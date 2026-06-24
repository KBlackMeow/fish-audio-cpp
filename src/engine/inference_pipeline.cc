// src/engine/inference_pipeline.cc
#include "engine/inference_pipeline.h"
#include "kernels/kernels.h"
#include "utils/cuda_utils.h"
#include <spdlog/spdlog.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string_view>
#include <numeric>
#include <random>
#include <unordered_map>

namespace fish {

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t kMaxRefCodeCacheEntries = 4;

void maybe_dump_codes(const std::vector<int32_t>& codes_flat, int B, int N, int T) {
    const char* path = std::getenv("FISH_DUMP_CODES");
    if (!path || !*path) return;
    std::ofstream f(path, std::ios::binary);
    if (!f.good()) {
        spdlog::warn("Cannot open FISH_DUMP_CODES path: {}", path);
        return;
    }
    int32_t hdr[3] = {B, N, T};
    f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(codes_flat.data()),
            codes_flat.size() * sizeof(int32_t));
    spdlog::info("  Dumped DAC codes to {} ([{}, {}, {}])", path, B, N, T);
}

double elapsed_ms(const Clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void log_profiling_summary(const TTSProfiling& p) {
    spdlog::info(
        "  Profiling: total={:.2f}ms tok={:.2f} prefill={:.2f} ar={:.2f} "
        "(embed={:.2f} decode={:.2f} logits={:.2f} sem_sample={:.2f} "
        "cb_decode={:.2f} cb_sample={:.2f} seq={:.2f}) dac={:.2f} copy={:.2f}",
        p.total_ms, p.tokenize_ms, p.prefill_ms, p.ar_decode_ms,
        p.decode_embed_ms, p.decode_step_ms, p.decode_logits_ms, p.semantic_sample_ms,
        p.codebook_decode_ms, p.codebook_sample_ms, p.seq_update_ms,
        p.dac_decode_ms, p.audio_copy_ms);
}

bool inference_debug_enabled() {
    const char* value = std::getenv("FISH_DEBUG_INFERENCE");
    return value && *value && std::string(value) != "0";
}

std::size_t hash_ref_audio(const float* samples, int count) {
    constexpr std::size_t kOffset = 1469598103934665603ull;
    constexpr std::size_t kPrime = 1099511628211ull;
    std::size_t hash = kOffset;
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(samples);
    const std::size_t nbytes = static_cast<std::size_t>(count) * sizeof(float);
    for (std::size_t i = 0; i < nbytes; ++i) {
        hash ^= static_cast<std::size_t>(bytes[i]);
        hash *= kPrime;
    }
    hash ^= static_cast<std::size_t>(count);
    hash *= kPrime;
    return hash;
}

std::size_t utf8_prefix_len(std::string_view text, std::size_t max_bytes) {
    if (text.size() <= max_bytes) return text.size();
    std::size_t cut = max_bytes;
    while (cut > 0 && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80) {
        --cut;
    }
    if (cut == 0) return max_bytes;
    return cut;
}

std::string trim_ascii(std::string_view text) {
    std::size_t start = 0;
    std::size_t end = text.size();
    while (start < end && std::isspace(static_cast<unsigned char>(text[start]))) ++start;
    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
    return std::string(text.substr(start, end - start));
}

bool starts_with_speaker_tag(std::string_view text, std::size_t pos) {
    return pos + 10 <= text.size() && text.substr(pos, 10) == "<|speaker:";
}

bool is_utf8_punctuation_at(std::string_view text, std::size_t pos, std::size_t* punct_len) {
    static const char* kPuncts[] = {"。", "！", "？", "，", "；", "：", "、"};
    for (const char* punct : kPuncts) {
        const std::size_t len = std::char_traits<char>::length(punct);
        if (pos + len <= text.size() && text.substr(pos, len) == punct) {
            *punct_len = len;
            return true;
        }
    }
    return false;
}

std::vector<std::string> split_text_units(const std::string& text) {
    std::vector<std::string> units;
    const std::string_view view(text);
    bool has_speaker = text.find("<|speaker:") != std::string::npos;

    if (has_speaker) {
        std::size_t pos = 0;
        while (pos < view.size()) {
            std::size_t next = view.size();
            for (std::size_t i = pos + 1; i < view.size(); ++i) {
                if (starts_with_speaker_tag(view, i)) {
                    next = i;
                    break;
                }
            }
            std::string piece = trim_ascii(view.substr(pos, next - pos));
            if (!piece.empty()) units.push_back(std::move(piece));
            pos = next;
        }
        return units;
    }

    std::size_t start = 0;
    for (std::size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        std::size_t punct_len = 0;
        const bool sentence_break =
            c == '\n' || c == '.' || c == '!' || c == '?' || c == ';' ||
            c == ',' || c == ':' ||
            is_utf8_punctuation_at(view, i, &punct_len);
        if (!sentence_break) {
            ++i;
            continue;
        }

        const std::size_t end = i + (punct_len > 0 ? punct_len : 1);
        std::string piece = trim_ascii(view.substr(start, end - start));
        if (!piece.empty()) units.push_back(std::move(piece));
        start = end;
        i = end;
    }
    if (start < text.size()) {
        std::string tail = trim_ascii(view.substr(start));
        if (!tail.empty()) units.push_back(std::move(tail));
    }
    if (units.empty()) {
        units.push_back(text);
    }
    return units;
}

std::vector<std::string> split_text_chunks(const std::string& text, int chunk_length) {
    if (chunk_length <= 0 || static_cast<int>(text.size()) <= chunk_length) {
        return {text};
    }

    std::vector<std::string> units = split_text_units(text);
    std::vector<std::string> chunks;
    std::string current;
    int current_bytes = 0;
    int speaker_count = 0;

    auto flush_current = [&]() {
        if (!current.empty()) {
            chunks.push_back(current);
            current.clear();
            current_bytes = 0;
            speaker_count = 0;
        }
    };

    auto merge_short_chunks = [&](std::vector<std::string>& out) {
        if (out.size() < 2 || chunk_length <= 0) return;
        const int short_tail_threshold = std::max(12, chunk_length / 3);
        bool changed = true;
        while (changed && out.size() >= 2) {
            changed = false;
            for (std::size_t i = 0; i < out.size(); ++i) {
                const int cur_bytes = static_cast<int>(out[i].size());
                if (cur_bytes > short_tail_threshold) continue;

                if (i + 1 < out.size()) {
                    const int merged_next =
                        cur_bytes + 1 + static_cast<int>(out[i + 1].size());
                    if (merged_next <= chunk_length + short_tail_threshold) {
                        out[i + 1] = out[i] + "\n" + out[i + 1];
                        out.erase(out.begin() + static_cast<std::ptrdiff_t>(i));
                        changed = true;
                        break;
                    }
                }

                if (i > 0) {
                    const int merged_prev =
                        static_cast<int>(out[i - 1].size()) + 1 + cur_bytes;
                    if (merged_prev <= chunk_length + short_tail_threshold) {
                        out[i - 1].append("\n").append(out[i]);
                        out.erase(out.begin() + static_cast<std::ptrdiff_t>(i));
                        changed = true;
                        break;
                    }
                }
            }
        }
    };

    for (const std::string& unit : units) {
        int unit_bytes = static_cast<int>(unit.size());
        bool unit_has_speaker = unit.find("<|speaker:") != std::string::npos;
        bool exceeds_speakers = unit_has_speaker && speaker_count >= 5 && !current.empty();
        bool exceeds_bytes = current_bytes + unit_bytes > chunk_length && !current.empty();

        if (exceeds_speakers || exceeds_bytes) {
            flush_current();
        }

        if (unit_bytes > chunk_length) {
            std::size_t pos = 0;
            while (pos < unit.size()) {
                std::string_view rest(unit.data() + pos, unit.size() - pos);
                std::size_t take = utf8_prefix_len(rest, static_cast<std::size_t>(chunk_length));
                std::size_t split = take;
                for (std::size_t i = take; i > 0; --i) {
                    const char ch = rest[i - 1];
                    if (ch == ' ' || ch == '\n' || ch == '\t' ||
                        ch == ',' || ch == '.' || ch == '!' || ch == '?' || ch == ';') {
                        split = i;
                        break;
                    }
                }
                chunks.emplace_back(trim_ascii(rest.substr(0, split)));
                pos += split;
            }
            continue;
        }

        if (!current.empty()) current.push_back('\n');
        current += unit;
        current_bytes += unit_bytes;
        if (unit_has_speaker) ++speaker_count;
    }

    flush_current();
    merge_short_chunks(chunks);
    return chunks;
}

void append_ids(std::vector<int32_t>& out, const std::vector<int>& ids) {
    for (int id : ids) out.push_back(static_cast<int32_t>(id));
}

void append_text_segment(
    std::vector<std::vector<int32_t>>& rows,
    const std::vector<int>& ids
) {
    if (rows.empty()) return;
    append_ids(rows[0], ids);
    for (std::size_t cb = 1; cb < rows.size(); ++cb) {
        rows[cb].insert(rows[cb].end(), ids.size(), 0);
    }
}

void append_code_segment(
    std::vector<std::vector<int32_t>>& rows,
    const std::vector<int32_t>& codes_flat,
    int num_codebooks,
    int code_len,
    int semantic_begin_id
) {
    if (rows.size() != static_cast<std::size_t>(num_codebooks + 1)) return;
    for (int t = 0; t < code_len; ++t) {
        rows[0].push_back(codes_flat[t] + semantic_begin_id);
    }
    for (int cb = 0; cb < num_codebooks; ++cb) {
        const int32_t* src = codes_flat.data() + static_cast<std::size_t>(cb) * code_len;
        rows[cb + 1].insert(rows[cb + 1].end(), src, src + code_len);
    }
}

std::vector<int32_t> flatten_rows(const std::vector<std::vector<int32_t>>& rows) {
    if (rows.empty()) return {};
    const int prompt_len = static_cast<int>(rows[0].size());
    std::vector<int32_t> prompt(rows.size() * static_cast<std::size_t>(prompt_len), 0);
    for (std::size_t row = 0; row < rows.size(); ++row) {
        std::copy(rows[row].begin(), rows[row].end(),
                  prompt.begin() + row * static_cast<std::size_t>(prompt_len));
    }
    return prompt;
}

std::vector<int32_t> tail_code_frames(
    const std::vector<int32_t>& codes_flat,
    int num_codebooks,
    int code_len,
    int keep_frames
) {
    if (keep_frames <= 0 || code_len <= keep_frames) {
        return codes_flat;
    }
    const int start = code_len - keep_frames;
    std::vector<int32_t> trimmed(static_cast<std::size_t>(num_codebooks) * keep_frames);
    for (int cb = 0; cb < num_codebooks; ++cb) {
        const int32_t* src = codes_flat.data() + static_cast<std::size_t>(cb) * code_len + start;
        std::copy(src, src + keep_frames, trimmed.begin() + static_cast<std::size_t>(cb) * keep_frames);
    }
    return trimmed;
}

// Simple audio decoder: one-shot DAC decode with pre-allocated GPU buffers.
// Audio chunking for streaming is handled by the caller.
class IncrementalAudioStreamer {
public:
    IncrementalAudioStreamer(DACEngine* dac, int num_codebooks, TTSProfiling* profiling)
        : dac_(dac), num_codebooks_(num_codebooks), profiling_(profiling) {
        samples_per_frame_ = dac_->config().hop_length() * 4;
    }

    ~IncrementalAudioStreamer() {
        if (d_codes_) cudaFree(d_codes_);
        if (d_audio_) cudaFree(d_audio_);
    }

    IncrementalAudioStreamer(const IncrementalAudioStreamer&) = delete;
    IncrementalAudioStreamer& operator=(const IncrementalAudioStreamer&) = delete;

    // Decode all code frames in one shot — full DAC context, correct audio.
    std::vector<float> decode_full(const std::vector<int32_t>& codes_flat) {
        std::vector<float> audio;
        const int code_len = num_codebooks_ > 0
            ? static_cast<int>(codes_flat.size() / static_cast<std::size_t>(num_codebooks_))
            : 0;
        if (code_len > 0) decode_window(codes_flat.data(), code_len, audio);
        return audio;
    }

private:
    void decode_window(const int32_t* codes, int frames, std::vector<float>& audio) {
        const int max_out = frames * samples_per_frame_ * 2;
        int audio_len = 0;

        // Reuse pre-allocated GPU buffers — avoid cudaMalloc/cudaFree stalls.
        const size_t codes_bytes = static_cast<size_t>(num_codebooks_) * frames * sizeof(int32_t);
        if (codes_bytes > d_codes_cap_) {
            if (d_codes_) cudaFree(d_codes_);
            CUDA_CHECK(cudaMalloc(&d_codes_, codes_bytes));
            d_codes_cap_ = codes_bytes;
        }
        const size_t audio_bytes = static_cast<size_t>(max_out) * sizeof(float);
        if (audio_bytes > d_audio_cap_) {
            if (d_audio_) cudaFree(d_audio_);
            CUDA_CHECK(cudaMalloc(&d_audio_, audio_bytes));
            d_audio_cap_ = audio_bytes;
        }

        CUDA_CHECK(cudaMemcpy(d_codes_, codes, codes_bytes, cudaMemcpyHostToDevice));

        const auto dac_start = Clock::now();
        dac_->decode(d_codes_, 1, frames, d_audio_, &audio_len, max_out);
        if (profiling_) profiling_->dac_decode_ms += elapsed_ms(dac_start);

        const auto copy_start = Clock::now();
        audio.resize(static_cast<std::size_t>(audio_len));
        CUDA_CHECK(cudaMemcpy(audio.data(), d_audio_, audio.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if (profiling_) profiling_->audio_copy_ms += elapsed_ms(copy_start);
    }

    DACEngine* dac_;
    int num_codebooks_;
    TTSProfiling* profiling_;
    int samples_per_frame_ = 0;
    int32_t* d_codes_ = nullptr;
    float* d_audio_ = nullptr;
    size_t d_codes_cap_ = 0;
    size_t d_audio_cap_ = 0;
};

}  // namespace

InferencePipeline::InferencePipeline(
    std::unique_ptr<DualAREngine> dual_ar,
    std::unique_ptr<DACEngine> dac,
    std::unique_ptr<BlockManager> block_mgr,
    std::unique_ptr<Tokenizer> tokenizer
) : dual_ar_(std::move(dual_ar)),
    dac_(std::move(dac)),
    block_mgr_(std::move(block_mgr)),
    tokenizer_(std::move(tokenizer))
{
    // All GPU operations use dual_ar_->stream() — no separate pipeline stream.
    spdlog::info("InferencePipeline: using unified CUDA stream");
}

InferencePipeline::~InferencePipeline() {
    // No separate stream to destroy — DualAR owns the stream.
}

int InferencePipeline::sample_rate() const {
    return dac_->config().sample_rate;
}

TTSOutput InferencePipeline::run(
    const std::string& text,
    int max_new_tokens,
    float temperature,
    float top_p,
    int top_k,
    int seed
) {
    StreamCallback noop;
    return run_streaming(text, max_new_tokens, temperature, top_p, top_k, seed, noop);
}

TTSOutput InferencePipeline::run_streaming(
    const std::string& text,
    int max_new_tokens,
    float temperature,
    float top_p,
    int top_k,
    int seed,
    StreamCallback callback
) {
    spdlog::info("InferencePipeline::run: text='{}' max_new={}", text, max_new_tokens);
    const auto total_start = Clock::now();
    TTSProfiling profiling;
    const bool debug_inference = inference_debug_enabled();

    // ── Use DualAR's stream for ALL GPU operations (no separate pipeline stream) ──
    cudaStream_t stream = dual_ar_->stream();

    int dim = dual_ar_->config().dim;
    int num_codebooks = dual_ar_->config().num_codebooks;
    int codebook_dim = 1 + num_codebooks;
    int vocab_size = dual_ar_->config().vocab_size;

    // Step 1: Tokenize
    const auto tokenize_start = Clock::now();
    auto token_ids = tokenizer_->encode(text);
    profiling.tokenize_ms = elapsed_ms(tokenize_start);
    int prompt_len = static_cast<int>(token_ids.size());
    profiling.prompt_tokens = prompt_len;
    spdlog::info("  Tokenized: {} tokens", prompt_len);

    // Step 2: Build prompt tensor [1, codebook_dim, prompt_len]
    std::vector<int32_t> h_prompt(codebook_dim * prompt_len, 0);
    for (int t = 0; t < prompt_len; t++) {
        h_prompt[t] = token_ids[t];  // semantic channel
    }

    int32_t* d_prompt;
    CUDA_CHECK(cudaMalloc(&d_prompt, h_prompt.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_prompt, h_prompt.data(),
                          h_prompt.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    // Step 3: Allocate blocks
    Sequence seq;
    seq.seq_id = 0;
    int n_blocks = (prompt_len + max_new_tokens + 15) / 16 + 1;
    if (!block_mgr_->allocate_blocks(seq, n_blocks)) {
        spdlog::error("Not enough blocks");
        CUDA_CHECK(cudaFree(d_prompt));
        return {};
    }
    seq.seq_len = prompt_len;
    seq.status = Sequence::RUNNING;

    // Step 4: Sync block table to GPU
    int32_t* d_block_table;
    int32_t* d_seq_len;
    CUDA_CHECK(cudaMalloc(&d_block_table, 256 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_seq_len, sizeof(int32_t)));

    std::vector<int32_t> h_bt(256, -1);
    for (size_t i = 0; i < seq.block_table.size(); i++)
        h_bt[i] = static_cast<int32_t>(seq.block_table[i]);
    CUDA_CHECK(cudaMemcpy(d_block_table, h_bt.data(), 256 * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq_len, &seq.seq_len, sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    // Step 5: Prefill
    __half *d_hidden, *d_fast;
    CUDA_CHECK(cudaMalloc(&d_hidden, dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_fast, dim * sizeof(__half)));

    const auto prefill_start = Clock::now();
    dual_ar_->prefill(d_prompt, 1, prompt_len, 1,
                      block_mgr_->k_cache(), block_mgr_->v_cache(),
                      d_block_table, d_seq_len,
                      d_hidden, d_fast);
    profiling.prefill_ms = elapsed_ms(prefill_start);

    // Diagnostic: verify prefill hidden state varies with input
    if (debug_inference) {
        std::vector<__half> h_hidden(dim);
        CUDA_CHECK(cudaMemcpy(h_hidden.data(), d_hidden, dim * sizeof(__half), cudaMemcpyDeviceToHost));
        float l2 = 0.f, first10 = 0.f;
        for (int i = 0; i < dim; i++) {
            float v = __half2float(h_hidden[i]);
            l2 += v * v;
            if (i < 10) first10 += std::abs(v);
        }
        spdlog::info("  [DEBUG] Prefill hidden L2-norm={:.3f} first10-abs={:.3f}", std::sqrt(l2), first10);
    }

    // Step 6: Decode loop
    int im_end_id = tokenizer_->im_end_id();
    std::vector<int32_t> generated;
    IncrementalAudioStreamer audio_streamer(dac_.get(), num_codebooks, &profiling);

    // Pre-allocate logits and token buffers on GPU
    __half* d_logits;
    int32_t* d_tokens;
    __half* d_codebook_logits;
    __half* d_embed;  // combined embedding for each decode step
    int32_t* d_codebook_tokens;
    CUDA_CHECK(cudaMalloc(&d_logits,          vocab_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_tokens,           2 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_codebook_logits,
                          dual_ar_->config().codebook_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_embed,            dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_codebook_tokens,  num_codebooks * sizeof(int32_t)));
    int32_t h_sample_token = 0;
    int32_t h_high_sample_token = 0;

    auto sample_range = [&](int sz, int id_start, float temp, float tp, int tk, uint64_t rng) -> int32_t {
        const auto sample_start = Clock::now();
        kernels::sample_top_k_top_p_range(
            d_codebook_logits, d_tokens, 1, dual_ar_->config().codebook_size,
            id_start, sz, temp, tp, tk, rng, stream);
        CUDA_CHECK(cudaMemcpy(&h_sample_token, d_tokens, sizeof(int32_t), cudaMemcpyDeviceToHost));
        profiling.codebook_sample_ms += elapsed_ms(sample_start);
        return h_sample_token;
    };

    auto sample_range_device = [&](int32_t* out_token, int sz, int id_start,
                                   float temp, float tp, int tk, uint64_t rng) {
        const auto sample_start = Clock::now();
        kernels::sample_top_k_top_p_range(
            d_codebook_logits, out_token, 1, dual_ar_->config().codebook_size,
            id_start, sz, temp, tp, tk, rng, stream);
        profiling.codebook_sample_ms += elapsed_ms(sample_start);
    };

    // Semantic sampling: allow [sem_start, sem_end] + eos_token
    int sem_start = dual_ar_->config().semantic_begin_id;  // 151678
    int sem_end   = dual_ar_->config().semantic_end_id;    // 155773
    int eos_id    = dual_ar_->config().im_end_token_id;    // 151645
    int sem_range = sem_end - sem_start + 1;
    std::vector<__half> h_logits_fp16;

    auto sample_semantic = [&](float temp, float tp, int tk, uint64_t rng) -> int32_t {
        const auto sample_start = Clock::now();
        kernels::sample_top_k_top_p_semantic_eos(
            d_logits, d_tokens, 1, vocab_size, sem_start, sem_range, eos_id,
            temp, tp, tk, rng, stream);
        CUDA_CHECK(cudaMemcpy(&h_sample_token, d_tokens, sizeof(int32_t), cudaMemcpyDeviceToHost));
        profiling.semantic_sample_ms += elapsed_ms(sample_start);
        return h_sample_token;
    };

    // ---- Prefill logits → first semantic token ----
    {
        const auto logits_start = Clock::now();
        dual_ar_->get_logits(d_hidden, d_logits, 1);
        profiling.decode_logits_ms += elapsed_ms(logits_start);
    }

    // Debug: dump top-10 semantic logits after prefill
    if (debug_inference) {
        int total_dbg = sem_start + sem_range;
        h_logits_fp16.resize(total_dbg);
        CUDA_CHECK(cudaMemcpy(h_logits_fp16.data(), d_logits,
                              total_dbg * sizeof(__half), cudaMemcpyDeviceToHost));
        std::vector<int> dbg_idx(sem_range);
        std::iota(dbg_idx.begin(), dbg_idx.end(), 0);
        std::partial_sort(dbg_idx.begin(), dbg_idx.begin() + 10, dbg_idx.end(),
            [&](int a, int b){ return __half2float(h_logits_fp16[sem_start+a]) > __half2float(h_logits_fp16[sem_start+b]); });
        spdlog::info("  [DEBUG] Prefill top-10 sem logits (id=begin+offset):");
        for (int i = 0; i < 10; i++) {
            int id = sem_start + dbg_idx[i];
            spdlog::info("    id={} logit={:.3f}", id, __half2float(h_logits_fp16[id]));
        }
        // Also show eos logit (copy from GPU first)
        {
            __half h_eos;
            CUDA_CHECK(cudaMemcpy(&h_eos, d_logits + eos_id, sizeof(__half),
                                  cudaMemcpyDeviceToHost));
            spdlog::info("    id={} (eos) logit={:.3f}", eos_id,
                         __half2float(h_eos));
        }
    }

    int32_t sem = sample_semantic(temperature, top_p, top_k,
                                  static_cast<uint64_t>(seed));
    spdlog::info("  Prefill → first sem token: {}", sem);
    if (sem == eos_id) {
        spdlog::info("  EOS after prefill — no tokens generated");
        goto decode_done;
    }

    {
    // ---- Generate frame-0 codebooks from the PREFILL hidden state ----
    // Python flow (decode_one_token_ar):
    //   cb[0] = clamp(sem - sem_begin, 0, cb_size-1)  -- deterministic
    //   cb[1..N-1] = sample(fast_decoder_logits)       -- full range, no clamp
    int cb_size  = dual_ar_->config().codebook_size;
    int sem_begin = dual_ar_->config().semantic_begin_id;

    auto gen_codebooks = [&](int32_t cur_sem, uint64_t base_rng,
                             std::vector<int32_t>& out_cbs,
                             float cb_temp, float cb_top_p, int cb_top_k) {
        const auto cb_start = Clock::now();
        out_cbs.resize(num_codebooks);
        // position 0: populate fast KV cache (logits discarded in Python too)
        dual_ar_->fast_codebook_decode(d_hidden, nullptr, d_codebook_logits,
                                       1, 0, 0, false);
        // cb[0] = deterministic: clamp(sem - sem_begin, 0, cb_size-1)
        out_cbs[0] = std::max(0, std::min(cb_size - 1, cur_sem - sem_begin));
        CUDA_CHECK(cudaMemcpyAsync(d_codebook_tokens, out_cbs.data(),
                                   sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        // cb[1..num_codebooks-1]: sample from fast decoder with FULL logits
        // (Python does NOT clamp to acoustic range — model learns to avoid invalid codes)
        for (int cb = 1; cb < num_codebooks; cb++) {
            dual_ar_->fast_codebook_decode_device(d_hidden, nullptr, d_codebook_logits,
                                                  1, cb, d_codebook_tokens + (cb - 1));
            // Sample from full [0, cb_size) range (matches Python), using pipeline params
            sample_range_device(d_codebook_tokens + cb, cb_size, 0,
                                cb_temp, cb_top_p, cb_top_k, base_rng + cb);
        }
        CUDA_CHECK(cudaMemcpy(out_cbs.data(), d_codebook_tokens,
                              num_codebooks * sizeof(int32_t), cudaMemcpyDeviceToHost));
        profiling.codebook_decode_ms += elapsed_ms(cb_start);
    };

    std::vector<int32_t> curr_cbs;
    gen_codebooks(sem, static_cast<uint64_t>(seed), curr_cbs, temperature, top_p, top_k);

    // ---- RAS (Repetition Aware Sampling) state ----
    // Python: previous_tokens shape [num_codebooks+1, RAS_WIN_SIZE]
    // RAS only checks row-0 (semantic tokens) for repetition, so a ring buffer suffices.
    static constexpr int RAS_WIN_SIZE = 10;
    static constexpr float RAS_HIGH_TEMP = 1.0f;
    static constexpr float RAS_HIGH_TOP_P = 0.9f;
    std::vector<int32_t> sem_history(RAS_WIN_SIZE, -1);
    sem_history[0] = sem;  // first semantic token enters history

    // ---- Autoregressive loop ----
    // Python flow at each step k:
    //   1. slow_decode( embed[sem_k, cbs_k] ) → hidden
    //   2. get_logits(hidden) → sample sem_{k+1}
    //   3. fast_decoder(hidden) → cbs_{k+1}
    //   4. record frame k = [sem_k, cbs_k]
    const auto ar_decode_start = Clock::now();
    for (int step = 0; step < max_new_tokens; step++) {
        // 1. Embed [sem, cbs] for THIS frame (same step)
        const auto embed_start = Clock::now();
        dual_ar_->embed_for_decode(sem, curr_cbs.data(), d_embed, stream);
        profiling.decode_embed_ms += elapsed_ms(embed_start);

        // seq_len must include the current position being written (prompt_len + step),
        // so the attention covers positions 0..prompt_len+step (inclusive).
        int32_t cur_seq_len = prompt_len + step + 1;
        const auto seq_start = Clock::now();
        CUDA_CHECK(cudaMemcpy(d_seq_len, &cur_seq_len, sizeof(int32_t),
                              cudaMemcpyHostToDevice));
        profiling.seq_update_ms += elapsed_ms(seq_start);

        // 2. Slow decode at position prompt_len+step
        const auto decode_start = Clock::now();
        dual_ar_->decode_step(d_embed, 1, prompt_len + step,
                              block_mgr_->k_cache(), block_mgr_->v_cache(),
                              d_block_table, d_seq_len,
                              d_hidden, d_hidden);
        profiling.decode_step_ms += elapsed_ms(decode_start);

        // 3. Sample next semantic token (normal + RAS high-temp fallback)
        {
            const auto logits_start = Clock::now();
            dual_ar_->get_logits(d_hidden, d_logits, 1);
            profiling.decode_logits_ms += elapsed_ms(logits_start);
        }
        int32_t sem_normal = sample_semantic(temperature, top_p, top_k,
                                              static_cast<uint64_t>(seed) + step + 1);
        // RAS: if sem_normal is in semantic range AND appeared in recent history, use high-temp fallback
        bool is_semantic = (sem_normal >= sem_start && sem_normal <= sem_end);
        bool in_window = false;
        if (is_semantic) {
            for (int w = 0; w < RAS_WIN_SIZE; w++) {
                if (sem_history[w] == sem_normal) { in_window = true; break; }
            }
        }
        int32_t sem_high = sem_normal;
        if (is_semantic && in_window) {
            const auto sample_start = Clock::now();
            kernels::sample_top_k_top_p_semantic_eos(
                d_logits, d_tokens + 1, 1, vocab_size, sem_start, sem_range, eos_id,
                RAS_HIGH_TEMP, RAS_HIGH_TOP_P, top_k,
                static_cast<uint64_t>(seed) + step + 1 + 1000000, stream);
            CUDA_CHECK(cudaMemcpy(&h_high_sample_token, d_tokens + 1, sizeof(int32_t), cudaMemcpyDeviceToHost));
            sem_high = h_high_sample_token;
            profiling.semantic_sample_ms += elapsed_ms(sample_start);
        }
        int32_t next_sem = (is_semantic && in_window) ? sem_high : sem_normal;

        // 4. Generate NEXT frame's codebooks from this hidden state
        std::vector<int32_t> next_cbs;
        gen_codebooks(next_sem, static_cast<uint64_t>(seed) + (step + 1) * num_codebooks,
                      next_cbs, temperature, top_p, top_k);

        // 5. Record current frame
        generated.push_back(sem);
        for (auto c : curr_cbs) generated.push_back(c);

        // Streaming: report progress (audio chunks are sent after full DAC decode)
        if (callback.on_progress) {
            callback.on_progress(step, max_new_tokens);
        }

        // Diagnostic: log first 5 semantic tokens + codebook ranges
        if (debug_inference && step < 5) {
            spdlog::info("  step={} sem={} (normal={} high={}) cb0={} cb1={} cb_last={}",
                         step, sem, sem_normal, sem_high, curr_cbs[0], curr_cbs[1],
                         curr_cbs[num_codebooks - 1]);
        }

        // Roll RAS history left, insert next semantic token at end
        for (int w = 0; w < RAS_WIN_SIZE - 1; w++)
            sem_history[w] = sem_history[w + 1];
        sem_history[RAS_WIN_SIZE - 1] = next_sem;

        // Advance
        sem = next_sem;
        curr_cbs = std::move(next_cbs);
        if (sem == eos_id) break;
    }
    profiling.ar_decode_ms = elapsed_ms(ar_decode_start);
    }
    decode_done:

    int code_len = static_cast<int>(generated.size()) / (1 + num_codebooks);
    profiling.generated_frames = code_len;
    spdlog::info("  Generated {} frames ({}+{} tokens each)",
                 code_len, 1, num_codebooks);

    // Safety: if nothing generated, return silence instead of crashing
    if (code_len == 0) {
        spdlog::info("  No frames generated — returning silence");
        block_mgr_->free_blocks(seq);
        CUDA_CHECK(cudaFree(d_prompt));
        CUDA_CHECK(cudaFree(d_block_table));
        CUDA_CHECK(cudaFree(d_seq_len));
        CUDA_CHECK(cudaFree(d_hidden));
        CUDA_CHECK(cudaFree(d_fast));
        CUDA_CHECK(cudaFree(d_logits));
        CUDA_CHECK(cudaFree(d_tokens));
        CUDA_CHECK(cudaFree(d_codebook_logits));
        CUDA_CHECK(cudaFree(d_embed));
        CUDA_CHECK(cudaFree(d_codebook_tokens));
        profiling.total_ms = elapsed_ms(total_start);
        log_profiling_summary(profiling);
        return {std::vector<float>(), std::vector<int32_t>(), dac_->config().sample_rate, profiling};
    }

    // Step 7: DAC decode — decode all codes at once, then stream audio in chunks
    // generated layout: [sem_0, cb0_0, cb1_0, ..., sem_1, cb0_1, ...] × code_len
    int stride  = 1 + num_codebooks;
    spdlog::info("  DAC input: {} code frames x {} codebooks", code_len, num_codebooks);
    std::vector<int32_t> codes_flat(num_codebooks * code_len, 0);
    for (int i = 0; i < code_len; i++) {
        for (int cb = 0; cb < num_codebooks; cb++) {
            codes_flat[cb * code_len + i] = generated[i * stride + 1 + cb];
        }
    }
    // Diagnostic: log code value ranges per codebook
    for (int cb = 0; cb < num_codebooks; cb++) {
        int cb_min = INT32_MAX, cb_max = INT32_MIN;
        for (int i = 0; i < code_len; i++) {
            int v = codes_flat[cb * code_len + i];
            if (v < cb_min) cb_min = v;
            if (v > cb_max) cb_max = v;
        }
        spdlog::info("  cb[{}] range: [{}, {}]", cb, cb_min, cb_max);
    }
    maybe_dump_codes(codes_flat, 1, num_codebooks, code_len);

    // One-shot DAC decode of all codes (full context, no incremental windows)
    std::vector<float> full_audio = audio_streamer.decode_full(codes_flat);

    // Stream audio in ~100ms chunks when a callback is registered
    if (callback.on_audio_chunk) {
        const int chunk_samples = dac_->config().sample_rate / 10;  // 100 ms
        for (int off = 0; off < static_cast<int>(full_audio.size()); off += chunk_samples) {
            int n = std::min(chunk_samples, static_cast<int>(full_audio.size()) - off);
            callback.on_audio_chunk(full_audio.data() + off, n);
            if (profiling.streamed_audio_chunks == 0 && profiling.first_audio_ms == 0.0)
                profiling.first_audio_ms = elapsed_ms(total_start);
            profiling.streamed_audio_chunks += 1;
        }
    }

    TTSOutput audio_result;
    audio_result.generated_codes = codes_flat;
    audio_result.sample_rate = dac_->config().sample_rate;
    audio_result.audio_samples = std::move(full_audio);
    const int audio_len = static_cast<int>(audio_result.audio_samples.size());

    spdlog::info("  Audio: {} samples ({:.2f}s @ {}Hz)",
                 audio_len, static_cast<double>(audio_len) / dac_->config().sample_rate,
                 dac_->config().sample_rate);

    // Step 8: Cleanup — all synchronous, unified stream
    block_mgr_->free_blocks(seq);
    CUDA_CHECK(cudaFree(d_prompt));
    CUDA_CHECK(cudaFree(d_block_table));
    CUDA_CHECK(cudaFree(d_seq_len));
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_fast));
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_tokens));
    CUDA_CHECK(cudaFree(d_codebook_logits));
    CUDA_CHECK(cudaFree(d_embed));
    CUDA_CHECK(cudaFree(d_codebook_tokens));
    profiling.total_ms = elapsed_ms(total_start);
    audio_result.profiling = profiling;
    log_profiling_summary(audio_result.profiling);
    return audio_result;
}

TTSOutput InferencePipeline::run_with_prompt_file(
    const std::string& prompt_path,
    int max_new_tokens,
    float temperature,
    float top_p,
    int top_k,
    int seed
) {
    spdlog::info("InferencePipeline::run_with_prompt_file: path='{}'", prompt_path);

    std::ifstream pf(prompt_path, std::ios::binary);
    if (!pf.good()) {
        spdlog::error("Cannot open prompt file: {}", prompt_path);
        return {};
    }
    int32_t hdr[2];
    if (!pf.read(reinterpret_cast<char*>(hdr), 2 * sizeof(int32_t))) {
        spdlog::error("Prompt file is too small: {}", prompt_path);
        return {};
    }
    int file_num_cb = hdr[0];
    int prompt_len   = hdr[1];
    if (file_num_cb <= 0 || prompt_len <= 0) {
        spdlog::error("Invalid prompt header: num_codebooks={} prompt_len={}",
                      file_num_cb, prompt_len);
        return {};
    }
    int cb_dim = file_num_cb + 1;
    int model_num_cb = dual_ar_->config().num_codebooks;
    if (file_num_cb != model_num_cb) {
        spdlog::error("Prompt num_codebooks={} ≠ model num_codebooks={}", file_num_cb, model_num_cb);
        return {};
    }
    size_t data_sz = static_cast<size_t>(cb_dim) * prompt_len;
    std::vector<int32_t> h_prompt(data_sz);
    if (!pf.read(reinterpret_cast<char*>(h_prompt.data()), data_sz * sizeof(int32_t))) {
        spdlog::error("Prompt file is truncated: expected {} int32 payload values", data_sz);
        return {};
    }
    spdlog::info("  Prompt file: num_codebooks={} prompt_len={}", file_num_cb, prompt_len);
    return run_with_prompt_tensor(
        h_prompt, file_num_cb, prompt_len, max_new_tokens, temperature, top_p, top_k, seed);
}

TTSOutput InferencePipeline::run_with_prompt_tensor(
    const std::vector<int32_t>& h_prompt,
    int num_codebooks,
    int prompt_len,
    int max_new_tokens,
    float temperature,
    float top_p,
    int top_k,
    int seed,
    StreamCallback callback
) {
    const auto total_start = Clock::now();
    TTSProfiling profiling;
    profiling.prompt_tokens = prompt_len;

    cudaStream_t stream = dual_ar_->stream();
    int dim = dual_ar_->config().dim;
    int cb_dim = num_codebooks + 1;
    int vocab_size = dual_ar_->config().vocab_size;
    const bool debug_inference = inference_debug_enabled();

    int32_t* d_prompt;
    const size_t data_sz = h_prompt.size();
    CUDA_CHECK(cudaMalloc(&d_prompt, data_sz * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_prompt, h_prompt.data(),
                          data_sz * sizeof(int32_t), cudaMemcpyHostToDevice));

    // Block manager
    Sequence seq;
    seq.seq_id = 0;
    int n_blocks = (prompt_len + max_new_tokens + 15) / 16 + 1;
    if (!block_mgr_->allocate_blocks(seq, n_blocks)) {
        spdlog::error("Not enough blocks");
        CUDA_CHECK(cudaFree(d_prompt));
        return {};
    }
    seq.seq_len = prompt_len;
    seq.status = Sequence::RUNNING;

    int32_t* d_block_table;
    int32_t* d_seq_len;
    CUDA_CHECK(cudaMalloc(&d_block_table, 256 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_seq_len, sizeof(int32_t)));
    std::vector<int32_t> h_bt(256, -1);
    for (size_t i = 0; i < seq.block_table.size(); i++)
        h_bt[i] = static_cast<int32_t>(seq.block_table[i]);
    CUDA_CHECK(cudaMemcpy(d_block_table, h_bt.data(), 256 * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq_len, &seq.seq_len, sizeof(int32_t), cudaMemcpyHostToDevice));

    // GPU buffers
    __half *d_hidden, *d_fast;
    CUDA_CHECK(cudaMalloc(&d_hidden, dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_fast, dim * sizeof(__half)));

    const auto prefill_start = Clock::now();
    dual_ar_->prefill(d_prompt, 1, prompt_len, cb_dim,
                      block_mgr_->k_cache(), block_mgr_->v_cache(),
                      d_block_table, d_seq_len,
                      d_hidden, d_fast);
    profiling.prefill_ms = elapsed_ms(prefill_start);

    int eos_id    = dual_ar_->config().im_end_token_id;
    int sem_start = dual_ar_->config().semantic_begin_id;
    int sem_end   = dual_ar_->config().semantic_end_id;
    int sem_range = sem_end - sem_start + 1;

    std::vector<int32_t> generated;
    IncrementalAudioStreamer audio_streamer(dac_.get(), num_codebooks, &profiling);
    __half* d_logits;
    int32_t* d_tokens;
    __half* d_codebook_logits;
    __half* d_embed;
    int32_t* d_codebook_tokens;
    CUDA_CHECK(cudaMalloc(&d_logits,          vocab_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_tokens,          2 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_codebook_logits, dual_ar_->config().codebook_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_embed,           dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_codebook_tokens, num_codebooks * sizeof(int32_t)));

    int32_t h_sample_token = 0;
    int32_t h_high_sample_token = 0;

    auto sample_range = [&](int sz, int id_start, float temp, float tp, int tk, uint64_t rng) -> int32_t {
        const auto sample_start = Clock::now();
        kernels::sample_top_k_top_p_range(
            d_codebook_logits, d_tokens, 1, dual_ar_->config().codebook_size,
            id_start, sz, temp, tp, tk, rng, stream);
        CUDA_CHECK(cudaMemcpy(&h_sample_token, d_tokens, sizeof(int32_t), cudaMemcpyDeviceToHost));
        profiling.codebook_sample_ms += elapsed_ms(sample_start);
        return h_sample_token;
    };

    auto sample_range_device = [&](int32_t* out_token, int sz, int id_start,
                                   float temp, float tp, int tk, uint64_t rng) {
        const auto sample_start = Clock::now();
        kernels::sample_top_k_top_p_range(
            d_codebook_logits, out_token, 1, dual_ar_->config().codebook_size,
            id_start, sz, temp, tp, tk, rng, stream);
        profiling.codebook_sample_ms += elapsed_ms(sample_start);
    };

    auto sample_semantic = [&](float temp, float tp, int tk, uint64_t rng) -> int32_t {
        const auto sample_start = Clock::now();
        kernels::sample_top_k_top_p_semantic_eos(
            d_logits, d_tokens, 1, vocab_size, sem_start, sem_range, eos_id,
            temp, tp, tk, rng, stream);
        CUDA_CHECK(cudaMemcpy(&h_sample_token, d_tokens, sizeof(int32_t), cudaMemcpyDeviceToHost));
        profiling.semantic_sample_ms += elapsed_ms(sample_start);
        return h_sample_token;
    };

    {
        const auto logits_start = Clock::now();
        dual_ar_->get_logits(d_hidden, d_logits, 1);
        profiling.decode_logits_ms += elapsed_ms(logits_start);
    }
    int32_t sem = sample_semantic(temperature, top_p, top_k, static_cast<uint64_t>(seed));
    spdlog::info("  Prefill → first sem token: {}", sem);
    if (sem == eos_id) { spdlog::info("  EOS after prefill"); goto prompt_decode_done; }

    {
    int cb_size = dual_ar_->config().codebook_size;
    int sem_begin = dual_ar_->config().semantic_begin_id;
    auto gen_codebooks = [&](int32_t cur_sem, uint64_t base_rng, std::vector<int32_t>& out_cbs, float cbt, float cbp, int cbk) {
        const auto cb_start = Clock::now();
        out_cbs.resize(num_codebooks);
        dual_ar_->fast_codebook_decode(d_hidden, nullptr, d_codebook_logits, 1, 0, 0, false);
        out_cbs[0] = std::max(0, std::min(cb_size - 1, cur_sem - sem_begin));
        CUDA_CHECK(cudaMemcpyAsync(d_codebook_tokens, out_cbs.data(),
                                   sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        for (int cb = 1; cb < num_codebooks; cb++) {
            dual_ar_->fast_codebook_decode_device(d_hidden, nullptr, d_codebook_logits,
                                                  1, cb, d_codebook_tokens + (cb - 1));
            sample_range_device(d_codebook_tokens + cb, cb_size, 0,
                                cbt, cbp, cbk, base_rng + cb);
        }
        CUDA_CHECK(cudaMemcpy(out_cbs.data(), d_codebook_tokens,
                              num_codebooks * sizeof(int32_t), cudaMemcpyDeviceToHost));
        profiling.codebook_decode_ms += elapsed_ms(cb_start);
    };

    std::vector<int32_t> curr_cbs;
    gen_codebooks(sem, static_cast<uint64_t>(seed), curr_cbs, temperature, top_p, top_k);

    static constexpr int RAS_WIN_SIZE = 10;
    static constexpr float RAS_HIGH_TEMP = 1.0f, RAS_HIGH_TOP_P = 0.9f;
    std::vector<int32_t> sem_history(RAS_WIN_SIZE, -1);
    sem_history[0] = sem;

    const auto ar_decode_start = Clock::now();
    for (int step = 0; step < max_new_tokens; step++) {
        const auto embed_start = Clock::now();
        dual_ar_->embed_for_decode(sem, curr_cbs.data(), d_embed, stream);
        profiling.decode_embed_ms += elapsed_ms(embed_start);
        int32_t cur_seq_len = prompt_len + step + 1;
        const auto seq_start = Clock::now();
        CUDA_CHECK(cudaMemcpy(d_seq_len, &cur_seq_len, sizeof(int32_t), cudaMemcpyHostToDevice));
        profiling.seq_update_ms += elapsed_ms(seq_start);
        const auto decode_start = Clock::now();
        dual_ar_->decode_step(d_embed, 1, prompt_len + step,
                              block_mgr_->k_cache(), block_mgr_->v_cache(),
                              d_block_table, d_seq_len, d_hidden, d_hidden);
        profiling.decode_step_ms += elapsed_ms(decode_start);
        {
            const auto logits_start = Clock::now();
            dual_ar_->get_logits(d_hidden, d_logits, 1);
            profiling.decode_logits_ms += elapsed_ms(logits_start);
        }
        int32_t sem_normal = sample_semantic(temperature, top_p, top_k, static_cast<uint64_t>(seed) + step + 1);
        bool is_sem = (sem_normal >= sem_start && sem_normal <= sem_end);
        bool in_win = false;
        if (is_sem) { for (int w = 0; w < RAS_WIN_SIZE; w++) if (sem_history[w] == sem_normal) { in_win = true; break; } }
        int32_t sem_high = sem_normal;
        if (is_sem && in_win) {
            const auto sample_start = Clock::now();
            kernels::sample_top_k_top_p_semantic_eos(
                d_logits, d_tokens + 1, 1, vocab_size, sem_start, sem_range, eos_id,
                RAS_HIGH_TEMP, RAS_HIGH_TOP_P, top_k,
                static_cast<uint64_t>(seed) + step + 1 + 1000000, stream);
            CUDA_CHECK(cudaMemcpy(&h_high_sample_token, d_tokens + 1, sizeof(int32_t), cudaMemcpyDeviceToHost));
            sem_high = h_high_sample_token;
            profiling.semantic_sample_ms += elapsed_ms(sample_start);
        }
        int32_t next_sem = (is_sem && in_win) ? sem_high : sem_normal;
        std::vector<int32_t> next_cbs;
        gen_codebooks(next_sem, static_cast<uint64_t>(seed) + (step + 1) * num_codebooks, next_cbs, temperature, top_p, top_k);
        generated.push_back(sem);
        for (auto c : curr_cbs) generated.push_back(c);
        if (callback.on_progress) callback.on_progress(step, max_new_tokens);
        if (debug_inference && step < 5) {
            spdlog::info("  step={} sem={} (normal={} high={}) cb0={} cb1={} cb_last={}",
                         step, sem, sem_normal, sem_high, curr_cbs[0], curr_cbs[1],
                         curr_cbs[num_codebooks - 1]);
        }
        for (int w = 0; w < RAS_WIN_SIZE - 1; w++) sem_history[w] = sem_history[w + 1];
        sem_history[RAS_WIN_SIZE - 1] = next_sem;
        sem = next_sem; curr_cbs = std::move(next_cbs);
        if (sem == eos_id) break;
    }
    profiling.ar_decode_ms = elapsed_ms(ar_decode_start);
    }
    prompt_decode_done:

    int code_len = static_cast<int>(generated.size()) / (1 + num_codebooks);
    profiling.generated_frames = code_len;
    spdlog::info("  Generated {} frames", code_len);
    if (code_len == 0) {
        block_mgr_->free_blocks(seq);
        CUDA_CHECK(cudaFree(d_prompt)); CUDA_CHECK(cudaFree(d_block_table)); CUDA_CHECK(cudaFree(d_seq_len));
        CUDA_CHECK(cudaFree(d_hidden)); CUDA_CHECK(cudaFree(d_fast));
        CUDA_CHECK(cudaFree(d_logits)); CUDA_CHECK(cudaFree(d_tokens)); CUDA_CHECK(cudaFree(d_codebook_logits)); CUDA_CHECK(cudaFree(d_embed)); CUDA_CHECK(cudaFree(d_codebook_tokens));
        profiling.total_ms = elapsed_ms(total_start);
        log_profiling_summary(profiling);
        return {std::vector<float>(), std::vector<int32_t>(), dac_->config().sample_rate, profiling};
    }

    int stride = 1 + num_codebooks;
    std::vector<int32_t> codes_flat(num_codebooks * code_len, 0);
    for (int i = 0; i < code_len; i++)
        for (int cb = 0; cb < num_codebooks; cb++)
            codes_flat[cb * code_len + i] = generated[i * stride + 1 + cb];
    maybe_dump_codes(codes_flat, 1, num_codebooks, code_len);

    // One-shot DAC decode, then stream audio in chunks
    std::vector<float> full_audio = audio_streamer.decode_full(codes_flat);
    if (callback.on_audio_chunk) {
        const int chunk_frames = 24;
        const int chunk_samples = chunk_frames * dac_->config().hop_length() * 4;
        for (int off = 0; off < static_cast<int>(full_audio.size()); off += chunk_samples) {
            int n = std::min(chunk_samples, static_cast<int>(full_audio.size()) - off);
            callback.on_audio_chunk(full_audio.data() + off, n);
            if (profiling.streamed_audio_chunks == 0 && profiling.first_audio_ms == 0.0)
                profiling.first_audio_ms = elapsed_ms(total_start);
            profiling.streamed_audio_chunks += 1;
        }
    }

    TTSOutput audio_result;
    audio_result.generated_codes = codes_flat;
    audio_result.sample_rate = dac_->config().sample_rate;
    audio_result.audio_samples = std::move(full_audio);
    const int audio_len = static_cast<int>(audio_result.audio_samples.size());

    spdlog::info("  Audio: {} samples ({:.2f}s)", audio_len, static_cast<double>(audio_len) / dac_->config().sample_rate);

    block_mgr_->free_blocks(seq);
    CUDA_CHECK(cudaFree(d_prompt)); CUDA_CHECK(cudaFree(d_block_table)); CUDA_CHECK(cudaFree(d_seq_len));
    CUDA_CHECK(cudaFree(d_hidden)); CUDA_CHECK(cudaFree(d_fast));
    CUDA_CHECK(cudaFree(d_logits)); CUDA_CHECK(cudaFree(d_tokens)); CUDA_CHECK(cudaFree(d_codebook_logits)); CUDA_CHECK(cudaFree(d_embed)); CUDA_CHECK(cudaFree(d_codebook_tokens));
    profiling.total_ms = elapsed_ms(total_start);
    audio_result.profiling = profiling;
    log_profiling_summary(audio_result.profiling);
    return audio_result;
}

std::vector<int32_t> InferencePipeline::build_ref_prompt(
    const int32_t* codes, int num_codebooks, int code_len,
    const std::string& ref_text, const std::string& target_text,
    int* prompt_len_out
) {
    int sem_begin = dual_ar_->config().semantic_begin_id;

    std::vector<int32_t> row0;
    append_ids(row0, tokenizer_->encode_raw(
        "<|im_start|>system\n"
        "convert the provided text to speech reference to the following:\n\n"
        "Text:\n"));
    append_ids(row0, tokenizer_->encode_raw(
        "<|speaker:0|>" + ref_text + "\n\nSpeech:\n"));

    const int vq_start = static_cast<int>(row0.size());
    for (int t = 0; t < code_len; ++t)
        row0.push_back(codes[t] + sem_begin);

    append_ids(row0, tokenizer_->encode_raw("<|im_end|>\n<|im_start|>user\n"));
    append_ids(row0, tokenizer_->encode_raw(target_text));
    append_ids(row0, tokenizer_->encode_raw("<|im_end|>\n<|im_start|>assistant\n<|voice|>"));

    const int cb_dim = num_codebooks + 1;
    const int prompt_len = static_cast<int>(row0.size());
    std::vector<int32_t> prompt(static_cast<size_t>(cb_dim) * prompt_len, 0);
    std::copy(row0.begin(), row0.end(), prompt.begin());
    for (int cb = 0; cb < num_codebooks; ++cb) {
        for (int t = 0; t < code_len; ++t) {
            prompt[static_cast<size_t>(cb + 1) * prompt_len + vq_start + t] =
                codes[static_cast<size_t>(cb) * code_len + t];
        }
    }
    if (prompt_len_out) *prompt_len_out = prompt_len;

    // Diagnostic: dump first/last few codes per codebook
    for (int cb = 0; cb < num_codebooks && cb < 3; cb++) {
        int cb_min = INT32_MAX, cb_max = INT32_MIN;
        for (int t = 0; t < code_len; t++) {
            int v = codes[static_cast<size_t>(cb) * code_len + t];
            if (v < cb_min) cb_min = v;
            if (v > cb_max) cb_max = v;
        }
        spdlog::info("  Ref codes cb[{}]: range [{}, {}] over {} frames", cb, cb_min, cb_max, code_len);
    }
    spdlog::info("  Prompt: cb_dim={} prompt_len={} vq_start={} row0[0..3]={} {} {} {}",
                 cb_dim, prompt_len, vq_start,
                 prompt[0], prompt[1], prompt[2], prompt[3]);
    spdlog::info("Built ref prompt in memory ({} cb, {} tokens, ref frames={})",
                 num_codebooks, prompt_len, code_len);
    return prompt;
}

TTSOutput InferencePipeline::run_with_ref_audio(
    const float* ref_audio, int ref_num_samples,
    const std::string& ref_text,
    const std::string& target_text,
    int max_new_tokens, float temperature, float top_p, int top_k, int seed,
    int chunk_length,
    int history_frames
) {
    return run_with_ref_audio_streaming(
        ref_audio, ref_num_samples, ref_text, target_text,
        max_new_tokens, temperature, top_p, top_k, seed, chunk_length, history_frames, {});
}

TTSOutput InferencePipeline::run_with_ref_audio_streaming(
    const float* ref_audio, int ref_num_samples,
    const std::string& ref_text,
    const std::string& target_text,
    int max_new_tokens, float temperature, float top_p, int top_k, int seed,
    int chunk_length,
    int history_frames,
    StreamCallback callback
) {
    const auto total_start = Clock::now();
    TTSProfiling ref_profiling;
    int num_codebooks = dual_ar_->config().num_codebooks;
    int code_len = 0;
    std::vector<int32_t> ref_codes;
    const std::size_t cache_key = hash_ref_audio(ref_audio, ref_num_samples);

    {
        std::lock_guard<std::mutex> lock(ref_code_cache_mutex_);
        auto it = ref_code_cache_.find(cache_key);
        if (it != ref_code_cache_.end()) {
            ref_codes = it->second.codes;
            code_len = it->second.code_len;
            ref_profiling.ref_cache_hit = true;
        }
    }

    if (!ref_profiling.ref_cache_hit) {
        const auto ref_encode_start = Clock::now();
        int max_cb = (ref_num_samples / (dac_->config().hop_length() * 4)) + 16;
        ref_codes.resize(static_cast<size_t>(DAC_TOTAL_CODEBOOKS) * max_cb);

        float* d_audio;
        CUDA_CHECK(cudaMalloc(&d_audio, ref_num_samples * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_audio, ref_audio, ref_num_samples * sizeof(float),
                              cudaMemcpyHostToDevice));
        dac_->encode(d_audio, 1, ref_num_samples, ref_codes.data(), &code_len);
        CUDA_CHECK(cudaFree(d_audio));
        ref_codes.resize(static_cast<size_t>(DAC_TOTAL_CODEBOOKS) * code_len);
        ref_profiling.ref_encode_ms = elapsed_ms(ref_encode_start);

        {
            std::lock_guard<std::mutex> lock(ref_code_cache_mutex_);
            if (ref_code_cache_.size() >= kMaxRefCodeCacheEntries) {
                ref_code_cache_.clear();
            }
            ref_code_cache_[cache_key] = RefCodeCacheEntry{ref_codes, code_len};
        }
        spdlog::info("Encoded ref audio: {} samples -> {} code frames", ref_num_samples, code_len);
    } else {
        spdlog::info("Reused cached ref audio encoding: {} samples -> {} code frames",
                     ref_num_samples, code_len);
    }

    ref_profiling.ref_code_frames = code_len;

    const auto prompt_build_start = Clock::now();
    const int cb_dim = num_codebooks + 1;
    const int sem_begin = dual_ar_->config().semantic_begin_id;
    std::vector<std::string> text_chunks = split_text_chunks(target_text, chunk_length);
    std::vector<std::vector<int32_t>> rows(cb_dim);

    append_text_segment(rows, tokenizer_->encode_raw(
        "<|im_start|>system\n"
        "convert the provided text to speech reference to the following:\n\n"
        "Text:\n"));
    append_text_segment(rows, tokenizer_->encode_raw(
        "<|speaker:0|>" + ref_text + "\n\nSpeech:\n"));
    append_code_segment(rows, ref_codes, num_codebooks, code_len, sem_begin);
    append_text_segment(rows, tokenizer_->encode_raw("<|im_end|>\n<|im_start|>user\n"));
    append_text_segment(rows, tokenizer_->encode_raw(text_chunks.front()));
    append_text_segment(rows, tokenizer_->encode_raw("<|im_end|>\n<|im_start|>assistant\n<|voice|>"));
    ref_profiling.prompt_build_ms = elapsed_ms(prompt_build_start);

    TTSOutput aggregate;
    aggregate.sample_rate = dac_->config().sample_rate;
    aggregate.profiling.ref_encode_ms = ref_profiling.ref_encode_ms;
    aggregate.profiling.prompt_build_ms = ref_profiling.prompt_build_ms;
    aggregate.profiling.ref_code_frames = ref_profiling.ref_code_frames;
    aggregate.profiling.ref_cache_hit = ref_profiling.ref_cache_hit;
    std::vector<int32_t> prev_chunk_codes;
    int prev_chunk_frames = 0;

    for (std::size_t chunk_idx = 0; chunk_idx < text_chunks.size(); ++chunk_idx) {
        if (chunk_idx > 0) {
            std::vector<int32_t> history_codes = tail_code_frames(
                prev_chunk_codes,
                num_codebooks,
                prev_chunk_frames,
                history_frames);
            const int history_len = history_frames > 0
                ? std::min(prev_chunk_frames, history_frames)
                : prev_chunk_frames;
            append_code_segment(
                rows,
                history_codes,
                num_codebooks,
                history_len,
                sem_begin);
            append_text_segment(rows, tokenizer_->encode_raw("<|im_end|>\n<|im_start|>user\n"));
            append_text_segment(rows, tokenizer_->encode_raw(text_chunks[chunk_idx]));
            append_text_segment(rows, tokenizer_->encode_raw("<|im_end|>\n<|im_start|>assistant\n<|voice|>"));
        }

        const int prompt_len = static_cast<int>(rows[0].size());
        std::vector<int32_t> prompt = flatten_rows(rows);
        TTSOutput chunk_result = run_with_prompt_tensor(
            prompt,
            num_codebooks,
            prompt_len,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            seed + static_cast<int>(chunk_idx) * 9973,
            callback);

        aggregate.audio_samples.insert(
            aggregate.audio_samples.end(),
            chunk_result.audio_samples.begin(),
            chunk_result.audio_samples.end());
        prev_chunk_frames = chunk_result.profiling.generated_frames;
        prev_chunk_codes = chunk_result.generated_codes;
        aggregate.generated_codes = std::move(chunk_result.generated_codes);
        aggregate.profiling.tokenize_ms += chunk_result.profiling.tokenize_ms;
        aggregate.profiling.prefill_ms += chunk_result.profiling.prefill_ms;
        aggregate.profiling.ar_decode_ms += chunk_result.profiling.ar_decode_ms;
        aggregate.profiling.dac_decode_ms += chunk_result.profiling.dac_decode_ms;
        aggregate.profiling.audio_copy_ms += chunk_result.profiling.audio_copy_ms;
        aggregate.profiling.response_encode_ms += chunk_result.profiling.response_encode_ms;
        aggregate.profiling.prompt_tokens = prompt_len;
        aggregate.profiling.generated_frames += chunk_result.profiling.generated_frames;
    }

    aggregate.profiling.total_ms = elapsed_ms(total_start);
    return aggregate;
}

}  // namespace fish
