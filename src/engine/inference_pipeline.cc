// src/engine/inference_pipeline.cc
#include "engine/inference_pipeline.h"
#include "kernels/kernels.h"
#include "utils/cuda_utils.h"
#include <spdlog/spdlog.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <fstream>
#include <numeric>
#include <random>

namespace fish {

namespace {

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

    // ── Use DualAR's stream for ALL GPU operations (no separate pipeline stream) ──
    cudaStream_t stream = dual_ar_->stream();

    int dim = dual_ar_->config().dim;
    int num_codebooks = dual_ar_->config().num_codebooks;
    int codebook_dim = 1 + num_codebooks;
    int vocab_size = dual_ar_->config().vocab_size;

    // Step 1: Tokenize
    auto token_ids = tokenizer_->encode(text);
    int prompt_len = static_cast<int>(token_ids.size());
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

    dual_ar_->prefill(d_prompt, 1, prompt_len, 1,
                      block_mgr_->k_cache(), block_mgr_->v_cache(),
                      d_block_table, d_seq_len,
                      d_hidden, d_fast);

    // Diagnostic: verify prefill hidden state varies with input
    {
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

    // Pre-allocate logits and token buffers on GPU
    __half* d_logits;
    int32_t* d_tokens;
    __half* d_codebook_logits;
    __half* d_embed;  // combined embedding for each decode step
    CUDA_CHECK(cudaMalloc(&d_logits,          vocab_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_tokens,           sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_codebook_logits,
                          dual_ar_->config().codebook_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_embed,            dim * sizeof(__half)));

    // CPU sampling: copy fp16 logits → host, do softmax/top-k/top-p there.
    // Avoids the shared-memory limitation when vocab_size > ~6000.
    //
    // Matches Python reference (logits_to_probs + sample in inference.py):
    //   1. Sort logits descending
    //   2. softmax(sorted) → cumsum → top-p cutoff
    //   3. Mask tokens beyond top-k or beyond top-p with -inf
    //   4. Apply temperature AFTER filtering
    //   5. softmax → multinomial sample
    std::vector<float> h_slice;
    std::vector<__half> h_logits_fp16;

    auto sample_range = [&](int sz, int id_start, float temp, float tp, int tk, uint64_t rng) -> int32_t {
        int total = id_start + sz;
        h_logits_fp16.resize(total);
        CUDA_CHECK(cudaMemcpy(h_logits_fp16.data(), d_logits,
                              total * sizeof(__half),
                              cudaMemcpyDeviceToHost));

        // Convert slice to fp32 (RAW logits, no temperature yet)
        h_slice.resize(sz);
        for (int i = 0; i < sz; i++)
            h_slice[i] = __half2float(h_logits_fp16[id_start + i]);

        // Step 1-2: sort descending and compute softmax on sorted logits
        std::vector<int> sorted_idx(sz);
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::sort(sorted_idx.begin(), sorted_idx.end(),
                  [&](int a, int b) { return h_slice[a] > h_slice[b]; });

        // softmax(sorted) for cumsum → top-p cutoff
        std::vector<float> sorted_logits(sz);
        float max_l = -1e30f;
        for (int i = 0; i < sz; i++) {
            sorted_logits[i] = h_slice[sorted_idx[i]];
            if (sorted_logits[i] > max_l) max_l = sorted_logits[i];
        }
        float sum_e = 0.f;
        for (int i = 0; i < sz; i++) {
            sorted_logits[i] = std::exp(sorted_logits[i] - max_l);
            sum_e += sorted_logits[i];
        }
        for (int i = 0; i < sz; i++) sorted_logits[i] /= sum_e;

        // Step 3: top-p cumulative sum + top-k mask
        // Python: ensure at least index 0 survives
        std::vector<bool> to_remove(sz, true);  // default: remove
        float cumsum = 0.f;
        int actual_k = std::min(tk, sz);
        for (int i = 0; i < sz; i++) {
            cumsum += sorted_logits[i];
            bool beyond_topk = (i >= actual_k);
            bool beyond_topp = (i > 0 && cumsum > tp);  // i==0 always kept
            if (beyond_topk || beyond_topp) break;
            to_remove[sorted_idx[i]] = false;
        }

        // Step 4: mask logits → apply temperature AFTER filtering
        float inv_temp = 1.0f / std::max(temp, 1e-5f);
        for (int i = 0; i < sz; i++) {
            if (to_remove[i])
                h_slice[i] = -1e30f;  // effectively -inf
            else
                h_slice[i] *= inv_temp;
        }

        // Step 5: softmax on masked+scaled logits
        max_l = -1e30f;
        for (int i = 0; i < sz; i++)
            if (h_slice[i] > max_l) max_l = h_slice[i];
        sum_e = 0.f;
        for (int i = 0; i < sz; i++) {
            h_slice[i] = std::exp(h_slice[i] - max_l);
            sum_e += h_slice[i];
        }
        for (int i = 0; i < sz; i++) h_slice[i] /= sum_e;

        // Step 6: multinomial sample
        std::mt19937_64 gen(rng);
        std::uniform_real_distribution<float> dist(0.f, 1.f);
        float r = dist(gen);
        float cs = 0.f;
        for (int i = 0; i < sz; i++) {
            cs += h_slice[i];
            if (r < cs) return static_cast<int32_t>(id_start + i);
        }
        return static_cast<int32_t>(id_start);  // fallback
    };

    // Semantic sampling: allow [sem_start, sem_end] + eos_token
    int sem_start = dual_ar_->config().semantic_begin_id;  // 151678
    int sem_end   = dual_ar_->config().semantic_end_id;    // 155773
    int eos_id    = dual_ar_->config().im_end_token_id;    // 151645
    int sem_range = sem_end - sem_start + 1;

    auto sample_semantic = [&](float temp, float tp, int tk, uint64_t rng) -> int32_t {
        // Copy both semantic range logits AND eos logit from GPU
        int total = sem_start + sem_range;  // covers up to sem_end
        h_logits_fp16.resize(total + 1);    // +1 for eos which may be outside range
        CUDA_CHECK(cudaMemcpy(h_logits_fp16.data(), d_logits,
                              total * sizeof(__half),
                              cudaMemcpyDeviceToHost));
        // Also copy eos logit
        CUDA_CHECK(cudaMemcpy(&h_logits_fp16[total], d_logits + eos_id,
                              sizeof(__half),
                              cudaMemcpyDeviceToHost));

        // Build combined array: [sem_range semantics] + [1 eos]
        int sz = sem_range + 1;
        h_slice.resize(sz);
        for (int i = 0; i < sem_range; i++)
            h_slice[i] = __half2float(h_logits_fp16[sem_start + i]);
        h_slice[sem_range] = __half2float(h_logits_fp16[total]);  // eos logit

        // Same Python-compatible sampling as sample_range
        std::vector<int> sorted_idx(sz);
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::sort(sorted_idx.begin(), sorted_idx.end(),
                  [&](int a, int b) { return h_slice[a] > h_slice[b]; });

        std::vector<float> sorted_logits(sz);
        float max_l = -1e30f;
        for (int i = 0; i < sz; i++) {
            sorted_logits[i] = h_slice[sorted_idx[i]];
            if (sorted_logits[i] > max_l) max_l = sorted_logits[i];
        }
        float sum_e = 0.f;
        for (int i = 0; i < sz; i++) { sorted_logits[i] = std::exp(sorted_logits[i] - max_l); sum_e += sorted_logits[i]; }
        for (int i = 0; i < sz; i++) sorted_logits[i] /= sum_e;

        std::vector<bool> to_remove(sz, true);
        float cumsum = 0.f;
        int actual_k = std::min(tk, sz);
        for (int i = 0; i < sz; i++) {
            cumsum += sorted_logits[i];
            bool beyond_topk = (i >= actual_k);
            bool beyond_topp = (i > 0 && cumsum > tp);
            if (beyond_topk || beyond_topp) break;
            to_remove[sorted_idx[i]] = false;
        }

        float inv_temp = 1.0f / std::max(temp, 1e-5f);
        for (int i = 0; i < sz; i++) {
            if (to_remove[i]) h_slice[i] = -1e30f;
            else h_slice[i] *= inv_temp;
        }

        max_l = -1e30f;
        for (int i = 0; i < sz; i++)
            if (h_slice[i] > max_l) max_l = h_slice[i];
        sum_e = 0.f;
        for (int i = 0; i < sz; i++) { h_slice[i] = std::exp(h_slice[i] - max_l); sum_e += h_slice[i]; }
        for (int i = 0; i < sz; i++) h_slice[i] /= sum_e;

        std::mt19937_64 gen(rng);
        std::uniform_real_distribution<float> dist(0.f, 1.f);
        float r = dist(gen);
        float cs = 0.f;
        for (int i = 0; i < sz; i++) {
            cs += h_slice[i];
            if (r < cs) {
                if (i < sem_range) return static_cast<int32_t>(sem_start + i);
                else return static_cast<int32_t>(eos_id);
            }
        }
        return static_cast<int32_t>(sem_start);
    };

    // ---- Prefill logits → first semantic token ----
    dual_ar_->get_logits(d_hidden, d_logits, 1);

    // Debug: dump top-10 semantic logits after prefill
    {
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
        out_cbs.resize(num_codebooks);
        // position 0: populate fast KV cache (logits discarded in Python too)
        dual_ar_->fast_codebook_decode(d_hidden, d_embed, d_codebook_logits,
                                       1, 0, 0);
        // cb[0] = deterministic: clamp(sem - sem_begin, 0, cb_size-1)
        out_cbs[0] = std::max(0, std::min(cb_size - 1, cur_sem - sem_begin));
        int32_t prev_cb = out_cbs[0];
        // cb[1..num_codebooks-1]: sample from fast decoder with FULL logits
        // (Python does NOT clamp to acoustic range — model learns to avoid invalid codes)
        for (int cb = 1; cb < num_codebooks; cb++) {
            dual_ar_->fast_codebook_decode(d_hidden, d_embed, d_codebook_logits,
                                           1, cb, prev_cb);
            CUDA_CHECK(cudaMemcpy(d_logits, d_codebook_logits,
                                  cb_size * sizeof(__half),
                                  cudaMemcpyDeviceToDevice));
            // Sample from full [0, cb_size) range (matches Python), using pipeline params
            prev_cb = sample_range(cb_size, 0, cb_temp, cb_top_p, cb_top_k, base_rng + cb);
            out_cbs[cb] = prev_cb;
        }
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
    for (int step = 0; step < max_new_tokens; step++) {
        // 1. Embed [sem, cbs] for THIS frame (same step)
        dual_ar_->embed_for_decode(sem, curr_cbs.data(), d_embed, stream);

        // seq_len must include the current position being written (prompt_len + step),
        // so the attention covers positions 0..prompt_len+step (inclusive).
        int32_t cur_seq_len = prompt_len + step + 1;
        CUDA_CHECK(cudaMemcpy(d_seq_len, &cur_seq_len, sizeof(int32_t),
                              cudaMemcpyHostToDevice));

        // 2. Slow decode at position prompt_len+step
        dual_ar_->decode_step(d_embed, 1, prompt_len + step,
                              block_mgr_->k_cache(), block_mgr_->v_cache(),
                              d_block_table, d_seq_len,
                              d_hidden, d_hidden);

        // 3. Sample next semantic token (normal + RAS high-temp fallback)
        dual_ar_->get_logits(d_hidden, d_logits, 1);
        int32_t sem_normal = sample_semantic(temperature, top_p, top_k,
                                              static_cast<uint64_t>(seed) + step + 1);
        int32_t sem_high = sample_semantic(RAS_HIGH_TEMP, RAS_HIGH_TOP_P, top_k,
                                            static_cast<uint64_t>(seed) + step + 1 + 1000000);
        // RAS: if sem_normal is in semantic range AND appeared in recent history, use high-temp fallback
        bool is_semantic = (sem_normal >= sem_start && sem_normal <= sem_end);
        bool in_window = false;
        if (is_semantic) {
            for (int w = 0; w < RAS_WIN_SIZE; w++) {
                if (sem_history[w] == sem_normal) { in_window = true; break; }
            }
        }
        int32_t next_sem = (is_semantic && in_window) ? sem_high : sem_normal;

        // 4. Generate NEXT frame's codebooks from this hidden state
        std::vector<int32_t> next_cbs;
        gen_codebooks(next_sem, static_cast<uint64_t>(seed) + (step + 1) * num_codebooks,
                      next_cbs, temperature, top_p, top_k);

        // 5. Record current frame
        generated.push_back(sem);
        for (auto c : curr_cbs) generated.push_back(c);

        // Streaming: report progress
        if (callback.on_progress) {
            callback.on_progress(step, max_new_tokens);
        }

        // Diagnostic: log first 5 semantic tokens + codebook ranges
        if (step < 5) {
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
    }
    decode_done:
    CUDA_CHECK(cudaDeviceSynchronize());

    int code_len = static_cast<int>(generated.size()) / (1 + num_codebooks);
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
        return {std::vector<float>(), dac_->config().sample_rate};
    }

    // Step 7: DAC decode
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

    int32_t* d_codes;
    float* d_audio;
    CUDA_CHECK(cudaMalloc(&d_codes, codes_flat.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_codes, codes_flat.data(),
                          codes_flat.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    // 2048 samples per code frame (encoder_hop=512 × quantizer_downsample=4), 2× headroom
    int max_out = code_len * 2048 * 2;
    CUDA_CHECK(cudaMalloc(&d_audio, max_out * sizeof(float)));

    int audio_len = 0;
    dac_->decode(d_codes, 1, code_len, d_audio, &audio_len, max_out);

    std::vector<float> h_audio(audio_len);
    CUDA_CHECK(cudaMemcpy(h_audio.data(), d_audio,
                          audio_len * sizeof(float), cudaMemcpyDeviceToHost));

    // Streaming: emit audio chunks (~50ms each)
    if (callback.on_audio_chunk) {
        constexpr int kChunkSize = 2205;  // ~50ms at 44.1kHz
        int offset = 0;
        while (offset < audio_len) {
            int chunk = std::min(kChunkSize, audio_len - offset);
            callback.on_audio_chunk(h_audio.data() + offset, chunk);
            offset += chunk;
        }
    }

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
    CUDA_CHECK(cudaFree(d_codes));
    CUDA_CHECK(cudaFree(d_audio));
    CUDA_CHECK(cudaDeviceSynchronize());

    return {h_audio, dac_->config().sample_rate};
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

    // ── Read prompt binary file ──
    // Format: int32 N, int32 T, int32[(N+1)*T] row-major
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
    pf.close();
    spdlog::info("  Prompt file: num_codebooks={} prompt_len={}", file_num_cb, prompt_len);

    cudaStream_t stream = dual_ar_->stream();
    int dim = dual_ar_->config().dim;
    int num_codebooks = file_num_cb;
    int vocab_size = dual_ar_->config().vocab_size;

    // Upload full prompt [cb_dim, prompt_len] to GPU
    int32_t* d_prompt;
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

    // Prefill — passes the FULL [cb_dim, prompt_len] tensor
    // The modified prefill reads all rows for combined embedding
    dual_ar_->prefill(d_prompt, 1, prompt_len, cb_dim,
                      block_mgr_->k_cache(), block_mgr_->v_cache(),
                      d_block_table, d_seq_len,
                      d_hidden, d_fast);

    // ---- Same decode loop as run() ----
    int eos_id    = dual_ar_->config().im_end_token_id;
    int sem_start = dual_ar_->config().semantic_begin_id;
    int sem_end   = dual_ar_->config().semantic_end_id;
    int sem_range = sem_end - sem_start + 1;

    std::vector<int32_t> generated;
    __half* d_logits;
    __half* d_codebook_logits;
    __half* d_embed;
    CUDA_CHECK(cudaMalloc(&d_logits,          vocab_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_codebook_logits, dual_ar_->config().codebook_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_embed,           dim * sizeof(__half)));

    std::vector<float> h_slice;
    std::vector<__half> h_logits_fp16;

    auto sample_range = [&](int sz, int id_start, float temp, float tp, int tk, uint64_t rng) -> int32_t {
        int total = id_start + sz;
        h_logits_fp16.resize(total);
        CUDA_CHECK(cudaMemcpy(h_logits_fp16.data(), d_logits, total * sizeof(__half), cudaMemcpyDeviceToHost));
        h_slice.resize(sz);
        for (int i = 0; i < sz; i++) h_slice[i] = __half2float(h_logits_fp16[id_start + i]);
        std::vector<int> sorted_idx(sz);
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b) { return h_slice[a] > h_slice[b]; });
        std::vector<float> sorted_logits(sz);
        float max_l = -1e30f;
        for (int i = 0; i < sz; i++) { sorted_logits[i] = h_slice[sorted_idx[i]]; if (sorted_logits[i] > max_l) max_l = sorted_logits[i]; }
        float sum_e = 0.f;
        for (int i = 0; i < sz; i++) { sorted_logits[i] = std::exp(sorted_logits[i] - max_l); sum_e += sorted_logits[i]; }
        for (int i = 0; i < sz; i++) sorted_logits[i] /= sum_e;
        std::vector<bool> to_remove(sz, true);
        float cumsum = 0.f; int actual_k = std::min(tk, sz);
        for (int i = 0; i < sz; i++) { cumsum += sorted_logits[i]; if ((i >= actual_k) || (i > 0 && cumsum > tp)) break; to_remove[sorted_idx[i]] = false; }
        float inv_temp = 1.0f / std::max(temp, 1e-5f);
        for (int i = 0; i < sz; i++) { if (to_remove[i]) h_slice[i] = -1e30f; else h_slice[i] *= inv_temp; }
        max_l = -1e30f;
        for (int i = 0; i < sz; i++) if (h_slice[i] > max_l) max_l = h_slice[i];
        sum_e = 0.f;
        for (int i = 0; i < sz; i++) { h_slice[i] = std::exp(h_slice[i] - max_l); sum_e += h_slice[i]; }
        for (int i = 0; i < sz; i++) h_slice[i] /= sum_e;
        std::mt19937_64 gen(rng);
        std::uniform_real_distribution<float> dist(0.f, 1.f);
        float r = dist(gen); float cs = 0.f;
        for (int i = 0; i < sz; i++) { cs += h_slice[i]; if (r < cs) return static_cast<int32_t>(id_start + i); }
        return static_cast<int32_t>(id_start);
    };

    auto sample_semantic = [&](float temp, float tp, int tk, uint64_t rng) -> int32_t {
        int total = sem_start + sem_range;
        h_logits_fp16.resize(total + 1);
        CUDA_CHECK(cudaMemcpy(h_logits_fp16.data(), d_logits, total * sizeof(__half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_logits_fp16[total], d_logits + eos_id, sizeof(__half), cudaMemcpyDeviceToHost));
        int sz = sem_range + 1;
        h_slice.resize(sz);
        for (int i = 0; i < sem_range; i++) h_slice[i] = __half2float(h_logits_fp16[sem_start + i]);
        h_slice[sem_range] = __half2float(h_logits_fp16[total]);
        // Same top-k/top-p/temp logic as sample_range (abbreviated for clarity)
        std::vector<int> sorted_idx(sz); std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b) { return h_slice[a] > h_slice[b]; });
        std::vector<float> sl(sz); float ml = -1e30f;
        for (int i = 0; i < sz; i++) { sl[i] = h_slice[sorted_idx[i]]; if (sl[i] > ml) ml = sl[i]; }
        float se = 0.f; for (int i = 0; i < sz; i++) { sl[i] = std::exp(sl[i] - ml); se += sl[i]; }
        for (int i = 0; i < sz; i++) sl[i] /= se;
        std::vector<bool> tr(sz, true); float cs2 = 0.f; int ak = std::min(tk, sz);
        for (int i = 0; i < sz; i++) { cs2 += sl[i]; if ((i >= ak) || (i > 0 && cs2 > tp)) break; tr[sorted_idx[i]] = false; }
        float it = 1.0f / std::max(temp, 1e-5f);
        for (int i = 0; i < sz; i++) { if (tr[i]) h_slice[i] = -1e30f; else h_slice[i] *= it; }
        ml = -1e30f; for (int i = 0; i < sz; i++) if (h_slice[i] > ml) ml = h_slice[i];
        se = 0.f; for (int i = 0; i < sz; i++) { h_slice[i] = std::exp(h_slice[i] - ml); se += h_slice[i]; }
        for (int i = 0; i < sz; i++) h_slice[i] /= se;
        std::mt19937_64 gen(rng); std::uniform_real_distribution<float> dist(0.f, 1.f);
        float r = dist(gen); float cs = 0.f;
        for (int i = 0; i < sz; i++) { cs += h_slice[i]; if (r < cs) { if (i < sem_range) return static_cast<int32_t>(sem_start + i); else return static_cast<int32_t>(eos_id); } }
        return static_cast<int32_t>(sem_start);
    };

    dual_ar_->get_logits(d_hidden, d_logits, 1);
    int32_t sem = sample_semantic(temperature, top_p, top_k, static_cast<uint64_t>(seed));
    spdlog::info("  Prefill → first sem token: {}", sem);
    if (sem == eos_id) { spdlog::info("  EOS after prefill"); goto prompt_decode_done; }

    {
    int cb_size = dual_ar_->config().codebook_size;
    int sem_begin = dual_ar_->config().semantic_begin_id;
    auto gen_codebooks = [&](int32_t cur_sem, uint64_t base_rng, std::vector<int32_t>& out_cbs, float cbt, float cbp, int cbk) {
        out_cbs.resize(num_codebooks);
        dual_ar_->fast_codebook_decode(d_hidden, d_embed, d_codebook_logits, 1, 0, 0);
        out_cbs[0] = std::max(0, std::min(cb_size - 1, cur_sem - sem_begin));
        int32_t prev_cb = out_cbs[0];
        for (int cb = 1; cb < num_codebooks; cb++) {
            dual_ar_->fast_codebook_decode(d_hidden, d_embed, d_codebook_logits, 1, cb, prev_cb);
            CUDA_CHECK(cudaMemcpy(d_logits, d_codebook_logits, cb_size * sizeof(__half), cudaMemcpyDeviceToDevice));
            prev_cb = sample_range(cb_size, 0, cbt, cbp, cbk, base_rng + cb);
            out_cbs[cb] = prev_cb;
        }
    };

    std::vector<int32_t> curr_cbs;
    gen_codebooks(sem, static_cast<uint64_t>(seed), curr_cbs, temperature, top_p, top_k);

    static constexpr int RAS_WIN_SIZE = 10;
    static constexpr float RAS_HIGH_TEMP = 1.0f, RAS_HIGH_TOP_P = 0.9f;
    std::vector<int32_t> sem_history(RAS_WIN_SIZE, -1);
    sem_history[0] = sem;

    for (int step = 0; step < max_new_tokens; step++) {
        dual_ar_->embed_for_decode(sem, curr_cbs.data(), d_embed, stream);
        int32_t cur_seq_len = prompt_len + step + 1;
        CUDA_CHECK(cudaMemcpy(d_seq_len, &cur_seq_len, sizeof(int32_t), cudaMemcpyHostToDevice));
        dual_ar_->decode_step(d_embed, 1, prompt_len + step,
                              block_mgr_->k_cache(), block_mgr_->v_cache(),
                              d_block_table, d_seq_len, d_hidden, d_hidden);
        dual_ar_->get_logits(d_hidden, d_logits, 1);
        int32_t sem_normal = sample_semantic(temperature, top_p, top_k, static_cast<uint64_t>(seed) + step + 1);
        int32_t sem_high   = sample_semantic(RAS_HIGH_TEMP, RAS_HIGH_TOP_P, top_k, static_cast<uint64_t>(seed) + step + 1 + 1000000);
        bool is_sem = (sem_normal >= sem_start && sem_normal <= sem_end);
        bool in_win = false;
        if (is_sem) { for (int w = 0; w < RAS_WIN_SIZE; w++) if (sem_history[w] == sem_normal) { in_win = true; break; } }
        int32_t next_sem = (is_sem && in_win) ? sem_high : sem_normal;
        std::vector<int32_t> next_cbs;
        gen_codebooks(next_sem, static_cast<uint64_t>(seed) + (step + 1) * num_codebooks, next_cbs, temperature, top_p, top_k);
        generated.push_back(sem);
        for (auto c : curr_cbs) generated.push_back(c);
        for (int w = 0; w < RAS_WIN_SIZE - 1; w++) sem_history[w] = sem_history[w + 1];
        sem_history[RAS_WIN_SIZE - 1] = next_sem;
        sem = next_sem; curr_cbs = std::move(next_cbs);
        if (sem == eos_id) break;
    }
    }
    prompt_decode_done:
    CUDA_CHECK(cudaDeviceSynchronize());

    int code_len = static_cast<int>(generated.size()) / (1 + num_codebooks);
    spdlog::info("  Generated {} frames", code_len);
    if (code_len == 0) {
        block_mgr_->free_blocks(seq);
        CUDA_CHECK(cudaFree(d_prompt)); CUDA_CHECK(cudaFree(d_block_table)); CUDA_CHECK(cudaFree(d_seq_len));
        CUDA_CHECK(cudaFree(d_hidden)); CUDA_CHECK(cudaFree(d_fast));
        CUDA_CHECK(cudaFree(d_logits)); CUDA_CHECK(cudaFree(d_codebook_logits)); CUDA_CHECK(cudaFree(d_embed));
        return {std::vector<float>(), dac_->config().sample_rate};
    }

    int stride = 1 + num_codebooks;
    std::vector<int32_t> codes_flat(num_codebooks * code_len, 0);
    for (int i = 0; i < code_len; i++)
        for (int cb = 0; cb < num_codebooks; cb++)
            codes_flat[cb * code_len + i] = generated[i * stride + 1 + cb];
    maybe_dump_codes(codes_flat, 1, num_codebooks, code_len);

    int32_t* d_codes; float* d_audio;
    CUDA_CHECK(cudaMalloc(&d_codes, codes_flat.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_codes, codes_flat.data(), codes_flat.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    int max_out = code_len * 2048 * 2;
    CUDA_CHECK(cudaMalloc(&d_audio, max_out * sizeof(float)));
    int audio_len = 0;
    dac_->decode(d_codes, 1, code_len, d_audio, &audio_len, max_out);
    std::vector<float> h_audio(audio_len);
    CUDA_CHECK(cudaMemcpy(h_audio.data(), d_audio, audio_len * sizeof(float), cudaMemcpyDeviceToHost));
    spdlog::info("  Audio: {} samples ({:.2f}s)", audio_len, static_cast<double>(audio_len) / dac_->config().sample_rate);

    block_mgr_->free_blocks(seq);
    CUDA_CHECK(cudaFree(d_prompt)); CUDA_CHECK(cudaFree(d_block_table)); CUDA_CHECK(cudaFree(d_seq_len));
    CUDA_CHECK(cudaFree(d_hidden)); CUDA_CHECK(cudaFree(d_fast));
    CUDA_CHECK(cudaFree(d_logits)); CUDA_CHECK(cudaFree(d_codebook_logits)); CUDA_CHECK(cudaFree(d_embed));
    CUDA_CHECK(cudaFree(d_codes)); CUDA_CHECK(cudaFree(d_audio));
    CUDA_CHECK(cudaDeviceSynchronize());
    return {h_audio, dac_->config().sample_rate};
}

std::string InferencePipeline::build_ref_prompt_file(
    const int32_t* codes, int num_codebooks, int code_len,
    const std::string& ref_text, const std::string& target_text
) {
    auto append_ids = [](std::vector<int32_t>& out, const std::vector<int>& ids) {
        for (int id : ids) out.push_back(static_cast<int32_t>(id));
    };

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

    std::filesystem::create_directories("/tmp/fish-audio-cpp");
    std::string prompt_path = "/tmp/fish-audio-cpp/reference_prompt_api.bin";
    std::ofstream pf(prompt_path, std::ios::binary);
    if (!pf.good())
        throw std::runtime_error("Cannot write prompt file: " + prompt_path);
    int32_t hdr[2] = {num_codebooks, prompt_len};
    pf.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    pf.write(reinterpret_cast<const char*>(prompt.data()), prompt.size() * sizeof(int32_t));
    pf.close();

    spdlog::info("Built ref prompt: {} ({} cb, {} tokens, ref frames={})",
                 prompt_path, num_codebooks, prompt_len, code_len);
    return prompt_path;
}

TTSOutput InferencePipeline::run_with_ref_audio(
    const float* ref_audio, int ref_num_samples,
    const std::string& ref_text,
    const std::string& target_text,
    int max_new_tokens, float temperature, float top_p, int top_k, int seed
) {
    int num_codebooks = dual_ar_->config().num_codebooks;
    int max_cb = (ref_num_samples / (dac_->config().hop_length() * 4)) + 16;
    std::vector<int32_t> ref_codes(static_cast<size_t>(DAC_TOTAL_CODEBOOKS) * max_cb);
    int code_len = 0;

    // Upload audio to GPU, encode — codes returned to host by DAC
    float* d_audio;
    CUDA_CHECK(cudaMalloc(&d_audio, ref_num_samples * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_audio, ref_audio, ref_num_samples * sizeof(float),
                          cudaMemcpyHostToDevice));
    dac_->encode(d_audio, 1, ref_num_samples, ref_codes.data(), &code_len);
    CUDA_CHECK(cudaFree(d_audio));

    spdlog::info("Encoded ref audio: {} samples → {} code frames", ref_num_samples, code_len);

    std::string prompt_path = build_ref_prompt_file(
        ref_codes.data(), num_codebooks, code_len, ref_text, target_text);

    return run_with_prompt_file(prompt_path, max_new_tokens, temperature, top_p, top_k, seed);
}

TTSOutput InferencePipeline::run_with_ref_audio_streaming(
    const float* ref_audio, int ref_num_samples,
    const std::string& ref_text,
    const std::string& target_text,
    int max_new_tokens, float temperature, float top_p, int top_k, int seed,
    StreamCallback callback
) {
    // Encode and build prompt (reuse non-streaming path)
    int num_codebooks = dual_ar_->config().num_codebooks;
    int max_cb = (ref_num_samples / (dac_->config().hop_length() * 4)) + 16;
    std::vector<int32_t> ref_codes(static_cast<size_t>(DAC_TOTAL_CODEBOOKS) * max_cb);
    int code_len = 0;

    float* d_audio;
    CUDA_CHECK(cudaMalloc(&d_audio, ref_num_samples * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_audio, ref_audio, ref_num_samples * sizeof(float),
                          cudaMemcpyHostToDevice));
    dac_->encode(d_audio, 1, ref_num_samples, ref_codes.data(), &code_len);
    CUDA_CHECK(cudaFree(d_audio));

    spdlog::info("Encoded ref audio: {} samples → {} code frames", ref_num_samples, code_len);

    std::string prompt_path = build_ref_prompt_file(
        ref_codes.data(), num_codebooks, code_len, ref_text, target_text);

    // For now, delegate to non-streaming and then manually chunk the result.
    TTSOutput result = run_with_prompt_file(prompt_path, max_new_tokens,
                                             temperature, top_p, top_k, seed);

    // Emit audio chunks
    if (callback.on_audio_chunk && !result.audio_samples.empty()) {
        constexpr int kChunkSize = 2205;
        const float* data = result.audio_samples.data();
        int total = static_cast<int>(result.audio_samples.size());
        for (int off = 0; off < total; off += kChunkSize) {
            int n = std::min(kChunkSize, total - off);
            callback.on_audio_chunk(data + off, n);
        }
    }
    return result;
}

}  // namespace fish
