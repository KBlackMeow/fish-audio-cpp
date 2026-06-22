// src/main.cc — Fish Audio S2 Pro TTS inference (pure C++)
//
// Pipeline (C++ throughout):
//   1. DualAREngine (Qwen3 transformer)  — text → VQ codes
//   2. DACEngine (audio codec)           — VQ codes → PCM audio
//   3. write_wav()                       — float32 PCM → WAV

#include "engine/inference_pipeline.h"
#include "engine/dual_ar_engine.h"
#include "engine/dac_engine.h"
#include "engine/block_manager.h"
#include "model/dual_ar_config.h"
#include "model/dac_config.h"
#include "model/loader.h"
#include "tokenizer/tokenizer.h"
#include "utils/cuda_utils.h"
#include "server/http_server.h"
#include <spdlog/spdlog.h>
#include <cxxopts.hpp>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <numeric>

// ---------------------------------------------------------------------------
// WAV writer (float32 PCM → 16-bit WAV)
// ---------------------------------------------------------------------------
static void write_wav(const std::string& path, const float* samples, int n_samples,
                      int sample_rate) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { spdlog::error("Cannot open WAV output: {}", path); return; }

    std::vector<int16_t> pcm(n_samples);
    for (int i = 0; i < n_samples; i++) {
        float v = std::max(-1.0f, std::min(1.0f, samples[i]));
        pcm[i] = static_cast<int16_t>(v * 32767.0f);
    }

    uint32_t data_bytes  = static_cast<uint32_t>(n_samples) * 2;
    uint32_t riff_size   = 36 + data_bytes;
    uint16_t num_ch      = 1;
    uint32_t byte_rate   = static_cast<uint32_t>(sample_rate) * 2;
    uint16_t block_align = 2;
    uint16_t bits        = 16;
    uint16_t audio_fmt   = 1;

    fwrite("RIFF",       1, 4, f); fwrite(&riff_size,   4, 1, f);
    fwrite("WAVE",       1, 4, f); fwrite("fmt ",       1, 4, f);
    uint32_t fmt_sz = 16;
    fwrite(&fmt_sz,      4, 1, f); fwrite(&audio_fmt,   2, 1, f);
    fwrite(&num_ch,      2, 1, f); fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate,   4, 1, f); fwrite(&block_align, 2, 1, f);
    fwrite(&bits,        2, 1, f); fwrite("data",       1, 4, f);
    fwrite(&data_bytes,  4, 1, f); fwrite(pcm.data(),   2, n_samples, f);
    fclose(f);
    spdlog::info("WAV saved: {} ({} samples, {}Hz)", path, n_samples, sample_rate);
}

static std::vector<float> resample_sinc(const std::vector<float>& in,
                                        int in_rate,
                                        int out_rate) {
    if (in_rate == out_rate || in.empty())
        return in;

    const int gcd = std::gcd(in_rate, out_rate);
    const int orig_freq = in_rate / gcd;
    const int new_freq = out_rate / gcd;
    size_t out_len = static_cast<size_t>(
        std::ceil(static_cast<double>(in.size()) * new_freq / orig_freq));
    out_len = std::max<size_t>(out_len, 1);
    std::vector<float> out(out_len);

    constexpr double kPi = 3.14159265358979323846264338327950288;
    constexpr int kLowpassFilterWidth = 6;
    constexpr double kRolloff = 0.99;
    const double base_freq = std::min(orig_freq, new_freq) * kRolloff;
    const int width = static_cast<int>(
        std::ceil(kLowpassFilterWidth * static_cast<double>(orig_freq) / base_freq));
    const int kernel_width = width * 2 + orig_freq;
    const double scale = base_freq / orig_freq;

    std::vector<double> kernel(static_cast<size_t>(new_freq) * kernel_width);
    for (int phase = 0; phase < new_freq; ++phase) {
        for (int k = 0; k < kernel_width; ++k) {
            const int idx = k - width;
            double t = (static_cast<double>(idx) / orig_freq) -
                       (static_cast<double>(phase) / new_freq);
            t *= base_freq;
            t = std::max(-static_cast<double>(kLowpassFilterWidth),
                         std::min(static_cast<double>(kLowpassFilterWidth), t));
            const double window = std::pow(std::cos(t * kPi / kLowpassFilterWidth / 2.0), 2.0);
            const double pix = t * kPi;
            const double s = std::abs(pix) < 1e-8 ? 1.0 : std::sin(pix) / pix;
            kernel[static_cast<size_t>(phase) * kernel_width + k] = s * window * scale;
        }
    }

    for (size_t i = 0; i < out_len; ++i) {
        const int frame = static_cast<int>(i / new_freq);
        const int phase = static_cast<int>(i % new_freq);
        double sum = 0.0;
        const size_t kernel_offset = static_cast<size_t>(phase) * kernel_width;
        for (int k = 0; k < kernel_width; ++k) {
            const int src = frame * orig_freq + k - width;
            if (src < 0 || src >= static_cast<int>(in.size()))
                continue;
            sum += static_cast<double>(in[static_cast<size_t>(src)]) *
                   kernel[kernel_offset + k];
        }

        out[i] = static_cast<float>(sum);
    }
    return out;
}

static std::vector<float> read_wav_mono_f32(const std::string& path, int target_rate) {
    FILE* wf = fopen(path.c_str(), "rb");
    if (!wf)
        throw std::runtime_error("Cannot open WAV file: " + path);

    char riff[5] = {};
    if (fread(riff, 1, 4, wf) != 4 || std::string(riff) != "RIFF") {
        fclose(wf);
        throw std::runtime_error("Not a WAV file: " + path);
    }
    fseek(wf, 4, SEEK_CUR);
    if (fread(riff, 1, 4, wf) != 4 || std::string(riff) != "WAVE") {
        fclose(wf);
        throw std::runtime_error("Not a WAVE file: " + path);
    }

    uint16_t audio_fmt = 0, channels = 0, bits = 0, block_align = 0;
    uint32_t sample_rate = 0;
    std::vector<uint8_t> data;
    bool have_fmt = false;

    while (true) {
        char ck[5] = {};
        uint32_t ck_sz = 0;
        if (fread(ck, 1, 4, wf) != 4)
            break;
        if (fread(&ck_sz, 4, 1, wf) != 1)
            break;

        std::string name(ck, 4);
        if (name == "fmt ") {
            if (ck_sz < 16) {
                fclose(wf);
                throw std::runtime_error("Invalid WAV fmt chunk: " + path);
            }
            uint32_t byte_rate = 0;
            fread(&audio_fmt, 2, 1, wf);
            fread(&channels, 2, 1, wf);
            fread(&sample_rate, 4, 1, wf);
            fread(&byte_rate, 4, 1, wf);
            fread(&block_align, 2, 1, wf);
            fread(&bits, 2, 1, wf);
            if (ck_sz > 16)
                fseek(wf, ck_sz - 16, SEEK_CUR);
            have_fmt = true;
        } else if (name == "data") {
            data.resize(ck_sz);
            if (ck_sz && fread(data.data(), 1, ck_sz, wf) != ck_sz) {
                fclose(wf);
                throw std::runtime_error("Truncated WAV data: " + path);
            }
        } else {
            fseek(wf, ck_sz, SEEK_CUR);
        }

        if (ck_sz & 1)
            fseek(wf, 1, SEEK_CUR);
        if (have_fmt && !data.empty())
            break;
    }
    fclose(wf);

    if (!have_fmt || data.empty())
        throw std::runtime_error("WAV missing fmt or data chunk: " + path);
    bool is_pcm = (audio_fmt == 1) && (bits == 8 || bits == 16 || bits == 32);
    bool is_float = (audio_fmt == 3) && (bits == 32);
    if ((!is_pcm && !is_float) || channels == 0) {
        throw std::runtime_error(
            "Unsupported WAV format for native reference audio: " + path +
            " (need PCM s8/s16/s32 or IEEE float32, got fmt=" +
            std::to_string(audio_fmt) + " bits=" + std::to_string(bits) + ")");
    }
    int frames = static_cast<int>(data.size() / block_align);
    std::vector<float> mono(frames);

    if (is_float) {
        // IEEE float32
        const float* fpcm = reinterpret_cast<const float*>(data.data());
        for (int i = 0; i < frames; ++i) {
            float sum = 0.0f;
            for (int ch = 0; ch < channels; ++ch)
                sum += fpcm[i * channels + ch];
            mono[i] = sum / channels;
        }
    } else if (bits == 32) {
        // s32 PCM
        const int32_t* pcm = reinterpret_cast<const int32_t*>(data.data());
        for (int i = 0; i < frames; ++i) {
            float sum = 0.0f;
            for (int ch = 0; ch < channels; ++ch)
                sum += static_cast<float>(pcm[i * channels + ch]) / 2147483648.0f;
            mono[i] = sum / channels;
        }
    } else if (bits == 16) {
        // s16 PCM
        const int16_t* pcm = reinterpret_cast<const int16_t*>(data.data());
        for (int i = 0; i < frames; ++i) {
            float sum = 0.0f;
            for (int ch = 0; ch < channels; ++ch)
                sum += static_cast<float>(pcm[i * channels + ch]) / 32768.0f;
            mono[i] = sum / channels;
        }
    } else {
        // s8 PCM (unsigned)
        const uint8_t* pcm = reinterpret_cast<const uint8_t*>(data.data());
        for (int i = 0; i < frames; ++i) {
            float sum = 0.0f;
            for (int ch = 0; ch < channels; ++ch)
                sum += (static_cast<float>(pcm[i * channels + ch]) - 128.0f) / 128.0f;
            mono[i] = sum / channels;
        }
    }

    auto out = resample_sinc(mono, static_cast<int>(sample_rate), target_rate);
    spdlog::info("Read WAV: {} input={}Hz samples={} -> {}Hz samples={}",
                 path, sample_rate, mono.size(), target_rate, out.size());
    return out;
}

struct RefCodes {
    int num_codebooks = 0;
    int code_len = 0;
    std::vector<int32_t> codes;  // [num_codebooks, code_len], row-major
};

static RefCodes read_ref_codes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good())
        throw std::runtime_error("Cannot open ref codes file: " + path);

    int32_t hdr[2] = {0, 0};
    if (!f.read(reinterpret_cast<char*>(hdr), sizeof(hdr)))
        throw std::runtime_error("Ref codes file is too small: " + path);

    RefCodes out;
    out.num_codebooks = hdr[0];
    out.code_len = hdr[1];
    if (out.num_codebooks <= 0 || out.code_len <= 0)
        throw std::runtime_error("Invalid ref codes header: " + path);

    const size_t n = static_cast<size_t>(out.num_codebooks) * out.code_len;
    out.codes.resize(n);
    if (!f.read(reinterpret_cast<char*>(out.codes.data()), n * sizeof(int32_t)))
        throw std::runtime_error("Ref codes file is truncated: " + path);
    return out;
}

static void append_ids(std::vector<int32_t>& out, const std::vector<int>& ids) {
    for (int id : ids) out.push_back(static_cast<int32_t>(id));
}

static std::string build_reference_prompt_file(
    fish::Tokenizer& tokenizer,
    const fish::DualARConfig& cfg,
    const std::string& target_text,
    const std::string& ref_text,
    const std::string& ref_codes_path
) {
    RefCodes ref = read_ref_codes(ref_codes_path);
    if (ref.num_codebooks != cfg.num_codebooks) {
        throw std::runtime_error(
            "Ref codes num_codebooks does not match DualAR config: " +
            std::to_string(ref.num_codebooks) + " vs " + std::to_string(cfg.num_codebooks));
    }

    std::vector<int32_t> row0;
    append_ids(row0, tokenizer.encode_raw(
        "<|im_start|>system\n"
        "convert the provided text to speech reference to the following:\n\n"
        "Text:\n"));
    append_ids(row0, tokenizer.encode_raw(
        "<|speaker:0|>" + ref_text + "\n\nSpeech:\n"));

    const int vq_start = static_cast<int>(row0.size());
    for (int t = 0; t < ref.code_len; ++t)
        row0.push_back(ref.codes[t] + cfg.semantic_begin_id);

    append_ids(row0, tokenizer.encode_raw("<|im_end|>\n<|im_start|>user\n"));
    append_ids(row0, tokenizer.encode_raw(target_text));
    append_ids(row0, tokenizer.encode_raw("<|im_end|>\n<|im_start|>assistant\n<|voice|>"));

    const int cb_dim = ref.num_codebooks + 1;
    const int prompt_len = static_cast<int>(row0.size());
    std::vector<int32_t> prompt(static_cast<size_t>(cb_dim) * prompt_len, 0);
    std::copy(row0.begin(), row0.end(), prompt.begin());
    for (int cb = 0; cb < ref.num_codebooks; ++cb) {
        for (int t = 0; t < ref.code_len; ++t) {
            prompt[static_cast<size_t>(cb + 1) * prompt_len + vq_start + t] =
                ref.codes[static_cast<size_t>(cb) * ref.code_len + t];
        }
    }

    std::filesystem::create_directories("/tmp/fish-audio-cpp");
    std::string prompt_path = "/tmp/fish-audio-cpp/reference_prompt.bin";
    std::ofstream pf(prompt_path, std::ios::binary);
    if (!pf.good())
        throw std::runtime_error("Cannot write prompt file: " + prompt_path);
    int32_t hdr[2] = {ref.num_codebooks, prompt_len};
    pf.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    pf.write(reinterpret_cast<const char*>(prompt.data()), prompt.size() * sizeof(int32_t));
    if (!pf.good())
        throw std::runtime_error("Failed writing prompt file: " + prompt_path);

    spdlog::info("Built C++ reference prompt: {} ({} codebooks, {} tokens, ref frames={})",
                 prompt_path, ref.num_codebooks, prompt_len, ref.code_len);
    return prompt_path;
}

// Resolve model file path.  Directory layout:
//   model_dir/models/model-fp16/dual_ar.bin    (new: recommended)
//   model_dir/models/model-int8-w8a16/dac.bin
//   model_dir/dual_ar_fp16.bin                 (legacy: flat, still works)
// --dtype maps shorthand: int8 → int8-w8a16
// Auto-detect priority: int8-w8a16 > fp16 > bf16 > unsuffixed (legacy).
static std::string resolve_model_path(
    const std::string& model_dir,
    const std::string& basename,   // "dual_ar" or "dac"
    const std::string& dtype_override = "")
{
    auto suffix_for = [](const std::string& dt) -> std::string {
        if (dt == "int8") return "int8-w8a16";
        return dt;
    };

    auto try_resolve = [&](const std::vector<std::string>& candidates) -> std::string {
        for (const auto& p : candidates) {
            if (std::filesystem::exists(p)) {
                spdlog::info("Resolved model: {}", p);
                return p;
            }
        }
        return "";
    };

    auto dir = std::filesystem::path(model_dir);
    auto model_name = dir.filename().string();  // e.g. "s2-pro"
    // Search models/<model>-<dtype>/ at project root, then parent, then model_dir
    auto candidates = [&](const std::string& suffix) -> std::vector<std::string> {
        auto sub = "/models/" + model_name + "-" + suffix + "/" + basename + ".bin";
        return {
            dir.parent_path().parent_path().string() + sub,  // ../../models/
            dir.parent_path().string() + sub,                // ../models/
            model_dir + sub,                                  // models/ inside model_dir
            model_dir + "/" + basename + "_" + suffix + ".bin",  // legacy flat
        };
    };

    if (!dtype_override.empty()) {
        auto suf = suffix_for(dtype_override);
        auto path = try_resolve(candidates(suf));
        if (!path.empty()) return path;
        throw std::runtime_error(
            "Requested dtype '" + dtype_override + "', not found in " + model_dir);
    }

    static const char* suffixes[] = {"int8-w8a16", "fp16", "bf16", ""};
    for (const char* suf : suffixes) {
        auto path = try_resolve(candidates(suf));
        if (!path.empty()) return path;
    }
    return model_dir + "/" + basename + "_fp16.bin";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
static int run_main(int argc, char* argv[]) {
    spdlog::set_pattern("[%H:%M:%S] [%^%l%$] %v");

    cxxopts::Options opts("fish-server", "Fish Audio S2 Pro TTS (pure C++)");
    opts.add_options()
        ("m,model-dir",    "Model directory (dual_ar_<dtype>.bin, dac_<dtype>.bin, configs)",
         cxxopts::value<std::string>()->default_value("checkpoints/s2-pro"))
        ("t,text",         "Text to synthesize",
         cxxopts::value<std::string>()->default_value("你好"))
        ("max-tokens",     "Max new codebook frames",
         cxxopts::value<int>()->default_value("512"))
        ("temperature",    "Sampling temperature",
         cxxopts::value<float>()->default_value("0.7"))
        ("top-p",          "Top-p nucleus sampling",
         cxxopts::value<float>()->default_value("0.9"))
        ("top-k",          "Top-k sampling",
         cxxopts::value<int>()->default_value("50"))
        ("seed",           "Random seed",
         cxxopts::value<int>()->default_value("42"))
        ("o,output",       "Output WAV path",
         cxxopts::value<std::string>()->default_value("output/speech.wav"))
        ("p,prompt-file",  "Pre-built prompt file for voice cloning",
         cxxopts::value<std::string>()->default_value(""))
        ("ref-audio",      "Reference audio file (WAV) for one-shot voice cloning — encodes internally",
         cxxopts::value<std::string>()->default_value(""))
        ("ref-codes",      "Reference VQ codes binary ([num_codebooks,T]+codes) for C++ prompt building",
         cxxopts::value<std::string>()->default_value(""))
        ("ref-text",       "Transcript text (or path to .txt file) for reference audio",
         cxxopts::value<std::string>()->default_value(""))
        ("tokenize-only",   "Only tokenize --text and print token IDs",
         cxxopts::value<bool>()->default_value("false"))
        ("encode-audio",    "Encode audio file to VQ codes (no TTS generation)",
         cxxopts::value<std::string>()->default_value(""))
        ("output-codes",    "Output path for encoded VQ codes (use with --encode-audio)",
         cxxopts::value<std::string>()->default_value(""))
        ("decode-codes", "Decode VQ codes file to WAV (roundtrip test)", cxxopts::value<std::string>()->default_value(""))
        ("server",          "Run in HTTP server mode instead of CLI",
         cxxopts::value<bool>()->default_value("false"))
        ("host",            "Server bind address",
         cxxopts::value<std::string>()->default_value("0.0.0.0"))
        ("port",            "Server port",
         cxxopts::value<int>()->default_value("8080"))
        ("h,help",         "Print usage")
        ("dtype",          "Precision: fp16, int8 (→int8-w8a16). Default: auto-detect",
         cxxopts::value<std::string>()->default_value(""));

    auto args = opts.parse(argc, argv);
    if (args.count("help")) { spdlog::info("{}", opts.help()); return 0; }

    const std::string model_dir   = args["model-dir"].as<std::string>();
    const std::string text        = args["text"].as<std::string>();
    const int         max_tokens  = args["max-tokens"].as<int>();
    const float       temperature = args["temperature"].as<float>();
    const float       top_p       = args["top-p"].as<float>();
    const int         top_k       = args["top-k"].as<int>();
    const int         seed        = args["seed"].as<int>();
    const std::string output_wav  = args["output"].as<std::string>();
    std::string       prompt_file = args["prompt-file"].as<std::string>();
    const std::string ref_audio   = args["ref-audio"].as<std::string>();
    const std::string ref_codes   = args["ref-codes"].as<std::string>();
    std::string       ref_text    = args["ref-text"].as<std::string>();
    const bool        tokenize_only = args["tokenize-only"].as<bool>();
    const std::string encode_audio  = args["encode-audio"].as<std::string>();
    const std::string output_codes  = args["output-codes"].as<std::string>();
    const std::string decode_codes = args["decode-codes"].as<std::string>();
    const bool        server_mode  = args["server"].as<bool>();
    const std::string server_host  = args["host"].as<std::string>();
    const int         server_port  = args["port"].as<int>();
    const std::string dtype_override = args["dtype"].as<std::string>();

    spdlog::info("=== Fish Audio S2 Pro TTS ===");
    spdlog::info("  model-dir : {}", model_dir);
    spdlog::info("  text      : '{}'", text);
    spdlog::info("  output    : {}", output_wav);
    if (!dtype_override.empty())
        spdlog::info("  dtype     : {}", dtype_override);

    if (!ref_codes.empty() && !prompt_file.empty())
        throw std::runtime_error("Use either --prompt-file or --ref-codes, not both");
    if (!ref_codes.empty() && ref_text.empty())
        throw std::runtime_error("--ref-text is required when --ref-codes is provided");

    const auto output_parent = std::filesystem::path(output_wav).parent_path();
    if (!output_parent.empty())
        std::filesystem::create_directories(output_parent);

    if (tokenize_only) {
        auto tokenizer = std::make_unique<fish::Tokenizer>();
        if (!tokenizer->load(model_dir))
            throw std::runtime_error("Failed to load tokenizer");
        auto ids = tokenizer->encode(text);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) std::printf(" ");
            std::printf("%d", ids[i]);
        }
        std::printf("\n");
        return 0;
    }

    // -----------------------------------------------------------------------
    // CUDA / cuBLAS / cuDNN
    // -----------------------------------------------------------------------
    cudaStream_t   stream;
    cublasHandle_t cublas;
    cudnnHandle_t  cudnn;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetStream(cublas, stream));
    CUDNN_CHECK(cudnnCreate(&cudnn));
    CUDNN_CHECK(cudnnSetStream(cudnn, stream));

    // -----------------------------------------------------------------------
    // DualAR model
    // -----------------------------------------------------------------------
    spdlog::info("Loading DualAR config …");
    auto dual_ar_cfg = fish::DualARConfig::from_json(model_dir + "/dual_ar_config.json");
    spdlog::info("  text: layers={} dim={} n_head={} n_kv_head={} vocab={}",
                 dual_ar_cfg.n_layer, dual_ar_cfg.dim,
                 dual_ar_cfg.n_head, dual_ar_cfg.n_local_heads,
                 dual_ar_cfg.vocab_size);
    spdlog::info("  fast: layers={} codebook_size={} num_codebooks={}",
                 dual_ar_cfg.n_fast_layer,
                 dual_ar_cfg.codebook_size, dual_ar_cfg.num_codebooks);

    fish::ModelLoader dual_ar_loader;
    std::string dual_ar_path = resolve_model_path(model_dir, "dual_ar", dtype_override);
    if (!dual_ar_loader.load(dual_ar_path))
        throw std::runtime_error("Failed to load " + dual_ar_path);
    // pin_memory() registers 8 GB with CUDA which is very slow in WSL — skip it.

    auto dual_ar = std::make_unique<fish::DualAREngine>(
        dual_ar_cfg, std::move(dual_ar_loader), cublas, stream);
    spdlog::info("Uploading DualAR weights to GPU …");
    dual_ar->init();

    // -----------------------------------------------------------------------
    // DAC model
    // -----------------------------------------------------------------------
    spdlog::info("Loading DAC config …");
    auto dac_cfg = fish::DACConfig::from_json(model_dir + "/dac_config.json");

    fish::ModelLoader dac_loader;
    std::string dac_path = resolve_model_path(model_dir, "dac", dtype_override);
    if (!dac_loader.load(dac_path))
        throw std::runtime_error("Failed to load " + dac_path);

    auto dac = std::make_unique<fish::DACEngine>(
        dac_cfg, std::move(dac_loader), cublas, cudnn, stream);
    dac->init();

    // ---- Encode-only path (audio → VQ codes) ----
    if (!encode_audio.empty()) {
        if (output_codes.empty())
            throw std::runtime_error("--output-codes is required with --encode-audio");

        std::vector<float> audio_f32 = read_wav_mono_f32(encode_audio, dac_cfg.sample_rate);
        int num_samples = static_cast<int>(audio_f32.size());

        spdlog::info("Read audio: {} samples ({:.2f}s)", num_samples,
                     static_cast<double>(num_samples) / dac_cfg.sample_rate);

        int max_code_len = (num_samples / (dac_cfg.hop_length() * 4)) + 16;
        int n_cb = fish::DAC_TOTAL_CODEBOOKS;
        std::vector<int32_t> codes(n_cb * max_code_len);
        int code_len = 0;
        dac->encode(audio_f32.data(), 1, num_samples, codes.data(), &code_len);

        std::ofstream cf(output_codes, std::ios::binary);
        int32_t hdr[2] = {n_cb, code_len};
        cf.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
        cf.write(reinterpret_cast<const char*>(codes.data()),
                 static_cast<size_t>(n_cb) * code_len * sizeof(int32_t));
        spdlog::info("Wrote codes: {} ({} codebooks × {} frames)",
                     output_codes, n_cb, code_len);

        cudnnDestroy(cudnn);
        cublasDestroy(cublas);
        cudaStreamDestroy(stream);
        return 0;
    }

    // ---- Decode-only path (VQ codes → WAV) ----
    if (!decode_codes.empty()) {
        std::ifstream cf(decode_codes, std::ios::binary);
        if (!cf.good())
            throw std::runtime_error("Cannot open codes file: " + decode_codes);
        int32_t hdr[2] = {0, 0};
        if (!cf.read(reinterpret_cast<char*>(hdr), sizeof(hdr)))
            throw std::runtime_error("Codes file too small: " + decode_codes);
        int n_cb = hdr[0], code_len = hdr[1];
        if (n_cb <= 0 || code_len <= 0)
            throw std::runtime_error("Invalid codes header: " + decode_codes);
        std::vector<int32_t> codes(static_cast<size_t>(n_cb) * code_len);
        if (!cf.read(reinterpret_cast<char*>(codes.data()),
                     static_cast<size_t>(n_cb) * code_len * sizeof(int32_t)))
            throw std::runtime_error("Codes file truncated: " + decode_codes);

        spdlog::info("Decoding {} codes: {} codebooks x {} frames", decode_codes, n_cb, code_len);

        int32_t* d_codes;
        float* d_audio;
        size_t cb_bytes = static_cast<size_t>(n_cb) * code_len * sizeof(int32_t);
        CUDA_CHECK(cudaMalloc(&d_codes, cb_bytes));
        CUDA_CHECK(cudaMemcpy(d_codes, codes.data(), cb_bytes, cudaMemcpyHostToDevice));
        int max_audio = code_len * 2048 * 2;
        CUDA_CHECK(cudaMalloc(&d_audio, max_audio * sizeof(float)));

        int audio_len = 0;
        dac->decode(d_codes, 1, code_len, d_audio, &audio_len, max_audio);

        std::vector<float> h_audio(audio_len);
        CUDA_CHECK(cudaMemcpy(h_audio.data(), d_audio, audio_len * sizeof(float), cudaMemcpyDeviceToHost));
        write_wav(output_wav, h_audio.data(), audio_len, dac_cfg.sample_rate);

        CUDA_CHECK(cudaFree(d_codes));
        CUDA_CHECK(cudaFree(d_audio));
        cudnnDestroy(cudnn);
        cublasDestroy(cublas);
        cudaStreamDestroy(stream);
        return 0;
    }

    // -----------------------------------------------------------------------
    // Tokenizer
    // -----------------------------------------------------------------------
    spdlog::info("Loading tokenizer …");
    auto tokenizer = std::make_unique<fish::Tokenizer>();
    if (!tokenizer->load(model_dir))
        throw std::runtime_error("Failed to load tokenizer");

    if (!ref_audio.empty() && !ref_text.empty()) {
        // Auto-encode reference audio and build prompt.
        std::vector<float> audio_f32 = read_wav_mono_f32(ref_audio, dac_cfg.sample_rate);
        int n_samples = static_cast<int>(audio_f32.size());

        // 2. Read ref text (as file or literal)
        std::string resolved_ref_text = ref_text;
        std::ifstream tf(ref_text);
        if (tf.good()) {
            std::stringstream ss; ss << tf.rdbuf();
            resolved_ref_text = ss.str();
            // Trim trailing newline
            while (!resolved_ref_text.empty() && (resolved_ref_text.back() == '\n' || resolved_ref_text.back() == '\r'))
                resolved_ref_text.pop_back();
        }

        // 3. Encode
        spdlog::info("Encoding reference audio: {} samples", n_samples);
        int max_cb = (n_samples / (dac_cfg.hop_length() * 4)) + 16;
        std::vector<int32_t> ref_cb(fish::DAC_TOTAL_CODEBOOKS * max_cb);
        int cb_len = 0;
        dac->encode(audio_f32.data(), 1, n_samples, ref_cb.data(), &cb_len);

        // 4. Write to temp file and build prompt
        std::filesystem::create_directories("/tmp/fish-audio-cpp");
        std::string tmp_codes = "/tmp/fish-audio-cpp/ref_auto_codes.bin";
        std::ofstream cf(tmp_codes, std::ios::binary);
        int32_t hdr2[2] = {fish::DAC_TOTAL_CODEBOOKS, cb_len};
        cf.write(reinterpret_cast<const char*>(hdr2), sizeof(hdr2));
        cf.write(reinterpret_cast<const char*>(ref_cb.data()),
                 static_cast<size_t>(fish::DAC_TOTAL_CODEBOOKS) * cb_len * sizeof(int32_t));
        cf.close();

        prompt_file = build_reference_prompt_file(*tokenizer, dual_ar_cfg, text, resolved_ref_text, tmp_codes);
    }

    if (!ref_codes.empty())
        prompt_file = build_reference_prompt_file(*tokenizer, dual_ar_cfg, text, ref_text, ref_codes);

    int prompt_context_len = 256;
    if (!prompt_file.empty()) {
        std::ifstream pf(prompt_file, std::ios::binary);
        int32_t hdr[2] = {0, 0};
        if (pf.read(reinterpret_cast<char*>(hdr), sizeof(hdr))) {
            prompt_context_len = std::max(prompt_context_len, hdr[1]);
            spdlog::info("  prompt    : {} ({} tokens)", prompt_file, hdr[1]);
        } else {
            spdlog::warn("Cannot read prompt header now; pipeline will report a detailed error later");
        }
    }

    // -----------------------------------------------------------------------
    // Block manager (paged KV cache)
    // -----------------------------------------------------------------------
    fish::BlockManager::Config bm_cfg;
    bm_cfg.block_size = 16;
    bm_cfg.n_layers   = dual_ar_cfg.n_layer;
    bm_cfg.n_heads    = dual_ar_cfg.n_local_heads;
    bm_cfg.head_dim   = dual_ar_cfg.head_dim;
    bm_cfg.max_blocks = (max_tokens + prompt_context_len + 15) / 16 + 16;

    auto block_mgr = std::make_unique<fish::BlockManager>(bm_cfg, stream);
    block_mgr->init_gpu_memory();

    // -----------------------------------------------------------------------
    // Run inference
    // -----------------------------------------------------------------------
    fish::InferencePipeline pipeline(
        std::move(dual_ar),
        std::move(dac),
        std::move(block_mgr),
        std::move(tokenizer));

    if (server_mode) {
        fish::HttpServer::Config srv_cfg;
        srv_cfg.host = server_host;
        srv_cfg.port = server_port;
        fish::HttpServer srv(srv_cfg, &pipeline);
        spdlog::info("Starting HTTP server on {}:{}", server_host, server_port);
        srv.start();  // blocking
        // server stopped -- cleanup below
    } else {
        fish::TTSOutput result;
        if (!prompt_file.empty()) {
            spdlog::info("Using pre-built prompt file: {}", prompt_file);
            result = pipeline.run_with_prompt_file(prompt_file, max_tokens, temperature, top_p, top_k, seed);
        } else {
            result = pipeline.run(text, max_tokens, temperature, top_p, top_k, seed);
        }

        if (result.audio_samples.empty()) {
            spdlog::error("No audio generated");
            cudnnDestroy(cudnn);
            cublasDestroy(cublas);
            cudaStreamDestroy(stream);
            return 1;
        }

        write_wav(output_wav, result.audio_samples.data(),
                  static_cast<int>(result.audio_samples.size()), result.sample_rate);
    }

    cudnnDestroy(cudnn);
    cublasDestroy(cublas);
    cudaStreamDestroy(stream);
    spdlog::info("=== done ===");
    return 0;
}

int main(int argc, char* argv[]) {
    try {
        return run_main(argc, argv);
    } catch (const std::exception& e) {
        spdlog::critical("{}", e.what());
        return 1;
    }
}
