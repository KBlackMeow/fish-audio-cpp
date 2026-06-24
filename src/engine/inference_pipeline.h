// src/engine/inference_pipeline.h
#pragma once
#include "engine/dual_ar_engine.h"
#include "engine/dac_engine.h"
#include "engine/block_manager.h"
#include "tokenizer/tokenizer.h"
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace fish {

struct TTSProfiling {
    double total_ms = 0.0;
    double tokenize_ms = 0.0;
    double prefill_ms = 0.0;
    double ar_decode_ms = 0.0;
    double decode_embed_ms = 0.0;
    double decode_step_ms = 0.0;
    double decode_logits_ms = 0.0;
    double semantic_sample_ms = 0.0;
    double codebook_decode_ms = 0.0;
    double codebook_sample_ms = 0.0;
    double seq_update_ms = 0.0;
    double dac_decode_ms = 0.0;
    double audio_copy_ms = 0.0;
    double first_audio_ms = 0.0;
    double ref_decode_ms = 0.0;
    double ref_encode_ms = 0.0;
    double prompt_build_ms = 0.0;
    double response_encode_ms = 0.0;
    int prompt_tokens = 0;
    int generated_frames = 0;
    int ref_code_frames = 0;
    int streamed_dac_calls = 0;
    int streamed_audio_chunks = 0;
    bool ref_cache_hit = false;
};

struct TTSOutput {
    std::vector<float> audio_samples;  // PCM float32, mono
    std::vector<int32_t> generated_codes;  // [num_codebooks, generated_frames]
    int sample_rate = 0;
    TTSProfiling profiling;
};

// Callbacks for streaming inference.
// All callbacks are called from the inference thread.
struct StreamCallback {
    // Called once per AR decode step. current is 0-indexed step number.
    std::function<void(int current, int total)> on_progress;
    // Called for each audio chunk after DAC decode completes.
    // Chunks are ~50ms (2205 samples at 44.1kHz).
    std::function<void(const float* samples, int count)> on_audio_chunk;
};

class InferencePipeline {
public:
    InferencePipeline(
        std::unique_ptr<DualAREngine> dual_ar,
        std::unique_ptr<DACEngine> dac,
        std::unique_ptr<BlockManager> block_mgr,
        std::unique_ptr<Tokenizer> tokenizer
    );

    ~InferencePipeline();

    // Non-copyable, movable
    InferencePipeline(const InferencePipeline&) = delete;
    InferencePipeline& operator=(const InferencePipeline&) = delete;
    InferencePipeline(InferencePipeline&&) = default;
    InferencePipeline& operator=(InferencePipeline&&) = default;

    // Run TTS inference for a single text prompt (no reference audio)
    TTSOutput run(
        const std::string& text,
        int max_new_tokens = 512,
        float temperature = 0.7f,
        float top_p = 0.9f,
        int top_k = 30,
        int seed = 42
    );

    // Run TTS inference with a pre-built prompt file (supports reference audio).
    // Prompt file format:
    //   int32: num_codebooks
    //   int32: prompt_len (T)
    //   int32[(num_codebooks+1) * T]: prompt tensor (row-major, row 0 = text tokens)
    TTSOutput run_with_prompt_file(
        const std::string& prompt_path,
        int max_new_tokens = 512,
        float temperature = 0.7f,
        float top_p = 0.9f,
        int top_k = 30,
        int seed = 42
    );

    // Streaming variant of run(). Reports progress during AR generation
    // and streams audio chunks after DAC decode. Returns full TTSOutput.
    TTSOutput run_streaming(
        const std::string& text,
        int max_new_tokens,
        float temperature, float top_p, int top_k, int seed,
        StreamCallback callback
    );

    // Run TTS with reference audio for voice cloning.
    // ref_audio: float32 mono PCM at model sample_rate (44.1kHz)
    // ref_text: transcript of the reference audio
    TTSOutput run_with_ref_audio(
        const float* ref_audio, int ref_num_samples,
        const std::string& ref_text,
        const std::string& target_text,
        int max_new_tokens, float temperature, float top_p, int top_k, int seed,
        int chunk_length = 0,
        int history_frames = 96
    );

    // Streaming variant of run_with_ref_audio().
    TTSOutput run_with_ref_audio_streaming(
        const float* ref_audio, int ref_num_samples,
        const std::string& ref_text,
        const std::string& target_text,
        int max_new_tokens, float temperature, float top_p, int top_k, int seed,
        int chunk_length,
        int history_frames,
        StreamCallback callback
    );

    // Expose the pipeline's output sample rate (from DAC config)
    int sample_rate() const;

private:
    std::vector<int32_t> build_ref_prompt(
        const int32_t* codes, int num_codebooks, int code_len,
        const std::string& ref_text, const std::string& target_text,
        int* prompt_len_out = nullptr
    );
    TTSOutput run_with_prompt_tensor(
        const std::vector<int32_t>& prompt,
        int num_codebooks,
        int prompt_len,
        int max_new_tokens,
        float temperature,
        float top_p,
        int top_k,
        int seed,
        StreamCallback callback = {}
    );
    struct RefCodeCacheEntry {
        std::vector<int32_t> codes;
        int code_len = 0;
    };
    std::unique_ptr<DualAREngine> dual_ar_;
    std::unique_ptr<DACEngine> dac_;
    std::unique_ptr<BlockManager> block_mgr_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::mutex ref_code_cache_mutex_;
    std::unordered_map<std::size_t, RefCodeCacheEntry> ref_code_cache_;
};

}  // namespace fish
