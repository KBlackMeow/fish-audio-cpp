// src/engine/inference_pipeline.h
#pragma once
#include "engine/dual_ar_engine.h"
#include "engine/dac_engine.h"
#include "engine/block_manager.h"
#include "tokenizer/tokenizer.h"
#include <memory>
#include <vector>

namespace fish {

struct TTSOutput {
    std::vector<float> audio_samples;  // PCM float32, mono
    int sample_rate = 0;
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

private:
    std::unique_ptr<DualAREngine> dual_ar_;
    std::unique_ptr<DACEngine> dac_;
    std::unique_ptr<BlockManager> block_mgr_;
    std::unique_ptr<Tokenizer> tokenizer_;
};

}  // namespace fish
