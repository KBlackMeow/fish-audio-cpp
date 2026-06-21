#pragma once
#include <string>
#include <nlohmann/json_fwd.hpp>

namespace fish {

struct DualARConfig {
    // === Text model (Qwen3-style) ===
    int vocab_size = 155776;
    int n_layer = 36;
    int n_head = 32;
    int dim = 2560;
    int intermediate_size = 9728;
    int n_local_heads = 8;     // KV heads for GQA
    int head_dim = 128;
    float rope_base = 1000000.0f;
    float norm_eps = 1e-6f;
    int max_seq_len = 32768;
    bool attention_qk_norm = true;

    // === Audio decoder (fast codebook predictor) ===
    int n_fast_layer = 4;
    int fast_dim = 2560;
    int fast_n_head = 32;
    int fast_n_local_heads = 8;
    int fast_head_dim = 128;
    int fast_intermediate_size = 9728;
    bool fast_attention_qk_norm = false;
    int codebook_size = 4096;
    int num_codebooks = 10;

    // === Special token IDs ===
    int semantic_begin_id = 151678;
    int semantic_end_id = 155773;
    int im_end_token_id = 151645;   // eos_token_id
    int audio_pad_token_id = 151677;
    int pad_token_id = 151669;

    // === Derived ===
    int head_size() const { return head_dim * n_head; }
    int kv_head_size() const { return head_dim * n_local_heads; }
    int fast_head_size() const { return fast_head_dim * fast_n_head; }
    int fast_kv_head_size() const { return fast_head_dim * fast_n_local_heads; }
    int codebook_stride() const { return codebook_size; }

    static DualARConfig from_json(const std::string& path);
    static DualARConfig from_json(const nlohmann::json& j);
};

}  // namespace fish
