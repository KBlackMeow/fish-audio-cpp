// src/model/dual_ar_config.cc
#include "model/dual_ar_config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <spdlog/spdlog.h>

namespace fish {

DualARConfig DualARConfig::from_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Failed to open config: " + path);
    nlohmann::json j;
    f >> j;
    return from_json(j);
}

static void parse_transformer_config(const nlohmann::json& j,
                                     int& dim, int& n_layer, int& n_head,
                                     int& n_local_heads, int& head_dim,
                                     int& intermediate_size,
                                     float& rope_base, float& norm_eps,
                                     bool& qk_norm, int& vocab_size,
                                     int& max_seq_len) {
    if (j.contains("dim")) dim = j["dim"].get<int>();
    if (j.contains("n_layer")) n_layer = j["n_layer"].get<int>();
    if (j.contains("n_head")) n_head = j["n_head"].get<int>();
    if (j.contains("n_local_heads")) n_local_heads = j["n_local_heads"].get<int>();
    if (j.contains("head_dim")) head_dim = j["head_dim"].get<int>();
    if (j.contains("intermediate_size")) intermediate_size = j["intermediate_size"].get<int>();
    if (j.contains("rope_base")) rope_base = j["rope_base"].get<float>();
    if (j.contains("norm_eps")) norm_eps = j["norm_eps"].get<float>();
    if (j.contains("attention_qk_norm")) qk_norm = j["attention_qk_norm"].get<bool>();
    if (j.contains("vocab_size")) vocab_size = j["vocab_size"].get<int>();
    if (j.contains("max_seq_len")) max_seq_len = j["max_seq_len"].get<int>();
}

DualARConfig DualARConfig::from_json(const nlohmann::json& j) {
    DualARConfig cfg;

    // Support both nested and flat JSON structures.
    // Nested: {"text_config": {...}, "audio_decoder_config": {...}, ...}
    // Flat:   {"n_layer": 12, "dim": 1024, ...}

    bool has_text_config = j.contains("text_config");
    bool has_audio_config = j.contains("audio_decoder_config");

    if (has_text_config || has_audio_config) {
        // Nested structure
        if (has_text_config) {
            const auto& tc = j["text_config"];
            parse_transformer_config(tc,
                cfg.dim, cfg.n_layer, cfg.n_head,
                cfg.n_local_heads, cfg.head_dim,
                cfg.intermediate_size, cfg.rope_base, cfg.norm_eps,
                cfg.attention_qk_norm, cfg.vocab_size, cfg.max_seq_len);
        }
        if (has_audio_config) {
            const auto& ac = j["audio_decoder_config"];
            int vocab = cfg.codebook_size;
            int max_seq = 0;
            parse_transformer_config(ac,
                cfg.fast_dim, cfg.n_fast_layer, cfg.fast_n_head,
                cfg.fast_n_local_heads, cfg.fast_head_dim,
                cfg.fast_intermediate_size, cfg.rope_base, cfg.norm_eps,
                cfg.fast_attention_qk_norm, vocab, max_seq);
            if (vocab > 0) cfg.codebook_size = vocab;
            if (ac.contains("num_codebooks")) cfg.num_codebooks = ac["num_codebooks"].get<int>();
        }
    } else {
        // Flat structure — apply directly
        parse_transformer_config(j,
            cfg.dim, cfg.n_layer, cfg.n_head,
            cfg.n_local_heads, cfg.head_dim,
            cfg.intermediate_size, cfg.rope_base, cfg.norm_eps,
            cfg.attention_qk_norm, cfg.vocab_size, cfg.max_seq_len);
        // Fast decoder from flat keys
        if (j.contains("fast_dim")) cfg.fast_dim = j["fast_dim"].get<int>();
        if (j.contains("n_fast_layer")) cfg.n_fast_layer = j["n_fast_layer"].get<int>();
        if (j.contains("fast_n_head")) cfg.fast_n_head = j["fast_n_head"].get<int>();
        if (j.contains("fast_n_local_heads")) cfg.fast_n_local_heads = j["fast_n_local_heads"].get<int>();
        if (j.contains("fast_head_dim")) cfg.fast_head_dim = j["fast_head_dim"].get<int>();
        if (j.contains("fast_intermediate_size")) cfg.fast_intermediate_size = j["fast_intermediate_size"].get<int>();
        if (j.contains("fast_attention_qk_norm")) cfg.fast_attention_qk_norm = j["fast_attention_qk_norm"].get<bool>();
        if (j.contains("codebook_size")) cfg.codebook_size = j["codebook_size"].get<int>();
        if (j.contains("num_codebooks")) cfg.num_codebooks = j["num_codebooks"].get<int>();
    }

    if (j.contains("semantic_start_token_id"))
        cfg.semantic_begin_id = j["semantic_start_token_id"].get<int>();
    if (j.contains("semantic_end_token_id"))
        cfg.semantic_end_id = j["semantic_end_token_id"].get<int>();
    if (j.contains("eos_token_id"))
        cfg.im_end_token_id = j["eos_token_id"].get<int>();
    if (j.contains("audio_pad_token_id"))
        cfg.audio_pad_token_id = j["audio_pad_token_id"].get<int>();
    if (j.contains("pad_token_id"))
        cfg.pad_token_id = j["pad_token_id"].get<int>();

    spdlog::info("DualARConfig: dim={} n_layer={} n_head={} n_kv={} head_dim={} qk_norm={}",
                 cfg.dim, cfg.n_layer, cfg.n_head, cfg.n_local_heads,
                 cfg.head_dim, cfg.attention_qk_norm);
    spdlog::info("  fast: dim={} n_layer={} n_head={} n_kv={} codebook={}x{}",
                 cfg.fast_dim, cfg.n_fast_layer, cfg.fast_n_head,
                 cfg.fast_n_local_heads, cfg.codebook_size, cfg.num_codebooks);
    spdlog::info("  special: semantic=[{},{}] eos={} vocab={}",
                 cfg.semantic_begin_id, cfg.semantic_end_id,
                 cfg.im_end_token_id, cfg.vocab_size);
    return cfg;
}

}  // namespace fish
