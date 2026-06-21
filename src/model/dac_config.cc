#include "model/dac_config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <spdlog/spdlog.h>

namespace fish {

DACConfig DACConfig::from_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Failed to open config: " + path);
    nlohmann::json j;
    f >> j;
    return from_json(j);
}

DACConfig DACConfig::from_json(const nlohmann::json& j) {
    DACConfig cfg;
    if (j.contains("sample_rate")) cfg.sample_rate = j["sample_rate"].get<int>();
    if (j.contains("block_size")) cfg.block_size = j["block_size"].get<int>();
    if (j.contains("n_layer")) cfg.n_layer = j["n_layer"].get<int>();
    if (j.contains("n_head")) cfg.n_head = j["n_head"].get<int>();
    if (j.contains("dim")) cfg.dim = j["dim"].get<int>();
    if (j.contains("intermediate_size"))
        cfg.intermediate_size = j["intermediate_size"].get<int>();
    if (j.contains("n_local_heads")) cfg.n_local_heads = j["n_local_heads"].get<int>();
    if (j.contains("head_dim")) cfg.head_dim = j["head_dim"].get<int>();
    if (j.contains("rope_base")) cfg.rope_base = j["rope_base"].get<float>();
    if (j.contains("norm_eps")) cfg.norm_eps = j["norm_eps"].get<float>();
    if (j.contains("codebook_size")) cfg.codebook_size = j["codebook_size"].get<int>();
    if (j.contains("num_codebooks")) cfg.num_codebooks = j["num_codebooks"].get<int>();
    if (j.contains("latent_dim")) cfg.latent_dim = j["latent_dim"].get<int>();
    if (j.contains("channels_first")) cfg.channels_first = j["channels_first"].get<bool>();
    if (j.contains("pos_embed_type")) cfg.pos_embed_type = j["pos_embed_type"].get<std::string>();
    if (j.contains("encoder_rates"))
        cfg.encoder_rates = j["encoder_rates"].get<std::vector<int>>();
    if (j.contains("decoder_rates"))
        cfg.decoder_rates = j["decoder_rates"].get<std::vector<int>>();

    spdlog::info("DACConfig: dim={} n_layer={} hop={} pos={}",
                 cfg.dim, cfg.n_layer, cfg.hop_length(), cfg.pos_embed_type);
    return cfg;
}

}  // namespace fish
