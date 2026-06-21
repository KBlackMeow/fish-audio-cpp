#pragma once
#include <string>
#include <vector>
#include <nlohmann/json_fwd.hpp>

namespace fish {

struct DACConfig {
    int sample_rate = 44100;
    int block_size = 2048;
    int n_layer = 8;
    int n_head = 8;
    int dim = 512;
    int intermediate_size = 1536;
    int n_local_heads = 8;
    int head_dim = 64;
    float rope_base = 10000.0f;
    float norm_eps = 1e-5f;
    int codebook_size = 1024;
    int num_codebooks = 10;
    int latent_dim = 1024;
    bool channels_first = true;
    std::string pos_embed_type = "rope";  // "rope" or "conformer"

    std::vector<int> encoder_rates = {2, 4, 8, 8};
    std::vector<int> decoder_rates = {8, 8, 4, 2};

    static DACConfig from_json(const std::string& path);
    static DACConfig from_json(const nlohmann::json& j);

    int hop_length() const {
        int hop = 1;
        for (int r : encoder_rates) hop *= r;
        return hop;  // typically 2*4*8*8 = 512
    }
};

}  // namespace fish
