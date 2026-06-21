// src/server/routes.cc — shared route utilities for the HTTP TTS server
#include "server/routes.h"
#include <cstdint>

namespace fish::routes {

// Build a JSON error response body
std::string error_json(const std::string& msg) {
    nlohmann::json j;
    j["error"] = msg;
    return j.dump();
}

// Encode a float32 PCM buffer as base64 for JSON transport
std::string encode_pcm_base64(const float* samples, int n) {
    // Simple inline base64 encoding
    static const char kBase64[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    auto* bytes = reinterpret_cast<const uint8_t*>(samples);
    int nbytes = n * static_cast<int>(sizeof(float));
    std::string out;
    out.reserve(((nbytes + 2) / 3) * 4);
    for (int i = 0; i < nbytes; i += 3) {
        uint32_t val = static_cast<uint32_t>(bytes[i]) << 16;
        if (i + 1 < nbytes) val |= static_cast<uint32_t>(bytes[i + 1]) << 8;
        if (i + 2 < nbytes) val |= static_cast<uint32_t>(bytes[i + 2]);
        out.push_back(kBase64[(val >> 18) & 0x3F]);
        out.push_back(kBase64[(val >> 12) & 0x3F]);
        out.push_back((i + 1 < nbytes) ? kBase64[(val >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < nbytes) ? kBase64[val & 0x3F] : '=');
    }
    return out;
}

std::string sse_event(const std::string& type, const nlohmann::json& data) {
    nlohmann::json j = data;
    j["type"] = type;
    return "data: " + j.dump() + "\n\n";
}

}  // namespace fish::routes
