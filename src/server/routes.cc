// src/server/routes.cc — shared route utilities for the HTTP TTS server
#include "server/routes.h"

namespace fish::routes {

// Build a JSON error response body
std::string error_json(const std::string& msg) {
    nlohmann::json j;
    j["error"] = msg;
    return j.dump();
}

std::string sse_event(const std::string& type, const nlohmann::json& data) {
    nlohmann::json j = data;
    j["type"] = type;
    return "data: " + j.dump() + "\n\n";
}

std::string base64_decode(const std::string& encoded) {
    static const unsigned char kDecode[256] = {
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,255,255, 62,255,255,255, 63,
         52, 53, 54, 55, 56, 57, 58, 59, 60, 61,255,255,255,  0,255,255,
        255,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,255,255,255,255,255,
        255, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
         41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    };
    std::string out;
    out.reserve((encoded.size() / 4) * 3);
    int val = 0, valb = -8;
    for (unsigned char c : encoded) {
        if (c == '=' || kDecode[c] == 255) continue;
        val = (val << 6) | kDecode[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

}  // namespace fish::routes
