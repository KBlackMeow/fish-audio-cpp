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

}  // namespace fish::routes
