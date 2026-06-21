// src/server/routes.h
#pragma once
#include <nlohmann/json.hpp>
#include <string>

namespace fish::routes {

std::string error_json(const std::string& msg);
std::string sse_event(const std::string& type, const nlohmann::json& data);
std::string base64_decode(const std::string& encoded);

}  // namespace fish::routes
