// src/server/routes.h
#pragma once
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace fish::routes {

struct DecodedAudio {
    std::vector<float> samples;
    int sample_rate = 0;
    std::string format;
};

std::string error_json(const std::string& msg);
std::string sse_event(const std::string& type, const nlohmann::json& data);
std::string base64_decode(const std::string& encoded);
std::string encode_pcm_f32_base64(const float* samples, int n_samples);
std::string encode_wav_base64(const float* samples, int n_samples, int sample_rate);
DecodedAudio decode_audio_base64(const std::string& encoded,
                                  int target_rate,
                                  int default_pcm_rate = 44100,
                                  const std::string& format_hint = "auto");

}  // namespace fish::routes
