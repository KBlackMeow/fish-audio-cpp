// src/server/routes.cc — shared route utilities for the HTTP TTS server
#include "server/routes.h"
#include <httplib.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace fish::routes {

namespace {

uint16_t read_le16(const std::string& s, size_t off) {
    if (off + 2 > s.size()) throw std::runtime_error("Truncated WAV header");
    return static_cast<uint16_t>(static_cast<unsigned char>(s[off]) |
           (static_cast<unsigned char>(s[off + 1]) << 8));
}

uint32_t read_le32(const std::string& s, size_t off) {
    if (off + 4 > s.size()) throw std::runtime_error("Truncated WAV header");
    return static_cast<uint32_t>(static_cast<unsigned char>(s[off]) |
           (static_cast<unsigned char>(s[off + 1]) << 8) |
           (static_cast<unsigned char>(s[off + 2]) << 16) |
           (static_cast<unsigned char>(s[off + 3]) << 24));
}

void append_le16(std::string& out, uint16_t v) {
    out.push_back(static_cast<char>(v & 0xff));
    out.push_back(static_cast<char>((v >> 8) & 0xff));
}

void append_le32(std::string& out, uint32_t v) {
    out.push_back(static_cast<char>(v & 0xff));
    out.push_back(static_cast<char>((v >> 8) & 0xff));
    out.push_back(static_cast<char>((v >> 16) & 0xff));
    out.push_back(static_cast<char>((v >> 24) & 0xff));
}

bool is_wav_bytes(const std::string& bytes) {
    return bytes.size() >= 12 &&
           bytes.compare(0, 4, "RIFF") == 0 &&
           bytes.compare(8, 4, "WAVE") == 0;
}

float sanitize_sample(float v) {
    if (!std::isfinite(v)) return 0.0f;
    return std::max(-1.0f, std::min(1.0f, v));
}

std::vector<float> resample_linear(const std::vector<float>& in, int in_rate, int out_rate) {
    if (in.empty() || in_rate <= 0 || out_rate <= 0 || in_rate == out_rate)
        return in;

    size_t out_len = std::max<size_t>(
        1, static_cast<size_t>(std::llround(
               static_cast<double>(in.size()) * out_rate / in_rate)));
    std::vector<float> out(out_len);
    const double scale = static_cast<double>(in_rate) / out_rate;
    for (size_t i = 0; i < out_len; ++i) {
        double pos = static_cast<double>(i) * scale;
        size_t i0 = static_cast<size_t>(pos);
        if (i0 >= in.size() - 1) {
            out[i] = in.back();
            continue;
        }
        double frac = pos - static_cast<double>(i0);
        out[i] = static_cast<float>(
            static_cast<double>(in[i0]) * (1.0 - frac) +
            static_cast<double>(in[i0 + 1]) * frac);
    }
    return out;
}

DecodedAudio decode_wav_bytes(const std::string& bytes, int target_rate) {
    if (!is_wav_bytes(bytes))
        throw std::runtime_error("Reference audio is not a WAV file");

    uint16_t audio_fmt = 0;
    uint16_t channels = 0;
    uint16_t bits = 0;
    uint16_t block_align = 0;
    uint32_t sample_rate = 0;
    size_t data_off = 0;
    size_t data_size = 0;

    for (size_t off = 12; off + 8 <= bytes.size();) {
        std::string name = bytes.substr(off, 4);
        uint32_t chunk_size = read_le32(bytes, off + 4);
        size_t payload = off + 8;
        if (payload + chunk_size > bytes.size())
            throw std::runtime_error("Truncated WAV chunk");

        if (name == "fmt ") {
            if (chunk_size < 16)
                throw std::runtime_error("Invalid WAV fmt chunk");
            audio_fmt = read_le16(bytes, payload);
            channels = read_le16(bytes, payload + 2);
            sample_rate = read_le32(bytes, payload + 4);
            block_align = read_le16(bytes, payload + 12);
            bits = read_le16(bytes, payload + 14);
        } else if (name == "data") {
            data_off = payload;
            data_size = chunk_size;
        }

        off = payload + chunk_size + (chunk_size & 1u);
    }

    if (!audio_fmt || !channels || !sample_rate || !block_align || !data_size)
        throw std::runtime_error("WAV missing fmt or data chunk");

    bool is_pcm = audio_fmt == 1 && (bits == 8 || bits == 16 || bits == 32);
    bool is_float = audio_fmt == 3 && bits == 32;
    if (!is_pcm && !is_float)
        throw std::runtime_error("Unsupported WAV reference audio format");

    int frames = static_cast<int>(data_size / block_align);
    std::vector<float> mono(frames);
    const char* data = bytes.data() + data_off;

    for (int i = 0; i < frames; ++i) {
        float sum = 0.0f;
        for (int ch = 0; ch < channels; ++ch) {
            size_t off = static_cast<size_t>(i) * block_align +
                         static_cast<size_t>(ch) * (bits / 8);
            if (is_float) {
                float v;
                std::memcpy(&v, data + off, sizeof(float));
                sum += v;
            } else if (bits == 32) {
                int32_t v;
                std::memcpy(&v, data + off, sizeof(int32_t));
                sum += static_cast<float>(v) / 2147483648.0f;
            } else if (bits == 16) {
                int16_t v;
                std::memcpy(&v, data + off, sizeof(int16_t));
                sum += static_cast<float>(v) / 32768.0f;
            } else {
                uint8_t v = static_cast<uint8_t>(data[off]);
                sum += (static_cast<float>(v) - 128.0f) / 128.0f;
            }
        }
        mono[i] = sanitize_sample(sum / channels);
    }

    DecodedAudio decoded;
    decoded.sample_rate = target_rate > 0 ? target_rate : static_cast<int>(sample_rate);
    decoded.samples = resample_linear(mono, static_cast<int>(sample_rate), decoded.sample_rate);
    decoded.format = "wav";
    return decoded;
}

DecodedAudio decode_pcm_f32_bytes(const std::string& bytes, int sample_rate) {
    if (bytes.size() % sizeof(float) != 0)
        throw std::runtime_error("float32 PCM reference audio byte length is not divisible by 4");
    DecodedAudio decoded;
    decoded.sample_rate = sample_rate;
    decoded.format = "pcm_f32";
    decoded.samples.resize(bytes.size() / sizeof(float));
    std::memcpy(decoded.samples.data(), bytes.data(), bytes.size());
    for (float& sample : decoded.samples)
        sample = sanitize_sample(sample);
    return decoded;
}

}  // namespace

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

std::string encode_pcm_f32_base64(const float* samples, int n_samples) {
    std::vector<float> clean(static_cast<size_t>(std::max(0, n_samples)));
    for (int i = 0; i < n_samples; ++i)
        clean[static_cast<size_t>(i)] = sanitize_sample(samples[i]);
    std::string raw(reinterpret_cast<const char*>(clean.data()),
                    clean.size() * sizeof(float));
    return httplib::detail::base64_encode(raw);
}

std::string encode_wav_base64(const float* samples, int n_samples, int sample_rate) {
    const int safe_samples = std::max(0, n_samples);
    const uint32_t data_bytes = static_cast<uint32_t>(safe_samples) * 2;

    std::string wav;
    wav.reserve(44 + data_bytes);
    wav.append("RIFF", 4);
    append_le32(wav, 36 + data_bytes);
    wav.append("WAVEfmt ", 8);
    append_le32(wav, 16);
    append_le16(wav, 1);
    append_le16(wav, 1);
    append_le32(wav, static_cast<uint32_t>(sample_rate));
    append_le32(wav, static_cast<uint32_t>(sample_rate) * 2);
    append_le16(wav, 2);
    append_le16(wav, 16);
    wav.append("data", 4);
    append_le32(wav, data_bytes);

    for (int i = 0; i < safe_samples; ++i) {
        float v = sanitize_sample(samples[i]);
        int16_t pcm = static_cast<int16_t>(std::lrint(v * 32767.0f));
        append_le16(wav, static_cast<uint16_t>(pcm));
    }
    return httplib::detail::base64_encode(wav);
}

DecodedAudio decode_audio_base64(const std::string& encoded,
                                  int target_rate,
                                  int default_pcm_rate,
                                  const std::string& format_hint) {
    std::string bytes = base64_decode(encoded);
    if (bytes.empty())
        throw std::runtime_error("Reference audio base64 decoded to empty bytes");

    if (format_hint == "wav" || (format_hint == "auto" && is_wav_bytes(bytes)))
        return decode_wav_bytes(bytes, target_rate);
    if (format_hint == "pcm_f32" || format_hint == "float32" || format_hint == "auto")
        return decode_pcm_f32_bytes(bytes, default_pcm_rate > 0 ? default_pcm_rate : target_rate);

    throw std::runtime_error("Unsupported reference audio format: " + format_hint);
}

}  // namespace fish::routes
