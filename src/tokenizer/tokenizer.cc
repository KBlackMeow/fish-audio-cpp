// src/tokenizer/tokenizer.cc
// Native tokenizer implementations for C++ inference.
#include "tokenizer/tokenizer.h"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <sentencepiece_processor.h>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <unistd.h>
#include <vector>
#include <string>

namespace fish {

// PIMPL definition: wraps sentencepiece::SentencePieceProcessor
struct Tokenizer::Impl {
    sentencepiece::SentencePieceProcessor sp;
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<std::string, int> special_vocab;
    std::vector<std::string> special_tokens;
    std::unordered_map<std::string, int> merge_ranks;
    std::array<std::string, 256> byte_encoder;
    std::unordered_map<std::string, std::vector<int>> bpe_cache;
};

namespace {

std::string utf8_encode(uint32_t cp) {
    std::string out;
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

std::array<std::string, 256> build_byte_encoder() {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    std::array<std::string, 256> out{};
    for (size_t i = 0; i < bs.size(); ++i)
        out[static_cast<uint8_t>(bs[i])] = utf8_encode(static_cast<uint32_t>(cs[i]));
    return out;
}

uint32_t next_codepoint(const std::string& s, size_t i, size_t* next) {
    const auto c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
        *next = i + 1;
        return c;
    }
    if ((c >> 5) == 0x6 && i + 1 < s.size()) {
        *next = i + 2;
        return ((c & 0x1F) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3F);
    }
    if ((c >> 4) == 0xE && i + 2 < s.size()) {
        *next = i + 3;
        return ((c & 0x0F) << 12) |
               ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6) |
               (static_cast<unsigned char>(s[i + 2]) & 0x3F);
    }
    if ((c >> 3) == 0x1E && i + 3 < s.size()) {
        *next = i + 4;
        return ((c & 0x07) << 18) |
               ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12) |
               ((static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6) |
               (static_cast<unsigned char>(s[i + 3]) & 0x3F);
    }
    *next = i + 1;
    return c;
}

bool is_space_cp(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == 0x0B || cp == 0x0C;
}

bool is_newline_cp(uint32_t cp) {
    return cp == '\n' || cp == '\r';
}

bool is_digit_cp(uint32_t cp) {
    return cp >= '0' && cp <= '9';
}

bool is_letter_cp(uint32_t cp) {
    return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || cp >= 0x80;
}

bool is_letter_or_num_cp(uint32_t cp) {
    return is_letter_cp(cp) || is_digit_cp(cp);
}

bool starts_with_at(const std::string& s, size_t pos, const std::string& needle) {
    return pos + needle.size() <= s.size() &&
           s.compare(pos, needle.size(), needle) == 0;
}

std::vector<std::string> split_bytelevel_pretokens(const std::string& text) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < text.size()) {
        size_t n = i;
        uint32_t cp = next_codepoint(text, i, &n);

        if (cp == '\'' && n < text.size()) {
            static const std::vector<std::string> suffixes =
                {"s", "t", "re", "ve", "m", "ll", "d",
                 "S", "T", "RE", "VE", "M", "LL", "D"};
            bool matched = false;
            for (const auto& suf : suffixes) {
                if (starts_with_at(text, n, suf)) {
                    out.push_back(text.substr(i, 1 + suf.size()));
                    i = n + suf.size();
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }

        if (is_space_cp(cp)) {
            size_t j = i;
            bool has_newline = false;
            while (j < text.size()) {
                size_t nn = j;
                uint32_t cc = next_codepoint(text, j, &nn);
                if (!is_space_cp(cc)) break;
                has_newline = has_newline || is_newline_cp(cc);
                j = nn;
            }
            if (!has_newline && j < text.size()) {
                size_t after = j;
                uint32_t next_cp = next_codepoint(text, j, &after);
                if (is_letter_cp(next_cp)) {
                    size_t k = after;
                    while (k < text.size()) {
                        size_t kk = k;
                        uint32_t c2 = next_codepoint(text, k, &kk);
                        if (!is_letter_cp(c2)) break;
                        k = kk;
                    }
                    out.push_back(text.substr(i, k - i));
                    i = k;
                    continue;
                }
                if (!is_space_cp(next_cp) && !is_letter_or_num_cp(next_cp)) {
                    size_t k = after;
                    while (k < text.size()) {
                        size_t kk = k;
                        uint32_t c2 = next_codepoint(text, k, &kk);
                        if (is_space_cp(c2) || is_letter_or_num_cp(c2)) break;
                        k = kk;
                    }
                    while (k < text.size()) {
                        size_t kk = k;
                        uint32_t c2 = next_codepoint(text, k, &kk);
                        if (!is_newline_cp(c2)) break;
                        k = kk;
                    }
                    out.push_back(text.substr(i, k - i));
                    i = k;
                    continue;
                }
            }
            out.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        if (is_letter_cp(cp)) {
            size_t j = n;
            while (j < text.size()) {
                size_t nn = j;
                uint32_t cc = next_codepoint(text, j, &nn);
                if (!is_letter_cp(cc)) break;
                j = nn;
            }
            out.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        if (is_digit_cp(cp)) {
            out.push_back(text.substr(i, n - i));
            i = n;
            continue;
        }

        size_t j = n;
        while (j < text.size()) {
            size_t nn = j;
            uint32_t cc = next_codepoint(text, j, &nn);
            if (is_space_cp(cc) || is_letter_or_num_cp(cc)) break;
            j = nn;
        }
        while (j < text.size()) {
            size_t nn = j;
            uint32_t cc = next_codepoint(text, j, &nn);
            if (!is_newline_cp(cc)) break;
            j = nn;
        }
        out.push_back(text.substr(i, j - i));
        i = j;
    }
    return out;
}

std::string pair_key(const std::string& a, const std::string& b) {
    std::string key;
    key.reserve(a.size() + b.size() + 1);
    key.append(a);
    key.push_back('\t');
    key.append(b);
    return key;
}

}  // namespace

// ── Tokenizer ────────────────────────────────────────────────────────────────

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;

std::string Tokenizer::build_prompt(const std::string& text) const {
    // Matches fish_speech inference.py generate_long:
    //   System prompt: "convert the provided text to speech" (TTS instruction)
    //   <|voice|> only at the start of the assistant message.
    return "<|im_start|>system\n"
           "convert the provided text to speech<|im_end|>\n"
           "<|im_start|>user\n" + text +
           "<|im_end|>\n<|im_start|>assistant\n<|voice|>";
}

bool Tokenizer::load(const std::string& model_path) {
    model_dir_ = model_path;

    std::string json_path = model_path + "/tokenizer.json";
    if (access(json_path.c_str(), F_OK) == 0) {
        try {
            std::ifstream f(json_path);
            nlohmann::json j;
            f >> j;

            if (j.value("model", nlohmann::json{}).value("type", "") != "BPE")
                throw std::runtime_error("tokenizer.json model is not BPE");

            sp_impl_ = std::make_unique<Impl>();
            sp_impl_->byte_encoder = build_byte_encoder();

            const auto& vocab = j["model"]["vocab"];
            for (auto it = vocab.begin(); it != vocab.end(); ++it)
                sp_impl_->vocab[it.key()] = it.value().get<int>();

            if (j.contains("added_tokens")) {
                for (const auto& tok : j["added_tokens"]) {
                    std::string content = tok.value("content", "");
                    int id = tok.value("id", -1);
                    if (!content.empty() && id >= 0) {
                        sp_impl_->special_vocab[content] = id;
                        sp_impl_->special_tokens.push_back(content);
                    }
                }
            }

            std::sort(sp_impl_->special_tokens.begin(), sp_impl_->special_tokens.end(),
                      [](const std::string& a, const std::string& b) {
                          return a.size() > b.size();
                      });

            const auto& merges = j["model"]["merges"];
            int rank = 0;
            for (const auto& m : merges) {
                std::string a;
                std::string b;
                if (m.is_array() && m.size() == 2) {
                    a = m[0].get<std::string>();
                    b = m[1].get<std::string>();
                } else if (m.is_string()) {
                    std::string line = m.get<std::string>();
                    size_t sep = line.find(' ');
                    if (sep == std::string::npos) continue;
                    a = line.substr(0, sep);
                    b = line.substr(sep + 1);
                } else {
                    continue;
                }
                sp_impl_->merge_ranks[pair_key(a, b)] = rank++;
            }

            use_byte_bpe_ = true;
            use_native_ = true;
            spdlog::info("Tokenizer: native ByteLevel BPE → {} (vocab={}, merges={}, specials={})",
                         json_path, sp_impl_->vocab.size(), sp_impl_->merge_ranks.size(),
                         sp_impl_->special_vocab.size());
            return true;
        } catch (const std::exception& e) {
            spdlog::warn("Tokenizer: failed to load tokenizer.json: {}", e.what());
            sp_impl_.reset();
            use_byte_bpe_ = false;
            use_native_ = false;
        }
    }

    std::string sp_path = model_path + "/tokenizer.model";
    if (access(sp_path.c_str(), F_OK) == 0) {
        try {
            sp_impl_ = std::make_unique<Impl>();
            auto status = sp_impl_->sp.Load(sp_path);
            if (status.ok()) {
                use_byte_bpe_ = false;
                use_native_ = true;
                spdlog::info("Tokenizer: native sentencepiece → {}", sp_path);
                return true;
            }
            spdlog::warn("Tokenizer: failed to load sentencepiece model: {}",
                         status.ToString());
            sp_impl_.reset();
        } catch (const std::exception& e) {
            spdlog::warn("Tokenizer: sentencepiece exception: {}", e.what());
            sp_impl_.reset();
        }
    }

    spdlog::error("Tokenizer: neither tokenizer.json ByteLevel BPE nor tokenizer.model could be loaded in {}",
                  model_path);
    return false;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    return encode_native(build_prompt(text));
}

std::vector<int> Tokenizer::encode_raw(const std::string& text) const {
    return encode_native(text);
}

std::vector<int> Tokenizer::encode_native(const std::string& text) const {
    if (!use_native_ || !sp_impl_)
        throw std::runtime_error("Tokenizer is not loaded");

    if (use_byte_bpe_) {
        std::vector<int> ids;
        size_t i = 0;
        while (i < text.size()) {
            bool special_matched = false;
            for (const auto& special : sp_impl_->special_tokens) {
                if (starts_with_at(text, i, special)) {
                    ids.push_back(sp_impl_->special_vocab.at(special));
                    i += special.size();
                    special_matched = true;
                    break;
                }
            }
            if (special_matched) continue;

            size_t next_special = text.size();
            for (const auto& special : sp_impl_->special_tokens) {
                size_t pos = text.find(special, i);
                if (pos != std::string::npos)
                    next_special = std::min(next_special, pos);
            }

            std::string segment = text.substr(i, next_special - i);
            for (const auto& pretoken : split_bytelevel_pretokens(segment)) {
                std::string byte_token;
                for (unsigned char b : pretoken)
                    byte_token += sp_impl_->byte_encoder[b];

                auto cache_it = sp_impl_->bpe_cache.find(byte_token);
                if (cache_it != sp_impl_->bpe_cache.end()) {
                    ids.insert(ids.end(), cache_it->second.begin(), cache_it->second.end());
                    continue;
                }

                std::vector<std::string> parts;
                for (unsigned char b : pretoken)
                    parts.push_back(sp_impl_->byte_encoder[b]);

                while (parts.size() > 1) {
                    int best_rank = std::numeric_limits<int>::max();
                    size_t best_idx = parts.size();
                    for (size_t p = 0; p + 1 < parts.size(); ++p) {
                        auto it = sp_impl_->merge_ranks.find(pair_key(parts[p], parts[p + 1]));
                        if (it != sp_impl_->merge_ranks.end() && it->second < best_rank) {
                            best_rank = it->second;
                            best_idx = p;
                        }
                    }
                    if (best_idx == parts.size()) break;
                    parts[best_idx] += parts[best_idx + 1];
                    parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(best_idx + 1));
                }

                std::vector<int> cached_ids;
                cached_ids.reserve(parts.size());
                for (const auto& part : parts) {
                    auto vit = sp_impl_->vocab.find(part);
                    if (vit == sp_impl_->vocab.end())
                        throw std::runtime_error("Tokenizer BPE piece not in vocab: " + part);
                    cached_ids.push_back(vit->second);
                }
                sp_impl_->bpe_cache.emplace(byte_token, cached_ids);
                ids.insert(ids.end(), cached_ids.begin(), cached_ids.end());
            }

            i = next_special;
        }

        spdlog::info("  Tokenized {} chars → {} tokens (native BPE)",
                     text.size(), ids.size());
        return ids;
    }

    std::vector<int> ids;
    sp_impl_->sp.Encode(text, &ids);
    spdlog::info("  Tokenized {} chars → {} tokens (native)",
                 text.size(), ids.size());
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    if (use_native_ && sp_impl_) {
        std::string text;
        sp_impl_->sp.Decode(ids, &text);
        return text;
    }
    return "";
}

}  // namespace fish
