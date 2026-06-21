// src/tokenizer/tokenizer.h
#pragma once
#include <string>
#include <vector>
#include <memory>

// Forward-declare to avoid pulling in the full sentencepiece header
// in translation units that only use Tokenizer as a member.
namespace sentencepiece { class SentencePieceProcessor; }

namespace fish {

// Tokenizer backed by native C++ runtimes:
//   - tokenizer.json ByteLevel BPE (Qwen/Fish S2 Pro)
//   - tokenizer.model SentencePiece fallback for older checkpoints
class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    // model_path: directory containing tokenizer.json or tokenizer.model
    bool load(const std::string& model_path);

    // Encode text → token IDs.
    // The prompt is automatically wrapped in the im_start/im_end template.
    std::vector<int> encode(const std::string& text) const;

    // Encode an already-built prompt fragment without adding chat wrappers.
    std::vector<int> encode_raw(const std::string& text) const;

    std::string decode(const std::vector<int>& ids) const;

    int vocab_size()    const { return 155776; }
    int bos_id()        const { return 151643; }
    int eos_id()        const { return 151643; }
    int im_start_id()   const { return 151644; }
    int im_end_id()     const { return 151645; }

    // Build the prompt with chat template
    std::string build_prompt(const std::string& text) const;

private:
    std::vector<int> encode_native(const std::string& text) const;

    std::string model_dir_;
    bool use_native_ = false;
    bool use_byte_bpe_ = false;

    // PIMPL: sentencepiece::SentencePieceProcessor hidden behind raw pointer.
    // The unique_ptr destruction is handled in tokenizer.cc where the full
    // definition is visible.
    struct Impl;
    std::unique_ptr<Impl> sp_impl_;
};

}  // namespace fish
