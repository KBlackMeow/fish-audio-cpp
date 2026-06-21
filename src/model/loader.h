#pragma once
#include "model/tensor.h"
#include <string>
#include <unordered_map>

namespace fish {

class ModelLoader {
public:
    ModelLoader() = default;
    ~ModelLoader();

    // Disable copy
    ModelLoader(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;

    // Move allowed
    ModelLoader(ModelLoader&& other) noexcept;
    ModelLoader& operator=(ModelLoader&& other) noexcept;

    // Load model from .bin file
    bool load(const std::string& bin_path);

    // Get tensor by name (zero-copy: points into mmap'd region)
    TensorView get(const std::string& name) const;

    // Check if tensor exists
    bool has(const std::string& name) const;

    // Number of loaded tensors
    size_t num_tensors() const { return headers_.size(); }

    // Pin mmap'd memory for GPU access (required before GPU inference)
    bool pin_memory();

    // Unload (munmap + close)
    void unload();

private:
    int fd_ = -1;
    void* mapped_data_ = nullptr;
    size_t mapped_size_ = 0;
    std::unordered_map<std::string, TensorHeader> headers_;
    uint64_t data_section_offset_ = 0;  // file offset to start of tensor data (for validation/logging)
};

}  // namespace fish
