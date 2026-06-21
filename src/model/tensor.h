#pragma once
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace fish {

enum class DType : uint32_t {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
};

constexpr int64_t dtype_size(DType dt) {
    switch (dt) {
        case DType::FP32: return 4;
        case DType::FP16: return 2;
        case DType::BF16: return 2;
    }
    return 0;  // unreachable for valid enum values
}

/// Non-owning view into tensor data. The caller must ensure the backing
/// memory (mmap'd region or GPU buffer) outlives this view.
struct TensorView {
    void* data;
    DType dtype;
    std::vector<int64_t> shape;

    template<typename T>
    T* as() { return static_cast<T*>(data); }

    template<typename T>
    const T* as() const { return static_cast<const T*>(data); }

    [[nodiscard]] int64_t numel() const {
        int64_t n = 1;
        for (int64_t s : shape) {
            // Guard against signed integer overflow (undefined behavior)
            if (n > INT64_MAX / s) {
                throw std::overflow_error("TensorView::numel overflow");
            }
            n *= s;
        }
        return n;
    }

    [[nodiscard]] int64_t nbytes() const { return numel() * dtype_size(dtype); }
};

/// POD struct matching the .bin file format header entry.
/// Must stay at 280 bytes for binary compatibility.
struct TensorHeader {
    char name[256];          // zero-padded tensor name
    uint32_t dtype_val;       // DType enum value
    uint32_t ndim;            // number of dimensions, ≤ 8
    int64_t shape[8];         // dimension sizes (0-padded)
    uint64_t data_offset;     // byte offset within .bin file (256B aligned)
    uint64_t data_size;       // byte count of tensor data
};
// 256 + 4 + 4 + (pad to 8) + 64 + 8 + 8 = 344
static_assert(sizeof(TensorHeader) == 344, "TensorHeader size must be 344 bytes for .bin format compatibility");

inline constexpr uint32_t MAGIC = 0x46495348;        // "FISH"
inline constexpr uint32_t FORMAT_VERSION = 1;

}  // namespace fish
