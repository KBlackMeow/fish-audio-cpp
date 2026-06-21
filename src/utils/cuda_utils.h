// src/utils/cuda_utils.h
#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <stdexcept>
#include <string>
#include <spdlog/spdlog.h>

#define CUDA_CHECK(call) do {                              \
    cudaError_t err = (call);                              \
    if (err != cudaSuccess) {                              \
        throw std::runtime_error(std::string("CUDA error at ") \
            + __FILE__ + ":" + std::to_string(__LINE__)    \
            + ": " + cudaGetErrorString(err));             \
    }                                                      \
} while(0)

#define CUBLAS_CHECK(call) do {                            \
    cublasStatus_t err = (call);                           \
    if (err != CUBLAS_STATUS_SUCCESS) {                    \
        throw std::runtime_error(std::string("cuBLAS error at ") \
            + __FILE__ + ":" + std::to_string(__LINE__)    \
            + ": code=" + std::to_string(err));            \
    }                                                      \
} while(0)

#define CUDNN_CHECK(call) do {                             \
    cudnnStatus_t err = (call);                            \
    if (err != CUDNN_STATUS_SUCCESS) {                     \
        throw std::runtime_error(std::string("cuDNN error at ") \
            + __FILE__ + ":" + std::to_string(__LINE__)    \
            + ": " + cudnnGetErrorString(err));            \
    }                                                      \
} while(0)

namespace fish {

inline constexpr int div_up(int a, int b) { return (a + b - 1) / b; }

inline constexpr size_t next_pow2(size_t x) {
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

}  // namespace fish
