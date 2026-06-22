// src/kernels/int8_gemm.cu
// INT8 weight-only dequant + cuBLAS FP16 GEMM.
//
// Dequant: W_fp16[m,k] = W_int8[m,k] * scale[m] * smooth_inv[k]
// (smooth_inv is optional — nullptr means no calibration smoothing)
// Then standard cuBLAS FP16 GEMM (Tensor Cores).
//
// Cached: each unique INT8 weight ptr is dequantized exactly once.

#include "kernels/kernels.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <unordered_map>

namespace fish::kernels {
namespace {

__global__ void dequant_weights_kernel(
    const int8_t* __restrict__ W_int8,
    const __half* __restrict__ scale,
    const __half* __restrict__ smooth_inv,  // [K] or nullptr
    __half* __restrict__ W_fp16,
    int M, int K)
{
    int row = blockIdx.x;
    if (row >= M) return;
    float s = __half2float(scale[row]);
    const int8_t* src = W_int8 + static_cast<size_t>(row) * K;
    __half* dst = W_fp16 + static_cast<size_t>(row) * K;

    if (smooth_inv) {
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            float si = __half2float(smooth_inv[k]);
            dst[k] = __float2half(static_cast<float>(src[k]) * s * si);
        }
    } else {
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            dst[k] = __float2half(static_cast<float>(src[k]) * s);
        }
    }
}

// Cache key combines INT8 pointer and smooth_inv pointer
struct CacheKey {
    const int8_t* w;
    const __half* si;
    bool operator==(const CacheKey& o) const { return w == o.w && si == o.si; }
};
struct CacheKeyHash {
    size_t operator()(const CacheKey& k) const {
        return reinterpret_cast<size_t>(k.w) ^
               (reinterpret_cast<size_t>(k.si) << 1);
    }
};

}  // namespace

void int8_dequant_gemm_fp16(
    const int8_t* W,
    const __half* scale,
    const __half* smooth_inv,
    const __half* X,
    __half* Y,
    int M, int N, int K,
    cublasHandle_t cublas,
    cudaStream_t stream)
{
    static std::unordered_map<CacheKey, __half*, CacheKeyHash> cache;

    CacheKey key{W, smooth_inv};
    __half* w_fp16 = nullptr;
    auto it = cache.find(key);
    if (it != cache.end()) {
        w_fp16 = it->second;
    } else {
        cudaMalloc(&w_fp16, static_cast<size_t>(M) * K * sizeof(__half));
        dequant_weights_kernel<<<M, 256, 0, stream>>>(
            W, scale, smooth_inv, w_fp16, M, K);
        cache[key] = w_fp16;
    }

    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        w_fp16, CUDA_R_16F, K,
        X, CUDA_R_16F, K,
        &beta,
        Y, CUDA_R_16F, M,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

}  // namespace fish::kernels
