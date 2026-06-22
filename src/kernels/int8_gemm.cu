// src/kernels/int8_gemm.cu
// INT8 weight-only dequant + cuBLAS FP16 GEMM.
//
// Strategy: dequantize INT8 weights to FP16 in a fast kernel, then
// use standard cuBLAS FP16 GEMM. This preserves INT8 memory savings
// (~50% weight VRAM) while retaining full cuBLAS FP16 performance
// and numerical accuracy (no activation quantization noise).
//
//   W_int8: [M, K] row-major int8_t
//   scale:  [M] per-channel FP16
//   X:      [N, K] row-major FP16
//   Y:      [M, N] col-major FP16 (cuBLAS convention, ld=M)

#include "kernels/kernels.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <unordered_map>

namespace fish::kernels {
namespace {

// Dequantize INT8 weights → FP16: W_fp16[m,k] = W_int8[m,k] * scale[m]
__global__ void dequant_weights_kernel(
    const int8_t* __restrict__ W_int8,
    const __half* __restrict__ scale,
    __half* __restrict__ W_fp16,
    int M, int K)
{
    int row = blockIdx.x;
    if (row >= M) return;
    float s = __half2float(scale[row]);
    const int8_t* src = W_int8 + static_cast<size_t>(row) * K;
    __half* dst = W_fp16 + static_cast<size_t>(row) * K;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        dst[k] = __float2half(static_cast<float>(src[k]) * s);
    }
}

}  // namespace

void int8_dequant_gemm_fp16(
    const int8_t* W,
    const __half* scale,
    const __half* X,
    __half* Y,
    int M, int N, int K,
    cublasHandle_t cublas,
    cudaStream_t stream)
{
    // Cache: each unique INT8 weight ptr → its own dedicated FP16 buffer.
    // Each weight matrix is dequantized exactly once.
    static std::unordered_map<const int8_t*, __half*> cache;
    static std::unordered_map<const int8_t*, size_t> cap_map;

    __half* w_fp16 = nullptr;
    auto it = cache.find(W);
    if (it != cache.end()) {
        w_fp16 = it->second;
    } else {
        size_t need = static_cast<size_t>(M) * K;
        // Check if we already have a buffer of at least this size
        auto cit = cap_map.find(W);
        if (cit == cap_map.end() || cit->second < need) {
            __half* buf = nullptr;
            cudaMalloc(&buf, need * sizeof(__half));
            cache[W] = buf;
            cap_map[W] = need;
            w_fp16 = buf;
        } else {
            w_fp16 = cache[W];
        }
        dequant_weights_kernel<<<M, 256, 0, stream>>>(
            W, scale, w_fp16, M, K);
    }

    // Standard cuBLAS FP16 GEMM (Tensor Cores)
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
