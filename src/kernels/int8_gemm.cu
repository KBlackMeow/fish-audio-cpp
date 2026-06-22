// src/kernels/int8_gemm.cu
// INT8 dequant + FP16 GEMM kernel.
//
// Computes Y = (W_int8 * row_scale) * X^T
//   W_int8: [M, K] row-major  (int8_t weights)
//   scale:  [M]               (__half, per-channel scale factors)
//   X:      [N, K] row-major  (__half, activation)
//   Y:      [M, N] col-major  (__half, ld=M — matches cuBLAS convention)
//
// Grid: dim3(M, N).  Block: 32 threads (1 warp).
// Each block handles one output element at (row=M, col=N).
// Single-warp design: no cross-warp reduction needed; each warp
// fully computes one dot product across K.

#include "kernels/kernels.h"
#include <cuda_fp16.h>
#include <cstdint>

namespace fish::kernels {
namespace {

static constexpr int TILE_K = 128;
static constexpr int WARP_SIZE = 32;

__global__ void int8_dequant_gemm_fp16_kernel(
    const int8_t* __restrict__ W,
    const __half* __restrict__ scale,
    const __half* __restrict__ X,
    __half* __restrict__ Y,
    int M, int N, int K)
{
    int row = blockIdx.x;
    int col = blockIdx.y;
    if (row >= M || col >= N) return;

    float s = __half2float(scale[row]);
    const int8_t* w_row = W + static_cast<size_t>(row) * K;
    const __half* x_row = X + static_cast<size_t>(col) * K;
    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        int k_lim = min(TILE_K, K - k0);

        // Each thread loads one element per iteration; loop over k_lim/WARP_SIZE
        for (int ki = threadIdx.x; ki < k_lim; ki += WARP_SIZE) {
            float w_val = static_cast<float>(w_row[k0 + ki]);
            float x_val = __half2float(x_row[k0 + ki]);
            acc += w_val * s * x_val;
        }
    }

    // Warp reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Lane 0 writes result
    if (threadIdx.x == 0) {
        Y[row + static_cast<size_t>(col) * M] = __float2half(acc);
    }
}

}  // namespace

void int8_dequant_gemm_fp16(
    const int8_t* W,
    const __half* scale,
    const __half* X,
    __half* Y,
    int M, int N, int K,
    cudaStream_t stream)
{
    dim3 grid(M, N);
    dim3 block(WARP_SIZE);
    int8_dequant_gemm_fp16_kernel<<<grid, block, 0, stream>>>(
        W, scale, X, Y, M, N, K);
}

}  // namespace fish::kernels
