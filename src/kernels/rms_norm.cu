// src/kernels/rms_norm.cu
#include "kernels/kernels.h"
#include "utils/cuda_utils.h"
#include <cuda_fp16.h>

namespace fish::kernels {

static constexpr int KWARP = 32;

__global__ void rms_norm_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    __half* __restrict__ out,
    int dim,
    float eps
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float s_rms[];

    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = __half2float(x[token_idx * dim + i]);
        local_sum += val * val;
    }

    for (int offset = KWARP / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if ((tid & (KWARP - 1)) == 0) {
        s_rms[tid / KWARP] = local_sum;
    }
    __syncthreads();

    float total_sum = 0.0f;
    int num_warps = (blockDim.x + KWARP - 1) / KWARP;
    if (tid < num_warps) {
        total_sum = s_rms[tid];
    }
    for (int offset = KWARP / 2; offset > 0; offset /= 2) {
        total_sum += __shfl_down_sync(0xffffffff, total_sum, offset);
    }

    float rms = rsqrtf(total_sum / dim + eps);
    if (tid == 0) {
        s_rms[0] = rms;
    }
    __syncthreads();
    rms = s_rms[0];

    for (int i = tid; i < dim; i += blockDim.x) {
        float val = __half2float(x[token_idx * dim + i]) * rms;
        float w = __half2float(weight[i]);
        out[token_idx * dim + i] = __float2half(val * w);
    }
}

void rms_norm(
    const __half* x,
    const __half* weight,
    __half* out,
    int n_tokens,
    int dim,
    float eps,
    cudaStream_t stream
) {
    int threads = static_cast<int>(next_pow2(static_cast<size_t>(dim)));
    if (threads > 1024) threads = 1024;
    int shmem = ((threads + KWARP - 1) / KWARP) * sizeof(float);

    rms_norm_kernel<<<n_tokens, threads, shmem, stream>>>(
        x, weight, out, dim, eps);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace fish::kernels
