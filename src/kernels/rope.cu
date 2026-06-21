// src/kernels/rope.cu
#include "kernels/kernels.h"
#include "utils/cuda_utils.h"
#include <cuda_fp16.h>
#include <cmath>

namespace fish::kernels {

__global__ void rope_qk_kernel(
    __half* __restrict__ q,
    __half* __restrict__ k,
    const float* __restrict__ freqs,
    int n_heads_q,       // number of Q heads
    int n_heads_k,       // number of K heads (≤ n_heads_q for GQA)
    int seq_len,
    int head_dim,
    int offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total_pairs_q = n_heads_q * seq_len * (head_dim / 2);
    int total_pairs_k = n_heads_k * seq_len * (head_dim / 2);
    int total_pairs = max(total_pairs_q, total_pairs_k);
    if (idx >= total_pairs) return;

    int b = blockIdx.y;  // batch index
    float freq = freqs[idx % (head_dim / 2)];

    // Q rotation
    if (idx < total_pairs_q) {
        int head = idx / (seq_len * (head_dim / 2));
        int remain = idx % (seq_len * (head_dim / 2));
        int pos = remain / (head_dim / 2);
        int pair = remain % (head_dim / 2);

        float angle = freq * static_cast<float>(pos + offset);
        float cos_a = cosf(angle), sin_a = sinf(angle);

        int q_base = b * n_heads_q * seq_len * head_dim + head * seq_len * head_dim + pos * head_dim;
        // Don't reuse 'pair' from above since we recomputed freq
        int q_idx0 = q_base + pair * 2;
        int q_idx1 = q_idx0 + 1;
        float q0 = __half2float(q[q_idx0]), q1 = __half2float(q[q_idx1]);
        q[q_idx0] = __float2half(q0 * cos_a - q1 * sin_a);
        q[q_idx1] = __float2half(q0 * sin_a + q1 * cos_a);
    }

    // K rotation
    if (idx < total_pairs_k) {
        int head = idx / (seq_len * (head_dim / 2));
        int remain = idx % (seq_len * (head_dim / 2));
        int pos = remain / (head_dim / 2);
        int pair = remain % (head_dim / 2);

        float angle = freq * static_cast<float>(pos + offset);
        float cos_a = cosf(angle), sin_a = sinf(angle);

        int k_base = b * n_heads_k * seq_len * head_dim + head * seq_len * head_dim + pos * head_dim;
        int k_idx0 = k_base + pair * 2;
        int k_idx1 = k_idx0 + 1;
        float k0 = __half2float(k[k_idx0]), k1 = __half2float(k[k_idx1]);
        k[k_idx0] = __float2half(k0 * cos_a - k1 * sin_a);
        k[k_idx1] = __float2half(k0 * sin_a + k1 * cos_a);
    }
}

void rope_qk(
    const __half* q,
    const __half* k,
    const float* freqs,
    int batch_size,
    int n_heads_q,
    int n_heads_k,
    int seq_len,
    int head_dim,
    int offset,
    cudaStream_t stream
) {
    int total_pairs_q = n_heads_q * seq_len * (head_dim / 2);
    int total_pairs_k = n_heads_k * seq_len * (head_dim / 2);
    int total_pairs = (total_pairs_q > total_pairs_k) ? total_pairs_q : total_pairs_k;
    int threads = 256;
    int blocks = fish::div_up(total_pairs, threads);

    dim3 grid(blocks, batch_size);
    rope_qk_kernel<<<grid, threads, 0, stream>>>(
        const_cast<__half*>(q), const_cast<__half*>(k), freqs,
        n_heads_q, n_heads_k, seq_len, head_dim, offset);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace fish::kernels
