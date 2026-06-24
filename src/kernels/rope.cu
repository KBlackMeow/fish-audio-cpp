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

__global__ void fast_qk_split_rope_cache_kernel(
    const __half* __restrict__ qkv,
    __half* __restrict__ q_out,
    __half* __restrict__ k_cache_layer,
    const float* __restrict__ freqs,
    int n_q,
    int n_kv,
    int head_dim,
    int position,
    int max_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q_pairs = n_q * (head_dim / 2);
    int k_pairs = n_kv * (head_dim / 2);
    int total = max(q_pairs, k_pairs);
    if (idx >= total) return;

    if (idx < q_pairs) {
        int head = idx / (head_dim / 2);
        int pair = idx % (head_dim / 2);
        float angle = freqs[pair] * static_cast<float>(position);
        float c = cosf(angle), s = sinf(angle);
        int src = head * head_dim + pair * 2;
        float x0 = __half2float(qkv[src]);
        float x1 = __half2float(qkv[src + 1]);
        int dst = src;
        q_out[dst] = __float2half(x0 * c - x1 * s);
        q_out[dst + 1] = __float2half(x0 * s + x1 * c);
    }

    if (idx < k_pairs) {
        int head = idx / (head_dim / 2);
        int pair = idx % (head_dim / 2);
        float angle = freqs[pair] * static_cast<float>(position);
        float c = cosf(angle), s = sinf(angle);
        int src = n_q * head_dim + head * head_dim + pair * 2;
        float x0 = __half2float(qkv[src]);
        float x1 = __half2float(qkv[src + 1]);
        int dst = (head * max_len + position) * head_dim + pair * 2;
        k_cache_layer[dst] = __float2half(x0 * c - x1 * s);
        k_cache_layer[dst + 1] = __float2half(x0 * s + x1 * c);
    }
}

__global__ void fast_v_cache_scatter_kernel(
    const __half* __restrict__ qkv,
    __half* __restrict__ v_cache_layer,
    int n_q,
    int n_kv,
    int head_dim,
    int position,
    int max_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_kv * head_dim;
    if (idx >= total) return;
    int head = idx / head_dim;
    int d = idx % head_dim;
    int src = (n_q + n_kv) * head_dim + idx;
    int dst = (head * max_len + position) * head_dim + d;
    v_cache_layer[dst] = qkv[src];
}

void fast_qkv_split_rope_cache(
    const __half* qkv,
    __half* q_out,
    __half* k_cache_layer,
    __half* v_cache_layer,
    const float* freqs,
    int n_q,
    int n_kv,
    int head_dim,
    int position,
    int max_len,
    cudaStream_t stream)
{
    int threads = 256;
    int qk_pairs = max(n_q, n_kv) * (head_dim / 2);
    int qk_blocks = fish::div_up(qk_pairs, threads);
    fast_qk_split_rope_cache_kernel<<<qk_blocks, threads, 0, stream>>>(
        qkv, q_out, k_cache_layer, freqs, n_q, n_kv, head_dim, position, max_len);
    CUDA_CHECK(cudaGetLastError());

    int v_total = n_kv * head_dim;
    int v_blocks = fish::div_up(v_total, threads);
    fast_v_cache_scatter_kernel<<<v_blocks, threads, 0, stream>>>(
        qkv, v_cache_layer, n_q, n_kv, head_dim, position, max_len);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace fish::kernels
