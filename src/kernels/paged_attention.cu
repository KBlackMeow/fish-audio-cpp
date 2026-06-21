// src/kernels/paged_attention.cu
#include "kernels/kernels.h"
#include "utils/cuda_utils.h"
#include <cuda_fp16.h>
#include <cmath>

namespace fish::kernels {

constexpr int BLOCK_SIZE = 16;   // tokens per physical block
constexpr int WARP_SIZE = 32;
constexpr int MAX_DIM = 128;     // max head_dim supported

__global__ void paged_attention_decode_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    __half* __restrict__ output,
    int max_blocks_per_seq,
    int n_heads,           // number of Q heads
    int head_dim,
    float sm_scale,
    int layer_idx,
    int n_cache_layers,    // stride for layer dimension in K/V cache
    int n_kv_heads         // number of KV heads (≤ n_heads for GQA)
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;  // Q head index (0..n_heads-1)
    int lane_id = threadIdx.x;

    int H = n_heads;
    int KvH = n_kv_heads;
    int D = head_dim;
    int kv_head = (head_idx * n_kv_heads) / n_heads;  // GQA mapping

    // Guard against unsupported head_dim
    if (D > MAX_DIM) return;

    int seq_len = seq_lens[batch_idx];

    // Guard against empty sequence (would produce NaN from 1.0/0.0)
    if (seq_len <= 0) {
        int out_base = ((batch_idx * H + head_idx)) * D;
        for (int d = lane_id; d < D; d += WARP_SIZE) {
            output[out_base + d] = __float2half(0.0f);
        }
        return;
    }

    // Load Q for this (batch, head) into registers
    // Q is [B, H, 1, D] → access as [B*H*D + head*D + d]
    float q_val[MAX_DIM];
    for (int d = lane_id; d < D; d += WARP_SIZE) {
        int q_idx = ((batch_idx * H + head_idx)) * D + d;
        q_val[d] = __half2float(q[q_idx]);
    }

    float local_max = -1e30f;
    float local_sum_exp = 0.0f;
    float local_acc[MAX_DIM] = {0.0f};

    int n_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Iterate over physical blocks in this sequence
    for (int block_i = 0; block_i < n_blocks; block_i++) {
        int phys_block = block_table[batch_idx * max_blocks_per_seq + block_i];
        int tokens_in_block = min(BLOCK_SIZE, seq_len - block_i * BLOCK_SIZE);

        for (int t = 0; t < tokens_in_block; t++) {
            // Compute dot product: q[d] · k[phys_block][layer][head][t][d]
            float dot = 0.0f;
            for (int d = lane_id; d < D; d += WARP_SIZE) {
                int k_idx = (((phys_block * n_cache_layers + layer_idx) * KvH + kv_head) * BLOCK_SIZE + t) * D + d;
                dot += q_val[d] * __half2float(k_cache[k_idx]);
            }

            // Warp reduction for dot product
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }

            // Broadcast dot to all lanes
            dot = __shfl_sync(0xffffffff, dot, 0);

            // Online softmax update
            float score = dot * sm_scale;
            float new_max = fmaxf(local_max, score);
            float scale = expf(local_max - new_max);
            float exp_score = expf(score - new_max);

            local_sum_exp = local_sum_exp * scale + exp_score;

            for (int d = lane_id; d < D; d += WARP_SIZE) {
                int v_idx = (((phys_block * n_cache_layers + layer_idx) * KvH + kv_head) * BLOCK_SIZE + t) * D + d;
                float v_val = __half2float(v_cache[v_idx]);
                local_acc[d] = local_acc[d] * scale + exp_score * v_val;
            }

            local_max = new_max;
        }
    }

    // Write final output (each lane writes its own dimensions)
    float inv_sum = 1.0f / local_sum_exp;
    for (int d = lane_id; d < D; d += WARP_SIZE) {
        int out_idx = ((batch_idx * H + head_idx)) * D + d;
        output[out_idx] = __float2half(local_acc[d] * inv_sum);
    }
}

void paged_attention_decode(
    const half* q,
    const half* k_cache,
    const half* v_cache,
    const int32_t* block_table,
    const int32_t* seq_lens,
    half* output,
    int batch_size,
    int n_heads,
    int head_dim,
    int max_blocks_per_seq,
    float sm_scale,
    int layer_idx,
    int n_cache_layers,
    int n_kv_heads,
    cudaStream_t stream
) {
    if (head_dim > MAX_DIM) {
        throw std::runtime_error("head_dim " + std::to_string(head_dim)
                                 + " exceeds MAX_DIM " + std::to_string(MAX_DIM));
    }

    dim3 grid(batch_size, n_heads);
    paged_attention_decode_kernel<<<grid, WARP_SIZE, 0, stream>>>(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        output,
        max_blocks_per_seq,
        n_heads,
        head_dim,
        sm_scale,
        layer_idx,
        n_cache_layers,
        n_kv_heads
    );
    CUDA_CHECK(cudaGetLastError());
}

// Fast decoder single-query attention — contiguous cache, short sequences.
__global__ void fast_attention_decode_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    __half* __restrict__ output,
    int n_q, int n_kv, int D, int T_kv, int max_len,
    float sm_scale)
{
    int b = blockIdx.x;
    int hq = blockIdx.y;
    int lane_id = threadIdx.x;
    int hkv = (hq * n_kv) / n_q;
    if (D > MAX_DIM) return;
    float q_val[MAX_DIM];
    for (int d = lane_id; d < D; d += WARP_SIZE)
        q_val[d] = __half2float(q[(b * n_q + hq) * D + d]);
    float local_max = -1e30f;
    float local_sum_exp = 0.0f;
    float local_acc[MAX_DIM] = {0.0f};
    for (int t = 0; t < T_kv; t++) {
        float dot = 0.0f;
        for (int d = lane_id; d < D; d += WARP_SIZE)
            dot += q_val[d] * __half2float(k_cache[(hkv * max_len + t) * D + d]);
        for (int off = WARP_SIZE / 2; off > 0; off /= 2)
            dot += __shfl_down_sync(0xffffffff, dot, off);
        dot = __shfl_sync(0xffffffff, dot, 0);
        float score = dot * sm_scale;
        float new_max = fmaxf(local_max, score);
        float scale = expf(local_max - new_max);
        float exp_score = expf(score - new_max);
        local_sum_exp = local_sum_exp * scale + exp_score;
        for (int d = lane_id; d < D; d += WARP_SIZE) {
            float v_val = __half2float(v_cache[(hkv * max_len + t) * D + d]);
            local_acc[d] = local_acc[d] * scale + exp_score * v_val;
        }
        local_max = new_max;
    }
    float inv_sum = 1.0f / local_sum_exp;
    for (int d = lane_id; d < D; d += WARP_SIZE)
        output[(b * n_q + hq) * D + d] = __float2half(local_acc[d] * inv_sum);
}

void fast_attention_decode(
    const __half* q, const __half* k_cache, const __half* v_cache,
    __half* output, int B, int n_q, int n_kv, int D, int T_kv, int max_len,
    float sm_scale, cudaStream_t stream)
{
    if (T_kv <= 0) {
        CUDA_CHECK(cudaMemsetAsync(output, 0, (size_t)B * n_q * D * sizeof(__half), stream));
        return;
    }
    if (D > MAX_DIM)
        throw std::runtime_error("fast_attention_decode: head_dim exceeds MAX_DIM");
    dim3 grid(B, n_q);
    fast_attention_decode_kernel<<<grid, WARP_SIZE, 0, stream>>>(
        q, k_cache, v_cache, output, n_q, n_kv, D, T_kv, max_len, sm_scale);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace fish::kernels
