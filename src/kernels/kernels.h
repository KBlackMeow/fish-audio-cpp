// src/kernels/kernels.h
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>

namespace fish::kernels {

// RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight
void rms_norm(
    const __half* x,        // [B*T, D]  flattened input
    const __half* weight,   // [D]
    __half* out,            // [B*T, D]
    int n_tokens,           // batch * seq_len
    int dim,
    float eps,
    cudaStream_t stream = 0
);

// RoPE: apply rotary position embedding to q and k in-place
void rope_qk(
    const __half* q,         // [B, Hq, T, D]
    const __half* k,         // [B, Hk, T, D]
    const float* freqs,      // [D/2]
    int batch_size,
    int n_heads_q,           // number of Q heads
    int n_heads_k,           // number of K heads (≤ Hq for GQA)
    int seq_len,
    int head_dim,
    int offset,
    cudaStream_t stream = 0
);

// Fast decoder specialization for B=1, T=1: split fused QKV, apply RoPE to
// Q/K, and scatter K/V directly into the current layer cache slot.
void fast_qkv_split_rope_cache(
    const __half* qkv,       // [n_q*D + 2*n_kv*D]
    __half* q_out,           // [n_q, D]
    __half* k_cache_layer,   // [n_kv, max_len, D]
    __half* v_cache_layer,   // [n_kv, max_len, D]
    const float* freqs,      // [D/2]
    int n_q,
    int n_kv,
    int head_dim,
    int position,
    int max_len,
    cudaStream_t stream = 0
);

// PagedAttention decode for single query per batch element
// Each CUDA block processes one (batch, head) pair
void paged_attention_decode(
    const __half* q,               // [B, H, 1, D]
    const __half* k_cache,         // [MAX_BLOCKS, N_LAYERS, H, 16, D]
    const __half* v_cache,         // same layout
    const int32_t* block_table,    // [B, MAX_BLOCKS_PER_SEQ]
    const int32_t* seq_lens,       // [B]
    __half* output,                // [B, H, 1, D]
    int batch_size,
    int n_heads,
    int head_dim,
    int max_blocks_per_seq,
    float sm_scale,                // 1.0 / sqrt(head_dim)
    int layer_idx,
    int n_cache_layers,            // number of layers in K/V cache
    int n_kv_heads,                // number of KV heads (≤ n_heads for GQA)
    cudaStream_t stream = 0
);

// RAS (Repetition Aware Sampling): replaces main token with
// high-temp token if semantic repetition detected
void ras_sample(
    int32_t* main_token,           // [B, 1]  in-place
    const int32_t* high_token,     // [B, 1]
    const int32_t* previous_tokens, // [B, 5, RAS_WIN_SIZE]
    int32_t semantic_begin_id,
    int32_t semantic_end_id,
    int batch_size,
    int win_size,
    cudaStream_t stream = 0
);

// SiLU activation: x = x * sigmoid(x)  (in-place)
void silu_forward(
    __half* x,
    int n,
    cudaStream_t stream = 0
);

// Fused SwiGLU activation: a = clamp(silu(a) * b, -limit, limit)
void silu_mul_clamp_forward(
    __half* a,
    const __half* b,
    int n,
    float limit,
    cudaStream_t stream = 0
);

// Element-wise multiply: a = a * b
void mul_forward(
    __half* a,
    const __half* b,
    int n,
    cudaStream_t stream = 0
);

// Element-wise add: a = a + b
void residual_add(
    __half* a,
    const __half* b,
    int n,
    cudaStream_t stream = 0
);

// Snake activation forward
void snake_forward(
    const half* x,
    half* out,
    int n,
    float alpha,
    cudaStream_t stream = 0
);

void snake_forward_channels(
    const half* x,
    half* out,
    const half* alpha,
    int B,
    int C,
    int L,
    cudaStream_t stream = 0
);

// GPU-side top-k/top-p sampling. Returns sampled token indices.
// logits: [B, vocab_size] in FP16 — modified in-place (temperature scaling)
void sample_top_k_top_p(
    __half* logits,           // [B, vocab_size], modified in-place
    int32_t* out_tokens,      // [B]
    int batch_size,
    int vocab_size,
    float temperature,
    float top_p,
    int top_k,
    uint64_t seed,
    cudaStream_t stream = 0
);

// Sample from logits[:, id_start:id_start+vocab_size] and return absolute token ids.
void sample_top_k_top_p_range(
    const __half* logits,     // [B, total_vocab_size]
    int32_t* out_tokens,      // [B]
    int batch_size,
    int total_vocab_size,
    int id_start,
    int vocab_size,
    float temperature,
    float top_p,
    int top_k,
    uint64_t seed,
    cudaStream_t stream = 0
);

// Sample from semantic range plus a separate eos logit and return absolute token ids.
void sample_top_k_top_p_semantic_eos(
    const __half* logits,     // [B, total_vocab_size]
    int32_t* out_tokens,      // [B]
    int batch_size,
    int total_vocab_size,
    int semantic_start,
    int semantic_size,
    int eos_id,
    float temperature,
    float top_p,
    int top_k,
    uint64_t seed,
    cudaStream_t stream = 0
);

// RVQ codebook lookup and sum
void rvq_lookup_decode(
    const int32_t* codes,          // [B, num_codebooks, T]
    const half* embeddings,        // [num_codebooks, codebook_size, latent_dim]
    half* latents,                  // [B, latent_dim, T]
    int batch_size,
    int num_codebooks,
    int codebook_size,
    int seq_len,
    int latent_dim,
    cudaStream_t stream = 0
);

// RVQ per-codebook step: lookup + out_proj projection into float32 latent accumulator.
//
//   codes_bt:    [B, T]  int32 — codes for this single codebook (row-major)
//   codebook:    [cb_size, D]  FP16
//   out_proj_w:  [L, D]  FP16  (linear weight, from Conv1d weight [L,D,1])
//   out_proj_b:  [L]     FP16
//   latent:      [B, L, T]  float32 — result accumulated (+=)
void rvq_step_per_codebook(
    const int32_t* codes_bt,
    const __half* codebook,
    const __half* out_proj_w,
    const __half* out_proj_b,
    float* latent,
    int B, int T, int cb_size, int D, int L,
    cudaStream_t stream = 0
);

// Cast float32 array to FP16 in-place (writes to dst)
void f32_to_f16_cast(const float* src, __half* dst, int n, cudaStream_t stream = 0);

// Cast FP16 array to float32 (writes to dst)
void f16_to_f32_cast(const __half* src, float* dst, int n, cudaStream_t stream = 0);

// INT8 weight + INT8 activation GEMM with FP16 output.
// W:          [M, K] row-major int8_t weights
// scale:      [M] row-wise or [M, G] group-wise FP16 scale factors
// act_scale:  [G] static activation scales or nullptr for dynamic quantization
// smooth_inv: [K] per-input-channel FP16 smooth (null if no calibration)
// X:          [N, K] row-major FP16 activation
// Y:          [M, N] col-major FP16 output (ld=M — matches cuBLAS)
void int8_dequant_gemm_fp16(
    const int8_t* W,
    const __half* scale,
    int group_size,
    const __half* act_scale,
    const __half* smooth_inv,
    const __half* X,
    __half* Y,
    int M, int N, int K,
    cublasHandle_t cublas,
    cudaStream_t stream = 0);

// GPU embedding lookup: indices → embeddings
void embedding_lookup(
    const int32_t* indices,
    const __half* table,
    __half* output,
    int n_tokens,
    int dim,
    int vocab_size,
    cudaStream_t stream = 0
);

void transpose_fp16(const __half* in, __half* out, int rows, int cols, int ld_in, int ld_out, cudaStream_t stream = 0);

// Scale all elements in-place: x *= scale
void scale_inplace(__half* x, int n, float scale, cudaStream_t stream = 0);
void clamp_finite_inplace(__half* x, int n, float limit, cudaStream_t stream = 0);

// Crop/copy [B,C,in_len] to [B,C,out_len] from start offset.
void crop_time_fp16(const __half* src, __half* dst, int B, int C,
                    int in_len, int out_len, int start, cudaStream_t stream = 0);

// Zero-pad time dimension: dst[..., left:left+in_len] = src, others zero.
void left_pad_time_fp16(const __half* src, __half* dst, int B, int C,
                        int in_len, int left, int right, cudaStream_t stream = 0);

// out = tanh(in) converted to float32.
void tanh_half_to_float(const __half* in, float* out, int n, cudaStream_t stream = 0);

// LayerNorm over channel dimension for channels-first tensors [B,C,T].
void layer_norm_channels_fp16(const __half* x, const __half* weight, const __half* bias,
                              __half* out, int B, int C, int T, float eps,
                              cudaStream_t stream = 0);

void gelu_forward(__half* x, int n, cudaStream_t stream = 0);

void channel_scale_residual(__half* x, const __half* y, const __half* gamma,
                            int B, int C, int T, cudaStream_t stream = 0);

void add_channel_bias(__half* x, const __half* bias, int B, int C, int T,
                      cudaStream_t stream = 0);

// x[row,dim] += y[row,dim] * gamma[dim]
void dim_scale_residual(__half* x, const __half* y, const __half* gamma,
                        int rows, int dim, cudaStream_t stream = 0);

// x[row,dim] += y[row,dim] * gamma[dim], where y is a float32 GEMM result.
void dim_scale_residual_from_f32(__half* x, const float* y, const __half* gamma,
                                 int rows, int dim, float limit, cudaStream_t stream = 0);

// GPU prefill attention using cuBLAS batched GEMM.
//   q:         [n_q, T, D] head-major
//   k, v:      [n_kv, T, D] head-major
//   out:       [n_q, T, D] head-major
//   workspace: scratch buffer, size >= (n_q*T*T + 2*n_q*T*D)*sizeof(half)
void prefill_attention_gpu(
    const __half* q, const __half* k, const __half* v,
    __half* out, __half* workspace,
    int n_q, int n_kv, int T, int D, float sm_scale,
    cublasHandle_t cublas, cudaStream_t stream,
    int window_size = 0);

void qkv_split_heads(const __half* qkv, __half* q, __half* k, __half* v,
                     int T, int n_heads, int head_dim, cudaStream_t stream = 0);

// GQA-aware: n_q may be > n_kv.  qkv [T, n_q*D+2*n_kv*D], q [n_q,T,D], k/v [n_kv,T,D].
void qkv_split_heads_gqa(const __half* qkv, __half* q, __half* k, __half* v,
                         int T, int n_q, int n_kv, int head_dim, cudaStream_t stream = 0);

void merge_heads(const __half* heads, __half* out,
                 int T, int n_heads, int head_dim, cudaStream_t stream = 0);

// Fast-decoder single-query attention over contiguous K/V cache.
void fast_attention_decode(
    const __half* q, const __half* k_cache, const __half* v_cache,
    __half* output, int B, int n_q, int n_kv, int D, int T_kv, int max_len,
    float sm_scale, cudaStream_t stream = 0);

void rope_qk_cis(__half* q, __half* k, const __half* freqs_cis,
                 int n_heads, int T, int head_dim, cudaStream_t stream = 0);

// Simple 1D conv for Cin=1 (used when cuDNN rejects narrow inputs).
void conv1d_cin1_fp16(
    const __half* x, const __half* weight, const __half* bias, __half* y,
    int B, int Cout, int L, int k, int paddedL,
    cudaStream_t stream = 0);

// General 1D conv with stride+dilation (encoder workhorse, replaces cuDNN).
void strided_conv1d_fp16(
    const __half* x, const __half* weight, const __half* bias, __half* y,
    int B, int Cin, int Cout, int L, int outL, int k, int stride, int dilation, int paddedL,
    cudaStream_t stream = 0);

// Depthwise 1D convolution with left-padded input (replaces cuDNN grouped conv).
void depthwise_conv1d_fp16(
    const __half* x, const __half* weight, const __half* bias, __half* y,
    int B, int C, int L, int k, int paddedL,
    cudaStream_t stream = 0);

// RVQ encode step: project residual to codebook space, find nearest neighbour,
// update residual for next codebook.
void rvq_encode_step(
    const float* residual_in, const __half* in_proj_w, const __half* in_proj_b,
    const __half* codebook, const __half* out_proj_w, const __half* out_proj_b,
    int32_t* codes_out, float* residual_out,
    int B, int T, int cb_size, int D, int L,
    cudaStream_t stream = 0);

}  // namespace fish::kernels
