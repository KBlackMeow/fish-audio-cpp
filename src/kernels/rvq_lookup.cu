// src/kernels/rvq_lookup.cu
#include "kernels/kernels.h"
#include "utils/cuda_utils.h"
#include <cuda_fp16.h>

namespace fish::kernels {

__global__ void rvq_lookup_decode_kernel(
    const int32_t* __restrict__ codes,
    const __half* __restrict__ embeddings,
    __half* __restrict__ latents,
    int num_codebooks,
    int codebook_size,
    int seq_len,
    int latent_dim
) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;

    if (d >= latent_dim) return;

    float sum = 0.0f;
    for (int cb = 0; cb < num_codebooks; cb++) {
        int code = codes[((b * num_codebooks + cb) * seq_len) + t];
        // Bound check
        if (code < 0) code = 0;
        if (code >= codebook_size) code = codebook_size - 1;
        int emb_idx = ((cb * codebook_size) + code) * latent_dim + d;
        sum += __half2float(embeddings[emb_idx]);
    }

    // Output: [B, latent_dim, T] — channel-first layout
    int out_idx = ((b * latent_dim) + d) * seq_len + t;
    latents[out_idx] = __float2half(sum);
}

void rvq_lookup_decode(
    const int32_t* codes,
    const half* embeddings,
    half* latents,
    int batch_size,
    int num_codebooks,
    int codebook_size,
    int seq_len,
    int latent_dim,
    cudaStream_t stream
) {
    dim3 grid(batch_size, seq_len);
    int threads = min(256, static_cast<int>(next_pow2(static_cast<size_t>(latent_dim))));
    rvq_lookup_decode_kernel<<<grid, threads, 0, stream>>>(
        codes, (const __half*)embeddings, (__half*)latents,
        num_codebooks, codebook_size, seq_len, latent_dim);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// rvq_step_kernel — one codebook: lookup + out_proj dot-product into FP32
// ---------------------------------------------------------------------------
__global__ static void rvq_step_kernel_impl(
    const int32_t* __restrict__ codes_bt,   // [B, T]
    const __half*  __restrict__ codebook,   // [cb_size, D]
    const __half*  __restrict__ out_proj_w, // [L, D]
    const __half*  __restrict__ out_proj_b, // [L]
    float*         __restrict__ latent,     // [B, L, T]  accumulated
    int B, int T, int cb_size, int D, int L
) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int l = threadIdx.x + blockIdx.z * blockDim.x;

    if (b >= B || t >= T || l >= L) return;

    int code = codes_bt[b * T + t];
    if (code < 0) code = 0;
    if (code >= cb_size) code = cb_size - 1;
    const __half* emb = codebook + code * D;

    float acc = __half2float(out_proj_b[l]);
    const __half* w_row = out_proj_w + l * D;
    for (int d = 0; d < D; d++)
        acc += __half2float(w_row[d]) * __half2float(emb[d]);

    latent[(b * L + l) * T + t] += acc;
}

void rvq_step_per_codebook(
    const int32_t* codes_bt,
    const __half* codebook,
    const __half* out_proj_w,
    const __half* out_proj_b,
    float* latent,
    int B, int T, int cb_size, int D, int L,
    cudaStream_t stream
) {
    int block = 256;
    int latent_tiles = (L + block - 1) / block;
    dim3 grid(B, T, latent_tiles);
    rvq_step_kernel_impl<<<grid, block, 0, stream>>>(
        codes_bt, codebook, out_proj_w, out_proj_b, latent,
        B, T, cb_size, D, L);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// f32_to_f16_cast
// ---------------------------------------------------------------------------
__global__ static void f32_to_f16_kernel_impl(const float* src, __half* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

void f32_to_f16_cast(const float* src, __half* dst, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    f32_to_f16_kernel_impl<<<grid, block, 0, stream>>>(src, dst, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// f16_to_f32_cast
// ---------------------------------------------------------------------------
__global__ static void f16_to_f32_kernel_impl(const __half* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

void f16_to_f32_cast(const __half* src, float* dst, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    f16_to_f32_kernel_impl<<<grid, block, 0, stream>>>(src, dst, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// rvq_encode_step — one codebook: latent → nearest-neighbour → update residual
// ---------------------------------------------------------------------------
__global__ static void rvq_encode_step_kernel(
    const float* __restrict__ residual_in,   // [B, L, T]
    const __half*  __restrict__ in_proj_w,   // [D, L]
    const __half*  __restrict__ in_proj_b,   // [D] may be null
    const __half*  __restrict__ codebook,    // [cb_size, D]
    const __half*  __restrict__ out_proj_w,  // [L, D]
    const __half*  __restrict__ out_proj_b,  // [L] may be null
    int32_t*       __restrict__ codes_out,    // [B, T]
    float*         __restrict__ residual_out, // [B, L, T]
    int B, int T, int cb_size, int D, int L)
{
    int b = blockIdx.x;
    int t = blockIdx.y;
    if (b >= B || t >= T) return;

    // 1. Project through in_proj: query[d] = Σ_l residual[b,l,t] * in_proj[d,l]
    // in_proj is [D, L]; residual for this (b,t) is [L] with stride T
    float query[8];  // D ≤ 8 for DAC
    const float* res_bt = residual_in + (static_cast<size_t>(b) * L) * T + t;
    for (int d = 0; d < D; ++d) {
        float acc = 0.f;
        const __half* ip = in_proj_w + static_cast<size_t>(d) * L;
        for (int l = 0; l < L; ++l)
            acc += res_bt[static_cast<size_t>(l) * T] * __half2float(ip[l]);
        query[d] = acc + (in_proj_b ? __half2float(in_proj_b[d]) : 0.f);
    }

    // 2. L2-normalize query (matching Python F.normalize)
    float q_norm = 0.f;
    for (int d = 0; d < D; ++d) q_norm += query[d] * query[d];
    q_norm = sqrtf(fmaxf(q_norm, 1e-12f));
    float q_norm_inv = 1.f / q_norm;
    float query_normed[8];
    for (int d = 0; d < D; ++d) query_normed[d] = query[d] * q_norm_inv;

    // 3. Find nearest codebook entry (cosine similarity via normalized L2 dist)
    float best_dist = 1e30f;
    int   best_c    = 0;
    for (int c = 0; c < cb_size; ++c) {
        const __half* cb_vec = codebook + static_cast<size_t>(c) * D;
        // Normalize codebook entry
        float c_norm = 0.f;
        for (int d = 0; d < D; ++d) { float v = __half2float(cb_vec[d]); c_norm += v * v; }
        float c_norm_inv = 1.f / sqrtf(fmaxf(c_norm, 1e-12f));
        // Compute L2 distance between normalized vectors
        float dist = 0.f;
        for (int d = 0; d < D; ++d) {
            float diff = query_normed[d] - __half2float(cb_vec[d]) * c_norm_inv;
            dist += diff * diff;
        }
        if (dist < best_dist) { best_dist = dist; best_c = c; }
    }
    codes_out[b * T + t] = best_c;

    // 3. Subtract contribution from residual
    // residual_out[b,:,t] = residual_in[b,:,t] - (out_proj @ codebook[best_c] + out_proj_b)
    const __half* cb_vec = codebook + static_cast<size_t>(best_c) * D;
    float* r_out = residual_out + (static_cast<size_t>(b) * L) * T + t;
    for (int l = 0; l < L; ++l) {
        float dot = 0.f;
        const __half* op = out_proj_w + static_cast<size_t>(l) * D;
        for (int d = 0; d < D; ++d)
            dot += __half2float(op[d]) * __half2float(cb_vec[d]);
        float bias = out_proj_b ? __half2float(out_proj_b[l]) : 0.f;
        r_out[static_cast<size_t>(l) * T] = res_bt[static_cast<size_t>(l) * T] - (dot + bias);
    }
}

void rvq_encode_step(
    const float* residual_in,
    const __half* in_proj_w,
    const __half* in_proj_b,
    const __half* codebook,
    const __half* out_proj_w,
    const __half* out_proj_b,
    int32_t* codes_out,
    float* residual_out,
    int B, int T, int cb_size, int D, int L,
    cudaStream_t stream)
{
    dim3 grid(B, T);
    rvq_encode_step_kernel<<<grid, 1, 0, stream>>>(
        residual_in, in_proj_w, in_proj_b, codebook, out_proj_w, out_proj_b,
        codes_out, residual_out, B, T, cb_size, D, L);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace fish::kernels
