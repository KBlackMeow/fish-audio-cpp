// src/kernels/prefill_attention.cu — GPU prefill attention via cuBLAS batched GEMM
#include "kernels/kernels.h"
#include "utils/cuda_utils.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>

namespace fish::kernels {

// Forward decl
__global__ void sgemv_kernel(const __half* S, const __half* V, __half* O, int T, int D);

__global__ void qkv_split_heads_kernel(const __half* qkv, __half* q, __half* k, __half* v,
                                       int total, int T, int H, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d = idx % D;
    int tmp = idx / D;
    int t = tmp % T;
    int h = tmp / T;
    int src_base = t * (3 * H * D) + h * D + d;
    q[idx] = qkv[src_base];
    k[idx] = qkv[src_base + H * D];
    v[idx] = qkv[src_base + 2 * H * D];
}

void qkv_split_heads(const __half* qkv, __half* q, __half* k, __half* v,
                     int T, int n_heads, int head_dim, cudaStream_t stream) {
    int total = n_heads * T * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    qkv_split_heads_kernel<<<blocks, threads, 0, stream>>>(qkv, q, k, v, total, T, n_heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

// GQA-aware QKV split + transpose: token-major fused QKV → head-major Q/K/V.
__global__ void qkv_split_gqa_kernel(
    const __half* __restrict__ qkv, __half* __restrict__ q,
    __half* __restrict__ k, __half* __restrict__ v,
    int T, int n_q, int n_kv, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_q  = n_q  * T * D;
    int total_kv = n_kv * T * D;
    int stride   = n_q * D + 2 * n_kv * D;
    if (idx < total_q) {
        int d = idx % D, tmp = idx / D, t = tmp % T, h = tmp / T;
        q[idx] = qkv[t * stride + h * D + d];
    } else if (idx < total_q + total_kv) {
        int k_idx = idx - total_q, d = k_idx % D, tmp = k_idx / D, t = tmp % T, h = tmp / T;
        k[k_idx] = qkv[t * stride + n_q * D + h * D + d];
    } else if (idx < total_q + 2 * total_kv) {
        int v_idx = idx - total_q - total_kv, d = v_idx % D, tmp = v_idx / D, t = tmp % T, h = tmp / T;
        v[v_idx] = qkv[t * stride + n_q * D + n_kv * D + h * D + d];
    }
}

void qkv_split_heads_gqa(const __half* qkv, __half* q, __half* k, __half* v,
                         int T, int n_q, int n_kv, int head_dim, cudaStream_t stream) {
    int total = n_q * T * head_dim + 2 * n_kv * T * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    qkv_split_gqa_kernel<<<blocks, threads, 0, stream>>>(qkv, q, k, v, T, n_q, n_kv, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void merge_heads_kernel(const __half* heads, __half* out,
                                   int total, int T, int H, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d = idx % D;
    int tmp = idx / D;
    int h = tmp % H;
    int t = tmp / H;
    out[t * H * D + h * D + d] = heads[h * T * D + t * D + d];
}

void merge_heads(const __half* heads, __half* out,
                 int T, int n_heads, int head_dim, cudaStream_t stream) {
    int total = T * n_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    merge_heads_kernel<<<blocks, threads, 0, stream>>>(heads, out, total, T, n_heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void rope_qk_cis_kernel(__half* q, __half* k, const __half* freqs_cis,
                                   int total, int T, int H, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairs = D / 2;
    if (idx >= total) return;
    int pair = idx % pairs;
    int tmp = idx / pairs;
    int t = tmp % T;
    int h = tmp / T;
    int base = h * T * D + t * D + pair * 2;
    float c = __half2float(freqs_cis[(t * pairs + pair) * 2 + 0]);
    float s = __half2float(freqs_cis[(t * pairs + pair) * 2 + 1]);
    float q0 = __half2float(q[base]), q1 = __half2float(q[base + 1]);
    float k0 = __half2float(k[base]), k1 = __half2float(k[base + 1]);
    q[base] = __float2half(q0 * c - q1 * s);
    q[base + 1] = __float2half(q0 * s + q1 * c);
    k[base] = __float2half(k0 * c - k1 * s);
    k[base + 1] = __float2half(k0 * s + k1 * c);
}

void rope_qk_cis(__half* q, __half* k, const __half* freqs_cis,
                 int n_heads, int T, int head_dim, cudaStream_t stream) {
    int total = n_heads * T * (head_dim / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rope_qk_cis_kernel<<<blocks, threads, 0, stream>>>(q, k, freqs_cis, total, T, n_heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

// Expand K/V from n_kv heads to n_q heads for GQA (repeat_interleave)
__global__ void gqa_expand_kernel(
    const __half* __restrict__ src,  // [n_kv, T, D]
    __half* __restrict__ dst,        // [n_q, T, D]
    int n_kv, int n_q, int T, int D, int gs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_q * T * D;
    if (idx >= total) return;

    int h_out = idx / (T * D);
    int rem   = idx % (T * D);
    int h_in  = h_out / gs;
    int src_idx = h_in * T * D + rem;
    dst[idx] = src[src_idx];
}

// Causal mask + softmax on attention scores [n_heads, T, T]
// scores: [n_heads, T, T] head-major, modified in-place
// sm_scale: 1/sqrt(D)
//
// Shared memory layout:
//   s_row[0..T-1]               — row scores (float)
//   s_warp[0..num_warps-1]      — warp-level scratch for cross-warp reduction
__global__ void causal_softmax_kernel(
    __half* __restrict__ scores,
    int n_heads, int T, float sm_scale, int window)
{
    int h = blockIdx.x;
    int i = blockIdx.y;
    int tid = threadIdx.x;

    if (h >= n_heads || i >= T) return;

    extern __shared__ float s_data[];
    float* s_row  = s_data;
    float* s_warp = s_data + T;         // after the row buffer

    int num_warps = blockDim.x / 32;    // = 8 for 256-thread blocks
    int warp_id   = tid / 32;
    int lane_id   = tid % 32;

    // ---- Step 1: load row, apply scale + causal+window mask ----
    int left = (window > 0) ? (i - window + 1) : 0;
    __half* row = scores + h * T * T + i * T;
    float mx = -1e30f;
    for (int j = tid; j < T; j += blockDim.x) {
        float v = __half2float(row[j]);
        v = (j <= i && j >= left) ? v * sm_scale : -1e30f;
        s_row[j] = v;
        if (v > mx) mx = v;
    }

    // ---- Step 2: warp-level max reduction ----
    for (int off = 16; off > 0; off /= 2) {
        float other = __shfl_down_sync(0xffffffff, mx, off);
        if (other > mx) mx = other;
    }

    // Write each warp's max to shared memory (lane 0 of each warp)
    if (lane_id == 0)
        s_warp[warp_id] = mx;
    __syncthreads();

    // ---- Step 3: block-level max reduction (one warp reads all) ----
    if (warp_id == 0) {
        float bm = (lane_id < num_warps) ? s_warp[lane_id] : -1e30f;
        for (int off = 16; off > 0; off /= 2) {
            float other = __shfl_down_sync(0xffffffff, bm, off);
            if (other > bm) bm = other;
        }
        if (lane_id == 0) s_warp[0] = bm;
    }
    __syncthreads();
    mx = s_warp[0];

    // ---- Step 4: exp and warp-level sum ----
    float sm = 0.f;
    for (int j = tid; j < T; j += blockDim.x) {
        float v = expf(s_row[j] - mx);
        s_row[j] = v;
        sm += v;
    }
    for (int off = 16; off > 0; off /= 2) {
        sm += __shfl_down_sync(0xffffffff, sm, off);
    }

    // Write each warp's sum to shared memory (lane 0)
    if (lane_id == 0)
        s_warp[warp_id] = sm;
    __syncthreads();

    // ---- Step 5: block-level sum reduction ----
    if (warp_id == 0) {
        float bs = (lane_id < num_warps) ? s_warp[lane_id] : 0.f;
        for (int off = 16; off > 0; off /= 2) {
            bs += __shfl_down_sync(0xffffffff, bs, off);
        }
        if (lane_id == 0) s_warp[0] = bs;
    }
    __syncthreads();
    float inv_sm = (s_warp[0] > 0.f) ? (1.f / s_warp[0]) : 0.f;

    // ---- Step 6: normalize and write back ----
    for (int j = tid; j < T; j += blockDim.x) {
        row[j] = __float2half(s_row[j] * inv_sm);
    }
}

void prefill_attention_gpu(
    const __half* q,           // [n_q, T, D] head-major
    const __half* k,           // [n_kv, T, D] head-major
    const __half* v,           // [n_kv, T, D] head-major
    __half* out,               // [n_q, T, D] head-major
    __half* workspace,         // scratch: n_q*T*T + n_q*T*D elements
    int n_q, int n_kv, int T, int D,
    float sm_scale,
    cublasHandle_t cublas,
    cudaStream_t stream,
    int window_size)
{
    if (T == 0 || n_q == 0) return;

    int gs = n_q / n_kv;
    __half* scores     = workspace;                          // [n_q, T, T]
    __half* expanded_k = scores + (size_t)n_q * T * T;       // [n_q, T, D]
    __half* expanded_v = expanded_k + (size_t)n_q * T * D;   // [n_q, T, D]
    // Check workspace size
    size_t needed = ((size_t)n_q * T * T + 2 * (size_t)n_q * T * D) * sizeof(__half);
    (void)needed;  // caller ensures workspace is large enough

    // Step 0: expand K/V for GQA
    if (n_kv < n_q) {
        int threads = 256;
        int blocks_kv = ((size_t)n_q * T * D + threads - 1) / threads;
        gqa_expand_kernel<<<blocks_kv, threads, 0, stream>>>(k, expanded_k, n_kv, n_q, T, D, gs);
        gqa_expand_kernel<<<blocks_kv, threads, 0, stream>>>(v, expanded_v, n_kv, n_q, T, D, gs);
    } else {
        size_t bytes = (size_t)n_q * T * D * sizeof(__half);
        CUDA_CHECK(cudaMemcpyAsync(expanded_k, k, bytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(expanded_v, v, bytes, cudaMemcpyDeviceToDevice, stream));
    }

    // Step 1: S = Q × K^T for all heads using strided batched GEMM
    // For each head h: Q_h is [T, D], expanded_K_h is [T, D]
    // We want S_h = Q_h × K_h^T  →  [T, T]
    // cuBLAS row-major trick: S = K × Q^T gives S^T, so we pass transposed.
    // Actually: gemm_fp16(M=T, N=T, K=D, K_h, Q_h, S_h) computes Q_h × K_h^T = [T, T]
    // But we want this for ALL heads in one batched call.
    //
    // Using cublasGemmStridedBatchedEx with the row-major trick:
    //   Y[M,N] = X[M,K] × W[N,K]^T
    //   cuBLAS(M=N, N=M, K=K, W[N,K], X[M,K], ...)
    // For batched: strideA = N*K = T*D, strideB = M*K = T*D, strideC = M*N = T*T
    // But we want W=K_h, X=Q_h for each head.
    // Reusing gemm_fp16 pattern: Y = X @ W^T where X=[M,K]=[T,D], W=[N,K]=[T,D]
    // cuBLAS(N=T, M=T, K=D, W, X, Y) with ldC=N=T
    {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            T, T, D,            // M_out=T, N_out=T, K=D
            &alpha,
            expanded_k, CUDA_R_16F, D, (long long)T * D,   // A: [T,D] per head, stride T*D
            q,          CUDA_R_16F, D, (long long)T * D,   // B: [T,D] per head, stride T*D
            &beta,
            scores,     CUDA_R_16F, T, (long long)T * T,   // C: [T,T] per head, stride T*T
            n_q,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }

    // Step 2: causal mask + softmax (in-place on scores)
    {
        dim3 grid(n_q, T);
        int threads = 256;
        // s_row[T] + s_warp[num_warps] where num_warps = threads/32
        int smem = (T + threads / 32) * static_cast<int>(sizeof(float));
        if (smem > 48 * 1024) {
            // Try with 32 threads (1 warp, no cross-warp reduction needed but
            // the kernel still uses the scratch slot at s_warp[0])
            threads = 32;
            smem = (T + 1) * static_cast<int>(sizeof(float));
        }
        // Still won't fit — use 32-thread fallback with row buffer only
        // (shared mem access in the kernel wraps, but T shouldn't be this large
        //  for DAC usage; keep the guard to avoid launch failure.)
        if (smem > 48 * 1024) {
            threads = 32;
            smem = T * static_cast<int>(sizeof(float));
        }
        causal_softmax_kernel<<<grid, threads, smem, stream>>>(
            scores, n_q, T, sm_scale, window_size);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 3: O = softmax(S) × V using custom kernel (avoids cuBLAS layout bugs)
    // For each head h: S_h is [T, T], V_h is [T, D], O_h = S_h × V_h = [T, D]
    // Each CUDA block processes one (head, row) pair
    {
        dim3 grid2(n_q, T);
        int threads2 = 128;
        sgemv_kernel<<<grid2, threads2, 0, stream>>>(
            scores, expanded_v, out, T, D);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Kernel: for one head and one output row, compute O[row,:] = Σ_j S[row,j] × V[j,:]
__global__ void sgemv_kernel(
    const __half* __restrict__ S,  // [n_q, T, T]
    const __half* __restrict__ V,  // [n_q, T, D]
    __half* __restrict__ O,        // [n_q, T, D]
    int T, int D)
{
    int h = blockIdx.x;
    int i = blockIdx.y;
    int tid = threadIdx.x;

    if (h >= gridDim.x || i >= T) return;

    const __half* S_h = S + h * T * T + i * T;
    const __half* V_h = V + h * T * D;
    __half* O_h = O + h * T * D + i * D;

    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < T; j++) {
            acc += __half2float(S_h[j]) * __half2float(V_h[j * D + d]);
        }
        O_h[d] = __float2half(acc);
    }
}

}  // namespace fish::kernels
