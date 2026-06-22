// src/kernels/sampling.cu
// GPU-side top-k/top-p sampling with curand for DualAREngine codebook decode
#include "kernels/kernels.h"
#include "utils/cuda_utils.h"
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cmath>

namespace fish::kernels {

namespace {

__device__ int top_k_top_p_sample_sorted(
    float* sorted_logits,
    int* sorted_indices,
    int sz,
    float temperature,
    float top_p,
    int top_k,
    uint64_t seed
) {
    const int actual_k = min(max(top_k, 1), sz);
    for (int i = 0; i < actual_k; ++i) {
        int best = i;
        for (int j = i + 1; j < sz; ++j) {
            if (sorted_logits[j] > sorted_logits[best]) best = j;
        }
        if (best != i) {
            float tmp_logit = sorted_logits[i];
            sorted_logits[i] = sorted_logits[best];
            sorted_logits[best] = tmp_logit;
            int tmp_idx = sorted_indices[i];
            sorted_indices[i] = sorted_indices[best];
            sorted_indices[best] = tmp_idx;
        }
    }

    float max_l = -1e30f;
    for (int i = 0; i < actual_k; ++i)
        if (sorted_logits[i] > max_l) max_l = sorted_logits[i];

    float sum_e = 0.f;
    for (int i = 0; i < actual_k; ++i) {
        sorted_logits[i] = expf(sorted_logits[i] - max_l);
        sum_e += sorted_logits[i];
    }
    const float inv_sum = 1.0f / fmaxf(sum_e, 1e-20f);
    for (int i = 0; i < actual_k; ++i) sorted_logits[i] *= inv_sum;

    float cumsum = 0.f;
    int keep = 0;
    for (int i = 0; i < actual_k; ++i) {
        cumsum += sorted_logits[i];
        if (i > 0 && cumsum > top_p) break;
        keep = i + 1;
    }
    keep = max(keep, 1);

    float inv_temp = 1.0f / fmaxf(temperature, 1e-5f);
    max_l = -1e30f;
    for (int i = 0; i < keep; ++i) {
        sorted_logits[i] = logf(fmaxf(sorted_logits[i], 1e-20f)) * inv_temp;
        if (sorted_logits[i] > max_l) max_l = sorted_logits[i];
    }

    sum_e = 0.f;
    for (int i = 0; i < keep; ++i) {
        sorted_logits[i] = expf(sorted_logits[i] - max_l);
        sum_e += sorted_logits[i];
    }
    const float inv_sum_temp = 1.0f / fmaxf(sum_e, 1e-20f);
    for (int i = 0; i < keep; ++i) sorted_logits[i] *= inv_sum_temp;

    curandState_t state;
    curand_init(seed, 0, 0, &state);
    float r = curand_uniform(&state);
    float acc = 0.f;
    for (int i = 0; i < keep; ++i) {
        acc += sorted_logits[i];
        if (r < acc) return sorted_indices[i];
    }
    return sorted_indices[0];
}

__global__ void sampling_range_kernel(
    const __half* logits,
    int32_t* out_tokens,
    int total_vocab_size,
    int id_start,
    int vocab_size,
    float temperature,
    float top_p,
    int top_k,
    uint64_t seed
) {
    int b = blockIdx.x;
    extern __shared__ unsigned char smem[];
    float* s_logits = reinterpret_cast<float*>(smem);
    int* s_indices = reinterpret_cast<int*>(s_logits + vocab_size);

    if (threadIdx.x == 0) {
        for (int i = 0; i < vocab_size; ++i) {
            s_logits[i] = __half2float(logits[b * total_vocab_size + id_start + i]);
            s_indices[i] = id_start + i;
        }
        out_tokens[b] = top_k_top_p_sample_sorted(
            s_logits, s_indices, vocab_size, temperature, top_p, top_k, seed + b);
    }
}

__global__ void sampling_semantic_eos_kernel(
    const __half* logits,
    int32_t* out_tokens,
    int total_vocab_size,
    int semantic_start,
    int semantic_size,
    int eos_id,
    float temperature,
    float top_p,
    int top_k,
    uint64_t seed
) {
    int b = blockIdx.x;
    const int sz = semantic_size + 1;
    extern __shared__ unsigned char smem[];
    float* s_logits = reinterpret_cast<float*>(smem);
    int* s_indices = reinterpret_cast<int*>(s_logits + sz);

    if (threadIdx.x == 0) {
        for (int i = 0; i < semantic_size; ++i) {
            s_logits[i] = __half2float(logits[b * total_vocab_size + semantic_start + i]);
            s_indices[i] = semantic_start + i;
        }
        s_logits[semantic_size] = __half2float(logits[b * total_vocab_size + eos_id]);
        s_indices[semantic_size] = eos_id;
        out_tokens[b] = top_k_top_p_sample_sorted(
            s_logits, s_indices, sz, temperature, top_p, top_k, seed + b);
    }
}

}  // namespace

__global__ void sampling_kernel(
    __half* logits,
    int32_t* out_tokens,
    int vocab_size,
    float temperature,
    float top_p,
    int top_k,
    uint64_t seed
) {
    int b = blockIdx.x;
    int tid = threadIdx.x;

    // Shared memory for this batch element's logits in FP32
    extern __shared__ float s_data[];
    float* s_logits = s_data;
    int* s_indices = (int*)(s_data + vocab_size);

    // Load logit for this thread's index
    if (tid < vocab_size) {
        s_logits[tid] = __half2float(logits[b * vocab_size + tid]);
        s_indices[tid] = tid;
    }
    __syncthreads();

    // Temperature scaling
    float inv_temp = 1.0f / fmaxf(temperature, 1e-5f);
    if (tid < vocab_size) {
        s_logits[tid] *= inv_temp;
    }
    __syncthreads();

    // Softmax: find max (for numerical stability), then exp, then normalize
    // Thread 0 does sequential reduction for simplicity (vocab_size <= 32000)
    if (tid == 0) {
        float max_val = -1e30f;
        for (int i = 0; i < vocab_size; i++) {
            if (s_logits[i] > max_val) max_val = s_logits[i];
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            float v = expf(s_logits[i] - max_val);
            s_logits[i] = v;
            sum_exp += v;
        }
        float inv_sum = 1.0f / sum_exp;
        for (int i = 0; i < vocab_size; i++) {
            s_logits[i] *= inv_sum;
        }
    }
    __syncthreads();

    // Top-k filter: sort in shared memory by swapping
    // Use selection sort to find top-k elements
    if (tid == 0) {
        for (int i = 0; i < vocab_size && i < top_k; i++) {
            int best_j = i;
            for (int j = i + 1; j < vocab_size; j++) {
                if (s_logits[j] > s_logits[best_j]) best_j = j;
            }
            // Swap
            float tmp_v = s_logits[i];
            s_logits[i] = s_logits[best_j];
            s_logits[best_j] = tmp_v;
            int tmp_i = s_indices[i];
            s_indices[i] = s_indices[best_j];
            s_indices[best_j] = tmp_i;
        }
        // Zero out below top-k
        for (int i = top_k; i < vocab_size; i++) {
            s_logits[i] = 0.0f;
        }
    }
    __syncthreads();

    // Top-p: cumulative sum, truncate at top_p
    if (tid == 0) {
        float cumsum = 0.0f;
        int cutoff = vocab_size;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += s_logits[i];
            if (cumsum >= top_p && cutoff == vocab_size) {
                cutoff = i + 1;
            }
        }
        // Ensure at least 1 token survives
        if (cutoff < 1) cutoff = 1;
        // Zero out beyond cutoff
        for (int i = cutoff; i < vocab_size; i++) {
            s_logits[i] = 0.0f;
        }

        // Renormalize
        float sum_exp = 0.0f;
        for (int i = 0; i < cutoff; i++) sum_exp += s_logits[i];
        float inv_sum = 1.0f / fmaxf(sum_exp, 1e-10f);
        for (int i = 0; i < cutoff; i++) s_logits[i] *= inv_sum;
    }
    __syncthreads();

    // Multinomial sample (single sample per batch element)
    if (tid == 0) {
        curandState_t state;
        curand_init(seed + b, 0, 0, &state);
        float r = curand_uniform(&state);

        float cumsum = 0.0f;
        int chosen = 0;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += s_logits[i];
            if (r < cumsum) {
                chosen = s_indices[i];
                break;
            }
        }
        out_tokens[b] = chosen;
    }
}

void sample_top_k_top_p(
    half* logits,
    int32_t* out_tokens,
    int batch_size,
    int vocab_size,
    float temperature,
    float top_p,
    int top_k,
    uint64_t seed,
    cudaStream_t stream
) {
    int shmem = vocab_size * (sizeof(float) + sizeof(int));
    sampling_kernel<<<batch_size, 256, shmem, stream>>>(
        (__half*)logits, out_tokens, vocab_size,
        temperature, top_p, top_k, seed);
    CUDA_CHECK(cudaGetLastError());
}

void sample_top_k_top_p_range(
    const __half* logits,
    int32_t* out_tokens,
    int batch_size,
    int total_vocab_size,
    int id_start,
    int vocab_size,
    float temperature,
    float top_p,
    int top_k,
    uint64_t seed,
    cudaStream_t stream
) {
    if (vocab_size <= 0) return;
    const int shmem = vocab_size * (sizeof(float) + sizeof(int));
    sampling_range_kernel<<<batch_size, 1, shmem, stream>>>(
        logits, out_tokens, total_vocab_size, id_start, vocab_size,
        temperature, top_p, top_k, seed);
    CUDA_CHECK(cudaGetLastError());
}

void sample_top_k_top_p_semantic_eos(
    const __half* logits,
    int32_t* out_tokens,
    int batch_size,
    int total_vocab_size,
    int semantic_start,
    int semantic_size,
    int eos_id,
    float temperature,
    float top_p,
    int top_k,
    uint64_t seed,
    cudaStream_t stream
) {
    const int sz = semantic_size + 1;
    if (sz <= 0) return;
    const int shmem = sz * (sizeof(float) + sizeof(int));
    sampling_semantic_eos_kernel<<<batch_size, 1, shmem, stream>>>(
        logits, out_tokens, total_vocab_size, semantic_start, semantic_size, eos_id,
        temperature, top_p, top_k, seed);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace fish::kernels
