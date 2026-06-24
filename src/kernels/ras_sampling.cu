// src/kernels/ras_sampling.cu
#include "kernels/kernels.h"
#include "utils/cuda_utils.h"

namespace fish::kernels {

__global__ void ras_sample_kernel(
    int32_t* main_token,
    const int32_t* high_token,
    const int32_t* previous_tokens,
    int32_t semantic_begin_id,
    int32_t semantic_end_id,
    int batch_size,
    int win_size
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    int32_t token = main_token[b];
    bool is_semantic = (token >= semantic_begin_id) && (token <= semantic_end_id);

    if (!is_semantic) return;

    // Check if token exists in previous_tokens window
    // previous_tokens: [B, 5, win_size] — 5 = 1 semantic + 4 codebooks
    bool in_window = false;
    for (int w = 0; w < win_size && !in_window; w++) {
        for (int c = 0; c < 5; c++) {
            if (previous_tokens[(b * 5 + c) * win_size + w] == token) {
                in_window = true;
                break;
            }
        }
    }

    if (in_window) {
        main_token[b] = high_token[b];
    }
}

void ras_sample(
    int32_t* main_token,
    const int32_t* high_token,
    const int32_t* previous_tokens,
    int32_t semantic_begin_id,
    int32_t semantic_end_id,
    int batch_size,
    int win_size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = div_up(batch_size, threads);
    ras_sample_kernel<<<blocks, threads, 0, stream>>>(
        main_token, high_token, previous_tokens,
        semantic_begin_id, semantic_end_id, batch_size, win_size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void semantic_history_update_kernel(
    int32_t* previous_tokens,
    const int32_t* next_token,
    int batch_size,
    int win_size)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    int32_t* semantic_hist = previous_tokens + (b * 5) * win_size;
    for (int i = 0; i < win_size - 1; ++i) {
        semantic_hist[i] = semantic_hist[i + 1];
    }
    semantic_hist[win_size - 1] = next_token[b];
}

void semantic_history_update(
    int32_t* previous_tokens,
    const int32_t* next_token,
    int batch_size,
    int win_size,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = div_up(batch_size, threads);
    semantic_history_update_kernel<<<blocks, threads, 0, stream>>>(
        previous_tokens, next_token, batch_size, win_size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void semantic_to_codebook0_kernel(
    const int32_t* semantic_token,
    int32_t* codebook0,
    int semantic_begin_id,
    int codebook_size,
    int batch_size)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    int32_t v = semantic_token[b] - semantic_begin_id;
    if (v < 0) v = 0;
    if (v >= codebook_size) v = codebook_size - 1;
    codebook0[b] = v;
}

void semantic_to_codebook0(
    const int32_t* semantic_token,
    int32_t* codebook0,
    int semantic_begin_id,
    int codebook_size,
    int batch_size,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = div_up(batch_size, threads);
    semantic_to_codebook0_kernel<<<blocks, threads, 0, stream>>>(
        semantic_token, codebook0, semantic_begin_id, codebook_size, batch_size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void pack_semantic_codebooks_kernel(
    const int32_t* semantic_token,
    const int32_t* codebook_tokens,
    int32_t* packed,
    int num_codebooks,
    int batch_size)
{
    int b = blockIdx.x;
    int tid = threadIdx.x;
    if (b >= batch_size) return;
    if (tid == 0) packed[b * (1 + num_codebooks)] = semantic_token[b];
    if (tid < num_codebooks) {
        packed[b * (1 + num_codebooks) + 1 + tid] =
            codebook_tokens[b * num_codebooks + tid];
    }
}

void pack_semantic_codebooks(
    const int32_t* semantic_token,
    const int32_t* codebook_tokens,
    int32_t* packed,
    int num_codebooks,
    int batch_size,
    cudaStream_t stream)
{
    const int threads = num_codebooks > 0 ? num_codebooks : 1;
    pack_semantic_codebooks_kernel<<<batch_size, threads, 0, stream>>>(
        semantic_token, codebook_tokens, packed, num_codebooks, batch_size);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace fish::kernels
