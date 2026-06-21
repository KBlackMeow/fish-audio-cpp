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

}  // namespace fish::kernels
