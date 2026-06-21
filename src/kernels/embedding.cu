// src/kernels/embedding.cu — GPU embedding lookup
#include "kernels/kernels.h"
#include "utils/cuda_utils.h"
#include <cuda_fp16.h>

namespace fish::kernels {

__global__ void embedding_lookup_kernel(
    const int32_t* __restrict__ indices,
    const __half* __restrict__ table,
    __half* __restrict__ output,
    int n_tokens,
    int dim,
    int vocab_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tok = idx / dim;
    int d = idx % dim;
    if (tok >= n_tokens) return;

    int token = indices[tok];
    if (token < 0 || token >= vocab_size) token = 0;

    output[idx] = table[token * dim + d];
}

void embedding_lookup(
    const int32_t* indices,
    const __half* table,
    __half* output,
    int n_tokens,
    int dim,
    int vocab_size,
    cudaStream_t stream
) {
    int threads = 256;
    int total = n_tokens * dim;
    int blocks = (total + threads - 1) / threads;
    embedding_lookup_kernel<<<blocks, threads, 0, stream>>>(
        indices, table, output, n_tokens, dim, vocab_size);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace fish::kernels
