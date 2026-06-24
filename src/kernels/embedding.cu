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

__global__ void decode_embedding_lookup_kernel(
    const int32_t* __restrict__ semantic_token,
    const int32_t* __restrict__ codebook_tokens,
    const __half* __restrict__ text_table,
    const __half* __restrict__ codebook_table,
    __half* __restrict__ output,
    int batch_size,
    int dim,
    int vocab_size,
    int num_codebooks,
    int codebook_size,
    float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / dim;
    int d = idx % dim;
    if (b >= batch_size) return;

    int sem = semantic_token[b];
    if (sem < 0 || sem >= vocab_size) sem = 0;
    float acc = __half2float(text_table[sem * dim + d]);
    for (int cb = 0; cb < num_codebooks; ++cb) {
        int tok = codebook_tokens[b * num_codebooks + cb];
        int table_idx = tok + cb * codebook_size;
        int limit = num_codebooks * codebook_size;
        if (table_idx < 0 || table_idx >= limit) table_idx = 0;
        acc += __half2float(codebook_table[table_idx * dim + d]);
    }
    output[idx] = __float2half(acc * scale);
}

void decode_embedding_lookup(
    const int32_t* semantic_token,
    const int32_t* codebook_tokens,
    const __half* text_table,
    const __half* codebook_table,
    __half* output,
    int batch_size,
    int dim,
    int vocab_size,
    int num_codebooks,
    int codebook_size,
    float scale,
    cudaStream_t stream)
{
    int threads = 256;
    int total = batch_size * dim;
    int blocks = (total + threads - 1) / threads;
    decode_embedding_lookup_kernel<<<blocks, threads, 0, stream>>>(
        semantic_token, codebook_tokens, text_table, codebook_table, output,
        batch_size, dim, vocab_size, num_codebooks, codebook_size, scale);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace fish::kernels
