// src/kernels/snake.cu
#include "kernels/kernels.h"
#include "utils/cuda_utils.h"
#include <cuda_fp16.h>
#include <cmath>

namespace fish::kernels {

__global__ void snake_forward_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int n,
    float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = __half2float(x[idx]);
    float sin_val = sinf(alpha * val);
    // Snake: f(x) = x + (1/α) * sin²(α·x)
    // (Standard BigVGAN / Snake activation, NOT 1/α²)
    float alpha_inv = 1.0f / alpha;
    out[idx] = __float2half(val + alpha_inv * sin_val * sin_val);
}

void snake_forward(
    const half* x,
    half* out,
    int n,
    float alpha,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = div_up(n, threads);
    snake_forward_kernel<<<blocks, threads, 0, stream>>>(
        (const __half*)x, (__half*)out, n, alpha);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void snake_forward_channels_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    const __half* __restrict__ alpha,
    int total,
    int C,
    int L
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int c = (idx / L) % C;
    float a = fmaxf(__half2float(alpha[c]), 1e-9f);
    float val = __half2float(x[idx]);
    float sin_val = sinf(a * val);
    out[idx] = __float2half(val + (sin_val * sin_val) / a);
}

void snake_forward_channels(
    const half* x,
    half* out,
    const half* alpha,
    int B,
    int C,
    int L,
    cudaStream_t stream
) {
    int total = B * C * L;
    int threads = 256;
    int blocks = div_up(total, threads);
    snake_forward_channels_kernel<<<blocks, threads, 0, stream>>>(
        (const __half*)x, (__half*)out, (const __half*)alpha, total, C, L);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void crop_time_kernel(const __half* src, __half* dst,
                                 int total, int in_len, int out_len, int start) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int t = idx % out_len;
    int row = idx / out_len;
    dst[idx] = src[row * in_len + start + t];
}

void crop_time_fp16(const __half* src, __half* dst, int B, int C,
                    int in_len, int out_len, int start, cudaStream_t stream) {
    int total = B * C * out_len;
    int threads = 256;
    int blocks = div_up(total, threads);
    crop_time_kernel<<<blocks, threads, 0, stream>>>(src, dst, total, in_len, out_len, start);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void left_pad_time_kernel(const __half* src, __half* dst,
                                     int total, int in_len, int out_len, int left) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int t = idx % in_len;
    int row = idx / in_len;
    dst[row * out_len + left + t] = src[idx];
}

void left_pad_time_fp16(const __half* src, __half* dst, int B, int C,
                        int in_len, int left, int right, cudaStream_t stream) {
    int out_len = in_len + left + right;
    CUDA_CHECK(cudaMemsetAsync(dst, 0, static_cast<size_t>(B) * C * out_len * sizeof(__half), stream));
    int total = B * C * in_len;
    int threads = 256;
    int blocks = div_up(total, threads);
    left_pad_time_kernel<<<blocks, threads, 0, stream>>>(src, dst, total, in_len, out_len, left);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void tanh_half_to_float_kernel(const __half* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = tanhf(__half2float(in[idx]));
}

void tanh_half_to_float(const __half* in, float* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = div_up(n, threads);
    tanh_half_to_float_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
    CUDA_CHECK(cudaGetLastError());
}

// Depthwise 1D convolution: y[b,c,t] = bias[c] + Σ_w weight[c*k+w] * x[b*c*L+c*L+t+w]
// Input x is already left-padded (k-1 zeros on the left).
// Weight layout: [C, k] flat, kernel_size = k.
__global__ void depthwise_conv1d_kernel(
    const __half* __restrict__ x,       // [B, C, paddedL] — already left-padded
    const __half* __restrict__ weight,  // [C, k]
    const __half* __restrict__ bias,    // [C]
    __half* __restrict__ y,             // [B, C, L]
    int B, int C, int L, int k, int paddedL)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * L;
    if (idx >= total) return;

    int t = idx % L;
    int tmp = idx / L;
    int c = tmp % C;
    int b = tmp / C;

    float acc = __half2float(bias[c]);
    const __half* w = weight + static_cast<size_t>(c) * k;
    const __half* xc = x + static_cast<size_t>(b * C + c) * paddedL + t;
    for (int i = 0; i < k; ++i)
        acc += __half2float(w[i]) * __half2float(xc[i]);

    y[idx] = __float2half(acc);
}

// Simple 1D convolution for Cin=1 (used by encoder.block.0 which cuDNN rejects)
// weight layout: [Cout, k], bias: [Cout], x layout: [B, 1, paddedL], y: [B, Cout, L]
__global__ void conv1d_cin1_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ y,
    int B, int Cout, int L, int k, int paddedL)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * Cout * L;
    if (idx >= total) return;

    int t = idx % L;
    int tmp = idx / L;
    int c = tmp % Cout;
    int b = tmp / Cout;

    float acc = __half2float(bias[c]);
    const __half* w = weight + static_cast<size_t>(c) * k;
    const __half* xb = x + static_cast<size_t>(b) * paddedL + t;
    for (int i = 0; i < k; ++i)
        acc += __half2float(w[i]) * __half2float(xb[i]);

    y[idx] = __float2half(acc);
}

void conv1d_cin1_fp16(
    const __half* x, const __half* weight, const __half* bias, __half* y,
    int B, int Cout, int L, int k, int paddedL,
    cudaStream_t stream)
{
    int total = B * Cout * L;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv1d_cin1_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, bias, y, B, Cout, L, k, paddedL);
    CUDA_CHECK(cudaGetLastError());
}

// General 1D conv with dilation (encoder workhorse — replaces cuDNN)
// weight: [Cout, Cin, k], bias: [Cout], x: [B, Cin, paddedL], y: [B, Cout, outL]
__global__ void strided_conv1d_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ y,
    int B, int Cin, int Cout, int outL, int k, int stride, int dilation, int paddedL)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * Cout * outL;
    if (idx >= total) return;

    int t = idx % outL;
    int tmp = idx / outL;
    int c = tmp % Cout;
    int b = tmp / Cout;

    float acc = __half2float(bias[c]);
    const __half* w = weight + static_cast<size_t>(c) * Cin * k;
    const __half* xb = x + static_cast<size_t>(b) * Cin * paddedL + t * stride;
    for (int ci = 0; ci < Cin; ++ci) {
        for (int i = 0; i < k; ++i)
            acc += __half2float(w[ci * k + i]) * __half2float(xb[ci * paddedL + i * dilation]);
    }

    y[idx] = __float2half(acc);
}

void strided_conv1d_fp16(
    const __half* x, const __half* weight, const __half* bias, __half* y,
    int B, int Cin, int Cout, int L, int outL, int k, int stride, int dilation, int paddedL,
    cudaStream_t stream)
{
    int total = B * Cout * outL;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    strided_conv1d_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, bias, y, B, Cin, Cout, outL, k, stride, dilation, paddedL);
    CUDA_CHECK(cudaGetLastError());
}

void depthwise_conv1d_fp16(
    const __half* x,
    const __half* weight,
    const __half* bias,
    __half* y,
    int B, int C, int L, int k, int paddedL,
    cudaStream_t stream)
{
    int total = B * C * L;
    int threads = 256;
    int blocks = div_up(total, threads);
    depthwise_conv1d_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, bias, y, B, C, L, k, paddedL);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace fish::kernels
