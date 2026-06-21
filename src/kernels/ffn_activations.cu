// src/kernels/ffn_activations.cu
// CUDA kernels for FFN activation functions
#include "kernels/kernels.h"
#include "utils/cuda_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace fish::kernels {

// --- device kernels ---

__global__ void silu_kernel(__half* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float val = __half2float(x[i]);
    x[i] = __float2half(val / (1.0f + expf(-val)));
}

__global__ void silu_mul_clamp_kernel(__half* a, const __half* b, int n, float limit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float av = __half2float(a[i]);
    float bv = __half2float(b[i]);
    float y = (av / (1.0f + expf(-av))) * bv;
    if (!isfinite(y)) {
        y = 0.0f;
    } else if (y > limit) {
        y = limit;
    } else if (y < -limit) {
        y = -limit;
    }
    a[i] = __float2half(y);
}

__global__ void mul_kernel(__half* a, const __half* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = __float2half(__half2float(a[i]) * __half2float(b[i]));
}

__global__ void residual_add_kernel(__half* a, const __half* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

__global__ void gelu_kernel(__half* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(x[i]);
    // Match PyTorch nn.GELU() default, approximate="none":
    // 0.5 * x * (1 + erf(x / sqrt(2))).
    float y = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
    x[i] = __float2half(y);
}

__global__ void add_channel_bias_kernel(__half* x, const __half* bias, int total, int C, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = (idx / T) % C;
    x[idx] = __float2half(__half2float(x[idx]) + __half2float(bias[c]));
}

__global__ void channel_scale_residual_kernel(__half* x, const __half* y, const __half* gamma,
                                              int total, int C, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = (idx / T) % C;
    float v = __half2float(x[idx]) + __half2float(y[idx]) * __half2float(gamma[c]);
    x[idx] = __float2half(v);
}

__global__ void dim_scale_residual_kernel(__half* x, const __half* y, const __half* gamma,
                                          int total, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d = idx % dim;
    float v = __half2float(x[idx]) + __half2float(y[idx]) * __half2float(gamma[d]);
    x[idx] = __float2half(v);
}

__global__ void dim_scale_residual_from_f32_kernel(__half* x, const float* y, const __half* gamma,
                                                   int total, int dim, float limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d = idx % dim;
    float v = __half2float(x[idx]) + y[idx] * __half2float(gamma[d]);
    if (isnan(v)) {
        v = 0.0f;
    } else if (v > limit) {
        v = limit;
    } else if (v < -limit) {
        v = -limit;
    }
    x[idx] = __float2half(v);
}

__global__ void layer_norm_channels_kernel(const __half* x, const __half* weight, const __half* bias,
                                           __half* out, int rows, int C, int T, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int b = row / T;
    int t = row % T;

    float mean = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        mean += __half2float(x[(b * C + c) * T + t]);
    __shared__ float smem[256];
    smem[threadIdx.x] = mean;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    mean = smem[0] / C;

    float var = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float d = __half2float(x[(b * C + c) * T + t]) - mean;
        var += d * d;
    }
    smem[threadIdx.x] = var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(smem[0] / C + eps);
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        int idx = (b * C + c) * T + t;
        float y = (__half2float(x[idx]) - mean) * inv;
        y = y * __half2float(weight[c]) + __half2float(bias[c]);
        out[idx] = __float2half(y);
    }
}

// --- host wrapper functions ---

void silu_forward(__half* x, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_kernel<<<blocks, threads, 0, stream>>>(x, n);
}

void silu_mul_clamp_forward(__half* a, const __half* b, int n, float limit, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_mul_clamp_kernel<<<blocks, threads, 0, stream>>>(a, b, n, limit);
}

void mul_forward(__half* a, const __half* b, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mul_kernel<<<blocks, threads, 0, stream>>>(a, b, n);
}

void residual_add(__half* a, const __half* b, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads, 0, stream>>>(a, b, n);
}

void gelu_forward(__half* x, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_kernel<<<blocks, threads, 0, stream>>>(x, n);
    CUDA_CHECK(cudaGetLastError());
}

void add_channel_bias(__half* x, const __half* bias, int B, int C, int T, cudaStream_t stream) {
    int total = B * C * T;
    constexpr int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_channel_bias_kernel<<<blocks, threads, 0, stream>>>(x, bias, total, C, T);
    CUDA_CHECK(cudaGetLastError());
}

void channel_scale_residual(__half* x, const __half* y, const __half* gamma,
                            int B, int C, int T, cudaStream_t stream) {
    int total = B * C * T;
    constexpr int threads = 256;
    int blocks = (total + threads - 1) / threads;
    channel_scale_residual_kernel<<<blocks, threads, 0, stream>>>(x, y, gamma, total, C, T);
    CUDA_CHECK(cudaGetLastError());
}

void layer_norm_channels_fp16(const __half* x, const __half* weight, const __half* bias,
                              __half* out, int B, int C, int T, float eps,
                              cudaStream_t stream) {
    int rows = B * T;
    layer_norm_channels_kernel<<<rows, 256, 0, stream>>>(x, weight, bias, out, rows, C, T, eps);
    CUDA_CHECK(cudaGetLastError());
}

void dim_scale_residual(__half* x, const __half* y, const __half* gamma,
                        int rows, int dim, cudaStream_t stream) {
    int total = rows * dim;
    constexpr int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dim_scale_residual_kernel<<<blocks, threads, 0, stream>>>(x, y, gamma, total, dim);
    CUDA_CHECK(cudaGetLastError());
}

void dim_scale_residual_from_f32(__half* x, const float* y, const __half* gamma,
                                 int rows, int dim, float limit, cudaStream_t stream) {
    int total = rows * dim;
    constexpr int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dim_scale_residual_from_f32_kernel<<<blocks, threads, 0, stream>>>(x, y, gamma, total, dim, limit);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void transpose_fp16_kernel(const __half* in, __half* out,
                                      int rows, int cols, int ld_in, int ld_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int i = idx / cols, j = idx % cols;
    // Input row-major in[row*ld_in + col]  →  Output row-major out[col*ld_out + row]
    out[j * ld_out + i] = in[i * ld_in + j];
}

void transpose_fp16(const __half* in, __half* out, int rows, int cols, int ld_in, int ld_out, cudaStream_t stream) {
    int th = 256;
    transpose_fp16_kernel<<<(rows*cols+th-1)/th, th, 0, stream>>>(in, out, rows, cols, ld_in, ld_out);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void scale_inplace_kernel(__half* x, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = __float2half(__half2float(x[i]) * scale);
}

void scale_inplace(__half* x, int n, float scale, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scale_inplace_kernel<<<blocks, threads, 0, stream>>>(x, n, scale);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void clamp_finite_inplace_kernel(__half* x, int n, float limit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(x[i]);
    if (isinf(v)) {
        v = v > 0.0f ? limit : -limit;
    } else if (isnan(v)) {
        v = 0.0f;
    } else if (v > limit) {
        v = limit;
    } else if (v < -limit) {
        v = -limit;
    }
    x[i] = __float2half(v);
}

void clamp_finite_inplace(__half* x, int n, float limit, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;
    clamp_finite_inplace_kernel<<<blocks, threads, 0, stream>>>(x, n, limit);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace fish::kernels
