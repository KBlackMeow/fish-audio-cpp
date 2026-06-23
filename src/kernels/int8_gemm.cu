// src/kernels/int8_gemm.cu
// W8A8 GEMM path for INT8-weight models.
//
// We keep the on-disk model format unchanged:
//   - W: INT8
//   - scale: row-wise or group-wise FP16 weight scales
//   - smooth_inv: optional per-input-channel SmoothQuant factor
//
// At runtime we dynamically quantize FP16 activations to INT8 per token/group,
// run INT8xINT8 -> INT32 GEMMs group by group, and fuse the dequant scales into
// a FP32 accumulator before casting back to FP16 output.

#include "kernels/kernels.h"
#include "utils/cuda_utils.h"
#include <spdlog/spdlog.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace fish::kernels {
namespace {

constexpr int kThreads = 256;

struct Workspace {
    int8_t* x_q = nullptr;
    float* x_scale = nullptr;
    int32_t* partial = nullptr;
    float* accum = nullptr;
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_stop = nullptr;
    size_t x_q_bytes = 0;
    size_t x_scale_bytes = 0;
    size_t partial_bytes = 0;
    size_t accum_bytes = 0;
};

thread_local Workspace g_ws;

struct ProfileBucket {
    uint64_t calls = 0;
    uint64_t groups_total = 0;
    double total_ms = 0.0;
    double quant_ms = 0.0;
    double memset_ms = 0.0;
    double gemm_ms = 0.0;
    double accum_ms = 0.0;
    double cast_ms = 0.0;
};

struct ProfileState {
    std::mutex mu;
    std::unordered_map<std::string, ProfileBucket> buckets;

    ~ProfileState() {
        if (buckets.empty()) return;
        std::lock_guard<std::mutex> lock(mu);
        spdlog::info("INT8 profile summary:");
        for (const auto& [key, bucket] : buckets) {
            double calls = static_cast<double>(bucket.calls);
            spdlog::info(
                "  {} calls={} avg_groups={:.1f} total={:.3f}ms avg={:.3f}ms "
                "(quant={:.3f} memset={:.3f} gemm={:.3f} accum={:.3f} cast={:.3f})",
                key,
                bucket.calls,
                calls > 0.0 ? static_cast<double>(bucket.groups_total) / calls : 0.0,
                bucket.total_ms,
                calls > 0.0 ? bucket.total_ms / calls : 0.0,
                calls > 0.0 ? bucket.quant_ms / calls : 0.0,
                calls > 0.0 ? bucket.memset_ms / calls : 0.0,
                calls > 0.0 ? bucket.gemm_ms / calls : 0.0,
                calls > 0.0 ? bucket.accum_ms / calls : 0.0,
                calls > 0.0 ? bucket.cast_ms / calls : 0.0);
        }
    }
};

ProfileState& profile_state() {
    static ProfileState state;
    return state;
}

bool int8_profile_enabled() {
    static bool enabled = [] {
        const char* value = std::getenv("FISH_INT8_PROFILE");
        return value && *value && std::string(value) != "0";
    }();
    return enabled;
}

void ensure_buffer(void** ptr, size_t* have, size_t need) {
    if (*have >= need) return;
    if (*ptr) CUDA_CHECK(cudaFree(*ptr));
    CUDA_CHECK(cudaMalloc(ptr, need));
    *have = need;
}

void ensure_profile_events(Workspace& ws) {
    if (ws.event_start) return;
    CUDA_CHECK(cudaEventCreate(&ws.event_start));
    CUDA_CHECK(cudaEventCreate(&ws.event_stop));
}

float measure_stream_op(cudaStream_t stream, const std::function<void()>& op) {
    ensure_profile_events(g_ws);
    CUDA_CHECK(cudaEventRecord(g_ws.event_start, stream));
    op();
    CUDA_CHECK(cudaEventRecord(g_ws.event_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(g_ws.event_stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, g_ws.event_start, g_ws.event_stop));
    return elapsed_ms;
}

std::string profile_key(int M, int N, int K, int group_size, bool static_act, bool smooth) {
    return "M=" + std::to_string(M) +
           " N=" + std::to_string(N) +
           " K=" + std::to_string(K) +
           " group=" + std::to_string(group_size) +
           " act=" + (static_act ? "static" : "dynamic") +
           " smooth=" + (smooth ? "yes" : "no");
}

void record_profile(int M, int N, int K, int group_size, int groups,
                    bool static_act, bool smooth,
                    double quant_ms, double memset_ms, double gemm_ms,
                    double accum_ms, double cast_ms) {
    auto& state = profile_state();
    std::lock_guard<std::mutex> lock(state.mu);
    auto& bucket = state.buckets[profile_key(M, N, K, group_size, static_act, smooth)];
    bucket.calls += 1;
    bucket.groups_total += static_cast<uint64_t>(groups);
    bucket.quant_ms += quant_ms;
    bucket.memset_ms += memset_ms;
    bucket.gemm_ms += gemm_ms;
    bucket.accum_ms += accum_ms;
    bucket.cast_ms += cast_ms;
    bucket.total_ms += quant_ms + memset_ms + gemm_ms + accum_ms + cast_ms;
}

__global__ void quantize_activation_groups_kernel(
    const __half* __restrict__ X,
    const __half* __restrict__ static_scale,
    const __half* __restrict__ smooth_inv,
    int8_t* __restrict__ X_q,
    float* __restrict__ X_scale,
    int N,
    int K,
    int group_size,
    int groups)
{
    int row = blockIdx.x;
    int group = blockIdx.y;
    if (row >= N || group >= groups) return;

    int start = group * group_size;
    int end = min(start + group_size, K);
    int width = end - start;
    if (width <= 0) return;

    const __half* x_row = X + static_cast<size_t>(row) * K;
    int8_t* q_row = X_q + static_cast<size_t>(row) * K;

    float scale = 0.0f;
    if (static_scale) {
        scale = __half2float(static_scale[group]);
        if (scale < 1e-8f) scale = 1e-8f;
        if (threadIdx.x == 0)
            X_scale[static_cast<size_t>(row) * groups + group] = scale;
        __syncthreads();
    } else {
        __shared__ float smax[kThreads];
        float local_max = 0.0f;
        for (int off = threadIdx.x; off < width; off += blockDim.x) {
            int k = start + off;
            float v = __half2float(x_row[k]);
            if (smooth_inv) v *= __half2float(smooth_inv[k]);
            local_max = fmaxf(local_max, fabsf(v));
        }
        smax[threadIdx.x] = local_max;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + stride]);
            __syncthreads();
        }

        scale = smax[0] / 127.0f;
        if (scale < 1e-8f) scale = 1e-8f;
        if (threadIdx.x == 0)
            X_scale[static_cast<size_t>(row) * groups + group] = scale;
        __syncthreads();
    }

    float inv_scale = 1.0f / scale;
    for (int off = threadIdx.x; off < width; off += blockDim.x) {
        int k = start + off;
        float v = __half2float(x_row[k]);
        if (smooth_inv) v *= __half2float(smooth_inv[k]);
        int q = __float2int_rn(v * inv_scale);
        q = max(-127, min(127, q));
        q_row[k] = static_cast<int8_t>(q);
    }
}

__global__ void accumulate_group_kernel(
    const int32_t* __restrict__ partial,
    const __half* __restrict__ w_scale,
    const float* __restrict__ x_scale,
    float* __restrict__ accum,
    int M,
    int N,
    int groups,
    int group)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;

    int m = idx % M;
    int n = idx / M;
    float ws = groups == 1
        ? __half2float(w_scale[m])
        : __half2float(w_scale[static_cast<size_t>(m) * groups + group]);
    float xs = x_scale[static_cast<size_t>(n) * groups + group];
    accum[idx] += static_cast<float>(partial[idx]) * ws * xs;
}

__global__ void float_to_half_kernel(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(src[idx]);
}

}  // namespace

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
    cudaStream_t stream)
{
    if (group_size <= 0 || group_size > K) group_size = K;
    int groups = (K + group_size - 1) / group_size;
    bool profile = int8_profile_enabled();
    double quant_ms = 0.0;
    double memset_ms = 0.0;
    double gemm_ms = 0.0;
    double accum_ms = 0.0;
    double cast_ms = 0.0;

    size_t x_q_bytes = static_cast<size_t>(N) * K * sizeof(int8_t);
    size_t x_scale_bytes = static_cast<size_t>(N) * groups * sizeof(float);
    size_t partial_bytes = static_cast<size_t>(M) * N * sizeof(int32_t);
    size_t accum_bytes = static_cast<size_t>(M) * N * sizeof(float);

    ensure_buffer(reinterpret_cast<void**>(&g_ws.x_q), &g_ws.x_q_bytes, x_q_bytes);
    ensure_buffer(reinterpret_cast<void**>(&g_ws.x_scale), &g_ws.x_scale_bytes, x_scale_bytes);
    ensure_buffer(reinterpret_cast<void**>(&g_ws.partial), &g_ws.partial_bytes, partial_bytes);
    ensure_buffer(reinterpret_cast<void**>(&g_ws.accum), &g_ws.accum_bytes, accum_bytes);

    dim3 quant_grid(N, groups);
    auto launch_quant = [&] {
        quantize_activation_groups_kernel<<<quant_grid, kThreads, 0, stream>>>(
            X, act_scale, smooth_inv, g_ws.x_q, g_ws.x_scale, N, K, group_size, groups);
    };
    if (profile) quant_ms = measure_stream_op(stream, launch_quant);
    else launch_quant();

    auto launch_memset = [&] {
        CUDA_CHECK(cudaMemsetAsync(g_ws.accum, 0, accum_bytes, stream));
    };
    if (profile) memset_ms = measure_stream_op(stream, launch_memset);
    else launch_memset();

    int total = M * N;
    int blocks = (total + kThreads - 1) / kThreads;
    int32_t alpha = 1;
    int32_t beta = 0;

    for (int g = 0; g < groups; ++g) {
        int start = g * group_size;
        int width = min(group_size, K - start);
        const int8_t* W_g = W + start;
        const int8_t* X_g = g_ws.x_q + start;

        auto launch_gemm = [&] {
            CUBLAS_CHECK(cublasGemmEx(
                cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                M, N, width,
                &alpha,
                W_g, CUDA_R_8I, K,
                X_g, CUDA_R_8I, K,
                &beta,
                g_ws.partial, CUDA_R_32I, M,
                CUBLAS_COMPUTE_32I,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };
        if (profile) gemm_ms += measure_stream_op(stream, launch_gemm);
        else launch_gemm();

        auto launch_accum = [&] {
            accumulate_group_kernel<<<blocks, kThreads, 0, stream>>>(
                g_ws.partial, scale, g_ws.x_scale, g_ws.accum, M, N, groups, g);
        };
        if (profile) accum_ms += measure_stream_op(stream, launch_accum);
        else launch_accum();
    }

    auto launch_cast = [&] {
        float_to_half_kernel<<<blocks, kThreads, 0, stream>>>(
            g_ws.accum, Y, total);
    };
    if (profile) cast_ms = measure_stream_op(stream, launch_cast);
    else launch_cast();

    if (profile) {
        record_profile(M, N, K, group_size, groups, act_scale != nullptr, smooth_inv != nullptr,
                       quant_ms, memset_ms, gemm_ms, accum_ms, cast_ms);
    }
}

}  // namespace fish::kernels
