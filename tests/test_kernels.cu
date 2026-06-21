// tests/test_kernels.cu — unit tests for all GPU kernels
#include "kernels/kernels.h"
#include "utils/cuda_utils.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <random>
#include <cstring>

namespace fish::kernels {
namespace {

// ── Helpers ──────────────────────────────────────────────────────────────────
static float to_float(__half h) { return __half2float(h); }
static __half to_half(float v)  { return __float2half(v); }

class KernelTest : public ::testing::Test {
protected:
    void SetUp() override    { CUDA_CHECK(cudaStreamCreate(&stream_)); }
    void TearDown() override { CUDA_CHECK(cudaStreamDestroy(stream_)); }
    cudaStream_t stream_{};
};

static std::vector<float> prefill_attention_ref(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    int H, int T, int D, float scale, int window)
{
    std::vector<float> out(static_cast<size_t>(H) * T * D, 0.0f);
    std::vector<float> scores(T);
    for (int h = 0; h < H; ++h) {
        for (int i = 0; i < T; ++i) {
            float mx = -1e30f;
            int left = window > 0 ? std::max(0, i - window + 1) : 0;
            for (int j = 0; j < T; ++j) {
                float s = -1e30f;
                if (j <= i && j >= left) {
                    s = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        size_t qi = (static_cast<size_t>(h) * T + i) * D + d;
                        size_t kj = (static_cast<size_t>(h) * T + j) * D + d;
                        s += q[qi] * k[kj];
                    }
                    s *= scale;
                }
                scores[j] = s;
                mx = std::max(mx, s);
            }
            float denom = 0.0f;
            for (int j = 0; j < T; ++j) {
                scores[j] = std::exp(scores[j] - mx);
                denom += scores[j];
            }
            for (int d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (int j = 0; j < T; ++j) {
                    size_t vj = (static_cast<size_t>(h) * T + j) * D + d;
                    acc += (scores[j] / denom) * v[vj];
                }
                out[(static_cast<size_t>(h) * T + i) * D + d] = acc;
            }
        }
    }
    return out;
}

TEST_F(KernelTest, PrefillAttentionMatchesCausalWindowReference) {
    constexpr int H = 2, T = 4, D = 8;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const int window = 3;
    const size_t n = static_cast<size_t>(H) * T * D;

    std::vector<float> qf(n), kf(n), vf(n);
    for (size_t i = 0; i < n; ++i) {
        qf[i] = std::sin(static_cast<float>(i) * 0.13f) * 0.7f;
        kf[i] = std::cos(static_cast<float>(i) * 0.11f) * 0.6f;
        vf[i] = std::sin(static_cast<float>(i) * 0.17f) * 0.5f;
    }
    std::vector<__half> qh(n), kh(n), vh(n);
    for (size_t i = 0; i < n; ++i) {
        qh[i] = to_half(qf[i]);
        kh[i] = to_half(kf[i]);
        vh[i] = to_half(vf[i]);
    }

    __half *dq, *dk, *dv, *dout, *dws;
    CUDA_CHECK(cudaMalloc(&dq, n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dk, n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dv, n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dout, n * sizeof(__half)));
    size_t ws_elems = static_cast<size_t>(H) * T * T + 2ULL * H * T * D;
    CUDA_CHECK(cudaMalloc(&dws, ws_elems * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dq, qh.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk, kh.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, vh.data(), n * sizeof(__half), cudaMemcpyHostToDevice));

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetStream(cublas, stream_));
    prefill_attention_gpu(dq, dk, dv, dout, dws, H, H, T, D, scale, cublas, stream_, window);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> hout(n);
    CUDA_CHECK(cudaMemcpy(hout.data(), dout, n * sizeof(__half), cudaMemcpyDeviceToHost));
    auto ref = prefill_attention_ref(qf, kf, vf, H, T, D, scale, window);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(to_float(hout[i]), ref[i], 2e-3f) << "idx=" << i;
    }

    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dk)); CUDA_CHECK(cudaFree(dv));
    CUDA_CHECK(cudaFree(dout)); CUDA_CHECK(cudaFree(dws));
}

// ── RMSNorm ─────────────────────────────────────────────────────────────────
TEST_F(KernelTest, RmsNormBasic) {
    constexpr int N = 4, D = 8;
    float eps = 1e-6f;

    std::vector<__half> h_x(N * D);
    std::vector<__half> h_w(D);
    for (int i = 0; i < D; i++) h_w[i] = to_half(1.0f);
    for (int i = 0; i < N * D; i++) h_x[i] = to_half(1.0f);

    __half *d_x, *d_w, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x,   N * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_w,   D     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, N * D * sizeof(__half)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), D     * sizeof(__half), cudaMemcpyHostToDevice));

    rms_norm(d_x, d_w, d_out, N, D, eps, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(N * D);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * D * sizeof(__half), cudaMemcpyDeviceToHost));

    // x=1, w=1, rms=1 → output ≈ 1
    for (int i = 0; i < N * D; i++)
        EXPECT_NEAR(to_float(h_out[i]), 1.0f, 1e-3f);

    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_out));
}

TEST_F(KernelTest, RmsNormZeroInput) {
    constexpr int N = 2, D = 4;
    float eps = 1e-5f;

    std::vector<__half> h_x(N * D, to_half(0.0f));
    std::vector<__half> h_w(D, to_half(1.0f));

    __half *d_x, *d_w, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x,   N * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_w,   D     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, N * D * sizeof(__half)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), D     * sizeof(__half), cudaMemcpyHostToDevice));

    rms_norm(d_x, d_w, d_out, N, D, eps, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(N * D);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * D * sizeof(__half), cudaMemcpyDeviceToHost));

    // Zero input → output should be finite (divided by sqrt(eps))
    for (int i = 0; i < N * D; i++) {
        EXPECT_FALSE(std::isnan(to_float(h_out[i])));
        EXPECT_FALSE(std::isinf(to_float(h_out[i])));
    }

    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_out));
}

TEST_F(KernelTest, RmsNormScale) {
    // If weights are 2.0, output = rms_norm(x) * 2
    constexpr int N = 1, D = 4;
    float eps = 1e-6f;

    std::vector<__half> h_x = {to_half(1.f), to_half(3.f), to_half(1.f), to_half(3.f)};
    std::vector<__half> h_w(D, to_half(2.0f));

    __half *d_x, *d_w, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x,   N * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_w,   D     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, N * D * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), D     * sizeof(__half), cudaMemcpyHostToDevice));

    rms_norm(d_x, d_w, d_out, N, D, eps, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(N * D);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * D * sizeof(__half), cudaMemcpyDeviceToHost));

    // Manual RMSNorm: rms = sqrt(mean(x^2) + eps)
    float rms = std::sqrt((1.f + 9.f + 1.f + 9.f) / 4.f + eps);  // sqrt(5 + eps)
    for (int i = 0; i < D; i++) {
        float expected = (to_float(h_x[i]) / rms) * 2.0f;
        EXPECT_NEAR(to_float(h_out[i]), expected, 1e-2f);
    }

    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_out));
}

TEST_F(KernelTest, RmsNormMultiToken) {
    constexpr int N = 3, D = 16;
    float eps = 1e-6f;

    std::vector<__half> h_x(N * D);
    std::vector<__half> h_w(D);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < D; i++) h_w[i] = to_half(1.0f);
    for (int i = 0; i < N * D; i++) h_x[i] = to_half(dist(gen));

    __half *d_x, *d_w, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x,   N * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_w,   D     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, N * D * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), D     * sizeof(__half), cudaMemcpyHostToDevice));

    rms_norm(d_x, d_w, d_out, N, D, eps, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(N * D);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * D * sizeof(__half), cudaMemcpyDeviceToHost));

    // Verify per-token normalization
    for (int n = 0; n < N; n++) {
        float sum_sq = 0.f;
        for (int d = 0; d < D; d++)
            sum_sq += to_float(h_out[n * D + d]) * to_float(h_out[n * D + d]);
        float rms_out = std::sqrt(sum_sq / D);
        // RMS of output ≈ RMS of weight = sqrt(mean(1^2)) = 1.0
        EXPECT_NEAR(rms_out, 1.0f, 1e-3f);
    }

    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_out));
}

// ── RoPE ────────────────────────────────────────────────────────────────────
TEST_F(KernelTest, RopeBasic) {
    constexpr int B = 1, Hq = 2, Hk = 1, T = 2, D = 8;
    float inv_freq_base = 10000.0f;

    std::vector<float> h_freqs(D / 2);
    for (int i = 0; i < D / 2; i++)
        h_freqs[i] = 1.0f / std::pow(inv_freq_base, (2.0f * i) / D);

    size_t q_elems = B * Hq * T * D;
    size_t k_elems = B * Hk * T * D;

    std::vector<__half> h_q(q_elems, to_half(0.0f));
    std::vector<__half> h_k(k_elems, to_half(0.0f));

    // Set q = k = all 1s (so we can verify rotation)
    for (auto& v : h_q) v = to_half(1.0f);
    for (auto& v : h_k) v = to_half(1.0f);

    float *d_freqs;
    __half *d_q, *d_k;
    CUDA_CHECK(cudaMalloc(&d_freqs, (D / 2) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q, q_elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_k, k_elems * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_freqs, h_freqs.data(), (D / 2) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), k_elems * sizeof(__half), cudaMemcpyHostToDevice));

    rope_qk(d_q, d_k, d_freqs, B, Hq, Hk, T, D, 0, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_q_out(q_elems), h_k_out(k_elems);
    CUDA_CHECK(cudaMemcpy(h_q_out.data(), d_q, q_elems * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_k_out.data(), d_k, k_elems * sizeof(__half), cudaMemcpyDeviceToHost));

    // RoPE rotates pairs: (x0, x1) → (x0*cos - x1*sin, x0*sin + x1*cos)
    // For [1,1] pair at position 0: cos(1)=0.54, sin(0)=0
    // Result should be cos(0)*1 - sin(0)*1 = 1, sin(0)*1 + cos(0)*1 = 1
    // So all values should remain non-zero and finite
    for (size_t i = 0; i < q_elems; i++) {
        EXPECT_FALSE(std::isnan(to_float(h_q_out[i])));
        EXPECT_FALSE(std::isinf(to_float(h_q_out[i])));
    }
    for (size_t i = 0; i < k_elems; i++) {
        EXPECT_FALSE(std::isnan(to_float(h_k_out[i])));
        EXPECT_FALSE(std::isinf(to_float(h_k_out[i])));
    }

    CUDA_CHECK(cudaFree(d_freqs)); CUDA_CHECK(cudaFree(d_q)); CUDA_CHECK(cudaFree(d_k));
}

TEST_F(KernelTest, RopePositionOffset) {
    constexpr int B = 1, Hq = 1, Hk = 1, T = 1, D = 8;
    float inv_freq_base = 10000.0f;

    std::vector<float> h_freqs(D / 2);
    for (int i = 0; i < D / 2; i++)
        h_freqs[i] = 1.0f / std::pow(inv_freq_base, (2.0f * i) / D);

    size_t elems = B * Hq * T * D;

    // Create two copies: one for pos=0, one for pos=3
    std::vector<__half> h_q0(elems, to_half(1.0f));
    std::vector<__half> h_q3(elems, to_half(1.0f));
    std::vector<__half> h_k0(elems, to_half(0.0f));
    std::vector<__half> h_k3(elems, to_half(0.0f));

    float *d_freqs;
    __half *d_q0, *d_k0, *d_q3, *d_k3;
    CUDA_CHECK(cudaMalloc(&d_freqs, (D / 2) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q0, elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_k0, elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_q3, elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_k3, elems * sizeof(__half)));

    CUDA_CHECK(cudaMemcpy(d_freqs, h_freqs.data(), (D / 2) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q0, h_q0.data(), elems * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k0, h_k0.data(), elems * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q3, h_q3.data(), elems * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k3, h_k3.data(), elems * sizeof(__half), cudaMemcpyHostToDevice));

    rope_qk(d_q0, d_k0, d_freqs, B, Hq, Hk, T, D, 0, stream_);
    rope_qk(d_q3, d_k3, d_freqs, B, Hq, Hk, T, D, 3, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_q0_out(elems), h_q3_out(elems);
    CUDA_CHECK(cudaMemcpy(h_q0_out.data(), d_q0, elems * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_q3_out.data(), d_q3, elems * sizeof(__half), cudaMemcpyDeviceToHost));

    // Different positions should produce different rotations
    bool any_diff = false;
    for (size_t i = 0; i < elems; i++) {
        if (std::abs(to_float(h_q0_out[i]) - to_float(h_q3_out[i])) > 1e-4f) {
            any_diff = true;
            break;
        }
    }
    EXPECT_TRUE(any_diff);

    CUDA_CHECK(cudaFree(d_freqs));
    CUDA_CHECK(cudaFree(d_q0)); CUDA_CHECK(cudaFree(d_k0));
    CUDA_CHECK(cudaFree(d_q3)); CUDA_CHECK(cudaFree(d_k3));
}

// ── SiLU ────────────────────────────────────────────────────────────────────
TEST_F(KernelTest, SiluBasic) {
    constexpr int N = 16;
    std::vector<__half> h_x(N);
    for (int i = 0; i < N; i++) h_x[i] = to_half(static_cast<float>(i - 8));  // -8..7

    __half* d_x;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(__half), cudaMemcpyHostToDevice));

    silu_forward(d_x, N, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_x, N * sizeof(__half), cudaMemcpyDeviceToHost));

    // SiLU(x) = x * sigmoid(x)
    for (int i = 0; i < N; i++) {
        float x = static_cast<float>(i - 8);
        float expected = x * (1.0f / (1.0f + std::exp(-x)));
        EXPECT_NEAR(to_float(h_out[i]), expected, 1e-2f);
    }

    CUDA_CHECK(cudaFree(d_x));
}

TEST_F(KernelTest, SiluLarge) {
    constexpr int N = 1024;
    std::vector<__half> h_x(N);
    std::mt19937 gen(123);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    for (int i = 0; i < N; i++) h_x[i] = to_half(dist(gen));

    __half* d_x;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(__half), cudaMemcpyHostToDevice));

    silu_forward(d_x, N, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_x, N * sizeof(__half), cudaMemcpyDeviceToHost));

    // All values should be finite
    for (int i = 0; i < N; i++) {
        EXPECT_FALSE(std::isnan(to_float(h_out[i])));
        EXPECT_FALSE(std::isinf(to_float(h_out[i])));
    }

    CUDA_CHECK(cudaFree(d_x));
}

// ── GELU ────────────────────────────────────────────────────────────────────
TEST_F(KernelTest, GeluMatchesPytorchDefaultExact) {
    std::vector<float> h_vals = {-4.0f, -1.5f, -0.5f, 0.0f, 0.5f, 1.5f, 4.0f};
    const int N = static_cast<int>(h_vals.size());
    std::vector<__half> h_x(N);
    for (int i = 0; i < N; ++i) h_x[i] = to_half(h_vals[i]);

    __half* d_x;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(__half), cudaMemcpyHostToDevice));

    gelu_forward(d_x, N, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_x, N * sizeof(__half), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        float x = h_vals[i];
        float expected = 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
        EXPECT_NEAR(to_float(h_out[i]), expected, 2e-3f);
    }

    CUDA_CHECK(cudaFree(d_x));
}

// ── Element-wise multiply ───────────────────────────────────────────────────
TEST_F(KernelTest, MulForward) {
    constexpr int N = 8;
    std::vector<__half> h_a = {to_half(1.f),to_half(2.f),to_half(3.f),to_half(4.f),
                               to_half(5.f),to_half(6.f),to_half(7.f),to_half(8.f)};
    std::vector<__half> h_b = {to_half(2.f),to_half(3.f),to_half(4.f),to_half(5.f),
                               to_half(1.f),to_half(1.f),to_half(1.f),to_half(1.f)};

    __half *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(__half), cudaMemcpyHostToDevice));

    mul_forward(d_a, d_b, N, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_a, N * sizeof(__half), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++)
        EXPECT_NEAR(to_float(h_out[i]), to_float(h_a[i]) * to_float(h_b[i]), 1e-2f);

    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b));
}

// ── Residual add ────────────────────────────────────────────────────────────
TEST_F(KernelTest, ResidualAdd) {
    constexpr int N = 8;
    std::vector<__half> h_a = {to_half(1.f),to_half(2.f),to_half(3.f),to_half(4.f),
                               to_half(5.f),to_half(6.f),to_half(7.f),to_half(8.f)};
    std::vector<__half> h_b = {to_half(10.f),to_half(20.f),to_half(30.f),to_half(40.f),
                               to_half(0.f),to_half(-1.f),to_half(-2.f),to_half(-3.f)};

    __half *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(__half), cudaMemcpyHostToDevice));

    residual_add(d_a, d_b, N, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_a, N * sizeof(__half), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++)
        EXPECT_NEAR(to_float(h_out[i]), to_float(h_a[i]) + to_float(h_b[i]), 1e-1f);

    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b));
}

// ── Scale inplace ───────────────────────────────────────────────────────────
TEST_F(KernelTest, ScaleInplace) {
    constexpr int N = 8;
    float scale = 0.5f;
    std::vector<__half> h_x(N, to_half(2.0f));

    __half* d_x;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(__half), cudaMemcpyHostToDevice));

    scale_inplace(d_x, N, scale, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_x, N * sizeof(__half), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++)
        EXPECT_NEAR(to_float(h_out[i]), 1.0f, 1e-3f);

    CUDA_CHECK(cudaFree(d_x));
}

TEST_F(KernelTest, ClampFiniteInplaceSaturatesInfAndClearsNan) {
    constexpr int N = 5;
    constexpr float limit = 100.0f;
    std::vector<__half> h_x = {
        to_half(-150.0f),
        to_half(-INFINITY),
        to_half(NAN),
        to_half(INFINITY),
        to_half(150.0f),
    };

    __half* d_x;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(__half), cudaMemcpyHostToDevice));

    clamp_finite_inplace(d_x, N, limit, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_x, N * sizeof(__half), cudaMemcpyDeviceToHost));

    EXPECT_NEAR(to_float(h_out[0]), -limit, 1e-3f);
    EXPECT_NEAR(to_float(h_out[1]), -limit, 1e-3f);
    EXPECT_NEAR(to_float(h_out[2]), 0.0f, 1e-3f);
    EXPECT_NEAR(to_float(h_out[3]), limit, 1e-3f);
    EXPECT_NEAR(to_float(h_out[4]), limit, 1e-3f);

    CUDA_CHECK(cudaFree(d_x));
}

TEST_F(KernelTest, LayerNormChannelsMatchesTorchLayout) {
    constexpr int B = 2, C = 3, T = 2;
    constexpr float eps = 1e-6f;
    std::vector<float> vals = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        -1.0f, 0.5f,
        2.0f, -3.0f,
        4.0f, 1.0f,
    };
    std::vector<__half> h_x(B * C * T);
    for (int i = 0; i < B * C * T; ++i) h_x[i] = to_half(vals[i]);
    std::vector<__half> h_w = {to_half(1.0f), to_half(1.5f), to_half(-0.5f)};
    std::vector<__half> h_b = {to_half(0.0f), to_half(0.25f), to_half(-1.0f)};

    __half *d_x, *d_w, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, B * C * T * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_w, C * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b, C * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, B * C * T * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), B * C * T * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), C * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), C * sizeof(__half), cudaMemcpyHostToDevice));

    layer_norm_channels_fp16(d_x, d_w, d_b, d_out, B, C, T, eps, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(B * C * T);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, B * C * T * sizeof(__half), cudaMemcpyDeviceToHost));

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            float mean = 0.0f;
            for (int c = 0; c < C; ++c) mean += vals[(b * C + c) * T + t];
            mean /= C;
            float var = 0.0f;
            for (int c = 0; c < C; ++c) {
                float d = vals[(b * C + c) * T + t] - mean;
                var += d * d;
            }
            float inv = 1.0f / std::sqrt(var / C + eps);
            for (int c = 0; c < C; ++c) {
                int idx = (b * C + c) * T + t;
                float expected = (vals[idx] - mean) * inv * to_float(h_w[c]) + to_float(h_b[c]);
                EXPECT_NEAR(to_float(h_out[idx]), expected, 2e-3f);
            }
        }
    }

    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_out));
}

// ── Embedding lookup ────────────────────────────────────────────────────────
TEST_F(KernelTest, EmbeddingLookup) {
    constexpr int n_tokens = 3, dim = 4, vocab = 5;
    // Build a small embedding table [vocab, dim]
    std::vector<__half> h_table(vocab * dim);
    for (int v = 0; v < vocab; v++)
        for (int d = 0; d < dim; d++)
            h_table[v * dim + d] = to_half(static_cast<float>(v * 100 + d));

    std::vector<int32_t> h_indices = {2, 0, 4};

    int32_t *d_indices;
    __half *d_table, *d_out;
    CUDA_CHECK(cudaMalloc(&d_indices, n_tokens * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_table,  vocab * dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out,    n_tokens * dim * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), vocab * dim * sizeof(__half), cudaMemcpyHostToDevice));

    embedding_lookup(d_indices, d_table, d_out, n_tokens, dim, vocab, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(n_tokens * dim);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n_tokens * dim * sizeof(__half), cudaMemcpyDeviceToHost));

    // Token 2 → row 2 of table
    for (int t = 0; t < n_tokens; t++) {
        int token = h_indices[t];
        for (int d = 0; d < dim; d++) {
            float expected = static_cast<float>(token * 100 + d);
            EXPECT_NEAR(to_float(h_out[t * dim + d]), expected, 1e-2f);
        }
    }

    CUDA_CHECK(cudaFree(d_indices)); CUDA_CHECK(cudaFree(d_table)); CUDA_CHECK(cudaFree(d_out));
}

TEST_F(KernelTest, EmbeddingLookupOutOfRange) {
    // Out-of-range indices should be clamped to 0
    constexpr int n_tokens = 2, dim = 4, vocab = 5;
    std::vector<__half> h_table(vocab * dim);
    for (int v = 0; v < vocab; v++)
        for (int d = 0; d < dim; d++)
            h_table[v * dim + d] = to_half(static_cast<float>(v * 10 + d));

    std::vector<int32_t> h_indices = {999, -5};  // both out of range

    int32_t *d_indices;
    __half *d_table, *d_out;
    CUDA_CHECK(cudaMalloc(&d_indices, n_tokens * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_table,  vocab * dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out,    n_tokens * dim * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), vocab * dim * sizeof(__half), cudaMemcpyHostToDevice));

    embedding_lookup(d_indices, d_table, d_out, n_tokens, dim, vocab, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(n_tokens * dim);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n_tokens * dim * sizeof(__half), cudaMemcpyDeviceToHost));

    // Should get row 0 (token 0 clamped)
    for (int d = 0; d < dim; d++) {
        EXPECT_NEAR(to_float(h_out[0 * dim + d]), static_cast<float>(d), 1e-2f);
        EXPECT_NEAR(to_float(h_out[1 * dim + d]), static_cast<float>(d), 1e-2f);
    }

    CUDA_CHECK(cudaFree(d_indices)); CUDA_CHECK(cudaFree(d_table)); CUDA_CHECK(cudaFree(d_out));
}

// ── Snake activation ────────────────────────────────────────────────────────
TEST_F(KernelTest, SnakeBasic) {
    constexpr int N = 16;
    float alpha = 1.0f;
    std::vector<__half> h_x(N);
    for (int i = 0; i < N; i++) h_x[i] = to_half(0.1f * (i + 1));

    __half *d_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x,   N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(__half), cudaMemcpyHostToDevice));

    snake_forward(d_x, d_out, N, alpha, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(__half), cudaMemcpyDeviceToHost));

    // snake(x, a) = x + (1/a) * sin^2(a * x)
    for (int i = 0; i < N; i++) {
        float x = 0.1f * (i + 1);
        float expected = x + (1.0f / alpha) * std::sin(alpha * x) * std::sin(alpha * x);
        EXPECT_NEAR(to_float(h_out[i]), expected, 1e-2f);
    }

    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_out));
}

TEST_F(KernelTest, SnakeChannelsUsesPerChannelAlpha) {
    constexpr int B = 1, C = 3, L = 4;
    std::vector<float> vals = {
        -0.5f, 0.0f, 0.5f, 1.0f,
        -1.0f, 0.25f, 0.75f, 1.25f,
        -0.25f, 0.5f, 1.0f, 1.5f,
    };
    std::vector<__half> h_x(B * C * L);
    for (int i = 0; i < B * C * L; ++i) h_x[i] = to_half(vals[i]);
    std::vector<__half> h_alpha = {to_half(0.5f), to_half(1.0f), to_half(2.0f)};

    __half *d_x, *d_alpha, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, B * C * L * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_alpha, C * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, B * C * L * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), B * C * L * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_alpha, h_alpha.data(), C * sizeof(__half), cudaMemcpyHostToDevice));

    snake_forward_channels(d_x, d_out, d_alpha, B, C, L, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(B * C * L);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, B * C * L * sizeof(__half), cudaMemcpyDeviceToHost));

    for (int i = 0; i < B * C * L; ++i) {
        int c = (i / L) % C;
        float a = to_float(h_alpha[c]);
        float expected = vals[i] + std::sin(a * vals[i]) * std::sin(a * vals[i]) / a;
        EXPECT_NEAR(to_float(h_out[i]), expected, 2e-3f);
    }

    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_alpha)); CUDA_CHECK(cudaFree(d_out));
}

// ── Float32 to Float16 cast ─────────────────────────────────────────────────
TEST_F(KernelTest, F32ToF16Cast) {
    constexpr int N = 8;
    std::vector<float> h_src = {0.f, 1.f, -1.f, 0.5f, 2.f, -3.5f, 100.f, -0.125f};

    float *d_src;
    __half *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    f32_to_f16_cast(d_src, d_dst, N, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_dst, N * sizeof(__half), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++)
        EXPECT_NEAR(to_float(h_out[i]), h_src[i], 1e-2f);

    CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_dst));
}

// ── Transpose FP16 ──────────────────────────────────────────────────────────
TEST_F(KernelTest, TransposeFp16) {
    constexpr int rows = 4, cols = 3;
    std::vector<__half> h_in(rows * cols);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            h_in[r * cols + c] = to_half(static_cast<float>(r * 10 + c));

    __half *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  rows * cols * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, rows * cols * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), rows * cols * sizeof(__half), cudaMemcpyHostToDevice));

    transpose_fp16(d_in, d_out, rows, cols, cols, rows, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(rows * cols);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, rows * cols * sizeof(__half), cudaMemcpyDeviceToHost));

    // Output is [cols, rows], so h_out[c * ld_out + r] = h_in[r * ld_in + c]
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            EXPECT_NEAR(to_float(h_out[c * rows + r]), to_float(h_in[r * cols + c]), 1e-2f);

    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
}

TEST_F(KernelTest, LeftPadTimeFp16PreservesRowsAndZerosPadding) {
    constexpr int B = 2, C = 2, L = 3, left = 2, right = 1;
    constexpr int outL = L + left + right;
    std::vector<__half> h_x(B * C * L);
    for (int i = 0; i < B * C * L; ++i) h_x[i] = to_half(static_cast<float>(i + 1));

    __half *d_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, B * C * L * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, B * C * outL * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), B * C * L * sizeof(__half), cudaMemcpyHostToDevice));

    left_pad_time_fp16(d_x, d_out, B, C, L, left, right, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(B * C * outL);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, B * C * outL * sizeof(__half), cudaMemcpyDeviceToHost));

    for (int row = 0; row < B * C; ++row) {
        for (int t = 0; t < outL; ++t) {
            float expected = 0.0f;
            if (t >= left && t < left + L) expected = to_float(h_x[row * L + (t - left)]);
            EXPECT_NEAR(to_float(h_out[row * outL + t]), expected, 1e-3f);
        }
    }

    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_out));
}

TEST_F(KernelTest, Conv1dCin1MatchesReference) {
    constexpr int B = 1, Cout = 2, L = 4, k = 3, paddedL = 6;
    std::vector<__half> h_x = {
        to_half(0.0f), to_half(0.0f), to_half(1.0f),
        to_half(2.0f), to_half(3.0f), to_half(4.0f),
    };
    std::vector<__half> h_w = {
        to_half(1.0f), to_half(0.0f), to_half(-1.0f),
        to_half(0.5f), to_half(1.0f), to_half(0.5f),
    };
    std::vector<__half> h_b = {to_half(0.25f), to_half(-0.5f)};

    __half *d_x, *d_w, *d_b, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, B * paddedL * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_w, Cout * k * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b, Cout * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_y, B * Cout * L * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), B * paddedL * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), Cout * k * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), Cout * sizeof(__half), cudaMemcpyHostToDevice));

    conv1d_cin1_fp16(d_x, d_w, d_b, d_y, B, Cout, L, k, paddedL, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_y(B * Cout * L);
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, B * Cout * L * sizeof(__half), cudaMemcpyDeviceToHost));

    for (int c = 0; c < Cout; ++c) {
        for (int t = 0; t < L; ++t) {
            float expected = to_float(h_b[c]);
            for (int i = 0; i < k; ++i)
                expected += to_float(h_w[c * k + i]) * to_float(h_x[t + i]);
            EXPECT_NEAR(to_float(h_y[c * L + t]), expected, 2e-3f);
        }
    }

    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_y));
}

TEST_F(KernelTest, StridedConv1dMatchesReferenceWithDilation) {
    constexpr int B = 1, Cin = 2, Cout = 2, L = 3, outL = 3, k = 2, stride = 2, dilation = 2, paddedL = 7;
    std::vector<__half> h_x(B * Cin * paddedL);
    for (int i = 0; i < static_cast<int>(h_x.size()); ++i)
        h_x[i] = to_half(static_cast<float>(i - 3) * 0.25f);
    std::vector<__half> h_w = {
        to_half(1.0f), to_half(-0.5f),
        to_half(0.25f), to_half(0.75f),
        to_half(-1.0f), to_half(0.5f),
        to_half(0.5f), to_half(1.0f),
    };
    std::vector<__half> h_b = {to_half(0.1f), to_half(-0.2f)};

    __half *d_x, *d_w, *d_b, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, B * Cin * paddedL * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_w, Cout * Cin * k * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b, Cout * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_y, B * Cout * outL * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), B * Cin * paddedL * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), Cout * Cin * k * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), Cout * sizeof(__half), cudaMemcpyHostToDevice));

    strided_conv1d_fp16(d_x, d_w, d_b, d_y, B, Cin, Cout, L, outL, k, stride, dilation, paddedL, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_y(B * Cout * outL);
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, B * Cout * outL * sizeof(__half), cudaMemcpyDeviceToHost));

    for (int c = 0; c < Cout; ++c) {
        for (int t = 0; t < outL; ++t) {
            float expected = to_float(h_b[c]);
            for (int ci = 0; ci < Cin; ++ci) {
                for (int i = 0; i < k; ++i) {
                    int x_idx = ci * paddedL + t * stride + i * dilation;
                    int w_idx = (c * Cin + ci) * k + i;
                    expected += to_float(h_w[w_idx]) * to_float(h_x[x_idx]);
                }
            }
            EXPECT_NEAR(to_float(h_y[c * outL + t]), expected, 2e-3f);
        }
    }

    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_y));
}

// ── RVQ lookup decode ───────────────────────────────────────────────────────
TEST_F(KernelTest, RvqLookupDecode) {
    constexpr int B = 1, N = 2, T = 2, cb_size = 4, dim = 3;
    // codes [B, N, T] = [1, 2, 2]
    std::vector<int32_t> h_codes = {0, 1, 2, 3};  // cb0: [0,1]; cb1: [2,3]
    // embeddings [N, cb_size, dim] = [2, 4, 3]
    std::vector<__half> h_emb(N * cb_size * dim);
    for (int n = 0; n < N; n++)
        for (int c = 0; c < cb_size; c++)
            for (int d = 0; d < dim; d++)
                h_emb[(n * cb_size + c) * dim + d] =
                    to_half(static_cast<float>(n * 100 + c * 10 + d));

    int32_t *d_codes;
    __half *d_emb, *d_latents;
    CUDA_CHECK(cudaMalloc(&d_codes,   B * N * T * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_emb,     N * cb_size * dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_latents, B * dim * T * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_codes, h_codes.data(), B * N * T * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_emb, h_emb.data(), N * cb_size * dim * sizeof(__half), cudaMemcpyHostToDevice));

    rvq_lookup_decode(d_codes, d_emb, d_latents, B, N, cb_size, T, dim, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(B * dim * T);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_latents, B * dim * T * sizeof(__half), cudaMemcpyDeviceToHost));

    // Manual check for (b=0, t=0, d=0):
    //   sum = emb[0*4+0 + 0*dim] + emb[1*4+2 + 0*dim]
    //       = emb[0,D=0] + emb[6,D=0] ... layout = [N, cb_size, dim]
    // Actually emb[n*cb_size*dim + c*dim + d]
    // n=0, c=0: 0*100+0*10+0 = 0
    // n=1, c=2: 1*100+2*10+0 = 120
    // sum = 0 + 120 = 120
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < dim; d++) {
            float expected = 0;
            for (int n = 0; n < N; n++) {
                int code = h_codes[(0 * N + n) * T + t];  // [b*N*T + n*T + t]
                expected += static_cast<float>(n * 100 + code * 10 + d);
            }
            float got = to_float(h_out[(0 * dim + d) * T + t]);
            EXPECT_NEAR(got, expected, 1e-2f);
        }
    }

    CUDA_CHECK(cudaFree(d_codes)); CUDA_CHECK(cudaFree(d_emb)); CUDA_CHECK(cudaFree(d_latents));
}

// ── RVQ step per codebook ───────────────────────────────────────────────────
TEST_F(KernelTest, RvqStepPerCodebook) {
    constexpr int B = 1, T = 2, cb_size = 4, D = 3, L = 5;
    // codes_bt [B, T] = [1, 2]
    std::vector<int32_t> h_codes = {0, 3};
    // codebook [cb_size, D] = [4, 3]
    std::vector<__half> h_cb(cb_size * D);
    for (int c = 0; c < cb_size; c++)
        for (int d = 0; d < D; d++)
            h_cb[c * D + d] = to_half(static_cast<float>(c * 10 + d));
    // out_proj_w [L, D] = [5, 3]
    std::vector<__half> h_w(L * D);
    for (int l = 0; l < L; l++)
        for (int d = 0; d < D; d++)
            h_w[l * D + d] = to_half(static_cast<float>(l + d));
    // out_proj_b [L] = [5]
    std::vector<__half> h_b(L);
    for (int l = 0; l < L; l++) h_b[l] = to_half(static_cast<float>(l));

    int32_t *d_codes;
    __half *d_cb, *d_w, *d_b;
    float *d_latent;
    CUDA_CHECK(cudaMalloc(&d_codes,  B * T * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_cb,     cb_size * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_w,      L * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b,      L * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_latent, B * L * T * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_codes, h_codes.data(), B * T * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cb, h_cb.data(), cb_size * D * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), L * D * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), L * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_latent, 0, B * L * T * sizeof(float)));

    rvq_step_per_codebook(d_codes, d_cb, d_w, d_b, d_latent, B, T, cb_size, D, L, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<float> h_out(B * L * T);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_latent, B * L * T * sizeof(float), cudaMemcpyDeviceToHost));

    // For each (b,t,l): out = bias[l] + sum_d(w[l,d] * emb[code[t], d])
    for (int t = 0; t < T; t++) {
        int code = h_codes[t];
        for (int l = 0; l < L; l++) {
            float expected = static_cast<float>(l);  // bias
            for (int d = 0; d < D; d++) {
                expected += static_cast<float>(l + d) * static_cast<float>(code * 10 + d);
            }
            float got = h_out[(0 * L + l) * T + t];
            EXPECT_NEAR(got, expected, 1e-1f);
        }
    }

    CUDA_CHECK(cudaFree(d_codes)); CUDA_CHECK(cudaFree(d_cb));
    CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_latent));
}

TEST_F(KernelTest, RvqStepClampsHighCodeToLastEntry) {
    constexpr int B = 1, T = 1, cb_size = 4, D = 2, L = 2;
    std::vector<int32_t> h_codes = {99};
    std::vector<__half> h_cb = {
        to_half(0.0f), to_half(0.0f),
        to_half(1.0f), to_half(1.0f),
        to_half(2.0f), to_half(2.0f),
        to_half(3.0f), to_half(4.0f),
    };
    std::vector<__half> h_w = {
        to_half(1.0f), to_half(0.0f),
        to_half(0.0f), to_half(1.0f),
    };
    std::vector<__half> h_b(L, to_half(0.0f));

    int32_t* d_codes;
    __half *d_cb, *d_w, *d_b;
    float* d_latent;
    CUDA_CHECK(cudaMalloc(&d_codes, B * T * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_cb, cb_size * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_w, L * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b, L * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_latent, B * L * T * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_codes, h_codes.data(), B * T * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cb, h_cb.data(), cb_size * D * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), L * D * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), L * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_latent, 0, B * L * T * sizeof(float)));

    rvq_step_per_codebook(d_codes, d_cb, d_w, d_b, d_latent, B, T, cb_size, D, L, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<float> h_out(B * L * T);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_latent, B * L * T * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_NEAR(h_out[0], 3.0f, 1e-3f);
    EXPECT_NEAR(h_out[1], 4.0f, 1e-3f);

    CUDA_CHECK(cudaFree(d_codes)); CUDA_CHECK(cudaFree(d_cb));
    CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_latent));
}

TEST_F(KernelTest, RvqStepAccumulates) {
    // Verify that multiple calls to rvq_step_per_codebook accumulate correctly
    constexpr int B = 1, T = 1, cb_size = 3, D = 2, L = 2;
    std::vector<int32_t> h_codes0 = {0};
    std::vector<int32_t> h_codes1 = {2};
    std::vector<__half> h_cb(cb_size * D);
    for (int c = 0; c < cb_size; c++)
        for (int d = 0; d < D; d++)
            h_cb[c * D + d] = to_half(static_cast<float>(c + d));
    std::vector<__half> h_w(L * D, to_half(1.0f));
    std::vector<__half> h_b(L, to_half(0.0f));

    int32_t *d_c0, *d_c1;
    __half *d_cb, *d_w, *d_b;
    float *d_latent;
    CUDA_CHECK(cudaMalloc(&d_c0, B * T * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_c1, B * T * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_cb, cb_size * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_w, L * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b, L * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_latent, B * L * T * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_c0, h_codes0.data(), B * T * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c1, h_codes1.data(), B * T * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cb, h_cb.data(), cb_size * D * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), L * D * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), L * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_latent, 0, B * L * T * sizeof(float)));

    // Two steps, different codebooks
    rvq_step_per_codebook(d_c0, d_cb, d_w, d_b, d_latent, B, T, cb_size, D, L, stream_);
    rvq_step_per_codebook(d_c1, d_cb, d_w, d_b, d_latent, B, T, cb_size, D, L, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<float> h_out(B * L * T);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_latent, B * L * T * sizeof(float), cudaMemcpyDeviceToHost));

    // step 0: code=0, emb=[0,1]; step 1: code=2, emb=[2,3]; sum=[2,4]
    // w=[1,1], so dot = 2+4 = 6 per output channel
    for (int l = 0; l < L; l++)
        EXPECT_NEAR(h_out[l], 6.0f, 1e-1f);

    CUDA_CHECK(cudaFree(d_c0)); CUDA_CHECK(cudaFree(d_c1));
    CUDA_CHECK(cudaFree(d_cb)); CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_latent));
}

TEST_F(KernelTest, RvqEncodeStepUsesProjectionBiases) {
    constexpr int B = 1, T = 1, cb_size = 3, D = 2, L = 2;

    std::vector<float> h_residual = {3.0f, 4.0f};  // [B,L,T]
    std::vector<__half> h_in_w(D * L, to_half(0.0f));
    std::vector<__half> h_in_b = {to_half(1.0f), to_half(0.0f)};
    std::vector<__half> h_cb = {
        to_half(0.0f),  to_half(1.0f),
        to_half(1.0f),  to_half(0.0f),
        to_half(-1.0f), to_half(0.0f),
    };
    std::vector<__half> h_out_w = {
        to_half(1.0f), to_half(0.0f),
        to_half(0.0f), to_half(1.0f),
    };
    std::vector<__half> h_out_b = {to_half(0.25f), to_half(-0.5f)};

    float *d_residual, *d_residual_out;
    __half *d_in_w, *d_in_b, *d_cb, *d_out_w, *d_out_b;
    int32_t* d_codes;
    CUDA_CHECK(cudaMalloc(&d_residual, B * L * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual_out, B * L * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_in_w, D * L * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_in_b, D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_cb, cb_size * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out_w, L * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out_b, L * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_codes, B * T * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_residual, h_residual.data(), B * L * T * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_w, h_in_w.data(), D * L * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_b, h_in_b.data(), D * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cb, h_cb.data(), cb_size * D * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_w, h_out_w.data(), L * D * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_b, h_out_b.data(), L * sizeof(__half), cudaMemcpyHostToDevice));

    rvq_encode_step(d_residual, d_in_w, d_in_b, d_cb, d_out_w, d_out_b,
                    d_codes, d_residual_out, B, T, cb_size, D, L, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    int32_t h_code = -1;
    std::vector<float> h_out(B * L * T);
    CUDA_CHECK(cudaMemcpy(&h_code, d_codes, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_residual_out, B * L * T * sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_code, 1);
    EXPECT_NEAR(h_out[0], 1.75f, 1e-3f);  // 3 - (1 + 0.25)
    EXPECT_NEAR(h_out[1], 4.5f, 1e-3f);   // 4 - (0 - 0.5)

    CUDA_CHECK(cudaFree(d_residual)); CUDA_CHECK(cudaFree(d_residual_out));
    CUDA_CHECK(cudaFree(d_in_w)); CUDA_CHECK(cudaFree(d_in_b));
    CUDA_CHECK(cudaFree(d_cb)); CUDA_CHECK(cudaFree(d_out_w));
    CUDA_CHECK(cudaFree(d_out_b)); CUDA_CHECK(cudaFree(d_codes));
}

TEST_F(KernelTest, RvqEncodeStepUsesNormalizedNearestNeighbor) {
    constexpr int B = 1, T = 1, cb_size = 3, D = 2, L = 2;

    std::vector<float> h_residual = {1.0f, 0.0f};  // [B,L,T]
    std::vector<__half> h_in_w = {
        to_half(1.0f), to_half(0.0f),
        to_half(0.0f), to_half(1.0f),
    };
    std::vector<__half> h_in_b(D, to_half(0.0f));
    std::vector<__half> h_cb = {
        to_half(10.0f), to_half(0.0f),
        to_half(0.9f),  to_half(0.4f),
        to_half(0.0f),  to_half(1.0f),
    };
    std::vector<__half> h_out_w = {
        to_half(1.0f), to_half(0.0f),
        to_half(0.0f), to_half(1.0f),
    };
    std::vector<__half> h_out_b(L, to_half(0.0f));

    float *d_residual, *d_residual_out;
    __half *d_in_w, *d_in_b, *d_cb, *d_out_w, *d_out_b;
    int32_t* d_codes;
    CUDA_CHECK(cudaMalloc(&d_residual, B * L * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual_out, B * L * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_in_w, D * L * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_in_b, D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_cb, cb_size * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out_w, L * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out_b, L * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_codes, B * T * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_residual, h_residual.data(), B * L * T * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_w, h_in_w.data(), D * L * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_b, h_in_b.data(), D * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cb, h_cb.data(), cb_size * D * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_w, h_out_w.data(), L * D * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_b, h_out_b.data(), L * sizeof(__half), cudaMemcpyHostToDevice));

    rvq_encode_step(d_residual, d_in_w, d_in_b, d_cb, d_out_w, d_out_b,
                    d_codes, d_residual_out, B, T, cb_size, D, L, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    int32_t h_code = -1;
    std::vector<float> h_out(B * L * T);
    CUDA_CHECK(cudaMemcpy(&h_code, d_codes, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_residual_out, B * L * T * sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_code, 0);
    EXPECT_NEAR(h_out[0], -9.0f, 1e-3f);
    EXPECT_NEAR(h_out[1], 0.0f, 1e-3f);

    CUDA_CHECK(cudaFree(d_residual)); CUDA_CHECK(cudaFree(d_residual_out));
    CUDA_CHECK(cudaFree(d_in_w)); CUDA_CHECK(cudaFree(d_in_b));
    CUDA_CHECK(cudaFree(d_cb)); CUDA_CHECK(cudaFree(d_out_w));
    CUDA_CHECK(cudaFree(d_out_b)); CUDA_CHECK(cudaFree(d_codes));
}

// ── Sampling ────────────────────────────────────────────────────────────────
TEST_F(KernelTest, SamplingTopK) {
    // With temperature=1, top_p=1, top_k=1, seed=fixed: should pick argmax
    constexpr int B = 2, V = 8;
    std::vector<__half> h_logits(B * V);
    for (int i = 0; i < V; i++) {
        h_logits[0 * V + i] = to_half(static_cast<float>(i));      // max = 7
        h_logits[1 * V + i] = to_half(static_cast<float>(V - i));  // max = 8 (index 0)
    }

    __half *d_logits;
    int32_t *d_tokens;
    CUDA_CHECK(cudaMalloc(&d_logits, B * V * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_tokens, B * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(), B * V * sizeof(__half), cudaMemcpyHostToDevice));

    sample_top_k_top_p(d_logits, d_tokens, B, V, 1.0f, 1.0f, 1, 42, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<int32_t> h_tokens(B);
    CUDA_CHECK(cudaMemcpy(h_tokens.data(), d_tokens, B * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // Batch 0: max is index 7; Batch 1: max is index 0
    EXPECT_EQ(h_tokens[0], 7);
    EXPECT_EQ(h_tokens[1], 0);

    CUDA_CHECK(cudaFree(d_logits)); CUDA_CHECK(cudaFree(d_tokens));
}

TEST_F(KernelTest, SamplingWithTemperature) {
    // Very high temperature → uniform-ish distribution
    // Very low temperature → deterministic argmax
    constexpr int B = 1, V = 4;
    std::vector<__half> h_logits = {to_half(0.f), to_half(100.f), to_half(0.f), to_half(0.f)};

    __half *d_logits;
    int32_t *d_tokens;
    CUDA_CHECK(cudaMalloc(&d_logits, B * V * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_tokens, B * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(), B * V * sizeof(__half), cudaMemcpyHostToDevice));

    // Temperature near zero → pick max (index 1)
    sample_top_k_top_p(d_logits, d_tokens, B, V, 0.01f, 1.0f, 4, 42, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    int32_t token;
    CUDA_CHECK(cudaMemcpy(&token, d_tokens, sizeof(int32_t), cudaMemcpyDeviceToHost));
    EXPECT_EQ(token, 1);

    CUDA_CHECK(cudaFree(d_logits)); CUDA_CHECK(cudaFree(d_tokens));
}

// ── Thread count stress test ────────────────────────────────────────────────
TEST_F(KernelTest, LargeEmbeddingLookup) {
    // Test embedding lookup with sizes that span multiple thread blocks
    constexpr int n_tokens = 256, dim = 1024, vocab = 32000;
    size_t total = static_cast<size_t>(n_tokens) * dim;

    int32_t *d_indices;
    __half *d_table, *d_out;
    CUDA_CHECK(cudaMalloc(&d_indices, n_tokens * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_table,  static_cast<size_t>(vocab) * dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out,    total * sizeof(__half)));

    // Fill with simple patterns
    std::vector<int32_t> h_indices(n_tokens);
    for (int i = 0; i < n_tokens; i++) h_indices[i] = i % vocab;
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_table, 0, static_cast<size_t>(vocab) * dim * sizeof(__half)));

    embedding_lookup(d_indices, d_table, d_out, n_tokens, dim, vocab, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<__half> h_out(total);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total * sizeof(__half), cudaMemcpyDeviceToHost));

    // All outputs should be finite
    for (size_t i = 0; i < total; i++)
        EXPECT_FALSE(std::isnan(to_float(h_out[i])));

    CUDA_CHECK(cudaFree(d_indices)); CUDA_CHECK(cudaFree(d_table)); CUDA_CHECK(cudaFree(d_out));
}

}  // namespace
}  // namespace fish::kernels
