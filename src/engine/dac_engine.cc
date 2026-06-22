// src/engine/dac_engine.cc
#include "engine/dac_engine.h"
#include "utils/cuda_utils.h"
#include "kernels/kernels.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace fish {

namespace {

void gemm_fp16(int M_out, int N_out, int K,
               const __half* W, const __half* X, __half* Y,
               cublasHandle_t cublas) {
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        M_out, N_out, K,
        &alpha,
        W, CUDA_R_16F, K,
        X, CUDA_R_16F, K,
        &beta,
        Y, CUDA_R_16F, M_out,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

void gemm_fp16_to_f32(int M_out, int N_out, int K,
                      const __half* W, const __half* X, float* Y,
                      cublasHandle_t cublas) {
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        M_out, N_out, K,
        &alpha,
        W, CUDA_R_16F, K,
        X, CUDA_R_16F, K,
        &beta,
        Y, CUDA_R_32F, M_out,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

// Dispatch GEMM: INT8 dequant+GEMM if scale is provided, else FP16 cuBLAS.
static void quantized_gemm_fp16(int M_out, int N_out, int K,
                                 const __half* W, const __half* scale,
                                 const __half* X, __half* Y,
                                 cublasHandle_t cublas, cudaStream_t stream) {
    if (scale != nullptr) {
        kernels::int8_dequant_gemm_fp16(
            reinterpret_cast<const int8_t*>(W), scale,
            X, Y, M_out, N_out, K, stream);
    } else {
        gemm_fp16(M_out, N_out, K, W, X, Y, cublas);
    }
}

std::string rvq_prefix_for_codebook(int cb) {
    if (cb == 0)
        return "quantizer.semantic_quantizer.quantizers.0";
    return "quantizer.quantizer.quantizers." + std::to_string(cb - 1);
}

__half half_from_tv(const TensorView& tv, int idx) {
    if (tv.dtype == DType::FP16)
        return static_cast<const __half*>(tv.data)[idx];
    if (tv.dtype == DType::FP32)
        return __float2half(static_cast<const float*>(tv.data)[idx]);
    return static_cast<const __half*>(tv.data)[idx];
}

float float_from_tv(const TensorView& tv, int idx) {
    return __half2float(half_from_tv(tv, idx));
}

std::string dac_dump_prefix() {
    const char* p = std::getenv("FISH_DUMP_DAC_PREFIX");
    return (p && *p) ? std::string(p) : std::string();
}

void write_f32_tensor(const std::string& path, const std::vector<int32_t>& shape,
                      const float* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    if (!f.good()) {
        spdlog::warn("Cannot write DAC dump: {}", path);
        return;
    }
    int32_t ndim = static_cast<int32_t>(shape.size());
    f.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    f.write(reinterpret_cast<const char*>(shape.data()), shape.size() * sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(data), n * sizeof(float));
    spdlog::info("DAC dump: {} ({} values)", path, n);
}

void dump_device_f32(const std::string& name, const float* d, const std::vector<int32_t>& shape) {
    std::string prefix = dac_dump_prefix();
    if (prefix.empty()) return;
    size_t n = 1;
    for (int32_t s : shape) n *= static_cast<size_t>(s);
    std::vector<float> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    write_f32_tensor(prefix + "_" + name + ".bin", shape, h.data(), n);
}

void dump_device_f16(const std::string& name, const __half* d, const std::vector<int32_t>& shape) {
    std::string prefix = dac_dump_prefix();
    if (prefix.empty()) return;
    size_t n = 1;
    for (int32_t s : shape) n *= static_cast<size_t>(s);
    std::vector<__half> hh(n);
    std::vector<float> hf(n);
    CUDA_CHECK(cudaMemcpy(hh.data(), d, n * sizeof(__half), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) hf[i] = __half2float(hh[i]);
    write_f32_tensor(prefix + "_" + name + ".bin", shape, hf.data(), n);
}

std::vector<__half> fuse_weight_norm(const TensorView& g_tv, const TensorView& v_tv) {
    int c0 = static_cast<int>(v_tv.shape[0]);
    int c1 = static_cast<int>(v_tv.shape[1]);
    int k = static_cast<int>(v_tv.shape[2]);
    std::vector<__half> fused(static_cast<size_t>(c0) * c1 * k);
    for (int o = 0; o < c0; ++o) {
        double ss = 0.0;
        for (int i = 0; i < c1; ++i) {
            for (int x = 0; x < k; ++x) {
                int idx = (o * c1 + i) * k + x;
                float v = float_from_tv(v_tv, idx);
                ss += static_cast<double>(v) * v;
            }
        }
        float scale = float_from_tv(g_tv, o) / std::sqrt(std::max(ss, 1e-20));
        for (int i = 0; i < c1; ++i) {
            for (int x = 0; x < k; ++x) {
                int idx = (o * c1 + i) * k + x;
                fused[idx] = __float2half(float_from_tv(v_tv, idx) * scale);
            }
        }
    }
    return fused;
}

}  // namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
DACEngine::DACEngine(const DACConfig& cfg, ModelLoader&& loader,
                     cublasHandle_t cublas, cudnnHandle_t cudnn, cudaStream_t stream)
    : cfg_(cfg), loader_(std::move(loader)), cublas_(cublas), cudnn_(cudnn), stream_(stream)
{
    spdlog::info("DACEngine: hop={} sample_rate={} num_codebooks={}",
                 cfg_.hop_length(), cfg_.sample_rate, cfg_.num_codebooks);
    CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
DACEngine::~DACEngine() {
    for (__half* p : gpu_codebooks_) if (p) cudaFree(p);
    for (__half* p : gpu_out_proj_w_) if (p) cudaFree(p);
    for (__half* p : gpu_out_proj_b_) if (p) cudaFree(p);
    for (auto& kv : conv_weights_) {
        if (kv.second.weight) cudaFree(kv.second.weight);
        if (kv.second.bias) cudaFree(kv.second.bias);
    }
    for (auto& kv : dense_weights_) {
        if (kv.second.weight) cudaFree(kv.second.weight);
        if (kv.second.bias) cudaFree(kv.second.bias);
    }
    for (auto& kv : vector_weights_) {
        if (kv.second) cudaFree(kv.second);
    }
    for (__half* p : gpu_in_proj_w_) if (p) cudaFree(p);
    for (__half* p : gpu_in_proj_b_) if (p) cudaFree(p);
    if (latent_buf_) { cudaFree(latent_buf_); latent_buf_ = nullptr; }
    if (decoder_buf_a_) { cudaFree(decoder_buf_a_); decoder_buf_a_ = nullptr; }
    if (decoder_buf_b_) { cudaFree(decoder_buf_b_); decoder_buf_b_ = nullptr; }
    if (decoder_buf_c_) { cudaFree(decoder_buf_c_); decoder_buf_c_ = nullptr; }
    if (decoder_buf_d_) { cudaFree(decoder_buf_d_); decoder_buf_d_ = nullptr; }
    if (encoder_buf_h_) { cudaFree(encoder_buf_h_); encoder_buf_h_ = nullptr; }
    if (encoder_buf_)  { cudaFree(encoder_buf_);  encoder_buf_  = nullptr; }
}

void DACEngine::load_decoder_weights() {
    auto load_conv = [&](const std::string& base, bool transpose) {
        std::string fused_w_key = base + ".conv.weight";
        std::string fused_b_key = base + ".conv.bias";
        if (loader_.has(fused_w_key) && loader_.has(fused_b_key)) {
            auto w_tv = loader_.get(fused_w_key);
            auto b_tv = loader_.get(fused_b_key);
            ConvWeight cw;
            cw.c0 = static_cast<int>(w_tv.shape[0]);
            cw.c1 = static_cast<int>(w_tv.shape[1]);
            cw.k = static_cast<int>(w_tv.shape[2]);
            cw.transpose = transpose;
            std::vector<__half> w(static_cast<size_t>(cw.c0) * cw.c1 * cw.k);
            for (size_t i = 0; i < w.size(); ++i)
                w[i] = half_from_tv(w_tv, static_cast<int>(i));
            CUDA_CHECK(cudaMalloc(&cw.weight, w.size() * sizeof(__half)));
            CUDA_CHECK(cudaMemcpy(cw.weight, w.data(), w.size() * sizeof(__half),
                                  cudaMemcpyHostToDevice));
            int bias_n = static_cast<int>(b_tv.numel());
            std::vector<__half> bias(bias_n);
            for (int i = 0; i < bias_n; ++i) bias[i] = half_from_tv(b_tv, i);
            CUDA_CHECK(cudaMalloc(&cw.bias, bias.size() * sizeof(__half)));
            CUDA_CHECK(cudaMemcpy(cw.bias, bias.data(), bias.size() * sizeof(__half),
                                  cudaMemcpyHostToDevice));
            conv_weights_[base] = cw;
            return;
        }

        std::string g_key = base + ".conv.parametrizations.weight.original0";
        std::string v_key = base + ".conv.parametrizations.weight.original1";
        std::string b_key = base + ".conv.bias";
        if (!loader_.has(g_key) || !loader_.has(v_key) || !loader_.has(b_key))
            throw std::runtime_error("DACEngine: missing decoder conv tensors for " + base);

        auto g_tv = loader_.get(g_key);
        auto v_tv = loader_.get(v_key);
        auto b_tv = loader_.get(b_key);
        auto fused = fuse_weight_norm(g_tv, v_tv);

        ConvWeight cw;
        cw.c0 = static_cast<int>(v_tv.shape[0]);
        cw.c1 = static_cast<int>(v_tv.shape[1]);
        cw.k = static_cast<int>(v_tv.shape[2]);
        cw.transpose = transpose;
        CUDA_CHECK(cudaMalloc(&cw.weight, fused.size() * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(cw.weight, fused.data(), fused.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));

        int bias_n = static_cast<int>(b_tv.numel());
        std::vector<__half> bias(bias_n);
        for (int i = 0; i < bias_n; ++i) bias[i] = half_from_tv(b_tv, i);
        CUDA_CHECK(cudaMalloc(&cw.bias, bias.size() * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(cw.bias, bias.data(), bias.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));
        conv_weights_[base] = cw;
    };

    auto load_plain_conv = [&](const std::string& base, bool transpose) {
        std::string w_key = base + ".conv.weight";
        std::string b_key = base + ".conv.bias";
        if (!loader_.has(w_key) || !loader_.has(b_key))
            throw std::runtime_error("DACEngine: missing conv tensors for " + base);

        auto w_tv = loader_.get(w_key);
        auto b_tv = loader_.get(b_key);
        ConvWeight cw;
        cw.c0 = static_cast<int>(w_tv.shape[0]);
        cw.c1 = static_cast<int>(w_tv.shape[1]);
        cw.k = static_cast<int>(w_tv.shape[2]);
        cw.transpose = transpose;

        std::vector<__half> w(static_cast<size_t>(cw.c0) * cw.c1 * cw.k);
        for (size_t i = 0; i < w.size(); ++i)
            w[i] = half_from_tv(w_tv, static_cast<int>(i));
        CUDA_CHECK(cudaMalloc(&cw.weight, w.size() * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(cw.weight, w.data(), w.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));

        int bias_n = static_cast<int>(b_tv.numel());
        std::vector<__half> bias(bias_n);
        for (int i = 0; i < bias_n; ++i) bias[i] = half_from_tv(b_tv, i);
        CUDA_CHECK(cudaMalloc(&cw.bias, bias.size() * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(cw.bias, bias.data(), bias.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));
        conv_weights_[base] = cw;
    };

    auto load_dense = [&](const std::string& base) {
        std::string w_key = base + ".weight";
        std::string b_key = base + ".bias";
        if (!loader_.has(w_key) || !loader_.has(b_key))
            throw std::runtime_error("DACEngine: missing dense tensors for " + base);
        auto w_tv = loader_.get(w_key);
        auto b_tv = loader_.get(b_key);
        DenseWeight dw;
        dw.out = static_cast<int>(w_tv.shape[0]);
        dw.in = static_cast<int>(w_tv.shape[1]);
        size_t n = static_cast<size_t>(dw.out) * dw.in;
        std::vector<__half> w(n);
        for (size_t i = 0; i < n; ++i) w[i] = half_from_tv(w_tv, static_cast<int>(i));
        CUDA_CHECK(cudaMalloc(&dw.weight, n * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(dw.weight, w.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
        std::vector<__half> b(dw.out);
        for (int i = 0; i < dw.out; ++i) b[i] = half_from_tv(b_tv, i);
        CUDA_CHECK(cudaMalloc(&dw.bias, b.size() * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(dw.bias, b.data(), b.size() * sizeof(__half), cudaMemcpyHostToDevice));

        // Check for INT8 scale
        std::string scale_key = base + "_scale";
        if (loader_.has(scale_key)) {
            use_int8_ = true;
            auto s_tv = loader_.get(scale_key);
            CUDA_CHECK(cudaMalloc(&dw.scale, s_tv.nbytes()));
            CUDA_CHECK(cudaMemcpy(dw.scale, s_tv.data, s_tv.nbytes(), cudaMemcpyHostToDevice));
        }
        dense_weights_[base] = dw;
    };

    auto load_matrix = [&](const std::string& key) {
        auto w_tv = loader_.get(key);
        DenseWeight dw;
        dw.out = static_cast<int>(w_tv.shape[0]);
        dw.in = static_cast<int>(w_tv.shape[1]);
        size_t n = static_cast<size_t>(dw.out) * dw.in;
        std::vector<__half> w(n);
        for (size_t i = 0; i < n; ++i) w[i] = half_from_tv(w_tv, static_cast<int>(i));
        CUDA_CHECK(cudaMalloc(&dw.weight, n * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(dw.weight, w.data(), n * sizeof(__half), cudaMemcpyHostToDevice));

        // Check for INT8 scale
        std::string scale_key = key + "_scale";
        if (loader_.has(scale_key)) {
            use_int8_ = true;
            auto s_tv = loader_.get(scale_key);
            CUDA_CHECK(cudaMalloc(&dw.scale, s_tv.nbytes()));
            CUDA_CHECK(cudaMemcpy(dw.scale, s_tv.data, s_tv.nbytes(), cudaMemcpyHostToDevice));
        }
        dense_weights_[key] = dw;
    };

    auto load_vector = [&](const std::string& key) {
        auto tv = loader_.get(key);
        std::vector<__half> v(tv.numel());
        for (int64_t i = 0; i < tv.numel(); ++i) v[i] = half_from_tv(tv, static_cast<int>(i));
        __half* gp = nullptr;
        CUDA_CHECK(cudaMalloc(&gp, v.size() * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(gp, v.data(), v.size() * sizeof(__half), cudaMemcpyHostToDevice));
        vector_weights_[key] = gp;
    };

    load_plain_conv("quantizer.upsample.0.0", true);
    load_plain_conv("quantizer.upsample.1.0", true);
    for (int i = 0; i < 2; ++i) {
        std::string p = "quantizer.upsample." + std::to_string(i) + ".1";
        load_plain_conv(p + ".dwconv", false);
        load_vector(p + ".norm.weight");
        load_vector(p + ".norm.bias");
        load_vector(p + ".gamma");
        load_dense(p + ".pwconv1");
        load_dense(p + ".pwconv2");
    }
    load_vector("quantizer.post_module.freqs_cis");
    load_vector("quantizer.post_module.norm.weight");
    for (int l = 0; l < 8; ++l) {
        std::string p = "quantizer.post_module.layers." + std::to_string(l);
        load_matrix(p + ".attention.wqkv.weight");
        load_matrix(p + ".attention.wo.weight");
        load_matrix(p + ".feed_forward.w1.weight");
        load_matrix(p + ".feed_forward.w2.weight");
        load_matrix(p + ".feed_forward.w3.weight");
        load_vector(p + ".attention_norm.weight");
        load_vector(p + ".ffn_norm.weight");
        load_vector(p + ".attention_layer_scale.gamma");
        load_vector(p + ".ffn_layer_scale.gamma");
    }

    load_conv("decoder.model.0", false);
    load_conv("decoder.model.1.block.1", true);
    load_conv("decoder.model.2.block.1", true);
    load_conv("decoder.model.3.block.1", true);
    load_conv("decoder.model.4.block.1", true);
    load_conv("decoder.model.6", false);
    load_vector("decoder.model.5.alpha");

    for (int block = 1; block <= 4; ++block) {
        load_vector("decoder.model." + std::to_string(block) + ".block.0.alpha");
        for (int ru = 2; ru <= 4; ++ru) {
            load_vector("decoder.model." + std::to_string(block) +
                        ".block." + std::to_string(ru) + ".block.0.alpha");
            load_conv("decoder.model." + std::to_string(block) +
                      ".block." + std::to_string(ru) + ".block.1", false);
            load_vector("decoder.model." + std::to_string(block) +
                        ".block." + std::to_string(ru) + ".block.2.alpha");
            load_conv("decoder.model." + std::to_string(block) +
                      ".block." + std::to_string(ru) + ".block.3", false);
        }
    }
    spdlog::info("DACEngine::init: loaded {} native decoder conv weights", conv_weights_.size());

    // ===== Encoder weights (audio → latent z) =====

    // block.0: initial conv (1→64, k=7), weight_norm fused by converter
    load_conv("encoder.block.0", false);

    // blocks 1-4: each has 3× residual_unit + snake + strided downsample conv
    // C: 64→128 (stride 2), 128→256 (stride 4), 256→512 (stride 8), 512→1024 (stride 8)
    for (int b = 1; b <= 4; ++b) {
        std::string p = "encoder.block." + std::to_string(b);
        for (int ru = 0; ru < 3; ++ru) {
            std::string rp = p + ".block." + std::to_string(ru);
            load_vector(rp + ".block.0.alpha");
            load_conv(rp + ".block.1", false);     // dilated conv (k=7)
            load_vector(rp + ".block.2.alpha");
            load_conv(rp + ".block.3", false);     // pointwise conv (k=1)
        }
        load_vector(p + ".block.3.alpha");          // snake before downsample
        load_plain_conv(p + ".block.4", false);     // strided downsample conv
    }

    // encoder.block.4 has a 4-layer WindowLimitedTransformer after the stride-8 conv.
    if (loader_.has("encoder.block.4.block.5.norm.weight")) {
        std::string ep = "encoder.block.4.block.5";
        load_vector(ep + ".freqs_cis");
        load_vector(ep + ".norm.weight");
        for (int l = 0; l < 4; ++l) {
            std::string lp = ep + ".layers." + std::to_string(l);
            load_matrix(lp + ".attention.wqkv.weight");
            load_matrix(lp + ".attention.wo.weight");
            load_matrix(lp + ".feed_forward.w1.weight");
            load_matrix(lp + ".feed_forward.w2.weight");
            load_matrix(lp + ".feed_forward.w3.weight");
            load_vector(lp + ".attention_norm.weight");
            load_vector(lp + ".ffn_norm.weight");
            load_vector(lp + ".attention_layer_scale.gamma");
            load_vector(lp + ".ffn_layer_scale.gamma");
        }
    }

    // quantizer.downsample: 2 stages, each stride-2 conv + ConvNeXt
    for (int i = 0; i < 2; ++i) {
        std::string dp = "quantizer.downsample." + std::to_string(i);
        load_plain_conv(dp + ".0", false);
        load_plain_conv(dp + ".1.dwconv", false);
        load_vector(dp + ".1.norm.weight");
        load_vector(dp + ".1.norm.bias");
        load_vector(dp + ".1.gamma");
        load_dense(dp + ".1.pwconv1");
        load_dense(dp + ".1.pwconv2");
    }

    // quantizer.pre_module: 8-layer transformer (same structure as post_module)
    load_vector("quantizer.pre_module.freqs_cis");
    load_vector("quantizer.pre_module.norm.weight");
    for (int l = 0; l < 8; ++l) {
        std::string pp = "quantizer.pre_module.layers." + std::to_string(l);
        load_matrix(pp + ".attention.wqkv.weight");
        load_matrix(pp + ".attention.wo.weight");
        load_matrix(pp + ".feed_forward.w1.weight");
        load_matrix(pp + ".feed_forward.w2.weight");
        load_matrix(pp + ".feed_forward.w3.weight");
        load_vector(pp + ".attention_norm.weight");
        load_vector(pp + ".ffn_norm.weight");
        load_vector(pp + ".attention_layer_scale.gamma");
        load_vector(pp + ".ffn_layer_scale.gamma");
    }

    // RVQ in_proj weights + biases (latent → codebook space projection for quantization)
    gpu_in_proj_w_.resize(DAC_TOTAL_CODEBOOKS, nullptr);
    gpu_in_proj_b_.resize(DAC_TOTAL_CODEBOOKS, nullptr);
    int loaded_in_proj = 0;
    for (int i = 0; i < DAC_TOTAL_CODEBOOKS; i++) {
        std::string key = rvq_prefix_for_codebook(i) + ".in_proj.weight";
        if (loader_.has(key)) {
            auto tv = loader_.get(key);
            __half* gp = nullptr;
            CUDA_CHECK(cudaMalloc(&gp, tv.nbytes()));
            CUDA_CHECK(cudaMemcpy(gp, tv.data, tv.nbytes(), cudaMemcpyHostToDevice));
            gpu_in_proj_w_[i] = gp;
        }
        std::string bkey = rvq_prefix_for_codebook(i) + ".in_proj.bias";
        if (loader_.has(bkey)) {
            auto tv = loader_.get(bkey);
            __half* gp = nullptr;
            CUDA_CHECK(cudaMalloc(&gp, tv.nbytes()));
            CUDA_CHECK(cudaMemcpy(gp, tv.data, tv.nbytes(), cudaMemcpyHostToDevice));
            gpu_in_proj_b_[i] = gp;
            if (gpu_in_proj_w_[i]) loaded_in_proj++;
        }
    }

    // encoder.block.5/6: final Snake + conv before quantization (if present)
    if (loader_.has("encoder.block.6.conv.weight"))
        load_vector("encoder.block.5.alpha");
    if (loader_.has("encoder.block.6.conv.weight"))
        load_conv("encoder.block.6", false);

    spdlog::info("DACEngine::init: encoder weights loaded ({} conv + {} dense + {} vector total, {} RVQ in_proj)",
                 conv_weights_.size(), dense_weights_.size(), vector_weights_.size(), loaded_in_proj);
}

// ---------------------------------------------------------------------------
// init() — load RVQ codebook weights from .bin loader when available,
// upload to GPU.
// ---------------------------------------------------------------------------
void DACEngine::init() {
    // Check if we have codebook weights in the loader
    for (int i = 0; i < DAC_TOTAL_CODEBOOKS; i++) {
        const std::string prefix = rvq_prefix_for_codebook(i);
        const std::string cb_key = prefix + ".codebook.weight";
        const std::string proj_key = prefix + ".out_proj.weight";
        const std::string bias_key = prefix + ".out_proj.bias";
        if (!loader_.has(cb_key) || !loader_.has(proj_key) || !loader_.has(bias_key)) {
            throw std::runtime_error(
                "DACEngine::init: missing RVQ tensors for codebook " + std::to_string(i) +
                " (expected " + cb_key + ", " + proj_key + ", " + bias_key + ")");
        }
    }

    // Load and upload codebook embeddings and out_proj weights
    spdlog::info("DACEngine::init: loading {} codebooks to GPU", DAC_TOTAL_CODEBOOKS);
    gpu_codebooks_.resize(DAC_TOTAL_CODEBOOKS, nullptr);
    gpu_out_proj_w_.resize(DAC_TOTAL_CODEBOOKS, nullptr);
    gpu_out_proj_b_.resize(DAC_TOTAL_CODEBOOKS, nullptr);

    for (int i = 0; i < DAC_TOTAL_CODEBOOKS; i++) {
        const std::string prefix = rvq_prefix_for_codebook(i);

        // Codebook embeddings
        std::string cb_key = prefix + ".codebook.weight";
        auto cb_tv = loader_.get(cb_key);
        __half* cb_gp = nullptr;
        CUDA_CHECK(cudaMalloc(&cb_gp, cb_tv.nbytes()));
        CUDA_CHECK(cudaMemcpy(cb_gp, cb_tv.data, cb_tv.nbytes(), cudaMemcpyHostToDevice));
        gpu_codebooks_[i] = cb_gp;

        // out_proj linear layers
        std::string proj_key = prefix + ".out_proj.weight";
        auto proj_tv = loader_.get(proj_key);
        __half* proj_gp = nullptr;
        CUDA_CHECK(cudaMalloc(&proj_gp, proj_tv.nbytes()));
        CUDA_CHECK(cudaMemcpy(proj_gp, proj_tv.data, proj_tv.nbytes(), cudaMemcpyHostToDevice));
        gpu_out_proj_w_[i] = proj_gp;

        std::string bias_key = prefix + ".out_proj.bias";
        auto bias_tv = loader_.get(bias_key);
        __half* bias_gp = nullptr;
        CUDA_CHECK(cudaMalloc(&bias_gp, bias_tv.nbytes()));
        CUDA_CHECK(cudaMemcpy(bias_gp, bias_tv.data, bias_tv.nbytes(), cudaMemcpyHostToDevice));
        gpu_out_proj_b_[i] = bias_gp;
    }

    // Allocate latent workspace: [B, DAC_LATENT_DIM, T] float32 accumulator
    // Size for a reasonable max T
    size_t latent_bytes = (size_t)1 * DAC_LATENT_DIM * 1024 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&latent_buf_, latent_bytes));
    CUDA_CHECK(cudaMemset(latent_buf_, 0, latent_bytes));

    load_decoder_weights();

    spdlog::info("DACEngine::init: GPU RVQ ready, {} codebooks uploaded", DAC_TOTAL_CODEBOOKS);
}

int DACEngine::decode_waveform_native(int batch_size, int code_len, float* audio, int max_audio_cap) {
    if (!cudnn_)
        throw std::runtime_error("DACEngine::decode: cuDNN handle is null");

    const int final_len = code_len * cfg_.hop_length() * 4;
    if (final_len > max_audio_cap)
        throw std::runtime_error("DACEngine::decode: output audio buffer is too small");

    // Worst intermediate in the partial native path:
    //   quantizer upsample:        1024 * (4T)
    //   decoder block 2 output:     384 * (256T)
    //   decoder block 3 output:     192 * (1024T)
    //   decoder block 4 output:      96 * (2048T)
    // = 196608*T half values. Keep headroom for temporary/residual buffers.
    size_t need = static_cast<size_t>(batch_size) * code_len * 260000;
    if (need > decoder_buf_elems_) {
        if (decoder_buf_a_) cudaFree(decoder_buf_a_);
        if (decoder_buf_b_) cudaFree(decoder_buf_b_);
        if (decoder_buf_c_) cudaFree(decoder_buf_c_);
        if (decoder_buf_d_) cudaFree(decoder_buf_d_);
        CUDA_CHECK(cudaMalloc(&decoder_buf_a_, need * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&decoder_buf_b_, need * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&decoder_buf_c_, need * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&decoder_buf_d_, need * sizeof(__half)));
        decoder_buf_elems_ = need;
    }

    kernels::f32_to_f16_cast(latent_buf_, decoder_buf_a_,
                             batch_size * DAC_LATENT_DIM * code_len, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    dump_device_f32("rvq", latent_buf_, {batch_size, DAC_LATENT_DIM, code_len});

    auto add_bias = [&](const ConvWeight& cw, __half* dst, int B, int C, int L) {
        cudnnTensorDescriptor_t y_desc, b_desc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&b_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_HALF, B, C, 1, L));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(b_desc, CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_HALF, 1, C, 1, 1));
        float alpha = 1.0f, beta = 1.0f;
        CUDNN_CHECK(cudnnAddTensor(cudnn_, &alpha, b_desc, cw.bias, &beta, y_desc, dst));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(b_desc));
    };

    auto conv1d = [&](const std::string& key, const __half* src, __half* dst,
                      int B, int Cin, int L, int dilation, int groups = 1,
                      int stride = 1) -> int {
        const auto& cw = conv_weights_.at(key);
        int Cout = cw.c0;
        int k_eff = (cw.k - 1) * dilation + 1;
        int pad_left = k_eff - stride;
        int pad_right = 0;
        int paddedL = L + pad_left + pad_right;
        int outL = (L - 1) / stride + 1;

        // Depthwise conv: use custom kernel
        if (groups > 1 && groups == Cin && groups == Cout && dilation == 1 && stride == 1) {
            __half* padded = nullptr;
            CUDA_CHECK(cudaMalloc(&padded, static_cast<size_t>(B) * Cin * paddedL * sizeof(__half)));
            kernels::left_pad_time_fp16(src, padded, B, Cin, L, pad_left, pad_right, stream_);
            kernels::depthwise_conv1d_fp16(padded, cw.weight, cw.bias, dst,
                                           B, Cin, L, cw.k, paddedL, stream_);
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            cudaFree(padded);
            return L;
        }

        __half* padded = nullptr;
        CUDA_CHECK(cudaMalloc(&padded, static_cast<size_t>(B) * Cin * paddedL * sizeof(__half)));
        kernels::left_pad_time_fp16(src, padded, B, Cin, L, pad_left, pad_right, stream_);

        cudnnTensorDescriptor_t x_desc, y_desc;
        cudnnFilterDescriptor_t w_desc;
        cudnnConvolutionDescriptor_t conv_desc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_HALF, B, Cin, 1, paddedL));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_HALF,
                                               CUDNN_TENSOR_NCHW, Cout, Cin, 1, cw.k));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, stride, 1, 1, dilation,
                                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, groups));
        CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_HALF, B, Cout, 1, outL));
        size_t ws_size = 0;
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn_, x_desc, w_desc, conv_desc, y_desc,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, &ws_size));
        void* ws = nullptr;
        if (ws_size) CUDA_CHECK(cudaMalloc(&ws, ws_size));
        float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnConvolutionForward(
            cudnn_, &alpha, x_desc, padded, w_desc, cw.weight, conv_desc,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, ws, ws_size,
            &beta, y_desc, dst));
        if (ws) cudaFree(ws);
        add_bias(cw, dst, B, Cout, outL);
        cudaFree(padded);
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
        CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_desc));
        CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
        return outL;
    };

    auto dense_cf = [&](const std::string& key, const __half* src, __half* dst,
                        int B, int Cin, int L) -> int {
        const auto& dw = dense_weights_.at(key);
        int N = B * L;
        kernels::transpose_fp16(src, decoder_buf_d_, Cin, N, L, Cin, stream_);
        quantized_gemm_fp16(dw.out, N, dw.in, dw.weight, dw.scale,
                            decoder_buf_d_, decoder_buf_c_, cublas_, stream_);
        kernels::transpose_fp16(decoder_buf_c_, dst, N, dw.out, dw.out, L, stream_);
        kernels::add_channel_bias(dst, dw.bias, B, dw.out, L, stream_);
        return dw.out;
    };

    auto convnext = [&](const std::string& prefix, __half* x, int B, int C, int L) {
        // scratch ownership inside this block:
        //   decoder_buf_b_: after dwconv / normalized / pwconv1 / gelu
        //   decoder_buf_d_: pwconv2 result (separate buffer to avoid
        //                    in-place transpose aliasing in dense_cf)
        conv1d(prefix + ".dwconv", x, decoder_buf_b_, B, C, L, 1, C);
        kernels::layer_norm_channels_fp16(decoder_buf_b_,
                                          vector_weights_.at(prefix + ".norm.weight"),
                                          vector_weights_.at(prefix + ".norm.bias"),
                                          decoder_buf_c_, B, C, L, 1e-6f, stream_);
        dense_cf(prefix + ".pwconv1", decoder_buf_c_, decoder_buf_b_, B, C, L);
        kernels::gelu_forward(decoder_buf_b_, B * 4096 * L, stream_);
        // NOTE: write pwconv2 result to decoder_buf_d_ (NOT decoder_buf_c_)
        // because dense_cf internally writes GEMM output to decoder_buf_c_
        // and then transpose reads it — aliasing if dst == decoder_buf_c_.
        dense_cf(prefix + ".pwconv2", decoder_buf_b_, decoder_buf_d_, B, 4096, L);
        kernels::channel_scale_residual(x, decoder_buf_d_,
                                        vector_weights_.at(prefix + ".gamma"),
                                        B, C, L, stream_);
    };

    auto dense_nobias = [&](const std::string& key, const __half* src, __half* dst,
                            int rows, int in_dim) {
        const auto& dw = dense_weights_.at(key);
        quantized_gemm_fp16(dw.out, rows, in_dim, dw.weight, dw.scale, src, dst, cublas_, stream_);
        return dw.out;
    };

    auto post_module = [&](__half* cf, int B, int C, int L) {
        if (B != 1 || C != 1024)
            throw std::runtime_error("DAC post_module currently supports B=1,C=1024");
        const int D = 1024;
        const int H = 16;
        const int HD = 64;
        const int I = 3072;
        const int T = L;
        kernels::transpose_fp16(cf, decoder_buf_b_, D, T, T, D, stream_); // [T,D]
        __half* x = decoder_buf_b_;
        __half* norm = decoder_buf_c_;
        __half* qkv = decoder_buf_d_;
        __half* q = decoder_buf_a_;
        __half* k = q + static_cast<size_t>(D) * T;
        __half* v = k + static_cast<size_t>(D) * T;
        __half* attn_heads = decoder_buf_c_;
        __half* merged = decoder_buf_d_;
        __half* out = decoder_buf_a_;

        size_t attn_ws_elems = static_cast<size_t>(H) * T * T + 2ULL * H * T * HD;
        __half* attn_ws = nullptr;
        CUDA_CHECK(cudaMalloc(&attn_ws, attn_ws_elems * sizeof(__half)));

        for (int l = 0; l < 8; ++l) {
            std::string p = "quantizer.post_module.layers." + std::to_string(l);
            kernels::rms_norm(x, vector_weights_.at(p + ".attention_norm.weight"),
                              norm, T, D, 1e-5f, stream_);
            dense_nobias(p + ".attention.wqkv.weight", norm, qkv, T, D);
            kernels::qkv_split_heads(qkv, q, k, v, T, H, HD, stream_);
            kernels::rope_qk_cis(q, k, vector_weights_.at("quantizer.post_module.freqs_cis"),
                                 H, T, HD, stream_);
            kernels::prefill_attention_gpu(q, k, v, attn_heads, attn_ws,
                                           H, H, T, HD, 1.0f / std::sqrt(static_cast<float>(HD)),
                                           cublas_, stream_, 128 /*window_size*/);
            kernels::merge_heads(attn_heads, merged, T, H, HD, stream_);
            dense_nobias(p + ".attention.wo.weight", merged, out, T, D);
            kernels::dim_scale_residual(x, out,
                                        vector_weights_.at(p + ".attention_layer_scale.gamma"),
                                        T, D, stream_);

            kernels::rms_norm(x, vector_weights_.at(p + ".ffn_norm.weight"),
                              norm, T, D, 1e-5f, stream_);
            dense_nobias(p + ".feed_forward.w1.weight", norm, qkv, T, D);
            dense_nobias(p + ".feed_forward.w3.weight", norm, out, T, D);
            kernels::silu_forward(qkv, T * I, stream_);
            kernels::mul_forward(qkv, out, T * I, stream_);
            dense_nobias(p + ".feed_forward.w2.weight", qkv, out, T, I);
            kernels::dim_scale_residual(x, out,
                                        vector_weights_.at(p + ".ffn_layer_scale.gamma"),
                                        T, D, stream_);
        }

        kernels::rms_norm(x, vector_weights_.at("quantizer.post_module.norm.weight"),
                          norm, T, D, 1e-5f, stream_);
        kernels::transpose_fp16(norm, cf, T, D, D, T, stream_);
        cudaFree(attn_ws);
    };

    auto transconv1d = [&](const std::string& key, const __half* src, __half* dst,
                           int B, int Cin, int L, int stride) -> int {
        const auto& cw = conv_weights_.at(key);
        int Cout = cw.c1;
        int fullL = (L - 1) * stride + cw.k;
        int outL = L * stride;
        __half* out_full = nullptr;
        CUDA_CHECK(cudaMalloc(&out_full, static_cast<size_t>(B) * Cout * fullL * sizeof(__half)));

        cudnnTensorDescriptor_t x_desc, y_desc;
        cudnnFilterDescriptor_t w_desc;
        cudnnConvolutionDescriptor_t conv_desc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_HALF, B, Cin, 1, L));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_HALF,
                                               CUDNN_TENSOR_NCHW, Cin, Cout, 1, cw.k));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, stride, 1, 1,
                                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_HALF, B, Cout, 1, fullL));
        size_t ws_size = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
            cudnn_, w_desc, x_desc, conv_desc, y_desc,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, &ws_size));
        void* ws = nullptr;
        if (ws_size) CUDA_CHECK(cudaMalloc(&ws, ws_size));
        float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnConvolutionBackwardData(
            cudnn_, &alpha, w_desc, cw.weight, x_desc, src, conv_desc,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, ws, ws_size,
            &beta, y_desc, out_full));
        if (ws) cudaFree(ws);
        add_bias(cw, out_full, B, Cout, fullL);
        kernels::crop_time_fp16(out_full, dst, B, Cout, fullL, outL, 0, stream_);
        cudaFree(out_full);
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
        CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_desc));
        CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
        return outL;
    };

    auto snake_cf = [&](const std::string& key, const __half* src, __half* dst,
                        int Bc, int Cc, int Lc) {
        kernels::snake_forward_channels(
            src, dst, vector_weights_.at(key + ".alpha"), Bc, Cc, Lc, stream_);
    };

    auto residual_unit = [&](const std::string& prefix, __half* x,
                             __half* s1, __half* s2, int B, int C, int L, int dilation) {
        snake_cf(prefix + ".block.0", x, s1, B, C, L);
        conv1d(prefix + ".block.1", s1, s2, B, C, L, dilation);
        snake_cf(prefix + ".block.2", s2, s1, B, C, L);
        conv1d(prefix + ".block.3", s1, s2, B, C, L, 1);
        kernels::residual_add(x, s2, B * C * L, stream_);
    };

    int B = batch_size;
    int C = DAC_LATENT_DIM;
    int L = code_len;
    __half* x = decoder_buf_a_;
    __half* s1 = decoder_buf_b_;
    __half* s2 = decoder_buf_c_;

    post_module(x, B, C, L);

    L = transconv1d("quantizer.upsample.0.0", x, decoder_buf_b_, B, C, L, 2);
    CUDA_CHECK(cudaMemcpyAsync(decoder_buf_a_, decoder_buf_b_,
                               static_cast<size_t>(B) * C * L * sizeof(__half),
                               cudaMemcpyDeviceToDevice, stream_));
    x = decoder_buf_a_;
    convnext("quantizer.upsample.0.1", x, B, C, L);

    L = transconv1d("quantizer.upsample.1.0", x, decoder_buf_b_, B, C, L, 2);
    CUDA_CHECK(cudaMemcpyAsync(decoder_buf_a_, decoder_buf_b_,
                               static_cast<size_t>(B) * C * L * sizeof(__half),
                               cudaMemcpyDeviceToDevice, stream_));
    x = decoder_buf_a_;
    convnext("quantizer.upsample.1.1", x, B, C, L);
    s1 = decoder_buf_b_; s2 = decoder_buf_c_;
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    dump_device_f16("upsample", x, {B, C, L});

    conv1d("decoder.model.0", x, s1, B, C, L, 1);
    x = s1; s1 = decoder_buf_a_; s2 = decoder_buf_c_; C = 1536;

    struct BlockSpec { int idx; int in_c; int out_c; int stride; };
    const BlockSpec blocks[] = {
        {1, 1536, 768, 8},
        {2, 768, 384, 8},
        {3, 384, 192, 4},
        {4, 192, 96, 2},
    };
    for (const auto& b : blocks) {
        snake_cf("decoder.model." + std::to_string(b.idx) + ".block.0",
                 x, s1, B, b.in_c, L);
        L = transconv1d("decoder.model." + std::to_string(b.idx) + ".block.1",
                        s1, s2, B, b.in_c, L, b.stride);
        x = s2;
        C = b.out_c;
        __half* r_s1 = (x == decoder_buf_a_) ? decoder_buf_b_ : decoder_buf_a_;
        __half* r_s2 = (x == decoder_buf_c_) ? decoder_buf_b_ : decoder_buf_c_;
        residual_unit("decoder.model." + std::to_string(b.idx) + ".block.2",
                      x, r_s1, r_s2, B, C, L, 1);
        residual_unit("decoder.model." + std::to_string(b.idx) + ".block.3",
                      x, r_s1, r_s2, B, C, L, 3);
        residual_unit("decoder.model." + std::to_string(b.idx) + ".block.4",
                      x, r_s1, r_s2, B, C, L, 9);
        s1 = r_s1;
        s2 = r_s2;
    }

    snake_cf("decoder.model.5", x, s1, B, C, L);
    conv1d("decoder.model.6", s1, s2, B, C, L, 1);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    dump_device_f16("pre_tanh", s2, {B, 1, L});
    kernels::tanh_half_to_float(s2, audio, B * L, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    dump_device_f32("audio", audio, {B, 1, L});
    return L;
}

// ---------------------------------------------------------------------------
// encode_waveform_native() — audio → latent z
// ---------------------------------------------------------------------------
int DACEngine::encode_waveform_native(const float* audio, int batch_size, int audio_len,
                                      float* latent_out) {
    int B = batch_size;
    if (B != 1)
        throw std::runtime_error("DACEngine::encode: native encoder currently supports batch_size=1");
    if (!cudnn_)
        throw std::runtime_error("DACEngine::encode: cuDNN handle is null");
    const int frame_length = cfg_.hop_length() * 4;
    const int padded_audio_len =
        ((audio_len + frame_length - 1) / frame_length) * frame_length;
    int L = padded_audio_len;

    // Ensure decoder scratch + encoder buffers are allocated
    size_t need = static_cast<size_t>(B) * padded_audio_len * 1024;
    if (!decoder_buf_a_ || need > decoder_buf_elems_) {
        if (decoder_buf_a_) { cudaFree(decoder_buf_a_); cudaFree(decoder_buf_b_);
                              cudaFree(decoder_buf_c_); cudaFree(decoder_buf_d_); }
        CUDA_CHECK(cudaMalloc(&decoder_buf_a_, need * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&decoder_buf_b_, need * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&decoder_buf_c_, need * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&decoder_buf_d_, need * sizeof(__half)));
        decoder_buf_elems_ = need;
    }

    // Copy audio to GPU, right-pad to the same frame boundary as Python DAC.encode(),
    // then cast to FP16.
    float* d_audio;
    CUDA_CHECK(cudaMalloc(&d_audio, static_cast<size_t>(B) * padded_audio_len * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_audio, 0,
                               static_cast<size_t>(B) * padded_audio_len * sizeof(float),
                               stream_));
    for (int b = 0; b < B; ++b) {
        CUDA_CHECK(cudaMemcpyAsync(d_audio + static_cast<size_t>(b) * padded_audio_len,
                                   audio + static_cast<size_t>(b) * audio_len,
                                   static_cast<size_t>(audio_len) * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_));
    }
    kernels::f32_to_f16_cast(d_audio, encoder_buf_h_, B * padded_audio_len, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    cudaFree(d_audio);

    bool debug_encoder = std::getenv("FISH_DEBUG_DAC_ENCODER") != nullptr;
    auto log_half_stats = [&](const char* name, const __half* ptr, int Bc, int Cc, int Lc) {
        if (!debug_encoder) return;
        std::vector<__half> h(static_cast<size_t>(Bc) * Cc * Lc);
        CUDA_CHECK(cudaMemcpy(h.data(), ptr, h.size() * sizeof(__half), cudaMemcpyDeviceToHost));
        float mn = 1e30f, mx = -1e30f;
        int n_nan = 0, n_inf = 0;
        for (auto hv : h) {
            float v = __half2float(hv);
            if (std::isnan(v)) { n_nan++; continue; }
            if (std::isinf(v)) { n_inf++; continue; }
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        spdlog::info("DAC encoder {} stats: shape=[{},{},{}] min={:.6f} max={:.6f} nan={} inf={}",
                     name, Bc, Cc, Lc, mn, mx, n_nan, n_inf);
    };
    log_half_stats("input", encoder_buf_h_, B, 1, L);

    // Shared helpers
    auto enc_snake_cf = [&](const std::string& key, const __half* src, __half* dst,
                            int Bc, int Cc, int Lc) {
        kernels::snake_forward_channels(
            src, dst, vector_weights_.at(key + ".alpha"), Bc, Cc, Lc, stream_);
    };

    auto dense_nobias = [&](const std::string& key, const __half* src, __half* dst,
                            int rows, int in_dim) {
        const auto& dw = dense_weights_.at(key);
        quantized_gemm_fp16(dw.out, rows, in_dim, dw.weight, dw.scale, src, dst, cublas_, stream_);
        return dw.out;
    };

    // Custom conv (no cuDNN, handles dilation correctly)
    auto enc_conv1d = [&](const std::string& key, const __half* src, __half* dst,
                          int Bc, int Cin, int Lc, int dilation, int groups, int stride) -> int {
        const auto& cw = conv_weights_.at(key);
        int Cout = cw.c0, k = cw.k;
        int k_eff = (k - 1) * dilation + 1;
        int outL = (Lc + stride - 1) / stride;
        int pad_left = std::max(0, k_eff - stride);
        int pad_right = std::max(0, (outL - 1) * stride + k_eff - pad_left - Lc);
        int paddedL = Lc + pad_left + pad_right;

        __half* padded = nullptr;
        CUDA_CHECK(cudaMalloc(&padded, static_cast<size_t>(Bc) * Cin * paddedL * sizeof(__half)));
        kernels::left_pad_time_fp16(src, padded, Bc, Cin, Lc, pad_left, pad_right, stream_);

        if (groups > 1 && groups == Cin && groups == Cout && dilation == 1 && stride == 1) {
            kernels::depthwise_conv1d_fp16(padded, cw.weight, cw.bias, dst,
                                           Bc, Cin, Lc, k, paddedL, stream_);
        } else if (Cin == 1 && groups == 1 && stride == 1 && dilation == 1) {
            kernels::conv1d_cin1_fp16(padded, cw.weight, cw.bias, dst,
                                      Bc, Cout, Lc, k, paddedL, stream_);
        } else {
            kernels::strided_conv1d_fp16(padded, cw.weight, cw.bias, dst,
                                         Bc, Cin, Cout, Lc, outL, k, stride, dilation, paddedL, stream_);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        cudaFree(padded);
        return outL;
    };

    auto enc_dense_cf = [&](const std::string& key, const __half* src, __half* dst,
                            int Bc, int Cin, int Lc) -> int {
        const auto& dw = dense_weights_.at(key);
        int N = Bc * Lc;
        kernels::transpose_fp16(src, decoder_buf_d_, Cin, N, Lc, Cin, stream_);
        quantized_gemm_fp16(dw.out, N, dw.in, dw.weight, dw.scale,
                            decoder_buf_d_, decoder_buf_c_, cublas_, stream_);
        kernels::transpose_fp16(decoder_buf_c_, dst, N, dw.out, dw.out, Lc, stream_);
        kernels::add_channel_bias(dst, dw.bias, Bc, dw.out, Lc, stream_);
        return dw.out;
    };

    auto enc_convnext = [&](const std::string& prefix, const std::string& dump_base,
                            __half* cx, int Bc, int Cc, int Lc) {
        enc_conv1d(prefix + ".dwconv", cx, decoder_buf_b_, Bc, Cc, Lc, 1, Cc, 1);
        dump_device_f16(dump_base + "_dwconv", decoder_buf_b_, {Bc, Cc, Lc});
        kernels::layer_norm_channels_fp16(decoder_buf_b_,
            vector_weights_.at(prefix + ".norm.weight"),
            vector_weights_.at(prefix + ".norm.bias"),
            decoder_buf_c_, Bc, Cc, Lc, 1e-6f, stream_);
        dump_device_f16(dump_base + "_norm", decoder_buf_c_, {Bc, Cc, Lc});
        enc_dense_cf(prefix + ".pwconv1", decoder_buf_c_, decoder_buf_b_, Bc, Cc, Lc);
        dump_device_f16(dump_base + "_pwconv1", decoder_buf_b_, {Bc, 4 * Cc, Lc});
        kernels::gelu_forward(decoder_buf_b_, Bc * 4 * Cc * Lc, stream_);
        dump_device_f16(dump_base + "_gelu", decoder_buf_b_, {Bc, 4 * Cc, Lc});
        enc_dense_cf(prefix + ".pwconv2", decoder_buf_b_, decoder_buf_d_, Bc, 4 * Cc, Lc);
        dump_device_f16(dump_base + "_pwconv2", decoder_buf_d_, {Bc, Cc, Lc});
        kernels::channel_scale_residual(cx, decoder_buf_d_,
            vector_weights_.at(prefix + ".gamma"), Bc, Cc, Lc, stream_);
        dump_device_f16(dump_base + "_out", cx, {Bc, Cc, Lc});
    };

    auto enc_transformer_cf = [&](const std::string& prefix, __half* cf,
                                  int Bc, int Cc, int Lc, int n_layers,
                                  int window_size) {
        if (Bc != 1 || Cc % 64 != 0)
            throw std::runtime_error("DAC encoder transformer currently supports B=1,C%64=0");

        const int D = Cc;
        const int H = Cc / 64;
        const int HD = 64;
        const int I = Cc * 3;
        const int T = Lc;
        kernels::transpose_fp16(cf, decoder_buf_b_, D, T, T, D, stream_);
        __half* tx = decoder_buf_b_;
        __half* norm = decoder_buf_c_, *qkv = decoder_buf_d_;
        __half* q = decoder_buf_a_, *k = q + static_cast<size_t>(D) * T;
        __half* v = k + static_cast<size_t>(D) * T;
        __half* attn_heads = decoder_buf_c_, *merged = decoder_buf_d_, *out = decoder_buf_a_;

        size_t aws = static_cast<size_t>(H) * T * T + 2ULL * H * T * HD;
        __half* attn_ws = nullptr;
        CUDA_CHECK(cudaMalloc(&attn_ws, aws * sizeof(__half)));

        for (int l = 0; l < n_layers; ++l) {
            std::string pp = prefix + ".layers." + std::to_string(l);
            kernels::rms_norm(tx, vector_weights_.at(pp + ".attention_norm.weight"),
                              norm, T, D, 1e-5f, stream_);
            dense_nobias(pp + ".attention.wqkv.weight", norm, qkv, T, D);
            kernels::qkv_split_heads(qkv, q, k, v, T, H, HD, stream_);
            kernels::rope_qk_cis(q, k, vector_weights_.at(prefix + ".freqs_cis"),
                                 H, T, HD, stream_);
            kernels::prefill_attention_gpu(q, k, v, attn_heads, attn_ws,
                                           H, H, T, HD, 1.0f / sqrtf(static_cast<float>(HD)),
                                           cublas_, stream_, window_size);
            kernels::merge_heads(attn_heads, merged, T, H, HD, stream_);
            dense_nobias(pp + ".attention.wo.weight", merged, out, T, D);
            kernels::dim_scale_residual(tx, out,
                vector_weights_.at(pp + ".attention_layer_scale.gamma"), T, D, stream_);
            kernels::rms_norm(tx, vector_weights_.at(pp + ".ffn_norm.weight"),
                              norm, T, D, 1e-5f, stream_);
            dense_nobias(pp + ".feed_forward.w1.weight", norm, qkv, T, D);
            dense_nobias(pp + ".feed_forward.w3.weight", norm, out, T, D);
            kernels::silu_mul_clamp_forward(qkv, out, T * I, 65504.0f, stream_);
            dense_nobias(pp + ".feed_forward.w2.weight", qkv, out, T, I);
            kernels::dim_scale_residual(tx, out,
                vector_weights_.at(pp + ".ffn_layer_scale.gamma"), T, D, stream_);
            log_half_stats(("pre_module.layer" + std::to_string(l)).c_str(), tx, B, D, T);
        }
        kernels::rms_norm(tx, vector_weights_.at(prefix + ".norm.weight"),
                          norm, T, D, 1e-5f, stream_);
        kernels::transpose_fp16(norm, cf, T, D, D, T, stream_);
        cudaFree(attn_ws);
    };

    __half* x  = encoder_buf_h_;
    __half* s1 = decoder_buf_a_;
    __half* s2 = decoder_buf_b_;

    // Diagnostic: dump input audio (FP16→FP32)
    {
        std::string prefix = dac_dump_prefix();
        if (!prefix.empty()) {
            std::vector<__half> hh(B * L);
            CUDA_CHECK(cudaMemcpy(hh.data(), x, B * L * sizeof(__half), cudaMemcpyDeviceToHost));
            float mn = 1e30f, mx = -1e30f; double sm = 0;
            for (auto v : hh) { float f = __half2float(v); if (f < mn) mn = f; if (f > mx) mx = f; sm += f; }
            spdlog::info("  [DIAG] input-audio: shape=[{},{}] min={:.4f} max={:.4f} mean={:.4f} "
                         "first5=[{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}]",
                         B, L, mn, mx, sm/hh.size(),
                         __half2float(hh[0]), __half2float(hh[1]), __half2float(hh[2]),
                         __half2float(hh[3]), __half2float(hh[4]));
        }
    }

    // === Step 1: encoder.block.0 (conv 1→64, k=7) ===
    L = enc_conv1d("encoder.block.0", x, s1, B, 1, L, 1, 1, 1);
    x = s1; s1 = decoder_buf_c_; s2 = decoder_buf_d_;
    int C = 64;
    log_half_stats("block0", x, B, C, L);
    dump_device_f16("enc_block0", x, {B, C, L});

    // === Step 2: encoder blocks 1-4 (residual units + strided downsample) ===
    struct EncBlock { int idx; int out_c; int stride; };
    const EncBlock eblocks[] = {
        {1, 128, 2}, {2, 256, 4}, {3, 512, 8}, {4, 1024, 8},
    };
    for (const auto& eb : eblocks) {
        std::string p = "encoder.block." + std::to_string(eb.idx);
        int dilations[] = {1, 3, 9};
        for (int ru = 0; ru < 3; ++ru) {
            std::string rp = p + ".block." + std::to_string(ru);
            enc_snake_cf(rp + ".block.0", x, s1, B, C, L);
            enc_conv1d(rp + ".block.1", s1, s2, B, C, L, dilations[ru], 1, 1);
            enc_snake_cf(rp + ".block.2", s2, s1, B, C, L);
            enc_conv1d(rp + ".block.3", s1, s2, B, C, L, 1, 1, 1);
            kernels::residual_add(x, s2, B * C * L, stream_);
            __half* tmp = s1; s1 = s2; s2 = tmp;
        }
        enc_snake_cf(p + ".block.3", x, s1, B, C, L);
        L = enc_conv1d(p + ".block.4", s1, s2, B, C, L, 1, 1, eb.stride);
        x = s2; C = eb.out_c;
        if (eb.idx == 4 && vector_weights_.count("encoder.block.4.block.5.norm.weight")) {
            CUDA_CHECK(cudaMemcpyAsync(encoder_buf_h_, x,
                                       static_cast<size_t>(B) * C * L * sizeof(__half),
                                       cudaMemcpyDeviceToDevice, stream_));
            x = encoder_buf_h_;
            enc_transformer_cf("encoder.block.4.block.5", x, B, C, L, 4, 512);
        }
        s1 = (x == decoder_buf_a_) ? decoder_buf_b_ : decoder_buf_a_;
        s2 = (x == decoder_buf_c_) ? decoder_buf_d_ : decoder_buf_c_;
        log_half_stats(("block" + std::to_string(eb.idx)).c_str(), x, B, C, L);
        dump_device_f16("enc_block" + std::to_string(eb.idx), x, {B, C, L});
    }

    // === Step 3: encoder.block.6 (final encoder conv, if present) ===
    if (conv_weights_.count("encoder.block.6")) {
        enc_snake_cf("encoder.block.5", x, s1, B, C, L);
        enc_conv1d("encoder.block.6", s1, s2, B, C, L, 1, 1, 1);
        x = s2; s1 = (x == decoder_buf_a_) ? decoder_buf_b_ : decoder_buf_a_;
        log_half_stats("block6", x, B, C, L);
        dump_device_f16("enc_block6", x, {B, C, L});
    }

    // Diagnostic: dump after encoder blocks, before quantizer downsample
    {
        std::string prefix = dac_dump_prefix();
        if (!prefix.empty()) {
            std::vector<__half> hh(static_cast<size_t>(B) * C * L);
            CUDA_CHECK(cudaMemcpy(hh.data(), x, hh.size() * sizeof(__half), cudaMemcpyDeviceToHost));
            float mn = 1e30f, mx = -1e30f;
            double sm = 0, s2 = 0;
            int nans = 0;
            for (auto v : hh) {
                float fv = __half2float(v);
                if (std::isnan(fv)) { nans++; continue; }
                if (fv < mn) mn = fv; if (fv > mx) mx = fv;
                sm += fv; s2 += (double)fv * fv;
            }
            size_t valid = hh.size() - nans;
            spdlog::info("  [DIAG] post-encoder: shape=[{},{},{}] min={:.4f} max={:.4f} NaN={} mean={:.4f} std={:.4f}",
                         B, C, L, mn, mx, nans,
                         valid ? sm/valid : 0.0, valid ? std::sqrt(s2/valid) : 0.0);
        }
    }

    // === Step 4: quantizer.downsample (2 ConvNeXt stages, stride 2) ===
    for (int i = 0; i < 2; ++i) {
        std::string dp = "quantizer.downsample." + std::to_string(i);
        L = enc_conv1d(dp + ".0", x, s1, B, C, L, 1, 1, 2);
        CUDA_CHECK(cudaMemcpyAsync(encoder_buf_h_, s1, static_cast<size_t>(B) * C * L * sizeof(__half),
                                   cudaMemcpyDeviceToDevice, stream_));
        x = encoder_buf_h_;
        dump_device_f16("enc_downsample" + std::to_string(i) + "_conv", x, {B, C, L});
        enc_convnext(dp + ".1", "enc_downsample" + std::to_string(i), x, B, C, L);
        s1 = (x == decoder_buf_a_) ? decoder_buf_b_ : decoder_buf_a_;
        s2 = (x == decoder_buf_c_) ? decoder_buf_d_ : decoder_buf_c_;
        log_half_stats(("downsample" + std::to_string(i)).c_str(), x, B, C, L);
        dump_device_f16("enc_downsample" + std::to_string(i), x, {B, C, L});
    }

    // === Diagnostic: dump post-downsample, pre-pre_module latent ===
    {
        std::string prefix = dac_dump_prefix();
        if (!prefix.empty()) {
            std::vector<__half> hh(static_cast<size_t>(B) * C * L);
            CUDA_CHECK(cudaMemcpy(hh.data(), x, hh.size() * sizeof(__half), cudaMemcpyDeviceToHost));
            std::vector<float> hf(hh.size());
            float mn = 1e30f, mx = -1e30f; double sm = 0, s2 = 0; int nans = 0;
            for (size_t i = 0; i < hh.size(); ++i) {
                float fv = __half2float(hh[i]);
                if (std::isnan(fv)) { nans++; hf[i] = 0; continue; }
                hf[i] = fv;
                if (fv < mn) mn = fv; if (fv > mx) mx = fv;
                sm += fv; s2 += (double)fv * fv;
            }
            size_t valid = hh.size() - nans;
            write_f32_tensor(prefix + "_cpp_enc.bin", {B, C, L}, hf.data(), hh.size());
            spdlog::info("  [DIAG] post-downsample: shape=[{},{},{}] min={:.4f} max={:.4f} NaN={} mean={:.4f} std={:.4f}",
                         B, C, L, mn, mx, nans,
                         valid ? sm/valid : 0.0, valid ? std::sqrt(s2/valid) : 0.0);
        }
    }

    // === Step 5: pre_module (8-layer transformer) ===
    {
        const int D = 1024, H = 16, HD = 64, I = 3072, T = L;
        if (B != 1 || C != 1024)
            throw std::runtime_error("DAC pre_module currently supports B=1,C=1024");

        kernels::transpose_fp16(x, decoder_buf_b_, D, T, T, D, stream_);
        __half* tx = decoder_buf_b_;
        __half* norm = decoder_buf_c_, *qkv = decoder_buf_d_;
        __half* q = decoder_buf_a_, *k = q + (size_t)D * T, *v = k + (size_t)D * T;
        __half* attn_heads = decoder_buf_c_, *merged = decoder_buf_d_, *out = decoder_buf_a_;

        size_t aws = (size_t)H * T * T + 2ULL * H * T * HD;
        __half* attn_ws = nullptr;
        CUDA_CHECK(cudaMalloc(&attn_ws, aws * sizeof(__half)));
        float* ffn_out_f32 = nullptr;
        CUDA_CHECK(cudaMalloc(&ffn_out_f32, static_cast<size_t>(T) * D * sizeof(float)));

        for (int l = 0; l < 8; ++l) {
            std::string pp = "quantizer.pre_module.layers." + std::to_string(l);
            kernels::rms_norm(tx, vector_weights_.at(pp + ".attention_norm.weight"),
                              norm, T, D, 1e-5f, stream_);
            dense_nobias(pp + ".attention.wqkv.weight", norm, qkv, T, D);
            if (l == 0) {
                dump_device_f16("enc_pre_module_layer0_attention_norm", norm, {B, T, D});
                dump_device_f16("enc_pre_module_layer0_wqkv", qkv, {B, T, 3 * D});
            }
            kernels::qkv_split_heads(qkv, q, k, v, T, H, HD, stream_);
            kernels::rope_qk_cis(q, k, vector_weights_.at("quantizer.pre_module.freqs_cis"),
                                 H, T, HD, stream_);
            kernels::prefill_attention_gpu(q, k, v, attn_heads, attn_ws,
                                           H, H, T, HD, 1.0f/sqrtf((float)HD),
                                           cublas_, stream_, 128 /*window_size*/);
            kernels::merge_heads(attn_heads, merged, T, H, HD, stream_);
            dense_nobias(pp + ".attention.wo.weight", merged, out, T, D);
            dump_device_f16("enc_pre_module_layer" + std::to_string(l) + "_attention", out, {B, T, D});
            kernels::dim_scale_residual(tx, out,
                vector_weights_.at(pp + ".attention_layer_scale.gamma"), T, D, stream_);
            kernels::rms_norm(tx, vector_weights_.at(pp + ".ffn_norm.weight"),
                              norm, T, D, 1e-5f, stream_);
            dense_nobias(pp + ".feed_forward.w1.weight", norm, qkv, T, D);
            dense_nobias(pp + ".feed_forward.w3.weight", norm, out, T, D);
            kernels::silu_mul_clamp_forward(qkv, out, T * I, 65504.0f, stream_);
            {
                const auto& dw = dense_weights_.at(pp + ".feed_forward.w2.weight");
                if (dw.scale != nullptr) {
                    // INT8 path: compute FP16 result, then cast to FP32
                    quantized_gemm_fp16(dw.out, T, I, dw.weight, dw.scale,
                                        qkv, decoder_buf_c_, cublas_, stream_);
                    kernels::f16_to_f32_cast(decoder_buf_c_, ffn_out_f32,
                                             T * dw.out, stream_);
                } else {
                    gemm_fp16_to_f32(dw.out, T, I, dw.weight, qkv, ffn_out_f32, cublas_);
                }
            }
            dump_device_f32("enc_pre_module_layer" + std::to_string(l) + "_ffn", ffn_out_f32, {B, T, D});
            kernels::dim_scale_residual_from_f32(tx, ffn_out_f32,
                vector_weights_.at(pp + ".ffn_layer_scale.gamma"), T, D, 65504.0f, stream_);
            log_half_stats(("quantizer.pre_module.layer" + std::to_string(l)).c_str(), tx, B, D, T);
            dump_device_f16("enc_pre_module_layer" + std::to_string(l), tx, {B, T, D});
        }
        kernels::rms_norm(tx, vector_weights_.at("quantizer.pre_module.norm.weight"),
                          norm, T, D, 1e-5f, stream_);
        kernels::transpose_fp16(norm, x, T, D, D, T, stream_);
        cudaFree(ffn_out_f32);
        cudaFree(attn_ws);
        dump_device_f16("enc_pre_module", x, {B, D, T});
    }

    // Copy latent to output
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    kernels::f16_to_f32_cast(x, latent_out, B * C * L, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    return L;
}

void DACEngine::decode(const int32_t* codes, int batch_size, int code_len,
                       float* audio, int* audio_len, int max_audio_cap) {

    // ── Phase 1: GPU RVQ lookup (when codebook weights are loaded) ──────
    bool have_gpu_rvq = !gpu_codebooks_.empty() && gpu_codebooks_[0] != nullptr;
    if (have_gpu_rvq) {
        // Zero the latent accumulator
        size_t latent_bytes = (size_t)batch_size * DAC_LATENT_DIM * code_len * sizeof(float);
        CUDA_CHECK(cudaMemsetAsync(latent_buf_, 0, latent_bytes, stream_));

        // Accumulate each codebook's contribution
        for (int cb = 0; cb < DAC_TOTAL_CODEBOOKS; cb++) {
            if (!gpu_codebooks_[cb] || !gpu_out_proj_w_[cb] || !gpu_out_proj_b_[cb])
                continue;
            int cb_size = (cb == 0) ? DAC_SEMANTIC_CODEBOOK_SIZE : DAC_ACOUSTIC_CODEBOOK_SIZE;
            // codes for this codebook: [B, T] slice from codes [B, N, T]
            const int32_t* cb_codes = codes + cb * code_len;  // B=1, so flat
            kernels::rvq_step_per_codebook(cb_codes, gpu_codebooks_[cb],
                                  gpu_out_proj_w_[cb],
                                  gpu_out_proj_b_[cb],
                                  latent_buf_, batch_size, code_len,
                                  cb_size, DAC_CODEBOOK_DIM, DAC_LATENT_DIM, stream_);
        }
    }

    (void)codes;
    *audio_len = decode_waveform_native(batch_size, code_len, audio, max_audio_cap);
    spdlog::info("DACEngine::decode: {} code frames → {} samples ({:.2f}s)",
                 code_len, *audio_len,
                 static_cast<double>(*audio_len) / cfg_.sample_rate);
}

// ---------------------------------------------------------------------------
// encode() — audio → codes (encoder + RVQ quantization)
// ---------------------------------------------------------------------------
void DACEngine::encode(const float* audio, int batch_size, int audio_len,
                       int32_t* codes, int* code_len) {
    int B = batch_size;
    if (B != 1)
        throw std::runtime_error("DACEngine::encode: native encoder currently supports batch_size=1");
    if (!cudnn_)
        throw std::runtime_error("DACEngine::encode: cuDNN handle is null");
    const int frame_length = cfg_.hop_length() * 4;
    const int padded_audio_len =
        ((audio_len + frame_length - 1) / frame_length) * frame_length;

    // Pre-allocate encoder buffers (encode_waveform_native latches latent_out to encoder_buf_)
    size_t enc_need = static_cast<size_t>(B) * padded_audio_len * 1024;
    if (enc_need > encoder_buf_elems_) {
        if (encoder_buf_h_) cudaFree(encoder_buf_h_);
        if (encoder_buf_)  cudaFree(encoder_buf_);
        CUDA_CHECK(cudaMalloc(&encoder_buf_h_, enc_need * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&encoder_buf_,  enc_need * sizeof(float)));
        encoder_buf_elems_ = enc_need;
    }

    // Step 1: encoder forward (audio → latent z)
    int T_code = encode_waveform_native(audio, B, audio_len, encoder_buf_);
    // encoder_buf_ now holds latent z [B, 1024, T_code]
    if (std::getenv("FISH_DEBUG_DAC_ENCODER")) {
        std::vector<float> h_lat(static_cast<size_t>(B) * DAC_LATENT_DIM * T_code);
        CUDA_CHECK(cudaMemcpy(h_lat.data(), encoder_buf_,
                              h_lat.size() * sizeof(float), cudaMemcpyDeviceToHost));
        float mn = 1e30f, mx = -1e30f;
        int n_nan = 0, n_inf = 0;
        for (float v : h_lat) {
            if (std::isnan(v)) { n_nan++; continue; }
            if (std::isinf(v)) { n_inf++; continue; }
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        spdlog::info("DACEngine::encode latent stats: min={:.6f} max={:.6f} nan={} inf={}",
                     mn, mx, n_nan, n_inf);
    }

    // Step 2: RVQ quantization on GPU, copy codes back to CPU
    {
        int32_t* d_codes;
        size_t cb_bytes = static_cast<size_t>(DAC_TOTAL_CODEBOOKS) * T_code * sizeof(int32_t);
        CUDA_CHECK(cudaMalloc(&d_codes, cb_bytes));
        CUDA_CHECK(cudaMemsetAsync(d_codes, 0, cb_bytes, stream_));

        float *d_res, *d_next;
        size_t res_b = static_cast<size_t>(B) * DAC_LATENT_DIM * T_code * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_res, res_b));
        CUDA_CHECK(cudaMalloc(&d_next, res_b));
        CUDA_CHECK(cudaMemcpyAsync(d_res, encoder_buf_, res_b, cudaMemcpyDeviceToDevice, stream_));

        for (int cb = 0; cb < DAC_TOTAL_CODEBOOKS; cb++) {
            if (!gpu_codebooks_[cb] || !gpu_in_proj_w_[cb] || !gpu_in_proj_b_[cb] ||
                !gpu_out_proj_w_[cb] || !gpu_out_proj_b_[cb])
                throw std::runtime_error("DACEngine::encode: missing RVQ encode tensors for codebook " + std::to_string(cb));
            int cb_size = (cb == 0) ? DAC_SEMANTIC_CODEBOOK_SIZE : DAC_ACOUSTIC_CODEBOOK_SIZE;
            int32_t* cb_d = d_codes + static_cast<size_t>(cb) * T_code;
            float* r_in  = (cb % 2 == 0) ? d_res : d_next;
            float* r_out = (cb % 2 == 0) ? d_next : d_res;

            kernels::rvq_encode_step(
                r_in, gpu_in_proj_w_[cb], gpu_in_proj_b_[cb],
                gpu_codebooks_[cb], gpu_out_proj_w_[cb], gpu_out_proj_b_[cb],
                cb_d, r_out,
                B, T_code, cb_size, DAC_CODEBOOK_DIM, DAC_LATENT_DIM, stream_);
        }

        CUDA_CHECK(cudaMemcpy(codes, d_codes, cb_bytes, cudaMemcpyDeviceToHost));
        cudaFree(d_codes); cudaFree(d_res); cudaFree(d_next);
    }

    *code_len = T_code;
    spdlog::info("DACEngine::encode: {} samples → {} code frames", audio_len, T_code);
}

}  // namespace fish
