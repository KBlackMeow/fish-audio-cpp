// src/engine/dual_ar_engine.cc — Qwen3 DualAR inference engine
#include "engine/dual_ar_engine.h"
#include "utils/cuda_utils.h"
#include "kernels/kernels.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <cstring>

namespace fish {

// ============================================================================
// GEMM helper: Y[M,N] = X[M,K] × W[N,K]^T   (all row-major)
//
// cuBLAS produces col-major output which is byte-identical to our row-major
// result.  Call as: gemm(M_out=N, N_out=M, K=K, W_weight[N,K], X_input[M,K], Y_out)
// ============================================================================
static void gemm_fp16(int M_out, int N_out, int K,
                      const __half* W, const __half* X, __half* Y,
                      cublasHandle_t cublas)
{
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        M_out, N_out, K,
        &alpha,
        W, CUDA_R_16F, K,     // W is [M_out, K] row-major
        X, CUDA_R_16F, K,     // X is [N_out, K] row-major
        &beta,
        Y, CUDA_R_16F, M_out, // ldc=M_out: col-major [M_out, N_out] = row-major [N_out, M_out]
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

// ============================================================================
// Constructor — bind TensorViews from ModelLoader
// ============================================================================
DualAREngine::DualAREngine(const DualARConfig& cfg, ModelLoader&& loader,
                           cublasHandle_t cublas, cudaStream_t stream)
    : cfg_(cfg), loader_(std::move(loader)), cublas_(cublas), stream_(stream)
{
    for (int l = 0; l < cfg_.n_layer; l++) {
        auto p = "text_model.model.layers." + std::to_string(l) + ".";
        TextLayerWeights lw;
        lw.wqkv      = loader_.get(p + "attention.wqkv.weight");
        lw.wo        = loader_.get(p + "attention.wo.weight");
        lw.q_norm    = loader_.get(p + "attention.q_norm.weight");
        lw.k_norm    = loader_.get(p + "attention.k_norm.weight");
        lw.attn_norm = loader_.get(p + "attention_norm.weight");
        lw.w_gate    = loader_.get(p + "feed_forward.w1.weight");
        lw.w_down    = loader_.get(p + "feed_forward.w2.weight");
        lw.w_up      = loader_.get(p + "feed_forward.w3.weight");
        lw.ffn_norm  = loader_.get(p + "ffn_norm.weight");
        layers_.push_back(std::move(lw));
    }
    w_embedding_ = loader_.get("text_model.model.embeddings.weight");
    w_text_norm_ = loader_.get("text_model.model.norm.weight");

    for (int l = 0; l < cfg_.n_fast_layer; l++) {
        auto p = "audio_decoder.layers." + std::to_string(l) + ".";
        FastLayerWeights fl;
        fl.wqkv      = loader_.get(p + "attention.wqkv.weight");
        fl.wo        = loader_.get(p + "attention.wo.weight");
        fl.attn_norm = loader_.get(p + "attention_norm.weight");
        fl.w_gate    = loader_.get(p + "feed_forward.w1.weight");
        fl.w_down    = loader_.get(p + "feed_forward.w2.weight");
        fl.w_up      = loader_.get(p + "feed_forward.w3.weight");
        fl.ffn_norm  = loader_.get(p + "ffn_norm.weight");
        fast_layers_.push_back(std::move(fl));
    }
    w_fast_embeddings_      = loader_.get("audio_decoder.embeddings.weight");
    w_codebook_embeddings_  = loader_.get("audio_decoder.codebook_embeddings.weight");
    w_fast_norm_            = loader_.get("audio_decoder.norm.weight");
    w_fast_output_          = loader_.get("audio_decoder.output.weight");
}

// ============================================================================
// Destructor
// ============================================================================
DualAREngine::~DualAREngine() {
    auto sf = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    sf(reinterpret_cast<void*&>(rope_freqs_));
    sf(reinterpret_cast<void*&>(fast_rope_freqs_));
    sf(reinterpret_cast<void*&>(fast_k_cache_));
    sf(reinterpret_cast<void*&>(fast_v_cache_));
    sf(reinterpret_cast<void*&>(ws1_)); sf(reinterpret_cast<void*&>(ws2_));
    sf(reinterpret_cast<void*&>(ws3_));
    sf(reinterpret_cast<void*&>(q_buf_)); sf(reinterpret_cast<void*&>(k_buf_));
    sf(reinterpret_cast<void*&>(v_buf_)); sf(reinterpret_cast<void*&>(attn_out_buf_));
    sf(reinterpret_cast<void*&>(expanded_k_buf_));
    sf(reinterpret_cast<void*&>(attn_ws_));
    sf(reinterpret_cast<void*&>(cb_idx_buf_));
    sf(reinterpret_cast<void*&>(d_scratch_int_));
    sf(cublas_ws_);
    for (void* p : gpu_weight_ptrs_) { if (p) cudaFree(p); }
}

// ============================================================================
// init() — allocate workspace, RoPE freqs, fast KV cache, copy weights → GPU
// ============================================================================
void DualAREngine::init() {
    int dim = cfg_.dim;

    // Text model RoPE frequencies — per head_dim (kernel reads freqs[pair % (head_dim/2)])
    int D = cfg_.head_dim;
    std::vector<float> hf(D / 2);
    for (int i = 0; i < D / 2; i++)
        hf[i] = 1.0f / std::pow(cfg_.rope_base, (2.0f * i) / D);
    CUDA_CHECK(cudaMalloc(&rope_freqs_, (D / 2) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(rope_freqs_, hf.data(), (D / 2) * sizeof(float), cudaMemcpyHostToDevice));

    // Fast decoder RoPE frequencies (uses fast_head_dim)
    int fD = cfg_.fast_head_dim;
    std::vector<float> fhf(fD / 2);
    for (int i = 0; i < fD / 2; i++)
        fhf[i] = 1.0f / std::pow(cfg_.rope_base, (2.0f * i) / fD);
    CUDA_CHECK(cudaMalloc(&fast_rope_freqs_, (fD / 2) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(fast_rope_freqs_, fhf.data(), (fD / 2) * sizeof(float), cudaMemcpyHostToDevice));

    // Fast decoder KV cache: [n_fast_layer, n_kv_heads, num_codebooks, head_dim]
    // num_codebooks+1 for the initial fast_forward at position 0 before semantic sampling
    size_t fkv_sz = (size_t)cfg_.n_fast_layer * cfg_.fast_n_local_heads
                    * (cfg_.num_codebooks + 1) * fD * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&fast_k_cache_, fkv_sz));
    CUDA_CHECK(cudaMalloc(&fast_v_cache_, fkv_sz));
    CUDA_CHECK(cudaMemset(fast_k_cache_, 0, fkv_sz));
    CUDA_CHECK(cudaMemset(fast_v_cache_, 0, fkv_sz));

    ensure_workspace_tokens(256);

    // cuBLAS workspace
    cublas_ws_size_ = 16 * 1024 * 1024;
    CUDA_CHECK(cudaMalloc(&cublas_ws_, cublas_ws_size_));
    CUBLAS_CHECK(cublasSetWorkspace(cublas_, cublas_ws_, cublas_ws_size_));
    CUBLAS_CHECK(cublasSetStream(cublas_, stream_));

    // Copy all weights to GPU (FP16 or INT8 with per-channel scales)
    struct NamedTV { std::string name; TensorView* tv; };
    std::vector<NamedTV> named_weights;

    named_weights.push_back({"text_model.model.embeddings.weight", &w_embedding_});
    named_weights.push_back({"text_model.model.norm.weight", &w_text_norm_});
    named_weights.push_back({"audio_decoder.embeddings.weight", &w_fast_embeddings_});
    named_weights.push_back({"audio_decoder.codebook_embeddings.weight", &w_codebook_embeddings_});
    named_weights.push_back({"audio_decoder.norm.weight", &w_fast_norm_});
    named_weights.push_back({"audio_decoder.output.weight", &w_fast_output_});

    for (int l = 0; l < cfg_.n_layer; l++) {
        auto p = "text_model.model.layers." + std::to_string(l) + ".";
        auto& lw = layers_[l];
        named_weights.push_back({p + "attention.wqkv.weight", &lw.wqkv});
        named_weights.push_back({p + "attention.wo.weight", &lw.wo});
        named_weights.push_back({p + "attention.q_norm.weight", &lw.q_norm});
        named_weights.push_back({p + "attention.k_norm.weight", &lw.k_norm});
        named_weights.push_back({p + "attention_norm.weight", &lw.attn_norm});
        named_weights.push_back({p + "ffn_norm.weight", &lw.ffn_norm});
        named_weights.push_back({p + "feed_forward.w1.weight", &lw.w_gate});
        named_weights.push_back({p + "feed_forward.w3.weight", &lw.w_up});
        named_weights.push_back({p + "feed_forward.w2.weight", &lw.w_down});
    }

    for (int l = 0; l < cfg_.n_fast_layer; l++) {
        auto p = "audio_decoder.layers." + std::to_string(l) + ".";
        auto& fl = fast_layers_[l];
        named_weights.push_back({p + "attention.wqkv.weight", &fl.wqkv});
        named_weights.push_back({p + "attention.wo.weight", &fl.wo});
        named_weights.push_back({p + "attention_norm.weight", &fl.attn_norm});
        named_weights.push_back({p + "ffn_norm.weight", &fl.ffn_norm});
        named_weights.push_back({p + "feed_forward.w1.weight", &fl.w_gate});
        named_weights.push_back({p + "feed_forward.w3.weight", &fl.w_up});
        named_weights.push_back({p + "feed_forward.w2.weight", &fl.w_down});
    }

    size_t tg = 0;
    for (auto& nw : named_weights) {
        TensorView* tv = nw.tv;
        size_t nb = tv->nbytes();
        void* gp = nullptr;
        CUDA_CHECK(cudaMalloc(&gp, nb));
        CUDA_CHECK(cudaMemcpy(gp, tv->data, nb, cudaMemcpyHostToDevice));
        gpu_weight_ptrs_.push_back(gp);

        // Look up INT8 per-channel scale
        std::string scale_name = nw.name + "_scale";
        __half* scale_gp = nullptr;
        if (loader_.has(scale_name)) {
            use_int8_ = true;
            auto scale_tv = loader_.get(scale_name);
            CUDA_CHECK(cudaMalloc(&scale_gp, scale_tv.nbytes()));
            CUDA_CHECK(cudaMemcpy(scale_gp, scale_tv.data, scale_tv.nbytes(),
                                  cudaMemcpyHostToDevice));
        }
        weight_to_scale_[gp] = scale_gp;
        tv->data = gp;
        tg += nb;
    }
    spdlog::info("GPU weights: {} MB ({})", tg / (1024 * 1024),
                 use_int8_ ? "INT8" : "FP16");
}

void DualAREngine::ensure_workspace_tokens(int tokens) {
    if (tokens <= workspace_tokens_) return;

    int dim = cfg_.dim;
    int qkv_dim = (cfg_.n_head + 2 * cfg_.n_local_heads) * cfg_.head_dim;
    int max_inner = std::max({qkv_dim, cfg_.intermediate_size, cfg_.fast_intermediate_size});

    auto free_if_set = [](void*& p) {
        if (p) {
            cudaFree(p);
            p = nullptr;
        }
    };
    free_if_set(reinterpret_cast<void*&>(ws1_));
    free_if_set(reinterpret_cast<void*&>(ws2_));
    free_if_set(reinterpret_cast<void*&>(ws3_));
    free_if_set(reinterpret_cast<void*&>(q_buf_));
    free_if_set(reinterpret_cast<void*&>(k_buf_));
    free_if_set(reinterpret_cast<void*&>(v_buf_));
    free_if_set(reinterpret_cast<void*&>(attn_out_buf_));
    free_if_set(reinterpret_cast<void*&>(expanded_k_buf_));
    free_if_set(reinterpret_cast<void*&>(attn_ws_));

    CUDA_CHECK(cudaMalloc(&ws1_, (size_t)tokens * max_inner * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&ws2_, (size_t)tokens * max_inner * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&ws3_, (size_t)tokens * dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&q_buf_, (size_t)tokens * cfg_.n_head * cfg_.head_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&k_buf_, (size_t)tokens * cfg_.n_local_heads * cfg_.head_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&v_buf_, (size_t)tokens * cfg_.n_local_heads * cfg_.head_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&attn_out_buf_, (size_t)tokens * max_inner * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&expanded_k_buf_, (size_t)tokens * cfg_.n_head * cfg_.head_dim * sizeof(__half)));
    int n_q = cfg_.n_head, D = cfg_.head_dim;
    CUDA_CHECK(cudaMalloc(&attn_ws_, (n_q * tokens * tokens + 2ULL * n_q * tokens * D) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_scratch_int_, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&cb_idx_buf_, (size_t)cfg_.num_codebooks * tokens * sizeof(int32_t)));
    workspace_tokens_ = tokens;
    spdlog::info("DualAREngine: workspace={}MB each (max_inner={}, tokens={})",
                 (size_t)tokens * max_inner * sizeof(__half) / (1024*1024),
                 max_inner, tokens);
}

// ============================================================================
// quantized_gemm — FP16 cuBLAS or INT8 dequant+GEMM dispatch
// ============================================================================
void DualAREngine::quantized_gemm(int M_out, int N_out, int K,
                                   const TensorView& weight,
                                   const __half* X, __half* Y) {
    auto it = weight_to_scale_.find(weight.data);
    if (it != weight_to_scale_.end() && it->second != nullptr) {
        // INT8 path: dequant + GEMM in one kernel
        kernels::int8_dequant_gemm_fp16(
            static_cast<const int8_t*>(weight.data),
            it->second, X, Y, M_out, N_out, K, stream_);
    } else {
        // FP16 path: standard cuBLAS GEMM
        gemm_fp16(M_out, N_out, K, weight.as<__half>(), X, Y, cublas_);
    }
}

// ============================================================================
// prefill — process the full prompt through the text transformer
//
// QKV split: GPU-side qkv_split_heads_gqa transforms to head-major in one kernel.
// ============================================================================
void DualAREngine::prefill(const int32_t* tokens, int B, int T, int prompt_stride,
                           __half* k_cache, __half* v_cache,
                           const int32_t* block_table, const int32_t* /*seq_lens*/,
                           __half* last_hidden_state, __half* hidden_states_for_fast)
{
    ensure_workspace_tokens(B * T);

    int dim = cfg_.dim, n_tokens = B * T;
    int n_q = cfg_.n_head, n_kv = cfg_.n_local_heads, D = cfg_.head_dim;
    int qkv_dim = (n_q + 2 * n_kv) * D;
    float sm_scale = 1.0f / std::sqrt(float(D));
    // --- Embedding lookup (all GPU-side) ---
    // tokens points to row 0 of the prompt tensor. prompt_stride is 1 for
    // text-only prompts and num_codebooks+1 for reference prompts.
    int stride = prompt_stride;

    // Step 1: text embedding for row 0 (all positions) — on GPU
    kernels::embedding_lookup(tokens, w_embedding_.as<__half>(), ws3_,
                              n_tokens, dim, cfg_.vocab_size, stream_);

    // Step 2: add codebook embeddings for rows 1..num_codebooks (GPU-side)
    // Read prompt rows to host only for index construction (small: stride * T int32)
    {
        std::vector<int32_t> h_full(stride * T);
        CUDA_CHECK(cudaMemcpy(h_full.data(), tokens,
                              stride * T * sizeof(int32_t), cudaMemcpyDeviceToHost));

        int cb_size = cfg_.codebook_size;
        int sem_begin = cfg_.semantic_begin_id;
        int sem_end   = cfg_.semantic_end_id;
        bool any_vq = false;

        bool has_vq_positions = false;
        std::vector<uint8_t> h_vq_mask(T, 0);
        for (int t = 0; t < T; t++) {
            int32_t tok = h_full[t];
            bool is_sem = (tok >= sem_begin && tok <= sem_end);
            h_vq_mask[t] = is_sem ? 1 : 0;
            has_vq_positions = has_vq_positions || is_sem;
        }

        for (int cb = 0; cb < cfg_.num_codebooks && stride > cb + 1; cb++) {
            if (!has_vq_positions) continue;
            any_vq = true;

            // Build indices and per-position mask. Code 0 is a valid DAC code, so
            // VQ positions must be detected from row 0 semantic tokens, not code != 0.
            std::vector<int32_t> h_idx(T);
            std::vector<__half> h_mask(T * dim);
            for (int t = 0; t < T; t++) {
                int32_t code = h_full[(1 + cb) * T + t];
                int idx = code + cb * cb_size;
                h_idx[t] = (idx >= 0 && idx < cfg_.num_codebooks * cb_size) ? idx : 0;
                __half mv = __float2half(h_vq_mask[t] ? 1.0f : 0.0f);
                for (int d = 0; d < dim; d++) h_mask[t * dim + d] = mv;
            }
            int32_t* d_idx = cb_idx_buf_ + cb * T;
            __half* d_emb   = attn_out_buf_;
            __half* d_mask  = expanded_k_buf_;
            CUDA_CHECK(cudaMemcpyAsync(d_idx, h_idx.data(), T * sizeof(int32_t),
                                        cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_mask, h_mask.data(), T * dim * sizeof(__half),
                                        cudaMemcpyHostToDevice, stream_));
            kernels::embedding_lookup(d_idx, w_codebook_embeddings_.as<__half>(),
                                      d_emb, T, dim, cfg_.num_codebooks * cb_size, stream_);
            kernels::mul_forward(d_emb, d_mask, T * dim, stream_);
            kernels::residual_add(ws3_, d_emb, T * dim, stream_);
        }

        // Step 3: scale_codebook_embeddings for semantic-token positions
        if (any_vq) {
            std::vector<__half> h_scale(T * dim);
            float sem_scale = 1.0f / std::sqrt(float(cfg_.num_codebooks + 1));
            for (int t = 0; t < T; t++) {
                int32_t tok = h_full[t];
                bool is_sem = (tok >= sem_begin && tok <= sem_end);
                __half s = __float2half(is_sem ? sem_scale : 1.0f);
                for (int d = 0; d < dim; d++) h_scale[t * dim + d] = s;
            }
            __half* d_scale = expanded_k_buf_;
            CUDA_CHECK(cudaMemcpyAsync(d_scale, h_scale.data(),
                                        T * dim * sizeof(__half), cudaMemcpyHostToDevice, stream_));
            kernels::mul_forward(ws3_, d_scale, T * dim, stream_);
        }
    }
    __half* curr = ws3_;

    // Read block table from GPU for KV cache writes
    std::vector<int32_t> h_bt(128, -1);
    if (block_table)
        CUDA_CHECK(cudaMemcpy(h_bt.data(), block_table, 128 * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // --- Layer loop ---
    for (int l = 0; l < cfg_.n_layer; l++) {
        auto& lw = layers_[l];

        // RMSNorm → fused QKV
        kernels::rms_norm(curr, lw.attn_norm.as<__half>(), ws1_,
                          n_tokens, dim, cfg_.norm_eps, stream_);

        quantized_gemm(qkv_dim, n_tokens, dim, lw.wqkv, ws1_, ws2_);

        // GPU QKV split + transpose: ws2_ [T, qkv_dim] → q_buf_/k_buf_/v_buf_ head-major
        kernels::qkv_split_heads_gqa(ws2_, q_buf_, k_buf_, v_buf_,
                                     T, n_q, n_kv, D, stream_);

        // QK-norm BEFORE RoPE (Qwen3 spec: normalize, then rotate)
        if (cfg_.attention_qk_norm) {
            kernels::rms_norm(q_buf_, lw.q_norm.as<__half>(), q_buf_,
                              B * n_q * T, D, cfg_.norm_eps, stream_);
            kernels::rms_norm(k_buf_, lw.k_norm.as<__half>(), k_buf_,
                              B * n_kv * T, D, cfg_.norm_eps, stream_);
        }

        // RoPE on Q and K (head-major layout)
        kernels::rope_qk(q_buf_, k_buf_, rope_freqs_, B, n_q, n_kv, T, D, 0, stream_);

        // Write K/V to paged cache
        if (k_cache && v_cache) {
            int bs = 16;
            int n_blk = (T + bs - 1) / bs;
            for (int blk = 0; blk < n_blk; blk++) {
                int phys = h_bt[blk];
                if (phys < 0) continue;
                int st = blk * bs, en = std::min(st + bs, T);
                int n_copy = en - st;
                for (int h = 0; h < n_kv; h++) {
                    size_t doff = ((((size_t)phys * cfg_.n_layer + l) * n_kv + h) * bs + 0) * D;
                    size_t soff = ((size_t)h * T + st) * D;
                    size_t bytes = n_copy * D * sizeof(__half);
                    CUDA_CHECK(cudaMemcpyAsync(k_cache + doff, k_buf_ + soff, bytes,
                                               cudaMemcpyDeviceToDevice, stream_));
                    CUDA_CHECK(cudaMemcpyAsync(v_cache + doff, v_buf_ + soff, bytes,
                                               cudaMemcpyDeviceToDevice, stream_));
                }
            }
        }

        // GPU prefill attention
        if (B != 1) {
            int per_batch_q = n_q * T * D, per_batch_kv = n_kv * T * D, per_batch_out = n_q * T * D;
            for (int b = 0; b < B; b++) {
                kernels::prefill_attention_gpu(
                    q_buf_ + b * per_batch_q, k_buf_ + b * per_batch_kv,
                    v_buf_ + b * per_batch_kv, ws2_ + b * per_batch_out,
                    attn_ws_, n_q, n_kv, T, D, sm_scale, cublas_, stream_, 0);
            }
            kernels::merge_heads(ws2_, attn_out_buf_, B * T, n_q, D, stream_);
        } else {
            kernels::prefill_attention_gpu(
                q_buf_, k_buf_, v_buf_, ws2_, attn_ws_,
                n_q, n_kv, T, D, sm_scale, cublas_, stream_, 0);
            kernels::merge_heads(ws2_, attn_out_buf_, T, n_q, D, stream_);
        }

        // Wo projection + residual
        quantized_gemm(dim, n_tokens, n_q * D, lw.wo, attn_out_buf_, ws1_);

        kernels::residual_add(curr, ws1_, n_tokens * dim, stream_);

        // --- SwiGLU FFN ---
        int inter = cfg_.intermediate_size;
        kernels::rms_norm(curr, lw.ffn_norm.as<__half>(), ws2_, n_tokens, dim, cfg_.norm_eps, stream_);

        // gate = silu(X @ w_gate^T)
        quantized_gemm(inter, n_tokens, dim, lw.w_gate, ws2_, ws1_);
        kernels::silu_forward(ws1_, n_tokens * inter, stream_);

        // up = X @ w_up^T
        quantized_gemm(inter, n_tokens, dim, lw.w_up, ws2_, attn_out_buf_);

        // gate *= up (element-wise)
        kernels::mul_forward(ws1_, attn_out_buf_, n_tokens * inter, stream_);

        // down = (gate*up) @ w_down^T  → output
        quantized_gemm(dim, n_tokens, inter, lw.w_down, ws1_, ws2_);
        kernels::residual_add(curr, ws2_, n_tokens * dim, stream_);

    }

    // --- Final RMSNorm ---
    kernels::rms_norm(curr, w_text_norm_.as<__half>(), ws1_,
                      n_tokens, dim, cfg_.norm_eps, stream_);

    // Extract last-token hidden states
    for (int b = 0; b < B; b++) {
        size_t src = ((size_t)b * T + (T - 1)) * dim;
        size_t dst = (size_t)b * dim;
        // last_hidden_state = post-norm (for logits via LM head)
        CUDA_CHECK(cudaMemcpyAsync(last_hidden_state + dst, ws1_ + src,
                                   dim * sizeof(__half), cudaMemcpyDeviceToDevice, stream_));
        // hidden_states_for_fast = post-norm (norm_fastlayer_input=True hardcoded in loader)
        CUDA_CHECK(cudaMemcpyAsync(hidden_states_for_fast + dst, ws1_ + src,
                                   dim * sizeof(__half), cudaMemcpyDeviceToDevice, stream_));
    }
}

// ============================================================================
// decode_step — single-token autoregressive step through the text transformer
//
// Computes Q, K, V from the input embedding, writes K/V to paged cache,
// runs paged attention, then FFN.  Output is the next hidden state.
// ============================================================================
void DualAREngine::decode_step(const __half* input_embed, int B, int token_pos,
                               __half* k_cache, __half* v_cache,
                               const int32_t* block_table, const int32_t* seq_lens,
                               __half* output_embed, __half* fast_hidden)
{
    int dim = cfg_.dim, n_q = cfg_.n_head, n_kv = cfg_.n_local_heads, D = cfg_.head_dim;
    int qkv_dim = (n_q + 2 * n_kv) * D;
    float sm_scale = 1.0f / std::sqrt(float(D));

    // Copy input → ws3_ (curr)
    CUDA_CHECK(cudaMemcpyAsync(ws3_, input_embed, B * dim * sizeof(__half),
                               cudaMemcpyDeviceToDevice, stream_));
    __half* curr = ws3_;

    // Read block table from GPU for KV cache writes
    std::vector<int32_t> h_bt(128, -1);
    if (block_table)
        CUDA_CHECK(cudaMemcpy(h_bt.data(), block_table, 128 * sizeof(int32_t), cudaMemcpyDeviceToHost));

    int bs = 16;  // block size

    // --- Layer loop ---
    for (int l = 0; l < cfg_.n_layer; l++) {
        auto& lw = layers_[l];

        // RMSNorm → fused QKV projection
        kernels::rms_norm(curr, lw.attn_norm.as<__half>(), ws1_,
                          B, dim, cfg_.norm_eps, stream_);
        quantized_gemm(qkv_dim, B, dim, lw.wqkv, ws1_, ws2_);

        // Split Q, K, V (B=1, so flat layout = head-major layout)
        // ws2_ layout: [B, n_q*D | n_kv*D | n_kv*D]
        size_t q_elems = (size_t)B * n_q * D;
        size_t kv_elems = (size_t)B * n_kv * D;
        CUDA_CHECK(cudaMemcpyAsync(q_buf_, ws2_, q_elems * sizeof(__half),
                                   cudaMemcpyDeviceToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(k_buf_, ws2_ + B * n_q * D, kv_elems * sizeof(__half),
                                   cudaMemcpyDeviceToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(v_buf_, ws2_ + B * (n_q + n_kv) * D, kv_elems * sizeof(__half),
                                   cudaMemcpyDeviceToDevice, stream_));

        // QK-norm BEFORE RoPE (Qwen3 spec: normalize, then rotate)
        if (cfg_.attention_qk_norm) {
            kernels::rms_norm(q_buf_, lw.q_norm.as<__half>(), q_buf_,
                              B * n_q, D, cfg_.norm_eps, stream_);
            kernels::rms_norm(k_buf_, lw.k_norm.as<__half>(), k_buf_,
                              B * n_kv, D, cfg_.norm_eps, stream_);
        }

        // RoPE on Q and K (offset = token_pos for the new token)
        kernels::rope_qk(q_buf_, k_buf_, rope_freqs_, B, n_q, n_kv, 1, D, token_pos, stream_);

        // Write K/V to paged cache at position token_pos
        if (k_cache && v_cache) {
            int blk_idx = token_pos / bs;
            int off_in_blk = token_pos % bs;
            int phys = h_bt[blk_idx];
            if (phys >= 0) {
                for (int h = 0; h < n_kv; h++) {
                    size_t doff = ((((size_t)phys * cfg_.n_layer + l) * n_kv + h) * bs + off_in_blk) * D;
                    size_t soff = (size_t)h * D;
                    CUDA_CHECK(cudaMemcpyAsync(k_cache + doff, k_buf_ + soff,
                                               D * sizeof(__half), cudaMemcpyDeviceToDevice, stream_));
                    CUDA_CHECK(cudaMemcpyAsync(v_cache + doff, v_buf_ + soff,
                                               D * sizeof(__half), cudaMemcpyDeviceToDevice, stream_));
                }
            }
        }

        // Paged attention (decode kernel)
        kernels::paged_attention_decode(q_buf_, k_cache, v_cache, block_table,
                                        seq_lens,
                                        ws2_, B, n_q, D, 128 /*max_blocks_per_seq*/,
                                        sm_scale, l, cfg_.n_layer, n_kv, stream_);

        // Wo projection + residual
        quantized_gemm(dim, B, n_q * D, lw.wo, ws2_, ws1_);
        kernels::residual_add(curr, ws1_, B * dim, stream_);

        // --- SwiGLU FFN ---
        int inter = cfg_.intermediate_size;
        kernels::rms_norm(curr, lw.ffn_norm.as<__half>(), ws2_,
                          B, dim, cfg_.norm_eps, stream_);

        quantized_gemm(inter, B, dim, lw.w_gate, ws2_, ws1_);
        kernels::silu_forward(ws1_, B * inter, stream_);

        quantized_gemm(inter, B, dim, lw.w_up, ws2_, attn_out_buf_);
        kernels::mul_forward(ws1_, attn_out_buf_, B * inter, stream_);

        quantized_gemm(dim, B, inter, lw.w_down, ws1_, ws2_);
        kernels::residual_add(curr, ws2_, B * dim, stream_);
    }

    // Final RMSNorm → output_embed (for logits via LM head)
    kernels::rms_norm(curr, w_text_norm_.as<__half>(), ws1_,
                      B, dim, cfg_.norm_eps, stream_);
    CUDA_CHECK(cudaMemcpyAsync(output_embed, ws1_, B * dim * sizeof(__half),
                               cudaMemcpyDeviceToDevice, stream_));
    // fast_hidden = post-norm (norm_fastlayer_input=True hardcoded in loader)
    CUDA_CHECK(cudaMemcpyAsync(fast_hidden, ws1_, B * dim * sizeof(__half),
                               cudaMemcpyDeviceToDevice, stream_));
}

// ============================================================================
// get_logits — project hidden state to vocabulary logits via shared embedding
// ============================================================================
void DualAREngine::get_logits(const __half* h, __half* logits, int B) {
    quantized_gemm(cfg_.vocab_size, B, cfg_.dim, w_embedding_, h, logits);
}

// ============================================================================
// embed_tokens — GPU embedding lookup (text model)
// ============================================================================
void DualAREngine::embed_tokens(const int32_t* tokens, __half* embeddings, int n_tokens) {
    kernels::embedding_lookup(tokens, w_embedding_.as<__half>(), embeddings,
                              n_tokens, cfg_.dim, cfg_.vocab_size, stream_);
}

// ============================================================================
// fast_embed_tokens — GPU embedding lookup (fast decoder)
// ============================================================================
void DualAREngine::fast_embed_tokens(const int32_t* tokens, __half* embeddings, int n_tokens) {
    kernels::embedding_lookup(tokens, w_fast_embeddings_.as<__half>(), embeddings,
                              n_tokens, cfg_.fast_dim, cfg_.codebook_size, stream_);
}

// ============================================================================
// embed_for_decode — combined embedding: text_embed[semantic] + Σ codebook_embeds
//
// This is the embedding used as input to decode_step at each AR step:
//   out = w_embedding_[semantic_token]
//   for j in 0..num_codebooks-1:
//       out += w_codebook_embeddings_[codebook_tokens[j] + j * codebook_size]
// If codebook_tokens == nullptr, codebook terms are skipped (all-zero context).
// ============================================================================
void DualAREngine::embed_for_decode(int32_t semantic_token, const int32_t* codebook_tokens,
                                     __half* out, cudaStream_t stream)
{
    int dim = cfg_.dim;
    int cb_size = cfg_.codebook_size;

    // Text embedding (GPU lookup)
    CUDA_CHECK(cudaMemcpyAsync(d_scratch_int_, &semantic_token, sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));
    kernels::embedding_lookup(d_scratch_int_, w_embedding_.as<__half>(), out,
                               1, dim, cfg_.vocab_size, stream);

    if (!codebook_tokens) return;

    // Add codebook embeddings (host-side loop; fine since N=9)
    std::vector<int32_t> h_cb(cfg_.num_codebooks);
    std::memcpy(h_cb.data(), codebook_tokens, cfg_.num_codebooks * sizeof(int32_t));

    for (int j = 0; j < cfg_.num_codebooks; j++) {
        int32_t idx = h_cb[j] + j * cb_size;
        CUDA_CHECK(cudaMemcpyAsync(d_scratch_int_, &idx, sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream));
        kernels::embedding_lookup(d_scratch_int_, w_codebook_embeddings_.as<__half>(), attn_out_buf_,
                                  1, dim, cfg_.num_codebooks * cb_size, stream);
        kernels::residual_add(out, attn_out_buf_, dim, stream);
    }

    // scale_codebook_embeddings=True: divide combined embedding by sqrt(num_codebooks+1)
    // Applied when token is a semantic audio token (always true during audio generation).
    float scale = 1.0f / std::sqrt(float(cfg_.num_codebooks + 1));
    kernels::scale_inplace(out, dim, scale, stream);
}

// ============================================================================
// fast_codebook_decode — single codebook step through the fast decoder
//
// The fast decoder has 4 transformer layers with its own KV cache.
// Q, K, V are projected from the input hidden state, RoPE applied,
// K/V written to fast_k_cache_/fast_v_cache_ at position codebook_step,
// then attention runs over all T_kv = codebook_step + 1 positions.
// ============================================================================
void DualAREngine::fast_codebook_decode(const __half* fast_hidden,
                                        __half* codebook_hidden, __half* logits,
                                        int B, int codebook_step, int32_t prev_token,
                                        bool compute_logits)
{
    int dim = cfg_.fast_dim, fq = cfg_.fast_n_head, fkv = cfg_.fast_n_local_heads;
    int fD = cfg_.fast_head_dim, cb = cfg_.codebook_size;
    int fqkv_dim = (fq + 2 * fkv) * fD;
    int T_kv = codebook_step + 1;  // number of K/V positions available
    float sm_scale = 1.0f / std::sqrt(float(fD));

    // Choose input: slow hidden state for step 0, fast embedding for step > 0
    if (codebook_step == 0) {
        CUDA_CHECK(cudaMemcpyAsync(ws3_, fast_hidden, B * dim * sizeof(__half),
                                   cudaMemcpyDeviceToDevice, stream_));
    } else {
        // Embed the previous codebook token via the fast embeddings table
        // fast_embed_tokens expects GPU token array, so upload prev_token
        CUDA_CHECK(cudaMemcpyAsync(d_scratch_int_, &prev_token, sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream_));
        fast_embed_tokens(d_scratch_int_, ws3_, 1);  // ws3_: [1, dim]
    }
    __half* curr = ws3_;

    // --- Fast layer loop ---
    for (int l = 0; l < cfg_.n_fast_layer; l++) {
        auto& fl = fast_layers_[l];

        // RMSNorm → fused QKV
        kernels::rms_norm(curr, fl.attn_norm.as<__half>(), ws1_,
                          B, dim, cfg_.norm_eps, stream_);
        quantized_gemm(fqkv_dim, B, dim, fl.wqkv, ws1_, ws2_);

        // Split Q, K, V (B=1)
        size_t q_elems = (size_t)B * fq * fD;
        size_t kv_elems = (size_t)B * fkv * fD;
        CUDA_CHECK(cudaMemcpyAsync(q_buf_, ws2_, q_elems * sizeof(__half),
                                   cudaMemcpyDeviceToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(k_buf_, ws2_ + B * fq * fD, kv_elems * sizeof(__half),
                                   cudaMemcpyDeviceToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(v_buf_, ws2_ + B * (fq + fkv) * fD, kv_elems * sizeof(__half),
                                   cudaMemcpyDeviceToDevice, stream_));

        // RoPE (fast decoder uses fast_rope_freqs_, offset = codebook_step)
        kernels::rope_qk(q_buf_, k_buf_, fast_rope_freqs_,
                         B, fq, fkv, 1, fD, codebook_step, stream_);

        // Write K/V to fast decoder cache at position codebook_step
        // Cache layout: [n_fast_layer, n_kv_heads, num_codebooks, head_dim]
        for (int h = 0; h < fkv; h++) {
            size_t off = (((size_t)l * fkv + h) * cfg_.num_codebooks + codebook_step) * fD;
            CUDA_CHECK(cudaMemcpyAsync(fast_k_cache_ + off, k_buf_ + h * fD,
                                       fD * sizeof(__half), cudaMemcpyDeviceToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(fast_v_cache_ + off, v_buf_ + h * fD,
                                       fD * sizeof(__half), cudaMemcpyDeviceToDevice, stream_));
        }

        const size_t layer_cache_off =
            static_cast<size_t>(l) * fkv * cfg_.num_codebooks * fD;
        kernels::fast_attention_decode(
            q_buf_, fast_k_cache_ + layer_cache_off, fast_v_cache_ + layer_cache_off,
            attn_out_buf_, B, fq, fkv, fD, T_kv, cfg_.num_codebooks,
            sm_scale, stream_);

        // Wo projection + residual
        quantized_gemm(dim, B, fq * fD, fl.wo, attn_out_buf_, ws1_);
        kernels::residual_add(curr, ws1_, B * dim, stream_);

        // --- SwiGLU FFN ---
        kernels::rms_norm(curr, fl.ffn_norm.as<__half>(), ws2_,
                          B, dim, cfg_.norm_eps, stream_);

        int fi = cfg_.fast_intermediate_size;
        quantized_gemm(fi, B, dim, fl.w_gate, ws2_, ws1_);
        kernels::silu_forward(ws1_, B * fi, stream_);

        quantized_gemm(fi, B, dim, fl.w_up, ws2_, attn_out_buf_);
        kernels::mul_forward(ws1_, attn_out_buf_, B * fi, stream_);

        quantized_gemm(dim, B, fi, fl.w_down, ws1_, ws2_);
        kernels::residual_add(curr, ws2_, B * dim, stream_);
    }

    if (compute_logits) {
        kernels::rms_norm(curr, w_fast_norm_.as<__half>(), ws1_,
                          B, dim, cfg_.norm_eps, stream_);
        quantized_gemm(cb, B, dim, w_fast_output_, ws1_, logits);
    }

    // Copy pre-norm hidden → codebook_hidden for next step's input
    if (codebook_hidden) {
        CUDA_CHECK(cudaMemcpyAsync(codebook_hidden, curr, B * dim * sizeof(__half),
                                   cudaMemcpyDeviceToDevice, stream_));
    }
}

}  // namespace fish
