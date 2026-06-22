// src/engine/dual_ar_engine.h
#pragma once
#include "model/dual_ar_config.h"
#include "model/loader.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <memory>
#include <vector>

namespace fish {

struct TextLayerWeights {
    TensorView wqkv;           // fused QKV [dim, (n_q+2*n_kv)*head_dim]
    TensorView wo;
    TensorView q_norm, k_norm; // QK-norm [head_dim] per-head
    TensorView attn_norm;
    TensorView w_gate, w_up, w_down;
    TensorView ffn_norm;
};

struct FastLayerWeights {
    TensorView wqkv;           // fused QKV [dim, (n_q+2*n_kv)*head_dim]
    TensorView wo;
    TensorView attn_norm;
    TensorView w_gate, w_up, w_down;
    TensorView ffn_norm;
};

class DualAREngine {
public:
    DualAREngine(const DualARConfig& cfg, ModelLoader&& loader,
                 cublasHandle_t cublas, cudaStream_t stream);
    ~DualAREngine();

    DualAREngine(const DualAREngine&) = delete;
    DualAREngine& operator=(const DualAREngine&) = delete;
    DualAREngine(DualAREngine&&) = default;
    DualAREngine& operator=(DualAREngine&&) = default;

    void init();
    const DualARConfig& config() const { return cfg_; }
    cudaStream_t stream() const { return stream_; }

    void prefill(const int32_t* tokens, int batch_size, int prompt_len,
                 int prompt_stride,
                 __half* k_cache, __half* v_cache,
                 const int32_t* block_table, const int32_t* seq_lens,
                 __half* last_hidden_state, __half* hidden_states_for_fast);

    void decode_step(const __half* input_embed, int batch_size, int seq_len,
                     __half* k_cache, __half* v_cache,
                     const int32_t* block_table, const int32_t* seq_lens,
                     __half* output_embed, __half* fast_hidden);

    void get_logits(const __half* hidden_state, __half* logits, int batch_size);

    // Embedding lookup for a token (uses GPU-side embedding table)
    void embed_tokens(const int32_t* tokens, __half* embeddings, int n_tokens);
    // Fast embedding lookup (audio_decoder.embeddings for codebook hidden state)
    void fast_embed_tokens(const int32_t* tokens, __half* embeddings, int n_tokens);

    // Combined embedding: text_embed[sem] + sum(codebook_embed[cb_j + j*codebook_size])
    // codebook_tokens is null (zero out codebook terms) or points to n_codebooks int32_t values
    void embed_for_decode(int32_t semantic_token, const int32_t* codebook_tokens,
                          __half* out, cudaStream_t stream);

    // fast_hidden : slow-decoder output (used only when codebook_step == 0)
    // prev_token  : token sampled at codebook_step-1 (ignored when step == 0)
    void fast_codebook_decode(const __half* fast_hidden, __half* codebook_hidden,
                              __half* logits, int batch_size, int codebook_step,
                              int32_t prev_token = 0, bool compute_logits = true);

private:
    void text_attention_prefill(const __half* input, __half* output,
                                __half* k_out, __half* v_out,
                                const TextLayerWeights& lw, int B, int T, int layer_idx);
    void fast_attention_prefill(const __half* input, __half* output,
                                const FastLayerWeights& lw, int B, int T, int layer_idx);
    void ffn_forward(const __half* input, __half* output,
                     const TensorView& w_gate, const TensorView& w_up,
                     const TensorView& w_down, int n_tokens, int dim, int inter);
    void ensure_workspace_tokens(int tokens);

    // GPU weight tracking (each weight individually copied)
    std::vector<void*> gpu_weight_ptrs_;

    DualARConfig cfg_;
    ModelLoader loader_;
    cublasHandle_t cublas_;
    cudaStream_t stream_;

    std::vector<TextLayerWeights> layers_;
    std::vector<FastLayerWeights> fast_layers_;

    // Embedding (shared input/output via tie_word_embeddings)
    TensorView w_embedding_;
    TensorView w_text_norm_;   // final norm after text transformer
    // Audio decoder top-level weights
    TensorView w_fast_embeddings_;
    TensorView w_codebook_embeddings_;  // [num_codebooks*codebook_size, dim]
    TensorView w_fast_norm_;
    TensorView w_fast_output_;  // [codebook_size, dim] for codebook logits

    float* rope_freqs_ = nullptr;       // text: [dim/2]
    float* fast_rope_freqs_ = nullptr;  // fast: [fast_head_dim/2]

    // Fast decoder KV cache: [n_fast_layer, n_kv_heads, num_codebooks, head_dim]
    __half* fast_k_cache_ = nullptr;
    __half* fast_v_cache_ = nullptr;

    // Workspace buffers
    __half* ws1_ = nullptr, *ws2_ = nullptr, *ws3_ = nullptr;
    __half* q_buf_ = nullptr, *k_buf_ = nullptr, *v_buf_ = nullptr;
    __half* attn_out_buf_ = nullptr;
    __half* expanded_k_buf_ = nullptr;  // for GQA K/V expansion
    __half* attn_ws_ = nullptr;         // prefill attention workspace
    int32_t* cb_idx_buf_ = nullptr;     // codebook index scratch
    int32_t* d_scratch_int_ = nullptr;  // tiny scratch for single-int uploads
    int workspace_tokens_ = 0;

    void* cublas_ws_ = nullptr;
    size_t cublas_ws_size_ = 0;
};

}  // namespace fish
