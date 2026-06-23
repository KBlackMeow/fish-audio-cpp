// src/engine/dac_engine.h
#pragma once
#include "model/dac_config.h"
#include "model/loader.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <cudnn.h>

namespace fish {

// Total codebooks understood by the DAC:
//   1 semantic (index 0, codebook_size=4096) +
//   9 acoustic (indices 1-9, codebook_size=1024)
// = 10 codebooks matching DualARConfig::num_codebooks
static constexpr int DAC_TOTAL_CODEBOOKS = 10;
static constexpr int DAC_ACOUSTIC_CODEBOOKS = 9;
static constexpr int DAC_CODEBOOK_DIM = 8;     // internal dimension of each codebook entry
static constexpr int DAC_LATENT_DIM = 1024;    // dimension output by out_proj (decoder input)
static constexpr int DAC_SEMANTIC_CODEBOOK_SIZE = 4096;
static constexpr int DAC_ACOUSTIC_CODEBOOK_SIZE = 1024;

class DACEngine {
public:
    DACEngine(const DACConfig& cfg, ModelLoader&& loader,
              cublasHandle_t cublas, cudnnHandle_t cudnn, cudaStream_t stream);
    ~DACEngine();

    DACEngine(const DACEngine&) = delete;
    DACEngine& operator=(const DACEngine&) = delete;

    // Allocate GPU memory and upload codebook weights
    void init();

    // Decode VQ codes to audio waveform.
    //   codes:          [B, num_codebooks, T_code] on GPU (int32)
    //   audio:          [B, T_audio] on GPU (float32) — caller allocates
    //   audio_len:      set to actual number of samples on return
    //   max_audio_cap:  maximum samples the caller allocated in `audio`
    //
    // Pure C++ path. The RVQ stage is implemented; the convolutional decoder
    // must be present in the converted dac.bin weights.
    void decode(const int32_t* codes, int batch_size, int code_len,
                float* audio, int* audio_len, int max_audio_cap);

    // Encode audio waveform to VQ codes.
    void encode(const float* audio, int batch_size, int audio_len,
                int32_t* codes, int* code_len);

    const DACConfig& config() const { return cfg_; }

private:
    DACConfig cfg_;
    ModelLoader loader_;
    cublasHandle_t cublas_;
    cudnnHandle_t cudnn_;
    cudaStream_t stream_;

    bool use_int8_ = false;

    // --- RVQ weights on GPU ---
    // Codebook embeddings: [codebook_size, DAC_CODEBOOK_DIM] (FP16)
    // Index 0 = semantic, indices 1..9 = acoustic
    std::vector<__half*> gpu_codebooks_;        // [DAC_TOTAL_CODEBOOKS]

    // out_proj linear weights: [DAC_LATENT_DIM, DAC_CODEBOOK_DIM] (FP16, from [L,D,1] conv)
    std::vector<__half*> gpu_out_proj_w_;       // [DAC_TOTAL_CODEBOOKS]

    // out_proj biases: [DAC_LATENT_DIM] (FP16)
    std::vector<__half*> gpu_out_proj_b_;       // [DAC_TOTAL_CODEBOOKS]

    // --- Encoder (audio → codes) ---
    // in_proj linear weights + biases for RVQ quantization
    std::vector<__half*> gpu_in_proj_w_;        // [DAC_TOTAL_CODEBOOKS]
    std::vector<__half*> gpu_in_proj_b_;        // [DAC_TOTAL_CODEBOOKS]

    // Workspace for RVQ decode:
    //   latent_buf_: [B, DAC_LATENT_DIM, T] accumulator (FP32 then converted to FP16)
    float* latent_buf_ = nullptr;

    struct ConvWeight {
        __half* weight = nullptr;
        __half* bias = nullptr;
        int c0 = 0;
        int c1 = 0;
        int k = 0;
        bool transpose = false;
    };

    struct DenseWeight {
        __half* weight = nullptr;
        __half* bias = nullptr;
        __half* scale = nullptr;      // INT8 per-channel scale (null if FP16)
        __half* act_scale = nullptr;  // static activation quant scale (null => dynamic)
        __half* smooth_inv = nullptr; // calibration smooth factor (null if none)
        int group_size = 0;           // K for row-wise, smaller for group-wise
        int out = 0;
        int in = 0;
    };

    std::unordered_map<std::string, ConvWeight> conv_weights_;
    std::unordered_map<std::string, DenseWeight> dense_weights_;
    std::unordered_map<std::string, __half*> vector_weights_;
    __half* decoder_buf_a_ = nullptr;
    __half* decoder_buf_b_ = nullptr;
    __half* decoder_buf_c_ = nullptr;
    __half* decoder_buf_d_ = nullptr;
    size_t decoder_buf_elems_ = 0;

    // Encoder workspace
    __half* encoder_buf_h_ = nullptr;   // FP16 scratch for encoder forward pass
    float* encoder_buf_ = nullptr;      // FP32 scratch for RVQ residual
    size_t encoder_buf_elems_ = 0;

    void load_decoder_weights();
    int decode_waveform_native(int batch_size, int code_len, float* audio, int max_audio_cap);
    int encode_waveform_native(const float* audio, int batch_size, int audio_len,
                               float* latent_out);
};

}  // namespace fish
