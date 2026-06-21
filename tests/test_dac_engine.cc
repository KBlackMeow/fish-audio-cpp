// tests/test_dac_engine.cc
#include "engine/dac_engine.h"
#include "model/dac_config.h"
#include "model/loader.h"
#include "utils/cuda_utils.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include <fstream>
#include <cstring>

namespace fish {
namespace {

class DACEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUBLAS_CHECK(cublasCreate(&cublas_));
        CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
    }
    void TearDown() override {
        CUBLAS_CHECK(cublasDestroy(cublas_));
        CUDA_CHECK(cudaStreamDestroy(stream_));
    }
    cudaStream_t stream_{};
    cublasHandle_t cublas_{};
};

TEST_F(DACEngineTest, ConstructionAndConfig) {
    DACConfig cfg;
    cfg.sample_rate = 44100;
    cfg.encoder_rates = {2, 4, 5, 8};
    cfg.decoder_rates = {8, 5, 4, 2};
    cfg.num_codebooks = 10;

    ModelLoader loader;  // empty loader (no tensors)

    DACEngine engine(cfg, std::move(loader), cublas_,
                     static_cast<cudnnHandle_t>(nullptr), stream_);

    EXPECT_EQ(engine.config().sample_rate, 44100);
    EXPECT_EQ(engine.config().num_codebooks, 10);
    EXPECT_EQ(engine.config().hop_length(), 2 * 4 * 5 * 8);  // 320
}

TEST_F(DACEngineTest, DefaultsMatchModdedDacConfig) {
    DACConfig cfg;

    EXPECT_EQ(cfg.sample_rate, 44100);
    EXPECT_EQ(cfg.num_codebooks, 10);
    EXPECT_EQ(cfg.latent_dim, 1024);
    ASSERT_EQ(cfg.encoder_rates.size(), 4);
    EXPECT_EQ(cfg.encoder_rates[0], 2);
    EXPECT_EQ(cfg.encoder_rates[1], 4);
    EXPECT_EQ(cfg.encoder_rates[2], 8);
    EXPECT_EQ(cfg.encoder_rates[3], 8);
    EXPECT_EQ(cfg.hop_length(), 2 * 4 * 8 * 8);
}

TEST_F(DACEngineTest, Init) {
    DACConfig cfg;
    ModelLoader loader;
    DACEngine engine(cfg, std::move(loader), cublas_,
                     static_cast<cudnnHandle_t>(nullptr), stream_);
    EXPECT_THROW(engine.init(), std::runtime_error);
}

TEST_F(DACEngineTest, DecodeRequiresNativeDecoder) {
    DACConfig cfg;
    cfg.sample_rate = 44100;
    cfg.encoder_rates = {2, 4, 5, 8};
    cfg.decoder_rates = {8, 5, 4, 2};

    ModelLoader loader;
    DACEngine engine(cfg, std::move(loader), cublas_,
                     static_cast<cudnnHandle_t>(nullptr), stream_);

    // Allocate small code and audio buffers
    constexpr int B = 1, N = DAC_TOTAL_CODEBOOKS, T = 2;
    int32_t* d_codes;
    float* d_audio;
    int audio_len = 0;
    int max_audio = T * cfg.hop_length() * 2;

    CUDA_CHECK(cudaMalloc(&d_codes, B * N * T * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_audio, max_audio * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_codes, 0, B * N * T * sizeof(int32_t)));

    EXPECT_THROW(engine.decode(d_codes, B, T, d_audio, &audio_len, max_audio), std::runtime_error);

    CUDA_CHECK(cudaFree(d_codes));
    CUDA_CHECK(cudaFree(d_audio));
}

TEST_F(DACEngineTest, EncodeRequiresNativeEncoder) {
    DACConfig cfg;
    cfg.sample_rate = 44100;
    cfg.encoder_rates = {2, 4, 5, 8};

    ModelLoader loader;
    DACEngine engine(cfg, std::move(loader), cublas_,
                     static_cast<cudnnHandle_t>(nullptr), stream_);

    constexpr int B = 1;
    int audio_len = cfg.hop_length() * 5;
    int32_t* d_codes;
    float* d_audio;
    int code_len = 0;

    CUDA_CHECK(cudaMalloc(&d_codes, B * DAC_TOTAL_CODEBOOKS * 5 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_audio, audio_len * sizeof(float)));

    EXPECT_THROW(engine.encode(d_audio, B, audio_len, d_codes, &code_len), std::runtime_error);

    CUDA_CHECK(cudaFree(d_codes));
    CUDA_CHECK(cudaFree(d_audio));
}

TEST_F(DACEngineTest, EncodeRejectsBatchGreaterThanOne) {
    DACConfig cfg;
    ModelLoader loader;
    DACEngine engine(cfg, std::move(loader), cublas_,
                     static_cast<cudnnHandle_t>(nullptr), stream_);

    constexpr int B = 2;
    int audio_len = cfg.hop_length();
    int32_t* d_codes;
    float* d_audio;
    int code_len = 0;

    CUDA_CHECK(cudaMalloc(&d_codes, B * DAC_TOTAL_CODEBOOKS * 2 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_audio, B * audio_len * sizeof(float)));

    EXPECT_THROW(engine.encode(d_audio, B, audio_len, d_codes, &code_len), std::runtime_error);

    CUDA_CHECK(cudaFree(d_codes));
    CUDA_CHECK(cudaFree(d_audio));
}

TEST_F(DACEngineTest, ConstantsAreConsistent) {
    // Verify DAC constants are internally consistent
    EXPECT_EQ(DAC_TOTAL_CODEBOOKS, DAC_ACOUSTIC_CODEBOOKS + 1);  // 1 semantic + 9 acoustic
    EXPECT_EQ(DAC_CODEBOOK_DIM, 8);
    EXPECT_EQ(DAC_LATENT_DIM, 1024);
    EXPECT_EQ(DAC_SEMANTIC_CODEBOOK_SIZE, 4096);
    EXPECT_EQ(DAC_ACOUSTIC_CODEBOOK_SIZE, 1024);
}

}  // namespace
}  // namespace fish
