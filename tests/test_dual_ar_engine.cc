// tests/test_dual_ar_engine.cc
#include "engine/dual_ar_engine.h"
#include "model/dual_ar_config.h"
#include "model/loader.h"
#include "utils/cuda_utils.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include <fstream>

namespace fish {
namespace {

class DualAREngineTest : public ::testing::Test {
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

TEST_F(DualAREngineTest, ConfigDefaults) {
    DualARConfig cfg;
    EXPECT_EQ(cfg.vocab_size, 155776);
    EXPECT_EQ(cfg.n_layer, 36);
    EXPECT_EQ(cfg.n_head, 32);
    EXPECT_EQ(cfg.dim, 2560);
    EXPECT_EQ(cfg.intermediate_size, 9728);
    EXPECT_EQ(cfg.n_local_heads, 8);
    EXPECT_EQ(cfg.head_dim, 128);
    EXPECT_FLOAT_EQ(cfg.rope_base, 1000000.0f);
    EXPECT_FLOAT_EQ(cfg.norm_eps, 1e-6f);
    EXPECT_TRUE(cfg.attention_qk_norm);

    EXPECT_EQ(cfg.n_fast_layer, 4);
    EXPECT_EQ(cfg.fast_dim, 2560);
    EXPECT_EQ(cfg.fast_n_head, 32);
    EXPECT_EQ(cfg.fast_n_local_heads, 8);
    EXPECT_EQ(cfg.fast_head_dim, 128);
    EXPECT_EQ(cfg.fast_intermediate_size, 9728);
    EXPECT_EQ(cfg.codebook_size, 4096);
    EXPECT_EQ(cfg.num_codebooks, 10);
}

TEST_F(DualAREngineTest, DerivedConfigValues) {
    DualARConfig cfg;
    EXPECT_EQ(cfg.head_size(), cfg.head_dim * cfg.n_head);
    EXPECT_EQ(cfg.kv_head_size(), cfg.head_dim * cfg.n_local_heads);
    EXPECT_EQ(cfg.fast_head_size(), cfg.fast_head_dim * cfg.fast_n_head);
    EXPECT_EQ(cfg.fast_kv_head_size(), cfg.fast_head_dim * cfg.fast_n_local_heads);
    EXPECT_EQ(cfg.codebook_stride(), cfg.codebook_size);
    EXPECT_EQ(cfg.head_size(), 128 * 32);
    EXPECT_EQ(cfg.kv_head_size(), 128 * 8);
}

TEST_F(DualAREngineTest, SpecialTokenIds) {
    DualARConfig cfg;
    EXPECT_EQ(cfg.semantic_begin_id, 151678);
    EXPECT_EQ(cfg.semantic_end_id, 155773);
    EXPECT_EQ(cfg.im_end_token_id, 151645);
    EXPECT_EQ(cfg.audio_pad_token_id, 151677);
    EXPECT_EQ(cfg.pad_token_id, 151669);

    // Semantic range is reasonable
    int sem_range = cfg.semantic_end_id - cfg.semantic_begin_id + 1;
    EXPECT_EQ(sem_range, 4096);  // one per codebook entry
}

TEST_F(DualAREngineTest, ConfigFromJsonString) {
    nlohmann::json j;
    j["vocab_size"] = 1000;
    j["n_layer"] = 12;
    j["n_head"] = 16;
    j["dim"] = 1024;
    j["intermediate_size"] = 4096;
    j["n_local_heads"] = 4;
    j["head_dim"] = 64;
    j["rope_base"] = 10000.0;
    j["norm_eps"] = 1e-5;
    j["attention_qk_norm"] = false;
    j["n_fast_layer"] = 2;
    j["fast_dim"] = 1024;
    j["fast_n_head"] = 16;
    j["fast_n_local_heads"] = 4;
    j["fast_head_dim"] = 64;
    j["fast_intermediate_size"] = 4096;
    j["fast_attention_qk_norm"] = false;
    j["codebook_size"] = 2048;
    j["num_codebooks"] = 8;

    auto cfg = DualARConfig::from_json(j);
    EXPECT_EQ(cfg.vocab_size, 1000);
    EXPECT_EQ(cfg.n_layer, 12);
    EXPECT_EQ(cfg.n_head, 16);
    EXPECT_EQ(cfg.dim, 1024);
    EXPECT_EQ(cfg.codebook_size, 2048);
    EXPECT_EQ(cfg.num_codebooks, 8);
    EXPECT_FALSE(cfg.attention_qk_norm);
}

// MoveSemantics: requires actual model weights to construct DualAREngine.
// The engine constructor resolves weight tensor names from the loader,
// so an empty loader will throw. This test is marked as a documentation
// of the expected behavior.
TEST_F(DualAREngineTest, ConstructorRejectsEmptyLoader) {
    DualARConfig cfg;
    cfg.n_layer = 2;
    cfg.n_fast_layer = 1;
    cfg.dim = 32;
    cfg.n_head = 2;
    cfg.n_local_heads = 1;
    cfg.head_dim = 16;
    cfg.intermediate_size = 64;
    cfg.fast_dim = 32;
    cfg.fast_n_head = 2;
    cfg.fast_n_local_heads = 1;
    cfg.fast_head_dim = 16;
    cfg.fast_intermediate_size = 64;
    cfg.codebook_size = 64;
    cfg.num_codebooks = 1;

    ModelLoader loader;  // empty loader
    EXPECT_THROW(
        DualAREngine engine(cfg, std::move(loader), cublas_, stream_),
        std::runtime_error);
}

}  // namespace
}  // namespace fish
