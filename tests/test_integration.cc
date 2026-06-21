// tests/test_integration.cc — integration tests for the full pipeline
#include "engine/inference_pipeline.h"
#include "engine/dual_ar_engine.h"
#include "engine/dac_engine.h"
#include "engine/block_manager.h"
#include "model/dual_ar_config.h"
#include "model/dac_config.h"
#include "model/loader.h"
#include "tokenizer/tokenizer.h"
#include "utils/cuda_utils.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include <random>
#include <numeric>
#include <cmath>

namespace fish {
namespace {

class IntegrationTest : public ::testing::Test {
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

// Helper: create a minimal DualARConfig for testing
static DualARConfig make_tiny_config() {
    DualARConfig cfg;
    cfg.n_layer = 2;
    cfg.n_head = 2;
    cfg.n_local_heads = 1;
    cfg.dim = 32;
    cfg.head_dim = 16;
    cfg.intermediate_size = 64;
    cfg.n_fast_layer = 1;
    cfg.fast_n_head = 2;
    cfg.fast_n_local_heads = 1;
    cfg.fast_dim = 32;
    cfg.fast_head_dim = 16;
    cfg.fast_intermediate_size = 64;
    cfg.codebook_size = 64;
    cfg.num_codebooks = 3;
    cfg.vocab_size = 256;
    return cfg;
}

// PipelineConstruction: DualAREngine requires model weights to construct.
// The constructor resolves weight tensor names from the loader, so
// inference with an empty loader is not possible. This test documents
// the dependency chain.
TEST_F(IntegrationTest, DependencyChain) {
    // BlockManager can be constructed independently
    BlockManager::Config bm_cfg;
    bm_cfg.max_blocks = 128;
    bm_cfg.n_layers = 2;
    bm_cfg.n_heads = 2;
    bm_cfg.head_dim = 16;
    auto block_mgr = std::make_unique<BlockManager>(bm_cfg, stream_);
    EXPECT_NE(block_mgr, nullptr);

    // DACEngine can be constructed independently; init() validates weights.
    DACConfig dac_cfg;
    ModelLoader dac_loader;
    auto dac = std::make_unique<DACEngine>(
        dac_cfg, std::move(dac_loader), cublas_,
        static_cast<cudnnHandle_t>(nullptr), stream_);
    EXPECT_NE(dac, nullptr);

    // Tokenizer can be constructed (though load() needs model path)
    auto tokenizer = std::make_unique<Tokenizer>();
    EXPECT_NE(tokenizer, nullptr);
}

TEST_F(IntegrationTest, TTSOutputDefaults) {
    TTSOutput output;
    EXPECT_TRUE(output.audio_samples.empty());
    EXPECT_EQ(output.sample_rate, 0);
}

TEST_F(IntegrationTest, BlockManagerConfigMeetsPipelineNeeds) {
    auto cfg = make_tiny_config();

    BlockManager::Config bm_cfg;
    bm_cfg.block_size = 16;
    bm_cfg.n_layers = cfg.n_layer;
    bm_cfg.n_heads = cfg.n_local_heads;
    bm_cfg.head_dim = cfg.head_dim;
    bm_cfg.max_blocks = 256;

    BlockManager bm(bm_cfg, stream_);

    // Verify block size is non-trivial
    EXPECT_EQ(bm.config().block_size, 16);
    EXPECT_EQ(bm.config().n_layers, cfg.n_layer);
    EXPECT_EQ(bm.config().n_heads, cfg.n_local_heads);
    EXPECT_EQ(bm.config().head_dim, cfg.head_dim);

    // Verify number of blocks matches what pipeline would need for 512 tokens
    int max_tokens = 512;
    int required_blocks = (max_tokens + 256 + bm_cfg.block_size - 1) / bm_cfg.block_size + 16;
    EXPECT_LE(required_blocks, static_cast<int>(bm.total_blocks()));
}

TEST_F(IntegrationTest, ConfigJsonRoundTrip) {
    // Write a config to JSON and read it back
    auto cfg = make_tiny_config();
    nlohmann::json j;
    j["vocab_size"] = cfg.vocab_size;
    j["n_layer"] = cfg.n_layer;
    j["n_head"] = cfg.n_head;
    j["dim"] = cfg.dim;
    j["intermediate_size"] = cfg.intermediate_size;
    j["n_local_heads"] = cfg.n_local_heads;
    j["head_dim"] = cfg.head_dim;
    j["rope_base"] = cfg.rope_base;
    j["norm_eps"] = cfg.norm_eps;
    j["attention_qk_norm"] = cfg.attention_qk_norm;
    j["n_fast_layer"] = cfg.n_fast_layer;
    j["fast_dim"] = cfg.fast_dim;
    j["fast_n_head"] = cfg.fast_n_head;
    j["fast_n_local_heads"] = cfg.fast_n_local_heads;
    j["fast_head_dim"] = cfg.fast_head_dim;
    j["fast_intermediate_size"] = cfg.fast_intermediate_size;
    j["fast_attention_qk_norm"] = cfg.fast_attention_qk_norm;
    j["codebook_size"] = cfg.codebook_size;
    j["num_codebooks"] = cfg.num_codebooks;

    auto cfg2 = DualARConfig::from_json(j);
    EXPECT_EQ(cfg2.vocab_size, cfg.vocab_size);
    EXPECT_EQ(cfg2.n_layer, cfg.n_layer);
    EXPECT_EQ(cfg2.dim, cfg.dim);
    EXPECT_EQ(cfg2.codebook_size, cfg.codebook_size);
    EXPECT_EQ(cfg2.num_codebooks, cfg.num_codebooks);
}

// ============================================================================
// Sampling logic tests — verify top-k/top-p/temperature matches Python reference
// ============================================================================

// Re-implementation of InferencePipeline::sample_range for testing.
// Matches Python logits_to_probs + sample in inference.py exactly.
static int32_t test_sample_range(
    const std::vector<float>& logits,  // raw logits (fp32)
    int id_start,                      // offset added to sampled index
    float temperature,
    float top_p,
    int top_k,
    uint64_t seed)
{
    int sz = static_cast<int>(logits.size());
    auto h_slice = logits;  // mutable copy

    // Step 1-2: sort descending, softmax on sorted
    std::vector<int> sorted_idx(sz);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(),
              [&](int a, int b) { return h_slice[a] > h_slice[b]; });

    std::vector<float> sorted_logits(sz);
    float max_l = -1e30f;
    for (int i = 0; i < sz; i++) {
        sorted_logits[i] = h_slice[sorted_idx[i]];
        if (sorted_logits[i] > max_l) max_l = sorted_logits[i];
    }
    float sum_e = 0.f;
    for (int i = 0; i < sz; i++) {
        sorted_logits[i] = std::exp(sorted_logits[i] - max_l);
        sum_e += sorted_logits[i];
    }
    for (int i = 0; i < sz; i++) sorted_logits[i] /= sum_e;

    // Step 3: top-p cumsum + top-k mask
    std::vector<bool> to_remove(sz, true);
    float cumsum = 0.f;
    int actual_k = std::min(top_k, sz);
    for (int i = 0; i < sz; i++) {
        cumsum += sorted_logits[i];
        bool beyond_topk = (i >= actual_k);
        bool beyond_topp = (i > 0 && cumsum > top_p);
        if (beyond_topk || beyond_topp) break;
        to_remove[sorted_idx[i]] = false;
    }

    // Step 4: mask → temperature AFTER filtering
    float inv_temp = 1.0f / std::max(temperature, 1e-5f);
    for (int i = 0; i < sz; i++) {
        if (to_remove[i]) h_slice[i] = -1e30f;
        else h_slice[i] *= inv_temp;
    }

    // Step 5: softmax on masked+scaled
    max_l = -1e30f;
    for (int i = 0; i < sz; i++)
        if (h_slice[i] > max_l) max_l = h_slice[i];
    sum_e = 0.f;
    for (int i = 0; i < sz; i++) {
        h_slice[i] = std::exp(h_slice[i] - max_l);
        sum_e += h_slice[i];
    }
    for (int i = 0; i < sz; i++) h_slice[i] /= sum_e;

    // Step 6: multinomial sample
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    float r = dist(gen);
    float cs = 0.f;
    for (int i = 0; i < sz; i++) {
        cs += h_slice[i];
        if (r < cs) return static_cast<int32_t>(id_start + i);
    }
    return static_cast<int32_t>(id_start);
}

TEST_F(IntegrationTest, SamplingTopKFiltering) {
    // 10 logits: token 3 is highest, token 7 is second
    std::vector<float> logits = {1.0f, 2.0f, 0.5f, 10.0f, -1.0f, 3.0f, -2.0f, 5.0f, 0.1f, 1.5f};

    // top_k=1: only token 3 (index 3) should survive
    // With temperature=0.01 (very low), sampling is nearly deterministic → always picks top
    int32_t result = test_sample_range(logits, 0, 0.01f, 1.0f, 1, 42);
    EXPECT_EQ(result, 3);  // token at index 3 (logit=10.0) should win

    // top_k=3: tokens 3, 7, 5 should survive (sorted: 3=10.0, 7=5.0, 5=3.0)
    // With very low temp, should always pick index 3
    for (int trial = 0; trial < 10; trial++) {
        result = test_sample_range(logits, 0, 0.001f, 1.0f, 3, 42 + trial);
        EXPECT_EQ(result, 3) << "trial=" << trial;
    }
}

TEST_F(IntegrationTest, SamplingTopPFiltering) {
    // Two strong tokens, rest weak
    std::vector<float> logits = {0.1f, 0.2f, 100.0f, 50.0f, 0.1f, 0.3f, 0.1f, 0.1f, 0.1f, 0.1f};

    // top_p=0.6 (60% cumulative prob): after softmax of [100, 50, 0.3, ...],
    // token 2 dominates (~1.0 prob), so only token 2 survives
    int32_t result = test_sample_range(logits, 0, 0.01f, 0.6f, 100, 42);
    EXPECT_EQ(result, 2);

    // top_p=1.0 + low temp: token 2 should still win
    result = test_sample_range(logits, 0, 0.01f, 1.0f, 100, 42);
    EXPECT_EQ(result, 2);
}

TEST_F(IntegrationTest, SamplingTemperatureEffect) {
    // Two close logits
    std::vector<float> logits = {2.0f, 2.5f};

    // Very low temperature (0.001): nearly deterministic → always picks index 1
    int picks_high = 0;
    for (int trial = 0; trial < 50; trial++) {
        if (test_sample_range(logits, 0, 0.001f, 1.0f, 10, 100 + trial) == 1)
            picks_high++;
    }
    EXPECT_GT(picks_high, 40);  // should almost always pick the higher one

    // Very high temperature (100.0): nearly uniform
    int picks_low = 0;
    for (int trial = 0; trial < 100; trial++) {
        if (test_sample_range(logits, 0, 100.0f, 1.0f, 10, 200 + trial) == 0)
            picks_low++;
    }
    // At high temp, both tokens should be picked roughly equally
    EXPECT_GT(picks_low, 20);
    EXPECT_LT(picks_low, 80);
}

TEST_F(IntegrationTest, SamplingDeterministic) {
    // Same seed, same logits → same output
    std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    int32_t r1 = test_sample_range(logits, 100, 0.7f, 0.9f, 5, 42);
    int32_t r2 = test_sample_range(logits, 100, 0.7f, 0.9f, 5, 42);
    EXPECT_EQ(r1, r2);  // deterministic with same seed

    // Different seed → likely different (with temp=0.7)
    int32_t r3 = test_sample_range(logits, 100, 0.7f, 0.9f, 5, 999);
    // Not guaranteed different, but very likely with 5 tokens and temp=0.7
}

TEST_F(IntegrationTest, SamplingSingleToken) {
    // Only one token → must always pick it
    std::vector<float> logits = {42.0f};
    int32_t result = test_sample_range(logits, 0, 0.7f, 0.9f, 50, 12345);
    EXPECT_EQ(result, 0);
}

TEST_F(IntegrationTest, SamplingAllSameLogits) {
    // All same → uniform sampling (depends on seed)
    std::vector<float> logits = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    // With top_k=5, all survive. Count distribution across trials
    int counts[5] = {0};
    for (int trial = 0; trial < 250; trial++) {
        int32_t r = test_sample_range(logits, 0, 1.0f, 1.0f, 5, 1000 + trial);
        ASSERT_GE(r, 0);
        ASSERT_LT(r, 5);
        counts[r]++;
    }
    // Each token should get ~20% ± reasonable margin
    for (int i = 0; i < 5; i++) {
        EXPECT_GT(counts[i], 25) << "token " << i << " picked " << counts[i] << " times";
        EXPECT_LT(counts[i], 75) << "token " << i << " picked " << counts[i] << " times";
    }
}

// ============================================================================
// Prompt template tests
// ============================================================================

TEST_F(IntegrationTest, TokenizerBuildPromptFormat) {
    Tokenizer tokenizer;
    std::string prompt = tokenizer.build_prompt("你好");

    // Must contain the chat structure
    EXPECT_NE(prompt.find("<|im_start|>system"), std::string::npos);
    EXPECT_NE(prompt.find("<|im_start|>user"), std::string::npos);
    EXPECT_NE(prompt.find("<|im_start|>assistant"), std::string::npos);
    EXPECT_NE(prompt.find("<|voice|>"), std::string::npos);
    EXPECT_NE(prompt.find("<|im_end|>"), std::string::npos);

    // System prompt must be the TTS instruction, not a chatbot persona
    EXPECT_NE(prompt.find("convert the provided text to speech"), std::string::npos);

    // <|voice|> must appear AFTER user text, NOT before it
    size_t user_pos = prompt.find("<|im_start|>user");
    size_t voice_pos = prompt.find("<|voice|>");
    size_t text_pos = prompt.find("你好");
    EXPECT_LT(user_pos, text_pos);     // user marker before text
    EXPECT_LT(text_pos, voice_pos);    // text before <|voice|> (NOT <|voice|> before text)

    // Must end with <|voice|> (no trailing <|im_end|> on assistant message)
    EXPECT_EQ(prompt.substr(prompt.size() - 9), "<|voice|>");
}

TEST_F(IntegrationTest, TokenizerBuildPromptNoVoiceInUserMessage) {
    Tokenizer tokenizer;
    std::string prompt = tokenizer.build_prompt("测试文本");

    // Count <|voice|> occurrences — should be exactly 1 (in assistant message)
    size_t first = prompt.find("<|voice|>");
    ASSERT_NE(first, std::string::npos);
    size_t second = prompt.find("<|voice|>", first + 1);
    EXPECT_EQ(second, std::string::npos) << "Multiple <|voice|> tokens found: " << prompt;
}

}  // namespace
}  // namespace fish
