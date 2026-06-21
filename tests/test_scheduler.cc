// tests/test_scheduler.cc
#include "engine/scheduler.h"
#include "engine/block_manager.h"
#include "utils/cuda_utils.h"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace fish {
namespace {

class SchedulerTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }
    void TearDown() override {
        CUDA_CHECK(cudaStreamDestroy(stream_));
    }
    cudaStream_t stream_{};
};

TEST_F(SchedulerTest, Construction) {
    SchedulerConfig cfg;
    cfg.max_batch_size = 8;
    cfg.max_seqs = 64;

    BlockManager::Config bm_cfg;
    bm_cfg.max_blocks = 64;
    bm_cfg.n_layers = 2;
    bm_cfg.n_heads = 2;
    bm_cfg.head_dim = 8;
    bm_cfg.block_size = 16;

    BlockManager bm(bm_cfg, stream_);
    Scheduler sched(cfg, &bm);
    EXPECT_FALSE(sched.has_work());
}

TEST_F(SchedulerTest, SubmitAndAdmit) {
    SchedulerConfig cfg;
    cfg.max_batch_size = 4;
    cfg.max_seqs = 16;
    cfg.block_size = 16;

    BlockManager::Config bm_cfg;
    bm_cfg.max_blocks = 256;
    bm_cfg.n_layers = 2;
    bm_cfg.n_heads = 2;
    bm_cfg.head_dim = 8;
    bm_cfg.block_size = 16;

    BlockManager bm(bm_cfg, stream_);
    Scheduler sched(cfg, &bm);

    auto req = std::make_shared<InferenceRequest>();
    req->request_id = 1;
    req->prompt_tokens = {1, 2, 3, 4, 5};
    req->max_new_tokens = 10;

    EXPECT_TRUE(sched.submit(req));
    EXPECT_TRUE(sched.has_work());

    int finished = sched.step();
    EXPECT_EQ(finished, 0);
    EXPECT_EQ(sched.active_count(), 1);
}

TEST_F(SchedulerTest, MultipleRequests) {
    SchedulerConfig cfg;
    cfg.max_batch_size = 3;
    cfg.max_seqs = 16;

    BlockManager::Config bm_cfg;
    bm_cfg.max_blocks = 256;
    bm_cfg.n_layers = 2;
    bm_cfg.n_heads = 2;
    bm_cfg.head_dim = 8;
    bm_cfg.block_size = 16;

    BlockManager bm(bm_cfg, stream_);
    Scheduler sched(cfg, &bm);

    for (int i = 0; i < 5; i++) {
        auto req = std::make_shared<InferenceRequest>();
        req->request_id = i;
        req->prompt_tokens = {1, 2, 3};
        req->max_new_tokens = 5;
        EXPECT_TRUE(sched.submit(req));
    }

    sched.step();
    EXPECT_EQ(sched.active_count(), 3);

    sched.finish_request(0);
    sched.finish_request(1);

    sched.step();
    EXPECT_EQ(sched.active_count(), 3);
}

TEST_F(SchedulerTest, OutOfBlocksFailsGracefully) {
    SchedulerConfig cfg;
    cfg.max_batch_size = 4;
    cfg.max_seqs = 8;
    cfg.block_size = 16;

    BlockManager::Config bm_cfg;
    bm_cfg.max_blocks = 2;
    bm_cfg.n_layers = 2;
    bm_cfg.n_heads = 2;
    bm_cfg.head_dim = 8;
    bm_cfg.block_size = 16;

    BlockManager bm(bm_cfg, stream_);
    Scheduler sched(cfg, &bm);

    auto req = std::make_shared<InferenceRequest>();
    req->request_id = 1;
    req->prompt_tokens = std::vector<int32_t>(100, 0);
    req->max_new_tokens = 500;

    EXPECT_TRUE(sched.submit(req));
    int finished = sched.step();
    EXPECT_EQ(finished, 1);
    EXPECT_EQ(sched.active_count(), 0);
}

TEST_F(SchedulerTest, QueueFullRejects) {
    SchedulerConfig cfg;
    cfg.max_batch_size = 1;
    cfg.max_seqs = 2;

    BlockManager::Config bm_cfg;
    bm_cfg.max_blocks = 256;
    bm_cfg.n_layers = 2;
    bm_cfg.n_heads = 2;
    bm_cfg.head_dim = 8;
    bm_cfg.block_size = 16;

    BlockManager bm(bm_cfg, stream_);
    Scheduler sched(cfg, &bm);

    auto req1 = std::make_shared<InferenceRequest>();
    req1->request_id = 1;
    req1->prompt_tokens = {1};
    req1->max_new_tokens = 1;
    EXPECT_TRUE(sched.submit(req1));

    auto req2 = std::make_shared<InferenceRequest>();
    req2->request_id = 2;
    req2->prompt_tokens = {2};
    req2->max_new_tokens = 1;
    EXPECT_TRUE(sched.submit(req2));

    auto req3 = std::make_shared<InferenceRequest>();
    req3->request_id = 3;
    req3->prompt_tokens = {3};
    req3->max_new_tokens = 1;
    EXPECT_FALSE(sched.submit(req3));
}

TEST_F(SchedulerTest, HasWorkLifecycle) {
    SchedulerConfig cfg;
    cfg.max_batch_size = 4;
    cfg.max_seqs = 8;

    BlockManager::Config bm_cfg;
    bm_cfg.max_blocks = 256;
    bm_cfg.n_layers = 2;
    bm_cfg.n_heads = 2;
    bm_cfg.head_dim = 8;
    bm_cfg.block_size = 16;

    BlockManager bm(bm_cfg, stream_);
    Scheduler sched(cfg, &bm);

    EXPECT_FALSE(sched.has_work());

    auto req = std::make_shared<InferenceRequest>();
    req->request_id = 1;
    req->prompt_tokens = {1, 2, 3};
    req->max_new_tokens = 5;
    EXPECT_TRUE(sched.submit(req));
    EXPECT_TRUE(sched.has_work());

    sched.step();
    EXPECT_TRUE(sched.has_work());

    sched.finish_request(1);
    sched.step();
    EXPECT_FALSE(sched.has_work());
}

}  // namespace
}  // namespace fish
