#include "engine/block_manager.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <memory>
#include <thread>
#include <vector>

namespace fish {
namespace {

class BlockManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaStreamCreate(&stream_);
    }

    void TearDown() override {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }

    cudaStream_t stream_{};
};

TEST_F(BlockManagerTest, ConstructionAndConfig) {
    BlockManager::Config cfg;
    cfg.block_size = 16;
    cfg.n_layers = 32;
    cfg.n_heads = 32;
    cfg.head_dim = 64;
    cfg.max_blocks = 8;

    BlockManager bm(cfg, stream_);

    EXPECT_EQ(bm.config().block_size, 16);
    EXPECT_EQ(bm.config().n_layers, 32);
    EXPECT_EQ(bm.config().n_heads, 32);
    EXPECT_EQ(bm.config().head_dim, 64);
    EXPECT_EQ(bm.config().max_blocks, 8);
    EXPECT_EQ(bm.total_blocks(), 8);
    EXPECT_EQ(bm.free_blocks(), 8);
}

TEST_F(BlockManagerTest, InitGpuMemory) {
    BlockManager::Config cfg;
    cfg.max_blocks = 4;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    cfg.head_dim = 4;
    cfg.block_size = 2;

    BlockManager bm(cfg, stream_);
    bm.init_gpu_memory();

    // Verify GPU pointers are non-null after init
    EXPECT_NE(bm.k_cache(), nullptr);
    EXPECT_NE(bm.v_cache(), nullptr);
    EXPECT_NE(bm.block_table_gpu(), nullptr);
    EXPECT_NE(bm.seq_lens_gpu(), nullptr);

    cudaStreamSynchronize(stream_);
}

TEST_F(BlockManagerTest, AllocateBlocks) {
    BlockManager::Config cfg;
    cfg.max_blocks = 8;

    BlockManager bm(cfg, stream_);

    Sequence seq;
    seq.seq_id = 1;

    // Allocate 4 blocks
    bool ok = bm.allocate_blocks(seq, 4);
    EXPECT_TRUE(ok);
    EXPECT_EQ(seq.block_table.size(), 4);
    EXPECT_EQ(bm.free_blocks(), 4);

    // Verify block IDs are unique and in range
    for (auto bid : seq.block_table) {
        EXPECT_LT(bid, cfg.max_blocks);
    }
}

TEST_F(BlockManagerTest, GrowSequence) {
    BlockManager::Config cfg;
    cfg.max_blocks = 8;

    BlockManager bm(cfg, stream_);

    Sequence seq;
    seq.seq_id = 1;

    // Grow block by block
    for (int i = 0; i < 4; i++) {
        bool ok = bm.grow(seq);
        EXPECT_TRUE(ok);
        EXPECT_EQ(seq.block_table.size(), i + 1);
    }
    EXPECT_EQ(bm.free_blocks(), 4);
}

TEST_F(BlockManagerTest, FreeBlocks) {
    BlockManager::Config cfg;
    cfg.max_blocks = 8;

    BlockManager bm(cfg, stream_);

    Sequence seq;
    seq.seq_id = 1;

    ASSERT_TRUE(bm.allocate_blocks(seq, 3));
    EXPECT_EQ(bm.free_blocks(), 5);

    bm.free_blocks(seq);
    EXPECT_EQ(bm.free_blocks(), 8);
    EXPECT_TRUE(seq.block_table.empty());
}

TEST_F(BlockManagerTest, AllocateFreeGrowCycle) {
    BlockManager::Config cfg;
    cfg.max_blocks = 8;

    BlockManager bm(cfg, stream_);

    // Full cycle: allocate, grow, free, repeat
    for (int cycle = 0; cycle < 3; cycle++) {
        Sequence seq;
        seq.seq_id = cycle;

        // Allocate 3 blocks
        ASSERT_TRUE(bm.allocate_blocks(seq, 3));
        EXPECT_EQ(bm.free_blocks(), 5);

        // Grow 2 more
        EXPECT_TRUE(bm.grow(seq));
        EXPECT_TRUE(bm.grow(seq));
        EXPECT_EQ(bm.free_blocks(), 3);
        EXPECT_EQ(seq.block_table.size(), 5);

        // Free all
        bm.free_blocks(seq);
        EXPECT_EQ(bm.free_blocks(), 8);
    }
}

TEST_F(BlockManagerTest, OutOfBlocks) {
    BlockManager::Config cfg;
    cfg.max_blocks = 4;

    BlockManager bm(cfg, stream_);

    // Allocate all blocks
    Sequence seq1;
    seq1.seq_id = 1;
    ASSERT_TRUE(bm.allocate_blocks(seq1, 4));
    EXPECT_EQ(bm.free_blocks(), 0);

    // Try to allocate more — should fail
    Sequence seq2;
    seq2.seq_id = 2;
    bool ok = bm.allocate_blocks(seq2, 1);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(seq2.block_table.empty());

    // Try to grow — should fail
    ok = bm.grow(seq1);
    EXPECT_FALSE(ok);
    EXPECT_EQ(seq1.block_table.size(), 4);

    // Free seq1 and retry
    bm.free_blocks(seq1);
    EXPECT_EQ(bm.free_blocks(), 4);

    ok = bm.allocate_blocks(seq2, 2);
    EXPECT_TRUE(ok);
    EXPECT_EQ(seq2.block_table.size(), 2);
}

TEST_F(BlockManagerTest, MultipleSequences) {
    BlockManager::Config cfg;
    cfg.max_blocks = 12;

    BlockManager bm(cfg, stream_);

    Sequence seq_a;
    seq_a.seq_id = 100;
    Sequence seq_b;
    seq_b.seq_id = 200;
    Sequence seq_c;
    seq_c.seq_id = 300;

    ASSERT_TRUE(bm.allocate_blocks(seq_a, 4));
    ASSERT_TRUE(bm.allocate_blocks(seq_b, 3));
    ASSERT_TRUE(bm.allocate_blocks(seq_c, 2));
    EXPECT_EQ(bm.free_blocks(), 3);

    // Free seq_b, should return 3 blocks
    bm.free_blocks(seq_b);
    EXPECT_EQ(bm.free_blocks(), 6);

    // Grow seq_c with the freed blocks
    EXPECT_TRUE(bm.grow(seq_c));
    EXPECT_TRUE(bm.grow(seq_c));
    EXPECT_EQ(bm.free_blocks(), 4);
    EXPECT_EQ(seq_c.block_table.size(), 4);
}

TEST_F(BlockManagerTest, SyncBlockTables) {
    BlockManager::Config cfg;
    cfg.max_blocks = 16;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    cfg.head_dim = 4;
    cfg.block_size = 2;

    BlockManager bm(cfg, stream_);
    bm.init_gpu_memory();

    // Create sequences with known block tables
    std::vector<std::unique_ptr<Sequence>> seqs;
    constexpr int n_seqs = 3;
    constexpr int max_bps = 8;

    for (int i = 0; i < n_seqs; i++) {
        auto seq = std::make_unique<Sequence>();
        seq->seq_id = i;
        seq->seq_len = (i + 1) * 16;  // 16, 32, 48
        int n_blocks = i + 1;          // 1, 2, 3
        ASSERT_TRUE(bm.allocate_blocks(*seq, n_blocks));
        seqs.push_back(std::move(seq));
    }

    bm.sync_block_tables(seqs, max_bps);

    // Copy back from GPU and verify
    std::vector<int32_t> h_block_table(n_seqs * max_bps);
    std::vector<int32_t> h_seq_lens(n_seqs);

    cudaStreamSynchronize(stream_);

    cudaMemcpy(h_block_table.data(), bm.block_table_gpu(),
               n_seqs * max_bps * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_seq_lens.data(), bm.seq_lens_gpu(),
               n_seqs * sizeof(int32_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n_seqs; i++) {
        EXPECT_EQ(h_seq_lens[i], static_cast<int32_t>(seqs[i]->seq_len));
        for (int j = 0; j < static_cast<int>(seqs[i]->block_table.size()); j++) {
            EXPECT_EQ(h_block_table[i * max_bps + j],
                      static_cast<int32_t>(seqs[i]->block_table[j]));
        }
        // Remaining slots should be -1
        for (int j = seqs[i]->block_table.size(); j < max_bps; j++) {
            EXPECT_EQ(h_block_table[i * max_bps + j], -1);
        }
    }
}

TEST_F(BlockManagerTest, SequenceStatusDefaults) {
    Sequence seq;
    seq.seq_id = 42;
    EXPECT_EQ(seq.seq_id, 42U);
    EXPECT_EQ(seq.seq_len, 0U);
    EXPECT_EQ(seq.status, Sequence::WAITING);
    EXPECT_TRUE(seq.block_table.empty());
}

// Basic thread-safety test: multiple threads allocating and freeing
TEST_F(BlockManagerTest, ThreadSafetyBasic) {
    BlockManager::Config cfg;
    cfg.max_blocks = 64;

    BlockManager bm(cfg, stream_);

    constexpr int n_threads = 4;
    constexpr int n_ops = 50;

    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; t++) {
        threads.emplace_back([&bm, t]() {
            for (int i = 0; i < n_ops; i++) {
                Sequence seq;
                seq.seq_id = t * 1000 + i;
                // Try to allocate; if it fails, that's OK for this test
                bm.allocate_blocks(seq, 2);
                if (!seq.block_table.empty()) {
                    bm.grow(seq);
                    bm.free_blocks(seq);
                }
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    // After all ops, all blocks should be free
    EXPECT_EQ(bm.free_blocks(), bm.total_blocks());
}

}  // namespace
}  // namespace fish
