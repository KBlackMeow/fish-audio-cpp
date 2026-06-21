// src/engine/block_manager.h
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <stack>
#include <mutex>
#include <memory>
#include <stdexcept>

namespace fish {

struct Sequence {
    uint32_t seq_id;
    std::vector<uint32_t> block_table;  // logical->physical block mapping
    uint32_t seq_len = 0;
    enum Status { WAITING, RUNNING, FINISHED } status = WAITING;
};

class BlockManager {
public:
    struct Config {
        int block_size = 16;      // tokens per block
        int n_layers = 32;
        int n_heads = 32;
        int head_dim = 64;
        size_t max_blocks = 1024; // total physical blocks in GPU pool
    };

    BlockManager(const Config& cfg, cudaStream_t stream);
    ~BlockManager();

    // Initialize GPU memory pools (allocate k_cache and v_cache)
    void init_gpu_memory();

    // Allocate N blocks for a sequence (prefill).
    // Returns false if not enough free blocks.
    bool allocate_blocks(Sequence& seq, int n_blocks);

    // Grow sequence by one block (decode).
    bool grow(Sequence& seq);

    // Free all blocks of a sequence.
    void free_blocks(Sequence& seq);

    // Access GPU caches
    __half* k_cache() { return k_cache_; }
    __half* v_cache() { return v_cache_; }
    int32_t* block_table_gpu() { return block_table_gpu_; }
    int32_t* seq_lens_gpu() { return seq_lens_gpu_; }

    // Sync block tables for active sequences to GPU.
    // Call before each batch step.
    void sync_block_tables(const std::vector<std::unique_ptr<Sequence>>& seqs,
                           int max_blocks_per_seq);

    // Available blocks
    size_t free_blocks() const { return free_list_.size(); }
    size_t total_blocks() const { return cfg_.max_blocks; }

    const Config& config() const { return cfg_; }

private:
    Config cfg_;
    cudaStream_t stream_;

    // CPU-side free list
    std::stack<uint32_t> free_list_;
    std::mutex mutex_;

    // GPU-side memory
    __half* k_cache_ = nullptr;        // [max_blocks, n_layers, n_heads, block_size, head_dim]
    __half* v_cache_ = nullptr;        // same layout
    int32_t* block_table_gpu_ = nullptr;  // [max_seqs=64, max_blocks_per_seq]
    int32_t* seq_lens_gpu_ = nullptr;     // [max_seqs=64]

    size_t kv_cache_bytes() const;
};

}  // namespace fish
