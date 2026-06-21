// src/engine/block_manager.cc
#include "engine/block_manager.h"
#include "utils/cuda_utils.h"
#include <spdlog/spdlog.h>
#include <algorithm>

namespace fish {

BlockManager::BlockManager(const Config& cfg, cudaStream_t stream)
    : cfg_(cfg), stream_(stream)
{
    // Initialize free list: all blocks free
    for (uint32_t i = 0; i < cfg_.max_blocks; i++) {
        free_list_.push(i);
    }
    spdlog::info("BlockManager: {} total blocks, {} MB KV cache",
                 cfg_.max_blocks, kv_cache_bytes() / (1024 * 1024));
}

BlockManager::~BlockManager() {
    if (k_cache_) cudaFree(k_cache_);
    if (v_cache_) cudaFree(v_cache_);
    if (block_table_gpu_) cudaFree(block_table_gpu_);
    if (seq_lens_gpu_) cudaFree(seq_lens_gpu_);
}

size_t BlockManager::kv_cache_bytes() const {
    // k_cache + v_cache
    return 2ULL * cfg_.max_blocks * cfg_.n_layers * cfg_.n_heads
           * cfg_.block_size * cfg_.head_dim * sizeof(__half);
}

void BlockManager::init_gpu_memory() {
    size_t kv_size = kv_cache_bytes();
    size_t per_cache = kv_size / 2;

    spdlog::info("Allocating {} MB for KV cache ({} MB per k/v)",
                 kv_size / (1024 * 1024), per_cache / (1024 * 1024));

    CUDA_CHECK(cudaMallocAsync(&k_cache_, per_cache, stream_));
    CUDA_CHECK(cudaMallocAsync(&v_cache_, per_cache, stream_));
    CUDA_CHECK(cudaMemsetAsync(k_cache_, 0, per_cache, stream_));
    CUDA_CHECK(cudaMemsetAsync(v_cache_, 0, per_cache, stream_));

    // Block table + seq lens for up to 64 concurrent seqs
    static constexpr int max_seqs = 64;
    static constexpr int max_blocks_per_seq = 256;  // 256*16 = 4096 tokens max
    CUDA_CHECK(cudaMallocAsync(&block_table_gpu_,
                               max_seqs * max_blocks_per_seq * sizeof(int32_t), stream_));
    CUDA_CHECK(cudaMallocAsync(&seq_lens_gpu_,
                               max_seqs * sizeof(int32_t), stream_));
}

bool BlockManager::allocate_blocks(Sequence& seq, int n_blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_list_.size() < static_cast<size_t>(n_blocks)) {
        return false;
    }
    for (int i = 0; i < n_blocks; i++) {
        uint32_t block_id = free_list_.top();
        free_list_.pop();
        seq.block_table.push_back(block_id);
    }
    return true;
}

bool BlockManager::grow(Sequence& seq) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_list_.empty()) return false;
    uint32_t block_id = free_list_.top();
    free_list_.pop();
    seq.block_table.push_back(block_id);
    return true;
}

void BlockManager::free_blocks(Sequence& seq) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (uint32_t block_id : seq.block_table) {
        free_list_.push(block_id);
    }
    seq.block_table.clear();
}

void BlockManager::sync_block_tables(
    const std::vector<std::unique_ptr<Sequence>>& seqs,
    int max_blocks_per_seq
) {
    if (!block_table_gpu_ || !seq_lens_gpu_) {
        spdlog::warn("sync_block_tables called before init_gpu_memory, skipping");
        return;
    }

    static constexpr int max_seqs = 64;
    int n = std::min(static_cast<int>(seqs.size()), max_seqs);

    std::vector<int32_t> h_block_table(n * max_blocks_per_seq, -1);
    std::vector<int32_t> h_seq_lens(n, 0);

    // Hold mutex while reading sequence state to prevent data race
    // with concurrent allocate/free/grow
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int i = 0; i < n; i++) {
            const auto& seq = seqs[i];
            h_seq_lens[i] = static_cast<int32_t>(seq->seq_len);
            size_t n_blk = std::min(seq->block_table.size(),
                                    static_cast<size_t>(max_blocks_per_seq));
            for (size_t j = 0; j < n_blk; j++) {
                h_block_table[i * max_blocks_per_seq + j] =
                    static_cast<int32_t>(seq->block_table[j]);
            }
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(block_table_gpu_, h_block_table.data(),
                               n * max_blocks_per_seq * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(seq_lens_gpu_, h_seq_lens.data(),
                               n * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream_));
}

}  // namespace fish
