// src/engine/scheduler.h
#pragma once
#include "engine/block_manager.h"
#include <cstdint>
#include <vector>
#include <memory>
#include <queue>
#include <mutex>
#include <functional>

namespace fish {

struct SchedulerConfig {
    int max_batch_size = 8;             // max sequences per batch
    int max_seqs = 64;                  // max concurrent sequences
    int max_blocks_per_seq = 256;       // 256*16=4096 tokens
    int block_size = 16;
};

// A request to be processed by the scheduler
struct InferenceRequest {
    uint64_t request_id;
    std::vector<int32_t> prompt_tokens;  // tokenized prompt
    int max_new_tokens = 512;

    // Sampling parameters
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 50;
    int seed = 42;

    // Output
    std::vector<int32_t> generated_codes;  // [stride * code_len] flat
    bool finished = false;
};

class Scheduler {
public:
    Scheduler(const SchedulerConfig& cfg, BlockManager* block_mgr);
    ~Scheduler();

    // Submit a request for processing. Returns false if the queue is full.
    bool submit(std::shared_ptr<InferenceRequest> req);

    // Process one batch: schedule prefill or decode for active sequences.
    // Returns the number of requests that finished this step.
    // This is a simplified single-sequence scheduler for now;
    // full batching support is wired in but delegates to serial execution.
    int step();

    // Check if there are any pending or active requests
    bool has_work() const;

    // Access the block manager
    BlockManager* block_mgr() { return block_mgr_; }

    // Get read-only count of active entries
    size_t active_count() const;

    // Mark a specific request as finished (used by pipeline after decode completes)
    void finish_request(uint64_t request_id);

private:
    SchedulerConfig cfg_;
    BlockManager* block_mgr_;

    // Pending request queue (FIFO)
    std::queue<std::shared_ptr<InferenceRequest>> pending_;
    mutable std::mutex mutex_;

    // Active sequences mapped to requests
    struct ActiveEntry {
        std::unique_ptr<Sequence> seq;
        std::shared_ptr<InferenceRequest> req;
        int tokens_generated = 0;
    };
    std::vector<std::unique_ptr<ActiveEntry>> active_;
};

}  // namespace fish
