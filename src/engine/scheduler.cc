// src/engine/scheduler.cc
#include "engine/scheduler.h"
#include <spdlog/spdlog.h>
#include <algorithm>

namespace fish {

Scheduler::Scheduler(const SchedulerConfig& cfg, BlockManager* block_mgr)
    : cfg_(cfg), block_mgr_(block_mgr)
{
    spdlog::info("Scheduler: max_batch={} max_seqs={}", cfg_.max_batch_size, cfg_.max_seqs);
}

Scheduler::~Scheduler() {
    // Release all active blocks back to the pool
    for (auto& entry : active_) {
        if (entry && entry->seq) {
            block_mgr_->free_blocks(*entry->seq);
        }
    }
}

bool Scheduler::submit(std::shared_ptr<InferenceRequest> req) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_.size() >= static_cast<size_t>(cfg_.max_seqs)) {
        spdlog::warn("Scheduler: queue full ({} pending), rejecting request {}",
                     pending_.size(), req->request_id);
        return false;
    }
    pending_.push(std::move(req));
    return true;
}

bool Scheduler::has_work() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !pending_.empty() || !active_.empty();
}

int Scheduler::step() {
    int finished_this_step = 0;

    // Phase 1: Clean up finished sequences
    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_.erase(
            std::remove_if(active_.begin(), active_.end(),
                [&](std::unique_ptr<ActiveEntry>& entry) -> bool {
                    if (!entry || !entry->req) return true;
                    if (entry->req->finished) {
                        block_mgr_->free_blocks(*entry->seq);
                        finished_this_step++;
                        spdlog::debug("Scheduler: request {} finished ({} tokens)",
                                      entry->req->request_id,
                                      entry->tokens_generated);
                        return true;
                    }
                    return false;
                }),
            active_.end());
    }

    // Phase 2: Admit pending requests into active slots
    {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!pending_.empty() &&
               active_.size() < static_cast<size_t>(cfg_.max_batch_size)) {
            auto req = pending_.front();
            pending_.pop();

            auto entry = std::make_unique<ActiveEntry>();
            entry->req = req;
            entry->seq = std::make_unique<Sequence>();
            entry->seq->seq_id = req->request_id;

            int prompt_len = static_cast<int>(req->prompt_tokens.size());
            int n_blocks = (prompt_len + req->max_new_tokens + cfg_.block_size - 1)
                           / cfg_.block_size + 1;

            if (!block_mgr_->allocate_blocks(*entry->seq, n_blocks)) {
                spdlog::warn("Scheduler: not enough blocks for request {} (need {})",
                             req->request_id, n_blocks);
                req->finished = true;  // mark as error
                finished_this_step++;
                continue;
            }

            entry->seq->seq_len = prompt_len;
            entry->seq->status = Sequence::RUNNING;
            active_.push_back(std::move(entry));

            spdlog::info("Scheduler: admitted request {} (prompt_len={}, {} blocks)",
                         req->request_id, prompt_len, n_blocks);
        }
    }

    return finished_this_step;
}

size_t Scheduler::active_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_.size();
}

void Scheduler::finish_request(uint64_t request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& entry : active_) {
        if (entry && entry->req && entry->req->request_id == request_id) {
            entry->req->finished = true;
            return;
        }
    }
    // Might also be in the pending queue — but that's fine, it'll
    // be caught when it's admitted. For now, just log.
    spdlog::debug("Scheduler: finish_request for unknown id {}", request_id);
}

}  // namespace fish
