#pragma once

#include "gc_request.h"
#include "gc_kvcache.h"

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ── Scheduling policy ─────────────────────────────────────────────────────────

typedef enum gc_sched_policy_t {
    GC_SCHED_FCFS     = 0,   // first-come-first-served
    GC_SCHED_PRIORITY = 1,   // lower priority value = higher priority
} gc_sched_policy_t;

// ── Scheduler output ──────────────────────────────────────────────────────────
// Mirrors vllm v1 SchedulerOutput, stripped of multimodal/spec-decode/PP fields.

struct gc_new_req_data_t {
    std::string           req_id;
    std::vector<int32_t>  prompt_token_ids;
    gc_sampling_params_t  sampling_params;
    std::vector<int32_t>  block_ids;         // flat: one KV group
    int                   num_computed_tokens = 0;
};

struct gc_cached_req_entry_t {
    std::string          req_id;
    std::vector<int32_t> new_block_ids;     // blocks newly allocated this step
    int                  num_computed_tokens = 0;
    int                  num_output_tokens   = 0;
    int                  num_prompt_tokens   = 0;
};

struct gc_sched_output_t {
    std::vector<gc_new_req_data_t>    new_reqs;      // first-time scheduled
    std::vector<gc_cached_req_entry_t> cached_reqs;  // running / resumed

    // req_id → num tokens scheduled this step
    std::unordered_map<std::string, int> num_scheduled_tokens;
    int total_scheduled_tokens = 0;

    // Requests that finished between previous and current step
    std::unordered_set<std::string> finished_req_ids;

    // Requests preempted this step
    std::unordered_set<std::string> preempted_req_ids;

    bool empty() const { return total_scheduled_tokens == 0; }
};

// ── Scheduler params ──────────────────────────────────────────────────────────

struct gc_scheduler_params_t {
    // KV cache
    int  num_blocks        = 256;
    int  block_size        = 16;

    // Scheduling limits
    int  max_num_running   = 256;         // max concurrent requests
    int  max_num_tokens    = 4096;        // max tokens per scheduling step (chunked prefill)
    int  max_model_len     = 4096;

    // Policy
    gc_sched_policy_t policy = GC_SCHED_FCFS;

    // Prefix caching
    bool enable_prefix_cache = true;
};

// ── Scheduler ─────────────────────────────────────────────────────────────────
// Owns gc_kv_manager_t and all request state.
// Thread-safety: single-threaded (same as vllm v1 scheduler).

class gc_scheduler_t {
public:
    explicit gc_scheduler_t(const gc_scheduler_params_t & p);
    ~gc_scheduler_t() = default;

    gc_scheduler_t(const gc_scheduler_t &)            = delete;
    gc_scheduler_t & operator=(const gc_scheduler_t &) = delete;

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    // Add a new request to the waiting queue.
    void add_request(std::unique_ptr<gc_request_t> req);

    // Abort a request by id (remove from wherever it lives).
    void abort_request(const std::string & req_id);

    // ── Main scheduling step ──────────────────────────────────────────────────
    gc_sched_output_t schedule();

    // ── Post-step update ──────────────────────────────────────────────────────
    // Called by the engine after model execution:
    //   new_token_ids[i] = token generated for running_req[i] this step.
    // Handles: finish detection, output_token append, block_hash update.
    void update_from_output(
        const std::unordered_map<std::string, int32_t> & new_tokens);

    // ── Queries ───────────────────────────────────────────────────────────────
    int  num_waiting()  const;
    int  num_running()  const;
    bool has_work()     const { return num_waiting() > 0 || num_running() > 0; }
    float kv_usage()    const { return kv_mgr_.usage(); }

    const gc_scheduler_params_t & params() const { return params_; }

    // Proxy: used by gc_engine_t to expose KV block info per request.
    const gc_req_blocks_t * kv_mgr_proxy(const std::string & req_id) const {
        return kv_mgr_.get_blocks(req_id);
    }

private:
    gc_scheduler_params_t params_;
    gc_kv_manager_t       kv_mgr_;

    // ── Request storage ───────────────────────────────────────────────────────
    // Ownership: all requests live here.
    std::unordered_map<std::string, std::unique_ptr<gc_request_t>> all_reqs_;

    // FCFS queue: deque of raw pointers (owned by all_reqs_)
    std::deque<gc_request_t *> waiting_fcfs_;

    // Priority queue: max-heap by operator> (highest priority = smallest value)
    struct PriorityLess {
        bool operator()(const gc_request_t * a, const gc_request_t * b) const {
            return *a > *b;  // min-heap by < (lower priority value = front)
        }
    };
    std::priority_queue<gc_request_t *, std::vector<gc_request_t *>, PriorityLess>
        waiting_prio_;

    std::vector<gc_request_t *> running_;   // ordered: index 0 = oldest

    // Requests finished since last schedule() call (to include in output)
    std::unordered_set<std::string> finished_req_ids_;

    // ── Helpers ───────────────────────────────────────────────────────────────

    bool         waiting_empty()   const;
    gc_request_t * waiting_front() const;
    gc_request_t * waiting_pop();
    void           waiting_push_front(gc_request_t * req);  // preempt / re-queue

    void   preempt(gc_request_t * req);
    void   finish_request(gc_request_t * req, gc_req_status_t status);

    // How many new tokens can be scheduled for this request this step.
    int  num_new_tokens_for(const gc_request_t * req, int token_budget) const;

    // Build gc_new_req_data_t from a newly scheduled request.
    gc_new_req_data_t make_new_req_data(const gc_request_t * req) const;

    // Build gc_cached_req_entry_t for a running/resumed request.
    gc_cached_req_entry_t make_cached_entry(
        const gc_request_t * req,
        const std::vector<int32_t> & new_block_ids) const;
};
