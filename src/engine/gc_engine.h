#pragma once

#include "gc_scheduler.h"   // gc_scheduler_t, gc_sched_output_t, gc_request_t
#include "gc_kvcache.h"     // gc_block_hash_t

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

// ── Batch input (scheduler → model runner) ────────────────────────────────────
// One entry per scheduled request-slot in the current step.
// Phase 10 will add position_ids, attention_mask, etc. from the arch layer.

struct gc_batch_entry_t {
    std::string          req_id;
    std::vector<int32_t> token_ids;         // tokens to process this step
    std::vector<int32_t> block_ids;         // KV block ids for this request
    int                  num_computed       = 0;  // tokens already in KV cache
    int                  num_new_tokens     = 0;  // tokens in token_ids this step
    bool                 is_prefill         = false;  // true if still in prefill phase
};

struct gc_batch_t {
    std::vector<gc_batch_entry_t> entries;
    int total_tokens = 0;
    bool empty() const { return entries.empty(); }
};

// ── Model output (model runner → engine) ─────────────────────────────────────
// sampled_tokens[i] = token sampled for entries[i] in the batch.
// -1 means "no token produced" (prefill-only step for that req).

struct gc_model_output_t {
    std::vector<std::string> req_ids;
    std::vector<int32_t>     sampled_tokens;  // parallel to req_ids; -1 = no sample
};

// ── Engine output (engine → caller) ──────────────────────────────────────────
// One entry per request that produced output this step.

struct gc_req_output_t {
    std::string     req_id;
    int32_t         token        = -1;     // token generated (-1 = prefill step)
    bool            finished     = false;
    gc_req_status_t finish_reason = GC_REQ_RUNNING;
};

struct gc_step_output_t {
    std::vector<gc_req_output_t> outputs;  // one per active request
    int total_scheduled_tokens = 0;
    bool had_work = false;
};

// ── Model runner interface ────────────────────────────────────────────────────
// Abstraction boundary between engine and actual GGML forward pass.
// Phase 10 provides a concrete implementation per architecture.
// Tests can inject a mock.

class gc_model_runner_t {
public:
    virtual ~gc_model_runner_t() = default;

    // Execute one forward step.  Returns sampled token per request.
    // Implementations may return -1 for prefill-only entries.
    virtual gc_model_output_t execute(const gc_batch_t & batch) = 0;

    // Number of KV layers (used to validate block_ids size).
    virtual int num_layers() const = 0;
};

// ── Engine params ─────────────────────────────────────────────────────────────

struct gc_engine_params_t {
    gc_scheduler_params_t sched;  // passed to gc_scheduler_t
};

// ── Engine ────────────────────────────────────────────────────────────────────
// Mirrors vllm v1 EngineCore.step():
//   schedule() → build_batch() → execute_model() → update_from_output()

class gc_engine_t {
public:
    gc_engine_t(const gc_engine_params_t & p, gc_model_runner_t * runner);
    ~gc_engine_t() = default;

    gc_engine_t(const gc_engine_t &)            = delete;
    gc_engine_t & operator=(const gc_engine_t &) = delete;

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    void add_request(std::unique_ptr<gc_request_t> req);
    void abort_request(const std::string & req_id);

    // ── Main loop step ────────────────────────────────────────────────────────
    // Returns outputs for this step (empty if nothing was scheduled).
    gc_step_output_t step();

    // ── Queries ───────────────────────────────────────────────────────────────
    bool  has_work()      const { return sched_.has_work(); }
    int   num_waiting()   const { return sched_.num_waiting(); }
    int   num_running()   const { return sched_.num_running(); }
    float kv_usage()      const { return sched_.kv_usage(); }

    const gc_engine_params_t & params() const { return params_; }

    // KV block info for a request (null if not running).
    const gc_req_blocks_t * get_blocks(const std::string & req_id) const;

private:
    gc_engine_params_t  params_;
    gc_scheduler_t      sched_;
    gc_model_runner_t * runner_;   // non-owning; caller manages lifetime

    // Build batch from scheduler output.
    gc_batch_t build_batch(const gc_sched_output_t & sched_out) const;

    // Merge model output back → scheduler update.
    gc_step_output_t process_output(
        const gc_sched_output_t  & sched_out,
        const gc_model_output_t  & model_out);
};
