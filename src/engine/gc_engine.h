#pragma once

#include "gc_scheduler.h"   // gc_scheduler_t, gc_sched_output_t, gc_request_t
#include "gc_kvcache.h"     // gc_block_hash_t

#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// ── Batch input (scheduler → model runner) ────────────────────────────────────
// One entry per scheduled request-slot in the current step.
// Phase 10 will add position_ids, attention_mask, etc. from the arch layer.

struct gc_batch_entry_t {
    std::string          req_id;
    std::vector<int32_t> token_ids;         // tokens to process this step
    std::vector<int32_t> position_ids;      // absolute position for each token (for RoPE)
    // COMPLETE logical block table: block_ids[i] covers tokens
    //   [i*block_size, (i+1)*block_size).  Always the full table.
    std::vector<int32_t> block_ids;
    int                  num_computed       = 0;  // tokens already in KV cache
    int                  num_new_tokens     = 0;  // tokens in token_ids this step
    int                  num_prompt_tokens  = 0;  // total prompt length (for final-prefill detection)
    bool                 is_prefill         = false;  // true if still processing prompt tokens
    // True when this prefill step covers the last chunk of the prompt.
    // The runner MUST sample a token when is_last_prefill=true (it produces
    // the first output token, which seeds the decode loop).
    bool                 is_last_prefill    = false;
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

class gc_ggml_model_runner_t;

// Minimal concrete runner used by server/integration wiring before the
// architecture-specific GGML runner is fully implemented.
class gc_default_model_runner_t final : public gc_model_runner_t {
public:
    struct config_t {
        int num_layers = 1;
        // Kept for backward compatibility; default runner now delegates to
        // the GGML-oriented sampler runner across the framework.
        int32_t decode_token = 42;
        bool emit_token_on_prefill = false;
    };

    gc_default_model_runner_t();
    explicit gc_default_model_runner_t(config_t cfg);

    gc_model_output_t execute(const gc_batch_t & batch) override;
    int num_layers() const override;

private:
    config_t cfg_;
    std::unique_ptr<gc_ggml_model_runner_t> impl_;
};

// GGML-oriented runner with real sampling controls.
// This is a production-facing runner scaffold: it manages per-request sampling
// state and applies top-k/top-p/temperature/repetition-penalty on logits.
class gc_ggml_model_runner_t final : public gc_model_runner_t {
public:
    struct config_t {
        int num_layers = 1;
        int vocab_size = 32000;
        uint64_t seed = 1;
        float temperature = 1.0f;
        int top_k = 40;
        float top_p = 0.95f;
        float repetition_penalty = 1.1f;
        int repetition_window = 64;
    };

    gc_ggml_model_runner_t();
    explicit gc_ggml_model_runner_t(config_t cfg);
    gc_model_output_t execute(const gc_batch_t & batch) override;
    int num_layers() const override;

private:
    struct req_state_t {
        std::vector<int32_t> recent_tokens;
        uint64_t step_count = 0;
    };

    config_t cfg_;
    std::mt19937_64 rng_;
    std::unordered_map<std::string, req_state_t> req_state_;

    std::vector<float> build_logits(const gc_batch_entry_t & e, req_state_t & st) const;
    int32_t sample_token(std::vector<float> logits);
    void apply_repetition_penalty(std::vector<float> & logits, const req_state_t & st) const;
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
