#pragma once

#include "gc_kvcache.h"   // gc_block_hash_t, GC_BLOCK_HASH_NONE

#include <cstdint>
#include <string>
#include <vector>

// ── Request status ────────────────────────────────────────────────────────────

typedef enum gc_req_status_t {
    GC_REQ_WAITING    = 0,
    GC_REQ_RUNNING    = 1,
    GC_REQ_PREEMPTED  = 2,
    // finished (any value >= GC_REQ_FINISHED_STOPPED is finished)
    GC_REQ_FINISHED_STOPPED       = 10,
    GC_REQ_FINISHED_LENGTH_CAPPED = 11,
    GC_REQ_FINISHED_ABORTED       = 12,
} gc_req_status_t;

static inline bool gc_req_is_finished(gc_req_status_t s) {
    return s >= GC_REQ_FINISHED_STOPPED;
}

// Validate a state transition.  Returns true if the transition is legal.
// Legal transitions mirror vllm v1 SequenceStatus:
//   WAITING  → RUNNING   (first schedule / resume after preemption)
//   RUNNING  → PREEMPTED (KV eviction)
//   RUNNING  → FINISHED  (EOS / length / abort)
//   PREEMPTED→ WAITING   (re-queued for recompute)
// All other transitions are bugs.
static inline bool gc_req_transition_valid(gc_req_status_t from, gc_req_status_t to) {
    if (from == GC_REQ_WAITING   && to == GC_REQ_RUNNING)           return true;
    if (from == GC_REQ_RUNNING   && to == GC_REQ_PREEMPTED)         return true;
    if (from == GC_REQ_RUNNING   && gc_req_is_finished(to))         return true;
    if (from == GC_REQ_PREEMPTED && to == GC_REQ_WAITING)           return true;
    // Re-admit after preemption (re-scheduled directly to RUNNING)
    if (from == GC_REQ_PREEMPTED && to == GC_REQ_RUNNING)           return true;
    return false;
}

// ── Sampling params ────────────────────────────────────────────────────────────

struct gc_sampling_params_t {
    // Generation budget
    int     max_tokens    = 512;

    // Sampling controls (applied per-request by the model runner)
    float   temperature   = 1.0f;
    float   top_p         = 1.0f;
    int     top_k         = -1;      // -1 = disabled (sample from full vocab)
    float   min_p         = 0.0f;    // 0 = disabled
    float   repetition_penalty = 1.0f; // 1.0 = disabled
    int     repetition_window  = 64;
    uint64_t seed         = 0;       // 0 = use engine global seed

    // Stop conditions
    int32_t eos_token_id  = 2;       // < 0 → disable EOS-by-id
    bool    ignore_eos    = false;

    // Multi-token stop strings: each inner vector is a tokenized stop sequence.
    // Checked after every output token. First match wins.
    std::vector<std::vector<int32_t>> stop_token_ids;
};

// ── Request ───────────────────────────────────────────────────────────────────
// Mirrors vllm v1 Request, stripped of multimodal/LoRA/spec-decode fields.

struct gc_request_t {
    // Construction
    gc_request_t(
        std::string             request_id,
        std::vector<int32_t>    prompt_token_ids,
        gc_sampling_params_t    sampling_params = {},
        int                     priority        = 0,
        double                  arrival_time    = 0.0);

    // ── Identity ──────────────────────────────────────────────────────────────
    std::string          request_id;
    int                  priority      = 0;
    double               arrival_time  = 0.0;
    gc_req_status_t      status        = GC_REQ_WAITING;
    gc_sampling_params_t sampling_params;

    // ── Token state ───────────────────────────────────────────────────────────
    std::vector<int32_t> prompt_token_ids;
    std::vector<int32_t> output_token_ids;  // tokens generated so far

    int num_prompt_tokens()  const { return (int)prompt_token_ids.size(); }
    int num_output_tokens()  const { return (int)output_token_ids.size(); }
    int num_tokens()         const { return num_prompt_tokens() + num_output_tokens(); }
    int max_tokens()         const { return sampling_params.max_tokens; }

    // num_computed_tokens: tokens for which KV cache is filled/reserved
    int num_computed_tokens = 0;

    // ── Block hashes ─────────────────────────────────────────────────────────
    // Precomputed prefix-cache hashes, one per full block.
    std::vector<gc_block_hash_t> block_hashes;

    // Recompute block_hashes up to all full blocks in prompt+output.
    void update_block_hashes(int block_size);

    // Append one output token and update block_hashes if a new block is full.
    void append_output_token(int32_t token_id, int block_size);

    // ── Scheduling state ──────────────────────────────────────────────────────
    int  num_preemptions = 0;

    // Ordering for priority queue: lower priority value = higher priority.
    // Ties broken by earlier arrival_time.
    bool operator<(const gc_request_t & o) const {
        if (priority != o.priority) return priority < o.priority;
        return arrival_time < o.arrival_time;
    }
    bool operator>(const gc_request_t & o) const { return o < *this; }
};
