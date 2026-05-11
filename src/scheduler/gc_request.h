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

// ── Sampling params (minimal — extended in Phase 8) ───────────────────────────

struct gc_sampling_params_t {
    int     max_tokens    = 512;
    float   temperature   = 1.0f;
    float   top_p         = 1.0f;
    int     top_k         = -1;
    bool    ignore_eos    = false;
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
