#pragma once

#include "gc_engine.h"
#include "gc_gguf_loader.h"
#include "gc_hparams.h"
#include "gc_arch_runtime.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// ── Paged KV buffer ───────────────────────────────────────────────────────────
// Flat F32 buffer: [n_layer * 2 * num_blocks * block_size * n_kv_head * head_dim]
// K = kv_idx 0, V = kv_idx 1. Block 0 is the null block (never written).

struct gc_kv_buf_t {
    std::vector<float> data;

    uint32_t n_layer    = 0;
    uint32_t num_blocks = 0;
    uint32_t block_size = 0;
    uint32_t n_kv_head  = 0;
    uint32_t head_dim   = 0;

    // Strides in floats
    size_t slot_stride       = 0;   // n_kv_head * head_dim
    size_t block_stride      = 0;   // block_size * slot_stride
    size_t layer_kv_stride   = 0;   // num_blocks * block_stride

    bool init(uint32_t n_layer, uint32_t num_blocks, uint32_t block_size,
              uint32_t n_kv_head, uint32_t head_dim);

    float * slot(uint32_t layer, int kv_idx, int32_t block_id, uint32_t slot_in_block) {
        return data.data()
            + (size_t)layer    * 2 * layer_kv_stride
            + (size_t)kv_idx   * layer_kv_stride
            + (size_t)block_id * block_stride
            + (size_t)slot_in_block * slot_stride;
    }
    const float * slot(uint32_t layer, int kv_idx, int32_t block_id, uint32_t slot_in_block) const {
        return data.data()
            + (size_t)layer    * 2 * layer_kv_stride
            + (size_t)kv_idx   * layer_kv_stride
            + (size_t)block_id * block_stride
            + (size_t)slot_in_block * slot_stride;
    }
};

// ── Graph runner (concrete gc_model_runner_t) ─────────────────────────────────
// Loads GGUF weights via gc_model_loader_t, builds a GGML graph per step,
// uses paged KV cache. Arch-agnostic: all arch-specific decisions go through
// gc_arch_runtime_* dispatchers.

class gc_graph_runner_t final : public gc_model_runner_t {
public:
    struct config_t {
        int num_kv_blocks = 256;   // must match scheduler kv block count
        int kv_block_size = 16;    // tokens per KV block
        uint64_t seed         = 1;
        float    temperature  = 0.8f;
        int      top_k        = 40;
        float    top_p        = 0.95f;
        float    repetition_penalty = 1.1f;
        int      repetition_window  = 64;
    };

    gc_graph_runner_t(const gc_model_loader_t * loader,
                      const gc_hparams_t      & hp,
                      config_t                  cfg);

    bool        ok()    const { return ok_; }
    const std::string & error() const { return err_; }

    gc_model_output_t execute(const gc_batch_t & batch) override;
    int num_layers() const override { return (int)hp_.n_layer; }

private:
    const gc_model_loader_t * loader_ = nullptr;
    gc_hparams_t              hp_{};
    config_t                  cfg_{};
    bool                      ok_  = false;
    std::string               err_;

    // Cached weight pointers (set in init)
    const gc_tensor_weight_t * w_tok_embd_    = nullptr;
    const gc_tensor_weight_t * w_output_      = nullptr;
    const gc_tensor_weight_t * w_output_norm_ = nullptr;

    // Paged KV buffer
    gc_kv_buf_t kv_buf_;

    // Arch-level cached params (computed once in init)
    gc_norm_params_t norm_params_{};
    gc_ffn_params_t  ffn_params_{};
    gc_rope_params_t rope_params_{};

    // Per-request sampling state
    struct req_state_t {
        std::vector<int32_t> recent_tokens;
    };
    std::mt19937_64 rng_;
    std::unordered_map<std::string, req_state_t> req_state_;

    bool init();

    const gc_tensor_weight_t * find_w(gc_tensor_role_t role,
                                      const char * suffix,
                                      int layer = -1) const;

    // Build forward graph for one request and return logits over vocab.
    // token_ids and position_ids must be aligned. block_ids is the paged KV table.
    bool forward(const std::vector<int32_t> & token_ids,
                 const std::vector<int32_t> & position_ids,
                 const std::vector<int32_t> & block_ids,
                 int                          num_computed,
                 bool                         is_prefill,
                 std::vector<float>         & logits_out);

    void   apply_rep_penalty(std::vector<float> & logits, const req_state_t & st) const;
    int32_t sample(std::vector<float> logits);
};
