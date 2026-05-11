#pragma once

#include "gc_engine.h"
#include "gc_gguf_loader.h"
#include "gc_hparams.h"
#include "gc_arch_runtime.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// ── gc_kv_pool_t ──────────────────────────────────────────────────────────────
// Backend-resident persistent KV cache.
//
// Architecture:
//   - One ggml_context (no_alloc) + one ggml_backend_buffer_t per backend device.
//   - Per-layer K and V: ggml_tensor* with shape [row_size, total_slots].
//       row_size   = n_kv_head * head_dim
//       total_slots = num_blocks * block_size
//   - Slots are laid out contiguously: slot [block_id * block_size + slot_in_block].
//   - Block table maps logical page index → physical block_id.
//   - KV writes happen via ggml_cpy nodes inside the step graph — no CPU copies.
//   - KV reads (attention) use ggml_view_2d views into this buffer — zero copy.
//
// This eliminates:
//   - std::vector<float> CPU ownership
//   - Per-slot memcpy scatter/gather on every forward pass
//   - CPU-side contiguous KV reconstruction
//
// Invariants:
//   - kv_ctx_ and kv_buf_ live for the lifetime of gc_graph_runner_t
//   - k_[il] and v_[il] are views backed by kv_buf_, never reallocated
//   - Null block (block_id=0) is never written; reads from it return zeros

struct gc_kv_pool_t {
    // Persistent context — holds tensor metadata for K/V layers.
    // Backed entirely by kv_buf_; no ggml_init alloc pool needed.
    ggml_context *         kv_ctx  = nullptr;
    ggml_backend_buffer_t  kv_buf  = nullptr;

    // Per-layer K/V tensors. Shape: F32 [row_size, total_slots].
    std::vector<ggml_tensor *> k;   // k[il]
    std::vector<ggml_tensor *> v;   // v[il]

    uint32_t n_layer    = 0;
    uint32_t num_blocks = 0;
    uint32_t block_size = 0;
    uint32_t n_kv_head  = 0;
    uint32_t head_dim   = 0;

    // Derived
    int64_t row_size    = 0;   // n_kv_head * head_dim (floats per slot)
    int64_t total_slots = 0;   // num_blocks * block_size

    // Initialize: allocate persistent KV tensors on the given backend.
    // Returns false on allocation failure.
    bool init(ggml_backend_t backend,
              uint32_t n_layer, uint32_t num_blocks, uint32_t block_size,
              uint32_t n_kv_head, uint32_t head_dim);

    void free();

    ~gc_kv_pool_t() { free(); }

    // Non-copyable
    gc_kv_pool_t()                               = default;
    gc_kv_pool_t(const gc_kv_pool_t &)            = delete;
    gc_kv_pool_t & operator=(const gc_kv_pool_t &) = delete;

    bool ok() const { return kv_buf != nullptr; }

    // Return byte offset into k[il] or v[il] for a given (block_id, slot_in_block).
    // Used to build ggml_view_1d for a single slot's row.
    size_t slot_byte_offset(int32_t block_id, uint32_t slot_in_block) const {
        const size_t slot_idx = (size_t)block_id * block_size + slot_in_block;
        return slot_idx * (size_t)row_size * sizeof(float);
    }

    // Return the physical slot index for a logical token position given the block table.
    size_t token_slot_idx(const std::vector<int32_t> & block_ids,
                          int token_pos) const {
        const int32_t blk_id  = (token_pos / (int)block_size < (int)block_ids.size())
                                 ? block_ids[(size_t)(token_pos / (int)block_size)] : 0;
        const uint32_t slot   = (uint32_t)(token_pos % (int)block_size);
        return (size_t)blk_id * block_size + slot;
    }
};

// ── gc_graph_runner_t ─────────────────────────────────────────────────────────
// Real model runner: backend-aware, persistent KV, unified step graph.
//
// Execution model (mirrors vllm GPUModelRunner lifecycle):
//   init()   — enumerate devices, init backend, allocate persistent KV tensors
//   execute()— for each step:
//              1. build one ggml_cgraph covering token_embd + all layers + lm_head
//              2. KV writes = ggml_cpy nodes from Kcur/Vcur into KV pool views
//              3. KV reads  = ggml_view_2d into persistent KV tensors
//              4. dispatch via ggml_gallocr + ggml_backend_graph_compute
//              5. read logits back to CPU, sample
//
// No CPU fallback paths, no per-layer micro-contexts, no F32 promotion loops.

class gc_graph_runner_t final : public gc_model_runner_t {
public:
    struct config_t {
        int num_kv_blocks = 256;
        int kv_block_size = 16;
        uint64_t seed              = 1;
        float    temperature       = 0.8f;
        int      top_k             = 40;
        float    top_p             = 0.95f;
        float    repetition_penalty = 1.1f;
        int      repetition_window  = 64;
        int      n_threads          = 4;    // CPU thread count
        bool     force_cpu          = false; // skip GPU/Metal, use CPU only
    };

    gc_graph_runner_t(const gc_model_loader_t * loader,
                      const gc_hparams_t      & hp,
                      config_t                  cfg);

    ~gc_graph_runner_t();

    bool              ok()    const { return ok_; }
    const std::string & error() const { return err_; }

    gc_model_output_t execute(const gc_batch_t & batch) override;
    int num_layers() const override { return (int)hp_.n_layer; }
    void release_request(const std::string & req_id) override;

    // Debug/validation only: raw logits read from the last executed entry,
    // before repetition penalty and sampling. Used by manual golden tests.
    const std::vector<float> & debug_last_logits() const { return last_logits_; }

private:
    const gc_model_loader_t * loader_ = nullptr;
    gc_hparams_t              hp_{};
    config_t                  cfg_{};
    bool                      ok_  = false;
    std::string               err_;

    // ── Backend ──────────────────────────────────────────────────────────────
    ggml_backend_t  backend_    = nullptr;   // primary (GPU/Metal/CPU)
    ggml_backend_t  backend_cpu_= nullptr;   // always CPU (fallback + inputs)
    bool            backend_is_cpu_ = true;  // true if primary == CPU

    // Graph allocator: reuses compute buffers across steps.
    ggml_gallocr_t  galloc_ = nullptr;

    // ── Persistent KV pool ────────────────────────────────────────────────────
    gc_kv_pool_t kv_pool_;

    // ── Static input tensors (CPU-backed, reused each step) ──────────────────
    // Avoids per-step malloc for the input ids and position ids.
    ggml_context *        inp_ctx_ = nullptr;
    ggml_backend_buffer_t inp_buf_ = nullptr;
    ggml_tensor *         inp_tokens_   = nullptr;  // [max_batch] I32
    ggml_tensor *         inp_pos_      = nullptr;  // [max_batch] I32

    int max_inp_tokens_ = 0;  // current capacity of inp_tokens_ / inp_pos_

    // ── Cached weight pointers ────────────────────────────────────────────────
    const gc_tensor_weight_t * w_tok_embd_    = nullptr;
    const gc_tensor_weight_t * w_output_      = nullptr;
    const gc_tensor_weight_t * w_output_norm_ = nullptr;

    // ── Arch params (computed once in init) ──────────────────────────────────
    gc_norm_params_t norm_params_{};
    gc_ffn_params_t  ffn_params_{};
    gc_rope_params_t rope_params_{};

    // ── Per-request sampling state ────────────────────────────────────────────
    struct req_state_t {
        std::vector<int32_t> recent_tokens;
    };
    std::mt19937_64                               rng_;
    std::unordered_map<std::string, req_state_t>  req_state_;
    std::vector<float>                            last_logits_;

    // ── Instrumentation ───────────────────────────────────────────────────────
    // Counters exposed for debugging / assertions.
    struct stats_t {
        uint64_t steps_executed      = 0;
        uint64_t kv_slots_written    = 0;  // total KV slot writes (n_layer * n_new)
        uint64_t graph_alloc_reused  = 0;  // times galloc reused existing buffer
        uint64_t graph_alloc_resized = 0;  // times galloc had to resize
    };
    stats_t stats_;

    // ── Lifecycle ─────────────────────────────────────────────────────────────
    bool init();
    void destroy();
    bool ensure_inp_capacity(int n_tokens);

    // ── Graph builder ─────────────────────────────────────────────────────────
    // Build the full forward graph for one request.
    // Returns the logits tensor (output, F32 [vocab_size, n_new]).
    // All tensors are no_alloc; allocation is deferred to galloc_.
    // *kq_mask_out receives the causal mask tensor so execute() can populate it
    // after ggml_gallocr_alloc_graph (that is when the tensor gets a buffer).
    ggml_tensor * build_graph(ggml_context * ctx,
                              ggml_cgraph  * gf,
                              const gc_batch_entry_t & e,
                              ggml_tensor ** kq_mask_out);

    // ── Weight view ───────────────────────────────────────────────────────────
    const gc_tensor_weight_t * find_w(gc_tensor_role_t role,
                                      const char * suffix,
                                      int layer = -1) const;

    ggml_tensor * weight_view(ggml_context * ctx,
                              const gc_tensor_weight_t * w) const;

    // ── Sampling ──────────────────────────────────────────────────────────────
    void    apply_rep_penalty(std::vector<float> & logits,
                               const req_state_t & st,
                               float rep_penalty, int rep_window) const;
    int32_t sample(std::vector<float> logits,
                   float temperature, int top_k, float top_p);
};
