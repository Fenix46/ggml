#pragma once

#include "gc_rope.h"
#include "ggml.h"

#include <cstdint>
#include <functional>

struct ggml_cgraph;

// ── Enums ─────────────────────────────────────────────────────────────────────

typedef enum gc_ffn_op_t {
    GC_FFN_SILU,
    GC_FFN_GELU,
    GC_FFN_RELU,
    GC_FFN_RELU_SQR,
    GC_FFN_SWIGLU,
    GC_FFN_GEGLU,
    GC_FFN_REGLU,
} gc_ffn_op_t;

typedef enum gc_ffn_gate_t {
    GC_FFN_SEQ,   // gate is sequential (gated activation)
    GC_FFN_PAR,   // gate is parallel to up projection
} gc_ffn_gate_t;

typedef enum gc_norm_t {
    GC_NORM_LAYER,  // LayerNorm
    GC_NORM_RMS,    // RMSNorm
    GC_NORM_GROUP,  // GroupNorm
} gc_norm_t;

// ── Callback type (for debug / tensor naming) ─────────────────────────────────

// Called after each significant tensor is created. Implementations may name the
// tensor (ggml_set_name) or schedule it to a specific backend.
using gc_graph_cb_t = std::function<void(ggml_tensor * cur, const char * name, int il)>;

// ── Norm builder ──────────────────────────────────────────────────────────────

// Applies layer/RMS/group norm.
//   cur  — input tensor
//   w    — weight (scale), may be nullptr
//   b    — bias, may be nullptr
//   eps  — epsilon for numerical stability
struct ggml_tensor * gc_build_norm(
        struct ggml_context * ctx,
        struct ggml_tensor  * cur,
        struct ggml_tensor  * w,
        struct ggml_tensor  * b,
        gc_norm_t             type,
        float                 eps,
        const gc_graph_cb_t & cb,
        int il);

// ── FFN builder ───────────────────────────────────────────────────────────────

// Builds a standard FFN sublayer:
//   cur  → up → gate (optional) → activation → down
// All bias/scale tensors are optional (pass nullptr to skip).
struct ggml_tensor * gc_build_ffn(
        struct ggml_context * ctx,
        struct ggml_tensor  * cur,
        struct ggml_tensor  * up,   struct ggml_tensor * up_b,
        struct ggml_tensor  * gate, struct ggml_tensor * gate_b,
        struct ggml_tensor  * down, struct ggml_tensor * down_b,
        gc_ffn_op_t           type_op,
        gc_ffn_gate_t         type_gate,
        const gc_graph_cb_t & cb,
        int il);

// ── Attention parameter block ─────────────────────────────────────────────────

struct gc_attn_params_t {
    int64_t n_embd_head_q = 0;  // head dimension for Q (and K)
    int64_t n_embd_head_v = 0;  // head dimension for V (= n_embd_head_q for standard)
    int64_t n_head_q      = 0;  // number of Q heads
    int64_t n_head_kv     = 0;  // number of KV heads (< n_head_q for GQA/MQA)
    int64_t n_tokens      = 0;  // sequence length (batch size)

    float   kq_scale      = 1.0f;  // sqrt(1 / n_embd_head_q) typically

    bool    use_flash     = false;  // use ggml_flash_attn_ext
    float   max_alibi_bias= 0.0f;   // ALiBi max bias (0 = disabled)
    float   logit_softcap = 0.0f;   // Gemma2 logit softcapping (0 = disabled)
};

// ── MHA graph builder ─────────────────────────────────────────────────────────

// Builds scaled dot-product attention (no KV cache — pure forward pass).
// Inputs already-projected Q, K, V tensors and a causal mask.
//
//   q        — [n_embd_head_q, n_head_q,  n_tokens]
//   k        — [n_embd_head_q, n_head_kv, n_tokens]
//   v        — [n_embd_head_v, n_head_kv, n_tokens]  (v_trans = false)
//   kq_mask  — F32 [n_tokens, n_tokens] causal mask; pass nullptr for no mask
//   kq_b     — optional additive KQ bias [n_tokens, n_tokens]
//
// Returns [n_embd_head_v * n_head_q, n_tokens] (ready for output projection).
struct ggml_tensor * gc_build_attn_mha(
        struct ggml_context      * ctx,
        struct ggml_cgraph       * gf,
        struct ggml_tensor       * q,
        struct ggml_tensor       * k,
        struct ggml_tensor       * v,
        struct ggml_tensor       * kq_mask,
        struct ggml_tensor       * kq_b,
        const gc_attn_params_t   & p,
        const gc_graph_cb_t      & cb,
        int il);

// Full attention sublayer:
//   1. Q/K/V projections (wq, wk, wv — weight matrices; bias optional)
//   2. RoPE on Q and K
//   3. MHA
//   4. Output projection (wo; bias optional)
//
//   cur    — input residual [n_embd, n_tokens]
//   wq/wk/wv/wo — weight matrices; biases bq/bk/bv/bo optional (nullptr)
//   pos    — I32 position tensor [n_tokens]
//   kq_mask— causal mask [n_tokens, n_tokens]; nullptr = no mask
struct ggml_tensor * gc_build_attn(
        struct ggml_context      * ctx,
        struct ggml_cgraph       * gf,
        struct ggml_tensor       * cur,
        struct ggml_tensor       * wq,  struct ggml_tensor * bq,
        struct ggml_tensor       * wk,  struct ggml_tensor * bk,
        struct ggml_tensor       * wv,  struct ggml_tensor * bv,
        struct ggml_tensor       * wo,  struct ggml_tensor * bo,
        struct ggml_tensor       * pos,
        struct ggml_tensor       * kq_mask,
        const gc_attn_params_t   & ap,
        const gc_rope_params_t   & rp,
        const gc_graph_cb_t      & cb,
        int il);
