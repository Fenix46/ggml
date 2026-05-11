#pragma once

#include "ggml.h"

#include <cstdint>

// ── RoPE scaling variants ─────────────────────────────────────────────────────

typedef enum gc_rope_scaling_t {
    GC_ROPE_SCALING_NONE    = 0,  // no scaling
    GC_ROPE_SCALING_LINEAR  = 1,  // linear freq_scale
    GC_ROPE_SCALING_YARN    = 2,  // YaRN (context-window extension)
    GC_ROPE_SCALING_LONGROPE= 3,  // LongRoPE (per-dim factors)
} gc_rope_scaling_t;

// ── RoPE parameters ───────────────────────────────────────────────────────────

struct gc_rope_params_t {
    int     n_dims      = 0;           // rotated dimensions (= n_rot from hparams)
    int     mode        = GGML_ROPE_TYPE_NEOX; // GGML_ROPE_TYPE_NORMAL/NEOX/MROPE/etc.
    int     n_ctx_orig  = 0;           // original training context length (for YaRN)

    float   freq_base   = 10000.0f;
    float   freq_scale  = 1.0f;        // inverse of the scaling factor
    float   ext_factor  = 0.0f;        // YaRN ext_factor (0 = disabled)
    float   attn_factor = 1.0f;        // YaRN attention scaling
    float   beta_fast   = 32.0f;       // YaRN beta_fast
    float   beta_slow   = 1.0f;        // YaRN beta_slow
};

// ── API ───────────────────────────────────────────────────────────────────────

// Apply standard RoPE (ggml_rope_ext wrapper).
//   a        — input tensor  [n_embd_head, n_head, n_tokens]
//   pos      — I32 position tensor [n_tokens]
//   factors  — optional per-dimension freq factors (LongRoPE); pass nullptr for none
// Returns the rotated tensor.
struct ggml_tensor * gc_rope_apply(
        struct ggml_context   * ctx,
        struct ggml_tensor    * a,
        struct ggml_tensor    * pos,
        struct ggml_tensor    * factors,
        const gc_rope_params_t & p);

// Apply multi-dimensional RoPE (ggml_rope_multi wrapper, for Qwen3/M-RoPE).
//   sections — 4-element array: [t, y, x, 0] for MROPE; [y, x, 0, 0] for VISION
struct ggml_tensor * gc_rope_apply_multi(
        struct ggml_context   * ctx,
        struct ggml_tensor    * a,
        struct ggml_tensor    * pos,
        struct ggml_tensor    * factors,
        const gc_rope_params_t & p,
        int sections[GGML_MROPE_SECTIONS]);
