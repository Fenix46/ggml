#pragma once

#include "gc_hparams.h"
#include "../attention/gc_attention.h"
#include "../attention/gc_rope.h"

#include <cstdint>
#include <string>

// Central architecture dispatcher: callers never pass/select architecture
// explicitly. Dispatch is resolved from hp.arch loaded from GGUF metadata.

bool gc_arch_runtime_validate_hparams(
        const gc_hparams_t & hp,
        std::string * err_msg = nullptr);

bool gc_arch_runtime_build_attn_params(
        const gc_hparams_t & hp,
        uint32_t layer,
        uint32_t n_tokens,
        gc_attn_params_t & out,
        std::string * err_msg = nullptr);

bool gc_arch_runtime_build_rope_params(
        const gc_hparams_t & hp,
        gc_rope_params_t & out,
        std::string * err_msg = nullptr);

// ── Norm parameters ───────────────────────────────────────────────────────────

struct gc_norm_params_t {
    gc_norm_t type     = GC_NORM_RMS;
    float     eps      = 1e-5f;
};

bool gc_arch_runtime_build_norm_params(
        const gc_hparams_t & hp,
        gc_norm_params_t   & out,
        std::string        * err_msg = nullptr);

// ── FFN parameters ────────────────────────────────────────────────────────────

struct gc_ffn_params_t {
    gc_ffn_op_t   act       = GC_FFN_SILU;
    gc_ffn_gate_t gate_mode = GC_FFN_PAR;
    bool          has_gate  = true;   // false → no gate projection (GPT-2 style)
};

bool gc_arch_runtime_build_ffn_params(
        const gc_hparams_t & hp,
        gc_ffn_params_t    & out,
        std::string        * err_msg = nullptr);

// ── Per-layer structural flags ────────────────────────────────────────────────
// Controls which optional sublayers are present at a given layer index.

struct gc_layer_flags_t {
    bool has_post_attn_norm = false;   // norm between attn output and residual add
    bool has_post_ffn_norm  = false;   // norm between ffn output and residual add
    bool has_attn_q_norm    = false;   // per-head Q norm (e.g. Gemma3)
    bool has_attn_k_norm    = false;   // per-head K norm
    bool apply_rope         = true;    // RoPE on Q/K
    bool causal             = true;    // causal attention mask
};

bool gc_arch_runtime_build_layer_flags(
        const gc_hparams_t & hp,
        uint32_t             layer,
        gc_layer_flags_t   & out,
        std::string        * err_msg = nullptr);
