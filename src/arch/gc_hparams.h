#pragma once

#include "gc_arch.h"

#include <cstdint>
#include <vector>

// ── Model hyperparameters ────────────────────────────────────────────────────
// Populated by gc_hparams_load() from a gc_model_loader_t.
// Per-layer arrays have size n_layer; a scalar fallback is used when the
// array is empty.

struct gc_hparams_t {
    gc_arch_t arch = GC_ARCH_UNKNOWN;

    // Core dimensions
    uint32_t n_vocab       = 0;
    uint32_t n_ctx_train   = 0;  // context length seen during training
    uint32_t n_embd        = 0;
    uint32_t n_layer       = 0;
    uint32_t n_ff          = 0;  // default feed-forward width

    // Attention — per-layer arrays (empty → use scalar)
    uint32_t n_head        = 0;
    uint32_t n_head_kv     = 0;
    uint32_t n_embd_head_k = 0;  // key head dim  (0 → n_embd / n_head)
    uint32_t n_embd_head_v = 0;  // value head dim (0 → n_embd / n_head)

    std::vector<uint32_t> n_head_arr;     // per-layer overrides
    std::vector<uint32_t> n_head_kv_arr;
    std::vector<uint32_t> n_ff_arr;

    // MLA (DeepSeek)
    uint32_t n_lora_q      = 0;
    uint32_t n_lora_kv     = 0;
    uint32_t n_embd_k_mla  = 0;
    uint32_t n_embd_v_mla  = 0;

    // MoE
    uint32_t n_expert      = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_expert_shared = 0;
    uint32_t n_ff_exp      = 0;
    uint32_t n_ff_shexp    = 0;
    uint32_t moe_every_n_layers     = 0;
    uint32_t leading_dense_block_count = 0;

    // Normalisation
    float    f_norm_eps     = 1e-5f;
    float    f_norm_rms_eps = 1e-5f;

    // Softcapping / scaling
    float    f_logit_scale            = 0.0f;
    float    f_residual_scale         = 0.0f;
    float    f_embedding_scale        = 0.0f;
    float    f_attn_logit_softcapping = 0.0f;
    float    f_final_logit_softcapping = 0.0f;
    float    f_attn_scale             = 0.0f;
    float    f_max_alibi_bias         = 0.0f;
    float    f_clamp_kqv              = 0.0f;

    // RoPE
    uint32_t n_rot          = 0;  // rope dimension count (0 → n_embd_head_k)
    float    rope_freq_base = 10000.0f;
    float    rope_scale_linear = 1.0f;

    // Sliding window attention
    uint32_t n_swa          = 0;  // 0 = disabled
    bool     causal_attn    = true;
    bool     use_par_res    = false;

    // SSM (Mamba)
    uint32_t ssm_d_conv     = 0;
    uint32_t ssm_d_inner    = 0;
    uint32_t ssm_d_state    = 0;
    uint32_t ssm_dt_rank    = 0;
    uint32_t ssm_n_group    = 1;

    // LFM2 short-conv
    uint32_t shortconv_l_cache = 0;

    // ── Helpers ───────────────────────────────────────────────────────────────
    uint32_t n_head_for_layer   (uint32_t layer) const;
    uint32_t n_head_kv_for_layer(uint32_t layer) const;
    uint32_t n_ff_for_layer     (uint32_t layer) const;

    uint32_t n_embd_k_head() const { return n_embd_head_k ? n_embd_head_k : (n_head ? n_embd / n_head : 0); }
    uint32_t n_embd_v_head() const { return n_embd_head_v ? n_embd_head_v : (n_head ? n_embd / n_head : 0); }
};

// ── Loader ───────────────────────────────────────────────────────────────────

// Forward declaration — avoids pulling in the full loader header here.
struct gc_model_loader_t;

// Populate hp from the metadata in loader.  Returns false on hard errors.
bool gc_hparams_load(gc_hparams_t & hp, const gc_model_loader_t & loader);
