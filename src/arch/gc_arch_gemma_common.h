#pragma once

#include "gc_hparams.h"
#include "../attention/gc_attention.h"
#include "../attention/gc_rope.h"

#include <cstdint>
#include <string>

struct gc_gemma_variant_cfg_t {
    float explicit_attn_scale = 0.0f; // >0 forces this scale
    bool  use_llama27b_rule   = false; // Gemma2/Gemma3: 27B variant scaling rule
    bool  causal_attn_required = true;
};

bool gc_arch_gemma_common_validate(
        const gc_hparams_t & hp,
        const gc_gemma_variant_cfg_t & cfg,
        std::string * err_msg = nullptr);

bool gc_arch_gemma_common_build_attn(
        const gc_hparams_t & hp,
        const gc_gemma_variant_cfg_t & cfg,
        uint32_t layer,
        uint32_t n_tokens,
        gc_attn_params_t & out,
        std::string * err_msg = nullptr);

bool gc_arch_gemma_common_build_rope(
        const gc_hparams_t & hp,
        gc_rope_params_t & out,
        std::string * err_msg = nullptr);
