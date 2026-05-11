#pragma once

#include "gc_hparams.h"
#include "../attention/gc_attention.h"
#include "../attention/gc_rope.h"

#include <cstdint>
#include <string>

// Validate core LLaMA-family hyperparameters before graph construction.
bool gc_arch_llama_validate_hparams(const gc_hparams_t & hp, std::string * err_msg = nullptr);

// Build per-layer attention parameters from loaded hparams.
bool gc_arch_llama_build_attn_params(
        const gc_hparams_t & hp,
        uint32_t layer,
        uint32_t n_tokens,
        gc_attn_params_t & out,
        std::string * err_msg = nullptr);

// Build RoPE parameters for LLaMA-family attention.
bool gc_arch_llama_build_rope_params(
        const gc_hparams_t & hp,
        gc_rope_params_t & out,
        std::string * err_msg = nullptr);
