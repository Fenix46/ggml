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
