#pragma once
#include "gc_arch_gemma.h"
bool gc_arch_gemma2_validate_hparams(const gc_hparams_t & hp, std::string * err_msg = nullptr);
bool gc_arch_gemma2_build_attn_params(const gc_hparams_t & hp, uint32_t layer, uint32_t n_tokens, gc_attn_params_t & out, std::string * err_msg = nullptr);
bool gc_arch_gemma2_build_rope_params(const gc_hparams_t & hp, gc_rope_params_t & out, std::string * err_msg = nullptr);
