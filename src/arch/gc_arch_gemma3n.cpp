#include "gc_arch_gemma3n.h"
#include "gc_arch_gemma_common.h"

static const gc_gemma_variant_cfg_t GC_CFG = { 1.0f, false, true };

bool gc_arch_gemma3n_validate_hparams(const gc_hparams_t & hp, std::string * err_msg) {
    if (hp.arch != GC_ARCH_GEMMA3N) {
        if (err_msg) *err_msg = "gc_arch_gemma3n requires GC_ARCH_GEMMA3N";
        return false;
    }
    return gc_arch_gemma_common_validate(hp, GC_CFG, err_msg);
}

bool gc_arch_gemma3n_build_attn_params(const gc_hparams_t & hp, uint32_t layer, uint32_t n_tokens, gc_attn_params_t & out, std::string * err_msg) {
    return gc_arch_gemma_common_build_attn(hp, GC_CFG, layer, n_tokens, out, err_msg);
}

bool gc_arch_gemma3n_build_rope_params(const gc_hparams_t & hp, gc_rope_params_t & out, std::string * err_msg) {
    return gc_arch_gemma_common_build_rope(hp, out, err_msg);
}
