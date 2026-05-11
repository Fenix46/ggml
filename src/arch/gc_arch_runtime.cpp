#include "gc_arch_runtime.h"

#include "gc_arch.h"
#include "gc_arch_llama.h"

#include <cstdio>

static bool gc__unsupported_arch(gc_arch_t arch, std::string * err_msg) {
    const char * name = gc_arch_name(arch);
    if (err_msg) {
        *err_msg = "architecture not implemented yet in runtime dispatcher: ";
        *err_msg += name ? name : "unknown";
    }
    return false;
}

bool gc_arch_runtime_validate_hparams(const gc_hparams_t & hp, std::string * err_msg) {
    switch (hp.arch) {
        case GC_ARCH_LLAMA:
        case GC_ARCH_LLAMA_EMBED:
            return gc_arch_llama_validate_hparams(hp, err_msg);
        default:
            return gc__unsupported_arch(hp.arch, err_msg);
    }
}

bool gc_arch_runtime_build_attn_params(
        const gc_hparams_t & hp,
        uint32_t layer,
        uint32_t n_tokens,
        gc_attn_params_t & out,
        std::string * err_msg) {
    switch (hp.arch) {
        case GC_ARCH_LLAMA:
        case GC_ARCH_LLAMA_EMBED:
            return gc_arch_llama_build_attn_params(hp, layer, n_tokens, out, err_msg);
        default:
            return gc__unsupported_arch(hp.arch, err_msg);
    }
}

bool gc_arch_runtime_build_rope_params(
        const gc_hparams_t & hp,
        gc_rope_params_t & out,
        std::string * err_msg) {
    switch (hp.arch) {
        case GC_ARCH_LLAMA:
        case GC_ARCH_LLAMA_EMBED:
            return gc_arch_llama_build_rope_params(hp, out, err_msg);
        default:
            return gc__unsupported_arch(hp.arch, err_msg);
    }
}
