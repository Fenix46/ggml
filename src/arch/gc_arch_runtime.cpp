#include "gc_arch_runtime.h"

#include "gc_arch.h"
#include "gc_arch_gemma.h"
#include "gc_arch_gemma2.h"
#include "gc_arch_gemma3.h"
#include "gc_arch_gemma3n.h"
#include "gc_arch_gemma4.h"
#include "gc_arch_gemma_embedding.h"
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
        case GC_ARCH_GEMMA:
            return gc_arch_gemma_validate_hparams(hp, err_msg);
        case GC_ARCH_GEMMA2:
            return gc_arch_gemma2_validate_hparams(hp, err_msg);
        case GC_ARCH_GEMMA3:
            return gc_arch_gemma3_validate_hparams(hp, err_msg);
        case GC_ARCH_GEMMA3N:
            return gc_arch_gemma3n_validate_hparams(hp, err_msg);
        case GC_ARCH_GEMMA4:
            return gc_arch_gemma4_validate_hparams(hp, err_msg);
        case GC_ARCH_GEMMA_EMBEDDING:
            return gc_arch_gemma_embedding_validate_hparams(hp, err_msg);
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
        case GC_ARCH_GEMMA:
            return gc_arch_gemma_build_attn_params(hp, layer, n_tokens, out, err_msg);
        case GC_ARCH_GEMMA2:
            return gc_arch_gemma2_build_attn_params(hp, layer, n_tokens, out, err_msg);
        case GC_ARCH_GEMMA3:
            return gc_arch_gemma3_build_attn_params(hp, layer, n_tokens, out, err_msg);
        case GC_ARCH_GEMMA3N:
            return gc_arch_gemma3n_build_attn_params(hp, layer, n_tokens, out, err_msg);
        case GC_ARCH_GEMMA4:
            return gc_arch_gemma4_build_attn_params(hp, layer, n_tokens, out, err_msg);
        case GC_ARCH_GEMMA_EMBEDDING:
            return gc_arch_gemma_embedding_build_attn_params(hp, layer, n_tokens, out, err_msg);
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
        case GC_ARCH_GEMMA:
            return gc_arch_gemma_build_rope_params(hp, out, err_msg);
        case GC_ARCH_GEMMA2:
            return gc_arch_gemma2_build_rope_params(hp, out, err_msg);
        case GC_ARCH_GEMMA3:
            return gc_arch_gemma3_build_rope_params(hp, out, err_msg);
        case GC_ARCH_GEMMA3N:
            return gc_arch_gemma3n_build_rope_params(hp, out, err_msg);
        case GC_ARCH_GEMMA4:
            return gc_arch_gemma4_build_rope_params(hp, out, err_msg);
        case GC_ARCH_GEMMA_EMBEDDING:
            return gc_arch_gemma_embedding_build_rope_params(hp, out, err_msg);
        default:
            return gc__unsupported_arch(hp.arch, err_msg);
    }
}

// ── Norm parameters ───────────────────────────────────────────────────────────

bool gc_arch_runtime_build_norm_params(
        const gc_hparams_t & hp,
        gc_norm_params_t   & out,
        std::string        * err_msg) {
    (void)err_msg;
    out = {};
    switch (hp.arch) {
        // GPT-2 style: LayerNorm
        case GC_ARCH_GPT2:
        case GC_ARCH_GPTJ:
        case GC_ARCH_GPTNEOX:
        case GC_ARCH_MPT:
        case GC_ARCH_BLOOM:
            out.type = GC_NORM_LAYER;
            out.eps  = hp.f_norm_eps > 0.0f ? hp.f_norm_eps : 1e-5f;
            break;
        // All RMSNorm architectures (default)
        default:
            out.type = GC_NORM_RMS;
            out.eps  = hp.f_norm_rms_eps > 0.0f ? hp.f_norm_rms_eps : 1e-5f;
            break;
    }
    return true;
}

// ── FFN parameters ────────────────────────────────────────────────────────────

bool gc_arch_runtime_build_ffn_params(
        const gc_hparams_t & hp,
        gc_ffn_params_t    & out,
        std::string        * err_msg) {
    (void)err_msg;
    out = {};
    switch (hp.arch) {
        // GELU, no gate (GPT-2 family)
        case GC_ARCH_GPT2:
        case GC_ARCH_GPTJ:
        case GC_ARCH_STARCODER:
        case GC_ARCH_STARCODER2:
        case GC_ARCH_BLOOM:
            out.act       = GC_FFN_GELU;
            out.has_gate  = false;
            out.gate_mode = GC_FFN_SEQ;
            break;
        // GELU with gate (Gemma family)
        case GC_ARCH_GEMMA:
        case GC_ARCH_GEMMA2:
        case GC_ARCH_GEMMA3:
        case GC_ARCH_GEMMA3N:
        case GC_ARCH_GEMMA4:
        case GC_ARCH_GEMMA_EMBEDDING:
            out.act       = GC_FFN_GEGLU;
            out.has_gate  = true;
            out.gate_mode = GC_FFN_PAR;
            break;
        // SwiGLU (everything else: Llama, Qwen, Mistral, Falcon, etc.)
        default:
            out.act       = GC_FFN_SWIGLU;
            out.has_gate  = true;
            out.gate_mode = GC_FFN_PAR;
            break;
    }
    return true;
}

// ── Per-layer structural flags ────────────────────────────────────────────────

bool gc_arch_runtime_build_layer_flags(
        const gc_hparams_t & hp,
        uint32_t             layer,
        gc_layer_flags_t   & out,
        std::string        * err_msg) {
    (void)layer; (void)err_msg;
    out = {};
    out.causal    = hp.causal_attn;
    out.apply_rope = true;

    switch (hp.arch) {
        case GC_ARCH_GEMMA2:
        case GC_ARCH_GEMMA3:
        case GC_ARCH_GEMMA3N:
        case GC_ARCH_GEMMA4:
            out.has_post_attn_norm = true;
            out.has_post_ffn_norm  = true;
            // Gemma3+ adds per-head Q/K norms
            out.has_attn_q_norm = (hp.arch == GC_ARCH_GEMMA3  ||
                                   hp.arch == GC_ARCH_GEMMA3N ||
                                   hp.arch == GC_ARCH_GEMMA4);
            out.has_attn_k_norm = out.has_attn_q_norm;
            break;
        // GPT-NeoX / Falcon: no RoPE on some variants — conservative: keep apply_rope=true,
        // rope dispatcher will return n_dims=0 if not applicable.
        default:
            break;
    }
    return true;
}
