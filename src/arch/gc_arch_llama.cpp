#include "gc_arch_llama.h"

#include <cmath>

static bool gc__llama_fail(const char * msg, std::string * err_msg) {
    if (err_msg) {
        *err_msg = msg;
    }
    return false;
}

bool gc_arch_llama_validate_hparams(const gc_hparams_t & hp, std::string * err_msg) {
    if (hp.arch != GC_ARCH_LLAMA && hp.arch != GC_ARCH_LLAMA_EMBED) {
        return gc__llama_fail("gc_arch_llama requires GC_ARCH_LLAMA or GC_ARCH_LLAMA_EMBED", err_msg);
    }

    if (hp.n_embd == 0 || hp.n_layer == 0) {
        return gc__llama_fail("n_embd and n_layer must be non-zero", err_msg);
    }

    for (uint32_t il = 0; il < hp.n_layer; ++il) {
        const uint32_t n_head = hp.n_head_for_layer(il);
        if (n_head == 0) {
            return gc__llama_fail("attention.head_count must be non-zero", err_msg);
        }
        if (hp.n_embd % n_head != 0) {
            return gc__llama_fail("n_embd must be divisible by attention.head_count", err_msg);
        }

        const uint32_t n_head_kv = hp.n_head_kv_for_layer(il);
        if (n_head_kv == 0) {
            return gc__llama_fail("attention.head_count_kv must be non-zero", err_msg);
        }
        if (n_head % n_head_kv != 0) {
            return gc__llama_fail("attention.head_count must be divisible by attention.head_count_kv", err_msg);
        }
    }

    const uint32_t rope_dims = hp.n_rot ? hp.n_rot : hp.n_embd_k_head();
    if (rope_dims == 0) {
        return gc__llama_fail("rope dimensions resolved to zero", err_msg);
    }

    return true;
}

bool gc_arch_llama_build_attn_params(
        const gc_hparams_t & hp,
        uint32_t layer,
        uint32_t n_tokens,
        gc_attn_params_t & out,
        std::string * err_msg) {
    if (!gc_arch_llama_validate_hparams(hp, err_msg)) {
        return false;
    }

    if (layer >= hp.n_layer) {
        return gc__llama_fail("layer index out of range", err_msg);
    }

    const uint32_t n_head_q = hp.n_head_for_layer(layer);
    const uint32_t n_head_kv = hp.n_head_kv_for_layer(layer);

    const uint32_t n_embd_head_q = hp.n_embd_head_k ? hp.n_embd_head_k : hp.n_embd / n_head_q;
    const uint32_t n_embd_head_v = hp.n_embd_head_v ? hp.n_embd_head_v : n_embd_head_q;

    out = {};
    out.n_embd_head_q = n_embd_head_q;
    out.n_embd_head_v = n_embd_head_v;
    out.n_head_q = n_head_q;
    out.n_head_kv = n_head_kv;
    out.n_tokens = n_tokens;

    if (hp.f_attn_scale > 0.0f) {
        out.kq_scale = hp.f_attn_scale;
    } else {
        out.kq_scale = 1.0f / std::sqrt((float)n_embd_head_q);
    }

    out.use_flash = false;
    out.max_alibi_bias = hp.f_max_alibi_bias;
    out.logit_softcap = hp.f_attn_logit_softcapping;

    return true;
}

bool gc_arch_llama_build_rope_params(
        const gc_hparams_t & hp,
        gc_rope_params_t & out,
        std::string * err_msg) {
    if (!gc_arch_llama_validate_hparams(hp, err_msg)) {
        return false;
    }

    const uint32_t n_head_ref = hp.n_head_for_layer(0);
    const uint32_t default_rot = n_head_ref ? (hp.n_embd / n_head_ref) : 0;

    out = {};
    out.n_dims = (int) (hp.n_rot ? hp.n_rot : default_rot);
    out.mode = GGML_ROPE_TYPE_NEOX;
    out.n_ctx_orig = (int) hp.n_ctx_train;
    out.freq_base = hp.rope_freq_base;

    // GGUF stores a scaling factor where >1 extends context. ggml expects
    // freq_scale = 1/factor for linear scaling.
    const float factor = hp.rope_scale_linear > 0.0f ? hp.rope_scale_linear : 1.0f;
    out.freq_scale = 1.0f / factor;

    out.ext_factor = 0.0f;
    out.attn_factor = 1.0f;
    out.beta_fast = 32.0f;
    out.beta_slow = 1.0f;

    if (out.n_dims <= 0) {
        return gc__llama_fail("resolved rope dimensions are invalid", err_msg);
    }

    return true;
}
