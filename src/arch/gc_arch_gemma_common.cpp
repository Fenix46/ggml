#include "gc_arch_gemma_common.h"

#include <cmath>

static bool gc__gemma_fail(const char * msg, std::string * err_msg) {
    if (err_msg) {
        *err_msg = msg;
    }
    return false;
}

bool gc_arch_gemma_common_validate(
        const gc_hparams_t & hp,
        const gc_gemma_variant_cfg_t & cfg,
        std::string * err_msg) {
    if (hp.n_embd == 0 || hp.n_layer == 0 || hp.n_head == 0) {
        return gc__gemma_fail("gemma hparams invalid: n_embd/n_layer/n_head must be non-zero", err_msg);
    }

    if (hp.n_embd % hp.n_head != 0) {
        return gc__gemma_fail("gemma hparams invalid: n_embd must be divisible by n_head", err_msg);
    }

    for (uint32_t il = 0; il < hp.n_layer; ++il) {
        const uint32_t hq = hp.n_head_for_layer(il);
        const uint32_t hkv = hp.n_head_kv_for_layer(il);
        if (hq == 0 || hkv == 0) {
            return gc__gemma_fail("gemma hparams invalid: per-layer head counts must be non-zero", err_msg);
        }
        if (hq % hkv != 0) {
            return gc__gemma_fail("gemma hparams invalid: n_head must be divisible by n_head_kv", err_msg);
        }
    }

    if (cfg.causal_attn_required && !hp.causal_attn) {
        return gc__gemma_fail("gemma causal variant requires causal_attn=true", err_msg);
    }

    const uint32_t n_rot = hp.n_rot ? hp.n_rot : hp.n_embd_k_head();
    if (n_rot == 0) {
        return gc__gemma_fail("gemma hparams invalid: resolved n_rot is zero", err_msg);
    }

    return true;
}

bool gc_arch_gemma_common_build_attn(
        const gc_hparams_t & hp,
        const gc_gemma_variant_cfg_t & cfg,
        uint32_t layer,
        uint32_t n_tokens,
        gc_attn_params_t & out,
        std::string * err_msg) {
    if (!gc_arch_gemma_common_validate(hp, cfg, err_msg)) {
        return false;
    }
    if (layer >= hp.n_layer) {
        return gc__gemma_fail("layer index out of range", err_msg);
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

    if (cfg.explicit_attn_scale > 0.0f) {
        out.kq_scale = cfg.explicit_attn_scale;
    } else if (cfg.use_llama27b_rule && (hp.n_layer == 46 || hp.n_layer == 62)) {
        out.kq_scale = 1.0f / std::sqrt((float)(hp.n_embd / n_head_q));
    } else if (hp.f_attn_scale > 0.0f) {
        out.kq_scale = hp.f_attn_scale;
    } else {
        out.kq_scale = 1.0f / std::sqrt((float)n_embd_head_q);
    }

    out.use_flash = false;
    out.max_alibi_bias = hp.f_max_alibi_bias;
    out.logit_softcap = hp.f_attn_logit_softcapping;

    return true;
}

bool gc_arch_gemma_common_build_rope(
        const gc_hparams_t & hp,
        gc_rope_params_t & out,
        std::string * err_msg) {
    if (hp.n_embd == 0 || hp.n_head == 0) {
        return gc__gemma_fail("invalid gemma hparams for rope", err_msg);
    }

    const uint32_t default_rot = hp.n_embd / hp.n_head;
    out = {};
    out.n_dims = (int) (hp.n_rot ? hp.n_rot : default_rot);
    out.mode = GGML_ROPE_TYPE_NEOX;
    out.n_ctx_orig = (int) hp.n_ctx_train;
    out.freq_base = hp.rope_freq_base;
    const float factor = hp.rope_scale_linear > 0.0f ? hp.rope_scale_linear : 1.0f;
    out.freq_scale = 1.0f / factor;
    out.ext_factor = 0.0f;
    out.attn_factor = 1.0f;
    out.beta_fast = 32.0f;
    out.beta_slow = 1.0f;

    if (out.n_dims <= 0) {
        return gc__gemma_fail("resolved rope dimensions are invalid", err_msg);
    }
    return true;
}
