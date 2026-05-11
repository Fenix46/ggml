#include "gc_hparams.h"
#include "../loader/gc_gguf_loader.h"
#include "../loader/gc_common.h"

// ── Per-layer helpers ─────────────────────────────────────────────────────────

uint32_t gc_hparams_t::n_head_for_layer(uint32_t layer) const {
    if (layer < n_head_arr.size()) return n_head_arr[layer];
    return n_head;
}

uint32_t gc_hparams_t::n_head_kv_for_layer(uint32_t layer) const {
    if (layer < n_head_kv_arr.size()) return n_head_kv_arr[layer];
    return n_head_kv;
}

uint32_t gc_hparams_t::n_ff_for_layer(uint32_t layer) const {
    if (layer < n_ff_arr.size()) return n_ff_arr[layer];
    return n_ff;
}

// ── Loader ────────────────────────────────────────────────────────────────────

bool gc_hparams_load(gc_hparams_t & hp, const gc_model_loader_t & loader) {
    // Resolve arch
    hp.arch = gc_arch_from_string(loader.arch_name);
    if (hp.arch == GC_ARCH_UNKNOWN) {
        GC_LOG_WARN("gc_hparams_load: unknown arch '%s'\n", loader.arch_name.c_str());
    }

    auto kv = [&](gc_kv_key_t k) { return gc_kv_resolve(hp.arch, k); };

    // Core dimensions
    loader.get_u32(kv(GC_KV_VOCAB_SIZE),     hp.n_vocab,     false);
    loader.get_u32(kv(GC_KV_CONTEXT_LENGTH),  hp.n_ctx_train, true);
    loader.get_u32(kv(GC_KV_EMBEDDING_LENGTH),hp.n_embd,      true);
    loader.get_u32(kv(GC_KV_BLOCK_COUNT),     hp.n_layer,     true);
    loader.get_u32(kv(GC_KV_FEED_FORWARD_LENGTH), hp.n_ff,    false);
    if (hp.n_ff == 0) {
        loader.get_arr_u32(kv(GC_KV_FEED_FORWARD_LENGTH), hp.n_ff_arr, false);
        if (!hp.n_ff_arr.empty()) {
            hp.n_ff = hp.n_ff_arr.front();
        }
    }

    // If vocab_size not in KV, derive it from token_embd tensor shape later;
    // for now set a sentinel so callers can detect.
    // (The tokenizer phase will fill it properly.)

    // Attention scalar defaults
    loader.get_u32(kv(GC_KV_ATTN_HEAD_COUNT),    hp.n_head,       true);
    loader.get_u32(kv(GC_KV_ATTN_HEAD_COUNT_KV), hp.n_head_kv,    false);
    if (hp.n_head == 0) {
        loader.get_arr_u32(kv(GC_KV_ATTN_HEAD_COUNT), hp.n_head_arr, false);
        if (!hp.n_head_arr.empty()) {
            hp.n_head = hp.n_head_arr.front();
        }
    }
    if (hp.n_head_kv == 0) {
        loader.get_arr_u32(kv(GC_KV_ATTN_HEAD_COUNT_KV), hp.n_head_kv_arr, false);
        if (!hp.n_head_kv_arr.empty()) {
            hp.n_head_kv = hp.n_head_kv_arr.front();
        }
    }
    if (hp.n_head_kv == 0) hp.n_head_kv = hp.n_head;

    loader.get_u32(kv(GC_KV_ATTN_KEY_LENGTH),   hp.n_embd_head_k, false);
    loader.get_u32(kv(GC_KV_ATTN_VALUE_LENGTH),  hp.n_embd_head_v, false);

    // Norm eps
    loader.get_f32(kv(GC_KV_ATTN_LAYERNORM_EPS),     hp.f_norm_eps,     false);
    loader.get_f32(kv(GC_KV_ATTN_LAYERNORM_RMS_EPS),  hp.f_norm_rms_eps, false);

    // Softcapping / scaling
    loader.get_f32(kv(GC_KV_LOGIT_SCALE),               hp.f_logit_scale,             false);
    loader.get_f32(kv(GC_KV_RESIDUAL_SCALE),             hp.f_residual_scale,          false);
    loader.get_f32(kv(GC_KV_EMBEDDING_SCALE),            hp.f_embedding_scale,         false);
    loader.get_f32(kv(GC_KV_ATTN_LOGIT_SOFTCAPPING),    hp.f_attn_logit_softcapping,  false);
    loader.get_f32(kv(GC_KV_FINAL_LOGIT_SOFTCAPPING),   hp.f_final_logit_softcapping, false);
    loader.get_f32(kv(GC_KV_ATTN_SCALE),                 hp.f_attn_scale,              false);
    loader.get_f32(kv(GC_KV_ATTN_MAX_ALIBI_BIAS),        hp.f_max_alibi_bias,          false);
    loader.get_f32(kv(GC_KV_ATTN_CLAMP_KQV),             hp.f_clamp_kqv,               false);

    // Parallel residual
    loader.get_bool(kv(GC_KV_USE_PARALLEL_RESIDUAL), hp.use_par_res, false);

    // Causal attention flag
    loader.get_bool(kv(GC_KV_ATTN_CAUSAL), hp.causal_attn, false);

    // Sliding window
    loader.get_u32(kv(GC_KV_ATTN_SLIDING_WINDOW), hp.n_swa, false);

    // RoPE
    loader.get_u32(kv(GC_KV_ROPE_DIMENSION_COUNT), hp.n_rot,          false);
    loader.get_f32(kv(GC_KV_ROPE_FREQ_BASE),        hp.rope_freq_base, false);
    loader.get_f32(kv(GC_KV_ROPE_SCALE_LINEAR),     hp.rope_scale_linear, false);

    // MLA
    loader.get_u32(kv(GC_KV_ATTN_Q_LORA_RANK),      hp.n_lora_q,      false);
    loader.get_u32(kv(GC_KV_ATTN_KV_LORA_RANK),     hp.n_lora_kv,     false);
    loader.get_u32(kv(GC_KV_ATTN_KEY_LENGTH_MLA),   hp.n_embd_k_mla,  false);
    loader.get_u32(kv(GC_KV_ATTN_VALUE_LENGTH_MLA),  hp.n_embd_v_mla,  false);

    // MoE
    loader.get_u32(kv(GC_KV_EXPERT_COUNT),                    hp.n_expert,              false);
    loader.get_u32(kv(GC_KV_EXPERT_USED_COUNT),               hp.n_expert_used,         false);
    loader.get_u32(kv(GC_KV_EXPERT_SHARED_COUNT),             hp.n_expert_shared,       false);
    loader.get_u32(kv(GC_KV_EXPERT_FEED_FORWARD_LENGTH),      hp.n_ff_exp,              false);
    loader.get_u32(kv(GC_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH), hp.n_ff_shexp,          false);
    loader.get_u32(kv(GC_KV_MOE_EVERY_N_LAYERS),              hp.moe_every_n_layers,    false);
    loader.get_u32(kv(GC_KV_LEADING_DENSE_BLOCK_COUNT),       hp.leading_dense_block_count, false);

    // SSM / Mamba
    loader.get_u32(kv(GC_KV_SSM_CONV_KERNEL),      hp.ssm_d_conv,  false);
    loader.get_u32(kv(GC_KV_SSM_INNER_SIZE),        hp.ssm_d_inner, false);
    loader.get_u32(kv(GC_KV_SSM_STATE_SIZE),        hp.ssm_d_state, false);
    loader.get_u32(kv(GC_KV_SSM_TIME_STEP_RANK),    hp.ssm_dt_rank, false);
    loader.get_u32(kv(GC_KV_SSM_GROUP_COUNT),       hp.ssm_n_group, false);
    if (hp.ssm_n_group == 0) hp.ssm_n_group = 1;

    // LFM2 short-conv
    loader.get_u32(kv(GC_KV_SHORTCONV_L_CACHE), hp.shortconv_l_cache, false);

    // Derive n_rot default
    if (hp.n_rot == 0 && hp.n_head > 0) {
        hp.n_rot = hp.n_embd / hp.n_head;
    }

    return true;
}
