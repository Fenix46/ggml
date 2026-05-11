#include "gc_attention.h"

#include "ggml.h"

#include <cassert>
#include <cmath>

// ── Norm ──────────────────────────────────────────────────────────────────────

struct ggml_tensor * gc_build_norm(
        struct ggml_context * ctx,
        struct ggml_tensor  * cur,
        struct ggml_tensor  * w,
        struct ggml_tensor  * b,
        gc_norm_t             type,
        float                 eps,
        const gc_graph_cb_t & cb,
        int il) {
    switch (type) {
        case GC_NORM_LAYER: cur = ggml_norm    (ctx, cur, eps); break;
        case GC_NORM_RMS:   cur = ggml_rms_norm(ctx, cur, eps); break;
        case GC_NORM_GROUP: cur = ggml_group_norm(ctx, cur, /*n_groups=*/32, eps); break;
    }
    cb(cur, "norm", il);

    if (w) {
        cur = ggml_mul(ctx, cur, w);
        cb(cur, "norm_w", il);
    }
    if (b) {
        cur = ggml_add(ctx, cur, b);
        cb(cur, "norm_b", il);
    }
    return cur;
}

// ── FFN ───────────────────────────────────────────────────────────────────────

struct ggml_tensor * gc_build_ffn(
        struct ggml_context * ctx,
        struct ggml_tensor  * cur,
        struct ggml_tensor  * up,   struct ggml_tensor * up_b,
        struct ggml_tensor  * gate, struct ggml_tensor * gate_b,
        struct ggml_tensor  * down, struct ggml_tensor * down_b,
        gc_ffn_op_t           type_op,
        gc_ffn_gate_t         type_gate,
        const gc_graph_cb_t & cb,
        int il) {
    const bool use_gate = (gate != nullptr);

    struct ggml_tensor * up_cur  = ggml_mul_mat(ctx, up, cur);
    if (up_b)   up_cur = ggml_add(ctx, up_cur, up_b);
    cb(up_cur, "ffn_up", il);

    struct ggml_tensor * gate_cur = nullptr;
    if (use_gate && type_gate == GC_FFN_PAR) {
        gate_cur = ggml_mul_mat(ctx, gate, cur);
        if (gate_b) gate_cur = ggml_add(ctx, gate_cur, gate_b);
        cb(gate_cur, "ffn_gate", il);
    }

    // activation
    struct ggml_tensor * act = nullptr;
    switch (type_op) {
        case GC_FFN_SILU:
            act = use_gate ? ggml_silu(ctx, gate_cur ? gate_cur : up_cur) : ggml_silu(ctx, up_cur);
            break;
        case GC_FFN_GELU:
            act = use_gate ? ggml_gelu(ctx, gate_cur ? gate_cur : up_cur) : ggml_gelu(ctx, up_cur);
            break;
        case GC_FFN_RELU:
            act = ggml_relu(ctx, up_cur);
            break;
        case GC_FFN_RELU_SQR:
            act = ggml_sqr(ctx, ggml_relu(ctx, up_cur));
            break;
        case GC_FFN_SWIGLU:
            act = use_gate ? ggml_mul(ctx, ggml_silu(ctx, gate_cur ? gate_cur : up_cur), up_cur) : ggml_silu(ctx, up_cur);
            break;
        case GC_FFN_GEGLU:
            act = use_gate ? ggml_mul(ctx, ggml_gelu(ctx, gate_cur ? gate_cur : up_cur), up_cur) : ggml_gelu(ctx, up_cur);
            break;
        case GC_FFN_REGLU:
            act = use_gate ? ggml_mul(ctx, ggml_relu(ctx, gate_cur ? gate_cur : up_cur), up_cur) : ggml_relu(ctx, up_cur);
            break;
    }
    cb(act, "ffn_act", il);

    struct ggml_tensor * res = ggml_mul_mat(ctx, down, act);
    if (down_b) res = ggml_add(ctx, res, down_b);
    cb(res, "ffn_out", il);

    return res;
}

// ── MHA ───────────────────────────────────────────────────────────────────────

struct ggml_tensor * gc_build_attn_mha(
        struct ggml_context      * ctx,
        struct ggml_cgraph       * gf,
        struct ggml_tensor       * q,
        struct ggml_tensor       * k,
        struct ggml_tensor       * v,
        struct ggml_tensor       * kq_mask,
        struct ggml_tensor       * kq_b,
        const gc_attn_params_t   & p,
        const gc_graph_cb_t      & cb,
        int il) {
    // ensure Q/K/V are materialised before attention (avoids bad graph ordering)
    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, k);
    ggml_build_forward_expand(gf, v);

    // permute: [n_embd_head, n_head, n_tokens] → [n_embd_head, n_tokens, n_head, 1]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    struct ggml_tensor * cur;

    if (p.use_flash) {
        // Flash attention path
        // Flash attention expects k/v in F16
        if (k->type == GGML_TYPE_F32) k = ggml_cast(ctx, k, GGML_TYPE_F16);
        if (v->type == GGML_TYPE_F32) v = ggml_cast(ctx, v, GGML_TYPE_F16);

        cur = ggml_flash_attn_ext(ctx, q, k, v, kq_mask, p.kq_scale,
                                  p.max_alibi_bias, p.logit_softcap);
        cb(cur, "attn_flash", il);

        // flash_attn_ext output: [n_embd_head_v, n_tokens, n_head, 1]
        cur = ggml_reshape_2d(ctx, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
    } else {
        // Standard attention path
        // kq = Q * K^T → [n_kv_tokens, n_q_tokens, n_head, 1]
        struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
        cb(kq, "attn_kq", il);

        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        if (p.logit_softcap != 0.0f) {
            kq = ggml_scale(ctx, kq, 1.0f / p.logit_softcap);
            kq = ggml_tanh (ctx, kq);
            kq = ggml_scale(ctx, kq, p.logit_softcap);
            cb(kq, "attn_kq_softcapped", il);
        }

        if (kq_b) {
            kq = ggml_add(ctx, kq, kq_b);
            cb(kq, "attn_kq_biased", il);
        }

        kq = ggml_soft_max_ext(ctx, kq, kq_mask, p.kq_scale, p.max_alibi_bias);
        cb(kq, "attn_kq_softmax", il);

        // V after permute: [n_embd_h, n_kv_tokens, n_head_kv, 1]
        // For kqv = V^T * kq, need V as [n_kv_tokens, n_embd_h, n_head_kv, 1]
        // kq: [n_kv_tokens, n_q_tokens, n_head_q, 1]
        // mul_mat(V_t, kq): ne[0] of V_t = n_kv_tokens = ne[0] of kq ✓
        struct ggml_tensor * v_t = ggml_cont(ctx, ggml_transpose(ctx, v));
        cb(v_t, "attn_v_t", il);

        // kqv = softmax(kq) * V → [n_embd_head_v, n_q_tokens, n_head, 1]
        struct ggml_tensor * kqv = ggml_mul_mat(ctx, v_t, kq);
        cb(kqv, "attn_kqv", il);

        // permute back: [n_embd_head_v, n_head, n_q_tokens, 1]
        cur = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
    }

    cb(cur, "attn_out_mha", il);
    ggml_build_forward_expand(gf, cur);
    return cur;
}

// ── Full attention sublayer ───────────────────────────────────────────────────

struct ggml_tensor * gc_build_attn(
        struct ggml_context      * ctx,
        struct ggml_cgraph       * gf,
        struct ggml_tensor       * cur,
        struct ggml_tensor       * wq,  struct ggml_tensor * bq,
        struct ggml_tensor       * wk,  struct ggml_tensor * bk,
        struct ggml_tensor       * wv,  struct ggml_tensor * bv,
        struct ggml_tensor       * wo,  struct ggml_tensor * bo,
        struct ggml_tensor       * pos,
        struct ggml_tensor       * kq_mask,
        const gc_attn_params_t   & ap,
        const gc_rope_params_t   & rp,
        const gc_graph_cb_t      & cb,
        int il) {
    const int64_t n_tokens    = ap.n_tokens;
    const int64_t n_head_q    = ap.n_head_q;
    const int64_t n_head_kv   = ap.n_head_kv;
    const int64_t n_embd_head = ap.n_embd_head_q;

    // Q projection
    struct ggml_tensor * Qcur = ggml_mul_mat(ctx, wq, cur);
    if (bq) Qcur = ggml_add(ctx, Qcur, bq);
    Qcur = ggml_reshape_3d(ctx, Qcur, n_embd_head, n_head_q,  n_tokens);
    cb(Qcur, "attn_q", il);

    // K projection
    struct ggml_tensor * Kcur = ggml_mul_mat(ctx, wk, cur);
    if (bk) Kcur = ggml_add(ctx, Kcur, bk);
    Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_tokens);
    cb(Kcur, "attn_k", il);

    // V projection
    struct ggml_tensor * Vcur = ggml_mul_mat(ctx, wv, cur);
    if (bv) Vcur = ggml_add(ctx, Vcur, bv);
    Vcur = ggml_reshape_3d(ctx, Vcur, ap.n_embd_head_v, n_head_kv, n_tokens);
    cb(Vcur, "attn_v", il);

    // RoPE on Q and K
    if (pos != nullptr && rp.n_dims > 0) {
        Qcur = gc_rope_apply(ctx, Qcur, pos, nullptr, rp);
        cb(Qcur, "attn_q_rope", il);
        Kcur = gc_rope_apply(ctx, Kcur, pos, nullptr, rp);
        cb(Kcur, "attn_k_rope", il);
    }

    // MHA
    struct ggml_tensor * out = gc_build_attn_mha(ctx, gf, Qcur, Kcur, Vcur, kq_mask, nullptr, ap, cb, il);

    // Output projection
    if (wo) {
        out = ggml_mul_mat(ctx, wo, out);
        if (bo) out = ggml_add(ctx, out, bo);
        cb(out, "attn_out_proj", il);
    }

    return out;
}
