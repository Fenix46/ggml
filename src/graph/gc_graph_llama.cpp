#include "gc_graph_llama.h"
#include "gc_graph_runner.h"    // gc_kv_pool_t
#include "gc_attention.h"       // gc_build_norm, gc_build_ffn, gc_build_attn_mha
#include "gc_arch_runtime.h"    // gc_arch_runtime_build_attn_params, etc.
#include "gc_rope.h"            // gc_rope_apply

#include <string>

ggml_tensor * GcGraphLlama::build(const GcGraphBuildParams & p) {
    const int n_new = (int)p.entry->token_ids.size();
    const int n_ctx = p.entry->num_computed + n_new;

    // ── Token embedding ───────────────────────────────────────────────────────
    ggml_tensor * w_embd = p.get_weight(p.ctx, GC_TENSOR_TOKEN_EMBD, "weight", -1);
    if (!w_embd) return nullptr;

    ggml_tensor * tok_view = ggml_view_1d(p.ctx, p.inp_tokens, (int64_t)n_new, 0);
    ggml_tensor * pos_view = ggml_view_1d(p.ctx, p.inp_pos,    (int64_t)n_new, 0);
    ggml_tensor * cur = ggml_get_rows(p.ctx, w_embd, tok_view);

    // ── Causal mask ───────────────────────────────────────────────────────────
    ggml_tensor * kq_mask = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32,
                                                (int64_t)n_ctx, (int64_t)n_new);
    ggml_set_name(kq_mask, "kq_mask");
    ggml_set_input(kq_mask);
    if (p.kq_mask_out) *p.kq_mask_out = kq_mask;

    const gc_norm_t norm_type = GC_NORM_RMS;
    const float     norm_eps  = p.hp->f_norm_rms_eps > 0 ? p.hp->f_norm_rms_eps : 1e-5f;

    // ── Transformer layers ────────────────────────────────────────────────────
    for (uint32_t il = 0; il < p.hp->n_layer; ++il) {
        std::string lerr;
        gc_attn_params_t ap{};
        gc_layer_flags_t lf{};
        if (!gc_arch_runtime_build_attn_params(*p.hp, il, (uint32_t)n_new, ap, &lerr) ||
            !gc_arch_runtime_build_layer_flags(*p.hp, il, lf, &lerr)) {
            return nullptr;
        }

        const int64_t n_embd_hq = ap.n_embd_head_q;
        const int64_t n_embd_hv = ap.n_embd_head_v;
        const int64_t nq        = ap.n_head_q;
        const int64_t nkv       = ap.n_head_kv;

        // Pre-attention norm
        ggml_tensor * w_an = p.get_weight(p.ctx, GC_TENSOR_ATTN_NORM, "weight", (int)il);
        if (!w_an) return nullptr;

        ggml_tensor * normed_attn = gc_build_norm(p.ctx, cur, w_an, nullptr,
                                                   norm_type, norm_eps,
                                                   [](ggml_tensor *, const char *, int) {}, (int)il);

        // Q/K/V projections
        ggml_tensor * w_q = p.get_weight(p.ctx, GC_TENSOR_ATTN_Q, "weight", (int)il);
        ggml_tensor * w_k = p.get_weight(p.ctx, GC_TENSOR_ATTN_K, "weight", (int)il);
        ggml_tensor * w_v = p.get_weight(p.ctx, GC_TENSOR_ATTN_V, "weight", (int)il);
        if (!w_q || !w_k || !w_v) return nullptr;

        ggml_tensor * Qcur = ggml_reshape_3d(p.ctx,
            ggml_mul_mat(p.ctx, w_q, normed_attn), n_embd_hq, nq, (int64_t)n_new);
        ggml_tensor * Kcur = ggml_reshape_3d(p.ctx,
            ggml_mul_mat(p.ctx, w_k, normed_attn), n_embd_hq, nkv, (int64_t)n_new);
        ggml_tensor * Vcur = ggml_reshape_3d(p.ctx,
            ggml_mul_mat(p.ctx, w_v, normed_attn), n_embd_hv, nkv, (int64_t)n_new);

        // RoPE parameters from arch runtime (no layer-specific override)
        gc_rope_params_t rp{};
        if (!gc_arch_runtime_build_rope_params(*p.hp, rp, &lerr))
            return nullptr;

        if (lf.apply_rope && rp.n_dims > 0) {
            Qcur = gc_rope_apply(p.ctx, Qcur, pos_view, nullptr, rp);
            Kcur = gc_rope_apply(p.ctx, Kcur, pos_view, nullptr, rp);
        }

        // KV write
        ggml_tensor * Kcur_2d = ggml_reshape_2d(p.ctx, Kcur,
            (int64_t)(nkv * n_embd_hq), (int64_t)n_new);
        ggml_tensor * Vcur_2d = ggml_reshape_2d(p.ctx, Vcur,
            (int64_t)(nkv * n_embd_hv), (int64_t)n_new);

        for (int i = 0; i < n_new; ++i) {
            const int token_pos = p.entry->num_computed + i;
            const size_t slot_idx = p.kv_pool->token_slot_idx(p.entry->block_ids, token_pos);

            const size_t byte_off_k = slot_idx * (size_t)(nkv * n_embd_hq) * sizeof(float);
            const size_t byte_off_v = slot_idx * (size_t)(nkv * n_embd_hv) * sizeof(float);

            ggml_tensor * k_src = ggml_view_2d(p.ctx, Kcur_2d,
                (int64_t)(nkv * n_embd_hq), 1, Kcur_2d->nb[1], (size_t)i * Kcur_2d->nb[1]);
            ggml_tensor * k_dst = ggml_view_2d(p.ctx, p.kv_pool->k[il],
                (int64_t)(nkv * n_embd_hq), 1, p.kv_pool->k[il]->nb[1], byte_off_k);
            ggml_build_forward_expand(p.gf, ggml_cpy(p.ctx, k_src, k_dst));

            ggml_tensor * v_src = ggml_view_2d(p.ctx, Vcur_2d,
                (int64_t)(nkv * n_embd_hv), 1, Vcur_2d->nb[1], (size_t)i * Vcur_2d->nb[1]);
            ggml_tensor * v_dst = ggml_view_2d(p.ctx, p.kv_pool->v[il],
                (int64_t)(nkv * n_embd_hv), 1, p.kv_pool->v[il]->nb[1], byte_off_v);
            ggml_build_forward_expand(p.gf, ggml_cpy(p.ctx, v_src, v_dst));
        }

        // KV read + attention
        ggml_tensor * Q3 = Qcur;
        ggml_tensor * K3 = nullptr;
        ggml_tensor * V3 = nullptr;

        if (p.entry->num_computed == 0 && n_ctx == n_new) {
            K3 = Kcur;
            V3 = Vcur;
        } else {
            ggml_tensor * K_full = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32,
                (int64_t)(nkv * n_embd_hq), (int64_t)n_ctx);
            ggml_set_name(K_full, "K_gathered");
            ggml_tensor * V_full = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32,
                (int64_t)(nkv * n_embd_hv), (int64_t)n_ctx);
            ggml_set_name(V_full, "V_gathered");

            for (int t = 0; t < n_ctx; ++t) {
                const size_t dst_off_k = (size_t)t * (size_t)(nkv * n_embd_hq) * sizeof(float);
                const size_t dst_off_v = (size_t)t * (size_t)(nkv * n_embd_hv) * sizeof(float);

                ggml_tensor * ks = nullptr;
                ggml_tensor * vs = nullptr;
                if (t < p.entry->num_computed) {
                    const size_t slot_idx = p.kv_pool->token_slot_idx(p.entry->block_ids, t);
                    const size_t src_off_k = slot_idx * (size_t)(nkv * n_embd_hq) * sizeof(float);
                    const size_t src_off_v = slot_idx * (size_t)(nkv * n_embd_hv) * sizeof(float);
                    ks = ggml_view_2d(p.ctx, p.kv_pool->k[il],
                        (int64_t)(nkv * n_embd_hq), 1, p.kv_pool->k[il]->nb[1], src_off_k);
                    vs = ggml_view_2d(p.ctx, p.kv_pool->v[il],
                        (int64_t)(nkv * n_embd_hv), 1, p.kv_pool->v[il]->nb[1], src_off_v);
                } else {
                    const int cur_idx = t - p.entry->num_computed;
                    ks = ggml_view_2d(p.ctx, Kcur_2d,
                        (int64_t)(nkv * n_embd_hq), 1, Kcur_2d->nb[1],
                        (size_t)cur_idx * Kcur_2d->nb[1]);
                    vs = ggml_view_2d(p.ctx, Vcur_2d,
                        (int64_t)(nkv * n_embd_hv), 1, Vcur_2d->nb[1],
                        (size_t)cur_idx * Vcur_2d->nb[1]);
                }

                ggml_tensor * kd = ggml_view_2d(p.ctx, K_full,
                    (int64_t)(nkv * n_embd_hq), 1, K_full->nb[1], dst_off_k);
                ggml_build_forward_expand(p.gf, ggml_cpy(p.ctx, ks, kd));

                ggml_tensor * vd = ggml_view_2d(p.ctx, V_full,
                    (int64_t)(nkv * n_embd_hv), 1, V_full->nb[1], dst_off_v);
                ggml_build_forward_expand(p.gf, ggml_cpy(p.ctx, vs, vd));
            }

            K3 = ggml_reshape_3d(p.ctx, K_full, n_embd_hq, nkv, (int64_t)n_ctx);
            V3 = ggml_reshape_3d(p.ctx, V_full, n_embd_hv, nkv, (int64_t)n_ctx);
        }

        // MHA — n_tokens is the number of QUERY tokens (n_new), not total context
        gc_attn_params_t ap_mha = ap;
        ap_mha.n_tokens = (int64_t)n_new;
        auto cb = [](ggml_tensor *, const char *, int) {};
        ggml_tensor * attn_out = gc_build_attn_mha(p.ctx, p.gf,
            Q3, K3, V3, kq_mask, nullptr, ap_mha, cb, (int)il);

        // Output projection
        ggml_tensor * w_o = p.get_weight(p.ctx, GC_TENSOR_ATTN_OUT, "weight", (int)il);
        if (!w_o) return nullptr;
        attn_out = ggml_mul_mat(p.ctx, w_o, attn_out);

        if (lf.has_post_attn_norm) {
            ggml_tensor * w_pan = p.get_weight(p.ctx, GC_TENSOR_POST_ATTN_NORM, "weight", (int)il);
            if (w_pan)
                attn_out = gc_build_norm(p.ctx, attn_out, w_pan, nullptr,
                                         norm_type, norm_eps, cb, (int)il);
        }
        cur = ggml_add(p.ctx, cur, attn_out);

        // FFN
        ggml_tensor * w_fn = p.get_weight(p.ctx, GC_TENSOR_FFN_NORM, "weight", (int)il);
        ggml_tensor * w_up = p.get_weight(p.ctx, GC_TENSOR_FFN_UP,   "weight", (int)il);
        ggml_tensor * w_dn = p.get_weight(p.ctx, GC_TENSOR_FFN_DOWN, "weight", (int)il);
        ggml_tensor * w_gt = p.get_weight(p.ctx, GC_TENSOR_FFN_GATE, "weight", (int)il);
        if (!w_fn || !w_up || !w_dn) return nullptr;

        ggml_tensor * normed_ffn = gc_build_norm(p.ctx, cur, w_fn, nullptr,
                                                  norm_type, norm_eps, cb, (int)il);
        ggml_tensor * ffn_out = gc_build_ffn(p.ctx, normed_ffn,
            w_up, nullptr, w_gt, nullptr, w_dn, nullptr,
            GC_FFN_SILU, GC_FFN_SEQ, cb, (int)il);

        if (lf.has_post_ffn_norm) {
            ggml_tensor * w_pfn = p.get_weight(p.ctx, GC_TENSOR_POST_MLP_NORM, "weight", (int)il);
            if (w_pfn)
                ffn_out = gc_build_norm(p.ctx, ffn_out, w_pfn, nullptr,
                                        norm_type, norm_eps, cb, (int)il);
        }
        cur = ggml_add(p.ctx, cur, ffn_out);
    }

    // Output norm + lm_head
    ggml_tensor * w_on = p.get_weight(p.ctx, GC_TENSOR_OUTPUT_NORM, "weight", -1);
    if (w_on)
        cur = gc_build_norm(p.ctx, cur, w_on, nullptr, norm_type, norm_eps,
                            [](ggml_tensor *, const char *, int) {}, -1);

    ggml_tensor * w_out = p.get_weight(p.ctx, GC_TENSOR_OUTPUT, "weight", -1);
    if (!w_out) return nullptr;

    ggml_tensor * logits = ggml_mul_mat(p.ctx, w_out, cur);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(p.gf, logits);

    return logits;
}
