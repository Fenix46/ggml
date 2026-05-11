#include "gc_graph_runner.h"

#include "gc_arch.h"
#include "gc_attention.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

// ── gc_kv_buf_t ───────────────────────────────────────────────────────────────

bool gc_kv_buf_t::init(uint32_t n_layer_, uint32_t num_blocks_, uint32_t block_size_,
                        uint32_t n_kv_head_, uint32_t head_dim_) {
    n_layer    = n_layer_;
    num_blocks = num_blocks_;
    block_size = block_size_;
    n_kv_head  = n_kv_head_;
    head_dim   = head_dim_;

    slot_stride     = (size_t)n_kv_head * head_dim;
    block_stride    = (size_t)block_size * slot_stride;
    layer_kv_stride = (size_t)num_blocks * block_stride;

    const size_t total = (size_t)n_layer * 2 * layer_kv_stride;
    if (total == 0) return false;
    data.assign(total, 0.0f);
    return true;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

static ggml_tensor * gc__weight_view(ggml_context * ctx,
                                      const gc_model_loader_t * loader,
                                      const gc_tensor_weight_t * w) {
    if (!ctx || !loader || !w || !w->tensor) return nullptr;
    const int64_t ne0 = w->tensor->ne[0];
    const int64_t ne1 = w->tensor->ne[1];
    const bool prev = ggml_get_no_alloc(ctx);
    ggml_set_no_alloc(ctx, true);
    ggml_tensor * t = ggml_new_tensor_2d(ctx, w->tensor->type, ne0, ne1);
    ggml_set_no_alloc(ctx, prev);
    if (!t) return nullptr;
    const auto & mm = loader->mmaps[(size_t)w->shard_idx];
    if (!mm) return nullptr;
    t->data = static_cast<uint8_t *>(mm->addr()) + w->file_offs;
    return t;
}

// Build a 1D F32 view into the KV buffer for a contiguous run of tokens.
// Returns a GGML tensor with shape [n_kv_head * head_dim, n_tokens].
// token_slots[i] = (block_id, slot_in_block) for token i.
static ggml_tensor * gc__kv_view(ggml_context * ctx,
                                   gc_kv_buf_t & kv,
                                   uint32_t layer, int kv_idx,
                                   const std::vector<std::pair<int32_t,uint32_t>> & slots,
                                   uint32_t n_tokens) {
    // Allocate a fresh F32 tensor and copy KV data into it (gather).
    // For performance a scatter/gather kernel would be ideal, but correctness
    // first: copy from the paged buffer into a contiguous tensor.
    const int64_t row = (int64_t)(kv.n_kv_head * kv.head_dim);
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, row, (int64_t)n_tokens);
    if (!t) return nullptr;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const float * src = kv.slot(layer, kv_idx, slots[i].first, slots[i].second);
        float * dst = static_cast<float *>(t->data) + (size_t)i * row;
        std::memcpy(dst, src, sizeof(float) * (size_t)row);
    }
    return t;
}

// Write new K or V values back to the KV buffer for a set of slots.
static void gc__kv_write(gc_kv_buf_t & kv,
                          uint32_t layer, int kv_idx,
                          const std::vector<std::pair<int32_t,uint32_t>> & slots,
                          const float * src_data,
                          uint32_t n_tokens) {
    const size_t row = kv.n_kv_head * kv.head_dim;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        float * dst = kv.slot(layer, kv_idx, slots[i].first, slots[i].second);
        std::memcpy(dst, src_data + (size_t)i * row, sizeof(float) * row);
    }
}

// Convert flat block_ids + num_computed + n_new to (block_id, slot) pairs
// for the range [num_computed, num_computed + n_new).
static std::vector<std::pair<int32_t,uint32_t>>
gc__make_slots(const std::vector<int32_t> & block_ids,
               uint32_t block_size,
               int num_computed, int n_new) {
    std::vector<std::pair<int32_t,uint32_t>> slots;
    slots.reserve((size_t)n_new);
    for (int i = 0; i < n_new; ++i) {
        const int pos        = num_computed + i;
        const int blk_idx    = pos / (int)block_size;
        const uint32_t slot  = (uint32_t)(pos % (int)block_size);
        const int32_t blk_id = (blk_idx < (int)block_ids.size())
                                ? block_ids[(size_t)blk_idx] : 0;
        slots.push_back({blk_id, slot});
    }
    return slots;
}

// ── gc_graph_runner_t::init() ─────────────────────────────────────────────────

bool gc_graph_runner_t::init() {
    if (!loader_) { err_ = "null loader"; return false; }

    std::string err;
    if (!gc_arch_runtime_validate_hparams(hp_, &err)) {
        err_ = "hparams validation failed: " + err;
        return false;
    }
    if (!gc_arch_runtime_build_norm_params(hp_, norm_params_, &err)) {
        err_ = "norm params: " + err; return false;
    }
    if (!gc_arch_runtime_build_ffn_params(hp_, ffn_params_, &err)) {
        err_ = "ffn params: " + err; return false;
    }
    if (!gc_arch_runtime_build_rope_params(hp_, rope_params_, &err)) {
        err_ = "rope params: " + err; return false;
    }

    const gc_arch_t arch = hp_.arch;
    w_tok_embd_ = loader_->find_weight(gc_tn(arch, GC_TENSOR_TOKEN_EMBD, "weight").c_str());
    w_output_   = loader_->find_weight(gc_tn(arch, GC_TENSOR_OUTPUT,     "weight").c_str());
    if (!w_tok_embd_) { err_ = "missing token_embd.weight"; return false; }
    if (!w_output_)   { w_output_ = w_tok_embd_; } // weight tying
    w_output_norm_ = loader_->find_weight(gc_tn(arch, GC_TENSOR_OUTPUT_NORM, "weight").c_str());

    // Validate embedding dimensions
    if (w_tok_embd_->tensor->ne[0] != (int64_t)hp_.n_embd) {
        err_ = "token_embd dim mismatch vs n_embd"; return false;
    }

    // Init KV buffer: use layer 0 attn params to get kv head / head_dim
    gc_attn_params_t ap0{};
    if (!gc_arch_runtime_build_attn_params(hp_, 0, 1, ap0, &err)) {
        err_ = "attn params layer0: " + err; return false;
    }
    const uint32_t n_kv_head = (uint32_t)ap0.n_head_kv;
    const uint32_t head_dim  = (uint32_t)ap0.n_embd_head_v;

    if (!kv_buf_.init(hp_.n_layer,
                      (uint32_t)cfg_.num_kv_blocks,
                      (uint32_t)cfg_.kv_block_size,
                      n_kv_head, head_dim)) {
        err_ = "KV buffer allocation failed"; return false;
    }

    return true;
}

gc_graph_runner_t::gc_graph_runner_t(const gc_model_loader_t * loader,
                                      const gc_hparams_t      & hp,
                                      config_t                  cfg)
    : loader_(loader), hp_(hp), cfg_(cfg), rng_(cfg.seed) {
    if (cfg_.num_kv_blocks < 1) cfg_.num_kv_blocks = 1;
    if (cfg_.kv_block_size  < 1) cfg_.kv_block_size = 16;
    if (!(cfg_.temperature > 0.0f)) cfg_.temperature = 1.0f;
    if (cfg_.top_k == 0 || cfg_.top_k < -1) cfg_.top_k = 40;
    if (cfg_.top_p <= 0.0f || cfg_.top_p > 1.0f) cfg_.top_p = 0.95f;
    if (cfg_.repetition_penalty < 1.0f) cfg_.repetition_penalty = 1.0f;
    if (cfg_.repetition_window  < 1)    cfg_.repetition_window  = 1;
    ok_ = init();
}

const gc_tensor_weight_t * gc_graph_runner_t::find_w(gc_tensor_role_t role,
                                                       const char * suffix,
                                                       int layer) const {
    if (!loader_) return nullptr;
    const std::string name = (layer >= 0)
        ? gc_tn(hp_.arch, role, suffix, layer)
        : gc_tn(hp_.arch, role, suffix);
    return loader_->find_weight(name.c_str());
}

// ── gc__ctx_make: convenience wrapper ────────────────────────────────────────

static ggml_context * gc__ctx_make(size_t extra_tensors) {
    // Per-layer budget: weights (no_alloc views) + intermediate activations.
    // 64MB covers even large hidden dims; views are zero-copy.
    const size_t mem = 64ULL * 1024 * 1024
                     + ggml_tensor_overhead() * (extra_tensors + 64)
                     + ggml_graph_overhead();
    ggml_init_params ip{mem, nullptr, false};
    return ggml_init(ip);
}

// ── forward() ─────────────────────────────────────────────────────────────────
// One ggml_context per layer — freed at end of each layer so memory stays
// bounded regardless of model depth or hidden size.
// The residual stream is kept in a std::vector<float> between layers.

bool gc_graph_runner_t::forward(const std::vector<int32_t> & token_ids,
                                 const std::vector<int32_t> & position_ids,
                                 const std::vector<int32_t> & block_ids,
                                 int                          num_computed,
                                 bool                         /*is_prefill*/,
                                 std::vector<float>         & logits_out) {
    if (token_ids.empty()) return false;
    const int     n_new   = (int)token_ids.size();
    const int     n_ctx   = num_computed + n_new;
    const int64_t n_embd  = (int64_t)hp_.n_embd;
    const gc_graph_cb_t cb = [](ggml_tensor *, const char *, int) {};

    const auto new_slots = gc__make_slots(block_ids, (uint32_t)cfg_.kv_block_size,
                                           num_computed, n_new);
    const auto all_slots = gc__make_slots(block_ids, (uint32_t)cfg_.kv_block_size,
                                           0, n_ctx);

    // ── Step 1: token embedding → residual[n_embd * n_new] ───────────────────
    std::vector<float> residual((size_t)n_embd * (size_t)n_new);
    {
        ggml_context * ctx = gc__ctx_make(4);
        if (!ctx) { err_ = "ctx alloc failed (embd)"; return false; }

        ggml_tensor * w_embd = gc__weight_view(ctx, loader_, w_tok_embd_);
        if (!w_embd) { ggml_free(ctx); err_ = "token_embd view failed"; return false; }

        ggml_tensor * tok_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)n_new);
        std::memcpy(tok_t->data, token_ids.data(), sizeof(int32_t) * (size_t)n_new);

        ggml_tensor * embd_out = ggml_get_rows(ctx, w_embd, tok_t);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, embd_out);
        if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
            ggml_free(ctx); err_ = "embedding compute failed"; return false;
        }
        // embd_out may be quantized — dequant via ggml_cast if needed
        if (embd_out->type != GGML_TYPE_F32) {
            ggml_tensor * f32 = ggml_cast(ctx, embd_out, GGML_TYPE_F32);
            ggml_cgraph * gf2 = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf2, f32);
            ggml_graph_compute_with_ctx(ctx, gf2, 1);
            std::memcpy(residual.data(), f32->data, sizeof(float) * residual.size());
        } else {
            std::memcpy(residual.data(), embd_out->data, sizeof(float) * residual.size());
        }
        ggml_free(ctx);
    }

    // Build causal mask (persistent across layers — just floats)
    std::vector<float> kq_mask_data((size_t)n_ctx * (size_t)n_new);
    for (int q = 0; q < n_new; ++q) {
        const int q_pos = num_computed + q;
        for (int k = 0; k < n_ctx; ++k) {
            kq_mask_data[(size_t)q * (size_t)n_ctx + k] =
                (k <= q_pos) ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }

    // ── Step 2: transformer layers ────────────────────────────────────────────
    for (uint32_t il = 0; il < hp_.n_layer; ++il) {

        std::string err;
        gc_attn_params_t ap{};
        gc_layer_flags_t lf{};
        if (!gc_arch_runtime_build_attn_params(hp_, il, (uint32_t)n_new, ap, &err) ||
            !gc_arch_runtime_build_layer_flags(hp_, il, lf, &err)) {
            err_ = err; return false;
        }

        const int64_t n_embd_hq = ap.n_embd_head_q;
        const int64_t n_embd_hv = ap.n_embd_head_v;
        const int64_t nq        = ap.n_head_q;
        const int64_t nkv       = ap.n_head_kv;

        // ── 2a: Q/K/V projection + RoPE ──────────────────────────────────────
        std::vector<float> Q_data((size_t)n_embd_hq * (size_t)nq  * (size_t)n_new);
        std::vector<float> K_data((size_t)n_embd_hq * (size_t)nkv * (size_t)n_new);
        std::vector<float> V_data((size_t)n_embd_hv * (size_t)nkv * (size_t)n_new);
        {
            ggml_context * ctx = gc__ctx_make(16);
            if (!ctx) { err_ = "ctx alloc failed (qkv)"; return false; }

            ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, (int64_t)n_new);
            std::memcpy(inp->data, residual.data(), sizeof(float) * residual.size());

            ggml_tensor * w_an = gc__weight_view(ctx, loader_, find_w(GC_TENSOR_ATTN_NORM, "weight", (int)il));
            ggml_tensor * w_q  = gc__weight_view(ctx, loader_, find_w(GC_TENSOR_ATTN_Q,    "weight", (int)il));
            ggml_tensor * w_k  = gc__weight_view(ctx, loader_, find_w(GC_TENSOR_ATTN_K,    "weight", (int)il));
            ggml_tensor * w_v  = gc__weight_view(ctx, loader_, find_w(GC_TENSOR_ATTN_V,    "weight", (int)il));
            if (!w_an || !w_q || !w_k || !w_v) {
                ggml_free(ctx); err_ = "attn weight missing"; return false;
            }

            ggml_tensor * pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)n_new);
            std::memcpy(pos_t->data, position_ids.data(), sizeof(int32_t) * (size_t)n_new);

            ggml_tensor * normed = gc_build_norm(ctx, inp, w_an, nullptr,
                                                  norm_params_.type, norm_params_.eps, cb, (int)il);
            ggml_tensor * Qcur = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w_q, normed),
                                                   n_embd_hq, nq, (int64_t)n_new);
            ggml_tensor * Kcur = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w_k, normed),
                                                   n_embd_hq, nkv, (int64_t)n_new);
            ggml_tensor * Vcur = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w_v, normed),
                                                   n_embd_hv, nkv, (int64_t)n_new);

            if (lf.has_attn_q_norm) {
                ggml_tensor * w_qn = gc__weight_view(ctx, loader_, find_w(GC_TENSOR_ATTN_Q_NORM, "weight", (int)il));
                if (w_qn) {
                    Qcur = ggml_reshape_3d(ctx,
                        gc_build_norm(ctx, ggml_reshape_2d(ctx, Qcur, n_embd_hq, nq*(int64_t)n_new),
                                      w_qn, nullptr, norm_params_.type, norm_params_.eps, cb, (int)il),
                        n_embd_hq, nq, (int64_t)n_new);
                }
            }
            if (lf.has_attn_k_norm) {
                ggml_tensor * w_kn = gc__weight_view(ctx, loader_, find_w(GC_TENSOR_ATTN_K_NORM, "weight", (int)il));
                if (w_kn) {
                    Kcur = ggml_reshape_3d(ctx,
                        gc_build_norm(ctx, ggml_reshape_2d(ctx, Kcur, n_embd_hq, nkv*(int64_t)n_new),
                                      w_kn, nullptr, norm_params_.type, norm_params_.eps, cb, (int)il),
                        n_embd_hq, nkv, (int64_t)n_new);
                }
            }

            if (lf.apply_rope && rope_params_.n_dims > 0) {
                Qcur = gc_rope_apply(ctx, Qcur, pos_t, nullptr, rope_params_);
                Kcur = gc_rope_apply(ctx, Kcur, pos_t, nullptr, rope_params_);
            }

            // Materialise Q/K/V
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, Kcur);
            ggml_build_forward_expand(gf, Vcur);
            if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
                ggml_free(ctx); err_ = "qkv compute failed"; return false;
            }

            std::memcpy(Q_data.data(), Qcur->data, sizeof(float) * Q_data.size());
            std::memcpy(K_data.data(), Kcur->data, sizeof(float) * K_data.size());
            std::memcpy(V_data.data(), Vcur->data, sizeof(float) * V_data.size());
            ggml_free(ctx);
        }

        // Write new K/V into paged buffer
        gc__kv_write(kv_buf_, il, 0, new_slots, K_data.data(), (uint32_t)n_new);
        gc__kv_write(kv_buf_, il, 1, new_slots, V_data.data(), (uint32_t)n_new);

        // ── 2b: Attention + output proj ───────────────────────────────────────
        std::vector<float> attn_result((size_t)n_embd * (size_t)n_new);
        {
            ggml_context * ctx = gc__ctx_make(16);
            if (!ctx) { err_ = "ctx alloc failed (attn)"; return false; }

            ggml_tensor * Qcur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                n_embd_hq, nq, (int64_t)n_new);
            std::memcpy(Qcur->data, Q_data.data(), sizeof(float) * Q_data.size());

            // Read full K/V from paged buffer
            ggml_tensor * K_ctx = gc__kv_view(ctx, kv_buf_, il, 0, all_slots, (uint32_t)n_ctx);
            ggml_tensor * V_ctx = gc__kv_view(ctx, kv_buf_, il, 1, all_slots, (uint32_t)n_ctx);
            if (!K_ctx || !V_ctx) { ggml_free(ctx); err_ = "KV view failed"; return false; }
            K_ctx = ggml_reshape_3d(ctx, K_ctx, n_embd_hq, nkv, (int64_t)n_ctx);
            V_ctx = ggml_reshape_3d(ctx, V_ctx, n_embd_hv, nkv, (int64_t)n_ctx);

            ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                (int64_t)n_ctx, (int64_t)n_new);
            std::memcpy(mask->data, kq_mask_data.data(),
                        sizeof(float) * kq_mask_data.size());

            gc_attn_params_t ap_full = ap;
            ap_full.n_tokens = (int64_t)n_new;

            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_tensor * attn_out = gc_build_attn_mha(ctx, gf,
                Qcur, K_ctx, V_ctx, mask, nullptr, ap_full, cb, (int)il);

            ggml_tensor * w_o = gc__weight_view(ctx, loader_, find_w(GC_TENSOR_ATTN_OUT, "weight", (int)il));
            if (!w_o) { ggml_free(ctx); err_ = "attn_out weight missing"; return false; }
            attn_out = ggml_mul_mat(ctx, w_o, attn_out);

            if (lf.has_post_attn_norm) {
                ggml_tensor * w_pan = gc__weight_view(ctx, loader_,
                    find_w(GC_TENSOR_POST_ATTN_NORM, "weight", (int)il));
                if (w_pan)
                    attn_out = gc_build_norm(ctx, attn_out, w_pan, nullptr,
                                             norm_params_.type, norm_params_.eps, cb, (int)il);
            }

            ggml_build_forward_expand(gf, attn_out);
            if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
                ggml_free(ctx); err_ = "attn compute failed"; return false;
            }
            std::memcpy(attn_result.data(), attn_out->data,
                        sizeof(float) * attn_result.size());
            ggml_free(ctx);
        }

        // Residual add after attention
        for (size_t i = 0; i < residual.size(); ++i) residual[i] += attn_result[i];

        // ── 2c: FFN ───────────────────────────────────────────────────────────
        std::vector<float> ffn_result((size_t)n_embd * (size_t)n_new);
        {
            ggml_context * ctx = gc__ctx_make(8);
            if (!ctx) { err_ = "ctx alloc failed (ffn)"; return false; }

            ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, (int64_t)n_new);
            std::memcpy(inp->data, residual.data(), sizeof(float) * residual.size());

            ggml_tensor * w_fn = gc__weight_view(ctx, loader_, find_w(GC_TENSOR_FFN_NORM, "weight", (int)il));
            ggml_tensor * w_up = gc__weight_view(ctx, loader_, find_w(GC_TENSOR_FFN_UP,   "weight", (int)il));
            ggml_tensor * w_dn = gc__weight_view(ctx, loader_, find_w(GC_TENSOR_FFN_DOWN, "weight", (int)il));
            ggml_tensor * w_gt = ffn_params_.has_gate
                ? gc__weight_view(ctx, loader_, find_w(GC_TENSOR_FFN_GATE, "weight", (int)il))
                : nullptr;
            if (!w_fn || !w_up || !w_dn) {
                ggml_free(ctx); err_ = "ffn weight missing"; return false;
            }

            ggml_tensor * normed = gc_build_norm(ctx, inp, w_fn, nullptr,
                                                  norm_params_.type, norm_params_.eps, cb, (int)il);
            ggml_tensor * ffn_out = gc_build_ffn(ctx, normed,
                w_up, nullptr, w_gt, nullptr, w_dn, nullptr,
                ffn_params_.act, ffn_params_.gate_mode, cb, (int)il);

            if (lf.has_post_ffn_norm) {
                ggml_tensor * w_pfn = gc__weight_view(ctx, loader_,
                    find_w(GC_TENSOR_POST_MLP_NORM, "weight", (int)il));
                if (w_pfn)
                    ffn_out = gc_build_norm(ctx, ffn_out, w_pfn, nullptr,
                                            norm_params_.type, norm_params_.eps, cb, (int)il);
            }

            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, ffn_out);
            if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
                ggml_free(ctx); err_ = "ffn compute failed"; return false;
            }
            std::memcpy(ffn_result.data(), ffn_out->data,
                        sizeof(float) * ffn_result.size());
            ggml_free(ctx);
        }

        // Residual add after FFN
        for (size_t i = 0; i < residual.size(); ++i) residual[i] += ffn_result[i];
    }

    // ── Step 3: output norm + lm_head ─────────────────────────────────────────
    {
        ggml_context * ctx = gc__ctx_make(8);
        if (!ctx) { err_ = "ctx alloc failed (lm_head)"; return false; }

        ggml_tensor * cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, (int64_t)n_new);
        std::memcpy(cur->data, residual.data(), sizeof(float) * residual.size());

        if (w_output_norm_) {
            ggml_tensor * w_on = gc__weight_view(ctx, loader_, w_output_norm_);
            if (w_on)
                cur = gc_build_norm(ctx, cur, w_on, nullptr,
                                    norm_params_.type, norm_params_.eps, cb, -1);
        }

        ggml_tensor * w_out = gc__weight_view(ctx, loader_, w_output_);
        if (!w_out) { ggml_free(ctx); err_ = "output weight view failed"; return false; }

        ggml_tensor * logits_t = ggml_mul_mat(ctx, w_out, cur);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, logits_t);
        if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
            ggml_free(ctx); err_ = "lm_head compute failed"; return false;
        }

        const int64_t vocab = logits_t->ne[0];
        const float * data  = static_cast<float *>(logits_t->data);
        // Take logits for the last token position
        const size_t offset = (size_t)(n_new - 1) * (size_t)vocab;
        logits_out.assign(data + offset, data + offset + (size_t)vocab);
        ggml_free(ctx);
    }

    return true;
}

// ── Sampling ──────────────────────────────────────────────────────────────────

void gc_graph_runner_t::apply_rep_penalty(std::vector<float> & logits,
                                           const req_state_t & st) const {
    if (cfg_.repetition_penalty <= 1.0f || st.recent_tokens.empty()) return;
    const size_t window = std::min(st.recent_tokens.size(), (size_t)cfg_.repetition_window);
    for (size_t i = st.recent_tokens.size() - window; i < st.recent_tokens.size(); ++i) {
        const int32_t tok = st.recent_tokens[i];
        if (tok < 0 || tok >= (int32_t)logits.size()) continue;
        float & l = logits[(size_t)tok];
        l = (l > 0.0f) ? l / cfg_.repetition_penalty : l * cfg_.repetition_penalty;
    }
}

int32_t gc_graph_runner_t::sample(std::vector<float> logits) {
    const int vocab = (int)logits.size();

    for (float & l : logits) l /= cfg_.temperature;

    std::vector<int> idx((size_t)vocab);
    std::iota(idx.begin(), idx.end(), 0);
    const int k = (cfg_.top_k > 0 && cfg_.top_k < vocab) ? cfg_.top_k : vocab;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
        [&](int a, int b) { return logits[(size_t)a] > logits[(size_t)b]; });
    idx.resize((size_t)k);

    float max_l = -std::numeric_limits<float>::infinity();
    for (int i : idx) max_l = std::max(max_l, logits[(size_t)i]);

    std::vector<float> probs;
    probs.reserve(idx.size());
    float sum = 0.0f;
    for (int i : idx) {
        const float p = std::exp(logits[(size_t)i] - max_l);
        probs.push_back(p);
        sum += p;
    }
    if (sum <= 0.0f) return idx.front();
    for (float & p : probs) p /= sum;

    if (cfg_.top_p < 1.0f) {
        float cum = 0.0f;
        size_t keep = 0;
        for (; keep < probs.size(); ++keep) {
            cum += probs[keep];
            if (cum >= cfg_.top_p) break;
        }
        keep = std::min(keep + 1, probs.size());
        idx.resize(keep);
        probs.resize(keep);
        float renorm = 0.0f;
        for (float p : probs) renorm += p;
        if (renorm > 0.0f) for (float & p : probs) p /= renorm;
    }

    std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
    return idx[dist(rng_)];
}

// ── execute() ─────────────────────────────────────────────────────────────────

gc_model_output_t gc_graph_runner_t::execute(const gc_batch_t & batch) {
    gc_model_output_t out;
    out.req_ids.reserve(batch.entries.size());
    out.sampled_tokens.reserve(batch.entries.size());

    std::unordered_map<std::string, bool> active;
    active.reserve(batch.entries.size());

    for (const auto & e : batch.entries) {
        out.req_ids.push_back(e.req_id);
        active[e.req_id] = true;
        req_state_t & st = req_state_[e.req_id];

        if (e.is_prefill) {
            std::vector<float> logits;
            forward(e.token_ids, e.position_ids, e.block_ids,
                    e.num_computed, true, logits);

            // Track all prompt tokens for repetition penalty.
            st.recent_tokens.insert(st.recent_tokens.end(),
                                    e.token_ids.begin(), e.token_ids.end());
            if (st.recent_tokens.size() > (size_t)cfg_.repetition_window) {
                st.recent_tokens.erase(st.recent_tokens.begin(),
                    st.recent_tokens.end() - cfg_.repetition_window);
            }

            // On the final prefill chunk we have logits for the last prompt
            // position — sample the first output token here (mirrors vllm
            // GPUModelRunner: prefill produces logits and samples in one step).
            if (e.is_last_prefill && !logits.empty()) {
                apply_rep_penalty(logits, st);
                const int32_t tok = sample(std::move(logits));
                out.sampled_tokens.push_back(tok);
                st.recent_tokens.push_back(tok);
                if (st.recent_tokens.size() > (size_t)cfg_.repetition_window) {
                    st.recent_tokens.erase(st.recent_tokens.begin());
                }
            } else {
                // Mid-prefill chunk: no output token yet.
                out.sampled_tokens.push_back(-1);
            }
            continue;
        }

        // Decode step
        std::vector<float> logits;
        if (!forward(e.token_ids, e.position_ids, e.block_ids,
                     e.num_computed, false, logits) || logits.empty()) {
            out.sampled_tokens.push_back(0);
            continue;
        }

        apply_rep_penalty(logits, st);
        const int32_t tok = sample(std::move(logits));
        out.sampled_tokens.push_back(tok);

        st.recent_tokens.push_back(tok);
        if (st.recent_tokens.size() > (size_t)cfg_.repetition_window) {
            st.recent_tokens.erase(st.recent_tokens.begin());
        }
    }

    // Purge state for completed requests
    for (auto it = req_state_.begin(); it != req_state_.end();) {
        it = active.count(it->first) ? std::next(it) : req_state_.erase(it);
    }

    return out;
}
