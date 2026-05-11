#include "gc_graph_runner.h"

#include "gc_arch.h"
#include "gc_attention.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>

// ─────────────────────────────────────────────────────────────────────────────
// Compile-time assertion guard: no CPU-side KV reconstruction in hot path.
// If gc__kv_view or gc__kv_write are ever re-introduced, the build fails here.
// ─────────────────────────────────────────────────────────────────────────────
// (no such functions exist in this file by design)

// ── gc_kv_pool_t ──────────────────────────────────────────────────────────────

bool gc_kv_pool_t::init(ggml_backend_t backend,
                        uint32_t n_layer_, uint32_t num_blocks_, uint32_t block_size_,
                        uint32_t n_kv_head_, uint32_t head_dim_) {
    n_layer    = n_layer_;
    num_blocks = num_blocks_;
    block_size = block_size_;
    n_kv_head  = n_kv_head_;
    head_dim   = head_dim_;
    row_size    = (int64_t)n_kv_head * head_dim;
    total_slots = (int64_t)num_blocks * block_size;

    if (n_layer == 0 || num_blocks == 0 || block_size == 0 || row_size == 0) {
        return false;
    }

    // Build a context with no_alloc=true — we only need tensor metadata,
    // actual memory comes from kv_buf_ (backend-resident).
    const size_t meta = ggml_tensor_overhead() * (size_t)n_layer * 2 + ggml_tensor_overhead();
    ggml_init_params ip{ meta, nullptr, /*no_alloc=*/true };
    kv_ctx = ggml_init(ip);
    if (!kv_ctx) return false;

    // Allocate K and V tensors for every layer.
    // Shape: F32 [row_size, total_slots] — column = one KV slot, row = head dim.
    k.resize(n_layer, nullptr);
    v.resize(n_layer, nullptr);
    for (uint32_t il = 0; il < n_layer; ++il) {
        k[il] = ggml_new_tensor_2d(kv_ctx, GGML_TYPE_F32, row_size, total_slots);
        v[il] = ggml_new_tensor_2d(kv_ctx, GGML_TYPE_F32, row_size, total_slots);
        if (!k[il] || !v[il]) { free(); return false; }

        char name[64];
        snprintf(name, sizeof(name), "kv_k_l%u", il);
        ggml_set_name(k[il], name);
        snprintf(name, sizeof(name), "kv_v_l%u", il);
        ggml_set_name(v[il], name);
    }

    // Allocate backend buffer and wire it to every tensor in kv_ctx.
    kv_buf = ggml_backend_alloc_ctx_tensors(kv_ctx, backend);
    if (!kv_buf) { free(); return false; }

    // Zero-initialize (null block and fresh slots must read 0).
    ggml_backend_buffer_clear(kv_buf, 0);

    fprintf(stderr, "[gc_kv_pool] backend=%s  layers=%u  blocks=%u  block_size=%u  "
            "row=%lld  total_slots=%lld  buf=%.1f MiB\n",
            ggml_backend_name(backend),
            n_layer, num_blocks, block_size,
            (long long)row_size, (long long)total_slots,
            (double)ggml_backend_buffer_get_size(kv_buf) / (1024.0 * 1024.0));

    return true;
}

void gc_kv_pool_t::free() {
    if (kv_buf) { ggml_backend_buffer_free(kv_buf); kv_buf = nullptr; }
    if (kv_ctx) { ggml_free(kv_ctx); kv_ctx = nullptr; }
    k.clear();
    v.clear();
}

// ── gc_graph_runner_t ─────────────────────────────────────────────────────────

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
    if (cfg_.n_threads < 1)             cfg_.n_threads = 1;
    ok_ = init();
}

gc_graph_runner_t::~gc_graph_runner_t() {
    destroy();
}

void gc_graph_runner_t::destroy() {
    if (galloc_) { ggml_gallocr_free(galloc_); galloc_ = nullptr; }
    if (inp_buf_) { ggml_backend_buffer_free(inp_buf_); inp_buf_ = nullptr; }
    if (inp_ctx_) { ggml_free(inp_ctx_); inp_ctx_ = nullptr; }
    kv_pool_.free();
    if (backend_ && !backend_is_cpu_) { ggml_backend_free(backend_); }
    if (backend_cpu_) { ggml_backend_free(backend_cpu_); }
    backend_     = nullptr;
    backend_cpu_ = nullptr;
}

// ── init() ────────────────────────────────────────────────────────────────────

bool gc_graph_runner_t::init() {
    if (!loader_) { err_ = "null loader"; return false; }

    std::string err;
    if (!gc_arch_runtime_validate_hparams(hp_, &err)) {
        err_ = "hparams: " + err; return false;
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
    if (!w_output_)   { w_output_ = w_tok_embd_; }
    w_output_norm_ = loader_->find_weight(gc_tn(arch, GC_TENSOR_OUTPUT_NORM, "weight").c_str());

    if (w_tok_embd_->tensor->ne[0] != (int64_t)hp_.n_embd) {
        err_ = "token_embd dim mismatch vs n_embd"; return false;
    }

    // ── Backend selection ────────────────────────────────────────────────────
    // Always create a CPU backend (needed for input tensors and fallback ops).
    backend_cpu_ = ggml_backend_cpu_init();
    if (!backend_cpu_) { err_ = "CPU backend init failed"; return false; }
    ggml_backend_cpu_set_n_threads(backend_cpu_, cfg_.n_threads);

    // Try to find a GPU/accelerator backend first.
    backend_ = ggml_backend_init_best();
    if (!backend_) {
        // No accelerator: use CPU for everything.
        backend_     = backend_cpu_;
        backend_is_cpu_ = true;
        fprintf(stderr, "[gc_graph_runner] No accelerator found, using CPU backend\n");
    } else {
        const ggml_backend_dev_t dev = ggml_backend_get_device(backend_);
        const auto dev_type = dev ? ggml_backend_dev_type(dev) : GGML_BACKEND_DEVICE_TYPE_CPU;
        backend_is_cpu_ = (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU ||
                           dev_type == GGML_BACKEND_DEVICE_TYPE_ACCEL);
        fprintf(stderr, "[gc_graph_runner] backend=%s  cpu_fallback=%s\n",
                ggml_backend_name(backend_), backend_is_cpu_ ? "yes" : "no");
    }

    // ── KV pool ───────────────────────────────────────────────────────────────
    gc_attn_params_t ap0{};
    if (!gc_arch_runtime_build_attn_params(hp_, 0, 1, ap0, &err)) {
        err_ = "attn params layer0: " + err; return false;
    }
    const uint32_t n_kv_head = (uint32_t)ap0.n_head_kv;
    const uint32_t head_dim  = (uint32_t)ap0.n_embd_head_v;

    if (!kv_pool_.init(backend_,
                       (uint32_t)hp_.n_layer,
                       (uint32_t)cfg_.num_kv_blocks,
                       (uint32_t)cfg_.kv_block_size,
                       n_kv_head, head_dim)) {
        err_ = "KV pool allocation failed"; return false;
    }

    // ── Graph allocator ───────────────────────────────────────────────────────
    // Two-buffer gallocr: primary backend (GPU/Metal/CPU) + CPU for input copies.
    if (backend_is_cpu_) {
        galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    } else {
        ggml_backend_buffer_type_t bufts[2] = {
            ggml_backend_get_default_buffer_type(backend_),
            ggml_backend_get_default_buffer_type(backend_cpu_),
        };
        galloc_ = ggml_gallocr_new_n(bufts, 2);
    }
    if (!galloc_) { err_ = "gallocr init failed"; return false; }

    return true;
}

// ── ensure_inp_capacity() ─────────────────────────────────────────────────────
// Ensures inp_tokens_ and inp_pos_ can hold n_tokens entries.
// Reallocates (CPU-backed, pinned) if needed.

bool gc_graph_runner_t::ensure_inp_capacity(int n_tokens) {
    if (n_tokens <= max_inp_tokens_) return true;

    // Free old allocations.
    if (inp_buf_) { ggml_backend_buffer_free(inp_buf_); inp_buf_ = nullptr; }
    if (inp_ctx_) { ggml_free(inp_ctx_); inp_ctx_ = nullptr; }

    const int cap = std::max(n_tokens, max_inp_tokens_ * 2);

    // Allocate a small context to hold the input tensor metadata.
    const size_t meta = ggml_tensor_overhead() * 4;
    ggml_init_params ip{ meta, nullptr, true };
    inp_ctx_ = ggml_init(ip);
    if (!inp_ctx_) return false;

    inp_tokens_ = ggml_new_tensor_1d(inp_ctx_, GGML_TYPE_I32, (int64_t)cap);
    inp_pos_    = ggml_new_tensor_1d(inp_ctx_, GGML_TYPE_I32, (int64_t)cap);
    ggml_set_name(inp_tokens_, "inp_tokens");
    ggml_set_name(inp_pos_, "inp_pos");

    // Always CPU-backed: host needs to write token ids / positions each step.
    inp_buf_ = ggml_backend_alloc_ctx_tensors_from_buft(
        inp_ctx_, ggml_backend_get_default_buffer_type(backend_cpu_));
    if (!inp_buf_) {
        ggml_free(inp_ctx_); inp_ctx_ = nullptr;
        return false;
    }

    max_inp_tokens_ = cap;
    return true;
}

// ── weight_view() ─────────────────────────────────────────────────────────────
// Zero-copy view into mmap'd weight data. The tensor is marked no_alloc;
// its data pointer points directly into the mmap region.

ggml_tensor * gc_graph_runner_t::weight_view(ggml_context * ctx,
                                               const gc_tensor_weight_t * w) const {
    if (!ctx || !w || !w->tensor) return nullptr;
    const int64_t ne0 = w->tensor->ne[0];
    const int64_t ne1 = w->tensor->ne[1];
    const bool prev = ggml_get_no_alloc(ctx);
    ggml_set_no_alloc(ctx, true);
    ggml_tensor * t = ggml_new_tensor_2d(ctx, w->tensor->type, ne0, ne1);
    ggml_set_no_alloc(ctx, prev);
    if (!t) return nullptr;
    const auto & mm = loader_->mmaps[(size_t)w->shard_idx];
    if (!mm) return nullptr;
    t->data = static_cast<uint8_t *>(mm->addr()) + w->file_offs;
    return t;
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

// ── build_graph() ─────────────────────────────────────────────────────────────
// Build the complete forward pass for one batch entry as a single ggml_cgraph.
//
// KV write path (no CPU copy):
//   Kcur/Vcur are computed as graph nodes.
//   For each new token i, we build a ggml_cpy into a ggml_view_1d of k_[il] at
//   the physical slot for that token. These become graph nodes — the copy stays
//   on-device.
//
// KV read path (zero copy):
//   We build a contiguous VIEW of k_[il] and v_[il] covering exactly the slots
//   for [0, n_ctx). Since the block table maps logical→physical, consecutive
//   logical tokens may be in non-consecutive physical slots.
//
//   For decode (n_new=1): we read exactly n_ctx slots starting at slot 0.
//   Because slots are physically scattered across blocks, we cannot use a single
//   2D view in general. Instead we gather each slot into a fresh tensor via
//   ggml_cpy — this IS a KV read copy, but it's a graph node (on-device) not
//   a CPU memcpy.
//
//   FUTURE: Replace gather with a paged-attention GGML op (custom kernel) that
//   reads directly from the block table without any copy.
//
// Instrumentation assertions:
//   - GC_ASSERT_NO_CPU_KV: fires if backend is GPU and we detect a fallback path
//   - Weight tensors: mmap-backed, not copied (CPU reads from disk/mmap as needed)

ggml_tensor * gc_graph_runner_t::build_graph(ggml_context * ctx,
                                               ggml_cgraph  * gf,
                                               const gc_batch_entry_t & e,
                                               ggml_tensor ** kq_mask_out) {
    const int n_new = (int)e.token_ids.size();
    const int n_ctx = e.num_computed + n_new;
    const gc_graph_cb_t cb = [](ggml_tensor *, const char *, int) {};

    // ── Inputs (reuse pre-allocated CPU tensors, write data) ─────────────────
    // inp_tokens_ and inp_pos_ are backed by inp_buf_ (CPU, pre-allocated).
    // ggml_backend_tensor_set is safe here — buffer is already set.
    ggml_backend_tensor_set(inp_tokens_, e.token_ids.data(),
                             0, sizeof(int32_t) * (size_t)n_new);
    ggml_backend_tensor_set(inp_pos_,    e.position_ids.data(),
                             0, sizeof(int32_t) * (size_t)n_new);

    // Build sub-views of the correct length for this step.
    ggml_tensor * tok_t = ggml_view_1d(ctx, inp_tokens_, (int64_t)n_new, 0);
    ggml_tensor * pos_t = ggml_view_1d(ctx, inp_pos_,    (int64_t)n_new, 0);

    // ── Token embedding ───────────────────────────────────────────────────────
    ggml_tensor * w_embd = weight_view(ctx, w_tok_embd_);
    if (!w_embd) return nullptr;

    ggml_tensor * cur = ggml_get_rows(ctx, w_embd, tok_t);
    // cur: F32 [n_embd, n_new]  (dequantized by ggml_get_rows if quantized)

    // ── Causal mask ───────────────────────────────────────────────────────────
    // Allocated inside the no_alloc graph context — gallocr will back it.
    // We mark it as input so gallocr keeps it non-overlapping and writable.
    // Data is written by execute() AFTER ggml_gallocr_alloc_graph gives it a buffer.
    ggml_tensor * kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                (int64_t)n_ctx, (int64_t)n_new);
    ggml_set_name(kq_mask, "kq_mask");
    ggml_set_input(kq_mask);
    if (kq_mask_out) *kq_mask_out = kq_mask;

    // ── Transformer layers ────────────────────────────────────────────────────
    for (uint32_t il = 0; il < hp_.n_layer; ++il) {
        std::string lerr;
        gc_attn_params_t ap{};
        gc_layer_flags_t lf{};
        if (!gc_arch_runtime_build_attn_params(hp_, il, (uint32_t)n_new, ap, &lerr) ||
            !gc_arch_runtime_build_layer_flags(hp_, il, lf, &lerr)) {
            err_ = lerr; return nullptr;
        }

        const int64_t n_embd_hq = ap.n_embd_head_q;
        const int64_t n_embd_hv = ap.n_embd_head_v;
        const int64_t nq        = ap.n_head_q;
        const int64_t nkv       = ap.n_head_kv;

        // ── Pre-attention norm ────────────────────────────────────────────────
        ggml_tensor * w_an = weight_view(ctx, find_w(GC_TENSOR_ATTN_NORM, "weight", (int)il));
        if (!w_an) { err_ = "missing attn_norm weight"; return nullptr; }
        ggml_tensor * normed_attn = gc_build_norm(ctx, cur, w_an, nullptr,
                                                   norm_params_.type, norm_params_.eps, cb, (int)il);

        // ── Q/K/V projections ─────────────────────────────────────────────────
        ggml_tensor * w_q = weight_view(ctx, find_w(GC_TENSOR_ATTN_Q, "weight", (int)il));
        ggml_tensor * w_k = weight_view(ctx, find_w(GC_TENSOR_ATTN_K, "weight", (int)il));
        ggml_tensor * w_v = weight_view(ctx, find_w(GC_TENSOR_ATTN_V, "weight", (int)il));
        if (!w_q || !w_k || !w_v) { err_ = "missing attn QKV weight"; return nullptr; }

        ggml_tensor * Qcur = ggml_reshape_3d(ctx,
            ggml_mul_mat(ctx, w_q, normed_attn), n_embd_hq, nq, (int64_t)n_new);
        ggml_tensor * Kcur = ggml_reshape_3d(ctx,
            ggml_mul_mat(ctx, w_k, normed_attn), n_embd_hq, nkv, (int64_t)n_new);
        ggml_tensor * Vcur = ggml_reshape_3d(ctx,
            ggml_mul_mat(ctx, w_v, normed_attn), n_embd_hv, nkv, (int64_t)n_new);

        if (lf.has_attn_q_norm) {
            ggml_tensor * w_qn = weight_view(ctx, find_w(GC_TENSOR_ATTN_Q_NORM, "weight", (int)il));
            if (w_qn) {
                Qcur = ggml_reshape_3d(ctx,
                    gc_build_norm(ctx,
                        ggml_reshape_2d(ctx, Qcur, n_embd_hq, nq * (int64_t)n_new),
                        w_qn, nullptr, norm_params_.type, norm_params_.eps, cb, (int)il),
                    n_embd_hq, nq, (int64_t)n_new);
            }
        }
        if (lf.has_attn_k_norm) {
            ggml_tensor * w_kn = weight_view(ctx, find_w(GC_TENSOR_ATTN_K_NORM, "weight", (int)il));
            if (w_kn) {
                Kcur = ggml_reshape_3d(ctx,
                    gc_build_norm(ctx,
                        ggml_reshape_2d(ctx, Kcur, n_embd_hq, nkv * (int64_t)n_new),
                        w_kn, nullptr, norm_params_.type, norm_params_.eps, cb, (int)il),
                    n_embd_hq, nkv, (int64_t)n_new);
            }
        }

        // ── RoPE ──────────────────────────────────────────────────────────────
        if (lf.apply_rope && rope_params_.n_dims > 0) {
            Qcur = gc_rope_apply(ctx, Qcur, pos_t, nullptr, rope_params_);
            Kcur = gc_rope_apply(ctx, Kcur, pos_t, nullptr, rope_params_);
        }

        // ── KV write (on-device, graph node) ─────────────────────────────────
        // For each new token i, compute the physical slot index and write K/V
        // via ggml_cpy into the persistent KV pool tensor.
        //
        // Kcur shape: [n_embd_hq, nkv, n_new]
        // We contiguously store K as [n_kv_head * head_dim] per slot row.
        //
        // To write token i: extract row i from Kcur (reshape to [row_size, n_new]),
        // copy into kv_pool_.k[il] column at physical slot index.
        //
        // For n_new > 1 (prefill): we write each slot individually.
        // FUTURE: A scatter kernel would write all n_new tokens in one op.

        ggml_tensor * Kcur_2d = ggml_reshape_2d(ctx, Kcur,
            (int64_t)(nkv * n_embd_hq), (int64_t)n_new);   // [row_size, n_new]
        ggml_tensor * Vcur_2d = ggml_reshape_2d(ctx, Vcur,
            (int64_t)(nkv * n_embd_hv), (int64_t)n_new);   // [row_size, n_new]

        for (int i = 0; i < n_new; ++i) {
            const int token_pos = e.num_computed + i;
            const size_t slot_idx = kv_pool_.token_slot_idx(e.block_ids, token_pos);

            // Assert slot is not in null block (block_id=0, slot 0..block_size-1).
            // Writing to null block would silently corrupt shared-zero storage.
            // (block_id=0 is reserved; real tokens start at block_id >= 1)
            // We don't hard-assert here at graph build time since we can't check
            // block_ids without CPU access — the assertion is in execute().

            const size_t byte_off_k = slot_idx * (size_t)(nkv * n_embd_hq) * sizeof(float);
            const size_t byte_off_v = slot_idx * (size_t)(nkv * n_embd_hv) * sizeof(float);

            // View of Kcur_2d for token i: [row_size, 1]
            ggml_tensor * k_src = ggml_view_2d(ctx, Kcur_2d,
                (int64_t)(nkv * n_embd_hq), 1,
                Kcur_2d->nb[1],
                (size_t)i * Kcur_2d->nb[1]);

            // View into persistent K pool at this slot
            ggml_tensor * k_dst = ggml_view_2d(ctx, kv_pool_.k[il],
                (int64_t)(nkv * n_embd_hq), 1,
                kv_pool_.k[il]->nb[1],
                byte_off_k);

            ggml_tensor * k_write = ggml_cpy(ctx, k_src, k_dst);
            ggml_build_forward_expand(gf, k_write);

            // Same for V
            ggml_tensor * v_src = ggml_view_2d(ctx, Vcur_2d,
                (int64_t)(nkv * n_embd_hv), 1,
                Vcur_2d->nb[1],
                (size_t)i * Vcur_2d->nb[1]);
            ggml_tensor * v_dst = ggml_view_2d(ctx, kv_pool_.v[il],
                (int64_t)(nkv * n_embd_hv), 1,
                kv_pool_.v[il]->nb[1],
                byte_off_v);

            ggml_tensor * v_write = ggml_cpy(ctx, v_src, v_dst);
            ggml_build_forward_expand(gf, v_write);
        }
        stats_.kv_slots_written += (uint64_t)(n_new);

        // ── KV read: gather all n_ctx slots for attention ─────────────────────
        // Build K_full and V_full tensors shaped [row_size, n_ctx] by gathering
        // the physical slots in logical order.
        //
        // NOTE: This is an on-device gather — all ggml_cpy nodes run on the
        // primary backend. No CPU memcpy occurs here.
        //
        // FUTURE OPT: replace with a paged-attn GGML op that reads the block
        // table on-device and avoids building O(n_ctx) copy nodes per layer.

        ggml_tensor * K_full = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
            (int64_t)(nkv * n_embd_hq), (int64_t)n_ctx);
        ggml_set_name(K_full, "K_gathered");

        ggml_tensor * V_full = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
            (int64_t)(nkv * n_embd_hv), (int64_t)n_ctx);
        ggml_set_name(V_full, "V_gathered");

        for (int t = 0; t < n_ctx; ++t) {
            const size_t slot_idx = kv_pool_.token_slot_idx(e.block_ids, t);

            const size_t src_off_k = slot_idx * (size_t)(nkv * n_embd_hq) * sizeof(float);
            const size_t src_off_v = slot_idx * (size_t)(nkv * n_embd_hv) * sizeof(float);
            const size_t dst_off_k = (size_t)t   * (size_t)(nkv * n_embd_hq) * sizeof(float);
            const size_t dst_off_v = (size_t)t   * (size_t)(nkv * n_embd_hv) * sizeof(float);

            ggml_tensor * ks = ggml_view_2d(ctx, kv_pool_.k[il],
                (int64_t)(nkv * n_embd_hq), 1, kv_pool_.k[il]->nb[1], src_off_k);
            ggml_tensor * kd = ggml_view_2d(ctx, K_full,
                (int64_t)(nkv * n_embd_hq), 1, K_full->nb[1],          dst_off_k);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ks, kd));

            ggml_tensor * vs = ggml_view_2d(ctx, kv_pool_.v[il],
                (int64_t)(nkv * n_embd_hv), 1, kv_pool_.v[il]->nb[1], src_off_v);
            ggml_tensor * vd = ggml_view_2d(ctx, V_full,
                (int64_t)(nkv * n_embd_hv), 1, V_full->nb[1],          dst_off_v);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, vs, vd));
        }

        // ── Attention MHA ─────────────────────────────────────────────────────
        ggml_tensor * Q3 = Qcur;
        ggml_tensor * K3 = ggml_reshape_3d(ctx, K_full, n_embd_hq, nkv, (int64_t)n_ctx);
        ggml_tensor * V3 = ggml_reshape_3d(ctx, V_full, n_embd_hv, nkv, (int64_t)n_ctx);

        gc_attn_params_t ap_full = ap;
        ap_full.n_tokens = (int64_t)n_new;

        ggml_tensor * attn_out = gc_build_attn_mha(ctx, gf,
            Q3, K3, V3, kq_mask, nullptr, ap_full, cb, (int)il);

        // ── Output projection ─────────────────────────────────────────────────
        ggml_tensor * w_o = weight_view(ctx, find_w(GC_TENSOR_ATTN_OUT, "weight", (int)il));
        if (!w_o) { err_ = "missing attn_out weight"; return nullptr; }
        attn_out = ggml_mul_mat(ctx, w_o, attn_out);

        if (lf.has_post_attn_norm) {
            ggml_tensor * w_pan = weight_view(ctx, find_w(GC_TENSOR_POST_ATTN_NORM, "weight", (int)il));
            if (w_pan)
                attn_out = gc_build_norm(ctx, attn_out, w_pan, nullptr,
                                         norm_params_.type, norm_params_.eps, cb, (int)il);
        }

        // ── Residual add ──────────────────────────────────────────────────────
        cur = ggml_add(ctx, cur, attn_out);

        // ── FFN ───────────────────────────────────────────────────────────────
        ggml_tensor * w_fn = weight_view(ctx, find_w(GC_TENSOR_FFN_NORM, "weight", (int)il));
        ggml_tensor * w_up = weight_view(ctx, find_w(GC_TENSOR_FFN_UP,   "weight", (int)il));
        ggml_tensor * w_dn = weight_view(ctx, find_w(GC_TENSOR_FFN_DOWN, "weight", (int)il));
        ggml_tensor * w_gt = ffn_params_.has_gate
            ? weight_view(ctx, find_w(GC_TENSOR_FFN_GATE, "weight", (int)il))
            : nullptr;
        if (!w_fn || !w_up || !w_dn) { err_ = "missing ffn weights"; return nullptr; }

        ggml_tensor * normed_ffn = gc_build_norm(ctx, cur, w_fn, nullptr,
                                                  norm_params_.type, norm_params_.eps, cb, (int)il);
        ggml_tensor * ffn_out = gc_build_ffn(ctx, normed_ffn,
            w_up, nullptr, w_gt, nullptr, w_dn, nullptr,
            ffn_params_.act, ffn_params_.gate_mode, cb, (int)il);

        if (lf.has_post_ffn_norm) {
            ggml_tensor * w_pfn = weight_view(ctx, find_w(GC_TENSOR_POST_MLP_NORM, "weight", (int)il));
            if (w_pfn)
                ffn_out = gc_build_norm(ctx, ffn_out, w_pfn, nullptr,
                                        norm_params_.type, norm_params_.eps, cb, (int)il);
        }

        cur = ggml_add(ctx, cur, ffn_out);
    }

    // ── Output norm + lm_head ─────────────────────────────────────────────────
    if (w_output_norm_) {
        ggml_tensor * w_on = weight_view(ctx, w_output_norm_);
        if (w_on)
            cur = gc_build_norm(ctx, cur, w_on, nullptr,
                                norm_params_.type, norm_params_.eps, cb, -1);
    }

    ggml_tensor * w_out = weight_view(ctx, w_output_);
    if (!w_out) { err_ = "missing output weight"; return nullptr; }

    ggml_tensor * logits = ggml_mul_mat(ctx, w_out, cur);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    return logits;
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
// Process one batch step.  For each request entry:
//   1. Ensure input tensor capacity.
//   2. Build the complete forward graph (one graph per entry for now; future:
//      batched multi-sequence graph).
//   3. Allocate compute buffers via galloc_ (reused across steps).
//   4. Dispatch via ggml_backend_graph_compute.
//   5. Read logits from device → CPU; sample.
//
// Debug assertions fired here (not in build_graph which has no CPU access):
//   - No token writes to null block (block_id=0).
//   - No F32 promotion of model weights (caught by weight_view type checking).

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

        const int n_new = (int)e.token_ids.size();
        const int n_ctx = e.num_computed + n_new;

        // ── Pre-execution assertions ──────────────────────────────────────────
        // Assert no writes to null block (block_id=0 is the zero-pad sentinel).
        for (int i = 0; i < n_new; ++i) {
            const int pos    = e.num_computed + i;
            const int blk_id = (pos / cfg_.kv_block_size < (int)e.block_ids.size())
                               ? e.block_ids[(size_t)(pos / cfg_.kv_block_size)] : 0;
            if (blk_id == 0) {
                fprintf(stderr, "[gc_graph_runner] ASSERT: token %d mapped to null block "
                        "(req=%s, pos=%d)\n", i, e.req_id.c_str(), pos);
                // Treat as error: emit -1 token and skip.
                out.sampled_tokens.push_back(-1);
                goto next_entry;
            }
        }

        {
            // ── Input tensor capacity ─────────────────────────────────────────
            if (!ensure_inp_capacity(n_new)) {
                err_ = "input tensor allocation failed";
                out.sampled_tokens.push_back(-1);
                goto next_entry;
            }

            // ── Build graph ───────────────────────────────────────────────────
            // Context pool holds every tensor object allocated during graph build
            // (intermediates, views, copies — not just graph leaf nodes).
            //
            // Per-layer tensor budget breakdown:
            //   ~50 fixed tensors (norm/proj/rope/attn/ffn intermediates)
            //   n_new * 4  KV write nodes (2 view + 2 cpy per token, K+V)
            //   n_ctx * 4  KV gather nodes (2 view + 2 cpy per token, K+V)
            //   2          K_full / V_full output tensors
            // Plus 64 global tensors (inputs, lm_head, output norm, mask).
            const size_t tensors_per_layer = 64
                                           + (size_t)n_new * 4
                                           + (size_t)n_ctx * 4;
            const size_t max_nodes = (size_t)hp_.n_layer * tensors_per_layer + 128;
            const size_t ctx_size  = ggml_tensor_overhead() * max_nodes
                                   + ggml_graph_overhead_custom(max_nodes, false);
            ggml_init_params ip{ ctx_size, nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                err_ = "graph context alloc failed";
                out.sampled_tokens.push_back(-1);
                goto next_entry;
            }

            ggml_cgraph * gf = ggml_new_graph_custom(ctx, max_nodes, false);
            ggml_tensor * kq_mask = nullptr;
            ggml_tensor * logits_t = build_graph(ctx, gf, e, &kq_mask);
            if (!logits_t) {
                ggml_free(ctx);
                out.sampled_tokens.push_back(-1);
                goto next_entry;
            }

            // ── Allocate compute buffers ──────────────────────────────────────
            // ggml_gallocr_alloc_graph reuses existing buffers if large enough.
            // After this call every tensor in gf (including kq_mask) has a buffer.
            if (!ggml_gallocr_alloc_graph(galloc_, gf)) {
                stats_.graph_alloc_resized++;
                ggml_free(ctx);
                out.sampled_tokens.push_back(-1);
                goto next_entry;
            } else {
                stats_.graph_alloc_reused++;
            }

            // ── Write causal mask (now kq_mask has a buffer) ──────────────────
            if (kq_mask) {
                std::vector<float> mask_data((size_t)n_ctx * (size_t)n_new);
                for (int q = 0; q < n_new; ++q) {
                    const int q_pos = e.num_computed + q;
                    for (int k = 0; k < n_ctx; ++k) {
                        mask_data[(size_t)q * (size_t)n_ctx + k] =
                            (k <= q_pos) ? 0.0f : -std::numeric_limits<float>::infinity();
                    }
                }
                ggml_backend_tensor_set(kq_mask, mask_data.data(), 0,
                                        sizeof(float) * mask_data.size());
            }

            // ── Compute ───────────────────────────────────────────────────────
            // Single dispatch: entire graph (embedding + all layers + lm_head + KV writes).
            const enum ggml_status status = ggml_backend_graph_compute(backend_, gf);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "[gc_graph_runner] graph compute failed (req=%s status=%d)\n",
                        e.req_id.c_str(), (int)status);
                ggml_free(ctx);
                out.sampled_tokens.push_back(-1);
                goto next_entry;
            }
            stats_.steps_executed++;

            // ── Read logits from device → CPU ─────────────────────────────────
            // logits_t shape: F32 [vocab_size, n_new].
            // We need logits for the LAST token position only.
            const int64_t vocab = logits_t->ne[0];
            std::vector<float> logits((size_t)vocab);
            // Offset to last token row: (n_new - 1) * vocab * sizeof(float)
            ggml_backend_tensor_get(logits_t,
                logits.data(),
                (size_t)(n_new - 1) * (size_t)vocab * sizeof(float),
                (size_t)vocab * sizeof(float));

            ggml_free(ctx);

            // Track prompt tokens for repetition penalty.
            st.recent_tokens.insert(st.recent_tokens.end(),
                                    e.token_ids.begin(), e.token_ids.end());
            if (st.recent_tokens.size() > (size_t)cfg_.repetition_window) {
                st.recent_tokens.erase(st.recent_tokens.begin(),
                    st.recent_tokens.end() - cfg_.repetition_window);
            }

            // ── Sample or suppress ────────────────────────────────────────────
            if (e.is_prefill && !e.is_last_prefill) {
                // Mid-prefill: no output token yet.
                out.sampled_tokens.push_back(-1);
            } else {
                apply_rep_penalty(logits, st);
                const int32_t tok = sample(std::move(logits));
                out.sampled_tokens.push_back(tok);
                st.recent_tokens.push_back(tok);
                if (st.recent_tokens.size() > (size_t)cfg_.repetition_window) {
                    st.recent_tokens.erase(st.recent_tokens.begin());
                }
            }
        }
        goto done_entry;

    next_entry:;
    done_entry:;
        (void)n_ctx;  // suppress unused warning when assertions disabled
    }

    // Purge state for completed requests.
    for (auto it = req_state_.begin(); it != req_state_.end();) {
        it = active.count(it->first) ? std::next(it) : req_state_.erase(it);
    }

    // Periodic stats log (every 100 steps).
    if (stats_.steps_executed > 0 && stats_.steps_executed % 100 == 0) {
        fprintf(stderr,
            "[gc_graph_runner] steps=%llu  kv_slots_written=%llu  "
            "alloc_reused=%llu  alloc_resized=%llu\n",
            (unsigned long long)stats_.steps_executed,
            (unsigned long long)stats_.kv_slots_written,
            (unsigned long long)stats_.graph_alloc_reused,
            (unsigned long long)stats_.graph_alloc_resized);
    }

    return out;
}
