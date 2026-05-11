// test_gc_attention.cpp — unit tests for gc_rope + gc_attention graph builders

#include "gc_rope.h"
#include "gc_attention.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
} while (0)

// ── Helpers ───────────────────────────────────────────────────────────────────

static ggml_context * make_ctx(size_t mem_mb = 64) {
    struct ggml_init_params p = {
        /*.mem_size   =*/ mem_mb * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    return ggml_init(p);
}

static void null_cb(ggml_tensor *, const char *, int) {}

// ── RoPE params struct ────────────────────────────────────────────────────────

static void test_rope_params_defaults() {
    gc_rope_params_t p;
    CHECK(p.freq_base   == 10000.0f);
    CHECK(p.freq_scale  == 1.0f);
    CHECK(p.ext_factor  == 0.0f);
    CHECK(p.attn_factor == 1.0f);
    CHECK(p.beta_fast   == 32.0f);
    CHECK(p.beta_slow   == 1.0f);
    CHECK(p.mode        == GGML_ROPE_TYPE_NEOX);
}

// ── RoPE apply ────────────────────────────────────────────────────────────────

static void test_rope_apply_shape() {
    ggml_context * ctx = make_ctx();

    const int64_t n_head   = 4;
    const int64_t n_head_k = 4;
    const int64_t n_rot    = 8;
    const int64_t n_tokens = 16;

    // q: [n_rot*2, n_head, n_tokens] (full head dim = n_rot*2 for simplicity)
    ggml_tensor * q   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_rot*2, n_head,   n_tokens);
    ggml_tensor * k   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_rot*2, n_head_k, n_tokens);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);

    gc_rope_params_t rp;
    rp.n_dims     = (int)n_rot;
    rp.n_ctx_orig = 4096;
    rp.freq_base  = 10000.0f;

    ggml_tensor * q_rope = gc_rope_apply(ctx, q, pos, nullptr, rp);
    ggml_tensor * k_rope = gc_rope_apply(ctx, k, pos, nullptr, rp);

    // shapes must be preserved
    CHECK(q_rope->ne[0] == n_rot*2);
    CHECK(q_rope->ne[1] == n_head);
    CHECK(q_rope->ne[2] == n_tokens);
    CHECK(k_rope->ne[0] == n_rot*2);
    CHECK(k_rope->ne[1] == n_head_k);
    CHECK(k_rope->ne[2] == n_tokens);

    ggml_free(ctx);
}

// ── Norm ─────────────────────────────────────────────────────────────────────

static void test_norm_shapes() {
    ggml_context * ctx = make_ctx();

    const int64_t n_embd   = 64;
    const int64_t n_tokens = 8;

    ggml_tensor * cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * w   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    ggml_tensor * b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

    ggml_tensor * rms  = gc_build_norm(ctx, cur, w, nullptr, GC_NORM_RMS,   1e-5f, null_cb, 0);
    ggml_tensor * ln   = gc_build_norm(ctx, cur, w, b,       GC_NORM_LAYER, 1e-5f, null_cb, 0);

    CHECK(rms->ne[0] == n_embd);
    CHECK(rms->ne[1] == n_tokens);
    CHECK(ln->ne[0]  == n_embd);
    CHECK(ln->ne[1]  == n_tokens);

    ggml_free(ctx);
}

// ── FFN ──────────────────────────────────────────────────────────────────────

static void test_ffn_shapes() {
    ggml_context * ctx = make_ctx();

    const int64_t n_embd   = 32;
    const int64_t n_ff     = 128;
    const int64_t n_tokens = 4;

    ggml_tensor * cur  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * up   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff);
    ggml_tensor * gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff);
    ggml_tensor * down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff,   n_embd);

    // SwiGLU gate-parallel
    ggml_tensor * out = gc_build_ffn(ctx, cur,
        up, nullptr, gate, nullptr, down, nullptr,
        GC_FFN_SILU, GC_FFN_PAR, null_cb, 0);
    CHECK(out->ne[0] == n_embd);
    CHECK(out->ne[1] == n_tokens);

    // plain GELU, no gate
    ggml_tensor * out2 = gc_build_ffn(ctx, cur,
        up, nullptr, nullptr, nullptr, down, nullptr,
        GC_FFN_GELU, GC_FFN_SEQ, null_cb, 0);
    CHECK(out2->ne[0] == n_embd);
    CHECK(out2->ne[1] == n_tokens);

    ggml_free(ctx);
}

// ── MHA shapes ────────────────────────────────────────────────────────────────

static void test_mha_shapes() {
    ggml_context * ctx = make_ctx(128);

    const int64_t n_head    = 4;
    const int64_t n_head_kv = 2;   // GQA: 2 KV heads for 4 Q heads
    const int64_t n_embd_h  = 16;
    const int64_t n_tokens  = 8;

    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_h, n_head,    n_tokens);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_h, n_head_kv, n_tokens);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_h, n_head_kv, n_tokens);

    gc_attn_params_t ap;
    ap.n_embd_head_q = n_embd_h;
    ap.n_embd_head_v = n_embd_h;
    ap.n_head_q      = n_head;
    ap.n_head_kv     = n_head_kv;
    ap.n_tokens      = n_tokens;
    ap.kq_scale      = 1.0f / sqrtf((float)n_embd_h);
    ap.use_flash     = false;

    // build a minimal graph
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * out = gc_build_attn_mha(ctx, gf, q, k, v, nullptr, nullptr, ap, null_cb, 0);

    // output shape: [n_embd_h * n_head, n_tokens]
    CHECK(out->ne[0] == n_embd_h * n_head);
    CHECK(out->ne[1] == n_tokens);

    ggml_free(ctx);
}

// ── Attn params sanity ────────────────────────────────────────────────────────

static void test_attn_params_fields() {
    gc_attn_params_t p;
    p.n_embd_head_q = 128;
    p.n_embd_head_v = 128;
    p.n_head_q      = 32;
    p.n_head_kv     = 8;
    p.n_tokens      = 512;
    p.kq_scale      = 1.0f / sqrtf(128.0f);

    CHECK(p.n_head_q    == 32);
    CHECK(p.n_head_kv   == 8);
    CHECK(p.n_tokens    == 512);
    CHECK(fabsf(p.kq_scale - 1.0f/sqrtf(128.0f)) < 1e-6f);
}

// ── Full gc_build_attn shape ──────────────────────────────────────────────────

static void test_build_attn_shape() {
    ggml_context * ctx = make_ctx(256);

    const int64_t n_embd   = 64;
    const int64_t n_head   = 4;
    const int64_t n_head_kv= 2;
    const int64_t n_embd_h = n_embd / n_head;  // 16
    const int64_t n_tokens = 6;

    // token embeddings
    ggml_tensor * cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);

    // weight matrices
    ggml_tensor * wq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
    ggml_tensor * wk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd_h * n_head_kv);
    ggml_tensor * wv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd_h * n_head_kv);
    ggml_tensor * wo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
    ggml_tensor * pos= ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);

    gc_attn_params_t ap;
    ap.n_embd_head_q = n_embd_h;
    ap.n_embd_head_v = n_embd_h;
    ap.n_head_q      = n_head;
    ap.n_head_kv     = n_head_kv;
    ap.n_tokens      = n_tokens;
    ap.kq_scale      = 1.0f / sqrtf((float)n_embd_h);

    gc_rope_params_t rp;
    rp.n_dims     = (int)(n_embd_h / 2);
    rp.n_ctx_orig = 4096;
    rp.mode       = GGML_ROPE_TYPE_NEOX;

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * out = gc_build_attn(
            ctx, gf,
            cur,
            wq, nullptr, wk, nullptr, wv, nullptr, wo, nullptr,
            pos, nullptr,
            ap, rp, null_cb, 0);

    // output should be [n_embd, n_tokens]
    CHECK(out->ne[0] == n_embd);
    CHECK(out->ne[1] == n_tokens);

    ggml_free(ctx);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_rope_params_defaults();
    test_rope_apply_shape();
    test_norm_shapes();
    test_ffn_shapes();
    test_mha_shapes();
    test_attn_params_fields();
    test_build_attn_shape();

    fprintf(stderr, "\n%s  pass=%d  fail=%d\n",
            g_fail ? "FAILED" : "PASSED", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
