#include "gc_arch.h"
#include "gc_arch_runtime.h"
#include "gc_hparams.h"
#include "gc_gguf_loader.h"
#include "gc_common.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) \
    do { \
        if (cond) { ++g_pass; } \
        else { fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); ++g_fail; } \
    } while (0)

// ── Arch name round-trip ──────────────────────────────────────────────────────
static void test_arch_names() {
    // Every arch returned by gc_arch_all() must round-trip through name→enum
    std::vector<gc_arch_t> all = gc_arch_all();
    CHECK(!all.empty(), "gc_arch_all returns non-empty list");

    int mismatch = 0;
    for (gc_arch_t a : all) {
        const char * name = gc_arch_name(a);
        gc_arch_t    back = gc_arch_from_string(name);
        if (back != a) {
            fprintf(stderr, "  round-trip fail: arch %d  name='%s'  back=%d\n", (int)a, name, (int)back);
            mismatch++;
        }
    }
    CHECK(mismatch == 0, "all arch names round-trip");

    // unknown arch must return GC_ARCH_UNKNOWN
    CHECK(gc_arch_from_string("not_a_real_arch_xyz") == GC_ARCH_UNKNOWN,
          "unknown arch string returns GC_ARCH_UNKNOWN");
}

// ── Capability flags ──────────────────────────────────────────────────────────
static void test_capability_flags() {
    CHECK( gc_arch_is_recurrent(GC_ARCH_MAMBA),   "mamba is recurrent");
    CHECK( gc_arch_is_recurrent(GC_ARCH_RWKV7),   "rwkv7 is recurrent");
    CHECK(!gc_arch_is_recurrent(GC_ARCH_LLAMA),   "llama is not recurrent");

    CHECK( gc_arch_is_encoder(GC_ARCH_BERT),       "bert is encoder");
    CHECK( gc_arch_is_encoder(GC_ARCH_T5ENCODER),  "t5encoder is encoder");
    CHECK(!gc_arch_is_encoder(GC_ARCH_LLAMA),      "llama is not encoder");

    CHECK( gc_arch_is_moe(GC_ARCH_QWEN2MOE),      "qwen2moe is moe");
    CHECK( gc_arch_is_moe(GC_ARCH_DEEPSEEK2),     "deepseek2 is moe");
    CHECK(!gc_arch_is_moe(GC_ARCH_LLAMA),         "llama is not moe");
}

// ── KV key resolution ─────────────────────────────────────────────────────────
static void test_kv_resolve() {
    // arch-independent key
    std::string garch = gc_kv_resolve(GC_ARCH_LLAMA, GC_KV_GENERAL_ARCHITECTURE);
    CHECK(garch == "general.architecture", "general.architecture key correct");

    // arch-prefixed key
    std::string head = gc_kv_resolve(GC_ARCH_LLAMA, GC_KV_ATTN_HEAD_COUNT);
    CHECK(head == "llama.attention.head_count", "llama attn head_count key correct");

    std::string head2 = gc_kv_resolve(GC_ARCH_QWEN2, GC_KV_ATTN_HEAD_COUNT);
    CHECK(head2 == "qwen2.attention.head_count", "qwen2 attn head_count key correct");

    // tokenizer key (global, no arch prefix)
    std::string tok = gc_kv_resolve(GC_ARCH_LLAMA, GC_KV_TOKENIZER_MODEL);
    CHECK(tok == "tokenizer.ggml.model", "tokenizer.ggml.model key correct");
}

// ── Tensor name resolution ────────────────────────────────────────────────────
static void test_tensor_names() {
    // global tensors (no bid)
    std::string oe = gc_tn(GC_ARCH_LLAMA, GC_TENSOR_OUTPUT, "weight");
    CHECK(oe == "output.weight", "output.weight name correct");

    std::string te = gc_tn(GC_ARCH_LLAMA, GC_TENSOR_TOKEN_EMBD, "weight");
    CHECK(te == "token_embd.weight", "token_embd.weight name correct");

    // per-layer tensor
    std::string aq = gc_tn(GC_ARCH_LLAMA, GC_TENSOR_ATTN_Q, "weight", 3);
    CHECK(aq == "blk.3.attn_q.weight", "blk.3.attn_q.weight name correct");

    // per-layer, no suffix
    std::string an = gc_tn(GC_ARCH_LLAMA, GC_TENSOR_ATTN_NORM, nullptr, 7);
    CHECK(an == "blk.7.attn_norm", "blk.7.attn_norm name correct");

    // expert tensor (bid + xid)
    std::string exp = gc_tn(GC_ARCH_LLAMA, GC_TENSOR_FFN_DOWN_EXP, "weight", 2, 5);
    CHECK(exp == "blk.2.ffn_down.5.weight", "blk.2.ffn_down.5.weight name correct");

    // missing bid should throw
    bool threw = false;
    try { gc_tn(GC_ARCH_LLAMA, GC_TENSOR_ATTN_Q, "weight"); }
    catch (const std::exception &) { threw = true; }
    CHECK(threw, "gc_tn throws when bid missing for per-layer role");
}

// ── Tensor info ───────────────────────────────────────────────────────────────
static void test_tensor_info() {
    const gc_tensor_info_t & ti = gc_tensor_info_for(GC_TENSOR_OUTPUT);
    CHECK(ti.layer == GC_TENSOR_LAYER_OUTPUT, "output tensor layer is OUTPUT");
    CHECK(ti.op    == GGML_OP_MUL_MAT,        "output tensor op is MUL_MAT");

    const gc_tensor_info_t & te = gc_tensor_info_for(GC_TENSOR_TOKEN_EMBD);
    CHECK(te.layer == GC_TENSOR_LAYER_INPUT,  "token_embd layer is INPUT");
    CHECK(te.op    == GGML_OP_GET_ROWS,       "token_embd op is GET_ROWS");
}

// ── LLaMA architecture adapter ───────────────────────────────────────────────
static void test_llama_arch_adapter() {
    gc_hparams_t hp;
    hp.arch = GC_ARCH_LLAMA;
    hp.n_embd = 4096;
    hp.n_layer = 2;
    hp.n_head = 32;
    hp.n_head_kv = 8;
    hp.n_ctx_train = 8192;
    hp.rope_freq_base = 500000.0f;
    hp.rope_scale_linear = 8.0f;
    hp.f_attn_logit_softcapping = 0.0f;
    hp.f_max_alibi_bias = 0.0f;

    std::string err;
    CHECK(gc_arch_runtime_validate_hparams(hp, &err), "runtime llama hparams validation passes");

    gc_attn_params_t ap;
    CHECK(gc_arch_runtime_build_attn_params(hp, 0, 128, ap, &err), "runtime llama attn params build passes");
    CHECK(ap.n_head_q == 32, "attn params q heads");
    CHECK(ap.n_head_kv == 8, "attn params kv heads");
    CHECK(ap.n_embd_head_q == 128, "attn params q head dim");
    CHECK(ap.n_embd_head_v == 128, "attn params v head dim");
    CHECK(ap.n_tokens == 128, "attn params token count");
    CHECK(fabsf(ap.kq_scale - (1.0f / sqrtf(128.0f))) < 1e-6f, "attn params default kq scale");

    gc_rope_params_t rp;
    CHECK(gc_arch_runtime_build_rope_params(hp, rp, &err), "runtime llama rope params build passes");
    CHECK(rp.n_dims == 128, "rope dims default to head dim");
    CHECK(rp.mode == GGML_ROPE_TYPE_NEOX, "rope mode is NEOX");
    CHECK(rp.n_ctx_orig == 8192, "rope n_ctx_orig");
    CHECK(fabsf(rp.freq_base - 500000.0f) < 1e-6f, "rope freq_base");
    CHECK(fabsf(rp.freq_scale - 0.125f) < 1e-6f, "rope linear freq scale");

    hp.f_attn_scale = 0.5f;
    CHECK(gc_arch_runtime_build_attn_params(hp, 0, 64, ap, &err), "runtime attn params build with explicit scale");
    CHECK(fabsf(ap.kq_scale - 0.5f) < 1e-6f, "attn params explicit kq scale");
}

static void test_runtime_dispatch_gemma_variants() {
    gc_hparams_t hp;
    hp.arch = GC_ARCH_GEMMA2;
    hp.n_embd = 2304;
    hp.n_layer = 46;
    hp.n_head = 8;
    hp.n_head_kv = 4;
    hp.n_ctx_train = 8192;
    hp.causal_attn = true;

    gc_attn_params_t ap;
    gc_rope_params_t rp;
    std::string err;

    CHECK(gc_arch_runtime_validate_hparams(hp, &err), "runtime supports gemma2 validate");
    CHECK(gc_arch_runtime_build_attn_params(hp, 0, 16, ap, &err), "runtime supports gemma2 attn build");
    CHECK(gc_arch_runtime_build_rope_params(hp, rp, &err), "runtime supports gemma2 rope build");
    CHECK(fabsf(ap.kq_scale - (1.0f / sqrtf(288.0f))) < 1e-6f, "gemma2 27b rule attention scale");

    hp.arch = GC_ARCH_GEMMA3N;
    hp.n_layer = 35;
    CHECK(gc_arch_runtime_validate_hparams(hp, &err), "runtime supports gemma3n validate");
    CHECK(gc_arch_runtime_build_attn_params(hp, 0, 16, ap, &err), "runtime supports gemma3n attn build");
    CHECK(fabsf(ap.kq_scale - 1.0f) < 1e-6f, "gemma3n fixed attention scale = 1.0");

    hp.arch = GC_ARCH_GEMMA_EMBEDDING;
    hp.causal_attn = false;
    hp.n_layer = 24;
    CHECK(gc_arch_runtime_validate_hparams(hp, &err), "runtime supports gemma embedding validate");
    CHECK(gc_arch_runtime_build_attn_params(hp, 0, 16, ap, &err), "runtime supports gemma embedding attn build");

    hp.causal_attn = true;
    CHECK(!gc_arch_runtime_validate_hparams(hp, &err), "gemma embedding rejects causal attention");
}

static void test_runtime_dispatch_still_unsupported() {
    gc_hparams_t hp;
    hp.arch = GC_ARCH_QWEN2;
    hp.n_embd = 1536;
    hp.n_layer = 28;
    hp.n_head = 12;
    hp.n_head_kv = 2;

    gc_attn_params_t ap;
    gc_rope_params_t rp;
    std::string err;

    CHECK(!gc_arch_runtime_validate_hparams(hp, &err), "runtime still rejects non-implemented arch");
    CHECK(!gc_arch_runtime_build_attn_params(hp, 0, 16, ap, &err), "runtime still rejects non-implemented attn");
    CHECK(!gc_arch_runtime_build_rope_params(hp, rp, &err), "runtime still rejects non-implemented rope");
}

// ── Hparams from file (optional) ──────────────────────────────────────────────
static void test_hparams_from_file(const char * path) {
    gc_loader_params_t params;
    params.use_mmap = true;

    gc_model_loader_t loader(path, params);
    gc_hparams_t hp;
    bool ok = gc_hparams_load(hp, loader);
    CHECK(ok,                        "gc_hparams_load returns true");
    CHECK(hp.arch != GC_ARCH_UNKNOWN,"arch resolved from file");
    CHECK(hp.n_embd > 0,             "n_embd > 0");
    CHECK(hp.n_layer > 0,            "n_layer > 0");
    CHECK(hp.n_head > 0,             "n_head > 0");

    if (hp.arch == GC_ARCH_LLAMA || hp.arch == GC_ARCH_LLAMA_EMBED ||
        hp.arch == GC_ARCH_GEMMA || hp.arch == GC_ARCH_GEMMA2 || hp.arch == GC_ARCH_GEMMA3 ||
        hp.arch == GC_ARCH_GEMMA3N || hp.arch == GC_ARCH_GEMMA4 || hp.arch == GC_ARCH_GEMMA_EMBEDDING) {
        std::string err;
        gc_attn_params_t ap;
        gc_rope_params_t rp;
        CHECK(gc_arch_runtime_validate_hparams(hp, &err), "runtime arch file hparams valid");
        CHECK(gc_arch_runtime_build_attn_params(hp, 0, 16, ap, &err), "runtime arch file attn params build");
        CHECK(gc_arch_runtime_build_rope_params(hp, rp, &err), "runtime arch file rope params build");
        CHECK(ap.n_embd_head_q > 0, "arch file attn head dim > 0");
        CHECK(rp.n_dims > 0, "arch file rope dims > 0");
    }

    fprintf(stdout,
        "  arch=%s  n_vocab=%u  n_embd=%u  n_layer=%u  n_head=%u  n_head_kv=%u  n_ff=%u\n",
        gc_arch_name(hp.arch), hp.n_vocab, hp.n_embd, hp.n_layer,
        hp.n_head, hp.n_head_kv, hp.n_ff);
}

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    gc_log_set_callback([](ggml_log_level, const char * t, void *) {
        fputs(t, stderr);
    }, nullptr);

    test_arch_names();
    test_capability_flags();
    test_kv_resolve();
    test_tensor_names();
    test_tensor_info();
    test_llama_arch_adapter();
    test_runtime_dispatch_gemma_variants();
    test_runtime_dispatch_still_unsupported();

    if (argc >= 2) {
        fprintf(stdout, "\n[hparams from file: %s]\n", argv[1]);
        try {
            test_hparams_from_file(argv[1]);
        } catch (const std::exception & e) {
            fprintf(stderr, "  exception: %s\n", e.what());
            ++g_fail;
        }
    }

    fprintf(stdout, "\n%s  pass=%d  fail=%d\n",
            g_fail == 0 ? "PASS" : "FAIL", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
