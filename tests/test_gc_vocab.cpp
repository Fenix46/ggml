// test_gc_vocab.cpp — unit tests for gc_vocab_t (no model file needed)
//
// With a GGUF model: ./test_gc_vocab <model.gguf>  (runs load + round-trip tests)
// Without arguments: exercises enums and predicates only.

#include "gc_vocab.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
} while (0)

// ── Enum sanity ───────────────────────────────────────────────────────────────

static void test_vocab_type_enum() {
    CHECK(GC_VOCAB_TYPE_NONE   == 0);
    CHECK(GC_VOCAB_TYPE_SPM    == 1);
    CHECK(GC_VOCAB_TYPE_BPE    == 2);
    CHECK(GC_VOCAB_TYPE_WPM    == 3);
    CHECK(GC_VOCAB_TYPE_UGM    == 4);
    CHECK(GC_VOCAB_TYPE_RWKV   == 5);
    CHECK(GC_VOCAB_TYPE_PLAMO2 == 6);
}

static void test_vocab_pre_enum() {
    CHECK(GC_VOCAB_PRE_DEFAULT     ==  0);
    CHECK(GC_VOCAB_PRE_LLAMA3      ==  1);
    CHECK(GC_VOCAB_PRE_GPT2        ==  7);
    CHECK(GC_VOCAB_PRE_QWEN2       == 11);
    CHECK(GC_VOCAB_PRE_KIMI_K2     == 37);
    CHECK(GC_VOCAB_PRE_SARVAM_MOE  == 51);
}

static void test_token_attr_flags() {
    CHECK(GC_TOKEN_ATTR_UNDEFINED    == 0);
    CHECK(GC_TOKEN_ATTR_UNKNOWN      == (1 << 0));
    CHECK(GC_TOKEN_ATTR_UNUSED       == (1 << 1));
    CHECK(GC_TOKEN_ATTR_NORMAL       == (1 << 2));
    CHECK(GC_TOKEN_ATTR_CONTROL      == (1 << 3));
    CHECK(GC_TOKEN_ATTR_USER_DEFINED == (1 << 4));
    CHECK(GC_TOKEN_ATTR_BYTE         == (1 << 5));
    CHECK(GC_TOKEN_ATTR_NORMALIZED   == (1 << 6));
    CHECK(GC_TOKEN_ATTR_LSTRIP       == (1 << 7));
    CHECK(GC_TOKEN_ATTR_RSTRIP       == (1 << 8));
    CHECK(GC_TOKEN_ATTR_SINGLE_WORD  == (1 << 9));
}

static void test_gc_token_null() {
    CHECK(GC_TOKEN_NULL == -1);
}

// ── Default-constructed vocab ─────────────────────────────────────────────────

static void test_default_vocab() {
    gc_vocab_t v;
    // no tokens until loaded
    CHECK(v.n_tokens() == 0);
    // optional / nullable special tokens
    CHECK(v.token_eot() == GC_TOKEN_NULL);
    CHECK(v.token_eom() == GC_TOKEN_NULL);
    CHECK(v.token_sep() == GC_TOKEN_NULL);
    CHECK(v.token_pad() == GC_TOKEN_NULL);
    CHECK(v.add_bos()   == false);
    CHECK(v.add_eos()   == false);
    CHECK(v.max_token_len() == 0);
    CHECK(v.text_to_token("anything") == GC_TOKEN_NULL);
}

// ── Alias accessors agree with canonical methods ──────────────────────────────

static void test_alias_consistency() {
    gc_vocab_t v;
    CHECK(v.get_type()     == v.type());
    CHECK(v.get_pre_type() == v.pre_type());
    CHECK(v.get_add_bos()  == v.add_bos());
    CHECK(v.get_add_eos()  == v.add_eos());
    CHECK(v.get_add_sep()  == v.add_sep());
    CHECK(v.get_add_space_prefix()           == v.add_space_prefix());
    CHECK(v.get_ignore_merges()              == v.ignore_merges());
    CHECK(v.get_clean_spaces()               == v.clean_spaces());
    CHECK(v.get_remove_extra_whitespaces()   == v.remove_extra_whitespaces());
    CHECK(v.get_escape_whitespaces()         == v.escape_whitespaces());
    CHECK(v.get_treat_whitespace_as_suffix() == v.treat_whitespace_as_suffix());
}

// ── Load + round-trip (optional, requires a GGUF model) ──────────────────────

#ifdef HAVE_LOADER
#include "gc_gguf_loader.h"

static void test_load_and_roundtrip(const char * path) {
    gc_model_loader_t ml;
    gc_status_t st = ml.open(path);
    if (st != GC_STATUS_OK) {
        fprintf(stderr, "SKIP  cannot open %s\n", path);
        return;
    }

    gc_vocab_t v;
    st = v.load(ml);
    CHECK(st == GC_STATUS_OK);

    if (st != GC_STATUS_OK) return;

    CHECK(v.type() != GC_VOCAB_TYPE_NONE);
    CHECK(v.n_tokens() > 0);
    fprintf(stderr, "INFO  vocab type=%s  n_tokens=%u\n", v.type_name(), v.n_tokens());

    // tokenize + detokenize round-trip
    const std::string sample = "Hello, world!";
    auto tokens = v.tokenize(sample, false, false);
    CHECK(!tokens.empty());

    std::string rebuilt = v.detokenize(tokens, false);
    CHECK(!rebuilt.empty());
    fprintf(stderr, "INFO  '%s' → %zu tokens → '%s'\n",
            sample.c_str(), tokens.size(), rebuilt.c_str());

    // token_to_piece for each token
    for (gc_token_t tok : tokens) {
        const std::string & piece = v.token_to_piece(tok);
        CHECK(!piece.empty() || tok == v.token_bos() || tok == v.token_eos());
    }

    // bos / eos special tokens round-trip
    auto toks_special = v.tokenize(sample, true, false);
    CHECK(toks_special.size() >= tokens.size());

    // byte_to_token / token_to_byte consistency for BPE
    if (v.type() == GC_VOCAB_TYPE_BPE || v.type() == GC_VOCAB_TYPE_SPM) {
        for (int b = 0; b < 256; ++b) {
            gc_token_t bt = v.byte_to_token((uint8_t)b);
            if (bt != GC_TOKEN_NULL) {
                CHECK(v.is_byte(bt));
                CHECK(v.token_to_byte(bt) == (uint8_t)b);
            }
        }
    }
}
#endif

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    test_vocab_type_enum();
    test_vocab_pre_enum();
    test_token_attr_flags();
    test_gc_token_null();
    test_default_vocab();
    test_alias_consistency();

#ifdef HAVE_LOADER
    if (argc > 1) {
        test_load_and_roundtrip(argv[1]);
    }
#else
    (void)argc; (void)argv;
#endif

    fprintf(stderr, "\n%s  pass=%d  fail=%d\n",
            g_fail ? "FAILED" : "PASSED", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
