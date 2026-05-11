// test_gc_chat.cpp — unit tests for gc_chat template detection and rendering

#include "gc_chat.h"

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

#define CHECK_CONTAINS(str, sub) do { \
    if ((str).find(sub) != std::string::npos) { ++g_pass; } \
    else { ++g_fail; fprintf(stderr, "FAIL  %s:%d  '%s' not found in '%s'\n", __FILE__, __LINE__, sub, (str).c_str()); } \
} while (0)

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::string apply(gc_chat_template_t tmpl,
                         const std::vector<std::pair<const char*, const char*>> & msgs,
                         bool add_ass = false) {
    std::vector<gc_chat_message_t> storage;
    std::vector<const gc_chat_message_t *> chat;
    for (auto & p : msgs) {
        storage.push_back({p.first, p.second});
    }
    for (auto & m : storage) chat.push_back(&m);
    std::string out;
    gc_chat_apply_template(tmpl, chat, out, add_ass);
    return out;
}

// ── Template name resolution ──────────────────────────────────────────────────

static void test_from_str() {
    CHECK(gc_chat_template_from_str("chatml")  == GC_CHAT_TEMPLATE_CHATML);
    CHECK(gc_chat_template_from_str("llama3")  == GC_CHAT_TEMPLATE_LLAMA_3);
    CHECK(gc_chat_template_from_str("gemma")   == GC_CHAT_TEMPLATE_GEMMA);
    CHECK(gc_chat_template_from_str("llama2")  == GC_CHAT_TEMPLATE_LLAMA_2);
    CHECK(gc_chat_template_from_str("phi3")    == GC_CHAT_TEMPLATE_PHI_3);
    CHECK(gc_chat_template_from_str("phi4")    == GC_CHAT_TEMPLATE_PHI_4);
    CHECK(gc_chat_template_from_str("kimi-k2") == GC_CHAT_TEMPLATE_KIMI_K2);
    CHECK(gc_chat_template_from_str("grok-2")  == GC_CHAT_TEMPLATE_GROK_2);

    bool threw = false;
    try { gc_chat_template_from_str("this-does-not-exist"); }
    catch (const std::out_of_range &) { threw = true; }
    CHECK(threw);
}

// ── Template detection ────────────────────────────────────────────────────────

static void test_detect() {
    CHECK(gc_chat_detect_template("chatml")  == GC_CHAT_TEMPLATE_CHATML);
    CHECK(gc_chat_detect_template("llama3")  == GC_CHAT_TEMPLATE_LLAMA_3);

    // heuristic: <|im_start|> without <|im_sep|> → chatml
    CHECK(gc_chat_detect_template("{% if msg %}<|im_start|>{{ role }}") == GC_CHAT_TEMPLATE_CHATML);

    // heuristic: <start_of_turn> → gemma
    CHECK(gc_chat_detect_template("<start_of_turn>user") == GC_CHAT_TEMPLATE_GEMMA);

    // heuristic: <|start_header_id|> → llama3
    CHECK(gc_chat_detect_template("<|start_header_id|>role<|end_header_id|>") == GC_CHAT_TEMPLATE_LLAMA_3);

    // unknown
    CHECK(gc_chat_detect_template("some_totally_random_jinja_template") == GC_CHAT_TEMPLATE_UNKNOWN);
}

// ── Template rendering ────────────────────────────────────────────────────────

static void test_chatml() {
    auto out = apply(GC_CHAT_TEMPLATE_CHATML, {
        {"system",    "You are helpful."},
        {"user",      "Hello"},
        {"assistant", "Hi"},
    });
    CHECK_CONTAINS(out, "<|im_start|>system\nYou are helpful.<|im_end|>");
    CHECK_CONTAINS(out, "<|im_start|>user\nHello<|im_end|>");
    CHECK_CONTAINS(out, "<|im_start|>assistant\nHi<|im_end|>");

    auto out2 = apply(GC_CHAT_TEMPLATE_CHATML, {{"user", "Hello"}}, true);
    CHECK_CONTAINS(out2, "<|im_start|>assistant\n");
}

static void test_llama3() {
    auto out = apply(GC_CHAT_TEMPLATE_LLAMA_3, {
        {"user",      "Hello"},
        {"assistant", "Hi there"},
    });
    CHECK(out.rfind("<|begin_of_text|>", 0) == 0);
    CHECK_CONTAINS(out, "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>");
    CHECK_CONTAINS(out, "<|start_header_id|>assistant<|end_header_id|>\n\nHi there<|eot_id|>");
}

static void test_gemma() {
    auto out = apply(GC_CHAT_TEMPLATE_GEMMA, {
        {"user",      "Hello"},
        {"assistant", "Hi"},
    });
    CHECK_CONTAINS(out, "<start_of_turn>user\nHello<end_of_turn>");
    CHECK_CONTAINS(out, "<start_of_turn>model\nHi<end_of_turn>");
}

static void test_llama2() {
    auto out = apply(GC_CHAT_TEMPLATE_LLAMA_2_SYS, {
        {"system", "You are helpful."},
        {"user",   "Hello"},
    });
    CHECK_CONTAINS(out, "<<SYS>>\nYou are helpful.\n<</SYS>>");
    CHECK_CONTAINS(out, "[INST]");
    CHECK_CONTAINS(out, "[/INST]");
}

static void test_mistral_v1() {
    auto out = apply(GC_CHAT_TEMPLATE_MISTRAL_V1, {
        {"user",      "Hello"},
        {"assistant", "Hi"},
    });
    CHECK_CONTAINS(out, "[INST]");
    CHECK_CONTAINS(out, "[/INST]");
    CHECK_CONTAINS(out, "</s>");
}

static void test_phi3() {
    auto out = apply(GC_CHAT_TEMPLATE_PHI_3, {
        {"system", "Be helpful."},
        {"user",   "Hello"},
    });
    CHECK_CONTAINS(out, "<|system|>\nBe helpful.<|end|>");
    CHECK_CONTAINS(out, "<|user|>\nHello<|end|>");
}

static void test_deepseek3() {
    auto out = apply(GC_CHAT_TEMPLATE_DEEPSEEK_3, {
        {"user",      "Explain RL"},
        {"assistant", "RL is..."},
    });
    CHECK_CONTAINS(out, "Explain RL");
    CHECK_CONTAINS(out, "RL is...");
}

static void test_unsupported_returns_minus1() {
    std::vector<const gc_chat_message_t *> empty;
    std::string out;
    int32_t ret = gc_chat_apply_template(GC_CHAT_TEMPLATE_UNKNOWN, empty, out, false);
    CHECK(ret == -1);
}

static void test_builtin_templates_list() {
    const char * names[128];
    int32_t n = gc_chat_builtin_templates(names, 128);
    CHECK(n > 40);  // we have 54 entries
    bool found_chatml = false, found_llama3 = false;
    for (int32_t i = 0; i < n; i++) {
        if (std::string(names[i]) == "chatml")  found_chatml = true;
        if (std::string(names[i]) == "llama3")  found_llama3 = true;
    }
    CHECK(found_chatml);
    CHECK(found_llama3);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_from_str();
    test_detect();
    test_chatml();
    test_llama3();
    test_gemma();
    test_llama2();
    test_mistral_v1();
    test_phi3();
    test_deepseek3();
    test_unsupported_returns_minus1();
    test_builtin_templates_list();

    fprintf(stderr, "\n%s  pass=%d  fail=%d\n",
            g_fail ? "FAILED" : "PASSED", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
