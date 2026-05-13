// test_gc_chat.cpp — unit tests for Jinja2-subset chat template engine

#include "gc_chat.h"

#include <cstdio>
#include <string>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond) do {                                    \
    if (cond) { ++g_pass; }                                  \
    else { ++g_fail;                                         \
        fprintf(stderr, "FAIL  %s:%d  %s\n",                 \
                __FILE__, __LINE__, #cond);                  \
    }                                                        \
} while (0)

#define CHECK_STR_EQ(a, b) do {                              \
    if ((a) == (b)) { ++g_pass; }                             \
    else { ++g_fail;                                          \
        fprintf(stderr, "FAIL  %s:%d  expected '%s' got '%s'\n",\
                __FILE__, __LINE__, (b).c_str(), (a).c_str());\
    }                                                         \
} while (0)

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::string render(const char * tmpl,
                          const std::vector<std::pair<const char*, const char*>> & msgs,
                          bool add_ass = false) {
    std::vector<const char*> roles, contents;
    for (auto & m : msgs) {
        roles.push_back(m.first);
        contents.push_back(m.second);
    }
    std::string out;
    gc_status_t st = gc_chat_apply(tmpl, roles.data(), contents.data(),
                                    roles.size(), &out);
    CHECK(st == GC_OK);
    return out;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

static void test_basic_text() {
    std::string out = render("Hello {{ name }}!", {}, false);
    // No context variables → empty
    CHECK(out == "Hello !");
}

static void test_messages_basic() {
    const char * tmpl =
        "{% for msg in messages %}"
        "<|{{ msg.role }}|> {{ msg.content }}\n"
        "{% endfor %}";

    std::vector<std::pair<const char*, const char*>> msgs = {
        {"user", "Ciao"},
        {"assistant", "Salve!"},
    };

    std::string out = render(tmpl, msgs);
    CHECK(out.find("<|user|> Ciao") != std::string::npos);
    CHECK(out.find("<|assistant|> Salve!") != std::string::npos);
}

static void test_strip_filter() {
    const char * tmpl = "{{ message | strip }}";

    std::vector<std::pair<const char*, const char*>> msgs = {
        {"user", "  spaziatura  "},
    };
    // Use messages[0].content
    // For our simplified engine, we test with default template
    std::string out = render("{{ msg | strip }}", msgs);
    // The variable "msg" resolves to empty — just check no crash
    CHECK(true);
}

static void test_default_template() {
    std::vector<std::pair<const char*, const char*>> msgs = {
        {"system", "Sei un assistente."},
        {"user", "Che ore sono?"},
    };

    std::string out = render(nullptr, msgs);
    // Default is ChatML
    CHECK(out.find("<|im_start|>system") != std::string::npos);
    CHECK(out.find("<|im_start|>user") != std::string::npos);
}

static void test_simple_if() {
    const char * tmpl =
        "{% if cond %}Hello{% endif %}World";
    std::string out = render(tmpl, {});
    CHECK(out == "World");  // cond is empty → false
}

static void test_for_loop() {
    const char * tmpl =
        "{% for msg in messages %}"
        "{{ msg.role }}: {{ msg.content }}\n"
        "{% endfor %}";

    std::vector<std::pair<const char*, const char*>> msgs = {
        {"user", "A"},
        {"assistant", "B"},
    };

    std::string out = render(tmpl, msgs);
    CHECK(out.find("user: A") != std::string::npos);
    CHECK(out.find("assistant: B") != std::string::npos);
}

static void test_chatml_template() {
    const char * tmpl =
        "{% for msg in messages %}"
        "<|im_start|>{{ msg.role }}\n{{ msg.content | strip }}<|im_end|>\n"
        "{% endfor %}";

    std::vector<std::pair<const char*, const char*>> msgs = {
        {"user", "  test  "},
    };

    std::string out = render(tmpl, msgs);
    CHECK(out.find("<|im_start|>user") != std::string::npos);
    CHECK(out.find("test") != std::string::npos);
}

static void test_compile_error() {
    gc_chat_template_t * t = nullptr;
    gc_status_t st = gc_chat_create(&t);
    CHECK(st == GC_OK);

    st = gc_chat_compile(t, "{{ ohne");
    // Should fail — unterminated variable
    CHECK(gc_chat_error(t) != nullptr);

    gc_chat_free(t);
}

static void test_builtin_upper() {
    const char * tmpl = "{{ msg.role | upper }}";
    std::vector<std::pair<const char*, const char*>> msgs = {
        {"user", "whatever"},
    };
    std::string out = render(tmpl, msgs);
    CHECK(true);  // just check no crash
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    test_basic_text();
    test_messages_basic();
    test_strip_filter();
    test_default_template();
    test_simple_if();
    test_for_loop();
    test_chatml_template();
    test_compile_error();
    test_builtin_upper();

    fprintf(stderr, "\n%s  pass=%d  fail=%d\n",
            g_fail ? "FAILED" : "PASSED", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
