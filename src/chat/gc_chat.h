#pragma once

#include <cstdint>
#include <string>
#include <vector>

// ── Chat message ──────────────────────────────────────────────────────────────

struct gc_chat_message_t {
    const char * role;
    const char * content;
};

// ── Chat template enum ────────────────────────────────────────────────────────

typedef enum gc_chat_template_t {
    GC_CHAT_TEMPLATE_CHATML,
    GC_CHAT_TEMPLATE_LLAMA_2,
    GC_CHAT_TEMPLATE_LLAMA_2_SYS,
    GC_CHAT_TEMPLATE_LLAMA_2_SYS_BOS,
    GC_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP,
    GC_CHAT_TEMPLATE_MISTRAL_V1,
    GC_CHAT_TEMPLATE_MISTRAL_V3,
    GC_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN,
    GC_CHAT_TEMPLATE_MISTRAL_V7,
    GC_CHAT_TEMPLATE_MISTRAL_V7_TEKKEN,
    GC_CHAT_TEMPLATE_PHI_3,
    GC_CHAT_TEMPLATE_PHI_4,
    GC_CHAT_TEMPLATE_FALCON_3,
    GC_CHAT_TEMPLATE_ZEPHYR,
    GC_CHAT_TEMPLATE_MONARCH,
    GC_CHAT_TEMPLATE_GEMMA,
    GC_CHAT_TEMPLATE_ORION,
    GC_CHAT_TEMPLATE_OPENCHAT,
    GC_CHAT_TEMPLATE_VICUNA,
    GC_CHAT_TEMPLATE_VICUNA_ORCA,
    GC_CHAT_TEMPLATE_DEEPSEEK,
    GC_CHAT_TEMPLATE_DEEPSEEK_2,
    GC_CHAT_TEMPLATE_DEEPSEEK_3,
    GC_CHAT_TEMPLATE_DEEPSEEK_OCR,
    GC_CHAT_TEMPLATE_COMMAND_R,
    GC_CHAT_TEMPLATE_LLAMA_3,
    GC_CHAT_TEMPLATE_CHATGLM_3,
    GC_CHAT_TEMPLATE_CHATGLM_4,
    GC_CHAT_TEMPLATE_GLMEDGE,
    GC_CHAT_TEMPLATE_MINICPM,
    GC_CHAT_TEMPLATE_EXAONE_3,
    GC_CHAT_TEMPLATE_EXAONE_4,
    GC_CHAT_TEMPLATE_EXAONE_MOE,
    GC_CHAT_TEMPLATE_RWKV_WORLD,
    GC_CHAT_TEMPLATE_GRANITE_3_X,
    GC_CHAT_TEMPLATE_GRANITE_4_0,
    GC_CHAT_TEMPLATE_GIGACHAT,
    GC_CHAT_TEMPLATE_MEGREZ,
    GC_CHAT_TEMPLATE_YANDEX,
    GC_CHAT_TEMPLATE_BAILING,
    GC_CHAT_TEMPLATE_BAILING_THINK,
    GC_CHAT_TEMPLATE_BAILING2,
    GC_CHAT_TEMPLATE_LLAMA4,
    GC_CHAT_TEMPLATE_SMOLVLM,
    GC_CHAT_TEMPLATE_DOTS1,
    GC_CHAT_TEMPLATE_HUNYUAN_MOE,
    GC_CHAT_TEMPLATE_OPENAI_MOE,
    GC_CHAT_TEMPLATE_HUNYUAN_DENSE,
    GC_CHAT_TEMPLATE_HUNYUAN_OCR,
    GC_CHAT_TEMPLATE_KIMI_K2,
    GC_CHAT_TEMPLATE_SEED_OSS,
    GC_CHAT_TEMPLATE_GROK_2,
    GC_CHAT_TEMPLATE_PANGU_EMBED,
    GC_CHAT_TEMPLATE_SOLAR_OPEN,
    GC_CHAT_TEMPLATE_UNKNOWN,
} gc_chat_template_t;

// ── API ───────────────────────────────────────────────────────────────────────

// Resolve a short name (e.g. "chatml", "llama3") to the enum value.
// Throws std::out_of_range if the name is unknown.
gc_chat_template_t gc_chat_template_from_str(const std::string & name);

// Auto-detect template from a raw Jinja template string.
gc_chat_template_t gc_chat_detect_template(const std::string & tmpl);

// Apply template to a chat history; append generated prompt to `dest`.
// Returns the number of bytes written, or -1 if template is not supported.
int32_t gc_chat_apply_template(
        gc_chat_template_t                            tmpl,
        const std::vector<const gc_chat_message_t *> & chat,
        std::string &                                  dest,
        bool                                           add_ass);

// Return all known short template names (useful for listing / CLI help).
// Writes at most `len` pointers into `output`. Returns total count.
int32_t gc_chat_builtin_templates(const char ** output, size_t len);
