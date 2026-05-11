#include "gc_chat.h"

#include <algorithm>
#include <map>
#include <sstream>

#if __cplusplus >= 202000L
    #define GCU8(x) (const char*)(u8##x)
#else
    #define GCU8(x) u8##x
#endif

static std::string gc__chat_trim(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();
    while (start < end && isspace(static_cast<unsigned char>(str[start]))) {
        start += 1;
    }
    while (end > start && isspace(static_cast<unsigned char>(str[end - 1]))) {
        end -= 1;
    }
    return str.substr(start, end - start);
}

static const std::map<std::string, gc_chat_template_t> GC_CHAT_TEMPLATES = {
    { "chatml",            GC_CHAT_TEMPLATE_CHATML            },
    { "llama2",            GC_CHAT_TEMPLATE_LLAMA_2           },
    { "llama2-sys",        GC_CHAT_TEMPLATE_LLAMA_2_SYS       },
    { "llama2-sys-bos",    GC_CHAT_TEMPLATE_LLAMA_2_SYS_BOS   },
    { "llama2-sys-strip",  GC_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP },
    { "mistral-v1",        GC_CHAT_TEMPLATE_MISTRAL_V1        },
    { "mistral-v3",        GC_CHAT_TEMPLATE_MISTRAL_V3        },
    { "mistral-v3-tekken", GC_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN },
    { "mistral-v7",        GC_CHAT_TEMPLATE_MISTRAL_V7        },
    { "mistral-v7-tekken", GC_CHAT_TEMPLATE_MISTRAL_V7_TEKKEN },
    { "phi3",              GC_CHAT_TEMPLATE_PHI_3             },
    { "phi4",              GC_CHAT_TEMPLATE_PHI_4             },
    { "falcon3",           GC_CHAT_TEMPLATE_FALCON_3          },
    { "zephyr",            GC_CHAT_TEMPLATE_ZEPHYR            },
    { "monarch",           GC_CHAT_TEMPLATE_MONARCH           },
    { "gemma",             GC_CHAT_TEMPLATE_GEMMA             },
    { "orion",             GC_CHAT_TEMPLATE_ORION             },
    { "openchat",          GC_CHAT_TEMPLATE_OPENCHAT          },
    { "vicuna",            GC_CHAT_TEMPLATE_VICUNA            },
    { "vicuna-orca",       GC_CHAT_TEMPLATE_VICUNA_ORCA       },
    { "deepseek",          GC_CHAT_TEMPLATE_DEEPSEEK          },
    { "deepseek2",         GC_CHAT_TEMPLATE_DEEPSEEK_2        },
    { "deepseek3",         GC_CHAT_TEMPLATE_DEEPSEEK_3        },
    { "deepseek-ocr",      GC_CHAT_TEMPLATE_DEEPSEEK_OCR      },
    { "command-r",         GC_CHAT_TEMPLATE_COMMAND_R         },
    { "llama3",            GC_CHAT_TEMPLATE_LLAMA_3           },
    { "chatglm3",          GC_CHAT_TEMPLATE_CHATGLM_3         },
    { "chatglm4",          GC_CHAT_TEMPLATE_CHATGLM_4         },
    { "glmedge",           GC_CHAT_TEMPLATE_GLMEDGE           },
    { "minicpm",           GC_CHAT_TEMPLATE_MINICPM           },
    { "exaone3",           GC_CHAT_TEMPLATE_EXAONE_3          },
    { "exaone4",           GC_CHAT_TEMPLATE_EXAONE_4          },
    { "exaone-moe",        GC_CHAT_TEMPLATE_EXAONE_MOE        },
    { "rwkv-world",        GC_CHAT_TEMPLATE_RWKV_WORLD        },
    { "granite",           GC_CHAT_TEMPLATE_GRANITE_3_X       },
    { "granite-4.0",       GC_CHAT_TEMPLATE_GRANITE_4_0       },
    { "gigachat",          GC_CHAT_TEMPLATE_GIGACHAT          },
    { "megrez",            GC_CHAT_TEMPLATE_MEGREZ            },
    { "yandex",            GC_CHAT_TEMPLATE_YANDEX            },
    { "bailing",           GC_CHAT_TEMPLATE_BAILING           },
    { "bailing-think",     GC_CHAT_TEMPLATE_BAILING_THINK     },
    { "bailing2",          GC_CHAT_TEMPLATE_BAILING2          },
    { "llama4",            GC_CHAT_TEMPLATE_LLAMA4            },
    { "smolvlm",           GC_CHAT_TEMPLATE_SMOLVLM           },
    { "hunyuan-moe",       GC_CHAT_TEMPLATE_HUNYUAN_MOE       },
    { "gpt-oss",           GC_CHAT_TEMPLATE_OPENAI_MOE        },
    { "hunyuan-dense",     GC_CHAT_TEMPLATE_HUNYUAN_DENSE     },
    { "hunyuan-ocr",       GC_CHAT_TEMPLATE_HUNYUAN_OCR       },
    { "kimi-k2",           GC_CHAT_TEMPLATE_KIMI_K2           },
    { "seed_oss",          GC_CHAT_TEMPLATE_SEED_OSS          },
    { "grok-2",            GC_CHAT_TEMPLATE_GROK_2            },
    { "pangu-embedded",    GC_CHAT_TEMPLATE_PANGU_EMBED       },
    { "solar-open",        GC_CHAT_TEMPLATE_SOLAR_OPEN        },
};

gc_chat_template_t gc_chat_template_from_str(const std::string & name) {
    return GC_CHAT_TEMPLATES.at(name);
}

gc_chat_template_t gc_chat_detect_template(const std::string & tmpl) {
    try {
        return gc_chat_template_from_str(tmpl);
    } catch (const std::out_of_range &) {
        // not a short name — fall through to heuristic detection
    }

    auto contains = [&tmpl](const char * hay) -> bool {
        return tmpl.find(hay) != std::string::npos;
    };

    if (contains("<|im_start|>")) {
        return contains("<|im_sep|>")
            ? GC_CHAT_TEMPLATE_PHI_4
            : contains("<end_of_utterance>")
                ? GC_CHAT_TEMPLATE_SMOLVLM
                : GC_CHAT_TEMPLATE_CHATML;
    } else if (tmpl.find("mistral") == 0 || contains("[INST]")) {
        if (contains("[SYSTEM_PROMPT]")) {
            return GC_CHAT_TEMPLATE_MISTRAL_V7;
        } else if (contains("' [INST] ' + system_message") || contains("[AVAILABLE_TOOLS]")) {
            if (contains(" [INST]")) {
                return GC_CHAT_TEMPLATE_MISTRAL_V1;
            } else if (contains("\"[INST]\"")) {
                return GC_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN;
            }
            return GC_CHAT_TEMPLATE_MISTRAL_V3;
        } else {
            bool support_system_message  = contains("<<SYS>>");
            bool add_bos_inside_history  = contains("bos_token + '[INST]");
            bool strip_message           = contains("content.strip()");
            if (strip_message)              return GC_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP;
            else if (add_bos_inside_history) return GC_CHAT_TEMPLATE_LLAMA_2_SYS_BOS;
            else if (support_system_message) return GC_CHAT_TEMPLATE_LLAMA_2_SYS;
            else                             return GC_CHAT_TEMPLATE_LLAMA_2;
        }
    } else if (contains("<|assistant|>") && contains("<|end|>")) {
        return GC_CHAT_TEMPLATE_PHI_3;
    } else if (contains("[gMASK]<sop>")) {
        return GC_CHAT_TEMPLATE_CHATGLM_4;
    } else if (contains("<|assistant|>") && contains("<|user|>")) {
        if (contains("<|tool_declare|>")) return GC_CHAT_TEMPLATE_EXAONE_MOE;
        return contains("</s>") ? GC_CHAT_TEMPLATE_FALCON_3 : GC_CHAT_TEMPLATE_GLMEDGE;
    } else if (contains("<|{{ item['role'] }}|>") && contains("<|begin_of_image|>")) {
        return GC_CHAT_TEMPLATE_GLMEDGE;
    } else if (contains("<|user|>") && contains("<|endoftext|>")) {
        return GC_CHAT_TEMPLATE_ZEPHYR;
    } else if (contains("bos_token + message['role']")) {
        return GC_CHAT_TEMPLATE_MONARCH;
    } else if (contains("<start_of_turn>")) {
        return GC_CHAT_TEMPLATE_GEMMA;
    } else if (contains("'\\n\\nAssistant: ' + eos_token")) {
        return GC_CHAT_TEMPLATE_ORION;
    } else if (contains("GPT4 Correct ")) {
        return GC_CHAT_TEMPLATE_OPENCHAT;
    } else if (contains("USER: ") && contains("ASSISTANT: ")) {
        return contains("SYSTEM: ") ? GC_CHAT_TEMPLATE_VICUNA_ORCA : GC_CHAT_TEMPLATE_VICUNA;
    } else if (contains("### Instruction:") && contains("<|EOT|>")) {
        return GC_CHAT_TEMPLATE_DEEPSEEK;
    } else if (contains("<|START_OF_TURN_TOKEN|>") && contains("<|USER_TOKEN|>")) {
        return GC_CHAT_TEMPLATE_COMMAND_R;
    } else if (contains("<|start_header_id|>") && contains("<|end_header_id|>")) {
        return GC_CHAT_TEMPLATE_LLAMA_3;
    } else if (contains("[gMASK]sop")) {
        return GC_CHAT_TEMPLATE_CHATGLM_3;
    } else if (contains(GCU8("<用户>"))) {
        return GC_CHAT_TEMPLATE_MINICPM;
    } else if (contains("'Assistant: ' + message['content'] + eos_token")) {
        return GC_CHAT_TEMPLATE_DEEPSEEK_2;
    } else if (contains(GCU8("<｜Assistant｜>")) && contains(GCU8("<｜User｜>")) && contains(GCU8("<｜end▁of▁sentence｜>"))) {
        return GC_CHAT_TEMPLATE_DEEPSEEK_3;
    } else if (contains("[|system|]") && contains("[|assistant|]") && contains("[|endofturn|]")) {
        return contains("[|tool|]") ? GC_CHAT_TEMPLATE_EXAONE_4 : GC_CHAT_TEMPLATE_EXAONE_3;
    } else if (contains("rwkv-world") || contains("{{- 'User: ' + message['content']|trim + '\\n\\n' -}}")) {
        return GC_CHAT_TEMPLATE_RWKV_WORLD;
    } else if (contains("<|start_of_role|>")) {
        return (contains("<tool_call>") || contains("<tools>")) ? GC_CHAT_TEMPLATE_GRANITE_4_0 : GC_CHAT_TEMPLATE_GRANITE_3_X;
    } else if (contains("message['role'] + additional_special_tokens[0] + message['content'] + additional_special_tokens[1]")) {
        return GC_CHAT_TEMPLATE_GIGACHAT;
    } else if (contains("<|role_start|>")) {
        return GC_CHAT_TEMPLATE_MEGREZ;
    } else if (contains(" Ассистент:")) {
        return GC_CHAT_TEMPLATE_YANDEX;
    } else if (contains("<role>ASSISTANT</role>") && contains("'HUMAN'")) {
        return GC_CHAT_TEMPLATE_BAILING;
    } else if (contains("<role>ASSISTANT</role>") && contains("\"HUMAN\"") && contains("<think>")) {
        return GC_CHAT_TEMPLATE_BAILING_THINK;
    } else if (contains("<role>ASSISTANT</role>") && contains("<role>HUMAN</role>") && contains("<|role_end|>")) {
        return GC_CHAT_TEMPLATE_BAILING2;
    } else if (contains("<|header_start|>") && contains("<|header_end|>")) {
        return GC_CHAT_TEMPLATE_LLAMA4;
    } else if (contains("<|endofuserprompt|>")) {
        return GC_CHAT_TEMPLATE_DOTS1;
    } else if (contains("<|extra_0|>") && contains("<|extra_4|>")) {
        return GC_CHAT_TEMPLATE_HUNYUAN_MOE;
    } else if (contains("<|start|>") && contains("<|channel|>")) {
        return GC_CHAT_TEMPLATE_OPENAI_MOE;
    } else if (contains(GCU8("<｜hy_Assistant｜>")) && contains(GCU8("<｜hy_begin▁of▁sentence｜>"))) {
        return GC_CHAT_TEMPLATE_HUNYUAN_OCR;
    } else if (contains(GCU8("<｜hy_Assistant｜>")) && contains(GCU8("<｜hy_place▁holder▁no▁3｜>"))) {
        return GC_CHAT_TEMPLATE_HUNYUAN_DENSE;
    } else if (contains("<|im_assistant|>assistant<|im_middle|>")) {
        return GC_CHAT_TEMPLATE_KIMI_K2;
    } else if (contains("<seed:bos>")) {
        return GC_CHAT_TEMPLATE_SEED_OSS;
    } else if (contains("'Assistant: '  + message['content'] + '<|separator|>")) {
        return GC_CHAT_TEMPLATE_GROK_2;
    } else if (contains(GCU8("[unused9]系统：[unused10]"))) {
        return GC_CHAT_TEMPLATE_PANGU_EMBED;
    } else if (contains("<|begin|>") && contains("<|end|>") && contains("<|content|>")) {
        return GC_CHAT_TEMPLATE_SOLAR_OPEN;
    }
    return GC_CHAT_TEMPLATE_UNKNOWN;
}

int32_t gc_chat_apply_template(
        gc_chat_template_t                            tmpl,
        const std::vector<const gc_chat_message_t *> & chat,
        std::string &                                  dest,
        bool                                           add_ass) {
    std::stringstream ss;

    if (tmpl == GC_CHAT_TEMPLATE_CHATML) {
        for (auto msg : chat) {
            ss << "<|im_start|>" << msg->role << "\n" << msg->content << "<|im_end|>\n";
        }
        if (add_ass) ss << "<|im_start|>assistant\n";
    } else if (tmpl == GC_CHAT_TEMPLATE_MISTRAL_V7 || tmpl == GC_CHAT_TEMPLATE_MISTRAL_V7_TEKKEN) {
        const char * trailing = (tmpl == GC_CHAT_TEMPLATE_MISTRAL_V7) ? " " : "";
        for (auto msg : chat) {
            std::string role(msg->role), content(msg->content);
            if (role == "system") {
                ss << "[SYSTEM_PROMPT]" << trailing << content << "[/SYSTEM_PROMPT]";
            } else if (role == "user") {
                ss << "[INST]" << trailing << content << "[/INST]";
            } else {
                ss << trailing << content << "</s>";
            }
        }
    } else if (tmpl == GC_CHAT_TEMPLATE_MISTRAL_V1
            || tmpl == GC_CHAT_TEMPLATE_MISTRAL_V3
            || tmpl == GC_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN) {
        std::string leading  = (tmpl == GC_CHAT_TEMPLATE_MISTRAL_V1) ? " " : "";
        std::string trailing = (tmpl == GC_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN) ? "" : " ";
        bool trim_ass = (tmpl == GC_CHAT_TEMPLATE_MISTRAL_V3);
        bool inside = false;
        for (auto msg : chat) {
            if (!inside) { ss << leading << "[INST]" << trailing; inside = true; }
            std::string role(msg->role), content(msg->content);
            if (role == "system") {
                ss << content << "\n\n";
            } else if (role == "user") {
                ss << content << leading << "[/INST]";
            } else {
                ss << trailing << (trim_ass ? gc__chat_trim(content) : content) << "</s>";
                inside = false;
            }
        }
    } else if (tmpl == GC_CHAT_TEMPLATE_LLAMA_2
            || tmpl == GC_CHAT_TEMPLATE_LLAMA_2_SYS
            || tmpl == GC_CHAT_TEMPLATE_LLAMA_2_SYS_BOS
            || tmpl == GC_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP) {
        bool support_sys  = (tmpl != GC_CHAT_TEMPLATE_LLAMA_2);
        bool bos_history  = (tmpl == GC_CHAT_TEMPLATE_LLAMA_2_SYS_BOS);
        bool strip_msg    = (tmpl == GC_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP);
        bool inside = true;
        ss << "[INST] ";
        for (auto msg : chat) {
            std::string content = strip_msg ? gc__chat_trim(msg->content) : std::string(msg->content);
            std::string role(msg->role);
            if (!inside) { ss << (bos_history ? "<s>[INST] " : "[INST] "); inside = true; }
            if (role == "system") {
                ss << (support_sys ? "<<SYS>>\n" + content + "\n<</SYS>>\n\n" : content + "\n");
            } else if (role == "user") {
                ss << content << " [/INST]";
            } else {
                ss << content << "</s>";
                inside = false;
            }
        }
    } else if (tmpl == GC_CHAT_TEMPLATE_PHI_3) {
        for (auto msg : chat) {
            ss << "<|" << msg->role << "|>\n" << msg->content << "<|end|>\n";
        }
        if (add_ass) ss << "<|assistant|>\n";
    } else if (tmpl == GC_CHAT_TEMPLATE_PHI_4) {
        for (auto msg : chat) {
            ss << "<|im_start|>" << msg->role << "<|im_sep|>" << msg->content << "<|im_end|>";
        }
        if (add_ass) ss << "<|im_start|>assistant<|im_sep|>";
    } else if (tmpl == GC_CHAT_TEMPLATE_FALCON_3) {
        for (auto msg : chat) {
            ss << "<|" << msg->role << "|>\n" << msg->content << "\n";
        }
        if (add_ass) ss << "<|assistant|>\n";
    } else if (tmpl == GC_CHAT_TEMPLATE_ZEPHYR) {
        for (auto msg : chat) {
            ss << "<|" << msg->role << "|>\n" << msg->content << "<|endoftext|>\n";
        }
        if (add_ass) ss << "<|assistant|>\n";
    } else if (tmpl == GC_CHAT_TEMPLATE_MONARCH) {
        for (auto msg : chat) {
            std::string bos = (msg == chat.front()) ? "" : "<s>";
            ss << bos << msg->role << "\n" << msg->content << "</s>\n";
        }
        if (add_ass) ss << "<s>assistant\n";
    } else if (tmpl == GC_CHAT_TEMPLATE_GEMMA) {
        std::string sys_prompt;
        for (auto msg : chat) {
            std::string role(msg->role);
            if (role == "system") { sys_prompt += gc__chat_trim(msg->content); continue; }
            role = (role == "assistant") ? "model" : role;
            ss << "<start_of_turn>" << role << "\n";
            if (!sys_prompt.empty() && role != "model") { ss << sys_prompt << "\n\n"; sys_prompt.clear(); }
            ss << gc__chat_trim(msg->content) << "<end_of_turn>\n";
        }
        if (add_ass) ss << "<start_of_turn>model\n";
    } else if (tmpl == GC_CHAT_TEMPLATE_ORION) {
        std::string sys_prompt;
        for (auto msg : chat) {
            std::string role(msg->role);
            if (role == "system") { sys_prompt += msg->content; continue; }
            if (role == "user") {
                ss << "Human: ";
                if (!sys_prompt.empty()) { ss << sys_prompt << "\n\n"; sys_prompt.clear(); }
                ss << msg->content << "\n\nAssistant: </s>";
            } else {
                ss << msg->content << "</s>";
            }
        }
    } else if (tmpl == GC_CHAT_TEMPLATE_OPENCHAT) {
        for (auto msg : chat) {
            std::string role(msg->role);
            if (role == "system") {
                ss << msg->content << "<|end_of_turn|>";
            } else {
                role[0] = (char)toupper((unsigned char)role[0]);
                ss << "GPT4 Correct " << role << ": " << msg->content << "<|end_of_turn|>";
            }
        }
        if (add_ass) ss << "GPT4 Correct Assistant:";
    } else if (tmpl == GC_CHAT_TEMPLATE_VICUNA || tmpl == GC_CHAT_TEMPLATE_VICUNA_ORCA) {
        for (auto msg : chat) {
            std::string role(msg->role);
            if (role == "system") {
                ss << (tmpl == GC_CHAT_TEMPLATE_VICUNA_ORCA ? "SYSTEM: " + std::string(msg->content) + "\n" : std::string(msg->content) + "\n\n");
            } else if (role == "user") {
                ss << "USER: " << msg->content << "\n";
            } else if (role == "assistant") {
                ss << "ASSISTANT: " << msg->content << "</s>\n";
            }
        }
        if (add_ass) ss << "ASSISTANT:";
    } else if (tmpl == GC_CHAT_TEMPLATE_DEEPSEEK) {
        for (auto msg : chat) {
            std::string role(msg->role);
            if (role == "system") {
                ss << msg->content;
            } else if (role == "user") {
                ss << "### Instruction:\n" << msg->content << "\n";
            } else if (role == "assistant") {
                ss << "### Response:\n" << msg->content << "\n<|EOT|>\n";
            }
        }
        if (add_ass) ss << "### Response:\n";
    } else if (tmpl == GC_CHAT_TEMPLATE_COMMAND_R) {
        for (auto msg : chat) {
            std::string role(msg->role);
            if (role == "system") {
                ss << "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>" << gc__chat_trim(msg->content) << "<|END_OF_TURN_TOKEN|>";
            } else if (role == "user") {
                ss << "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>" << gc__chat_trim(msg->content) << "<|END_OF_TURN_TOKEN|>";
            } else if (role == "assistant") {
                ss << "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>" << gc__chat_trim(msg->content) << "<|END_OF_TURN_TOKEN|>";
            }
        }
        if (add_ass) ss << "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>";
    } else if (tmpl == GC_CHAT_TEMPLATE_LLAMA_3) {
        ss << "<|begin_of_text|>";
        for (auto msg : chat) {
            ss << "<|start_header_id|>" << msg->role << "<|end_header_id|>\n\n"
               << gc__chat_trim(msg->content) << "<|eot_id|>";
        }
        if (add_ass) ss << "<|start_header_id|>assistant<|end_header_id|>\n\n";
    } else if (tmpl == GC_CHAT_TEMPLATE_CHATGLM_3) {
        ss << "[gMASK]sop";
        for (auto msg : chat) {
            ss << "<|" << msg->role << "|>\n " << msg->content;
        }
        if (add_ass) ss << "<|assistant|>";
    } else if (tmpl == GC_CHAT_TEMPLATE_CHATGLM_4) {
        ss << "[gMASK]<sop>";
        for (auto msg : chat) {
            ss << "<|" << msg->role << "|>\n" << msg->content;
        }
        if (add_ass) ss << "<|assistant|>\n";
    } else if (tmpl == GC_CHAT_TEMPLATE_GLMEDGE) {
        for (auto msg : chat) {
            ss << "<|" << msg->role << "|>\n" << msg->content;
        }
        if (add_ass) ss << "<|assistant|>";
    } else if (tmpl == GC_CHAT_TEMPLATE_MINICPM) {
        for (auto msg : chat) {
            std::string role(msg->role);
            if (role == "user") {
                ss << GCU8("<用户>") << gc__chat_trim(msg->content) << "<AI>";
            } else {
                ss << gc__chat_trim(msg->content);
            }
        }
    } else if (tmpl == GC_CHAT_TEMPLATE_DEEPSEEK_2) {
        for (auto msg : chat) {
            std::string role(msg->role);
            if (role == "system")         ss << msg->content << "\n\n";
            else if (role == "user")      ss << "User: " << msg->content << "\n\n";
            else if (role == "assistant") ss << "Assistant: " << msg->content << GCU8("<｜end▁of▁sentence｜>");
        }
        if (add_ass) ss << "Assistant:";
    } else if (tmpl == GC_CHAT_TEMPLATE_DEEPSEEK_3) {
        for (auto msg : chat) {
            std::string role(msg->role);
            if (role == "system")         ss << msg->content << "\n\n";
            else if (role == "user")      ss << GCU8("<｜User｜>") << msg->content;
            else if (role == "assistant") ss << GCU8("<｜Assistant｜>") << msg->content << GCU8("<｜end▁of▁sentence｜>");
        }
        if (add_ass) ss << GCU8("<｜Assistant｜>");
    } else if (tmpl == GC_CHAT_TEMPLATE_DEEPSEEK_OCR) {
        for (auto msg : chat) ss << msg->content;
    } else if (tmpl == GC_CHAT_TEMPLATE_EXAONE_3) {
        for (auto msg : chat) {
            std::string role(msg->role);
            if (role == "system")         ss << "[|system|]"    << gc__chat_trim(msg->content) << "[|endofturn|]\n";
            else if (role == "user")      ss << "[|user|]"      << gc__chat_trim(msg->content) << "\n";
            else if (role == "assistant") ss << "[|assistant|]" << gc__chat_trim(msg->content) << "[|endofturn|]\n";
        }
        if (add_ass) ss << "[|assistant|]";
    } else if (tmpl == GC_CHAT_TEMPLATE_EXAONE_4) {
        for (auto msg : chat) {
            std::string role(msg->role);
            if (role == "system")         ss << "[|system|]"    << gc__chat_trim(msg->content) << "[|endofturn|]\n";
            else if (role == "user")      ss << "[|user|]"      << gc__chat_trim(msg->content) << "\n";
            else if (role == "assistant") ss << "[|assistant|]" << gc__chat_trim(msg->content) << "[|endofturn|]\n";
            else if (role == "tool")      ss << "[|tool|]"      << gc__chat_trim(msg->content) << "[|endofturn|]\n";
        }
        if (add_ass) ss << "[|assistant|]";
    } else if (tmpl == GC_CHAT_TEMPLATE_EXAONE_MOE) {
        for (auto msg : chat) {
            std::string role(msg->role);
            if (role == "system")         ss << "<|system|>\n"    << gc__chat_trim(msg->content) << "<|endofturn|>\n";
            else if (role == "user")      ss << "<|user|>\n"      << gc__chat_trim(msg->content) << "<|endofturn|>\n";
            else if (role == "assistant") ss << "<|assistant|>\n" << gc__chat_trim(msg->content) << "<|endofturn|>\n";
            else if (role == "tool")      ss << "<|tool|>\n"      << gc__chat_trim(msg->content) << "<|endofturn|>\n";
        }
        if (add_ass) ss << "<|assistant|>\n";
    } else if (tmpl == GC_CHAT_TEMPLATE_RWKV_WORLD) {
        for (size_t i = 0; i < chat.size(); i++) {
            std::string role(chat[i]->role);
            if (role == "system") {
                ss << "System: " << gc__chat_trim(chat[i]->content) << "\n\n";
            } else if (role == "user") {
                ss << "User: " << gc__chat_trim(chat[i]->content) << "\n\n";
                if (i == chat.size() - 1) ss << "Assistant:";
            } else if (role == "assistant") {
                ss << "Assistant: " << gc__chat_trim(chat[i]->content) << "\n\n";
            }
        }
    } else if (tmpl == GC_CHAT_TEMPLATE_GRANITE_3_X) {
        for (const auto & msg : chat) {
            ss << "<|start_of_role|>" << msg->role << "<|end_of_role|>";
            if (std::string(msg->role) == "assistant_tool_call") ss << "<|tool_call|>";
            ss << msg->content << "<|end_of_text|>\n";
        }
        if (add_ass) ss << "<|start_of_role|>assistant<|end_of_role|>";
    } else if (tmpl == GC_CHAT_TEMPLATE_GRANITE_4_0) {
        for (const auto & msg : chat) {
            if (std::string(msg->role) == "assistant_tool_call") {
                ss << "<|start_of_role|>assistant<|end_of_role|><|tool_call|>";
            } else {
                ss << "<|start_of_role|>" << msg->role << "<|end_of_role|>";
            }
            ss << msg->content << "<|end_of_text|>\n";
        }
        if (add_ass) ss << "<|start_of_role|>assistant<|end_of_role|>";
    } else if (tmpl == GC_CHAT_TEMPLATE_GIGACHAT) {
        bool has_sys = !chat.empty() && std::string(chat[0]->role) == "system";
        ss << (has_sys ? "<s>" + std::string(chat[0]->content) + "<|message_sep|>" : "<s>");
        for (size_t i = has_sys ? 1 : 0; i < chat.size(); i++) {
            std::string role(chat[i]->role);
            if (role == "user") {
                ss << "user<|role_sep|>" << chat[i]->content << "<|message_sep|>"
                   << "available functions<|role_sep|>[]<|message_sep|>";
            } else if (role == "assistant") {
                ss << "assistant<|role_sep|>" << chat[i]->content << "<|message_sep|>";
            }
        }
        if (add_ass) ss << "assistant<|role_sep|>";
    } else if (tmpl == GC_CHAT_TEMPLATE_MEGREZ) {
        for (auto msg : chat) {
            ss << "<|role_start|>" << msg->role << "<|role_end|>" << msg->content << "<|turn_end|>";
        }
        if (add_ass) ss << "<|role_start|>assistant<|role_end|>";
    } else if (tmpl == GC_CHAT_TEMPLATE_YANDEX) {
        for (size_t i = 0; i < chat.size(); i++) {
            std::string role(chat[i]->role);
            if (role == "user")           ss << " Пользователь: " << chat[i]->content << "\n\n";
            else if (role == "assistant") ss << " Ассистент: "    << chat[i]->content << "\n\n";
        }
        if (add_ass) ss << " Ассистент:[SEP]";
    } else if (tmpl == GC_CHAT_TEMPLATE_BAILING || tmpl == GC_CHAT_TEMPLATE_BAILING_THINK) {
        for (auto msg : chat) {
            std::string role(msg->role);
            if (role == "user") role = "HUMAN";
            else std::transform(role.begin(), role.end(), role.begin(), ::toupper);
            ss << "<role>" << role << "</role>" << msg->content;
        }
        if (add_ass) {
            ss << "<role>ASSISTANT</role>";
            if (tmpl == GC_CHAT_TEMPLATE_BAILING_THINK) ss << "<think>";
        }
    } else if (tmpl == GC_CHAT_TEMPLATE_BAILING2) {
        bool has_sys = !chat.empty() && std::string(chat[0]->role) == "system";
        if (!has_sys) ss << "<role>SYSTEM</role>detailed thinking off<|role_end|>";
        for (auto msg : chat) {
            std::string role(msg->role);
            if (role == "user") role = "HUMAN";
            else std::transform(role.begin(), role.end(), role.begin(), ::toupper);
            ss << "<role>" << role << "</role>" << msg->content << "<|role_end|>";
        }
        if (add_ass) ss << "<role>ASSISTANT</role>";
    } else if (tmpl == GC_CHAT_TEMPLATE_LLAMA4) {
        for (auto msg : chat) {
            ss << "<|header_start|>" << msg->role << "<|header_end|>\n\n"
               << gc__chat_trim(msg->content) << "<|eot|>";
        }
        if (add_ass) ss << "<|header_start|>assistant<|header_end|>\n\n";
    } else if (tmpl == GC_CHAT_TEMPLATE_SMOLVLM) {
        ss << "<|im_start|>";
        for (auto msg : chat) {
            std::string role(msg->role);
            if (role == "system")         ss << msg->content << "\n\n";
            else if (role == "user")      ss << "User: " << msg->content << "<end_of_utterance>\n";
            else                          ss << "Assistant: " << msg->content << "<end_of_utterance>\n";
        }
        if (add_ass) ss << "Assistant:";
    } else if (tmpl == GC_CHAT_TEMPLATE_DOTS1) {
        for (auto msg : chat) {
            std::string role(msg->role);
            if (role == "system")         ss << "<|system|>" << msg->content << "<|endofsystem|>";
            else if (role == "user")      ss << "<|userprompt|>" << msg->content << "<|endofuserprompt|>";
            else                          ss << "<|response|>" << msg->content << "<|endofresponse|>";
        }
        if (add_ass) ss << "<|response|>";
    } else if (tmpl == GC_CHAT_TEMPLATE_HUNYUAN_MOE) {
        for (auto msg : chat) {
            std::string role(msg->role);
            if (role == "system")         ss << "<|startoftext|>" << msg->content << "<|extra_4|>";
            else if (role == "assistant") ss << msg->content << "<|eos|>";
            else                          ss << "<|startoftext|>" << msg->content << "<|extra_0|>";
        }
    } else if (tmpl == GC_CHAT_TEMPLATE_OPENAI_MOE) {
        for (auto msg : chat) {
            std::string role(msg->role);
            ss << "<|start|>" << role << "<|message|>" << msg->content;
            ss << (role == "assistant" ? "<|return|>" : "<|end|>");
        }
        if (add_ass) ss << "<|start|>assistant";
    } else if (tmpl == GC_CHAT_TEMPLATE_HUNYUAN_DENSE) {
        for (size_t i = 0; i < chat.size(); i++) {
            std::string role(chat[i]->role);
            if (i == 0 && role == "system") {
                ss << chat[i]->content << GCU8("<｜hy_place▁holder▁no▁3｜>");
            }
            if (role == "assistant")     ss << GCU8("<｜hy_Assistant｜>") << chat[i]->content << GCU8("<｜hy_place▁holder▁no▁2｜>");
            else if (role == "user")     ss << GCU8("<｜hy_User｜>") << chat[i]->content << GCU8("<｜hy_Assistant｜>");
        }
    } else if (tmpl == GC_CHAT_TEMPLATE_HUNYUAN_OCR) {
        ss << GCU8("<｜hy_begin▁of▁sentence｜>");
        for (size_t i = 0; i < chat.size(); i++) {
            std::string role(chat[i]->role);
            if (i == 0 && role == "system") {
                ss << chat[i]->content << GCU8("<｜hy_place▁holder▁no▁3｜>");
                continue;
            }
            if (role == "user")           ss << chat[i]->content << GCU8("<｜hy_User｜>");
            else if (role == "assistant") ss << chat[i]->content << GCU8("<｜hy_Assistant｜>");
        }
    } else if (tmpl == GC_CHAT_TEMPLATE_KIMI_K2) {
        for (auto msg : chat) {
            std::string role(msg->role);
            if (role == "system")         ss << "<|im_system|>system<|im_middle|>";
            else if (role == "user")      ss << "<|im_user|>user<|im_middle|>";
            else if (role == "assistant") ss << "<|im_assistant|>assistant<|im_middle|>";
            else if (role == "tool")      ss << "<|im_system|>tool<|im_middle|>";
            ss << msg->content << "<|im_end|>";
        }
        if (add_ass) ss << "<|im_assistant|>assistant<|im_middle|>";
    } else if (tmpl == GC_CHAT_TEMPLATE_SEED_OSS) {
        for (auto msg : chat) {
            std::string role(msg->role);
            ss << "<seed:bos>" << role << "\n"
               << (role == "assistant" ? gc__chat_trim(msg->content) : std::string(msg->content))
               << "<seed:eos>";
        }
        if (add_ass) ss << "<seed:bos>assistant\n";
    } else if (tmpl == GC_CHAT_TEMPLATE_GROK_2) {
        for (auto msg : chat) {
            std::string role(msg->role);
            if (role == "system")         ss << "System: "    << gc__chat_trim(msg->content) << "<|separator|>\n\n";
            else if (role == "user")      ss << "Human: "     << gc__chat_trim(msg->content) << "<|separator|>\n\n";
            else if (role == "assistant") ss << "Assistant: " << msg->content                << "<|separator|>\n\n";
        }
        if (add_ass) ss << "Assistant:";
    } else if (tmpl == GC_CHAT_TEMPLATE_PANGU_EMBED) {
        for (size_t i = 0; i < chat.size(); ++i) {
            const std::string role(chat[i]->role), content(chat[i]->content);
            if (i == 0 && role != "system") ss << GCU8("[unused9]系统：[unused10]");
            if (role == "system")           ss << GCU8("[unused9]系统：") << content << GCU8("[unused10]");
            else if (role == "user")        ss << GCU8("[unused9]用户：") << content << GCU8("[unused10]");
            else if (role == "assistant")   ss << GCU8("[unused9]助手：") << content << GCU8("[unused10]");
            else if (role == "tool")        ss << GCU8("[unused9]工具：") << content << GCU8("[unused10]");
            else if (role == "function")    ss << GCU8("[unused9]方法：") << content << GCU8("[unused10]");
        }
        if (add_ass) ss << GCU8("[unused9]助手：");
    } else if (tmpl == GC_CHAT_TEMPLATE_SOLAR_OPEN) {
        for (auto msg : chat) {
            ss << "<|begin|>" << msg->role << "<|content|>" << msg->content << "<|end|>";
        }
        if (add_ass) ss << "<|begin|>assistant";
    } else {
        return -1;
    }

    dest = ss.str();
    return (int32_t)dest.size();
}

int32_t gc_chat_builtin_templates(const char ** output, size_t len) {
    auto it = GC_CHAT_TEMPLATES.begin();
    for (size_t i = 0; i < std::min(len, GC_CHAT_TEMPLATES.size()); i++) {
        output[i] = it->first.c_str();
        std::advance(it, 1);
    }
    return (int32_t)GC_CHAT_TEMPLATES.size();
}
