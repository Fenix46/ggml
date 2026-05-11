#include "gc_arch.h"

#include "../loader/gc_common.h"

#include <cassert>
#include <map>
#include <stdexcept>

// ── Architecture name table ───────────────────────────────────────────────────

static const std::map<gc_arch_t, const char *> GC_ARCH_NAMES = {
    { GC_ARCH_UNKNOWN,          "(unknown)"        },
    { GC_ARCH_CLIP,             "clip"             },
    { GC_ARCH_LLAMA,            "llama"            },
    { GC_ARCH_LLAMA4,           "llama4"           },
    { GC_ARCH_DECI,             "deci"             },
    { GC_ARCH_FALCON,           "falcon"           },
    { GC_ARCH_BAICHUAN,         "baichuan"         },
    { GC_ARCH_GROK,             "grok"             },
    { GC_ARCH_GPT2,             "gpt2"             },
    { GC_ARCH_GPTJ,             "gptj"             },
    { GC_ARCH_GPTNEOX,          "gptneox"          },
    { GC_ARCH_MPT,              "mpt"              },
    { GC_ARCH_STARCODER,        "starcoder"        },
    { GC_ARCH_STARCODER2,       "starcoder2"       },
    { GC_ARCH_REFACT,           "refact"           },
    { GC_ARCH_BLOOM,            "bloom"            },
    { GC_ARCH_STABLELM,         "stablelm"         },
    { GC_ARCH_QWEN,             "qwen"             },
    { GC_ARCH_QWEN2,            "qwen2"            },
    { GC_ARCH_QWEN2MOE,         "qwen2moe"         },
    { GC_ARCH_QWEN2VL,          "qwen2vl"          },
    { GC_ARCH_QWEN3,            "qwen3"            },
    { GC_ARCH_QWEN3MOE,         "qwen3moe"         },
    { GC_ARCH_QWEN3NEXT,        "qwen3next"        },
    { GC_ARCH_QWEN3VL,          "qwen3vl"          },
    { GC_ARCH_QWEN3VLMOE,       "qwen3vlmoe"       },
    { GC_ARCH_QWEN35,           "qwen35"           },
    { GC_ARCH_QWEN35MOE,        "qwen35moe"        },
    { GC_ARCH_PHI2,             "phi2"             },
    { GC_ARCH_PHI3,             "phi3"             },
    { GC_ARCH_PHIMOE,           "phimoe"           },
    { GC_ARCH_PLAMO,            "plamo"            },
    { GC_ARCH_PLAMO2,           "plamo2"           },
    { GC_ARCH_PLAMO3,           "plamo3"           },
    { GC_ARCH_CODESHELL,        "codeshell"        },
    { GC_ARCH_ORION,            "orion"            },
    { GC_ARCH_INTERNLM2,        "internlm2"        },
    { GC_ARCH_MINICPM,          "minicpm"          },
    { GC_ARCH_MINICPM3,         "minicpm3"         },
    { GC_ARCH_GEMMA,            "gemma"            },
    { GC_ARCH_GEMMA2,           "gemma2"           },
    { GC_ARCH_GEMMA3,           "gemma3"           },
    { GC_ARCH_GEMMA3N,          "gemma3n"          },
    { GC_ARCH_GEMMA4,           "gemma4"           },
    { GC_ARCH_GEMMA_EMBEDDING,  "gemma-embedding"  },
    { GC_ARCH_MAMBA,            "mamba"            },
    { GC_ARCH_MAMBA2,           "mamba2"           },
    { GC_ARCH_JAMBA,            "jamba"            },
    { GC_ARCH_FALCON_H1,        "falcon-h1"        },
    { GC_ARCH_XVERSE,           "xverse"           },
    { GC_ARCH_COMMAND_R,        "command-r"        },
    { GC_ARCH_COHERE2,          "cohere2"          },
    { GC_ARCH_DBRX,             "dbrx"             },
    { GC_ARCH_OLMO,             "olmo"             },
    { GC_ARCH_OLMO2,            "olmo2"            },
    { GC_ARCH_OLMOE,            "olmoe"            },
    { GC_ARCH_OPENELM,          "openelm"          },
    { GC_ARCH_ARCTIC,           "arctic"           },
    { GC_ARCH_DEEPSEEK,         "deepseek"         },
    { GC_ARCH_DEEPSEEK2,        "deepseek2"        },
    { GC_ARCH_DEEPSEEK2OCR,     "deepseek2-ocr"    },
    { GC_ARCH_CHATGLM,          "chatglm"          },
    { GC_ARCH_GLM4,             "glm4"             },
    { GC_ARCH_GLM4_MOE,         "glm4moe"          },
    { GC_ARCH_GLM_DSA,          "glm-dsa"          },
    { GC_ARCH_BITNET,           "bitnet"           },
    { GC_ARCH_T5,               "t5"               },
    { GC_ARCH_T5ENCODER,        "t5encoder"        },
    { GC_ARCH_JAIS,             "jais"             },
    { GC_ARCH_JAIS2,            "jais2"            },
    { GC_ARCH_NEMOTRON,         "nemotron"         },
    { GC_ARCH_NEMOTRON_H,       "nemotron_h"       },
    { GC_ARCH_NEMOTRON_H_MOE,   "nemotron_h_moe"   },
    { GC_ARCH_EXAONE,           "exaone"           },
    { GC_ARCH_EXAONE4,          "exaone4"          },
    { GC_ARCH_EXAONE_MOE,       "exaone-moe"       },
    { GC_ARCH_RWKV6,            "rwkv6"            },
    { GC_ARCH_RWKV6QWEN2,       "rwkv6qwen2"       },
    { GC_ARCH_RWKV7,            "rwkv7"            },
    { GC_ARCH_ARWKV7,           "arwkv7"           },
    { GC_ARCH_GRANITE,          "granite"          },
    { GC_ARCH_GRANITE_MOE,      "granitemoe"       },
    { GC_ARCH_GRANITE_HYBRID,   "granitehybrid"    },
    { GC_ARCH_CHAMELEON,        "chameleon"        },
    { GC_ARCH_WAVTOKENIZER_DEC, "wavtokenizer-dec" },
    { GC_ARCH_PLM,              "plm"              },
    { GC_ARCH_BAILINGMOE,       "bailingmoe"       },
    { GC_ARCH_BAILINGMOE2,      "bailingmoe2"      },
    { GC_ARCH_DOTS1,            "dots1"            },
    { GC_ARCH_ARCEE,            "arcee"            },
    { GC_ARCH_AFMOE,            "afmoe"            },
    { GC_ARCH_ERNIE4_5,         "ernie4_5"         },
    { GC_ARCH_ERNIE4_5_MOE,     "ernie4_5-moe"     },
    { GC_ARCH_HUNYUAN_MOE,      "hunyuan-moe"      },
    { GC_ARCH_HUNYUAN_DENSE,    "hunyuan-dense"    },
    { GC_ARCH_HUNYUAN_VL,       "hunyuan_vl"       },
    { GC_ARCH_SMOLLM3,          "smollm3"          },
    { GC_ARCH_OPENAI_MOE,       "gpt-oss"          },
    { GC_ARCH_LFM2,             "lfm2"             },
    { GC_ARCH_LFM2MOE,          "lfm2moe"          },
    { GC_ARCH_DREAM,            "dream"            },
    { GC_ARCH_SMALLTHINKER,     "smallthinker"     },
    { GC_ARCH_LLADA,            "llada"            },
    { GC_ARCH_LLADA_MOE,        "llada-moe"        },
    { GC_ARCH_SEED_OSS,         "seed_oss"         },
    { GC_ARCH_GROVEMOE,         "grovemoe"         },
    { GC_ARCH_APERTUS,          "apertus"          },
    { GC_ARCH_MINIMAX_M2,       "minimax-m2"       },
    { GC_ARCH_COGVLM,           "cogvlm"           },
    { GC_ARCH_RND1,             "rnd1"             },
    { GC_ARCH_PANGU_EMBED,      "pangu-embedded"   },
    { GC_ARCH_MISTRAL3,         "mistral3"         },
    { GC_ARCH_MISTRAL4,         "mistral4"         },
    { GC_ARCH_PADDLEOCR,        "paddleocr"        },
    { GC_ARCH_MIMO2,            "mimo2"            },
    { GC_ARCH_STEP35,           "step35"           },
    { GC_ARCH_LLAMA_EMBED,      "llama-embed"      },
    { GC_ARCH_MAINCODER,        "maincoder"        },
    { GC_ARCH_KIMI_LINEAR,      "kimi-linear"      },
    { GC_ARCH_BERT,             "bert"             },
    { GC_ARCH_MODERN_BERT,      "modern-bert"      },
    { GC_ARCH_NOMIC_BERT,       "nomic-bert"       },
    { GC_ARCH_NOMIC_BERT_MOE,   "nomic-bert-moe"   },
    { GC_ARCH_NEO_BERT,         "neo-bert"         },
    { GC_ARCH_JINA_BERT_V2,     "jina-bert-v2"     },
    { GC_ARCH_JINA_BERT_V3,     "jina-bert-v3"     },
    { GC_ARCH_EUROBERT,         "eurobert"         },
};

const char * gc_arch_name(gc_arch_t arch) {
    auto it = GC_ARCH_NAMES.find(arch);
    return it != GC_ARCH_NAMES.end() ? it->second : "(unknown)";
}

gc_arch_t gc_arch_from_string(const std::string & name) {
    for (auto & kv : GC_ARCH_NAMES) {
        if (name == kv.second) return kv.first;
    }
    return GC_ARCH_UNKNOWN;
}

std::vector<gc_arch_t> gc_arch_all() {
    std::vector<gc_arch_t> out;
    out.reserve(GC_ARCH_NAMES.size());
    for (auto & kv : GC_ARCH_NAMES) {
        if (kv.first != GC_ARCH_UNKNOWN) out.push_back(kv.first);
    }
    return out;
}

// ── Architecture capability flags ────────────────────────────────────────────

bool gc_arch_is_recurrent(gc_arch_t arch) {
    switch (arch) {
        case GC_ARCH_MAMBA:
        case GC_ARCH_MAMBA2:
        case GC_ARCH_RWKV6:
        case GC_ARCH_RWKV6QWEN2:
        case GC_ARCH_RWKV7:
        case GC_ARCH_ARWKV7:
            return true;
        default:
            return false;
    }
}

bool gc_arch_is_encoder(gc_arch_t arch) {
    switch (arch) {
        case GC_ARCH_BERT:
        case GC_ARCH_MODERN_BERT:
        case GC_ARCH_NOMIC_BERT:
        case GC_ARCH_NOMIC_BERT_MOE:
        case GC_ARCH_NEO_BERT:
        case GC_ARCH_JINA_BERT_V2:
        case GC_ARCH_JINA_BERT_V3:
        case GC_ARCH_EUROBERT:
        case GC_ARCH_T5ENCODER:
        case GC_ARCH_GEMMA_EMBEDDING:
        case GC_ARCH_LLAMA_EMBED:
        case GC_ARCH_PANGU_EMBED:
            return true;
        default:
            return false;
    }
}

bool gc_arch_is_moe(gc_arch_t arch) {
    switch (arch) {
        case GC_ARCH_QWEN2MOE:
        case GC_ARCH_QWEN3MOE:
        case GC_ARCH_QWEN35MOE:
        case GC_ARCH_PHIMOE:
        case GC_ARCH_DBRX:
        case GC_ARCH_OLMOE:
        case GC_ARCH_DEEPSEEK2:
        case GC_ARCH_DEEPSEEK2OCR:
        case GC_ARCH_GRANITE_MOE:
        case GC_ARCH_BAILINGMOE:
        case GC_ARCH_BAILINGMOE2:
        case GC_ARCH_ERNIE4_5_MOE:
        case GC_ARCH_HUNYUAN_MOE:
        case GC_ARCH_LFM2MOE:
        case GC_ARCH_LLADA_MOE:
        case GC_ARCH_GROVEMOE:
        case GC_ARCH_OPENAI_MOE:
        case GC_ARCH_GLM4_MOE:
        case GC_ARCH_NOMIC_BERT_MOE:
        case GC_ARCH_NEMOTRON_H_MOE:
        case GC_ARCH_EXAONE_MOE:
        case GC_ARCH_AFMOE:
        case GC_ARCH_JAMBA:
            return true;
        default:
            return false;
    }
}

// ── KV key tables ─────────────────────────────────────────────────────────────
// Keys with "%s" have the arch name substituted.

static const std::map<gc_kv_key_t, const char *> GC_KV_TEMPLATES = {
    // general (no arch prefix)
    { GC_KV_GENERAL_ARCHITECTURE,          "general.architecture"                  },
    { GC_KV_GENERAL_NAME,                  "general.name"                          },
    { GC_KV_GENERAL_FILE_TYPE,             "general.file_type"                     },
    { GC_KV_GENERAL_QUANTIZATION_VERSION,  "general.quantization_version"          },

    // arch-prefixed
    { GC_KV_VOCAB_SIZE,                    "%s.vocab_size"                         },
    { GC_KV_CONTEXT_LENGTH,                "%s.context_length"                     },
    { GC_KV_EMBEDDING_LENGTH,              "%s.embedding_length"                   },
    { GC_KV_BLOCK_COUNT,                   "%s.block_count"                        },
    { GC_KV_FEED_FORWARD_LENGTH,           "%s.feed_forward_length"                },
    { GC_KV_EXPERT_FEED_FORWARD_LENGTH,    "%s.expert_feed_forward_length"         },
    { GC_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, "%s.expert_shared_feed_forward_length" },
    { GC_KV_EXPERT_COUNT,                  "%s.expert_count"                       },
    { GC_KV_EXPERT_USED_COUNT,             "%s.expert_used_count"                  },
    { GC_KV_EXPERT_SHARED_COUNT,           "%s.expert_shared_count"                },
    { GC_KV_EXPERT_GATING_FUNC,            "%s.expert_gating_func"                 },
    { GC_KV_MOE_EVERY_N_LAYERS,            "%s.moe_every_n_layers"                 },
    { GC_KV_LEADING_DENSE_BLOCK_COUNT,     "%s.leading_dense_block_count"          },
    { GC_KV_USE_PARALLEL_RESIDUAL,         "%s.use_parallel_residual"              },
    { GC_KV_LOGIT_SCALE,                   "%s.logit_scale"                        },
    { GC_KV_RESIDUAL_SCALE,                "%s.residual_scale"                     },
    { GC_KV_EMBEDDING_SCALE,               "%s.embedding_scale"                    },
    { GC_KV_ATTN_LOGIT_SOFTCAPPING,        "%s.attn_logit_softcapping"             },
    { GC_KV_FINAL_LOGIT_SOFTCAPPING,       "%s.final_logit_softcapping"            },

    // attention
    { GC_KV_ATTN_HEAD_COUNT,               "%s.attention.head_count"               },
    { GC_KV_ATTN_HEAD_COUNT_KV,            "%s.attention.head_count_kv"            },
    { GC_KV_ATTN_KEY_LENGTH,               "%s.attention.key_length"               },
    { GC_KV_ATTN_VALUE_LENGTH,             "%s.attention.value_length"             },
    { GC_KV_ATTN_LAYERNORM_EPS,            "%s.attention.layer_norm_epsilon"       },
    { GC_KV_ATTN_LAYERNORM_RMS_EPS,        "%s.attention.layer_norm_rms_epsilon"   },
    { GC_KV_ATTN_CAUSAL,                   "%s.attention.causal"                   },
    { GC_KV_ATTN_SLIDING_WINDOW,           "%s.attention.sliding_window"           },
    { GC_KV_ATTN_SCALE,                    "%s.attention.scale"                    },
    { GC_KV_ATTN_Q_LORA_RANK,              "%s.attention.q_lora_rank"              },
    { GC_KV_ATTN_KV_LORA_RANK,             "%s.attention.kv_lora_rank"             },
    { GC_KV_ATTN_KEY_LENGTH_MLA,           "%s.attention.key_length_mla"           },
    { GC_KV_ATTN_VALUE_LENGTH_MLA,         "%s.attention.value_length_mla"         },
    { GC_KV_ATTN_CLAMP_KQV,               "%s.attention.clamp_kqv"               },
    { GC_KV_ATTN_MAX_ALIBI_BIAS,           "%s.attention.max_alibi_bias"           },

    // RoPE
    { GC_KV_ROPE_DIMENSION_COUNT,          "%s.rope.dimension_count"               },
    { GC_KV_ROPE_FREQ_BASE,                "%s.rope.freq_base"                     },
    { GC_KV_ROPE_SCALE_LINEAR,             "%s.rope.scale_linear"                  },
    { GC_KV_ROPE_SCALING_TYPE,             "%s.rope.scaling.type"                  },
    { GC_KV_ROPE_SCALING_FACTOR,           "%s.rope.scaling.factor"                },
    { GC_KV_ROPE_SCALING_ORIG_CTX_LEN,     "%s.rope.scaling.original_context_length" },
    { GC_KV_ROPE_SCALING_FINETUNED,        "%s.rope.scaling.finetuned"             },
    { GC_KV_ROPE_SCALING_YARN_LOG_MUL,     "%s.rope.scaling.yarn_log_multiplier"   },
    { GC_KV_ROPE_SCALING_YARN_EXT_FACTOR,  "%s.rope.scaling.yarn_ext_factor"       },
    { GC_KV_ROPE_SCALING_YARN_ATTN_FACTOR, "%s.rope.scaling.yarn_attn_factor"      },
    { GC_KV_ROPE_SCALING_YARN_BETA_FAST,   "%s.rope.scaling.yarn_beta_fast"        },
    { GC_KV_ROPE_SCALING_YARN_BETA_SLOW,   "%s.rope.scaling.yarn_beta_slow"        },
    { GC_KV_ROPE_DIMENSION_SECTIONS,       "%s.rope.dimension_sections"            },

    // SSM / Mamba
    { GC_KV_SSM_CONV_KERNEL,               "%s.ssm.conv_kernel"                    },
    { GC_KV_SSM_INNER_SIZE,                "%s.ssm.inner_size"                     },
    { GC_KV_SSM_STATE_SIZE,                "%s.ssm.state_size"                     },
    { GC_KV_SSM_TIME_STEP_RANK,            "%s.ssm.time_step_rank"                 },
    { GC_KV_SSM_GROUP_COUNT,               "%s.ssm.group_count"                    },

    // LFM2
    { GC_KV_SHORTCONV_L_CACHE,             "%s.shortconv.l_cache"                  },

    // tokenizer (global keys)
    { GC_KV_TOKENIZER_MODEL,               "tokenizer.ggml.model"                  },
    { GC_KV_TOKENIZER_PRE,                 "tokenizer.ggml.pre"                    },
    { GC_KV_TOKENIZER_LIST,                "tokenizer.ggml.tokens"                 },
    { GC_KV_TOKENIZER_TOKEN_TYPE,          "tokenizer.ggml.token_type"             },
    { GC_KV_TOKENIZER_SCORES,              "tokenizer.ggml.scores"                 },
    { GC_KV_TOKENIZER_MERGES,              "tokenizer.ggml.merges"                 },
    { GC_KV_TOKENIZER_BOS_ID,              "tokenizer.ggml.bos_token_id"           },
    { GC_KV_TOKENIZER_EOS_ID,              "tokenizer.ggml.eos_token_id"           },
    { GC_KV_TOKENIZER_EOT_ID,              "tokenizer.ggml.eot_token_id"           },
    { GC_KV_TOKENIZER_UNK_ID,              "tokenizer.ggml.unknown_token_id"       },
    { GC_KV_TOKENIZER_PAD_ID,              "tokenizer.ggml.padding_token_id"       },
    { GC_KV_TOKENIZER_ADD_BOS,             "tokenizer.ggml.add_bos_token"          },
    { GC_KV_TOKENIZER_ADD_EOS,             "tokenizer.ggml.add_eos_token"          },
    { GC_KV_TOKENIZER_CHAT_TEMPLATE,       "tokenizer.chat_template"               },
    { GC_KV_TOKENIZER_HF_JSON,             "tokenizer.huggingface.json"            },

    // shard
    { GC_KV_SPLIT_NO,            "split.no"            },
    { GC_KV_SPLIT_COUNT,         "split.count"         },
    { GC_KV_SPLIT_TENSORS_COUNT, "split.tensors.count" },
};

std::string gc_kv_resolve(gc_arch_t arch, gc_kv_key_t key) {
    auto it = GC_KV_TEMPLATES.find(key);
    if (it == GC_KV_TEMPLATES.end()) {
        throw std::runtime_error(gc__format("gc_kv_resolve: unknown key %d", (int)key));
    }
    const char * tmpl = it->second;
    // substitute %s → arch name if present
    if (tmpl[0] == '%' && tmpl[1] == 's') {
        return gc__format(tmpl, gc_arch_name(arch));
    }
    for (const char * p = tmpl; *p; ++p) {
        if (*p == '%' && *(p+1) == 's') {
            return gc__format(tmpl, gc_arch_name(arch));
        }
    }
    return tmpl;
}

// ── Tensor name table ─────────────────────────────────────────────────────────
// Templates use printf-style %d for block/expert indices.

static const std::map<gc_tensor_role_t, const char *> GC_TENSOR_NAMES = {
    { GC_TENSOR_TOKEN_EMBD,          "token_embd"                   },
    { GC_TENSOR_TOKEN_EMBD_NORM,     "token_embd_norm"              },
    { GC_TENSOR_POS_EMBD,            "position_embd"                },
    { GC_TENSOR_OUTPUT,              "output"                       },
    { GC_TENSOR_OUTPUT_NORM,         "output_norm"                  },
    { GC_TENSOR_ROPE_FREQS,          "rope_freqs"                   },
    { GC_TENSOR_ROPE_FACTORS_LONG,   "rope_factors_long"            },
    { GC_TENSOR_ROPE_FACTORS_SHORT,  "rope_factors_short"           },

    { GC_TENSOR_ATTN_NORM,           "blk.%d.attn_norm"             },
    { GC_TENSOR_ATTN_Q,              "blk.%d.attn_q"                },
    { GC_TENSOR_ATTN_K,              "blk.%d.attn_k"                },
    { GC_TENSOR_ATTN_V,              "blk.%d.attn_v"                },
    { GC_TENSOR_ATTN_QKV,            "blk.%d.attn_qkv"             },
    { GC_TENSOR_ATTN_OUT,            "blk.%d.attn_output"           },
    { GC_TENSOR_ATTN_Q_NORM,         "blk.%d.attn_q_norm"           },
    { GC_TENSOR_ATTN_K_NORM,         "blk.%d.attn_k_norm"           },

    { GC_TENSOR_FFN_NORM,            "blk.%d.ffn_norm"              },
    { GC_TENSOR_FFN_GATE,            "blk.%d.ffn_gate"              },
    { GC_TENSOR_FFN_DOWN,            "blk.%d.ffn_down"              },
    { GC_TENSOR_FFN_UP,              "blk.%d.ffn_up"                },
    { GC_TENSOR_FFN_GATE_INP,        "blk.%d.ffn_gate_inp"          },

    { GC_TENSOR_FFN_DOWN_EXP,        "blk.%d.ffn_down.%d"           },
    { GC_TENSOR_FFN_GATE_EXP,        "blk.%d.ffn_gate.%d"           },
    { GC_TENSOR_FFN_UP_EXP,          "blk.%d.ffn_up.%d"             },
    { GC_TENSOR_FFN_DOWN_EXPS,       "blk.%d.ffn_down_exps"         },
    { GC_TENSOR_FFN_GATE_EXPS,       "blk.%d.ffn_gate_exps"         },
    { GC_TENSOR_FFN_UP_EXPS,         "blk.%d.ffn_up_exps"           },
    { GC_TENSOR_FFN_DOWN_SHEXP,      "blk.%d.ffn_down_shexp"        },
    { GC_TENSOR_FFN_GATE_SHEXP,      "blk.%d.ffn_gate_shexp"        },
    { GC_TENSOR_FFN_UP_SHEXP,        "blk.%d.ffn_up_shexp"          },

    { GC_TENSOR_LAYER_OUT_NORM,      "blk.%d.layer_output_norm"     },
    { GC_TENSOR_POST_ATTN_NORM,      "blk.%d.post_attention_norm"   },
    { GC_TENSOR_POST_MLP_NORM,       "blk.%d.post_ffw_norm"         },

    { GC_TENSOR_SSM_IN,              "blk.%d.ssm_in"                },
    { GC_TENSOR_SSM_CONV1D,          "blk.%d.ssm_conv1d"            },
    { GC_TENSOR_SSM_X,               "blk.%d.ssm_x"                 },
    { GC_TENSOR_SSM_DT,              "blk.%d.ssm_dt"                },
    { GC_TENSOR_SSM_A,               "blk.%d.ssm_a"                 },
    { GC_TENSOR_SSM_D,               "blk.%d.ssm_d"                 },
    { GC_TENSOR_SSM_OUT,             "blk.%d.ssm_out"               },

    { GC_TENSOR_SHORTCONV_CONV,      "blk.%d.shortconv.conv"        },
    { GC_TENSOR_SHORTCONV_INPROJ,    "blk.%d.shortconv.in_proj"     },
    { GC_TENSOR_SHORTCONV_OUTPROJ,   "blk.%d.shortconv.out_proj"    },

    { GC_TENSOR_UNKNOWN,             "(unknown)"                    },
};

std::string gc_tn(gc_arch_t arch, gc_tensor_role_t role,
                  const char * suffix, int bid, int xid)
{
    (void)arch; // tensor names are currently arch-independent in GGUF
    auto it = GC_TENSOR_NAMES.find(role);
    if (it == GC_TENSOR_NAMES.end()) {
        throw std::runtime_error(gc__format("gc_tn: unknown role %d", (int)role));
    }
    const char * tmpl = it->second;

    std::string base;
    // count %d occurrences
    int pct_count = 0;
    for (const char * p = tmpl; *p; ++p) {
        if (*p == '%' && *(p+1) == 'd') pct_count++;
    }

    if (pct_count == 0) {
        base = tmpl;
    } else if (pct_count == 1) {
        if (bid < 0) {
            throw std::runtime_error(gc__format("gc_tn: role %d needs bid", (int)role));
        }
        base = gc__format(tmpl, bid);
    } else {
        if (bid < 0 || xid < 0) {
            throw std::runtime_error(gc__format("gc_tn: role %d needs bid+xid", (int)role));
        }
        base = gc__format(tmpl, bid, xid);
    }

    if (suffix) {
        base += '.';
        base += suffix;
    }
    return base;
}

// ── Tensor info table ─────────────────────────────────────────────────────────

static const gc_tensor_info_t GC_FALLBACK_INFO = { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT };

static const std::map<gc_tensor_role_t, gc_tensor_info_t> GC_TENSOR_INFOS = {
    { GC_TENSOR_TOKEN_EMBD,      { GC_TENSOR_LAYER_INPUT,     GGML_OP_GET_ROWS  } },
    { GC_TENSOR_POS_EMBD,        { GC_TENSOR_LAYER_INPUT,     GGML_OP_GET_ROWS  } },
    { GC_TENSOR_TOKEN_EMBD_NORM, { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL       } },
    { GC_TENSOR_OUTPUT,          { GC_TENSOR_LAYER_OUTPUT,    GGML_OP_MUL_MAT   } },
    { GC_TENSOR_OUTPUT_NORM,     { GC_TENSOR_LAYER_OUTPUT,    GGML_OP_MUL       } },
    { GC_TENSOR_ROPE_FREQS,      { GC_TENSOR_LAYER_REPEATING, GGML_OP_ROPE      } },
    { GC_TENSOR_ROPE_FACTORS_LONG,  { GC_TENSOR_LAYER_REPEATING, GGML_OP_ROPE   } },
    { GC_TENSOR_ROPE_FACTORS_SHORT, { GC_TENSOR_LAYER_REPEATING, GGML_OP_ROPE   } },
    { GC_TENSOR_ATTN_NORM,       { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL       } },
    { GC_TENSOR_ATTN_Q,          { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_ATTN_K,          { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_ATTN_V,          { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_ATTN_QKV,        { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_ATTN_OUT,        { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_ATTN_Q_NORM,     { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL       } },
    { GC_TENSOR_ATTN_K_NORM,     { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL       } },
    { GC_TENSOR_FFN_NORM,        { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL       } },
    { GC_TENSOR_FFN_GATE,        { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_FFN_DOWN,        { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_FFN_UP,          { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_FFN_GATE_INP,    { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_FFN_DOWN_EXP,    { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_FFN_GATE_EXP,    { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_FFN_UP_EXP,      { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_FFN_DOWN_EXPS,   { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_FFN_GATE_EXPS,   { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_FFN_UP_EXPS,     { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_FFN_DOWN_SHEXP,  { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_FFN_GATE_SHEXP,  { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_FFN_UP_SHEXP,    { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_LAYER_OUT_NORM,  { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL       } },
    { GC_TENSOR_POST_ATTN_NORM,  { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL       } },
    { GC_TENSOR_POST_MLP_NORM,   { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL       } },
    { GC_TENSOR_SSM_IN,          { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_SSM_CONV1D,      { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_SSM_X,           { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_SSM_DT,          { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_SSM_A,           { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL       } },
    { GC_TENSOR_SSM_D,           { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL       } },
    { GC_TENSOR_SSM_OUT,         { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT   } },
    { GC_TENSOR_SHORTCONV_CONV,  { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL       } },
    { GC_TENSOR_SHORTCONV_INPROJ,  { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT } },
    { GC_TENSOR_SHORTCONV_OUTPROJ, { GC_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT } },
};

const gc_tensor_info_t & gc_tensor_info_for(gc_tensor_role_t role) {
    auto it = GC_TENSOR_INFOS.find(role);
    return it != GC_TENSOR_INFOS.end() ? it->second : GC_FALLBACK_INFO;
}
