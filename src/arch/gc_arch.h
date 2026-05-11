#pragma once

#include "ggml.h"

#include <string>
#include <vector>

// ── Architecture identifiers ──────────────────────────────────────────────────
// One enum value per supported GGUF architecture string.

typedef enum gc_arch_t {
    GC_ARCH_UNKNOWN = 0,

    // Transformer decoder (causal LM)
    GC_ARCH_LLAMA,
    GC_ARCH_LLAMA4,
    GC_ARCH_DECI,
    GC_ARCH_FALCON,
    GC_ARCH_BAICHUAN,
    GC_ARCH_GROK,
    GC_ARCH_GPT2,
    GC_ARCH_GPTJ,
    GC_ARCH_GPTNEOX,
    GC_ARCH_MPT,
    GC_ARCH_STARCODER,
    GC_ARCH_STARCODER2,
    GC_ARCH_REFACT,
    GC_ARCH_BLOOM,
    GC_ARCH_STABLELM,
    GC_ARCH_QWEN,
    GC_ARCH_QWEN2,
    GC_ARCH_QWEN2MOE,
    GC_ARCH_QWEN2VL,
    GC_ARCH_QWEN3,
    GC_ARCH_QWEN3MOE,
    GC_ARCH_QWEN3NEXT,
    GC_ARCH_QWEN3VL,
    GC_ARCH_QWEN3VLMOE,
    GC_ARCH_QWEN35,
    GC_ARCH_QWEN35MOE,
    GC_ARCH_PHI2,
    GC_ARCH_PHI3,
    GC_ARCH_PHIMOE,
    GC_ARCH_PLAMO,
    GC_ARCH_PLAMO2,
    GC_ARCH_PLAMO3,
    GC_ARCH_CODESHELL,
    GC_ARCH_ORION,
    GC_ARCH_INTERNLM2,
    GC_ARCH_MINICPM,
    GC_ARCH_MINICPM3,
    GC_ARCH_GEMMA,
    GC_ARCH_GEMMA2,
    GC_ARCH_GEMMA3,
    GC_ARCH_GEMMA3N,
    GC_ARCH_GEMMA4,
    GC_ARCH_GEMMA_EMBEDDING,
    GC_ARCH_MAMBA,
    GC_ARCH_MAMBA2,
    GC_ARCH_JAMBA,
    GC_ARCH_FALCON_H1,
    GC_ARCH_XVERSE,
    GC_ARCH_COMMAND_R,
    GC_ARCH_COHERE2,
    GC_ARCH_DBRX,
    GC_ARCH_OLMO,
    GC_ARCH_OLMO2,
    GC_ARCH_OLMOE,
    GC_ARCH_OPENELM,
    GC_ARCH_ARCTIC,
    GC_ARCH_DEEPSEEK,
    GC_ARCH_DEEPSEEK2,
    GC_ARCH_DEEPSEEK2OCR,
    GC_ARCH_CHATGLM,
    GC_ARCH_GLM4,
    GC_ARCH_GLM4_MOE,
    GC_ARCH_GLM_DSA,
    GC_ARCH_BITNET,
    GC_ARCH_T5,
    GC_ARCH_T5ENCODER,
    GC_ARCH_JAIS,
    GC_ARCH_JAIS2,
    GC_ARCH_NEMOTRON,
    GC_ARCH_NEMOTRON_H,
    GC_ARCH_NEMOTRON_H_MOE,
    GC_ARCH_EXAONE,
    GC_ARCH_EXAONE4,
    GC_ARCH_EXAONE_MOE,
    GC_ARCH_RWKV6,
    GC_ARCH_RWKV6QWEN2,
    GC_ARCH_RWKV7,
    GC_ARCH_ARWKV7,
    GC_ARCH_GRANITE,
    GC_ARCH_GRANITE_MOE,
    GC_ARCH_GRANITE_HYBRID,
    GC_ARCH_CHAMELEON,
    GC_ARCH_WAVTOKENIZER_DEC,
    GC_ARCH_PLM,
    GC_ARCH_BAILINGMOE,
    GC_ARCH_BAILINGMOE2,
    GC_ARCH_DOTS1,
    GC_ARCH_ARCEE,
    GC_ARCH_AFMOE,
    GC_ARCH_ERNIE4_5,
    GC_ARCH_ERNIE4_5_MOE,
    GC_ARCH_HUNYUAN_MOE,
    GC_ARCH_HUNYUAN_DENSE,
    GC_ARCH_HUNYUAN_VL,
    GC_ARCH_SMOLLM3,
    GC_ARCH_OPENAI_MOE,
    GC_ARCH_LFM2,
    GC_ARCH_LFM2MOE,
    GC_ARCH_DREAM,
    GC_ARCH_SMALLTHINKER,
    GC_ARCH_LLADA,
    GC_ARCH_LLADA_MOE,
    GC_ARCH_SEED_OSS,
    GC_ARCH_GROVEMOE,
    GC_ARCH_APERTUS,
    GC_ARCH_MINIMAX_M2,
    GC_ARCH_COGVLM,
    GC_ARCH_RND1,
    GC_ARCH_PANGU_EMBED,
    GC_ARCH_MISTRAL3,
    GC_ARCH_MISTRAL4,
    GC_ARCH_PADDLEOCR,
    GC_ARCH_MIMO2,
    GC_ARCH_STEP35,
    GC_ARCH_LLAMA_EMBED,
    GC_ARCH_MAINCODER,
    GC_ARCH_KIMI_LINEAR,

    // BERT-style encoders
    GC_ARCH_BERT,
    GC_ARCH_MODERN_BERT,
    GC_ARCH_NOMIC_BERT,
    GC_ARCH_NOMIC_BERT_MOE,
    GC_ARCH_NEO_BERT,
    GC_ARCH_JINA_BERT_V2,
    GC_ARCH_JINA_BERT_V3,
    GC_ARCH_EUROBERT,

    // Vision/multi-modal (stub — for completeness)
    GC_ARCH_CLIP,
} gc_arch_t;

// ── Architecture → string name (matches GGUF general.architecture values) ────

const char *  gc_arch_name(gc_arch_t arch);
gc_arch_t     gc_arch_from_string(const std::string & name);
std::vector<gc_arch_t> gc_arch_all();

// ── Architecture capability flags ────────────────────────────────────────────

bool gc_arch_is_recurrent (gc_arch_t arch); // Mamba, RWKV, …
bool gc_arch_is_encoder   (gc_arch_t arch); // BERT variants
bool gc_arch_is_moe       (gc_arch_t arch); // mixture-of-experts

// ── Tensor role identifier ────────────────────────────────────────────────────
// Logical role of a tensor in the model graph.

typedef enum gc_tensor_role_t {
    GC_TENSOR_TOKEN_EMBD,
    GC_TENSOR_TOKEN_EMBD_NORM,
    GC_TENSOR_POS_EMBD,
    GC_TENSOR_OUTPUT,
    GC_TENSOR_OUTPUT_NORM,
    GC_TENSOR_ROPE_FREQS,
    GC_TENSOR_ROPE_FACTORS_LONG,
    GC_TENSOR_ROPE_FACTORS_SHORT,

    // Attention
    GC_TENSOR_ATTN_NORM,
    GC_TENSOR_ATTN_Q,
    GC_TENSOR_ATTN_K,
    GC_TENSOR_ATTN_V,
    GC_TENSOR_ATTN_QKV,
    GC_TENSOR_ATTN_OUT,
    GC_TENSOR_ATTN_Q_NORM,
    GC_TENSOR_ATTN_K_NORM,

    // FFN
    GC_TENSOR_FFN_NORM,
    GC_TENSOR_FFN_GATE,
    GC_TENSOR_FFN_DOWN,
    GC_TENSOR_FFN_UP,
    GC_TENSOR_FFN_GATE_INP,

    // Expert FFN (MoE)
    GC_TENSOR_FFN_DOWN_EXP,
    GC_TENSOR_FFN_GATE_EXP,
    GC_TENSOR_FFN_UP_EXP,
    GC_TENSOR_FFN_DOWN_EXPS,
    GC_TENSOR_FFN_GATE_EXPS,
    GC_TENSOR_FFN_UP_EXPS,
    GC_TENSOR_FFN_DOWN_SHEXP,
    GC_TENSOR_FFN_GATE_SHEXP,
    GC_TENSOR_FFN_UP_SHEXP,

    // Layer output norm
    GC_TENSOR_LAYER_OUT_NORM,
    GC_TENSOR_POST_ATTN_NORM,
    GC_TENSOR_POST_MLP_NORM,

    // SSM (Mamba)
    GC_TENSOR_SSM_IN,
    GC_TENSOR_SSM_CONV1D,
    GC_TENSOR_SSM_X,
    GC_TENSOR_SSM_DT,
    GC_TENSOR_SSM_A,
    GC_TENSOR_SSM_D,
    GC_TENSOR_SSM_OUT,

    // LFM2 short-conv
    GC_TENSOR_SHORTCONV_CONV,
    GC_TENSOR_SHORTCONV_INPROJ,
    GC_TENSOR_SHORTCONV_OUTPROJ,

    GC_TENSOR_UNKNOWN,
} gc_tensor_role_t;

// Return the role layer type (input / repeating / output)
typedef enum gc_tensor_layer_t {
    GC_TENSOR_LAYER_INPUT,
    GC_TENSOR_LAYER_REPEATING,
    GC_TENSOR_LAYER_OUTPUT,
} gc_tensor_layer_t;

struct gc_tensor_info_t {
    gc_tensor_layer_t layer;
    ggml_op           op;
};

const gc_tensor_info_t & gc_tensor_info_for(gc_tensor_role_t role);

// ── KV key resolver ──────────────────────────────────────────────────────────
// Given an arch and a logical KV enum, returns the concrete GGUF key string.
// Keys with "%s" in their template have the arch name substituted in.

typedef enum gc_kv_key_t {
    // general (no arch prefix)
    GC_KV_GENERAL_ARCHITECTURE,
    GC_KV_GENERAL_NAME,
    GC_KV_GENERAL_FILE_TYPE,
    GC_KV_GENERAL_QUANTIZATION_VERSION,

    // arch-prefixed model hyperparams
    GC_KV_VOCAB_SIZE,
    GC_KV_CONTEXT_LENGTH,
    GC_KV_EMBEDDING_LENGTH,
    GC_KV_BLOCK_COUNT,
    GC_KV_FEED_FORWARD_LENGTH,
    GC_KV_EXPERT_FEED_FORWARD_LENGTH,
    GC_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH,
    GC_KV_EXPERT_COUNT,
    GC_KV_EXPERT_USED_COUNT,
    GC_KV_EXPERT_SHARED_COUNT,
    GC_KV_EXPERT_GATING_FUNC,
    GC_KV_MOE_EVERY_N_LAYERS,
    GC_KV_LEADING_DENSE_BLOCK_COUNT,
    GC_KV_USE_PARALLEL_RESIDUAL,
    GC_KV_LOGIT_SCALE,
    GC_KV_RESIDUAL_SCALE,
    GC_KV_EMBEDDING_SCALE,
    GC_KV_ATTN_LOGIT_SOFTCAPPING,
    GC_KV_FINAL_LOGIT_SOFTCAPPING,

    // attention
    GC_KV_ATTN_HEAD_COUNT,
    GC_KV_ATTN_HEAD_COUNT_KV,
    GC_KV_ATTN_KEY_LENGTH,
    GC_KV_ATTN_VALUE_LENGTH,
    GC_KV_ATTN_LAYERNORM_EPS,
    GC_KV_ATTN_LAYERNORM_RMS_EPS,
    GC_KV_ATTN_CAUSAL,
    GC_KV_ATTN_SLIDING_WINDOW,
    GC_KV_ATTN_SCALE,
    GC_KV_ATTN_Q_LORA_RANK,
    GC_KV_ATTN_KV_LORA_RANK,
    GC_KV_ATTN_KEY_LENGTH_MLA,
    GC_KV_ATTN_VALUE_LENGTH_MLA,
    GC_KV_ATTN_CLAMP_KQV,
    GC_KV_ATTN_MAX_ALIBI_BIAS,

    // RoPE
    GC_KV_ROPE_DIMENSION_COUNT,
    GC_KV_ROPE_FREQ_BASE,
    GC_KV_ROPE_SCALE_LINEAR,
    GC_KV_ROPE_SCALING_TYPE,
    GC_KV_ROPE_SCALING_FACTOR,
    GC_KV_ROPE_SCALING_ORIG_CTX_LEN,
    GC_KV_ROPE_SCALING_FINETUNED,
    GC_KV_ROPE_SCALING_YARN_LOG_MUL,
    GC_KV_ROPE_SCALING_YARN_EXT_FACTOR,
    GC_KV_ROPE_SCALING_YARN_ATTN_FACTOR,
    GC_KV_ROPE_SCALING_YARN_BETA_FAST,
    GC_KV_ROPE_SCALING_YARN_BETA_SLOW,
    GC_KV_ROPE_DIMENSION_SECTIONS,

    // SSM / Mamba
    GC_KV_SSM_CONV_KERNEL,
    GC_KV_SSM_INNER_SIZE,
    GC_KV_SSM_STATE_SIZE,
    GC_KV_SSM_TIME_STEP_RANK,
    GC_KV_SSM_GROUP_COUNT,

    // LFM2
    GC_KV_SHORTCONV_L_CACHE,

    // tokenizer
    GC_KV_TOKENIZER_MODEL,
    GC_KV_TOKENIZER_PRE,
    GC_KV_TOKENIZER_LIST,
    GC_KV_TOKENIZER_TOKEN_TYPE,
    GC_KV_TOKENIZER_SCORES,
    GC_KV_TOKENIZER_MERGES,
    GC_KV_TOKENIZER_BOS_ID,
    GC_KV_TOKENIZER_EOS_ID,
    GC_KV_TOKENIZER_EOT_ID,
    GC_KV_TOKENIZER_UNK_ID,
    GC_KV_TOKENIZER_PAD_ID,
    GC_KV_TOKENIZER_ADD_BOS,
    GC_KV_TOKENIZER_ADD_EOS,
    GC_KV_TOKENIZER_CHAT_TEMPLATE,
    GC_KV_TOKENIZER_HF_JSON,

    // shard
    GC_KV_SPLIT_NO,
    GC_KV_SPLIT_COUNT,
    GC_KV_SPLIT_TENSORS_COUNT,
} gc_kv_key_t;

// Resolve: gc_kv_resolve(arch, GC_KV_ATTN_HEAD_COUNT) → "llama.attention.head_count"
std::string gc_kv_resolve(gc_arch_t arch, gc_kv_key_t key);

// ── Tensor name resolver ─────────────────────────────────────────────────────
// Translate (arch, role, suffix, block_idx) → GGUF tensor name string.
//
//   gc_tn(GC_ARCH_LLAMA, GC_TENSOR_OUTPUT, "weight")       → "output.weight"
//   gc_tn(GC_ARCH_LLAMA, GC_TENSOR_ATTN_Q, "weight", 3)    → "blk.3.attn_q.weight"
//
std::string gc_tn(gc_arch_t arch, gc_tensor_role_t role,
                  const char * suffix  = nullptr,
                  int          bid     = -1,
                  int          xid     = -1);
