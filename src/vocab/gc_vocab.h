#pragma once

#include "gc_unicode.h"
#include "../loader/gc_common.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// ── Token id type ─────────────────────────────────────────────────────────────
typedef int32_t gc_token_t;

static constexpr gc_token_t GC_TOKEN_NULL = -1;

// ── Vocab / tokenizer type ────────────────────────────────────────────────────
typedef enum gc_vocab_type_t {
    GC_VOCAB_TYPE_NONE   = 0,
    GC_VOCAB_TYPE_SPM    = 1,  // SentencePiece unigram (Llama, Mistral, …)
    GC_VOCAB_TYPE_BPE    = 2,  // Byte-pair encoding (GPT-2, Qwen, …)
    GC_VOCAB_TYPE_WPM    = 3,  // WordPiece (BERT, …)
    GC_VOCAB_TYPE_UGM    = 4,  // Unigram + precompiled charsmap (Gemma, …)
    GC_VOCAB_TYPE_RWKV   = 5,  // RWKV world tokenizer
    GC_VOCAB_TYPE_PLAMO2 = 6,  // PLaMo2 bespoke tokenizer
} gc_vocab_type_t;

// Pre-tokenization regex variant — controls which regex splits are applied
// before BPE/SPM encoding.
typedef enum gc_vocab_pre_t {
    GC_VOCAB_PRE_DEFAULT        =  0,
    GC_VOCAB_PRE_LLAMA3         =  1,
    GC_VOCAB_PRE_DEEPSEEK_LLM   =  2,
    GC_VOCAB_PRE_DEEPSEEK_CODER =  3,
    GC_VOCAB_PRE_FALCON         =  4,
    GC_VOCAB_PRE_MPT            =  5,
    GC_VOCAB_PRE_STARCODER      =  6,
    GC_VOCAB_PRE_GPT2           =  7,
    GC_VOCAB_PRE_REFACT         =  8,
    GC_VOCAB_PRE_COMMAND_R      =  9,
    GC_VOCAB_PRE_STABLELM2      = 10,
    GC_VOCAB_PRE_QWEN2          = 11,
    GC_VOCAB_PRE_OLMO           = 12,
    GC_VOCAB_PRE_DBRX           = 13,
    GC_VOCAB_PRE_SMAUG          = 14,
    GC_VOCAB_PRE_PORO           = 15,
    GC_VOCAB_PRE_CHATGLM3       = 16,
    GC_VOCAB_PRE_CHATGLM4       = 17,
    GC_VOCAB_PRE_VIKING         = 18,
    GC_VOCAB_PRE_JAIS           = 19,
    GC_VOCAB_PRE_TEKKEN         = 20,
    GC_VOCAB_PRE_SMOLLM         = 21,
    GC_VOCAB_PRE_CODESHELL      = 22,
    GC_VOCAB_PRE_BLOOM          = 23,
    GC_VOCAB_PRE_GPT3_FINNISH   = 24,
    GC_VOCAB_PRE_EXAONE         = 25,
    GC_VOCAB_PRE_CHAMELEON      = 26,
    GC_VOCAB_PRE_MINERVA        = 27,
    GC_VOCAB_PRE_DEEPSEEK3_LLM  = 28,
    GC_VOCAB_PRE_GPT4O          = 29,
    GC_VOCAB_PRE_SUPERBPE       = 30,
    GC_VOCAB_PRE_TRILLION       = 31,
    GC_VOCAB_PRE_BAILINGMOE     = 32,
    GC_VOCAB_PRE_LLAMA4         = 33,
    GC_VOCAB_PRE_PIXTRAL        = 34,
    GC_VOCAB_PRE_SEED_CODER     = 35,
    GC_VOCAB_PRE_HUNYUAN        = 36,
    GC_VOCAB_PRE_KIMI_K2        = 37,
    GC_VOCAB_PRE_HUNYUAN_DENSE  = 38,
    GC_VOCAB_PRE_GROK_2         = 39,
    GC_VOCAB_PRE_GRANITE_DOCLING= 40,
    GC_VOCAB_PRE_MINIMAX_M2     = 41,
    GC_VOCAB_PRE_AFMOE          = 42,
    GC_VOCAB_PRE_SOLAR_OPEN     = 43,
    GC_VOCAB_PRE_YOUTU          = 44,
    GC_VOCAB_PRE_EXAONE_MOE     = 45,
    GC_VOCAB_PRE_QWEN35         = 46,
    GC_VOCAB_PRE_TINY_AYA       = 47,
    GC_VOCAB_PRE_JOYAI_LLM      = 48,
    GC_VOCAB_PRE_JAIS2          = 49,
    GC_VOCAB_PRE_GEMMA4         = 50,
    GC_VOCAB_PRE_SARVAM_MOE     = 51,
} gc_vocab_pre_t;

// ── Token attribute flags ─────────────────────────────────────────────────────
typedef enum gc_token_attr_t {
    GC_TOKEN_ATTR_UNDEFINED    = 0,
    GC_TOKEN_ATTR_UNKNOWN      = 1 << 0,
    GC_TOKEN_ATTR_UNUSED       = 1 << 1,
    GC_TOKEN_ATTR_NORMAL       = 1 << 2,
    GC_TOKEN_ATTR_CONTROL      = 1 << 3,
    GC_TOKEN_ATTR_USER_DEFINED = 1 << 4,
    GC_TOKEN_ATTR_BYTE         = 1 << 5,
    GC_TOKEN_ATTR_NORMALIZED   = 1 << 6,
    GC_TOKEN_ATTR_LSTRIP       = 1 << 7,
    GC_TOKEN_ATTR_RSTRIP       = 1 << 8,
    GC_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
} gc_token_attr_t;

// ── Forward declarations ──────────────────────────────────────────────────────
struct gc_model_loader_t;

// ── Vocab struct ──────────────────────────────────────────────────────────────
struct gc_vocab_t {
    gc_vocab_t();
    ~gc_vocab_t();

    // Load vocab from GGUF metadata via the model loader.
    gc_status_t load(const gc_model_loader_t & loader);

    // ── Type / pre-type ──────────────────────────────────────────────────────
    gc_vocab_type_t type()              const;
    gc_vocab_pre_t  pre_type()          const;
    const char *    type_name()         const;
    std::string     tokenizer_model()   const;
    std::string     tokenizer_pre_str() const;

    // Aliases used internally by tokenizer classes
    gc_vocab_type_t get_type()     const { return type(); }
    gc_vocab_pre_t  get_pre_type() const { return pre_type(); }

    // ── Vocabulary size ──────────────────────────────────────────────────────
    uint32_t n_tokens()      const;
    uint32_t n_token_types() const;

    // ── Token predicates ─────────────────────────────────────────────────────
    bool is_normal      (gc_token_t id) const;
    bool is_unknown     (gc_token_t id) const;
    bool is_control     (gc_token_t id) const;
    bool is_byte        (gc_token_t id) const;
    bool is_user_defined(gc_token_t id) const;
    bool is_unused      (gc_token_t id) const;
    bool is_eog         (gc_token_t id) const;  // end-of-generation

    // ── Token ↔ byte ─────────────────────────────────────────────────────────
    uint8_t    token_to_byte(gc_token_t id) const;
    gc_token_t byte_to_token(uint8_t ch)    const;

    // ── Token data ────────────────────────────────────────────────────────────
    const char *      token_text (gc_token_t id) const;
    float             token_score(gc_token_t id) const;
    gc_token_attr_t   token_attr (gc_token_t id) const;

    // ── Special tokens ────────────────────────────────────────────────────────
    gc_token_t token_bos()    const;
    gc_token_t token_eos()    const;
    gc_token_t token_eot()    const;  // end-of-turn
    gc_token_t token_eom()    const;  // end-of-message
    gc_token_t token_unk()    const;
    gc_token_t token_sep()    const;
    gc_token_t token_nl()     const;
    gc_token_t token_pad()    const;
    gc_token_t token_mask()   const;
    gc_token_t token_prefix() const;
    gc_token_t token_middle() const;
    gc_token_t token_suffix() const;
    gc_token_t token_fim_pre() const;
    gc_token_t token_fim_suf() const;
    gc_token_t token_fim_mid() const;
    gc_token_t token_fim_pad() const;
    gc_token_t token_fim_rep() const;
    gc_token_t token_fim_sep() const;

    // ── Flags ─────────────────────────────────────────────────────────────────
    bool add_bos()                    const;
    bool add_eos()                    const;
    bool add_sep()                    const;
    bool add_space_prefix()           const;
    bool ignore_merges()              const;
    bool clean_spaces()               const;
    bool remove_extra_whitespaces()   const;
    bool escape_whitespaces()         const;
    bool treat_whitespace_as_suffix() const;

    // Aliases used internally by tokenizer classes
    bool get_add_bos()                    const { return add_bos(); }
    bool get_add_eos()                    const { return add_eos(); }
    bool get_add_sep()                    const { return add_sep(); }
    bool get_add_space_prefix()           const { return add_space_prefix(); }
    bool get_ignore_merges()              const { return ignore_merges(); }
    bool get_clean_spaces()               const { return clean_spaces(); }
    bool get_remove_extra_whitespaces()   const { return remove_extra_whitespaces(); }
    bool get_escape_whitespaces()         const { return escape_whitespaces(); }
    bool get_treat_whitespace_as_suffix() const { return treat_whitespace_as_suffix(); }

    // ── Internal helpers for tokenizer implementations ────────────────────────
    // Look up a token by its text string; returns GC_TOKEN_NULL if not found.
    gc_token_t text_to_token(const std::string & text) const;

    struct token_data {
        std::string     text;
        float           score;
        gc_token_attr_t attr;
    };
    const token_data & get_token_data(gc_token_t id) const;

    // ── BPE internals ─────────────────────────────────────────────────────────
    int                      max_token_len()  const;
    int                      find_bpe_rank(const std::string & left, const std::string & right) const;
    std::vector<std::string> bpe_merges()     const;
    std::vector<char>        precompiled_charsmap() const;

    // ── Encoding / decoding ───────────────────────────────────────────────────

    // Tokenize text into token ids.
    // add_special: prepend/append BOS/EOS per vocab settings.
    // parse_special: allow special token text matches inside the input string.
    std::vector<gc_token_t> tokenize(
            const std::string & text,
            bool                add_special,
            bool                parse_special = false) const;

    int32_t tokenize(
            const char  * text,
            int32_t       text_len,
            gc_token_t  * out,
            int32_t       out_max,
            bool          add_special,
            bool          parse_special) const;

    // Convert a single token to its text piece.
    // lstrip: number of leading bytes to strip (for space-prefixed tokens).
    // special: if true, render special tokens as their text form.
    int32_t token_to_piece(
            gc_token_t  token,
            char      * buf,
            int32_t     buf_len,
            int32_t     lstrip,
            bool        special) const;

    const std::string & token_to_piece(gc_token_t token) const;

    // Detokenize a token array to a string.
    std::string detokenize(
            const std::vector<gc_token_t> & tokens,
            bool                            special) const;

    int32_t detokenize(
            const gc_token_t * tokens,
            int32_t            n_tokens,
            char             * text,
            int32_t            text_len_max,
            bool               remove_special,
            bool               unparse_special) const;

    void print_info() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
