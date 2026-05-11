#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Unicode codepoint category flags (mirrors llama.cpp unicode_cpt_flags, renamed)

struct gc_cpt_flags_t {
    enum {
        UNDEFINED       = 0x0001,
        NUMBER          = 0x0002,
        LETTER          = 0x0004,
        SEPARATOR       = 0x0008,
        ACCENT_MARK     = 0x0010,
        PUNCTUATION     = 0x0020,
        SYMBOL          = 0x0040,
        CONTROL         = 0x0080,
        MASK_CATEGORIES = 0x00FF,
        WHITESPACE      = 0x0100,
        LOWERCASE       = 0x0200,
        UPPERCASE       = 0x0400,
        NFD             = 0x0800,
    };

    uint16_t is_undefined   : 1;
    uint16_t is_number      : 1;
    uint16_t is_letter      : 1;
    uint16_t is_separator   : 1;
    uint16_t is_accent_mark : 1;
    uint16_t is_punctuation : 1;
    uint16_t is_symbol      : 1;
    uint16_t is_control     : 1;
    uint16_t is_whitespace  : 1;
    uint16_t is_lowercase   : 1;
    uint16_t is_uppercase   : 1;
    uint16_t is_nfd         : 1;

    inline gc_cpt_flags_t(uint16_t flags = 0) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        *reinterpret_cast<uint16_t*>(this) = flags;
#else
        is_undefined   = (flags & UNDEFINED)   ? 1 : 0;
        is_number      = (flags & NUMBER)      ? 1 : 0;
        is_letter      = (flags & LETTER)      ? 1 : 0;
        is_separator   = (flags & SEPARATOR)   ? 1 : 0;
        is_accent_mark = (flags & ACCENT_MARK) ? 1 : 0;
        is_punctuation = (flags & PUNCTUATION) ? 1 : 0;
        is_symbol      = (flags & SYMBOL)      ? 1 : 0;
        is_control     = (flags & CONTROL)     ? 1 : 0;
        is_whitespace  = (flags & WHITESPACE)  ? 1 : 0;
        is_lowercase   = (flags & LOWERCASE)   ? 1 : 0;
        is_uppercase   = (flags & UPPERCASE)   ? 1 : 0;
        is_nfd         = (flags & NFD)         ? 1 : 0;
#endif
    }

    inline uint16_t as_uint() const {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        return *reinterpret_cast<const uint16_t*>(this);
#else
        return (uint16_t)(
              is_undefined   * UNDEFINED
            + is_number      * NUMBER
            + is_letter      * LETTER
            + is_separator   * SEPARATOR
            + is_accent_mark * ACCENT_MARK
            + is_punctuation * PUNCTUATION
            + is_symbol      * SYMBOL
            + is_control     * CONTROL
            + is_whitespace  * WHITESPACE
            + is_lowercase   * LOWERCASE
            + is_uppercase   * UPPERCASE
            + is_nfd         * NFD);
#endif
    }

    inline uint16_t category_flag() const { return as_uint() & MASK_CATEGORIES; }
};

size_t      gc_utf8_len       (char src);
std::string gc_cpt_to_utf8    (uint32_t cpt);
uint32_t    gc_cpt_from_utf8  (const std::string & utf8, size_t & offset);

std::vector<uint32_t> gc_cpts_from_utf8    (const std::string & utf8);
std::vector<uint32_t> gc_cpts_normalize_nfd(const std::vector<uint32_t> & cpts);

gc_cpt_flags_t gc_cpt_flags    (uint32_t cpt);
gc_cpt_flags_t gc_cpt_flags_utf8(const std::string & utf8);

std::string gc_byte_to_utf8(uint8_t byte);
uint8_t     gc_utf8_to_byte(const std::string & utf8);

uint32_t gc_tolower(uint32_t cpt);

bool gc_cpt_is_han(uint32_t cpt);

// Split text by regex patterns (pre-tokenization).
// byte_encode=true wraps unmatched bytes in their utf8 representation.
std::vector<std::string> gc_regex_split(
        const std::string              & text,
        const std::vector<std::string> & regex_exprs,
        bool                             byte_encode = true);
