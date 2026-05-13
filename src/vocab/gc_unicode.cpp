#include "gc_unicode.h"
#include "gc_unicode_data.h"
#include "gc_debug.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

size_t gc_utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

static std::string gc__cpts_to_utf8(const std::vector<uint32_t> & cps) {
    std::string result;
    for (size_t i = 0; i < cps.size(); ++i) {
        result.append(gc_cpt_to_utf8(cps[i]));
    }
    return result;
}

uint32_t gc_cpt_from_utf8(const std::string & utf8, size_t & offset) {
    assert(offset < utf8.size());
    if (!(utf8[offset + 0] & 0x80)) {
        auto result = utf8[offset + 0];
        offset += 1;
        return result;
    }
    if (!(utf8[offset + 0] & 0x40)) {
        throw std::invalid_argument("invalid character");
    }
    if (!(utf8[offset + 0] & 0x20)) {
        if (offset + 1 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
        offset += 2;
        return result;
    }
    if (!(utf8[offset + 0] & 0x10)) {
        if (offset + 2 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) || !((utf8[offset + 2] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x0f) << 12) | ((utf8[offset + 1] & 0x3f) << 6) | (utf8[offset + 2] & 0x3f);
        offset += 3;
        return result;
    }
    if (!(utf8[offset + 0] & 0x08)) {
        if (offset + 3 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) || !((utf8[offset + 2] & 0xc0) == 0x80) || !((utf8[offset + 3] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x07) << 18) | ((utf8[offset + 1] & 0x3f) << 12) | ((utf8[offset + 2] & 0x3f) << 6) | (utf8[offset + 3] & 0x3f);
        offset += 4;
        return result;
    }
    throw std::invalid_argument("failed to convert utf8 to codepoint");
}

static std::vector<gc_cpt_flags_t> gc__cpt_flags_array() {
    std::vector<gc_cpt_flags_t> cpt_flags(GC_MAX_CODEPOINTS, gc_cpt_flags_t::UNDEFINED);

    assert(gc_unicode_ranges_flags.begin()[0].first == 0);
    assert(gc_unicode_ranges_flags.begin()[gc_unicode_ranges_flags.size()-1].first == GC_MAX_CODEPOINTS);
    for (size_t i = 1; i < gc_unicode_ranges_flags.size(); ++i) {
        const auto range_ini = gc_unicode_ranges_flags.begin()[i-1];
        const auto range_end = gc_unicode_ranges_flags.begin()[i];
        for (uint32_t cpt = range_ini.first; cpt < range_end.first; ++cpt) {
            cpt_flags[cpt] = range_ini.second;
        }
    }

    for (auto cpt : gc_unicode_set_whitespace) {
        cpt_flags[cpt].is_whitespace = true;
    }
    for (auto p : gc_unicode_map_lowercase) {
        cpt_flags[p.second].is_lowercase = true;
    }
    for (auto p : gc_unicode_map_uppercase) {
        cpt_flags[p.second].is_uppercase = true;
    }
    for (auto & range : gc_unicode_ranges_nfd) {
        cpt_flags[range.nfd].is_nfd = true;
    }

    return cpt_flags;
}

static std::unordered_map<uint8_t, std::string> gc__byte_to_utf8_map() {
    std::unordered_map<uint8_t, std::string> map;
    for (int ch = 0x21; ch <= 0x7E; ++ch) { map[ch] = gc_cpt_to_utf8(ch); }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) { map[ch] = gc_cpt_to_utf8(ch); }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) { map[ch] = gc_cpt_to_utf8(ch); }
    int n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(ch) == map.end()) {
            map[ch] = gc_cpt_to_utf8(256 + n);
            ++n;
        }
    }
    return map;
}

static std::unordered_map<std::string, uint8_t> gc__utf8_to_byte_map() {
    std::unordered_map<std::string, uint8_t> map;
    for (int ch = 0x21; ch <= 0x7E; ++ch) { map[gc_cpt_to_utf8(ch)] = ch; }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) { map[gc_cpt_to_utf8(ch)] = ch; }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) { map[gc_cpt_to_utf8(ch)] = ch; }
    int n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(gc_cpt_to_utf8(ch)) == map.end()) {
            map[gc_cpt_to_utf8(256 + n)] = ch;
            ++n;
        }
    }
    return map;
}

static std::vector<std::string> gc__byte_encoding_process(const std::vector<std::string> & bpe_words) {
    std::vector<std::string> encoded;
    for (const auto & word : bpe_words) {
        std::string text_utf;
        auto utf_word = gc_cpts_from_utf8(word);
        for (size_t i = 0; i < utf_word.size(); ++i) {
            text_utf += gc_cpt_to_utf8(utf_word[i]);
        }
        std::string tok;
        for (char & c : text_utf) {
            tok += gc_byte_to_utf8(c);
        }
        encoded.emplace_back(tok);
    }
    return encoded;
}

// ── Custom fast-path regex splits ─────────────────────────────────────────────

static std::vector<size_t> gc__regex_split_gpt2(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());
    const auto cpts = gc_cpts_from_utf8(text);
    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;
        static const uint32_t OOR = 0xFFFFFFFF;
        auto _get_cpt   = [&](size_t pos) -> uint32_t      { return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OOR; };
        auto _get_flags = [&](size_t pos) -> gc_cpt_flags_t { return (offset_ini <= pos && pos < offset_end) ? gc_cpt_flags(cpts[pos]) : gc_cpt_flags_t{}; };
        size_t _prev_end = offset_ini;
        auto _add = [&](size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) bpe_offsets.push_back(len);
            _prev_end = end;
            return len;
        };
        for (size_t pos = offset_ini; pos < offset_end; ) {
            const uint32_t cpt   = _get_cpt(pos);
            const auto     flags = _get_flags(pos);
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t n = _get_cpt(pos+1);
                if (n == 's' || n == 't' || n == 'm' || n == 'd') { pos += _add(pos+2); continue; }
                if (pos+2 < offset_end) {
                    uint32_t nn = _get_cpt(pos+2);
                    if ((n=='r'&&nn=='e')||(n=='v'&&nn=='e')||(n=='l'&&nn=='l')) { pos += _add(pos+3); continue; }
                }
            }
            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            if (flags2.is_letter) {
                pos += (cpt == ' ');
                while (flags2.is_letter) flags2 = _get_flags(++pos);
                _add(pos); continue;
            }
            if (flags2.is_number) {
                pos += (cpt == ' ');
                while (flags2.is_number) flags2 = _get_flags(++pos);
                _add(pos); continue;
            }
            if (!(flags2.is_whitespace|flags2.is_letter|flags2.is_number) && flags2.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace|flags2.is_letter|flags2.is_number) && flags2.as_uint()) flags2 = _get_flags(++pos);
                _add(pos); continue;
            }
            size_t nws = 0;
            while (_get_flags(pos+nws).is_whitespace) nws++;
            if (nws > 1 && _get_cpt(pos+nws) != OOR) { pos += nws-1; _add(pos); continue; }
            if (nws > 0) { pos += nws; _add(pos); continue; }
            _add(++pos);
        }
    }
    return bpe_offsets;
}

static std::vector<size_t> gc__regex_split_llama3(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());
    const auto cpts = gc_cpts_from_utf8(text);
    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;
        static const uint32_t OOR = 0xFFFFFFFF;
        auto _get_cpt   = [&](size_t pos) -> uint32_t      { return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OOR; };
        auto _get_flags = [&](size_t pos) -> gc_cpt_flags_t { return (offset_ini <= pos && pos < offset_end) ? gc_cpt_flags(cpts[pos]) : gc_cpt_flags_t{}; };
        size_t _prev_end = offset_ini;
        auto _add = [&](size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) bpe_offsets.push_back(len);
            _prev_end = end;
            return len;
        };
        for (size_t pos = offset_ini; pos < offset_end; ) {
            const uint32_t cpt   = _get_cpt(pos);
            const auto     flags = _get_flags(pos);
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t n = gc_tolower(_get_cpt(pos+1));
                if (n == 's' || n == 't' || n == 'm' || n == 'd') { pos += _add(pos+2); continue; }
                if (pos+2 < offset_end) {
                    uint32_t nn = gc_tolower(_get_cpt(pos+2));
                    if ((n=='r'&&nn=='e')||(n=='v'&&nn=='e')||(n=='l'&&nn=='l')) { pos += _add(pos+3); continue; }
                }
            }
            if (!(cpt=='\r'||cpt=='\n'||flags.is_number)) {
                if (flags.is_letter || _get_flags(pos+1).is_letter) {
                    pos++;
                    while (_get_flags(pos).is_letter) pos++;
                    _add(pos); continue;
                }
            }
            if (flags.is_number) {
                size_t ini = pos;
                while (_get_flags(pos).is_number) {
                    if (++pos - ini >= 3) { _add(pos); ini = pos; }
                }
                _add(pos); continue;
            }
            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            if (!(flags2.is_whitespace|flags2.is_letter|flags2.is_number) && flags.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace|flags2.is_letter|flags2.is_number) && flags2.as_uint()) flags2 = _get_flags(++pos);
                uint32_t c2 = _get_cpt(pos);
                while (c2=='\r'||c2=='\n') c2 = _get_cpt(++pos);
                _add(pos); continue;
            }
            size_t nws = 0; size_t last_rn = 0;
            while (_get_flags(pos+nws).is_whitespace) {
                uint32_t c2 = _get_cpt(pos+nws);
                if (c2=='\r'||c2=='\n') last_rn = pos+nws+1;
                nws++;
            }
            if (last_rn > 0) { pos = last_rn; _add(pos); continue; }
            if (nws > 1 && _get_cpt(pos+nws) != OOR) { pos += nws-1; _add(pos); continue; }
            if (nws > 0) { pos += nws; _add(pos); continue; }
            _add(++pos);
        }
    }
    return bpe_offsets;
}

static std::vector<size_t> gc__regex_split_qwen2(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());
    const auto cpts = gc_cpts_from_utf8(text);
    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;
        static const uint32_t OOR = 0xFFFFFFFF;
        auto _get_cpt   = [&](size_t pos) -> uint32_t      { return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OOR; };
        auto _get_flags = [&](size_t pos) -> gc_cpt_flags_t { return (offset_ini <= pos && pos < offset_end) ? gc_cpt_flags(cpts[pos]) : gc_cpt_flags_t{}; };
        size_t _prev_end = offset_ini;
        auto _add = [&](size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) bpe_offsets.push_back(len);
            _prev_end = end;
            return len;
        };
        for (size_t pos = offset_ini; pos < offset_end; ) {
            const uint32_t cpt   = _get_cpt(pos);
            const auto     flags = _get_flags(pos);
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t n = gc_tolower(_get_cpt(pos+1));
                if (n == 's' || n == 't' || n == 'm' || n == 'd') { pos += _add(pos+2); continue; }
                if (pos+2 < offset_end) {
                    uint32_t nn = gc_tolower(_get_cpt(pos+2));
                    if ((n=='r'&&nn=='e')||(n=='v'&&nn=='e')||(n=='l'&&nn=='l')) { pos += _add(pos+3); continue; }
                }
            }
            if (!(cpt=='\r'||cpt=='\n'||flags.is_number)) {
                if (flags.is_letter || _get_flags(pos+1).is_letter) {
                    pos++;
                    while (_get_flags(pos).is_letter) pos++;
                    _add(pos); continue;
                }
            }
            if (flags.is_number) { pos++; _add(pos); continue; }
            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            if (!(flags2.is_whitespace|flags2.is_letter|flags2.is_number) && flags.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace|flags2.is_letter|flags2.is_number) && flags2.as_uint()) flags2 = _get_flags(++pos);
                uint32_t c2 = _get_cpt(pos);
                while (c2=='\r'||c2=='\n') c2 = _get_cpt(++pos);
                _add(pos); continue;
            }
            size_t nws = 0; size_t last_rn = 0;
            while (_get_flags(pos+nws).is_whitespace) {
                uint32_t c2 = _get_cpt(pos+nws);
                if (c2=='\r'||c2=='\n') last_rn = pos+nws+1;
                nws++;
            }
            if (last_rn > 0) { pos = last_rn; _add(pos); continue; }
            if (nws > 1 && _get_cpt(pos+nws) != OOR) { pos += nws-1; _add(pos); continue; }
            if (nws > 0) { pos += nws; _add(pos); continue; }
            _add(++pos);
        }
    }
    return bpe_offsets;
}

static std::vector<size_t> gc__regex_split_kimi_k2(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());
    const auto cpts = gc_cpts_from_utf8(text);
    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;
        static const uint32_t OOR = 0xFFFFFFFF;
        auto _get_cpt   = [&](size_t pos) -> uint32_t      { return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OOR; };
        auto _get_flags = [&](size_t pos) -> gc_cpt_flags_t { return (offset_ini <= pos && pos < offset_end) ? gc_cpt_flags(cpts[pos]) : gc_cpt_flags_t{}; };
        size_t _prev_end = offset_ini;
        auto _add = [&](size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) bpe_offsets.push_back(len);
            _prev_end = end;
            return len;
        };
        for (size_t pos = offset_ini; pos < offset_end; ) {
            const uint32_t cpt   = _get_cpt(pos);
            const auto     flags = _get_flags(pos);
            if (gc_cpt_is_han(cpt)) {
                while (gc_cpt_is_han(_get_cpt(pos))) pos++;
                _add(pos); continue;
            }
            bool is_letter_pat = (flags.is_letter && !gc_cpt_is_han(cpt)) ||
                                 (!(cpt=='\r'||cpt=='\n'||flags.is_letter||flags.is_number) &&
                                  _get_flags(pos+1).is_letter && !gc_cpt_is_han(_get_cpt(pos+1)));
            if (is_letter_pat) {
                bool had_lead = false;
                if (!(cpt=='\r'||cpt=='\n'||flags.is_letter||flags.is_number)) { had_lead = true; pos++; }
                bool has_letters = false;
                while (_get_flags(pos).is_letter && !gc_cpt_is_han(_get_cpt(pos))) { has_letters = true; pos++; }
                if (has_letters || (!had_lead && _get_flags(pos).is_letter && !gc_cpt_is_han(_get_cpt(pos)))) {
                    if (!has_letters) pos++;
                    while (_get_flags(pos).is_letter && !gc_cpt_is_han(_get_cpt(pos))) pos++;
                    if (_get_cpt(pos) == '\'' && pos+1 < offset_end) {
                        uint32_t n = gc_tolower(_get_cpt(pos+1));
                        if (n=='s'||n=='t'||n=='m'||n=='d') { pos += 2; }
                        else if (pos+2 < offset_end) {
                            uint32_t nn = gc_tolower(_get_cpt(pos+2));
                            if ((n=='r'&&nn=='e')||(n=='v'&&nn=='e')||(n=='l'&&nn=='l')) pos += 3;
                        }
                    }
                    _add(pos); continue;
                } else if (had_lead) pos--;
            }
            if (flags.is_number) {
                size_t ini = pos;
                while (_get_flags(pos).is_number) {
                    if (++pos - ini >= 3) { _add(pos); ini = pos; }
                }
                _add(pos); continue;
            }
            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            if (!(flags2.is_whitespace||flags2.is_letter||flags2.is_number) && flags2.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace||flags2.is_letter||flags2.is_number) && flags2.as_uint()) flags2 = _get_flags(++pos);
                uint32_t c2 = _get_cpt(pos);
                while (c2=='\r'||c2=='\n') c2 = _get_cpt(++pos);
                _add(pos); continue;
            }
            size_t nws = 0; size_t last_rn = 0;
            while (_get_flags(pos+nws).is_whitespace) {
                uint32_t c2 = _get_cpt(pos+nws);
                if (c2=='\r'||c2=='\n') last_rn = pos+nws+1;
                nws++;
            }
            if (last_rn > 0) { pos = last_rn; _add(pos); continue; }
            if (nws > 1 && _get_cpt(pos+nws) != OOR) { pos += nws-1; _add(pos); continue; }
            if (nws > 0) { pos += nws; _add(pos); continue; }
            _add(++pos);
        }
    }
    return bpe_offsets;
}

static std::vector<size_t> gc__regex_split_afmoe(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());
    const auto cpts = gc_cpts_from_utf8(text);
    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;
        auto _get_flags = [&](size_t pos) -> gc_cpt_flags_t {
            return (offset_ini <= pos && pos < offset_end) ? gc_cpt_flags(cpts[pos]) : gc_cpt_flags_t{};
        };
        size_t _prev_end = offset_ini;
        auto _add = [&](size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) bpe_offsets.push_back(len);
            _prev_end = end;
            return len;
        };
        for (size_t pos = offset_ini; pos < offset_end; ) {
            const auto flags = _get_flags(pos);
            if (flags.is_number) {
                size_t digit_start = pos, digit_count = 0;
                while (_get_flags(pos).is_number && pos < offset_end) { digit_count++; pos++; }
                size_t rem = digit_count % 3, cur = digit_start;
                if (rem > 0) { _add(cur + rem); cur += rem; }
                while (cur < digit_start + digit_count) { _add(cur + 3); cur += 3; }
                continue;
            }
            pos++;
        }
        if (_prev_end < offset_end) _add(offset_end);
    }
    return bpe_offsets;
}

static std::vector<size_t> gc__regex_split_newlines(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());
    const auto cpts = gc_cpts_from_utf8(text);
    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;
        size_t pos = offset_ini;
        while (pos < offset_end) {
            const bool is_nl = (cpts[pos] == '\n');
            const size_t run_start = pos;
            while (pos < offset_end && (cpts[pos] == '\n') == is_nl) pos++;
            bpe_offsets.push_back(pos - run_start);
        }
    }
    return bpe_offsets;
}

static std::vector<size_t> gc__regex_split_custom(const std::string & text, const std::string & expr, const std::vector<size_t> & offsets) {
    if (expr == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)") {
        return gc__regex_split_gpt2(text, offsets);
    }
    if (expr == "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" ||
        expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
        return gc__regex_split_llama3(text, offsets);
    }
    if (expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
        return gc__regex_split_qwen2(text, offsets);
    }
    if (expr == "\\p{Han}+") {
        return gc__regex_split_kimi_k2(text, offsets);
    }
    if (expr == "\\p{AFMoE_digits}" || expr == "\\d{1,3}(?=(?:\\d{3})*\\b)") {
        return gc__regex_split_afmoe(text, offsets);
    }
    if (expr == "[^\\n]+|[\\n]+") {
        return gc__regex_split_newlines(text, offsets);
    }
    return {};
}

template <typename CharT>
static std::vector<size_t> gc__regex_split_stl(
        const std::basic_string<CharT> & text,
        const std::basic_string<CharT> & regex,
        const std::vector<size_t>      & offsets)
{
    using BidirIt = typename std::basic_string<CharT>::const_iterator;
#ifdef _MSC_VER
    constexpr auto rf = std::regex_constants::ECMAScript;
#else
    constexpr auto rf = std::regex_constants::optimize | std::regex_constants::nosubs;
#endif
    std::basic_regex<CharT> expr(regex, rf);
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());
    size_t start = 0;
    for (auto offset : offsets) {
        std::regex_iterator<BidirIt> it(text.begin() + start, text.begin() + start + offset, expr);
        std::regex_iterator<BidirIt> end;
        int64_t start_idx = 0;
        while (it != end) {
            auto match = *it;
            if (match.position() > start_idx) bpe_offsets.emplace_back(match.position() - start_idx);
            bpe_offsets.emplace_back(match.length());
            start_idx = match.position() + match.length();
            ++it;
        }
        if (start_idx < (int64_t) offset) bpe_offsets.emplace_back(offset - start_idx);
        start += offset;
    }
    return bpe_offsets;
}

// ── Public interface ──────────────────────────────────────────────────────────

std::string gc_cpt_to_utf8(uint32_t cpt) {
    std::string result;
    if (cpt <= 0x7f)    { result.push_back(cpt); return result; }
    if (cpt <= 0x7ff)   { result.push_back(0xc0 | ((cpt >> 6) & 0x1f)); result.push_back(0x80 | (cpt & 0x3f)); return result; }
    if (cpt <= 0xffff)  { result.push_back(0xe0 | ((cpt >> 12) & 0x0f)); result.push_back(0x80 | ((cpt >> 6) & 0x3f)); result.push_back(0x80 | (cpt & 0x3f)); return result; }
    if (cpt <= 0x10ffff){ result.push_back(0xf0 | ((cpt >> 18) & 0x07)); result.push_back(0x80 | ((cpt >> 12) & 0x3f)); result.push_back(0x80 | ((cpt >> 6) & 0x3f)); result.push_back(0x80 | (cpt & 0x3f)); return result; }
    throw std::invalid_argument("invalid codepoint");
}

std::vector<uint32_t> gc_cpts_normalize_nfd(const std::vector<uint32_t> & cpts) {
    auto comp = [](const uint32_t cpt, const gc_range_nfd_t & range) { return cpt < range.first; };
    std::vector<uint32_t> result(cpts.size());
    for (size_t i = 0; i < cpts.size(); ++i) {
        const uint32_t cpt = cpts[i];
        auto it = std::upper_bound(gc_unicode_ranges_nfd.begin(), gc_unicode_ranges_nfd.end(), cpt, comp) - 1;
        result[i] = (it->first <= cpt && cpt <= it->last) ? it->nfd : cpt;
    }
    return result;
}

std::vector<uint32_t> gc_cpts_from_utf8(const std::string & utf8) {
    std::vector<uint32_t> result;
    result.reserve(utf8.size());
    size_t offset = 0;
    while (offset < utf8.size()) {
        try {
            result.push_back(gc_cpt_from_utf8(utf8, offset));
        } catch (const std::invalid_argument &) {
            ++offset;
            result.emplace_back(0xFFFD);
        }
    }
    return result;
}

gc_cpt_flags_t gc_cpt_flags(const uint32_t cpt) {
    static const gc_cpt_flags_t undef(gc_cpt_flags_t::UNDEFINED);
    static const auto cpt_flags = gc__cpt_flags_array();
    return cpt < cpt_flags.size() ? cpt_flags[cpt] : undef;
}

gc_cpt_flags_t gc_cpt_flags_utf8(const std::string & utf8) {
    static const gc_cpt_flags_t undef(gc_cpt_flags_t::UNDEFINED);
    if (utf8.empty()) return undef;
    size_t offset = 0;
    return gc_cpt_flags(gc_cpt_from_utf8(utf8, offset));
}

std::string gc_byte_to_utf8(uint8_t byte) {
    static std::unordered_map<uint8_t, std::string> map = gc__byte_to_utf8_map();
    return map.at(byte);
}

uint8_t gc_utf8_to_byte(const std::string & utf8) {
    static std::unordered_map<std::string, uint8_t> map = gc__utf8_to_byte_map();
    return map.at(utf8);
}

uint32_t gc_tolower(uint32_t cpt) {
    auto it = std::lower_bound(gc_unicode_map_lowercase.begin(), gc_unicode_map_lowercase.end(), cpt,
        [](const std::pair<uint32_t,uint32_t> & p, uint32_t v) { return p.first < v; });
    if (it != gc_unicode_map_lowercase.end() && it->first == cpt) return it->second;
    return cpt;
}

bool gc_cpt_is_han(uint32_t cpt) {
    return (cpt >= 0x4E00 && cpt <= 0x9FFF)
        || (cpt >= 0x3400 && cpt <= 0x4DBF)
        || (cpt >= 0x20000 && cpt <= 0x2A6DF)
        || (cpt >= 0x2A700 && cpt <= 0x2B73F)
        || (cpt >= 0x2B740 && cpt <= 0x2B81F)
        || (cpt >= 0x2B820 && cpt <= 0x2CEAF)
        || (cpt >= 0x2CEB0 && cpt <= 0x2EBEF)
        || (cpt >= 0xF900  && cpt <= 0xFAFF)
        || (cpt >= 0x2F800 && cpt <= 0x2FA1F);
}

std::vector<std::string> gc_regex_split(const std::string & text, const std::vector<std::string> & regex_exprs, bool byte_encode) {
    static const std::map<std::string, int> k_ucat_enum = {
        { "\\p{N}", gc_cpt_flags_t::NUMBER },
        { "\\p{L}", gc_cpt_flags_t::LETTER },
        { "\\p{P}", gc_cpt_flags_t::PUNCTUATION },
        { "\\p{M}", gc_cpt_flags_t::ACCENT_MARK },
        { "\\p{S}", gc_cpt_flags_t::SYMBOL },
        { "\\p{Lu}", gc_cpt_flags_t::LETTER },
        { "\\p{Ll}", gc_cpt_flags_t::LETTER },
        { "\\p{Lt}", gc_cpt_flags_t::LETTER },
        { "\\p{Lm}", gc_cpt_flags_t::LETTER },
        { "\\p{Lo}", gc_cpt_flags_t::LETTER },
    };
    static const std::map<int, int> k_ucat_cpt = {
        { gc_cpt_flags_t::NUMBER,      0xD1 },
        { gc_cpt_flags_t::LETTER,      0xD2 },
        { gc_cpt_flags_t::PUNCTUATION, 0xD3 },
        { gc_cpt_flags_t::ACCENT_MARK, 0xD4 },
        { gc_cpt_flags_t::SYMBOL,      0xD5 },
    };
    static const std::map<int, std::string> k_ucat_map = {
        { gc_cpt_flags_t::NUMBER,      "\x30-\x39" },
        { gc_cpt_flags_t::LETTER,      "\x41-\x5A\x61-\x7A" },
        { gc_cpt_flags_t::PUNCTUATION, "\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\\\x5B-\\\x5D\x5F\\\x7B\\\x7D" },
        { gc_cpt_flags_t::ACCENT_MARK, "" },
        { gc_cpt_flags_t::SYMBOL,      "\\\x24\\\x2B\x3C-\x3E\x5E\x60\\\x7C" },
    };

    bool need_collapse = false;
    for (const auto & expr : regex_exprs) {
        for (const auto & ucat : k_ucat_enum) {
            if (std::string::npos != expr.find(ucat.first)) { need_collapse = true; break; }
        }
    }

    const auto cpts = gc_cpts_from_utf8(text);
    std::string text_collapsed;
    if (need_collapse) {
        text_collapsed.resize(cpts.size());
        for (size_t i = 0; i < cpts.size(); ++i) {
            if (cpts[i] < 128) { text_collapsed[i] = cpts[i]; continue; }
            const auto flags = gc_cpt_flags(cpts[i]);
            if (flags.is_whitespace) { text_collapsed[i] = (char) 0x0B; }
            else if (k_ucat_cpt.find(flags.category_flag()) != k_ucat_cpt.end()) { text_collapsed[i] = k_ucat_cpt.at(flags.category_flag()); }
            else { text_collapsed[i] = (char) 0xD0; }
        }
    }

    std::vector<size_t> bpe_offsets = { cpts.size() };

    for (const auto & expr : regex_exprs) {
        auto tmp = gc__regex_split_custom(text, expr, bpe_offsets);
        if (!tmp.empty()) { bpe_offsets = std::move(tmp); continue; }

        try {
            bool use_collapsed = false;
            for (const auto & ucat : k_ucat_enum) {
                if (std::string::npos != expr.find(ucat.first)) { use_collapsed = true; break; }
            }
            const auto cpts_re = gc_cpts_from_utf8(expr);
            if (use_collapsed) {
                for (size_t i = 0; i < cpts_re.size(); ++i) {
                    if (cpts_re[i] >= 128) throw std::runtime_error("Regex has unicode categories + non-ASCII — unsupported");
                }
                std::string expr_collapsed;
                bool inside = false;
                for (size_t i = 0; i < expr.size(); ++i) {
                    if (expr[i] == '[' && (i == 0 || expr[i-1] != '\\')) { expr_collapsed += '['; inside = true; continue; }
                    if (inside && expr[i] == ']' && expr[i-1] != '\\') { expr_collapsed += ']'; inside = false; continue; }
                    if (expr[i+0] == '\\' && i+3 < expr.size() && expr[i+1] == 'p' && expr[i+2] == '{') {
                        size_t cb = expr.find('}', i+3);
                        if (cb != std::string::npos && cb <= i+10) {
                            const std::string pat = expr.substr(i, cb-i+1);
                            if (k_ucat_enum.find(pat) != k_ucat_enum.end()) {
                                if (!inside) expr_collapsed += '[';
                                expr_collapsed += k_ucat_cpt.at(k_ucat_enum.at(pat));
                                expr_collapsed += k_ucat_map.at(k_ucat_enum.at(pat));
                                if (!inside) expr_collapsed += ']';
                                i = cb; continue;
                            }
                        }
                    }
                    expr_collapsed += expr[i];
                }
                bpe_offsets = gc__regex_split_stl(text_collapsed, expr_collapsed, bpe_offsets);
            } else {
                std::wstring wexpr(cpts_re.begin(), cpts_re.end());
                std::wstring wtext(cpts.begin(), cpts.end());
                for (size_t i = 0; i < wtext.size(); ++i) {
                    if (wtext[i] > 0x7F && gc_cpt_flags(wtext[i]).is_whitespace) wtext[i] = 0x0B;
                }
                bpe_offsets = gc__regex_split_stl(wtext, wexpr, bpe_offsets);
            }
        } catch (std::regex_error & e) {
            GC_LOG_ERR("gc_regex_split: failed for regex '%s': %s", expr.c_str(), e.what());
            throw std::runtime_error("gc_regex_split failed");
        }
    }

    std::vector<std::string> bpe_words;
    bpe_words.reserve(bpe_offsets.size());
    size_t s = 0;
    for (size_t & off : bpe_offsets) {
        bpe_words.emplace_back();
        for (size_t i = s; i < s + off; ++i) bpe_words.back() += gc_cpt_to_utf8(cpts[i]);
        s += off;
    }

    if (byte_encode) return gc__byte_encoding_process(bpe_words);
    return bpe_words;
}
