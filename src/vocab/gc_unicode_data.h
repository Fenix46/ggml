#pragma once

#include "gc_unicode.h"

#include <cstdint>
#include <initializer_list>
#include <unordered_set>
#include <utility>
#include <vector>

struct gc_range_nfd_t {
    uint32_t first;
    uint32_t last;
    uint32_t nfd;
};

static const uint32_t GC_MAX_CODEPOINTS = 0x110000;

extern const std::initializer_list<std::pair<uint32_t, uint16_t>> gc_unicode_ranges_flags;
extern const std::unordered_set<uint32_t>                          gc_unicode_set_whitespace;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> gc_unicode_map_lowercase;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> gc_unicode_map_uppercase;
extern const std::initializer_list<gc_range_nfd_t>                gc_unicode_ranges_nfd;
