#include "gc_debug.h"

#include <cstdarg>
#include <cstdio>

static int g_gc_debug_level = 0;

int gc_debug_level() {
    return g_gc_debug_level;
}

void gc_debug_set_level(int level) {
    g_gc_debug_level = level;
}

void gc_debug_log(int level, const char * tag, const char * fmt, ...) {
    (void)level;
    fprintf(stderr, "[gc/%s] ", tag ? tag : "?");
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}
