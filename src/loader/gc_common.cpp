#include "gc_common.h"

#include <cstdio>
#include <cstdarg>
#include <cstring>

// ── Logging ───────────────────────────────────────────────────────────────────

static gc_log_callback_t s_log_cb   = nullptr;
static void *            s_log_ud   = nullptr;

void gc_log_set_callback(gc_log_callback_t cb, void * user_data) {
    s_log_cb = cb;
    s_log_ud = user_data;
}

void gc__log(ggml_log_level level, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buf[2048];
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (s_log_cb) {
        s_log_cb(level, buf, s_log_ud);
    } else {
        FILE * out = (level == GGML_LOG_LEVEL_ERROR) ? stderr : stdout;
        fputs(buf, out);
        fputc('\n', out);
    }
}

// ── String helpers ────────────────────────────────────────────────────────────

std::string gc__format(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    va_list args2;
    va_copy(args2, args);
    int n = vsnprintf(nullptr, 0, fmt, args);
    va_end(args);
    std::string result(n, '\0');
    vsnprintf(result.data(), n + 1, fmt, args2);
    va_end(args2);
    return result;
}
