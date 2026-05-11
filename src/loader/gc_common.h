#pragma once

#include "ggml.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string>

// ── Status codes ──────────────────────────────────────────────────────────────

typedef enum gc_status_t {
    GC_OK              = 0,
    GC_ERR_IO          = 1,
    GC_ERR_ALLOC       = 2,
    GC_ERR_INVALID     = 3,
    GC_ERR_UNSUPPORTED = 4,
    GC_ERR_NOT_FOUND   = 5,
    GC_ERR_CORRUPT     = 6,
} gc_status_t;

// ── Logging ───────────────────────────────────────────────────────────────────

typedef void (*gc_log_callback_t)(ggml_log_level level, const char * text, void * user_data);

void gc_log_set_callback(gc_log_callback_t cb, void * user_data);

void gc__log(ggml_log_level level, const char * fmt, ...);

#define GC_LOG_INFO(...)  gc__log(GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define GC_LOG_WARN(...)  gc__log(GGML_LOG_LEVEL_WARN,  __VA_ARGS__)
#define GC_LOG_ERROR(...) gc__log(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GC_LOG_DEBUG(...) gc__log(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)

// ── String helpers ────────────────────────────────────────────────────────────

std::string gc__format(const char * fmt, ...);
