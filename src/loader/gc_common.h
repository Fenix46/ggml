#pragma once

#include "ggml.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string>

// ── Status codes ──────────────────────────────────────────────────────────────
// Use the canonical gc_status_t from the public header.
#include <gc/gc_status.h>  // GC_OK, GC_ERR_*, etc.

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
