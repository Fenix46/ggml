#pragma once

// ── gc_debug.h — Tiered debug/trace infrastructure ────────────────────────────
//
// Usage:
//   GC_DEBUG_LEVEL=0  silent (production default)
//   GC_DEBUG_LEVEL=1  errors only
//   GC_DEBUG_LEVEL=2  + scheduler step summaries
//   GC_DEBUG_LEVEL=3  + per-request state transitions
//   GC_DEBUG_LEVEL=4  + KV alloc/free traces
//   GC_DEBUG_LEVEL=5  + graph execution (logits, sampling)
//
// Set at runtime:
//   gc_debug_set_level(3);
//
// Or at compile time:
//   -DGC_DEBUG_LEVEL=3
//
// All output goes to stderr.  Zero overhead when level is below threshold.
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <cstdarg>

// ── Compile-time ceiling ──────────────────────────────────────────────────────
// Define GC_DEBUG_MAX_LEVEL at compile time to strip all logs above it.
// Default: allow all levels (5).
#ifndef GC_DEBUG_MAX_LEVEL
#define GC_DEBUG_MAX_LEVEL 5
#endif

// ── Subsystem tags ────────────────────────────────────────────────────────────
#define GC_TRACE_SCHED    "sched"
#define GC_TRACE_KV       "kv"
#define GC_TRACE_ALLOC    "alloc"
#define GC_TRACE_REQ      "req"
#define GC_TRACE_GRAPH    "graph"
#define GC_TRACE_SAMPLE   "sample"
#define GC_TRACE_ENGINE   "engine"

// ── Runtime level ─────────────────────────────────────────────────────────────

// Get / set the runtime debug level (thread-unsafe; must be set before first use).
int  gc_debug_level();
void gc_debug_set_level(int level);

// Low-level log: always emits regardless of level checks.
// Internal — prefer the macros below.
void gc_debug_log(int level, const char * tag, const char * fmt, ...)
    __attribute__((format(printf, 3, 4)));

// ── Level-guarded macros ──────────────────────────────────────────────────────
// Each macro checks compile-time ceiling first (free), then runtime level.

#define GC_LOG(lvl, tag, ...) do { \
    if constexpr ((lvl) <= GC_DEBUG_MAX_LEVEL) { \
        if (gc_debug_level() >= (lvl)) gc_debug_log((lvl), (tag), __VA_ARGS__); \
    } \
} while (0)

// Convenience aliases
#define GC_LOG_ERR(...)    GC_LOG(1, GC_TRACE_ENGINE, __VA_ARGS__)
#define GC_LOG_SCHED(...)  GC_LOG(2, GC_TRACE_SCHED,  __VA_ARGS__)
#define GC_LOG_REQ(...)    GC_LOG(3, GC_TRACE_REQ,    __VA_ARGS__)
#define GC_LOG_KV(...)     GC_LOG(4, GC_TRACE_KV,     __VA_ARGS__)
#define GC_LOG_GRAPH(...)  GC_LOG(5, GC_TRACE_GRAPH,  __VA_ARGS__)
