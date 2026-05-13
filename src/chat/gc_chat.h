#pragma once

#include <gc/gc_status.h>

#include <cstdint>
#include <string>
#include <vector>

// ── Chat template engine ─────────────────────────────────────────────────────
// Full Jinja2-subset parser/evaluator.
// Supports: {{ var }}, {{ expr | filter }}, {% if %}, {% for %}, {% set %}.
// The engine owns the compiled template (AST) and can be reused.

struct gc_chat_template_t;
struct gc_chat_message_t;

// ── Lifecycle ─────────────────────────────────────────────────────────────────

gc_status_t gc_chat_create(gc_chat_template_t ** out);
void        gc_chat_free(gc_chat_template_t * t);

// Compile a raw Jinja2 template string into internal AST.
// Returns GC_ERR_UNSUPPORTED if template has syntax errors.
gc_status_t gc_chat_compile(gc_chat_template_t * t, const char * template_str);

// Apply the compiled template with the given messages and context.
//   messages[i] = {role, content}
//   system      = optional system message override (pass "" to ignore)
//   add_ass    = if true, append assistant turn start to output
gc_status_t gc_chat_render(gc_chat_template_t * t,
                           const char * const * roles,
                           const char * const * contents,
                           size_t n_messages,
                           const char * system,
                           bool add_ass,
                           std::string * out);

// Convenience: compile + render in one call (uses default built-in template if
// template_str is nullptr).
gc_status_t gc_chat_apply(const char * template_str,
                          const char * const * roles,
                          const char * const * contents,
                          size_t n_messages,
                          std::string * out);

// ── Diagnostics ───────────────────────────────────────────────────────────────

// Get last compile error (valid until next compile call).
const char * gc_chat_error(gc_chat_template_t * t);
