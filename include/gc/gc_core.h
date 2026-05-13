#pragma once

// ── GGUF.CORE public aggregate header ─────────────────────────────────────────
// Include this to get all public APIs.  Each sub-header delegates to the
// corresponding internal implementation in src/<subsystem>/.
// All public headers are self-contained and use include guards.

#include "gc_status.h"
#include "gc_loader.h"
#include "gc_arch.h"
#include "gc_vocab.h"
#include "gc_chat.h"
#include "gc_attention.h"
#include "gc_kvcache.h"
#include "gc_scheduler.h"
#include "gc_engine.h"
#include "gc_server.h"
