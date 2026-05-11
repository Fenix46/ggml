# GGUF.CORE — Agent Guidelines

## Project Identity

Name: **GGUF.CORE**  
Engine: GGML (this repo) as the tensor graph backend  
Language: C/C++17  
Source references: `llama-original/` and `vllm/` (read-only — never modify them)

---

## Cardinal Rules

1. **No `llama_` symbols in public API.** All ported code is renamed under `gc_` / `GC_` / `GcXxx`. Violation = broken contract.
2. **No Python at runtime.** Server, engine, scheduler: pure C/C++.
3. **GGML is the only tensor library.** No PyTorch, no ONNX, no custom kernels outside GGML backends.
4. **Source directories are read-only.** `llama-original/` and `vllm/` are reference only. Write only inside `gguf.core/` (or the ggml root if adding a subdir).
5. **One phase at a time.** Do not start Phase N+1 until Phase N has a passing test.

---

## Porting Protocol

When porting a file from llama-original or vllm:

1. **Read the source** — understand data flow, not just syntax.
2. **Strip Python/C++ idioms that don't carry over** — no `self`, no `Optional[X]`, no decorators.
3. **Rename everything** according to the rename map in PLAN.md.
4. **Write the `.h` first** — define the public struct/function surface before any `.cpp`.
5. **Write a minimal test** in `tests/` that exercises the new unit before moving on.
6. **Never copy-paste llama.cpp comment blocks verbatim** — rewrite or omit. Avoid attribution trails in source.

---

## Naming Rules

| Context | Convention | Example |
|---------|-----------|---------|
| Public C API function | `gc_` + snake_case | `gc_tokenizer_create` |
| Public C++ class | `Gc` + PascalCase | `GcScheduler` |
| Internal function | `gc__` + snake_case | `gc__block_hash` |
| Enum type | `gc_` + snake_case + `_t` | `gc_req_status_t` |
| Enum value | `GC_` + SCREAMING_SNAKE | `GC_REQ_WAITING` |
| Struct type | `gc_` + snake_case + `_t` | `gc_block_pool_t` |
| Macro/constant | `GC_` + SCREAMING_SNAKE | `GC_MAX_BATCH_SIZE` |

---

## File Layout Rules

- One logical unit per `{name}.cpp` + `{name}.h` pair.
- Headers are self-contained — every header compiles cleanly with `-c`.
- No circular includes. Dependency order: `ggml.h` → loader → arch → vocab → chat → attention → kvcache → scheduler → engine → server.
- Public headers live in `include/`. Internal headers stay in `src/<subsystem>/`.

---

## Code Style

- C++17, no exceptions in hot paths (use return codes or `std::optional`).
- No RTTI.
- Memory: manual `malloc/free` for GGML-adjacent code; `std::unique_ptr` acceptable in engine/server layer.
- No global mutable state outside of the ggml context.
- Error handling: return `gc_status_t` (enum) from all public API functions.
  ```c
  typedef enum gc_status_t {
      GC_OK            = 0,
      GC_ERR_IO        = 1,
      GC_ERR_ALLOC     = 2,
      GC_ERR_INVALID   = 3,
      GC_ERR_UNSUPPORTED = 4,
  } gc_status_t;
  ```
- No `printf` in library code — use a `gc_log_callback` registered at init.

---

## Phase-Specific Guidelines

### Phase 1 — GGUF Loader
- Mirror the GGUF spec from `ggml/docs/gguf.md` exactly.
- mmap path must work on macOS (`MAP_PRIVATE`) and Linux.
- Validate magic bytes and version before any allocation.
- Test: load a small GGUF (e.g., Qwen 0.5B), print all metadata KV and tensor names.

### Phase 2 — Architecture Registry
- The arch enum must cover all architectures in `llama-arch.h` at port time.
- Adding a new arch = add enum value + tensor name map + hparam parser. No other files change.
- Test: for each supported arch, verify all expected tensor names resolve correctly.

### Phase 3 — Tokenizer
- BPE and SentencePiece must both pass round-trip (encode→decode) tests.
- Special tokens must not appear mid-sequence unless explicitly requested.
- Test: tokenize "Hello, world!" and a CJK string for each tokenizer type.

### Phase 4 — Chat Templates
- Template engine must handle `if/else`, `for`, `set`, string filters (`strip`, `upper`, etc.).
- Fail loudly on unknown template syntax rather than silently producing wrong output.
- Test: apply a Llama-3 and a ChatML template, compare output to reference strings.

### Phase 5 — RoPE + Attention
- RoPE variant is selected at runtime from `gc_hparams_t.rope_type`.
- GQA: K/V head count may differ from Q head count — handle `n_kv_heads != n_heads`.
- Test: single forward pass on a 2-layer toy model, check output shape.

### Phase 6 — Paged KV Cache
- Block size is a compile-time constant (`GC_KV_BLOCK_SIZE`, default 16 tokens).
- Prefix hash = FNV-1a over the token IDs of the block's content.
- Evict LRU free blocks first; never evict blocks owned by running sequences.
- Test: fill cache to 90%, trigger eviction, verify LRU order and no use-after-free.

### Phase 7 — Scheduler
- Chunked prefill max tokens per step: configurable via `gc_sched_config_t.max_prefill_tokens`.
- Preemption: recompute (re-prefill) is the default strategy; swap-to-CPU is optional.
- Continuous batching: every `gc_scheduler_step()` call may mix prefill and decode sequences.
- Test: submit 10 requests with varying lengths, verify all complete and outputs are ordered.

### Phase 8 — Engine
- `gc_engine_step()` is the single hot-loop function: scheduler → batch → ggml forward → output.
- Worker threads own ggml contexts; the engine owns the scheduler and kv manager.
- Streaming: deliver tokens via a registered `gc_token_callback` per request.
- Test: generate 100 tokens from a loaded model, verify no memory growth over iterations.

### Phase 9 — Server
- Use cpp-httplib vendored in `src/server/vendor/httplib.h`.
- Use nlohmann/json vendored in `src/server/vendor/json.hpp`.
- SSE streaming: flush after every token chunk.
- Validate all request fields; return 400 with `{"error": {"message": ...}}` on bad input.
- Test: `curl` the `/v1/chat/completions` endpoint, verify streaming and non-streaming responses.

---

## Testing Rules

- Every phase ships with at least one test in `tests/test_gc_<phase>.cpp`.
- Tests link against ggml and the gc subsystems built so far.
- Tests must pass with `-fsanitize=address,undefined` on debug builds.
- No mocking of GGML — use real ggml contexts in tests.

---

## What NOT to Port

Skip these from llama.cpp (out of scope for v1):
- `llama-grammar.*` — constrained decoding
- `llama-adapter.*` — LoRA
- `llama-memory-recurrent.*` — Mamba/recurrent states
- `llama-quant.*` — quantization conversion (we load, not convert)
- Anything under `llama-original/common/` — CLI helpers, not needed

Skip these from vllm:
- Anything under `vllm/distributed/` — no tensor parallelism in v1
- `vllm/lora/` — no LoRA
- `vllm/multimodal/` — no vision
- `vllm/compilation/` — torch.compile, irrelevant
- `vllm/ray/` — no Ray

---

## Commit Messages

Follow Conventional Commits:
```
feat(loader): implement GGUF mmap loader with metadata KV parsing
feat(vocab): add BPE tokenizer with special token support
fix(kvcache): correct LRU eviction when block refcount > 0
refactor(scheduler): extract request_queue into separate translation unit
```

Scope = subsystem directory name (loader, arch, vocab, chat, attention, kvcache, scheduler, engine, server).
