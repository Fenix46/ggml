# GGUF.CORE — Development Plan

## Vision

Build a production-grade LLM inference engine on top of the GGML tensor graph engine.
Name: **GGUF.CORE**. Language: C/C++17. No Python runtime dependency.

Source strategy:
- **Port from llama.cpp**: GGUF loader, tokenizer, chat template, architecture definitions, RoPE
- **Port from vllm (v1 engine)**: paged KV cache, block pool, scheduler, request queue, output processor
- **Original**: C++ server (OpenAI-compatible REST API via a header-only HTTP lib)
- All ported code **renamed** under the `gc_` / `GC_` / `GcXxx` namespace. No `llama_` or `vllm` symbols survive in the public API.

---

## Directory Layout

```
gguf.core/
├── include/
│   ├── gc_model.h          # public model handle
│   ├── gc_tokenizer.h      # public tokenizer API
│   ├── gc_inference.h      # public inference/generation API
│   └── gc_server.h         # public server API
├── src/
│   ├── loader/             # Phase 1 — GGUF model loading
│   │   ├── gc_gguf_loader.cpp/.h
│   │   ├── gc_mmap.cpp/.h
│   │   └── gc_io.cpp/.h
│   ├── arch/               # Phase 2 — architecture registry
│   │   ├── gc_arch.cpp/.h
│   │   ├── gc_hparams.cpp/.h
│   │   └── gc_model.cpp/.h
│   ├── vocab/              # Phase 3 — tokenizer
│   │   ├── gc_vocab.cpp/.h
│   │   ├── gc_unicode.cpp/.h
│   │   └── gc_unicode_data.cpp/.h
│   ├── chat/               # Phase 4 — chat templates
│   │   └── gc_chat.cpp/.h
│   ├── attention/          # Phase 5 — RoPE + attention
│   │   ├── gc_rope.cpp/.h
│   │   └── gc_attention.cpp/.h
│   ├── kvcache/            # Phase 6 — paged KV cache
│   │   ├── gc_block_pool.cpp/.h
│   │   ├── gc_kv_cache_manager.cpp/.h
│   │   ├── gc_kv_cache_utils.cpp/.h
│   │   └── gc_kv_cache_metrics.cpp/.h
│   ├── scheduler/          # Phase 7 — request scheduler
│   │   ├── gc_request.cpp/.h
│   │   ├── gc_request_queue.cpp/.h
│   │   ├── gc_scheduler.cpp/.h
│   │   └── gc_scheduler_output.cpp/.h
│   ├── engine/             # Phase 8 — inference engine
│   │   ├── gc_engine.cpp/.h
│   │   ├── gc_worker.cpp/.h
│   │   ├── gc_batch.cpp/.h
│   │   └── gc_output_processor.cpp/.h
│   └── server/             # Phase 9 — HTTP server
│       ├── gc_server.cpp/.h
│       ├── gc_api_openai.cpp/.h
│       └── gc_api_types.cpp/.h
├── tests/
├── examples/
├── CMakeLists.txt
├── PLAN.md
└── AGENTS.md
```

---

## Phases

### Phase 1 — GGUF Model Loader
**Source**: `llama-original/src/llama-model-loader.{cpp,h}`, `llama-mmap.{cpp,h}`, `llama-io.{cpp,h}`

Goals:
- Read `.gguf` file format: header, metadata KV, tensor index, tensor data
- Memory-mapped loading (mmap) with optional preload
- Quantization-aware tensor type registry (all GGML quant types)
- API: `gc_model_loader_create()`, `gc_model_loader_load()`, `gc_model_loader_free()`

Rename map:
```
llama_model_loader   → gc_model_loader
llama_mmap           → gc_mmap
llama_file           → gc_file
llm_kv               → gc_kv
```

---

### Phase 2 — Architecture Registry
**Source**: `llama-original/src/llama-arch.{cpp,h}`, `llama-hparams.{cpp,h}`, `llama-model.{cpp,h}`

Goals:
- Enumerate all supported `llm_arch` values (Llama, Qwen, Gemma, Mistral, Phi, etc.)
- Map GGUF metadata keys to hyperparameter structs
- Tensor name → role mapping per architecture
- API: `gc_arch_from_string()`, `gc_hparams_load()`, `gc_model_build_graph()`

Rename map:
```
llm_arch             → gc_arch
llama_hparams        → gc_hparams
llama_model          → gc_model_t
llm_build_context    → gc_graph_ctx
```

---

### Phase 3 — Tokenizer
**Source**: `llama-original/src/llama-vocab.{cpp,h}`, `unicode.{cpp,h}`, `unicode-data.{cpp,h}`

Goals:
- BPE, SentencePiece, WordPiece tokenizers (per-model selection from GGUF metadata)
- Encode: `string → []token_id`
- Decode: `[]token_id → string`
- Special token handling (BOS, EOS, PAD, UNK, system, tool)
- API: `gc_tokenizer_create()`, `gc_tokenize()`, `gc_detokenize()`, `gc_tokenizer_free()`

Rename map:
```
llama_vocab          → gc_vocab
llama_token          → gc_token_id
llm_tokenizer_*      → gc_tokenizer_*
```

---

### Phase 4 — Chat Templates
**Source**: `llama-original/src/llama-chat.{cpp,h}`

Goals:
- Jinja2-subset template engine (already in llama.cpp)
- Apply chat template to `[]gc_chat_message` → formatted prompt string
- Template auto-detection from GGUF metadata (`tokenizer.chat_template`)
- API: `gc_chat_apply_template()`, `gc_chat_format_single()`

Rename map:
```
llama_chat_message   → gc_chat_message
llama_chat_apply_template → gc_chat_apply_template
```

---

### Phase 5 — RoPE + Attention
**Source**: `llama-original/src/llama-model.cpp` (graph builders), `llama-graph.{cpp,h}`

Goals:
- RoPE variants: standard, NTK, YaRN, Llama3, Qwen3, Gemma3, etc.
- GQA (grouped-query attention) support
- Sliding window attention
- Build ggml subgraph for attention block: Q/K/V projections, RoPE, scaled dot-product, softmax, output proj
- API: `gc_rope_apply()`, `gc_attention_build_graph()`

Rename map:
```
llm_build_kv_cache_upd → gc_kv_cache_update
llm_build_attn         → gc_attention_build
ggml_rope_ext          → kept as-is (ggml internal)
```

---

### Phase 6 — Paged KV Cache
**Source**: `vllm/vllm/v1/core/block_pool.py`, `kv_cache_manager.py`, `kv_cache_utils.py`, `kv_cache_metrics.py`

Goals:
- Fixed-size block pool: `gc_block_pool_t` — alloc/free physical KV blocks
- Per-sequence block table: maps logical block index → physical block id
- Prefix caching (hash-based reuse of KV blocks for common prefixes)
- Eviction policy: LRU on free-list
- Metrics: hit rate, fragmentation, utilization
- API: `gc_block_pool_create()`, `gc_kv_alloc()`, `gc_kv_free()`, `gc_kv_prefix_match()`

Port strategy: translate Python class methods to C++ structs + free functions. Use `std::vector` for block tables, `std::unordered_map` for prefix hash → block_id.

Key types:
```c
typedef struct gc_block_t       { int32_t id; uint64_t content_hash; bool is_free; } gc_block_t;
typedef struct gc_block_table_t { int32_t *block_ids; int32_t n_blocks; }            gc_block_table_t;
typedef struct gc_block_pool_t  { /* ... */ }                                         gc_block_pool_t;
typedef struct gc_kv_manager_t  { /* ... */ }                                         gc_kv_manager_t;
```

---

### Phase 7 — Request Scheduler
**Source**: `vllm/vllm/v1/core/sched/scheduler.py`, `request_queue.py`, `output.py`, `interface.py`

Goals:
- Request lifecycle: `WAITING → RUNNING → PREEMPTED → FINISHED`
- Priority queue with FCFS default, optional priority field
- Chunked prefill: split long prompts across multiple steps
- Preemption: swap out low-priority sequences when KV budget exhausted
- Continuous batching: fill decode slots with new prefill tokens each step
- Output: `gc_scheduler_output_t` — which sequences run this step, new tokens, finished flags
- API: `gc_scheduler_create()`, `gc_scheduler_add_request()`, `gc_scheduler_step()`, `gc_scheduler_free()`

Key types:
```c
typedef enum   gc_req_status_t  { GC_REQ_WAITING, GC_REQ_RUNNING, GC_REQ_PREEMPTED, GC_REQ_DONE } gc_req_status_t;
typedef struct gc_request_t     { uint64_t id; int32_t *prompt_tokens; int32_t n_prompt; /* ... */ } gc_request_t;
typedef struct gc_sched_output_t { gc_request_t **running; int32_t n_running; /* ... */ }           gc_sched_output_t;
```

---

### Phase 8 — Inference Engine
**Source**: `vllm/vllm/v1/engine/core.py`, `worker_base.py`, `gpu_model_runner.py`, `output_processor.py`

Goals:
- Drive the forward pass loop: scheduler → batch builder → ggml graph execute → output processor
- `gc_engine_t` owns: model, tokenizer, kv manager, scheduler, worker thread pool
- Synchronous and async (callback-based) generation modes
- Streaming token delivery per-request
- Sampling: greedy, top-k, top-p, temperature, repetition penalty (port from llama-sampler.cpp)
- API: `gc_engine_create()`, `gc_engine_submit()`, `gc_engine_step()`, `gc_engine_free()`

---

### Phase 9 — OpenAI-Compatible HTTP Server
**Source**: `vllm/vllm/entrypoints/openai/api_server.py` (design reference only — full C++ rewrite)

Goals:
- Endpoints: `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`, `GET /health`
- SSE streaming for `stream: true`
- JSON request/response types matching OpenAI spec
- HTTP library: [cpp-httplib](https://github.com/yhirose/cpp-httplib) (header-only, zero dependencies)
- JSON library: [nlohmann/json](https://github.com/nlohmann/json) (header-only)
- Thread pool: one thread per connection, dispatch to `gc_engine_submit()`
- API: `gc_server_create()`, `gc_server_start()`, `gc_server_stop()`

---

## Build System

CMake 3.21+. The existing GGML `CMakeLists.txt` is the base — `add_subdirectory(ggml)` and link `ggml`.

```
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGC_BACKEND=metal   # macOS
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGC_BACKEND=cuda    # NVIDIA
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGC_BACKEND=cpu     # CPU-only
```

---

## Milestone Schedule

| Milestone | Phases | Deliverable |
|-----------|--------|-------------|
| M1 | 1–2 | Load any GGUF, inspect metadata and tensors |
| M2 | 3–4 | Tokenize + apply chat template, round-trip test |
| M3 | 5 | Single-sequence greedy decode (no paging) |
| M4 | 6–7 | Paged KV + scheduler, multi-sequence batching |
| M5 | 8 | Engine loop, streaming output, sampling |
| M6 | 9 | OpenAI-compatible server, curl-testable |

---

## Namespace & Naming Conventions

- Public C API: `gc_` prefix, snake_case (e.g. `gc_engine_create`)
- Public C++ classes: `Gc` prefix, PascalCase (e.g. `GcEngine`)
- Internal symbols: `gc__` double-underscore prefix
- Constants/enums: `GC_` prefix, SCREAMING_SNAKE_CASE
- No `llama_`, `vllm`, `gguf_` symbols in public headers (GGML internal types are OK)

---

## Out of Scope (v1)

- Multi-GPU tensor parallelism (vllm distributed worker)
- LoRA adapters
- Speculative decoding
- Multimodal (vision encoders)
- Python bindings
