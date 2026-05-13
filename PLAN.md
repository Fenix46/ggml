# GGUF.CORE — Development Plan

## Vision

Build a production-grade LLM inference engine on top of the GGML tensor graph engine.
Name: **GGUF.CORE**. Language: C/C++17. No Python runtime dependency.

Source strategy — **hard boundary**:

| Layer | Source | What we take |
|-------|--------|-------------|
| Loader (Phase 1) | llama.cpp | GGUF file parsing, mmap, KV getters |
| Arch registry (Phase 2) | llama.cpp | Arch enum, hparams struct, tensor name map |
| Tokenizer (Phase 3) | llama.cpp | BPE / SentencePiece / WordPiece logic |
| Chat template (Phase 4) | llama.cpp | Jinja2-subset template engine |
| Attention + RoPE (Phase 5) | llama.cpp | RoPE math, GQA graph builder |
| **KV cache (Phase 6)** | **vllm v1** | **Block pool, KV manager, prefix cache** |
| **Scheduler (Phase 7)** | **vllm v1** | **Continuous batching, chunked prefill, preemption** |
| **Engine (Phase 8)** | **vllm v1** | **Engine core loop, worker, output processor** |
| **Server (Phase 9)** | **vllm v1 + original** | **OpenAI endpoint design, C++ rewrite** |

**From llama.cpp we take ONLY**: file I/O, metadata parsing, tokenizer math, chat template, RoPE coefficients, and GQA graph construction. **Nothing else.**

**We do NOT port from llama.cpp**:
- `llama_kv_cache_*` — replaced entirely by vllm block pool
- `llama_batch` / `llama_ubatch` — replaced by `gc_batch_t` modelled on vllm's `SchedulerOutput`
- `llama_context` / `llama_state` — no concept of a "context slot"; requests are stateless from the engine's perspective, KV state lives in paged blocks
- `llama_decode` / `llama_encode` — replaced by `gc_engine_step()`
- `llama_sampler*` — sampler logic ported independently, not via llama's chain API
- Any KV defrag, sequential slot allocation, or shift logic — vllm's block table makes these unnecessary

All ported code **renamed** under the `gc_` / `GC_` / `GcXxx` namespace. No `llama_` or `vllm` symbols survive in the public API.

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

**This is a direct C++ translation of vllm v1's block allocator — NOT a port of llama.cpp's kv cache.**
No slot arrays, no sequential defrag, no `llama_kv_cache_seq_*`. Pure paged allocation.

Goals:
- Fixed-size block pool: `gc_block_pool_t` — alloc/free physical KV blocks, ref-counted
- Per-sequence block table: maps logical page index → physical block id (vllm `BlockTable`)
- Prefix caching: FNV-1a hash of token content → reuse matching blocks across requests
- Eviction: LRU free-list, never evict ref'd blocks
- Metrics: hit rate, fragmentation, utilization (`gc_kv_cache_metrics_t`)
- API: `gc_block_pool_create()`, `gc_kv_alloc()`, `gc_kv_free()`, `gc_kv_prefix_match()`

Port map:
```
BlockPool             → gc_block_pool_t
KVCacheManager        → gc_kv_manager_t
BlockTable            → std::vector<int32_t>  (logical → physical block id)
PrefixCache           → std::unordered_map<uint64_t, int32_t>  (hash → block_id)
```

Key types:
```c
typedef struct gc_block_t       { int32_t id; uint64_t content_hash; int32_t ref_count; } gc_block_t;
typedef struct gc_block_pool_t  { /* total_blocks, free_list, hash map */ }               gc_block_pool_t;
typedef struct gc_kv_manager_t  { gc_block_pool_t *pool; /* per-seq block tables */ }     gc_kv_manager_t;
```

---

### Phase 7 — Request Scheduler
**Source**: `vllm/vllm/v1/core/sched/scheduler.py`, `request_queue.py`, `output.py`, `interface.py`

**Direct C++ translation of vllm v1 Scheduler — NOT llama.cpp's decode/batch loop.**
No `llama_batch`, no `llama_ubatch`. The scheduler owns the step budget, not the model runner.

Goals:
- Request lifecycle: `WAITING → RUNNING → PREEMPTED → FINISHED` (mirrors vllm `SequenceStatus`)
- Priority queue: FCFS default, optional priority field per request
- Chunked prefill: configurable `max_num_batched_tokens` split across steps
- Preemption: recompute strategy (re-prefill evicted sequences) as default; swap-to-CPU optional
- Continuous batching: every `gc_scheduler_step()` mixes prefill + decode sequences in one batch
- Output: `gc_sched_output_t` — running seqs, new token slots, block table updates, finished flags

Port map:
```
Scheduler             → gc_scheduler_t
SchedulerOutput       → gc_sched_output_t
Request               → gc_request_t
RequestQueue          → gc_request_queue_t  (priority deque)
SequenceStatus        → gc_req_status_t
```

Key types:
```c
typedef enum gc_req_status_t { GC_REQ_WAITING, GC_REQ_RUNNING, GC_REQ_PREEMPTED, GC_REQ_DONE } gc_req_status_t;
typedef struct gc_request_t  { uint64_t id; gc_req_status_t status; int32_t *tokens; int32_t n_tokens; /* sampling params */ } gc_request_t;
typedef struct gc_sched_output_t { gc_request_t **prefill_reqs; int32_t n_prefill;
                                   gc_request_t **decode_reqs;  int32_t n_decode;
                                   gc_request_t **finished_reqs; int32_t n_finished; } gc_sched_output_t;
```

---

### Phase 8 — Inference Engine
**Source**: `vllm/vllm/v1/engine/core.py`, `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/engine/output_processor.py`

**Modelled on vllm v1 EngineCore — NOT llama.cpp's llama_decode() loop.**
Forward pass driven by `gc_sched_output_t`, not by caller-managed batches.

Goals:
- Hot loop: `gc_engine_step()` = scheduler.step() → batch_builder → ggml_graph_compute() → output_processor
- `gc_engine_t` owns: loaded model tensors, tokenizer, kv manager, scheduler, ggml backend contexts
- Batch builder (`gc_batch_t`): assembles token ids + position ids + block tables from `gc_sched_output_t`
- Output processor: logit extraction → sampling → token delivery via `gc_token_callback_t` per request
- Sampling ops: greedy, top-k, top-p, temperature, repetition penalty (ported independently, not via llama sampler chain)
- Streaming: per-token callback fired immediately after sampling, before next step
- API: `gc_engine_create()`, `gc_engine_submit()`, `gc_engine_step()`, `gc_engine_cancel()`, `gc_engine_free()`

Port map:
```
EngineCore            → gc_engine_t
GPUModelRunner        → gc_model_runner_t  (drives ggml graph)
ModelInputForGPUWithSamplingMetadata → gc_batch_t
OutputProcessor       → gc_output_processor_t
SamplerOutput         → gc_sample_result_t
```

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

| Milestone | Phases | Deliverable | Stato |
|:---------:|:-----:|------------|:----:|
| M1 | 1–2 | Load any GGUF, inspect metadata and tensors | ✅ 55 test pass |
| M2 | 3–4 | Tokenize + apply chat template, round-trip test | ⚠️ Tokenizer OK (vocab 0+), chat 10/12 fail |
| M3 | 5 | Single-sequence greedy decode (no paging) | ❌ Da testare |
| M4 | 6–7 | Paged KV + scheduler, multi-sequence batching | ✅ 144 test pass |
| M5 | 8 | Engine loop, streaming output, sampling | ✅ 56 test (mock runner) |
| M6 | 9 | OpenAI-compatible server, curl-testable | ⚠️ Compila, chat rotta blocca prompt |

---

# Piano Esecutivo — Task Sequenziali

Ogni task deve essere completato e verificato (test pass) prima di passare al successivo.
Flaggo ogni task come `[x]` quando è pronto.

## Fase A — Graph Engine Ristrutturato (per-arch forward builders)

Ristrutturare `gc_graph_runner.cpp` (901 linee, monolitico) in builder separati per architettura,
seguendo il pattern di `llama-original/src/models/` e il coordinator pattern di vllm `GPUModelRunner`.

Nuova struttura:
```
src/graph/
├── gc_graph.h                 # Interfaccia base: GcGraphBuilder
├── gc_graph_llama.cpp/.h      # Forward per Llama arch
├── gc_graph_gemma.cpp/.h      # Forward per Gemma arch
├── gc_graph_qwen2.cpp/.h      # Forward per Qwen2 arch
├── gc_graph_runner.cpp        # Coordinatore (seleziona builder, pesi, KV, esecuzione)
├── gc_graph_runner.h
└── CMakeLists.txt
```

- [x] **A1**: Creare `src/graph/` directory, `gc_graph.h` con interfaccia astratta `GcGraphBuilder`.
- [x] **A2**: Estrarre builder Llama da `gc_graph_runner.cpp` → `gc_graph_llama.cpp/.h` (294 linee).
- [x] **A3**: Refactor `gc_graph_runner.cpp` (da 901→644 linee) come coordinatore: dispatcher via switch(arch) sul builder giusto, peso view in lambda inline.
- [ ] **A4**: Copiare modello Llama-3.2-1B in `models/` e testare forward pass reale.
- [ ] **A5**: Verificare output logits non degeneri e memory growth zero.
- [ ] **A6**: Aggiungere `test_gc_graph` a CTest.

## Fase B — Fix Chat Template Engine

- [ ] **B1**: Diagnosticare `test_gc_chat` fallimenti. `{{ var }}` senza contesto produce stringa vuota (corretto), ma messaggi e loop `{% for %}` non funzionano.
- [ ] **B2**: Fixare risoluzione variabili nel renderer: `messages`, `msg.role`, `msg.content` devono funzionare.
- [ ] **B3**: Fixare filtri: `| strip`, `| upper` devono modificare il testo.
- [ ] **B4**: Tutti i 12 test in `test_gc_chat.cpp` devono passare.

## Fase C — Server Funzionante

- [x] **C1**: Server compila, `test_gc_server` passa (36 check, mock runtime).
- [x] **C2**: `/v1/chat/completions` risponde con JSON valido nel test.
- [x] **C3**: SSE streaming funziona nel test.
- [x] **C4**: Error handling: richieste malformate → 400 con messaggio di errore.
- [ ] **C5**: `gc_server_main` con modello reale: funziona ma output degenere (stesso token ripetuto). Da investigare: attenzione/RoPE/KV cache.

## Fase D — Port Forward Pass Reale (gc_graph_runner_t)

- [ ] **D1**: Verificare che `gc_graph_runner_t` con pesi reali da modello GGUF produca logits non degenere.
- [ ] **D2**: Testare architettura Llama (la più comune) con forward pass e confronto golden.
- [ ] **D3**: Testare GQA (`n_kv_heads != n_heads`).
- [ ] **D4**: Testare RoPE varianti: standard, NEOX, YaRN.

## Fase E — Pulizia e Robustezza

- [ ] **E1**: Rimuovere `gc_mock_runtime_t` dal server (produzione).
- [ ] **E2**: Aggiungere `-fsanitize=address,undefined` ai target debug.
- [ ] **E3**: Test di concorrenza: scheduler con 10 richieste contemporanee di lunghezze variabili.
- [ ] **E4**: Documentare API pubblica in `include/gc/`.

---

## Note di Architettura

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
