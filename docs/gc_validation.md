# GGUF.CORE Validation Lane

This repo must validate the model forward path before trusting paged KV,
continuous batching, or the HTTP server.

## Phase Gate Order

1. Loader, hparams, vocab, and chat template load a real GGUF.
2. CPU-only single-request Llama forward produces a stable greedy token.
3. Full-prefill and chunked-prefill produce the same first greedy token.
4. The first greedy token and top-k logits match a local llama.cpp reference.
5. Only then validate paged KV decode, scheduler batching, and server output.

## Core Harness

Build:

```sh
cmake --build build --target test_gc_llama_forward -j4
```

Run smoke validation on the current Llama 3.2 1B model:

```sh
build/bin/test_gc_llama_forward \
  /Users/emanuelescarlata/Downloads/Llama-3.2-1B-Instruct-Q3_K_S.gguf \
  --chat \
  --prompt 'Say hello in one short sentence.' \
  --check-chunking
```

Run with a llama.cpp golden token once available:

```sh
build/bin/test_gc_llama_forward \
  /Users/emanuelescarlata/Downloads/Llama-3.2-1B-Instruct-Q3_K_S.gguf \
  --chat \
  --prompt 'Say hello in one short sentence.' \
  --expect-token <TOKEN_ID_FROM_LLAMA_CPP> \
  --check-chunking
```

## llama.cpp Logits Golden

Reference checkout used locally:

```sh
/Users/emanuelescarlata/Desktop/llama-original.cpp
```

The existing `llama-debug` binary may need dylib symlinks because its rpath can
point at an older directory name. A non-invasive setup in `/private/tmp`:

```sh
mkdir -p /private/tmp/gc_llama_libs
ln -sf /Users/emanuelescarlata/Desktop/llama-original.cpp/build-cpu/bin/libllama-common.0.0.9140.dylib /private/tmp/gc_llama_libs/libllama-common.0.dylib
ln -sf /Users/emanuelescarlata/Desktop/llama-original.cpp/build-cpu/bin/libllama.0.0.9140.dylib /private/tmp/gc_llama_libs/libllama.0.dylib
ln -sf /Users/emanuelescarlata/Desktop/llama-original.cpp/build-cpu/bin/libggml.0.10.0.dylib /private/tmp/gc_llama_libs/libggml.0.dylib
ln -sf /Users/emanuelescarlata/Desktop/llama-original.cpp/build-cpu/bin/libggml-cpu.0.10.0.dylib /private/tmp/gc_llama_libs/libggml-cpu.0.dylib
ln -sf /Users/emanuelescarlata/Desktop/llama-original.cpp/build-cpu/bin/libggml-blas.0.10.0.dylib /private/tmp/gc_llama_libs/libggml-blas.0.dylib
ln -sf /Users/emanuelescarlata/Desktop/llama-original.cpp/build-cpu/bin/libggml-base.0.10.0.dylib /private/tmp/gc_llama_libs/libggml-base.0.dylib
```

Generate a plain-text logits golden:

```sh
rm -rf /private/tmp/gc_llama_ref_hello
DYLD_LIBRARY_PATH=/private/tmp/gc_llama_libs \
  /Users/emanuelescarlata/Desktop/llama-original.cpp/build-cpu/bin/llama-debug \
  -m /Users/emanuelescarlata/Downloads/Llama-3.2-1B-Instruct-Q3_K_S.gguf \
  -p 'Hello' \
  --save-logits \
  --logits-output-dir /private/tmp/gc_llama_ref_hello \
  --no-warmup
```

Compare GGUF.CORE against it:

```sh
build/bin/test_gc_llama_forward \
  /Users/emanuelescarlata/Downloads/Llama-3.2-1B-Instruct-Q3_K_S.gguf \
  --prompt 'Hello' \
  --save-logits /private/tmp/gc_core_hello.bin \
  --compare-logits /private/tmp/gc_llama_ref_hello/llamacpp-Llama-3.2-1B-Instruct-Q3_K_S.bin
```

Current accepted baseline after the GQA fix:

```text
top10_overlap=10/10
mean_abs≈0.132
max_abs≈0.740
```

## Rules

- Do not use `gc_ggml_model_runner_t` to validate model quality; it is a synthetic runner for scheduler tests.
- Do not accept server output as a model-correctness signal until `test_gc_llama_forward` matches llama.cpp.
- Fix one subsystem at a time: tokenizer/template, hparams/RoPE, FFN, attention/mask, residual/norm, then KV.
