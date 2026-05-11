#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-build}"

LLAMA_MODEL="/Users/emanuele/Downloads/Llama-3.2-3B-Instruct-Q5_K_M.gguf"
GEMMA_MODEL="/Users/emanuele/Downloads/gemma-4-E2B-it-Q5_K_M.gguf"
LFM2_MODEL="/Users/emanuele/Downloads/LFM2-2.6B-Q4_0.gguf"

for m in "$LLAMA_MODEL" "$GEMMA_MODEL" "$LFM2_MODEL"; do
  if [[ ! -f "$m" ]]; then
    echo "error: model not found: $m"
    exit 1
  fi
done

cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/${BUILD_DIR}"
cmake --build "${ROOT_DIR}/${BUILD_DIR}" -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

"${ROOT_DIR}/${BUILD_DIR}/bin/test_gc_model_integration" "$LLAMA_MODEL" --expect-arch=llama
"${ROOT_DIR}/${BUILD_DIR}/bin/test_gc_model_integration" "$GEMMA_MODEL" --expect-arch=gemma4
"${ROOT_DIR}/${BUILD_DIR}/bin/test_gc_model_integration" "$LFM2_MODEL" --expect-arch=lfm2

# Keep core unit suite in the same matrix run.
ctest --test-dir "${ROOT_DIR}/${BUILD_DIR}" -R "test_gc_(arch|vocab|chat|attention|kvcache|scheduler|engine|server)" --output-on-failure
