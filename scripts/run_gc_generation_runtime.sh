#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-build}"
MODEL_PATH="${2:-/Users/emanuele/Downloads/Llama-3.2-3B-Instruct-Q5_K_M.gguf}"
ROUNDS="${3:-5}"
MAX_TOKENS="${4:-64}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "error: model not found: ${MODEL_PATH}"
  exit 1
fi

cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/${BUILD_DIR}"
cmake --build "${ROOT_DIR}/${BUILD_DIR}" -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

"${ROOT_DIR}/${BUILD_DIR}/bin/test_gc_generation_runtime" "${MODEL_PATH}" \
  "--rounds=${ROUNDS}" "--max-tokens=${MAX_TOKENS}"
