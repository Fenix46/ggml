#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-sanitize"
MODEL_PATH="${1:-}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DGGML_SANITIZE_ADDRESS=ON \
  -DGGML_SANITIZE_UNDEFINED=ON

cmake --build "${BUILD_DIR}" -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

ctest --test-dir "${BUILD_DIR}" -R "test_gc_" --output-on-failure

if [[ -n "${MODEL_PATH}" ]]; then
  if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "error: model not found: ${MODEL_PATH}"
    exit 1
  fi
  "${BUILD_DIR}/bin/test_gc_model_integration" "${MODEL_PATH}"
fi
