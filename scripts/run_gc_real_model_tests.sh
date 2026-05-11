#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <model.gguf> [build_dir]"
  exit 1
fi

MODEL_PATH="$1"
BUILD_DIR="${2:-build}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "error: model not found: $MODEL_PATH"
  exit 1
fi

cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/${BUILD_DIR}"
cmake --build "${ROOT_DIR}/${BUILD_DIR}" -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

"${ROOT_DIR}/${BUILD_DIR}/bin/test_gc_loader" "$MODEL_PATH"
"${ROOT_DIR}/${BUILD_DIR}/bin/test_gc_arch" "$MODEL_PATH"
"${ROOT_DIR}/${BUILD_DIR}/bin/test_gc_model_integration" "$MODEL_PATH"

# Keep unit coverage in the same run.
ctest --test-dir "${ROOT_DIR}/${BUILD_DIR}" -R "test_gc_(chat|attention|kvcache|scheduler|engine|server)" --output-on-failure

