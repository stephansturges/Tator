#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${1:-facebook/dinov3-vitb16-pretrain-lvd1689m}"
export GIT_CONFIG_GLOBAL="${GIT_CONFIG_GLOBAL:-/dev/null}"
SAFE_MODEL_ID="$(python3 - <<'PY' "$MODEL_ID"
import sys
print("".join(ch if ch.isalnum() else "_" for ch in sys.argv[1]).strip("_") or "dinov3_model")
PY
)"
OUT_DIR="${2:-$ROOT/uploads/model_cache/mlx_dinov3/$SAFE_MODEL_ID}"
BUILD_DIR="$ROOT/tools/mlx_dinov3_worker/.build/release"

"$BUILD_DIR/mlx-dinov3-convert" "$MODEL_ID" "$OUT_DIR"
echo "$OUT_DIR"
