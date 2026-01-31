#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
FUZZ_PACK="${FUZZ_PACK:-tests/fixtures/fuzz_pack}"
OUT_DIR="${OUT_DIR:-/tmp/tator_fuzz}"
SKIP_GPU="${SKIP_GPU:-1}"

mkdir -p "$OUT_DIR"

echo "Running Tier-0 smoke..."
python tools/fuzz_tier0.py "$BASE_URL" | tee "$OUT_DIR/tier0.json"

echo "Running Tier-1 fuzz (skip_gpu=${SKIP_GPU})..."
if [[ "$SKIP_GPU" == "1" ]]; then
  python tools/fuzz_tier1.py --base-url "$BASE_URL" --fuzz-pack "$FUZZ_PACK" --skip-gpu --out "$OUT_DIR/tier1.json"
else
  python tools/fuzz_tier1.py --base-url "$BASE_URL" --fuzz-pack "$FUZZ_PACK" --out "$OUT_DIR/tier1.json"
fi

echo "Fuzz summary written to $OUT_DIR"
