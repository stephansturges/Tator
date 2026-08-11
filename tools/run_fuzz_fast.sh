#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
FUZZ_PACK="${FUZZ_PACK:-tests/fixtures/fuzz_pack}"
OUT_DIR="${OUT_DIR:-/tmp/tator_fuzz}"
SKIP_GPU="${SKIP_GPU:-1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-60}"

resolve_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    if [[ -x "${PYTHON}" ]]; then
      printf '%s\n' "${PYTHON}"
      return
    fi
    if command -v "${PYTHON}" >/dev/null 2>&1; then
      command -v "${PYTHON}"
      return
    fi
    echo "Configured PYTHON is not executable: ${PYTHON}" >&2
    exit 127
  fi

  for candidate in \
    "${REPO_ROOT}/.venv-macos/bin/python" \
    "${REPO_ROOT}/.venv/bin/python" \
    python3 \
    python
  do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return
    fi
  done

  echo "No Python interpreter found. Set PYTHON=/path/to/python." >&2
  exit 127
}

PYTHON_BIN="$(resolve_python)"

cd "${REPO_ROOT}"
mkdir -p "$OUT_DIR"

echo "Running Tier-0 smoke..."
"${PYTHON_BIN}" tools/fuzz_tier0.py "$BASE_URL" | tee "$OUT_DIR/tier0.json"

echo "Running Tier-1 fuzz (skip_gpu=${SKIP_GPU})..."
if [[ "$SKIP_GPU" == "1" ]]; then
  "${PYTHON_BIN}" tools/fuzz_tier1.py --base-url "$BASE_URL" --fuzz-pack "$FUZZ_PACK" --skip-gpu --request-timeout "$REQUEST_TIMEOUT" --out "$OUT_DIR/tier1.json"
else
  "${PYTHON_BIN}" tools/fuzz_tier1.py --base-url "$BASE_URL" --fuzz-pack "$FUZZ_PACK" --request-timeout "$REQUEST_TIMEOUT" --out "$OUT_DIR/tier1.json"
fi

echo "Fuzz summary written to $OUT_DIR"
