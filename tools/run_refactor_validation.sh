#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
SKIP_GPU="${SKIP_GPU:-1}"

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

echo "==> PyCompile core modules"
"${PYTHON_BIN}" -m py_compile \
  localinferenceapi.py \
  app/__init__.py \
  models/*.py \
  api/*.py \
  services/*.py \
  utils/*.py \
  tools/*.py

if [[ "${RUN_UNUSED_SCAN:-0}" == "1" ]]; then
  echo "==> Unused-def scan"
  "${PYTHON_BIN}" tools/scan_unused_defs.py
fi

if [[ "${SKIP_FUZZ:-0}" == "1" ]]; then
  echo "==> Fuzz tests skipped (SKIP_FUZZ=1)"
else
  echo "==> Tier-0/Tier-1 fuzz (skip_gpu=${SKIP_GPU})"
  if [[ "${SKIP_GPU}" == "1" ]]; then
    PYTHON="${PYTHON_BIN}" SKIP_GPU=1 BASE_URL="${BASE_URL}" tools/run_fuzz_fast.sh
  else
    PYTHON="${PYTHON_BIN}" SKIP_GPU=0 BASE_URL="${BASE_URL}" tools/run_fuzz_fast.sh
  fi
fi

echo "==> Refactor validation complete"
