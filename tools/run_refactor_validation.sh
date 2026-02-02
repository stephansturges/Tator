#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
SKIP_GPU="${SKIP_GPU:-1}"

echo "==> PyCompile core modules"
python -m py_compile \
  localinferenceapi.py \
  app/__init__.py \
  models/*.py \
  api/*.py \
  services/*.py \
  utils/*.py \
  tools/*.py

if [[ "${RUN_UNUSED_SCAN:-0}" == "1" ]]; then
  echo "==> Unused-def scan"
  python tools/scan_unused_defs.py
fi

if [[ "${SKIP_FUZZ:-0}" == "1" ]]; then
  echo "==> Fuzz tests skipped (SKIP_FUZZ=1)"
else
  echo "==> Tier-0/Tier-1 fuzz (skip_gpu=${SKIP_GPU})"
  if [[ "${SKIP_GPU}" == "1" ]]; then
    SKIP_GPU=1 BASE_URL="${BASE_URL}" tools/run_fuzz_fast.sh
  else
    SKIP_GPU=0 BASE_URL="${BASE_URL}" tools/run_fuzz_fast.sh
  fi
fi

echo "==> Refactor validation complete"
