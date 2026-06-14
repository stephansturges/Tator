#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -x ".venv-macos/bin/python" ]]; then
  PYTHON_BIN=".venv-macos/bin/python"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
else
  PYTHON_BIN="python3"
fi
VENV_DIR="${QWEN36_SWIR_VENV:-.venv-qwen36-swir}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements-qwen36-swir.txt

echo "Qwen3.6/SwiReasoning environment ready: $VENV_DIR"
echo "Run: $VENV_DIR/bin/python tools/qwen36_swir_smoke.py"
