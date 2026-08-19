#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${MTPLX_VENV_DIR:-${ROOT_DIR}/.venv-mtplx}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  elif [[ -x "${ROOT_DIR}/.venv-macos/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv-macos/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/requirements-macos-mtplx.txt"
"${VENV_DIR}/bin/mtplx" --version

echo "MTPLX runtime ready at ${VENV_DIR}."
