#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-macos}"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env.macos}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing ${VENV_DIR}/bin/python. Run tools/setup_venv_macos_inference.sh first." >&2
  exit 1
fi

cd "${ROOT_DIR}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

export TATOR_INFERENCE_DEVICE="${TATOR_INFERENCE_DEVICE:-auto}"
export TATOR_ALLOW_MPS="${TATOR_ALLOW_MPS:-1}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export SAM3_DEVICE="${SAM3_DEVICE:-auto}"
export QWEN_DEVICE="${QWEN_DEVICE:-auto}"

exec "${VENV_DIR}/bin/python" -m uvicorn app:app --host "${HOST}" --port "${PORT}"
