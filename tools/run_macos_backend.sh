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
RESTART_ON_CRASH="${TATOR_BACKEND_RESTART_ON_CRASH:-1}"
RESTART_MAX="${TATOR_BACKEND_RESTART_MAX:-0}"
RESTART_DELAY="${TATOR_BACKEND_RESTART_DELAY:-1}"
RESTART_MAX_DELAY="${TATOR_BACKEND_RESTART_MAX_DELAY:-30}"
RESTART_COUNT=0
export TATOR_BACKEND_LAUNCHER="tools/run_macos_backend.sh"
export TATOR_BACKEND_LAUNCHER_RESTARTS_CRASHES="${RESTART_ON_CRASH}"
export TATOR_BACKEND_LAUNCHER_RESTART_MAX="${RESTART_MAX}"
export TATOR_BACKEND_LAUNCHER_RESTART_DELAY="${RESTART_DELAY}"
export TATOR_BACKEND_LAUNCHER_RESTART_MAX_DELAY="${RESTART_MAX_DELAY}"

while true; do
  set +e
  "${VENV_DIR}/bin/python" -m uvicorn app:app --host "${HOST}" --port "${PORT}"
  status=$?
  set -e
  if [[ "${status}" == "0" ]]; then
    exit 0
  fi
  if [[ "${status}" == "130" || "${status}" == "143" ]]; then
    exit "${status}"
  fi
  if [[ "${status}" == "${TATOR_QWEN_CANCEL_RESTART_EXIT_CODE:-75}" ]]; then
    echo "Backend exited after Qwen cancellation; restarting..." >&2
    sleep "${RESTART_DELAY}"
    continue
  fi
  if [[ "${RESTART_ON_CRASH}" == "1" || "${RESTART_ON_CRASH}" == "true" || "${RESTART_ON_CRASH}" == "yes" ]]; then
    RESTART_COUNT=$((RESTART_COUNT + 1))
    if [[ "${RESTART_MAX}" != "0" && "${RESTART_COUNT}" -gt "${RESTART_MAX}" ]]; then
      echo "Backend exited with status ${status}; restart limit ${RESTART_MAX} reached." >&2
      exit "${status}"
    fi
    delay="${RESTART_DELAY}"
    if command -v python3 >/dev/null 2>&1; then
      delay="$(python3 - <<PY
base = float("${RESTART_DELAY}")
cap = float("${RESTART_MAX_DELAY}")
count = int("${RESTART_COUNT}")
print(min(cap, base * (2 ** max(0, count - 1))))
PY
)"
    fi
    limit_text=""
    if [[ "${RESTART_MAX}" != "0" ]]; then
      limit_text="/${RESTART_MAX}"
    fi
    echo "Backend exited with status ${status}; restarting in ${delay}s (restart ${RESTART_COUNT}${limit_text})." >&2
    sleep "${delay}"
    continue
  fi
  exit "${status}"
done
