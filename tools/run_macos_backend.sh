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
PORT_CONFLICT_EXIT_CODE="${TATOR_BACKEND_PORT_CONFLICT_EXIT_CODE:-98}"
RESTART_COUNT=0
export TATOR_BACKEND_LAUNCHER="tools/run_macos_backend.sh"
export TATOR_BACKEND_LAUNCHER_RESTARTS_CRASHES="${RESTART_ON_CRASH}"
export TATOR_BACKEND_LAUNCHER_RESTART_MAX="${RESTART_MAX}"
export TATOR_BACKEND_LAUNCHER_RESTART_DELAY="${RESTART_DELAY}"
export TATOR_BACKEND_LAUNCHER_RESTART_MAX_DELAY="${RESTART_MAX_DELAY}"

backend_port_available() {
  HOST="${HOST}" PORT="${PORT}" "${VENV_DIR}/bin/python" - <<'PY'
import errno
import os
import socket
import sys

host = os.environ.get("HOST", "127.0.0.1")
raw_port = os.environ.get("PORT", "8000")
try:
    port = int(raw_port)
except (TypeError, ValueError):
    print(f"Invalid backend PORT {raw_port!r}", file=sys.stderr)
    sys.exit(64)

try:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
except OSError as exc:
    print(f"Cannot resolve backend bind address {host}:{port}: {exc}", file=sys.stderr)
    sys.exit(65)

busy = False
denied = False
last_error = None
for family, socktype, proto, _canonname, sockaddr in infos:
    sock = socket.socket(family, socktype, proto)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(sockaddr)
        sys.exit(0)
    except OSError as exc:
        last_error = exc
        if exc.errno == errno.EADDRINUSE:
            busy = True
        elif exc.errno == errno.EACCES:
            denied = True
    finally:
        sock.close()

if busy:
    sys.exit(98)
if denied:
    sys.exit(13)
print(f"Cannot probe backend bind address {host}:{port}: {last_error}", file=sys.stderr)
sys.exit(1)
PY
}

print_backend_port_owner() {
  if command -v lsof >/dev/null 2>&1; then
    local owner
    owner="$(lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "${owner}" ]]; then
      echo "Current listener on ${HOST}:${PORT}:" >&2
      echo "${owner}" >&2
    fi
  fi
}

exit_if_backend_port_busy() {
  local context="${1:-start}"
  local probe_status=0
  if backend_port_available; then
    return 0
  else
    probe_status=$?
  fi
  if [[ "${probe_status}" == "98" ]]; then
    if [[ "${context}" == "restart" ]]; then
      echo "Backend port ${HOST}:${PORT} is now in use; not restarting into a bind loop." >&2
    else
      echo "Backend port ${HOST}:${PORT} is already in use; not starting another backend." >&2
    fi
    print_backend_port_owner
    echo "Use the existing backend, stop the listener above, or choose another port, e.g. PORT=8080 tools/run_macos_backend.sh." >&2
    exit "${PORT_CONFLICT_EXIT_CODE}"
  fi
  if [[ "${probe_status}" == "13" ]]; then
    echo "Backend port ${HOST}:${PORT} cannot be bound due to permissions." >&2
    exit "${probe_status}"
  fi
  exit "${probe_status}"
}

exit_if_backend_port_busy "start"

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
    exit_if_backend_port_busy "restart"
    continue
  fi
  if [[ "${RESTART_ON_CRASH}" == "1" || "${RESTART_ON_CRASH}" == "true" || "${RESTART_ON_CRASH}" == "yes" ]]; then
    exit_if_backend_port_busy "restart"
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
