#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-macos}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
ARGS=(macos --python "${PYTHON_BIN}" --venv-dir "${VENV_DIR}")
DRY_RUN=0
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  fi
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python ${PYTHON_BIN} not found. Install Python 3.11 first, for example: brew install python@3.11" >&2
  exit 1
fi

if [[ "${INSTALL_LOCAL_CLIP:-0}" == "1" ]]; then
  ARGS+=(--install-local-clip)
fi

cd "${ROOT_DIR}"
"${PYTHON_BIN}" "${ROOT_DIR}/tools/setup_env.py" "${ARGS[@]}" "$@"
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

"${VENV_DIR}/bin/python" - <<'PY'
import platform
import torch

print("python", platform.python_version())
print("machine", platform.machine())
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
mps = getattr(torch.backends, "mps", None)
print("mps_available", bool(mps and mps.is_available()))
print("mps_built", bool(mps and getattr(mps, "is_built", lambda: False)()))
PY

cat <<EOF

macOS inference environment ready:
  source "${VENV_DIR}/bin/activate"
  cp .env.macos.example .env.macos
  tools/run_macos_backend.sh
EOF
