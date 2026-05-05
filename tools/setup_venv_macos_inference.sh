#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-macos}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python ${PYTHON_BIN} not found. Install Python 3.11 first, for example: brew install python@3.11" >&2
  exit 1
fi

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip wheel "setuptools<81"
"${VENV_DIR}/bin/python" -m pip install -r requirements-macos-inference.txt

if [[ "${INSTALL_LOCAL_CLIP:-0}" == "1" && -d "${ROOT_DIR}/CLIP" && -f "${ROOT_DIR}/CLIP/setup.py" ]]; then
  "${VENV_DIR}/bin/python" -m pip install --no-build-isolation -e "${ROOT_DIR}/CLIP"
fi

if [[ "${INSTALL_VLM:-0}" == "1" ]]; then
  "${VENV_DIR}/bin/python" -m pip install -r requirements-macos-vlm.txt
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
