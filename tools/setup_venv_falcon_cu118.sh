#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${1:-$ROOT_DIR/.venv}"
INSTALL_DEV="${INSTALL_DEV:-0}"

python3 -m venv "$ENV_DIR"
# shellcheck disable=SC1090
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Falcon-Perception works better on newer PyTorch FlexAttention, but we do not
# need a local CUDA toolkit install for this wheel-based path.
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

python -m pip install -r "$ROOT_DIR/requirements.txt" -c "$ROOT_DIR/constraints/falcon-cu118.txt"

if [[ "$INSTALL_DEV" == "1" ]]; then
  python -m pip install -r "$ROOT_DIR/requirements-dev.txt" -c "$ROOT_DIR/constraints/falcon-cu118.txt"
fi

PIP_CHECK_OUTPUT="$(python -m pip check 2>&1 || true)"
if [[ -n "$PIP_CHECK_OUTPUT" ]]; then
  if [[ "$PIP_CHECK_OUTPUT" == "decord 0.6.0 is not supported on this platform" ]]; then
    echo "Ignoring known pip check false positive: $PIP_CHECK_OUTPUT"
  else
    echo "$PIP_CHECK_OUTPUT" >&2
    exit 1
  fi
fi

cat <<EOF

Falcon GPU environment is ready in: $ENV_DIR

Validation:
  source "$ENV_DIR/bin/activate"
  python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
print("devices", torch.cuda.device_count())
PY

EOF
