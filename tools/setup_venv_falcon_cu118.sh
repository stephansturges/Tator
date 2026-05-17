#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != --* ]]; then
  ENV_DIR="$1"
  shift
else
  ENV_DIR="$ROOT_DIR/.venv"
fi
INSTALL_DEV="${INSTALL_DEV:-0}"
ARGS=(falcon-cu118 --venv-dir "$ENV_DIR")
DRY_RUN=0
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  fi
done

if [[ "$INSTALL_DEV" == "1" ]]; then
  ARGS+=(--dev)
fi

python3 "$ROOT_DIR/tools/setup_env.py" "${ARGS[@]}" "$@"
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
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
