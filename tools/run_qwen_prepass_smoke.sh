#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

resolve_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    if [[ -x "${PYTHON}" ]]; then
      printf '%s\n' "${PYTHON}"
      return
    fi
    if command -v "${PYTHON}" >/dev/null 2>&1; then
      command -v "${PYTHON}"
      return
    fi
    echo "Configured PYTHON is not executable: ${PYTHON}" >&2
    exit 127
  fi

  for candidate in \
    "${REPO_ROOT}/.venv-macos/bin/python" \
    "${REPO_ROOT}/.venv/bin/python" \
    python3 \
    python
  do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return
    fi
  done

  echo "No Python interpreter found. Set PYTHON=/path/to/python." >&2
  exit 127
}

PYTHON_BIN="$(resolve_python)"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" - "$@" <<'PY'
import argparse
import base64
import json
import random
import time
from pathlib import Path

import requests


parser = argparse.ArgumentParser(description="Run a Qwen prepass 10-image smoke test.")
parser.add_argument("--count", type=int, default=10, help="Number of images to sample.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
parser.add_argument("--dataset", default="qwen_dataset", help="Dataset id under uploads/qwen_runs/datasets.")
parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Thinking", help="Qwen model id.")
parser.add_argument("--variant", default="Thinking", choices=["auto", "Instruct", "Thinking"], help="Model variant.")
parser.add_argument("--output", default=None, help="Output JSONL path.")
parser.add_argument("--api-root", default="http://127.0.0.1:8000", help="API root.")
args = parser.parse_args()

random.seed(args.seed)

dataset_root = Path("uploads/qwen_runs/datasets") / args.dataset
coco_path = dataset_root / "val" / "_annotations.coco.json"
if not coco_path.exists():
    raise SystemExit(f"Missing COCO annotations at {coco_path}")

coco = json.loads(coco_path.read_text())
images = list(coco.get("images") or [])
if len(images) < args.count:
    raise SystemExit(f"Not enough images ({len(images)}) for {args.count}-image smoke test.")

selected = random.sample(images, args.count)
out_path = Path(args.output or f"qwen_prepass_smoke_{args.count}img_seed{args.seed}.jsonl")

api_url = f"{args.api_root.rstrip('/')}/qwen/prepass"
unload_url = f"{args.api_root.rstrip('/')}/runtime/unload"

requests.post(unload_url, timeout=None)

def emit(record: dict) -> None:
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")

for idx, info in enumerate(selected, 1):
    img_name = info.get("file_name")
    if not img_name:
        continue
    img_path = dataset_root / "val" / img_name
    img_bytes = img_path.read_bytes()
    image_base64 = base64.b64encode(img_bytes).decode("utf-8")
    payload = {
        "dataset_id": args.dataset,
        "image_base64": image_base64,
        "model_id": args.model_id,
        "model_variant": args.variant,
        "max_new_tokens": 1200,
    }
    print(f"[{idx}/{args.count}] {img_name}")
    record = {
        "ts": time.time(),
        "image": img_name,
        "model_id": args.model_id,
        "variant": args.variant,
        "payload_bytes": len(json.dumps(payload)),
    }
    try:
        resp = requests.post(api_url, json=payload, timeout=None)
        record["status"] = resp.status_code
        if resp.status_code == 200:
            data = resp.json()
            record["detections"] = data.get("detections", [])
            record["trace"] = data.get("trace", [])
            record["warnings"] = data.get("warnings", [])
        else:
            record["error"] = resp.text[:300]
    except Exception as exc:  # noqa: BLE001
        record["status"] = "exception"
        record["error"] = str(exc)[:300]
    emit(record)

print("Smoke test complete.")
print("Output:", out_path)
PY
