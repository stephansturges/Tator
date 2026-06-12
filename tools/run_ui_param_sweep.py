#!/usr/bin/env python3
"""Lightweight negative/param sweep to catch endpoint contract regressions."""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:8000"

PARAM_SWEEPS = [
    ("/qwen/caption", "POST", [
        {"image_base64": "", "caption_mode": "full"},
        {"image_token": "missing", "caption_mode": "windowed"},
        {"caption_mode": "windowed"},
    ]),
    ("/qwen/prepass", "POST", [
        {"dataset_id": "missing"},
        {"dataset_id": "qwen_dataset", "max_images": -1},
        {"dataset_id": "qwen_dataset", "max_images": 0},
    ]),
    ("/yolo/predict_full", "POST", [
        {"image_base64": ""},
        {"image_token": "missing"},
    ]),
    ("/rfdetr/predict_full", "POST", [
        {"image_base64": ""},
        {"image_token": "missing"},
    ]),
    ("/sam_bbox", "POST", [
        {"image_token": "missing", "bbox_xyxy_px": [0, 0, 10, 10]},
        {"image_base64": "", "bbox_xyxy_px": [0, 0, 10, 10]},
    ]),
]

# Negative-path probes are expected to fail cleanly. HTTP 412 is a valid
# precondition response for optional detector routes when no active model is set.
OK_STATUS = {400, 404, 405, 412, 422, 428, 500, 503}


def run_sweep(base_url: str = DEFAULT_BASE_URL) -> dict:
    base = base_url.rstrip("/")
    failures = []
    for path, method, payloads in PARAM_SWEEPS:
        for payload in payloads:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{base}{path}",
                data=data,
                headers={"Content-Type": "application/json"},
                method=method,
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    status = resp.getcode()
                    body = resp.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                status = exc.code
                body = exc.read().decode("utf-8")
            except Exception as exc:
                failures.append({"path": path, "payload": payload, "error": str(exc)})
                continue
            if status not in OK_STATUS:
                failures.append({"path": path, "payload": payload, "status": status, "body": body})
    return {"failures": failures}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    base_url = args[0] if args else DEFAULT_BASE_URL
    summary = run_sweep(base_url)
    print(json.dumps(summary, indent=2))
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
