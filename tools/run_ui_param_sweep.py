#!/usr/bin/env python3
"""Lightweight negative/param sweep to catch endpoint contract regressions."""
import json
import urllib.request

BASE_URL = "http://127.0.0.1:8000"

param_sweeps = [
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

ok_status = {400, 404, 405, 422, 428, 500, 503}

failures = []
for path, method, payloads in param_sweeps:
    for payload in payloads:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{BASE_URL}{path}",
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
        if status not in ok_status:
            failures.append({"path": path, "payload": payload, "status": status, "body": body})

print(json.dumps({"failures": failures}, indent=2))
