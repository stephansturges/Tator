#!/usr/bin/env python3
import argparse
import base64
import json
import os
from pathlib import Path
from urllib import request, error

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
FUZZ_PACK = Path("tests/fixtures/fuzz_pack")


def _post(base_url: str, path: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60) as resp:
            return resp.getcode(), json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        try:
            return exc.code, json.loads(body)
        except Exception:
            return exc.code, {"raw": body}
    except Exception as exc:
        return None, {"error": str(exc)}


def _get(base_url: str, path: str):
    req = request.Request(f"{base_url}{path}")
    try:
        with request.urlopen(req, timeout=30) as resp:
            return resp.getcode(), json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        try:
            return exc.code, json.loads(body)
        except Exception:
            return exc.code, {"raw": body}
    except Exception as exc:
        return None, {"error": str(exc)}


def _load_sample_image() -> str:
    img_dir = FUZZ_PACK / "images"
    img_path = next(iter(img_dir.iterdir()))
    data = img_path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UI smoke checks against a Tator backend."
    )
    parser.add_argument(
        "base_url",
        nargs="?",
        default=None,
        help=f"Backend base URL. Defaults to BASE_URL or {DEFAULT_BASE_URL}.",
    )
    parser.add_argument(
        "--base-url",
        dest="base_url_flag",
        default=None,
        help="Backend base URL. Overrides the positional URL and BASE_URL.",
    )
    args = parser.parse_args(argv)
    args.base_url = (
        args.base_url_flag
        or args.base_url
        or os.environ.get("BASE_URL", DEFAULT_BASE_URL)
    )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_url = args.base_url.rstrip("/")
    sample_b64 = _load_sample_image()

    checks = []
    # GET endpoints (lightweight)
    dataset_list = []
    for path in [
        "/system/gpu",
        "/system/storage_check",
        "/yolo/active",
        "/rfdetr/active",
        "/clip/classifiers",
        "/sam3/models",
        "/qwen/models",
        "/qwen/status",
        "/datasets",
        "/glossaries",
        "/qwen/datasets",
        "/sam3/datasets",
    ]:
        code, payload = _get(base_url, path)
        checks.append((path, "GET", code, payload))
        if path == "/datasets" and code and code < 400:
            if isinstance(payload, list):
                dataset_list = payload

    # POST endpoints used by UI
    post_payloads = {
        "/predict_base64": {"image_base64": sample_b64, "uuid": "smoke"},
        "/yolo/predict_full": {"image_base64": sample_b64},
        "/yolo/predict_windowed": {
            "image_base64": sample_b64,
            "slice_size": 640,
            "overlap": 0.2,
            "merge_iou": 0.5,
        },
        "/rfdetr/predict_full": {"image_base64": sample_b64},
        "/rfdetr/predict_windowed": {
            "image_base64": sample_b64,
            "slice_size": 640,
            "overlap": 0.2,
            "merge_iou": 0.5,
        },
        "/sam_bbox": {
            "image_base64": sample_b64,
            "bbox_left": 5,
            "bbox_top": 5,
            "bbox_width": 32,
            "bbox_height": 32,
        },
        "/sam_point": {
            "image_base64": sample_b64,
            "point_x": 10,
            "point_y": 10,
        },
    }

    for path, payload in post_payloads.items():
        code, resp = _post(base_url, path, payload)
        checks.append((path, "POST", code, resp))

    if dataset_list:
        first = dataset_list[0]
        dataset_id = first.get("id") if isinstance(first, dict) else None
        if dataset_id:
            for extra in [
                f"/datasets/{dataset_id}/glossary",
                f"/datasets/{dataset_id}/check",
            ]:
                code, payload = _get(base_url, extra)
                checks.append((extra, "GET", code, payload))

    print(f"UI smoke summary for {base_url}:")
    failures = 0
    for path, method, code, resp in checks:
        status = "ok"
        if code is None:
            status = "error"
            failures += 1
        elif code >= 500:
            status = "error"
            failures += 1
        elif code >= 400:
            status = "warn"
        print(f"- {method} {path}: {code} ({status})")
        if status in {"warn", "error"}:
            print("  ", resp)

    if failures:
        print(f"\nFailures: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
