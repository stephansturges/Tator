#!/usr/bin/env python3
import argparse
import base64
import json
import socket
from pathlib import Path
import urllib.request
from urllib.error import URLError, HTTPError


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"URL error for {url}: {exc}") from exc
    return json.loads(data)


def _cancel_active_qwen(cancel_url: str) -> None:
    data = b"{}"
    req = urllib.request.Request(
        cancel_url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except Exception:
        pass


def _post(url: str, payload: dict, *, timeout: float = 60.0, cancel_url: str = "") -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (TimeoutError, socket.timeout) as exc:
        if cancel_url:
            _cancel_active_qwen(cancel_url)
        raise RuntimeError(f"timeout_after_{timeout:g}s:{url}") from exc
    except HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    except URLError as exc:
        if isinstance(getattr(exc, "reason", None), socket.timeout):
            if cancel_url:
                _cancel_active_qwen(cancel_url)
            raise RuntimeError(f"timeout_after_{timeout:g}s:{url}") from exc
        raise RuntimeError(f"URL error for {url}: {exc}") from exc


def _record_failure(summary: dict, out_path: str, name: str, exc: Exception) -> int:
    summary["steps"].append({"name": name, "failed": True, "error": str(exc)})
    Path(out_path).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 1


def _load_manifest(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if "images" not in manifest:
        raise RuntimeError("manifest_missing_images")
    return manifest


def _pick_image(root: Path, manifest: dict) -> Path:
    img_name = sorted(manifest["images"].keys())[0]
    img_path = root / "images" / img_name
    if not img_path.exists():
        raise RuntimeError(f"missing_image:{img_name}")
    return img_path


def _b64(img_path: Path) -> str:
    return base64.b64encode(img_path.read_bytes()).decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--fuzz-pack", default="tests/fixtures/fuzz_pack")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--out", default="fuzz_tier1_summary.json")
    parser.add_argument("--request-timeout", type=float, default=60.0)
    args = parser.parse_args()

    root = Path(args.fuzz_pack)
    manifest = _load_manifest(root)
    img_path = _pick_image(root, manifest)
    img_b64 = _b64(img_path)
    glossary = (root / "glossary.json").read_text()
    labelmap = (root / "labelmap.txt").read_text().strip().splitlines()
    try:
        yolo_active = _get(args.base_url + "/yolo/active")
        labelmap_path = yolo_active.get("labelmap_path")
        if labelmap_path:
            labelmap_file = Path(labelmap_path)
            if not labelmap_file.is_absolute():
                labelmap_file = Path.cwd() / labelmap_file
            if labelmap_file.exists():
                labelmap = labelmap_file.read_text().strip().splitlines()
    except Exception:
        pass

    summary = {
        "skip_gpu": args.skip_gpu,
        "steps": [],
    }

    if args.skip_gpu:
        summary["steps"].append({"name": "prepass", "skipped": True})
        summary["steps"].append({"name": "caption_full", "skipped": True})
        summary["steps"].append({"name": "caption_windowed", "skipped": True})
        summary["steps"].append({"name": "calibration", "skipped": True})
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 0

    # Ensure runtime is unloaded so caption models reload on a single device.
    try:
        _post(args.base_url + "/runtime/unload", {})
    except Exception:
        pass

    # Prepass: baseline (non-windowed SAM3) then windowed
    prepass_base = {
        "image_base64": img_b64,
        "labelmap": labelmap,
        "labelmap_glossary": glossary,
        "enable_yolo": True,
        "enable_rfdetr": True,
        "enable_sam3_text": True,
        "enable_sam3_similarity": True,
        "sam3_text_window_extension": False,
        "similarity_window_extension": False,
        "prepass_caption": False,
        "prepass_only": True,
    }
    prepass_win = dict(prepass_base)
    prepass_win["sam3_text_window_extension"] = True
    prepass_win["similarity_window_extension"] = True

    qwen_cancel_url = args.base_url + "/qwen/cancel?force=false"
    try:
        summary["steps"].append(
            {
                "name": "prepass_base",
                "result": _post(
                    args.base_url + "/qwen/prepass",
                    prepass_base,
                    timeout=args.request_timeout,
                    cancel_url=qwen_cancel_url,
                ),
            }
        )
    except Exception as exc:
        return _record_failure(summary, args.out, "prepass_base", exc)
    try:
        summary["steps"].append(
            {
                "name": "prepass_windowed",
                "result": _post(
                    args.base_url + "/qwen/prepass",
                    prepass_win,
                    timeout=args.request_timeout,
                    cancel_url=qwen_cancel_url,
                ),
            }
        )
    except Exception as exc:
        return _record_failure(summary, args.out, "prepass_windowed", exc)

    # Caption: full + windowed
    caption_full = {
        "image_base64": img_b64,
        "labelmap_glossary": glossary,
        "caption_mode": "full",
        "caption_all_windows": True,
    }
    caption_win = dict(caption_full)
    caption_win["caption_mode"] = "windowed"

    try:
        summary["steps"].append(
            {
                "name": "caption_full",
                "result": _post(
                    args.base_url + "/qwen/caption",
                    caption_full,
                    timeout=args.request_timeout,
                    cancel_url=qwen_cancel_url,
                ),
            }
        )
    except Exception as exc:
        return _record_failure(summary, args.out, "caption_full", exc)
    try:
        summary["steps"].append(
            {
                "name": "caption_windowed",
                "result": _post(
                    args.base_url + "/qwen/caption",
                    caption_win,
                    timeout=args.request_timeout,
                    cancel_url=qwen_cancel_url,
                ),
            }
        )
    except Exception as exc:
        return _record_failure(summary, args.out, "caption_windowed", exc)

    # Calibration: tiny job with max_images=1 (uses cached prepass)
    classifier_id = None
    try:
        classifiers = _get(args.base_url + "/clip/classifiers")
        if isinstance(classifiers, list):
            entries = classifiers
        else:
            entries = classifiers.get("classifiers", [])
        if isinstance(entries, list) and entries:
            first = entries[0]
            if isinstance(first, dict):
                classifier_id = first.get("path") or first.get("id")
            elif isinstance(first, str):
                classifier_id = first
    except Exception as exc:
        summary["steps"].append(
            {
                "name": "calibration",
                "skipped": True,
                "reason": f"classifier_lookup_failed:{exc}",
            }
        )
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 0

    if not classifier_id:
        summary["steps"].append(
            {
                "name": "calibration",
                "skipped": True,
                "reason": "classifier_required",
            }
        )
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 0

    cal_payload = {
        "dataset_id": "qwen_dataset",
        "max_images": 1,
        "enable_yolo": True,
        "enable_rfdetr": True,
        "label_iou": 0.5,
        "eval_iou": 0.75,
        "classifier_id": classifier_id,
    }
    try:
        summary["steps"].append(
            {
                "name": "calibration",
                "result": _post(
                    args.base_url + "/calibration/jobs",
                    cal_payload,
                    timeout=args.request_timeout,
                ),
            }
        )
    except Exception as exc:
        summary["steps"].append(
            {
                "name": "calibration",
                "skipped": True,
                "reason": f"calibration_failed:{exc}",
            }
        )

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
