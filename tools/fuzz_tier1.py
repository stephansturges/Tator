#!/usr/bin/env python3
import argparse
import base64
import json
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


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"URL error for {url}: {exc}") from exc


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
    args = parser.parse_args()

    root = Path(args.fuzz_pack)
    manifest = _load_manifest(root)
    img_path = _pick_image(root, manifest)
    img_b64 = _b64(img_path)
    glossary = (root / "glossary.json").read_text()
    labelmap = (root / "labelmap.txt").read_text().strip().splitlines()

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

    summary["steps"].append({"name": "prepass_base", "result": _post(args.base_url + "/qwen/prepass", prepass_base)})
    summary["steps"].append({"name": "prepass_windowed", "result": _post(args.base_url + "/qwen/prepass", prepass_win)})

    # Caption: full + windowed
    caption_full = {
        "image_base64": img_b64,
        "labelmap_glossary": glossary,
        "caption_mode": "full",
        "caption_all_windows": True,
    }
    caption_win = dict(caption_full)
    caption_win["caption_mode"] = "windowed"

    summary["steps"].append({"name": "caption_full", "result": _post(args.base_url + "/qwen/caption", caption_full)})
    summary["steps"].append({"name": "caption_windowed", "result": _post(args.base_url + "/qwen/caption", caption_win)})

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
        "dataset_id": "fuzz_pack",
        "max_images": 1,
        "enable_yolo": True,
        "enable_rfdetr": True,
        "label_iou": 0.5,
        "eval_iou": 0.75,
        "classifier_id": classifier_id,
    }
    summary["steps"].append({"name": "calibration", "result": _post(args.base_url + "/calibration/jobs", cal_payload)})

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
