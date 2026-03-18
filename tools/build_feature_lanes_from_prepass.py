#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}", flush=True)


def _run(cmd: Sequence[str]) -> None:
    _log("RUN " + " ".join(str(c) for c in cmd))
    subprocess.run(list(cmd), cwd=REPO_ROOT, check=True, text=True)


def _ensure_symlink_or_copy(src: Path, dst: Path, *, force: bool) -> None:
    if not src.exists():
        raise SystemExit(f"precomputed_path_missing:{src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not force:
            return
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except Exception:
        import shutil

        shutil.copy2(src, dst)


def _npz_image_set(npz_path: Path) -> Set[str]:
    data = np.load(npz_path, allow_pickle=True)
    out: Set[str] = set()
    for row in data["meta"]:
        try:
            payload = json.loads(str(row))
        except Exception:
            continue
        image = str(payload.get("image") or "").strip()
        if image:
            out.add(image)
    return out


def _npz_has_label_targets(npz_path: Path) -> bool:
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception:
        return False
    return "y" in data.files


def _subset_npz_by_images(src: Path, dst: Path, keep_images: Set[str]) -> Dict[str, Any]:
    data = np.load(src, allow_pickle=True)
    meta = data["meta"]
    keep_idx: List[int] = []
    for idx, row in enumerate(meta):
        try:
            payload = json.loads(str(row))
        except Exception:
            continue
        image = str(payload.get("image") or "").strip()
        if image in keep_images:
            keep_idx.append(idx)
    subset: Dict[str, Any] = {}
    for key in data.files:
        arr = data[key]
        if hasattr(arr, "shape") and arr.shape and arr.shape[0] == len(meta):
            subset[key] = arr[keep_idx]
        else:
            subset[key] = arr
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **subset)
    return {"rows": int(len(keep_idx)), "src_rows": int(len(meta))}


def _lane_config(prepass_key: str) -> str:
    return str(prepass_key).strip()


def _lane_id(variant: str) -> str:
    return str(variant).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build calibration feature lanes from fixed prepass cache keys."
    )
    parser.add_argument("--dataset", default="qwen_dataset")
    parser.add_argument("--run-root", required=True, help="Output run root directory.")
    parser.add_argument("--nonwindow-key", required=True)
    parser.add_argument("--window-key", required=True)
    parser.add_argument("--classifier-id", required=True)
    parser.add_argument("--candidate-embed-dim", type=int, default=1024)
    parser.add_argument(
        "--lane-selection",
        default="compare_both",
        choices=["window", "nonwindow", "compare_both"],
        help="Restrict lane construction to the selected lane family or build both.",
    )
    parser.add_argument("--support-iou", type=float, default=0.5)
    parser.add_argument("--context-radius", type=float, default=0.075)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="Rebuild existing outputs.")
    parser.add_argument("--nonwindow-prepass-jsonl", default="", help="Optional precomputed nonwindow prepass JSONL.")
    parser.add_argument("--window-prepass-jsonl", default="", help="Optional precomputed window prepass JSONL.")
    parser.add_argument("--nonwindow-features", default="", help="Optional precomputed nonwindow features npz.")
    parser.add_argument("--nonwindow-labeled", default="", help="Optional precomputed nonwindow labeled npz.")
    parser.add_argument("--window-features", default="", help="Optional precomputed window features npz.")
    parser.add_argument("--window-labeled", default="", help="Optional precomputed window labeled npz.")
    args = parser.parse_args()

    run_root = (REPO_ROOT / args.run_root).resolve()
    prepass_dir = run_root / "prepass"
    lanes_dir = run_root / "lanes"
    views_dir = run_root / "views"
    prepass_dir.mkdir(parents=True, exist_ok=True)
    lanes_dir.mkdir(parents=True, exist_ok=True)
    views_dir.mkdir(parents=True, exist_ok=True)

    base_variants_all = {
        "nonwindow": str(args.nonwindow_key).strip(),
        "window": str(args.window_key).strip(),
    }
    if args.lane_selection == "window":
        active_variants = ["window"]
    elif args.lane_selection == "nonwindow":
        active_variants = ["nonwindow"]
    else:
        active_variants = ["nonwindow", "window"]
    base_variants = {
        variant: cache_key
        for variant, cache_key in base_variants_all.items()
        if variant in active_variants
    }

    prepass_jsonl_paths: Dict[str, Path] = {}
    for variant, cache_key in base_variants.items():
        precomputed = str(
            args.nonwindow_prepass_jsonl if variant == "nonwindow" else args.window_prepass_jsonl
        ).strip()
        out_jsonl = prepass_dir / f"{variant}.jsonl"
        if precomputed:
            _ensure_symlink_or_copy(Path(precomputed).resolve(), out_jsonl, force=bool(args.force))
        elif args.force or (not out_jsonl.exists()):
            _run(
                [
                    sys.executable,
                    "tools/materialize_prepass_from_cache.py",
                    "--cache-key",
                    cache_key,
                    "--output",
                    str(out_jsonl),
                ]
            )
        prepass_jsonl_paths[variant] = out_jsonl

    lane_manifest: Dict[str, Any] = {
        "generated_utc": _ts(),
        "dataset": str(args.dataset),
        "run_root": str(run_root),
        "base_variants": base_variants,
        "prepass_jsonl": {k: str(v) for k, v in prepass_jsonl_paths.items()},
        "lanes": {},
    }

    lane_precomputed: Dict[str, Dict[str, Optional[Path]]] = {
        "nonwindow": {
            "features": Path(str(args.nonwindow_features).strip()).resolve()
            if str(args.nonwindow_features).strip()
            else None,
            "labeled": Path(str(args.nonwindow_labeled).strip()).resolve()
            if str(args.nonwindow_labeled).strip()
            else None,
        },
        "window": {
            "features": Path(str(args.window_features).strip()).resolve()
            if str(args.window_features).strip()
            else None,
            "labeled": Path(str(args.window_labeled).strip()).resolve()
            if str(args.window_labeled).strip()
            else None,
        },
    }

    for variant, cache_key in base_variants.items():
        lane_id = _lane_id(variant)
        lane_dir = lanes_dir / lane_id
        lane_dir.mkdir(parents=True, exist_ok=True)
        features_path = lane_dir / "features.npz"
        labeled_path = lane_dir / "labeled.npz"
        precomputed = lane_precomputed.get(lane_id, {})
        precomputed_features = precomputed.get("features")
        precomputed_labeled = precomputed.get("labeled")
        if precomputed_labeled is None and precomputed_features is not None:
            if _npz_has_label_targets(precomputed_features):
                precomputed_labeled = precomputed_features
        if precomputed_features is not None:
            _ensure_symlink_or_copy(precomputed_features, features_path, force=bool(args.force))
        if precomputed_labeled is not None:
            _ensure_symlink_or_copy(precomputed_labeled, labeled_path, force=bool(args.force))
        if args.force or (not features_path.exists()):
            _run(
                [
                    sys.executable,
                    "tools/build_ensemble_features.py",
                    "--input",
                    str(prepass_jsonl_paths[variant]),
                    "--dataset",
                    str(args.dataset),
                    "--output",
                    str(features_path),
                    "--classifier-id",
                    str(args.classifier_id),
                    "--require-classifier",
                    "--support-iou",
                    str(float(args.support_iou)),
                    "--context-radius",
                    str(float(args.context_radius)),
                    "--embed-proj-dim",
                    str(int(args.candidate_embed_dim)),
                    "--image-embed-proj-dim",
                    "0",
                    "--device",
                    str(args.device),
                ]
            )
        if args.force or (not labeled_path.exists()):
            _run(
                [
                    sys.executable,
                    "tools/label_candidates_iou90.py",
                    "--input",
                    str(features_path),
                    "--dataset",
                    str(args.dataset),
                    "--output",
                    str(labeled_path),
                    "--iou",
                    "0.5",
                ]
            )
        lane_manifest["lanes"][lane_id] = {
            "variant": variant,
            "cache_key": cache_key,
            "image_embed_dim": 0,
            "features": str(features_path),
            "labeled": str(labeled_path),
            "prepass_jsonl": str(prepass_jsonl_paths[variant]),
            "lane_config_key": _lane_config(cache_key),
            "precomputed_features": str(precomputed_features) if precomputed_features else "",
            "precomputed_labeled": str(precomputed_labeled) if precomputed_labeled else "",
        }

    lane_image_sets: Dict[str, Set[str]] = {}
    for lane_id, lane in lane_manifest["lanes"].items():
        lane_image_sets[lane_id] = _npz_image_set(Path(lane["labeled"]))
    if "nonwindow" in lane_image_sets and "window" in lane_image_sets:
        intersection = sorted(lane_image_sets["nonwindow"] & lane_image_sets["window"])
    elif "window" in lane_image_sets:
        intersection = sorted(lane_image_sets["window"])
    elif "nonwindow" in lane_image_sets:
        intersection = sorted(lane_image_sets["nonwindow"])
    else:
        raise SystemExit("no_lane_built")
    intersection_file = views_dir / "intersection_images.json"
    intersection_file.write_text(json.dumps(intersection, indent=2), encoding="utf-8")

    lane_manifest["views"] = {
        "full": {
            "nonwindow_images": int(len(lane_image_sets.get("nonwindow", set()))),
            "window_images": int(len(lane_image_sets.get("window", set()))),
        },
        "intersection": {
            "images_file": str(intersection_file),
            "count": int(len(intersection)),
        },
    }

    intersection_prepass: Dict[str, str] = {}
    for variant, cache_key in base_variants.items():
        out_jsonl = views_dir / f"{variant}_intersection.jsonl"
        if args.force or (not out_jsonl.exists()):
            _run(
                [
                    sys.executable,
                    "tools/materialize_prepass_from_cache.py",
                    "--cache-key",
                    cache_key,
                    "--output",
                    str(out_jsonl),
                    "--images-file",
                    str(intersection_file),
                ]
            )
        intersection_prepass[variant] = str(out_jsonl)

    lane_manifest["intersection_prepass_jsonl"] = intersection_prepass
    lane_manifest["intersection_labeled"] = {}
    keep_images = set(intersection)
    for lane_id, lane in lane_manifest["lanes"].items():
        src_labeled = Path(lane["labeled"])
        dst_labeled = views_dir / f"{lane_id}_intersection.labeled.npz"
        stats = _subset_npz_by_images(src_labeled, dst_labeled, keep_images)
        lane_manifest["intersection_labeled"][lane_id] = {
            "path": str(dst_labeled),
            "rows": int(stats["rows"]),
            "src_rows": int(stats["src_rows"]),
        }

    manifest_path = run_root / "lane_manifest.json"
    manifest_path.write_text(json.dumps(lane_manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
