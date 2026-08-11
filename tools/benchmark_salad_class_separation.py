#!/usr/bin/env python3
"""Benchmark supported pooled DINOv3/C-RADIO recipes for class separation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import localinferenceapi as api
from services.data_ingestion import IMAGE_EXTS


def _read_labelmap(path: Path) -> List[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise SystemExit(f"Labelmap is empty: {path}")
    return labels


def _find_image_for_label(image_dir: Path, label_path: Path) -> Optional[Path]:
    for ext in sorted(IMAGE_EXTS):
        candidate = image_dir / f"{label_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _build_manifest(
    *,
    image_dir: Path,
    label_dir: Path,
    labelmap: Sequence[str],
    image_cap: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = _find_image_for_label(image_dir, label_path)
        if image_path is None:
            continue
        label_lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not label_lines:
            continue
        rows.append(
            {
                "split": "train",
                "image_relpath": image_path.name,
                "image_name": image_path.name,
                "label_lines": label_lines,
            }
        )
        if image_cap > 0 and len(rows) >= image_cap:
            break
    if not rows:
        raise SystemExit(f"No labeled images found in {image_dir} with labels from {label_dir}")
    return {
        "dataset_label": "Class-separation benchmark",
        "labelmap": list(labelmap),
        "images": rows,
        "yolo_layout": "flat",
        "source_mode": "active_workspace",
    }


def _run_class_analysis(
    *,
    manifest: Dict[str, Any],
    image_dir: Path,
    recipe_id: str,
    encoder_type: str,
    encoder_model: str,
    cradio_pooling: str,
    embedding_aggregation: str,
    embedding_view_mode: str,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], float]:
    request = api._normalize_class_analysis_request(
        {
            "source_mode": "active_workspace",
            "workspace_id": f"class_bench_{recipe_id}",
            "workspace_dir": str(image_dir.resolve()),
            "workspace_manifest": manifest,
            "yolo_layout": "flat",
            "labelmap": manifest.get("labelmap") or [],
            "analysis_scope": "all_classes",
            "encoder_type": encoder_type,
            "encoder_model": encoder_model,
            "projection": args.projection,
            "projection_neighbor_k": args.projection_neighbors,
            "crop_mode": "padded_square",
            "padding_ratio": args.padding_ratio,
            "preprocess_mode": "canonical",
            "canonical_size": args.canonical_size,
            "dinov3_pooling": "pooler",
            "cradio_pooling": cradio_pooling,
            "embedding_aggregation": embedding_aggregation,
            "embedding_salad_head_id": "",
            "background_mode": "full_crop",
            "embedding_view_mode": embedding_view_mode,
            "embedding_adjustment": "remove_size_bias",
            "sample_cap": args.object_sample_cap,
            "neighbor_k": args.neighbor_k,
            "batch_size": args.batch_size,
            "seed": args.seed,
        }
    )
    job_id = f"bench_ca_{recipe_id}_{uuid.uuid4().hex[:8]}"
    job = api.ClassAnalysisJob(job_id=job_id, request=request)
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS[job_id] = job
    started = time.perf_counter()
    api._run_class_analysis_job(job)
    runtime = time.perf_counter() - started
    if job.status != "completed":
        raise RuntimeError(f"{recipe_id} class analysis failed: {job.error or job.message}")
    return dict(job.result or {}), runtime


def _summarize_result(recipe_id: str, result: Dict[str, Any], runtime_seconds: float) -> Dict[str, Any]:
    points = list(result.get("points") or [])
    ratios_by_class: Dict[str, List[float]] = defaultdict(list)
    for point in points:
        cls = str(point.get("class_name") or "")
        ratios_by_class[cls].append(float(point.get("same_class_neighbor_ratio") or 0.0))
    class_mean_purity = {
        cls: float(np.mean(values)) if values else 0.0
        for cls, values in sorted(ratios_by_class.items())
    }
    strongest = ((result.get("diagnostics") or {}).get("strongest_size_axis") or {})
    object_purity = float(np.mean([float(p.get("same_class_neighbor_ratio") or 0.0) for p in points])) if points else 0.0
    balanced_purity = float(np.mean(list(class_mean_purity.values()))) if class_mean_purity else 0.0
    return {
        "recipe_id": recipe_id,
        "runtime_seconds": runtime_seconds,
        "object_count": len(points),
        "object_weighted_nn_purity": object_purity,
        "class_balanced_nn_purity": balanced_purity,
        "worst_class_nn_purity": float(min(class_mean_purity.values())) if class_mean_purity else 0.0,
        "class_mean_nn_purity": class_mean_purity,
        "wrong_class_candidate_count": len(result.get("wrong_class_candidates") or []),
        "size_axis_abs_correlation": abs(float(strongest.get("correlation") or 0.0)),
        "size_axis": strongest,
        "summary": result.get("summary") or {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("uploads/datasets/labeling_session_1"))
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--labelmap", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("uploads/class_analysis/benchmarks/class_separation"))
    parser.add_argument("--image-cap", type=int, default=128)
    parser.add_argument("--object-sample-cap", type=int, default=256)
    parser.add_argument("--include-cradio", action="store_true")
    parser.add_argument("--cradio-model", default=api.CRADIO_DEFAULT_MODEL)
    parser.add_argument(
        "--cradio-pooling",
        choices=["summary", "spatial_mean", "summary_spatial_concat"],
        default="summary",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--canonical-size", type=int, default=336)
    parser.add_argument("--padding-ratio", type=float, default=0.08)
    parser.add_argument("--projection", choices=["pca", "umap"], default="pca")
    parser.add_argument("--projection-neighbors", type=int, default=50)
    parser.add_argument("--neighbor-k", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    image_dir = (args.image_dir or (dataset_root / "train" / "images")).expanduser().resolve()
    label_dir = (args.label_dir or (dataset_root / "train" / "labels")).expanduser().resolve()
    labelmap_path = (args.labelmap or (dataset_root / "labelmap.txt")).expanduser().resolve()
    labelmap = _read_labelmap(labelmap_path)
    manifest = _build_manifest(image_dir=image_dir, label_dir=label_dir, labelmap=labelmap, image_cap=args.image_cap)

    output_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    recipes = [
        ("pooled_balanced", "dinov3", api.CLASS_ANALYSIS_DEFAULT_DINOV3_MODEL, "summary", "pooled", "single"),
        ("pooled_precise", "dinov3", api.CLASS_ANALYSIS_DEFAULT_DINOV3_MODEL, "summary", "pooled", "tight_context"),
    ]
    if args.include_cradio:
        recipes.append(
            (
                "cradio_summary",
                "cradio",
                args.cradio_model,
                args.cradio_pooling,
                "pooled",
                "single",
            )
        )
    results: Dict[str, Any] = {
        "inputs": {
            "dataset_root": str(dataset_root),
            "image_dir": str(image_dir),
            "label_dir": str(label_dir),
            "labelmap": str(labelmap_path),
            "image_count": len(manifest.get("images") or []),
            "object_sample_cap": int(args.object_sample_cap),
            "local_salad_policy": api.LOCAL_SALAD_POLICY,
            "loaded_external_salad_checkpoint": False,
            "include_cradio": bool(args.include_cradio),
            "crop_level_local_salad": "disabled",
            "cradio_model": args.cradio_model,
            "cradio_pooling": args.cradio_pooling,
        },
        "recipes": {},
    }
    for recipe_id, encoder_type, encoder_model, cradio_pooling, aggregation, view_mode in recipes:
        result, runtime = _run_class_analysis(
            manifest=manifest,
            image_dir=image_dir,
            recipe_id=recipe_id,
            encoder_type=encoder_type,
            encoder_model=encoder_model,
            cradio_pooling=cradio_pooling,
            embedding_aggregation=aggregation,
            embedding_view_mode=view_mode,
            args=args,
        )
        summary = _summarize_result(recipe_id, result, runtime)
        results["recipes"][recipe_id] = summary

    ranked = sorted(
        results["recipes"].values(),
        key=lambda item: (
            -float(item.get("class_balanced_nn_purity") or 0.0),
            -float(item.get("object_weighted_nn_purity") or 0.0),
            float(item.get("size_axis_abs_correlation") or 1.0),
            float(item.get("runtime_seconds") or 0.0),
        ),
    )
    recipe_lookup = {
        "pooled_balanced": {
            "id": "balanced",
            "encoder_type": "dinov3",
            "embedding_aggregation": "pooled",
            "embedding_view_mode": "single",
        },
        "pooled_precise": {
            "id": "precise",
            "encoder_type": "dinov3",
            "embedding_aggregation": "pooled",
            "embedding_view_mode": "tight_context",
        },
        "cradio_summary": {
            "id": "cradio_summary",
            "encoder_type": "cradio",
            "encoder_model": args.cradio_model,
            "cradio_pooling": args.cradio_pooling,
            "embedding_aggregation": "pooled",
            "embedding_view_mode": "single",
        },
    }
    recommendations: List[Dict[str, Any]] = []
    for item in ranked:
        recipe = dict(recipe_lookup.get(str(item.get("recipe_id") or ""), {}))
        if not recipe:
            continue
        recipe.update(
            {
                "canonical_size": args.canonical_size,
                "padding_ratio": args.padding_ratio,
                "embedding_adjustment": "remove_size_bias",
                "measured_class_balanced_nn_purity": item.get("class_balanced_nn_purity"),
                "measured_object_weighted_nn_purity": item.get("object_weighted_nn_purity"),
                "measured_size_axis_abs_correlation": item.get("size_axis_abs_correlation"),
                "reason": "Ranked by this benchmark's class-balanced nearest-neighbor purity, then object-weighted purity and size leakage.",
            }
        )
        recommendations.append(recipe)
    results["recommended_recipes"] = {"class_separation": recommendations}

    report_path = output_dir / "benchmark.json"
    report_path.write_text(json.dumps(api._class_analysis_json_safe(results), indent=2), encoding="utf-8")
    print(json.dumps(api._class_analysis_json_safe({"output": str(report_path), **results}), indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    raise SystemExit(main())
