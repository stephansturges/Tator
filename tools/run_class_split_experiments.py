#!/usr/bin/env python3
"""Run Class Split embedding experiments from the command line."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import localinferenceapi as api


DEFAULT_DINOV3 = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_CRADIO = "nvidia/C-RADIOv4-SO400M"


def _read_labelmap(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_zip_labels(path: Path) -> Dict[str, List[str]]:
    labels: Dict[str, List[str]] = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".txt"):
                continue
            name = Path(info.filename).name
            if not name:
                continue
            text = zf.read(info).decode("utf-8", errors="replace")
            labels[Path(name).stem] = [line.strip() for line in text.splitlines() if line.strip()]
    return labels


def _build_manifest(*, image_dir: Path, label_zip: Path, labelmap: List[str]) -> Dict[str, Any]:
    labels_by_stem = _load_zip_labels(label_zip)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    rows: List[Dict[str, Any]] = []
    for image_path in sorted(image_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in valid_exts:
            continue
        rel = image_path.relative_to(image_dir).as_posix()
        rows.append(
            {
                "split": "train",
                "image_relpath": rel,
                "image_name": image_path.name,
                "frontend_image_key": rel,
                "label_lines": labels_by_stem.get(image_path.stem, []),
            }
        )
    return {
        "dataset_label": image_dir.name or "active snapshot",
        "labelmap": labelmap,
        "images": rows,
        "yolo_layout": "flat",
    }


def _class_scopes(classes: Sequence[str]) -> List[Tuple[str, str]]:
    scopes = [(str(class_name), "selected_class") for class_name in classes if str(class_name or "").strip()]
    scopes.append(("all_classes", "all_classes"))
    return scopes


def _minimum_matrix(sample_cap: int, classes: Sequence[str]) -> List[Dict[str, Any]]:
    base = {
        "encoder_type": "dinov3",
        "encoder_model": DEFAULT_DINOV3,
        "canonical_size": 336,
        "padding_ratio": 0.08,
        "crop_mode": "padded_square",
        "neighbor_k": 15,
        "projection_neighbor_k": 50,
        "dinov3_pooling": "pooler",
        "embedding_aggregation": "pooled",
        "background_mode": "full_crop",
        "embedding_view_mode": "single",
        "embedding_postprocess": "none",
        "batch_size": 64,
        "seed": 42,
        "sample_cap": sample_cap,
    }
    variants = {
        "B1": {"preprocess_mode": "native", "embedding_adjustment": "none", "projection": "pca"},
        "B2": {"preprocess_mode": "canonical", "embedding_adjustment": "none", "projection": "pca"},
        "B3": {"preprocess_mode": "canonical", "embedding_adjustment": "remove_size_bias", "projection": "pca"},
        "B5": {"preprocess_mode": "canonical", "embedding_adjustment": "remove_size_bias", "projection": "umap"},
    }
    scopes = _class_scopes(classes)
    runs: List[Dict[str, Any]] = []
    for class_name, scope in scopes:
        for variant_id, settings in variants.items():
            run = dict(base)
            run.update(settings)
            run["analysis_scope"] = scope
            run["class_name"] = "" if scope == "all_classes" else class_name
            run["run_id"] = f"{variant_id}_{class_name}"
            runs.append(run)
    return runs


def _remaining_lever_matrix(sample_cap: int, classes: Sequence[str]) -> List[Dict[str, Any]]:
    base = {
        "encoder_type": "dinov3",
        "encoder_model": DEFAULT_DINOV3,
        "canonical_size": 336,
        "padding_ratio": 0.08,
        "crop_mode": "padded_square",
        "neighbor_k": 15,
        "projection": "pca",
        "projection_neighbor_k": 50,
        "dinov3_pooling": "pooler",
        "embedding_aggregation": "pooled",
        "background_mode": "full_crop",
        "embedding_view_mode": "single",
        "embedding_adjustment": "remove_size_bias",
        "embedding_postprocess": "none",
        "preprocess_mode": "canonical",
        "batch_size": 64,
        "seed": 42,
        "sample_cap": sample_cap,
    }
    variants = [
        ("baseline", {}),
        ("fast_native_raw", {"preprocess_mode": "native", "embedding_adjustment": "none"}),
        ("fast_224_raw", {"canonical_size": 224, "padding_ratio": 0.04, "embedding_adjustment": "none"}),
        ("crop_tight", {"crop_mode": "tight", "padding_ratio": 0.0}),
        ("crop_padded_bbox", {"crop_mode": "padded", "padding_ratio": 0.08}),
        ("crop_pad_0", {"padding_ratio": 0.0}),
        ("crop_pad_04", {"padding_ratio": 0.04}),
        ("crop_pad_25", {"padding_ratio": 0.25}),
        ("bg_mean_fill", {"background_mode": "mean_fill_outside_box"}),
        ("bg_blur", {"background_mode": "blur_outside_box"}),
        ("bg_darken", {"background_mode": "darken_outside_box"}),
        ("view_tight_standard", {"embedding_view_mode": "tight_standard"}),
        ("view_standard_context", {"embedding_view_mode": "standard_context"}),
        ("view_tight_context", {"embedding_view_mode": "tight_context"}),
        ("pool_cls", {"dinov3_pooling": "cls"}),
        ("pool_patch_mean", {"dinov3_pooling": "patch_mean"}),
        ("pool_cls_patch", {"dinov3_pooling": "cls_patch_concat"}),
        (
            "cradio_summary",
            {
                "encoder_type": "cradio",
                "encoder_model": DEFAULT_CRADIO,
                "canonical_size": 432,
                "cradio_pooling": "summary",
                "batch_size": 16,
            },
        ),
        (
            "cradio_spatial_mean",
            {
                "encoder_type": "cradio",
                "encoder_model": DEFAULT_CRADIO,
                "canonical_size": 432,
                "cradio_pooling": "spatial_mean",
                "batch_size": 16,
            },
        ),
        ("post_pca64", {"embedding_postprocess": "pca64"}),
        ("post_whiten64", {"embedding_postprocess": "whiten64"}),
        ("post_image_bias", {"embedding_postprocess": "remove_image_bias"}),
        ("umap15", {"projection": "umap", "projection_neighbor_k": 15}),
        ("umap50", {"projection": "umap", "projection_neighbor_k": 50}),
        ("clip_vitb32", {"encoder_type": "clip", "encoder_model": "ViT-B/32", "dinov3_pooling": "pooler"}),
    ]
    scopes = _class_scopes(classes)
    runs: List[Dict[str, Any]] = []
    for class_name, scope in scopes:
        for variant_id, settings in variants:
            run = dict(base)
            run.update(settings)
            run["analysis_scope"] = scope
            run["class_name"] = "" if scope == "all_classes" else class_name
            run["run_id"] = f"{variant_id}_{class_name}"
            runs.append(run)
    return runs


def _finalist_matrix(sample_cap: int, classes: Sequence[str]) -> List[Dict[str, Any]]:
    base = {
        "encoder_type": "dinov3",
        "encoder_model": DEFAULT_DINOV3,
        "canonical_size": 336,
        "padding_ratio": 0.08,
        "crop_mode": "padded_square",
        "neighbor_k": 15,
        "projection_neighbor_k": 50,
        "dinov3_pooling": "pooler",
        "embedding_aggregation": "pooled",
        "background_mode": "full_crop",
        "embedding_view_mode": "single",
        "embedding_adjustment": "remove_size_bias",
        "embedding_postprocess": "none",
        "preprocess_mode": "canonical",
        "batch_size": 64,
        "seed": 42,
        "sample_cap": sample_cap,
    }
    variants = [
        ("fast", {"canonical_size": 224, "padding_ratio": 0.04, "embedding_adjustment": "remove_size_bias", "projection": "pca"}),
        ("balanced", {"projection": "pca"}),
        ("balanced_umap", {"projection": "umap", "projection_neighbor_k": 50}),
        ("precise_tight_context", {"embedding_view_mode": "tight_context", "projection": "pca"}),
        ("precise_tight_context_umap", {"embedding_view_mode": "tight_context", "projection": "umap", "projection_neighbor_k": 50}),
    ]
    scopes = _class_scopes(classes)
    runs: List[Dict[str, Any]] = []
    for class_name, scope in scopes:
        for variant_id, settings in variants:
            run = dict(base)
            run.update(settings)
            run["analysis_scope"] = scope
            run["class_name"] = "" if scope == "all_classes" else class_name
            run["run_id"] = f"{variant_id}_{class_name}"
            runs.append(run)
    return runs


def _cradio_matrix(sample_cap: int, classes: Sequence[str]) -> List[Dict[str, Any]]:
    base = {
        "encoder_type": "cradio",
        "encoder_model": DEFAULT_CRADIO,
        "canonical_size": 432,
        "padding_ratio": 0.08,
        "crop_mode": "padded_square",
        "neighbor_k": 15,
        "projection_neighbor_k": 50,
        "dinov3_pooling": "pooler",
        "cradio_pooling": "summary",
        "embedding_aggregation": "pooled",
        "background_mode": "full_crop",
        "embedding_view_mode": "single",
        "embedding_adjustment": "remove_size_bias",
        "embedding_postprocess": "none",
        "preprocess_mode": "canonical",
        "batch_size": 16,
        "seed": 42,
        "sample_cap": sample_cap,
    }
    variants = [
        ("cradio_summary_pca", {"projection": "pca", "cradio_pooling": "summary"}),
        ("cradio_summary_umap", {"projection": "umap", "projection_neighbor_k": 50, "cradio_pooling": "summary"}),
        ("cradio_spatial_mean_pca", {"projection": "pca", "cradio_pooling": "spatial_mean"}),
        (
            "cradio_summary_spatial_concat_pca",
            {"projection": "pca", "cradio_pooling": "summary_spatial_concat"},
        ),
        (
            "cradio_precise_tight_context_pca",
            {"projection": "pca", "embedding_view_mode": "tight_context", "cradio_pooling": "summary"},
        ),
    ]
    scopes = _class_scopes(classes)
    runs: List[Dict[str, Any]] = []
    for class_name, scope in scopes:
        for variant_id, settings in variants:
            run = dict(base)
            run.update(settings)
            run["analysis_scope"] = scope
            run["class_name"] = "" if scope == "all_classes" else class_name
            run["run_id"] = f"{variant_id}_{class_name}"
            runs.append(run)
    return runs


def _json_safe(value: Any) -> Any:
    return api._class_analysis_json_safe(value)


def _variant_from_run_id(run_id: str, class_name: Any, analysis_scope: Any) -> str:
    class_suffix = str(class_name or "").strip()
    if str(analysis_scope or "") == "all_classes" or class_suffix == "all_classes":
        class_suffix = "all_classes"
    if class_suffix:
        suffix = f"_{class_suffix}"
        if run_id.endswith(suffix):
            return run_id[: -len(suffix)] or run_id
    return run_id


def _apply_embedding_postprocess(
    embeddings: np.ndarray,
    records: List[Dict[str, Any]],
    mode: str,
) -> tuple[np.ndarray, Dict[str, Any]]:
    mode_norm = str(mode or "none").strip().lower()
    arr = api._class_analysis_normalize_rows(np.asarray(embeddings, dtype=np.float32))
    if mode_norm in {"", "none", "off"}:
        return arr, {"mode": "none", "applied": False}
    if mode_norm == "remove_image_bias":
        groups: Dict[str, List[int]] = {}
        for idx, record in enumerate(records):
            groups.setdefault(str(record.get("image_relpath") or ""), []).append(idx)
        adjusted = arr.copy()
        applied_groups = 0
        for idxs in groups.values():
            if len(idxs) < 2:
                continue
            group_mean = arr[idxs].mean(axis=0, keepdims=True)
            adjusted[idxs] = arr[idxs] - group_mean
            applied_groups += 1
        return api._class_analysis_normalize_rows(adjusted), {
            "mode": mode_norm,
            "applied": applied_groups > 0,
            "groups": applied_groups,
        }
    if mode_norm in {"pca64", "whiten64"}:
        n_components = min(64, arr.shape[0] - 1, arr.shape[1])
        if n_components < 2:
            return arr, {"mode": mode_norm, "applied": False, "skipped_reason": "not_enough_points"}
        reducer = PCA(n_components=n_components, whiten=(mode_norm == "whiten64"), random_state=42)
        reduced = reducer.fit_transform(arr).astype(np.float32)
        return api._class_analysis_normalize_rows(reduced), {
            "mode": mode_norm,
            "applied": True,
            "n_components": int(n_components),
            "explained_variance": float(np.sum(reducer.explained_variance_ratio_)),
        }
    return arr, {"mode": mode_norm, "applied": False, "skipped_reason": "unknown_mode"}


def _run_one(
    run: Dict[str, Any],
    *,
    manifest: Dict[str, Any],
    image_dir: Path,
    output_root: Path,
    force: bool,
) -> Dict[str, Any]:
    run_id = str(run["run_id"])
    out_dir = output_root / run_id
    result_path = out_dir / "result.json"
    metrics_path = out_dir / "metrics.json"
    if not force and result_path.exists() and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    request = {
        **run,
        "source_mode": "active_workspace",
        "workspace_id": "class_split_experiment",
        "workspace_dir": str(image_dir.resolve()),
        "workspace_manifest": manifest,
        "yolo_layout": "flat",
    }
    job = api.ClassAnalysisJob(job_id=run_id, request=request)
    start = time.perf_counter()
    records, crops, summary = api._class_analysis_collect_records(job.request, job=job, out_dir=out_dir)
    encoder_type = str(run.get("encoder_type") or "dinov3").strip().lower()
    encoder_model = str(run.get("encoder_model") or (DEFAULT_CRADIO if encoder_type == "cradio" else DEFAULT_DINOV3))
    cache_stats: Dict[str, int] = {}
    feats = api._class_analysis_encode_crops(
        crops,
        job=job,
        head={
            "encoder_type": encoder_type,
            "encoder_model": encoder_model,
            "clip_model": encoder_model,
            "normalize_embeddings": True,
            "dinov3_pooling": run.get("dinov3_pooling") or "pooler",
            "cradio_pooling": run.get("cradio_pooling") or "summary",
            "embedding_aggregation": run.get("embedding_aggregation") or "pooled",
            "embedding_salad_head_id": run.get("embedding_salad_head_id") or "",
        },
        batch_size=int(run.get("batch_size") or 64),
        records=records,
        cache_stats=cache_stats,
    )
    if feats is None or feats.size == 0:
        raise RuntimeError(f"{run_id}: embedding_failed")
    raw_embeddings = api._class_analysis_normalize_rows(np.asarray(feats, dtype=np.float32))
    embeddings, adjustment_info = api._class_analysis_apply_embedding_adjustment(
        raw_embeddings,
        records,
        mode=str(run.get("embedding_adjustment") or "none"),
    )
    embeddings, postprocess_info = _apply_embedding_postprocess(
        embeddings,
        records,
        str(run.get("embedding_postprocess") or "none"),
    )
    result = api._class_analysis_build_result(
        records,
        embeddings,
        summary={
            **summary,
            "encoder_type": encoder_type,
            "encoder_model": encoder_model,
            "dinov3_pooling": run.get("dinov3_pooling") or "pooler",
            "cradio_pooling": run.get("cradio_pooling") or "summary",
            "embedding_aggregation": run.get("embedding_aggregation") or "pooled",
            "embedding_salad_head_id": run.get("embedding_salad_head_id") or "",
            "embedding_adjustment": str(run.get("embedding_adjustment") or "none"),
            "embedding_adjustment_info": adjustment_info,
            "embedding_postprocess": str(run.get("embedding_postprocess") or "none"),
            "embedding_postprocess_info": postprocess_info,
            "embedding_cache": cache_stats,
        },
        projection=str(run.get("projection") or "pca"),
        projection_neighbor_k=int(run.get("projection_neighbor_k") or 50),
        neighbor_k=int(run.get("neighbor_k") or 15),
        seed=int(run.get("seed") or 42),
    )
    elapsed = time.perf_counter() - start
    np.savez_compressed(out_dir / "embeddings.npz", embeddings=embeddings, raw_embeddings=raw_embeddings)
    with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")
    result_path.write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(_json_safe(request), indent=2), encoding="utf-8")
    metrics = _metrics_from_result(run_id, run, result, elapsed)
    metrics_path.write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    return metrics


def _metrics_from_result(run_id: str, run: Dict[str, Any], result: Dict[str, Any], elapsed: float) -> Dict[str, Any]:
    summary = result.get("summary") or {}
    diagnostics = result.get("diagnostics") or {}
    strongest = diagnostics.get("strongest_size_axis") or {}
    axis_corr = diagnostics.get("axis_correlations") or {}
    corr_values: List[float] = []
    for axis in axis_corr.values():
        if isinstance(axis, dict):
            corr_values.extend(abs(float(v)) for v in axis.values() if v is not None)
    clusters = result.get("clusters") or {}
    candidates = [c for c in clusters.get("candidates") or [] if isinstance(c, dict) and c.get("silhouette") is not None]
    best_silhouette = max((float(c.get("silhouette")) for c in candidates), default=0.0)
    points = result.get("points") or []
    same_ratios = [float(p.get("same_class_neighbor_ratio") or 0.0) for p in points if isinstance(p, dict)]
    class_ratios: Dict[str, List[float]] = {}
    for point in points:
        if not isinstance(point, dict):
            continue
        class_name = str(point.get("class_name") or point.get("className") or "")
        class_ratios.setdefault(class_name, []).append(float(point.get("same_class_neighbor_ratio") or 0.0))
    class_means = {
        class_name: float(sum(values) / len(values))
        for class_name, values in class_ratios.items()
        if values
    }
    worst_class = min(class_means, key=class_means.get) if class_means else ""
    signed_size_corr = float(strongest.get("correlation") or 0.0)
    cache = summary.get("embedding_cache") or {}
    total_cache = int(cache.get("total") or 0)
    cache_hit_rate = float(cache.get("hits") or 0) / max(1, total_cache)
    metric_class_name = run.get("class_name") or "all_classes"
    return {
        "run_id": run_id,
        "variant": run.get("variant") or _variant_from_run_id(run_id, metric_class_name, run.get("analysis_scope")),
        "analysis_scope": run.get("analysis_scope"),
        "class_name": metric_class_name,
        "encoder_type": run.get("encoder_type"),
        "encoder_model": run.get("encoder_model"),
        "preprocess_mode": run.get("preprocess_mode"),
        "canonical_size": run.get("canonical_size"),
        "crop_mode": run.get("crop_mode"),
        "padding_ratio": run.get("padding_ratio"),
        "dinov3_pooling": run.get("dinov3_pooling"),
        "cradio_pooling": run.get("cradio_pooling"),
        "embedding_aggregation": run.get("embedding_aggregation"),
        "background_mode": run.get("background_mode"),
        "embedding_view_mode": run.get("embedding_view_mode"),
        "embedding_adjustment": run.get("embedding_adjustment"),
        "embedding_postprocess": run.get("embedding_postprocess"),
        "projection": summary.get("projection") or run.get("projection"),
        "projection_neighbor_k": summary.get("projection_neighbor_k"),
        "neighbor_k": summary.get("neighbor_k"),
        "object_count": summary.get("object_count"),
        "raw_object_count": summary.get("raw_object_count"),
        "sample_cap": summary.get("sample_cap"),
        "cache_hit_rate": cache_hit_rate,
        "runtime_seconds": elapsed,
        "strongest_size_axis_metric": strongest.get("metric") or "",
        "strongest_size_axis_correlation": signed_size_corr,
        "strongest_size_axis_abs_correlation": abs(signed_size_corr),
        "mean_abs_size_correlation": float(sum(corr_values) / len(corr_values)) if corr_values else 0.0,
        "kmeans_best_k": (clusters.get("best_k") if isinstance(clusters, dict) else None),
        "kmeans_best_silhouette": best_silhouette,
        "mean_neighbor_same_class_ratio": float(sum(same_ratios) / len(same_ratios)) if same_ratios else 0.0,
        "class_balanced_neighbor_same_class_ratio": float(sum(class_means.values()) / len(class_means)) if class_means else 0.0,
        "worst_class_neighbor_same_class": worst_class,
        "worst_class_neighbor_same_class_ratio": class_means.get(worst_class, 0.0) if worst_class else 0.0,
        "worst_class_neighbor_same_class_count": len(class_ratios.get(worst_class, [])) if worst_class else 0,
        "wrong_class_candidate_count": summary.get("wrong_class_candidate_count"),
    }


def _write_leaderboard(output_root: Path, metrics_rows: List[Dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "variant",
        "analysis_scope",
        "class_name",
        "encoder_type",
        "encoder_model",
        "preprocess_mode",
        "canonical_size",
        "crop_mode",
        "padding_ratio",
        "dinov3_pooling",
        "cradio_pooling",
        "embedding_aggregation",
        "background_mode",
        "embedding_view_mode",
        "embedding_adjustment",
        "embedding_postprocess",
        "projection",
        "projection_neighbor_k",
        "object_count",
        "raw_object_count",
        "sample_cap",
        "cache_hit_rate",
        "runtime_seconds",
        "strongest_size_axis_metric",
        "strongest_size_axis_correlation",
        "strongest_size_axis_abs_correlation",
        "mean_abs_size_correlation",
        "kmeans_best_k",
        "kmeans_best_silhouette",
        "mean_neighbor_same_class_ratio",
        "class_balanced_neighbor_same_class_ratio",
        "worst_class_neighbor_same_class",
        "worst_class_neighbor_same_class_ratio",
        "worst_class_neighbor_same_class_count",
        "wrong_class_candidate_count",
    ]
    with (output_root / "leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)
    (output_root / "metrics.json").write_text(json.dumps(_json_safe(metrics_rows), indent=2), encoding="utf-8")
    lines = ["# Class Split Experiment Report", ""]
    for row in metrics_rows:
        lines.append(
            "- {run_id}: n={object_count}, projection={projection}, size_abs={strongest_size_axis_abs_correlation:.3f}, "
            "size_signed={strongest_size_axis_correlation:.3f}, mean_size_abs={mean_abs_size_correlation:.3f}, "
            "nn_purity={mean_neighbor_same_class_ratio:.3f}, class_balanced_nn={class_balanced_neighbor_same_class_ratio:.3f}, "
            "silhouette={kmeans_best_silhouette:.3f}".format(**row)
        )
    lines.append("")
    all_class_rows = [row for row in metrics_rows if row.get("analysis_scope") == "all_classes"]
    if all_class_rows:
        lines.append("## All-Class Ranking")
        lines.append("")
        for row in sorted(
            all_class_rows,
            key=lambda item: (
                float(item.get("strongest_size_axis_abs_correlation") or 0.0) > 0.35,
                -float(item.get("class_balanced_neighbor_same_class_ratio") or 0.0),
                -float(item.get("mean_neighbor_same_class_ratio") or 0.0),
                float(item.get("strongest_size_axis_abs_correlation") or 0.0),
            ),
        ):
            leakage_state = "high" if float(row.get("strongest_size_axis_abs_correlation") or 0.0) > 0.35 else "ok"
            lines.append(
                "- {variant}: class_balanced_nn={class_balanced_neighbor_same_class_ratio:.3f}, "
                "object_weighted_nn={mean_neighbor_same_class_ratio:.3f}, size_abs={strongest_size_axis_abs_correlation:.3f}, "
                f"leakage={leakage_state}, wrong_candidates={{wrong_class_candidate_count}}, worst_class={{worst_class_neighbor_same_class}} "
                "({worst_class_neighbor_same_class_ratio:.3f})".format(**row)
            )
        lines.append("")
    lines.append("Recommendation rule: compare absolute size-axis leakage, not signed correlation direction. Prefer the simplest recipe that improves class-balanced nearest-neighbor purity while keeping size-axis leakage low; reserve multi-view and background suppression for precise/slow presets only when they improve the screening and finalist matrices.")
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--label-zip", type=Path, required=True)
    parser.add_argument("--labelmap", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("uploads/class_analysis/experiments/class_split"))
    parser.add_argument("--sample-cap", type=int, default=0, help="0 means all objects in each scope.")
    parser.add_argument("--matrix", choices=["minimum", "remaining", "finalists", "cradio"], default="minimum")
    parser.add_argument("--classes", default="", help="Optional comma-separated selected classes. Defaults to the first labelmap classes.")
    parser.add_argument("--selected-class-count", type=int, default=0, help="How many labelmap classes to test when --classes is omitted.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _matrix_default_class_count(matrix: str) -> int:
    if matrix == "finalists":
        return 8
    if matrix == "remaining":
        return 4
    return 2


def _selected_classes(args: argparse.Namespace, labelmap: Sequence[str]) -> List[str]:
    requested = [item.strip() for item in str(args.classes or "").split(",") if item.strip()]
    if requested:
        labelmap_set = set(labelmap)
        unknown = [class_name for class_name in requested if class_name not in labelmap_set]
        if unknown:
            raise SystemExit(f"--classes contains labels not found in --labelmap: {', '.join(unknown)}")
        return requested
    count = int(args.selected_class_count or 0) or _matrix_default_class_count(str(args.matrix or "minimum"))
    return [str(class_name) for class_name in labelmap[: max(0, count)] if str(class_name or "").strip()]


def main() -> None:
    args = parse_args()
    labelmap = _read_labelmap(args.labelmap.expanduser())
    selected_classes = _selected_classes(args, labelmap)
    manifest = _build_manifest(
        image_dir=args.image_dir.expanduser(),
        label_zip=args.label_zip.expanduser(),
        labelmap=labelmap,
    )
    if args.matrix == "remaining":
        runs = _remaining_lever_matrix(sample_cap=max(0, int(args.sample_cap or 0)), classes=selected_classes)
    elif args.matrix == "finalists":
        runs = _finalist_matrix(sample_cap=max(0, int(args.sample_cap or 0)), classes=selected_classes)
    elif args.matrix == "cradio":
        runs = _cradio_matrix(sample_cap=max(0, int(args.sample_cap or 0)), classes=selected_classes)
    else:
        runs = _minimum_matrix(sample_cap=max(0, int(args.sample_cap or 0)), classes=selected_classes)
    if args.dry_run:
        for run in runs:
            print(run["run_id"], json.dumps(run, sort_keys=True))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    metrics_rows: List[Dict[str, Any]] = []
    for index, run in enumerate(runs, start=1):
        print(f"[{index}/{len(runs)}] {run['run_id']} ...", flush=True)
        row = _run_one(
            run,
            manifest=manifest,
            image_dir=args.image_dir.expanduser(),
            output_root=args.output_root,
            force=bool(args.force),
        )
        metrics_rows.append(row)
        _write_leaderboard(args.output_root, metrics_rows)
        print(
            f"  n={row['object_count']} size_abs={row['strongest_size_axis_abs_correlation']:.3f} "
            f"nn={row['mean_neighbor_same_class_ratio']:.3f} cache={row['cache_hit_rate']:.2f}",
            flush=True,
        )
    _write_leaderboard(args.output_root, metrics_rows)


if __name__ == "__main__":
    main()
