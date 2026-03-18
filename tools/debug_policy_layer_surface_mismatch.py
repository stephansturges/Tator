#!/usr/bin/env python3
"""Debug the mismatch between internal policy-layer selection metrics and external detection metrics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools import eval_ensemble_xgb_dedupe as exgb
from tools.policy_runtime import (
    apply_hand_policy,
    apply_selected_policy,
    load_selected_policy,
    predict_base_probabilities,
    resolve_thresholds,
    transform_base_features,
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _parse_meta_rows(meta_raw: Iterable[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in meta_raw:
        if isinstance(row, dict):
            out.append(dict(row))
            continue
        try:
            out.append(json.loads(str(row)))
        except Exception:
            out.append({})
    return out


def _strip_policy_to_gate_only(policy: Dict[str, Any]) -> Dict[str, Any]:
    gate_only: Dict[str, Any] = {}
    for key in (
        "sam_only_min_prob_default",
        "sam_only_min_prob_by_class",
        "consensus_iou_default",
        "consensus_iou_by_class",
        "consensus_iou_by_source_class",
        "consensus_class_aware",
        "sam_bias_scope",
    ):
        if key in policy:
            gate_only[key] = policy[key]
    return gate_only


def _build_detection_rows(
    *,
    meta_rows: Sequence[Dict[str, Any]],
    decision_rows: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    detections_by_image: Dict[str, List[Dict[str, Any]]] = {}
    total_rows = 0
    accepted_rows = 0
    accepted_by_source: Dict[str, int] = {}
    accepted_by_subgroup = {
        "sam_only": 0,
        "sam3_similarity_primary": 0,
        "sam3_text_primary": 0,
        "detector_supported": 0,
    }
    accepted_per_image: Dict[str, int] = {}
    for payload, decision in zip(meta_rows, decision_rows):
        total_rows += 1
        image = str(payload.get("image") or "")
        label = str(payload.get("label") or "").strip().lower()
        bbox = payload.get("bbox_xyxy_px")
        if not image or not label or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        primary, source_list = exgb._normalize_source_fields(payload)
        if not bool(decision.get("accept")):
            continue
        accepted_rows += 1
        accepted_by_source[primary] = int(accepted_by_source.get(primary, 0)) + 1
        if primary == "sam3_similarity":
            accepted_by_subgroup["sam3_similarity_primary"] += 1
        if primary == "sam3_text":
            accepted_by_subgroup["sam3_text_primary"] += 1
        has_detector_support = ("yolo" in source_list) or ("rfdetr" in source_list)
        if primary in {"sam3_text", "sam3_similarity"} and not has_detector_support:
            accepted_by_subgroup["sam_only"] += 1
        if has_detector_support:
            accepted_by_subgroup["detector_supported"] += 1
        accepted_per_image[image] = int(accepted_per_image.get(image, 0)) + 1
        detections_by_image.setdefault(image, []).append(
            {
                "label": label,
                "bbox_xyxy_px": [float(v) for v in bbox[:4]],
                "score": float(decision.get("prob", 0.0)),
                "score_source": primary,
                "source": primary,
                "source_list": sorted(source_list),
            }
        )
    return detections_by_image, {
        "total_rows": int(total_rows),
        "accepted_rows": int(accepted_rows),
        "accepted_rate": (accepted_rows / total_rows) if total_rows > 0 else 0.0,
        "accepted_by_primary_source": accepted_by_source,
        "accepted_by_subgroup": accepted_by_subgroup,
        "accepted_per_image": accepted_per_image,
    }


def _per_image_count_stats(counts: Sequence[int]) -> Dict[str, float]:
    if not counts:
        return {"mean": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    values = np.asarray(list(counts), dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _deduped_counts(
    detections_by_image: Dict[str, List[Dict[str, Any]]],
    *,
    dedupe_iou: float,
    scoreless_iou: float,
) -> Dict[str, Any]:
    deduped_total = 0
    deduped_per_image: Dict[str, int] = {}
    for image, dets in detections_by_image.items():
        merged, _ = exgb._merge_prepass_detections(dets, iou_thr=float(dedupe_iou))
        if float(scoreless_iou) > 0.0:
            merged, _ = exgb._filter_scoreless_detections(merged, iou_thr=float(scoreless_iou))
        deduped_total += len(merged)
        deduped_per_image[image] = len(merged)
    return {
        "deduped_total": int(deduped_total),
        "deduped_per_image": deduped_per_image,
        "deduped_per_image_stats": _per_image_count_stats(deduped_per_image.values()),
    }


def _load_gt(
    *,
    dataset_id: str,
    yolo_root: Path,
    images: Sequence[str],
) -> Tuple[List[str], Dict[str, int], Dict[str, Dict[int, List[List[float]]]], Dict[str, Tuple[int, int]]]:
    labelmap_path = yolo_root / "labelmap.txt"
    labelmap = [line.strip().lower() for line in labelmap_path.read_text().splitlines() if line.strip()]
    name_to_cat = {name: idx for idx, name in enumerate(labelmap)}
    dataset_root = ROOT / "uploads" / "qwen_runs" / "datasets" / dataset_id
    gt_by_image: Dict[str, Dict[int, List[List[float]]]] = {}
    image_sizes: Dict[str, Tuple[int, int]] = {}
    for image in sorted(set(str(x) for x in images if str(x))):
        label_path = yolo_root / "labels" / "val" / f"{Path(image).stem}.txt"
        if not label_path.exists():
            label_path = yolo_root / "labels" / "train" / f"{Path(image).stem}.txt"
        if not label_path.exists():
            continue
        img_path = None
        for split in ("val", "train"):
            candidate = dataset_root / split / image
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception:
            continue
        image_sizes[image] = (int(img_w), int(img_h))
        gt_by_class: Dict[int, List[List[float]]] = {}
        for raw in label_path.read_text().splitlines():
            parts = [p for p in raw.strip().split() if p]
            if len(parts) < 5:
                continue
            try:
                cat_id = int(float(parts[0]))
                cx = float(parts[1])
                cy = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
            except ValueError:
                continue
            gt_by_class.setdefault(cat_id, []).append(exgb._yolo_to_xyxy([cx, cy, bw, bh], img_w, img_h))
        if gt_by_class:
            gt_by_image[image] = gt_by_class
    return labelmap, name_to_cat, gt_by_image, image_sizes


def _evaluate_variant(
    *,
    name: str,
    meta_rows: Sequence[Dict[str, Any]],
    decision_rows: Sequence[Dict[str, Any]],
    gt_by_image: Dict[str, Dict[int, List[List[float]]]],
    name_to_cat: Dict[str, int],
    labelmap: Sequence[str],
    dedupe_iou: float,
    eval_iou: float,
    scoreless_iou: float,
) -> Dict[str, Any]:
    detections_by_image, acceptance = _build_detection_rows(meta_rows=meta_rows, decision_rows=decision_rows)
    metrics = exgb._evaluate_predictions(
        detections_by_image,
        gt_by_image,
        name_to_cat=name_to_cat,
        dedupe_iou=float(dedupe_iou),
        eval_iou=float(eval_iou),
        scoreless_iou=float(scoreless_iou),
        fusion_mode="primary",
        source_weights=None,
        apply_dedupe=True,
    )
    detail = exgb._evaluate_predictions_detailed(
        detections_by_image,
        gt_by_image,
        name_to_cat=name_to_cat,
        labelmap=labelmap,
        dedupe_iou=float(dedupe_iou),
        eval_iou=float(eval_iou),
        scoreless_iou=float(scoreless_iou),
        fusion_mode="primary",
        source_weights=None,
        apply_dedupe=True,
    )
    dedupe = _deduped_counts(
        detections_by_image,
        dedupe_iou=float(dedupe_iou),
        scoreless_iou=float(scoreless_iou),
    )
    return {
        "name": name,
        "metrics": metrics,
        "per_class": detail.get("per_class", []),
        "per_class_per_source": detail.get("per_class_per_source", []),
        "acceptance_audit": {
            **acceptance,
            "accepted_per_image_stats": _per_image_count_stats(acceptance["accepted_per_image"].values()),
        },
        "duplicate_density": {
            "accepted_total": int(acceptance["accepted_rows"]),
            "accepted_per_image_stats": _per_image_count_stats(acceptance["accepted_per_image"].values()),
            **dedupe,
            "accepted_to_deduped_ratio": (
                acceptance["accepted_rows"] / dedupe["deduped_total"]
                if dedupe["deduped_total"] > 0
                else 0.0
            ),
        },
    }


def _decision_rows_threshold_only(
    probs: Sequence[float],
    meta_rows: Sequence[Dict[str, Any]],
    *,
    default_threshold: float,
    thresholds_by_label: Dict[str, float],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, payload in enumerate(meta_rows):
        label = str(payload.get("label") or "").strip().lower()
        thr = float(thresholds_by_label.get(label, default_threshold))
        prob = _safe_float(probs[idx], 0.0)
        accept = bool(prob >= thr)
        out.append(
            {
                "prob_raw": prob,
                "prob": prob,
                "accept": accept,
                "threshold": thr,
                "blocked_reason": None if accept else "threshold",
            }
        )
    return out


def _subset_arrays(
    *,
    meta_rows: Sequence[Dict[str, Any]],
    y: np.ndarray,
    base_probs: np.ndarray,
    learned_probs: np.ndarray,
    val_images: Optional[set[str]],
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    if not val_images:
        return list(meta_rows), y.copy(), base_probs.copy(), learned_probs.copy()
    keep = [idx for idx, row in enumerate(meta_rows) if str(row.get("image") or "") in val_images]
    return (
        [meta_rows[idx] for idx in keep],
        y[keep].copy(),
        base_probs[keep].copy(),
        learned_probs[keep].copy(),
    )


def _top_source_rows(rows: Sequence[Dict[str, Any]], *, limit: int = 8) -> List[Dict[str, Any]]:
    out = list(rows)
    out.sort(key=lambda row: (-int(row.get("fp", 0)), str(row.get("label")), str(row.get("primary_source") or "")))
    return out[:limit]


def _resolve_single_path(parent: Path, pattern: str, *, label: str) -> Path:
    matches = sorted(parent.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {label} in {parent}, got {len(matches)}")
    return matches[0]


def _model_json_from_meta(meta_path: Path) -> Path:
    model_json = meta_path.with_name(meta_path.name.replace(".meta.json", ".json"))
    if not model_json.exists():
        raise FileNotFoundError(f"Missing model json for {meta_path}")
    return model_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seed", default="42")
    parser.add_argument("--lane", default="window")
    parser.add_argument("--dataset", default="qwen_dataset")
    parser.add_argument("--refined-tag", default="text_m1p4__sim_m1p2")
    parser.add_argument("--eval-iou", type=float, default=0.5)
    parser.add_argument("--dedupe-iou", type=float, default=0.75)
    parser.add_argument("--scoreless-iou", type=float, default=0.0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    seed = str(args.seed)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (run_root / "debug_policy_layer_surface_mismatch" / f"seed_{seed}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_npz = run_root / "lanes" / args.lane / "labeled.npz"
    final_seed_dir = run_root / "final_matrix" / args.lane / "full" / f"seed_{seed}"
    base_meta_path = _resolve_single_path(
        final_seed_dir,
        "model_*.meta.json",
        label="final-matrix full-view meta",
    )
    base_model_path = _model_json_from_meta(base_meta_path)
    baseline_seed_dir = (
        run_root / "postrun_sam_bias_magnitude_sweep" / "full" / args.refined_tag / "full" / f"seed_{seed}"
    )
    baseline_meta_path = _resolve_single_path(
        baseline_seed_dir,
        "model_*.meta.json",
        label="magnitude-sweep promoted meta",
    )
    baseline_policy_path = baseline_seed_dir / "policy.json"
    learned_seed_dir = run_root / "postrun_learned_xgb_full_window_eval" / f"seed_{seed}"
    learned_meta_path = _resolve_single_path(
        learned_seed_dir,
        "model_*.meta.json",
        label="learned-xgb promoted meta",
    )

    baseline_meta = _load_json(baseline_meta_path)
    baseline_policy = _load_json(baseline_policy_path)
    learned_meta = _load_json(learned_meta_path)

    data = np.load(labeled_npz, allow_pickle=True)
    X_raw = data["X"].astype(np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    feature_names = [str(name) for name in data.get("feature_names", [])]
    meta_rows = _parse_meta_rows(data["meta"])

    X_base = transform_base_features(X_raw, feature_names, baseline_meta)
    base_probs = predict_base_probabilities(
        X_base,
        meta_rows=meta_rows,
        model_path=base_model_path,
        meta=baseline_meta,
    )
    selected_policy = load_selected_policy(
        learned_meta,
        base_dir=base_model_path.parent,
        meta_dir=learned_meta_path.parent,
    )
    if selected_policy is None:
        raise SystemExit("selected_policy_not_found")
    learned_probs, _ = apply_selected_policy(
        X_full=X_raw,
        feature_names_full=feature_names,
        meta_rows=meta_rows,
        base_probs=base_probs,
        selected_policy=selected_policy,
    )
    selected_policy_meta = selected_policy["meta"]

    base_default_threshold, base_thresholds = resolve_thresholds(baseline_meta)
    learned_default_threshold = float(selected_policy_meta.get("calibrated_threshold") or base_default_threshold)
    learned_thresholds = {
        str(k): float(v)
        for k, v in (selected_policy_meta.get("calibrated_thresholds") or {}).items()
    } or dict(base_thresholds)

    gate_only_policy = _strip_policy_to_gate_only(baseline_policy)

    val_images = {str(v) for v in (baseline_meta.get("split_val_images") or []) if str(v)}
    labelmap, name_to_cat, gt_by_image, _ = _load_gt(
        dataset_id=str(args.dataset),
        yolo_root=ROOT / "uploads" / "clip_dataset_uploads" / f"{args.dataset}_yolo",
        images=[row.get("image") for row in meta_rows],
    )

    variants: Dict[str, Dict[str, Any]] = {}
    for split_name, val_only in (("full", None), ("val", val_images)):
        split_meta_rows, split_y, split_base_probs, split_learned_probs = _subset_arrays(
            meta_rows=meta_rows,
            y=y,
            base_probs=base_probs,
            learned_probs=learned_probs,
            val_images=val_only,
        )
        split_gt = {image: gt for image, gt in gt_by_image.items() if not val_only or image in val_only}

        baseline_decisions = apply_hand_policy(
            probs=split_base_probs,
            meta_rows=split_meta_rows,
            policy=baseline_policy,
            default_threshold=base_default_threshold,
            thresholds_by_label=base_thresholds,
        )
        learned_selected_decisions = _decision_rows_threshold_only(
            split_learned_probs,
            split_meta_rows,
            default_threshold=learned_default_threshold,
            thresholds_by_label=learned_thresholds,
        )
        learned_base_thresholds_decisions = _decision_rows_threshold_only(
            split_learned_probs,
            split_meta_rows,
            default_threshold=base_default_threshold,
            thresholds_by_label=base_thresholds,
        )
        learned_gate_only_decisions = apply_hand_policy(
            probs=split_learned_probs,
            meta_rows=split_meta_rows,
            policy=gate_only_policy,
            default_threshold=base_default_threshold,
            thresholds_by_label=base_thresholds,
        )
        learned_full_hand_decisions = apply_hand_policy(
            probs=split_learned_probs,
            meta_rows=split_meta_rows,
            policy=baseline_policy,
            default_threshold=base_default_threshold,
            thresholds_by_label=base_thresholds,
        )

        split_results = {
            "baseline_hand": _evaluate_variant(
                name="baseline_hand",
                meta_rows=split_meta_rows,
                decision_rows=baseline_decisions,
                gt_by_image=split_gt,
                name_to_cat=name_to_cat,
                labelmap=labelmap,
                dedupe_iou=float(args.dedupe_iou),
                eval_iou=float(args.eval_iou),
                scoreless_iou=float(args.scoreless_iou),
            ),
            "learned_selected": _evaluate_variant(
                name="learned_selected",
                meta_rows=split_meta_rows,
                decision_rows=learned_selected_decisions,
                gt_by_image=split_gt,
                name_to_cat=name_to_cat,
                labelmap=labelmap,
                dedupe_iou=float(args.dedupe_iou),
                eval_iou=float(args.eval_iou),
                scoreless_iou=float(args.scoreless_iou),
            ),
            "learned_base_thresholds": _evaluate_variant(
                name="learned_base_thresholds",
                meta_rows=split_meta_rows,
                decision_rows=learned_base_thresholds_decisions,
                gt_by_image=split_gt,
                name_to_cat=name_to_cat,
                labelmap=labelmap,
                dedupe_iou=float(args.dedupe_iou),
                eval_iou=float(args.eval_iou),
                scoreless_iou=float(args.scoreless_iou),
            ),
            "learned_gate_only": _evaluate_variant(
                name="learned_gate_only",
                meta_rows=split_meta_rows,
                decision_rows=learned_gate_only_decisions,
                gt_by_image=split_gt,
                name_to_cat=name_to_cat,
                labelmap=labelmap,
                dedupe_iou=float(args.dedupe_iou),
                eval_iou=float(args.eval_iou),
                scoreless_iou=float(args.scoreless_iou),
            ),
            "learned_full_hand": _evaluate_variant(
                name="learned_full_hand",
                meta_rows=split_meta_rows,
                decision_rows=learned_full_hand_decisions,
                gt_by_image=split_gt,
                name_to_cat=name_to_cat,
                labelmap=labelmap,
                dedupe_iou=float(args.dedupe_iou),
                eval_iou=float(args.eval_iou),
                scoreless_iou=float(args.scoreless_iou),
            ),
        }
        baseline_metrics = split_results["baseline_hand"]["metrics"]
        for variant_name, payload in split_results.items():
            payload["compare_to_baseline_hand"] = {
                f"delta_{field}": _safe_float(payload["metrics"].get(field), 0.0) - _safe_float(baseline_metrics.get(field), 0.0)
                for field in ("tp", "fp", "fn", "precision", "recall", "f1")
            }
            payload["top_fp_sources"] = _top_source_rows(payload.get("per_class_per_source", []))
        variants[split_name] = split_results

    summary = {
        "run_root": str(run_root),
        "seed": seed,
        "lane": str(args.lane),
        "refined_tag": str(args.refined_tag),
        "thresholds": {
            "base_default": base_default_threshold,
            "learned_default": learned_default_threshold,
            "base_by_label": base_thresholds,
            "learned_by_label": learned_thresholds,
        },
        "policy_components": {
            "baseline_policy_keys": sorted(baseline_policy.keys()),
            "gate_only_policy_keys": sorted(gate_only_policy.keys()),
            "selected_policy_variant": str(selected_policy.get("variant") or ""),
        },
        "variants": variants,
    }
    _write_json(output_dir / "results_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
