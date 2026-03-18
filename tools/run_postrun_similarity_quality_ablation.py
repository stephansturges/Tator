#!/usr/bin/env python3
"""Evaluate the SAM3 similarity-quality branch on the frozen postrun baseline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _alpha_tag(alpha: float) -> str:
    return f"a{str(float(alpha)).replace('-', 'm').replace('.', 'p')}"


def _resolve_best_refined_tag(run_root: Path) -> str:
    ranked = _load_json(run_root / "postrun_sam_bias_magnitude_sweep" / "results_ranked.json")
    for section in ("full", "pilot"):
        rows = ranked.get(section) if isinstance(ranked.get(section), list) else []
        if rows:
            tag = str((rows[0] or {}).get("tag") or "").strip()
            if tag:
                return tag
    raise SystemExit("missing_refined_policy_rankings")


def _extract_eval_metrics(payload: Dict[str, Any]) -> Dict[str, float]:
    post_xgb = payload.get("metric_tiers", {}).get("post_xgb", {}).get("accepted_all", {})
    post_cluster_union = (
        payload.get("metric_tiers", {})
        .get("post_cluster", {})
        .get("source_attributed", {})
        .get("yolo_rfdetr_union", {})
    )
    coverage = payload.get("coverage_upper_bound", {}).get("candidate_all", {})
    precision = _safe_float(post_xgb.get("precision"), _safe_float(payload.get("precision"), 0.0))
    recall = _safe_float(post_xgb.get("recall"), _safe_float(payload.get("recall"), 0.0))
    f1 = _safe_float(post_xgb.get("f1"), _safe_float(payload.get("f1"), 0.0))
    union_f1 = _safe_float(post_cluster_union.get("f1"), 0.0)
    coverage_ub = _safe_float(coverage.get("recall_upper_bound"), 0.0)
    coverage_pres = (recall / coverage_ub) if coverage_ub > 0.0 else 0.0
    return {
        "tp": _safe_float(payload.get("tp"), 0.0),
        "fp": _safe_float(payload.get("fp"), 0.0),
        "fn": _safe_float(payload.get("fn"), 0.0),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "union_f1": union_f1,
        "delta_vs_union_f1": f1 - union_f1,
        "coverage_upper_bound": coverage_ub,
        "coverage_preservation": coverage_pres,
    }


def _compare_to_baseline(metrics: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for field in ("tp", "fp", "fn", "precision", "recall", "f1", "coverage_preservation"):
        out[f"delta_{field}"] = _safe_float(metrics.get(field), 0.0) - _safe_float(baseline.get(field), 0.0)
    return out


def _write_val_images_file(path: Path, images: Iterable[str]) -> None:
    rows = [str(image).strip() for image in images if str(image).strip()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _resolve_view_context(run_root: Path, lane: str, view: str) -> Dict[str, Path]:
    if view == "intersection":
        return {
            "labeled_npz": run_root / "views" / f"{lane}_intersection.labeled.npz",
            "prepass_jsonl": run_root / "views" / "window_intersection.jsonl",
        }
    if view == "full":
        return {
            "labeled_npz": run_root / "lanes" / lane / "labeled.npz",
            "prepass_jsonl": run_root / "prepass" / "window.jsonl",
        }
    raise SystemExit(f"unsupported_view:{view}")


def _resolve_baseline_dir(run_root: Path, refined_tag: str, view: str, seed: str) -> Path:
    stage = "pilot" if view == "intersection" else "full"
    path = run_root / "postrun_sam_bias_magnitude_sweep" / stage / refined_tag / view / f"seed_{seed}"
    if not path.exists():
        raise SystemExit(f"missing_baseline_dir:{path}")
    return path


def _resolve_final_meta(run_root: Path, lane: str, view: str, seed: str) -> Path:
    path = run_root / "final_matrix" / lane / view / f"seed_{seed}" / "model_7ce3c1b42a_9e14a28553.meta.json"
    if not path.exists():
        raise SystemExit(f"missing_final_meta:{path}")
    return path


def _mean_dict(rows: List[Dict[str, float]], fields: Iterable[str]) -> Dict[str, float]:
    if not rows:
        return {}
    return {f"mean_{field}": mean(_safe_float(row.get(field), 0.0) for row in rows) for field in fields}


def _render_report(summary: Dict[str, Any]) -> str:
    lines = ["# Postrun SAM3 Similarity Quality Ablation", ""]
    lines.append(f"- Run root: `{summary['run_root']}`")
    lines.append(f"- Lane: `{summary['lane']}`")
    lines.append(f"- Refined baseline tag: `{summary['refined_tag']}`")
    lines.append(f"- Text-quality alpha: `{summary['sam3_text_quality_alpha']}`")
    lines.append("")
    for section in ("pilot", "full"):
        rows = summary.get(section) or []
        if not rows:
            continue
        lines.append(f"## `{section}`")
        lines.append("")
        for row in rows:
            lines.append(
                f"- `{row['tag']}`: P={row['mean_precision']:.4f} R={row['mean_recall']:.4f} "
                f"F1={row['mean_f1']:.4f} dF1={row['mean_delta_f1']:+.4f} "
                f"dTP={row['mean_delta_tp']:+.1f} dFP={row['mean_delta_fp']:+.1f} dFN={row['mean_delta_fn']:+.1f}"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run postrun SAM3 similarity-quality ablation.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--lane", default="window")
    parser.add_argument("--refined-tag", default=None)
    parser.add_argument("--alphas", default="0.2,0.35,0.5,0.8")
    parser.add_argument("--seeds", default="42,1337,2025")
    parser.add_argument("--top-k-full", type=int, default=2)
    parser.add_argument("--sam3-text-quality-alpha", type=float, default=0.8)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (run_root / "postrun_similarity_quality_ablation").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    refined_tag = args.refined_tag or _resolve_best_refined_tag(run_root)
    alphas = [float(chunk.strip()) for chunk in str(args.alphas).split(",") if chunk.strip()]
    seeds = [chunk.strip() for chunk in str(args.seeds).split(",") if chunk.strip()]

    raw_rows: List[Dict[str, Any]] = []
    pilot_scores: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for alpha in alphas:
        tag = _alpha_tag(alpha)
        for seed in seeds:
            view = "intersection"
            context = _resolve_view_context(run_root, args.lane, view)
            baseline_dir = _resolve_baseline_dir(run_root, refined_tag, view, seed)
            baseline_payload = _load_json(baseline_dir / "eval.json")
            baseline_metrics = _extract_eval_metrics(baseline_payload)
            base_meta_path = _resolve_final_meta(run_root, args.lane, view, seed)
            base_meta = _load_json(base_meta_path)
            xgb_params = base_meta.get("xgb_params") if isinstance(base_meta.get("xgb_params"), dict) else {}
            split_val_images = base_meta.get("split_val_images") if isinstance(base_meta.get("split_val_images"), list) else []
            run_dir = output_dir / "pilot" / tag / view / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            model_prefix = run_dir / "model"
            eval_json = run_dir / "eval.json"
            if eval_json.exists() and not args.force:
                payload = _load_json(eval_json)
                metrics = _extract_eval_metrics(payload)
            else:
                val_images_file = run_dir / "val_images.txt"
                _write_val_images_file(val_images_file, split_val_images)
                train_cmd = [
                    sys.executable,
                    "tools/train_ensemble_xgb.py",
                    "--input",
                    str(context["labeled_npz"]),
                    "--output",
                    str(model_prefix),
                    "--seed",
                    str(int(seed)),
                    "--val-ratio",
                    "0.2",
                    "--max-depth",
                    str(int(_safe_float(xgb_params.get("max_depth"), 8))),
                    "--n-estimators",
                    str(int(_safe_float(base_meta.get("n_estimators"), 900))),
                    "--learning-rate",
                    str(_safe_float(xgb_params.get("eta"), 0.01)),
                    "--subsample",
                    str(_safe_float(xgb_params.get("subsample"), 0.8)),
                    "--colsample-bytree",
                    str(_safe_float(xgb_params.get("colsample_bytree"), 0.8)),
                    "--min-child-weight",
                    str(_safe_float(xgb_params.get("min_child_weight"), 1.0)),
                    "--gamma",
                    str(_safe_float(xgb_params.get("gamma"), 0.0)),
                    "--reg-lambda",
                    str(_safe_float(xgb_params.get("lambda"), 1.0)),
                    "--reg-alpha",
                    str(_safe_float(xgb_params.get("alpha"), 0.0)),
                    "--tree-method",
                    str(xgb_params.get("tree_method") or "hist"),
                    "--max-bin",
                    str(int(_safe_float(xgb_params.get("max_bin"), 256))),
                    "--early-stopping-rounds",
                    "50",
                    "--threshold-steps",
                    "300",
                    "--optimize",
                    "f1",
                    "--target-fp-ratio",
                    "0.2",
                    "--min-recall",
                    "0.6",
                    "--per-class",
                    "--fixed-val-images",
                    str(val_images_file),
                    "--train-sam3-text-quality",
                    "--sam3-text-quality-alpha",
                    str(float(args.sam3_text_quality_alpha)),
                    "--train-sam3-similarity-quality",
                    "--sam3-similarity-quality-alpha",
                    str(float(alpha)),
                ]
                if bool(base_meta.get("log1p_counts")):
                    train_cmd.append("--log1p-counts")
                if bool(base_meta.get("standardize")):
                    train_cmd.append("--standardize")
                split_head = base_meta.get("split_head") if isinstance(base_meta.get("split_head"), dict) else {}
                if bool(split_head.get("enabled")):
                    train_cmd.append("--split-head-by-support")
                _run(train_cmd)
                model_json = model_prefix.with_suffix(".json")
                model_meta = model_prefix.with_suffix(".meta.json")
                _run(
                    [
                        sys.executable,
                        "tools/tune_ensemble_thresholds_xgb.py",
                        "--model",
                        str(model_json),
                        "--meta",
                        str(model_meta),
                        "--data",
                        str(context["labeled_npz"]),
                        "--dataset",
                        "qwen_dataset",
                        "--optimize",
                        "f1",
                        "--target-fp-ratio",
                        "0.2",
                        "--relax-fp-ratio",
                        "0.2",
                        "--min-recall",
                        "0.6",
                        "--steps",
                        "300",
                        "--eval-iou",
                        "0.5",
                        "--dedupe-iou",
                        "0.75",
                        "--scoreless-iou",
                        "0.0",
                        "--use-val-split",
                    ]
                )
                result = subprocess.run(
                    [
                        sys.executable,
                        "tools/eval_ensemble_xgb_dedupe.py",
                        "--model",
                        str(model_json),
                        "--meta",
                        str(model_meta),
                        "--data",
                        str(context["labeled_npz"]),
                        "--dataset",
                        "qwen_dataset",
                        "--prepass-jsonl",
                        str(context["prepass_jsonl"]),
                        "--eval-iou",
                        "0.5",
                        "--eval-iou-grid",
                        "0.5",
                        "--dedupe-iou",
                        "0.75",
                        "--scoreless-iou",
                        "0.0",
                        "--use-val-split",
                        "--policy-json",
                        str(baseline_dir / "policy.json"),
                    ],
                    cwd=str(ROOT),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                eval_json.write_text(result.stdout.strip() + "\n", encoding="utf-8")
                payload = json.loads(result.stdout)
                metrics = _extract_eval_metrics(payload)
            compare = _compare_to_baseline(metrics, baseline_metrics)
            row = {
                "tag": tag,
                "alpha": float(alpha),
                "view": view,
                "seed": seed,
                "metrics": metrics,
                "baseline": baseline_metrics,
                "compare_to_baseline": compare,
            }
            raw_rows.append(row)
            pilot_scores[tag].append(compare | metrics)

    pilot_summary: List[Dict[str, Any]] = []
    for tag, rows in pilot_scores.items():
        pilot_summary.append(
            {
                "tag": tag,
                "mean_precision": mean(_safe_float(row.get("precision"), 0.0) for row in rows),
                "mean_recall": mean(_safe_float(row.get("recall"), 0.0) for row in rows),
                "mean_f1": mean(_safe_float(row.get("f1"), 0.0) for row in rows),
                "mean_delta_f1": mean(_safe_float(row.get("delta_f1"), 0.0) for row in rows),
                "mean_delta_tp": mean(_safe_float(row.get("delta_tp"), 0.0) for row in rows),
                "mean_delta_fp": mean(_safe_float(row.get("delta_fp"), 0.0) for row in rows),
                "mean_delta_fn": mean(_safe_float(row.get("delta_fn"), 0.0) for row in rows),
            }
        )
    pilot_summary.sort(key=lambda row: (-_safe_float(row.get("mean_f1")), -_safe_float(row.get("mean_delta_f1"))))

    promoted = [row["tag"] for row in pilot_summary if _safe_float(row.get("mean_delta_f1"), 0.0) > 0.0][: max(0, int(args.top_k_full))]

    full_summary: List[Dict[str, Any]] = []
    for tag in promoted:
        alpha = next(item["alpha"] for item in pilot_summary if item["tag"] == tag)
        full_rows: List[Dict[str, float]] = []
        for seed in seeds:
            view = "full"
            context = _resolve_view_context(run_root, args.lane, view)
            baseline_dir = _resolve_baseline_dir(run_root, refined_tag, view, seed)
            baseline_payload = _load_json(baseline_dir / "eval.json")
            baseline_metrics = _extract_eval_metrics(baseline_payload)
            base_meta_path = _resolve_final_meta(run_root, args.lane, view, seed)
            base_meta = _load_json(base_meta_path)
            xgb_params = base_meta.get("xgb_params") if isinstance(base_meta.get("xgb_params"), dict) else {}
            split_val_images = base_meta.get("split_val_images") if isinstance(base_meta.get("split_val_images"), list) else []
            run_dir = output_dir / "full" / tag / view / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            model_prefix = run_dir / "model"
            eval_json = run_dir / "eval.json"
            if eval_json.exists() and not args.force:
                payload = _load_json(eval_json)
                metrics = _extract_eval_metrics(payload)
            else:
                val_images_file = run_dir / "val_images.txt"
                _write_val_images_file(val_images_file, split_val_images)
                train_cmd = [
                    sys.executable,
                    "tools/train_ensemble_xgb.py",
                    "--input",
                    str(context["labeled_npz"]),
                    "--output",
                    str(model_prefix),
                    "--seed",
                    str(int(seed)),
                    "--val-ratio",
                    "0.2",
                    "--max-depth",
                    str(int(_safe_float(xgb_params.get("max_depth"), 8))),
                    "--n-estimators",
                    str(int(_safe_float(base_meta.get("n_estimators"), 900))),
                    "--learning-rate",
                    str(_safe_float(xgb_params.get("eta"), 0.01)),
                    "--subsample",
                    str(_safe_float(xgb_params.get("subsample"), 0.8)),
                    "--colsample-bytree",
                    str(_safe_float(xgb_params.get("colsample_bytree"), 0.8)),
                    "--min-child-weight",
                    str(_safe_float(xgb_params.get("min_child_weight"), 1.0)),
                    "--gamma",
                    str(_safe_float(xgb_params.get("gamma"), 0.0)),
                    "--reg-lambda",
                    str(_safe_float(xgb_params.get("lambda"), 1.0)),
                    "--reg-alpha",
                    str(_safe_float(xgb_params.get("alpha"), 0.0)),
                    "--tree-method",
                    str(xgb_params.get("tree_method") or "hist"),
                    "--max-bin",
                    str(int(_safe_float(xgb_params.get("max_bin"), 256))),
                    "--early-stopping-rounds",
                    "50",
                    "--threshold-steps",
                    "300",
                    "--optimize",
                    "f1",
                    "--target-fp-ratio",
                    "0.2",
                    "--min-recall",
                    "0.6",
                    "--per-class",
                    "--fixed-val-images",
                    str(val_images_file),
                    "--train-sam3-text-quality",
                    "--sam3-text-quality-alpha",
                    str(float(args.sam3_text_quality_alpha)),
                    "--train-sam3-similarity-quality",
                    "--sam3-similarity-quality-alpha",
                    str(float(alpha)),
                ]
                if bool(base_meta.get("log1p_counts")):
                    train_cmd.append("--log1p-counts")
                if bool(base_meta.get("standardize")):
                    train_cmd.append("--standardize")
                split_head = base_meta.get("split_head") if isinstance(base_meta.get("split_head"), dict) else {}
                if bool(split_head.get("enabled")):
                    train_cmd.append("--split-head-by-support")
                _run(train_cmd)
                model_json = model_prefix.with_suffix(".json")
                model_meta = model_prefix.with_suffix(".meta.json")
                _run(
                    [
                        sys.executable,
                        "tools/tune_ensemble_thresholds_xgb.py",
                        "--model",
                        str(model_json),
                        "--meta",
                        str(model_meta),
                        "--data",
                        str(context["labeled_npz"]),
                        "--dataset",
                        "qwen_dataset",
                        "--optimize",
                        "f1",
                        "--target-fp-ratio",
                        "0.2",
                        "--relax-fp-ratio",
                        "0.2",
                        "--min-recall",
                        "0.6",
                        "--steps",
                        "300",
                        "--eval-iou",
                        "0.5",
                        "--dedupe-iou",
                        "0.75",
                        "--scoreless-iou",
                        "0.0",
                        "--use-val-split",
                    ]
                )
                result = subprocess.run(
                    [
                        sys.executable,
                        "tools/eval_ensemble_xgb_dedupe.py",
                        "--model",
                        str(model_json),
                        "--meta",
                        str(model_meta),
                        "--data",
                        str(context["labeled_npz"]),
                        "--dataset",
                        "qwen_dataset",
                        "--prepass-jsonl",
                        str(context["prepass_jsonl"]),
                        "--eval-iou",
                        "0.5",
                        "--eval-iou-grid",
                        "0.5",
                        "--dedupe-iou",
                        "0.75",
                        "--scoreless-iou",
                        "0.0",
                        "--use-val-split",
                        "--policy-json",
                        str(baseline_dir / "policy.json"),
                    ],
                    cwd=str(ROOT),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                eval_json.write_text(result.stdout.strip() + "\n", encoding="utf-8")
                payload = json.loads(result.stdout)
                metrics = _extract_eval_metrics(payload)
            compare = _compare_to_baseline(metrics, baseline_metrics)
            row = {
                "tag": tag,
                "alpha": float(alpha),
                "view": view,
                "seed": seed,
                "metrics": metrics,
                "baseline": baseline_metrics,
                "compare_to_baseline": compare,
            }
            raw_rows.append(row)
            full_rows.append(compare | metrics)
        full_summary.append(
            {
                "tag": tag,
                "mean_precision": mean(_safe_float(row.get("precision"), 0.0) for row in full_rows),
                "mean_recall": mean(_safe_float(row.get("recall"), 0.0) for row in full_rows),
                "mean_f1": mean(_safe_float(row.get("f1"), 0.0) for row in full_rows),
                "mean_delta_f1": mean(_safe_float(row.get("delta_f1"), 0.0) for row in full_rows),
                "mean_delta_tp": mean(_safe_float(row.get("delta_tp"), 0.0) for row in full_rows),
                "mean_delta_fp": mean(_safe_float(row.get("delta_fp"), 0.0) for row in full_rows),
                "mean_delta_fn": mean(_safe_float(row.get("delta_fn"), 0.0) for row in full_rows),
            }
        )
    full_summary.sort(key=lambda row: (-_safe_float(row.get("mean_f1")), -_safe_float(row.get("mean_delta_f1"))))

    summary = {
        "run_root": str(run_root),
        "lane": args.lane,
        "refined_tag": refined_tag,
        "sam3_text_quality_alpha": float(args.sam3_text_quality_alpha),
        "pilot": pilot_summary,
        "full": full_summary,
        "promoted": promoted,
    }
    _write_json(output_dir / "results_raw.json", {"runs": raw_rows})
    _write_json(output_dir / "results_ranked.json", summary)
    (output_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
