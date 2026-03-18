#!/usr/bin/env python3
"""Run a corrected full-window external eval for the SAM3 similarity-quality head."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _alpha_tag(alpha: float) -> str:
    return f"a{str(float(alpha)).replace('-', 'm').replace('.', 'p')}"


def _resolve_single_path(parent: Path, pattern: str, *, label: str) -> Path:
    matches = sorted(parent.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {label} in {parent}, got {len(matches)}")
    return matches[0]


def _wait_for_process_pattern(pattern: str, *, poll_seconds: int) -> None:
    raw = str(pattern or "").strip()
    if not raw:
        return
    self_pid = os.getpid()
    while True:
        result = subprocess.run(
            ["pgrep", "-f", raw],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return
        pids = {
            int(line.strip())
            for line in str(result.stdout or "").splitlines()
            if line.strip().isdigit()
        }
        if not {pid for pid in pids if pid != self_pid}:
            return
        time.sleep(max(5, int(poll_seconds)))


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


def _compare(metrics: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for field in ("tp", "fp", "fn", "precision", "recall", "f1", "coverage_preservation"):
        out[f"delta_{field}"] = _safe_float(metrics.get(field), 0.0) - _safe_float(baseline.get(field), 0.0)
    return out


def _build_decision_summary(
    *,
    run_root: Path,
    lane: str,
    refined_tag: str,
    ranked: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    winner = dict(ranked[0]) if ranked else {}
    winner_tag = str(winner.get("tag") or "").strip()
    winner_rows = [row for row in rows if str(row.get("tag") or "") == winner_tag]
    required_non_negative = max(1, min(2, len(winner_rows)))
    non_negative_seed_count = sum(
        1 for row in winner_rows if _safe_float(row.get("compare_to_baseline", {}).get("delta_f1"), 0.0) >= 0.0
    )
    status = "rejected"
    reason_codes: List[str] = []
    if winner and _safe_float(winner.get("mean_delta_f1"), 0.0) >= 0.001 and non_negative_seed_count >= required_non_negative:
        status = "promoted"
        reason_codes.append("mean_delta_f1_cleared_gate")
        reason_codes.append("seed_non_negative_gate_cleared")
    else:
        if _safe_float(winner.get("mean_delta_f1"), 0.0) < 0.001:
            reason_codes.append("mean_delta_f1_below_gate")
        if non_negative_seed_count < required_non_negative:
            reason_codes.append("seed_non_negative_gate_failed")
    promoted_alpha = float(winner["alpha"]) if status == "promoted" and winner.get("alpha") is not None else None
    promoted_config = (
        {
            "train_sam3_similarity_quality": True,
            "sam3_similarity_quality_alpha": promoted_alpha,
        }
        if promoted_alpha is not None
        else {
            "train_sam3_similarity_quality": False,
            "sam3_similarity_quality_alpha": None,
        }
    )
    return {
        "stage": "postrun_similarity_quality_full_window_eval",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_root": str(run_root),
        "lane": lane,
        "refined_tag": refined_tag,
        "winner_tag": winner_tag or None,
        "winner_metrics": winner,
        "non_negative_seed_count": int(non_negative_seed_count),
        "required_non_negative_seed_count": int(required_non_negative),
        "seed_count": len(winner_rows),
        "status": status,
        "promoted_alpha": promoted_alpha,
        "promoted_config": promoted_config,
        "reason_codes": reason_codes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--lane", default="window")
    parser.add_argument("--refined-tag", default="text_m1p4__sim_m1p2")
    parser.add_argument("--alphas", default="0.2,0.35,0.5,0.8")
    parser.add_argument("--seeds", default="42,1337,2025")
    parser.add_argument("--sam3-text-quality-alpha", type=float, default=0.8)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--wait-for-process-pattern", default="")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (run_root / "postrun_similarity_quality_full_window_eval").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _wait_for_process_pattern(args.wait_for_process_pattern, poll_seconds=int(args.poll_seconds))
    alphas = [float(chunk.strip()) for chunk in str(args.alphas).split(",") if chunk.strip()]
    seeds = [chunk.strip() for chunk in str(args.seeds).split(",") if chunk.strip()]

    labeled_npz = run_root / "lanes" / args.lane / "labeled.npz"
    prepass_jsonl = run_root / "prepass" / "window.jsonl"
    rows: List[Dict[str, Any]] = []

    for alpha in alphas:
        tag = _alpha_tag(alpha)
        for seed in seeds:
            final_seed_dir = run_root / "final_matrix" / args.lane / "full" / f"seed_{seed}"
            base_meta_path = _resolve_single_path(
                final_seed_dir,
                "model_*.meta.json",
                label="final-matrix full-view meta",
            )
            base_meta = _load_json(base_meta_path)
            xgb_params = base_meta.get("xgb_params") if isinstance(base_meta.get("xgb_params"), dict) else {}
            split_val_images = base_meta.get("split_val_images") if isinstance(base_meta.get("split_val_images"), list) else []

            baseline_dir = (
                run_root
                / "postrun_sam_bias_magnitude_sweep"
                / "full"
                / args.refined_tag
                / "full"
                / f"seed_{seed}"
            )
            baseline_payload = _load_json(baseline_dir / "eval.json")
            baseline_metrics = _extract_eval_metrics(baseline_payload)

            run_dir = output_dir / tag / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            model_prefix = run_dir / "model"
            model_json = model_prefix.with_suffix(".json")
            model_meta = model_prefix.with_suffix(".meta.json")
            eval_json = run_dir / "eval.json"
            analysis_json = run_dir / "analysis.json"
            val_images_file = run_dir / "val_images.txt"

            if not eval_json.exists() or args.force:
                val_images_file.write_text(
                    "\n".join(str(image).strip() for image in split_val_images if str(image).strip()) + "\n",
                    encoding="utf-8",
                )
                train_cmd = [
                    sys.executable,
                    "tools/train_ensemble_xgb.py",
                    "--input",
                    str(labeled_npz),
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
                subprocess.run(train_cmd, cwd=str(ROOT), check=True)

                subprocess.run(
                    [
                        sys.executable,
                        "tools/tune_ensemble_thresholds_xgb.py",
                        "--model",
                        str(model_json),
                        "--meta",
                        str(model_meta),
                        "--data",
                        str(labeled_npz),
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
                    ],
                    cwd=str(ROOT),
                    check=True,
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
                        str(labeled_npz),
                        "--dataset",
                        "qwen_dataset",
                        "--prepass-jsonl",
                        str(prepass_jsonl),
                        "--eval-iou",
                        "0.5",
                        "--eval-iou-grid",
                        "0.5",
                        "--dedupe-iou",
                        "0.75",
                        "--scoreless-iou",
                        "0.0",
                        "--policy-json",
                        str(baseline_dir / "policy.json"),
                        "--analysis-json",
                        str(analysis_json),
                    ],
                    cwd=str(ROOT),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                eval_json.write_text(result.stdout.strip() + "\n", encoding="utf-8")

            payload = _load_json(eval_json)
            metrics = _extract_eval_metrics(payload)
            rows.append(
                {
                    "tag": tag,
                    "alpha": float(alpha),
                    "seed": seed,
                    "metrics": metrics,
                    "baseline_metrics": baseline_metrics,
                    "compare_to_baseline": _compare(metrics, baseline_metrics),
                    "eval_json": str(eval_json.resolve()),
                    "analysis_json": str(analysis_json.resolve()),
                }
            )

    summary_rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        tag = _alpha_tag(alpha)
        tagged = [row for row in rows if row["tag"] == tag]
        summary_rows.append(
            {
                "tag": tag,
                "alpha": float(alpha),
                "mean_precision": mean(_safe_float(row["metrics"].get("precision"), 0.0) for row in tagged),
                "mean_recall": mean(_safe_float(row["metrics"].get("recall"), 0.0) for row in tagged),
                "mean_f1": mean(_safe_float(row["metrics"].get("f1"), 0.0) for row in tagged),
                "mean_delta_f1": mean(_safe_float(row["compare_to_baseline"].get("delta_f1"), 0.0) for row in tagged),
                "mean_delta_tp": mean(_safe_float(row["compare_to_baseline"].get("delta_tp"), 0.0) for row in tagged),
                "mean_delta_fp": mean(_safe_float(row["compare_to_baseline"].get("delta_fp"), 0.0) for row in tagged),
                "mean_delta_fn": mean(_safe_float(row["compare_to_baseline"].get("delta_fn"), 0.0) for row in tagged),
                "mean_coverage_preservation": mean(
                    _safe_float(row["metrics"].get("coverage_preservation"), 0.0) for row in tagged
                ),
            }
        )
    summary_rows.sort(
        key=lambda row: (
            -_safe_float(row.get("mean_f1"), 0.0),
            -_safe_float(row.get("mean_delta_f1"), 0.0),
            float(row.get("alpha", 0.0)),
        )
    )

    summary = {
        "run_root": str(run_root),
        "lane": args.lane,
        "refined_tag": args.refined_tag,
        "sam3_text_quality_alpha": float(args.sam3_text_quality_alpha),
        "rows": rows,
        "ranked": summary_rows,
    }
    decision_summary = _build_decision_summary(
        run_root=run_root,
        lane=args.lane,
        refined_tag=args.refined_tag,
        ranked=summary_rows,
        rows=rows,
    )
    _write_json(output_dir / "results_summary.json", summary)
    _write_json(output_dir / "decision_summary.json", decision_summary)
    subprocess.run(
        [
            sys.executable,
            "tools/build_calibration_report_bundle.py",
            "--results-summary-json",
            str(output_dir / "results_summary.json"),
            "--analysis-json-glob",
            str(output_dir / "a*" / "seed_*" / "analysis.json"),
            "--output-json",
            str(output_dir / "report_bundle.json"),
            "--output-md",
            str(output_dir / "report_bundle.md"),
        ],
        cwd=str(ROOT),
        check=True,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
