#!/usr/bin/env python3
"""Train/evaluate the forced learned-XGB second stage on the full 4000-image windowed lane."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forced learned-XGB full-windowed evaluation.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--refined-tag", default="text_m1p4__sim_m1p2")
    parser.add_argument("--seeds", default="42,1337,2025")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (run_root / "postrun_learned_xgb_full_window_eval").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [chunk.strip() for chunk in str(args.seeds).split(",") if chunk.strip()]

    labeled_npz = run_root / "lanes" / "window" / "labeled.npz"
    prepass_jsonl = run_root / "prepass" / "window.jsonl"
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        final_seed_dir = run_root / "final_matrix" / "window" / "full" / f"seed_{seed}"
        final_meta = _resolve_single_path(
            final_seed_dir,
            "model_*.meta.json",
            label="final-matrix full-view meta",
        )
        final_model = _model_json_from_meta(final_meta)
        baseline_dir = run_root / "postrun_sam_bias_magnitude_sweep" / "full" / args.refined_tag / "full" / f"seed_{seed}"
        meta_source = _resolve_single_path(
            baseline_dir,
            "model_*.meta.json",
            label="magnitude-sweep promoted meta",
        )
        baseline_eval = _load_json(baseline_dir / "eval.json")
        baseline_metrics = _extract_eval_metrics(baseline_eval)

        seed_out = output_dir / f"seed_{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        meta_copy = seed_out / meta_source.name
        shutil.copy2(meta_source, meta_copy)
        analysis_json = seed_out / "analysis.json"

        subprocess.run(
            [
                sys.executable,
                "tools/train_policy_layer.py",
                "--input",
                str(labeled_npz),
                "--base-model",
                str(final_model),
                "--base-meta",
                str(meta_copy),
                "--output-dir",
                str(seed_out / "policy_layer"),
                "--variant",
                "xgb",
                "--seed",
                str(int(seed)),
                "--nested-folds",
                "5",
            ],
            cwd=str(ROOT),
            check=True,
        )

        result = subprocess.run(
            [
                sys.executable,
                "tools/eval_ensemble_xgb_dedupe.py",
                "--model",
                str(final_model),
                "--meta",
                str(meta_copy),
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
                "--analysis-json",
                str(analysis_json),
            ],
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        (seed_out / "eval.json").write_text(result.stdout.strip() + "\n", encoding="utf-8")
        metrics = _extract_eval_metrics(payload)
        rows.append(
            {
                "seed": seed,
                "metrics": metrics,
                "baseline_metrics": baseline_metrics,
                "compare_to_baseline": _compare(metrics, baseline_metrics),
                "eval_json": str((seed_out / "eval.json").resolve()),
                "analysis_json": str(analysis_json.resolve()),
            }
        )

    summary = {
        "run_root": str(run_root),
        "refined_tag": args.refined_tag,
        "rows": rows,
        "metrics_mean": {
            f"mean_{field}": mean(_safe_float(row["metrics"].get(field), 0.0) for row in rows)
            for field in rows[0]["metrics"].keys()
        }
        if rows
        else {},
        "baseline_mean": {
            f"mean_{field}": mean(_safe_float(row["baseline_metrics"].get(field), 0.0) for row in rows)
            for field in rows[0]["baseline_metrics"].keys()
        }
        if rows
        else {},
        "compare_mean": {
            f"mean_{field}": mean(_safe_float(row["compare_to_baseline"].get(field), 0.0) for row in rows)
            for field in rows[0]["compare_to_baseline"].keys()
        }
        if rows
        else {},
    }
    _write_json(output_dir / "results_summary.json", summary)
    subprocess.run(
        [
            sys.executable,
            "tools/build_calibration_report_bundle.py",
            "--results-summary-json",
            str(output_dir / "results_summary.json"),
            "--analysis-json-glob",
            str(output_dir / "seed_*" / "analysis.json"),
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
