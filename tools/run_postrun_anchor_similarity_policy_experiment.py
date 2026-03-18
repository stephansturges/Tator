#!/usr/bin/env python3
"""Queue a winner-only anchor-similarity learned-XGB experiment behind the active postrun tests."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

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


def _wait_for_process_pattern(pattern: str, *, poll_seconds: int) -> None:
    raw = str(pattern or "").strip()
    if not raw:
        return
    self_pid = os.getpid()
    while True:
        result = subprocess.run(["pgrep", "-f", raw], cwd=str(ROOT), capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return
        pids = {int(line.strip()) for line in str(result.stdout or "").splitlines() if line.strip().isdigit()}
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


def _resolve_best_similarity(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ranked = summary.get("ranked") if isinstance(summary.get("ranked"), list) else []
    if not ranked:
        return None
    best = ranked[0]
    if _safe_float(best.get("mean_delta_f1"), 0.0) <= 0.0:
        return None
    return best


def _resolve_base_source(run_root: Path, *, refined_tag: str) -> Dict[str, Any]:
    refined_rows = []
    for seed in ("42", "1337", "2025"):
        eval_path = run_root / "postrun_sam_bias_magnitude_sweep" / "full" / refined_tag / "full" / f"seed_{seed}" / "eval.json"
        refined_rows.append(_extract_eval_metrics(_load_json(eval_path)))
    refined_mean_f1 = mean(row["f1"] for row in refined_rows)

    sim_summary_path = run_root / "postrun_similarity_quality_full_window_eval" / "results_summary.json"
    best_similarity = _resolve_best_similarity(_load_json(sim_summary_path)) if sim_summary_path.exists() else None
    if best_similarity and _safe_float(best_similarity.get("mean_f1"), 0.0) > refined_mean_f1:
        return {
            "name": "similarity_quality",
            "tag": str(best_similarity["tag"]),
            "mean_f1": _safe_float(best_similarity.get("mean_f1"), 0.0),
        }
    return {"name": "refined_hand", "tag": refined_tag, "mean_f1": refined_mean_f1}


def _resolve_baseline_to_beat(run_root: Path, *, refined_tag: str) -> Dict[str, Any]:
    refined_rows = {}
    for seed in ("42", "1337", "2025"):
        eval_path = run_root / "postrun_sam_bias_magnitude_sweep" / "full" / refined_tag / "full" / f"seed_{seed}" / "eval.json"
        refined_rows[seed] = _extract_eval_metrics(_load_json(eval_path))
    candidates = [
        {
            "name": "refined_hand",
            "mean_f1": mean(row["f1"] for row in refined_rows.values()),
            "per_seed": refined_rows,
        }
    ]

    learned_path = run_root / "postrun_learned_xgb_full_window_eval" / "results_summary.json"
    if learned_path.exists():
        learned = _load_json(learned_path)
        learned_rows = {str(row["seed"]): row["metrics"] for row in learned.get("rows", [])}
        candidates.append(
            {
                "name": "learned_xgb",
                "mean_f1": _safe_float(learned.get("metrics_mean", {}).get("mean_f1"), 0.0),
                "per_seed": learned_rows,
            }
        )

    sim_path = run_root / "postrun_similarity_quality_full_window_eval" / "results_summary.json"
    if sim_path.exists():
        sim = _load_json(sim_path)
        best = _resolve_best_similarity(sim)
        if best:
            tag = str(best["tag"])
            sim_rows = {}
            for row in sim.get("rows", []):
                if str(row.get("tag")) == tag:
                    sim_rows[str(row["seed"])] = row["metrics"]
            candidates.append(
                {
                    "name": f"similarity_quality::{tag}",
                    "mean_f1": _safe_float(best.get("mean_f1"), 0.0),
                    "per_seed": sim_rows,
                }
            )
    return max(candidates, key=lambda item: _safe_float(item.get("mean_f1"), 0.0))


def _variant_grid() -> List[Dict[str, Any]]:
    return [
        {"tag": "p0p90_k1", "min_base_prob": 0.90, "topk_same_label": 1, "topk_any": 8},
        {"tag": "p0p90_k4", "min_base_prob": 0.90, "topk_same_label": 4, "topk_any": 8},
        {"tag": "p0p95_k1", "min_base_prob": 0.95, "topk_same_label": 1, "topk_any": 8},
        {"tag": "p0p95_k4", "min_base_prob": 0.95, "topk_same_label": 4, "topk_any": 8},
    ]


def _base_paths(run_root: Path, *, base_source: Dict[str, Any], seed: str) -> Dict[str, Path]:
    if base_source["name"] == "similarity_quality":
        tag = str(base_source["tag"])
        model_prefix = run_root / "postrun_similarity_quality_full_window_eval" / tag / f"seed_{seed}" / "model"
        return {"model": model_prefix.with_suffix(".json"), "meta": model_prefix.with_suffix(".meta.json")}
    return {
        "model": run_root / "final_matrix" / "window" / "full" / f"seed_{seed}" / "model_7ce3c1b42a_9e14a28553.json",
        "meta": run_root / "postrun_sam_bias_magnitude_sweep" / "full" / str(base_source["tag"]) / "full" / f"seed_{seed}" / "model_7ce3c1b42a_9e14a28553.meta.json",
    }


def _run_eval(
    *,
    run_root: Path,
    output_root: Path,
    variant: Dict[str, Any],
    seed: str,
    base_source: Dict[str, Any],
    baseline_to_beat: Dict[str, Any],
) -> Dict[str, Any]:
    run_dir = output_root / variant["tag"] / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    base_paths = _base_paths(run_root, base_source=base_source, seed=seed)
    model_json = base_paths["model"]
    meta_src = base_paths["meta"]
    meta_copy = run_dir / meta_src.name
    if not meta_copy.exists():
        meta_payload = _load_json(meta_src)
        meta_payload.pop("selected_policy_layer", None)
        meta_payload.pop("policy_layer_summary", None)
        _write_json(meta_copy, meta_payload)
    else:
        meta_payload = _load_json(meta_copy)

    subprocess.run(
        [
            sys.executable,
            "tools/train_policy_layer.py",
            "--input",
            str(run_root / "lanes" / "window" / "labeled.npz"),
            "--base-model",
            str(model_json),
            "--base-meta",
            str(meta_copy),
            "--output-dir",
            str(run_dir / "policy_layer"),
            "--variant",
            "xgb",
            "--seed",
            str(int(seed)),
            "--nested-folds",
            "5",
            "--enable-anchor-similarity",
            "--anchor-min-base-prob",
            str(float(variant["min_base_prob"])),
            "--anchor-topk-same-label",
            str(int(variant["topk_same_label"])),
            "--anchor-topk-any",
            str(int(variant["topk_any"])),
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
            str(meta_copy),
            "--data",
            str(run_root / "lanes" / "window" / "labeled.npz"),
            "--dataset",
            "qwen_dataset",
            "--prepass-jsonl",
            str(run_root / "prepass" / "window.jsonl"),
            "--eval-iou",
            "0.5",
            "--eval-iou-grid",
            "0.5",
            "--dedupe-iou",
            "0.75",
            "--scoreless-iou",
            "0.0",
            "--analysis-json",
            str(run_dir / "analysis.json"),
        ],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    eval_payload = json.loads(result.stdout)
    (run_dir / "eval.json").write_text(result.stdout.strip() + "\n", encoding="utf-8")
    metrics = _extract_eval_metrics(eval_payload)
    baseline_metrics = baseline_to_beat["per_seed"][seed]
    return {
        "tag": variant["tag"],
        "seed": seed,
        "anchor_similarity": {
            "enabled": True,
            "min_base_prob": float(variant["min_base_prob"]),
            "topk_same_label": int(variant["topk_same_label"]),
            "topk_any": int(variant["topk_any"]),
            "require_detector_support": True,
        },
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "compare_to_baseline": _compare(metrics, baseline_metrics),
        "eval_json": str((run_dir / "eval.json").resolve()),
        "analysis_json": str((run_dir / "analysis.json").resolve()),
    }


def _promoted(pilot_rows: List[Dict[str, Any]], *, topk: int) -> List[str]:
    ranked = sorted(
        pilot_rows,
        key=lambda row: (
            _safe_float(row["compare_to_baseline"].get("delta_f1"), 0.0),
            -_safe_float(row["compare_to_baseline"].get("delta_fp"), 0.0),
        ),
        reverse=True,
    )
    out: List[str] = []
    for row in ranked:
        delta_f1 = _safe_float(row["compare_to_baseline"].get("delta_f1"), 0.0)
        delta_tp = _safe_float(row["compare_to_baseline"].get("delta_tp"), 0.0)
        delta_fp = _safe_float(row["compare_to_baseline"].get("delta_fp"), 0.0)
        if delta_f1 <= 0.0:
            continue
        if delta_fp >= 0.0:
            continue
        if abs(delta_tp) > max(20.0, abs(delta_fp) / 20.0):
            continue
        out.append(str(row["tag"]))
        if len(out) >= int(topk):
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--refined-tag", default="text_m1p4__sim_m1p2")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--promote-topk", type=int, default=2)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_root = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (run_root / "postrun_anchor_similarity_policy_experiment").resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    _wait_for_process_pattern(
        f"tools/run_postrun_learned_xgb_full_window_eval.py --run-root {args.run_root}",
        poll_seconds=int(args.poll_seconds),
    )
    _wait_for_process_pattern(
        f"tools/run_postrun_similarity_quality_full_window_eval.py --run-root {args.run_root}",
        poll_seconds=int(args.poll_seconds),
    )
    while not (run_root / "postrun_learned_xgb_full_window_eval" / "results_summary.json").exists():
        time.sleep(max(5, int(args.poll_seconds)))
    while not (run_root / "postrun_similarity_quality_full_window_eval" / "results_summary.json").exists():
        time.sleep(max(5, int(args.poll_seconds)))

    base_source = _resolve_base_source(run_root, refined_tag=str(args.refined_tag))
    baseline_to_beat = _resolve_baseline_to_beat(run_root, refined_tag=str(args.refined_tag))

    pilot_rows: List[Dict[str, Any]] = []
    for variant in _variant_grid():
        pilot_rows.append(
            _run_eval(
                run_root=run_root,
                output_root=output_root / "pilot",
                variant=variant,
                seed="42",
                base_source=base_source,
                baseline_to_beat=baseline_to_beat,
            )
        )

    promoted = _promoted(pilot_rows, topk=int(args.promote_topk))
    full_rows: List[Dict[str, Any]] = []
    if promoted:
        variants_by_tag = {row["tag"]: row["anchor_similarity"] for row in pilot_rows}
        for tag in promoted:
            cfg = variants_by_tag[tag]
            variant = {"tag": tag, **cfg}
            for seed in ("42", "1337", "2025"):
                full_rows.append(
                    _run_eval(
                        run_root=run_root,
                        output_root=output_root / "full",
                        variant=variant,
                        seed=seed,
                        base_source=base_source,
                        baseline_to_beat=baseline_to_beat,
                    )
                )

    summary = {
        "run_root": str(run_root),
        "base_source": base_source,
        "baseline_to_beat": {
            "name": baseline_to_beat["name"],
            "mean_f1": baseline_to_beat["mean_f1"],
        },
        "pilot_rows": pilot_rows,
        "promoted_tags": promoted,
        "full_rows": full_rows,
    }
    if full_rows:
        grouped = {}
        for row in full_rows:
            grouped.setdefault(row["tag"], []).append(row)
        ranked = []
        for tag, rows in grouped.items():
            ranked.append(
                {
                    "tag": tag,
                    "mean_f1": mean(_safe_float(row["metrics"].get("f1"), 0.0) for row in rows),
                    "mean_delta_f1": mean(_safe_float(row["compare_to_baseline"].get("delta_f1"), 0.0) for row in rows),
                    "mean_delta_tp": mean(_safe_float(row["compare_to_baseline"].get("delta_tp"), 0.0) for row in rows),
                    "mean_delta_fp": mean(_safe_float(row["compare_to_baseline"].get("delta_fp"), 0.0) for row in rows),
                }
            )
        ranked.sort(key=lambda row: row["mean_f1"], reverse=True)
        summary["ranked"] = ranked
    _write_json(output_root / "results_summary.json", summary)
    subprocess.run(
        [
            sys.executable,
            "tools/build_calibration_report_bundle.py",
            "--results-summary-json",
            str(output_root / "results_summary.json"),
            "--analysis-json-glob",
            str(output_root / "*" / "seed_*" / "analysis.json"),
            "--analysis-json-glob",
            str(output_root / "full" / "*" / "seed_*" / "analysis.json"),
            "--output-json",
            str(output_root / "report_bundle.json"),
            "--output-md",
            str(output_root / "report_bundle.md"),
        ],
        cwd=str(ROOT),
        check=True,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
