#!/usr/bin/env python3
"""Chain the postrun similarity-quality and learned-policy follow-up automatically."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


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


def _wait_for(path: Path, *, poll_seconds: float) -> None:
    while not path.exists():
        time.sleep(max(1.0, float(poll_seconds)))


def _wait_for_all(paths: Iterable[Path], *, poll_seconds: float) -> None:
    pending = [path for path in paths if not path.exists()]
    while pending:
        time.sleep(max(1.0, float(poll_seconds)))
        pending = [path for path in paths if not path.exists()]


def _choose_similarity_tag(results_ranked: Dict[str, Any]) -> Optional[str]:
    rows = results_ranked.get("pilot") if isinstance(results_ranked.get("pilot"), list) else []
    for row in rows:
        if _safe_float((row or {}).get("mean_delta_f1"), 0.0) > 0.0:
            tag = str((row or {}).get("tag") or "").strip()
            if tag:
                return tag
    return None


def _compare_metrics(metrics: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for field in ("tp", "fp", "fn", "precision", "recall", "f1", "coverage_preservation"):
        out[f"delta_{field}"] = _safe_float(metrics.get(field), 0.0) - _safe_float(baseline.get(field), 0.0)
    return out


def _mean_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    fields = {
        key
        for row in rows
        for key in row.keys()
        if isinstance(row.get(key), (int, float))
    }
    return {f"mean_{field}": mean(_safe_float(row.get(field), 0.0) for row in rows) for field in sorted(fields)}


def _run_eval(
    *,
    model_path: Path,
    meta_path: Path,
    labeled_npz: Path,
    prepass_jsonl: Path,
    output_json: Path,
) -> Dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            "tools/eval_ensemble_xgb_dedupe.py",
            "--model",
            str(model_path),
            "--meta",
            str(meta_path),
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
            "--use-val-split",
        ],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    output_json.write_text(result.stdout.strip() + "\n", encoding="utf-8")
    return payload


def _evaluate_existing_bakeoff(
    *,
    run_root: Path,
    refined_tag: str,
    output_dir: Path,
    seeds: List[str],
) -> Dict[str, Any]:
    labeled_npz = run_root / "views" / "window_intersection.labeled.npz"
    prepass_jsonl = run_root / "views" / "window_intersection.jsonl"
    rows: List[Dict[str, Any]] = []
    selected_counts = Counter()
    for seed in seeds:
        seed_dir = run_root / "postrun_policy_layer_bakeoff" / "bakeoff" / f"seed_{seed}"
        selection_path = seed_dir / "policy_layer" / "policy_layer_selection.json"
        _wait_for(selection_path, poll_seconds=10)
        selection = _load_json(selection_path)
        selected = str(selection.get("selected_variant") or "").strip().lower()
        selected_counts[selected] += 1
        meta_copy = sorted(seed_dir.glob("*.meta.json"))[0]
        meta = _load_json(meta_copy)
        model_path = Path(str(meta.get("model_path") or "")).resolve()
        eval_json = output_dir / "learned_policy_only" / f"seed_{seed}" / "eval.json"
        eval_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _run_eval(
            model_path=model_path,
            meta_path=meta_copy,
            labeled_npz=labeled_npz,
            prepass_jsonl=prepass_jsonl,
            output_json=eval_json,
        )
        metrics = _extract_eval_metrics(payload)
        baseline_eval = _load_json(
            run_root
            / "postrun_sam_bias_magnitude_sweep"
            / "pilot"
            / refined_tag
            / "intersection"
            / f"seed_{seed}"
            / "eval.json"
        )
        baseline_metrics = _extract_eval_metrics(baseline_eval)
        rows.append(
            {
                "seed": seed,
                "selected_variant": selected,
                "metrics": metrics,
                "baseline_metrics": baseline_metrics,
                "compare_to_baseline": _compare_metrics(metrics, baseline_metrics),
            }
        )
    summary = {
        "selected_variant_counts": dict(selected_counts),
        "rows": rows,
        "metrics_mean": _mean_rows([row["metrics"] for row in rows]),
        "baseline_mean": _mean_rows([row["baseline_metrics"] for row in rows]),
        "compare_mean": _mean_rows([row["compare_to_baseline"] for row in rows]),
    }
    _write_json(output_dir / "learned_policy_only" / "summary.json", summary)
    return summary


def _run_similarity_plus_policy(
    *,
    run_root: Path,
    refined_tag: str,
    similarity_tag: str,
    output_dir: Path,
    seeds: List[str],
) -> Dict[str, Any]:
    labeled_npz = run_root / "views" / "window_intersection.labeled.npz"
    prepass_jsonl = run_root / "views" / "window_intersection.jsonl"
    rows: List[Dict[str, Any]] = []
    selected_counts = Counter()
    for seed in seeds:
        base_seed_dir = (
            run_root
            / "postrun_similarity_quality_ablation"
            / "pilot"
            / similarity_tag
            / "intersection"
            / f"seed_{seed}"
        )
        meta_source = sorted(base_seed_dir.glob("*.meta.json"))[0]
        seed_out = output_dir / "similarity_plus_policy" / similarity_tag / f"seed_{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        meta_copy = seed_out / meta_source.name
        shutil.copy2(meta_source, meta_copy)
        meta = _load_json(meta_copy)
        model_path = Path(str(meta.get("model_path") or "")).resolve()
        subprocess.run(
            [
                sys.executable,
                "tools/train_policy_layer.py",
                "--input",
                str(labeled_npz),
                "--base-model",
                str(model_path),
                "--base-meta",
                str(meta_copy),
                "--output-dir",
                str(seed_out / "policy_layer"),
                "--variant",
                "bakeoff",
                "--seed",
                str(int(seed)),
                "--nested-folds",
                "5",
            ],
            cwd=str(ROOT),
            check=True,
        )
        selection = _load_json(seed_out / "policy_layer" / "policy_layer_selection.json")
        selected = str(selection.get("selected_variant") or "").strip().lower()
        selected_counts[selected] += 1
        eval_json = seed_out / "policy_layer" / "eval_selected.json"
        payload = _run_eval(
            model_path=model_path,
            meta_path=meta_copy,
            labeled_npz=labeled_npz,
            prepass_jsonl=prepass_jsonl,
            output_json=eval_json,
        )
        metrics = _extract_eval_metrics(payload)
        refined_eval = _load_json(
            run_root
            / "postrun_sam_bias_magnitude_sweep"
            / "pilot"
            / refined_tag
            / "intersection"
            / f"seed_{seed}"
            / "eval.json"
        )
        refined_metrics = _extract_eval_metrics(refined_eval)
        similarity_eval = _load_json(base_seed_dir / "eval.json")
        similarity_metrics = _extract_eval_metrics(similarity_eval)
        rows.append(
            {
                "seed": seed,
                "selected_variant": selected,
                "metrics": metrics,
                "refined_baseline_metrics": refined_metrics,
                "similarity_only_metrics": similarity_metrics,
                "compare_to_refined_baseline": _compare_metrics(metrics, refined_metrics),
                "compare_to_similarity_only": _compare_metrics(metrics, similarity_metrics),
            }
        )
    summary = {
        "tag": similarity_tag,
        "selected_variant_counts": dict(selected_counts),
        "rows": rows,
        "metrics_mean": _mean_rows([row["metrics"] for row in rows]),
        "refined_baseline_mean": _mean_rows([row["refined_baseline_metrics"] for row in rows]),
        "similarity_only_mean": _mean_rows([row["similarity_only_metrics"] for row in rows]),
        "compare_to_refined_baseline_mean": _mean_rows([row["compare_to_refined_baseline"] for row in rows]),
        "compare_to_similarity_only_mean": _mean_rows([row["compare_to_similarity_only"] for row in rows]),
    }
    _write_json(output_dir / "similarity_plus_policy" / similarity_tag / "summary.json", summary)
    return summary


def _render_report(summary: Dict[str, Any]) -> str:
    lines = ["# Postrun Similarity + Policy Follow-up", ""]
    lines.append(f"- Run root: `{summary['run_root']}`")
    lines.append(f"- Refined hand-policy tag: `{summary['refined_tag']}`")
    lines.append(f"- Selected similarity-quality tag: `{summary.get('similarity_tag') or 'none'}`")
    lines.append("")
    learned = summary.get("learned_policy_only") or {}
    if learned:
        lines.append("## Learned Policy Only")
        lines.append("")
        lines.append(f"- Selected family counts: `{learned.get('selected_variant_counts', {})}`")
        metrics = learned.get("metrics_mean") or {}
        compare = learned.get("compare_mean") or {}
        lines.append(
            f"- Mean: P={metrics.get('mean_precision', 0.0):.4f} "
            f"R={metrics.get('mean_recall', 0.0):.4f} F1={metrics.get('mean_f1', 0.0):.4f}"
        )
        lines.append(
            f"- Delta vs refined hand baseline: dF1={compare.get('mean_delta_f1', 0.0):+.4f} "
            f"dTP={compare.get('mean_delta_tp', 0.0):+.1f} "
            f"dFP={compare.get('mean_delta_fp', 0.0):+.1f}"
        )
        lines.append("")
    similarity = summary.get("similarity_plus_policy") or {}
    if similarity:
        lines.append("## Similarity Quality + Learned Policy")
        lines.append("")
        lines.append(f"- Selected family counts: `{similarity.get('selected_variant_counts', {})}`")
        metrics = similarity.get("metrics_mean") or {}
        compare_ref = similarity.get("compare_to_refined_baseline_mean") or {}
        compare_sim = similarity.get("compare_to_similarity_only_mean") or {}
        lines.append(
            f"- Mean: P={metrics.get('mean_precision', 0.0):.4f} "
            f"R={metrics.get('mean_recall', 0.0):.4f} F1={metrics.get('mean_f1', 0.0):.4f}"
        )
        lines.append(
            f"- Delta vs refined hand baseline: dF1={compare_ref.get('mean_delta_f1', 0.0):+.4f} "
            f"dTP={compare_ref.get('mean_delta_tp', 0.0):+.1f} "
            f"dFP={compare_ref.get('mean_delta_fp', 0.0):+.1f}"
        )
        lines.append(
            f"- Delta vs similarity-only: dF1={compare_sim.get('mean_delta_f1', 0.0):+.4f} "
            f"dTP={compare_sim.get('mean_delta_tp', 0.0):+.1f} "
            f"dFP={compare_sim.get('mean_delta_fp', 0.0):+.1f}"
        )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for current tests, then run similarity+policy follow-up.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--seeds", default="42,1337,2025")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (run_root / "postrun_similarity_policy_followup").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [chunk.strip() for chunk in str(args.seeds).split(",") if chunk.strip()]

    # Wait only for the bakeoff selections we actually need, not the entire forced-mode runner tail.
    bakeoff_selection_paths = [
        run_root / "postrun_policy_layer_bakeoff" / "bakeoff" / f"seed_{seed}" / "policy_layer" / "policy_layer_selection.json"
        for seed in seeds
    ]
    similarity_ranked = run_root / "postrun_similarity_quality_ablation" / "results_ranked.json"
    _wait_for_all(bakeoff_selection_paths + [similarity_ranked], poll_seconds=float(args.poll_seconds))

    similarity_results = _load_json(similarity_ranked)
    similarity_tag = _choose_similarity_tag(similarity_results)
    refined_tag = str(similarity_results.get("refined_tag") or "").strip()
    if not refined_tag:
        refined_tag = str(
            (
                ((similarity_results.get("pilot") or [{}])[0] or {}).get("tag")
                if similarity_results.get("pilot")
                else ""
            )
        ).strip()
    # True refined baseline tag comes from the magnitude sweep winner, not the similarity tag.
    if refined_tag == similarity_tag or not refined_tag:
        ranked = _load_json(run_root / "postrun_sam_bias_magnitude_sweep" / "results_ranked.json")
        full_rows = ranked.get("full") if isinstance(ranked.get("full"), list) else []
        pilot_rows = ranked.get("pilot") if isinstance(ranked.get("pilot"), list) else []
        source_rows = full_rows or pilot_rows
        refined_tag = str((source_rows[0] or {}).get("tag") or "").strip()

    learned_summary = _evaluate_existing_bakeoff(
        run_root=run_root,
        refined_tag=refined_tag,
        output_dir=output_dir,
        seeds=seeds,
    )

    similarity_plus_policy_summary: Optional[Dict[str, Any]] = None
    if similarity_tag:
        similarity_plus_policy_summary = _run_similarity_plus_policy(
            run_root=run_root,
            refined_tag=refined_tag,
            similarity_tag=similarity_tag,
            output_dir=output_dir,
            seeds=seeds,
        )

    summary = {
        "run_root": str(run_root),
        "refined_tag": refined_tag,
        "similarity_tag": similarity_tag,
        "learned_policy_only": learned_summary,
        "similarity_plus_policy": similarity_plus_policy_summary,
    }
    _write_json(output_dir / "results_summary.json", summary)
    (output_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
