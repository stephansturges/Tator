#!/usr/bin/env python3
"""Run the final learned-policy trust analysis on the real detection surface."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
LEARNED_VARIANTS = (
    "learned_selected",
    "learned_base_thresholds",
    "learned_gate_only",
    "learned_full_hand",
)


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


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _resolve_single_path(parent: Path, pattern: str, *, label: str) -> Path:
    matches = sorted(parent.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {label} in {parent}, got {len(matches)}")
    return matches[0]


def _alpha_from_tag(tag: str) -> Optional[float]:
    raw = str(tag or "").strip()
    if not raw.startswith("a"):
        return None
    try:
        return float(raw[1:].replace("m", "-").replace("p", "."))
    except ValueError:
        return None


def _load_or_run_debug_summary(
    *,
    run_root: Path,
    seed: str,
    lane: str,
    refined_tag: str,
    dataset: str,
    output_root: Path,
    force: bool,
) -> Dict[str, Any]:
    out_dir = output_root / f"seed_{seed}"
    summary_path = out_dir / "results_summary.json"
    if summary_path.exists() and not force:
        return _load_json(summary_path)
    subprocess.run(
        [
            sys.executable,
            "tools/debug_policy_layer_surface_mismatch.py",
            "--run-root",
            str(run_root),
            "--seed",
            str(seed),
            "--lane",
            str(lane),
            "--dataset",
            str(dataset),
            "--refined-tag",
            str(refined_tag),
            "--output-dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return _load_json(summary_path)


def _ensure_learned_eval_outputs(
    *,
    run_root: Path,
    refined_tag: str,
    seeds: Sequence[str],
    force: bool,
) -> None:
    output_root = run_root / "postrun_learned_xgb_full_window_eval"
    required = [output_root / f"seed_{seed}" / "eval.json" for seed in seeds]
    if all(path.exists() for path in required) and not force:
        return
    subprocess.run(
        [
            sys.executable,
            "tools/run_postrun_learned_xgb_full_window_eval.py",
            "--run-root",
            str(run_root),
            "--refined-tag",
            str(refined_tag),
            "--seeds",
            ",".join(str(seed) for seed in seeds),
        ],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )


def _flatten_top_fp_sources(rows: Sequence[Dict[str, Any]], *, limit: int = 8) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in rows:
        for item in row.get("top_fp_sources") or []:
            label = str(item.get("label") or "").strip().lower()
            source = str(item.get("primary_source") or "").strip().lower()
            if not label or not source:
                continue
            bucket = buckets.setdefault((label, source), {"tp": 0.0, "fp": 0.0, "accepted": 0.0})
            bucket["tp"] += _safe_float(item.get("tp"), 0.0)
            bucket["fp"] += _safe_float(item.get("fp"), 0.0)
            bucket["accepted"] += _safe_float(item.get("accepted"), 0.0)
    out: List[Dict[str, Any]] = []
    for (label, source), payload in buckets.items():
        out.append(
            {
                "label": label,
                "primary_source": source,
                "tp": int(payload["tp"]),
                "fp": int(payload["fp"]),
                "accepted": int(payload["accepted"]),
            }
        )
    out.sort(key=lambda row: (-int(row["fp"]), str(row["label"]), str(row["primary_source"])))
    return out[:limit]


def _extract_debug_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seed = str(summary.get("seed") or "")
    thresholds = summary.get("thresholds") if isinstance(summary.get("thresholds"), dict) else {}
    base_default = _safe_float(thresholds.get("base_default"), 0.0)
    learned_default = _safe_float(thresholds.get("learned_default"), 0.0)
    variants = summary.get("variants") if isinstance(summary.get("variants"), dict) else {}
    for split_name, split_payload in variants.items():
        if not isinstance(split_payload, dict):
            continue
        baseline = split_payload.get("baseline_hand") if isinstance(split_payload.get("baseline_hand"), dict) else {}
        baseline_acc = baseline.get("acceptance_audit") if isinstance(baseline.get("acceptance_audit"), dict) else {}
        baseline_density = baseline.get("duplicate_density") if isinstance(baseline.get("duplicate_density"), dict) else {}
        baseline_sam_only = _safe_float((baseline_acc.get("accepted_by_subgroup") or {}).get("sam_only"), 0.0)
        baseline_p99 = _safe_float(
            ((baseline_density.get("accepted_per_image_stats") or {}).get("p99")),
            _safe_float(((baseline_density.get("accepted_per_image_stats") or {}).get("p95")), 0.0),
        )
        for variant_name, payload in split_payload.items():
            if not isinstance(payload, dict):
                continue
            metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
            compare = payload.get("compare_to_baseline_hand") if isinstance(payload.get("compare_to_baseline_hand"), dict) else {}
            acc = payload.get("acceptance_audit") if isinstance(payload.get("acceptance_audit"), dict) else {}
            density = payload.get("duplicate_density") if isinstance(payload.get("duplicate_density"), dict) else {}
            subgroup = acc.get("accepted_by_subgroup") if isinstance(acc.get("accepted_by_subgroup"), dict) else {}
            accepted_stats = density.get("accepted_per_image_stats") if isinstance(density.get("accepted_per_image_stats"), dict) else {}
            deduped_stats = density.get("deduped_per_image_stats") if isinstance(density.get("deduped_per_image_stats"), dict) else {}
            sam_only_count = _safe_float(subgroup.get("sam_only"), 0.0)
            accepted_p99 = _safe_float(accepted_stats.get("p99"), _safe_float(accepted_stats.get("p95"), 0.0))
            row = {
                "tag": variant_name,
                "seed": seed,
                "split": str(split_name),
                "metrics": {
                    key: _safe_float(metrics.get(key), 0.0)
                    for key in ("tp", "fp", "fn", "precision", "recall", "f1")
                },
                "baseline_metrics": {
                    key: _safe_float((baseline.get("metrics") or {}).get(key), 0.0)
                    for key in ("tp", "fp", "fn", "precision", "recall", "f1")
                },
                "compare_to_baseline": {
                    key: _safe_float(compare.get(key), 0.0)
                    for key in (
                        "delta_tp",
                        "delta_fp",
                        "delta_fn",
                        "delta_precision",
                        "delta_recall",
                        "delta_f1",
                    )
                },
                "acceptance": {
                    "accepted_total": int(_safe_float(acc.get("accepted_rows"), 0.0)),
                    "accepted_rate": _safe_float(acc.get("accepted_rate"), 0.0),
                    "sam_only_accepted": int(sam_only_count),
                    "sam3_similarity_primary_accepted": int(_safe_float(subgroup.get("sam3_similarity_primary"), 0.0)),
                    "sam3_text_primary_accepted": int(_safe_float(subgroup.get("sam3_text_primary"), 0.0)),
                    "detector_supported_accepted": int(_safe_float(subgroup.get("detector_supported"), 0.0)),
                },
                "duplicate_density": {
                    "accepted_mean": _safe_float(accepted_stats.get("mean"), 0.0),
                    "accepted_p95": _safe_float(accepted_stats.get("p95"), 0.0),
                    "accepted_p99": accepted_p99,
                    "accepted_max": _safe_float(accepted_stats.get("max"), 0.0),
                    "deduped_mean": _safe_float(deduped_stats.get("mean"), 0.0),
                    "deduped_p95": _safe_float(deduped_stats.get("p95"), 0.0),
                    "deduped_p99": _safe_float(deduped_stats.get("p99"), _safe_float(deduped_stats.get("p95"), 0.0)),
                    "deduped_max": _safe_float(deduped_stats.get("max"), 0.0),
                    "accepted_to_deduped_ratio": _safe_float(density.get("accepted_to_deduped_ratio"), 0.0),
                },
                "baseline_comparators": {
                    "baseline_sam_only_accepted": int(baseline_sam_only),
                    "sam_only_ratio_vs_baseline": (sam_only_count / baseline_sam_only) if baseline_sam_only > 0 else 0.0,
                    "baseline_accepted_p99": baseline_p99,
                    "accepted_p99_ratio_vs_baseline": (accepted_p99 / baseline_p99) if baseline_p99 > 0 else 0.0,
                },
                "thresholds": {
                    "base_default": base_default,
                    "learned_default": learned_default,
                },
                "top_fp_sources": payload.get("top_fp_sources") if isinstance(payload.get("top_fp_sources"), list) else [],
            }
            rows.append(row)
    return rows


def _load_similarity_summary(run_root: Path) -> Dict[str, Any]:
    path = run_root / "postrun_similarity_quality_full_window_eval" / "results_summary.json"
    return _load_json(path)


def _select_similarity_tag(summary: Dict[str, Any], requested: str) -> str:
    if requested and requested != "auto":
        return requested
    ranked = summary.get("ranked") if isinstance(summary.get("ranked"), list) else []
    if ranked:
        return str(ranked[0].get("tag") or "a0p5")
    return "a0p5"


def _extract_similarity_rows(summary: Dict[str, Any], *, tag: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in summary.get("rows") or []:
        if str(raw.get("tag") or "") != str(tag):
            continue
        rows.append(
            {
                "tag": "similarity_quality_best",
                "seed": str(raw.get("seed") or ""),
                "split": "full",
                "metrics": {
                    key: _safe_float((raw.get("metrics") or {}).get(key), 0.0)
                    for key in ("tp", "fp", "fn", "precision", "recall", "f1", "coverage_preservation")
                },
                "baseline_metrics": {
                    key: _safe_float((raw.get("baseline_metrics") or {}).get(key), 0.0)
                    for key in ("tp", "fp", "fn", "precision", "recall", "f1", "coverage_preservation")
                },
                "compare_to_baseline": {
                    key: _safe_float((raw.get("compare_to_baseline") or {}).get(key), 0.0)
                    for key in (
                        "delta_tp",
                        "delta_fp",
                        "delta_fn",
                        "delta_precision",
                        "delta_recall",
                        "delta_f1",
                        "delta_coverage_preservation",
                    )
                },
                "analysis_json": raw.get("analysis_json"),
                "alpha": _safe_float(raw.get("alpha"), _alpha_from_tag(tag) or 0.0),
            }
        )
    return rows


def _aggregate_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row.get("split") or "full")][str(row.get("tag") or "")].append(row)
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for split_name, variants in grouped.items():
        split_bucket: Dict[str, Dict[str, Any]] = {}
        for tag, tagged_rows in variants.items():
            metrics_fields = sorted({key for row in tagged_rows for key in (row.get("metrics") or {}).keys()})
            compare_fields = sorted({key for row in tagged_rows for key in (row.get("compare_to_baseline") or {}).keys()})
            agg_metrics = {f"mean_{field}": mean(_safe_float((row.get("metrics") or {}).get(field), 0.0) for row in tagged_rows) for field in metrics_fields}
            agg_compare = {f"mean_{field}": mean(_safe_float((row.get("compare_to_baseline") or {}).get(field), 0.0) for row in tagged_rows) for field in compare_fields}
            delta_f1_values = [_safe_float((row.get("compare_to_baseline") or {}).get("delta_f1"), 0.0) for row in tagged_rows]
            sam_only_values = [_safe_float((row.get("acceptance") or {}).get("sam_only_accepted"), 0.0) for row in tagged_rows if isinstance(row.get("acceptance"), dict)]
            sam_only_ratios = [_safe_float((row.get("baseline_comparators") or {}).get("sam_only_ratio_vs_baseline"), 0.0) for row in tagged_rows if isinstance(row.get("baseline_comparators"), dict)]
            p99_ratios = [_safe_float((row.get("baseline_comparators") or {}).get("accepted_p99_ratio_vs_baseline"), 0.0) for row in tagged_rows if isinstance(row.get("baseline_comparators"), dict)]
            split_bucket[tag] = {
                "tag": tag,
                "split": split_name,
                "seed_count": len(tagged_rows),
                "rows": tagged_rows,
                **agg_metrics,
                **agg_compare,
                "min_delta_f1": min(delta_f1_values) if delta_f1_values else 0.0,
                "max_delta_f1": max(delta_f1_values) if delta_f1_values else 0.0,
                "mean_sam_only_accepted": mean(sam_only_values) if sam_only_values else 0.0,
                "mean_sam_only_ratio_vs_baseline": mean(sam_only_ratios) if sam_only_ratios else 0.0,
                "mean_accepted_p99_ratio_vs_baseline": mean(p99_ratios) if p99_ratios else 0.0,
                "top_fp_sources": _flatten_top_fp_sources(tagged_rows),
            }
        out[split_name] = split_bucket
    return out


def _rank_split(aggregated_split: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = list(aggregated_split.values())
    ranked.sort(key=lambda row: (_safe_float(row.get("mean_f1"), row.get("mean_f1", 0.0) or row.get("mean_f1", 0.0)),), reverse=True)
    # mean_f1 is stored as mean_f1 already; keep explicit normalization
    ranked.sort(key=lambda row: _safe_float(row.get("mean_f1"), 0.0), reverse=True)
    return ranked


def _evaluate_learned_variant(candidate: Optional[Dict[str, Any]]) -> Tuple[str, List[str]]:
    if not candidate:
        return "rejected", ["candidate_missing"]
    reasons: List[str] = []
    mean_delta_f1 = _safe_float(candidate.get("mean_delta_f1"), 0.0)
    min_delta_f1 = _safe_float(candidate.get("min_delta_f1"), 0.0)
    sam_only_ratio = _safe_float(candidate.get("mean_sam_only_ratio_vs_baseline"), 0.0)
    p99_ratio = _safe_float(candidate.get("mean_accepted_p99_ratio_vs_baseline"), 0.0)
    if mean_delta_f1 < -0.002:
        reasons.append("mean_f1_below_trust_gate")
    if min_delta_f1 < -0.004:
        reasons.append("worst_seed_below_trust_gate")
    if sam_only_ratio > 1.10:
        reasons.append("sam_only_acceptance_flood")
    if p99_ratio > 1.25:
        reasons.append("per_image_density_flood")
    if reasons:
        return "rejected", reasons
    if mean_delta_f1 >= 0.0 and min_delta_f1 >= -0.002:
        return "promotable", []
    return "retain_for_future_under_hand_policy_only", []


def _evaluate_similarity(candidate: Optional[Dict[str, Any]]) -> Tuple[str, List[str]]:
    if not candidate:
        return "rejected", ["candidate_missing"]
    reasons: List[str] = []
    mean_delta_f1 = _safe_float(candidate.get("mean_delta_f1"), 0.0)
    rows = candidate.get("rows") if isinstance(candidate.get("rows"), list) else []
    non_negative = sum(
        1
        for row in rows
        if _safe_float((row.get("compare_to_baseline") or {}).get("delta_f1"), 0.0) >= 0.0
    )
    mean_delta_fp = _safe_float(candidate.get("mean_delta_fp"), 0.0)
    if mean_delta_f1 < 0.001:
        reasons.append("mean_f1_gain_too_small")
    if non_negative < 2:
        reasons.append("insufficient_seed_consistency")
    if mean_delta_fp > 0.0:
        reasons.append("fp_not_reduced")
    if reasons:
        return "rejected", reasons
    return "promoted", []


def _load_windowed_recipe(run_root: Path, refined_tag: str, similarity_status: str, similarity_tag: str) -> Dict[str, Any]:
    baseline_dir = run_root / "postrun_sam_bias_magnitude_sweep" / "full" / refined_tag / "full" / "seed_42"
    meta = _load_json(
        _resolve_single_path(
            baseline_dir,
            "model_*.meta.json",
            label="magnitude-sweep promoted meta",
        )
    )
    policy = _load_json(baseline_dir / "policy.json")
    xgb_params = meta.get("xgb_params") if isinstance(meta.get("xgb_params"), dict) else {}
    return {
        "validation_status": "validated_on_full_window_3_seed_detection_surface",
        "lane": "window",
        "base_model_family": "xgb",
        "xgb_hparams": {
            "max_depth": int(_safe_float(xgb_params.get("max_depth"), 8)),
            "n_estimators": int(_safe_float(meta.get("n_estimators"), 900)),
            "learning_rate": _safe_float(xgb_params.get("eta"), 0.01),
            "subsample": _safe_float(xgb_params.get("subsample"), 0.8),
            "colsample_bytree": _safe_float(xgb_params.get("colsample_bytree"), 0.8),
            "min_child_weight": _safe_float(xgb_params.get("min_child_weight"), 1.0),
            "gamma": _safe_float(xgb_params.get("gamma"), 0.0),
            "reg_lambda": _safe_float(xgb_params.get("lambda"), 1.0),
            "reg_alpha": _safe_float(xgb_params.get("alpha"), 0.0),
        },
        "scenario": {
            "split_head": bool(((meta.get("split_head") or {}).get("enabled")) if isinstance(meta.get("split_head"), dict) else False),
            "sam3_text_quality_enabled": True,
            "sam3_text_quality_alpha": 0.8,
            "sam3_similarity_quality_enabled": similarity_status == "promoted",
            "sam3_similarity_quality_alpha": _alpha_from_tag(similarity_tag) if similarity_status == "promoted" else None,
        },
        "policy": policy,
        "second_stage_learned_policy": {
            "enabled": False,
            "reason": "not_trusted_on_external_detection_surface",
        },
    }


def _load_nonwindowed_recipe(run_root: Path) -> Dict[str, Any]:
    base_dir = run_root / "final_matrix" / "nonwindow" / "full" / "seed_42"
    meta_paths = sorted(base_dir.glob("*.meta.json")) if base_dir.exists() else []
    policy_paths = sorted(base_dir.glob("policy_*.json")) if base_dir.exists() else []
    if not meta_paths or not policy_paths:
        # Window-only recipe builds do not materialize a nonwindow fallback.
        # Trust analysis should still emit its learned-policy verdict rather
        # than failing at the final reporting/canonicalization step.
        return {
            "validation_status": "not_available_for_window_only_run",
            "lane": "nonwindow",
            "base_model_family": "xgb",
            "xgb_hparams": {},
            "scenario": {
                "split_head": False,
                "sam3_text_quality_enabled": False,
                "sam3_text_quality_alpha": None,
                "sam3_similarity_quality_enabled": False,
                "sam3_similarity_quality_alpha": None,
            },
            "policy": {},
            "second_stage_learned_policy": {
                "enabled": False,
                "reason": "window_only_run_no_nonwindow_fallback",
            },
        }
    meta = _load_json(meta_paths[0])
    policy = _load_json(policy_paths[0])
    xgb_params = meta.get("xgb_params") if isinstance(meta.get("xgb_params"), dict) else {}
    return {
        "validation_status": "fallback_only_final_sweep_winner_not_retested_in_trust_analysis",
        "lane": "nonwindow",
        "base_model_family": "xgb",
        "xgb_hparams": {
            "max_depth": int(_safe_float(xgb_params.get("max_depth"), 12)),
            "n_estimators": int(_safe_float(meta.get("n_estimators"), 600)),
            "learning_rate": _safe_float(xgb_params.get("eta"), 0.05),
            "subsample": _safe_float(xgb_params.get("subsample"), 0.8),
            "colsample_bytree": _safe_float(xgb_params.get("colsample_bytree"), 0.8),
            "min_child_weight": _safe_float(xgb_params.get("min_child_weight"), 1.0),
            "gamma": _safe_float(xgb_params.get("gamma"), 0.0),
            "reg_lambda": _safe_float(xgb_params.get("lambda"), 1.0),
            "reg_alpha": _safe_float(xgb_params.get("alpha"), 0.0),
        },
        "scenario": {
            "split_head": bool(((meta.get("split_head") or {}).get("enabled")) if isinstance(meta.get("split_head"), dict) else False),
            "sam3_text_quality_enabled": True,
            "sam3_text_quality_alpha": 0.5,
            "sam3_similarity_quality_enabled": False,
            "sam3_similarity_quality_alpha": None,
        },
        "policy": policy,
        "second_stage_learned_policy": {
            "enabled": False,
            "reason": "not_retested_on_nonwindow_lane",
        },
    }


def _render_markdown(
    *,
    decision_summary: Dict[str, Any],
    ranked_full: Sequence[Dict[str, Any]],
    ranked_val: Sequence[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# Final Policy Trust Analysis")
    lines.append("")
    lines.append(f"- Learned scoring trust status: `{decision_summary.get('learned_scoring_trust_status')}`")
    lines.append(f"- Best learned variant: `{decision_summary.get('best_learned_variant')}`")
    lines.append(f"- Similarity-quality status: `{decision_summary.get('similarity_quality_status')}`")
    lines.append(f"- Canonical windowed lane: `{decision_summary.get('canonical_windowed_recipe', {}).get('lane')}`")
    lines.append("")
    lines.append("## Full-Surface Ranking")
    lines.append("")
    lines.append("| Variant | Mean F1 | ΔF1 vs baseline | Mean P | Mean R | Mean sam_only ratio | Mean p99 ratio |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in ranked_full:
        lines.append(
            "| {tag} | {f1:.4f} | {df1:+.4f} | {p:.4f} | {r:.4f} | {sam:.2f} | {p99:.2f} |".format(
                tag=str(row.get("tag")),
                f1=_safe_float(row.get("mean_f1"), 0.0),
                df1=_safe_float(row.get("mean_delta_f1"), 0.0),
                p=_safe_float(row.get("mean_precision"), 0.0),
                r=_safe_float(row.get("mean_recall"), 0.0),
                sam=_safe_float(row.get("mean_sam_only_ratio_vs_baseline"), 0.0),
                p99=_safe_float(row.get("mean_accepted_p99_ratio_vs_baseline"), 0.0),
            )
        )
    lines.append("")
    lines.append("## Outer-Val Ranking")
    lines.append("")
    lines.append("| Variant | Mean F1 | ΔF1 vs baseline | Mean P | Mean R |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in ranked_val:
        lines.append(
            "| {tag} | {f1:.4f} | {df1:+.4f} | {p:.4f} | {r:.4f} |".format(
                tag=str(row.get("tag")),
                f1=_safe_float(row.get("mean_f1"), 0.0),
                df1=_safe_float(row.get("mean_delta_f1"), 0.0),
                p=_safe_float(row.get("mean_precision"), 0.0),
                r=_safe_float(row.get("mean_recall"), 0.0),
            )
        )
    lines.append("")
    best_learned = decision_summary.get("best_learned_aggregate") if isinstance(decision_summary.get("best_learned_aggregate"), dict) else {}
    if best_learned:
        lines.append("## Learned Verdict")
        lines.append("")
        lines.append(f"- Candidate: `{best_learned.get('tag')}`")
        lines.append(f"- Full mean F1 delta: `{_safe_float(best_learned.get('mean_delta_f1'), 0.0):+.4f}`")
        lines.append(f"- Full worst-seed delta: `{_safe_float(best_learned.get('min_delta_f1'), 0.0):+.4f}`")
        lines.append(f"- Mean `sam_only` acceptance ratio vs baseline: `{_safe_float(best_learned.get('mean_sam_only_ratio_vs_baseline'), 0.0):.2f}`")
        lines.append(f"- Mean accepted-count p99 ratio vs baseline: `{_safe_float(best_learned.get('mean_accepted_p99_ratio_vs_baseline'), 0.0):.2f}`")
        reasons = decision_summary.get("learned_scoring_reasons") if isinstance(decision_summary.get("learned_scoring_reasons"), list) else []
        if reasons:
            lines.append(f"- Rejection reasons: `{', '.join(str(item) for item in reasons)}`")
        top_fp = best_learned.get("top_fp_sources") if isinstance(best_learned.get("top_fp_sources"), list) else []
        if top_fp:
            lines.append("")
            lines.append("Top FP-heavy label/source pairs:")
            for row in top_fp[:5]:
                lines.append(
                    f"- `{row.get('label')}/{row.get('primary_source')}`: fp=`{int(row.get('fp', 0))}`, tp=`{int(row.get('tp', 0))}`, accepted=`{int(row.get('accepted', 0))}`"
                )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _render_canonical_recipe_md(decision_summary: Dict[str, Any]) -> str:
    windowed = decision_summary.get("canonical_windowed_recipe") if isinstance(decision_summary.get("canonical_windowed_recipe"), dict) else {}
    nonwindowed = decision_summary.get("canonical_nonwindowed_recipe") if isinstance(decision_summary.get("canonical_nonwindowed_recipe"), dict) else {}
    lines = [
        "# Canonical Prepass Recipe",
        "",
        "## Windowed",
        "",
        f"- Lane: `{windowed.get('lane')}`",
        f"- Validation status: `{windowed.get('validation_status')}`",
        f"- Learned second stage enabled: `{bool(((windowed.get('second_stage_learned_policy') or {}).get('enabled')) if isinstance(windowed.get('second_stage_learned_policy'), dict) else False)}`",
        f"- SAM3 text quality alpha: `{(windowed.get('scenario') or {}).get('sam3_text_quality_alpha')}`",
        f"- SAM3 similarity quality alpha: `{(windowed.get('scenario') or {}).get('sam3_similarity_quality_alpha')}`",
        "",
        "## Non-Windowed Fallback",
        "",
        f"- Lane: `{nonwindowed.get('lane')}`",
        f"- Validation status: `{nonwindowed.get('validation_status')}`",
        f"- Learned second stage enabled: `{bool(((nonwindowed.get('second_stage_learned_policy') or {}).get('enabled')) if isinstance(nonwindowed.get('second_stage_learned_policy'), dict) else False)}`",
        "",
    ]
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--lane", default="window")
    parser.add_argument("--dataset", default="qwen_dataset")
    parser.add_argument("--refined-tag", default="text_m1p4__sim_m1p2")
    parser.add_argument("--seeds", default="42,1337,2025")
    parser.add_argument("--similarity-tag", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (run_root / "final_policy_trust_analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_root = output_dir / "debug_policy_layer_surface_mismatch"
    seeds = [chunk.strip() for chunk in str(args.seeds).split(",") if chunk.strip()]
    debug_summaries_exist = all(
        (debug_root / f"seed_{seed}" / "results_summary.json").exists()
        for seed in seeds
    )
    if bool(args.force) or not debug_summaries_exist:
        _ensure_learned_eval_outputs(
            run_root=run_root,
            refined_tag=str(args.refined_tag),
            seeds=seeds,
            force=bool(args.force),
        )

    raw_rows: List[Dict[str, Any]] = []
    raw_debug: Dict[str, Any] = {}
    for seed in seeds:
        summary = _load_or_run_debug_summary(
            run_root=run_root,
            seed=seed,
            lane=str(args.lane),
            refined_tag=str(args.refined_tag),
            dataset=str(args.dataset),
            output_root=debug_root,
            force=bool(args.force),
        )
        raw_debug[str(seed)] = {
            "summary_path": str((debug_root / f"seed_{seed}" / "results_summary.json").resolve()),
            "thresholds": summary.get("thresholds"),
        }
        raw_rows.extend(_extract_debug_rows(summary))

    similarity_summary = _load_similarity_summary(run_root)
    similarity_tag = _select_similarity_tag(similarity_summary, str(args.similarity_tag))
    similarity_rows = _extract_similarity_rows(similarity_summary, tag=similarity_tag)
    raw_rows.extend(similarity_rows)

    aggregated = _aggregate_rows(raw_rows)
    ranked_full = _rank_split(aggregated.get("full", {}))
    ranked_val = _rank_split(aggregated.get("val", {}))

    best_learned = None
    for name in ("learned_full_hand", "learned_gate_only", "learned_base_thresholds", "learned_selected"):
        candidate = aggregated.get("full", {}).get(name)
        if candidate is None:
            continue
        if best_learned is None or _safe_float(candidate.get("mean_f1"), 0.0) > _safe_float(best_learned.get("mean_f1"), 0.0):
            best_learned = candidate
    learned_status, learned_reasons = _evaluate_learned_variant(best_learned)
    similarity_status, similarity_reasons = _evaluate_similarity(aggregated.get("full", {}).get("similarity_quality_best"))
    recommended_second_stage_policy = {
        "enabled": learned_status == "promotable",
        "variant": "xgb" if learned_status == "promotable" else "none",
        "runtime_mode": best_learned.get("tag") if isinstance(best_learned, dict) and learned_status == "promotable" else None,
        "reason": (
            "trusted_on_external_detection_surface"
            if learned_status == "promotable"
            else "not_trusted_on_external_detection_surface"
        ),
    }

    canonical_windowed = _load_windowed_recipe(run_root, str(args.refined_tag), similarity_status, similarity_tag)
    canonical_nonwindowed = _load_nonwindowed_recipe(run_root)

    decision_summary = {
        "run_root": str(run_root),
        "lane": str(args.lane),
        "dataset": str(args.dataset),
        "refined_tag": str(args.refined_tag),
        "similarity_tag_evaluated": similarity_tag,
        "learned_scoring_trust_status": learned_status,
        "learned_scoring_reasons": learned_reasons,
        "best_learned_variant": best_learned.get("tag") if isinstance(best_learned, dict) else None,
        "best_learned_aggregate": best_learned,
        "recommended_second_stage_policy": recommended_second_stage_policy,
        "similarity_quality_status": similarity_status,
        "similarity_quality_reasons": similarity_reasons,
        "canonical_windowed_recipe": canonical_windowed,
        "canonical_nonwindowed_recipe": canonical_nonwindowed,
        "recommended_canonical_stack": {
            "windowed": "refined_hand_policy_plus_similarity_quality" if similarity_status == "promoted" else "refined_hand_policy",
            "nonwindowed": "final_sweep_nonwindow_fallback",
        },
    }

    full_rows_for_bundle = [row for row in raw_rows if str(row.get("split") or "") == "full"]
    results_summary = {
        "run_root": str(run_root),
        "baseline_to_beat": {"name": "baseline_hand"},
        "rows": full_rows_for_bundle,
        "ranked": ranked_full,
        "decision_summary_path": str((output_dir / "decision_summary.json").resolve()),
    }
    results_ranked = {
        "full": ranked_full,
        "val": ranked_val,
    }
    results_raw = {
        "debug_seed_summaries": raw_debug,
        "rows": raw_rows,
        "aggregated": aggregated,
        "similarity_summary_path": str((run_root / "postrun_similarity_quality_full_window_eval" / "results_summary.json").resolve()),
    }

    _write_json(output_dir / "results_raw.json", results_raw)
    _write_json(output_dir / "results_summary.json", results_summary)
    _write_json(output_dir / "results_ranked.json", results_ranked)
    _write_json(output_dir / "decision_summary.json", decision_summary)
    _write_text(output_dir / "report.md", _render_markdown(decision_summary=decision_summary, ranked_full=ranked_full, ranked_val=ranked_val))
    _write_text(output_dir / "canonical_recipe.md", _render_canonical_recipe_md(decision_summary))

    subprocess.run(
        [
            sys.executable,
            "tools/build_calibration_report_bundle.py",
            "--results-summary-json",
            str(output_dir / "results_summary.json"),
            "--output-json",
            str(output_dir / "report_bundle.json"),
            "--output-md",
            str(output_dir / "report_bundle.md"),
        ],
        cwd=str(ROOT),
        check=True,
    )

    print(json.dumps(decision_summary, indent=2))


if __name__ == "__main__":
    main()
