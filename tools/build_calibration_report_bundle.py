#!/usr/bin/env python3
"""Build a canonical calibration reporting bundle from eval and post-run artifacts."""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPORT_BUNDLE_VERSION = 1


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return parsed


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _mean(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    return float(sum(seq) / len(seq)) if seq else 0.0


def _std(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if len(seq) <= 1:
        return 0.0
    return float(statistics.pstdev(seq))


def _extract_overall_metrics(eval_payload: Dict[str, Any]) -> Dict[str, float]:
    metric_tiers = eval_payload.get("metric_tiers") if isinstance(eval_payload.get("metric_tiers"), dict) else {}
    post_xgb = metric_tiers.get("post_xgb") if isinstance(metric_tiers.get("post_xgb"), dict) else {}
    accepted_all = post_xgb.get("accepted_all") if isinstance(post_xgb.get("accepted_all"), dict) else {}
    coverage = eval_payload.get("coverage_upper_bound") if isinstance(eval_payload.get("coverage_upper_bound"), dict) else {}
    coverage_candidate = (
        coverage.get("candidate_all") if isinstance(coverage.get("candidate_all"), dict) else {}
    )
    reference = eval_payload.get("reference_iou") if isinstance(eval_payload.get("reference_iou"), dict) else {}
    reference_xgb = reference.get("xgb_ensemble") if isinstance(reference.get("xgb_ensemble"), dict) else {}
    union_metrics = (
        metric_tiers.get("post_cluster", {})
        .get("source_attributed", {})
        .get("yolo_rfdetr_union", {})
        if isinstance(metric_tiers.get("post_cluster"), dict)
        else {}
    )
    precision = _safe_float(accepted_all.get("precision"), _safe_float(reference_xgb.get("precision"), _safe_float(eval_payload.get("precision"), 0.0)))
    recall = _safe_float(accepted_all.get("recall"), _safe_float(reference_xgb.get("recall"), _safe_float(eval_payload.get("recall"), 0.0)))
    f1 = _safe_float(accepted_all.get("f1"), _safe_float(reference_xgb.get("f1"), _safe_float(eval_payload.get("f1"), 0.0)))
    union_f1 = _safe_float(union_metrics.get("f1"), 0.0)
    coverage_ub = _safe_float(coverage_candidate.get("recall_upper_bound"), 0.0)
    coverage_pres = (recall / coverage_ub) if coverage_ub > 0.0 else 0.0
    return {
        "tp": _safe_float(accepted_all.get("tp"), _safe_float(reference_xgb.get("tp"), _safe_float(eval_payload.get("tp"), 0.0))),
        "fp": _safe_float(accepted_all.get("fp"), _safe_float(reference_xgb.get("fp"), _safe_float(eval_payload.get("fp"), 0.0))),
        "fn": _safe_float(accepted_all.get("fn"), _safe_float(reference_xgb.get("fn"), _safe_float(eval_payload.get("fn"), 0.0))),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "union_f1": union_f1,
        "delta_vs_union_f1": f1 - union_f1,
        "coverage_upper_bound": coverage_ub,
        "coverage_preservation": coverage_pres,
    }


def _compute_prf(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def _flatten_analysis_globs(patterns: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()
    for raw in patterns:
        pattern = str(raw or "").strip()
        if not pattern:
            continue
        for match in glob.glob(pattern, recursive=True):
            resolved = str(Path(match).resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(Path(resolved))
    out.sort()
    return out


def _aggregate_per_class(analysis_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[str, Dict[str, float]] = {}
    for payload in analysis_payloads:
        for row in payload.get("per_class") or []:
            label = str(row.get("label") or "").strip().lower()
            if not label:
                continue
            bucket = counts.setdefault(
                label,
                {"tp": 0.0, "fp": 0.0, "fn": 0.0, "support_gt": 0.0, "support_pred": 0.0},
            )
            for field in ("tp", "fp", "fn", "support_gt", "support_pred"):
                bucket[field] += _safe_float(row.get(field), 0.0)
    out: List[Dict[str, Any]] = []
    for label, row in sorted(counts.items()):
        precision, recall, f1 = _compute_prf(row["tp"], row["fp"], row["fn"])
        out.append(
            {
                "label": label,
                "tp": int(row["tp"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "support_gt": int(row["support_gt"]),
                "support_pred": int(row["support_pred"]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    out.sort(key=lambda row: (-int(row["fp"]), str(row["label"])))
    return out


def _aggregate_per_class_per_source(analysis_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[Tuple[str, str], Dict[str, float]] = {}
    for payload in analysis_payloads:
        for row in payload.get("per_class_per_source") or []:
            label = str(row.get("label") or "").strip().lower()
            source = str(row.get("primary_source") or row.get("source") or "").strip().lower()
            if not label or not source:
                continue
            bucket = counts.setdefault((label, source), {"tp": 0.0, "fp": 0.0, "accepted": 0.0})
            bucket["tp"] += _safe_float(row.get("tp"), 0.0)
            bucket["fp"] += _safe_float(row.get("fp"), 0.0)
            bucket["accepted"] += _safe_float(row.get("accepted"), _safe_float(row.get("support_pred"), 0.0))
    out: List[Dict[str, Any]] = []
    for (label, source), row in sorted(counts.items()):
        precision = row["tp"] / (row["tp"] + row["fp"]) if (row["tp"] + row["fp"]) > 0 else 0.0
        out.append(
            {
                "label": label,
                "primary_source": source,
                "tp": int(row["tp"]),
                "fp": int(row["fp"]),
                "accepted": int(row["accepted"]),
                "precision": float(precision),
            }
        )
    out.sort(key=lambda row: (-int(row["fp"]), str(row["label"]), str(row["primary_source"])))
    return out


def _aggregate_boundary_hits(analysis_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg: Dict[str, Any] = {"decision_rows": 0, "hard_gate_rejections": {}, "buckets": {}}
    for payload in analysis_payloads:
        boundary = payload.get("boundary_hits") if isinstance(payload.get("boundary_hits"), dict) else {}
        agg["decision_rows"] += int(boundary.get("decision_rows") or 0)
        hard_gate = boundary.get("hard_gate_rejections") if isinstance(boundary.get("hard_gate_rejections"), dict) else {}
        for key, value in hard_gate.items():
            agg["hard_gate_rejections"][str(key)] = int(agg["hard_gate_rejections"].get(str(key), 0)) + int(value or 0)
        buckets = boundary.get("buckets") if isinstance(boundary.get("buckets"), dict) else {}
        for bucket_name, raw_bucket in buckets.items():
            if not isinstance(raw_bucket, dict):
                continue
            bucket = agg["buckets"].setdefault(
                str(bucket_name),
                {
                    "accepted": 0,
                    "rejected": 0,
                    "positive_rows": 0,
                    "negative_rows": 0,
                    "by_label": {},
                    "by_source": {},
                },
            )
            for field in ("accepted", "rejected", "positive_rows", "negative_rows"):
                bucket[field] += int(raw_bucket.get(field) or 0)
            for group_name in ("by_label", "by_source"):
                raw_group = raw_bucket.get(group_name) if isinstance(raw_bucket.get(group_name), dict) else {}
                group = bucket[group_name]
                for key, payload_counts in raw_group.items():
                    if not isinstance(payload_counts, dict):
                        continue
                    slot = group.setdefault(str(key), {"accepted": 0, "rejected": 0, "positive_rows": 0, "negative_rows": 0})
                    for field in ("accepted", "rejected", "positive_rows", "negative_rows"):
                        slot[field] += int(payload_counts.get(field) or 0)
    return agg


def _aggregate_calibration_diagnostics(analysis_payloads: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    total_count = 0
    total_positive = 0.0
    brier_sum_sq = 0.0
    bins: Dict[int, Dict[str, float]] = {}
    for payload in analysis_payloads:
        diag = payload.get("calibration_diagnostics") if isinstance(payload.get("calibration_diagnostics"), dict) else {}
        if not diag:
            continue
        total_count += int(diag.get("candidate_count") or 0)
        total_positive += _safe_float(diag.get("positive_count"), 0.0)
        brier_sum_sq += _safe_float(diag.get("brier_sum_sq"), 0.0)
        for row in diag.get("bins") or []:
            try:
                idx = int(row.get("bin_index"))
            except (TypeError, ValueError):
                continue
            bucket = bins.setdefault(
                idx,
                {
                    "bin_index": idx,
                    "lower": _safe_float(row.get("lower"), idx / 10.0),
                    "upper": _safe_float(row.get("upper"), (idx + 1) / 10.0),
                    "count": 0.0,
                    "sum_prob": 0.0,
                    "sum_positive": 0.0,
                },
            )
            bucket["count"] += _safe_float(row.get("count"), 0.0)
            bucket["sum_prob"] += _safe_float(row.get("sum_prob"), 0.0)
            bucket["sum_positive"] += _safe_float(row.get("sum_positive"), 0.0)
    if total_count <= 0:
        return None
    out_bins: List[Dict[str, Any]] = []
    ece = 0.0
    for idx in sorted(bins):
        row = bins[idx]
        count = int(row["count"])
        mean_prob = (row["sum_prob"] / row["count"]) if row["count"] > 0 else 0.0
        positive_rate = (row["sum_positive"] / row["count"]) if row["count"] > 0 else 0.0
        ece += (abs(mean_prob - positive_rate) * row["count"]) / total_count
        out_bins.append(
            {
                "bin_index": idx,
                "lower": row["lower"],
                "upper": row["upper"],
                "count": count,
                "sum_prob": row["sum_prob"],
                "sum_positive": row["sum_positive"],
                "mean_prob": mean_prob,
                "positive_rate": positive_rate,
            }
        )
    return {
        "candidate_count": int(total_count),
        "positive_count": int(total_positive),
        "brier_sum_sq": float(brier_sum_sq),
        "brier_score": float(brier_sum_sq / total_count) if total_count > 0 else 0.0,
        "ece_10": float(ece),
        "bins": out_bins,
    }


def _build_seed_uncertainty_from_rows(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    by_metric: Dict[str, List[float]] = defaultdict(list)
    row_summaries: List[Dict[str, Any]] = []
    for row in rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        seed = row.get("seed") or row.get("row_id") or row.get("tag") or len(row_summaries)
        row_summaries.append(
            {
                "seed": seed,
                "precision": _safe_float(metrics.get("precision"), 0.0),
                "recall": _safe_float(metrics.get("recall"), 0.0),
                "f1": _safe_float(metrics.get("f1"), 0.0),
                "coverage_preservation": _safe_float(metrics.get("coverage_preservation"), 0.0),
            }
        )
        for field in ("precision", "recall", "f1", "coverage_preservation"):
            by_metric[field].append(_safe_float(metrics.get(field), 0.0))
    metrics_out = {}
    for field, values in by_metric.items():
        metrics_out[field] = {
            "mean": _mean(values),
            "std": _std(values),
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }
    return {"row_count": len(rows), "rows": row_summaries, "metrics": metrics_out}


def _extract_policy_layer_summary(meta_payload: Dict[str, Any], policy_selection: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not meta_payload and not policy_selection:
        return None
    selected = meta_payload.get("policy_layer_summary") if isinstance(meta_payload.get("policy_layer_summary"), dict) else {}
    selection = policy_selection if isinstance(policy_selection, dict) else {}
    return {
        "requested_variant": selection.get("requested_variant") or selected.get("requested_variant"),
        "selected_variant": selection.get("selected_variant") or selected.get("selected_variant"),
        "trained_variants": selection.get("trained_variants") if isinstance(selection.get("trained_variants"), list) else [],
        "baseline_f1": _safe_float(selected.get("baseline_f1"), 0.0),
        "selected_f1": _safe_float(selected.get("selected_f1"), 0.0),
        "delta_vs_baseline_f1": _safe_float(selected.get("delta_vs_baseline_f1"), 0.0),
        "candidates": selection.get("candidates") if isinstance(selection.get("candidates"), dict) else {},
    }


def _build_single_run_bundle(
    *,
    eval_payload: Dict[str, Any],
    analysis_payload: Optional[Dict[str, Any]],
    meta_payload: Optional[Dict[str, Any]],
    policy_selection: Optional[Dict[str, Any]],
    artifact_refs: Dict[str, Any],
    model_family: str,
) -> Dict[str, Any]:
    analysis_payload = analysis_payload or {}
    meta_payload = meta_payload or {}
    policy_selection = policy_selection or {}
    overall_metrics = _extract_overall_metrics(eval_payload)
    per_class = _aggregate_per_class([analysis_payload]) if analysis_payload else []
    per_class_per_source = _aggregate_per_class_per_source([analysis_payload]) if analysis_payload else []
    boundary_hits = _aggregate_boundary_hits([analysis_payload]) if analysis_payload else {}
    calibration_diag = _aggregate_calibration_diagnostics([analysis_payload]) if analysis_payload else None
    bundle = {
        "version": REPORT_BUNDLE_VERSION,
        "report_kind": "calibration_job",
        "generated_utc": meta_payload.get("generated_utc"),
        "source_root": artifact_refs.get("output_dir"),
        "model_family": str(model_family or "xgb"),
        "selection_summary": {
            "selected_policy_variant": (
                (policy_selection or {}).get("selected_variant")
                or ((meta_payload.get("policy_layer_summary") or {}).get("selected_variant") if isinstance(meta_payload.get("policy_layer_summary"), dict) else None)
            ),
        },
        "overall_metrics": overall_metrics,
        "policy_layer": _extract_policy_layer_summary(meta_payload, policy_selection),
        "per_class": per_class,
        "per_class_per_source": per_class_per_source,
        "boundary_hits": boundary_hits,
        "seed_uncertainty": {"row_count": 1, "rows": [{"seed": meta_payload.get("seed", "single"), **{k: overall_metrics[k] for k in ("precision", "recall", "f1", "coverage_preservation")}}], "metrics": {field: {"mean": overall_metrics[field], "std": 0.0, "min": overall_metrics[field], "max": overall_metrics[field]} for field in ("precision", "recall", "f1", "coverage_preservation")}},
        "calibration_diagnostics": calibration_diag,
        "artifact_refs": artifact_refs,
        "warnings": list(analysis_payload.get("warnings") or []),
    }
    return bundle


def _build_postrun_bundle(
    *,
    summary_payload: Dict[str, Any],
    analysis_payloads: List[Dict[str, Any]],
    artifact_refs: Dict[str, Any],
    model_family: str,
) -> Dict[str, Any]:
    ranked = summary_payload.get("ranked") if isinstance(summary_payload.get("ranked"), list) else []
    rows = summary_payload.get("rows") if isinstance(summary_payload.get("rows"), list) else []
    best_row = ranked[0] if ranked else {}
    overall_metrics = {
        "precision": _safe_float(best_row.get("mean_precision"), _safe_float(summary_payload.get("metrics_mean", {}).get("mean_precision"), 0.0)),
        "recall": _safe_float(best_row.get("mean_recall"), _safe_float(summary_payload.get("metrics_mean", {}).get("mean_recall"), 0.0)),
        "f1": _safe_float(best_row.get("mean_f1"), _safe_float(summary_payload.get("metrics_mean", {}).get("mean_f1"), 0.0)),
        "coverage_preservation": _safe_float(best_row.get("mean_coverage_preservation"), _safe_float(summary_payload.get("metrics_mean", {}).get("mean_coverage_preservation"), 0.0)),
    }
    compare_mean = summary_payload.get("compare_mean") if isinstance(summary_payload.get("compare_mean"), dict) else {}
    selection_summary = {
        "winner": best_row.get("tag") or best_row.get("lane") or "single_variant",
        "baseline_name": (
            ((summary_payload.get("baseline_to_beat") or {}).get("name") if isinstance(summary_payload.get("baseline_to_beat"), dict) else None)
            or "baseline"
        ),
        "delta_vs_baseline_f1": _safe_float(best_row.get("mean_delta_f1"), _safe_float(compare_mean.get("mean_delta_f1"), 0.0)),
        "comparison_rows": ranked if ranked else [],
    }
    bundle = {
        "version": REPORT_BUNDLE_VERSION,
        "report_kind": "postrun_comparison",
        "generated_utc": None,
        "source_root": summary_payload.get("run_root"),
        "model_family": str(model_family or "xgb"),
        "selection_summary": selection_summary,
        "overall_metrics": overall_metrics,
        "policy_layer": None,
        "per_class": _aggregate_per_class(analysis_payloads),
        "per_class_per_source": _aggregate_per_class_per_source(analysis_payloads),
        "boundary_hits": _aggregate_boundary_hits(analysis_payloads) if analysis_payloads else {},
        "seed_uncertainty": _build_seed_uncertainty_from_rows(rows) if rows else None,
        "calibration_diagnostics": _aggregate_calibration_diagnostics(analysis_payloads),
        "artifact_refs": artifact_refs,
        "warnings": [],
    }
    return bundle


def _build_final_sweep_bundle(
    *,
    ranked_payload: Dict[str, Any],
    analysis_payloads: List[Dict[str, Any]],
    artifact_refs: Dict[str, Any],
    model_family: str,
) -> Dict[str, Any]:
    winner = ranked_payload.get("winner") if isinstance(ranked_payload.get("winner"), dict) else {}
    views = ranked_payload.get("views") if isinstance(ranked_payload.get("views"), dict) else {}
    intersection_rows = views.get("intersection", {}).get("ranked_lanes", []) if isinstance(views.get("intersection"), dict) else []
    selection_summary = {
        "winner_lane": winner.get("lane"),
        "selection_view": winner.get("selection_view"),
        "guardrail_pass": bool(winner.get("guardrail_pass")),
        "comparison_rows": intersection_rows,
    }
    overall_metrics = {
        "precision": _safe_float(winner.get("mean_precision"), 0.0),
        "recall": _safe_float(winner.get("mean_recall"), 0.0),
        "f1": _safe_float(winner.get("mean_f1"), 0.0),
        "delta_vs_union_f1": _safe_float(winner.get("mean_delta_vs_union_f1"), 0.0),
        "coverage_preservation": _safe_float(winner.get("mean_coverage_preservation"), 0.0),
    }
    bundle = {
        "version": REPORT_BUNDLE_VERSION,
        "report_kind": "final_sweep",
        "generated_utc": ranked_payload.get("generated_utc"),
        "source_root": artifact_refs.get("run_root"),
        "model_family": str(model_family or "xgb"),
        "selection_summary": selection_summary,
        "overall_metrics": overall_metrics,
        "policy_layer": None,
        "per_class": _aggregate_per_class(analysis_payloads),
        "per_class_per_source": _aggregate_per_class_per_source(analysis_payloads),
        "boundary_hits": _aggregate_boundary_hits(analysis_payloads) if analysis_payloads else {},
        "seed_uncertainty": None,
        "calibration_diagnostics": _aggregate_calibration_diagnostics(analysis_payloads),
        "artifact_refs": artifact_refs,
        "warnings": [],
    }
    return bundle


def _render_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _render_bundle_md(bundle: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Calibration Report Bundle")
    lines.append("")
    lines.append(f"- Report kind: `{bundle.get('report_kind')}`")
    selection_summary = bundle.get("selection_summary") if isinstance(bundle.get("selection_summary"), dict) else {}
    overall = bundle.get("overall_metrics") if isinstance(bundle.get("overall_metrics"), dict) else {}
    if selection_summary:
        winner = selection_summary.get("winner") or selection_summary.get("winner_lane") or selection_summary.get("selected_policy_variant")
        if winner:
            lines.append(f"- Selection winner: `{winner}`")
    if overall:
        lines.append(f"- Precision: `{_safe_float(overall.get('precision'), 0.0):.4f}`")
        lines.append(f"- Recall: `{_safe_float(overall.get('recall'), 0.0):.4f}`")
        lines.append(f"- F1: `{_safe_float(overall.get('f1'), 0.0):.4f}`")
    lines.append("")

    policy = bundle.get("policy_layer") if isinstance(bundle.get("policy_layer"), dict) else None
    if policy:
        lines.append("## Policy Layer")
        lines.append(f"- Requested variant: `{policy.get('requested_variant')}`")
        lines.append(f"- Selected variant: `{policy.get('selected_variant')}`")
        trained = policy.get("trained_variants") if isinstance(policy.get("trained_variants"), list) else []
        if trained:
            lines.append(f"- Trained variants: `{', '.join(str(v) for v in trained)}`")
        lines.append(f"- Δ vs baseline F1: `{_safe_float(policy.get('delta_vs_baseline_f1'), 0.0):+.4f}`")
        lines.append("")

    per_class = bundle.get("per_class") if isinstance(bundle.get("per_class"), list) else []
    if per_class:
        lines.append("## Per-Class")
        lines.extend(
            _render_table(
                ["Label", "TP", "FP", "FN", "P", "R", "F1"],
                [
                    [
                        str(row.get("label")),
                        str(int(row.get("tp", 0))),
                        str(int(row.get("fp", 0))),
                        str(int(row.get("fn", 0))),
                        f"{_safe_float(row.get('precision'), 0.0):.4f}",
                        f"{_safe_float(row.get('recall'), 0.0):.4f}",
                        f"{_safe_float(row.get('f1'), 0.0):.4f}",
                    ]
                    for row in per_class
                ],
            )
        )
        lines.append("")

    per_source = bundle.get("per_class_per_source") if isinstance(bundle.get("per_class_per_source"), list) else []
    if per_source:
        lines.append("## Per-Class / Source Attribution")
        lines.extend(
            _render_table(
                ["Label", "Source", "TP", "FP", "Accepted", "Precision"],
                [
                    [
                        str(row.get("label")),
                        str(row.get("primary_source")),
                        str(int(row.get("tp", 0))),
                        str(int(row.get("fp", 0))),
                        str(int(row.get("accepted", 0))),
                        f"{_safe_float(row.get('precision'), 0.0):.4f}",
                    ]
                    for row in per_source[:80]
                ],
            )
        )
        lines.append("")

    boundary = bundle.get("boundary_hits") if isinstance(bundle.get("boundary_hits"), dict) else {}
    if boundary:
        lines.append("## Boundary Hits")
        lines.append(f"- Decision rows analyzed: `{int(boundary.get('decision_rows') or 0)}`")
        buckets = boundary.get("buckets") if isinstance(boundary.get("buckets"), dict) else {}
        for bucket_name in sorted(buckets):
            row = buckets[bucket_name]
            lines.append(
                f"- `{bucket_name}`: accepted=`{int(row.get('accepted') or 0)}`, rejected=`{int(row.get('rejected') or 0)}`, positives=`{int(row.get('positive_rows') or 0)}`, negatives=`{int(row.get('negative_rows') or 0)}`"
            )
        lines.append("")

    seed = bundle.get("seed_uncertainty") if isinstance(bundle.get("seed_uncertainty"), dict) else {}
    metrics = seed.get("metrics") if isinstance(seed.get("metrics"), dict) else {}
    if metrics:
        lines.append("## Seed Uncertainty")
        lines.extend(
            _render_table(
                ["Metric", "Mean", "Std", "Min", "Max"],
                [
                    [
                        field,
                        f"{_safe_float(payload.get('mean'), 0.0):.4f}",
                        f"{_safe_float(payload.get('std'), 0.0):.4f}",
                        f"{_safe_float(payload.get('min'), 0.0):.4f}",
                        f"{_safe_float(payload.get('max'), 0.0):.4f}",
                    ]
                    for field, payload in metrics.items()
                    if isinstance(payload, dict)
                ],
            )
        )
        lines.append("")

    diag = bundle.get("calibration_diagnostics") if isinstance(bundle.get("calibration_diagnostics"), dict) else {}
    if diag:
        lines.append("## Calibration Diagnostics")
        lines.append(f"- Candidate count: `{int(diag.get('candidate_count') or 0)}`")
        lines.append(f"- Brier score: `{_safe_float(diag.get('brier_score'), 0.0):.6f}`")
        lines.append(f"- ECE (10 bins): `{_safe_float(diag.get('ece_10'), 0.0):.6f}`")
        lines.append("")

    warnings = bundle.get("warnings") if isinstance(bundle.get("warnings"), list) else []
    if warnings:
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- `{warning}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_bundle_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    analysis_paths = _flatten_analysis_globs(args.analysis_json_glob or [])
    if args.analysis_json:
        analysis_paths.extend(_flatten_analysis_globs([str(args.analysis_json)]))
    analysis_payloads = [_load_json(path) for path in analysis_paths if path.exists()]
    model_family = str(getattr(args, "model_family", None) or "xgb").strip().lower() or "xgb"

    if args.eval_json:
        eval_path = Path(args.eval_json)
        eval_payload = _load_json(eval_path)
        meta_payload = _load_json(Path(args.meta_json)) if args.meta_json and Path(args.meta_json).exists() else {}
        policy_selection = (
            _load_json(Path(args.policy_selection_json))
            if args.policy_selection_json and Path(args.policy_selection_json).exists()
            else {}
        )
        return _build_single_run_bundle(
            eval_payload=eval_payload,
            analysis_payload=analysis_payloads[0] if analysis_payloads else None,
            meta_payload=meta_payload,
            policy_selection=policy_selection,
            model_family=model_family,
            artifact_refs={
                "eval_json": str(eval_path),
                "analysis_json": str(analysis_paths[0]) if analysis_paths else None,
                "meta_json": args.meta_json,
                "policy_selection_json": args.policy_selection_json,
                "output_dir": str(eval_path.parent),
            },
        )

    if args.results_summary_json:
        summary_path = Path(args.results_summary_json)
        summary_payload = _load_json(summary_path)
        rows = summary_payload.get("rows") if isinstance(summary_payload.get("rows"), list) else []
        ranked = summary_payload.get("ranked") if isinstance(summary_payload.get("ranked"), list) else []
        if rows:
            best_tag = None
            if ranked:
                best_tag = ranked[0].get("tag") or ranked[0].get("lane")
            row_analysis_paths: List[Path] = []
            for row in rows:
                row_tag = row.get("tag") or row.get("lane")
                if best_tag is not None and row_tag != best_tag:
                    continue
                raw_path = row.get("analysis_json")
                if not raw_path:
                    continue
                path = Path(str(raw_path))
                if path.exists():
                    row_analysis_paths.append(path)
            if row_analysis_paths:
                analysis_paths = row_analysis_paths
                analysis_payloads = [_load_json(path) for path in analysis_paths]
        return _build_postrun_bundle(
            summary_payload=summary_payload,
            analysis_payloads=analysis_payloads,
            model_family=model_family,
            artifact_refs={
                "results_summary_json": str(summary_path),
                "analysis_jsons": [str(path) for path in analysis_paths],
            },
        )

    if args.ranked_json:
        ranked_path = Path(args.ranked_json)
        ranked_payload = _load_json(ranked_path)
        return _build_final_sweep_bundle(
            ranked_payload=ranked_payload,
            analysis_payloads=analysis_payloads,
            model_family=model_family,
            artifact_refs={
                "ranked_json": str(ranked_path),
                "analysis_jsons": [str(path) for path in analysis_paths],
                "run_root": str(ranked_path.parent),
            },
        )

    raise SystemExit("one_input_mode_required")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json")
    parser.add_argument("--analysis-json")
    parser.add_argument("--analysis-json-glob", action="append", default=[])
    parser.add_argument("--meta-json")
    parser.add_argument("--policy-selection-json")
    parser.add_argument("--results-summary-json")
    parser.add_argument("--ranked-json")
    parser.add_argument("--model-family", default="xgb")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    bundle = _build_bundle_from_args(args)
    _write_json(Path(args.output_json), bundle)
    _write_text(Path(args.output_md), _render_bundle_md(bundle))
    print(json.dumps({"status": "ok", "output_json": str(Path(args.output_json).resolve()), "output_md": str(Path(args.output_md).resolve())}, indent=2))


if __name__ == "__main__":
    main()
