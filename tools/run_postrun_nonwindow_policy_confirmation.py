#!/usr/bin/env python3
"""Confirm the canonical non-window fallback policy on the promoted non-window lane."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.run_postrun_alpha_extension as alpha_ext


REFINED_TEXT_BIAS = -1.4
REFINED_SIM_BIAS = -1.2
REFINED_SAM_ONLY_MIN_PROB = 0.15
REFINED_CONSENSUS_IOU = 0.7


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def _variant_tag(name: str, *, alpha: float | None = None) -> str:
    if alpha is None:
        return str(name)
    return f"{name}_simq_a{str(float(alpha)).replace('-', 'm').replace('.', 'p')}"


def _parse_alphas(raw: str) -> List[float]:
    out: List[float] = []
    for chunk in str(raw or "").split(","):
        token = chunk.strip()
        if not token:
            continue
        value = float(token)
        if value not in out:
            out.append(value)
    return out


def _select_nonwindow_lane(run_root: Path) -> str:
    ranked = _load_json(run_root / "results_ranked.json")
    rows = ranked.get("views", {}).get("intersection", {}).get("ranked_lanes", [])
    for row in rows:
        lane = str(row.get("lane") or "")
        if lane == "nonwindow":
            return lane
    raise RuntimeError("nonwindow_lane_missing_from_main_sweep")


def _load_best_stack(run_root: Path, lane: str) -> Dict[str, Any]:
    raw = _load_json(run_root / "results_raw.json")
    best_stack = raw.get("search_results", {}).get("best_stack", {})
    payload = best_stack.get(lane)
    if not isinstance(payload, dict):
        raise RuntimeError(f"best_stack_missing_for_lane:{lane}")
    return payload


def _objective_args(meta: Dict[str, Any]) -> Dict[str, Any]:
    params = meta.get("calibration_objective_params") if isinstance(meta.get("calibration_objective_params"), dict) else {}
    return {
        "optimize": str(params.get("optimize") or meta.get("calibration_optimize") or "f1"),
        "target_fp_ratio": float(params.get("target_fp_ratio", 0.2)),
        "min_recall": float(params.get("min_recall", 0.6)),
        "steps": int(params.get("steps", 300)),
        "eval_iou": float(params.get("eval_iou", 0.5)),
        "dedupe_iou": float(params.get("dedupe_iou", 0.75)),
        "scoreless_iou": float(params.get("scoreless_iou", 0.0)),
        "use_val_split": bool(params.get("use_val_split", True)),
        "relax_fp_ratio": float(params.get("target_fp_ratio", 0.2)),
    }


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


def _refined_policy(base_policy: Dict[str, Any]) -> Dict[str, Any]:
    payload = copy.deepcopy(base_policy)
    payload["sam_bias_scope"] = "sam_only"
    bias_map = payload.get("logit_bias_by_source_class") if isinstance(payload.get("logit_bias_by_source_class"), dict) else {}
    text_map = bias_map.get("sam3_text") if isinstance(bias_map.get("sam3_text"), dict) else {}
    sim_map = bias_map.get("sam3_similarity") if isinstance(bias_map.get("sam3_similarity"), dict) else {}
    text_map["__default__"] = REFINED_TEXT_BIAS
    sim_map["__default__"] = REFINED_SIM_BIAS
    bias_map["sam3_text"] = text_map
    bias_map["sam3_similarity"] = sim_map
    payload["logit_bias_by_source_class"] = bias_map
    payload["sam_only_min_prob_default"] = REFINED_SAM_ONLY_MIN_PROB
    payload["consensus_iou_default"] = REFINED_CONSENSUS_IOU
    payload["consensus_class_aware"] = True
    source_consensus = payload.get("consensus_iou_by_source_class") if isinstance(payload.get("consensus_iou_by_source_class"), dict) else {}
    source_consensus["sam3_text"] = {"__default__": REFINED_CONSENSUS_IOU}
    source_consensus["sam3_similarity"] = {"__default__": REFINED_CONSENSUS_IOU}
    payload["consensus_iou_by_source_class"] = source_consensus
    return payload


def _baseline_rows(contexts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ctx in contexts:
        payload = _load_json(ctx["baseline_eval_json"])
        metrics = _extract_eval_metrics(payload)
        rows.append(
            {
                "tag": "baseline_hand",
                "seed": int(ctx["seed"]),
                "metrics": metrics,
                "baseline_metrics": metrics,
                "compare_to_baseline": _compare(metrics, metrics),
                "eval_json": str(ctx["baseline_eval_json"]),
                "analysis_json": None,
            }
        )
    return rows


def _contexts(run_root: Path, lane: str) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    for ctx in alpha_ext._winner_contexts(run_root, lane):
        if ctx.view != "full":
            continue
        base_meta = _load_json(ctx.meta_json)
        base_policy = _load_json(ctx.policy_json)
        contexts.append(
            {
                "lane": lane,
                "seed": int(ctx.seed),
                "model_json": ctx.model_json,
                "meta_json": ctx.meta_json,
                "policy_json": ctx.policy_json,
                "labeled_npz": ctx.labeled_npz,
                "prepass_jsonl": ctx.prepass_jsonl,
                "baseline_eval_json": ctx.baseline_eval_json,
                "base_meta": base_meta,
                "base_policy": base_policy,
            }
        )
    return contexts


def _run(cmd: Sequence[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=str(ROOT),
        check=True,
        capture_output=capture,
        text=True,
    )


def _run_policy_eval(
    ctx: Dict[str, Any],
    *,
    tag: str,
    policy_payload: Dict[str, Any],
    output_dir: Path,
    force: bool = False,
) -> Dict[str, Any]:
    run_dir = output_dir / tag / f"seed_{ctx['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    policy_json = run_dir / "policy.json"
    meta_json = run_dir / Path(str(ctx["meta_json"])).name
    eval_json = run_dir / "eval.json"
    analysis_json = run_dir / "analysis.json"
    tune_done = run_dir / "thresholds.done"

    _write_json(policy_json, policy_payload)
    meta_payload = copy.deepcopy(ctx["base_meta"])
    meta_payload["ensemble_policy"] = policy_payload
    _write_json(meta_json, meta_payload)
    params = _objective_args(meta_payload)
    if force or not tune_done.exists():
        cmd = [
            sys.executable,
            "tools/tune_ensemble_thresholds_xgb.py",
            "--model",
            str(ctx["model_json"]),
            "--meta",
            str(meta_json),
            "--data",
            str(ctx["labeled_npz"]),
            "--dataset",
            "qwen_dataset",
            "--optimize",
            str(params["optimize"]),
            "--target-fp-ratio",
            str(params["target_fp_ratio"]),
            "--relax-fp-ratio",
            str(params["relax_fp_ratio"]),
            "--min-recall",
            str(params["min_recall"]),
            "--steps",
            str(params["steps"]),
            "--eval-iou",
            str(params["eval_iou"]),
            "--dedupe-iou",
            str(params["dedupe_iou"]),
            "--scoreless-iou",
            str(params["scoreless_iou"]),
        ]
        if params["use_val_split"]:
            cmd.append("--use-val-split")
        _run(cmd)
        tune_done.write_text("ok\n", encoding="utf-8")

    if force or not eval_json.exists():
        result = _run(
            [
                sys.executable,
                "tools/eval_ensemble_xgb_dedupe.py",
                "--model",
                str(ctx["model_json"]),
                "--meta",
                str(meta_json),
                "--data",
                str(ctx["labeled_npz"]),
                "--dataset",
                "qwen_dataset",
                "--prepass-jsonl",
                str(ctx["prepass_jsonl"]),
                "--eval-iou",
                str(params["eval_iou"]),
                "--eval-iou-grid",
                str(params["eval_iou"]),
                "--dedupe-iou",
                str(params["dedupe_iou"]),
                "--scoreless-iou",
                str(params["scoreless_iou"]),
                "--policy-json",
                str(policy_json),
                "--analysis-json",
                str(analysis_json),
            ],
            capture=True,
        )
        eval_json.write_text(result.stdout.strip() + "\n", encoding="utf-8")
    payload = _load_json(eval_json)
    metrics = _extract_eval_metrics(payload)
    baseline_metrics = _extract_eval_metrics(_load_json(ctx["baseline_eval_json"]))
    return {
        "tag": tag,
        "seed": int(ctx["seed"]),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "compare_to_baseline": _compare(metrics, baseline_metrics),
        "eval_json": str(eval_json.resolve()),
        "analysis_json": str(analysis_json.resolve()),
    }


def _run_retrained_eval(
    ctx: Dict[str, Any],
    *,
    tag: str,
    policy_payload: Dict[str, Any],
    similarity_alpha: float,
    output_dir: Path,
    force: bool = False,
) -> Dict[str, Any]:
    run_dir = output_dir / tag / f"seed_{ctx['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = run_dir / "model"
    model_json = model_prefix.with_suffix(".json")
    model_meta = model_prefix.with_suffix(".meta.json")
    eval_json = run_dir / "eval.json"
    analysis_json = run_dir / "analysis.json"
    policy_json = run_dir / "policy.json"
    val_images_file = run_dir / "val_images.txt"

    _write_json(policy_json, policy_payload)
    base_meta = ctx["base_meta"]
    xgb_params = base_meta.get("xgb_params") if isinstance(base_meta.get("xgb_params"), dict) else {}
    split_val_images = base_meta.get("split_val_images") if isinstance(base_meta.get("split_val_images"), list) else []
    val_images_file.write_text(
        "\n".join(str(image).strip() for image in split_val_images if str(image).strip()) + "\n",
        encoding="utf-8",
    )
    if force or not eval_json.exists():
        train_cmd = [
            sys.executable,
            "tools/train_ensemble_xgb.py",
            "--input",
            str(ctx["labeled_npz"]),
            "--output",
            str(model_prefix),
            "--seed",
            str(int(ctx["seed"])),
            "--val-ratio",
            "0.2",
            "--max-depth",
            str(int(_safe_float(xgb_params.get("max_depth"), 12))),
            "--n-estimators",
            str(int(_safe_float(base_meta.get("n_estimators"), 600))),
            "--learning-rate",
            str(_safe_float(xgb_params.get("eta"), 0.05)),
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
            "--policy-json",
            str(policy_json),
        ]
        text_quality_cfg = base_meta.get("sam3_text_quality") if isinstance(base_meta.get("sam3_text_quality"), dict) else {}
        if bool(text_quality_cfg.get("enabled")):
            train_cmd += [
                "--train-sam3-text-quality",
                "--sam3-text-quality-alpha",
                str(_safe_float(text_quality_cfg.get("alpha"), 0.5)),
            ]
        if bool(base_meta.get("log1p_counts")):
            train_cmd.append("--log1p-counts")
        if bool(base_meta.get("standardize")):
            train_cmd.append("--standardize")
        split_head = base_meta.get("split_head") if isinstance(base_meta.get("split_head"), dict) else {}
        if bool(split_head.get("enabled")):
            train_cmd.append("--split-head-by-support")
        train_cmd += [
            "--train-sam3-similarity-quality",
            "--sam3-similarity-quality-alpha",
            str(float(similarity_alpha)),
        ]
        _run(train_cmd)
        _run(
            [
                sys.executable,
                "tools/tune_ensemble_thresholds_xgb.py",
                "--model",
                str(model_json),
                "--meta",
                str(model_meta),
                "--data",
                str(ctx["labeled_npz"]),
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
        result = _run(
            [
                sys.executable,
                "tools/eval_ensemble_xgb_dedupe.py",
                "--model",
                str(model_json),
                "--meta",
                str(model_meta),
                "--data",
                str(ctx["labeled_npz"]),
                "--dataset",
                "qwen_dataset",
                "--prepass-jsonl",
                str(ctx["prepass_jsonl"]),
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
            capture=True,
        )
        eval_json.write_text(result.stdout.strip() + "\n", encoding="utf-8")

    payload = _load_json(eval_json)
    metrics = _extract_eval_metrics(payload)
    baseline_metrics = _extract_eval_metrics(_load_json(ctx["baseline_eval_json"]))
    return {
        "tag": tag,
        "seed": int(ctx["seed"]),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "compare_to_baseline": _compare(metrics, baseline_metrics),
        "eval_json": str(eval_json.resolve()),
        "analysis_json": str(analysis_json.resolve()),
    }


def _aggregate(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["tag"]), []).append(row)
    ranked: List[Dict[str, Any]] = []
    for tag, items in grouped.items():
        deltas = [_safe_float(item.get("compare_to_baseline", {}).get("delta_f1"), 0.0) for item in items]
        ranked.append(
            {
                "tag": tag,
                "seed_count": len(items),
                "mean_precision": mean(_safe_float(item.get("metrics", {}).get("precision"), 0.0) for item in items),
                "mean_recall": mean(_safe_float(item.get("metrics", {}).get("recall"), 0.0) for item in items),
                "mean_f1": mean(_safe_float(item.get("metrics", {}).get("f1"), 0.0) for item in items),
                "mean_delta_f1": mean(deltas),
                "min_delta_f1": min(deltas) if deltas else 0.0,
                "non_negative_seed_count": sum(1 for value in deltas if value >= 0.0),
            }
        )
    ranked.sort(key=lambda row: (-_safe_float(row.get("mean_f1"), 0.0), row.get("tag")))
    return ranked


def _row_by_tag(rows: Sequence[Dict[str, Any]], tag: str) -> Dict[str, Any]:
    for row in rows:
        if str(row.get("tag") or "") == tag:
            return dict(row)
    return {}


def _decision_summary(
    *,
    run_root: Path,
    lane: str,
    best_stack: Dict[str, Any],
    ranked: Sequence[Dict[str, Any]],
    similarity_alphas: Sequence[float],
) -> Dict[str, Any]:
    baseline = _row_by_tag(ranked, "baseline_hand")
    refined = _row_by_tag(ranked, "refined_hand")
    refined_promoted = (
        bool(refined)
        and _safe_float(refined.get("mean_delta_f1"), 0.0) >= -0.0015
        and _safe_float(refined.get("min_delta_f1"), 0.0) >= -0.004
    )
    similarity_candidates = [row for row in ranked if str(row.get("tag") or "").startswith("refined_hand_simq_")]
    promoted_similarity = None
    if refined_promoted:
        for row in similarity_candidates:
            delta_vs_refined = _safe_float(row.get("mean_f1"), 0.0) - _safe_float(refined.get("mean_f1"), 0.0)
            required_non_negative = max(1, min(2, int(row.get("seed_count", 0))))
            if delta_vs_refined >= 0.001 and int(row.get("non_negative_seed_count", 0)) >= required_non_negative:
                promoted_similarity = dict(row)
                promoted_similarity["mean_delta_f1_vs_refined"] = delta_vs_refined
                promoted_similarity["required_non_negative_seed_count"] = required_non_negative
                break
    policy = _refined_policy(best_stack.get("policy") if isinstance(best_stack.get("policy"), dict) else {}) if refined_promoted else copy.deepcopy(best_stack.get("policy") or {})
    scenario = best_stack.get("scenario") if isinstance(best_stack.get("scenario"), dict) else {}
    sam_text_alpha = _safe_float(scenario.get("alpha"), 0.5)
    canonical_recipe = {
        "validation_status": (
            "validated_nonwindow_refined_policy_with_similarity_quality"
            if promoted_similarity
            else "validated_nonwindow_refined_policy" if refined_promoted else "fallback_to_main_sweep_nonwindow_policy"
        ),
        "winner_lane": lane,
        "xgb_hparams": copy.deepcopy(best_stack.get("hp") or {}),
        "scenario": {
            "split_head": bool(scenario.get("split_head")),
            "train_sam3_text_quality": bool(scenario.get("sam_quality", True)),
            "sam3_text_quality_alpha": sam_text_alpha,
            "train_sam3_similarity_quality": promoted_similarity is not None,
            "sam3_similarity_quality_alpha": None if promoted_similarity is None else float(str(promoted_similarity.get("tag")).split("_a", 1)[1].replace("p", ".")),
        },
        "policy": policy,
        "second_stage_policy_layer": {
            "enabled": False,
            "variant": "none",
            "reason": "not_promoted_for_nonwindow_fallback",
        },
        "expected_metrics": {
            "full_mean_f1": _safe_float((promoted_similarity or refined or baseline).get("mean_f1"), 0.0),
            "full_mean_delta_f1_vs_baseline": _safe_float((promoted_similarity or refined or baseline).get("mean_delta_f1"), 0.0),
        },
    }
    return {
        "stage": "postrun_nonwindow_policy_confirmation",
        "generated_utc": _ts(),
        "run_root": str(run_root),
        "nonwindow_lane": lane,
        "tested_similarity_alphas": [float(alpha) for alpha in similarity_alphas],
        "baseline_metrics": baseline,
        "refined_metrics": refined,
        "refined_policy_status": "promoted" if refined_promoted else "rejected",
        "similarity_quality_status": "promoted" if promoted_similarity else "rejected",
        "promoted_similarity_variant": promoted_similarity,
        "canonical_recipe": canonical_recipe,
        "reason_codes": [
            "refined_policy_non_regressive" if refined_promoted else "refined_policy_below_gate",
            "nonwindow_similarity_quality_promoted" if promoted_similarity else "nonwindow_similarity_quality_not_promoted",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--similarity-alphas", default="0.2,0.5")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (run_root / "postrun_nonwindow_policy_confirmation").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lane = _select_nonwindow_lane(run_root)
    best_stack = _load_best_stack(run_root, lane)
    contexts = _contexts(run_root, lane)
    rows = _baseline_rows(contexts)
    refined_policy = _refined_policy(best_stack.get("policy") if isinstance(best_stack.get("policy"), dict) else {})
    for ctx in contexts:
        rows.append(
            _run_policy_eval(
                ctx,
                tag="refined_hand",
                policy_payload=refined_policy,
                output_dir=output_dir,
                force=bool(args.force),
            )
        )
    similarity_alphas = _parse_alphas(args.similarity_alphas)
    for alpha in similarity_alphas:
        tag = _variant_tag("refined_hand", alpha=alpha)
        for ctx in contexts:
            rows.append(
                _run_retrained_eval(
                    ctx,
                    tag=tag,
                    policy_payload=refined_policy,
                    similarity_alpha=float(alpha),
                    output_dir=output_dir,
                    force=bool(args.force),
                )
            )

    ranked = _aggregate(rows)
    summary = {
        "run_root": str(run_root),
        "nonwindow_lane": lane,
        "rows": rows,
        "ranked": ranked,
    }
    decision_summary = _decision_summary(
        run_root=run_root,
        lane=lane,
        best_stack=best_stack,
        ranked=ranked,
        similarity_alphas=similarity_alphas,
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
            str(output_dir / "*" / "seed_*" / "analysis.json"),
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
