#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrainContext:
    lane: str
    view: str
    seed: int
    labeled_npz: Path
    prepass_jsonl: Path
    val_images_file: Path


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}", flush=True)


def _run(cmd: Sequence[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    _log("RUN " + " ".join(str(c) for c in cmd))
    return subprocess.run(
        list(cmd),
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=capture,
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _parse_seed_list(raw: str) -> List[int]:
    out: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise SystemExit("no seeds provided")
    return out


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _lane_image_set(labeled_npz: Path) -> List[str]:
    data = np.load(labeled_npz, allow_pickle=True)
    images = set()
    for row in data["meta"]:
        try:
            payload = json.loads(str(row))
        except Exception:
            continue
        image = str(payload.get("image") or "").strip()
        if image:
            images.add(image)
    return sorted(images)


def _build_val_split(images: Sequence[str], *, seed: int, val_ratio: float) -> List[str]:
    values = sorted({str(x).strip() for x in images if str(x).strip()})
    if not values:
        return []
    rng = random.Random(int(seed))
    values = list(values)
    rng.shuffle(values)
    val_count = max(1, int(round(len(values) * float(val_ratio))))
    val_count = min(val_count, len(values) - 1) if len(values) > 1 else 1
    return sorted(values[:val_count])


def _hash_obj(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _extract_eval_metrics(payload: Dict[str, Any]) -> Dict[str, float]:
    post_xgb = (
        payload.get("metric_tiers", {})
        .get("post_xgb", {})
        .get("accepted_all", {})
    )
    post_cluster_union = (
        payload.get("metric_tiers", {})
        .get("post_cluster", {})
        .get("source_attributed", {})
        .get("yolo_rfdetr_union", {})
    )
    coverage = payload.get("coverage_upper_bound", {}).get("candidate_all", {})

    p = _safe_float(post_xgb.get("precision"), _safe_float(payload.get("precision"), 0.0))
    r = _safe_float(post_xgb.get("recall"), _safe_float(payload.get("recall"), 0.0))
    f1 = _safe_float(post_xgb.get("f1"), _safe_float(payload.get("f1"), 0.0))
    union_f1 = _safe_float(post_cluster_union.get("f1"), 0.0)
    coverage_ub = _safe_float(coverage.get("recall_upper_bound"), 0.0)
    coverage_pres = (r / coverage_ub) if coverage_ub > 0.0 else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "union_f1": union_f1,
        "delta_vs_union_f1": f1 - union_f1,
        "coverage_upper_bound": coverage_ub,
        "coverage_preservation": coverage_pres,
    }


def _threshold_source(meta: Dict[str, Any]) -> Dict[str, float]:
    for key in (
        "calibrated_thresholds_objective",
        "calibrated_thresholds_relaxed_smoothed",
        "calibrated_thresholds_relaxed",
        "calibrated_thresholds",
    ):
        raw = meta.get(key)
        if not isinstance(raw, dict):
            continue
        out: Dict[str, float] = {}
        for label, value in raw.items():
            name = str(label or "").strip().lower()
            if not name:
                continue
            out[name] = _safe_float(value, 0.5)
        if out:
            return out
    return {}


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _logit(value: float) -> float:
    p = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _build_policy(
    *,
    base_thresholds: Dict[str, float],
    threshold_shift: float,
    source_aware: bool,
    bias_text: float,
    bias_sim: float,
    sam_floor: float,
    consensus_text: float,
    consensus_sim: float,
) -> Dict[str, Any]:
    policy: Dict[str, Any] = {}
    if abs(float(threshold_shift)) > 1e-12:
        overrides: Dict[str, float] = {}
        for label, threshold in base_thresholds.items():
            overrides[label] = _sigmoid(_logit(threshold) + float(threshold_shift))
        policy["threshold_by_class_override"] = overrides
    if source_aware:
        policy["logit_bias_by_source_class"] = {
            "sam3_text": {"__default__": float(bias_text)},
            "sam3_similarity": {"__default__": float(bias_sim)},
        }
        policy["sam_only_min_prob_default"] = float(sam_floor)
        policy["consensus_iou_by_source_class"] = {
            "sam3_text": {"__default__": float(consensus_text)},
            "sam3_similarity": {"__default__": float(consensus_sim)},
        }
        policy["consensus_iou_default"] = float(min(consensus_text, consensus_sim))
        policy["consensus_class_aware"] = True
    return policy


def _coarse_param_grid() -> List[Dict[str, Any]]:
    return [
        {"max_depth": 6, "n_estimators": 600, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1.0, "gamma": 0.0, "reg_lambda": 1.0, "reg_alpha": 0.0},
        {"max_depth": 8, "n_estimators": 600, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1.0, "gamma": 0.0, "reg_lambda": 1.0, "reg_alpha": 0.0},
        {"max_depth": 10, "n_estimators": 600, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1.0, "gamma": 0.0, "reg_lambda": 1.0, "reg_alpha": 0.0},
        {"max_depth": 8, "n_estimators": 900, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1.0, "gamma": 0.0, "reg_lambda": 1.0, "reg_alpha": 0.0},
        {"max_depth": 8, "n_estimators": 1200, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1.0, "gamma": 0.0, "reg_lambda": 1.0, "reg_alpha": 0.0},
        {"max_depth": 6, "n_estimators": 900, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 1.0, "gamma": 0.0, "reg_lambda": 1.0, "reg_alpha": 0.0},
        {"max_depth": 8, "n_estimators": 900, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 0.85, "min_child_weight": 1.0, "gamma": 0.0, "reg_lambda": 1.0, "reg_alpha": 0.0},
        {"max_depth": 8, "n_estimators": 900, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 3.0, "gamma": 0.0, "reg_lambda": 1.0, "reg_alpha": 0.0},
        {"max_depth": 8, "n_estimators": 900, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3.0, "gamma": 0.2, "reg_lambda": 1.0, "reg_alpha": 0.0},
        {"max_depth": 10, "n_estimators": 1200, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3.0, "gamma": 0.2, "reg_lambda": 3.0, "reg_alpha": 0.1},
        {"max_depth": 6, "n_estimators": 1200, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 1.0, "min_child_weight": 1.0, "gamma": 0.2, "reg_lambda": 3.0, "reg_alpha": 0.0},
        {"max_depth": 10, "n_estimators": 900, "learning_rate": 0.08, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1.0, "gamma": 0.0, "reg_lambda": 1.0, "reg_alpha": 0.0},
    ]


def _refine_neighbors(base: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for lr_delta in (-0.02, -0.01, 0.01, 0.02):
        cand = dict(base)
        cand["learning_rate"] = min(max(_safe_float(base["learning_rate"]) + lr_delta, 0.01), 0.2)
        out.append(cand)
    for depth_delta in (-2, 2):
        cand = dict(base)
        cand["max_depth"] = min(max(int(base["max_depth"]) + depth_delta, 4), 14)
        out.append(cand)
    for child in (1.0, 2.0, 3.0, 5.0):
        cand = dict(base)
        cand["min_child_weight"] = float(child)
        out.append(cand)
    for field in ("subsample", "colsample_bytree"):
        for delta in (-0.1, 0.1):
            cand = dict(base)
            cand[field] = min(max(_safe_float(base[field]) + delta, 0.6), 1.0)
            out.append(cand)
    dedup: Dict[str, Dict[str, Any]] = {}
    for cand in out:
        dedup[_hash_obj(cand)] = cand
    return list(dedup.values())


def _policy_candidates(base_thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    # Wide shift grid (including negative side) to avoid boundary-clipped optima.
    for shift in (-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4):
        out.append(
            _build_policy(
                base_thresholds=base_thresholds,
                threshold_shift=shift,
                source_aware=False,
                bias_text=0.0,
                bias_sim=0.0,
                sam_floor=0.0,
                consensus_text=0.0,
                consensus_sim=0.0,
            )
        )
    presets = [
        (-1.0, -0.8, 0.15, 0.7, 0.7),
        (-0.8, -0.6, 0.15, 0.7, 0.5),
        (-0.8, -0.4, 0.10, 0.5, 0.5),
        (-0.6, -0.4, 0.10, 0.5, 0.5),
        (-0.6, -0.3, 0.05, 0.5, 0.3),
        (-0.5, -0.3, 0.05, 0.3, 0.3),
        (-0.4, -0.2, 0.05, 0.3, 0.3),
        (-0.3, -0.2, 0.00, 0.3, 0.3),
    ]
    for shift in (-0.8, -0.6, -0.4, -0.2, 0.0, 0.2):
        for bias_text, bias_sim, floor, con_text, con_sim in presets:
            out.append(
                _build_policy(
                    base_thresholds=base_thresholds,
                    threshold_shift=shift,
                    source_aware=True,
                    bias_text=bias_text,
                    bias_sim=bias_sim,
                    sam_floor=floor,
                    consensus_text=con_text,
                    consensus_sim=con_sim,
                )
            )
    dedup: Dict[str, Dict[str, Any]] = {}
    for policy in out:
        dedup[_hash_obj(policy)] = policy
    return list(dedup.values())


def _train_tune_eval(
    *,
    context: TrainContext,
    hp: Dict[str, Any],
    split_head: bool,
    sam_quality: bool,
    sam_quality_alpha: float,
    policy: Dict[str, Any],
    out_dir: Path,
    optimize: str,
    target_fp_ratio: float,
    min_recall: float,
    relax_fp_ratio: float,
    threshold_steps: int,
    eval_iou: float,
    dedupe_iou: float,
    scoreless_iou: float,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    hp_id = _hash_obj(hp)
    scenario = {
        "split_head": bool(split_head),
        "sam_quality": bool(sam_quality),
        "alpha": round(float(sam_quality_alpha), 4),
    }
    sc_id = _hash_obj(scenario)
    model_prefix = out_dir / f"model_{hp_id}_{sc_id}"
    model_json = Path(f"{model_prefix}.json")
    model_meta = Path(f"{model_prefix}.meta.json")

    if not model_json.exists() or not model_meta.exists():
        cmd = [
            sys.executable,
            "tools/train_ensemble_xgb.py",
            "--input",
            str(context.labeled_npz),
            "--output",
            str(model_prefix),
            "--seed",
            str(int(context.seed)),
            "--val-ratio",
            "0.2",
            "--max-depth",
            str(int(hp["max_depth"])),
            "--n-estimators",
            str(int(hp["n_estimators"])),
            "--learning-rate",
            str(float(hp["learning_rate"])),
            "--subsample",
            str(float(hp["subsample"])),
            "--colsample-bytree",
            str(float(hp["colsample_bytree"])),
            "--min-child-weight",
            str(float(hp["min_child_weight"])),
            "--gamma",
            str(float(hp["gamma"])),
            "--reg-lambda",
            str(float(hp["reg_lambda"])),
            "--reg-alpha",
            str(float(hp["reg_alpha"])),
            "--tree-method",
            "hist",
            "--max-bin",
            "256",
            "--early-stopping-rounds",
            "50",
            "--threshold-steps",
            str(int(threshold_steps)),
            "--optimize",
            str(optimize),
            "--target-fp-ratio",
            str(float(target_fp_ratio)),
            "--min-recall",
            str(float(min_recall)),
            "--per-class",
            "--fixed-val-images",
            str(context.val_images_file),
        ]
        if split_head:
            cmd.append("--split-head-by-support")
        if sam_quality:
            cmd += ["--train-sam3-text-quality", "--sam3-text-quality-alpha", str(float(sam_quality_alpha))]
        _run(cmd)

    tune_tag = f"tuned_{_hash_obj({'hp': hp, 'sc': scenario, 'opt': optimize, 'tfr': target_fp_ratio, 'mr': min_recall, 'rfr': relax_fp_ratio, 'steps': threshold_steps, 'diou': dedupe_iou, 'siou': scoreless_iou})}"
    tune_done = out_dir / f"{model_prefix.name}.{tune_tag}.done"
    if not tune_done.exists():
        _run(
            [
                sys.executable,
                "tools/tune_ensemble_thresholds_xgb.py",
                "--model",
                str(model_json),
                "--meta",
                str(model_meta),
                "--data",
                str(context.labeled_npz),
                "--dataset",
                "qwen_dataset",
                "--optimize",
                str(optimize),
                "--target-fp-ratio",
                str(float(target_fp_ratio)),
                "--relax-fp-ratio",
                str(float(relax_fp_ratio)),
                "--min-recall",
                str(float(min_recall)),
                "--steps",
                str(int(threshold_steps)),
                "--eval-iou",
                str(float(eval_iou)),
                "--dedupe-iou",
                str(float(dedupe_iou)),
                "--scoreless-iou",
                str(float(scoreless_iou)),
                "--use-val-split",
            ]
        )
        tune_done.write_text("ok\n", encoding="utf-8")

    eval_id = _hash_obj(policy)
    eval_out = out_dir / f"eval_{hp_id}_{sc_id}_{eval_id}.json"
    if not eval_out.exists():
        policy_path = out_dir / f"policy_{eval_id}.json"
        policy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
        result = _run(
            [
                sys.executable,
                "tools/eval_ensemble_xgb_dedupe.py",
                "--model",
                str(model_json),
                "--meta",
                str(model_meta),
                "--data",
                str(context.labeled_npz),
                "--dataset",
                "qwen_dataset",
                "--prepass-jsonl",
                str(context.prepass_jsonl),
                "--eval-iou",
                str(float(eval_iou)),
                "--eval-iou-grid",
                str(float(eval_iou)),
                "--dedupe-iou",
                str(float(dedupe_iou)),
                "--scoreless-iou",
                str(float(scoreless_iou)),
                "--use-val-split",
                "--policy-json",
                str(policy_path),
            ],
            capture=True,
        )
        eval_out.write_text(result.stdout.strip() + "\n", encoding="utf-8")
    payload = _load_json(eval_out)
    metrics = _extract_eval_metrics(payload)
    return {
        "model_json": str(model_json),
        "model_meta": str(model_meta),
        "eval_json": str(eval_out),
        "hp": hp,
        "scenario": scenario,
        "policy": policy,
        "metrics": metrics,
    }


def _rank_lane_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    guardrail_delta_min: float,
    coverage_tolerance: float,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["lane"]), []).append(row)

    coverage_best = 0.0
    for lane_rows in grouped.values():
        lane_cov = mean(_safe_float(x.get("coverage_preservation"), 0.0) for x in lane_rows)
        coverage_best = max(coverage_best, lane_cov)

    out: List[Dict[str, Any]] = []
    for lane, lane_rows in grouped.items():
        mean_p = mean(_safe_float(x.get("precision"), 0.0) for x in lane_rows)
        mean_r = mean(_safe_float(x.get("recall"), 0.0) for x in lane_rows)
        mean_f1 = mean(_safe_float(x.get("f1"), 0.0) for x in lane_rows)
        mean_du = mean(_safe_float(x.get("delta_vs_union_f1"), 0.0) for x in lane_rows)
        mean_cov = mean(_safe_float(x.get("coverage_preservation"), 0.0) for x in lane_rows)
        std_f1 = pstdev([_safe_float(x.get("f1"), 0.0) for x in lane_rows]) if len(lane_rows) > 1 else 0.0
        no_negative = all(_safe_float(x.get("delta_vs_union_f1"), 0.0) >= 0.0 for x in lane_rows)
        cov_ok = mean_cov >= (coverage_best - float(coverage_tolerance))
        pass_guardrail = bool(mean_du >= float(guardrail_delta_min) and no_negative and cov_ok)
        out.append(
            {
                "lane": lane,
                "mean_precision": mean_p,
                "mean_recall": mean_r,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "mean_delta_vs_union_f1": mean_du,
                "mean_coverage_preservation": mean_cov,
                "guardrail_pass": pass_guardrail,
            }
        )
    out.sort(key=lambda x: (-int(bool(x["guardrail_pass"])), -_safe_float(x["mean_f1"]), -_safe_float(x["mean_delta_vs_union_f1"]), -_safe_float(x["mean_coverage_preservation"])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final calibration sweep for default prepass decision.")
    parser.add_argument("--dataset", default="qwen_dataset")
    parser.add_argument("--run-root", default=f"tmp/final_calibration_sweep_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}")
    parser.add_argument("--nonwindow-key", default="20c8d44d69f51b2ffe528fb500e75672a306f67d")
    parser.add_argument("--window-key", default="ceab65b2bff24d316ca5f858addaffed8abfdb11")
    parser.add_argument("--classifier-id", default="uploads/classifiers/DinoV3_best_model_large.pkl")
    parser.add_argument("--seeds", default="42,1337,2025")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--candidate-embed-dim", type=int, default=1024)
    parser.add_argument("--image-embed-dims", default="0,1024")
    parser.add_argument("--support-iou", type=float, default=0.5)
    parser.add_argument("--context-radius", type=float, default=0.075)
    parser.add_argument("--optimize", default="f1", choices=["f1", "recall", "tp"])
    parser.add_argument("--target-fp-ratio", type=float, default=0.2)
    parser.add_argument("--relax-fp-ratio", type=float, default=0.2)
    parser.add_argument("--min-recall", type=float, default=0.6)
    parser.add_argument("--threshold-steps", type=int, default=300)
    parser.add_argument("--eval-iou", type=float, default=0.5)
    parser.add_argument("--dedupe-iou", type=float, default=0.75)
    parser.add_argument("--scoreless-iou", type=float, default=0.0)
    parser.add_argument("--guardrail-delta-min", type=float, default=0.02)
    parser.add_argument("--coverage-tolerance", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nonwindow-prepass-jsonl", default="")
    parser.add_argument("--window-prepass-jsonl", default="")
    parser.add_argument("--nonwindow-noimg-features", default="")
    parser.add_argument("--nonwindow-noimg-labeled", default="")
    parser.add_argument("--nonwindow-imgctx-features", default="")
    parser.add_argument("--nonwindow-imgctx-labeled", default="")
    parser.add_argument("--window-noimg-features", default="")
    parser.add_argument("--window-noimg-labeled", default="")
    parser.add_argument("--window-imgctx-features", default="")
    parser.add_argument("--window-imgctx-labeled", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_root = (REPO_ROOT / args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    lane_cmd: List[str] = [
        sys.executable,
        "tools/build_feature_lanes_from_prepass.py",
        "--dataset",
        str(args.dataset),
        "--run-root",
        str(run_root),
        "--nonwindow-key",
        str(args.nonwindow_key),
        "--window-key",
        str(args.window_key),
        "--classifier-id",
        str(args.classifier_id),
        "--candidate-embed-dim",
        str(int(args.candidate_embed_dim)),
        "--image-embed-dims",
        str(args.image_embed_dims),
        "--support-iou",
        str(float(args.support_iou)),
        "--context-radius",
        str(float(args.context_radius)),
        "--device",
        str(args.device),
    ]
    passthrough = {
        "--nonwindow-prepass-jsonl": str(args.nonwindow_prepass_jsonl).strip(),
        "--window-prepass-jsonl": str(args.window_prepass_jsonl).strip(),
        "--nonwindow-noimg-features": str(args.nonwindow_noimg_features).strip(),
        "--nonwindow-noimg-labeled": str(args.nonwindow_noimg_labeled).strip(),
        "--nonwindow-imgctx-features": str(args.nonwindow_imgctx_features).strip(),
        "--nonwindow-imgctx-labeled": str(args.nonwindow_imgctx_labeled).strip(),
        "--window-noimg-features": str(args.window_noimg_features).strip(),
        "--window-noimg-labeled": str(args.window_noimg_labeled).strip(),
        "--window-imgctx-features": str(args.window_imgctx_features).strip(),
        "--window-imgctx-labeled": str(args.window_imgctx_labeled).strip(),
    }
    for flag, value in passthrough.items():
        if value:
            lane_cmd += [flag, value]
    if args.force:
        lane_cmd.append("--force")
    _run(lane_cmd)

    manifest = _load_json(run_root / "lane_manifest.json")
    lanes = manifest["lanes"]
    lane_ids = sorted(lanes.keys())
    seeds = _parse_seed_list(args.seeds)

    lane_images_full: Dict[str, List[str]] = {}
    lane_images_intersection = _load_json(Path(manifest["views"]["intersection"]["images_file"]))
    for lane_id in lane_ids:
        lane_images_full[lane_id] = _lane_image_set(Path(lanes[lane_id]["labeled"]))

    split_root = run_root / "splits"
    contexts: Dict[Tuple[str, str, int], TrainContext] = {}
    for lane_id in lane_ids:
        variant = str(lanes[lane_id]["variant"])
        for view in ("full", "intersection"):
            if view == "full":
                labeled = Path(lanes[lane_id]["labeled"])
                prepass = Path(lanes[lane_id]["prepass_jsonl"])
                images = lane_images_full[lane_id]
            else:
                labeled = Path(manifest["intersection_labeled"][lane_id]["path"])
                prepass = Path(manifest["intersection_prepass_jsonl"][variant])
                images = list(lane_images_intersection)
            for seed in seeds:
                val_images = _build_val_split(images, seed=seed, val_ratio=float(args.val_ratio))
                split_file = split_root / view / lane_id / f"seed_{seed}.val_images.json"
                split_file.parent.mkdir(parents=True, exist_ok=True)
                split_file.write_text(json.dumps(val_images, indent=2), encoding="utf-8")
                contexts[(lane_id, view, seed)] = TrainContext(
                    lane=lane_id,
                    view=view,
                    seed=int(seed),
                    labeled_npz=labeled,
                    prepass_jsonl=prepass,
                    val_images_file=split_file,
                )

    coarse = _coarse_param_grid()
    search_results: Dict[str, Any] = {"coarse": {}, "refine": {}, "best_hp": {}, "best_stack": {}}

    # Stage A/B: hyperparam search per lane on full-view first seed.
    primary_seed = int(seeds[0])
    for lane_id in lane_ids:
        context = contexts[(lane_id, "full", primary_seed)]
        lane_model_dir = run_root / "search" / lane_id / "hp"
        coarse_rows: List[Dict[str, Any]] = []
        for idx, hp in enumerate(coarse):
            result = _train_tune_eval(
                context=context,
                hp=hp,
                split_head=True,
                sam_quality=True,
                sam_quality_alpha=0.35,
                policy={},
                out_dir=lane_model_dir / f"coarse_{idx:02d}",
                optimize=str(args.optimize),
                target_fp_ratio=float(args.target_fp_ratio),
                min_recall=float(args.min_recall),
                relax_fp_ratio=float(args.relax_fp_ratio),
                threshold_steps=int(args.threshold_steps),
                eval_iou=float(args.eval_iou),
                dedupe_iou=float(args.dedupe_iou),
                scoreless_iou=float(args.scoreless_iou),
            )
            coarse_rows.append(result)
        coarse_rows.sort(key=lambda row: -_safe_float(row["metrics"]["f1"]))
        search_results["coarse"][lane_id] = coarse_rows

        top3 = coarse_rows[:3]
        refine_candidates: Dict[str, Dict[str, Any]] = {}
        for row in top3:
            for cand in _refine_neighbors(row["hp"]):
                refine_candidates[_hash_obj(cand)] = cand
        refine_rows: List[Dict[str, Any]] = []
        for ridx, hp in enumerate(refine_candidates.values()):
            result = _train_tune_eval(
                context=context,
                hp=hp,
                split_head=True,
                sam_quality=True,
                sam_quality_alpha=0.35,
                policy={},
                out_dir=lane_model_dir / f"refine_{ridx:02d}",
                optimize=str(args.optimize),
                target_fp_ratio=float(args.target_fp_ratio),
                min_recall=float(args.min_recall),
                relax_fp_ratio=float(args.relax_fp_ratio),
                threshold_steps=int(args.threshold_steps),
                eval_iou=float(args.eval_iou),
                dedupe_iou=float(args.dedupe_iou),
                scoreless_iou=float(args.scoreless_iou),
            )
            refine_rows.append(result)
        refine_rows.sort(key=lambda row: -_safe_float(row["metrics"]["f1"]))
        search_results["refine"][lane_id] = refine_rows
        best_hp = refine_rows[0]["hp"] if refine_rows else coarse_rows[0]["hp"]
        search_results["best_hp"][lane_id] = best_hp

        # Stage C: split-head + source policy search on same context.
        stack_rows: List[Dict[str, Any]] = []
        scenario_opts: List[Tuple[bool, bool, float]] = []
        for split_head in (True, False):
            scenario_opts.append((split_head, False, 0.35))
            for alpha in (0.25, 0.35, 0.5):
                scenario_opts.append((split_head, True, alpha))

        for split_head, sam_quality, alpha in scenario_opts:
            baseline_result = _train_tune_eval(
                context=context,
                hp=best_hp,
                split_head=split_head,
                sam_quality=sam_quality,
                sam_quality_alpha=alpha,
                policy={},
                out_dir=run_root / "search" / lane_id / "stack" / f"s{int(split_head)}_q{int(sam_quality)}_a{int(alpha*100)}",
                optimize=str(args.optimize),
                target_fp_ratio=float(args.target_fp_ratio),
                min_recall=float(args.min_recall),
                relax_fp_ratio=float(args.relax_fp_ratio),
                threshold_steps=int(args.threshold_steps),
                eval_iou=float(args.eval_iou),
                dedupe_iou=float(args.dedupe_iou),
                scoreless_iou=float(args.scoreless_iou),
            )
            base_meta = _load_json(Path(baseline_result["model_meta"]))
            thresholds = _threshold_source(base_meta)
            for policy in _policy_candidates(thresholds):
                result = _train_tune_eval(
                    context=context,
                    hp=best_hp,
                    split_head=split_head,
                    sam_quality=sam_quality,
                    sam_quality_alpha=alpha,
                    policy=policy,
                    out_dir=run_root / "search" / lane_id / "stack" / f"s{int(split_head)}_q{int(sam_quality)}_a{int(alpha*100)}",
                    optimize=str(args.optimize),
                    target_fp_ratio=float(args.target_fp_ratio),
                    min_recall=float(args.min_recall),
                    relax_fp_ratio=float(args.relax_fp_ratio),
                    threshold_steps=int(args.threshold_steps),
                    eval_iou=float(args.eval_iou),
                    dedupe_iou=float(args.dedupe_iou),
                    scoreless_iou=float(args.scoreless_iou),
                )
                stack_rows.append(result)

        stack_rows.sort(key=lambda row: -_safe_float(row["metrics"]["f1"]))
        search_results["best_stack"][lane_id] = {
            "hp": best_hp,
            "scenario": stack_rows[0]["scenario"],
            "policy": stack_rows[0]["policy"],
            "search_context": {"view": "full", "seed": primary_seed},
        }

    # Final matrix across all lanes/views/seeds with selected hp+scenario+policy.
    matrix_rows: List[Dict[str, Any]] = []
    for lane_id in lane_ids:
        selected = search_results["best_stack"][lane_id]
        hp = selected["hp"]
        split_head = bool(selected["scenario"]["split_head"])
        sam_quality = bool(selected["scenario"]["sam_quality"])
        alpha = float(selected["scenario"]["alpha"])
        policy = selected["policy"]
        for view in ("intersection", "full"):
            for seed in seeds:
                context = contexts[(lane_id, view, int(seed))]
                result = _train_tune_eval(
                    context=context,
                    hp=hp,
                    split_head=split_head,
                    sam_quality=sam_quality,
                    sam_quality_alpha=alpha,
                    policy=policy,
                    out_dir=run_root / "final_matrix" / lane_id / view / f"seed_{seed}",
                    optimize=str(args.optimize),
                    target_fp_ratio=float(args.target_fp_ratio),
                    min_recall=float(args.min_recall),
                    relax_fp_ratio=float(args.relax_fp_ratio),
                    threshold_steps=int(args.threshold_steps),
                    eval_iou=float(args.eval_iou),
                    dedupe_iou=float(args.dedupe_iou),
                    scoreless_iou=float(args.scoreless_iou),
                )
                row = {
                    "lane": lane_id,
                    "view": view,
                    "seed": int(seed),
                    "precision": result["metrics"]["precision"],
                    "recall": result["metrics"]["recall"],
                    "f1": result["metrics"]["f1"],
                    "delta_vs_union_f1": result["metrics"]["delta_vs_union_f1"],
                    "coverage_preservation": result["metrics"]["coverage_preservation"],
                    "coverage_upper_bound": result["metrics"]["coverage_upper_bound"],
                    "eval_json": result["eval_json"],
                    "model_json": result["model_json"],
                    "model_meta": result["model_meta"],
                }
                matrix_rows.append(row)

    raw_payload = {
        "generated_utc": _ts(),
        "dataset": str(args.dataset),
        "run_root": str(run_root),
        "assumptions": {
            "selection_objective": "max_f1_with_guardrail",
            "guardrail_delta_min": float(args.guardrail_delta_min),
            "coverage_tolerance": float(args.coverage_tolerance),
            "eval_iou": float(args.eval_iou),
            "dedupe_iou": float(args.dedupe_iou),
            "scoreless_iou": float(args.scoreless_iou),
            "search_scope": "xgb_family_only",
            "selection_view": "intersection",
            "validation_protocol": "3_seed_fixed_split",
        },
        "seeds": seeds,
        "search_results": search_results,
        "matrix_rows": matrix_rows,
    }
    raw_json = run_root / "results_raw.json"
    _write_json(raw_json, raw_payload)

    view_rankings: Dict[str, Any] = {}
    for view in ("intersection", "full"):
        rows = [row for row in matrix_rows if row["view"] == view]
        ranked = _rank_lane_rows(
            rows,
            guardrail_delta_min=float(args.guardrail_delta_min),
            coverage_tolerance=float(args.coverage_tolerance),
        )
        view_rankings[view] = {"ranked_lanes": ranked}

    intersection_ranked = view_rankings["intersection"]["ranked_lanes"]
    winner = intersection_ranked[0] if intersection_ranked else {}
    ranked_payload = {
        "generated_utc": _ts(),
        "run_root": str(run_root),
        "assumptions": raw_payload["assumptions"],
        "views": view_rankings,
        "winner": {
            "lane": winner.get("lane"),
            "selection_view": "intersection",
            "mean_f1": winner.get("mean_f1"),
            "mean_delta_vs_union_f1": winner.get("mean_delta_vs_union_f1"),
            "mean_coverage_preservation": winner.get("mean_coverage_preservation"),
            "guardrail_pass": winner.get("guardrail_pass"),
        },
    }
    ranked_json = run_root / "results_ranked.json"
    _write_json(ranked_json, ranked_payload)

    final_default = {
        "generated_utc": _ts(),
        "dataset": str(args.dataset),
        "winner_lane": winner.get("lane"),
        "lane_settings": search_results["best_stack"].get(str(winner.get("lane")), {}),
        "selection_policy": raw_payload["assumptions"],
    }
    default_json = run_root / "final_default_recipe.json"
    _write_json(default_json, final_default)

    report_md = run_root / "final_report.md"
    _run(
        [
            sys.executable,
            "tools/report_final_calibration_decision.py",
            "--ranked-json",
            str(ranked_json),
            "--output-md",
            str(report_md),
        ]
    )

    print(
        json.dumps(
            {
                "status": "completed",
                "run_root": str(run_root),
                "results_raw": str(raw_json),
                "results_ranked": str(ranked_json),
                "final_default_recipe": str(default_json),
                "final_report_md": str(report_md),
                "winner_lane": winner.get("lane"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
