#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from tools.context_feature_variants import (
    CANONICAL_BASE_VARIANT,
    COMBINED_VARIANT,
    IMGRAW_VARIANT,
    SCENE_SUMMARY_VARIANT,
    TRUSTED_CENTROID_VARIANT,
    compute_payload_feature_block_stats,
    infer_feature_schema_hash,
    load_npz_payload,
    payload_context_variant_id,
    save_npz_payload,
    subset_payload_by_images,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOISE_BAND = 0.001
BASELINE_METHODS = {"noimg", "imgraw", "legacy_imgraw_plus_probs"}
REQUIRED_SCHEMA_KEYS = {
    "feature_schema_hash",
    "context_variant_id",
    "parent_feature_npz",
    "parent_feature_schema_hash",
    "variant_config_json",
}


@dataclass(frozen=True)
class LaneConfig:
    lane_id: str
    dataset_variant: str
    method: str
    labeled_npz: Path
    prepass_jsonl: Path
    derivation_summary_json: Path | None = None


@dataclass(frozen=True)
class LaneRuntimeInfo:
    schema_hash: str
    context_variant_id: str
    derivation_stats: dict[str, Any]
    feature_overhead: int


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}", flush=True)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _run(cmd: Sequence[str]) -> None:
    _log("RUN " + " ".join(str(part) for part in cmd))
    subprocess.run(list(cmd), cwd=REPO_ROOT, text=True, check=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_lane_config(path: Path) -> list[LaneConfig]:
    raw = _load_json(path)
    items = raw["lanes"] if isinstance(raw, dict) and "lanes" in raw else raw
    if not isinstance(items, list):
        raise SystemExit("lane config must be a list or {\"lanes\": [...]} object")
    lanes: list[LaneConfig] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        lanes.append(
            LaneConfig(
                lane_id=str(item["lane_id"]),
                dataset_variant=str(item["dataset_variant"]),
                method=str(item["method"]),
                labeled_npz=Path(item["labeled_npz"]).resolve(),
                prepass_jsonl=Path(item["prepass_jsonl"]).resolve(),
                derivation_summary_json=Path(item["derivation_summary_json"]).resolve()
                if item.get("derivation_summary_json")
                else None,
            )
        )
    if not lanes:
        raise SystemExit("no lanes in lane config")
    return lanes


def _parse_seed_list(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    if not out:
        raise SystemExit("no seeds provided")
    return out


def _parse_views(raw: str) -> list[str]:
    values = [token.strip() for token in str(raw or "").split(",") if token.strip()]
    if not values:
        raise SystemExit("no views provided")
    for value in values:
        if value not in {"full", "intersection"}:
            raise SystemExit(f"invalid view: {value}")
    return values


def _default_hp_grid(mode: str) -> list[dict[str, Any]]:
    conservative = {
        "max_depth": 8,
        "n_estimators": 600,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3.0,
        "gamma": 0.2,
        "reg_lambda": 3.0,
        "reg_alpha": 0.1,
    }
    best_baseline = {
        "max_depth": 10,
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3.0,
        "gamma": 0.2,
        "reg_lambda": 3.0,
        "reg_alpha": 0.1,
    }
    best_imgraw = {
        "max_depth": 10,
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "min_child_weight": 1.0,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    }
    if mode == "pilot_1000":
        return [best_baseline, best_imgraw, conservative]
    return [best_baseline]


def _load_hp_grid(path_or_json: str, *, mode: str) -> list[dict[str, Any]]:
    raw = str(path_or_json or "").strip()
    if not raw:
        return _default_hp_grid(mode)
    path = Path(raw)
    text = path.read_text(encoding="utf-8") if path.exists() else raw
    parsed = json.loads(text)
    if not isinstance(parsed, list) or not parsed:
        raise SystemExit("hp grid must be a non-empty JSON list")
    out: list[dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict):
            out.append(item)
    if not out:
        raise SystemExit("hp grid contained no objects")
    return out


def _lane_images(path: Path) -> list[str]:
    payload = load_npz_payload(path)
    meta = payload.get("meta", [])
    images = []
    for row in meta:
        try:
            parsed = json.loads(str(row))
        except Exception:
            continue
        image = str(parsed.get("image") or "").strip()
        if image:
            images.append(image)
    return sorted(set(images))


def _select_images(
    lanes: Sequence[LaneConfig],
    *,
    view: str,
    pilot_max_images: int,
) -> dict[str, list[str]]:
    lane_images = {lane.lane_id: _lane_images(lane.labeled_npz) for lane in lanes}
    if view == "intersection":
        common: set[str] | None = None
        for values in lane_images.values():
            common = set(values) if common is None else common.intersection(values)
        selected = sorted(common or set())
        if pilot_max_images > 0:
            selected = selected[:pilot_max_images]
        return {lane_id: list(selected) for lane_id in lane_images}
    out: dict[str, list[str]] = {}
    for lane_id, values in lane_images.items():
        chosen = list(values)
        if pilot_max_images > 0:
            chosen = chosen[:pilot_max_images]
        out[lane_id] = chosen
    return out


def _build_val_split(images: Sequence[str], *, seed: int, val_ratio: float) -> list[str]:
    values = sorted({str(item).strip() for item in images if str(item).strip()})
    if not values:
        return []
    rng = random.Random(int(seed))
    values = list(values)
    rng.shuffle(values)
    val_count = max(1, int(round(len(values) * float(val_ratio))))
    if len(values) > 1:
        val_count = min(val_count, len(values) - 1)
    return sorted(values[:val_count])


def _write_subset_artifact(
    lane: LaneConfig,
    *,
    run_root: Path,
    images: Sequence[str],
) -> Path:
    payload = load_npz_payload(lane.labeled_npz)
    subset = subset_payload_by_images(payload, images)
    out_path = run_root / "subsets" / lane.lane_id / "labeled.npz"
    save_npz_payload(out_path, subset)
    return out_path


def _expected_context_variant_id(method: str) -> str | None:
    normalized = str(method or "").strip().lower()
    if normalized == "noimg":
        return CANONICAL_BASE_VARIANT
    if normalized in {
        IMGRAW_VARIANT,
        SCENE_SUMMARY_VARIANT,
        TRUSTED_CENTROID_VARIANT,
        COMBINED_VARIANT,
    }:
        return normalized
    return None


def _validate_summary_schema(lane: LaneConfig, *, expected_schema_hash: str) -> None:
    if lane.derivation_summary_json is None or (not lane.derivation_summary_json.exists()):
        return
    payload = _load_json(lane.derivation_summary_json)
    summary_hash = str(payload.get("feature_schema_hash") or "").strip()
    if summary_hash and summary_hash != str(expected_schema_hash):
        raise SystemExit(
            f"derivation_summary_schema_mismatch:{lane.lane_id}:{summary_hash}:{expected_schema_hash}"
        )


def _load_lane_runtime_info(lane: LaneConfig) -> LaneRuntimeInfo:
    payload = load_npz_payload(lane.labeled_npz)
    missing_keys = sorted(key for key in REQUIRED_SCHEMA_KEYS if key not in payload)
    if missing_keys:
        raise SystemExit(f"lane_missing_schema_metadata:{lane.lane_id}:{','.join(missing_keys)}")
    schema_hash = infer_feature_schema_hash(payload)
    stored_schema_hash = str(payload["feature_schema_hash"].item() if np.asarray(payload["feature_schema_hash"]).shape == () else payload["feature_schema_hash"])
    if stored_schema_hash != schema_hash:
        raise SystemExit(
            f"lane_feature_schema_hash_mismatch:{lane.lane_id}:{stored_schema_hash}:{schema_hash}"
        )
    context_variant_id = payload_context_variant_id(payload)
    expected_variant_id = _expected_context_variant_id(lane.method)
    if expected_variant_id and context_variant_id != expected_variant_id:
        raise SystemExit(
            f"lane_context_variant_mismatch:{lane.lane_id}:{context_variant_id}:{expected_variant_id}"
        )
    _validate_summary_schema(lane, expected_schema_hash=schema_hash)
    derivation_stats = (
        compute_payload_feature_block_stats(payload, expected_variant_id=expected_variant_id)
        if lane.method not in {"noimg", "legacy_imgraw_plus_probs"}
        else {
            "variant_type": "global-only",
            "new_feature_count": 0,
            "context_variant_id": context_variant_id,
        }
    )
    return LaneRuntimeInfo(
        schema_hash=schema_hash,
        context_variant_id=context_variant_id,
        derivation_stats=derivation_stats,
        feature_overhead=int(np.asarray(payload["X"]).shape[1]),
    )


def _extract_eval_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    post_xgb = payload.get("metric_tiers", {}).get("post_xgb", {}).get("accepted_all", {})
    post_cluster_union = (
        payload.get("metric_tiers", {})
        .get("post_cluster", {})
        .get("source_attributed", {})
        .get("yolo_rfdetr_union", {})
    )
    coverage = payload.get("coverage_upper_bound", {}).get("candidate_all", {})
    recall = _safe_float(post_xgb.get("recall"), _safe_float(payload.get("recall"), 0.0))
    coverage_upper_bound = _safe_float(coverage.get("recall_upper_bound"), 0.0)
    return {
        "precision": _safe_float(post_xgb.get("precision"), _safe_float(payload.get("precision"), 0.0)),
        "recall": recall,
        "f1": _safe_float(post_xgb.get("f1"), _safe_float(payload.get("f1"), 0.0)),
        "delta_vs_union_f1": _safe_float(post_xgb.get("f1"), _safe_float(payload.get("f1"), 0.0))
        - _safe_float(post_cluster_union.get("f1"), 0.0),
        "coverage_upper_bound": coverage_upper_bound,
        "coverage_preservation": (recall / coverage_upper_bound) if coverage_upper_bound > 0.0 else 0.0,
    }


def _feature_gain_share(model_path: Path, labeled_npz: Path, *, family_prefixes: Sequence[str]) -> float:
    import xgboost as xgb

    data = np.load(labeled_npz, allow_pickle=True)
    feature_names = [str(name) for name in data["feature_names"]]
    booster = xgb.Booster()
    booster.load_model(model_path)
    scores = booster.get_score(importance_type="gain")
    total = sum(float(value) for value in scores.values()) or 1.0
    used = 0.0
    for raw_name, value in scores.items():
        if not str(raw_name).startswith("f"):
            continue
        idx = int(str(raw_name)[1:])
        if idx >= len(feature_names):
            continue
        name = feature_names[idx]
        if any(name.startswith(prefix) for prefix in family_prefixes):
            used += float(value)
    return float(used / total)


def _train_tune_eval(
    *,
    dataset: str,
    lane: LaneConfig,
    labeled_npz: Path,
    prepass_jsonl: Path,
    out_dir: Path,
    hp: Mapping[str, Any],
    eval_iou: float,
    dedupe_iou: float,
    scoreless_iou: float,
    fixed_val_images: Path,
    threshold_steps: int,
    target_fp_ratio: float,
    min_recall: float,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = out_dir / "model"
    train_cmd = [
        sys.executable,
        "tools/train_ensemble_xgb.py",
        "--input",
        str(labeled_npz),
        "--output",
        str(model_prefix),
        "--seed",
        "42",
        "--val-ratio",
        "0.2",
        "--tree-method",
        "hist",
        "--max-bin",
        "256",
        "--early-stopping-rounds",
        "50",
        "--threshold-steps",
        str(int(threshold_steps)),
        "--optimize",
        "f1",
        "--target-fp-ratio",
        str(float(target_fp_ratio)),
        "--min-recall",
        str(float(min_recall)),
        "--per-class",
        "--fixed-val-images",
        str(fixed_val_images),
    ]
    for flag, value in hp.items():
        train_cmd.extend([f"--{flag.replace('_', '-')}", str(value)])
    _run(train_cmd)

    model_json = Path(f"{model_prefix}.json")
    model_meta = Path(f"{model_prefix}.meta.json")
    tune_cmd = [
        sys.executable,
        "tools/tune_ensemble_thresholds_xgb.py",
        "--model",
        str(model_json),
        "--meta",
        str(model_meta),
        "--data",
        str(labeled_npz),
        "--dataset",
        str(dataset),
        "--optimize",
        "f1",
        "--target-fp-ratio",
        str(float(target_fp_ratio)),
        "--relax-fp-ratio",
        str(float(target_fp_ratio)),
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
    _run(tune_cmd)

    eval_json = out_dir / "eval.json"
    eval_cmd = [
        sys.executable,
        "tools/eval_ensemble_xgb_dedupe.py",
        "--model",
        str(model_json),
        "--meta",
        str(model_meta),
        "--data",
        str(labeled_npz),
        "--dataset",
        str(dataset),
        "--prepass-jsonl",
        str(prepass_jsonl),
        "--eval-iou",
        str(float(eval_iou)),
        "--eval-iou-grid",
        str(float(eval_iou)),
        "--dedupe-iou",
        str(float(dedupe_iou)),
        "--scoreless-iou",
        str(float(scoreless_iou)),
        "--use-val-split",
    ]
    proc = subprocess.run(eval_cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=True)
    eval_json.write_text(proc.stdout, encoding="utf-8")
    eval_payload = json.loads(proc.stdout)
    metrics = _extract_eval_metrics(eval_payload)
    return {
        "metrics": metrics,
        "model_json": str(model_json),
        "model_meta": str(model_meta),
        "eval_json": str(eval_json),
    }


def _gate0_decision(stats: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    zero_fraction = _safe_float(stats.get("zero_fraction"), 0.0)
    varying_fraction = _safe_float(stats.get("varying_fraction"), 1.0)
    duplicate_fraction = _safe_float(stats.get("duplicate_fraction"), 0.0)
    if zero_fraction > 0.95:
        reasons.append("zero_fraction_exceeds_0.95")
    if varying_fraction < 0.20:
        reasons.append("varying_fraction_below_0.20")
    if duplicate_fraction > 0.90:
        reasons.append("duplicate_fraction_exceeds_0.90")
    variant_type = str(stats.get("variant_type") or "candidate-specific")
    return {
        "pass": not reasons,
        "variant_type": variant_type,
        "reasons": reasons,
        "stats": dict(stats),
    }


def _delta_band(delta: float) -> str:
    if -NOISE_BAND <= delta <= NOISE_BAND:
        return "noise"
    return "signal"


def _gate1_decision(method_type: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(row.get("delta_f1", 0.0)) for row in rows]
    coverage_drops = [float(row.get("coverage_drop", 0.0)) for row in rows]
    gain_shares = [float(row.get("feature_gain_share", 0.0)) for row in rows]
    reasons: list[str] = []
    if method_type == "global-only":
        if max(deltas or [0.0]) < 0.0035:
            reasons.append("best_delta_below_0.0035")
        if mean(deltas or [0.0]) < 0.0020:
            reasons.append("mean_delta_below_0.0020")
        if any(delta < -0.0005 for delta in deltas):
            reasons.append("variant_below_negative_global_floor")
        if any(drop > 0.003 for drop in coverage_drops):
            reasons.append("coverage_drop_exceeds_0.003")
    else:
        if max(deltas or [0.0]) < 0.0025:
            reasons.append("best_delta_below_0.0025")
        if mean(deltas or [0.0]) < 0.0015:
            reasons.append("mean_delta_below_0.0015")
        if any(delta < -0.0010 for delta in deltas):
            reasons.append("variant_below_negative_floor")
        if any(drop > 0.005 for drop in coverage_drops):
            reasons.append("coverage_drop_exceeds_0.005")
        if mean(gain_shares or [0.0]) < 0.03:
            reasons.append("feature_gain_share_below_0.03")
    if deltas and max(abs(delta) for delta in deltas) <= NOISE_BAND:
        reasons.append("noise_band")
    return {"pass": not reasons, "reasons": reasons, "rows": list(rows)}


def _gate2_decision(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(row.get("delta_f1", 0.0)) for row in rows]
    coverage_drops = [float(row.get("coverage_drop", 0.0)) for row in rows]
    gain_shares = [float(row.get("feature_gain_share", 0.0)) for row in rows]
    reasons: list[str] = []
    if mean(deltas or [0.0]) <= 0.0:
        reasons.append("mean_delta_non_positive")
    if any(delta < -0.0015 for delta in deltas):
        reasons.append("variant_below_negative_floor")
    if any(drop > 0.007 for drop in coverage_drops):
        reasons.append("coverage_drop_exceeds_0.007")
    if mean(gain_shares or [0.0]) < 0.05:
        reasons.append("feature_gain_share_below_0.05")
    return {"pass": not reasons, "reasons": reasons, "rows": list(rows)}


def _method_gate_type(rows: Sequence[Mapping[str, Any]]) -> str:
    return (
        "global-only"
        if any(str(row.get("variant_type") or "") == "global-only" for row in rows)
        else "candidate-specific"
    )


def _rank_method_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    ranked: list[dict[str, Any]] = []
    for method, items in grouped.items():
        ranked.append(
            {
                "method": method,
                "mean_f1": mean(float(item["f1"]) for item in items),
                "mean_delta_vs_union_f1": mean(float(item["delta_vs_union_f1"]) for item in items),
                "mean_coverage_preservation": mean(float(item["coverage_preservation"]) for item in items),
                "mean_delta_f1": mean(float(item["delta_f1"]) for item in items),
                "feature_overhead": max(int(item.get("feature_overhead", 0)) for item in items),
            }
        )
    ranked.sort(
        key=lambda row: (
            -row["mean_f1"],
            -row["mean_delta_vs_union_f1"],
            -row["mean_coverage_preservation"],
            row["feature_overhead"],
        )
    )
    return ranked


def _build_markdown_report(
    *,
    mode: str,
    rows: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
) -> str:
    lines = [
        f"# Context Feature Ablation Report ({mode})",
        "",
        "| Method | Variant | Precision | Recall | F1 | Delta vs Baseline | Coverage Preservation | Coverage Drop | Gain Share |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {variant} | {p:.4f} | {r:.4f} | {f1:.4f} | {df1:+.4f} | {cov:.4f} | {drop:+.4f} | {gain:.4f} |".format(
                method=row["method"],
                variant=row["dataset_variant"],
                p=float(row["precision"]),
                r=float(row["recall"]),
                f1=float(row["f1"]),
                df1=float(row["delta_f1"]),
                cov=float(row["coverage_preservation"]),
                drop=float(row["coverage_drop"]),
                gain=float(row["feature_gain_share"]),
            )
        )
    lines.append("")
    for method, decision in decisions.items():
        lines.append(f"## {method}")
        lines.append("")
        lines.append(f"- pass: `{bool(decision.get('pass'))}`")
        lines.append(f"- reasons: `{','.join(decision.get('reasons', [])) or 'none'}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fast-fail context feature ablations over existing labeled artifacts.")
    parser.add_argument("--mode", required=True, choices=["sanity", "pilot_1000", "full_4000", "final_validate"])
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--lane-config-json", required=True)
    parser.add_argument("--dataset", default="qwen_dataset")
    parser.add_argument("--eval-iou", type=float, default=0.5)
    parser.add_argument("--dedupe-iou", type=float, default=0.75)
    parser.add_argument("--scoreless-iou", type=float, default=0.0)
    parser.add_argument("--target-fp-ratio", type=float, default=0.2)
    parser.add_argument("--min-recall", type=float, default=0.6)
    parser.add_argument("--threshold-steps", type=int, default=300)
    parser.add_argument("--pilot-max-images", type=int, default=1000)
    parser.add_argument("--views", default="")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--fixed-val-images", default="")
    parser.add_argument("--hp-grid-json", default="")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    lanes = _load_lane_config(Path(args.lane_config_json).resolve())
    lane_runtime: dict[str, LaneRuntimeInfo] = {
        lane.lane_id: _load_lane_runtime_info(lane) for lane in lanes
    }

    if args.mode == "sanity":
        payload = {"generated_utc": _ts(), "mode": args.mode, "decisions": {}}
        for lane in lanes:
            if lane.method in BASELINE_METHODS:
                continue
            stats = lane_runtime[lane.lane_id].derivation_stats
            payload["decisions"][lane.lane_id] = _gate0_decision(stats)
        _write_json(run_root / "results_raw.json", payload)
        _write_json(run_root / "results_ranked.json", payload)
        (run_root / "final_report.md").write_text(
            _build_markdown_report(mode=args.mode, rows=[], decisions=payload["decisions"]),
            encoding="utf-8",
        )
        return

    hp_grid = _load_hp_grid(args.hp_grid_json, mode=args.mode)
    views = _parse_views(args.views or ("intersection" if args.mode == "pilot_1000" else "full"))
    seeds = _parse_seed_list(args.seeds)
    fixed_val_all = []
    if args.fixed_val_images:
        fixed_val_all = json.loads(Path(args.fixed_val_images).read_text(encoding="utf-8"))
        fixed_val_all = [str(item).strip() for item in fixed_val_all if str(item).strip()]

    all_rows: list[dict[str, Any]] = []
    for view in views:
        selected_images_by_lane = _select_images(
            lanes,
            view=view,
            pilot_max_images=args.pilot_max_images if args.mode == "pilot_1000" else 0,
        )
        for seed in seeds:
            for lane in lanes:
                selected_images = selected_images_by_lane[lane.lane_id]
                subset_npz = _write_subset_artifact(lane, run_root=run_root / view / f"seed_{seed}", images=selected_images)
                val_images = sorted(set(selected_images).intersection(fixed_val_all)) if fixed_val_all else _build_val_split(
                    selected_images,
                    seed=int(seed),
                    val_ratio=float(args.val_ratio),
                )
                val_file = run_root / view / f"seed_{seed}" / lane.lane_id / "val_images.json"
                _write_json(val_file, val_images)
                for hp_idx, hp in enumerate(hp_grid):
                    out_dir = run_root / view / f"seed_{seed}" / lane.lane_id / f"hp_{hp_idx:02d}"
                    result = _train_tune_eval(
                        dataset=str(args.dataset),
                        lane=lane,
                        labeled_npz=subset_npz,
                        prepass_jsonl=lane.prepass_jsonl,
                        out_dir=out_dir,
                        hp=hp,
                        eval_iou=float(args.eval_iou),
                        dedupe_iou=float(args.dedupe_iou),
                        scoreless_iou=float(args.scoreless_iou),
                        fixed_val_images=val_file,
                        threshold_steps=int(args.threshold_steps),
                        target_fp_ratio=float(args.target_fp_ratio),
                        min_recall=float(args.min_recall),
                    )
                    row = {
                        "lane_id": lane.lane_id,
                        "dataset_variant": lane.dataset_variant,
                        "method": lane.method,
                        "view": view,
                        "seed": int(seed),
                        "hp_index": hp_idx,
                        **result["metrics"],
                        "model_json": result["model_json"],
                        "model_meta": result["model_meta"],
                        "eval_json": result["eval_json"],
                    }
                    if lane.method not in BASELINE_METHODS:
                        prefixes = ["imgctx_scene_", "imgctx_trusted_"]
                        row["feature_gain_share"] = _feature_gain_share(
                            Path(result["model_json"]),
                            subset_npz,
                            family_prefixes=prefixes,
                        )
                    else:
                        row["feature_gain_share"] = 0.0
                    all_rows.append(row)

    baselines: dict[tuple[str, str, int], float] = {}
    for row in all_rows:
        if row["method"] not in BASELINE_METHODS:
            continue
        key = (row["dataset_variant"], row["view"], int(row["seed"]))
        baselines[key] = max(float(row["f1"]), baselines.get(key, float("-inf")))

    for row in all_rows:
        key = (row["dataset_variant"], row["view"], int(row["seed"]))
        baseline_f1 = baselines.get(key, float(row["f1"]))
        baseline_cov = max(
            (
                float(candidate["coverage_preservation"])
                for candidate in all_rows
                if candidate["method"] in BASELINE_METHODS
                and candidate["dataset_variant"] == row["dataset_variant"]
                and candidate["view"] == row["view"]
                and int(candidate["seed"]) == int(row["seed"])
            ),
            default=float(row["coverage_preservation"]),
        )
        row["delta_f1"] = float(row["f1"]) - float(baseline_f1)
        row["coverage_drop"] = max(0.0, float(baseline_cov) - float(row["coverage_preservation"]))
        row["delta_band"] = _delta_band(float(row["delta_f1"]))
        row["feature_overhead"] = int(lane_runtime[row["lane_id"]].feature_overhead)
        row["variant_type"] = str(
            lane_runtime[row["lane_id"]].derivation_stats.get("variant_type") or "candidate-specific"
        )

    decisions: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    for method, rows in grouped.items():
        if method in BASELINE_METHODS:
            continue
        best_by_variant: dict[str, dict[str, Any]] = {}
        for row in rows:
            variant = str(row["dataset_variant"])
            current = best_by_variant.get(variant)
            if current is None or float(row["f1"]) > float(current["f1"]):
                best_by_variant[variant] = row
        selected_rows = list(best_by_variant.values())
        method_type = _method_gate_type(selected_rows)
        if args.mode == "pilot_1000":
            decisions[method] = _gate1_decision(method_type, selected_rows)
        else:
            decisions[method] = _gate2_decision(selected_rows)

    raw_payload = {
        "generated_utc": _ts(),
        "mode": args.mode,
        "rows": all_rows,
        "ranked_methods": _rank_method_rows([row for row in all_rows if row["method"] not in BASELINE_METHODS]),
        "decisions": decisions,
    }
    _write_json(run_root / "results_raw.json", raw_payload)
    _write_json(run_root / "results_ranked.json", {"ranked_methods": raw_payload["ranked_methods"], "decisions": decisions})
    (run_root / "final_report.md").write_text(
        _build_markdown_report(mode=args.mode, rows=all_rows, decisions=decisions),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
