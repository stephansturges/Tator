#!/usr/bin/env python3
"""Run full-stack prepass/calibration follow-up and leave-one-out ablations.

This suite reuses existing 4000-image labeled feature matrices and compares:
- full_stack: source-aware policy + global threshold-shift tuning + split-head quality modeling
- no_source_policy: disable source-aware policy, keep global threshold-shift tuning
- no_joint_tune: keep source-aware policy, disable global threshold-shift tuning
- no_split_head_quality: disable split-head + sam3-text quality head, keep full stack tuning

All evaluations are apples-to-apples on fixed validation images, IoU=0.5.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class VariantSpec:
    name: str
    dataset: str
    labeled_npz: Path
    prepass_jsonl: Path
    baseline_eval_json: Path
    fixed_val_meta_json: Path


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _run(cmd: List[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    _log("RUN " + " ".join(cmd))
    return subprocess.run(
        cmd,
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=capture,
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


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


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_val_images(meta_path: Path, out_path: Path) -> List[str]:
    meta = _load_json(meta_path)
    val_images = [str(x) for x in (meta.get("split_val_images") or []) if str(x)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sorted(set(val_images)), indent=2), encoding="utf-8")
    return sorted(set(val_images))


def _extract_metrics(payload: Dict[str, Any], *, gate_margin: float) -> Dict[str, Any]:
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
    coverage_obj = payload.get("coverage_upper_bound", {}).get("candidate_all", {})
    coverage_ub = _safe_float(coverage_obj.get("recall_upper_bound"), 0.0)

    p = _safe_float(post_xgb.get("precision"), _safe_float(payload.get("precision"), 0.0))
    r = _safe_float(post_xgb.get("recall"), _safe_float(payload.get("recall"), 0.0))
    f1 = _safe_float(post_xgb.get("f1"), _safe_float(payload.get("f1"), 0.0))

    comparator_f1 = _safe_float(post_cluster_union.get("f1"), 0.0)
    delta_vs_union = f1 - comparator_f1
    coverage_preservation = (r / coverage_ub) if coverage_ub > 0 else 0.0

    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "comparator_union_f1": comparator_f1,
        "delta_vs_union_f1": delta_vs_union,
        "coverage_upper_bound": coverage_ub,
        "coverage_preservation": coverage_preservation,
        "gate_margin": float(gate_margin),
        "gate_pass": bool(delta_vs_union >= float(gate_margin)),
    }


class Evaluator:
    def __init__(
        self,
        *,
        variant: VariantSpec,
        model_json: Path,
        model_meta: Path,
        eval_iou: float,
        dedupe_iou: float,
        scoreless_iou: float,
        run_dir: Path,
    ) -> None:
        self.variant = variant
        self.model_json = model_json
        self.model_meta = model_meta
        self.eval_iou = float(eval_iou)
        self.dedupe_iou = float(dedupe_iou)
        self.scoreless_iou = float(scoreless_iou)
        self.run_dir = run_dir
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._counter = 0

    def _policy_key(self, policy: Dict[str, Any]) -> str:
        return json.dumps(policy, sort_keys=True, separators=(",", ":"))

    def evaluate(self, *, policy: Dict[str, Any], tag: str) -> Dict[str, Any]:
        key = self._policy_key(policy)
        if key in self._cache:
            return self._cache[key]

        policy_path: Optional[Path] = None
        if policy:
            policy_path = self.run_dir / "policies" / f"{tag}_{self._counter:04d}.json"
            self._counter += 1
            _write_json(policy_path, policy)

        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "eval_ensemble_xgb_dedupe.py"),
            "--model",
            str(self.model_json),
            "--meta",
            str(self.model_meta),
            "--data",
            str(self.variant.labeled_npz),
            "--dataset",
            self.variant.dataset,
            "--prepass-jsonl",
            str(self.variant.prepass_jsonl),
            "--eval-iou",
            str(self.eval_iou),
            "--eval-iou-grid",
            str(self.eval_iou),
            "--dedupe-iou",
            str(self.dedupe_iou),
            "--scoreless-iou",
            str(self.scoreless_iou),
            "--use-val-split",
        ]
        if policy_path is not None:
            cmd += ["--policy-json", str(policy_path)]

        result = _run(cmd, capture=True)
        payload = json.loads(result.stdout)
        self._cache[key] = payload
        return payload


def _build_policy(
    *,
    base_thresholds: Dict[str, float],
    threshold_logit_shift: float,
    source_aware: bool,
    bias_sam3_text: float,
    bias_sam3_similarity: float,
    sam_floor: float,
    consensus_iou_sam3_text: float,
    consensus_iou_sam3_similarity: float,
    bias_sam3_text_by_class: Optional[Dict[str, float]] = None,
    bias_sam3_similarity_by_class: Optional[Dict[str, float]] = None,
    sam_floor_by_class: Optional[Dict[str, float]] = None,
    consensus_iou_sam3_text_by_class: Optional[Dict[str, float]] = None,
    consensus_iou_sam3_similarity_by_class: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    policy: Dict[str, Any] = {}

    # Lever 2: global threshold shift (absolute per-class override derived from base thresholds).
    if abs(threshold_logit_shift) > 1e-12:
        overrides: Dict[str, float] = {}
        for label, threshold in base_thresholds.items():
            overrides[label] = _sigmoid(_logit(threshold) + float(threshold_logit_shift))
        policy["threshold_by_class_override"] = overrides

    # Lever 1: source-aware acceptance policy.
    if source_aware:
        text_map: Dict[str, float] = {"__default__": float(bias_sam3_text)}
        sim_map: Dict[str, float] = {"__default__": float(bias_sam3_similarity)}
        for key, value in (bias_sam3_text_by_class or {}).items():
            label = str(key or "").strip().lower()
            if label:
                text_map[label] = float(value)
        for key, value in (bias_sam3_similarity_by_class or {}).items():
            label = str(key or "").strip().lower()
            if label:
                sim_map[label] = float(value)
        policy["logit_bias_by_source_class"] = {
            "sam3_text": text_map,
            "sam3_similarity": sim_map,
        }
        policy["sam_only_min_prob_default"] = float(sam_floor)
        if sam_floor_by_class:
            floor_map: Dict[str, float] = {}
            for key, value in sam_floor_by_class.items():
                label = str(key or "").strip().lower()
                if label:
                    floor_map[label] = float(value)
            if floor_map:
                policy["sam_only_min_prob_by_class"] = floor_map
        text_consensus_map: Dict[str, float] = {"__default__": float(consensus_iou_sam3_text)}
        sim_consensus_map: Dict[str, float] = {"__default__": float(consensus_iou_sam3_similarity)}
        for key, value in (consensus_iou_sam3_text_by_class or {}).items():
            label = str(key or "").strip().lower()
            if label:
                text_consensus_map[label] = float(value)
        for key, value in (consensus_iou_sam3_similarity_by_class or {}).items():
            label = str(key or "").strip().lower()
            if label:
                sim_consensus_map[label] = float(value)
        policy["consensus_iou_by_source_class"] = {
            "sam3_text": text_consensus_map,
            "sam3_similarity": sim_consensus_map,
        }
        # Keep a legacy fallback field for older consumers that only read class/default keys.
        policy["consensus_iou_default"] = float(
            min(consensus_iou_sam3_text, consensus_iou_sam3_similarity)
        )
        policy["consensus_class_aware"] = True

    return policy


def _policy_param_search(
    evaluator: Evaluator,
    *,
    base_thresholds: Dict[str, float],
    threshold_logit_shift: float,
    source_aware: bool,
    initial: Dict[str, float],
    search_grids: Dict[str, List[float]],
    tag_prefix: str,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    state = dict(initial)

    def _eval_state(local_state: Dict[str, float], local_tag: str) -> Dict[str, Any]:
        policy = _build_policy(
            base_thresholds=base_thresholds,
            threshold_logit_shift=threshold_logit_shift,
            source_aware=source_aware,
            bias_sam3_text=float(local_state.get("bias_sam3_text", 0.0)),
            bias_sam3_similarity=float(local_state.get("bias_sam3_similarity", 0.0)),
            sam_floor=float(local_state.get("sam_floor", 0.0)),
            consensus_iou_sam3_text=float(local_state.get("consensus_iou_sam3_text", 0.0)),
            consensus_iou_sam3_similarity=float(
                local_state.get("consensus_iou_sam3_similarity", 0.0)
            ),
        )
        return evaluator.evaluate(policy=policy, tag=local_tag)

    best_payload = _eval_state(state, f"{tag_prefix}_init")
    best_f1 = _safe_float(best_payload.get("f1"), 0.0)

    for param in (
        "bias_sam3_text",
        "bias_sam3_similarity",
        "sam_floor",
        "consensus_iou_sam3_text",
        "consensus_iou_sam3_similarity",
    ):
        grid = list(search_grids.get(param) or [])
        local_best_state = dict(state)
        local_best_payload = best_payload
        local_best_f1 = best_f1
        for value in grid:
            if abs(float(value) - float(state.get(param, 0.0))) <= 1e-12:
                continue
            candidate = dict(state)
            candidate[param] = float(value)
            payload = _eval_state(candidate, f"{tag_prefix}_{param}_{value:+.3f}")
            f1 = _safe_float(payload.get("f1"), 0.0)
            if f1 > local_best_f1 + 1e-12:
                local_best_f1 = f1
                local_best_state = candidate
                local_best_payload = payload
        state = local_best_state
        best_payload = local_best_payload
        best_f1 = local_best_f1

    return state, best_payload


def _label_rank_from_payload(
    payload: Dict[str, Any],
    *,
    fallback_labels: Sequence[str],
    limit: int,
) -> List[str]:
    labels: List[Tuple[str, float]] = []
    by_label = (
        payload.get("metric_tiers", {})
        .get("post_xgb", {})
        .get("accepted_all", {})
        .get("by_label", {})
    )
    if isinstance(by_label, dict):
        for key, metrics in by_label.items():
            label = str(key or "").strip().lower()
            if not label:
                continue
            if not isinstance(metrics, dict):
                continue
            support = float(_safe_float(metrics.get("tp"), 0.0) + _safe_float(metrics.get("fn"), 0.0))
            labels.append((label, support))
    if not labels:
        labels = [(str(lbl).strip().lower(), 0.0) for lbl in fallback_labels if str(lbl).strip()]
    labels = sorted(labels, key=lambda item: (-item[1], item[0]))
    unique: List[str] = []
    seen = set()
    for label, _ in labels:
        if label in seen:
            continue
        seen.add(label)
        unique.append(label)
    if limit > 0:
        unique = unique[:limit]
    return unique


def _policy_param_search_per_class(
    evaluator: Evaluator,
    *,
    base_thresholds: Dict[str, float],
    threshold_logit_shift: float,
    global_params: Dict[str, float],
    labels: Sequence[str],
    tag_prefix: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    per_class_maps: Dict[str, Dict[str, float]] = {
        "bias_sam3_text_by_class": {},
        "bias_sam3_similarity_by_class": {},
        "sam_floor_by_class": {},
        "consensus_iou_sam3_text_by_class": {},
        "consensus_iou_sam3_similarity_by_class": {},
    }

    def _evaluate(
        *,
        local_maps: Dict[str, Dict[str, float]],
        local_tag: str,
    ) -> Dict[str, Any]:
        policy = _build_policy(
            base_thresholds=base_thresholds,
            threshold_logit_shift=threshold_logit_shift,
            source_aware=True,
            bias_sam3_text=float(global_params.get("bias_sam3_text", 0.0)),
            bias_sam3_similarity=float(global_params.get("bias_sam3_similarity", 0.0)),
            sam_floor=float(global_params.get("sam_floor", 0.0)),
            consensus_iou_sam3_text=float(global_params.get("consensus_iou_sam3_text", 0.0)),
            consensus_iou_sam3_similarity=float(
                global_params.get("consensus_iou_sam3_similarity", 0.0)
            ),
            bias_sam3_text_by_class=local_maps.get("bias_sam3_text_by_class"),
            bias_sam3_similarity_by_class=local_maps.get("bias_sam3_similarity_by_class"),
            sam_floor_by_class=local_maps.get("sam_floor_by_class"),
            consensus_iou_sam3_text_by_class=local_maps.get("consensus_iou_sam3_text_by_class"),
            consensus_iou_sam3_similarity_by_class=local_maps.get(
                "consensus_iou_sam3_similarity_by_class"
            ),
        )
        return evaluator.evaluate(policy=policy, tag=local_tag)

    current_maps = {
        "bias_sam3_text_by_class": {},
        "bias_sam3_similarity_by_class": {},
        "sam_floor_by_class": {},
        "consensus_iou_sam3_text_by_class": {},
        "consensus_iou_sam3_similarity_by_class": {},
    }
    best_payload = _evaluate(local_maps=current_maps, local_tag=f"{tag_prefix}_perclass_init")
    best_f1 = _safe_float(best_payload.get("f1"), 0.0)

    global_text = float(global_params.get("bias_sam3_text", 0.0))
    global_sim = float(global_params.get("bias_sam3_similarity", 0.0))
    global_floor = float(global_params.get("sam_floor", 0.0))
    global_consensus_text = float(global_params.get("consensus_iou_sam3_text", 0.0))
    global_consensus_sim = float(global_params.get("consensus_iou_sam3_similarity", 0.0))

    def _unique(values: Sequence[float]) -> List[float]:
        seen = set()
        out: List[float] = []
        for value in values:
            key = round(float(value), 6)
            if key in seen:
                continue
            seen.add(key)
            out.append(float(value))
        return out

    for label in labels:
        label = str(label or "").strip().lower()
        if not label:
            continue
        label_updates = [
            (
                "bias_sam3_text_by_class",
                _unique(
                    [
                        global_text - 0.4,
                        global_text - 0.2,
                        global_text,
                        global_text + 0.2,
                    ]
                ),
                -1.2,
                0.6,
            ),
            (
                "bias_sam3_similarity_by_class",
                _unique(
                    [
                        global_sim - 0.4,
                        global_sim - 0.2,
                        global_sim,
                        global_sim + 0.2,
                    ]
                ),
                -1.2,
                0.6,
            ),
            (
                "sam_floor_by_class",
                _unique([global_floor - 0.05, global_floor, global_floor + 0.05, global_floor + 0.1]),
                0.0,
                0.4,
            ),
            (
                "consensus_iou_sam3_text_by_class",
                _unique([global_consensus_text, 0.0, 0.3, 0.5, 0.7]),
                0.0,
                0.9,
            ),
            (
                "consensus_iou_sam3_similarity_by_class",
                _unique([global_consensus_sim, 0.0, 0.3, 0.5, 0.7]),
                0.0,
                0.9,
            ),
        ]
        for map_name, raw_grid, min_value, max_value in label_updates:
            grid = [min(max(float(v), min_value), max_value) for v in raw_grid]
            local_best_maps = current_maps
            local_best_payload = best_payload
            local_best_f1 = best_f1
            for value in grid:
                candidate_maps = {
                    key: dict(value_map)
                    for key, value_map in current_maps.items()
                }
                candidate_maps.setdefault(map_name, {})
                candidate_maps[map_name][label] = float(value)
                payload = _evaluate(
                    local_maps=candidate_maps,
                    local_tag=f"{tag_prefix}_{label}_{map_name}_{value:+.3f}",
                )
                f1 = _safe_float(payload.get("f1"), 0.0)
                if f1 > local_best_f1 + 1e-12:
                    local_best_f1 = f1
                    local_best_maps = candidate_maps
                    local_best_payload = payload
            current_maps = local_best_maps
            best_payload = local_best_payload
            best_f1 = local_best_f1

    # Drop redundant overrides equal to global defaults.
    for label, value in list(current_maps["bias_sam3_text_by_class"].items()):
        if abs(float(value) - global_text) <= 1e-12:
            current_maps["bias_sam3_text_by_class"].pop(label, None)
    for label, value in list(current_maps["bias_sam3_similarity_by_class"].items()):
        if abs(float(value) - global_sim) <= 1e-12:
            current_maps["bias_sam3_similarity_by_class"].pop(label, None)
    for label, value in list(current_maps["sam_floor_by_class"].items()):
        if abs(float(value) - global_floor) <= 1e-12:
            current_maps["sam_floor_by_class"].pop(label, None)
    for label, value in list(current_maps["consensus_iou_sam3_text_by_class"].items()):
        if abs(float(value) - global_consensus_text) <= 1e-12:
            current_maps["consensus_iou_sam3_text_by_class"].pop(label, None)
    for label, value in list(current_maps["consensus_iou_sam3_similarity_by_class"].items()):
        if abs(float(value) - global_consensus_sim) <= 1e-12:
            current_maps["consensus_iou_sam3_similarity_by_class"].pop(label, None)

    return current_maps, best_payload


def _threshold_shift_search(
    evaluator: Evaluator,
    *,
    base_thresholds: Dict[str, float],
    shifts: Sequence[float],
    source_aware: bool,
    policy_params: Dict[str, float],
    per_class_maps: Optional[Dict[str, Dict[str, float]]],
    tag_prefix: str,
) -> Tuple[float, Dict[str, Any]]:
    best_shift = 0.0
    best_payload: Optional[Dict[str, Any]] = None
    best_f1 = -1.0
    for shift in shifts:
        policy = _build_policy(
            base_thresholds=base_thresholds,
            threshold_logit_shift=float(shift),
            source_aware=source_aware,
            bias_sam3_text=float(policy_params.get("bias_sam3_text", 0.0)),
            bias_sam3_similarity=float(policy_params.get("bias_sam3_similarity", 0.0)),
            sam_floor=float(policy_params.get("sam_floor", 0.0)),
            consensus_iou_sam3_text=float(policy_params.get("consensus_iou_sam3_text", 0.0)),
            consensus_iou_sam3_similarity=float(
                policy_params.get("consensus_iou_sam3_similarity", 0.0)
            ),
            bias_sam3_text_by_class=(per_class_maps or {}).get("bias_sam3_text_by_class"),
            bias_sam3_similarity_by_class=(per_class_maps or {}).get("bias_sam3_similarity_by_class"),
            sam_floor_by_class=(per_class_maps or {}).get("sam_floor_by_class"),
            consensus_iou_sam3_text_by_class=(per_class_maps or {}).get(
                "consensus_iou_sam3_text_by_class"
            ),
            consensus_iou_sam3_similarity_by_class=(per_class_maps or {}).get(
                "consensus_iou_sam3_similarity_by_class"
            ),
        )
        payload = evaluator.evaluate(policy=policy, tag=f"{tag_prefix}_shift_{shift:+.3f}")
        f1 = _safe_float(payload.get("f1"), 0.0)
        if f1 > best_f1 + 1e-12:
            best_f1 = f1
            best_shift = float(shift)
            best_payload = payload
    if best_payload is None:
        best_payload = evaluator.evaluate(policy={}, tag=f"{tag_prefix}_shift_fallback")
    return best_shift, best_payload


def _train_and_tune_xgb(
    *,
    variant: VariantSpec,
    out_prefix: Path,
    fixed_val_images_path: Path,
    target_fp_ratio: float,
    min_recall: float,
    relax_fp_ratio: float,
    steps: int,
    optimize: str,
    split_head_quality: bool,
    sam3_text_quality_alpha: float,
) -> Tuple[Path, Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "train_ensemble_xgb.py"),
        "--input",
        str(variant.labeled_npz),
        "--output",
        str(out_prefix),
        "--seed",
        "42",
        "--target-fp-ratio",
        str(target_fp_ratio),
        "--min-recall",
        str(min_recall),
        "--threshold-steps",
        str(int(steps)),
        "--optimize",
        str(optimize),
        "--per-class",
        "--fixed-val-images",
        str(fixed_val_images_path),
    ]
    if split_head_quality:
        train_cmd.append("--split-head-by-support")
        train_cmd.append("--train-sam3-text-quality")
        train_cmd += ["--sam3-text-quality-alpha", str(float(sam3_text_quality_alpha))]
    _run(train_cmd)

    model_json = Path(str(out_prefix) + ".json")
    meta_json = Path(str(out_prefix) + ".meta.json")

    relax_cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "relax_ensemble_thresholds_xgb.py"),
        "--model",
        str(model_json),
        "--data",
        str(variant.labeled_npz),
        "--meta",
        str(meta_json),
        "--fp-ratio-cap",
        str(relax_fp_ratio),
    ]
    _run(relax_cmd)

    tune_cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "tune_ensemble_thresholds_xgb.py"),
        "--model",
        str(model_json),
        "--meta",
        str(meta_json),
        "--data",
        str(variant.labeled_npz),
        "--dataset",
        str(variant.dataset),
        "--optimize",
        str(optimize),
        "--target-fp-ratio",
        str(target_fp_ratio),
        "--min-recall",
        str(min_recall),
        "--steps",
        str(int(steps)),
        "--eval-iou",
        "0.5",
        "--dedupe-iou",
        "0.75",
        "--scoreless-iou",
        "0.0",
        "--use-val-split",
    ]
    _run(tune_cmd)

    return model_json, meta_json


def _scenario_run(
    *,
    evaluator: Evaluator,
    base_thresholds: Dict[str, float],
    use_source_policy: bool,
    use_joint_tune: bool,
    per_class_label_limit: int,
    gate_margin: float,
    scenario_tag: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    initial_params = {
        "bias_sam3_text": 0.0,
        "bias_sam3_similarity": 0.0,
        "sam_floor": 0.0,
        "consensus_iou_sam3_text": 0.0,
        "consensus_iou_sam3_similarity": 0.0,
    }
    grids = {
        "bias_sam3_text": [-0.8, -0.4, 0.0],
        "bias_sam3_similarity": [-0.8, -0.4, 0.0],
        "sam_floor": [0.0, 0.1, 0.2],
        "consensus_iou_sam3_text": [0.0, 0.3, 0.5, 0.7],
        "consensus_iou_sam3_similarity": [0.0, 0.3, 0.5, 0.7],
    }
    shifts = [-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4]

    if use_source_policy:
        policy_params, payload_after_policy = _policy_param_search(
            evaluator,
            base_thresholds=base_thresholds,
            threshold_logit_shift=0.0,
            source_aware=True,
            initial=initial_params,
            search_grids=grids,
            tag_prefix=f"{scenario_tag}_policy",
        )
        ranked_labels = _label_rank_from_payload(
            payload_after_policy,
            fallback_labels=sorted(base_thresholds.keys()),
            limit=(int(per_class_label_limit) if int(per_class_label_limit) > 0 else len(base_thresholds)),
        )
        per_class_maps, payload_after_per_class = _policy_param_search_per_class(
            evaluator,
            base_thresholds=base_thresholds,
            threshold_logit_shift=0.0,
            global_params=policy_params,
            labels=ranked_labels,
            tag_prefix=f"{scenario_tag}_perclass",
        )
    else:
        policy_params = dict(initial_params)
        payload_after_policy = evaluator.evaluate(policy={}, tag=f"{scenario_tag}_nopolicy_init")
        payload_after_per_class = payload_after_policy
        ranked_labels = []
        per_class_maps = {
            "bias_sam3_text_by_class": {},
            "bias_sam3_similarity_by_class": {},
            "sam_floor_by_class": {},
            "consensus_iou_sam3_text_by_class": {},
            "consensus_iou_sam3_similarity_by_class": {},
        }

    if use_joint_tune:
        best_shift, payload_after_shift = _threshold_shift_search(
            evaluator,
            base_thresholds=base_thresholds,
            shifts=shifts,
            source_aware=use_source_policy,
            policy_params=policy_params,
            per_class_maps=per_class_maps,
            tag_prefix=f"{scenario_tag}_joint",
        )
    else:
        best_shift = 0.0
        policy_no_shift = _build_policy(
            base_thresholds=base_thresholds,
            threshold_logit_shift=0.0,
            source_aware=use_source_policy,
            bias_sam3_text=float(policy_params.get("bias_sam3_text", 0.0)),
            bias_sam3_similarity=float(policy_params.get("bias_sam3_similarity", 0.0)),
            sam_floor=float(policy_params.get("sam_floor", 0.0)),
            consensus_iou_sam3_text=float(policy_params.get("consensus_iou_sam3_text", 0.0)),
            consensus_iou_sam3_similarity=float(
                policy_params.get("consensus_iou_sam3_similarity", 0.0)
            ),
            bias_sam3_text_by_class=per_class_maps.get("bias_sam3_text_by_class"),
            bias_sam3_similarity_by_class=per_class_maps.get("bias_sam3_similarity_by_class"),
            sam_floor_by_class=per_class_maps.get("sam_floor_by_class"),
            consensus_iou_sam3_text_by_class=per_class_maps.get("consensus_iou_sam3_text_by_class"),
            consensus_iou_sam3_similarity_by_class=per_class_maps.get(
                "consensus_iou_sam3_similarity_by_class"
            ),
        )
        payload_after_shift = evaluator.evaluate(policy=policy_no_shift, tag=f"{scenario_tag}_nojoint")

    final_policy = _build_policy(
        base_thresholds=base_thresholds,
        threshold_logit_shift=best_shift,
        source_aware=use_source_policy,
        bias_sam3_text=float(policy_params.get("bias_sam3_text", 0.0)),
        bias_sam3_similarity=float(policy_params.get("bias_sam3_similarity", 0.0)),
        sam_floor=float(policy_params.get("sam_floor", 0.0)),
        consensus_iou_sam3_text=float(policy_params.get("consensus_iou_sam3_text", 0.0)),
        consensus_iou_sam3_similarity=float(policy_params.get("consensus_iou_sam3_similarity", 0.0)),
        bias_sam3_text_by_class=per_class_maps.get("bias_sam3_text_by_class"),
        bias_sam3_similarity_by_class=per_class_maps.get("bias_sam3_similarity_by_class"),
        sam_floor_by_class=per_class_maps.get("sam_floor_by_class"),
        consensus_iou_sam3_text_by_class=per_class_maps.get("consensus_iou_sam3_text_by_class"),
        consensus_iou_sam3_similarity_by_class=per_class_maps.get(
            "consensus_iou_sam3_similarity_by_class"
        ),
    )
    final_payload = evaluator.evaluate(policy=final_policy, tag=f"{scenario_tag}_final")

    details = {
        "labels_tuned_per_class": ranked_labels,
        "policy_params": policy_params,
        "policy_params_per_class": per_class_maps,
        "threshold_logit_shift": float(best_shift),
        "policy_after_policy_search_f1": _safe_float(payload_after_policy.get("f1"), 0.0),
        "policy_after_per_class_search_f1": _safe_float(payload_after_per_class.get("f1"), 0.0),
        "policy_after_joint_tune_f1": _safe_float(payload_after_shift.get("f1"), 0.0),
        "final_policy": final_policy,
        "final_metrics": _extract_metrics(final_payload, gate_margin=gate_margin),
    }
    return final_payload, details


def _mk_variants(args: argparse.Namespace) -> List[VariantSpec]:
    variants: List[VariantSpec] = []
    if args.variants in {"both", "nonwindow"}:
        variants.append(
            VariantSpec(
                name="nonwindow",
                dataset=args.dataset,
                labeled_npz=Path(args.nonwindow_labeled),
                prepass_jsonl=Path(args.nonwindow_prepass_jsonl),
                baseline_eval_json=Path(args.nonwindow_baseline_eval_json),
                fixed_val_meta_json=Path(args.nonwindow_fixed_val_meta_json),
            )
        )
    if args.variants in {"both", "window"}:
        variants.append(
            VariantSpec(
                name="window",
                dataset=args.dataset,
                labeled_npz=Path(args.window_labeled),
                prepass_jsonl=Path(args.window_prepass_jsonl),
                baseline_eval_json=Path(args.window_baseline_eval_json),
                fixed_val_meta_json=Path(args.window_fixed_val_meta_json),
            )
        )
    return variants


def _check_files(variant: VariantSpec) -> None:
    for path in (
        variant.labeled_npz,
        variant.prepass_jsonl,
        variant.baseline_eval_json,
        variant.fixed_val_meta_json,
    ):
        if not path.exists():
            raise SystemExit(f"missing_required_file:{variant.name}:{path}")


def _scenario_name(use_source_policy: bool, use_joint_tune: bool, split_head_quality: bool) -> str:
    if use_source_policy and use_joint_tune and split_head_quality:
        return "full_stack"
    if (not use_source_policy) and use_joint_tune and split_head_quality:
        return "no_source_policy"
    if use_source_policy and (not use_joint_tune) and split_head_quality:
        return "no_joint_tune"
    if use_source_policy and use_joint_tune and (not split_head_quality):
        return "no_split_head_quality"
    return "custom"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-stack + ablation benchmark suite.")
    parser.add_argument("--dataset", default="qwen_dataset")
    parser.add_argument("--variants", choices=["both", "nonwindow", "window"], default="both")
    parser.add_argument(
        "--output-root",
        default=f"tmp/emb1024_calibration_20260219_161507/fullstack_ablation_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}",
    )
    parser.add_argument("--target-fp-ratio", type=float, default=0.2)
    parser.add_argument("--min-recall", type=float, default=0.6)
    parser.add_argument("--relax-fp-ratio", type=float, default=0.2)
    parser.add_argument("--threshold-steps", type=int, default=200)
    parser.add_argument("--optimize", choices=["f1", "recall", "tp"], default="f1")
    parser.add_argument("--sam3-text-quality-alpha", type=float, default=0.35)
    parser.add_argument(
        "--per-class-label-limit",
        type=int,
        default=12,
        help="Maximum labels to tune with per-class source policy refinement (<=0 means all).",
    )
    parser.add_argument(
        "--full-stack-only",
        action="store_true",
        help="Run only the full_stack scenario (source policy + joint tune + split-head quality).",
    )
    parser.add_argument("--gate-margin", type=float, default=0.02)

    parser.add_argument(
        "--nonwindow-labeled",
        default="tmp/emb1024_calibration_20260219_161507/image_context_ablation_20260223_112551_nonwindow_gpu0/nonwindow_4000.imgctx_d1024.labeled.npz",
    )
    parser.add_argument(
        "--window-labeled",
        default="tmp/emb1024_calibration_20260219_161507/image_context_ablation_20260223_112551_window_gpu1/window_4000.imgctx_d1024.labeled.npz",
    )
    parser.add_argument(
        "--nonwindow-prepass-jsonl",
        default="uploads/calibration_cache/features_backfill/20c8d44d69f51b2ffe528fb500e75672a306f67d/prepass.jsonl",
    )
    parser.add_argument(
        "--window-prepass-jsonl",
        default="uploads/calibration_cache/features_backfill/ceab65b2bff24d316ca5f858addaffed8abfdb11/prepass.jsonl",
    )
    parser.add_argument(
        "--nonwindow-baseline-eval-json",
        default="tmp/emb1024_calibration_20260219_161507/image_context_ablation_20260223_112551_nonwindow_gpu0/nonwindow_4000.imgctx_d1024.eval.json",
    )
    parser.add_argument(
        "--window-baseline-eval-json",
        default="tmp/emb1024_calibration_20260219_161507/image_context_ablation_20260223_112551_window_gpu1/window_4000.imgctx_d1024.eval.json",
    )
    parser.add_argument(
        "--nonwindow-fixed-val-meta-json",
        default="tmp/emb1024_calibration_20260219_161507/image_context_ablation_20260223_112551_nonwindow_gpu0/nonwindow_4000.meta.json",
    )
    parser.add_argument(
        "--window-fixed-val-meta-json",
        default="tmp/emb1024_calibration_20260219_161507/image_context_ablation_20260223_112551_window_gpu1/window_4000.meta.json",
    )

    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    variants = _mk_variants(args)
    if not variants:
        raise SystemExit("no_variants_selected")

    summary: Dict[str, Any] = {
        "generated_utc": _ts(),
        "dataset": args.dataset,
        "config": {
            "target_fp_ratio": float(args.target_fp_ratio),
            "min_recall": float(args.min_recall),
            "relax_fp_ratio": float(args.relax_fp_ratio),
            "threshold_steps": int(args.threshold_steps),
            "optimize": args.optimize,
            "per_class_label_limit": int(args.per_class_label_limit),
            "gate_margin": float(args.gate_margin),
            "eval_iou": 0.5,
            "dedupe_iou": 0.75,
            "scoreless_iou": 0.0,
        },
        "variants": {},
    }

    for variant in variants:
        _check_files(variant)
        _log(f"=== VARIANT {variant.name} START ===")
        variant_dir = output_root / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)

        baseline_payload = _load_json(variant.baseline_eval_json)
        baseline_metrics = _extract_metrics(baseline_payload, gate_margin=float(args.gate_margin))

        fixed_val_path = variant_dir / "fixed_val_images.json"
        val_images = _save_val_images(variant.fixed_val_meta_json, fixed_val_path)
        _log(f"{variant.name}: fixed val images={len(val_images)}")

        trained: Dict[str, Tuple[Path, Path]] = {}
        for split_head_quality in (True, False):
            train_key = "split_on" if split_head_quality else "split_off"
            out_prefix = variant_dir / train_key / "model"
            model_json, meta_json = _train_and_tune_xgb(
                variant=variant,
                out_prefix=out_prefix,
                fixed_val_images_path=fixed_val_path,
                target_fp_ratio=float(args.target_fp_ratio),
                min_recall=float(args.min_recall),
                relax_fp_ratio=float(args.relax_fp_ratio),
                steps=int(args.threshold_steps),
                optimize=args.optimize,
                split_head_quality=split_head_quality,
                sam3_text_quality_alpha=float(args.sam3_text_quality_alpha),
            )
            trained[train_key] = (model_json, meta_json)

        variant_summary: Dict[str, Any] = {
            "baseline": {
                "metrics": baseline_metrics,
                "path": str(variant.baseline_eval_json),
            },
            "scenarios": {},
        }

        if args.full_stack_only:
            scenarios = [
                (True, True, True),  # full_stack
            ]
        else:
            scenarios = [
                (True, True, True),   # full_stack
                (False, True, True),  # no_source_policy
                (True, False, True),  # no_joint_tune
                (True, True, False),  # no_split_head_quality
            ]

        for use_source_policy, use_joint_tune, split_head_quality in scenarios:
            scenario = _scenario_name(use_source_policy, use_joint_tune, split_head_quality)
            train_key = "split_on" if split_head_quality else "split_off"
            model_json_src, meta_json_src = trained[train_key]
            scenario_dir = variant_dir / "scenarios" / scenario
            scenario_dir.mkdir(parents=True, exist_ok=True)

            # Use scenario-local copies so each scenario can persist its own policy/meta outputs.
            model_json = scenario_dir / f"{variant.name}.{scenario}.json"
            meta_json = scenario_dir / f"{variant.name}.{scenario}.meta.json"
            shutil.copy2(model_json_src, model_json)
            shutil.copy2(meta_json_src, meta_json)

            base_meta = _load_json(meta_json)
            base_thresholds = _threshold_source(base_meta)
            if not base_thresholds:
                raise RuntimeError(f"missing_thresholds:{variant.name}:{scenario}")

            evaluator = Evaluator(
                variant=variant,
                model_json=model_json,
                model_meta=meta_json,
                eval_iou=0.5,
                dedupe_iou=0.75,
                scoreless_iou=0.0,
                run_dir=scenario_dir,
            )

            final_payload, details = _scenario_run(
                evaluator=evaluator,
                base_thresholds=base_thresholds,
                use_source_policy=use_source_policy,
                use_joint_tune=use_joint_tune,
                per_class_label_limit=int(args.per_class_label_limit),
                gate_margin=float(args.gate_margin),
                scenario_tag=f"{variant.name}_{scenario}",
            )

            eval_path = scenario_dir / "eval.final.json"
            _write_json(eval_path, final_payload)
            _write_json(scenario_dir / "details.json", details)

            metrics = details["final_metrics"]
            metrics["delta_vs_baseline_f1"] = float(metrics["f1"]) - float(baseline_metrics["f1"])
            metrics["scenario"] = scenario
            metrics["split_head_quality"] = bool(split_head_quality)
            metrics["use_source_policy"] = bool(use_source_policy)
            metrics["use_joint_tune"] = bool(use_joint_tune)
            metrics["model_json"] = str(model_json)
            metrics["meta_json"] = str(meta_json)
            metrics["eval_json"] = str(eval_path)

            variant_summary["scenarios"][scenario] = {
                "metrics": metrics,
                "details": details,
            }
            _log(
                f"{variant.name}/{scenario}: "
                f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
                f"delta_vs_union={metrics['delta_vs_union_f1']:+.4f} "
                f"delta_vs_baseline={metrics['delta_vs_baseline_f1']:+.4f} "
                f"gate_pass={metrics['gate_pass']}"
            )

        summary["variants"][variant.name] = variant_summary
        _log(f"=== VARIANT {variant.name} END ===")

    summary_path = output_root / "summary.json"
    _write_json(summary_path, summary)

    # Simple markdown report for quick scan.
    lines = [
        "# Full-Stack + LOO Ablation Suite",
        "",
        f"- Generated (UTC): {summary['generated_utc']}",
        f"- Dataset: `{summary['dataset']}`",
        f"- Gate margin (delta vs detector-union comparator): `{summary['config']['gate_margin']:+.3f}` F1",
        "",
    ]
    for variant_name, payload in summary["variants"].items():
        base = payload["baseline"]["metrics"]
        lines += [
            f"## Variant: `{variant_name}`",
            "",
            (
                "Baseline (current constrained): "
                f"P={base['precision']:.4f} R={base['recall']:.4f} F1={base['f1']:.4f} "
                f"delta_vs_union={base['delta_vs_union_f1']:+.4f}"
            ),
            "",
            "| Scenario | Precision | Recall | F1 | Δ vs Union | Δ vs Baseline | Coverage Pres. | Gate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for scenario in ("full_stack", "no_source_policy", "no_joint_tune", "no_split_head_quality"):
            row = payload["scenarios"].get(scenario, {}).get("metrics")
            if not row:
                continue
            lines.append(
                "| {scenario} | {p:.4f} | {r:.4f} | {f1:.4f} | {du:+.4f} | {db:+.4f} | {cp:.4f} | {gate} |".format(
                    scenario=scenario,
                    p=row["precision"],
                    r=row["recall"],
                    f1=row["f1"],
                    du=row["delta_vs_union_f1"],
                    db=row["delta_vs_baseline_f1"],
                    cp=row["coverage_preservation"],
                    gate="PASS" if row["gate_pass"] else "FAIL",
                )
            )
        lines.append("")

    report_path = output_root / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    _log(f"Suite complete: {summary_path}")
    _log(f"Report: {report_path}")


if __name__ == "__main__":
    main()
