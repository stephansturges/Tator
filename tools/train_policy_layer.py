#!/usr/bin/env python3
"""Train learned second-stage policy models on top of the base ensemble XGB scorer."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools import train_ensemble_xgb as texgb
from tools.policy_layer_features import build_policy_feature_matrix
from tools.policy_runtime import (
    apply_hand_policy,
    predict_base_probabilities,
    resolve_thresholds,
    transform_base_features,
)

POLICY_LAYER_VERSION = 1


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def _compute_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_true == 1) & (pred == 1)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _split_outer_indices(meta_rows: Sequence[Dict[str, Any]], base_meta: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    train_images = {str(v) for v in (base_meta.get("split_train_images") or []) if str(v)}
    val_images = {str(v) for v in (base_meta.get("split_val_images") or []) if str(v)}
    if not train_images or not val_images:
        raise SystemExit("base_meta_missing_split_images")
    train_idx = [idx for idx, row in enumerate(meta_rows) if str(row.get("image") or "") in train_images]
    val_idx = [idx for idx, row in enumerate(meta_rows) if str(row.get("image") or "") in val_images]
    if not train_idx or not val_idx:
        raise SystemExit("policy_layer_empty_outer_split")
    return train_idx, val_idx


def _fold_splits(groups: Sequence[str], n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    unique_groups = sorted({g for g in groups if g})
    if len(unique_groups) < 2:
        raise SystemExit("policy_layer_not_enough_groups")
    n_splits = max(2, min(int(n_splits), len(unique_groups)))
    splitter = GroupKFold(n_splits=n_splits)
    index_arr = np.arange(len(groups))
    return [(train, val) for train, val in splitter.split(index_arr, groups=groups)]


def _holdout_split(groups: Sequence[str], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    unique_groups = {g for g in groups if g}
    if len(unique_groups) < 2:
        idx = np.arange(len(groups))
        return idx, idx
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=int(seed))
    index_arr = np.arange(len(groups))
    return next(splitter.split(index_arr, groups=groups))


def _prepare_fold_matrix(
    X_raw: np.ndarray,
    feature_names: Sequence[str],
    *,
    train_idx: Sequence[int],
    standardize: bool,
    log1p_counts: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    X_work = np.asarray(X_raw, dtype=np.float32).copy()
    if log1p_counts:
        X_work = texgb._apply_log1p_counts(X_work, list(feature_names))
    mean = None
    std = None
    if standardize:
        mean = X_work[list(train_idx)].mean(axis=0)
        std = X_work[list(train_idx)].std(axis=0)
        X_work = texgb._standardize(X_work, mean, std)
    return X_work, mean, std


def _train_base_fold_family(
    X_raw: np.ndarray,
    y: np.ndarray,
    meta_rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    *,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    base_meta: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    X_work, mean, std = _prepare_fold_matrix(
        X_raw,
        feature_names,
        train_idx=train_idx,
        standardize=bool(base_meta.get("standardize")),
        log1p_counts=bool(base_meta.get("log1p_counts")),
    )
    params = dict(base_meta.get("xgb_params") or {})
    params["seed"] = int(seed)
    num_boost_round = int(base_meta.get("n_estimators") or 600)
    early_stopping_rounds = 50
    dtrain = xgb.DMatrix(X_work[list(train_idx)], label=y[list(train_idx)])
    dval = xgb.DMatrix(X_work[list(val_idx)], label=y[list(val_idx)])
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    detector_head = None
    sam_only_head = None
    split_meta = base_meta.get("split_head") if isinstance(base_meta.get("split_head"), dict) else {}
    if bool(split_meta.get("enabled")):
        det_train_idx = [idx for idx in train_idx if texgb._has_detector_support(meta_rows[idx])]
        sam_train_idx = [idx for idx in train_idx if not texgb._has_detector_support(meta_rows[idx])]
        det_val_idx = [idx for idx in val_idx if texgb._has_detector_support(meta_rows[idx])]
        sam_val_idx = [idx for idx in val_idx if not texgb._has_detector_support(meta_rows[idx])]
        detector_head = texgb._train_head_model(
            X=X_work,
            y=y,
            train_idx=det_train_idx,
            val_idx=det_val_idx,
            params=params,
            n_estimators=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
        sam_only_head = texgb._train_head_model(
            X=X_work,
            y=y,
            train_idx=sam_train_idx,
            val_idx=sam_val_idx,
            params=params,
            n_estimators=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )

    quality_model = None
    quality_meta = base_meta.get("sam3_text_quality") if isinstance(base_meta.get("sam3_text_quality"), dict) else {}
    alpha = _safe_float(quality_meta.get("alpha"), 0.0)
    if bool(quality_meta.get("enabled")):
        q_train_idx = [idx for idx in train_idx if texgb._is_sam3_text_only(meta_rows[idx])]
        q_val_idx = [idx for idx in val_idx if texgb._is_sam3_text_only(meta_rows[idx])]
        quality_model = texgb._train_head_model(
            X=X_work,
            y=y,
            train_idx=q_train_idx,
            val_idx=q_val_idx,
            params={**params, "max_depth": int(min(int(params.get("max_depth", 8)), 4))},
            n_estimators=int(min(num_boost_round, 400)),
            early_stopping_rounds=early_stopping_rounds,
        )
    similarity_quality_model = None
    similarity_quality_meta = (
        base_meta.get("sam3_similarity_quality")
        if isinstance(base_meta.get("sam3_similarity_quality"), dict)
        else {}
    )
    similarity_alpha = _safe_float(similarity_quality_meta.get("alpha"), 0.0)
    if bool(similarity_quality_meta.get("enabled")):
        q_train_idx = [idx for idx in train_idx if texgb._is_sam3_similarity_only(meta_rows[idx])]
        q_val_idx = [idx for idx in val_idx if texgb._is_sam3_similarity_only(meta_rows[idx])]
        similarity_quality_model = texgb._train_head_model(
            X=X_work,
            y=y,
            train_idx=q_train_idx,
            val_idx=q_val_idx,
            params={**params, "max_depth": int(min(int(params.get("max_depth", 8)), 4))},
            n_estimators=int(min(num_boost_round, 400)),
            early_stopping_rounds=early_stopping_rounds,
        )

    return {
        "booster": booster,
        "detector_head": detector_head,
        "sam_only_head": sam_only_head,
        "quality_model": quality_model,
        "quality_alpha": alpha,
        "similarity_quality_model": similarity_quality_model,
        "similarity_quality_alpha": similarity_alpha,
        "X_work": X_work,
    }


def _predict_with_fold_family(family: Dict[str, Any], meta_rows: Sequence[Dict[str, Any]], subset_idx: Sequence[int]) -> np.ndarray:
    X_work = family["X_work"]
    idx = np.asarray(list(subset_idx), dtype=np.int64)
    probs = np.asarray(family["booster"].predict(xgb.DMatrix(X_work[idx])), dtype=np.float32)
    det_head = family.get("detector_head")
    sam_head = family.get("sam_only_head")
    if det_head is not None or sam_head is not None:
        det_mask = np.asarray([texgb._has_detector_support(meta_rows[int(i)]) for i in idx], dtype=bool)
        if det_head is not None and det_mask.any():
            probs[det_mask] = det_head.predict(xgb.DMatrix(X_work[idx[det_mask]]))
        if sam_head is not None and (~det_mask).any():
            probs[~det_mask] = sam_head.predict(xgb.DMatrix(X_work[idx[~det_mask]]))
    q_model = family.get("quality_model")
    alpha = _safe_float(family.get("quality_alpha"), 0.0)
    if q_model is not None and alpha > 0.0:
        text_mask = np.asarray([texgb._is_sam3_text_only(meta_rows[int(i)]) for i in idx], dtype=bool)
        if text_mask.any():
            q_probs = np.asarray(q_model.predict(xgb.DMatrix(X_work[idx[text_mask]])), dtype=np.float32)
            probs[text_mask] = texgb._blend_probabilities(probs[text_mask], q_probs, alpha)
    sim_q_model = family.get("similarity_quality_model")
    sim_alpha = _safe_float(family.get("similarity_quality_alpha"), 0.0)
    if sim_q_model is not None and sim_alpha > 0.0:
        sim_mask = np.asarray([texgb._is_sam3_similarity_only(meta_rows[int(i)]) for i in idx], dtype=bool)
        if sim_mask.any():
            q_probs = np.asarray(sim_q_model.predict(xgb.DMatrix(X_work[idx[sim_mask]])), dtype=np.float32)
            probs[sim_mask] = texgb._blend_probabilities(probs[sim_mask], q_probs, sim_alpha)
    return probs.astype(np.float32)


def _build_oof_base_predictions(
    X_raw: np.ndarray,
    y: np.ndarray,
    meta_rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    train_idx: Sequence[int],
    base_meta: Dict[str, Any],
    seed: int,
    n_folds: int,
) -> np.ndarray:
    train_idx = list(train_idx)
    groups = [str(meta_rows[idx].get("image") or f"row_{idx}") for idx in train_idx]
    splits = _fold_splits(groups, n_splits=n_folds)
    out = np.zeros(len(train_idx), dtype=np.float32)
    for fold_id, (local_train, local_val) in enumerate(splits):
        fold_train_idx = [train_idx[i] for i in local_train]
        fold_val_idx = [train_idx[i] for i in local_val]
        family = _train_base_fold_family(
            X_raw,
            y,
            meta_rows,
            feature_names,
            train_idx=fold_train_idx,
            val_idx=fold_val_idx,
            base_meta=base_meta,
            seed=seed + fold_id,
        )
        preds = _predict_with_fold_family(family, meta_rows, fold_val_idx)
        out[local_val] = preds
    return out


def _calibrate_thresholds(
    probs: np.ndarray,
    y_true: np.ndarray,
    labels: Sequence[str],
    *,
    target_fp_ratio: float,
    optimize: str,
    min_recall: float,
    steps: int,
) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    thresholds = np.linspace(0.0, 1.0, int(max(20, steps)))
    best = texgb._select_threshold(probs, y_true, thresholds, target_fp_ratio, optimize, min_recall)
    if best is None:
        best = texgb._select_threshold(probs, y_true, thresholds, target_fp_ratio, "recall", 0.0)
    default_threshold = float(best["threshold"]) if best else 0.5
    by_label: Dict[str, float] = {}
    for label in sorted({str(lbl).strip() for lbl in labels if str(lbl).strip()}):
        mask = np.asarray([str(lbl).strip() == label for lbl in labels], dtype=bool)
        if not mask.any():
            continue
        label_best = texgb._select_threshold(probs[mask], y_true[mask], thresholds, target_fp_ratio, optimize, min_recall)
        if label_best is None:
            label_best = texgb._select_threshold(probs[mask], y_true[mask], thresholds, target_fp_ratio, "recall", 0.0)
        if label_best is not None:
            by_label[label] = float(label_best["threshold"])
    return default_threshold, by_label, best or {}


def _apply_thresholds(probs: np.ndarray, labels: Sequence[str], default_threshold: float, thresholds_by_label: Dict[str, float]) -> np.ndarray:
    pred = np.zeros(len(labels), dtype=np.int64)
    for idx, label in enumerate(labels):
        thr = float(thresholds_by_label.get(str(label).strip(), default_threshold))
        pred[idx] = 1 if float(probs[idx]) >= thr else 0
    return pred


def _subgroup_metrics(y_true: np.ndarray, pred: np.ndarray, subgroup_flags: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, mask in subgroup_flags.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != y_true.shape[0] or not mask.any():
            out[name] = {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            continue
        out[name] = _compute_metrics(y_true[mask], pred[mask])
    return out


def _evaluate_probs(
    probs: np.ndarray,
    y_true: np.ndarray,
    labels: Sequence[str],
    subgroup_flags: Dict[str, np.ndarray],
    *,
    default_threshold: float,
    thresholds_by_label: Dict[str, float],
) -> Dict[str, Any]:
    pred = _apply_thresholds(probs, labels, default_threshold, thresholds_by_label)
    metrics = _compute_metrics(y_true, pred)
    metrics["positive_rate"] = float(pred.mean()) if pred.size else 0.0
    metrics["subgroups"] = _subgroup_metrics(y_true, pred, subgroup_flags)
    return metrics


def _compare_to_baseline(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "delta_f1": float(candidate["f1"] - baseline["f1"]),
        "delta_precision": float(candidate["precision"] - baseline["precision"]),
        "delta_recall": float(candidate["recall"] - baseline["recall"]),
        "delta_tp": int(candidate["tp"] - baseline["tp"]),
        "delta_fp": int(candidate["fp"] - baseline["fp"]),
        "delta_fn": int(candidate["fn"] - baseline["fn"]),
    }
    subgroups: Dict[str, Any] = {}
    cand_sub = candidate.get("subgroups") if isinstance(candidate.get("subgroups"), dict) else {}
    base_sub = baseline.get("subgroups") if isinstance(baseline.get("subgroups"), dict) else {}
    for name in sorted(set(cand_sub) | set(base_sub)):
        c = cand_sub.get(name) or {}
        b = base_sub.get(name) or {}
        subgroups[name] = {
            "delta_f1": float(_safe_float(c.get("f1")) - _safe_float(b.get("f1"))),
            "delta_tp": int(_safe_float(c.get("tp")) - _safe_float(b.get("tp"))),
            "delta_fp": int(_safe_float(c.get("fp")) - _safe_float(b.get("fp"))),
            "delta_fn": int(_safe_float(c.get("fn")) - _safe_float(b.get("fn"))),
        }
    out["subgroups"] = subgroups
    return out


def _candidate_sort_key(name: str, row: Dict[str, Any]) -> Tuple[float, float, float, float, int]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    compare = row.get("compare_to_baseline") if isinstance(row.get("compare_to_baseline"), dict) else {}
    subgroup_deltas = compare.get("subgroups") if isinstance(compare.get("subgroups"), dict) else {}
    sam_only_delta = subgroup_deltas.get("sam_only") if isinstance(subgroup_deltas.get("sam_only"), dict) else {}
    f1 = _safe_float(metrics.get("f1"), 0.0)
    recall = _safe_float(metrics.get("recall"), 0.0)
    fp = _safe_float(metrics.get("fp"), 0.0)
    sam_only_delta_tp = _safe_float(sam_only_delta.get("delta_tp"), 0.0)
    simplicity = 1 if name == "lreg" else 0
    # Higher is better for the first four terms; LR wins exact ties by simplicity.
    return (f1, recall, -fp, sam_only_delta_tp, simplicity)


def _select_policy_candidate(candidates: Dict[str, Dict[str, Any]]) -> str:
    if not candidates:
        raise ValueError("no policy candidates")
    if len(candidates) == 1:
        return next(iter(candidates.keys()))

    ordered = sorted(
        candidates.items(),
        key=lambda item: _candidate_sort_key(item[0], item[1]),
        reverse=True,
    )
    best_name, best_row = ordered[0]
    runner_up_name, runner_up_row = ordered[1]

    best_metrics = best_row.get("metrics") if isinstance(best_row.get("metrics"), dict) else {}
    runner_metrics = runner_up_row.get("metrics") if isinstance(runner_up_row.get("metrics"), dict) else {}
    best_f1 = _safe_float(best_metrics.get("f1"), 0.0)
    runner_f1 = _safe_float(runner_metrics.get("f1"), 0.0)

    if abs(best_f1 - runner_f1) >= 0.0015:
        return best_name

    best_recall = _safe_float(best_metrics.get("recall"), 0.0)
    runner_recall = _safe_float(runner_metrics.get("recall"), 0.0)
    if abs(best_recall - runner_recall) <= 0.0015:
        best_fp = _safe_float(best_metrics.get("fp"), 0.0)
        runner_fp = _safe_float(runner_metrics.get("fp"), 0.0)
        if best_fp != runner_fp:
            return best_name if best_fp < runner_fp else runner_up_name

    best_compare = best_row.get("compare_to_baseline") if isinstance(best_row.get("compare_to_baseline"), dict) else {}
    runner_compare = runner_up_row.get("compare_to_baseline") if isinstance(runner_up_row.get("compare_to_baseline"), dict) else {}
    best_subgroups = best_compare.get("subgroups") if isinstance(best_compare.get("subgroups"), dict) else {}
    runner_subgroups = runner_compare.get("subgroups") if isinstance(runner_compare.get("subgroups"), dict) else {}
    best_sam_only = best_subgroups.get("sam_only") if isinstance(best_subgroups.get("sam_only"), dict) else {}
    runner_sam_only = runner_subgroups.get("sam_only") if isinstance(runner_subgroups.get("sam_only"), dict) else {}
    best_sam_only_tp = _safe_float(best_sam_only.get("delta_tp"), 0.0)
    runner_sam_only_tp = _safe_float(runner_sam_only.get("delta_tp"), 0.0)
    if best_sam_only_tp != runner_sam_only_tp:
        return best_name if best_sam_only_tp > runner_sam_only_tp else runner_up_name

    return "lreg" if "lreg" in candidates else best_name


def _fit_lr_candidate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    base_probs_train: np.ndarray,
    labels_train: Sequence[str],
    groups_train: Sequence[str],
    *,
    seed: int,
    target_fp_ratio: float,
    optimize: str,
    min_recall: float,
    steps: int,
) -> Tuple[LogisticRegression, np.ndarray, np.ndarray, Dict[str, Any]]:
    inner_train, inner_val = _holdout_split(groups_train, seed)
    best = None
    for l1_ratio in [0.0, 0.2, 0.5]:
        for c_val in [0.05, 0.1, 0.2, 0.5, 1.0]:
            mean = X_train[inner_train].mean(axis=0)
            std = X_train[inner_train].std(axis=0)
            std = np.where(std < 1e-6, 1.0, std)
            train_scaled = (X_train[inner_train] - mean) / std
            val_scaled = (X_train[inner_val] - mean) / std
            model = LogisticRegression(
                solver="saga",
                penalty="elasticnet",
                l1_ratio=float(l1_ratio),
                C=float(c_val),
                class_weight="balanced",
                max_iter=4000,
                random_state=int(seed),
            )
            model.fit(train_scaled, y_train[inner_train])
            train_probs = _sigmoid(np.asarray([math.log(max(min(float(p), 1.0 - 1e-6), 1e-6) / (1.0 - max(min(float(p), 1.0 - 1e-6), 1e-6))) for p in base_probs_train[inner_train]], dtype=np.float32) + model.decision_function(train_scaled))
            default_thr, thr_by_label, _ = _calibrate_thresholds(train_probs, y_train[inner_train], [labels_train[i] for i in inner_train], target_fp_ratio=target_fp_ratio, optimize=optimize, min_recall=min_recall, steps=steps)
            val_probs = _sigmoid(np.asarray([math.log(max(min(float(p), 1.0 - 1e-6), 1e-6) / (1.0 - max(min(float(p), 1.0 - 1e-6), 1e-6))) for p in base_probs_train[inner_val]], dtype=np.float32) + model.decision_function(val_scaled))
            metrics = _evaluate_probs(val_probs, y_train[inner_val], [labels_train[i] for i in inner_val], {"sam_only": np.zeros(len(inner_val), dtype=bool)}, default_threshold=default_thr, thresholds_by_label=thr_by_label)
            rank = (metrics["f1"], -metrics["fp"], metrics["recall"])
            if best is None or rank > best["rank"]:
                best = {"rank": rank, "params": {"l1_ratio": float(l1_ratio), "C": float(c_val)}}
    assert best is not None
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    X_scaled = (X_train - mean) / std
    model = LogisticRegression(
        solver="saga",
        penalty="elasticnet",
        l1_ratio=float(best["params"]["l1_ratio"]),
        C=float(best["params"]["C"]),
        class_weight="balanced",
        max_iter=4000,
        random_state=int(seed),
    )
    model.fit(X_scaled, y_train)
    return model, mean.astype(np.float32), std.astype(np.float32), best["params"]


def _predict_lr_final(model: LogisticRegression, X: np.ndarray, mean: np.ndarray, std: np.ndarray, base_probs: np.ndarray) -> np.ndarray:
    X_scaled = (X - mean) / np.where(std < 1e-6, 1.0, std)
    delta = np.asarray(model.decision_function(X_scaled), dtype=np.float32)
    base_logits = np.asarray([math.log(max(min(float(p), 1.0 - 1e-6), 1e-6) / (1.0 - max(min(float(p), 1.0 - 1e-6), 1e-6))) for p in base_probs], dtype=np.float32)
    return _sigmoid(base_logits + delta)


def _fit_xgb_candidate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    labels_train: Sequence[str],
    groups_train: Sequence[str],
    *,
    seed: int,
    target_fp_ratio: float,
    optimize: str,
    min_recall: float,
    steps: int,
) -> Tuple[xgb.Booster, Dict[str, Any]]:
    inner_train, inner_val = _holdout_split(groups_train, seed)
    best = None
    for max_depth in [2, 3, 4]:
        for n_estimators in [120, 240]:
            for learning_rate in [0.03, 0.07]:
                for min_child_weight in [4, 8]:
                    for gamma in [0.0, 0.2]:
                        for reg_lambda in [2.0, 5.0]:
                            for reg_alpha in [0.0, 0.5]:
                                params = {
                                    "objective": "binary:logistic",
                                    "eval_metric": "logloss",
                                    "max_depth": int(max_depth),
                                    "eta": float(learning_rate),
                                    "subsample": 0.8,
                                    "colsample_bytree": 0.8,
                                    "min_child_weight": float(min_child_weight),
                                    "gamma": float(gamma),
                                    "lambda": float(reg_lambda),
                                    "alpha": float(reg_alpha),
                                    "tree_method": "hist",
                                    "seed": int(seed),
                                }
                                dtrain = xgb.DMatrix(X_train[inner_train], label=y_train[inner_train])
                                dval = xgb.DMatrix(X_train[inner_val], label=y_train[inner_val])
                                booster = xgb.train(
                                    params,
                                    dtrain,
                                    num_boost_round=int(n_estimators),
                                    evals=[(dtrain, "train"), (dval, "val")],
                                    early_stopping_rounds=30,
                                    verbose_eval=False,
                                )
                                train_probs = np.asarray(booster.predict(xgb.DMatrix(X_train[inner_train])), dtype=np.float32)
                                default_thr, thr_by_label, _ = _calibrate_thresholds(train_probs, y_train[inner_train], [labels_train[i] for i in inner_train], target_fp_ratio=target_fp_ratio, optimize=optimize, min_recall=min_recall, steps=steps)
                                val_probs = np.asarray(booster.predict(xgb.DMatrix(X_train[inner_val])), dtype=np.float32)
                                metrics = _evaluate_probs(val_probs, y_train[inner_val], [labels_train[i] for i in inner_val], {"sam_only": np.zeros(len(inner_val), dtype=bool)}, default_threshold=default_thr, thresholds_by_label=thr_by_label)
                                rank = (metrics["f1"], -metrics["fp"], metrics["recall"])
                                if best is None or rank > best["rank"]:
                                    best = {
                                        "rank": rank,
                                        "params": params,
                                        "num_boost_round": int(n_estimators),
                                    }
    assert best is not None
    dtrain = xgb.DMatrix(X_train, label=y_train)
    booster = xgb.train(best["params"], dtrain, num_boost_round=best["num_boost_round"], evals=[(dtrain, "train")], verbose_eval=False)
    return booster, {**best["params"], "num_boost_round": best["num_boost_round"]}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _render_report(payload: Dict[str, Any]) -> str:
    lines = ["# Policy Layer Selection Report", ""]
    lines.append(f"Requested variant: `{payload.get('requested_variant')}`")
    lines.append(f"Selected variant: `{payload.get('selected_variant')}`")
    lines.append("")
    baseline = payload.get("baseline") or {}
    lines.append("## Baseline hand policy")
    lines.append(f"- F1 `{_safe_float(baseline.get('f1')):.4f}`")
    lines.append(f"- Precision `{_safe_float(baseline.get('precision')):.4f}`")
    lines.append(f"- Recall `{_safe_float(baseline.get('recall')):.4f}`")
    lines.append("")
    for name in ["lreg", "xgb"]:
        row = payload.get("candidates", {}).get(name)
        if not row:
            continue
        metrics = row.get("metrics") or {}
        compare = row.get("compare_to_baseline") or {}
        lines.append(f"## {name}")
        lines.append(f"- F1 `{_safe_float(metrics.get('f1')):.4f}` (`{_safe_float(compare.get('delta_f1')):+.4f}` vs baseline)")
        lines.append(f"- Precision `{_safe_float(metrics.get('precision')):.4f}`")
        lines.append(f"- Recall `{_safe_float(metrics.get('recall')):.4f}`")
        lines.append(f"- TP `{int(_safe_float(metrics.get('tp'))):d}` FP `{int(_safe_float(metrics.get('fp'))):d}` FN `{int(_safe_float(metrics.get('fn'))):d}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train second-stage learned policy layer for ensemble XGB.")
    parser.add_argument("--input", required=True, help="Input labeled .npz file.")
    parser.add_argument("--base-model", required=True, help="Base ensemble_xgb.json path.")
    parser.add_argument("--base-meta", required=True, help="Base ensemble_xgb.meta.json path.")
    parser.add_argument("--output-dir", required=True, help="Directory to write policy-layer artifacts.")
    parser.add_argument("--variant", default="bakeoff", choices=["bakeoff", "xgb", "lreg"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nested-folds", type=int, default=5)
    parser.add_argument("--target-fp-ratio", type=float, default=0.2)
    parser.add_argument("--min-recall", type=float, default=0.6)
    parser.add_argument("--threshold-steps", type=int, default=200)
    parser.add_argument("--optimize", default="f1", choices=["f1", "tp", "recall"])
    parser.add_argument("--enable-anchor-similarity", action="store_true")
    parser.add_argument("--anchor-min-base-prob", type=float, default=0.9)
    parser.add_argument("--anchor-topk-same-label", type=int, default=4)
    parser.add_argument("--anchor-topk-any", type=int, default=8)
    args = parser.parse_args()

    labeled = np.load(args.input, allow_pickle=True)
    X_raw = np.asarray(labeled["X"], dtype=np.float32)
    y = np.asarray(labeled["y"], dtype=np.int64)
    feature_names = [str(name) for name in labeled.get("feature_names", [])]
    meta_rows = [json.loads(str(row)) for row in labeled["meta"]]

    base_meta_path = Path(args.base_meta)
    base_model_path = Path(args.base_model)
    base_meta = json.loads(base_meta_path.read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    job_dir = base_meta_path.parent
    anchor_similarity_cfg = {
        "enabled": bool(args.enable_anchor_similarity),
        "min_base_prob": float(args.anchor_min_base_prob),
        "topk_same_label": int(args.anchor_topk_same_label),
        "topk_any": int(args.anchor_topk_any),
        "require_detector_support": True,
    }

    outer_train_idx, outer_val_idx = _split_outer_indices(meta_rows, base_meta)
    oof_probs = _build_oof_base_predictions(
        X_raw,
        y,
        meta_rows,
        feature_names,
        outer_train_idx,
        base_meta,
        int(args.seed),
        int(args.nested_folds),
    )
    X_base_full = transform_base_features(X_raw, feature_names, base_meta)
    base_probs_val = predict_base_probabilities(
        X_base_full[list(outer_val_idx)],
        meta_rows=[meta_rows[idx] for idx in outer_val_idx],
        model_path=base_model_path,
        meta=base_meta,
    )

    train_bundle = build_policy_feature_matrix(
        X_raw[list(outer_train_idx)],
        feature_names,
        [meta_rows[idx] for idx in outer_train_idx],
        oof_probs,
        anchor_similarity=anchor_similarity_cfg,
    )
    val_bundle = build_policy_feature_matrix(
        X_raw[list(outer_val_idx)],
        feature_names,
        [meta_rows[idx] for idx in outer_val_idx],
        base_probs_val,
        anchor_similarity=anchor_similarity_cfg,
    )
    schema_hash = str(train_bundle["feature_schema_hash"])
    if schema_hash != str(val_bundle["feature_schema_hash"]):
        raise SystemExit("policy_feature_schema_hash_mismatch")

    policy_feature_schema = {
        "version": POLICY_LAYER_VERSION,
        "feature_schema_hash": schema_hash,
        "feature_names": train_bundle["feature_names"],
        "anchor_similarity": train_bundle.get("anchor_similarity"),
    }
    _write_json(output_dir / "policy_layer_feature_schema.json", policy_feature_schema)

    train_X = np.asarray(train_bundle["X"], dtype=np.float32)
    train_y = y[list(outer_train_idx)]
    train_labels = [str(meta_rows[idx].get("label") or "").strip().lower() for idx in outer_train_idx]
    train_groups = [str(meta_rows[idx].get("image") or f"row_{idx}") for idx in outer_train_idx]
    val_X = np.asarray(val_bundle["X"], dtype=np.float32)
    val_y = y[list(outer_val_idx)]
    val_labels = [str(meta_rows[idx].get("label") or "").strip().lower() for idx in outer_val_idx]

    # Baseline development comparison against the current hand policy stored in base meta.
    base_default_threshold, base_thresholds_by_label = resolve_thresholds(base_meta)
    baseline_policy = base_meta.get("ensemble_policy") if isinstance(base_meta.get("ensemble_policy"), dict) else {}
    baseline_rows = apply_hand_policy(
        probs=base_probs_val,
        meta_rows=[meta_rows[idx] for idx in outer_val_idx],
        policy=baseline_policy,
        default_threshold=base_default_threshold,
        thresholds_by_label=base_thresholds_by_label,
    )
    baseline_pred = np.asarray([1 if row.get("accept") else 0 for row in baseline_rows], dtype=np.int64)
    baseline_metrics = _compute_metrics(val_y, baseline_pred)
    baseline_metrics["subgroups"] = _subgroup_metrics(val_y, baseline_pred, val_bundle["subgroup_flags"])

    candidates: Dict[str, Any] = {}

    if args.variant in {"bakeoff", "lreg"}:
        lr_model, lr_mean, lr_std, lr_params = _fit_lr_candidate(
            train_X,
            train_y,
            oof_probs,
            train_labels,
            train_groups,
            seed=int(args.seed),
            target_fp_ratio=float(args.target_fp_ratio),
            optimize=str(args.optimize),
            min_recall=float(args.min_recall),
            steps=int(args.threshold_steps),
        )
        train_probs_lr = _predict_lr_final(lr_model, train_X, lr_mean, lr_std, oof_probs)
        lr_default_threshold, lr_thresholds, lr_cal = _calibrate_thresholds(
            train_probs_lr,
            train_y,
            train_labels,
            target_fp_ratio=float(args.target_fp_ratio),
            optimize=str(args.optimize),
            min_recall=float(args.min_recall),
            steps=int(args.threshold_steps),
        )
        val_probs_lr = _predict_lr_final(lr_model, val_X, lr_mean, lr_std, base_probs_val)
        lr_metrics = _evaluate_probs(
            val_probs_lr,
            val_y,
            val_labels,
            val_bundle["subgroup_flags"],
            default_threshold=lr_default_threshold,
            thresholds_by_label=lr_thresholds,
        )
        lr_path = output_dir / "ensemble_policy_lr.joblib"
        joblib.dump({"model": lr_model, "feature_mean": lr_mean, "feature_std": lr_std}, lr_path, compress=3)
        lr_meta = {
            "version": POLICY_LAYER_VERSION,
            "variant": "lreg",
            "feature_schema_hash": schema_hash,
            "feature_names": train_bundle["feature_names"],
            "anchor_similarity": train_bundle.get("anchor_similarity"),
            "calibrated_threshold": lr_default_threshold,
            "calibrated_thresholds": lr_thresholds,
            "calibration_metrics": lr_cal,
            "params": lr_params,
            "metrics": lr_metrics,
        }
        lr_meta_path = output_dir / "ensemble_policy_lr.meta.json"
        _write_json(lr_meta_path, lr_meta)
        candidates["lreg"] = {
            "variant": "lreg",
            "model_path": str(lr_path.relative_to(job_dir)),
            "meta_path": str(lr_meta_path.relative_to(job_dir)),
            "metrics": lr_metrics,
            "compare_to_baseline": _compare_to_baseline(lr_metrics, baseline_metrics),
        }

    if args.variant in {"bakeoff", "xgb"}:
        xgb_model, xgb_params = _fit_xgb_candidate(
            train_X,
            train_y,
            train_labels,
            train_groups,
            seed=int(args.seed),
            target_fp_ratio=float(args.target_fp_ratio),
            optimize=str(args.optimize),
            min_recall=float(args.min_recall),
            steps=int(args.threshold_steps),
        )
        train_probs_xgb = np.asarray(xgb_model.predict(xgb.DMatrix(train_X)), dtype=np.float32)
        xgb_default_threshold, xgb_thresholds, xgb_cal = _calibrate_thresholds(
            train_probs_xgb,
            train_y,
            train_labels,
            target_fp_ratio=float(args.target_fp_ratio),
            optimize=str(args.optimize),
            min_recall=float(args.min_recall),
            steps=int(args.threshold_steps),
        )
        val_probs_xgb = np.asarray(xgb_model.predict(xgb.DMatrix(val_X)), dtype=np.float32)
        xgb_metrics = _evaluate_probs(
            val_probs_xgb,
            val_y,
            val_labels,
            val_bundle["subgroup_flags"],
            default_threshold=xgb_default_threshold,
            thresholds_by_label=xgb_thresholds,
        )
        xgb_path = output_dir / "ensemble_policy_xgb.json"
        xgb_model.save_model(str(xgb_path))
        xgb_meta = {
            "version": POLICY_LAYER_VERSION,
            "variant": "xgb",
            "feature_schema_hash": schema_hash,
            "feature_names": train_bundle["feature_names"],
            "anchor_similarity": train_bundle.get("anchor_similarity"),
            "calibrated_threshold": xgb_default_threshold,
            "calibrated_thresholds": xgb_thresholds,
            "calibration_metrics": xgb_cal,
            "params": xgb_params,
            "metrics": xgb_metrics,
        }
        xgb_meta_path = output_dir / "ensemble_policy_xgb.meta.json"
        _write_json(xgb_meta_path, xgb_meta)
        candidates["xgb"] = {
            "variant": "xgb",
            "model_path": str(xgb_path.relative_to(job_dir)),
            "meta_path": str(xgb_meta_path.relative_to(job_dir)),
            "metrics": xgb_metrics,
            "compare_to_baseline": _compare_to_baseline(xgb_metrics, baseline_metrics),
        }

    if not candidates:
        raise SystemExit("policy_layer_no_candidates_trained")

    if args.variant == "lreg":
        selected_key = "lreg"
    elif args.variant == "xgb":
        selected_key = "xgb"
    else:
        selected_key = _select_policy_candidate(candidates)

    selected = candidates[selected_key]
    selection_payload = {
        "version": POLICY_LAYER_VERSION,
        "requested_variant": str(args.variant),
        "trained_variants": sorted(candidates.keys()),
        "selected_variant": selected_key,
        "anchor_similarity": train_bundle.get("anchor_similarity"),
        "baseline": baseline_metrics,
        "candidates": candidates,
        "feature_schema_hash": schema_hash,
        "nested_folds": int(args.nested_folds),
        "seed": int(args.seed),
    }
    _write_json(output_dir / "policy_layer_selection.json", selection_payload)
    (output_dir / "policy_layer_report.md").write_text(_render_report(selection_payload), encoding="utf-8")

    base_meta["selected_policy_layer"] = {
        "version": POLICY_LAYER_VERSION,
        "requested_variant": str(args.variant),
        "variant": selected_key,
        "model_path": str((job_dir / selected["model_path"]).resolve()),
        "meta_path": str((job_dir / selected["meta_path"]).resolve()),
        "selection_path": str((output_dir / "policy_layer_selection.json").resolve()),
        "feature_schema_hash": schema_hash,
    }
    base_meta["policy_layer_summary"] = {
        "requested_variant": str(args.variant),
        "selected_variant": selected_key,
        "baseline_f1": _safe_float((baseline_metrics or {}).get("f1"), 0.0),
        "selected_f1": _safe_float((((selected.get("metrics") or {})).get("f1")), 0.0),
        "delta_vs_baseline_f1": _safe_float((((selected.get("compare_to_baseline") or {})).get("delta_f1")), 0.0),
    }
    base_meta_path.write_text(json.dumps(base_meta, indent=2), encoding="utf-8")

    print(json.dumps({
        "selected_variant": selected_key,
        "selection_path": str((output_dir / "policy_layer_selection.json").resolve()),
        "baseline": baseline_metrics,
        "selected_metrics": selected.get("metrics") or {},
    }))


if __name__ == "__main__":
    main()
