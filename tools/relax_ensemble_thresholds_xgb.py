#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import xgboost as xgb


def _apply_log1p_counts(X: np.ndarray, feature_names: List[str]) -> np.ndarray:
    if not feature_names:
        return X
    count_tokens = ("count", "support_count", "sam3_text_count", "sam3_sim_count")
    for idx, name in enumerate(feature_names):
        if any(token in name for token in count_tokens):
            X[:, idx] = np.log1p(np.maximum(X[:, idx], 0.0))
    return X


def _standardize(X: np.ndarray, mean: Optional[List[float]], std: Optional[List[float]]) -> np.ndarray:
    if mean is None or std is None:
        return X
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape[0] != X.shape[1] or std.shape[0] != X.shape[1]:
        return X
    std = np.where(std < 1e-6, 1.0, std)
    return (X - mean) / std


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fp_ratio = fp / tp if tp else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "fp_ratio": fp_ratio}


def _select_relaxed_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    base_threshold: float,
    fp_ratio_cap: float,
) -> Optional[dict]:
    if probs.size == 0:
        return None
    thresholds = np.unique(probs)
    thresholds = thresholds[thresholds <= base_threshold]
    if thresholds.size == 0:
        thresholds = np.asarray([base_threshold], dtype=np.float32)
    thresholds = np.sort(thresholds)[::-1]
    best = None
    for thr in thresholds[::-1]:
        pred = (probs >= thr).astype(np.int64)
        metrics = _compute_metrics(y_true, pred)
        if metrics["tp"] == 0:
            continue
        if metrics["fp_ratio"] > fp_ratio_cap:
            continue
        if best is None or metrics["recall"] > best["recall"]:
            best = {**metrics, "threshold": float(thr)}
    return best


def _compute_global_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    labels: List[str],
    thresholds: Dict[str, float],
    default_threshold: float,
) -> dict:
    preds = np.zeros_like(probs, dtype=np.int64)
    for idx, (prob, label) in enumerate(zip(probs, labels)):
        thr = thresholds.get(label, default_threshold)
        preds[idx] = 1 if prob >= thr else 0
    return _compute_metrics(y_true, preds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Relax per-class thresholds for XGBoost.")
    parser.add_argument("--model", required=True, help="Model .json path.")
    parser.add_argument("--data", required=True, help="Labeled .npz data.")
    parser.add_argument("--meta", required=True, help="Model meta json to update.")
    parser.add_argument("--fp-ratio-cap", type=float, default=0.2, help="Max FP/TP ratio per class.")
    parser.add_argument("--global-fp-cap", type=float, default=0.2, help="Max FP/TP ratio globally.")
    parser.add_argument("--smooth-alpha", type=float, default=0.2, help="Blend per-class threshold toward global.")
    parser.add_argument("--smooth-step", type=float, default=0.05, help="Step size when increasing alpha.")
    args = parser.parse_args()

    meta_path = Path(args.meta)
    meta = json.loads(meta_path.read_text())
    default_threshold = float(meta.get("calibrated_threshold") or 0.5)
    thresholds_by_label = meta.get("calibrated_thresholds") if isinstance(meta.get("calibrated_thresholds"), dict) else {}

    data = np.load(args.data, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    feature_names = [str(name) for name in data.get("feature_names", [])]
    meta_rows = [json.loads(str(row)) for row in data["meta"]]
    labels = [str(row.get("label") or "").strip() for row in meta_rows]

    if meta.get("log1p_counts"):
        X = _apply_log1p_counts(X, feature_names)
    X = _standardize(X, meta.get("feature_mean"), meta.get("feature_std"))

    booster = xgb.Booster()
    booster.load_model(str(Path(args.model)))
    probs = booster.predict(xgb.DMatrix(X))

    unique_labels = sorted({lbl for lbl in labels if lbl})
    relaxed: Dict[str, float] = {}
    metrics_by_label: Dict[str, Optional[dict]] = {}
    for label in unique_labels:
        mask = np.array([lbl == label for lbl in labels], dtype=bool)
        if not mask.any():
            continue
        base = float(thresholds_by_label.get(label, default_threshold))
        best = _select_relaxed_threshold(
            probs[mask],
            y[mask],
            base_threshold=base,
            fp_ratio_cap=float(args.fp_ratio_cap),
        )
        if best is None:
            relaxed[label] = base
            metrics_by_label[label] = None
        else:
            relaxed[label] = float(best["threshold"])
            metrics_by_label[label] = best

    meta["calibrated_thresholds_base"] = thresholds_by_label
    meta["calibrated_thresholds_relaxed"] = relaxed
    meta["relaxation_metrics_by_label"] = metrics_by_label
    meta["relaxation_fp_ratio_cap"] = float(args.fp_ratio_cap)

    alpha = max(0.0, min(1.0, float(args.smooth_alpha)))
    smoothed: Dict[str, float] = {}
    if relaxed:
        for label, thr in relaxed.items():
            smoothed[label] = alpha * default_threshold + (1.0 - alpha) * float(thr)

    global_cap = float(args.global_fp_cap)
    global_metrics = None
    if smoothed:
        global_metrics = _compute_global_metrics(probs, y, labels, smoothed, default_threshold)
        if global_metrics["fp_ratio"] > global_cap:
            step = max(0.01, float(args.smooth_step))
            best_alpha = alpha
            best_metrics = global_metrics
            probe = alpha
            while probe < 1.0:
                probe = min(1.0, probe + step)
                candidate = {
                    label: probe * default_threshold + (1.0 - probe) * float(thr)
                    for label, thr in relaxed.items()
                }
                metrics = _compute_global_metrics(probs, y, labels, candidate, default_threshold)
                if metrics["fp_ratio"] <= global_cap:
                    best_alpha = probe
                    best_metrics = metrics
                    smoothed = candidate
                    break
                best_metrics = metrics
            meta["relaxation_alpha"] = best_alpha
            global_metrics = best_metrics
        else:
            meta["relaxation_alpha"] = alpha

    if smoothed:
        meta["calibrated_thresholds_relaxed_smoothed"] = smoothed
    if global_metrics:
        meta["relaxation_global_metrics"] = global_metrics
        meta["relaxation_global_fp_cap"] = global_cap
    meta["relaxation_smooth_alpha"] = alpha
    meta_path.write_text(json.dumps(meta, indent=2))

    print(json.dumps({"labels": len(relaxed)}, indent=2))


if __name__ == "__main__":
    main()
