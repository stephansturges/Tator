#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import List

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


def _standardize(X: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    if mean is None or std is None:
        return X
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape[0] != X.shape[1] or std.shape[0] != X.shape[1]:
        return X
    std = np.where(std < 1e-6, 1.0, std)
    return (X - mean) / std


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidates with ensemble XGBoost.")
    parser.add_argument("--model", required=True, help="Model .json path.")
    parser.add_argument("--meta", required=True, help="Model meta json.")
    parser.add_argument("--data", required=True, help="Input .npz data.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold.")
    args = parser.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    thresholds_by_label = meta.get("calibrated_thresholds_relaxed_smoothed")
    if not isinstance(thresholds_by_label, dict):
        thresholds_by_label = meta.get("calibrated_thresholds_relaxed") if isinstance(meta.get("calibrated_thresholds_relaxed"), dict) else {}
    if not isinstance(thresholds_by_label, dict):
        thresholds_by_label = meta.get("calibrated_thresholds") if isinstance(meta.get("calibrated_thresholds"), dict) else {}
    default_threshold = float(meta.get("calibrated_threshold") or 0.5)
    threshold_override = float(args.threshold) if args.threshold is not None else None

    data = np.load(args.data, allow_pickle=True)
    X = data["X"].astype(np.float32)
    meta_rows = list(data["meta"])
    feature_names = [str(name) for name in data.get("feature_names", [])]

    if meta.get("log1p_counts"):
        X = _apply_log1p_counts(X, feature_names)
    X = _standardize(X, meta.get("feature_mean"), meta.get("feature_std"))

    booster = xgb.Booster()
    booster.load_model(str(Path(args.model)))
    probs = booster.predict(xgb.DMatrix(X))

    parsed_meta = []
    for row in meta_rows:
        try:
            parsed_meta.append(json.loads(str(row)))
        except json.JSONDecodeError:
            parsed_meta.append({})

    preds = np.zeros_like(probs, dtype=np.int64)
    thresholds_used = []
    for idx, payload in enumerate(parsed_meta):
        label = str(payload.get("label") or "").strip()
        if threshold_override is not None:
            thr = threshold_override
        elif label and label in thresholds_by_label:
            thr = float(thresholds_by_label[label])
        else:
            thr = default_threshold
        thresholds_used.append(float(thr))
        preds[idx] = 1 if probs[idx] >= thr else 0

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, entry in enumerate(parsed_meta):
            entry["ensemble_prob"] = float(probs[idx])
            entry["ensemble_accept"] = bool(preds[idx])
            entry["ensemble_threshold"] = float(thresholds_used[idx])
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
