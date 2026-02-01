#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import xgboost as xgb

from tools import train_ensemble_mlp as tem


def _apply_log1p_counts(X: np.ndarray, feature_names: List[str]) -> np.ndarray:
    if not feature_names:
        return X
    count_tokens = ("count", "support_count", "sam3_text_count", "sam3_sim_count")
    for idx, name in enumerate(feature_names):
        if any(token in name for token in count_tokens):
            X[:, idx] = np.log1p(np.maximum(X[:, idx], 0.0))
    return X


def _standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(std < 1e-6, 1.0, std)
    return (X - mean) / std


def _select_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
    target_fp_ratio: float,
    optimize: str,
    min_recall: float,
) -> Optional[dict]:
    best = None
    for thr in thresholds:
        pred = (probs >= thr).astype(np.int64)
        tp = int(((y_true == 1) & (pred == 1)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        fn = int(((y_true == 1) & (pred == 0)).sum())
        if tp == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if recall < min_recall:
            continue
        if fp > target_fp_ratio * tp:
            continue
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        metrics = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
        if best is None:
            best = {**metrics, "threshold": float(thr)}
            continue
        if optimize == "recall":
            if recall > best["recall"]:
                best = {**metrics, "threshold": float(thr)}
        elif optimize == "tp":
            if tp > best["tp"] or (tp == best["tp"] and f1 > best["f1"]):
                best = {**metrics, "threshold": float(thr)}
        else:
            if f1 > best["f1"]:
                best = {**metrics, "threshold": float(thr)}
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost accept/reject model.")
    parser.add_argument("--input", required=True, help="Input labeled .npz file.")
    parser.add_argument("--output", required=True, help="Output model prefix (no extension).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--max-depth", type=int, default=8, help="Max tree depth.")
    parser.add_argument("--n-estimators", type=int, default=600, help="Boosting rounds.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample ratio.")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, help="Column sample per tree.")
    parser.add_argument("--min-child-weight", type=float, default=1.0, help="Min child weight.")
    parser.add_argument("--gamma", type=float, default=0.0, help="Gamma (min split loss).")
    parser.add_argument("--reg-lambda", type=float, default=1.0, help="L2 regularization.")
    parser.add_argument("--reg-alpha", type=float, default=0.0, help="L1 regularization.")
    parser.add_argument("--scale-pos-weight", type=float, default=None, help="Positive class weight.")
    parser.add_argument("--tree-method", default="hist", help="Tree method (hist/gpu_hist).")
    parser.add_argument("--max-bin", type=int, default=256, help="Max bin for hist.")
    parser.add_argument("--early-stopping-rounds", type=int, default=50, help="Early stopping rounds.")
    parser.add_argument("--log1p-counts", action="store_true", help="Apply log1p to count features.")
    parser.add_argument("--standardize", action="store_true", help="Z-score standardize features.")
    parser.add_argument("--target-fp-ratio", type=float, default=0.2, help="Max FP/TP ratio.")
    parser.add_argument("--min-recall", type=float, default=0.6, help="Minimum recall floor.")
    parser.add_argument("--threshold-steps", type=int, default=200, help="Calibration threshold steps.")
    parser.add_argument("--per-class", action="store_true", help="Calibrate per-class thresholds.")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    meta_raw = list(data["meta"])
    feature_names = [str(name) for name in data.get("feature_names", [])]
    meta_rows = [json.loads(str(row)) for row in meta_raw]

    split = tem._split_by_image(meta_rows, seed=int(args.seed), val_ratio=float(args.val_ratio))
    train_idx = split["train"]
    val_idx = split["val"]
    if not train_idx or not val_idx:
        raise SystemExit("Empty train/val split.")

    if args.log1p_counts:
        X = _apply_log1p_counts(X, feature_names)

    mean = None
    std = None
    if args.standardize:
        mean = X[train_idx].mean(axis=0)
        std = X[train_idx].std(axis=0)
        X = _standardize(X, mean, std)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    scale_pos_weight = args.scale_pos_weight
    if scale_pos_weight is None:
        pos = max(1, int((y_train == 1).sum()))
        neg = max(1, int((y_train == 0).sum()))
        scale_pos_weight = float(neg) / float(pos)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": int(args.max_depth),
        "eta": float(args.learning_rate),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "min_child_weight": float(args.min_child_weight),
        "gamma": float(args.gamma),
        "lambda": float(args.reg_lambda),
        "alpha": float(args.reg_alpha),
        "scale_pos_weight": float(scale_pos_weight),
        "tree_method": str(args.tree_method),
        "max_bin": int(args.max_bin),
        "seed": int(args.seed),
    }
    evals = [(dtrain, "train"), (dval, "val")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=int(args.n_estimators),
        evals=evals,
        early_stopping_rounds=int(args.early_stopping_rounds),
        verbose_eval=False,
    )

    # calibration thresholds on val
    probs_val = model.predict(dval)
    thresholds = np.linspace(0.0, 1.0, int(args.threshold_steps))
    best_thr = _select_threshold(
        probs_val,
        y_val,
        thresholds,
        float(args.target_fp_ratio),
        "f1",
        float(args.min_recall),
    )
    if best_thr is None:
        best_thr = _select_threshold(
            probs_val,
            y_val,
            thresholds,
            float(args.target_fp_ratio),
            "recall",
            0.0,
        )
    calibrated_threshold = float(best_thr["threshold"]) if best_thr else 0.5

    thresholds_by_label: Dict[str, float] = {}
    if args.per_class:
        labels_val = [meta_rows[idx].get("label", "") for idx in val_idx]
        unique_labels = sorted({str(lbl).strip() for lbl in labels_val if str(lbl).strip()})
        for lbl in unique_labels:
            mask = np.array([str(x).strip() == lbl for x in labels_val], dtype=bool)
            if not mask.any():
                continue
            best_label = _select_threshold(
                probs_val[mask],
                y_val[mask],
                thresholds,
                float(args.target_fp_ratio),
                "f1",
                float(args.min_recall),
            )
            if best_label is None:
                best_label = _select_threshold(
                    probs_val[mask],
                    y_val[mask],
                    thresholds,
                    float(args.target_fp_ratio),
                    "recall",
                    0.0,
                )
            if best_label is None:
                continue
            thresholds_by_label[lbl] = float(best_label["threshold"])

    model_path = Path(args.output).with_suffix(".json")
    model.save_model(str(model_path))
    meta_path = Path(args.output).with_suffix(".meta.json")
    meta_out = {
        "model_path": str(model_path),
        "calibrated_threshold": calibrated_threshold,
        "calibration_metrics": best_thr,
        "calibrated_thresholds": thresholds_by_label,
        "feature_mean": mean.tolist() if mean is not None else None,
        "feature_std": std.tolist() if std is not None else None,
        "log1p_counts": bool(args.log1p_counts),
        "standardize": bool(args.standardize),
        "split_seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "split_val_images": sorted({meta_rows[idx].get("image") for idx in val_idx if meta_rows[idx].get("image")}),
        "split_train_images": [],
        "xgb_params": params,
        "n_estimators": int(args.n_estimators),
        "best_iteration": int(model.best_iteration or 0),
    }
    meta_path.write_text(json.dumps(meta_out, indent=2))


if __name__ == "__main__":
    main()
