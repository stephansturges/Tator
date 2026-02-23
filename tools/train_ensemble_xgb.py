#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import xgboost as xgb

try:
    from tools import train_ensemble_mlp as tem
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
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


def _load_fixed_val_images(path: str) -> Set[str]:
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        return set()
    try:
        payload: Sequence[Any] = json.loads(raw)
    except json.JSONDecodeError:
        payload = [line.strip() for line in raw.splitlines() if line.strip()]
    return {str(v).strip() for v in payload if str(v).strip()}


def _normalize_source_fields(row: Dict[str, Any]) -> tuple[str, Set[str]]:
    primary = str(row.get("score_source") or row.get("source") or "unknown").strip().lower() or "unknown"
    source_set: Set[str] = set()
    raw = row.get("source_list")
    if isinstance(raw, (list, tuple, set)):
        for src in raw:
            name = str(src or "").strip().lower()
            if name:
                source_set.add(name)
    elif isinstance(raw, str):
        name = raw.strip().lower()
        if name:
            source_set.add(name)
    raw_scores = row.get("score_by_source")
    if isinstance(raw_scores, dict):
        for src in raw_scores.keys():
            name = str(src or "").strip().lower()
            if name:
                source_set.add(name)
    source_set.add(primary)
    return primary, source_set


def _has_detector_support(row: Dict[str, Any]) -> bool:
    _, sources = _normalize_source_fields(row)
    return ("yolo" in sources) or ("rfdetr" in sources)


def _is_sam3_text_only(row: Dict[str, Any]) -> bool:
    primary, sources = _normalize_source_fields(row)
    if primary != "sam3_text":
        return False
    return ("yolo" not in sources) and ("rfdetr" not in sources)


def _blend_probabilities(base: np.ndarray, aux: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(max(0.0, min(1.0, alpha)))
    if alpha <= 0.0:
        return np.asarray(base, dtype=np.float32)
    if alpha >= 1.0:
        return np.asarray(aux, dtype=np.float32)
    return np.asarray((1.0 - alpha) * base + alpha * aux, dtype=np.float32)


def _train_head_model(
    *,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: List[int],
    val_idx: List[int],
    params: Dict[str, Any],
    n_estimators: int,
    early_stopping_rounds: int,
) -> Optional[xgb.Booster]:
    if len(train_idx) < 32:
        return None
    y_train = y[train_idx]
    if int((y_train == 1).sum()) == 0 or int((y_train == 0).sum()) == 0:
        return None
    dtrain = xgb.DMatrix(X[train_idx], label=y_train)
    evals = [(dtrain, "train")]
    if val_idx:
        y_val = y[val_idx]
        if int((y_val == 1).sum()) > 0 and int((y_val == 0).sum()) > 0:
            dval = xgb.DMatrix(X[val_idx], label=y_val)
            evals.append((dval, "val"))
    kwargs: Dict[str, Any] = {"verbose_eval": False}
    if len(evals) > 1:
        kwargs["early_stopping_rounds"] = int(max(5, early_stopping_rounds))
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=int(max(20, n_estimators)),
        evals=evals,
        **kwargs,
    )
    return booster


def _load_policy(path_or_json: Optional[str]) -> Dict[str, Any]:
    if not path_or_json:
        return {}
    raw = str(path_or_json).strip()
    if not raw:
        return {}
    maybe_path = Path(raw)
    payload = raw
    if maybe_path.exists() and maybe_path.is_file():
        payload = maybe_path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --policy-json payload: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("Invalid --policy-json payload: expected JSON object.")
    return parsed


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
    parser.add_argument("--optimize", default="f1", choices=["f1", "tp", "recall"], help="Calibration objective.")
    parser.add_argument("--per-class", action="store_true", help="Calibrate per-class thresholds.")
    parser.add_argument(
        "--split-head-by-support",
        action="store_true",
        help="Train split heads routed by detector support (detector-backed vs SAM-only).",
    )
    parser.add_argument(
        "--train-sam3-text-quality",
        action="store_true",
        help="Train a dedicated SAM3-text-only quality model and blend during threshold calibration.",
    )
    parser.add_argument(
        "--sam3-text-quality-alpha",
        type=float,
        default=0.35,
        help="Blend weight for SAM3-text-only quality model on primary SAM3-text candidates.",
    )
    parser.add_argument(
        "--policy-json",
        default=None,
        help="Optional policy JSON file/string to persist in model meta as ensemble_policy.",
    )
    parser.add_argument(
        "--fixed-val-images",
        help="Optional path to a JSON list or newline-separated image ids for fixed validation split.",
    )
    args = parser.parse_args()
    policy = _load_policy(args.policy_json)

    data = np.load(args.input, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    meta_raw = list(data["meta"])
    feature_names = [str(name) for name in data.get("feature_names", [])]
    meta_rows = [json.loads(str(row)) for row in meta_raw]

    fixed_val_images = (
        _load_fixed_val_images(args.fixed_val_images)
        if args.fixed_val_images
        else set()
    )
    if fixed_val_images:
        val_idx = [idx for idx, row in enumerate(meta_rows) if row.get("image") in fixed_val_images]
        train_idx = [idx for idx, row in enumerate(meta_rows) if row.get("image") not in fixed_val_images]
        if not val_idx or not train_idx:
            raise SystemExit("Fixed validation split is empty for train or val.")
    else:
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

    split_head_meta: Dict[str, Any] = {"enabled": False, "route": "detector_support", "models": {}}
    detector_head: Optional[xgb.Booster] = None
    sam_only_head: Optional[xgb.Booster] = None
    if args.split_head_by_support:
        det_train_idx = [idx for idx in train_idx if _has_detector_support(meta_rows[idx])]
        sam_train_idx = [idx for idx in train_idx if not _has_detector_support(meta_rows[idx])]
        det_val_idx = [idx for idx in val_idx if _has_detector_support(meta_rows[idx])]
        sam_val_idx = [idx for idx in val_idx if not _has_detector_support(meta_rows[idx])]

        detector_head = _train_head_model(
            X=X,
            y=y,
            train_idx=det_train_idx,
            val_idx=det_val_idx,
            params=params,
            n_estimators=int(args.n_estimators),
            early_stopping_rounds=int(args.early_stopping_rounds),
        )
        sam_only_head = _train_head_model(
            X=X,
            y=y,
            train_idx=sam_train_idx,
            val_idx=sam_val_idx,
            params=params,
            n_estimators=int(args.n_estimators),
            early_stopping_rounds=int(args.early_stopping_rounds),
        )
        split_head_meta = {
            "enabled": bool(detector_head is not None or sam_only_head is not None),
            "route": "detector_support",
            "models": {},
            "train_counts": {
                "detector_supported": len(det_train_idx),
                "sam_only": len(sam_train_idx),
            },
            "val_counts": {
                "detector_supported": len(det_val_idx),
                "sam_only": len(sam_val_idx),
            },
        }
        if detector_head is not None:
            det_path = Path(str(args.output) + ".detector.json")
            detector_head.save_model(str(det_path))
            split_head_meta["models"]["detector_supported"] = str(det_path)
        if sam_only_head is not None:
            sam_path = Path(str(args.output) + ".sam_only.json")
            sam_only_head.save_model(str(sam_path))
            split_head_meta["models"]["sam_only"] = str(sam_path)

    # calibration thresholds on val
    probs_val = np.asarray(model.predict(dval), dtype=np.float32)
    if split_head_meta.get("enabled"):
        val_support_mask = np.asarray([_has_detector_support(meta_rows[idx]) for idx in val_idx], dtype=bool)
        if detector_head is not None and val_support_mask.any():
            probs_val[val_support_mask] = detector_head.predict(xgb.DMatrix(X_val[val_support_mask]))
        if sam_only_head is not None and (~val_support_mask).any():
            probs_val[~val_support_mask] = sam_only_head.predict(xgb.DMatrix(X_val[~val_support_mask]))

    sam3_text_quality_meta: Dict[str, Any] = {"enabled": False}
    if args.train_sam3_text_quality:
        q_train_idx = [idx for idx in train_idx if _is_sam3_text_only(meta_rows[idx])]
        q_val_idx = [idx for idx in val_idx if _is_sam3_text_only(meta_rows[idx])]
        q_model = _train_head_model(
            X=X,
            y=y,
            train_idx=q_train_idx,
            val_idx=q_val_idx,
            params={**params, "max_depth": int(min(int(args.max_depth), 4))},
            n_estimators=int(min(int(args.n_estimators), 400)),
            early_stopping_rounds=int(args.early_stopping_rounds),
        )
        if q_model is not None and q_val_idx:
            q_path = Path(str(args.output) + ".sam3_text_quality.json")
            q_model.save_model(str(q_path))
            alpha = float(max(0.0, min(1.0, float(args.sam3_text_quality_alpha))))
            local_mask = np.asarray([_is_sam3_text_only(meta_rows[idx]) for idx in val_idx], dtype=bool)
            if local_mask.any():
                q_probs = np.asarray(q_model.predict(xgb.DMatrix(X_val[local_mask])), dtype=np.float32)
                probs_val[local_mask] = _blend_probabilities(probs_val[local_mask], q_probs, alpha)
            sam3_text_quality_meta = {
                "enabled": True,
                "model_path": str(q_path),
                "alpha": alpha,
                "train_count": len(q_train_idx),
                "val_count": len(q_val_idx),
            }
    thresholds = np.linspace(0.0, 1.0, int(args.threshold_steps))
    best_thr = _select_threshold(
        probs_val,
        y_val,
        thresholds,
        float(args.target_fp_ratio),
        str(args.optimize),
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
                str(args.optimize),
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
        "split_type": "fixed" if fixed_val_images else "seeded_image_split",
        "xgb_params": params,
        "calibration_optimize": str(args.optimize),
        "n_estimators": int(args.n_estimators),
        "best_iteration": int(model.best_iteration or 0),
        "split_train_images": sorted(
            {meta_rows[idx].get("image") for idx in train_idx if meta_rows[idx].get("image")}
        ),
        "ensemble_policy": policy,
        "split_head": split_head_meta,
        "sam3_text_quality": sam3_text_quality_meta,
    }
    meta_path.write_text(json.dumps(meta_out, indent=2))


if __name__ == "__main__":
    main()
