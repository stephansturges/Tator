#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


def _load_model(model_path: Path):
    payload = torch.load(model_path, map_location="cpu")
    return payload


class _MLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden: List[int], dropout: float) -> None:
        super().__init__()
        layers: List[torch.nn.Module] = []
        last = input_dim
        for width in hidden:
            layers.append(torch.nn.Linear(last, width))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            last = width
        layers.append(torch.nn.Linear(last, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


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
    tp = int(((y_true == 1) & (preds == 1)).sum())
    fp = int(((y_true == 0) & (preds == 1)).sum())
    fn = int(((y_true == 1) & (preds == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fp_ratio = fp / tp if tp else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_ratio": fp_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Relax per-class thresholds with FP/TP cap.")
    parser.add_argument("--model", required=True, help="Model .pt path.")
    parser.add_argument("--data", required=True, help="Labeled .npz data.")
    parser.add_argument("--meta", required=True, help="Model meta json to update.")
    parser.add_argument("--fp-ratio-cap", type=float, default=0.2, help="Max FP/TP ratio per class.")
    parser.add_argument("--global-fp-cap", type=float, default=0.2, help="Max FP/TP ratio globally.")
    parser.add_argument("--smooth-alpha", type=float, default=0.2, help="Blend per-class threshold toward global.")
    parser.add_argument("--smooth-step", type=float, default=0.05, help="Step size when increasing alpha.")
    args = parser.parse_args()

    model_blob = _load_model(Path(args.model))
    input_dim = int(model_blob.get("input_dim") or 0)
    hidden = model_blob.get("hidden") or []
    dropout = float(model_blob.get("dropout") or 0.0)

    state_dict = model_blob["state_dict"]
    if any(key.startswith("net.") for key in state_dict):
        model = _MLP(input_dim, hidden, dropout)
    else:
        layers: List[torch.nn.Module] = []
        last = input_dim
        for width in hidden:
            layers.append(torch.nn.Linear(last, width))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            last = width
        layers.append(torch.nn.Linear(last, 1))
        model = torch.nn.Sequential(*layers)
    model.load_state_dict(state_dict)
    model.eval()

    data = np.load(args.data, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    meta_rows = data["meta"] if "meta" in data else None
    if meta_rows is None:
        raise SystemExit("Missing meta rows for per-class relaxation.")

    meta_path = Path(args.meta)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    val_images = set(meta.get("split_val_images") or [])
    if val_images:
        keep_mask = []
        for row in meta_rows:
            try:
                payload = json.loads(str(row))
            except json.JSONDecodeError:
                payload = {}
            keep_mask.append(str(payload.get("image") or "") in val_images)
        mask = np.asarray(keep_mask, dtype=bool)
        if mask.any():
            X = X[mask]
            y = y[mask]
            meta_rows = meta_rows[mask]

    labels: List[str] = []
    for row in meta_rows:
        try:
            payload = json.loads(str(row))
        except json.JSONDecodeError:
            payload = {}
        labels.append(str(payload.get("label") or "").strip())

    with torch.no_grad():
        logits = model(torch.tensor(X)).numpy().reshape(-1)
    probs = _sigmoid(logits)

    thresholds_by_label = meta.get("calibrated_thresholds") if isinstance(meta.get("calibrated_thresholds"), dict) else {}
    default_threshold = float(meta.get("calibrated_threshold") or 0.5)

    relaxed: Dict[str, float] = {}
    metrics_by_label: Dict[str, Optional[dict]] = {}
    unique_labels = sorted({lbl for lbl in labels if lbl})
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

    # optional smoothing toward global threshold
    alpha = max(0.0, min(1.0, float(args.smooth_alpha)))
    smoothed: Dict[str, float] = {}
    if relaxed:
        for label, thr in relaxed.items():
            smoothed[label] = alpha * default_threshold + (1.0 - alpha) * float(thr)

    # enforce global FP/TP cap if needed by increasing alpha
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
