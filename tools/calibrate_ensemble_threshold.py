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


def _resolve_temperature(meta: Dict[str, object], override: Optional[float]) -> float:
    if override is not None:
        temp = float(override)
    else:
        temp = float(meta.get("calibrated_temperature") or 1.0)
    if not np.isfinite(temp) or temp <= 0.0:
        return 1.0
    return float(temp)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


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
        metrics = _compute_metrics(y_true, pred)
        if metrics["tp"] == 0:
            continue
        if metrics["recall"] < min_recall:
            continue
        if metrics["fp"] > target_fp_ratio * metrics["tp"]:
            continue
        if best is None:
            best = {**metrics, "threshold": float(thr)}
            continue
        if optimize == "recall":
            if metrics["recall"] > best["recall"]:
                best = {**metrics, "threshold": float(thr)}
        elif optimize == "tp":
            if metrics["tp"] > best["tp"] or (
                metrics["tp"] == best["tp"] and metrics["f1"] > best["f1"]
            ):
                best = {**metrics, "threshold": float(thr)}
        else:
            if metrics["f1"] > best["f1"]:
                best = {**metrics, "threshold": float(thr)}
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate ensemble threshold.")
    parser.add_argument("--model", required=True, help="Model .pt path.")
    parser.add_argument("--data", required=True, help="Labeled .npz data.")
    parser.add_argument("--meta", required=True, help="Model meta json to update.")
    parser.add_argument("--target-fp-ratio", type=float, default=0.1, help="Max FP/TP ratio.")
    parser.add_argument("--min-recall", type=float, default=0.0, help="Minimum recall floor for calibration.")
    parser.add_argument("--min-threshold", type=float, default=0.0, help="Min threshold.")
    parser.add_argument("--max-threshold", type=float, default=1.0, help="Max threshold.")
    parser.add_argument("--steps", type=int, default=200, help="Threshold steps.")
    parser.add_argument("--per-class", action="store_true", help="Calibrate per-label thresholds.")
    parser.add_argument("--optimize", default="f1", choices=["f1", "tp", "recall"], help="Optimization metric.")
    parser.add_argument("--temperature", type=float, default=None, help="Override logit temperature.")
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
    meta_path = Path(args.meta)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    with torch.no_grad():
        logits = model(torch.tensor(X)).numpy().reshape(-1)
    temperature = _resolve_temperature(meta, args.temperature)
    probs = _sigmoid(logits / max(temperature, 1e-6))
    meta["calibrated_temperature"] = float(temperature)
    val_images = set(meta.get("split_val_images") or [])
    if val_images:
        meta_rows = data["meta"] if "meta" in data else None
        if meta_rows is not None:
            keep_mask = []
            for row in meta_rows:
                try:
                    payload = json.loads(str(row))
                except json.JSONDecodeError:
                    payload = {}
                keep_mask.append(str(payload.get("image") or "") in val_images)
            mask = np.asarray(keep_mask, dtype=bool)
            if mask.any():
                probs = probs[mask]
                y = y[mask]
                meta_rows = meta_rows[mask]
        else:
            meta_rows = None
    else:
        meta_rows = data["meta"] if "meta" in data else None

    thresholds = np.linspace(args.min_threshold, args.max_threshold, args.steps)
    best = _select_threshold(
        probs,
        y,
        thresholds,
        args.target_fp_ratio,
        args.optimize,
        float(args.min_recall),
    )
    fallback_used = None
    if best is None and float(args.min_recall) > 0.0:
        best = _select_threshold(
            probs,
            y,
            thresholds,
            args.target_fp_ratio,
            "recall",
            0.0,
        )
        if best is not None:
            fallback_used = "recall"
    meta["calibration_min_recall"] = float(args.min_recall)
    if fallback_used:
        meta["calibration_fallback"] = fallback_used
    else:
        meta["calibration_fallback"] = None
    if best is not None:
        meta["calibrated_threshold"] = best["threshold"]
        meta["calibration_metrics"] = best
    else:
        meta["calibrated_threshold"] = float(args.max_threshold)
        meta["calibration_metrics"] = None

    if args.per_class:
        if meta_rows is None:
            raise SystemExit("Missing meta rows for per-class calibration.")
        labels: List[str] = []
        for row in meta_rows:
            try:
                payload = json.loads(str(row))
            except json.JSONDecodeError:
                payload = {}
            labels.append(str(payload.get("label") or "").strip())
        thresholds_by_label: Dict[str, float] = {}
        metrics_by_label: Dict[str, Optional[dict]] = {}
        fallback_by_label: Dict[str, Optional[str]] = {}
        unique_labels = sorted({lbl for lbl in labels if lbl})
        for label in unique_labels:
            mask = np.array([lbl == label for lbl in labels], dtype=bool)
            if not mask.any():
                continue
            best_label = _select_threshold(
                probs[mask],
                y[mask],
                thresholds,
                args.target_fp_ratio,
                args.optimize,
                float(args.min_recall),
            )
            fallback_label = None
            if best_label is None and float(args.min_recall) > 0.0:
                best_label = _select_threshold(
                    probs[mask],
                    y[mask],
                    thresholds,
                    args.target_fp_ratio,
                    "recall",
                    0.0,
                )
                if best_label is not None:
                    fallback_label = "recall"
            if best_label is None:
                thresholds_by_label[label] = float(args.max_threshold)
                metrics_by_label[label] = None
                fallback_by_label[label] = fallback_label
                continue
            thresholds_by_label[label] = float(best_label["threshold"])
            metrics_by_label[label] = best_label
            fallback_by_label[label] = fallback_label
        meta["calibrated_thresholds"] = thresholds_by_label
        meta["calibration_metrics_by_label"] = metrics_by_label
        meta["calibration_fallback_by_label"] = fallback_by_label

    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta.get("calibration_metrics") or {}, indent=2))


if __name__ == "__main__":
    main()
