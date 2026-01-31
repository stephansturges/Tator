#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _normalize_features(X: np.ndarray, meta: dict, feature_names: List[str]) -> np.ndarray:
    mean = meta.get("feature_mean")
    std = meta.get("feature_std")
    if mean is None or std is None:
        return X
    if feature_names:
        count_tokens = ("count", "support_count", "sam3_text_count", "sam3_sim_count")
        for idx, name in enumerate(feature_names):
            if any(token in name for token in count_tokens):
                X[:, idx] = np.log1p(np.maximum(X[:, idx], 0.0))
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
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidates with ensemble MLP.")
    parser.add_argument("--model", required=True, help="Model .pt path.")
    parser.add_argument("--meta", required=True, help="Model meta json.")
    parser.add_argument("--data", required=True, help="Input .npz data.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold.")
    args = parser.parse_args()

    model_blob = torch.load(Path(args.model), map_location="cpu")
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
    y = data["y"] if "y" in data else None
    feature_names = [str(name) for name in data.get("feature_names", [])]

    X = _normalize_features(X, meta, feature_names)
    with torch.no_grad():
        logits = model(torch.tensor(X)).numpy().reshape(-1)
    probs = _sigmoid(logits)
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

    if y is not None:
        metrics = _compute_metrics(y.astype(np.int64), preds)
        metrics["threshold"] = float(threshold_override) if threshold_override is not None else float(default_threshold)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
