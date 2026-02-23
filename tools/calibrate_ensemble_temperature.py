#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F


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


def _load_model(model_path: Path) -> torch.nn.Module:
    blob = torch.load(model_path, map_location="cpu")
    input_dim = int(blob.get("input_dim") or 0)
    hidden = blob.get("hidden") or []
    dropout = float(blob.get("dropout") or 0.0)
    state_dict = blob["state_dict"]
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
    return model


def _normalize_features(X: np.ndarray, meta: Dict[str, Any], feature_names: List[str]) -> np.ndarray:
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


def _nll_loss(logits: np.ndarray, y_true: np.ndarray, temperature: float) -> float:
    logits_t = torch.tensor(logits / max(float(temperature), 1e-6), dtype=torch.float32)
    targets_t = torch.tensor(y_true, dtype=torch.float32)
    return float(F.binary_cross_entropy_with_logits(logits_t, targets_t).item())


def _f1_from_logits(logits: np.ndarray, y_true: np.ndarray, temperature: float) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-(logits / max(float(temperature), 1e-6))))
    pred = (probs >= 0.5).astype(np.int64)
    y_int = y_true.astype(np.int64)
    tp = int(((y_int == 1) & (pred == 1)).sum())
    fp = int(((y_int == 0) & (pred == 1)).sum())
    fn = int(((y_int == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _select_temperature(
    logits: np.ndarray,
    y_true: np.ndarray,
    *,
    min_temp: float,
    max_temp: float,
    steps: int,
    objective: str,
) -> Dict[str, Any]:
    lo = max(float(min_temp), 1e-3)
    hi = max(float(max_temp), lo + 1e-6)
    temps = np.exp(np.linspace(np.log(lo), np.log(hi), max(int(steps), 2)))
    best: Dict[str, Any] = {}
    for temp in temps:
        nll = _nll_loss(logits, y_true, float(temp))
        f1_metrics = _f1_from_logits(logits, y_true, float(temp))
        if not best:
            best = {
                "temperature": float(temp),
                "nll": float(nll),
                "f1@0.5": float(f1_metrics["f1"]),
                "precision@0.5": float(f1_metrics["precision"]),
                "recall@0.5": float(f1_metrics["recall"]),
            }
            continue
        if objective == "f1":
            if (
                f1_metrics["f1"] > best["f1@0.5"]
                or (f1_metrics["f1"] == best["f1@0.5"] and nll < best["nll"])
            ):
                best = {
                    "temperature": float(temp),
                    "nll": float(nll),
                    "f1@0.5": float(f1_metrics["f1"]),
                    "precision@0.5": float(f1_metrics["precision"]),
                    "recall@0.5": float(f1_metrics["recall"]),
                }
        else:
            if nll < best["nll"]:
                best = {
                    "temperature": float(temp),
                    "nll": float(nll),
                    "f1@0.5": float(f1_metrics["f1"]),
                    "precision@0.5": float(f1_metrics["precision"]),
                    "recall@0.5": float(f1_metrics["recall"]),
                }
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a scalar temperature for MLP logits.")
    parser.add_argument("--model", required=True, help="Model .pt path.")
    parser.add_argument("--data", required=True, help="Labeled .npz data.")
    parser.add_argument("--meta", required=True, help="Model meta json to update.")
    parser.add_argument("--min-temp", type=float, default=0.5, help="Minimum temperature.")
    parser.add_argument("--max-temp", type=float, default=3.0, help="Maximum temperature.")
    parser.add_argument("--steps", type=int, default=200, help="Grid size over log-temperature space.")
    parser.add_argument("--objective", default="nll", choices=["nll", "f1"], help="Selection objective.")
    parser.add_argument("--use-val-split", action="store_true", help="Restrict fitting to split_val_images.")
    args = parser.parse_args()

    model_path = Path(args.model)
    meta_path = Path(args.meta)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    data = np.load(args.data, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    feature_names = [str(name) for name in data.get("feature_names", [])]
    meta_rows = list(data["meta"]) if "meta" in data else []
    parsed_meta: List[Dict[str, Any]] = []
    for row in meta_rows:
        try:
            parsed_meta.append(json.loads(str(row)))
        except json.JSONDecodeError:
            parsed_meta.append({})

    X = _normalize_features(X, meta, feature_names)
    model = _load_model(model_path)
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).numpy().reshape(-1)

    if args.use_val_split and parsed_meta:
        val_images = set(meta.get("split_val_images") or [])
        if val_images:
            keep_mask = np.asarray(
                [str(row.get("image") or "") in val_images for row in parsed_meta],
                dtype=bool,
            )
            if keep_mask.any():
                logits = logits[keep_mask]
                y = y[keep_mask]

    best = _select_temperature(
        logits,
        y,
        min_temp=float(args.min_temp),
        max_temp=float(args.max_temp),
        steps=int(args.steps),
        objective=str(args.objective),
    )

    meta["calibrated_temperature"] = float(best["temperature"])
    meta["temperature_calibration"] = {
        "objective": str(args.objective),
        "min_temp": float(args.min_temp),
        "max_temp": float(args.max_temp),
        "steps": int(args.steps),
        "metrics": best,
        "samples": int(len(y)),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["temperature_calibration"], indent=2))


if __name__ == "__main__":
    main()
