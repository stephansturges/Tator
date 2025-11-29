from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Sampler, WeightedRandomSampler

from sam3_lite.config import DatasetConfig


def _compute_class_weights(counts: Dict[int, int], cfg: DatasetConfig) -> Dict[int, float]:
    if not counts:
        return {}
    weights: Dict[int, float] = {}
    strategy = (cfg.balance_strategy or "none").lower()
    total = sum(counts.values())
    beta = cfg.balance_beta
    gamma = cfg.balance_gamma
    power = cfg.balance_power
    clip_value = max(1e-6, float(cfg.balance_clip))

    for cls_id, cnt in counts.items():
        cnt = max(1, int(cnt))
        if strategy == "effective_num":
            weight = (1.0 - beta) / (1.0 - math.pow(beta, cnt))
        elif strategy == "focal":
            prob = cnt / float(total)
            weight = math.pow(max(1e-6, 1.0 - prob), gamma)
        elif strategy == "clipped_inv":
            weight = min(clip_value, math.pow(cnt, -power))
        elif strategy == "inv_sqrt":
            weight = math.pow(cnt, -power)
        else:
            weight = 1.0
        weights[cls_id] = float(weight)
    return weights


def _normalize_weights(weights: Iterable[float]) -> List[float]:
    vals = list(weights)
    if not vals:
        return vals
    max_v = max(vals)
    if max_v <= 0:
        return [1.0 for _ in vals]
    return [v / max_v for v in vals]


def compute_image_weights(image_labels: Sequence[Sequence[int]], cfg: DatasetConfig) -> Tuple[List[float], Dict[int, int]]:
    counts = Counter()
    for labels in image_labels:
        counts.update(labels)
    class_weights = _compute_class_weights(dict(counts), cfg)
    sample_weights: List[float] = []
    for labels in image_labels:
        if not labels:
            sample_weights.append(1.0)
            continue
        vals = [class_weights.get(int(l), 1.0) for l in labels]
        sample_weights.append(sum(vals) / float(len(vals)))
    return _normalize_weights(sample_weights), dict(counts)


class ClassBalancedSampler(WeightedRandomSampler):
    def __init__(self, weights: List[float], num_samples: int):
        super().__init__(weights=weights, num_samples=num_samples, replacement=True)
        self.weights_tensor = torch.tensor(weights, dtype=torch.float32)


def build_sampler(image_labels: Sequence[Sequence[int]], cfg: DatasetConfig, epoch_size: int) -> Tuple[Sampler[int] | None, Dict[str, float]]:
    if not cfg.class_balance or cfg.balance_strategy == "none":
        return None, {}
    weights, counts = compute_image_weights(image_labels, cfg)
    stats = {
        "classes": float(len(counts)),
        "min": float(min(weights) if weights else 0.0),
        "max": float(max(weights) if weights else 0.0),
        "avg": float(sum(weights) / len(weights) if weights else 0.0),
    }
    sampler = ClassBalancedSampler(weights, num_samples=max(epoch_size, len(weights)))
    return sampler, stats
