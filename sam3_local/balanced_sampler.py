from __future__ import annotations

import logging
import math
from typing import Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler

_logger = logging.getLogger(__name__)
_LOGGED_SUMMARY = False
# Smoothing power for inverse-frequency weighting; <1.0 reduces aggressiveness.
BALANCE_POWER = 0.5
# Minimum raw weight to avoid zero-probability datapoints.
MIN_RAW_WEIGHT = 1e-8


def _compute_image_weights(coco, ids: Sequence[int]) -> List[float]:
    """
    Compute per-image weights based on inverse class frequency.
    Each image weight = sum(1 / freq[c]) over its categories.
    """
    global _LOGGED_SUMMARY
    cat_counts = {}
    for ann in coco.getAnnIds():
        ann_obj = coco.loadAnns([ann])[0]
        cid = ann_obj.get("category_id")
        if cid is None:
            continue
        cat_counts[cid] = cat_counts.get(cid, 0) + 1
    weights: List[float] = []
    for img_id in ids:
        ann_ids = coco.getAnnIds(imgIds=[int(img_id)])
        anns = coco.loadAnns(ann_ids)
        cats = {ann.get("category_id") for ann in anns if ann.get("category_id") is not None}
        w = 0.0
        for cid in cats:
            freq = cat_counts.get(cid, 1)
            w += (1.0 / max(1, freq)) ** BALANCE_POWER
        # Avoid zero so every datapoint remains sampleable.
        weights.append(max(w, MIN_RAW_WEIGHT))
    total = sum(weights) or 1.0
    weights = [w / total for w in weights]
    if not _LOGGED_SUMMARY:
        _LOGGED_SUMMARY = True
        try:
            cat_items = sorted(cat_counts.items(), key=lambda kv: kv[1])
            smallest = cat_items[: min(5, len(cat_items))]
            largest = cat_items[-min(5, len(cat_items)) :] if cat_items else []
            w_min, w_max = min(weights), max(weights)
            w_avg = sum(weights) / len(weights)
            def _fmt(v: float) -> str:
                # Use scientific notation to avoid rounding small weights to 0.0000 in logs.
                return f"{v:.2e}"
            msg = (
                f"[sam3-balance] classes={len(cat_counts)} images={len(ids)} "
                f"min_w={_fmt(w_min)} avg_w={_fmt(w_avg)} max_w={_fmt(w_max)} "
                f"smallest={smallest} largest={largest}"
            )
            print(msg, flush=True)
            _logger.info(msg)
        except Exception:
            pass
    return weights


class BalancedSampler(Sampler[int]):
    """
    Weighted sampler over image ids using inverse class frequency.
    Uses replacement to keep epoch size stable.
    """

    def __init__(self, indices: Sequence[int], weights: Sequence[float], num_samples: Optional[int] = None) -> None:
        self.indices = list(indices)
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples or len(self.indices)

    def __iter__(self) -> Iterator[int]:
        sampled = torch.multinomial(self.weights, self.num_samples, replacement=True)
        for idx in sampled.tolist():
            yield self.indices[idx]

    def __len__(self) -> int:
        return self.num_samples


class DistributedBalancedSampler(DistributedSampler):
    """
    Wraps BalancedSampler for DDP.
    """

    def __init__(
        self, indices: Sequence[int], weights: Sequence[float], num_replicas: Optional[int] = None,
        rank: Optional[int] = None, replacement: bool = True, num_samples: Optional[int] = None
    ) -> None:
        super().__init__(indices, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.replacement = replacement
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.indices = list(indices)
        self.num_samples = num_samples or math.ceil(len(self.indices) / self.num_replicas)

    def __iter__(self) -> Iterator[int]:
        # Evenly split across replicas
        generator = torch.Generator()
        generator.manual_seed(self.epoch)
        sampled = torch.multinomial(self.weights, self.num_samples * self.num_replicas, replacement=True, generator=generator)
        sampled = sampled.tolist()
        # Deterministic per-rank stride
        subsampled = sampled[self.rank : self.num_replicas * self.num_samples : self.num_replicas]
        for idx in subsampled:
            yield self.indices[idx]

    def __len__(self) -> int:
        return self.num_samples
