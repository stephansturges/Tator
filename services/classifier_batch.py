"""Classifier batch inference helpers."""

from __future__ import annotations

import os
from typing import Callable, Optional, Sequence

import numpy as np
from PIL import Image


def _resolve_classifier_batch_size(env_key: str = "AGENT_CLASSIFIER_BATCH_SIZE") -> int:
    raw = os.environ.get(env_key)
    try:
        if raw is not None and str(raw).strip():
            return max(1, min(int(raw), 512))
    except Exception:
        pass
    return 64


def _predict_proba_batched(
    crops: Sequence[Image.Image],
    head: dict,
    *,
    batch_size: int,
    encode_batch_fn: Callable[[Sequence[Image.Image], dict, int], Optional[np.ndarray]],
    predict_proba_fn: Callable[[np.ndarray, dict], Optional[np.ndarray]],
    empty_cache_fn: Optional[Callable[[], None]] = None,
) -> Optional[np.ndarray]:
    if not crops:
        return None
    results: list[np.ndarray] = []
    idx = 0
    bs = max(1, int(batch_size))
    while idx < len(crops):
        chunk = list(crops[idx : idx + bs])
        feats = encode_batch_fn(chunk, head, bs)
        if feats is None:
            if bs > 1:
                bs = max(1, bs // 2)
                if empty_cache_fn is not None:
                    try:
                        empty_cache_fn()
                    except Exception:
                        pass
                continue
            return None
        proba = predict_proba_fn(feats, head)
        if proba is None or proba.ndim != 2 or proba.shape[0] != len(chunk):
            if bs > 1:
                bs = max(1, bs // 2)
                if empty_cache_fn is not None:
                    try:
                        empty_cache_fn()
                    except Exception:
                        pass
                continue
            return None
        results.append(proba)
        idx += len(chunk)
    if not results:
        return None
    return np.vstack(results)
