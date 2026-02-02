"""Classifier batch inference helpers."""

from __future__ import annotations

import os


def _resolve_classifier_batch_size(env_key: str = "AGENT_CLASSIFIER_BATCH_SIZE") -> int:
    raw = os.environ.get(env_key)
    try:
        if raw is not None and str(raw).strip():
            return max(1, min(int(raw), 512))
    except Exception:
        pass
    return 64

