"""Data-ingestion diversity helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def normalize_keep_fraction(value: Any, default: float = 0.2) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    if parsed > 1.0:
        parsed = parsed / 100.0
    return max(0.01, min(1.0, parsed))


def normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return (arr / norms).astype(np.float32, copy=False)


def cosine_distance_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = normalize_rows(left)
    right_norm = normalize_rows(right)
    return np.maximum(0.0, 1.0 - np.clip(left_norm @ right_norm.T, -1.0, 1.0)).astype(np.float32)


def greedy_diverse_indices(
    embeddings: np.ndarray,
    *,
    keep_fraction: float,
    reference_embeddings: Optional[np.ndarray] = None,
) -> Tuple[List[int], np.ndarray]:
    """Return farthest-first diverse indices and per-item novelty scores."""

    emb = normalize_rows(embeddings)
    total = int(emb.shape[0])
    if total == 0:
        return [], np.empty((0,), dtype=np.float32)
    keep_count = max(1, min(total, int(math.ceil(total * normalize_keep_fraction(keep_fraction)))))
    if reference_embeddings is not None and np.asarray(reference_embeddings).size:
        ref = normalize_rows(np.asarray(reference_embeddings, dtype=np.float32))
        min_distance = cosine_distance_matrix(emb, ref).min(axis=1)
    else:
        centroid = normalize_rows(emb.mean(axis=0, keepdims=True))
        min_distance = cosine_distance_matrix(emb, centroid).reshape(-1)
    selected: List[int] = []
    candidate_scores = min_distance.astype(np.float32, copy=True)
    remaining = np.ones(total, dtype=bool)
    for _ in range(keep_count):
        masked = np.where(remaining, candidate_scores, -1.0)
        idx = int(np.argmax(masked))
        if not remaining[idx]:
            break
        selected.append(idx)
        remaining[idx] = False
        if len(selected) >= keep_count:
            break
        distances_to_new = cosine_distance_matrix(emb, emb[idx : idx + 1]).reshape(-1)
        candidate_scores = np.minimum(candidate_scores, distances_to_new)
    return selected, min_distance.astype(np.float32, copy=False)


def diversity_summary(
    embeddings: np.ndarray,
    *,
    selected_indices: Sequence[int],
    reference_embeddings: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    emb = normalize_rows(embeddings)
    total = int(emb.shape[0])
    selected = [int(idx) for idx in selected_indices if 0 <= int(idx) < total]
    if total == 0:
        return {
            "item_count": 0,
            "selected_count": 0,
            "mean_nearest_neighbor_distance": 0.0,
            "mean_reference_distance": None,
        }
    if total >= 2:
        dmat = cosine_distance_matrix(emb, emb)
        np.fill_diagonal(dmat, np.inf)
        mean_nn = float(np.min(dmat, axis=1).mean())
    else:
        mean_nn = 0.0
    mean_ref: Optional[float] = None
    if reference_embeddings is not None and np.asarray(reference_embeddings).size:
        mean_ref = float(cosine_distance_matrix(emb, reference_embeddings).min(axis=1).mean())
    return {
        "item_count": total,
        "selected_count": len(selected),
        "mean_nearest_neighbor_distance": mean_nn,
        "mean_reference_distance": mean_ref,
    }


def safe_media_name(path_or_name: str, fallback: str = "media", *, max_length: int = 96) -> str:
    stem = Path(str(path_or_name or "")).stem or fallback
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in stem).strip("._-")
    if max_length > 0:
        cleaned = cleaned[:max_length].strip("._-")
    return cleaned or fallback


def iter_media_files(paths: Iterable[Path]) -> List[Path]:
    found: List[Path] = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
                    found.append(child)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
            found.append(p)
    return found
