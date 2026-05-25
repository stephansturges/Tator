"""Data-ingestion diversity helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
LOCAL_VENDI_MAX_PATCHES = 256


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


def _score_percentiles(values: np.ndarray, *, mask: Optional[np.ndarray] = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=np.finfo(np.float32).min,
    )
    out = np.zeros(arr.shape[0], dtype=np.float32)
    if arr.size == 0:
        return out
    if mask is None:
        valid_indices = np.arange(arr.shape[0])
    else:
        valid_mask = np.asarray(mask, dtype=bool).reshape(-1)
        if valid_mask.shape[0] != arr.shape[0]:
            valid_mask = np.zeros(arr.shape[0], dtype=bool)
        valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        return out
    if valid_indices.size == 1:
        out[int(valid_indices[0])] = 1.0
        return out
    ordered = valid_indices[np.argsort(arr[valid_indices], kind="mergesort")]
    denom = max(1, ordered.size - 1)
    rank = 0
    while rank < ordered.size:
        end = rank + 1
        value = float(arr[int(ordered[rank])])
        while end < ordered.size and float(arr[int(ordered[end])]) == value:
            end += 1
        percentile = float(((rank + end - 1) / 2.0) / denom)
        for idx in ordered[rank:end]:
            out[int(idx)] = percentile
        rank = end
    return out


def score_percentiles(values: Sequence[float]) -> np.ndarray:
    """Return 0..1 percentiles where larger input values receive higher scores."""

    return _score_percentiles(np.asarray(values, dtype=np.float32))


def local_vendi_metric(
    patch_tokens: np.ndarray,
    *,
    max_patches: int = LOCAL_VENDI_MAX_PATCHES,
) -> Dict[str, float]:
    """Return a Vendi-style local diversity metric for one image's patch tokens."""

    tokens = np.asarray(patch_tokens, dtype=np.float32)
    if tokens.ndim != 2 or tokens.shape[0] <= 0 or tokens.shape[1] <= 0:
        return {
            "local_vendi_score": 0.0,
            "local_vendi_effective_patches": 0.0,
            "local_vendi_patch_count": 0.0,
            "local_vendi_used_patch_count": 0.0,
        }
    tokens = np.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
    patch_count = int(tokens.shape[0])
    cap = max(1, int(max_patches or LOCAL_VENDI_MAX_PATCHES))
    if patch_count > cap:
        keep = np.linspace(0, patch_count - 1, cap, dtype=np.int64)
        tokens = tokens[keep]
    used_count = int(tokens.shape[0])
    if used_count <= 1:
        return {
            "local_vendi_score": 0.0,
            "local_vendi_effective_patches": 1.0 if used_count else 0.0,
            "local_vendi_patch_count": float(patch_count),
            "local_vendi_used_patch_count": float(used_count),
        }
    normalized = normalize_rows(tokens)
    kernel = normalized @ normalized.T
    weighted = kernel / float(used_count)
    eigvals = np.linalg.eigvalsh(weighted.astype(np.float64, copy=False))
    eigvals = np.clip(eigvals, 0.0, None)
    total = float(eigvals.sum())
    if total <= 1e-12:
        effective = 1.0
    else:
        probs = eigvals / total
        probs = probs[probs > 1e-12]
        entropy = float(-(probs * np.log(probs)).sum())
        effective = float(np.exp(entropy))
    score = 0.0 if used_count <= 1 else float(np.log(max(effective, 1.0)) / np.log(float(used_count)))
    return {
        "local_vendi_score": float(max(0.0, min(1.0, score))),
        "local_vendi_effective_patches": float(effective),
        "local_vendi_patch_count": float(patch_count),
        "local_vendi_used_patch_count": float(used_count),
    }


def local_vendi_metrics_from_patch_tokens(
    patch_tokens: np.ndarray,
    *,
    max_patches: int = LOCAL_VENDI_MAX_PATCHES,
) -> List[Dict[str, float]]:
    tokens = np.asarray(patch_tokens, dtype=np.float32)
    if tokens.ndim != 3:
        return []
    return [local_vendi_metric(sample, max_patches=max_patches) for sample in tokens]


def greedy_diverse_indices(
    embeddings: np.ndarray,
    *,
    keep_fraction: float,
    reference_embeddings: Optional[np.ndarray] = None,
) -> Tuple[List[int], np.ndarray]:
    """Return farthest-first diverse indices and per-item novelty scores."""

    selected, novelty_scores, _coverage_scores = greedy_diverse_indices_with_scores(
        embeddings,
        keep_fraction=keep_fraction,
        reference_embeddings=reference_embeddings,
    )
    return selected, novelty_scores


def greedy_diverse_indices_with_scores(
    embeddings: np.ndarray,
    *,
    keep_fraction: float,
    reference_embeddings: Optional[np.ndarray] = None,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Return selected indices, reference novelty, and coverage scores.

    The reference novelty score is the nearest-reference distance. The coverage
    score is the distance to the closest already-covered item at the moment an
    item is selected; for skipped items it is their final distance to the
    reference/selected coverage set.
    """

    emb = normalize_rows(embeddings)
    total = int(emb.shape[0])
    if total == 0:
        empty = np.empty((0,), dtype=np.float32)
        return [], empty, empty
    keep_count = max(1, min(total, int(math.ceil(total * normalize_keep_fraction(keep_fraction)))))
    if reference_embeddings is not None and np.asarray(reference_embeddings).size:
        ref = normalize_rows(np.asarray(reference_embeddings, dtype=np.float32))
        min_distance = cosine_distance_matrix(emb, ref).min(axis=1)
    else:
        centroid = normalize_rows(emb.mean(axis=0, keepdims=True))
        min_distance = cosine_distance_matrix(emb, centroid).reshape(-1)
    selected: List[int] = []
    candidate_scores = min_distance.astype(np.float32, copy=True)
    coverage_scores = np.zeros(total, dtype=np.float32)
    remaining = np.ones(total, dtype=bool)
    for _ in range(keep_count):
        masked = np.where(remaining, candidate_scores, -1.0)
        idx = int(np.argmax(masked))
        if not remaining[idx]:
            break
        selected.append(idx)
        coverage_scores[idx] = float(candidate_scores[idx])
        remaining[idx] = False
        if len(selected) >= keep_count:
            break
        distances_to_new = cosine_distance_matrix(emb, emb[idx : idx + 1]).reshape(-1)
        candidate_scores = np.minimum(candidate_scores, distances_to_new)
    coverage_scores[remaining] = candidate_scores[remaining]
    return selected, min_distance.astype(np.float32, copy=False), coverage_scores.astype(np.float32, copy=False)


def greedy_diverse_indices_with_local_scores(
    embeddings: np.ndarray,
    *,
    keep_fraction: float,
    reference_embeddings: Optional[np.ndarray] = None,
    local_scores: Optional[Sequence[float]] = None,
    local_weight: float = 0.0,
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    """Return diverse indices plus raw coverage and final selection scores."""

    try:
        weight = float(local_weight)
    except (TypeError, ValueError):
        weight = 0.0
    if not math.isfinite(weight):
        weight = 0.0
    weight = max(0.0, min(1.0, weight))
    emb = normalize_rows(embeddings)
    total = int(emb.shape[0])
    local_arr = np.asarray([] if local_scores is None else local_scores, dtype=np.float32).reshape(-1)
    if total == 0:
        empty = np.empty((0,), dtype=np.float32)
        return [], empty, empty, empty
    if weight <= 0.0 or local_arr.shape[0] != total:
        selected, novelty_scores, coverage_scores = greedy_diverse_indices_with_scores(
            emb,
            keep_fraction=keep_fraction,
            reference_embeddings=reference_embeddings,
        )
        return selected, novelty_scores, coverage_scores, coverage_scores.astype(np.float32, copy=True)
    keep_count = max(1, min(total, int(math.ceil(total * normalize_keep_fraction(keep_fraction)))))
    if reference_embeddings is not None and np.asarray(reference_embeddings).size:
        ref = normalize_rows(np.asarray(reference_embeddings, dtype=np.float32))
        min_distance = cosine_distance_matrix(emb, ref).min(axis=1)
    else:
        centroid = normalize_rows(emb.mean(axis=0, keepdims=True))
        min_distance = cosine_distance_matrix(emb, centroid).reshape(-1)
    local_percentiles = _score_percentiles(local_arr)
    selected: List[int] = []
    candidate_scores = min_distance.astype(np.float32, copy=True)
    coverage_scores = np.zeros(total, dtype=np.float32)
    selection_scores = np.zeros(total, dtype=np.float32)
    remaining = np.ones(total, dtype=bool)
    for _ in range(keep_count):
        coverage_percentiles = _score_percentiles(candidate_scores, mask=remaining)
        combined = (1.0 - weight) * coverage_percentiles + weight * local_percentiles
        masked = np.where(remaining, combined, -1.0)
        idx = int(np.argmax(masked))
        if not remaining[idx]:
            break
        selected.append(idx)
        coverage_scores[idx] = float(candidate_scores[idx])
        selection_scores[idx] = float(combined[idx])
        remaining[idx] = False
        if len(selected) >= keep_count:
            break
        distances_to_new = cosine_distance_matrix(emb, emb[idx : idx + 1]).reshape(-1)
        candidate_scores = np.minimum(candidate_scores, distances_to_new)
    coverage_scores[remaining] = candidate_scores[remaining]
    if np.any(remaining):
        final_coverage_percentiles = _score_percentiles(candidate_scores, mask=remaining)
        final_combined = (1.0 - weight) * final_coverage_percentiles + weight * local_percentiles
        selection_scores[remaining] = final_combined[remaining]
    return (
        selected,
        min_distance.astype(np.float32, copy=False),
        coverage_scores.astype(np.float32, copy=False),
        selection_scores.astype(np.float32, copy=False),
    )


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
