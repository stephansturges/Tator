"""Data-ingestion diversity helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    values = arr[ordered]
    # Tied values share the mean of their ranks.
    changed = values[1:] != values[:-1]
    starts = np.flatnonzero(np.concatenate(([True], changed)))
    counts = np.diff(np.concatenate((starts, [values.size])))
    group = np.concatenate(([0], np.cumsum(changed)))
    mean_rank = starts + (counts - 1) / 2.0
    out[ordered] = (mean_rank[group] / max(1, ordered.size - 1)).astype(np.float32)
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


def usable_embedding_rows(values: np.ndarray) -> np.ndarray:
    """Rows a descriptor can be compared on: finite and not all zero.

    A blank row is at distance 1 from everything, so it would otherwise look
    like the most novel image.
    """

    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    finite = np.isfinite(arr).all(axis=1)
    norms = np.linalg.norm(np.where(np.isfinite(arr), arr, 0.0), axis=1)
    return finite & (norms > 1e-12)


def reference_spacing(reference_embeddings: np.ndarray, *, max_queries: int = 2000, chunk: int = 256) -> Optional[float]:
    """Median distance from a reference image to its nearest other reference.

    This is the cutoff for "new": an image closer than this to what is already
    covered adds less than a typical reference image did. Large reference sets
    are measured from evenly spaced rows against every row.
    """

    ref = np.asarray(reference_embeddings, dtype=np.float32)
    if ref.ndim != 2 or not ref.size:
        return None
    ref = normalize_rows(ref[usable_embedding_rows(ref)])
    total = int(ref.shape[0])
    if total < 2:
        return None
    queries = np.arange(total) if total <= max_queries else np.unique(np.linspace(0, total - 1, max_queries).astype(np.int64))
    nearest = np.empty(queries.shape[0], dtype=np.float32)
    for start in range(0, queries.shape[0], chunk):
        rows = queries[start : start + chunk]
        distances = np.maximum(0.0, 1.0 - np.clip(ref[rows] @ ref.T, -1.0, 1.0))
        distances[np.arange(rows.shape[0]), rows] = np.inf
        nearest[start : start + rows.shape[0]] = distances.min(axis=1)
    return float(np.median(nearest))


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
    initial_indices: Optional[Sequence[int]] = None,
    target_count: Optional[int] = None,
    distance_update_fn=None,
    on_select=None,
    min_coverage: Optional[float] = None,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Return selected indices, reference novelty, and coverage scores.

    The reference novelty score is the nearest-reference distance. The coverage
    score is the distance to the closest already-covered item at the moment an
    item is selected; for skipped items it is their final distance to the
    reference/selected coverage set.

    With ``min_coverage`` the keep fraction no longer sets the count: selection
    continues while the best remaining item is at least that far from what is
    covered, so it may keep every item or none.
    """

    emb = normalize_rows(embeddings)
    total = int(emb.shape[0])
    if total == 0:
        empty = np.empty((0,), dtype=np.float32)
        return [], empty, empty
    keep_count = max(1, min(total, int(math.ceil(total * normalize_keep_fraction(keep_fraction)))))
    if target_count is not None:
        if type(target_count) is not int or not 1 <= target_count <= total:
            raise ValueError('target_count_out_of_range')
        keep_count = target_count
    elif min_coverage is not None:
        keep_count = total
    if reference_embeddings is not None and np.asarray(reference_embeddings).size:
        ref = normalize_rows(np.asarray(reference_embeddings, dtype=np.float32))
        min_distance = cosine_distance_matrix(emb, ref).min(axis=1)
    else:
        centroid = normalize_rows(emb.mean(axis=0, keepdims=True))
        min_distance = cosine_distance_matrix(emb, centroid).reshape(-1)
    # Optional adapters let dataset pruning use this same retained-prefix loop
    # with protected examples and task-aware distances. Default ingestion
    # callers keep the original initialization, ordering and return values.
    initial = list(dict.fromkeys(int(i) for i in (initial_indices or [])))
    if any(i < 0 or i >= total for i in initial):
        raise ValueError("initial_indices_out_of_range")
    keep_count = max(keep_count, len(initial))
    selected: List[int] = []
    if distance_update_fn is not None:
        min_distance = np.ones(total, dtype=np.float32)
    candidate_scores = min_distance.astype(np.float32, copy=True)
    coverage_scores = np.zeros(total, dtype=np.float32)
    remaining = np.ones(total, dtype=bool)
    for step in range(keep_count):
        masked = np.where(remaining, candidate_scores, -1.0)
        idx = initial[step] if step < len(initial) else int(np.argmax(masked))
        if not remaining[idx]:
            break
        if min_coverage is not None and step >= len(initial) and float(candidate_scores[idx]) < min_coverage:
            break
        selected.append(idx)
        coverage_scores[idx] = float(candidate_scores[idx])
        remaining[idx] = False
        if len(selected) >= keep_count and distance_update_fn is None and on_select is None:
            break
        distances_to_new = (
            np.asarray(distance_update_fn(idx), dtype=np.float32).reshape(-1)
            if distance_update_fn is not None
            else cosine_distance_matrix(emb, emb[idx : idx + 1]).reshape(-1)
        )
        if distances_to_new.shape != (total,) or not np.all(np.isfinite(distances_to_new)):
            raise ValueError("coverage_distance_update_invalid")
        candidate_scores = np.minimum(candidate_scores, distances_to_new)
        if on_select is not None:
            on_select(idx, len(selected), candidate_scores)
    coverage_scores[remaining] = candidate_scores[remaining]
    return selected, min_distance.astype(np.float32, copy=False), coverage_scores.astype(np.float32, copy=False)


def greedy_diverse_indices_with_local_scores(
    embeddings: np.ndarray,
    *,
    keep_fraction: float,
    reference_embeddings: Optional[np.ndarray] = None,
    local_scores: Optional[Sequence[float]] = None,
    local_weight: float = 0.0,
    min_coverage: Optional[float] = None,
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
            min_coverage=min_coverage,
        )
        return selected, novelty_scores, coverage_scores, coverage_scores.astype(np.float32, copy=True)
    keep_count = max(1, min(total, int(math.ceil(total * normalize_keep_fraction(keep_fraction)))))
    if min_coverage is not None:
        keep_count = total
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
        # Below the cutoff an item is not new enough, however varied its own
        # patches are; local variety only orders the items that are.
        eligible = remaining if min_coverage is None else remaining & (candidate_scores >= min_coverage)
        if not eligible.any():
            break
        coverage_percentiles = _score_percentiles(candidate_scores, mask=eligible)
        combined = (1.0 - weight) * coverage_percentiles + weight * local_percentiles
        masked = np.where(eligible, combined, -1.0)
        idx = int(np.argmax(masked))
        if not eligible[idx]:
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
