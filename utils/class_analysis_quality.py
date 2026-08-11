"""Quality scoring and adaptive review ranking for Class Analysis.

The helpers in this module are intentionally independent of HTTP and UI state.
They operate on real analysis records only. Maximum-fidelity execution uses
every eligible object; memory-limited execution uses deterministic, auditable
working sets and disk-backed arrays so the same recipes scale to large data.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np


THOROUGH_QUALITY_RECIPE = "thorough_quality_v1"
PRECISE_COMPACT_RECIPE = "precise_compact_v1"
FAST_MAP_RECIPE = "fast_map_v1"
CUSTOM_RECIPE = "custom"
QUALITY_MEMORY_POLICIES = ("auto", "full", "budgeted")
QUALITY_MEMORY_MIN_MB = 1024
QUALITY_MEMORY_MAX_MB = 262_144
QUALITY_MEMORY_AUTO_MAX_MB = 32_768
QUALITY_LOW_DETAIL_MIN_SIDE_PX = 32.0

QUALITY_RECIPE_ALIASES = {
    "balanced": THOROUGH_QUALITY_RECIPE,
    "fusion": THOROUGH_QUALITY_RECIPE,
    "salad": THOROUGH_QUALITY_RECIPE,
    "precise": PRECISE_COMPACT_RECIPE,
    "fast": FAST_MAP_RECIPE,
    THOROUGH_QUALITY_RECIPE: THOROUGH_QUALITY_RECIPE,
    PRECISE_COMPACT_RECIPE: PRECISE_COMPACT_RECIPE,
    FAST_MAP_RECIPE: FAST_MAP_RECIPE,
    CUSTOM_RECIPE: CUSTOM_RECIPE,
}


@dataclass(frozen=True)
class QualityRecipe:
    recipe_id: str
    label: str
    compact_weight: float
    cradio_weight: float
    local_weight: float
    logistic_weight: float
    late_compact_weight: float
    late_cradio_weight: float
    late_weight: float
    el2n_weight: float
    review_fraction: float
    use_cradio: bool
    use_el2n: bool


RECIPES = {
    THOROUGH_QUALITY_RECIPE: QualityRecipe(
        recipe_id=THOROUGH_QUALITY_RECIPE,
        label="Thorough quality",
        compact_weight=0.75,
        cradio_weight=0.25,
        local_weight=0.35,
        logistic_weight=0.65,
        late_compact_weight=0.60,
        late_cradio_weight=0.40,
        late_weight=0.70,
        el2n_weight=0.30,
        review_fraction=0.05,
        use_cradio=True,
        use_el2n=True,
    ),
    PRECISE_COMPACT_RECIPE: QualityRecipe(
        recipe_id=PRECISE_COMPACT_RECIPE,
        label="Precise compact",
        compact_weight=1.0,
        cradio_weight=0.0,
        local_weight=0.35,
        logistic_weight=0.65,
        late_compact_weight=1.0,
        late_cradio_weight=0.0,
        late_weight=1.0,
        el2n_weight=0.0,
        review_fraction=0.05,
        use_cradio=False,
        use_el2n=False,
    ),
    FAST_MAP_RECIPE: QualityRecipe(
        recipe_id=FAST_MAP_RECIPE,
        label="Fast map",
        compact_weight=1.0,
        cradio_weight=0.0,
        local_weight=1.0,
        logistic_weight=0.0,
        late_compact_weight=1.0,
        late_cradio_weight=0.0,
        late_weight=1.0,
        el2n_weight=0.0,
        review_fraction=0.05,
        use_cradio=False,
        use_el2n=False,
    ),
}


def available_quality_memory_mb() -> int:
    try:
        import psutil

        return max(1, int(psutil.virtual_memory().available // (1024 * 1024)))
    except Exception:
        try:
            pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return max(1, int((pages * page_size) // (1024 * 1024)))
        except (AttributeError, OSError, TypeError, ValueError):
            # Capability reporting must remain available in minimal Python
            # environments. Production installs provide psutil; this
            # conservative fallback selects Budgeted more readily.
            return 8192


def recommended_quality_memory_mb(available_mb: int | None = None) -> int:
    available = max(1, int(available_mb or available_quality_memory_mb()))
    return max(
        QUALITY_MEMORY_MIN_MB,
        min(QUALITY_MEMORY_AUTO_MAX_MB, int(available * 0.50)),
    )


def estimate_quality_workspace_mb(
    record_count: int,
    compact_dimensions: int,
    cradio_dimensions: int = 0,
) -> int:
    count = max(0, int(record_count))
    compact_dims = max(0, int(compact_dimensions))
    cradio_dims = max(0, int(cradio_dimensions))
    merged_dims = compact_dims + cradio_dims
    # Compact, C-RADIO, normalized branches, merged output, score vectors,
    # blocked exact-neighbour queries, projections, and conservative library
    # headroom. Model residency and immutable input records are reported
    # separately by the API preflight.
    feature_bytes = count * (compact_dims * 2 + cradio_dims * 2 + merged_dims) * 4
    score_bytes = count * 18 * 8
    projection_bytes = count * (merged_dims + 24) * 4
    neighbour_bytes = count * 16 * (8 + 4)
    exact_query_bytes = min(count, 256) * count * 4
    estimate = int(
        (feature_bytes + score_bytes + projection_bytes + neighbour_bytes + exact_query_bytes)
        * 1.20
        + 256 * 1024 * 1024
    )
    return max(1, int(math.ceil(estimate / (1024 * 1024))))


def plan_quality_execution(
    *,
    policy: str,
    budget_mb: int | None,
    record_count: int,
    compact_dimensions: int,
    cradio_dimensions: int = 0,
    available_mb: int | None = None,
) -> dict[str, Any]:
    requested = str(policy or "auto").strip().lower()
    if requested not in QUALITY_MEMORY_POLICIES:
        requested = "auto"
    explicit_budget = budget_mb is not None
    target_mb = int(
        budget_mb
        if explicit_budget
        else recommended_quality_memory_mb(available_mb)
    )
    target_mb = max(QUALITY_MEMORY_MIN_MB, min(QUALITY_MEMORY_MAX_MB, target_mb))
    estimated_mb = estimate_quality_workspace_mb(
        record_count,
        compact_dimensions,
        cradio_dimensions,
    )
    resolved = requested
    if requested == "auto":
        resolved = "full" if estimated_mb <= target_mb else "budgeted"
    return {
        "requested_policy": requested,
        "resolved_policy": resolved,
        "budget_mb": None if requested == "full" else target_mb,
        "budget_source": "user" if explicit_budget else "system_derived",
        "available_memory_mb": int(available_mb or available_quality_memory_mb()),
        "estimated_full_workspace_mb": estimated_mb,
        "budget_contract": "analysis_incremental_working_set_budget_v2",
        "budget_scope": "scoring_projection_clustering_neighbors_serialization",
        "model_and_input_baseline_excluded": True,
        "safety_headroom_fraction": 0.20,
        "never_aborts_for_configured_budget": True,
    }


def _quality_cancelled(cancel_callback: Callable[[], bool] | None) -> None:
    if cancel_callback is not None and bool(cancel_callback()):
        raise RuntimeError("cancelled")


def _quality_progress(
    callback: Callable[[float, str], None] | None,
    progress: float,
    message: str,
) -> None:
    if callback is not None:
        callback(float(progress), str(message))


def resolve_quality_recipe(
    value: Any,
    overrides: Mapping[str, Any] | None = None,
) -> QualityRecipe:
    recipe_id = QUALITY_RECIPE_ALIASES.get(str(value or "").strip().lower())
    if not recipe_id:
        recipe_id = THOROUGH_QUALITY_RECIPE
    if recipe_id == CUSTOM_RECIPE:
        values = overrides or {}

        def custom_bool(key: str, default: bool = False) -> bool:
            raw = values.get(key, default)
            if isinstance(raw, str):
                return raw.strip().lower() in {"1", "true", "yes", "on"}
            return bool(raw)

        def normalized_pair(
            first_key: str,
            second_key: str,
            first_default: float,
            second_default: float,
        ) -> tuple[float, float]:
            try:
                first = float(values.get(first_key, first_default))
            except (TypeError, ValueError):
                first = first_default
            try:
                second = float(values.get(second_key, second_default))
            except (TypeError, ValueError):
                second = second_default
            first = max(0.0, min(1.0, first)) if math.isfinite(first) else first_default
            second = max(0.0, min(1.0, second)) if math.isfinite(second) else second_default
            total = first + second
            if total <= 0.0:
                first, second, total = first_default, second_default, first_default + second_default
            return first / total, second / total

        use_cradio = custom_bool("quality_use_cradio")
        use_el2n = custom_bool("quality_use_el2n")
        compact_weight, cradio_weight = normalized_pair(
            "quality_compact_weight",
            "quality_cradio_weight",
            0.75 if use_cradio else 1.0,
            0.25 if use_cradio else 0.0,
        )
        if not use_cradio:
            compact_weight, cradio_weight = 1.0, 0.0
        local_weight, logistic_weight = normalized_pair(
            "quality_local_weight",
            "quality_logistic_weight",
            0.35,
            0.65,
        )
        late_compact_weight, late_cradio_weight = normalized_pair(
            "quality_late_compact_weight",
            "quality_late_cradio_weight",
            0.60 if use_cradio else 1.0,
            0.40 if use_cradio else 0.0,
        )
        if not use_cradio:
            late_compact_weight, late_cradio_weight = 1.0, 0.0
        late_weight, el2n_weight = normalized_pair(
            "quality_late_weight",
            "quality_el2n_weight",
            0.70 if use_el2n else 1.0,
            0.30 if use_el2n else 0.0,
        )
        if not use_el2n:
            late_weight, el2n_weight = 1.0, 0.0
        return QualityRecipe(
            recipe_id=CUSTOM_RECIPE,
            label="Custom",
            compact_weight=compact_weight,
            cradio_weight=cradio_weight,
            local_weight=local_weight,
            logistic_weight=logistic_weight,
            late_compact_weight=late_compact_weight,
            late_cradio_weight=late_cradio_weight,
            late_weight=late_weight,
            el2n_weight=el2n_weight,
            review_fraction=max(
                0.01,
                min(0.25, float(values.get("quality_review_fraction") or 0.05)),
            ),
            use_cradio=use_cradio,
            use_el2n=use_el2n,
        )
    return RECIPES[recipe_id]


def stable_record_id(record: Mapping[str, Any], index: int = 0) -> str:
    for key in ("point_id", "object_id", "bbox_id", "id", "uuid"):
        value = record.get(key)
        if value is not None and str(value):
            return str(value)
    return f"record:{index}"


def record_label(record: Mapping[str, Any]) -> str:
    for key in ("class_name", "label", "class_id", "category", "category_id"):
        value = record.get(key)
        if value is not None and str(value):
            return str(value)
    return "__unknown__"


def record_source_group(record: Mapping[str, Any], index: int = 0) -> str:
    for key in (
        "source_group",
        "source_image_id",
        "image_id",
        "image_key",
        "relative_path",
        "image_path",
        "source",
        "filename",
    ):
        value = record.get(key)
        if value is not None and str(value):
            return str(value)
    return stable_record_id(record, index)


def _stable_u64(value: str, seed: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{value}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("quality embeddings must be a two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def merge_quality_features(
    compact_embeddings: np.ndarray,
    cradio_embeddings: np.ndarray | None,
    *,
    compact_weight: float = 0.75,
    cradio_weight: float = 0.25,
    output_path: str | Path | None = None,
    chunk_size: int = 2048,
    cancel_callback: Callable[[], bool] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> np.ndarray:
    if output_path is not None:
        compact_source = np.asarray(compact_embeddings)
        if compact_source.ndim != 2:
            raise ValueError("quality embeddings must be a two-dimensional matrix")
        cradio_source = None if cradio_embeddings is None else np.asarray(cradio_embeddings)
        if cradio_source is not None and compact_source.shape[0] != cradio_source.shape[0]:
            raise ValueError("quality feature branches must contain the same records")
        use_cradio = cradio_source is not None and cradio_weight > 0.0
        output_dimensions = compact_source.shape[1] + (cradio_source.shape[1] if use_cradio else 0)
        output = np.lib.format.open_memmap(
            str(output_path),
            mode="w+",
            dtype=np.float32,
            shape=(compact_source.shape[0], output_dimensions),
        )
        safe_chunk = max(1, int(chunk_size))
        for start in range(0, compact_source.shape[0], safe_chunk):
            _quality_cancelled(cancel_callback)
            end = min(compact_source.shape[0], start + safe_chunk)
            compact_chunk = _unit_rows(compact_source[start:end])
            if use_cradio:
                cradio_chunk = _unit_rows(cradio_source[start:end])
                chunk = np.concatenate(
                    (
                        compact_chunk * math.sqrt(max(0.0, compact_weight)),
                        cradio_chunk * math.sqrt(max(0.0, cradio_weight)),
                    ),
                    axis=1,
                )
                output[start:end] = _unit_rows(chunk)
            else:
                output[start:end] = compact_chunk
            _quality_progress(
                progress_callback,
                end / max(1, compact_source.shape[0]),
                f"Fusing quality features: {end}/{compact_source.shape[0]}",
            )
        output.flush()
        return output
    compact = _unit_rows(compact_embeddings)
    if cradio_embeddings is None or cradio_weight <= 0.0:
        return compact
    cradio = _unit_rows(cradio_embeddings)
    if compact.shape[0] != cradio.shape[0]:
        raise ValueError("quality feature branches must contain the same records")
    merged = np.concatenate(
        (
            compact * math.sqrt(max(0.0, compact_weight)),
            cradio * math.sqrt(max(0.0, cradio_weight)),
        ),
        axis=1,
    )
    return _unit_rows(merged)


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    count = values.size
    finite_mask = np.isfinite(values)
    finite_count = int(np.sum(finite_mask))
    if count <= 1 or finite_count <= 1:
        return np.zeros(count, dtype=np.float64)
    finite_values = values[finite_mask]
    if float(np.max(finite_values)) == float(np.min(finite_values)):
        return np.zeros(count, dtype=np.float64)
    order = np.argsort(finite_values, kind="mergesort")
    sorted_values = finite_values[order]
    finite_ranks = np.zeros(finite_count, dtype=np.float64)
    denominator = float(max(1, finite_count - 1))
    start = 0
    while start < finite_count:
        end = start + 1
        while end < finite_count and sorted_values[end] == sorted_values[start]:
            end += 1
        finite_ranks[order[start:end]] = ((start + end - 1) * 0.5) / denominator
        start = end
    ranks = np.zeros(count, dtype=np.float64)
    ranks[finite_mask] = finite_ranks
    return ranks


def _size_percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Midranks for descriptive size evidence; a constant population is neutral."""
    values = np.asarray(values, dtype=np.float64)
    ranks = _percentile_ranks(values)
    finite = np.isfinite(values)
    if int(np.sum(finite)) <= 1:
        ranks[finite] = 0.5
    elif float(np.max(values[finite])) == float(np.min(values[finite])):
        ranks[finite] = 0.5
    ranks[~finite] = np.nan
    return ranks


def _bounded_indices(
    ids: Sequence[str],
    labels: Sequence[str],
    groups: Sequence[str],
    limit: int,
    seed: int,
) -> np.ndarray:
    count = len(ids)
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    by_label: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(str(label), []).append(index)
    selected: list[int] = []
    per_label = max(1, limit // max(1, len(by_label)))
    for label in sorted(by_label):
        candidates = sorted(
            by_label[label],
            key=lambda index: (
                _stable_u64(f"{groups[index]}:{ids[index]}", seed),
                ids[index],
            ),
        )
        selected.extend(candidates[:per_label])
    selected_set = set(selected)
    if len(selected) < limit:
        remaining = sorted(
            (index for index in range(count) if index not in selected_set),
            key=lambda index: (
                _stable_u64(f"{labels[index]}:{groups[index]}:{ids[index]}", seed + 1),
                ids[index],
            ),
        )
        selected.extend(remaining[: limit - len(selected)])
    return np.asarray(sorted(selected[:limit]), dtype=np.int64)


def _group_folds(
    labels: Sequence[str],
    groups: Sequence[str],
    seed: int,
    folds: int = 5,
) -> np.ndarray:
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        return np.zeros(len(groups), dtype=np.int64)
    fold_count = min(folds, len(unique_groups))
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        assignment = np.full(len(groups), -1, dtype=np.int64)
        splitter = StratifiedGroupKFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=seed,
        )
        placeholders = np.zeros((len(groups), 1), dtype=np.float32)
        for fold, (_, test) in enumerate(splitter.split(placeholders, labels, groups)):
            assignment[test] = fold
        if np.all(assignment >= 0):
            return assignment
    except (ImportError, ValueError):
        pass
    mapping = {group: _stable_u64(group, seed) % fold_count for group in unique_groups}
    return np.asarray([mapping[group] for group in groups], dtype=np.int64)


def _fallback_disagreement(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes, counts = np.unique(labels, return_counts=True)
    majority = classes[int(np.argmax(counts))]
    predictions = np.full(labels.shape[0], majority, dtype=object)
    return (predictions != labels).astype(np.float64), predictions


def _oof_logistic_disagreement(
    features: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[str],
    ids: Sequence[str],
    *,
    seed: int,
    fit_limit: int,
    predict_chunk_size: int = 2048,
    cancel_callback: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import LogisticRegression

    if np.unique(labels).size < 2:
        return np.zeros(labels.size, dtype=np.float64), labels.copy()
    folds = _group_folds(labels.tolist(), groups, seed)
    disagreement = np.full(labels.size, np.nan, dtype=np.float64)
    predictions = np.empty(labels.size, dtype=object)
    predictions[:] = None
    for fold in sorted(set(folds.tolist())):
        _quality_cancelled(cancel_callback)
        test = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        if not test.size or np.unique(labels[train]).size < 2:
            continue
        bounded = _bounded_indices(
            [ids[i] for i in train],
            [str(labels[i]) for i in train],
            [groups[i] for i in train],
            fit_limit,
            seed + int(fold),
        )
        train = train[bounded]
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=600,
            random_state=seed,
        )
        model.fit(features[train], labels[train])
        class_lookup = {str(value): idx for idx, value in enumerate(model.classes_)}
        valid_test = np.asarray(
            [index for index in test if str(labels[index]) in class_lookup],
            dtype=np.int64,
        )
        for start in range(0, valid_test.size, max(1, int(predict_chunk_size))):
            _quality_cancelled(cancel_callback)
            chunk = valid_test[start : start + max(1, int(predict_chunk_size))]
            probabilities = model.predict_proba(features[chunk])
            true_probability = np.asarray(
                [
                    probabilities[row, class_lookup[str(labels[index])]]
                    for row, index in enumerate(chunk)
                ],
                dtype=np.float64,
            )
            disagreement[chunk] = 1.0 - true_probability
            predictions[chunk] = model.classes_[np.argmax(probabilities, axis=1)]
    return disagreement, predictions


def _oof_el2n(
    features: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[str],
    ids: Sequence[str],
    *,
    seed: int,
    fit_limit: int,
    epochs: int = 5,
    predict_chunk_size: int = 2048,
    cancel_callback: Callable[[], bool] | None = None,
) -> np.ndarray:
    from sklearn.linear_model import SGDClassifier
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(labels)
    if classes.size < 2:
        return np.zeros(labels.size, dtype=np.float64)
    class_lookup = {str(value): idx for idx, value in enumerate(classes)}
    folds = _group_folds(labels.tolist(), groups, seed + 23)
    scores = np.full(labels.size, np.nan, dtype=np.float64)
    for fold in sorted(set(folds.tolist())):
        _quality_cancelled(cancel_callback)
        test = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        if not test.size or np.unique(labels[train]).size < 2:
            continue
        bounded = _bounded_indices(
            [ids[i] for i in train],
            [str(labels[i]) for i in train],
            [groups[i] for i in train],
            fit_limit,
            seed + 100 + int(fold),
        )
        train = train[bounded]
        present_classes = np.unique(labels[train])
        present_weights = compute_class_weight(
            class_weight="balanced",
            classes=present_classes,
            y=labels[train],
        )
        class_weights = {
            value: 1.0 for value in classes
        }
        class_weights.update(
            {
                value: float(weight)
                for value, weight in zip(present_classes, present_weights)
            }
        )
        model = SGDClassifier(
            loss="log_loss",
            alpha=0.0001,
            class_weight=class_weights,
            random_state=seed,
        )
        valid_test = np.asarray(
            [index for index in test if labels[index] in set(present_classes.tolist())],
            dtype=np.int64,
        )
        epoch_scores = np.zeros(valid_test.size, dtype=np.float64)
        for epoch in range(epochs):
            _quality_cancelled(cancel_callback)
            order = np.asarray(
                sorted(train, key=lambda idx: _stable_u64(ids[int(idx)], seed + epoch)),
                dtype=np.int64,
            )
            model.partial_fit(features[order], labels[order], classes=classes)
            for start in range(0, valid_test.size, max(1, int(predict_chunk_size))):
                chunk = valid_test[start : start + max(1, int(predict_chunk_size))]
                probabilities = model.predict_proba(features[chunk])
                one_hot = np.zeros_like(probabilities)
                for row, idx in enumerate(chunk):
                    one_hot[row, class_lookup[str(labels[idx])]] = 1.0
                epoch_scores[start : start + chunk.size] += np.linalg.norm(
                    probabilities - one_hot,
                    axis=1,
                )
        scores[valid_test] = epoch_scores / float(epochs)
    return scores


def _oof_rbf_proposals(
    features: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[str],
    ids: Sequence[str],
    *,
    seed: int,
    fit_limit: int,
    predict_chunk_size: int = 2048,
    cancel_callback: Callable[[], bool] | None = None,
) -> np.ndarray:
    from sklearn.svm import SVC

    if np.unique(labels).size < 2:
        return labels.copy()
    folds = _group_folds(labels.tolist(), groups, seed + 47)
    proposals = np.empty(labels.size, dtype=object)
    proposals[:] = None
    for fold in sorted(set(folds.tolist())):
        _quality_cancelled(cancel_callback)
        test = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        if not test.size or np.unique(labels[train]).size < 2:
            continue
        bounded = _bounded_indices(
            [ids[i] for i in train],
            [str(labels[i]) for i in train],
            [groups[i] for i in train],
            fit_limit,
            seed + 200 + int(fold),
        )
        train = train[bounded]
        model = SVC(C=3.0, gamma="scale", kernel="rbf")
        model.fit(features[train], labels[train])
        present = set(str(value) for value in model.classes_)
        valid_test = np.asarray(
            [index for index in test if str(labels[index]) in present],
            dtype=np.int64,
        )
        for start in range(0, valid_test.size, max(1, int(predict_chunk_size))):
            _quality_cancelled(cancel_callback)
            chunk = valid_test[start : start + max(1, int(predict_chunk_size))]
            proposals[chunk] = model.predict(features[chunk])
    return proposals


def _oof_bounded_rbf_proposals(
    features: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[str],
    ids: Sequence[str],
    *,
    seed: int,
    fit_limit: int,
    components: int,
    predict_chunk_size: int,
    cancel_callback: Callable[[], bool] | None = None,
) -> np.ndarray:
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import LogisticRegression

    proposals = np.empty(labels.size, dtype=object)
    proposals[:] = None
    if np.unique(labels).size < 2:
        return proposals
    folds = _group_folds(labels.tolist(), groups, seed + 47)
    for fold in sorted(set(folds.tolist())):
        _quality_cancelled(cancel_callback)
        test = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        if not test.size or np.unique(labels[train]).size < 2:
            continue
        bounded = _bounded_indices(
            [ids[index] for index in train],
            [str(labels[index]) for index in train],
            [groups[index] for index in train],
            fit_limit,
            seed + 200 + int(fold),
        )
        train = train[bounded]
        component_count = max(32, min(int(components), train.size))
        mapper = Nystroem(
            kernel="rbf",
            gamma=None,
            n_components=component_count,
            random_state=seed + int(fold),
        )
        transformed_train = mapper.fit_transform(features[train])
        model = LogisticRegression(
            C=3.0,
            class_weight="balanced",
            max_iter=600,
            random_state=seed,
        )
        model.fit(transformed_train, labels[train])
        present = set(str(value) for value in model.classes_)
        valid_test = np.asarray(
            [index for index in test if str(labels[index]) in present],
            dtype=np.int64,
        )
        for start in range(0, valid_test.size, max(1, int(predict_chunk_size))):
            _quality_cancelled(cancel_callback)
            chunk = valid_test[start : start + max(1, int(predict_chunk_size))]
            proposals[chunk] = model.predict(mapper.transform(features[chunk]))
    return proposals


def _cosine_neighbour_disagreement(
    features: np.ndarray,
    labels: np.ndarray,
    ids: Sequence[str],
    groups: Sequence[str],
    *,
    seed: int,
    reference_limit: int,
    neighbours: int = 15,
    query_chunk_size: int | None = None,
    cancel_callback: Callable[[], bool] | None = None,
    approximate: bool = False,
) -> np.ndarray:
    count = labels.size
    if count <= 1:
        return np.zeros(count, dtype=np.float64)
    reference = _bounded_indices(ids, labels.tolist(), groups, reference_limit, seed + 311)
    reference_features = features[reference]
    neighbour_count = min(neighbours + 1, reference.size)
    if approximate:
        from pynndescent import NNDescent

        model = NNDescent(
            reference_features,
            n_neighbors=neighbour_count,
            metric="cosine",
            random_state=seed,
            low_memory=True,
        )
        model.prepare()
    else:
        from sklearn.neighbors import NearestNeighbors

        model = NearestNeighbors(
            n_neighbors=neighbour_count,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        model.fit(reference_features)
    result = np.zeros(count, dtype=np.float64)
    chunk_size = query_chunk_size or max(32, min(2048, int(2_000_000 / max(1, reference.size))))
    for start in range(0, count, chunk_size):
        _quality_cancelled(cancel_callback)
        end = min(count, start + chunk_size)
        if approximate:
            relative, _ = model.query(features[start:end], k=neighbour_count)
        else:
            _, relative = model.kneighbors(features[start:end], return_distance=True)
        absolute = reference[relative]
        for row, item_index in enumerate(range(start, end)):
            neighbour_indices = [
                int(candidate)
                for candidate in absolute[row]
                if int(candidate) != item_index
            ][:neighbours]
            if neighbour_indices:
                result[item_index] = float(
                    np.mean(labels[neighbour_indices] != labels[item_index])
                )
    return result


def _bbox_dimensions(record: Mapping[str, Any]) -> tuple[float, float, float, float]:
    bbox = (
        record.get("bbox_xyxy")
        or record.get("bbox")
        or record.get("box")
        or record.get("bounds")
    )
    width = height = 0.0
    if isinstance(bbox, Mapping):
        width = float(bbox.get("width") or bbox.get("w") or 0.0)
        height = float(bbox.get("height") or bbox.get("h") or 0.0)
        if width <= 0.0 and bbox.get("x2") is not None:
            width = float(bbox.get("x2", 0.0)) - float(bbox.get("x1", 0.0))
        if height <= 0.0 and bbox.get("y2") is not None:
            height = float(bbox.get("y2", 0.0)) - float(bbox.get("y1", 0.0))
    elif isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)) and len(bbox) >= 4:
        x1, y1, third, fourth = [float(value) for value in bbox[:4]]
        mode = str(record.get("bbox_mode") or "").lower()
        if mode in {"xywh", "coco"}:
            width, height = third, fourth
        else:
            width, height = max(0.0, third - x1), max(0.0, fourth - y1)
            if width <= 0.0 or height <= 0.0:
                width, height = max(0.0, third), max(0.0, fourth)
    width = float(record.get("bbox_width") or width)
    height = float(record.get("bbox_height") or height)
    image_width = float(
        record.get("source_width")
        or record.get("image_width")
        or record.get("frame_width")
        or 0.0
    )
    image_height = float(
        record.get("source_height")
        or record.get("image_height")
        or record.get("frame_height")
        or 0.0
    )
    return max(width, 0.0), max(height, 0.0), image_width, image_height


def compute_size_evidence(
    records: Sequence[Mapping[str, Any]],
    *,
    low_detail_min_side_px: float = QUALITY_LOW_DETAIL_MIN_SIDE_PX,
) -> list[dict[str, Any]]:
    count = len(records)
    if not count:
        return []
    labels = np.asarray([record_label(record) for record in records], dtype=object)
    dimensions = [_bbox_dimensions(record) for record in records]
    pixel_areas = np.asarray([width * height for width, height, _, _ in dimensions], dtype=np.float64)
    normalized_areas = np.asarray(
        [
            (width * height) / (image_width * image_height)
            if image_width > 0.0 and image_height > 0.0
            else np.nan
            for width, height, image_width, image_height in dimensions
        ],
        dtype=np.float64,
    )
    global_percentile = _size_percentile_ranks(pixel_areas)
    normalized_percentile = _size_percentile_ranks(normalized_areas)
    class_percentile = np.zeros(count, dtype=np.float64)
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        class_percentile[indices] = _size_percentile_ranks(pixel_areas[indices])
    rows: list[dict[str, Any]] = []
    for index, (width, height, _, _) in enumerate(dimensions):
        low_source_detail = min(width, height) < max(1.0, float(low_detail_min_side_px))
        global_quintile = min(5, int(global_percentile[index] * 5.0) + 1)
        relative_small = bool(global_percentile[index] < 0.20)
        rows.append(
            {
                "bbox_pixel_area": float(pixel_areas[index]),
                "bbox_source_area": (
                    float(normalized_areas[index])
                    if math.isfinite(float(normalized_areas[index]))
                    else None
                ),
                "bbox_global_percentile": float(global_percentile[index]),
                "bbox_normalized_percentile": (
                    float(normalized_percentile[index])
                    if math.isfinite(float(normalized_percentile[index]))
                    else None
                ),
                "bbox_class_percentile": float(class_percentile[index]),
                "bbox_global_quintile": int(global_quintile),
                "low_source_detail": bool(low_source_detail),
                "low_detail_min_side_px": float(low_detail_min_side_px),
                "relative_small_object": relative_small,
                # Compatibility alias. Relative smallness is evidence only.
                "tiny_object": bool(low_source_detail),
            }
        )
    return rows


def _normalize_features_to_memmap(
    values: np.ndarray,
    path: Path,
    *,
    chunk_size: int,
    cancel_callback: Callable[[], bool] | None,
) -> np.memmap:
    source = np.asarray(values)
    if source.ndim != 2:
        raise ValueError("quality embeddings must be a two-dimensional matrix")
    output = np.lib.format.open_memmap(
        str(path),
        mode="w+",
        dtype=np.float32,
        shape=source.shape,
    )
    for start in range(0, source.shape[0], max(1, int(chunk_size))):
        _quality_cancelled(cancel_callback)
        end = min(source.shape[0], start + max(1, int(chunk_size)))
        output[start:end] = _unit_rows(source[start:end])
    output.flush()
    return output


def _weighted_rank_blend(
    components: Sequence[tuple[np.ndarray, float, np.ndarray | None]],
) -> tuple[np.ndarray, np.ndarray]:
    if not components:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=bool)
    count = np.asarray(components[0][0]).size
    numerator = np.zeros(count, dtype=np.float64)
    denominator = np.zeros(count, dtype=np.float64)
    for values, raw_weight, upstream_valid in components:
        weight = max(0.0, float(raw_weight))
        array = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(array)
        finite_values = array[finite]
        discriminating = bool(
            finite_values.size > 1
            and float(np.max(finite_values)) != float(np.min(finite_values))
        )
        valid = finite if discriminating else np.zeros(count, dtype=bool)
        if upstream_valid is not None:
            valid &= np.asarray(upstream_valid, dtype=bool)
        ranks = _percentile_ranks(array)
        numerator[valid] += weight * ranks[valid]
        denominator[valid] += weight
    output = np.zeros(count, dtype=np.float64)
    valid = denominator > 0.0
    output[valid] = numerator[valid] / denominator[valid]
    return output, valid


def _priority_indices_without_boundary_tie(
    scores: np.ndarray,
    eligible: np.ndarray,
    ids: Sequence[str],
    target_count: int,
) -> list[int]:
    ordered = sorted(
        (int(index) for index in np.flatnonzero(eligible)),
        key=lambda index: (-float(scores[index]), ids[index]),
    )
    if target_count <= 0 or not ordered:
        return []
    if len(ordered) <= target_count:
        return ordered
    boundary = float(scores[ordered[target_count - 1]])
    tied = [index for index in ordered if float(scores[index]) == boundary]
    above = [index for index in ordered if float(scores[index]) > boundary]
    if len(above) + len(tied) > target_count:
        return above
    return ordered[:target_count]


def score_quality_records(
    records: Sequence[Mapping[str, Any]],
    compact_embeddings: np.ndarray,
    cradio_embeddings: np.ndarray | None = None,
    *,
    recipe_id: str = THOROUGH_QUALITY_RECIPE,
    recipe_overrides: Mapping[str, Any] | None = None,
    seed: int = 17,
    logistic_fit_limit: int = 8192,
    rbf_fit_limit: int = 4096,
    neighbour_reference_limit: int = 20_000,
    review_fraction: float | None = None,
    low_detail_min_side_px: float = QUALITY_LOW_DETAIL_MIN_SIDE_PX,
    memory_policy: str = "full",
    memory_budget_mb: int | None = None,
    scratch_dir: str | Path | None = None,
    execution_metadata: MutableMapping[str, Any] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], list[str]]:
    recipe = resolve_quality_recipe(recipe_id, recipe_overrides)
    compact_source = np.asarray(compact_embeddings)
    if compact_source.ndim != 2 or compact_source.shape[0] != len(records):
        raise ValueError("quality records and compact embeddings must align")
    if recipe.use_cradio and cradio_embeddings is None:
        raise ValueError("thorough_quality_v1 requires the C-RADIO feature branch")
    cradio_source = None if cradio_embeddings is None else np.asarray(cradio_embeddings)
    plan = plan_quality_execution(
        policy=memory_policy,
        budget_mb=memory_budget_mb,
        record_count=len(records),
        compact_dimensions=compact_source.shape[1],
        cradio_dimensions=0 if cradio_source is None else cradio_source.shape[1],
    )
    if execution_metadata is not None:
        execution_metadata.update(plan)
    resolved_policy = str(plan["resolved_policy"])
    owner: tempfile.TemporaryDirectory[str] | None = None
    if resolved_policy == "budgeted":
        if scratch_dir is None:
            owner = tempfile.TemporaryDirectory(prefix="tator-quality-")
            workspace = Path(owner.name)
        else:
            workspace = Path(scratch_dir)
            workspace.mkdir(parents=True, exist_ok=True)
        target_bytes = int(plan.get("budget_mb") or QUALITY_MEMORY_MIN_MB) * 1024 * 1024
        dimensions = compact_source.shape[1] + (0 if cradio_source is None else cradio_source.shape[1])
        chunk_size = max(32, min(4096, int((target_bytes * 0.08) / max(1, dimensions * 4 * 4))))
        compact = _normalize_features_to_memmap(
            compact_source,
            workspace / "compact.npy",
            chunk_size=chunk_size,
            cancel_callback=cancel_callback,
        )
        cradio = (
            _normalize_features_to_memmap(
                cradio_source,
                workspace / "cradio.npy",
                chunk_size=chunk_size,
                cancel_callback=cancel_callback,
            )
            if cradio_source is not None
            else None
        )
        merged = merge_quality_features(
            compact,
            cradio,
            compact_weight=recipe.compact_weight,
            cradio_weight=recipe.cradio_weight,
            output_path=workspace / "merged.npy",
            chunk_size=chunk_size,
            cancel_callback=cancel_callback,
            progress_callback=progress_callback,
        )
        largest_branch = max(compact.shape[1], merged.shape[1], cradio.shape[1] if cradio is not None else 0)
        bounded_reference = max(
            256,
            min(neighbour_reference_limit, int((target_bytes * 0.18) / max(1, largest_branch * 4 * 2))),
        )
        bounded_fit = max(256, min(logistic_fit_limit, bounded_reference))
        bounded_rbf_fit = max(256, min(rbf_fit_limit, bounded_reference))
        rbf_components = max(128, min(2048, int(plan.get("budget_mb") or 1024)))
        if execution_metadata is not None:
            execution_metadata.update(
                {
                    "fusion_storage": "temporary_memmap",
                    "neighbour_algorithm": "pynndescent",
                    "proposal_algorithm": "budget_planned",
                    "chunk_size": chunk_size,
                    "logistic_fit_limit": bounded_fit,
                    "rbf_fit_limit": bounded_rbf_fit,
                    "neighbour_reference_limit": bounded_reference,
                    "rbf_components": rbf_components,
                    "_scratch_owner": owner,
                    "_scratch_dir": str(workspace),
                }
            )
    else:
        compact = _unit_rows(compact_source)
        cradio = _unit_rows(cradio_source) if cradio_source is not None else None
        merged = merge_quality_features(
            compact,
            cradio,
            compact_weight=recipe.compact_weight,
            cradio_weight=recipe.cradio_weight,
        )
        chunk_size = 256
        # Maximum fidelity has no fixed fit/reference reservoirs. Exact
        # algorithms consume every eligible object; API preflight warns before
        # a quadratic run that is unlikely to fit the current machine.
        bounded_reference = len(records)
        bounded_fit = len(records)
        bounded_rbf_fit = len(records)
        rbf_components = 0
        if execution_metadata is not None:
            execution_metadata.update(
                {
                    "fusion_storage": "in_memory",
                    "neighbour_algorithm": "exact_brute_cosine",
                    "proposal_algorithm": "exact_rbf_svc",
                    "full_population_fit": True,
                    "logistic_fit_limit": len(records),
                    "rbf_fit_limit": len(records),
                    "neighbour_reference_limit": len(records),
                }
            )
    ids = [stable_record_id(record, index) for index, record in enumerate(records)]
    labels = np.asarray([record_label(record) for record in records], dtype=object)
    groups = [record_source_group(record, index) for index, record in enumerate(records)]

    _quality_progress(progress_callback, 0.10, "Scoring compact logistic disagreement ...")
    compact_logistic, _ = _oof_logistic_disagreement(
        compact,
        labels,
        groups,
        ids,
        seed=seed,
        fit_limit=bounded_fit,
        predict_chunk_size=chunk_size,
        cancel_callback=cancel_callback,
    )
    compact_local = _cosine_neighbour_disagreement(
        compact,
        labels,
        ids,
        groups,
        seed=seed,
        reference_limit=bounded_reference,
        query_chunk_size=chunk_size,
        cancel_callback=cancel_callback,
        approximate=resolved_policy == "budgeted",
    )
    compact_score, compact_valid = _weighted_rank_blend(
        (
            (compact_logistic, recipe.logistic_weight, None),
            (compact_local, recipe.local_weight, None),
        )
    )

    if recipe.use_cradio and cradio is not None:
        cradio_logistic, _ = _oof_logistic_disagreement(
            cradio,
            labels,
            groups,
            ids,
            seed=seed + 1,
            fit_limit=bounded_fit,
            predict_chunk_size=chunk_size,
            cancel_callback=cancel_callback,
        )
        cradio_local = _cosine_neighbour_disagreement(
            cradio,
            labels,
            ids,
            groups,
            seed=seed + 1,
            reference_limit=bounded_reference,
            query_chunk_size=chunk_size,
            cancel_callback=cancel_callback,
            approximate=resolved_policy == "budgeted",
        )
        cradio_score, cradio_valid = _weighted_rank_blend(
            (
                (cradio_logistic, recipe.logistic_weight, None),
                (cradio_local, recipe.local_weight, None),
            )
        )
    else:
        cradio_logistic = np.zeros(labels.size, dtype=np.float64)
        cradio_local = np.zeros(labels.size, dtype=np.float64)
        cradio_score = compact_score.copy()
        cradio_valid = compact_valid.copy()

    late_score, late_valid = _weighted_rank_blend(
        (
            (compact_score, recipe.late_compact_weight, compact_valid),
            (cradio_score, recipe.late_cradio_weight, cradio_valid),
        )
    )
    if recipe.use_el2n:
        el2n = _oof_el2n(
            merged,
            labels,
            groups,
            ids,
            seed=seed,
            fit_limit=bounded_fit,
            predict_chunk_size=chunk_size,
            cancel_callback=cancel_callback,
        )
    else:
        el2n = np.zeros(labels.size, dtype=np.float64)
    final_score, signal_available = _weighted_rank_blend(
        (
            (late_score, recipe.late_weight, late_valid),
            (el2n, recipe.el2n_weight, None),
        )
    )
    _quality_progress(progress_callback, 0.82, "Generating source-disjoint class proposals ...")
    if resolved_policy == "budgeted" and labels.size > 2000:
        proposals = _oof_bounded_rbf_proposals(
            merged,
            labels,
            groups,
            ids,
            seed=seed,
            fit_limit=bounded_rbf_fit,
            components=rbf_components,
            predict_chunk_size=chunk_size,
            cancel_callback=cancel_callback,
        )
    else:
        proposals = _oof_rbf_proposals(
            merged,
            labels,
            groups,
            ids,
            seed=seed,
            fit_limit=bounded_rbf_fit,
            predict_chunk_size=chunk_size,
            cancel_callback=cancel_callback,
        )
    if execution_metadata is not None:
        execution_metadata["proposal_algorithm"] = (
            "nystroem_rbf_logistic"
            if resolved_policy == "budgeted" and labels.size > 2000
            else "exact_fit_bounded_rbf_svc"
            if resolved_policy == "budgeted"
            else "exact_rbf_svc"
        )
    size_rows = compute_size_evidence(
        records,
        low_detail_min_side_px=low_detail_min_side_px,
    )
    effective_fraction = float(recipe.review_fraction if review_fraction is None else review_fraction)
    effective_fraction = max(0.01, min(0.25, effective_fraction))
    review_count = min(labels.size, max(1, int(math.ceil(labels.size * effective_fraction))))
    eligible = np.asarray(
        [
            bool(signal_available[index])
            for index in range(labels.size)
        ],
        dtype=bool,
    )
    priority_indices = _priority_indices_without_boundary_tie(
        final_score,
        eligible,
        ids,
        review_count,
    )
    selected = {ids[index] for index in priority_indices}
    rows: list[dict[str, Any]] = []
    for index, point_id in enumerate(ids):
        rows.append(
            {
                "point_id": point_id,
                "quality_recipe": recipe.recipe_id,
                "quality_score": float(final_score[index]),
                "compact_logistic_disagreement": float(compact_logistic[index]),
                "compact_neighbor_disagreement": float(compact_local[index]),
                "cradio_logistic_disagreement": float(cradio_logistic[index]),
                "cradio_neighbor_disagreement": float(cradio_local[index]),
                "el2n_score": float(el2n[index]),
                "quality_signal_available": bool(signal_available[index]),
                "quality_abstention_reason": None if signal_available[index] else "no_discriminating_quality_signal",
                "quality_review_fraction": effective_fraction,
                "proposed_class": None if proposals[index] is None else str(proposals[index]),
                "proposed_class_differs": bool(
                    proposals[index] is not None
                    and str(proposals[index]) != str(labels[index])
                ),
                "quality_review_candidate": point_id in selected,
                "quality_queue_bucket": (
                    "priority"
                    if point_id in selected
                    else "low_detail"
                    if bool(size_rows[index].get("low_source_detail"))
                    else "scored"
                ),
                "quality_flag_sources": (
                    ["quality_priority"]
                    if point_id in selected
                    else ["low_source_detail"]
                    if bool(size_rows[index].get("low_source_detail"))
                    else []
                ),
                **size_rows[index],
            }
        )
    _quality_progress(progress_callback, 1.0, "Quality scoring complete")
    return merged, rows, [ids[index] for index in priority_indices]


def apply_quality_rows(
    result: dict[str, Any],
    quality_rows: Sequence[Mapping[str, Any]],
    review_ids: Sequence[str],
) -> dict[str, Any]:
    by_id = {str(row.get("point_id")): dict(row) for row in quality_rows}
    points = result.get("points") if isinstance(result.get("points"), list) else []
    point_by_id: dict[str, dict[str, Any]] = {}
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            continue
        point_id = stable_record_id(point, index)
        point_by_id[point_id] = point
        quality = by_id.get(point_id)
        if quality:
            point.update(quality)
            point["wrong_class_suspicion"] = float(quality.get("quality_score") or 0.0)
            point["review_priority_score"] = float(quality.get("quality_score") or 0.0)
            if quality.get("quality_review_candidate"):
                point["review_candidate"] = True
    existing = result.get("wrong_class_candidates")
    existing_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(existing, list):
        for index, item in enumerate(existing):
            if isinstance(item, dict):
                existing_by_id[stable_record_id(item, index)] = item
    existing_ids = list(existing_by_id)
    for point_id in existing_ids:
        point = point_by_id.get(point_id)
        if point is not None:
            point["review_candidate"] = True
            sources = list(point.get("quality_flag_sources") or [])
            if "legacy_heuristic" not in sources:
                sources.append("legacy_heuristic")
            point["quality_flag_sources"] = sources
    ordered_size_rows = sorted(
        quality_rows,
        key=lambda row: (-float(row.get("quality_score") or 0.0), str(row.get("point_id") or "")),
    )
    low_detail_ids = [
        str(row.get("point_id") or "")
        for row in ordered_size_rows
        if str(row.get("point_id") or "")
        and bool(row.get("low_source_detail"))
    ]
    # `tiny_ids` remains a response-compatibility alias. V2 consumers should
    # use the threshold-bound `low_detail_ids` field.
    tiny_ids = list(low_detail_ids)
    rich_candidate_ids: list[str] = []
    seen_rich_ids: set[str] = set()
    # Size evidence is a separate review axis. Keeping every low-detail point in
    # the rich wrong-class queue duplicates arbitrary portions of the graph and
    # incorrectly promotes resolution risk into label-error suspicion. The
    # dedicated ID-only queue below remains complete and drives the UI filter.
    for point_id in [*review_ids, *existing_ids]:
        safe_id = str(point_id)
        if safe_id and safe_id not in seen_rich_ids:
            seen_rich_ids.add(safe_id)
            rich_candidate_ids.append(safe_id)
    ordered: list[dict[str, Any]] = []
    for point_id in rich_candidate_ids:
        point = point_by_id.get(str(point_id))
        if point is None:
            continue
        candidate = dict(existing_by_id.get(str(point_id)) or point)
        candidate.update(by_id.get(str(point_id), {}))
        sources = list(candidate.get("quality_flag_sources") or [])
        if str(point_id) in existing_by_id and "legacy_heuristic" not in sources:
            sources.append("legacy_heuristic")
        candidate["quality_flag_sources"] = sources
        candidate["wrong_class_suspicion"] = float(candidate.get("quality_score") or 0.0)
        candidate["review_priority_score"] = float(candidate.get("quality_score") or 0.0)
        ordered.append(candidate)
    result["wrong_class_candidates"] = ordered
    priority_ids: list[str] = []
    priority_seen: set[str] = set()
    for point_id in [*review_ids, *existing_ids]:
        safe_id = str(point_id)
        if safe_id in point_by_id and safe_id not in priority_seen:
            priority_seen.add(safe_id)
            priority_ids.append(safe_id)
    relative_small_ids = [
        str(row.get("point_id") or "")
        for row in quality_rows
        if str(row.get("point_id") or "") and bool(row.get("relative_small_object"))
    ]
    all_flagged_ids: list[str] = []
    seen_flagged_ids: set[str] = set()
    for point_id in [*rich_candidate_ids, *low_detail_ids, *relative_small_ids]:
        safe_id = str(point_id)
        if safe_id in point_by_id and safe_id not in seen_flagged_ids:
            seen_flagged_ids.add(safe_id)
            all_flagged_ids.append(safe_id)
    result["quality_review_queue"] = {
        "schema": "class-analysis-quality-review-queue-v2",
        "size_evidence_schema": "class-analysis-size-evidence-v2",
        "tiny_ids_semantics": "legacy_alias_of_low_detail_ids",
        "priority_ids": priority_ids,
        "tiny_ids": [point_id for point_id in tiny_ids if point_id in point_by_id],
        "low_detail_ids": [point_id for point_id in low_detail_ids if point_id in point_by_id],
        "relative_small_ids": [point_id for point_id in relative_small_ids if point_id in point_by_id],
        "all_flagged_ids": [point_id for point_id in all_flagged_ids if point_id in point_by_id],
        "review_fraction": float(quality_rows[0].get("quality_review_fraction") or 0.05) if quality_rows else 0.05,
    }
    summary = result.setdefault("summary", {})
    summary["quality_recipe"] = quality_rows[0].get("quality_recipe") if quality_rows else None
    summary["quality_review_budget"] = len(review_ids)
    summary["quality_flagged_count"] = len(ordered)
    summary["tiny_object_count"] = sum(bool(row.get("tiny_object")) for row in quality_rows)
    summary["low_source_detail_count"] = sum(bool(row.get("low_source_detail")) for row in quality_rows)
    summary["size_evidence"] = {
        "schema": "class-analysis-size-evidence-v2",
        "low_detail_min_side_px": (
            float(quality_rows[0].get("low_detail_min_side_px") or QUALITY_LOW_DETAIL_MIN_SIDE_PX)
            if quality_rows
            else float(QUALITY_LOW_DETAIL_MIN_SIDE_PX)
        ),
        "tiny_object_semantics": "legacy_alias_of_low_source_detail",
        "low_source_detail_semantics": "absolute_bbox_min_side_pixels",
    }
    summary["quality_signal_available_count"] = sum(bool(row.get("quality_signal_available")) for row in quality_rows)
    return result


def adaptive_ranking_features(point: Mapping[str, Any]) -> list[float]:
    return [
        float(point.get("quality_score") or point.get("review_priority_score") or 0.0),
        float(point.get("compact_logistic_disagreement") or 0.0),
        float(point.get("compact_neighbor_disagreement") or 0.0),
        float(point.get("cradio_logistic_disagreement") or 0.0),
        float(point.get("cradio_neighbor_disagreement") or 0.0),
        float(point.get("el2n_score") or 0.0),
        float(point.get("same_class_neighbor_ratio") or 0.0),
        1.0 if point.get("proposed_class_differs") else 0.0,
        1.0 if point.get("tiny_object") else 0.0,
    ]


def build_adaptive_review_ranking(
    points: Sequence[Mapping[str, Any]],
    *,
    seed: int = 17,
    minimum_receipts: int = 30,
    minimum_relabels: int = 3,
    minimum_audits: int = 3,
    minimum_classes: int = 2,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    def normalized_disposition(point: Mapping[str, Any]) -> str:
        raw = str(
            point.get("review_disposition")
            or point.get("human_review_disposition")
            or ""
        ).strip().lower()
        return {
            "confirm_current": "confirm",
            "reassign_class": "reassign",
        }.get(raw, raw)

    reviewable = [
        point
        for point in points
        if isinstance(point, Mapping)
        and normalized_disposition(point) not in {"dismiss", "skip"}
    ]
    receipts = [
        point
        for point in reviewable
        if normalized_disposition(point) in {"confirm", "reassign"}
    ]
    relabel_count = sum(normalized_disposition(point) == "reassign" for point in receipts)
    audit_count = sum(str(point.get("review_sampling_source") or "") == "audit" for point in receipts)
    receipt_classes = {
        str(point.get("class_id") or point.get("class_name") or point.get("label") or "")
        for point in receipts
    }
    readiness = {
        "receipt_count": len(receipts),
        "relabel_count": relabel_count,
        "audit_outcome_count": audit_count,
        "class_count": len(receipt_classes - {""}),
        "minimum_receipts": minimum_receipts,
        "minimum_relabels": minimum_relabels,
        "minimum_audits": minimum_audits,
        "minimum_classes": minimum_classes,
    }
    ready = (
        len(receipts) >= minimum_receipts
        and relabel_count >= minimum_relabels
        and audit_count >= minimum_audits
        and len(receipt_classes - {""}) >= minimum_classes
    )
    readiness["ready"] = ready

    pending = [
        point
        for point in reviewable
        if not normalized_disposition(point)
        and not bool(point.get("reviewed"))
    ]
    if not pending:
        return {"ready": ready, "readiness": readiness, "ordered_point_ids": [], "audit_point_ids": [], "ranking": []}
    ids = [stable_record_id(point, index) for index, point in enumerate(pending)]

    def strong_flag_sources(point: Mapping[str, Any]) -> list[str]:
        sources: list[str] = []
        checks = (
            ("quality", point.get("quality_review_candidate")),
            ("legacy", point.get("is_wrong_class_candidate")),
            ("rough", point.get("is_rough_outlier_candidate")),
            ("review", point.get("review_candidate")),
            ("overlap", point.get("is_close_overlap_candidate")),
            ("dual_bbox", point.get("is_dual_bbox_conflict")),
        )
        for name, enabled in checks:
            if bool(enabled):
                sources.append(name)
        if point.get("quality_flag_sources"):
            sources.append("quality_flag_sources")
        if point.get("review_signals"):
            sources.append("review_signals")
        refined = point.get("refined_outlier")
        if isinstance(refined, Mapping) and bool(
            refined.get("review_candidate")
            or refined.get("selected")
            or refined.get("actionable")
        ):
            sources.append("deep_evidence")
        return sorted(set(sources))

    def interleave_real_audits(
        scores: np.ndarray,
    ) -> tuple[list[int], list[str], list[dict[str, Any]], dict[str, int]]:
        ranked = sorted(range(len(pending)), key=lambda index: (-scores[index], ids[index]))
        explicit_priority = [
            index
            for index in ranked
            if strong_flag_sources(pending[index])
        ]
        priority = explicit_priority
        priority_set = set(priority)
        audit_pool = [
            index
            for index in ranked
            if index not in priority_set
            and not bool(
                pending[index].get("tiny_object")
                or pending[index].get("low_source_detail")
            )
        ]
        strata: dict[tuple[str, str, str], list[int]] = {}
        for index in audit_pool:
            key = (
                record_label(pending[index]),
                record_source_group(pending[index], index),
                str(pending[index].get("bbox_global_quintile") or "unknown"),
            )
            strata.setdefault(key, []).append(index)
        for key, values in strata.items():
            values.sort(key=lambda index: (_stable_u64(ids[index], seed + 503), ids[index]))
        stratum_keys = sorted(
            strata,
            key=lambda key: (_stable_u64(":".join(key), seed + 509), key),
        )
        audit_target = min(
            len(audit_pool),
            max(3 if not ready else 1, int(math.ceil(len(priority) / 9.0))),
        )
        selected_audit_indices: list[int] = []
        while len(selected_audit_indices) < audit_target and stratum_keys:
            next_keys: list[tuple[str, str, str]] = []
            for key in stratum_keys:
                if strata[key] and len(selected_audit_indices) < audit_target:
                    selected_audit_indices.append(strata[key].pop(0))
                if strata[key]:
                    next_keys.append(key)
            stratum_keys = next_keys
        ordered_indices: list[int] = []
        selected_audits: list[str] = []
        audit_cursor = 0
        for offset in range(0, len(priority), 9):
            ordered_indices.extend(priority[offset : offset + 9])
            if audit_cursor < len(selected_audit_indices):
                audit_index = selected_audit_indices[audit_cursor]
                audit_cursor += 1
                ordered_indices.append(audit_index)
                selected_audits.append(ids[audit_index])
        while audit_cursor < len(selected_audit_indices):
            audit_index = selected_audit_indices[audit_cursor]
            audit_cursor += 1
            ordered_indices.append(audit_index)
            selected_audits.append(ids[audit_index])
        audit_strata = [
            {
                "point_id": ids[index],
                "class": record_label(pending[index]),
                "source_group": record_source_group(pending[index], index),
                "size_band": str(pending[index].get("bbox_global_quintile") or "unknown"),
            }
            for index in selected_audit_indices
        ]
        source_counts: dict[str, int] = {}
        for index in priority:
            for source in strong_flag_sources(pending[index]):
                source_counts[source] = source_counts.get(source, 0) + 1
        return ordered_indices, selected_audits, audit_strata, source_counts

    baseline = np.asarray(
        [float(point.get("review_priority_score") or point.get("quality_score") or 0.0) for point in pending],
        dtype=np.float64,
    )
    if not ready:
        baseline_ranks = _percentile_ranks(baseline)
        ordered, audit_ids, audit_strata, source_counts = interleave_real_audits(baseline_ranks)
        audit_set = set(audit_ids)
        return {
            "ready": False,
            "readiness": readiness,
            "ordered_point_ids": [ids[index] for index in ordered],
            "audit_point_ids": audit_ids,
            "audit_strata": audit_strata,
            "priority_source_counts": source_counts,
            "ranking": [
                {
                    "point_id": ids[index],
                    "adaptive_review_score": float(baseline_ranks[index]),
                    "learned_error_probability": None,
                    "review_sampling_source": "audit" if ids[index] in audit_set else "priority",
                }
                for index in ordered
            ],
            "model": {
                "type": "bootstrap_baseline",
                "audit_interval": 9,
                "audit_population": "all_non_priority_real_objects",
            },
            "population_count": len(pending),
            "scheduled_count": len(ordered),
        }

    receipt_features = np.asarray([adaptive_ranking_features(point) for point in receipts], dtype=np.float64)
    outcomes = np.asarray(
        [1 if normalized_disposition(point) == "reassign" else 0 for point in receipts],
        dtype=np.int64,
    )
    if np.unique(outcomes).size < 2:
        readiness["ready"] = False
        readiness["reason"] = "both confirm and reassign outcomes are required"
        baseline_ranks = _percentile_ranks(baseline)
        ordered, audit_ids, audit_strata, source_counts = interleave_real_audits(baseline_ranks)
        audit_set = set(audit_ids)
        return {
            "ready": False,
            "readiness": readiness,
            "ordered_point_ids": [ids[index] for index in ordered],
            "audit_point_ids": audit_ids,
            "audit_strata": audit_strata,
            "priority_source_counts": source_counts,
            "ranking": [
                {
                    "point_id": ids[index],
                    "adaptive_review_score": float(baseline_ranks[index]),
                    "learned_error_probability": None,
                    "review_sampling_source": "audit" if ids[index] in audit_set else "priority",
                }
                for index in ordered
            ],
            "model": {"type": "bootstrap_baseline", "reason": readiness["reason"], "audit_interval": 9},
            "population_count": len(pending),
            "scheduled_count": len(ordered),
        }
    model = LogisticRegression(
        C=0.25,
        class_weight="balanced",
        max_iter=600,
        random_state=seed,
    )
    model.fit(receipt_features, outcomes)

    learned = model.predict_proba(
        np.asarray([adaptive_ranking_features(point) for point in pending], dtype=np.float64)
    )[:, 1]
    adaptive = 0.70 * _percentile_ranks(baseline) + 0.30 * _percentile_ranks(learned)
    ordered, audit_ids, audit_strata, source_counts = interleave_real_audits(adaptive)
    audit_set = set(audit_ids)
    ranking_rows = [
        {
            "point_id": ids[index],
            "adaptive_review_score": float(adaptive[index]),
            "learned_error_probability": float(learned[index]),
            "review_sampling_source": "audit" if ids[index] in audit_set else "priority",
        }
        for index in ordered
    ]
    digest_source = "|".join(
        f"{stable_record_id(point, index)}:{point.get('human_review_revision', '')}:{normalized_disposition(point)}"
        for index, point in enumerate(receipts)
    )
    return {
        "ready": True,
        "readiness": readiness,
        "ordered_point_ids": [ids[index] for index in ordered],
        "audit_point_ids": audit_ids,
        "audit_strata": audit_strata,
        "priority_source_counts": source_counts,
        "ranking": ranking_rows,
        "receipt_digest": hashlib.sha256(digest_source.encode("utf-8")).hexdigest(),
        "model": {
            "type": "logistic_regression",
            "C": 0.25,
            "baseline_weight": 0.70,
            "learned_weight": 0.30,
            "audit_interval": 9,
            "audit_population": "all_non_priority_real_objects",
        },
        "population_count": len(pending),
        "scheduled_count": len(ordered),
    }
