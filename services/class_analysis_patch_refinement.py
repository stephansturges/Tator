"""Deterministic spatial qualification for class-analysis outlier candidates.

The class-analysis graph intentionally remains a pooled-embedding discovery
mechanism.  This module consumes only the resulting rough candidates and a
bounded, source-balanced set of DINOv3 spatial tokens.  It never mutates labels
and it deliberately fails closed to ``unresolved`` when reference support or
spatial evidence is inadequate.

The orchestration layer owns image loading, model execution, cache paths and
progress reporting.  Keeping those concerns outside this module makes the
scoring logic testable without loading a vision backbone.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from services.class_analysis_selector_v6 import SELECTOR_PRIORITY_CONTRACT


REFINEMENT_SCHEMA = "class-analysis-patch-refinement-v5"
# The artifact schema describes field/layout compatibility.  Decision behavior
# is versioned independently so a status-logic fix invalidates completed-job
# reuse without forcing a rebuild of compatible image or reference-bank data.
REFINEMENT_DECISION_CONTRACT = "class-analysis-patch-decision-v9"
CALIBRATION_STATUS_SOURCE_AWARE = (
    "source_heldout_exact_two_view_directed_pair_probe_v5"
)
SOURCE_EXCLUSION_CONTRACT = (
    "global-stratified-heldout-source-role-exact-view-query-loso-v8"
)
COMPETITOR_BANK_CONTRACT = (
    "intrinsic-own-background-exact-two-view-pair-probe-v8"
)
PAIR_PROBE_CONTRACT = (
    "source-disjoint-exact-two-view-paired-exclusive-fit-thresholds-angle-grid-l2-sign-v4"
)
PAIR_PROBE_VIEW_CONTRACT = (
    "tight-context-weighted-target-mass-paired-exclusive-mean-v1"
)
PAIR_PROBE_LOWER_BOUND_CONTRACT = (
    "hanley-mcneil-shrunk-two-sided-95-v1"
)
DIAGNOSTIC_PAIR_RELIABILITY_CONTRACT = (
    "source-disjoint-exact-two-view-diagnostic-pair-v1"
)
HUMAN_REVIEW_QUALIFICATION_CONTRACT = (
    "class-analysis-qualified-human-review-v1"
)
HUMAN_REVIEW_RANK_CONTRACT = (
    "confirmed-band-stage1-suspicion-probe-excess-v1"
)
FREQUENT_OVERLAP_PRIOR_CONTRACT = (
    "capture-aware-directed-class-overlap-trusted-label-stratified-beta-smoothed-loo-v4"
)
CAPTURE_GROUP_CONTRACT = (
    "explicit-exporter-sequence-perceptual-cluster-tiered-independence-v2"
)
FREQUENT_OVERLAP_FIT_ELIGIBILITY_CONTRACT = (
    "exclude-stage1-rough-labels-both-directed-roles-v1"
)
FREQUENT_OVERLAP_FIT_REGISTRY_CONTRACT = (
    "capture-overlap-sufficient-statistics-digest-v1"
)
FREQUENT_OVERLAP_TRIAGE_CONTRACT = (
    "review-unresolved-annotated-overlap-frequency-rank-only-v2"
)
# An overlap prior is allowed to change review order, never candidate
# membership or semantic status.  Source-level minimums and a conservative
# confidence bound keep a visually plausible coincidence in a handful of
# images from becoming a dataset-wide rule.
FREQUENT_OVERLAP_MIN_SOURCE_GROUPS = 20
FREQUENT_OVERLAP_MIN_POSITIVE_SOURCE_GROUPS = 5
FREQUENT_OVERLAP_MIN_WILSON_LOWER_BOUND = 0.10
FREQUENT_OVERLAP_BETA_ALPHA = 1.0
FREQUENT_OVERLAP_BETA_BETA = 3.0
FREQUENT_OVERLAP_MAX_PRIORITY_ADJUSTMENT = 0.35
FREQUENT_DUPLICATE_OVERLAP_MAX_PRIORITY_ADJUSTMENT = 0.15
FREQUENT_OVERLAP_MIN_TARGET_COVERAGE = 0.20
FREQUENT_OVERLAP_MIN_IOU = 0.10
FREQUENT_OVERLAP_LOWER_CONFIDENCE_MIN_GROUPS = 80
FREQUENT_OVERLAP_LOWER_CONFIDENCE_MIN_POSITIVE_GROUPS = 20
FREQUENT_OVERLAP_LOWER_CONFIDENCE_MIN_WILSON_LOWER_BOUND = 0.15
FREQUENT_OVERLAP_LOWER_CONFIDENCE_MAX_PRIORITY_ADJUSTMENT = 0.18
FREQUENT_DUPLICATE_LOWER_CONFIDENCE_MAX_PRIORITY_ADJUSTMENT = 0.075
FREQUENT_OVERLAP_PROVISIONAL_MIN_GROUPS = 200
FREQUENT_OVERLAP_PROVISIONAL_MIN_POSITIVE_GROUPS = 50
FREQUENT_OVERLAP_PROVISIONAL_MIN_WILSON_LOWER_BOUND = 0.20
FREQUENT_OVERLAP_PROVISIONAL_MAX_PRIORITY_ADJUSTMENT = 0.08
FREQUENT_DUPLICATE_PROVISIONAL_MAX_PRIORITY_ADJUSTMENT = 0.03
# This second, deliberately smaller gate is a review-order heuristic, not a
# correctness claim.  It exists because useful capture provenance is often
# too weak for the semantic overlap-explanation gate while still supporting a
# conservative observation such as "this annotated class pair commonly
# overlaps in this dataset".  In particular, the provisional thresholds are
# attainable for large still-image exports without pretending those images
# are source-independent captures.
FREQUENT_OVERLAP_TRIAGE_MIN_SOURCE_GROUPS = 20
FREQUENT_OVERLAP_TRIAGE_MIN_POSITIVE_SOURCE_GROUPS = 5
FREQUENT_OVERLAP_TRIAGE_MIN_WILSON_LOWER_BOUND = 0.05
FREQUENT_OVERLAP_TRIAGE_MAX_PRIORITY_ADJUSTMENT = 0.06
FREQUENT_OVERLAP_TRIAGE_LOWER_CONFIDENCE_MIN_GROUPS = 50
FREQUENT_OVERLAP_TRIAGE_LOWER_CONFIDENCE_MIN_POSITIVE_GROUPS = 10
FREQUENT_OVERLAP_TRIAGE_LOWER_CONFIDENCE_MIN_WILSON_LOWER_BOUND = 0.06
FREQUENT_OVERLAP_TRIAGE_LOWER_CONFIDENCE_MAX_PRIORITY_ADJUSTMENT = 0.035
FREQUENT_OVERLAP_TRIAGE_PROVISIONAL_MIN_GROUPS = 100
FREQUENT_OVERLAP_TRIAGE_PROVISIONAL_MIN_POSITIVE_GROUPS = 20
FREQUENT_OVERLAP_TRIAGE_PROVISIONAL_MIN_WILSON_LOWER_BOUND = 0.08
FREQUENT_OVERLAP_TRIAGE_PROVISIONAL_MAX_PRIORITY_ADJUSTMENT = 0.02
CAPTURE_PERCEPTUAL_HASH_BITS = 128
CAPTURE_PERCEPTUAL_MAX_HAMMING_DISTANCE = 12
PAIR_PROBE_FEATURE_NAMES = (
    "current_patch_exclusive_support",
    "alternative_patch_exclusive_support",
)
PAIR_PROBE_ANGLE_COUNT = 181
PAIR_PROBE_ANGLE_REGULARIZATION = 0.02
PAIR_PROBE_MAX_FOLDS = 5
# Historical intrinsic-bank construction can retain a pair at its 0.70 AUROC
# floor. The exact-view human-review diagnostic intentionally keeps the same
# stricter 0.80 point-estimate and 0.60 lower-bound floors as confirmation; it
# broadens coverage only by separating discrimination quality from the two
# destructive operating-point yield checks below.
MIN_POSITIVE_CONFIRMATION_PAIR_PROBE_AUROC = 0.80
MIN_POSITIVE_CONFIRMATION_PAIR_PROBE_AUROC_LOWER_BOUND = 0.60
MIN_PAIR_PROBE_FIT_SOURCES_PER_CLASS = 8
MIN_PAIR_PROBE_EVAL_SOURCES_PER_CLASS = 8
MIN_PAIR_CALIBRATION_HELDOUT_SOURCE_GROUPS = (
    MIN_PAIR_PROBE_FIT_SOURCES_PER_CLASS
    + MIN_PAIR_PROBE_EVAL_SOURCES_PER_CLASS
)
PAIR_PROBE_SCORE_ABS_BOUND = 4.0 * math.sqrt(2.0)
SOURCE_GROUP_SEMANTICS = "image_sha256_or_split_relpath"
LEGACY_CALIBRATION_STATUS = "heldout_source_margin_v1"
LEGACY_CALIBRATION_STATUSES = frozenset(
    {
        LEGACY_CALIBRATION_STATUS,
        "heldout_source_margin_loso_v2",
        "global_heldout_source_margin_loso_v3",
        "global_heldout_source_margin_loso_v4",
        "global_stratified_heldout_source_margin_loso_v5",
        "source_heldout_bag_intrinsic_directed_pair_v1",
        "source_heldout_bag_intrinsic_directed_pair_probe_v2",
    }
)
MIN_SOURCE_INDEPENDENT_PROTOTYPES = 3
MIN_SOURCE_INDEPENDENT_GROUPS = 2
MIN_RELIABLE_FIT_SOURCE_GROUPS = MIN_SOURCE_INDEPENDENT_GROUPS + 1
MIN_HELDOUT_SOURCE_GROUPS = 2
MIN_HELDOUT_TARGET_SOURCE_PASS_FRACTION = 0.75
MIN_PAIR_PROBE_METRIC_FRACTION = 0.50
MIN_RELIABLE_TOTAL_SOURCE_GROUPS = (
    MIN_RELIABLE_FIT_SOURCE_GROUPS + MIN_HELDOUT_SOURCE_GROUPS
)
MIN_PAIR_CALIBRATION_TOTAL_SOURCE_GROUPS = (
    MIN_RELIABLE_FIT_SOURCE_GROUPS
    + MIN_PAIR_CALIBRATION_HELDOUT_SOURCE_GROUPS
)
GLOBAL_HELDOUT_SEARCH_STATE_LIMIT = 50_000

STATUS_CONFIRMED_OUTLIER = "confirmed_outlier"
STATUS_EXPLAINED_NOT_OUTLIER = "explained_not_outlier"
STATUS_MIXED_OR_COMPOSITE = "mixed_or_composite"
STATUS_UNRESOLVED = "unresolved"
STATUS_PAIR_CONFLICT = "pair_conflict"
REFINEMENT_STATUSES = (
    STATUS_CONFIRMED_OUTLIER,
    STATUS_EXPLAINED_NOT_OUTLIER,
    STATUS_MIXED_OR_COMPOSITE,
    STATUS_UNRESOLVED,
    STATUS_PAIR_CONFLICT,
)


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("spatial_tokens_must_be_2d")
    if not np.all(np.isfinite(array)):
        raise ValueError("class_analysis_refinement_tokens_nonfinite")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    normalised = array / np.maximum(norms, 1e-12)
    if not np.all(np.isfinite(normalised)):
        raise ValueError("class_analysis_refinement_tokens_nonfinite")
    return normalised


def _source_fingerprint(source_key: str) -> str:
    """Return the bounded source identifier stored beside prototype rows."""

    clean_source = str(source_key or "").strip()
    if not clean_source:
        return ""
    return hashlib.sha256(clean_source.encode("utf-8")).hexdigest()[:16]


def _calibration_source_split_digest(
    heldout_source_ids: Sequence[str],
    fit_source_ids: Sequence[str],
) -> str:
    hasher = hashlib.sha256()
    for value in (
        CALIBRATION_STATUS_SOURCE_AWARE,
        "heldout",
        *sorted(str(item) for item in heldout_source_ids),
        "fit",
        *sorted(str(item) for item in fit_source_ids),
    ):
        encoded = str(value).encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.hexdigest()


def _global_heldout_sources(
    source_names: Iterable[str],
    *,
    source_groups: Optional[Iterable[Iterable[str]]] = None,
) -> set[str]:
    """Choose one deterministic, class-stratified global source split.

    A per-class fold is not globally held out: a source reserved while
    calibrating class A can otherwise enter PCA or prototypes through class B.
    Splitting the union once makes the no-fit guarantee true for the complete
    reference bank, including shared source images containing multiple classes.
    When class target/background groups are supplied, the same global split is
    repaired so feasible groups retain both held-out and fit support.
    """

    ordered = sorted(
        {
            str(source).strip()
            for source in source_names
            if str(source).strip()
        },
        key=lambda source: (
            hashlib.sha256(source.encode("utf-8")).hexdigest(),
            source,
        ),
    )
    heldout_count = (
        min(
            len(ordered) - 1,
            max(1, int(math.ceil(len(ordered) * 0.20))),
        )
        if len(ordered) >= 2
        else 0
    )
    selected = set(ordered[:heldout_count])
    if source_groups is None:
        return selected

    universe = set(ordered)
    groups: List[frozenset[str]] = []
    seen_groups: set[frozenset[str]] = set()
    for raw_group in source_groups:
        group = frozenset(
            str(source).strip()
            for source in raw_group
            if str(source).strip() in universe
        )
        # A one-source group cannot be both globally held out and represented
        # in the fit bank. Leave it to the reliability gate to fail closed.
        if len(group) < 2 or group in seen_groups:
            continue
        seen_groups.add(group)
        groups.append(group)
    if not groups:
        return selected

    rank = {source: index for index, source in enumerate(ordered)}
    full_mask = (1 << len(ordered)) - 1
    base_mask = sum(1 << rank[source] for source in selected)

    constraints: List[Tuple[int, int, int]] = []
    for group in sorted(
        groups,
        key=lambda value: (len(value), tuple(sorted(value))),
    ):
        mask = sum(1 << rank[source] for source in group)
        # A confirm-capable exact-view pair needs eight fit and eight untouched
        # evaluation sources for *each* class. Reserve that full sixteen-source
        # pool whenever the class can also retain the three fit sources required
        # by the intrinsic prototype bank. Smaller groups retain the intrinsic
        # bank's ordinary heldout/fit constraint and fail the pair gate closed.
        if len(group) >= MIN_PAIR_CALIBRATION_TOTAL_SOURCE_GROUPS:
            constraints.append(
                (
                    mask,
                    MIN_PAIR_CALIBRATION_HELDOUT_SOURCE_GROUPS,
                    MIN_RELIABLE_FIT_SOURCE_GROUPS,
                )
            )
        elif len(group) >= MIN_RELIABLE_TOTAL_SOURCE_GROUPS:
            constraints.append(
                (
                    mask,
                    MIN_HELDOUT_SOURCE_GROUPS,
                    MIN_RELIABLE_FIT_SOURCE_GROUPS,
                )
            )
        else:
            constraints.append((mask, 1, 1))
    # Preserve at least one global fit and calibration source even if a caller
    # supplies groups that omit part of the source universe.
    constraints.append((full_mask, 1, 1))

    def violated_constraints(mask: int) -> List[Tuple[int, int, int]]:
        return [
            (group_mask, minimum_selected, minimum_fit)
            for group_mask, minimum_selected, minimum_fit in constraints
            if (
                (mask & group_mask).bit_count() < minimum_selected
                or group_mask.bit_count() - (mask & group_mask).bit_count()
                < minimum_fit
            )
        ]

    def repair_bits_for_constraint(
        mask: int,
        constraint: Tuple[int, int, int],
    ) -> int:
        group_mask, minimum_selected, minimum_fit = constraint
        selected_count = (mask & group_mask).bit_count()
        if selected_count < minimum_selected:
            return group_mask & ~mask
        if group_mask.bit_count() - selected_count < minimum_fit:
            return group_mask & mask
        return 0

    def violation_count(mask: int) -> int:
        return len(violated_constraints(mask))

    def next_violated_constraint(
        mask: int,
    ) -> Optional[Tuple[int, int, int]]:
        violations = violated_constraints(mask)
        if not violations:
            return None
        # Branch on the violation with the fewest legal one-bit repairs. This
        # fail-first heuristic prevents independent groups from consuming the
        # bounded search budget in breadth before any full repair is reached.
        return min(
            violations,
            key=lambda constraint: (
                repair_bits_for_constraint(mask, constraint).bit_count(),
                constraint[0].bit_count(),
                constraint,
            ),
        )

    def selected_sources(mask: int) -> set[str]:
        return {
            source
            for source, source_rank in rank.items()
            if mask & (1 << source_rank)
        }

    # Bounded deterministic depth-first repair. Every branch flips a member of
    # a violated constraint, which every satisfying assignment must change.
    # Children that reduce violations while staying near the original 20% fold
    # are explored first. If the budget is exhausted, the least-violating split
    # is returned and every affected class fails its downstream source gates.
    target_count = heldout_count
    stack: List[int] = [base_mask]
    seen_masks = {base_mask}
    best_mask = base_mask
    best_score = (
        violation_count(base_mask),
        0,
        abs(base_mask.bit_count() - target_count),
        base_mask,
    )
    visited = 0
    while stack and visited < GLOBAL_HELDOUT_SEARCH_STATE_LIMIT:
        mask = stack.pop()
        visited += 1
        violation = next_violated_constraint(mask)
        if violation is None:
            return selected_sources(mask)
        candidate_bits = repair_bits_for_constraint(mask, violation)
        children: List[Tuple[Tuple[int, int, int, int], int]] = []
        while candidate_bits:
            bit = candidate_bits & -candidate_bits
            candidate_bits ^= bit
            next_mask = mask ^ bit
            if next_mask in seen_masks:
                continue
            seen_masks.add(next_mask)
            distance = (next_mask ^ base_mask).bit_count()
            size_delta = abs(next_mask.bit_count() - target_count)
            score = (
                violation_count(next_mask),
                distance,
                size_delta,
                next_mask,
            )
            if score < best_score:
                best_score = score
                best_mask = next_mask
            children.append((score, next_mask))
        # The stack pops from the end, so append siblings in reverse priority.
        children.sort(key=lambda item: item[0], reverse=True)
        stack.extend(next_mask for _score, next_mask in children)
    return selected_sources(best_mask)


def _resolve_reliability_active_set(
    intrinsic_eligible: set[str],
    evaluate_active_set: Callable[[set[str], set[str]], set[str]],
) -> Tuple[set[str], bool]:
    """Find a bounded self-consistent class set with fail-closed cycles.

    The callback receives the active competitor set and the class set to
    reevaluate. During ordinary convergence every intrinsically eligible class
    can re-enter after a bad competitor drops. On a cycle or state-budget
    exhaustion, only the intersection of the cycle survives and is then
    monotonically revalidated, so oscillating classes cannot publish as usable.
    """

    eligible = set(intrinsic_eligible)
    if not eligible:
        return set(), False
    active = set(eligible)
    seen_states: Dict[frozenset[str], int] = {}
    history: List[frozenset[str]] = []
    state_limit = max(8, min(128, 4 * len(eligible) + 8))
    cycle_survivors: Optional[set[str]] = None
    cycle_or_limit = False
    for _iteration in range(state_limit):
        state = frozenset(active)
        if state in seen_states:
            cycle_start = seen_states[state]
            cycle_states = history[cycle_start:]
            cycle_survivors = (
                set.intersection(*(set(value) for value in cycle_states))
                if cycle_states
                else set()
            )
            cycle_or_limit = True
            break
        seen_states[state] = len(history)
        history.append(state)
        next_active = (
            set(evaluate_active_set(set(active), set(eligible))) & eligible
        )
        if next_active == active:
            return active, False
        active = next_active
    else:
        cycle_survivors = (
            set.intersection(*(set(value) for value in history))
            if history
            else set()
        )
        cycle_or_limit = True

    active = set(cycle_survivors or set())
    while True:
        next_active = (
            set(evaluate_active_set(set(active), set(active))) & active
        )
        if next_active == active:
            return active, cycle_or_limit
        active = next_active


def _exclude_source_prototypes(
    prototypes: np.ndarray,
    source_ids: np.ndarray,
    *,
    exclude_source_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exclude query-source and unprovenanced rows from a prototype pool."""

    values = np.asarray(prototypes, dtype=np.float32)
    sources = np.asarray(source_ids).astype(str, copy=False).reshape(-1)
    if values.ndim != 2 or sources.shape != (values.shape[0],):
        raise ValueError("class_analysis_refinement_bank_provenance_invalid")
    clean_source = str(exclude_source_key or "").strip()
    keep = sources != ""
    if clean_source:
        excluded_id = _source_fingerprint(clean_source)
        keep &= sources != excluded_id
    return values[keep], sources[keep]


def _prototype_pool_is_source_independent(
    prototypes: np.ndarray,
    source_ids: np.ndarray,
) -> bool:
    """Require enough rows and source groups for the mean-top-3 statistic."""

    values = np.asarray(prototypes)
    sources = np.asarray(source_ids).astype(str, copy=False).reshape(-1)
    if values.ndim != 2 or sources.shape != (values.shape[0],):
        return False
    nonempty_sources = {source for source in sources.tolist() if source}
    return bool(
        values.shape[0] >= MIN_SOURCE_INDEPENDENT_PROTOTYPES
        and len(nonempty_sources) >= MIN_SOURCE_INDEPENDENT_GROUPS
    )


def _assigned_cluster_medoid_indices(
    values: np.ndarray,
    centres: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Select one unique real exemplar from each non-empty assigned cluster."""

    projected = _normalise_rows(np.asarray(values, dtype=np.float32))
    normalized_centres = _normalise_rows(
        np.asarray(centres, dtype=np.float32)
    )
    assignments = np.asarray(labels, dtype=np.int64).reshape(-1)
    if assignments.shape != (projected.shape[0],):
        raise ValueError("class_analysis_refinement_cluster_labels_invalid")
    selected: List[int] = []
    for cluster_index, centre in enumerate(normalized_centres):
        members = np.flatnonzero(assignments == cluster_index)
        if members.size <= 0:
            continue
        similarities = projected[members] @ centre
        selected.append(int(members[int(np.argmax(similarities))]))
    return np.asarray(selected, dtype=np.int64)


def _source_balanced_cluster_exemplar_indices(
    values: np.ndarray,
    centres: np.ndarray,
    labels: np.ndarray,
    source_ids: np.ndarray,
    *,
    limit: int,
) -> np.ndarray:
    """Select a bounded, source-balanced set of real cluster exemplars.

    Selecting only one medoid per non-empty cluster loses provenance when many
    distinct exact-image source groups contain the same visual pattern: duplicate points can
    collapse into one KMeans cluster and make a well-supported class look like
    it came from one source.  This selector instead round-robins across sources
    and, within each source, prefers an under-represented assigned cluster and
    the row closest to that cluster centre.  It never invents duplicate rows or
    lets one source consume a second slot while another source still has an
    eligible row.
    """

    projected = _normalise_rows(np.asarray(values, dtype=np.float32))
    normalized_centres = _normalise_rows(
        np.asarray(centres, dtype=np.float32)
    )
    assignments = np.asarray(labels, dtype=np.int64).reshape(-1)
    sources = np.asarray(source_ids).astype(str, copy=False).reshape(-1)
    if (
        assignments.shape != (projected.shape[0],)
        or sources.shape != (projected.shape[0],)
        or np.any(assignments < 0)
        or np.any(assignments >= normalized_centres.shape[0])
    ):
        raise ValueError("class_analysis_refinement_cluster_labels_invalid")
    maximum = min(max(1, int(limit)), projected.shape[0])
    similarities = np.einsum(
        "ij,ij->i",
        projected,
        normalized_centres[assignments],
    )
    rows_by_source: Dict[str, List[int]] = {}
    for row_index, source_id in enumerate(sources.tolist()):
        if source_id:
            rows_by_source.setdefault(source_id, []).append(row_index)
    if not rows_by_source:
        return _assigned_cluster_medoid_indices(
            projected,
            normalized_centres,
            assignments,
        )[:maximum]

    selected: List[int] = []
    selected_set: set[int] = set()
    cluster_counts = np.zeros(normalized_centres.shape[0], dtype=np.int32)
    depth = 0
    ordered_sources = sorted(rows_by_source)
    while len(selected) < maximum:
        added = False
        for source_id in ordered_sources:
            candidates = [
                row_index
                for row_index in rows_by_source[source_id]
                if row_index not in selected_set
            ]
            if depth >= len(rows_by_source[source_id]) or not candidates:
                continue
            row_index = min(
                candidates,
                key=lambda candidate: (
                    int(cluster_counts[assignments[candidate]]),
                    -float(similarities[candidate]),
                    int(candidate),
                ),
            )
            selected.append(row_index)
            selected_set.add(row_index)
            cluster_counts[assignments[row_index]] += 1
            added = True
            if len(selected) >= maximum:
                break
        if not added:
            break
        depth += 1
    return np.asarray(selected, dtype=np.int64)


def _round_robin_class_rows(
    rows_by_class: Mapping[str, Sequence[Tuple[int, str, np.ndarray]]],
    *,
    limit: int,
) -> List[Tuple[int, str, np.ndarray]]:
    """Take a deterministic class-balanced prefix for shared fitting.

    Hash-ranking every patch globally lets large classes monopolize the PCA
    pool.  Round-robin selection preserves the per-class reservoir ordering
    while giving every represented class an equal opportunity at each depth.
    """

    maximum = max(1, int(limit))
    ordered = {
        str(class_name): sorted(
            list(rows),
            key=lambda row: (int(row[0]), str(row[1])),
        )
        for class_name, rows in rows_by_class.items()
        if rows
    }
    selected: List[Tuple[int, str, np.ndarray]] = []
    depth = 0
    while len(selected) < maximum:
        added = False
        for class_name in sorted(ordered):
            rows = ordered[class_name]
            if depth >= len(rows):
                continue
            selected.append(rows[depth])
            added = True
            if len(selected) >= maximum:
                break
        if not added:
            break
        depth += 1
    return selected


def _mean_top_source_similarity(
    projected_values: np.ndarray,
    prototypes: np.ndarray,
    source_ids: np.ndarray,
) -> np.ndarray:
    """Mean the best match from each of up to three distinct source groups."""

    values = np.asarray(projected_values, dtype=np.float32)
    references = np.asarray(prototypes, dtype=np.float32)
    sources = np.asarray(source_ids).astype(str, copy=False).reshape(-1)
    if values.ndim != 2:
        raise ValueError("class_analysis_refinement_projected_values_invalid")
    if (
        references.ndim != 2
        or references.shape[0] <= 0
        or sources.shape != (references.shape[0],)
    ):
        return np.full(values.shape[0], -1.0, dtype=np.float32)
    unique_sources = sorted({source for source in sources.tolist() if source})
    if not unique_sources:
        return np.full(values.shape[0], -1.0, dtype=np.float32)
    similarities = values @ _normalise_rows(references).T
    source_maxima = np.stack(
        [
            np.max(similarities[:, sources == source], axis=1)
            for source in unique_sources
        ],
        axis=1,
    )
    top_count = min(3, source_maxima.shape[1])
    if top_count == 1:
        return source_maxima[:, 0].astype(np.float32, copy=False)
    top = np.partition(
        source_maxima,
        -top_count,
        axis=1,
    )[:, -top_count:]
    return np.mean(top, axis=1, dtype=np.float32)


def _source_consensus_similarity(
    projected_values: np.ndarray,
    source_centroids: np.ndarray,
) -> np.ndarray:
    """Return the median similarity across distinct exact-image source centroids.

    Inter-class filtering is destructive: one repeated foreign-looking mode
    must not delete another class's clean target bank. A source-level median
    requires the mode to recur across the competitor's sources, while ordinary
    candidate scoring can still use sensitive top-k prototype evidence.
    """

    values = np.asarray(projected_values, dtype=np.float32)
    references = np.asarray(source_centroids, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("class_analysis_refinement_projected_values_invalid")
    if references.ndim != 2 or references.shape[0] <= 0:
        return np.full(values.shape[0], -1.0, dtype=np.float32)
    similarities = values @ _normalise_rows(references).T
    # Use the lower median so an even two-source pool requires both sources to
    # support the mode; arithmetic median would invent midpoint evidence from
    # one clean and one contaminated source.
    consensus_index = (similarities.shape[1] - 1) // 2
    consensus = np.partition(
        similarities,
        consensus_index,
        axis=1,
    )[:, consensus_index]
    return consensus.astype(np.float32, copy=False)


def _weighted_top_fraction_score(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    selected_fraction: float = 0.05,
) -> float:
    """Average the best values covering an exact fraction of target mass."""

    scores = np.asarray(values, dtype=np.float32).reshape(-1)
    mass = np.asarray(weights, dtype=np.float32).reshape(-1)
    if scores.shape != mass.shape:
        raise ValueError("class_analysis_refinement_weight_shape_invalid")
    finite = np.isfinite(scores) & np.isfinite(mass) & (mass > 0.0)
    scores = scores[finite]
    mass = np.clip(mass[finite], 0.0, 1.0)
    total_mass = float(mass.sum())
    if scores.size <= 0 or total_mass <= 1e-12:
        return -1.0
    fraction = min(1.0, max(1e-6, float(selected_fraction)))
    requested_mass = max(1e-12, total_mass * fraction)
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_mass = mass[order]
    cumulative_before = np.concatenate(
        [
            np.zeros(1, dtype=np.float32),
            np.cumsum(ordered_mass[:-1], dtype=np.float32),
        ]
    )
    selected_mass = np.clip(
        requested_mass - cumulative_before,
        0.0,
        ordered_mass,
    )
    used_mass = float(selected_mass.sum())
    if used_mass <= 1e-12:
        return -1.0
    return float(
        np.sum(ordered_scores * selected_mass, dtype=np.float32)
        / used_mass
    )


def _source_bag_score(
    values: np.ndarray,
    *,
    selected_fraction: float = 0.05,
) -> float:
    """Return the top-fraction intrinsic-margin score for one source bag.

    This shares its top-fraction aggregation with per-view inference.  It is
    not the complete inference statistic: calibration has one unordered bag
    per held-out source, while inference applies paired-view, spatial,
    overlap, and view-consistency gates.
    """

    scores = np.asarray(values, dtype=np.float32).reshape(-1)
    return _weighted_top_fraction_score(
        scores,
        np.ones(scores.shape, dtype=np.float32),
        selected_fraction=selected_fraction,
    )


def _source_bag_score_vector(
    bags_by_source: Mapping[str, np.ndarray],
    source_keys: Sequence[str],
    *,
    selected_fraction: float = 0.05,
) -> np.ndarray:
    """Reduce each source exactly once before fitting an operating point.

    Pair calibration retains aligned per-patch margins for its exclusivity
    probe.  Intrinsic-presence thresholds are nevertheless source-balanced:
    patch count within a source must not give that image more votes.
    """

    return np.asarray(
        [
            _source_bag_score(
                bags_by_source[source_key],
                selected_fraction=selected_fraction,
            )
            for source_key in source_keys
        ],
        dtype=np.float32,
    )


def _paired_patch_exclusive_bag_features(
    current_intrinsic_margins: np.ndarray,
    alternative_intrinsic_margins: np.ndarray,
    *,
    selected_fraction: float = 0.05,
) -> np.ndarray:
    """Preserve patch identity while measuring directed class exclusivity.

    Independent top-k pooling can select the strongest current-class patch and
    strongest alternative-class patch from different object parts.  That
    makes a clean object and a nested/composite crop share the same two scalar
    maxima.  Pairing first asks whether either class wins on the *same* patch;
    the two directed top-fraction summaries then remain compatible with the
    sign-constrained probe used for confirmation.
    """

    current = np.asarray(
        current_intrinsic_margins, dtype=np.float32
    ).reshape(-1)
    alternative = np.asarray(
        alternative_intrinsic_margins, dtype=np.float32
    ).reshape(-1)
    if current.shape != alternative.shape or current.size <= 0:
        raise ValueError(
            "class_analysis_pair_probe_paired_patch_shape_invalid"
        )
    finite = np.isfinite(current) & np.isfinite(alternative)
    if not np.any(finite):
        raise ValueError(
            "class_analysis_pair_probe_paired_patch_nonfinite"
        )
    current = current[finite]
    alternative = alternative[finite]
    return np.asarray(
        [
            _source_bag_score(
                current - alternative,
                selected_fraction=selected_fraction,
            ),
            _source_bag_score(
                alternative - current,
                selected_fraction=selected_fraction,
            ),
        ],
        dtype=np.float32,
    )


def exact_two_view_pair_features(
    current_margin_views: Sequence[np.ndarray],
    alternative_margin_views: Sequence[np.ndarray],
    target_masks: Sequence[np.ndarray],
    *,
    selected_fraction: float = 0.05,
) -> np.ndarray:
    """Return the exact deployed two-view paired-exclusive feature vector."""

    if not (
        len(current_margin_views)
        == len(alternative_margin_views)
        == len(target_masks)
        == 2
    ):
        raise ValueError(
            "class_analysis_pair_probe_exact_view_contract_invalid"
        )
    per_view: List[np.ndarray] = []
    for current_raw, alternative_raw, mask_raw in zip(
        current_margin_views,
        alternative_margin_views,
        target_masks,
    ):
        current = np.asarray(current_raw, dtype=np.float32).reshape(-1)
        alternative = np.asarray(
            alternative_raw, dtype=np.float32
        ).reshape(-1)
        mask = np.asarray(mask_raw, dtype=np.float32).reshape(-1)
        if (
            current.shape != alternative.shape
            or current.shape != mask.shape
            or current.size <= 0
            or not np.all(np.isfinite(current))
            or not np.all(np.isfinite(alternative))
            or not np.all(np.isfinite(mask))
            or np.any(mask < 0.0)
            or np.any(mask > 1.0)
            or float(mask.sum()) <= 1e-12
        ):
            raise ValueError(
                "class_analysis_pair_probe_exact_view_contract_invalid"
            )
        per_view.append(
            np.asarray(
                [
                    _weighted_top_fraction_score(
                        current - alternative,
                        mask,
                        selected_fraction=selected_fraction,
                    ),
                    _weighted_top_fraction_score(
                        alternative - current,
                        mask,
                        selected_fraction=selected_fraction,
                    ),
                ],
                dtype=np.float32,
            )
        )
    return np.mean(np.stack(per_view), axis=0, dtype=np.float32)


def _conservative_auroc_lower_bound(
    auroc: float,
    positive_count: int,
    negative_count: int,
) -> float:
    """Return a finite-sample-shrunk two-sided 95% Hanley-McNeil bound."""

    positives = max(0, int(positive_count))
    negatives = max(0, int(negative_count))
    if positives <= 0 or negatives <= 0 or not math.isfinite(float(auroc)):
        return 0.0
    pair_count = positives * negatives
    estimate = min(1.0, max(0.0, float(auroc)))
    # A small Jeffreys-style shrink prevents an observed 1.0 from publishing
    # a zero-variance bound on only a handful of sources.
    shrunk = (estimate * pair_count + 1.0) / (pair_count + 2.0)
    q1 = shrunk / max(1e-12, 2.0 - shrunk)
    q2 = (2.0 * shrunk * shrunk) / max(1e-12, 1.0 + shrunk)
    variance = (
        shrunk * (1.0 - shrunk)
        + (positives - 1) * (q1 - shrunk * shrunk)
        + (negatives - 1) * (q2 - shrunk * shrunk)
    ) / float(positives * negatives)
    return float(max(0.0, shrunk - 1.96 * math.sqrt(max(0.0, variance))))


def pair_metrics_are_reliable(
    *,
    current_class_reliable: bool,
    alternative_class_reliable: bool,
    fit_current_source_count: int,
    fit_alternative_source_count: int,
    eval_current_source_count: int,
    eval_alternative_source_count: int,
    eval_auroc: float,
    eval_auroc_lower_bound: float,
    fit_balanced_accuracy: float,
    eval_sensitivity: float,
    eval_specificity: float,
    current_absence_eval_fraction: float,
    alternative_strong_eval_fraction: float,
) -> bool:
    """Return whether a pair is safe for automatic positive confirmation.

    This deliberately includes the two operating-point yield checks.  Those
    checks answer whether the fitted thresholds can support the destructive
    positive-confirmation claim; they do not answer whether the source-
    disjoint pair probe is useful for ranking a human-review queue.  Keep that
    lower-risk question in :func:`pair_metrics_are_diagnostic`.
    """

    return bool(
        pair_metrics_are_diagnostic(
            current_class_reliable=current_class_reliable,
            alternative_class_reliable=alternative_class_reliable,
            fit_current_source_count=fit_current_source_count,
            fit_alternative_source_count=fit_alternative_source_count,
            eval_current_source_count=eval_current_source_count,
            eval_alternative_source_count=eval_alternative_source_count,
            eval_auroc=eval_auroc,
            eval_auroc_lower_bound=eval_auroc_lower_bound,
            fit_balanced_accuracy=fit_balanced_accuracy,
            eval_sensitivity=eval_sensitivity,
            eval_specificity=eval_specificity,
        )
        and math.isfinite(float(current_absence_eval_fraction))
        and math.isfinite(float(alternative_strong_eval_fraction))
        and float(current_absence_eval_fraction)
        >= MIN_PAIR_PROBE_METRIC_FRACTION
        and float(alternative_strong_eval_fraction)
        >= MIN_PAIR_PROBE_METRIC_FRACTION
    )


def pair_metrics_are_diagnostic(
    *,
    current_class_reliable: bool,
    alternative_class_reliable: bool,
    fit_current_source_count: int,
    fit_alternative_source_count: int,
    eval_current_source_count: int,
    eval_alternative_source_count: int,
    eval_auroc: float,
    eval_auroc_lower_bound: float,
    fit_balanced_accuracy: float,
    eval_sensitivity: float,
    eval_specificity: float,
) -> bool:
    """Return whether an exact-view pair probe is useful for human triage.

    Diagnostic reliability keeps every source-count, source-disjoint
    evaluation and discrimination-quality gate used by confirmation.  It
    intentionally excludes ``current_absence_eval_fraction`` and
    ``alternative_strong_eval_fraction``: those are operating-point yield
    checks and can fail for legitimate co-occurring classes even when the
    held-out probe separates the directed pair well.
    """

    source_counts = (
        fit_current_source_count,
        fit_alternative_source_count,
        eval_current_source_count,
        eval_alternative_source_count,
    )
    metrics = (
        eval_auroc,
        eval_auroc_lower_bound,
        fit_balanced_accuracy,
        eval_sensitivity,
        eval_specificity,
    )
    return bool(
        current_class_reliable
        and alternative_class_reliable
        and all(_is_non_bool_integral(value) for value in source_counts)
        and fit_current_source_count >= MIN_PAIR_PROBE_FIT_SOURCES_PER_CLASS
        and fit_alternative_source_count
        >= MIN_PAIR_PROBE_FIT_SOURCES_PER_CLASS
        and eval_current_source_count >= MIN_PAIR_PROBE_EVAL_SOURCES_PER_CLASS
        and eval_alternative_source_count
        >= MIN_PAIR_PROBE_EVAL_SOURCES_PER_CLASS
        and all(math.isfinite(float(value)) for value in metrics)
        and float(eval_auroc) >= MIN_POSITIVE_CONFIRMATION_PAIR_PROBE_AUROC
        and float(eval_auroc_lower_bound)
        >= MIN_POSITIVE_CONFIRMATION_PAIR_PROBE_AUROC_LOWER_BOUND
        and float(fit_balanced_accuracy)
        >= MIN_PAIR_PROBE_METRIC_FRACTION
        and float(eval_sensitivity)
        >= MIN_PAIR_PROBE_METRIC_FRACTION
        and float(eval_specificity)
        >= MIN_PAIR_PROBE_METRIC_FRACTION
    )


def _is_non_bool_integral(value: Any) -> bool:
    """Return whether persisted provenance is an actual integral scalar."""

    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _is_sha256_hex_digest(value: Any) -> bool:
    """Require the canonical lowercase encoding used by persisted SHA-256s."""

    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_finite_real(value: Any) -> bool:
    """Reject booleans and coercible strings from persisted numeric proofs."""

    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ) and math.isfinite(float(value))


def _as_finite_float32(value: Any) -> float:
    """Return the exact persisted threshold value or reject overflow."""

    converted = np.float32(value)
    if not np.isfinite(converted):
        raise ValueError("class_analysis_pair_threshold_nonfinite")
    return float(converted)


def _float32_strictly_below(value: Any) -> float:
    """Return the adjacent persisted float32 below ``value``.

    Thresholds are serialized as float32. Taking ``nextafter`` in float64 and
    casting afterwards can round straight back to the boundary, violating the
    absence/presence invariant only when a real streamed bank is finalized.
    """

    boundary = np.float32(value)
    if not np.isfinite(boundary):
        raise ValueError("class_analysis_pair_threshold_nonfinite")
    lowered = np.nextafter(boundary, np.float32(-np.inf))
    if not np.isfinite(lowered) or not lowered < boundary:
        raise ValueError("class_analysis_pair_threshold_boundary_invalid")
    return float(lowered)


def _balanced_source_operating_point(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    *,
    fallback: float,
) -> Tuple[float, float, float]:
    """Fit a held-out source-bag threshold without abstention-by-default.

    The threshold maximises Youden's J. Ties favour positive recall and then
    the lower threshold. The final return is the 95th-percentile negative
    boundary used by calibrated weak-evidence gates.
    """

    positives = np.asarray(positive_scores, dtype=np.float32).reshape(-1)
    negatives = np.asarray(negative_scores, dtype=np.float32).reshape(-1)
    positives = positives[np.isfinite(positives)]
    negatives = negatives[np.isfinite(negatives)]
    if positives.size <= 0 or negatives.size <= 0:
        clean_fallback = float(fallback)
        return clean_fallback, 0.0, clean_fallback
    observed = np.unique(np.concatenate([positives, negatives]))
    candidates: List[float] = [float(observed[0] - 1e-6)]
    candidates.extend(
        float((left + right) * 0.5)
        for left, right in zip(observed[:-1], observed[1:])
    )
    candidates.append(float(observed[-1] + 1e-6))
    best_key: Optional[Tuple[float, float, float]] = None
    threshold = float(fallback)
    for candidate in candidates:
        true_positive_rate = float(np.mean(positives >= candidate))
        false_positive_rate = float(np.mean(negatives >= candidate))
        key = (
            true_positive_rate - false_positive_rate,
            true_positive_rate,
            -candidate,
        )
        if best_key is None or key > best_key:
            best_key = key
            threshold = candidate
    auroc = _binary_auroc(positives, negatives)
    negative_boundary = float(np.quantile(negatives, 0.95))
    return threshold, auroc, negative_boundary


def _binary_auroc(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
) -> float:
    """Return deterministic tie-aware binary AUROC without a dependency.

    The pair probe is fitted while the reference bank is being constructed,
    including in lightweight workers where scikit-learn is optional.  Average
    ranks preserve the conventional half-credit treatment for tied scores.
    """

    positives = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    negatives = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    positives = positives[np.isfinite(positives)]
    negatives = negatives[np.isfinite(negatives)]
    if positives.size <= 0 or negatives.size <= 0:
        return 0.0
    values = np.concatenate([positives, negatives])
    labels = np.concatenate(
        [
            np.ones(positives.size, dtype=np.uint8),
            np.zeros(negatives.size, dtype=np.uint8),
        ]
    )
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_labels = labels[order]
    _unique, starts, tie_counts = np.unique(
        ordered_values,
        return_index=True,
        return_counts=True,
    )
    positive_counts = np.add.reduceat(
        ordered_labels.astype(np.float64, copy=False), starts
    )
    average_ranks = starts.astype(np.float64) + 0.5 * (
        tie_counts.astype(np.float64) + 1.0
    )
    positive_rank_sum = float(np.sum(average_ranks * positive_counts))
    return float(
        (
            positive_rank_sum
            - positives.size * (positives.size + 1) * 0.5
        )
        / float(positives.size * negatives.size)
    )


def _stable_source_group_folds(
    source_keys: Sequence[str],
    labels: Sequence[int],
    *,
    maximum_folds: int = PAIR_PROBE_MAX_FOLDS,
) -> Tuple[np.ndarray, int, str]:
    """Assign every exact-image source group to one stable cross-fit fold.

    A source may contribute objects to both classes.  Grouping happens before
    fold assignment, so every row from that source always remains on the same
    side of a train/held-out boundary.  The digest makes this property
    auditable in a persisted reference bank.
    """

    sources = np.asarray([str(value) for value in source_keys])
    binary_labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    if (
        sources.ndim != 1
        or sources.shape[0] != binary_labels.shape[0]
        or sources.size <= 0
        or np.any(~np.isin(binary_labels, np.asarray([0, 1])))
    ):
        return np.full(binary_labels.shape, -1, dtype=np.int32), 0, ""
    grouped_counts: Dict[str, np.ndarray] = {}
    for source, label in zip(sources.tolist(), binary_labels.tolist()):
        clean_source = str(source).strip()
        if not clean_source:
            return np.full(binary_labels.shape, -1, dtype=np.int32), 0, ""
        counts = grouped_counts.setdefault(
            clean_source, np.zeros(2, dtype=np.int64)
        )
        counts[int(label)] += 1
    carrying = [
        sum(int(counts[label] > 0) for counts in grouped_counts.values())
        for label in (0, 1)
    ]
    fold_count = min(max(0, int(maximum_folds)), min(carrying))
    if fold_count < 2:
        return np.full(binary_labels.shape, -1, dtype=np.int32), 0, ""

    # Seed every fold with both labels before balancing remaining groups.
    # With ``fold_count <= carrying[label]`` this construction is always
    # possible: mixed groups cover one fold each, and the remaining folds use
    # one disjoint negative-only plus one positive-only source.  This avoids a
    # greedy corner case where all support for a label accidentally landed in
    # one fold despite enough independent groups being available.
    def stable_group_order(source: str) -> Tuple[int, int, str, str]:
        return (
            -int(np.max(grouped_counts[source])),
            -int(np.sum(grouped_counts[source])),
            hashlib.sha256(source.encode("utf-8")).hexdigest(),
            source,
        )

    mixed_groups = sorted(
        [
            source
            for source, counts in grouped_counts.items()
            if counts[0] > 0 and counts[1] > 0
        ],
        key=stable_group_order,
    )
    negative_only_groups = sorted(
        [
            source
            for source, counts in grouped_counts.items()
            if counts[0] > 0 and counts[1] == 0
        ],
        key=stable_group_order,
    )
    positive_only_groups = sorted(
        [
            source
            for source, counts in grouped_counts.items()
            if counts[1] > 0 and counts[0] == 0
        ],
        key=stable_group_order,
    )
    fold_label_counts = np.zeros((fold_count, 2), dtype=np.int64)
    fold_group_counts = np.zeros(fold_count, dtype=np.int64)
    source_to_fold: Dict[str, int] = {}

    def assign(source: str, fold: int) -> None:
        source_to_fold[source] = int(fold)
        fold_label_counts[fold] += grouped_counts[source]
        fold_group_counts[fold] += 1

    mixed_seed_count = min(fold_count, len(mixed_groups))
    for fold, source in enumerate(mixed_groups[:mixed_seed_count]):
        assign(source, fold)
    uncovered_folds = list(range(mixed_seed_count, fold_count))
    if (
        len(negative_only_groups) < len(uncovered_folds)
        or len(positive_only_groups) < len(uncovered_folds)
    ):
        return np.full(binary_labels.shape, -1, dtype=np.int32), 0, ""
    for offset, fold in enumerate(uncovered_folds):
        assign(negative_only_groups[offset], fold)
        assign(positive_only_groups[offset], fold)

    ordered_groups = sorted(
        [source for source in grouped_counts if source not in source_to_fold],
        key=stable_group_order,
    )
    total_label_counts = np.maximum(
        1,
        np.sum(np.stack(list(grouped_counts.values())), axis=0),
    )
    for source in ordered_groups:
        counts = grouped_counts[source]
        best_fold = min(
            range(fold_count),
            key=lambda fold: (
                float(
                    np.max(
                        (fold_label_counts[fold] + counts)
                        / total_label_counts
                    )
                ),
                float(
                    np.sum(
                        (fold_label_counts[fold] + counts)
                        / total_label_counts
                    )
                ),
                int(fold_group_counts[fold]),
                fold,
            ),
        )
        assign(source, best_fold)

    assignments = np.asarray(
        [source_to_fold[str(source)] for source in sources.tolist()],
        dtype=np.int32,
    )
    # Every held-out fold and every training complement must retain both
    # labels.  This is stronger than leakage prevention alone and keeps OOF
    # operating-point metrics class-balanced in each source split.
    for fold in range(fold_count):
        heldout_labels = binary_labels[assignments == fold]
        train_labels = binary_labels[assignments != fold]
        if not (
            np.any(heldout_labels == 0)
            and np.any(heldout_labels == 1)
            and np.any(train_labels == 0)
            and np.any(train_labels == 1)
        ):
            return np.full(binary_labels.shape, -1, dtype=np.int32), 0, ""
    digest_payload = "\n".join(
        [PAIR_PROBE_CONTRACT]
        + [
            (
                f"{source}={source_to_fold[source]}:"
                f"{int(grouped_counts[source][0])},"
                f"{int(grouped_counts[source][1])}"
            )
            for source in sorted(source_to_fold)
        ]
    )
    digest = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()
    return assignments, fold_count, digest


def _select_sign_constrained_pair_weights(
    features: np.ndarray,
    labels: Sequence[int],
) -> Optional[np.ndarray]:
    """Fit a unit two-feature direction with semantically safe signs.

    The current-class intrinsic score may only oppose an outlier decision and
    the alternative-class score may only support it.  A small angular penalty
    keeps the probe near the ordinary ``alternative - current`` direction
    unless held-out source evidence supports a material reweighting.
    """

    values = np.asarray(features, dtype=np.float32)
    binary_labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or values.shape[0] != binary_labels.shape[0]
        or values.shape[0] <= 0
        or not np.all(np.isfinite(values))
        or not np.any(binary_labels == 0)
        or not np.any(binary_labels == 1)
    ):
        return None
    angles = np.linspace(
        0.0,
        0.5 * math.pi,
        PAIR_PROBE_ANGLE_COUNT,
        dtype=np.float64,
    )
    weights = np.stack([-np.cos(angles), np.sin(angles)], axis=1)
    target = np.asarray(
        [-math.sqrt(0.5), math.sqrt(0.5)], dtype=np.float64
    )
    scores = np.asarray(values, dtype=np.float64) @ weights.T
    positive = binary_labels == 1
    negative = binary_labels == 0
    best_index = 0
    best_key: Optional[Tuple[float, float, int]] = None
    for index in range(weights.shape[0]):
        auroc = _binary_auroc(scores[positive, index], scores[negative, index])
        distance_squared = float(np.sum((weights[index] - target) ** 2))
        objective = auroc - PAIR_PROBE_ANGLE_REGULARIZATION * distance_squared
        key = (objective, -distance_squared, -index)
        if best_key is None or key > best_key:
            best_key = key
            best_index = index
    fitted = weights[best_index].astype(np.float32)
    # These assertions are repeated by persisted-bank validation.  Keeping the
    # local guard protects callers that use the helper independently.
    if fitted[0] > 1e-7 or fitted[1] < -1e-7:
        raise RuntimeError("class_analysis_pair_probe_sign_constraint_broken")
    fitted /= max(1e-12, float(np.linalg.norm(fitted)))
    return fitted


def _fit_source_cross_fitted_pair_probe(
    current_features: np.ndarray,
    current_sources: Sequence[str],
    alternative_features: np.ndarray,
    alternative_sources: Sequence[str],
) -> Dict[str, Any]:
    """Cross-fit and fully refit one directed intrinsic two-feature probe."""

    negative = np.asarray(current_features, dtype=np.float32)
    positive = np.asarray(alternative_features, dtype=np.float32)
    fallback_weights = np.asarray(
        [-math.sqrt(0.5), math.sqrt(0.5)], dtype=np.float32
    )
    result: Dict[str, Any] = {
        "weights": fallback_weights,
        "threshold": 0.0,
        "oof_auroc": 0.0,
        "oof_positive_pass_fraction": 0.0,
        "fold_count": 0,
        "fit_status": "insufficient_sources",
        "fold_digest": "",
    }
    if (
        negative.ndim != 2
        or positive.ndim != 2
        or negative.shape[1:] != (2,)
        or positive.shape[1:] != (2,)
        or negative.shape[0] != len(current_sources)
        or positive.shape[0] != len(alternative_sources)
        or negative.shape[0] < MIN_HELDOUT_SOURCE_GROUPS
        or positive.shape[0] < MIN_HELDOUT_SOURCE_GROUPS
    ):
        return result
    if not np.all(np.isfinite(negative)) or not np.all(np.isfinite(positive)):
        result["fit_status"] = "nonfinite"
        return result

    features = np.concatenate([negative, positive], axis=0)
    labels = np.concatenate(
        [
            np.zeros(negative.shape[0], dtype=np.int8),
            np.ones(positive.shape[0], dtype=np.int8),
        ]
    )
    sources = [str(value) for value in current_sources] + [
        str(value) for value in alternative_sources
    ]
    assignments, fold_count, fold_digest = _stable_source_group_folds(
        sources,
        labels,
    )
    if fold_count < 2:
        result["fit_status"] = "fold_invalid"
        return result

    oof_scores = np.full(features.shape[0], np.nan, dtype=np.float32)
    for fold in range(fold_count):
        train = assignments != fold
        heldout = assignments == fold
        weights = _select_sign_constrained_pair_weights(
            features[train], labels[train]
        )
        if weights is None or not np.any(heldout):
            result["fit_status"] = "fold_invalid"
            return result
        oof_scores[heldout] = features[heldout] @ weights
    if not np.all(np.isfinite(oof_scores)):
        result["fit_status"] = "nonfinite"
        return result

    oof_threshold, oof_auroc, _unused = _balanced_source_operating_point(
        oof_scores[labels == 1],
        oof_scores[labels == 0],
        fallback=0.0,
    )
    oof_positive_pass_fraction = float(
        np.mean(oof_scores[labels == 1] >= float(oof_threshold))
    )
    fitted_weights = _select_sign_constrained_pair_weights(features, labels)
    if fitted_weights is None:
        result["fit_status"] = "fold_invalid"
        return result
    result.update(
        {
            "weights": fitted_weights.astype(np.float32, copy=False),
            # Deploy the source-cross-fitted operating point.  Re-fitting the
            # safe unit direction on every bag is useful, but re-selecting its
            # threshold on those same bags would reintroduce resubstitution
            # optimism into the production decision gate.
            "threshold": float(oof_threshold),
            "oof_auroc": float(oof_auroc),
            "oof_positive_pass_fraction": oof_positive_pass_fraction,
            "fold_count": int(fold_count),
            "fit_status": "ok",
            "fold_digest": fold_digest,
        }
    )
    return result


def infer_square_grid(token_count: int) -> Tuple[int, int]:
    """Return a square spatial grid or reject an ambiguous token contract."""

    count = int(token_count)
    side = int(round(math.sqrt(max(0, count))))
    if count <= 0 or side * side != count:
        raise ValueError("dinov3_patch_grid_not_square")
    return side, side


def dinov3_spatial_token_offset(num_register_tokens: int = 0) -> int:
    """Return the first spatial-token index for a CLS/register sequence."""

    return 1 + max(0, int(num_register_tokens or 0))


def strip_torch_dinov3_special_tokens(
    last_hidden_state: Any,
    *,
    num_register_tokens: int = 0,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Extract DINO spatial tokens from a Torch-like last-hidden-state value.

    DINOv3 sequences contain a CLS token followed by optional learned register
    tokens.  Treating registers as image patches corrupts spatial pooling and
    heatmaps, so the start offset is always ``1 + num_register_tokens``.
    """

    value = last_hidden_state
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] <= 0:
        raise ValueError("dinov3_last_hidden_state_shape_invalid")
    register_count = max(0, int(num_register_tokens or 0))
    start = dinov3_spatial_token_offset(register_count)
    if array.shape[1] <= start:
        raise ValueError("dinov3_last_hidden_state_missing_patches")
    patches = array[:, start:, :]
    grid = infer_square_grid(patches.shape[1])
    flat = _normalise_rows(patches.reshape(-1, patches.shape[-1]))
    return flat.reshape(patches.shape).astype(np.float32, copy=False), grid


def validate_mlx_patch_tokens(
    patch_tokens: Any,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Validate and L2-normalise the MLX worker's already-clean patch tensor."""

    patches = np.asarray(patch_tokens, dtype=np.float32)
    if patches.ndim != 3 or patches.shape[0] <= 0 or patches.shape[2] <= 0:
        raise ValueError("mlx_dinov3_patch_tokens_shape_invalid")
    grid = infer_square_grid(patches.shape[1])
    flat = _normalise_rows(patches.reshape(-1, patches.shape[-1]))
    return flat.reshape(patches.shape).astype(np.float32, copy=False), grid


@dataclass(frozen=True)
class RefinementConfig:
    schema: str = REFINEMENT_SCHEMA
    input_size: int = 224
    selected_fraction: float = 0.05
    max_candidates: int = 5_000
    anchors_per_class: int = 128
    patches_per_anchor: int = 32
    patch_reservoir_per_class: int = 1_024
    prototypes_per_class: int = 128
    minimum_distinct_sources: int = MIN_RELIABLE_TOTAL_SOURCE_GROUPS
    support_margin: float = 0.08
    strong_support_margin: float = 0.12
    weak_support_margin: float = 0.02
    overlap_localized_fraction: float = 0.80
    outside_overlap_confirm_fraction: float = 0.35
    exclusive_support_margin: float = 0.02
    minimum_component_cells: int = 2
    minimum_component_mass_fraction: float = 0.20
    component_correspondence_distance_fraction: float = 0.25
    component_separation_distance_fraction: float = 0.15
    # Full V3.2 confirmation audit found visually invalid confirmations through
    # 17x16 (272 px^2), while the first consistently legible corrections began
    # at 22x21. Policy therefore requires a 16 px short side plus 324 px^2 of
    # visible area (an 18x18-equivalent area), intentionally allowing elongated
    # objects that retain enough resolved pixels; it does not require 18x18.
    minimum_confirmation_bbox_short_side: float = 16.0
    minimum_confirmation_bbox_area: float = 324.0
    seed: int = 42

    def validate(self) -> None:
        if self.schema != REFINEMENT_SCHEMA:
            raise ValueError("class_analysis_refinement_schema_unsupported")
        if (
            isinstance(self.input_size, bool)
            or not isinstance(self.input_size, (int, np.integer))
            or int(self.input_size) != 224
        ):
            raise ValueError("class_analysis_refinement_input_size_unsupported")
        if isinstance(self.selected_fraction, bool):
            raise ValueError(
                "class_analysis_refinement_selected_fraction_invalid"
            )
        try:
            selected_fraction = float(self.selected_fraction)
        except (TypeError, ValueError):
            raise ValueError(
                "class_analysis_refinement_selected_fraction_invalid"
            ) from None
        if not math.isfinite(selected_fraction) or not 0.0 < selected_fraction <= 1.0:
            raise ValueError("class_analysis_refinement_selected_fraction_invalid")

        positive_integer_fields = {
            "candidate_cap": self.max_candidates,
            "anchor_count": self.anchors_per_class,
            "patches_per_anchor": self.patches_per_anchor,
            "patch_reservoir": self.patch_reservoir_per_class,
            "prototype_count": self.prototypes_per_class,
            "minimum_distinct_sources": self.minimum_distinct_sources,
            "minimum_component_cells": self.minimum_component_cells,
        }
        for detail, value in positive_integer_fields.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) <= 0
            ):
                raise ValueError(
                    f"class_analysis_refinement_{detail}_invalid"
                )
        margin_fields = {
            "support_margin": self.support_margin,
            "strong_support_margin": self.strong_support_margin,
            "weak_support_margin": self.weak_support_margin,
            "exclusive_support_margin": self.exclusive_support_margin,
        }
        margins: Dict[str, float] = {}
        for detail, value in margin_fields.items():
            if isinstance(value, bool):
                raise ValueError(
                    f"class_analysis_refinement_{detail}_invalid"
                )
            try:
                margin = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"class_analysis_refinement_{detail}_invalid"
                ) from None
            if not math.isfinite(margin) or not -2.0 <= margin <= 2.0:
                raise ValueError(
                    f"class_analysis_refinement_{detail}_invalid"
                )
            margins[detail] = margin
        if not (
            margins["weak_support_margin"]
            <= margins["support_margin"]
            <= margins["strong_support_margin"]
        ):
            raise ValueError(
                "class_analysis_refinement_support_margin_order_invalid"
            )

        for detail, value in (
            ("overlap_localized_fraction", self.overlap_localized_fraction),
            (
                "outside_overlap_confirm_fraction",
                self.outside_overlap_confirm_fraction,
            ),
            (
                "minimum_component_mass_fraction",
                self.minimum_component_mass_fraction,
            ),
            (
                "component_correspondence_distance_fraction",
                self.component_correspondence_distance_fraction,
            ),
            (
                "component_separation_distance_fraction",
                self.component_separation_distance_fraction,
            ),
        ):
            if isinstance(value, bool):
                raise ValueError(
                    f"class_analysis_refinement_{detail}_invalid"
                )
            try:
                fraction = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"class_analysis_refinement_{detail}_invalid"
                ) from None
            if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    f"class_analysis_refinement_{detail}_invalid"
                )

        for detail, value in (
            (
                "minimum_confirmation_bbox_short_side",
                self.minimum_confirmation_bbox_short_side,
            ),
            (
                "minimum_confirmation_bbox_area",
                self.minimum_confirmation_bbox_area,
            ),
        ):
            if isinstance(value, bool):
                raise ValueError(
                    f"class_analysis_refinement_{detail}_invalid"
                )
            try:
                threshold = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"class_analysis_refinement_{detail}_invalid"
                ) from None
            if not math.isfinite(threshold) or threshold < 0.0:
                raise ValueError(
                    f"class_analysis_refinement_{detail}_invalid"
                )

        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, (int, np.integer))
            or not 0 <= int(self.seed) <= np.iinfo(np.uint32).max
        ):
            raise ValueError("class_analysis_refinement_seed_invalid")


def select_within_class_outlier_candidates(
    points: Sequence[Mapping[str, Any]],
    *,
    fraction: float = 0.05,
    cap: int = 5_000,
) -> List[Dict[str, Any]]:
    """Select a stable top tail of existing Stage-1 within-class scores."""

    eligible = [
        point
        for point in points
        if isinstance(point, Mapping) and str(point.get("point_id") or "").strip()
    ]
    if len(eligible) < 20:
        return []
    count = max(1, int(math.ceil(len(eligible) * max(0.0, min(1.0, float(fraction))))))
    count = min(count, max(1, int(cap)))
    ordered = sorted(
        eligible,
        key=lambda point: (
            -float(point.get("outlier_score") or 0.0),
            str(point.get("point_id") or ""),
        ),
    )
    selected: List[Dict[str, Any]] = []
    for point in ordered[:count]:
        selected.append(
            {
                "point_id": str(point.get("point_id") or ""),
                "review_object_key": point.get("review_object_key"),
                "class_name": str(point.get("class_name") or ""),
                "suggested_neighbor_class": "",
                "outlier_score": float(point.get("outlier_score") or 0.0),
                "wrong_class_suspicion": float(
                    point.get("wrong_class_suspicion")
                    or point.get("outlier_score")
                    or 0.0
                ),
                "image_relpath": str(point.get("image_relpath") or ""),
                "split": str(point.get("split") or "train"),
                "rough_candidate_reason": "within_class_top_5_percent",
            }
        )
    return selected


def bbox_overlap_geometry(
    target_bbox: Sequence[float],
    other_bbox: Sequence[float],
) -> Optional[Dict[str, Any]]:
    try:
        ax1, ay1, ax2, ay2 = [float(value) for value in list(target_bbox)[:4]]
        bx1, by1, bx2, by2 = [float(value) for value in list(other_bbox)[:4]]
    except Exception:
        return None
    if not all(
        math.isfinite(value)
        for value in (ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)
    ):
        return None
    aw, ah = max(0.0, ax2 - ax1), max(0.0, ay2 - ay1)
    bw, bh = max(0.0, bx2 - bx1), max(0.0, by2 - by1)
    if min(aw, ah, bw, bh) <= 0.0:
        return None
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return None
    target_area, other_area = aw * ah, bw * bh
    epsilon = float(np.finfo(np.float32).eps)
    iou = intersection / max(
        epsilon,
        target_area + other_area - intersection,
    )
    target_coverage = intersection / max(epsilon, target_area)
    other_coverage = intersection / max(epsilon, other_area)
    if iou >= 0.85 or (target_coverage >= 0.85 and other_coverage >= 0.85):
        relation = "duplicate_like"
    elif target_coverage >= 0.75:
        relation = "other_contains_target"
    elif other_coverage >= 0.75:
        relation = "target_contains_other"
    else:
        relation = "partial_contamination"
    return {
        "iou": float(iou),
        "target_area_covered": float(target_coverage),
        "other_area_covered": float(other_coverage),
        "relation": relation,
        "intersection_xyxy": [ix1, iy1, ix2, iy2],
    }


_CONFIRMATION_REQUIRED_GATES = (
    "directed_pair_reliable",
    "directed_pair_candidate_source_independent",
    "directed_pair_exact_calibration_contracts",
    "intrinsic_references_reliable",
    "positive_confirmation_pair_reliable",
    "positive_confirmation_pair_probe_auroc_sufficient",
    "positive_confirmation_pair_probe_lower_bound_sufficient",
    "source_resolution_sufficient",
    "current_absent",
    "alternative_strong",
    "directed_pair_dominates",
    "alternative_exclusive_component_corresponds",
    "view_consistent",
    "alternative_evidence_external_to_overlap",
)


def confirmation_invariants_hold(evidence: Mapping[str, Any]) -> bool:
    """Reject a terminal confirmation missing any V4 proof obligation."""

    if str(evidence.get("status") or "") != STATUS_CONFIRMED_OUTLIER:
        return True
    gates = evidence.get("decision_gates")
    if not isinstance(gates, Mapping) or not all(
        gates.get(name) is True for name in _CONFIRMATION_REQUIRED_GATES
    ):
        return False
    fold_digest = evidence.get("directed_pair_probe_fold_digest")
    split_digest = evidence.get("directed_pair_probe_fit_eval_split_digest")
    finite_metrics = (
        evidence.get("directed_pair_heldout_auroc"),
        evidence.get("directed_pair_eval_auroc_lower_bound"),
        evidence.get("directed_pair_probe_fit_balanced_accuracy"),
        evidence.get("directed_pair_probe_eval_sensitivity"),
        evidence.get("directed_pair_probe_eval_specificity"),
        evidence.get("directed_pair_current_absence_eval_fraction"),
        evidence.get("directed_pair_alternative_strong_eval_fraction"),
    )
    integral_provenance = (
        evidence.get("directed_pair_probe_fold_count"),
        evidence.get("directed_pair_probe_fit_current_source_count"),
        evidence.get("directed_pair_probe_fit_alternative_source_count"),
        evidence.get("directed_pair_probe_eval_current_source_count"),
        evidence.get("directed_pair_probe_eval_alternative_source_count"),
    )
    if not all(_is_non_bool_integral(value) for value in integral_provenance):
        return False
    try:
        metrics = tuple(float(value) for value in finite_metrics)
        features = np.asarray(
            evidence.get("directed_pair_probe_features"), dtype=np.float64
        )
        weights = np.asarray(
            evidence.get("directed_pair_probe_weights"), dtype=np.float64
        )
    except (TypeError, ValueError, OverflowError):
        return False
    scalar_keys = (
        "current_support_score",
        "alternative_support_score",
        "intrinsic_current_support",
        "intrinsic_alternative_support",
        "directed_pair_margin",
        "directed_pair_raw_margin",
        "directed_pair_probe_score",
        "directed_pair_probe_threshold",
        "directed_pair_threshold",
        "directed_pair_current_exclusive_support",
        "directed_pair_alternative_exclusive_support",
        "current_support_threshold",
        "current_negative_threshold",
        "current_strong_threshold",
        "alternative_negative_threshold",
        "alternative_support_threshold",
        "alternative_strong_threshold",
        "visible_target_bbox_width",
        "visible_target_bbox_height",
        "visible_target_bbox_area",
        "minimum_confirmation_bbox_short_side",
        "minimum_confirmation_bbox_area",
        "positive_confirmation_pair_probe_auroc_floor",
        "positive_confirmation_pair_probe_auroc_lower_bound_floor",
    )
    if not all(_is_finite_real(evidence.get(key)) for key in scalar_keys):
        return False
    try:
        current_view_supports = np.asarray(
            evidence.get("current_view_support_scores"), dtype=np.float64
        )
        alternative_view_supports = np.asarray(
            evidence.get("alternative_view_support_scores"), dtype=np.float64
        )
        feature_names = list(
            evidence.get("directed_pair_probe_feature_names") or []
        )
    except (TypeError, ValueError, OverflowError):
        return False
    if (
        features.shape != (2,)
        or weights.shape != (2,)
        or current_view_supports.shape != (2,)
        or alternative_view_supports.shape != (2,)
        or not np.all(np.isfinite(features))
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(current_view_supports))
        or not np.all(np.isfinite(alternative_view_supports))
    ):
        return False
    scalar = {key: float(evidence[key]) for key in scalar_keys}
    probe_score = float(np.dot(weights, features))
    numeric_proof_holds = bool(
        feature_names == list(PAIR_PROBE_FEATURE_NAMES)
        and np.all(np.abs(features) <= 4.0 + 1e-6)
        and np.all(np.abs(current_view_supports) <= 2.0 + 1e-6)
        and np.all(np.abs(alternative_view_supports) <= 2.0 + 1e-6)
        and weights[0] <= 1e-6
        and weights[1] >= -1e-6
        and math.isclose(
            float(np.linalg.norm(weights)), 1.0, rel_tol=0.0, abs_tol=1e-5
        )
        and abs(scalar["directed_pair_probe_threshold"])
        <= PAIR_PROBE_SCORE_ABS_BOUND
        and math.isclose(
            scalar["directed_pair_probe_score"],
            probe_score,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and probe_score >= scalar["directed_pair_probe_threshold"]
        and math.isclose(
            scalar["directed_pair_threshold"],
            scalar["directed_pair_probe_threshold"],
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        and math.isclose(
            scalar["directed_pair_current_exclusive_support"],
            float(features[0]),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and math.isclose(
            scalar["directed_pair_alternative_exclusive_support"],
            float(features[1]),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and math.isclose(
            scalar["current_support_score"],
            float(np.mean(current_view_supports)),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and math.isclose(
            scalar["alternative_support_score"],
            float(np.mean(alternative_view_supports)),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and math.isclose(
            scalar["intrinsic_current_support"],
            scalar["current_support_score"],
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        and math.isclose(
            scalar["intrinsic_alternative_support"],
            scalar["alternative_support_score"],
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        and math.isclose(
            scalar["directed_pair_margin"],
            scalar["alternative_support_score"]
            - scalar["current_support_score"],
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and math.isclose(
            scalar["directed_pair_raw_margin"],
            scalar["directed_pair_margin"],
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        and max(current_view_supports)
        <= scalar["current_negative_threshold"] + 1e-7
        and scalar["current_negative_threshold"]
        < scalar["current_support_threshold"]
        and scalar["current_support_threshold"]
        <= scalar["current_strong_threshold"]
        and scalar["alternative_negative_threshold"]
        < scalar["alternative_support_threshold"]
        and min(alternative_view_supports)
        >= scalar["alternative_strong_threshold"] - 1e-7
        and scalar["alternative_strong_threshold"]
        >= scalar["alternative_support_threshold"]
        and math.isclose(
            scalar["visible_target_bbox_area"],
            scalar["visible_target_bbox_width"]
            * scalar["visible_target_bbox_height"],
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and min(
            scalar["visible_target_bbox_width"],
            scalar["visible_target_bbox_height"],
        )
        >= scalar["minimum_confirmation_bbox_short_side"]
        and scalar["visible_target_bbox_area"]
        >= scalar["minimum_confirmation_bbox_area"]
        and scalar["positive_confirmation_pair_probe_auroc_floor"]
        == MIN_POSITIVE_CONFIRMATION_PAIR_PROBE_AUROC
        and scalar[
            "positive_confirmation_pair_probe_auroc_lower_bound_floor"
        ]
        == MIN_POSITIVE_CONFIRMATION_PAIR_PROBE_AUROC_LOWER_BOUND
        and evidence.get("directed_pair_reliable") is True
        and evidence.get("directed_pair_bank_reliable") is True
        and evidence.get("directed_pair_candidate_source_excluded") is False
        and evidence.get(
            "directed_pair_candidate_source_membership_roles"
        )
        == []
        and evidence.get("reference_reliable") is True
        and evidence.get("intrinsic_references_reliable") is True
        and evidence.get("support_threshold_source")
        == "fit_only_directed_pair"
        and evidence.get("directed_pair_tier") in {"usable", "high"}
    )
    (
        fold_count,
        fit_current_count,
        fit_alternative_count,
        eval_current_count,
        eval_alternative_count,
    ) = integral_provenance
    return bool(
        evidence.get("schema") == REFINEMENT_SCHEMA
        and evidence.get("decision_contract") == REFINEMENT_DECISION_CONTRACT
        and evidence.get("directed_pair_probe_contract")
        == PAIR_PROBE_CONTRACT
        and evidence.get("directed_pair_probe_view_contract")
        == PAIR_PROBE_VIEW_CONTRACT
        and evidence.get("directed_pair_probe_lower_bound_contract")
        == PAIR_PROBE_LOWER_BOUND_CONTRACT
        and evidence.get("directed_pair_probe_fit_status") == "ok"
        and fold_count == 1
        and _is_sha256_hex_digest(fold_digest)
        and _is_sha256_hex_digest(split_digest)
        and fold_digest == split_digest
        and fit_current_count >= MIN_PAIR_PROBE_FIT_SOURCES_PER_CLASS
        and fit_alternative_count >= MIN_PAIR_PROBE_FIT_SOURCES_PER_CLASS
        and eval_current_count >= MIN_PAIR_PROBE_EVAL_SOURCES_PER_CLASS
        and eval_alternative_count >= MIN_PAIR_PROBE_EVAL_SOURCES_PER_CLASS
        and all(math.isfinite(value) for value in metrics)
        and all(0.0 <= value <= 1.0 for value in metrics)
        and metrics[1] <= metrics[0]
        and metrics[0] >= MIN_POSITIVE_CONFIRMATION_PAIR_PROBE_AUROC
        and metrics[1]
        >= MIN_POSITIVE_CONFIRMATION_PAIR_PROBE_AUROC_LOWER_BOUND
        and all(value >= MIN_PAIR_PROBE_METRIC_FRACTION for value in metrics[2:])
        and pair_metrics_are_reliable(
            current_class_reliable=True,
            alternative_class_reliable=True,
            fit_current_source_count=fit_current_count,
            fit_alternative_source_count=fit_alternative_count,
            eval_current_source_count=eval_current_count,
            eval_alternative_source_count=eval_alternative_count,
            eval_auroc=metrics[0],
            eval_auroc_lower_bound=metrics[1],
            fit_balanced_accuracy=metrics[2],
            eval_sensitivity=metrics[3],
            eval_specificity=metrics[4],
            current_absence_eval_fraction=metrics[5],
            alternative_strong_eval_fraction=metrics[6],
        )
        and numeric_proof_holds
    )


def _overlap_prior_source_key(record: Mapping[str, Any]) -> str:
    """Return the dataset source group used by the overlap prior.

    A content hash groups copied images across paths/splits.  Portable datasets
    without a hash retain the same split/relative-path fallback used by the
    reference-bank source contract.
    """

    digest = str(record.get("_image_sha256") or "").strip().lower()
    if digest:
        return f"sha256:{digest}"
    split = str(record.get("split") or "train").strip() or "train"
    relpath = str(record.get("image_relpath") or "").strip()
    return f"path:{split}/{relpath}" if relpath else ""


def _overlap_prior_material_geometry(
    geometry: Optional[Mapping[str, Any]],
) -> bool:
    """Return whether another box materially contaminates the target box."""

    if not isinstance(geometry, Mapping):
        return False
    if str(geometry.get("relation") or "") == "duplicate_like":
        # Duplicate-like geometry is learned in its own lower-impact stratum;
        # it must not contaminate the normal co-occurrence prevalence.
        return False
    try:
        target_coverage = float(geometry.get("target_area_covered") or 0.0)
        iou = float(geometry.get("iou") or 0.0)
    except (TypeError, ValueError):
        return False
    return bool(
        math.isfinite(target_coverage)
        and math.isfinite(iou)
        and (
            target_coverage >= FREQUENT_OVERLAP_MIN_TARGET_COVERAGE
            or iou >= FREQUENT_OVERLAP_MIN_IOU
        )
    )


def _overlap_prior_geometry_stratum(
    geometry: Optional[Mapping[str, Any]],
) -> str:
    if not isinstance(geometry, Mapping):
        return ""
    if str(geometry.get("relation") or "") == "duplicate_like":
        return "duplicate_like"
    if _overlap_prior_material_geometry(geometry):
        return "material_nonduplicate"
    return ""


def overlap_annotation_selection_key(
    geometry: Mapping[str, Any],
) -> Tuple[bool, bool, float, float, float, str]:
    """Return the exact stable priority for one annotated overlap object.

    Duplicate-like ownership conflicts remain first. Among ordinary overlaps,
    prefer geometry that meets the learned-prior materiality contract before a
    slightly higher-IoU incidental intersection. This keeps the single public
    annotation identity aligned with the overlap row used by selector triage,
    while patch masking continues to consume every intersection.
    """

    relation = str(geometry.get("relation") or "")
    iou = float(geometry.get("iou") or 0.0)
    target_coverage = float(geometry.get("target_area_covered") or 0.0)
    return (
        relation == "duplicate_like",
        _overlap_prior_material_geometry(geometry),
        max(iou, target_coverage),
        target_coverage,
        iou,
        relation,
    )


def _wilson_lower_bound(successes: int, total: int) -> float:
    """Conservative two-sided 95% Wilson lower bound for a source rate."""

    count = max(0, int(total))
    positive = max(0, min(count, int(successes)))
    if count <= 0:
        return 0.0
    z = 1.959963984540054
    proportion = positive / count
    z2 = z * z
    denominator = 1.0 + z2 / count
    centre = proportion + z2 / (2.0 * count)
    radius = z * math.sqrt(
        (proportion * (1.0 - proportion) + z2 / (4.0 * count))
        / count
    )
    return max(0.0, min(1.0, (centre - radius) / denominator))


def frequent_overlap_wilson_lower_bound(successes: int, total: int) -> float:
    """Public pure derivation shared by persistence validation."""

    return _wilson_lower_bound(successes, total)


def frequent_overlap_cohort_specifications() -> Tuple[Dict[str, Any], ...]:
    """Return the ordered, version-bound prior cohorts and quality thresholds."""

    return (
        {
            "reliability_tier": "strong",
            "minimum_capture_groups": FREQUENT_OVERLAP_MIN_SOURCE_GROUPS,
            "minimum_positive_capture_groups": (
                FREQUENT_OVERLAP_MIN_POSITIVE_SOURCE_GROUPS
            ),
            "minimum_wilson_lower_bound": (
                FREQUENT_OVERLAP_MIN_WILSON_LOWER_BOUND
            ),
            "material_cap": FREQUENT_OVERLAP_MAX_PRIORITY_ADJUSTMENT,
            "duplicate_cap": (
                FREQUENT_DUPLICATE_OVERLAP_MAX_PRIORITY_ADJUSTMENT
            ),
            "source_independence_verified": True,
        },
        {
            "reliability_tier": "lower_confidence",
            "minimum_capture_groups": (
                FREQUENT_OVERLAP_LOWER_CONFIDENCE_MIN_GROUPS
            ),
            "minimum_positive_capture_groups": (
                FREQUENT_OVERLAP_LOWER_CONFIDENCE_MIN_POSITIVE_GROUPS
            ),
            "minimum_wilson_lower_bound": (
                FREQUENT_OVERLAP_LOWER_CONFIDENCE_MIN_WILSON_LOWER_BOUND
            ),
            "material_cap": (
                FREQUENT_OVERLAP_LOWER_CONFIDENCE_MAX_PRIORITY_ADJUSTMENT
            ),
            "duplicate_cap": (
                FREQUENT_DUPLICATE_LOWER_CONFIDENCE_MAX_PRIORITY_ADJUSTMENT
            ),
            "source_independence_verified": False,
        },
        {
            "reliability_tier": "provisional_unlineaged",
            "minimum_capture_groups": FREQUENT_OVERLAP_PROVISIONAL_MIN_GROUPS,
            "minimum_positive_capture_groups": (
                FREQUENT_OVERLAP_PROVISIONAL_MIN_POSITIVE_GROUPS
            ),
            "minimum_wilson_lower_bound": (
                FREQUENT_OVERLAP_PROVISIONAL_MIN_WILSON_LOWER_BOUND
            ),
            "material_cap": (
                FREQUENT_OVERLAP_PROVISIONAL_MAX_PRIORITY_ADJUSTMENT
            ),
            "duplicate_cap": (
                FREQUENT_DUPLICATE_PROVISIONAL_MAX_PRIORITY_ADJUSTMENT
            ),
            "source_independence_verified": False,
        },
    )


def frequent_overlap_triage_cohort_specifications() -> Tuple[Dict[str, Any], ...]:
    """Return the independent, rank-only frequent-overlap triage gates.

    These gates intentionally do not inherit semantic ``adjustment_eligible``.
    They can therefore identify a well-supported dataset tendency for an
    unresolved row while leaving the row unresolved and preserving its human
    review requirement.
    """

    return (
        {
            "reliability_tier": "strong",
            "minimum_capture_groups": (
                FREQUENT_OVERLAP_TRIAGE_MIN_SOURCE_GROUPS
            ),
            "minimum_positive_capture_groups": (
                FREQUENT_OVERLAP_TRIAGE_MIN_POSITIVE_SOURCE_GROUPS
            ),
            "minimum_wilson_lower_bound": (
                FREQUENT_OVERLAP_TRIAGE_MIN_WILSON_LOWER_BOUND
            ),
            "maximum_priority_adjustment": (
                FREQUENT_OVERLAP_TRIAGE_MAX_PRIORITY_ADJUSTMENT
            ),
            "source_independence_verified": True,
        },
        {
            "reliability_tier": "lower_confidence",
            "minimum_capture_groups": (
                FREQUENT_OVERLAP_TRIAGE_LOWER_CONFIDENCE_MIN_GROUPS
            ),
            "minimum_positive_capture_groups": (
                FREQUENT_OVERLAP_TRIAGE_LOWER_CONFIDENCE_MIN_POSITIVE_GROUPS
            ),
            "minimum_wilson_lower_bound": (
                FREQUENT_OVERLAP_TRIAGE_LOWER_CONFIDENCE_MIN_WILSON_LOWER_BOUND
            ),
            "maximum_priority_adjustment": (
                FREQUENT_OVERLAP_TRIAGE_LOWER_CONFIDENCE_MAX_PRIORITY_ADJUSTMENT
            ),
            "source_independence_verified": False,
        },
        {
            "reliability_tier": "provisional_unlineaged",
            "minimum_capture_groups": (
                FREQUENT_OVERLAP_TRIAGE_PROVISIONAL_MIN_GROUPS
            ),
            "minimum_positive_capture_groups": (
                FREQUENT_OVERLAP_TRIAGE_PROVISIONAL_MIN_POSITIVE_GROUPS
            ),
            "minimum_wilson_lower_bound": (
                FREQUENT_OVERLAP_TRIAGE_PROVISIONAL_MIN_WILSON_LOWER_BOUND
            ),
            "maximum_priority_adjustment": (
                FREQUENT_OVERLAP_TRIAGE_PROVISIONAL_MAX_PRIORITY_ADJUSTMENT
            ),
            "source_independence_verified": False,
        },
    )


def frequent_overlap_evidence_multiplier(
    candidate_refinement_evidence: Optional[Mapping[str, Any]],
) -> Tuple[float, str]:
    """Derive whether overlap prevalence may lower this candidate's rank."""

    refinement = (
        candidate_refinement_evidence
        if isinstance(candidate_refinement_evidence, Mapping)
        else {}
    )
    gates = (
        refinement.get("decision_gates")
        if isinstance(refinement.get("decision_gates"), Mapping)
        else {}
    )
    current_absent = gates.get("current_absent") is True
    external_alternative = (
        gates.get("alternative_evidence_external_to_overlap") is True
        or refinement.get("alternative_evidence_external_to_overlap") is True
    )
    current_present = (
        gates.get("current_present") is True
        or gates.get("current_strong") is True
    )
    overlap_explains_alternative = bool(
        gates.get("alternative_evidence_localized_to_overlap") is True
        or gates.get("nested_overlap") is True
        or gates.get("proved_overlap_decomposition") is True
        or gates.get("current_overlap_explanation") is True
        or refinement.get("alternative_evidence_localized_to_overlap") is True
        or refinement.get("nested_overlap") is True
        or refinement.get("proved_overlap_decomposition") is True
        or refinement.get("current_overlap_explanation") is True
    )
    known_patch_evidence = any(
        isinstance(refinement.get(name), (int, float, np.number))
        and not isinstance(refinement.get(name), bool)
        and math.isfinite(float(refinement.get(name)))
        for name in (
            "current_support_score",
            "alternative_support_score",
            "intrinsic_current_support",
            "intrinsic_alternative_support",
        )
    )
    if current_absent:
        return 0.0, "current_class_absent_no_overlap_decrease"
    if external_alternative:
        return 0.0, "alternative_external_no_overlap_decrease"
    if str(refinement.get("status") or "").strip() == STATUS_PAIR_CONFLICT:
        return 0.0, "pair_conflict_no_overlap_decrease"
    if current_present and overlap_explains_alternative:
        return 1.0, "current_present_overlap_explains_alternative"
    if not known_patch_evidence:
        return 0.0, "patch_evidence_unavailable_no_overlap_decrease"
    return 0.0, "patch_evidence_inconclusive_no_overlap_decrease"


def frequent_overlap_fit_source_digest(
    *,
    fit_registry_digest: str,
    current_class: str,
    alternative_class: str,
    candidate_capture_group_id: str,
    candidate_capture_group_dependency_tier: str = "unavailable",
    cohort_statistics: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    """Bind a row to the summary-anchored fit registry and its LOO group."""

    digest = hashlib.sha256()
    for value in (
        FREQUENT_OVERLAP_PRIOR_CONTRACT,
        CAPTURE_GROUP_CONTRACT,
        FREQUENT_OVERLAP_FIT_ELIGIBILITY_CONTRACT,
        FREQUENT_OVERLAP_FIT_REGISTRY_CONTRACT,
        str(fit_registry_digest or ""),
        str(current_class or "").strip(),
        str(alternative_class or "").strip(),
        str(candidate_capture_group_id or "").strip(),
        str(candidate_capture_group_dependency_tier or "").strip(),
    ):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    for stratum in cohort_statistics or ():
        for value in (
            str(stratum.get("geometry_stratum") or ""),
        ):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        for diagnostic_field in (
            "cohort_diagnostics",
            "triage_cohort_diagnostics",
        ):
            encoded = diagnostic_field.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            diagnostics = stratum.get(diagnostic_field)
            if not isinstance(diagnostics, Sequence):
                diagnostics = ()
            for diagnostic in diagnostics:
                if not isinstance(diagnostic, Mapping):
                    continue
                values = (
                    str(diagnostic.get("reliability_tier") or ""),
                    int(
                        diagnostic.get("eligible_capture_group_count") or 0
                    ),
                    int(
                        diagnostic.get("overlap_capture_group_count") or 0
                    ),
                    float(diagnostic.get("group_rate_sum") or 0.0).hex(),
                )
                for value in values:
                    encoded = str(value).encode("utf-8")
                    digest.update(len(encoded).to_bytes(8, "big"))
                    digest.update(encoded)
    return digest.hexdigest()


def frequent_overlap_fit_query_key(
    *,
    current_class: str,
    alternative_class: str,
    candidate_capture_group_id: str,
) -> str:
    digest = hashlib.sha256()
    for value in (
        str(current_class or "").strip(),
        str(alternative_class or "").strip(),
        str(candidate_capture_group_id or "").strip(),
    ):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass
class CaptureGroupIndex:
    """Conservative image-to-capture grouping for independence claims.

    Exact hashes and perceptual similarity can prove that images are *not*
    independent; they cannot prove that two visually distinct UUID-named
    frames came from independent captures.  Only explicit or conservatively
    parsed parent/sequence metadata receives the strong tier.  A content-bound
    perceptual signature permits a lower-confidence tier with stricter gates.
    Missing lineage remains visible. Content-hash-only groups may contribute
    only to a separately labelled provisional heuristic with much stricter
    gates and a small cap; they never claim source-independent reliability.
    """

    source_to_group: Dict[str, str]
    group_tiers: Dict[str, str]
    group_methods: Dict[str, Tuple[str, ...]]
    group_image_counts: Dict[str, int]
    image_count: int
    source_tiers: Dict[str, str] = field(default_factory=dict)
    source_methods: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    perceptual_near_duplicate_merge_count: int = 0

    def summary(self) -> Dict[str, Any]:
        tier_counts: Dict[str, int] = {}
        method_counts: Dict[str, int] = {}
        for group, tier in self.group_tiers.items():
            tier_counts[tier] = int(tier_counts.get(tier, 0)) + 1
            for method in self.group_methods.get(group, ()):
                method_counts[method] = int(method_counts.get(method, 0)) + 1
        return {
            "contract": CAPTURE_GROUP_CONTRACT,
            "image_count_semantics": (
                "unique_split_image_relpath_representatives"
            ),
            "image_count": int(self.image_count),
            "capture_group_count": len(self.group_tiers),
            "capture_group_tier_counts": dict(sorted(tier_counts.items())),
            "capture_group_method_counts": dict(sorted(method_counts.items())),
            "perceptual_near_duplicate_merge_count": int(
                self.perceptual_near_duplicate_merge_count
            ),
            "strong_independence_methods": [
                "explicit_capture_group",
                "exporter_parent",
                "sequence_filename",
            ],
            "conservative_nonindependence_methods": ["directory_parent"],
            "lower_confidence_requires_content_bound_perceptual_hash": True,
            "provisional_unlineaged_is_source_independent": False,
            "unresolved_provenance_counts_toward_adjustment": False,
            "perceptual_hash_bits": CAPTURE_PERCEPTUAL_HASH_BITS,
            "perceptual_max_hamming_distance": (
                CAPTURE_PERCEPTUAL_MAX_HAMMING_DISTANCE
            ),
        }


def _capture_explicit_group_hint(record: Mapping[str, Any]) -> str:
    for field in (
        "capture_group_id",
        "sequence_id",
        "video_id",
        "flight_id",
    ):
        value = str(record.get(field) or "").strip()
        if value:
            # A declared capture/sequence is the independence boundary. Do not
            # split it by camera: synchronized or adjacent multi-camera views
            # are still correlated unless the exporter declares distinct
            # capture groups itself.
            return f"{field}:{value}"
    return ""


_CAPTURE_EXPORTER_RE = re.compile(
    r"^(?P<parent>.+?)_raw_images_"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"_crop\d+$",
    re.IGNORECASE,
)
_CAPTURE_UUID_STEM_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_CAPTURE_SEQUENCE_RE = re.compile(
    r"^(?P<prefix>.*(?:frame|image|img|capture|shot|tile|[_-]))[_-]?"
    r"(?P<index>\d{3,})$",
    re.IGNORECASE,
)


def _capture_parsed_group_hint(record: Mapping[str, Any]) -> Tuple[str, str]:
    relpath = str(record.get("image_relpath") or "").strip().replace("\\", "/")
    if not relpath:
        return "", ""
    path = PurePosixPath(relpath)
    stem = path.stem
    exporter = _CAPTURE_EXPORTER_RE.fullmatch(stem)
    if exporter:
        return (
            f"exporter:{path.parent.as_posix()}:{exporter.group('parent')}",
            "exporter_parent",
        )
    if not _CAPTURE_UUID_STEM_RE.fullmatch(stem):
        sequence = _CAPTURE_SEQUENCE_RE.fullmatch(stem)
        if sequence:
            return (
                f"sequence:{path.parent.as_posix()}:{sequence.group('prefix').lower()}",
                "sequence_filename",
            )
    parent = path.parent.as_posix().strip()
    if parent not in {"", ".", "images", "train", "val", "test"}:
        return f"directory:{parent}", "directory_parent"
    return "", ""


def _capture_bound_perceptual_signature(
    record: Mapping[str, Any],
) -> Optional[int]:
    signature = str(
        record.get("capture_perceptual_hash")
        or record.get("_capture_perceptual_hash")
        or ""
    ).strip().lower()
    attested_sha = str(
        record.get("capture_perceptual_image_sha256")
        or record.get("_capture_perceptual_image_sha256")
        or ""
    ).strip().lower()
    image_sha = str(record.get("_image_sha256") or "").strip().lower()
    if (
        not re.fullmatch(r"[0-9a-f]{32}", signature)
        or not re.fullmatch(r"[0-9a-f]{64}", attested_sha)
        or attested_sha != image_sha
    ):
        return None
    return int(signature, 16)


def _capture_aspect_bucket(record: Mapping[str, Any]) -> int:
    try:
        width = float(record.get("_image_width") or 0.0)
        height = float(record.get("_image_height") or 0.0)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(width) or not math.isfinite(height) or min(width, height) <= 0:
        return 0
    return int(round(math.log2(width / height) * 8.0))


def _build_capture_group_index(
    image_records: Sequence[Mapping[str, Any]],
    *,
    should_cancel: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    memory_check: Optional[Callable[[], Any]] = None,
) -> CaptureGroupIndex:
    ordered = sorted(
        (
            record
            for record in image_records
            if isinstance(record, Mapping)
            and _overlap_prior_source_key(record)
        ),
        key=lambda record: (
            str(record.get("split") or "train"),
            str(record.get("image_relpath") or ""),
            _overlap_prior_source_key(record),
        ),
    )
    count = len(ordered)
    parent = list(range(count))
    rank = [0] * count
    methods_by_index: List[set[str]] = [set() for _ in range(count)]

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> bool:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return False
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1
        return True

    def checkpoint(processed: int) -> None:
        if processed != count and processed % 256:
            return
        if should_cancel is not None and should_cancel():
            raise RuntimeError("cancelled")
        if memory_check is not None:
            memory_check()
        if progress_callback is not None:
            progress_callback("capture_groups", processed, count)

    exact_sources: Dict[str, int] = {}
    strong_hints: Dict[str, int] = {}
    perceptual_signatures: Dict[int, int] = {}
    unique_signature_indices: List[int] = []
    for index, record in enumerate(ordered):
        source = _overlap_prior_source_key(record)
        previous_source = exact_sources.setdefault(source, index)
        if previous_source != index:
            union(index, previous_source)
            methods_by_index[index].add("exact_content")
            methods_by_index[previous_source].add("exact_content")
        explicit = _capture_explicit_group_hint(record)
        if explicit:
            hint = f"explicit:{explicit}"
            method = "explicit_capture_group"
        else:
            hint, method = _capture_parsed_group_hint(record)
        if hint:
            previous_hint = strong_hints.setdefault(hint, index)
            if previous_hint != index:
                union(index, previous_hint)
            methods_by_index[index].add(method)
            methods_by_index[previous_hint].add(method)
        signature = _capture_bound_perceptual_signature(record)
        if signature is not None:
            previous_signature = perceptual_signatures.setdefault(
                signature, index
            )
            if previous_signature != index:
                union(index, previous_signature)
                methods_by_index[index].add("perceptual_exact")
                methods_by_index[previous_signature].add("perceptual_exact")
            else:
                unique_signature_indices.append(index)
            methods_by_index[index].add("content_bound_perceptual_hash")
        else:
            methods_by_index[index].add("unresolved_perceptual_provenance")
        checkpoint(index + 1)

    # Sixteen eight-bit LSH bands guarantee that signatures within twelve bit
    # flips share several candidate buckets, while avoiding a quadratic scan.
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    near_merge_count = 0
    near_comparison_count = 0
    unique_total = len(unique_signature_indices)
    for position, index in enumerate(unique_signature_indices, start=1):
        record = ordered[index]
        signature = _capture_bound_perceptual_signature(record)
        if signature is None:
            continue
        aspect = _capture_aspect_bucket(record)
        candidates: set[int] = set()
        for band in range(16):
            value = (signature >> (band * 8)) & 0xFF
            for aspect_candidate in (aspect - 1, aspect, aspect + 1):
                candidates.update(
                    buckets.get((band, value, aspect_candidate), ())
                )
        for other_index in sorted(candidates):
            near_comparison_count += 1
            if near_comparison_count % 4096 == 0:
                if should_cancel is not None and should_cancel():
                    raise RuntimeError("cancelled")
                if memory_check is not None:
                    memory_check()
            other_signature = _capture_bound_perceptual_signature(
                ordered[other_index]
            )
            if other_signature is None:
                continue
            if (signature ^ other_signature).bit_count() <= (
                CAPTURE_PERCEPTUAL_MAX_HAMMING_DISTANCE
            ):
                if union(index, other_index):
                    near_merge_count += 1
                methods_by_index[index].add("perceptual_near_duplicate")
                methods_by_index[other_index].add(
                    "perceptual_near_duplicate"
                )
        for band in range(16):
            value = (signature >> (band * 8)) & 0xFF
            buckets.setdefault((band, value, aspect), []).append(index)
        if position == unique_total or position % 256 == 0:
            if should_cancel is not None and should_cancel():
                raise RuntimeError("cancelled")
            if memory_check is not None:
                memory_check()
            if progress_callback is not None:
                progress_callback(
                    "capture_perceptual_clusters", position, unique_total
                )

    group_indices: Dict[int, List[int]] = {}
    for index in range(count):
        group_indices.setdefault(find(index), []).append(index)
    source_to_group: Dict[str, str] = {}
    group_tiers: Dict[str, str] = {}
    group_methods: Dict[str, Tuple[str, ...]] = {}
    group_image_counts: Dict[str, int] = {}
    source_tiers: Dict[str, str] = {}
    source_methods: Dict[str, Tuple[str, ...]] = {}
    strong_methods = {
        "explicit_capture_group",
        "exporter_parent",
        "sequence_filename",
    }
    for indices in group_indices.values():
        sources = sorted({_overlap_prior_source_key(ordered[i]) for i in indices})
        methods = sorted(
            {method for i in indices for method in methods_by_index[i]}
        )
        digest = hashlib.sha256()
        digest.update(CAPTURE_GROUP_CONTRACT.encode("utf-8"))
        for source in sources:
            encoded = source.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        group = f"capture:{digest.hexdigest()}"
        perceptual_grouped = bool(
            {"perceptual_exact", "perceptual_near_duplicate"}.intersection(
                methods
            )
        )
        local_methods_by_source: Dict[str, set[str]] = {}
        for index in indices:
            local_methods_by_source.setdefault(
                _overlap_prior_source_key(ordered[index]), set()
            ).update(methods_by_index[index])
        for source in sources:
            local_methods = tuple(
                sorted(local_methods_by_source.get(source, ()))
            )
            if strong_methods.intersection(local_methods):
                source_tier = "strong"
            elif (
                "content_bound_perceptual_hash" in local_methods
                and perceptual_grouped
            ):
                source_tier = "lower_confidence"
            elif source.startswith("sha256:"):
                source_tier = "provisional_unlineaged"
            else:
                source_tier = "unresolved_provenance"
            source_tiers[source] = source_tier
            source_methods[source] = local_methods
        # A dependency component is only as trustworthy as its weakest member.
        # Perceptual similarity may conservatively merge an unlineaged image
        # into an explicitly lineaged capture so leave-one-out excludes both,
        # but that merge must never promote the unlineaged member into the
        # strong fit cohort.  The broad group remains useful for leakage
        # control while its weakest local tier controls statistical claims.
        tier_order = {
            "strong": 0,
            "lower_confidence": 1,
            "provisional_unlineaged": 2,
            "unresolved_provenance": 3,
        }
        tier = max(
            (source_tiers[source] for source in sources),
            key=lambda candidate: tier_order[candidate],
        )
        for source in sources:
            source_to_group[source] = group
        group_tiers[group] = tier
        group_methods[group] = tuple(methods)
        group_image_counts[group] = len(indices)
    return CaptureGroupIndex(
        source_to_group=source_to_group,
        group_tiers=group_tiers,
        group_methods=group_methods,
        group_image_counts=group_image_counts,
        image_count=count,
        source_tiers=source_tiers,
        source_methods=source_methods,
        perceptual_near_duplicate_merge_count=near_merge_count,
    )


@dataclass
class FrequentOverlapPrior:
    """Capture-balanced observed class-overlap prevalence.

    The prior is descriptive, not ground truth: it learns which *annotated*
    class pairs frequently overlap. Candidate scoring always removes the
    candidate's complete capture group, so adjacent frames or derivative crops
    cannot manufacture their own explanation. Counts are capture-balanced and
    beta-smoothed; rare pairs never adjust priority. Only strong lineage is
    described as source-independent. Perceptual and unlineaged evidence remain
    explicitly lower-confidence/provisional and use stricter gates and caps.
    """

    class_source_object_counts: Dict[str, Dict[str, int]]
    pair_source_overlap_object_counts: Dict[
        Tuple[str, str, str], Dict[str, int]
    ]
    capture_groups: CaptureGroupIndex
    record_count: int = 0
    input_record_count: int = 0
    context_record_count: int = 0
    stage1_screened_point_id_count: int = 0
    stage1_screened_record_count: int = 0
    excluded_unscreened_annotation_record_count: int = 0
    excluded_unusable_provenance_annotation_record_count: int = 0
    fit_screening_scope: str = "all_classes"
    fit_screening_exhaustive: bool = True
    stage1_screened_point_id_digest: str = ""
    excluded_suspicious_record_count: int = 0
    excluded_directed_overlap_observation_count: int = 0
    _fit_cache_ready: bool = field(default=False, init=False, repr=False)
    _class_tier_group_counts: Dict[Tuple[str, str], int] = field(
        default_factory=dict, init=False, repr=False
    )
    _pair_tier_positive_counts: Dict[
        Tuple[str, str, str, str], int
    ] = field(default_factory=dict, init=False, repr=False)
    _pair_tier_group_rate_sums: Dict[
        Tuple[str, str, str, str], float
    ] = field(default_factory=dict, init=False, repr=False)
    _fit_registry_digest_cache: str = field(
        default="", init=False, repr=False
    )
    _fit_query_registry: Dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        # Freeze owned copies before any cache is built. A warmed aggregate
        # must never silently diverge from mutable public fit maps.
        self.class_source_object_counts = MappingProxyType(
            {
                str(class_name): MappingProxyType(
                    {str(group): int(count) for group, count in counts.items()}
                )
                for class_name, counts in self.class_source_object_counts.items()
            }
        )
        self.pair_source_overlap_object_counts = MappingProxyType(
            {
                tuple(key): MappingProxyType(
                    {str(group): int(count) for group, count in counts.items()}
                )
                for key, counts in self.pair_source_overlap_object_counts.items()
            }
        )
        capture = self.capture_groups
        self.capture_groups = CaptureGroupIndex(
            source_to_group=MappingProxyType(dict(capture.source_to_group)),
            group_tiers=MappingProxyType(dict(capture.group_tiers)),
            group_methods=MappingProxyType(
                {
                    str(group): tuple(methods)
                    for group, methods in capture.group_methods.items()
                }
            ),
            group_image_counts=MappingProxyType(
                {
                    str(group): int(count)
                    for group, count in capture.group_image_counts.items()
                }
            ),
            image_count=int(capture.image_count),
            source_tiers=MappingProxyType(dict(capture.source_tiers)),
            source_methods=MappingProxyType(
                {
                    str(source): tuple(methods)
                    for source, methods in capture.source_methods.items()
                }
            ),
            perceptual_near_duplicate_merge_count=int(
                capture.perceptual_near_duplicate_merge_count
            ),
        )
        count_fields = (
            self.record_count,
            self.input_record_count,
            self.context_record_count,
            self.stage1_screened_point_id_count,
            self.stage1_screened_record_count,
            self.excluded_unscreened_annotation_record_count,
            self.excluded_unusable_provenance_annotation_record_count,
            self.excluded_suspicious_record_count,
            self.excluded_directed_overlap_observation_count,
        )
        mapped_record_count = sum(
            count
            for class_counts in self.class_source_object_counts.values()
            for count in class_counts.values()
        )
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in count_fields
            )
            or mapped_record_count != self.record_count
            or self.stage1_screened_record_count
            != self.record_count
            + self.excluded_suspicious_record_count
            + self.excluded_unusable_provenance_annotation_record_count
            or self.context_record_count
            != self.stage1_screened_record_count
            + self.excluded_unscreened_annotation_record_count
            or self.context_record_count > self.input_record_count
            or self.stage1_screened_record_count
            > self.stage1_screened_point_id_count
        ):
            raise ValueError("frequent_overlap_prior_fit_counts_invalid")
        self.fit_screening_scope = str(
            self.fit_screening_scope or ""
        ).strip().lower()
        if self.fit_screening_scope not in {"selected_class", "all_classes"}:
            raise ValueError("frequent_overlap_prior_screening_scope_invalid")
        if type(self.fit_screening_exhaustive) is not bool:
            raise ValueError(
                "frequent_overlap_prior_screening_exhaustive_invalid"
            )
        normalized_screened_digest = str(
            self.stage1_screened_point_id_digest or ""
        ).strip().lower()
        if not re.fullmatch(
            r"[0-9a-f]{64}",
            normalized_screened_digest,
        ):
            if self.stage1_screened_point_id_count:
                raise ValueError(
                    "frequent_overlap_prior_screened_point_id_digest_invalid"
                )
            digest = hashlib.sha256()
            digest.update(
                FREQUENT_OVERLAP_FIT_ELIGIBILITY_CONTRACT.encode("utf-8")
            )
            self.stage1_screened_point_id_digest = digest.hexdigest()
        else:
            self.stage1_screened_point_id_digest = normalized_screened_digest

    def _ensure_fit_cache(self) -> None:
        if self._fit_cache_ready:
            return
        class_tier_group_counts: Dict[Tuple[str, str], int] = {}
        pair_tier_positive_counts: Dict[
            Tuple[str, str, str, str], int
        ] = {}
        pair_tier_group_rate_sums: Dict[
            Tuple[str, str, str, str], float
        ] = {}
        pair_tier_group_rates: Dict[
            Tuple[str, str, str, str], List[float]
        ] = {}
        digest = hashlib.sha256()

        def digest_field(value: Any) -> None:
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)

        for value in (
            FREQUENT_OVERLAP_PRIOR_CONTRACT,
            CAPTURE_GROUP_CONTRACT,
            FREQUENT_OVERLAP_FIT_ELIGIBILITY_CONTRACT,
            FREQUENT_OVERLAP_FIT_REGISTRY_CONTRACT,
            FREQUENT_OVERLAP_TRIAGE_CONTRACT,
            FREQUENT_OVERLAP_BETA_ALPHA,
            FREQUENT_OVERLAP_BETA_BETA,
            FREQUENT_OVERLAP_MIN_TARGET_COVERAGE,
            FREQUENT_OVERLAP_MIN_IOU,
            str(self.fit_screening_scope or ""),
            bool(self.fit_screening_exhaustive),
            int(self.stage1_screened_point_id_count),
            str(self.stage1_screened_point_id_digest or ""),
        ):
            digest_field(value)
        for spec in frequent_overlap_cohort_specifications():
            for key in (
                "reliability_tier",
                "minimum_capture_groups",
                "minimum_positive_capture_groups",
                "minimum_wilson_lower_bound",
                "material_cap",
                "duplicate_cap",
                "source_independence_verified",
            ):
                digest_field(key)
                digest_field(spec[key])
        for spec in frequent_overlap_triage_cohort_specifications():
            for key in (
                "reliability_tier",
                "minimum_capture_groups",
                "minimum_positive_capture_groups",
                "minimum_wilson_lower_bound",
                "maximum_priority_adjustment",
                "source_independence_verified",
            ):
                digest_field("triage_gate")
                digest_field(key)
                digest_field(spec[key])
        for group in sorted(self.capture_groups.group_tiers):
            digest_field("capture_group")
            digest_field(group)
            digest_field(self.capture_groups.group_tiers[group])
            for method in self.capture_groups.group_methods.get(group, ()):
                digest_field(method)
        for source in sorted(self.capture_groups.source_to_group):
            digest_field("capture_source")
            digest_field(source)
            digest_field(self.capture_groups.source_to_group[source])
            digest_field(
                self.capture_groups.source_tiers.get(
                    source, "unresolved_provenance"
                )
            )
            for method in self.capture_groups.source_methods.get(source, ()):
                digest_field(method)
        for class_name in sorted(self.class_source_object_counts):
            group_counts = self.class_source_object_counts[class_name]
            for group, object_count in sorted(group_counts.items()):
                tier = self.capture_groups.group_tiers.get(
                    group, "unresolved_provenance"
                )
                key = (class_name, tier)
                class_tier_group_counts[key] = (
                    int(class_tier_group_counts.get(key, 0)) + 1
                )
                digest_field("class_group")
                digest_field(class_name)
                digest_field(group)
                digest_field(tier)
                digest_field(int(object_count))
        for pair_key in sorted(self.pair_source_overlap_object_counts):
            current, alternative, stratum = pair_key
            class_counts = self.class_source_object_counts.get(current, {})
            for group, overlap_count in sorted(
                self.pair_source_overlap_object_counts[pair_key].items()
            ):
                object_count = max(0, int(class_counts.get(group, 0)))
                overlap_count = max(0, int(overlap_count))
                if overlap_count > 0 and object_count <= 0:
                    raise ValueError(
                        "frequent_overlap_prior_invalid_pair_group_without_"
                        "current_class_objects"
                    )
                tier = self.capture_groups.group_tiers.get(
                    group, "unresolved_provenance"
                )
                aggregate_key = (current, alternative, stratum, tier)
                if overlap_count > 0 and object_count > 0:
                    pair_tier_positive_counts[aggregate_key] = (
                        int(pair_tier_positive_counts.get(aggregate_key, 0))
                        + 1
                    )
                    pair_tier_group_rates.setdefault(
                        aggregate_key, []
                    ).append(min(1.0, overlap_count / object_count))
                digest_field("pair_group")
                digest_field(current)
                digest_field(alternative)
                digest_field(stratum)
                digest_field(group)
                digest_field(tier)
                digest_field(overlap_count)
        pair_tier_group_rate_sums = {
            key: math.fsum(values)
            for key, values in pair_tier_group_rates.items()
        }
        self._class_tier_group_counts = class_tier_group_counts
        self._pair_tier_positive_counts = pair_tier_positive_counts
        self._pair_tier_group_rate_sums = pair_tier_group_rate_sums
        self._fit_registry_digest_cache = digest.hexdigest()
        self._fit_cache_ready = True

    def fit_registry_digest(self) -> str:
        self._ensure_fit_cache()
        return self._fit_registry_digest_cache

    def _cohort_statistics(
        self,
        *,
        current_class: str,
        alternative_class: str,
        geometry_stratum: str,
        reliability_tier: str,
        candidate_capture_group_id: str,
    ) -> Dict[str, Any]:
        """Return O(1) leave-one-capture-out sufficient statistics."""

        self._ensure_fit_cache()
        aggregate_key = (
            current_class,
            alternative_class,
            geometry_stratum,
            reliability_tier,
        )
        group_count = int(
            self._class_tier_group_counts.get(
                (current_class, reliability_tier), 0
            )
        )
        positive_group_count = int(
            self._pair_tier_positive_counts.get(aggregate_key, 0)
        )
        group_rate_sum = float(
            self._pair_tier_group_rate_sums.get(aggregate_key, 0.0)
        )
        class_counts = self.class_source_object_counts.get(current_class, {})
        candidate_object_count = max(
            0, int(class_counts.get(candidate_capture_group_id, 0))
        )
        if (
            candidate_capture_group_id
            and candidate_object_count > 0
            and self.capture_groups.group_tiers.get(candidate_capture_group_id)
            == reliability_tier
        ):
            group_count = max(0, group_count - 1)
            pair_count = max(
                0,
                int(
                    self.pair_source_overlap_object_counts.get(
                        (
                            current_class,
                            alternative_class,
                            geometry_stratum,
                        ),
                        {},
                    ).get(candidate_capture_group_id, 0)
                ),
            )
            if pair_count > 0:
                positive_group_count = max(0, positive_group_count - 1)
                group_rate_sum = max(
                    0.0,
                    group_rate_sum
                    - min(1.0, pair_count / candidate_object_count),
                )
        alpha = FREQUENT_OVERLAP_BETA_ALPHA
        beta = FREQUENT_OVERLAP_BETA_BETA
        denominator = group_count + alpha + beta
        return {
            "group_count": group_count,
            "positive_group_count": positive_group_count,
            "group_rate_sum": group_rate_sum,
            "smoothed_incidence": (
                (positive_group_count + alpha) / denominator
                if denominator > 0.0
                else 0.0
            ),
            "smoothed_object_rate": (
                (group_rate_sum + alpha) / denominator
                if denominator > 0.0
                else 0.0
            ),
            "lower_bound": _wilson_lower_bound(
                positive_group_count, group_count
            ),
        }

    def summary(self) -> Dict[str, Any]:
        self._ensure_fit_cache()
        screening_adjustment_eligible = bool(
            self.fit_screening_scope == "all_classes"
            and self.fit_screening_exhaustive
        )
        screening_reason = (
            "exhaustive_all_classes"
            if screening_adjustment_eligible
            else "screening_scope_ineligible"
        )
        all_sources = {
            source
            for counts in self.class_source_object_counts.values()
            for source in counts
        }
        return {
            "contract": FREQUENT_OVERLAP_PRIOR_CONTRACT,
            "capture_group_contract": CAPTURE_GROUP_CONTRACT,
            "fit_eligibility_contract": (
                FREQUENT_OVERLAP_FIT_ELIGIBILITY_CONTRACT
            ),
            "fit_registry_contract": FREQUENT_OVERLAP_FIT_REGISTRY_CONTRACT,
            "fit_registry_digest": self._fit_registry_digest_cache,
            "fit_query_registry": dict(sorted(self._fit_query_registry.items())),
            "source_group_semantics": "capture_group_id",
            "record_count": int(self.record_count),
            "record_count_semantics": (
                "eligible_deduplicated_annotation_records"
            ),
            "eligible_annotation_record_count": int(self.record_count),
            "fit_candidate_record_count": int(
                self.record_count + self.excluded_suspicious_record_count
            ),
            "input_annotation_record_count": int(self.input_record_count),
            "context_annotation_record_count": int(
                self.context_record_count
            ),
            "stage1_screened_point_id_count": int(
                self.stage1_screened_point_id_count
            ),
            "stage1_screened_record_count": int(
                self.stage1_screened_record_count
            ),
            "excluded_unscreened_annotation_record_count": int(
                self.excluded_unscreened_annotation_record_count
            ),
            "excluded_unusable_provenance_annotation_record_count": int(
                self.excluded_unusable_provenance_annotation_record_count
            ),
            "fit_screening_scope": str(self.fit_screening_scope or ""),
            "fit_screening_exhaustive": bool(
                self.fit_screening_exhaustive
            ),
            "fit_screening_adjustment_eligible": (
                screening_adjustment_eligible
            ),
            "fit_screening_quality_gate": {
                "passed": screening_adjustment_eligible,
                "reason": screening_reason,
                "ordering_adjustments_enabled": (
                    screening_adjustment_eligible
                ),
            },
            "stage1_screened_point_id_digest": str(
                self.stage1_screened_point_id_digest or ""
            ),
            "excluded_suspicious_annotation_record_count": int(
                self.excluded_suspicious_record_count
            ),
            "excluded_directed_overlap_observation_count": int(
                self.excluded_directed_overlap_observation_count
            ),
            "fit_requires_both_annotation_roles_trusted": True,
            "fit_requires_both_annotation_roles_stage1_screened": True,
            "source_group_count": len(all_sources),
            "capture_group_count": len(all_sources),
            "class_count": len(self.class_source_object_counts),
            "observed_directed_pair_count": len(
                self.pair_source_overlap_object_counts
            ),
            "minimum_source_groups": FREQUENT_OVERLAP_MIN_SOURCE_GROUPS,
            "minimum_positive_source_groups": (
                FREQUENT_OVERLAP_MIN_POSITIVE_SOURCE_GROUPS
            ),
            "minimum_wilson_lower_bound": (
                FREQUENT_OVERLAP_MIN_WILSON_LOWER_BOUND
            ),
            "beta_prior": {
                "alpha": FREQUENT_OVERLAP_BETA_ALPHA,
                "beta": FREQUENT_OVERLAP_BETA_BETA,
            },
            "material_overlap_thresholds": {
                "minimum_target_coverage": (
                    FREQUENT_OVERLAP_MIN_TARGET_COVERAGE
                ),
                "minimum_iou": FREQUENT_OVERLAP_MIN_IOU,
                "geometry_strata": [
                    "material_nonduplicate",
                    "duplicate_like",
                ],
            },
            "maximum_priority_adjustment": (
                FREQUENT_OVERLAP_MAX_PRIORITY_ADJUSTMENT
            ),
            "maximum_duplicate_like_priority_adjustment": (
                FREQUENT_DUPLICATE_OVERLAP_MAX_PRIORITY_ADJUSTMENT
            ),
            "lower_confidence_gate": {
                "minimum_capture_groups": (
                    FREQUENT_OVERLAP_LOWER_CONFIDENCE_MIN_GROUPS
                ),
                "minimum_positive_capture_groups": (
                    FREQUENT_OVERLAP_LOWER_CONFIDENCE_MIN_POSITIVE_GROUPS
                ),
                "minimum_wilson_lower_bound": (
                    FREQUENT_OVERLAP_LOWER_CONFIDENCE_MIN_WILSON_LOWER_BOUND
                ),
                "maximum_priority_adjustment": (
                    FREQUENT_OVERLAP_LOWER_CONFIDENCE_MAX_PRIORITY_ADJUSTMENT
                ),
                "maximum_duplicate_like_priority_adjustment": (
                    FREQUENT_DUPLICATE_LOWER_CONFIDENCE_MAX_PRIORITY_ADJUSTMENT
                ),
                "source_independence_verified": False,
            },
            "provisional_unlineaged_gate": {
                "minimum_capture_groups": FREQUENT_OVERLAP_PROVISIONAL_MIN_GROUPS,
                "minimum_positive_capture_groups": (
                    FREQUENT_OVERLAP_PROVISIONAL_MIN_POSITIVE_GROUPS
                ),
                "minimum_wilson_lower_bound": (
                    FREQUENT_OVERLAP_PROVISIONAL_MIN_WILSON_LOWER_BOUND
                ),
                "maximum_priority_adjustment": (
                    FREQUENT_OVERLAP_PROVISIONAL_MAX_PRIORITY_ADJUSTMENT
                ),
                "maximum_duplicate_like_priority_adjustment": (
                    FREQUENT_DUPLICATE_PROVISIONAL_MAX_PRIORITY_ADJUSTMENT
                ),
                "source_independence_verified": False,
            },
            "candidate_source_leave_one_out": True,
            "candidate_capture_group_leave_one_out": True,
            "triage_contract": FREQUENT_OVERLAP_TRIAGE_CONTRACT,
            "triage_cohort_specifications": [
                dict(spec)
                for spec in frequent_overlap_triage_cohort_specifications()
            ],
            "triage_review_statuses": [
                STATUS_UNRESOLVED,
                STATUS_MIXED_OR_COMPOSITE,
            ],
            "triage_requires_annotated_alternative_overlap": True,
            "triage_is_rank_only": True,
            "triage_changes_correctness_claim": False,
            "capture_groups": self.capture_groups.summary(),
            "changes_candidate_membership": False,
            "changes_semantic_status": False,
        }

    def candidate_evidence(
        self,
        *,
        current_class: str,
        alternative_class: str,
        query_source_key: str,
        overlap_matches: Sequence[Mapping[str, Any]],
        candidate_refinement_evidence: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        screening_adjustment_eligible = bool(
            self.fit_screening_scope == "all_classes"
            and self.fit_screening_exhaustive
        )
        screening_adjustment_reason = (
            "exhaustive_all_classes"
            if screening_adjustment_eligible
            else "screening_scope_ineligible"
        )
        current = str(current_class or "").strip()
        alternative = str(alternative_class or "").strip()
        source = str(query_source_key or "").strip()
        candidate_capture_group = self.capture_groups.source_to_group.get(
            source, ""
        )
        candidate_capture_tier = self.capture_groups.source_tiers.get(
            source,
            "unavailable" if not candidate_capture_group else "unresolved_provenance",
        )
        candidate_capture_dependency_tier = self.capture_groups.group_tiers.get(
            candidate_capture_group,
            "unavailable" if not candidate_capture_group else "unresolved_provenance",
        )
        candidate_capture_methods = list(
            self.capture_groups.source_methods.get(source, ())
        )
        class_sources = self.class_source_object_counts.get(current, {})
        source_excluded = bool(
            candidate_capture_group
            and candidate_capture_group in class_sources
        )
        pair_matches = [
            match
            for match in overlap_matches
            if isinstance(match, Mapping)
            and str(match.get("class_name") or "").strip() == alternative
        ]
        matches_by_stratum: Dict[str, List[Mapping[str, Any]]] = {
            "material_nonduplicate": [],
            "duplicate_like": [],
        }
        candidate_overlap_evidence: List[Dict[str, Any]] = []
        for match in pair_matches:
            stratum = _overlap_prior_geometry_stratum(match)
            if stratum:
                matches_by_stratum[stratum].append(match)
                try:
                    iou = float(match.get("iou") or 0.0)
                    target_coverage = float(
                        match.get("target_area_covered") or 0.0
                    )
                except (TypeError, ValueError):
                    iou = 0.0
                    target_coverage = 0.0
                candidate_overlap_evidence.append(
                    {
                        "point_id": str(match.get("point_id") or ""),
                        "class_name": alternative,
                        "relation": str(match.get("relation") or ""),
                        "geometry_stratum": stratum,
                        "iou": max(0.0, min(1.0, iou)),
                        "target_area_covered": max(
                            0.0, min(1.0, target_coverage)
                        ),
                    }
                )
        candidate_overlap_evidence.sort(
            key=lambda item: (
                item["geometry_stratum"],
                -float(item["target_area_covered"]),
                -float(item["iou"]),
                item["point_id"],
            )
        )

        evidence_multiplier, evidence_reason = (
            frequent_overlap_evidence_multiplier(
                candidate_refinement_evidence
            )
        )
        refinement = (
            candidate_refinement_evidence
            if isinstance(candidate_refinement_evidence, Mapping)
            else {}
        )
        refinement_gates = (
            refinement.get("decision_gates")
            if isinstance(refinement.get("decision_gates"), Mapping)
            else {}
        )
        refinement_status = str(refinement.get("status") or "").strip()
        refinement_qualified = (
            refinement.get("qualified_for_human_review") is True
        )
        triage_review_statuses = {
            STATUS_UNRESOLVED,
            STATUS_MIXED_OR_COMPOSITE,
        }
        triage_blocked_current_absent = (
            refinement_gates.get("current_absent") is True
        )
        triage_blocked_external_alternative = bool(
            refinement_gates.get(
                "alternative_evidence_external_to_overlap"
            )
            is True
            or refinement.get("alternative_evidence_external_to_overlap")
            is True
        )

        def stratum_statistics(stratum: str) -> Dict[str, Any]:
            cohort_specs = frequent_overlap_cohort_specifications()
            cohort_diagnostics: List[Dict[str, Any]] = []
            cohort_statistics_by_tier: Dict[str, Dict[str, Any]] = {}
            selected_cohort: Optional[Dict[str, Any]] = None
            selected_spec: Optional[Mapping[str, Any]] = None
            fallback_cohort: Optional[Dict[str, Any]] = None
            for spec in cohort_specs:
                tier = str(spec["reliability_tier"])
                minimum_groups = int(spec["minimum_capture_groups"])
                minimum_positive = int(
                    spec["minimum_positive_capture_groups"]
                )
                minimum_lower_bound = float(
                    spec["minimum_wilson_lower_bound"]
                )
                source_independent = bool(
                    spec["source_independence_verified"]
                )
                stats = self._cohort_statistics(
                    current_class=current,
                    alternative_class=alternative,
                    geometry_stratum=stratum,
                    reliability_tier=tier,
                    candidate_capture_group_id=candidate_capture_group,
                )
                cohort_statistics_by_tier[tier] = stats
                passes = bool(
                    int(stats["group_count"]) >= int(minimum_groups)
                    and int(stats["positive_group_count"])
                    >= int(minimum_positive)
                    and float(stats["lower_bound"])
                    >= float(minimum_lower_bound)
                )
                cohort_diagnostics.append(
                    {
                        "reliability_tier": tier,
                        "eligible_capture_group_count": int(
                            stats["group_count"]
                        ),
                        "overlap_capture_group_count": int(
                            stats["positive_group_count"]
                        ),
                        "source_incidence_wilson_lower_bound": float(
                            stats["lower_bound"]
                        ),
                        "group_rate_sum": float(stats["group_rate_sum"]),
                        "smoothed_capture_group_incidence": float(
                            stats["smoothed_incidence"]
                        ),
                        "smoothed_capture_group_balanced_object_rate": float(
                            stats["smoothed_object_rate"]
                        ),
                        "minimum_capture_groups": int(minimum_groups),
                        "minimum_positive_capture_groups": int(
                            minimum_positive
                        ),
                        "minimum_wilson_lower_bound": float(
                            minimum_lower_bound
                        ),
                        "source_independence_verified": bool(
                            source_independent
                        ),
                        "passes": passes,
                    }
                )
                if fallback_cohort is None and int(stats["group_count"]) > 0:
                    fallback_cohort = stats
                if passes and selected_cohort is None:
                    selected_cohort = stats
                    selected_spec = spec

            if selected_cohort is None:
                # Keep the strongest cohort visible for audit, but do not let
                # any cross-tier pool manufacture an independence claim or
                # affect ordering unless one versioned tier passes.
                selected_cohort = fallback_cohort or self._cohort_statistics(
                    current_class=current,
                    alternative_class=alternative,
                    geometry_stratum=stratum,
                    reliability_tier="strong",
                    candidate_capture_group_id=candidate_capture_group,
                )
                reliability_tier = "none"
                adjustment_eligible = False
                reliable = False
                provisional = False
                source_independence_verified = False
                material_cap = 0.0
                duplicate_cap = 0.0
            else:
                assert selected_spec is not None
                reliability_tier = str(selected_spec["reliability_tier"])
                material_cap = float(selected_spec["material_cap"])
                duplicate_cap = float(selected_spec["duplicate_cap"])
                source_independence_verified = bool(
                    selected_spec["source_independence_verified"]
                )
                adjustment_eligible = screening_adjustment_eligible
                provisional = reliability_tier == "provisional_unlineaged"
                # Lower-confidence perceptual grouping has enough evidence to
                # be operationally reliable, but is never called independent.
                reliable = not provisional

            triage_cohort_diagnostics: List[Dict[str, Any]] = []
            selected_triage_cohort: Optional[Dict[str, Any]] = None
            selected_triage_spec: Optional[Mapping[str, Any]] = None
            fallback_triage_cohort: Optional[Dict[str, Any]] = None
            for triage_spec in frequent_overlap_triage_cohort_specifications():
                triage_tier = str(triage_spec["reliability_tier"])
                triage_stats = cohort_statistics_by_tier[triage_tier]
                triage_minimum_groups = int(
                    triage_spec["minimum_capture_groups"]
                )
                triage_minimum_positive = int(
                    triage_spec["minimum_positive_capture_groups"]
                )
                triage_minimum_lower_bound = float(
                    triage_spec["minimum_wilson_lower_bound"]
                )
                triage_passes = bool(
                    int(triage_stats["group_count"])
                    >= triage_minimum_groups
                    and int(triage_stats["positive_group_count"])
                    >= triage_minimum_positive
                    and float(triage_stats["lower_bound"])
                    >= triage_minimum_lower_bound
                )
                triage_cohort_diagnostics.append(
                    {
                        "reliability_tier": triage_tier,
                        "eligible_capture_group_count": int(
                            triage_stats["group_count"]
                        ),
                        "overlap_capture_group_count": int(
                            triage_stats["positive_group_count"]
                        ),
                        "source_incidence_wilson_lower_bound": float(
                            triage_stats["lower_bound"]
                        ),
                        "group_rate_sum": float(
                            triage_stats["group_rate_sum"]
                        ),
                        "smoothed_capture_group_incidence": float(
                            triage_stats["smoothed_incidence"]
                        ),
                        "smoothed_capture_group_balanced_object_rate": float(
                            triage_stats["smoothed_object_rate"]
                        ),
                        "minimum_capture_groups": triage_minimum_groups,
                        "minimum_positive_capture_groups": (
                            triage_minimum_positive
                        ),
                        "minimum_wilson_lower_bound": (
                            triage_minimum_lower_bound
                        ),
                        "maximum_priority_adjustment": float(
                            triage_spec["maximum_priority_adjustment"]
                        ),
                        "source_independence_verified": bool(
                            triage_spec["source_independence_verified"]
                        ),
                        "passes": triage_passes,
                    }
                )
                if (
                    fallback_triage_cohort is None
                    and int(triage_stats["group_count"]) > 0
                ):
                    fallback_triage_cohort = triage_stats
                if triage_passes and selected_triage_cohort is None:
                    selected_triage_cohort = triage_stats
                    selected_triage_spec = triage_spec

            if selected_triage_cohort is None:
                selected_triage_cohort = (
                    fallback_triage_cohort
                    or cohort_statistics_by_tier["strong"]
                )
                triage_reliability_tier = "none"
                triage_adjustment_eligible = False
                triage_source_independence_verified = False
                triage_provisional = False
                maximum_triage_adjustment = 0.0
            else:
                assert selected_triage_spec is not None
                triage_reliability_tier = str(
                    selected_triage_spec["reliability_tier"]
                )
                triage_adjustment_eligible = screening_adjustment_eligible
                triage_source_independence_verified = bool(
                    selected_triage_spec["source_independence_verified"]
                )
                triage_provisional = (
                    triage_reliability_tier == "provisional_unlineaged"
                )
                maximum_triage_adjustment = float(
                    selected_triage_spec["maximum_priority_adjustment"]
                )
            source_count = int(selected_cohort["group_count"])
            positive_source_count = int(
                selected_cohort["positive_group_count"]
            )
            smoothed_incidence = float(
                selected_cohort["smoothed_incidence"]
            )
            smoothed_object_rate = float(
                selected_cohort["smoothed_object_rate"]
            )
            lower_bound = float(selected_cohort["lower_bound"])
            matches = matches_by_stratum[stratum]
            overlap_strength = max(
                (
                    max(
                        float(match.get("target_area_covered") or 0.0),
                        float(match.get("iou") or 0.0),
                    )
                    for match in matches
                ),
                default=0.0,
            )
            overlap_strength = max(0.0, min(1.0, overlap_strength))
            prior_strength = min(
                smoothed_incidence,
                smoothed_object_rate,
                lower_bound,
            )
            maximum_adjustment = (
                duplicate_cap if stratum == "duplicate_like" else material_cap
            )
            applies = bool(
                adjustment_eligible
                and matches
                and evidence_multiplier > 0.0
            )
            semantic_adjustment = (
                maximum_adjustment
                * overlap_strength
                * prior_strength
                * evidence_multiplier
                if applies
                else 0.0
            )
            triage_source_count = int(
                selected_triage_cohort["group_count"]
            )
            triage_positive_source_count = int(
                selected_triage_cohort["positive_group_count"]
            )
            triage_smoothed_incidence = float(
                selected_triage_cohort["smoothed_incidence"]
            )
            triage_smoothed_object_rate = float(
                selected_triage_cohort["smoothed_object_rate"]
            )
            triage_lower_bound = float(
                selected_triage_cohort["lower_bound"]
            )
            triage_prior_strength = min(
                triage_smoothed_incidence,
                triage_smoothed_object_rate,
                triage_lower_bound,
            )
            if not screening_adjustment_eligible:
                triage_reason = "screening_scope_ineligible"
                triage_applies = False
            elif refinement_qualified:
                triage_reason = (
                    "qualified_human_review_no_triage_decrease"
                )
                triage_applies = False
            elif refinement_status == STATUS_PAIR_CONFLICT:
                triage_reason = "pair_conflict_no_triage_decrease"
                triage_applies = False
            elif refinement_status not in triage_review_statuses:
                triage_reason = (
                    "not_review_unresolved_no_triage_decrease"
                )
                triage_applies = False
            elif triage_blocked_current_absent:
                triage_reason = "current_class_absent_no_triage_decrease"
                triage_applies = False
            elif triage_blocked_external_alternative:
                triage_reason = "alternative_external_no_triage_decrease"
                triage_applies = False
            elif applies:
                triage_reason = (
                    "semantic_overlap_adjustment_already_applies"
                )
                triage_applies = False
            elif not matches:
                triage_reason = (
                    "no_annotated_alternative_overlap_no_triage_decrease"
                )
                triage_applies = False
            elif not triage_adjustment_eligible:
                triage_reason = "triage_frequency_gate_not_met"
                triage_applies = False
            else:
                triage_applies = maximum_triage_adjustment > 0.0
                triage_reason = "frequent_overlap_review_triage_rank_only"
            triage_adjustment = (
                maximum_triage_adjustment
                * overlap_strength
                * triage_prior_strength
                if triage_applies
                else 0.0
            )
            total_adjustment = semantic_adjustment + triage_adjustment
            return {
                "geometry_stratum": stratum,
                "fit_screening_adjustment_eligible": (
                    screening_adjustment_eligible
                ),
                "fit_screening_adjustment_reason": (
                    screening_adjustment_reason
                ),
                "capture_group_contract": CAPTURE_GROUP_CONTRACT,
                "fit_eligibility_contract": (
                    FREQUENT_OVERLAP_FIT_ELIGIBILITY_CONTRACT
                ),
                "reliability_tier": reliability_tier,
                "source_independence_verified": (
                    source_independence_verified
                ),
                "provisional": provisional,
                "adjustment_eligible": adjustment_eligible,
                "eligible_capture_group_count": source_count,
                "overlap_capture_group_count": positive_source_count,
                "eligible_source_count": source_count,
                "overlap_source_count": positive_source_count,
                "raw_source_incidence": (
                    positive_source_count / source_count
                    if source_count
                    else 0.0
                ),
                "raw_capture_group_incidence": (
                    positive_source_count / source_count
                    if source_count
                    else 0.0
                ),
                "smoothed_source_incidence": smoothed_incidence,
                "smoothed_capture_group_incidence": smoothed_incidence,
                "smoothed_source_balanced_object_rate": (
                    smoothed_object_rate
                ),
                "smoothed_capture_group_balanced_object_rate": (
                    smoothed_object_rate
                ),
                "group_rate_sum": float(selected_cohort["group_rate_sum"]),
                "source_incidence_wilson_lower_bound": lower_bound,
                "capture_group_incidence_wilson_lower_bound": lower_bound,
                "reliable": reliable,
                "cohort_diagnostics": cohort_diagnostics,
                "candidate_overlap": bool(matches),
                "candidate_overlap_count": len(matches),
                "candidate_overlap_strength": overlap_strength,
                "conservative_prior_strength": prior_strength,
                "maximum_priority_adjustment": maximum_adjustment,
                "evidence_multiplier": evidence_multiplier,
                "evidence_multiplier_reason": evidence_reason,
                "applies": applies,
                "semantic_reliability_tier": reliability_tier,
                "semantic_adjustment_eligible": adjustment_eligible,
                "semantic_priority_adjustment": float(
                    semantic_adjustment
                ),
                "triage_contract": FREQUENT_OVERLAP_TRIAGE_CONTRACT,
                "triage_reliability_tier": triage_reliability_tier,
                "triage_source_independence_verified": (
                    triage_source_independence_verified
                ),
                "triage_provisional": triage_provisional,
                "triage_adjustment_eligible": triage_adjustment_eligible,
                "triage_eligible_capture_group_count": triage_source_count,
                "triage_overlap_capture_group_count": (
                    triage_positive_source_count
                ),
                "triage_group_rate_sum": float(
                    selected_triage_cohort["group_rate_sum"]
                ),
                "triage_smoothed_capture_group_incidence": (
                    triage_smoothed_incidence
                ),
                "triage_smoothed_capture_group_balanced_object_rate": (
                    triage_smoothed_object_rate
                ),
                "triage_source_incidence_wilson_lower_bound": (
                    triage_lower_bound
                ),
                "triage_conservative_prior_strength": (
                    triage_prior_strength
                ),
                "triage_cohort_diagnostics": triage_cohort_diagnostics,
                "triage_candidate_annotated_overlap": bool(matches),
                "maximum_triage_frequency_adjustment": (
                    maximum_triage_adjustment
                ),
                "triage_frequency_adjustment_reason": triage_reason,
                "triage_applies": triage_applies,
                "triage_frequency_adjustment": float(
                    triage_adjustment
                ),
                "priority_adjustment": float(total_adjustment),
            }

        strata = [
            stratum_statistics("material_nonduplicate"),
            stratum_statistics("duplicate_like"),
        ]
        selected = max(
            strata,
            key=lambda item: (
                float(item.get("priority_adjustment") or 0.0),
                int(bool(item.get("candidate_overlap"))),
                int(bool(item.get("triage_adjustment_eligible"))),
                int(bool(item.get("adjustment_eligible"))),
                item.get("geometry_stratum") == "material_nonduplicate",
            ),
        )
        reasons: List[str] = []
        if source_excluded:
            reasons.append("candidate_capture_group_left_out")
        elif source:
            reasons.append("candidate_capture_group_not_in_prior_population")
        else:
            reasons.append("candidate_capture_group_unavailable")
        selected_stratum = str(selected.get("geometry_stratum") or "")
        if selected.get("candidate_overlap"):
            reasons.append(f"annotated_overlap_observed:{selected_stratum}")
        elif alternative:
            reasons.append("no_material_overlap_with_alternative")
        else:
            reasons.append("alternative_class_unavailable")
        reliability_tier = str(selected.get("reliability_tier") or "none")
        if reliability_tier == "strong":
            reasons.append("frequent_overlap_prior_strong_capture_reliable")
        elif reliability_tier == "lower_confidence":
            reasons.append("frequent_overlap_prior_lower_confidence")
        elif reliability_tier == "provisional_unlineaged":
            reasons.append("frequent_overlap_prior_provisional_unlineaged")
        else:
            reasons.append("frequent_overlap_prior_no_eligible_capture_tier")
        if not screening_adjustment_eligible:
            reasons.append("screening_scope_ineligible")
        reasons.append(evidence_reason)
        if selected.get("applies") is True:
            reasons.append("common_overlap_priority_decrease")
        if selected.get("triage_applies") is True:
            reasons.append("frequent_overlap_triage_priority_decrease")
        if (
            selected.get("applies") is not True
            and selected.get("triage_applies") is not True
        ):
            reasons.append("selector_priority_unchanged_by_overlap_prior")

        fit_registry_digest = self.fit_registry_digest()
        fit_source_digest = frequent_overlap_fit_source_digest(
            fit_registry_digest=fit_registry_digest,
            current_class=current,
            alternative_class=alternative,
            candidate_capture_group_id=candidate_capture_group,
            candidate_capture_group_dependency_tier=(
                candidate_capture_dependency_tier
            ),
            cohort_statistics=strata,
        )
        fit_query_key = frequent_overlap_fit_query_key(
            current_class=current,
            alternative_class=alternative,
            candidate_capture_group_id=candidate_capture_group,
        )
        previous_query_digest = self._fit_query_registry.setdefault(
            fit_query_key, fit_source_digest
        )
        if previous_query_digest != fit_source_digest:
            raise RuntimeError("frequent_overlap_prior_query_digest_mismatch")
        return {
            "contract": FREQUENT_OVERLAP_PRIOR_CONTRACT,
            "capture_group_contract": CAPTURE_GROUP_CONTRACT,
            "fit_eligibility_contract": (
                FREQUENT_OVERLAP_FIT_ELIGIBILITY_CONTRACT
            ),
            "fit_screening_scope": str(self.fit_screening_scope or ""),
            "fit_screening_exhaustive": bool(
                self.fit_screening_exhaustive
            ),
            "fit_screening_adjustment_eligible": (
                screening_adjustment_eligible
            ),
            "fit_screening_adjustment_reason": (
                screening_adjustment_reason
            ),
            "stage1_screened_point_id_count": int(
                self.stage1_screened_point_id_count
            ),
            "stage1_screened_point_id_digest": str(
                self.stage1_screened_point_id_digest or ""
            ),
            "current_class": current,
            "alternative_class": alternative,
            "candidate_capture_group_id": candidate_capture_group,
            "candidate_capture_group_tier": candidate_capture_tier,
            "candidate_capture_group_dependency_tier": (
                candidate_capture_dependency_tier
            ),
            "candidate_capture_group_methods": candidate_capture_methods,
            "candidate_capture_group_excluded": source_excluded,
            "candidate_source_excluded": source_excluded,
            "fit_registry_contract": FREQUENT_OVERLAP_FIT_REGISTRY_CONTRACT,
            "fit_registry_digest": fit_registry_digest,
            "fit_query_key": fit_query_key,
            "fit_source_digest": fit_source_digest,
            **selected,
            "candidate_material_overlap": bool(
                matches_by_stratum["material_nonduplicate"]
            ),
            "candidate_material_overlap_count": len(
                matches_by_stratum["material_nonduplicate"]
            ),
            "candidate_duplicate_like_overlap_count": len(
                matches_by_stratum["duplicate_like"]
            ),
            "candidate_overlap_evidence": candidate_overlap_evidence,
            "strata": strata,
            "reasons": reasons,
        }


def build_frequent_overlap_prior(
    spatial_context_records: Sequence[Mapping[str, Any]],
    *,
    trusted_screened_point_ids: Iterable[str],
    fit_screening_scope: str = "all_classes",
    fit_screening_exhaustive: bool = True,
    excluded_suspicious_point_ids: Optional[Iterable[str]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    memory_check: Optional[Callable[[], Any]] = None,
) -> FrequentOverlapPrior:
    """Learn directed overlap prevalence without object-count domination.

    Capture grouping receives exactly one representative per source image, not
    one copy per annotation. The annotation rows remain referenced (not copied)
    by image, and identity de-duplication is image-local, bounding peak memory
    without changing deterministic output.
    """

    total_records = len(spatial_context_records)
    if isinstance(trusted_screened_point_ids, (str, bytes)):
        raise ValueError(
            "frequent_overlap_prior_screened_point_id_allowlist_invalid"
        )
    screened_point_ids = frozenset(
        str(point_id).strip()
        for point_id in trusted_screened_point_ids
        if str(point_id).strip()
    )
    screening_scope = str(fit_screening_scope or "").strip().lower()
    if screening_scope not in {"selected_class", "all_classes"}:
        raise ValueError("frequent_overlap_prior_screening_scope_invalid")
    screened_point_id_digest_builder = hashlib.sha256()
    screened_point_id_digest_builder.update(
        FREQUENT_OVERLAP_FIT_ELIGIBILITY_CONTRACT.encode("utf-8")
    )
    for point_id in sorted(screened_point_ids):
        encoded = point_id.encode("utf-8")
        screened_point_id_digest_builder.update(
            len(encoded).to_bytes(8, "big")
        )
        screened_point_id_digest_builder.update(encoded)
    screened_point_id_digest = screened_point_id_digest_builder.hexdigest()
    excluded_suspicious_ids = frozenset(
        str(point_id).strip()
        for point_id in (excluded_suspicious_point_ids or ())
        if str(point_id).strip()
    )

    def checkpoint(
        phase: str,
        processed: int,
        total: int,
        *,
        interval: int,
    ) -> None:
        if processed != total and processed % interval:
            return
        if should_cancel is not None and should_cancel():
            raise RuntimeError("cancelled")
        if memory_check is not None:
            memory_check()
        if progress_callback is not None:
            progress_callback(phase, processed, total)

    by_image: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    context_point_ids: set[str] = set()
    screened_context_point_ids: set[str] = set()
    excluded_unscreened_point_ids: set[str] = set()
    for position, record in enumerate(spatial_context_records, start=1):
        if not isinstance(record, Mapping):
            checkpoint(
                "scan_records", position, total_records, interval=2048
            )
            continue
        class_name = str(record.get("class_name") or "").strip()
        relpath = str(record.get("image_relpath") or "").strip()
        split = str(record.get("split") or "train").strip() or "train"
        source = _overlap_prior_source_key(record)
        bbox = list(record.get("bbox_xyxy") or [])
        point_id = str(
            record.get("point_id") or record.get("review_object_key") or ""
        ).strip()
        if (
            class_name
            and relpath
            and source
            and len(bbox) >= 4
            and point_id
        ):
            context_point_ids.add(point_id)
            if point_id in screened_point_ids:
                screened_context_point_ids.add(point_id)
                by_image.setdefault((split, relpath), []).append(record)
            else:
                excluded_unscreened_point_ids.add(point_id)
        # Input order is not an identity. A row without a stable object key
        # cannot participate, but it must still reach cancellation/progress.
        checkpoint("scan_records", position, total_records, interval=2048)

    def representative_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
        explicit = _capture_explicit_group_hint(record)
        parsed, _method = _capture_parsed_group_hint(record)
        perceptual = _capture_bound_perceptual_signature(record)
        return (
            int(bool(explicit)),
            int(bool(parsed)),
            int(perceptual is not None),
            explicit,
            parsed,
            str(record.get("capture_perceptual_hash") or ""),
            str(record.get("point_id") or record.get("review_object_key") or ""),
        )

    image_keys = sorted(by_image)
    image_representatives = [
        max(by_image[key], key=representative_key) for key in image_keys
    ]
    capture_groups = _build_capture_group_index(
        image_representatives,
        should_cancel=should_cancel,
        progress_callback=progress_callback,
        memory_check=memory_check,
    )

    class_source_object_counts: Dict[str, Dict[str, int]] = {}
    pair_source_overlap_object_counts: Dict[
        Tuple[str, str, str], Dict[str, int]
    ] = {}
    record_count = 0
    excluded_suspicious_record_count = 0
    excluded_unusable_provenance_record_count = 0
    excluded_directed_overlap_observation_count = 0
    image_total = len(image_keys)
    geometry_comparison_count = 0
    for image_position, image_key in enumerate(image_keys, start=1):
        image_rows = by_image[image_key]
        representative = max(image_rows, key=representative_key)
        exact_source = _overlap_prior_source_key(representative)
        capture_group = capture_groups.source_to_group.get(exact_source, "")
        capture_tier = capture_groups.group_tiers.get(
            capture_group, "unresolved_provenance"
        )
        # Unresolved path-only provenance remains auditable in the capture
        # summary but cannot contribute to any ordering adjustment.
        if not capture_group or capture_tier == "unresolved_provenance":
            excluded_unusable_provenance_record_count += len(
                {
                    str(
                        record.get("point_id")
                        or record.get("review_object_key")
                        or ""
                    ).strip()
                    for record in image_rows
                    if str(
                        record.get("point_id")
                        or record.get("review_object_key")
                        or ""
                    ).strip()
                }
            )
            checkpoint(
                "overlap_geometry",
                image_position,
                image_total,
                interval=128,
            )
            continue

        def row_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
            point_id = str(
                record.get("point_id")
                or record.get("review_object_key")
                or ""
            ).strip()
            bbox_key: Tuple[Any, ...]
            try:
                bbox_key = tuple(
                    float(value)
                    for value in list(record.get("bbox_xyxy") or [])[:4]
                )
            except (TypeError, ValueError):
                bbox_key = ()
            return (
                point_id,
                str(record.get("class_name") or ""),
                bbox_key,
            )

        unique_rows: List[Mapping[str, Any]] = []
        seen_point_ids: set[str] = set()
        for record in sorted(image_rows, key=row_key):
            point_id = str(
                record.get("point_id")
                or record.get("review_object_key")
                or ""
            ).strip()
            if not point_id or point_id in seen_point_ids:
                continue
            seen_point_ids.add(point_id)
            unique_rows.append(record)
            if point_id in excluded_suspicious_ids:
                excluded_suspicious_record_count += 1
                continue
            record_count += 1
            class_name = str(record.get("class_name") or "").strip()
            if class_name:
                class_counts = class_source_object_counts.setdefault(
                    class_name, {}
                )
                class_counts[capture_group] = int(
                    class_counts.get(capture_group, 0)
                ) + 1

        sortable: List[Tuple[float, float, Mapping[str, Any]]] = []
        for record in unique_rows:
            try:
                x1, _y1, x2, _y2 = [
                    float(value)
                    for value in list(record.get("bbox_xyxy") or [])[:4]
                ]
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (x1, x2)) or x2 <= x1:
                continue
            sortable.append((x1, x2, record))
        sortable.sort(key=lambda item: (item[0], item[1]))
        # Accumulate each directed source object once per pair/stratum.  The
        # set is image-local, keeping memory bounded even when the complete
        # dataset contains hundreds of thousands of objects.
        image_pair_objects: set[Tuple[str, str, str, str]] = set()
        image_excluded_pair_objects: set[
            Tuple[str, str, str, str]
        ] = set()
        for left_index, (_left_x1, left_x2, left) in enumerate(
            sortable
        ):
            left_class = str(left.get("class_name") or "").strip()
            left_id = str(
                left.get("point_id") or left.get("review_object_key") or ""
            ).strip()
            left_trusted = left_id not in excluded_suspicious_ids
            for right_index in range(left_index + 1, len(sortable)):
                geometry_comparison_count += 1
                if geometry_comparison_count % 8192 == 0:
                    if should_cancel is not None and should_cancel():
                        raise RuntimeError("cancelled")
                    if memory_check is not None:
                        memory_check()
                right_x1, _right_x2, right = sortable[right_index]
                if right_x1 >= left_x2:
                    break
                right_class = str(right.get("class_name") or "").strip()
                if not left_class or not right_class or left_class == right_class:
                    continue
                # Geometry is evaluated in both directions because target-area
                # contamination is intentionally asymmetric.
                left_geometry = bbox_overlap_geometry(
                    left.get("bbox_xyxy") or [],
                    right.get("bbox_xyxy") or [],
                )
                if left_geometry is None:
                    continue
                right_geometry = bbox_overlap_geometry(
                    right.get("bbox_xyxy") or [],
                    left.get("bbox_xyxy") or [],
                )
                right_id = str(
                    right.get("point_id")
                    or right.get("review_object_key")
                    or ""
                ).strip()
                right_trusted = right_id not in excluded_suspicious_ids
                fit_pair_objects = (
                    image_pair_objects
                    if left_trusted and right_trusted
                    else image_excluded_pair_objects
                )
                left_stratum = _overlap_prior_geometry_stratum(left_geometry)
                right_stratum = _overlap_prior_geometry_stratum(right_geometry)
                if left_stratum:
                    fit_pair_objects.add(
                        (
                            left_class,
                            right_class,
                            left_stratum,
                            left_id,
                        )
                    )
                if right_stratum:
                    fit_pair_objects.add(
                        (
                            right_class,
                            left_class,
                            right_stratum,
                            right_id,
                        )
                    )
        for (
            current_class,
            alternative_class,
            stratum,
            _point_id,
        ) in image_pair_objects:
            pair_counts = pair_source_overlap_object_counts.setdefault(
                (current_class, alternative_class, stratum), {}
            )
            pair_counts[capture_group] = int(
                pair_counts.get(capture_group, 0)
            ) + 1
        excluded_directed_overlap_observation_count += len(
            image_excluded_pair_objects
        )
        checkpoint(
            "overlap_geometry",
            image_position,
            image_total,
            interval=128,
        )

    return FrequentOverlapPrior(
        class_source_object_counts=class_source_object_counts,
        pair_source_overlap_object_counts=(
            pair_source_overlap_object_counts
        ),
        capture_groups=capture_groups,
        record_count=record_count,
        input_record_count=total_records,
        context_record_count=len(context_point_ids),
        stage1_screened_point_id_count=len(screened_point_ids),
        stage1_screened_record_count=len(screened_context_point_ids),
        excluded_unscreened_annotation_record_count=len(
            excluded_unscreened_point_ids
        ),
        excluded_unusable_provenance_annotation_record_count=(
            excluded_unusable_provenance_record_count
        ),
        fit_screening_scope=screening_scope,
        fit_screening_exhaustive=bool(fit_screening_exhaustive),
        stage1_screened_point_id_digest=screened_point_id_digest,
        excluded_suspicious_record_count=(
            excluded_suspicious_record_count
        ),
        excluded_directed_overlap_observation_count=(
            excluded_directed_overlap_observation_count
        ),
    )


def build_overlap_index(
    query_records: Sequence[Mapping[str, Any]],
    spatial_context_records: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Index all material same-image overlaps, including containment."""

    query_image_keys = {
        (
            str(record.get("split") or "train"),
            str(record.get("image_relpath") or ""),
        )
        for record in query_records
    }
    by_image: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for record in spatial_context_records:
        key = (
            str(record.get("split") or "train"),
            str(record.get("image_relpath") or ""),
        )
        if key not in query_image_keys:
            continue
        by_image.setdefault(key, []).append(record)
    result: Dict[str, List[Dict[str, Any]]] = {}
    for query in query_records:
        point_id = str(query.get("point_id") or "")
        if not point_id:
            continue
        key = (
            str(query.get("split") or "train"),
            str(query.get("image_relpath") or ""),
        )
        matches: List[Dict[str, Any]] = []
        for other in by_image.get(key, []):
            other_id = str(other.get("point_id") or "")
            if other_id and other_id == point_id:
                continue
            geometry = bbox_overlap_geometry(
                query.get("bbox_xyxy") or [],
                other.get("bbox_xyxy") or [],
            )
            if geometry is None:
                continue
            matches.append(
                {
                    "point_id": other_id,
                    "class_name": str(other.get("class_name") or ""),
                    "bbox_xyxy": [
                        float(value)
                        for value in list(other.get("bbox_xyxy") or [])[:4]
                    ],
                    **geometry,
                }
            )
        matches.sort(
            key=lambda row: (
                -float(row.get("target_area_covered") or 0.0),
                -float(row.get("iou") or 0.0),
                str(row.get("point_id") or ""),
            )
        )
        result[point_id] = matches
    return result


def patch_source_centres(
    crop_xyxy: Sequence[float],
    grid_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Map cells to source centres and exact valid-image area fractions.

    A centre-only validity bit erases thin objects when no patch centre lands
    inside their letterboxed content.  The second return value is therefore a
    float mask giving the fraction of each patch cell covered by real image
    pixels.
    """

    x1, y1, x2, y2 = [float(value) for value in list(crop_xyxy)[:4]]
    crop_width = max(1e-6, x2 - x1)
    crop_height = max(1e-6, y2 - y1)
    side = max(crop_width, crop_height)
    # Canonical preprocessing pastes into an integer-sized square canvas with
    # floor offsets. Match that exact transform; a symmetric float half-offset
    # drifts by 0.5 source pixel whenever the dimension difference is odd.
    offset_x = float(math.floor((side - crop_width) / 2.0))
    offset_y = float(math.floor((side - crop_height) / 2.0))
    grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
    centres = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
    valid = np.zeros((grid_h, grid_w), dtype=np.float32)
    cell_width = side / max(1, grid_w)
    cell_height = side / max(1, grid_h)
    content_x1 = offset_x
    content_y1 = offset_y
    content_x2 = offset_x + crop_width
    content_y2 = offset_y + crop_height
    for row in range(grid_h):
        square_y = (row + 0.5) * side / grid_h
        local_y = square_y - offset_y
        cell_y1 = row * cell_height
        cell_y2 = (row + 1) * cell_height
        covered_height = max(
            0.0,
            min(cell_y2, content_y2) - max(cell_y1, content_y1),
        )
        for col in range(grid_w):
            square_x = (col + 0.5) * side / grid_w
            local_x = square_x - offset_x
            centres[row, col] = [x1 + local_x, y1 + local_y]
            cell_x1 = col * cell_width
            cell_x2 = (col + 1) * cell_width
            covered_width = max(
                0.0,
                min(cell_x2, content_x2) - max(cell_x1, content_x1),
            )
            valid[row, col] = float(
                (covered_width * covered_height)
                / max(1e-12, cell_width * cell_height)
            )
    return centres, valid


def rasterize_box_fractions(
    crop_xyxy: Sequence[float],
    grid_shape: Tuple[int, int],
    boxes: Sequence[Sequence[float]],
    *,
    supersample: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize source boxes onto the patch grid using exact area coverage.

    ``supersample`` remains in the signature for compatibility with existing
    callers. Axis-aligned annotation boxes do not need sampling: an analytic
    rectangle-union calculation is cheaper and cannot erase a thin box that
    happens to fall between sample centres.
    """

    x1, y1, x2, y2 = [float(value) for value in list(crop_xyxy)[:4]]
    crop_width = max(1e-6, x2 - x1)
    crop_height = max(1e-6, y2 - y1)
    side = max(crop_width, crop_height)
    offset_x = float(math.floor((side - crop_width) / 2.0))
    offset_y = float(math.floor((side - crop_height) / 2.0))
    grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
    # Validate the legacy argument rather than silently accepting surprising
    # objects. Its numeric value no longer affects exact geometry.
    max(1, int(supersample))
    valid_fractions = np.zeros((grid_h, grid_w), dtype=np.float32)
    box_fractions = np.zeros((grid_h, grid_w), dtype=np.float32)
    clean_boxes: List[Tuple[float, float, float, float]] = []
    for box in boxes:
        try:
            bx1, by1, bx2, by2 = [
                float(value) for value in list(box)[:4]
            ]
        except Exception:
            continue
        if (
            all(math.isfinite(value) for value in (bx1, by1, bx2, by2))
            and bx2 > bx1
            and by2 > by1
        ):
            clean_boxes.append((bx1, by1, bx2, by2))

    def rectangle_union_area(
        rectangles: Sequence[Tuple[float, float, float, float]],
    ) -> float:
        if not rectangles:
            return 0.0
        x_edges = sorted(
            {
                coordinate
                for rectangle in rectangles
                for coordinate in (rectangle[0], rectangle[2])
            }
        )
        area = 0.0
        for edge_index in range(len(x_edges) - 1):
            left = x_edges[edge_index]
            right = x_edges[edge_index + 1]
            if right <= left:
                continue
            intervals = sorted(
                (top, bottom)
                for rect_left, top, rect_right, bottom in rectangles
                if rect_left < right and rect_right > left and bottom > top
            )
            if not intervals:
                continue
            covered_y = 0.0
            current_top, current_bottom = intervals[0]
            for top, bottom in intervals[1:]:
                if top <= current_bottom:
                    current_bottom = max(current_bottom, bottom)
                    continue
                covered_y += current_bottom - current_top
                current_top, current_bottom = top, bottom
            covered_y += current_bottom - current_top
            area += (right - left) * covered_y
        return float(area)

    cell_width = side / max(1, grid_w)
    cell_height = side / max(1, grid_h)
    cell_area = max(1e-12, cell_width * cell_height)
    content_x1 = offset_x
    content_y1 = offset_y
    content_x2 = offset_x + crop_width
    content_y2 = offset_y + crop_height
    for row in range(grid_h):
        cell_y1 = row * cell_height
        cell_y2 = (row + 1) * cell_height
        valid_square_y1 = max(cell_y1, content_y1)
        valid_square_y2 = min(cell_y2, content_y2)
        for col in range(grid_w):
            cell_x1 = col * cell_width
            cell_x2 = (col + 1) * cell_width
            valid_square_x1 = max(cell_x1, content_x1)
            valid_square_x2 = min(cell_x2, content_x2)
            if (
                valid_square_x2 <= valid_square_x1
                or valid_square_y2 <= valid_square_y1
            ):
                continue
            source_cell = (
                x1 + valid_square_x1 - offset_x,
                y1 + valid_square_y1 - offset_y,
                x1 + valid_square_x2 - offset_x,
                y1 + valid_square_y2 - offset_y,
            )
            valid_area = (
                (source_cell[2] - source_cell[0])
                * (source_cell[3] - source_cell[1])
            )
            valid_fractions[row, col] = float(valid_area / cell_area)
            intersections: List[Tuple[float, float, float, float]] = []
            for bx1, by1, bx2, by2 in clean_boxes:
                ix1 = max(source_cell[0], bx1)
                iy1 = max(source_cell[1], by1)
                ix2 = min(source_cell[2], bx2)
                iy2 = min(source_cell[3], by2)
                if ix2 > ix1 and iy2 > iy1:
                    intersections.append((ix1, iy1, ix2, iy2))
            box_fractions[row, col] = float(
                rectangle_union_area(intersections) / cell_area
            )
    return (
        np.clip(box_fractions, 0.0, 1.0),
        np.clip(valid_fractions, 0.0, 1.0),
    )


def rasterize_overlap_centres(
    crop_xyxy: Sequence[float],
    grid_shape: Tuple[int, int],
    overlap_boxes: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper returning exact fractional overlap/valid masks."""

    return rasterize_box_fractions(
        crop_xyxy,
        grid_shape,
        overlap_boxes,
        supersample=4,
    )


@dataclass
class ReferenceBank:
    class_names: List[str]
    prototypes: np.ndarray
    prototype_counts: np.ndarray
    prototype_source_ids: np.ndarray
    background_prototypes: np.ndarray
    background_prototype_counts: np.ndarray
    background_prototype_source_ids: np.ndarray
    anchor_counts: np.ndarray
    distinct_source_counts: np.ndarray
    reliable: np.ndarray
    reliability_tiers: np.ndarray
    heldout_aurocs: np.ndarray
    support_thresholds: np.ndarray
    strong_support_thresholds: np.ndarray
    projection_mean: np.ndarray
    projection_components: np.ndarray
    calibration_status: str
    calibration_split_digest: str
    calibration_heldout_source_count: int
    calibration_fit_source_count: int
    calibration_target_patch_counts: np.ndarray
    calibration_background_patch_counts: np.ndarray
    calibration_target_source_counts: np.ndarray
    calibration_background_source_counts: np.ndarray
    calibration_target_passing_source_counts: np.ndarray
    calibration_target_source_pass_fractions: np.ndarray
    fit_target_patch_counts: np.ndarray
    fit_background_patch_counts: np.ndarray
    fit_target_source_counts: np.ndarray
    fit_background_source_counts: np.ndarray
    calibration_heldout_source_ids: np.ndarray
    calibration_fit_source_ids: np.ndarray
    schema: str = REFINEMENT_SCHEMA
    negative_support_thresholds: Optional[np.ndarray] = None
    pair_reliable: Optional[np.ndarray] = None
    pair_reliability_tiers: Optional[np.ndarray] = None
    pair_heldout_aurocs: Optional[np.ndarray] = None
    pair_dominance_thresholds: Optional[np.ndarray] = None
    pair_current_negative_thresholds: Optional[np.ndarray] = None
    pair_current_presence_thresholds: Optional[np.ndarray] = None
    pair_current_strong_thresholds: Optional[np.ndarray] = None
    pair_alternative_negative_thresholds: Optional[np.ndarray] = None
    pair_alternative_presence_thresholds: Optional[np.ndarray] = None
    pair_alternative_strong_thresholds: Optional[np.ndarray] = None
    pair_current_source_counts: Optional[np.ndarray] = None
    pair_alternative_source_counts: Optional[np.ndarray] = None
    pair_current_patch_counts: Optional[np.ndarray] = None
    pair_alternative_patch_counts: Optional[np.ndarray] = None
    pair_alternative_passing_source_fractions: Optional[np.ndarray] = None
    pair_probe_contract: str = PAIR_PROBE_CONTRACT
    pair_probe_view_contract: str = PAIR_PROBE_VIEW_CONTRACT
    pair_probe_lower_bound_contract: str = PAIR_PROBE_LOWER_BOUND_CONTRACT
    pair_probe_weights: Optional[np.ndarray] = None
    pair_probe_thresholds: Optional[np.ndarray] = None
    pair_probe_oof_aurocs: Optional[np.ndarray] = None
    pair_probe_fold_counts: Optional[np.ndarray] = None
    pair_probe_fit_statuses: Optional[np.ndarray] = None
    pair_probe_fold_digests: Optional[np.ndarray] = None
    pair_probe_eval_auroc_lower_bounds: Optional[np.ndarray] = None
    pair_probe_fit_current_source_counts: Optional[np.ndarray] = None
    pair_probe_fit_alternative_source_counts: Optional[np.ndarray] = None
    pair_probe_eval_current_source_counts: Optional[np.ndarray] = None
    pair_probe_eval_alternative_source_counts: Optional[np.ndarray] = None
    pair_probe_fit_balanced_accuracies: Optional[np.ndarray] = None
    pair_probe_eval_sensitivities: Optional[np.ndarray] = None
    pair_probe_eval_specificities: Optional[np.ndarray] = None
    pair_current_absence_eval_fractions: Optional[np.ndarray] = None
    pair_alternative_strong_eval_fractions: Optional[np.ndarray] = None
    pair_probe_fit_eval_split_digests: Optional[np.ndarray] = None
    pair_calibration_class_source_ids: Optional[np.ndarray] = None
    pair_calibration_class_source_counts: Optional[np.ndarray] = None

    def _effective_negative_support_thresholds(self) -> np.ndarray:
        count = len(self.class_names)
        if self.negative_support_thresholds is None:
            return np.full(count, 0.02, dtype=np.float32)
        return np.asarray(self.negative_support_thresholds, dtype=np.float32)

    def _effective_pair_calibration_source_membership(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        count = len(self.class_names)
        source_ids = (
            np.asarray(self.pair_calibration_class_source_ids)
            if self.pair_calibration_class_source_ids is not None
            else np.full((count, 0), "", dtype="<U16")
        )
        source_counts = (
            np.asarray(self.pair_calibration_class_source_counts)
            if self.pair_calibration_class_source_counts is not None
            else np.zeros(count, dtype=np.int32)
        )
        return source_ids, source_counts

    def _effective_pair_arrays(self) -> Dict[str, np.ndarray]:
        count = len(self.class_names)
        # V4 never fabricates learned pair reliability from class reliability.
        # A pair is confirm-capable only after exact-view source-disjoint
        # calibration has explicitly populated every required field.
        default_reliable = np.zeros((count, count), dtype=bool)
        default_tiers = np.full((count, count), "low", dtype="<U8")
        default_sources = np.zeros((count, count), dtype=np.int32)
        default_aurocs = np.zeros((count, count), dtype=np.float32)
        default_thresholds = np.full(
            (count, count), 0.02, dtype=np.float32
        )
        equal_weights = np.asarray(
            [-math.sqrt(0.5), math.sqrt(0.5)], dtype=np.float32
        )
        default_weights = np.broadcast_to(
            equal_weights,
            (count, count, 2),
        ).copy()
        default_fold_digest = hashlib.sha256(
            f"{PAIR_PROBE_CONTRACT}:in-memory-default".encode("utf-8")
        ).hexdigest()
        support_thresholds = np.asarray(
            self.support_thresholds, dtype=np.float32
        )
        feasible_current_negative = np.minimum(
            self._effective_negative_support_thresholds(),
            np.nextafter(support_thresholds, np.float32(-np.inf)),
        )
        strong_thresholds = np.asarray(
            self.strong_support_thresholds, dtype=np.float32
        )
        negative_thresholds = self._effective_negative_support_thresholds()
        feasible_alternative_negative = np.minimum(
            negative_thresholds,
            np.nextafter(support_thresholds, np.float32(-np.inf)),
        )
        defaults: Dict[str, np.ndarray] = {
            "pair_reliable": default_reliable,
            "pair_reliability_tiers": default_tiers,
            "pair_heldout_aurocs": default_aurocs.copy(),
            "pair_dominance_thresholds": default_thresholds.copy(),
            "pair_current_negative_thresholds": np.broadcast_to(
                feasible_current_negative[:, None],
                (count, count),
            ).copy(),
            "pair_current_presence_thresholds": np.broadcast_to(
                support_thresholds[:, None],
                (count, count),
            ).copy(),
            "pair_current_strong_thresholds": np.broadcast_to(
                strong_thresholds[:, None],
                (count, count),
            ).copy(),
            "pair_alternative_negative_thresholds": np.broadcast_to(
                feasible_alternative_negative[None, :],
                (count, count),
            ).copy(),
            "pair_alternative_presence_thresholds": np.broadcast_to(
                support_thresholds[None, :],
                (count, count),
            ).copy(),
            "pair_alternative_strong_thresholds": np.broadcast_to(
                strong_thresholds[None, :],
                (count, count),
            ).copy(),
            "pair_current_source_counts": default_sources.copy(),
            "pair_alternative_source_counts": default_sources.T.copy(),
            "pair_current_patch_counts": np.broadcast_to(
                np.asarray(self.calibration_target_patch_counts, dtype=np.int32)[
                    :, None
                ],
                (count, count),
            ).copy(),
            "pair_alternative_patch_counts": np.broadcast_to(
                np.asarray(self.calibration_target_patch_counts, dtype=np.int32)[
                    None, :
                ],
                (count, count),
            ).copy(),
            "pair_alternative_passing_source_fractions": default_reliable.astype(
                np.float32
            ),
            "pair_probe_weights": default_weights,
            "pair_probe_thresholds": default_thresholds.copy(),
            "pair_probe_oof_aurocs": default_aurocs.copy(),
            "pair_probe_fold_counts": np.where(
                default_reliable, 2, 0
            ).astype(np.int32),
            "pair_probe_fit_statuses": np.full(
                (count, count), "not_fitted", dtype="<U24"
            ),
            "pair_probe_fold_digests": np.full(
                (count, count), "", dtype="<U64"
            ),
            "pair_probe_eval_auroc_lower_bounds": np.zeros(
                (count, count), dtype=np.float32
            ),
            "pair_probe_fit_current_source_counts": default_sources.copy(),
            "pair_probe_fit_alternative_source_counts": default_sources.copy(),
            "pair_probe_eval_current_source_counts": default_sources.copy(),
            "pair_probe_eval_alternative_source_counts": default_sources.copy(),
            "pair_probe_fit_balanced_accuracies": default_aurocs.copy(),
            "pair_probe_eval_sensitivities": default_aurocs.copy(),
            "pair_probe_eval_specificities": default_aurocs.copy(),
            "pair_current_absence_eval_fractions": default_aurocs.copy(),
            "pair_alternative_strong_eval_fractions": default_aurocs.copy(),
            "pair_probe_fit_eval_split_digests": np.full(
                (count, count), "", dtype="<U64"
            ),
        }
        for field_name in (
            "pair_reliable",
            "pair_reliability_tiers",
            "pair_heldout_aurocs",
            "pair_dominance_thresholds",
            "pair_current_negative_thresholds",
            "pair_current_presence_thresholds",
            "pair_current_strong_thresholds",
            "pair_alternative_negative_thresholds",
            "pair_alternative_presence_thresholds",
            "pair_alternative_strong_thresholds",
            "pair_current_source_counts",
            "pair_alternative_source_counts",
            "pair_current_patch_counts",
            "pair_alternative_patch_counts",
            "pair_alternative_passing_source_fractions",
        ):
            supplied = getattr(self, field_name)
            if supplied is not None:
                defaults[field_name] = np.asarray(supplied)
        for field_name in (
            "pair_probe_weights",
            "pair_probe_thresholds",
            "pair_probe_oof_aurocs",
            "pair_probe_fold_counts",
            "pair_probe_fit_statuses",
            "pair_probe_fold_digests",
            "pair_probe_eval_auroc_lower_bounds",
            "pair_probe_fit_current_source_counts",
            "pair_probe_fit_alternative_source_counts",
            "pair_probe_eval_current_source_counts",
            "pair_probe_eval_alternative_source_counts",
            "pair_probe_fit_balanced_accuracies",
            "pair_probe_eval_sensitivities",
            "pair_probe_eval_specificities",
            "pair_current_absence_eval_fractions",
            "pair_alternative_strong_eval_fractions",
            "pair_probe_fit_eval_split_digests",
        ):
            supplied = getattr(self, field_name)
            if supplied is not None:
                defaults[field_name] = np.asarray(supplied)

        # The old names remain exact aliases in the v3 artifact so downstream
        # readers can migrate without interpreting two competing calibrations.
        if self.pair_probe_thresholds is None:
            defaults["pair_probe_thresholds"] = np.asarray(
                defaults["pair_dominance_thresholds"], dtype=np.float32
            ).copy()
        defaults["pair_dominance_thresholds"] = np.asarray(
            defaults["pair_probe_thresholds"], dtype=np.float32
        ).copy()
        if self.pair_probe_oof_aurocs is None:
            defaults["pair_probe_oof_aurocs"] = np.asarray(
                defaults["pair_heldout_aurocs"], dtype=np.float32
            ).copy()
        defaults["pair_heldout_aurocs"] = np.asarray(
            defaults["pair_probe_oof_aurocs"], dtype=np.float32
        ).copy()
        if self.pair_probe_fold_counts is None:
            defaults["pair_probe_fold_counts"] = np.zeros(
                (count, count), dtype=np.int32
            )
        if self.pair_probe_fit_statuses is None:
            defaults["pair_probe_fit_statuses"] = np.full(
                (count, count), "not_fitted", dtype="<U24"
            )
        if self.pair_probe_fold_digests is None:
            defaults["pair_probe_fold_digests"] = np.full(
                (count, count), "", dtype="<U64"
            )
        if count and self.pair_probe_fold_counts is None:
            np.fill_diagonal(defaults["pair_probe_fold_counts"], 0)
        if count and self.pair_probe_fit_statuses is None:
            np.fill_diagonal(
                defaults["pair_probe_fit_statuses"], "not_applicable"
            )
        if count and self.pair_probe_fold_digests is None:
            np.fill_diagonal(defaults["pair_probe_fold_digests"], "")
        return defaults

    def _uncalibrated_pair_arrays(self) -> Dict[str, np.ndarray]:
        """Return intrinsic defaults without carrying stale pair-fit state."""

        pair_contract_fields = {
            "pair_probe_contract",
            "pair_probe_view_contract",
            "pair_probe_lower_bound_contract",
        }
        cleared = {
            field_name: None
            for field_name in self.__dataclass_fields__
            if field_name.startswith("pair_")
            and field_name not in pair_contract_fields
        }
        return replace(self, **cleared)._effective_pair_arrays()

    def validate(self) -> None:
        if self.schema != REFINEMENT_SCHEMA:
            raise ValueError("class_analysis_refinement_bank_schema_invalid")
        if str(self.pair_probe_contract or "") != PAIR_PROBE_CONTRACT:
            raise ValueError(
                "class_analysis_refinement_bank_pair_probe_contract_invalid"
            )
        if str(self.pair_probe_view_contract or "") != PAIR_PROBE_VIEW_CONTRACT:
            raise ValueError(
                "class_analysis_refinement_bank_pair_probe_view_contract_invalid"
            )
        if (
            str(self.pair_probe_lower_bound_contract or "")
            != PAIR_PROBE_LOWER_BOUND_CONTRACT
        ):
            raise ValueError(
                "class_analysis_refinement_bank_pair_probe_lower_bound_contract_invalid"
            )
        calibration_status = str(self.calibration_status or "")
        if calibration_status in LEGACY_CALIBRATION_STATUSES:
            raise ValueError(
                "class_analysis_refinement_bank_calibration_legacy"
            )
        if calibration_status != CALIBRATION_STATUS_SOURCE_AWARE:
            raise ValueError(
                "class_analysis_refinement_bank_calibration_unsupported"
            )
        calibration_split_digest = str(
            self.calibration_split_digest or ""
        ).strip().lower()
        if (
            len(calibration_split_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in calibration_split_digest
            )
        ):
            raise ValueError(
                "class_analysis_refinement_bank_calibration_provenance_invalid"
            )
        if (
            int(self.calibration_heldout_source_count) < 0
            or int(self.calibration_fit_source_count) < 0
        ):
            raise ValueError(
                "class_analysis_refinement_bank_calibration_provenance_invalid"
            )
        heldout_source_ids = np.asarray(
            self.calibration_heldout_source_ids
        ).astype(str, copy=False).reshape(-1)
        fit_source_ids = np.asarray(
            self.calibration_fit_source_ids
        ).astype(str, copy=False).reshape(-1)
        source_id_sets = (
            set(heldout_source_ids.tolist()),
            set(fit_source_ids.tolist()),
        )
        if (
            heldout_source_ids.size
            != int(self.calibration_heldout_source_count)
            or fit_source_ids.size != int(self.calibration_fit_source_count)
            or len(source_id_sets[0]) != heldout_source_ids.size
            or len(source_id_sets[1]) != fit_source_ids.size
            or not source_id_sets[0].isdisjoint(source_id_sets[1])
            or any(
                len(source_id) != 16
                or any(character not in "0123456789abcdef" for character in source_id)
                for source_id in source_id_sets[0] | source_id_sets[1]
            )
            or calibration_split_digest
            != _calibration_source_split_digest(
                heldout_source_ids.tolist(), fit_source_ids.tolist()
            )
        ):
            raise ValueError(
                "class_analysis_refinement_bank_calibration_provenance_invalid"
            )
        prototypes = np.asarray(self.prototypes)
        background = np.asarray(self.background_prototypes)
        counts = np.asarray(self.prototype_counts)
        prototype_sources = np.asarray(self.prototype_source_ids)
        background_counts = np.asarray(self.background_prototype_counts)
        background_sources = np.asarray(
            self.background_prototype_source_ids
        )
        anchor_counts = np.asarray(self.anchor_counts)
        sources = np.asarray(self.distinct_source_counts)
        reliable = np.asarray(self.reliable)
        heldout_aurocs = np.asarray(self.heldout_aurocs)
        calibration_count_arrays = (
            np.asarray(self.calibration_target_patch_counts),
            np.asarray(self.calibration_background_patch_counts),
            np.asarray(self.calibration_target_source_counts),
            np.asarray(self.calibration_background_source_counts),
            np.asarray(self.calibration_target_passing_source_counts),
            np.asarray(self.fit_target_patch_counts),
            np.asarray(self.fit_background_patch_counts),
            np.asarray(self.fit_target_source_counts),
            np.asarray(self.fit_background_source_counts),
        )
        if prototypes.ndim != 3 or prototypes.shape[0] != len(self.class_names):
            raise ValueError("class_analysis_refinement_bank_shape_invalid")
        if background.shape != prototypes.shape:
            raise ValueError("class_analysis_refinement_bank_background_shape_invalid")
        if counts.shape != (len(self.class_names),):
            raise ValueError("class_analysis_refinement_bank_counts_invalid")
        if prototype_sources.shape != prototypes.shape[:2]:
            raise ValueError(
                "class_analysis_refinement_bank_provenance_invalid"
            )
        if (
            background_counts.shape != counts.shape
            or background_sources.shape != background.shape[:2]
            or anchor_counts.shape != counts.shape
            or sources.shape != counts.shape
            or reliable.shape != counts.shape
            or np.asarray(self.reliability_tiers).shape != counts.shape
            or np.asarray(self.heldout_aurocs).shape != counts.shape
            or np.asarray(self.support_thresholds).shape != counts.shape
            or np.asarray(self.strong_support_thresholds).shape != counts.shape
            or np.asarray(
                self.calibration_target_source_pass_fractions
            ).shape
            != counts.shape
            or any(
                values.shape != counts.shape
                for values in calibration_count_arrays
            )
        ):
            raise ValueError("class_analysis_refinement_bank_metadata_invalid")
        negative_thresholds = self._effective_negative_support_thresholds()
        pair_arrays = self._effective_pair_arrays()
        (
            pair_calibration_class_source_ids,
            pair_calibration_class_source_counts,
        ) = self._effective_pair_calibration_source_membership()
        pair_shape = (len(self.class_names), len(self.class_names))
        if (
            negative_thresholds.shape != counts.shape
            or pair_calibration_class_source_ids.ndim != 2
            or pair_calibration_class_source_ids.shape[0] != counts.shape[0]
            or pair_calibration_class_source_counts.shape != counts.shape
            or not np.issubdtype(
                pair_calibration_class_source_counts.dtype,
                np.integer,
            )
            or np.asarray(pair_arrays["pair_probe_weights"]).shape
            != pair_shape + (2,)
            or any(
                np.asarray(values).shape != pair_shape
                for field_name, values in pair_arrays.items()
                if field_name != "pair_probe_weights"
            )
        ):
            raise ValueError(
                "class_analysis_refinement_bank_pair_metadata_invalid"
            )
        pair_calibration_source_width = int(
            pair_calibration_class_source_ids.shape[1]
        )
        heldout_source_id_set = set(heldout_source_ids.tolist())
        for class_position in range(len(self.class_names)):
            source_count = int(
                pair_calibration_class_source_counts[class_position]
            )
            if source_count < 0 or source_count > pair_calibration_source_width:
                raise ValueError(
                    "class_analysis_refinement_bank_pair_source_provenance_invalid"
                )
            active_source_ids = (
                pair_calibration_class_source_ids[
                    class_position, :source_count
                ]
                .astype(str, copy=False)
                .tolist()
            )
            padding_source_ids = (
                pair_calibration_class_source_ids[
                    class_position, source_count:
                ]
                .astype(str, copy=False)
                .tolist()
            )
            if (
                active_source_ids != sorted(active_source_ids)
                or len(set(active_source_ids)) != len(active_source_ids)
                or any(
                    len(source_id) != 16
                    or any(
                        character not in "0123456789abcdef"
                        for character in source_id
                    )
                    or source_id not in heldout_source_id_set
                    for source_id in active_source_ids
                )
                or any(padding_source_ids)
            ):
                raise ValueError(
                    "class_analysis_refinement_bank_pair_source_provenance_invalid"
                )
        pair_reliable = np.asarray(pair_arrays["pair_reliable"], dtype=bool)
        pair_tiers = np.asarray(
            pair_arrays["pair_reliability_tiers"]
        ).astype(str, copy=False)
        pair_aurocs = np.asarray(
            pair_arrays["pair_heldout_aurocs"], dtype=np.float32
        )
        pair_dominance = np.asarray(
            pair_arrays["pair_dominance_thresholds"], dtype=np.float32
        )
        pair_current_negative = np.asarray(
            pair_arrays["pair_current_negative_thresholds"],
            dtype=np.float32,
        )
        pair_current_presence = np.asarray(
            pair_arrays["pair_current_presence_thresholds"],
            dtype=np.float32,
        )
        pair_current_strong = np.asarray(
            pair_arrays["pair_current_strong_thresholds"],
            dtype=np.float32,
        )
        pair_alternative_negative = np.asarray(
            pair_arrays["pair_alternative_negative_thresholds"],
            dtype=np.float32,
        )
        pair_alternative_presence = np.asarray(
            pair_arrays["pair_alternative_presence_thresholds"],
            dtype=np.float32,
        )
        pair_alternative_strong = np.asarray(
            pair_arrays["pair_alternative_strong_thresholds"],
            dtype=np.float32,
        )
        pair_current_sources = np.asarray(
            pair_arrays["pair_current_source_counts"]
        )
        pair_alternative_sources = np.asarray(
            pair_arrays["pair_alternative_source_counts"]
        )
        pair_current_patches = np.asarray(
            pair_arrays["pair_current_patch_counts"]
        )
        pair_alternative_patches = np.asarray(
            pair_arrays["pair_alternative_patch_counts"]
        )
        pair_pass_fractions = np.asarray(
            pair_arrays["pair_alternative_passing_source_fractions"],
            dtype=np.float32,
        )
        pair_probe_weights = np.asarray(
            pair_arrays["pair_probe_weights"], dtype=np.float32
        )
        pair_probe_thresholds = np.asarray(
            pair_arrays["pair_probe_thresholds"], dtype=np.float32
        )
        pair_probe_oof_aurocs = np.asarray(
            pair_arrays["pair_probe_oof_aurocs"], dtype=np.float32
        )
        pair_probe_fold_counts = np.asarray(
            pair_arrays["pair_probe_fold_counts"]
        )
        pair_probe_fit_statuses = np.asarray(
            pair_arrays["pair_probe_fit_statuses"]
        ).astype(str, copy=False)
        pair_probe_fold_digests = np.asarray(
            pair_arrays["pair_probe_fold_digests"]
        ).astype(str, copy=False)
        pair_probe_eval_lower_bounds = np.asarray(
            pair_arrays["pair_probe_eval_auroc_lower_bounds"],
            dtype=np.float32,
        )
        pair_probe_fit_current_sources = np.asarray(
            pair_arrays["pair_probe_fit_current_source_counts"]
        )
        pair_probe_fit_alternative_sources = np.asarray(
            pair_arrays["pair_probe_fit_alternative_source_counts"]
        )
        pair_probe_eval_current_sources = np.asarray(
            pair_arrays["pair_probe_eval_current_source_counts"]
        )
        pair_probe_eval_alternative_sources = np.asarray(
            pair_arrays["pair_probe_eval_alternative_source_counts"]
        )
        pair_probe_fit_balanced_accuracies = np.asarray(
            pair_arrays["pair_probe_fit_balanced_accuracies"],
            dtype=np.float32,
        )
        pair_probe_eval_sensitivities = np.asarray(
            pair_arrays["pair_probe_eval_sensitivities"],
            dtype=np.float32,
        )
        pair_probe_eval_specificities = np.asarray(
            pair_arrays["pair_probe_eval_specificities"],
            dtype=np.float32,
        )
        pair_current_absence_eval_fractions = np.asarray(
            pair_arrays["pair_current_absence_eval_fractions"],
            dtype=np.float32,
        )
        pair_alternative_strong_eval_fractions = np.asarray(
            pair_arrays["pair_alternative_strong_eval_fractions"],
            dtype=np.float32,
        )
        pair_probe_split_digests = np.asarray(
            pair_arrays["pair_probe_fit_eval_split_digests"]
        ).astype(str, copy=False)
        probe_norms = np.linalg.norm(pair_probe_weights, axis=2)
        valid_probe_statuses = np.asarray(
            [
                "ok",
                "insufficient_sources",
                "fold_invalid",
                "nonfinite",
                "not_fitted",
                "not_applicable",
            ]
        )
        if (
            np.any(~np.isin(pair_tiers, np.asarray(["low", "usable", "high"])))
            or np.any(pair_reliable != np.isin(pair_tiers, ["usable", "high"]))
            or not np.all(np.isfinite(negative_thresholds))
            or not np.all(np.isfinite(pair_aurocs))
            or not np.all(np.isfinite(pair_dominance))
            or not np.all(np.isfinite(pair_current_negative))
            or not np.all(np.isfinite(pair_current_presence))
            or not np.all(np.isfinite(pair_current_strong))
            or not np.all(np.isfinite(pair_alternative_negative))
            or not np.all(np.isfinite(pair_alternative_presence))
            or not np.all(np.isfinite(pair_alternative_strong))
            or not np.all(np.isfinite(pair_pass_fractions))
            or not np.all(np.isfinite(pair_probe_weights))
            or not np.all(np.isfinite(pair_probe_thresholds))
            or not np.all(np.isfinite(pair_probe_oof_aurocs))
            or not np.all(np.isfinite(pair_probe_eval_lower_bounds))
            or any(
                not np.all(np.isfinite(values))
                or np.any(values < 0.0)
                or np.any(values > 1.0)
                for values in (
                    pair_probe_fit_balanced_accuracies,
                    pair_probe_eval_sensitivities,
                    pair_probe_eval_specificities,
                    pair_current_absence_eval_fractions,
                    pair_alternative_strong_eval_fractions,
                )
            )
            or np.any(pair_aurocs < 0.0)
            or np.any(pair_aurocs > 1.0)
            or np.any(pair_probe_oof_aurocs < 0.0)
            or np.any(pair_probe_oof_aurocs > 1.0)
            or np.any(pair_probe_eval_lower_bounds < 0.0)
            or np.any(pair_probe_eval_lower_bounds > pair_probe_oof_aurocs)
            or np.any(pair_pass_fractions < 0.0)
            or np.any(pair_pass_fractions > 1.0)
            or not np.array_equal(
                pair_pass_fractions,
                pair_probe_eval_sensitivities,
            )
            or np.any(pair_probe_weights[..., 0] > 1e-6)
            or np.any(pair_probe_weights[..., 1] < -1e-6)
            or not np.allclose(probe_norms, 1.0, rtol=0.0, atol=1e-5)
            or not np.array_equal(pair_aurocs, pair_probe_oof_aurocs)
            or not np.array_equal(pair_dominance, pair_probe_thresholds)
            or np.any(~np.isin(pair_probe_fit_statuses, valid_probe_statuses))
            or any(
                not np.issubdtype(values.dtype, np.integer)
                for values in (
                    pair_current_sources,
                    pair_alternative_sources,
                    pair_current_patches,
                    pair_alternative_patches,
                    pair_probe_fold_counts,
                    pair_probe_fit_current_sources,
                    pair_probe_fit_alternative_sources,
                    pair_probe_eval_current_sources,
                    pair_probe_eval_alternative_sources,
                )
            )
            or np.any(pair_current_sources < 0)
            or np.any(pair_alternative_sources < 0)
            or np.any(pair_current_patches < 0)
            or np.any(pair_alternative_patches < 0)
            or np.any(pair_probe_fold_counts < 0)
            or np.any(pair_probe_fit_current_sources < 0)
            or np.any(pair_probe_fit_alternative_sources < 0)
            or np.any(pair_probe_eval_current_sources < 0)
            or np.any(pair_probe_eval_alternative_sources < 0)
            or np.any(pair_current_negative >= pair_current_presence)
            or np.any(pair_current_presence > pair_current_strong)
            or np.any(pair_alternative_negative >= pair_alternative_presence)
            or np.any(pair_alternative_presence > pair_alternative_strong)
            or np.any(np.abs(pair_probe_thresholds) > PAIR_PROBE_SCORE_ABS_BOUND)
            or np.any(np.diag(pair_reliable))
        ):
            raise ValueError(
                "class_analysis_refinement_bank_pair_metadata_invalid"
            )
        for current in range(pair_shape[0]):
            for alternative in range(pair_shape[1]):
                status = str(
                    pair_probe_fit_statuses[current, alternative]
                )
                fold_count = int(
                    pair_probe_fold_counts[current, alternative]
                )
                digest = str(
                    pair_probe_fold_digests[current, alternative]
                )
                split_digest = str(
                    pair_probe_split_digests[current, alternative]
                )
                if current == alternative:
                    if (
                        status != "not_applicable"
                        or fold_count != 0
                        or digest
                        or split_digest
                    ):
                        raise ValueError(
                            "class_analysis_refinement_bank_pair_metadata_invalid"
                        )
                    continue
                if status == "ok":
                    if (
                        fold_count != 1
                        or len(digest) != 64
                        or len(split_digest) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in digest + split_digest
                        )
                        or digest != split_digest
                        or int(pair_current_sources[current, alternative])
                        != int(
                            pair_probe_eval_current_sources[
                                current, alternative
                            ]
                        )
                        or int(pair_alternative_sources[current, alternative])
                        != int(
                            pair_probe_eval_alternative_sources[
                                current, alternative
                            ]
                        )
                        or int(pair_current_patches[current, alternative])
                        != int(pair_current_sources[current, alternative])
                        or int(pair_alternative_patches[current, alternative])
                        != int(pair_alternative_sources[current, alternative])
                        or int(
                            pair_calibration_class_source_counts[current]
                        )
                        != int(
                            pair_probe_fit_current_sources[
                                current, alternative
                            ]
                            + pair_probe_eval_current_sources[
                                current, alternative
                            ]
                        )
                        or int(
                            pair_calibration_class_source_counts[alternative]
                        )
                        != int(
                            pair_probe_fit_alternative_sources[
                                current, alternative
                            ]
                            + pair_probe_eval_alternative_sources[
                                current, alternative
                            ]
                        )
                    ):
                        raise ValueError(
                            "class_analysis_refinement_bank_pair_metadata_invalid"
                        )
                elif fold_count != 0 or digest or split_digest:
                    raise ValueError(
                        "class_analysis_refinement_bank_pair_metadata_invalid"
                    )
        for current, alternative in np.argwhere(pair_reliable):
            if (
                not pair_metrics_are_reliable(
                    current_class_reliable=bool(reliable[int(current)]),
                    alternative_class_reliable=bool(
                        reliable[int(alternative)]
                    ),
                    fit_current_source_count=int(
                        pair_probe_fit_current_sources[current, alternative]
                    ),
                    fit_alternative_source_count=int(
                        pair_probe_fit_alternative_sources[
                            current, alternative
                        ]
                    ),
                    eval_current_source_count=int(
                        pair_probe_eval_current_sources[current, alternative]
                    ),
                    eval_alternative_source_count=int(
                        pair_probe_eval_alternative_sources[
                            current, alternative
                        ]
                    ),
                    eval_auroc=float(pair_aurocs[current, alternative]),
                    eval_auroc_lower_bound=float(
                        pair_probe_eval_lower_bounds[current, alternative]
                    ),
                    fit_balanced_accuracy=float(
                        pair_probe_fit_balanced_accuracies[
                            current, alternative
                        ]
                    ),
                    eval_sensitivity=float(
                        pair_probe_eval_sensitivities[current, alternative]
                    ),
                    eval_specificity=float(
                        pair_probe_eval_specificities[current, alternative]
                    ),
                    current_absence_eval_fraction=float(
                        pair_current_absence_eval_fractions[
                            current, alternative
                        ]
                    ),
                    alternative_strong_eval_fraction=float(
                        pair_alternative_strong_eval_fractions[
                            current, alternative
                        ]
                    ),
                )
                or int(pair_current_patches[current, alternative]) <= 0
                or int(pair_alternative_patches[current, alternative]) <= 0
                or str(pair_probe_fit_statuses[current, alternative]) != "ok"
                or int(pair_probe_fold_counts[current, alternative]) != 1
                or len(
                    str(pair_probe_fold_digests[current, alternative])
                )
                != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in str(
                        pair_probe_fold_digests[current, alternative]
                    )
                )
            ):
                raise ValueError(
                    "class_analysis_refinement_bank_pair_metadata_invalid"
                )
        if any(
            not np.issubdtype(values.dtype, np.integer)
            for values in (
                counts,
                background_counts,
                anchor_counts,
                sources,
            )
        ):
            raise ValueError("class_analysis_refinement_bank_metadata_invalid")
        if (
            np.any(anchor_counts < 0)
            or np.any(sources < 0)
            or np.any(sources > anchor_counts)
            or np.any(
                sources
                > (
                    int(self.calibration_heldout_source_count)
                    + int(self.calibration_fit_source_count)
                )
            )
        ):
            raise ValueError("class_analysis_refinement_bank_metadata_invalid")
        if any(np.any(values < 0) for values in calibration_count_arrays):
            raise ValueError(
                "class_analysis_refinement_bank_calibration_provenance_invalid"
            )
        if any(
            not np.issubdtype(values.dtype, np.integer)
            for values in calibration_count_arrays
        ):
            raise ValueError(
                "class_analysis_refinement_bank_calibration_provenance_invalid"
            )
        for patch_counts, source_counts in (
            (
                calibration_count_arrays[0],
                calibration_count_arrays[2],
            ),
            (
                calibration_count_arrays[1],
                calibration_count_arrays[3],
            ),
            (
                calibration_count_arrays[5],
                calibration_count_arrays[7],
            ),
            (
                calibration_count_arrays[6],
                calibration_count_arrays[8],
            ),
        ):
            if np.any(source_counts > patch_counts):
                raise ValueError(
                    "class_analysis_refinement_bank_calibration_provenance_invalid"
                )
        if (
            np.any(
                calibration_count_arrays[2]
                > int(self.calibration_heldout_source_count)
            )
            or np.any(
                calibration_count_arrays[3]
                > int(self.calibration_heldout_source_count)
            )
            or np.any(
                calibration_count_arrays[7]
                > int(self.calibration_fit_source_count)
            )
            or np.any(
                calibration_count_arrays[8]
                > int(self.calibration_fit_source_count)
            )
        ):
            raise ValueError(
                "class_analysis_refinement_bank_calibration_provenance_invalid"
            )
        passing_source_counts = calibration_count_arrays[4]
        pass_fractions = np.asarray(
            self.calibration_target_source_pass_fractions,
            dtype=np.float32,
        )
        target_source_counts = calibration_count_arrays[2]
        expected_pass_fractions = np.divide(
            passing_source_counts,
            target_source_counts,
            out=np.zeros_like(pass_fractions),
            where=target_source_counts > 0,
        )
        if (
            np.any(passing_source_counts > target_source_counts)
            or not np.all(np.isfinite(pass_fractions))
            or np.any(pass_fractions < 0.0)
            or np.any(pass_fractions > 1.0)
            or not np.allclose(
                pass_fractions,
                expected_pass_fractions,
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise ValueError(
                "class_analysis_refinement_bank_calibration_provenance_invalid"
            )
        if np.any(counts < 0) or np.any(counts > prototypes.shape[1]):
            raise ValueError("class_analysis_refinement_bank_counts_invalid")
        if np.any(background_counts < 0) or np.any(
            background_counts > background.shape[1]
        ):
            raise ValueError(
                "class_analysis_refinement_bank_background_counts_invalid"
            )
        tiers = np.asarray(self.reliability_tiers).astype(
            str,
            copy=False,
        )
        if np.any(
            ~np.isin(tiers, np.asarray(["low", "usable", "high"]))
        ) or np.any(reliable != np.isin(tiers, ["usable", "high"])):
            raise ValueError("class_analysis_refinement_bank_metadata_invalid")
        reliable_positions = np.flatnonzero(reliable)
        for position in reliable_positions.tolist():
            required_anchor_count = 64 if tiers[position] == "high" else 24
            required_source_count = 8 if tiers[position] == "high" else 5
            if (
                int(anchor_counts[position]) < required_anchor_count
                or int(sources[position]) < required_source_count
                or float(self.heldout_aurocs[position]) < 0.70
                or int(calibration_count_arrays[2][position])
                < MIN_HELDOUT_SOURCE_GROUPS
                or int(calibration_count_arrays[3][position])
                < MIN_HELDOUT_SOURCE_GROUPS
                or int(calibration_count_arrays[4][position])
                < MIN_HELDOUT_SOURCE_GROUPS
                or float(pass_fractions[position])
                < MIN_HELDOUT_TARGET_SOURCE_PASS_FRACTION
                or int(calibration_count_arrays[7][position])
                < MIN_RELIABLE_FIT_SOURCE_GROUPS
                or int(calibration_count_arrays[8][position])
                < MIN_RELIABLE_FIT_SOURCE_GROUPS
            ):
                raise ValueError(
                    "class_analysis_refinement_bank_calibration_provenance_invalid"
                )
            target_source_set = {
                str(value)
                for value in prototype_sources[
                    position, : int(counts[position])
                ].tolist()
                if str(value)
            }
            background_source_set = {
                str(value)
                for value in background_sources[
                    position, : int(background_counts[position])
                ].tolist()
                if str(value)
            }
            if (
                len(target_source_set) < MIN_RELIABLE_FIT_SOURCE_GROUPS
                or len(background_source_set)
                < MIN_RELIABLE_FIT_SOURCE_GROUPS
            ):
                raise ValueError(
                    "class_analysis_refinement_bank_calibration_provenance_invalid"
                )
        if (
            not np.all(np.isfinite(prototypes))
            or not np.all(np.isfinite(background))
            or not np.all(np.isfinite(self.projection_mean))
            or not np.all(np.isfinite(self.projection_components))
            or not np.all(np.isfinite(heldout_aurocs))
            or not np.all(np.isfinite(self.support_thresholds))
            or not np.all(np.isfinite(self.strong_support_thresholds))
        ):
            raise ValueError("class_analysis_refinement_bank_nonfinite")
        if np.any(heldout_aurocs < 0.0) or np.any(heldout_aurocs > 1.0):
            raise ValueError(
                "class_analysis_refinement_bank_calibration_provenance_invalid"
            )
        if (
            any(not str(class_name).strip() for class_name in self.class_names)
            or len(set(self.class_names)) != len(self.class_names)
        ):
            raise ValueError("class_analysis_refinement_bank_classes_invalid")
        if np.any(
            np.asarray(self.strong_support_thresholds, dtype=np.float32)
            < np.asarray(self.support_thresholds, dtype=np.float32)
        ) or np.any(
            negative_thresholds
            >= np.asarray(self.support_thresholds, dtype=np.float32)
        ) or np.any(
            np.abs(np.asarray(self.support_thresholds, dtype=np.float32)) > 2.0
        ) or np.any(
            np.abs(
                np.asarray(self.strong_support_thresholds, dtype=np.float32)
            )
            > 2.0
        ) or np.any(np.abs(negative_thresholds) > 2.0):
            raise ValueError("class_analysis_refinement_bank_thresholds_invalid")
        components = np.asarray(self.projection_components)
        mean = np.asarray(self.projection_mean)
        if (
            components.ndim != 2
            or mean.ndim != 1
            or components.shape[1] != mean.shape[0]
            or components.shape[0] != prototypes.shape[2]
        ):
            raise ValueError("class_analysis_refinement_bank_projection_invalid")

    def class_position(self, class_name: str) -> Optional[int]:
        try:
            return self.class_names.index(str(class_name or ""))
        except ValueError:
            return None

    def _class_reference_pool(
        self,
        class_name: str,
        *,
        background: bool,
        exclude_source_key: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        position = self.class_position(class_name)
        if position is None:
            width = int(self.prototypes.shape[-1])
            return (
                np.empty((0, width), dtype=np.float32),
                np.empty((0,), dtype="<U16"),
            )
        if background:
            count = int(self.background_prototype_counts[position])
            values = np.asarray(
                self.background_prototypes[position, :count],
                dtype=np.float32,
            )
            source_ids = np.asarray(
                self.background_prototype_source_ids[position, :count]
            )
        else:
            count = int(self.prototype_counts[position])
            values = np.asarray(
                self.prototypes[position, :count],
                dtype=np.float32,
            )
            source_ids = np.asarray(
                self.prototype_source_ids[position, :count]
            )
        return _exclude_source_prototypes(
            values,
            source_ids,
            exclude_source_key=exclude_source_key,
        )

    def class_prototypes(
        self,
        class_name: str,
        *,
        exclude_source_key: str = "",
    ) -> np.ndarray:
        values, _source_ids = self._class_reference_pool(
            class_name,
            background=False,
            exclude_source_key=exclude_source_key,
        )
        return values

    def class_background_prototypes(
        self,
        class_name: str,
        *,
        exclude_source_key: str = "",
    ) -> np.ndarray:
        values, _source_ids = self._class_reference_pool(
            class_name,
            background=True,
            exclude_source_key=exclude_source_key,
        )
        return values

    def class_has_source_independent_support(
        self,
        class_name: str,
        *,
        exclude_source_key: str,
    ) -> bool:
        target_values, target_sources = self._class_reference_pool(
            class_name,
            background=False,
            exclude_source_key=exclude_source_key,
        )
        background_values, background_sources = self._class_reference_pool(
            class_name,
            background=True,
            exclude_source_key=exclude_source_key,
        )
        return bool(
            _prototype_pool_is_source_independent(
                target_values,
                target_sources,
            )
            and _prototype_pool_is_source_independent(
                background_values,
                background_sources,
            )
        )

    def calibration_source_role(self, source_key: str) -> str:
        source_id = _source_fingerprint(source_key)
        if not source_id:
            return "unknown"
        if source_id in set(
            np.asarray(self.calibration_heldout_source_ids)
            .astype(str, copy=False)
            .reshape(-1)
            .tolist()
        ):
            return "heldout"
        if source_id in set(
            np.asarray(self.calibration_fit_source_ids)
            .astype(str, copy=False)
            .reshape(-1)
            .tolist()
        ):
            return "fit"
        return "unknown"

    def project_tokens(self, values: np.ndarray) -> np.ndarray:
        raw = np.asarray(values, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != self.projection_mean.shape[0]:
            raise ValueError("class_analysis_refinement_projection_input_invalid")
        projected = (
            raw - np.asarray(self.projection_mean, dtype=np.float32)
        ) @ np.asarray(self.projection_components, dtype=np.float32).T
        return _normalise_rows(projected)

    def class_support_threshold(self, class_name: str) -> float:
        position = self.class_position(class_name)
        return (
            float(self.support_thresholds[position])
            if position is not None
            # Cosine-margin support is bounded by [-2, 2]. Keep an
            # unavailable class conservatively impossible to confirm without
            # emitting non-finite values into persisted evidence.
            else 2.0
        )

    def class_strong_support_threshold(self, class_name: str) -> float:
        position = self.class_position(class_name)
        return (
            float(self.strong_support_thresholds[position])
            if position is not None
            else 2.0
        )

    def class_negative_support_threshold(self, class_name: str) -> float:
        position = self.class_position(class_name)
        thresholds = self._effective_negative_support_thresholds()
        return (
            float(thresholds[position])
            if position is not None
            else -2.0
        )

    def directed_pair_calibration_source_roles(
        self,
        current_class: str,
        alternative_class: str,
        *,
        source_key: str = "",
    ) -> Tuple[str, ...]:
        """Return pair roles for which ``source_key`` supplied an exact view."""

        clean_source = str(source_key or "").strip()
        current = self.class_position(current_class)
        alternative = self.class_position(alternative_class)
        if (
            not clean_source
            or current is None
            or alternative is None
            or current == alternative
        ):
            return ()
        source_id = _source_fingerprint(clean_source)
        source_ids, source_counts = (
            self._effective_pair_calibration_source_membership()
        )
        roles: List[str] = []
        for role, position in (
            ("current_class", current),
            ("alternative_class", alternative),
        ):
            count = int(source_counts[position])
            if source_id in set(
                source_ids[position, :count]
                .astype(str, copy=False)
                .tolist()
            ):
                roles.append(role)
        return tuple(roles)

    def directed_pair_metadata(
        self,
        current_class: str,
        alternative_class: str,
        *,
        query_source_key: str = "",
    ) -> Dict[str, Any]:
        current = self.class_position(current_class)
        alternative = self.class_position(alternative_class)
        query_source = str(query_source_key or "").strip()
        query_source_id = (
            _source_fingerprint(query_source) if query_source else ""
        )
        if current is None or alternative is None or current == alternative:
            return {
                "reliable": False,
                "bank_reliable": False,
                "diagnostic_reliable": False,
                "diagnostic_bank_reliable": False,
                "diagnostic_reliability_contract": (
                    DIAGNOSTIC_PAIR_RELIABILITY_CONTRACT
                ),
                "candidate_source_excluded": False,
                "candidate_source_fingerprint": query_source_id,
                "candidate_source_membership_roles": [],
                "tier": "low",
                "heldout_auroc": 0.0,
                "dominance_threshold": 2.0,
                "probe_contract": PAIR_PROBE_CONTRACT,
                "probe_view_contract": PAIR_PROBE_VIEW_CONTRACT,
                "probe_lower_bound_contract": PAIR_PROBE_LOWER_BOUND_CONTRACT,
                "probe_feature_names": list(PAIR_PROBE_FEATURE_NAMES),
                "probe_weights": [-math.sqrt(0.5), math.sqrt(0.5)],
                "probe_threshold": 2.0,
                "probe_oof_auroc": 0.0,
                "probe_eval_auroc_lower_bound": 0.0,
                "probe_fold_count": 0,
                "probe_fit_status": "not_applicable",
                "probe_fold_digest": "",
                "current_negative_threshold": -2.0,
                "current_presence_threshold": 2.0,
                "current_strong_threshold": 2.0,
                "alternative_negative_threshold": -2.0,
                "alternative_presence_threshold": 2.0,
                "alternative_strong_threshold": 2.0,
                "fit_current_source_count": 0,
                "fit_alternative_source_count": 0,
                "eval_current_source_count": 0,
                "eval_alternative_source_count": 0,
                "probe_fit_balanced_accuracy": 0.0,
                "probe_eval_sensitivity": 0.0,
                "probe_eval_specificity": 0.0,
                "current_absence_eval_fraction": 0.0,
                "alternative_strong_eval_fraction": 0.0,
                "fit_eval_split_digest": "",
                "current_source_count": 0,
                "alternative_source_count": 0,
                "current_patch_count": 0,
                "alternative_patch_count": 0,
                "alternative_passing_source_fraction": 0.0,
            }
        arrays = self._effective_pair_arrays()
        candidate_source_membership_roles = list(
            self.directed_pair_calibration_source_roles(
                current_class,
                alternative_class,
                source_key=query_source,
            )
        )
        bank_reliable = bool(
            arrays["pair_reliable"][current, alternative]
        )
        candidate_source_excluded = bool(
            candidate_source_membership_roles
        )
        diagnostic_bank_reliable = bool(
            str(self.pair_probe_contract) == PAIR_PROBE_CONTRACT
            and str(self.pair_probe_view_contract)
            == PAIR_PROBE_VIEW_CONTRACT
            and str(self.pair_probe_lower_bound_contract)
            == PAIR_PROBE_LOWER_BOUND_CONTRACT
            and str(
                arrays["pair_probe_fit_statuses"][current, alternative]
            )
            == "ok"
            and int(
                arrays["pair_probe_fold_counts"][current, alternative]
            )
            == 1
            and str(
                arrays["pair_probe_fold_digests"][current, alternative]
            )
            == str(
                arrays["pair_probe_fit_eval_split_digests"][
                    current, alternative
                ]
            )
            and pair_metrics_are_diagnostic(
                current_class_reliable=self.class_is_reliable(
                    current_class
                ),
                alternative_class_reliable=self.class_is_reliable(
                    alternative_class
                ),
                fit_current_source_count=int(
                    arrays["pair_probe_fit_current_source_counts"][
                        current, alternative
                    ]
                ),
                fit_alternative_source_count=int(
                    arrays["pair_probe_fit_alternative_source_counts"][
                        current, alternative
                    ]
                ),
                eval_current_source_count=int(
                    arrays["pair_probe_eval_current_source_counts"][
                        current, alternative
                    ]
                ),
                eval_alternative_source_count=int(
                    arrays["pair_probe_eval_alternative_source_counts"][
                        current, alternative
                    ]
                ),
                eval_auroc=float(
                    arrays["pair_probe_oof_aurocs"][current, alternative]
                ),
                eval_auroc_lower_bound=float(
                    arrays["pair_probe_eval_auroc_lower_bounds"][
                        current, alternative
                    ]
                ),
                fit_balanced_accuracy=float(
                    arrays["pair_probe_fit_balanced_accuracies"][
                        current, alternative
                    ]
                ),
                eval_sensitivity=float(
                    arrays["pair_probe_eval_sensitivities"][
                        current, alternative
                    ]
                ),
                eval_specificity=float(
                    arrays["pair_probe_eval_specificities"][
                        current, alternative
                    ]
                ),
            )
        )
        return {
            "reliable": bool(bank_reliable and not candidate_source_excluded),
            "bank_reliable": bank_reliable,
            "diagnostic_reliable": bool(
                diagnostic_bank_reliable and not candidate_source_excluded
            ),
            "diagnostic_bank_reliable": diagnostic_bank_reliable,
            "diagnostic_reliability_contract": (
                DIAGNOSTIC_PAIR_RELIABILITY_CONTRACT
            ),
            "candidate_source_excluded": candidate_source_excluded,
            "candidate_source_fingerprint": query_source_id,
            "candidate_source_membership_roles": (
                candidate_source_membership_roles
            ),
            "tier": str(
                arrays["pair_reliability_tiers"][current, alternative]
            ),
            "heldout_auroc": float(
                arrays["pair_heldout_aurocs"][current, alternative]
            ),
            "dominance_threshold": float(
                arrays["pair_dominance_thresholds"][current, alternative]
            ),
            "probe_contract": str(self.pair_probe_contract),
            "probe_view_contract": str(self.pair_probe_view_contract),
            "probe_lower_bound_contract": str(
                self.pair_probe_lower_bound_contract
            ),
            "probe_feature_names": list(PAIR_PROBE_FEATURE_NAMES),
            "probe_weights": [
                float(value)
                for value in arrays["pair_probe_weights"][
                    current, alternative
                ].tolist()
            ],
            "probe_threshold": float(
                arrays["pair_probe_thresholds"][current, alternative]
            ),
            "probe_oof_auroc": float(
                arrays["pair_probe_oof_aurocs"][current, alternative]
            ),
            "probe_eval_auroc_lower_bound": float(
                arrays["pair_probe_eval_auroc_lower_bounds"][
                    current, alternative
                ]
            ),
            "probe_fold_count": int(
                arrays["pair_probe_fold_counts"][current, alternative]
            ),
            "probe_fit_status": str(
                arrays["pair_probe_fit_statuses"][current, alternative]
            ),
            "probe_fold_digest": str(
                arrays["pair_probe_fold_digests"][current, alternative]
            ),
            "current_negative_threshold": float(
                arrays["pair_current_negative_thresholds"][
                    current, alternative
                ]
            ),
            "current_presence_threshold": float(
                arrays["pair_current_presence_thresholds"][
                    current, alternative
                ]
            ),
            "current_strong_threshold": float(
                arrays["pair_current_strong_thresholds"][
                    current, alternative
                ]
            ),
            "alternative_negative_threshold": float(
                arrays["pair_alternative_negative_thresholds"][
                    current, alternative
                ]
            ),
            "alternative_presence_threshold": float(
                arrays["pair_alternative_presence_thresholds"][
                    current, alternative
                ]
            ),
            "alternative_strong_threshold": float(
                arrays["pair_alternative_strong_thresholds"][
                    current, alternative
                ]
            ),
            "fit_current_source_count": int(
                arrays["pair_probe_fit_current_source_counts"][
                    current, alternative
                ]
            ),
            "fit_alternative_source_count": int(
                arrays["pair_probe_fit_alternative_source_counts"][
                    current, alternative
                ]
            ),
            "eval_current_source_count": int(
                arrays["pair_probe_eval_current_source_counts"][
                    current, alternative
                ]
            ),
            "eval_alternative_source_count": int(
                arrays["pair_probe_eval_alternative_source_counts"][
                    current, alternative
                ]
            ),
            "probe_fit_balanced_accuracy": float(
                arrays["pair_probe_fit_balanced_accuracies"][
                    current, alternative
                ]
            ),
            "probe_eval_sensitivity": float(
                arrays["pair_probe_eval_sensitivities"][
                    current, alternative
                ]
            ),
            "probe_eval_specificity": float(
                arrays["pair_probe_eval_specificities"][
                    current, alternative
                ]
            ),
            "current_absence_eval_fraction": float(
                arrays["pair_current_absence_eval_fractions"][
                    current, alternative
                ]
            ),
            "alternative_strong_eval_fraction": float(
                arrays["pair_alternative_strong_eval_fractions"][
                    current, alternative
                ]
            ),
            "fit_eval_split_digest": str(
                arrays["pair_probe_fit_eval_split_digests"][
                    current, alternative
                ]
            ),
            "current_source_count": int(
                arrays["pair_current_source_counts"][current, alternative]
            ),
            "alternative_source_count": int(
                arrays["pair_alternative_source_counts"][
                    current, alternative
                ]
            ),
            "current_patch_count": int(
                arrays["pair_current_patch_counts"][current, alternative]
            ),
            "alternative_patch_count": int(
                arrays["pair_alternative_patch_counts"][
                    current, alternative
                ]
            ),
            "alternative_passing_source_fraction": float(
                arrays["pair_alternative_passing_source_fractions"][
                    current, alternative
                ]
            ),
        }

    def directed_pair_is_reliable(
        self,
        current_class: str,
        alternative_class: str,
        *,
        query_source_key: str = "",
    ) -> bool:
        return bool(
            self.directed_pair_metadata(
                current_class,
                alternative_class,
                query_source_key=query_source_key,
            )["reliable"]
        )

    def pair_calibration_provenance(self) -> Dict[str, Any]:
        pairs: Dict[str, Dict[str, Any]] = {}
        reliable_count = 0
        diagnostic_reliable_count = 0
        for current in self.class_names:
            for alternative in self.class_names:
                if current == alternative:
                    continue
                metadata = self.directed_pair_metadata(current, alternative)
                reliable_count += int(bool(metadata["reliable"]))
                diagnostic_reliable_count += int(
                    bool(metadata["diagnostic_reliable"])
                )
                pairs[f"{current}->{alternative}"] = metadata
        return {
            "contract": str(self.calibration_status or ""),
            "probe_contract": str(self.pair_probe_contract or ""),
            "probe_view_contract": str(self.pair_probe_view_contract or ""),
            "probe_lower_bound_contract": str(
                self.pair_probe_lower_bound_contract or ""
            ),
            "diagnostic_reliability_contract": (
                DIAGNOSTIC_PAIR_RELIABILITY_CONTRACT
            ),
            "directed_pair_count": len(pairs),
            "reliable_directed_pair_count": reliable_count,
            "reliable_directed_pair_fraction": (
                float(reliable_count) / float(len(pairs)) if pairs else 0.0
            ),
            "diagnostic_reliable_directed_pair_count": (
                diagnostic_reliable_count
            ),
            "diagnostic_reliable_directed_pair_fraction": (
                float(diagnostic_reliable_count) / float(len(pairs))
                if pairs
                else 0.0
            ),
            "pairs": pairs,
        }

    def class_is_reliable(self, class_name: str) -> bool:
        position = self.class_position(class_name)
        return bool(position is not None and self.reliable[position])

    def class_distinct_sources(self, class_name: str) -> int:
        position = self.class_position(class_name)
        return int(self.distinct_source_counts[position]) if position is not None else 0

    def class_reliability_tier(self, class_name: str) -> str:
        position = self.class_position(class_name)
        return (
            str(self.reliability_tiers[position])
            if position is not None
            else "low"
        )

    def class_heldout_auroc(self, class_name: str) -> float:
        position = self.class_position(class_name)
        return (
            float(self.heldout_aurocs[position])
            if position is not None
            else 0.0
        )

    def calibration_split_provenance(self) -> Dict[str, Any]:
        """Return bounded, non-identifying calibration audit metadata."""

        per_class: Dict[str, Dict[str, int]] = {}
        for position, class_name in enumerate(self.class_names):
            per_class[class_name] = {
                "heldout_target_patch_count": int(
                    self.calibration_target_patch_counts[position]
                ),
                "heldout_background_patch_count": int(
                    self.calibration_background_patch_counts[position]
                ),
                "heldout_target_source_count": int(
                    self.calibration_target_source_counts[position]
                ),
                "heldout_background_source_count": int(
                    self.calibration_background_source_counts[position]
                ),
                "heldout_target_passing_source_count": int(
                    self.calibration_target_passing_source_counts[position]
                ),
                "heldout_target_source_pass_fraction": float(
                    self.calibration_target_source_pass_fractions[position]
                ),
                "fit_target_patch_count": int(
                    self.fit_target_patch_counts[position]
                ),
                "fit_background_patch_count": int(
                    self.fit_background_patch_counts[position]
                ),
                "fit_target_source_count": int(
                    self.fit_target_source_counts[position]
                ),
                "fit_background_source_count": int(
                    self.fit_background_source_counts[position]
                ),
            }
        return {
            "contract": str(self.calibration_status or ""),
            "source_group_semantics": SOURCE_GROUP_SEMANTICS,
            "source_group_independence_scope": (
                "exact_image_or_split_relpath_only_not_capture_or_site_v1"
            ),
            "anchor_patch_selection_contract": (
                "source_recurring_target_vs_own_context_ring_v1"
            ),
            "calibration_bag_feature_contract": (
                "class_intrinsic_top_fraction_and_pair_paired_patch_"
                "exclusivity_without_adjacency_v2"
            ),
            "calibration_inference_alignment": {
                "exact": False,
                "shared_statistic": (
                    "class_intrinsic_and_pair_paired_patch_top_fraction"
                ),
                "calibration_unit": "single_heldout_source_bag",
                "inference_unit": (
                    "paired_tight_context_views_with_conservative_spatial_"
                    "overlap_and_view_consistency_gates"
                ),
            },
            "known_limitations": [
                "thin_or_subpatch_targets_can_remain_underrepresented_when_"
                "the_upstream_target_mask_contains_too_few_object_cells",
                "source_reservoir_rows_do_not_preserve_anchor_grid_adjacency_"
                "so_coherence_remains_an_inference_only_gate",
                "heldout_exact_image_groups_do_not_guarantee_capture_or_site_"
                "independence",
            ],
            "digest": str(self.calibration_split_digest or ""),
            "heldout_source_count": int(
                self.calibration_heldout_source_count
            ),
            "fit_source_count": int(self.calibration_fit_source_count),
            "per_class": per_class,
        }

    def to_arrays(self) -> Dict[str, np.ndarray]:
        self.validate()
        pair_arrays = self._effective_pair_arrays()
        (
            pair_calibration_class_source_ids,
            pair_calibration_class_source_counts,
        ) = self._effective_pair_calibration_source_membership()
        return {
            "schema": np.asarray([self.schema]),
            "class_names": np.asarray(self.class_names),
            # Status boundaries are defined in cosine-margin space.  Float16
            # cache quantisation was enough to move borderline values across a
            # threshold, so cold and warm runs could disagree.  Preserve the
            # fitted float32 representation exactly.
            "prototypes": np.asarray(self.prototypes, dtype=np.float32),
            "prototype_counts": np.asarray(self.prototype_counts, dtype=np.int32),
            "prototype_source_ids": np.asarray(
                self.prototype_source_ids
            ),
            "background_prototypes": np.asarray(
                self.background_prototypes, dtype=np.float32
            ),
            "background_prototype_counts": np.asarray(
                self.background_prototype_counts, dtype=np.int32
            ),
            "background_prototype_source_ids": np.asarray(
                self.background_prototype_source_ids
            ),
            "anchor_counts": np.asarray(self.anchor_counts, dtype=np.int32),
            "distinct_source_counts": np.asarray(
                self.distinct_source_counts, dtype=np.int32
            ),
            "reliable": np.asarray(self.reliable, dtype=np.uint8),
            "reliability_tiers": np.asarray(self.reliability_tiers),
            "heldout_aurocs": np.asarray(
                self.heldout_aurocs, dtype=np.float32
            ),
            "support_thresholds": np.asarray(
                self.support_thresholds, dtype=np.float32
            ),
            "strong_support_thresholds": np.asarray(
                self.strong_support_thresholds, dtype=np.float32
            ),
            "negative_support_thresholds": np.asarray(
                self._effective_negative_support_thresholds(),
                dtype=np.float32,
            ),
            "pair_reliable": np.asarray(
                pair_arrays["pair_reliable"], dtype=np.uint8
            ),
            "pair_reliability_tiers": np.asarray(
                pair_arrays["pair_reliability_tiers"]
            ),
            "pair_heldout_aurocs": np.asarray(
                pair_arrays["pair_heldout_aurocs"], dtype=np.float32
            ),
            "pair_dominance_thresholds": np.asarray(
                pair_arrays["pair_dominance_thresholds"], dtype=np.float32
            ),
            "pair_current_negative_thresholds": np.asarray(
                pair_arrays["pair_current_negative_thresholds"],
                dtype=np.float32,
            ),
            "pair_current_presence_thresholds": np.asarray(
                pair_arrays["pair_current_presence_thresholds"],
                dtype=np.float32,
            ),
            "pair_current_strong_thresholds": np.asarray(
                pair_arrays["pair_current_strong_thresholds"],
                dtype=np.float32,
            ),
            "pair_alternative_negative_thresholds": np.asarray(
                pair_arrays["pair_alternative_negative_thresholds"],
                dtype=np.float32,
            ),
            "pair_alternative_presence_thresholds": np.asarray(
                pair_arrays["pair_alternative_presence_thresholds"],
                dtype=np.float32,
            ),
            "pair_alternative_strong_thresholds": np.asarray(
                pair_arrays["pair_alternative_strong_thresholds"],
                dtype=np.float32,
            ),
            "pair_current_source_counts": np.asarray(
                pair_arrays["pair_current_source_counts"], dtype=np.int32
            ),
            "pair_alternative_source_counts": np.asarray(
                pair_arrays["pair_alternative_source_counts"],
                dtype=np.int32,
            ),
            "pair_current_patch_counts": np.asarray(
                pair_arrays["pair_current_patch_counts"], dtype=np.int32
            ),
            "pair_alternative_patch_counts": np.asarray(
                pair_arrays["pair_alternative_patch_counts"],
                dtype=np.int32,
            ),
            "pair_alternative_passing_source_fractions": np.asarray(
                pair_arrays["pair_alternative_passing_source_fractions"],
                dtype=np.float32,
            ),
            "pair_probe_contract": np.asarray([self.pair_probe_contract]),
            "pair_probe_view_contract": np.asarray(
                [self.pair_probe_view_contract]
            ),
            "pair_probe_lower_bound_contract": np.asarray(
                [self.pair_probe_lower_bound_contract]
            ),
            "pair_probe_weights": np.asarray(
                pair_arrays["pair_probe_weights"], dtype=np.float32
            ),
            "pair_probe_thresholds": np.asarray(
                pair_arrays["pair_probe_thresholds"], dtype=np.float32
            ),
            "pair_probe_oof_aurocs": np.asarray(
                pair_arrays["pair_probe_oof_aurocs"], dtype=np.float32
            ),
            "pair_probe_fold_counts": np.asarray(
                pair_arrays["pair_probe_fold_counts"], dtype=np.int32
            ),
            "pair_probe_fit_statuses": np.asarray(
                pair_arrays["pair_probe_fit_statuses"]
            ),
            "pair_probe_fold_digests": np.asarray(
                pair_arrays["pair_probe_fold_digests"]
            ),
            "pair_probe_eval_auroc_lower_bounds": np.asarray(
                pair_arrays["pair_probe_eval_auroc_lower_bounds"],
                dtype=np.float32,
            ),
            "pair_probe_fit_current_source_counts": np.asarray(
                pair_arrays["pair_probe_fit_current_source_counts"],
                dtype=np.int32,
            ),
            "pair_probe_fit_alternative_source_counts": np.asarray(
                pair_arrays["pair_probe_fit_alternative_source_counts"],
                dtype=np.int32,
            ),
            "pair_probe_eval_current_source_counts": np.asarray(
                pair_arrays["pair_probe_eval_current_source_counts"],
                dtype=np.int32,
            ),
            "pair_probe_eval_alternative_source_counts": np.asarray(
                pair_arrays["pair_probe_eval_alternative_source_counts"],
                dtype=np.int32,
            ),
            "pair_probe_fit_balanced_accuracies": np.asarray(
                pair_arrays["pair_probe_fit_balanced_accuracies"],
                dtype=np.float32,
            ),
            "pair_probe_eval_sensitivities": np.asarray(
                pair_arrays["pair_probe_eval_sensitivities"],
                dtype=np.float32,
            ),
            "pair_probe_eval_specificities": np.asarray(
                pair_arrays["pair_probe_eval_specificities"],
                dtype=np.float32,
            ),
            "pair_current_absence_eval_fractions": np.asarray(
                pair_arrays["pair_current_absence_eval_fractions"],
                dtype=np.float32,
            ),
            "pair_alternative_strong_eval_fractions": np.asarray(
                pair_arrays["pair_alternative_strong_eval_fractions"],
                dtype=np.float32,
            ),
            "pair_probe_fit_eval_split_digests": np.asarray(
                pair_arrays["pair_probe_fit_eval_split_digests"]
            ),
            "pair_calibration_class_source_ids": np.asarray(
                pair_calibration_class_source_ids
            ),
            "pair_calibration_class_source_counts": np.asarray(
                pair_calibration_class_source_counts,
                dtype=np.int32,
            ),
            "projection_mean": np.asarray(
                self.projection_mean, dtype=np.float32
            ),
            "projection_components": np.asarray(
                self.projection_components, dtype=np.float32
            ),
            "calibration_status": np.asarray([self.calibration_status]),
            "calibration_split_digest": np.asarray(
                [self.calibration_split_digest]
            ),
            "calibration_heldout_source_count": np.asarray(
                [self.calibration_heldout_source_count], dtype=np.int32
            ),
            "calibration_fit_source_count": np.asarray(
                [self.calibration_fit_source_count], dtype=np.int32
            ),
            "calibration_target_patch_counts": np.asarray(
                self.calibration_target_patch_counts, dtype=np.int32
            ),
            "calibration_background_patch_counts": np.asarray(
                self.calibration_background_patch_counts, dtype=np.int32
            ),
            "calibration_target_source_counts": np.asarray(
                self.calibration_target_source_counts, dtype=np.int32
            ),
            "calibration_background_source_counts": np.asarray(
                self.calibration_background_source_counts, dtype=np.int32
            ),
            "calibration_target_passing_source_counts": np.asarray(
                self.calibration_target_passing_source_counts,
                dtype=np.int32,
            ),
            "calibration_target_source_pass_fractions": np.asarray(
                self.calibration_target_source_pass_fractions,
                dtype=np.float32,
            ),
            "fit_target_patch_counts": np.asarray(
                self.fit_target_patch_counts, dtype=np.int32
            ),
            "fit_background_patch_counts": np.asarray(
                self.fit_background_patch_counts, dtype=np.int32
            ),
            "fit_target_source_counts": np.asarray(
                self.fit_target_source_counts, dtype=np.int32
            ),
            "fit_background_source_counts": np.asarray(
                self.fit_background_source_counts, dtype=np.int32
            ),
            "calibration_heldout_source_ids": np.asarray(
                self.calibration_heldout_source_ids
            ),
            "calibration_fit_source_ids": np.asarray(
                self.calibration_fit_source_ids
            ),
        }

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, Any]) -> "ReferenceBank":
        required_calibration_provenance = (
            "calibration_split_digest",
            "calibration_heldout_source_count",
            "calibration_fit_source_count",
            "calibration_target_patch_counts",
            "calibration_background_patch_counts",
            "calibration_target_source_counts",
            "calibration_background_source_counts",
            "calibration_target_passing_source_counts",
            "calibration_target_source_pass_fractions",
            "fit_target_patch_counts",
            "fit_background_patch_counts",
            "fit_target_source_counts",
            "fit_background_source_counts",
            "negative_support_thresholds",
            "calibration_heldout_source_ids",
            "calibration_fit_source_ids",
            "pair_reliable",
            "pair_reliability_tiers",
            "pair_heldout_aurocs",
            "pair_dominance_thresholds",
            "pair_current_negative_thresholds",
            "pair_current_presence_thresholds",
            "pair_current_strong_thresholds",
            "pair_alternative_negative_thresholds",
            "pair_alternative_presence_thresholds",
            "pair_alternative_strong_thresholds",
            "pair_current_source_counts",
            "pair_alternative_source_counts",
            "pair_current_patch_counts",
            "pair_alternative_patch_counts",
            "pair_alternative_passing_source_fractions",
            "pair_probe_contract",
            "pair_probe_view_contract",
            "pair_probe_lower_bound_contract",
            "pair_probe_weights",
            "pair_probe_thresholds",
            "pair_probe_oof_aurocs",
            "pair_probe_fold_counts",
            "pair_probe_fit_statuses",
            "pair_probe_fold_digests",
            "pair_probe_eval_auroc_lower_bounds",
            "pair_probe_fit_current_source_counts",
            "pair_probe_fit_alternative_source_counts",
            "pair_probe_eval_current_source_counts",
            "pair_probe_eval_alternative_source_counts",
            "pair_probe_fit_balanced_accuracies",
            "pair_probe_eval_sensitivities",
            "pair_probe_eval_specificities",
            "pair_current_absence_eval_fractions",
            "pair_alternative_strong_eval_fractions",
            "pair_probe_fit_eval_split_digests",
            "pair_calibration_class_source_ids",
            "pair_calibration_class_source_counts",
        )
        if any(
            key not in arrays for key in required_calibration_provenance
        ):
            raise ValueError(
                "class_analysis_refinement_bank_calibration_provenance_invalid"
            )
        for storage_key in (
            "prototype_counts",
            "background_prototype_counts",
            "anchor_counts",
            "distinct_source_counts",
        ):
            if not np.issubdtype(
                np.asarray(arrays.get(storage_key)).dtype,
                np.integer,
            ):
                raise ValueError(
                    "class_analysis_refinement_bank_metadata_invalid"
                )
        for storage_key in (
            "calibration_heldout_source_count",
            "calibration_fit_source_count",
            "calibration_target_patch_counts",
            "calibration_background_patch_counts",
            "calibration_target_source_counts",
            "calibration_background_source_counts",
            "calibration_target_passing_source_counts",
            "fit_target_patch_counts",
            "fit_background_patch_counts",
            "fit_target_source_counts",
            "fit_background_source_counts",
            "pair_current_source_counts",
            "pair_alternative_source_counts",
            "pair_current_patch_counts",
            "pair_alternative_patch_counts",
            "pair_probe_fold_counts",
            "pair_probe_fit_current_source_counts",
            "pair_probe_fit_alternative_source_counts",
            "pair_probe_eval_current_source_counts",
            "pair_probe_eval_alternative_source_counts",
            "pair_calibration_class_source_counts",
        ):
            if not np.issubdtype(
                np.asarray(arrays.get(storage_key)).dtype,
                np.integer,
            ):
                raise ValueError(
                    "class_analysis_refinement_bank_calibration_provenance_invalid"
                )
        if not np.issubdtype(
            np.asarray(
                arrays.get("calibration_target_source_pass_fractions")
            ).dtype,
            np.floating,
        ):
            raise ValueError(
                "class_analysis_refinement_bank_calibration_provenance_invalid"
            )
        reliable_storage = np.asarray(arrays.get("reliable"))
        if (
            not np.issubdtype(reliable_storage.dtype, np.bool_)
            and not np.issubdtype(reliable_storage.dtype, np.integer)
        ) or np.any(
            (reliable_storage != 0) & (reliable_storage != 1)
        ):
            raise ValueError("class_analysis_refinement_bank_metadata_invalid")
        pair_reliable_storage = np.asarray(arrays.get("pair_reliable"))
        if (
            not np.issubdtype(pair_reliable_storage.dtype, np.bool_)
            and not np.issubdtype(pair_reliable_storage.dtype, np.integer)
        ) or np.any(
            (pair_reliable_storage != 0) & (pair_reliable_storage != 1)
        ):
            raise ValueError(
                "class_analysis_refinement_bank_pair_metadata_invalid"
            )
        for storage_key in (
            "prototypes",
            "background_prototypes",
            "projection_mean",
            "projection_components",
            "negative_support_thresholds",
            "pair_heldout_aurocs",
            "pair_dominance_thresholds",
            "pair_current_negative_thresholds",
            "pair_current_presence_thresholds",
            "pair_current_strong_thresholds",
            "pair_alternative_negative_thresholds",
            "pair_alternative_presence_thresholds",
            "pair_alternative_strong_thresholds",
            "pair_alternative_passing_source_fractions",
            "pair_probe_weights",
            "pair_probe_thresholds",
            "pair_probe_oof_aurocs",
            "pair_probe_eval_auroc_lower_bounds",
            "pair_probe_fit_balanced_accuracies",
            "pair_probe_eval_sensitivities",
            "pair_probe_eval_specificities",
            "pair_current_absence_eval_fractions",
            "pair_alternative_strong_eval_fractions",
        ):
            if np.asarray(arrays.get(storage_key)).dtype != np.float32:
                raise ValueError(
                    "class_analysis_refinement_bank_storage_dtype_invalid"
                )
        schema_raw = arrays.get("schema")
        schema_values = (
            np.asarray(schema_raw)
            if schema_raw is not None
            else np.asarray([])
        )
        schema = str(schema_values.reshape(-1)[0]) if schema_values.size else ""
        bank = cls(
            schema=schema,
            class_names=[
                str(value)
                for value in np.asarray(arrays["class_names"]).reshape(-1).tolist()
            ],
            prototypes=np.asarray(arrays["prototypes"], dtype=np.float32),
            prototype_counts=np.asarray(
                arrays["prototype_counts"], dtype=np.int32
            ),
            prototype_source_ids=np.asarray(
                arrays["prototype_source_ids"]
            ),
            background_prototypes=np.asarray(
                arrays["background_prototypes"], dtype=np.float32
            ),
            background_prototype_counts=np.asarray(
                arrays["background_prototype_counts"], dtype=np.int32
            ),
            background_prototype_source_ids=np.asarray(
                arrays["background_prototype_source_ids"]
            ),
            anchor_counts=np.asarray(arrays["anchor_counts"], dtype=np.int32),
            distinct_source_counts=np.asarray(
                arrays["distinct_source_counts"], dtype=np.int32
            ),
            reliable=np.asarray(arrays["reliable"], dtype=bool),
            reliability_tiers=np.asarray(arrays["reliability_tiers"]),
            heldout_aurocs=np.asarray(
                arrays["heldout_aurocs"], dtype=np.float32
            ),
            support_thresholds=np.asarray(
                arrays["support_thresholds"], dtype=np.float32
            ),
            strong_support_thresholds=np.asarray(
                arrays["strong_support_thresholds"], dtype=np.float32
            ),
            negative_support_thresholds=np.asarray(
                arrays["negative_support_thresholds"], dtype=np.float32
            ),
            pair_reliable=np.asarray(arrays["pair_reliable"], dtype=bool),
            pair_reliability_tiers=np.asarray(
                arrays["pair_reliability_tiers"]
            ),
            pair_heldout_aurocs=np.asarray(
                arrays["pair_heldout_aurocs"], dtype=np.float32
            ),
            pair_dominance_thresholds=np.asarray(
                arrays["pair_dominance_thresholds"], dtype=np.float32
            ),
            pair_current_negative_thresholds=np.asarray(
                arrays["pair_current_negative_thresholds"], dtype=np.float32
            ),
            pair_current_presence_thresholds=np.asarray(
                arrays["pair_current_presence_thresholds"], dtype=np.float32
            ),
            pair_current_strong_thresholds=np.asarray(
                arrays["pair_current_strong_thresholds"], dtype=np.float32
            ),
            pair_alternative_negative_thresholds=np.asarray(
                arrays["pair_alternative_negative_thresholds"], dtype=np.float32
            ),
            pair_alternative_presence_thresholds=np.asarray(
                arrays["pair_alternative_presence_thresholds"], dtype=np.float32
            ),
            pair_alternative_strong_thresholds=np.asarray(
                arrays["pair_alternative_strong_thresholds"], dtype=np.float32
            ),
            pair_current_source_counts=np.asarray(
                arrays["pair_current_source_counts"], dtype=np.int32
            ),
            pair_alternative_source_counts=np.asarray(
                arrays["pair_alternative_source_counts"], dtype=np.int32
            ),
            pair_current_patch_counts=np.asarray(
                arrays["pair_current_patch_counts"], dtype=np.int32
            ),
            pair_alternative_patch_counts=np.asarray(
                arrays["pair_alternative_patch_counts"], dtype=np.int32
            ),
            pair_alternative_passing_source_fractions=np.asarray(
                arrays["pair_alternative_passing_source_fractions"],
                dtype=np.float32,
            ),
            pair_probe_contract=str(
                np.asarray(arrays["pair_probe_contract"]).reshape(-1)[0]
            ),
            pair_probe_view_contract=str(
                np.asarray(arrays["pair_probe_view_contract"]).reshape(-1)[0]
            ),
            pair_probe_lower_bound_contract=str(
                np.asarray(arrays["pair_probe_lower_bound_contract"])
                .reshape(-1)[0]
            ),
            pair_probe_weights=np.asarray(
                arrays["pair_probe_weights"], dtype=np.float32
            ),
            pair_probe_thresholds=np.asarray(
                arrays["pair_probe_thresholds"], dtype=np.float32
            ),
            pair_probe_oof_aurocs=np.asarray(
                arrays["pair_probe_oof_aurocs"], dtype=np.float32
            ),
            pair_probe_fold_counts=np.asarray(
                arrays["pair_probe_fold_counts"], dtype=np.int32
            ),
            pair_probe_fit_statuses=np.asarray(
                arrays["pair_probe_fit_statuses"]
            ),
            pair_probe_fold_digests=np.asarray(
                arrays["pair_probe_fold_digests"]
            ),
            pair_probe_eval_auroc_lower_bounds=np.asarray(
                arrays["pair_probe_eval_auroc_lower_bounds"], dtype=np.float32
            ),
            pair_probe_fit_current_source_counts=np.asarray(
                arrays["pair_probe_fit_current_source_counts"], dtype=np.int32
            ),
            pair_probe_fit_alternative_source_counts=np.asarray(
                arrays["pair_probe_fit_alternative_source_counts"],
                dtype=np.int32,
            ),
            pair_probe_eval_current_source_counts=np.asarray(
                arrays["pair_probe_eval_current_source_counts"], dtype=np.int32
            ),
            pair_probe_eval_alternative_source_counts=np.asarray(
                arrays["pair_probe_eval_alternative_source_counts"],
                dtype=np.int32,
            ),
            pair_probe_fit_balanced_accuracies=np.asarray(
                arrays["pair_probe_fit_balanced_accuracies"], dtype=np.float32
            ),
            pair_probe_eval_sensitivities=np.asarray(
                arrays["pair_probe_eval_sensitivities"], dtype=np.float32
            ),
            pair_probe_eval_specificities=np.asarray(
                arrays["pair_probe_eval_specificities"], dtype=np.float32
            ),
            pair_current_absence_eval_fractions=np.asarray(
                arrays["pair_current_absence_eval_fractions"], dtype=np.float32
            ),
            pair_alternative_strong_eval_fractions=np.asarray(
                arrays["pair_alternative_strong_eval_fractions"],
                dtype=np.float32,
            ),
            pair_probe_fit_eval_split_digests=np.asarray(
                arrays["pair_probe_fit_eval_split_digests"]
            ),
            pair_calibration_class_source_ids=np.asarray(
                arrays["pair_calibration_class_source_ids"]
            ),
            pair_calibration_class_source_counts=np.asarray(
                arrays["pair_calibration_class_source_counts"],
                dtype=np.int32,
            ),
            projection_mean=np.asarray(
                arrays["projection_mean"], dtype=np.float32
            ),
            projection_components=np.asarray(
                arrays["projection_components"], dtype=np.float32
            ),
            calibration_status=str(
                np.asarray(arrays["calibration_status"]).reshape(-1)[0]
            ),
            calibration_split_digest=str(
                np.asarray(arrays["calibration_split_digest"])
                .reshape(-1)[0]
            ),
            calibration_heldout_source_count=int(
                np.asarray(arrays["calibration_heldout_source_count"])
                .reshape(-1)[0]
            ),
            calibration_fit_source_count=int(
                np.asarray(arrays["calibration_fit_source_count"])
                .reshape(-1)[0]
            ),
            calibration_target_patch_counts=np.asarray(
                arrays["calibration_target_patch_counts"], dtype=np.int32
            ),
            calibration_background_patch_counts=np.asarray(
                arrays["calibration_background_patch_counts"],
                dtype=np.int32,
            ),
            calibration_target_source_counts=np.asarray(
                arrays["calibration_target_source_counts"], dtype=np.int32
            ),
            calibration_background_source_counts=np.asarray(
                arrays["calibration_background_source_counts"],
                dtype=np.int32,
            ),
            calibration_target_passing_source_counts=np.asarray(
                arrays["calibration_target_passing_source_counts"],
                dtype=np.int32,
            ),
            calibration_target_source_pass_fractions=np.asarray(
                arrays["calibration_target_source_pass_fractions"],
                dtype=np.float32,
            ),
            fit_target_patch_counts=np.asarray(
                arrays["fit_target_patch_counts"], dtype=np.int32
            ),
            fit_background_patch_counts=np.asarray(
                arrays["fit_background_patch_counts"], dtype=np.int32
            ),
            fit_target_source_counts=np.asarray(
                arrays["fit_target_source_counts"], dtype=np.int32
            ),
            fit_background_source_counts=np.asarray(
                arrays["fit_background_source_counts"], dtype=np.int32
            ),
            calibration_heldout_source_ids=np.asarray(
                arrays["calibration_heldout_source_ids"]
            ),
            calibration_fit_source_ids=np.asarray(
                arrays["calibration_fit_source_ids"]
            ),
        )
        bank.validate()
        return bank


class StreamingReferenceBankBuilder:
    """Bounded deterministic reservoir for source-balanced class patches."""

    def __init__(self, config: RefinementConfig):
        config.validate()
        self.config = config
        self._rows: Dict[str, List[Tuple[int, str, np.ndarray]]] = {}
        self._background_rows: Dict[
            str, List[Tuple[int, str, np.ndarray]]
        ] = {}
        self._sources: Dict[str, set[str]] = {}
        self._anchor_counts: Dict[str, int] = {}
        self._dimension: Optional[int] = None

    @staticmethod
    def _rank_key(class_name: str, source_key: str, patch_index: int) -> int:
        digest = hashlib.sha256(
            f"{class_name}\0{source_key}\0{patch_index}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    @staticmethod
    def _coverage_rank(coverage: float, stable_rank: int) -> int:
        """Sort higher target coverage before the deterministic hash rank."""

        clipped = min(1.0, max(0.0, float(coverage)))
        coverage_priority = int(round((1.0 - clipped) * 1_000_000_000.0))
        return (coverage_priority << 64) | int(stable_rank)

    @staticmethod
    def _trim_source_balanced(
        rows: List[Tuple[int, str, np.ndarray]],
        *,
        limit: int,
    ) -> None:
        """Retain a deterministic round-robin sample across source images."""

        maximum = max(1, int(limit))
        by_source: Dict[str, List[Tuple[int, str, np.ndarray]]] = {}
        for row in rows:
            by_source.setdefault(row[1], []).append(row)
        for source_rows in by_source.values():
            source_rows.sort(key=lambda item: item[0])
        retained: List[Tuple[int, str, np.ndarray]] = []
        offset = 0
        while len(retained) < maximum:
            added = False
            for source_key in sorted(by_source):
                source_rows = by_source[source_key]
                if offset >= len(source_rows):
                    continue
                retained.append(source_rows[offset])
                added = True
                if len(retained) >= maximum:
                    break
            if not added:
                break
            offset += 1
        rows[:] = sorted(retained, key=lambda item: (item[0], item[1]))

    def add(
        self,
        *,
        class_name: str,
        source_key: str,
        patch_tokens: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        background_tokens: Optional[np.ndarray] = None,
        background_valid_mask: Optional[np.ndarray] = None,
    ) -> None:
        clean_class = str(class_name or "").strip()
        clean_source = str(source_key or "").strip()
        tokens = np.asarray(patch_tokens, dtype=np.float32)
        if not clean_class or not clean_source or tokens.ndim != 2 or not tokens.size:
            return
        if self._dimension is None:
            self._dimension = int(tokens.shape[1])
        if int(tokens.shape[1]) != self._dimension:
            raise ValueError("class_analysis_refinement_token_dimension_changed")
        tokens = _normalise_rows(tokens)
        valid = (
            np.asarray(valid_mask, dtype=np.float32).reshape(-1)
            if valid_mask is not None
            else np.ones(tokens.shape[0], dtype=np.float32)
        )
        if (
            valid.shape[0] != tokens.shape[0]
            or not np.all(np.isfinite(valid))
            or np.any(valid < 0.0)
            or np.any(valid > 1.0)
        ):
            raise ValueError("class_analysis_refinement_anchor_mask_invalid")
        indices = np.flatnonzero(valid > 0.0)
        if indices.size <= 0:
            return
        ranked = sorted(
            (
                -float(valid[int(index)]),
                self._rank_key(clean_class, clean_source, int(index)),
                int(index),
            )
            for index in indices.tolist()
        )
        ranked = ranked[: max(1, int(self.config.patches_per_anchor))]
        rows = self._rows.setdefault(clean_class, [])
        for negative_coverage, rank, patch_index in ranked:
            retained_rank = self._coverage_rank(-negative_coverage, rank)
            rows.append(
                (retained_rank, clean_source, tokens[patch_index].copy())
            )
        self._trim_source_balanced(
            rows,
            limit=self.config.patch_reservoir_per_class,
        )
        if background_tokens is not None:
            background = np.asarray(background_tokens, dtype=np.float32)
            if (
                background.ndim != 2
                or background.shape[1] != self._dimension
            ):
                raise ValueError(
                    "class_analysis_refinement_background_tokens_invalid"
                )
            background = _normalise_rows(background)
            background_valid = (
                np.asarray(background_valid_mask, dtype=bool).reshape(-1)
                if background_valid_mask is not None
                else np.ones(background.shape[0], dtype=bool)
            )
            if background_valid.shape[0] != background.shape[0]:
                raise ValueError(
                    "class_analysis_refinement_background_mask_invalid"
                )
            background_indices = np.flatnonzero(background_valid)
            background_ranked = sorted(
                (
                    self._rank_key(
                        clean_class,
                        f"background:{clean_source}",
                        int(index),
                    ),
                    int(index),
                )
                for index in background_indices.tolist()
            )[: max(1, int(self.config.patches_per_anchor))]
            background_rows = self._background_rows.setdefault(
                clean_class, []
            )
            for rank, patch_index in background_ranked:
                background_rows.append(
                    (rank, clean_source, background[patch_index].copy())
                )
            self._trim_source_balanced(
                background_rows,
                limit=self.config.patch_reservoir_per_class,
            )
        self._sources.setdefault(clean_class, set()).add(clean_source)
        self._anchor_counts[clean_class] = (
            int(self._anchor_counts.get(clean_class) or 0) + 1
        )

    def finalize(self) -> ReferenceBank:
        if self._dimension is None or not self._rows:
            raise ValueError("class_analysis_refinement_reference_bank_empty")
        try:
            from sklearn.cluster import MiniBatchKMeans
            from sklearn.decomposition import PCA
        except Exception as exc:
            raise ValueError(
                "class_analysis_refinement_reduction_unavailable"
            ) from exc

        class_names = sorted(self._rows)
        fit_rows: Dict[str, List[Tuple[int, str, np.ndarray]]] = {}
        calibration_rows: Dict[str, List[Tuple[int, str, np.ndarray]]] = {}
        fit_background: Dict[str, List[Tuple[int, str, np.ndarray]]] = {}
        calibration_background: Dict[
            str, List[Tuple[int, str, np.ndarray]]
        ] = {}
        heldout_sources = _global_heldout_sources(
            (
                row[1]
                for class_name in class_names
                for row in (
                    self._rows.get(class_name, [])
                    + self._background_rows.get(class_name, [])
                )
            ),
            source_groups=(
                tuple(row[1] for row in rows)
                for class_name in class_names
                for rows in (
                    self._rows.get(class_name, []),
                    self._background_rows.get(class_name, []),
                )
            ),
        )
        all_sources = {
            str(row[1])
            for class_name in class_names
            for row in (
                self._rows.get(class_name, [])
                + self._background_rows.get(class_name, [])
            )
            if str(row[1])
        }
        fit_sources = all_sources.difference(heldout_sources)
        calibration_heldout_source_ids = np.asarray(
            sorted(_source_fingerprint(source) for source in heldout_sources),
            dtype="<U16",
        )
        calibration_fit_source_ids = np.asarray(
            sorted(_source_fingerprint(source) for source in fit_sources),
            dtype="<U16",
        )
        calibration_split_digest = _calibration_source_split_digest(
            calibration_heldout_source_ids.tolist(),
            calibration_fit_source_ids.tolist(),
        )
        for class_name in class_names:
            fit_rows[class_name] = []
            calibration_rows[class_name] = []
            fit_background[class_name] = []
            calibration_background[class_name] = []
            for row in self._rows.get(class_name, []):
                (
                    calibration_rows[class_name]
                    if row[1] in heldout_sources
                    else fit_rows[class_name]
                ).append(row)
            for row in self._background_rows.get(class_name, []):
                (
                    calibration_background[class_name]
                    if row[1] in heldout_sources
                    else fit_background[class_name]
                ).append(row)
            # Never backfill either fit side from a held-out source. The fold is
            # global, so a source image reserved through one class cannot enter
            # PCA or a prototype through another class. A class missing either
            # fit side stays unreliable instead of publishing a nominally
            # held-out AUROC.

        def row_count_array(
            rows_by_class: Mapping[
                str, Sequence[Tuple[int, str, np.ndarray]]
            ],
        ) -> np.ndarray:
            return np.asarray(
                [len(rows_by_class[name]) for name in class_names],
                dtype=np.int32,
            )

        def source_count_array(
            rows_by_class: Mapping[
                str, Sequence[Tuple[int, str, np.ndarray]]
            ],
        ) -> np.ndarray:
            return np.asarray(
                [
                    len({str(row[1]) for row in rows_by_class[name]})
                    for name in class_names
                ],
                dtype=np.int32,
            )

        calibration_target_patch_counts = row_count_array(
            calibration_rows
        )
        calibration_background_patch_counts = row_count_array(
            calibration_background
        )
        calibration_target_source_counts = source_count_array(
            calibration_rows
        )
        calibration_background_source_counts = source_count_array(
            calibration_background
        )
        fit_target_patch_counts = row_count_array(fit_rows)
        fit_background_patch_counts = row_count_array(fit_background)
        fit_target_source_counts = source_count_array(fit_rows)
        fit_background_source_counts = source_count_array(fit_background)

        pca_pool = _round_robin_class_rows(
            {
                class_name: (
                    fit_rows[class_name] + fit_background[class_name]
                )
                for class_name in class_names
            },
            limit=4096,
        )
        pca_values = np.stack(
            [row[2] for row in pca_pool], axis=0
        ).astype(np.float32, copy=False)
        component_count = min(
            64,
            int(self._dimension),
            max(1, pca_values.shape[0] - 1),
        )
        if component_count < 2:
            raise ValueError("class_analysis_refinement_reduction_data_insufficient")
        pca = PCA(
            n_components=component_count,
            svd_solver="randomized",
            random_state=int(self.config.seed),
        )
        pca.fit(pca_values)
        projection_mean = np.asarray(pca.mean_, dtype=np.float32)
        projection_components = np.asarray(
            pca.components_, dtype=np.float32
        )

        def project_rows(
            rows: Sequence[Tuple[int, str, np.ndarray]],
        ) -> np.ndarray:
            if not rows:
                return np.empty((0, component_count), dtype=np.float32)
            values = np.stack([row[2] for row in rows], axis=0)
            projected = (
                values - projection_mean
            ) @ projection_components.T
            return _normalise_rows(projected)

        def reduce_projected(
            values: np.ndarray,
            source_keys: Sequence[str],
        ) -> Tuple[np.ndarray, np.ndarray]:
            values = np.asarray(values, dtype=np.float32)
            source_ids = np.asarray(
                [
                    hashlib.sha256(
                        str(source).encode("utf-8")
                    ).hexdigest()[:16]
                    for source in source_keys
                ],
                dtype="<U16",
            )
            maximum = max(1, int(self.config.prototypes_per_class))
            if (
                values.ndim != 2
                or values.shape[0] <= 0
                or source_ids.shape != (values.shape[0],)
            ):
                return (
                    np.empty((0, component_count), dtype=np.float32),
                    np.empty((0,), dtype="<U16"),
                )
            if values.shape[0] <= maximum:
                return values, source_ids
            cluster_count = min(maximum, values.shape[0])
            model = MiniBatchKMeans(
                n_clusters=cluster_count,
                random_state=int(self.config.seed),
                batch_size=min(1024, max(64, values.shape[0])),
                n_init=3,
                max_iter=100,
            )
            model.fit(values)
            centres = _normalise_rows(
                np.asarray(model.cluster_centers_, dtype=np.float32)
            )
            nearest_rows = _source_balanced_cluster_exemplar_indices(
                values,
                centres,
                np.asarray(model.labels_, dtype=np.int64),
                source_ids,
                limit=maximum,
            )
            return (
                values[nearest_rows].astype(np.float32, copy=False),
                source_ids[nearest_rows],
            )

        projected_fit_targets = {
            class_name: project_rows(fit_rows[class_name])
            for class_name in class_names
        }
        projected_fit_background = {
            class_name: project_rows(fit_background[class_name])
            for class_name in class_names
        }
        projected_fit_target_sources = {
            class_name: np.asarray(
                [row[1] for row in fit_rows[class_name]],
                dtype=object,
            )
            for class_name in class_names
        }
        projected_fit_background_sources = {
            class_name: np.asarray(
                [row[1] for row in fit_background[class_name]],
                dtype=object,
            )
            for class_name in class_names
        }

        def build_intrinsic_specific_fit_rows() -> Dict[
            str, List[Tuple[int, str, np.ndarray]]
        ]:
            """Keep target patches recurring beyond their own context ring.

            V3 never lets another class destructively filter this bank. Each
            query source is compared only with independent same-class sources
            and independent paired-background sources. Thin targets can still
            be under-represented when the upstream anchor mask contains too few
            object cells; calibration exposes that as low intrinsic coverage.
            """

            specific: Dict[str, List[Tuple[int, str, np.ndarray]]] = {}
            for class_name in class_names:
                rows = fit_rows[class_name]
                values = projected_fit_targets[class_name]
                keep = np.zeros(values.shape[0], dtype=bool)
                target_sources = projected_fit_target_sources[class_name]
                background_values = projected_fit_background[class_name]
                background_sources = projected_fit_background_sources[
                    class_name
                ]
                for source_key in sorted(set(target_sources.tolist())):
                    query_indices = np.flatnonzero(target_sources == source_key)
                    target_pool_mask = target_sources != source_key
                    background_pool_mask = background_sources != source_key
                    target_pool = values[target_pool_mask]
                    target_pool_sources = target_sources[target_pool_mask]
                    background_pool = background_values[background_pool_mask]
                    background_pool_sources = background_sources[
                        background_pool_mask
                    ]
                    if not (
                        _prototype_pool_is_source_independent(
                            target_pool, target_pool_sources
                        )
                        and _prototype_pool_is_source_independent(
                            background_pool, background_pool_sources
                        )
                    ):
                        continue
                    target_score = _mean_top_source_similarity(
                        values[query_indices],
                        target_pool,
                        target_pool_sources,
                    )
                    background_score = _mean_top_source_similarity(
                        values[query_indices],
                        background_pool,
                        background_pool_sources,
                    )
                    keep[query_indices] = (
                        target_score - background_score
                        >= float(self.config.weak_support_margin)
                    )
                specific[class_name] = [
                    row for index, row in enumerate(rows) if bool(keep[index])
                ]
            return specific

        def build_reduced_targets() -> Tuple[
            Dict[str, np.ndarray], Dict[str, np.ndarray]
        ]:
            specific = build_intrinsic_specific_fit_rows()
            target_values: Dict[str, np.ndarray] = {}
            target_sources: Dict[str, np.ndarray] = {}
            for class_name in class_names:
                target_values[class_name], target_sources[class_name] = (
                    reduce_projected(
                        project_rows(specific[class_name]),
                        [row[1] for row in specific[class_name]],
                    )
                )
            return target_values, target_sources

        reduced_background: Dict[str, np.ndarray] = {}
        reduced_background_sources: Dict[str, np.ndarray] = {}
        for class_name in class_names:
            (
                reduced_background[class_name],
                reduced_background_sources[class_name],
            ) = reduce_projected(
                projected_fit_background[class_name],
                [row[1] for row in fit_background[class_name]],
            )

        reduced_targets, reduced_target_sources = build_reduced_targets()
        maximum = max(
            1,
            min(
                int(self.config.prototypes_per_class),
                max(
                    max(
                        reduced_targets[name].shape[0],
                        reduced_background[name].shape[0],
                    )
                    for name in class_names
                ),
            ),
        )
        prototypes = np.zeros(
            (len(class_names), maximum, component_count), dtype=np.float32
        )
        prototype_source_ids = np.full(
            (len(class_names), maximum),
            "",
            dtype="<U16",
        )
        background_prototypes = np.zeros_like(prototypes)
        background_prototype_source_ids = np.full(
            (len(class_names), maximum),
            "",
            dtype="<U16",
        )
        counts = np.zeros(len(class_names), dtype=np.int32)
        background_counts = np.zeros(len(class_names), dtype=np.int32)
        anchor_counts = np.zeros(len(class_names), dtype=np.int32)
        sources = np.zeros(len(class_names), dtype=np.int32)
        reliable = np.zeros(len(class_names), dtype=bool)
        reliability_tiers = np.full(
            len(class_names), "low", dtype="<U8"
        )
        heldout_aurocs = np.zeros(len(class_names), dtype=np.float32)
        calibration_target_passing_source_counts = np.zeros(
            len(class_names),
            dtype=np.int32,
        )
        calibration_target_source_pass_fractions = np.zeros(
            len(class_names),
            dtype=np.float32,
        )
        support_thresholds = np.full(
            len(class_names), float(self.config.support_margin), dtype=np.float32
        )
        strong_thresholds = np.full(
            len(class_names),
            float(self.config.strong_support_margin),
            dtype=np.float32,
        )
        negative_thresholds = np.full(
            len(class_names),
            float(self.config.weak_support_margin),
            dtype=np.float32,
        )

        def projected_intrinsic_margin(
            values: np.ndarray,
            class_position: int,
            *,
            query_source_key: str,
        ) -> Optional[np.ndarray]:
            if values.size <= 0:
                return np.empty((0,), dtype=np.float32)
            class_name = class_names[class_position]
            target, target_sources = _exclude_source_prototypes(
                reduced_targets[class_name],
                reduced_target_sources[class_name],
                exclude_source_key=query_source_key,
            )
            background, background_sources = _exclude_source_prototypes(
                reduced_background[class_name],
                reduced_background_sources[class_name],
                exclude_source_key=query_source_key,
            )
            if not (
                _prototype_pool_is_source_independent(
                    target,
                    target_sources,
                )
                and _prototype_pool_is_source_independent(
                    background,
                    background_sources,
                )
            ):
                return None
            target_sim = _mean_top_source_similarity(
                values,
                target,
                target_sources,
            )
            background_sim = _mean_top_source_similarity(
                values,
                background,
                background_sources,
            )
            return (target_sim - background_sim).astype(
                np.float32,
                copy=False,
            )

        def projected_intrinsic_parts_for_rows(
            rows: Sequence[Tuple[int, str, np.ndarray]],
            class_position: int,
        ) -> List[Tuple[str, np.ndarray]]:
            if not rows:
                return []
            rows_by_source: Dict[
                str, List[Tuple[int, str, np.ndarray]]
            ] = {}
            for row in rows:
                rows_by_source.setdefault(str(row[1]), []).append(row)
            parts: List[Tuple[str, np.ndarray]] = []
            for source_key in sorted(rows_by_source):
                projected = project_rows(rows_by_source[source_key])
                margins = projected_intrinsic_margin(
                    projected,
                    class_position,
                    query_source_key=source_key,
                )
                if margins is not None and margins.size:
                    parts.append((source_key, margins))
            return parts

        def source_bag_scores(
            parts: Sequence[Tuple[str, np.ndarray]],
        ) -> Dict[str, float]:
            return {
                source_key: _source_bag_score(
                    margins,
                    selected_fraction=float(self.config.selected_fraction),
                )
                for source_key, margins in parts
                if np.asarray(margins).size
            }

        for position, class_name in enumerate(class_names):
            anchor_counts[position] = int(
                self._anchor_counts.get(class_name) or 0
            )
            sources[position] = len(self._sources.get(class_name) or set())

        def evaluate_calibration(
            position: int,
        ) -> Dict[str, Any]:
            """Calibrate intrinsic presence on held-out source-level bags."""

            class_name = class_names[position]
            target_values = reduced_targets[class_name]
            background_values = reduced_background[class_name]
            background_margin_parts = projected_intrinsic_parts_for_rows(
                calibration_background[class_name],
                position,
            )
            target_margin_parts = projected_intrinsic_parts_for_rows(
                calibration_rows[class_name],
                position,
            )
            target_bags = source_bag_scores(target_margin_parts)
            background_bags = source_bag_scores(background_margin_parts)
            target_source_scores = np.asarray(
                list(target_bags.values()), dtype=np.float32
            )
            background_source_scores = np.asarray(
                list(background_bags.values()), dtype=np.float32
            )
            heldout_target_source_count = len(target_bags)
            heldout_negative_source_count = len(background_bags)
            heldout_own_background_source_count = len(
                {str(row[1]) for row in calibration_background[class_name]}
            )
            support_threshold, heldout_auroc, negative_threshold = (
                _balanced_source_operating_point(
                    target_source_scores,
                    background_source_scores,
                    fallback=float(self.config.support_margin),
                )
            )
            # Presence and absence are mutually exclusive states.  A noisy
            # held-out background tail can otherwise put its p95 above the
            # fitted presence boundary, allowing one score to satisfy both
            # gates.  Keep the calibrated negative boundary, but cap it just
            # below the source-bag operating point.
            negative_threshold = min(
                float(negative_threshold),
                float(np.nextafter(support_threshold, -np.inf)),
            )
            if (
                target_source_scores.size < MIN_HELDOUT_SOURCE_GROUPS
                or background_source_scores.size
                < MIN_HELDOUT_SOURCE_GROUPS
            ):
                heldout_auroc = 0.0
            strong_threshold = support_threshold
            if target_source_scores.size:
                strong_threshold = max(
                    strong_threshold,
                    float(np.quantile(target_source_scores, 0.50)),
                )
            target_passing_source_count = int(
                np.count_nonzero(target_source_scores >= support_threshold)
            )
            target_source_pass_fraction = (
                float(target_passing_source_count)
                / float(target_source_scores.size)
                if target_source_scores.size
                else 0.0
            )
            base_calibrated = bool(
                target_values.shape[0] >= 4
                and background_values.shape[0] >= 4
                and _prototype_pool_is_source_independent(
                    target_values,
                    reduced_target_sources[class_name],
                )
                and _prototype_pool_is_source_independent(
                    background_values,
                    reduced_background_sources[class_name],
                )
                and heldout_target_source_count
                >= MIN_HELDOUT_SOURCE_GROUPS
                and heldout_negative_source_count
                >= MIN_HELDOUT_SOURCE_GROUPS
                and heldout_own_background_source_count
                >= MIN_HELDOUT_SOURCE_GROUPS
                and target_passing_source_count
                >= MIN_HELDOUT_SOURCE_GROUPS
                and target_source_pass_fraction
                >= MIN_HELDOUT_TARGET_SOURCE_PASS_FRACTION
                and len(set(reduced_target_sources[class_name].tolist()))
                >= MIN_RELIABLE_FIT_SOURCE_GROUPS
                and len(
                    set(reduced_background_sources[class_name].tolist())
                )
                >= MIN_RELIABLE_FIT_SOURCE_GROUPS
                and heldout_auroc >= 0.70
            )
            tier = "low"
            if (
                base_calibrated
                and anchor_counts[position] >= 64
                and sources[position]
                >= max(8, int(self.config.minimum_distinct_sources))
            ):
                tier = "high"
            elif (
                base_calibrated
                and anchor_counts[position] >= 24
                and sources[position]
                >= max(
                    MIN_RELIABLE_TOTAL_SOURCE_GROUPS,
                    int(self.config.minimum_distinct_sources),
                )
            ):
                tier = "usable"
            return {
                "support_threshold": support_threshold,
                "strong_threshold": strong_threshold,
                "negative_threshold": negative_threshold,
                "heldout_auroc": heldout_auroc,
                "heldout_target_source_count": heldout_target_source_count,
                "heldout_target_passing_source_count": (
                    target_passing_source_count
                ),
                "heldout_target_source_pass_fraction": (
                    target_source_pass_fraction
                ),
                "heldout_negative_source_count": (
                    heldout_negative_source_count
                ),
                "tier": tier,
            }

        # Intrinsic class reliability is independent: a weak or ambiguous class
        # cannot globally demote unrelated classes. Directed pair calibration
        # below decides whether a particular current->alternative comparison is
        # usable.
        calibration_results = {
            class_name: evaluate_calibration(position)
            for position, class_name in enumerate(class_names)
        }
        active_names = {
            class_name
            for class_name, result in calibration_results.items()
            if result["tier"] in {"high", "usable"}
        }

        for position, class_name in enumerate(class_names):
            target_values = reduced_targets[class_name][:maximum]
            background_values = reduced_background[class_name][:maximum]
            counts[position] = target_values.shape[0]
            background_counts[position] = background_values.shape[0]
            if target_values.size:
                prototypes[position, : target_values.shape[0]] = target_values
                prototype_source_ids[
                    position, : target_values.shape[0]
                ] = reduced_target_sources[class_name][
                    : target_values.shape[0]
                ]
            if background_values.size:
                background_prototypes[
                    position, : background_values.shape[0]
                ] = background_values
                background_prototype_source_ids[
                    position, : background_values.shape[0]
                ] = reduced_background_sources[class_name][
                    : background_values.shape[0]
                ]
            result = calibration_results[class_name]
            support_thresholds[position] = float(
                result["support_threshold"]
            )
            strong_thresholds[position] = float(result["strong_threshold"])
            negative_thresholds[position] = min(
                float(result["negative_threshold"]),
                float(
                    np.nextafter(
                        support_thresholds[position],
                        np.float32(-np.inf),
                    )
                ),
            )
            heldout_aurocs[position] = float(result["heldout_auroc"])
            calibration_target_passing_source_counts[position] = int(
                result["heldout_target_passing_source_count"]
            )
            calibration_target_source_pass_fractions[position] = float(
                result["heldout_target_source_pass_fraction"]
            )
            reliability_tiers[position] = (
                str(result["tier"])
                if class_name in active_names
                else "low"
            )
            reliable[position] = class_name in active_names

        pair_shape = (len(class_names), len(class_names))
        pair_reliable = np.zeros(pair_shape, dtype=bool)
        pair_tiers = np.full(pair_shape, "low", dtype="<U8")
        pair_aurocs = np.zeros(pair_shape, dtype=np.float32)
        pair_dominance_thresholds = np.zeros(
            pair_shape, dtype=np.float32
        )
        feasible_current_negative = np.minimum(
            negative_thresholds,
            np.nextafter(support_thresholds, np.float32(-np.inf)),
        )
        pair_current_negative_thresholds = np.broadcast_to(
            feasible_current_negative[:, None], pair_shape
        ).copy()
        pair_alternative_strong_thresholds = np.broadcast_to(
            strong_thresholds[None, :], pair_shape
        ).copy()
        pair_current_source_counts = np.zeros(pair_shape, dtype=np.int32)
        pair_alternative_source_counts = np.zeros(
            pair_shape, dtype=np.int32
        )
        pair_current_patch_counts = np.zeros(pair_shape, dtype=np.int32)
        pair_alternative_patch_counts = np.zeros(pair_shape, dtype=np.int32)
        pair_alternative_passing_source_fractions = np.zeros(
            pair_shape, dtype=np.float32
        )
        equal_probe_weights = np.asarray(
            [-math.sqrt(0.5), math.sqrt(0.5)], dtype=np.float32
        )
        pair_probe_weights = np.broadcast_to(
            equal_probe_weights,
            pair_shape + (2,),
        ).copy()
        pair_probe_thresholds = np.zeros(pair_shape, dtype=np.float32)
        pair_probe_oof_aurocs = np.zeros(pair_shape, dtype=np.float32)
        pair_probe_fold_counts = np.zeros(pair_shape, dtype=np.int32)
        pair_probe_fit_statuses = np.full(
            pair_shape, "not_fitted", dtype="<U24"
        )
        np.fill_diagonal(pair_probe_fit_statuses, "not_applicable")
        pair_probe_fold_digests = np.full(
            pair_shape, "", dtype="<U64"
        )
        pair_probe_eval_auroc_lower_bounds = np.zeros(
            pair_shape, dtype=np.float32
        )
        pair_probe_fit_current_source_counts = np.zeros(
            pair_shape, dtype=np.int32
        )
        pair_probe_fit_alternative_source_counts = np.zeros(
            pair_shape, dtype=np.int32
        )
        pair_probe_eval_current_source_counts = np.zeros(
            pair_shape, dtype=np.int32
        )
        pair_probe_eval_alternative_source_counts = np.zeros(
            pair_shape, dtype=np.int32
        )
        pair_probe_fit_eval_split_digests = np.full(
            pair_shape, "", dtype="<U64"
        )

        def intrinsic_parts_for_rows(
            rows: Sequence[Tuple[int, str, np.ndarray]],
            class_position: int,
        ) -> Dict[str, np.ndarray]:
            return {
                source_key: np.asarray(margins, dtype=np.float32)
                for source_key, margins in projected_intrinsic_parts_for_rows(
                    rows, class_position
                )
            }

        heldout_intrinsic_parts = {
            (row_class, score_position): intrinsic_parts_for_rows(
                calibration_rows[row_class], score_position
            )
            for row_class in class_names
            for score_position in range(len(class_names))
        }

        # V4 pair reliability is populated only by the streaming exact-view
        # calibrator after this intrinsic bank has been finalized.  The old
        # unordered tight-bag loop remains below as migration reference but is
        # deliberately unreachable so it cannot publish confirm-capable pairs.
        for current_position, current_name in enumerate(()):
            for alternative_position, alternative_name in enumerate(
                class_names
            ):
                if current_position == alternative_position:
                    continue
                current_on_current = heldout_intrinsic_parts[
                    (current_name, current_position)
                ]
                alternative_on_current = heldout_intrinsic_parts[
                    (current_name, alternative_position)
                ]
                alternative_on_alternative = heldout_intrinsic_parts[
                    (alternative_name, alternative_position)
                ]
                current_on_alternative = heldout_intrinsic_parts[
                    (alternative_name, current_position)
                ]
                current_sources_for_pair = sorted(
                    set(current_on_current).intersection(alternative_on_current)
                )
                alternative_sources_for_pair = sorted(
                    set(alternative_on_alternative).intersection(
                        current_on_alternative
                    )
                )
                current_features = np.asarray(
                    [
                        _paired_patch_exclusive_bag_features(
                            current_on_current[source],
                            alternative_on_current[source],
                            selected_fraction=float(
                                self.config.selected_fraction
                            ),
                        )
                        for source in current_sources_for_pair
                    ],
                    dtype=np.float32,
                )
                alternative_features = np.asarray(
                    [
                        _paired_patch_exclusive_bag_features(
                            current_on_alternative[source],
                            alternative_on_alternative[source],
                            selected_fraction=float(
                                self.config.selected_fraction
                            ),
                        )
                        for source in alternative_sources_for_pair
                    ],
                    dtype=np.float32,
                )
                probe = _fit_source_cross_fitted_pair_probe(
                    current_features,
                    current_sources_for_pair,
                    alternative_features,
                    alternative_sources_for_pair,
                )
                dominance_threshold = float(probe["threshold"])
                pair_auroc = float(probe["oof_auroc"])
                current_positive_scores = _source_bag_score_vector(
                    current_on_current,
                    current_sources_for_pair,
                    selected_fraction=float(
                        self.config.selected_fraction
                    ),
                )
                current_negative_scores = _source_bag_score_vector(
                    current_on_alternative,
                    alternative_sources_for_pair,
                    selected_fraction=float(
                        self.config.selected_fraction
                    ),
                )
                (
                    current_presence_threshold,
                    _current_presence_auroc,
                    current_negative_threshold,
                ) = _balanced_source_operating_point(
                    current_positive_scores,
                    current_negative_scores,
                    fallback=float(negative_thresholds[current_position]),
                )
                current_negative_threshold = min(
                    float(current_negative_threshold),
                    float(
                        np.nextafter(
                            min(
                                float(current_presence_threshold),
                                float(
                                    support_thresholds[current_position]
                                ),
                            ),
                            -np.inf,
                        )
                    ),
                )
                pass_fraction = float(
                    probe["oof_positive_pass_fraction"]
                )
                pair_current_source_counts[
                    current_position, alternative_position
                ] = len(current_sources_for_pair)
                pair_alternative_source_counts[
                    current_position, alternative_position
                ] = len(alternative_sources_for_pair)
                pair_current_patch_counts[
                    current_position, alternative_position
                ] = sum(
                    1
                    for row in calibration_rows[current_name]
                    if str(row[1]) in current_sources_for_pair
                )
                pair_alternative_patch_counts[
                    current_position, alternative_position
                ] = sum(
                    1
                    for row in calibration_rows[alternative_name]
                    if str(row[1]) in alternative_sources_for_pair
                )
                pair_aurocs[current_position, alternative_position] = float(
                    pair_auroc
                )
                pair_dominance_thresholds[
                    current_position, alternative_position
                ] = float(dominance_threshold)
                pair_current_negative_thresholds[
                    current_position, alternative_position
                ] = float(current_negative_threshold)
                pair_alternative_passing_source_fractions[
                    current_position, alternative_position
                ] = float(pass_fraction)
                pair_probe_weights[
                    current_position, alternative_position
                ] = np.asarray(probe["weights"], dtype=np.float32)
                pair_probe_thresholds[
                    current_position, alternative_position
                ] = float(dominance_threshold)
                pair_probe_oof_aurocs[
                    current_position, alternative_position
                ] = float(pair_auroc)
                pair_probe_fold_counts[
                    current_position, alternative_position
                ] = int(probe["fold_count"])
                pair_probe_fit_statuses[
                    current_position, alternative_position
                ] = str(probe["fit_status"])
                pair_probe_fold_digests[
                    current_position, alternative_position
                ] = str(probe["fold_digest"])
                pair_is_reliable = bool(
                    current_name in active_names
                    and alternative_name in active_names
                    and len(current_sources_for_pair)
                    >= MIN_HELDOUT_SOURCE_GROUPS
                    and len(alternative_sources_for_pair)
                    >= MIN_HELDOUT_SOURCE_GROUPS
                    and str(probe["fit_status"]) == "ok"
                    and int(probe["fold_count"]) >= 2
                    and pair_auroc >= 0.70
                    and pass_fraction
                    >= MIN_HELDOUT_TARGET_SOURCE_PASS_FRACTION
                )
                if not pair_is_reliable:
                    continue
                pair_reliable[current_position, alternative_position] = True
                high_pair = bool(
                    reliability_tiers[current_position] == "high"
                    and reliability_tiers[alternative_position] == "high"
                    and pair_auroc >= 0.85
                    and min(
                        len(current_sources_for_pair),
                        len(alternative_sources_for_pair),
                    )
                    >= 4
                )
                pair_tiers[current_position, alternative_position] = (
                    "high" if high_pair else "usable"
                )
        bank = ReferenceBank(
            class_names=class_names,
            prototypes=prototypes,
            prototype_counts=counts,
            prototype_source_ids=prototype_source_ids,
            background_prototypes=background_prototypes,
            background_prototype_counts=background_counts,
            background_prototype_source_ids=(
                background_prototype_source_ids
            ),
            anchor_counts=anchor_counts,
            distinct_source_counts=sources,
            reliable=reliable,
            reliability_tiers=reliability_tiers,
            heldout_aurocs=heldout_aurocs,
            support_thresholds=support_thresholds,
            strong_support_thresholds=strong_thresholds,
            negative_support_thresholds=negative_thresholds,
            pair_reliable=pair_reliable,
            pair_reliability_tiers=pair_tiers,
            pair_heldout_aurocs=pair_aurocs,
            pair_dominance_thresholds=pair_dominance_thresholds,
            pair_current_negative_thresholds=(
                pair_current_negative_thresholds
            ),
            pair_alternative_strong_thresholds=(
                pair_alternative_strong_thresholds
            ),
            pair_current_source_counts=pair_current_source_counts,
            pair_alternative_source_counts=pair_alternative_source_counts,
            pair_current_patch_counts=pair_current_patch_counts,
            pair_alternative_patch_counts=pair_alternative_patch_counts,
            pair_alternative_passing_source_fractions=(
                pair_alternative_passing_source_fractions
            ),
            pair_probe_contract=PAIR_PROBE_CONTRACT,
            pair_probe_view_contract=PAIR_PROBE_VIEW_CONTRACT,
            pair_probe_lower_bound_contract=(
                PAIR_PROBE_LOWER_BOUND_CONTRACT
            ),
            pair_probe_weights=pair_probe_weights,
            pair_probe_thresholds=pair_probe_thresholds,
            pair_probe_oof_aurocs=pair_probe_oof_aurocs,
            pair_probe_fold_counts=pair_probe_fold_counts,
            pair_probe_fit_statuses=pair_probe_fit_statuses,
            pair_probe_fold_digests=pair_probe_fold_digests,
            pair_probe_eval_auroc_lower_bounds=(
                pair_probe_eval_auroc_lower_bounds
            ),
            pair_probe_fit_current_source_counts=(
                pair_probe_fit_current_source_counts
            ),
            pair_probe_fit_alternative_source_counts=(
                pair_probe_fit_alternative_source_counts
            ),
            pair_probe_eval_current_source_counts=(
                pair_probe_eval_current_source_counts
            ),
            pair_probe_eval_alternative_source_counts=(
                pair_probe_eval_alternative_source_counts
            ),
            pair_probe_fit_eval_split_digests=(
                pair_probe_fit_eval_split_digests
            ),
            projection_mean=projection_mean,
            projection_components=projection_components,
            calibration_status=CALIBRATION_STATUS_SOURCE_AWARE,
            calibration_split_digest=calibration_split_digest,
            calibration_heldout_source_count=len(heldout_sources),
            calibration_fit_source_count=len(fit_sources),
            calibration_target_patch_counts=(
                calibration_target_patch_counts
            ),
            calibration_background_patch_counts=(
                calibration_background_patch_counts
            ),
            calibration_target_source_counts=(
                calibration_target_source_counts
            ),
            calibration_background_source_counts=(
                calibration_background_source_counts
            ),
            calibration_target_passing_source_counts=(
                calibration_target_passing_source_counts
            ),
            calibration_target_source_pass_fractions=(
                calibration_target_source_pass_fractions
            ),
            fit_target_patch_counts=fit_target_patch_counts,
            fit_background_patch_counts=fit_background_patch_counts,
            fit_target_source_counts=fit_target_source_counts,
            fit_background_source_counts=fit_background_source_counts,
            calibration_heldout_source_ids=calibration_heldout_source_ids,
            calibration_fit_source_ids=calibration_fit_source_ids,
        )
        bank.validate()
        return bank


class ExactTwoViewPairCalibrationBuilder:
    """Stream exact deployed-view examples into source-disjoint pair probes."""

    def __init__(self, bank: ReferenceBank, config: RefinementConfig):
        config.validate()
        bank.validate()
        self.bank = bank
        self.config = config
        self._rows: Dict[
            str, Dict[str, Dict[str, Any]]
        ] = {class_name: {} for class_name in bank.class_names}

    @staticmethod
    def _example_rank(
        class_name: str,
        source_id: str,
        anchor_id: str,
    ) -> str:
        return hashlib.sha256(
            f"{class_name}\0{source_id}\0{anchor_id}".encode("utf-8")
        ).hexdigest()

    def add_example(
        self,
        *,
        class_name: str,
        source_key: str,
        anchor_id: str,
        token_views: Sequence[np.ndarray],
        target_masks: Sequence[np.ndarray],
    ) -> bool:
        """Add one tight/context object and retain only reduced statistics.

        Views are ordered ``tight, context``.  A source can contribute at most
        one object per class; deterministic anchor rank makes batched call
        order irrelevant.  Only globally held-out sources are accepted.
        """

        clean_class = str(class_name or "").strip()
        clean_source = str(source_key or "").strip()
        clean_anchor = str(anchor_id or "").strip()
        if (
            clean_class not in self._rows
            or not clean_source
            or not clean_anchor
            or self.bank.calibration_source_role(clean_source) != "heldout"
            or len(token_views) != 2
            or len(target_masks) != 2
        ):
            return False
        source_id = _source_fingerprint(clean_source)
        rank = self._example_rank(clean_class, source_id, clean_anchor)
        previous = self._rows[clean_class].get(source_id)
        if previous is not None and str(previous["rank"]) <= rank:
            return False

        clean_tokens: List[np.ndarray] = []
        clean_masks: List[np.ndarray] = []
        for raw_tokens, raw_mask in zip(token_views, target_masks):
            tokens = np.asarray(raw_tokens, dtype=np.float32)
            mask = np.asarray(raw_mask, dtype=np.float32).reshape(-1)
            if (
                tokens.ndim != 2
                or tokens.shape[1] != self.bank.projection_mean.shape[0]
                or tokens.shape[0] != mask.shape[0]
                or tokens.shape[0] <= 0
                or not np.all(np.isfinite(tokens))
                or not np.all(np.isfinite(mask))
                or np.any(mask < 0.0)
                or np.any(mask > 1.0)
                or float(mask.sum()) <= 1e-12
            ):
                raise ValueError(
                    "class_analysis_pair_probe_exact_view_contract_invalid"
                )
            clean_tokens.append(tokens)
            clean_masks.append(mask)

        heatmaps_by_view = [
            all_class_margin_heatmaps(
                tokens,
                self.bank,
                query_source_key=clean_source,
            )
            for tokens in clean_tokens
        ]
        class_count = len(self.bank.class_names)
        supports = np.zeros((class_count, 2), dtype=np.float32)
        for class_position, scored_class in enumerate(self.bank.class_names):
            for view_index in range(2):
                supports[class_position, view_index] = _support_from_heat(
                    heatmaps_by_view[view_index][scored_class],
                    clean_masks[view_index],
                    positive_margin=self.bank.class_support_threshold(
                        scored_class
                    ),
                    selected_fraction=self.config.selected_fraction,
                )[0]
        pair_features = np.zeros(
            (class_count, class_count, 2), dtype=np.float32
        )
        for current_position, current_name in enumerate(self.bank.class_names):
            for alternative_position, alternative_name in enumerate(
                self.bank.class_names
            ):
                if current_position == alternative_position:
                    continue
                pair_features[current_position, alternative_position] = (
                    exact_two_view_pair_features(
                        [
                            heatmaps[current_name]
                            for heatmaps in heatmaps_by_view
                        ],
                        [
                            heatmaps[alternative_name]
                            for heatmaps in heatmaps_by_view
                        ],
                        clean_masks,
                        selected_fraction=self.config.selected_fraction,
                    )
                )
        self._rows[clean_class][source_id] = {
            "rank": rank,
            "supports": supports,
            "pair_features": pair_features,
        }
        return True

    @staticmethod
    def _split_digest(
        pair_name: str,
        rows: Sequence[Tuple[str, int]],
        assignments: np.ndarray,
    ) -> str:
        payload = "\n".join(
            [PAIR_PROBE_CONTRACT, PAIR_PROBE_VIEW_CONTRACT, pair_name]
            + [
                f"{source_id}:{label}:{'eval' if int(role) == 0 else 'fit'}"
                for (source_id, label), role in sorted(
                    zip(rows, assignments.tolist()),
                    key=lambda item: (item[0][0], item[0][1]),
                )
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def finalize(self) -> ReferenceBank:
        arrays = {
            name: np.asarray(value).copy()
            for name, value in self.bank._uncalibrated_pair_arrays().items()
        }
        class_count = len(self.bank.class_names)
        calibration_sources_by_class = [
            sorted(self._rows.get(class_name, {}))
            for class_name in self.bank.class_names
        ]
        calibration_source_width = max(
            (len(source_ids) for source_ids in calibration_sources_by_class),
            default=0,
        )
        pair_calibration_class_source_ids = np.full(
            (class_count, calibration_source_width),
            "",
            dtype="<U16",
        )
        pair_calibration_class_source_counts = np.zeros(
            class_count,
            dtype=np.int32,
        )
        for class_position, source_ids in enumerate(
            calibration_sources_by_class
        ):
            pair_calibration_class_source_counts[class_position] = len(
                source_ids
            )
            if source_ids:
                pair_calibration_class_source_ids[
                    class_position, : len(source_ids)
                ] = np.asarray(source_ids, dtype="<U16")
        for current_position, current_name in enumerate(self.bank.class_names):
            for alternative_position, alternative_name in enumerate(
                self.bank.class_names
            ):
                if current_position == alternative_position:
                    continue
                current_rows = self._rows.get(current_name, {})
                alternative_rows = self._rows.get(alternative_name, {})
                rows: List[Tuple[str, int]] = [
                    (source_id, 0) for source_id in sorted(current_rows)
                ] + [
                    (source_id, 1) for source_id in sorted(alternative_rows)
                ]
                sources = [row[0] for row in rows]
                labels = np.asarray([row[1] for row in rows], dtype=np.int8)
                assignments, fold_count, _fold_digest = (
                    _stable_source_group_folds(
                        sources,
                        labels,
                        maximum_folds=2,
                    )
                )
                if fold_count != 2:
                    arrays["pair_probe_fit_statuses"][
                        current_position, alternative_position
                    ] = "insufficient_sources"
                    continue
                eval_mask = assignments == 0
                fit_mask = assignments != 0
                fit_current_count = int(
                    np.count_nonzero(fit_mask & (labels == 0))
                )
                fit_alternative_count = int(
                    np.count_nonzero(fit_mask & (labels == 1))
                )
                eval_current_count = int(
                    np.count_nonzero(eval_mask & (labels == 0))
                )
                eval_alternative_count = int(
                    np.count_nonzero(eval_mask & (labels == 1))
                )
                for field_name, value in (
                    ("pair_probe_fit_current_source_counts", fit_current_count),
                    (
                        "pair_probe_fit_alternative_source_counts",
                        fit_alternative_count,
                    ),
                    ("pair_probe_eval_current_source_counts", eval_current_count),
                    (
                        "pair_probe_eval_alternative_source_counts",
                        eval_alternative_count,
                    ),
                ):
                    arrays[field_name][current_position, alternative_position] = value
                if (
                    fit_current_count < MIN_PAIR_PROBE_FIT_SOURCES_PER_CLASS
                    or fit_alternative_count
                    < MIN_PAIR_PROBE_FIT_SOURCES_PER_CLASS
                    or eval_current_count
                    < MIN_PAIR_PROBE_EVAL_SOURCES_PER_CLASS
                    or eval_alternative_count
                    < MIN_PAIR_PROBE_EVAL_SOURCES_PER_CLASS
                ):
                    arrays["pair_probe_fit_statuses"][
                        current_position, alternative_position
                    ] = "insufficient_sources"
                    continue

                feature_rows = np.stack(
                    [
                        (
                            current_rows[source_id]
                            if label == 0
                            else alternative_rows[source_id]
                        )["pair_features"][
                            current_position, alternative_position
                        ]
                        for source_id, label in rows
                    ],
                    axis=0,
                ).astype(np.float32, copy=False)
                weights = _select_sign_constrained_pair_weights(
                    feature_rows[fit_mask], labels[fit_mask]
                )
                if weights is None:
                    arrays["pair_probe_fit_statuses"][
                        current_position, alternative_position
                    ] = "fold_invalid"
                    continue
                fit_scores = feature_rows[fit_mask] @ weights
                fit_labels = labels[fit_mask]
                eval_scores = feature_rows[eval_mask] @ weights
                eval_labels = labels[eval_mask]
                threshold, fit_auroc, _unused = _balanced_source_operating_point(
                    fit_scores[fit_labels == 1],
                    fit_scores[fit_labels == 0],
                    fallback=0.0,
                )
                # Evaluate exactly the scalar that will be persisted and used at
                # inference. A float64 midpoint can otherwise classify a tied
                # float32 score differently before and after a cache roundtrip.
                threshold = _as_finite_float32(threshold)
                # The eval partition is a one-shot measurement set.  Neither
                # the probe direction nor any deployed operating threshold may
                # look at it.  This keeps the persisted eval metrics honest.
                auroc = _binary_auroc(
                    eval_scores[eval_labels == 1],
                    eval_scores[eval_labels == 0],
                )
                lower_bound = _conservative_auroc_lower_bound(
                    auroc,
                    eval_alternative_count,
                    eval_current_count,
                )
                fit_sensitivity = float(
                    np.mean(
                        fit_scores[fit_labels == 1] >= float(threshold)
                    )
                )
                fit_specificity = float(
                    np.mean(
                        fit_scores[fit_labels == 0] < float(threshold)
                    )
                )
                fit_balanced_accuracy = float(
                    0.5 * (fit_sensitivity + fit_specificity)
                )
                eval_sensitivity = float(
                    np.mean(
                        eval_scores[eval_labels == 1] >= float(threshold)
                    )
                )
                eval_specificity = float(
                    np.mean(
                        eval_scores[eval_labels == 0] < float(threshold)
                    )
                )
                pass_fraction = eval_sensitivity

                support_rows = np.stack(
                    [
                        (
                            current_rows[source_id]
                            if label == 0
                            else alternative_rows[source_id]
                        )["supports"]
                        for source_id, label in rows
                    ],
                    axis=0,
                ).astype(np.float32, copy=False)
                # Match every fit/eval statistic to its deployed view rule.
                # Presence and current strength are tight-view-first; absence is
                # conservative across both views. Alternative presence is the
                # two-view mean, strong evidence must survive both views, and
                # absence is again conservative across both views.
                current_presence_stat = support_rows[:, current_position, 0]
                current_strong_stat = current_presence_stat
                current_absence_stat = np.max(
                    support_rows[:, current_position, :], axis=1
                )
                alternative_presence_stat = np.mean(
                    support_rows[:, alternative_position, :], axis=1
                )
                alternative_strong_stat = np.min(
                    support_rows[:, alternative_position, :], axis=1
                )
                alternative_absence_stat = np.max(
                    support_rows[:, alternative_position, :], axis=1
                )
                (
                    current_presence_threshold,
                    _current_auroc,
                    _unused_current_negative,
                ) = _balanced_source_operating_point(
                    current_presence_stat[fit_mask & (labels == 0)],
                    current_presence_stat[fit_mask & (labels == 1)],
                    fallback=0.0,
                )
                current_presence_threshold = _as_finite_float32(
                    current_presence_threshold
                )
                current_negative_threshold = _as_finite_float32(
                    np.quantile(
                        current_absence_stat[fit_mask & (labels == 1)],
                        0.95,
                    )
                )
                current_negative_threshold = min(
                    current_negative_threshold,
                    _float32_strictly_below(current_presence_threshold),
                )
                current_negative_threshold = _as_finite_float32(
                    current_negative_threshold
                )
                current_strong_threshold = _as_finite_float32(
                    max(
                        current_presence_threshold,
                        float(
                            np.quantile(
                                current_strong_stat[
                                    fit_mask & (labels == 0)
                                ],
                                0.50,
                            )
                        ),
                    )
                )
                (
                    alternative_presence_threshold,
                    _alternative_presence_auroc,
                    _unused_alternative_negative,
                ) = (
                    _balanced_source_operating_point(
                        alternative_presence_stat[fit_mask & (labels == 1)],
                        alternative_presence_stat[fit_mask & (labels == 0)],
                        fallback=0.0,
                    )
                )
                alternative_presence_threshold = _as_finite_float32(
                    alternative_presence_threshold
                )
                alternative_negative_threshold = _as_finite_float32(
                    np.quantile(
                        alternative_absence_stat[
                            fit_mask & (labels == 0)
                        ],
                        0.95,
                    )
                )
                alternative_negative_threshold = min(
                    alternative_negative_threshold,
                    _float32_strictly_below(
                        alternative_presence_threshold
                    ),
                )
                alternative_negative_threshold = _as_finite_float32(
                    alternative_negative_threshold
                )
                (
                    alternative_strong_operating_threshold,
                    _alternative_strong_auroc,
                    _unused_alternative_strong_negative,
                ) = _balanced_source_operating_point(
                    alternative_strong_stat[fit_mask & (labels == 1)],
                    alternative_strong_stat[fit_mask & (labels == 0)],
                    fallback=0.0,
                )
                alternative_strong_threshold = _as_finite_float32(
                    max(
                        alternative_presence_threshold,
                        float(alternative_strong_operating_threshold),
                        float(
                            np.quantile(
                                alternative_strong_stat[
                                    fit_mask & (labels == 1)
                                ],
                                0.50,
                            )
                        ),
                    )
                )
                current_absence_eval_fraction = float(
                    np.mean(
                        current_absence_stat[eval_mask & (labels == 1)]
                        <= float(current_negative_threshold)
                    )
                )
                alternative_strong_eval_fraction = float(
                    np.mean(
                        alternative_strong_stat[eval_mask & (labels == 1)]
                        >= float(alternative_strong_threshold)
                    )
                )
                split_digest = self._split_digest(
                    f"{current_name}->{alternative_name}", rows, assignments
                )
                arrays["pair_probe_weights"][
                    current_position, alternative_position
                ] = weights
                arrays["pair_probe_thresholds"][
                    current_position, alternative_position
                ] = float(threshold)
                arrays["pair_dominance_thresholds"][
                    current_position, alternative_position
                ] = float(threshold)
                arrays["pair_probe_oof_aurocs"][
                    current_position, alternative_position
                ] = float(auroc)
                arrays["pair_heldout_aurocs"][
                    current_position, alternative_position
                ] = float(auroc)
                arrays["pair_probe_eval_auroc_lower_bounds"][
                    current_position, alternative_position
                ] = float(lower_bound)
                arrays["pair_probe_fit_balanced_accuracies"][
                    current_position, alternative_position
                ] = fit_balanced_accuracy
                arrays["pair_probe_eval_sensitivities"][
                    current_position, alternative_position
                ] = eval_sensitivity
                arrays["pair_probe_eval_specificities"][
                    current_position, alternative_position
                ] = eval_specificity
                arrays["pair_current_absence_eval_fractions"][
                    current_position, alternative_position
                ] = current_absence_eval_fraction
                arrays["pair_alternative_strong_eval_fractions"][
                    current_position, alternative_position
                ] = alternative_strong_eval_fraction
                arrays["pair_current_negative_thresholds"][
                    current_position, alternative_position
                ] = float(current_negative_threshold)
                arrays["pair_current_presence_thresholds"][
                    current_position, alternative_position
                ] = float(current_presence_threshold)
                arrays["pair_current_strong_thresholds"][
                    current_position, alternative_position
                ] = float(current_strong_threshold)
                arrays["pair_alternative_negative_thresholds"][
                    current_position, alternative_position
                ] = float(alternative_negative_threshold)
                arrays["pair_alternative_presence_thresholds"][
                    current_position, alternative_position
                ] = float(alternative_presence_threshold)
                arrays["pair_alternative_strong_thresholds"][
                    current_position, alternative_position
                ] = float(alternative_strong_threshold)
                arrays["pair_alternative_passing_source_fractions"][
                    current_position, alternative_position
                ] = pass_fraction
                arrays["pair_current_source_counts"][
                    current_position, alternative_position
                ] = eval_current_count
                arrays["pair_alternative_source_counts"][
                    current_position, alternative_position
                ] = eval_alternative_count
                arrays["pair_current_patch_counts"][
                    current_position, alternative_position
                ] = eval_current_count
                arrays["pair_alternative_patch_counts"][
                    current_position, alternative_position
                ] = eval_alternative_count
                arrays["pair_probe_fold_counts"][
                    current_position, alternative_position
                ] = 1
                arrays["pair_probe_fit_statuses"][
                    current_position, alternative_position
                ] = "ok"
                arrays["pair_probe_fold_digests"][
                    current_position, alternative_position
                ] = split_digest
                arrays["pair_probe_fit_eval_split_digests"][
                    current_position, alternative_position
                ] = split_digest
                pair_reliable = pair_metrics_are_reliable(
                    current_class_reliable=self.bank.class_is_reliable(
                        current_name
                    ),
                    alternative_class_reliable=self.bank.class_is_reliable(
                        alternative_name
                    ),
                    fit_current_source_count=fit_current_count,
                    fit_alternative_source_count=fit_alternative_count,
                    eval_current_source_count=eval_current_count,
                    eval_alternative_source_count=eval_alternative_count,
                    eval_auroc=auroc,
                    eval_auroc_lower_bound=lower_bound,
                    fit_balanced_accuracy=fit_balanced_accuracy,
                    eval_sensitivity=eval_sensitivity,
                    eval_specificity=eval_specificity,
                    current_absence_eval_fraction=(
                        current_absence_eval_fraction
                    ),
                    alternative_strong_eval_fraction=(
                        alternative_strong_eval_fraction
                    ),
                )
                arrays["pair_reliable"][
                    current_position, alternative_position
                ] = pair_reliable
                arrays["pair_reliability_tiers"][
                    current_position, alternative_position
                ] = (
                    "high"
                    if pair_reliable
                    and auroc >= 0.90
                    and lower_bound >= 0.70
                    and min(
                        fit_current_count,
                        fit_alternative_count,
                        eval_current_count,
                        eval_alternative_count,
                    )
                    >= 12
                    else "usable" if pair_reliable else "low"
                )

        calibrated = replace(
            self.bank,
            pair_reliable=np.asarray(arrays["pair_reliable"], dtype=bool),
            pair_reliability_tiers=np.asarray(
                arrays["pair_reliability_tiers"]
            ),
            pair_heldout_aurocs=np.asarray(
                arrays["pair_heldout_aurocs"], dtype=np.float32
            ),
            pair_dominance_thresholds=np.asarray(
                arrays["pair_dominance_thresholds"], dtype=np.float32
            ),
            pair_current_negative_thresholds=np.asarray(
                arrays["pair_current_negative_thresholds"], dtype=np.float32
            ),
            pair_current_presence_thresholds=np.asarray(
                arrays["pair_current_presence_thresholds"], dtype=np.float32
            ),
            pair_current_strong_thresholds=np.asarray(
                arrays["pair_current_strong_thresholds"], dtype=np.float32
            ),
            pair_alternative_negative_thresholds=np.asarray(
                arrays["pair_alternative_negative_thresholds"], dtype=np.float32
            ),
            pair_alternative_presence_thresholds=np.asarray(
                arrays["pair_alternative_presence_thresholds"], dtype=np.float32
            ),
            pair_alternative_strong_thresholds=np.asarray(
                arrays["pair_alternative_strong_thresholds"], dtype=np.float32
            ),
            pair_current_source_counts=np.asarray(
                arrays["pair_current_source_counts"], dtype=np.int32
            ),
            pair_alternative_source_counts=np.asarray(
                arrays["pair_alternative_source_counts"], dtype=np.int32
            ),
            pair_current_patch_counts=np.asarray(
                arrays["pair_current_patch_counts"], dtype=np.int32
            ),
            pair_alternative_patch_counts=np.asarray(
                arrays["pair_alternative_patch_counts"], dtype=np.int32
            ),
            pair_alternative_passing_source_fractions=np.asarray(
                arrays["pair_alternative_passing_source_fractions"],
                dtype=np.float32,
            ),
            pair_probe_weights=np.asarray(
                arrays["pair_probe_weights"], dtype=np.float32
            ),
            pair_probe_thresholds=np.asarray(
                arrays["pair_probe_thresholds"], dtype=np.float32
            ),
            pair_probe_oof_aurocs=np.asarray(
                arrays["pair_probe_oof_aurocs"], dtype=np.float32
            ),
            pair_probe_fold_counts=np.asarray(
                arrays["pair_probe_fold_counts"], dtype=np.int32
            ),
            pair_probe_fit_statuses=np.asarray(
                arrays["pair_probe_fit_statuses"]
            ),
            pair_probe_fold_digests=np.asarray(
                arrays["pair_probe_fold_digests"]
            ),
            pair_probe_eval_auroc_lower_bounds=np.asarray(
                arrays["pair_probe_eval_auroc_lower_bounds"], dtype=np.float32
            ),
            pair_probe_fit_current_source_counts=np.asarray(
                arrays["pair_probe_fit_current_source_counts"], dtype=np.int32
            ),
            pair_probe_fit_alternative_source_counts=np.asarray(
                arrays["pair_probe_fit_alternative_source_counts"],
                dtype=np.int32,
            ),
            pair_probe_eval_current_source_counts=np.asarray(
                arrays["pair_probe_eval_current_source_counts"], dtype=np.int32
            ),
            pair_probe_eval_alternative_source_counts=np.asarray(
                arrays["pair_probe_eval_alternative_source_counts"],
                dtype=np.int32,
            ),
            pair_probe_fit_balanced_accuracies=np.asarray(
                arrays["pair_probe_fit_balanced_accuracies"], dtype=np.float32
            ),
            pair_probe_eval_sensitivities=np.asarray(
                arrays["pair_probe_eval_sensitivities"], dtype=np.float32
            ),
            pair_probe_eval_specificities=np.asarray(
                arrays["pair_probe_eval_specificities"], dtype=np.float32
            ),
            pair_current_absence_eval_fractions=np.asarray(
                arrays["pair_current_absence_eval_fractions"], dtype=np.float32
            ),
            pair_alternative_strong_eval_fractions=np.asarray(
                arrays["pair_alternative_strong_eval_fractions"],
                dtype=np.float32,
            ),
            pair_probe_fit_eval_split_digests=np.asarray(
                arrays["pair_probe_fit_eval_split_digests"]
            ),
            pair_calibration_class_source_ids=(
                pair_calibration_class_source_ids
            ),
            pair_calibration_class_source_counts=(
                pair_calibration_class_source_counts
            ),
        )
        calibrated.validate()
        return calibrated


def _class_similarity_projected(
    projected: np.ndarray,
    bank: ReferenceBank,
    class_name: str,
    *,
    query_source_key: str = "",
) -> np.ndarray:
    prototypes, source_ids = bank._class_reference_pool(
        class_name,
        background=False,
        exclude_source_key=query_source_key,
    )
    return _mean_top_source_similarity(projected, prototypes, source_ids)


def _background_similarity_projected(
    projected: np.ndarray,
    bank: ReferenceBank,
    class_name: str,
    *,
    query_source_key: str = "",
) -> np.ndarray:
    prototypes, source_ids = bank._class_reference_pool(
        class_name,
        background=True,
        exclude_source_key=query_source_key,
    )
    return _mean_top_source_similarity(projected, prototypes, source_ids)


def _class_is_eligible_competitor(
    bank: ReferenceBank,
    class_name: str,
    *,
    query_source_key: str,
) -> bool:
    """Require calibrated reliability and both distinct-source margin sides.

    Source identity here is exact image SHA-256 (or split plus relative path),
    not an assertion that nearby frames are statistically independent.
    """

    return bool(
        bank.class_is_reliable(class_name)
        and bank.class_has_source_independent_support(
            class_name,
            exclude_source_key=query_source_key,
        )
    )


def all_class_margin_heatmaps(
    tokens: np.ndarray,
    bank: ReferenceBank,
    *,
    query_source_key: str = "",
) -> Dict[str, np.ndarray]:
    """Project once and derive intrinsic class-vs-own-background maps."""

    projected = bank.project_tokens(tokens)
    target_scores = {
        class_name: _class_similarity_projected(
            projected,
            bank,
            class_name,
            query_source_key=query_source_key,
        )
        for class_name in bank.class_names
    }
    background_scores = {
        class_name: _background_similarity_projected(
            projected,
            bank,
            class_name,
            query_source_key=query_source_key,
        )
        for class_name in bank.class_names
    }
    return {
        class_name: (
            target_scores[class_name] - background_scores[class_name]
        ).astype(np.float32, copy=False)
        for class_name in bank.class_names
    }


def class_margin_heatmap(
    tokens: np.ndarray,
    bank: ReferenceBank,
    class_name: str,
    *,
    query_source_key: str = "",
) -> np.ndarray:
    projected = bank.project_tokens(tokens)
    target = _class_similarity_projected(
        projected,
        bank,
        class_name,
        query_source_key=query_source_key,
    )
    background = _background_similarity_projected(
        projected,
        bank,
        class_name,
        query_source_key=query_source_key,
    )
    return (target - background).astype(np.float32, copy=False)


def _support_from_heat(
    heat: np.ndarray,
    valid: np.ndarray,
    *,
    positive_margin: float,
    selected_fraction: float = 0.05,
) -> Tuple[float, float]:
    values = np.asarray(heat, dtype=np.float32).reshape(-1)
    weights = np.asarray(valid, dtype=np.float32).reshape(-1)
    weights = np.clip(weights, 0.0, 1.0)
    eligible = weights > 0.0
    valid_values = values[eligible]
    if valid_values.size <= 0:
        return -1.0, 0.0
    eligible_weights = weights[eligible]
    # Selection is an exact fraction of *target mass*, including a fractional
    # boundary cell.  This is deliberately the same primitive used by exact
    # two-view calibration: a thin target touching many cells must not select
    # many times more evidence at inference than it did during calibration.
    top_mean = _weighted_top_fraction_score(
        valid_values,
        eligible_weights,
        selected_fraction=selected_fraction,
    )
    coverage = float(
        np.sum(
            eligible_weights
            * (valid_values >= float(positive_margin)).astype(np.float32)
        )
        / max(1e-12, float(eligible_weights.sum()))
    )
    # The score remains a cosine-margin evidence value, not a probability.
    return top_mean, coverage


def _largest_positive_component(
    heat: np.ndarray,
    valid: np.ndarray,
    *,
    grid_shape: Tuple[int, int],
    threshold: float,
) -> Tuple[float, int]:
    target_weights = np.clip(
        np.asarray(valid, dtype=np.float32).reshape(grid_shape),
        0.0,
        1.0,
    )
    # Fractional target coverage controls evidence mass, not the margin
    # threshold.  Multiplying before thresholding double-penalised thin
    # targets (for example a 0.20 margin in a quarter-covered patch became
    # 0.05 and disappeared below a 0.08 threshold).
    positive = (
        np.asarray(heat, dtype=np.float32).reshape(grid_shape)
        >= float(threshold)
    ) & (target_weights > 0.0)
    positive_count = int(positive.sum())
    valid_mass = max(1e-12, float(target_weights.sum()))
    if positive_count <= 0:
        return 0.0, 0
    seen = np.zeros(positive.shape, dtype=bool)
    largest = 0
    largest_mass = 0.0
    for row, col in np.argwhere(positive):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        stack = [(row, col)]
        seen[row, col] = True
        component = 0
        component_mass = 0.0
        while stack:
            current_row, current_col = stack.pop()
            component += 1
            component_mass += float(
                target_weights[current_row, current_col]
            )
            for next_row, next_col in (
                (current_row - 1, current_col),
                (current_row + 1, current_col),
                (current_row, current_col - 1),
                (current_row, current_col + 1),
                (current_row - 1, current_col - 1),
                (current_row - 1, current_col + 1),
                (current_row + 1, current_col - 1),
                (current_row + 1, current_col + 1),
            ):
                if (
                    0 <= next_row < positive.shape[0]
                    and 0 <= next_col < positive.shape[1]
                    and positive[next_row, next_col]
                    and not seen[next_row, next_col]
                ):
                    seen[next_row, next_col] = True
                    stack.append((next_row, next_col))
        largest = max(largest, component)
        largest_mass = max(largest_mass, component_mass)
    return float(largest_mass / valid_mass), int(largest)


def _grid_source_cell_boxes(
    crop_xyxy: Sequence[float],
    grid_shape: Tuple[int, int],
) -> np.ndarray:
    """Return each valid canonical patch cell's clipped source-space box."""

    x1, y1, x2, y2 = [float(value) for value in list(crop_xyxy)[:4]]
    crop_width = max(1e-6, x2 - x1)
    crop_height = max(1e-6, y2 - y1)
    side = max(crop_width, crop_height)
    offset_x = float(math.floor((side - crop_width) / 2.0))
    offset_y = float(math.floor((side - crop_height) / 2.0))
    grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
    cell_width = side / max(1, grid_w)
    cell_height = side / max(1, grid_h)
    boxes = np.full((grid_h, grid_w, 4), np.nan, dtype=np.float32)
    for row in range(grid_h):
        square_y1 = max(row * cell_height, offset_y)
        square_y2 = min((row + 1) * cell_height, offset_y + crop_height)
        if square_y2 <= square_y1:
            continue
        for col in range(grid_w):
            square_x1 = max(col * cell_width, offset_x)
            square_x2 = min((col + 1) * cell_width, offset_x + crop_width)
            if square_x2 <= square_x1:
                continue
            boxes[row, col] = [
                x1 + square_x1 - offset_x,
                y1 + square_y1 - offset_y,
                x1 + square_x2 - offset_x,
                y1 + square_y2 - offset_y,
            ]
    return boxes


def _largest_component_geometry(
    heat: np.ndarray,
    target_weights: np.ndarray,
    *,
    crop_xyxy: Sequence[float],
    target_bbox: Sequence[float],
    grid_shape: Tuple[int, int],
    threshold: float,
    competitor_heat: Optional[np.ndarray] = None,
    exclusive_margin: float = 0.0,
) -> Dict[str, Any]:
    """Return target-mass and source geometry for the largest valid component."""

    weights = np.clip(
        np.asarray(target_weights, dtype=np.float32).reshape(grid_shape),
        0.0,
        1.0,
    )
    values = np.asarray(heat, dtype=np.float32).reshape(grid_shape)
    positive = (values >= float(threshold)) & (weights > 0.0)
    if competitor_heat is not None:
        competitor = np.asarray(
            competitor_heat, dtype=np.float32
        ).reshape(grid_shape)
        positive &= values - competitor >= float(exclusive_margin)
    target_mass = max(1e-12, float(weights.sum()))
    cell_boxes = _grid_source_cell_boxes(crop_xyxy, grid_shape)
    tx1, ty1, tx2, ty2 = [
        float(value) for value in list(target_bbox)[:4]
    ]
    seen = np.zeros(positive.shape, dtype=bool)
    best: Dict[str, Any] = {
        "cell_count": 0,
        "mass": 0.0,
        "mass_fraction": 0.0,
        "source_bbox": None,
        "source_centroid": None,
    }
    for row, col in np.argwhere(positive):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        stack = [(row, col)]
        seen[row, col] = True
        members: List[Tuple[int, int]] = []
        while stack:
            current_row, current_col = stack.pop()
            members.append((current_row, current_col))
            for next_row, next_col in (
                (current_row - 1, current_col),
                (current_row + 1, current_col),
                (current_row, current_col - 1),
                (current_row, current_col + 1),
                (current_row - 1, current_col - 1),
                (current_row - 1, current_col + 1),
                (current_row + 1, current_col - 1),
                (current_row + 1, current_col + 1),
            ):
                if (
                    0 <= next_row < positive.shape[0]
                    and 0 <= next_col < positive.shape[1]
                    and positive[next_row, next_col]
                    and not seen[next_row, next_col]
                ):
                    seen[next_row, next_col] = True
                    stack.append((next_row, next_col))
        component_mass = float(sum(weights[r, c] for r, c in members))
        clipped_boxes: List[Tuple[float, float, float, float, float]] = []
        for member_row, member_col in members:
            raw_box = cell_boxes[member_row, member_col]
            if not np.all(np.isfinite(raw_box)):
                continue
            bx1, by1, bx2, by2 = [float(value) for value in raw_box]
            ix1, iy1 = max(tx1, bx1), max(ty1, by1)
            ix2, iy2 = min(tx2, bx2), min(ty2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            clipped_boxes.append(
                (ix1, iy1, ix2, iy2, float(weights[member_row, member_col]))
            )
        if not clipped_boxes:
            continue
        geometry_mass = max(
            1e-12, float(sum(item[4] for item in clipped_boxes))
        )
        centroid_x = sum(
            0.5 * (item[0] + item[2]) * item[4] for item in clipped_boxes
        ) / geometry_mass
        centroid_y = sum(
            0.5 * (item[1] + item[3]) * item[4] for item in clipped_boxes
        ) / geometry_mass
        candidate = {
            "cell_count": len(members),
            "mass": component_mass,
            "mass_fraction": float(component_mass / target_mass),
            "source_bbox": [
                min(item[0] for item in clipped_boxes),
                min(item[1] for item in clipped_boxes),
                max(item[2] for item in clipped_boxes),
                max(item[3] for item in clipped_boxes),
            ],
            "source_centroid": [float(centroid_x), float(centroid_y)],
        }
        if (
            float(candidate["mass"]) > float(best["mass"])
            or (
                float(candidate["mass"]) == float(best["mass"])
                and int(candidate["cell_count"]) > int(best["cell_count"])
            )
        ):
            best = candidate
    return best


def _component_is_coherent(
    component: Mapping[str, Any],
    config: RefinementConfig,
) -> bool:
    return bool(
        int(component.get("cell_count") or 0)
        >= int(config.minimum_component_cells)
        and float(component.get("mass_fraction") or 0.0)
        >= float(config.minimum_component_mass_fraction)
        and component.get("source_centroid") is not None
    )


def _component_corresponds(
    tight: Mapping[str, Any],
    context: Mapping[str, Any],
    target_bbox: Sequence[float],
    config: RefinementConfig,
) -> bool:
    if not (
        _component_is_coherent(tight, config)
        and _component_is_coherent(context, config)
    ):
        return False
    tight_centroid = list(tight.get("source_centroid") or [])
    context_centroid = list(context.get("source_centroid") or [])
    if len(tight_centroid) < 2 or len(context_centroid) < 2:
        return False
    tx1, ty1, tx2, ty2 = [float(value) for value in list(target_bbox)[:4]]
    width = max(1e-6, tx2 - tx1)
    height = max(1e-6, ty2 - ty1)
    per_axis_distance = max(
        abs(float(tight_centroid[0]) - float(context_centroid[0])) / width,
        abs(float(tight_centroid[1]) - float(context_centroid[1])) / height,
    )
    return bool(
        per_axis_distance
        <= float(config.component_correspondence_distance_fraction)
    )


def _components_are_spatially_separated(
    current_components: Sequence[Mapping[str, Any]],
    alternative_components: Sequence[Mapping[str, Any]],
    target_bbox: Sequence[float],
    config: RefinementConfig,
) -> bool:
    if len(current_components) != 2 or len(alternative_components) != 2:
        return False
    tx1, ty1, tx2, ty2 = [float(value) for value in list(target_bbox)[:4]]
    width = max(1e-6, tx2 - tx1)
    height = max(1e-6, ty2 - ty1)
    distances: List[float] = []
    for current, alternative in zip(current_components, alternative_components):
        current_centroid = list(current.get("source_centroid") or [])
        alternative_centroid = list(alternative.get("source_centroid") or [])
        if len(current_centroid) < 2 or len(alternative_centroid) < 2:
            return False
        distances.append(
            max(
                abs(
                    float(current_centroid[0])
                    - float(alternative_centroid[0])
                )
                / width,
                abs(
                    float(current_centroid[1])
                    - float(alternative_centroid[1])
                )
                / height,
            )
        )
    return bool(
        min(distances)
        >= float(config.component_separation_distance_fraction)
    )


def strongest_alternative_class(
    token_views: Sequence[np.ndarray],
    valid_views: Sequence[np.ndarray],
    *,
    current_class: str,
    bank: ReferenceBank,
    config: RefinementConfig,
    query_source_key: str = "",
) -> str:
    best: Tuple[float, str] = (-float("inf"), "")
    heatmaps_by_view = [
        all_class_margin_heatmaps(
            tokens,
            bank,
            query_source_key=query_source_key,
        )
        for tokens in token_views
    ]
    for class_name in bank.class_names:
        if (
            class_name == current_class
            or not bank.class_is_reliable(class_name)
            or not bank.class_has_source_independent_support(
                class_name, exclude_source_key=query_source_key
            )
        ):
            continue
        scores = []
        for heatmaps, valid in zip(heatmaps_by_view, valid_views):
            heat = heatmaps[class_name]
            support, _coverage = _support_from_heat(
                heat,
                valid,
                positive_margin=config.support_margin,
                selected_fraction=config.selected_fraction,
            )
            scores.append(support)
        score = float(np.mean(scores)) if scores else -1.0
        candidate = (score, class_name)
        if candidate > best:
            best = candidate
    return best[1]


def score_candidate(
    *,
    point_id: str,
    current_class: str,
    alternative_class: str,
    token_views: Sequence[np.ndarray],
    crop_boxes: Sequence[Sequence[float]],
    target_bbox: Sequence[float],
    grid_shape: Tuple[int, int],
    alternative_overlap_boxes: Sequence[Sequence[float]],
    overlap_boxes_by_class: Optional[
        Mapping[str, Sequence[Sequence[float]]]
    ] = None,
    bank: ReferenceBank,
    config: RefinementConfig,
    query_source_key: str = "",
) -> Dict[str, Any]:
    """Score one candidate and return compact evidence plus private heatmaps."""

    config.validate()
    bank.validate()
    if len(token_views) != 2 or len(crop_boxes) != 2:
        raise ValueError("class_analysis_refinement_candidate_views_invalid")
    current_class = str(current_class or "").strip()
    alternative_class = str(alternative_class or "").strip()
    query_source_key = str(query_source_key or "").strip()
    expected_tokens = int(grid_shape[0]) * int(grid_shape[1])
    valid_views: List[np.ndarray] = []
    target_views: List[np.ndarray] = []
    for crop_box, tokens in zip(crop_boxes, token_views):
        if np.asarray(tokens).shape[0] != expected_tokens:
            raise ValueError("class_analysis_refinement_candidate_grid_mismatch")
        _empty, valid = rasterize_overlap_centres(
            crop_box, grid_shape, []
        )
        target, _target_valid = rasterize_box_fractions(
            crop_box,
            grid_shape,
            [target_bbox],
            supersample=4,
        )
        target = np.minimum(target, valid)
        valid_views.append(valid.reshape(-1))
        target_views.append(target.reshape(-1))
    heatmaps_by_view = [
        all_class_margin_heatmaps(
            tokens,
            bank,
            query_source_key=query_source_key,
        )
        for tokens in token_views
    ]
    if not alternative_class:
        best_alternative: Tuple[float, str] = (-float("inf"), "")
        for class_name in bank.class_names:
            if (
                class_name == current_class
                or not bank.class_is_reliable(class_name)
                or not bank.class_has_source_independent_support(
                    class_name, exclude_source_key=query_source_key
                )
            ):
                continue
            supports = [
                _support_from_heat(
                    heatmaps[class_name],
                    target,
                    positive_margin=bank.class_support_threshold(
                        class_name
                    ),
                    selected_fraction=config.selected_fraction,
                )[0]
                for heatmaps, target in zip(
                    heatmaps_by_view,
                    target_views,
                )
            ]
            candidate = (
                float(np.mean(supports)) if supports else -1.0,
                class_name,
            )
            if candidate > best_alternative:
                best_alternative = candidate
        alternative_class = best_alternative[1]

    # Geometry ownership takes precedence over appearance scoring.  In a
    # selected-class run Stage 1 sees only the selected class, so a
    # near-identical cross-class box can first become visible here through the
    # full spatial-context index.  Choose that box's class as the diagnostic
    # alternative and retain the candidate for explicit pair resolution even
    # when its reference bank is unavailable.  The orchestration layer adds
    # the matching object identity; this service-level check keeps the status
    # contract fail-safe for direct callers as well.
    duplicate_overlap_choice: Optional[
        Tuple[float, float, str]
    ] = None
    if overlap_boxes_by_class is not None:
        for overlap_class, overlap_boxes in overlap_boxes_by_class.items():
            clean_overlap_class = str(overlap_class or "").strip()
            if (
                not clean_overlap_class
                or clean_overlap_class == current_class
            ):
                continue
            for overlap_box in overlap_boxes or []:
                geometry = bbox_overlap_geometry(target_bbox, overlap_box)
                if (
                    geometry is None
                    or str(geometry.get("relation") or "")
                    != "duplicate_like"
                ):
                    continue
                choice = (
                    float(geometry.get("iou") or 0.0),
                    float(geometry.get("target_area_covered") or 0.0),
                    clean_overlap_class,
                )
                if (
                    duplicate_overlap_choice is None
                    or choice > duplicate_overlap_choice
                ):
                    duplicate_overlap_choice = choice
    if duplicate_overlap_choice is not None:
        alternative_class = duplicate_overlap_choice[2]
    resolved_overlap_boxes = list(alternative_overlap_boxes or [])
    if overlap_boxes_by_class is not None and alternative_class:
        resolved_overlap_boxes = list(
            overlap_boxes_by_class.get(alternative_class) or []
        )
    overlap_geometry_rows = [
        (geometry, [float(value) for value in list(overlap_box)[:4]])
        for overlap_box in resolved_overlap_boxes
        for geometry in [bbox_overlap_geometry(target_bbox, overlap_box)]
        if geometry is not None
    ]
    overlap_geometries = [row[0] for row in overlap_geometry_rows]
    dominant_overlap = (
        max(
            overlap_geometries,
            key=overlap_annotation_selection_key,
        )
        if overlap_geometries
        else {}
    )
    overlap_relation = str(dominant_overlap.get("relation") or "none")
    annotated_overlap_alternative_bbox_xyxy = next(
        (
            bbox
            for geometry, bbox in overlap_geometry_rows
            if geometry is dominant_overlap
        ),
        None,
    )
    intersection_boxes = [
        list(geometry.get("intersection_xyxy") or [])
        for geometry in overlap_geometries
        if len(list(geometry.get("intersection_xyxy") or [])) >= 4
    ]
    overlap_views: List[np.ndarray] = []
    for crop_box, target in zip(crop_boxes, target_views):
        intersection_fraction = rasterize_box_fractions(
            crop_box,
            grid_shape,
            intersection_boxes,
            supersample=4,
        )[0].reshape(-1)
        # Evidence mass is already weighted by the fraction of each patch that
        # belongs to the target box.  Overlap must therefore be conditional on
        # that target mass, not another unconditional whole-cell fraction.
        # Using explicit geometric intersections also prevents disjoint boxes
        # that merely share a coarse patch cell from explaining each other.
        conditional_overlap = np.divide(
            intersection_fraction,
            target,
            out=np.zeros_like(intersection_fraction, dtype=np.float32),
            where=np.asarray(target, dtype=np.float32) > 1e-12,
        )
        overlap_views.append(
            np.clip(conditional_overlap, 0.0, 1.0).astype(
                np.float32,
                copy=False,
            )
        )

    current_bank_reliable = bank.class_is_reliable(current_class)
    alternative_bank_reliable = bool(
        alternative_class and bank.class_is_reliable(alternative_class)
    )
    current_source_independent = bank.class_has_source_independent_support(
        current_class,
        exclude_source_key=query_source_key,
    )
    alternative_source_independent = bool(
        alternative_class
        and bank.class_has_source_independent_support(
            alternative_class,
            exclude_source_key=query_source_key,
        )
    )
    current_reliable = bool(
        current_bank_reliable and current_source_independent
    )
    alternative_reliable = bool(
        alternative_bank_reliable and alternative_source_independent
    )
    pair_metadata = bank.directed_pair_metadata(
        current_class,
        alternative_class,
        query_source_key=query_source_key,
    )
    directed_pair_candidate_source_independent = not bool(
        pair_metadata["candidate_source_excluded"]
    )
    directed_pair_reliable = bool(
        pair_metadata["reliable"]
        and current_source_independent
        and alternative_source_independent
    )
    directed_pair_exact_calibration_contracts = bool(
        str(pair_metadata["probe_contract"]) == PAIR_PROBE_CONTRACT
        and str(pair_metadata["probe_view_contract"])
        == PAIR_PROBE_VIEW_CONTRACT
        and str(pair_metadata["probe_lower_bound_contract"])
        == PAIR_PROBE_LOWER_BOUND_CONTRACT
        and str(pair_metadata["probe_fit_status"]) == "ok"
        and int(pair_metadata["probe_fold_count"]) == 1
        and str(pair_metadata["probe_fold_digest"])
        == str(pair_metadata["fit_eval_split_digest"])
        and len(str(pair_metadata["probe_fold_digest"])) == 64
    )
    positive_confirmation_pair_probe_auroc_sufficient = bool(
        float(pair_metadata["probe_oof_auroc"])
        >= MIN_POSITIVE_CONFIRMATION_PAIR_PROBE_AUROC
    )
    positive_confirmation_pair_probe_lower_bound_sufficient = bool(
        float(pair_metadata["probe_eval_auroc_lower_bound"])
        >= MIN_POSITIVE_CONFIRMATION_PAIR_PROBE_AUROC_LOWER_BOUND
    )
    intrinsic_references_reliable = bool(
        current_reliable and alternative_reliable
    )
    diagnostic_pair_reliable = bool(
        intrinsic_references_reliable
        and pair_metadata["diagnostic_reliable"]
        and current_source_independent
        and alternative_source_independent
        and directed_pair_exact_calibration_contracts
    )
    positive_confirmation_pair_reliable = bool(
        intrinsic_references_reliable
        and directed_pair_reliable
        and directed_pair_exact_calibration_contracts
        and positive_confirmation_pair_probe_auroc_sufficient
        and positive_confirmation_pair_probe_lower_bound_sufficient
    )
    directed_pair_fit_thresholds_active = bool(
        directed_pair_reliable
        and directed_pair_exact_calibration_contracts
    )
    support_threshold_source = (
        "fit_only_directed_pair"
        if directed_pair_fit_thresholds_active
        else "intrinsic_fallback"
    )
    current_heats: List[np.ndarray] = []
    alternative_heats: List[np.ndarray] = []
    current_supports: List[float] = []
    alternative_supports: List[float] = []
    if directed_pair_fit_thresholds_active:
        current_threshold = float(
            pair_metadata["current_presence_threshold"]
        )
        current_strong_threshold = float(
            pair_metadata["current_strong_threshold"]
        )
        alternative_threshold = float(
            pair_metadata["alternative_presence_threshold"]
        )
        alternative_strong_threshold = float(
            pair_metadata["alternative_strong_threshold"]
        )
        alternative_negative_threshold = float(
            pair_metadata["alternative_negative_threshold"]
        )
        current_negative_threshold = float(
            pair_metadata["current_negative_threshold"]
        )
    else:
        current_threshold = bank.class_support_threshold(current_class)
        current_strong_threshold = bank.class_strong_support_threshold(
            current_class
        )
        current_negative_threshold = bank.class_negative_support_threshold(
            current_class
        )
        alternative_threshold = bank.class_support_threshold(
            alternative_class
        )
        alternative_strong_threshold = bank.class_strong_support_threshold(
            alternative_class
        )
        alternative_negative_threshold = bank.class_negative_support_threshold(
            alternative_class
        )
    directed_pair_threshold = float(pair_metadata["probe_threshold"])
    directed_pair_probe_weights = np.asarray(
        pair_metadata["probe_weights"], dtype=np.float32
    )
    current_intrinsic_components: List[Dict[str, Any]] = []
    alternative_intrinsic_components: List[Dict[str, Any]] = []
    current_exclusive_components: List[Dict[str, Any]] = []
    alternative_exclusive_components: List[Dict[str, Any]] = []
    current_outside_exclusive_components: List[Dict[str, Any]] = []
    alternative_outside_exclusive_components: List[Dict[str, Any]] = []
    for heatmaps, target, crop_box, overlap in zip(
        heatmaps_by_view, target_views, crop_boxes, overlap_views
    ):
        current_heat = heatmaps.get(
            current_class,
            np.full(expected_tokens, -1.0, dtype=np.float32),
        )
        alternative_heat = (
            heatmaps.get(
                alternative_class,
                np.full(expected_tokens, -1.0, dtype=np.float32),
            )
            if alternative_class
            else np.full(expected_tokens, -1.0, dtype=np.float32)
        )
        current_heats.append(current_heat)
        alternative_heats.append(alternative_heat)
        current_supports.append(
            _support_from_heat(
                current_heat,
                target,
                positive_margin=current_threshold,
                selected_fraction=config.selected_fraction,
            )[0]
        )
        alternative_supports.append(
            _support_from_heat(
                alternative_heat,
                target,
                positive_margin=alternative_threshold,
                selected_fraction=config.selected_fraction,
            )[0]
        )
        component_kwargs = {
            "crop_xyxy": crop_box,
            "target_bbox": target_bbox,
            "grid_shape": grid_shape,
        }
        current_intrinsic_components.append(
            _largest_component_geometry(
                current_heat,
                target,
                threshold=current_threshold,
                **component_kwargs,
            )
        )
        alternative_intrinsic_components.append(
            _largest_component_geometry(
                alternative_heat,
                target,
                threshold=alternative_threshold,
                **component_kwargs,
            )
        )
        current_exclusive_components.append(
            _largest_component_geometry(
                current_heat,
                target,
                threshold=current_threshold,
                competitor_heat=alternative_heat,
                exclusive_margin=config.exclusive_support_margin,
                **component_kwargs,
            )
        )
        alternative_exclusive_components.append(
            _largest_component_geometry(
                alternative_heat,
                target,
                threshold=alternative_threshold,
                competitor_heat=current_heat,
                exclusive_margin=config.exclusive_support_margin,
                **component_kwargs,
            )
        )
        outside_target = target * (1.0 - np.clip(overlap, 0.0, 1.0))
        current_outside_exclusive_components.append(
            _largest_component_geometry(
                current_heat,
                outside_target,
                threshold=current_threshold,
                competitor_heat=alternative_heat,
                exclusive_margin=config.exclusive_support_margin,
                **component_kwargs,
            )
        )
        alternative_outside_exclusive_components.append(
            _largest_component_geometry(
                alternative_heat,
                outside_target,
                threshold=alternative_threshold,
                competitor_heat=current_heat,
                exclusive_margin=config.exclusive_support_margin,
                **component_kwargs,
            )
        )
    current_support = float(np.mean(current_supports))
    alternative_support = float(np.mean(alternative_supports))
    directed_pair_margin = float(alternative_support - current_support)
    directed_pair_probe_features = exact_two_view_pair_features(
        current_heats,
        alternative_heats,
        target_views,
        selected_fraction=config.selected_fraction,
    )
    current_exclusive_support = float(directed_pair_probe_features[0])
    alternative_exclusive_support = float(directed_pair_probe_features[1])
    directed_pair_probe_score = float(
        np.dot(
            directed_pair_probe_weights,
            directed_pair_probe_features,
        )
    )
    if len(current_supports) > 1:
        view_delta = max(
            abs(current_supports[0] - current_supports[1]),
            abs(alternative_supports[0] - alternative_supports[1]),
        )
        view_agreement = float(max(0.0, 1.0 - min(1.0, view_delta)))
    else:
        view_agreement = 1.0

    alternative_mass = 0.0
    alternative_inside = 0.0
    current_mass = 0.0
    current_outside = 0.0
    for current_heat, alternative_heat, target, overlap in zip(
        current_heats, alternative_heats, target_views, overlap_views
    ):
        current_positive = (
            np.maximum(
                0.0,
                current_heat
                - alternative_heat
                - float(config.exclusive_support_margin),
            )
            * (current_heat >= float(current_threshold)).astype(np.float32)
            * target
        )
        alternative_positive = (
            np.maximum(
                0.0,
                alternative_heat
                - current_heat
                - float(config.exclusive_support_margin),
            )
            * (alternative_heat >= float(alternative_threshold)).astype(
                np.float32
            )
            * target
        )
        alternative_mass += float(alternative_positive.sum())
        alternative_inside += float(
            (alternative_positive * np.minimum(1.0, overlap)).sum()
        )
        current_mass += float(current_positive.sum())
        current_outside += float(
            (current_positive * (1.0 - np.minimum(1.0, overlap))).sum()
        )
    inside_fraction = (
        float(alternative_inside / alternative_mass)
        if alternative_mass > 1e-12
        else 0.0
    )
    outside_fraction = (
        float(1.0 - inside_fraction) if alternative_mass > 1e-12 else 0.0
    )
    current_outside_fraction = (
        float(current_outside / current_mass) if current_mass > 1e-12 else 0.0
    )

    current_strong = bool(
        current_supports
        and current_supports[0] >= float(current_strong_threshold)
    )
    # Presence preservation is tight-view-first, while absence remains
    # conservative across all views. This prevents a context-diluted small
    # object from being called absent.
    current_supported = bool(
        current_supports
        and current_supports[0] >= float(current_threshold)
    )
    current_weak = bool(
        current_supports
        and max(current_supports) <= float(current_negative_threshold)
    )
    alternative_supported = bool(
        alternative_support >= float(alternative_threshold)
    )
    alternative_absent = bool(
        alternative_supports
        and max(alternative_supports)
        <= float(alternative_negative_threshold)
    )
    alternative_strong = bool(
        alternative_supports
        and min(alternative_supports)
        >= float(alternative_strong_threshold)
    )
    directed_pair_dominates = bool(
        directed_pair_probe_score >= float(directed_pair_threshold)
    )
    current_intrinsic_corresponds = _component_corresponds(
        current_intrinsic_components[0],
        current_intrinsic_components[1],
        target_bbox,
        config,
    )
    alternative_intrinsic_corresponds = _component_corresponds(
        alternative_intrinsic_components[0],
        alternative_intrinsic_components[1],
        target_bbox,
        config,
    )
    current_exclusive_corresponds = _component_corresponds(
        current_exclusive_components[0],
        current_exclusive_components[1],
        target_bbox,
        config,
    )
    alternative_exclusive_corresponds = _component_corresponds(
        alternative_exclusive_components[0],
        alternative_exclusive_components[1],
        target_bbox,
        config,
    )
    current_outside_exclusive_corresponds = _component_corresponds(
        current_outside_exclusive_components[0],
        current_outside_exclusive_components[1],
        target_bbox,
        config,
    )
    alternative_outside_exclusive_corresponds = _component_corresponds(
        alternative_outside_exclusive_components[0],
        alternative_outside_exclusive_components[1],
        target_bbox,
        config,
    )
    exclusive_components_separated = _components_are_spatially_separated(
        current_exclusive_components,
        alternative_exclusive_components,
        target_bbox,
        config,
    )
    overlap_localized = bool(
        resolved_overlap_boxes
        and inside_fraction >= float(config.overlap_localized_fraction)
    )
    external_alternative_evidence = bool(
        alternative_exclusive_corresponds
        and (
            not resolved_overlap_boxes
            or (
                alternative_outside_exclusive_corresponds
                and outside_fraction
                >= float(config.outside_overlap_confirm_fraction)
            )
        )
    )
    nested_overlap = any(
        np.asarray(overlap, dtype=np.float32).max() > 0.0
        and np.asarray(valid, dtype=np.float32).max() > 0.0
        and float(
            np.sum(
                np.asarray(overlap, dtype=np.float32)
                * np.asarray(valid, dtype=np.float32)
            )
            / max(
                1e-12,
                float(np.asarray(valid, dtype=np.float32).sum()),
            )
        )
        >= 0.90
        for overlap, valid in zip(overlap_views, target_views)
    )
    view_consistent = bool(view_agreement >= 0.50)
    tx1, ty1, tx2, ty2 = [float(value) for value in list(target_bbox)[:4]]
    target_width = max(0.0, tx2 - tx1)
    target_height = max(0.0, ty2 - ty1)
    tight_x1, tight_y1, tight_x2, tight_y2 = [
        float(value) for value in list(crop_boxes[0])[:4]
    ]
    visible_width = max(0.0, min(tx2, tight_x2) - max(tx1, tight_x1))
    visible_height = max(0.0, min(ty2, tight_y2) - max(ty1, tight_y1))
    source_resolution_sufficient = bool(
        min(visible_width, visible_height)
        >= float(config.minimum_confirmation_bbox_short_side)
        and visible_width * visible_height
        >= float(config.minimum_confirmation_bbox_area)
    )
    annotated_overlap = bool(
        overlap_geometries and overlap_relation != "duplicate_like"
    )
    current_overlap_explanation = bool(
        (
            nested_overlap
            and current_strong
            and current_exclusive_corresponds
        )
        or (
            not nested_overlap
            and current_outside_exclusive_corresponds
            and current_outside_fraction
            >= float(config.outside_overlap_confirm_fraction)
        )
    )
    proved_overlap_decomposition = bool(
        annotated_overlap
        and overlap_localized
        and current_overlap_explanation
    )
    qualified_for_human_review = bool(
        diagnostic_pair_reliable
        and source_resolution_sufficient
        and directed_pair_dominates
        and alternative_exclusive_corresponds
        and view_consistent
        and external_alternative_evidence
    )
    decision_gates = {
        "directed_pair_reliable": directed_pair_reliable,
        "diagnostic_pair_reliable": diagnostic_pair_reliable,
        "directed_pair_candidate_source_independent": (
            directed_pair_candidate_source_independent
        ),
        "directed_pair_exact_calibration_contracts": (
            directed_pair_exact_calibration_contracts
        ),
        "intrinsic_references_reliable": intrinsic_references_reliable,
        "positive_confirmation_pair_reliable": (
            positive_confirmation_pair_reliable
        ),
        "positive_confirmation_pair_probe_auroc_sufficient": (
            positive_confirmation_pair_probe_auroc_sufficient
        ),
        "positive_confirmation_pair_probe_lower_bound_sufficient": (
            positive_confirmation_pair_probe_lower_bound_sufficient
        ),
        "source_resolution_sufficient": source_resolution_sufficient,
        "current_present": current_supported,
        "current_strong": current_strong,
        "current_absent": current_weak,
        "alternative_present": alternative_supported,
        "alternative_absent": alternative_absent,
        "alternative_strong": alternative_strong,
        "directed_pair_dominates": directed_pair_dominates,
        "current_spatially_coherent": current_intrinsic_corresponds,
        "alternative_spatially_coherent_both_views": (
            alternative_intrinsic_corresponds
        ),
        "current_exclusive_component_corresponds": (
            current_exclusive_corresponds
        ),
        "alternative_exclusive_component_corresponds": (
            alternative_exclusive_corresponds
        ),
        "current_outside_overlap_exclusive_component_corresponds": (
            current_outside_exclusive_corresponds
        ),
        "alternative_outside_overlap_exclusive_component_corresponds": (
            alternative_outside_exclusive_corresponds
        ),
        "exclusive_components_spatially_separated": (
            exclusive_components_separated
        ),
        "annotated_overlap": annotated_overlap,
        "proved_overlap_decomposition": proved_overlap_decomposition,
        "current_overlap_explanation": current_overlap_explanation,
        "view_consistent": view_consistent,
        "alternative_evidence_external_to_overlap": (
            external_alternative_evidence
        ),
        "alternative_evidence_localized_to_overlap": overlap_localized,
        "nested_overlap": bool(nested_overlap),
        "qualified_for_human_review": qualified_for_human_review,
    }

    reason_codes: List[str] = []
    if not current_bank_reliable:
        reason_codes.append("current_reference_unreliable")
    elif not current_source_independent:
        reason_codes.append(
            "current_reference_source_independent_support_insufficient"
        )
    if not alternative_bank_reliable:
        reason_codes.append("alternative_reference_unreliable")
    elif not alternative_source_independent:
        reason_codes.append(
            "alternative_reference_source_independent_support_insufficient"
        )
    if bool(pair_metadata["candidate_source_excluded"]):
        reason_codes.append(
            "directed_pair_candidate_source_in_calibration"
        )
    elif not bool(pair_metadata["bank_reliable"]):
        reason_codes.append("directed_pair_calibration_unreliable")
    elif not positive_confirmation_pair_probe_auroc_sufficient:
        reason_codes.append(
            "directed_pair_confirmation_auroc_below_floor"
        )
    if overlap_relation == "duplicate_like":
        status = STATUS_PAIR_CONFLICT
        reason_codes.append(
            "near_identical_cross_class_bbox_requires_review"
        )
    elif not intrinsic_references_reliable:
        status = STATUS_UNRESOLVED
    else:
        if (
            positive_confirmation_pair_reliable
            and alternative_strong
            and current_weak
            and directed_pair_dominates
            and alternative_exclusive_corresponds
            and view_consistent
            and external_alternative_evidence
            and source_resolution_sufficient
        ):
            status = STATUS_CONFIRMED_OUTLIER
            reason_codes.append(
                "corresponding_alternative_exclusive_evidence_dominates"
            )
        elif (
            current_supported
            and alternative_supported
            and current_exclusive_corresponds
            and alternative_exclusive_corresponds
            and not nested_overlap
            and (
                exclusive_components_separated
                or proved_overlap_decomposition
            )
        ):
            status = STATUS_MIXED_OR_COMPOSITE
            reason_codes.append(
                "annotated_overlap_with_both_exclusive_components"
                if proved_overlap_decomposition
                else "spatially_separated_exclusive_components"
            )
        elif (
            current_supported
            and current_exclusive_corresponds
            and alternative_absent
        ):
            status = STATUS_EXPLAINED_NOT_OUTLIER
            reason_codes.append("current_spatial_evidence_supported")
        elif (
            current_supported
            and overlap_localized
            and current_overlap_explanation
        ):
            status = STATUS_EXPLAINED_NOT_OUTLIER
            reason_codes.append(
                "nested_overlap_with_strong_current_exclusive_evidence"
                if nested_overlap
                else "overlap_localized_with_current_exclusive_evidence"
            )
        else:
            status = STATUS_UNRESOLVED
            if (
                positive_confirmation_pair_reliable
                and alternative_strong
                and current_weak
                and directed_pair_dominates
                and alternative_exclusive_corresponds
                and view_consistent
                and external_alternative_evidence
                and not source_resolution_sufficient
            ):
                reason_codes.append(
                    "source_resolution_insufficient_for_confirmation"
                )
            elif (
                current_exclusive_corresponds
                and alternative_exclusive_corresponds
                and not annotated_overlap
                and not exclusive_components_separated
            ):
                reason_codes.append("coincident_exclusive_components")
            elif overlap_localized and not current_overlap_explanation:
                reason_codes.append(
                    "overlap_explanation_lacks_current_exclusive_evidence"
                )
            else:
                reason_codes.append("spatial_evidence_not_decisive")

    if (
        status
        in {
            STATUS_CONFIRMED_OUTLIER,
            STATUS_EXPLAINED_NOT_OUTLIER,
            STATUS_MIXED_OR_COMPOSITE,
        }
        and not source_resolution_sufficient
    ):
        resolution_reason = {
            STATUS_CONFIRMED_OUTLIER: (
                "source_resolution_insufficient_for_confirmation"
            ),
            STATUS_EXPLAINED_NOT_OUTLIER: (
                "source_resolution_insufficient_for_explanation"
            ),
            STATUS_MIXED_OR_COMPOSITE: (
                "source_resolution_insufficient_for_mixed_composite"
            ),
        }[status]
        status = STATUS_UNRESOLVED
        if resolution_reason not in reason_codes:
            reason_codes.append(resolution_reason)

    shaped_current = np.stack(
        [heat.reshape(grid_shape) for heat in current_heats], axis=0
    )
    shaped_alternative = np.stack(
        [heat.reshape(grid_shape) for heat in alternative_heats], axis=0
    )
    shaped_valid = np.stack(
        [valid.reshape(grid_shape) for valid in valid_views], axis=0
    )
    shaped_target = np.stack(
        [target.reshape(grid_shape) for target in target_views], axis=0
    )
    shaped_overlap = np.stack(
        [overlap.reshape(grid_shape) for overlap in overlap_views], axis=0
    )
    evidence = {
        "schema": REFINEMENT_SCHEMA,
        "decision_contract": REFINEMENT_DECISION_CONTRACT,
        "status": status,
        "reason_codes": reason_codes,
        "current_class": current_class,
        "alternative_class": alternative_class,
        "current_support_score": current_support,
        "alternative_support_score": alternative_support,
        "current_view_support_scores": [
            float(value) for value in current_supports
        ],
        "alternative_view_support_scores": [
            float(value) for value in alternative_supports
        ],
        "intrinsic_current_support": current_support,
        "intrinsic_alternative_support": alternative_support,
        "directed_pair_margin": directed_pair_margin,
        "directed_pair_raw_margin": directed_pair_margin,
        "directed_pair_probe_score": directed_pair_probe_score,
        "directed_pair_probe_features": [
            float(value) for value in directed_pair_probe_features.tolist()
        ],
        "directed_pair_probe_feature_names": list(PAIR_PROBE_FEATURE_NAMES),
        "directed_pair_current_exclusive_support": (
            current_exclusive_support
        ),
        "directed_pair_alternative_exclusive_support": (
            alternative_exclusive_support
        ),
        "directed_pair_probe_threshold": directed_pair_threshold,
        "directed_pair_probe_weights": [
            float(value) for value in directed_pair_probe_weights.tolist()
        ],
        "directed_pair_probe_contract": str(
            pair_metadata["probe_contract"]
        ),
        "directed_pair_probe_view_contract": str(
            pair_metadata["probe_view_contract"]
        ),
        "directed_pair_probe_lower_bound_contract": str(
            pair_metadata["probe_lower_bound_contract"]
        ),
        "directed_pair_probe_fold_count": int(
            pair_metadata["probe_fold_count"]
        ),
        "directed_pair_probe_fit_status": str(
            pair_metadata["probe_fit_status"]
        ),
        "directed_pair_probe_fold_digest": str(
            pair_metadata["probe_fold_digest"]
        ),
        "directed_pair_probe_fit_eval_split_digest": str(
            pair_metadata["fit_eval_split_digest"]
        ),
        "directed_pair_probe_fit_current_source_count": int(
            pair_metadata["fit_current_source_count"]
        ),
        "directed_pair_probe_fit_alternative_source_count": int(
            pair_metadata["fit_alternative_source_count"]
        ),
        "directed_pair_probe_eval_current_source_count": int(
            pair_metadata["eval_current_source_count"]
        ),
        "directed_pair_probe_eval_alternative_source_count": int(
            pair_metadata["eval_alternative_source_count"]
        ),
        "directed_pair_threshold": directed_pair_threshold,
        "current_negative_threshold": current_negative_threshold,
        "current_support_threshold": current_threshold,
        "current_strong_threshold": current_strong_threshold,
        "alternative_negative_threshold": alternative_negative_threshold,
        "alternative_support_threshold": alternative_threshold,
        "alternative_strong_threshold": alternative_strong_threshold,
        "support_threshold_source": support_threshold_source,
        "directed_pair_tier": str(pair_metadata["tier"]),
        "directed_pair_reliable": directed_pair_reliable,
        "directed_pair_bank_reliable": bool(
            pair_metadata["bank_reliable"]
        ),
        "diagnostic_pair_reliability_contract": str(
            pair_metadata["diagnostic_reliability_contract"]
        ),
        "diagnostic_pair_reliable": diagnostic_pair_reliable,
        "diagnostic_pair_bank_reliable": bool(
            pair_metadata["diagnostic_bank_reliable"]
        ),
        "positive_confirmation_pair_reliable": (
            positive_confirmation_pair_reliable
        ),
        "directed_pair_candidate_source_excluded": bool(
            pair_metadata["candidate_source_excluded"]
        ),
        "directed_pair_candidate_source_fingerprint": str(
            pair_metadata["candidate_source_fingerprint"]
        ),
        "directed_pair_candidate_source_membership_roles": list(
            pair_metadata["candidate_source_membership_roles"]
        ),
        "directed_pair_heldout_auroc": float(
            pair_metadata["probe_oof_auroc"]
        ),
        "directed_pair_eval_auroc_lower_bound": float(
            pair_metadata["probe_eval_auroc_lower_bound"]
        ),
        "positive_confirmation_pair_probe_auroc_floor": float(
            MIN_POSITIVE_CONFIRMATION_PAIR_PROBE_AUROC
        ),
        "positive_confirmation_pair_probe_auroc_lower_bound_floor": float(
            MIN_POSITIVE_CONFIRMATION_PAIR_PROBE_AUROC_LOWER_BOUND
        ),
        "directed_pair_probe_fit_balanced_accuracy": float(
            pair_metadata["probe_fit_balanced_accuracy"]
        ),
        "directed_pair_probe_eval_sensitivity": float(
            pair_metadata["probe_eval_sensitivity"]
        ),
        "directed_pair_probe_eval_specificity": float(
            pair_metadata["probe_eval_specificity"]
        ),
        "directed_pair_current_absence_eval_fraction": float(
            pair_metadata["current_absence_eval_fraction"]
        ),
        "directed_pair_alternative_strong_eval_fraction": float(
            pair_metadata["alternative_strong_eval_fraction"]
        ),
        "directed_pair_current_source_count": int(
            pair_metadata["current_source_count"]
        ),
        "directed_pair_alternative_source_count": int(
            pair_metadata["alternative_source_count"]
        ),
        "directed_pair_current_patch_count": int(
            pair_metadata["current_patch_count"]
        ),
        "directed_pair_alternative_patch_count": int(
            pair_metadata["alternative_patch_count"]
        ),
        "directed_pair_alternative_passing_source_fraction": float(
            pair_metadata["alternative_passing_source_fraction"]
        ),
        "decision_gates": decision_gates,
        "human_review_qualification_contract": (
            HUMAN_REVIEW_QUALIFICATION_CONTRACT
        ),
        "human_review_rank_contract": HUMAN_REVIEW_RANK_CONTRACT,
        "qualified_for_human_review": qualified_for_human_review,
        # The service scores one candidate at a time. The orchestration layer
        # assigns a stable one-based rank after every candidate is available.
        "human_review_rank": None,
        "alternative_evidence_inside_overlap_fraction": inside_fraction,
        "alternative_evidence_outside_overlap_fraction": outside_fraction,
        "current_evidence_outside_overlap_fraction": current_outside_fraction,
        "overlap_relation": overlap_relation,
        "overlap_object_count": len(overlap_geometries),
        "annotated_overlap_alternative_bbox_xyxy": (
            annotated_overlap_alternative_bbox_xyxy
        ),
        # The scorer owns geometry selection but does not receive annotation
        # identities. The orchestration layer binds this selected box back to
        # the exact, deterministically ordered overlap-index row.
        "annotated_overlap_alternative_point_id": None,
        "reference_reliable": directed_pair_reliable,
        "intrinsic_references_reliable": intrinsic_references_reliable,
        "reference_distinct_source_count": int(
            min(
                bank.class_distinct_sources(current_class),
                bank.class_distinct_sources(alternative_class),
            )
        ),
        "current_reference_tier": bank.class_reliability_tier(
            current_class
        ),
        "alternative_reference_tier": bank.class_reliability_tier(
            alternative_class
        ),
        "current_reference_heldout_auroc": bank.class_heldout_auroc(
            current_class
        ),
        "alternative_reference_heldout_auroc": bank.class_heldout_auroc(
            alternative_class
        ),
        "view_agreement": view_agreement,
        "current_largest_component_fraction": float(
            max(
                (
                    float(component.get("mass_fraction") or 0.0)
                    for component in current_intrinsic_components
                ),
                default=0.0,
            )
        ),
        "alternative_largest_component_fraction": float(
            max(
                (
                    float(component.get("mass_fraction") or 0.0)
                    for component in alternative_intrinsic_components
                ),
                default=0.0,
            )
        ),
        "current_intrinsic_components": current_intrinsic_components,
        "alternative_intrinsic_components": alternative_intrinsic_components,
        "current_exclusive_components": current_exclusive_components,
        "alternative_exclusive_components": alternative_exclusive_components,
        "current_outside_overlap_exclusive_components": (
            current_outside_exclusive_components
        ),
        "alternative_outside_overlap_exclusive_components": (
            alternative_outside_exclusive_components
        ),
        "target_bbox_width": target_width,
        "target_bbox_height": target_height,
        "target_bbox_area": target_width * target_height,
        "visible_target_bbox_width": visible_width,
        "visible_target_bbox_height": visible_height,
        "visible_target_bbox_area": visible_width * visible_height,
        "minimum_confirmation_bbox_short_side": float(
            config.minimum_confirmation_bbox_short_side
        ),
        "minimum_confirmation_bbox_area": float(
            config.minimum_confirmation_bbox_area
        ),
        "calibration_status": bank.calibration_status,
        "_sidecar": {
            "point_id": str(point_id or ""),
            "current_heatmap": shaped_current.astype(np.float16),
            "alternative_heatmap": shaped_alternative.astype(np.float16),
            "target_mask": (
                shaped_target * np.float32(255.0)
            ).astype(np.uint8),
            "valid_mask": (
                shaped_valid > 0.0
            ).astype(np.uint8),
            "overlap_mask": (
                np.clip(shaped_overlap, 0.0, 1.0) * np.float32(255.0)
            ).astype(np.uint8),
        },
    }
    if not confirmation_invariants_hold(evidence):
        evidence["status"] = STATUS_UNRESOLVED
        evidence["reason_codes"] = list(evidence["reason_codes"]) + [
            "confirmed_outlier_invariant_failed"
        ]
    return evidence


def unresolved_evidence(
    *,
    current_class: str,
    alternative_class: str,
    reason: str,
    status: str = STATUS_UNRESOLVED,
) -> Dict[str, Any]:
    safe_status = (
        status
        if status in {STATUS_UNRESOLVED, STATUS_PAIR_CONFLICT}
        else STATUS_UNRESOLVED
    )
    return {
        "schema": REFINEMENT_SCHEMA,
        "decision_contract": REFINEMENT_DECISION_CONTRACT,
        "status": safe_status,
        "reason_codes": [str(reason or "refinement_unavailable")],
        "current_class": str(current_class or ""),
        "alternative_class": str(alternative_class or ""),
        "current_support_score": None,
        "alternative_support_score": None,
        "current_view_support_scores": None,
        "alternative_view_support_scores": None,
        "intrinsic_current_support": None,
        "intrinsic_alternative_support": None,
        "directed_pair_margin": None,
        "directed_pair_raw_margin": None,
        "directed_pair_probe_score": None,
        "directed_pair_probe_features": None,
        "directed_pair_probe_feature_names": list(PAIR_PROBE_FEATURE_NAMES),
        "directed_pair_current_exclusive_support": None,
        "directed_pair_alternative_exclusive_support": None,
        "directed_pair_probe_threshold": None,
        "directed_pair_probe_weights": None,
        "directed_pair_probe_contract": PAIR_PROBE_CONTRACT,
        "directed_pair_probe_view_contract": PAIR_PROBE_VIEW_CONTRACT,
        "directed_pair_probe_lower_bound_contract": (
            PAIR_PROBE_LOWER_BOUND_CONTRACT
        ),
        "directed_pair_probe_fold_count": 0,
        "directed_pair_probe_fit_status": "not_applicable",
        "directed_pair_probe_fold_digest": "",
        "directed_pair_probe_fit_eval_split_digest": "",
        "directed_pair_probe_fit_current_source_count": 0,
        "directed_pair_probe_fit_alternative_source_count": 0,
        "directed_pair_probe_eval_current_source_count": 0,
        "directed_pair_probe_eval_alternative_source_count": 0,
        "directed_pair_threshold": None,
        "current_negative_threshold": None,
        "current_support_threshold": None,
        "current_strong_threshold": None,
        "alternative_negative_threshold": None,
        "alternative_support_threshold": None,
        "alternative_strong_threshold": None,
        "support_threshold_source": "intrinsic_fallback",
        "directed_pair_tier": "low",
        "directed_pair_reliable": False,
        "directed_pair_bank_reliable": False,
        "diagnostic_pair_reliability_contract": (
            DIAGNOSTIC_PAIR_RELIABILITY_CONTRACT
        ),
        "diagnostic_pair_reliable": False,
        "diagnostic_pair_bank_reliable": False,
        "positive_confirmation_pair_reliable": False,
        "directed_pair_candidate_source_excluded": False,
        "directed_pair_candidate_source_fingerprint": "",
        "directed_pair_candidate_source_membership_roles": [],
        "directed_pair_heldout_auroc": 0.0,
        "directed_pair_eval_auroc_lower_bound": 0.0,
        "positive_confirmation_pair_probe_auroc_floor": float(
            MIN_POSITIVE_CONFIRMATION_PAIR_PROBE_AUROC
        ),
        "positive_confirmation_pair_probe_auroc_lower_bound_floor": float(
            MIN_POSITIVE_CONFIRMATION_PAIR_PROBE_AUROC_LOWER_BOUND
        ),
        "directed_pair_probe_fit_balanced_accuracy": 0.0,
        "directed_pair_probe_eval_sensitivity": 0.0,
        "directed_pair_probe_eval_specificity": 0.0,
        "directed_pair_current_absence_eval_fraction": 0.0,
        "directed_pair_alternative_strong_eval_fraction": 0.0,
        "directed_pair_current_source_count": 0,
        "directed_pair_alternative_source_count": 0,
        "directed_pair_current_patch_count": 0,
        "directed_pair_alternative_patch_count": 0,
        "directed_pair_alternative_passing_source_fraction": 0.0,
        "decision_gates": {
            "directed_pair_reliable": False,
            "diagnostic_pair_reliable": False,
            # No pair-calibration membership was consulted for this synthetic
            # pre-scoring abstention. The source-exclusion gate therefore
            # mirrors candidate_source_excluded=False; every confirm-capable
            # pair and intrinsic gate remains false.
            "directed_pair_candidate_source_independent": True,
            "directed_pair_exact_calibration_contracts": False,
            "intrinsic_references_reliable": False,
            "positive_confirmation_pair_reliable": False,
            "positive_confirmation_pair_probe_auroc_sufficient": False,
            "positive_confirmation_pair_probe_lower_bound_sufficient": False,
            "source_resolution_sufficient": False,
            "current_present": False,
            "current_strong": False,
            "current_absent": False,
            "alternative_present": False,
            "alternative_absent": False,
            "alternative_strong": False,
            "directed_pair_dominates": False,
            "current_spatially_coherent": False,
            "alternative_spatially_coherent_both_views": False,
            "current_exclusive_component_corresponds": False,
            "alternative_exclusive_component_corresponds": False,
            "current_outside_overlap_exclusive_component_corresponds": False,
            "alternative_outside_overlap_exclusive_component_corresponds": False,
            "exclusive_components_spatially_separated": False,
            "annotated_overlap": False,
            "proved_overlap_decomposition": False,
            "current_overlap_explanation": False,
            "view_consistent": False,
            "alternative_evidence_external_to_overlap": False,
            "alternative_evidence_localized_to_overlap": False,
            "nested_overlap": False,
            "qualified_for_human_review": False,
        },
        "human_review_qualification_contract": (
            HUMAN_REVIEW_QUALIFICATION_CONTRACT
        ),
        "human_review_rank_contract": HUMAN_REVIEW_RANK_CONTRACT,
        "qualified_for_human_review": False,
        "human_review_rank": None,
        "alternative_evidence_inside_overlap_fraction": None,
        "alternative_evidence_outside_overlap_fraction": None,
        "current_evidence_outside_overlap_fraction": None,
        "overlap_relation": "none",
        "overlap_object_count": 0,
        "annotated_overlap_alternative_bbox_xyxy": None,
        "annotated_overlap_alternative_point_id": None,
        "reference_reliable": False,
        "intrinsic_references_reliable": False,
        "reference_distinct_source_count": 0,
        "current_reference_tier": "low",
        "alternative_reference_tier": "low",
        "current_reference_heldout_auroc": 0.0,
        "alternative_reference_heldout_auroc": 0.0,
        "view_agreement": None,
        "current_largest_component_fraction": None,
        "alternative_largest_component_fraction": None,
        "current_intrinsic_components": [],
        "alternative_intrinsic_components": [],
        "current_exclusive_components": [],
        "alternative_exclusive_components": [],
        "current_outside_overlap_exclusive_components": [],
        "alternative_outside_overlap_exclusive_components": [],
        "target_bbox_width": None,
        "target_bbox_height": None,
        "target_bbox_area": None,
        "visible_target_bbox_width": None,
        "visible_target_bbox_height": None,
        "visible_target_bbox_area": None,
        "minimum_confirmation_bbox_short_side": None,
        "minimum_confirmation_bbox_area": None,
        "calibration_status": "unavailable",
        "sidecar_row": None,
    }


def sidecar_arrays(
    sidecar_rows: Sequence[Mapping[str, Any]],
    *,
    grid_shape: Tuple[int, int],
    view_count: int = 2,
) -> Dict[str, np.ndarray]:
    count = len(sidecar_rows)
    shape = (count, int(view_count), int(grid_shape[0]), int(grid_shape[1]))
    current = np.zeros(shape, dtype=np.float16)
    alternative = np.zeros(shape, dtype=np.float16)
    valid = np.zeros(shape, dtype=np.uint8)
    target = np.zeros(shape, dtype=np.uint8)
    overlap = np.zeros(shape, dtype=np.uint8)
    point_ids: List[str] = []
    for index, row in enumerate(sidecar_rows):
        point_ids.append(str(row.get("point_id") or ""))
        for destination, key in (
            (current, "current_heatmap"),
            (alternative, "alternative_heatmap"),
            (valid, "valid_mask"),
            (target, "target_mask"),
            (overlap, "overlap_mask"),
        ):
            value = np.asarray(row.get(key))
            if value.shape != shape[1:]:
                raise ValueError("class_analysis_refinement_sidecar_shape_invalid")
            destination[index] = value
    return {
        "point_ids": np.asarray(point_ids),
        "current_heatmaps": current,
        "alternative_heatmaps": alternative,
        "valid_masks": valid,
        "target_masks": target,
        "overlap_masks": overlap,
    }
