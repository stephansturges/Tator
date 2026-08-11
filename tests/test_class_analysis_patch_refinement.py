from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import localinferenceapi as api
from services.class_analysis_patch_refinement import (
    CALIBRATION_STATUS_SOURCE_AWARE,
    CAPTURE_GROUP_CONTRACT,
    DIAGNOSTIC_PAIR_RELIABILITY_CONTRACT,
    HUMAN_REVIEW_QUALIFICATION_CONTRACT,
    HUMAN_REVIEW_RANK_CONTRACT,
    FREQUENT_OVERLAP_PRIOR_CONTRACT,
    FREQUENT_OVERLAP_FIT_ELIGIBILITY_CONTRACT,
    FREQUENT_OVERLAP_TRIAGE_CONTRACT,
    SELECTOR_PRIORITY_CONTRACT,
    REFINEMENT_DECISION_CONTRACT,
    REFINEMENT_SCHEMA,
    STATUS_CONFIRMED_OUTLIER,
    STATUS_EXPLAINED_NOT_OUTLIER,
    STATUS_MIXED_OR_COMPOSITE,
    STATUS_PAIR_CONFLICT,
    STATUS_UNRESOLVED,
    CaptureGroupIndex,
    FrequentOverlapPrior,
    ReferenceBank,
    RefinementConfig,
    StreamingReferenceBankBuilder,
    _assigned_cluster_medoid_indices,
    _calibration_source_split_digest,
    _build_capture_group_index,
    _global_heldout_sources,
    _largest_positive_component,
    _mean_top_source_similarity,
    _round_robin_class_rows,
    _resolve_reliability_active_set,
    _source_consensus_similarity,
    _support_from_heat,
    bbox_overlap_geometry,
    build_frequent_overlap_prior as _build_frequent_overlap_prior,
    build_overlap_index,
    class_margin_heatmap,
    overlap_annotation_selection_key,
    patch_source_centres,
    pair_metrics_are_diagnostic,
    pair_metrics_are_reliable,
    rasterize_box_fractions,
    score_candidate,
    select_within_class_outlier_candidates,
    sidecar_arrays,
    strip_torch_dinov3_special_tokens,
)


def build_frequent_overlap_prior(records, **kwargs):
    """Build a prior from an explicitly Stage-1-screened test universe."""

    kwargs.setdefault(
        "trusted_screened_point_ids",
        {
            str(
                record.get("point_id")
                or record.get("review_object_key")
                or ""
            ).strip()
            for record in records
            if isinstance(record, dict)
            and str(
                record.get("point_id")
                or record.get("review_object_key")
                or ""
            ).strip()
        },
    )
    return _build_frequent_overlap_prior(records, **kwargs)


def _npy_member_bytes(value: np.ndarray) -> bytes:
    handle = io.BytesIO()
    np.save(handle, np.asarray(value), allow_pickle=False)
    return handle.getvalue()


def _npy_header_only_bytes(*, shape, dtype) -> bytes:
    handle = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        handle,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
            "fortran_order": False,
            "shape": tuple(shape),
        },
    )
    return handle.getvalue()


def _write_npz_members(path: Path, members) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, encoded in members.items():
            archive.writestr(f"{name}.npy", encoded)


def _synthetic_bank() -> ReferenceBank:
    class_a = np.asarray([[1.0, 0.0]] * 6, dtype=np.float32)
    class_b = np.asarray([[0.0, 1.0]] * 6, dtype=np.float32)
    class_a_background = np.asarray([[-1.0, 0.0]] * 6, dtype=np.float32)
    class_b_background = np.asarray([[0.0, -1.0]] * 6, dtype=np.float32)
    pair_bool = np.asarray([[False, True], [True, False]], dtype=bool)
    pair_metrics = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    pair_counts = np.asarray([[0, 8], [8, 0]], dtype=np.int32)
    pair_digests = np.asarray([["", "a" * 64], ["b" * 64, ""]])
    return ReferenceBank(
        class_names=["A", "B"],
        prototypes=np.stack([class_a, class_b]),
        prototype_counts=np.asarray([6, 6], dtype=np.int32),
        prototype_source_ids=np.asarray(
            [
                ["a1", "a1", "a2", "a2", "a3", "a3"],
                ["b1", "b1", "b2", "b2", "b3", "b3"],
            ]
        ),
        background_prototypes=np.stack(
            [class_a_background, class_b_background]
        ),
        background_prototype_counts=np.asarray([6, 6], dtype=np.int32),
        background_prototype_source_ids=np.asarray(
            [
                ["abg1", "abg1", "abg2", "abg2", "abg3", "abg3"],
                ["bbg1", "bbg1", "bbg2", "bbg2", "bbg3", "bbg3"],
            ]
        ),
        anchor_counts=np.asarray([64, 64], dtype=np.int32),
        distinct_source_counts=np.asarray([8, 8], dtype=np.int32),
        reliable=np.asarray([True, True]),
        reliability_tiers=np.asarray(["high", "high"]),
        heldout_aurocs=np.asarray([1.0, 1.0], dtype=np.float32),
        support_thresholds=np.asarray([0.08, 0.08], dtype=np.float32),
        strong_support_thresholds=np.asarray([0.12, 0.12], dtype=np.float32),
        negative_support_thresholds=np.asarray([0.02, 0.02], dtype=np.float32),
        pair_reliable=pair_bool,
        pair_reliability_tiers=np.asarray(
            [["low", "high"], ["high", "low"]]
        ),
        pair_heldout_aurocs=pair_metrics,
        pair_dominance_thresholds=np.zeros((2, 2), dtype=np.float32),
        pair_current_negative_thresholds=np.full(
            (2, 2), 0.02, dtype=np.float32
        ),
        pair_alternative_strong_thresholds=np.full(
            (2, 2), 0.12, dtype=np.float32
        ),
        pair_current_source_counts=pair_counts,
        pair_alternative_source_counts=pair_counts.copy(),
        pair_current_patch_counts=pair_counts.copy(),
        pair_alternative_patch_counts=pair_counts.copy(),
        pair_alternative_passing_source_fractions=pair_metrics.copy(),
        pair_probe_weights=np.broadcast_to(
            np.asarray([-np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32),
            (2, 2, 2),
        ).copy(),
        pair_probe_thresholds=np.zeros((2, 2), dtype=np.float32),
        pair_probe_oof_aurocs=pair_metrics.copy(),
        pair_probe_fold_counts=np.asarray([[0, 1], [1, 0]], dtype=np.int32),
        pair_probe_fit_statuses=np.asarray(
            [["not_applicable", "ok"], ["ok", "not_applicable"]]
        ),
        pair_probe_fold_digests=pair_digests,
        pair_probe_eval_auroc_lower_bounds=np.asarray(
            [[0.0, 0.90], [0.90, 0.0]], dtype=np.float32
        ),
        pair_probe_fit_current_source_counts=pair_counts.copy(),
        pair_probe_fit_alternative_source_counts=pair_counts.copy(),
        pair_probe_eval_current_source_counts=pair_counts.copy(),
        pair_probe_eval_alternative_source_counts=pair_counts.copy(),
        pair_probe_fit_balanced_accuracies=pair_metrics.copy(),
        pair_probe_eval_sensitivities=pair_metrics.copy(),
        pair_probe_eval_specificities=pair_metrics.copy(),
        pair_current_absence_eval_fractions=pair_metrics.copy(),
        pair_alternative_strong_eval_fractions=pair_metrics.copy(),
        pair_probe_fit_eval_split_digests=pair_digests.copy(),
        pair_calibration_class_source_ids=np.asarray(
            [
                [f"{index:016x}" for index in range(16)],
                [f"{index:016x}" for index in range(16, 32)],
            ]
        ),
        pair_calibration_class_source_counts=np.asarray(
            [16, 16], dtype=np.int32
        ),
        projection_mean=np.zeros(2, dtype=np.float32),
        projection_components=np.eye(2, dtype=np.float32),
        calibration_status=CALIBRATION_STATUS_SOURCE_AWARE,
        **_synthetic_calibration_provenance(2),
    )


def test_diagnostic_pair_reliability_does_not_consume_confirmation_yield_gates():
    shared = {
        "current_class_reliable": True,
        "alternative_class_reliable": True,
        "fit_current_source_count": 12,
        "fit_alternative_source_count": 12,
        "eval_current_source_count": 12,
        "eval_alternative_source_count": 12,
        "eval_auroc": 0.93,
        "eval_auroc_lower_bound": 0.81,
        "fit_balanced_accuracy": 0.82,
        "eval_sensitivity": 0.59,
        "eval_specificity": 1.0,
    }

    assert pair_metrics_are_diagnostic(**shared) is True
    assert pair_metrics_are_reliable(
        **shared,
        current_absence_eval_fraction=0.88,
        alternative_strong_eval_fraction=0.24,
    ) is False


def test_diagnostic_human_review_qualification_is_advisory_and_ranked():
    bank = _synthetic_bank()
    # Keep a discriminative, source-disjoint exact-view probe while making its
    # positive operating-point yield ineligible for automatic confirmation.
    bank.pair_reliable[0, 1] = False
    bank.pair_reliability_tiers[0, 1] = "low"
    bank.pair_alternative_strong_eval_fractions[0, 1] = 0.24
    class_b = np.asarray([0.0, 1.0], dtype=np.float32)
    tokens = np.stack([class_b] * 16)

    evidence = score_candidate(
        point_id="diagnostic-only",
        current_class="A",
        alternative_class="B",
        token_views=[tokens, tokens],
        crop_boxes=[[0, 0, 32, 32], [0, 0, 32, 32]],
        target_bbox=[0, 0, 32, 32],
        grid_shape=(4, 4),
        alternative_overlap_boxes=[],
        bank=bank,
        config=RefinementConfig(),
    )

    assert evidence["diagnostic_pair_reliability_contract"] == (
        DIAGNOSTIC_PAIR_RELIABILITY_CONTRACT
    )
    assert evidence["diagnostic_pair_bank_reliable"] is True
    assert evidence["diagnostic_pair_reliable"] is True
    assert evidence["positive_confirmation_pair_reliable"] is False
    assert evidence["decision_gates"]["diagnostic_pair_reliable"] is True
    assert evidence["qualified_for_human_review"] is True
    assert evidence["decision_gates"]["qualified_for_human_review"] is True
    assert evidence["human_review_qualification_contract"] == (
        HUMAN_REVIEW_QUALIFICATION_CONTRACT
    )
    assert evidence["human_review_rank_contract"] == HUMAN_REVIEW_RANK_CONTRACT
    assert evidence["human_review_rank"] is None
    assert evidence["status"] == STATUS_UNRESOLVED

    candidate = {
        "point_id": "diagnostic-only",
        "class_name": "A",
        "wrong_class_suspicion": 0.91,
        "refined_outlier": evidence,
    }
    assert api._class_analysis_assign_human_review_ranks([candidate]) == 1
    assert evidence["human_review_rank"] == 1
    selector_summary = api._class_analysis_assign_selector_priority_ranks(
        [candidate]
    )
    assert selector_summary["candidate_count"] == 1
    assert evidence["selector_priority_contract"] == SELECTOR_PRIORITY_CONTRACT
    assert evidence["selector_priority_rank"] == 1
    rail = api._class_analysis_compact_refinement_rail(candidate)
    assert rail is not None
    assert rail["diagnostic_pair_reliable"] is True
    assert rail["positive_confirmation_pair_reliable"] is False
    assert rail["qualified_for_human_review"] is True
    assert rail["human_review_rank"] == 1
    selector_rail = rail["selector_priority"]
    assert selector_rail["contract"] == SELECTOR_PRIORITY_CONTRACT
    assert 0.0 <= selector_rail["base_score"] <= 1.0
    assert selector_rail["base_rank"] == 1
    assert selector_rail["score"] == pytest.approx(
        selector_rail["expected_review_utility"]
    )
    assert selector_rail["rank"] == 1
    assert selector_rail["candidate_count"] == 1
    assert selector_rail["current_evidence_state"] in {
        "present",
        "absent",
        "indeterminate",
        "unavailable",
    }
    assert selector_rail["expected_review_utility"] == pytest.approx(
        selector_rail["actionable_probability"]
        * (0.75 + 0.25 * selector_rail["reviewability_probability"])
    )
    assert selector_rail["same_image_context"]["available"] is False
    missing_band_candidate = copy.deepcopy(candidate)
    missing_band_candidate["refined_outlier"].pop(
        "selector_priority_band_rank"
    )
    assert api._class_analysis_compact_refinement_rail(
        missing_band_candidate
    ) is None
    summary = api._class_analysis_refinement_v3_observability(
        refinement_candidates=[candidate],
        bank=bank,
        anchor_selection={},
    )
    assert summary["human_review_qualification"] == {
        "contract": HUMAN_REVIEW_QUALIFICATION_CONTRACT,
        "diagnostic_pair_reliability_contract": (
            DIAGNOSTIC_PAIR_RELIABILITY_CONTRACT
        ),
        "rank_contract": HUMAN_REVIEW_RANK_CONTRACT,
        "required_gates": [
            "diagnostic_pair_reliable",
            "source_resolution_sufficient",
            "directed_pair_dominates",
            "alternative_exclusive_component_corresponds",
            "view_consistent",
            "alternative_evidence_external_to_overlap",
        ],
        "qualified_candidate_count": 1,
        "ranked_candidate_count": 1,
        "changes_semantic_status": False,
        "changes_default_queue": False,
    }
    assert summary["pair_coverage"]["reliable_candidate_count"] == 0
    assert (
        summary["pair_coverage"]["diagnostic_reliable_candidate_count"]
        == 1
    )
    assert summary["queue_policy"]["default_queue"] == (
        "selector_ranked_stage1_candidates"
    )
    assert summary["queue_policy"]["effective_default_candidate_count"] == 1


def test_human_review_rank_is_stable_and_does_not_change_candidate_order():
    candidates = [
        {
            "point_id": "unresolved-high",
            "wrong_class_suspicion": 0.99,
            "refined_outlier": {
                "status": STATUS_UNRESOLVED,
                "qualified_for_human_review": True,
                "directed_pair_probe_score": 0.8,
                "directed_pair_probe_threshold": 0.2,
            },
        },
        {
            "point_id": "confirmed-low",
            "wrong_class_suspicion": 0.20,
            "refined_outlier": {
                "status": STATUS_CONFIRMED_OUTLIER,
                "qualified_for_human_review": True,
                "directed_pair_probe_score": 0.3,
                "directed_pair_probe_threshold": 0.2,
            },
        },
        {
            "point_id": "not-qualified",
            "wrong_class_suspicion": 1.0,
            "refined_outlier": {
                "status": STATUS_UNRESOLVED,
                "qualified_for_human_review": False,
            },
        },
    ]
    original_order = [candidate["point_id"] for candidate in candidates]

    assert api._class_analysis_assign_human_review_ranks(candidates) == 2

    assert [candidate["point_id"] for candidate in candidates] == original_order
    assert candidates[0]["refined_outlier"]["human_review_rank"] == 2
    assert candidates[1]["refined_outlier"]["human_review_rank"] == 1
    assert candidates[2]["refined_outlier"]["human_review_rank"] is None


def _overlap_prior_record(
    source_index: int,
    point_suffix: str,
    class_name: str,
    bbox: list[float],
) -> dict:
    return {
        "point_id": f"p-{source_index}-{point_suffix}",
        "class_name": class_name,
        "split": "train",
        "image_relpath": f"source-{source_index}.jpg",
        "_image_sha256": f"{source_index + 1:064x}",
        "capture_group_id": f"capture-{source_index}",
        "bbox_xyxy": bbox,
    }


def _affirmative_overlap_refinement_evidence(
    *, status: str = STATUS_UNRESOLVED
) -> dict:
    return {
        "status": status,
        "intrinsic_current_support": 0.42,
        "intrinsic_alternative_support": 0.31,
        "decision_gates": {
            "current_present": True,
            "current_absent": False,
            "alternative_evidence_localized_to_overlap": True,
            "alternative_evidence_external_to_overlap": False,
        },
    }


def test_frequent_overlap_prior_is_source_balanced_smoothed_and_loso():
    records = []
    for source_index in range(30):
        records.append(
            _overlap_prior_record(
                source_index, "bike", "Bike", [0, 0, 100, 100]
            )
        )
        if source_index < 20:
            records.append(
                _overlap_prior_record(
                    source_index, "person", "Person", [10, 0, 90, 100]
                )
            )
    prior = build_frequent_overlap_prior(records)
    evidence = prior.candidate_evidence(
        current_class="Bike",
        alternative_class="Person",
        query_source_key=f"sha256:{1:064x}",
        overlap_matches=[
            {
                "class_name": "Person",
                "iou": 0.8,
                "target_area_covered": 0.8,
                "other_area_covered": 1.0,
                "relation": "partial_contamination",
            }
        ],
        candidate_refinement_evidence=(
            _affirmative_overlap_refinement_evidence()
        ),
    )

    assert evidence["contract"] == FREQUENT_OVERLAP_PRIOR_CONTRACT
    assert evidence["candidate_source_excluded"] is True
    assert evidence["eligible_source_count"] == 29
    assert evidence["overlap_source_count"] == 19
    assert evidence["smoothed_source_incidence"] < 19 / 29
    assert evidence["source_incidence_wilson_lower_bound"] > 0.10
    assert evidence["reliable"] is True
    assert evidence["applies"] is True
    assert 0.0 < evidence["priority_adjustment"] <= 0.35
    assert "common_overlap_priority_decrease" in evidence["reasons"]
    assert len(evidence["fit_source_digest"]) == 64
    other_pair_evidence = prior.candidate_evidence(
        current_class="Bike",
        alternative_class="Boat",
        query_source_key=f"sha256:{1:064x}",
        overlap_matches=[],
    )
    assert other_pair_evidence["fit_source_digest"] != (
        evidence["fit_source_digest"]
    )
    assert prior.summary()["changes_candidate_membership"] is False


def test_frequent_overlap_prior_does_not_trust_rare_duplicate_pairs():
    records = []
    for source_index in range(25):
        records.append(
            _overlap_prior_record(
                source_index, "bike", "Bike", [0, 0, 100, 100]
            )
        )
        if source_index < 3:
            # Near-identical ownership conflicts are learned separately, but
            # three sources are intentionally too few to trust.
            records.append(
                _overlap_prior_record(
                    source_index, "person", "Person", [0, 0, 100, 100]
                )
            )
    prior = build_frequent_overlap_prior(records)
    evidence = prior.candidate_evidence(
        current_class="Bike",
        alternative_class="Person",
        query_source_key="sha256:ffffffff",
        overlap_matches=[
            {
                "class_name": "Person",
                "iou": 1.0,
                "target_area_covered": 1.0,
                "other_area_covered": 1.0,
                "relation": "duplicate_like",
            }
        ],
    )

    assert evidence["geometry_stratum"] == "duplicate_like"
    assert evidence["overlap_source_count"] == 3
    assert evidence["reliable"] is False
    assert evidence["candidate_material_overlap"] is False
    assert evidence["candidate_duplicate_like_overlap_count"] == 1
    assert evidence["priority_adjustment"] == 0.0
    assert "frequent_overlap_prior_no_eligible_capture_tier" in (
        evidence["reasons"]
    )


def test_frequent_overlap_prior_does_not_let_one_dense_source_dominate():
    records = []
    for source_index in range(25):
        bike_count = 100 if source_index == 0 else 1
        for object_index in range(bike_count):
            records.append(
                _overlap_prior_record(
                    source_index,
                    f"bike-{object_index}",
                    "Bike",
                    [0, 0, 100, 100],
                )
            )
        if source_index < 8:
            records.append(
                _overlap_prior_record(
                    source_index,
                    "person",
                    "Person",
                    [10, 0, 90, 100],
                )
            )
    prior = build_frequent_overlap_prior(records)
    evidence = prior.candidate_evidence(
        current_class="Bike",
        alternative_class="Person",
        query_source_key="sha256:ffffffff",
        overlap_matches=[
            {
                "class_name": "Person",
                "iou": 0.8,
                "target_area_covered": 0.8,
                "other_area_covered": 1.0,
                "relation": "target_contains_other",
            }
        ],
    )

    assert evidence["eligible_source_count"] == 25
    assert evidence["overlap_source_count"] == 8
    assert evidence["raw_source_incidence"] == pytest.approx(8 / 25)
    # The 100 overlapping Bike objects in source zero carry the same total
    # weight as a source containing one Bike; an object-weighted rate here
    # would be above 0.85.
    assert evidence["smoothed_source_balanced_object_rate"] < 0.40
    assert evidence["reliable"] is True


def test_common_overlap_requires_supporting_patch_evidence_for_full_decrease():
    records = []
    for source_index in range(30):
        records.append(
            _overlap_prior_record(
                source_index, "pole", "ElevatedFixture", [0, 0, 100, 100]
            )
        )
        if source_index < 22:
            records.append(
                _overlap_prior_record(
                    source_index, "building", "Building", [0, 0, 80, 100]
                )
            )
    prior = build_frequent_overlap_prior(records)
    match = {
        "class_name": "Building",
        "iou": 0.8,
        "target_area_covered": 0.8,
        "other_area_covered": 1.0,
        "relation": "partial_contamination",
    }
    base = {
        "current_class": "ElevatedFixture",
        "alternative_class": "Building",
        "query_source_key": f"sha256:{1:064x}",
        "overlap_matches": [match],
    }
    supported = prior.candidate_evidence(
        **base,
        candidate_refinement_evidence={
            "status": STATUS_UNRESOLVED,
            "intrinsic_current_support": 0.42,
            "intrinsic_alternative_support": 0.31,
            "decision_gates": {
                "current_present": True,
                "current_absent": False,
                "alternative_evidence_localized_to_overlap": True,
                "alternative_evidence_external_to_overlap": False,
            },
        },
    )
    missing_patch_evidence = prior.candidate_evidence(**base)
    current_absent = prior.candidate_evidence(
        **base,
        candidate_refinement_evidence={
            "status": STATUS_UNRESOLVED,
            "intrinsic_current_support": -0.2,
            "intrinsic_alternative_support": 0.51,
            "decision_gates": {
                "current_present": False,
                "current_absent": True,
                "alternative_evidence_localized_to_overlap": True,
                "alternative_evidence_external_to_overlap": False,
            },
        },
    )
    external = prior.candidate_evidence(
        **base,
        candidate_refinement_evidence={
            "status": STATUS_UNRESOLVED,
            "intrinsic_current_support": 0.42,
            "intrinsic_alternative_support": 0.51,
            "decision_gates": {
                "current_present": True,
                "current_absent": False,
                "alternative_evidence_localized_to_overlap": False,
                "alternative_evidence_external_to_overlap": True,
            },
        },
    )

    assert supported["evidence_multiplier"] == 1.0
    assert supported["priority_adjustment"] > 0.0
    assert missing_patch_evidence["evidence_multiplier"] == 0.0
    assert missing_patch_evidence["evidence_multiplier_reason"] == (
        "patch_evidence_unavailable_no_overlap_decrease"
    )
    assert missing_patch_evidence["priority_adjustment"] == 0.0
    assert current_absent["evidence_multiplier"] == 0.0
    assert current_absent["priority_adjustment"] == 0.0
    assert external["evidence_multiplier"] == 0.0
    assert external["priority_adjustment"] == 0.0


def test_common_duplicate_pair_conflict_is_never_demoted_as_harmless_overlap():
    records = []
    for source_index in range(30):
        records.append(
            _overlap_prior_record(
                source_index, "bike", "Bike", [0, 0, 100, 100]
            )
        )
        if source_index < 22:
            records.append(
                _overlap_prior_record(
                    source_index, "person", "Person", [0, 0, 100, 100]
                )
            )
    prior = build_frequent_overlap_prior(records)
    pair_conflict = _affirmative_overlap_refinement_evidence(
        status=STATUS_PAIR_CONFLICT
    )
    evidence = prior.candidate_evidence(
        current_class="Bike",
        alternative_class="Person",
        query_source_key=f"sha256:{1:064x}",
        overlap_matches=[
            {
                "class_name": "Person",
                "iou": 1.0,
                "target_area_covered": 1.0,
                "other_area_covered": 1.0,
                "relation": "duplicate_like",
            }
        ],
        candidate_refinement_evidence=pair_conflict,
    )

    assert evidence["geometry_stratum"] == "duplicate_like"
    assert evidence["reliable"] is True
    assert evidence["evidence_multiplier"] == 0.0
    assert evidence["evidence_multiplier_reason"] == (
        "pair_conflict_no_overlap_decrease"
    )
    assert evidence["applies"] is False
    assert evidence["semantic_priority_adjustment"] == 0.0
    assert evidence["triage_contract"] == FREQUENT_OVERLAP_TRIAGE_CONTRACT
    assert evidence["triage_reliability_tier"] == "strong"
    assert evidence["triage_adjustment_eligible"] is True
    assert evidence["triage_applies"] is False
    assert evidence["triage_frequency_adjustment_reason"] == (
        "pair_conflict_no_triage_decrease"
    )
    assert evidence["triage_frequency_adjustment"] == 0.0
    assert evidence["priority_adjustment"] == 0.0
    assert pair_conflict["status"] == STATUS_PAIR_CONFLICT

    candidates = [
        {
            "point_id": "pair-conflict",
            "class_name": "Bike",
            "suggested_neighbor_class": "Person",
            "wrong_class_suspicion": 0.99,
            "split": "train",
            "image_relpath": "source-1.jpg",
                "refined_outlier": {
                    **pair_conflict,
                    "current_class": "Bike",
                    "alternative_class": "Person",
                    "overlap_relation": "duplicate_like",
                    "qualified_for_human_review": False,
                },
        },
        {
            "point_id": "unresolved-control",
            "class_name": "Bike",
            "suggested_neighbor_class": "Boat",
            "wrong_class_suspicion": 0.98,
            "split": "train",
            "image_relpath": "source-control.jpg",
            "refined_outlier": {
                "status": STATUS_UNRESOLVED,
                "current_class": "Bike",
                "alternative_class": "Boat",
                "qualified_for_human_review": False,
                "decision_gates": {},
            },
        },
    ]
    summary = api._class_analysis_assign_selector_priority_ranks(
        candidates,
        overlap_prior=prior,
        overlap_index={
            "pair-conflict": [
                {
                    "point_id": "overlap-person",
                    "class_name": "Person",
                    "iou": 1.0,
                    "target_area_covered": 1.0,
                    "other_area_covered": 1.0,
                    "relation": "duplicate_like",
                }
            ]
        },
        records_by_id={
            "pair-conflict": {
                "point_id": "pair-conflict",
                "_image_sha256": f"{2:064x}",
                "split": "train",
                "image_relpath": "source-1.jpg",
            }
        },
    )
    ranked_evidence = candidates[0]["refined_outlier"]
    assert summary["dataset_overlap_applied_candidate_count"] == 0
    assert summary["utility_model"][
        "dataset_overlap_scoring_effect_enabled"
    ] is True
    assert sorted(
        row["refined_outlier"]["selector_priority_rank"]
        for row in candidates
    ) == [1, 2]
    assert "selector_priority_overlap_adjustment" not in ranked_evidence
    assert (
        "selector_priority_triage_frequency_adjustment"
        not in ranked_evidence
    )
    assert ranked_evidence["selector_v6"]["overlap_evidence_state"] == (
        "duplicate_conflict"
    )
    assert ranked_evidence["selector_priority_score"] == pytest.approx(
        ranked_evidence["selector_v6"]["expected_review_utility"]
    )
    assert ranked_evidence["selector_v6"]["dataset_overlap"][
        "applied"
    ] is False
    assert ranked_evidence["selector_v6"]["dataset_overlap"][
        "utility_delta"
    ] == 0.0


def _provisional_overlap_prior(
    *, source_count: int = 120, overlap_source_count: int = 30
) -> tuple[FrequentOverlapPrior, str]:
    sources = [f"sha256:{index + 1:064x}" for index in range(source_count)]
    groups = [f"capture:{index + 1:064x}" for index in range(source_count)]
    source_to_group = dict(zip(sources, groups))
    return (
        FrequentOverlapPrior(
            class_source_object_counts={
                "ElevatedFixture": {group: 1 for group in groups},
                "Building": {
                    group: 1 for group in groups[:overlap_source_count]
                },
            },
            pair_source_overlap_object_counts={
                ("ElevatedFixture", "Building", "material_nonduplicate"): {
                    group: 1 for group in groups[:overlap_source_count]
                }
            },
            capture_groups=CaptureGroupIndex(
                source_to_group=source_to_group,
                group_tiers={
                    group: "provisional_unlineaged" for group in groups
                },
                group_methods={group: ("exact_content",) for group in groups},
                group_image_counts={group: 1 for group in groups},
                image_count=source_count,
                source_tiers={
                    source: "provisional_unlineaged" for source in sources
                },
                source_methods={
                    source: ("exact_content",) for source in sources
                },
            ),
            record_count=source_count + overlap_source_count,
            input_record_count=source_count + overlap_source_count,
            context_record_count=source_count + overlap_source_count,
            stage1_screened_point_id_count=(
                source_count + overlap_source_count
            ),
            stage1_screened_record_count=(
                source_count + overlap_source_count
            ),
            stage1_screened_point_id_digest=hashlib.sha256(
                b"provisional-overlap-prior-screened-fixture"
            ).hexdigest(),
        ),
        sources[0],
    )


def test_unresolved_provisional_overlap_uses_separate_triage_gate():
    prior, query_source = _provisional_overlap_prior()
    match = {
        "point_id": "building-overlap",
        "class_name": "Building",
        "iou": 0.65,
        "target_area_covered": 0.90,
        "other_area_covered": 0.75,
        "relation": "partial_contamination",
    }
    evidence = prior.candidate_evidence(
        current_class="ElevatedFixture",
        alternative_class="Building",
        query_source_key=query_source,
        overlap_matches=[match],
        candidate_refinement_evidence={
            "status": STATUS_UNRESOLVED,
            "qualified_for_human_review": False,
            "intrinsic_current_support": 0.1,
            "intrinsic_alternative_support": 0.2,
            "decision_gates": {
                "current_present": False,
                "current_absent": False,
                "alternative_evidence_localized_to_overlap": False,
                "alternative_evidence_external_to_overlap": False,
            },
        },
    )

    assert evidence["reliability_tier"] == "none"
    assert evidence["adjustment_eligible"] is False
    assert evidence["semantic_priority_adjustment"] == 0.0
    assert evidence["triage_reliability_tier"] == (
        "provisional_unlineaged"
    )
    assert evidence["triage_adjustment_eligible"] is True
    assert evidence["triage_candidate_annotated_overlap"] is True
    assert evidence["triage_eligible_capture_group_count"] == 119
    assert evidence["triage_overlap_capture_group_count"] == 29
    assert evidence["triage_source_incidence_wilson_lower_bound"] > 0.08
    assert evidence["maximum_triage_frequency_adjustment"] == 0.02
    assert evidence["triage_applies"] is True
    assert evidence["triage_frequency_adjustment"] == pytest.approx(
        0.02 * 0.90 * evidence["triage_conservative_prior_strength"]
    )
    assert evidence["priority_adjustment"] == evidence[
        "triage_frequency_adjustment"
    ]


@pytest.mark.parametrize(
    ("refinement_evidence", "overlap_matches", "reason"),
    [
        (
            {
                "status": STATUS_UNRESOLVED,
                "qualified_for_human_review": True,
                "decision_gates": {},
            },
            "match",
            "qualified_human_review_no_triage_decrease",
        ),
        (
            {
                "status": STATUS_UNRESOLVED,
                "qualified_for_human_review": False,
                "decision_gates": {"current_absent": True},
            },
            "match",
            "current_class_absent_no_triage_decrease",
        ),
        (
            {
                "status": STATUS_UNRESOLVED,
                "qualified_for_human_review": False,
                "decision_gates": {
                    "alternative_evidence_external_to_overlap": True
                },
            },
            "match",
            "alternative_external_no_triage_decrease",
        ),
        (
            {
                "status": STATUS_UNRESOLVED,
                "qualified_for_human_review": False,
                "decision_gates": {},
            },
            "none",
            "no_annotated_alternative_overlap_no_triage_decrease",
        ),
    ],
)
def test_frequency_triage_blocks_unsafe_or_non_geometry_rows(
    refinement_evidence, overlap_matches, reason
):
    prior, query_source = _provisional_overlap_prior()
    match = {
        "point_id": "building-overlap",
        "class_name": "Building",
        "iou": 0.65,
        "target_area_covered": 0.90,
        "other_area_covered": 0.75,
        "relation": "partial_contamination",
    }
    evidence = prior.candidate_evidence(
        current_class="ElevatedFixture",
        alternative_class="Building",
        query_source_key=query_source,
        overlap_matches=[match] if overlap_matches == "match" else [],
        candidate_refinement_evidence=refinement_evidence,
    )

    assert evidence["triage_applies"] is False
    assert evidence["triage_frequency_adjustment"] == 0.0
    assert evidence["triage_frequency_adjustment_reason"] == reason


def test_semantic_overlap_adjustment_suppresses_triage_double_counting():
    prior, query_source = _provisional_overlap_prior(
        source_count=240, overlap_source_count=100
    )
    evidence = prior.candidate_evidence(
        current_class="ElevatedFixture",
        alternative_class="Building",
        query_source_key=query_source,
        overlap_matches=[
            {
                "point_id": "building-overlap",
                "class_name": "Building",
                "iou": 0.65,
                "target_area_covered": 0.90,
                "other_area_covered": 0.75,
                "relation": "partial_contamination",
            }
        ],
        candidate_refinement_evidence=(
            _affirmative_overlap_refinement_evidence()
        ),
    )

    assert evidence["semantic_priority_adjustment"] > 0.0
    assert evidence["triage_adjustment_eligible"] is True
    assert evidence["triage_applies"] is False
    assert evidence["triage_frequency_adjustment"] == 0.0
    assert evidence["triage_frequency_adjustment_reason"] == (
        "semantic_overlap_adjustment_already_applies"
    )


def test_overlap_prior_fit_maps_are_frozen_after_construction():
    prior = build_frequent_overlap_prior(
        [_overlap_prior_record(0, "bike", "Bike", [0, 0, 100, 100])]
    )
    group = next(iter(prior.class_source_object_counts["Bike"]))

    with pytest.raises(TypeError):
        prior.class_source_object_counts["Bike"][group] = 999
    with pytest.raises(TypeError):
        prior.capture_groups.group_tiers[group] = "strong"


def test_overlap_prior_constructor_rejects_unbound_fit_counts():
    group = "capture:" + "1" * 64
    with pytest.raises(
        ValueError,
        match="frequent_overlap_prior_fit_counts_invalid",
    ):
        FrequentOverlapPrior(
            class_source_object_counts={"Bike": {group: 1}},
            pair_source_overlap_object_counts={},
            capture_groups=CaptureGroupIndex(
                source_to_group={},
                group_tiers={group: "strong"},
                group_methods={group: ("explicit_capture_group",)},
                group_image_counts={group: 1},
                image_count=1,
            ),
        )

    with pytest.raises(
        ValueError,
        match="frequent_overlap_prior_screened_point_id_digest_invalid",
    ):
        FrequentOverlapPrior(
            class_source_object_counts={"Bike": {group: 1}},
            pair_source_overlap_object_counts={},
            capture_groups=CaptureGroupIndex(
                source_to_group={},
                group_tiers={group: "strong"},
                group_methods={group: ("explicit_capture_group",)},
                group_image_counts={group: 1},
                image_count=1,
            ),
            record_count=1,
            input_record_count=1,
            context_record_count=1,
            stage1_screened_point_id_count=1,
            stage1_screened_record_count=1,
        )


def test_overlap_prior_rejects_pair_group_without_current_objects():
    group = "capture:" + "1" * 64
    prior = FrequentOverlapPrior(
        class_source_object_counts={"Bike": {}},
        pair_source_overlap_object_counts={
            ("Bike", "Person", "material_nonduplicate"): {group: 1}
        },
        capture_groups=CaptureGroupIndex(
            source_to_group={},
            group_tiers={group: "strong"},
            group_methods={group: ("explicit_capture_group",)},
            group_image_counts={group: 1},
            image_count=1,
        ),
    )

    with pytest.raises(
        ValueError,
        match="invalid_pair_group_without_current_class_objects",
    ):
        prior.summary()


def test_overlap_prior_registry_digest_is_order_stable_and_data_bound():
    records = []
    for source_index in range(6):
        records.append(
            _overlap_prior_record(
                source_index, "bike", "Bike", [0, 0, 100, 100]
            )
        )
        if source_index < 3:
            records.append(
                _overlap_prior_record(
                    source_index, "person", "Person", [10, 0, 90, 100]
                )
            )
    forward = build_frequent_overlap_prior(records)
    reversed_prior = build_frequent_overlap_prior(list(reversed(records)))
    changed = build_frequent_overlap_prior(
        [
            *records,
            _overlap_prior_record(5, "person", "Person", [10, 0, 90, 100]),
        ]
    )

    assert forward.fit_registry_digest() == reversed_prior.fit_registry_digest()
    assert forward.fit_registry_digest() != changed.fit_registry_digest()


def test_fit_digest_is_population_only_across_candidate_geometry_strata():
    records = []
    for source_index in range(25):
        records.append(
            _overlap_prior_record(
                source_index, "bike", "Bike", [0, 0, 100, 100]
            )
        )
        if source_index < 20:
            records.append(
                _overlap_prior_record(
                    source_index, "person", "Person", [10, 0, 90, 100]
                )
            )
    prior = build_frequent_overlap_prior(records)
    common = {
        "current_class": "Bike",
        "alternative_class": "Person",
        "query_source_key": f"sha256:{1:064x}",
        "candidate_refinement_evidence": (
            _affirmative_overlap_refinement_evidence()
        ),
    }
    material = prior.candidate_evidence(
        **common,
        overlap_matches=[
            {
                "class_name": "Person",
                "iou": 0.8,
                "target_area_covered": 0.8,
                "relation": "partial_contamination",
            }
        ],
    )
    duplicate = prior.candidate_evidence(
        **common,
        overlap_matches=[
            {
                "class_name": "Person",
                "iou": 1.0,
                "target_area_covered": 1.0,
                "relation": "duplicate_like",
            }
        ],
    )

    assert material["geometry_stratum"] == "material_nonduplicate"
    assert duplicate["geometry_stratum"] == "duplicate_like"
    assert material["fit_query_key"] == duplicate["fit_query_key"]
    assert material["fit_source_digest"] == duplicate["fit_source_digest"]


def test_capture_groups_count_images_not_annotation_rows():
    records = [
        _overlap_prior_record(
            0,
            f"bike-{object_index}",
            "Bike",
            [object_index, 0, object_index + 10, 10],
        )
        for object_index in range(200)
    ]
    prior = build_frequent_overlap_prior(records)

    summary = prior.summary()
    assert summary["input_annotation_record_count"] == 200
    assert summary["eligible_annotation_record_count"] == 200
    assert summary["capture_groups"]["image_count"] == 1
    assert summary["capture_groups"]["capture_group_count"] == 1
    assert list(prior.capture_groups.group_image_counts.values()) == [1]


def test_exporter_derivative_crops_share_one_strong_capture_group():
    records = []
    for crop_index in (1, 2):
        record = _overlap_prior_record(
            crop_index,
            "bike",
            "Bike",
            [0, 0, 100, 100],
        )
        record.pop("capture_group_id")
        record["image_relpath"] = (
            "synthetic-sequence_raw_images_"
            "00000000-0000-0000-0000-000000000000_"
            f"crop{crop_index}.jpg"
        )
        records.append(record)

    prior = build_frequent_overlap_prior(records)
    capture_summary = prior.capture_groups.summary()
    groups = set(prior.capture_groups.source_to_group.values())

    assert len(groups) == 1
    group = next(iter(groups))
    assert prior.capture_groups.group_tiers[group] == "strong"
    assert "exporter_parent" in prior.capture_groups.group_methods[group]
    assert capture_summary["image_count"] == 2
    assert capture_summary["capture_group_count"] == 1


def test_generic_directory_parent_is_not_strong_capture_lineage():
    records = []
    for source_index in range(3):
        record = _overlap_prior_record(
            source_index, "bike", "Bike", [0, 0, 100, 100]
        )
        record.pop("capture_group_id")
        record["image_relpath"] = f"nested-export/source-{source_index}.jpg"
        records.append(record)

    index = _build_capture_group_index(records)
    groups = set(index.source_to_group.values())

    assert len(groups) == 1
    group = next(iter(groups))
    assert index.group_tiers[group] == "provisional_unlineaged"
    assert "directory_parent" in index.group_methods[group]
    assert not {
        "explicit_capture_group",
        "exporter_parent",
        "sequence_filename",
    }.intersection(index.group_methods[group])


def test_overlap_prior_excludes_stage1_suspicious_labels_in_both_roles():
    records = []
    for source_index in range(22):
        records.extend(
            [
                _overlap_prior_record(
                    source_index, "bike", "Bike", [0, 0, 100, 100]
                ),
                _overlap_prior_record(
                    source_index, "person", "Person", [10, 0, 90, 100]
                ),
            ]
        )
    excluded_ids = {"p-0-bike", "p-1-person"}

    prior = build_frequent_overlap_prior(
        records,
        excluded_suspicious_point_ids=excluded_ids,
    )
    summary = prior.summary()
    evidence = prior.candidate_evidence(
        current_class="Bike",
        alternative_class="Person",
        query_source_key="sha256:" + "f" * 64,
        overlap_matches=[
            {
                "class_name": "Person",
                "iou": 0.8,
                "target_area_covered": 0.8,
                "relation": "partial_contamination",
            }
        ],
        candidate_refinement_evidence=(
            _affirmative_overlap_refinement_evidence()
        ),
    )

    assert summary["fit_eligibility_contract"] == (
        FREQUENT_OVERLAP_FIT_ELIGIBILITY_CONTRACT
    )
    assert summary["fit_requires_both_annotation_roles_trusted"] is True
    assert summary[
        "fit_requires_both_annotation_roles_stage1_screened"
    ] is True
    assert summary["fit_screening_scope"] == "all_classes"
    assert summary["fit_screening_exhaustive"] is True
    assert summary["stage1_screened_point_id_count"] == 44
    assert summary["stage1_screened_record_count"] == 44
    assert summary["fit_candidate_record_count"] == 44
    assert summary["excluded_unscreened_annotation_record_count"] == 0
    assert summary[
        "excluded_unusable_provenance_annotation_record_count"
    ] == 0
    assert summary["input_annotation_record_count"] == 44
    assert summary["context_annotation_record_count"] == 44
    assert summary["eligible_annotation_record_count"] == 42
    assert summary["excluded_suspicious_annotation_record_count"] == 2
    assert summary["excluded_directed_overlap_observation_count"] == 4
    assert evidence["eligible_capture_group_count"] == 21
    assert evidence["overlap_capture_group_count"] == 20
    assert prior.class_source_object_counts["Bike"] != (
        prior.class_source_object_counts["Person"]
    )


def test_selected_class_prior_ignores_unscreened_alternative_context():
    screened = [
        _overlap_prior_record(
            source_index,
            "pole",
            "ElevatedFixture",
            [0, 0, 100, 100],
        )
        for source_index in range(30)
    ]
    unscreened_alternatives = [
        _overlap_prior_record(
            source_index,
            "building",
            "Building",
            [10, 0, 90, 100],
        )
        for source_index in range(30)
    ]
    screened_ids = {record["point_id"] for record in screened}
    baseline = _build_frequent_overlap_prior(
        screened,
        trusted_screened_point_ids=screened_ids,
        fit_screening_scope="selected_class",
        fit_screening_exhaustive=True,
    )
    injected = _build_frequent_overlap_prior(
        [*screened, *unscreened_alternatives],
        trusted_screened_point_ids=screened_ids,
        fit_screening_scope="selected_class",
        fit_screening_exhaustive=True,
    )

    assert injected.fit_registry_digest() == baseline.fit_registry_digest()
    assert injected.class_source_object_counts == baseline.class_source_object_counts
    assert injected.pair_source_overlap_object_counts == {}
    summary = injected.summary()
    assert summary["fit_screening_scope"] == "selected_class"
    assert summary["fit_screening_exhaustive"] is True
    assert summary["fit_screening_adjustment_eligible"] is False
    assert summary["fit_screening_quality_gate"] == {
        "passed": False,
        "reason": "screening_scope_ineligible",
        "ordering_adjustments_enabled": False,
    }
    assert summary["stage1_screened_record_count"] == 30
    assert summary["excluded_unscreened_annotation_record_count"] == 30
    assert summary["eligible_annotation_record_count"] == 30
    evidence = injected.candidate_evidence(
        current_class="ElevatedFixture",
        alternative_class="Building",
        query_source_key="sha256:" + "f" * 64,
        overlap_matches=[
            {
                "class_name": "Building",
                "iou": 0.8,
                "target_area_covered": 0.8,
                "relation": "partial_contamination",
            }
        ],
        candidate_refinement_evidence=(
            _affirmative_overlap_refinement_evidence()
        ),
    )
    assert evidence["reliability_tier"] == "none"
    assert evidence["fit_screening_adjustment_eligible"] is False
    assert evidence["fit_screening_adjustment_reason"] == (
        "screening_scope_ineligible"
    )
    assert "screening_scope_ineligible" in evidence["reasons"]
    assert evidence["priority_adjustment"] == 0.0


def test_sample_capped_prior_ignores_unscreened_all_class_rows():
    screened = []
    unscreened = []
    for source_index in range(30):
        screened.extend(
            [
                _overlap_prior_record(
                    source_index, "bike", "Bike", [0, 0, 100, 100]
                ),
                _overlap_prior_record(
                    source_index,
                    "person",
                    "Person",
                    [10, 0, 90, 100],
                ),
            ]
        )
    for source_index in range(30, 130):
        unscreened.extend(
            [
                _overlap_prior_record(
                    source_index, "bike", "Bike", [0, 0, 100, 100]
                ),
                _overlap_prior_record(
                    source_index,
                    "person",
                    "Person",
                    [10, 0, 90, 100],
                ),
            ]
        )
    screened_ids = {record["point_id"] for record in screened}
    baseline = _build_frequent_overlap_prior(
        screened,
        trusted_screened_point_ids=screened_ids,
        fit_screening_scope="all_classes",
        fit_screening_exhaustive=False,
    )
    injected = _build_frequent_overlap_prior(
        [*screened, *unscreened],
        trusted_screened_point_ids=screened_ids,
        fit_screening_scope="all_classes",
        fit_screening_exhaustive=False,
    )

    assert injected.fit_registry_digest() == baseline.fit_registry_digest()
    assert injected.class_source_object_counts == baseline.class_source_object_counts
    assert (
        injected.pair_source_overlap_object_counts
        == baseline.pair_source_overlap_object_counts
    )
    summary = injected.summary()
    assert summary["fit_screening_exhaustive"] is False
    assert summary["fit_screening_adjustment_eligible"] is False
    assert summary["fit_screening_quality_gate"] == {
        "passed": False,
        "reason": "screening_scope_ineligible",
        "ordering_adjustments_enabled": False,
    }
    assert summary["stage1_screened_record_count"] == 60
    assert summary["excluded_unscreened_annotation_record_count"] == 200
    assert summary["eligible_annotation_record_count"] == 60

    evidence = injected.candidate_evidence(
        current_class="Bike",
        alternative_class="Person",
        query_source_key="sha256:" + "f" * 64,
        overlap_matches=[
            {
                "class_name": "Person",
                "iou": 0.8,
                "target_area_covered": 0.8,
                "relation": "partial_contamination",
            }
        ],
        candidate_refinement_evidence=(
            _affirmative_overlap_refinement_evidence()
        ),
    )
    assert evidence["reliability_tier"] == "strong"
    assert evidence["fit_screening_adjustment_eligible"] is False
    assert evidence["adjustment_eligible"] is False
    assert evidence["triage_adjustment_eligible"] is False
    assert evidence["fit_screening_adjustment_reason"] == (
        "screening_scope_ineligible"
    )
    assert evidence["priority_adjustment"] == 0.0

    changed_ids = {*screened_ids, unscreened[0]["point_id"]}
    changed = _build_frequent_overlap_prior(
        [*screened, *unscreened],
        trusted_screened_point_ids=changed_ids,
        fit_screening_scope="all_classes",
        fit_screening_exhaustive=False,
    )
    assert changed.fit_registry_digest() != injected.fit_registry_digest()


@pytest.mark.parametrize(
    ("fit_screening_scope", "fit_screening_exhaustive"),
    (("selected_class", True), ("all_classes", False)),
)
def test_ineligible_screening_cannot_apply_a_direct_priority_adjustment(
    fit_screening_scope,
    fit_screening_exhaustive,
):
    fit_records = []
    for source_index in range(30):
        fit_records.extend(
            [
                _overlap_prior_record(
                    source_index, "bike", "Bike", [0, 0, 100, 100]
                ),
                _overlap_prior_record(
                    source_index,
                    "person",
                    "Person",
                    [10, 0, 90, 100],
                ),
            ]
        )
    prior = _build_frequent_overlap_prior(
        fit_records,
        trusted_screened_point_ids={
            record["point_id"] for record in fit_records
        },
        fit_screening_scope=fit_screening_scope,
        fit_screening_exhaustive=fit_screening_exhaustive,
    )
    candidates = []
    records_by_id = {}
    for index, suspicion in enumerate((0.92, 0.88, 0.84)):
        point_id = f"candidate-{index}"
        candidate = {
            "point_id": point_id,
            "class_name": "Bike",
            "suggested_neighbor_class": "Person",
            "wrong_class_suspicion": suspicion,
            "refined_outlier": {
                "status": STATUS_UNRESOLVED,
                "current_class": "Bike",
                "alternative_class": "Person",
                "qualified_for_human_review": False,
                **_affirmative_overlap_refinement_evidence(),
            },
        }
        candidates.append(candidate)
        records_by_id[point_id] = {
            "point_id": point_id,
            "split": "train",
            "image_relpath": f"candidate-{index}.jpg",
            "_image_sha256": f"{1000 + index:064x}",
        }
    overlap_index = {
        "candidate-1": [
            {
                "point_id": "person-overlap",
                "class_name": "Person",
                "iou": 0.8,
                "target_area_covered": 0.8,
                "relation": "partial_contamination",
            }
        ]
    }

    summary = api._class_analysis_assign_selector_priority_ranks(
        candidates,
        overlap_prior=prior,
        overlap_index=overlap_index,
        records_by_id=records_by_id,
    )

    assert summary["dataset_overlap_applied_candidate_count"] == 0
    assert summary["frequent_overlap_prior"][
        "fit_screening_adjustment_eligible"
    ] is False
    for candidate in candidates:
        evidence = candidate["refined_outlier"]
        assert "selector_priority_overlap_adjustment" not in evidence
        assert "selector_priority_semantic_overlap_adjustment" not in evidence
        assert "selector_priority_triage_frequency_adjustment" not in evidence
        assert evidence["selector_priority_status_band_name"] == (
            "expected_review_utility"
        )
        assert evidence["selector_v6"]["dataset_overlap"]["applied"] is False
        assert evidence["selector_v6"]["dataset_overlap"][
            "utility_delta"
        ] == 0.0
        assert evidence["frequent_overlap_prior"][
            "fit_screening_adjustment_reason"
        ] == "screening_scope_ineligible"
        assert "ranked_by_expected_review_utility" in (
            evidence["selector_priority_reasons"]
        )


def test_overlap_prior_summary_counts_partition_screening_and_provenance():
    eligible = _overlap_prior_record(
        0, "eligible", "Bike", [0, 0, 100, 100]
    )
    suspicious = _overlap_prior_record(
        1, "rough", "Bike", [0, 0, 100, 100]
    )
    unusable = _overlap_prior_record(
        2, "unusable", "Bike", [0, 0, 100, 100]
    )
    unusable.pop("_image_sha256")
    unusable.pop("capture_group_id")
    unscreened = _overlap_prior_record(
        3, "context", "Person", [0, 0, 100, 100]
    )
    prior = _build_frequent_overlap_prior(
        [eligible, suspicious, unusable, unscreened],
        trusted_screened_point_ids={
            eligible["point_id"],
            suspicious["point_id"],
            unusable["point_id"],
        },
        excluded_suspicious_point_ids={suspicious["point_id"]},
    )

    summary = prior.summary()
    assert summary["context_annotation_record_count"] == 4
    assert summary["stage1_screened_record_count"] == 3
    assert summary["excluded_unscreened_annotation_record_count"] == 1
    assert summary[
        "excluded_unusable_provenance_annotation_record_count"
    ] == 1
    assert summary["fit_candidate_record_count"] == 2
    assert summary["excluded_suspicious_annotation_record_count"] == 1
    assert summary["eligible_annotation_record_count"] == 1


def test_explicit_capture_group_is_not_split_by_synchronized_camera():
    records = []
    for source_index, camera_id in enumerate(("left", "right")):
        record = _overlap_prior_record(
            source_index, "bike", "Bike", [0, 0, 100, 100]
        )
        record["capture_group_id"] = "simultaneous-capture-7"
        record["camera_id"] = camera_id
        records.append(record)

    index = _build_capture_group_index(records)
    assert len(set(index.source_to_group.values())) == 1
    group = next(iter(index.source_to_group.values()))
    assert index.group_tiers[group] == "strong"


def _bare_uuid_overlap_record(
    source_index: int,
    point_suffix: str,
    class_name: str,
    bbox: list[float],
) -> dict:
    record = _overlap_prior_record(
        source_index, point_suffix, class_name, bbox
    )
    record.pop("capture_group_id")
    record["image_relpath"] = (
        f"00000000-0000-0000-0000-{source_index + 1:012x}.jpg"
    )
    return record


def test_unlineaged_uuid_groups_are_provisional_and_strictly_capped():
    too_small = []
    for source_index in range(30):
        too_small.append(
            _bare_uuid_overlap_record(
                source_index, "bike", "Bike", [0, 0, 100, 100]
            )
        )
        if source_index < 20:
            too_small.append(
                _bare_uuid_overlap_record(
                    source_index,
                    "person",
                    "Person",
                    [10, 0, 90, 100],
                )
            )
    small_prior = build_frequent_overlap_prior(too_small)
    small = small_prior.candidate_evidence(
        current_class="Bike",
        alternative_class="Person",
        query_source_key=f"sha256:{1:064x}",
        overlap_matches=[
            {
                "class_name": "Person",
                "iou": 0.8,
                "target_area_covered": 0.8,
                "relation": "partial_contamination",
            }
        ],
    )
    assert small["reliability_tier"] == "none"
    assert small["adjustment_eligible"] is False
    assert small["priority_adjustment"] == 0.0
    provisional_diagnostic = next(
        row
        for row in small["cohort_diagnostics"]
        if row["reliability_tier"] == "provisional_unlineaged"
    )
    assert provisional_diagnostic["eligible_capture_group_count"] == 29
    assert provisional_diagnostic["passes"] is False

    records = []
    for source_index in range(220):
        records.append(
            _bare_uuid_overlap_record(
                source_index, "bike", "Bike", [0, 0, 100, 100]
            )
        )
        if source_index < 110:
            records.append(
                _bare_uuid_overlap_record(
                    source_index,
                    "person",
                    "Person",
                    [10, 0, 90, 100],
                )
            )
    prior = build_frequent_overlap_prior(records)
    evidence = prior.candidate_evidence(
        current_class="Bike",
        alternative_class="Person",
        query_source_key=f"sha256:{1:064x}",
        overlap_matches=[
            {
                "class_name": "Person",
                "iou": 0.8,
                "target_area_covered": 0.8,
                "relation": "partial_contamination",
            }
        ],
        candidate_refinement_evidence=(
            _affirmative_overlap_refinement_evidence()
        ),
    )

    assert evidence["candidate_capture_group_tier"] == (
        "provisional_unlineaged"
    )
    assert evidence["reliability_tier"] == "provisional_unlineaged"
    assert evidence["source_independence_verified"] is False
    assert evidence["provisional"] is True
    assert evidence["reliable"] is False
    assert evidence["adjustment_eligible"] is True
    assert evidence["applies"] is True
    assert 0.0 < evidence["priority_adjustment"] <= 0.08


def test_content_bound_perceptual_hash_can_only_create_lower_tier_group():
    records = []
    signatures = ("0" * 32, "0" * 31 + "1")
    for source_index, signature in enumerate(signatures):
        record = _bare_uuid_overlap_record(
            source_index, "bike", "Bike", [0, 0, 100, 100]
        )
        record["capture_perceptual_hash"] = signature
        record["capture_perceptual_image_sha256"] = record[
            "_image_sha256"
        ]
        record["_image_width"] = 100
        record["_image_height"] = 100
        records.append(record)
    invalid = _bare_uuid_overlap_record(
        2, "bike", "Bike", [0, 0, 100, 100]
    )
    invalid["capture_perceptual_hash"] = "0" * 32
    invalid["capture_perceptual_image_sha256"] = "f" * 64
    records.append(invalid)

    index = _build_capture_group_index(records)
    first = index.source_to_group[f"sha256:{1:064x}"]
    second = index.source_to_group[f"sha256:{2:064x}"]
    invalid_group = index.source_to_group[f"sha256:{3:064x}"]

    assert first == second
    assert index.group_tiers[first] == "lower_confidence"
    assert "perceptual_near_duplicate" in index.group_methods[first]
    assert invalid_group != first
    assert index.group_tiers[invalid_group] == "provisional_unlineaged"
    assert index.source_tiers[f"sha256:{3:064x}"] == (
        "provisional_unlineaged"
    )


def test_unlineaged_member_of_strong_group_downgrades_fit_group_and_keeps_local_tier():
    strong = _bare_uuid_overlap_record(
        0, "bike", "Bike", [0, 0, 100, 100]
    )
    strong["capture_group_id"] = "attested-capture"
    unlineaged = _bare_uuid_overlap_record(
        1, "bike", "Bike", [0, 0, 100, 100]
    )
    for record, signature in (
        (strong, "0" * 32),
        (unlineaged, "0" * 31 + "1"),
    ):
        record["capture_perceptual_hash"] = signature
        record["capture_perceptual_image_sha256"] = record["_image_sha256"]
        record["_image_width"] = 100
        record["_image_height"] = 100

    prior = build_frequent_overlap_prior([strong, unlineaged])
    index = prior.capture_groups
    strong_source = f"sha256:{1:064x}"
    unlineaged_source = f"sha256:{2:064x}"
    group = index.source_to_group[strong_source]

    assert index.source_to_group[unlineaged_source] == group
    assert index.group_tiers[group] == "lower_confidence"
    assert index.source_tiers[strong_source] == "strong"
    assert index.source_tiers[unlineaged_source] == "lower_confidence"
    evidence = prior.candidate_evidence(
        current_class="Bike",
        alternative_class="Person",
        query_source_key=unlineaged_source,
        overlap_matches=[],
    )
    assert evidence["candidate_capture_group_tier"] == "lower_confidence"
    assert "explicit_capture_group" not in evidence[
        "candidate_capture_group_methods"
    ]


def test_perceptually_joined_unlineaged_overlap_cannot_create_strong_prior():
    records = []
    for source_index in range(25):
        strong = _bare_uuid_overlap_record(
            source_index * 2, "bike", "Bike", [0, 0, 100, 100]
        )
        strong["capture_group_id"] = f"attested-{source_index}"
        unlineaged = _bare_uuid_overlap_record(
            source_index * 2 + 1,
            "person",
            "Person",
            [10, 0, 90, 100],
        )
        signature = f"{source_index:032x}"
        for record in (strong, unlineaged):
            record["capture_perceptual_hash"] = signature
            record["capture_perceptual_image_sha256"] = record[
                "_image_sha256"
            ]
            record["_image_width"] = 100
            record["_image_height"] = 100
        records.extend((strong, unlineaged))

    prior = build_frequent_overlap_prior(records)
    evidence = prior.candidate_evidence(
        current_class="Bike",
        alternative_class="Person",
        query_source_key="sha256:" + "f" * 64,
        overlap_matches=[
            {
                "class_name": "Person",
                "iou": 0.8,
                "target_area_covered": 0.8,
                "relation": "partial_contamination",
            }
        ],
        candidate_refinement_evidence=(
            _affirmative_overlap_refinement_evidence()
        ),
    )

    assert set(prior.capture_groups.group_tiers.values()) == {
        "lower_confidence"
    }
    strong_diagnostic = next(
        row
        for row in evidence["cohort_diagnostics"]
        if row["reliability_tier"] == "strong"
    )
    assert strong_diagnostic["eligible_capture_group_count"] == 0
    assert strong_diagnostic["passes"] is False
    assert evidence["reliability_tier"] != "strong"


def test_candidate_leave_one_out_excludes_complete_capture_group():
    records = []
    for source_index in range(22):
        capture_index = 0 if source_index < 2 else source_index - 1
        record = _overlap_prior_record(
            source_index, "bike", "Bike", [0, 0, 100, 100]
        )
        record["capture_group_id"] = f"capture-{capture_index}"
        records.append(record)
        person = _overlap_prior_record(
            source_index, "person", "Person", [10, 0, 90, 100]
        )
        person["capture_group_id"] = f"capture-{capture_index}"
        records.append(person)
    prior = build_frequent_overlap_prior(records)
    evidence = prior.candidate_evidence(
        current_class="Bike",
        alternative_class="Person",
        query_source_key=f"sha256:{1:064x}",
        overlap_matches=[
            {
                "class_name": "Person",
                "iou": 0.8,
                "target_area_covered": 0.8,
                "relation": "partial_contamination",
            }
        ],
    )

    assert prior.capture_groups.summary()["image_count"] == 22
    assert prior.capture_groups.summary()["capture_group_count"] == 21
    assert evidence["candidate_capture_group_excluded"] is True
    assert evidence["eligible_capture_group_count"] == 20
    assert evidence["overlap_capture_group_count"] == 20


def test_overlap_prior_tiers_are_not_pooled_for_reliability():
    groups = {}
    methods = {}
    counts = {}
    class_counts = {"Bike": {}}
    pair_counts = {("Bike", "Person", "material_nonduplicate"): {}}
    for tier, count in (
        ("strong", 19),
        ("lower_confidence", 79),
        ("provisional_unlineaged", 199),
    ):
        for index in range(count):
            group = f"{tier}-{index}"
            groups[group] = tier
            methods[group] = (tier,)
            counts[group] = 1
            class_counts["Bike"][group] = 1
            pair_counts[("Bike", "Person", "material_nonduplicate")][
                group
            ] = 1
    prior = FrequentOverlapPrior(
        class_source_object_counts=class_counts,
        pair_source_overlap_object_counts=pair_counts,
        capture_groups=CaptureGroupIndex(
            source_to_group={},
            group_tiers=groups,
            group_methods=methods,
            group_image_counts=counts,
            image_count=sum(counts.values()),
        ),
        record_count=sum(counts.values()),
        input_record_count=sum(counts.values()),
        context_record_count=sum(counts.values()),
        stage1_screened_point_id_count=sum(counts.values()),
        stage1_screened_record_count=sum(counts.values()),
        stage1_screened_point_id_digest=hashlib.sha256(
            b"tiered-overlap-prior-screened-fixture"
        ).hexdigest(),
    )
    evidence = prior.candidate_evidence(
        current_class="Bike",
        alternative_class="Person",
        query_source_key="",
        overlap_matches=[
            {
                "class_name": "Person",
                "iou": 0.8,
                "target_area_covered": 0.8,
                "relation": "partial_contamination",
            }
        ],
    )

    assert evidence["reliability_tier"] == "none"
    assert evidence["adjustment_eligible"] is False
    assert evidence["priority_adjustment"] == 0.0
    assert {
        row["reliability_tier"]: row["eligible_capture_group_count"]
        for row in evidence["cohort_diagnostics"]
    } == {
        "strong": 19,
        "lower_confidence": 79,
        "provisional_unlineaged": 199,
    }


def test_overlap_prior_cooperative_hooks_cover_cancel_progress_and_observation():
    record = _overlap_prior_record(
        0, "bike", "Bike", [0, 0, 100, 100]
    )
    with pytest.raises(RuntimeError, match="cancelled"):
        build_frequent_overlap_prior(
            [record], should_cancel=lambda: True
        )

    progress = []
    memory_observations = []
    prior = build_frequent_overlap_prior(
        [record],
        should_cancel=lambda: False,
        progress_callback=lambda phase, processed, total: progress.append(
            (phase, processed, total)
        ),
        memory_check=lambda: memory_observations.append(True),
    )
    assert prior.record_count == 1
    assert memory_observations
    assert {phase for phase, _processed, _total in progress} >= {
        "scan_records",
        "capture_groups",
        "overlap_geometry",
    }

    def observation_failure():
        raise RuntimeError("resource_observer_failed")

    with pytest.raises(RuntimeError, match="resource_observer_failed"):
        build_frequent_overlap_prior(
            [record], memory_check=observation_failure
        )


def test_v6_selector_priority_ranks_every_row_without_legacy_adjustments():
    class _Prior:
        def summary(self):
            return {"contract": FREQUENT_OVERLAP_PRIOR_CONTRACT}

        def candidate_evidence(self, **kwargs):
            applies = kwargs["alternative_class"] == "Person"
            return {
                "contract": FREQUENT_OVERLAP_PRIOR_CONTRACT,
                "reliable": applies,
                "applies": applies,
                "semantic_priority_adjustment": (
                    0.30 if applies else 0.0
                ),
                "triage_frequency_adjustment": 0.0,
                "priority_adjustment": 0.30 if applies else 0.0,
                "reasons": [
                    "common_overlap_priority_decrease"
                    if applies
                    else "selector_priority_unchanged_by_overlap_prior"
                ],
            }

    candidates = []
    for index, suspicion in enumerate((0.99, 0.98, 0.97, 0.96, 0.95, 0.94)):
        candidates.append(
            {
                "point_id": f"candidate-{index}",
                "class_name": "Bike",
                "suggested_neighbor_class": "Person" if index == 0 else "Boat",
                "wrong_class_suspicion": suspicion,
                "split": "train",
                "image_relpath": f"candidate-{index}.jpg",
                "refined_outlier": {
                    "status": STATUS_UNRESOLVED,
                    "current_class": "Bike",
                    "alternative_class": "Person" if index == 0 else "Boat",
                    "qualified_for_human_review": True,
                    "directed_pair_probe_score": 0.5,
                    "directed_pair_probe_threshold": 0.2,
                },
            }
        )
    original_order = [row["point_id"] for row in candidates]
    assert api._class_analysis_assign_human_review_ranks(candidates) == 6
    original_human_ranks = [
        row["refined_outlier"]["human_review_rank"] for row in candidates
    ]

    summary = api._class_analysis_assign_selector_priority_ranks(
        candidates,
        overlap_prior=_Prior(),
        overlap_index={
            "candidate-0": [
                {
                    "class_name": "Person",
                    "relation": "partial_contamination",
                    "target_area_covered": 0.9,
                    "iou": 0.6,
                }
            ]
        },
    )

    assert [row["point_id"] for row in candidates] == original_order
    assert summary["contract"] == SELECTOR_PRIORITY_CONTRACT
    assert summary["candidate_count"] == 6
    assert summary["ranked_candidate_count"] == 6
    assert summary["dataset_overlap_applied_candidate_count"] == 0
    ranks = [
        row["refined_outlier"]["selector_priority_rank"]
        for row in candidates
    ]
    assert sorted(ranks) == [1, 2, 3, 4, 5, 6]
    assert [
        row["refined_outlier"]["human_review_rank"] for row in candidates
    ] == original_human_ranks
    assert (
        "selector_priority_overlap_adjustment"
        not in candidates[0]["refined_outlier"]
    )
    assert candidates[0]["refined_outlier"]["selector_v6"][
        "selector_contract"
    ] == SELECTOR_PRIORITY_CONTRACT
    assert candidates[0]["refined_outlier"]["selector_v6"][
        "dataset_overlap_scoring_effect_enabled"
    ] is True
    assert candidates[0]["refined_outlier"]["selector_v6"][
        "dataset_overlap"
    ]["rank_only"] is True


def test_selector_priority_uses_one_global_expected_utility_order():
    class _Prior:
        def summary(self):
            return {"contract": FREQUENT_OVERLAP_PRIOR_CONTRACT}

        def candidate_evidence(self, **kwargs):
            penalized = kwargs["current_class"] == "Penalized"
            return {
                "contract": FREQUENT_OVERLAP_PRIOR_CONTRACT,
                "reliable": penalized,
                "applies": penalized,
                "semantic_priority_adjustment": (
                    0.35 if penalized else 0.0
                ),
                "triage_frequency_adjustment": 0.0,
                "priority_adjustment": 0.35 if penalized else 0.0,
                "reasons": [],
            }

    candidates = [
        {
            "point_id": "pair-first",
            "wrong_class_suspicion": 0.99,
            "refined_outlier": {
                "status": STATUS_PAIR_CONFLICT,
                "current_class": "Unpenalized",
                    "alternative_class": "Person",
                    "qualified_for_human_review": False,
                    "overlap_object_count": 0,
                    "annotated_overlap_alternative_bbox_xyxy": None,
                },
        },
        {
            "point_id": "pair-last-penalized",
            "wrong_class_suspicion": 0.01,
            "refined_outlier": {
                "status": STATUS_PAIR_CONFLICT,
                "current_class": "Penalized",
                "alternative_class": "Person",
                "qualified_for_human_review": False,
            },
        },
        {
            "point_id": "mixed-first",
            "wrong_class_suspicion": 1.0,
            "refined_outlier": {
                "status": STATUS_MIXED_OR_COMPOSITE,
                "current_class": "Unpenalized",
                "alternative_class": "Person",
                "qualified_for_human_review": False,
            },
        },
    ]

    summary = api._class_analysis_assign_selector_priority_ranks(
        candidates, overlap_prior=_Prior()
    )
    evidence = {
        row["point_id"]: row["refined_outlier"] for row in candidates
    }

    assert summary["status_band_partitioned"] is False
    assert summary["cross_status_band_reordering"] is True
    assert sorted(
        row["selector_priority_rank"] for row in evidence.values()
    ) == [1, 2, 3]
    assert {
        row["selector_priority_status_band_name"]
        for row in evidence.values()
    } == {"expected_review_utility"}
    assert all(
        "selector_priority_overlap_adjustment" not in row
        and row["selector_v6"]["dataset_overlap"]["applied"] is False
        and row["selector_v6"]["dataset_overlap"]["utility_delta"] == 0.0
        for row in evidence.values()
    )


def test_v6_priority_has_no_semantic_status_reservation():
    candidates = []
    for index in range(8):
        candidates.append(
            {
                "point_id": f"pair-{index}",
                "wrong_class_suspicion": 0.20 - index * 0.001,
                "refined_outlier": {
                    "status": STATUS_PAIR_CONFLICT,
                    "current_class": "Bike",
                    "alternative_class": "Person",
                    "qualified_for_human_review": False,
                },
            }
        )
    for index, status in enumerate(
        (
            STATUS_UNRESOLVED,
            STATUS_MIXED_OR_COMPOSITE,
            STATUS_UNRESOLVED,
            STATUS_MIXED_OR_COMPOSITE,
        )
    ):
        candidates.append(
            {
                "point_id": f"review-{index}",
                "wrong_class_suspicion": 0.95 - index * 0.01,
                "refined_outlier": {
                    "status": status,
                    "current_class": "ElevatedFixture",
                    "alternative_class": "Building",
                    "qualified_for_human_review": False,
                },
            }
        )

    summary = api._class_analysis_assign_selector_priority_ranks(candidates)
    assert summary["status_band_counts"]["expected_review_utility"] == 12
    assert summary["status_band_partitioned"] is False
    assert summary["cross_status_band_reordering"] is True
    assert {
        row["refined_outlier"]["selector_priority_status_band_name"]
        for row in candidates
    } == {"expected_review_utility"}
    assert {
        row["refined_outlier"]["selector_priority_status_band_index"]
        for row in candidates
    } == {0}


def test_selector_prior_evaluation_failure_is_observable_but_score_neutral():
    record = _overlap_prior_record(
        0, "bike", "Bike", [0, 0, 100, 100]
    )
    prior = build_frequent_overlap_prior([record])

    class _FailingPrior:
        def summary(self):
            return prior.summary()

        def candidate_evidence(self, **_kwargs):
            raise RuntimeError("synthetic_prior_failure")

    candidate = {
        "point_id": record["point_id"],
        "class_name": "Bike",
        "suggested_neighbor_class": "Person",
        "wrong_class_suspicion": 0.9,
        "split": "train",
        "image_relpath": record["image_relpath"],
        "refined_outlier": {
            "schema": REFINEMENT_SCHEMA,
            "decision_contract": REFINEMENT_DECISION_CONTRACT,
            "status": STATUS_UNRESOLVED,
            "current_class": "Bike",
            "alternative_class": "Person",
            "qualified_for_human_review": False,
            "overlap_object_count": 0,
            "annotated_overlap_alternative_bbox_xyxy": None,
        },
    }
    summary = api._class_analysis_assign_selector_priority_ranks(
        [candidate],
        overlap_prior=_FailingPrior(),
        records_by_id={record["point_id"]: record},
    )
    prior_evidence = candidate["refined_outlier"][
        "frequent_overlap_prior"
    ]

    assert summary["prior_evaluation_failure_count"] == 1
    assert summary["prior_evaluation_failures"] == [
        {
            "point_id": record["point_id"],
            "reason": "RuntimeError:synthetic_prior_failure",
        }
    ]
    assert summary["quality_gate"] == {
        "passed": False,
        "reasons": ["frequent_overlap_prior_evaluation_failed"],
    }
    assert prior_evidence["evidence_multiplier_reason"] == (
        "frequent_overlap_prior_evaluation_failed"
    )
    assert prior_evidence["evaluation_failure_reason"] == (
        "RuntimeError:synthetic_prior_failure"
    )
    assert prior_evidence["priority_adjustment"] == 0.0

    refinement = {
        "selector_priority_contract": SELECTOR_PRIORITY_CONTRACT,
        "selector_priority": summary,
        "selector_priority_candidate_count": 1,
    }
    api._class_analysis_validate_selector_priority_artifact(
        refinement=refinement,
        refinement_rows=[candidate],
        configured_contract=SELECTOR_PRIORITY_CONTRACT,
    )

    diagnostic_only = copy.deepcopy(refinement)
    diagnostic_only["selector_priority"]["prior_evaluation_failure_count"] = 0
    diagnostic_only["selector_priority"]["prior_evaluation_failures"] = []
    diagnostic_only["selector_priority"]["quality_gate"] = {
        "passed": True,
        "reasons": [],
    }
    # V6 binds and validates the score-bearing selector_v6 payload. Historical
    # overlap-prior failure diagnostics remain observable, but they have zero
    # scoring effect and are deliberately outside the rank-validity contract.
    api._class_analysis_validate_selector_priority_artifact(
        refinement=diagnostic_only,
        refinement_rows=[candidate],
        configured_contract=SELECTOR_PRIORITY_CONTRACT,
    )
    assert candidate["refined_outlier"]["selector_v6"][
        "dataset_overlap_scoring_effect_enabled"
    ] is True
    assert candidate["refined_outlier"]["selector_v6"]["dataset_overlap"][
        "utility_delta"
    ] == 0.0


@pytest.mark.parametrize("failure_call", (1, 2))
def test_selector_prior_summary_failure_aborts_publication(failure_call):
    class _SummaryFailingPrior:
        def __init__(self):
            self.calls = 0

        def summary(self):
            self.calls += 1
            if self.calls == failure_call:
                raise RuntimeError("synthetic_summary_failure")
            return {"contract": FREQUENT_OVERLAP_PRIOR_CONTRACT}

        def candidate_evidence(self, **_kwargs):
            return {
                "contract": FREQUENT_OVERLAP_PRIOR_CONTRACT,
                "reliable": False,
                "applies": False,
                "semantic_priority_adjustment": 0.0,
                "triage_frequency_adjustment": 0.0,
                "priority_adjustment": 0.0,
                "reasons": [
                    "selector_priority_unchanged_by_overlap_prior"
                ],
            }

    candidate = {
        "point_id": "summary-failure-candidate",
        "wrong_class_suspicion": 0.9,
        "refined_outlier": {
            "status": STATUS_UNRESOLVED,
            "current_class": "Bike",
            "alternative_class": "Person",
            "qualified_for_human_review": False,
        },
    }
    expected_phase = "initial" if failure_call == 1 else "final"

    with pytest.raises(
        RuntimeError,
        match=(
            "class_analysis_overlap_prior_summary_failed:"
            f"{expected_phase}:RuntimeError:synthetic_summary_failure"
        ),
    ):
        api._class_analysis_assign_selector_priority_ranks(
            [candidate], overlap_prior=_SummaryFailingPrior()
        )


def test_overlap_prior_and_selector_priority_are_input_order_deterministic():
    records = []
    for source_index in range(30):
        records.extend(
            [
                _overlap_prior_record(
                    source_index, "bike", "Bike", [0, 0, 100, 100]
                ),
                _overlap_prior_record(
                    source_index, "person", "Person", [10, 0, 90, 100]
                ),
            ]
        )
    prior_a = build_frequent_overlap_prior(records)
    prior_b = build_frequent_overlap_prior(list(reversed(records)))
    match = {
        "class_name": "Person",
        "iou": 0.8,
        "target_area_covered": 0.8,
        "other_area_covered": 1.0,
        "relation": "partial_contamination",
    }
    patch_evidence = {
        "status": STATUS_UNRESOLVED,
        "decision_gates": {
            "current_present": True,
            "current_absent": False,
            "alternative_evidence_localized_to_overlap": True,
            "alternative_evidence_external_to_overlap": False,
        },
    }
    kwargs = {
        "current_class": "Bike",
        "alternative_class": "Person",
        "query_source_key": f"sha256:{1:064x}",
        "overlap_matches": [match],
        "candidate_refinement_evidence": patch_evidence,
    }
    assert prior_a.candidate_evidence(**kwargs) == prior_b.candidate_evidence(
        **kwargs
    )

    candidates = []
    records_by_id = {}
    overlap_index = {}
    for index in range(8):
        point_id = f"rank-{index}"
        alternative = "Person" if index == 0 else "Boat"
        candidates.append(
            {
                "point_id": point_id,
                "class_name": "Bike",
                "suggested_neighbor_class": alternative,
                "wrong_class_suspicion": 0.9,
                "split": "train",
                "image_relpath": f"rank-{index}.jpg",
                "refined_outlier": {
                    **copy.deepcopy(patch_evidence),
                    "current_class": "Bike",
                    "alternative_class": alternative,
                    "qualified_for_human_review": False,
                    "directed_pair_probe_score": 0.0,
                    "directed_pair_probe_threshold": 0.0,
                },
            }
        )
        records_by_id[point_id] = {
            "point_id": point_id,
            "split": "train",
            "image_relpath": f"rank-{index}.jpg",
            "_image_sha256": f"{index + 100:064x}",
        }
    overlap_index["rank-0"] = [match]
    forward = copy.deepcopy(candidates)
    reverse = list(reversed(copy.deepcopy(candidates)))
    api._class_analysis_assign_selector_priority_ranks(
        forward,
        overlap_prior=prior_a,
        overlap_index=overlap_index,
        records_by_id=records_by_id,
    )
    api._class_analysis_assign_selector_priority_ranks(
        reverse,
        overlap_prior=prior_b,
        overlap_index=overlap_index,
        records_by_id=records_by_id,
    )
    fields = (
        "selector_priority_base_rank",
        "selector_priority_base_score",
        "selector_priority_score",
        "selector_priority_rank",
        "selector_v6",
    )
    forward_by_id = {
        row["point_id"]: {
            field: row["refined_outlier"][field] for field in fields
        }
        for row in forward
    }
    reverse_by_id = {
        row["point_id"]: {
            field: row["refined_outlier"][field] for field in fields
        }
        for row in reverse
    }
    assert forward_by_id == reverse_by_id
    assert sorted(
        row["selector_priority_rank"] for row in forward_by_id.values()
    ) == list(range(1, 9))


def test_persisted_selector_priority_validation_fails_closed_on_corruption():
    candidates = [
        {
            "point_id": f"persisted-{index}",
            "class_name": "Bike",
            "suggested_neighbor_class": "Person",
            "wrong_class_suspicion": 0.9 - index * 0.1,
            "refined_outlier": {
                "schema": REFINEMENT_SCHEMA,
                "decision_contract": REFINEMENT_DECISION_CONTRACT,
                "status": STATUS_UNRESOLVED,
                "current_class": "Bike",
                "alternative_class": "Person",
                "qualified_for_human_review": False,
                "overlap_object_count": 0,
                "annotated_overlap_alternative_bbox_xyxy": None,
            },
        }
        for index in range(3)
    ]
    selector_summary = api._class_analysis_assign_selector_priority_ranks(
        candidates
    )
    refinement = {
        "selector_priority_contract": SELECTOR_PRIORITY_CONTRACT,
        "selector_priority": selector_summary,
        "selector_priority_candidate_count": len(candidates),
    }
    api._class_analysis_validate_selector_priority_artifact(
        refinement=refinement,
        refinement_rows=candidates,
        configured_contract=SELECTOR_PRIORITY_CONTRACT,
    )
    assert candidates[0]["refined_outlier"][
        "selector_priority_base_score"
    ] == pytest.approx(
        candidates[0]["refined_outlier"]["selector_v6"][
            "base_expected_review_utility"
        ]
    )
    assert (
        "selector_priority_overlap_adjustment"
        not in candidates[0]["refined_outlier"]
    )

    duplicate_rank_rows = copy.deepcopy(candidates)
    duplicate_rank_rows[1]["refined_outlier"]["selector_priority_rank"] = 1
    duplicate_rank_rows[1]["refined_outlier"][
        "selector_priority_band_rank"
    ] = 1
    with pytest.raises(ValueError, match="ranks_not_unique_contiguous"):
        api._class_analysis_validate_selector_priority_artifact(
            refinement=refinement,
            refinement_rows=duplicate_rank_rows,
            configured_contract=SELECTOR_PRIORITY_CONTRACT,
        )

    corrupt_score_rows = copy.deepcopy(candidates)
    corrupt_score_rows[0]["refined_outlier"]["selector_priority_score"] += 0.01
    with pytest.raises(ValueError, match="row_utility_arithmetic"):
        api._class_analysis_validate_selector_priority_artifact(
            refinement=refinement,
            refinement_rows=corrupt_score_rows,
            configured_contract=SELECTOR_PRIORITY_CONTRACT,
        )

    corrupt_model_binding = copy.deepcopy(candidates)
    corrupt_model_binding[0]["refined_outlier"]["selector_v6"][
        "model_digest"
    ] = "0" * 64
    with pytest.raises(ValueError, match="row_contract"):
        api._class_analysis_validate_selector_priority_artifact(
            refinement=refinement,
            refinement_rows=corrupt_model_binding,
            configured_contract=SELECTOR_PRIORITY_CONTRACT,
        )

    corrupt_summary = copy.deepcopy(refinement)
    corrupt_summary["selector_priority_candidate_count"] = 2
    with pytest.raises(ValueError, match="summary_counts_or_invariants"):
        api._class_analysis_validate_selector_priority_artifact(
            refinement=corrupt_summary,
            refinement_rows=candidates,
            configured_contract=SELECTOR_PRIORITY_CONTRACT,
        )

    corrupt_utility_rows = copy.deepcopy(candidates)
    corrupt_utility_rows[0]["refined_outlier"]["selector_v6"][
        "reviewability_probability"
    ] = 0.0
    with pytest.raises(ValueError, match="row_utility_arithmetic"):
        api._class_analysis_validate_selector_priority_artifact(
            refinement=refinement,
            refinement_rows=corrupt_utility_rows,
            configured_contract=SELECTOR_PRIORITY_CONTRACT,
        )


def _selector_priority_validation_fixture_with_overlap_effects():
    records = []
    for source_index in range(30):
        records.extend(
            [
                _overlap_prior_record(
                    source_index, "bike", "Bike", [0, 0, 100, 100]
                ),
                _overlap_prior_record(
                    source_index, "person", "Person", [10, 0, 90, 100]
                ),
            ]
        )
    prior = build_frequent_overlap_prior(records)
    candidates = []
    records_by_id = {}
    overlap_index = {}
    for source_index, point_id in enumerate(
        ("selector-float-z", "selector-float-a", "selector-float-m")
    ):
        alternative_point_id = f"p-{source_index}-person"
        candidates.append(
            {
                "point_id": point_id,
                "class_name": "Bike",
                "suggested_neighbor_class": "Person",
                "wrong_class_suspicion": 0.95 - source_index * 0.1,
                "split": "train",
                "image_relpath": f"source-{source_index}.jpg",
                "refined_outlier": {
                    "schema": REFINEMENT_SCHEMA,
                    "decision_contract": REFINEMENT_DECISION_CONTRACT,
                    "status": STATUS_UNRESOLVED,
                    "current_class": "Bike",
                    "alternative_class": "Person",
                    "qualified_for_human_review": False,
                    "intrinsic_current_support": 0.35 + source_index * 0.03,
                    "intrinsic_alternative_support": 0.65 - source_index * 0.02,
                    "overlap_object_count": 1,
                    "annotated_overlap_alternative_bbox_xyxy": [
                        10,
                        0,
                        90,
                        100,
                    ],
                    "annotated_overlap_alternative_point_id": (
                        alternative_point_id
                    ),
                    "decision_gates": {
                        "current_present": True,
                        "current_absent": False,
                        "alternative_present": True,
                        "annotated_overlap": True,
                        "alternative_evidence_localized_to_overlap": True,
                        "alternative_evidence_external_to_overlap": False,
                    },
                },
            }
        )
        records_by_id[point_id] = {
            "point_id": point_id,
            "class_name": "Bike",
            "bbox_xyxy": [0, 0, 100, 100],
            "_image_width": 100,
            "_image_height": 100,
            "_image_sha256": f"{source_index + 1:064x}",
            "split": "train",
            "image_relpath": f"source-{source_index}.jpg",
        }
        overlap_index[point_id] = [
            {
                "point_id": alternative_point_id,
                "class_name": "Person",
                "iou": 0.8,
                "target_area_covered": 0.8,
                "other_area_covered": 1.0,
                "relation": "partial_contamination",
            }
        ]
    selector_summary = api._class_analysis_assign_selector_priority_ranks(
        candidates,
        overlap_prior=prior,
        overlap_index=overlap_index,
        records_by_id=records_by_id,
    )
    refinement = {
        "selector_priority_contract": SELECTOR_PRIORITY_CONTRACT,
        "selector_priority": selector_summary,
        "selector_priority_candidate_count": len(candidates),
    }
    assert selector_summary["dataset_overlap_applied_candidate_count"] == len(
        candidates
    )
    assert selector_summary["utility_model"]["dataset_overlap"][
        "mean_absolute_utility_effect"
    ] > 0.0
    return json.loads(
        json.dumps({"refinement": refinement, "rows": candidates})
    )


def test_selector_priority_validation_accepts_last_bit_summary_mean_drift():
    artifact = _selector_priority_validation_fixture_with_overlap_effects()
    refinement = artifact["refinement"]
    overlap_summary = refinement["selector_priority"]["utility_model"][
        "dataset_overlap"
    ]
    original_mean = overlap_summary["mean_absolute_utility_effect"]
    overlap_summary["mean_absolute_utility_effect"] = math.nextafter(
        original_mean, math.inf
    )

    api._class_analysis_validate_selector_priority_artifact(
        refinement=refinement,
        refinement_rows=artifact["rows"],
        configured_contract=SELECTOR_PRIORITY_CONTRACT,
    )


def test_selector_priority_validation_rejects_material_summary_mean_tamper():
    artifact = _selector_priority_validation_fixture_with_overlap_effects()
    refinement = artifact["refinement"]
    overlap_summary = refinement["selector_priority"]["utility_model"][
        "dataset_overlap"
    ]
    overlap_summary["mean_absolute_utility_effect"] += 1e-6

    with pytest.raises(ValueError, match="summary_model_counts"):
        api._class_analysis_validate_selector_priority_artifact(
            refinement=refinement,
            refinement_rows=artifact["rows"],
            configured_contract=SELECTOR_PRIORITY_CONTRACT,
        )


def test_v6_selector_keeps_pair_conflict_metadata_outside_score_contract():
    candidate = {
        "point_id": "bike-box",
        "class_name": "Bike",
        "suggested_neighbor_class": "Person",
        "wrong_class_suspicion": 0.9,
        "bbox_xyxy": [0.0, 0.0, 100.0, 100.0],
        "refined_outlier": {
            "schema": REFINEMENT_SCHEMA,
            "decision_contract": REFINEMENT_DECISION_CONTRACT,
            "status": STATUS_PAIR_CONFLICT,
            "current_class": "Bike",
            "alternative_class": "Person",
            "qualified_for_human_review": False,
            "overlap_relation": "duplicate_like",
            "overlap_object_count": 1,
            "annotated_overlap_alternative_bbox_xyxy": [
                1.0,
                1.0,
                99.0,
                99.0,
            ],
            "annotated_overlap_alternative_point_id": "person-box",
            "pair_conflict": {
                "point_id": "bike-box",
                "current_class": "Bike",
                "other_point_id": "person-box",
                "other_class_name": "Person",
                "target_bbox_xyxy": [0.0, 0.0, 100.0, 100.0],
                "other_bbox_xyxy": [1.0, 1.0, 99.0, 99.0],
                "relation": "duplicate_like",
            },
        },
    }
    selector_summary = api._class_analysis_assign_selector_priority_ranks(
        [candidate]
    )
    refinement = {
        "selector_priority_contract": SELECTOR_PRIORITY_CONTRACT,
                "selector_priority": selector_summary,
                "selector_priority_candidate_count": 1,
                "rough_candidate_count": 1,
            }

    api._class_analysis_validate_selector_priority_artifact(
        refinement=refinement,
        refinement_rows=[candidate],
        configured_contract=SELECTOR_PRIORITY_CONTRACT,
    )

    original_score = candidate["refined_outlier"]["selector_priority_score"]
    assert candidate["refined_outlier"]["selector_v6"][
        "overlap_evidence_state"
    ] == "duplicate_conflict"
    assert candidate["refined_outlier"]["selector_v6"]["dataset_overlap"][
        "applied"
    ] is False
    for field, value in (
        ("other_point_id", "different-person"),
        ("other_class_name", "Building"),
        ("other_bbox_xyxy", [2.0, 1.0, 99.0, 99.0]),
    ):
        corrupted = copy.deepcopy(candidate)
        corrupted["refined_outlier"]["pair_conflict"][field] = value
        api._class_analysis_validate_selector_priority_artifact(
            refinement=refinement,
            refinement_rows=[corrupted],
            configured_contract=SELECTOR_PRIORITY_CONTRACT,
        )
        assert corrupted["refined_outlier"][
            "selector_priority_score"
        ] == original_score


def test_selector_restore_ignores_legacy_triage_policy_fields():
    prior, query_source = _provisional_overlap_prior()
    source_sha256 = query_source.split(":", 1)[1]
    candidate = {
        "point_id": "restore-triage",
        "class_name": "ElevatedFixture",
        "suggested_neighbor_class": "Building",
        "wrong_class_suspicion": 1.0,
        "refined_outlier": {
            "schema": REFINEMENT_SCHEMA,
            "decision_contract": REFINEMENT_DECISION_CONTRACT,
            "status": STATUS_UNRESOLVED,
            "current_class": "ElevatedFixture",
            "alternative_class": "Building",
            "qualified_for_human_review": False,
            "intrinsic_current_support": 0.1,
            "intrinsic_alternative_support": 0.2,
            "decision_gates": {
                "current_present": False,
                "current_absent": False,
                "alternative_evidence_localized_to_overlap": False,
                "alternative_evidence_external_to_overlap": False,
            },
            "overlap_object_count": 1,
            "annotated_overlap_alternative_bbox_xyxy": [10, 10, 90, 90],
            "annotated_overlap_alternative_point_id": "building-overlap",
            "source_image_sha256": source_sha256,
        },
    }
    match = {
        "point_id": "building-overlap",
        "class_name": "Building",
        "iou": 0.65,
        "target_area_covered": 0.90,
        "other_area_covered": 0.75,
        "relation": "partial_contamination",
    }
    selector = api._class_analysis_assign_selector_priority_ranks(
        [candidate],
        overlap_prior=prior,
        overlap_index={"restore-triage": [match]},
        records_by_id={
            "restore-triage": {
                "_image_sha256": source_sha256,
                "split": "train",
                "image_relpath": "restore-triage.jpg",
            }
        },
    )
    refinement = {
        "selector_priority_contract": SELECTOR_PRIORITY_CONTRACT,
        "selector_priority": selector,
        "selector_priority_candidate_count": 1,
    }
    api._class_analysis_validate_selector_priority_artifact(
        refinement=refinement,
        refinement_rows=[candidate],
        configured_contract=SELECTOR_PRIORITY_CONTRACT,
    )

    tampered = copy.deepcopy(candidate)
    tampered["refined_outlier"]["frequent_overlap_prior"]["strata"][0][
        "triage_cohort_diagnostics"
    ][2]["minimum_wilson_lower_bound"] = 0.01
    api._class_analysis_validate_selector_priority_artifact(
        refinement=refinement,
        refinement_rows=[tampered],
        configured_contract=SELECTOR_PRIORITY_CONTRACT,
    )


def _empty_exact_view_calibration() -> dict:
    return {
        "contract": "class-analysis-exact-view-calibration-pass-v1",
        "eligible_example_count": 0,
        "accepted_example_count": 0,
        "skipped_example_count": 0,
        "per_class_accepted_source_counts": {},
    }


def _synthetic_calibration_provenance(class_count: int) -> dict:
    counts = np.full(class_count, 8, dtype=np.int32)
    heldout_sources = np.full(class_count, 2, dtype=np.int32)
    fit_sources = np.full(class_count, 3, dtype=np.int32)
    heldout_source_ids = np.asarray(
        [f"{index:016x}" for index in range(32)]
    )
    fit_source_ids = np.asarray(
        [f"{index:016x}" for index in range(100, 108)]
    )
    return {
        "calibration_split_digest": _calibration_source_split_digest(
            heldout_source_ids.tolist(), fit_source_ids.tolist()
        ),
        "calibration_heldout_source_count": 32,
        "calibration_fit_source_count": 8,
        "calibration_heldout_source_ids": heldout_source_ids,
        "calibration_fit_source_ids": fit_source_ids,
        "calibration_target_patch_counts": counts.copy(),
        "calibration_background_patch_counts": counts.copy(),
        "calibration_target_source_counts": heldout_sources.copy(),
        "calibration_background_source_counts": heldout_sources.copy(),
        "calibration_target_passing_source_counts": heldout_sources.copy(),
        "calibration_target_source_pass_fractions": np.ones(
            class_count,
            dtype=np.float32,
        ),
        "fit_target_patch_counts": counts.copy(),
        "fit_background_patch_counts": counts.copy(),
        "fit_target_source_counts": fit_sources.copy(),
        "fit_background_source_counts": fit_sources.copy(),
    }


def _tokens_with(indices: set[int], positive: np.ndarray) -> np.ndarray:
    negative = np.asarray([1.0, 0.0], dtype=np.float32)
    return np.stack(
        [positive if index in indices else negative for index in range(16)],
        axis=0,
    )


@pytest.mark.parametrize(
    ("overrides", "detail"),
    [
        ({"input_size": True}, "class_analysis_refinement_input_size_unsupported"),
        (
            {"selected_fraction": float("nan")},
            "class_analysis_refinement_selected_fraction_invalid",
        ),
        (
            {"max_candidates": True},
            "class_analysis_refinement_candidate_cap_invalid",
        ),
        (
            {"anchors_per_class": 0},
            "class_analysis_refinement_anchor_count_invalid",
        ),
        (
            {"patches_per_anchor": -1},
            "class_analysis_refinement_patches_per_anchor_invalid",
        ),
        (
            {"patch_reservoir_per_class": 0},
            "class_analysis_refinement_patch_reservoir_invalid",
        ),
        (
            {"prototypes_per_class": 0},
            "class_analysis_refinement_prototype_count_invalid",
        ),
        (
            {"minimum_distinct_sources": 0},
            "class_analysis_refinement_minimum_distinct_sources_invalid",
        ),
        (
            {"support_margin": float("inf")},
            "class_analysis_refinement_support_margin_invalid",
        ),
        (
            {"weak_support_margin": 0.2},
            "class_analysis_refinement_support_margin_order_invalid",
        ),
        (
            {"overlap_localized_fraction": 1.01},
            "class_analysis_refinement_overlap_localized_fraction_invalid",
        ),
        (
            {"outside_overlap_confirm_fraction": False},
            "class_analysis_refinement_outside_overlap_confirm_fraction_invalid",
        ),
        (
            {"seed": -1},
            "class_analysis_refinement_seed_invalid",
        ),
    ],
)
def test_refinement_config_rejects_malformed_values(overrides, detail):
    with pytest.raises(ValueError, match=f"^{detail}$"):
        RefinementConfig(**overrides).validate()


def test_refinement_config_accepts_numpy_integer_fields():
    RefinementConfig(
        max_candidates=np.int64(5_000),
        anchors_per_class=np.int32(128),
        seed=np.int64(42),
    ).validate()


def test_dinov3_register_tokens_are_not_spatial_patches():
    hidden = np.arange(1 * 21 * 3, dtype=np.float32).reshape(1, 21, 3)

    patches, grid = strip_torch_dinov3_special_tokens(
        hidden,
        num_register_tokens=4,
    )

    assert grid == (4, 4)
    assert patches.shape == (1, 16, 3)
    expected = hidden[:, 5:, :]
    expected /= np.maximum(
        np.linalg.norm(expected, axis=-1, keepdims=True),
        1e-12,
    )
    np.testing.assert_allclose(patches, expected)
    assert api._class_analysis_dinov3_spatial_token_offset(4) == 5


def test_selected_class_tail_is_stable_and_small_classes_fail_closed():
    small = [
        {"point_id": f"p{index}", "outlier_score": index}
        for index in range(19)
    ]
    assert select_within_class_outlier_candidates(small) == []

    points = [
        {
            "point_id": f"p{index:04d}",
            "class_name": "A",
            "outlier_score": float(index % 11),
        }
        for index in range(100)
    ]
    selected = select_within_class_outlier_candidates(
        list(reversed(points)),
        fraction=0.05,
        cap=5_000,
    )
    assert len(selected) == 5
    assert [row["point_id"] for row in selected] == [
        "p0010",
        "p0021",
        "p0032",
        "p0043",
        "p0054",
    ]


def test_fractional_rasterization_preserves_partial_target_coverage():
    target, valid = rasterize_box_fractions(
        [0, 0, 4, 4],
        (2, 2),
        [[0, 0, 1, 4]],
        supersample=4,
    )

    np.testing.assert_allclose(valid, np.ones((2, 2), dtype=np.float32))
    assert np.all((target >= 0.0) & (target <= 1.0))
    assert 0.0 < float(target[0, 0]) < 1.0
    assert float(target[0, 1]) == 0.0


def test_exact_rasterization_cannot_miss_subsample_thin_box():
    target, valid = rasterize_box_fractions(
        [0, 0, 100, 100],
        (1, 1),
        [[1.01, 0, 1.02, 100]],
        supersample=1,
    )

    assert valid[0, 0] == pytest.approx(1.0)
    assert target[0, 0] == pytest.approx(0.0001)


def test_exact_rasterization_uses_box_union_without_double_counting():
    target, valid = rasterize_box_fractions(
        [0, 0, 10, 10],
        (1, 1),
        [[0, 0, 6, 10], [4, 0, 10, 10]],
        supersample=4,
    )

    assert valid[0, 0] == pytest.approx(1.0)
    assert target[0, 0] == pytest.approx(1.0)


def test_odd_letterbox_dimension_uses_integer_floor_paste_offset():
    _centres, centre_valid = patch_source_centres(
        [0, 0, 5, 2],
        (5, 5),
    )
    target, sampled_valid = rasterize_box_fractions(
        [0, 0, 5, 2],
        (5, 5),
        [[0, 0, 5, 2]],
        supersample=1,
    )

    # The 5x2 crop is pasted at y=1 on the 5x5 canonical canvas. A float
    # symmetric offset of 1.5 would incorrectly admit a third boundary row.
    assert int(centre_valid.sum()) == 10
    assert int(sampled_valid.sum()) == 10
    np.testing.assert_array_equal(target, sampled_valid)


def test_thin_letterboxed_target_retains_fractional_patch_support():
    _centres, valid = patch_source_centres(
        [0, 0, 5, 100],
        (16, 16),
    )

    assert valid.dtype == np.float32
    assert float(valid.sum()) > 0.0
    assert np.count_nonzero(valid) >= 16

    heat = np.full((16, 16), 0.20, dtype=np.float32)
    target = np.where(valid > 0.0, 0.25, 0.0).astype(np.float32)
    support, coverage = _support_from_heat(
        heat,
        target,
        positive_margin=0.08,
    )
    component_fraction, component_cells = _largest_positive_component(
        heat,
        target,
        grid_shape=(16, 16),
        threshold=0.08,
    )

    assert support == pytest.approx(0.20)
    assert coverage == pytest.approx(1.0)
    assert component_fraction == pytest.approx(1.0)
    assert component_cells > 0


@pytest.mark.parametrize(
    "target_bbox",
    (
        [70.25, 4.0, 75.75, 92.0],
        [7.0, 31.25, 151.0, 45.75],
    ),
    ids=("skinny_vertical", "non_square_horizontal"),
)
def test_exact_view_calibration_masks_match_candidate_scoring_masks(
    target_bbox,
):
    image = Image.new("RGB", (160, 96), (31, 47, 63))
    views = []
    try:
        views, _primary, metadata = api._embedding_make_crop_views(
            image,
            target_bbox,
            crop_mode="tight",
            padding_ratio=0.0,
            preprocess_mode="canonical",
            canonical_size=224,
            background_mode="full_crop",
            view_mode="tight_context",
        )
        grid_shape = (14, 14)
        calibration_masks = (
            api._class_analysis_refinement_exact_view_target_masks(
                view_metadata=metadata,
                target_bbox=target_bbox,
                grid_shape=grid_shape,
            )
        )
        tokens = np.tile(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            (grid_shape[0] * grid_shape[1], 1),
        )
        evidence = score_candidate(
            point_id="mask-parity",
            current_class="A",
            alternative_class="B",
            token_views=[tokens, tokens],
            crop_boxes=[row["crop_xyxy"] for row in metadata],
            target_bbox=target_bbox,
            grid_shape=grid_shape,
            alternative_overlap_boxes=[],
            bank=_synthetic_bank(),
            config=RefinementConfig(input_size=224),
        )
        candidate_masks = evidence["_sidecar"]["target_mask"]
        assert len(calibration_masks) == 2
        for index, calibration_mask in enumerate(calibration_masks):
            assert calibration_mask.shape == (196,)
            assert 0.0 < float(calibration_mask.sum()) < 196.0
            expected = (
                calibration_mask.reshape(grid_shape) * np.float32(255.0)
            ).astype(np.uint8)
            np.testing.assert_array_equal(candidate_masks[index], expected)
    finally:
        image.close()
        for view in views:
            view.close()


def test_refinement_source_key_normalizes_path_fallback_for_all_passes():
    record = {
        "split": None,
        "image_relpath": "nested/frame.jpg",
    }
    assert api._class_analysis_refinement_source_key(record) == (
        "train/nested/frame.jpg"
    )
    record["_image_sha256"] = " a1b2c3 "
    assert api._class_analysis_refinement_source_key(record) == "a1b2c3"


def test_exact_view_calibration_fails_when_every_eligible_anchor_is_skipped(
    monkeypatch,
    tmp_path,
):
    image_path = tmp_path / "eligible.jpg"
    Image.new("RGB", (64, 64), (21, 34, 55)).save(image_path)
    anchor = {
        "point_id": "heldout-anchor",
        "class_name": "A",
        "split": "train",
        "image_relpath": "eligible.jpg",
        "bbox_xyxy": [8, 8, 56, 56],
        "_image_path": str(image_path),
        "_image_sha256": "eligible-source",
    }
    monkeypatch.setattr(
        api,
        "_class_analysis_refinement_exact_view_calibration_anchors",
        lambda *_args, **_kwargs: [anchor],
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_encode_dinov3_spatial_batch",
        lambda *_args, **_kwargs: (
            np.full((2, 196, 2), np.nan, dtype=np.float32),
            (14, 14),
            "test",
        ),
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_refinement_memory_snapshot",
        lambda *_args, **_kwargs: {
            "combined_rss_bytes": 100,
            "incremental_combined_rss_bytes": 0,
        },
    )
    job = api.ClassAnalysisJob(
        job_id="ca_exact_calibration_empty",
        status="running",
        request={"encoder_model": "test/dinov3"},
    )

    with pytest.raises(
        ValueError,
        match="class_analysis_refinement_exact_view_calibration_empty",
    ):
        api._class_analysis_calibrate_exact_two_view_bank(
            bank=_synthetic_bank(),
            anchors=[anchor],
            request=job.request,
            config=RefinementConfig(input_size=224),
            grid_shape=(14, 14),
            job=job,
            baseline_memory={"combined_rss_bytes": 100},
            initial_backend="test",
        )


def test_subpixel_identical_boxes_are_duplicate_like():
    geometry = bbox_overlap_geometry(
        [10.0, 10.0, 10.5, 10.5],
        [10.0, 10.0, 10.5, 10.5],
    )

    assert geometry is not None
    assert geometry["iou"] == pytest.approx(1.0)
    assert geometry["target_area_covered"] == pytest.approx(1.0)
    assert geometry["other_area_covered"] == pytest.approx(1.0)
    assert geometry["relation"] == "duplicate_like"
    assert api._class_analysis_bbox_overlap_geometry(
        [10.0, 10.0, 10.5, 10.5],
        [10.0, 10.0, 10.5, 10.5],
    ) == geometry
    conflict = api._class_analysis_dual_bbox_conflict(
        {
            "point_id": "a",
            "class_name": "A",
            "bbox_xyxy": [10.0, 10.0, 10.5, 10.5],
        },
        {
            "point_id": "b",
            "class_name": "B",
            "bbox_xyxy": [10.0, 10.0, 10.5, 10.5],
        },
    )
    assert conflict is not None
    assert conflict["iou"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("target_bbox", "other_bbox"),
    [
        ([0.0, 0.0, float("inf"), 4.0], [0.0, 0.0, 1.0, 4.0]),
        ([0.0, 0.0, 4.0, 4.0], [0.0, 0.0, float("-inf"), 4.0]),
        ([0.0, 0.0, float("nan"), 4.0], [0.0, 0.0, 1.0, 4.0]),
    ],
)
def test_overlap_geometry_rejects_nonfinite_boxes(target_bbox, other_bbox):
    assert bbox_overlap_geometry(target_bbox, other_bbox) is None


def test_spatial_tokens_reject_nonfinite_values():
    with pytest.raises(
        ValueError,
        match="class_analysis_refinement_tokens_nonfinite",
    ):
        api._class_analysis_validate_mlx_patch_tokens(
            np.asarray([[[np.nan, 1.0]]], dtype=np.float32)
        )


def test_query_encoder_isolates_nonfinite_samples_for_candidate_boundary(
    monkeypatch,
):
    patches = np.asarray(
        [
            [[1.0, 0.0]] * 4,
            [[np.nan, 1.0]] * 4,
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        api,
        "_data_ingestion_get_dinov3",
        lambda _model: (object(), object(), "test", "cpu"),
    )
    monkeypatch.setattr(
        api,
        "_data_ingestion_dinov3_tokens",
        lambda *_args, **_kwargs: (
            patches,
            np.zeros((2, 2), dtype=np.float32),
        ),
    )
    images = [
        Image.new("RGB", (4, 4), (0, 0, 0)),
        Image.new("RGB", (4, 4), (0, 0, 0)),
    ]
    try:
        with pytest.raises(
            ValueError,
            match="class_analysis_refinement_tokens_nonfinite",
        ):
            api._class_analysis_encode_dinov3_spatial_batch(
                images,
                model_name="test",
            )

        isolated, grid, _backend = (
            api._class_analysis_encode_dinov3_spatial_batch(
                images,
                model_name="test",
                isolate_nonfinite_samples=True,
            )
        )
        assert grid == (2, 2)
        assert np.all(np.isfinite(isolated[0]))
        assert not np.any(np.isfinite(isolated[1]))
        with pytest.raises(
            ValueError,
            match="class_analysis_refinement_tokens_nonfinite",
        ):
            class_margin_heatmap(isolated[1], _synthetic_bank(), "A")
    finally:
        for image in images:
            image.close()


def test_reference_bank_float32_cache_roundtrip_preserves_scores():
    bank = _synthetic_bank()
    tokens = np.asarray(
        [[0.67891234, 0.73421985]],
        dtype=np.float32,
    )
    before = class_margin_heatmap(tokens, bank, "A")

    arrays = bank.to_arrays()
    assert arrays["prototypes"].dtype == np.float32
    assert arrays["background_prototypes"].dtype == np.float32
    restored = ReferenceBank.from_arrays(arrays)
    after = class_margin_heatmap(tokens, restored, "A")

    np.testing.assert_array_equal(after, before)


def test_legacy_float16_reference_bank_cache_is_rejected(
    monkeypatch,
    tmp_path,
):
    cache_root = tmp_path / "cache"
    bank_root = cache_root / "patch_reference_banks"
    bank_root.mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    fingerprint = "b" * 64
    arrays = _synthetic_bank().to_arrays()
    arrays["prototypes"] = arrays["prototypes"].astype(np.float16)
    arrays["background_prototypes"] = arrays[
        "background_prototypes"
    ].astype(np.float16)
    arrays["grid_shape"] = np.asarray([2, 2], dtype=np.int32)
    np.savez_compressed(
        bank_root / f"{fingerprint}.npz",
        **arrays,
    )

    assert api._class_analysis_load_refinement_bank(fingerprint) is None


def test_reference_bank_cache_rejects_valid_payload_renamed_to_other_fingerprint(
    monkeypatch,
    tmp_path,
):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    source_fingerprint = "a" * 64
    other_fingerprint = "b" * 64
    api._class_analysis_write_refinement_bank(
        source_fingerprint,
        bank=_synthetic_bank(),
        grid_shape=(2, 2),
        calibration_diagnostics=_empty_exact_view_calibration(),
    )
    bank_root = cache_root / "patch_reference_banks"
    source_path = bank_root / f"{source_fingerprint}.npz"
    other_path = bank_root / f"{other_fingerprint}.npz"
    other_path.write_bytes(source_path.read_bytes())

    assert api._class_analysis_load_refinement_bank(source_fingerprint) is not None
    assert api._class_analysis_load_refinement_bank(other_fingerprint) is None


def test_v33_reference_bank_backend_cache_write_load_roundtrip(
    monkeypatch,
    tmp_path,
):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    fingerprint = "9" * 64
    bank = _synthetic_bank()
    diagnostics = {
        "contract": "class-analysis-exact-view-calibration-pass-v1",
        "eligible_example_count": 3,
        "accepted_example_count": 2,
        "skipped_example_count": 1,
        "transient_failure_example_count": 0,
        "deterministic_rejected_example_count": 1,
        "intrinsic_anchor_eligible_count": 2,
        "intrinsic_anchor_accepted_count": 2,
        "intrinsic_anchor_transient_failure_count": 0,
        "transient_failure_count": 0,
        "cache_reusable": True,
        "cache_suppression_reasons": [],
        "per_class_accepted_source_counts": {"A": 1, "B": 1},
    }

    api._class_analysis_write_refinement_bank(
        fingerprint,
        bank=bank,
        grid_shape=(2, 2),
        calibration_diagnostics=diagnostics,
    )
    loaded = api._class_analysis_load_refinement_bank(fingerprint)

    assert loaded is not None
    restored, grid_shape, diagnostics = loaded
    assert grid_shape == (2, 2)
    assert diagnostics == {
        "contract": "class-analysis-exact-view-calibration-pass-v1",
        "eligible_example_count": 3,
        "accepted_example_count": 2,
        "skipped_example_count": 1,
        "transient_failure_example_count": 0,
        "deterministic_rejected_example_count": 1,
        "intrinsic_anchor_eligible_count": 2,
        "intrinsic_anchor_accepted_count": 2,
        "intrinsic_anchor_transient_failure_count": 0,
        "transient_failure_count": 0,
        "cache_reusable": True,
        "cache_suppression_reasons": [],
        "per_class_accepted_source_counts": {"A": 1, "B": 1},
    }
    assert restored.schema == api.CLASS_ANALYSIS_REFINEMENT_V33_SCHEMA
    assert (
        restored.pair_probe_contract
        == api.CLASS_ANALYSIS_REFINEMENT_V33_PAIR_PROBE_CONTRACT
    )
    assert (
        restored.pair_probe_view_contract
        == api.CLASS_ANALYSIS_REFINEMENT_V33_VIEW_FEATURE_CONTRACT
    )
    assert (
        restored.pair_probe_lower_bound_contract
        == api.CLASS_ANALYSIS_REFINEMENT_V33_LOWER_BOUND_CONTRACT
    )
    restored_arrays = restored.to_arrays()
    for field in (
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
        "pair_probe_fit_eval_split_digests",
        "pair_probe_fit_balanced_accuracies",
        "pair_probe_eval_sensitivities",
        "pair_probe_eval_specificities",
        "pair_current_absence_eval_fractions",
        "pair_alternative_strong_eval_fractions",
    ):
        np.testing.assert_array_equal(
            restored_arrays[field],
            bank.to_arrays()[field],
        )


def test_reference_bank_cache_suppresses_transient_partial_builds(
    monkeypatch,
    tmp_path,
):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    fingerprint = "7" * 64
    diagnostics = {
        "contract": "class-analysis-exact-view-calibration-pass-v1",
        "eligible_example_count": 2,
        "accepted_example_count": 1,
        "skipped_example_count": 1,
        "transient_failure_example_count": 1,
        "deterministic_rejected_example_count": 0,
        "intrinsic_anchor_eligible_count": 2,
        "intrinsic_anchor_accepted_count": 2,
        "intrinsic_anchor_transient_failure_count": 0,
        "transient_failure_count": 1,
        "cache_reusable": False,
        "cache_suppression_reasons": [
            "transient_reference_build_failure"
        ],
        "per_class_accepted_source_counts": {"A": 1},
    }

    api._class_analysis_write_refinement_bank(
        fingerprint,
        bank=_synthetic_bank(),
        grid_shape=(2, 2),
        calibration_diagnostics=diagnostics,
    )

    assert api._class_analysis_load_refinement_bank(fingerprint) is None
    assert not (
        cache_root / "patch_reference_banks" / f"{fingerprint}.npz"
    ).exists()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (
            "exact_view_calibration_eligible_count",
            np.asarray([3.0], dtype=np.float32),
        ),
        (
            "exact_view_calibration_accepted_count",
            np.asarray([True], dtype=np.bool_),
        ),
        (
            "exact_view_calibration_class_accepted_counts",
            np.asarray([1.0, 1.0], dtype=np.float32),
        ),
        (
            "exact_view_calibration_class_accepted_counts",
            np.asarray([True, True], dtype=np.bool_),
        ),
        (
            "exact_view_calibration_class_names",
            np.asarray(["A", "A"]),
        ),
        (
            "exact_view_calibration_class_names",
            np.asarray(["A", ""]),
        ),
    ],
)
def test_reference_bank_cache_rejects_tampered_exact_calibration_diagnostics(
    monkeypatch,
    tmp_path,
    field,
    replacement,
):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    fingerprint = "8" * 64
    api._class_analysis_write_refinement_bank(
        fingerprint,
        bank=_synthetic_bank(),
        grid_shape=(2, 2),
        calibration_diagnostics={
            "contract": "class-analysis-exact-view-calibration-pass-v1",
            "eligible_example_count": 3,
            "accepted_example_count": 2,
            "skipped_example_count": 1,
            "per_class_accepted_source_counts": {"A": 1, "B": 1},
        },
    )
    cache_path = (
        cache_root / "patch_reference_banks" / f"{fingerprint}.npz"
    )
    with np.load(cache_path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays[field] = replacement
    np.savez_compressed(cache_path, **arrays)

    assert api._class_analysis_load_refinement_bank(fingerprint) is None


@pytest.mark.parametrize(
    "field",
    [
        "pair_current_presence_thresholds",
        "pair_current_strong_thresholds",
        "pair_alternative_negative_thresholds",
        "pair_alternative_presence_thresholds",
    ],
)
def test_reference_bank_cache_requires_float32_fit_threshold_arrays(
    monkeypatch,
    tmp_path,
    field,
):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    fingerprint = "7" * 64
    api._class_analysis_write_refinement_bank(
        fingerprint,
        bank=_synthetic_bank(),
        grid_shape=(2, 2),
        calibration_diagnostics=_empty_exact_view_calibration(),
    )
    cache_path = cache_root / "patch_reference_banks" / f"{fingerprint}.npz"
    with np.load(cache_path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays[field] = np.asarray(arrays[field], dtype=np.float64)
    np.savez_compressed(cache_path, **arrays)

    assert api._class_analysis_load_refinement_bank(fingerprint) is None


def test_reference_bank_cache_refreshes_exact_open_inode(
    monkeypatch,
    tmp_path,
):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    fingerprint = "e" * 64
    api._class_analysis_write_refinement_bank(
        fingerprint,
        bank=_synthetic_bank(),
        grid_shape=(2, 2),
        calibration_diagnostics=_empty_exact_view_calibration(),
    )
    touched = []

    def record_utime(target, *args, **kwargs):
        touched.append(target)

    monkeypatch.setattr(api.os, "utime", record_utime)

    assert api._class_analysis_load_refinement_bank(fingerprint) is not None
    assert touched
    assert all(isinstance(target, int) for target in touched)


def test_reference_bank_cache_preflights_huge_declared_array_before_np_load(
    monkeypatch,
    tmp_path,
):
    cache_root = tmp_path / "cache"
    bank_root = cache_root / "patch_reference_banks"
    bank_root.mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    fingerprint = "d" * 64
    arrays = {
        **_synthetic_bank().to_arrays(),
        "grid_shape": np.asarray([2, 2], dtype=np.int32),
        "cache_fingerprint": np.asarray(fingerprint, dtype="<U64"),
    }
    members = {
        name: _npy_member_bytes(value)
        for name, value in arrays.items()
    }
    members["prototypes"] = _npy_header_only_bytes(
        shape=(1, 1, 100_000_000),
        dtype=np.float32,
    )
    cache_path = bank_root / f"{fingerprint}.npz"
    _write_npz_members(cache_path, members)
    load_calls = []

    def reject_np_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        raise AssertionError("np.load must not run before bounded NPY preflight")

    monkeypatch.setattr(api.np, "load", reject_np_load)

    assert api._class_analysis_load_refinement_bank(fingerprint) is None
    assert load_calls == []
    assert not cache_path.exists()


def test_context_patches_cannot_vote_and_sidecar_masks_remain_distinct():
    bank = _synthetic_bank()
    class_a = np.asarray([1.0, 0.0], dtype=np.float32)
    class_b = np.asarray([0.0, 1.0], dtype=np.float32)
    tokens = np.stack([class_b] * 16)
    tokens[[0, 1, 4, 5]] = class_a

    evidence = score_candidate(
        point_id="p",
        current_class="A",
        alternative_class="B",
        token_views=[tokens, tokens],
        crop_boxes=[[0, 0, 4, 4], [0, 0, 4, 4]],
        target_bbox=[0, 0, 2, 2],
        grid_shape=(4, 4),
        alternative_overlap_boxes=[],
        bank=bank,
        config=RefinementConfig(),
    )

    assert evidence["status"] == STATUS_UNRESOLVED
    assert "source_resolution_insufficient_for_explanation" in evidence[
        "reason_codes"
    ]
    assert evidence["current_support_score"] > 0.12
    assert evidence["alternative_support_score"] <= 0.02
    sidecar = evidence["_sidecar"]
    assert int(sidecar["valid_mask"][0].sum()) == 16
    assert int((sidecar["target_mask"][0] > 0).sum()) == 4
    arrays = sidecar_arrays([sidecar], grid_shape=(4, 4))
    assert arrays["valid_masks"].shape == (1, 2, 4, 4)
    assert not np.array_equal(
        arrays["valid_masks"],
        arrays["target_masks"],
    )


def test_context_view_dilution_abstains_without_current_correspondence():
    bank = _synthetic_bank()
    class_a = np.asarray([1.0, 0.0], dtype=np.float32)
    background = np.asarray([-1.0, -1.0], dtype=np.float32)
    tight = np.stack([class_a] * 16)
    diluted_context = np.stack([background] * 16)

    evidence = score_candidate(
        point_id="small-clean-target",
        current_class="A",
        alternative_class="B",
        token_views=[tight, diluted_context],
        crop_boxes=[[0, 0, 4, 4], [0, 0, 4, 4]],
        target_bbox=[0, 0, 4, 4],
        grid_shape=(4, 4),
        alternative_overlap_boxes=[],
        bank=bank,
        config=RefinementConfig(),
    )

    # V3.3 keeps the clean tight-view score observable, but a terminal keep now
    # requires current-exclusive evidence to correspond across both views.
    assert evidence["current_support_score"] < 2.0
    assert evidence["status"] == STATUS_UNRESOLVED
    assert (
        evidence["decision_gates"][
            "current_exclusive_component_corresponds"
        ]
        is False
    )


def test_duplicate_like_alternative_geometry_preempts_patch_decision():
    bank = _synthetic_bank()
    class_b = np.asarray([0.0, 1.0], dtype=np.float32)
    tokens = np.stack([class_b] * 16)

    evidence = score_candidate(
        point_id="duplicate-owner",
        current_class="A",
        alternative_class="B",
        token_views=[tokens, tokens],
        crop_boxes=[[0, 0, 4, 4], [0, 0, 4, 4]],
        target_bbox=[0, 0, 4, 4],
        grid_shape=(4, 4),
        alternative_overlap_boxes=[[0, 0, 4, 4]],
        bank=bank,
        config=RefinementConfig(),
    )

    assert evidence["status"] == STATUS_PAIR_CONFLICT
    assert evidence["overlap_relation"] == "duplicate_like"
    assert "near_identical_cross_class_bbox_requires_review" in evidence[
        "reason_codes"
    ]


def test_duplicate_like_overlap_precedes_larger_containing_box():
    bank = _synthetic_bank()
    class_b = np.asarray([0.0, 1.0], dtype=np.float32)
    tokens = np.stack([class_b] * 16)

    evidence = score_candidate(
        point_id="duplicate-among-overlaps",
        current_class="A",
        alternative_class="B",
        token_views=[tokens, tokens],
        crop_boxes=[[0, 0, 4, 4], [0, 0, 4, 4]],
        target_bbox=[0, 0, 4, 4],
        grid_shape=(4, 4),
        alternative_overlap_boxes=[
            [0, 0, 3.8, 4],
            [-1, -1, 5, 5],
        ],
        bank=bank,
        config=RefinementConfig(),
    )

    assert evidence["status"] == STATUS_PAIR_CONFLICT
    assert evidence["overlap_relation"] == "duplicate_like"
    assert evidence["overlap_object_count"] == 2


def test_material_overlap_identity_precedes_higher_iou_incidental_intersection():
    bank = _synthetic_bank()
    class_b = np.asarray([0.0, 1.0], dtype=np.float32)
    tokens = np.stack([class_b] * 16)
    incidental_higher_iou = [0.0, 0.0, 19.0, 50.0]
    material_lower_iou = [-1000.0, 0.0, 21.0, 1000.0]

    evidence = score_candidate(
        point_id="material-overlap-owner",
        current_class="A",
        alternative_class="B",
        token_views=[tokens, tokens],
        crop_boxes=[[0, 0, 100, 100], [0, 0, 100, 100]],
        target_bbox=[0, 0, 100, 100],
        grid_shape=(4, 4),
        alternative_overlap_boxes=[
            incidental_higher_iou,
            material_lower_iou,
        ],
        bank=bank,
        config=RefinementConfig(),
    )

    assert evidence["overlap_object_count"] == 2
    assert evidence["annotated_overlap_alternative_bbox_xyxy"] == (
        material_lower_iou
    )
    assert evidence["overlap_relation"] == "partial_contamination"


def test_refinement_sidecar_accumulator_streams_rows_to_memmaps(tmp_path):
    accumulator = api._ClassAnalysisRefinementSidecarAccumulator(
        work_dir=tmp_path,
        capacity=2,
        grid_shape=(4, 4),
        point_id_width=20,
    )
    base = {
        "current_heatmap": np.ones((2, 4, 4), dtype=np.float16),
        "alternative_heatmap": np.zeros((2, 4, 4), dtype=np.float16),
        "valid_mask": np.ones((2, 4, 4), dtype=np.uint8),
        "target_mask": np.full((2, 4, 4), 127, dtype=np.uint8),
        "overlap_mask": np.zeros((2, 4, 4), dtype=np.uint8),
    }

    assert accumulator.append({"point_id": "p1", **base}) == 0
    assert accumulator.append({"point_id": "p2", **base}) == 1
    arrays = accumulator.arrays()

    assert all(isinstance(value, np.memmap) for value in arrays.values())
    assert arrays["current_heatmaps"].shape == (2, 2, 4, 4)
    assert list(arrays["point_ids"]) == ["p1", "p2"]
    assert accumulator.point_rows == {"p1": 0, "p2": 1}

    arrays.clear()
    accumulator.close()
    assert accumulator._closed is True
    assert not accumulator.work_dir.exists()


def test_refinement_failure_closes_work_maps_and_removes_partial_publication(
    monkeypatch,
    tmp_path,
):
    out_dir = tmp_path / "job"
    out_dir.mkdir()
    job = api.ClassAnalysisJob(job_id="ca_refinement_cleanup")

    def fail_after_partial_publication(**_kwargs):
        accumulator = api._ClassAnalysisRefinementSidecarAccumulator(
            work_dir=out_dir / "work",
            capacity=1,
            grid_shape=(2, 2),
            point_id_width=8,
        )
        job._refinement_sidecar_accumulator = accumulator
        base = {
            "current_heatmap": np.ones((2, 2, 2), dtype=np.float16),
            "alternative_heatmap": np.zeros((2, 2, 2), dtype=np.float16),
            "valid_mask": np.ones((2, 2, 2), dtype=np.uint8),
            "target_mask": np.ones((2, 2, 2), dtype=np.uint8),
            "overlap_mask": np.zeros((2, 2, 2), dtype=np.uint8),
        }
        accumulator.append({"point_id": "p1", **base})
        (out_dir / api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME).write_bytes(
            b"partial-sidecar"
        )
        (out_dir / api.CLASS_ANALYSIS_REFINEMENT_MANIFEST_FILENAME).write_text(
            "{}",
            encoding="utf-8",
        )
        preview_dir = out_dir / api.CLASS_ANALYSIS_REFINEMENT_PREVIEW_DIRNAME
        preview_dir.mkdir()
        (preview_dir / "partial.png").write_bytes(b"partial-preview")
        raise RuntimeError("synthetic_artifact_failure")

    monkeypatch.setattr(
        api,
        "_class_analysis_refine_result_impl",
        fail_after_partial_publication,
    )

    with pytest.raises(RuntimeError, match="synthetic_artifact_failure"):
        api._class_analysis_refine_result(
            records=[],
            spatial_context_records=[],
            result={},
            request={},
            job=job,
            out_dir=out_dir,
        )

    assert job._refinement_sidecar_accumulator is None
    assert not (out_dir / "work" / "patch-refinement-sidecar").exists()
    assert not (
        out_dir / api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME
    ).exists()
    assert not (
        out_dir / api.CLASS_ANALYSIS_REFINEMENT_MANIFEST_FILENAME
    ).exists()
    assert not (
        out_dir / api.CLASS_ANALYSIS_REFINEMENT_PREVIEW_DIRNAME
    ).exists()


def test_inferred_alternative_uses_its_overlap_geometry():
    bank = _synthetic_bank()
    class_a = np.asarray([1.0, 0.0], dtype=np.float32)
    class_b = np.asarray([0.0, 1.0], dtype=np.float32)
    tokens = np.stack([class_a] * 16)
    tokens[[0, 1, 2, 3]] = class_b

    evidence = score_candidate(
        point_id="p",
        current_class="A",
        alternative_class="",
        token_views=[tokens, tokens],
        crop_boxes=[[0, 0, 4, 4], [0, 0, 4, 4]],
        target_bbox=[0, 0, 4, 4],
        grid_shape=(4, 4),
        alternative_overlap_boxes=[],
        overlap_boxes_by_class={"B": [[0, 0, 4, 1]]},
        bank=bank,
        config=RefinementConfig(),
    )

    assert evidence["alternative_class"] == "B"
    assert evidence["status"] == STATUS_UNRESOLVED
    assert "source_resolution_insufficient_for_mixed_composite" in evidence[
        "reason_codes"
    ]
    assert evidence["alternative_evidence_inside_overlap_fraction"] > 0.95
    assert evidence["overlap_relation"] == "target_contains_other"
    assert evidence["overlap_object_count"] == 1
    assert "annotated_overlap_with_both_exclusive_components" in evidence[
        "reason_codes"
    ]


def test_overlap_fraction_is_conditional_on_thin_target_mass():
    bank = _synthetic_bank()
    class_a = np.asarray([1.0, 0.0], dtype=np.float32)
    class_b = np.asarray([0.0, 1.0], dtype=np.float32)
    alternative_tokens = np.stack([class_b] * 4)

    identical = score_candidate(
        point_id="thin-identical",
        current_class="A",
        alternative_class="B",
        token_views=[alternative_tokens, alternative_tokens],
        crop_boxes=[[0, 0, 4, 4], [0, 0, 4, 4]],
        target_bbox=[0, 0, 1, 4],
        grid_shape=(2, 2),
        alternative_overlap_boxes=[[0, 0, 1, 4]],
        bank=bank,
        config=RefinementConfig(),
    )
    assert identical["alternative_evidence_inside_overlap_fraction"] == pytest.approx(
        1.0
    )
    assert identical["alternative_evidence_outside_overlap_fraction"] == pytest.approx(
        0.0
    )
    assert identical["status"] != STATUS_CONFIRMED_OUTLIER

    same_cell_but_disjoint = score_candidate(
        point_id="thin-disjoint",
        current_class="A",
        alternative_class="B",
        token_views=[alternative_tokens, alternative_tokens],
        crop_boxes=[[0, 0, 4, 4], [0, 0, 4, 4]],
        target_bbox=[0, 0, 1, 4],
        grid_shape=(2, 2),
        alternative_overlap_boxes=[[1.1, 0, 2, 4]],
        bank=bank,
        config=RefinementConfig(),
    )
    assert same_cell_but_disjoint[
        "alternative_evidence_inside_overlap_fraction"
    ] == pytest.approx(0.0)
    assert same_cell_but_disjoint[
        "alternative_evidence_outside_overlap_fraction"
    ] == pytest.approx(1.0)

    # A fully nested thin target must still activate the conservative
    # strong-current guard. Raising the calibrated strong threshold makes the
    # otherwise clean current evidence deliberately insufficient.
    guarded_bank = _synthetic_bank()
    guarded_bank.strong_support_thresholds[0] = 1.9
    guarded_current = np.asarray([0.8, 0.6], dtype=np.float32)
    current_tokens = np.stack([guarded_current] * 4)
    nested = score_candidate(
        point_id="thin-nested",
        current_class="A",
        alternative_class="B",
        token_views=[current_tokens, current_tokens],
        crop_boxes=[[0, 0, 4, 4], [0, 0, 4, 4]],
        target_bbox=[0, 0, 1, 4],
        grid_shape=(2, 2),
        # A containing alternative box exercises ambiguous ownership without
        # becoming the stricter near-identical pair-conflict case.
        alternative_overlap_boxes=[[0, 0, 2, 4]],
        bank=guarded_bank,
        config=RefinementConfig(),
    )
    assert nested["status"] == STATUS_UNRESOLVED
    assert "spatial_evidence_not_decisive" in nested["reason_codes"]


def test_confirmation_requires_strong_coherent_alternative_in_both_views():
    bank = _synthetic_bank()
    class_a = np.asarray([1.0, 0.0], dtype=np.float32)
    class_b = np.asarray([0.0, 1.0], dtype=np.float32)
    tight = np.stack([class_b] * 16)
    context = np.stack([class_a] * 16)

    evidence = score_candidate(
        point_id="p",
        current_class="A",
        alternative_class="B",
        token_views=[tight, context],
        crop_boxes=[[0, 0, 4, 4], [0, 0, 4, 4]],
        target_bbox=[0, 0, 4, 4],
        grid_shape=(4, 4),
        alternative_overlap_boxes=[],
        bank=bank,
        config=RefinementConfig(),
    )

    assert evidence["status"] != STATUS_CONFIRMED_OUTLIER


def test_isolated_alternative_cell_cannot_create_mixed_status():
    bank = _synthetic_bank()
    class_b = np.asarray([0.0, 1.0], dtype=np.float32)
    tokens = _tokens_with({0}, class_b)

    evidence = score_candidate(
        point_id="p",
        current_class="A",
        alternative_class="B",
        token_views=[tokens, tokens],
        crop_boxes=[[0, 0, 4, 4], [0, 0, 4, 4]],
        target_bbox=[0, 0, 4, 4],
        grid_shape=(4, 4),
        alternative_overlap_boxes=[],
        bank=bank,
        config=RefinementConfig(),
    )

    assert evidence["status"] != STATUS_MIXED_OR_COMPOSITE


def test_similarity_top_k_counts_each_source_only_once():
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    prototypes = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    source_ids = np.asarray(["s1", "s1", "s1", "s2", "s3"])

    score = _mean_top_source_similarity(query, prototypes, source_ids)

    # Three duplicate rows from s1 contribute one vote, not a false 1.0 score.
    assert score[0] == pytest.approx(1.0 / 3.0)


def test_source_consensus_uses_lower_median_for_even_pools():
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    centroids = np.asarray(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    score = _source_consensus_similarity(query, centroids)

    # A single contaminated source cannot invent arithmetic midpoint support.
    assert score[0] == pytest.approx(0.0)


def test_reliability_fixed_point_cycle_fails_closed():
    transitions = {
        frozenset({"A", "B"}): {"A"},
        frozenset({"A"}): {"B"},
        frozenset({"B"}): {"A"},
        frozenset(): set(),
    }

    def evaluate(active: set[str], candidates: set[str]) -> set[str]:
        return set(transitions[frozenset(active)]) & candidates

    resolved, cycle_detected = _resolve_reliability_active_set(
        {"A", "B"},
        evaluate,
    )

    # A and B oscillate; neither appears throughout the cycle, so neither may
    # publish as reliable merely because iteration stopped on one phase.
    assert cycle_detected is True
    assert resolved == set()


def test_reference_builder_is_source_balanced_calibrated_and_roundtrips():
    config = RefinementConfig(
        patches_per_anchor=6,
        patch_reservoir_per_class=512,
        prototypes_per_class=16,
    )
    builder = StreamingReferenceBankBuilder(config)
    rng = np.random.default_rng(4)
    for class_name, base in (
        ("A", np.asarray([1.0, 0.0, 0.0, 0.0])),
        ("B", np.asarray([0.0, 1.0, 0.0, 0.0])),
    ):
        for index in range(32):
            target = base + rng.normal(0.0, 0.025, (6, 4))
            background = np.asarray([-1.0, -1.0, 0.0, 0.0])
            background = background + rng.normal(0.0, 0.025, (6, 4))
            builder.add(
                class_name=class_name,
                source_key=f"{class_name}-source-{index}",
                patch_tokens=target,
                background_tokens=background,
            )

    bank = builder.finalize()

    assert bank.reliability_tiers.tolist() == ["usable", "usable"]
    assert np.all(bank.heldout_aurocs >= 0.70)
    assert np.all(bank.strong_support_thresholds >= 0.12)
    assert np.all(bank.prototype_source_ids[:, :1] != "")
    restored = ReferenceBank.from_arrays(bank.to_arrays())
    np.testing.assert_array_equal(
        restored.prototype_source_ids,
        bank.prototype_source_ids,
    )
    provenance = bank.calibration_split_provenance()
    assert len(provenance["digest"]) == 64
    assert provenance["heldout_source_count"] > 0
    assert provenance["fit_source_count"] > 0
    assert set(provenance["per_class"]) == {"A", "B"}
    assert provenance == restored.calibration_split_provenance()


def test_under_supported_class_cannot_disable_reliable_calibration_pair():
    config = RefinementConfig(
        patches_per_anchor=6,
        patch_reservoir_per_class=512,
        prototypes_per_class=16,
    )
    builder = StreamingReferenceBankBuilder(config)
    rng = np.random.default_rng(41)
    for class_name, base in (
        ("A", np.asarray([1.0, 0.0, 0.0, 0.0])),
        ("B", np.asarray([0.0, 1.0, 0.0, 0.0])),
    ):
        for index in range(32):
            builder.add(
                class_name=class_name,
                source_key=f"{class_name}-source-{index}",
                patch_tokens=(
                    base + rng.normal(0.0, 0.025, (6, 4))
                ),
                background_tokens=(
                    np.asarray([-1.0, -1.0, 0.0, 0.0])
                    + rng.normal(0.0, 0.025, (6, 4))
                ),
            )
    for index in range(2):
        builder.add(
            class_name="C",
            source_key=f"C-source-{index}",
            patch_tokens=(
                np.asarray([0.0, 0.0, 1.0, 0.0])
                + rng.normal(0.0, 0.025, (6, 4))
            ),
            background_tokens=(
                np.asarray([0.0, 0.0, -1.0, 0.0])
                + rng.normal(0.0, 0.025, (6, 4))
            ),
        )

    bank = builder.finalize()

    assert bank.class_reliability_tier("A") == "usable"
    assert bank.class_reliability_tier("B") == "usable"
    assert bank.class_reliability_tier("C") == "low"
    assert bank.class_is_reliable("A") is True
    assert bank.class_is_reliable("B") is True
    assert bank.class_is_reliable("C") is False


def test_under_supported_lookalike_cannot_filter_reliable_target_bank():
    config = RefinementConfig(
        patches_per_anchor=6,
        patch_reservoir_per_class=512,
        prototypes_per_class=16,
    )
    builder = StreamingReferenceBankBuilder(config)
    rng = np.random.default_rng(411)
    background_base = np.asarray([-1.0, -1.0, 0.0, 0.0])
    for class_name, base in (
        ("A", np.asarray([1.0, 0.0, 0.0, 0.0])),
        ("B", np.asarray([0.0, 1.0, 0.0, 0.0])),
    ):
        for index in range(32):
            builder.add(
                class_name=class_name,
                source_key=f"{class_name}-source-{index}",
                patch_tokens=base + rng.normal(0.0, 0.025, (6, 4)),
                background_tokens=(
                    background_base + rng.normal(0.0, 0.025, (6, 4))
                ),
            )

    # C has enough raw patches and source groups to look admissible to a weak
    # pre-calibration check, but too few anchors for a usable bank.  Its visual
    # distribution deliberately duplicates A, so allowing C into specificity
    # filtering would erase A's target prototypes.
    for index in range(8):
        builder.add(
            class_name="C",
            source_key=f"C-source-{index}",
            patch_tokens=(
                np.asarray([1.0, 0.0, 0.0, 0.0])
                + rng.normal(0.0, 0.025, (6, 4))
            ),
            background_tokens=(
                background_base + rng.normal(0.0, 0.025, (6, 4))
            ),
        )

    bank = builder.finalize()

    assert bank.class_reliability_tier("A") == "usable"
    assert bank.class_reliability_tier("B") == "usable"
    assert bank.class_reliability_tier("C") == "low"
    assert bank.prototype_counts[bank.class_position("A")] >= 4


def test_intrinsic_reliability_does_not_use_global_fixed_point(monkeypatch):
    heldout_sources = {"ah0", "ah1", "bh0", "bh1", "ch0", "ch1"}
    monkeypatch.setattr(
        "services.class_analysis_patch_refinement._global_heldout_sources",
        lambda *_args, **_kwargs: set(heldout_sources),
    )
    builder = StreamingReferenceBankBuilder(
        RefinementConfig(
            patches_per_anchor=6,
            patch_reservoir_per_class=512,
            prototypes_per_class=64,
        )
    )
    rng = np.random.default_rng(913)
    u = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    v = np.asarray([0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
    a_mid = (u + v) / np.linalg.norm(u + v)
    class_b = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    class_c = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    background = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    def add(
        class_name: str,
        source_key: str,
        target_base: np.ndarray,
        repeats: int,
    ) -> None:
        for _ in range(repeats):
            builder.add(
                class_name=class_name,
                source_key=source_key,
                patch_tokens=(
                    target_base + rng.normal(0.0, 0.001, (6, 7))
                ),
                background_tokens=(
                    background + rng.normal(0.0, 0.001, (6, 7))
                ),
            )

    for source_key, base in (
        ("ah0", a_mid),
        ("ah1", a_mid),
        ("shared_u", u),
        ("shared_v", v),
        ("af3", a_mid),
    ):
        add("A", source_key, base, 6)
    for source_key, base in (
        ("bh0", class_b),
        ("bh1", class_b),
        ("bc0", a_mid),
        ("bc1", a_mid),
        ("bc2", a_mid),
        ("shared_u", class_b),
        ("shared_v", class_b),
    ):
        add("B", source_key, base, 4)
    for source_key, base in (
        ("ch0", class_c),
        ("ch1", class_c),
        ("cc0", class_c),
        ("cc1", class_c),
        ("cb0", class_b),
        ("cb1", class_b),
        ("cb2", class_b),
    ):
        add("C", source_key, base, 4)

    bank = builder.finalize()

    # V3 evaluates every class against its own paired background. Ambiguity in
    # B versus C therefore cannot globally demote A or B; directed pair
    # calibration owns those local comparisons.
    assert bank.class_reliability_tier("A") == "usable"
    assert bank.class_reliability_tier("B") == "usable"
    assert bank.class_reliability_tier("C") == "usable"


def test_reliable_competitor_minority_contamination_cannot_erase_clean_class():
    """A repeated background patch is not a destructive class prototype.

    Every B crop contains one A-like patch and five unambiguous B patches. The
    old raw mean-top-3 competitor pool treated those minority patches as a
    perfect B match and deleted A's complete target bank before B's own
    specificity was considered.
    """

    config = RefinementConfig(
        patches_per_anchor=6,
        patch_reservoir_per_class=512,
        prototypes_per_class=32,
    )
    builder = StreamingReferenceBankBuilder(config)
    rng = np.random.default_rng(2026)
    background_base = np.asarray([-1.0, -1.0, 0.0, 0.0])
    for index in range(32):
        builder.add(
            class_name="A",
            source_key=f"A-source-{index}",
            patch_tokens=(
                np.asarray([1.0, 0.0, 0.0, 0.0])
                + rng.normal(0.0, 0.01, (6, 4))
            ),
            background_tokens=(
                background_base + rng.normal(0.0, 0.01, (6, 4))
            ),
        )
        class_b = np.repeat(
            np.asarray([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
            6,
            axis=0,
        )
        class_b += rng.normal(0.0, 0.01, class_b.shape)
        class_b[0] = (
            np.asarray([1.0, 0.0, 0.0, 0.0])
            + rng.normal(0.0, 0.01, 4)
        )
        builder.add(
            class_name="B",
            source_key=f"B-source-{index}",
            patch_tokens=class_b,
            background_tokens=(
                background_base + rng.normal(0.0, 0.01, (6, 4))
            ),
        )

    bank = builder.finalize()

    assert bank.class_reliability_tier("A") == "usable"
    assert bank.class_reliability_tier("B") == "usable"
    assert bank.class_is_reliable("A") is True
    assert bank.prototype_counts[bank.class_position("A")] >= 4


def test_minority_contaminated_competitor_sources_need_consensus(monkeypatch):
    heldout_sources = {"ah0", "ah1", "bh0", "bh1"}
    config = RefinementConfig(
        patches_per_anchor=6,
        patch_reservoir_per_class=512,
        prototypes_per_class=64,
    )
    builder = StreamingReferenceBankBuilder(config)
    rng = np.random.default_rng(916)
    class_a = np.asarray([1.0, 0.0, 0.0, 0.0])
    class_b = np.asarray([0.0, 1.0, 0.0, 0.0])
    background = np.asarray([0.0, 0.0, 1.0, 0.0])

    def add(
        class_name: str,
        source_key: str,
        target_base: np.ndarray,
        repeats: int,
    ) -> None:
        for _ in range(repeats):
            builder.add(
                class_name=class_name,
                source_key=source_key,
                patch_tokens=(
                    target_base + rng.normal(0.0, 0.002, (6, 4))
                ),
                background_tokens=(
                    background + rng.normal(0.0, 0.002, (6, 4))
                ),
            )

    for source_key in ("ah0", "ah1", "af0", "af1", "af2"):
        add("A", source_key, class_a, 6)
    for source_key in ("bh0", "bh1", "bf0", "bf1", "bf2"):
        add("B", source_key, class_b, 4)
    # Two coherent but minority B sources contain only A-like target patches.
    # Raw/source-top-k competitors erase A; strict source consensus does not.
    for source_key in ("bc0", "bc1"):
        add("B", source_key, class_a, 4)

    monkeypatch.setattr(
        "services.class_analysis_patch_refinement._global_heldout_sources",
        lambda *_args, **_kwargs: set(heldout_sources),
    )
    bank = builder.finalize()

    assert bank.class_reliability_tier("A") == "usable"
    assert bank.class_reliability_tier("B") == "usable"
    assert bank.class_is_reliable("A") is True
    assert len(
        set(
            bank.prototype_source_ids[
                bank.class_position("A"),
                : bank.prototype_counts[bank.class_position("A")],
            ].tolist()
        )
    ) >= 3


def test_cluster_compression_preserves_identical_cross_source_support():
    config = RefinementConfig(
        patches_per_anchor=6,
        patch_reservoir_per_class=512,
        prototypes_per_class=16,
    )
    builder = StreamingReferenceBankBuilder(config)
    rng = np.random.default_rng(412)
    background = np.repeat(
        np.asarray([[-1.0, -1.0, 0.0, 0.0]], dtype=np.float32),
        6,
        axis=0,
    )
    for class_name, base in (
        ("A", np.asarray([1.0, 0.0, 0.0, 0.0])),
        ("B", np.asarray([0.0, 1.0, 0.0, 0.0])),
    ):
        for index in range(32):
            builder.add(
                class_name=class_name,
                source_key=f"{class_name}-source-{index}",
                patch_tokens=base + rng.normal(0.0, 0.025, (6, 4)),
                background_tokens=background,
            )

    bank = builder.finalize()

    for class_name in ("A", "B"):
        position = bank.class_position(class_name)
        count = int(bank.background_prototype_counts[position])
        source_ids = bank.background_prototype_source_ids[position, :count]
        assert count == config.prototypes_per_class
        assert len(set(source_ids.tolist())) == config.prototypes_per_class
        assert bank.class_reliability_tier(class_name) == "usable"
        query_source = next(
            f"{class_name}-source-{index}"
            for index in range(32)
            if hashlib.sha256(
                f"{class_name}-source-{index}".encode("utf-8")
            ).hexdigest()[:16]
            in set(source_ids.tolist())
        )
        assert bank.class_has_source_independent_support(
            class_name,
            exclude_source_key=query_source,
        )


def test_unreliable_bank_cannot_poison_reliable_inference_margin():
    clean = _synthetic_bank()
    prototypes = np.zeros((3, 6, 2), dtype=np.float32)
    prototypes[0] = np.asarray([[1.0, 0.0]] * 6, dtype=np.float32)
    prototypes[1] = np.asarray([[0.0, 1.0]] * 6, dtype=np.float32)
    prototypes[2, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    backgrounds = np.zeros_like(prototypes)
    backgrounds[0] = np.asarray([[-1.0, 0.0]] * 6, dtype=np.float32)
    backgrounds[1] = np.asarray([[0.0, -1.0]] * 6, dtype=np.float32)
    backgrounds[2, 0] = np.asarray([-1.0, 0.0], dtype=np.float32)
    bank = ReferenceBank(
        class_names=["A", "B", "C"],
        prototypes=prototypes,
        prototype_counts=np.asarray([6, 6, 1], dtype=np.int32),
        prototype_source_ids=np.asarray(
            [
                ["a1", "a1", "a2", "a2", "a3", "a3"],
                ["b1", "b1", "b2", "b2", "b3", "b3"],
                ["c1", "", "", "", "", ""],
            ]
        ),
        background_prototypes=backgrounds,
        background_prototype_counts=np.asarray([6, 6, 1], dtype=np.int32),
        background_prototype_source_ids=np.asarray(
            [
                ["abg1", "abg1", "abg2", "abg2", "abg3", "abg3"],
                ["bbg1", "bbg1", "bbg2", "bbg2", "bbg3", "bbg3"],
                ["cbg1", "", "", "", "", ""],
            ]
        ),
        anchor_counts=np.asarray([64, 64, 1], dtype=np.int32),
        distinct_source_counts=np.asarray([8, 8, 1], dtype=np.int32),
        reliable=np.asarray([True, True, False]),
        reliability_tiers=np.asarray(["high", "high", "low"]),
        heldout_aurocs=np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        support_thresholds=np.asarray([0.08, 0.08, 0.08], dtype=np.float32),
        strong_support_thresholds=np.asarray(
            [0.12, 0.12, 0.12], dtype=np.float32
        ),
        projection_mean=np.zeros(2, dtype=np.float32),
        projection_components=np.eye(2, dtype=np.float32),
        calibration_status=CALIBRATION_STATUS_SOURCE_AWARE,
        **_synthetic_calibration_provenance(3),
    )
    tokens = np.asarray([[1.0, 0.0]] * 16, dtype=np.float32)

    np.testing.assert_allclose(
        class_margin_heatmap(tokens, bank, "A"),
        class_margin_heatmap(tokens, clean, "A"),
    )
    evidence = score_candidate(
        point_id="query",
        current_class="A",
        alternative_class="B",
        token_views=[tokens, tokens],
        crop_boxes=[[0, 0, 4, 4], [0, 0, 4, 4]],
        target_bbox=[0, 0, 4, 4],
        grid_shape=(4, 4),
        alternative_overlap_boxes=[],
        bank=bank,
        config=RefinementConfig(),
    )
    assert evidence["status"] == STATUS_UNRESOLVED
    assert "source_resolution_insufficient_for_explanation" in evidence[
        "reason_codes"
    ]
    assert evidence["current_support_score"] > 0.12


def test_cluster_medoids_are_unique_members_of_their_assigned_clusters():
    values = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.10],
            [0.95, 0.20],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    centres = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.05],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 1, 1], dtype=np.int64)

    selected = _assigned_cluster_medoid_indices(
        values,
        centres,
        labels,
    )

    assert selected.shape == (2,)
    assert len(set(selected.tolist())) == 2
    assert labels[selected].tolist() == [0, 1]
    assert selected[0] == 0
    assert selected[1] != 0


def test_heldout_only_background_is_not_reused_for_fit_or_calibration():
    config = RefinementConfig(
        patches_per_anchor=6,
        patch_reservoir_per_class=512,
        prototypes_per_class=16,
    )
    builder = StreamingReferenceBankBuilder(config)
    rng = np.random.default_rng(17)
    class_a_sources = [f"A-source-{index}" for index in range(32)]
    class_b_sources = [f"B-source-{index}" for index in range(32)]
    heldout_sources = _global_heldout_sources(
        class_a_sources + class_b_sources
    )
    class_a_heldout = sorted(heldout_sources.intersection(class_a_sources))
    assert class_a_heldout
    background_only_source = class_a_heldout[0]

    for source_key in class_a_sources:
        target = (
            np.asarray([1.0, 0.0, 0.0, 0.0])
            + rng.normal(0.0, 0.025, (6, 4))
        )
        background = None
        if source_key == background_only_source:
            background = (
                np.asarray([-1.0, -1.0, 0.0, 0.0])
                + rng.normal(0.0, 0.025, (6, 4))
            )
        builder.add(
            class_name="A",
            source_key=source_key,
            patch_tokens=target,
            background_tokens=background,
        )
    for source_key in class_b_sources:
        builder.add(
            class_name="B",
            source_key=source_key,
            patch_tokens=(
                np.asarray([0.0, 1.0, 0.0, 0.0])
                + rng.normal(0.0, 0.025, (6, 4))
            ),
            background_tokens=(
                np.asarray([-1.0, -1.0, 0.0, 0.0])
                + rng.normal(0.0, 0.025, (6, 4))
            ),
        )

    assert background_only_source in heldout_sources
    bank = builder.finalize()
    class_a_position = bank.class_position("A")

    assert class_a_position is not None
    assert int(bank.background_prototype_counts[class_a_position]) == 0
    assert bank.class_reliability_tier("A") == "low"
    assert bank.class_is_reliable("A") is False
    assert bank.calibration_status == CALIBRATION_STATUS_SOURCE_AWARE


def test_global_heldout_split_repairs_small_class_calibration_coverage():
    large_group = [f"large-source-{index}" for index in range(96)]
    rare_candidates = [f"rare-source-{index}" for index in range(64)]
    base_heldout = _global_heldout_sources(large_group + rare_candidates)
    rare_group = [
        source for source in rare_candidates if source not in base_heldout
    ][:5]
    assert len(rare_group) == 5
    assert not base_heldout.intersection(rare_group)

    repaired = _global_heldout_sources(
        large_group + rare_candidates,
        source_groups=[large_group, rare_group],
    )

    # The same global split remains leak-free, and a five-source class group
    # receives two held-out plus three fit sources for candidate-safe LOSO.
    assert 0 < len(repaired.intersection(large_group)) < len(large_group)
    assert len(repaired.intersection(rare_group)) == 2
    assert len(set(rare_group) - repaired) == 3


def test_global_heldout_solver_repairs_feasible_shared_source_path():
    sources = list("abcde")
    groups = [
        {"a", "b"},
        {"a", "c"},
        {"c", "d"},
        {"d", "e"},
    ]

    repaired = _global_heldout_sources(sources, source_groups=groups)

    # The former local greedy repair cycled to {b, d}, leaving {a, c}
    # entirely in the fit fold even though {a, d} is a valid global solution.
    assert all(0 < len(group & repaired) < len(group) for group in groups)
    assert repaired == {"a", "d"}


def test_global_heldout_solver_reaches_deep_disjoint_feasible_repair():
    sources = [f"source-{index}" for index in range(400)]
    base_heldout = _global_heldout_sources(sources)
    initially_fit = [source for source in sources if source not in base_heldout]
    groups = [
        set(initially_fit[offset : offset + 5])
        for offset in range(0, 60, 5)
    ]

    repaired = _global_heldout_sources(sources, source_groups=groups)

    # Breadth-by-Hamming repair exhausted 50k states halfway through this
    # feasible case. Constraint-directed backtracking reaches all 12 groups.
    assert all(len(group & repaired) >= 2 for group in groups)
    assert all(len(group - repaired) >= 3 for group in groups)


def test_reference_reliability_requires_multiple_heldout_sources(monkeypatch):
    sources = [f"source-{index}" for index in range(4)]
    monkeypatch.setattr(
        "services.class_analysis_patch_refinement._global_heldout_sources",
        lambda *_args, **_kwargs: {sources[0]},
    )
    builder = StreamingReferenceBankBuilder(
        RefinementConfig(
            patches_per_anchor=8,
            patch_reservoir_per_class=256,
            prototypes_per_class=64,
        )
    )
    rng = np.random.default_rng(2027)
    for index in range(24):
        builder.add(
            class_name="A",
            source_key=sources[index % len(sources)],
            patch_tokens=(
                np.asarray([1.0, 0.0, 0.0, 0.0])
                + rng.normal(0.0, 0.005, (8, 4))
            ),
            background_tokens=(
                np.asarray([0.0, 1.0, 0.0, 0.0])
                + rng.normal(0.0, 0.005, (8, 4))
            ),
        )

    bank = builder.finalize()

    # Patch-level separation is perfect, but source-level AUROC is deliberately
    # undefined with one held-out source on each side and therefore reports 0.
    assert bank.class_heldout_auroc("A") == pytest.approx(0.0)
    assert bank.class_reliability_tier("A") == "low"
    assert bank.class_is_reliable("A") is False


def test_reference_reliability_balances_heldout_sources(monkeypatch):
    heldout_sources = {"s0", "s1"}
    monkeypatch.setattr(
        "services.class_analysis_patch_refinement._global_heldout_sources",
        lambda *_args, **_kwargs: set(heldout_sources),
    )
    builder = StreamingReferenceBankBuilder(
        RefinementConfig(
            patches_per_anchor=8,
            patch_reservoir_per_class=512,
            prototypes_per_class=64,
        )
    )
    rng = np.random.default_rng(77)
    good = np.asarray([1.0, 0.0, 0.0, 0.0])
    background = np.asarray([0.0, 1.0, 0.0, 0.0])

    def add_anchor(source_key: str, target_base: np.ndarray) -> None:
        builder.add(
            class_name="A",
            source_key=source_key,
            patch_tokens=(
                target_base + rng.normal(0.0, 0.002, (8, 4))
            ),
            background_tokens=(
                background + rng.normal(0.0, 0.002, (8, 4))
            ),
        )

    for _ in range(20):
        add_anchor("s0", good)
    add_anchor("s1", background)
    for source_key in ("s2", "s3", "s4"):
        for _ in range(2):
            add_anchor(source_key, good)

    bank = builder.finalize()

    # Patch-weighted AUROC was dominated by s0 and called this usable even
    # though the complete second held-out source lacked target evidence.
    assert bank.class_heldout_auroc("A") < 0.70
    assert bank.class_reliability_tier("A") == "low"
    assert bank.class_is_reliable("A") is False


def test_reliable_tier_requires_three_fit_sources_for_query_loso(monkeypatch):
    heldout_sources = {"s0", "s1", "s2"}
    monkeypatch.setattr(
        "services.class_analysis_patch_refinement._global_heldout_sources",
        lambda *_args, **_kwargs: set(heldout_sources),
    )
    builder = StreamingReferenceBankBuilder(
        RefinementConfig(
            patches_per_anchor=8,
            patch_reservoir_per_class=512,
            prototypes_per_class=64,
        )
    )
    rng = np.random.default_rng(78)
    for index in range(25):
        builder.add(
            class_name="A",
            source_key=f"s{index % 5}",
            patch_tokens=(
                np.asarray([1.0, 0.0, 0.0, 0.0])
                + rng.normal(0.0, 0.002, (8, 4))
            ),
            background_tokens=(
                np.asarray([0.0, 1.0, 0.0, 0.0])
                + rng.normal(0.0, 0.002, (8, 4))
            ),
        )

    bank = builder.finalize()

    assert bank.calibration_split_provenance()["per_class"]["A"][
        "fit_target_source_count"
    ] == 2
    assert bank.class_reliability_tier("A") == "low"
    assert bank.class_is_reliable("A") is False


def test_global_heldout_source_never_enters_pca_through_another_class(
    monkeypatch,
):
    from sklearn.decomposition import PCA as RealPCA

    captured_fit_values = []

    class SpyPCA(RealPCA):
        def fit(self, values, y=None):
            captured_fit_values.append(np.asarray(values).copy())
            return super().fit(values, y)

    monkeypatch.setattr("sklearn.decomposition.PCA", SpyPCA)
    sources = [f"shared-scene-{index}" for index in range(32)]
    heldout_sources = _global_heldout_sources(sources)
    assert heldout_sources
    shared_heldout = sorted(heldout_sources)[0]
    marker = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    builder = StreamingReferenceBankBuilder(
        RefinementConfig(
            patches_per_anchor=1,
            patch_reservoir_per_class=256,
            prototypes_per_class=16,
        )
    )
    for source_key in sources:
        builder.add(
            class_name="A",
            source_key=source_key,
            patch_tokens=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            background_tokens=np.asarray(
                [[-1.0, -1.0, 0.0, 0.0]], dtype=np.float32
            ),
        )
        builder.add(
            class_name="B",
            source_key=source_key,
            patch_tokens=np.asarray(
                [
                    marker
                    if source_key == shared_heldout
                    else np.asarray([0.0, 1.0, 0.0, 0.0])
                ],
                dtype=np.float32,
            ),
            background_tokens=np.asarray(
                [[-1.0, -1.0, 0.0, 0.0]], dtype=np.float32
            ),
        )

    builder.finalize()

    assert len(captured_fit_values) == 1
    assert not np.any(
        np.all(np.isclose(captured_fit_values[0], marker), axis=1)
    )


def test_query_source_exclusion_fails_closed_without_independent_support():
    query_source = "query-image"
    query_id = hashlib.sha256(query_source.encode("utf-8")).hexdigest()[:16]
    source_one = hashlib.sha256(b"independent-one").hexdigest()[:16]
    source_two = hashlib.sha256(b"independent-two").hexdigest()[:16]
    source_three = hashlib.sha256(b"independent-three").hexdigest()[:16]
    bank = ReferenceBank(
        class_names=["A", "B"],
        prototypes=np.asarray(
            [
                [[1.0, 0.0]] * 4,
                [[0.0, 1.0]] * 4,
            ],
            dtype=np.float32,
        ),
        prototype_counts=np.asarray([4, 4], dtype=np.int32),
        prototype_source_ids=np.asarray(
            [
                [query_id, query_id, source_one, source_two],
                [source_one, source_two, source_three, source_three],
            ]
        ),
        background_prototypes=np.asarray(
            [
                [[-1.0, -1.0]] * 4,
                [[-1.0, -1.0]] * 4,
            ],
            dtype=np.float32,
        ),
        background_prototype_counts=np.asarray([4, 4], dtype=np.int32),
        background_prototype_source_ids=np.asarray(
            [
                [query_id, query_id, source_one, source_two],
                [source_one, source_two, source_three, source_three],
            ]
        ),
        anchor_counts=np.asarray([64, 64], dtype=np.int32),
        distinct_source_counts=np.asarray([8, 8], dtype=np.int32),
        reliable=np.asarray([True, True]),
        reliability_tiers=np.asarray(["high", "high"]),
        heldout_aurocs=np.asarray([1.0, 1.0], dtype=np.float32),
        support_thresholds=np.asarray([0.08, 0.08], dtype=np.float32),
        strong_support_thresholds=np.asarray([0.12, 0.12], dtype=np.float32),
        projection_mean=np.zeros(2, dtype=np.float32),
        projection_components=np.eye(2, dtype=np.float32),
        calibration_status=CALIBRATION_STATUS_SOURCE_AWARE,
        **_synthetic_calibration_provenance(2),
    )
    class_a = np.asarray([[1.0, 0.0]] * 16, dtype=np.float32)

    evidence = score_candidate(
        point_id="query",
        current_class="A",
        alternative_class="B",
        token_views=[class_a, class_a],
        crop_boxes=[[0, 0, 4, 4], [0, 0, 4, 4]],
        target_bbox=[0, 0, 4, 4],
        grid_shape=(4, 4),
        alternative_overlap_boxes=[],
        bank=bank,
        config=RefinementConfig(),
        query_source_key=query_source,
    )

    assert not bank.class_has_source_independent_support(
        "A",
        exclude_source_key=query_source,
    )
    assert bank.class_has_source_independent_support(
        "B",
        exclude_source_key=query_source,
    )
    assert evidence["status"] == STATUS_UNRESOLVED
    assert evidence["reference_reliable"] is False
    assert (
        "current_reference_source_independent_support_insufficient"
        in evidence["reason_codes"]
    )


def test_unprovenanced_rows_never_satisfy_source_independent_support():
    bank = copy.deepcopy(_synthetic_bank())
    bank.prototype_source_ids[0] = ["", "", "", "", "a1", "a2"]

    assert not bank.class_has_source_independent_support(
        "A",
        exclude_source_key="",
    )


@pytest.mark.parametrize(
    "legacy_status",
    [
        "heldout_source_margin_v1",
        "heldout_source_margin_loso_v2",
        "global_heldout_source_margin_loso_v3",
        "global_heldout_source_margin_loso_v4",
        "global_stratified_heldout_source_margin_loso_v5",
    ],
)
def test_legacy_calibration_cache_is_rejected(legacy_status):
    arrays = _synthetic_bank().to_arrays()
    arrays["calibration_status"] = np.asarray([legacy_status])

    with pytest.raises(
        ValueError,
        match="class_analysis_refinement_bank_calibration_legacy",
    ):
        ReferenceBank.from_arrays(arrays)


def test_unknown_calibration_cache_contract_is_rejected():
    arrays = _synthetic_bank().to_arrays()
    arrays["calibration_status"] = np.asarray(["future-or-corrupt-contract"])

    with pytest.raises(
        ValueError,
        match="class_analysis_refinement_bank_calibration_unsupported",
    ):
        ReferenceBank.from_arrays(arrays)


def test_reference_bank_without_calibration_split_provenance_is_rejected():
    arrays = _synthetic_bank().to_arrays()
    arrays.pop("calibration_split_digest")

    with pytest.raises(
        ValueError,
        match=(
            "class_analysis_refinement_bank_calibration_provenance_invalid"
        ),
    ):
        ReferenceBank.from_arrays(arrays)


def test_reference_bank_rejects_incoherent_calibration_split_counts():
    arrays = _synthetic_bank().to_arrays()
    arrays["calibration_target_source_counts"] = np.asarray(
        [5, 2],
        dtype=np.int32,
    )

    with pytest.raises(
        ValueError,
        match=(
            "class_analysis_refinement_bank_calibration_provenance_invalid"
        ),
    ):
        ReferenceBank.from_arrays(arrays)


@pytest.mark.parametrize(
    "storage_key",
    [
        "calibration_heldout_source_count",
        "calibration_target_source_counts",
        "calibration_target_passing_source_counts",
        "fit_background_source_counts",
    ],
)
def test_reference_bank_rejects_fractional_count_storage(storage_key):
    arrays = _synthetic_bank().to_arrays()
    arrays[storage_key] = np.asarray(arrays[storage_key], dtype=np.float32)
    arrays[storage_key].reshape(-1)[0] = 1.9

    with pytest.raises(
        ValueError,
        match="class_analysis_refinement_bank_calibration_provenance_invalid",
    ):
        ReferenceBank.from_arrays(arrays)


def test_reference_bank_rejects_fractional_reliable_storage():
    arrays = _synthetic_bank().to_arrays()
    arrays["reliable"] = np.asarray([1.0, 1.0], dtype=np.float32)

    with pytest.raises(
        ValueError,
        match="class_analysis_refinement_bank_metadata_invalid",
    ):
        ReferenceBank.from_arrays(arrays)


def test_reference_bank_rejects_reliable_tier_with_one_heldout_source():
    arrays = _synthetic_bank().to_arrays()
    arrays["calibration_target_source_counts"][0] = 1

    with pytest.raises(
        ValueError,
        match="class_analysis_refinement_bank_calibration_provenance_invalid",
    ):
        ReferenceBank.from_arrays(arrays)


def test_reference_bank_rejects_reliable_tier_with_one_passing_source():
    arrays = _synthetic_bank().to_arrays()
    arrays["calibration_target_passing_source_counts"][0] = 1
    arrays["calibration_target_source_pass_fractions"][0] = 0.5

    with pytest.raises(
        ValueError,
        match="class_analysis_refinement_bank_calibration_provenance_invalid",
    ):
        ReferenceBank.from_arrays(arrays)


@pytest.mark.parametrize(
    ("storage_key", "replacement"),
    [
        ("anchor_counts", np.asarray([-1, 64], dtype=np.int32)),
        ("distinct_source_counts", np.asarray([-1, 8], dtype=np.int32)),
        ("anchor_counts", np.asarray([7, 64], dtype=np.int32)),
        ("distinct_source_counts", np.asarray([41, 8], dtype=np.int32)),
    ],
)
def test_reference_bank_rejects_impossible_anchor_source_counts(
    storage_key,
    replacement,
):
    arrays = _synthetic_bank().to_arrays()
    arrays[storage_key] = replacement

    with pytest.raises(
        ValueError,
        match="class_analysis_refinement_bank_metadata_invalid",
    ):
        ReferenceBank.from_arrays(arrays)


@pytest.mark.parametrize("invalid_auroc", [-0.01, 1.01])
def test_reference_bank_rejects_auroc_outside_unit_interval(invalid_auroc):
    arrays = _synthetic_bank().to_arrays()
    arrays["heldout_aurocs"][0] = invalid_auroc

    with pytest.raises(
        ValueError,
        match="class_analysis_refinement_bank_calibration_provenance_invalid",
    ):
        ReferenceBank.from_arrays(arrays)


@pytest.mark.parametrize("invalid_name", ["", "   "])
def test_reference_bank_rejects_empty_class_names(invalid_name):
    arrays = _synthetic_bank().to_arrays()
    arrays["class_names"] = np.asarray([invalid_name, "B"])

    with pytest.raises(
        ValueError,
        match="class_analysis_refinement_bank_classes_invalid",
    ):
        ReferenceBank.from_arrays(arrays)


def test_reference_patch_reservoir_balances_sources_exactly():
    builder = StreamingReferenceBankBuilder(
        RefinementConfig(
            patches_per_anchor=8,
            patch_reservoir_per_class=8,
            prototypes_per_class=4,
        )
    )
    for source_index in range(4):
        builder.add(
            class_name="A",
            source_key=f"source-{source_index}",
            patch_tokens=np.eye(8, dtype=np.float32),
        )

    counts: dict[str, int] = {}
    for _rank, source, _value in builder._rows["A"]:
        counts[source] = counts.get(source, 0) + 1
    assert counts == {
        "source-0": 2,
        "source-1": 2,
        "source-2": 2,
        "source-3": 2,
    }


def test_shared_pca_pool_is_class_balanced_before_large_classes_repeat():
    vector = np.asarray([1.0, 0.0], dtype=np.float32)
    rows = {
        "large": [
            (index, f"large-{index}", vector)
            for index in range(10)
        ],
        "small": [
            (100 + index, f"small-{index}", vector)
            for index in range(2)
        ],
    }

    selected = _round_robin_class_rows(rows, limit=4)

    assert [row[1] for row in selected] == [
        "large-0",
        "small-0",
        "large-1",
        "small-1",
    ]


def test_reference_bank_rejects_nonfinite_calibration():
    bank = _synthetic_bank()
    arrays = bank.to_arrays()
    arrays["support_thresholds"] = np.asarray([np.nan, 0.08])

    with pytest.raises(
        ValueError,
        match="class_analysis_refinement_bank_nonfinite",
    ):
        ReferenceBank.from_arrays(arrays)


def test_overlap_index_only_materializes_candidate_images():
    context = [
        {
            "point_id": f"p{index}",
            "split": "train",
            "image_relpath": f"image-{index}.jpg",
            "class_name": "A",
            "bbox_xyxy": [0, 0, 10, 10],
        }
        for index in range(1_000)
    ]
    context.append(
        {
            "point_id": "overlap",
            "split": "train",
            "image_relpath": "image-7.jpg",
            "class_name": "B",
            "bbox_xyxy": [1, 1, 9, 9],
        }
    )

    index = build_overlap_index([context[7]], context)

    assert set(index) == {"p7"}
    assert [row["point_id"] for row in index["p7"]] == ["overlap"]


def test_overlap_annotation_binding_uses_sorted_point_id_for_identical_boxes():
    query = {
        "point_id": "target",
        "class_name": "Truck",
        "split": "train",
        "image_relpath": "frame.jpg",
        "bbox_xyxy": [0.0, 0.0, 100.0, 100.0],
    }
    duplicate_bbox = [10.0, 10.0, 90.0, 90.0]
    context = [
        query,
        {
            **query,
            "point_id": "z-light-vehicle",
            "class_name": "LightVehicle",
            "bbox_xyxy": duplicate_bbox,
        },
        {
            **query,
            "point_id": "a-light-vehicle",
            "class_name": "LightVehicle",
            "bbox_xyxy": duplicate_bbox,
        },
    ]
    overlap_index = build_overlap_index([query], context)
    assert [row["point_id"] for row in overlap_index["target"]] == [
        "a-light-vehicle",
        "z-light-vehicle",
    ]
    evidence = {
        "alternative_class": "LightVehicle",
        "overlap_object_count": 2,
        "annotated_overlap_alternative_bbox_xyxy": duplicate_bbox,
        "annotated_overlap_alternative_point_id": None,
    }

    api._class_analysis_bind_annotated_overlap_alternative_point_id(
        point_id="target",
        evidence=evidence,
        overlap_matches=overlap_index["target"],
    )

    assert evidence["annotated_overlap_alternative_point_id"] == (
        "a-light-vehicle"
    )


def test_overlap_annotation_binding_replays_scorer_dominance_and_fails_closed():
    overlap_matches = [
        {
            "point_id": "incidental-person",
            "class_name": "Person",
            "bbox_xyxy": [0.0, 0.0, 19.0, 50.0],
            "relation": "target_contains_other",
            "iou": 0.095,
            "target_area_covered": 0.095,
        },
        {
            "point_id": "material-person",
            "class_name": "Person",
            "bbox_xyxy": [-1000.0, 0.0, 21.0, 1000.0],
            "relation": "partial_contamination",
            "iou": 0.002057,
            "target_area_covered": 0.21,
        },
    ]
    valid = {
        "alternative_class": "Person",
        "overlap_object_count": 2,
        "annotated_overlap_alternative_bbox_xyxy": [
            -1000.0,
            0.0,
            21.0,
            1000.0,
        ],
        "annotated_overlap_alternative_point_id": None,
    }

    api._class_analysis_bind_annotated_overlap_alternative_point_id(
        point_id="target",
        evidence=valid,
        overlap_matches=overlap_matches,
    )

    assert valid["annotated_overlap_alternative_point_id"] == "material-person"

    non_dominant = {
        **valid,
        "annotated_overlap_alternative_bbox_xyxy": [0.0, 0.0, 19.0, 50.0],
        "annotated_overlap_alternative_point_id": None,
    }
    with pytest.raises(
        ValueError,
        match="class_analysis_refinement_overlap_annotation_binding_failed:target",
    ):
        api._class_analysis_bind_annotated_overlap_alternative_point_id(
            point_id="target",
            evidence=non_dominant,
            overlap_matches=overlap_matches,
        )

    missing_geometry = {
        "alternative_class": "Person",
        "overlap_object_count": 1,
        "annotated_overlap_alternative_bbox_xyxy": None,
        "annotated_overlap_alternative_point_id": None,
    }
    with pytest.raises(
        ValueError,
        match="class_analysis_refinement_overlap_annotation_binding_failed:target",
    ):
        api._class_analysis_bind_annotated_overlap_alternative_point_id(
            point_id="target",
            evidence=missing_geometry,
            overlap_matches=overlap_matches[:1],
        )

    wrong_count = {
        **valid,
        "overlap_object_count": 1,
        "annotated_overlap_alternative_point_id": None,
    }
    with pytest.raises(
        ValueError,
        match="class_analysis_refinement_overlap_annotation_binding_failed:target",
    ):
        api._class_analysis_bind_annotated_overlap_alternative_point_id(
            point_id="target",
            evidence=wrong_count,
            overlap_matches=overlap_matches,
        )


def test_overlap_annotation_selection_prefers_stronger_material_owner():
    higher_iou = {
        "point_id": "higher-iou",
        "class_name": "Person",
        "bbox_xyxy": [0.0, 0.0, 30.0, 70.0],
        "relation": "partial_contamination",
        "iou": 0.24,
        "target_area_covered": 0.21,
    }
    higher_coverage = {
        "point_id": "higher-coverage",
        "class_name": "Person",
        "bbox_xyxy": [-10.0, 0.0, 35.0, 100.0],
        "relation": "partial_contamination",
        "iou": 0.18,
        "target_area_covered": 0.35,
    }
    overlap_matches = [higher_iou, higher_coverage]

    assert max(
        overlap_matches,
        key=overlap_annotation_selection_key,
    )["point_id"] == "higher-coverage"

    evidence = {
        "alternative_class": "Person",
        "overlap_object_count": 2,
        "annotated_overlap_alternative_bbox_xyxy": higher_coverage[
            "bbox_xyxy"
        ],
        "annotated_overlap_alternative_point_id": None,
    }
    api._class_analysis_bind_annotated_overlap_alternative_point_id(
        point_id="target",
        evidence=evidence,
        overlap_matches=overlap_matches,
    )

    assert evidence["annotated_overlap_alternative_point_id"] == (
        "higher-coverage"
    )


def test_all_class_rough_queue_is_not_capped():
    candidates = [
        {
            "point_id": f"p{index}",
            "class_name": "A",
            "suggested_neighbor_class": "B",
        }
        for index in range(5_488)
    ]
    rough, within = api._class_analysis_refinement_rough_candidates(
        {
            "points": copy.deepcopy(candidates),
            "wrong_class_candidates": candidates,
        },
        scope="all_classes",
        config=RefinementConfig(max_candidates=5_000),
    )

    assert len(rough) == 5_488
    assert within == []


def test_normalized_request_persists_refinement_schema_and_alias():
    request = api._normalize_class_analysis_request(
        {
            "refine_wrong_class_candidates": "true",
            "encoder_type": "dinov3",
        }
    )

    assert request["refine_outliers"] is True
    assert request["refinement_schema"] == REFINEMENT_SCHEMA
    assert request["refinement_decision_contract"] == (
        REFINEMENT_DECISION_CONTRACT
    )
    assert request["selector_priority_contract"] == SELECTOR_PRIORITY_CONTRACT
    assert request["capture_group_contract"] == CAPTURE_GROUP_CONTRACT
    capabilities = api._class_analysis_capabilities()[
        "fine_grained_refinement"
    ]
    assert capabilities["api_version"] == 5
    assert capabilities["supported_model_families"] == ["vit"]
    assert capabilities["decision_contract"] == REFINEMENT_DECISION_CONTRACT
    assert capabilities["selector_priority_contract"] == (
        SELECTOR_PRIORITY_CONTRACT
    )
    assert capabilities["frequent_overlap_prior_contract"] == (
        FREQUENT_OVERLAP_PRIOR_CONTRACT
    )
    assert capabilities["capture_group_contract"] == CAPTURE_GROUP_CONTRACT
    assert capabilities["default_enabled"] is False
    assert capabilities["precise_default_enabled"] is False
    assert capabilities["experimental"] is True
    assert capabilities["blocks_use"] is False
    assert "ready_for_default_use" not in capabilities
    assert "release_status" not in capabilities
    with pytest.raises(api.HTTPException) as exc:
        api._normalize_class_analysis_request(
            {
                "refine_outliers": True,
                "refinement_schema": "wrong",
            }
        )
    assert exc.value.detail == "class_analysis_refinement_schema_unsupported"
    convnext_base = api._normalize_class_analysis_request(
        {
            "refine_outliers": True,
            "encoder_type": "dinov3",
            "encoder_model": (
                "facebook/dinov3-convnext-base-pretrain-lvd1689m"
            ),
        }
    )
    assert "convnext" in convnext_base["encoder_model"]
    assert "vit" in convnext_base["deep_evidence_encoder_model"]
    unknown_base = api._normalize_class_analysis_request(
        {
            "refine_outliers": True,
            "encoder_type": "dinov3",
            "encoder_model": "local/unknown-backbone",
        }
    )
    assert unknown_base["encoder_model"] == "local/unknown-backbone"
    with pytest.raises(api.HTTPException) as invalid_evidence_model:
        api._normalize_class_analysis_request(
            {
                "refine_outliers": True,
                "encoder_type": "cradio",
                "deep_evidence_encoder_model": (
                    "facebook/dinov3-convnext-base-pretrain-lvd1689m"
                ),
            }
        )
    assert invalid_evidence_model.value.detail == (
        "class_analysis_refinement_requires_dinov3_vit"
    )


def test_active_workspace_capture_metadata_is_allowlisted_and_preserved():
    rows = api._class_analysis_active_workspace_rows(
        {
            "images": [
                {
                    "upload_name": "original.jpg",
                    "image_name": "original.jpg",
                    "label_lines": ["0 0.5 0.5 0.2 0.2"],
                    "capture_group_id": "capture-17",
                    "sequence_id": "sequence-a",
                    "camera_id": "left",
                    "frame_index": 42,
                    "capture_perceptual_hash": "a" * 32,
                    "capture_perceptual_image_sha256": "b" * 64,
                    "nested_untrusted": {"value": "ignored"},
                }
            ]
        },
        {"original.jpg": "safe.jpg", "safe.jpg": "safe.jpg"},
        {"original.jpg": "b" * 64, "safe.jpg": "b" * 64},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["capture_group_id"] == "capture-17"
    assert row["sequence_id"] == "sequence-a"
    assert row["camera_id"] == "left"
    assert row["frame_index"] == 42
    assert row["capture_perceptual_hash"] == "a" * 32
    assert row["capture_perceptual_image_sha256"] == "b" * 64
    assert "nested_untrusted" not in row


def test_refinement_capability_fails_closed_for_mixed_v33_contract(
    monkeypatch,
):
    monkeypatch.setattr(
        api,
        "CLASS_ANALYSIS_REFINEMENT_DECISION_CONTRACT",
        "class-analysis-patch-decision-v3",
    )

    capability = api._class_analysis_capabilities()[
        "fine_grained_refinement"
    ]

    assert capability["api_version"] == 5
    assert capability["supported"] is False


def test_refinement_decision_contract_invalidates_only_refined_job_reuse(
    monkeypatch,
    tmp_path,
):
    source = {
        "source_mode": "active",
        "source_id": "snapshot-contract",
        "dataset_root": tmp_path,
        "manifest": {"images": []},
        "labelmap": ["A", "B"],
    }
    monkeypatch.setattr(api, "_class_analysis_source", lambda _payload: source)
    monkeypatch.setattr(api, "_class_analysis_require_cleanup_complete", lambda: None)

    class NoopThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

    monkeypatch.setattr(api.threading, "Thread", NoopThread)
    current_request = api._normalize_class_analysis_request(
        {"refine_outliers": True, "encoder_type": "dinov3"}
    )
    legacy_request = dict(current_request)
    legacy_request.pop("refinement_decision_contract")
    legacy_fingerprint = api._class_analysis_run_fingerprint(legacy_request)
    current_fingerprint = api._class_analysis_run_fingerprint(current_request)
    assert current_fingerprint != legacy_fingerprint
    pre_selector_request = dict(current_request)
    pre_selector_request.pop("selector_priority_contract")
    assert api._class_analysis_run_fingerprint(pre_selector_request) != (
        current_fingerprint
    )

    existing = api.ClassAnalysisJob(
        job_id="ca_legacy_contract",
        request={
            **legacy_request,
            "run_fingerprint": legacy_fingerprint,
        },
        status="completed",
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS[existing.job_id] = existing
    with api.CLASS_ANALYSIS_RUN_FINGERPRINTS_LOCK:
        api.CLASS_ANALYSIS_RUN_FINGERPRINTS[legacy_fingerprint] = existing.job_id
    response = None
    try:
        response = api._enqueue_class_analysis_job(current_request)
        assert response["reused"] is False
        assert response["job_id"] != existing.job_id
        assert response["run_fingerprint"] == current_fingerprint
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.pop(existing.job_id, None)
            if response is not None:
                api.CLASS_ANALYSIS_JOBS.pop(response["job_id"], None)
        with api.CLASS_ANALYSIS_RUN_FINGERPRINTS_LOCK:
            api.CLASS_ANALYSIS_RUN_FINGERPRINTS.pop(
                legacy_fingerprint,
                None,
            )
            api.CLASS_ANALYSIS_RUN_FINGERPRINTS.pop(
                current_fingerprint,
                None,
            )

    disabled_plain = api._normalize_class_analysis_request(
        {"refine_outliers": False, "encoder_type": "dinov3"}
    )
    disabled_spoofed = api._normalize_class_analysis_request(
        {
            "refine_outliers": False,
            "encoder_type": "dinov3",
            "refinement_decision_contract": "stale-client-value",
            "selector_priority_contract": "stale-selector-value",
            "capture_group_contract": "stale-capture-value",
        }
    )
    assert "refinement_decision_contract" not in disabled_plain
    assert "selector_priority_contract" not in disabled_plain
    assert "capture_group_contract" not in disabled_plain
    assert disabled_spoofed == disabled_plain
    assert api._class_analysis_run_fingerprint(disabled_spoofed) == (
        api._class_analysis_run_fingerprint(disabled_plain)
    )


def test_refinement_model_family_uses_local_config_not_directory_name(
    tmp_path,
):
    neutral_vit = tmp_path / "checkpoint"
    neutral_vit.mkdir()
    (neutral_vit / "config.json").write_text(
        json.dumps(
            {
                "model_type": "dinov3_vit",
                "architectures": ["DINOv3ViTModel"],
                "patch_size": 16,
                "num_hidden_layers": 12,
            }
        ),
        encoding="utf-8",
    )
    accepted = api._normalize_class_analysis_request(
        {
            "refine_outliers": True,
            "encoder_type": "dinov3",
            "encoder_model": str(neutral_vit),
        }
    )
    assert accepted["encoder_model"] == str(neutral_vit)

    misleading = tmp_path / "looks-like-vit"
    misleading.mkdir()
    (misleading / "config.json").write_text(
        json.dumps(
            {
                "model_type": "dinov3_convnext",
                "architectures": ["DINOv3ConvNextModel"],
            }
        ),
        encoding="utf-8",
    )
    base_accepted = api._normalize_class_analysis_request(
        {
            "refine_outliers": True,
            "encoder_type": "dinov3",
            "encoder_model": str(misleading),
        }
    )
    assert base_accepted["encoder_model"] == str(misleading)
    with pytest.raises(api.HTTPException) as exc:
        api._normalize_class_analysis_request(
            {
                "refine_outliers": True,
                "encoder_type": "dinov3",
                "encoder_model": str(neutral_vit),
                "deep_evidence_encoder_model": str(misleading),
            }
        )
    assert exc.value.detail == (
        "class_analysis_refinement_requires_dinov3_vit"
    )


def test_reference_bank_fingerprint_includes_exact_anchor_set(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        api,
        "mlx_dinov3_status",
        lambda *_args, **_kwargs: SimpleNamespace(
            requested="auto",
            resolved="torch",
            model_dir=tmp_path / "missing-model",
            worker_path=tmp_path / "missing-worker",
        ),
    )
    records = [
        {
            "point_id": point_id,
            "_image_sha256": f"{index + 1:064x}",
            "split": "train",
            "image_relpath": f"{point_id}.jpg",
            "class_name": "A",
            "bbox_xyxy": [0, 0, 10, 10],
        }
        for index, point_id in enumerate(("anchor-a", "anchor-b"))
    ]
    request = {"encoder_model": "test/dinov3", "seed": 42}
    config = RefinementConfig(seed=42)

    first = api._class_analysis_refinement_bank_fingerprint(
        records,
        anchor_records=[records[0]],
        request=request,
        config=config,
    )
    second = api._class_analysis_refinement_bank_fingerprint(
        records,
        anchor_records=[records[1]],
        request=request,
        config=config,
    )

    assert first != second


def test_reference_bank_fingerprint_binds_anchor_ids_to_visual_rows(
    monkeypatch,
):
    monkeypatch.setattr(
        api,
        "_class_analysis_refinement_encoder_identity",
        lambda *_args, **_kwargs: ({"model": "stable-test-model"}, True),
    )
    records = [
        {
            "point_id": "anchor-a",
            "_image_sha256": "1" * 64,
            "split": "train",
            "image_relpath": "a.jpg",
            "class_name": "A",
            "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
        },
        {
            "point_id": "anchor-b",
            "_image_sha256": "2" * 64,
            "split": "train",
            "image_relpath": "b.jpg",
            "class_name": "B",
            "bbox_xyxy": [10.0, 10.0, 20.0, 20.0],
        },
    ]
    swapped = copy.deepcopy(records)
    swapped[0]["point_id"], swapped[1]["point_id"] = (
        swapped[1]["point_id"],
        swapped[0]["point_id"],
    )
    request = {"encoder_model": "test/dinov3", "seed": 42}
    config = RefinementConfig(seed=42)

    first = api._class_analysis_refinement_bank_fingerprint(
        records,
        anchor_records=records,
        request=request,
        config=config,
    )
    second = api._class_analysis_refinement_bank_fingerprint(
        swapped,
        anchor_records=swapped,
        request=request,
        config=config,
    )

    assert first != second


def test_reference_bank_fingerprint_binds_spatial_runtime_versions_and_device(
    monkeypatch,
):
    status = SimpleNamespace(resolved="torch")
    versions = {
        "Pillow": "10.4.0",
        "torch": "2.7.0",
        "transformers": "4.53.0",
    }
    monkeypatch.setattr(
        api.importlib_metadata,
        "version",
        lambda name: versions[name],
    )
    monkeypatch.setattr(
        api,
        "_dinov3_resolve_device_impl",
        lambda *_args, **_kwargs: "mps",
    )
    monkeypatch.setattr(api.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(api.platform, "release", lambda: "25.0.0")
    monkeypatch.setattr(api.platform, "machine", lambda: "arm64")

    first = api._class_analysis_refinement_spatial_runtime_identity(status)
    assert first["pillow_version"] == "10.4.0"
    assert first["torch_version"] == "2.7.0"
    assert first["transformers_version"] == "4.53.0"
    assert first["torch_device"] == "mps"

    versions["transformers"] = "4.54.0"
    second = api._class_analysis_refinement_spatial_runtime_identity(status)
    assert second != first


def test_selected_fraction_invalidates_reference_bank_and_result_fingerprints(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        api,
        "_class_analysis_refinement_encoder_identity",
        lambda *_args, **_kwargs: ({"model": "stable-test-model"}, True),
    )
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    record = {
        "point_id": "anchor",
        "_image_sha256": "1" * 64,
        "split": "train",
        "image_relpath": "anchor.jpg",
        "class_name": "A",
        "bbox_xyxy": [0.0, 0.0, 20.0, 20.0],
    }
    request = {
        "encoder_model": "test/dinov3",
        "seed": 42,
        "refine_outliers": True,
        "refinement_schema": api.CLASS_ANALYSIS_REFINEMENT_SCHEMA,
        "refinement_decision_contract": (
            api.CLASS_ANALYSIS_REFINEMENT_DECISION_CONTRACT
        ),
    }
    base_config = RefinementConfig(seed=42, selected_fraction=0.05)
    changed_config = RefinementConfig(seed=42, selected_fraction=0.20)

    base_bank_fingerprint = api._class_analysis_refinement_bank_fingerprint(
        [record],
        anchor_records=[record],
        request=request,
        config=base_config,
    )
    changed_bank_fingerprint = api._class_analysis_refinement_bank_fingerprint(
        [record],
        anchor_records=[record],
        request=request,
        config=changed_config,
    )
    assert changed_bank_fingerprint != base_bank_fingerprint

    api._class_analysis_write_refinement_bank(
        base_bank_fingerprint,
        bank=_synthetic_bank(),
        grid_shape=(2, 2),
        calibration_diagnostics=_empty_exact_view_calibration(),
    )
    assert api._class_analysis_load_refinement_bank(base_bank_fingerprint) is not None
    assert api._class_analysis_load_refinement_bank(changed_bank_fingerprint) is None

    source = {
        "source_mode": "active",
        "source_id": "selected-fraction-snapshot",
        "dataset_root": tmp_path,
        "manifest": {"images": []},
        "labelmap": ["A", "B"],
    }
    monkeypatch.setattr(api, "_class_analysis_source", lambda _payload: source)
    base_result_fingerprint = api._class_analysis_run_fingerprint(request)
    monkeypatch.setattr(
        api,
        "CLASS_ANALYSIS_REFINEMENT_SELECTED_FRACTION",
        0.20,
    )
    changed_result_fingerprint = api._class_analysis_run_fingerprint(request)
    assert changed_result_fingerprint != base_result_fingerprint


def test_reference_bank_builder_epoch_invalidates_interim_v33_cache(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        api,
        "_class_analysis_refinement_encoder_identity",
        lambda *_args, **_kwargs: ({"model": "stable-test-model"}, True),
    )
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    record = {
        "point_id": "anchor",
        "_image_sha256": "1" * 64,
        "split": "train",
        "image_relpath": "anchor.jpg",
        "class_name": "A",
        "bbox_xyxy": [0.0, 0.0, 20.0, 20.0],
    }
    request = {"encoder_model": "test/dinov3", "seed": 42}
    config = RefinementConfig(seed=42)
    current_epoch = api.CLASS_ANALYSIS_REFINEMENT_V33_BANK_BUILDER_EPOCH
    monkeypatch.setattr(
        api,
        "CLASS_ANALYSIS_REFINEMENT_V33_BANK_BUILDER_EPOCH",
        "interim-v33-before-coverage-priority",
    )
    interim_fingerprint = api._class_analysis_refinement_bank_fingerprint(
        [record],
        anchor_records=[record],
        request=request,
        config=config,
    )
    api._class_analysis_write_refinement_bank(
        interim_fingerprint,
        bank=_synthetic_bank(),
        grid_shape=(2, 2),
        calibration_diagnostics=_empty_exact_view_calibration(),
    )

    monkeypatch.setattr(
        api,
        "CLASS_ANALYSIS_REFINEMENT_V33_BANK_BUILDER_EPOCH",
        current_epoch,
    )
    current_fingerprint = api._class_analysis_refinement_bank_fingerprint(
        [record],
        anchor_records=[record],
        request=request,
        config=config,
    )

    assert current_fingerprint != interim_fingerprint
    assert api._class_analysis_load_refinement_bank(current_fingerprint) is None
    assert api._class_analysis_load_refinement_bank(interim_fingerprint) is not None


def test_reference_bank_fingerprint_preserves_sub_millipixel_geometry(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        api,
        "mlx_dinov3_status",
        lambda *_args, **_kwargs: SimpleNamespace(
            requested="auto",
            resolved="torch",
            model_dir=tmp_path / "missing-model",
            worker_path=tmp_path / "missing-worker",
        ),
    )
    first_record = {
        "point_id": "anchor",
        "_image_sha256": "1" * 64,
        "split": "train",
        "image_relpath": "anchor.jpg",
        "class_name": "A",
        "bbox_xyxy": [10.9996, 0.0, 20.0, 20.0],
    }
    second_record = copy.deepcopy(first_record)
    second_record["bbox_xyxy"] = [11.0004, 0.0, 20.0, 20.0]
    request = {"encoder_model": "test/dinov3", "seed": 42}
    config = RefinementConfig(seed=42)

    first = api._class_analysis_refinement_bank_fingerprint(
        [first_record],
        anchor_records=[first_record],
        request=request,
        config=config,
    )
    second = api._class_analysis_refinement_bank_fingerprint(
        [second_record],
        anchor_records=[second_record],
        request=request,
        config=config,
    )

    assert first != second


def test_reference_bank_fingerprint_tracks_resolved_torch_checkpoint_assets(
    monkeypatch,
    tmp_path,
):
    torch_model = tmp_path / "torch-model"
    torch_model.mkdir()
    (torch_model / "config.json").write_text(
        json.dumps({"model_type": "dinov3_vit"}),
        encoding="utf-8",
    )
    torch_weights = torch_model / "model.safetensors"
    torch_weights.write_bytes(b"weight-version-a")
    unrelated_mlx = tmp_path / "mlx-conversion"
    unrelated_mlx.mkdir()
    (unrelated_mlx / "config.json").write_text("{}", encoding="utf-8")
    (unrelated_mlx / "model.safetensors").write_bytes(b"unrelated-a")
    monkeypatch.setattr(
        api,
        "mlx_dinov3_status",
        lambda *_args, **_kwargs: SimpleNamespace(
            requested="auto",
            resolved="torch",
            model_dir=unrelated_mlx,
            worker_path=tmp_path / "mlx-worker.py",
        ),
    )
    monkeypatch.setattr(api, "dinov3_model", None)
    monkeypatch.setattr(api, "dinov3_model_name", None)
    record = {
        "point_id": "anchor",
        "_image_sha256": "1" * 64,
        "split": "train",
        "image_relpath": "anchor.jpg",
        "class_name": "A",
        "bbox_xyxy": [0.0, 0.0, 20.0, 20.0],
    }
    request = {"encoder_model": str(torch_model), "seed": 42}
    config = RefinementConfig(seed=42)

    first = api._class_analysis_refinement_bank_fingerprint(
        [record],
        anchor_records=[record],
        request=request,
        config=config,
    )
    assert getattr(first, "cache_reusable") is True
    (unrelated_mlx / "model.safetensors").write_bytes(b"unrelated-b")
    after_unrelated_mlx_change = api._class_analysis_refinement_bank_fingerprint(
        [record],
        anchor_records=[record],
        request=request,
        config=config,
    )
    assert after_unrelated_mlx_change == first

    original_stat = torch_weights.stat()
    torch_weights.write_bytes(b"weight-version-b")
    os.utime(
        torch_weights,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    after_actual_weight_change = api._class_analysis_refinement_bank_fingerprint(
        [record],
        anchor_records=[record],
        request=request,
        config=config,
    )
    assert after_actual_weight_change != first
    unverified_revision = api._class_analysis_refinement_bank_fingerprint(
        [record],
        anchor_records=[record],
        request={**request, "encoder_revision": "deadbeef"},
        config=config,
    )
    assert getattr(unverified_revision, "cache_reusable") is False


def test_unresolved_refinement_model_identity_disables_bank_cache(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        api,
        "mlx_dinov3_status",
        lambda *_args, **_kwargs: SimpleNamespace(
            requested="auto",
            resolved="torch",
            model_dir=tmp_path / "unrelated",
            worker_path=tmp_path / "missing-worker",
        ),
    )
    monkeypatch.setattr(api, "_qwen_cache_snapshot_path", lambda _model: None)
    monkeypatch.setattr(api, "dinov3_model", None)
    monkeypatch.setattr(api, "dinov3_model_name", None)
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    record = {
        "point_id": "anchor",
        "_image_sha256": "2" * 64,
        "split": "train",
        "image_relpath": "anchor.jpg",
        "class_name": "A",
        "bbox_xyxy": [0.0, 0.0, 20.0, 20.0],
    }

    fingerprint = api._class_analysis_refinement_bank_fingerprint(
        [record],
        anchor_records=[record],
        request={"encoder_model": "uncached/mutable-model", "seed": 42},
        config=RefinementConfig(seed=42),
    )

    assert getattr(fingerprint, "cache_reusable") is False
    assert api._class_analysis_refinement_bank_cache_path(
        fingerprint,
        create=True,
    ) is None


def test_truncated_checkpoint_asset_scan_disables_bank_cache(
    monkeypatch,
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setattr(
        api,
        "mlx_dinov3_status",
        lambda *_args, **_kwargs: SimpleNamespace(
            requested="torch",
            resolved="torch",
            model_dir=None,
            worker_path=None,
        ),
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_model_asset_directory_identity",
        lambda _path: (
            {
                "path": str(checkpoint),
                "assets": [{"name": "model.safetensors"}],
                "asset_scan_truncated": True,
            },
            True,
        ),
    )
    monkeypatch.setattr(api, "dinov3_model", None)
    monkeypatch.setattr(api, "dinov3_model_name", None)

    identity, reusable = api._class_analysis_refinement_encoder_identity(
        str(checkpoint),
        requested_revision="",
    )

    assert reusable is False
    assert identity["checkpoint_assets"]["asset_scan_truncated"] is True
    assert identity["cache_reusable"] is False


def test_job_memory_tracking_keeps_pre_collection_baseline():
    job = api.ClassAnalysisJob(job_id="ca_memory_baseline")
    first = api._class_analysis_track_job_memory(
        job,
        current={
            "backend_rss_bytes": 80,
            "worker_rss_bytes": 20,
            "combined_rss_bytes": 100,
            "system_available_bytes": 900,
            "system_total_bytes": 1_000,
        },
    )
    second = api._class_analysis_track_job_memory(
        job,
        current={
            "backend_rss_bytes": 130,
            "worker_rss_bytes": 30,
            "combined_rss_bytes": 160,
            "system_available_bytes": 840,
            "system_total_bytes": 1_000,
        },
    )

    assert first["job_start_baseline_combined_rss_bytes"] == 100
    assert second["job_start_baseline_combined_rss_bytes"] == 100
    assert second["peak_job_combined_rss_bytes"] == 160
    assert second["peak_job_incremental_combined_rss_bytes"] == 60
    assert job.runtime["job_memory"] == second


def test_publication_memory_metrics_update_both_result_summaries():
    result = {
        "summary": {
            "runtime": {"existing": True},
            "refinement": {
                "status": "completed",
                "resource_metrics": {"sidecar_bytes": 123},
            },
        },
        "refinement_summary": {
            "status": "completed",
            "resource_metrics": {"sidecar_bytes": 123},
        },
    }
    metrics = {
        "job_start_baseline_combined_rss_bytes": 100,
        "peak_job_combined_rss_bytes": 180,
        "peak_job_incremental_combined_rss_bytes": 80,
    }

    api._class_analysis_embed_publication_memory_metrics(result, metrics)

    top = result["refinement_summary"]["resource_metrics"]
    nested = result["summary"]["refinement"]["resource_metrics"]
    assert top == nested
    assert top["sidecar_bytes"] == 123
    assert top["peak_job_incremental_combined_rss_bytes"] == 80
    assert top["publication_memory_scope"].startswith(
        "through_public_result_materialization"
    )
    assert top["result_json_writer"] == "json_encoder_iterencode_atomic"
    assert result["summary"]["runtime"]["existing"] is True
    assert (
        result["summary"]["runtime"]["job_memory"]
        ["peak_job_combined_rss_bytes"]
        == 180
    )


def test_record_collection_preserves_sub_millipixel_crop_identity(
    monkeypatch,
    tmp_path,
):
    class_root = tmp_path / "class-analysis"
    workspace = class_root / "workspace"
    images_dir = workspace / "images"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), (20, 40, 60)).save(
        images_dir / "frame.png"
    )
    manifest_path = workspace / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_label": "exact geometry",
                "labelmap": ["A"],
                "images": [
                    {
                        "split": "train",
                        "image_relpath": "frame.png",
                        "label_lines": [
                            "0 0.500000 0.500000 0.250000 0.250000",
                            "0 0.500006 0.500000 0.250000 0.250000",
                        ],
                    }
                ],
                "yolo_layout": "flat",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job = api.ClassAnalysisJob(job_id="ca_exact_geometry")

    records, crops, summary = api._class_analysis_collect_records(
        {
            "source_mode": "active_workspace",
            "workspace_id": "ca_exact_geometry",
            "workspace_dir": str(workspace),
            "workspace_manifest_path": str(manifest_path),
            "analysis_scope": "all_classes",
            "encoder_type": "dinov3",
            "encoder_model": "test-dino",
        },
        job=job,
        out_dir=class_root / job.job_id,
        materialize_crops=False,
    )

    assert crops == []
    assert summary["object_count"] == 2
    first_x = float(records[0]["bbox_xyxy"][0])
    second_x = float(records[1]["bbox_xyxy"][0])
    assert 0.0 < abs(second_x - first_x) < 0.001
    assert records[0]["crop_cache_key"] != records[1]["crop_cache_key"]


def test_selected_class_spatial_context_is_retained_only_for_refinement(
    monkeypatch,
    tmp_path,
):
    class_root = tmp_path / "class-analysis"
    workspace = class_root / "workspace"
    images_dir = workspace / "images"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), (20, 40, 60)).save(
        images_dir / "frame.png"
    )
    manifest_path = workspace / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_label": "spatial context gate",
                "labelmap": ["A", "B"],
                "images": [
                    {
                        "split": "train",
                        "image_relpath": "frame.png",
                        "label_lines": [
                            "0 0.25 0.50 0.20 0.20",
                            "1 0.75 0.50 0.20 0.20",
                        ],
                    }
                ],
                "yolo_layout": "flat",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    base_payload = {
        "source_mode": "active_workspace",
        "workspace_id": "ca_spatial_context",
        "workspace_dir": str(workspace),
        "workspace_manifest_path": str(manifest_path),
        "analysis_scope": "selected_class",
        "class_name": "A",
        "encoder_type": "dinov3",
        "encoder_model": "test-dino",
    }

    stage1_records, stage1_crops, stage1_summary = (
        api._class_analysis_collect_records(
            {**base_payload, "refine_outliers": False},
            job=api.ClassAnalysisJob(job_id="ca_stage1_context_gate"),
            out_dir=class_root / "ca_stage1_context_gate",
            materialize_crops=False,
        )
    )
    refined_records, refined_crops, refined_summary = (
        api._class_analysis_collect_records(
            {**base_payload, "refine_outliers": True},
            job=api.ClassAnalysisJob(job_id="ca_refined_context_gate"),
            out_dir=class_root / "ca_refined_context_gate",
            materialize_crops=False,
        )
    )

    assert stage1_crops == refined_crops == []
    assert [row["class_name"] for row in stage1_records] == ["A"]
    assert [row["class_name"] for row in refined_records] == ["A"]
    assert stage1_summary["_spatial_context_records"] == []
    assert sorted(
        row["class_name"]
        for row in refined_summary["_spatial_context_records"]
    ) == ["A", "B"]


@pytest.mark.parametrize(
    "scope",
    ["selected_class", "all_classes"],
)
def test_stage2_discovers_cross_class_pair_conflict_outside_stage1_records(
    monkeypatch,
    tmp_path,
    scope,
):
    target = {
        "point_id": "target",
        "class_name": "A",
        "split": "train",
        "image_relpath": "frame.jpg",
        "bbox_xyxy": [4.0, 4.0, 28.0, 28.0],
        "width": 24,
        "height": 24,
        "outlier_score": 1.0,
        "wrong_class_suspicion": 0.95,
        "is_wrong_class_candidate": scope == "all_classes",
    }
    paired = {
        "point_id": "paired-building",
        "class_name": "B",
        "split": "train",
        "image_relpath": "frame.jpg",
        "bbox_xyxy": [4.0, 4.0, 28.0, 28.0],
        "width": 24,
        "height": 24,
    }
    records = [dict(target)]
    points = [dict(target)]
    wrong_candidates = []
    if scope == "selected_class":
        for index in range(1, 20):
            point = {
                **target,
                "point_id": f"ordinary-{index}",
                "image_relpath": f"ordinary-{index}.jpg",
                "outlier_score": 0.0,
                "wrong_class_suspicion": 0.0,
                "is_wrong_class_candidate": False,
            }
            points.append(dict(point))
            records.append(dict(point))
    else:
        wrong_candidates = [
            {
                "point_id": "target",
                "class_name": "A",
                "suggested_neighbor_class": "B",
                "wrong_class_suspicion": 0.95,
            }
        ]
    result = {
        "summary": {
            "analysis_scope": scope,
            "sampled": scope == "all_classes",
            "source_mode": "linked",
            "source_id": "dataset",
            "labelmap": ["A", "B"],
        },
        "points": points,
        "wrong_class_candidates": wrong_candidates,
    }
    job = api.ClassAnalysisJob(
        job_id=f"ca_pair_{scope}",
        status="running",
        request={"refine_outliers": True},
    )
    out_dir = tmp_path / scope
    out_dir.mkdir()
    monkeypatch.setattr(
        api,
        "_class_analysis_memory_snapshot",
        lambda: {
            "combined_rss_bytes": 100,
            "system_total_bytes": 1_000,
            "system_available_bytes": 900,
        },
    )

    refined = api._class_analysis_refine_result(
        records=records,
        # The paired B object deliberately exists only in full spatial
        # context, as happens for selected-class and sample-limited runs.
        spatial_context_records=[*records, paired],
        result=result,
        request=job.request,
        job=job,
        out_dir=out_dir,
    )

    assert refined["wrong_class_candidates"] == wrong_candidates
    assert [
        candidate["point_id"]
        for candidate in refined["vignette_candidates"]
    ] == ["target"]
    assert refined["refinement_summary"]["status"] == "completed"
    prior_summary = refined["refinement_summary"]["selector_priority"][
        "frequent_overlap_prior"
    ]
    assert prior_summary["fit_screening_scope"] == scope
    assert prior_summary["fit_screening_exhaustive"] is (
        scope != "all_classes"
    )
    assert prior_summary["stage1_screened_point_id_count"] == len(records)
    assert prior_summary["stage1_screened_record_count"] == len(records)
    assert prior_summary["excluded_unscreened_annotation_record_count"] == 1
    assert prior_summary["observed_directed_pair_count"] == 0
    assert refined["refinement_summary"]["category_counts"] == {
        STATUS_PAIR_CONFLICT: 1
    }
    assert refined["refinement_summary"][
        "qualified_human_review_candidate_count"
    ] == 0
    assert refined["refinement_summary"]["queue_policy"] == {
        "mode": "selector_ranked_complete_stage1",
        "automatic_rough_fallback": False,
        "fallback_reason": "",
        "default_queue": "selector_ranked_stage1_candidates",
        "effective_default_candidate_count": 1,
        "confirmed_count": 0,
        "pair_conflict_count": 1,
        "refined_review_candidate_count": 1,
        "rough_count": 1,
    }
    row = refined["refinement_candidates"][0]
    assert row["point_id"] == "target"
    assert row["refined_outlier"]["status"] == STATUS_PAIR_CONFLICT
    assert row["refined_outlier"]["overlap_relation"] == "duplicate_like"
    conflict = row["refined_outlier"]["pair_conflict"]
    assert conflict["point_id"] == "target"
    assert conflict["other_point_id"] == "paired-building"
    assert conflict["other_class_name"] == "B"
    assert conflict["iou"] == pytest.approx(1.0)
    assert conflict["target_area_covered"] == pytest.approx(1.0)
    assert conflict["other_area_covered"] == pytest.approx(1.0)
    assert conflict["target_bbox_xyxy"] == [4.0, 4.0, 28.0, 28.0]
    assert conflict["other_bbox_xyxy"] == [4.0, 4.0, 28.0, 28.0]
    assert conflict["discovered_by"] == (
        "patch_refinement_spatial_context"
    )
    assert row["refined_outlier"]["overlap_object_count"] == 1
    assert row["refined_outlier"][
        "annotated_overlap_alternative_bbox_xyxy"
    ] == [4.0, 4.0, 28.0, 28.0]
    assert row["refined_outlier"][
        "annotated_overlap_alternative_point_id"
    ] == "paired-building"
    refined_point = next(
        point
        for point in refined["points"]
        if point["point_id"] == "target"
    )
    assert refined_point["is_dual_bbox_conflict"] is True
    assert refined_point["dual_bbox_conflict"] == conflict
    qwen_conflict = api._class_analysis_qwen_review_dual_bbox_conflict(
        refined_point
    )
    assert qwen_conflict["review_mode"] == "dual_bbox_annotation_resolution"
    assert qwen_conflict["other_point_id"] == "paired-building"


def test_pair_conflict_merge_preserves_complete_scorer_overlap_accounting():
    scored = {
        "status": "unresolved",
        "reason_codes": ["pair_probe_unreliable"],
        "alternative_class": "Building",
        "overlap_object_count": 2,
        "annotated_overlap_alternative_bbox_xyxy": [10.0, 10.0, 30.0, 30.0],
        "annotated_overlap_alternative_point_id": None,
        "decision_gates": {"annotated_overlap": True},
        "_sidecar": {"point_id": "target"},
    }
    pair = {
        "reason_codes": ["near_identical_cross_class_bbox_requires_review"],
        "overlap_relation": "duplicate_like",
        "overlap_object_count": 1,
        "annotated_overlap_alternative_bbox_xyxy": [10.0, 10.0, 30.0, 30.0],
        "annotated_overlap_alternative_point_id": "exact",
        "pair_conflict": {"other_point_id": "exact"},
        "decision_gates": {"nested_overlap": True},
    }
    overlaps = [
        {
            "point_id": "weak",
            "class_name": "Building",
            "bbox_xyxy": [24.0, 24.0, 44.0, 44.0],
            "iou": 0.1,
            "target_area_covered": 0.2,
        },
        {
            "point_id": "exact",
            "class_name": "Building",
            "bbox_xyxy": [10.0, 10.0, 30.0, 30.0],
            "iou": 1.0,
            "target_area_covered": 1.0,
        },
    ]

    merged = api._class_analysis_merge_pair_conflict_scored_evidence(
        scored_evidence=scored,
        pair_conflict_evidence=pair,
        alternative_class="Building",
        overlap_matches=overlaps,
    )
    api._class_analysis_bind_annotated_overlap_alternative_point_id(
        point_id="target",
        evidence=merged,
        overlap_matches=overlaps,
    )

    assert merged["status"] == api.CLASS_ANALYSIS_REFINEMENT_PAIR_CONFLICT
    assert merged["overlap_object_count"] == 2
    assert merged["annotated_overlap_alternative_point_id"] == "exact"
    assert merged["pair_conflict"]["other_point_id"] == "exact"
    assert merged["_sidecar"] == {"point_id": "target"}


def test_v9_no_candidate_summary_is_valid_experimental_selector_result(tmp_path):
    job = api.ClassAnalysisJob(
        job_id="ca_v9_no_candidates",
        status="running",
        request={"refine_outliers": True},
    )
    result = {
        "summary": {"analysis_scope": "all_classes"},
        "points": [],
        "wrong_class_candidates": [],
    }

    refined = api._class_analysis_refine_result(
        records=[],
        spatial_context_records=[],
        result=result,
        request=job.request,
        job=job,
        out_dir=tmp_path,
    )

    summary = refined["refinement_summary"]
    assert summary["quality_status"] == "no_candidates"
    assert summary["diagnostic_pair_reliability_contract"] == (
        DIAGNOSTIC_PAIR_RELIABILITY_CONTRACT
    )
    assert summary["human_review_qualification_contract"] == (
        HUMAN_REVIEW_QUALIFICATION_CONTRACT
    )
    assert summary["human_review_rank_contract"] == HUMAN_REVIEW_RANK_CONTRACT
    assert summary["selector_priority_contract"] == SELECTOR_PRIORITY_CONTRACT
    assert summary["selector_priority_candidate_count"] == 0
    assert summary["selector_priority"]["candidate_count"] == 0
    assert summary["selector_evaluation"] == {
        "status": "experimental",
        "blocks_use": False,
    }
    assert summary["qualified_human_review_candidate_count"] == 0
    assert summary["queue_policy"] == {
        "mode": "selector_ranked_complete_stage1",
        "automatic_rough_fallback": False,
        "fallback_reason": "",
        "default_queue": "selector_ranked_stage1_candidates",
        "effective_default_candidate_count": 0,
        "confirmed_count": 0,
        "pair_conflict_count": 0,
        "refined_review_candidate_count": 0,
        "rough_count": 0,
    }


def test_within_class_review_disposition_is_counted_once_and_hidden(
    monkeypatch,
):
    key = "carv_" + ("a" * 64)
    monkeypatch.setattr(
        api,
        "_class_analysis_lookup_review_dispositions",
        lambda _keys: {
            key: {
                "disposition": "confirm_current",
                "updated_at": 123.0,
                "origin": "desktop",
            }
        },
    )
    result = {
        "summary": {},
        "points": [
            {
                "point_id": "within",
                "review_object_key": key,
                "is_wrong_class_candidate": True,
            }
        ],
        "wrong_class_candidates": [],
        "refinement_candidates": [
            {"point_id": "within", "review_object_key": key}
        ],
        "vignette_candidates": [
            {"point_id": "within", "review_object_key": key}
        ],
        "within_class_outlier_candidates": [
            {"point_id": "within", "review_object_key": key}
        ],
    }

    overlaid = api._class_analysis_apply_review_dispositions(result)

    assert overlaid["within_class_outlier_candidates"] == []
    assert overlaid["refinement_candidates"] == []
    assert overlaid["vignette_candidates"] == []
    assert overlaid["summary"]["human_review_disposition_counts"] == {
        "confirm_current": 1
    }
    assert overlaid["summary"]["human_reviewed_candidate_count"] == 1
    assert (
        overlaid["summary"]["human_reviewed_wrong_class_candidate_count"]
        == 1
    )
    assert overlaid["points"][0]["is_wrong_class_candidate"] is False


def test_selector_review_disposition_preserves_complete_stage1_artifact(
    monkeypatch,
):
    key = "carv_" + ("b" * 64)
    monkeypatch.setattr(
        api,
        "_class_analysis_lookup_review_dispositions",
        lambda _keys: {
            key: {
                "disposition": "confirm_current",
                "updated_at": 123.0,
                "origin": "desktop",
            }
        },
    )
    candidate = {"point_id": "rough", "review_object_key": key}
    result = {
        "summary": {},
        "refinement_summary": {
            "selector_priority_contract": SELECTOR_PRIORITY_CONTRACT,
            "selector_priority": {
                "contract": SELECTOR_PRIORITY_CONTRACT,
                "candidate_count": 1,
            },
        },
        "points": [
            {
                "point_id": "rough",
                "review_object_key": key,
                "is_wrong_class_candidate": True,
            }
        ],
        "wrong_class_candidates": [dict(candidate)],
        "refinement_candidates": [dict(candidate)],
        "vignette_candidates": [dict(candidate)],
        "within_class_outlier_candidates": [],
    }

    overlaid = api._class_analysis_apply_review_dispositions(result)

    assert len(overlaid["wrong_class_candidates"]) == 1
    assert len(overlaid["refinement_candidates"]) == 1
    assert len(overlaid["vignette_candidates"]) == 1
    assert overlaid["summary"]["complete_selector_queue_preserved"] is True
    assert overlaid["summary"]["wrong_class_candidate_count"] == 1
    assert overlaid["summary"]["visible_wrong_class_candidate_count"] == 0
    assert overlaid["summary"]["refinement_candidate_count"] == 1
    assert overlaid["summary"]["visible_refinement_candidate_count"] == 0
    assert overlaid["points"][0]["human_review_disposition"] == (
        "confirm_current"
    )


def test_refinement_orchestration_success_reaches_completed_summary(
    monkeypatch,
    tmp_path,
):
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (32, 32), (80, 100, 120)).save(image_path)
    point = {
        "point_id": "candidate",
        "class_name": "A",
        "suggested_neighbor_class": "B",
        "split": "train",
        "image_relpath": "frame.jpg",
        "bbox_xyxy": [0, 0, 32, 32],
        "width": 32,
        "height": 32,
        "wrong_class_suspicion": 0.9,
        "is_wrong_class_candidate": True,
    }
    record = {
        **point,
        "_image_path": str(image_path),
        "_image_sha256": api._class_analysis_file_sha256(image_path),
    }
    result = {
        "summary": {"analysis_scope": "all_classes"},
        "points": [point],
        "wrong_class_candidates": [
            {
                "point_id": "candidate",
                "class_name": "A",
                "suggested_neighbor_class": "B",
                "wrong_class_suspicion": 0.9,
            }
        ],
    }
    job = api.ClassAnalysisJob(
        job_id="ca_refinement_success",
        status="running",
        request={
            "encoder_model": "test/dinov3",
            "refine_outliers": True,
        },
    )
    out_dir = tmp_path / "job"
    out_dir.mkdir()
    corrupt_cache = out_dir / f"{'f' * 64}.npz"
    corrupt_cache.write_bytes(b"not-an-npz")
    bank = _synthetic_bank()
    class_b = np.asarray([0.0, 1.0], dtype=np.float32)
    encoded = np.stack(
        [np.stack([class_b] * 4), np.stack([class_b] * 4)],
        axis=0,
    )
    exact_calibration = {
        "contract": "class-analysis-exact-view-calibration-pass-v1",
        "eligible_example_count": 1,
        "accepted_example_count": 1,
        "skipped_example_count": 0,
        "per_class_accepted_source_counts": {"A": 1},
    }
    monkeypatch.setattr(
        api,
        "_class_analysis_refinement_anchor_records",
        lambda *_args, **_kwargs: [record],
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_refinement_bank_fingerprint",
        lambda *_args, **_kwargs: "f" * 64,
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_load_refinement_bank",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_refinement_bank_cache_path",
        lambda *_args, **_kwargs: corrupt_cache,
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_build_refinement_bank",
        lambda **_kwargs: (
            bank,
            (2, 2),
            "test",
            exact_calibration,
        ),
    )
    wrote_bank = []
    monkeypatch.setattr(
        api,
        "_class_analysis_write_refinement_bank",
        lambda *_args, **_kwargs: wrote_bank.append(True),
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_encode_dinov3_spatial_batch",
        lambda *_args, **_kwargs: (encoded, (2, 2), "test"),
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_memory_snapshot",
        lambda: {
            "combined_rss_bytes": 100,
            "system_total_bytes": 1_000,
            "system_available_bytes": 900,
        },
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_refinement_memory_snapshot",
        lambda _baseline, **_kwargs: {
            "combined_rss_bytes": 100,
            "incremental_combined_rss_bytes": 0,
        },
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_refinement_progress",
        lambda *_args, **_kwargs: None,
    )

    refined = api._class_analysis_refine_result(
        records=[record],
        spatial_context_records=[record],
        result=result,
        request=job.request,
        job=job,
        out_dir=out_dir,
    )

    assert refined["refinement_summary"]["status"] == "completed"
    assert refined["refinement_summary"]["evaluated_count"] == 1
    assert refined["refinement_summary"]["reference_bank_cache_hit"] is False
    assert refined["refinement_summary"]["calibration_split"] == (
        bank.calibration_split_provenance()
    )
    assert refined["refinement_summary"]["warnings"] == [
        "reference_bank_cache_invalid_rebuilt"
    ]
    assert wrote_bank == [True]
    assert any(
        "Invalid spatial reference-bank cache entry" in log["message"]
        for log in job.logs
    )
    assert refined["refinement_summary"]["phase_timings_seconds"][
        "rough_selection"
    ] >= 0.0
    assert refined["refinement_candidates"][0]["point_id"] == "candidate"
    assert refined["vignette_candidates"][0]["point_id"] == "candidate"
    assert (
        refined["vignette_candidates"][0]["refined_outlier"]["status"]
        == STATUS_CONFIRMED_OUTLIER
    )
    assert refined["refinement_summary"]["queue_policy"][
        "default_queue"
    ] == "selector_ranked_stage1_candidates"
    assert refined["refinement_summary"]["queue_policy"][
        "automatic_rough_fallback"
    ] is False
    assert refined["refinement_summary"]["queue_policy"][
        "effective_default_candidate_count"
    ] == len(refined["refinement_candidates"])
    assert (
        refined["vignette_candidates"][0]["refined_outlier"][
            "source_image_sha256"
        ]
        == record["_image_sha256"]
    )
    manifest = json.loads(
        (
            out_dir / api.CLASS_ANALYSIS_REFINEMENT_MANIFEST_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert manifest["calibration_split"] == (
        bank.calibration_split_provenance()
    )
    assert refined["refinement_summary"][
        "exact_view_pair_calibration"
    ] == exact_calibration
    assert manifest["exact_view_pair_calibration"] == (
        refined["refinement_summary"]["exact_view_pair_calibration"]
    )


@pytest.mark.parametrize(
    "cancel_on_filename",
    [
        None,
        "result.json",
        api.CLASS_ANALYSIS_JOB_STATE_FILENAME,
        "fail_corrective",
    ],
)
@pytest.mark.parametrize("refine_outliers", [True, False])
def test_cancel_arriving_during_final_persistence_cannot_publish_completed(
    monkeypatch,
    tmp_path,
    cancel_on_filename,
    refine_outliers,
):
    root = tmp_path / "class-analysis"
    root.mkdir()
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", root)
    job = api.ClassAnalysisJob(
        job_id="ca_cancel_during_persist",
        status="queued",
        request={
            "source_mode": "active_workspace",
            "encoder_type": "dinov3",
            "refine_outliers": refine_outliers,
            "refinement_schema": REFINEMENT_SCHEMA,
            "run_fingerprint": "d" * 64,
            **(
                {
                    "refinement_decision_contract": (
                        api.CLASS_ANALYSIS_REFINEMENT_DECISION_CONTRACT
                    ),
                    "selector_priority_contract": (
                        api.CLASS_ANALYSIS_SELECTOR_PRIORITY_CONTRACT
                    ),
                    "selector_feature_contract": (
                        api.CLASS_ANALYSIS_SELECTOR_FEATURE_CONTRACT
                    ),
                    "selector_model_digest": str(
                        api._class_analysis_load_default_selector_model_v6().get(
                            "model_digest"
                        )
                        or ""
                    ),
                    "selector_utility_policy_contract": (
                        api.CLASS_ANALYSIS_SELECTOR_UTILITY_POLICY_CONTRACT
                    ),
                    "selector_dataset_overlap_application_contract": (
                        api.CLASS_ANALYSIS_SELECTOR_DATASET_OVERLAP_APPLICATION_CONTRACT
                    ),
                    "selector_dataset_overlap_diagnostic_contract": (
                        api.CLASS_ANALYSIS_SELECTOR_DATASET_OVERLAP_DIAGNOSTIC_CONTRACT
                    ),
                    "selector_global_actionability_model_contract": (
                        api.CLASS_ANALYSIS_SELECTOR_GLOBAL_ACTIONABILITY_MODEL_CONTRACT
                    ),
                    "capture_group_contract": (
                        api.CLASS_ANALYSIS_CAPTURE_GROUP_CONTRACT
                    ),
                }
                if refine_outliers
                else {}
            ),
        },
    )
    records = [
        {
            "point_id": point_id,
            "class_name": class_name,
            "split": "train",
            "image_relpath": f"{point_id}.jpg",
            "bbox_xyxy": [0, 0, 10, 10],
            "width": 10,
            "height": 10,
            "_image_path": str(tmp_path / f"{point_id}.jpg"),
            "_image_sha256": point_id,
        }
        for point_id, class_name in (("p1", "A"), ("p2", "B"))
    ]

    monkeypatch.setattr(
        api,
        "_class_analysis_collect_records",
        lambda *_args, **_kwargs: (
            records,
            [],
            {"_spatial_context_records": list(records)},
        ),
    )

    embedding_maps = {}

    def encoded(
        _records,
        _request,
        *,
        out_dir,
        **_kwargs,
    ):
        work = out_dir / "work"
        work.mkdir(parents=True, exist_ok=True)
        values = np.lib.format.open_memmap(
            work / "raw.npy",
            mode="w+",
            dtype=np.float32,
            shape=(2, 2),
        )
        values[:] = np.eye(2, dtype=np.float32)
        embedding_maps["raw"] = values
        return values

    monkeypatch.setattr(
        api,
        "_class_analysis_stream_encode_records",
        encoded,
    )

    def adjusted(
        _raw,
        _records,
        *,
        output_path,
        **_kwargs,
    ):
        values = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=(2, 2),
        )
        values[:] = np.eye(2, dtype=np.float32)
        embedding_maps["adjusted"] = values
        return values, {"mode": "none"}

    monkeypatch.setattr(
        api,
        "_class_analysis_adjust_embeddings_to_memmap",
        adjusted,
    )
    rough_result = {
        "summary": {
            "analysis_scope": "all_classes",
            "analysis_job_id": job.job_id,
            "analysis_run_instance_id": job.job_id,
            "analysis_input_digest": "d" * 64,
            "run_fingerprint": "d" * 64,
        },
        "points": [
            {
                "point_id": "p1",
                "class_name": "A",
                "projection": [0.0, 0.0],
            },
            {
                "point_id": "p2",
                "class_name": "B",
                "projection": [1.0, 1.0],
            },
        ],
        "wrong_class_candidates": [
            {
                "point_id": "p1",
                "class_name": "A",
                "suggested_neighbor_class": "B",
            }
        ],
    }
    monkeypatch.setattr(
        api,
        "_class_analysis_build_result",
        lambda *_args, **_kwargs: copy.deepcopy(rough_result),
    )

    def refined(**kwargs):
        assert embedding_maps["raw"]._mmap.closed is True
        assert embedding_maps["adjusted"]._mmap.closed is True
        payload = kwargs["result"]
        payload["refinement_candidates"] = [
            {
                **payload["wrong_class_candidates"][0],
                "refined_outlier": {
                    "schema": REFINEMENT_SCHEMA,
                    "decision_contract": (
                        api.CLASS_ANALYSIS_REFINEMENT_DECISION_CONTRACT
                    ),
                    "status": STATUS_CONFIRMED_OUTLIER,
                    "reason_codes": ["test"],
                },
            }
        ]
        payload["vignette_candidates"] = list(
            payload["refinement_candidates"]
        )
        payload["within_class_outlier_candidates"] = []
        selector_summary = api._class_analysis_assign_selector_priority_ranks(
            payload["refinement_candidates"]
        )
        payload["refinement_summary"] = {
            "enabled": True,
            "status": "completed",
            "schema": REFINEMENT_SCHEMA,
            "decision_contract": (
                api.CLASS_ANALYSIS_REFINEMENT_DECISION_CONTRACT
            ),
            "capture_group_contract": (
                api.CLASS_ANALYSIS_CAPTURE_GROUP_CONTRACT
            ),
            "selector_priority_contract": (
                api.CLASS_ANALYSIS_SELECTOR_PRIORITY_CONTRACT
            ),
            "selector_priority": selector_summary,
            "selector_priority_candidate_count": 1,
            "rough_candidate_count": 1,
        }
        payload["summary"]["refinement"] = payload["refinement_summary"]
        return payload

    monkeypatch.setattr(api, "_class_analysis_refine_result", refined)
    original_write_json = api._class_analysis_write_json
    cancelled_during_persist = {"value": False}

    def cancelling_write(
        path,
        root_path,
        payload,
        *,
        artifact_snapshot=None,
    ):
        if (
            cancel_on_filename == "fail_corrective"
            and Path(path).name == api.CLASS_ANALYSIS_JOB_STATE_FILENAME
            and str((payload or {}).get("status") or "") == "completed"
        ):
            raise OSError("simulated corrective state write failure")
        written = original_write_json(
            path,
            root_path,
            payload,
            artifact_snapshot=artifact_snapshot,
        )
        if (
            cancel_on_filename is not None
            and
            Path(path).name == cancel_on_filename
            and not cancelled_during_persist["value"]
            and (
                cancel_on_filename == "result.json"
                or str((payload or {}).get("status") or "") == "completed"
            )
        ):
            cancelled_during_persist["value"] = True
            job.cancel_event.set()
        return written

    monkeypatch.setattr(api, "_class_analysis_write_json", cancelling_write)
    monkeypatch.setattr(
        api,
        "_class_analysis_enforce_cache_budget",
        lambda: None,
    )

    api._run_class_analysis_job(job)

    job_dir = api._class_analysis_job_dir(job.job_id, create=False)
    persisted = json.loads((job_dir / "result.json").read_text("utf-8"))
    state = json.loads(
        (
            job_dir / api.CLASS_ANALYSIS_JOB_STATE_FILENAME
        ).read_text("utf-8")
    )
    if cancel_on_filename is None:
        assert cancelled_during_persist["value"] is False
        assert job.status == "completed"
        assert state["status"] == "completed"
        assert job.runtime["job_memory"] == state["job_memory"]
        assert job.runtime["publication_memory"] == state[
            "publication_memory"
        ]
        assert state["publication_memory"] == {
                "result_json_writer": "sqlite_points_plus_json_encoder_iterencode_atomic",
            "measured_through": "first_terminal_state_write",
            "pre_result_peak_combined_rss_bytes": state[
                "publication_memory"
            ]["pre_result_peak_combined_rss_bytes"],
            "post_result_peak_combined_rss_bytes": state[
                "publication_memory"
            ]["post_result_peak_combined_rss_bytes"],
            "first_terminal_state_peak_combined_rss_bytes": state[
                "job_memory"
            ]["peak_job_combined_rss_bytes"],
            "final_corrective_write": (
                "bounded_atomic_no_recursive_sample"
            ),
        }
        return
    if cancel_on_filename == "fail_corrective":
        assert cancelled_during_persist["value"] is False
        assert job.status == "failed"
        assert state["status"] == "publishing"
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)
        restored = api._get_class_analysis_job(job.job_id)
        try:
            assert restored.status == "failed"
            assert restored.result_path is None
            assert "job_state_status_invalid" in str(restored.error)
        finally:
            with api.CLASS_ANALYSIS_JOBS_LOCK:
                api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)
        return
    assert cancelled_during_persist["value"] is True
    assert job.status == "cancelled"
    assert state["status"] == "cancelled"
    assert persisted["refinement_summary"]["status"] == (
        "cancelled" if refine_outliers else "disabled"
    )
    assert persisted["vignette_candidates"] == []
    assert persisted["wrong_class_candidates"][0]["point_id"] == "p1"


def test_cancel_after_stage_one_persists_raw_fallback_and_no_sidecar(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "class-analysis"
    root.mkdir()
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", root)
    job = api.ClassAnalysisJob(
        job_id="ca_cancelled_refinement",
        status="running",
        request={
            "source_mode": "active_workspace",
            "refine_outliers": True,
            "refinement_schema": REFINEMENT_SCHEMA,
            "refinement_decision_contract": (
                api.CLASS_ANALYSIS_REFINEMENT_DECISION_CONTRACT
            ),
            "selector_priority_contract": (
                api.CLASS_ANALYSIS_SELECTOR_PRIORITY_CONTRACT
            ),
            "selector_feature_contract": (
                api.CLASS_ANALYSIS_SELECTOR_FEATURE_CONTRACT
            ),
            "selector_model_digest": str(
                api._class_analysis_load_default_selector_model_v6().get(
                    "model_digest"
                )
                or ""
            ),
            "selector_utility_policy_contract": (
                api.CLASS_ANALYSIS_SELECTOR_UTILITY_POLICY_CONTRACT
            ),
            "selector_dataset_overlap_application_contract": (
                api.CLASS_ANALYSIS_SELECTOR_DATASET_OVERLAP_APPLICATION_CONTRACT
            ),
            "selector_dataset_overlap_diagnostic_contract": (
                api.CLASS_ANALYSIS_SELECTOR_DATASET_OVERLAP_DIAGNOSTIC_CONTRACT
            ),
            "selector_global_actionability_model_contract": (
                api.CLASS_ANALYSIS_SELECTOR_GLOBAL_ACTIONABILITY_MODEL_CONTRACT
            ),
            "capture_group_contract": (
                api.CLASS_ANALYSIS_CAPTURE_GROUP_CONTRACT
            ),
            "run_fingerprint": "c" * 64,
        },
    )
    job_dir = api._class_analysis_job_dir(job.job_id, create=True)
    (job_dir / api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME).write_bytes(
        b"partial"
    )
    (job_dir / api.CLASS_ANALYSIS_REFINEMENT_MANIFEST_FILENAME).write_text(
        "{}",
        encoding="utf-8",
    )
    preview_dir = (
        job_dir / api.CLASS_ANALYSIS_REFINEMENT_PREVIEW_DIRNAME
    )
    preview_dir.mkdir()
    (preview_dir / "partial.png").write_bytes(b"partial")
    point = {
        "point_id": "rough",
        "class_name": "A",
        "split": "train",
        "image_relpath": "frame.jpg",
        "bbox_xyxy": [0, 0, 10, 10],
        "projection": [0.25, -0.5],
        "suggested_neighbor_class": "B",
        "is_wrong_class_candidate": True,
        "is_rough_outlier_candidate": True,
        "include_in_refined_vignettes": True,
        "refined_outlier": {
            "status": STATUS_CONFIRMED_OUTLIER,
            "reason_codes": ["partial_should_not_survive"],
        },
    }
    record = {
        **point,
        "_image_path": "synthetic-frame.jpg",
        "_image_sha256": "synthetic",
    }
    result = {
        "summary": {
            "analysis_scope": "all_classes",
            "analysis_job_id": job.job_id,
            "analysis_run_instance_id": job.job_id,
            "analysis_input_digest": "c" * 64,
            "run_fingerprint": "c" * 64,
            "refinement": {"status": "running"},
        },
        "points": [point],
        "wrong_class_candidates": [
            {
                "point_id": "rough",
                "class_name": "A",
                "suggested_neighbor_class": "B",
            }
        ],
        "projection_options": {
            "selected": "global_pca",
            "coordinates": {
                "global_pca": np.asarray(
                    [[0.25, -0.5]],
                    dtype=np.float32,
                )
            },
        },
        "refinement_candidates": [{"point_id": "partial"}],
        "vignette_candidates": [{"point_id": "partial"}],
        "refinement_summary": {"status": "running"},
    }

    result_path = api._class_analysis_persist_cancelled_refinement_result(
        result=result,
        records=[record],
        request=job.request,
        job=job,
        out_dir=job_dir,
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["wrong_class_candidates"][0]["point_id"] == "rough"
    assert payload["refinement_summary"]["status"] == "cancelled"
    assert payload["vignette_candidates"] == []
    evidence = payload["refinement_candidates"][0]["refined_outlier"]
    assert evidence["status"] == "unresolved"
    assert evidence["reason_codes"] == ["refinement_cancelled"]
    assert "_image_path" not in record
    assert not (
        job_dir / api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME
    ).exists()
    assert not (
        job_dir / api.CLASS_ANALYSIS_REFINEMENT_MANIFEST_FILENAME
    ).exists()
    assert not preview_dir.exists()
    assert job.result_path == str(result_path)
    assert job.summary["refinement"]["status"] == "cancelled"
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)
    restored = api._get_class_analysis_job(job.job_id)
    try:
        assert restored.status == "cancelled"
        restored_result = api.get_class_analysis_result(job.job_id)
        assert restored_result["refinement_summary"]["status"] == "cancelled"
        assert restored_result["wrong_class_candidates"][0]["point_id"] == "rough"
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)


def test_stage1_only_cancel_persists_cancelled_state_bound_to_result(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "class-analysis"
    root.mkdir()
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", root)
    job = api.ClassAnalysisJob(
        job_id="ca_cancelled_stage1_only",
        status="running",
        progress=0.9,
        request={
            "source_mode": "active_workspace",
            "refine_outliers": False,
            "run_fingerprint": "d" * 64,
        },
    )
    job_dir = api._class_analysis_job_dir(job.job_id, create=True)
    record = {
        "point_id": "p0",
        "class_name": "A",
        "projection": [0.25, -0.5],
        "_image_path": "synthetic-frame.jpg",
        "_image_sha256": "synthetic",
    }
    result = {
        "summary": {
            "analysis_job_id": job.job_id,
            "analysis_run_instance_id": job.job_id,
            "analysis_input_digest": "d" * 64,
            "run_fingerprint": "d" * 64,
        },
        "points": [{"point_id": "p0", "class_name": "A", "projection": [0.25, -0.5]}],
        "wrong_class_candidates": [],
        "projection_options": {
            "selected": "global_pca",
            "coordinates": {"global_pca": np.asarray([[0.25, -0.5]], dtype=np.float32)},
        },
    }

    result_path = api._class_analysis_persist_cancelled_stage1_result(
        result=result,
        records=[record],
        request=job.request,
        job=job,
        out_dir=job_dir,
    )

    state = json.loads(
        (job_dir / api.CLASS_ANALYSIS_JOB_STATE_FILENAME).read_text(encoding="utf-8")
    )
    assert state["status"] == "cancelled"
    assert state["analysis_run_instance_id"] == job.job_id
    assert state["analysis_input_digest"] == "d" * 64
    assert state["result_file"] == "result.json"
    assert state["result_sha256"] == api._class_analysis_file_sha256(result_path)
    assert state["sidecar_file"] == ""
    assert state["sidecar_sha256"] == ""
    assert state["refinement_manifest_file"] == ""
    assert state["refinement_manifest_sha256"] == ""
    assert job.result_path == str(result_path)
    assert "_image_path" not in record
    assert "_image_sha256" not in record
    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    assert persisted["refinement_summary"]["status"] == "disabled"
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)
    restored = api._get_class_analysis_job(job.job_id)
    try:
        assert restored.status == "cancelled"
        assert restored.result_path == str(result_path)
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)


@pytest.mark.parametrize(
    ("status", "source_resolution", "expected_quality"),
    [
        (
            STATUS_UNRESOLVED,
            False,
            "completed_non_actionable",
        ),
        (
            STATUS_EXPLAINED_NOT_OUTLIER,
            True,
            "actionable",
        ),
    ],
)
def test_v33_observability_does_not_gate_complete_v6_queue(
    status,
    source_resolution,
    expected_quality,
):
    gates = {
        gate: False
        for gate in api.CLASS_ANALYSIS_REFINEMENT_V33_CONFIRMATION_REQUIRED_GATES
    }
    gates["source_resolution_sufficient"] = source_resolution
    candidate = {
        "point_id": "rough-1",
        "refined_outlier": {
            "status": status,
            "reason_codes": ["test_fixture"],
            "decision_gates": gates,
        },
    }
    bank = SimpleNamespace(pair_calibration_provenance=lambda: {})

    summary = api._class_analysis_refinement_v3_observability(
        refinement_candidates=[candidate],
        bank=bank,
        anchor_selection={},
    )

    assert summary["quality_status"] == expected_quality
    assert summary["selector_evaluation"] == {
        "status": "experimental",
        "blocks_use": False,
    }
    assert summary["queue_policy"] == {
        "mode": "selector_ranked_complete_stage1",
        "automatic_rough_fallback": False,
        "fallback_reason": "",
        "default_queue": "selector_ranked_stage1_candidates",
        "effective_default_candidate_count": 1,
        "confirmed_count": 0,
        "pair_conflict_count": 0,
        "refined_review_candidate_count": 0,
        "rough_count": 1,
    }


def test_v9_high_resolved_run_uses_complete_v6_queue_without_certification():
    gates = {
        gate: True
        for gate in api.CLASS_ANALYSIS_REFINEMENT_V33_CONFIRMATION_REQUIRED_GATES
    }
    candidate = {
        "point_id": "confirmed-but-not-release-ready",
        "refined_outlier": {
            "status": STATUS_CONFIRMED_OUTLIER,
            "reason_codes": ["synthetic_confirmed_outlier"],
            "decision_gates": gates,
        },
    }
    bank = SimpleNamespace(pair_calibration_provenance=lambda: {})

    summary = api._class_analysis_refinement_v3_observability(
        refinement_candidates=[candidate],
        bank=bank,
        anchor_selection={},
    )

    assert summary["quality_status"] == "actionable"
    assert summary["quality_gate"]["metrics"]["resolved_rate"] == 1.0
    assert summary["queue_policy"] == {
        "mode": "selector_ranked_complete_stage1",
        "automatic_rough_fallback": False,
        "fallback_reason": "",
        "default_queue": "selector_ranked_stage1_candidates",
        "effective_default_candidate_count": 1,
        "confirmed_count": 1,
        "pair_conflict_count": 0,
        "refined_review_candidate_count": 1,
        "rough_count": 1,
    }


def test_refinement_preview_is_generated_lazily(monkeypatch, tmp_path):
    root = tmp_path / "class-analysis"
    root.mkdir()
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", root)
    job_id = "ca_preview"
    point_id = "point-preview"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    workspace = root / "workspace"
    images = workspace / "images"
    images.mkdir(parents=True)
    source_path = images / "frame.jpg"
    Image.new("RGB", (80, 60), (80, 100, 120)).save(source_path)
    source_sha256 = api._class_analysis_file_sha256(source_path)
    point = {
        "point_id": point_id,
        "split": "train",
        "image_relpath": "frame.jpg",
        "class_name": "A",
        "bbox_xyxy": [10, 10, 50, 50],
        "refined_outlier": {
            "status": "mixed_or_composite",
            "current_class": "A",
            "alternative_class": "B",
            "sidecar_row": 0,
            "source_image_sha256": source_sha256,
        },
    }
    api._class_analysis_write_indexed_metadata_jsonl(
        job_dir / api.CLASS_ANALYSIS_METADATA_FILENAME,
        job_dir,
        [point],
    )
    heat = np.ones((1, 2, 4, 4), dtype=np.float16)
    arrays = {
        "point_ids": np.asarray([point_id]),
        "current_heatmaps": heat,
        "alternative_heatmaps": heat * 0.5,
        "valid_masks": np.ones_like(heat, dtype=np.uint8),
        "target_masks": np.full_like(heat, 255, dtype=np.uint8),
        "overlap_masks": np.zeros_like(heat, dtype=np.uint8),
    }
    sidecar_path = api._class_analysis_write_npz(
        job_dir / api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME,
        job_dir,
        **arrays,
    )
    manifest = {
        "schema": REFINEMENT_SCHEMA,
        "decision_contract": api.CLASS_ANALYSIS_REFINEMENT_DECISION_CONTRACT,
        "sidecar_file": api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME,
        "exact_view_pair_calibration": {
            "contract": "class-analysis-exact-view-calibration-pass-v1",
            "eligible_example_count": 1,
            "accepted_example_count": 1,
            "skipped_example_count": 0,
            "per_class_accepted_source_counts": {"A": 1},
        },
        "sidecar": {
            "sha256": api._class_analysis_file_sha256(sidecar_path),
            "bytes": sidecar_path.stat().st_size,
            "arrays": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                for name, value in arrays.items()
            },
        },
        "point_rows": {point_id: 0},
    }
    api._class_analysis_write_json(
        job_dir / api.CLASS_ANALYSIS_REFINEMENT_MANIFEST_FILENAME,
        job_dir,
        {
            **manifest,
            "sidecar": {
                **manifest["sidecar"],
                "sha256": "0" * 64,
            },
        },
    )
    job = api.ClassAnalysisJob(
        job_id=job_id,
        status="completed",
        request={
            "source_mode": "active_workspace",
            "workspace_dir": str(workspace),
            "yolo_layout": "flat",
        },
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS[job_id] = job

    preview_path = (
        job_dir
        / api.CLASS_ANALYSIS_REFINEMENT_PREVIEW_DIRNAME
        / (
            f"{point_id}-v33-exclusive-{manifest['sidecar']['sha256'][:16]}-"
            f"{source_sha256[:16]}.png"
        )
    )
    try:
        assert not preview_path.exists()
        with pytest.raises(api.HTTPException) as checksum_error:
            api.get_class_analysis_refinement_preview(job_id, point_id)
        assert checksum_error.value.detail == "refinement_preview_not_found"
        api._class_analysis_write_json(
            job_dir / api.CLASS_ANALYSIS_REFINEMENT_MANIFEST_FILENAME,
            job_dir,
            manifest,
        )
        original_source_snapshot = (
            api._class_analysis_file_snapshot_cached
        )
        swapped_source = False

        def snapshot_then_replace_source(path):
            nonlocal swapped_source
            snapshot = original_source_snapshot(path)
            if Path(path) == source_path and not swapped_source:
                swapped_source = True
                replacement = source_path.with_name("frame.replacement.jpg")
                Image.new("RGB", (80, 60), (10, 20, 30)).save(
                    replacement
                )
                os.replace(replacement, source_path)
            return snapshot

        monkeypatch.setattr(
            api,
            "_class_analysis_file_snapshot_cached",
            snapshot_then_replace_source,
        )
        with pytest.raises(api.HTTPException) as source_race_error:
            api.get_class_analysis_refinement_preview(job_id, point_id)
        assert swapped_source is True
        assert source_race_error.value.detail == (
            "refinement_preview_not_found"
        )
        monkeypatch.setattr(
            api,
            "_class_analysis_file_snapshot_cached",
            original_source_snapshot,
        )
        Image.new("RGB", (80, 60), (80, 100, 120)).save(source_path)
        assert api._class_analysis_file_sha256(source_path) == source_sha256
        response = api.get_class_analysis_refinement_preview(
            job_id,
            point_id,
        )
        assert response.media_type == "image/png"
        assert Path(response.path) == preview_path
        assert preview_path.is_file()
        before_hash = hashlib.sha256(preview_path.read_bytes()).hexdigest()
        second = api.get_class_analysis_refinement_preview(job_id, point_id)
        assert second.media_type == "image/png"
        assert hashlib.sha256(preview_path.read_bytes()).hexdigest() == before_hash
        Image.new("RGB", (80, 60), (10, 20, 30)).save(source_path)
        with pytest.raises(api.HTTPException) as source_error:
            api.get_class_analysis_refinement_preview(job_id, point_id)
        assert source_error.value.detail == "refinement_preview_not_found"
        Image.new("RGB", (80, 60), (80, 100, 120)).save(source_path)
        assert api._class_analysis_file_sha256(source_path) == source_sha256
        sidecar_path.write_bytes(sidecar_path.read_bytes() + b"tampered")
        with pytest.raises(api.HTTPException) as sidecar_error:
            api.get_class_analysis_refinement_preview(job_id, point_id)
        assert sidecar_error.value.detail == "refinement_preview_not_found"
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.pop(job_id, None)


def test_refinement_preview_preflights_huge_declared_array_before_np_load(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "class-analysis"
    root.mkdir()
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", root)
    job_id = "ca_preview_huge_header"
    point_id = "point-preview"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    workspace = root / "workspace"
    images = workspace / "images"
    images.mkdir(parents=True)
    source_path = images / "frame.jpg"
    Image.new("RGB", (80, 60), (80, 100, 120)).save(source_path)
    source_sha256 = api._class_analysis_file_sha256(source_path)
    point = {
        "point_id": point_id,
        "split": "train",
        "image_relpath": "frame.jpg",
        "class_name": "A",
        "bbox_xyxy": [10, 10, 50, 50],
        "refined_outlier": {
            "status": "mixed_or_composite",
            "current_class": "A",
            "alternative_class": "B",
            "sidecar_row": 0,
            "source_image_sha256": source_sha256,
        },
    }
    api._class_analysis_write_indexed_metadata_jsonl(
        job_dir / api.CLASS_ANALYSIS_METADATA_FILENAME,
        job_dir,
        [point],
    )
    point_ids = np.asarray([point_id])
    spatial = np.ones((1, 2, 4, 4), dtype=np.float16)
    members = {
        "point_ids": _npy_member_bytes(point_ids),
        "current_heatmaps": _npy_header_only_bytes(
            shape=(1, 2, 100_000_000, 4),
            dtype=np.float16,
        ),
        "alternative_heatmaps": _npy_member_bytes(spatial),
        "valid_masks": _npy_member_bytes(
            np.ones_like(spatial, dtype=np.uint8)
        ),
        "target_masks": _npy_member_bytes(
            np.ones_like(spatial, dtype=np.uint8)
        ),
        "overlap_masks": _npy_member_bytes(
            np.zeros_like(spatial, dtype=np.uint8)
        ),
    }
    sidecar_path = (
        job_dir / api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME
    )
    _write_npz_members(sidecar_path, members)
    declared_arrays = {
        "point_ids": {
            "shape": list(point_ids.shape),
            "dtype": str(point_ids.dtype),
        },
        "current_heatmaps": {
            "shape": [1, 2, 100_000_000, 4],
            "dtype": "float16",
        },
        "alternative_heatmaps": {
            "shape": list(spatial.shape),
            "dtype": "float16",
        },
        "valid_masks": {
            "shape": list(spatial.shape),
            "dtype": "uint8",
        },
        "target_masks": {
            "shape": list(spatial.shape),
            "dtype": "uint8",
        },
        "overlap_masks": {
            "shape": list(spatial.shape),
            "dtype": "uint8",
        },
    }
    api._class_analysis_write_json(
        job_dir / api.CLASS_ANALYSIS_REFINEMENT_MANIFEST_FILENAME,
        job_dir,
        {
            "schema": REFINEMENT_SCHEMA,
            "sidecar_file": api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME,
            "sidecar": {
                "sha256": api._class_analysis_file_sha256(sidecar_path),
                "bytes": sidecar_path.stat().st_size,
                "arrays": declared_arrays,
            },
            "point_rows": {point_id: 0},
        },
    )
    job = api.ClassAnalysisJob(
        job_id=job_id,
        status="completed",
        request={
            "source_mode": "active_workspace",
            "workspace_dir": str(workspace),
            "yolo_layout": "flat",
        },
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS[job_id] = job
    load_calls = []

    def reject_np_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        raise AssertionError("np.load must not run before bounded NPY preflight")

    monkeypatch.setattr(api.np, "load", reject_np_load)
    try:
        with pytest.raises(api.HTTPException) as error:
            api.get_class_analysis_refinement_preview(job_id, point_id)
        assert error.value.status_code == 404
        assert error.value.detail == "refinement_preview_not_found"
        assert load_calls == []
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.pop(job_id, None)
