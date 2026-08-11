import copy
import math

from PIL import Image

import localinferenceapi as api


def _refined_point() -> dict:
    return {
        "point_id": "point_refined",
        "class_name": "ElevatedFixture",
        "suggested_neighbor_class": "Building",
        "refined_outlier": {
            "schema": api.CLASS_ANALYSIS_REFINEMENT_SCHEMA,
            "decision_contract": (
                api.CLASS_ANALYSIS_REFINEMENT_DECISION_CONTRACT
            ),
            "status": "explained_not_outlier",
            "reason_codes": [
                "alternative_evidence_localized_to_overlap",
            ],
            "current_class": "ElevatedFixture",
            "alternative_class": "Building",
            "current_support_score": 0.41,
            "alternative_support_score": 0.37,
            "intrinsic_current_support": 0.39,
            "intrinsic_alternative_support": 0.52,
            "directed_pair_margin": 0.13,
            "directed_pair_raw_margin": 0.13,
            "directed_pair_probe_score": 0.264,
            "directed_pair_probe_features": [0.12, 0.42],
            "directed_pair_probe_feature_names": [
                "current_patch_exclusive_support",
                "alternative_patch_exclusive_support",
            ],
            "directed_pair_current_exclusive_support": 0.12,
            "directed_pair_alternative_exclusive_support": 0.42,
            "directed_pair_probe_threshold": 0.18,
            "directed_pair_probe_weights": [-0.6, 0.8],
            "directed_pair_probe_contract": (
                api.CLASS_ANALYSIS_REFINEMENT_V33_PAIR_PROBE_CONTRACT
            ),
            "directed_pair_probe_view_contract": (
                api.CLASS_ANALYSIS_REFINEMENT_V33_VIEW_FEATURE_CONTRACT
            ),
            "directed_pair_probe_lower_bound_contract": (
                api.CLASS_ANALYSIS_REFINEMENT_V33_LOWER_BOUND_CONTRACT
            ),
            "directed_pair_probe_fold_count": 1,
            "directed_pair_probe_fit_status": "ok",
            "directed_pair_probe_fold_digest": "cd" * 32,
            "directed_pair_probe_fit_eval_split_digest": "cd" * 32,
            "directed_pair_threshold": 0.18,
            "current_negative_threshold": 0.07,
            "current_support_threshold": 0.15,
            "current_strong_threshold": 0.25,
            "alternative_negative_threshold": 0.09,
            "alternative_support_threshold": 0.15,
            "alternative_strong_threshold": 0.25,
            "support_threshold_source": "fit_only_directed_pair",
            "directed_pair_tier": "usable",
            "directed_pair_reliable": True,
            "directed_pair_bank_reliable": True,
            "diagnostic_pair_reliability_contract": (
                api.CLASS_ANALYSIS_DIAGNOSTIC_PAIR_RELIABILITY_CONTRACT
            ),
            "diagnostic_pair_reliable": True,
            "diagnostic_pair_bank_reliable": True,
            "positive_confirmation_pair_reliable": True,
            "human_review_qualification_contract": (
                api.CLASS_ANALYSIS_HUMAN_REVIEW_QUALIFICATION_CONTRACT
            ),
            "human_review_rank_contract": (
                api.CLASS_ANALYSIS_HUMAN_REVIEW_RANK_CONTRACT
            ),
            "qualified_for_human_review": False,
            "human_review_rank": None,
            "directed_pair_candidate_source_excluded": False,
            "directed_pair_candidate_source_fingerprint": "12" * 8,
            "directed_pair_candidate_source_membership_roles": [],
            "directed_pair_heldout_auroc": 0.82,
            "directed_pair_eval_auroc_lower_bound": 0.68,
            "positive_confirmation_pair_probe_auroc_floor": 0.80,
            "positive_confirmation_pair_probe_auroc_lower_bound_floor": 0.60,
            "directed_pair_probe_fit_current_source_count": 12,
            "directed_pair_probe_fit_alternative_source_count": 13,
            "directed_pair_probe_eval_current_source_count": 9,
            "directed_pair_probe_eval_alternative_source_count": 10,
            "directed_pair_probe_fit_balanced_accuracy": 0.76,
            "directed_pair_probe_eval_sensitivity": 0.71,
            "directed_pair_probe_eval_specificity": 0.73,
            "directed_pair_current_absence_eval_fraction": 0.68,
            "directed_pair_alternative_strong_eval_fraction": 0.70,
            "directed_pair_current_source_count": 21,
            "directed_pair_alternative_source_count": 29,
            "directed_pair_current_patch_count": 144,
            "directed_pair_alternative_patch_count": 201,
            "directed_pair_alternative_passing_source_fraction": 0.79,
            "decision_gates": {
                "directed_pair_reliable": True,
                "directed_pair_candidate_source_independent": True,
                "directed_pair_exact_calibration_contracts": True,
                "intrinsic_references_reliable": True,
                "diagnostic_pair_reliable": True,
                "positive_confirmation_pair_reliable": True,
                "qualified_for_human_review": False,
                "positive_confirmation_pair_probe_auroc_sufficient": True,
                "positive_confirmation_pair_probe_lower_bound_sufficient": True,
                "source_resolution_sufficient": True,
                "directed_pair_dominates": False,
                "current_absent": False,
                "alternative_strong": True,
                "alternative_exclusive_component_corresponds": True,
                "view_consistent": True,
                "alternative_evidence_external_to_overlap": False,
            },
            "alternative_evidence_inside_overlap_fraction": 0.94,
            "alternative_evidence_outside_overlap_fraction": 0.06,
            "current_evidence_outside_overlap_fraction": 0.71,
            "overlap_relation": "other_contains_target",
            "overlap_object_count": 1,
            "annotated_overlap_alternative_bbox_xyxy": [
                10.0,
                12.0,
                90.0,
                92.0,
            ],
            "annotated_overlap_alternative_point_id": "overlap-point",
            "reference_reliable": True,
            "reference_distinct_source_count": 63,
            "current_reference_tier": "high",
            "alternative_reference_tier": "usable",
            "current_reference_heldout_auroc": 0.86,
            "alternative_reference_heldout_auroc": 0.78,
            "view_agreement": 0.92,
            "sidecar_row": 7,
            "source_image_sha256": "ab" * 32,
            "_private_heatmaps": [[0.2, 0.8]],
        },
    }


def test_compact_refinement_rail_is_bounded_and_advisory():
    rail = api._class_analysis_compact_refinement_rail(
        _refined_point()
    )

    assert rail["status"] == "explained_not_outlier"
    assert rail["reason_codes"] == [
        "alternative_evidence_localized_to_overlap"
    ]
    assert rail["current_support_score"] == 0.41
    assert rail["alternative_support_score"] == 0.37
    assert rail["rail_version"] == "patch_refinement_advisory_v7"
    assert rail["directed_pair_probe_score"] == 0.264
    assert rail["directed_pair_probe_features"] == [0.12, 0.42]
    assert rail["directed_pair_probe_feature_names"] == [
        "current_patch_exclusive_support",
        "alternative_patch_exclusive_support",
    ]
    assert rail["directed_pair_current_exclusive_support"] == 0.12
    assert rail["directed_pair_alternative_exclusive_support"] == 0.42
    assert rail["positive_confirmation_pair_probe_auroc_floor"] == 0.8
    assert rail["directed_pair_eval_auroc_lower_bound"] == 0.68
    assert (
        rail["positive_confirmation_pair_probe_auroc_lower_bound_floor"]
        == 0.6
    )
    assert rail["directed_pair_raw_margin"] == 0.13
    assert rail["current_negative_threshold"] == 0.07
    assert rail["current_support_threshold"] == 0.15
    assert rail["current_strong_threshold"] == 0.25
    assert rail["alternative_negative_threshold"] == 0.09
    assert rail["alternative_support_threshold"] == 0.15
    assert rail["alternative_strong_threshold"] == 0.25
    assert rail["support_threshold_source"] == "fit_only_directed_pair"
    assert rail["directed_pair_bank_reliable"] is True
    assert rail["directed_pair_candidate_source_excluded"] is False
    assert rail["directed_pair_candidate_source_fingerprint"] == "12" * 8
    assert rail["directed_pair_candidate_source_membership_roles"] == []
    assert rail["gate_diagnostics"][
        "directed_pair_candidate_source_independent"
    ] is True
    assert "directed_pair_margin" not in rail
    assert rail["alternative_evidence_inside_overlap_fraction"] == 0.94
    assert rail["overlap_relation"] == "other_contains_target"
    assert rail["annotated_overlap_alternative_bbox_xyxy"] == [
        10.0,
        12.0,
        90.0,
        92.0,
    ]
    assert rail["annotated_overlap_alternative_point_id"] == (
        "overlap-point"
    )
    assert rail["reference_reliable"] is True
    assert rail["reference_distinct_source_count"] == 63
    assert rail["sidecar_row"] == 7
    assert rail["advisory_only"] is True
    assert "cannot override" in rail["policy"]
    assert "_private_heatmaps" not in rail


def test_compact_refinement_rail_preserves_close_fitted_threshold_order():
    point = _refined_point()
    evidence = point["refined_outlier"]
    evidence["alternative_negative_threshold"] = 0.16439378261566162
    evidence["alternative_support_threshold"] = 0.16439379751682281
    evidence["alternative_strong_threshold"] = 0.16439379751682281

    rail = api._class_analysis_compact_refinement_rail(point)

    assert rail is not None
    assert (
        rail["alternative_negative_threshold"]
        < rail["alternative_support_threshold"]
        <= rail["alternative_strong_threshold"]
    )


def test_compact_refinement_rail_rejects_foreign_schema_and_bad_fractions():
    foreign = _refined_point()
    foreign["refined_outlier"]["schema"] = "legacy-refinement-v0"
    assert api._class_analysis_compact_refinement_rail(foreign) is None

    for stale_contract in (
        None,
        "class-analysis-patch-decision-v1",
        "class-analysis-patch-decision-v8",
    ):
        stale = _refined_point()
        if stale_contract is None:
            stale["refined_outlier"].pop("decision_contract")
        else:
            stale["refined_outlier"]["decision_contract"] = stale_contract
        assert api._class_analysis_compact_refinement_rail(stale) is None

    malformed = _refined_point()
    malformed["refined_outlier"][
        "alternative_evidence_inside_overlap_fraction"
    ] = 1.2
    malformed["refined_outlier"]["view_agreement"] = -0.1
    rail = api._class_analysis_compact_refinement_rail(malformed)

    assert rail["alternative_evidence_inside_overlap_fraction"] is None
    assert rail["view_agreement"] is None
    assert rail["current_support_score"] == 0.41


def test_compact_refinement_rail_preserves_unfitted_unresolved_evidence():
    point = {
        "point_id": "unresolved-point",
        "class_name": "ElevatedFixture",
        "suggested_neighbor_class": "Building",
        "refined_outlier": api._class_analysis_unresolved_refinement_evidence(
            current_class="ElevatedFixture",
            alternative_class="Building",
            reason="candidate_scoring_failed",
        ),
    }

    rail = api._class_analysis_compact_refinement_rail(point)

    assert rail is not None
    assert rail["status"] == "unresolved"
    assert rail["directed_pair_probe_fit_status"] == "not_applicable"
    assert rail["directed_pair_probe_features"] is None
    assert rail["directed_pair_probe_fit_eval_split_digest"] == ""
    assert rail["gate_diagnostics"]["directed_pair_reliable"] is False


def test_compact_refinement_rail_requires_exact_overlap_geometry():
    missing = _refined_point()
    missing["refined_outlier"].pop(
        "annotated_overlap_alternative_bbox_xyxy"
    )
    assert api._class_analysis_compact_refinement_rail(missing) is None

    malformed = _refined_point()
    malformed["refined_outlier"][
        "annotated_overlap_alternative_bbox_xyxy"
    ] = [10.0, 12.0, 10.0, 92.0]
    assert api._class_analysis_compact_refinement_rail(malformed) is None

    contradictory = _refined_point()
    contradictory["refined_outlier"]["overlap_object_count"] = 0
    assert api._class_analysis_compact_refinement_rail(contradictory) is None


def test_compact_refinement_rail_selector_bundle_is_optional_but_atomic():
    no_selector_bundle = _refined_point()
    rail = api._class_analysis_compact_refinement_rail(no_selector_bundle)
    assert rail is not None
    assert rail["selector_priority"] is None

    partial_selector_bundle = _refined_point()
    partial_selector_bundle["refined_outlier"][
        "selector_priority_contract"
    ] = api.CLASS_ANALYSIS_SELECTOR_PRIORITY_CONTRACT
    assert (
        api._class_analysis_compact_refinement_rail(partial_selector_bundle)
        is None
    )


def test_compact_refinement_rail_fails_closed_without_v33_probe_provenance():
    for field in (
        "directed_pair_probe_score",
        "directed_pair_probe_features",
        "directed_pair_probe_feature_names",
        "directed_pair_current_exclusive_support",
        "directed_pair_alternative_exclusive_support",
        "directed_pair_probe_threshold",
        "directed_pair_probe_weights",
        "directed_pair_probe_contract",
        "directed_pair_probe_view_contract",
        "directed_pair_probe_lower_bound_contract",
        "directed_pair_probe_fold_count",
        "directed_pair_probe_fit_status",
        "directed_pair_probe_fold_digest",
        "directed_pair_raw_margin",
        "directed_pair_heldout_auroc",
        "directed_pair_eval_auroc_lower_bound",
        "positive_confirmation_pair_probe_auroc_floor",
        "positive_confirmation_pair_probe_auroc_lower_bound_floor",
        "directed_pair_probe_fit_current_source_count",
        "directed_pair_probe_fit_alternative_source_count",
        "directed_pair_probe_eval_current_source_count",
        "directed_pair_probe_eval_alternative_source_count",
        "directed_pair_probe_fit_eval_split_digest",
        "directed_pair_probe_fit_balanced_accuracy",
        "directed_pair_probe_eval_sensitivity",
        "directed_pair_probe_eval_specificity",
        "directed_pair_current_absence_eval_fraction",
        "directed_pair_alternative_strong_eval_fraction",
        "current_negative_threshold",
        "current_support_threshold",
        "current_strong_threshold",
        "alternative_negative_threshold",
        "alternative_support_threshold",
        "alternative_strong_threshold",
        "support_threshold_source",
        "directed_pair_bank_reliable",
        "directed_pair_candidate_source_excluded",
        "directed_pair_candidate_source_fingerprint",
        "directed_pair_candidate_source_membership_roles",
        "diagnostic_pair_reliability_contract",
        "diagnostic_pair_reliable",
        "diagnostic_pair_bank_reliable",
        "positive_confirmation_pair_reliable",
        "human_review_qualification_contract",
        "human_review_rank_contract",
        "qualified_for_human_review",
        "human_review_rank",
    ):
        point = _refined_point()
        point["refined_outlier"].pop(field)
        assert api._class_analysis_compact_refinement_rail(point) is None

    wrong_probe = _refined_point()
    wrong_probe["refined_outlier"]["directed_pair_probe_contract"] = (
        "unknown-probe-v0"
    )
    assert api._class_analysis_compact_refinement_rail(wrong_probe) is None


def test_compact_refinement_rail_rejects_unsafe_probe_serialization():
    mutations = (
        ("directed_pair_probe_weights", [0.6, 0.8]),
        ("directed_pair_probe_weights", [-0.5, -0.8660254]),
        ("directed_pair_probe_weights", [-0.4, 0.4]),
        ("directed_pair_probe_fold_count", 2),
        ("directed_pair_probe_fold_digest", "not-a-digest"),
        ("directed_pair_probe_score", float("nan")),
        ("directed_pair_probe_features", [0.12, float("nan")]),
        ("directed_pair_probe_feature_names", ["wrong", "features"]),
        ("directed_pair_current_exclusive_support", 0.13),
        ("directed_pair_probe_features", [-4.01, 0.42]),
        ("positive_confirmation_pair_probe_auroc_floor", 0.79),
        (
            "positive_confirmation_pair_probe_auroc_lower_bound_floor",
            0.59,
        ),
        ("directed_pair_candidate_source_fingerprint", "not-a-fingerprint"),
        (
            "directed_pair_candidate_source_membership_roles",
            ["unknown_role"],
        ),
    )
    for field, value in mutations:
        point = _refined_point()
        point["refined_outlier"][field] = value
        assert api._class_analysis_compact_refinement_rail(point) is None


def test_compact_refinement_rail_accepts_signed_unbounded_exclusive_margins():
    point = _refined_point()
    evidence = point["refined_outlier"]
    evidence["directed_pair_probe_features"] = [-1.2, 1.3]
    evidence["directed_pair_current_exclusive_support"] = -1.2
    evidence["directed_pair_alternative_exclusive_support"] = 1.3
    evidence["directed_pair_probe_score"] = 1.76

    rail = api._class_analysis_compact_refinement_rail(point)

    assert rail is not None
    assert rail["directed_pair_probe_features"] == [-1.2, 1.3]
    assert rail["directed_pair_current_exclusive_support"] == -1.2
    assert rail["directed_pair_alternative_exclusive_support"] == 1.3


def test_compact_rail_accepts_intrinsic_unreliability_after_source_clearance():
    point = _refined_point()
    evidence = point["refined_outlier"]
    evidence["directed_pair_reliable"] = False
    evidence["diagnostic_pair_reliable"] = False
    evidence["positive_confirmation_pair_reliable"] = False
    evidence["support_threshold_source"] = "intrinsic_fallback"
    evidence["decision_gates"].update(
        {
            "directed_pair_reliable": False,
            "intrinsic_references_reliable": False,
            "diagnostic_pair_reliable": False,
            "positive_confirmation_pair_reliable": False,
        }
    )

    rail = api._class_analysis_compact_refinement_rail(point)

    assert rail is not None
    assert rail["directed_pair_bank_reliable"] is True
    assert rail["directed_pair_candidate_source_excluded"] is False
    assert rail["gate_diagnostics"][
        "directed_pair_candidate_source_independent"
    ] is True
    assert rail["gate_diagnostics"]["intrinsic_references_reliable"] is False
    assert rail["directed_pair_reliable"] is False
    assert rail["support_threshold_source"] == "intrinsic_fallback"


def test_compact_rail_preserves_candidate_calibration_source_exclusion():
    point = _refined_point()
    evidence = point["refined_outlier"]
    evidence["status"] = "unresolved"
    evidence["reason_codes"].append(
        "directed_pair_candidate_source_in_calibration"
    )
    evidence["directed_pair_reliable"] = False
    evidence["diagnostic_pair_reliable"] = False
    evidence["positive_confirmation_pair_reliable"] = False
    evidence["directed_pair_candidate_source_excluded"] = True
    evidence["directed_pair_candidate_source_membership_roles"] = [
        "current_class"
    ]
    evidence["support_threshold_source"] = "intrinsic_fallback"
    evidence["decision_gates"].update(
        {
            "directed_pair_reliable": False,
            "directed_pair_candidate_source_independent": False,
            "diagnostic_pair_reliable": False,
            "positive_confirmation_pair_reliable": False,
        }
    )

    rail = api._class_analysis_compact_refinement_rail(point)

    assert rail is not None
    assert rail["directed_pair_bank_reliable"] is True
    assert rail["directed_pair_candidate_source_excluded"] is True
    assert rail["directed_pair_candidate_source_membership_roles"] == [
        "current_class"
    ]
    assert rail["gate_diagnostics"][
        "directed_pair_candidate_source_independent"
    ] is False
    assert rail["directed_pair_reliable"] is False
    assert rail["support_threshold_source"] == "intrinsic_fallback"
    assert "directed_pair_candidate_source_in_calibration" in rail[
        "reason_codes"
    ]


def test_confirmed_rail_requires_positive_confirmation_reliability():
    valid = _refined_point()
    evidence = valid["refined_outlier"]
    evidence.update(
        {
            "status": "confirmed_outlier",
            "current_support_score": 0.035,
            "alternative_support_score": 0.43,
            "intrinsic_current_support": 0.035,
            "intrinsic_alternative_support": 0.43,
            "current_view_support_scores": [0.03, 0.04],
            "alternative_view_support_scores": [0.42, 0.44],
            "directed_pair_margin": 0.395,
            "directed_pair_raw_margin": 0.395,
            "directed_pair_probe_features": [-0.1, 0.5],
            "directed_pair_current_exclusive_support": -0.1,
            "directed_pair_alternative_exclusive_support": 0.5,
            "directed_pair_probe_score": 0.46,
            "visible_target_bbox_width": 40.0,
            "visible_target_bbox_height": 50.0,
            "visible_target_bbox_area": 2000.0,
            "minimum_confirmation_bbox_short_side": 16.0,
            "minimum_confirmation_bbox_area": 324.0,
            "intrinsic_references_reliable": True,
        }
    )
    evidence["decision_gates"].update(
        {
            gate: True
            for gate in api.CLASS_ANALYSIS_REFINEMENT_V33_CONFIRMATION_REQUIRED_GATES
        }
    )
    evidence["qualified_for_human_review"] = True
    evidence["human_review_rank"] = 1
    evidence["decision_gates"]["qualified_for_human_review"] = True
    assert api._class_analysis_compact_refinement_rail(valid) is not None

    point = _refined_point()
    point["refined_outlier"]["status"] = "confirmed_outlier"
    point["refined_outlier"]["decision_gates"][
        "positive_confirmation_pair_reliable"
    ] = False

    assert api._class_analysis_compact_refinement_rail(point) is None

    point = _refined_point()
    point["refined_outlier"]["status"] = "confirmed_outlier"
    point["refined_outlier"]["directed_pair_reliable"] = False
    assert api._class_analysis_compact_refinement_rail(point) is None

    point = _refined_point()
    point["refined_outlier"]["status"] = "confirmed_outlier"
    point["refined_outlier"]["directed_pair_heldout_auroc"] = 0.79
    point["refined_outlier"]["decision_gates"][
        "positive_confirmation_pair_probe_auroc_sufficient"
    ] = False
    assert api._class_analysis_compact_refinement_rail(point) is None


def test_attachment_boundary_downgrades_unproven_confirmation():
    point = _refined_point()
    evidence = point["refined_outlier"]
    evidence["status"] = "confirmed_outlier"
    result_point = {"point_id": point["point_id"]}

    attached = api._class_analysis_attach_refinement(
        candidate=point,
        evidence=evidence,
        result_points_by_id={point["point_id"]: result_point},
        records_by_id={},
    )

    assert attached["refined_outlier"]["status"] == "unresolved"
    assert attached["include_in_refined_vignettes"] is False
    assert attached["refined_outlier"]["status_before_attachment_guard"] == (
        "confirmed_outlier"
    )
    assert (
        "confirmed_outlier_invariants_failed_at_attachment"
        in attached["refined_outlier"]["reason_codes"]
    )
    assert result_point["include_in_refined_vignettes"] is False


def test_compact_rail_rejects_semantic_terminal_without_source_resolution():
    for status in (
        "confirmed_outlier",
        "explained_not_outlier",
        "mixed_or_composite",
    ):
        point = _refined_point()
        point["refined_outlier"]["status"] = status
        point["refined_outlier"]["decision_gates"][
            "source_resolution_sufficient"
        ] = False
        assert api._class_analysis_compact_refinement_rail(point) is None

    pair_conflict = _refined_point()
    pair_conflict["refined_outlier"]["status"] = "pair_conflict"
    pair_conflict["refined_outlier"][
        "annotated_overlap_alternative_point_id"
    ] = "building-box"
    pair_conflict["refined_outlier"]["overlap_relation"] = "duplicate_like"
    pair_conflict["refined_outlier"]["pair_conflict"] = {
        "enabled": True,
        "point_id": "point_refined",
        "current_class": "ElevatedFixture",
        "other_point_id": "building-box",
        "other_class_name": "Building",
        "target_bbox_xyxy": [11.0, 13.0, 89.0, 91.0],
        "other_bbox_xyxy": [10.0, 12.0, 90.0, 92.0],
        "relation": "duplicate_like",
    }
    pair_conflict["refined_outlier"]["decision_gates"][
        "source_resolution_sufficient"
    ] = False
    assert api._class_analysis_compact_refinement_rail(pair_conflict) is not None


def test_attachment_boundary_downgrades_all_unresolved_semantic_terminals():
    for status in ("explained_not_outlier", "mixed_or_composite"):
        point = _refined_point()
        evidence = point["refined_outlier"]
        evidence["status"] = status
        evidence["decision_gates"]["source_resolution_sufficient"] = False
        result_point = {"point_id": point["point_id"]}

        attached = api._class_analysis_attach_refinement(
            candidate=point,
            evidence=evidence,
            result_points_by_id={point["point_id"]: result_point},
            records_by_id={},
        )

        assert attached["refined_outlier"]["status"] == "unresolved"
        assert attached["refined_outlier"]["status_before_attachment_guard"] == status
        assert (
            "source_resolution_insufficient_for_terminal_status"
            in attached["refined_outlier"]["reason_codes"]
        )
        assert attached["include_in_refined_vignettes"] is False


def test_compact_pair_conflict_rail_preserves_bounded_object_provenance():
    point = _refined_point()
    point["refined_outlier"]["status"] = "pair_conflict"
    point["refined_outlier"][
        "annotated_overlap_alternative_point_id"
    ] = "building-box"
    point["refined_outlier"]["overlap_relation"] = "duplicate_like"
    point["refined_outlier"]["pair_conflict"] = {
        "enabled": True,
        "kind": "near_identical_cross_class_bbox",
        "review_mode": "dual_bbox_class_resolution",
        "point_id": "point_refined",
        "current_class": "ElevatedFixture",
        "other_point_id": "building-box",
        "target_bbox_xyxy": [11.0, 13.0, 89.0, 91.0],
        "other_bbox_xyxy": [10.0, 12.0, 90.0, 92.0],
        "other_class_name": "Building",
        "classes": ["ElevatedFixture", "Building"],
        "image_relpath": "train/frame.jpg",
        "split": "train",
        "iou": 0.97,
        "corner_similarity": 0.98,
        "target_area_covered": 0.99,
        "other_area_covered": 0.98,
        "relation": "duplicate_like",
        "score": 0.97,
        "question": "Which box owns the object?",
        "discovered_by": "patch_refinement_spatial_context",
        "private_extra": {"must": "not leak"},
    }

    rail = api._class_analysis_compact_refinement_rail(point)

    assert rail["status"] == "pair_conflict"
    assert rail["annotated_overlap_alternative_point_id"] == "building-box"
    assert rail["pair_conflict"] == {
        "enabled": True,
        "kind": "near_identical_cross_class_bbox",
        "review_mode": "dual_bbox_class_resolution",
        "point_id": "point_refined",
        "current_class": "ElevatedFixture",
        "other_point_id": "building-box",
        "target_bbox_xyxy": [11.0, 13.0, 89.0, 91.0],
        "other_bbox_xyxy": [10.0, 12.0, 90.0, 92.0],
        "other_class_name": "Building",
        "classes": ["ElevatedFixture", "Building"],
        "image_relpath": "train/frame.jpg",
        "split": "train",
        "iou": 0.97,
        "corner_similarity": 0.98,
        "target_area_covered": 0.99,
        "other_area_covered": 0.98,
        "relation": "duplicate_like",
        "score": 0.97,
        "question": "Which box owns the object?",
        "discovered_by": "patch_refinement_spatial_context",
    }

    mismatched = _refined_point()
    mismatched["refined_outlier"]["status"] = "pair_conflict"
    mismatched["refined_outlier"]["pair_conflict"] = dict(
        point["refined_outlier"]["pair_conflict"]
    )
    mismatched["refined_outlier"]["pair_conflict"][
        "other_bbox_xyxy"
    ] = [11.0, 12.0, 90.0, 92.0]
    assert api._class_analysis_compact_refinement_rail(mismatched) is None

    misbound = copy.deepcopy(point)
    misbound["refined_outlier"][
        "annotated_overlap_alternative_point_id"
    ] = "different-building-box"
    assert api._class_analysis_compact_refinement_rail(misbound) is None


def test_compact_refinement_rail_rejects_misbound_current_class():
    point = _refined_point()
    point["refined_outlier"]["current_class"] = "Building"

    assert api._class_analysis_compact_refinement_rail(point) is None


def test_qwen_refinement_observation_attaches_only_validated_preview(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    parent_dir = class_root / "ca_refined"
    preview_dir = parent_dir / api.CLASS_ANALYSIS_REFINEMENT_PREVIEW_DIRNAME
    preview_dir.mkdir(parents=True)
    sidecar_path = (
        parent_dir / api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME
    )
    sidecar_path.write_bytes(b"checksum-bound-sidecar")
    sidecar_sha256 = "cd" * 32
    source_sha256 = "ab" * 32
    preview_path = (
        preview_dir
        / f"point_refined-{sidecar_sha256[:16]}-{source_sha256[:16]}.png"
    )
    Image.new("RGB", (120, 80), (30, 80, 120)).save(preview_path)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_analysis_job = api.ClassAnalysisJob(
        job_id="ca_refined",
        status="completed",
    )
    monkeypatch.setattr(
        api,
        "_get_class_analysis_job",
        lambda _job_id: parent_analysis_job,
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_refinement_sidecar_contract",
        lambda **_kwargs: {
            "arrays": {},
            "sidecar_sha256": sidecar_sha256,
        },
    )
    monkeypatch.setattr(
        api,
        "get_class_analysis_refinement_preview",
        lambda _job_id, _point_id: api.FileResponse(
            str(preview_path),
            media_type="image/png",
        ),
    )
    job = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_refined",
        parent_job_id="ca_refined",
        point_id="point_refined",
    )

    observation = (
        api._class_analysis_qwen_review_refinement_observation(
            job,
            _refined_point(),
        )
    )

    assert observation["preview_status"] == "validated"
    assert len(observation["image_paths"]) == 1
    assert len(observation["evidence"]) == 2
    evidence, machine_evidence = observation["evidence"]
    assert evidence["kind"] == "patch_refinement_preview"
    assert machine_evidence["kind"] == "patch_refinement_machine_preview"
    assert observation["image_paths"][0].endswith(
        f"{machine_evidence['evidence_id']}.jpg"
    )
    assert machine_evidence["metadata"]["palette"] == {
        "current": "#00e8c4",
        "alternative": "#ff6a2a",
        "shared": "#bf5cff",
        "background": "neutral_grayscale",
    }
    integrity = evidence["metadata"]["integrity"]
    assert integrity["status"] == "validated"
    assert integrity["sidecar_sha256"] == sidecar_sha256
    assert integrity["source_image_sha256"] == source_sha256
    assert "intrinsic current support=0.39" in observation["summary"]
    assert "directed-pair learned probe score/threshold=0.264/0.18" in (
        observation["summary"]
    )
    assert "raw intrinsic margin (audit-only, never the decision score)=0.13" in (
        observation["summary"]
    )
    assert "current/alternative calibration sources=21/29" in (
        observation["summary"]
    )
    assert '"directed_pair_dominates": false' in (
        observation["summary"]
    )
    assert "cannot override" in observation["summary"]


def test_qwen_refinement_preview_failure_keeps_scalar_rail(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    parent_dir = class_root / "ca_refined_failed"
    parent_dir.mkdir(parents=True)
    (
        parent_dir / api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME
    ).write_bytes(b"stale")
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)

    def reject_contract(**_kwargs):
        raise ValueError("refinement_sidecar_checksum_mismatch")

    monkeypatch.setattr(
        api,
        "_class_analysis_refinement_sidecar_contract",
        reject_contract,
    )
    job = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_refined_failed",
        parent_job_id="ca_refined_failed",
        point_id="point_refined",
    )

    observation = (
        api._class_analysis_qwen_review_refinement_observation(
            job,
            _refined_point(),
        )
    )

    assert observation["preview_status"] == "unavailable"
    assert observation["image_paths"] == []
    assert observation["evidence"] == []
    assert observation["patch_refinement"]["status"] == (
        "explained_not_outlier"
    )
    assert "cannot override" in observation["summary"]


def test_final_vlm_context_keeps_refinement_preview_advisory(
    monkeypatch,
):
    monkeypatch.setattr(
        api,
        "CLASS_ANALYSIS_QWEN_REVIEW_FINAL_MAX_IMAGES",
        4,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Tool result for inspect_target_detail.\n"
                        "Evidence ids: target_detail_1"
                    ),
                },
                {"type": "image", "image": "/tmp/target.jpg"},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Tool result for inspect_patch_refinement.\n"
                        "Stage-2 status=explained_not_outlier.\n"
                        "Evidence ids: patch_refinement_preview_2"
                    ),
                },
                {"type": "image", "image": "/tmp/refinement.jpg"},
            ],
        },
    ]

    final_messages, policy = (
        api._class_analysis_qwen_review_final_context_messages(
            messages
        )
    )
    images = [
        item["image"]
        for message in final_messages
        for item in message.get("content") or []
        if item.get("type") == "image"
    ]
    text = "\n".join(
        str(item.get("text") or "")
        for message in final_messages
        for item in message.get("content") or []
        if item.get("type") == "text"
    )

    assert images == ["/tmp/target.jpg", "/tmp/refinement.jpg"]
    assert "inspect_patch_refinement" in policy["image_observations"]
    assert "cannot override clean pixels" in text


def test_refinement_and_consensus_share_the_four_image_final_context(
    monkeypatch,
):
    monkeypatch.setattr(
        api,
        "CLASS_ANALYSIS_QWEN_REVIEW_FINAL_MAX_IMAGES",
        4,
    )

    def observation(tool: str, image: str | None = None) -> dict:
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Tool result for {tool}.\n"
                        f"Evidence ids: {tool}_1"
                    ),
                },
                {"type": "image", "image": image or f"/tmp/{tool}.jpg"},
            ],
        }

    final_messages, policy = (
        api._class_analysis_qwen_review_final_context_messages(
            [
                observation("inspect_target_detail"),
                observation("zoom_source_region"),
                observation("inspect_specificity_region_contrast"),
                observation("inspect_source_overlay"),
                observation(
                    "inspect_patch_refinement",
                    "/tmp/patch_refinement_machine_preview.jpg",
                ),
                observation(
                    "inspect_local_consensus_context",
                    "/tmp/local_consensus_dot_map.jpg",
                ),
            ]
        )
    )
    images = [
        item["image"]
        for message in final_messages
        for item in message.get("content") or []
        if item.get("type") == "image"
    ]

    assert images == [
        "/tmp/inspect_target_detail.jpg",
        "/tmp/zoom_source_region.jpg",
        "/tmp/patch_refinement_machine_preview.jpg",
        "/tmp/local_consensus_dot_map.jpg",
    ]
    assert "inspect_patch_refinement" in policy["image_observations"]
    assert "inspect_local_consensus_context" in policy["image_observations"]
    assert "inspect_source_overlay" not in policy["image_observations"]
    assert (
        "inspect_specificity_region_contrast"
        not in policy["image_observations"]
    )


def test_refinement_rail_is_separate_from_clean_visual_evidence():
    job = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_ledger",
        parent_job_id="ca_ledger",
        point_id="point_refined",
    )
    job.evidence = [
        {
            "evidence_id": "patch_refinement_preview_1",
            "kind": "patch_refinement_preview",
            "title": "Stage-2 patch refinement",
            "summary": "advisory heatmap",
            "metadata": {
                "integrity": {
                    "status": "validated",
                    "sidecar_sha256": "cd" * 32,
                }
            },
        }
    ]
    rail = api._class_analysis_compact_refinement_rail(
        _refined_point()
    )

    ledger = api._class_analysis_qwen_review_evidence_ledger(
        job,
        patch_refinement=rail,
    )
    deterministic = (
        api._class_analysis_qwen_review_deterministic_context(
            job,
            patch_refinement=rail,
        )
    )

    assert ledger["clean_visual_evidence_ids"] == []
    assert ledger["patch_refinement_evidence_ids"] == [
        "patch_refinement_preview_1"
    ]
    assert ledger["patch_refinement"]["advisory_only"] is True
    assert (
        deterministic["patch_refinement"]["status"]
        == "explained_not_outlier"
    )
    assert (
        deterministic["patch_refinement_preview"]["integrity"][
            "status"
        ]
        == "validated"
    )


def test_compact_rail_normalizes_legacy_infinite_threshold_sentinels():
    normalized = (
        api._class_analysis_normalize_legacy_refinement_threshold_sentinels(
            {
                "current_negative_threshold": -float("inf"),
                "current_support_threshold": float("inf"),
                "current_strong_threshold": float("inf"),
                "alternative_negative_threshold": float("nan"),
            }
        )
    )

    assert normalized["current_negative_threshold"] == -2.0
    assert normalized["current_support_threshold"] == 2.0
    assert normalized["current_strong_threshold"] == 2.0
    assert math.isnan(normalized["alternative_negative_threshold"])
