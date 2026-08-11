from __future__ import annotations

import copy
import math

import pytest

import localinferenceapi as api
from services.class_analysis_selector_v6 import (
    DATASET_OVERLAP_APPLICATION_CONTRACT,
    DATASET_OVERLAP_DIAGNOSTIC_CONTRACT,
    DATASET_OVERLAP_DIAGNOSTIC_FEATURES,
    FORBIDDEN_MODEL_FEATURE_NAMES,
    FORBIDDEN_MODEL_FEATURE_PREFIXES,
    GLOBAL_ACTIONABILITY_MODEL_CONTRACT,
    GLOBAL_CATEGORICAL_FEATURES,
    GLOBAL_DENSE_FEATURES,
    GLOBAL_NUMERIC_FEATURES,
    LEGACY_V6_DATASET_OVERLAP_APPLICATION_CONTRACT,
    LEGACY_V6_DATASET_OVERLAP_DIAGNOSTIC_CONTRACT,
    LEGACY_V6_SELECTOR_PRIORITY_CONTRACT,
    SELECTOR_FEATURE_CONTRACT,
    SELECTOR_PRIORITY_CONTRACT,
    _score_hist_gradient_boosting_model,
    _serialize_hist_gradient_boosting,
    build_selector_feature_row,
    load_default_selector_model,
    score_selector_candidates,
    selector_model_digest,
    validate_selector_model_artifact,
)


def _candidate(
    point_id: str,
    *,
    current_class: str = "Bike",
    alternative_class: str = "Person",
    suspicion: float = 0.9,
    eligible_dataset_overlap: bool = False,
) -> dict:
    alternative_point_id = f"{point_id}-alternative"
    evidence = {
        "current_class": current_class,
        "alternative_class": alternative_class,
        "current_support_score": 0.45,
        "current_support_threshold": 0.50,
        "alternative_support_score": 0.70,
        "alternative_support_threshold": 0.50,
        "intrinsic_current_support": 0.30,
        "intrinsic_alternative_support": 0.65,
        "directed_pair_probe_score": 0.70,
        "directed_pair_probe_threshold": 0.50,
        "directed_pair_heldout_auroc": 0.80,
        "directed_pair_eval_auroc_lower_bound": 0.65,
        "view_agreement": 0.80,
        "target_bbox_area": 0.01,
        "target_bbox_width": 0.10,
        "target_bbox_height": 0.10,
        "annotated_overlap_alternative_point_id": alternative_point_id,
        "decision_gates": {
            "source_resolution_sufficient": True,
            "current_present": True,
            "alternative_present": True,
            "annotated_overlap": True,
            "alternative_evidence_localized_to_overlap": True,
        },
        "frequent_overlap_prior": {
            "candidate_capture_group_id": (
                f"capture-{point_id}" if eligible_dataset_overlap else ""
            ),
            "candidate_capture_group_excluded": eligible_dataset_overlap,
            "candidate_source_excluded": eligible_dataset_overlap,
            "fit_screening_adjustment_eligible": eligible_dataset_overlap,
            "candidate_overlap_evidence": (
                [
                    {
                        "point_id": alternative_point_id,
                        "class_name": alternative_class,
                        "geometry_stratum": "material_nonduplicate",
                    }
                ]
                if eligible_dataset_overlap
                else []
            ),
            "strata": (
                [
                    {
                        "geometry_stratum": "material_nonduplicate",
                        "fit_screening_adjustment_eligible": True,
                        "reliable": True,
                        "source_independence_verified": True,
                        "adjustment_eligible": True,
                        "provisional": False,
                        "candidate_overlap": True,
                        "candidate_overlap_count": 1,
                        "candidate_overlap_strength": 0.9,
                        "smoothed_capture_group_incidence": 0.75,
                        "capture_group_incidence_wilson_lower_bound": 0.4,
                        "conservative_prior_strength": 0.35,
                        "eligible_capture_group_count": 40,
                        "overlap_capture_group_count": 30,
                        "reliability_tier": "trusted",
                    }
                ]
                if eligible_dataset_overlap
                else []
            ),
        },
    }
    return {
        "point_id": point_id,
        "class_name": current_class,
        "suggested_neighbor_class": alternative_class,
        "wrong_class_suspicion": suspicion,
        "embedding_wrong_class_suspicion": suspicion,
        "refined_outlier": evidence,
    }


def _context(point_id: str) -> dict:
    return {
        "point_id": point_id,
        "available": True,
        "image_object_count": 13,
        "same_class_count": 4,
        "same_class_fraction": 4 / 13,
        "trusted_same_class_anchor_count": 3,
        "trusted_alternative_class_anchor_count": 5,
        "same_class_scale_reference_available": True,
        "alternative_scale_reference_available": True,
        "scale_contrast_available": True,
        "perspective_available": True,
        "same_class_log_width_residual": -0.8,
        "same_class_log_height_residual": -0.5,
        "same_class_log_area_residual": -1.3,
        "same_class_log_aspect_residual": -0.3,
        "perspective_log_scale_residual": -0.7,
        "current_minus_alternative_abs_scale_residual": 0.9,
        "local_object_count_r10": 6,
        "local_same_class_count_r10": 2,
        "max_other_class_iou": 0.55,
        "max_target_coverage_by_other": 0.95,
        "bbox_touches_border": False,
        "bbox_outside_fraction": 0.0,
        "bbox_width_norm": 0.10,
        "bbox_height_norm": 0.08,
        "bbox_area_fraction": 0.008,
        "same_class_peer_width_median_norm": 0.20,
        "same_class_peer_height_median_norm": 0.14,
        "same_class_peer_area_median_fraction": 0.028,
    }


def test_default_v6_model_is_digest_bound_pair_blind_and_hgb_only():
    artifact = load_default_selector_model()
    validate_selector_model_artifact(artifact)

    assert artifact["selector_contract"] == (
        LEGACY_V6_SELECTOR_PRIORITY_CONTRACT
    )
    assert artifact["feature_contract"] == SELECTOR_FEATURE_CONTRACT
    assert artifact["global_actionability_model_contract"] == (
        GLOBAL_ACTIONABILITY_MODEL_CONTRACT
    )
    assert artifact["dataset_overlap_application_contract"] == (
        LEGACY_V6_DATASET_OVERLAP_APPLICATION_CONTRACT
    )
    assert artifact["dataset_overlap_diagnostic_contract"] == (
        LEGACY_V6_DATASET_OVERLAP_DIAGNOSTIC_CONTRACT
    )
    assert artifact["dataset_overlap_scoring_effect_enabled"] is False
    assert selector_model_digest(artifact) == artifact["model_digest"]
    assert artifact["global_numeric_features"] == list(GLOBAL_NUMERIC_FEATURES)
    assert artifact["global_categorical_features"] == list(
        GLOBAL_CATEGORICAL_FEATURES
    )
    assert artifact["global_dense_features"] == list(GLOBAL_DENSE_FEATURES)
    assert artifact["dataset_overlap_diagnostic_features"] == list(
        DATASET_OVERLAP_DIAGNOSTIC_FEATURES
    )
    assert not set(DATASET_OVERLAP_DIAGNOSTIC_FEATURES).intersection(
        GLOBAL_DENSE_FEATURES
    )
    assert "pair_adapters" not in artifact
    assert "pair_adapter_features" not in artifact
    assert "pair_adapter_application_contract" not in artifact
    for head in ("actionability_model", "reviewability_model"):
        model = artifact[head]
        assert model["kind"] == "binary-numeric-hist-gradient-boosting-v1"
        assert model["feature_names"] == list(GLOBAL_DENSE_FEATURES)
        assert model["trees"]
    assert not {
        "current_class",
        "alternative_class",
        "directed_pair",
    }.intersection(GLOBAL_DENSE_FEATURES)
    for feature_name in (
        *GLOBAL_NUMERIC_FEATURES,
        *GLOBAL_CATEGORICAL_FEATURES,
        *GLOBAL_DENSE_FEATURES,
    ):
        assert feature_name not in FORBIDDEN_MODEL_FEATURE_NAMES
        assert not any(
            feature_name.startswith(prefix)
            for prefix in FORBIDDEN_MODEL_FEATURE_PREFIXES
        )


def test_v6_rejects_auxiliary_head_feature_outside_global_allowlist():
    artifact = copy.deepcopy(load_default_selector_model())
    strict_model = artifact["strict_mislabeled_given_actionable_model"]
    assert strict_model["kind"] == "scaled-sparse-logistic-v1"
    strict_model["feature_names"].append("overlap_incidence")
    strict_model["scale"].append(1.0)
    strict_model["coefficients"].append(0.1)
    artifact["model_digest"] = selector_model_digest(artifact)

    with pytest.raises(
        ValueError,
        match="strict_mislabeled_feature_allowlist",
    ):
        validate_selector_model_artifact(artifact)


def test_v6_portable_hgb_export_matches_sklearn_missing_value_routes():
    np = pytest.importorskip("numpy")
    ensemble = pytest.importorskip("sklearn.ensemble")
    rng = np.random.default_rng(20260802)
    matrix = rng.normal(size=(400, 7))
    matrix[::5, 1] = np.nan
    matrix[1::11, 5] = np.nan
    latent = (
        1.3 * matrix[:, 0]
        - 0.8 * np.nan_to_num(matrix[:, 1], nan=-1.5)
        + 0.5 * matrix[:, 3]
        + 0.7 * np.isnan(matrix[:, 5])
        + rng.normal(scale=0.7, size=len(matrix))
    )
    labels = (latent > np.quantile(latent, 0.72)).astype(np.int64)
    config = {
        "learning_rate": 0.07,
        "max_iter": 37,
        "max_leaf_nodes": 5,
        "min_samples_leaf": 8,
        "l2_regularization": 3.0,
        "early_stopping": False,
        "random_state": 20260802,
    }
    sklearn_model = ensemble.HistGradientBoostingClassifier(**config).fit(
        matrix,
        labels,
    )
    feature_names = [f"f{index}" for index in range(matrix.shape[1])]
    portable = _serialize_hist_gradient_boosting(
        sklearn_model,
        feature_names=feature_names,
        training_config=config,
    )

    split_nodes = [
        node
        for tree in portable["trees"]
        for node in tree
        if "leaf" not in node
    ]
    assert {node["missing_left"] for node in split_nodes} == {False, True}
    assert abs(portable["baseline_log_odds"]) > 0.1
    probes = [row.copy() for row in matrix[:100]]
    probes.extend(
        [
            np.full(matrix.shape[1], np.nan),
            np.zeros(matrix.shape[1]),
        ]
    )
    for node in split_nodes[:20]:
        row = np.zeros(matrix.shape[1])
        row[node["feature"]] = node["threshold"]
        probes.append(row)
    probe_matrix = np.asarray(probes, dtype=np.float64)
    expected_raw = sklearn_model.decision_function(probe_matrix)
    expected_probability = sklearn_model.predict_proba(probe_matrix)[:, 1]
    actual = [
        _score_hist_gradient_boosting_model(
            portable,
            dict(zip(feature_names, row)),
        )
        for row in probe_matrix
    ]
    np.testing.assert_allclose(
        [item[1] for item in actual],
        expected_raw,
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        [item[0] for item in actual],
        expected_probability,
        rtol=0.0,
        atol=2e-15,
    )


def test_v6_explicit_empty_artifact_fails_closed():
    candidate = _candidate("empty-artifact")
    with pytest.raises(ValueError, match="selector_model_invalid"):
        score_selector_candidates(
            [candidate],
            same_image_context_by_point_id={
                "empty-artifact": _context("empty-artifact")
            },
            artifact={},
        )


def test_v4_and_v5_policy_outputs_cannot_change_v6_features():
    candidate = _candidate(
        "same-input", eligible_dataset_overlap=True
    )
    expected = build_selector_feature_row(candidate, _context("same-input"))
    mutated = copy.deepcopy(candidate)
    mutated_evidence = mutated["refined_outlier"]
    mutated_evidence.update(
        {
            "status": "confirmed_outlier",
            "qualified_for_human_review": True,
            "human_review_rank": 1,
            "selector_priority_rank": 999,
            "selector_priority_status_band_name": "confirmed_outlier",
            "selector_priority_overlap_adjustment": 0.35,
            "selector_v5": {"expected_review_utility": 1.0},
        }
    )
    mutated_evidence["frequent_overlap_prior"].update(
        {
            "applies": True,
            "semantic_priority_adjustment": 0.35,
            "triage_frequency_adjustment": 0.06,
            "priority_adjustment": 0.41,
        }
    )

    assert build_selector_feature_row(
        mutated, _context("same-input")
    ) == expected


def test_v6_payload_contains_counts_scale_and_exact_hgb_utility():
    candidate = _candidate("scale-row")
    scored, summary = score_selector_candidates(
        [candidate],
        same_image_context_by_point_id={
            "scale-row": _context("scale-row")
        },
    )
    payload = scored["scale-row"]

    assert payload["global_model"]["contract"] == (
        GLOBAL_ACTIONABILITY_MODEL_CONTRACT
    )
    assert payload["global_model"]["actionable_probability"] == pytest.approx(
        payload["actionable_probability"]
    )
    assert math.isfinite(payload["global_model"]["raw_margin"])
    assert summary["context_available_count"] == 1
    context = payload["same_image_context"]
    assert context["image_object_count"] == 13
    assert context["same_class_count"] == 4
    assert context["bbox_width_norm"] == pytest.approx(0.10)
    assert context["same_class_log_width_residual"] < 0.0
    assert payload["expected_review_utility"] == pytest.approx(
        payload["actionable_probability"]
        * (0.75 + 0.25 * payload["reviewability_probability"])
    )


def test_dataset_overlap_gate_applies_bounded_rank_only_effect():
    candidate = _candidate(
        "overlap-row", eligible_dataset_overlap=True
    )
    scored, summary = score_selector_candidates(
        [candidate],
        same_image_context_by_point_id={
            "overlap-row": _context("overlap-row")
        },
    )
    payload = scored["overlap-row"]
    overlap = payload["dataset_overlap"]

    assert payload["dataset_overlap_application_contract"] == (
        DATASET_OVERLAP_APPLICATION_CONTRACT
    )
    assert payload["dataset_overlap_diagnostic_contract"] == (
        DATASET_OVERLAP_DIAGNOSTIC_CONTRACT
    )
    assert payload["dataset_overlap_scoring_effect_enabled"] is True
    assert overlap["applicable"] is True
    assert overlap["application_reason"] == (
        "eligible_dataset_overlap_explanation"
    )
    assert overlap["scoring_effect_enabled"] is True
    assert overlap["rank_only"] is True
    assert overlap["uses_human_review_labels"] is False
    assert overlap["applied"] is True
    assert overlap["probability_delta"] == 0.0
    assert overlap["utility_delta"] < 0.0
    assert 0.0 < overlap["rank_discount_fraction"] <= 0.25
    assert overlap["counterfactual_actionable_probability"] == pytest.approx(
        payload["actionable_probability"]
    )
    assert overlap["counterfactual_expected_review_utility"] == pytest.approx(
        payload["base_expected_review_utility"]
    )
    assert payload["expected_review_utility"] < (
        payload["base_expected_review_utility"]
    )
    overlap_summary = summary["dataset_overlap"]
    assert overlap_summary["applicable_candidate_count"] == 1
    assert overlap_summary["applied_candidate_count"] == 1
    assert overlap_summary["effect_candidate_count"] == 1
    assert overlap_summary["zero_effect_candidate_count"] == 0
    assert overlap_summary["maximum_absolute_probability_effect"] == 0.0
    assert overlap_summary["maximum_absolute_utility_effect"] > 0.0


def test_dataset_overlap_requires_affirmative_current_class_evidence():
    candidate = _candidate(
        "indeterminate-current", eligible_dataset_overlap=True
    )
    gates = candidate["refined_outlier"]["decision_gates"]
    gates.pop("current_present")
    scored, _summary = score_selector_candidates(
        [candidate],
        same_image_context_by_point_id={
            "indeterminate-current": _context("indeterminate-current")
        },
    )
    overlap = scored["indeterminate-current"]["dataset_overlap"]

    assert overlap["affirmative_current_evidence"] is False
    assert overlap["applicable"] is False
    assert overlap["application_reason"] == (
        "current_class_not_affirmatively_present"
    )
    assert overlap["applied"] is False
    assert overlap["probability_delta"] == 0.0
    assert overlap["utility_delta"] == 0.0


def test_dataset_overlap_rank_discount_preserves_pair_conflicts():
    candidate = _candidate(
        "pair-conflict", eligible_dataset_overlap=True
    )
    candidate["refined_outlier"]["status"] = "pair_conflict"
    scored, _summary = score_selector_candidates(
        [candidate],
        same_image_context_by_point_id={
            "pair-conflict": _context("pair-conflict")
        },
    )
    overlap = scored["pair-conflict"]["dataset_overlap"]

    assert overlap["applicable"] is False
    assert overlap["application_reason"] == "pair_conflict_preserved"
    assert overlap["applied"] is False
    assert overlap["probability_delta"] == 0.0
    assert overlap["utility_delta"] == 0.0


def test_dataset_overlap_rank_discount_requires_capture_and_source_loo():
    candidate = _candidate(
        "not-left-out", eligible_dataset_overlap=True
    )
    prior = candidate["refined_outlier"]["frequent_overlap_prior"]
    prior["candidate_source_excluded"] = False
    scored, _summary = score_selector_candidates(
        [candidate],
        same_image_context_by_point_id={
            "not-left-out": _context("not-left-out")
        },
    )
    overlap = scored["not-left-out"]["dataset_overlap"]

    assert overlap["applicable"] is False
    assert overlap["application_reason"] == (
        "candidate_capture_group_not_left_out"
    )
    assert overlap["applied"] is False
    assert overlap["utility_delta"] == 0.0


def test_candidate_scoring_rejects_duplicate_identity_without_mutation():
    candidate = _candidate("duplicate")
    before = copy.deepcopy(candidate)
    with pytest.raises(ValueError, match="selector_v6_candidate_identity_invalid"):
        score_selector_candidates(
            [candidate, copy.deepcopy(candidate)],
            same_image_context_by_point_id={},
        )
    assert candidate == before


def test_active_rank_publication_is_atomic_when_v6_scoring_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    candidates = [_candidate("one"), _candidate("two")]
    before = copy.deepcopy(candidates)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic_selector_failure")

    monkeypatch.setattr(
        api,
        "_class_analysis_score_selector_candidates_v6",
        fail,
    )
    with pytest.raises(RuntimeError, match="synthetic_selector_failure"):
        api._class_analysis_assign_selector_priority_ranks(candidates)
    assert candidates == before


def test_active_rank_is_global_complete_v6_and_preserves_human_fields():
    candidates = [
        _candidate("row-c", suspicion=0.55),
        _candidate("row-a", suspicion=0.95),
        _candidate("row-b", suspicion=0.75),
    ]
    for index, candidate in enumerate(candidates, start=1):
        candidate["refined_outlier"].update(
            {
                "status": "unresolved" if index % 2 else "pair_conflict",
                "qualified_for_human_review": index == 1,
                "human_review_rank": index if index == 1 else None,
            }
        )
    original_human_fields = [
        (
            row["refined_outlier"]["status"],
            row["refined_outlier"]["qualified_for_human_review"],
            row["refined_outlier"]["human_review_rank"],
        )
        for row in candidates
    ]

    summary = api._class_analysis_assign_selector_priority_ranks(candidates)
    ranks = [
        row["refined_outlier"]["selector_priority_rank"]
        for row in candidates
    ]

    assert summary["contract"] == SELECTOR_PRIORITY_CONTRACT
    assert summary["candidate_count"] == len(candidates)
    assert summary["status_band_partitioned"] is False
    assert summary["cross_status_band_reordering"] is True
    assert summary["dataset_overlap_applied_candidate_count"] == 0
    assert sorted(ranks) == [1, 2, 3]
    assert [
        (
            row["refined_outlier"]["status"],
            row["refined_outlier"]["qualified_for_human_review"],
            row["refined_outlier"]["human_review_rank"],
        )
        for row in candidates
    ] == original_human_fields
    for row in candidates:
        evidence = row["refined_outlier"]
        payload = evidence["selector_v6"]
        assert "selector_v5" not in evidence
        assert evidence["selector_priority_score"] == pytest.approx(
            payload["expected_review_utility"]
        )
        assert math.isfinite(evidence["selector_priority_score"])
        assert payload["dataset_overlap"]["applied"] is False
        assert payload["dataset_overlap"]["utility_delta"] == 0.0


def test_stale_v5_selector_contract_fails_closed():
    with pytest.raises(
        ValueError,
        match="class_analysis_selector_priority_artifact_invalid:contract",
    ):
        api._class_analysis_validate_selector_priority_artifact(
            refinement={},
            refinement_rows=[],
            configured_contract=(
                "expected-review-utility-global-plus-guarded-pair-scale-v5"
            ),
        )
