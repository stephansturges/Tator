from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

import services.class_analysis_patch_refinement as refinement
from services.class_analysis_patch_refinement import (
    CALIBRATION_STATUS_SOURCE_AWARE,
    MIN_PAIR_PROBE_EVAL_SOURCES_PER_CLASS,
    MIN_PAIR_PROBE_FIT_SOURCES_PER_CLASS,
    MIN_PAIR_CALIBRATION_HELDOUT_SOURCE_GROUPS,
    MIN_HELDOUT_TARGET_SOURCE_PASS_FRACTION,
    MIN_PAIR_PROBE_METRIC_FRACTION,
    PAIR_PROBE_CONTRACT,
    PAIR_PROBE_LOWER_BOUND_CONTRACT,
    PAIR_PROBE_VIEW_CONTRACT,
    REFINEMENT_DECISION_CONTRACT,
    REFINEMENT_SCHEMA,
    STATUS_CONFIRMED_OUTLIER,
    STATUS_EXPLAINED_NOT_OUTLIER,
    STATUS_MIXED_OR_COMPOSITE,
    STATUS_UNRESOLVED,
    ExactTwoViewPairCalibrationBuilder,
    ReferenceBank,
    RefinementConfig,
    StreamingReferenceBankBuilder,
    _calibration_source_split_digest,
    _component_corresponds,
    _largest_component_geometry,
    _source_fingerprint,
    _stable_source_group_folds,
    _global_heldout_sources,
    _support_from_heat,
    _weighted_top_fraction_score,
    confirmation_invariants_hold,
    exact_two_view_pair_features,
    pair_metrics_are_reliable,
    score_candidate,
    unresolved_evidence,
)


HELDOUT_KEYS = tuple(
    [f"heldout-a-{index}" for index in range(16)]
    + [f"heldout-b-{index}" for index in range(16)]
)
FIT_KEYS = tuple(f"fit-source-{index}" for index in range(6))


def _off_diagonal(value: float, diagonal: float = 0.0, *, dtype=np.float32):
    return np.asarray([[diagonal, value], [value, diagonal]], dtype=dtype)


def _bank(*, pair_reliable: bool = True, include_pair: bool = True) -> ReferenceBank:
    heldout_ids = np.asarray(
        sorted(_source_fingerprint(key) for key in HELDOUT_KEYS), dtype="<U16"
    )
    fit_ids = np.asarray(
        sorted(_source_fingerprint(key) for key in FIT_KEYS), dtype="<U16"
    )
    prototype_sources = np.asarray(
        [
            [fit_ids[0], fit_ids[0], fit_ids[1], fit_ids[1], fit_ids[2], fit_ids[2]],
            [fit_ids[3], fit_ids[3], fit_ids[4], fit_ids[4], fit_ids[5], fit_ids[5]],
        ]
    )
    background_sources = np.asarray(
        [
            [fit_ids[3], fit_ids[3], fit_ids[4], fit_ids[4], fit_ids[5], fit_ids[5]],
            [fit_ids[0], fit_ids[0], fit_ids[1], fit_ids[1], fit_ids[2], fit_ids[2]],
        ]
    )
    pair_kwargs = {}
    if include_pair:
        reliable = _off_diagonal(pair_reliable, False, dtype=bool)
        pair_tiers = np.asarray(
            [
                ["low", "high" if pair_reliable else "low"],
                ["high" if pair_reliable else "low", "low"],
            ]
        )
        metrics = _off_diagonal(1.0 if pair_reliable else 0.0)
        counts = _off_diagonal(8, 0, dtype=np.int32)
        statuses = np.asarray(
            [["not_applicable", "ok"], ["ok", "not_applicable"]]
        )
        digests = np.asarray([["", "a" * 64], ["b" * 64, ""]])
        pair_kwargs = {
            "pair_reliable": reliable,
            "pair_reliability_tiers": pair_tiers,
            "pair_heldout_aurocs": metrics,
            "pair_dominance_thresholds": _off_diagonal(0.0),
            "pair_current_negative_thresholds": np.full(
                (2, 2), 0.10, dtype=np.float32
            ),
            "pair_alternative_strong_thresholds": np.full(
                (2, 2), 0.40, dtype=np.float32
            ),
            "pair_current_source_counts": counts,
            "pair_alternative_source_counts": counts.copy(),
            "pair_current_patch_counts": counts.copy(),
            "pair_alternative_patch_counts": counts.copy(),
            "pair_alternative_passing_source_fractions": metrics.copy(),
            "pair_probe_weights": np.broadcast_to(
                np.asarray([-np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32),
                (2, 2, 2),
            ).copy(),
            "pair_probe_thresholds": _off_diagonal(0.0),
            "pair_probe_oof_aurocs": metrics.copy(),
            "pair_probe_fold_counts": _off_diagonal(1, 0, dtype=np.int32),
            "pair_probe_fit_statuses": statuses,
            "pair_probe_fold_digests": digests,
            "pair_probe_eval_auroc_lower_bounds": (
                _off_diagonal(0.90 if pair_reliable else 0.0)
            ),
            "pair_probe_fit_current_source_counts": counts.copy(),
            "pair_probe_fit_alternative_source_counts": counts.copy(),
            "pair_probe_eval_current_source_counts": counts.copy(),
            "pair_probe_eval_alternative_source_counts": counts.copy(),
            "pair_probe_fit_balanced_accuracies": metrics.copy(),
            "pair_probe_eval_sensitivities": metrics.copy(),
            "pair_probe_eval_specificities": metrics.copy(),
            "pair_current_absence_eval_fractions": metrics.copy(),
            "pair_alternative_strong_eval_fractions": metrics.copy(),
            "pair_probe_fit_eval_split_digests": digests.copy(),
            "pair_calibration_class_source_ids": np.asarray(
                [
                    sorted(
                        _source_fingerprint(f"heldout-a-{index}")
                        for index in range(16)
                    ),
                    sorted(
                        _source_fingerprint(f"heldout-b-{index}")
                        for index in range(16)
                    ),
                ]
            ),
            "pair_calibration_class_source_counts": np.asarray(
                [16, 16], dtype=np.int32
            ),
        }
    bank = ReferenceBank(
        class_names=["A", "B"],
        prototypes=np.asarray(
            [[[1.0, 0.0, 0.0]] * 6, [[0.0, 1.0, 0.0]] * 6],
            dtype=np.float32,
        ),
        prototype_counts=np.asarray([6, 6], dtype=np.int32),
        prototype_source_ids=prototype_sources,
        background_prototypes=np.asarray(
            [[[-1.0, 0.0, 0.0]] * 6, [[0.0, -1.0, 0.0]] * 6],
            dtype=np.float32,
        ),
        background_prototype_counts=np.asarray([6, 6], dtype=np.int32),
        background_prototype_source_ids=background_sources,
        anchor_counts=np.asarray([64, 64], dtype=np.int32),
        distinct_source_counts=np.asarray([8, 8], dtype=np.int32),
        reliable=np.asarray([True, True]),
        reliability_tiers=np.asarray(["high", "high"]),
        heldout_aurocs=np.asarray([1.0, 1.0], dtype=np.float32),
        support_thresholds=np.asarray([0.20, 0.20], dtype=np.float32),
        strong_support_thresholds=np.asarray([0.40, 0.40], dtype=np.float32),
        negative_support_thresholds=np.asarray([0.10, 0.10], dtype=np.float32),
        projection_mean=np.zeros(3, dtype=np.float32),
        projection_components=np.eye(3, dtype=np.float32),
        calibration_status=CALIBRATION_STATUS_SOURCE_AWARE,
        calibration_split_digest=_calibration_source_split_digest(
            heldout_ids.tolist(), fit_ids.tolist()
        ),
        calibration_heldout_source_count=len(heldout_ids),
        calibration_fit_source_count=len(fit_ids),
        calibration_target_patch_counts=np.asarray([8, 8], dtype=np.int32),
        calibration_background_patch_counts=np.asarray([8, 8], dtype=np.int32),
        calibration_target_source_counts=np.asarray([2, 2], dtype=np.int32),
        calibration_background_source_counts=np.asarray([2, 2], dtype=np.int32),
        calibration_target_passing_source_counts=np.asarray([2, 2], dtype=np.int32),
        calibration_target_source_pass_fractions=np.ones(2, dtype=np.float32),
        fit_target_patch_counts=np.asarray([12, 12], dtype=np.int32),
        fit_background_patch_counts=np.asarray([12, 12], dtype=np.int32),
        fit_target_source_counts=np.asarray([3, 3], dtype=np.int32),
        fit_background_source_counts=np.asarray([3, 3], dtype=np.int32),
        calibration_heldout_source_ids=heldout_ids,
        calibration_fit_source_ids=fit_ids,
        pair_probe_contract=PAIR_PROBE_CONTRACT,
        pair_probe_view_contract=PAIR_PROBE_VIEW_CONTRACT,
        pair_probe_lower_bound_contract=PAIR_PROBE_LOWER_BOUND_CONTRACT,
        **pair_kwargs,
    )
    bank.validate()
    return bank


E_A = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
E_B = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
NEUTRAL = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
ALIAS = np.asarray([np.sqrt(0.5), np.sqrt(0.5), 0.0], dtype=np.float32)


def _score(
    tight: np.ndarray,
    context: np.ndarray | None = None,
    *,
    target_bbox=(0.0, 0.0, 40.0, 40.0),
    crop_boxes=None,
    overlaps=(),
    bank: ReferenceBank | None = None,
    query_source_key: str = "",
) -> dict:
    return score_candidate(
        point_id="candidate",
        current_class="A",
        alternative_class="B",
        token_views=[tight, tight if context is None else context],
        crop_boxes=(
            [[0.0, 0.0, 40.0, 40.0], [0.0, 0.0, 40.0, 40.0]]
            if crop_boxes is None
            else crop_boxes
        ),
        target_bbox=target_bbox,
        grid_shape=(4, 4),
        alternative_overlap_boxes=list(overlaps),
        bank=_bank() if bank is None else bank,
        config=RefinementConfig(),
        query_source_key=query_source_key,
    )


def test_v5_contracts_and_weighted_fractional_mask_parity():
    assert REFINEMENT_SCHEMA == "class-analysis-patch-refinement-v5"
    assert REFINEMENT_DECISION_CONTRACT == "class-analysis-patch-decision-v9"
    assert PAIR_PROBE_CONTRACT.endswith("fit-thresholds-angle-grid-l2-sign-v4")
    assert MIN_HELDOUT_TARGET_SOURCE_PASS_FRACTION == 0.75
    assert MIN_PAIR_PROBE_METRIC_FRACTION == 0.50
    values = np.asarray([1.0, 0.9, -0.5], dtype=np.float32)
    weights = np.asarray([0.01, 0.49, 0.50], dtype=np.float32)
    expected = _weighted_top_fraction_score(
        values, weights, selected_fraction=0.05
    )
    inferred, _coverage = _support_from_heat(
        values,
        weights,
        positive_margin=0.0,
        selected_fraction=0.05,
    )
    features = exact_two_view_pair_features(
        [values, values],
        [np.zeros_like(values), np.zeros_like(values)],
        [weights, weights],
        selected_fraction=0.05,
    )
    assert expected == pytest.approx(0.92, abs=1e-6)
    assert inferred == pytest.approx(expected, abs=1e-7)
    assert features[0] == pytest.approx(expected, abs=1e-7)


def test_streaming_builder_preserves_coverage_priority_across_same_source_anchors():
    builder = StreamingReferenceBankBuilder(
        RefinementConfig(patches_per_anchor=1, patch_reservoir_per_class=1)
    )
    builder.add(
        class_name="A",
        source_key="same-source",
        patch_tokens=np.asarray([E_A], dtype=np.float32),
        valid_mask=np.asarray([0.20], dtype=np.float32),
    )
    builder.add(
        class_name="A",
        source_key="same-source",
        patch_tokens=np.asarray([E_B], dtype=np.float32),
        valid_mask=np.asarray([0.90], dtype=np.float32),
    )
    assert len(builder._rows["A"]) == 1
    np.testing.assert_array_equal(builder._rows["A"][0][2], E_B)


def test_global_holdout_reserves_exact_pair_fit_and_eval_pool_deterministically():
    groups = [
        [f"class-{class_index}-source-{source_index}" for source_index in range(19)]
        for class_index in range(3)
    ]
    sources = sorted({source for group in groups for source in group})
    heldout = _global_heldout_sources(sources, source_groups=groups)
    reversed_heldout = _global_heldout_sources(
        reversed(sources),
        source_groups=(reversed(group) for group in reversed(groups)),
    )

    assert MIN_PAIR_CALIBRATION_HELDOUT_SOURCE_GROUPS == 16
    assert heldout == reversed_heldout
    for group in groups:
        group_sources = set(group)
        assert len(group_sources.intersection(heldout)) >= 16
        assert len(group_sources.difference(heldout)) >= 3


def test_pair_reliability_has_one_shared_predicate_and_each_gate_fails():
    good = {
        "current_class_reliable": True,
        "alternative_class_reliable": True,
        "fit_current_source_count": 8,
        "fit_alternative_source_count": 8,
        "eval_current_source_count": 8,
        "eval_alternative_source_count": 8,
        "eval_auroc": 0.95,
        "eval_auroc_lower_bound": 0.80,
        "fit_balanced_accuracy": 0.75,
        "eval_sensitivity": 0.75,
        "eval_specificity": 0.75,
        "current_absence_eval_fraction": 0.75,
        "alternative_strong_eval_fraction": 0.75,
    }
    assert pair_metrics_are_reliable(**good)
    failures = {
        "current_class_reliable": False,
        "alternative_class_reliable": False,
        "fit_current_source_count": 7,
        "fit_alternative_source_count": 7,
        "eval_current_source_count": 7,
        "eval_alternative_source_count": 7,
        "eval_auroc": 0.79,
        "eval_auroc_lower_bound": 0.59,
        "fit_balanced_accuracy": 0.49,
        "eval_sensitivity": 0.49,
        "eval_specificity": 0.49,
        "current_absence_eval_fraction": 0.49,
        "alternative_strong_eval_fraction": 0.49,
    }
    for field, bad_value in failures.items():
        candidate = dict(good)
        candidate[field] = bad_value
        assert not pair_metrics_are_reliable(**candidate), field


def test_reference_bank_roundtrip_and_tamper_detection():
    bank = _bank()
    arrays = bank.to_arrays()
    restored = ReferenceBank.from_arrays(arrays)
    assert restored.directed_pair_metadata("A", "B") == bank.directed_pair_metadata(
        "A", "B"
    )
    assert arrays["pair_probe_fit_balanced_accuracies"].dtype == np.float32
    np.testing.assert_array_equal(
        restored.pair_calibration_class_source_counts,
        np.asarray([16, 16], dtype=np.int32),
    )
    excluded_metadata = restored.directed_pair_metadata(
        "A", "B", query_source_key="heldout-a-0"
    )
    assert excluded_metadata["bank_reliable"]
    assert not excluded_metadata["reliable"]
    assert excluded_metadata["candidate_source_excluded"]
    assert excluded_metadata["candidate_source_membership_roles"] == [
        "current_class"
    ]

    digest_tamper = {name: value.copy() for name, value in arrays.items()}
    digest_tamper["pair_probe_fit_eval_split_digests"][0, 1] = "c" * 64
    with pytest.raises(ValueError, match="pair_metadata_invalid"):
        ReferenceBank.from_arrays(digest_tamper)

    count_tamper = {name: value.copy() for name, value in arrays.items()}
    count_tamper["pair_current_patch_counts"][0, 1] = 9
    with pytest.raises(ValueError, match="pair_metadata_invalid"):
        ReferenceBank.from_arrays(count_tamper)

    threshold_tamper = {name: value.copy() for name, value in arrays.items()}
    threshold_tamper["pair_current_negative_thresholds"][0, 1] = 0.20
    with pytest.raises(ValueError, match="pair_metadata_invalid"):
        ReferenceBank.from_arrays(threshold_tamper)

    source_tamper = {name: value.copy() for name, value in arrays.items()}
    source_tamper["pair_calibration_class_source_ids"][0, 0] = "f" * 16
    with pytest.raises(ValueError, match="pair_source_provenance_invalid"):
        ReferenceBank.from_arrays(source_tamper)


def test_missing_pair_calibration_never_synthesizes_reliability():
    bank = _bank(include_pair=False)
    assert not bank.directed_pair_is_reliable("A", "B")
    metadata = bank.directed_pair_metadata("A", "B")
    assert metadata["probe_fit_status"] == "not_fitted"
    assert metadata["fit_current_source_count"] == 0


def _calibrated_pair_bank(
    *,
    corrupt_eval: bool,
    bank: ReferenceBank | None = None,
) -> ReferenceBank:
    bank = _bank(include_pair=False) if bank is None else bank
    rows = [(_source_fingerprint(f"heldout-a-{i}"), 0) for i in range(16)]
    rows += [(_source_fingerprint(f"heldout-b-{i}"), 1) for i in range(16)]
    assignments, fold_count, _digest = _stable_source_group_folds(
        [source for source, _label in rows],
        np.asarray([label for _source, label in rows], dtype=np.int8),
        maximum_folds=2,
    )
    assert fold_count == 2
    eval_sources = {
        source for (source, _label), role in zip(rows, assignments) if role == 0
    }
    builder = ExactTwoViewPairCalibrationBuilder(bank, RefinementConfig())
    fractional_mask = np.asarray(
        [0.01, 0.49] + [1.0] * 14, dtype=np.float32
    )
    for class_name, prefix, clean_token in (
        ("A", "heldout-a", E_A),
        ("B", "heldout-b", E_B),
    ):
        for index in range(16):
            source_key = f"{prefix}-{index}"
            source_id = _source_fingerprint(source_key)
            token = clean_token
            if corrupt_eval and source_id in eval_sources:
                token = E_B if class_name == "A" else E_A
            tokens = np.repeat(token[None, :], 16, axis=0)
            assert builder.add_example(
                class_name=class_name,
                source_key=source_key,
                anchor_id=f"anchor-{index}",
                token_views=[tokens, tokens],
                target_masks=[fractional_mask, fractional_mask],
            )
    return builder.finalize()


def test_exact_pair_calibration_fits_all_thresholds_without_eval_leakage():
    clean = _calibrated_pair_bank(corrupt_eval=False)
    corrupted_eval = _calibrated_pair_bank(corrupt_eval=True)
    clean_meta = clean.directed_pair_metadata("A", "B")
    corrupt_meta = corrupted_eval.directed_pair_metadata("A", "B")
    assert clean_meta["fit_current_source_count"] == 8
    assert clean_meta["fit_alternative_source_count"] == 8
    assert clean_meta["eval_current_source_count"] == 8
    assert clean_meta["eval_alternative_source_count"] == 8
    np.testing.assert_array_equal(
        clean.pair_probe_weights[0, 1], corrupted_eval.pair_probe_weights[0, 1]
    )
    for threshold_name in (
        "probe_threshold",
        "current_negative_threshold",
        "current_presence_threshold",
        "current_strong_threshold",
        "alternative_negative_threshold",
        "alternative_presence_threshold",
        "alternative_strong_threshold",
    ):
        assert clean_meta[threshold_name] == pytest.approx(
            corrupt_meta[threshold_name]
        ), threshold_name
    assert clean_meta["probe_oof_auroc"] > corrupt_meta["probe_oof_auroc"]
    assert clean_meta["reliable"]
    assert not corrupt_meta["reliable"]
    assert not clean.directed_pair_is_reliable(
        "A", "B", query_source_key="heldout-a-0"
    )
    assert not clean.directed_pair_is_reliable(
        "A", "B", query_source_key="heldout-b-0"
    )


def test_same_source_pair_calibration_blocks_only_positive_confirmation():
    bank = _calibrated_pair_bank(corrupt_eval=False)
    alternative_tokens = np.repeat(E_B[None, :], 16, axis=0)

    independent = _score(
        alternative_tokens,
        bank=bank,
        query_source_key="candidate-source-not-in-calibration",
    )
    same_source = _score(
        alternative_tokens,
        bank=bank,
        query_source_key="heldout-a-0",
    )

    assert independent["status"] == STATUS_CONFIRMED_OUTLIER
    assert independent["directed_pair_reliable"]
    assert not independent["directed_pair_candidate_source_excluded"]
    assert same_source["status"] == STATUS_UNRESOLVED
    assert same_source["intrinsic_references_reliable"]
    assert same_source["directed_pair_bank_reliable"]
    assert not same_source["directed_pair_reliable"]
    assert same_source["directed_pair_candidate_source_excluded"]
    assert same_source[
        "directed_pair_candidate_source_membership_roles"
    ] == ["current_class"]
    assert same_source["support_threshold_source"] == "intrinsic_fallback"
    assert (
        "directed_pair_candidate_source_in_calibration"
        in same_source["reason_codes"]
    )

    current_tokens = np.repeat(E_A[None, :], 16, axis=0)
    explained = _score(
        current_tokens,
        bank=bank,
        query_source_key="heldout-a-0",
    )
    assert explained["status"] == STATUS_EXPLAINED_NOT_OUTLIER
    assert explained["support_threshold_source"] == "intrinsic_fallback"


def test_exact_recalibration_cannot_retain_stale_pair_fit_state():
    calibrated = _calibrated_pair_bank(corrupt_eval=False)
    assert calibrated.directed_pair_is_reliable("A", "B")

    recalibrated = ExactTwoViewPairCalibrationBuilder(
        calibrated, RefinementConfig()
    ).finalize()
    metadata = recalibrated.directed_pair_metadata("A", "B")

    assert not metadata["reliable"]
    assert metadata["probe_fit_status"] == "insufficient_sources"
    assert metadata["current_presence_threshold"] == pytest.approx(
        recalibrated.class_support_threshold("A")
    )
    assert metadata["alternative_presence_threshold"] == pytest.approx(
        recalibrated.class_support_threshold("B")
    )


def test_exact_pair_thresholds_do_not_reuse_intrinsic_heldout_thresholds():
    base = _bank(include_pair=False)
    low_intrinsic = replace(
        base,
        support_thresholds=np.asarray([-1.50, -1.50], dtype=np.float32),
        strong_support_thresholds=np.asarray([-1.40, -1.40], dtype=np.float32),
        negative_support_thresholds=np.asarray([-1.60, -1.60], dtype=np.float32),
    )
    high_intrinsic = replace(
        base,
        support_thresholds=np.asarray([1.50, 1.50], dtype=np.float32),
        strong_support_thresholds=np.asarray([1.60, 1.60], dtype=np.float32),
        negative_support_thresholds=np.asarray([1.40, 1.40], dtype=np.float32),
    )
    low_intrinsic.validate()
    high_intrinsic.validate()
    low = _calibrated_pair_bank(corrupt_eval=False, bank=low_intrinsic)
    high = _calibrated_pair_bank(corrupt_eval=False, bank=high_intrinsic)
    low_meta = low.directed_pair_metadata("A", "B")
    high_meta = high.directed_pair_metadata("A", "B")
    for threshold_name in (
        "probe_threshold",
        "current_negative_threshold",
        "current_presence_threshold",
        "current_strong_threshold",
        "alternative_negative_threshold",
        "alternative_presence_threshold",
        "alternative_strong_threshold",
    ):
        assert low_meta[threshold_name] == pytest.approx(
            high_meta[threshold_name]
        ), threshold_name


def test_exact_pair_thresholds_match_each_deployed_view_statistic():
    bank = _bank(include_pair=False)
    builder = ExactTwoViewPairCalibrationBuilder(bank, RefinementConfig())
    for class_name, prefix in (("A", "heldout-a"), ("B", "heldout-b")):
        for index in range(16):
            source_id = _source_fingerprint(f"{prefix}-{index}")
            if class_name == "A":
                supports = np.asarray(
                    [[0.80, -0.50], [-0.70, 0.10]], dtype=np.float32
                )
                pair_features = np.asarray([1.0, -1.0], dtype=np.float32)
            else:
                supports = np.asarray(
                    [[-0.80, -0.40], [0.90, 0.50]], dtype=np.float32
                )
                pair_features = np.asarray([-1.0, 1.0], dtype=np.float32)
            pair_matrix = np.zeros((2, 2, 2), dtype=np.float32)
            pair_matrix[0, 1] = pair_features
            pair_matrix[1, 0] = pair_features[::-1]
            builder._rows[class_name][source_id] = {
                "rank": f"{index:04d}",
                "supports": supports,
                "pair_features": pair_matrix,
            }
    calibrated = builder.finalize()
    metadata = calibrated.directed_pair_metadata("A", "B")
    assert metadata["current_presence_threshold"] == pytest.approx(0.0)
    assert metadata["current_negative_threshold"] == pytest.approx(-0.40)
    assert metadata["current_strong_threshold"] == pytest.approx(0.80)
    assert metadata["alternative_presence_threshold"] == pytest.approx(0.20)
    assert metadata["alternative_negative_threshold"] == pytest.approx(0.10)
    assert metadata["alternative_strong_threshold"] == pytest.approx(0.50)
    assert metadata["current_negative_threshold"] < metadata[
        "current_presence_threshold"
    ] <= metadata["current_strong_threshold"]
    assert metadata["alternative_negative_threshold"] < metadata[
        "alternative_presence_threshold"
    ] <= metadata["alternative_strong_threshold"]


def test_unreliable_pair_thresholds_cannot_change_intrinsic_nonconfirm_decision():
    intrinsic_only = _bank(include_pair=False)
    unreliable_pair = _bank(pair_reliable=False)
    pathological_presence = np.full((2, 2), 1.70, dtype=np.float32)
    unreliable_pair = replace(
        unreliable_pair,
        pair_current_negative_thresholds=np.full(
            (2, 2), -1.90, dtype=np.float32
        ),
        pair_current_presence_thresholds=pathological_presence.copy(),
        pair_current_strong_thresholds=np.full(
            (2, 2), 1.80, dtype=np.float32
        ),
        pair_alternative_negative_thresholds=np.full(
            (2, 2), -1.90, dtype=np.float32
        ),
        pair_alternative_presence_thresholds=pathological_presence.copy(),
        pair_alternative_strong_thresholds=np.full(
            (2, 2), 1.80, dtype=np.float32
        ),
    )
    unreliable_pair.validate()
    current_tokens = np.repeat(E_A[None, :], 16, axis=0)

    baseline = _score(current_tokens, bank=intrinsic_only)
    scored = _score(current_tokens, bank=unreliable_pair)

    assert baseline["status"] == STATUS_EXPLAINED_NOT_OUTLIER
    assert scored["status"] == baseline["status"]
    assert scored["support_threshold_source"] == "intrinsic_fallback"
    assert scored["current_support_threshold"] == pytest.approx(
        unreliable_pair.class_support_threshold("A")
    )
    assert scored["alternative_support_threshold"] == pytest.approx(
        unreliable_pair.class_support_threshold("B")
    )


def test_streamed_bank_exact_thresholds_remain_strict_after_float32_roundtrip():
    config = RefinementConfig(
        patches_per_anchor=1,
        patch_reservoir_per_class=128,
        prototypes_per_class=8,
    )
    class_tokens = {
        "A": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "B": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    }
    builder = StreamingReferenceBankBuilder(config)
    for class_name, background_name in (("A", "B"), ("B", "A")):
        for source_index in range(24):
            builder.add(
                class_name=class_name,
                source_key=f"{class_name}-streamed-{source_index}",
                patch_tokens=class_tokens[class_name][None, :],
                valid_mask=np.ones(1, dtype=np.float32),
                background_tokens=class_tokens[background_name][None, :],
                background_valid_mask=np.ones(1, dtype=bool),
            )
    bank = builder.finalize()
    exact = ExactTwoViewPairCalibrationBuilder(bank, config)
    for class_name in ("A", "B"):
        for source_index in range(24):
            source_key = f"{class_name}-streamed-{source_index}"
            if bank.calibration_source_role(source_key) != "heldout":
                continue
            source_id = _source_fingerprint(source_key)
            # These fit statistics deliberately tie at the same float32 value.
            # The persisted negative threshold must move by one float32 ULP,
            # rather than by a float64 epsilon that disappears in the cache.
            pair_features = np.zeros((2, 2, 2), dtype=np.float32)
            pair_features[0, 1] = (
                np.asarray([1.0, -1.0], dtype=np.float32)
                if class_name == "A"
                else np.asarray([-1.0, 1.0], dtype=np.float32)
            )
            pair_features[1, 0] = pair_features[0, 1][::-1]
            exact._rows[class_name][source_id] = {
                "rank": f"{source_index:04d}",
                "supports": np.full((2, 2), 0.25, dtype=np.float32),
                "pair_features": pair_features,
            }

    calibrated = exact.finalize()
    restored = ReferenceBank.from_arrays(calibrated.to_arrays())
    for current_name, alternative_name in (("A", "B"), ("B", "A")):
        metadata = restored.directed_pair_metadata(
            current_name, alternative_name
        )
        assert np.float32(metadata["current_negative_threshold"]) < np.float32(
            metadata["current_presence_threshold"]
        )
        assert np.float32(
            metadata["alternative_negative_threshold"]
        ) < np.float32(metadata["alternative_presence_threshold"])


def test_no_overlap_keep_requires_current_exclusive_correspondence():
    current = np.repeat(E_A[None, :], 16, axis=0)
    alias = np.repeat(ALIAS[None, :], 16, axis=0)
    assert _score(current)["status"] == STATUS_EXPLAINED_NOT_OUTLIER
    alias_result = _score(alias)
    assert alias_result["status"] == STATUS_UNRESOLVED
    assert not alias_result["decision_gates"][
        "current_exclusive_component_corresponds"
    ]


def test_mixed_requires_corresponding_separated_exclusive_components():
    separated = np.repeat(NEUTRAL[None, :], 16, axis=0)
    separated[:8] = E_A
    separated[8:] = E_B
    result = _score(separated)
    assert result["status"] == STATUS_MIXED_OR_COMPOSITE
    assert result["decision_gates"]["exclusive_components_spatially_separated"]

    alias = np.repeat(ALIAS[None, :], 16, axis=0)
    alias_result = _score(alias, overlaps=([10.0, 0.0, 30.0, 40.0],))
    assert alias_result["status"] == STATUS_UNRESOLVED
    assert not alias_result["decision_gates"]["proved_overlap_decomposition"]


def test_cross_view_component_mismatch_cannot_confirm():
    tight = np.repeat(NEUTRAL[None, :], 16, axis=0)
    context = tight.copy()
    tight[:8] = E_B
    context[8:] = E_B
    result = _score(tight, context)
    assert result["status"] == STATUS_UNRESOLVED
    assert not result["decision_gates"][
        "alternative_exclusive_component_corresponds"
    ]


def test_components_use_eight_neighbors_and_per_axis_correspondence():
    heat = np.zeros(16, dtype=np.float32)
    heat[[0, 5, 10, 15]] = 1.0
    component = _largest_component_geometry(
        heat,
        np.ones(16, dtype=np.float32),
        crop_xyxy=[0.0, 0.0, 40.0, 40.0],
        target_bbox=[0.0, 0.0, 40.0, 40.0],
        grid_shape=(4, 4),
        threshold=0.5,
    )
    assert component["cell_count"] == 4
    assert component["mass_fraction"] == pytest.approx(0.25)

    tight = {
        "cell_count": 4,
        "mass_fraction": 0.25,
        "source_centroid": [50.0, 2.0],
    }
    context = {
        "cell_count": 4,
        "mass_fraction": 0.25,
        "source_centroid": [50.0, 7.0],
    }
    assert not _component_corresponds(
        tight,
        context,
        [0.0, 0.0, 100.0, 10.0],
        RefinementConfig(),
    )


def test_nested_containment_explains_only_strong_corresponding_current_evidence():
    strong = np.repeat(E_B[None, :], 16, axis=0)
    strong[:4] = E_A
    overlap = ([-5.0, -5.0, 45.0, 45.0],)
    strong_result = _score(strong, overlaps=overlap)
    assert strong_result["status"] == STATUS_EXPLAINED_NOT_OUTLIER
    assert strong_result["decision_gates"]["nested_overlap"]
    assert strong_result["decision_gates"]["proved_overlap_decomposition"]

    weak_current = np.asarray([0.15, 0.0, np.sqrt(1.0 - 0.15**2)], dtype=np.float32)
    weak = np.repeat(E_B[None, :], 16, axis=0)
    weak[:4] = weak_current
    weak_result = _score(weak, overlaps=overlap)
    assert weak_result["status"] == STATUS_UNRESOLVED
    assert not weak_result["decision_gates"]["current_strong"]


def test_partial_overlap_decomposition_requires_current_evidence_outside_overlap():
    overlap = ([20.0, 0.0, 50.0, 40.0],)
    proved = np.repeat(E_B[None, :], 16, axis=0)
    proved.reshape(4, 4, 3)[:, :2] = E_A
    proved_result = _score(proved, overlaps=overlap)
    assert proved_result["decision_gates"]["proved_overlap_decomposition"]
    assert proved_result["decision_gates"][
        "current_outside_overlap_exclusive_component_corresponds"
    ]

    inside_only = np.repeat(NEUTRAL[None, :], 16, axis=0)
    shaped = inside_only.reshape(4, 4, 3)
    shaped[:2, 2:] = E_A
    shaped[2:, 2:] = E_B
    inside_result = _score(inside_only, overlaps=overlap)
    assert not inside_result["decision_gates"]["proved_overlap_decomposition"]
    assert not inside_result["decision_gates"][
        "current_outside_overlap_exclusive_component_corresponds"
    ]


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        ((0.0, 0.0, 17.0, 16.0), STATUS_UNRESOLVED),
        ((0.0, 0.0, 18.0, 18.0), STATUS_CONFIRMED_OUTLIER),
    ],
)
def test_confirmation_source_resolution_boundary(bbox, expected):
    tokens = np.repeat(E_B[None, :], 16, axis=0)
    result = _score(
        tokens,
        target_bbox=bbox,
        crop_boxes=[list(bbox), list(bbox)],
    )
    assert result["status"] == expected


def test_confirmation_resolution_uses_visible_clipped_target_not_raw_bbox():
    tokens = np.repeat(E_B[None, :], 16, axis=0)
    result = _score(
        tokens,
        target_bbox=(-32.0, 0.0, 8.0, 40.0),
        crop_boxes=[[0.0, 0.0, 40.0, 40.0]] * 2,
    )
    assert result["target_bbox_width"] == 40.0
    assert result["visible_target_bbox_width"] == 8.0
    assert result["status"] == STATUS_UNRESOLVED
    assert not result["decision_gates"]["source_resolution_sufficient"]


def test_source_resolution_gates_explained_and_mixed_terminal_statuses():
    all_current = np.repeat(E_A[None, :], 16, axis=0)
    explained = _score(
        all_current,
        target_bbox=(0.0, 0.0, 8.0, 8.0),
        crop_boxes=[[0.0, 0.0, 8.0, 8.0]] * 2,
    )
    assert explained["status"] == STATUS_UNRESOLVED
    assert "source_resolution_insufficient_for_explanation" in explained[
        "reason_codes"
    ]

    mixed_tokens = np.repeat(E_A[None, :], 16, axis=0)
    mixed_tokens[8:] = E_B
    mixed = _score(
        mixed_tokens,
        target_bbox=(0.0, 0.0, 8.0, 8.0),
        crop_boxes=[[0.0, 0.0, 8.0, 8.0]] * 2,
    )
    assert mixed["status"] == STATUS_UNRESOLVED
    assert "source_resolution_insufficient_for_mixed_composite" in mixed[
        "reason_codes"
    ]


def test_confirmation_invariants_are_central_and_unresolved_helper_cannot_forge():
    tokens = np.repeat(E_B[None, :], 16, axis=0)
    result = _score(tokens)
    assert result["status"] == STATUS_CONFIRMED_OUTLIER
    assert result["support_threshold_source"] == "fit_only_directed_pair"
    assert confirmation_invariants_hold(result)
    for gate in refinement._CONFIRMATION_REQUIRED_GATES:
        tampered = deepcopy(result)
        tampered["decision_gates"][gate] = False
        assert not confirmation_invariants_hold(tampered), gate
    malformed = deepcopy(result)
    malformed["directed_pair_probe_eval_current_source_count"] = "not-a-count"
    assert not confirmation_invariants_hold(malformed)
    for malformed_count in (8.5, True, np.float64(8.0)):
        malformed = deepcopy(result)
        malformed["directed_pair_probe_eval_current_source_count"] = (
            malformed_count
        )
        assert not confirmation_invariants_hold(malformed)
    for malformed_digest in ("A" * 64, "g" * 64, 7):
        malformed = deepcopy(result)
        malformed["directed_pair_probe_fold_digest"] = malformed_digest
        malformed["directed_pair_probe_fit_eval_split_digest"] = (
            malformed_digest
        )
        assert not confirmation_invariants_hold(malformed)
    for field_name, replacement in (
        ("directed_pair_probe_features", None),
        ("directed_pair_probe_weights", [1.0, 0.0]),
        ("directed_pair_probe_feature_names", 7),
        ("directed_pair_probe_score", None),
        ("directed_pair_probe_score", -999.0),
        (
            "directed_pair_probe_score",
            result["directed_pair_probe_score"] + 0.25,
        ),
        (
            "directed_pair_current_exclusive_support",
            result["directed_pair_current_exclusive_support"] + 0.25,
        ),
        ("current_negative_threshold", -999.0),
        ("alternative_strong_threshold", 999.0),
        ("current_view_support_scores", [1.0, 1.0]),
        ("alternative_view_support_scores", [0.0, 0.0]),
        ("visible_target_bbox_area", 1.0),
        ("directed_pair_heldout_auroc", 0.1),
        ("directed_pair_reliable", "true"),
        ("support_threshold_source", "intrinsic_fallback"),
    ):
        malformed = deepcopy(result)
        malformed[field_name] = replacement
        assert not confirmation_invariants_hold(malformed), field_name
    for forged_status in (
        STATUS_CONFIRMED_OUTLIER,
        STATUS_EXPLAINED_NOT_OUTLIER,
        STATUS_MIXED_OR_COMPOSITE,
    ):
        forged = unresolved_evidence(
            current_class="A",
            alternative_class="B",
            reason="failure",
            status=forged_status,
        )
        assert forged["status"] == STATUS_UNRESOLVED


def test_property_every_confirmed_result_satisfies_all_invariants():
    rng = np.random.default_rng(42)
    vocabulary = np.stack([E_A, E_B, NEUTRAL, ALIAS], axis=0)
    for _index in range(100):
        tight = vocabulary[rng.integers(0, len(vocabulary), size=16)]
        context = vocabulary[rng.integers(0, len(vocabulary), size=16)]
        result = _score(tight, context)
        if result["status"] == STATUS_CONFIRMED_OUTLIER:
            assert confirmation_invariants_hold(result)
            assert all(
                result["decision_gates"][gate]
                for gate in refinement._CONFIRMATION_REQUIRED_GATES
            )
