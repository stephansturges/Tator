import asyncio
import copy
import hashlib
import inspect
import json
import math
import os
import re
import time
import types
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw
from starlette.datastructures import UploadFile

import localinferenceapi as api
from services.classifier import _load_clip_head_from_classifier_impl
from tools import clip_training
from utils.embedding_recipe import normalize_embedding_aggregation
from utils.cradio_embedding import (
    CRADIO_DEFAULT_MODEL,
    CRadioBackendStatus,
    _unpack_cradio_outputs,
    cradio_backend_status,
    encode_cradio_images,
    normalize_cradio_pooling,
)
from utils import cradio_embedding as cradio_embedding_utils
from utils.local_salad import LocalSALADConfig, LocalSALADHead, symmetric_infonce_loss
from utils.local_salad_mlx import (
    MLXLocalSALADHead,
    encode_local_salad_mlx,
    local_salad_mlx_available,
    make_mlx_local_salad_optimizer,
    mlx_local_salad_state_dict,
    mlx_local_salad_train_step,
)


def _record(point_id: str, class_name: str) -> dict:
    return {
        "point_id": point_id,
        "class_name": class_name,
        "image_relpath": f"{point_id}.jpg",
        "split": "train",
        "bbox_xyxy": [0, 0, 10, 10],
    }


def _class_analysis_test_npy_bytes(value) -> bytes:
    handle = BytesIO()
    np.save(handle, np.asarray(value), allow_pickle=False)
    return handle.getvalue()


def _class_analysis_test_npy_header(*, shape, dtype) -> bytes:
    handle = BytesIO()
    np.lib.format.write_array_header_1_0(
        handle,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
            "fortran_order": False,
            "shape": tuple(shape),
        },
    )
    return handle.getvalue()


def _class_analysis_test_write_npz_members(path, members) -> None:
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for name, encoded in members.items():
            archive.writestr(f"{name}.npy", encoded)


def test_class_analysis_v3_anchor_selection_prefers_stage1_strong_positives():
    def anchor_record(point_id, *, source, bbox):
        return {
            **_record(point_id, "Bike"),
            "_image_sha256": source,
            "width": int(bbox[2] - bbox[0]),
            "height": int(bbox[3] - bbox[1]),
            "_image_width": 1000,
            "_image_height": 1000,
            "bbox_xyxy": bbox,
        }

    records = [
        anchor_record("strong-small", source="11" * 32, bbox=[0, 0, 30, 30]),
        anchor_record("strong-large", source="22" * 32, bbox=[0, 0, 180, 120]),
        anchor_record("missing-stage1", source="33" * 32, bbox=[0, 0, 80, 40]),
        anchor_record("low-trust", source="44" * 32, bbox=[0, 0, 90, 90]),
    ]
    points = {
        "strong-small": {
            "same_class_neighbor_ratio": 0.96,
            "outlier_score": 0.08,
            "embedding_wrong_class_suspicion": 0.0,
        },
        "strong-large": {
            "same_class_neighbor_ratio": 0.91,
            "outlier_score": 0.18,
            "embedding_wrong_class_suspicion": 0.0,
        },
        "low-trust": {
            "same_class_neighbor_ratio": 0.20,
            "outlier_score": 0.92,
            "embedding_wrong_class_suspicion": 0.70,
            "is_wrong_class_candidate": True,
        },
    }
    diagnostics = {}

    selected = api._class_analysis_refinement_anchor_records(
        records,
        excluded_point_ids=set(),
        config=types.SimpleNamespace(seed=42, anchors_per_class=2),
        result_points_by_id=points,
        diagnostics_out=diagnostics,
    )

    assert {row["point_id"] for row in selected} == {
        "strong-small",
        "strong-large",
    }
    assert diagnostics["schema"] == "class-analysis-strong-positive-anchors-v3"
    bike = diagnostics["per_class"]["Bike"]
    assert bike["selected_tier_counts"] == {"strong_positive": 2}
    assert bike["distinct_source_count"] == 2
    assert set(bike["scale_buckets"]) == {"small", "large"}
    assert bike["used_fallback"] is False


def test_class_analysis_v3_anchor_scale_uses_source_image_dimensions_and_excludes_low_trust_fill():
    records = []
    points = {}
    for index, (point_id, same_ratio, outlier_score) in enumerate(
        [
            ("clean-tiny", 0.96, 0.05),
            ("clean-large", 0.84, 0.45),
            ("low-1", 0.20, 0.95),
            ("low-2", 0.10, 0.99),
        ]
    ):
        bbox = [0, 0, 10, 20] if point_id == "clean-tiny" else [0, 0, 200, 160]
        records.append(
            {
                **_record(point_id, "TransitObject"),
                "bbox_xyxy": bbox,
                # Public width/height are object dimensions, not source image
                # dimensions. The private fields prevent every box from being
                # misclassified as a full-frame/large object.
                "width": bbox[2] - bbox[0],
                "height": bbox[3] - bbox[1],
                "_image_width": 1000,
                "_image_height": 1000,
                "_image_sha256": f"{index + 1:064x}",
            }
        )
        points[point_id] = {
            "same_class_neighbor_ratio": same_ratio,
            "outlier_score": outlier_score,
            "embedding_wrong_class_suspicion": max(0.0, 1.0 - same_ratio),
            "is_wrong_class_candidate": point_id.startswith("low"),
        }
    diagnostics = {}

    selected = api._class_analysis_refinement_anchor_records(
        records,
        excluded_point_ids=set(),
        config=types.SimpleNamespace(seed=42, anchors_per_class=4),
        result_points_by_id=points,
        diagnostics_out=diagnostics,
    )

    assert {row["point_id"] for row in selected} == {
        "clean-tiny",
        "clean-large",
    }
    transit_object = diagnostics["per_class"]["TransitObject"]
    assert transit_object["clean_candidate_count"] == 2
    assert transit_object["low_trust_candidate_count"] == 2
    assert transit_object["selected_clean_count"] == 2
    assert transit_object["low_trust_rows_excluded"] == 2
    assert set(transit_object["scale_buckets"]) == {"tiny", "large"}


def test_class_analysis_v3_observability_fails_closed_on_pair_coverage():
    class FakeBank:
        class_names = ["A", "B", "C"]
        anchor_counts = np.asarray([8, 8, 8], dtype=np.int32)

        def class_position(self, name):
            return self.class_names.index(name) if name in self.class_names else None

        def class_is_reliable(self, _name):
            return True

        def class_reliability_tier(self, _name):
            return "usable"

        def class_distinct_sources(self, _name):
            return 8

        def class_negative_support_threshold(self, _name):
            return 0.25

        def directed_pair_metadata(self, current, alternative):
            reliable = (current, alternative) != ("A", "C")
            return {
                "reliable": reliable,
                "tier": "usable" if reliable else "insufficient",
                "dominance_threshold": 0.15,
                "current_negative_threshold": 0.05,
                "current_source_count": 8,
                "alternative_source_count": 8,
                "current_patch_count": 64,
                "alternative_patch_count": 64,
                "alternative_passing_source_fraction": 0.75 if reliable else 0.25,
                "heldout_auroc": 0.91 if reliable else 0.55,
            }

        def directed_pair_is_reliable(self, current, alternative):
            return (current, alternative) != ("A", "C")

        def pair_calibration_provenance(self):
            return {"schema": "directed-pair-calibration-v1"}

    candidates = [
        {
            "class_name": "A",
            "suggested_neighbor_class": "B",
            "refined_outlier": {
                "status": api.CLASS_ANALYSIS_REFINEMENT_CONFIRMED_OUTLIER,
                "current_class": "A",
                "alternative_class": "B",
                "directed_pair_reliable": True,
                "decision_gates": {
                    "directed_pair_reliable": True,
                    "directed_pair_candidate_source_independent": True,
                    "directed_pair_exact_calibration_contracts": True,
                    "directed_pair_dominates": True,
                    "intrinsic_references_reliable": True,
                    "positive_confirmation_pair_reliable": True,
                    "positive_confirmation_pair_probe_auroc_sufficient": True,
                    "positive_confirmation_pair_probe_lower_bound_sufficient": True,
                    "source_resolution_sufficient": True,
                    "alternative_strong": True,
                    "current_absent": True,
                    "alternative_exclusive_component_corresponds": True,
                    "view_consistent": True,
                    "alternative_evidence_external_to_overlap": True,
                },
                "reason_codes": ["directed_pair_margin_passed"],
            },
        },
        {
            "class_name": "A",
            "suggested_neighbor_class": "C",
            "refined_outlier": {
                "status": api.CLASS_ANALYSIS_REFINEMENT_UNRESOLVED,
                "current_class": "A",
                "alternative_class": "C",
                "directed_pair_reliable": False,
                "decision_gates": {
                    "directed_pair_reliable": False,
                    "directed_pair_candidate_source_independent": True,
                    "directed_pair_exact_calibration_contracts": False,
                    "directed_pair_dominates": True,
                    "intrinsic_references_reliable": True,
                    "positive_confirmation_pair_reliable": False,
                    "positive_confirmation_pair_probe_auroc_sufficient": False,
                    "positive_confirmation_pair_probe_lower_bound_sufficient": False,
                    "source_resolution_sufficient": True,
                    "alternative_strong": True,
                    "current_absent": True,
                    "alternative_exclusive_component_corresponds": True,
                    "view_consistent": True,
                    "alternative_evidence_external_to_overlap": True,
                },
                "reason_codes": ["directed_pair_unreliable"],
            },
        },
        {
            "class_name": "B",
            "suggested_neighbor_class": "A",
            "refined_outlier": {
                "status": api.CLASS_ANALYSIS_REFINEMENT_EXPLAINED_NOT_OUTLIER,
                "current_class": "B",
                "alternative_class": "A",
                "directed_pair_reliable": True,
                "decision_gates": {
                    "directed_pair_reliable": True,
                    "directed_pair_candidate_source_independent": True,
                    "directed_pair_exact_calibration_contracts": True,
                    "directed_pair_dominates": True,
                    "intrinsic_references_reliable": True,
                    "positive_confirmation_pair_reliable": True,
                    "positive_confirmation_pair_probe_auroc_sufficient": True,
                    "positive_confirmation_pair_probe_lower_bound_sufficient": True,
                    "source_resolution_sufficient": True,
                    "alternative_strong": False,
                    "current_absent": False,
                    "alternative_exclusive_component_corresponds": False,
                    "view_consistent": True,
                    "alternative_evidence_external_to_overlap": False,
                },
                "reason_codes": ["overlap_localized"],
            },
        },
        {
            "class_name": "A",
            "suggested_neighbor_class": "",
            "refined_outlier": {
                "status": api.CLASS_ANALYSIS_REFINEMENT_UNRESOLVED,
                "current_class": "A",
                "alternative_class": "",
                "directed_pair_reliable": False,
                "decision_gates": {
                    "directed_pair_reliable": False,
                    "directed_pair_candidate_source_independent": True,
                    "directed_pair_exact_calibration_contracts": False,
                    "directed_pair_dominates": False,
                    "intrinsic_references_reliable": False,
                    "positive_confirmation_pair_reliable": False,
                    "positive_confirmation_pair_probe_auroc_sufficient": False,
                    "positive_confirmation_pair_probe_lower_bound_sufficient": False,
                    "source_resolution_sufficient": False,
                    "alternative_strong": False,
                    "current_absent": False,
                    "alternative_exclusive_component_corresponds": False,
                    "view_consistent": False,
                    "alternative_evidence_external_to_overlap": False,
                },
                "reason_codes": ["alternative_class_missing"],
            },
        },
    ]

    summary = api._class_analysis_refinement_v3_observability(
        refinement_candidates=candidates,
        bank=FakeBank(),
        anchor_selection={"schema": "anchors-v3"},
    )

    assert summary["quality_status"] == "completed_non_actionable"
    assert summary["observability_contract"] == (
        "class-analysis-refinement-observability-v3.3"
    )
    assert summary["triage_semantics"] == (
        "class-analysis-terminal-triage-v3.3"
    )
    assert summary["quality_gate"]["metrics"]["resolved_rate"] == pytest.approx(0.5)
    assert summary["quality_gate"]["metrics"]["terminal_decisive_rate"] == pytest.approx(0.5)
    assert summary["quality_gate"]["metrics"]["unresolved_rate"] == pytest.approx(0.5)
    assert summary["pair_coverage"]["candidate_weighted_coverage"] == pytest.approx(0.5)
    assert summary["pair_coverage"]["candidate_count"] == 4
    assert summary["pair_coverage"]["paired_candidate_count"] == 3
    assert summary["pair_coverage"]["missing_pair_candidate_count"] == 1
    assert summary["quality_gate"]["reasons"] == [
        "confirmation_eligible_pair_coverage_below_release_gate"
    ]
    assert summary["queue_policy"] == {
        "mode": "selector_ranked_complete_stage1",
        "automatic_rough_fallback": False,
        "fallback_reason": "",
        "default_queue": "selector_ranked_stage1_candidates",
        "effective_default_candidate_count": 4,
        "confirmed_count": 1,
        "pair_conflict_count": 0,
        "refined_review_candidate_count": 1,
        "rough_count": 4,
    }
    assert summary["gate_counts"]["boolean_gates"]["directed_pair_reliable"] == {
        "passed": 2,
        "failed": 2,
    }
    assert summary["gate_counts"]["boolean_gates"]["directed_pair_dominates"] == {
        "passed": 3,
        "failed": 1,
    }
    pair_row = next(
        row
        for row in summary["pair_coverage"]["pairs"]
        if row["current_class"] == "A" and row["alternative_class"] == "B"
    )
    assert pair_row["dominance_threshold"] == 0.15
    assert pair_row["current_source_count"] == 8
    assert pair_row["alternative_source_count"] == 8
    assert pair_row["current_patch_count"] == 64
    assert pair_row["alternative_patch_count"] == 64


def test_class_analysis_v3_compact_vlm_rail_preserves_pair_calibration_gates():
    point = {
        "point_id": "candidate-1",
        "class_name": "Bike",
        "suggested_neighbor_class": "Person",
        "refined_outlier": {
            "schema": api.CLASS_ANALYSIS_REFINEMENT_SCHEMA,
            "decision_contract": (
                api.CLASS_ANALYSIS_REFINEMENT_DECISION_CONTRACT
            ),
            "status": api.CLASS_ANALYSIS_REFINEMENT_UNRESOLVED,
            "reason_codes": ["spatial_evidence_not_decisive"],
            "current_class": "Bike",
            "alternative_class": "Person",
            "current_support_score": 0.22,
            "alternative_support_score": 0.37,
            "intrinsic_current_support": 0.22,
            "intrinsic_alternative_support": 0.37,
            "directed_pair_margin": 0.15,
            "directed_pair_raw_margin": 0.15,
            "directed_pair_probe_score": 0.16,
            "directed_pair_probe_features": [0.4, 0.5],
            "directed_pair_probe_feature_names": [
                "current_patch_exclusive_support",
                "alternative_patch_exclusive_support",
            ],
            "directed_pair_current_exclusive_support": 0.4,
            "directed_pair_alternative_exclusive_support": 0.5,
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
            "directed_pair_probe_fold_digest": "ab" * 32,
            "directed_pair_probe_fit_eval_split_digest": "ab" * 32,
            "directed_pair_threshold": 0.18,
            "current_negative_threshold": 0.08,
            "current_support_threshold": 0.14,
            "current_strong_threshold": 0.22,
            "alternative_negative_threshold": 0.09,
            "alternative_support_threshold": 0.14,
            "alternative_strong_threshold": 0.25,
            "support_threshold_source": "fit_only_directed_pair",
            "directed_pair_tier": "usable",
            "directed_pair_reliable": True,
            "directed_pair_bank_reliable": True,
            "directed_pair_candidate_source_excluded": False,
            "directed_pair_candidate_source_fingerprint": "34" * 8,
            "directed_pair_candidate_source_membership_roles": [],
            "directed_pair_heldout_auroc": 0.83,
            "directed_pair_eval_auroc_lower_bound": 0.68,
            "positive_confirmation_pair_probe_auroc_floor": 0.8,
            "positive_confirmation_pair_probe_auroc_lower_bound_floor": 0.6,
            "directed_pair_probe_fit_current_source_count": 12,
            "directed_pair_probe_fit_alternative_source_count": 13,
            "directed_pair_probe_eval_current_source_count": 9,
            "directed_pair_probe_eval_alternative_source_count": 10,
            "directed_pair_probe_fit_balanced_accuracy": 0.74,
            "directed_pair_probe_eval_sensitivity": 0.71,
            "directed_pair_probe_eval_specificity": 0.72,
            "directed_pair_current_absence_eval_fraction": 0.67,
            "directed_pair_alternative_strong_eval_fraction": 0.69,
            "directed_pair_current_source_count": 21,
            "directed_pair_alternative_source_count": 29,
            "directed_pair_current_patch_count": 144,
            "directed_pair_alternative_patch_count": 201,
            "directed_pair_alternative_passing_source_fraction": 0.79,
            "current_reference_tier": "high",
            "alternative_reference_tier": "high",
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
            "decision_gates": {
                "diagnostic_pair_reliable": True,
                "directed_pair_reliable": True,
                "directed_pair_candidate_source_independent": True,
                "directed_pair_exact_calibration_contracts": True,
                "intrinsic_references_reliable": True,
                "positive_confirmation_pair_reliable": True,
                "positive_confirmation_pair_probe_auroc_sufficient": True,
                "positive_confirmation_pair_probe_lower_bound_sufficient": True,
                "source_resolution_sufficient": True,
                "directed_pair_dominates": False,
                "current_absent": False,
                "alternative_strong": True,
                "alternative_exclusive_component_corresponds": True,
                "view_consistent": True,
                "alternative_evidence_external_to_overlap": True,
                "qualified_for_human_review": False,
            },
        },
    }

    rail = api._class_analysis_compact_refinement_rail(point)

    assert rail is not None
    assert rail["rail_version"] == "patch_refinement_advisory_v7"
    assert rail["schema"] == "class-analysis-patch-refinement-v5"
    assert rail["decision_contract"] == "class-analysis-patch-decision-v9"
    assert rail["directed_pair_probe_score"] == pytest.approx(0.16)
    assert rail["directed_pair_raw_margin"] == pytest.approx(0.15)
    assert rail["alternative_negative_threshold"] == pytest.approx(0.09)
    assert rail["alternative_support_threshold"] == pytest.approx(0.14)
    assert rail["alternative_strong_threshold"] == pytest.approx(0.25)
    assert rail["support_threshold_source"] == "fit_only_directed_pair"
    assert rail["directed_pair_bank_reliable"] is True
    assert rail["directed_pair_candidate_source_excluded"] is False
    assert rail["directed_pair_candidate_source_fingerprint"] == "34" * 8
    assert rail["directed_pair_candidate_source_membership_roles"] == []
    assert rail["directed_pair_heldout_auroc"] == pytest.approx(0.83)
    assert rail["directed_pair_current_source_count"] == 21
    assert rail["directed_pair_alternative_source_count"] == 29
    assert rail["directed_pair_current_patch_count"] == 144
    assert rail["directed_pair_alternative_patch_count"] == 201
    assert rail[
        "directed_pair_alternative_passing_source_fraction"
    ] == pytest.approx(0.79)
    assert rail["diagnostic_pair_reliable"] is True
    assert rail["positive_confirmation_pair_reliable"] is True
    assert rail["qualified_for_human_review"] is False
    assert rail["human_review_rank"] is None
    assert rail["gate_diagnostics"] == {
        "diagnostic_pair_reliable": True,
        "directed_pair_reliable": True,
        "directed_pair_candidate_source_independent": True,
        "directed_pair_exact_calibration_contracts": True,
        "intrinsic_references_reliable": True,
        "positive_confirmation_pair_reliable": True,
        "positive_confirmation_pair_probe_auroc_sufficient": True,
        "positive_confirmation_pair_probe_lower_bound_sufficient": True,
        "source_resolution_sufficient": True,
        "directed_pair_dominates": False,
        "current_absent": False,
        "alternative_strong": True,
        "alternative_exclusive_component_corresponds": True,
        "view_consistent": True,
        "alternative_evidence_external_to_overlap": True,
        "qualified_for_human_review": False,
    }


def test_class_analysis_parses_bbox_polygon_and_crop_bounds():
    bbox = api._class_analysis_parse_yolo_geometry(
        "1 0.5 0.5 0.2 0.4",
        image_width=100,
        image_height=100,
    )
    assert bbox["kind"] == "bbox"
    assert bbox["class_id"] == 1
    assert bbox["bbox_xyxy"] == [40.0, 30.0, 60.0, 70.0]

    bbox_with_confidence = api._class_analysis_parse_yolo_geometry(
        "1 0.5 0.5 0.2 0.4 0.99",
        image_width=100,
        image_height=100,
    )
    assert bbox_with_confidence["kind"] == "bbox"
    assert bbox_with_confidence["bbox_xyxy"] == [40.0, 30.0, 60.0, 70.0]


@pytest.mark.parametrize(
    "line",
    [
        "0 0.5 0.5 inf 0.2",
        "0 0.5 0.5 0.2 -inf",
        "0 0.0 0.0 nan 0.0 1.0 1.0",
        "0 0.0 0.0 broken 0.0 1.0 1.0",
    ],
)
def test_class_analysis_rejects_nonfinite_or_malformed_geometry(line):
    assert (
        api._class_analysis_parse_yolo_geometry(
            line,
            image_width=100,
            image_height=100,
        )
        is None
    )


def test_class_analysis_exact_reassignment_geometry_fails_closed():
    assert api._class_analysis_parse_yolo_geometry(
        "1.5 0.5 0.5 0.2 0.4",
        image_width=100,
        image_height=100,
    ) is None
    bbox_event = {
        "point": {
            "bbox": {
                "kind": "bbox",
                "xyxy": [40.0, 30.0, 60.0, 70.0],
                "label_line": "0 0.5 0.5 0.2 0.4",
            }
        }
    }
    assert api._class_analysis_resolve_persisted_review_class(
        event=bbox_event,
        label_lines=["1 0.5 0.5 0.2 0.4"],
        labelmap=["Boat", "Building"],
        target_bbox=[40.0, 30.0, 60.0, 70.0],
        image_width=100,
        image_height=100,
        require_unique=True,
    ) == "Building"
    assert api._class_analysis_resolve_persisted_review_class(
        event=bbox_event,
        label_lines=[
            "1 0.5 0.5 0.2 0.4",
            "999 0.5 0.5 0.2 0.4",
        ],
        labelmap=["Boat", "Building"],
        target_bbox=[40.0, 30.0, 60.0, 70.0],
        image_width=100,
        image_height=100,
        require_unique=True,
    ) is None
    assert api._class_analysis_resolve_persisted_review_class(
        event=bbox_event,
        label_lines=["1.5 0.5 0.5 0.2 0.4"],
        labelmap=["Boat", "Building"],
        target_bbox=[40.0, 30.0, 60.0, 70.0],
        image_width=100,
        image_height=100,
        require_unique=True,
    ) is None
    assert api._class_analysis_resolve_persisted_review_class(
        event=bbox_event,
        label_lines=["1 0.5001 0.5 0.2 0.4"],
        labelmap=["Boat", "Building"],
        target_bbox=[40.0, 30.0, 60.0, 70.0],
        image_width=100,
        image_height=100,
        require_unique=True,
    ) is None

    polygon_event = {
        "point": {
            "bbox": {
                "kind": "polygon",
                "xyxy": [10.0, 10.0, 30.0, 30.0],
                "label_line": (
                    "0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3"
                ),
            }
        }
    }
    assert api._class_analysis_resolve_persisted_review_class(
        event=polygon_event,
        label_lines=[
            # Same envelope, different polygon interior.
            "1 0.1 0.1 0.3 0.1 0.3 0.3 0.2 0.2 0.1 0.3"
        ],
        labelmap=["Boat", "Building"],
        target_bbox=[10.0, 10.0, 30.0, 30.0],
        image_width=100,
        image_height=100,
        require_unique=True,
    ) is None


def test_class_analysis_review_object_key_is_order_independent_but_content_scoped():
    base = {
        "source_key": "active:cas_123",
        "image_sha256": "ab" * 32,
        "split": "train",
        "image_relpath": "nested/frame.jpg",
        "class_name": "Boat",
        "image_width": 100,
        "image_height": 100,
    }
    geometry = {
        "kind": "polygon",
        "bbox_xyxy": [10.0, 10.0, 30.0, 30.0],
        "points": [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]],
    }
    rotated = {
        **geometry,
        "points": [[30.0, 30.0], [10.0, 30.0], [10.0, 10.0], [30.0, 10.0]],
    }
    reversed_ring = {**geometry, "points": list(reversed(geometry["points"]))}

    key = api._class_analysis_review_object_key(**base, geometry=geometry)

    assert key == api._class_analysis_review_object_key(**base, geometry=rotated)
    assert key == api._class_analysis_review_object_key(
        **base,
        geometry=reversed_ring,
    )
    assert key == api._class_analysis_review_object_key(
        **{**base, "source_key": "active:cas_changed_by_unrelated_image"},
        geometry=geometry,
    )
    assert key != api._class_analysis_review_object_key(
        **{**base, "class_name": "Building"},
        geometry=geometry,
    )
    assert key != api._class_analysis_review_object_key(
        **{**base, "image_sha256": "cd" * 32},
        geometry=geometry,
    )
    assert key != api._class_analysis_review_object_key(
        **base,
        geometry={**geometry, "points": [[11.0, 10.0], *geometry["points"][1:]]},
    )


def test_class_analysis_review_object_key_survives_yolo_six_decimal_round_trip():
    base = {
        "source_key": "linked:dataset",
        "image_sha256": "ef" * 32,
        "split": "train",
        "image_relpath": "frame.jpg",
        "class_name": "Boat",
        "image_width": 10_000,
        "image_height": 8_000,
    }
    before_save = api._class_analysis_parse_yolo_geometry(
        "0 0.50000049 0.50000049 0.20000049 0.40000049",
        image_width=base["image_width"],
        image_height=base["image_height"],
    )
    after_save = api._class_analysis_parse_yolo_geometry(
        "0 0.5 0.5 0.2 0.4",
        image_width=base["image_width"],
        image_height=base["image_height"],
    )
    meaningfully_moved = api._class_analysis_parse_yolo_geometry(
        "0 0.5001 0.5 0.2 0.4",
        image_width=base["image_width"],
        image_height=base["image_height"],
    )

    assert api._class_analysis_review_object_key(
        **base,
        geometry=before_save,
    ) == api._class_analysis_review_object_key(
        **base,
        geometry=after_save,
    )
    assert api._class_analysis_review_object_key(
        **base,
        geometry=after_save,
    ) != api._class_analysis_review_object_key(
        **base,
        geometry=meaningfully_moved,
    )


def test_local_salad_head_is_trainable_normalized_and_fixed_width():
    gen = torch.Generator(device="cpu")
    gen.manual_seed(123)
    patches = torch.randn(3, 12, 32, generator=gen)
    global_token = torch.randn(3, 32, generator=gen)
    head = LocalSALADHead(
        LocalSALADConfig(
            num_channels=32,
            num_clusters=4,
            cluster_dim=8,
            token_dim=16,
            hidden_dim=24,
            dropout=0.0,
        )
    )

    desc_a = head(patches, global_token=global_token)
    desc_b = head(patches, global_token=global_token)
    mismatched_global = torch.randn(3, 64, generator=gen)
    desc_mismatch = head(patches, global_token=mismatched_global)

    assert desc_a.shape == (3, 48)
    assert desc_mismatch.shape == (3, 48)
    assert torch.allclose(desc_a, desc_b, atol=1e-6)
    assert torch.isfinite(desc_a).all()
    assert torch.isfinite(desc_mismatch).all()
    assert torch.allclose(desc_a.norm(dim=1), torch.ones(3), atol=1e-5)
    cluster_blocks = desc_a[:, 16:].reshape(3, 4, 8).transpose(1, 2)
    assert torch.isfinite(cluster_blocks).all()
    loss = symmetric_infonce_loss(desc_a[:2], desc_b[:2], temperature=0.2)
    assert torch.isfinite(loss)
    assert normalize_embedding_aggregation("salad") == "local_salad"
    assert normalize_embedding_aggregation("local_salad") == "local_salad"
    assert normalize_embedding_aggregation("anything_else") == "pooled"

    polygon = api._class_analysis_parse_yolo_geometry(
        "2 0.1 0.1 0.2 0.1 0.2 0.2",
        image_width=100,
        image_height=100,
    )
    assert polygon["kind"] == "polygon"
    assert polygon["class_id"] == 2
    assert polygon["bbox_xyxy"] == [10.0, 10.0, 20.0, 20.0]

    crop_bounds = api._class_analysis_crop_bounds(
        [40, 30, 60, 70],
        image_width=100,
        image_height=100,
        crop_mode="padded_square",
        padding_ratio=0.1,
    )
    assert crop_bounds == (26, 26, 74, 74)


def test_mlx_local_salad_matches_torch_state_and_trains_one_step():
    if not local_salad_mlx_available():
        pytest.skip("MLX is not available")
    torch.manual_seed(321)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(321)
    config = LocalSALADConfig(
        num_channels=8,
        num_clusters=3,
        cluster_dim=5,
        token_dim=7,
        hidden_dim=11,
        dropout=0.0,
    )
    torch_head = LocalSALADHead(config)
    torch_head.eval()
    mlx_head = MLXLocalSALADHead(config)
    mlx_head.load_torch_state_dict(torch_head.state_dict())
    patches = torch.randn(4, 9, 8, generator=gen)
    global_token = torch.randn(4, 8, generator=gen)

    with torch.no_grad():
        torch_out = torch_head(patches, global_token=global_token).detach().numpy()
    mlx_out = encode_local_salad_mlx(mlx_head, patches, global_token=global_token)

    assert mlx_out.shape == torch_out.shape == (4, 22)
    assert np.max(np.abs(torch_out - mlx_out)) < 1e-3
    assert np.allclose(np.linalg.norm(mlx_out, axis=1), np.ones(4), atol=1e-5)

    optimizer = make_mlx_local_salad_optimizer(learning_rate=1e-4, weight_decay=0.0)
    loss_value = mlx_local_salad_train_step(
        mlx_head,
        optimizer,
        patches,
        global_token,
        patches + 0.01,
        global_token + 0.01,
        temperature=0.2,
    )
    state_dict = mlx_local_salad_state_dict(mlx_head)

    assert np.isfinite(loss_value)
    assert set(torch_head.state_dict()) == set(state_dict)
    assert state_dict["token_features.0.weight"].shape == torch_head.state_dict()["token_features.0.weight"].shape


def test_class_analysis_flags_neighbor_disagreement_only_in_all_classes():
    records = [
        _record("p0", "car"),
        _record("p1", "boat"),
        _record("p2", "boat"),
        _record("p3", "boat"),
        _record("p4", "car"),
        _record("p5", "car"),
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.01, 0.0],
            [1.0, -0.01, 0.0],
            [0.99, 0.02, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.99, 0.0],
        ],
        dtype=np.float32,
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    all_classes = api._class_analysis_build_result(
        records,
        embeddings,
        summary={"analysis_scope": "all_classes"},
        projection="pca",
        projection_neighbor_k=15,
        neighbor_k=3,
        seed=13,
    )
    candidate_ids = {item["point_id"] for item in all_classes["wrong_class_candidates"]}
    assert "p0" in candidate_ids
    p0 = next(point for point in all_classes["points"] if point["point_id"] == "p0")
    assert p0["suggested_neighbor_class"] == "boat"
    assert p0["is_wrong_class_candidate"] is True
    assert records[0]["wrong_class_suspicion"] == p0["wrong_class_suspicion"]
    assert records[0]["wrong_class_review_reason"] == "embedding_outlier"
    assert records[0]["review_signals"] == ["wrong_class"]
    assert records[0]["neighbor_class_counts"] == p0["neighbor_class_counts"]
    assert "class_cluster_id" not in p0
    assert all_classes["class_clusters"] == {}
    assert all_classes["summary"]["class_cluster_count"] == 0
    assert all_classes["summary"]["class_cluster_class_count"] == 0

    selected_class = api._class_analysis_build_result(
        records,
        embeddings,
        summary={"analysis_scope": "selected_class"},
        projection="pca",
        projection_neighbor_k=15,
        neighbor_k=3,
        seed=13,
    )
    assert selected_class["wrong_class_candidates"] == []
    assert all(point["is_wrong_class_candidate"] is False for point in selected_class["points"])
    assert all("class_cluster_id" not in point for point in selected_class["points"])
    assert all("neighbor_ids" not in point for point in selected_class["points"])
    assert all("neighbor_distances" not in point for point in selected_class["points"])
    assert all("neighbor_class_counts" not in point for point in selected_class["points"])


def test_class_analysis_marks_dual_bbox_conflict_on_near_identical_cross_class_boxes():
    records = [
        _record("p0", "car"),
        _record("p1", "boat"),
        _record("p2", "boat"),
        _record("p3", "boat"),
        _record("p4", "car"),
        _record("p5", "car"),
    ]
    records[0]["image_relpath"] = "shared.jpg"
    records[1]["image_relpath"] = "shared.jpg"
    records[0]["bbox_xyxy"] = [10, 20, 110, 120]
    records[1]["bbox_xyxy"] = [11, 20, 111, 120]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.01, 0.0],
            [1.0, -0.01, 0.0],
            [0.99, 0.02, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.99, 0.0],
        ],
        dtype=np.float32,
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    result = api._class_analysis_build_result(
        records,
        embeddings,
        summary={"analysis_scope": "all_classes"},
        projection="pca",
        projection_neighbor_k=15,
        neighbor_k=3,
        seed=13,
    )

    p0 = next(point for point in result["points"] if point["point_id"] == "p0")
    conflict = p0["dual_bbox_conflict"]
    assert p0["is_dual_bbox_conflict"] is True
    assert "dual_bbox_conflict" in p0["review_signals"]
    assert conflict["review_mode"] == "dual_bbox_annotation_resolution"
    assert conflict["other_class_name"] == "boat"
    assert conflict["iou"] >= 0.98
    assert conflict["target_bbox_xyxy"] == [10.0, 20.0, 110.0, 120.0]
    assert conflict["other_bbox_xyxy"] == [11.0, 20.0, 111.0, 120.0]
    candidate = next(item for item in result["wrong_class_candidates"] if item["point_id"] == "p0")
    assert candidate["is_dual_bbox_conflict"] is True
    assert candidate["dual_bbox_conflict"]["other_class_name"] == "boat"
    same_class_candidate = next(item for item in result["wrong_class_candidates"] if item["point_id"] == "p1")
    assert same_class_candidate["wrong_class_review_reason"] == "dual_bbox_conflict"
    assert same_class_candidate["embedding_wrong_class_suspicion"] < same_class_candidate["wrong_class_suspicion"]
    assert same_class_candidate["dual_bbox_conflict"]["other_class_name"] == "car"
    assert same_class_candidate["dual_bbox_conflict"][
        "other_bbox_xyxy"
    ] == [10.0, 20.0, 110.0, 120.0]
    assert result["summary"]["dual_bbox_conflict_count"] >= 2


def test_class_analysis_dual_bbox_conflict_rejects_polygon_envelope_matches():
    current = {
        **_record("polygon", "building"),
        "kind": "polygon",
        "image_relpath": "shared.jpg",
        "bbox_xyxy": [10, 20, 110, 120],
    }
    other = {
        **_record("bbox", "boat"),
        "kind": "bbox",
        "image_relpath": "shared.jpg",
        "bbox_xyxy": [10, 20, 110, 120],
    }

    assert api._class_analysis_dual_bbox_conflict(current, other) is None
    assert api._class_analysis_dual_bbox_conflict(other, current) is None


def test_class_analysis_qwen_review_parses_tool_call_payloads():
    payload, error = api._class_analysis_qwen_review_parse_payload(
        '<tool_call>{"name":"inspect_target_context","arguments":{}}</tool_call>'
    )
    assert error is None
    assert payload == {"name": "inspect_target_context", "arguments": {}}

    fenced, fenced_error = api._class_analysis_qwen_review_parse_payload(
        'thinking...\n```json\n{"name":"finalize_review","arguments":{"decision":"skip_uncertain"}}\n```'
    )
    assert fenced_error is None
    assert fenced["name"] == "finalize_review"

    trailing, trailing_error = api._class_analysis_qwen_review_parse_payload(
        '{"name":"inspect_overlap_evidence","arguments":{}}</tool_call> stray prose {"bad":'
    )
    assert trailing_error is None
    assert trailing == {"name": "inspect_overlap_evidence", "arguments": {}}

    multi, multi_error = api._class_analysis_qwen_review_parse_payload(
        '{"name":"inspect_target_context","arguments":{}} '
        '{"name":"inspect_source_overlay","arguments":{}} '
        '{"name":"final_review","arguments":{"decision":"accept_suggested","confidence":0. 65}}'
    )
    assert multi_error is None
    assert multi["name"] == "final_review"
    assert multi["arguments"]["decision"] == "accept_suggested"
    assert multi["arguments"]["confidence"] == pytest.approx(0.65)

    fragment, fragment_error = api._class_analysis_qwen_review_parse_payload(
        '":"{","decision":"skip_uncertain","target_class":"Truck","confidence":0.600}'
    )
    assert fragment_error is None
    assert fragment["decision"] == "skip_uncertain"

    percent, percent_error = api._class_analysis_qwen_review_parse_payload(
        '{%"decision":"skip_uncertain","target_class":"Truck","confidence":0.600}'
    )
    assert percent_error is None
    assert percent["decision"] == "skip_uncertain"


def test_class_analysis_qwen_review_detects_degenerate_final_text():
    assert api._class_analysis_qwen_review_text_is_degenerate("!" * 120)
    assert api._class_analysis_qwen_review_text_is_degenerate("-lfs" * 80)
    assert not api._class_analysis_qwen_review_text_is_degenerate(
        json.dumps(
            {
                "decision": "skip_uncertain",
                "target_class": "CandidateClass",
                "confidence": 0.42,
                "rationale_short": "Target evidence is not clear enough for a class change.",
            }
        )
    )


def test_class_analysis_qwen_review_serializes_recent_model_output_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", tmp_path)
    job = api.ClassAnalysisQwenReviewJob(
        review_id="review_trace",
        parent_job_id="parent_trace",
        point_id="point_1",
    )

    api._class_analysis_qwen_review_append_event(
        job,
        {"type": "model_input", "phase": "final", "messages": [{"image": "large"}]},
    )
    api._class_analysis_qwen_review_append_event(
        job,
        {"type": "model_output", "phase": "final", "text": "complete model reasoning\nfinal json"},
    )

    payload = api._serialize_class_analysis_qwen_review_job(job)

    assert payload["trace_events"] == [
        {
            "timestamp": payload["trace_events"][0]["timestamp"],
            "type": "model_output",
            "phase": "final",
            "text": "complete model reasoning\nfinal json",
        }
    ]


def test_class_analysis_qwen_review_audit_append_failure_is_terminal(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", tmp_path)
    job = api.ClassAnalysisQwenReviewJob(
        review_id="review_audit_failure",
        parent_job_id="parent_audit_failure",
        point_id="point_1",
    )
    monkeypatch.setattr(
        api.os,
        "fsync",
        lambda _fd: (_ for _ in ()).throw(OSError("disk unavailable")),
    )

    with pytest.raises(RuntimeError, match="qwen_review_audit_append_failed"):
        api._class_analysis_qwen_review_append_event(
            job,
            {"type": "model_output", "phase": "final_attempt_1", "text": "raw"},
        )

    assert job.audit_error == "qwen_review_audit_append_failed"
    assert job.trace_events == []


def test_class_analysis_qwen_model_provenance_uses_bounded_checkpoint_identity(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "snapshots" / ("a" * 40)
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
    (checkpoint / "model.safetensors.index.json").write_text(
        '{"weight_map":{"x":"model-00001-of-00001.safetensors"}}',
        encoding="utf-8",
    )
    (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    hashed = []
    original_hash = api._class_analysis_file_sha256_cached

    def record_hash(path):
        hashed.append(Path(path).name)
        return original_hash(path)

    monkeypatch.setattr(api, "_class_analysis_file_sha256_cached", record_hash)
    runtime = api.QwenRuntime(
        model=object(),
        processor=object(),
        platform=api.QWEN_PLATFORM_MLX,
        model_id=str(checkpoint),
    )

    provenance = api._class_analysis_qwen_review_model_provenance(
        "default", runtime
    )

    assert provenance["resolved_model_id"] == str(checkpoint)
    assert provenance["checkpoint_revision"] == "a" * 40
    assert provenance["checkpoint_identity_verified"] is True
    assert provenance["checkpoint_fingerprint_verified"] is False
    assert provenance["fingerprint_strength"] == "revision_config_weight_manifest"
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["checkpoint_fingerprint"])
    assert "config.json" in hashed
    assert "model.safetensors.index.json" in hashed
    assert "model-00001-of-00001.safetensors" not in hashed


def test_class_analysis_qwen_review_final_context_keeps_decision_images_scoped():
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "system"}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tool result for inspect_target_context.\nEvidence ids: target_context_1"},
                {"type": "image", "image": "target.jpg"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tool result for inspect_source_overlay.\nEvidence ids: source_overlay_3"},
                {"type": "image", "image": "source_clean.jpg"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tool result for inspect_class_context_pack.\nEvidence ids: class_context_pack_5"},
                {"type": "image", "image": "class_context.jpg"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tool result for zoom_source_region.\nEvidence ids: zoom_region_6"},
                {"type": "image", "image": "zoom.jpg"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tool result for inspect_local_consensus_context.\nEvidence ids: local_consensus_context_7"},
                {"type": "image", "image": "local_consensus.jpg"},
            ],
        },
    ]

    compacted, policy = api._class_analysis_qwen_review_final_context_messages(messages)
    image_values = [
        item["image"]
        for message in compacted
        for item in (message.get("content") or [])
        if isinstance(item, dict) and item.get("type") == "image"
    ]

    assert policy["input_image_count"] == 5
    assert policy["output_image_count"] == 4
    assert image_values == [
        "target.jpg",
        "zoom.jpg",
        "local_consensus.jpg",
        "source_clean.jpg",
    ]
    assert "inspect_class_context_pack" in policy["text_only_observations"]
    assert "inspect_source_overlay" not in policy["text_only_observations"]
    assert "inspect_source_overlay" in policy["image_observations"]
    assert "inspect_local_consensus_context" not in policy["text_only_observations"]
    assert "inspect_local_consensus_context" in policy["image_observations"]


def test_class_analysis_qwen_review_final_context_keeps_region_contrast_but_not_composite_images():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tool result for inspect_target_detail.\nEvidence ids: target_detail_2"},
                {"type": "image", "image": "target_detail.jpg"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tool result for inspect_source_overlay.\nEvidence ids: source_clean_3, source_overlay_4"},
                {"type": "image", "image": "source_clean.jpg"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tool result for inspect_class_context_pack.\nEvidence ids: class_context_pack_6"},
                {"type": "image", "image": "class_context.jpg"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tool result for inspect_specificity_region_contrast.\nEvidence ids: specificity_region_contrast_7"},
                {"type": "image", "image": "region_contrast.jpg"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tool result for inspect_local_consensus_context.\nEvidence ids: local_consensus_context_11"},
                {"type": "image", "image": "local_consensus.jpg"},
            ],
        },
    ]

    compacted, policy = api._class_analysis_qwen_review_final_context_messages(messages)
    final_text = "\n".join(
        str(item.get("text") or "")
        for message in compacted
        for item in (message.get("content") or [])
        if isinstance(item, dict) and item.get("type") == "text"
    )
    image_values = [
        item["image"]
        for message in compacted
        for item in (message.get("content") or [])
        if isinstance(item, dict) and item.get("type") == "image"
    ]

    assert image_values == [
        "target_detail.jpg",
        "region_contrast.jpg",
        "local_consensus.jpg",
        "source_clean.jpg",
    ]
    assert "class_context_pack_6" in final_text
    assert "local_consensus_context_11" in final_text
    assert "class_context.jpg" not in image_values
    assert "local_consensus.jpg" in image_values
    assert policy["input_image_count"] == 5
    assert policy["output_image_count"] == 4
    assert policy["text_only_observations"] == ["inspect_class_context_pack"]
    assert "inspect_specificity_region_contrast" in policy["image_observations"]


def test_class_analysis_qwen_review_system_prompt_gates_local_consensus_tool():
    default_text = api._class_analysis_qwen_review_system_prompt(3)
    enabled_text = api._class_analysis_qwen_review_system_prompt(3, allow_local_consensus=True)

    assert "inspect_local_consensus_context" not in default_text
    assert "Available tools" not in default_text
    assert "active schema" in default_text
    assert "inspect_local_consensus_context" not in enabled_text
    assert "controller renders one local-consensus context" in enabled_text
    default_router = api._class_analysis_qwen_review_router_tool_spec(allow_local_consensus=False)
    enabled_router = api._class_analysis_qwen_review_router_tool_spec(allow_local_consensus=True)
    assert default_router["parameters"]["properties"]["action"]["enum"] == ["finalize_now"]
    assert "inspect_local_consensus_context" in enabled_router["parameters"]["properties"]["action"]["enum"]


def test_class_analysis_qwen_review_system_prompt_preserves_limited_advisory_changes():
    text = api._class_analysis_qwen_review_system_prompt(3)

    assert "advisory human-triage opinion" in text
    assert "class-changing" in text
    assert "advisory only" in text
    assert "state the VLM" in text
    assert "evidence-based decision" in text
    assert "Class-changing decisions require clear backend quality" not in text
    assert "may only return an advisory confirm_current" not in text


def test_class_analysis_qwen_review_router_policy_allows_limited_local_consensus():
    point = {"class_name": "PoleFixture", "suggested_neighbor_class": "SmallVehicle"}
    limited_quality = {"tier": "limited"}
    payload = {
        "name": "route_review",
        "arguments": {
            "action": "inspect_local_consensus_context",
            "reason_code": "needs_same_image_consensus",
            "confidence": 0.9,
            "rationale_short": "Need dot context.",
        },
    }

    router = api._class_analysis_qwen_review_validate_router(
        payload,
        local_consensus_enabled=True,
        visual_quality=limited_quality,
        point=point,
        executed_tools=set(),
    )

    assert router["action"] == "inspect_local_consensus_context"
    assert router["reason_code"] == "needs_same_image_consensus"
    assert router["policy_allowed_local_consensus"] is True
    assert router["policy_reasons"] == []


def test_class_analysis_qwen_review_router_policy_masks_poor_local_consensus():
    point = {"class_name": "PoleFixture", "suggested_neighbor_class": "SmallVehicle"}
    poor_quality = {"tier": "poor"}
    payload = {
        "name": "route_review",
        "arguments": {
            "action": "inspect_local_consensus_context",
            "reason_code": "needs_same_image_consensus",
            "confidence": 0.9,
            "rationale_short": "Need dot context.",
        },
    }

    router = api._class_analysis_qwen_review_validate_router(
        payload,
        local_consensus_enabled=True,
        visual_quality=poor_quality,
        point=point,
        executed_tools=set(),
    )

    assert router["action"] == "finalize_now"
    assert router["reason_code"] == "policy_blocked"
    assert router["confidence"] <= 0.35
    assert "target_quality_not_reviewable" in router["policy_reasons"]


def test_class_analysis_qwen_review_compact_final_schema_expands_to_full_audit_payload():
    result = {"summary": {"labelmap": ["Truck", "SmallVehicle"]}}
    point = {
        "point_id": "p0",
        "class_name": "Truck",
        "suggested_neighbor_class": "SmallVehicle",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 124.0,
        "bbox_height": 70.0,
        "bbox_min_dim": 70.0,
        "bbox_area": 8680.0,
        "crop_contrast": 63.8,
        "crop_dynamic_range": 197.0,
        "crop_sharpness": 10.4,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    spec = api._class_analysis_qwen_review_final_tool_spec(["Truck", "SmallVehicle"])
    required = set(spec["parameters"]["required"])

    assert "anchor_evidence_current" not in required
    assert "evidence_ids" not in required
    assert {
        "decision",
        "final_class",
        "current_evidence",
        "suggested_evidence",
        "specificity_alignment",
        "target_background_contrast",
        "target_identity_summary",
        "target_identity_uncertainty",
        "target_identity_evidence_ids",
        "whole_target_extent_supported",
        "whole_target_extent_reason",
    } <= required

    expanded = api._class_analysis_qwen_review_expand_compact_final(
        {
            "decision": "accept_suggested",
            "final_class": "SmallVehicle",
            "confidence": 0.88,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "global_context_evidence": "strong",
            "overlap_assessment": "no material overlap",
            "overlap_explains_candidate_similarity": False,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "target_identity_summary": "compact road-vehicle body with open rear bed",
            "target_identity_uncertainty": "low",
            "target_identity_evidence_ids": ["target_context_1", "zoom_region_6"],
            "whole_target_extent_supported": True,
            "whole_target_extent_reason": "The suggested class explains the full target extent.",
            "visible_target_cues": ["compact road-vehicle body", "visible cargo bed"],
            "rationale_short": "clean pickup-like light vehicle",
        },
        point=point,
        evidence_ids={"target_context_1", "zoom_region_6"},
        visual_quality=clear_quality,
        executed_tools={"inspect_target_context", "zoom_source_region"},
        labelmap_glossary='{"Truck":"heavy goods vehicles"}',
        review_guidance="Prefer visible target evidence.",
    )
    final = api._class_analysis_qwen_review_validate_final(
        expanded,
        result,
        point,
        {"target_context_1", "zoom_region_6"},
        clear_quality,
    )

    assert expanded["_expanded_by_controller"] is True
    assert expanded["overlap_assessment"] == "none"
    assert expanded["local_consensus_evidence"] == "not_applicable"
    assert expanded["glossary_or_guidance_used"] is True
    assert final["decision"] == "accept_suggested"
    assert final["target_class"] == "SmallVehicle"
    assert final["evidence_ids"] == ["target_context_1", "zoom_region_6"]


def test_class_analysis_qwen_review_preserves_compact_skip_without_class_name_promotion():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 124.0,
        "bbox_height": 70.0,
        "bbox_min_dim": 70.0,
        "bbox_area": 8680.0,
        "crop_contrast": 63.8,
        "crop_dynamic_range": 197.0,
        "crop_sharpness": 10.4,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    expanded = api._class_analysis_qwen_review_expand_compact_final(
        {
            "decision": "skip_uncertain",
            "target_class": "CurrentClass",
            "confidence": 0.47,
            "visual_quality": "clear",
            "object_visibility": "partial",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "rationale": "target crop shows suggested class is visually better",
        },
        point=point,
        evidence_ids={"target_context_1", "zoom_region_6"},
        visual_quality=clear_quality,
        executed_tools={"inspect_target_context", "zoom_source_region"},
    )
    final = api._class_analysis_qwen_review_validate_final(
        expanded,
        result,
        point,
        {"target_context_1", "zoom_region_6"},
        clear_quality,
    )

    assert expanded["_controller_reconciliation"]["applied"] is False
    assert final["decision"] == "skip_uncertain"
    assert final["target_class"] == "CurrentClass"
    assert final["human_review_needed"] is True
    assert "model object visibility is partial" in final["advisory_reasons"]


def test_class_analysis_qwen_review_blocks_class_change_without_whole_extent_support():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 124.0,
        "bbox_height": 70.0,
        "bbox_min_dim": 70.0,
        "bbox_area": 8680.0,
        "crop_contrast": 63.8,
        "crop_dynamic_range": 197.0,
        "crop_sharpness": 10.4,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.92,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "global_context_evidence": "strong",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "whole_target_extent_supported": False,
            "whole_target_extent_reason": (
                "SuggestedClass explains only a smaller subcomponent, not the attached structure."
            ),
            "visible_target_cues": ["compact front section", "distinct edge line"],
            "supporting_clean_evidence_ids": ["target_context_1", "zoom_region_6"],
            "rationale_short": "Target front section matches SuggestedClass.",
            "counter_evidence": "Large attached structure remains unexplained.",
            "human_review_needed": True,
            "glossary_or_guidance_used": False,
        },
        result,
        point,
        {"target_context_1", "zoom_region_6"},
        clear_quality,
    )

    assert final["decision"] == "skip_uncertain"
    assert final["guarded_recommendation"]["decision"] == "accept_suggested"
    assert any("whole target extent" in reason for reason in final["guardrail_reasons"])


def test_class_analysis_qwen_review_reconciles_self_contradictory_accept_to_confirm_current():
    result = {"summary": {"labelmap": ["Boat", "SmallVehicle"]}}
    point = {
        "point_id": "p0",
        "class_name": "Boat",
        "suggested_neighbor_class": "SmallVehicle",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 180.0,
        "bbox_height": 90.0,
        "bbox_min_dim": 90.0,
        "bbox_area": 16200.0,
        "crop_contrast": 60.0,
        "crop_dynamic_range": 190.0,
        "crop_sharpness": 20.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    expanded = api._class_analysis_qwen_review_expand_compact_final(
        {
            "decision": "accept_suggested",
            "target_class": "Boat",
            "confidence": 0.85,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "rationale_short": (
                "Target crop clearly shows a boat with a cabin. "
                "The current class is weak due to partial overlap, but target pixels are unambiguous."
            ),
        },
        point=point,
        evidence_ids={"target_context_1", "zoom_region_6"},
        visual_quality=clear_quality,
        executed_tools={"inspect_target_context", "zoom_source_region"},
    )
    final = api._class_analysis_qwen_review_validate_final(
        expanded,
        result,
        point,
        {"target_context_1", "zoom_region_6"},
        clear_quality,
    )

    assert expanded["_controller_reconciliation"]["applied"] is True
    assert expanded["_controller_reconciliation"]["from_decision"] == "accept_suggested"
    assert expanded["decision"] == "confirm_current"
    assert expanded["current_evidence"] == "strong"
    assert expanded["suggested_evidence"] == "weak"
    assert final["decision"] == "confirm_current"
    assert final["target_class"] == "Boat"
    assert final["confidence"] <= 0.72
    assert final["human_review_needed"] is True


def test_class_analysis_qwen_review_does_not_confirm_when_self_contradictory_accept_rejects_current():
    point = {
        "point_id": "p0",
        "class_name": "PoleFixture",
        "suggested_neighbor_class": "SmallVehicle",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 160.0,
        "bbox_height": 90.0,
        "bbox_min_dim": 90.0,
        "bbox_area": 14400.0,
        "crop_contrast": 60.0,
        "crop_dynamic_range": 190.0,
        "crop_sharpness": 20.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    expanded = api._class_analysis_qwen_review_expand_compact_final(
        {
            "decision": "accept_suggested",
            "target_class": "PoleFixture",
            "confidence": 0.85,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "rationale_short": (
                "Target crop clearly shows a car fitting SmallVehicle. "
                "The current PoleFixture label is likely a misclassification."
            ),
        },
        point=point,
        evidence_ids={"target_context_1", "zoom_region_6"},
        visual_quality=clear_quality,
        executed_tools={"inspect_target_context", "zoom_source_region"},
    )

    assert expanded["_controller_reconciliation"]["applied"] is False
    assert expanded["decision"] == "accept_suggested"
    assert expanded["target_class"] == "SmallVehicle"


def test_class_analysis_qwen_review_does_not_reconcile_non_adjacent_skip():
    result = {"summary": {"labelmap": ["SolarArray", "SmallVehicle"]}}
    point = {
        "point_id": "p0",
        "class_name": "SolarArray",
        "suggested_neighbor_class": "SmallVehicle",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 64.0,
        "bbox_height": 64.0,
        "bbox_min_dim": 64.0,
        "bbox_area": 4096.0,
        "crop_contrast": 63.8,
        "crop_dynamic_range": 197.0,
        "crop_sharpness": 10.4,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    expanded = api._class_analysis_qwen_review_expand_compact_final(
        {
            "decision": "skip_uncertain",
            "confidence": 0.73,
            "visual_quality": "clear",
            "object_visibility": "visible",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "rationale": "suggested class is visually better",
        },
        point=point,
        evidence_ids={"target_context_1"},
        visual_quality=clear_quality,
        executed_tools={"inspect_target_context"},
    )
    final = api._class_analysis_qwen_review_validate_final(
        expanded,
        result,
        point,
        {"target_context_1"},
        clear_quality,
    )

    assert expanded["_controller_reconciliation"]["applied"] is False
    assert final["decision"] == "skip_uncertain"


def test_class_analysis_qwen_review_compact_uncertain_class_alias_maps_to_suggested():
    result = {"summary": {"labelmap": ["PoleFixture", "SolarArray"]}}
    point = {
        "point_id": "p0",
        "class_name": "PoleFixture",
        "suggested_neighbor_class": "SolarArray",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 64.0,
        "bbox_height": 64.0,
        "bbox_min_dim": 64.0,
        "bbox_area": 4096.0,
        "crop_contrast": 63.8,
        "crop_dynamic_range": 197.0,
        "crop_sharpness": 10.4,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    expanded = api._class_analysis_qwen_review_expand_compact_final(
        {
            "decision": "change_to_other",
            "uncertain_class": "SolarArray",
            "confidence": 0.86,
            "visual_quality": "clear",
            "visual_visibility": "visible",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "global_context_evidence": "strong",
            "overlap_assessment": "clear",
            "overlap_explains_candidate": False,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "target_identity_summary": "rectangular gridded flat panel surface",
            "target_identity_uncertainty": "low",
            "target_identity_evidence_ids": ["target_context_1"],
            "whole_target_extent_supported": True,
            "whole_target_extent_reason": "The suggested class explains the full target extent.",
            "visible_target_cues": ["rectangular panel surface", "grid-like panel texture"],
            "rationale": "target shows a solar panel",
        },
        point=point,
        evidence_ids={"target_context_1"},
        visual_quality=clear_quality,
        executed_tools={"inspect_target_context"},
    )
    final = api._class_analysis_qwen_review_validate_final(
        expanded,
        result,
        point,
        {"target_context_1"},
        clear_quality,
    )

    assert expanded["decision"] == "accept_suggested"
    assert expanded["overlap_assessment"] == "none"
    assert final["decision"] == "accept_suggested"
    assert final["target_class"] == "SolarArray"


def test_class_analysis_qwen_review_final_validation_is_non_mutating_and_labelmap_guarded():
    result = {"summary": {"labelmap": ["car", "boat", "building"]}}
    point = {
        "point_id": "p0",
        "class_name": "building",
        "suggested_neighbor_class": "boat",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 80.0,
        "bbox_height": 70.0,
        "bbox_min_dim": 70.0,
        "bbox_area": 5600.0,
        "crop_contrast": 42.0,
        "crop_dynamic_range": 120.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "boat",
            "confidence": 0.87,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "moderate",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1", "missing"],
            "visible_target_cues": ["hull-like outline", "open deck shape"],
            "rationale_short": "looks like a boat",
            "human_review_needed": False,
        },
        result,
        point,
        {"ctx_1"},
        clear_quality,
    )
    assert final["decision"] == "accept_suggested"
    assert final["target_class"] == "boat"
    assert final["confidence"] == pytest.approx(0.87)
    assert final["evidence_ids"] == ["ctx_1"]
    assert final["applied"] is False

    with pytest.raises(ValueError, match="labelmap"):
        api._class_analysis_qwen_review_validate_final(
            {
                "decision": "change_to_other",
                "target_class": "airplane",
                "confidence": 1.0,
                "visual_quality": "clear",
                "object_visibility": "clear",
                "current_evidence": "weak",
                "suggested_evidence": "weak",
                "target_evidence": "strong",
                "overlap_assessment": "none",
                "overlap_explains_candidate_similarity": False,
                "anchor_evidence_current": "weak",
                "anchor_evidence_suggested": "weak",
                "local_context_evidence": "weak",
                "local_consensus_evidence": "mixed",
                "global_context_evidence": "weak",
                "glossary_or_guidance_used": False,
            },
            result,
            point,
            set(),
            clear_quality,
        )


def test_class_analysis_qwen_review_quality_gate_forces_uncertain_skip():
    result = {"summary": {"labelmap": ["car", "boat", "building"]}}
    point = {
        "point_id": "p0",
        "class_name": "building",
        "suggested_neighbor_class": "boat",
    }
    poor_quality = {
        "tier": "poor",
        "bbox_width": 13.0,
        "bbox_height": 12.0,
        "bbox_min_dim": 12.0,
        "bbox_area": 156.0,
        "crop_contrast": 73.0,
        "crop_dynamic_range": 220.0,
        "crop_sharpness": 24.0,
        "edge_clipped": False,
        "reasons": ["bbox area is 156px^2"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "confirm_current",
            "target_class": "building",
            "confidence": 0.91,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "strong",
            "suggested_evidence": "weak",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "strong",
            "anchor_evidence_suggested": "weak",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": False,
            "evidence_ids": ["ctx_1"],
            "rationale_short": "model claims the current class is obvious",
            "human_review_needed": False,
        },
        result,
        point,
        {"ctx_1"},
        poor_quality,
    )

    assert final["decision"] == "skip_uncertain"
    assert final["target_class"] == "building"
    assert final["confidence"] <= 0.25
    assert final["human_review_needed"] is True
    assert "backend visual-quality tier is poor" in final["guardrail_reasons"]
    assert final["applied"] is False


def test_class_analysis_qwen_review_quality_gate_caps_direct_high_confidence_skip():
    result = {"summary": {"labelmap": ["Person", "Bike"]}}
    point = {
        "point_id": "p0",
        "class_name": "Person",
        "suggested_neighbor_class": "Bike",
    }
    poor_quality = {
        "tier": "poor",
        "bbox_width": 12.0,
        "bbox_height": 9.0,
        "bbox_min_dim": 9.0,
        "bbox_area": 108.0,
        "crop_contrast": 21.0,
        "crop_dynamic_range": 60.0,
        "crop_sharpness": 3.0,
        "edge_clipped": False,
        "reasons": ["bbox is tiny and blurry"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "skip_uncertain",
            "target_class": "Person",
            "confidence": 0.8,
            "visual_quality": "poor",
            "object_visibility": "partial",
            "current_evidence": "strong",
            "suggested_evidence": "weak",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "strong",
            "anchor_evidence_suggested": "weak",
            "local_context_evidence": "moderate",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "moderate",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": "too small to relabel safely",
            "human_review_needed": True,
        },
        result,
        point,
        {"ctx_1"},
        poor_quality,
    )

    assert final["decision"] == "skip_uncertain"
    assert final["confidence"] <= 0.25
    assert final["human_review_needed"] is True
    assert "backend visual-quality tier is poor" in final["guardrail_reasons"]
    assert "model visual-quality self-check is poor" in final["guardrail_reasons"]


def test_class_analysis_qwen_review_blocks_class_change_on_material_overlap():
    result = {"summary": {"labelmap": ["Truck", "SmallVehicle"]}}
    point = {
        "point_id": "p0",
        "class_name": "Truck",
        "suggested_neighbor_class": "SmallVehicle",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SmallVehicle",
            "confidence": 0.9,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "moderate",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": True,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "moderate",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": "Class comparison indicates SmallVehicle is a better fit.",
            "counter_evidence": "",
            "human_review_needed": False,
        },
        result,
        point,
        {"ctx_1"},
        clear_quality,
    )

    assert final["decision"] == "skip_uncertain"
    assert final["target_class"] == "Truck"
    assert final["confidence"] <= 0.45
    assert final["human_review_needed"] is True
    assert "overlap assessment partial_contamination is too entangled for relabel recommendation" in final["guardrail_reasons"]
    assert any("overlap decomposition" in reason for reason in final["advisory_reasons"])


def test_class_analysis_qwen_review_allows_verifier_backed_partial_overlap_rebuttal():
    result = {"summary": {"labelmap": ["Truck", "Building"]}}
    point = {
        "point_id": "p0",
        "class_name": "Truck",
        "suggested_neighbor_class": "Building",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    evidence_ids = {"target_context_1", "target_detail_2", "source_clean_3", "zoom_region_9"}
    evidence_ledger = {
        "clean_visual_evidence_ids": sorted(evidence_ids),
        "clean_target_source_evidence_ids": sorted(evidence_ids),
        "rows": [
            {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
            {"evidence_id": "target_detail_2", "kind": "target_detail", "use": "clean_visual"},
            {"evidence_id": "source_clean_3", "kind": "source_clean", "use": "clean_visual"},
            {"evidence_id": "zoom_region_9", "kind": "zoom_region", "use": "clean_visual"},
        ],
        "overlap_decomposition": {
            "overlaps": [
                {
                    "point_id": "p1",
                    "class_name": "OtherClass",
                    "relation": "partial_contamination",
                    "target_area_covered": 0.18,
                    "other_area_covered": 0.22,
                    "iou": 0.08,
                }
            ]
        },
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "Building",
            "confidence": 0.88,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "overlap_adjudication_verified": True,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "questions_current",
            "same_image_embedding_evidence": "questions_current",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "glossary_or_guidance_used": True,
            "evidence_ids": sorted(evidence_ids),
            "visible_target_cues": ["fixed rectangular roof", "corrugated roof texture"],
            "supporting_clean_evidence_ids": ["target_detail_2", "source_clean_3"],
            "rationale_short": (
                "Target pixels show fixed rectangular roof and corrugated texture; "
                "overlap does not explain the target-contained building features."
            ),
            "counter_evidence": "Truck anchors are only a moderate match.",
            "human_review_needed": True,
        },
        result,
        point,
        evidence_ids,
        clear_quality,
        evidence_ledger,
    )

    assert final["decision"] == "accept_suggested"
    assert final["target_class"] == "Building"
    assert final["overlap_adjudication_verified"] is True
    assert final["guardrail_reasons"] == []
    assert any("moderate suggested-anchor" in reason for reason in final["advisory_reasons"])
    assert any("partial overlap present" in reason for reason in final["advisory_reasons"])


def test_class_analysis_qwen_review_verified_overlap_path_does_not_depend_on_rebuttal_regex():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    evidence_ids = {"target_context_1", "target_detail_2", "source_clean_3", "zoom_region_9"}
    evidence_ledger = {
        "clean_visual_evidence_ids": sorted(evidence_ids),
        "clean_target_source_evidence_ids": sorted(evidence_ids),
        "rows": [
            {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
            {"evidence_id": "target_detail_2", "kind": "target_detail", "use": "clean_visual"},
            {"evidence_id": "source_clean_3", "kind": "source_clean", "use": "clean_visual"},
            {"evidence_id": "zoom_region_9", "kind": "zoom_region", "use": "clean_visual"},
        ],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.88,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "overlap_adjudication_verified": True,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "neutral",
            "same_image_embedding_evidence": "questions_current",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "glossary_or_guidance_used": False,
            "evidence_ids": sorted(evidence_ids),
            "visible_target_cues": ["spiral conduit ridges", "triangular bracket lattice"],
            "supporting_clean_evidence_ids": ["target_detail_2", "source_clean_3"],
            "rationale_short": "Verifier isolated target-specific visible features in the clean crop.",
            "counter_evidence": "Current-class anchors are weak.",
            "human_review_needed": True,
        },
        result,
        point,
        evidence_ids,
        clear_quality,
        evidence_ledger,
    )

    assert final["decision"] == "accept_suggested"
    assert final["target_class"] == "SuggestedClass"
    assert final["overlap_adjudication_verified"] is True
    assert final["guardrail_reasons"] == []


def test_class_analysis_qwen_review_overlap_guarded_suggestion_runs_cue_verifier():
    final_result = {
        "decision": "skip_uncertain",
        "guarded_recommendation": {
            "blocked": True,
            "decision": "accept_suggested",
            "current_class": "Truck",
            "suggested_neighbor_class": "Building",
            "target_class": "Building",
            "backend_tier": "clear",
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "visible_target_cues": ["fixed rectangular roof", "corrugated roof texture"],
            "guardrail_reasons": [
                "accept_suggested requires strong suggested-anchor agreement, got moderate",
                "overlap assessment partial_contamination is too entangled for relabel recommendation",
            ],
        },
    }

    assert api._class_analysis_qwen_review_should_run_cue_verifier(final_result) is True

    payload, error = api._class_analysis_qwen_review_parse_cue_verifier_payload(
        json.dumps(
            {
                "verified": True,
                "target_class": "Building",
                "cue_confidence": 0.91,
                "positive_visible_target_cues": ["fixed rectangular roof", "corrugated roof texture"],
                "current_class_positive_cues": [],
                "current_class_plausible": False,
                "current_class_plausibility_reason": "No truck-valid shape or parts are visible in the clean target pixels.",
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "The proposed class explains the entire clean target extent.",
                "overlap_rebutted": True,
                "overlap_risk": "target_specific",
                "overlap_rebuttal": "Overlap does not explain the roof texture inside the target.",
                "supporting_clean_evidence_ids": ["target_detail_2", "source_clean_3"],
                "rejection_reason": "",
            }
        ),
        current_class="Truck",
        target_class="Building",
        evidence_ids={"target_detail_2", "source_clean_3"},
    )

    assert error is None
    assert payload["verified"] is True
    assert payload["overlap_rebutted"] is True
    assert payload["overlap_risk"] == "target_specific"

    reconciled_payload, error = api._class_analysis_qwen_review_parse_cue_verifier_payload(
        json.dumps(
            {
                "verified": True,
                "target_class": "Building",
                "cue_confidence": 0.92,
                "positive_visible_target_cues": ["rectangular footprint", "corrugated roof texture"],
                "current_class_positive_cues": [],
                "current_class_plausible": False,
                "current_class_plausibility_reason": "Clean target pixels show a fixed rectangular roof, not a truck-valid body.",
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "The proposed class explains the full rectangular target extent.",
                "overlap_rebutted": True,
                "overlap_risk": "overlap_explains",
                "overlap_rebuttal": (
                    "The rectangular footprint and corrugated roof texture are intrinsic "
                    "to the target object's geometry, not merely artifacts of the partial overlap."
                ),
                "supporting_clean_evidence_ids": ["target_detail_2", "source_clean_3"],
                "rejection_reason": "",
            }
        ),
        current_class="Truck",
        target_class="Building",
        evidence_ids={"target_detail_2", "source_clean_3"},
    )

    assert error is None
    assert reconciled_payload["verified"] is True
    assert reconciled_payload["overlap_risk"] == "target_specific"
    assert reconciled_payload["overlap_risk_reconciled"] is True


def test_class_analysis_qwen_review_visible_cues_are_domain_generic():
    cues = api._class_analysis_qwen_review_normalize_visible_cues(
        [
            "matches class",
            "visible target",
            "not a target object",
            "overhead scene context",
            "dark specular highlight",
            "accordion folded fabric boundary",
            "spiral translucent membrane pattern",
            "hexagonal clasp geometry",
        ],
        current_class="SourceLabel",
        suggested_class="CandidateLabel",
        target_class="CandidateLabel",
    )

    assert cues == [
        "accordion folded fabric boundary",
        "spiral translucent membrane pattern",
        "hexagonal clasp geometry",
    ]
    source = inspect.getsource(api._class_analysis_qwen_review_normalize_visible_cues)
    assert "concrete_visual_tokens" not in source
    for benchmark_term in ("wheel", "roof", "pole", "panel", "cab", "hull", "cargo"):
        assert re.search(rf"\b{re.escape(benchmark_term)}\b", source) is None


def test_class_analysis_qwen_review_moderate_anchor_requires_current_plausibility_verifier():
    result = {"summary": {"labelmap": ["Truck", "Building"]}}
    point = {
        "point_id": "p0",
        "class_name": "Truck",
        "suggested_neighbor_class": "Building",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    evidence_ids = {"target_context_1", "target_detail_2", "source_clean_3", "zoom_region_9"}
    evidence_ledger = {
        "clean_visual_evidence_ids": sorted(evidence_ids),
        "clean_target_source_evidence_ids": sorted(evidence_ids),
        "rows": [
            {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
            {"evidence_id": "target_detail_2", "kind": "target_detail", "use": "clean_visual"},
            {"evidence_id": "source_clean_3", "kind": "source_clean", "use": "clean_visual"},
            {"evidence_id": "zoom_region_9", "kind": "zoom_region", "use": "clean_visual"},
        ],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "Building",
            "confidence": 0.88,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "insufficient",
            "same_image_embedding_evidence": "insufficient",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "glossary_or_guidance_used": True,
            "evidence_ids": sorted(evidence_ids),
            "visible_target_cues": ["rectangular roof", "flat roof surface"],
            "supporting_clean_evidence_ids": ["target_detail_2", "source_clean_3"],
            "rationale_short": "Target looks like a rectangular fixed roof.",
            "counter_evidence": "Truck anchors are only a moderate match.",
            "human_review_needed": True,
        },
        result,
        point,
        evidence_ids,
        clear_quality,
        evidence_ledger,
    )

    assert final["decision"] == "skip_uncertain"
    assert any("current-class plausibility verification" in reason for reason in final["guardrail_reasons"])
    assert api._class_analysis_qwen_review_should_run_cue_verifier(final) is True


def test_class_analysis_qwen_review_cue_verifier_refuses_current_class_plausibility():
    parsed, error = api._class_analysis_qwen_review_parse_cue_verifier_payload(
        json.dumps(
            {
                "verified": True,
                "target_class": "Building",
                "cue_confidence": 0.94,
                "positive_visible_target_cues": ["arched lattice canopy", "riveted panel seam"],
                "current_class_positive_cues": ["long trailer-like rectangular body"],
                "current_class_plausibility_basis": "direct_positive_cues",
                "current_class_plausible": True,
                "current_class_plausibility_reason": (
                    "The clean target still plausibly fits Truck because it is an isolated long "
                    "rectangular trailer-like body."
                ),
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "The proposed class covers the full visible target.",
                "overlap_rebutted": True,
                "overlap_risk": "target_specific",
                "overlap_rebuttal": "Overlap does not explain the arched canopy and panel seam.",
                "supporting_clean_evidence_ids": ["target_detail_2", "source_clean_3"],
                "rejection_reason": "",
            }
        ),
        current_class="Truck",
        target_class="Building",
        evidence_ids={"target_detail_2", "source_clean_3"},
    )

    assert error is None
    assert parsed["verified"] is False
    assert parsed["current_class_plausible"] is True
    assert "trailer-like body" in parsed["rejection_reason"]


def test_class_analysis_qwen_review_cue_verifier_reconciles_hypothetical_plausibility():
    parsed, error = api._class_analysis_qwen_review_parse_cue_verifier_payload(
        json.dumps(
            {
                "verified": True,
                "target_class": "SuggestedClass",
                "cue_confidence": 0.91,
                "positive_visible_target_cues": [
                    "spiral translucent membrane pattern",
                    "hexagonal clasp geometry",
                ],
                "current_class_positive_cues": [],
                "current_class_plausibility_basis": "hypothetical_or_uncertain",
                "current_class_plausible": True,
                "current_class_plausibility_reason": (
                    "The current class is only imaginable as an edge case, with no direct "
                    "current-class pixels visible."
                ),
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "The proposed class explains the full clean target extent.",
                "overlap_rebutted": True,
                "overlap_risk": "target_specific",
                "overlap_rebuttal": "Overlap does not explain the membrane and clasp features.",
                "supporting_clean_evidence_ids": ["target_detail_2", "source_clean_3"],
                "rejection_reason": "",
            }
        ),
        current_class="CurrentClass",
        target_class="SuggestedClass",
        evidence_ids={"target_detail_2", "source_clean_3"},
    )

    assert error is None
    assert parsed["verified"] is True
    assert parsed["raw_current_class_plausible"] is True
    assert parsed["current_class_plausible"] is False
    assert parsed["current_class_plausibility_basis"] == "hypothetical_or_uncertain"


def test_class_analysis_qwen_review_cue_verifier_reconciles_overlap_risk_contradiction():
    assert api._class_analysis_qwen_review_cue_verifier_text_rebuts_overlap(
        "The partial contamination is accounted for; the target's own pixels clearly display the defining features."
    )

    parsed, error = api._class_analysis_qwen_review_parse_cue_verifier_payload(
        json.dumps(
            {
                "verified": False,
                "target_class": "SuggestedClass",
                "cue_confidence": 0.92,
                "positive_visible_target_cues": [
                    "spiral translucent membrane pattern",
                    "hexagonal clasp geometry",
                ],
                "current_class_positive_cues": [],
                "current_class_plausibility_basis": "none",
                "current_class_plausible": False,
                "current_class_plausibility_reason": "No direct current-class cue is visible.",
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "The proposed class explains the full clean target extent.",
                "overlap_rebutted": True,
                "overlap_risk": "overlap_explains",
                "overlap_rebuttal": (
                    "Overlap does not explain the membrane and clasp features inside the target pixels."
                ),
                "supporting_clean_evidence_ids": ["target_detail_2", "source_clean_3"],
                "rejection_reason": "Overlap risk remained marked as overlap_explains.",
            }
        ),
        current_class="CurrentClass",
        target_class="SuggestedClass",
        evidence_ids={"target_detail_2", "source_clean_3"},
    )

    assert error is None
    assert parsed["verified"] is True
    assert parsed["raw_verified"] is False
    assert parsed["reconciled_to_verified"] is True
    assert parsed["overlap_risk"] == "target_specific"
    assert parsed["overlap_risk_reconciled"] is True


def test_class_analysis_qwen_review_cue_verifier_rejects_shared_target_current_cues():
    parsed, error = api._class_analysis_qwen_review_parse_cue_verifier_payload(
        json.dumps(
            {
                "verified": True,
                "target_class": "SuggestedClass",
                "cue_confidence": 0.95,
                "positive_visible_target_cues": [
                    "ribbed membrane surface",
                    "hexagonal clasp geometry",
                ],
                "current_class_positive_cues": [
                    "ribbed membrane surface",
                    "hexagonal clasp geometry",
                ],
                "current_class_plausibility_basis": "shared_generic_cues",
                "current_class_plausible": False,
                "current_class_plausibility_reason": "Only shared generic cues are visible.",
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "The proposed class explains the full clean target extent.",
                "overlap_rebutted": True,
                "overlap_risk": "target_specific",
                "overlap_rebuttal": "Overlap does not explain the shared surface details.",
                "supporting_clean_evidence_ids": ["target_detail_2", "source_clean_3"],
                "rejection_reason": "",
            }
        ),
        current_class="CurrentClass",
        target_class="SuggestedClass",
        evidence_ids={"target_detail_2", "source_clean_3"},
    )

    assert error is None
    assert parsed["verified"] is False
    assert parsed["shared_current_class_positive_cues"]
    assert "independent positive target cues" in parsed["rejection_reason"]


def test_class_analysis_qwen_review_dual_bbox_mode_allows_resolved_overlap_class_switch():
    result = {"summary": {"labelmap": ["Truck", "SmallVehicle"]}}
    point = {
        "point_id": "p0",
        "class_name": "Truck",
        "suggested_neighbor_class": "SmallVehicle",
        "dual_bbox_conflict": {
            "enabled": True,
            "kind": "near_identical_cross_class_bbox",
            "review_mode": "dual_bbox_class_resolution",
            "point_id": "p0",
            "current_class": "Truck",
            "other_point_id": "p1",
            "other_class_name": "SmallVehicle",
            "class_name": "SmallVehicle",
            "classes": ["Truck", "SmallVehicle"],
            "iou": 0.96,
            "corner_similarity": 0.97,
            "target_area_covered": 0.98,
            "other_area_covered": 0.97,
            "relation": "duplicate_like",
        },
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    evidence_ledger = {
        "clean_visual_evidence_ids": ["target_context_1", "target_detail_2", "zoom_region_8"],
        "clean_target_source_evidence_ids": ["target_context_1", "target_detail_2", "zoom_region_8"],
        "rows": [
            {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
            {"evidence_id": "target_detail_2", "kind": "target_detail", "use": "clean_visual"},
            {"evidence_id": "zoom_region_8", "kind": "zoom_region", "use": "clean_visual"},
        ],
        "overlap_decomposition": {
            "overlaps": [
                {
                    "point_id": "p1",
                    "class_name": "SmallVehicle",
                    "relation": "duplicate_like",
                    "target_area_covered": 0.98,
                    "other_area_covered": 0.97,
                    "iou": 0.96,
                }
            ]
        },
    }
    instruction = api._class_analysis_qwen_review_final_instruction(
        required_tools={"inspect_target_context", "inspect_overlap_decomposition", "zoom_source_region_clean"},
        evidence_ids={"target_context_1", "target_detail_2", "zoom_region_8"},
        point=point,
        visual_quality=clear_quality,
        dual_bbox_conflict=point["dual_bbox_conflict"],
    )
    instruction_text = instruction["content"][0]["text"]
    assert "Dual-bbox conflict mode is active" in instruction_text
    assert "dual_bbox_resolution" in instruction_text

    expanded = api._class_analysis_qwen_review_expand_compact_final(
        {
            "decision": "accept_suggested",
            "final_class": "SmallVehicle",
            "confidence": 0.9,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "duplicate_like",
            "overlap_explains_candidate_similarity": False,
            "dual_bbox_resolution": "overlap_box_class",
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "supports_suggested",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "target_identity_summary": "compact object body with visible windshield",
            "target_identity_uncertainty": "low",
            "target_identity_evidence_ids": ["target_detail_2", "zoom_region_8"],
            "whole_target_extent_supported": True,
            "whole_target_extent_reason": "The overlapping class explains the full target extent.",
            "visible_target_cues": ["compact vehicle body", "visible windshield"],
            "supporting_clean_evidence_ids": ["target_detail_2", "zoom_region_8"],
            "rationale_short": "target pixels match the overlapping SmallVehicle box",
            "counter_evidence": "Truck label is only from the duplicate box metadata.",
            "human_review_needed": False,
        },
        point=point,
        evidence_ids={"target_context_1", "target_detail_2", "zoom_region_8"},
        visual_quality=clear_quality,
        executed_tools={"inspect_target_context", "inspect_target_detail", "zoom_source_region"},
    )
    final = api._class_analysis_qwen_review_validate_final(
        expanded,
        result,
        point,
        {"target_context_1", "target_detail_2", "zoom_region_8"},
        clear_quality,
        evidence_ledger,
    )

    assert expanded["dual_bbox_resolution"] == "overlap_box_class"
    assert final["decision"] == "accept_suggested"
    assert final["target_class"] == "SmallVehicle"
    assert final["dual_bbox_resolution"] == "overlap_box_class"
    assert final["guardrail_reasons"] == []
    disposition = api._class_analysis_qwen_review_disposition(
        {
            **final,
            "current_class": point["class_name"],
            "suggested_neighbor_class": point["suggested_neighbor_class"],
        }
    )
    assert disposition["disposition"] == "dual_bbox_switch_overlap_class"

    dynamic_point = {
        key: value
        for key, value in point.items()
        if key != "dual_bbox_conflict"
    }
    dynamic_expanded = api._class_analysis_qwen_review_expand_compact_final(
        dict(expanded["_compact_model_arguments"]),
        point=dynamic_point,
        evidence_ids={"target_context_1", "target_detail_2", "zoom_region_8"},
        visual_quality=clear_quality,
        executed_tools={"inspect_target_context", "inspect_target_detail", "zoom_source_region"},
        evidence_ledger=evidence_ledger,
    )
    dynamic_final = api._class_analysis_qwen_review_validate_final(
        dynamic_expanded,
        result,
        dynamic_point,
        {"target_context_1", "target_detail_2", "zoom_region_8"},
        clear_quality,
        evidence_ledger,
    )

    assert dynamic_expanded["dual_bbox_resolution"] == "overlap_box_class"
    assert dynamic_expanded["dual_bbox_conflict"]["source"] == "overlap_decomposition"
    assert dynamic_final["decision"] == "accept_suggested"
    assert dynamic_final["dual_bbox_resolution"] == "overlap_box_class"

    inconsistent_compact = api._class_analysis_qwen_review_expand_compact_final(
        {
            "decision": "accept_suggested",
            "final_class": "SmallVehicle",
            "confidence": 0.92,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "high_overlap",
            "overlap_explains_candidate_similarity": True,
            "dual_bbox_resolution": "both_valid_overlapping_objects",
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "target_identity_summary": "compact object body with visible windshield",
            "target_identity_uncertainty": "low",
            "target_identity_evidence_ids": ["target_detail_2", "zoom_region_8"],
            "whole_target_extent_supported": True,
            "whole_target_extent_reason": "The overlapping class explains the full target extent.",
            "visible_target_cues": ["compact vehicle body", "visible windshield"],
            "supporting_clean_evidence_ids": ["target_detail_2", "zoom_region_8"],
            "rationale_short": "target pixels match the near-identical SmallVehicle box, not Truck.",
            "counter_evidence": "Truck label is only from the duplicate box metadata.",
            "human_review_needed": False,
        },
        point=dynamic_point,
        evidence_ids={"target_context_1", "target_detail_2", "zoom_region_8"},
        visual_quality=clear_quality,
        executed_tools={"inspect_target_context", "inspect_target_detail", "zoom_source_region"},
        evidence_ledger=evidence_ledger,
    )
    inconsistent_final = api._class_analysis_qwen_review_validate_final(
        inconsistent_compact,
        result,
        dynamic_point,
        {"target_context_1", "target_detail_2", "zoom_region_8"},
        clear_quality,
        evidence_ledger,
    )

    assert inconsistent_compact["overlap_assessment"] == "duplicate_like"
    assert inconsistent_compact["dual_bbox_resolution"] == "overlap_box_class"
    assert inconsistent_final["decision"] == "accept_suggested"
    assert inconsistent_final["target_class"] == "SmallVehicle"
    assert inconsistent_final["dual_bbox_resolution"] == "overlap_box_class"
    assert "accept_suggested has only moderate suggested-anchor agreement" in inconsistent_final["advisory_reasons"]


def test_class_analysis_qwen_review_allows_verifier_backed_limited_dual_bbox_switch():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
        "dual_bbox_conflict": {
            "enabled": True,
            "kind": "near_identical_cross_class_bbox",
            "review_mode": "dual_bbox_class_resolution",
            "point_id": "p0",
            "current_class": "CurrentClass",
            "other_point_id": "p1",
            "other_class_name": "SuggestedClass",
            "class_name": "SuggestedClass",
            "classes": ["CurrentClass", "SuggestedClass"],
            "iou": 1.0,
            "target_area_covered": 1.0,
            "other_area_covered": 1.0,
            "relation": "duplicate_like",
        },
    }
    limited_quality = {
        "tier": "limited",
        "bbox_width": 42.0,
        "bbox_height": 30.0,
        "bbox_min_dim": 30.0,
        "bbox_area": 1260.0,
        "crop_contrast": 20.0,
        "crop_dynamic_range": 90.0,
        "crop_sharpness": 8.0,
        "edge_clipped": False,
        "reasons": ["small_but_reviewable"],
    }
    evidence_ledger = {
        "clean_visual_evidence_ids": ["target_context_1", "target_detail_2", "zoom_region_8"],
        "clean_target_source_evidence_ids": ["target_context_1", "target_detail_2", "zoom_region_8"],
        "rows": [
            {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
            {"evidence_id": "target_detail_2", "kind": "target_detail", "use": "clean_visual"},
            {"evidence_id": "zoom_region_8", "kind": "zoom_region", "use": "clean_visual"},
        ],
        "specificity_probe": {
            "status": "completed",
            "confidence": 0.9,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "specificity_margin": "suggested_target_favored",
            "best_supported_class": "SuggestedClass",
        },
        "overlap_decomposition": {
            "overlaps": [
                {
                    "point_id": "p1",
                    "class_name": "SuggestedClass",
                    "relation": "duplicate_like",
                    "target_area_covered": 1.0,
                    "other_area_covered": 1.0,
                    "iou": 1.0,
                }
            ]
        },
    }
    expanded = api._class_analysis_qwen_review_expand_compact_final(
        {
            "decision": "accept_suggested",
            "final_class": "SuggestedClass",
            "confidence": 0.88,
            "visual_quality": "limited",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "duplicate_like",
            "overlap_explains_candidate_similarity": False,
            "dual_bbox_resolution": "overlap_box_class",
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "questions_current",
            "same_image_embedding_evidence": "neutral",
            "glossary_or_guidance_used": True,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "target_identity_summary": "target pixels visibly match the overlapping class",
            "target_identity_uncertainty": "low",
            "target_identity_evidence_ids": ["target_context_1", "target_detail_2"],
            "whole_target_extent_supported": True,
            "whole_target_extent_reason": "The overlapping class explains the full target extent.",
            "visible_target_cues": ["target-specific body shape", "target-specific front detail"],
            "supporting_clean_evidence_ids": ["target_context_1", "target_detail_2"],
            "rationale_short": "target pixels match the overlapping class",
            "counter_evidence": "current class cues are missing",
            "human_review_needed": True,
        },
        point=point,
        evidence_ids={"target_context_1", "target_detail_2", "zoom_region_8"},
        visual_quality=limited_quality,
        executed_tools={"inspect_target_context", "inspect_target_detail", "zoom_source_region"},
        evidence_ledger=evidence_ledger,
    )
    expanded.update(
        {
            "anchor_adjudication_verified": True,
            "_anchor_adjudication_verified": True,
            "current_class_plausible": False,
            "current_class_plausibility_reason": "Current-class defining cues are absent.",
            "_cue_verifier_class_change_verified": True,
            "_cue_verifier_confidence": 0.93,
            "_cue_verifier_overlap_rebutted": True,
            "_cue_verifier_overlap_risk": "target_specific",
            "_cue_verifier_edge_clip_recoverable": True,
        }
    )

    final = api._class_analysis_qwen_review_validate_final(
        expanded,
        result,
        point,
        {"target_context_1", "target_detail_2", "zoom_region_8"},
        limited_quality,
        evidence_ledger,
    )

    assert final["decision"] == "accept_suggested"
    assert final["target_class"] == "SuggestedClass"
    assert final["dual_bbox_resolution"] == "overlap_box_class"
    assert final["guardrail_reasons"] == []
    assert any("limited" in reason for reason in final["advisory_reasons"])
    disposition = api._class_analysis_qwen_review_disposition(
        {
            **final,
            "current_class": point["class_name"],
            "suggested_neighbor_class": point["suggested_neighbor_class"],
        }
    )
    assert disposition["disposition"] == "dual_bbox_switch_overlap_class"

    clipped_quality = {**limited_quality, "edge_clipped": True}
    clipped_final = api._class_analysis_qwen_review_validate_final(
        expanded,
        result,
        point,
        {"target_context_1", "target_detail_2", "zoom_region_8"},
        clipped_quality,
        evidence_ledger,
    )
    assert clipped_final["decision"] == "skip_uncertain"
    assert any("clipped" in reason for reason in clipped_final["guardrail_reasons"])


def test_class_analysis_qwen_review_allows_clear_accept_without_named_class_guard():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass", "OtherClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.9,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "visible_target_cues": ["rectangular target body", "ribbed target surface"],
            "rationale_short": "Target has clear SuggestedClass-specific features and does not match CurrentClass.",
            "counter_evidence": "OtherClass is listed but does not visibly match the target.",
            "human_review_needed": False,
        },
        result,
        point,
        {"ctx_1"},
        clear_quality,
    )

    assert final["decision"] == "accept_suggested"
    assert final["target_class"] == "SuggestedClass"
    assert final["guardrail_reasons"] == []


def test_class_analysis_qwen_review_blocks_class_change_when_specificity_is_background_dominated():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.91,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "background_dominated",
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": False,
            "visible_target_cues": ["suggested-class texture near target", "scene-compatible surroundings"],
            "rationale_short": "Suggested class is plausible from surrounding context.",
            "counter_evidence": "The visible target itself is not distinctive.",
            "human_review_needed": False,
        },
        result,
        point,
        {"target_context_1"},
        clear_quality,
    )

    assert final["decision"] == "skip_uncertain"
    assert final["target_class"] == "CurrentClass"
    assert final["guarded_recommendation"]["target_background_contrast"] == "background_dominated"
    assert any("target_background_contrast=target_specific" in reason for reason in final["guardrail_reasons"])


def test_class_analysis_qwen_review_specificity_probe_conflict_guards_class_change():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    evidence_ledger = {
        "rows": [
            {"evidence_id": "target_detail_1", "kind": "target_detail", "use": "clean_visual"},
            {"evidence_id": "zoom_region_2", "kind": "zoom_region", "use": "clean_visual"},
        ],
        "clean_visual_evidence_ids": ["target_detail_1", "zoom_region_2"],
        "clean_target_source_evidence_ids": ["target_detail_1", "zoom_region_2"],
        "specificity_probe": {
            "enabled": True,
            "status": "completed",
            "version": api.CLASS_ANALYSIS_QWEN_REVIEW_SPECIFICITY_PROBE_VERSION,
            "specificity_alignment": "supports_current",
            "target_background_contrast": "target_specific",
            "best_supported_class": "CurrentClass",
            "confidence": 0.86,
            "target_specific_cues": ["current-class target structure"],
            "background_or_overlap_cues": ["suggested-class object is nearby"],
        },
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.91,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": False,
            "visible_target_cues": ["rectangular target body", "ribbed target surface"],
            "supporting_clean_evidence_ids": ["target_detail_1", "zoom_region_2"],
            "rationale_short": "Final pass sees suggested-class target cues.",
            "counter_evidence": "Probe disagrees, so verifier should re-check.",
            "human_review_needed": False,
        },
        result,
        point,
        {"target_detail_1", "zoom_region_2"},
        clear_quality,
        evidence_ledger,
    )

    assert final["decision"] == "skip_uncertain"
    assert final["guarded_recommendation"]["target_class"] == "SuggestedClass"
    assert any("specificity probe" in reason for reason in final["guardrail_reasons"])
    assert api._class_analysis_qwen_review_should_run_cue_verifier(final) is True


def test_class_analysis_qwen_review_specificity_margin_blocks_background_favored_change():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    evidence_ledger = {
        "rows": [
            {"evidence_id": "target_detail_1", "kind": "target_detail", "use": "clean_visual"},
            {"evidence_id": "zoom_region_2", "kind": "zoom_region", "use": "clean_visual"},
        ],
        "clean_visual_evidence_ids": ["target_detail_1", "zoom_region_2"],
        "clean_target_source_evidence_ids": ["target_detail_1", "zoom_region_2"],
        "specificity_probe": {
            "enabled": True,
            "status": "completed",
            "version": api.CLASS_ANALYSIS_QWEN_REVIEW_SPECIFICITY_PROBE_VERSION,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "specificity_margin": "background_or_overlap_favored",
            "best_supported_class": "SuggestedClass",
            "confidence": 0.86,
            "target_specific_cues": ["suggested-class texture near target"],
            "background_or_overlap_cues": ["suggested-class texture is outside target"],
            "subdescription_assessments": [
                {
                    "class_name": "SuggestedClass",
                    "subdescription": "suggested-class texture",
                    "target_support": "weak",
                    "background_or_overlap_support": "strong",
                    "support_location": "background",
                    "supporting_clean_evidence_ids": [],
                    "note": "texture is background-dominated",
                }
            ],
        },
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.91,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": False,
            "visible_target_cues": ["suggested-class texture", "elongated target edge"],
            "supporting_clean_evidence_ids": ["target_detail_1", "zoom_region_2"],
            "rationale_short": "Final pass sees suggested-class target cues.",
            "human_review_needed": False,
        },
        result,
        point,
        {"target_detail_1", "zoom_region_2"},
        clear_quality,
        evidence_ledger,
    )

    assert final["decision"] == "skip_uncertain"
    assert final["guarded_recommendation"]["target_class"] == "SuggestedClass"
    assert any("sub-description margin favors background/overlap" in reason for reason in final["guardrail_reasons"])


def test_class_analysis_qwen_review_specificity_probe_parses_loose_qwen_output():
    probe, error = api._class_analysis_qwen_review_parse_specificity_probe_payload(
        json.dumps(
            {
                "best_supported_class": "CurrentClass",
                "specificity_alignment": "current",
                "target_background_contrast": 0.85,
                "target_background_cues": ["nearby suggested-class object", "road texture"],
                "target_specific_cues": ["vertical target edge", "compact target cap"],
                "whole_target_extent_supported": True,
                "rationale_short": "Target cues support current class; suggested cue is nearby.",
            }
        ),
        current_class="CurrentClass",
        suggested_class="SuggestedClass",
        labelmap=["CurrentClass", "SuggestedClass"],
        evidence_ids={"target_detail_1", "zoom_region_2"},
    )

    assert error is None
    assert probe["specificity_alignment"] == "supports_current"
    assert probe["target_background_contrast"] == "target_specific"
    assert probe["confidence"] == pytest.approx(0.85)
    assert probe["background_or_overlap_cues"] == ["nearby suggested-class object", "road texture"]
    assert probe["target_specific_cues"] == ["vertical target edge", "compact target cap"]

    high_probe, high_error = api._class_analysis_qwen_review_parse_specificity_probe_payload(
        json.dumps(
            {
                "best_supported_class": "CurrentClass",
                "specificity_alignment": "high",
                "target_background_contrast": "high",
                "target_specific_cues": ["whole target outline", "distinct target surface"],
                "whole_target_extent_supported": True,
                "confidence": 0.91,
                "rationale_short": "Target cues strongly support current class.",
            }
        ),
        current_class="CurrentClass",
        suggested_class="SuggestedClass",
        labelmap=["CurrentClass", "SuggestedClass"],
        evidence_ids={"target_detail_1", "zoom_region_2"},
    )

    assert high_error is None
    assert high_probe["specificity_alignment"] == "supports_current"
    assert high_probe["target_background_contrast"] == "target_specific"
    assert high_probe["confidence"] == pytest.approx(0.91)
    assert api._class_analysis_qwen_review_specificity_probe_validation_errors(
        high_probe,
        evidence_ids={"target_detail_1", "zoom_region_2"},
    ) == [
        "target_identity_summary is required",
        "high-confidence probe cannot leave target_identity_uncertainty=high",
        "target_specific probe requires supporting_clean_evidence_ids",
        "high-confidence probe requires subdescription_assessments",
    ]


def test_class_analysis_qwen_review_specificity_probe_normalizes_subdescription_assessments():
    probe, error = api._class_analysis_qwen_review_parse_specificity_probe_payload(
        json.dumps(
            {
                "target_identity_summary": "compact target with visible current-class cap",
                "target_identity_uncertainty": "low",
                "specificity_alignment": "supports_current",
                "target_background_contrast": "target_specific",
                "best_supported_class": "CurrentClass",
                "target_specific_cues": ["visible cap", "compact target outline"],
                "background_or_overlap_cues": ["suggested-class texture appears behind target"],
                "contrastive_subdescriptions": [
                    {
                        "class": "CurrentClass",
                        "description": "compact target cap",
                        "target_support": "visible",
                        "background_support": "absent",
                        "location": "inside bbox",
                        "evidence_ids": ["target_detail_1"],
                        "reason": "cap is part of target",
                    },
                    {
                        "class": "SuggestedClass",
                        "description": "suggested-class texture",
                        "target_support": "weak",
                        "context_support": "strong",
                        "location": "context",
                        "evidence_ids": ["source_overlay_99"],
                        "reason": "texture is behind target",
                    },
                ],
                "specificity_margin": "current",
                "margin_rationale": "target sub-descriptions favor current class",
                "current_class_cues": ["visible cap"],
                "suggested_class_cues": [],
                "whole_target_extent_supported": True,
                "supporting_clean_evidence_ids": ["target_detail_1"],
                "confidence": 0.9,
                "rationale_short": "Target cues support current class.",
            }
        ),
        current_class="CurrentClass",
        suggested_class="SuggestedClass",
        labelmap=["CurrentClass", "SuggestedClass"],
        evidence_ids={"target_detail_1", "zoom_region_2"},
    )

    assert error is None
    assert probe["specificity_margin"] == "current_target_favored"
    assert probe["subdescription_assessments"] == [
        {
            "class_name": "CurrentClass",
            "subdescription": "compact target cap",
            "target_support": "strong",
            "background_or_overlap_support": "none",
            "support_location": "target",
            "supporting_clean_evidence_ids": ["target_detail_1"],
            "note": "cap is part of target",
        },
        {
            "class_name": "SuggestedClass",
            "subdescription": "suggested-class texture",
            "target_support": "weak",
            "background_or_overlap_support": "strong",
            "support_location": "background",
            "supporting_clean_evidence_ids": [],
            "note": "texture is behind target",
        },
    ]
    assert api._class_analysis_qwen_review_specificity_probe_validation_errors(
        probe,
        evidence_ids={"target_detail_1", "zoom_region_2"},
    ) == []


def test_class_analysis_qwen_review_specificity_derivation_treats_equal_top_classes_as_mixed():
    derived = api._class_analysis_qwen_review_derive_specificity_from_subdescriptions(
        [
            {
                "class_name": "CurrentClass",
                "subdescription": "compact target frame",
                "target_support": "moderate",
                "background_or_overlap_support": "weak",
                "support_location": "target",
            },
            {
                "class_name": "SuggestedClass",
                "subdescription": "compact target silhouette",
                "target_support": "moderate",
                "background_or_overlap_support": "weak",
                "support_location": "target",
            },
        ],
        current_class="CurrentClass",
        suggested_class="SuggestedClass",
    )

    assert derived["specificity_alignment"] == "mixed"
    assert derived["specificity_margin"] == "low_contrast"
    assert derived["target_background_contrast"] == "mixed"
    assert derived["best_supported_class"] == ""


def test_class_analysis_qwen_review_specificity_probe_reconciles_context_favored_scalars():
    probe, error = api._class_analysis_qwen_review_parse_specificity_probe_payload(
        json.dumps(
            {
                "target_identity_summary": "white open-deck object with canopy beside parking context",
                "target_identity_uncertainty": "low",
                "specificity_alignment": "supports_suggested",
                "target_background_contrast": "background_dominated",
                "best_supported_class": "SuggestedClass",
                "target_specific_cues": ["open deck", "target canopy"],
                "background_or_overlap_cues": ["parking context", "nearby suggested-class objects"],
                "subdescription_assessments": [
                    {
                        "class_name": "CurrentClass",
                        "subdescription": "open-deck target structure",
                        "target_support": "strong",
                        "background_or_overlap_support": "weak",
                        "support_location": "target",
                        "supporting_clean_evidence_ids": ["target_detail_1"],
                        "note": "visible on the reviewed target",
                    },
                    {
                        "class_name": "SuggestedClass",
                        "subdescription": "suggested-class object in its usual scene",
                        "target_support": "moderate",
                        "background_or_overlap_support": "strong",
                        "support_location": "mixed",
                        "supporting_clean_evidence_ids": ["zoom_region_2"],
                        "note": "context supports suggested class more than target pixels do",
                    },
                ],
                "specificity_margin": "suggested_target_favored",
                "margin_rationale": "context made suggested class look plausible",
                "current_class_cues": ["open deck", "canopy"],
                "suggested_class_cues": ["parking context"],
                "whole_target_extent_supported": True,
                "supporting_clean_evidence_ids": ["target_detail_1", "zoom_region_2"],
                "confidence": 0.86,
                "rationale_short": "Target cues current; context suggested.",
            }
        ),
        current_class="CurrentClass",
        suggested_class="SuggestedClass",
        labelmap=["CurrentClass", "SuggestedClass"],
        evidence_ids={"target_detail_1", "zoom_region_2"},
    )

    assert error is None
    assert probe["specificity_alignment"] == "supports_current"
    assert probe["specificity_margin"] == "current_target_favored"
    assert probe["target_background_contrast"] == "target_specific"
    assert probe["best_supported_class"] == "CurrentClass"
    assert probe["reconciled_from_subdescription_assessments"] == [
        "specificity_margin_contradicted_assessments",
        "specificity_alignment_contradicted_assessments",
        "target_background_contrast_contradicted_assessments",
        "best_supported_class_contradicted_assessments",
    ]


def test_class_analysis_qwen_review_specificity_probe_repairs_incomplete_output(monkeypatch):
    loose_output = json.dumps(
        {
            "best_supported_class": "CurrentClass",
            "specificity_alignment": "high",
            "target_background_contrast": "high",
            "target_specific_cues": ["whole target outline", "distinct target surface"],
            "whole_target_extent_supported": True,
            "confidence": 0.91,
            "rationale_short": "Target cues strongly support current class.",
        }
    )
    repaired_output = json.dumps(
        {
            "target_identity_summary": "compact upright target with a distinct cap and visible vertical edge",
            "target_identity_uncertainty": "low",
            "specificity_alignment": "supports_current",
            "target_background_contrast": "target_specific",
            "best_supported_class": "CurrentClass",
            "target_specific_cues": ["whole target outline", "distinct target surface"],
            "background_or_overlap_cues": [],
            "subdescription_assessments": [
                {
                    "class_name": "CurrentClass",
                    "subdescription": "compact upright target outline",
                    "target_support": "strong",
                    "background_or_overlap_support": "none",
                    "support_location": "target",
                    "supporting_clean_evidence_ids": ["target_detail_1"],
                    "note": "visible on the reviewed target",
                },
                {
                    "class_name": "SuggestedClass",
                    "subdescription": "suggested-class background texture",
                    "target_support": "none",
                    "background_or_overlap_support": "moderate",
                    "support_location": "background",
                    "supporting_clean_evidence_ids": [],
                    "note": "only nearby context supports it",
                },
            ],
            "specificity_margin": "current_target_favored",
            "margin_rationale": "target cues favor the current class",
            "current_class_cues": ["whole target outline", "distinct target surface"],
            "suggested_class_cues": [],
            "whole_target_extent_supported": True,
            "supporting_clean_evidence_ids": ["target_detail_1", "zoom_region_2"],
            "confidence": 0.91,
            "rationale_short": "Target-contained cues support the current class.",
        }
    )
    outputs = iter([loose_output, repaired_output])
    calls = []
    events = []

    def fake_model_call(job, messages, **kwargs):
        calls.append({"messages": copy.deepcopy(messages), "kwargs": dict(kwargs)})
        return next(outputs)

    monkeypatch.setattr(api, "_class_analysis_qwen_review_model_call", fake_model_call)
    monkeypatch.setattr(api, "_class_analysis_qwen_review_append_event", lambda _job, payload: events.append(payload))
    job = api.ClassAnalysisQwenReviewJob(
        review_id="probe_repair",
        parent_job_id="parent",
        point_id="p0",
        request={},
    )

    probe = api._class_analysis_qwen_review_run_specificity_probe(
        job,
        final_base_messages=[{"role": "user", "content": [{"type": "text", "text": "base evidence"}]}],
        point={"class_name": "CurrentClass", "suggested_neighbor_class": "SuggestedClass"},
        visual_quality={"tier": "clear"},
        evidence_ledger={"clean_target_source_evidence_ids": ["target_detail_1", "zoom_region_2"]},
        evidence_ids={"target_detail_1", "zoom_region_2"},
        labelmap=["CurrentClass", "SuggestedClass"],
        class_concept_brief_text="",
        model_id="test-model",
    )

    assert len(calls) == 2
    assert calls[0]["kwargs"]["phase"] == "specificity_probe"
    assert calls[0]["kwargs"]["max_new_tokens"] == 800
    assert calls[1]["kwargs"]["phase"] == "specificity_probe"
    assert calls[1]["kwargs"]["max_new_tokens"] == 1000
    assert calls[1]["kwargs"]["event_extra"]["repair_attempt"] == 1
    repair_text = "\n".join(
        content.get("text") or ""
        for message in calls[1]["messages"]
        for content in message.get("content", [])
        if isinstance(content, dict)
    )
    assert "Your previous specificity probe output was incomplete" in repair_text
    assert "target_identity_summary is required" in repair_text
    assert probe["status"] == "completed"
    assert probe["target_identity_summary"] == "compact upright target with a distinct cap and visible vertical edge"
    assert probe["target_identity_uncertainty"] == "low"
    assert probe["supporting_clean_evidence_ids"] == ["target_detail_1", "zoom_region_2"]
    assert probe["specificity_margin"] == "current_target_favored"
    assert len(probe["subdescription_assessments"]) == 2
    assert "validation_errors" not in probe
    assert events[-1]["type"] == "specificity_probe_result"
    assert events[-1]["specificity_probe"]["target_identity_summary"] == probe["target_identity_summary"]


def test_class_analysis_qwen_review_specificity_probe_salvages_malformed_json():
    raw = (
        '{ "target_identity_summary": "long rectangular target with a flat segmented roof", '
        '"target_identity_uncertainty": "low", '
        '"specificity_alignment": "supports_suggested", '
        '"target_background_contrast": "target_specific", '
        '"best_supported_class": "SuggestedClass", '
        '"target_specific_cues": ["rectangular footprint", "flat roof", "rigid structure"], '
        '"background_or_overlap_cues": ["nearby water", "nearby objects"], '
        '"current_class_cues": ["nearby water"], '
        '"suggested_class_cues": ["rectangular footprint", "flat roof"'
    )

    probe, error = api._class_analysis_qwen_review_parse_specificity_probe_payload(
        raw,
        current_class="CurrentClass",
        suggested_class="SuggestedClass",
        labelmap=["CurrentClass", "SuggestedClass"],
        evidence_ids={"target_detail_1", "zoom_region_2"},
    )

    assert error is None
    assert probe["status"] == "completed"
    assert probe["target_identity_summary"] == "long rectangular target with a flat segmented roof"
    assert probe["specificity_alignment"] == "supports_suggested"
    assert probe["target_background_contrast"] == "target_specific"
    assert probe["best_supported_class"] == "SuggestedClass"
    assert probe["target_specific_cues"] == ["rectangular footprint", "flat roof", "rigid structure"]
    assert probe["background_or_overlap_cues"] == ["nearby water", "nearby objects"]
    assert api._class_analysis_qwen_review_specificity_probe_validation_errors(
        probe,
        evidence_ids={"target_detail_1", "zoom_region_2"},
    ) == ["target_specific probe requires supporting_clean_evidence_ids"]


def test_class_analysis_qwen_review_specificity_probe_salvages_after_inner_object_parse_failure():
    raw = (
        '{"target_identity_summary":"compact target with bright front details",'
        '"target_identity_uncertainty":"low",'
        '"specificity_alignment":"supports_suggested",'
        '"target_background_contrast":"target_specific",'
        '"best_supported_class":"SuggestedClass",'
        '"target_specific_cues":["solid enclosed body","bright front details"],'
        '"background_or_overlap_cues":["road texture"],'
        '"subdescription_assessments":['
        '{"class_name":"CurrentClass","subdescription":"open thin frame","target_support":"none",'
        '"background_or_overlap_support":"none","support_location":"absent",'
        '"supporting_clean_evidence_ids":[],"note":"not visible on target"},'
        '{"class_name":"SuggestedClass","subdescription":"solid enclosed body with bright front",'
        '"target_support":"strong","background_or_overlap_support":"none","support_location":"target",'
        '"supporting_clean_evidence_ids":["target_detail_1"],"note":"visible on target"}],'
        '"specificity_margin":"suggested_target_favored",'
        '"margin_rationale":"target descriptors favor the suggested class",'
        '"current_class_cues":[],'
        '"suggested_class_cues":["solid enclosed body"],'
        '"whole_target_extent_supported":true,'
        '"supporting_clean_evidence_ids":["target_detail_1"],'
        '"confidence":0. 95,'
        '"rationale_short":"target pixels fit suggested'
    )

    probe, error = api._class_analysis_qwen_review_parse_specificity_probe_payload(
        raw,
        current_class="CurrentClass",
        suggested_class="SuggestedClass",
        labelmap=["CurrentClass", "SuggestedClass"],
        evidence_ids={"target_detail_1", "source_clean_2"},
    )

    assert error is None
    assert probe["status"] == "completed"
    assert probe["target_identity_summary"] == "compact target with bright front details"
    assert probe["specificity_alignment"] == "supports_suggested"
    assert probe["target_background_contrast"] == "target_specific"
    assert probe["best_supported_class"] == "SuggestedClass"
    assert probe["confidence"] == pytest.approx(0.95)
    assert probe["supporting_clean_evidence_ids"] == ["target_detail_1"]
    assert len(probe["subdescription_assessments"]) == 2
    assert probe["subdescription_assessments"][1]["class_name"] == "SuggestedClass"
    assert api._class_analysis_qwen_review_specificity_probe_validation_errors(
        probe,
        evidence_ids={"target_detail_1", "source_clean_2"},
    ) == []


def test_class_analysis_qwen_review_blocks_expanded_class_change_missing_specificity_audit():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    expanded = api._class_analysis_qwen_review_expand_compact_final(
        {
            "decision": "accept_suggested",
            "final_class": "SuggestedClass",
            "confidence": 0.88,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "global_context_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "visible_target_cues": ["rectangular target body", "ribbed target surface"],
            "rationale_short": "Model omitted the specificity audit fields.",
        },
        point=point,
        evidence_ids={"target_context_1"},
        visual_quality=clear_quality,
        executed_tools={"inspect_target_context"},
    )
    final = api._class_analysis_qwen_review_validate_final(
        expanded,
        result,
        point,
        {"target_context_1"},
        clear_quality,
    )

    assert expanded["specificity_alignment"] == "insufficient"
    assert expanded["target_background_contrast"] == "insufficient"
    assert final["decision"] == "skip_uncertain"
    assert any("specificity_alignment=supports_suggested" in reason for reason in final["guardrail_reasons"])


def test_class_analysis_qwen_review_cue_verifier_promotes_guarded_clear_target(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_cue_verify"
    (class_root / parent_id).mkdir(parents=True)
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    evidence_ledger = {
        "clean_visual_evidence_ids": ["target_context_1", "zoom_region_8"],
        "clean_target_source_evidence_ids": ["target_context_1", "zoom_region_8"],
        "rows": [
            {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
            {"evidence_id": "zoom_region_8", "kind": "zoom_region", "use": "clean_visual"},
        ],
    }
    initial = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.91,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": False,
            "visible_target_cues": ["rectangular target body"],
            "supporting_clean_evidence_ids": ["target_context_1"],
            "rationale_short": "Target visibly fits SuggestedClass.",
            "counter_evidence": "CurrentClass cues are not visible.",
            "human_review_needed": True,
        },
        result,
        point,
        {"target_context_1", "zoom_region_8"},
        clear_quality,
        evidence_ledger,
    )
    assert initial["decision"] == "skip_uncertain"
    assert api._class_analysis_qwen_review_should_run_cue_verifier(initial) is True

    calls = []

    def fake_model_call(*args, **kwargs):
        calls.append(kwargs)
        return json.dumps(
            {
                "verified": True,
                "target_class": "SuggestedClass",
                "cue_confidence": 0.93,
                "positive_visible_target_cues": [
                    "rectangular target body",
                    "ribbed surface texture",
                ],
                "current_class_positive_cues": [],
                "current_class_plausible": False,
                "current_class_plausibility_reason": "Clean target pixels do not match the current class concept.",
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "The proposed class explains the whole target extent.",
                "overlap_rebutted": False,
                "overlap_risk": "not_applicable",
                "overlap_rebuttal": "",
                "supporting_clean_evidence_ids": ["target_context_1", "zoom_region_8"],
                "rejection_reason": "",
            }
        )

    monkeypatch.setattr(api, "_class_analysis_qwen_review_model_call", fake_model_call)
    job = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_cue_verify",
        parent_job_id=parent_id,
        point_id="p0",
        request={},
    )
    promoted = api._class_analysis_qwen_review_try_cue_verifier(
        job,
        final_result=initial,
        final_base_messages=[{"role": "user", "content": [{"type": "text", "text": "base"}]}],
        point=point,
        result=result,
        evidence_ids={"target_context_1", "zoom_region_8"},
        visual_quality=clear_quality,
        evidence_ledger=evidence_ledger,
        labelmap_glossary="",
        review_guidance="",
        deterministic_context={},
        model_id="test-model",
        executed_tools={"inspect_target_context", "zoom_source_region"},
        labelmap=["CurrentClass", "SuggestedClass"],
    )

    assert calls
    assert promoted["decision"] == "accept_suggested"
    assert promoted["target_class"] == "SuggestedClass"
    assert promoted["visible_target_cues"] == [
        "rectangular target body",
        "ribbed surface texture",
    ]
    assert promoted["cue_verifier"]["promoted_from_guarded_recommendation"] is True
    assert promoted["applied"] is False


def test_class_analysis_qwen_review_cue_verifier_promotes_limited_verified_target(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_cue_verify_limited"
    (class_root / parent_id).mkdir(parents=True)
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    limited_quality = {
        "tier": "limited",
        "bbox_width": 48.0,
        "bbox_height": 36.0,
        "bbox_min_dim": 36.0,
        "bbox_area": 1728.0,
        "crop_contrast": 28.0,
        "crop_dynamic_range": 90.0,
        "crop_sharpness": 9.0,
        "edge_clipped": False,
        "reasons": ["limited but reviewable"],
    }
    evidence_ledger = {
        "clean_visual_evidence_ids": ["target_context_1", "target_detail_2"],
        "clean_target_source_evidence_ids": ["target_context_1", "target_detail_2"],
        "specificity_probe": {
            "status": "completed",
            "confidence": 0.92,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "specificity_margin": "suggested_target_favored",
            "best_supported_class": "SuggestedClass",
        },
        "rows": [
            {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
            {"evidence_id": "target_detail_2", "kind": "target_detail", "use": "clean_visual"},
        ],
    }
    initial = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.91,
            "visual_quality": "limited",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "insufficient",
            "same_image_embedding_evidence": "neutral",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "glossary_or_guidance_used": True,
            "visible_target_cues": [
                "distinct target contour",
                "target-specific surface markings",
            ],
            "supporting_clean_evidence_ids": ["target_context_1", "target_detail_2"],
            "rationale_short": "Clean target cues fit SuggestedClass, but quality is limited.",
            "counter_evidence": "CurrentClass-specific cues are absent.",
            "human_review_needed": True,
        },
        result,
        point,
        {"target_context_1", "target_detail_2"},
        limited_quality,
        evidence_ledger,
    )
    assert initial["decision"] == "skip_uncertain"
    assert initial["guarded_recommendation"]["backend_tier"] == "limited"
    assert api._class_analysis_qwen_review_should_run_cue_verifier(initial) is True

    calls = []

    def fake_model_call(*args, **kwargs):
        calls.append(kwargs)
        return json.dumps(
            {
                "verified": True,
                "target_class": "SuggestedClass",
                "cue_confidence": 0.94,
                "positive_visible_target_cues": [
                    "distinct target contour",
                    "target-specific surface markings",
                ],
                "target_class_defining_cues": [
                    "target-specific surface markings",
                    "compact target silhouette",
                ],
                "current_class_positive_cues": [],
                "current_class_missing_or_inconsistent_cues": [
                    "no current-class edge pattern",
                ],
                "current_class_plausibility_basis": "none",
                "current_class_plausible": False,
                "current_class_plausibility_reason": "Clean target pixels lack current-class-specific cues.",
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "The proposed class explains the full target extent.",
                "overlap_rebutted": False,
                "overlap_risk": "not_applicable",
                "overlap_rebuttal": "",
                "anchor_support_verified": True,
                "anchor_support_basis": "target_specific_anchors",
                "anchor_support_reason": "Trusted anchors share the same target-internal markings.",
                "supporting_clean_evidence_ids": ["target_context_1", "target_detail_2"],
                "rejection_reason": "",
            }
        )

    monkeypatch.setattr(api, "_class_analysis_qwen_review_model_call", fake_model_call)
    job = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_cue_verify_limited",
        parent_job_id=parent_id,
        point_id="p0",
        request={},
    )
    promoted = api._class_analysis_qwen_review_try_cue_verifier(
        job,
        final_result=initial,
        final_base_messages=[{"role": "user", "content": [{"type": "text", "text": "base"}]}],
        point=point,
        result=result,
        evidence_ids={"target_context_1", "target_detail_2"},
        visual_quality=limited_quality,
        evidence_ledger=evidence_ledger,
        labelmap_glossary="CurrentClass: synthetic current class\nSuggestedClass: synthetic target class",
        review_guidance="",
        deterministic_context={
            "scale": {"signal": "insufficient"},
            "embedding": {"signal": "neutral"},
        },
        model_id="test-model",
        executed_tools={"inspect_target_context", "inspect_target_detail"},
        labelmap=["CurrentClass", "SuggestedClass"],
    )

    assert calls
    assert promoted["decision"] == "accept_suggested"
    assert promoted["target_class"] == "SuggestedClass"
    assert promoted["human_review_needed"] is True
    assert promoted["backend_visual_quality"]["tier"] == "limited"
    assert promoted["cue_verifier"]["promoted_from_guarded_recommendation"] is True
    assert promoted["confidence"] <= 0.65


def test_class_analysis_qwen_review_cue_verifier_promotes_limited_dual_bbox_unclear_first_pass(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_cue_verify_dual_limited"
    (class_root / parent_id).mkdir(parents=True)
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
        "dual_bbox_conflict": {
            "enabled": True,
            "kind": "near_identical_cross_class_bbox",
            "review_mode": "dual_bbox_class_resolution",
            "point_id": "p0",
            "current_class": "CurrentClass",
            "other_point_id": "p1",
            "other_class_name": "SuggestedClass",
            "class_name": "SuggestedClass",
            "classes": ["CurrentClass", "SuggestedClass"],
            "iou": 0.98,
            "target_area_covered": 0.98,
            "other_area_covered": 0.98,
            "relation": "duplicate_like",
        },
    }
    limited_quality = {
        "tier": "limited",
        "bbox_width": 44.0,
        "bbox_height": 32.0,
        "bbox_min_dim": 32.0,
        "bbox_area": 1408.0,
        "crop_contrast": 24.0,
        "crop_dynamic_range": 85.0,
        "crop_sharpness": 8.0,
        "edge_clipped": False,
        "reasons": ["limited but reviewable"],
    }
    evidence_ledger = {
        "clean_visual_evidence_ids": ["target_context_1", "target_detail_2", "source_clean_3"],
        "clean_target_source_evidence_ids": ["target_context_1", "target_detail_2", "source_clean_3"],
        "rows": [
            {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
            {"evidence_id": "target_detail_2", "kind": "target_detail", "use": "clean_visual"},
            {"evidence_id": "source_clean_3", "kind": "source_clean", "use": "clean_visual"},
        ],
        "dual_bbox_conflict": copy.deepcopy(point["dual_bbox_conflict"]),
        "specificity_probe": {
            "status": "completed",
            "confidence": 0.88,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "background_dominated",
            "specificity_margin": "suggested_target_favored",
            "best_supported_class": "SuggestedClass",
            "validation_errors": ["target/background contrast was context-mixed after subdescription reconciliation"],
        },
    }
    initial = {
        "decision": "skip_uncertain",
        "target_class": "CurrentClass",
        "confidence": 0.45,
        "guarded_recommendation": {
            "blocked": True,
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.96,
            "current_class": "CurrentClass",
            "suggested_neighbor_class": "SuggestedClass",
            "visual_quality": "limited",
            "object_visibility": "partial",
            "backend_tier": "limited",
            "backend_edge_clipped": False,
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "same_image_scale_evidence": "neutral",
            "same_image_embedding_evidence": "neutral",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "specificity_alignment": "insufficient",
            "target_background_contrast": "background_dominated",
            "target_identity_summary": "limited target shows suggested-class structure",
            "target_identity_uncertainty": "moderate",
            "target_identity_evidence_ids": ["target_context_1"],
            "whole_target_extent_supported": True,
            "whole_target_extent_reason": "The suggested class explains the whole target.",
            "overlap_assessment": "unclear",
            "dual_bbox_resolution": "overlap_box_class",
            "dual_bbox_conflict": copy.deepcopy(point["dual_bbox_conflict"]),
            "visible_target_cues": ["target-specific outline"],
            "supporting_clean_evidence_ids": ["target_context_1"],
            "guardrail_reasons": [
                "accept_suggested is advisory-only because backend visual-quality tier is limited",
                "target/background contrast is background_dominated",
                "accept_suggested requires at least two concrete visible target cues, got 1",
                "overlap assessment unclear is too entangled for relabel recommendation",
            ],
            "advisory_reasons": [],
            "rationale_short": "Clean pixels favor the overlapping class, but context was mixed.",
            "counter_evidence": "Current-class cues are not visible.",
        },
    }

    assert api._class_analysis_qwen_review_should_run_cue_verifier(initial) is True

    def fake_model_call(*args, **kwargs):
        return json.dumps(
            {
                "verified": True,
                "target_class": "SuggestedClass",
                "cue_confidence": 0.94,
                "positive_visible_target_cues": [
                    "target-specific outline",
                    "target-specific surface detail",
                ],
                "target_class_defining_cues": [
                    "target-specific surface detail",
                    "whole target silhouette",
                ],
                "current_class_positive_cues": [],
                "current_class_missing_or_inconsistent_cues": [
                    "no current-class defining structure",
                ],
                "current_class_plausibility_basis": "none",
                "current_class_plausible": False,
                "current_class_plausibility_reason": "Clean pixels lack current-class structure.",
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "The suggested class explains the whole target.",
                "overlap_rebutted": True,
                "overlap_risk": "target_specific",
                "overlap_rebuttal": "Overlap does not explain the target-specific details.",
                "anchor_support_verified": True,
                "anchor_support_basis": "target_specific_anchors",
                "anchor_support_reason": "Anchors share target-specific structure.",
                "supporting_clean_evidence_ids": ["target_context_1", "target_detail_2"],
                "rejection_reason": "",
            }
        )

    monkeypatch.setattr(api, "_class_analysis_qwen_review_model_call", fake_model_call)
    job = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_cue_verify_dual_limited",
        parent_job_id=parent_id,
        point_id="p0",
        request={},
    )

    promoted = api._class_analysis_qwen_review_try_cue_verifier(
        job,
        final_result=initial,
        final_base_messages=[{"role": "user", "content": [{"type": "text", "text": "base"}]}],
        point=point,
        result=result,
        evidence_ids={"target_context_1", "target_detail_2", "source_clean_3"},
        visual_quality=limited_quality,
        evidence_ledger=evidence_ledger,
        labelmap_glossary="",
        review_guidance="",
        deterministic_context={
            "scale": {"signal": "neutral"},
            "embedding": {"signal": "neutral"},
        },
        model_id="test-model",
        executed_tools={"inspect_target_context", "inspect_target_detail", "inspect_source_overlay"},
        labelmap=["CurrentClass", "SuggestedClass"],
    )

    assert promoted["decision"] == "accept_suggested"
    assert promoted["target_class"] == "SuggestedClass"
    assert promoted["overlap_assessment"] == "duplicate_like"
    assert promoted["dual_bbox_resolution"] == "overlap_box_class"
    assert promoted["cue_verifier"]["promoted_from_guarded_recommendation"] is True


def test_class_analysis_qwen_review_cue_verifier_runs_on_limited_current_supported_target_for_triage():
    final = {
        "decision": "skip_uncertain",
        "guarded_recommendation": {
            "blocked": True,
            "decision": "accept_suggested",
            "current_class": "CurrentClass",
            "suggested_neighbor_class": "SuggestedClass",
            "target_class": "SuggestedClass",
            "backend_tier": "limited",
            "visual_quality": "limited",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_suggested": "moderate",
            "same_image_scale_evidence": "supports_current",
            "same_image_embedding_evidence": "neutral",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "visible_target_cues": ["cue one", "cue two"],
            "guardrail_reasons": [
                "accept_suggested is advisory-only because backend visual-quality tier is limited",
                "moderate-anchor class change with no same-image deterministic support requires current-class plausibility verification",
            ],
        },
    }

    assert api._class_analysis_qwen_review_should_run_cue_verifier(final) is True


def test_class_analysis_qwen_review_cue_verifier_runs_on_limited_partial_target_for_triage():
    final = {
        "decision": "skip_uncertain",
        "guarded_recommendation": {
            "blocked": True,
            "decision": "accept_suggested",
            "current_class": "CurrentClass",
            "suggested_neighbor_class": "SuggestedClass",
            "target_class": "SuggestedClass",
            "backend_tier": "limited",
            "visual_quality": "limited",
            "object_visibility": "partial",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_suggested": "moderate",
            "same_image_scale_evidence": "neutral",
            "same_image_embedding_evidence": "supports_current",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "visible_target_cues": ["cue one", "cue two"],
            "guardrail_reasons": [
                "accept_suggested is advisory-only because backend visual-quality tier is limited",
                "overlap decomposition says overlapping-object pixels explain candidate-class similarity",
                "overlap assessment unclear is too entangled for relabel recommendation",
            ],
        },
    }

    assert api._class_analysis_qwen_review_should_run_cue_verifier(final) is True


def test_class_analysis_qwen_review_cue_verifier_runs_on_edge_clipped_limited_target_for_triage():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    edge_clipped_quality = {
        "tier": "limited",
        "bbox_width": 48.0,
        "bbox_height": 36.0,
        "bbox_min_dim": 36.0,
        "bbox_area": 1728.0,
        "crop_contrast": 28.0,
        "crop_dynamic_range": 90.0,
        "crop_sharpness": 9.0,
        "edge_clipped": True,
        "reasons": ["bbox touches the source image edge"],
    }
    evidence_ledger = {
        "clean_visual_evidence_ids": ["target_context_1", "target_detail_2"],
        "clean_target_source_evidence_ids": ["target_context_1", "target_detail_2"],
        "specificity_probe": {
            "status": "completed",
            "confidence": 0.92,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "specificity_margin": "suggested_target_favored",
            "best_supported_class": "SuggestedClass",
        },
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.91,
            "visual_quality": "limited",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "anchor_adjudication_verified": True,
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "insufficient",
            "same_image_embedding_evidence": "neutral",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "current_class_plausible": False,
            "whole_target_extent_supported": True,
            "_cue_verifier_class_change_verified": True,
            "_cue_verifier_confidence": 0.95,
            "glossary_or_guidance_used": True,
            "visible_target_cues": [
                "distinct target contour",
                "target-specific surface markings",
            ],
            "supporting_clean_evidence_ids": ["target_context_1", "target_detail_2"],
            "rationale_short": "Clean target cues fit SuggestedClass, but quality is edge clipped.",
            "counter_evidence": "CurrentClass-specific cues are absent.",
            "human_review_needed": True,
        },
        result,
        point,
        {"target_context_1", "target_detail_2"},
        edge_clipped_quality,
        evidence_ledger,
    )

    assert final["decision"] == "skip_uncertain"
    guarded = final["guarded_recommendation"]
    assert guarded["backend_edge_clipped"] is True
    assert "source image edge" in " ".join(guarded["guardrail_reasons"])
    assert api._class_analysis_qwen_review_should_run_cue_verifier(final) is True


def test_class_analysis_qwen_review_cue_verifier_enriches_edge_clipped_without_promotion(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_cue_verify_edge_triage"
    (class_root / parent_id).mkdir(parents=True)
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    edge_clipped_quality = {
        "tier": "limited",
        "bbox_width": 48.0,
        "bbox_height": 36.0,
        "bbox_min_dim": 36.0,
        "bbox_area": 1728.0,
        "crop_contrast": 28.0,
        "crop_dynamic_range": 90.0,
        "crop_sharpness": 9.0,
        "edge_clipped": True,
        "reasons": ["bbox touches the source image edge"],
    }
    evidence_ledger = {
        "clean_visual_evidence_ids": ["target_context_1", "target_detail_2"],
        "clean_target_source_evidence_ids": ["target_context_1", "target_detail_2"],
        "rows": [
            {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
            {"evidence_id": "target_detail_2", "kind": "target_detail", "use": "clean_visual"},
        ],
        "specificity_probe": {
            "status": "completed",
            "confidence": 0.92,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "specificity_margin": "suggested_target_favored",
            "best_supported_class": "SuggestedClass",
        },
    }
    initial = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.91,
            "visual_quality": "limited",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "anchor_adjudication_verified": True,
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "insufficient",
            "same_image_embedding_evidence": "neutral",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "current_class_plausible": False,
            "whole_target_extent_supported": True,
            "glossary_or_guidance_used": True,
            "visible_target_cues": [
                "distinct target contour",
                "target-specific surface markings",
            ],
            "supporting_clean_evidence_ids": ["target_context_1", "target_detail_2"],
            "rationale_short": "Clean target cues fit SuggestedClass, but quality is edge clipped.",
            "counter_evidence": "CurrentClass-specific cues are absent.",
            "human_review_needed": True,
        },
        result,
        point,
        {"target_context_1", "target_detail_2"},
        edge_clipped_quality,
        evidence_ledger,
    )
    assert initial["decision"] == "skip_uncertain"
    assert api._class_analysis_qwen_review_should_run_cue_verifier(initial) is True

    calls = []

    def fake_model_call(*args, **kwargs):
        calls.append(kwargs)
        return json.dumps(
            {
                "verified": True,
                "target_class": "SuggestedClass",
                "cue_confidence": 0.94,
                "positive_visible_target_cues": [
                    "distinct target contour",
                    "target-specific surface markings",
                ],
                "target_class_defining_cues": [
                    "target-specific surface markings",
                    "compact target silhouette",
                ],
                "current_class_positive_cues": [],
                "current_class_missing_or_inconsistent_cues": [
                    "no current-class edge pattern",
                ],
                "current_class_plausibility_basis": "none",
                "current_class_plausible": False,
                "current_class_plausibility_reason": "Clean target pixels lack current-class-specific cues.",
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "The proposed class explains the full target extent.",
                "overlap_rebutted": False,
                "overlap_risk": "not_applicable",
                "overlap_rebuttal": "",
                "anchor_support_verified": True,
                "anchor_support_basis": "target_specific_anchors",
                "anchor_support_reason": "Trusted anchors share the same target-internal markings.",
                "supporting_clean_evidence_ids": ["target_context_1", "target_detail_2"],
                "rejection_reason": "",
            }
        )

    monkeypatch.setattr(api, "_class_analysis_qwen_review_model_call", fake_model_call)
    job = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_cue_verify_edge_triage",
        parent_job_id=parent_id,
        point_id="p0",
        request={},
    )
    reviewed = api._class_analysis_qwen_review_try_cue_verifier(
        job,
        final_result=initial,
        final_base_messages=[{"role": "user", "content": [{"type": "text", "text": "base"}]}],
        point=point,
        result=result,
        evidence_ids={"target_context_1", "target_detail_2"},
        visual_quality=edge_clipped_quality,
        evidence_ledger=evidence_ledger,
        labelmap_glossary="CurrentClass: synthetic current class\nSuggestedClass: synthetic target class",
        review_guidance="",
        deterministic_context={
            "scale": {"signal": "insufficient"},
            "embedding": {"signal": "neutral"},
        },
        model_id="test-model",
        executed_tools={"inspect_target_context", "inspect_target_detail"},
        labelmap=["CurrentClass", "SuggestedClass"],
    )

    assert calls
    assert reviewed["decision"] == "skip_uncertain"
    assert reviewed["guarded_recommendation"]["backend_edge_clipped"] is True
    assert reviewed["cue_verifier"]["verified"] is True
    assert reviewed["cue_verifier"]["promoted_from_guarded_recommendation"] is False
    assert "source image edge" in " ".join(reviewed["guardrail_reasons"])


def test_class_analysis_qwen_review_cue_verifier_promotes_edge_clipped_when_visible_extent_is_diagnostic(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_cue_verify_edge_promote"
    (class_root / parent_id).mkdir(parents=True)
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    edge_clipped_quality = {
        "tier": "limited",
        "bbox_width": 64.0,
        "bbox_height": 52.0,
        "bbox_min_dim": 52.0,
        "bbox_area": 3328.0,
        "crop_contrast": 32.0,
        "crop_dynamic_range": 110.0,
        "crop_sharpness": 12.0,
        "edge_clipped": True,
        "reasons": ["bbox touches the source image edge"],
    }
    evidence_ledger = {
        "clean_visual_evidence_ids": ["target_context_1", "target_detail_2"],
        "clean_target_source_evidence_ids": ["target_context_1", "target_detail_2"],
        "rows": [
            {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
            {"evidence_id": "target_detail_2", "kind": "target_detail", "use": "clean_visual"},
        ],
        "specificity_probe": {
            "status": "completed",
            "confidence": 0.93,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "specificity_margin": "suggested_target_favored",
            "best_supported_class": "SuggestedClass",
        },
    }
    initial = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.91,
            "visual_quality": "limited",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "anchor_adjudication_verified": True,
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "insufficient",
            "same_image_embedding_evidence": "neutral",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "current_class_plausible": False,
            "whole_target_extent_supported": True,
            "glossary_or_guidance_used": True,
            "visible_target_cues": [
                "distinct target contour",
                "target-specific surface markings",
            ],
            "supporting_clean_evidence_ids": ["target_context_1", "target_detail_2"],
            "rationale_short": "Clean target cues fit SuggestedClass, but quality is edge clipped.",
            "counter_evidence": "CurrentClass-specific cues are absent.",
            "human_review_needed": True,
        },
        result,
        point,
        {"target_context_1", "target_detail_2"},
        edge_clipped_quality,
        evidence_ledger,
    )
    assert initial["decision"] == "skip_uncertain"
    assert api._class_analysis_qwen_review_should_run_cue_verifier(initial) is True

    def fake_model_call(*args, **kwargs):
        return json.dumps(
            {
                "verified": True,
                "target_class": "SuggestedClass",
                "cue_confidence": 0.94,
                "positive_visible_target_cues": [
                    "distinct target contour",
                    "target-specific surface markings",
                ],
                "target_class_defining_cues": [
                    "target-specific surface markings",
                    "compact target silhouette",
                ],
                "current_class_positive_cues": [],
                "current_class_missing_or_inconsistent_cues": [
                    "no current-class edge pattern",
                ],
                "current_class_plausibility_basis": "none",
                "current_class_plausible": False,
                "current_class_plausibility_reason": "Clean target pixels lack current-class-specific cues.",
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "Visible extent is explained by the proposed class.",
                "edge_clip_recoverable": True,
                "edge_clip_recoverability_reason": "Edge clipping does not hide class-critical parts.",
                "overlap_rebutted": False,
                "overlap_risk": "not_applicable",
                "overlap_rebuttal": "",
                "anchor_support_verified": True,
                "anchor_support_basis": "target_specific_anchors",
                "anchor_support_reason": "Trusted anchors share the same target-internal markings.",
                "supporting_clean_evidence_ids": ["target_context_1", "target_detail_2"],
                "rejection_reason": "",
            }
        )

    monkeypatch.setattr(api, "_class_analysis_qwen_review_model_call", fake_model_call)
    job = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_cue_verify_edge_promote",
        parent_job_id=parent_id,
        point_id="p0",
        request={},
    )
    reviewed = api._class_analysis_qwen_review_try_cue_verifier(
        job,
        final_result=initial,
        final_base_messages=[{"role": "user", "content": [{"type": "text", "text": "base"}]}],
        point=point,
        result=result,
        evidence_ids={"target_context_1", "target_detail_2"},
        visual_quality=edge_clipped_quality,
        evidence_ledger=evidence_ledger,
        labelmap_glossary="CurrentClass: synthetic current class\nSuggestedClass: synthetic target class",
        review_guidance="",
        deterministic_context={
            "scale": {"signal": "insufficient"},
            "embedding": {"signal": "neutral"},
        },
        model_id="test-model",
        executed_tools={"inspect_target_context", "inspect_target_detail"},
        labelmap=["CurrentClass", "SuggestedClass"],
    )

    assert reviewed["decision"] == "accept_suggested"
    assert reviewed["target_class"] == "SuggestedClass"
    assert reviewed["cue_verifier"]["edge_clip_recoverable"] is True
    assert reviewed["cue_verifier"]["promoted_from_guarded_recommendation"] is True
    assert "source image edge" not in " ".join(reviewed["guardrail_reasons"])


def test_class_analysis_qwen_review_cue_verifier_promotes_verified_moderate_anchor_overlap(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_cue_verify_moderate_anchor"
    (class_root / parent_id).mkdir(parents=True)
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    evidence_ledger = {
        "clean_visual_evidence_ids": ["target_detail_2", "source_clean_3"],
        "clean_target_source_evidence_ids": ["target_detail_2", "source_clean_3"],
        "rows": [
            {"evidence_id": "target_detail_2", "kind": "target_detail", "use": "clean_visual"},
            {"evidence_id": "source_clean_3", "kind": "source_clean", "use": "clean_visual"},
        ],
    }
    initial = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.9,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "insufficient",
            "same_image_embedding_evidence": "insufficient",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "glossary_or_guidance_used": True,
            "visible_target_cues": [
                "spiral translucent membrane pattern",
                "hexagonal clasp geometry",
            ],
            "supporting_clean_evidence_ids": ["target_detail_2", "source_clean_3"],
            "rationale_short": "Clean target cues fit SuggestedClass.",
            "counter_evidence": "Anchors are only moderate.",
            "human_review_needed": True,
        },
        result,
        point,
        {"target_detail_2", "source_clean_3"},
        clear_quality,
        evidence_ledger,
    )
    assert initial["decision"] == "skip_uncertain"
    assert initial["guarded_recommendation"]["anchor_evidence_suggested"] == "moderate"
    assert api._class_analysis_qwen_review_should_run_cue_verifier(initial) is True

    def fake_model_call(*args, **kwargs):
        return json.dumps(
            {
                "verified": True,
                "target_class": "SuggestedClass",
                "cue_confidence": 0.93,
                "positive_visible_target_cues": [
                    "spiral translucent membrane pattern",
                    "hexagonal clasp geometry",
                ],
                "current_class_missing_or_inconsistent_cues": [
                    "no paired support rails",
                ],
                "current_class_positive_cues": [],
                "current_class_plausibility_basis": "none",
                "current_class_plausible": False,
                "current_class_plausibility_reason": "No current-class-specific target pixels are visible.",
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "The proposed class explains the whole target extent.",
                "overlap_rebutted": True,
                "overlap_risk": "target_specific",
                "overlap_rebuttal": (
                    "The reviewed pixels carry the membrane and clasp cues inside the object extent, "
                    "with the nearby overlap kept separate in clean evidence."
                ),
                "anchor_support_verified": True,
                "anchor_support_basis": "target_specific_anchors",
                "anchor_support_reason": "Trusted anchors share the same target-internal membrane and clasp traits.",
                "supporting_clean_evidence_ids": ["target_detail_2", "source_clean_3"],
                "rejection_reason": "",
            }
        )

    monkeypatch.setattr(api, "_class_analysis_qwen_review_model_call", fake_model_call)
    job = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_cue_verify_moderate_anchor",
        parent_job_id=parent_id,
        point_id="p0",
        request={},
    )
    promoted = api._class_analysis_qwen_review_try_cue_verifier(
        job,
        final_result=initial,
        final_base_messages=[{"role": "user", "content": [{"type": "text", "text": "base"}]}],
        point=point,
        result=result,
        evidence_ids={"target_detail_2", "source_clean_3"},
        visual_quality=clear_quality,
        evidence_ledger=evidence_ledger,
        labelmap_glossary="",
        review_guidance="",
        deterministic_context={},
        model_id="test-model",
        executed_tools={"inspect_target_detail", "inspect_source_overlay"},
        labelmap=["CurrentClass", "SuggestedClass"],
    )

    assert promoted["decision"] == "accept_suggested"
    assert promoted["target_class"] == "SuggestedClass"
    assert promoted["anchor_adjudication_verified"] is True
    assert promoted["overlap_adjudication_verified"] is True
    assert promoted["cue_verifier"]["anchor_support_basis"] == "target_specific_anchors"
    assert promoted["cue_verifier"]["promoted_from_guarded_recommendation"] is True


def test_class_analysis_qwen_review_cue_verifier_blocks_neighbor_biased_moderate_overlap(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_cue_verify_neighbor_bias"
    (class_root / parent_id).mkdir(parents=True)
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 70.0,
        "bbox_height": 34.0,
        "bbox_min_dim": 34.0,
        "bbox_area": 2380.0,
        "crop_contrast": 60.0,
        "crop_dynamic_range": 200.0,
        "crop_sharpness": 25.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    evidence_ledger = {
        "clean_visual_evidence_ids": ["target_context_1", "target_detail_2", "source_clean_3", "zoom_region_9"],
        "clean_target_source_evidence_ids": ["target_context_1", "target_detail_2", "source_clean_3", "zoom_region_9"],
        "rows": [
            {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
            {"evidence_id": "target_detail_2", "kind": "target_detail", "use": "clean_visual"},
            {"evidence_id": "source_clean_3", "kind": "source_clean", "use": "clean_visual"},
            {"evidence_id": "zoom_region_9", "kind": "zoom_region", "use": "clean_visual"},
        ],
    }
    initial = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.9,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "supports_suggested",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "insufficient",
            "same_image_embedding_evidence": "insufficient",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "glossary_or_guidance_used": False,
            "visible_target_cues": [
                "smooth bright body",
                "compact top-down profile",
            ],
            "supporting_clean_evidence_ids": ["target_context_1", "target_detail_2"],
            "rationale_short": "Nearby examples make SuggestedClass plausible.",
            "counter_evidence": "Moderate anchors and overlap need verifier grounding.",
            "human_review_needed": True,
        },
        result,
        point,
        {"target_context_1", "target_detail_2", "source_clean_3", "zoom_region_9"},
        clear_quality,
        evidence_ledger,
    )
    assert initial["decision"] == "skip_uncertain"
    assert api._class_analysis_qwen_review_should_run_cue_verifier(initial) is True

    def fake_model_call(*args, **kwargs):
        return json.dumps(
            {
                "verified": True,
                "target_class": "SuggestedClass",
                "cue_confidence": 0.92,
                "positive_visible_target_cues": [
                    "smooth bright body",
                    "compact top-down profile",
                ],
                "target_class_defining_cues": [
                    "smooth bright body",
                    "compact top-down profile",
                ],
                "current_class_positive_cues": [],
                # This mirrors the audited failure: the model asserts a class
                # change but gives no surviving clean-pixel contradiction for
                # the current class, and deterministic reports are insufficient.
                "current_class_missing_or_inconsistent_cues": [],
                "current_class_plausibility_basis": "none",
                "current_class_plausible": False,
                "current_class_plausibility_reason": "",
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "The proposed class explains the whole target extent.",
                "overlap_rebutted": True,
                "overlap_risk": "target_specific",
                "overlap_rebuttal": "Overlap does not explain the proposed target cues.",
                "anchor_support_verified": True,
                "anchor_support_basis": "target_specific_anchors",
                "anchor_support_reason": "Trusted anchors share the asserted target traits.",
                "supporting_clean_evidence_ids": ["target_context_1", "target_detail_2", "source_clean_3", "zoom_region_9"],
                "rejection_reason": "",
            }
        )

    monkeypatch.setattr(api, "_class_analysis_qwen_review_model_call", fake_model_call)
    job = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_cue_verify_neighbor_bias",
        parent_job_id=parent_id,
        point_id="p0",
        request={},
    )
    guarded = api._class_analysis_qwen_review_try_cue_verifier(
        job,
        final_result=initial,
        final_base_messages=[{"role": "user", "content": [{"type": "text", "text": "base"}]}],
        point=point,
        result=result,
        evidence_ids={"target_context_1", "target_detail_2", "source_clean_3", "zoom_region_9"},
        visual_quality=clear_quality,
        evidence_ledger=evidence_ledger,
        labelmap_glossary="",
        review_guidance="",
        deterministic_context={
            "scale": {"signal": "insufficient"},
            "embedding": {"signal": "insufficient"},
        },
        model_id="test-model",
        executed_tools={"inspect_target_context", "inspect_target_detail", "zoom_source_region"},
        labelmap=["CurrentClass", "SuggestedClass"],
    )

    assert guarded["decision"] == "skip_uncertain"
    assert guarded["cue_verifier"]["verified"] is False
    assert "local consensus alone is not enough" in guarded["cue_verifier"]["rejection_reason"]


def test_class_analysis_qwen_review_cue_verifier_promotes_contrastive_moderate_anchor(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_cue_verify_contrastive_anchor"
    (class_root / parent_id).mkdir(parents=True)
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    evidence_ledger = {
        "clean_visual_evidence_ids": ["target_detail_2", "source_clean_3"],
        "clean_target_source_evidence_ids": ["target_detail_2", "source_clean_3"],
        "rows": [
            {"evidence_id": "target_detail_2", "kind": "target_detail", "use": "clean_visual"},
            {"evidence_id": "source_clean_3", "kind": "source_clean", "use": "clean_visual"},
        ],
    }
    initial = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.9,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "insufficient",
            "same_image_embedding_evidence": "insufficient",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "glossary_or_guidance_used": True,
            "visible_target_cues": [
                "rectangular footprint",
                "ribbed roof texture",
            ],
            "supporting_clean_evidence_ids": ["target_detail_2", "source_clean_3"],
            "rationale_short": "Target cues fit SuggestedClass.",
            "counter_evidence": "One cue is shared, so verifier must contrast classes.",
            "human_review_needed": True,
        },
        result,
        point,
        {"target_detail_2", "source_clean_3"},
        clear_quality,
        evidence_ledger,
    )
    assert initial["decision"] == "skip_uncertain"
    assert api._class_analysis_qwen_review_should_run_cue_verifier(initial) is True

    def fake_model_call(*args, **kwargs):
        return json.dumps(
            {
                "verified": False,
                "target_class": "SuggestedClass",
                "cue_confidence": 0.9,
                "positive_visible_target_cues": [
                    "rectangular footprint",
                    "ribbed roof texture",
                ],
                "target_class_defining_cues": [
                    "ribbed roof panels",
                    "flat roof plane",
                ],
                "current_class_positive_cues": ["rectangular footprint"],
                "current_class_missing_or_inconsistent_cues": [
                    "no rounded end caps",
                    "no curved exterior surface",
                ],
                "current_class_plausibility_basis": "shared_generic_cues",
                "current_class_plausible": False,
                "current_class_plausibility_reason": "The shared footprint is not independently current-class-specific.",
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "The proposed class explains the whole target extent.",
                "overlap_rebutted": True,
                "overlap_risk": "target_specific",
                "overlap_rebuttal": "Overlap does not explain the target-contained ribbed panels.",
                "anchor_support_verified": True,
                "anchor_support_basis": "target_specific_anchors",
                "anchor_support_reason": "Trusted anchors share ribbed panels and a flat roof plane.",
                "supporting_clean_evidence_ids": ["target_detail_2", "source_clean_3"],
                "rejection_reason": "",
            }
        )

    monkeypatch.setattr(api, "_class_analysis_qwen_review_model_call", fake_model_call)
    job = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_cue_verify_contrastive_anchor",
        parent_job_id=parent_id,
        point_id="p0",
        request={},
    )
    promoted = api._class_analysis_qwen_review_try_cue_verifier(
        job,
        final_result=initial,
        final_base_messages=[{"role": "user", "content": [{"type": "text", "text": "base"}]}],
        point=point,
        result=result,
        evidence_ids={"target_detail_2", "source_clean_3"},
        visual_quality=clear_quality,
        evidence_ledger=evidence_ledger,
        labelmap_glossary="CurrentClass: synthetic current class\nSuggestedClass: synthetic target class",
        review_guidance="",
        deterministic_context={},
        model_id="test-model",
        executed_tools={"inspect_target_detail", "inspect_source_overlay"},
        labelmap=["CurrentClass", "SuggestedClass"],
    )

    assert promoted["decision"] == "accept_suggested"
    assert promoted["target_class"] == "SuggestedClass"
    assert promoted["cue_verifier"]["raw_verified"] is False
    assert promoted["cue_verifier"]["reconciled_to_verified"] is True
    assert promoted["cue_verifier"]["promoted_from_guarded_recommendation"] is True
    assert "ribbed roof panels" in promoted["visible_target_cues"]
    assert "no rounded end caps" in promoted["counter_evidence"]


def test_class_analysis_qwen_review_cue_verifier_blocks_moderate_shared_anchors(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_cue_verify_shared_anchor"
    (class_root / parent_id).mkdir(parents=True)
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    evidence_ledger = {
        "clean_visual_evidence_ids": ["target_detail_2", "source_clean_3"],
        "clean_target_source_evidence_ids": ["target_detail_2", "source_clean_3"],
        "rows": [
            {"evidence_id": "target_detail_2", "kind": "target_detail", "use": "clean_visual"},
            {"evidence_id": "source_clean_3", "kind": "source_clean", "use": "clean_visual"},
        ],
    }
    initial = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.9,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "insufficient",
            "same_image_embedding_evidence": "insufficient",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "glossary_or_guidance_used": True,
            "visible_target_cues": [
                "large rectangular target footprint",
                "flat top surface",
            ],
            "supporting_clean_evidence_ids": ["target_detail_2", "source_clean_3"],
            "rationale_short": "Target may fit SuggestedClass.",
            "counter_evidence": "Anchor cues are broad.",
            "human_review_needed": True,
        },
        result,
        point,
        {"target_detail_2", "source_clean_3"},
        clear_quality,
        evidence_ledger,
    )
    assert api._class_analysis_qwen_review_should_run_cue_verifier(initial) is True

    def fake_model_call(*args, **kwargs):
        return json.dumps(
            {
                "verified": True,
                "target_class": "SuggestedClass",
                "cue_confidence": 0.93,
                "positive_visible_target_cues": [
                    "large rectangular target footprint",
                    "flat top surface",
                ],
                "current_class_positive_cues": [],
                "current_class_plausibility_basis": "shared_generic_cues",
                "current_class_plausible": False,
                "current_class_plausibility_reason": "The visible cues are broad and shared.",
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "The proposed class explains the whole target extent.",
                "overlap_rebutted": True,
                "overlap_risk": "target_specific",
                "overlap_rebuttal": "Overlap does not explain the visible broad shape.",
                "anchor_support_verified": False,
                "anchor_support_basis": "shared_generic_anchors",
                "anchor_support_reason": "Trusted anchors only share broad footprint and surface cues.",
                "supporting_clean_evidence_ids": ["target_detail_2", "source_clean_3"],
                "rejection_reason": "Anchors are shared generic cues.",
            }
        )

    monkeypatch.setattr(api, "_class_analysis_qwen_review_model_call", fake_model_call)
    job = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_cue_verify_shared_anchor",
        parent_job_id=parent_id,
        point_id="p0",
        request={},
    )
    guarded = api._class_analysis_qwen_review_try_cue_verifier(
        job,
        final_result=initial,
        final_base_messages=[{"role": "user", "content": [{"type": "text", "text": "base"}]}],
        point=point,
        result=result,
        evidence_ids={"target_detail_2", "source_clean_3"},
        visual_quality=clear_quality,
        evidence_ledger=evidence_ledger,
        labelmap_glossary="",
        review_guidance="",
        deterministic_context={},
        model_id="test-model",
        executed_tools={"inspect_target_detail", "inspect_source_overlay"},
        labelmap=["CurrentClass", "SuggestedClass"],
    )

    assert guarded["decision"] == "skip_uncertain"
    assert guarded["cue_verifier"]["verified"] is False
    assert "target-specific anchor support" in guarded["cue_verifier"]["rejection_reason"]


def test_class_analysis_qwen_review_cue_verifier_blocks_shared_generic_without_support(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_cue_verify_generic"
    (class_root / parent_id).mkdir(parents=True)
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    evidence_ledger = {
        "clean_visual_evidence_ids": ["target_context_1", "zoom_region_8"],
        "clean_target_source_evidence_ids": ["target_context_1", "zoom_region_8"],
        "rows": [
            {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
            {"evidence_id": "zoom_region_8", "kind": "zoom_region", "use": "clean_visual"},
        ],
    }
    initial = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.91,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "insufficient",
            "same_image_embedding_evidence": "insufficient",
            "glossary_or_guidance_used": False,
            "visible_target_cues": ["generic target shape"],
            "supporting_clean_evidence_ids": ["target_context_1"],
            "rationale_short": "Target uses generic shape language.",
            "counter_evidence": "CurrentClass is not independently excluded.",
            "human_review_needed": True,
        },
        result,
        point,
        {"target_context_1", "zoom_region_8"},
        clear_quality,
        evidence_ledger,
    )
    assert api._class_analysis_qwen_review_should_run_cue_verifier(initial) is True

    def fake_model_call(*args, **kwargs):
        return json.dumps(
            {
                "verified": True,
                "target_class": "SuggestedClass",
                "cue_confidence": 0.93,
                "positive_visible_target_cues": [
                    "generic rectangular target outline",
                    "flat top surface",
                    "stationary placement",
                ],
                "current_class_positive_cues": [],
                "current_class_plausibility_basis": "shared_generic_cues",
                "current_class_plausible": False,
                "current_class_plausibility_reason": "The cues are generic and shared rather than current-class-specific.",
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "The proposed class explains the whole target extent.",
                "overlap_rebutted": False,
                "overlap_risk": "not_applicable",
                "overlap_rebuttal": "",
                "supporting_clean_evidence_ids": ["target_context_1", "zoom_region_8"],
                "rejection_reason": "",
            }
        )

    monkeypatch.setattr(api, "_class_analysis_qwen_review_model_call", fake_model_call)
    job = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_cue_verify_generic",
        parent_job_id=parent_id,
        point_id="p0",
        request={},
    )
    guarded = api._class_analysis_qwen_review_try_cue_verifier(
        job,
        final_result=initial,
        final_base_messages=[{"role": "user", "content": [{"type": "text", "text": "base"}]}],
        point=point,
        result=result,
        evidence_ids={"target_context_1", "zoom_region_8"},
        visual_quality=clear_quality,
        evidence_ledger=evidence_ledger,
        labelmap_glossary="",
        review_guidance="",
        deterministic_context={},
        model_id="test-model",
        executed_tools={"inspect_target_context", "zoom_source_region"},
        labelmap=["CurrentClass", "SuggestedClass"],
    )

    assert guarded["decision"] == "skip_uncertain"
    assert guarded["cue_verifier"]["verified"] is False
    assert "shared generic" in guarded["cue_verifier"]["rejection_reason"]


def test_class_analysis_qwen_review_cue_verifier_instruction_names_strict_schema():
    instruction = api._class_analysis_qwen_review_cue_verifier_instruction(
        point={"class_name": "CurrentClass"},
        guarded_recommendation={
            "target_class": "SuggestedClass",
            "visible_target_cues": ["ribbed surface texture"],
            "rationale_short": "Target visibly fits SuggestedClass.",
            "guardrail_reasons": ["moderate suggested-anchor agreement"],
        },
        evidence_ledger={
            "clean_target_source_evidence_ids": ["target_detail_2", "zoom_region_9"],
        },
    )
    text = instruction["content"][0]["text"]

    for field_name in api.CLASS_ANALYSIS_QWEN_REVIEW_CUE_VERIFIER_REQUIRED_FIELDS:
        assert field_name in text
    for field_name in api.CLASS_ANALYSIS_QWEN_REVIEW_CUE_VERIFIER_OPTIONAL_FIELDS:
        assert field_name in text
    assert '"target_class": "SuggestedClass"' in text
    assert "Do not include legacy or diagnostic keys" in text
    assert "current_class, proposed_target_class, verified_evidence_ids" in text
    assert "Use supporting_clean_evidence_ids, not verified_evidence_ids." in text
    assert "whole reviewed bbox/object extent" in text
    assert "Output compact JSON" in text
    assert "Do not copy the same cue string into multiple arrays" in text
    assert "Optional keys, use only when they add non-duplicative validation evidence" in text
    assert "0.92 not 0. 92" in text
    assert "under 18 words" in text
    assert "subcomponent" in text


def test_class_analysis_qwen_review_cue_verifier_tool_schema_keeps_duplicate_cues_optional():
    spec = api._class_analysis_qwen_review_cue_verifier_tool_spec(["CurrentClass", "SuggestedClass"])
    required = spec["parameters"]["required"]

    assert required == list(api.CLASS_ANALYSIS_QWEN_REVIEW_CUE_VERIFIER_REQUIRED_FIELDS)
    for field_name in api.CLASS_ANALYSIS_QWEN_REVIEW_CUE_VERIFIER_OPTIONAL_FIELDS:
        assert field_name in spec["parameters"]["properties"]
        assert field_name not in required
    assert spec["parameters"]["properties"]["positive_visible_target_cues"]["items"]["maxLength"] == 90


def test_class_analysis_qwen_review_cue_verifier_accepts_compact_required_payload():
    parsed, error = api._class_analysis_qwen_review_parse_cue_verifier_payload(
        json.dumps(
            {
                "verified": True,
                "target_class": "SuggestedClass",
                "cue_confidence": 0.91,
                "positive_visible_target_cues": [
                    "round target wheel",
                    "upright handlebar",
                ],
                "current_class_plausibility_basis": "none",
                "current_class_plausible": False,
                "current_class_plausibility_reason": "No direct current cue.",
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "Target class explains full extent.",
                "overlap_rebutted": True,
                "overlap_risk": "target_specific",
                "anchor_support_verified": False,
                "anchor_support_basis": "not_applicable",
                "supporting_clean_evidence_ids": ["target_detail_2", "zoom_region_9"],
                "rejection_reason": "",
            }
        ),
        current_class="CurrentClass",
        target_class="SuggestedClass",
        evidence_ids={"target_detail_2", "zoom_region_9"},
    )

    assert error is None
    assert parsed["verified"] is True
    assert parsed["target_class_defining_cues"] == []
    assert parsed["current_class_positive_cues"] == []
    assert parsed["current_class_missing_or_inconsistent_cues"] == []


def test_class_analysis_qwen_review_cue_verifier_repairs_numeric_whitespace():
    raw = """
    {
      "verified": true,
      "target_class": "SuggestedClass",
      "cue_confidence": 0. 92,
      "positive_visible_target_cues": ["rectangular target body", "ribbed surface texture"],
      "target_class_defining_cues": ["rectangular target body", "ribbed surface texture"],
      "current_class_positive_cues": [],
      "current_class_missing_or_inconsistent_cues": ["no visible current-class parts"],
      "current_class_plausibility_basis": "none",
      "current_class_plausible": false,
      "current_class_plausibility_reason": "",
      "whole_target_extent_supported": true,
      "whole_target_extent_reason": "SuggestedClass explains the full target.",
      "overlap_rebutted": true,
      "overlap_risk": "target_specific",
      "overlap_rebuttal": "Target cues are visible inside the clean crop.",
      "anchor_support_verified": true,
      "anchor_support_basis": "target_specific_anchors",
      "anchor_support_reason": "Anchors share the same target structure.",
      "supporting_clean_evidence_ids": ["target_detail_2", "zoom_region_9"],
      "rejection_reason": ""
    }
    """

    payload, error = api._class_analysis_qwen_review_parse_cue_verifier_payload(
        raw,
        current_class="CurrentClass",
        target_class="SuggestedClass",
        evidence_ids={"target_detail_2", "zoom_region_9"},
    )

    assert error is None
    assert payload["cue_confidence"] == pytest.approx(0.92)
    assert payload["verified"] is True


def test_class_analysis_qwen_review_cue_verifier_refuses_partial_subcomponent_extent():
    parsed, error = api._class_analysis_qwen_review_parse_cue_verifier_payload(
        json.dumps(
            {
                "verified": True,
                "target_class": "SuggestedClass",
                "cue_confidence": 0.93,
                "positive_visible_target_cues": [
                    "compact front cabin",
                    "distinct hood boundary",
                ],
                "current_class_positive_cues": [],
                "current_class_plausibility_basis": "none",
                "current_class_plausible": False,
                "current_class_plausibility_reason": "No direct current-class cue is visible.",
                "whole_target_extent_supported": False,
                "whole_target_extent_reason": (
                    "The proposed class explains only the front subcomponent, not the large attached body "
                    "inside the same bbox."
                ),
                "overlap_rebutted": True,
                "overlap_risk": "target_specific",
                "overlap_rebuttal": "Overlap does not explain the front cabin cues.",
                "anchor_support_verified": True,
                "anchor_support_basis": "target_specific_anchors",
                "anchor_support_reason": "Anchors share the front-cabin appearance.",
                "supporting_clean_evidence_ids": ["target_detail_2", "source_clean_3"],
                "rejection_reason": "",
            }
        ),
        current_class="CurrentClass",
        target_class="SuggestedClass",
        evidence_ids={"target_detail_2", "source_clean_3"},
    )

    assert error is None
    assert parsed["verified"] is False
    assert parsed["whole_target_extent_supported"] is False
    assert "front subcomponent" in parsed["rejection_reason"]


def test_class_analysis_qwen_review_cue_verifier_reconciles_contrastive_target_support():
    parsed, error = api._class_analysis_qwen_review_parse_cue_verifier_payload(
        json.dumps(
            {
                "verified": False,
                "target_class": "SuggestedClass",
                "cue_confidence": 0.9,
                "positive_visible_target_cues": [
                    "rectangular roof footprint",
                    "ribbed roof texture",
                ],
                "target_class_defining_cues": [
                    "ribbed roof panels",
                    "flat building-like roof plane",
                ],
                "current_class_positive_cues": ["rectangular footprint"],
                "current_class_missing_or_inconsistent_cues": [
                    "no rounded end caps",
                    "no cylindrical body surface",
                ],
                "current_class_plausibility_basis": "shared_generic_cues",
                "current_class_plausible": False,
                "current_class_plausibility_reason": (
                    "Only the rectangular footprint is shared; no current-class-specific parts are visible."
                ),
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "The target class explains the full roof-like extent.",
                "overlap_rebutted": True,
                "overlap_risk": "target_specific",
                "overlap_rebuttal": "Overlap does not explain the ribbed roof panels.",
                "anchor_support_verified": True,
                "anchor_support_basis": "target_specific_anchors",
                "anchor_support_reason": "Trusted anchors share ribbed roof panels and flat roof planes.",
                "supporting_clean_evidence_ids": ["target_detail_2", "source_clean_3"],
                "rejection_reason": "",
            }
        ),
        current_class="CurrentClass",
        target_class="SuggestedClass",
        evidence_ids={"target_detail_2", "source_clean_3"},
    )

    assert error is None
    assert parsed["raw_verified"] is False
    assert parsed["verified"] is True
    assert parsed["reconciled_to_verified"] is True
    assert parsed["contrastively_supported_target"] is True
    assert parsed["target_defining_cue_count"] >= 2
    assert "no rounded end caps" in parsed["current_class_missing_or_inconsistent_cues"]


def test_class_analysis_qwen_review_filters_context_only_verifier_cues():
    parsed, error = api._class_analysis_qwen_review_parse_cue_verifier_payload(
        json.dumps(
            {
                "verified": True,
                "target_class": "SuggestedClass",
                "cue_confidence": 0.92,
                "positive_visible_target_cues": [
                    "ribbed roof panels",
                    "parked next to other objects",
                    "flat roof plane",
                ],
                "target_class_defining_cues": [
                    "ribbed roof panels",
                    "flat roof plane",
                ],
                "current_class_positive_cues": [],
                "current_class_missing_or_inconsistent_cues": [
                    "absence of water or outdoor environment",
                    "no rounded end caps",
                ],
                "current_class_plausibility_basis": "none",
                "current_class_plausible": False,
                "current_class_plausibility_reason": "No direct current-class cue is visible.",
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "The proposed class explains the whole target extent.",
                "overlap_rebutted": True,
                "overlap_risk": "target_specific",
                "overlap_rebuttal": "Overlap does not explain the roof panels.",
                "anchor_support_verified": True,
                "anchor_support_basis": "target_specific_anchors",
                "anchor_support_reason": "Anchors share ribbed panels and flat roof planes.",
                "supporting_clean_evidence_ids": ["target_detail_2", "source_clean_3"],
                "rejection_reason": "",
            }
        ),
        current_class="CurrentClass",
        target_class="SuggestedClass",
        evidence_ids={"target_detail_2", "source_clean_3"},
    )

    assert error is None
    assert "parked next to other objects" not in parsed["positive_visible_target_cues"]
    assert "absence of water or outdoor environment" not in parsed["current_class_missing_or_inconsistent_cues"]
    assert "no rounded end caps" in parsed["current_class_missing_or_inconsistent_cues"]


def test_class_analysis_qwen_review_cue_verifier_repairs_partial_schema(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_cue_verify_repair"
    (class_root / parent_id).mkdir(parents=True)
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    evidence_ledger = {
        "clean_visual_evidence_ids": ["target_context_1", "zoom_region_8"],
        "clean_target_source_evidence_ids": ["target_context_1", "zoom_region_8"],
        "rows": [
            {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
            {"evidence_id": "zoom_region_8", "kind": "zoom_region", "use": "clean_visual"},
        ],
    }
    initial = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.91,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": False,
            "visible_target_cues": ["rectangular target body"],
            "supporting_clean_evidence_ids": ["target_context_1"],
            "rationale_short": "Target visibly fits SuggestedClass.",
            "counter_evidence": "CurrentClass cues are not visible.",
            "human_review_needed": True,
        },
        result,
        point,
        {"target_context_1", "zoom_region_8"},
        clear_quality,
        evidence_ledger,
    )
    assert api._class_analysis_qwen_review_should_run_cue_verifier(initial) is True

    outputs = [
        '{"verified": false, "cue_confidence": 0.75}',
        json.dumps(
            {
                "verified": True,
                "target_class": "SuggestedClass",
                "cue_confidence": 0.93,
                "positive_visible_target_cues": [
                    "rectangular target body",
                    "ribbed surface texture",
                ],
                "current_class_positive_cues": [],
                "current_class_plausible": False,
                "current_class_plausibility_reason": "Clean target pixels do not match the current class concept.",
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "The proposed class explains the whole target extent.",
                "overlap_rebutted": False,
                "overlap_risk": "not_applicable",
                "overlap_rebuttal": "",
                "supporting_clean_evidence_ids": ["target_context_1", "zoom_region_8"],
                "rejection_reason": "",
            }
        ),
    ]
    calls = []

    def fake_model_call(*args, **kwargs):
        calls.append(kwargs)
        return outputs.pop(0)

    monkeypatch.setattr(api, "_class_analysis_qwen_review_model_call", fake_model_call)
    job = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_cue_verify_repair",
        parent_job_id=parent_id,
        point_id="p0",
        request={},
    )
    promoted = api._class_analysis_qwen_review_try_cue_verifier(
        job,
        final_result=initial,
        final_base_messages=[{"role": "user", "content": [{"type": "text", "text": "base"}]}],
        point=point,
        result=result,
        evidence_ids={"target_context_1", "zoom_region_8"},
        visual_quality=clear_quality,
        evidence_ledger=evidence_ledger,
        labelmap_glossary="",
        review_guidance="",
        deterministic_context={},
        model_id="test-model",
        executed_tools={"inspect_target_context", "zoom_source_region"},
        labelmap=["CurrentClass", "SuggestedClass"],
    )

    assert [call["phase"] for call in calls] == ["cue_verifier", "cue_verifier_repair"]
    assert promoted["decision"] == "accept_suggested"
    assert promoted["cue_verifier"]["promoted_from_guarded_recommendation"] is True


def test_class_analysis_qwen_review_accepts_one_cue_with_independent_support():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 120.0,
        "bbox_height": 90.0,
        "bbox_min_dim": 90.0,
        "bbox_area": 10800.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.95,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "near_context",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "supports_suggested",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "neutral",
            "same_image_embedding_evidence": "questions_current",
            "glossary_or_guidance_used": False,
            "visible_target_cues": ["compact target body"],
            "supporting_clean_evidence_ids": ["target_context_1", "zoom_region_8"],
            "rationale_short": "Target has one clear cue and independent local support.",
            "counter_evidence": "CurrentClass cues are weak.",
            "human_review_needed": True,
        },
        result,
        point,
        {"target_context_1", "zoom_region_8"},
        clear_quality,
    )

    assert final["decision"] == "accept_suggested"
    assert final["confidence"] == 0.86
    assert "one concrete visible cue" in " ".join(final["advisory_reasons"])


def test_class_analysis_qwen_review_blocks_one_cue_without_independent_support():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 120.0,
        "bbox_height": 90.0,
        "bbox_min_dim": 90.0,
        "bbox_area": 10800.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.95,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "near_context",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "neutral",
            "same_image_embedding_evidence": "neutral",
            "glossary_or_guidance_used": False,
            "visible_target_cues": ["compact target body"],
            "supporting_clean_evidence_ids": ["target_context_1"],
            "rationale_short": "Target has only one cue.",
            "counter_evidence": "CurrentClass cues are weak.",
            "human_review_needed": True,
        },
        result,
        point,
        {"target_context_1", "zoom_region_8"},
        clear_quality,
    )

    assert final["decision"] == "skip_uncertain"
    assert any("at least two concrete visible target cues" in item for item in final["guardrail_reasons"])


def test_class_analysis_qwen_review_cue_verifier_refuses_current_class_cues():
    parsed, error = api._class_analysis_qwen_review_parse_cue_verifier_payload(
        json.dumps(
            {
                "verified": True,
                "target_class": "SuggestedClass",
                "cue_confidence": 0.96,
                "positive_visible_target_cues": [
                    "rectangular target body",
                    "ribbed surface texture",
                ],
                "current_class_positive_cues": ["round current-class wheel"],
                "current_class_plausibility_basis": "direct_positive_cues",
                "current_class_plausible": True,
                "current_class_plausibility_reason": "A current-class wheel is visible in the clean target pixels.",
                "whole_target_extent_supported": True,
                "whole_target_extent_reason": "The proposed class explains the whole target extent.",
                "overlap_rebutted": False,
                "overlap_risk": "not_applicable",
                "overlap_rebuttal": "",
                "supporting_clean_evidence_ids": ["target_context_1"],
                "rejection_reason": "",
            }
        ),
        current_class="CurrentClass",
        target_class="SuggestedClass",
        evidence_ids={"target_context_1"},
    )

    assert error is None
    assert parsed["verified"] is False
    assert "current-class wheel" in parsed["rejection_reason"]


def test_class_analysis_qwen_review_disposition_separates_guarded_signal():
    disposition = api._class_analysis_qwen_review_disposition(
        {
            "decision": "skip_uncertain",
            "target_class": "CurrentClass",
            "current_class": "CurrentClass",
            "suggested_neighbor_class": "SuggestedClass",
            "visual_quality": "limited",
            "object_visibility": "partial",
            "guardrail_reasons": [
                "accept_suggested requires clear backend visual-quality tier, got limited"
            ],
            "guarded_recommendation": {
                "blocked": True,
                "decision": "accept_suggested",
                "target_class": "SuggestedClass",
                "confidence": 0.84,
                "backend_tier": "limited",
                "visual_quality": "limited",
                "object_visibility": "partial",
                "target_evidence": "strong",
                "current_evidence": "weak",
                "guardrail_reasons": [
                    "accept_suggested requires clear backend visual-quality tier, got limited"
                ],
            },
            "specificity_probe": {
                "status": "completed",
                "specificity_alignment": "supports_suggested",
                "target_background_contrast": "target_specific",
                "specificity_margin": "suggested_target_favored",
                "target_identity_uncertainty": "low",
            },
        }
    )

    assert disposition["signal"] == "guarded_human_triage"
    assert disposition["disposition"] == "guarded_visual_quality"
    assert disposition["signal_strength"] == "strong"
    assert disposition["priority"] == "high"
    assert disposition["label"].startswith("Strong guarded signal")
    assert disposition["advisory_target_class"] == "SuggestedClass"


def test_class_analysis_qwen_review_disposition_prioritizes_specificity_conflict():
    disposition = api._class_analysis_qwen_review_disposition(
        {
            "decision": "skip_uncertain",
            "target_class": "CurrentClass",
            "current_class": "CurrentClass",
            "suggested_neighbor_class": "SuggestedClass",
            "visual_quality": "clear",
            "object_visibility": "clear",
            "guardrail_reasons": [
                "class change contradicts Qwen specificity probe: target/background contrast is background_dominated"
            ],
            "guarded_recommendation": {
                "blocked": True,
                "decision": "accept_suggested",
                "target_class": "SuggestedClass",
                "confidence": 0.86,
                "backend_tier": "clear",
                "visual_quality": "clear",
                "object_visibility": "clear",
                "target_evidence": "strong",
                "current_evidence": "weak",
                "guardrail_reasons": [
                    "class change contradicts Qwen specificity probe: target/background contrast is background_dominated"
                ],
            },
            "specificity_probe": {
                "status": "completed",
                "specificity_alignment": "insufficient",
                "target_background_contrast": "background_dominated",
                "specificity_margin": "background_or_overlap_favored",
                "target_identity_uncertainty": "moderate",
            },
        }
    )

    assert disposition["signal"] == "guarded_human_triage"
    assert disposition["disposition"] == "guarded_specificity_conflict"
    assert "specificity probe" in disposition["label"]
    assert disposition["advisory_target_class"] == "SuggestedClass"


def test_class_analysis_qwen_review_disposition_marks_verified_guarded_limited_signal():
    disposition = api._class_analysis_qwen_review_disposition(
        {
            "decision": "skip_uncertain",
            "target_class": "CurrentClass",
            "current_class": "CurrentClass",
            "suggested_neighbor_class": "SuggestedClass",
            "visual_quality": "limited",
            "object_visibility": "partial",
            "guardrail_reasons": [
                "accept_suggested requires clear backend visual-quality tier, got limited"
            ],
            "guarded_recommendation": {
                "blocked": True,
                "decision": "accept_suggested",
                "target_class": "SuggestedClass",
                "confidence": 0.82,
                "backend_tier": "limited",
                "visual_quality": "limited",
                "object_visibility": "partial",
                "target_evidence": "strong",
                "current_evidence": "weak",
                "guardrail_reasons": [
                    "accept_suggested requires clear backend visual-quality tier, got limited"
                ],
            },
            "cue_verifier": {
                "verified": True,
                "cue_confidence": 0.91,
            },
            "specificity_probe": {
                "status": "completed",
                "specificity_alignment": "supports_suggested",
                "target_background_contrast": "target_specific",
                "specificity_margin": "suggested_target_favored",
                "target_identity_uncertainty": "moderate",
            },
        }
    )

    assert disposition["signal"] == "guarded_human_triage"
    assert disposition["disposition"] == "guarded_visual_quality"
    assert disposition["signal_strength"] == "moderate"
    assert disposition["label"].startswith("Verified guarded signal")


def test_class_analysis_qwen_review_disposition_marks_useful_negative_verifier():
    disposition = api._class_analysis_qwen_review_disposition(
        {
            "decision": "skip_uncertain",
            "target_class": "CurrentClass",
            "current_class": "CurrentClass",
            "cue_verifier": {
                "verified": False,
                "rejection_reason": "Verifier did not find concrete target cues.",
            },
        }
    )

    assert disposition["signal"] == "useful_negative"
    assert disposition["disposition"] == "verified_no_class_change"


def test_class_analysis_qwen_review_disposition_marks_current_overlap_false_alarm():
    disposition = api._class_analysis_qwen_review_disposition(
        {
            "decision": "skip_uncertain",
            "target_class": "CurrentClass",
            "current_class": "CurrentClass",
            "suggested_neighbor_class": "SuggestedClass",
            "visual_quality": "clear",
            "object_visibility": "clear",
            "guardrail_reasons": [
                "accept_suggested conflicts with overlap decomposition: current class CurrentClass dominates the target bbox (partial_contamination, current_cover=0.63, target_class_cover=0.15)"
            ],
            "guarded_recommendation": {
                "blocked": True,
                "decision": "accept_suggested",
                "target_class": "SuggestedClass",
                "backend_tier": "clear",
                "visual_quality": "clear",
                "object_visibility": "clear",
                "target_evidence": "strong",
                "guardrail_reasons": [
                    "accept_suggested conflicts with overlap decomposition: current class CurrentClass dominates the target bbox (partial_contamination, current_cover=0.63, target_class_cover=0.15)"
                ],
            },
        }
    )

    assert disposition["signal"] == "useful_negative"
    assert disposition["disposition"] == "verified_current_class_overlap"
    assert disposition["advisory_decision"] == "confirm_current"
    assert disposition["advisory_target_class"] == "CurrentClass"


def test_class_analysis_qwen_review_does_not_infer_other_label_from_text_without_decision():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass", "OtherClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.9,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "visible_target_cues": ["distinct target silhouette", "surface texture matches suggested anchors"],
            "rationale_short": "Target is a clear SuggestedClass example.",
            "counter_evidence": "OtherClass may share context, but the target pixels do not show it.",
            "human_review_needed": False,
        },
        result,
        point,
        {"ctx_1"},
        clear_quality,
    )

    assert final["decision"] == "accept_suggested"
    assert final["target_class"] == "SuggestedClass"
    assert not final["guardrail_reasons"]


def test_class_analysis_qwen_review_blocks_partial_overlap_accept_without_strong_suggested_anchor():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass", "OtherClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.85,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": (
                "Target clearly matches SuggestedClass. CurrentClass is only broad compatibility. "
                "Overlap is partial but does not explain target features."
            ),
            "counter_evidence": "OtherClass is not visually unambiguous.",
            "human_review_needed": False,
        },
        result,
        point,
        {"ctx_1"},
        clear_quality,
    )

    assert final["decision"] == "skip_uncertain"
    assert final["target_class"] == "CurrentClass"
    assert final["guarded_recommendation"]["decision"] == "accept_suggested"
    assert any("partial_contamination" in reason for reason in final["guardrail_reasons"])


def test_class_analysis_qwen_review_caps_clear_accept_with_moderate_suggested_anchor():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.9,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "current_class_plausible": False,
            "current_class_plausibility_reason": "Clean target pixels do not fit CurrentClass.",
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["target_context_1", "zoom_region_6"],
            "visible_target_cues": ["distinct target silhouette", "surface texture matches trusted anchors"],
            "supporting_clean_evidence_ids": ["target_context_1", "zoom_region_6"],
            "rationale_short": "Target pixels visibly fit SuggestedClass better.",
            "counter_evidence": "CurrentClass anchors do not match the target.",
            "human_review_needed": False,
        },
        result,
        point,
        {"target_context_1", "zoom_region_6"},
        clear_quality,
        evidence_ledger={
            "rows": [
                {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
                {"evidence_id": "zoom_region_6", "kind": "zoom_region", "use": "clean_visual"},
                {"evidence_id": "class_context_pack_5", "kind": "class_context_pack", "use": "clean_visual"},
            ],
            "clean_visual_evidence_ids": ["target_context_1", "zoom_region_6", "class_context_pack_5"],
            "clean_target_source_evidence_ids": ["target_context_1", "zoom_region_6"],
        },
    )

    assert final["decision"] == "accept_suggested"
    assert final["target_class"] == "SuggestedClass"
    assert final["human_review_needed"] is True
    assert final["confidence"] <= 0.72
    assert final["guarded_recommendation"] is None
    assert not final["guardrail_reasons"]
    assert any("moderate suggested-anchor agreement" in reason for reason in final["advisory_reasons"])


def test_class_analysis_qwen_review_blocks_class_change_with_label_only_visible_cues():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.9,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "visible_target_cues": ["SuggestedClass", "matches suggested class"],
            "evidence_ids": ["ctx_1"],
            "rationale_short": "Target matches SuggestedClass.",
            "counter_evidence": "No explicit counterevidence provided.",
            "human_review_needed": False,
        },
        result,
        point,
        {"ctx_1"},
        clear_quality,
    )

    assert final["decision"] == "skip_uncertain"
    assert final["visible_target_cues"] == []
    assert any("visible target cues" in reason for reason in final["guardrail_reasons"])


def test_class_analysis_qwen_review_ignores_context_only_visible_cues_for_class_change():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    ledger = {
        "rows": [
            {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
            {"evidence_id": "zoom_region_6", "kind": "zoom_region", "use": "clean_visual"},
        ],
        "clean_visual_evidence_ids": ["target_context_1", "zoom_region_6"],
        "clean_target_source_evidence_ids": ["target_context_1", "zoom_region_6"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.9,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "visible_target_cues": [
                "top-down perspective",
                "parked on pavement",
                "ribbed target surface",
            ],
            "supporting_clean_evidence_ids": ["target_context_1", "zoom_region_6"],
            "evidence_ids": ["target_context_1", "zoom_region_6"],
            "rationale_short": "Target pixels support SuggestedClass.",
            "counter_evidence": "No explicit counterevidence provided.",
            "human_review_needed": False,
        },
        result,
        point,
        {"target_context_1", "zoom_region_6"},
        clear_quality,
        ledger,
    )

    assert final["decision"] == "skip_uncertain"
    assert final["visible_target_cues"] == ["ribbed target surface"]
    assert any("visible target cues" in reason for reason in final["guardrail_reasons"])


def test_class_analysis_qwen_review_ignores_negative_and_color_only_visible_cues():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    ledger = {
        "rows": [
            {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
            {"evidence_id": "zoom_region_6", "kind": "zoom_region", "use": "clean_visual"},
        ],
        "clean_visual_evidence_ids": ["target_context_1", "zoom_region_6"],
        "clean_target_source_evidence_ids": ["target_context_1", "zoom_region_6"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.9,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "supports_suggested",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "visible_target_cues": [
                "overhead view of parked candidate class",
                "multiple object colors",
                "flat ground surface",
                "no current-class features",
            ],
            "supporting_clean_evidence_ids": ["target_context_1", "zoom_region_6"],
            "evidence_ids": ["target_context_1", "zoom_region_6"],
            "rationale_short": "Target is suggested by nearby local consensus.",
            "counter_evidence": "No explicit counterevidence provided.",
            "human_review_needed": False,
        },
        result,
        point,
        {"target_context_1", "zoom_region_6"},
        clear_quality,
        ledger,
    )

    assert final["decision"] == "skip_uncertain"
    assert final["visible_target_cues"] == []
    assert any("visible target cues" in reason for reason in final["guardrail_reasons"])


def test_class_analysis_qwen_review_blocks_class_change_with_overlay_only_supporting_evidence():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    ledger = {
        "rows": [
            {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
            {"evidence_id": "source_overlay_3", "kind": "source_overlay", "use": "geometry_overlay"},
        ],
        "clean_visual_evidence_ids": ["target_context_1"],
        "geometry_overlay_evidence_ids": ["source_overlay_3"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.9,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "visible_target_cues": ["distinct target outline", "visible surface detail"],
            "supporting_clean_evidence_ids": ["source_overlay_3"],
            "evidence_ids": ["target_context_1", "source_overlay_3"],
            "rationale_short": "Target pixels support SuggestedClass.",
            "counter_evidence": "No explicit counterevidence provided.",
            "human_review_needed": False,
        },
        result,
        point,
        {"target_context_1", "source_overlay_3"},
        clear_quality,
        ledger,
    )

    assert final["decision"] == "skip_uncertain"
    assert final["guarded_recommendation"]["supporting_clean_evidence_ids"] == ["source_overlay_3"]
    assert any("clean visual evidence" in reason for reason in final["guardrail_reasons"])


def test_class_analysis_qwen_review_allows_class_change_with_clean_supporting_evidence():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    ledger = {
        "rows": [
            {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
            {"evidence_id": "source_overlay_3", "kind": "source_overlay", "use": "geometry_overlay"},
        ],
        "clean_visual_evidence_ids": ["target_context_1"],
        "geometry_overlay_evidence_ids": ["source_overlay_3"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.9,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "visible_target_cues": ["distinct target outline", "visible surface detail"],
            "supporting_clean_evidence_ids": ["target_context_1"],
            "evidence_ids": ["target_context_1", "source_overlay_3"],
            "rationale_short": "Target pixels support SuggestedClass.",
            "counter_evidence": "No explicit counterevidence provided.",
            "human_review_needed": False,
        },
        result,
        point,
        {"target_context_1", "source_overlay_3"},
        clear_quality,
        ledger,
    )

    assert final["decision"] == "accept_suggested"
    assert final["supporting_clean_evidence_ids"] == ["target_context_1"]
    assert final["guardrail_reasons"] == []


def test_class_analysis_qwen_review_blocks_accept_when_text_rejects_suggested_alias():
    result = {"summary": {"labelmap": ["Building", "SmallVehicle"]}}
    point = {
        "point_id": "p0",
        "class_name": "Building",
        "suggested_neighbor_class": "SmallVehicle",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    ledger = {
        "rows": [
            {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
            {"evidence_id": "zoom_region_6", "kind": "zoom_region", "use": "clean_visual"},
        ],
        "clean_visual_evidence_ids": ["target_context_1", "zoom_region_6"],
        "clean_target_source_evidence_ids": ["target_context_1", "zoom_region_6"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SmallVehicle",
            "confidence": 0.95,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "near_context",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "visible_target_cues": ["roofline structure", "overhead shadows"],
            "supporting_clean_evidence_ids": ["target_context_1", "zoom_region_6"],
            "evidence_ids": ["target_context_1", "zoom_region_6"],
            "rationale_short": "Target is a roof, not vehicle; suggested context is nearby.",
            "counter_evidence": "No explicit counterevidence provided.",
            "human_review_needed": False,
        },
        result,
        point,
        {"target_context_1", "zoom_region_6"},
        clear_quality,
        ledger,
    )

    assert final["decision"] == "skip_uncertain"
    assert final["guarded_recommendation"]["decision"] == "accept_suggested"
    assert any("rejecting suggested-class cue" in reason for reason in final["guardrail_reasons"])


def test_class_analysis_qwen_review_semantic_rejection_stops_at_semicolon_positive_cue():
    payload = {
        "decision": "accept_suggested",
        "target_class": "SmallVehicle",
        "rationale_short": (
            "Target is small, compact, no cargo; matches SmallVehicle visual cues; "
            "no overlap contamination"
        ),
        "counter_evidence": "No explicit counterevidence provided.",
        "visible_target_cues": ["Compact size"],
    }

    conflict = api._class_analysis_qwen_review_text_conflicts_with_accept_suggested(
        current_class="Truck",
        suggested_class="SmallVehicle",
        payload=payload,
        labelmap=["Truck", "SmallVehicle"],
    )

    assert conflict is None


def test_class_analysis_qwen_review_allows_partial_overlap_accept_with_strong_independent_evidence():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass", "OtherClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.85,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": (
                "Target clearly matches SuggestedClass. CurrentClass is only broad compatibility. "
                "Overlap is partial but does not explain target features."
            ),
            "counter_evidence": "No explicit counterevidence provided.",
            "human_review_needed": False,
        },
        result,
        point,
        {"ctx_1"},
        clear_quality,
    )

    assert final["decision"] == "accept_suggested"
    assert final["target_class"] == "SuggestedClass"
    assert final["guardrail_reasons"] == []


def test_class_analysis_qwen_review_blocks_class_change_on_limited_quality():
    result = {"summary": {"labelmap": ["PoleFixture", "Person"]}}
    point = {
        "point_id": "p0",
        "class_name": "PoleFixture",
        "suggested_neighbor_class": "Person",
    }
    limited_quality = {
        "tier": "limited",
        "bbox_width": 30.0,
        "bbox_height": 42.0,
        "bbox_min_dim": 30.0,
        "bbox_area": 1260.0,
        "crop_contrast": 28.0,
        "crop_dynamic_range": 80.0,
        "crop_sharpness": 11.0,
        "edge_clipped": False,
        "reasons": ["bbox area is limited"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "Person",
            "confidence": 0.82,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": "suggested class looks better",
            "human_review_needed": False,
        },
        result,
        point,
        {"ctx_1"},
        limited_quality,
    )

    assert final["decision"] == "skip_uncertain"
    assert final["confidence"] <= 0.45
    assert any(
        "accept_suggested is advisory-only because backend visual-quality tier is limited" in reason
        for reason in final["guardrail_reasons"]
    )
    assert any(
        "accept_suggested requires clear backend visual-quality tier" in reason
        for reason in final["guardrail_reasons"]
    )
    assert final["target_class"] == "PoleFixture"
    assert final["guarded_recommendation"]["blocked"] is True
    assert final["guarded_recommendation"]["decision"] == "accept_suggested"
    assert final["guarded_recommendation"]["target_class"] == "Person"
    assert final["guarded_recommendation"]["confidence"] == 0.82
    assert "suggested class looks better" in final["guarded_recommendation"]["rationale_short"]


def test_class_analysis_qwen_review_allows_limited_confirm_current_with_specificity_rebuttal():
    result = {"summary": {"labelmap": ["PoleFixture", "SmallVehicle"]}}
    point = {
        "point_id": "p0",
        "class_name": "PoleFixture",
        "suggested_neighbor_class": "SmallVehicle",
    }
    limited_quality = {
        "tier": "limited",
        "bbox_width": 28.0,
        "bbox_height": 44.0,
        "bbox_min_dim": 28.0,
        "bbox_area": 1232.0,
        "crop_contrast": 26.0,
        "crop_dynamic_range": 90.0,
        "crop_sharpness": 11.0,
        "edge_clipped": False,
        "reasons": ["bbox area is limited"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "confirm_current",
            "target_class": "PoleFixture",
            "confidence": 0.88,
            "visual_quality": "limited",
            "object_visibility": "partial",
            "current_evidence": "strong",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "moderate",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "moderate",
            "local_consensus_evidence": "supports_suggested",
            "global_context_evidence": "moderate",
            "same_image_scale_evidence": "neutral",
            "same_image_embedding_evidence": "neutral",
            "specificity_alignment": "supports_current",
            "target_background_contrast": "target_specific",
            "target_identity_summary": "target pixels show a narrow vertical object",
            "target_identity_uncertainty": "moderate",
            "whole_target_extent_supported": True,
            "dual_bbox_resolution": "not_applicable",
            "visible_target_cues": ["narrow vertical shaft", "top fixture", "standing object outline"],
            "supporting_clean_evidence_ids": ["target_context_1"],
            "target_identity_evidence_ids": ["target_context_1"],
            "glossary_or_guidance_used": True,
            "evidence_ids": ["target_context_1"],
            "rationale_short": "The target pixels support the current class despite neighbor similarity.",
            "counter_evidence": "The suggested class evidence comes from neighboring context.",
            "human_review_needed": True,
        },
        result,
        point,
        {"target_context_1"},
        limited_quality,
        {
            "specificity_probe": {
                "status": "completed",
                "confidence": 0.86,
                "specificity_alignment": "supports_current",
                "target_background_contrast": "target_specific",
                "specificity_margin": "current_target_favored",
                "best_supported_class": "PoleFixture",
            },
        },
    )

    assert final["decision"] == "confirm_current"
    assert final["target_class"] == "PoleFixture"
    assert final["guardrail_reasons"] == []
    assert final["guarded_recommendation"] is None
    assert final["confidence"] <= 0.65
    assert final["human_review_needed"] is True
    assert any("backend visual-quality tier is limited" in reason for reason in final["advisory_reasons"])
    assert any("specificity probe supports the current target" in reason for reason in final["advisory_reasons"])


def test_class_analysis_qwen_review_promotes_limited_partial_cue_verified_change_with_one_context_veto():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    limited_quality = {
        "tier": "limited",
        "bbox_width": 34.0,
        "bbox_height": 48.0,
        "bbox_min_dim": 34.0,
        "bbox_area": 1632.0,
        "crop_contrast": 28.0,
        "crop_dynamic_range": 88.0,
        "crop_sharpness": 12.0,
        "edge_clipped": False,
        "reasons": ["reviewable but limited target"],
    }
    evidence_ledger = {
        "clean_visual_evidence_ids": ["target_detail_2", "source_clean_3", "zoom_region_10"],
        "clean_target_source_evidence_ids": ["target_detail_2", "source_clean_3", "zoom_region_10"],
        "rows": [
            {"evidence_id": "target_detail_2", "kind": "target_detail", "use": "clean_visual"},
            {"evidence_id": "source_clean_3", "kind": "source_clean", "use": "clean_visual"},
            {"evidence_id": "zoom_region_10", "kind": "zoom_region", "use": "clean_visual"},
        ],
        "specificity_probe": {
            "status": "completed",
            "confidence": 0.86,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "specificity_margin": "suggested_target_favored",
            "best_supported_class": "SuggestedClass",
        },
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "_expanded_by_controller": True,
            "_cue_verifier_class_change_verified": True,
            "_cue_verifier_confidence": 0.93,
            "_cue_verifier_overlap_rebutted": True,
            "_cue_verifier_overlap_risk": "target_specific",
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.9,
            "visual_quality": "limited",
            "object_visibility": "partial",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "anchor_adjudication_verified": True,
            "local_context_evidence": "strong",
            "global_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "same_image_scale_evidence": "neutral",
            "same_image_embedding_evidence": "supports_current",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "target_identity_summary": "clean target pixels show ridged texture, bracket lattice, and a continuous body",
            "target_identity_uncertainty": "moderate",
            "target_identity_evidence_ids": ["target_detail_2", "source_clean_3"],
            "whole_target_extent_supported": True,
            "whole_target_extent_reason": "The suggested class explains the whole reviewed target extent.",
            "overlap_assessment": "unclear",
            "overlap_explains_candidate_similarity": True,
            "overlap_adjudication_verified": True,
            "dual_bbox_resolution": "not_applicable",
            "visible_target_cues": ["spiral conduit ridges", "triangular bracket lattice", "translucent membrane fold"],
            "supporting_clean_evidence_ids": ["target_detail_2", "source_clean_3"],
            "current_class_plausible": False,
            "current_class_plausibility_reason": "Clean pixels lack current-class structure.",
            "glossary_or_guidance_used": False,
            "rationale_short": "Cue verifier finds target-specific suggested-class cues.",
            "human_review_needed": True,
        },
        result,
        point,
        {"target_detail_2", "source_clean_3", "zoom_region_10"},
        limited_quality,
        evidence_ledger,
    )

    assert final["decision"] == "accept_suggested"
    assert final["target_class"] == "SuggestedClass"
    assert final["guardrail_reasons"] == []
    assert final["guarded_recommendation"] is None
    assert final["confidence"] <= 0.65
    assert final["human_review_needed"] is True
    assert any("backend visual-quality tier is limited" in reason for reason in final["advisory_reasons"])
    assert any("cue verifier rebuts overlap" in reason for reason in final["advisory_reasons"])
    assert any("same-image embedding report supports the current class" in reason for reason in final["advisory_reasons"])


def test_class_analysis_qwen_review_blocks_limited_partial_change_when_multiple_context_reports_support_current():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    limited_quality = {
        "tier": "limited",
        "bbox_width": 34.0,
        "bbox_height": 48.0,
        "bbox_min_dim": 34.0,
        "bbox_area": 1632.0,
        "crop_contrast": 28.0,
        "crop_dynamic_range": 88.0,
        "crop_sharpness": 12.0,
        "edge_clipped": False,
        "reasons": ["reviewable but limited target"],
    }
    evidence_ledger = {
        "clean_visual_evidence_ids": ["target_detail_2", "source_clean_3"],
        "clean_target_source_evidence_ids": ["target_detail_2", "source_clean_3"],
        "rows": [
            {"evidence_id": "target_detail_2", "kind": "target_detail", "use": "clean_visual"},
            {"evidence_id": "source_clean_3", "kind": "source_clean", "use": "clean_visual"},
        ],
        "specificity_probe": {
            "status": "completed",
            "confidence": 0.86,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "specificity_margin": "suggested_target_favored",
            "best_supported_class": "SuggestedClass",
        },
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "_expanded_by_controller": True,
            "_cue_verifier_class_change_verified": True,
            "_cue_verifier_confidence": 0.94,
            "_cue_verifier_overlap_rebutted": True,
            "_cue_verifier_overlap_risk": "target_specific",
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.9,
            "visual_quality": "limited",
            "object_visibility": "partial",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "anchor_adjudication_verified": True,
            "local_context_evidence": "strong",
            "global_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "same_image_scale_evidence": "supports_current",
            "same_image_embedding_evidence": "supports_current",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "target_identity_summary": "clean target pixels show ridged texture, bracket lattice, and a continuous body",
            "target_identity_uncertainty": "moderate",
            "target_identity_evidence_ids": ["target_detail_2", "source_clean_3"],
            "whole_target_extent_supported": True,
            "whole_target_extent_reason": "The suggested class explains the whole reviewed target extent.",
            "overlap_assessment": "unclear",
            "overlap_explains_candidate_similarity": True,
            "overlap_adjudication_verified": True,
            "dual_bbox_resolution": "not_applicable",
            "visible_target_cues": ["spiral conduit ridges", "triangular bracket lattice", "translucent membrane fold"],
            "supporting_clean_evidence_ids": ["target_detail_2", "source_clean_3"],
            "current_class_plausible": False,
            "current_class_plausibility_reason": "Clean pixels lack current-class structure.",
            "glossary_or_guidance_used": False,
            "rationale_short": "Cue verifier finds target-specific suggested-class cues.",
            "human_review_needed": True,
        },
        result,
        point,
        {"target_detail_2", "source_clean_3"},
        limited_quality,
        evidence_ledger,
    )

    assert final["decision"] == "skip_uncertain"
    assert final["target_class"] == "CurrentClass"
    assert final["guarded_recommendation"]["decision"] == "accept_suggested"
    assert any("backend visual-quality tier is limited" in reason for reason in final["guardrail_reasons"])


def test_class_analysis_qwen_review_blocks_self_conflicting_class_recommendations():
    result = {"summary": {"labelmap": ["Truck", "SmallVehicle", "Container"]}}
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    accepted = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SmallVehicle",
            "confidence": 0.9,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "strong",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "strong",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": "suggested class looks better",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p0", "class_name": "Truck", "suggested_neighbor_class": "SmallVehicle"},
        {"ctx_1"},
        clear_quality,
    )
    confirmed = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "confirm_current",
            "target_class": "Truck",
            "confidence": 0.9,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "strong",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "strong",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": "current class looks better",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p1", "class_name": "Truck", "suggested_neighbor_class": "Container"},
        {"ctx_1"},
        clear_quality,
    )

    assert accepted["decision"] == "skip_uncertain"
    assert accepted["confidence"] <= 0.45
    assert "accept_suggested cannot override current_evidence=strong" in accepted["guardrail_reasons"]
    assert confirmed["decision"] == "skip_uncertain"
    assert confirmed["confidence"] <= 0.45
    assert (
        "confirm_current cannot override target-contained suggested_evidence=strong without overlap/near-context rebuttal"
        in confirmed["guardrail_reasons"]
    )


def test_class_analysis_qwen_review_blocks_accept_when_model_text_rejects_suggested_class():
    result = {"summary": {"labelmap": ["Boat", "Building", "SmallVehicle", "Truck"]}}
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    contradictory = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SmallVehicle",
            "confidence": 0.85,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": "The target crop clearly shows a small boat, contradicting the SmallVehicle suggestion.",
            "counter_evidence": "The object is clearly a small boat, not a car or light vehicle.",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p0", "class_name": "Boat", "suggested_neighbor_class": "SmallVehicle"},
        {"ctx_1"},
        clear_quality,
    )
    good_relabel = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "Building",
            "confidence": 0.85,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": "The target is a small red shed with a roof, clearly a Building. The current SmallVehicle label is incorrect.",
            "counter_evidence": "No vehicle features are visible.",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p1", "class_name": "SmallVehicle", "suggested_neighbor_class": "Building"},
        {"ctx_1"},
        clear_quality,
    )
    good_contradicting_current_label = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "Building",
            "confidence": 0.9,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": "The target crop clearly shows a building roof with vents, contradicting the Truck label.",
            "counter_evidence": "No vehicle features are visible.",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p2", "class_name": "Truck", "suggested_neighbor_class": "Building"},
        {"ctx_1"},
        clear_quality,
    )

    assert contradictory["decision"] == "skip_uncertain"
    assert contradictory["confidence"] <= 0.45
    assert any("model text" in reason for reason in contradictory["guardrail_reasons"])
    assert good_relabel["decision"] == "accept_suggested"
    assert good_relabel["target_class"] == "Building"
    assert good_contradicting_current_label["decision"] == "accept_suggested"
    assert good_contradicting_current_label["target_class"] == "Building"


def test_class_analysis_qwen_review_sentence_bounds_model_text_fields():
    result = {"summary": {"labelmap": ["Building", "SmallVehicle", "Truck"]}}
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    ledger = {
        "rows": [
            {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
            {"evidence_id": "zoom_region_8", "kind": "zoom_region", "use": "clean_visual"},
        ],
        "clean_visual_evidence_ids": ["target_context_1", "zoom_region_8"],
        "clean_target_source_evidence_ids": ["target_context_1", "zoom_region_8"],
    }

    accepted = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "Building",
            "confidence": 0.95,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "visible_target_cues": ["Flat roof structure", "Structural walls"],
            "supporting_clean_evidence_ids": ["target_context_1", "zoom_region_8"],
            "evidence_ids": ["target_context_1", "zoom_region_8"],
            "rationale_short": (
                "Target shows clear building features; no truck-like cargo or chassis; "
                "no overlap contamination."
            ),
            "counter_evidence": "",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p2", "class_name": "Truck", "suggested_neighbor_class": "Building"},
        {"target_context_1", "zoom_region_8"},
        clear_quality,
        ledger,
    )

    assert accepted["decision"] == "accept_suggested"
    assert accepted["target_class"] == "Building"
    assert accepted["guardrail_reasons"] == []


def test_class_analysis_qwen_review_blocks_dominant_current_overlap():
    result = {"summary": {"labelmap": ["Building", "SmallVehicle"]}}
    clear_quality = {
        "tier": "clear",
        "bbox_width": 183.0,
        "bbox_height": 178.0,
        "bbox_min_dim": 178.0,
        "bbox_area": 32574.0,
        "crop_contrast": 37.5,
        "crop_dynamic_range": 154.0,
        "crop_sharpness": 9.3,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    ledger = {
        "rows": [
            {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
            {"evidence_id": "zoom_region_8", "kind": "zoom_region", "use": "clean_visual"},
            {"evidence_id": "overlap_decomposition_4", "kind": "overlap_decomposition", "use": "geometry_overlay"},
        ],
        "clean_visual_evidence_ids": ["target_context_1", "zoom_region_8"],
        "clean_target_source_evidence_ids": ["target_context_1", "zoom_region_8"],
        "overlap_decomposition": {
            "overlap_count": 1,
            "relation_counts": {"partial_contamination": 1},
            "overlaps": [
                {
                    "point_id": "current_building",
                    "class_name": "Building",
                    "relation": "partial_contamination",
                    "target_area_covered": 0.63,
                    "other_area_covered": 0.20,
                    "iou": 0.18,
                },
                {
                    "point_id": "neighbor_vehicle",
                    "class_name": "SmallVehicle",
                    "relation": "partial_contamination",
                    "target_area_covered": 0.15,
                    "other_area_covered": 0.31,
                    "iou": 0.11,
                }
            ],
        },
    }

    accepted = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SmallVehicle",
            "confidence": 0.95,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "supports_suggested",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "visible_target_cues": ["parked object shape", "bright vehicle roof"],
            "supporting_clean_evidence_ids": ["target_context_1", "zoom_region_8"],
            "evidence_ids": ["target_context_1", "zoom_region_8", "overlap_decomposition_4"],
            "rationale_short": "Target matches SmallVehicle traits; overlap does not explain vehicle features.",
            "counter_evidence": "",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p0", "class_name": "Building", "suggested_neighbor_class": "SmallVehicle"},
        {"target_context_1", "zoom_region_8", "overlap_decomposition_4"},
        clear_quality,
        ledger,
    )

    assert accepted["decision"] == "skip_uncertain"
    assert accepted["guarded_recommendation"]["decision"] == "accept_suggested"
    assert any("overlap decomposition" in reason for reason in accepted["guardrail_reasons"])


def test_class_analysis_qwen_review_allows_confirm_current_when_current_evidence_strong():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 120.0,
        "bbox_height": 180.0,
        "bbox_min_dim": 120.0,
        "bbox_area": 21600.0,
        "crop_contrast": 48.0,
        "crop_dynamic_range": 174.0,
        "crop_sharpness": 4.7,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "confirm_current",
            "target_class": "CurrentClass",
            "confidence": 0.85,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "strong",
            "suggested_evidence": "weak",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": True,
            "anchor_evidence_current": "strong",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["target_context_1"],
            "rationale_short": (
                "Target has CurrentClass-specific features; "
                "the nearby SuggestedClass object is the source of the suggestion."
            ),
            "human_review_needed": False,
        },
        result,
        point,
        {"target_context_1"},
        clear_quality,
    )

    assert final["decision"] == "confirm_current"
    assert final["target_class"] == "CurrentClass"
    assert final["guardrail_reasons"] == []


def test_class_analysis_qwen_review_allows_confirm_current_when_overlap_explains_strong_suggestion():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 120.0,
        "bbox_height": 180.0,
        "bbox_min_dim": 120.0,
        "bbox_area": 21600.0,
        "crop_contrast": 48.0,
        "crop_dynamic_range": 174.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "confirm_current",
            "target_class": "CurrentClass",
            "confidence": 0.88,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "strong",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": True,
            "anchor_evidence_current": "strong",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "supports_suggested",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "visible_target_cues": ["current-class shape", "current-class surface detail"],
            "supporting_clean_evidence_ids": ["target_context_1"],
            "evidence_ids": ["target_context_1"],
            "rationale_short": "Target shows current-class cues; overlap explains suggested-class signal.",
            "counter_evidence": "Suggested-class object is adjacent/overlapping, not the target.",
            "human_review_needed": False,
        },
        result,
        point,
        {"target_context_1"},
        clear_quality,
    )

    assert final["decision"] == "confirm_current"
    assert final["target_class"] == "CurrentClass"
    assert final["guardrail_reasons"] == []
    assert any("rebuts suggested_evidence=strong" in reason for reason in final["advisory_reasons"])
    assert any("local consensus supports the suggested class" in reason for reason in final["advisory_reasons"])


def test_class_analysis_qwen_review_allows_confirm_current_when_specificity_probe_rebuts_strong_suggestion():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 120.0,
        "bbox_height": 180.0,
        "bbox_min_dim": 120.0,
        "bbox_area": 21600.0,
        "crop_contrast": 48.0,
        "crop_dynamic_range": 174.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    evidence_ledger = {
        "specificity_probe": {
            "enabled": True,
            "status": "completed",
            "version": api.CLASS_ANALYSIS_QWEN_REVIEW_SPECIFICITY_PROBE_VERSION,
            "specificity_alignment": "supports_current",
            "target_background_contrast": "target_specific",
            "best_supported_class": "CurrentClass",
            "confidence": 0.91,
            "target_specific_cues": ["whole target outline", "distinct target surface"],
        },
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "confirm_current",
            "target_class": "CurrentClass",
            "confidence": 0.88,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "strong",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "strong",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "supports_suggested",
            "global_context_evidence": "strong",
            "specificity_alignment": "supports_current",
            "target_background_contrast": "target_specific",
            "glossary_or_guidance_used": True,
            "visible_target_cues": ["whole target outline", "distinct target surface"],
            "supporting_clean_evidence_ids": ["target_context_1"],
            "evidence_ids": ["target_context_1"],
            "rationale_short": "Target-specific probe supports the current class.",
            "counter_evidence": "Suggested evidence comes from neighbor context.",
            "human_review_needed": False,
        },
        result,
        point,
        {"target_context_1"},
        clear_quality,
        evidence_ledger,
    )

    assert final["decision"] == "confirm_current"
    assert final["target_class"] == "CurrentClass"
    assert final["guardrail_reasons"] == []
    assert any("specificity-probe support" in reason for reason in final["advisory_reasons"])
    assert any("specificity probe supports the current target" in reason for reason in final["advisory_reasons"])


def test_class_analysis_qwen_review_blocks_confirm_current_when_probe_favors_background():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 80.0,
        "bbox_height": 64.0,
        "bbox_min_dim": 64.0,
        "bbox_area": 5120.0,
        "crop_contrast": 42.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 14.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }
    evidence_ledger = {
        "specificity_probe": {
            "enabled": True,
            "status": "completed",
            "version": api.CLASS_ANALYSIS_QWEN_REVIEW_SPECIFICITY_PROBE_VERSION,
            "specificity_alignment": "insufficient",
            "target_background_contrast": "background_dominated",
            "specificity_margin": "background_or_overlap_favored",
            "best_supported_class": "",
            "confidence": 0.65,
            "target_specific_cues": ["generic target shape"],
        },
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "confirm_current",
            "target_class": "CurrentClass",
            "confidence": 0.72,
            "visual_quality": "clear",
            "object_visibility": "partial",
            "current_evidence": "strong",
            "suggested_evidence": "moderate",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "strong",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "specificity_alignment": "supports_current",
            "target_background_contrast": "target_specific",
            "glossary_or_guidance_used": True,
            "visible_target_cues": ["generic target shape"],
            "supporting_clean_evidence_ids": ["target_context_1"],
            "evidence_ids": ["target_context_1"],
            "rationale_short": "Final answer tries to confirm current despite a background-favored probe.",
            "counter_evidence": "The independent specificity probe did not find target-specific current evidence.",
            "human_review_needed": False,
        },
        result,
        point,
        {"target_context_1"},
        clear_quality,
        evidence_ledger,
    )

    assert final["decision"] == "skip_uncertain"
    assert final["guarded_recommendation"]["decision"] == "confirm_current"
    assert any("specificity probe" in reason for reason in final["guardrail_reasons"])
    assert any("background_or_overlap_favored" in reason for reason in final["guardrail_reasons"])


def test_class_analysis_qwen_review_controller_preflight_confirms_current_overlap_false_alarm():
    result = api._class_analysis_qwen_review_current_overlap_false_alarm_result(
        {"point_id": "p0", "class_name": "Building", "suggested_neighbor_class": "SmallVehicle"},
        {"tier": "clear"},
        {
            "clean_visual_evidence_ids": ["target_detail_2", "zoom_region_9"],
            "clean_target_source_evidence_ids": ["target_detail_2", "zoom_region_9"],
            "overlap_decomposition": {
                "overlaps": [
                    {
                        "class_name": "Building",
                        "relation": "partial_contamination",
                        "target_area_covered": 0.63,
                        "other_area_covered": 0.20,
                        "iou": 0.18,
                    },
                    {
                        "class_name": "SmallVehicle",
                        "relation": "partial_contamination",
                        "target_area_covered": 0.15,
                        "other_area_covered": 0.31,
                        "iou": 0.11,
                    },
                ]
            },
        },
    )

    assert result is not None
    assert result["decision"] == "confirm_current"
    assert result["target_class"] == "Building"
    assert result["controller_preflight"]["kind"] == "current_overlap_false_alarm"
    assert result["supporting_clean_evidence_ids"] == ["target_detail_2", "zoom_region_9"]


def test_class_analysis_qwen_review_controller_preflight_ignores_balanced_overlap():
    result = api._class_analysis_qwen_review_current_overlap_false_alarm_result(
        {"point_id": "p0", "class_name": "Building", "suggested_neighbor_class": "SmallVehicle"},
        {"tier": "clear"},
        {
            "overlap_decomposition": {
                "overlaps": [
                    {"class_name": "Building", "relation": "partial_contamination", "target_area_covered": 0.52},
                    {"class_name": "SmallVehicle", "relation": "partial_contamination", "target_area_covered": 0.35},
                ]
            },
        },
    )

    assert result is None


def test_class_analysis_qwen_review_confirm_current_does_not_require_named_class_pairs():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 260.0,
        "bbox_height": 120.0,
        "bbox_min_dim": 120.0,
        "bbox_area": 31200.0,
        "crop_contrast": 60.0,
        "crop_dynamic_range": 190.0,
        "crop_sharpness": 20.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "confirm_current",
            "target_class": "CurrentClass",
            "confidence": 0.84,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "strong",
            "suggested_evidence": "weak",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "strong",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["target_context_1"],
            "rationale_short": "Target matches CurrentClass-specific cues, not the suggested class.",
            "human_review_needed": False,
        },
        result,
        point,
        {"target_context_1"},
        clear_quality,
    )

    assert final["decision"] == "confirm_current"
    assert final["target_class"] == "CurrentClass"
    assert final["guardrail_reasons"] == []


def test_class_analysis_qwen_review_blocks_partial_overlap_accept_without_overlap_rebuttal():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 70.0,
        "bbox_height": 34.0,
        "bbox_min_dim": 34.0,
        "bbox_area": 2380.0,
        "crop_contrast": 60.0,
        "crop_dynamic_range": 190.0,
        "crop_sharpness": 25.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.68,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["target_context_1"],
            "rationale_short": "Target visibly matches SuggestedClass more than CurrentClass.",
            "counter_evidence": "No CurrentClass-specific cues are visible.",
            "human_review_needed": False,
        },
        result,
        point,
        {"target_context_1"},
        clear_quality,
    )

    assert final["decision"] == "skip_uncertain"
    assert final["target_class"] == "CurrentClass"
    assert final["guarded_recommendation"]["decision"] == "accept_suggested"
    assert any("partial_contamination" in reason for reason in final["guardrail_reasons"])


def test_class_analysis_qwen_review_allows_accept_with_decisive_suggested_cues():
    result = {"summary": {"labelmap": ["CurrentClass", "SuggestedClass"]}}
    point = {
        "point_id": "p0",
        "class_name": "CurrentClass",
        "suggested_neighbor_class": "SuggestedClass",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 70.0,
        "crop_dynamic_range": 200.0,
        "crop_sharpness": 28.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SuggestedClass",
            "confidence": 0.82,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["target_context_1"],
            "visible_target_cues": ["distinct target shape", "visible surface texture"],
            "rationale_short": "Target clearly shows SuggestedClass-specific cues; no CurrentClass cues are visible.",
            "counter_evidence": "No CurrentClass-specific cues.",
            "human_review_needed": False,
        },
        result,
        point,
        {"target_context_1"},
        clear_quality,
    )

    assert final["decision"] == "accept_suggested"
    assert final["target_class"] == "SuggestedClass"
    assert final["guardrail_reasons"] == []


def test_class_analysis_qwen_review_blocks_accept_when_counter_evidence_supports_current_class():
    result = {"summary": {"labelmap": ["PoleFixture", "SmallVehicle"]}}
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    accepted = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SmallVehicle",
            "confidence": 0.85,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": "Target crop clearly shows a car, so SmallVehicle is plausible.",
            "counter_evidence": "A thin pole-like structure is visible, which could justify the PoleFixture label.",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p0", "class_name": "PoleFixture", "suggested_neighbor_class": "SmallVehicle"},
        {"ctx_1"},
        clear_quality,
    )

    assert accepted["decision"] == "skip_uncertain"
    assert any("model text supporting current class PoleFixture" in reason for reason in accepted["guardrail_reasons"])

    plausible_current = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SmallVehicle",
            "confidence": 0.85,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "visible_target_cues": ["compact object body", "visible surface detail"],
            "rationale_short": "Target resembles the suggested class.",
            "counter_evidence": "Current class Truck is plausible from visible target structure.",
            "human_review_needed": False,
        },
        {"summary": {"labelmap": ["Truck", "SmallVehicle"]}},
        {"point_id": "p1", "class_name": "Truck", "suggested_neighbor_class": "SmallVehicle"},
        {"ctx_1"},
        clear_quality,
    )

    assert plausible_current["decision"] == "skip_uncertain"
    assert any("model text supporting current class Truck" in reason for reason in plausible_current["guardrail_reasons"])

    mixed_reject_and_support = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SmallVehicle",
            "confidence": 0.85,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "visible_target_cues": ["compact object body", "single unit"],
            "rationale_short": "Target is a compact white object, not a large Truck.",
            "counter_evidence": "Current class Truck is plausible due to visible target structure.",
            "human_review_needed": False,
        },
        {"summary": {"labelmap": ["Truck", "SmallVehicle"]}},
        {"point_id": "p2", "class_name": "Truck", "suggested_neighbor_class": "SmallVehicle"},
        {"ctx_1"},
        clear_quality,
    )

    assert mixed_reject_and_support["decision"] == "skip_uncertain"
    assert any("model text supporting current class Truck" in reason for reason in mixed_reject_and_support["guardrail_reasons"])


@pytest.mark.parametrize(
    ("current_class", "suggested_class", "rationale"),
    [
        (
            "Boat",
            "SmallVehicle",
            "The target is a small white boat on a trailer, visually matching SmallVehicle.",
        ),
        (
            "Truck",
            "SmallVehicle",
            "Target crop shows a clear truck with a cab and open bed, distinct from the nearby car.",
        ),
        (
            "StorageTank",
            "Building",
            "Target is a small residential tank, visually matching Building anchors.",
        ),
    ],
)
def test_class_analysis_qwen_review_blocks_accept_when_visible_text_identifies_current_class(
    current_class,
    suggested_class,
    rationale,
):
    result = {"summary": {"labelmap": [current_class, suggested_class]}}
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    accepted = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": suggested_class,
            "confidence": 0.85,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "supports_suggested",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": rationale,
            "counter_evidence": "No explicit counterevidence provided.",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p0", "class_name": current_class, "suggested_neighbor_class": suggested_class},
        {"ctx_1"},
        clear_quality,
    )

    assert accepted["decision"] == "skip_uncertain"
    assert any(
        "visible target text supporting current class" in reason
        or "partial_contamination" in reason
        for reason in accepted["guardrail_reasons"]
    )


def test_class_analysis_qwen_review_allows_adjacent_accept_when_text_downgrades_current_label():
    result = {"summary": {"labelmap": ["Truck", "SmallVehicle"]}}
    clear_quality = {
        "tier": "clear",
        "bbox_width": 120.0,
        "bbox_height": 90.0,
        "bbox_min_dim": 90.0,
        "bbox_area": 10800.0,
        "crop_contrast": 60.0,
        "crop_dynamic_range": 170.0,
        "crop_sharpness": 25.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    accepted = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SmallVehicle",
            "confidence": 0.85,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "supports_suggested",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": (
                "Target crop clearly shows a pickup truck with an open bed, fitting SmallVehicle. "
                "Current Truck label is broad and weak."
            ),
            "counter_evidence": "No explicit counterevidence provided.",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p0", "class_name": "Truck", "suggested_neighbor_class": "SmallVehicle"},
        {"ctx_1"},
        clear_quality,
    )

    assert accepted["decision"] == "accept_suggested"
    assert accepted["target_class"] == "SmallVehicle"
    assert not accepted["guardrail_reasons"]


@pytest.mark.parametrize(
    ("current_class", "suggested_class", "rationale", "counter_evidence"),
    [
        (
            "PoleFixture",
            "StorageTank",
            "Target is a clear horizontal tank (StorageTank). Current PoleFixture label is weak as it lacks vertical pole features.",
            "Current PoleFixture anchors show vertical poles, while the target is a horizontal tank.",
        ),
        (
            "Container",
            "Building",
            "Target is a clear residential roof (Building). Current class (Container) is weak and visually mismatched.",
            "Current class (Container) anchors are industrial, while the target is a clear residential roof.",
        ),
        (
            "Building",
            "SolarArray",
            "Target crop clearly shows a solar panel array with grid structure, distinct from the large building roofs labeled as Building.",
            "Local consensus shows 12 Building anchors, but the target crop matches SolarArray anchors.",
        ),
    ],
)
def test_class_analysis_qwen_review_allows_accept_when_current_text_is_anchor_or_rejected_label(
    current_class,
    suggested_class,
    rationale,
    counter_evidence,
):
    result = {"summary": {"labelmap": [current_class, suggested_class]}}
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    accepted = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": suggested_class,
            "confidence": 0.85,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "supports_suggested",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "visible_target_cues": ["distinct target shape", "visible surface texture"],
            "rationale_short": rationale,
            "counter_evidence": counter_evidence,
            "human_review_needed": False,
        },
        result,
        {"point_id": "p0", "class_name": current_class, "suggested_neighbor_class": suggested_class},
        {"ctx_1"},
        clear_quality,
    )

    assert accepted["decision"] == "accept_suggested"
    assert accepted["target_class"] == suggested_class


def test_class_analysis_qwen_review_allows_rebutted_partial_overlap_for_clear_target():
    result = {"summary": {"labelmap": ["Container", "Building"]}}
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    accepted = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "Building",
            "confidence": 0.85,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": True,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": (
                "Target is a small shed with a pitched roof, visually a Building. "
                "Overlap with a larger Building box is present but does not explain "
                "the target's own building features."
            ),
            "counter_evidence": "No explicit counterevidence provided.",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p0", "class_name": "Container", "suggested_neighbor_class": "Building"},
        {"ctx_1"},
        clear_quality,
    )

    assert accepted["decision"] == "accept_suggested"
    assert accepted["target_class"] == "Building"
    assert accepted["confidence"] == 0.68
    assert any("partial overlap present" in reason for reason in accepted["advisory_reasons"])


def test_class_analysis_qwen_review_allows_background_element_partial_overlap_rebuttal():
    result = {"summary": {"labelmap": ["PoleFixture", "SmallVehicle"]}}
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    accepted = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SmallVehicle",
            "confidence": 0.85,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": (
                "Target crop clearly shows a light vehicle. Current PoleFixture label is weak; "
                "the vertical pole is a minor background element and overlap does not explain target features."
            ),
            "counter_evidence": "No explicit counterevidence provided.",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p0", "class_name": "PoleFixture", "suggested_neighbor_class": "SmallVehicle"},
        {"ctx_1"},
        clear_quality,
    )

    assert accepted["decision"] == "accept_suggested"
    assert accepted["target_class"] == "SmallVehicle"
    assert any("partial overlap present" in reason for reason in accepted["advisory_reasons"])


def test_class_analysis_qwen_review_allows_background_overlap_not_vehicle_rebuttal():
    result = {"summary": {"labelmap": ["PoleFixture", "SmallVehicle"]}}
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    accepted = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SmallVehicle",
            "confidence": 0.85,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": (
                "Target crop clearly shows a sedan. Overlap is background road markings, "
                "not a vehicle, and does not explain target features."
            ),
            "counter_evidence": "No explicit counterevidence provided.",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p0", "class_name": "PoleFixture", "suggested_neighbor_class": "SmallVehicle"},
        {"ctx_1"},
        clear_quality,
    )

    assert accepted["decision"] == "accept_suggested"
    assert accepted["target_class"] == "SmallVehicle"


def test_class_analysis_qwen_review_allows_minor_partial_overlap_wording():
    result = {"summary": {"labelmap": ["Container", "Building"]}}
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "Building",
            "confidence": 0.85,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": "Target is a small residential building with a pitched roof. Overlap is minor.",
            "counter_evidence": "No explicit counterevidence provided.",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p0", "class_name": "Container", "suggested_neighbor_class": "Building"},
        {"ctx_1"},
        clear_quality,
    )

    assert final["decision"] == "accept_suggested"
    assert final["target_class"] == "Building"
    assert final["confidence"] == 0.68


def test_class_analysis_qwen_review_allows_adjacent_not_target_overlap_wording():
    result = {"summary": {"labelmap": ["Container", "Building"]}}
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "Building",
            "confidence": 0.85,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "visible_target_cues": ["fixed roof plane", "rectangular roof edge"],
            "rationale_short": "Target is a clear building roof. Overlapping containers are adjacent, not the target itself.",
            "counter_evidence": "No explicit counterevidence provided.",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p0", "class_name": "Container", "suggested_neighbor_class": "Building"},
        {"ctx_1"},
        clear_quality,
    )

    assert final["decision"] == "accept_suggested"
    assert final["target_class"] == "Building"
    assert final["confidence"] == 0.68


def test_class_analysis_qwen_review_blocks_partial_overlap_when_model_says_overlap_explains():
    result = {"summary": {"labelmap": ["PoleFixture", "StorageTank"]}}
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "StorageTank",
            "confidence": 0.85,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": True,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "not_applicable",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": "Target crop shows a clear horizontal tank structure caused by overlap.",
            "counter_evidence": "No explicit counterevidence provided.",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p0", "class_name": "PoleFixture", "suggested_neighbor_class": "StorageTank"},
        {"ctx_1"},
        clear_quality,
    )

    assert final["decision"] == "skip_uncertain"
    assert any("partial_contamination" in reason for reason in final["guardrail_reasons"])


def test_class_analysis_qwen_review_blocks_partial_overlap_without_explicit_rebuttal():
    result = {"summary": {"labelmap": ["Boat", "SmallVehicle"]}}
    clear_quality = {
        "tier": "clear",
        "bbox_width": 120.0,
        "bbox_height": 70.0,
        "bbox_min_dim": 70.0,
        "bbox_area": 8400.0,
        "crop_contrast": 55.0,
        "crop_dynamic_range": 150.0,
        "crop_sharpness": 20.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    accepted = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SmallVehicle",
            "confidence": 0.84,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "weak",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "supports_suggested",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": "Target crop clearly shows a car. Current Boat class is weak.",
            "counter_evidence": "No explicit counterevidence provided.",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p0", "class_name": "Boat", "suggested_neighbor_class": "SmallVehicle"},
        {"ctx_1"},
        clear_quality,
    )

    assert accepted["decision"] == "skip_uncertain"
    assert any("partial_contamination" in reason for reason in accepted["guardrail_reasons"])


def test_class_analysis_qwen_review_local_consensus_guardrails():
    result = {"summary": {"labelmap": ["PoleFixture", "SmallVehicle"]}}
    point = {
        "point_id": "p0",
        "class_name": "PoleFixture",
        "suggested_neighbor_class": "SmallVehicle",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    accepted = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "SmallVehicle",
            "confidence": 0.84,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "moderate",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "moderate",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "moderate",
            "local_consensus_evidence": "supports_current",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": "suggested class looks better",
            "human_review_needed": False,
        },
        result,
        point,
        {"ctx_1"},
        clear_quality,
    )
    confirmed = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "confirm_current",
            "target_class": "PoleFixture",
            "confidence": 0.84,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "strong",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "strong",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "moderate",
            "local_consensus_evidence": "supports_suggested",
            "global_context_evidence": "strong",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": "current class looks better",
            "human_review_needed": False,
        },
        result,
        point,
        {"ctx_1"},
        clear_quality,
    )

    assert accepted["decision"] == "skip_uncertain"
    assert "accept_suggested conflicts with local_consensus_evidence=supports_current" in accepted["guardrail_reasons"]
    assert accepted["local_consensus_evidence"] == "supports_current"
    assert confirmed["decision"] == "skip_uncertain"
    assert "confirm_current conflicts with local_consensus_evidence=supports_suggested" in confirmed["guardrail_reasons"]
    assert confirmed["local_consensus_evidence"] == "supports_suggested"


def test_class_analysis_qwen_review_caps_direct_uncertain_skip_confidence():
    result = {"summary": {"labelmap": ["Boat", "Person"]}}
    point = {
        "point_id": "p0",
        "class_name": "Boat",
        "suggested_neighbor_class": "Person",
    }
    clear_quality = {
        "tier": "clear",
        "bbox_width": 90.0,
        "bbox_height": 60.0,
        "bbox_min_dim": 60.0,
        "bbox_area": 5400.0,
        "crop_contrast": 50.0,
        "crop_dynamic_range": 160.0,
        "crop_sharpness": 18.0,
        "edge_clipped": False,
        "reasons": ["usable"],
    }

    final = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "skip_uncertain",
            "target_class": "Boat",
            "confidence": 0.8,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "moderate",
            "suggested_evidence": "moderate",
            "target_evidence": "moderate",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "anchor_evidence_current": "moderate",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "moderate",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "moderate",
            "glossary_or_guidance_used": True,
            "evidence_ids": ["ctx_1"],
            "rationale_short": "ambiguous target",
            "human_review_needed": True,
        },
        result,
        point,
        {"ctx_1"},
        clear_quality,
    )

    assert final["decision"] == "skip_uncertain"
    assert final["confidence"] == pytest.approx(0.5)
    assert final["guardrail_reasons"] == []


def test_class_analysis_qwen_review_loop_enforces_evidence_and_writes_artifacts(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_parent"
    workspace_dir = class_root / parent_id / "active_workspace"
    images_dir = workspace_dir / "images"
    images_dir.mkdir(parents=True)
    for filename, color in (("target.jpg", (40, 60, 80)), ("boat.jpg", (40, 80, 180))):
        image = Image.new("RGB", (220, 180), color)
        draw = ImageDraw.Draw(image)
        for x in range(0, 220, 16):
            draw.line([(x, 0), (x, 180)], fill=(180, 200, 220), width=3)
        for y in range(0, 180, 18):
            draw.line([(0, y), (220, y)], fill=(20, 30, 40), width=2)
        image.save(images_dir / filename)
    api._class_analysis_write_json(
        workspace_dir / "manifest.json",
        workspace_dir,
        {
            "labelmap": ["car", "boat"],
            "images": [
                {"split": "train", "image_relpath": "target.jpg", "label_lines": ["0 0.5 0.5 0.4 0.4"]},
                {"split": "train", "image_relpath": "boat.jpg", "label_lines": ["1 0.5 0.5 0.4 0.4"]},
            ],
            "yolo_layout": "flat",
            "source_mode": "active_workspace",
        },
    )
    result = {
        "summary": {
            "source_mode": "active_workspace",
            "source_id": parent_id,
            "dataset_label": "test workspace",
            "labelmap": ["car", "boat"],
            "analysis_scope": "all_classes",
        },
        "points": [
            {
                "point_id": "p0",
                "class_name": "car",
                "suggested_neighbor_class": "boat",
                "wrong_class_suspicion": 0.91,
                "same_class_neighbor_ratio": 0.0,
                "top_other_neighbor_ratio": 1.0,
                "neighbor_class_counts": {"boat": 3},
                "neighbor_ids": ["p1"],
                "neighbor_distances": [0.12],
                "image_relpath": "target.jpg",
                "split": "train",
                "bbox_xyxy": [40, 35, 130, 120],
                "is_wrong_class_candidate": True,
                # The deterministic rail favors keeping the current label,
                # while the mocked VLM below independently chooses the
                # suggested class. The final assertion protects the product
                # invariant that Stage 2 cannot replace the VLM judgment.
                    "refined_outlier": {
                        "schema": api.CLASS_ANALYSIS_REFINEMENT_SCHEMA,
                        "decision_contract": (
                            api.CLASS_ANALYSIS_REFINEMENT_DECISION_CONTRACT
                        ),
                        "status": "explained_not_outlier",
                    "reason_codes": [
                        "current_spatial_evidence_supported"
                    ],
                    "current_class": "car",
                    "alternative_class": "boat",
                        "current_support_score": 0.52,
                        "alternative_support_score": 0.12,
                        "intrinsic_current_support": 0.52,
                        "intrinsic_alternative_support": 0.12,
                        "directed_pair_raw_margin": -0.40,
                        "directed_pair_probe_score": -0.35,
                        "directed_pair_probe_features": [0.85, 0.2],
                        "directed_pair_probe_feature_names": [
                            "current_patch_exclusive_support",
                            "alternative_patch_exclusive_support",
                        ],
                        "directed_pair_current_exclusive_support": 0.85,
                        "directed_pair_alternative_exclusive_support": 0.2,
                        "directed_pair_probe_threshold": 0.10,
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
                        "directed_pair_probe_fold_digest": "ef" * 32,
                        "directed_pair_probe_fit_eval_split_digest": "ef" * 32,
                        "current_negative_threshold": 0.08,
                        "current_support_threshold": 0.15,
                        "current_strong_threshold": 0.25,
                        "alternative_negative_threshold": 0.09,
                        "alternative_support_threshold": 0.15,
                        "alternative_strong_threshold": 0.25,
                        "support_threshold_source": "fit_only_directed_pair",
                        "directed_pair_reliable": True,
                        "directed_pair_bank_reliable": True,
                        "directed_pair_candidate_source_excluded": False,
                        "directed_pair_candidate_source_fingerprint": "56" * 8,
                        "directed_pair_candidate_source_membership_roles": [],
                        "directed_pair_heldout_auroc": 0.85,
                        "directed_pair_eval_auroc_lower_bound": 0.68,
                        "positive_confirmation_pair_probe_auroc_floor": 0.8,
                        "positive_confirmation_pair_probe_auroc_lower_bound_floor": 0.6,
                        "directed_pair_probe_fit_current_source_count": 12,
                        "directed_pair_probe_fit_alternative_source_count": 13,
                        "directed_pair_probe_eval_current_source_count": 9,
                        "directed_pair_probe_eval_alternative_source_count": 10,
                        "directed_pair_probe_fit_balanced_accuracy": 0.76,
                        "directed_pair_probe_eval_sensitivity": 0.72,
                        "directed_pair_probe_eval_specificity": 0.74,
                        "directed_pair_current_absence_eval_fraction": 0.66,
                        "directed_pair_alternative_strong_eval_fraction": 0.69,
                        "directed_pair_current_source_count": 21,
                        "directed_pair_alternative_source_count": 29,
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
                        "decision_gates": {
                            "diagnostic_pair_reliable": True,
                            "intrinsic_references_reliable": True,
                            "positive_confirmation_pair_reliable": True,
                            "directed_pair_reliable": True,
                            "directed_pair_candidate_source_independent": True,
                            "directed_pair_exact_calibration_contracts": True,
                            "positive_confirmation_pair_probe_auroc_sufficient": True,
                            "positive_confirmation_pair_probe_lower_bound_sufficient": True,
                            "source_resolution_sufficient": True,
                            "current_absent": False,
                            "alternative_strong": False,
                            "directed_pair_dominates": False,
                            "alternative_exclusive_component_corresponds": False,
                            "view_consistent": True,
                            "alternative_evidence_external_to_overlap": True,
                            "qualified_for_human_review": False,
                        },
                    "alternative_evidence_inside_overlap_fraction": 0.0,
                    "alternative_evidence_outside_overlap_fraction": 1.0,
                    "current_evidence_outside_overlap_fraction": 1.0,
                    "overlap_relation": "none",
                    "overlap_object_count": 0,
                    "reference_reliable": True,
                    "reference_distinct_source_count": 40,
                    "current_reference_tier": "high",
                    "alternative_reference_tier": "high",
                    "current_reference_heldout_auroc": 0.88,
                    "alternative_reference_heldout_auroc": 0.84,
                    "view_agreement": 0.91,
                    "sidecar_row": None,
                },
            },
            {
                "point_id": "p1",
                "class_name": "boat",
                "image_relpath": "boat.jpg",
                "split": "train",
                "bbox_xyxy": [45, 40, 150, 130],
            },
        ],
        "wrong_class_candidates": [{"point_id": "p0", "class_name": "car", "suggested_neighbor_class": "boat"}],
    }
    parent = api.ClassAnalysisJob(job_id=parent_id, status="completed", result=result)
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS[parent_id] = parent
    outputs = iter(
        [
            '{"target_identity_summary":"elongated bright target with visible grid texture","target_identity_uncertainty":"low","specificity_alignment":"supports_suggested","target_background_contrast":"target_specific","best_supported_class":"boat","target_specific_cues":["elongated bright target shape","visible grid texture"],"background_or_overlap_cues":[],"subdescription_assessments":[{"class_name":"car","subdescription":"vehicle-like fixture body","target_support":"weak","background_or_overlap_support":"none","support_location":"target","supporting_clean_evidence_ids":["target_detail_2"],"note":"only generic body shape is visible"},{"class_name":"boat","subdescription":"elongated bright grid-textured target","target_support":"strong","background_or_overlap_support":"none","support_location":"target","supporting_clean_evidence_ids":["target_detail_2","specificity_region_contrast_7","zoom_region_10"],"note":"visible on the reviewed target"}],"specificity_margin":"suggested_target_favored","margin_rationale":"target descriptors favor the suggested class","current_class_cues":[],"suggested_class_cues":["elongated bright target shape","visible grid texture"],"whole_target_extent_supported":true,"supporting_clean_evidence_ids":["target_detail_2","specificity_region_contrast_7","zoom_region_10"],"confidence":0.88,"rationale_short":"target pixels fit suggested class"}',
            '<tool_call>{"name":"route_review","arguments":{"action":"inspect_local_consensus_context","reason_code":"needs_same_image_consensus","confidence":0.78,"rationale_short":"same-image consensus may resolve this"}}</tool_call>',
            "{}}",
            '{"decision":"accept_suggested","target_class":"boat","confidence":0.82,"visual_quality":"clear","object_visibility":"clear","current_evidence":"weak","suggested_evidence":"strong","target_evidence":"strong","anchor_evidence_current":"weak","anchor_evidence_suggested":"strong","local_context_evidence":"strong","global_context_evidence":"strong","same_image_scale_evidence":"insufficient","same_image_embedding_evidence":"insufficient","overlap_assessment":"none","overlap_explains_candidate_similarity":false,"specificity_alignment":"supports_suggested","target_background_contrast":"target_specific","target_identity_summary":"elongated bright target with visible grid texture","target_identity_uncertainty":"low","target_identity_evidence_ids":["target_detail_2","specificity_region_contrast_7","zoom_region_10"],"whole_target_extent_supported":true,"whole_target_extent_reason":"the suggested class explains the full target extent","local_consensus_evidence":"mixed","visible_target_cues":["elongated bright target shape","visible grid texture"],"supporting_clean_evidence_ids":["target_detail_2","specificity_region_contrast_7","zoom_region_10"],"rationale_short":"target evidence and anchors fit better","counter_evidence":"synthetic fixture","human_review_needed":false}',
        ]
    )
    calls = []

    def fake_qwen_chat_stream(messages, **kwargs):
        calls.append({"messages": copy.deepcopy(messages), "kwargs": dict(kwargs)})
        yield next(outputs)

    monkeypatch.setattr(api, "_run_qwen_chat_stream", fake_qwen_chat_stream)
    monkeypatch.setattr(api, "_ensure_qwen_ready_for_caption", lambda *_args, **_kwargs: object())
    real_refinement_observation = (
        api._class_analysis_qwen_review_refinement_observation
    )

    def validated_refinement_observation(job, point):
        observation = real_refinement_observation(job, point)
        assert observation is not None
        observation["preview_status"] = "validated"
        observation["preview_integrity"] = {
            "status": "validated",
            "validation": "synthetic_unit_fixture",
        }
        return observation

    monkeypatch.setattr(
        api,
        "_class_analysis_qwen_review_refinement_observation",
        validated_refinement_observation,
    )
    unloads = []
    monkeypatch.setattr(
        api,
        "_unload_qwen_runtime",
        lambda: unloads.append("unload"),
    )
    review = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_test",
        parent_job_id=parent_id,
        point_id="p0",
        request={
            "max_turns": 8,
            "model_id": "test-model",
            "enable_local_consensus_context": True,
            "keep_vlm_loaded": True,
            "reset_qwen_runtime_after_review": True,
        },
    )
    with api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS_LOCK:
        api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS[review.review_id] = review

    api._run_class_analysis_qwen_review_job(review)

    assert unloads == []
    assert review.result["runtime_retention"]["keep_vlm_loaded"] is True
    assert review.result["runtime_retention"]["post_review_reset"] is False
    assert len(calls) == 4
    assert calls[0]["kwargs"].get("assistant_prefix") is None
    assert calls[-1]["kwargs"].get("assistant_prefix") is None
    assert all(call["kwargs"].get("chat_template_kwargs") == {"enable_thinking": False} for call in calls)
    assert all("tools" not in call["kwargs"] for call in calls)
    assert calls[0]["kwargs"].get("max_new_tokens") == 800
    assert calls[1]["kwargs"].get("max_new_tokens") == 1000
    assert calls[-1]["kwargs"].get("max_new_tokens") == 1000
    assert not any(message.get("role") == "assistant" for message in calls[-1]["messages"])
    specificity_prompt_text = "\n".join(
        str(item.get("text") or "")
        for message in calls[0]["messages"]
        for item in (message.get("content") or [])
        if isinstance(item, dict) and item.get("type") == "text"
    )
    assert "Specificity probe state" in specificity_prompt_text
    assert "target/background" in specificity_prompt_text
    assert "Required JSON skeleton and key order" in specificity_prompt_text
    assert '"target_identity_summary"' in specificity_prompt_text
    assert '"subdescription_assessments"' in specificity_prompt_text
    assert "V3 adds explicit region-contrast evidence" in specificity_prompt_text
    assert "specificity_region_contrast evidence" in specificity_prompt_text
    assert '"supporting_clean_evidence_ids"' in specificity_prompt_text
    assert "Scene, location, medium, surface, lighting, and nearby-object cues are context" in specificity_prompt_text
    assert "Switch blockers / hard negatives" in specificity_prompt_text
    final_prompt_text = "\n".join(
        str(item.get("text") or "")
        for message in calls[1]["messages"]
        for item in (message.get("content") or [])
        if isinstance(item, dict) and item.get("type") == "text"
    )
    assert "inspect_overlap_decomposition" in final_prompt_text
    assert "inspect_class_context_pack" in final_prompt_text
    assert "inspect_specificity_region_contrast" in final_prompt_text
    assert "inspect_target_detail" in final_prompt_text
    assert "zoom_source_region with draw_bbox=false" in final_prompt_text or "zoom_source_region(draw_bbox=false)" in final_prompt_text
    first_user_text = "\n".join(
        content.get("text") or ""
        for message in calls[0]["messages"]
        for content in message.get("content", [])
        if isinstance(content, dict)
    )
    assert "Router state" not in first_user_text
    assert "local_consensus_context_" in first_user_text
    assert any(
        content.get("type") == "image"
        for message in calls[0]["messages"]
        for content in message.get("content", [])
        if isinstance(content, dict)
    )
    final_user_text = "\n".join(
        content.get("text") or ""
        for message in calls[-1]["messages"]
        for content in message.get("content", [])
        if isinstance(content, dict)
    )
    final_image_values = [
        str(content.get("image") or "")
        for message in calls[-1]["messages"]
        for content in message.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    ]
    assert "compact arguments object" in final_user_text
    assert "Controller evidence ledger" in final_user_text
    assert "Clean visual evidence ids" in final_user_text
    assert "Use clean target/source/zoom pixels for visible_target_cues" in final_user_text
    assert "Use same-image scale and embedding reports to guide visual attention" in final_user_text
    assert "specificity_alignment" in final_user_text
    assert "target_background_contrast" in final_user_text
    assert "specificity_region_contrast panel" in final_user_text
    assert "Scene, location, medium, surface, lighting, and nearby-object cues are context" in final_user_text
    assert "Switch blockers / hard negatives" in final_user_text
    assert "Specificity probe result" in final_user_text
    assert "Probe target-specific cues" in final_user_text
    assert "Probe sub-description assessments" in final_user_text
    assert "Probe specificity margin" in final_user_text
    assert "supporting_clean_evidence_ids" in final_user_text
    assert "Local consensus evidence has been inspected" in final_user_text
    assert "previous final response failed validation" in final_user_text
    assert len(final_image_values) <= 4
    assert not any("source_clean_" in image for image in final_image_values)
    assert not any("class_context_pack_" in image for image in final_image_values)
    assert not any("local_consensus_context_" in image for image in final_image_values)
    assert any("local_consensus_dot_map_" in image for image in final_image_values)
    assert any("specificity_region_contrast_" in image for image in final_image_values)
    assert review.status == "completed"
    assert review.result["decision"] == "accept_suggested"
    assert (
        review.result["patch_refinement_rail"]["status"]
        == "explained_not_outlier"
    )
    assert review.result["patch_refinement_rail"]["advisory_only"] is True
    assert review.result["patch_refinement_policy"] == (
        "advisory_only_cannot_override_pixels_or_vlm"
    )
    assert review.result["patch_refinement_preview"]["status"] == (
        "validated"
    )
    assert review.result["target_class"] == "boat"
    assert review.result["specificity_alignment"] == "supports_suggested"
    assert review.result["target_background_contrast"] == "target_specific"
    assert review.result["specificity_probe"]["status"] == "completed"
    assert review.result["specificity_probe"]["specificity_alignment"] == "supports_suggested"
    assert review.result["specificity_probe"]["target_background_contrast"] == "target_specific"
    assert review.result["specificity_probe"]["specificity_margin"] == "suggested_target_favored"
    assert len(review.result["specificity_probe"]["subdescription_assessments"]) == 2
    assert review.result["specificity_probe"]["best_supported_class"] == "boat"
    assert review.result["supporting_clean_evidence_ids"] == [
        "target_detail_2",
        "specificity_region_contrast_7",
        "zoom_region_10",
    ]
    assert review.result["applied"] is False
    assert review.result["executed_tools"] == [
        "inspect_class_context_pack",
        "inspect_local_consensus_context",
        "inspect_overlap_decomposition",
        "inspect_patch_refinement",
        "inspect_same_image_embedding_report",
        "inspect_same_image_scale_report",
        "inspect_source_overlay",
        "inspect_specificity_region_contrast",
        "inspect_target_context",
        "inspect_target_detail",
        "zoom_source_region",
    ]
    assert "zoom_source_region(draw_bbox=false)" in review.result["satisfied_requirements"]
    assert result["points"][0]["class_name"] == "car"
    assert review.result["review_agent_controller"] == "state_machine_v2"
    assert review.result["evidence_ledger"]["clean_visual_evidence_ids"] == [
        "target_context_1",
        "target_detail_2",
        "source_clean_3",
        "class_context_pack_6",
        "specificity_region_contrast_7",
        "zoom_region_10",
    ]
    assert review.result["evidence_ledger"]["rows"]
    assert "source_overlay_4" in review.result["evidence_ledger"]["geometry_overlay_evidence_ids"]
    assert review.result["evidence_ledger"]["deterministic_context_evidence_ids"] == [
        "same_image_scale_report_8",
        "same_image_embedding_report_9",
    ]
    assert review.result["evidence_ledger"]["specificity_probe"]["status"] == "completed"
    assert review.result["deterministic_context"]["scale"]["signal"] == "insufficient"
    assert review.result["deterministic_context"]["embedding"]["signal"] == "insufficient"
    assert review.result["evidence_ledger"]["local_consensus_evidence_ids"] == [
        "local_consensus_context_11",
        "local_consensus_dot_map_13",
    ]
    assert review.result["expanded_by_controller"] is True
    assert review.result["model_compact_arguments"]["decision"] == "accept_suggested"
    review_dir = class_root / parent_id / "qwen_reviews" / review.review_id
    events = [
        json.loads(line)
        for line in (review_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event.get("type") == "final_validation_error" for event in events)
    assert (review_dir / "final.json").is_file()
    assert (review_dir / "prompt_sources.json").is_file()
    assert (review_dir / "evidence_ledger.json").is_file()
    assert (review_dir / "specificity_probe.json").is_file()
    assert (review_dir / "events.jsonl").is_file()
    event_lines = [
        json.loads(line)
        for line in (review_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model_inputs = [event for event in event_lines if event.get("type") == "model_input"]
    model_outputs = [event for event in event_lines if event.get("type") == "model_output"]
    assert len(model_inputs) == len(calls)
    assert len(model_outputs) == len(calls)
    assert all(isinstance(event.get("messages"), list) for event in model_inputs)
    assert model_inputs[0]["phase"] == "specificity_probe"
    assert [event["phase"] for event in model_inputs] == [
        "specificity_probe",
        "final_attempt_1",
        "final_attempt_2",
        "final_attempt_3",
    ]
    assert model_inputs[0]["tool_schema"][0]["name"] == "probe_specificity"
    assert model_inputs[1]["tool_schema"][0]["name"] == "finalize_review"
    assert model_inputs[-1]["tool_schema"][0]["name"] == "finalize_review"
    assert "evidence_ids" not in model_inputs[-1]["tool_schema"][0]["parameters"]["required"]
    assert all(event.get("tool_schema_chat_template_disabled") for event in model_inputs)
    assert model_inputs[-1]["assistant_prefix_strategy"] == "plain_json_arguments"
    assert "zoom_source_region(draw_bbox=false)" in model_inputs[-1]["satisfied_requirements"]
    controller_calls = [event for event in event_lines if event.get("type") == "controller_tool_call"]
    required_controller_calls = [event for event in controller_calls if event.get("required_phase")]
    assert [event.get("tool") for event in required_controller_calls] == [
        "inspect_target_context",
        "inspect_target_detail",
        "inspect_source_overlay",
        "inspect_overlap_decomposition",
        "inspect_class_context_pack",
        "inspect_specificity_region_contrast",
        "inspect_same_image_scale_report",
        "inspect_same_image_embedding_report",
        "zoom_source_region",
    ]
    assert any(event.get("tool") == "inspect_local_consensus_context" for event in controller_calls)
    router_events = [event for event in event_lines if event.get("type") == "router_decision"]
    assert router_events[-1]["router"]["action"] == "inspect_local_consensus_context"
    assert router_events[-1].get("skipped_model_call") is True
    assert router_events[-1]["router"]["controller_forced"] is True
    specificity_events = [event for event in event_lines if event.get("type") == "specificity_probe_result"]
    assert specificity_events[-1]["status"] == "completed"
    assert specificity_events[-1]["specificity_probe"]["specificity_alignment"] == "supports_suggested"
    ledger_events = [event for event in event_lines if event.get("type") == "evidence_ledger"]
    assert ledger_events[-1]["clean_visual_evidence_ids"] == [
        "target_context_1",
        "target_detail_2",
        "source_clean_3",
        "class_context_pack_6",
        "specificity_region_contrast_7",
        "zoom_region_10",
    ]
    expansion_events = [event for event in event_lines if event.get("type") == "compact_final_expanded"]
    assert expansion_events[-1]["expanded_arguments"]["evidence_ids"] == [
        "class_context_pack_6",
        "local_consensus_clean_context_12",
        "local_consensus_context_11",
        "local_consensus_dot_map_13",
        "overlap_decomposition_5",
        "same_image_embedding_report_9",
        "same_image_scale_report_8",
        "source_clean_3",
        "source_overlay_4",
        "specificity_region_contrast_7",
        "target_context_1",
        "target_detail_2",
        "zoom_region_10",
    ]
    assert expansion_events[-1]["expanded_arguments"]["supporting_clean_evidence_ids"] == [
        "target_detail_2",
        "specificity_region_contrast_7",
        "zoom_region_10",
    ]
    evidence_paths = sorted((review_dir / "evidence").glob("*.jpg"))
    assert len(evidence_paths) == 13
    assert any(path.name.startswith("target_detail_") for path in evidence_paths)
    assert any(path.name.startswith("source_clean_") for path in evidence_paths)
    assert any(path.name.startswith("specificity_region_contrast_") for path in evidence_paths)
    assert any(path.name.startswith("local_consensus_context_") for path in evidence_paths)
    assert any(path.name.startswith("local_consensus_clean_context_") for path in evidence_paths)
    assert any(path.name.startswith("local_consensus_dot_map_") for path in evidence_paths)
    assert any(path.name.startswith("zoom_region_") for path in evidence_paths)


def test_class_analysis_qwen_review_mlx_reset_cadence_is_generic_and_logged(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_reset_policy"
    (class_root / parent_id).mkdir(parents=True)
    with api.CLASS_ANALYSIS_QWEN_REVIEW_MLX_RESET_LOCK:
        api.CLASS_ANALYSIS_QWEN_REVIEW_MLX_RESET_STATE["completed_calls"] = 0
    monkeypatch.setattr(api, "qwen_runtime_platform", None)

    def fake_qwen_chat_stream(messages, **kwargs):
        api.qwen_runtime_platform = api.QWEN_PLATFORM_MLX
        yield '{"decision":"skip_uncertain"}'

    resets = []

    def fake_unload_qwen_runtime():
        resets.append("reset")

    monkeypatch.setattr(api, "_run_qwen_chat_stream", fake_qwen_chat_stream)
    monkeypatch.setattr(api, "_unload_qwen_runtime", fake_unload_qwen_runtime)
    review = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_reset_policy",
        parent_job_id=parent_id,
        point_id="p0",
        request={"mlx_reset_every": 2},
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": "test"}]}]
    api._class_analysis_qwen_review_model_call(
        review,
        messages,
        phase="first",
        model_id="test-model",
        tool_specs=[],
        max_new_tokens=16,
        progress=0.1,
        assistant_prefix=None,
    )
    assert resets == []

    api._class_analysis_qwen_review_model_call(
        review,
        messages,
        phase="second",
        model_id="test-model",
        tool_specs=[],
        max_new_tokens=16,
        progress=0.2,
        assistant_prefix=None,
    )

    assert resets == ["reset"]
    review_dir = class_root / parent_id / "qwen_reviews" / review.review_id
    events = [
        json.loads(line)
        for line in (review_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reset_events = [event for event in events if event.get("type") == "qwen_runtime_reset"]
    assert len(reset_events) == 1
    assert reset_events[0]["reason"] == "mlx_reset_every_2"
    assert reset_events[0]["completed_calls_before_reset"] == 2


def test_class_analysis_qwen_review_keep_loaded_disables_all_resets(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_keep_loaded"
    (class_root / parent_id).mkdir(parents=True)
    with api.CLASS_ANALYSIS_QWEN_REVIEW_MLX_RESET_LOCK:
        api.CLASS_ANALYSIS_QWEN_REVIEW_MLX_RESET_STATE["completed_calls"] = 0
    monkeypatch.setattr(api, "qwen_runtime_platform", None)

    retained_during_calls = []

    def fake_qwen_chat_stream(messages, **kwargs):
        retained_during_calls.append(
            api.qwen_caption_retain_runtime_var.get()
        )
        api.qwen_runtime_platform = api.QWEN_PLATFORM_MLX
        yield '{"decision":"skip_uncertain"}'

    resets = []
    monkeypatch.setattr(api, "_run_qwen_chat_stream", fake_qwen_chat_stream)
    monkeypatch.setattr(api, "_unload_qwen_runtime", lambda: resets.append("reset"))
    review = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_keep_loaded",
        parent_job_id=parent_id,
        point_id="p0",
        request={
            "keep_vlm_loaded": True,
            "mlx_reset_every": 1,
            "reset_qwen_runtime_after_review": True,
        },
    )

    messages = [
        {"role": "user", "content": [{"type": "text", "text": "test"}]}
    ]
    for phase in ("first", "second"):
        api._class_analysis_qwen_review_model_call(
            review,
            messages,
            phase=phase,
            model_id="test-model",
            tool_specs=[],
            max_new_tokens=16,
            progress=0.1,
            assistant_prefix=None,
        )

    status = api._class_analysis_qwen_review_runtime_retention(review)
    assert resets == []
    assert retained_during_calls == [True, True]
    assert status["keep_vlm_loaded"] is True
    assert status["periodic_reset_every"] == 0
    assert status["post_review_reset"] is False


def test_qwen_mlx_caption_runtime_can_be_retained_with_global_cache_disabled(
    monkeypatch,
):
    runtime = api.QwenRuntime(
        model=object(),
        processor=object(),
        platform=api.QWEN_PLATFORM_MLX,
        model_id="resolved-test-model",
    )
    loads = []
    monkeypatch.setattr(api, "QWEN_CAPTION_CACHE_LIMIT", 0)
    monkeypatch.setattr(api, "qwen_caption_cache", {})
    monkeypatch.setattr(api, "qwen_caption_order", api.deque())
    monkeypatch.setattr(api, "qwen_caption_pinned_cache_key", None)
    monkeypatch.setattr(
        api,
        "_effective_qwen_model_id_for_platform",
        lambda model_id, platform: "resolved-test-model",
    )
    monkeypatch.setattr(
        api,
        "_load_qwen_mlx_runtime",
        lambda model_id: loads.append(model_id) or runtime,
    )
    monkeypatch.setattr(api, "_clear_mlx_cache", lambda: None)

    first = api._ensure_qwen_mlx_ready_for_caption(
        "test-model",
        retain_runtime=True,
    )
    second = api._ensure_qwen_mlx_ready_for_caption(
        "test-model",
        retain_runtime=True,
    )

    assert first is runtime
    assert second is runtime
    assert loads == ["resolved-test-model"]
    assert list(api.qwen_caption_cache) == [
        "caption:mlx_vlm:resolved-test-model"
    ]
    assert api.qwen_caption_pinned_cache_key == (
        "caption:mlx_vlm:resolved-test-model"
    )


def test_qwen_mlx_retained_runtime_survives_unrelated_uncached_load(
    monkeypatch,
):
    runtimes = {
        name: api.QwenRuntime(
            model=object(),
            processor=object(),
            platform=api.QWEN_PLATFORM_MLX,
            model_id=name,
        )
        for name in ("retained-model", "one-shot-model")
    }
    monkeypatch.setattr(api, "QWEN_CAPTION_CACHE_LIMIT", 0)
    monkeypatch.setattr(api, "qwen_caption_cache", {})
    monkeypatch.setattr(api, "qwen_caption_order", api.deque())
    monkeypatch.setattr(api, "qwen_caption_pinned_cache_key", None)
    monkeypatch.setattr(
        api,
        "_effective_qwen_model_id_for_platform",
        lambda model_id, platform: model_id,
    )
    monkeypatch.setattr(
        api,
        "_load_qwen_mlx_runtime",
        lambda model_id: runtimes[model_id],
    )
    monkeypatch.setattr(api, "_clear_mlx_cache", lambda: None)

    api._ensure_qwen_mlx_ready_for_caption(
        "retained-model",
        retain_runtime=True,
    )
    one_shot = api._ensure_qwen_mlx_ready_for_caption(
        "one-shot-model",
        retain_runtime=False,
    )

    assert one_shot is runtimes["one-shot-model"]
    assert list(api.qwen_caption_cache) == [
        "caption:mlx_vlm:retained-model"
    ]
    assert api._qwen_loaded_for_model(
        "retained-model",
        api.QWEN_PLATFORM_MLX,
    ) is True


def test_qwen_loaded_for_transformers_caption_cache(monkeypatch):
    monkeypatch.setattr(
        api,
        "_effective_qwen_model_id_for_platform",
        lambda model_id, platform: model_id,
    )
    monkeypatch.setattr(
        api,
        "qwen_caption_cache",
        {"caption:org/model:cuda:0": (object(), object())},
    )

    assert api._qwen_loaded_for_model(
        "org/model",
        api.QWEN_PLATFORM_TRANSFORMERS,
    ) is True


def test_qwen_transformers_retained_runtime_survives_unrelated_uncached_load(
    monkeypatch,
):
    loads = []
    monkeypatch.setattr(api, "QWEN_CAPTION_CACHE_LIMIT", 0)
    monkeypatch.setattr(api, "QWEN_DEVICE_PREF", "cpu")
    monkeypatch.setattr(api, "qwen_caption_cache", {})
    monkeypatch.setattr(api, "qwen_caption_order", api.deque())
    monkeypatch.setattr(api, "qwen_caption_pinned_cache_key", None)
    monkeypatch.setattr(
        api,
        "_resolve_qwen_runtime_platform",
        lambda *_args, **_kwargs: api.QWEN_PLATFORM_TRANSFORMERS,
    )
    monkeypatch.setattr(
        api,
        "_effective_qwen_model_id_for_platform",
        lambda model_id, _platform: model_id,
    )
    monkeypatch.setattr(
        api,
        "_qwen_model_local_state",
        lambda *_args, **_kwargs: {"needs_download": False},
    )
    monkeypatch.setattr(api, "_qwen_progress_update", lambda **_kwargs: None)

    def fake_ensure(model_id, *, state, device_pref, caption_cache_limit, **_kwargs):
        cache_key = f"caption:{model_id}:{device_pref}"
        if caption_cache_limit == 0:
            state["qwen_caption_cache"].clear()
            state["qwen_caption_order"].clear()
        cached = state["qwen_caption_cache"].get(cache_key)
        if cached and caption_cache_limit:
            return cached
        loads.append(model_id)
        runtime = (object(), object())
        if caption_cache_limit:
            state["qwen_caption_cache"][cache_key] = runtime
            state["qwen_caption_order"].append(cache_key)
        return runtime

    monkeypatch.setattr(
        api,
        "_ensure_qwen_ready_for_caption_impl",
        fake_ensure,
    )

    retained = api._ensure_qwen_ready_for_caption(
        "org/retained",
        retain_runtime=True,
    )
    one_shot = api._ensure_qwen_ready_for_caption(
        "org/one-shot",
        retain_runtime=False,
    )

    assert retained.model is not None
    assert one_shot.model is not None
    assert loads == ["org/retained", "org/one-shot"]
    assert list(api.qwen_caption_cache) == ["caption:org/retained:cpu"]
    assert api.qwen_caption_pinned_cache_key == "caption:org/retained:cpu"
    assert api._qwen_loaded_for_model(
        "org/retained",
        api.QWEN_PLATFORM_TRANSFORMERS,
    ) is True


def test_qwen_mlx_switching_retained_model_clears_released_metal_cache(
    monkeypatch,
):
    runtimes = {
        model_id: api.QwenRuntime(
            model=object(),
            processor=object(),
            platform=api.QWEN_PLATFORM_MLX,
            model_id=model_id,
        )
        for model_id in ("first-model", "second-model")
    }
    clears = []
    monkeypatch.setattr(api, "QWEN_CAPTION_CACHE_LIMIT", 0)
    monkeypatch.setattr(api, "qwen_caption_cache", {})
    monkeypatch.setattr(api, "qwen_caption_order", api.deque())
    monkeypatch.setattr(api, "qwen_caption_pinned_cache_key", None)
    monkeypatch.setattr(
        api,
        "_effective_qwen_model_id_for_platform",
        lambda model_id, _platform: model_id,
    )
    monkeypatch.setattr(
        api,
        "_load_qwen_mlx_runtime",
        lambda model_id: runtimes[model_id],
    )
    monkeypatch.setattr(api, "_clear_mlx_cache", lambda: clears.append("clear"))

    api._ensure_qwen_mlx_ready_for_caption(
        "first-model",
        retain_runtime=True,
    )
    api._ensure_qwen_mlx_ready_for_caption(
        "second-model",
        retain_runtime=True,
    )

    assert clears == ["clear"]
    assert list(api.qwen_caption_cache) == [
        "caption:mlx_vlm:second-model"
    ]
    assert api.qwen_caption_pinned_cache_key == (
        "caption:mlx_vlm:second-model"
    )


def test_completed_qwen_review_exposes_full_durable_trace(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_complete_trace"
    (class_root / parent_id).mkdir(parents=True)
    review = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_complete_trace",
        parent_job_id=parent_id,
        point_id="p0",
    )
    api._class_analysis_qwen_review_append_event(
        review,
        {
            "type": "model_input",
            "phase": "final_attempt_1",
            "messages": [{"role": "user", "content": "pixels"}],
            "image_policy": {"images": [{"evidence_id": "target_detail_1"}]},
        },
    )
    api._class_analysis_qwen_review_append_event(
        review,
        {
            "type": "model_output",
            "phase": "final_attempt_1",
            "text": '{"decision":"accept_suggested"}',
        },
    )
    review.model_call_count = 1
    review.model_output_count = 1
    review.status = "completed"

    summary = api._serialize_class_analysis_qwen_review_job(review)
    complete = api._serialize_class_analysis_qwen_review_job(
        review,
        include_complete_trace=True,
    )
    assert summary["complete_trace"] == {
        "available": True,
        "included": False,
        "event_count": None,
        "model_input_count": 1,
        "model_output_count": 1,
    }
    assert complete["complete_trace"]["available"] is True
    assert complete["complete_trace"]["included"] is True
    assert complete["complete_trace"]["event_count"] == 2
    assert complete["complete_trace"]["model_input_count"] == 1
    assert complete["complete_trace"]["model_output_count"] == 1
    assert [
        item["type"] for item in complete["complete_trace"]["events"]
    ] == ["model_input", "model_output"]


def test_class_analysis_qwen_review_model_call_keeps_schema_calls_thinking_disabled(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    monkeypatch.setattr(api, "qwen_runtime_platform", None)
    calls = []

    def fake_qwen_chat_stream(_messages, **kwargs):
        calls.append(dict(kwargs))
        yield "ok"

    monkeypatch.setattr(api, "_run_qwen_chat_stream", fake_qwen_chat_stream)
    review = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_thinking_policy",
        parent_job_id="ca_thinking_policy",
        point_id="p0",
        request={"enable_thinking": True, "thinking_effort": "high", "thinking_scale_factor": 0.75},
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "test"}]}]

    api._class_analysis_qwen_review_model_call(
        review,
        messages,
        phase="schema_finalization",
        model_id="test-model",
        tool_specs=[],
        max_new_tokens=16,
        progress=0.1,
        assistant_prefix=None,
    )
    api._class_analysis_qwen_review_model_call(
        review,
        messages,
        phase="thinking_scratchpad",
        model_id="test-model",
        tool_specs=[],
        max_new_tokens=16,
        progress=0.2,
        assistant_prefix=None,
        enable_thinking=True,
    )

    assert calls[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "thinking_effort" not in calls[0]
    assert "thinking_scale_factor" not in calls[0]
    assert calls[1]["chat_template_kwargs"] == {"enable_thinking": True}
    assert calls[1]["thinking_effort"] == "high"
    assert calls[1]["thinking_scale_factor"] == 0.75


def test_class_analysis_qwen_review_model_call_publishes_live_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", tmp_path / "class_analysis")
    observed = []
    review = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_live",
        parent_job_id="ca_live",
        point_id="p0",
        request={},
    )

    def fake_stream(_messages, **_kwargs):
        yield "first partial output"
        observed.append(copy.deepcopy(review.active_generation))
        yield "first partial output and the completed answer with enough additional streamed detail"
        observed.append(copy.deepcopy(review.active_generation))

    monkeypatch.setattr(api, "_run_qwen_chat_stream", fake_stream)
    messages = [{"role": "user", "content": [{"type": "text", "text": "test"}]}]

    result = api._class_analysis_qwen_review_model_call(
        review,
        messages,
        phase="live_probe",
        model_id="test-model",
        tool_specs=[],
        max_new_tokens=64,
        progress=0.5,
        assistant_prefix=None,
    )

    assert result == "first partial output and the completed answer with enough additional streamed detail"
    assert observed[0]["phase"] == "live_probe"
    assert observed[0]["text"] == "first partial output"
    assert observed[-1]["text"] == result
    assert review.active_generation is None
    assert review.trace_events[-1]["type"] == "model_output"
    assert review.trace_events[-1]["text"] == result


def test_create_class_analysis_qwen_review_defaults_poor_target_to_vlm(
    monkeypatch,
):
    parent = api.ClassAnalysisJob(
        job_id="ca_default_poor_vlm",
        status="completed",
        result={"points": [{"point_id": "p0"}]},
    )
    monkeypatch.setattr(
        api,
        "get_class_analysis_result",
        lambda _job_id: {"points": [{"point_id": "p0"}]},
    )
    monkeypatch.setattr(api, "_get_class_analysis_job", lambda _job_id: parent)
    captured = {}

    def capture_job(**kwargs):
        captured["job"] = kwargs["job"]

    monkeypatch.setattr(api, "_register_job_and_start_thread", capture_job)

    payload = api.create_class_analysis_qwen_review(parent.job_id, "p0", {})

    assert captured["job"].request["allow_poor_final_review"] is True
    assert payload["request"]["allow_poor_final_review"] is True


def test_class_analysis_qwen_review_controller_skips_poor_target_without_qwen(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_limited"
    workspace_dir = class_root / parent_id / "active_workspace"
    images_dir = workspace_dir / "images"
    images_dir.mkdir(parents=True)
    for filename, color in (("target.jpg", (50, 60, 70)), ("same.jpg", (70, 80, 90)), ("other.jpg", (30, 80, 120))):
        image = Image.new("RGB", (220, 180), color)
        draw = ImageDraw.Draw(image)
        draw.rectangle([40, 35, 170, 140], outline=(180, 200, 220), width=4)
        image.save(images_dir / filename)
    api._class_analysis_write_json(
        workspace_dir / "manifest.json",
        workspace_dir,
        {
            "labelmap": ["ClassA", "ClassB"],
            "images": [
                {"split": "train", "image_relpath": "target.jpg", "label_lines": ["0 0.25 0.25 0.04 0.04"]},
                {"split": "train", "image_relpath": "same.jpg", "label_lines": ["0 0.5 0.5 0.4 0.4"]},
                {"split": "train", "image_relpath": "other.jpg", "label_lines": ["1 0.5 0.5 0.4 0.4"]},
            ],
            "yolo_layout": "flat",
            "source_mode": "active_workspace",
        },
    )
    result = {
        "summary": {
            "source_mode": "active_workspace",
            "source_id": parent_id,
            "dataset_label": "test workspace",
            "labelmap": ["ClassA", "ClassB"],
            "analysis_scope": "all_classes",
        },
        "points": [
            {
                "point_id": "p0",
                "class_name": "ClassA",
                "suggested_neighbor_class": "ClassB",
                "wrong_class_suspicion": 0.91,
                "same_class_neighbor_ratio": 0.0,
                "top_other_neighbor_ratio": 1.0,
                "neighbor_class_counts": {"ClassB": 3},
                "neighbor_ids": ["p2"],
                "neighbor_distances": [0.12],
                "image_relpath": "target.jpg",
                "split": "train",
                "bbox_xyxy": [48, 42, 58, 52],
                "is_wrong_class_candidate": True,
            },
            {
                "point_id": "p1",
                "class_name": "ClassA",
                "image_relpath": "same.jpg",
                "split": "train",
                "bbox_xyxy": [45, 40, 150, 130],
                "same_class_neighbor_ratio": 0.95,
                "top_other_neighbor_ratio": 0.02,
                "outlier_score": 0.02,
            },
            {
                "point_id": "p2",
                "class_name": "ClassB",
                "image_relpath": "other.jpg",
                "split": "train",
                "bbox_xyxy": [45, 40, 150, 130],
                "same_class_neighbor_ratio": 0.96,
                "top_other_neighbor_ratio": 0.01,
                "outlier_score": 0.02,
            },
        ],
        "wrong_class_candidates": [{"point_id": "p0", "class_name": "ClassA", "suggested_neighbor_class": "ClassB"}],
    }
    parent = api.ClassAnalysisJob(job_id=parent_id, status="completed", result=result)
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS[parent_id] = parent

    def fail_qwen_chat(*args, **kwargs):
        raise AssertionError("Qwen should not be called for unclear target quality")

    monkeypatch.setattr(api, "_run_qwen_chat", fail_qwen_chat)
    review = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_limited",
        parent_job_id=parent_id,
        point_id="p0",
        request={
            "max_turns": 8,
            "enable_local_consensus_context": True,
            "enable_class_concept_briefs": True,
            "allow_limited_final_review": True,
            "allow_poor_final_review": False,
        },
    )
    with api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS_LOCK:
        api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS[review.review_id] = review

    api._run_class_analysis_qwen_review_job(review)

    assert review.status == "completed"
    assert review.result["decision"] == "skip_uncertain"
    assert review.result["model_invoked"] is False
    assert review.result["model_final_completed"] is False
    assert review.result["reviewed_by_model"] is None
    assert review.result["decision_source"] == "controller_guardrail"
    assert review.result["backend_visual_quality"]["tier"] != "clear"
    assert review.result["class_concept_briefs"]["enabled"] is False
    assert any("Controller skipped Qwen final decision" in reason for reason in review.result["guardrail_reasons"])
    review_dir = class_root / parent_id / "qwen_reviews" / review.review_id
    events = [
        json.loads(line)
        for line in (review_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event.get("type") == "concept_briefs_skipped" for event in events)
    assert any(event.get("type") == "controller_final_skip" for event in events)
    model_preflight = [event for event in events if event.get("type") == "model_preflight"]
    assert model_preflight[-1]["requested_model_id"] == api.CLASS_ANALYSIS_QWEN_REVIEW_DEFAULT_MODEL_ID
    assert model_preflight[-1]["status"] == "skipped"


def test_class_analysis_qwen_review_poor_target_can_reach_guarded_advisory_review(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_poor_advisory"
    workspace_dir = class_root / parent_id / "active_workspace"
    images_dir = workspace_dir / "images"
    images_dir.mkdir(parents=True)
    image = Image.new("RGB", (220, 180), (40, 50, 60))
    draw = ImageDraw.Draw(image)
    draw.rectangle([48, 42, 58, 52], fill=(220, 220, 230))
    image.save(images_dir / "target.jpg")
    for filename, color in (("same.jpg", (70, 80, 90)), ("other.jpg", (30, 80, 120))):
        anchor = Image.new("RGB", (220, 180), color)
        ImageDraw.Draw(anchor).rectangle([45, 40, 150, 130], fill=(180, 200, 220))
        anchor.save(images_dir / filename)
    api._class_analysis_write_json(
        workspace_dir / "manifest.json",
        workspace_dir,
        {
            "labelmap": ["ClassA", "ClassB"],
            "images": [
                {"split": "train", "image_relpath": "target.jpg", "label_lines": ["0 0.25 0.25 0.04 0.04"]},
                {"split": "train", "image_relpath": "same.jpg", "label_lines": ["0 0.5 0.5 0.4 0.4"]},
                {"split": "train", "image_relpath": "other.jpg", "label_lines": ["1 0.5 0.5 0.4 0.4"]},
            ],
            "yolo_layout": "flat",
            "source_mode": "active_workspace",
        },
    )
    point = {
        "point_id": "p0",
        "class_name": "ClassA",
        "suggested_neighbor_class": "ClassB",
        "wrong_class_suspicion": 0.91,
        "same_class_neighbor_ratio": 0.0,
        "top_other_neighbor_ratio": 1.0,
        "neighbor_class_counts": {"ClassB": 3},
        "neighbor_ids": ["p2"],
        "neighbor_distances": [0.12],
        "image_relpath": "target.jpg",
        "split": "train",
        "bbox_xyxy": [48, 42, 58, 52],
        "is_wrong_class_candidate": True,
    }
    result = {
        "summary": {
            "source_mode": "active_workspace",
            "source_id": parent_id,
            "dataset_label": "test workspace",
            "labelmap": ["ClassA", "ClassB"],
            "analysis_scope": "all_classes",
        },
        "points": [
            point,
            {
                "point_id": "p1",
                "class_name": "ClassA",
                "image_relpath": "same.jpg",
                "split": "train",
                "bbox_xyxy": [45, 40, 150, 130],
                "same_class_neighbor_ratio": 0.95,
                "top_other_neighbor_ratio": 0.02,
                "outlier_score": 0.02,
            },
            {
                "point_id": "p2",
                "class_name": "ClassB",
                "image_relpath": "other.jpg",
                "split": "train",
                "bbox_xyxy": [45, 40, 150, 130],
                "same_class_neighbor_ratio": 0.96,
                "top_other_neighbor_ratio": 0.01,
                "outlier_score": 0.02,
            },
        ],
        "wrong_class_candidates": [{"point_id": "p0", "class_name": "ClassA", "suggested_neighbor_class": "ClassB"}],
    }
    parent = api.ClassAnalysisJob(job_id=parent_id, status="completed", result=result)
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS[parent_id] = parent
    calls = []

    def fake_qwen_chat_stream(messages, **kwargs):
        calls.append({"messages": copy.deepcopy(messages), "kwargs": dict(kwargs)})
        yield '{"decision":"accept_suggested","target_class":"ClassB","confidence":0.8,"visual_quality":"poor","object_visibility":"tiny_or_blurry","current_evidence":"weak","suggested_evidence":"strong","target_evidence":"strong","anchor_evidence_current":"weak","anchor_evidence_suggested":"strong","local_context_evidence":"strong","global_context_evidence":"strong","same_image_scale_evidence":"insufficient","same_image_embedding_evidence":"insufficient","overlap_assessment":"none","overlap_explains_candidate_similarity":false,"local_consensus_evidence":"not_applicable","visible_target_cues":["bright rectangular target","hard-edged target patch"],"supporting_clean_evidence_ids":["target_detail_2"],"rationale_short":"target appears closer to ClassB but is tiny","counter_evidence":"poor crop quality","human_review_needed":true}'

    monkeypatch.setattr(api, "_run_qwen_chat_stream", fake_qwen_chat_stream)
    monkeypatch.setattr(api, "_ensure_qwen_ready_for_caption", lambda *_args, **_kwargs: object())
    review = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_poor_advisory",
        parent_job_id=parent_id,
        point_id="p0",
        request={
            "max_turns": 2,
            "model_id": "test-model",
            "enable_local_consensus_context": True,
            "enable_class_concept_briefs": True,
            "allow_limited_final_review": True,
        },
    )
    with api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS_LOCK:
        api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS[review.review_id] = review

    api._run_class_analysis_qwen_review_job(review)

    assert calls
    assert review.status == "completed"
    assert review.result["backend_visual_quality"]["tier"] == "poor"
    assert review.result["decision"] == "skip_uncertain"
    assert review.result["guarded_recommendation"]["blocked"] is True
    assert review.result["guarded_recommendation"]["decision"] == "accept_suggested"
    assert review.result["review_disposition"]["signal"] == "guarded_human_triage"
    assert review.result["model_invoked"] is True
    assert review.result["model_final_completed"] is True
    assert review.result["model_final_validated"] is True
    assert review.result["reviewed_by_model"] == "test-model"
    assert review.result["decision_source"] == "vlm_validated"
    review_dir = class_root / parent_id / "qwen_reviews" / review.review_id
    events = [
        json.loads(line)
        for line in (review_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event.get("type") == "model_input" for event in events)


def test_class_analysis_qwen_review_limited_target_can_reach_advisory_final_review(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_limited_advisory"
    workspace_dir = class_root / parent_id / "active_workspace"
    images_dir = workspace_dir / "images"
    images_dir.mkdir(parents=True)
    for filename, color in (("target.jpg", (50, 60, 70)), ("same.jpg", (70, 80, 90)), ("other.jpg", (30, 80, 120))):
        image = Image.new("RGB", (220, 180), color)
        draw = ImageDraw.Draw(image)
        draw.rectangle([35, 30, 165, 135], outline=(180, 200, 220), width=4)
        if filename == "target.jpg":
            draw.rectangle([38, 34, 62, 60], fill=(235, 235, 235))
            draw.line([(38, 34), (62, 60)], fill=(20, 20, 20), width=3)
            draw.line([(38, 60), (62, 34)], fill=(20, 20, 20), width=3)
        image.save(images_dir / filename)
    api._class_analysis_write_json(
        workspace_dir / "manifest.json",
        workspace_dir,
        {
            "labelmap": ["ClassA", "ClassB"],
            "images": [
                {"split": "train", "image_relpath": "target.jpg", "label_lines": ["0 0.23 0.28 0.09 0.16"]},
                {"split": "train", "image_relpath": "same.jpg", "label_lines": ["0 0.5 0.5 0.4 0.4"]},
                {"split": "train", "image_relpath": "other.jpg", "label_lines": ["1 0.5 0.5 0.4 0.4"]},
            ],
            "yolo_layout": "flat",
            "source_mode": "active_workspace",
        },
    )
    result = {
        "summary": {
            "source_mode": "active_workspace",
            "source_id": parent_id,
            "dataset_label": "test workspace",
            "labelmap": ["ClassA", "ClassB"],
            "analysis_scope": "all_classes",
        },
        "points": [
            {
                "point_id": "p0",
                "class_name": "ClassA",
                "suggested_neighbor_class": "ClassB",
                "wrong_class_suspicion": 0.78,
                "same_class_neighbor_ratio": 0.35,
                "top_other_neighbor_ratio": 0.65,
                "neighbor_class_counts": {"ClassB": 2, "ClassA": 1},
                "neighbor_ids": ["p2", "p1"],
                "neighbor_distances": [0.12, 0.22],
                "image_relpath": "target.jpg",
                "split": "train",
                "bbox_xyxy": [38, 34, 62, 60],
                "is_wrong_class_candidate": True,
            },
            {
                "point_id": "p1",
                "class_name": "ClassA",
                "image_relpath": "same.jpg",
                "split": "train",
                "bbox_xyxy": [45, 40, 150, 130],
                "same_class_neighbor_ratio": 0.95,
                "top_other_neighbor_ratio": 0.02,
                "outlier_score": 0.02,
            },
            {
                "point_id": "p2",
                "class_name": "ClassB",
                "image_relpath": "other.jpg",
                "split": "train",
                "bbox_xyxy": [45, 40, 150, 130],
                "same_class_neighbor_ratio": 0.96,
                "top_other_neighbor_ratio": 0.01,
                "outlier_score": 0.02,
            },
        ],
        "wrong_class_candidates": [{"point_id": "p0", "class_name": "ClassA", "suggested_neighbor_class": "ClassB"}],
    }
    parent = api.ClassAnalysisJob(job_id=parent_id, status="completed", result=result)
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS[parent_id] = parent
    calls = []

    def fake_qwen_chat_stream(messages, **kwargs):
        calls.append({"messages": copy.deepcopy(messages), "kwargs": dict(kwargs)})
        yield json.dumps(
            {
                "decision": "confirm_current",
                "target_class": "ClassA",
                "confidence": 0.74,
                "visual_quality": "limited",
                "object_visibility": "partial",
                "current_evidence": "strong",
                "suggested_evidence": "weak",
                "target_evidence": "strong",
                "overlap_assessment": "none",
                "overlap_explains_candidate_similarity": False,
                "anchor_evidence_current": "moderate",
                "anchor_evidence_suggested": "weak",
                "local_context_evidence": "moderate",
                "local_consensus_evidence": "not_applicable",
                "global_context_evidence": "moderate",
                "specificity_alignment": "supports_current",
                "target_background_contrast": "target_specific",
                "target_identity_summary": "compact target outline",
                "target_identity_uncertainty": "moderate",
                "whole_target_extent_supported": True,
                "dual_bbox_resolution": "not_applicable",
                "visible_target_cues": ["compact target outline"],
                "supporting_clean_evidence_ids": ["target_context_1"],
                "target_identity_evidence_ids": ["target_context_1"],
                "glossary_or_guidance_used": True,
                "rationale_short": "limited crop still supports current class",
                "counter_evidence": "suggested class cues are not visible",
                "human_review_needed": True,
            }
        )

    monkeypatch.setattr(api, "_run_qwen_chat_stream", fake_qwen_chat_stream)
    monkeypatch.setattr(api, "_ensure_qwen_ready_for_caption", lambda *_args, **_kwargs: object())
    review = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_limited_advisory",
        parent_job_id=parent_id,
        point_id="p0",
        request={
            "max_turns": 8,
            "model_id": "test-model",
            "enable_local_consensus_context": True,
            "enable_class_concept_briefs": True,
            "allow_limited_final_review": True,
        },
    )
    with api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS_LOCK:
        api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS[review.review_id] = review

    api._run_class_analysis_qwen_review_job(review)

    assert calls
    assert review.status == "completed"
    assert review.result["backend_visual_quality"]["tier"] == "limited"
    assert review.result["decision"] == "confirm_current"
    assert review.result["target_class"] == "ClassA"
    assert review.result["confidence"] <= 0.65
    assert review.result["human_review_needed"] is True
    assert review.result["guarded_recommendation"] is None
    assert any("backend visual-quality tier is limited" in reason for reason in review.result["advisory_reasons"])
    assert review.result["guardrail_reasons"] == []
    assert review.result["review_disposition"]["disposition"] == "actionable_confirm_current"
    assert review.result["class_concept_briefs"]["enabled"] is True
    review_dir = class_root / parent_id / "qwen_reviews" / review.review_id
    events = [
        json.loads(line)
        for line in (review_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event.get("type") == "concept_briefs_ready" for event in events)
    assert not any(event.get("type") == "controller_final_skip" for event in events)
    assert any(event.get("type") == "model_input" for event in events)


def test_class_analysis_qwen_review_limited_final_instruction_requests_advisory_opinion():
    instruction = api._class_analysis_qwen_review_final_instruction(
        required_tools={"inspect_target_context"},
        evidence_ids={"target_context_1"},
        point={"class_name": "ClassA", "suggested_neighbor_class": "ClassB"},
        visual_quality={"tier": "limited", "reasons": ["small_target"]},
    )
    text = instruction["content"][0]["text"]
    assert "advisory-only" in text
    assert "human-triage opinion" in text
    assert "may preserve class-changing opinions as guarded human-triage" in text
    assert "block automatic mutation" in text
    assert "Choose accept_suggested, change_to_other, or confirm_current" in text
    assert "class-changing decisions are forbidden" not in text
    assert "will not allow an automatic label recommendation" not in text


def test_class_analysis_qwen_review_builds_cached_class_concept_briefs(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_concepts"
    workspace_dir = class_root / parent_id / "active_workspace"
    images_dir = workspace_dir / "images"
    images_dir.mkdir(parents=True)
    for filename, color in (("car_a.jpg", (40, 60, 80)), ("car_b.jpg", (50, 70, 90)), ("boat_a.jpg", (40, 80, 180))):
        image = Image.new("RGB", (240, 200), color)
        draw = ImageDraw.Draw(image)
        draw.rectangle([60, 55, 170, 145], outline=(220, 240, 255), width=8)
        image.save(images_dir / filename)
    api._class_analysis_write_json(
        workspace_dir / "manifest.json",
        workspace_dir,
        {
            "labelmap": ["car", "boat"],
            "images": [
                {"split": "train", "image_relpath": "car_a.jpg", "label_lines": ["0 0.5 0.5 0.4 0.4"]},
                {"split": "train", "image_relpath": "car_b.jpg", "label_lines": ["0 0.5 0.5 0.4 0.4"]},
                {"split": "train", "image_relpath": "boat_a.jpg", "label_lines": ["1 0.5 0.5 0.4 0.4"]},
            ],
            "yolo_layout": "flat",
            "source_mode": "active_workspace",
        },
    )
    result = {
        "summary": {
            "source_mode": "active_workspace",
            "source_id": parent_id,
            "dataset_label": "test workspace",
            "labelmap": ["car", "boat"],
            "analysis_scope": "all_classes",
        },
        "points": [
            {
                "point_id": "p0",
                "class_name": "car",
                "suggested_neighbor_class": "boat",
                "neighbor_class_counts": {"boat": 4},
                "image_relpath": "car_a.jpg",
                "split": "train",
                "bbox_xyxy": [55, 50, 175, 150],
                "same_class_neighbor_ratio": 0.95,
                "top_other_neighbor_ratio": 0.02,
                "outlier_score": 0.05,
            },
            {
                "point_id": "p1",
                "class_name": "car",
                "image_relpath": "car_b.jpg",
                "split": "train",
                "bbox_xyxy": [55, 50, 175, 150],
                "same_class_neighbor_ratio": 0.93,
                "top_other_neighbor_ratio": 0.01,
                "outlier_score": 0.05,
            },
            {
                "point_id": "p2",
                "class_name": "boat",
                "image_relpath": "boat_a.jpg",
                "split": "train",
                "bbox_xyxy": [55, 50, 175, 150],
                "same_class_neighbor_ratio": 0.92,
                "top_other_neighbor_ratio": 0.03,
                "outlier_score": 0.04,
            },
        ],
    }
    parent = api.ClassAnalysisJob(job_id=parent_id, status="completed", result=result)
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS[parent_id] = parent
    outputs = iter(
        [
            '{"class_name":"car","summary":"cars are compact road vehicles","visual_traits":["road vehicle body"],"valid_variations":["different colors"],"exclude_when":["hull or watercraft"],"common_confusions":["boat"],"uncertainty_triggers":["tiny crop"]}',
            '{"class_name":"boat","summary":"boats are watercraft","visual_traits":["hull shape"],"valid_variations":["deck layouts"],"exclude_when":["road vehicle body"],"common_confusions":["car"],"uncertainty_triggers":["partial crop"]}',
            '{"class_a":"car","class_b":"boat","summary":"distinguish road-vehicle bodies from watercraft hulls","choose_class_a_when":["wheeled compact road body"],"choose_class_b_when":["visible hull or deck"],"shared_or_ambiguous_cues":["rectangular bright crop"],"hard_negative_cues":["ignore adjacent context"],"must_skip_when":["target distinction is hidden"]}',
        ]
    )
    calls = []

    def fake_qwen_chat_stream(messages, **kwargs):
        calls.append({"messages": copy.deepcopy(messages), "kwargs": dict(kwargs)})
        yield next(outputs)

    monkeypatch.setattr(api, "_run_qwen_chat_stream", fake_qwen_chat_stream)
    review = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_concepts",
        parent_job_id=parent_id,
        point_id="p0",
        request={"model_id": "test-model", "enable_class_concept_briefs": True},
    )
    with api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS_LOCK:
        api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS[review.review_id] = review

    packet = api._class_analysis_qwen_review_build_concept_briefs(
        review,
        result,
        result["points"][0],
        labelmap_glossary='{"car":"road vehicles","boat":"watercraft"}',
        review_guidance="Prefer visible pixels.",
        model_id="test-model",
    )

    assert packet["enabled"] is True
    assert packet["classes"] == ["car", "boat"]
    assert "road vehicles" in packet["prompt_text"]
    assert "hull shape" in packet["prompt_text"]
    assert "Pair car vs boat" in packet["prompt_text"]
    assert "wheeled compact road body" in packet["prompt_text"]
    assert len(calls) == 3
    assert calls[0]["kwargs"]["assistant_prefix"] is None
    assert calls[0]["kwargs"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert any(content.get("type") == "image" for content in calls[0]["messages"][1]["content"])
    review_dir = class_root / parent_id / "qwen_reviews" / review.review_id
    cache_dir = class_root / parent_id / "qwen_reviews" / "class_concept_briefs"
    pair_cache_dir = class_root / parent_id / "qwen_reviews" / "class_pair_contrast_briefs"
    assert (review_dir / "concept_briefs.json").is_file()
    assert len(list(cache_dir.glob("*.json"))) == 2
    assert len(list(cache_dir.glob("*_examples.jpg"))) == 2
    assert len(list(pair_cache_dir.glob("*.json"))) == 1
    assert len(list(pair_cache_dir.glob("*_examples.jpg"))) == 1

    calls.clear()
    cached = api._class_analysis_qwen_review_build_concept_briefs(
        review,
        result,
        result["points"][0],
        labelmap_glossary='{"car":"road vehicles","boat":"watercraft"}',
        review_guidance="Prefer visible pixels.",
        model_id="test-model",
    )
    assert [item["cache_hit"] for item in cached["artifacts"]] == [True, True]
    assert [item["cache_hit"] for item in cached["pair_contrasts"]] == [True]
    assert calls == []
    instruction = api._class_analysis_qwen_review_final_instruction(
        required_tools={"inspect_class_context_pack"},
        evidence_ids={"class_context_pack_1"},
        point=result["points"][0],
        visual_quality={"tier": "clear", "reasons": []},
        class_concept_brief_text=cached["prompt_text"],
    )
    instruction_text = instruction["content"][0]["text"]
    assert "Advisory class concept and pairwise contrast briefs built from trusted exemplars" in instruction_text
    assert "Fresh target pixels" in instruction_text
    assert "cars are compact road vehicles" in instruction_text
    assert "distinguish road-vehicle bodies from watercraft hulls" in instruction_text
    assert "Dataset-specific pair contrast beats generic word meanings" in instruction_text


def test_class_analysis_qwen_review_concept_examples_are_trusted_but_diverse():
    def point(point_id, projection, image_relpath, same=0.96, other=0.01):
        return {
            "point_id": point_id,
            "class_name": "car",
            "image_relpath": image_relpath,
            "split": "train",
            "bbox_xyxy": [0, 0, 100, 100],
            "projection": list(projection),
            "same_class_neighbor_ratio": same,
            "top_other_neighbor_ratio": other,
            "outlier_score": 0.03,
        }

    result = {
        "points": [
            point("cluster_0", (0.00, 0.00), "same_a.jpg"),
            point("cluster_1", (0.01, 0.01), "same_b.jpg"),
            point("cluster_2", (0.02, 0.00), "same_c.jpg"),
            point("far_right", (8.0, 0.0), "right.jpg", same=0.94),
            point("far_top", (0.0, 8.0), "top.jpg", same=0.94),
            point("far_left", (-8.0, 0.0), "left.jpg", same=0.94),
            point("wrong", (0.0, -8.0), "wrong.jpg", same=0.98),
        ]
    }
    result["points"][-1]["is_wrong_class_candidate"] = True

    selected = api._class_analysis_qwen_review_select_class_concept_examples(result, "car", limit=4)
    selected_ids = [item["point_id"] for item in selected]

    assert selected_ids[0] == "cluster_0"
    assert "wrong" not in selected_ids
    assert {"far_right", "far_top", "far_left"} & set(selected_ids)
    assert len({item["image_relpath"] for item in selected}) == len(selected)


def test_class_analysis_qwen_review_pair_must_skip_drops_obvious_class_examples():
    brief = api._class_analysis_qwen_review_normalize_pair_contrast(
        {
            "class_a": "Boat",
            "class_b": "SmallVehicle",
            "summary": "separate boats and cars",
            "choose_class_a_when": ["visible hull"],
            "choose_class_b_when": ["visible wheels"],
            "must_skip_when": [
                "Object is clearly a car on a road",
                "Target is clearly a boat on open water",
                "target is clipped or hidden",
                "overlap contamination hides the target",
            ],
        },
        class_a="Boat",
        class_b="SmallVehicle",
        glossary_a="",
        glossary_b="",
        review_guidance="",
        examples_a=[],
        examples_b=[],
    )

    assert "Object is clearly a car on a road" not in brief["must_skip_when"]
    assert "Target is clearly a boat on open water" not in brief["must_skip_when"]
    assert "target is clipped or hidden" in brief["must_skip_when"]
    assert "overlap contamination hides the target" in brief["must_skip_when"]


def test_class_analysis_qwen_review_concept_parser_handles_fenced_json():
    payload, error = api._class_analysis_qwen_review_parse_concept_payload(
        """```json
        {
          "class_name": "Truck",
          "summary": "large mobile vehicle",
          "visual_traits": ["box body",],
          "valid_variations": ["trailers"],
          "exclude_when": ["fixed roof"],
          "common_confusions": ["Building"],
          "uncertainty_triggers": ["partial overlap"]
        }
        ```"""
    )
    assert error is None
    assert payload["class_name"] == "Truck"
    assert payload["visual_traits"] == ["box body"]


def test_class_analysis_qwen_review_context_image_can_render_clean_crop(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_clean_context"
    workspace_dir = class_root / parent_id / "active_workspace"
    images_dir = workspace_dir / "images"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), (20, 80, 30)).save(images_dir / "scene.jpg")
    api._class_analysis_write_json(
        workspace_dir / "manifest.json",
        workspace_dir,
        {
            "labelmap": ["PoleFixture"],
            "images": [{"split": "train", "image_relpath": "scene.jpg", "label_lines": []}],
            "yolo_layout": "flat",
            "source_mode": "active_workspace",
        },
    )
    point = {
        "point_id": "p0",
        "class_name": "PoleFixture",
        "image_relpath": "scene.jpg",
        "split": "train",
        "bbox_xyxy": [20, 20, 60, 60],
    }
    result = {
        "summary": {"source_mode": "active_workspace", "source_id": parent_id, "labelmap": ["PoleFixture"]},
        "points": [point],
    }
    job = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_clean",
        parent_job_id=parent_id,
        point_id="p0",
    )

    boxed = api._class_analysis_qwen_review_context_image(
        job, result, point, max_dim=1000, draw_bbox=True
    )
    clean = api._class_analysis_qwen_review_context_image(
        job, result, point, max_dim=1000, draw_bbox=False
    )
    boxed_arr = np.asarray(boxed.convert("RGB"))
    clean_arr = np.asarray(clean.convert("RGB"))
    orange = np.asarray([249, 115, 22], dtype=np.uint8)

    assert np.any(np.all(boxed_arr == orange, axis=-1))
    assert not np.any(np.all(clean_arr == orange, axis=-1))

    observation = api._class_analysis_qwen_review_tool_target_detail(job, result, point, {})
    assert observation["evidence"][0]["kind"] == "target_detail"
    assert observation["evidence"][0]["metadata"]["bbox_overlay"] is False
    assert observation["evidence"][0]["metadata"]["deterministic_upscale"] is True
    assert observation["image_paths"]
    detail_arr = np.asarray(Image.open(observation["image_paths"][0]).convert("RGB"))
    assert not np.any(np.all(detail_arr == orange, axis=-1))


def test_class_analysis_qwen_review_local_consensus_context_filters_and_renders(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_local_consensus"
    workspace_dir = class_root / parent_id / "active_workspace"
    images_dir = workspace_dir / "images"
    images_dir.mkdir(parents=True)
    image = Image.new("RGB", (320, 240), (30, 60, 80))
    draw = ImageDraw.Draw(image)
    draw.rectangle([30, 30, 285, 210], fill=(70, 110, 130))
    image.save(images_dir / "scene.jpg")
    api._class_analysis_write_json(
        workspace_dir / "manifest.json",
        workspace_dir,
        {
            "labelmap": ["PoleFixture", "SmallVehicle", "Boat"],
            "images": [{"split": "train", "image_relpath": "scene.jpg", "label_lines": []}],
            "yolo_layout": "flat",
            "source_mode": "active_workspace",
        },
    )
    target = {
        "point_id": "target",
        "class_name": "PoleFixture",
        "suggested_neighbor_class": "SmallVehicle",
        "image_relpath": "scene.jpg",
        "split": "train",
        "bbox_xyxy": [90, 70, 130, 155],
    }
    result = {
        "summary": {
            "source_mode": "active_workspace",
            "source_id": parent_id,
            "labelmap": ["PoleFixture", "SmallVehicle", "Boat"],
        },
        "points": [
            target,
            {
                "point_id": "current_near",
                "class_name": "PoleFixture",
                "image_relpath": "scene.jpg",
                "split": "train",
                "bbox_xyxy": [42, 72, 62, 158],
            },
            {
                "point_id": "current_far",
                "class_name": "PoleFixture",
                "image_relpath": "scene.jpg",
                "split": "train",
                "bbox_xyxy": [220, 60, 242, 150],
            },
            {
                "point_id": "suggested_near",
                "class_name": "SmallVehicle",
                "image_relpath": "scene.jpg",
                "split": "train",
                "bbox_xyxy": [145, 140, 260, 188],
            },
            {
                "point_id": "other_class",
                "class_name": "Boat",
                "image_relpath": "scene.jpg",
                "split": "train",
                "bbox_xyxy": [10, 10, 38, 38],
            },
        ],
    }
    job = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_local_consensus",
        parent_job_id=parent_id,
        point_id="target",
    )

    clean, dots, metadata = api._class_analysis_qwen_review_local_consensus_context(job, result, target)
    observation = api._class_analysis_qwen_review_tool_local_consensus_context(job, result, target, {})

    assert clean.width > 0
    assert dots.height > clean.height
    assert metadata["same_image_current_count"] == 2
    assert metadata["same_image_suggested_count"] == 1
    assert metadata["included_current_count"] == 2
    assert metadata["included_suggested_count"] == 1
    assert all(item["class_name"] in {"PoleFixture", "SmallVehicle"} for item in metadata["included_points"])
    assert [row["kind"] for row in observation["evidence"]] == [
        "local_consensus_context",
        "local_consensus_clean_context",
        "local_consensus_dot_map",
    ]
    assert api._class_analysis_qwen_review_evidence_use(
        "local_consensus_clean_context"
    ) == "audit_visual"
    assert all(Path(path).is_file() for path in observation["image_paths"])
    with Image.open(observation["image_paths"][0]) as rendered:
        assert rendered.width <= 1200
        assert rendered.height <= 900
    assert "cannot override unclear target pixels" in observation["summary"]


def test_class_analysis_qwen_review_overlap_decomposition_marks_partial_contamination():
    point = {
        "point_id": "pole",
        "class_name": "PoleFixture",
        "split": "train",
        "image_relpath": "scene.jpg",
        "bbox_xyxy": [50, 20, 80, 170],
    }
    result = {
        "points": [
            point,
            {
                "point_id": "car",
                "class_name": "SmallVehicle",
                "split": "train",
                "image_relpath": "scene.jpg",
                "bbox_xyxy": [40, 110, 160, 160],
            },
            {
                "point_id": "other",
                "class_name": "Boat",
                "split": "train",
                "image_relpath": "other.jpg",
                "bbox_xyxy": [0, 0, 10, 10],
            },
        ]
    }

    overlaps = api._class_analysis_qwen_review_overlap_decomposition(result, point)

    assert len(overlaps) == 1
    assert overlaps[0]["point_id"] == "car"
    assert overlaps[0]["class_name"] == "SmallVehicle"
    assert overlaps[0]["relation"] == "partial_contamination"
    assert overlaps[0]["target_area_covered"] > 0.25


def test_class_analysis_qwen_review_anchor_selection_prefers_clean_class_anchors():
    point = {
        "point_id": "target",
        "class_name": "PoleFixture",
        "split": "train",
        "image_relpath": "scene.jpg",
    }
    result = {
        "points": [
            point,
            {
                "point_id": "clean",
                "class_name": "PoleFixture",
                "split": "train",
                "image_relpath": "other.jpg",
                "bbox_xyxy": [0, 0, 80, 80],
                "same_class_neighbor_ratio": 0.95,
                "top_other_neighbor_ratio": 0.05,
                "outlier_score": 0.1,
            },
            {
                "point_id": "suspicious",
                "class_name": "PoleFixture",
                "split": "train",
                "image_relpath": "other2.jpg",
                "bbox_xyxy": [0, 0, 100, 100],
                "same_class_neighbor_ratio": 0.05,
                "top_other_neighbor_ratio": 0.95,
                "outlier_score": 0.9,
                "is_wrong_class_candidate": True,
            },
        ]
    }

    anchors = api._class_analysis_qwen_review_select_anchors(
        result, point, "PoleFixture", same_image=False, limit=3
    )

    assert [anchor["point_id"] for anchor in anchors] == ["clean"]


def test_class_analysis_qwen_review_same_image_scale_report_is_generic_outlier():
    point = {
        "point_id": "target",
        "class_name": "CurrentClass",
        "split": "train",
        "image_relpath": "scene.jpg",
        "bbox_xyxy": [100, 100, 500, 500],
        "is_wrong_class_candidate": True,
    }
    anchors = [
        {
            "point_id": f"anchor{i}",
            "class_name": "CurrentClass",
            "split": "train",
            "image_relpath": "scene.jpg",
            "bbox_xyxy": [10 + i * 60, 20, 60 + i * 60, 70],
            "same_class_neighbor_ratio": 0.95,
            "top_other_neighbor_ratio": 0.02,
            "outlier_score": 0.05,
        }
        for i in range(4)
    ]
    result = {"points": [point, *anchors]}

    report = api._class_analysis_qwen_review_same_image_scale_report(result, point)

    assert report["signal"] == "questions_current"
    assert report["same_image_anchor_count"] == 4
    assert report["target_to_anchor_median_ratios"]["area_px2"] > 10.0
    assert "perspective" in report["policy"]


def test_class_analysis_qwen_review_same_image_embedding_report_uses_existing_vectors(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_embed"
    parent_dir = class_root / parent_id
    parent_dir.mkdir(parents=True)
    point = {
        "point_id": "target",
        "class_name": "CurrentClass",
        "split": "train",
        "image_relpath": "scene.jpg",
        "bbox_xyxy": [100, 100, 160, 160],
        "is_wrong_class_candidate": True,
    }
    anchors = [
        {
            "point_id": f"anchor{i}",
            "class_name": "CurrentClass",
            "split": "train",
            "image_relpath": "scene.jpg",
            "bbox_xyxy": [10 + i * 70, 20, 70 + i * 70, 80],
            "same_class_neighbor_ratio": 0.95,
            "top_other_neighbor_ratio": 0.02,
            "outlier_score": 0.05,
        }
        for i in range(3)
    ]
    result = {"points": [point, *anchors]}
    embeddings = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.99, 0.08, 0.0],
            [0.99, -0.08, 0.0],
        ],
        dtype=np.float32,
    )
    np.savez(parent_dir / "embeddings.npz", embeddings=embeddings)
    review = api.ClassAnalysisQwenReviewJob(
        review_id="r_embed",
        parent_job_id=parent_id,
        point_id="target",
    )

    report = api._class_analysis_qwen_review_same_image_embedding_report(review, result, point)

    assert report["signal"] == "questions_current"
    assert report["same_image_anchor_count"] == 3
    assert report["target_median_distance_percentile_vs_anchor_pairs"] >= 90.0
    assert report["target_to_current_anchor_cosine_distance"]["median"] > 0.9


def test_class_analysis_qwen_review_compact_final_defaults_deterministic_context():
    expanded = api._class_analysis_qwen_review_expand_compact_final(
        {
            "decision": "skip_uncertain",
            "target_class": "CurrentClass",
            "confidence": 0.2,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "weak",
            "target_evidence": "weak",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
            "visible_target_cues": [],
            "rationale_short": "deterministic context is not enough alone",
        },
        point={
            "point_id": "p0",
            "class_name": "CurrentClass",
            "suggested_neighbor_class": "SuggestedClass",
        },
        evidence_ids={"target_context_1"},
        visual_quality={"tier": "clear"},
        executed_tools={"inspect_same_image_scale_report", "inspect_same_image_embedding_report"},
        deterministic_context={
            "scale": {"signal": "questions_current"},
            "embedding": {"signal": "supports_current"},
        },
    )

    assert expanded["same_image_scale_evidence"] == "questions_current"
    assert expanded["same_image_embedding_evidence"] == "supports_current"


def test_class_analysis_qwen_review_reasoning_image_cap_resizes_and_drops_extra(tmp_path):
    image_paths = []
    for idx in range(4):
        path = tmp_path / f"evidence_{idx}.jpg"
        Image.new("RGB", (1000, 760), (20 + idx, 30, 40)).save(path)
        image_paths.append(path)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": str(path)} for path in image_paths],
        }
    ]

    capped, policy = api._class_analysis_qwen_review_cap_message_images(
        messages,
        max_images=3,
        max_side=512,
    )

    capped_images = [
        item["image"]
        for message in capped
        for item in message["content"]
        if item.get("type") == "image"
    ]
    assert len(capped_images) == 3
    assert policy["input_image_count"] == 4
    assert policy["output_image_count"] == 3
    assert policy["dropped_image_count"] == 1
    assert policy["max_side"] == 512
    assert len(policy["images"]) == 4
    assert policy["images"][0]["original_size"] == [1000, 760]
    assert policy["images"][0]["prepared_size"] == [512, 389]
    assert policy["images"][0]["rewritten"] is True
    assert policy["images"][3]["dropped"] is True
    assert policy["images"][3]["prepared_size"] is None
    for path in capped_images:
        with Image.open(path) as image:
            assert max(image.size) <= 512
    assert all("model_inputs" in path for path in capped_images)


def test_class_analysis_qwen_review_final_context_prefers_clean_contrast_before_overlay(monkeypatch):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_QWEN_REVIEW_FINAL_MAX_IMAGES", 3)

    def obs(tool_name: str, path: str) -> dict:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Tool result for {tool_name}.\nsummary\nEvidence ids: {tool_name}_1"},
                {"type": "image", "image": path},
            ],
        }

    messages = [
        obs("inspect_target_detail", "/tmp/target.jpg"),
        obs("zoom_source_region", "/tmp/zoom.jpg"),
        obs("inspect_source_overlay", "/tmp/overlay.jpg"),
        obs("inspect_specificity_region_contrast", "/tmp/contrast.jpg"),
    ]

    final_messages, policy = api._class_analysis_qwen_review_final_context_messages(messages)
    final_images = [
        item["image"]
        for message in final_messages
        for item in message.get("content", [])
        if item.get("type") == "image"
    ]

    assert final_images == ["/tmp/target.jpg", "/tmp/zoom.jpg", "/tmp/contrast.jpg"]
    assert policy["output_image_count"] == 3
    assert "inspect_specificity_region_contrast" in policy["image_observations"]
    assert "inspect_source_overlay" not in policy["image_observations"]


def test_class_analysis_qwen_review_deterministic_triage_is_guarded_human_signal():
    review = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_triage",
        parent_job_id="ca_triage",
        point_id="target",
    )
    review.evidence = [
        {
            "evidence_id": "local_consensus_context_10",
            "kind": "local_consensus_context",
            "metadata": {
                "same_image_current_count": 3,
                "same_image_suggested_count": 44,
                "included_current_count": 2,
                "included_suggested_count": 42,
                "nearest_current_distance_px": 968.0,
                "nearest_suggested_distance_px": 62.0,
            },
        }
    ]
    point = {
        "point_id": "target",
        "class_name": "Truck",
        "suggested_neighbor_class": "Building",
    }
    result = api._class_analysis_qwen_review_deterministic_triage_result(
        review,
        point,
        {"tier": "clear"},
        {
            "clean_visual_evidence_ids": ["target_detail_2"],
            "clean_target_source_evidence_ids": ["target_detail_2"],
        },
        {"embedding": {"signal": "questions_current"}, "scale": {"signal": "insufficient"}},
    )

    assert result is not None
    assert result["decision"] == "skip_uncertain"
    assert result["guarded_recommendation"]["decision"] == "accept_suggested"
    assert result["guarded_recommendation"]["target_class"] == "Building"
    assert result["human_review_needed"] is True
    disposition = api._class_analysis_qwen_review_disposition(
        {**result, "current_class": "Truck", "suggested_neighbor_class": "Building"}
    )
    assert disposition["signal"] == "guarded_human_triage"
    assert disposition["advisory_target_class"] == "Building"


def test_class_analysis_qwen_review_deterministic_triage_ignores_consensus_without_feature_support():
    review = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_triage_weak",
        parent_job_id="ca_triage",
        point_id="target",
    )
    review.evidence = [
        {
            "evidence_id": "local_consensus_context_10",
            "kind": "local_consensus_context",
            "metadata": {
                "same_image_current_count": 0,
                "same_image_suggested_count": 20,
                "included_current_count": 0,
                "included_suggested_count": 12,
                "nearest_current_distance_px": 0.0,
                "nearest_suggested_distance_px": 80.0,
            },
        }
    ]

    result = api._class_analysis_qwen_review_deterministic_triage_result(
        review,
        {"point_id": "target", "class_name": "Boat", "suggested_neighbor_class": "SmallVehicle"},
        {"tier": "clear"},
        {
            "clean_visual_evidence_ids": ["target_detail_2"],
            "clean_target_source_evidence_ids": ["target_detail_2"],
        },
        {"embedding": {"signal": "insufficient"}, "scale": {"signal": "insufficient"}},
    )

    assert result is None


def test_class_analysis_qwen_review_deterministic_triage_confirms_current_with_feature_support():
    review = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_triage_current",
        parent_job_id="ca_triage",
        point_id="target",
    )

    result = api._class_analysis_qwen_review_deterministic_triage_result(
        review,
        {"point_id": "target", "class_name": "Truck", "suggested_neighbor_class": "Building"},
        {"tier": "clear"},
        {
            "clean_visual_evidence_ids": ["target_detail_2", "zoom_region_9"],
            "clean_target_source_evidence_ids": ["target_detail_2"],
        },
        {"embedding": {"signal": "supports_current"}, "scale": {"signal": "supports_current"}},
    )

    assert result is not None
    assert result["decision"] == "confirm_current"
    assert result["target_class"] == "Truck"
    assert result["controller_preflight"]["kind"] == "deterministic_current_triage"
    assert result["same_image_embedding_evidence"] == "supports_current"
    assert result["same_image_scale_evidence"] == "supports_current"


def test_class_analysis_qwen_review_deterministic_triage_does_not_override_current_overlap():
    review = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_triage_overlap",
        parent_job_id="ca_triage",
        point_id="target",
    )
    review.evidence = [
        {
            "evidence_id": "local_consensus_context_10",
            "kind": "local_consensus_context",
            "metadata": {
                "same_image_current_count": 2,
                "same_image_suggested_count": 30,
                "included_current_count": 1,
                "included_suggested_count": 20,
                "nearest_current_distance_px": 900.0,
                "nearest_suggested_distance_px": 50.0,
            },
        }
    ]
    point = {
        "point_id": "target",
        "class_name": "Building",
        "suggested_neighbor_class": "SmallVehicle",
    }
    result = api._class_analysis_qwen_review_deterministic_triage_result(
        review,
        point,
        {"tier": "clear"},
        {
            "clean_visual_evidence_ids": ["target_detail_2"],
            "clean_target_source_evidence_ids": ["target_detail_2"],
            "overlap_decomposition": {
                "overlaps": [
                    {
                        "class_name": "Building",
                        "relation": "partial_contamination",
                        "target_area_covered": 0.63,
                        "other_area_covered": 0.8,
                        "iou": 0.4,
                    },
                    {
                        "class_name": "SmallVehicle",
                        "relation": "partial_contamination",
                        "target_area_covered": 0.15,
                        "other_area_covered": 0.2,
                        "iou": 0.1,
                    },
                ]
            },
        },
        {"embedding": {"signal": "questions_current"}, "scale": {"signal": "neutral"}},
    )

    assert result is None


def test_class_analysis_qwen_review_mlx_final_disabled_returns_completed_skip():
    point = {
        "point_id": "target",
        "class_name": "Truck",
        "suggested_neighbor_class": "Building",
    }
    result = api._class_analysis_qwen_review_mlx_final_disabled_result(
        point,
        {"tier": "clear", "reasons": []},
        {"clean_visual_evidence_ids": ["target_detail_2"]},
    )

    assert result["decision"] == "skip_uncertain"
    assert result["backend_visual_quality"]["tier"] == "clear"
    assert result["visual_quality"] == "clear"
    assert result["controller_preflight"]["kind"] == "mlx_final_disabled"
    disposition = api._class_analysis_qwen_review_disposition(
        {**result, "current_class": "Truck", "suggested_neighbor_class": "Building"}
    )
    assert disposition["signal"] == "no_signal"
    assert "MLX Qwen final generation" in disposition["primary_reason"]


def test_class_analysis_qwen_review_initial_prompt_includes_glossary_and_guidance():
    text = api._class_analysis_qwen_review_initial_user_message(
        {"summary": {"labelmap": ["PoleFixture", "SmallVehicle"]}},
        {
            "point_id": "p0",
            "class_name": "PoleFixture",
            "suggested_neighbor_class": "SmallVehicle",
        },
        {"tier": "clear", "bbox_width": 50, "bbox_height": 100},
        labelmap_glossary='{"PoleFixture":["elevated fixture","mounted sign"]}',
        review_guidance="PoleFixture includes project-specific obstruction fixtures in this dataset.",
    )

    assert "Relevant class meaning glossary" in text
    assert "mounted sign" in text
    assert "Additional review guidance" in text
    assert "project-specific obstruction" in text


def test_class_analysis_flags_very_close_overlap_candidates():
    records = [
        {
            **_record("p0", "boat"),
            "image_relpath": "shared.jpg",
            "bbox_xyxy": [10, 10, 50, 50],
        },
        {
            **_record("p1", "building"),
            "image_relpath": "shared.jpg",
            "bbox_xyxy": [10.5, 10.5, 49.5, 49.5],
        },
        {
            **_record("p2", "tree"),
            "image_relpath": "shared.jpg",
            "bbox_xyxy": [80, 80, 120, 120],
        },
        {
            **_record("p3", "tree"),
            "image_relpath": "other.jpg",
            "bbox_xyxy": [10, 10, 50, 50],
        },
    ]
    embeddings = np.eye(4, dtype=np.float32)

    result = api._class_analysis_build_result(
        records,
        embeddings,
        summary={"analysis_scope": "all_classes"},
        projection="pca",
        projection_neighbor_k=15,
        neighbor_k=3,
        seed=13,
    )

    assert result["summary"]["close_overlap_pair_count"] == 1
    assert result["summary"]["close_overlap_candidate_count"] == 2
    overlap_ids = {
        point["point_id"]
        for point in result["points"]
        if point.get("is_close_overlap_candidate")
    }
    assert overlap_ids == {"p0", "p1"}
    assert result["close_overlap_candidates"][0]["class_name"] == "boat"
    assert result["close_overlap_candidates"][0]["other_class_name"] == "building"
    p0 = next(point for point in result["points"] if point["point_id"] == "p0")
    assert "close_overlap" in p0["review_signals"]


def test_class_analysis_cluster_search_reuses_selected_class_embeddings():
    records = [_record(f"p{i}", "vehicle") for i in range(8)]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.98, -0.01, 0.0],
            [1.0, 0.02, 0.0],
            [0.0, 1.0, 0.0],
            [0.01, 0.99, 0.0],
            [-0.01, 0.98, 0.0],
            [0.02, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    cluster_job = api.ClassAnalysisClusterJob(job_id="cac_unit", parent_job_id="ca_unit")
    result = api._class_analysis_cluster_search_result(
        job=cluster_job,
        points=records,
        embeddings=embeddings,
        payload={
            "proposal_source": "embedding_kmeans",
            "sensitivity": "sensitive",
            "max_clusters": 4,
            "min_cluster_size": 2,
            "seed": 13,
        },
    )

    clusters = result["clusters"]
    assert len(clusters) == 2
    assert result["summary"]["cluster_count"] == 2
    assert result["summary"]["proposal_source"] == "embedding_kmeans"
    assert result["summary"]["best_k"] == 2
    assert set(result["labels_by_point_id"]) == {record["point_id"] for record in records}
    assert sorted(cluster["size"] for cluster in clusters) == [4, 4]


def test_class_analysis_cluster_search_umap_islands_can_propose_visual_tail(monkeypatch):
    records = [_record(f"p{i}", "vehicle") for i in range(8)]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.98, -0.01, 0.0],
            [1.0, 0.02, 0.0],
            [0.0, 1.0, 0.0],
            [0.01, 0.99, 0.0],
            [-0.01, 0.98, 0.0],
            [0.02, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def fake_project(embeddings_arg, *, projection, projection_neighbor_k, projection_min_dist, seed, warnings):
        assert projection == "umap"
        assert projection_neighbor_k == 4
        assert projection_min_dist == 0.02
        return np.asarray(
            [
                [0.00, 0.00],
                [0.02, 0.00],
                [0.00, 0.02],
                [0.02, 0.02],
                [4.00, 0.00],
                [4.02, 0.00],
                [4.00, 0.02],
                [4.02, 0.02],
            ],
            dtype=np.float32,
        ), "umap"

    monkeypatch.setattr(api, "_class_analysis_project_embeddings", fake_project)
    cluster_job = api.ClassAnalysisClusterJob(job_id="cac_unit_umap", parent_job_id="ca_unit")
    result = api._class_analysis_cluster_search_result(
        job=cluster_job,
        points=records,
        embeddings=embeddings,
        payload={
            "proposal_source": "umap_islands",
            "sensitivity": "sensitive",
            "max_clusters": 4,
            "min_cluster_size": 2,
            "umap_neighbors": 4,
            "umap_min_dist": 0.02,
            "seed": 13,
        },
    )

    assert result["summary"]["proposal_source"] == "umap_islands"
    assert result["summary"]["method"] == "umap_dbscan"
    assert result["summary"]["cluster_count"] == 2
    assert sorted(cluster["size"] for cluster in result["clusters"]) == [4, 4]
    assert set(result["labels_by_point_id"]) == {record["point_id"] for record in records}


def test_class_analysis_stratified_sampling_keeps_classes_represented():
    records = [_record(f"a{i}", "alpha") for i in range(10)]
    records.extend(_record(f"b{i}", "beta") for i in range(10))

    selected = api._class_analysis_stratified_indices(records, cap=6, seed=7)
    selected_classes = [records[idx]["class_name"] for idx in selected]

    assert len(selected) == 6
    assert "alpha" in selected_classes
    assert "beta" in selected_classes


def test_class_analysis_sample_cap_defaults_to_unlimited():
    assert api._class_analysis_sample_cap(None) == 0
    assert api._class_analysis_sample_cap("") == 0
    assert api._class_analysis_sample_cap("0") == 0
    assert api._class_analysis_sample_cap("-5") == 0
    assert api._class_analysis_sample_cap("250") == 250

    records = [_record(f"p{i}", "alpha") for i in range(12)]
    assert api._class_analysis_stratified_indices(records, cap=0, seed=7) == list(range(12))


def test_class_analysis_normalizes_pca_projection_modes():
    request = api._normalize_class_analysis_request({"projection": "between_class_pca"})
    assert request["projection"] == "pca"
    assert request["projection_mode"] == "between_class_pca"
    assert request["embedding_view_mode"] == "tight_context"
    assert request["projection_metric"] == api.CLASS_ANALYSIS_DEFAULT_UMAP_METRIC
    assert request["projection_spread"] == api.CLASS_ANALYSIS_DEFAULT_UMAP_SPREAD

    fallback = api._normalize_class_analysis_request({"projection_mode": "unknown"})
    assert fallback["projection"] == "pca"
    assert fallback["projection_mode"] == api.CLASS_ANALYSIS_DEFAULT_PCA_PROJECTION_MODE


def test_class_analysis_normalizes_umap_projection_hyperparams():
    request = api._normalize_class_analysis_request(
        {
            "projection": "umap",
            "projection_metric": "cosine",
            "projection_spread": "2.5",
            "projection_min_dist": "0.2",
            "projection_preprocess": "l2",
        }
    )
    assert request["projection"] == "umap"
    assert request["projection_metric"] == "cosine"
    assert request["projection_spread"] == 2.5
    assert request["projection_min_dist"] == 0.2
    assert request["projection_preprocess"] == "l2"

    fallback = api._normalize_class_analysis_request(
        {
            "projection": "umap",
            "projection_metric": "invalid_metric",
            "projection_spread": 0.0,
            "projection_min_dist": 2.0,
            "projection_preprocess": "invalid_preprocess",
        }
    )
    assert fallback["projection_metric"] == api.CLASS_ANALYSIS_DEFAULT_UMAP_METRIC
    assert fallback["projection_spread"] == 0.1
    assert fallback["projection_min_dist"] == 0.99
    assert fallback["projection_preprocess"] == api.CLASS_ANALYSIS_DEFAULT_PROJECTION_PREPROCESS


def test_class_analysis_normalizes_projection_preprocess_modes():
    assert api._class_analysis_normalize_projection_preprocess_mode("center") == "center"
    assert api._class_analysis_normalize_projection_preprocess_mode("mean_center") == "center"
    assert api._class_analysis_normalize_projection_preprocess_mode("z_score") == "zscore"
    assert api._class_analysis_normalize_projection_preprocess_mode("invalid") == api.CLASS_ANALYSIS_DEFAULT_PROJECTION_PREPROCESS


def test_class_analysis_apply_projection_preprocess_modes_affect_embeddings():
    embeddings = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
        ],
        dtype=np.float32,
    )
    centered = api._class_analysis_apply_projection_preprocess(embeddings, "center")
    expected_centered = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    assert np.allclose(centered, expected_centered)

    zscore = api._class_analysis_apply_projection_preprocess(embeddings, "zscore")
    mean = np.mean(embeddings, axis=0, keepdims=True)
    std = np.std(embeddings, axis=0, keepdims=True)
    expected_zscore = (embeddings - mean) / np.where(std > 1e-12, std, 1.0)
    assert np.allclose(zscore, expected_zscore)

    unchanged = api._class_analysis_apply_projection_preprocess(embeddings, "none")
    assert np.allclose(unchanged, embeddings)


def test_class_analysis_result_carries_switchable_pca_coordinates():
    records = [
        _record("a0", "alpha"),
        _record("a1", "alpha"),
        _record("a2", "alpha"),
        _record("b0", "beta"),
        _record("b1", "beta"),
        _record("b2", "beta"),
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.98, 0.02, 0.0, 0.0],
            [0.99, -0.02, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.02, 0.98, 0.0, 0.0],
            [-0.02, 0.99, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    result = api._class_analysis_build_result(
        records,
        embeddings,
        summary={"analysis_scope": "all_classes"},
        projection="pca",
        projection_mode="between_class_pca",
        projection_neighbor_k=15,
        neighbor_k=3,
        seed=13,
    )

    options = result["projection_options"]
    assert options["selected"] == "between_class_pca"
    assert options["available"] == api.CLASS_ANALYSIS_PCA_PROJECTION_MODES
    assert set(options["coordinates"]) == set(api.CLASS_ANALYSIS_PCA_PROJECTION_MODES)
    for coords in options["coordinates"].values():
        assert coords.shape == (len(records), 2)
        assert np.isfinite(coords).all()
    first_point = result["points"][0]
    assert first_point["projection"] == pytest.approx(options["coordinates"]["between_class_pca"][0].tolist())
    assert result["summary"]["projection"] == "pca"
    assert result["summary"]["projection_mode"] == "between_class_pca"
    projection_metadata = result["summary"]["projection_metadata"]
    assert projection_metadata["requested"]["projection"] == "pca"
    assert projection_metadata["requested"]["projection_mode"] == "between_class_pca"
    assert projection_metadata["requested"]["projection_metric"] is None
    assert projection_metadata["requested"]["projection_min_dist"] is None
    assert projection_metadata["requested"]["projection_spread"] is None
    assert projection_metadata["resolved"]["projection"] == "pca"
    assert projection_metadata["resolved"]["projection_mode"] == "between_class_pca"
    assert projection_metadata["resolved"]["projection_neighbor_k"] is None
    assert result["summary"]["projection_recommended_mode"] in api.CLASS_ANALYSIS_PCA_PROJECTION_MODES
    assert isinstance(result["summary"]["projection_quality_by_mode"], dict)
    assert set(result["summary"]["projection_quality_by_mode"]) == set(api.CLASS_ANALYSIS_PCA_PROJECTION_MODES)
    for mode in api.CLASS_ANALYSIS_PCA_PROJECTION_MODES:
        quality = result["summary"]["projection_quality_by_mode"][mode]
        assert set(quality).issuperset(
            {
                "class_separation_ratio",
                "class_silhouette",
                "trustworthiness",
                "separation_score",
                "overview_score",
                "projection_recommendation",
            }
        )
        assert quality["projection_recommendation"] in {"overview", "class_separation", "balanced"}

    public = api._class_analysis_public_result(result)
    assert "coordinates" not in public["projection_options"]
    assert public["projection_options"]["coordinates_available"] == api.CLASS_ANALYSIS_PCA_PROJECTION_MODES
    assert public["points"][0]["projection"] == pytest.approx(first_point["projection"], abs=1e-6)


def test_class_analysis_public_result_strips_non_ui_point_bulk():
    result = {
        "summary": {"projection": "pca", "projection_mode": "class_balanced_pca"},
        "projection_options": {
            "selected": "class_balanced_pca",
            "coordinates": {
                "class_balanced_pca": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
            },
        },
        "points": [
            {
                "point_id": "p0",
                "class_name": "car",
                "bbox_xyxy": [0, 0, 10, 10],
                "projection": [0.0, 1.0],
                "thumbnail_url": "/thumb/p0",
                "neighbor_ids": ["p1"],
                "neighbor_distances": [0.12],
                "neighbor_class_counts": {"boat": 1},
                "embedding_views": [{"path": "internal"}],
                "crop_cache_key": "cache-key",
                "crop_cache_reused": True,
                "label_line": "0 0.5 0.5 0.1 0.1",
                "crop_xyxy": [0, 0, 12, 12],
                "is_wrong_class_candidate": False,
                "review_signals": [],
            },
            {
                "point_id": "p1",
                "class_name": "boat",
                "bbox_xyxy": [2, 2, 12, 12],
                "projection": [1.0, 0.0],
                "thumbnail_url": "/thumb/p1",
                "neighbor_ids": ["p0"],
                "neighbor_distances": [0.12],
                "neighbor_class_counts": {"car": 1},
                "embedding_views": [{"path": "internal"}],
                "crop_cache_key": "cache-key-2",
                "is_wrong_class_candidate": True,
                "review_signals": ["wrong_class"],
            },
        ],
    }

    public = api._class_analysis_public_result(result)

    assert "coordinates" not in public["projection_options"]
    assert public["projection_options"]["coordinates_available"] == ["class_balanced_pca"]
    graph_point = public["points"][0]
    assert graph_point["point_id"] == "p0"
    assert graph_point["projection"] == [0.0, 1.0]
    assert "neighbor_ids" not in graph_point
    assert "neighbor_distances" not in graph_point
    assert "neighbor_class_counts" not in graph_point
    assert "embedding_views" not in graph_point
    assert "crop_cache_key" not in graph_point
    assert "crop_cache_reused" not in graph_point
    assert "label_line" not in graph_point
    assert "crop_xyxy" not in graph_point

    review_point = public["points"][1]
    assert review_point["neighbor_ids"] == ["p0"]
    assert review_point["neighbor_distances"] == [0.12]
    assert review_point["neighbor_class_counts"] == {"car": 1}
    assert "embedding_views" not in review_point
    assert "crop_cache_key" not in review_point


def test_class_analysis_disk_result_publicizes_request_payload_in_place(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_dir = api._class_analysis_job_dir("job_public_in_place", create=True)
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(
        result_path,
        class_root,
        {
            "summary": {"analysis_scope": "all_classes"},
            "points": [{"point_id": "p0", "projection": [0.0, 1.0]}],
            "wrong_class_candidates": [],
        },
    )
    job = api.ClassAnalysisJob(
        job_id="job_public_in_place",
        status="completed",
        result_path=str(result_path),
    )
    public_calls = []
    original_public_result = api._class_analysis_public_result

    def track_public_result(result, *, in_place=False):
        public_calls.append(bool(in_place))
        return original_public_result(result, in_place=in_place)

    monkeypatch.setattr(api, "_class_analysis_public_result", track_public_result)
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS[job.job_id] = job
    try:
        payload = api.get_class_analysis_result(job.job_id)
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)

    assert payload["points"] == [{"point_id": "p0", "projection": [0, 1]}]
    assert public_calls == [True]


def test_class_analysis_review_overlay_preserves_point_without_dispositions(
    monkeypatch,
):
    point = {"point_id": "p0", "class_name": "Boat"}
    result = {
        "summary": {},
        "points": [point],
        "wrong_class_candidates": [
            {"point_id": "p0", "review_object_key": "cro_" + "1" * 64}
        ],
    }
    monkeypatch.setattr(
        api,
        "_class_analysis_lookup_review_dispositions",
        lambda _keys: {},
    )

    overlaid = api._class_analysis_apply_review_dispositions(result)

    assert overlaid["points"] == [point]
    assert "human_review_disposition" not in point
    assert overlaid["summary"]["wrong_class_candidate_count"] == 1
    assert overlaid["summary"]["human_review_disposition_counts"] == {}


def test_class_analysis_umap_fallback_is_labeled_as_global_pca(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "umap", None)
    records = [
        _record("a0", "alpha"),
        _record("a1", "alpha"),
        _record("a2", "alpha"),
        _record("b0", "beta"),
        _record("b1", "beta"),
        _record("b2", "beta"),
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.98, 0.02, 0.0, 0.0],
            [0.99, -0.02, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.02, 0.98, 0.0, 0.0],
            [-0.02, 0.99, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    result = api._class_analysis_build_result(
        records,
        embeddings,
        summary={"analysis_scope": "all_classes"},
        projection="umap",
        projection_mode="class_balanced_pca",
        projection_neighbor_k=15,
        neighbor_k=3,
        seed=13,
    )

    options = result["projection_options"]
    assert result["summary"]["projection"] == "pca"
    assert result["summary"]["projection_mode"] == "global_pca"
    projection_metadata = result["summary"]["projection_metadata"]
    assert projection_metadata["requested"]["projection"] == "umap"
    assert projection_metadata["requested"]["projection_mode"] == "class_balanced_pca"
    assert projection_metadata["requested"]["projection_metric"] == api._class_analysis_normalize_umap_projection_metric("cosine")
    assert projection_metadata["requested"]["projection_min_dist"] == 0.08
    assert projection_metadata["requested"]["projection_spread"] == 1.0
    assert projection_metadata["resolved"]["projection"] == "pca"
    assert projection_metadata["resolved"]["projection_mode"] == "global_pca"
    assert projection_metadata["resolved"]["projection_neighbor_k"] is None
    assert options["selected"] == "global_pca"
    assert result["points"][0]["projection"] == pytest.approx(options["coordinates"]["global_pca"][0].tolist())
    assert any("UMAP unavailable" in warning for warning in result["summary"]["warnings"])


def test_class_analysis_direct_job_rejects_missing_source():
    with pytest.raises(api.HTTPException) as exc_info:
        api.create_class_analysis_job({})

    assert exc_info.value.status_code == api.HTTP_400_BAD_REQUEST
    assert exc_info.value.detail == "dataset_id_required"


def test_class_analysis_rejects_local_salad_aggregation_before_queue():
    with pytest.raises(api.HTTPException) as disabled:
        api._normalize_class_analysis_request(
            {
                "encoder_type": "dinov3",
                "embedding_aggregation": "local_salad",
                "embedding_salad_head_id": "unit_head",
            }
        )
    assert disabled.value.status_code == 400
    assert disabled.value.detail == "local_salad_class_analysis_disabled"

    pooled = api._normalize_class_analysis_request(
        {
            "encoder_type": "dinov3",
            "embedding_aggregation": "pooled",
            "embedding_salad_head_id": "stale_head",
        }
    )
    assert pooled["embedding_aggregation"] == "pooled"
    assert pooled["embedding_salad_head_id"] == ""


def test_auto_class_training_rejects_local_salad_aggregation_before_dataset_validation():
    with pytest.raises(api.HTTPException) as disabled:
        asyncio.run(
            api.start_clip_training(
                embedding_aggregation="local_salad",
                embedding_salad_head_id="unit_head",
            )
        )
    assert disabled.value.status_code == 400
    assert disabled.value.detail == "local_salad_auto_class_disabled"


def test_auto_class_training_cleans_staged_upload_on_dataset_validation_error(
    tmp_path,
    monkeypatch,
):
    staged_root = tmp_path / "clip_train_fixed"

    def fake_mkdtemp(prefix=""):
        staged_root.mkdir(parents=True, exist_ok=True)
        return str(staged_root)

    monkeypatch.setattr(api.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(api.tempfile, "gettempdir", lambda: str(tmp_path))
    image = UploadFile(filename="a.jpg", file=BytesIO(b"image-bytes"))
    empty_label = UploadFile(filename="a.txt", file=BytesIO(b""))

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(
            api.start_clip_training(
                images=[image],
                labels=[empty_label],
                labelmap=None,
                clip_model_name=api.DEFAULT_CLIP_MODEL,
                encoder_type="clip",
                encoder_model=None,
                output_dir=".",
                images_path_native=None,
                labels_path_native=None,
                labelmap_path_native=None,
                solver="saga",
                classifier_type="logreg",
                embedding_aggregation="pooled",
                embedding_salad_head_id="",
                reuse_embeddings=None,
                hard_example_mining=None,
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "clip_labels_empty"
    assert not staged_root.exists()


def test_auto_class_runtime_rejects_local_salad_artifacts_before_encoding():
    with pytest.raises(api.HTTPException) as disabled:
        api._encode_pil_batch_for_head(
            [Image.new("RGB", (8, 8), (10, 20, 30))],
            head={
                "encoder_type": "dinov3",
                "embedding_aggregation": "local_salad",
                "embedding_salad_head_id": "unit_head",
            },
        )
    assert disabled.value.status_code == 400
    assert disabled.value.detail == "local_salad_auto_class_disabled"


def test_cradio_embedding_contract_and_capabilities(monkeypatch):
    assert normalize_cradio_pooling("spatial") == "spatial_mean"
    assert normalize_cradio_pooling("summary+spatial") == "summary_spatial_concat"
    assert normalize_cradio_pooling("anything_else") == "summary"

    monkeypatch.setattr(
        cradio_embedding_utils,
        "_cradio_mlx_backend_status",
        lambda model_name=None, *, requested="mlx": CRadioBackendStatus(
            requested=requested,
            resolved="mlx",
            available=True,
            detail="Local MLX C-RADIOv4 backend (/tmp/model.safetensors)",
        ),
    )

    mlx = cradio_backend_status("mlx")
    assert mlx.resolved == "mlx"
    assert mlx.available is True
    assert "Local MLX C-RADIOv4 backend" in mlx.detail

    def model_specific_mlx_status(model_name=None, *, requested="mlx"):
        model = model_name or CRADIO_DEFAULT_MODEL
        return CRadioBackendStatus(
            requested=requested,
            resolved="mlx",
            available=model == CRADIO_DEFAULT_MODEL,
            detail=f"mlx status for {model}",
        )

    monkeypatch.setattr(cradio_embedding_utils, "_cradio_mlx_backend_status", model_specific_mlx_status)
    monkeypatch.setattr(cradio_embedding_utils.platform, "system", lambda: "Darwin")
    assert cradio_backend_status("auto", model_name=CRADIO_DEFAULT_MODEL).resolved == "mlx"
    assert cradio_backend_status("auto", model_name="nvidia/C-RADIOv4-H").resolved != "mlx"

    summary = torch.ones(2, 3)
    spatial = torch.zeros(2, 4, 3)
    unpacked = _unpack_cradio_outputs({"summary": summary, "spatial_features": spatial})
    assert unpacked[0] is summary
    assert unpacked[1] is spatial

    class FakeMLXEncoder:
        def encode_batch(self, images, image_size=512):
            assert len(images) == 2
            assert image_size == 512
            return types.SimpleNamespace(
                summary=np.asarray([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32),
                spatial=np.asarray(
                    [
                        [[1.0, 0.0], [0.0, 1.0]],
                        [[2.0, 0.0], [0.0, 2.0]],
                    ],
                    dtype=np.float32,
                ),
            )

    mlx_images = [Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32))]
    mlx_feats, mlx_spatial, mlx_summary = encode_cradio_images(
        FakeMLXEncoder(),
        None,
        "mlx",
        mlx_images,
        pooling="summary_spatial_concat",
        normalize=True,
        return_tokens=True,
    )
    assert mlx_feats.shape == (2, 4)
    assert mlx_spatial.shape == (2, 2, 2)
    assert mlx_summary.shape == (2, 2)
    assert np.allclose(np.linalg.norm(mlx_feats, axis=1), np.ones(2), atol=1e-6)

    caps = api._class_analysis_capabilities()
    assert "cradio" in caps["encoders"]
    assert caps["default_cradio_model"] == CRADIO_DEFAULT_MODEL
    assert "summary_spatial_concat" in caps["cradio_pooling_modes"]
    assert any(recipe["id"] == "cradio_summary" for recipe in caps["class_separation_recipes"])

    request = api._normalize_class_analysis_request(
        {
            "encoder_type": "cradio",
            "encoder_model": "",
            "cradio_pooling": "summary+spatial",
            "embedding_aggregation": "pooled",
            "embedding_salad_head_id": "stale_head",
        }
    )
    assert request["encoder_model"] == CRADIO_DEFAULT_MODEL
    assert request["cradio_pooling"] == "summary_spatial_concat"
    assert request["embedding_salad_head_id"] == ""


def test_cradio_head_encoding_uses_saved_pooling(monkeypatch):
    captured = {}

    monkeypatch.setattr(api, "resolve_cradio_torch_device", lambda _backend=None, **_kwargs: "cpu")
    monkeypatch.setattr(
        api,
        "_load_cradio_backbone_cached",
        lambda model_name, target_device, raise_on_error=False: ("model", "processor", model_name, "cpu"),
    )

    def fake_encode(model, processor, device_name, images, *, pooling, normalize=True, return_tokens=False):
        captured["pooling"] = pooling
        captured["normalize"] = normalize
        captured["return_tokens"] = return_tokens
        return np.asarray([[1.0, 2.0, 3.0] for _ in images], dtype=np.float32)

    monkeypatch.setattr(api, "encode_cradio_images", fake_encode)
    feats = api._encode_pil_batch_for_head(
        [Image.new("RGB", (8, 8), (10, 20, 30))],
        head={
            "encoder_type": "cradio",
            "encoder_model": "nvidia/C-RADIOv4-SO400M",
            "cradio_pooling": "spatial_mean",
            "normalize_embeddings": True,
        },
    )

    assert captured == {"pooling": "spatial_mean", "normalize": False, "return_tokens": False}
    assert feats.shape == (1, 3)
    assert np.allclose(np.linalg.norm(feats, axis=1), 1.0)


def test_clip_head_encoding_uses_saved_clip_model_without_mutating_active(monkeypatch):
    loaded = []
    encoded = []

    class ActiveClipModel:
        def encode_image(self, _batch):
            raise AssertionError("active CLIP backbone should not be used")

    class SavedClipModel:
        def __init__(self, name):
            self.name = name

        def encode_image(self, batch):
            encoded.append(self.name)
            return torch.full((int(batch.shape[0]), 2), 3.0, dtype=torch.float32)

    def fake_load(name, device=None):
        loaded.append((name, device))
        return SavedClipModel(name), lambda _img: torch.zeros(3, 8, 8)

    monkeypatch.setattr(api, "clip_model", ActiveClipModel())
    monkeypatch.setattr(api, "clip_preprocess", lambda _img: torch.ones(3, 8, 8))
    monkeypatch.setattr(api, "clip_model_name", "ViT-B/32")
    monkeypatch.setattr(api, "_clip_reload_needed", False)
    api._agent_clip_backbones.clear()
    api._agent_clip_locks.clear()
    monkeypatch.setattr(api.clip, "load", fake_load)

    feats = api._encode_pil_batch_for_head(
        [Image.new("RGB", (8, 8), (10, 20, 30))],
        head={
            "encoder_type": "clip",
            "encoder_model": "ViT-L/14",
            "normalize_embeddings": False,
        },
        device_override="cpu",
    )

    assert loaded == [("ViT-L/14", "cpu")]
    assert encoded == ["ViT-L/14"]
    assert feats.shape == (1, 2)
    assert np.allclose(feats, np.asarray([[3.0, 3.0]], dtype=np.float32))
    assert api.clip_model_name == "ViT-B/32"
    assert isinstance(api.clip_model, ActiveClipModel)


def test_class_analysis_capabilities_expose_only_normal_recipe_controls():
    caps = api._class_analysis_capabilities()

    assert caps["review_disposition_api_version"] == 3
    assert caps["review_class_reassignment_api_version"] == 2
    assert caps["review_single_bbox_deletion_api_version"] == 2
    assert caps["review_history_delete_api_version"] == 1
    assert caps["dual_bbox_resolution_api_version"] == 1
    assert caps["dual_bbox_annotation_transaction_api_version"] == 2
    assert caps["preprocess_modes"] == ["canonical"]
    assert caps["embedding_adjustments"] == ["remove_size_bias"]
    assert caps["expert_preprocess_modes"] == ["native", "canonical"]
    history_route = next(
        route
        for route in api.app.routes
        if getattr(route, "path", "")
        == "/class_analysis/jobs/{job_id}/review_history/delete"
    )
    assert "POST" in history_route.methods
    assert caps["expert_embedding_adjustments"] == ["none", "remove_size_bias"]
    assert caps["default_preprocess_mode"] == "canonical"
    assert caps["default_embedding_adjustment"] == "remove_size_bias"
    assert caps["default_projection_neighbor_k"] == 50
    assert caps["default_projection_min_dist"] == 0.08
    assert caps["default_projection_preprocess"] == "none"
    assert set(caps["projection_preprocess_modes"]) == set(api.CLASS_ANALYSIS_PROJECTION_PREPROCESS_MODES)
    assert caps["default_projection_spread"] == 1.0
    assert caps["default_projection_metric"] == "cosine"
    assert set(caps["umap_projection_metrics"]) == set(api.CLASS_ANALYSIS_UMAP_PROJECTION_METRICS)
    assert caps["subclass_cluster_sources"] == ["umap_islands", "embedding_kmeans"]
    assert caps["default_subclass_cluster_source"] == "umap_islands"
    assert caps["default_subclass_umap_neighbors"] == 15
    assert caps["default_subclass_umap_min_dist"] == 0.02
    assert caps["default_pca_projection_mode"] == "class_balanced_pca"
    assert caps["pca_projection_modes"] == api.CLASS_ANALYSIS_PCA_PROJECTION_MODES
    assert "pooled" in caps["embedding_aggregation_modes"]
    assert api.CLASS_ANALYSIS_SAM3_MASK_FUSION_SCHEMA in caps["embedding_aggregation_modes"]
    assert api.CLASS_ANALYSIS_SAM3_SALAD_FUSION_SCHEMA in caps["embedding_aggregation_modes"]
    assert caps["sam3_salad_fusion"]["training_policy"] == (
        "bounded_source_diverse_label_free_reservoir_v1"
    )
    assert caps["sam3_salad_fusion"]["all_objects_encoded"] is True
    assert "local_salad_heads" not in caps
    assert "local_salad_policy" not in caps
    assert not any(recipe["id"] == "local_salad" for recipe in caps["class_separation_recipes"])


def test_dinov3_head_encoding_uses_default_model_constant(monkeypatch):
    class DummyProcessor:
        def __call__(self, images, return_tensors="pt"):
            assert return_tensors == "pt"
            return {"pixel_values": torch.zeros(len(images), 1)}

    class DummyModel:
        def __call__(self, **inputs):
            batch = int(inputs["pixel_values"].shape[0])
            return types.SimpleNamespace(
                last_hidden_state=torch.ones(batch, 2, 2),
                pooler_output=torch.tensor([[3.0, 4.0], [0.0, 5.0]], dtype=torch.float32)[:batch],
            )

    monkeypatch.setattr(api, "dinov3_model", DummyModel())
    monkeypatch.setattr(api, "dinov3_processor", DummyProcessor())
    monkeypatch.setattr(api, "dinov3_model_name", api.CLASS_ANALYSIS_DEFAULT_DINOV3_MODEL)
    monkeypatch.setattr(api, "dinov3_model_device", "cpu")
    monkeypatch.setattr(api, "_load_dinov3_backbone", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected load")))

    feats = api._encode_pil_batch_for_head(
        [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))],
        head={"encoder_type": "dinov3", "normalize_embeddings": True},
        device_override="cpu",
    )

    assert feats.shape == (2, 2)
    assert np.allclose(np.linalg.norm(feats, axis=1), np.ones(2), atol=1e-6)


def test_class_analysis_source_reads_active_workspace_manifest(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    workspace = class_root / "workspace"
    (workspace / "images").mkdir(parents=True)
    manifest_path = workspace / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_label": "browser snapshot",
                "labelmap": ["car", "boat"],
                "images": [
                    {
                        "split": "train",
                        "image_relpath": "example.jpg",
                        "frontend_image_key": "train/original/example.jpg",
                        "label_lines": ["0 0.5 0.5 0.2 0.2"],
                    }
                ],
                "yolo_layout": "flat",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    source = api._class_analysis_source(
        {
            "source_mode": "active_workspace",
            "workspace_id": "ca_test",
            "workspace_dir": str(workspace),
            "workspace_manifest_path": str(manifest_path),
        }
    )

    assert source["source_mode"] == "active_workspace"
    assert source["source_id"] == "ca_test"
    assert source["dataset_root"] == workspace.resolve()
    assert source["labelmap"] == ["car", "boat"]
    assert source["manifest"]["images"][0]["frontend_image_key"] == "train/original/example.jpg"


def test_class_analysis_chunked_active_workspace_preserves_frontend_keys(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", tmp_path)
    captured = {}

    def fake_enqueue(payload, *, job_id=None):
        captured["payload"] = payload
        return {"job_id": job_id}

    monkeypatch.setattr(api, "_enqueue_class_analysis_job", fake_enqueue)
    start = api.start_class_analysis_active_workspace_upload_session(
        {
            "dataset_label": "browser snapshot",
            "labelmap": ["car"],
            "request": {"analysis_scope": "all_classes"},
        }
    )
    session_id = start["session_id"]
    upload = UploadFile(filename="present.jpg", file=BytesIO(b"image-bytes"))
    batch_manifest = {
        "images": [
            {
                "upload_name": "present.jpg",
                "image_name": "Display Name.jpg",
                "frontend_image_key": "train/source/present.jpg",
                "label_lines": ["0 0.5 0.5 0.2 0.2"],
            }
        ]
    }

    try:
        batch = asyncio.run(
            api.batch_class_analysis_active_workspace_upload_session(
                session_id,
                json.dumps(batch_manifest),
                [upload],
            )
        )
        result = api.finalize_class_analysis_active_workspace_upload_session(session_id)
    finally:
        api.cancel_class_analysis_active_workspace_upload_session(session_id)

    assert batch["image_count"] == 1
    assert result["job_id"] == session_id
    assert result["snapshot_id"].startswith("cas_")
    manifest_path = Path(captured["payload"]["workspace_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_mode"] == "active_workspace"
    assert manifest["images"][0]["frontend_image_key"] == "train/source/present.jpg"
    assert manifest["images"][0]["image_relpath"] == "present.jpg"
    assert "dataset_id" not in captured["payload"]


def test_class_analysis_one_shot_reuse_discards_redundant_uploaded_workspace(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", tmp_path)
    monkeypatch.setattr(
        api,
        "_class_analysis_cache_digest",
        lambda _payload: "1" * 64,
    )
    snapshot_id = f"cas_{'1' * 20}"
    canonical_workspace = tmp_path / "ca_existing" / "active_workspace"
    canonical_workspace.mkdir(parents=True)
    previous_snapshot = {
        "snapshot_id": snapshot_id,
        "workspace_dir": str(canonical_workspace),
        "images": [],
        "labelmap": ["car"],
        "dataset_label": "canonical browser snapshot",
        "created_at": 1.0,
    }
    with api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS_LOCK:
        api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS[snapshot_id] = previous_snapshot
    captured = {}

    def fake_enqueue(payload, *, job_id=None):
        captured["payload"] = dict(payload)
        captured["requested_job_id"] = str(job_id or "")
        return {
            "job_id": "ca_existing",
            "run_fingerprint": "f" * 64,
            "reused": True,
        }

    monkeypatch.setattr(api, "_enqueue_class_analysis_job", fake_enqueue)
    manifest = {
        "dataset_label": "duplicate browser snapshot",
        "labelmap": ["car"],
        "images": [
            {
                "upload_name": "present.jpg",
                "label_lines": ["0 0.5 0.5 0.2 0.2"],
            }
        ],
    }
    upload = UploadFile(filename="present.jpg", file=BytesIO(b"image-bytes"))

    try:
        result = asyncio.run(
            api.create_class_analysis_active_workspace_job(
                json.dumps(manifest),
                [upload],
            )
        )
        redundant_root = Path(captured["payload"]["workspace_dir"]).parent
        with api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS_LOCK:
            restored_snapshot = dict(
                api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS[snapshot_id]
            )
    finally:
        with api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS_LOCK:
            api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS.pop(snapshot_id, None)

    assert result["job_id"] == "ca_existing"
    assert result["reused"] is True
    assert captured["requested_job_id"] != "ca_existing"
    assert not redundant_root.exists()
    assert canonical_workspace.is_dir()
    assert restored_snapshot == previous_snapshot


def test_class_analysis_chunked_reuse_discards_upload_and_rebinds_snapshot(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", tmp_path)
    monkeypatch.setattr(
        api,
        "_class_analysis_cache_digest",
        lambda _payload: "2" * 64,
    )
    snapshot_id = f"cas_{'2' * 20}"
    canonical_workspace = tmp_path / "ca_existing" / "active_workspace"
    canonical_workspace.mkdir(parents=True)
    existing_job = api.ClassAnalysisJob(
        job_id="ca_existing",
        status="completed",
        request={"workspace_dir": str(canonical_workspace)},
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS[existing_job.job_id] = existing_job

    def fake_enqueue(_payload, *, job_id=None):
        return {
            "job_id": "ca_existing",
            "run_fingerprint": "f" * 64,
            "reused": True,
        }

    monkeypatch.setattr(api, "_enqueue_class_analysis_job", fake_enqueue)
    start = api.start_class_analysis_active_workspace_upload_session(
        {
            "dataset_label": "duplicate chunked snapshot",
            "labelmap": ["car"],
            "request": {"analysis_scope": "all_classes"},
        }
    )
    session_id = start["session_id"]
    session_root = api.CLASS_ANALYSIS_ACTIVE_UPLOAD_SESSIONS[
        session_id
    ].root_dir
    upload = UploadFile(filename="present.jpg", file=BytesIO(b"image-bytes"))
    batch_manifest = {
        "images": [
            {
                "upload_name": "present.jpg",
                "label_lines": ["0 0.5 0.5 0.2 0.2"],
            }
        ]
    }

    try:
        asyncio.run(
            api.batch_class_analysis_active_workspace_upload_session(
                session_id,
                json.dumps(batch_manifest),
                [upload],
            )
        )
        result = api.finalize_class_analysis_active_workspace_upload_session(
            session_id
        )
        with api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS_LOCK:
            rebound_snapshot = dict(
                api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS[snapshot_id]
            )
    finally:
        api.cancel_class_analysis_active_workspace_upload_session(session_id)
        with api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS_LOCK:
            api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS.pop(snapshot_id, None)
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.pop(existing_job.job_id, None)

    assert result["job_id"] == existing_job.job_id
    assert result["reused"] is True
    assert not session_root.exists()
    assert session_id not in api.CLASS_ANALYSIS_ACTIVE_UPLOAD_SESSIONS
    assert canonical_workspace.is_dir()
    assert Path(rebound_snapshot["workspace_dir"]) == canonical_workspace


def test_class_analysis_reuse_replaces_stale_snapshot_registry_pointer(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", tmp_path)
    monkeypatch.setattr(
        api,
        "_class_analysis_cache_digest",
        lambda _payload: "3" * 64,
    )
    snapshot_id = f"cas_{'3' * 20}"
    canonical_workspace = tmp_path / "ca_existing" / "active_workspace"
    canonical_workspace.mkdir(parents=True)
    existing_job = api.ClassAnalysisJob(
        job_id="ca_existing",
        status="completed",
        request={"workspace_dir": str(canonical_workspace)},
    )
    image_sha256 = api.hashlib.sha256(b"image-bytes").hexdigest()
    stale_snapshot = {
        "snapshot_id": snapshot_id,
        "workspace_dir": str(tmp_path / "deleted_workspace"),
        "images": [
            {
                "image_relpath": "present.jpg",
                "image_sha256": image_sha256,
            }
        ],
        "labelmap": ["car"],
        "dataset_label": "stale browser snapshot",
        "created_at": 1.0,
    }
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS[existing_job.job_id] = existing_job
    with api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS_LOCK:
        api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS[snapshot_id] = stale_snapshot

    def fake_enqueue(_payload, *, job_id=None):
        return {
            "job_id": existing_job.job_id,
            "run_fingerprint": "f" * 64,
            "reused": True,
        }

    monkeypatch.setattr(api, "_enqueue_class_analysis_job", fake_enqueue)
    manifest = {
        "dataset_label": "duplicate browser snapshot",
        "labelmap": ["car"],
        "images": [
            {
                "upload_name": "present.jpg",
                "label_lines": ["0 0.5 0.5 0.2 0.2"],
            }
        ],
    }
    upload = UploadFile(
        filename="present.jpg",
        file=BytesIO(b"image-bytes"),
    )

    try:
        result = asyncio.run(
            api.create_class_analysis_active_workspace_job(
                json.dumps(manifest),
                [upload],
            )
        )
        with api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS_LOCK:
            rebound_snapshot = dict(
                api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS[snapshot_id]
            )
    finally:
        with api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS_LOCK:
            api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS.pop(snapshot_id, None)
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.pop(existing_job.job_id, None)

    assert result["job_id"] == existing_job.job_id
    assert result["reused"] is True
    assert Path(rebound_snapshot["workspace_dir"]) == canonical_workspace
    assert rebound_snapshot["images"] == [
        {
            "image_relpath": "present.jpg",
            "image_sha256": image_sha256,
        }
    ]


def test_class_analysis_snapshot_reuse_discards_temporary_job_and_manifest(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", tmp_path)
    snapshot_id = "cas_snapshot_reuse"
    canonical_workspace = tmp_path / "ca_existing" / "active_workspace"
    image_path = canonical_workspace / "images" / "present.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image-bytes")
    snapshot = {
        "snapshot_id": snapshot_id,
        "workspace_dir": str(canonical_workspace),
        "images": [
            {
                "image_relpath": "present.jpg",
                "image_sha256": api._class_analysis_file_sha256(image_path),
            }
        ],
        "labelmap": ["car"],
        "dataset_label": "canonical browser snapshot",
        "created_at": 1.0,
    }
    with api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS_LOCK:
        api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS[snapshot_id] = snapshot
    captured = {}

    def fake_enqueue(payload, *, job_id=None):
        captured["payload"] = dict(payload)
        captured["requested_job_id"] = str(job_id or "")
        return {
            "job_id": "ca_existing",
            "run_fingerprint": "f" * 64,
            "reused": True,
        }

    monkeypatch.setattr(api, "_enqueue_class_analysis_job", fake_enqueue)
    payload = {
        "labelmap": ["car"],
        "images": [
            {
                "upload_name": "present.jpg",
                "label_lines": ["0 0.5 0.5 0.2 0.2"],
            }
        ],
        "request": {"analysis_scope": "all_classes"},
    }

    try:
        result = api.create_class_analysis_active_workspace_snapshot_job(
            snapshot_id,
            payload,
        )
        temporary_job_dir = tmp_path / captured["requested_job_id"]
        generated_manifest = Path(
            captured["payload"]["workspace_manifest_path"]
        )
    finally:
        with api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS_LOCK:
            api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS.pop(snapshot_id, None)

    assert result["job_id"] == "ca_existing"
    assert result["reused"] is True
    assert not temporary_job_dir.exists()
    assert not generated_manifest.exists()
    assert image_path.read_bytes() == b"image-bytes"


def test_class_analysis_stale_snapshot_requests_fresh_upload_without_orphan(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", tmp_path)
    tmp_path.mkdir(exist_ok=True)
    snapshot_id = "cas_deleted_snapshot"
    snapshot = {
        "snapshot_id": snapshot_id,
        "workspace_dir": str(tmp_path / "deleted_workspace"),
        "images": [
            {
                "image_relpath": "present.jpg",
                "image_sha256": "a" * 64,
            }
        ],
        "labelmap": ["car"],
        "dataset_label": "deleted browser snapshot",
        "created_at": 1.0,
    }
    with api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS_LOCK:
        api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS[snapshot_id] = snapshot
    payload = {
        "labelmap": ["car"],
        "images": [
            {
                "upload_name": "present.jpg",
                "label_lines": ["0 0.5 0.5 0.2 0.2"],
            }
        ],
        "request": {"analysis_scope": "all_classes"},
    }

    try:
        with pytest.raises(api.HTTPException) as exc_info:
            api.create_class_analysis_active_workspace_snapshot_job(
                snapshot_id,
                payload,
            )
        with api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS_LOCK:
            snapshot_present = (
                snapshot_id in api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS
            )
    finally:
        with api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS_LOCK:
            api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS.pop(snapshot_id, None)

    assert exc_info.value.status_code == api.HTTP_409_CONFLICT
    assert exc_info.value.detail == "active_workspace_snapshot_stale"
    assert snapshot_present is False
    assert not any(path.name.startswith("ca_") for path in tmp_path.iterdir())


def test_class_analysis_chunked_active_workspace_rejects_overlapping_batch(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", tmp_path)
    start = api.start_class_analysis_active_workspace_upload_session(
        {
            "dataset_label": "browser snapshot",
            "labelmap": ["car"],
            "request": {"analysis_scope": "all_classes"},
        }
    )
    session_id = start["session_id"]
    session = api.CLASS_ANALYSIS_ACTIVE_UPLOAD_SESSIONS[session_id]
    assert session.lock.acquire(blocking=False)

    try:
        with pytest.raises(api.HTTPException) as exc_info:
            asyncio.run(
                api.batch_class_analysis_active_workspace_upload_session(
                    session_id,
                    json.dumps({"images": []}),
                    [],
                )
            )
    finally:
        session.lock.release()
        api.cancel_class_analysis_active_workspace_upload_session(session_id)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "active_workspace_upload_session_busy"


def test_class_analysis_chunked_upload_can_retry_after_partial_batch_failure(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", tmp_path)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ACTIVE_UPLOAD_MAX_BYTES", 4)
    start = api.start_class_analysis_active_workspace_upload_session(
        {
            "dataset_label": "retryable browser snapshot",
            "labelmap": ["car"],
            "request": {"analysis_scope": "all_classes"},
        }
    )
    session_id = start["session_id"]
    manifest = json.dumps(
        {
            "images": [
                {"upload_name": "first.jpg", "label_lines": ["0 0.5 0.5 0.2 0.2"]},
                {"upload_name": "second.jpg", "label_lines": ["0 0.5 0.5 0.2 0.2"]},
            ]
        }
    )

    try:
        with pytest.raises(api.HTTPException) as exc_info:
            asyncio.run(
                api.batch_class_analysis_active_workspace_upload_session(
                    session_id,
                    manifest,
                    [
                        UploadFile(filename="first.jpg", file=BytesIO(b"ok")),
                        UploadFile(filename="second.jpg", file=BytesIO(b"too-large")),
                    ],
                )
            )
        session = api.CLASS_ANALYSIS_ACTIVE_UPLOAD_SESSIONS[session_id]
        assert session.saved_uploads == {}
        assert session.rows == []
        assert session.bytes_written == 0

        retried = asyncio.run(
            api.batch_class_analysis_active_workspace_upload_session(
                session_id,
                manifest,
                [
                    UploadFile(filename="first.jpg", file=BytesIO(b"ok")),
                    UploadFile(filename="second.jpg", file=BytesIO(b"yes")),
                ],
            )
        )
    finally:
        api.cancel_class_analysis_active_workspace_upload_session(session_id)

    assert exc_info.value.detail == "active_workspace_upload_too_large"
    assert retried["image_count"] == 2
    assert retried["bytes_written"] == 5


def test_class_analysis_source_rejects_active_workspace_outside_class_root(
    tmp_path, monkeypatch
):
    class_root = tmp_path / "class_analysis"
    class_root.mkdir()
    workspace = tmp_path / "outside_workspace"
    (workspace / "images").mkdir(parents=True)
    manifest_path = workspace / "manifest.json"
    manifest_path.write_text(
        json.dumps({"labelmap": ["car"], "images": [], "yolo_layout": "flat"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)

    with pytest.raises(api.HTTPException) as exc_info:
        api._class_analysis_source(
            {
                "source_mode": "active_workspace",
                "workspace_id": "ca_outside",
                "workspace_dir": str(workspace),
                "workspace_manifest_path": str(manifest_path),
            }
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "active_workspace_path_invalid"


def test_class_analysis_source_rejects_active_workspace_manifest_escape(
    tmp_path, monkeypatch
):
    class_root = tmp_path / "class_analysis"
    workspace = class_root / "workspace"
    workspace.mkdir(parents=True)
    outside_manifest = class_root / "outside_manifest.json"
    outside_manifest.write_text(
        json.dumps({"labelmap": ["car"], "images": [], "yolo_layout": "flat"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)

    with pytest.raises(api.HTTPException) as exc_info:
        api._class_analysis_source(
            {
                "source_mode": "active_workspace",
                "workspace_id": "ca_manifest_escape",
                "workspace_dir": str(workspace),
                "workspace_manifest_path": str(outside_manifest),
            }
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "active_workspace_path_invalid"


def test_class_analysis_active_workspace_rejects_symlinked_root_before_upload(
    tmp_path, monkeypatch
):
    outside = tmp_path / "outside_class_analysis"
    outside.mkdir()
    class_root = tmp_path / "class_analysis"
    try:
        class_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    manifest = {
        "labelmap": ["car"],
        "images": [
            {
                "upload_name": "present.jpg",
                "label_lines": ["0 0.5 0.5 0.2 0.2"],
            }
        ],
    }
    upload = UploadFile(filename="present.jpg", file=BytesIO(b"image-bytes"))

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(api.create_class_analysis_active_workspace_job(json.dumps(manifest), [upload]))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "class_analysis_path_invalid"
    assert list(outside.iterdir()) == []


def test_class_analysis_active_workspace_rejects_symlinked_root_parent_before_upload(
    tmp_path, monkeypatch
):
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", linked_parent / "class_analysis")
    manifest = {
        "labelmap": ["car"],
        "images": [
            {
                "upload_name": "present.jpg",
                "label_lines": ["0 0.5 0.5 0.2 0.2"],
            }
        ],
    }
    upload = UploadFile(filename="present.jpg", file=BytesIO(b"image-bytes"))

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(api.create_class_analysis_active_workspace_job(json.dumps(manifest), [upload]))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "class_analysis_path_invalid"
    assert list(outside.iterdir()) == []


def test_class_analysis_active_workspace_cleans_partial_upload_on_bad_manifest(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", tmp_path)
    manifest = {
        "labelmap": ["car"],
        "images": [
            {
                "upload_name": "missing.jpg",
                "label_lines": ["0 0.5 0.5 0.2 0.2"],
            }
        ],
    }
    upload = UploadFile(filename="present.jpg", file=BytesIO(b"image-bytes"))

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(api.create_class_analysis_active_workspace_job(json.dumps(manifest), [upload]))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "active_workspace_image_upload_missing"
    assert list(tmp_path.iterdir()) == []
    assert upload.file.closed


def test_class_analysis_active_workspace_rejects_oversize_upload(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", tmp_path)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ACTIVE_UPLOAD_MAX_BYTES", 4)
    manifest = {
        "labelmap": ["car"],
        "images": [
            {
                "upload_name": "present.jpg",
                "label_lines": ["0 0.5 0.5 0.2 0.2"],
            }
        ],
    }
    upload = UploadFile(filename="present.jpg", file=BytesIO(b"too-large"))

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(api.create_class_analysis_active_workspace_job(json.dumps(manifest), [upload]))

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == "active_workspace_upload_too_large"
    assert list(tmp_path.iterdir()) == []
    assert upload.file.closed


def test_class_analysis_active_workspace_upload_is_atomic_over_symlink_leaves(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    job_root = class_root / "ca_fixed"
    images_dir = job_root / "active_workspace" / "images"
    images_dir.mkdir(parents=True)
    outside_tmp = tmp_path / "outside_tmp.jpg"
    outside_final = tmp_path / "outside_final.jpg"
    outside_tmp.write_bytes(b"external tmp")
    outside_final.write_bytes(b"external final")
    target = images_dir / "present.jpg"
    tmp_leaf = images_dir / "present.jpg.fixed.tmp"
    try:
        target.symlink_to(outside_final)
        tmp_leaf.symlink_to(outside_tmp)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    image_buffer = BytesIO()
    Image.new("RGB", (8, 8), (12, 24, 36)).save(image_buffer, format="JPEG")
    image_bytes = image_buffer.getvalue()
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    monkeypatch.setattr(api.uuid, "uuid4", lambda: types.SimpleNamespace(hex="fixed"))
    captured = {}

    def fake_enqueue(payload, *, job_id=None):
        captured["payload"] = payload
        return {"job_id": job_id}

    monkeypatch.setattr(api, "_enqueue_class_analysis_job", fake_enqueue)
    manifest = {
        "labelmap": ["car"],
        "images": [
            {
                "upload_name": "present.jpg",
                "label_lines": ["0 0.5 0.5 0.2 0.2"],
            }
        ],
    }
    upload = UploadFile(filename="present.jpg", file=BytesIO(image_bytes))

    result = asyncio.run(
        api.create_class_analysis_active_workspace_job(json.dumps(manifest), [upload])
    )

    assert result["job_id"] == "ca_fixed"
    assert result["snapshot_id"].startswith("cas_")
    assert captured["payload"]["workspace_id"] == result["snapshot_id"]
    assert outside_tmp.read_bytes() == b"external tmp"
    assert outside_final.read_bytes() == b"external final"
    assert not target.is_symlink()
    assert target.read_bytes() == image_bytes
    assert not tmp_leaf.exists()
    assert upload.file.closed


def test_class_analysis_prepare_write_path_rejects_symlinked_parent_without_write(tmp_path):
    root = tmp_path / "class_analysis"
    root.mkdir()
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = root / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    assert api._class_analysis_prepare_write_path(linked_parent / "result.json", root) is None
    assert list(outside.iterdir()) == []


def test_class_analysis_json_write_is_atomic_over_symlink_leaves(
    tmp_path, monkeypatch
):
    root = tmp_path / "class_analysis"
    root.mkdir()
    outside_tmp = tmp_path / "outside_tmp.json"
    outside_final = tmp_path / "outside_final.json"
    outside_tmp.write_text('{"tmp":true}', encoding="utf-8")
    outside_final.write_text('{"final":true}', encoding="utf-8")
    target = root / "result.json"
    tmp_leaf = root / "result.json.fixed.tmp"
    try:
        target.symlink_to(outside_final)
        tmp_leaf.symlink_to(outside_tmp)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api.uuid, "uuid4", lambda: types.SimpleNamespace(hex="fixed"))

    api._class_analysis_write_json(target, root, {"status": "ok"})

    assert outside_tmp.read_text(encoding="utf-8") == '{"tmp":true}'
    assert outside_final.read_text(encoding="utf-8") == '{"final":true}'
    assert not target.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "ok"}
    assert not tmp_leaf.exists()


def test_class_analysis_json_write_streams_without_monolithic_dumps(
    tmp_path, monkeypatch
):
    root = tmp_path / "class_analysis"
    root.mkdir()
    target = root / "result.json"

    def forbidden_dumps(*_args, **_kwargs):
        raise AssertionError("class analysis JSON must use iterencode")

    monkeypatch.setattr(api.json, "dumps", forbidden_dumps)

    api._class_analysis_write_json(
        target,
        root,
        {
            "label": "Bâtiment",
            "coordinates": np.asarray([[1.0, 2.0]], dtype=np.float32),
            "count": np.int64(3),
        },
    )

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "label": "Bâtiment",
        "coordinates": [[1.0, 2.0]],
        "count": 3,
    }


def test_class_analysis_binary_copy_is_atomic_over_symlink_leaves(
    tmp_path, monkeypatch
):
    source_root = tmp_path / "source"
    dest_root = tmp_path / "dest"
    source_root.mkdir()
    dest_root.mkdir()
    src = source_root / "source.bin"
    src.write_bytes(b"new payload")
    outside_tmp = tmp_path / "outside_tmp.bin"
    outside_final = tmp_path / "outside_final.bin"
    outside_tmp.write_bytes(b"external tmp")
    outside_final.write_bytes(b"external final")
    dest = dest_root / "copy.bin"
    tmp_leaf = dest_root / "copy.bin.fixed.tmp"
    try:
        dest.symlink_to(outside_final)
        tmp_leaf.symlink_to(outside_tmp)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api.uuid, "uuid4", lambda: types.SimpleNamespace(hex="fixed"))

    assert api._class_analysis_copy_file_within_roots(
        src,
        dest,
        source_root=source_root,
        dest_root=dest_root,
    )

    assert outside_tmp.read_bytes() == b"external tmp"
    assert outside_final.read_bytes() == b"external final"
    assert not dest.is_symlink()
    assert dest.read_bytes() == b"new payload"
    assert not tmp_leaf.exists()


def test_class_analysis_npz_write_is_atomic_over_symlink_leaves(
    tmp_path, monkeypatch
):
    root = tmp_path / "class_analysis"
    root.mkdir()
    outside_tmp = tmp_path / "outside_tmp.npz"
    outside_final = tmp_path / "outside_final.npz"
    outside_tmp.write_bytes(b"external tmp")
    outside_final.write_bytes(b"external final")
    target = root / "embeddings.npz"
    tmp_leaf = root / "embeddings.npz.fixed.tmp"
    try:
        target.symlink_to(outside_final)
        tmp_leaf.symlink_to(outside_tmp)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api.uuid, "uuid4", lambda: types.SimpleNamespace(hex="fixed"))

    api._class_analysis_write_npz(
        target,
        root,
        embeddings=np.asarray([[1.0, 2.0]], dtype=np.float32),
    )

    assert outside_tmp.read_bytes() == b"external tmp"
    assert outside_final.read_bytes() == b"external final"
    assert not target.is_symlink()
    with np.load(target) as loaded:
        assert np.allclose(loaded["embeddings"], [[1.0, 2.0]])
    assert not tmp_leaf.exists()


def test_class_analysis_projection_endpoint_returns_json_lists(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_dir = api._class_analysis_job_dir("job_projection", create=True)
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(
        result_path,
        class_root,
        {
            "summary": {"projection": "pca", "projection_mode": "global_pca"},
            "points": [
                {"point_id": "a", "projection": [1.0, 2.0]},
                {"point_id": "b", "projection": [3.0, 4.0]},
            ],
            "projection_options": {"selected": "global_pca", "coordinates_available": ["global_pca"]},
        },
    )
    api._class_analysis_write_npz(
        job_dir / api.CLASS_ANALYSIS_PROJECTION_COORDS_FILENAME,
        class_root,
        global_pca=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    job = api.ClassAnalysisJob(
        job_id="job_projection",
        status="completed",
        result_path=str(result_path),
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()
        api.CLASS_ANALYSIS_JOBS[job.job_id] = job

    try:
        payload = api.get_class_analysis_projection(job.job_id, "global_pca")
        assert payload["mode"] == "global_pca"
        assert payload["coordinates"] == [[1.0, 2.0], [3.0, 4.0]]
        assert isinstance(payload["coordinates"], list)
        assert isinstance(payload["coordinates"][0], list)
        for bad_mode in ("not_a_projection", "umap"):
            with pytest.raises(api.HTTPException) as exc_info:
                api.get_class_analysis_projection(job.job_id, bad_mode)
            assert exc_info.value.status_code == api.HTTP_404_NOT_FOUND
            assert exc_info.value.detail == "projection_not_found"
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


def test_class_analysis_get_job_restores_explicit_legacy_stage1_disk_result(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_dir = api._class_analysis_job_dir("job_restored", create=True)
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(
        result_path,
        class_root,
        {
            "summary": {"analysis_scope": "all_classes"},
            "points": [],
        },
    )
    api._class_analysis_write_json(
        job_dir / "config.json",
        class_root,
        {"analysis_scope": "all_classes", "encoder_type": "dinov3"},
    )
    thumb_dir = job_dir / "thumbnails"
    thumb_dir.mkdir()
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()

    try:
        job = api._get_class_analysis_job("job_restored")

        assert job.status == "completed"
        assert job.result_path == str(result_path)
        assert job.thumbnail_dir == str(thumb_dir)
        assert job.request["encoder_type"] == "dinov3"
        assert job.message == "Restored legacy Stage-1 class analysis job from disk."
        assert api.get_class_analysis_result(job.job_id)["summary"]["analysis_scope"] == "all_classes"
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            assert api.CLASS_ANALYSIS_JOBS[job.job_id] is job
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


def _class_analysis_test_selector_v6_binding():
    selector_model = api._class_analysis_load_default_selector_model_v6()
    return {
        "selector_feature_contract": (
            api.CLASS_ANALYSIS_SELECTOR_FEATURE_CONTRACT
        ),
        "selector_model_digest": str(
            selector_model.get("model_digest") or ""
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
    }


def _write_class_analysis_terminal_support(
    *,
    job_dir,
    class_root,
    result,
    digest,
):
    summary = (
        result.get("summary")
        if isinstance(result.get("summary"), dict)
        else {}
    )
    refinement = (
        summary.get("refinement")
        if isinstance(summary.get("refinement"), dict)
        else {}
    )
    config = {"run_fingerprint": digest}
    decision_contract = str(
        refinement.get("decision_contract") or ""
    ).strip()
    if decision_contract:
        config["refinement_decision_contract"] = decision_contract
    selector_priority_contract = str(
        refinement.get("selector_priority_contract") or ""
    ).strip()
    if selector_priority_contract:
        config["selector_priority_contract"] = selector_priority_contract
        for key in (
            "selector_feature_contract",
            "selector_model_digest",
            "selector_utility_policy_contract",
            "selector_dataset_overlap_application_contract",
            "selector_dataset_overlap_diagnostic_contract",
            "selector_global_actionability_model_contract",
        ):
            value = str(refinement.get(key) or "").strip()
            if value:
                config[key] = value
    capture_group_contract = str(
        refinement.get("capture_group_contract") or ""
    ).strip()
    if capture_group_contract:
        config["capture_group_contract"] = capture_group_contract
    api._class_analysis_write_json(
        job_dir / "config.json",
        class_root,
        config,
    )
    api._class_analysis_write_indexed_metadata_jsonl(
        job_dir / api.CLASS_ANALYSIS_METADATA_FILENAME,
        job_dir,
        list(result.get("points") or []),
    )


def _class_analysis_test_exact_view_pair_calibration():
    return {
        "contract": "class-analysis-exact-view-calibration-pass-v1",
        "eligible_example_count": 1,
        "accepted_example_count": 1,
        "skipped_example_count": 0,
        "per_class_accepted_source_counts": {"A": 1},
    }


def test_selector_priority_restore_accepts_singleton_content_bound_phash():
    source_sha256 = "1" * 64
    record = {
        "point_id": "point-phash-singleton",
        "review_object_key": "point-phash-singleton",
        "split": "train",
        "image_relpath": "00000000-0000-0000-0000-000000000001.jpg",
        "class_name": "Bike",
        "bbox_xyxy": [0, 0, 100, 100],
        "_image_sha256": source_sha256,
        "_image_width": 100,
        "_image_height": 100,
        "capture_perceptual_hash": "0" * 32,
        "capture_perceptual_image_sha256": source_sha256,
    }
    prior = api._class_analysis_build_frequent_overlap_prior(
        [record],
        trusted_screened_point_ids={record["point_id"]},
    )
    row = {
        "point_id": "point-phash-singleton",
        "split": "train",
        "image_relpath": record["image_relpath"],
        "class_name": "Bike",
        "suggested_neighbor_class": "Person",
        "wrong_class_suspicion": 0.9,
        "refined_outlier": {
            "schema": api.CLASS_ANALYSIS_REFINEMENT_SCHEMA,
            "decision_contract": (
                api.CLASS_ANALYSIS_REFINEMENT_DECISION_CONTRACT
            ),
            "status": api.CLASS_ANALYSIS_REFINEMENT_UNRESOLVED,
            "current_class": "Bike",
            "alternative_class": "Person",
            "qualified_for_human_review": False,
            "source_image_sha256": source_sha256,
            "overlap_object_count": 0,
            "annotated_overlap_alternative_bbox_xyxy": None,
            "decision_gates": {},
        },
    }
    selector = api._class_analysis_assign_selector_priority_ranks(
        [row],
        overlap_prior=prior,
        overlap_index={},
        records_by_id={row["point_id"]: record},
    )
    prior_evidence = row["refined_outlier"]["frequent_overlap_prior"]

    assert prior_evidence["candidate_capture_group_tier"] == (
        "provisional_unlineaged"
    )
    assert prior_evidence["candidate_capture_group_methods"] == [
        "content_bound_perceptual_hash"
    ]
    api._class_analysis_validate_selector_priority_artifact(
        refinement={
            "selector_priority": selector,
            "selector_priority_candidate_count": 1,
        },
        refinement_rows=[row],
        configured_contract=api.CLASS_ANALYSIS_SELECTOR_PRIORITY_CONTRACT,
    )


def test_v7_selector_binds_published_rank_effect_not_mutated_raw_overlap_provenance():
    records = []
    for index, explicit, signature in (
        (1, True, "0" * 32),
        (2, False, "0" * 31 + "1"),
    ):
        sha256 = f"{index:064x}"
        record = {
            "point_id": f"point-{index}",
            "review_object_key": f"point-{index}",
            "split": "train",
            "image_relpath": f"00000000-0000-0000-0000-{index:012x}.jpg",
            "class_name": "Bike",
            "bbox_xyxy": [0, 0, 100, 100],
            "_image_sha256": sha256,
            "_image_width": 100,
            "_image_height": 100,
            "capture_perceptual_hash": signature,
            "capture_perceptual_image_sha256": sha256,
        }
        if explicit:
            record["capture_group_id"] = "attested-capture"
        records.append(record)
    prior = api._class_analysis_build_frequent_overlap_prior(
        records,
        trusted_screened_point_ids={
            str(record.get("point_id") or "") for record in records
        },
    )
    rows = []
    overlap_index = {}
    for index, relation in ((1, "partial_contamination"), (2, "duplicate_like")):
        row = {
            "point_id": f"point-{index}",
            "split": "train",
            "image_relpath": records[index - 1]["image_relpath"],
            "class_name": "Bike",
            "suggested_neighbor_class": "Person",
            "wrong_class_suspicion": 0.9 - index * 0.01,
            "refined_outlier": {
                "schema": api.CLASS_ANALYSIS_REFINEMENT_SCHEMA,
                "decision_contract": (
                    api.CLASS_ANALYSIS_REFINEMENT_DECISION_CONTRACT
                ),
                "status": api.CLASS_ANALYSIS_REFINEMENT_UNRESOLVED,
                "current_class": "Bike",
                "alternative_class": "Person",
                "qualified_for_human_review": False,
                "source_image_sha256": records[index - 1]["_image_sha256"],
                "overlap_object_count": 1,
                "annotated_overlap_alternative_bbox_xyxy": [
                    10,
                    0,
                    90,
                    100,
                ],
                "annotated_overlap_alternative_point_id": f"person-{index}",
                "decision_gates": {},
            },
        }
        rows.append(row)
        overlap_index[row["point_id"]] = [
            {
                "point_id": f"person-{index}",
                "class_name": "Person",
                "iou": 1.0 if relation == "duplicate_like" else 0.8,
                "target_area_covered": 1.0 if relation == "duplicate_like" else 0.8,
                "relation": relation,
            }
        ]
    selector = api._class_analysis_assign_selector_priority_ranks(
        rows,
        overlap_prior=prior,
        overlap_index=overlap_index,
        records_by_id={record["point_id"]: record for record in records},
    )
    priors = [row["refined_outlier"]["frequent_overlap_prior"] for row in rows]

    assert {item["candidate_capture_group_tier"] for item in priors} == {
        "strong",
        "lower_confidence",
    }
    assert {
        item["candidate_capture_group_dependency_tier"] for item in priors
    } == {"lower_confidence"}
    assert {item["geometry_stratum"] for item in priors} == {
        "material_nonduplicate",
        "duplicate_like",
    }
    assert len({item["fit_source_digest"] for item in priors}) == 1
    refinement = {
        "selector_priority": selector,
        "selector_priority_candidate_count": 2,
    }
    api._class_analysis_validate_selector_priority_artifact(
        refinement=refinement,
        refinement_rows=rows,
        configured_contract=api.CLASS_ANALYSIS_SELECTOR_PRIORITY_CONTRACT,
    )

    corrupted = copy.deepcopy(rows)
    corrupted_prior = corrupted[0]["refined_outlier"][
        "frequent_overlap_prior"
    ]
    corrupted_prior["strata"][0]["cohort_diagnostics"][0][
        "group_rate_sum"
    ] = 0.5
    # V7 score validity binds the fully materialized selector payload. Raw
    # cohort diagnostics are retained for inspection, but mutating them after
    # ranking cannot retroactively change the published score.
    api._class_analysis_validate_selector_priority_artifact(
        refinement=refinement,
        refinement_rows=corrupted,
        configured_contract=api.CLASS_ANALYSIS_SELECTOR_PRIORITY_CONTRACT,
    )
    for row in corrupted:
        payload = row["refined_outlier"]["selector_v6"]
        assert payload["dataset_overlap_scoring_effect_enabled"] is True
        assert payload["dataset_overlap"]["rank_only"] is True
        assert payload["dataset_overlap"]["uses_human_review_labels"] is False
        assert payload["dataset_overlap"]["probability_delta"] == 0.0
        assert payload["dataset_overlap"]["applied"] is False
        assert payload["dataset_overlap"]["utility_delta"] == 0.0


def _write_bound_refinement_terminal_fixture(
    *,
    job_dir,
    class_root,
    job_id,
    digest,
):
    point = {
        "point_id": "point-0",
        "split": "train",
        "image_relpath": "frame.jpg",
        "class_name": "A",
        "bbox_xyxy": [0, 0, 10, 10],
    }
    sidecar_path = (
        job_dir / api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME
    )
    sidecar_bytes = b"bounded-sidecar"
    api._class_analysis_write_binary(
        sidecar_path,
        job_dir,
        lambda handle: handle.write(sidecar_bytes),
    )
    sidecar_sha256 = hashlib.sha256(sidecar_bytes).hexdigest()
    manifest_path = (
        job_dir / api.CLASS_ANALYSIS_REFINEMENT_MANIFEST_FILENAME
    )
    exact_view_pair_calibration = (
        _class_analysis_test_exact_view_pair_calibration()
    )
    selector_v6_binding = _class_analysis_test_selector_v6_binding()
    selector_priority = api._class_analysis_assign_selector_priority_ranks([])
    manifest = {
        "schema": api.CLASS_ANALYSIS_REFINEMENT_SCHEMA,
        "decision_contract": (
            api.CLASS_ANALYSIS_REFINEMENT_DECISION_CONTRACT
        ),
        "selector_priority_contract": (
            api.CLASS_ANALYSIS_SELECTOR_PRIORITY_CONTRACT
        ),
        **selector_v6_binding,
        "capture_group_contract": api.CLASS_ANALYSIS_CAPTURE_GROUP_CONTRACT,
        "exact_view_pair_calibration": exact_view_pair_calibration,
        "sidecar_file": api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME,
        "grid_shape": [3, 3],
        "sidecar": {
            "file": api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME,
            "sha256": sidecar_sha256,
            "bytes": len(sidecar_bytes),
            "arrays": {
                "point_ids": {"shape": [1], "dtype": "<U8"},
                "current_heatmaps": {
                    "shape": [1, 2, 3, 3],
                    "dtype": "float16",
                },
                "alternative_heatmaps": {
                    "shape": [1, 2, 3, 3],
                    "dtype": "float16",
                },
                "valid_masks": {
                    "shape": [1, 2, 3, 3],
                    "dtype": "uint8",
                },
                "target_masks": {
                    "shape": [1, 2, 3, 3],
                    "dtype": "uint8",
                },
                "overlap_masks": {
                    "shape": [1, 2, 3, 3],
                    "dtype": "uint8",
                },
            },
        },
        "point_rows": {point["point_id"]: 0},
    }
    api._class_analysis_write_json(
        manifest_path,
        class_root,
        manifest,
    )
    refinement = {
        "status": "completed",
        "schema": api.CLASS_ANALYSIS_REFINEMENT_SCHEMA,
        "decision_contract": (
            api.CLASS_ANALYSIS_REFINEMENT_DECISION_CONTRACT
        ),
        "selector_priority_contract": (
            api.CLASS_ANALYSIS_SELECTOR_PRIORITY_CONTRACT
        ),
        **selector_v6_binding,
        "capture_group_contract": api.CLASS_ANALYSIS_CAPTURE_GROUP_CONTRACT,
        "selector_priority": selector_priority,
        "selector_priority_candidate_count": 0,
        "rough_candidate_count": 0,
        "exact_view_pair_calibration": exact_view_pair_calibration,
        "sidecar_file": api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME,
        "sidecar_sha256": sidecar_sha256,
    }
    result = {
        "summary": {
            "analysis_scope": "all_classes",
            "analysis_job_id": job_id,
            "analysis_run_instance_id": job_id,
            "analysis_input_digest": digest,
            "run_fingerprint": digest,
            "refinement": refinement,
        },
        "refinement_summary": refinement,
        "refinement_candidates": [],
        "wrong_class_candidates": [],
        "vignette_candidates": [],
        "points": [point],
    }
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(result_path, class_root, result)
    _write_class_analysis_terminal_support(
        job_dir=job_dir,
        class_root=class_root,
        result=result,
        digest=digest,
    )
    binding = api._class_analysis_job_state_artifact_binding(
        result_path=result_path,
        result=result,
    )
    api._class_analysis_write_json(
        job_dir / api.CLASS_ANALYSIS_JOB_STATE_FILENAME,
        class_root,
        {
            "schema": "class-analysis-job-state-v1",
            "status": "completed",
            "progress": 1.0,
            **binding,
        },
    )
    return {
        "point": point,
        "result": result,
        "result_path": result_path,
        "sidecar_path": sidecar_path,
        "sidecar_bytes": sidecar_bytes,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }


@pytest.mark.parametrize(
    ("ranked_point_ids", "expected_reason"),
    [
        (["rough-0", "rough-1"], "summary_counts_or_invariants"),
        (
            ["rough-0", "rough-1", "not-in-stage1"],
            "rough_candidate_identity",
        ),
    ],
)
def test_class_analysis_terminal_binding_rejects_incomplete_v6_queue(
    tmp_path,
    monkeypatch,
    ranked_point_ids,
    expected_reason,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_id = f"job_v6_incomplete_{len(ranked_point_ids)}"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    digest = "7" * 64
    fixture = _write_bound_refinement_terminal_fixture(
        job_dir=job_dir,
        class_root=class_root,
        job_id=job_id,
        digest=digest,
    )
    result = fixture["result"]
    rough_rows = [
        {
            "point_id": f"rough-{index}",
            "class_name": "Bike",
            "suggested_neighbor_class": "Person",
            "wrong_class_suspicion": 0.9 - index * 0.1,
        }
        for index in range(3)
    ]
    ranked_rows = [
        {
            "point_id": point_id,
            "class_name": "Bike",
            "suggested_neighbor_class": "Person",
            "wrong_class_suspicion": 0.9 - index * 0.1,
            "refined_outlier": {
                "current_class": "Bike",
                "alternative_class": "Person",
                "decision_gates": {
                    "source_resolution_sufficient": True,
                },
            },
        }
        for index, point_id in enumerate(ranked_point_ids)
    ]
    selector_priority = api._class_analysis_assign_selector_priority_ranks(
        ranked_rows
    )
    refinement = dict(result["refinement_summary"])
    refinement.update(
        {
            "selector_priority": selector_priority,
            "selector_priority_candidate_count": len(ranked_rows),
            "rough_candidate_count": len(rough_rows),
        }
    )
    result["summary"] = {
        **dict(result["summary"]),
        "refinement": refinement,
    }
    result["refinement_summary"] = refinement
    result["wrong_class_candidates"] = rough_rows
    result["refinement_candidates"] = ranked_rows
    api._class_analysis_write_json(
        fixture["result_path"],
        class_root,
        result,
    )

    with pytest.raises(ValueError, match=expected_reason):
        api._class_analysis_job_state_artifact_binding(
            result_path=fixture["result_path"],
            result=result,
        )


def _class_analysis_projection_test_result(job_id, digest):
    refinement = {
        "status": "disabled",
        "sidecar_file": "",
        "sidecar_sha256": "",
    }
    return {
        "summary": {
            "analysis_job_id": job_id,
            "analysis_run_instance_id": job_id,
            "analysis_input_digest": digest,
            "run_fingerprint": digest,
            "refinement": refinement,
        },
        "projection_options": {
            "selected": "global_pca",
            "coordinates_available": ["global_pca"],
        },
        "refinement_summary": refinement,
        "refinement_candidates": [],
        "vignette_candidates": [],
        "points": [
            {"point_id": "point-1", "projection": [1.0, 2.0]},
            {"point_id": "point-2", "projection": [3.0, 4.0]},
        ],
    }


def test_class_analysis_get_job_restores_valid_bound_terminal_result(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_id = "job_bound_restore"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    digest = "a" * 64
    result = {
        "summary": {
            "analysis_scope": "all_classes",
            "analysis_job_id": job_id,
            "analysis_run_instance_id": job_id,
            "analysis_input_digest": digest,
            "run_fingerprint": digest,
            "refinement": {"status": "disabled", "sidecar_file": ""},
        },
        "refinement_summary": {"status": "disabled", "sidecar_file": ""},
        "refinement_candidates": [],
        "vignette_candidates": [],
        "points": [],
    }
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(result_path, class_root, result)
    _write_class_analysis_terminal_support(
        job_dir=job_dir,
        class_root=class_root,
        result=result,
        digest=digest,
    )
    api._class_analysis_write_json(
        job_dir / api.CLASS_ANALYSIS_JOB_STATE_FILENAME,
        class_root,
        {
            "schema": "class-analysis-job-state-v1",
            "status": "completed",
            "progress": 1.0,
            "message": "Class analysis completed.",
            **api._class_analysis_job_state_artifact_binding(
                result_path=result_path,
                result=result,
            ),
        },
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()

    try:
        job = api._get_class_analysis_job(job_id)

        assert job.status == "completed"
        assert job.result_path == str(result_path)
        assert job.error is None
        assert api.get_class_analysis_result(job_id)["summary"][
            "analysis_input_digest"
        ] == digest
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


@pytest.mark.parametrize(
    "artifact_name",
    ["result.json", "config.json", api.CLASS_ANALYSIS_JOB_STATE_FILENAME],
)
def test_class_analysis_restart_rejects_atomic_leaf_replacement_after_snapshot(
    tmp_path,
    monkeypatch,
    artifact_name,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_id = f"job_snapshot_swap_{artifact_name.replace('.', '_')}"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    digest = "6" * 64
    refinement = {
        "status": "disabled",
        "sidecar_file": "",
        "sidecar_sha256": "",
    }
    result = {
        "summary": {
            "analysis_job_id": job_id,
            "analysis_run_instance_id": job_id,
            "analysis_input_digest": digest,
            "run_fingerprint": digest,
            "refinement": refinement,
        },
        "refinement_summary": refinement,
        "refinement_candidates": [],
        "vignette_candidates": [],
        "points": [],
    }
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(result_path, class_root, result)
    _write_class_analysis_terminal_support(
        job_dir=job_dir,
        class_root=class_root,
        result=result,
        digest=digest,
    )
    state = {
        "schema": "class-analysis-job-state-v1",
        "status": "completed",
        "progress": 1.0,
        **api._class_analysis_job_state_artifact_binding(
            result_path=result_path,
            result=result,
        ),
    }
    state_path = job_dir / api.CLASS_ANALYSIS_JOB_STATE_FILENAME
    api._class_analysis_write_json(state_path, class_root, state)
    target = job_dir / artifact_name
    replacements = {
        "result.json": {
            "summary": {"analysis_job_id": "replacement"},
            "points": [{"point_id": "unvalidated-replacement"}],
        },
        "config.json": {
            "run_fingerprint": digest,
            "unvalidated_replacement": True,
        },
        api.CLASS_ANALYSIS_JOB_STATE_FILENAME: {
            **state,
            "message": "unvalidated replacement",
        },
    }
    original_snapshot = api._class_analysis_json_artifact_snapshot
    replaced = False

    def replace_after_snapshot(path):
        nonlocal replaced
        snapshot = original_snapshot(path)
        if Path(path) == target and not replaced:
            replaced = True
            api._class_analysis_write_json(
                target,
                job_dir,
                replacements[artifact_name],
            )
        return snapshot

    monkeypatch.setattr(
        api,
        "_class_analysis_json_artifact_snapshot",
        replace_after_snapshot,
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()

    try:
        restored = api._get_class_analysis_job(job_id)

        assert replaced is True
        assert restored.status == "failed"
        assert restored.result_path is None
        assert "artifact_changed" in str(restored.error)
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


def test_class_analysis_bound_result_get_rejects_replacement_after_restore(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_id = "job_result_swap_after_restore"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    digest = "5" * 64
    refinement = {
        "status": "disabled",
        "sidecar_file": "",
        "sidecar_sha256": "",
    }
    result = {
        "summary": {
            "analysis_job_id": job_id,
            "analysis_run_instance_id": job_id,
            "analysis_input_digest": digest,
            "run_fingerprint": digest,
            "refinement": refinement,
        },
        "refinement_summary": refinement,
        "refinement_candidates": [],
        "vignette_candidates": [],
        "points": [{"point_id": "validated"}],
    }
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(result_path, class_root, result)
    _write_class_analysis_terminal_support(
        job_dir=job_dir,
        class_root=class_root,
        result=result,
        digest=digest,
    )
    api._class_analysis_write_json(
        job_dir / api.CLASS_ANALYSIS_JOB_STATE_FILENAME,
        class_root,
        {
            "schema": "class-analysis-job-state-v1",
            "status": "completed",
            "progress": 1.0,
            **api._class_analysis_job_state_artifact_binding(
                result_path=result_path,
                result=result,
            ),
        },
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()

    try:
        restored = api._get_class_analysis_job(job_id)
        assert restored.status == "completed"
        assert restored._artifact_binding["result_sha256"]
        api._class_analysis_write_json(
            result_path,
            job_dir,
            {
                "summary": {"analysis_job_id": "replacement"},
                "points": [{"point_id": "unvalidated-replacement"}],
            },
        )

        with pytest.raises(api.HTTPException) as exc_info:
            api.get_class_analysis_result(job_id)

        assert exc_info.value.status_code == api.HTTP_409_CONFLICT
        assert exc_info.value.detail == (
            "class_analysis_result_artifact_changed"
        )
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


def test_class_analysis_restart_rejects_metadata_replacement_after_snapshot(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_id = "job_metadata_snapshot_swap"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    digest = "4" * 64
    result = {
        "summary": {
            "analysis_job_id": job_id,
            "analysis_run_instance_id": job_id,
            "analysis_input_digest": digest,
            "run_fingerprint": digest,
            "refinement": {
                "status": "disabled",
                "sidecar_file": "",
                "sidecar_sha256": "",
            },
        },
        "refinement_summary": {
            "status": "disabled",
            "sidecar_file": "",
            "sidecar_sha256": "",
        },
        "points": [
            {
                "point_id": "point-0",
                "split": "train",
                "image_relpath": "frame.jpg",
                "class_name": "A",
                "bbox_xyxy": [0, 0, 10, 10],
            }
        ],
    }
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(result_path, class_root, result)
    _write_class_analysis_terminal_support(
        job_dir=job_dir,
        class_root=class_root,
        result=result,
        digest=digest,
    )
    api._class_analysis_write_json(
        job_dir / api.CLASS_ANALYSIS_JOB_STATE_FILENAME,
        class_root,
        {
            "schema": "class-analysis-job-state-v1",
            "status": "completed",
            "progress": 1.0,
            **api._class_analysis_job_state_artifact_binding(
                result_path=result_path,
                result=result,
            ),
        },
    )
    metadata_path = job_dir / api.CLASS_ANALYSIS_METADATA_FILENAME
    original_snapshot = api._class_analysis_binary_artifact_snapshot
    replaced = False

    def replace_after_snapshot(path, **kwargs):
        nonlocal replaced
        snapshot = original_snapshot(path, **kwargs)
        if Path(path) == metadata_path and not replaced:
            replaced = True
            replacement = metadata_path.with_name("metadata.replacement")
            replacement.write_bytes(metadata_path.read_bytes())
            os.replace(replacement, metadata_path)
        return snapshot

    monkeypatch.setattr(
        api,
        "_class_analysis_binary_artifact_snapshot",
        replace_after_snapshot,
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()
    try:
        restored = api._get_class_analysis_job(job_id)

        assert replaced is True
        assert restored.status == "failed"
        assert restored.result_path is None
        assert "class_analysis_metadata_artifact_changed" in str(
            restored.error
        )
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


@pytest.mark.parametrize(
    "artifact_name",
    ["sidecar", "refinement_manifest"],
)
def test_class_analysis_restart_rejects_refinement_leaf_replacement_after_snapshot(
    tmp_path,
    monkeypatch,
    artifact_name,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_id = f"job_{artifact_name}_snapshot_swap"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    fixture = _write_bound_refinement_terminal_fixture(
        job_dir=job_dir,
        class_root=class_root,
        job_id=job_id,
        digest="3" * 64,
    )
    target = fixture[f"{artifact_name.replace('refinement_', '')}_path"]
    original_snapshot = (
        api._class_analysis_binary_artifact_snapshot
        if artifact_name == "sidecar"
        else api._class_analysis_json_artifact_snapshot
    )
    replaced = False

    def replace_after_snapshot(path, **kwargs):
        nonlocal replaced
        snapshot = original_snapshot(path, **kwargs)
        if Path(path) == target and not replaced:
            replaced = True
            replacement = target.with_name(f"{target.name}.replacement")
            replacement.write_bytes(target.read_bytes())
            os.replace(replacement, target)
        return snapshot

    monkeypatch.setattr(
        api,
        (
            "_class_analysis_binary_artifact_snapshot"
            if artifact_name == "sidecar"
            else "_class_analysis_json_artifact_snapshot"
        ),
        replace_after_snapshot,
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()
    try:
        restored = api._get_class_analysis_job(job_id)

        assert replaced is True
        assert restored.status == "failed"
        assert restored.result_path is None
        assert f"class_analysis_{artifact_name}_artifact_changed" in str(
            restored.error
        )
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


def test_class_analysis_bound_metadata_reads_reject_post_restore_replacement(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_id = "job_bound_metadata_read_swap"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    fixture = _write_bound_refinement_terminal_fixture(
        job_dir=job_dir,
        class_root=class_root,
        job_id=job_id,
        digest="2" * 64,
    )
    thumb_dir = job_dir / "thumbnails"
    thumb_dir.mkdir()
    (thumb_dir / "point-0.jpg").write_bytes(b"cached-thumbnail")
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()
    try:
        restored = api._get_class_analysis_job(job_id)
        assert restored.status == "completed"
        assert restored._artifact_binding["_file_identities"]["metadata"]

        metadata_path = job_dir / api.CLASS_ANALYSIS_METADATA_FILENAME
        before = metadata_path.stat()
        replacement = metadata_path.with_name("metadata.replacement")
        replacement.write_bytes(metadata_path.read_bytes())
        os.utime(
            replacement,
            ns=(int(before.st_atime_ns), int(before.st_mtime_ns)),
        )
        os.replace(replacement, metadata_path)

        monkeypatch.setattr(
            api,
            "_class_analysis_binary_artifact_snapshot",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("per-point metadata reads must not rehash")
            ),
        )
        assert (
            api._class_analysis_thumbnail_point_metadata(
                restored,
                job_dir,
                "point-0",
            )
            is None
        )
        with pytest.raises(api.HTTPException) as exc_info:
            api.get_class_analysis_thumbnail(job_id, "point-0")
        assert exc_info.value.status_code == api.HTTP_404_NOT_FOUND
        assert exc_info.value.detail == "thumbnail_not_found"
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


def test_class_analysis_bound_metadata_row_rejects_swap_after_index_lookup(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_id = "job_bound_metadata_index_read_swap"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    _write_bound_refinement_terminal_fixture(
        job_dir=job_dir,
        class_root=class_root,
        job_id=job_id,
        digest="9" * 64,
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()
    try:
        restored = api._get_class_analysis_job(job_id)
        assert restored.status == "completed"
        original_offsets = api._class_analysis_thumbnail_metadata_offsets
        metadata_path = job_dir / api.CLASS_ANALYSIS_METADATA_FILENAME
        replaced = False

        def offsets_then_replace(job, resolved_job_dir):
            nonlocal replaced
            offsets = original_offsets(job, resolved_job_dir)
            if not replaced:
                replaced = True
                replacement = metadata_path.with_name(
                    "metadata.after-index"
                )
                replacement.write_bytes(metadata_path.read_bytes())
                os.replace(replacement, metadata_path)
            return offsets

        monkeypatch.setattr(
            api,
            "_class_analysis_thumbnail_metadata_offsets",
            offsets_then_replace,
        )
        assert (
            api._class_analysis_thumbnail_point_metadata(
                restored,
                job_dir,
                "point-0",
            )
            is None
        )
        assert replaced is True
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


@pytest.mark.parametrize(
    "artifact_name",
    ["sidecar", "refinement_manifest"],
)
def test_class_analysis_bound_refinement_access_rejects_replacement(
    tmp_path,
    monkeypatch,
    artifact_name,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_id = f"job_bound_{artifact_name}_read_swap"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    fixture = _write_bound_refinement_terminal_fixture(
        job_dir=job_dir,
        class_root=class_root,
        job_id=job_id,
        digest="1" * 64,
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()
    try:
        restored = api._get_class_analysis_job(job_id)
        assert restored.status == "completed"
        target = fixture[
            f"{artifact_name.replace('refinement_', '')}_path"
        ]
        before = target.stat()
        replacement = target.with_name(f"{target.name}.replacement")
        replacement.write_bytes(target.read_bytes())
        os.utime(
            replacement,
            ns=(int(before.st_atime_ns), int(before.st_mtime_ns)),
        )
        os.replace(replacement, target)
        monkeypatch.setattr(
            api,
            "_class_analysis_file_sha256_cached",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("bound refinement access must not use path cache")
            ),
        )

        with pytest.raises(
            ValueError,
            match=f"class_analysis_{artifact_name}_artifact_changed",
        ):
            api._class_analysis_refinement_sidecar_contract(
                job_dir=job_dir,
                sidecar_path=fixture["sidecar_path"],
                point_id="point-0",
                sidecar_row=0,
                artifact_job=restored,
            )
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


def test_class_analysis_hash_cache_rejects_same_size_mtime_path_replacement(
    tmp_path,
):
    path = tmp_path / "source.bin"
    path.write_bytes(b"original-bytes")
    fixed_ns = 1_700_000_000_000_000_000
    os.utime(path, ns=(fixed_ns, fixed_ns))
    with api.CLASS_ANALYSIS_FILE_HASH_CACHE_LOCK:
        api.CLASS_ANALYSIS_FILE_HASH_CACHE.clear()
    first = api._class_analysis_file_sha256_cached(path)

    replacement = tmp_path / "source.replacement"
    replacement.write_bytes(b"replaced-bytes")
    os.utime(replacement, ns=(fixed_ns, fixed_ns))
    os.replace(replacement, path)
    second = api._class_analysis_file_sha256_cached(path)

    assert first == hashlib.sha256(b"original-bytes").hexdigest()
    assert second == hashlib.sha256(b"replaced-bytes").hexdigest()
    assert second != first


@pytest.mark.parametrize(
    "tamper",
    [
        "config_content",
        "config_identity",
        "metadata_content",
        "metadata_truncated",
        "projection_content",
        "projection_missing",
    ],
)
def test_class_analysis_restart_rejects_tampered_bound_ancillary_artifact(
    tmp_path,
    monkeypatch,
    tamper,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_id = f"job_bound_{tamper}"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    digest = "7" * 64
    refinement = {
        "status": "disabled",
        "sidecar_file": "",
        "sidecar_sha256": "",
    }
    point = {
        "point_id": "point-1",
        "split": "train",
        "image_relpath": "frame.jpg",
        "class_name": "A",
        "bbox_xyxy": [0, 0, 10, 10],
    }
    result = {
        "summary": {
            "analysis_job_id": job_id,
            "analysis_run_instance_id": job_id,
            "analysis_input_digest": digest,
            "run_fingerprint": digest,
            "refinement": refinement,
        },
        "projection_options": {
            "selected": "global_pca",
            "coordinates_available": ["global_pca"],
        },
        "refinement_summary": refinement,
        "refinement_candidates": [],
        "vignette_candidates": [],
        "points": [point],
    }
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(result_path, class_root, result)
    _write_class_analysis_terminal_support(
        job_dir=job_dir,
        class_root=class_root,
        result=result,
        digest=digest,
    )
    projection_path = (
        job_dir / api.CLASS_ANALYSIS_PROJECTION_COORDS_FILENAME
    )
    api._class_analysis_write_npz(
        projection_path,
        job_dir,
        global_pca=np.asarray([[1.0, 2.0]], dtype=np.float32),
    )
    binding = api._class_analysis_job_state_artifact_binding(
        result_path=result_path,
        result=result,
    )
    assert binding["config_file"] == "config.json"
    assert binding["metadata_file"] == api.CLASS_ANALYSIS_METADATA_FILENAME
    assert binding["metadata_row_count"] == 1
    assert binding["projection_file"] == (
        api.CLASS_ANALYSIS_PROJECTION_COORDS_FILENAME
    )
    api._class_analysis_write_json(
        job_dir / api.CLASS_ANALYSIS_JOB_STATE_FILENAME,
        job_dir,
        {
            "schema": "class-analysis-job-state-v1",
            "status": "completed",
            "progress": 1.0,
            **binding,
        },
    )

    if tamper == "config_content":
        api._class_analysis_write_json(
            job_dir / "config.json",
            job_dir,
            {"run_fingerprint": digest, "tampered": True},
        )
    elif tamper == "config_identity":
        api._class_analysis_write_json(
            job_dir / "config.json",
            job_dir,
            {"run_fingerprint": "8" * 64},
        )
    elif tamper == "metadata_content":
        metadata_path = job_dir / api.CLASS_ANALYSIS_METADATA_FILENAME
        metadata_path.write_text(
            metadata_path.read_text(encoding="utf-8").replace(
                '"class_name":"A"',
                '"class_name":"B"',
            ),
            encoding="utf-8",
        )
    elif tamper == "metadata_truncated":
        (job_dir / api.CLASS_ANALYSIS_METADATA_FILENAME).write_bytes(b"")
    elif tamper == "projection_content":
        projection_path.write_bytes(projection_path.read_bytes() + b"tamper")
    else:
        projection_path.unlink()

    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()
    try:
        restored = api._get_class_analysis_job(job_id)
        assert restored.status == "failed"
        assert restored.result_path is None
        assert "class_analysis_artifact_binding_invalid" in str(
            restored.error
        )
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


@pytest.mark.parametrize(
    ("defect", "error_fragment"),
    [
        ("extra_member", "class_analysis_npz_member_set_invalid"),
        ("wrong_rows", "class_analysis_npz_header_shape_mismatch:global_pca"),
        ("wrong_dtype", "class_analysis_npz_header_dtype_mismatch:global_pca"),
        ("nonfinite", "class_analysis_projection_nonfinite:global_pca"),
    ],
)
def test_class_analysis_terminal_binding_validates_projection_array_contract(
    tmp_path,
    monkeypatch,
    defect,
    error_fragment,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_id = f"job_projection_contract_{defect}"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    digest = "6" * 64
    result = _class_analysis_projection_test_result(job_id, digest)
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(result_path, job_dir, result)
    _write_class_analysis_terminal_support(
        job_dir=job_dir,
        class_root=class_root,
        result=result,
        digest=digest,
    )
    arrays = {
        "global_pca": np.asarray(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=np.float32,
        )
    }
    if defect == "extra_member":
        arrays["class_balanced_pca"] = arrays["global_pca"].copy()
    elif defect == "wrong_rows":
        arrays["global_pca"] = arrays["global_pca"][:1]
    elif defect == "wrong_dtype":
        arrays["global_pca"] = arrays["global_pca"].astype(np.float64)
    else:
        arrays["global_pca"][1, 0] = np.nan
    api._class_analysis_write_npz(
        job_dir / api.CLASS_ANALYSIS_PROJECTION_COORDS_FILENAME,
        job_dir,
        **arrays,
    )

    with pytest.raises(ValueError, match=error_fragment):
        api._class_analysis_job_state_artifact_binding(
            result_path=result_path,
            result=result,
        )


def test_class_analysis_projection_preflights_declared_shape_before_np_load(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_id = "job_projection_huge_header"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    digest = "5" * 64
    result = _class_analysis_projection_test_result(job_id, digest)
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(result_path, job_dir, result)
    _write_class_analysis_terminal_support(
        job_dir=job_dir,
        class_root=class_root,
        result=result,
        digest=digest,
    )
    encoded = BytesIO()
    np.lib.format.write_array_header_1_0(
        encoded,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
            "fortran_order": False,
            "shape": (100_000_000, 2),
        },
    )
    projection_path = job_dir / api.CLASS_ANALYSIS_PROJECTION_COORDS_FILENAME
    with zipfile.ZipFile(
        projection_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("global_pca.npy", encoded.getvalue())
    load_calls = []

    def reject_np_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        raise AssertionError("np.load must follow bounded NPY preflight")

    monkeypatch.setattr(api.np, "load", reject_np_load)

    with pytest.raises(ValueError):
        api._class_analysis_job_state_artifact_binding(
            result_path=result_path,
            result=result,
        )
    assert load_calls == []


@pytest.mark.parametrize("defect", ["unadvertised", "wrong_rows"])
def test_class_analysis_projection_endpoint_rejects_artifact_contract_drift(
    tmp_path,
    monkeypatch,
    defect,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_id = f"job_projection_endpoint_{defect}"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    result = _class_analysis_projection_test_result(job_id, "4" * 64)
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(result_path, job_dir, result)
    arrays = {
        "global_pca": np.asarray(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=np.float32,
        )
    }
    requested_mode = "global_pca"
    if defect == "unadvertised":
        arrays["class_balanced_pca"] = arrays["global_pca"].copy()
        requested_mode = "class_balanced_pca"
    else:
        arrays["global_pca"] = arrays["global_pca"][:1]
    api._class_analysis_write_npz(
        job_dir / api.CLASS_ANALYSIS_PROJECTION_COORDS_FILENAME,
        job_dir,
        **arrays,
    )
    job = api.ClassAnalysisJob(
        job_id=job_id,
        status="completed",
        result_path=str(result_path),
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS[job_id] = job
    try:
        with pytest.raises(api.HTTPException) as error:
            api.get_class_analysis_projection(job_id, requested_mode)
        assert error.value.status_code == api.HTTP_404_NOT_FOUND
        assert error.value.detail == "projection_not_found"
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.pop(job_id, None)


@pytest.mark.parametrize("tamper", ["projection", "result_contract"])
def test_class_analysis_projection_endpoint_rejects_post_restore_tamper(
    tmp_path,
    monkeypatch,
    tamper,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_id = f"job_projection_endpoint_tamper_{tamper}"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    digest = "3" * 64
    result = _class_analysis_projection_test_result(job_id, digest)
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(result_path, job_dir, result)
    _write_class_analysis_terminal_support(
        job_dir=job_dir,
        class_root=class_root,
        result=result,
        digest=digest,
    )
    projection_path = job_dir / api.CLASS_ANALYSIS_PROJECTION_COORDS_FILENAME
    api._class_analysis_write_npz(
        projection_path,
        job_dir,
        global_pca=np.asarray(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=np.float32,
        ),
    )
    binding = api._class_analysis_job_state_artifact_binding(
        result_path=result_path,
        result=result,
    )
    api._class_analysis_write_json(
        job_dir / api.CLASS_ANALYSIS_JOB_STATE_FILENAME,
        job_dir,
        {
            "schema": "class-analysis-job-state-v1",
            "status": "completed",
            "progress": 1.0,
            **binding,
        },
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()
    try:
        restored = api._get_class_analysis_job(job_id)
        assert restored.status == "completed"
        assert restored._artifact_binding["projection_sha256"] == binding[
            "projection_sha256"
        ]
        if tamper == "projection":
            api._class_analysis_write_npz(
                projection_path,
                job_dir,
                global_pca=np.asarray(
                    [[11.0, 12.0], [13.0, 14.0]],
                    dtype=np.float32,
                ),
            )
        else:
            tampered_result = copy.deepcopy(result)
            tampered_result.pop("projection_options", None)
            tampered_result["points"][0]["projection"] = [91.0, 92.0]
            tampered_result["points"][1]["projection"] = [93.0, 94.0]
            api._class_analysis_write_json(
                result_path,
                job_dir,
                tampered_result,
            )

        with pytest.raises(api.HTTPException) as error:
            api.get_class_analysis_projection(job_id, "global_pca")
        assert error.value.status_code == api.HTTP_404_NOT_FOUND
        assert error.value.detail == "projection_not_found"
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


def test_class_analysis_restart_restores_validated_terminal_memory_payload(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_id = "job_memory_restore"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    digest = "9" * 64
    result_memory = {
        "job_start_baseline_combined_rss_bytes": 1_000,
        "peak_job_combined_rss_bytes": 1_200,
        "peak_job_incremental_combined_rss_bytes": 200,
    }
    terminal_memory = {
        **result_memory,
        "backend_rss_bytes": 900,
        "worker_rss_bytes": 400,
        "combined_rss_bytes": 1_300,
        "system_available_bytes": 8_000,
        "system_total_bytes": 16_000,
        "peak_job_combined_rss_bytes": 1_300,
        "peak_job_incremental_combined_rss_bytes": 300,
    }
    publication_memory = {
        "result_json_writer": "json_encoder_iterencode_atomic",
        "measured_through": "first_terminal_state_write",
        "final_corrective_write": "bounded_atomic_no_recursive_sample",
        "pre_result_peak_combined_rss_bytes": 1_200,
        "post_result_peak_combined_rss_bytes": 1_300,
        "first_terminal_state_peak_combined_rss_bytes": 1_300,
    }
    refinement = {
        "status": "disabled",
        "sidecar_file": "",
        "sidecar_sha256": "",
        "resource_metrics": dict(result_memory),
    }
    result = {
        "summary": {
            "analysis_job_id": job_id,
            "analysis_run_instance_id": job_id,
            "analysis_input_digest": digest,
            "run_fingerprint": digest,
            "runtime": {
                "completed_objects": 2,
                "job_memory": dict(result_memory),
            },
            "refinement": refinement,
        },
        "refinement_summary": refinement,
        "refinement_candidates": [],
        "vignette_candidates": [],
        "points": [],
    }
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(result_path, class_root, result)
    request = {
        "analysis_scope": "all_classes",
        "run_fingerprint": digest,
    }
    _write_class_analysis_terminal_support(
        job_dir=job_dir,
        class_root=class_root,
        result=result,
        digest=digest,
    )
    api._class_analysis_write_json(
        job_dir / api.CLASS_ANALYSIS_JOB_STATE_FILENAME,
        class_root,
        {
            "schema": "class-analysis-job-state-v1",
            "status": "completed",
            "progress": 1.0,
            "message": "Class analysis completed.",
            "job_memory": terminal_memory,
            "publication_memory": publication_memory,
            **api._class_analysis_job_state_artifact_binding(
                result_path=result_path,
                result=result,
            ),
        },
    )
    live_job = api.ClassAnalysisJob(
        job_id=job_id,
        status="completed",
        progress=1.0,
        message="Class analysis completed.",
        request=request,
        summary=dict(result["summary"]),
        result_path=str(result_path),
        runtime={
            "completed_objects": 2,
            "job_memory": dict(terminal_memory),
            "publication_memory": dict(publication_memory),
        },
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()
        api.CLASS_ANALYSIS_JOBS[job_id] = live_job

    try:
        live_payload = api.get_class_analysis_job(job_id)
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()
        restored_payload = api.get_class_analysis_job(job_id)

        assert restored_payload["status"] == live_payload["status"]
        assert restored_payload["progress"] == live_payload["progress"]
        assert restored_payload["message"] == live_payload["message"]
        assert restored_payload["run_fingerprint"] == digest
        assert restored_payload["runtime"] == live_payload["runtime"]
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


def test_class_analysis_restart_does_not_surface_invalid_terminal_memory(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_id = "job_invalid_memory_restore"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    digest = "8" * 64
    refinement = {
        "status": "disabled",
        "sidecar_file": "",
        "sidecar_sha256": "",
    }
    result = {
        "summary": {
            "analysis_job_id": job_id,
            "analysis_run_instance_id": job_id,
            "analysis_input_digest": digest,
            "run_fingerprint": digest,
            "runtime": {
                "completed_objects": 2,
                "job_memory": {
                    "peak_job_incremental_combined_rss_bytes": 1,
                },
            },
            "refinement": refinement,
        },
        "refinement_summary": refinement,
        "points": [],
    }
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(result_path, class_root, result)
    _write_class_analysis_terminal_support(
        job_dir=job_dir,
        class_root=class_root,
        result=result,
        digest=digest,
    )
    api._class_analysis_write_json(
        job_dir / api.CLASS_ANALYSIS_JOB_STATE_FILENAME,
        class_root,
        {
            "schema": "class-analysis-job-state-v1",
            "status": "completed",
            "progress": 1.0,
            "job_memory": {
                "job_start_baseline_combined_rss_bytes": 1_000,
                "peak_job_combined_rss_bytes": 1_100,
                # Deliberately contradict peak - baseline.
                "peak_job_incremental_combined_rss_bytes": 1,
            },
            "publication_memory": {
                "result_json_writer": "json_encoder_iterencode_atomic",
                "measured_through": "first_terminal_state_write",
                "final_corrective_write": (
                    "bounded_atomic_no_recursive_sample"
                ),
                "pre_result_peak_combined_rss_bytes": 1_050,
                "post_result_peak_combined_rss_bytes": 1_100,
                "first_terminal_state_peak_combined_rss_bytes": 1_100,
            },
            **api._class_analysis_job_state_artifact_binding(
                result_path=result_path,
                result=result,
            ),
        },
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()

    try:
        payload = api.get_class_analysis_job(job_id)

        assert payload["status"] == "completed"
        assert payload["runtime"] == {"completed_objects": 2}
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


@pytest.mark.parametrize(
    ("state_status", "refinement_status"),
    [
        ("cancelled", "completed"),
        ("completed", ""),
        ("completed", "unknown"),
    ],
)
def test_class_analysis_restart_rejects_terminal_result_status_mismatch(
    tmp_path,
    monkeypatch,
    state_status,
    refinement_status,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_id = "job_cancel_status_mismatch"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    digest = "e" * 64
    refinement = {
        "status": refinement_status,
        "schema": api.CLASS_ANALYSIS_REFINEMENT_SCHEMA,
        "sidecar_file": "",
        "sidecar_sha256": "",
    }
    result = {
        "summary": {
            "analysis_job_id": job_id,
            "analysis_run_instance_id": job_id,
            "analysis_input_digest": digest,
            "run_fingerprint": digest,
            "refinement": refinement,
        },
        "refinement_summary": refinement,
        "refinement_candidates": [],
        "vignette_candidates": [],
        "points": [],
    }
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(result_path, class_root, result)
    _write_class_analysis_terminal_support(
        job_dir=job_dir,
        class_root=class_root,
        result=result,
        digest=digest,
    )
    api._class_analysis_write_json(
        job_dir / api.CLASS_ANALYSIS_JOB_STATE_FILENAME,
        class_root,
        {
            "schema": "class-analysis-job-state-v1",
            "status": state_status,
            "progress": 1.0,
            **api._class_analysis_job_state_artifact_binding(
                result_path=result_path,
                result=result,
            ),
        },
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()

    try:
        job = api._get_class_analysis_job(job_id)
        assert job.status == "failed"
        assert "job_state_result_status_mismatch" in str(job.error)
        assert job.result_path is None
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


def test_terminal_binding_rejects_contradictory_refinement_summaries(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_id = "job_summary_mismatch"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    digest = "f" * 64
    result = {
        "summary": {
            "analysis_job_id": job_id,
            "analysis_run_instance_id": job_id,
            "analysis_input_digest": digest,
            "run_fingerprint": digest,
            "refinement": {
                "status": "disabled",
                "schema": api.CLASS_ANALYSIS_REFINEMENT_SCHEMA,
                "sidecar_file": "",
                "sidecar_sha256": "",
            },
        },
        "refinement_summary": {
            "status": "completed",
            "schema": api.CLASS_ANALYSIS_REFINEMENT_SCHEMA,
            "sidecar_file": "",
            "sidecar_sha256": "",
        },
        "points": [],
    }
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(result_path, class_root, result)

    with pytest.raises(
        ValueError,
        match="class_analysis_refinement_summary_mismatch:status",
    ):
        api._class_analysis_job_state_artifact_binding(
            result_path=result_path,
            result=result,
        )


def test_class_analysis_restart_rejects_result_after_terminal_state_write_failure(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_id = "job_interrupted_publication"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    digest = "b" * 64
    sidecar_path = job_dir / api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME
    sidecar_path.write_bytes(b"published-before-state-write")
    sidecar_sha256 = api._class_analysis_file_sha256(sidecar_path)
    api._class_analysis_write_json(
        job_dir / api.CLASS_ANALYSIS_REFINEMENT_MANIFEST_FILENAME,
        class_root,
        {
            "schema": api.CLASS_ANALYSIS_REFINEMENT_SCHEMA,
            "decision_contract": (
                api.CLASS_ANALYSIS_REFINEMENT_DECISION_CONTRACT
            ),
            "selector_priority_contract": (
                api.CLASS_ANALYSIS_SELECTOR_PRIORITY_CONTRACT
            ),
            "capture_group_contract": (
                api.CLASS_ANALYSIS_CAPTURE_GROUP_CONTRACT
            ),
            "sidecar_file": api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME,
            "grid_shape": [14, 14],
            "sidecar": {
                "file": api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME,
                "sha256": sidecar_sha256,
                "bytes": sidecar_path.stat().st_size,
                "arrays": {
                    "point_ids": {"shape": [1], "dtype": "<U8"},
                    "current_heatmaps": {
                        "shape": [1, 2, 14, 14],
                        "dtype": "float16",
                    },
                    "alternative_heatmaps": {
                        "shape": [1, 2, 14, 14],
                        "dtype": "float16",
                    },
                    "valid_masks": {
                        "shape": [1, 2, 14, 14],
                        "dtype": "uint8",
                    },
                    "target_masks": {
                        "shape": [1, 2, 14, 14],
                        "dtype": "uint8",
                    },
                    "overlap_masks": {
                        "shape": [1, 2, 14, 14],
                        "dtype": "uint8",
                    },
                },
            },
            "point_rows": {"point-0": 0},
        },
    )
    result = {
        "summary": {
            "analysis_scope": "all_classes",
            "analysis_job_id": job_id,
            "analysis_run_instance_id": job_id,
            "analysis_input_digest": digest,
            "run_fingerprint": digest,
            "refinement": {
                "status": "completed",
                "sidecar_file": api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME,
                "sidecar_sha256": sidecar_sha256,
            },
        },
        "refinement_summary": {
            "status": "completed",
            "sidecar_file": api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME,
            "sidecar_sha256": sidecar_sha256,
        },
        "refinement_candidates": [],
        "vignette_candidates": [],
        "points": [],
    }
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(result_path, class_root, result)
    # Simulate the real outer-finally path after the terminal state write
    # fails: Stage-2 artifacts are removed, while result.json remains.
    sidecar_path.unlink()
    (job_dir / api.CLASS_ANALYSIS_REFINEMENT_MANIFEST_FILENAME).unlink()
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()

    try:
        job = api._get_class_analysis_job(job_id)

        assert job.status == "failed"
        assert job.result_path is None
        assert "job_state_missing" in str(job.error)
        with pytest.raises(api.HTTPException) as exc_info:
            api.get_class_analysis_result(job_id)
        assert exc_info.value.status_code == api.HTTP_404_NOT_FOUND
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


def test_class_analysis_restart_verifies_refinement_manifest_binding(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_id = "job_refinement_binding"
    job_dir = api._class_analysis_job_dir(job_id, create=True)
    digest = "c" * 64
    sidecar_path = job_dir / api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME
    sidecar_path.write_bytes(b"bounded-sidecar")
    sidecar_sha256 = api._class_analysis_file_sha256(sidecar_path)
    manifest_path = job_dir / api.CLASS_ANALYSIS_REFINEMENT_MANIFEST_FILENAME
    exact_view_pair_calibration = (
        _class_analysis_test_exact_view_pair_calibration()
    )
    selector_v6_binding = _class_analysis_test_selector_v6_binding()
    selector_priority = api._class_analysis_assign_selector_priority_ranks([])
    api._class_analysis_write_json(
        manifest_path,
        class_root,
        {
            "schema": api.CLASS_ANALYSIS_REFINEMENT_SCHEMA,
            "decision_contract": (
                api.CLASS_ANALYSIS_REFINEMENT_DECISION_CONTRACT
            ),
            "selector_priority_contract": (
                api.CLASS_ANALYSIS_SELECTOR_PRIORITY_CONTRACT
            ),
            **selector_v6_binding,
            "capture_group_contract": (
                api.CLASS_ANALYSIS_CAPTURE_GROUP_CONTRACT
            ),
            "exact_view_pair_calibration": exact_view_pair_calibration,
            "sidecar_file": api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME,
            "grid_shape": [3, 3],
            "sidecar": {
                "file": api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME,
                "sha256": sidecar_sha256,
                "bytes": sidecar_path.stat().st_size,
                "arrays": {
                    "point_ids": {"shape": [1], "dtype": "<U8"},
                    "current_heatmaps": {
                        "shape": [1, 2, 3, 3],
                        "dtype": "float16",
                    },
                    "alternative_heatmaps": {
                        "shape": [1, 2, 3, 3],
                        "dtype": "float16",
                    },
                    "valid_masks": {
                        "shape": [1, 2, 3, 3],
                        "dtype": "uint8",
                    },
                    "target_masks": {
                        "shape": [1, 2, 3, 3],
                        "dtype": "uint8",
                    },
                    "overlap_masks": {
                        "shape": [1, 2, 3, 3],
                        "dtype": "uint8",
                    },
                },
            },
            "point_rows": {"point-0": 0},
        },
    )
    refinement = {
        "status": "completed",
        "schema": api.CLASS_ANALYSIS_REFINEMENT_SCHEMA,
        "decision_contract": (
            api.CLASS_ANALYSIS_REFINEMENT_DECISION_CONTRACT
        ),
        "selector_priority_contract": (
            api.CLASS_ANALYSIS_SELECTOR_PRIORITY_CONTRACT
        ),
        **selector_v6_binding,
        "capture_group_contract": api.CLASS_ANALYSIS_CAPTURE_GROUP_CONTRACT,
        "selector_priority": selector_priority,
        "selector_priority_candidate_count": 0,
        "rough_candidate_count": 0,
        "exact_view_pair_calibration": exact_view_pair_calibration,
        "sidecar_file": api.CLASS_ANALYSIS_REFINEMENT_SIDECAR_FILENAME,
        "sidecar_sha256": sidecar_sha256,
    }
    result = {
        "summary": {
            "analysis_scope": "all_classes",
            "analysis_job_id": job_id,
            "analysis_run_instance_id": job_id,
            "analysis_input_digest": digest,
            "run_fingerprint": digest,
            "refinement": refinement,
        },
        "refinement_summary": refinement,
        "refinement_candidates": [],
        "wrong_class_candidates": [],
        "vignette_candidates": [],
        "points": [],
    }
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(result_path, class_root, result)
    _write_class_analysis_terminal_support(
        job_dir=job_dir,
        class_root=class_root,
        result=result,
        digest=digest,
    )
    binding = api._class_analysis_job_state_artifact_binding(
        result_path=result_path,
        result=result,
    )
    assert binding["refinement_manifest_file"] == manifest_path.name
    assert binding["refinement_manifest_sha256"] == (
        api._class_analysis_file_sha256(manifest_path)
    )
    api._class_analysis_write_json(
        job_dir / api.CLASS_ANALYSIS_JOB_STATE_FILENAME,
        class_root,
        {
            "schema": "class-analysis-job-state-v1",
            "status": "completed",
            "progress": 1.0,
            **binding,
        },
    )
    tampered_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered_manifest["point_rows"] = {"point-0": 0, "duplicate": 0}
    manifest_path.write_text(
        json.dumps(tampered_manifest),
        encoding="utf-8",
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()

    try:
        job = api._get_class_analysis_job(job_id)
        assert job.status == "failed"
        assert job.result_path is None
        assert "artifact_binding_invalid" in str(job.error)
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


def test_class_analysis_projection_endpoint_maps_legacy_pca_points_to_global(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_dir = api._class_analysis_job_dir("job_legacy_projection", create=True)
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(
        result_path,
        class_root,
        {
            "summary": {"projection": "pca"},
            "points": [
                {"point_id": "a", "projection": [1.0, 2.0]},
                {"point_id": "b", "projection": [3.0, 4.0]},
            ],
            "projection_options": {},
        },
    )
    job = api.ClassAnalysisJob(
        job_id="job_legacy_projection",
        status="completed",
        result_path=str(result_path),
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()
        api.CLASS_ANALYSIS_JOBS[job.job_id] = job

    try:
        payload = api.get_class_analysis_projection(job.job_id, "global_pca")
        assert payload["mode"] == "global_pca"
        assert payload["coordinates"] == [[1.0, 2.0], [3.0, 4.0]]
        with pytest.raises(api.HTTPException) as exc_info:
            api.get_class_analysis_projection(job.job_id, "class_balanced_pca")
        assert exc_info.value.status_code == api.HTTP_404_NOT_FOUND
        assert exc_info.value.detail == "projection_not_found"
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


def test_class_analysis_projection_endpoint_rejects_corrupt_legacy_points(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_dir = api._class_analysis_job_dir("job_corrupt_projection", create=True)
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(
        result_path,
        class_root,
        {
            "summary": {"projection": "pca"},
            "points": [
                {"point_id": "a", "projection": [1.0, 2.0]},
                {"point_id": "b", "projection": ["not-a-number", 4.0]},
            ],
            "projection_options": {},
        },
    )
    job = api.ClassAnalysisJob(
        job_id="job_corrupt_projection",
        status="completed",
        result_path=str(result_path),
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()
        api.CLASS_ANALYSIS_JOBS[job.job_id] = job

    try:
        with pytest.raises(api.HTTPException) as exc_info:
            api.get_class_analysis_projection(job.job_id, "global_pca")
        assert exc_info.value.status_code == api.HTTP_404_NOT_FOUND
        assert exc_info.value.detail == "projection_not_found"
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


def test_class_analysis_projection_endpoint_maps_unannotated_legacy_points_to_global(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_dir = api._class_analysis_job_dir("job_legacy_projection_no_summary", create=True)
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(
        result_path,
        class_root,
        {
            "summary": {},
            "points": [
                {"point_id": "a", "projection": [5.0, 6.0]},
                {"point_id": "b", "projection": [7.0, 8.0]},
            ],
        },
    )
    job = api.ClassAnalysisJob(
        job_id="job_legacy_projection_no_summary",
        status="completed",
        result_path=str(result_path),
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()
        api.CLASS_ANALYSIS_JOBS[job.job_id] = job

    try:
        payload = api.get_class_analysis_projection(job.job_id, "global_pca")
        assert payload["mode"] == "global_pca"
        assert payload["coordinates"] == [[5.0, 6.0], [7.0, 8.0]]
        with pytest.raises(api.HTTPException) as exc_info:
            api.get_class_analysis_projection(job.job_id, "class_balanced_pca")
        assert exc_info.value.status_code == api.HTTP_404_NOT_FOUND
        assert exc_info.value.detail == "projection_not_found"
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


def test_class_analysis_result_rejects_symlinked_result_escape(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    job_root = class_root / "job_escape"
    job_root.mkdir(parents=True)
    outside = tmp_path / "outside_result.json"
    outside.write_text('{"escaped":true}', encoding="utf-8")
    result_link = job_root / "result.json"
    try:
        result_link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job = api.ClassAnalysisJob(
        job_id="job_escape",
        status="completed",
        result_path=str(result_link),
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()
        api.CLASS_ANALYSIS_JOBS[job.job_id] = job

    try:
        with pytest.raises(api.HTTPException) as exc_info:
            api.get_class_analysis_result(job.job_id)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "result_not_found"
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


def test_class_analysis_thumbnail_rejects_symlinked_thumbnail_dir_escape(
    tmp_path, monkeypatch
):
    class_root = tmp_path / "class_analysis"
    job_root = class_root / "job_thumb"
    job_root.mkdir(parents=True)
    outside = tmp_path / "outside_thumbs"
    outside.mkdir()
    (outside / "pt1.jpg").write_bytes(b"jpeg")
    thumb_link = job_root / "thumbnails"
    try:
        thumb_link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job = api.ClassAnalysisJob(
        job_id="job_thumb",
        status="completed",
        thumbnail_dir=str(thumb_link),
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()
        api.CLASS_ANALYSIS_JOBS[job.job_id] = job

    try:
        with pytest.raises(api.HTTPException) as exc_info:
            api.get_class_analysis_thumbnail(job.job_id, "pt1")
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "thumbnail_not_found"
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


def test_class_analysis_indexed_metadata_uses_binary_offsets_for_unicode_rows(
    tmp_path, monkeypatch
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_dir = api._class_analysis_job_dir("job_unicode_offsets", create=True)
    metadata_path = job_dir / api.CLASS_ANALYSIS_METADATA_FILENAME
    rows = [
        {
            "point_id": "point_a",
            "image_relpath": "éolienne_日本.jpg",
            "split": "train",
            "bbox_xyxy": [1, 2, 3, 4],
        },
        {
            "point_id": "point_b",
            "image_relpath": "boat.jpg",
            "split": "train",
            "bbox_xyxy": [5, 6, 7, 8],
        },
    ]

    api._class_analysis_write_indexed_metadata_jsonl(
        metadata_path,
        job_dir,
        rows,
    )

    index_payload = json.loads(
        (job_dir / api.CLASS_ANALYSIS_THUMBNAIL_INDEX_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    metadata_lines = metadata_path.read_bytes().splitlines(keepends=True)
    assert index_payload["schema"] == api.CLASS_ANALYSIS_THUMBNAIL_INDEX_SCHEMA
    assert index_payload["offsets"]["point_a"] == 0
    assert index_payload["offsets"]["point_b"] == len(metadata_lines[0])
    with metadata_path.open("rb") as handle:
        handle.seek(index_payload["offsets"]["point_b"])
        assert json.loads(handle.readline())["point_id"] == "point_b"


def test_class_analysis_thumbnail_lazily_indexes_metadata_without_loading_result(
    tmp_path, monkeypatch
):
    class_root = tmp_path / "class_analysis"
    source_image = tmp_path / "source.jpg"
    Image.new("RGB", (100, 80), (20, 100, 180)).save(source_image)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_dir = api._class_analysis_job_dir("job_lazy_thumb", create=True)
    workspace_dir = job_dir / "active_workspace"
    workspace_dir.mkdir()
    api._class_analysis_write_jsonl(
        job_dir / api.CLASS_ANALYSIS_METADATA_FILENAME,
        job_dir,
        [
            {
                "point_id": "point_lazy",
                "image_relpath": "source.jpg",
                "split": "train",
                "bbox_xyxy": [10, 10, 70, 60],
            }
        ],
    )
    assert not (job_dir / api.CLASS_ANALYSIS_THUMBNAIL_INDEX_FILENAME).exists()
    job = api.ClassAnalysisJob(
        job_id="job_lazy_thumb",
        status="completed",
        request={
            "source_mode": "active_workspace",
            "workspace_dir": str(workspace_dir),
            "yolo_layout": "flat",
            "crop_mode": "padded_square",
            "padding_ratio": 0.08,
        },
    )
    result_calls = []

    def forbidden_result(*_args, **_kwargs):
        result_calls.append(True)
        raise AssertionError("thumbnail generation must not load result.json")

    monkeypatch.setattr(api, "get_class_analysis_result", forbidden_result)
    monkeypatch.setattr(
        api,
        "_class_analysis_source",
        lambda _request: (_ for _ in ()).throw(
            AssertionError("thumbnail generation must not parse the source manifest")
        ),
    )
    monkeypatch.setattr(
        api,
        "_resolve_annotation_image_path",
        lambda *_args, **_kwargs: source_image,
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_schedule_thumbnail_lru_prune",
        lambda **_kwargs: None,
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()
        api.CLASS_ANALYSIS_JOBS[job.job_id] = job

    try:
        first = api.get_class_analysis_thumbnail(job.job_id, "point_lazy")
        assert first.media_type == "image/jpeg"
        with Image.open(BytesIO(first.body)) as thumbnail:
            assert thumbnail.format == "JPEG"
            assert thumbnail.width > 0
            assert thumbnail.height > 0
        assert (job_dir / api.CLASS_ANALYSIS_THUMBNAIL_INDEX_FILENAME).is_file()

        monkeypatch.setattr(
            api,
            "_class_analysis_source_locator",
            lambda _request: (_ for _ in ()).throw(
                AssertionError("cached thumbnail must not reopen its source")
            ),
        )
        second = api.get_class_analysis_thumbnail(job.job_id, "point_lazy")
        assert second.body == first.body
        assert result_calls == []
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


def test_class_analysis_wide_thumbnail_is_read_only_context_with_bbox(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    source_image = tmp_path / "source_wide.jpg"
    Image.new("RGB", (1200, 800), (24, 92, 146)).save(source_image)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_dir = api._class_analysis_job_dir("job_wide_thumb", create=True)
    workspace_dir = job_dir / "active_workspace"
    workspace_dir.mkdir()
    api._class_analysis_write_jsonl(
        job_dir / api.CLASS_ANALYSIS_METADATA_FILENAME,
        job_dir,
        [
            {
                "point_id": "point_wide",
                "image_relpath": "source_wide.jpg",
                "split": "train",
                "bbox_xyxy": [550, 350, 650, 450],
            }
        ],
    )
    job = api.ClassAnalysisJob(
        job_id="job_wide_thumb",
        status="completed",
        request={
            "source_mode": "active_workspace",
            "workspace_dir": str(workspace_dir),
            "yolo_layout": "flat",
        },
    )
    monkeypatch.setattr(
        api,
        "_resolve_annotation_image_path",
        lambda *_args, **_kwargs: source_image,
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_schedule_thumbnail_lru_prune",
        lambda **_kwargs: None,
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()
        api.CLASS_ANALYSIS_JOBS[job.job_id] = job

    try:
        response = api.get_class_analysis_thumbnail(
            job.job_id,
            "point_wide",
            "wide",
        )
        assert response.media_type == "image/jpeg"
        assert response.headers["x-tator-thumbnail-context"] == "wide"
        with Image.open(BytesIO(response.body)) as thumbnail:
            assert max(thumbnail.size) > 256
            assert max(thumbnail.size) <= 900
        assert (job_dir / "thumbnails" / "point_wide.wide.jpg").is_file()
        assert job.status == "completed"
        assert job.request["source_mode"] == "active_workspace"
        with pytest.raises(api.HTTPException) as exc_info:
            api.get_class_analysis_thumbnail(
                job.job_id,
                "point_wide",
                "unknown",
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "thumbnail_context_invalid"
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


def test_class_analysis_thumbnail_stale_index_is_rebuilt_only_once(
    tmp_path, monkeypatch
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_dir = api._class_analysis_job_dir("job_stale_thumb_index", create=True)
    api._class_analysis_write_jsonl(
        job_dir / api.CLASS_ANALYSIS_METADATA_FILENAME,
        job_dir,
        [
            {
                "point_id": "point_stale",
                "image_relpath": "source.jpg",
                "split": "train",
                "bbox_xyxy": [1, 2, 3, 4],
            }
        ],
    )
    api._class_analysis_write_json(
        job_dir / api.CLASS_ANALYSIS_THUMBNAIL_INDEX_FILENAME,
        job_dir,
        {
            "schema": api.CLASS_ANALYSIS_THUMBNAIL_INDEX_SCHEMA,
            "metadata_file": api.CLASS_ANALYSIS_METADATA_FILENAME,
            "metadata_size": 1,
            "metadata_mtime_ns": 1,
            "point_count": 1,
            "offsets": {"point_stale": 0},
        },
    )
    job = api.ClassAnalysisJob(job_id="job_stale_thumb_index", status="completed")
    build_calls = []
    original_build = api._class_analysis_build_thumbnail_metadata_index

    def counted_build(resolved_job_dir):
        build_calls.append(resolved_job_dir)
        return original_build(resolved_job_dir)

    monkeypatch.setattr(
        api,
        "_class_analysis_build_thumbnail_metadata_index",
        counted_build,
    )

    first = api._class_analysis_thumbnail_metadata_offsets(job, job_dir)
    second = api._class_analysis_thumbnail_metadata_offsets(job, job_dir)

    assert first == {"point_stale": 0}
    assert second is first
    assert build_calls == [job_dir]


def test_class_analysis_thumbnail_rejects_index_offset_for_a_different_point(
    tmp_path, monkeypatch
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_dir = api._class_analysis_job_dir("job_wrong_thumb_offset", create=True)
    metadata_path = job_dir / api.CLASS_ANALYSIS_METADATA_FILENAME
    api._class_analysis_write_indexed_metadata_jsonl(
        metadata_path,
        job_dir,
        [
            {
                "point_id": "point_expected",
                "image_relpath": "expected.jpg",
                "split": "train",
                "bbox_xyxy": [1, 2, 3, 4],
            },
            {
                "point_id": "point_other",
                "image_relpath": "other.jpg",
                "split": "train",
                "bbox_xyxy": [5, 6, 7, 8],
            },
        ],
    )
    index_path = job_dir / api.CLASS_ANALYSIS_THUMBNAIL_INDEX_FILENAME
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    index_payload["offsets"]["point_expected"] = index_payload["offsets"]["point_other"]
    api._class_analysis_write_json(index_path, job_dir, index_payload)
    job = api.ClassAnalysisJob(job_id="job_wrong_thumb_offset", status="completed")

    assert (
        api._class_analysis_thumbnail_point_metadata(
            job,
            job_dir,
            "point_expected",
        )
        is None
    )


def test_class_analysis_thumbnail_coalesces_same_point_generation(
    tmp_path, monkeypatch
):
    class_root = tmp_path / "class_analysis"
    source_image = tmp_path / "source.jpg"
    Image.new("RGB", (80, 80), (200, 80, 40)).save(source_image)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job_dir = api._class_analysis_job_dir("job_coalesced_thumb", create=True)
    api._class_analysis_write_indexed_metadata_jsonl(
        job_dir / api.CLASS_ANALYSIS_METADATA_FILENAME,
        job_dir,
        [
            {
                "point_id": "point_shared",
                "image_relpath": "source.jpg",
                "split": "train",
                "bbox_xyxy": [5, 5, 65, 65],
            }
        ],
    )
    job = api.ClassAnalysisJob(
        job_id="job_coalesced_thumb",
        status="completed",
        request={"crop_mode": "padded_square", "padding_ratio": 0.08},
    )
    monkeypatch.setattr(
        api,
        "get_class_analysis_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("thumbnail generation must not load result.json")
        ),
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_source_locator",
        lambda _request: {"dataset_root": tmp_path, "yolo_layout": "flat"},
    )
    monkeypatch.setattr(
        api,
        "_resolve_annotation_image_path",
        lambda *_args, **_kwargs: source_image,
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_schedule_thumbnail_lru_prune",
        lambda **_kwargs: None,
    )
    original_write_jpeg = api._class_analysis_write_jpeg
    write_entered = api.threading.Event()
    release_write = api.threading.Event()
    write_calls = []

    def delayed_write(*args, **kwargs):
        write_calls.append(True)
        write_entered.set()
        assert release_write.wait(timeout=5.0)
        return original_write_jpeg(*args, **kwargs)

    monkeypatch.setattr(api, "_class_analysis_write_jpeg", delayed_write)
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.clear()
        api.CLASS_ANALYSIS_JOBS[job.job_id] = job

    try:
        with api.ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(
                api.get_class_analysis_thumbnail,
                job.job_id,
                "point_shared",
            )
            assert write_entered.wait(timeout=5.0)
            second_future = executor.submit(
                api.get_class_analysis_thumbnail,
                job.job_id,
                "point_shared",
            )
            release_write.set()
            first = first_future.result(timeout=5.0)
            second = second_future.result(timeout=5.0)
        assert first.body == second.body
        assert len(write_calls) == 1
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()


def test_class_analysis_copy_file_hardlinks_when_possible(tmp_path):
    source_root = tmp_path / "source"
    dest_root = tmp_path / "dest"
    source_root.mkdir()
    dest_root.mkdir()
    source = source_root / "thumb.jpg"
    dest = dest_root / "thumb.jpg"
    source.write_bytes(b"cached thumbnail")

    try:
        assert api._class_analysis_copy_file_within_roots(
            source,
            dest,
            source_root=source_root,
            dest_root=dest_root,
        )
    except OSError as exc:
        pytest.skip(f"hardlinks unsupported: {exc}")

    assert dest.read_bytes() == b"cached thumbnail"
    if hasattr(source.stat(), "st_ino"):
        assert source.stat().st_ino == dest.stat().st_ino


def test_class_analysis_encode_crops_reports_batch_progress(monkeypatch):
    calls = []

    def fake_encode(images, *, head, batch_size_override=None, device_override=None, geometry_records=None):
        calls.append(len(images))
        return np.ones((len(images), 4), dtype=np.float32)

    monkeypatch.setattr(api, "_encode_pil_batch_for_head", fake_encode)
    job = api.ClassAnalysisJob(job_id="ca_test")
    crops = [Image.new("RGB", (8, 8), (idx, idx, idx)) for idx in range(5)]

    feats = api._class_analysis_encode_crops(
        crops,
        job=job,
        head={"encoder_type": "dinov3", "normalize_embeddings": True},
        batch_size=2,
    )

    assert feats.shape == (5, 4)
    assert calls == [2, 2, 1]
    assert job.progress == 0.70
    assert any("batch 1/3" in entry["message"] for entry in job.logs)
    assert "Encoded 5/5 crops with DINOv3" in job.message


def test_class_analysis_umap_uses_projection_neighbors(monkeypatch):
    captured = {}

    class FakeUMAP:
        def __init__(self, *, n_components, n_neighbors, min_dist, spread, metric, random_state):
            captured.update(
                {
                    "n_components": n_components,
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "spread": spread,
                    "metric": metric,
                    "random_state": random_state,
                }
            )

        def fit_transform(self, embeddings):
            return np.zeros((embeddings.shape[0], 2), dtype=np.float32)

    monkeypatch.setitem(__import__("sys").modules, "umap", types.SimpleNamespace(UMAP=FakeUMAP))
    embeddings = np.eye(80, 8, dtype=np.float32)
    warnings = []

    coords, used = api._class_analysis_project_embeddings(
        embeddings,
        projection="umap",
        projection_neighbor_k=50,
        seed=99,
        warnings=warnings,
    )

    assert used == "umap"
    assert coords.shape == (80, 2)
    assert captured["n_neighbors"] == 50
    assert captured["metric"] == "cosine"
    assert captured["spread"] == api.CLASS_ANALYSIS_DEFAULT_UMAP_SPREAD
    assert warnings == []


def test_class_analysis_size_bias_adjustment_reduces_area_axis_signal():
    records = []
    raw = []
    for idx in range(30):
        side = 10 + idx * 4
        records.append(
            {
                "point_id": f"p{idx}",
                "class_name": "light_vehicle",
                "width": side,
                "height": side,
                "crop_xyxy": [0, 0, side + 4, side + 4],
            }
        )
        area_signal = np.log1p(side * side)
        semantic_signal = 1.0 if idx % 2 else -1.0
        raw.append([area_signal, semantic_signal, semantic_signal * 0.25])
    embeddings = np.asarray(raw, dtype=np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    before = api._class_analysis_projection_diagnostics(records, embeddings[:, :2])

    adjusted, info = api._class_analysis_apply_embedding_adjustment(
        embeddings,
        records,
        mode="remove_size_bias",
    )
    after = api._class_analysis_projection_diagnostics(records, adjusted[:, :2])

    assert info["applied"] is True
    assert "log_bbox_area" in info["covariates"]
    assert abs(before["strongest_size_axis"]["correlation"]) > 0.9
    assert abs(after["strongest_size_axis"]["correlation"]) < 0.25


def test_class_analysis_projection_diagnostics_reports_quality_scores():
    rng = np.random.default_rng(17)
    records = []
    projection_coords = []
    embeddings = []
    for idx in range(20):
        vector_a = [0.0, 0.0] + list(rng.normal(scale=0.05, size=8))
        records.append(
            {
                "point_id": f"a-{idx}",
                "class_name": "class_a",
                "width": 20,
                "height": 20,
                "crop_xyxy": [0, 0, 20, 20],
            }
        )
        projection_coords.append(vector_a[:2])
        embeddings.append(vector_a)
        vector_b = [10.0, 10.0] + list(rng.normal(scale=0.05, size=8))
        records.append(
            {
                "point_id": f"b-{idx}",
                "class_name": "class_b",
                "width": 20,
                "height": 20,
                "crop_xyxy": [0, 0, 20, 20],
            }
        )
        projection_coords.append(vector_b[:2])
        embeddings.append(vector_b)
    coords = np.asarray(projection_coords, dtype=np.float32)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    diagnostics = api._class_analysis_projection_diagnostics(
        records,
        coords,
        embeddings=embeddings,
        random_seed=7,
    )
    quality = diagnostics.get("projection_quality") or {}
    assert quality["class_silhouette"] is not None
    assert 0.0 <= quality["class_silhouette"] <= 1.0
    assert quality["class_silhouette"] > 0.5
    assert quality["class_separation_ratio"] is not None
    assert quality["class_separation_ratio"] > 1.0
    assert quality["trustworthiness"] is not None
    assert 0.0 <= quality["trustworthiness"] <= 1.0
    assert quality["trustworthiness_error"] == ""
    assert quality["class_quality_error"] == ""
    assert quality["separation_score"] is not None
    assert 0.0 <= quality["separation_score"] <= 1.0
    assert quality["overview_score"] is not None
    assert 0.0 <= quality["overview_score"] <= 1.0
    assert quality["projection_recommendation"] in {"overview", "class_separation", "balanced"}


def test_class_analysis_projection_recommendation_logic():
    assert api._class_analysis_projection_recommendation(overview_score=0.82, separation_score=0.18) == "overview"
    assert api._class_analysis_projection_recommendation(overview_score=0.18, separation_score=0.82) == "class_separation"
    assert api._class_analysis_projection_recommendation(overview_score=0.60, separation_score=0.60) == "balanced"
    assert api._class_analysis_projection_recommendation(overview_score=None, separation_score=0.82) == "class_separation"
    assert api._class_analysis_projection_recommendation(overview_score=0.82, separation_score=None) == "overview"
    assert api._class_analysis_projection_recommendation(overview_score=None, separation_score=None) == "balanced"


def test_class_analysis_projection_mode_recommendation_uses_overview_quality():
    quality_by_mode = {
        "global_pca": {"overview_score": 0.21},
        "class_balanced_pca": {"overview_score": 0.88},
        "between_class_pca": {"overview_score": 0.67},
        "within_filter_pca": {"overview_score": 0.42},
    }
    assert api._class_analysis_recommend_projection_mode(
        quality_by_mode,
        fallback="global_pca",
    ) == "class_balanced_pca"


def test_class_analysis_projection_mode_recommendation_includes_umap_and_exclusions():
    quality_by_mode = {
        "global_pca": {"overview_score": 0.61},
        "class_balanced_pca": {"overview_score": 0.72},
        "within_filter_pca": {"overview_score": 0.99},
        "umap": {"overview_score": 0.91},
    }
    assert api._class_analysis_recommend_projection_mode(
        quality_by_mode,
        fallback="class_balanced_pca",
        excluded_modes={"within_filter_pca"},
    ) == "umap"


def test_class_analysis_projection_trustworthiness_handles_small_samples():
    records = [_record(f"p{idx}", f"class_{idx % 2}") for idx in range(4)]
    embeddings = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=np.float32,
    )
    diagnostics = api._class_analysis_projection_diagnostics(
        records,
        embeddings.copy(),
        embeddings=embeddings,
        random_seed=7,
    )
    quality = diagnostics["projection_quality"]
    assert quality["trustworthiness_neighbors"] == 1
    assert quality["trustworthiness"] is not None
    assert quality["trustworthiness_error"] == ""


def test_class_analysis_projection_quality_by_mode_includes_all_modes():
    records = []
    for idx in range(12):
        records.append(_record(f"r{idx}", f"class_{idx % 3}"))
    embeddings = np.asarray(
        [
            [float(idx), float(idx), float(idx) * 0.01]
            if idx % 3 == 0 else [float(idx + 1), float(idx + 2), float(idx + 0.5)]
            for idx in range(12)
        ],
        dtype=np.float32,
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    coords_by_mode = api._class_analysis_build_pca_projection_coordinates(
        records,
        embeddings,
        seed=7,
        warnings=[],
    )
    quality_by_mode = api._class_analysis_projection_quality_by_mode(
        records,
        coords_by_mode,
        embeddings,
        random_seed=7,
    )
    assert set(quality_by_mode.keys()) == set(api.CLASS_ANALYSIS_PCA_PROJECTION_MODES)
    for mode in api.CLASS_ANALYSIS_PCA_PROJECTION_MODES:
        assert "overview_score" in quality_by_mode[mode]


def test_class_analysis_projection_quality_by_mode_includes_umap_when_requested():
    records = []
    for idx in range(12):
        records.append(_record(f"r{idx}", f"class_{idx % 3}"))
    embeddings = np.asarray(
        [
            [float(idx), float(idx), float(idx) * 0.01]
            if idx % 3 == 0 else [float(idx + 1), float(idx + 2), float(idx + 0.5)]
            for idx in range(12)
        ],
        dtype=np.float32,
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    coords_by_mode = api._class_analysis_build_pca_projection_coordinates(
        records,
        embeddings,
        seed=7,
        warnings=[],
    )
    umap_coords = np.zeros_like(embeddings[:, :2], dtype=np.float32)
    for idx in range(12):
        umap_coords[idx, 0] = idx * 0.5
        umap_coords[idx, 1] = (12 - idx) * 0.25
    coords_by_mode["umap"] = umap_coords
    quality_by_mode = api._class_analysis_projection_quality_by_mode(
        records,
        coords_by_mode,
        embeddings,
        random_seed=7,
        include_umap=True,
    )
    assert set(quality_by_mode.keys()) == set(api.CLASS_ANALYSIS_PCA_PROJECTION_MODES) | {"umap"}
    for mode in api.CLASS_ANALYSIS_PCA_PROJECTION_MODES:
        assert "overview_score" in quality_by_mode[mode]
    assert "overview_score" in quality_by_mode["umap"]


def test_class_analysis_umap_includes_quality_summary_in_result(monkeypatch):
    class FakeUMAP:
        def __init__(self, *, n_components, n_neighbors, min_dist, spread, metric, random_state):
            pass

        def fit_transform(self, embeddings):
            return np.asarray(
                [[float(idx) / 10.0, float(embeddings.shape[0] - idx)] for idx in range(embeddings.shape[0])],
                dtype=np.float32,
            )

    monkeypatch.setitem(__import__("sys").modules, "umap", types.SimpleNamespace(UMAP=FakeUMAP))
    records = [
        _record("a0", "alpha"),
        _record("a1", "alpha"),
        _record("a2", "alpha"),
        _record("b0", "beta"),
        _record("b1", "beta"),
        _record("b2", "beta"),
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.98, 0.02, 0.0, 0.0],
            [0.99, -0.02, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.02, 0.98, 0.0, 0.0],
            [-0.02, 0.99, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    result = api._class_analysis_build_result(
        records,
        embeddings,
        summary={"analysis_scope": "all_classes"},
        projection="umap",
        projection_mode="class_balanced_pca",
        projection_neighbor_k=15,
        neighbor_k=3,
        seed=13,
    )

    assert result["summary"]["projection"] == "umap"
    quality_by_mode = result["summary"]["projection_quality_by_mode"]
    assert set(quality_by_mode.keys()) == set(api.CLASS_ANALYSIS_PCA_PROJECTION_MODES) | {"umap"}
    assert "overview_score" in quality_by_mode["umap"]
    assert "projection" in result["summary"] and result["summary"]["projection"] == "umap"

@pytest.mark.parametrize(
    "preprocess_mode",
    ["none", "l2", "center", "zscore"],
)
def test_class_analysis_projection_preprocess_mode_affects_pca_coordinates(preprocess_mode):
    records = [
        {"point_id": "a", "class_name": "alpha", "width": 10, "height": 10, "crop_xyxy": [0, 0, 10, 10]},
        {"point_id": "b", "class_name": "alpha", "width": 12, "height": 12, "crop_xyxy": [0, 0, 12, 12]},
        {"point_id": "c", "class_name": "beta", "width": 20, "height": 20, "crop_xyxy": [0, 0, 20, 20]},
        {"point_id": "d", "class_name": "beta", "width": 22, "height": 22, "crop_xyxy": [0, 0, 22, 22]},
    ]
    embeddings = np.array(
        [
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [22.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    base_result = api._class_analysis_build_result(
        records,
        embeddings,
        summary={"analysis_scope": "selected_class"},
        projection="pca",
        projection_mode="global_pca",
        projection_neighbor_k=2,
        projection_preprocess="none",
        projection_min_dist=0.08,
        projection_metric=api.CLASS_ANALYSIS_DEFAULT_UMAP_METRIC,
        projection_spread=1.0,
        neighbor_k=1,
        seed=13,
    )
    mode_result = base_result if preprocess_mode == "none" else api._class_analysis_build_result(
        records,
        embeddings,
        summary={"analysis_scope": "selected_class"},
        projection="pca",
        projection_mode="global_pca",
        projection_neighbor_k=2,
        projection_preprocess=preprocess_mode,
        projection_min_dist=0.08,
        projection_metric=api.CLASS_ANALYSIS_DEFAULT_UMAP_METRIC,
        projection_spread=1.0,
        neighbor_k=1,
        seed=13,
    )

    none_coords = np.asarray(
        [item["projection"] for item in base_result["points"]],
        dtype=np.float32,
    )
    mode_coords = np.asarray(
        [item["projection"] for item in mode_result["points"]],
        dtype=np.float32,
    )
    assert none_coords.shape == (4, 2)
    assert mode_coords.shape == (4, 2)
    assert mode_result["summary"]["projection_preprocess"] == preprocess_mode
    if preprocess_mode in {"none", "center"}:
        # PCA centers inputs internally, so an explicit center pass is
        # intentionally translation-equivalent for this projection.
        assert np.allclose(none_coords, mode_coords)
    else:
        assert not np.allclose(none_coords, mode_coords)


@pytest.mark.parametrize("rank_deficient", [False, True])
def test_class_analysis_streamed_small_design_lstsq_matches_numpy(
    tmp_path,
    monkeypatch,
    rank_deficient,
):
    rng = np.random.default_rng(1701 if rank_deficient else 1700)
    design = rng.normal(size=(97, 5)).astype(np.float32)
    if rank_deficient:
        design[:, 2] = design[:, 1]
        design[:, 4] = design[:, 3] * 2.0
    rhs = np.lib.format.open_memmap(
        tmp_path / f"rhs-{rank_deficient}.npy",
        mode="w+",
        dtype=np.float32,
        shape=(97, 19),
    )
    rhs[:] = rng.normal(size=rhs.shape).astype(np.float32)
    expected, _residuals, expected_rank, _singular_values = np.linalg.lstsq(
        design,
        np.asarray(rhs),
        rcond=None,
    )
    if rank_deficient:
        assert expected_rank < design.shape[1]
    else:
        assert expected_rank == design.shape[1]

    def reject_full_rhs_lstsq(*_args, **_kwargs):
        raise AssertionError("streamed solve must not pass the embedding RHS to lstsq")

    monkeypatch.setattr(api.np.linalg, "lstsq", reject_full_rhs_lstsq)
    actual = api._class_analysis_streamed_small_design_lstsq(
        design,
        rhs,
        chunk_size=7,
    )

    assert actual.dtype == np.float32
    assert np.allclose(actual, expected, rtol=2e-5, atol=2e-6)
    api._class_analysis_close_memmap_arrays(rhs)


def test_class_analysis_memmap_adjustment_streams_rank_deficient_rhs(
    tmp_path,
    monkeypatch,
):
    rng = np.random.default_rng(1702)
    records = []
    for idx in range(80):
        width = 12 + idx
        height = 9 + (idx % 17)
        # Equal bbox/crop geometry deliberately makes the area and aspect
        # covariate pairs linearly dependent, as they often are for tight crops.
        records.append(
            {
                "point_id": f"p{idx}",
                "class_name": "object",
                "bbox_xyxy": [0, 0, width, height],
                "crop_xyxy": [0, 0, width, height],
                "width": width,
                "height": height,
            }
        )
    source = np.lib.format.open_memmap(
        tmp_path / "raw.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(records), 23),
    )
    source[:] = rng.normal(size=source.shape).astype(np.float32)
    source[:] /= np.maximum(
        np.linalg.norm(source, axis=1, keepdims=True),
        1e-12,
    )
    expected, expected_info = api._class_analysis_apply_embedding_adjustment(
        np.asarray(source),
        records,
        mode="remove_size_bias",
    )

    def reject_full_rhs_lstsq(*_args, **_kwargs):
        raise AssertionError("memmap adjustment must use the streamed solver")

    monkeypatch.setattr(api.np.linalg, "lstsq", reject_full_rhs_lstsq)
    actual, actual_info = api._class_analysis_adjust_embeddings_to_memmap(
        source,
        records,
        mode="remove_size_bias",
        output_path=tmp_path / "adjusted.npy",
        chunk_size=9,
    )

    assert actual_info == expected_info
    assert np.allclose(actual, expected, rtol=3e-5, atol=3e-6)
    assert np.allclose(np.linalg.norm(actual, axis=1), 1.0, atol=2e-6)
    api._class_analysis_close_memmap_arrays(source, actual)


def test_class_analysis_canonical_preprocess_and_embedding_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", tmp_path)
    calls = []

    def fake_encode(images, *, head, batch_size_override=None, device_override=None, geometry_records=None):
        calls.append(len(images))
        return np.asarray([[idx + 1, idx + 2, idx + 3] for idx in range(len(images))], dtype=np.float32)

    monkeypatch.setattr(api, "_encode_pil_batch_for_head", fake_encode)
    crop = Image.new("RGB", (20, 10), (120, 80, 40))
    canonical = api._class_analysis_preprocess_crop(crop, mode="canonical", canonical_size=96)
    assert canonical.size == (96, 96)

    records = [
        {"point_id": "a", "crop_cache_key": "crop-a"},
        {"point_id": "b", "crop_cache_key": "crop-b"},
    ]
    head = {"encoder_type": "dinov3", "encoder_model": "test-dino", "normalize_embeddings": True}
    stats = {}
    first = api._class_analysis_encode_crops(
        [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))],
        job=api.ClassAnalysisJob(job_id="cache_a"),
        head=head,
        batch_size=8,
        records=records,
        cache_stats=stats,
    )
    assert first.shape == (2, 3)
    assert stats["hits"] == 0
    assert stats["misses"] == 2
    assert calls == [2]

    def fail_encode(*args, **kwargs):
        raise AssertionError("cached embeddings should avoid encoder calls")

    monkeypatch.setattr(api, "_encode_pil_batch_for_head", fail_encode)
    stats = {}
    second = api._class_analysis_encode_crops(
        [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))],
        job=api.ClassAnalysisJob(job_id="cache_b"),
        head=head,
        batch_size=8,
        records=records,
        cache_stats=stats,
    )
    assert np.allclose(first, second)
    assert stats["hits"] == 2
    assert stats["misses"] == 0


def test_class_analysis_embedding_cache_rejects_invalid_arrays(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", tmp_path)
    bad_shape = tmp_path / "bad_shape.npy"
    bad_nan = tmp_path / "bad_nan.npy"
    bad_dtype = tmp_path / "bad_dtype.npy"
    good = tmp_path / "good.npy"

    np.save(bad_shape, np.zeros((1, 3), dtype=np.float32))
    np.save(bad_nan, np.asarray([1.0, np.nan], dtype=np.float32))
    np.save(bad_dtype, np.asarray([1.0, 2.0], dtype=np.float64))
    np.save(good, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))

    assert api._class_analysis_load_cached_embedding(bad_shape) is None
    assert api._class_analysis_load_cached_embedding(bad_nan) is None
    assert api._class_analysis_load_cached_embedding(bad_dtype) is None
    assert not bad_shape.exists()
    assert not bad_nan.exists()
    assert not bad_dtype.exists()
    assert np.allclose(api._class_analysis_load_cached_embedding(good), [1.0, 2.0, 3.0])


def test_class_analysis_embedding_cache_preflights_huge_header_before_np_load(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", tmp_path)
    cache_path = tmp_path / "huge.npy"
    cache_path.write_bytes(
        _class_analysis_test_npy_header(
            shape=(100_000_000,),
            dtype=np.float32,
        )
    )
    load_calls = []

    def reject_np_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        raise AssertionError("np.load must follow bounded NPY preflight")

    monkeypatch.setattr(api.np, "load", reject_np_load)

    assert api._class_analysis_load_cached_embedding(cache_path) is None
    assert load_calls == []
    assert not cache_path.exists()


def test_class_analysis_embedding_cache_does_not_delete_concurrent_replacement(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", tmp_path)
    cache_path = tmp_path / "embedding.npy"
    replacement_path = tmp_path / "replacement.npy"
    np.save(cache_path, np.asarray([1.0, 2.0], dtype=np.float32))
    np.save(replacement_path, np.asarray([7.0, 8.0], dtype=np.float32))
    original_preflight = api._class_analysis_preflight_npy_header

    def replace_then_fail(handle, *, max_array_bytes):
        original_preflight(handle, max_array_bytes=max_array_bytes)
        os.replace(replacement_path, cache_path)
        raise ValueError("simulated stale cache read")

    monkeypatch.setattr(
        api,
        "_class_analysis_preflight_npy_header",
        replace_then_fail,
    )

    assert api._class_analysis_load_cached_embedding(cache_path) is None
    assert cache_path.exists()
    assert np.array_equal(
        np.load(cache_path, allow_pickle=False),
        np.asarray([7.0, 8.0], dtype=np.float32),
    )


def test_class_analysis_embedding_cache_rejects_symlink_escape(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    outside = tmp_path / "outside.npy"
    np.save(outside, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    cache_link = cache_root / "linked.npy"
    try:
        cache_link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)

    assert api._class_analysis_load_cached_embedding(cache_link) is None


def test_class_analysis_embedding_cache_rejects_symlinked_cache_root(
    tmp_path, monkeypatch
):
    outside_cache = tmp_path / "outside_cache"
    outside_cache.mkdir()
    cache_link = tmp_path / "cache_link"
    try:
        cache_link.symlink_to(outside_cache, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    cache_file = cache_link / "good.npy"
    np.save(outside_cache / "good.npy", np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_link)

    assert api._class_analysis_load_cached_embedding(cache_file) is None


def test_class_analysis_corrupt_cache_rematerializes_real_crop(monkeypatch, tmp_path):
    class_root = tmp_path / "class_analysis"
    workspace = class_root / "workspace"
    images_dir = workspace / "images"
    images_dir.mkdir(parents=True)
    image_path = images_dir / "sample.jpg"
    Image.new("RGB", (80, 60), (20, 40, 60)).save(image_path)
    manifest_path = workspace / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_label": "browser snapshot",
                "labelmap": ["car"],
                "images": [
                    {
                        "split": "train",
                        "image_relpath": "sample.jpg",
                        "frontend_image_key": "train/original/sample.jpg",
                        "label_lines": ["0 0.5 0.5 0.25 0.25"],
                    }
                ],
                "yolo_layout": "flat",
            }
        ),
        encoding="utf-8",
    )
    corrupt_embedding = tmp_path / "corrupt.npy"
    cached_thumb = tmp_path / "cached_thumb.jpg"
    np.save(corrupt_embedding, np.zeros((1, 3), dtype=np.float32))
    Image.new("RGB", (8, 8), (1, 2, 3)).save(cached_thumb)

    monkeypatch.setattr(api, "_class_analysis_embedding_cache_path", lambda _cache_key: corrupt_embedding)
    monkeypatch.setattr(api, "_class_analysis_thumbnail_cache_path", lambda _crop_cache_key: cached_thumb)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)

    job = api.ClassAnalysisJob(job_id="ca_corrupt_cache")
    records, crops, summary = api._class_analysis_collect_records(
        {
            "source_mode": "active_workspace",
            "workspace_id": "ca_test",
            "workspace_dir": str(workspace),
            "workspace_manifest_path": str(manifest_path),
            "analysis_scope": "selected_class",
            "class_name": "car",
            "preprocess_mode": "canonical",
            "canonical_size": 64,
            "crop_mode": "padded_square",
            "padding_ratio": 0.08,
            "background_mode": "full_crop",
            "embedding_view_mode": "single",
            "encoder_type": "dinov3",
            "encoder_model": "test-dino",
            "dinov3_pooling": "pooler",
        },
        job=job,
        out_dir=tmp_path / "out",
    )

    try:
        assert len(records) == 1
        assert len(crops) == 1
        assert summary["object_count"] == 1
        assert records[0]["crop_cache_reused"] is False
        assert records[0]["embedding_views"]
        assert re.fullmatch(r"cro_[0-9a-f]{64}", records[0]["review_object_key"])
        assert crops[0].size == (64, 64)
    finally:
        for crop in crops:
            api._close_crop_item(crop)


@pytest.mark.parametrize("materialize_crops", [False, True])
def test_class_analysis_refinement_preserves_private_source_dimensions_until_stage2(
    materialize_crops,
    monkeypatch,
    tmp_path,
):
    class_root = tmp_path / "class_analysis"
    workspace = class_root / "workspace"
    images_dir = workspace / "images"
    images_dir.mkdir(parents=True)
    image_path = images_dir / "sample.jpg"
    Image.new("RGB", (1000, 1000), (20, 40, 60)).save(image_path)
    manifest_path = workspace / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_label": "source dimensions",
                "labelmap": ["car"],
                "images": [
                    {
                        "split": "train",
                        "image_relpath": "sample.jpg",
                        "label_lines": ["0 0.5 0.5 0.01 0.02"],
                    }
                ],
                "yolo_layout": "flat",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    job = api.ClassAnalysisJob(job_id=f"ca_dimensions_{materialize_crops}")

    records, crops, summary = api._class_analysis_collect_records(
        {
            "source_mode": "active_workspace",
            "workspace_id": "ca_dimensions",
            "workspace_dir": str(workspace),
            "workspace_manifest_path": str(manifest_path),
            "analysis_scope": "all_classes",
            "refine_outliers": True,
            "preprocess_mode": "canonical",
            "canonical_size": 64,
            "crop_mode": "padded_square",
            "padding_ratio": 0.08,
            "background_mode": "full_crop",
            "embedding_view_mode": "single",
            "encoder_type": "dinov3",
            "encoder_model": "test-dino",
            "dinov3_pooling": "pooler",
        },
        job=job,
        out_dir=tmp_path / f"out-{materialize_crops}",
        materialize_crops=materialize_crops,
    )

    try:
        assert len(records) == 1
        assert records[0]["width"] == 10
        assert records[0]["height"] == 20
        assert records[0]["_image_width"] == 1000
        assert records[0]["_image_height"] == 1000
        spatial = summary["_spatial_context_records"]
        assert spatial[0]["_image_width"] == 1000
        assert spatial[0]["_image_height"] == 1000
    finally:
        for crop in crops:
            api._close_crop_item(crop)


def test_class_analysis_thumbnail_cache_replaces_symlink_without_target_write(
    monkeypatch, tmp_path
):
    class_root = tmp_path / "class_analysis"
    workspace = class_root / "workspace"
    images_dir = workspace / "images"
    images_dir.mkdir(parents=True)
    image_path = images_dir / "sample.jpg"
    Image.new("RGB", (80, 60), (20, 40, 60)).save(image_path)
    manifest_path = workspace / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_label": "browser snapshot",
                "labelmap": ["car"],
                "images": [
                    {
                        "split": "train",
                        "image_relpath": "sample.jpg",
                        "frontend_image_key": "train/original/sample.jpg",
                        "label_lines": ["0 0.5 0.5 0.25 0.25"],
                    }
                ],
                "yolo_layout": "flat",
            }
        ),
        encoding="utf-8",
    )
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    outside = tmp_path / "outside_thumb.jpg"
    outside.write_bytes(b"external")
    cached_thumb = cache_root / "cached_thumb.jpg"
    try:
        cached_thumb.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    monkeypatch.setattr(api, "_class_analysis_thumbnail_cache_path", lambda _crop_cache_key: cached_thumb)
    monkeypatch.setattr(api, "_class_analysis_cached_embedding_valid", lambda *_args, **_kwargs: True)

    job = api.ClassAnalysisJob(job_id="ca_thumb_cache_symlink")
    records, crops, summary = api._class_analysis_collect_records(
        {
            "source_mode": "active_workspace",
            "workspace_id": "ca_test",
            "workspace_dir": str(workspace),
            "workspace_manifest_path": str(manifest_path),
            "analysis_scope": "selected_class",
            "class_name": "car",
            "preprocess_mode": "canonical",
            "canonical_size": 64,
            "crop_mode": "padded_square",
            "padding_ratio": 0.08,
            "background_mode": "full_crop",
            "embedding_view_mode": "single",
            "encoder_type": "dinov3",
            "encoder_model": "test-dino",
            "dinov3_pooling": "pooler",
        },
        job=job,
        out_dir=tmp_path / "out",
    )

    try:
        assert summary["object_count"] == 1
        assert records[0]["crop_cache_reused"] is False
        # Thumbnails are now generated only when the thumbnail endpoint is
        # requested, so collection must not touch a legacy cache leaf.
        assert cached_thumb.is_symlink()
        assert outside.read_bytes() == b"external"
    finally:
        for crop in crops:
            api._close_crop_item(crop)


def test_class_analysis_thumbnail_cache_ignores_symlinked_cache_root(
    monkeypatch, tmp_path
):
    class_root = tmp_path / "class_analysis"
    workspace = class_root / "workspace"
    images_dir = workspace / "images"
    images_dir.mkdir(parents=True)
    image_path = images_dir / "sample.jpg"
    Image.new("RGB", (80, 60), (20, 40, 60)).save(image_path)
    manifest_path = workspace / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_label": "browser snapshot",
                "labelmap": ["car"],
                "images": [
                    {
                        "split": "train",
                        "image_relpath": "sample.jpg",
                        "frontend_image_key": "train/original/sample.jpg",
                        "label_lines": ["0 0.5 0.5 0.25 0.25"],
                    }
                ],
                "yolo_layout": "flat",
            }
        ),
        encoding="utf-8",
    )
    outside_cache = tmp_path / "outside_cache"
    outside_cache.mkdir()
    cache_link = tmp_path / "cache_link"
    try:
        cache_link.symlink_to(outside_cache, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    cached_thumb = cache_link / "cached_thumb.jpg"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(outside_cache / "cached_thumb.jpg")
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_link)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    monkeypatch.setattr(api, "_class_analysis_thumbnail_cache_path", lambda _crop_cache_key: cached_thumb)
    monkeypatch.setattr(api, "_class_analysis_cached_embedding_valid", lambda *_args, **_kwargs: True)

    job = api.ClassAnalysisJob(job_id="ca_thumb_cache_root_symlink")
    records, crops, summary = api._class_analysis_collect_records(
        {
            "source_mode": "active_workspace",
            "workspace_id": "ca_test",
            "workspace_dir": str(workspace),
            "workspace_manifest_path": str(manifest_path),
            "analysis_scope": "selected_class",
            "class_name": "car",
            "preprocess_mode": "canonical",
            "canonical_size": 64,
            "crop_mode": "padded_square",
            "padding_ratio": 0.08,
            "background_mode": "full_crop",
            "embedding_view_mode": "single",
            "encoder_type": "dinov3",
            "encoder_model": "test-dino",
            "dinov3_pooling": "pooler",
        },
        job=job,
        out_dir=tmp_path / "out",
    )

    try:
        assert summary["object_count"] == 1
        assert records[0]["crop_cache_reused"] is False
        assert (outside_cache / "cached_thumb.jpg").exists()
    finally:
        for crop in crops:
            api._close_crop_item(crop)


def test_class_analysis_cache_validation_uses_cradio_recipe(monkeypatch, tmp_path):
    class_root = tmp_path / "class_analysis"
    workspace = class_root / "workspace"
    images_dir = workspace / "images"
    images_dir.mkdir(parents=True)
    image_path = images_dir / "sample.jpg"
    Image.new("RGB", (80, 60), (20, 40, 60)).save(image_path)
    manifest_path = workspace / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_label": "browser snapshot",
                "labelmap": ["car"],
                "images": [
                    {
                        "split": "train",
                        "image_relpath": "sample.jpg",
                        "frontend_image_key": "train/original/sample.jpg",
                        "label_lines": ["0 0.5 0.5 0.25 0.25"],
                    }
                ],
                "yolo_layout": "flat",
            }
        ),
        encoding="utf-8",
    )
    cached_thumb = tmp_path / "cached_thumb.jpg"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(cached_thumb)
    captured_heads = []

    monkeypatch.setattr(api, "_class_analysis_thumbnail_cache_path", lambda _crop_cache_key: cached_thumb)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)

    def fake_load_image_pack(_image_pack_key, head, _object_keys):
        captured_heads.append(dict(head))
        return None

    monkeypatch.setattr(api, "_class_analysis_load_image_pack", fake_load_image_pack)

    job = api.ClassAnalysisJob(job_id="ca_cradio_cache_recipe")
    records, crops, summary = api._class_analysis_collect_records(
        {
            "source_mode": "active_workspace",
            "workspace_id": "ca_test",
            "workspace_dir": str(workspace),
            "workspace_manifest_path": str(manifest_path),
            "analysis_scope": "selected_class",
            "class_name": "car",
            "preprocess_mode": "canonical",
            "canonical_size": 64,
            "crop_mode": "padded_square",
            "padding_ratio": 0.08,
            "background_mode": "full_crop",
            "embedding_view_mode": "single",
            "encoder_type": "cradio",
            "encoder_model": CRADIO_DEFAULT_MODEL,
            "cradio_pooling": "spatial_mean",
        },
        job=job,
        out_dir=tmp_path / "out",
    )

    try:
        assert len(records) == 1
        assert summary["object_count"] == 1
        assert captured_heads
        assert any(head["encoder_type"] == "cradio" for head in captured_heads)
        assert any(head["encoder_model"] == CRADIO_DEFAULT_MODEL for head in captured_heads)
        assert any(head["cradio_pooling"] == "spatial_mean" for head in captured_heads)
    finally:
        for crop in crops:
            api._close_crop_item(crop)


def test_class_analysis_multiview_embedding_composes_before_postprocess(monkeypatch):
    captured = {}

    def fake_encode(images, *, head, batch_size_override=None, device_override=None, geometry_records=None):
        captured["image_count"] = len(images)
        captured["geometry_records"] = geometry_records
        return np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)

    monkeypatch.setattr(api, "_encode_pil_batch_for_head", fake_encode)
    head = {"encoder_type": "dinov3", "normalize_embeddings": True}
    feats = api._encode_embedding_items_for_head(
        [(Image.new("RGB", (8, 8)), Image.new("RGB", (12, 12)))],
        head=head,
    )

    assert captured["image_count"] == 2
    assert captured["geometry_records"] is None
    assert feats.shape == (1, 4)
    assert np.allclose(np.linalg.norm(feats, axis=1), 1.0)


def test_classifier_crop_for_head_uses_saved_embedding_recipe(monkeypatch):
    captured = {}
    head = {
        "encoder_type": "dinov3",
        "normalize_embeddings": True,
        "preprocess_mode": "canonical",
        "canonical_size": 80,
        "embedding_crop_mode": "padded_square",
        "embedding_crop_padding_ratio": 0.5,
        "background_mode": "darken_outside_box",
        "embedding_view_mode": "single",
        "embedding_adjustment_transform": {"mode": "remove_size_bias"},
    }

    def fake_encode(images, *, head, batch_size_override=None, device_override=None, geometry_records=None):
        crop_pixels = np.asarray(images[0], dtype=np.float32)
        captured["image_size"] = images[0].size
        captured["outside_mean"] = float(crop_pixels[2, 2].mean())
        captured["inside_mean"] = float(crop_pixels[40, 40].mean())
        captured["geometry"] = geometry_records[0]
        captured["head"] = head
        return np.ones((1, 4), dtype=np.float32)

    monkeypatch.setattr(api, "_active_classifier_head_for_inference", lambda: head)
    monkeypatch.setattr(api, "_encode_pil_batch_for_head", fake_encode)

    image = Image.new("RGB", (100, 60), (20, 40, 60))
    feats = api._encode_classifier_xyxy_for_active(image, [40, 20, 60, 30])

    assert feats.shape == (1, 4)
    assert captured["image_size"] == (80, 80)
    assert captured["outside_mean"] < captured["inside_mean"]
    assert captured["geometry"]["bbox_xyxy"] == [40.0, 20.0, 60.0, 30.0]
    assert captured["geometry"]["crop_xyxy"] == [30, 5, 70, 45]
    assert captured["geometry"]["background_mode"] == "darken_outside_box"
    assert captured["geometry"]["embedding_view_mode"] == "single"
    assert captured["head"]["embedding_adjustment_transform"]["mode"] == "remove_size_bias"


def test_classifier_multiview_inference_composes_views_before_size_bias(monkeypatch):
    captured = {}
    transform = {"mode": "remove_size_bias", "sentinel": True}
    head = {
        "encoder_type": "dinov3",
        "normalize_embeddings": False,
        "preprocess_mode": "canonical",
        "canonical_size": 48,
        "embedding_crop_mode": "padded_square",
        "embedding_crop_padding_ratio": 0.08,
        "background_mode": "full_crop",
        "embedding_view_mode": "tight_context",
        "embedding_adjustment_transform": transform,
    }

    def fake_encode(images, *, head, batch_size_override=None, device_override=None, geometry_records=None):
        captured["raw_image_sizes"] = [image.size for image in images]
        captured["raw_geometry_records"] = geometry_records
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    def fake_residualizer(embeddings, covariates, residualizer, *, normalize=True):
        captured["residualizer_embedding_shape"] = embeddings.shape
        captured["residualizer_covariate_shape"] = covariates.shape
        captured["residualizer_transform"] = residualizer
        captured["residualizer_normalize"] = normalize
        return np.asarray(embeddings, dtype=np.float32) + 10.0

    monkeypatch.setattr(api, "_active_classifier_head_for_inference", lambda: head)
    monkeypatch.setattr(api, "_encode_pil_batch_for_head", fake_encode)
    monkeypatch.setattr(api, "apply_size_bias_residualizer", fake_residualizer)

    image = Image.new("RGB", (96, 72), (30, 60, 90))
    feats = api._encode_classifier_xyxy_for_active(image, [20, 18, 36, 34])

    assert feats.shape == (1, 4)
    assert captured["raw_image_sizes"] == [(64, 64), (64, 64)]
    assert captured["raw_geometry_records"] is None
    assert captured["residualizer_embedding_shape"] == (1, 4)
    assert captured["residualizer_covariate_shape"] == (1, 4)
    assert captured["residualizer_transform"] == transform
    assert captured["residualizer_normalize"] is True
    assert np.all(feats > 9.0)


def test_classifier_detection_scoring_closes_preprocessed_crops(monkeypatch):
    crop = Image.new("RGB", (16, 16), (10, 20, 30))
    original_close = crop.close
    closed = []

    def tracked_close():
        closed.append(True)
        original_close()

    crop.close = tracked_close

    def fake_crop_for_head(pil_img, xyxy, head):
        return crop, {
            "bbox_xyxy": [float(v) for v in xyxy],
            "crop_xyxy": [0, 0, 16, 16],
            "width": 10,
            "height": 10,
        }

    def fake_encode(crops, *, head, batch_size_override=None, device_override=None, geometry_records=None):
        assert crops == [crop]
        assert geometry_records and geometry_records[0]["crop_xyxy"] == [0, 0, 16, 16]
        return np.ones((1, 2), dtype=np.float32)

    monkeypatch.setattr(api, "_classifier_crop_for_head", fake_crop_for_head)
    monkeypatch.setattr(api, "_encode_pil_batch_for_head", fake_encode)
    monkeypatch.setattr(
        api,
        "_clip_head_predict_proba",
        lambda feats, head, empty_cache_fn=None: np.asarray([[0.2, 0.8]], dtype=np.float32),
    )

    image = Image.new("RGB", (100, 60), (20, 40, 60))
    detection = {"label": "boat", "bbox_xyxy_px": [1, 2, 11, 12]}
    scores = api._score_detections_with_clip_head(
        [detection],
        pil_img=image,
        clip_head={"classes": np.asarray(["car", "boat"], dtype=object)},
        score_mode="clip_head_prob",
    )

    assert set(scores) == {id(detection)}
    assert np.isclose(scores[id(detection)], 0.8)
    assert closed == [True]


def test_classifier_loader_preserves_embedding_recipe_metadata(tmp_path):
    class DummyClassifier:
        classes_ = np.asarray(["car", "boat"])
        coef_ = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        intercept_ = np.asarray([0.0], dtype=np.float32)

    classifier_path = tmp_path / "test_classifier.pkl"
    meta_path = tmp_path / "test_classifier.meta.pkl"
    classifier_path.write_bytes(b"classifier")
    meta_path.write_bytes(b"meta")
    transform = {
        "mode": "remove_size_bias",
        "keep_mask": [True, True, False, False],
        "mean": [1.0, 2.0],
        "std": [0.5, 0.25],
        "beta": [[0.0, 0.0, 0.0, 0.0]] * 3,
    }

    def fake_joblib_load(path):
        if path.endswith(".meta.pkl"):
            return {
                "encoder_type": "dinov3",
                "encoder_model": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "mlp_normalize_embeddings": True,
                "preprocess_mode": "canonical",
                "canonical_size": 336,
                "embedding_crop_mode": "padded_square",
                "embedding_crop_padding_ratio": 0.08,
                "background_mode": "blur_outside_box",
                "embedding_view_mode": "tight_context",
                "embedding_adjustment": "remove_size_bias",
                "embedding_adjustment_transform": transform,
                "dinov3_pooling": "pooler",
                "cradio_pooling": "summary_spatial_concat",
                "embedding_aggregation": "local_salad",
                "embedding_salad_head_id": "unit_head",
            }
        return DummyClassifier()

    class HttpError(Exception):
        def __init__(self, *, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    head = _load_clip_head_from_classifier_impl(
        classifier_path,
        joblib_load_fn=fake_joblib_load,
        http_exception_cls=HttpError,
        clip_head_background_indices_fn=lambda classes: [],
        resolve_head_normalize_embeddings_fn=lambda clf, default: default,
        infer_clip_model_fn=lambda dim, default: default,
        active_clip_model_name=None,
        default_clip_model="ViT-B/32",
        logger=type("Logger", (), {"warning": lambda *args, **kwargs: None})(),
    )

    assert head["encoder_type"] == "dinov3"
    assert head["preprocess_mode"] == "canonical"
    assert head["canonical_size"] == 336
    assert head["embedding_crop_mode"] == "padded_square"
    assert head["embedding_crop_padding_ratio"] == 0.08
    assert head["background_mode"] == "blur_outside_box"
    assert head["embedding_view_mode"] == "tight_context"
    assert head["embedding_adjustment"] == "remove_size_bias"
    assert head["embedding_adjustment_transform"] == transform
    assert head["dinov3_pooling"] == "pooler"
    assert head["cradio_pooling"] == "summary_spatial_concat"
    assert head["embedding_aggregation"] == "local_salad"
    assert head["embedding_salad_head_id"] == "unit_head"


def test_classifier_loader_preserves_mlp_gelu_activation(tmp_path):
    classifier_path = tmp_path / "gelu_head.pkl"
    meta_path = tmp_path / "gelu_head.meta.pkl"
    classifier_path.write_bytes(b"classifier")
    meta_path.write_bytes(b"meta")

    clf_obj = {
        "classifier_type": "mlp",
        "classes": np.asarray(["car", "boat"], dtype=object),
        "embedding_dim": 2,
        "layers": [
            {
                "weight": np.eye(2, dtype=np.float32),
                "bias": np.zeros(2, dtype=np.float32),
                "activation": "gelu",
            },
            {
                "weight": np.eye(2, dtype=np.float32),
                "bias": np.zeros(2, dtype=np.float32),
                "activation": "linear",
            },
        ],
    }

    def fake_joblib_load(path):
        if path.endswith(".meta.pkl"):
            return {
                "encoder_type": "dinov3",
                "encoder_model": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "mlp_normalize_embeddings": True,
            }
        return clf_obj

    class HttpError(Exception):
        def __init__(self, *, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    head = _load_clip_head_from_classifier_impl(
        classifier_path,
        joblib_load_fn=fake_joblib_load,
        http_exception_cls=HttpError,
        clip_head_background_indices_fn=lambda classes: [],
        resolve_head_normalize_embeddings_fn=lambda clf, default: default,
        infer_clip_model_fn=lambda dim, default: default,
        active_clip_model_name=None,
        default_clip_model="ViT-B/32",
        logger=type("Logger", (), {"warning": lambda *args, **kwargs: None})(),
    )

    assert head["layers"][0]["activation"] == "gelu"
    assert head["classes"] == ["car", "boat"]


def test_classifier_loader_rejects_mlp_layer_width_mismatch(tmp_path):
    classifier_path = tmp_path / "bad_head.pkl"
    meta_path = tmp_path / "bad_head.meta.pkl"
    classifier_path.write_bytes(b"classifier")
    meta_path.write_bytes(b"meta")

    clf_obj = {
        "classifier_type": "mlp",
        "classes": np.asarray(["car", "boat"], dtype=object),
        "embedding_dim": 2,
        "layers": [
            {
                "weight": np.eye(2, dtype=np.float32),
                "bias": np.zeros(2, dtype=np.float32),
                "activation": "gelu",
            },
            {
                "weight": np.zeros((2, 3), dtype=np.float32),
                "bias": np.zeros(2, dtype=np.float32),
                "activation": "linear",
            },
        ],
    }

    def fake_joblib_load(path):
        if path.endswith(".meta.pkl"):
            return {
                "encoder_type": "dinov3",
                "encoder_model": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "mlp_normalize_embeddings": True,
            }
        return clf_obj

    class HttpError(Exception):
        def __init__(self, *, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    with pytest.raises(HttpError) as exc:
        _load_clip_head_from_classifier_impl(
            classifier_path,
            joblib_load_fn=fake_joblib_load,
            http_exception_cls=HttpError,
            clip_head_background_indices_fn=lambda classes: [],
            resolve_head_normalize_embeddings_fn=lambda clf, default: default,
            infer_clip_model_fn=lambda dim, default: default,
            active_clip_model_name=None,
            default_clip_model="ViT-B/32",
            logger=type("Logger", (), {"warning": lambda *args, **kwargs: None})(),
        )

    assert exc.value.detail == "agent_clip_classifier_invalid_shape"


def test_classifier_loader_rejects_mlp_output_width_mismatch(tmp_path):
    classifier_path = tmp_path / "bad_output_head.pkl"
    meta_path = tmp_path / "bad_output_head.meta.pkl"
    classifier_path.write_bytes(b"classifier")
    meta_path.write_bytes(b"meta")

    clf_obj = {
        "classifier_type": "mlp",
        "classes": np.asarray(["car", "boat"], dtype=object),
        "embedding_dim": 2,
        "layers": [
            {
                "weight": np.zeros((3, 2), dtype=np.float32),
                "bias": np.zeros(3, dtype=np.float32),
                "activation": "linear",
            }
        ],
    }

    def fake_joblib_load(path):
        if path.endswith(".meta.pkl"):
            return {
                "encoder_type": "clip",
                "encoder_model": "ViT-B/32",
                "mlp_normalize_embeddings": True,
            }
        return clf_obj

    class HttpError(Exception):
        def __init__(self, *, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    with pytest.raises(HttpError) as exc:
        _load_clip_head_from_classifier_impl(
            classifier_path,
            joblib_load_fn=fake_joblib_load,
            http_exception_cls=HttpError,
            clip_head_background_indices_fn=lambda classes: [],
            resolve_head_normalize_embeddings_fn=lambda clf, default: default,
            infer_clip_model_fn=lambda dim, default: default,
            active_clip_model_name=None,
            default_clip_model="ViT-B/32",
            logger=type("Logger", (), {"warning": lambda *args, **kwargs: None})(),
        )

    assert exc.value.detail == "agent_clip_classifier_invalid_shape"


def test_classifier_loader_rejects_mlp_layer_norm_width_mismatch(tmp_path):
    classifier_path = tmp_path / "bad_layer_norm_head.pkl"
    meta_path = tmp_path / "bad_layer_norm_head.meta.pkl"
    classifier_path.write_bytes(b"classifier")
    meta_path.write_bytes(b"meta")

    clf_obj = {
        "classifier_type": "mlp",
        "classes": np.asarray(["car", "boat"], dtype=object),
        "embedding_dim": 2,
        "layers": [
            {
                "weight": np.eye(2, dtype=np.float32),
                "bias": np.zeros(2, dtype=np.float32),
                "layer_norm_weight": np.ones(3, dtype=np.float32),
                "activation": "linear",
            }
        ],
    }

    def fake_joblib_load(path):
        if path.endswith(".meta.pkl"):
            return {
                "encoder_type": "clip",
                "encoder_model": "ViT-B/32",
                "mlp_normalize_embeddings": True,
            }
        return clf_obj

    class HttpError(Exception):
        def __init__(self, *, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    with pytest.raises(HttpError) as exc:
        _load_clip_head_from_classifier_impl(
            classifier_path,
            joblib_load_fn=fake_joblib_load,
            http_exception_cls=HttpError,
            clip_head_background_indices_fn=lambda classes: [],
            resolve_head_normalize_embeddings_fn=lambda clf, default: default,
            infer_clip_model_fn=lambda dim, default: default,
            active_clip_model_name=None,
            default_clip_model="ViT-B/32",
            logger=type("Logger", (), {"warning": lambda *args, **kwargs: None})(),
        )

    assert exc.value.detail == "agent_clip_classifier_invalid_shape"


def test_clip_head_predict_proba_replays_mlp_gelu_activation():
    head = {
        "classifier_type": "mlp",
        "classes": ["car", "boat"],
        "proba_mode": "softmax",
        "layers": [
            {
                "weight": np.eye(2, dtype=np.float32),
                "bias": np.zeros(2, dtype=np.float32),
                "activation": "gelu",
            },
            {
                "weight": np.asarray([[1.0, -0.5], [-0.75, 0.25]], dtype=np.float32),
                "bias": np.asarray([0.1, -0.2], dtype=np.float32),
                "activation": "linear",
            },
        ],
    }
    feats = np.asarray([[-1.0, 2.0]], dtype=np.float32)

    hidden = 0.5 * feats * (1.0 + np.vectorize(math.erf)(feats / math.sqrt(2.0)))
    logits = hidden @ head["layers"][1]["weight"].T + head["layers"][1]["bias"]
    logits = logits - np.max(logits, axis=1, keepdims=True)
    expected = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    actual = api._clip_head_predict_proba(feats, head)

    assert np.allclose(actual, expected.astype(np.float32), atol=1e-6)


def test_clip_head_predict_proba_replays_mlp_arcface_output_layer():
    head = {
        "classifier_type": "mlp",
        "classes": ["car", "boat"],
        "proba_mode": "softmax",
        "arcface": True,
        "arcface_scale": 10.0,
        "layers": [
            {
                "weight": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                "bias": np.zeros(2, dtype=np.float32),
                "activation": "linear",
            },
        ],
    }
    feats = np.asarray([[3.0, 4.0]], dtype=np.float32)
    feats_norm = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    weight = head["layers"][0]["weight"]
    weight_norm = weight / np.linalg.norm(weight, axis=1, keepdims=True)
    logits = (feats_norm @ weight_norm.T) * 10.0
    logits = logits - np.max(logits, axis=1, keepdims=True)
    expected = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    actual = api._clip_head_predict_proba(feats, head)

    assert np.allclose(actual, expected.astype(np.float32), atol=1e-6)


def test_clip_head_predict_proba_normalizes_ovr_probabilities():
    feats = np.asarray([[1.0, 0.0]], dtype=np.float32)
    head = {
        "classifier_type": "logreg",
        "coef": np.asarray([[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
        "intercept": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "proba_mode": "ovr",
    }

    actual = api._clip_head_predict_proba(feats, head)
    raw = 1.0 / (1.0 + np.exp(-np.asarray([[1.0, 0.0, -1.0]], dtype=np.float32)))
    expected = raw / raw.sum(axis=1, keepdims=True)

    assert np.allclose(actual, expected, atol=1e-6)
    assert np.allclose(actual.sum(axis=1), [1.0])


def test_clip_head_predict_proba_applies_binary_logit_adjustment():
    feats = np.asarray([[0.0, 0.0]], dtype=np.float32)
    head = {
        "classifier_type": "logreg",
        "classes": ["negative", "positive"],
        "coef": np.asarray([[0.0, 0.0]], dtype=np.float32),
        "intercept": np.asarray([0.0], dtype=np.float32),
        "proba_mode": "binary",
        "logit_adjustment_inference": True,
        "logit_adjustment": [0.0, 2.0],
    }

    actual = api._clip_head_predict_proba(feats, head)
    expected_pos = 1.0 / (1.0 + np.exp(-2.0))

    assert actual is not None
    assert actual.shape == (1, 2)
    assert np.allclose(actual[0], [1.0 - expected_pos, expected_pos], atol=1e-6)


def test_clip_head_predict_proba_temperatures_adjusted_binary_logits():
    feats = np.asarray([[0.0, 0.0]], dtype=np.float32)
    head = {
        "classifier_type": "logreg",
        "classes": ["negative", "positive"],
        "coef": np.asarray([[0.0, 0.0]], dtype=np.float32),
        "intercept": np.asarray([0.0], dtype=np.float32),
        "proba_mode": "binary",
        "temperature": 2.0,
        "logit_adjustment_inference": True,
        "logit_adjustment": [0.0, 2.0],
    }

    actual = api._clip_head_predict_proba(feats, head)
    expected_pos = 1.0 / (1.0 + np.exp(-1.0))

    assert actual is not None
    assert actual.shape == (1, 2)
    assert np.allclose(actual[0], [1.0 - expected_pos, expected_pos], atol=1e-6)


def test_clip_head_predict_proba_accepts_numpy_array_classes():
    feats = np.asarray([[1.0, 0.0]], dtype=np.float32)
    head = {
        "classifier_type": "logreg",
        "classes": np.asarray(["car", "boat"], dtype=object),
        "coef": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "intercept": np.zeros(2, dtype=np.float32),
        "proba_mode": "softmax",
    }

    actual = api._clip_head_predict_proba(feats, head)

    assert actual is not None
    assert actual.shape == (1, 2)
    assert float(actual[0, 0]) > float(actual[0, 1])


def test_clip_auto_predict_details_accepts_numpy_array_classes(monkeypatch):
    head = {
        "classifier_type": "logreg",
        "classes": np.asarray(["car", "boat"], dtype=object),
        "coef": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "intercept": np.zeros(2, dtype=np.float32),
        "proba_mode": "softmax",
    }
    monkeypatch.setattr(api, "_active_classifier_head_for_inference", lambda: head)

    details = api._clip_auto_predict_details(
        np.asarray([[2.0, 0.0]], dtype=np.float32),
        background_guard=False,
    )

    assert details["error"] is None
    assert details["label"] == "car"
    assert details["second_label"] == "boat"


def test_clip_head_predict_proba_fails_closed_on_embedding_width_mismatch():
    feats = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    logreg_head = {
        "classifier_type": "logreg",
        "coef": np.zeros((2, 2), dtype=np.float32),
        "intercept": np.zeros(2, dtype=np.float32),
        "proba_mode": "softmax",
    }
    mlp_head = {
        "classifier_type": "mlp",
        "layers": [
            {
                "weight": np.zeros((2, 2), dtype=np.float32),
                "bias": np.zeros(2, dtype=np.float32),
                "activation": "linear",
            }
        ],
    }

    assert api._clip_head_predict_proba(feats, logreg_head) is None
    assert api._clip_head_predict_proba(feats, mlp_head) is None


def test_clip_head_predict_proba_fails_closed_on_class_count_mismatch():
    feats = np.asarray([[1.0, 2.0]], dtype=np.float32)
    head = {
        "classifier_type": "logreg",
        "classes": ["car", "boat", "plane"],
        "coef": np.zeros((2, 2), dtype=np.float32),
        "intercept": np.zeros(2, dtype=np.float32),
        "proba_mode": "softmax",
    }

    assert api._clip_head_predict_proba(feats, head) is None


def test_clip_head_predict_proba_fails_closed_on_layer_norm_shape_mismatch():
    feats = np.asarray([[1.0, 2.0]], dtype=np.float32)
    head = {
        "classifier_type": "mlp",
        "classes": ["car", "boat"],
        "layers": [
            {
                "weight": np.eye(2, dtype=np.float32),
                "bias": np.zeros(2, dtype=np.float32),
                "layer_norm_weight": np.ones(3, dtype=np.float32),
                "activation": "linear",
            }
        ],
    }

    assert api._clip_head_predict_proba(feats, head) is None


def test_clip_head_predict_proba_fails_closed_on_malformed_arrays():
    feats = np.asarray([[1.0, 2.0]], dtype=np.float32)
    bad_logreg = {
        "classifier_type": "logreg",
        "coef": [[object(), 0.0]],
        "intercept": [0.0],
        "proba_mode": "softmax",
    }
    bad_mlp_weight = {
        "classifier_type": "mlp",
        "classes": ["car", "boat"],
        "layers": [
            {
                "weight": [[object(), 0.0], [0.0, 1.0]],
                "bias": np.zeros(2, dtype=np.float32),
                "activation": "linear",
            }
        ],
    }
    bad_layer_norm = {
        "classifier_type": "mlp",
        "classes": ["car", "boat"],
        "layers": [
            {
                "weight": np.eye(2, dtype=np.float32),
                "bias": np.zeros(2, dtype=np.float32),
                "layer_norm_weight": [object(), 1.0],
                "activation": "linear",
            }
        ],
    }

    assert api._clip_head_predict_proba(feats, bad_logreg) is None
    assert api._clip_head_predict_proba(feats, bad_mlp_weight) is None
    assert api._clip_head_predict_proba(feats, bad_layer_norm) is None


def test_clip_head_predict_proba_ignores_malformed_logit_adjustment():
    feats = np.asarray([[1.0, 0.0]], dtype=np.float32)
    head = {
        "classifier_type": "logreg",
        "classes": ["car", "boat"],
        "coef": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "intercept": np.zeros(2, dtype=np.float32),
        "proba_mode": "softmax",
        "logit_adjustment_inference": True,
        "logit_adjustment": [object(), 0.0],
    }

    actual = api._clip_head_predict_proba(feats, head)

    assert actual is not None
    assert actual.shape == (1, 2)
    assert float(actual[0, 0]) > float(actual[0, 1])


def test_classifier_postprocess_matches_training_normalize_then_center_order():
    feats = np.asarray([[3.0, 4.0]], dtype=np.float32)
    head = {
        "classifier_type": "logreg",
        "normalize_embeddings": True,
        "embedding_center_values": [0.6, 0.8],
    }

    actual = api._postprocess_features_for_head(feats, head=head)

    assert np.allclose(actual, np.zeros((1, 2), dtype=np.float32), atol=1e-6)


def test_predict_base64_replays_classifier_crop_recipe_with_scaled_bbox(monkeypatch):
    captured = {}
    image = Image.new("RGB", (50, 100), (10, 20, 30))

    def fake_resolve(*args, **kwargs):
        return image, np.asarray(image), "token"

    def fake_encode(pil_img, xyxy):
        captured["image_size"] = pil_img.size
        captured["xyxy"] = [float(v) for v in xyxy]
        return np.asarray([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(api, "_active_encoder_ready", lambda: True)
    monkeypatch.setattr(api, "_resolve_detector_image_impl", fake_resolve)
    monkeypatch.setattr(api, "_encode_classifier_xyxy_for_active", fake_encode)
    monkeypatch.setattr(
        api,
        "_clip_auto_predict_details",
        lambda feats, background_guard=False: {
            "label": "car",
            "proba": 0.9,
            "second_label": "boat",
            "second_proba": 0.1,
            "margin": 0.8,
            "error": None,
        },
    )

    response = api.predict_base64(
        api.Base64Payload(
            image_base64="ignored",
            uuid="bbox-1",
            bbox_xyxy=[10.0, 20.0, 30.0, 60.0],
            image_width=100,
            image_height=200,
        )
    )

    assert response.prediction == "car"
    assert response.uuid == "bbox-1"
    assert captured["image_size"] == (50, 100)
    assert captured["xyxy"] == [5.0, 10.0, 15.0, 30.0]


def test_predict_base64_crop_only_uses_full_image_as_bbox(monkeypatch):
    captured = {}
    image = Image.new("RGB", (24, 16), (10, 20, 30))

    monkeypatch.setattr(api, "_active_encoder_ready", lambda: True)
    monkeypatch.setattr(
        api,
        "_resolve_detector_image_impl",
        lambda *args, **kwargs: (image, np.asarray(image), "token"),
    )
    def fake_encode(pil_img, xyxy):
        captured["xyxy"] = [float(v) for v in xyxy]
        return np.asarray([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(api, "_encode_classifier_xyxy_for_active", fake_encode)
    monkeypatch.setattr(
        api,
        "_clip_auto_predict_details",
        lambda feats, background_guard=False: {"label": "car", "error": None},
    )

    api.predict_base64(api.Base64Payload(image_base64="ignored", uuid="crop-1"))

    assert captured["xyxy"] == [0.0, 0.0, 24.0, 16.0]


def test_set_active_model_accepts_multiview_clip_embedding_width(tmp_path, monkeypatch):
    classifiers_root = tmp_path / "classifiers"
    labelmaps_root = tmp_path / "labelmaps"
    classifiers_root.mkdir()
    labelmaps_root.mkdir()
    classifier_path = classifiers_root / "clip_multiview.pkl"
    meta_path = classifiers_root / "clip_multiview.meta.pkl"
    labelmap_path = labelmaps_root / "labels.pkl"
    classifier = types.SimpleNamespace(
        classes_=np.asarray(["car", "boat"], dtype=object),
        coef_=np.zeros((2, 1536), dtype=np.float32),
        intercept_=np.zeros(2, dtype=np.float32),
        solver="lbfgs",
        multi_class="auto",
    )
    api.joblib.dump(classifier, classifier_path)
    api.joblib.dump(
        {
            "clip_model": "ViT-L/14",
            "encoder_type": "clip",
            "encoder_model": "ViT-L/14",
            "embedding_view_mode": "tight_context",
            "embedding_dim": 1536,
        },
        meta_path,
    )
    api.joblib.dump(["car", "boat"], labelmap_path)

    class FakeClipModel:
        visual = types.SimpleNamespace(output_dim=768)

    monkeypatch.setattr(api, "UPLOAD_ROOT", tmp_path)
    monkeypatch.setattr(api, "clip_model", None)
    monkeypatch.setattr(api, "clip_preprocess", None)
    monkeypatch.setattr(api, "clip_model_name", "ViT-B/32")
    monkeypatch.setattr(api.clip, "load", lambda name, device=None: (FakeClipModel(), object()))

    payload = api.set_active_model(
        api.ActiveModelRequest(
            classifier_path=str(classifier_path),
            labelmap_path=str(labelmap_path),
        )
    )

    assert payload["encoder_type"] == "clip"
    assert payload["encoder_ready"] is True
    assert api.active_classifier_head["embedding_dim"] == 1536
    assert api.active_classifier_head["embedding_view_mode"] == "tight_context"


def test_set_active_model_rejects_invalid_mlp_head_before_activation(tmp_path, monkeypatch):
    classifiers_root = tmp_path / "classifiers"
    labelmaps_root = tmp_path / "labelmaps"
    classifiers_root.mkdir()
    labelmaps_root.mkdir()
    classifier_path = classifiers_root / "bad_mlp.pkl"
    meta_path = classifiers_root / "bad_mlp.meta.pkl"
    labelmap_path = labelmaps_root / "labels.pkl"
    classifier = {
        "classifier_type": "mlp",
        "classes": np.asarray(["car", "boat"], dtype=object),
        "embedding_dim": 2,
        "layers": [
            {
                "weight": np.eye(2, dtype=np.float32),
                "bias": np.zeros(2, dtype=np.float32),
                "activation": "relu",
            },
            {
                "weight": np.zeros((2, 3), dtype=np.float32),
                "bias": np.zeros(2, dtype=np.float32),
                "activation": "linear",
            },
        ],
    }
    api.joblib.dump(classifier, classifier_path)
    api.joblib.dump(
        {
            "clip_model": "ViT-B/32",
            "encoder_type": "clip",
            "encoder_model": "ViT-B/32",
            "embedding_dim": 2,
        },
        meta_path,
    )
    api.joblib.dump(["car", "boat"], labelmap_path)

    class FakeClipModel:
        visual = types.SimpleNamespace(output_dim=2)

    previous_path = "/tmp/old-classifier.pkl"
    previous_head = {"classes": ["old"]}
    monkeypatch.setattr(api, "UPLOAD_ROOT", tmp_path)
    monkeypatch.setattr(api, "clip_model", None)
    monkeypatch.setattr(api, "clip_preprocess", None)
    monkeypatch.setattr(api, "clip_model_name", "ViT-B/32")
    monkeypatch.setattr(api, "active_classifier_path", previous_path)
    monkeypatch.setattr(api, "active_classifier_head", previous_head)
    monkeypatch.setattr(api.clip, "load", lambda name, device=None: (FakeClipModel(), object()))

    with pytest.raises(api.HTTPException) as exc:
        api.set_active_model(
            api.ActiveModelRequest(
                classifier_path=str(classifier_path),
                labelmap_path=str(labelmap_path),
            )
        )

    assert exc.value.detail == "agent_clip_classifier_invalid_shape"
    assert api.active_classifier_path == previous_path
    assert api.active_classifier_head is previous_head


def test_set_active_model_rejects_classifier_sibling_prefix_path(tmp_path, monkeypatch):
    upload_root = tmp_path / "uploads"
    (upload_root / "classifiers").mkdir(parents=True)
    sibling_root = tmp_path / "uploads" / "classifiers_evil"
    sibling_root.mkdir()
    classifier_path = sibling_root / "outside.pkl"
    classifier_path.write_bytes(b"not a classifier")
    monkeypatch.setattr(api, "UPLOAD_ROOT", upload_root)

    def fail_load(_path):
        raise AssertionError("classifier outside upload root should not be loaded")

    monkeypatch.setattr(api.joblib, "load", fail_load)

    with pytest.raises(api.HTTPException) as exc:
        api.set_active_model(api.ActiveModelRequest(classifier_path=str(classifier_path)))

    assert exc.value.detail == "classifier_path_not_allowed"


def test_set_active_model_rejects_labelmap_sibling_prefix_path(tmp_path, monkeypatch):
    upload_root = tmp_path / "uploads"
    classifiers_root = upload_root / "classifiers"
    labelmaps_root = upload_root / "labelmaps"
    sibling_labelmaps_root = upload_root / "labelmaps_evil"
    classifiers_root.mkdir(parents=True)
    labelmaps_root.mkdir()
    sibling_labelmaps_root.mkdir()
    classifier_path = classifiers_root / "head.pkl"
    meta_path = classifiers_root / "head.meta.pkl"
    labelmap_path = sibling_labelmaps_root / "labels.pkl"
    classifier = types.SimpleNamespace(
        classes_=np.asarray(["car"], dtype=object),
        coef_=np.zeros((1, 2), dtype=np.float32),
        intercept_=np.zeros(1, dtype=np.float32),
        solver="lbfgs",
        multi_class="auto",
    )
    api.joblib.dump(classifier, classifier_path)
    api.joblib.dump(
        {
            "clip_model": "ViT-B/32",
            "encoder_type": "clip",
            "encoder_model": "ViT-B/32",
            "embedding_dim": 2,
        },
        meta_path,
    )
    api.joblib.dump(["car"], labelmap_path)

    class FakeClipModel:
        visual = types.SimpleNamespace(output_dim=2)

    monkeypatch.setattr(api, "UPLOAD_ROOT", upload_root)
    monkeypatch.setattr(api, "clip_model", None)
    monkeypatch.setattr(api, "clip_preprocess", None)
    monkeypatch.setattr(api, "clip_model_name", "ViT-B/32")
    monkeypatch.setattr(api.clip, "load", lambda name, device=None: (FakeClipModel(), object()))

    with pytest.raises(api.HTTPException) as exc:
        api.set_active_model(
            api.ActiveModelRequest(
                classifier_path=str(classifier_path),
                labelmap_path=str(labelmap_path),
            )
        )

    assert exc.value.detail == "labelmap_path_not_allowed"


def test_set_active_model_accepts_multiview_dinov3_embedding_width(tmp_path, monkeypatch):
    classifiers_root = tmp_path / "classifiers"
    labelmaps_root = tmp_path / "labelmaps"
    classifiers_root.mkdir()
    labelmaps_root.mkdir()
    classifier_path = classifiers_root / "dino_multiview.pkl"
    meta_path = classifiers_root / "dino_multiview.meta.pkl"
    labelmap_path = labelmaps_root / "labels.pkl"
    classifier = types.SimpleNamespace(
        classes_=np.asarray(["car", "boat"], dtype=object),
        coef_=np.zeros((2, 2048), dtype=np.float32),
        intercept_=np.zeros(2, dtype=np.float32),
        solver="lbfgs",
        multi_class="auto",
    )
    api.joblib.dump(classifier, classifier_path)
    api.joblib.dump(
        {
            "encoder_type": "dinov3",
            "encoder_model": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "embedding_view_mode": "tight_context",
            "dinov3_pooling": "pooler",
            "embedding_dim": 2048,
        },
        meta_path,
    )
    api.joblib.dump(["car", "boat"], labelmap_path)

    class FakeDinoModel:
        config = types.SimpleNamespace(hidden_size=1024)

    monkeypatch.setattr(api, "UPLOAD_ROOT", tmp_path)
    monkeypatch.setattr(api, "dinov3_model", None)
    monkeypatch.setattr(api, "dinov3_processor", None)
    monkeypatch.setattr(api, "dinov3_initialized", False)
    monkeypatch.setattr(
        api,
        "_data_ingestion_get_dinov3",
        lambda model_name, device_name=None: (FakeDinoModel(), object(), model_name, device_name or "cpu"),
    )
    monkeypatch.setattr(api, "resolve_mlx_dinov3_backend", lambda *_args, **_kwargs: "torch")

    payload = api.set_active_model(
        api.ActiveModelRequest(
            classifier_path=str(classifier_path),
            labelmap_path=str(labelmap_path),
        )
    )

    assert payload["encoder_type"] == "dinov3"
    assert payload["encoder_ready"] is True
    assert api.active_classifier_head["embedding_dim"] == 2048
    assert api.active_classifier_head["embedding_view_mode"] == "tight_context"


def test_set_active_model_accepts_cradio_mlx_without_processor(tmp_path, monkeypatch):
    classifiers_root = tmp_path / "classifiers"
    labelmaps_root = tmp_path / "labelmaps"
    classifiers_root.mkdir()
    labelmaps_root.mkdir()
    classifier_path = classifiers_root / "cradio_mlx.pkl"
    meta_path = classifiers_root / "cradio_mlx.meta.pkl"
    labelmap_path = labelmaps_root / "labels.pkl"
    classifier = types.SimpleNamespace(
        classes_=np.asarray(["car", "boat"], dtype=object),
        coef_=np.zeros((2, 16), dtype=np.float32),
        intercept_=np.zeros(2, dtype=np.float32),
        solver="lbfgs",
        multi_class="auto",
    )
    api.joblib.dump(classifier, classifier_path)
    api.joblib.dump(
        {
            "encoder_type": "cradio",
            "encoder_model": CRADIO_DEFAULT_MODEL,
            "cradio_pooling": "summary_spatial_concat",
            "embedding_dim": 16,
        },
        meta_path,
    )
    api.joblib.dump(["car", "boat"], labelmap_path)

    fake_model = types.SimpleNamespace(output_dim=8)
    monkeypatch.setattr(api, "UPLOAD_ROOT", tmp_path)
    monkeypatch.setattr(api, "cradio_model", None)
    monkeypatch.setattr(api, "cradio_processor", None)
    monkeypatch.setattr(api, "cradio_model_name", None)
    monkeypatch.setattr(api, "cradio_model_device", None)
    monkeypatch.setattr(api, "cradio_initialized", False)
    monkeypatch.setattr(api, "resolve_cradio_torch_device", lambda **_kwargs: "mlx")
    monkeypatch.setattr(
        api,
        "_load_cradio_backbone_cached",
        lambda model_name, target_device, raise_on_error=False: (fake_model, None, model_name, "mlx"),
    )

    payload = api.set_active_model(
        api.ActiveModelRequest(
            classifier_path=str(classifier_path),
            labelmap_path=str(labelmap_path),
        )
    )

    assert payload["encoder_type"] == "cradio"
    assert payload["encoder_ready"] is True
    assert api.cradio_model is fake_model
    assert api.cradio_processor is None
    assert api.cradio_model_device == "mlx"
    assert api.active_classifier_head["embedding_dim"] == 16
    assert api.active_classifier_head["cradio_pooling"] == "summary_spatial_concat"


def test_set_active_model_rejects_cradio_embedding_width_mismatch(tmp_path, monkeypatch):
    classifiers_root = tmp_path / "classifiers"
    labelmaps_root = tmp_path / "labelmaps"
    classifiers_root.mkdir()
    labelmaps_root.mkdir()
    classifier_path = classifiers_root / "cradio_bad_width.pkl"
    meta_path = classifiers_root / "cradio_bad_width.meta.pkl"
    labelmap_path = labelmaps_root / "labels.pkl"
    classifier = types.SimpleNamespace(
        classes_=np.asarray(["car", "boat"], dtype=object),
        coef_=np.zeros((2, 15), dtype=np.float32),
        intercept_=np.zeros(2, dtype=np.float32),
        solver="lbfgs",
        multi_class="auto",
    )
    api.joblib.dump(classifier, classifier_path)
    api.joblib.dump(
        {
            "encoder_type": "cradio",
            "encoder_model": CRADIO_DEFAULT_MODEL,
            "cradio_pooling": "summary_spatial_concat",
            "embedding_dim": 15,
        },
        meta_path,
    )
    api.joblib.dump(["car", "boat"], labelmap_path)

    fake_model = types.SimpleNamespace(output_dim=8)
    monkeypatch.setattr(api, "UPLOAD_ROOT", tmp_path)
    monkeypatch.setattr(api, "resolve_cradio_torch_device", lambda **_kwargs: "mlx")
    monkeypatch.setattr(
        api,
        "_load_cradio_backbone_cached",
        lambda model_name, target_device, raise_on_error=False: (fake_model, None, model_name, "mlx"),
    )

    with pytest.raises(api.HTTPException) as exc:
        api.set_active_model(
            api.ActiveModelRequest(
                classifier_path=str(classifier_path),
                labelmap_path=str(labelmap_path),
            )
        )

    assert exc.value.detail == "dimension_mismatch:15!=16"


def test_training_multiview_items_compose_consistent_embedding_widths():
    def fake_encode(images):
        return np.asarray(
            [[float(idx + 1), float(idx + 2)] for idx, _image in enumerate(images)],
            dtype=np.float32,
        )

    image = Image.new("RGB", (96, 72), (30, 60, 90))
    positive_views, _positive_crop_xyxy, positive_meta = clip_training._embedding_make_crop_views(
        image,
        (20, 18, 36, 34),
        crop_mode="padded_square",
        padding_ratio=0.08,
        preprocess_mode="canonical",
        canonical_size=64,
        background_mode="blur_outside_box",
        view_mode="tight_context",
    )
    background_views, _background_crop_xyxy, background_meta = clip_training._embedding_make_crop_views(
        image,
        (54, 20, 70, 36),
        crop_mode="padded_square",
        padding_ratio=0.08,
        preprocess_mode="canonical",
        canonical_size=64,
        background_mode="blur_outside_box",
        view_mode="tight_context",
    )
    try:
        positive_item = tuple(positive_views)
        background_item = tuple(background_views)
        augmented_positive = clip_training._apply_augmenter_to_item(None, positive_item)
        augmented_background = clip_training._apply_augmenter_to_item(None, background_item)

        positive_embedding = clip_training._encode_embedding_items(
            [augmented_positive],
            encode_images_fn=fake_encode,
        )
        background_embedding = clip_training._encode_embedding_items(
            [augmented_background],
            encode_images_fn=fake_encode,
        )

        assert len(positive_item) == 2
        assert len(background_item) == 2
        assert positive_embedding.shape == (1, 4)
        assert background_embedding.shape == (1, 4)
        assert positive_embedding.shape[1] == background_embedding.shape[1]
        assert [entry["view"] for entry in positive_meta] == ["tight", "context"]
        assert [entry["view"] for entry in background_meta] == ["tight", "context"]
        assert all(view.size == (64, 64) for view in positive_item)
        assert all(view.size == (64, 64) for view in background_item)
    finally:
        for name in ("augmented_positive", "augmented_background"):
            item = locals().get(name)
            if item is not None:
                clip_training._close_crop_item(item)
        clip_training._close_crop_item(positive_views)
        clip_training._close_crop_item(background_views)
        image.close()


def test_class_analysis_image_pack_roundtrip_and_corrupt_recovery(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    head = {
        "encoder_type": "dinov3",
        "encoder_model": "test-dino",
        "dinov3_pooling": "pooler",
        "normalize_embeddings": True,
    }
    embeddings = [
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
    ]

    assert api._class_analysis_write_image_pack("ab" * 32, head, ["crop-a", "crop-b"], embeddings)
    loaded = api._class_analysis_load_image_pack("ab" * 32, head, ["crop-a", "crop-b"])

    assert loaded is not None
    assert np.allclose(np.stack(loaded), np.stack(embeddings))
    pack_path = api._class_analysis_image_pack_path("ab" * 32, head)
    pack_path.write_bytes(b"not-an-npz")
    assert api._class_analysis_load_image_pack("ab" * 32, head, ["crop-a", "crop-b"]) is None
    assert not pack_path.exists()


def test_class_analysis_image_pack_preflights_huge_header_before_np_load(
    tmp_path,
    monkeypatch,
):
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    head = {
        "encoder_type": "dinov3",
        "encoder_model": "test-dino",
        "dinov3_pooling": "pooler",
        "normalize_embeddings": True,
    }
    pack_path = api._class_analysis_image_pack_path("cd" * 32, head)
    pack_path.parent.mkdir(parents=True)
    _class_analysis_test_write_npz_members(
        pack_path,
        {
            "version": _class_analysis_test_npy_bytes(
                np.asarray([api.CLASS_ANALYSIS_CACHE_VERSION])
            ),
            "object_keys": _class_analysis_test_npy_bytes(
                np.asarray(["crop-a", "crop-b"])
            ),
            "embeddings": _class_analysis_test_npy_header(
                shape=(2, 100_000_000),
                dtype=np.float32,
            ),
        },
    )
    load_calls = []

    def reject_np_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        raise AssertionError("np.load must follow bounded NPZ preflight")

    monkeypatch.setattr(api.np, "load", reject_np_load)

    assert api._class_analysis_load_image_pack(
        "cd" * 32,
        head,
        ["crop-a", "crop-b"],
    ) is None
    assert load_calls == []
    assert not pack_path.exists()


@pytest.mark.parametrize("defect", ["wrong_dtype", "extra_member"])
def test_class_analysis_image_pack_rejects_exact_member_or_dtype_drift(
    tmp_path,
    monkeypatch,
    defect,
):
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    head = {
        "encoder_type": "dinov3",
        "encoder_model": "test-dino",
        "normalize_embeddings": True,
    }
    pack_path = api._class_analysis_image_pack_path("ef" * 32, head)
    pack_path.parent.mkdir(parents=True)
    members = {
        "version": np.asarray([api.CLASS_ANALYSIS_CACHE_VERSION]),
        "object_keys": np.asarray(["crop-a"]),
        "embeddings": np.asarray([[1.0, 2.0]], dtype=np.float32),
    }
    if defect == "wrong_dtype":
        members["embeddings"] = members["embeddings"].astype(np.float64)
    else:
        members["unexpected"] = np.asarray([1], dtype=np.int32)
    np.savez_compressed(pack_path, **members)

    assert api._class_analysis_load_image_pack(
        "ef" * 32,
        head,
        ["crop-a"],
    ) is None
    assert not pack_path.exists()


def test_class_analysis_saved_embeddings_prefer_memmap_and_support_legacy_npz(
    tmp_path,
):
    expected = np.arange(12, dtype=np.float32).reshape(4, 3)
    np.savez_compressed(tmp_path / "embeddings.npz", embeddings=expected + 100)
    np.save(tmp_path / "embeddings.npy", expected)

    matrix, path = api._class_analysis_load_saved_embeddings(tmp_path)

    assert path == tmp_path / "embeddings.npy"
    assert isinstance(matrix, np.memmap)
    assert np.array_equal(matrix, expected)

    (tmp_path / "embeddings.npy").unlink()
    legacy, legacy_path = api._class_analysis_load_saved_embeddings(tmp_path)

    assert legacy_path == tmp_path / "embeddings.npz"
    assert np.array_equal(legacy, expected + 100)


def test_class_analysis_saved_embeddings_preflight_huge_headers_without_deleting(
    tmp_path,
    monkeypatch,
):
    npy_path = tmp_path / "embeddings.npy"
    npy_path.write_bytes(
        _class_analysis_test_npy_header(
            shape=(100_000_000, 2),
            dtype=np.float32,
        )
    )
    memmap_calls = []
    load_calls = []

    def reject_memmap(*args, **kwargs):
        memmap_calls.append((args, kwargs))
        raise AssertionError("memmap must follow bounded NPY preflight")

    def reject_np_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        raise AssertionError("np.load must follow bounded NPZ preflight")

    monkeypatch.setattr(api.np, "memmap", reject_memmap)
    monkeypatch.setattr(api.np, "load", reject_np_load)

    matrix, path = api._class_analysis_load_saved_embeddings(tmp_path)
    assert matrix is None
    assert path == npy_path
    assert memmap_calls == []
    assert load_calls == []
    assert npy_path.exists()

    npy_path.unlink()
    npz_path = tmp_path / "embeddings.npz"
    _class_analysis_test_write_npz_members(
        npz_path,
        {
            "embeddings": _class_analysis_test_npy_header(
                shape=(1_000_000_000, 2),
                dtype=np.float32,
            )
        },
    )
    legacy, legacy_path = api._class_analysis_load_saved_embeddings(tmp_path)
    assert legacy is None
    assert legacy_path == npz_path
    assert load_calls == []
    assert npz_path.exists()


@pytest.mark.parametrize("legacy_npz", [False, True])
def test_class_analysis_saved_embeddings_reject_nonfinite_without_deleting(
    tmp_path,
    legacy_npz,
):
    values = np.asarray([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
    if legacy_npz:
        artifact_path = tmp_path / "embeddings.npz"
        np.savez_compressed(artifact_path, embeddings=values)
    else:
        artifact_path = tmp_path / "embeddings.npy"
        np.save(artifact_path, values)

    matrix, path = api._class_analysis_load_saved_embeddings(tmp_path)

    assert matrix is None
    assert path == artifact_path
    assert artifact_path.exists()


@pytest.mark.parametrize("legacy_npz", [False, True])
def test_class_analysis_saved_embeddings_reject_dtype_without_deleting(
    tmp_path,
    legacy_npz,
):
    values = np.asarray([[1.0, 2.0]], dtype=np.float64)
    if legacy_npz:
        artifact_path = tmp_path / "embeddings.npz"
        np.savez_compressed(artifact_path, embeddings=values)
    else:
        artifact_path = tmp_path / "embeddings.npy"
        np.save(artifact_path, values)

    matrix, path = api._class_analysis_load_saved_embeddings(tmp_path)

    assert matrix is None
    assert path == artifact_path
    assert artifact_path.exists()


def test_class_analysis_review_dispositions_survive_registry_restart_and_preserve_evidence(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    job_dir = class_root / "ca_reviews"
    job_dir.mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    confirm_key = api._class_analysis_review_object_key(
        source_key="active:cas_abc",
        image_sha256="11" * 32,
        split="train",
        image_relpath="confirm.jpg",
        class_name="Boat",
        geometry={"kind": "bbox", "bbox_xyxy": [1, 2, 30, 40]},
    )
    skip_key = api._class_analysis_review_object_key(
        source_key="active:cas_abc",
        image_sha256="22" * 32,
        split="train",
        image_relpath="skip.jpg",
        class_name="ElevatedFixture",
        geometry={"kind": "bbox", "bbox_xyxy": [5, 6, 20, 50]},
    )
    points = [
        {
            **_record("confirm", "Boat"),
            "image_relpath": "confirm.jpg",
            "bbox_xyxy": [1, 2, 30, 40],
            "review_object_key": confirm_key,
            "wrong_class_suspicion": 0.91,
            "wrong_class_review_reason": "embedding_outlier",
            "is_wrong_class_candidate": True,
            "review_signals": ["wrong_class"],
        },
        {
            **_record("skip", "ElevatedFixture"),
            "image_relpath": "skip.jpg",
            "bbox_xyxy": [5, 6, 20, 50],
            "review_object_key": skip_key,
            "wrong_class_suspicion": 0.87,
            "wrong_class_review_reason": "embedding_outlier",
            "is_wrong_class_candidate": True,
            "review_signals": ["wrong_class"],
        },
    ]
    result = {
        "summary": {
            "source_mode": "active_workspace",
            "source_id": "cas_abc",
            "source_key": "active:cas_abc",
            "wrong_class_candidate_count": 2,
        },
        "points": points,
        "wrong_class_candidates": [
            {"point_id": "confirm", "review_object_key": confirm_key},
            {"point_id": "skip", "review_object_key": skip_key},
        ],
    }
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(result_path, job_dir, result)
    (job_dir / "config.json").write_text("{}", encoding="utf-8")
    api.CLASS_ANALYSIS_JOBS["ca_reviews"] = api.ClassAnalysisJob(
        job_id="ca_reviews",
        status="completed",
        result_path=str(result_path),
        summary=dict(result["summary"]),
    )

    try:
        confirmed = api.record_class_analysis_review_disposition(
            "ca_reviews",
            "confirm",
            {
                "disposition": "confirm_current",
                "origin": "desktop",
                "client_action_id": "reviews-confirm-current",
            },
        )
        skipped = api.record_class_analysis_review_disposition(
            "ca_reviews",
            "skip",
            {
                "disposition": "skip",
                "origin": "desktop",
                "client_action_id": "reviews-skip-first",
            },
        )
        # An uncertain retry reuses its idempotency token and receipt; it
        # cannot accumulate duplicate rows or replace a newer choice.
        repeated_skip = api.record_class_analysis_review_disposition(
            "ca_reviews",
            "skip",
            {
                "disposition": "skip",
                "origin": "desktop",
                "client_action_id": "reviews-skip-first",
            },
        )
        assert repeated_skip["idempotent_replay"] is True
        assert (
            repeated_skip["human_review_revision"]
            == skipped["human_review_revision"]
        )

        overlaid = api.get_class_analysis_result("ca_reviews")
        assert overlaid["wrong_class_candidates"] == []
        assert overlaid["summary"]["wrong_class_candidate_count"] == 0
        assert overlaid["summary"]["wrong_class_candidate_count_before_human_review"] == 2
        assert overlaid["summary"]["human_review_disposition_counts"] == {
            "confirm_current": 1,
            "skip": 1,
        }
        overlaid_by_id = {point["point_id"]: point for point in overlaid["points"]}
        assert overlaid_by_id["confirm"]["is_wrong_class_candidate"] is False
        assert overlaid_by_id["skip"]["is_wrong_class_candidate"] is True
        assert overlaid_by_id["confirm"]["wrong_class_suspicion"] == pytest.approx(0.91)
        assert overlaid_by_id["skip"]["wrong_class_suspicion"] == pytest.approx(0.87)
        assert overlaid_by_id["confirm"]["review_signals"] == ["wrong_class"]
        assert overlaid_by_id["skip"]["review_signals"] == ["wrong_class"]
        assert confirmed["human_reviewed_at"] > 0
        assert api.CLASS_ANALYSIS_REVIEW_ENTRY_REVISION_PATTERN.fullmatch(
            confirmed["human_review_revision"]
        )
        assert (
            overlaid_by_id["confirm"]["human_review_revision"]
            == confirmed["human_review_revision"]
        )
        assert api.CLASS_ANALYSIS_REVIEW_ENTRY_REVISION_PATTERN.fullmatch(
            overlaid_by_id["skip"]["human_review_revision"]
        )
        assert skipped["disposition"] == "skip"

        raw = json.loads(result_path.read_text(encoding="utf-8"))
        assert len(raw["wrong_class_candidates"]) == 2
        assert all(point["is_wrong_class_candidate"] for point in raw["points"])

        api.CLASS_ANALYSIS_JOBS.pop("ca_reviews", None)
        restored = api.get_class_analysis_result("ca_reviews")
        assert restored["wrong_class_candidates"] == []
        assert {
            point["point_id"]: point["is_wrong_class_candidate"]
            for point in restored["points"]
        } == {"confirm": False, "skip": True}

        rerun = api._class_analysis_apply_review_dispositions(
            api._class_analysis_public_result(
                {
                    "summary": {
                        "source_mode": "active_workspace",
                        "source_id": "cas_changed_by_unrelated_image",
                        "source_key": "active:cas_changed_by_unrelated_image",
                    },
                    "points": [
                        {
                            **points[0],
                            "point_id": "confirm_new_line_order",
                            "is_wrong_class_candidate": True,
                        }
                    ],
                    "wrong_class_candidates": [
                        {
                            "point_id": "confirm_new_line_order",
                            "review_object_key": confirm_key,
                        }
                    ],
                }
            )
        )
        assert rerun["wrong_class_candidates"] == []
        assert rerun["points"][0]["is_wrong_class_candidate"] is False

        entries = {}
        for shard_path in (
            class_root / "audit" / "human_review_dispositions"
        ).glob("*.json"):
            entries.update(json.loads(shard_path.read_text(encoding="utf-8"))["entries"])
        assert set(entries) == {confirm_key, skip_key}
        assert entries[skip_key]["origin"] == "desktop"
    finally:
        api.CLASS_ANALYSIS_JOBS.pop("ca_reviews", None)


def test_selected_class_dispositions_preserve_scope_authoritative_selector_queue(
    monkeypatch,
):
    reviewed_key = "rk-reviewed"
    kept_key = "rk-kept"
    monkeypatch.setattr(
        api,
        "_class_analysis_lookup_review_dispositions",
        lambda keys: {
            reviewed_key: {
                "disposition": "skip",
                "updated_at": 123.0,
                "origin": "desktop",
            }
        },
    )
    within = [
        {"point_id": "reviewed", "review_object_key": reviewed_key},
        {"point_id": "kept", "review_object_key": kept_key},
    ]
    result = {
        "summary": {"analysis_scope": "selected_class"},
        "points": [
            {**within[0], "class_name": "Bike"},
            {**within[1], "class_name": "Bike"},
        ],
        # The selected-class selector is bound to `within`, while this broader
        # compatibility collection may have different membership.
        "wrong_class_candidates": [*within, {"point_id": "pair"}],
        "within_class_outlier_candidates": list(within),
        "refinement_candidates": list(within),
        "vignette_candidates": list(within),
        "refinement_summary": {
            "selector_priority_contract": api.CLASS_ANALYSIS_SELECTOR_PRIORITY_CONTRACT,
            "selector_priority": {
                "candidate_count": 2,
            },
        },
    }

    overlaid = api._class_analysis_apply_review_dispositions(result)

    assert [
        row["point_id"]
        for row in overlaid["within_class_outlier_candidates"]
    ] == ["reviewed", "kept"]
    assert [
        row["point_id"]
        for row in overlaid["refinement_candidates"]
    ] == ["reviewed", "kept"]
    assert overlaid["summary"]["complete_selector_queue_preserved"] is True
    assert overlaid["summary"][
        "visible_within_class_outlier_candidate_count"
    ] == 1
    assert overlaid["points"][0]["human_review_disposition"] == "skip"


def test_class_analysis_delete_bbox_disposition_hides_stale_completed_candidate(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    (class_root / "ca_deleted_bbox").mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    key = api._class_analysis_review_object_key(
        source_key="linked:dataset",
        image_sha256="dc" * 32,
        split="train",
        image_relpath="deleted.jpg",
        class_name="Boat",
        geometry={"kind": "bbox", "bbox_xyxy": [1, 2, 30, 40]},
    )
    point = {
        **_record("deleted", "Boat"),
        "image_relpath": "deleted.jpg",
        "bbox_xyxy": [1, 2, 30, 40],
        "review_object_key": key,
        "is_wrong_class_candidate": True,
    }
    job = api.ClassAnalysisJob(
        job_id="ca_deleted_bbox",
        status="completed",
        summary={
            "analysis_job_id": "ca_deleted_bbox",
            "source_mode": "linked",
            "source_id": "dataset",
            "source_key": "linked:dataset",
        },
    )
    api.CLASS_ANALYSIS_JOBS[job.job_id] = job
    monkeypatch.setattr(
        api,
        "_class_analysis_thumbnail_point_metadata",
        lambda *_args, **_kwargs: dict(point),
    )
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "dataset",
            "dataset_root": str(tmp_path / "dataset"),
        },
    )
    annotation_target = {
        "source_mode": "linked",
        "source_id": "dataset",
        "split": "train",
        "image_relpath": "deleted.jpg",
    }
    attestation = {
        "schema": api.CLASS_ANALYSIS_SINGLE_BBOX_DELETION_ATTESTATION_SCHEMA,
        "committed": True,
        "analysis_job_id": job.job_id,
        "point_id": point["point_id"],
        "review_object_key": key,
        "annotation_target": annotation_target,
        "source_identity": f"asi1_{'12' * 32}",
        "before_revision": f"alr1_{'34' * 32}",
        "committed_revision": f"alr1_{'56' * 32}",
        "image_sha256": "dc" * 32,
        "committed_label_count": 0,
        "deleted_label_line_sha256": "ab" * 32,
        "deleted_label_line_index": 0,
        "verification_method": (
            api.CLASS_ANALYSIS_SINGLE_BBOX_DELETION_VERIFICATION_METHOD
        ),
        "verified_at": 1.0,
    }
    attestation["attestation_sha256"] = (
        api._class_analysis_single_bbox_deletion_attestation_hash(
            attestation
        )
    )
    deletion_verifications = []

    def verify_deletion(**kwargs):
        deletion_verifications.append(kwargs)
        return dict(attestation)

    monkeypatch.setattr(
        api,
        "_class_analysis_validate_single_bbox_deletion_commit",
        verify_deletion,
    )
    captured_training_payloads = []

    def capture_training_action(job_id, payload):
        captured_training_payloads.append((job_id, payload))
        return {"status": "recorded", "recorded_count": 1}

    monkeypatch.setattr(
        api,
        "record_class_analysis_vignette_training_action",
        capture_training_action,
    )
    request = {
        "disposition": "delete_bbox",
        "origin": "desktop",
        "client_action_id": "delete-single-bbox",
        "capture_training_data": True,
        "label_commit_status": "committed",
        "annotation_target": annotation_target,
        "annotation_before_revision": attestation["before_revision"],
        "annotation_commit_revision": attestation["committed_revision"],
        "annotation_source_identity": attestation["source_identity"],
    }
    try:
        with pytest.raises(api.HTTPException) as unverified_exc:
            api.record_class_analysis_review_disposition(
                job.job_id,
                point["point_id"],
                {
                    "disposition": "delete_bbox",
                    "origin": "desktop",
                    "client_action_id": "delete-without-commit-proof",
                },
            )
        assert unverified_exc.value.status_code == 400
        assert unverified_exc.value.detail == (
            "single_bbox_deletion_label_commit_required"
        )
        receipt = api.record_class_analysis_review_disposition(
            job.job_id,
            point["point_id"],
            request,
        )
        replay = api.record_class_analysis_review_disposition(
            job.job_id,
            point["point_id"],
            request,
        )
        overlaid = api._class_analysis_apply_review_dispositions(
            {
                "summary": dict(job.summary),
                "points": [dict(point)],
                "wrong_class_candidates": [
                    {
                        "point_id": point["point_id"],
                        "review_object_key": key,
                    }
                ],
            },
            job=job,
        )
        deleted_history = api.delete_class_analysis_review_history(
            job.job_id,
            {
                "schema": api.CLASS_ANALYSIS_REVIEW_HISTORY_DELETE_SCHEMA,
                "client_action_id": "delete-single-history",
                "entries": [
                    {
                        "point_id": point["point_id"],
                        "expected_disposition": "delete_bbox",
                        "expected_revision": receipt[
                            "human_review_revision"
                        ],
                    }
                ],
            },
        )
        monkeypatch.setattr(
            api,
            "_class_analysis_validate_single_bbox_deletion_commit",
            lambda **_kwargs: pytest.fail(
                "tombstoned retry must be rejected before live verification"
            ),
        )
        with pytest.raises(api.HTTPException) as deleted_replay_exc:
            api.record_class_analysis_review_disposition(
                job.job_id,
                point["point_id"],
                request,
            )
    finally:
        api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)

    assert receipt["disposition"] == "delete_bbox"
    assert receipt["training_capture_requested"] is True
    assert receipt["annotation_commit_attestation"] == attestation
    assert replay["idempotent_replay"] is True
    assert deleted_history["status"] == "deleted"
    assert deleted_replay_exc.value.status_code == 409
    assert deleted_replay_exc.value.detail == {
        "code": "review_disposition_action_deleted",
        "review_object_key": key,
    }
    assert len(deletion_verifications) == 1
    assert len(captured_training_payloads) == 2
    assert all(
        captured_job_id == job.job_id
        and captured_payload["single_bbox_deletion_attestation"]
        == attestation
        for captured_job_id, captured_payload in captured_training_payloads
    )
    assert overlaid["wrong_class_candidates"] == []
    assert overlaid["points"][0]["human_review_disposition"] == "delete_bbox"
    assert overlaid["summary"]["human_review_disposition_counts"] == {
        "delete_bbox": 1,
    }


def test_single_bbox_duplicate_proofs_with_new_timestamps_replay_idempotently(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    review_key = "cro_" + ("9a" * 32)
    point = {
        **_record("deleted", "Boat"),
        "review_object_key": review_key,
        "single_bbox_deletion_attestation": None,
    }
    target = {
        "source_mode": "linked",
        "source_id": "dataset",
        "split": "train",
        "image_relpath": "deleted.jpg",
    }

    def proof(verified_at):
        value = {
            "schema": (
                api.CLASS_ANALYSIS_SINGLE_BBOX_DELETION_ATTESTATION_SCHEMA
            ),
            "committed": True,
            "analysis_job_id": "ca_stable_delete",
            "point_id": "deleted",
            "review_object_key": review_key,
            "annotation_target": target,
            "source_identity": f"asi1_{'12' * 32}",
            "before_revision": f"alr1_{'34' * 32}",
            "committed_revision": f"alr1_{'56' * 32}",
            "image_sha256": "78" * 32,
            "committed_label_count": 3,
            "verification_method": "exact_frozen_geometry_absent_v1",
            "verified_at": verified_at,
        }
        value["attestation_sha256"] = (
            api._class_analysis_single_bbox_deletion_attestation_hash(value)
        )
        return value

    first_proof = proof(100.0)
    second_proof = proof(101.0)
    first = api._class_analysis_record_review_disposition_entry(
        result={"summary": {"analysis_job_id": "ca_stable_delete"}},
        point={
            **point,
            "single_bbox_deletion_attestation": first_proof,
        },
        disposition="delete_bbox",
        origin="desktop",
        client_action_id="stable-delete-action",
    )
    replay = api._class_analysis_record_review_disposition_entry(
        result={"summary": {"analysis_job_id": "ca_stable_delete"}},
        point={
            **point,
            "single_bbox_deletion_attestation": second_proof,
        },
        disposition="delete_bbox",
        origin="desktop",
        client_action_id="stable-delete-action",
    )

    assert replay["_idempotent_replay"] is True
    assert replay["entry_revision"] == first["entry_revision"]
    assert replay["single_bbox_deletion_attestation"] == first_proof


def test_single_bbox_delete_reconciles_persisted_absence_idempotently(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    (class_root / "ca_absent_bbox").mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    review_key = "cro_" + ("7c" * 32)
    point = {
        **_record("already-gone", "Boat"),
        "image_relpath": "already-gone.jpg",
        "image_sha256": "dc" * 32,
        "bbox_xyxy": [1, 2, 30, 40],
        "review_object_key": review_key,
        "is_wrong_class_candidate": True,
    }
    job = api.ClassAnalysisJob(
        job_id="ca_absent_bbox",
        status="completed",
        summary={
            "analysis_job_id": "ca_absent_bbox",
            "source_mode": "linked",
            "source_id": "dataset",
            "source_key": "linked:dataset",
        },
    )
    api.CLASS_ANALYSIS_JOBS[job.job_id] = job
    monkeypatch.setattr(
        api,
        "_class_analysis_thumbnail_point_metadata",
        lambda *_args, **_kwargs: dict(point),
    )
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "dataset",
            "dataset_root": str(tmp_path / "dataset"),
        },
    )
    annotation_target = {
        "source_mode": "linked",
        "source_id": "dataset",
        "split": "train",
        "image_relpath": "already-gone.jpg",
    }
    current_revision = f"alr1_{'34' * 32}"
    source_identity = f"asi1_{'12' * 32}"
    attestation = {
        "schema": api.CLASS_ANALYSIS_SINGLE_BBOX_DELETION_ATTESTATION_SCHEMA,
        "committed": True,
        "deletion_state": "already_absent",
        "analysis_job_id": job.job_id,
        "point_id": point["point_id"],
        "review_object_key": review_key,
        "annotation_target": annotation_target,
        "source_identity": source_identity,
        "before_revision": current_revision,
        "committed_revision": current_revision,
        "image_sha256": point["image_sha256"],
        "committed_label_count": 0,
        "verification_method": (
            api.CLASS_ANALYSIS_SINGLE_BBOX_ALREADY_ABSENT_VERIFICATION_METHOD
        ),
        "verified_at": 1.0,
    }
    attestation["attestation_sha256"] = (
        api._class_analysis_single_bbox_deletion_attestation_hash(
            attestation
        )
    )
    verification_calls = []

    def verify_absence(**kwargs):
        verification_calls.append(kwargs)
        return dict(attestation)

    monkeypatch.setattr(
        api,
        "_class_analysis_validate_single_bbox_deletion_commit",
        verify_absence,
    )
    captured_training_payloads = []

    def capture_training_action(job_id, payload):
        captured_training_payloads.append((job_id, payload))
        return {"status": "recorded", "recorded_count": 1}

    monkeypatch.setattr(
        api,
        "record_class_analysis_vignette_training_action",
        capture_training_action,
    )
    request = {
        "disposition": "delete_bbox",
        "origin": "desktop",
        "client_action_id": "reconcile-absent-bbox",
        "capture_training_data": True,
        "label_commit_status": "already_absent",
        "annotation_target": annotation_target,
        "annotation_before_revision": current_revision,
        "annotation_commit_revision": current_revision,
        "annotation_source_identity": source_identity,
    }
    try:
        receipt = api.record_class_analysis_review_disposition(
            job.job_id,
            point["point_id"],
            request,
        )
        replay = api.record_class_analysis_review_disposition(
            job.job_id,
            point["point_id"],
            request,
        )
        overlaid = api._class_analysis_apply_review_dispositions(
            {
                "summary": dict(job.summary),
                "points": [dict(point)],
                "wrong_class_candidates": [
                    {
                        "point_id": point["point_id"],
                        "review_object_key": review_key,
                    }
                ],
            },
            job=job,
        )
    finally:
        api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)

    assert receipt["status"] == "already_absent"
    assert receipt["annotation_state"] == "already_absent"
    assert receipt["annotation_commit_attestation"] == attestation
    assert replay["status"] == "already_absent"
    assert replay["idempotent_replay"] is True
    assert replay["human_review_revision"] == receipt[
        "human_review_revision"
    ]
    assert replay["annotation_commit_attestation"] == attestation
    assert len(verification_calls) == 1
    assert verification_calls[0]["allow_already_absent"] is True
    assert overlaid["wrong_class_candidates"] == []
    assert overlaid["points"][0]["human_review_disposition"] == (
        "delete_bbox"
    )
    assert len(captured_training_payloads) == 2
    assert all(
        captured_job_id == job.job_id
        and captured_payload["label_commit_status"] == "already_absent"
        and captured_payload["single_bbox_deletion_attestation"]
        == attestation
        for captured_job_id, captured_payload in captured_training_payloads
    )


def test_review_action_tombstones_are_scoped_to_object_within_one_shard(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", tmp_path / "class_analysis")
    points = [
        {
            **_record("p0", "Boat"),
            "review_object_key": "cro_aa" + ("1" * 62),
        },
        {
            **_record("p1", "Bike"),
            "review_object_key": "cro_aa" + ("2" * 62),
        },
    ]
    entries = [
        api._class_analysis_record_review_disposition_entry(
            result={"summary": {"analysis_job_id": "ca_token_scope"}},
            point=point,
            disposition="skip",
            origin="desktop",
            client_action_id="shared-action-token",
        )
        for point in points
    ]

    api._class_analysis_clear_review_disposition_entry(
        points[0]["review_object_key"],
        expected_revision=entries[0]["entry_revision"],
        expected_reviewed_at=None,
    )
    replay = api._class_analysis_record_review_disposition_entry(
        result={"summary": {"analysis_job_id": "ca_token_scope"}},
        point=points[1],
        disposition="skip",
        origin="desktop",
        client_action_id="shared-action-token",
    )

    assert replay["_idempotent_replay"] is True
    assert replay["review_object_key"] == points[1]["review_object_key"]
    with pytest.raises(api.HTTPException) as deleted_action_exc:
        api._class_analysis_record_review_disposition_entry(
            result={"summary": {"analysis_job_id": "ca_token_scope"}},
            point=points[0],
            disposition="skip",
            origin="desktop",
            client_action_id="shared-action-token",
        )
    assert deleted_action_exc.value.detail["code"] == (
        "review_disposition_action_deleted"
    )


def test_legacy_unverified_single_bbox_delete_does_not_hide_candidate(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    review_key = "cro_" + ("8b" * 32)
    point = {
        **_record("legacy-delete", "Boat"),
        "review_object_key": review_key,
        "is_wrong_class_candidate": True,
    }
    api._class_analysis_record_review_disposition_entry(
        result={"summary": {"analysis_job_id": "ca_legacy_delete"}},
        point=point,
        disposition="delete_bbox",
        origin="legacy_import",
    )

    overlaid = api._class_analysis_apply_review_dispositions(
        {
            "summary": {"analysis_job_id": "ca_legacy_delete"},
            "points": [dict(point)],
            "wrong_class_candidates": [
                {
                    "point_id": point["point_id"],
                    "review_object_key": review_key,
                }
            ],
        }
    )

    assert [
        candidate["point_id"]
        for candidate in overlaid["wrong_class_candidates"]
    ] == [point["point_id"]]
    assert overlaid["summary"][
        "unverified_single_bbox_deletion_history_count"
    ] == 1
    assert overlaid["summary"]["human_reviewed_candidate_count"] == 0


def test_class_analysis_reassignment_is_verified_job_scoped_and_history_only(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    job_dir = class_root / "ca_reassign"
    job_dir.mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    source_key = api._class_analysis_review_object_key(
        source_key="linked:dataset",
        image_sha256="ad" * 32,
        split="train",
        image_relpath="reassign.jpg",
        class_name="Boat",
        geometry={"kind": "bbox", "bbox_xyxy": [1, 2, 30, 40]},
    )
    point = {
        **_record("reassigned", "Boat"),
        "image_sha256": "ad" * 32,
        "image_width": 64,
        "image_height": 64,
        "image_relpath": "reassign.jpg",
        "bbox_xyxy": [1, 2, 30, 40],
        "review_object_key": source_key,
        "source_mode": "linked",
        "source_id": "dataset",
        "source_key": "linked:dataset",
        "is_wrong_class_candidate": True,
    }
    before_line = api._class_analysis_browser_bbox_label_line(
        class_id=0,
        bbox_xyxy=point["bbox_xyxy"],
        image_width=point["image_width"],
        image_height=point["image_height"],
    )
    committed_line = api._class_analysis_browser_bbox_label_line(
        class_id=1,
        bbox_xyxy=point["bbox_xyxy"],
        image_width=point["image_width"],
        image_height=point["image_height"],
    )
    person_line = api._class_analysis_browser_bbox_label_line(
        class_id=2,
        bbox_xyxy=point["bbox_xyxy"],
        image_width=point["image_width"],
        image_height=point["image_height"],
    )
    point["label_line"] = before_line
    before_revision = api._annotation_image_label_revision([before_line])
    committed_revision = api._annotation_image_label_revision(
        [committed_line]
    )
    source_identity = "asi1_" + ("7d" * 32)
    job = api.ClassAnalysisJob(
        job_id="ca_reassign",
        status="completed",
        summary={
            "analysis_job_id": "ca_reassign",
            "source_mode": "linked",
            "source_id": "dataset",
            "source_key": "linked:dataset",
            "labelmap": ["Boat", "Building", "Person"],
        },
    )
    api.CLASS_ANALYSIS_JOBS[job.job_id] = job
    monkeypatch.setattr(
        api,
        "_class_analysis_thumbnail_point_metadata",
        lambda *_args, **_kwargs: dict(point),
    )
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "dataset",
            "dataset_root": str(tmp_path / "dataset"),
        },
    )
    annotation_target = {
        "source_mode": "linked",
        "source_id": "dataset",
        "split": "train",
        "image_relpath": "reassign.jpg",
    }
    annotation_state = {
        "target": annotation_target,
        "current_lines": [committed_line],
        "labelmap": ["Boat", "Building", "Person"],
        "image_width": point["image_width"],
        "image_height": point["image_height"],
        "source_identity": source_identity,
        "current_revision": committed_revision,
        "image_sha256": point["image_sha256"],
    }
    monkeypatch.setattr(
        api,
        "_class_analysis_reassignment_annotation_state",
        lambda _target: {
            **annotation_state,
            "current_lines": list(annotation_state["current_lines"]),
        },
    )
    geometry_edit = {
        "schema": api.CLASS_ANALYSIS_BBOX_GEOMETRY_EDIT_SCHEMA,
        "analysis_bbox_xyxy": list(point["bbox_xyxy"]),
        "edited_bbox_xyxy": list(point["bbox_xyxy"]),
        "changed": False,
    }
    request = {
        "disposition": "reassign_class",
        "origin": "desktop",
        "client_action_id": "reassign-boat-building",
        "label_commit_status": "committed",
        "target_class": "Building",
        "annotation_target": annotation_target,
        "annotation_before_revision": before_revision,
        "annotation_commit_revision": committed_revision,
        "annotation_source_identity": source_identity,
        "geometry_edit": geometry_edit,
    }
    try:
        with pytest.raises(api.HTTPException) as uncommitted_exc:
            api.record_class_analysis_review_disposition(
                job.job_id,
                point["point_id"],
                {**request, "label_commit_status": "pending"},
            )
        assert uncommitted_exc.value.status_code == 400
        assert uncommitted_exc.value.detail == (
            "review_reassignment_label_commit_required"
        )

        with pytest.raises(api.HTTPException) as target_mismatch_exc:
            api.record_class_analysis_review_disposition(
                job.job_id,
                point["point_id"],
                {**request, "target_class": "Person"},
            )
        assert target_mismatch_exc.value.status_code == 409
        assert target_mismatch_exc.value.detail["code"] == (
            "review_reassignment_annotation_unverified"
        )

        with pytest.raises(api.HTTPException) as source_target_exc:
            api.record_class_analysis_review_disposition(
                job.job_id,
                point["point_id"],
                {**request, "target_class": "Boat"},
            )
        assert source_target_exc.value.status_code == 400
        assert source_target_exc.value.detail == (
            "review_reassignment_target_invalid"
        )

        with pytest.raises(api.HTTPException) as annotation_target_exc:
            api.record_class_analysis_review_disposition(
                job.job_id,
                point["point_id"],
                {
                    **request,
                    "annotation_target": {
                        **annotation_target,
                        "image_relpath": "other.jpg",
                    },
                },
            )
        assert annotation_target_exc.value.status_code == 409
        assert annotation_target_exc.value.detail == (
            "review_reassignment_annotation_target_mismatch"
        )

        receipt = api.record_class_analysis_review_disposition(
            job.job_id,
            point["point_id"],
            request,
        )
        replay = api.record_class_analysis_review_disposition(
            job.job_id,
            point["point_id"],
            request,
        )
        assert replay["idempotent_replay"] is True
        assert replay["human_review_revision"] == receipt["human_review_revision"]
        assert receipt["target_class"] == "Building"
        assert receipt["review_object_key"].startswith("crj_")

        annotation_state["current_lines"] = [person_line]
        annotation_state["current_revision"] = (
            api._annotation_image_label_revision([person_line])
        )
        late_replay = api.record_class_analysis_review_disposition(
            job.job_id,
            point["point_id"],
            request,
        )
        assert late_replay["idempotent_replay"] is True
        assert late_replay["human_review_revision"] == receipt[
            "human_review_revision"
        ]
        annotation_state["current_lines"] = [committed_line]
        annotation_state["current_revision"] = committed_revision

        reassignment_key = api._class_analysis_review_reassignment_key(
            source_key,
            job.job_id,
        )
        stored = api._class_analysis_lookup_review_dispositions(
            [reassignment_key]
        )[reassignment_key]
        assert stored["source_review_object_key"] == source_key
        assert stored["target_class"] == "Building"
        assert stored["annotation_target"] == annotation_target

        same_job = api._class_analysis_apply_review_dispositions(
            {
                "summary": dict(job.summary),
                "points": [dict(point)],
                "wrong_class_candidates": [
                    {
                        "point_id": point["point_id"],
                        "review_object_key": source_key,
                    }
                ],
            },
            job=job,
        )
        assert same_job["wrong_class_candidates"] == []
        assert same_job["points"][0]["human_review_disposition"] == (
            "reassign_class"
        )
        assert same_job["points"][0]["human_review_before_class"] == "Boat"
        assert same_job["points"][0]["human_review_target_class"] == (
            "Building"
        )

        # A later analysis of an intentionally changed-back Boat must not be
        # hidden by this old A->B correction.
        fresh_job = api.ClassAnalysisJob(
            job_id="ca_reassign_later",
            status="completed",
            summary={
                **job.summary,
                "analysis_job_id": "ca_reassign_later",
            },
        )
        later = api._class_analysis_apply_review_dispositions(
            {
                "summary": dict(fresh_job.summary),
                "points": [dict(point)],
                "wrong_class_candidates": [
                    {
                        "point_id": point["point_id"],
                        "review_object_key": source_key,
                    }
                ],
            },
            job=fresh_job,
        )
        assert len(later["wrong_class_candidates"]) == 1
        assert "human_review_disposition" not in later["points"][0]

        deleted = api.delete_class_analysis_review_history(
            job.job_id,
            {
                "schema": api.CLASS_ANALYSIS_REVIEW_HISTORY_DELETE_SCHEMA,
                "client_action_id": "delete-reassignment-history",
                "entries": [
                    {
                        "point_id": point["point_id"],
                        "expected_disposition": "reassign_class",
                        "expected_revision": receipt[
                            "human_review_revision"
                        ],
                    }
                ],
            },
        )
        assert deleted["deleted_count"] == 1
        assert deleted["labels_changed"] is False
        assert deleted["annotations_changed"] is False
        assert api._class_analysis_lookup_review_dispositions(
            [reassignment_key]
        ) == {}
        assert annotation_state["current_lines"] == [committed_line]
    finally:
        api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)


@pytest.mark.parametrize(
    ("transition_mode", "before_line"),
    [
        (
            "geometry_precommitted",
            "0 0.2400000 0.260000 0.240000 0.280000",
        ),
        (
            "geometry_and_class_same_commit",
            "0 0.2 0.2 0.2 0.2",
        ),
    ],
)
def test_class_reassignment_attests_server_geometry_transitions(
    monkeypatch,
    transition_mode,
    before_line,
):
    source_key = "cro_" + ("3a" * 32)
    image_sha256 = "ab" * 32
    point = {
        "point_id": "edited-reassignment",
        "review_object_key": source_key,
        "class_name": "Boat",
        "source_mode": "linked",
        "source_id": "dataset",
        "source_key": "linked:dataset",
        "split": "train",
        "image_relpath": "edited.jpg",
        "image_sha256": image_sha256,
        "image_width": 100,
        "image_height": 100,
        "bbox_xyxy": [10, 10, 30, 30],
        "label_line": "0 0.2 0.2 0.2 0.2",
    }
    summary = {
        "analysis_job_id": "ca_geometry_reassign",
        "source_mode": "linked",
        "source_id": "dataset",
        "source_key": "linked:dataset",
        "labelmap": ["Boat", "Building", "Person"],
    }
    job = api.ClassAnalysisJob(
        job_id="ca_geometry_reassign",
        status="completed",
        summary=summary,
    )
    target = {
        "source_mode": "linked",
        "source_id": "dataset",
        "split": "train",
        "image_relpath": "edited.jpg",
    }
    committed_line = "1 0.2400000 0.260000 0.240000 0.280000"
    source_identity = "asi1_" + ("5b" * 32)
    before_revision = api._annotation_image_label_revision(
        [before_line, "2 0.8 0.8 0.1 0.1"]
    )
    current_lines = [committed_line, "2 0.8 0.8 0.1 0.1"]
    committed_revision = api._annotation_image_label_revision(current_lines)
    state = {
        "target": target,
        "current_lines": current_lines,
        "labelmap": ["Boat", "Building", "Person"],
        "image_width": 100,
        "image_height": 100,
        "source_identity": source_identity,
        "current_revision": committed_revision,
        "image_sha256": image_sha256,
    }
    monkeypatch.setattr(
        api,
        "_class_analysis_reassignment_annotation_state",
        lambda _target: dict(state),
    )
    geometry_edit = {
        "schema": api.CLASS_ANALYSIS_BBOX_GEOMETRY_EDIT_SCHEMA,
        "analysis_bbox_xyxy": [10, 10, 30, 30],
        "edited_bbox_xyxy": [12, 12, 36, 40],
        "changed": True,
    }

    attestation = api._class_analysis_validate_class_reassignment_commit(
        job=job,
        point=point,
        summary=summary,
        annotation_target=target,
        before_revision=before_revision,
        committed_revision=committed_revision,
        expected_source_identity=source_identity,
        target_class="Building",
        geometry_edit=geometry_edit,
    )

    assert attestation["transition_mode"] == transition_mode
    assert attestation["geometry_edit"]["analysis_bbox_xyxy"] == [
        10.0,
        10.0,
        30.0,
        30.0,
    ]
    assert attestation["geometry_edit"]["committed_bbox_xyxy"] == [
        12.0,
        12.0,
        36.0,
        40.0,
    ]
    assert attestation["geometry_edit"]["changed"] is True
    assert attestation["attestation_sha256"] == (
        api._class_analysis_class_reassignment_attestation_hash(attestation)
    )
    source_geometry_edit = api._class_analysis_bbox_geometry_edit_contract(
        geometry_edit,
        point_snapshot=api._class_analysis_vignette_training_point_snapshot(
            point,
            summary,
        ),
        image_width=100,
        image_height=100,
    )
    source_event = {
        "analysis_job_id": job.job_id,
        "review_object_key": source_key,
        "visual_object_key": "cvo_geometry_edit",
        "before_class": "Boat",
        "after_class": "Building",
        "annotation_target": target,
        "geometry_edit": source_geometry_edit,
        "point": api._class_analysis_vignette_training_point_snapshot(
            point,
            summary,
        ),
    }
    commit_event = {
        "analysis_job_id": job.job_id,
        "visual_object_key": "cvo_geometry_edit",
        "before_class": "Boat",
        "after_class": "Building",
        "annotation_target": target,
        "geometry_edit": attestation["geometry_edit"],
    }
    assert api._class_analysis_vignette_training_commit_attestation_valid(
        attestation,
        commit_event=commit_event,
        source_event=source_event,
    ) is True

    state["current_lines"] = [committed_line, committed_line]
    state["current_revision"] = api._annotation_image_label_revision(
        state["current_lines"]
    )
    with pytest.raises(api.HTTPException) as duplicate_exc:
        api._class_analysis_validate_class_reassignment_commit(
            job=job,
            point=point,
            summary=summary,
            annotation_target=target,
            before_revision=before_revision,
            committed_revision=state["current_revision"],
            expected_source_identity=source_identity,
            target_class="Building",
            geometry_edit=geometry_edit,
        )
    assert duplicate_exc.value.status_code == 409
    assert duplicate_exc.value.detail["code"] == (
        "review_reassignment_annotation_unverified"
    )

    state["current_lines"] = [
        committed_line,
        "2 0.75 0.75 0.1 0.1",
    ]
    state["current_revision"] = api._annotation_image_label_revision(
        state["current_lines"]
    )
    with pytest.raises(api.HTTPException) as unrelated_edit_exc:
        api._class_analysis_validate_class_reassignment_commit(
            job=job,
            point=point,
            summary=summary,
            annotation_target=target,
            before_revision=before_revision,
            committed_revision=state["current_revision"],
            expected_source_identity=source_identity,
            target_class="Building",
            geometry_edit=geometry_edit,
        )
    assert unrelated_edit_exc.value.detail["code"] == (
        "review_reassignment_transition_unverified"
    )

    state["current_lines"] = current_lines
    state["current_revision"] = committed_revision
    with pytest.raises(api.HTTPException) as stale_source_exc:
        api._class_analysis_validate_class_reassignment_commit(
            job=job,
            point=point,
            summary=summary,
            annotation_target=target,
            before_revision=before_revision,
            committed_revision=committed_revision,
            expected_source_identity="asi1_" + ("9f" * 32),
            target_class="Building",
            geometry_edit=geometry_edit,
        )
    assert stale_source_exc.value.detail == (
        "review_reassignment_commit_stale"
    )

    with pytest.raises(api.HTTPException) as malformed_geometry_exc:
        api._class_analysis_validate_class_reassignment_commit(
            job=job,
            point=point,
            summary=summary,
            annotation_target=target,
            before_revision=before_revision,
            committed_revision=committed_revision,
            expected_source_identity=source_identity,
            target_class="Building",
            geometry_edit={
                **geometry_edit,
                "edited_bbox_xyxy": [36, 12, 12, 40],
            },
        )
    assert malformed_geometry_exc.value.detail == (
        "review_reassignment_geometry_edit_invalid"
    )


def test_resized_class_reassignment_retry_accepts_yolo_quantized_geometry(
    tmp_path,
    monkeypatch,
):
    """A lost response must not turn harmless YOLO rounding into a conflict."""

    class_root = tmp_path / "class_analysis"
    (class_root / "ca_quantized_reassign").mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    image_sha256 = "ac" * 32
    source_key = api._class_analysis_review_object_key(
        source_key="linked:dataset",
        image_sha256=image_sha256,
        split="train",
        image_relpath="quantized.jpg",
        class_name="Boat",
        geometry={"kind": "bbox", "bbox_xyxy": [100, 100, 300, 300]},
    )
    point = {
        **_record("quantized-reassignment", "Boat"),
        "review_object_key": source_key,
        "source_mode": "linked",
        "source_id": "dataset",
        "source_key": "linked:dataset",
        "split": "train",
        "image_relpath": "quantized.jpg",
        "image_sha256": image_sha256,
        "image_width": 1000,
        "image_height": 1000,
        "bbox_xyxy": [100, 100, 300, 300],
        "label_line": "0 0.2 0.2 0.2 0.2",
    }
    job = api.ClassAnalysisJob(
        job_id="ca_quantized_reassign",
        status="completed",
        summary={
            "analysis_job_id": "ca_quantized_reassign",
            "source_mode": "linked",
            "source_id": "dataset",
            "source_key": "linked:dataset",
            "labelmap": ["Boat", "Building", "Person"],
        },
    )
    api.CLASS_ANALYSIS_JOBS[job.job_id] = job
    monkeypatch.setattr(
        api,
        "_class_analysis_thumbnail_point_metadata",
        lambda *_args, **_kwargs: dict(point),
    )
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "dataset",
            "dataset_root": str(tmp_path / "dataset"),
        },
    )
    annotation_target = {
        "source_mode": "linked",
        "source_id": "dataset",
        "split": "train",
        "image_relpath": "quantized.jpg",
    }
    before_line = "0 0.24 0.26 0.24 0.28"
    committed_line = "1 0.24 0.26 0.24 0.28"
    companion_line = "2 0.8 0.8 0.1 0.1"
    before_revision = api._annotation_image_label_revision(
        [before_line, companion_line]
    )
    current_lines = [committed_line, companion_line]
    committed_revision = api._annotation_image_label_revision(current_lines)
    source_identity = "asi1_" + ("5c" * 32)
    monkeypatch.setattr(
        api,
        "_class_analysis_reassignment_annotation_state",
        lambda _target: {
            "target": annotation_target,
            "current_lines": list(current_lines),
            "labelmap": ["Boat", "Building", "Person"],
            "image_width": 1000,
            "image_height": 1000,
            "source_identity": source_identity,
            "current_revision": committed_revision,
            "image_sha256": image_sha256,
        },
    )
    # These are the in-memory browser coordinates before its YOLO serializer
    # rounds normalized values to six decimals. The persisted row resolves to
    # [120, 120, 360, 400], but both geometries have one canonical identity.
    geometry_edit = {
        "schema": api.CLASS_ANALYSIS_BBOX_GEOMETRY_EDIT_SCHEMA,
        "analysis_bbox_xyxy": [100, 100, 300, 300],
        "edited_bbox_xyxy": [119.9997, 120.0002, 360.0001, 400.0003],
        "changed": True,
    }
    request = {
        "disposition": "reassign_class",
        "origin": "desktop",
        "client_action_id": "quantized-reassign-retry",
        "label_commit_status": "committed",
        "target_class": "Building",
        "annotation_target": annotation_target,
        "annotation_before_revision": before_revision,
        "annotation_commit_revision": committed_revision,
        "annotation_source_identity": source_identity,
        "geometry_edit": geometry_edit,
    }
    try:
        receipt = api.record_class_analysis_review_disposition(
            job.job_id,
            point["point_id"],
            request,
        )
        replay = api.record_class_analysis_review_disposition(
            job.job_id,
            point["point_id"],
            request,
        )
    finally:
        api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)

    assert receipt["idempotent_replay"] is False
    assert receipt["geometry_edit"]["committed_bbox_xyxy"] == [
        120.0,
        120.0,
        360.0,
        400.0,
    ]
    assert replay["idempotent_replay"] is True
    assert replay["human_review_revision"] == receipt[
        "human_review_revision"
    ]


def test_class_analysis_reassignment_atomically_supersedes_source_review(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    class_root.mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    source_key = api._class_analysis_review_object_key(
        source_key="linked:dataset",
        image_sha256="bc" * 32,
        split="train",
        image_relpath="atomic.jpg",
        class_name="Boat",
        geometry={"kind": "bbox", "bbox_xyxy": [1, 2, 30, 40]},
    )
    result = {
        "summary": {
            "analysis_job_id": "ca_reassign_atomic",
            "source_mode": "linked",
            "source_id": "dataset",
            "source_key": "linked:dataset",
        }
    }
    source_point = {
        **_record("atomic", "Boat"),
        "review_object_key": source_key,
        "image_relpath": "atomic.jpg",
        "bbox_xyxy": [1, 2, 30, 40],
    }
    confirmed = api._class_analysis_record_review_disposition_entry(
        result=result,
        point=source_point,
        disposition="confirm_current",
        origin="desktop",
        client_action_id="atomic-confirm-source",
    )
    reassignment_key = api._class_analysis_review_reassignment_key(
        source_key,
        "ca_reassign_atomic",
    )
    assert api._class_analysis_review_disposition_shard(source_key)[0] == (
        api._class_analysis_review_disposition_shard(reassignment_key)[0]
    )
    reassigned = api._class_analysis_record_review_disposition_entry(
        result=result,
        point={
            **source_point,
            "review_object_key": reassignment_key,
            "source_review_object_key": source_key,
            "review_target_class": "Building",
            "review_annotation_target": {
                "source_mode": "linked",
                "source_id": "dataset",
                "split": "train",
                "image_relpath": "atomic.jpg",
            },
        },
        disposition="reassign_class",
        origin="desktop",
        client_action_id="atomic-reassign-source",
    )
    assert reassigned["superseded_source_disposition"] == "confirm_current"
    assert reassigned["superseded_source_revision"] == confirmed[
        "entry_revision"
    ]
    assert api._class_analysis_lookup_review_dispositions(
        [source_key, reassignment_key]
    ) == {reassignment_key: reassigned}

    with pytest.raises(api.HTTPException) as stale_review_exc:
        api._class_analysis_record_review_disposition_entry(
            result=result,
            point=source_point,
            disposition="skip",
            origin="desktop",
            client_action_id="atomic-stale-skip",
        )
    assert stale_review_exc.value.status_code == 409
    assert stale_review_exc.value.detail["code"] == (
        "review_disposition_changed"
    )


def test_class_analysis_review_disposition_post_uses_indexed_point_fast_path(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    (class_root / "ca_fast").mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    key = api._class_analysis_review_object_key(
        source_key="linked:dataset",
        image_sha256="33" * 32,
        split="train",
        image_relpath="frame.jpg",
        class_name="Boat",
        geometry={"kind": "bbox", "bbox_xyxy": [1, 2, 3, 4]},
    )
    api.CLASS_ANALYSIS_JOBS["ca_fast"] = api.ClassAnalysisJob(
        job_id="ca_fast",
        status="completed",
        summary={
            "source_mode": "linked",
            "source_id": "dataset",
            "source_key": "linked:dataset",
        },
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_thumbnail_point_metadata",
        lambda *_args, **_kwargs: {
            "point_id": "p0",
            "review_object_key": key,
            "split": "train",
            "image_relpath": "frame.jpg",
            "class_name": "Boat",
            "bbox_xyxy": [1, 2, 3, 4],
            "wrong_class_suspicion": 0.8,
            "wrong_class_review_reason": "embedding_outlier",
            "embedding_wrong_class_suspicion": 0.75,
            "same_class_neighbor_ratio": 0.1,
            "top_other_neighbor_ratio": 0.9,
            "suggested_neighbor_class": "Building",
            "neighbor_class_counts": {"Boat": 1, "Building": 9},
            "review_signals": ["wrong_class"],
        },
    )
    monkeypatch.setattr(
        api,
        "_safe_job_result_json_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("indexed review action must not load result.json")
        ),
    )

    try:
        with pytest.raises(api.HTTPException) as missing_client_exc_info:
            api.record_class_analysis_review_disposition(
                "ca_fast",
                "p0",
                {"disposition": "confirm_current"},
            )
        assert missing_client_exc_info.value.status_code == 400
        assert missing_client_exc_info.value.detail == (
            "review_disposition_client_action_id_invalid"
        )
        response = api.record_class_analysis_review_disposition(
            "ca_fast",
            "p0",
            {
                "disposition": "confirm_current",
                "origin": "desktop",
                "client_action_id": "review-fast-confirm",
            },
        )
        stored = api._class_analysis_lookup_review_dispositions([key])[key]
        replayed = api.record_class_analysis_review_disposition(
            "ca_fast",
            "p0",
            {
                "disposition": "confirm_current",
                "origin": "desktop",
                "client_action_id": "review-fast-confirm",
            },
        )
        assert replayed["idempotent_replay"] is True
        assert (
            replayed["human_review_revision"]
            == response["human_review_revision"]
        )
        with pytest.raises(api.HTTPException) as consent_conflict_exc_info:
            api.record_class_analysis_review_disposition(
                "ca_fast",
                "p0",
                {
                    "disposition": "confirm_current",
                    "origin": "desktop",
                    "capture_training_data": True,
                    "client_action_id": "review-fast-confirm",
                },
            )
        assert consent_conflict_exc_info.value.status_code == 409
        assert consent_conflict_exc_info.value.detail == (
            "review_disposition_client_action_conflict"
        )
        with pytest.raises(api.HTTPException) as replay_conflict_exc_info:
            api.record_class_analysis_review_disposition(
                "ca_fast",
                "p0",
                {
                    "disposition": "confirm_current",
                    "origin": "external",
                    "client_action_id": "review-fast-confirm",
                },
            )
        assert replay_conflict_exc_info.value.status_code == 409
        assert replay_conflict_exc_info.value.detail == (
            "review_disposition_client_action_conflict"
        )
        with pytest.raises(api.HTTPException) as invalid_job_exc_info:
            api.record_class_analysis_review_disposition(
                "ca fast",
                "p0",
                {
                    "disposition": "skip",
                    "client_action_id": "review-invalid-job",
                },
            )
        assert invalid_job_exc_info.value.status_code == 400
        assert invalid_job_exc_info.value.detail == "job_id_invalid"
        with pytest.raises(api.HTTPException) as concurrent_write_exc_info:
            api.record_class_analysis_review_disposition(
                "ca_fast",
                "p0",
                {
                    "disposition": "skip",
                    "origin": "external",
                    "client_action_id": "review-newer-external",
                },
            )
        assert concurrent_write_exc_info.value.status_code == 409
        assert concurrent_write_exc_info.value.detail == {
            "code": "review_disposition_changed",
            "review_object_key": key,
        }
        assert api._class_analysis_lookup_review_dispositions([key])[key][
            "entry_revision"
        ] == response["human_review_revision"]
        with pytest.raises(api.HTTPException) as missing_cas_exc_info:
            api.record_class_analysis_review_disposition(
                "ca_fast",
                "p0",
                {
                    "disposition": "clear",
                    "client_action_id": "review-clear-no-cas",
                },
            )
        assert missing_cas_exc_info.value.status_code == 400
        first_cleared = api.record_class_analysis_review_disposition(
            "ca_fast",
            "p0",
            {
                "disposition": "clear",
                "client_action_id": "review-clear-first",
                "expected_revision": response["human_review_revision"],
            },
        )
        assert first_cleared["status"] == "cleared"
        assert first_cleared["training_capture_requested"] is False
        newer = api.record_class_analysis_review_disposition(
            "ca_fast",
            "p0",
            {
                "disposition": "skip",
                "origin": "external",
                "client_action_id": "review-newer-external",
            },
        )
        with pytest.raises(api.HTTPException) as delayed_retry_exc_info:
            api.record_class_analysis_review_disposition(
                "ca_fast",
                "p0",
                {
                    "disposition": "confirm_current",
                    "origin": "desktop",
                    "client_action_id": "review-fast-confirm",
                },
            )
        assert delayed_retry_exc_info.value.status_code == 409
        assert api._class_analysis_lookup_review_dispositions([key])[key][
            "entry_revision"
        ] == newer["human_review_revision"]
        with pytest.raises(api.HTTPException) as stale_exc_info:
            api.record_class_analysis_review_disposition(
                "ca_fast",
                "p0",
                {
                    "disposition": "clear",
                    "client_action_id": "review-clear-stale",
                    "expected_revision": response[
                        "human_review_revision"
                    ],
                },
            )
        assert stale_exc_info.value.status_code == 409
        assert api._class_analysis_lookup_review_dispositions([key])[key][
            "entry_revision"
        ] == newer["human_review_revision"]
        cleared = api.record_class_analysis_review_disposition(
            "ca_fast",
            "p0",
            {
                "disposition": "clear",
                "origin": "desktop",
                "client_action_id": "review-fast-clear",
                "expected_revision": newer["human_review_revision"],
            },
        )
    finally:
        api.CLASS_ANALYSIS_JOBS.pop("ca_fast", None)

    assert response["review_object_key"] == key
    assert response["schema"] == (
        api.CLASS_ANALYSIS_REVIEW_DISPOSITION_RECEIPT_SCHEMA
    )
    assert response["client_action_id"] == "review-fast-confirm"
    assert response["disposition"] == "confirm_current"
    assert stored["analysis_job_id"] == "ca_fast"
    assert stored["raw_wrong_class_suspicion"] == 0.8
    assert stored["raw_wrong_class_review_reason"] == "embedding_outlier"
    assert stored["raw_review_evidence"] == {
        "embedding_wrong_class_suspicion": 0.75,
        "same_class_neighbor_ratio": 0.1,
        "top_other_neighbor_ratio": 0.9,
        "suggested_neighbor_class": "Building",
        "neighbor_class_counts": {"Boat": 1, "Building": 9},
        "review_signals": ["wrong_class"],
        "is_close_overlap_candidate": None,
        "close_overlap_matches": None,
        "is_dual_bbox_conflict": None,
        "dual_bbox_conflict": None,
    }
    assert cleared["status"] == "cleared"
    assert cleared["schema"] == (
        api.CLASS_ANALYSIS_REVIEW_DISPOSITION_RECEIPT_SCHEMA
    )
    assert cleared["client_action_id"] == "review-fast-clear"
    assert cleared["previous_disposition"] == "skip"
    assert api._class_analysis_lookup_review_dispositions([key]) == {}


def test_class_analysis_record_review_disposition_rejects_corrupt_existing_row(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    key = "cro_50" + ("1" * 62)
    point = {
        **_record("p0", "Boat"),
        "review_object_key": key,
    }
    result = {"summary": {"analysis_job_id": "ca_corrupt_overwrite"}}
    api._class_analysis_record_review_disposition_entry(
        result=result,
        point=point,
        disposition="skip",
        origin="desktop",
        client_action_id="review-corrupt-first",
    )
    shard_id, shard_lock = api._class_analysis_review_disposition_shard(key)
    ledger_root = api._class_analysis_review_disposition_root(
        create=False,
        strict=True,
    )
    assert ledger_root is not None
    shard_path = ledger_root / f"{shard_id}.json"
    with shard_lock:
        with api._class_analysis_review_disposition_file_lock(
            ledger_root,
            shard_id,
        ):
            shard_payload = api._class_analysis_load_review_disposition_shard(
                ledger_root,
                shard_id,
                strict=True,
            )
            # A row carrying a modern client receipt without its revision is
            # neither a valid legacy row nor safe to replay/overwrite.
            shard_payload["entries"][key].pop("entry_revision")
            api._class_analysis_write_audit_json(
                shard_path,
                ledger_root,
                shard_payload,
                detail="review_disposition_ledger_unavailable",
            )
    corrupt_bytes = shard_path.read_bytes()

    with pytest.raises(api.HTTPException) as exc_info:
        api._class_analysis_record_review_disposition_entry(
            result=result,
            point=point,
            disposition="confirm_current",
            origin="desktop",
            client_action_id="review-corrupt-overwrite",
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "review_disposition_ledger_invalid"
    assert shard_path.read_bytes() == corrupt_bytes


def test_class_analysis_bulk_review_history_delete_is_preconditioned_and_scoped(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    job_dir = class_root / "ca_bulk_history"
    job_dir.mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    points = {
        "p0": {
            **_record("p0", "Boat"),
            "review_object_key": "cro_00" + ("1" * 62),
        },
        "p1": {
            **_record("p1", "Bike"),
            "review_object_key": "cro_ff" + ("2" * 62),
        },
    }
    job = api.ClassAnalysisJob(
        job_id="ca_bulk_history",
        status="completed",
        summary={"source_key": "linked:dataset"},
    )
    api.CLASS_ANALYSIS_JOBS[job.job_id] = job
    monkeypatch.setattr(
        api,
        "_class_analysis_thumbnail_point_metadata",
        lambda _job, _job_dir, point_id: dict(points[point_id]),
    )
    sentinels = {
        class_root / "linked_labels.txt": b"0 0.5 0.5 0.2 0.2\n",
        class_root / "audit" / "qwen_review.json": b'{"decision":"keep"}',
        class_root / "audit" / "future_training.jsonl": b'{"action":"keep"}\n',
    }
    for path, content in sentinels.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    try:
        api._class_analysis_record_review_disposition_entry(
            result={"summary": {"analysis_job_id": job.job_id}},
            point=points["p0"],
            disposition="confirm_current",
            origin="desktop",
            client_action_id="history-source-p0",
        )
        api._class_analysis_record_review_disposition_entry(
            result={"summary": {"analysis_job_id": job.job_id}},
            point=points["p1"],
            disposition="skip",
            origin="desktop",
            client_action_id="history-source-p1",
        )
        stored = api._class_analysis_lookup_review_dispositions(
            [point["review_object_key"] for point in points.values()]
        )
        payload = {
            "schema": api.CLASS_ANALYSIS_REVIEW_HISTORY_DELETE_SCHEMA,
            "client_action_id": "history-delete-bulk-scoped",
            "entries": [
                {
                    "point_id": point_id,
                    "expected_disposition": stored[point["review_object_key"]][
                        "disposition"
                    ],
                    "expected_revision": stored[point["review_object_key"]][
                        "entry_revision"
                    ],
                }
                for point_id, point in points.items()
            ],
        }

        with pytest.raises(api.HTTPException) as invalid_job_exc_info:
            api.delete_class_analysis_review_history(
                "ca bulk history",
                payload,
            )
        assert invalid_job_exc_info.value.status_code == 400
        assert invalid_job_exc_info.value.detail == "job_id_invalid"
        assert set(
            api._class_analysis_lookup_review_dispositions(
                [point["review_object_key"] for point in points.values()]
            )
        ) == {point["review_object_key"] for point in points.values()}

        deleted = api.delete_class_analysis_review_history(job.job_id, payload)
        repeated = api.delete_class_analysis_review_history(job.job_id, payload)

        assert deleted == {
            "schema": api.CLASS_ANALYSIS_REVIEW_HISTORY_DELETE_SCHEMA,
            "client_action_id": "history-delete-bulk-scoped",
            "job_id": job.job_id,
            "requested_point_count": 2,
            "resolved_review_key_count": 2,
            "labels_changed": False,
            "annotations_changed": False,
            "training_actions_deleted": 0,
            "qwen_audits_deleted": 0,
            "status": "deleted",
            "deleted_count": 2,
            "absent_count": 0,
            "deleted_disposition_counts": {
                "confirm_current": 1,
                "skip": 1,
            },
        }
        assert repeated["status"] == "nothing_to_delete"
        assert repeated["deleted_count"] == 0
        assert repeated["absent_count"] == 2
        assert api._class_analysis_lookup_review_dispositions(
            [point["review_object_key"] for point in points.values()]
        ) == {}
        with pytest.raises(api.HTTPException) as replay_exc_info:
            api._class_analysis_record_review_disposition_entry(
                result={"summary": {"analysis_job_id": job.job_id}},
                point=points["p0"],
                disposition="confirm_current",
                origin="desktop",
                client_action_id="history-source-p0",
            )
        assert replay_exc_info.value.status_code == 409
        assert replay_exc_info.value.detail == {
            "code": "review_disposition_action_deleted",
            "review_object_key": points["p0"]["review_object_key"],
        }
        assert api._class_analysis_lookup_review_dispositions(
            [point["review_object_key"] for point in points.values()]
        ) == {}
        for path, content in sentinels.items():
            assert path.read_bytes() == content
    finally:
        api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)


def test_class_analysis_bulk_review_history_delete_stale_aborts_all_shards(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    (class_root / "ca_bulk_stale").mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    points = {
        "p0": {**_record("p0", "Boat"), "review_object_key": "cro_10" + ("1" * 62)},
        "p1": {**_record("p1", "Bike"), "review_object_key": "cro_e0" + ("2" * 62)},
    }
    job = api.ClassAnalysisJob(
        job_id="ca_bulk_stale",
        status="completed",
        summary={"source_key": "linked:dataset"},
    )
    api.CLASS_ANALYSIS_JOBS[job.job_id] = job
    monkeypatch.setattr(
        api,
        "_class_analysis_thumbnail_point_metadata",
        lambda _job, _job_dir, point_id: dict(points[point_id]),
    )
    try:
        for point in points.values():
            api._class_analysis_record_review_disposition_entry(
                result={"summary": {"analysis_job_id": job.job_id}},
                point=point,
                disposition="skip",
                origin="desktop",
            )
        keys = [point["review_object_key"] for point in points.values()]
        stored = api._class_analysis_lookup_review_dispositions(keys)
        with pytest.raises(api.HTTPException) as exc_info:
            api.delete_class_analysis_review_history(
                job.job_id,
                {
                    "schema": api.CLASS_ANALYSIS_REVIEW_HISTORY_DELETE_SCHEMA,
                    "client_action_id": "history-delete-stale-shards",
                    "entries": [
                        {
                            "point_id": "p0",
                            "expected_disposition": "skip",
                            "expected_revision": stored[keys[0]]["entry_revision"],
                        },
                        {
                            "point_id": "p1",
                            "expected_disposition": "confirm_current",
                            "expected_revision": stored[keys[1]]["entry_revision"],
                        },
                    ],
                },
            )
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail == {
            "code": "review_history_changed",
            "point_ids": ["p1"],
        }
        assert set(api._class_analysis_lookup_review_dispositions(keys)) == set(keys)
    finally:
        api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)


def test_class_analysis_bulk_review_history_delete_deduplicates_pair_directions(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    (class_root / "ca_bulk_pair").mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    current_key = "cro_01" + ("4" * 62)
    other_key = "cro_02" + ("5" * 62)
    pair_key = api._class_analysis_dual_bbox_pair_key_from_object_keys(
        current_key,
        other_key,
    )
    points = {
        "current": {
            **_record("current", "Bike"),
            "review_object_key": current_key,
            "dual_bbox_conflict": {"enabled": True},
        },
        "other": {
            **_record("other", "Person"),
            "review_object_key": other_key,
            "dual_bbox_conflict": {"enabled": True},
        },
    }
    job = api.ClassAnalysisJob(job_id="ca_bulk_pair", status="completed")
    api.CLASS_ANALYSIS_JOBS[job.job_id] = job
    monkeypatch.setattr(
        api,
        "_class_analysis_thumbnail_point_metadata",
        lambda _job, _job_dir, point_id: dict(points[point_id]),
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_point_is_dual_bbox_resolution_task",
        lambda point: str(point.get("point_id")) in points,
    )

    def validated_pair(_job, point, _payload):
        if point["point_id"] == "current":
            return {}, pair_key, current_key, other_key
        return {}, pair_key, other_key, current_key

    monkeypatch.setattr(
        api,
        "_class_analysis_validated_dual_bbox_conflict",
        validated_pair,
    )
    commit_attestation = {
        "schema": api.CLASS_ANALYSIS_DUAL_BBOX_ANNOTATION_COMMIT_SCHEMA,
        "committed": True,
        "operation_id": "dual_bbox:test-pair-history",
        "pair_review_key": pair_key,
        "action": "delete_current_box",
    }
    commit_attestation["attestation_sha256"] = (
        api._class_analysis_dual_bbox_commit_attestation_hash(
            commit_attestation
        )
    )
    try:
        api._class_analysis_record_review_disposition_entry(
            result={"summary": {"analysis_job_id": job.job_id}},
            point={
                **points["current"],
                "review_object_key": pair_key,
                "dual_bbox_pair_resolution": {
                    "schema": api.CLASS_ANALYSIS_DUAL_BBOX_PAIR_KEY_SCHEMA,
                    "action": "delete_current_box",
                    "pair_review_key": pair_key,
                    "current_review_object_key": current_key,
                    "other_review_object_key": other_key,
                    "annotation_commit_attestation": commit_attestation,
                },
            },
            disposition="delete_current_box",
            origin="desktop",
        )
        review_revision = api._class_analysis_lookup_review_dispositions([pair_key])[
            pair_key
        ]["entry_revision"]

        deleted = api.delete_class_analysis_review_history(
            job.job_id,
            {
                "schema": api.CLASS_ANALYSIS_REVIEW_HISTORY_DELETE_SCHEMA,
                "client_action_id": "history-delete-pair-directions",
                "entries": [
                    {
                        "point_id": "current",
                        "expected_disposition": "delete_current_box",
                        "expected_revision": review_revision,
                    },
                    {
                        "point_id": "other",
                        "expected_disposition": "delete_overlapping_box",
                        "expected_revision": review_revision,
                    },
                ],
            },
        )

        assert deleted["requested_point_count"] == 2
        assert deleted["resolved_review_key_count"] == 1
        assert deleted["deleted_count"] == 1
        assert deleted["deleted_disposition_counts"] == {
            "delete_current_box": 1
        }
        assert api._class_analysis_lookup_review_dispositions([pair_key]) == {}
    finally:
        api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)


def test_class_analysis_bulk_review_history_delete_rolls_back_exact_shards_on_write_failure(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    (class_root / "ca_bulk_rollback").mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    points = {
        "p0": {**_record("p0", "Boat"), "review_object_key": "cro_20" + ("1" * 62)},
        "p1": {**_record("p1", "Bike"), "review_object_key": "cro_d0" + ("2" * 62)},
    }
    job = api.ClassAnalysisJob(
        job_id="ca_bulk_rollback",
        status="completed",
        summary={"source_key": "linked:dataset"},
    )
    api.CLASS_ANALYSIS_JOBS[job.job_id] = job
    monkeypatch.setattr(
        api,
        "_class_analysis_thumbnail_point_metadata",
        lambda _job, _job_dir, point_id: dict(points[point_id]),
    )
    try:
        for point in points.values():
            api._class_analysis_record_review_disposition_entry(
                result={"summary": {"analysis_job_id": job.job_id}},
                point=point,
                disposition="skip",
                origin="desktop",
            )
        keys = [point["review_object_key"] for point in points.values()]
        stored = api._class_analysis_lookup_review_dispositions(keys)
        ledger_root = (
            class_root / "audit" / "human_review_dispositions"
        )
        original_bytes = {
            shard: (ledger_root / f"{shard}.json").read_bytes()
            for shard in ("20", "d0")
        }
        real_write = api._class_analysis_write_audit_json
        write_count = 0

        def fail_second_write(path, root, value, *, detail):
            nonlocal write_count
            write_count += 1
            if write_count == 2:
                raise api.HTTPException(
                    status_code=500,
                    detail="injected_second_shard_failure",
                )
            return real_write(path, root, value, detail=detail)

        monkeypatch.setattr(
            api,
            "_class_analysis_write_audit_json",
            fail_second_write,
        )
        with pytest.raises(api.HTTPException) as exc_info:
            api.delete_class_analysis_review_history(
                job.job_id,
                {
                    "schema": api.CLASS_ANALYSIS_REVIEW_HISTORY_DELETE_SCHEMA,
                    "client_action_id": "history-delete-rollback-shards",
                    "entries": [
                        {
                            "point_id": point_id,
                            "expected_disposition": "skip",
                            "expected_revision": stored[
                                point["review_object_key"]
                            ]["entry_revision"],
                        }
                        for point_id, point in points.items()
                    ],
                },
            )
        assert exc_info.value.detail == "injected_second_shard_failure"
        assert write_count == 2
        assert {
            shard: (ledger_root / f"{shard}.json").read_bytes()
            for shard in ("20", "d0")
        } == original_bytes
        assert set(api._class_analysis_lookup_review_dispositions(keys)) == set(keys)
    finally:
        api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)


def test_class_analysis_bulk_review_history_delete_requires_revision_or_exact_timestamp(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    (class_root / "ca_bulk_legacy_timestamp").mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    point = {
        **_record("p0", "Boat"),
        "review_object_key": "cro_30" + ("1" * 62),
    }
    job = api.ClassAnalysisJob(
        job_id="ca_bulk_legacy_timestamp",
        status="completed",
        summary={"source_key": "linked:dataset"},
    )
    api.CLASS_ANALYSIS_JOBS[job.job_id] = job
    monkeypatch.setattr(
        api,
        "_class_analysis_thumbnail_point_metadata",
        lambda _job, _job_dir, _point_id: dict(point),
    )
    try:
        api._class_analysis_record_review_disposition_entry(
            result={"summary": {"analysis_job_id": job.job_id}},
            point=point,
            disposition="skip",
            origin="desktop",
        )
        stored = api._class_analysis_lookup_review_dispositions(
            [point["review_object_key"]]
        )[point["review_object_key"]]
        assert api.CLASS_ANALYSIS_REVIEW_ENTRY_REVISION_PATTERN.fullmatch(
            stored["entry_revision"]
        )
        deleted = api.delete_class_analysis_review_history(
            job.job_id,
            {
                "schema": api.CLASS_ANALYSIS_REVIEW_HISTORY_DELETE_SCHEMA,
                "client_action_id": "history-delete-exact-revision",
                "entries": [
                    {
                        "point_id": "p0",
                        "expected_disposition": "skip",
                        "expected_revision": stored["entry_revision"],
                    }
                ],
            },
        )
        assert deleted["deleted_count"] == 1
        assert deleted["client_action_id"] == "history-delete-exact-revision"

        api._class_analysis_record_review_disposition_entry(
            result={"summary": {"analysis_job_id": job.job_id}},
            point=point,
            disposition="skip",
            origin="desktop",
        )
        stored = api._class_analysis_lookup_review_dispositions(
            [point["review_object_key"]]
        )[point["review_object_key"]]
        with pytest.raises(api.HTTPException) as missing_exc_info:
            api.delete_class_analysis_review_history(
                job.job_id,
                {
                    "schema": api.CLASS_ANALYSIS_REVIEW_HISTORY_DELETE_SCHEMA,
                    "client_action_id": "history-delete-missing-cas",
                    "entries": [
                        {
                            "point_id": "p0",
                            "expected_disposition": "skip",
                        }
                    ],
                },
            )
        assert missing_exc_info.value.status_code == 400
        assert api._class_analysis_lookup_review_dispositions(
            [point["review_object_key"]]
        )

        # Timestamp CAS remains a compatibility path only for rows written
        # before revision tokens were introduced.  A current row must never
        # be deletable through the weaker timestamp contract.
        with pytest.raises(api.HTTPException) as current_timestamp_exc_info:
            api.delete_class_analysis_review_history(
                job.job_id,
                {
                    "schema": api.CLASS_ANALYSIS_REVIEW_HISTORY_DELETE_SCHEMA,
                    "client_action_id": "history-delete-current-timestamp-only",
                    "entries": [
                        {
                            "point_id": "p0",
                            "expected_disposition": "skip",
                            "expected_reviewed_at": stored["updated_at"],
                        }
                    ],
                },
            )
        assert current_timestamp_exc_info.value.status_code == 409
        shard_id, shard_lock = api._class_analysis_review_disposition_shard(
            point["review_object_key"]
        )
        ledger_root = api._class_analysis_review_disposition_root(
            create=False,
            strict=True,
        )
        assert ledger_root is not None
        with shard_lock:
            with api._class_analysis_review_disposition_file_lock(
                ledger_root,
                shard_id,
            ):
                shard_payload = (
                    api._class_analysis_load_review_disposition_shard(
                        ledger_root,
                        shard_id,
                        strict=True,
                    )
                )
                shard_payload["entries"][point["review_object_key"]].pop(
                    "entry_revision"
                )
                api._class_analysis_write_audit_json(
                    ledger_root / f"{shard_id}.json",
                    ledger_root,
                    shard_payload,
                    detail="review_disposition_ledger_unavailable",
                )
        stored = api._class_analysis_lookup_review_dispositions(
            [point["review_object_key"]]
        )[point["review_object_key"]]
        assert "entry_revision" not in stored

        with pytest.raises(api.HTTPException) as exc_info:
            api.delete_class_analysis_review_history(
                job.job_id,
                {
                    "schema": api.CLASS_ANALYSIS_REVIEW_HISTORY_DELETE_SCHEMA,
                    "client_action_id": "history-delete-stale-timestamp",
                    "entries": [
                        {
                            "point_id": "p0",
                            "expected_disposition": "skip",
                            "expected_reviewed_at": math.nextafter(
                                stored["updated_at"],
                                math.inf,
                            ),
                        }
                    ],
                },
            )
        assert exc_info.value.status_code == 409
        assert api._class_analysis_lookup_review_dispositions(
            [point["review_object_key"]]
        )

        deleted_legacy = api.delete_class_analysis_review_history(
            job.job_id,
            {
                "schema": api.CLASS_ANALYSIS_REVIEW_HISTORY_DELETE_SCHEMA,
                "client_action_id": "history-delete-exact-legacy-timestamp",
                "entries": [
                    {
                        "point_id": "p0",
                        "expected_disposition": "skip",
                        "expected_reviewed_at": stored["updated_at"],
                    }
                ],
            },
        )
        assert deleted_legacy["deleted_count"] == 1
    finally:
        api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)


def test_class_analysis_bulk_review_history_delete_rejects_corrupt_requested_entry(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    (class_root / "ca_bulk_corrupt_entry").mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    point = {
        **_record("p0", "Boat"),
        "review_object_key": "cro_40" + ("1" * 62),
    }
    job = api.ClassAnalysisJob(
        job_id="ca_bulk_corrupt_entry",
        status="completed",
        summary={"source_key": "linked:dataset"},
    )
    api.CLASS_ANALYSIS_JOBS[job.job_id] = job
    monkeypatch.setattr(
        api,
        "_class_analysis_thumbnail_point_metadata",
        lambda _job, _job_dir, _point_id: dict(point),
    )
    try:
        api._class_analysis_record_review_disposition_entry(
            result={"summary": {"analysis_job_id": job.job_id}},
            point=point,
            disposition="skip",
            origin="desktop",
        )
        shard_path = (
            class_root
            / "audit"
            / "human_review_dispositions"
            / "40.json"
        )
        payload = json.loads(shard_path.read_text(encoding="utf-8"))
        payload["entries"][point["review_object_key"]][
            "review_object_key"
        ] = "cro_41" + ("2" * 62)
        shard_path.write_text(json.dumps(payload), encoding="utf-8")
        corrupt_bytes = shard_path.read_bytes()

        with pytest.raises(api.HTTPException) as exc_info:
            api.delete_class_analysis_review_history(
                job.job_id,
                {
                    "schema": api.CLASS_ANALYSIS_REVIEW_HISTORY_DELETE_SCHEMA,
                    "client_action_id": "history-delete-corrupt-entry",
                    "entries": [
                        {
                            "point_id": "p0",
                            "expected_disposition": "skip",
                            "expected_reviewed_at": 1.0,
                        }
                    ],
                },
            )
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "review_disposition_ledger_invalid"
        assert shard_path.read_bytes() == corrupt_bytes
    finally:
        api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)


def test_class_analysis_review_hydration_rejects_only_requested_corrupt_rows(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    bad_key = "cro_70" + ("1" * 62)
    same_shard_good_key = "cro_70" + ("2" * 62)
    other_shard_good_key = "cro_80" + ("3" * 62)
    for index, key in enumerate(
        (bad_key, same_shard_good_key, other_shard_good_key)
    ):
        api._class_analysis_record_review_disposition_entry(
            result={"summary": {"analysis_job_id": "ca_hydration"}},
            point={
                **_record(f"p{index}", "Boat"),
                "review_object_key": key,
            },
            disposition="skip",
            origin="desktop",
            client_action_id=f"hydrate-review-{index}",
        )
    bad_shard_path = (
        class_root / "audit" / "human_review_dispositions" / "70.json"
    )
    bad_shard = json.loads(bad_shard_path.read_text(encoding="utf-8"))
    bad_shard["entries"][bad_key]["updated_at"] = True
    bad_shard_path.write_text(json.dumps(bad_shard), encoding="utf-8")

    with pytest.raises(api.HTTPException) as exc_info:
        api._class_analysis_lookup_review_dispositions([bad_key])
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "review_disposition_ledger_invalid"
    assert set(
        api._class_analysis_lookup_review_dispositions(
            [same_shard_good_key, other_shard_good_key]
        )
    ) == {same_shard_good_key, other_shard_good_key}


def test_class_analysis_review_hydration_rejects_requested_symlink_shard(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    key = "cro_90" + ("4" * 62)
    api._class_analysis_record_review_disposition_entry(
        result={"summary": {"analysis_job_id": "ca_hydration_link"}},
        point={**_record("p0", "Boat"), "review_object_key": key},
        disposition="skip",
        origin="desktop",
        client_action_id="hydrate-link-review",
    )
    shard_path = (
        class_root / "audit" / "human_review_dispositions" / "90.json"
    )
    outside = tmp_path / "outside-shard.json"
    outside.write_text(shard_path.read_text(encoding="utf-8"), encoding="utf-8")
    shard_path.unlink()
    shard_path.symlink_to(outside)

    with pytest.raises(api.HTTPException) as exc_info:
        api._class_analysis_lookup_review_dispositions([key])
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "review_disposition_ledger_invalid"


def test_class_analysis_review_hydration_rejects_malformed_pair_resolution(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    current_key = "cro_a0" + ("5" * 62)
    other_key = "cro_a1" + ("6" * 62)
    pair_key = api._class_analysis_dual_bbox_pair_key_from_object_keys(
        current_key,
        other_key,
    )
    api._class_analysis_record_review_disposition_entry(
        result={"summary": {"analysis_job_id": "ca_pair_hydration"}},
        point={
            **_record("p0", "Bike"),
            "review_object_key": pair_key,
            "dual_bbox_pair_resolution": {
                "schema": api.CLASS_ANALYSIS_DUAL_BBOX_PAIR_KEY_SCHEMA,
                "action": "keep_both_boxes",
                "pair_review_key": pair_key,
                "current_review_object_key": current_key,
                "other_review_object_key": other_key,
            },
        },
        disposition="keep_both_boxes",
        origin="desktop",
        client_action_id="hydrate-pair-review",
    )
    shard_id, _lock = api._class_analysis_review_disposition_shard(pair_key)
    shard_path = (
        class_root
        / "audit"
        / "human_review_dispositions"
        / f"{shard_id}.json"
    )
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    shard["entries"][pair_key]["dual_bbox_pair_resolution"][
        "action"
    ] = "unresolved"
    shard_path.write_text(json.dumps(shard), encoding="utf-8")

    with pytest.raises(api.HTTPException) as exc_info:
        api._class_analysis_lookup_review_dispositions([pair_key])
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "review_disposition_ledger_invalid"

    shard["entries"][pair_key]["disposition"] = "skip"
    shard["entries"][pair_key]["dual_bbox_pair_resolution"]["action"] = "skip"
    shard_path.write_text(json.dumps(shard), encoding="utf-8")
    with pytest.raises(api.HTTPException) as pair_action_exc_info:
        api._class_analysis_lookup_review_dispositions([pair_key])
    assert pair_action_exc_info.value.status_code == 500
    assert pair_action_exc_info.value.detail == (
        "review_disposition_ledger_invalid"
    )


def test_class_analysis_bulk_review_history_delete_does_not_hide_ledger_root_corruption(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    (class_root / "ca_bulk_corrupt_root").mkdir(parents=True)
    audit_root = class_root / "audit"
    audit_root.mkdir()
    outside = tmp_path / "outside_ledger"
    outside.mkdir()
    (audit_root / "human_review_dispositions").symlink_to(
        outside,
        target_is_directory=True,
    )
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    point = {
        **_record("p0", "Boat"),
        "review_object_key": "cro_50" + ("1" * 62),
    }
    job = api.ClassAnalysisJob(
        job_id="ca_bulk_corrupt_root",
        status="completed",
        summary={"source_key": "linked:dataset"},
    )
    api.CLASS_ANALYSIS_JOBS[job.job_id] = job
    monkeypatch.setattr(
        api,
        "_class_analysis_thumbnail_point_metadata",
        lambda _job, _job_dir, _point_id: dict(point),
    )
    try:
        with pytest.raises(api.HTTPException) as hydration_exc_info:
            api._class_analysis_lookup_review_dispositions(
                [point["review_object_key"]]
            )
        assert hydration_exc_info.value.status_code == 500
        assert hydration_exc_info.value.detail == (
            "review_disposition_ledger_unavailable"
        )
        with pytest.raises(api.HTTPException) as exc_info:
            api.delete_class_analysis_review_history(
                job.job_id,
                {
                    "schema": api.CLASS_ANALYSIS_REVIEW_HISTORY_DELETE_SCHEMA,
                    "client_action_id": "history-delete-corrupt-root",
                    "entries": [
                        {
                            "point_id": "p0",
                            "expected_disposition": "skip",
                            "expected_reviewed_at": 1.0,
                        }
                    ],
                },
            )
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "review_disposition_ledger_unavailable"
    finally:
        api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)


def test_class_analysis_bulk_review_history_delete_uses_persisted_pair_identity_without_live_source(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    (class_root / "ca_bulk_frozen_pair").mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    current_key = "cro_60" + ("1" * 62)
    other_key = "cro_61" + ("2" * 62)
    pair_key = api._class_analysis_dual_bbox_pair_key_from_object_keys(
        current_key,
        other_key,
    )
    conflict = {
        "enabled": True,
        "review_mode": "dual_bbox_annotation_resolution",
        "point_id": "p0",
        "other_point_id": "p1",
        "current_class": "Bike",
        "other_class_name": "Person",
        "current_geometry_kind": "bbox",
        "other_geometry_kind": "bbox",
        "target_bbox_xyxy": [0, 0, 10, 10],
        "other_bbox_xyxy": [0, 0, 10, 10],
        "split": "train",
        "image_relpath": "p0.jpg",
        "pair_review_key": pair_key,
        "current_review_object_key": current_key,
        "other_review_object_key": other_key,
    }
    point = {
        **_record("p0", "Bike"),
        "kind": "bbox",
        "review_object_key": current_key,
        "dual_bbox_conflict": conflict,
    }
    job = api.ClassAnalysisJob(
        job_id="ca_bulk_frozen_pair",
        status="completed",
        summary={"source_key": "linked:missing-dataset"},
    )
    api.CLASS_ANALYSIS_JOBS[job.job_id] = job
    monkeypatch.setattr(
        api,
        "_class_analysis_thumbnail_point_metadata",
        lambda _job, _job_dir, _point_id: copy.deepcopy(point),
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_backfill_review_object_key",
        lambda *_args, **_kwargs: pytest.fail(
            "modern frozen pair deletion must not backfill from live source"
        ),
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_validated_dual_bbox_conflict",
        lambda *_args, **_kwargs: pytest.fail(
            "modern frozen pair deletion must not invoke legacy validation"
        ),
    )
    pair_resolution = api._class_analysis_dual_bbox_pair_resolution_record(
        action="keep_both_boxes",
        point=point,
        conflict=conflict,
        pair_review_key=pair_key,
        current_review_object_key=current_key,
        other_review_object_key=other_key,
        annotation_commit_attestation=None,
    )
    try:
        api._class_analysis_record_review_disposition_entry(
            result={"summary": {"analysis_job_id": job.job_id}},
            point={
                **point,
                "review_object_key": pair_key,
                "dual_bbox_pair_resolution": pair_resolution,
            },
            disposition="keep_both_boxes",
            origin="desktop",
        )
        stored = api._class_analysis_lookup_review_dispositions([pair_key])[pair_key]
        deleted = api.delete_class_analysis_review_history(
            job.job_id,
            {
                "schema": api.CLASS_ANALYSIS_REVIEW_HISTORY_DELETE_SCHEMA,
                "client_action_id": "history-delete-persisted-pair",
                "entries": [
                    {
                        "point_id": "p0",
                        "expected_disposition": "keep_both_boxes",
                        "expected_revision": stored["entry_revision"],
                    }
                ],
            },
        )
        assert deleted["deleted_count"] == 1
        assert api._class_analysis_lookup_review_dispositions([pair_key]) == {}
    finally:
        api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)


def test_class_analysis_review_disposition_backfills_legacy_key_from_frozen_metadata(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    job_dir = class_root / "ca_legacy_review"
    workspace = class_root / "ca_workspace_owner" / "active_workspace"
    images_dir = workspace / "images"
    labels_dir = workspace / "labels"
    job_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)

    image_path = images_dir / "frame.jpg"
    Image.new("RGB", (100, 100), (30, 60, 90)).save(image_path)
    frozen_label_line = "5 0.5 0.5 0.2 0.4"
    # Simulate the user moving this bbox after the analysis snapshot completed.
    # The review action must use the frozen indexed geometry, not this live row.
    (labels_dir / "frame.txt").write_text(
        "5 0.7 0.6 0.2 0.4\n",
        encoding="utf-8",
    )
    api.CLASS_ANALYSIS_JOBS["ca_legacy_review"] = api.ClassAnalysisJob(
        job_id="ca_legacy_review",
        status="completed",
        request={
            "source_mode": "active_workspace",
            "workspace_id": "ca_workspace_owner",
            "snapshot_id": "cas_legacy",
            "workspace_dir": str(workspace),
            "yolo_layout": "flat",
        },
        summary={
            "analysis_job_id": "ca_legacy_review",
            "source_mode": "active_workspace",
            "source_id": "cas_legacy",
            "source_key": "active:cas_legacy",
        },
    )
    legacy_point = {
        "point_id": "p0",
        "source_mode": "active_workspace",
        "source_id": "cas_legacy",
        "source_key": "active:cas_legacy",
        "split": "train",
        "image_relpath": "frame.jpg",
        "class_name": "Bike",
        "kind": "bbox",
        "label_line": frozen_label_line,
        "bbox_xyxy": [40, 30, 60, 70],
        "wrong_class_suspicion": 0.91,
        "wrong_class_review_reason": "dual_bbox_conflict",
    }
    monkeypatch.setattr(
        api,
        "_class_analysis_thumbnail_point_metadata",
        lambda *_args, **_kwargs: dict(legacy_point),
    )
    monkeypatch.setattr(
        api,
        "_safe_job_result_json_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy indexed review action must not load result.json")
        ),
    )
    frozen_geometry = api._class_analysis_parse_yolo_geometry(
        frozen_label_line,
        image_width=100,
        image_height=100,
    )
    expected_key = api._class_analysis_review_object_key(
        source_key="active:cas_legacy",
        image_sha256=api._class_analysis_file_sha256(image_path),
        split="train",
        image_relpath="frame.jpg",
        class_name="Bike",
        geometry=frozen_geometry,
        image_width=100,
        image_height=100,
    )

    try:
        response = api.record_class_analysis_review_disposition(
            "ca_legacy_review",
            "p0",
            {
                "disposition": "skip",
                "origin": "desktop",
                "client_action_id": "legacy-review-skip",
            },
        )
        stored = api._class_analysis_lookup_review_dispositions([expected_key])
    finally:
        api.CLASS_ANALYSIS_JOBS.pop("ca_legacy_review", None)

    assert response["review_object_key"] == expected_key
    assert response["disposition"] == "skip"
    assert stored[expected_key]["bbox_xyxy"] == [40.0, 30.0, 60.0, 70.0]
    assert (labels_dir / "frame.txt").read_text(encoding="utf-8") == (
        "5 0.7 0.6 0.2 0.4\n"
    )


def test_dual_bbox_review_accepts_stage2_duplicate_with_subpixel_drift(monkeypatch):
    current_key = "cro_" + ("1" * 64)
    other_key = "cro_" + ("2" * 64)
    pair_key = api._class_analysis_dual_bbox_pair_key_from_object_keys(
        current_key,
        other_key,
    )
    conflict = {
        "enabled": True,
        "point_id": "truck",
        "current_class": "Truck",
        "current_geometry_kind": "bbox",
        "other_point_id": "container",
        "other_class_name": "Container",
        "other_geometry_kind": "bbox",
        "target_bbox_xyxy": [275.99952, 893.00016, 303.99984, 926.00016],
        "other_bbox_xyxy": [276.00048, 890.99952, 303.00048, 924.99984],
        "split": "train",
        "image_relpath": "frame.png",
        "pair_review_key": pair_key,
        "current_review_object_key": current_key,
        "other_review_object_key": other_key,
        "relation": "duplicate_like",
    }
    point = {
        "point_id": "truck",
        "class_name": "Truck",
        "kind": "bbox",
        "bbox_xyxy": [276.0, 893.0, 304.0, 926.0],
        "split": "train",
        "image_relpath": "frame.png",
        "review_object_key": current_key,
        "dual_bbox_conflict": conflict,
    }
    monkeypatch.setattr(
        api,
        "_class_analysis_backfill_review_object_key",
        lambda _job, value: dict(value),
    )
    job = api.ClassAnalysisJob(job_id="ca_subpixel_pair", status="completed")

    trusted, trusted_pair_key, _, _ = (
        api._class_analysis_validated_dual_bbox_conflict(
            job,
            point,
            {"dual_bbox_conflict": {"other_point_id": "container"}},
        )
    )
    assert trusted_pair_key == pair_key
    assert trusted["target_bbox_xyxy"] == conflict["target_bbox_xyxy"]
    assert api._class_analysis_bbox_overlap_geometry(
        conflict["target_bbox_xyxy"],
        conflict["other_bbox_xyxy"],
    )["iou"] < 0.90

    stale_point = copy.deepcopy(point)
    stale_point["dual_bbox_conflict"]["target_bbox_xyxy"][1] = 892.98
    with pytest.raises(api.HTTPException) as exc_info:
        api._class_analysis_validated_dual_bbox_conflict(
            job,
            stale_point,
            {"dual_bbox_conflict": {"other_point_id": "container"}},
        )
    assert exc_info.value.detail == "dual_bbox_conflict_stale"


def test_dual_bbox_resolution_uses_canonical_pair_ledger_and_requires_committed_delete(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    workspace = class_root / "ca_pair_owner" / "active_workspace"
    images_dir = workspace / "images"
    labels_dir = workspace / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    image_path = images_dir / "frame.jpg"
    Image.new("RGB", (100, 100), (40, 60, 80)).save(image_path)
    (labels_dir / "frame.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n1 0.5 0.5 0.4 0.4\n",
        encoding="utf-8",
    )
    image_sha256 = api._class_analysis_file_sha256(image_path)
    current_key = api._class_analysis_review_object_key(
        source_key="active:cas_pair",
        image_sha256=image_sha256,
        split="train",
        image_relpath="frame.jpg",
        class_name="Bike",
        geometry={"kind": "bbox", "bbox_xyxy": [30, 30, 70, 70]},
        image_width=100,
        image_height=100,
    )
    conflict = {
        "enabled": True,
        "kind": "near_identical_cross_class_bbox",
        "review_mode": "dual_bbox_annotation_resolution_v1",
        "point_id": "bike",
        "current_class": "Bike",
        "other_point_id": "person",
        "target_bbox_xyxy": [30, 30, 70, 70],
        "other_bbox_xyxy": [30, 30, 70, 70],
        "other_class_name": "Person",
        "class_name": "Person",
        "split": "train",
        "image_relpath": "frame.jpg",
        "iou": 1.0,
        "corner_similarity": 1.0,
    }
    point = {
        "point_id": "bike",
        "review_object_key": current_key,
        "source_mode": "active_workspace",
        "source_id": "cas_pair",
        "source_key": "active:cas_pair",
        "split": "train",
        "image_relpath": "frame.jpg",
        "class_name": "Bike",
        "bbox_xyxy": [30, 30, 70, 70],
        "is_dual_bbox_conflict": True,
        "dual_bbox_conflict": conflict,
        "is_wrong_class_candidate": True,
    }
    api.CLASS_ANALYSIS_JOBS["ca_pair"] = api.ClassAnalysisJob(
        job_id="ca_pair",
        status="completed",
        request={
            "source_mode": "active_workspace",
            "workspace_id": "ca_pair_owner",
            "snapshot_id": "cas_pair",
            "workspace_dir": str(workspace),
            "yolo_layout": "flat",
        },
        summary={
            "analysis_job_id": "ca_pair",
            "source_mode": "active_workspace",
            "source_id": "cas_pair",
            "source_key": "active:cas_pair",
        },
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_thumbnail_point_metadata",
        lambda *_args, **_kwargs: dict(point),
    )
    (
        canonical_conflict,
        expected_pair_key,
        expected_current_key,
        expected_other_key,
    ) = api._class_analysis_validated_dual_bbox_conflict(
        api.CLASS_ANALYSIS_JOBS["ca_pair"],
        point,
        {"dual_bbox_conflict": conflict},
    )
    fake_attestation = {
        "schema": api.CLASS_ANALYSIS_DUAL_BBOX_ANNOTATION_COMMIT_SCHEMA,
        "committed": True,
        "analysis_job_id": "ca_pair",
        "point_id": "bike",
        "pair_review_key": expected_pair_key,
        "action": "delete_current_box",
        "deleted_point_id": "bike",
        "surviving_point_id": "person",
    }
    fake_attestation["attestation_sha256"] = (
        api._class_analysis_dual_bbox_commit_attestation_hash(
            fake_attestation
        )
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_validate_dual_bbox_annotation_commit_attestation",
        lambda **kwargs: (
            dict(kwargs["value"])
            if isinstance(kwargs.get("value"), dict)
            else (_ for _ in ()).throw(
                api.HTTPException(
                    status_code=409,
                    detail="dual_bbox_deletion_attestation_required",
                )
            )
        ),
    )
    review_id = "cqr_pair_capture"
    review = api.ClassAnalysisQwenReviewJob(
        review_id=review_id,
        parent_job_id="ca_pair",
        point_id="bike",
        status="completed",
        request={"model_id": "test-pair-vlm"},
        result={
            "review_mode": "dual_bbox_annotation_resolution",
            "dual_bbox_action": "delete_current_box",
            "dual_bbox_conflict": canonical_conflict,
            "applied": False,
        },
    )
    review_dir = (
        class_root / "ca_pair" / "qwen_reviews" / review_id
    )
    evidence_dir = review_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), (20, 40, 60)).save(
        evidence_dir / "dual_bbox_pair_1.jpg"
    )
    (review_dir / "result.json").write_text(
        json.dumps(review.result),
        encoding="utf-8",
    )
    with api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS_LOCK:
        api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS[review_id] = review
    try:
        for generic_disposition in ("confirm_current", "skip"):
            with pytest.raises(api.HTTPException) as guard_exc:
                api.record_class_analysis_review_disposition(
                    "ca_pair",
                    "bike",
                    {
                        "disposition": generic_disposition,
                        "client_action_id": (
                            f"pair-guard-{generic_disposition}"
                        ),
                    },
                )
            assert guard_exc.value.status_code == 409
            assert (
                guard_exc.value.detail
                == "dual_bbox_pair_action_required"
            )
        for generic_action in (
            "confirm_current",
            "skip",
            "change_class",
        ):
            with pytest.raises(api.HTTPException) as guard_exc:
                api.record_class_analysis_vignette_training_action(
                    "ca_pair",
                    {
                        "capture_training_data": True,
                        "action_type": generic_action,
                        "point_id": "bike",
                        "after_class": "Person",
                    },
                )
            assert guard_exc.value.status_code == 409
            assert (
                guard_exc.value.detail
                == "dual_bbox_pair_action_required"
            )
        assert api._class_analysis_lookup_review_dispositions(
            [current_key, expected_pair_key]
        ) == {}
        assert list(
            api._class_analysis_iter_vignette_training_actions()
        ) == []

        with pytest.raises(api.HTTPException) as exc_info:
            api.record_class_analysis_review_disposition(
                "ca_pair",
                "bike",
                {
                    "disposition": "delete_current_box",
                    "dual_bbox_conflict": conflict,
                    "client_action_id": "pair-delete-no-attestation",
                },
            )
        assert exc_info.value.detail == "dual_bbox_deletion_attestation_required"

        with pytest.raises(api.HTTPException) as exc_info:
            api.record_class_analysis_review_disposition(
                "ca_pair",
                "bike",
                {
                    "disposition": "delete_current_box",
                    "dual_bbox_conflict": conflict,
                    "annotation_mutation_committed": True,
                    "origin": "desktop",
                    "client_action_id": "pair-delete-fake-commit",
                },
            )
        assert exc_info.value.detail == "dual_bbox_deletion_attestation_required"

        recorded = api.record_class_analysis_review_disposition(
            "ca_pair",
            "bike",
            {
                "disposition": "delete_current_box",
                "dual_bbox_conflict": {
                    **conflict,
                    # Caller-owned durable identity must never enter either
                    # the disposition ledger or the training corpus.
                    "pair_review_key": "crp_" + ("f" * 64),
                    "current_review_object_key": "cro_" + ("e" * 64),
                    "other_review_object_key": "cro_" + ("d" * 64),
                },
                "dual_bbox_pair_resolution": {
                    "pair_review_key": "crp_" + ("c" * 64),
                },
                "annotation_commit_attestation": fake_attestation,
                "origin": "desktop",
                "capture_training_data": True,
                "client_action_id": "canonical-pair-capture",
                "review_id": review_id,
            },
        )
        pair_key = recorded["pair_review_key"]
        assert pair_key == expected_pair_key
        assert re.fullmatch(r"crp_[0-9a-f]{64}", pair_key)
        stored = api._class_analysis_lookup_review_dispositions([pair_key])
        assert stored[pair_key]["disposition"] == "delete_current_box"
        assert stored[pair_key]["dual_bbox_pair_resolution"]["other_point_id"] == "person"
        assert stored[pair_key]["dual_bbox_pair_resolution"]["deleted_point_id"] == "bike"
        assert recorded["training_capture"]["status"] == "recorded"
        changed_attestation = {
            **fake_attestation,
            "committed_at": 12345.0,
        }
        changed_attestation["attestation_sha256"] = (
            api._class_analysis_dual_bbox_commit_attestation_hash(
                changed_attestation
            )
        )
        with pytest.raises(api.HTTPException) as replay_conflict:
            api.record_class_analysis_review_disposition(
                "ca_pair",
                "bike",
                {
                    "disposition": "delete_current_box",
                    "dual_bbox_conflict": conflict,
                    "annotation_commit_attestation": changed_attestation,
                    "origin": "desktop",
                    "capture_training_data": True,
                    "client_action_id": "canonical-pair-capture",
                    "review_id": review_id,
                },
            )
        assert (
            replay_conflict.value.detail
            == "review_disposition_client_action_conflict"
        )
        captured = list(
            api._class_analysis_iter_vignette_training_actions()
        )
        assert len(captured) == 1
        captured_pair = captured[0]["dual_bbox_resolution"]
        assert captured[0]["pair_review_key"] == expected_pair_key
        assert captured_pair["pair_review_key"] == expected_pair_key
        assert captured_pair["current_review_object_key"] == expected_current_key
        assert captured_pair["other_review_object_key"] == expected_other_key
        assert captured_pair["point_id"] == "bike"
        assert captured_pair["other_point_id"] == "person"
        assert captured_pair["current_class"] == "Bike"
        assert captured_pair["other_class"] == "Person"
        assert captured_pair["deleted_point_id"] == "bike"
        assert captured_pair["annotation_mutation_committed"] is True
        assert (
            captured_pair["conflict"]["pair_review_key"]
            == expected_pair_key
        )
        pair_artifacts = [
            artifact
            for artifact in captured[0]["artifacts"]
            if artifact.get("role") == "dual_bbox_pair_evidence"
        ]
        assert len(pair_artifacts) == 1
        assert pair_artifacts[0]["linked_review_id"] == review_id
        assert pair_artifacts[0]["pair_review_key"] == expected_pair_key
        pair_blob = class_root / pair_artifacts[0]["blob_relpath"]
        assert pair_blob.is_file()
        assert (
            api._class_analysis_file_sha256(pair_blob)
            == pair_artifacts[0]["sha256"]
        )
        export_rows = api._class_analysis_vignette_training_export_rows(
            captured
        )
        assert [
            row["action_id"]
            for row in export_rows["geometry_decisions"]
        ] == [captured[0]["action_id"]]
        exported_roles = {
            artifact["role"] for artifact in export_rows["artifacts"]
        }
        assert "object_crop" in exported_roles
        assert "dual_bbox_pair_evidence" in exported_roles
        exported = api.export_class_analysis_vignette_training_actions()
        exported_path = Path(exported.path)
        try:
            with zipfile.ZipFile(exported_path) as archive:
                manifest = json.loads(
                    archive.read("manifest.json").decode("utf-8")
                )
                geometry_rows = [
                    json.loads(line)
                    for line in archive.read(
                        "geometry_decisions.jsonl"
                    ).decode("utf-8").splitlines()
                    if line.strip()
                ]
                data_card = archive.read("DATA_CARD.md").decode("utf-8")
                archive_names = set(archive.namelist())
            assert manifest["geometry_decision_count"] == 1
            assert manifest["rules"][
                "dual_bbox_geometry_decisions_exported_with_linked_evidence"
            ] is True
            assert geometry_rows[0]["action_id"] == captured[0]["action_id"]
            assert geometry_rows[0]["dual_bbox_resolution"][
                "annotation_commit_attestation"
            ]["attestation_sha256"] == fake_attestation[
                "attestation_sha256"
            ]
            assert "geometry_decisions.jsonl" in data_card
            for artifact in export_rows["artifacts"]:
                suffix = {
                    "image/jpeg": ".jpg",
                    "image/png": ".png",
                    "image/webp": ".webp",
                }.get(artifact["media_type"], ".bin")
                assert (
                    f"media/{artifact['sha256']}{suffix}"
                    in archive_names
                )
        finally:
            exported_path.unlink(missing_ok=True)

        conflict_with_key = {**conflict, "pair_review_key": pair_key}
        overlaid = api._class_analysis_apply_review_dispositions(
            {
                "summary": {"analysis_scope": "all_classes"},
                "points": [dict(point)],
                "wrong_class_candidates": [
                    {
                        "point_id": "bike",
                        "review_object_key": current_key,
                        "dual_bbox_conflict": conflict_with_key,
                    }
                ],
            }
        )
        assert overlaid["wrong_class_candidates"] == []
        assert overlaid["points"][0]["human_review_disposition"] == "delete_current_box"
        assert overlaid["points"][0]["human_review_pair_key"] == pair_key

        other_key = recorded["other_review_object_key"]
        reverse_conflict = {
            **conflict,
            "point_id": "person",
            "current_class": "Person",
            "other_point_id": "bike",
            "other_class_name": "Bike",
            "class_name": "Bike",
            "pair_review_key": pair_key,
            "current_review_object_key": other_key,
            "other_review_object_key": current_key,
        }
        other_point = {
            **point,
            "point_id": "person",
            "review_object_key": other_key,
            "class_name": "Person",
            "dual_bbox_conflict": reverse_conflict,
        }
        both_directions = api._class_analysis_apply_review_dispositions(
            {
                "summary": {"analysis_scope": "all_classes"},
                "points": [dict(point), dict(other_point)],
                "wrong_class_candidates": [
                    {
                        "point_id": "person",
                        "review_object_key": other_key,
                        "dual_bbox_conflict": reverse_conflict,
                    },
                    {
                        "point_id": "bike",
                        "review_object_key": current_key,
                        "dual_bbox_conflict": conflict_with_key,
                    },
                ],
            }
        )
        assert both_directions["wrong_class_candidates"] == []
        reviewed_directions = [
            item
            for item in both_directions["points"]
            if item.get("human_review_disposition")
        ]
        assert [item["point_id"] for item in reviewed_directions] == ["bike"]
        assert reviewed_directions[0]["human_review_disposition"] == "delete_current_box"

        reverse_only = api._class_analysis_apply_review_dispositions(
            {
                "summary": {"analysis_scope": "selected_class"},
                "points": [dict(other_point)],
                "within_class_outlier_candidates": [
                    {
                        "point_id": "person",
                        "review_object_key": other_key,
                        "dual_bbox_conflict": reverse_conflict,
                    }
                ],
            }
        )
        assert reverse_only["within_class_outlier_candidates"] == []
        assert (
            reverse_only["points"][0]["human_review_disposition"]
            == "delete_overlapping_box"
        )

        cleared = api.record_class_analysis_review_disposition(
            "ca_pair",
            "bike",
            {
                "disposition": "clear",
                "dual_bbox_conflict": conflict,
                "origin": "desktop",
                "client_action_id": "canonical-pair-clear",
                "expected_revision": recorded["human_review_revision"],
            },
        )
        assert cleared["pair_review_key"] == pair_key
        assert cleared["previous_disposition"] == "delete_current_box"
        assert api._class_analysis_lookup_review_dispositions([pair_key]) == {}
        after_clear_rows = (
            api._class_analysis_vignette_training_export_rows(captured)
        )
        assert after_clear_rows["geometry_decisions"] == []
        assert "current_pair_review_state_changed" in next(
            row["reasons"]
            for row in after_clear_rows["excluded"]
            if row["action_id"] == captured[0]["action_id"]
        )
    finally:
        api.CLASS_ANALYSIS_JOBS.pop("ca_pair", None)
        with api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS_LOCK:
            api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS.pop(review_id, None)


def test_dual_bbox_annotation_transaction_is_single_image_cas_and_idempotent(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    (class_root / "ca_txn").mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 100), (40, 60, 80)).save(image_path)
    before_lines = [
        "0 0.5 0.5 0.4 0.4",
        "1 0.5 0.5 0.4 0.4",
    ]
    after_lines = ["1 0.5 0.5 0.4 0.4"]
    state = {"lines": list(before_lines), "save_payloads": []}
    concurrent_started = __import__("threading").Event()
    concurrent_finished = __import__("threading").Event()
    concurrent_outcome = {}
    concurrent_thread = None
    conflict = {
        "enabled": True,
        "point_id": "bike",
        "current_class": "Bike",
        "other_point_id": "person",
        "other_class_name": "Person",
        "class_name": "Person",
        "target_bbox_xyxy": [30, 30, 70, 70],
        "other_bbox_xyxy": [30, 30, 70, 70],
        "split": "train",
        "image_relpath": "frame.jpg",
    }
    current_review_key = "cro_" + ("1" * 64)
    other_review_key = "cro_" + ("3" * 64)
    pair_review_key = api._class_analysis_dual_bbox_pair_key_from_object_keys(
        current_review_key,
        other_review_key,
    )
    point = {
        "point_id": "bike",
        "review_object_key": current_review_key,
        "source_mode": "linked",
        "source_id": "ds",
        "split": "train",
        "image_relpath": "frame.jpg",
        "class_name": "Bike",
        "bbox_xyxy": [30, 30, 70, 70],
        "image_sha256": api._class_analysis_file_sha256(image_path),
        "dual_bbox_conflict": conflict,
    }
    job = api.ClassAnalysisJob(
        job_id="ca_txn",
        status="completed",
        request={
            "source_mode": "linked",
            "dataset_id": "ds",
        },
        summary={
            "analysis_job_id": "ca_txn",
            "source_mode": "linked",
            "source_id": "ds",
        },
    )
    api.CLASS_ANALYSIS_JOBS[job.job_id] = job
    monkeypatch.setattr(
        api,
        "_class_analysis_thumbnail_point_metadata",
        lambda *_args, **_kwargs: dict(point),
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_validated_dual_bbox_conflict",
        lambda *_args, **_kwargs: (
            dict(conflict),
            pair_review_key,
            current_review_key,
            other_review_key,
        ),
    )
    entry = {
        "id": "ds",
        "dataset_root": str(tmp_path),
        "yolo_layout": "flat",
        "classes": ["Bike", "Person"],
    }
    lock = __import__("threading").RLock()
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _source_id: entry)
    monkeypatch.setattr(api, "_dataset_annotation_mutation_lock", lambda _entry: lock)
    monkeypatch.setattr(api, "_annotation_load_or_create_meta", lambda _entry: (tmp_path / "dataset.json", {}))
    monkeypatch.setattr(api, "_require_annotation_lock_owner", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(api, "_dataset_effective_root_from_entry", lambda _entry: tmp_path)
    monkeypatch.setattr(api, "_resolve_annotation_image_path", lambda *_args, **_kwargs: image_path)
    monkeypatch.setattr(api, "_annotation_effective_label_lines", lambda *_args, **_kwargs: list(state["lines"]))

    def save_one(_dataset_id, payload):
        nonlocal concurrent_thread
        state["save_payloads"].append(payload)
        assert len(payload["records"]) == 1
        if concurrent_thread is None:
            def try_concurrent_keep_both():
                concurrent_started.set()
                try:
                    api.record_class_analysis_review_disposition(
                        "ca_txn",
                        "bike",
                        {
                            "disposition": "keep_both_boxes",
                            "dual_bbox_conflict": conflict,
                            "origin": "desktop",
                            "client_action_id": "concurrent-keep-both",
                        },
                    )
                    concurrent_outcome["status"] = "recorded"
                except api.HTTPException as exc:
                    concurrent_outcome["status"] = "rejected"
                    concurrent_outcome["detail"] = exc.detail
                finally:
                    concurrent_finished.set()

            concurrent_thread = __import__("threading").Thread(
                target=try_concurrent_keep_both,
                daemon=True,
            )
            concurrent_thread.start()
            assert concurrent_started.wait(timeout=1.0)
            # The competing review mutation must remain behind the combined
            # annotation+ledger transaction until its deletion row is durable.
            assert not concurrent_finished.wait(timeout=0.05)
        state["lines"] = list(payload["records"][0]["label_lines"])
        return {"status": "saved"}

    monkeypatch.setattr(api, "save_dataset_annotation_snapshot", save_one)
    source_identity = api._annotation_image_source_identity(
        source_mode="linked",
        source_id="ds",
        split="train",
        image_relpath=Path("frame.jpg"),
        image_path=image_path,
        yolo_layout="flat",
    )
    request = {
        "operation_id": "dual_bbox:test-operation",
        "session_id": "editor",
        "capture_training_data": True,
        "action": "delete_current_box",
        "annotation_target": {
            "source_mode": "linked",
            "source_id": "ds",
            "split": "train",
            "image_relpath": "frame.jpg",
        },
        "dual_bbox_conflict": conflict,
        "expected_record_revision": api._annotation_image_label_revision(before_lines),
        "expected_source_identity": source_identity,
        "record": {
            "split": "train",
            "image_relpath": "frame.jpg",
            "label_lines": after_lines,
        },
    }
    try:
        with pytest.raises(api.HTTPException) as source_stale_exc:
            api.commit_class_analysis_dual_bbox_annotation_transaction(
                "ca_txn",
                "bike",
                {
                    **request,
                    "expected_source_identity": "asi1_" + ("0" * 64),
                },
            )
        assert (
            source_stale_exc.value.detail
            == "dual_bbox_annotation_source_identity_stale"
        )
        with pytest.raises(api.HTTPException) as extra_change_exc:
            api.commit_class_analysis_dual_bbox_annotation_transaction(
                "ca_txn",
                "bike",
                {
                    **request,
                    "record": {
                        **request["record"],
                        "label_lines": [
                            *after_lines,
                            "9 0.1 0.1 0.05 0.05",
                        ],
                    },
                },
            )
        assert (
            extra_change_exc.value.detail
            == "dual_bbox_annotation_non_target_changes"
        )
        with pytest.raises(api.HTTPException) as stale_exc:
            api.commit_class_analysis_dual_bbox_annotation_transaction(
                "ca_txn",
                "bike",
                {
                    **request,
                    "expected_record_revision": "alr1_" + ("0" * 64),
                },
            )
        assert stale_exc.value.detail == "dual_bbox_annotation_revision_stale"
        assert state["save_payloads"] == []
        committed = api.commit_class_analysis_dual_bbox_annotation_transaction(
            "ca_txn", "bike", request
        )
        assert concurrent_thread is not None
        concurrent_thread.join(timeout=2.0)
        assert not concurrent_thread.is_alive()
        replayed = api.commit_class_analysis_dual_bbox_annotation_transaction(
            "ca_txn", "bike", request
        )
        stored_pair_review = api._class_analysis_lookup_review_dispositions(
            [pair_review_key]
        )[pair_review_key]
        captured_actions = list(
            api._class_analysis_iter_vignette_training_actions()
        )
        export_rows = api._class_analysis_vignette_training_export_rows(
            captured_actions
        )
        verified = api._class_analysis_validate_dual_bbox_annotation_commit_attestation(
            job=job,
            point=point,
            conflict=conflict,
            pair_review_key=pair_review_key,
            action="delete_current_box",
            value=committed["annotation_commit_attestation"],
        )
    finally:
        api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)

    assert committed["status"] == "committed"
    assert committed["review_disposition"]["disposition"] == "delete_current_box"
    assert committed["review_disposition"]["client_action_id"] == request["operation_id"]
    assert replayed["status"] == "already_committed"
    assert replayed["review_disposition"]["idempotent_replay"] is True
    assert committed["review_disposition"]["training_capture"]["status"] == "recorded"
    assert (
        replayed["review_disposition"]["training_capture"]["status"]
        == "already_recorded"
    )
    assert (
        replayed["annotation_commit_attestation"]
        == committed["annotation_commit_attestation"]
    )
    stored_attestation = stored_pair_review["dual_bbox_pair_resolution"][
        "annotation_commit_attestation"
    ]
    assert stored_attestation == committed["annotation_commit_attestation"]
    assert len(captured_actions) == 1
    captured_attestation = captured_actions[0]["dual_bbox_resolution"][
        "annotation_commit_attestation"
    ]
    assert captured_attestation == stored_attestation
    assert len(export_rows["geometry_decisions"]) == 1
    assert export_rows["geometry_decisions"][0]["dual_bbox_resolution"][
        "annotation_commit_attestation"
    ] == stored_attestation
    assert concurrent_outcome["status"] == "rejected"
    assert concurrent_outcome["detail"]["code"] == "review_disposition_changed"
    assert len(state["save_payloads"]) == 1
    assert state["lines"] == after_lines
    assert verified["committed_revision"] == api._annotation_image_label_revision(after_lines)


def test_dual_bbox_annotation_transaction_route_is_wired():
    assert (
        "/class_analysis/jobs/{job_id}/points/{point_id}/dual_bbox_annotation_transaction"
        in {getattr(route, "path", "") for route in api.app.routes}
    )


def test_dual_bbox_annotation_transaction_rejects_job_id_alias_before_lookup(
    monkeypatch,
):
    monkeypatch.setattr(
        api,
        "_class_analysis_review_point_for_mutation",
        lambda *_args, **_kwargs: pytest.fail(
            "an invalid mutation job id must not reach job lookup"
        ),
    )
    with pytest.raises(api.HTTPException) as exc_info:
        api.commit_class_analysis_dual_bbox_annotation_transaction(
            "ca transaction alias",
            "p0",
            {},
        )
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "job_id_invalid"


def test_dual_bbox_annotation_transaction_rejects_conflicting_saved_pair_review(
    monkeypatch,
):
    pair_key = "crp_" + ("e" * 64)
    monkeypatch.setattr(
        api,
        "_class_analysis_review_point_for_mutation",
        lambda _job_id, _point_id: (
            api.ClassAnalysisJob(job_id="ca_pair_preflight", status="completed"),
            {"point_id": "p0"},
        ),
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_validated_dual_bbox_conflict",
        lambda *_args, **_kwargs: (
            {"other_point_id": "p1"},
            pair_key,
            "cro_" + ("1" * 64),
            "cro_" + ("2" * 64),
        ),
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_lookup_review_dispositions",
        lambda _keys: {
            pair_key: {
                "disposition": "keep_both_boxes",
                "dual_bbox_pair_resolution": {},
            }
        },
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_dual_bbox_annotation_source_contract",
        lambda **_kwargs: pytest.fail(
            "annotation-source resolution must not run after review conflict"
        ),
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.commit_class_analysis_dual_bbox_annotation_transaction(
            "ca_pair_preflight",
            "p0",
            {
                "operation_id": "dual_bbox:preflight",
                "action": "delete_current_box",
            },
        )
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == {
        "code": "dual_bbox_review_disposition_exists",
        "pair_review_key": pair_key,
    }


def test_dual_bbox_annotation_transaction_commits_transient_overlay(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        api,
        "CLASS_ANALYSIS_ROOT",
        tmp_path / "class_analysis",
    )
    dataset_root = tmp_path / "transient_dataset"
    (dataset_root / "images").mkdir(parents=True)
    (dataset_root / "labels").mkdir(parents=True)
    image_path = dataset_root / "images" / "frame.jpg"
    Image.new("RGB", (100, 100), (30, 50, 70)).save(image_path)
    before_lines = [
        "0 0.5 0.5 0.4 0.4",
        "1 0.5 0.5 0.4 0.4",
    ]
    after_lines = ["0 0.5 0.5 0.4 0.4"]
    (dataset_root / "labels" / "frame.txt").write_text(
        "\n".join(before_lines) + "\n",
        encoding="utf-8",
    )
    (dataset_root / "labelmap.txt").write_text(
        "Bike\nPerson\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "DATASET_LINK_ROOTS", [tmp_path.resolve()])
    monkeypatch.setattr(
        api,
        "DATASET_TRANSIENT_STATE_ROOT",
        tmp_path / "transient_state",
    )
    session_id = api.open_dataset_path(
        str(dataset_root),
        strict=True,
    )["session_id"]
    editor_session_id = "transient-editor"
    api.start_transient_annotation_session(
        session_id,
        {"session_id": editor_session_id, "editor_name": "test"},
    )
    manifest = api.get_transient_annotation_manifest(session_id)
    row = manifest["images"][0]
    conflict = {
        "enabled": True,
        "point_id": "bike",
        "current_class": "Bike",
        "other_point_id": "person",
        "other_class_name": "Person",
        "class_name": "Person",
        "target_bbox_xyxy": [30, 30, 70, 70],
        "other_bbox_xyxy": [30, 30, 70, 70],
        "split": "train",
        "image_relpath": "frame.jpg",
    }
    point = {
        "point_id": "bike",
        "review_object_key": "cro_" + ("4" * 64),
        "source_mode": "transient",
        "source_id": session_id,
        "split": "train",
        "image_relpath": "frame.jpg",
        "class_name": "Bike",
        "bbox_xyxy": [30, 30, 70, 70],
        "image_sha256": api._class_analysis_file_sha256(image_path),
        "dual_bbox_conflict": conflict,
    }
    job = api.ClassAnalysisJob(
        job_id="ca_transient_txn",
        status="completed",
        summary={
            "analysis_job_id": "ca_transient_txn",
            "source_mode": "transient",
            "source_id": session_id,
        },
    )
    api.CLASS_ANALYSIS_JOBS[job.job_id] = job
    monkeypatch.setattr(
        api,
        "_class_analysis_thumbnail_point_metadata",
        lambda *_args, **_kwargs: dict(point),
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_validated_dual_bbox_conflict",
        lambda *_args, **_kwargs: (
            dict(conflict),
            "crp_" + ("5" * 64),
            "cro_" + ("4" * 64),
            "cro_" + ("6" * 64),
        ),
    )
    request = {
        "operation_id": "dual_bbox:transient-test",
        "session_id": editor_session_id,
        "action": "delete_overlapping_box",
        "annotation_target": {
            "source_mode": "transient",
            "source_id": session_id,
            "split": "train",
            "image_relpath": "frame.jpg",
        },
        "dual_bbox_conflict": conflict,
        "expected_record_revision": row["annotation_record_revision"],
        "expected_source_identity": row["annotation_source_identity"],
        "record": {
            "split": "train",
            "image_relpath": "frame.jpg",
            "label_lines": after_lines,
        },
    }
    try:
        committed = api.commit_class_analysis_dual_bbox_annotation_transaction(
            job.job_id,
            point["point_id"],
            request,
        )
        persisted = api._resolve_transient_session(session_id)
        verified = api._class_analysis_validate_dual_bbox_annotation_commit_attestation(
            job=job,
            point=point,
            conflict=conflict,
            pair_review_key="crp_" + ("5" * 64),
            action="delete_overlapping_box",
            value=committed["annotation_commit_attestation"],
        )
    finally:
        api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)
        api.delete_transient_dataset(session_id)

    assert committed["status"] == "committed"
    assert committed["session_revision"] > manifest["session_revision"]
    assert persisted["overlay_labels"]["train:frame.jpg"] == after_lines
    assert verified["committed_revision"] == (
        api._annotation_image_label_revision(after_lines)
    )


@pytest.mark.parametrize("disposition", ["keep_both_boxes", "unresolved"])
def test_legacy_dual_bbox_pair_identity_is_backfilled_on_result_reload(
    tmp_path,
    monkeypatch,
    disposition,
):
    class_root = tmp_path / "class_analysis"
    workspace = class_root / "ca_legacy_pair_owner" / "active_workspace"
    images_dir = workspace / "images"
    labels_dir = workspace / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    image_path = images_dir / "frame.jpg"
    Image.new("RGB", (100, 100), (50, 70, 90)).save(image_path)
    (labels_dir / "frame.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n1 0.5 0.5 0.4 0.4\n",
        encoding="utf-8",
    )
    image_sha256 = api._class_analysis_file_sha256(image_path)

    def object_key(class_name):
        return api._class_analysis_review_object_key(
            source_key="active:cas_legacy_pair",
            image_sha256=image_sha256,
            split="train",
            image_relpath="frame.jpg",
            class_name=class_name,
            geometry={
                "kind": "bbox",
                "bbox_xyxy": [30, 30, 70, 70],
            },
            image_width=100,
            image_height=100,
        )

    bike_key = object_key("Bike")
    person_key = object_key("Person")

    def legacy_conflict(
        point_id,
        current_class,
        other_point_id,
        other_class,
    ):
        # This is the exact persisted shape from the compatibility window:
        # immutable point ids/classes/geometry exist, but pair identity does
        # not. The read path may hydrate its public copy only.
        return {
            "enabled": True,
            "kind": "near_identical_cross_class_bbox",
            "review_mode": "dual_bbox_annotation_resolution",
            "point_id": point_id,
            "current_class": current_class,
            "current_geometry_kind": "bbox",
            "other_point_id": other_point_id,
            "other_geometry_kind": "bbox",
            "target_bbox_xyxy": [30, 30, 70, 70],
            "other_bbox_xyxy": [30, 30, 70, 70],
            "other_class_name": other_class,
            "class_name": other_class,
            "split": "train",
            "image_relpath": "frame.jpg",
            "iou": 1.0,
            "corner_similarity": 1.0,
        }

    bike_conflict = legacy_conflict("bike", "Bike", "person", "Person")
    person_conflict = legacy_conflict("person", "Person", "bike", "Bike")
    bike = {
        "point_id": "bike",
        "review_object_key": bike_key,
        "source_mode": "active_workspace",
        "source_id": "cas_legacy_pair",
        "source_key": "active:cas_legacy_pair",
        "split": "train",
        "image_relpath": "frame.jpg",
        "image_sha256": image_sha256,
        "class_name": "Bike",
        "kind": "bbox",
        "bbox_xyxy": [30, 30, 70, 70],
        "is_wrong_class_candidate": True,
        "is_dual_bbox_conflict": True,
        "dual_bbox_conflict": bike_conflict,
        "review_signals": ["dual_bbox_conflict"],
    }
    person = {
        **bike,
        "point_id": "person",
        "review_object_key": person_key,
        "class_name": "Person",
        "dual_bbox_conflict": person_conflict,
    }
    result = {
        "summary": {
            "analysis_job_id": "ca_legacy_pair",
            "analysis_scope": "all_classes",
            "source_mode": "active_workspace",
            "source_id": "cas_legacy_pair",
            "source_key": "active:cas_legacy_pair",
            "wrong_class_candidate_count": 2,
        },
        "points": [bike, person],
        "wrong_class_candidates": [dict(bike), dict(person)],
    }
    job_dir = class_root / "ca_legacy_pair"
    job_dir.mkdir(parents=True)
    result_path = job_dir / "result.json"
    api._class_analysis_write_json(result_path, job_dir, result)
    raw_before = result_path.read_bytes()
    job = api.ClassAnalysisJob(
        job_id="ca_legacy_pair",
        status="completed",
        result_path=str(result_path),
        request={
            "source_mode": "active_workspace",
            "workspace_id": "ca_legacy_pair_owner",
            "snapshot_id": "cas_legacy_pair",
            "workspace_dir": str(workspace),
            "yolo_layout": "flat",
        },
        summary=dict(result["summary"]),
    )
    api.CLASS_ANALYSIS_JOBS[job.job_id] = job
    monkeypatch.setattr(
        api,
        "_class_analysis_thumbnail_point_metadata",
        lambda *_args, **_kwargs: dict(bike),
    )
    try:
        recorded = api.record_class_analysis_review_disposition(
            job.job_id,
            "bike",
            {
                "disposition": disposition,
                "dual_bbox_conflict": bike_conflict,
                "origin": "desktop",
                "client_action_id": f"pair-restart-{disposition}",
            },
        )
        pair_key = recorded["pair_review_key"]
        assert pair_key == api._class_analysis_dual_bbox_pair_key_from_object_keys(
            bike_key,
            person_key,
        )

        reloaded = api.get_class_analysis_result(job.job_id)
        assert reloaded["wrong_class_candidates"] == []
        assert reloaded["summary"]["human_review_disposition_counts"] == {
            disposition: 1
        }
        reloaded_points = {
            row["point_id"]: row for row in reloaded["points"]
        }
        assert reloaded_points["bike"]["human_review_disposition"] == disposition
        assert reloaded_points["bike"]["human_review_pair_key"] == pair_key
        assert "human_review_disposition" not in reloaded_points["person"]
        assert (
            reloaded_points["person"]["dual_bbox_conflict"][
                "pair_review_key"
            ]
            == pair_key
        )

        # A selected-class result can contain only the reverse side. Its
        # canonical other-object key is reconstructed from the bound source
        # geometry, and the same ledger decision remains visible.
        reverse_only = api._class_analysis_apply_review_dispositions(
            {
                "summary": {"analysis_scope": "selected_class"},
                "points": [api._class_analysis_public_point(dict(person))],
                "within_class_outlier_candidates": [dict(person)],
            },
            job=job,
        )
        assert reverse_only["within_class_outlier_candidates"] == []
        assert (
            reverse_only["points"][0]["human_review_disposition"]
            == disposition
        )
        assert reverse_only["points"][0]["human_review_pair_key"] == pair_key

        # Reload migration is strictly in-memory: immutable result artifacts
        # and annotation labels are not repaired or rewritten as a side effect.
        assert result_path.read_bytes() == raw_before
        assert "pair_review_key" not in json.loads(
            result_path.read_text(encoding="utf-8")
        )["wrong_class_candidates"][0]["dual_bbox_conflict"]
        assert (labels_dir / "frame.txt").read_text(encoding="utf-8") == (
            "0 0.5 0.5 0.4 0.4\n1 0.5 0.5 0.4 0.4\n"
        )
    finally:
        api.CLASS_ANALYSIS_JOBS.pop(job.job_id, None)


def test_class_analysis_review_disposition_shard_updates_are_thread_safe(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", tmp_path / "class_analysis")
    key = api._class_analysis_review_object_key(
        source_key="linked:dataset",
        image_sha256="44" * 32,
        split="train",
        image_relpath="frame.jpg",
        class_name="Boat",
        geometry={"kind": "bbox", "bbox_xyxy": [1, 2, 3, 4]},
    )
    result = {
        "summary": {
            "source_mode": "linked",
            "source_id": "dataset",
            "source_key": "linked:dataset",
        }
    }
    point = {
        "point_id": "p0",
        "review_object_key": key,
        "split": "train",
        "image_relpath": "frame.jpg",
        "class_name": "Boat",
        "bbox_xyxy": [1, 2, 3, 4],
        "wrong_class_suspicion": 0.8,
    }

    with api.ThreadPoolExecutor(max_workers=8) as executor:
        writes = list(
            executor.map(
                lambda index: api._class_analysis_record_review_disposition_entry(
                    result=result,
                    point=point,
                    disposition="confirm_current" if index % 2 else "skip",
                    origin=f"thread_{index}",
                ),
                range(24),
            )
        )

    assert len(writes) == 24
    stored = api._class_analysis_lookup_review_dispositions([key])
    assert list(stored) == [key]
    assert stored[key]["disposition"] in {"confirm_current", "skip"}
    shard_path = next(
        (tmp_path / "class_analysis" / "audit" / "human_review_dispositions").glob(
            "*.json"
        )
    )
    payload = json.loads(shard_path.read_text(encoding="utf-8"))
    assert payload["schema"] == api.CLASS_ANALYSIS_REVIEW_DISPOSITION_SCHEMA
    assert list(payload["entries"]) == [key]


def test_class_analysis_corrupt_review_shard_reads_and_writes_closed(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    key = api._class_analysis_review_object_key(
        source_key="linked:dataset",
        image_sha256="55" * 32,
        split="train",
        image_relpath="frame.jpg",
        class_name="Boat",
        geometry={"kind": "bbox", "bbox_xyxy": [1, 2, 3, 4]},
    )
    shard_id, _lock = api._class_analysis_review_disposition_shard(key)
    ledger_root = class_root / "audit" / "human_review_dispositions"
    ledger_root.mkdir(parents=True)
    shard_path = ledger_root / f"{shard_id}.json"
    shard_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(api.HTTPException) as read_exc_info:
        api._class_analysis_lookup_review_dispositions([key])
    assert read_exc_info.value.status_code == 500
    assert read_exc_info.value.detail == "review_disposition_ledger_invalid"
    with pytest.raises(api.HTTPException) as exc_info:
        api._class_analysis_record_review_disposition_entry(
            result={"summary": {"source_key": "linked:dataset"}},
            point={
                "point_id": "p0",
                "review_object_key": key,
                "split": "train",
                "image_relpath": "frame.jpg",
                "class_name": "Boat",
                "bbox_xyxy": [1, 2, 3, 4],
            },
            disposition="skip",
            origin="desktop",
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "review_disposition_ledger_invalid"
    assert shard_path.read_text(encoding="utf-8") == "{not-json"


def test_class_analysis_startup_cleanup_purges_runtime_but_preserves_audit_and_exports(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    cache_root = class_root / "cache"
    review_dir = class_root / "ca_old" / "qwen_reviews" / "review_1"
    review_dir.mkdir(parents=True)
    (review_dir / "events.jsonl").write_text('{"type":"model_output"}\n', encoding="utf-8")
    cache_root.mkdir(parents=True)
    (cache_root / "stale.bin").write_bytes(b"cache")
    (class_root / "runtime").mkdir()
    (class_root / "runtime" / "state.json").write_text("{}", encoding="utf-8")
    (class_root / "exports").mkdir()
    (class_root / "exports" / "keep.json").write_text("{}", encoding="utf-8")
    (class_root / "bench_reference").mkdir()
    (class_root / "bench_reference" / "keep.json").write_text("{}", encoding="utf-8")
    review_ledger = class_root / "audit" / "human_review_dispositions" / "ab.json"
    review_ledger.parent.mkdir(parents=True)
    review_ledger.write_text('{"schema":"review-ledger"}', encoding="utf-8")
    (class_root / ".purge" / "interrupted").mkdir(parents=True)
    (class_root / ".purge" / "interrupted" / "stale.bin").write_bytes(b"stale")
    (class_root / "analysis.sqlite").write_bytes(b"sqlite")
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.update(
        {
            "status": "not_started",
            "started_at": None,
            "ready_at": None,
            "completed_at": None,
            "targets": 0,
            "error": None,
        }
    )

    api._class_analysis_startup_cleanup_worker()

    assert api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE["status"] == "completed"
    assert not (class_root / "ca_old").exists()
    assert not (class_root / "runtime").exists()
    assert not (class_root / "analysis.sqlite").exists()
    assert cache_root.is_dir()
    assert (cache_root / "stale.bin").read_bytes() == b"cache"
    assert (class_root / "audit" / "ca_old" / "review_1" / "events.jsonl").is_file()
    assert review_ledger.read_text(encoding="utf-8") == '{"schema":"review-ledger"}'
    assert (class_root / "exports" / "keep.json").is_file()
    assert (class_root / "bench_reference" / "keep.json").is_file()
    assert not (class_root / ".purge").exists()


def test_class_analysis_disabled_startup_purge_restores_active_snapshot(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    workspace = class_root / "ca_failed" / "active_workspace"
    workspace.mkdir(parents=True)
    snapshot = {
        "snapshot_id": "cas_restore",
        "workspace_dir": str(workspace),
        "images": [
            {"image_relpath": "frame.jpg", "image_sha256": "ab" * 32}
        ],
        "labelmap": ["Boat", "Building"],
        "dataset_label": "Active Label Images workspace",
        "created_at": 123.0,
    }
    (workspace / "snapshot.json").write_text(
        json.dumps(snapshot),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    monkeypatch.setattr(
        api,
        "CLASS_ANALYSIS_CACHE_ROOT",
        class_root / "cache",
    )
    monkeypatch.setattr(api, "CLASS_ANALYSIS_PURGE_ON_START", False)
    with api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS_LOCK:
        api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS.clear()
    api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.update(
        {
            "status": "not_started",
            "started_at": None,
            "ready_at": None,
            "completed_at": None,
            "targets": 0,
            "error": None,
        }
    )

    api._start_class_analysis_startup_cleanup()

    assert api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE["status"] == "disabled"
    assert (
        api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE[
            "restored_active_snapshots"
        ]
        == 1
    )
    with api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS_LOCK:
        restored = dict(
            api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS["cas_restore"]
        )
    assert restored == snapshot


def test_class_analysis_disabled_purge_rebases_moved_active_snapshot(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "moved" / "class_analysis"
    workspace = class_root / "ca_failed" / "active_workspace"
    workspace.mkdir(parents=True)
    snapshot = {
        "snapshot_id": "cas_moved",
        "workspace_dir": str(
            tmp_path / "old_machine" / "ca_failed" / "active_workspace"
        ),
        "images": [
            {"image_relpath": "frame.jpg", "image_sha256": "cd" * 32}
        ],
        "labelmap": ["Boat"],
        "dataset_label": "Moved active workspace",
        "created_at": 456.0,
    }
    (workspace / "snapshot.json").write_text(
        json.dumps(snapshot),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    with api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS_LOCK:
        api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS.clear()

    try:
        restored_count = (
            api._class_analysis_restore_active_snapshots_from_disk()
        )
        with api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS_LOCK:
            restored = dict(
                api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS["cas_moved"]
            )
    finally:
        with api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS_LOCK:
            api.CLASS_ANALYSIS_ACTIVE_SNAPSHOTS.clear()

    assert restored_count == 1
    assert restored["workspace_dir"] == str(workspace.resolve())
    assert restored["images"] == snapshot["images"]
    assert restored["labelmap"] == snapshot["labelmap"]


def test_class_analysis_cache_budget_removes_oldest_files_only(
    tmp_path,
    monkeypatch,
):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    oldest = cache_root / "oldest.npz"
    newest = cache_root / "newest.npz"
    oldest.write_bytes(b"a" * 80)
    newest.write_bytes(b"b" * 80)
    oldest.touch()
    newest.touch()
    os.utime(oldest, (1, 1))
    os.utime(newest, (2, 2))
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_MAX_BYTES", 100)

    result = api._class_analysis_enforce_cache_budget()

    assert not oldest.exists()
    assert newest.exists()
    assert result["bytes"] == 80
    assert result["removed_bytes"] == 80


def test_class_analysis_cache_budget_preserves_live_atomic_temp_files(
    tmp_path,
    monkeypatch,
):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    live_temp = cache_root / "bank.npz.writer.tmp"
    stale_temp = cache_root / "abandoned.npz.writer.tmp"
    ordinary = cache_root / "ordinary.npz"
    live_temp.write_bytes(b"l" * 80)
    stale_temp.write_bytes(b"s" * 20)
    ordinary.write_bytes(b"o" * 80)
    now = time.time()
    os.utime(live_temp, (now, now))
    os.utime(stale_temp, (1, 1))
    os.utime(ordinary, (2, 2))
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_MAX_BYTES", 100)
    monkeypatch.setattr(
        api,
        "CLASS_ANALYSIS_CACHE_STALE_TEMP_SECONDS",
        60,
    )

    result = api._class_analysis_enforce_cache_budget()

    assert live_temp.read_bytes() == b"l" * 80
    assert not stale_temp.exists()
    assert not ordinary.exists()
    assert result["bytes"] == 80
    assert result["files"] == 1


def test_class_analysis_stream_encoder_bounds_batches_and_reuses_image_packs(
    tmp_path,
    monkeypatch,
):
    cache_root = tmp_path / "cache"
    out_dir = tmp_path / "job"
    out_dir.mkdir()
    image_paths = []
    for image_idx in range(2):
        image_path = tmp_path / f"image-{image_idx}.png"
        Image.new(
            "RGB",
            (96, 72),
            (20 + image_idx * 30, 40, 80),
        ).save(image_path)
        image_paths.append(image_path)
    records = []
    for idx in range(5):
        image_idx = 0 if idx < 3 else 1
        records.append(
            {
                "point_id": f"p{idx}",
                "bbox_xyxy": [10 + idx, 10, 30 + idx, 30],
                "crop_cache_key": f"crop-{idx}",
                "image_pack_key": f"pack-{image_idx}",
                "_image_path": str(image_paths[image_idx]),
                "crop_cache_reused": False,
            }
        )
    calls = []

    def fake_encode(items, **_kwargs):
        calls.append(len(items))
        return np.asarray(
            [[float(len(calls)), float(idx + 1), 1.0] for idx in range(len(items))],
            dtype=np.float32,
        )

    stable_memory = {
        "backend_rss_bytes": 100,
        "worker_rss_bytes": 200,
        "combined_rss_bytes": 300,
        "system_available_bytes": 10_000,
        "system_total_bytes": 20_000,
    }
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    monkeypatch.setattr(api, "_class_analysis_memory_snapshot", lambda: dict(stable_memory))
    monkeypatch.setattr(api, "_encode_embedding_items_for_head", fake_encode)
    payload = {
        "analysis_scope": "all_classes",
        "crop_mode": "padded_square",
        "padding_ratio": 0.08,
        "preprocess_mode": "canonical",
        "canonical_size": 64,
        "background_mode": "full_crop",
        "embedding_view_mode": "single",
    }
    head = {
        "encoder_type": "dinov3",
        "encoder_model": "unit",
        "normalize_embeddings": True,
        "dinov3_pooling": "pooler",
    }
    job = api.ClassAnalysisJob(job_id="stream", request=dict(payload))

    encoded = api._class_analysis_stream_encode_records(
        records,
        payload,
        job=job,
        head=head,
        batch_size=2,
        out_dir=out_dir,
    )

    assert encoded.shape == (5, 3)
    assert calls == [2, 2, 1]
    assert max(calls) <= 2
    assert job.runtime["completed_objects"] == 5
    assert api._class_analysis_load_image_pack(
        "pack-0",
        head,
        ["crop-0", "crop-1", "crop-2"],
    )
    assert api._class_analysis_load_image_pack(
        "pack-1",
        head,
        ["crop-3", "crop-4"],
    )

    monkeypatch.setattr(
        api,
        "_encode_embedding_items_for_head",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("warm cache must bypass encoder")
        ),
    )
    warm_out = tmp_path / "warm-job"
    warm_out.mkdir()
    warm_job = api.ClassAnalysisJob(job_id="warm", request=dict(payload))
    cache_stats = {}
    warm = api._class_analysis_stream_encode_records(
        copy.deepcopy(records),
        payload,
        job=warm_job,
        head=head,
        batch_size=2,
        out_dir=warm_out,
        cache_stats=cache_stats,
    )

    assert np.allclose(warm, encoded)
    assert cache_stats == {"hits": 5, "misses": 0, "errors": 0, "total": 5}


def test_class_analysis_stage1_checkpoint_ignores_refinement_contract_and_migrates_legacy_key(
    tmp_path,
    monkeypatch,
):
    cache_root = tmp_path / "cache"
    source_dir = tmp_path / "source_job"
    resumed_dir = tmp_path / "resumed_job"
    legacy_resumed_dir = tmp_path / "legacy_resumed_job"
    migrated_resumed_dir = tmp_path / "migrated_resumed_job"
    source_dir.mkdir()
    resumed_dir.mkdir()
    legacy_resumed_dir.mkdir()
    migrated_resumed_dir.mkdir()
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    records = [
        {
            **_record("p0", "Boat"),
            "review_object_key": "review-p0",
            "class_id": 0,
            "_image_sha256": "11" * 32,
            "crop_cache_key": "crop-p0",
        },
        {
            **_record("p1", "Building"),
            "review_object_key": "review-p1",
            "class_id": 1,
            "_image_sha256": "22" * 32,
            "crop_cache_key": "crop-p1",
        },
    ]
    base_request = {
        "run_fingerprint": "aa" * 32,
        "crop_mode": "padded_square",
        "padding_ratio": 0.08,
        "preprocess_mode": "canonical",
        "canonical_size": 336,
        "background_mode": "full_crop",
        "embedding_view_mode": "single",
        "embedding_adjustment": "none",
        "refine_outliers": True,
        "refinement_schema": "legacy-refinement-schema",
        "refinement_decision_contract": "legacy-refinement-decision",
        "projection": "pca",
        "neighbor_k": 15,
    }
    head = {
        "encoder_type": "dinov3",
        "encoder_model": "unit",
        "normalize_embeddings": True,
        "dinov3_pooling": "pooler",
        "cradio_pooling": "summary",
        "embedding_aggregation": "pooled",
    }
    source_path = source_dir / "embeddings.npy"
    np.save(
        source_path,
        np.asarray([[0.6, 0.8], [0.0, 1.0]], dtype=np.float32),
        allow_pickle=False,
    )
    source_matrix = np.load(source_path, mmap_mode="r", allow_pickle=False)
    try:
        published = api._class_analysis_publish_embedding_checkpoint(
            embeddings_path=source_path,
            embeddings=source_matrix,
            records=records,
            request=base_request,
            head=head,
            adjustment_info={"mode": "none", "applied": False},
        )
    finally:
        api._class_analysis_close_memmap_arrays(source_matrix)
    assert published is not None
    stage1_fingerprint = api._class_analysis_stage1_embedding_fingerprint(
        records,
        base_request,
        head,
    )
    assert published["embedding_fingerprint"] == stage1_fingerprint
    assert stage1_fingerprint != base_request["run_fingerprint"]

    v3_request = {
        **base_request,
        "run_fingerprint": "bb" * 32,
        "refinement_schema": api.CLASS_ANALYSIS_REFINEMENT_SCHEMA,
        "refinement_decision_contract": (
            api.CLASS_ANALYSIS_REFINEMENT_DECISION_CONTRACT
        ),
        "projection": "umap",
        "neighbor_k": 31,
    }
    assert api._class_analysis_stage1_embedding_fingerprint(
        records,
        v3_request,
        head,
    ) == stage1_fingerprint
    matrix, info = api._class_analysis_load_embedding_checkpoint(
        records=records,
        request=v3_request,
        head=head,
        out_dir=resumed_dir,
        copy_on_write=True,
    )
    try:
        assert matrix is not None
        assert np.allclose(matrix, [[0.6, 0.8], [0.0, 1.0]])
        assert info["used"] is True
        assert info["checkpoint_source"] == "stage1_fingerprint"
    finally:
        api._class_analysis_close_memmap_arrays(matrix)

    # Simulate the run-keyed layout written before the Stage-1 key split.
    stage1_dir = api._class_analysis_resume_checkpoint_dir(
        stage1_fingerprint,
        create=False,
    )
    legacy_dir = api._class_analysis_resume_checkpoint_dir(
        base_request["run_fingerprint"],
        create=True,
    )
    assert stage1_dir is not None and legacy_dir is not None
    for child in list(legacy_dir.iterdir()):
        child.unlink()
    for child in list(stage1_dir.iterdir()):
        child.rename(legacy_dir / child.name)
    stage1_dir.rmdir()
    legacy_manifest_path = (
        legacy_dir / api.CLASS_ANALYSIS_RESUME_CHECKPOINT_MANIFEST
    )
    legacy_manifest = json.loads(legacy_manifest_path.read_text("utf-8"))
    legacy_manifest.pop("embedding_fingerprint", None)
    legacy_manifest.pop("embedding_fingerprint_schema", None)
    legacy_manifest_path.write_text(json.dumps(legacy_manifest), "utf-8")

    matrix, info = api._class_analysis_load_embedding_checkpoint(
        records=records,
        request=v3_request,
        head=head,
        out_dir=legacy_resumed_dir,
        copy_on_write=True,
    )
    try:
        assert matrix is not None
        assert info["checkpoint_source"] == "validated_legacy_run_fingerprint"
        assert info["migrated_to_stage1_fingerprint"] is True
    finally:
        api._class_analysis_close_memmap_arrays(matrix)

    matrix, info = api._class_analysis_load_embedding_checkpoint(
        records=records,
        request=v3_request,
        head=head,
        out_dir=migrated_resumed_dir,
        copy_on_write=True,
    )
    try:
        assert matrix is not None
        assert info["checkpoint_source"] == "stage1_fingerprint"
    finally:
        api._class_analysis_close_memmap_arrays(matrix)

    changed_recipe = {**v3_request, "canonical_size": 448}
    assert api._class_analysis_stage1_embedding_fingerprint(
        records,
        changed_recipe,
        head,
    ) != stage1_fingerprint


def test_class_analysis_checkpoint_attach_revalidates_copied_inode(
    tmp_path,
    monkeypatch,
):
    cache_root = tmp_path / "cache"
    source_dir = tmp_path / "source_job"
    resumed_dir = tmp_path / "resumed_job"
    source_dir.mkdir()
    resumed_dir.mkdir()
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    records = [
        {
            **_record("p0", "Boat"),
            "review_object_key": "review-p0",
            "class_id": 0,
            "_image_sha256": "11" * 32,
            "crop_cache_key": "crop-p0",
        },
        {
            **_record("p1", "Building"),
            "review_object_key": "review-p1",
            "class_id": 1,
            "_image_sha256": "22" * 32,
            "crop_cache_key": "crop-p1",
        },
    ]
    request = {
        "run_fingerprint": "ef" * 32,
        "crop_mode": "padded_square",
        "padding_ratio": 0.08,
        "preprocess_mode": "canonical",
        "canonical_size": 336,
        "background_mode": "full_crop",
        "embedding_view_mode": "single",
        "embedding_adjustment": "none",
    }
    head = {
        "encoder_type": "dinov3",
        "encoder_model": "unit",
        "normalize_embeddings": True,
        "dinov3_pooling": "pooler",
        "cradio_pooling": "summary",
        "embedding_aggregation": "pooled",
    }
    source_path = source_dir / "embeddings.npy"
    np.save(
        source_path,
        np.asarray(
            [[0.6, 0.8, 0.0], [0.0, 5.0 / 13.0, 12.0 / 13.0]],
            dtype=np.float32,
        ),
        allow_pickle=False,
    )
    source_matrix = np.load(source_path, mmap_mode="r", allow_pickle=False)
    try:
        published = api._class_analysis_publish_embedding_checkpoint(
            embeddings_path=source_path,
            embeddings=source_matrix,
            records=records,
            request=request,
            head=head,
            adjustment_info={"mode": "none", "applied": False},
        )
    finally:
        api._class_analysis_close_memmap_arrays(source_matrix)
    assert published is not None

    original_link = api._class_analysis_link_checkpoint_artifact

    def attach_different_inode(source_path, target_path, *, root):
        if Path(root).resolve() == resumed_dir.resolve():
            api._class_analysis_write_npy(
                Path(target_path),
                Path(root),
                np.asarray(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=np.float32,
                ),
            )
            return Path(target_path)
        return original_link(
            Path(source_path),
            Path(target_path),
            root=Path(root),
        )

    monkeypatch.setattr(
        api,
        "_class_analysis_link_checkpoint_artifact",
        attach_different_inode,
    )

    matrix, info = api._class_analysis_load_embedding_checkpoint(
        records=records,
        request=request,
        head=head,
        out_dir=resumed_dir,
        copy_on_write=True,
    )

    assert matrix is None
    assert info == {
        "used": False,
        "reason": "class_analysis_embedding_checkpoint_attach_mismatch",
    }


def test_class_analysis_job_publishes_and_resumes_private_embedding_checkpoint(
    tmp_path,
    monkeypatch,
):
    class_root = tmp_path / "class_analysis"
    cache_root = tmp_path / "class_analysis_cache"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    monkeypatch.setattr(
        api,
        "_class_analysis_memory_snapshot",
        lambda: {
            "backend_rss_bytes": 100,
            "worker_rss_bytes": 100,
            "combined_rss_bytes": 200,
            "system_available_bytes": 1,
            "system_total_bytes": 20_000,
        },
    )
    records = [
        {
            **_record("p0", "Boat"),
            "review_object_key": "review-p0",
            "class_id": 0,
            "_image_sha256": "11" * 32,
            "crop_cache_key": "crop-p0",
        },
        {
            **_record("p1", "Building"),
            "review_object_key": "review-p1",
            "class_id": 1,
            "_image_sha256": "22" * 32,
            "crop_cache_key": "crop-p1",
        },
    ]

    def collect_records(*_args, **_kwargs):
        return (
            copy.deepcopy(records),
            [],
            {
                "source_mode": "linked",
                "source_id": "unit",
                "_spatial_context_records": [],
            },
        )

    monkeypatch.setattr(
        api,
        "_class_analysis_collect_records",
        collect_records,
    )
    encoded_maps = []
    encode_job_ids = []

    def encode_records(
        _records,
        _request,
        *,
        job,
        out_dir,
        **_kwargs,
    ):
        encode_job_ids.append(job.job_id)
        work_dir = out_dir / "work"
        work_dir.mkdir(parents=True, exist_ok=True)
        matrix = np.lib.format.open_memmap(
            work_dir / "raw_embeddings.npy",
            mode="w+",
            dtype=np.float32,
            shape=(2, 3),
        )
        matrix[:] = np.asarray(
            [[3.0, 4.0, 0.0], [0.0, 5.0, 12.0]],
            dtype=np.float32,
        )
        matrix.flush()
        encoded_maps.append(matrix)
        return matrix

    monkeypatch.setattr(
        api,
        "_class_analysis_stream_encode_records",
        encode_records,
    )
    original_adjust = api._class_analysis_adjust_embeddings_to_memmap
    adjusted_maps = []

    def track_adjusted_map(*args, **kwargs):
        matrix, info = original_adjust(*args, **kwargs)
        adjusted_maps.append(matrix)
        return matrix, info

    monkeypatch.setattr(
        api,
        "_class_analysis_adjust_embeddings_to_memmap",
        track_adjusted_map,
    )
    graph_inputs = []
    graph_jobs = []

    def build_result(_records, embeddings, *, summary, job, **_kwargs):
        graph_jobs.append(job.job_id)
        assert embeddings.flags.writeable is True
        assert getattr(embeddings, "mode", None) == "c"
        original_value = float(embeddings[0, 0])
        embeddings[0, 0] = original_value + 10.0
        persisted = np.load(
            api._class_analysis_job_dir(job.job_id, create=False)
            / "embeddings.npy",
            mmap_mode="r",
        )
        assert float(persisted[0, 0]) == pytest.approx(original_value)
        api._class_analysis_close_memmap_arrays(persisted)
        embeddings[0, 0] = original_value
        graph_inputs.append(np.asarray(embeddings).copy())
        if job.job_id == "ca_checkpoint_fresh":
            assert encoded_maps[0]._mmap.closed is True
            assert adjusted_maps[0]._mmap.closed is True
            checkpoint_dir = api._class_analysis_resume_checkpoint_dir(
                api._class_analysis_stage1_embedding_fingerprint(
                    records,
                    job.request,
                    {
                        "encoder_type": "dinov3",
                        "encoder_model": "unit",
                        "normalize_embeddings": True,
                        "dinov3_pooling": "pooler",
                        "cradio_pooling": "summary",
                        "embedding_aggregation": "pooled",
                        "embedding_salad_head_id": "",
                    },
                ),
                create=False,
            )
            assert checkpoint_dir is not None
            assert (checkpoint_dir / "embeddings.npy").is_file()
            assert (
                checkpoint_dir
                / api.CLASS_ANALYSIS_RESUME_CHECKPOINT_MANIFEST
            ).is_file()
            raise RuntimeError("stop_after_fresh_graph_probe")
        return {
            "summary": dict(summary),
            "points": [],
            "projection_options": {"coordinates": {}},
            "wrong_class_candidates": [],
        }

    monkeypatch.setattr(api, "_class_analysis_build_result", build_result)
    refinement_jobs = []

    def refine_result(*, job, **_kwargs):
        refinement_jobs.append(job.job_id)
        raise RuntimeError("cancelled")

    monkeypatch.setattr(api, "_class_analysis_refine_result", refine_result)
    persisted_cancellations = []
    monkeypatch.setattr(
        api,
        "_class_analysis_persist_cancelled_refinement_result",
        lambda **kwargs: persisted_cancellations.append(kwargs["job"].job_id),
    )
    request = {
        "run_fingerprint": "ab" * 32,
        "snapshot_id": "cas_checkpoint",
        "encoder_type": "dinov3",
        "encoder_model": "unit",
        "embedding_adjustment": "none",
        "refine_outliers": True,
        "refinement_decision_contract": (
            api.CLASS_ANALYSIS_REFINEMENT_DECISION_CONTRACT
        ),
        "selector_priority_contract": (
            api.CLASS_ANALYSIS_SELECTOR_PRIORITY_CONTRACT
        ),
        **_class_analysis_test_selector_v6_binding(),
        "capture_group_contract": api.CLASS_ANALYSIS_CAPTURE_GROUP_CONTRACT,
    }
    fresh_job = api.ClassAnalysisJob(
        job_id="ca_checkpoint_fresh",
        request=dict(request),
    )

    api._run_class_analysis_job(fresh_job)

    assert fresh_job.status == "failed"
    assert fresh_job.error == "stop_after_fresh_graph_probe"
    fresh_resume = fresh_job.runtime["embedding_resume"]
    assert fresh_resume["used"] is False
    assert fresh_resume["published"] is True
    assert fresh_resume["checkpoint_schema"] == (
        api.CLASS_ANALYSIS_RESUME_CHECKPOINT_SCHEMA
    )
    assert fresh_resume["record_count"] == 2
    assert fresh_resume["reason"] == "fresh_embeddings_published"
    checkpoint_dir = api._class_analysis_resume_checkpoint_dir(
        fresh_resume["embedding_fingerprint"],
        create=False,
    )
    checkpoint_manifest = json.loads(
        (
            checkpoint_dir
            / api.CLASS_ANALYSIS_RESUME_CHECKPOINT_MANIFEST
        ).read_text("utf-8")
    )
    assert fresh_resume["checkpoint_sha256"] == (
        checkpoint_manifest["embedding_sha256"]
    )
    fresh_manifest = json.loads(
        (
            api._class_analysis_job_dir(
                "ca_checkpoint_fresh",
                create=False,
            )
            / "embeddings_manifest.json"
        ).read_text("utf-8")
    )
    assert fresh_manifest["checkpoint_sha256"] == (
        checkpoint_manifest["embedding_sha256"]
    )
    assert fresh_manifest["embedding_fingerprint"] == (
        fresh_resume["embedding_fingerprint"]
    )
    assert re.fullmatch(r"[0-9a-f]{64}", fresh_manifest["record_digest"])
    assert re.fullmatch(r"[0-9a-f]{64}", fresh_manifest["recipe_digest"])
    resumed_job = api.ClassAnalysisJob(
        job_id="ca_checkpoint_resumed",
        request=dict(request),
    )

    api._run_class_analysis_job(resumed_job)

    assert resumed_job.status == "cancelled"
    assert encode_job_ids == ["ca_checkpoint_fresh"]
    assert len(adjusted_maps) == 1
    assert graph_jobs == ["ca_checkpoint_fresh", "ca_checkpoint_resumed"]
    assert np.allclose(graph_inputs[0], graph_inputs[1])
    assert np.allclose(
        graph_inputs[1],
        np.asarray(
            [[0.6, 0.8, 0.0], [0.0, 5.0 / 13.0, 12.0 / 13.0]],
            dtype=np.float32,
        ),
    )
    assert resumed_job.runtime["embedding_resume"]["used"] is True
    assert resumed_job.runtime["embedding_resume"]["record_count"] == 2
    assert refinement_jobs == ["ca_checkpoint_resumed"]
    assert persisted_cancellations == ["ca_checkpoint_resumed"]
    resumed_dir = api._class_analysis_job_dir(
        "ca_checkpoint_resumed",
        create=False,
    )
    assert not (resumed_dir / "raw_embeddings.npy").exists()
    resumed_manifest = json.loads(
        (resumed_dir / "embeddings_manifest.json").read_text("utf-8")
    )
    assert resumed_manifest["schema"] == (
        "class-analysis-embeddings-v5-resumed-memmap"
    )


def test_class_analysis_global_pca_streams_fit_and_transform_batches(monkeypatch):
    embeddings = np.asarray(
        [
            [float(idx), float(idx % 3), float((idx * 2) % 5)]
            for idx in range(11)
        ],
        dtype=np.float32,
    )
    real_incremental_pca = api.IncrementalPCA
    fit_batch_sizes = []
    transform_batch_sizes = []

    class TrackingIncrementalPCA:
        def __init__(self, *args, **kwargs):
            self.reducer = real_incremental_pca(*args, **kwargs)

        def partial_fit(self, values):
            fit_batch_sizes.append(int(values.shape[0]))
            self.reducer.partial_fit(values)
            return self

        def transform(self, values):
            transform_batch_sizes.append(int(values.shape[0]))
            return self.reducer.transform(values)

    monkeypatch.setattr(api, "CLASS_ANALYSIS_PCA_STREAM_BATCH_SIZE", 4)
    monkeypatch.setattr(api, "IncrementalPCA", TrackingIncrementalPCA)

    coordinates = api._class_analysis_fit_pca_projection(
        embeddings,
        seed=42,
        label="Global PCA",
    )

    assert coordinates.shape == (11, 2)
    assert np.isfinite(coordinates).all()
    assert fit_batch_sizes == [4, 4, 3]
    assert transform_batch_sizes == [4, 4, 3]
    assert sum(fit_batch_sizes) == len(embeddings)


def test_class_analysis_approximate_neighbors_use_index_neighbor_graph(
    monkeypatch,
):
    embeddings = np.eye(4, dtype=np.float32)
    graph_indices = np.asarray(
        [
            [2, 0, 1],
            [3, 0, 1],
            [2, 3, 0],
            [1, 3, 2],
        ],
        dtype=np.int32,
    )
    graph_distances = np.asarray(
        [
            [0.3, 0.0, 0.1],
            [0.4, 0.2, 0.0],
            [0.0, 0.1, 0.5],
            [0.4, 0.0, 0.2],
        ],
        dtype=np.float32,
    )
    constructed = []

    class FakeNNDescent:
        def __init__(self, values, **params):
            constructed.append((values, params))
            self.neighbor_graph = (graph_indices, graph_distances)

        def query(self, *_args, **_kwargs):
            raise AssertionError(
                "all-data query must not run after graph construction"
            )

    import pynndescent

    monkeypatch.setattr(pynndescent, "NNDescent", FakeNNDescent)

    indices, distances, params = api._class_analysis_approximate_neighbor_attempt(
        embeddings,
        neighbor_k=2,
        seed=42,
        stronger=False,
    )

    assert np.array_equal(
        indices,
        np.asarray([[1, 2], [0, 3], [3, 0], [2, 1]], dtype=np.int64),
    )
    assert np.allclose(
        distances,
        np.asarray([[0.1, 0.3], [0.2, 0.4], [0.1, 0.5], [0.2, 0.4]]),
    )
    assert len(constructed) == 1
    assert constructed[0][1]["low_memory"] is True
    assert params["n_neighbors"] == 3


def test_class_analysis_large_neighbor_path_is_audited_and_deterministic(
    monkeypatch,
):
    embeddings = np.eye(6, dtype=np.float32)
    approximate = np.asarray(
        [[(idx + 1) % 6, (idx + 2) % 6] for idx in range(6)],
        dtype=np.int64,
    )
    distances = np.full((6, 2), 0.5, dtype=np.float32)
    attempts = []

    def fake_attempt(_embeddings, *, neighbor_k, seed, stronger):
        attempts.append((neighbor_k, seed, stronger))
        return approximate, distances, {"n_neighbors": 32}

    monkeypatch.setattr(api, "CLASS_ANALYSIS_EXACT_NEIGHBOR_LIMIT", 3)
    monkeypatch.setattr(
        api,
        "_class_analysis_approximate_neighbor_attempt",
        fake_attempt,
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_neighbor_recall_audit",
        lambda *_args, **_kwargs: {
            "sample_count": 6,
            "recall_at_k": 0.97,
            "minimum_query_recall_at_k": 0.5,
        },
    )

    indices, result_distances, info = api._class_analysis_compute_neighbors(
        embeddings,
        neighbor_k=2,
        seed=42,
    )

    assert np.array_equal(indices, approximate)
    assert np.array_equal(result_distances, distances)
    assert info["method"] == "pynndescent"
    assert info["audit"]["recall_at_k"] == pytest.approx(0.97)
    assert attempts == [(2, 42, False)]


def test_class_analysis_stream_memory_telemetry_never_stops_analysis(
    tmp_path,
    monkeypatch,
):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (64, 64), (40, 70, 100)).save(image_path)
    records = [
        {
            "point_id": f"p{idx}",
            "bbox_xyxy": [10, 10, 30, 30],
            "crop_cache_key": f"crop-{idx}",
            "image_pack_key": "pack",
            "_image_path": str(image_path),
        }
        for idx in range(4)
    ]
    snapshots = [
        {
            "backend_rss_bytes": 100,
            "worker_rss_bytes": 100,
            "combined_rss_bytes": 200,
            "system_available_bytes": 1,
            "system_total_bytes": 20_000,
        }
    ]
    snapshots.extend(
        {
            "backend_rss_bytes": 200,
            "worker_rss_bytes": 200,
            "combined_rss_bytes": 400,
            "system_available_bytes": 1,
            "system_total_bytes": 20_000,
        }
        for _ in range(5)
    )
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(
        api,
        "_class_analysis_memory_snapshot",
        lambda: dict(snapshots.pop(0) if snapshots else {
            "backend_rss_bytes": 200,
            "worker_rss_bytes": 200,
            "combined_rss_bytes": 400,
            "system_available_bytes": 10_000,
            "system_total_bytes": 20_000,
        }),
    )
    monkeypatch.setattr(api, "CLASS_ANALYSIS_MEMORY_SAMPLE_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(
        api,
        "_encode_embedding_items_for_head",
        lambda items, **_kwargs: np.ones((len(items), 3), dtype=np.float32),
    )
    payload = {
        "analysis_scope": "all_classes",
        "crop_mode": "padded_square",
        "padding_ratio": 0.08,
        "preprocess_mode": "canonical",
        "canonical_size": 64,
        "background_mode": "full_crop",
        "embedding_view_mode": "single",
    }
    head = {
        "encoder_type": "dinov3",
        "encoder_model": "unit",
        "normalize_embeddings": True,
    }
    out_dir = tmp_path / "job"
    out_dir.mkdir()
    job = api.ClassAnalysisJob(job_id="memory-telemetry", request=dict(payload))

    encoded = api._class_analysis_stream_encode_records(
        records,
        payload,
        job=job,
        head=head,
        batch_size=1,
        out_dir=out_dir,
    )

    serialized = api._serialize_class_analysis_job(job)
    assert encoded.shape == (4, 3)
    assert serialized["runtime"]["completed_objects"] == 4
    assert serialized["runtime"]["memory"]["incremental_combined_rss_bytes"] == 200
    assert "budget_bytes" not in serialized["runtime"]["memory"]
    assert serialized["runtime"]["job_memory"][
        "job_start_baseline_combined_rss_bytes"
    ] == 200
    assert serialized["runtime"]["job_memory"][
        "peak_job_incremental_combined_rss_bytes"
    ] == 200
    assert "job_memory_budget_bytes" not in serialized["runtime"]["job_memory"]


def test_class_analysis_exact_neighbor_search_honors_cancellation():
    cancel_event = api.threading.Event()
    cancel_event.set()

    with pytest.raises(RuntimeError, match="cancelled"):
        api._class_analysis_exact_neighbors_chunked(
            np.eye(4, dtype=np.float32),
            neighbor_k=2,
            cancel_event=cancel_event,
        )


def test_class_analysis_startup_cleanup_fails_closed_on_symlinked_audit_root(
    tmp_path,
    monkeypatch,
):
    original_cleanup_state = dict(api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE)
    class_root = tmp_path / "class_analysis"
    cache_root = class_root / "cache"
    outside = tmp_path / "outside_audit"
    class_root.mkdir()
    cache_root.mkdir()
    outside.mkdir()
    try:
        (class_root / "audit").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.update(
        {
            "status": "not_started",
            "started_at": None,
            "ready_at": None,
            "completed_at": None,
            "targets": 0,
            "error": None,
        }
    )

    try:
        api._class_analysis_startup_cleanup_worker()

        assert api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE["status"] == "failed"
        assert api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE["error"] == "class_analysis_audit_root_invalid"
        assert list(outside.iterdir()) == []
    finally:
        api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.clear()
        api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.update(original_cleanup_state)


def test_class_analysis_cleanup_deletion_does_not_block_new_analysis(
    tmp_path,
    monkeypatch,
):
    original_cleanup_state = dict(api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE)
    class_root = tmp_path / "class_analysis"
    cache_root = class_root / "cache"
    cache_root.mkdir(parents=True)
    (cache_root / "stale.bin").write_bytes(b"cache")
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    monkeypatch.setattr(
        api,
        "_class_analysis_delete_quarantined",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("delete interrupted")),
    )
    api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.clear()
    api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.update(
        {
            "status": "not_started",
            "started_at": None,
            "ready_at": None,
            "completed_at": None,
            "targets": 0,
            "error": None,
        }
    )

    try:
        api._class_analysis_startup_cleanup_worker()

        assert api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE["status"] == "deletion_failed"
        assert api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE["ready_at"] is not None
        assert api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE["error"] == "delete interrupted"
        assert cache_root.is_dir()
        api._class_analysis_require_cleanup_complete()
    finally:
        api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.clear()
        api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.update(original_cleanup_state)


def test_class_analysis_cleanup_supports_external_persistent_cache(
    tmp_path,
    monkeypatch,
):
    original_cleanup_state = dict(api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE)
    class_root = tmp_path / "class_analysis"
    cache_root = tmp_path / "persistent_cache"
    class_root.mkdir()
    cache_root.mkdir()
    cache_file = cache_root / "warm.safetensors"
    cache_file.write_bytes(b"cache")
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)
    monkeypatch.setattr(api, "_class_analysis_delete_quarantined", lambda *_args: None)
    api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.clear()
    api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.update(
        {
            "status": "not_started",
            "started_at": None,
            "ready_at": None,
            "completed_at": None,
            "targets": 0,
            "error": None,
        }
    )

    try:
        api._class_analysis_startup_cleanup_worker()

        assert api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE["status"] == "completed"
        assert cache_file.read_bytes() == b"cache"
    finally:
        api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.clear()
        api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.update(original_cleanup_state)


def test_class_analysis_cleanup_blocks_only_during_quarantine():
    original_cleanup_state = dict(api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE)
    try:
        for status in ("queued", "quarantining"):
            api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE["status"] = status
            with pytest.raises(api.HTTPException) as exc_info:
                api._class_analysis_require_cleanup_complete()
            assert exc_info.value.status_code == 503
            assert exc_info.value.detail == "class_analysis_startup_cleanup_in_progress"

        for status in (
            "not_started",
            "deleting",
            "completed",
            "deletion_failed",
            "disabled",
            "test_skipped",
        ):
            api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE["status"] = status
            api._class_analysis_require_cleanup_complete()
    finally:
        api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.clear()
        api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.update(original_cleanup_state)


def test_system_health_reports_class_analysis_startup_readiness(monkeypatch):
    original_cleanup_state = dict(api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE)
    monkeypatch.setattr(api, "_storage_check_payload", lambda: {"ok": True, "roots": []})
    monkeypatch.setattr(api, "_gpu_status_payload", lambda: {"available": True})
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [])
    monkeypatch.setattr(api, "qwen_status", lambda: {"loaded": False})
    monkeypatch.setattr(api, "_resolve_sam1_devices", lambda: [])
    monkeypatch.setattr(api, "mlx_sam_status", lambda: {})
    monkeypatch.setattr(api, "_list_sam3_runs_impl", lambda **_kwargs: [])
    monkeypatch.setattr(api, "_list_qwen_model_entries", lambda: [])
    monkeypatch.setattr(api, "_list_clip_classifiers_impl", lambda **_kwargs: [])
    try:
        api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.clear()
        api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.update(
            {"status": "quarantining", "ready_at": None, "error": None}
        )
        blocked = api._system_health_summary()
        assert blocked["ok"] is False
        assert blocked["class_analysis"]["ready"] is False
        assert "class_analysis_startup_cleanup_in_progress" in blocked["errors"]

        api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.update(
            {"status": "deleting", "ready_at": time.time(), "error": None}
        )
        ready = api._system_health_summary()
        assert ready["ok"] is True
        assert ready["class_analysis"]["ready"] is True
        assert ready["class_analysis"]["startup_cleanup"]["status"] == "deleting"
    finally:
        api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.clear()
        api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.update(original_cleanup_state)


def test_class_analysis_job_summary_does_not_require_retained_result():
    job = api.ClassAnalysisJob(
        job_id="summary-only",
        status="completed",
        summary={"object_count": 193_000, "neighbor_search": {"method": "pynndescent"}},
        result=None,
    )

    serialized = api._serialize_class_analysis_job(job)

    assert serialized["summary"]["object_count"] == 193_000
    assert serialized["summary"]["neighbor_search"]["method"] == "pynndescent"
    assert job.result is None


def test_class_analysis_duplicate_fingerprint_reuses_completed_job(monkeypatch):
    fingerprint = "f" * 64
    existing = api.ClassAnalysisJob(
        job_id="ca_existing",
        request={"run_fingerprint": fingerprint, "snapshot_id": "cas_existing"},
        status="completed",
    )
    monkeypatch.setattr(api, "_class_analysis_run_fingerprint", lambda _payload: fingerprint)
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS["ca_existing"] = existing
    with api.CLASS_ANALYSIS_RUN_FINGERPRINTS_LOCK:
        api.CLASS_ANALYSIS_RUN_FINGERPRINTS[fingerprint] = "ca_existing"
    try:
        response = api._enqueue_class_analysis_job({"snapshot_id": "cas_existing"})
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.pop("ca_existing", None)
        with api.CLASS_ANALYSIS_RUN_FINGERPRINTS_LOCK:
            api.CLASS_ANALYSIS_RUN_FINGERPRINTS.pop(fingerprint, None)

    assert response == {
        "job_id": "ca_existing",
        "run_fingerprint": fingerprint,
        "snapshot_id": "cas_existing",
        "reused": True,
    }


def test_class_analysis_explicit_rerun_gets_fresh_job_with_same_digest(
    monkeypatch,
):
    fingerprint = "e" * 64
    existing = api.ClassAnalysisJob(
        job_id="ca_existing",
        request={"run_fingerprint": fingerprint},
        status="completed",
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_run_fingerprint",
        lambda _payload: fingerprint,
    )
    class NoopThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

    monkeypatch.setattr(api.threading, "Thread", NoopThread)
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS["ca_existing"] = existing
    with api.CLASS_ANALYSIS_RUN_FINGERPRINTS_LOCK:
        api.CLASS_ANALYSIS_RUN_FINGERPRINTS[fingerprint] = "ca_existing"
    try:
        response = api._enqueue_class_analysis_job(
            {"force_new_run": True}
        )
        fresh = api.CLASS_ANALYSIS_JOBS[response["job_id"]]
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.pop("ca_existing", None)
            if "response" in locals():
                api.CLASS_ANALYSIS_JOBS.pop(response["job_id"], None)
        with api.CLASS_ANALYSIS_RUN_FINGERPRINTS_LOCK:
            api.CLASS_ANALYSIS_RUN_FINGERPRINTS.pop(
                fingerprint,
                None,
            )

    assert response["job_id"] != "ca_existing"
    assert response["run_fingerprint"] == fingerprint
    assert response["reused"] is False
    assert fresh.request["run_fingerprint"] == fingerprint


def test_class_analysis_bounded_ordered_map_limits_submitted_crop_rows():
    state = {"outstanding": 0, "maximum": 0}

    class _Future:
        def __init__(self, value):
            self.value = value

        def result(self):
            state["outstanding"] -= 1
            return self.value

    class _Executor:
        def submit(self, fn, item):
            state["outstanding"] += 1
            state["maximum"] = max(state["maximum"], state["outstanding"])
            return _Future(fn(item))

    values = list(
        api._bounded_ordered_executor_map(
            _Executor(),
            lambda value: value * 2,
            range(100),
            max_in_flight=6,
        )
    )

    assert values == [value * 2 for value in range(100)]
    assert state["maximum"] == 6
    assert state["outstanding"] == 0


def test_class_analysis_bounded_ordered_map_executes_rows_in_parallel():
    barrier = api.threading.Barrier(4)

    def prepare(value):
        if value < 4:
            barrier.wait(timeout=2)
        return value

    with api.ThreadPoolExecutor(max_workers=4) as executor:
        values = list(
            api._bounded_ordered_executor_map(
                executor,
                prepare,
                range(8),
                max_in_flight=8,
            )
        )

    assert values == list(range(8))


def test_qwen_stream_worker_error_is_reported_without_long_timeout():
    class _EmptyStreamer:
        def __iter__(self):
            return self

        def __next__(self):
            raise api.queue.Empty

    class _FinishedThread:
        def join(self, timeout=None):
            assert timeout == 1.0

        def is_alive(self):
            return False

    finished = api.threading.Event()
    finished.set()
    with pytest.raises(RuntimeError, match="qwen_stream_generation_failed:model exploded"):
        list(
            api._iterate_qwen_streamer_with_worker(
                _EmptyStreamer(),
                _FinishedThread(),
                [ValueError("model exploded")],
                finished,
                timeout_seconds=2,
            )
        )


def test_list_class_analysis_qwen_reviews_returns_latest_attempt_per_point():
    now = api.time.time()
    parent = api.ClassAnalysisJob(job_id="ca_restore_reviews", status="completed")
    old = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_old",
        parent_job_id=parent.job_id,
        point_id="point-a",
        status="failed",
        created_at=now - 4,
        updated_at=now - 3,
    )
    latest = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_latest",
        parent_job_id=parent.job_id,
        point_id="point-a",
        status="running",
        created_at=now - 1,
        updated_at=now,
    )
    other = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_other",
        parent_job_id=parent.job_id,
        point_id="point-b",
        status="completed",
        created_at=now - 2,
        updated_at=now - 1,
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS[parent.job_id] = parent
    with api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS_LOCK:
        api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS.update(
            {job.review_id: job for job in (old, latest, other)}
        )
    try:
        payload = api.list_class_analysis_qwen_reviews(parent.job_id)
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.pop(parent.job_id, None)
        with api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS_LOCK:
            for job in (old, latest, other):
                api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS.pop(job.review_id, None)

    assert payload["job_id"] == parent.job_id
    assert [review["review_id"] for review in payload["reviews"]] == [
        "cqr_latest",
        "cqr_other",
    ]


def test_qwen_cache_inventory_protects_caption_cached_model(monkeypatch, tmp_path):
    repo = types.SimpleNamespace(
        repo_id="owner/caption-model",
        repo_type="model",
        size_on_disk=1234,
        last_accessed=1.0,
        last_modified=2.0,
        revisions=[],
    )
    cache_info = types.SimpleNamespace(
        repos=[repo],
        cache_dir=tmp_path,
        size_on_disk=1234,
        warnings=[],
    )
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", lambda: cache_info)
    monkeypatch.setattr(
        api,
        "list_qwen_models",
        lambda: {
            "models": [
                {
                    "id": "owner/caption-model",
                    "metadata": {"model_id": "owner/caption-model"},
                    "review_compatibility": {"status": "ready_local", "reviewable": True},
                }
            ]
        },
    )
    monkeypatch.setattr(
        api,
        "qwen_caption_cache",
        {
            "caption:mlx_vlm:owner/caption-model": api.QwenRuntime(
                model=object(),
                processor=object(),
                platform=api.QWEN_PLATFORM_MLX,
                model_id="owner/caption-model",
            )
        },
    )
    monkeypatch.setattr(api, "qwen_loaded_effective_model_id", None)

    inventory = api._qwen_hf_cache_inventory()

    assert inventory["repos"][0]["loaded"] is True


def test_class_analysis_router_exposes_qwen_review_reconnect_endpoint():
    paths = {getattr(route, "path", "") for route in api.app.routes}
    assert "/class_analysis/jobs/{job_id}/qwen_reviews" in paths
    assert (
        "/class_analysis/jobs/{job_id}/points/{point_id}/review_disposition"
        in paths
    )


def test_qwen_review_compatibility_blocks_cuda_awq_on_mac(monkeypatch):
    monkeypatch.setattr(api.sys, "platform", "darwin")
    compatibility = api._qwen_review_compatibility(
        {"id": "owner/model-awq"},
        {
            "model_id": "owner/model-awq",
            "inference_supported": True,
            "vision_inference_supported": True,
            "quantization_backend": "awq",
        },
        {"local": True, "partial": False},
        api.QWEN_PLATFORM_TRANSFORMERS,
    )

    assert compatibility["status"] == "incompatible"
    assert compatibility["reviewable"] is False
    assert "MLX" in compatibility["reason"]


def test_qwen_review_preflight_falls_back_only_for_implicit_default(monkeypatch):
    entries = {
        "owner/partial": {"metadata": {"model_id": "owner/partial"}},
        "owner/ready": {"metadata": {"model_id": "owner/ready"}},
    }
    monkeypatch.setattr(
        api,
        "CLASS_ANALYSIS_QWEN_REVIEW_LOCAL_FALLBACK_MODEL_IDS",
        ("owner/ready",),
    )
    monkeypatch.setattr(
        api,
        "_get_builtin_qwen_model_entry",
        lambda model_id: entries.get(model_id),
    )
    monkeypatch.setattr(
        api,
        "_resolve_qwen_runtime_platform",
        lambda *_args, **_kwargs: api.QWEN_PLATFORM_MLX,
    )
    monkeypatch.setattr(
        api,
        "_qwen_model_local_state",
        lambda model_id, _platform: {
            "local": model_id == "owner/ready",
            "partial": model_id == "owner/partial",
        },
    )
    monkeypatch.setattr(
        api,
        "_qwen_review_compatibility",
        lambda _entry, metadata, _availability, _platform: {
            "reviewable": metadata["model_id"] == "owner/ready",
            "reason": "incomplete snapshot",
        },
    )

    selected = api._class_analysis_qwen_review_preflight_model_selection(
        "owner/partial",
        allow_local_fallback=True,
    )

    assert selected == {
        "model_id": "owner/ready",
        "fallback_applied": True,
        "fallback_reason": "incomplete snapshot",
    }
    with pytest.raises(api.HTTPException) as exc_info:
        api._class_analysis_qwen_review_preflight_model_selection(
            "owner/partial",
            allow_local_fallback=False,
        )
    assert exc_info.value.status_code == 400
    assert "qwen_review_model_incompatible:owner/partial" in str(
        exc_info.value.detail
    )
