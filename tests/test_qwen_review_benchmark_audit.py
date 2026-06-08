import inspect
import json

import tools.analyze_class_split_qwen_review_benchmark as audit_mod
from tools.analyze_class_split_qwen_review_benchmark import audit_records, compare_runs
from tools.run_class_split_qwen_review_benchmark import (
    _make_visual_sheet,
    _normalize_filter_values,
    _source_point_ids,
    _summarize,
)


def _record(**overrides):
    base = {
        "ordinal": 1,
        "point_id": "p1",
        "decision": "skip_uncertain",
        "current_class": "Truck",
        "suggested_neighbor_class": "LightVehicle",
        "target_class": "Truck",
        "confidence": 0.4,
        "backend_tier": "limited",
        "visual_quality": "limited",
        "object_visibility": "partial",
        "rationale_short": "target is too small",
        "visible_target_cues": ["distinct target outline", "visible surface detail"],
        "supporting_clean_evidence_ids": ["target_context_1"],
        "anchor_evidence_current": "weak",
        "anchor_evidence_suggested": "strong",
        "evidence_ledger": {
            "rows": [
                {"evidence_id": "target_context_1", "kind": "target_context", "use": "clean_visual"},
                {"evidence_id": "source_clean_2", "kind": "source_clean", "use": "clean_visual"},
                {"evidence_id": "zoom_region_6", "kind": "zoom_region", "use": "clean_visual"},
                {"evidence_id": "source_overlay_3", "kind": "source_overlay", "use": "geometry_overlay"},
                {"evidence_id": "class_context_pack_5", "kind": "class_context_pack", "use": "clean_visual"},
            ],
            "clean_visual_evidence_ids": ["target_context_1", "source_clean_2", "zoom_region_6"],
            "geometry_overlay_evidence_ids": ["source_overlay_3", "overlap_decomposition_4"],
            "local_consensus_evidence_ids": [],
            "clean_visual_reference_evidence_ids": ["class_context_pack_5"],
        },
        "guardrail_reasons": [],
        "advisory_reasons": [],
        "model_compact_arguments": {
            "current_evidence": "weak",
            "suggested_evidence": "weak",
            "target_evidence": "weak",
            "overlap_assessment": "unclear",
        },
    }
    base.update(overrides)
    return base


def test_qwen_review_benchmark_audit_flags_unsafe_low_quality_accept():
    record = _record(
        decision="accept_suggested",
        target_class="LightVehicle",
        backend_tier="limited",
        visual_quality="clear",
        object_visibility="clear",
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
        },
    )

    audit = audit_records([record])

    assert audit["unsafe_issue_count"] == 2
    assert audit["issue_counts"]["non_skip_low_quality"] == 1
    assert audit["issue_counts"]["class_change_low_backend_quality"] == 1


def test_qwen_review_benchmark_visual_sheet_can_select_guarded_recommendations(tmp_path):
    guarded = _record(
        guarded_recommendation={
            "blocked": True,
            "decision": "accept_suggested",
            "target_class": "LightVehicle",
            "confidence": 0.9,
            "guardrail_reasons": ["limited target"],
        },
    )
    unguarded = _record(ordinal=2, decision="accept_suggested", target_class="LightVehicle", backend_tier="clear")

    guarded_output = tmp_path / "guarded.jpg"
    _make_visual_sheet(
        [guarded, unguarded],
        parent_job_id="ca_test",
        output_path=guarded_output,
        title="guarded",
        guarded_only=True,
        limit=4,
    )

    assert guarded_output.is_file()
    assert guarded_output.stat().st_size > 0

    empty_output = tmp_path / "empty.jpg"
    _make_visual_sheet(
        [unguarded],
        parent_job_id="ca_test",
        output_path=empty_output,
        title="empty",
        guarded_only=True,
        limit=4,
    )

    assert not empty_output.exists()


def test_qwen_review_benchmark_audit_flags_class_change_without_visible_cues():
    record = _record(
        decision="accept_suggested",
        current_class="CurrentClass",
        suggested_neighbor_class="SuggestedClass",
        target_class="SuggestedClass",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        visible_target_cues=["SuggestedClass", "matches suggested class"],
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
        },
    )

    audit = audit_records([record])

    assert audit["issue_counts"]["class_change_missing_visible_target_cues"] == 1
    assert audit["issues"]["class_change_missing_visible_target_cues"][0]["point_id"] == "p1"


def test_qwen_review_benchmark_audit_flags_background_dominated_class_change():
    record = _record(
        decision="accept_suggested",
        current_class="CurrentClass",
        suggested_neighbor_class="SuggestedClass",
        target_class="SuggestedClass",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        current_evidence="weak",
        suggested_evidence="strong",
        target_evidence="strong",
        visible_target_cues=["rectangular target body", "visible target surface texture"],
        supporting_clean_evidence_ids=["target_context_1"],
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "background_dominated",
        },
    )

    audit = audit_records([record])

    assert audit["issue_counts"]["class_change_bad_target_background_contrast"] == 1
    assert audit["target_background_contrast_counts"]["background_dominated"] == 1


def test_qwen_review_benchmark_audit_flags_missing_specificity_for_class_change():
    record = _record(
        decision="accept_suggested",
        current_class="CurrentClass",
        suggested_neighbor_class="SuggestedClass",
        target_class="SuggestedClass",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        current_evidence="weak",
        suggested_evidence="strong",
        target_evidence="strong",
        visible_target_cues=["rectangular target body", "visible target surface texture"],
        supporting_clean_evidence_ids=["target_context_1"],
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
        },
    )

    audit = audit_records([record])

    assert audit["issue_counts"]["class_change_bad_specificity_alignment"] == 1
    assert audit["issue_counts"]["class_change_bad_target_background_contrast"] == 1
    assert audit["specificity_alignment_counts"]["missing"] == 1
    assert audit["target_background_contrast_counts"]["missing"] == 1


def test_qwen_review_benchmark_audit_allows_one_cue_with_independent_support():
    record = _record(
        decision="accept_suggested",
        current_class="CurrentClass",
        suggested_neighbor_class="SuggestedClass",
        target_class="SuggestedClass",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        current_evidence="weak",
        suggested_evidence="strong",
        target_evidence="strong",
        anchor_evidence_suggested="strong",
        local_context_evidence="strong",
        local_consensus_evidence="supports_suggested",
        global_context_evidence="strong",
        same_image_scale_evidence="neutral",
        same_image_embedding_evidence="questions_current",
        overlap_assessment="near_context",
        overlap_explains_candidate_similarity=False,
        visible_target_cues=["compact target body"],
        supporting_clean_evidence_ids=["target_context_1", "zoom_region_6"],
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "supports_suggested",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "neutral",
            "same_image_embedding_evidence": "questions_current",
            "overlap_assessment": "near_context",
            "overlap_explains_candidate_similarity": False,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
        },
    )

    audit = audit_records([record])

    assert "class_change_missing_visible_target_cues" not in audit["issue_counts"]


def test_qwen_review_benchmark_audit_semicolon_positive_cue_does_not_reject_target():
    record = _record(
        decision="accept_suggested",
        current_class="Truck",
        suggested_neighbor_class="LightVehicle",
        target_class="LightVehicle",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        current_evidence="weak",
        suggested_evidence="strong",
        target_evidence="strong",
        anchor_evidence_suggested="strong",
        local_context_evidence="strong",
        local_consensus_evidence="supports_suggested",
        global_context_evidence="strong",
        same_image_scale_evidence="neutral",
        same_image_embedding_evidence="questions_current",
        overlap_assessment="none",
        overlap_explains_candidate_similarity=False,
        visible_target_cues=["compact target body"],
        supporting_clean_evidence_ids=["target_context_1", "zoom_region_6"],
        rationale_short=(
            "Target is small, compact, no cargo; matches LightVehicle visual cues; "
            "no overlap contamination"
        ),
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_suggested": "strong",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "supports_suggested",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "neutral",
            "same_image_embedding_evidence": "questions_current",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
        },
    )

    audit = audit_records([record])

    assert "class_change_text_rejects_target" not in audit["issue_counts"]


def test_qwen_review_benchmark_audit_blocks_one_cue_without_independent_support():
    record = _record(
        decision="accept_suggested",
        current_class="CurrentClass",
        suggested_neighbor_class="SuggestedClass",
        target_class="SuggestedClass",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        current_evidence="weak",
        suggested_evidence="strong",
        target_evidence="strong",
        anchor_evidence_suggested="moderate",
        local_context_evidence="strong",
        local_consensus_evidence="mixed",
        global_context_evidence="strong",
        same_image_scale_evidence="neutral",
        same_image_embedding_evidence="neutral",
        overlap_assessment="near_context",
        visible_target_cues=["compact target body"],
        supporting_clean_evidence_ids=["target_context_1"],
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "neutral",
            "same_image_embedding_evidence": "neutral",
            "overlap_assessment": "near_context",
        },
    )

    audit = audit_records([record])

    assert audit["issue_counts"]["class_change_missing_visible_target_cues"] == 1


def test_qwen_review_benchmark_audit_flags_accept_without_strong_suggested_anchor():
    record = _record(
        decision="accept_suggested",
        current_class="CurrentClass",
        suggested_neighbor_class="SuggestedClass",
        target_class="SuggestedClass",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        anchor_evidence_suggested="moderate",
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_suggested": "moderate",
            "overlap_assessment": "none",
        },
    )

    audit = audit_records([record])

    assert audit["issue_counts"]["accept_without_strong_suggested_anchor"] == 1
    assert "moderate" in audit["issues"]["accept_without_strong_suggested_anchor"][0]["reason"]


def test_qwen_review_benchmark_audit_allows_moderate_anchor_on_clear_target_path():
    record = _record(
        decision="accept_suggested",
        current_class="CurrentClass",
        suggested_neighbor_class="SuggestedClass",
        target_class="SuggestedClass",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        anchor_evidence_suggested="moderate",
        target_evidence="strong",
        suggested_evidence="strong",
        current_evidence="weak",
        local_context_evidence="strong",
        global_context_evidence="strong",
        overlap_assessment="none",
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "global_context_evidence": "strong",
            "overlap_assessment": "none",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
        },
    )

    audit = audit_records([record])

    assert "accept_without_strong_suggested_anchor" not in audit["issue_counts"]


def test_qwen_review_benchmark_audit_ignores_context_only_visible_cues():
    record = _record(
        decision="accept_suggested",
        current_class="CurrentClass",
        suggested_neighbor_class="SuggestedClass",
        target_class="SuggestedClass",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        visible_target_cues=["top-down perspective", "parked on pavement", "ribbed target surface"],
        supporting_clean_evidence_ids=["target_context_1"],
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
        },
    )

    audit = audit_records([record])

    assert audit["issue_counts"]["class_change_missing_visible_target_cues"] == 1


def test_qwen_review_benchmark_audit_ignores_negative_and_color_only_visible_cues():
    record = _record(
        decision="accept_suggested",
        current_class="CurrentClass",
        suggested_neighbor_class="SuggestedClass",
        target_class="SuggestedClass",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        visible_target_cues=[
            "aerial view of parked candidate class",
            "multiple object colors",
            "no current-class features",
        ],
        supporting_clean_evidence_ids=["target_context_1"],
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
        },
    )

    audit = audit_records([record])

    assert audit["issue_counts"]["class_change_missing_visible_target_cues"] == 1


def test_qwen_review_benchmark_visible_cue_filter_is_domain_agnostic():
    cues = audit_mod._normalize_visible_cues(
        [
            "spiral conduit ridges",
            "triangular bracket lattice",
            "translucent membrane fold",
            "aerial view of parked candidate",
            "green background region",
            "matches suggested class",
        ],
        current_class="CurrentClass",
        suggested_class="SuggestedClass",
        target_class="SuggestedClass",
    )

    assert cues == [
        "spiral conduit ridges",
        "triangular bracket lattice",
        "translucent membrane fold",
    ]

    source = inspect.getsource(audit_mod._normalize_visible_cues)
    forbidden_terms = [
        "concrete_visual_tokens",
        '"wheel"',
        '"roof"',
        '"pole"',
        '"hull"',
        '"cab"',
        '"boom"',
        '"bucket"',
    ]
    for term in forbidden_terms:
        assert term not in source


def test_qwen_review_benchmark_audit_flags_class_change_without_clean_visual_ledger():
    record = _record(
        decision="accept_suggested",
        target_class="LightVehicle",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        evidence_ledger={},
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
        },
    )

    audit = audit_records([record])

    assert audit["issue_counts"]["class_change_missing_clean_visual_ledger"] == 1


def test_qwen_review_benchmark_audit_flags_class_change_without_supporting_clean_evidence():
    record = _record(
        decision="accept_suggested",
        target_class="LightVehicle",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        supporting_clean_evidence_ids=[],
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
        },
    )

    audit = audit_records([record])

    assert audit["issue_counts"]["class_change_missing_supporting_clean_evidence"] == 1


def test_qwen_review_benchmark_audit_flags_overlay_only_supporting_evidence():
    record = _record(
        decision="accept_suggested",
        target_class="LightVehicle",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        supporting_clean_evidence_ids=["source_overlay_3"],
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
        },
    )

    audit = audit_records([record])

    assert audit["issue_counts"]["class_change_missing_supporting_clean_evidence"] == 1
    assert "clean visual evidence" in audit["issues"]["class_change_missing_supporting_clean_evidence"][0]["reason"]


def test_qwen_review_benchmark_audit_flags_reference_only_supporting_evidence():
    record = _record(
        decision="accept_suggested",
        target_class="LightVehicle",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        supporting_clean_evidence_ids=["class_context_pack_5"],
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
        },
    )

    audit = audit_records([record])

    assert audit["issue_counts"]["class_change_missing_supporting_clean_evidence"] == 1
    assert "reference context" in audit["issues"]["class_change_missing_supporting_clean_evidence"][0]["reason"]


def test_qwen_review_benchmark_audit_flags_class_change_text_rejecting_target_alias():
    record = _record(
        decision="accept_suggested",
        current_class="Building",
        suggested_neighbor_class="LightVehicle",
        target_class="LightVehicle",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        visible_target_cues=["roofline structure", "overhead shadows"],
        supporting_clean_evidence_ids=["target_context_1"],
        rationale_short="Target is a roof, not vehicle; suggested context is nearby.",
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "near_context",
        },
    )

    audit = audit_records([record])

    assert audit["issue_counts"]["class_change_text_rejects_target"] == 1
    assert "vehicle" in audit["issues"]["class_change_text_rejects_target"][0]["reason"]


def test_qwen_review_benchmark_audit_keeps_text_fields_sentence_bounded():
    record = _record(
        decision="accept_suggested",
        current_class="Truck",
        suggested_neighbor_class="Building",
        target_class="Building",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        visible_target_cues=["Flat roof structure", "Multiple building units"],
        supporting_clean_evidence_ids=["target_context_1"],
        rationale_short="Target shows clear building features; no truck-like cargo or chassis; no overlap contamination.",
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_suggested": "strong",
            "overlap_assessment": "none",
            "visible_target_cues": ["Flat roof structure", "Multiple building units"],
            "rationale_short": "Target shows clear building features; no truck-like cargo or chassis; no overlap contamination.",
        },
    )

    audit = audit_records([record])

    assert audit["issue_counts"].get("class_change_text_rejects_target", 0) == 0


def test_qwen_review_benchmark_audit_flags_dominant_current_overlap():
    record = _record(
        decision="accept_suggested",
        current_class="Building",
        suggested_neighbor_class="LightVehicle",
        target_class="LightVehicle",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        visible_target_cues=["parked object shape", "bright vehicle roof"],
        supporting_clean_evidence_ids=["target_context_1"],
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_suggested": "strong",
            "overlap_assessment": "near_context",
        },
        evidence=[
            {
                "kind": "overlap_decomposition",
                "metadata": {
                    "overlaps": [
                        {
                            "class_name": "Building",
                            "relation": "partial_contamination",
                            "target_area_covered": 0.63,
                            "other_area_covered": 0.20,
                            "iou": 0.18,
                        },
                        {
                            "class_name": "LightVehicle",
                            "relation": "partial_contamination",
                            "target_area_covered": 0.15,
                            "other_area_covered": 0.31,
                            "iou": 0.11,
                        }
                    ]
                },
            }
        ],
    )

    audit = audit_records([record])

    assert audit["issue_counts"]["class_change_dominant_current_overlap"] == 1


def test_qwen_review_benchmark_audit_requires_partial_overlap_rebuttal():
    unsafe = _record(
        ordinal=1,
        point_id="unsafe",
        decision="accept_suggested",
        target_class="LightVehicle",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        rationale_short="target looks like a light vehicle",
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
        },
    )
    safe = _record(
        ordinal=2,
        point_id="safe",
        decision="accept_suggested",
        target_class="LightVehicle",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        rationale_short="overlap does not explain the target vehicle features",
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
        },
    )

    audit = audit_records([unsafe, safe])

    assert audit["issue_counts"]["partial_overlap_without_explicit_rebuttal"] == 1
    assert audit["issues"]["partial_overlap_without_explicit_rebuttal"][0]["point_id"] == "unsafe"


def test_qwen_review_benchmark_audit_accepts_minor_or_adjacent_overlap_rebuttals():
    minor = _record(
        ordinal=1,
        point_id="minor",
        decision="accept_suggested",
        target_class="Building",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        rationale_short="Target is a residential building roof. Overlap is minor.",
        current_evidence="weak",
        suggested_evidence="strong",
        target_evidence="strong",
        overlap_assessment="partial_contamination",
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
        },
    )
    adjacent = _record(
        ordinal=2,
        point_id="adjacent",
        decision="accept_suggested",
        target_class="Building",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        rationale_short="Target is a clear building roof. Overlapping containers are adjacent, not the target itself.",
        current_evidence="weak",
        suggested_evidence="strong",
        target_evidence="strong",
        overlap_assessment="partial_contamination",
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
        },
    )

    audit = audit_records([minor, adjacent])

    assert audit["unsafe_issue_count"] == 0


def test_qwen_review_benchmark_audit_accepts_background_not_vehicle_rebuttal():
    record = _record(
        decision="accept_suggested",
        target_class="LightVehicle",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        rationale_short="Target crop clearly shows a sedan. Overlap is background road markings, not a vehicle.",
        current_evidence="weak",
        suggested_evidence="strong",
        target_evidence="strong",
        overlap_assessment="partial_contamination",
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
        },
    )

    audit = audit_records([record])

    assert audit["unsafe_issue_count"] == 0


def test_qwen_review_benchmark_audit_accepts_verifier_backed_partial_overlap_rebuttal():
    record = _record(
        decision="accept_suggested",
        target_class="Building",
        current_class="Truck",
        suggested_neighbor_class="Building",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        current_evidence="weak",
        suggested_evidence="strong",
        target_evidence="strong",
        overlap_assessment="partial_contamination",
        overlap_explains_candidate_similarity=False,
        overlap_adjudication_verified=True,
        anchor_evidence_suggested="moderate",
        local_context_evidence="strong",
        global_context_evidence="strong",
        same_image_scale_evidence="questions_current",
        same_image_embedding_evidence="questions_current",
        specificity_alignment="supports_suggested",
        target_background_contrast="target_specific",
        visible_target_cues=["fixed rectangular roof", "corrugated roof texture"],
        supporting_clean_evidence_ids=["target_context_1", "source_clean_2"],
        rationale_short=(
            "Target pixels show a fixed rectangular roof and corrugated texture; "
            "overlap does not explain the target-contained building features."
        ),
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "questions_current",
            "same_image_embedding_evidence": "questions_current",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "overlap_adjudication_verified": True,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
        },
    )

    audit = audit_records([record])

    assert audit["unsafe_issue_count"] == 0
    assert audit["overlap_adjudication_verified_count"] == 1


def test_qwen_review_benchmark_audit_accepts_verified_overlap_without_rebuttal_phrase():
    record = _record(
        decision="accept_suggested",
        target_class="SuggestedClass",
        current_class="CurrentClass",
        suggested_neighbor_class="SuggestedClass",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        current_evidence="weak",
        suggested_evidence="strong",
        target_evidence="strong",
        overlap_assessment="partial_contamination",
        overlap_explains_candidate_similarity=False,
        overlap_adjudication_verified=True,
        anchor_evidence_suggested="moderate",
        local_context_evidence="strong",
        local_consensus_evidence="mixed",
        global_context_evidence="strong",
        same_image_scale_evidence="neutral",
        same_image_embedding_evidence="questions_current",
        specificity_alignment="supports_suggested",
        target_background_contrast="target_specific",
        visible_target_cues=["spiral conduit ridges", "triangular bracket lattice"],
        supporting_clean_evidence_ids=["target_context_1", "source_clean_2"],
        rationale_short="Verifier isolated target-specific visible features in the clean crop.",
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "neutral",
            "same_image_embedding_evidence": "questions_current",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "overlap_adjudication_verified": True,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
        },
    )

    audit = audit_records([record])

    assert audit["unsafe_issue_count"] == 0
    assert "class_change_bad_overlap" not in audit["issue_counts"]
    assert "partial_overlap_without_explicit_rebuttal" not in audit["issue_counts"]


def test_qwen_review_benchmark_audit_accepts_verified_moderate_anchor_overlap_without_rebuttal_phrase():
    record = _record(
        decision="accept_suggested",
        target_class="SuggestedClass",
        current_class="CurrentClass",
        suggested_neighbor_class="SuggestedClass",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        current_evidence="weak",
        suggested_evidence="strong",
        target_evidence="strong",
        overlap_assessment="partial_contamination",
        overlap_explains_candidate_similarity=False,
        overlap_adjudication_verified=True,
        anchor_adjudication_verified=True,
        current_class_plausible=False,
        anchor_evidence_suggested="moderate",
        local_context_evidence="strong",
        local_consensus_evidence="mixed",
        global_context_evidence="strong",
        same_image_scale_evidence="insufficient",
        same_image_embedding_evidence="insufficient",
        specificity_alignment="supports_suggested",
        target_background_contrast="target_specific",
        visible_target_cues=["spiral conduit ridges", "triangular bracket lattice"],
        supporting_clean_evidence_ids=["target_context_1", "source_clean_2"],
        rationale_short="Verifier isolated target-specific visible features in the clean crop.",
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "local_consensus_evidence": "mixed",
            "global_context_evidence": "strong",
            "same_image_scale_evidence": "insufficient",
            "same_image_embedding_evidence": "insufficient",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "overlap_adjudication_verified": True,
            "anchor_adjudication_verified": True,
            "current_class_plausible": False,
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
        },
    )

    audit = audit_records([record])

    assert audit["unsafe_issue_count"] == 0
    assert "class_change_bad_overlap" not in audit["issue_counts"]
    assert "partial_overlap_without_explicit_rebuttal" not in audit["issue_counts"]


def test_qwen_review_benchmark_audit_accepts_dual_bbox_overlap_switch_path():
    record = _record(
        decision="accept_suggested",
        target_class="Building",
        current_class="Truck",
        suggested_neighbor_class="Building",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        current_evidence="weak",
        suggested_evidence="strong",
        target_evidence="strong",
        overlap_assessment="duplicate_like",
        dual_bbox_resolution="overlap_box_class",
        dual_bbox_conflict={
            "enabled": True,
            "kind": "near_identical_cross_class_bbox",
            "current_class": "Truck",
            "other_class_name": "Building",
            "iou": 0.96,
            "relation": "duplicate_like",
        },
        review_disposition={"disposition": "dual_bbox_switch_overlap_class", "signal": "actionable"},
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_suggested": "moderate",
            "local_context_evidence": "strong",
            "global_context_evidence": "strong",
            "overlap_assessment": "duplicate_like",
            "dual_bbox_resolution": "overlap_box_class",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
        },
        specificity_alignment="supports_suggested",
        target_background_contrast="target_specific",
        visible_target_cues=["flat roof surface", "fixed building structure"],
    )

    audit = audit_records([record])

    assert audit["unsafe_issue_count"] == 0


def test_qwen_review_benchmark_audit_lists_guarded_clear_target_candidates():
    record = _record(
        decision="skip_uncertain",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "none",
        },
    )

    audit = audit_records([record])

    assert audit["unsafe_issue_count"] == 0
    assert audit["guarded_clear_target_candidate_count"] == 1
    assert audit["guarded_clear_target_candidates"][0]["point_id"] == "p1"


def test_qwen_review_benchmark_audit_lists_guarded_recommendations():
    record = _record(
        decision="skip_uncertain",
        target_class="Truck",
        backend_tier="limited",
        visual_quality="clear",
        object_visibility="clear",
        guarded_recommendation={
            "blocked": True,
            "decision": "accept_suggested",
            "target_class": "LightVehicle",
            "confidence": 0.82,
            "target_evidence": "strong",
            "current_evidence": "weak",
            "guardrail_reasons": ["accept_suggested requires clear backend visual-quality tier, got limited"],
            "rationale_short": "target looks like a pickup",
        },
        specificity_probe={
            "status": "completed",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "specificity_margin": "suggested_target_favored",
            "target_identity_uncertainty": "low",
        },
    )

    audit = audit_records([record])

    assert audit["unsafe_issue_count"] == 0
    assert audit["guarded_recommendation_count"] == 1
    assert audit["effective_human_signal_count"] == 1
    assert audit["guarded_human_triage_count"] == 1
    assert audit["review_disposition_signal_counts"] == {"guarded_human_triage": 1}
    guarded = audit["guarded_recommendations"][0]
    assert guarded["point_id"] == "p1"
    assert guarded["guarded_recommendation"]["target_class"] == "LightVehicle"
    assert guarded["review_disposition"]["disposition"] == "guarded_visual_quality"
    assert guarded["review_disposition"]["signal_strength"] == "strong"
    assert guarded["review_disposition"]["priority"] == "high"


def test_qwen_review_benchmark_audit_counts_current_overlap_false_alarm_as_useful_negative():
    record = _record(
        decision="skip_uncertain",
        current_class="Building",
        target_class="Building",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        guarded_recommendation={
            "blocked": True,
            "decision": "accept_suggested",
            "target_class": "LightVehicle",
            "confidence": 0.95,
            "guardrail_reasons": [
                "accept_suggested conflicts with overlap decomposition: current class Building dominates the target bbox (partial_contamination, current_cover=0.63, target_class_cover=0.15)"
            ],
            "rationale_short": "target looks like a vehicle",
        },
    )

    audit = audit_records([record])

    assert audit["unsafe_issue_count"] == 0
    assert audit["effective_human_signal_count"] == 1
    assert audit["review_disposition_signal_counts"] == {"useful_negative": 1}
    guarded = audit["guarded_recommendations"][0]
    assert guarded["review_disposition"]["disposition"] == "verified_current_class_overlap"
    assert guarded["review_disposition"]["advisory_decision"] == "confirm_current"
    assert guarded["review_disposition"]["advisory_target_class"] == "Building"


def test_qwen_review_benchmark_audit_prefers_controller_normalized_evidence():
    record = _record(
        decision="confirm_current",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        current_evidence="strong",
        suggested_evidence="weak",
        target_evidence="strong",
        overlap_assessment="partial_contamination",
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
        },
    )

    audit = audit_records([record])

    assert audit["unsafe_issue_count"] == 0


def test_qwen_review_benchmark_audit_flags_class_change_when_current_still_plausible():
    record = _record(
        decision="accept_suggested",
        current_class="Truck",
        suggested_neighbor_class="Building",
        target_class="Building",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        current_evidence="weak",
        suggested_evidence="strong",
        target_evidence="strong",
        anchor_evidence_suggested="strong",
        overlap_assessment="none",
        specificity_alignment="supports_suggested",
        target_background_contrast="target_specific",
        current_class_plausible=True,
        current_class_plausibility_reason="clean target still looks like an isolated trailer body",
        visible_target_cues=["rectangular white body", "flat roof surface"],
        supporting_clean_evidence_ids=["target_context_1"],
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_suggested": "strong",
            "overlap_assessment": "none",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
            "current_class_plausible": True,
            "current_class_plausibility_reason": "clean target still looks like an isolated trailer body",
        },
    )

    audit = audit_records([record])

    assert audit["current_class_plausible_count"] == 1
    assert audit["issue_counts"]["class_change_current_class_still_plausible"] == 1
    assert "isolated trailer" in audit["issues"]["class_change_current_class_still_plausible"][0]["reason"]


def test_qwen_review_benchmark_audit_counts_nested_cue_verifier_current_plausibility():
    record = _record(
        decision="accept_suggested",
        current_class="Truck",
        suggested_neighbor_class="Building",
        target_class="Building",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        current_evidence="weak",
        suggested_evidence="strong",
        target_evidence="strong",
        anchor_evidence_suggested="strong",
        overlap_assessment="none",
        specificity_alignment="supports_suggested",
        target_background_contrast="target_specific",
        current_class_plausible=False,
        current_class_plausibility_reason="",
        cue_verifier={
            "current_class_plausible": True,
            "current_class_plausibility_reason": "verifier sees a plausible truck/trailer body",
        },
        visible_target_cues=["rectangular white body", "flat roof surface"],
        supporting_clean_evidence_ids=["target_context_1"],
        model_compact_arguments={
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_suggested": "strong",
            "overlap_assessment": "none",
            "specificity_alignment": "supports_suggested",
            "target_background_contrast": "target_specific",
        },
    )

    audit = audit_records([record])
    summary = _summarize([record], run_id="r", job_id="j", model_id="m", started_at=0.0)

    assert audit["current_class_plausible_count"] == 1
    assert audit["issue_counts"]["class_change_current_class_still_plausible"] == 1
    assert "truck/trailer" in audit["issues"]["class_change_current_class_still_plausible"][0]["reason"]
    assert summary["current_class_plausible_count"] == 1


def test_qwen_review_benchmark_audit_counts_cue_verifier_contrastive_support():
    record = _record(
        cue_verifier={
            "contrastively_supported_target": True,
            "current_class_missing_or_inconsistent_cues": [
                "missing expected target part",
            ],
        },
    )

    audit = audit_records([record])
    summary = _summarize([record], run_id="r", job_id="j", model_id="m", started_at=0.0)

    assert audit["cue_verifier_contrastive_support_count"] == 1
    assert audit["cue_verifier_missing_current_cue_count"] == 1
    assert summary["cue_verifier_contrastive_support_count"] == 1
    assert summary["cue_verifier_missing_current_cue_count"] == 1


def test_qwen_review_benchmark_audit_counts_cue_verifier_parse_errors():
    record = _record(cue_verifier_parse_errors=2)

    audit = audit_records([record])
    summary = _summarize([record], run_id="r", job_id="j", model_id="m", started_at=0.0)

    assert audit["cue_verifier_parse_error_count"] == 2
    assert summary["cue_verifier_parse_error_count"] == 2


def test_qwen_review_benchmark_audit_allows_overlap_rebutted_confirm_current():
    record = _record(
        decision="confirm_current",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        current_evidence="strong",
        suggested_evidence="strong",
        target_evidence="strong",
        anchor_evidence_current="strong",
        overlap_assessment="partial_contamination",
        overlap_explains_candidate_similarity=True,
        rationale_short="Target shows current-class cues; overlap explains suggested-class signal.",
        model_compact_arguments={
            "current_evidence": "strong",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_current": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": True,
            "specificity_alignment": "supports_current",
            "target_background_contrast": "overlap_dominated",
        },
    )

    audit = audit_records([record])

    assert audit["unsafe_issue_count"] == 0


def test_qwen_review_benchmark_audit_allows_specificity_probe_rebutted_confirm_current():
    record = _record(
        decision="confirm_current",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        current_class="Truck",
        suggested_neighbor_class="LightVehicle",
        current_evidence="strong",
        suggested_evidence="strong",
        target_evidence="strong",
        anchor_evidence_current="moderate",
        overlap_assessment="unclear",
        overlap_explains_candidate_similarity=False,
        visible_target_cues=["white cab", "rectangular cargo bed", "white tank in bed"],
        rationale_short="Distinct truck features confirm the current class despite strong neighbor signal.",
        specificity_alignment="supports_current",
        target_background_contrast="target_specific",
        specificity_probe={
            "status": "completed",
            "confidence": 0.92,
            "specificity_alignment": "supports_current",
            "target_background_contrast": "target_specific",
            "best_supported_class": "Truck",
            "target_specific_cues": ["white cab", "rectangular cargo bed", "white tank in bed"],
        },
        model_compact_arguments={
            "current_evidence": "strong",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_current": "moderate",
            "overlap_assessment": "unclear",
            "overlap_explains_candidate_similarity": False,
            "specificity_alignment": "supports_current",
            "target_background_contrast": "target_specific",
        },
    )

    audit = audit_records([record])

    assert audit["unsafe_issue_count"] == 0


def test_qwen_review_benchmark_audit_blocks_weak_probe_rebutted_confirm_current():
    record = _record(
        decision="confirm_current",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        current_class="Truck",
        suggested_neighbor_class="LightVehicle",
        current_evidence="strong",
        suggested_evidence="strong",
        target_evidence="strong",
        anchor_evidence_current="moderate",
        overlap_assessment="unclear",
        overlap_explains_candidate_similarity=False,
        visible_target_cues=["white cab", "rectangular cargo bed"],
        specificity_alignment="supports_current",
        target_background_contrast="target_specific",
        specificity_probe={
            "status": "completed",
            "confidence": 0.55,
            "specificity_alignment": "supports_current",
            "target_background_contrast": "target_specific",
            "best_supported_class": "Truck",
        },
        model_compact_arguments={
            "current_evidence": "strong",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_current": "moderate",
            "overlap_assessment": "unclear",
            "overlap_explains_candidate_similarity": False,
            "specificity_alignment": "supports_current",
            "target_background_contrast": "target_specific",
        },
    )

    audit = audit_records([record])

    assert audit["issue_counts"]["confirm_overrides_strong_suggested"] == 1


def test_qwen_review_benchmark_audit_blocks_unrebutted_strong_suggestion_confirm_current():
    record = _record(
        decision="confirm_current",
        backend_tier="clear",
        visual_quality="clear",
        object_visibility="clear",
        current_evidence="strong",
        suggested_evidence="strong",
        target_evidence="strong",
        anchor_evidence_current="strong",
        overlap_assessment="none",
        overlap_explains_candidate_similarity=False,
        rationale_short="Target looks like current class.",
        model_compact_arguments={
            "current_evidence": "strong",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "anchor_evidence_current": "strong",
            "overlap_assessment": "none",
            "overlap_explains_candidate_similarity": False,
        },
    )

    audit = audit_records([record])

    assert audit["issue_counts"]["confirm_overrides_strong_suggested"] == 1


def test_qwen_review_benchmark_compare_runs_tracks_decision_drift():
    previous = [_record(point_id="p1", decision="accept_suggested", target_class="LightVehicle", confidence=0.8)]
    current = [_record(point_id="p1", decision="skip_uncertain", target_class="Truck", confidence=0.4)]

    comparison = compare_runs(current, previous)

    assert comparison["matched_current_records"] == 1
    assert comparison["changed_count"] == 1
    assert comparison["changes"][0]["changed"]["decision"] == {
        "previous": "accept_suggested",
        "current": "skip_uncertain",
    }


def test_qwen_review_benchmark_source_run_filters_before_slicing(tmp_path):
    source = tmp_path / "source.json"
    records = [
        {
            "point_id": "p1",
            "backend_tier": "limited",
            "decision": "skip_uncertain",
            "review_disposition": {"disposition": "guarded_visual_quality", "signal": "guarded_human_triage"},
            "guarded_recommendation": {"blocked": True},
        },
        {
            "point_id": "p2",
            "backend_tier": "clear",
            "decision": "skip_uncertain",
            "review_disposition": {"disposition": "guarded_overlap_risk", "signal": "guarded_human_triage"},
            "guarded_recommendation": {"blocked": True},
        },
        {
            "point_id": "p3",
            "backend_tier": "clear",
            "decision": "accept_suggested",
            "review_disposition": {"disposition": "actionable_class_change", "signal": "actionable"},
        },
        {
            "point_id": "p4",
            "backend_tier": "clear",
            "decision": "skip_uncertain",
            "review_disposition": {"disposition": "target_not_reviewable", "signal": "none"},
            "guarded_recommendation": {"blocked": True},
        },
        {"point_id": "p5", "backend_tier": "clear", "decision": "skip_uncertain"},
    ]
    source.write_text(json.dumps({"records": records}), encoding="utf-8")

    assert _normalize_filter_values([" clear, LIMITED ", ""]) == {"clear", "limited"}
    assert _source_point_ids(
        source,
        count=2,
        start=0,
        backend_tiers={"clear"},
        guarded_only=True,
        reviewable_only=True,
    ) == ["p2"]
    assert _source_point_ids(
        source,
        count=1,
        start=1,
        backend_tiers={"clear"},
        disposition_signals={"guarded_human_triage", "actionable"},
    ) == ["p3"]
