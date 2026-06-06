from tools.analyze_class_split_qwen_review_benchmark import audit_records, compare_runs


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
        },
    )

    audit = audit_records([record])

    assert audit["unsafe_issue_count"] == 2
    assert audit["issue_counts"]["non_skip_low_quality"] == 1
    assert audit["issue_counts"]["class_change_low_backend_quality"] == 1


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
        },
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
            "guardrail_reasons": ["accept_suggested requires clear backend visual-quality tier, got limited"],
            "rationale_short": "target looks like a pickup",
        },
    )

    audit = audit_records([record])

    assert audit["unsafe_issue_count"] == 0
    assert audit["guarded_recommendation_count"] == 1
    guarded = audit["guarded_recommendations"][0]
    assert guarded["point_id"] == "p1"
    assert guarded["guarded_recommendation"]["target_class"] == "LightVehicle"


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
