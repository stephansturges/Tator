import copy

import localinferenceapi as api


def test_refined_review_contract_requires_all_deterministic_tools():
    required, sequence = (
        api._class_analysis_qwen_review_required_tool_contract(
            patch_refinement={"sidecar_row": 0},
            dual_bbox_conflict=None,
        )
    )
    sequence_names = [name for name, _args in sequence]

    assert "inspect_patch_refinement" in required
    assert "inspect_same_image_scale_report" in required
    assert "inspect_same_image_embedding_report" in required
    assert "inspect_same_image_scale_report" in sequence_names
    assert "inspect_same_image_embedding_report" in sequence_names


def test_missing_expected_refinement_preview_blocks_class_change_and_preserves_raw():
    point = {
        "point_id": "pole-among-buses",
        "class_name": "TransitObject",
        "suggested_neighbor_class": "Elevated fixture",
    }
    model_result = {
        "decision": "accept_suggested",
        "final_class": "Elevated fixture",
        "confidence": 0.94,
        "rationale_short": "Visible pole shape.",
    }
    observation = {
        "preview_status": "unavailable",
        "preview_integrity": {"status": "unavailable"},
        "preview_error": "refinement_sidecar_checksum_mismatch",
    }

    guarded = (
        api._class_analysis_qwen_review_apply_refinement_preview_guard(
            model_result,
            point,
            observation,
        )
    )

    assert guarded["decision"] == "skip_uncertain"
    assert guarded["controller_reconciliation"]["applied"] is True
    assert (
        guarded["model_recommendation_before_guardrail"]
        == model_result
    )
    assert (
        guarded["patch_refinement_preview"]["status"]
        == "unavailable"
    )


def test_validated_refinement_preview_preserves_vlm_judgment():
    point = {"point_id": "p", "class_name": "TransitObject"}
    model_result = {
        "decision": "accept_suggested",
        "final_class": "Elevated fixture",
        "confidence": 0.91,
    }
    original = copy.deepcopy(model_result)

    result = (
        api._class_analysis_qwen_review_apply_refinement_preview_guard(
            model_result,
            point,
            {
                "preview_status": "validated",
                "preview_integrity": {
                    "status": "validated",
                    "sidecar_sha256": "ab" * 32,
                },
                "preview_error": "",
            },
        )
    )

    assert result["decision"] == original["decision"]
    assert result["final_class"] == original["final_class"]
    assert (
        result["patch_refinement_preview"]["status"]
        == "validated"
    )
    assert "controller_reconciliation" not in result
