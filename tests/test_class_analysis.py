import asyncio
import copy
import inspect
import json
import math
import re
import types
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
from tools import run_class_split_experiments as class_split_experiments
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
    assert conflict["review_mode"] == "dual_bbox_class_resolution"
    assert conflict["other_class_name"] == "boat"
    assert conflict["iou"] >= 0.98
    candidate = next(item for item in result["wrong_class_candidates"] if item["point_id"] == "p0")
    assert candidate["is_dual_bbox_conflict"] is True
    assert candidate["dual_bbox_conflict"]["other_class_name"] == "boat"
    same_class_candidate = next(item for item in result["wrong_class_candidates"] if item["point_id"] == "p1")
    assert same_class_candidate["wrong_class_review_reason"] == "dual_bbox_conflict"
    assert same_class_candidate["embedding_wrong_class_suspicion"] < same_class_candidate["wrong_class_suspicion"]
    assert same_class_candidate["dual_bbox_conflict"]["other_class_name"] == "car"
    assert result["summary"]["dual_bbox_conflict_count"] >= 2


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
                {"type": "image", "image": "source_overlay.jpg"},
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
    assert policy["output_image_count"] == 3
    assert image_values == ["target.jpg", "class_context.jpg", "zoom.jpg"]
    assert "inspect_class_context_pack" not in policy["text_only_observations"]
    assert "inspect_source_overlay" in policy["text_only_observations"]
    assert "inspect_local_consensus_context" in policy["text_only_observations"]


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


def test_class_analysis_qwen_review_router_policy_masks_disallowed_local_consensus():
    point = {"class_name": "UPole", "suggested_neighbor_class": "LightVehicle"}
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

    assert router["action"] == "finalize_now"
    assert router["reason_code"] == "policy_blocked"
    assert router["confidence"] <= 0.35
    assert "target_quality_not_clear" in router["policy_reasons"]


def test_class_analysis_qwen_review_compact_final_schema_expands_to_full_audit_payload():
    result = {"summary": {"labelmap": ["Truck", "LightVehicle"]}}
    point = {
        "point_id": "p0",
        "class_name": "Truck",
        "suggested_neighbor_class": "LightVehicle",
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
    spec = api._class_analysis_qwen_review_final_tool_spec(["Truck", "LightVehicle"])
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
            "final_class": "LightVehicle",
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
    assert final["target_class"] == "LightVehicle"
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
    result = {"summary": {"labelmap": ["Boat", "LightVehicle"]}}
    point = {
        "point_id": "p0",
        "class_name": "Boat",
        "suggested_neighbor_class": "LightVehicle",
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
        "class_name": "UPole",
        "suggested_neighbor_class": "LightVehicle",
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
            "target_class": "UPole",
            "confidence": 0.85,
            "visual_quality": "clear",
            "object_visibility": "clear",
            "current_evidence": "weak",
            "suggested_evidence": "strong",
            "target_evidence": "strong",
            "overlap_assessment": "partial_contamination",
            "overlap_explains_candidate_similarity": False,
            "rationale_short": (
                "Target crop clearly shows a car fitting LightVehicle. "
                "The current UPole label is likely a misclassification."
            ),
        },
        point=point,
        evidence_ids={"target_context_1", "zoom_region_6"},
        visual_quality=clear_quality,
        executed_tools={"inspect_target_context", "zoom_source_region"},
    )

    assert expanded["_controller_reconciliation"]["applied"] is False
    assert expanded["decision"] == "accept_suggested"
    assert expanded["target_class"] == "LightVehicle"


def test_class_analysis_qwen_review_does_not_reconcile_non_adjacent_skip():
    result = {"summary": {"labelmap": ["Solarpanels", "LightVehicle"]}}
    point = {
        "point_id": "p0",
        "class_name": "Solarpanels",
        "suggested_neighbor_class": "LightVehicle",
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
    result = {"summary": {"labelmap": ["UPole", "Solarpanels"]}}
    point = {
        "point_id": "p0",
        "class_name": "UPole",
        "suggested_neighbor_class": "Solarpanels",
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
            "uncertain_class": "Solarpanels",
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
    assert final["target_class"] == "Solarpanels"


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
    result = {"summary": {"labelmap": ["Truck", "LightVehicle"]}}
    point = {
        "point_id": "p0",
        "class_name": "Truck",
        "suggested_neighbor_class": "LightVehicle",
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
            "target_class": "LightVehicle",
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
            "rationale_short": "Class comparison indicates LightVehicle is a better fit.",
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
    result = {"summary": {"labelmap": ["Truck", "LightVehicle"]}}
    point = {
        "point_id": "p0",
        "class_name": "Truck",
        "suggested_neighbor_class": "LightVehicle",
        "dual_bbox_conflict": {
            "enabled": True,
            "kind": "near_identical_cross_class_bbox",
            "review_mode": "dual_bbox_class_resolution",
            "point_id": "p0",
            "current_class": "Truck",
            "other_point_id": "p1",
            "other_class_name": "LightVehicle",
            "class_name": "LightVehicle",
            "classes": ["Truck", "LightVehicle"],
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
                    "class_name": "LightVehicle",
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
            "final_class": "LightVehicle",
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
            "rationale_short": "target pixels match the overlapping LightVehicle box",
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
    assert final["target_class"] == "LightVehicle"
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
            "final_class": "LightVehicle",
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
            "rationale_short": "target pixels match the near-identical LightVehicle box, not Truck.",
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
    assert inconsistent_final["target_class"] == "LightVehicle"
    assert inconsistent_final["dual_bbox_resolution"] == "overlap_box_class"
    assert "accept_suggested has only moderate suggested-anchor agreement" in inconsistent_final["advisory_reasons"]


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
    assert '"target_class": "SuggestedClass"' in text
    assert "Do not include legacy or diagnostic keys" in text
    assert "current_class, proposed_target_class, verified_evidence_ids" in text
    assert "Use supporting_clean_evidence_ids, not verified_evidence_ids." in text
    assert "whole reviewed bbox/object extent" in text
    assert "Output compact JSON" in text
    assert "0.92 not 0. 92" in text
    assert "under 18 words" in text
    assert "subcomponent" in text


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
                "aerial view of parked candidate class",
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
    result = {"summary": {"labelmap": ["Building", "LightVehicle"]}}
    point = {
        "point_id": "p0",
        "class_name": "Building",
        "suggested_neighbor_class": "LightVehicle",
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
            "target_class": "LightVehicle",
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
        "target_class": "LightVehicle",
        "rationale_short": (
            "Target is small, compact, no cargo; matches LightVehicle visual cues; "
            "no overlap contamination"
        ),
        "counter_evidence": "No explicit counterevidence provided.",
        "visible_target_cues": ["Compact size"],
    }

    conflict = api._class_analysis_qwen_review_text_conflicts_with_accept_suggested(
        current_class="Truck",
        suggested_class="LightVehicle",
        payload=payload,
        labelmap=["Truck", "LightVehicle"],
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
    result = {"summary": {"labelmap": ["UPole", "Person"]}}
    point = {
        "point_id": "p0",
        "class_name": "UPole",
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
    assert final["target_class"] == "UPole"
    assert final["guarded_recommendation"]["blocked"] is True
    assert final["guarded_recommendation"]["decision"] == "accept_suggested"
    assert final["guarded_recommendation"]["target_class"] == "Person"
    assert final["guarded_recommendation"]["confidence"] == 0.82
    assert "suggested class looks better" in final["guarded_recommendation"]["rationale_short"]


def test_class_analysis_qwen_review_blocks_self_conflicting_class_recommendations():
    result = {"summary": {"labelmap": ["Truck", "LightVehicle", "Container"]}}
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
            "target_class": "LightVehicle",
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
        {"point_id": "p0", "class_name": "Truck", "suggested_neighbor_class": "LightVehicle"},
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
    result = {"summary": {"labelmap": ["Boat", "Building", "LightVehicle", "Truck"]}}
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
            "target_class": "LightVehicle",
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
            "rationale_short": "The target crop clearly shows a small boat, contradicting the LightVehicle suggestion.",
            "counter_evidence": "The object is clearly a small boat, not a car or light vehicle.",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p0", "class_name": "Boat", "suggested_neighbor_class": "LightVehicle"},
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
            "rationale_short": "The target is a small red shed with a roof, clearly a Building. The current LightVehicle label is incorrect.",
            "counter_evidence": "No vehicle features are visible.",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p1", "class_name": "LightVehicle", "suggested_neighbor_class": "Building"},
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
    result = {"summary": {"labelmap": ["Building", "LightVehicle", "Truck"]}}
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
    result = {"summary": {"labelmap": ["Building", "LightVehicle"]}}
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
                    "class_name": "LightVehicle",
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
            "target_class": "LightVehicle",
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
            "rationale_short": "Target matches LightVehicle traits; overlap does not explain vehicle features.",
            "counter_evidence": "",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p0", "class_name": "Building", "suggested_neighbor_class": "LightVehicle"},
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
        {"point_id": "p0", "class_name": "Building", "suggested_neighbor_class": "LightVehicle"},
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
                        "class_name": "LightVehicle",
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
        {"point_id": "p0", "class_name": "Building", "suggested_neighbor_class": "LightVehicle"},
        {"tier": "clear"},
        {
            "overlap_decomposition": {
                "overlaps": [
                    {"class_name": "Building", "relation": "partial_contamination", "target_area_covered": 0.52},
                    {"class_name": "LightVehicle", "relation": "partial_contamination", "target_area_covered": 0.35},
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
    result = {"summary": {"labelmap": ["UPole", "LightVehicle"]}}
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
            "target_class": "LightVehicle",
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
            "rationale_short": "Target crop clearly shows a car, so LightVehicle is plausible.",
            "counter_evidence": "A thin pole-like structure is visible, which could justify the UPole label.",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p0", "class_name": "UPole", "suggested_neighbor_class": "LightVehicle"},
        {"ctx_1"},
        clear_quality,
    )

    assert accepted["decision"] == "skip_uncertain"
    assert any("model text supporting current class UPole" in reason for reason in accepted["guardrail_reasons"])

    plausible_current = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "LightVehicle",
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
        {"summary": {"labelmap": ["Truck", "LightVehicle"]}},
        {"point_id": "p1", "class_name": "Truck", "suggested_neighbor_class": "LightVehicle"},
        {"ctx_1"},
        clear_quality,
    )

    assert plausible_current["decision"] == "skip_uncertain"
    assert any("model text supporting current class Truck" in reason for reason in plausible_current["guardrail_reasons"])

    mixed_reject_and_support = api._class_analysis_qwen_review_validate_final(
        {
            "decision": "accept_suggested",
            "target_class": "LightVehicle",
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
        {"summary": {"labelmap": ["Truck", "LightVehicle"]}},
        {"point_id": "p2", "class_name": "Truck", "suggested_neighbor_class": "LightVehicle"},
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
            "LightVehicle",
            "The target is a small white boat on a trailer, visually matching LightVehicle.",
        ),
        (
            "Truck",
            "LightVehicle",
            "Target crop shows a clear truck with a cab and open bed, distinct from the nearby car.",
        ),
        (
            "Gastank",
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
    result = {"summary": {"labelmap": ["Truck", "LightVehicle"]}}
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
            "target_class": "LightVehicle",
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
                "Target crop clearly shows a pickup truck with an open bed, fitting LightVehicle. "
                "Current Truck label is broad and weak."
            ),
            "counter_evidence": "No explicit counterevidence provided.",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p0", "class_name": "Truck", "suggested_neighbor_class": "LightVehicle"},
        {"ctx_1"},
        clear_quality,
    )

    assert accepted["decision"] == "accept_suggested"
    assert accepted["target_class"] == "LightVehicle"
    assert not accepted["guardrail_reasons"]


@pytest.mark.parametrize(
    ("current_class", "suggested_class", "rationale", "counter_evidence"),
    [
        (
            "UPole",
            "Gastank",
            "Target is a clear horizontal tank (Gastank). Current UPole label is weak as it lacks vertical pole features.",
            "Current UPole anchors show vertical poles, while the target is a horizontal tank.",
        ),
        (
            "Container",
            "Building",
            "Target is a clear residential roof (Building). Current class (Container) is weak and visually mismatched.",
            "Current class (Container) anchors are industrial, while the target is a clear residential roof.",
        ),
        (
            "Building",
            "Solarpanels",
            "Target crop clearly shows a solar panel array with grid structure, distinct from the large building roofs labeled as Building.",
            "Local consensus shows 12 Building anchors, but the target crop matches Solarpanels anchors.",
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
    result = {"summary": {"labelmap": ["UPole", "LightVehicle"]}}
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
            "target_class": "LightVehicle",
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
                "Target crop clearly shows a light vehicle. Current UPole label is weak; "
                "the vertical pole is a minor background element and overlap does not explain target features."
            ),
            "counter_evidence": "No explicit counterevidence provided.",
            "human_review_needed": False,
        },
        result,
        {"point_id": "p0", "class_name": "UPole", "suggested_neighbor_class": "LightVehicle"},
        {"ctx_1"},
        clear_quality,
    )

    assert accepted["decision"] == "accept_suggested"
    assert accepted["target_class"] == "LightVehicle"
    assert any("partial overlap present" in reason for reason in accepted["advisory_reasons"])


def test_class_analysis_qwen_review_allows_background_overlap_not_vehicle_rebuttal():
    result = {"summary": {"labelmap": ["UPole", "LightVehicle"]}}
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
            "target_class": "LightVehicle",
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
        {"point_id": "p0", "class_name": "UPole", "suggested_neighbor_class": "LightVehicle"},
        {"ctx_1"},
        clear_quality,
    )

    assert accepted["decision"] == "accept_suggested"
    assert accepted["target_class"] == "LightVehicle"


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
    result = {"summary": {"labelmap": ["UPole", "Gastank"]}}
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
            "target_class": "Gastank",
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
        {"point_id": "p0", "class_name": "UPole", "suggested_neighbor_class": "Gastank"},
        {"ctx_1"},
        clear_quality,
    )

    assert final["decision"] == "skip_uncertain"
    assert any("partial_contamination" in reason for reason in final["guardrail_reasons"])


def test_class_analysis_qwen_review_blocks_partial_overlap_without_explicit_rebuttal():
    result = {"summary": {"labelmap": ["Boat", "LightVehicle"]}}
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
            "target_class": "LightVehicle",
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
        {"point_id": "p0", "class_name": "Boat", "suggested_neighbor_class": "LightVehicle"},
        {"ctx_1"},
        clear_quality,
    )

    assert accepted["decision"] == "skip_uncertain"
    assert any("partial_contamination" in reason for reason in accepted["guardrail_reasons"])


def test_class_analysis_qwen_review_local_consensus_guardrails():
    result = {"summary": {"labelmap": ["UPole", "LightVehicle"]}}
    point = {
        "point_id": "p0",
        "class_name": "UPole",
        "suggested_neighbor_class": "LightVehicle",
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
            "target_class": "LightVehicle",
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
            "target_class": "UPole",
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

    def fake_qwen_chat(messages, **kwargs):
        calls.append({"messages": copy.deepcopy(messages), "kwargs": dict(kwargs)})
        return next(outputs)

    monkeypatch.setattr(api, "_run_qwen_chat", fake_qwen_chat)
    review = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_test",
        parent_job_id=parent_id,
        point_id="p0",
        request={"max_turns": 8, "model_id": "test-model", "enable_local_consensus_context": True},
    )
    with api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS_LOCK:
        api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS[review.review_id] = review

    api._run_class_analysis_qwen_review_job(review)

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
    assert review.status == "completed"
    assert review.result["decision"] == "accept_suggested"
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
    assert "local_consensus_context_11" in review.result["evidence_ledger"]["local_consensus_evidence_ids"]
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
        "local_consensus_context_11",
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
    assert len(evidence_paths) == 11
    assert any(path.name.startswith("target_detail_") for path in evidence_paths)
    assert any(path.name.startswith("source_clean_") for path in evidence_paths)
    assert any(path.name.startswith("specificity_region_contrast_") for path in evidence_paths)
    assert any(path.name.startswith("local_consensus_context_") for path in evidence_paths)
    assert any(path.name.startswith("zoom_region_") for path in evidence_paths)


def test_class_analysis_qwen_review_mlx_reset_cadence_is_generic_and_logged(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    parent_id = "ca_reset_policy"
    (class_root / parent_id).mkdir(parents=True)
    with api.CLASS_ANALYSIS_QWEN_REVIEW_MLX_RESET_LOCK:
        api.CLASS_ANALYSIS_QWEN_REVIEW_MLX_RESET_STATE["completed_calls"] = 0
    monkeypatch.setattr(api, "qwen_runtime_platform", None)

    def fake_qwen_chat(messages, **kwargs):
        api.qwen_runtime_platform = api.QWEN_PLATFORM_MLX
        return '{"decision":"skip_uncertain"}'

    resets = []

    def fake_reset_qwen_runtime():
        resets.append("reset")

    monkeypatch.setattr(api, "_run_qwen_chat", fake_qwen_chat)
    monkeypatch.setattr(api, "_reset_qwen_runtime", fake_reset_qwen_runtime)
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
            "model_id": "test-model",
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

    def fake_qwen_chat(messages, **kwargs):
        calls.append({"messages": copy.deepcopy(messages), "kwargs": dict(kwargs)})
        return '{"decision":"accept_suggested","target_class":"ClassB","confidence":0.8,"visual_quality":"poor","object_visibility":"tiny_or_blurry","current_evidence":"weak","suggested_evidence":"strong","target_evidence":"strong","anchor_evidence_current":"weak","anchor_evidence_suggested":"strong","local_context_evidence":"strong","global_context_evidence":"strong","same_image_scale_evidence":"insufficient","same_image_embedding_evidence":"insufficient","overlap_assessment":"none","overlap_explains_candidate_similarity":false,"local_consensus_evidence":"not_applicable","visible_target_cues":["bright rectangular target","hard-edged target patch"],"supporting_clean_evidence_ids":["target_detail_2"],"rationale_short":"target appears closer to ClassB but is tiny","counter_evidence":"poor crop quality","human_review_needed":true}'

    monkeypatch.setattr(api, "_run_qwen_chat", fake_qwen_chat)
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
            "allow_poor_final_review": True,
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

    def fake_qwen_chat(messages, **kwargs):
        calls.append({"messages": copy.deepcopy(messages), "kwargs": dict(kwargs)})
        return '{"decision":"confirm_current","target_class":"ClassA","confidence":0.74,"visual_quality":"limited","object_visibility":"partial","current_evidence":"strong","suggested_evidence":"weak","target_evidence":"strong","overlap_assessment":"none","overlap_explains_candidate_similarity":false,"local_consensus_evidence":"not_applicable","visible_target_cues":["compact target outline"],"supporting_clean_evidence_ids":["target_context_1"],"rationale_short":"limited crop still supports current class","counter_evidence":"suggested class cues are not visible","human_review_needed":true}'

    monkeypatch.setattr(api, "_run_qwen_chat", fake_qwen_chat)
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
    assert review.result["decision"] == "skip_uncertain"
    assert review.result["target_class"] == "ClassA"
    assert review.result["confidence"] <= 0.45
    assert review.result["human_review_needed"] is True
    guarded = review.result["guarded_recommendation"]
    assert guarded["blocked"] is True
    assert guarded["decision"] == "confirm_current"
    assert guarded["target_class"] == "ClassA"
    assert any("backend visual-quality tier is limited" in reason for reason in review.result["advisory_reasons"])
    assert any("advisory-only" in reason for reason in review.result["guardrail_reasons"])
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
    assert "Choose accept_suggested, change_to_other, or confirm_current" in text
    assert "class-changing decisions are forbidden" not in text


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

    def fake_qwen_chat(messages, **kwargs):
        calls.append({"messages": copy.deepcopy(messages), "kwargs": dict(kwargs)})
        return next(outputs)

    monkeypatch.setattr(api, "_run_qwen_chat", fake_qwen_chat)
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
            "class_b": "LightVehicle",
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
        class_b="LightVehicle",
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
            "labelmap": ["UPole"],
            "images": [{"split": "train", "image_relpath": "scene.jpg", "label_lines": []}],
            "yolo_layout": "flat",
            "source_mode": "active_workspace",
        },
    )
    point = {
        "point_id": "p0",
        "class_name": "UPole",
        "image_relpath": "scene.jpg",
        "split": "train",
        "bbox_xyxy": [20, 20, 60, 60],
    }
    result = {
        "summary": {"source_mode": "active_workspace", "source_id": parent_id, "labelmap": ["UPole"]},
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
            "labelmap": ["UPole", "LightVehicle", "Boat"],
            "images": [{"split": "train", "image_relpath": "scene.jpg", "label_lines": []}],
            "yolo_layout": "flat",
            "source_mode": "active_workspace",
        },
    )
    target = {
        "point_id": "target",
        "class_name": "UPole",
        "suggested_neighbor_class": "LightVehicle",
        "image_relpath": "scene.jpg",
        "split": "train",
        "bbox_xyxy": [90, 70, 130, 155],
    }
    result = {
        "summary": {
            "source_mode": "active_workspace",
            "source_id": parent_id,
            "labelmap": ["UPole", "LightVehicle", "Boat"],
        },
        "points": [
            target,
            {
                "point_id": "current_near",
                "class_name": "UPole",
                "image_relpath": "scene.jpg",
                "split": "train",
                "bbox_xyxy": [42, 72, 62, 158],
            },
            {
                "point_id": "current_far",
                "class_name": "UPole",
                "image_relpath": "scene.jpg",
                "split": "train",
                "bbox_xyxy": [220, 60, 242, 150],
            },
            {
                "point_id": "suggested_near",
                "class_name": "LightVehicle",
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
    assert all(item["class_name"] in {"UPole", "LightVehicle"} for item in metadata["included_points"])
    assert len(observation["evidence"]) == 1
    assert observation["evidence"][0]["kind"] == "local_consensus_context"
    assert all(Path(path).is_file() for path in observation["image_paths"])
    with Image.open(observation["image_paths"][0]) as rendered:
        assert rendered.width <= 1200
        assert rendered.height <= 900
    assert "cannot override unclear target pixels" in observation["summary"]


def test_class_analysis_qwen_review_overlap_decomposition_marks_partial_contamination():
    point = {
        "point_id": "pole",
        "class_name": "UPole",
        "split": "train",
        "image_relpath": "scene.jpg",
        "bbox_xyxy": [50, 20, 80, 170],
    }
    result = {
        "points": [
            point,
            {
                "point_id": "car",
                "class_name": "LightVehicle",
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
    assert overlaps[0]["class_name"] == "LightVehicle"
    assert overlaps[0]["relation"] == "partial_contamination"
    assert overlaps[0]["target_area_covered"] > 0.25


def test_class_analysis_qwen_review_anchor_selection_prefers_clean_class_anchors():
    point = {
        "point_id": "target",
        "class_name": "UPole",
        "split": "train",
        "image_relpath": "scene.jpg",
    }
    result = {
        "points": [
            point,
            {
                "point_id": "clean",
                "class_name": "UPole",
                "split": "train",
                "image_relpath": "other.jpg",
                "bbox_xyxy": [0, 0, 80, 80],
                "same_class_neighbor_ratio": 0.95,
                "top_other_neighbor_ratio": 0.05,
                "outlier_score": 0.1,
            },
            {
                "point_id": "suspicious",
                "class_name": "UPole",
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
        result, point, "UPole", same_image=False, limit=3
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
        {"point_id": "target", "class_name": "Boat", "suggested_neighbor_class": "LightVehicle"},
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
        "suggested_neighbor_class": "LightVehicle",
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
                        "class_name": "LightVehicle",
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
        {"summary": {"labelmap": ["UPole", "LightVehicle"]}},
        {
            "point_id": "p0",
            "class_name": "UPole",
            "suggested_neighbor_class": "LightVehicle",
        },
        {"tier": "clear", "bbox_width": 50, "bbox_height": 100},
        labelmap_glossary='{"UPole":["utility pole","satellite dish"]}',
        review_guidance="UPole includes drone-obstruction fixtures in this dataset.",
    )

    assert "Relevant class meaning glossary" in text
    assert "satellite dish" in text
    assert "Additional review guidance" in text
    assert "drone-obstruction" in text


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


def test_class_analysis_mobile_review_queue_and_actions(monkeypatch):
    api.CLASS_ANALYSIS_JOBS.clear()
    api.CLASS_ANALYSIS_MOBILE_REVIEW_SESSIONS.clear()

    monkeypatch.setattr(
        api,
        "_class_analysis_mobile_touch_lock",
        lambda session, required=False: {"status": "ok", "lock": {"session_id": session.annotation_session_id}},
    )
    points = [
        {
            **_record("p0", "building"),
            "is_wrong_class_candidate": True,
            "suggested_neighbor_class": "boat",
            "wrong_class_suspicion": 0.8,
        },
        {
            **_record("p1", "tree"),
            "is_wrong_class_candidate": True,
            "suggested_neighbor_class": "boat",
            "wrong_class_suspicion": 0.7,
        },
    ]
    api.CLASS_ANALYSIS_JOBS["ca_mobile"] = api.ClassAnalysisJob(
        job_id="ca_mobile",
        status="completed",
        result={
            "summary": {
                "source_mode": "linked",
                "source_id": "dataset_a",
                "dataset_label": "Dataset A",
                "labelmap": ["building", "boat", "tree"],
            },
            "points": points,
            "wrong_class_candidates": [
                {"point_id": "p0"},
                {"point_id": "p1"},
            ],
        },
    )

    created = api.create_class_analysis_mobile_review(
        "ca_mobile",
        {"annotation_session_id": "desktop-lock", "dismissed_point_ids": []},
    )
    session_id = created["session_id"]
    assert created["target_mode"] == "desktop_workspace"
    assert created["current"]["point_id"] == "p0"
    assert created["counts"]["remaining"] == 2
    assert created["mobile_url"] == f"/mobile_review.html?session={session_id}"

    skipped = api.class_analysis_mobile_review_action(session_id, {"action": "skip_next", "count": 1})
    assert skipped["counts"]["skipped"] == 1
    assert skipped["current"]["point_id"] == "p1"
    assert skipped["action"]["point_ids"] == ["p0"]
    assert skipped["action_log"][-1]["action_id"] == f"{session_id}:1"
    assert skipped["action_log"][-1]["sequence"] == 1
    assert skipped["action_log"][-1]["point_ids"] == ["p0"]

    changed = api.class_analysis_mobile_review_action(
        session_id,
        {"action": "change_class", "point_id": "p1", "target_class": "boat"},
    )
    assert changed["counts"]["changed"] == 1
    assert changed["counts"]["remaining"] == 0
    assert changed["action_log"][-1]["action_id"] == f"{session_id}:2"
    assert changed["action_log"][-1]["sequence"] == 2
    assert changed["action_log"][-1]["target_class"] == "boat"
    assert points[1]["class_name"] == "boat"
    assert points[1]["is_wrong_class_candidate"] is False
    api.CLASS_ANALYSIS_JOBS.clear()
    api.CLASS_ANALYSIS_MOBILE_REVIEW_SESSIONS.clear()


def test_class_analysis_mobile_review_forces_desktop_workspace_without_backend_write(monkeypatch):
    api.CLASS_ANALYSIS_JOBS.clear()
    api.CLASS_ANALYSIS_MOBILE_REVIEW_SESSIONS.clear()

    monkeypatch.setattr(
        api,
        "save_dataset_annotation_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("mobile review must not write backend snapshots")),
    )
    point = {
        **_record("p0", "building"),
        "is_wrong_class_candidate": True,
        "suggested_neighbor_class": "boat",
        "review_signals": ["wrong_class"],
        "wrong_class_suspicion": 0.8,
    }
    api.CLASS_ANALYSIS_JOBS["ca_mobile_desktop"] = api.ClassAnalysisJob(
        job_id="ca_mobile_desktop",
        status="completed",
        result={
            "summary": {
                "source_mode": "linked",
                "source_id": "uploaded_active_workspace",
                "dataset_label": "Uploaded active workspace",
                "labelmap": ["building", "boat"],
            },
            "points": [point],
            "wrong_class_candidates": [{"point_id": "p0"}],
        },
    )

    created = api.create_class_analysis_mobile_review(
        "ca_mobile_desktop",
        {"target_mode": "backend", "annotation_session_id": "desktop-lock"},
    )
    assert created["target_mode"] == "desktop_workspace"
    assert created["lock"] == {}

    changed = api.class_analysis_mobile_review_action(
        created["session_id"],
        {"action": "change_class", "point_id": "p0", "target_class": "boat"},
    )

    assert changed["target_mode"] == "desktop_workspace"
    assert changed["counts"]["changed"] == 1
    assert changed["counts"]["remaining"] == 0
    assert changed["action_log"][-1]["name"] == "change_class"
    assert changed["action_log"][-1]["action_id"] == f"{created['session_id']}:1"
    assert changed["action_log"][-1]["sequence"] == 1
    assert changed["action_log"][-1]["status"] == "changed"
    assert changed["action_log"][-1]["point_id"] == "p0"
    assert changed["action_log"][-1]["target_class"] == "boat"
    assert changed["action_log"][-1]["timestamp"] > 0
    assert point["class_name"] == "boat"
    assert point["is_wrong_class_candidate"] is False
    assert point["review_signals"] == []
    api.CLASS_ANALYSIS_JOBS.clear()
    api.CLASS_ANALYSIS_MOBILE_REVIEW_SESSIONS.clear()


def test_class_analysis_mobile_review_accepts_active_workspace_context(tmp_path, monkeypatch):
    api.CLASS_ANALYSIS_JOBS.clear()
    api.CLASS_ANALYSIS_MOBILE_REVIEW_SESSIONS.clear()
    class_root = tmp_path / "class_analysis"
    workspace = class_root / "ca_active_mobile" / "active_workspace"
    images_dir = workspace / "images"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (80, 60), (40, 80, 120)).save(images_dir / "frame.jpg")
    (workspace / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_label": "Live desktop workspace",
                "labelmap": ["building", "boat"],
                "yolo_layout": "flat",
                "images": [
                    {
                        "split": "train",
                        "image_relpath": "frame.jpg",
                        "label_lines": ["0 0.5 0.5 0.5 0.5"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    point = {
        **_record("p0", "building"),
        "image_relpath": "frame.jpg",
        "bbox_xyxy": [20, 15, 60, 45],
        "is_wrong_class_candidate": True,
        "suggested_neighbor_class": "boat",
    }
    api.CLASS_ANALYSIS_JOBS["ca_active_mobile"] = api.ClassAnalysisJob(
        job_id="ca_active_mobile",
        status="completed",
        result={
            "summary": {
                "source_mode": "active_workspace",
                "source_id": "ca_active_mobile",
                "dataset_label": "Live desktop workspace",
                "labelmap": ["building", "boat"],
            },
            "points": [point],
            "wrong_class_candidates": [{"point_id": "p0"}],
        },
    )

    created = api.create_class_analysis_mobile_review("ca_active_mobile", {})
    assert created["source_mode"] == "active_workspace"
    assert created["target_mode"] == "desktop_workspace"
    context = api.get_class_analysis_mobile_review_context(created["session_id"], "p0")
    assert context.media_type == "image/jpeg"
    assert context.body
    changed = api.class_analysis_mobile_review_action(
        created["session_id"],
        {"action": "change_class", "point_id": "p0", "target_class": "boat"},
    )
    assert changed["target_mode"] == "desktop_workspace"
    assert changed["counts"]["changed"] == 1
    api.CLASS_ANALYSIS_JOBS.clear()
    api.CLASS_ANALYSIS_MOBILE_REVIEW_SESSIONS.clear()


def test_class_analysis_mobile_review_prunes_stale_and_excess_sessions(monkeypatch):
    api.CLASS_ANALYSIS_MOBILE_REVIEW_SESSIONS.clear()
    monkeypatch.setattr(api, "CLASS_ANALYSIS_MOBILE_REVIEW_TTL_SECONDS", 100)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_MOBILE_REVIEW_MAX_SESSIONS", 2)
    api.CLASS_ANALYSIS_MOBILE_REVIEW_SESSIONS["old"] = api.ClassAnalysisMobileReviewSession(
        session_id="old",
        job_id="job",
        source_mode="linked",
        source_id="dataset",
        updated_at=50.0,
    )
    api.CLASS_ANALYSIS_MOBILE_REVIEW_SESSIONS["keep_a"] = api.ClassAnalysisMobileReviewSession(
        session_id="keep_a",
        job_id="job",
        source_mode="linked",
        source_id="dataset",
        updated_at=950.0,
    )
    api.CLASS_ANALYSIS_MOBILE_REVIEW_SESSIONS["keep_b"] = api.ClassAnalysisMobileReviewSession(
        session_id="keep_b",
        job_id="job",
        source_mode="linked",
        source_id="dataset",
        updated_at=970.0,
    )
    api.CLASS_ANALYSIS_MOBILE_REVIEW_SESSIONS["keep_c"] = api.ClassAnalysisMobileReviewSession(
        session_id="keep_c",
        job_id="job",
        source_mode="linked",
        source_id="dataset",
        updated_at=990.0,
    )

    api._prune_class_analysis_mobile_review_sessions(now=1000.0)

    assert set(api.CLASS_ANALYSIS_MOBILE_REVIEW_SESSIONS) == {"keep_b", "keep_c"}
    api.CLASS_ANALYSIS_MOBILE_REVIEW_SESSIONS.clear()


def test_class_analysis_mobile_review_excludes_overlap_only_candidates(monkeypatch):
    api.CLASS_ANALYSIS_JOBS.clear()
    api.CLASS_ANALYSIS_MOBILE_REVIEW_SESSIONS.clear()
    monkeypatch.setattr(
        api,
        "_class_analysis_mobile_touch_lock",
        lambda session, required=False: {"status": "ok", "lock": {"session_id": session.annotation_session_id}},
    )
    api.CLASS_ANALYSIS_JOBS["ca_overlap_only"] = api.ClassAnalysisJob(
        job_id="ca_overlap_only",
        status="completed",
        result={
            "summary": {
                "source_mode": "linked",
                "source_id": "dataset_a",
                "dataset_label": "Dataset A",
                "labelmap": ["building", "boat"],
            },
            "points": [
                {
                    **_record("p0", "building"),
                    "is_wrong_class_candidate": False,
                    "is_close_overlap_candidate": True,
                    "review_signals": ["close_overlap"],
                    "close_overlap_matches": [{"point_id": "p1", "class_name": "boat"}],
                }
            ],
            "wrong_class_candidates": [],
            "close_overlap_candidates": [{"point_id": "p0", "other_point_id": "p1"}],
        },
    )

    with pytest.raises(api.HTTPException) as exc:
        api.create_class_analysis_mobile_review("ca_overlap_only", {})

    assert exc.value.status_code == 400
    assert exc.value.detail == "mobile_review_no_candidates"
    api.CLASS_ANALYSIS_JOBS.clear()
    api.CLASS_ANALYSIS_MOBILE_REVIEW_SESSIONS.clear()


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

    fallback = api._normalize_class_analysis_request({"projection_mode": "unknown"})
    assert fallback["projection"] == "pca"
    assert fallback["projection_mode"] == api.CLASS_ANALYSIS_DEFAULT_PCA_PROJECTION_MODE


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

    public = api._class_analysis_public_result(result)
    assert "coordinates" not in public["projection_options"]
    assert public["projection_options"]["coordinates_available"] == api.CLASS_ANALYSIS_PCA_PROJECTION_MODES
    assert public["points"][0]["projection"] == pytest.approx(first_point["projection"])


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

    assert caps["preprocess_modes"] == ["canonical"]
    assert caps["embedding_adjustments"] == ["remove_size_bias"]
    assert caps["expert_preprocess_modes"] == ["native", "canonical"]
    assert caps["expert_embedding_adjustments"] == ["none", "remove_size_bias"]
    assert caps["default_preprocess_mode"] == "canonical"
    assert caps["default_embedding_adjustment"] == "remove_size_bias"
    assert caps["default_projection_neighbor_k"] == 50
    assert caps["default_projection_min_dist"] == 0.08
    assert caps["subclass_cluster_sources"] == ["umap_islands", "embedding_kmeans"]
    assert caps["default_subclass_cluster_source"] == "umap_islands"
    assert caps["default_subclass_umap_neighbors"] == 15
    assert caps["default_subclass_umap_min_dist"] == 0.02
    assert caps["default_pca_projection_mode"] == "class_balanced_pca"
    assert caps["pca_projection_modes"] == api.CLASS_ANALYSIS_PCA_PROJECTION_MODES
    assert caps["embedding_aggregation_modes"] == ["pooled"]
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


def test_class_split_experiment_metrics_use_absolute_leakage_and_macro_purity(tmp_path):
    result = {
        "summary": {
            "object_count": 3,
            "raw_object_count": 3,
            "sample_cap": 0,
            "projection": "umap",
            "projection_neighbor_k": 50,
            "neighbor_k": 15,
            "embedding_cache": {"hits": 1, "total": 4},
            "wrong_class_candidate_count": 1,
        },
        "diagnostics": {
            "strongest_size_axis": {"metric": "bbox_area", "correlation": -0.73},
            "axis_correlations": {
                "x": {"bbox_area": -0.73, "crop_area": 0.25},
                "y": {"bbox_area": 0.11},
            },
        },
        "clusters": {"best_k": 2, "candidates": [{"silhouette": 0.31}, {"silhouette": 0.12}]},
        "points": [
            {"class_name": "car", "same_class_neighbor_ratio": 1.0},
            {"class_name": "car", "same_class_neighbor_ratio": 0.8},
            {"class_name": "boat", "same_class_neighbor_ratio": 0.2},
        ],
    }
    run = {
        "analysis_scope": "all_classes",
        "class_name": "",
        "encoder_type": "dinov3",
        "encoder_model": "test",
        "preprocess_mode": "canonical",
        "canonical_size": 336,
        "crop_mode": "padded_square",
        "padding_ratio": 0.08,
        "dinov3_pooling": "pooler",
        "embedding_aggregation": "pooled",
        "background_mode": "full_crop",
        "embedding_view_mode": "tight_context",
        "embedding_adjustment": "remove_size_bias",
        "embedding_postprocess": "none",
    }

    metrics = class_split_experiments._metrics_from_result(
        "precise_tight_context_all_classes",
        run,
        result,
        12.5,
    )

    assert metrics["variant"] == "precise_tight_context"
    assert metrics["embedding_aggregation"] == "pooled"
    assert metrics["strongest_size_axis_correlation"] == -0.73
    assert np.isclose(metrics["strongest_size_axis_abs_correlation"], 0.73)
    assert np.isclose(metrics["mean_abs_size_correlation"], (0.73 + 0.25 + 0.11) / 3)
    assert np.isclose(metrics["mean_neighbor_same_class_ratio"], (1.0 + 0.8 + 0.2) / 3)
    assert np.isclose(metrics["class_balanced_neighbor_same_class_ratio"], ((1.0 + 0.8) / 2 + 0.2) / 2)
    assert metrics["worst_class_neighbor_same_class"] == "boat"
    assert metrics["worst_class_neighbor_same_class_count"] == 1

    class_split_experiments._write_leaderboard(tmp_path, [metrics])
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    leaderboard = (tmp_path / "leaderboard.csv").read_text(encoding="utf-8")
    assert "size_abs=0.730" in report
    assert "class_balanced_nn=0.550" in report
    assert "strongest_size_axis_abs_correlation" in leaderboard
    assert "class_balanced_neighbor_same_class_ratio" in leaderboard

    cradio_runs = class_split_experiments._cradio_matrix(sample_cap=11, classes=["ClassA", "ClassB"])
    assert cradio_runs
    assert all(run["encoder_type"] == "cradio" for run in cradio_runs)
    assert {run["cradio_pooling"] for run in cradio_runs} >= {"summary", "spatial_mean", "summary_spatial_concat"}
    assert all(run["sample_cap"] == 11 for run in cradio_runs)


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

    assert result == {"job_id": "ca_fixed"}
    assert captured["payload"]["workspace_id"] == "ca_fixed"
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
        def __init__(self, *, n_components, n_neighbors, min_dist, metric, random_state):
            captured.update(
                {
                    "n_components": n_components,
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
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
    good = tmp_path / "good.npy"

    np.save(bad_shape, np.zeros((1, 3), dtype=np.float32))
    np.save(bad_nan, np.asarray([1.0, np.nan], dtype=np.float32))
    np.save(good, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))

    assert api._class_analysis_load_cached_embedding(bad_shape) is None
    assert api._class_analysis_load_cached_embedding(bad_nan) is None
    assert np.allclose(api._class_analysis_load_cached_embedding(good), [1.0, 2.0, 3.0])


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
        assert crops[0].size == (64, 64)
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
        assert not cached_thumb.is_symlink()
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

    def fake_cached_valid(_crop_cache_key, head):
        captured_heads.append(dict(head))
        return False

    monkeypatch.setattr(api, "_class_analysis_cached_embedding_valid", fake_cached_valid)

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
