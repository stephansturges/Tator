import base64
import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

import localinferenceapi as api
from api.qwen_caption import build_qwen_caption_router
from localinferenceapi import (
    QwenCaptionHint,
    _build_caption_overlap_guidance,
    _build_qwen_caption_prompt,
    _group_hints_by_window,
    _normalize_caption_hints_for_decoded_image,
    _qwen_messages_with_no_think,
    _qwen_neutralize_thinking_prefill,
    _resolve_qwen_caption_refinement_model_id,
    qwen_caption_prompt_preview,
)
from models.schemas import (
    AutoLabelRequest,
    QwenCaptionPromptPreviewResponse,
    QwenCaptionRequest,
    QwenCaptionResponse,
    QwenPrepassRequest,
)
from services.qwen import (
    _caption_count_conflicts,
    _caption_demote_unstable_glossary_subtypes,
    _caption_degenerate_reason,
    _caption_is_degenerate_impl,
    _caption_has_meta,
    _caption_missing_labels,
    _caption_needs_english_rewrite,
    _caption_needs_completion,
    _caption_missing_exact_counts,
    _caption_needs_refine,
    _caption_repetition_loop_detected,
    _caption_trim_to_complete_sentences,
    _allowed_caption_labels_impl,
    _clean_caption_source_context_text,
    _extract_caption_from_text,
    _format_caption_source_output_context,
    _format_caption_window_observation_lines,
    _format_qwen_load_error_impl,
    _resolve_qwen_caption_decode,
    _resolve_qwen_window_overlap,
    _resolve_qwen_window_size,
    _resolve_caption_all_windows,
    _run_qwen_caption_cleanup,
    _run_qwen_caption_merge,
    _sanitize_qwen_caption,
    _thinking_caption_needs_cleanup,
    _truncate_repeated_caption_loop,
    _window_positions_impl,
)
from utils.glossary import _default_agent_glossary_for_labelmap


class _DecodePayload:
    use_sampling = None
    temperature = None
    top_p = None
    top_k = None
    presence_penalty = None


def _caption_test_image_data_url(width: int = 96, height: int = 96) -> str:
    img = Image.new("RGB", (width, height), (72, 92, 108))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def test_build_qwen_caption_prompt_counts_and_truncation():
    hints = [
        QwenCaptionHint(label="car", bbox=[0, 0, 10, 10], confidence=0.9),
        QwenCaptionHint(label="person", bbox=[20, 20, 30, 30], confidence=0.8),
    ]
    prompt, counts, used, truncated = _build_qwen_caption_prompt(
        "Test prompt",
        hints,
        image_width=100,
        image_height=100,
        include_counts=True,
        include_coords=True,
        max_boxes=1,
        detailed_mode=False,
        restrict_to_labels=True,
    )
    assert "COUNTS (state exactly in final caption)" in prompt
    assert "User caption request: Test prompt" in prompt
    assert "Treat the user caption request as required guidance" in prompt
    assert "Labeled class inventory" in prompt
    assert '"label":"car"' in prompt or '"label":"person"' in prompt
    assert counts.get("car") == 1
    assert counts.get("person") == 1
    assert used == 1
    assert truncated is True


def test_caption_prompt_accepts_dict_and_model_hint_shapes():
    prompt, counts, used, truncated = _build_qwen_caption_prompt(
        "Describe this image.",
        [
            {"label": "car", "bbox": [0, 0, 10, 10], "confidence": 0.8},
            QwenCaptionHint(label="person", bbox=[20, 20, 40, 60], confidence=0.9),
        ],
        image_width=100,
        image_height=100,
        include_counts=True,
        include_coords=True,
        max_boxes=0,
        detailed_mode=False,
        restrict_to_labels=True,
    )

    assert counts == {"car": 1, "person": 1}
    assert used == 2
    assert truncated is False
    assert '"label":"car"' in prompt
    assert '"label":"person"' in prompt
    assert _allowed_caption_labels_impl([QwenCaptionHint(label="car"), {"label": "person"}]) == [
        "car",
        "person",
    ]


def test_caption_prompt_accepts_frontend_context_policy():
    hints = [
        QwenCaptionHint(label="light_vehicle", bbox=[0, 0, 10, 10], confidence=0.9),
    ]

    prompt, counts, used, truncated = _build_qwen_caption_prompt(
        "Test prompt",
        hints,
        image_width=100,
        image_height=100,
        include_counts=True,
        include_coords=False,
        max_boxes=0,
        detailed_mode=False,
        restrict_to_labels=True,
        labelmap_glossary='{"light_vehicle":["small vehicle","delivery vehicle"]}',
        context_prompt="Frontend policy: use the exact prompt stack from the browser.",
    )

    assert "Frontend policy: use the exact prompt stack from the browser." in prompt
    assert "Caption policy:" in prompt
    assert "COUNTS (state exactly in final caption): small vehicle: 1" in prompt
    assert "Treat the user caption request as required guidance" not in prompt
    assert counts == {"light_vehicle": 1}
    assert used == 1
    assert truncated is False


def test_caption_prompt_rejects_malformed_glossary_terms():
    prompt, counts, used, truncated = _build_qwen_caption_prompt(
        "Describe this image.",
        [
            QwenCaptionHint(label="Building", bbox=[0, 0, 40, 40], confidence=0.9),
            QwenCaptionHint(label="Solarpanels", bbox=[50, 0, 90, 25], confidence=0.8),
            QwenCaptionHint(label="UPole", bbox=[10, 50, 20, 95], confidence=0.7),
        ],
        image_width=100,
        image_height=100,
        include_counts=True,
        include_coords=True,
        max_boxes=0,
        detailed_mode=True,
        restrict_to_labels=True,
        labelmap_glossary='Building: [\nSolarpanels: solar panels and rooftop arrays\nUPole: utility pole or antenna',
    )

    assert counts == {"Building": 1, "Solarpanels": 1, "UPole": 1}
    assert used == 3
    assert truncated is False
    assert "broad term \"[\"" not in prompt
    assert "COUNTS (state exactly in final caption): [" not in prompt
    assert '"label":"["' not in prompt
    assert "Building: 1" in prompt
    assert "solar panels: 1" in prompt
    assert "utility pole: 1" in prompt


def test_resolve_caption_all_windows_defaults_to_labeled_windows_only():
    assert (
        _resolve_caption_all_windows(
            "windowed",
            None,
            has_label_hints=True,
            restrict_to_labels=True,
        )
        is False
    )
    assert (
        _resolve_caption_all_windows(
            "windowed",
            None,
            has_label_hints=False,
            restrict_to_labels=True,
        )
        is True
    )
    assert (
        _resolve_caption_all_windows(
            "windowed",
            True,
            has_label_hints=True,
            restrict_to_labels=True,
        )
        is True
    )
    assert (
        _resolve_caption_all_windows(
            "full",
            None,
            has_label_hints=True,
            restrict_to_labels=True,
        )
        is False
    )


def test_caption_count_conflicts_catch_pluralized_singletons_with_glossary_terms():
    glossary = {"PoleFixture": ["utility pole"]}
    caption = (
        "A tall utility pole stands near the center-left. "
        "A few utility poles are visible throughout the neighborhood."
    )

    conflicts = _caption_count_conflicts(caption, {"PoleFixture": 1}, glossary)
    needs_refine, missing = _caption_needs_refine(
        caption,
        {"PoleFixture": 1},
        detailed_mode=True,
        include_counts=True,
        glossary_map=glossary,
    )

    assert conflicts == ["PoleFixture"]
    assert needs_refine is True
    assert missing == ["PoleFixture"]
    assert (
        _caption_count_conflicts(
            "A utility pole stands near the road, with another pole farther back.",
            {"PoleFixture": 2},
            glossary,
        )
        == []
    )


def test_caption_missing_exact_counts_requires_numeric_count_for_multiple_objects():
    glossary = {"SmallVehicle": ["small vehicle", "car", "van", "SUV"]}

    assert _caption_missing_exact_counts(
        "Several small vehicles are parked along the street.",
        {"SmallVehicle": 251},
        glossary,
    ) == ["SmallVehicle"]
    assert _caption_missing_exact_counts(
        "There are 251 small vehicles parked along the street.",
        {"SmallVehicle": 251},
        glossary,
    ) == []
    needs_refine, missing = _caption_needs_refine(
        "Several small vehicles are parked along the street.",
        {"SmallVehicle": 251},
        detailed_mode=False,
        include_counts=True,
        glossary_map=glossary,
    )
    assert needs_refine is True
    assert missing == ["SmallVehicle"]


def test_caption_english_rewrite_ignores_ascii_equivalent_punctuation():
    assert _caption_needs_english_rewrite("A top-down view shows a brown-roofed house.") is False
    assert _caption_needs_english_rewrite("A top\u2011down view shows a brown\u2011roofed house.") is False
    assert _caption_needs_english_rewrite("\u57ce\u5e02\u4e2d\u6709\u4e00\u8f86\u8eca\u3002") is True


def test_caption_demotes_disputed_glossary_subtype_to_broad_term():
    glossary = {
        "SmallVehicle": [
            "small vehicle",
            "light vehicle",
            "car",
            "van",
            "pickup truck",
            "personal pickup truck",
            "delivery vehicle",
        ]
    }
    caption = "A red pickup truck sits near the center of the dirt path."

    cleaned = _caption_demote_unstable_glossary_subtypes(
        caption,
        {"SmallVehicle": 1},
        glossary,
        source_outputs=[
            ("Window 1", "A small vehicle is parked near the buildings."),
            ("Window 2", "A red vehicle sits on the dirt path."),
            ("Window 3", "A red pickup truck sits on the path."),
        ],
    )

    assert "pickup truck" not in cleaned
    assert "red small vehicle" in cleaned


def test_caption_keeps_consistently_supported_glossary_subtype():
    glossary = {
        "SmallVehicle": [
            "small vehicle",
            "light vehicle",
            "car",
            "van",
            "pickup truck",
        ]
    }
    caption = "A red pickup truck sits near the center of the dirt path."

    cleaned = _caption_demote_unstable_glossary_subtypes(
        caption,
        {"SmallVehicle": 1},
        glossary,
        source_outputs=[
            ("Window 1", "A red pickup truck sits on the path."),
            ("Draft", "The pickup truck is parked near the buildings."),
        ],
    )

    assert cleaned == caption


def test_window_caption_hints_include_every_intersecting_crop_with_local_coords():
    source_id = "123e4567-e89b-12d3-a456-426614174000"
    hints = [
        QwenCaptionHint(label="truck", bbox=[90, 10, 130, 50], confidence=0.9, source_id=source_id),
    ]

    grouped = _group_hints_by_window(hints, x_positions=[0, 100], y_positions=[0], window=120)

    assert grouped[(0, 0)][0].bbox == [90.0, 10.0, 120.0, 50.0]
    assert grouped[(100, 0)][0].bbox == [0.0, 10.0, 30.0, 50.0]
    assert grouped[(0, 0)][0].source_id == source_id
    assert grouped[(100, 0)][0].source_id == source_id


def test_caption_hints_scale_to_decoded_image_before_windowing():
    source_id = "123e4567-e89b-12d3-a456-426614174000"
    hints = [
        QwenCaptionHint(label="truck", bbox=[900, 100, 1300, 500], confidence=0.9, source_id=source_id),
    ]

    normalized = _normalize_caption_hints_for_decoded_image(
        hints,
        source_width=2400,
        source_height=1200,
        decoded_width=240,
        decoded_height=120,
    )
    grouped = _group_hints_by_window(normalized, x_positions=[0, 100], y_positions=[0], window=120)

    assert normalized[0].bbox == [90.0, 10.0, 130.0, 50.0]
    assert normalized[0].source_id == source_id
    assert grouped[(0, 0)][0].bbox == [90.0, 10.0, 120.0, 50.0]
    assert grouped[(100, 0)][0].bbox == [0.0, 10.0, 30.0, 50.0]


def test_caption_overlap_guidance_never_exposes_bbox_ids():
    source_id = "123e4567-e89b-12d3-a456-426614174000"
    hints = [
        QwenCaptionHint(label="light_vehicle", bbox=[90, 10, 130, 50], confidence=0.9, source_id=source_id),
    ]
    grouped = _group_hints_by_window(hints, x_positions=[0, 100], y_positions=[0], window=120)

    guidance = _build_caption_overlap_guidance(
        hints,
        grouped,
        image_width=240,
        image_height=120,
        labelmap_glossary="light_vehicle: car or van",
    )

    assert "Overlap deduplication guidance" in guidance
    assert "car" in guidance
    assert source_id not in guidance
    assert "object ID" not in guidance
    assert "bbox ID" not in guidance


def test_window_observation_lines_ground_local_spatial_language_globally():
    lines = _format_caption_window_observation_lines(
        [(0, 0, 100, "A small object appears in the bottom-right corner of the crop.")],
        image_width=400,
        image_height=400,
    )
    text = "\n".join(lines)

    assert "Spatial grounding" in text
    assert "crop-relative" in text
    assert "global upper-left section" in text
    assert "covers x 0-25% and y 0-25%" in text
    assert "do not copy local spatial wording" in text
    assert "bottom-right corner of the crop" in text


def test_window_observation_lines_describe_overlapping_quadrants():
    lines = _format_caption_window_observation_lines(
        [
            (0, 0, 672, "Window one."),
            (328, 0, 672, "Window two."),
            (0, 328, 672, "Window three."),
            (328, 328, 672, "Window four."),
        ],
        image_width=1000,
        image_height=1000,
    )
    text = "\n".join(lines)

    assert "Window 1 (first-row, first-column global upper-left section" in text
    assert "Window 2 (first-row, last-column global upper-right section" in text
    assert "Window 3 (last-row, first-column global lower-left section" in text
    assert "Window 4 (last-row, last-column global lower-right section" in text
    assert "global region middle-center" not in text


def test_window_observation_lines_reconcile_window_hints_to_full_frame_objects():
    full_hints = [
        QwenCaptionHint(
            label="light_vehicle",
            bbox=[20, 20, 90, 90],
            confidence=0.9,
            source_id="full-car-1",
        ),
        QwenCaptionHint(label="person", bbox=[240, 240, 280, 320], confidence=0.8, source_id="person-1"),
    ]
    window_hints = {
        (0, 0): [
            QwenCaptionHint(
                label="light_vehicle",
                bbox=[20, 20, 90, 90],
                confidence=0.9,
                source_id="full-car-1",
            ),
            QwenCaptionHint(label="bike", bbox=[120, 120, 150, 150], confidence=0.6),
        ]
    }

    lines = _format_caption_window_observation_lines(
        [(0, 0, 200, "A close view shows a small vehicle near pavement.")],
        image_width=400,
        image_height=400,
        full_label_hints=full_hints,
        window_hints_by_window=window_hints,
        glossary_map={"light_vehicle": ["small vehicle"]},
    )
    text = "\n".join(lines)

    assert "Object reconciliation" in text
    assert "object_001 small vehicle" in text
    assert "same source id" in text
    assert "Window-only evidence not matched to the full-frame inventory" in text
    assert "bike" in text
    assert "never output them" in text


def test_window_observation_lines_reconcile_without_source_ids_by_geometry():
    full_hints = [
        QwenCaptionHint(label="car", bbox=[40, 40, 120, 120], confidence=0.9),
    ]
    window_hints = {
        (0, 0): [
            QwenCaptionHint(label="car", bbox=[40, 40, 100, 100], confidence=0.9),
        ]
    }

    lines = _format_caption_window_observation_lines(
        [(0, 0, 100, "The crop shows part of the car.")],
        image_width=200,
        image_height=200,
        full_label_hints=full_hints,
        window_hints_by_window=window_hints,
    )
    text = "\n".join(lines)

    assert "object_001 car" in text
    assert "window coverage" in text
    assert "Window-only evidence not matched" not in text


def test_caption_prompt_never_includes_bbox_source_ids():
    source_id = "123e4567-e89b-12d3-a456-426614174000"
    prompt, _, _, _ = _build_qwen_caption_prompt(
        "Describe this image.",
        [QwenCaptionHint(label="car", bbox=[0, 0, 10, 10], confidence=0.9, source_id=source_id)],
        image_width=100,
        image_height=100,
        include_counts=True,
        include_coords=True,
        max_boxes=0,
        detailed_mode=True,
    )

    assert source_id not in prompt


def test_caption_prompt_uses_glossary_as_semantic_class_meaning():
    glossary = '{"light_vehicle":["small vehicle","car","van","pickup truck"],"gas_tank":["storage tank","silo"]}'
    prompt, counts, used, truncated = _build_qwen_caption_prompt(
        "Describe this image.",
        [
            QwenCaptionHint(label="light_vehicle", bbox=[0, 0, 10, 10], confidence=0.9),
            QwenCaptionHint(label="gas_tank", bbox=[20, 20, 40, 45], confidence=0.8),
        ],
        image_width=100,
        image_height=100,
        include_counts=True,
        include_coords=True,
        max_boxes=0,
        detailed_mode=True,
        labelmap_glossary=glossary,
    )

    assert counts == {"light_vehicle": 1, "gas_tank": 1}
    assert used == 2
    assert truncated is False
    assert "Class meaning glossary" in prompt
    assert 'light_vehicle: broad term "small vehicle"' in prompt
    assert "possible variants include car, van, pickup truck" in prompt
    assert 'gas_tank: broad term "storage tank"' in prompt
    assert "Glossary variants are possible members of a class, not assertions" in prompt
    assert "Do not choose a subtype from the glossary unless the image clearly supports that subtype" in prompt
    assert "COUNTS (state exactly in final caption): small vehicle: 1, storage tank: 1" in prompt
    assert '"label":"small vehicle"' in prompt
    assert '"label":"storage tank"' in prompt


def test_default_caption_glossary_matches_camelcase_labelmap_names():
    glossary = _default_agent_glossary_for_labelmap(["SmallVehicle", "PoleFixture", "StorageTank"])

    assert '"SmallVehicle": [\n    "Small Vehicle"' in glossary
    assert '"PoleFixture": [\n    "Pole Fixture"' in glossary
    assert '"StorageTank": [\n    "Storage Tank"' in glossary


def test_window_caption_prompt_applies_max_boxes_after_window_clipping():
    hints = [
        QwenCaptionHint(label="truck", bbox=[90, 10, 130, 50], confidence=0.9),
        QwenCaptionHint(label="tree", bbox=[105, 20, 115, 40], confidence=0.8),
    ]
    grouped = _group_hints_by_window(hints, x_positions=[100], y_positions=[0], window=120)

    prompt, counts, used, truncated = _build_qwen_caption_prompt(
        "Describe this crop.",
        grouped[(100, 0)],
        image_width=120,
        image_height=120,
        include_counts=True,
        include_coords=True,
        max_boxes=1,
        detailed_mode=True,
        restrict_to_labels=True,
    )

    assert counts == {"truck": 1, "tree": 1}
    assert used == 1
    assert truncated is True
    assert '"label":"truck"' in prompt
    assert '"bbox_2d":[0,83,250,417]' in prompt
    assert '"label":"tree"' not in prompt


def test_short_caption_request_removes_conflicting_detail_instructions():
    prompt, _, _, _ = _build_qwen_caption_prompt(
        "Write a short caption (1-2 sentences) describing the scene and main objects.",
        [],
        image_width=3200,
        image_height=1800,
        include_counts=False,
        include_coords=False,
        max_boxes=0,
        detailed_mode=True,
        max_sentences=10,
    )

    assert "Write a short caption (1-2 sentences)" in prompt
    assert "Use the image as truth" in prompt
    assert "longer captions are acceptable" not in prompt
    assert "Be maximally descriptive" not in prompt


def test_sanitize_qwen_caption_removes_repeated_sentence_tail():
    repeated = (
        "An overhead view shows a dry site with a white path and several buildings. "
        "There are no visible signs of vegetation or large trees in the area, and the ground appears mostly bare and dry. "
        "There are no visible signs of vegetation or large trees in the area, and the ground appears mostly bare and dry. "
        "There are no visible signs of vegetation or large trees in the area, and the ground appears"
    )

    cleaned = _sanitize_qwen_caption(repeated)

    assert cleaned.count("There are no visible signs") == 1
    assert not cleaned.endswith("and the ground appears")


def test_sanitize_qwen_caption_removes_uuid_like_bbox_ids():
    raw = "A car is parked near the road. 123e4567-e89b-12d3-a456-426614174000"

    cleaned = _sanitize_qwen_caption(raw)

    assert cleaned == "A car is parked near the road."


def test_caption_loop_detection_catches_meta_planning_repetition():
    repeated = (
        "We can mention the street. We can mention the buildings' windows. "
        "We can mention the street. We can mention the buildings' windows. "
        "We can mention the street. We can mention the buildings' windows."
    )

    assert _caption_repetition_loop_detected(repeated) is True
    assert _caption_has_meta(_sanitize_qwen_caption(repeated)) is True


def test_caption_loop_detection_catches_numeric_character_run():
    repeated = "9980217" + ("8" * 120)

    assert _caption_repetition_loop_detected(repeated) is True
    assert _sanitize_qwen_caption(repeated) == "99802178"
    assert _caption_is_degenerate_impl(repeated) is True


def test_caption_loop_detection_catches_punctuation_run():
    repeated = "!" * 160

    assert _caption_repetition_loop_detected(repeated) is True
    assert _caption_degenerate_reason(repeated, allow_short_caption=True) == "punctuation_loop"
    assert _caption_is_degenerate_impl(repeated) is True
    assert _truncate_repeated_caption_loop(repeated) == "!"


def test_truncate_repeated_caption_loop_keeps_first_cycle_only():
    repeated = (
        "A street is lined with buildings. Windows face the road. "
        "A street is lined with buildings. Windows face the road. "
        "A street is lined with buildings. Windows face the road."
    )

    cleaned = _truncate_repeated_caption_loop(repeated)

    assert cleaned == "A street is lined with buildings. Windows face the road."


def test_source_output_context_strips_meta_loop_text():
    context = _format_caption_source_output_context(
        source_outputs=[
            (
                "Window raw output",
                "A street is lined with buildings and visible windows. "
                "We can mention the street. We can mention the buildings' windows. "
                "We can mention the street. We can mention the buildings' windows. "
                "We can mention the street. We can mention the buildings' windows.",
            )
        ]
    )

    assert "A street is lined with buildings" in context
    assert "We can mention" not in context


def test_source_output_context_drops_user_request_and_coordinate_meta():
    raw = (
        'The user request says: "Write a short caption (1-2 sentences)." '
        "The region of interest is from [1210, 0] to [1882, 672]. "
        "A street is lined with brick buildings and parked cars."
    )

    cleaned = _clean_caption_source_context_text(raw)
    context = _format_caption_source_output_context(source_outputs=[("Window raw output", raw)])

    assert cleaned == "A street is lined with brick buildings and parked cars."
    assert "user request" not in context.lower()
    assert "[1210" not in context
    assert "brick buildings and parked cars" in context


def test_valid_one_sentence_caption_is_not_degenerate():
    caption = "A top-down view shows a city block with streets, buildings, and parked cars."

    assert _caption_is_degenerate_impl(caption) is False


def test_caption_final_guard_removes_incomplete_trailing_sentence():
    raw = (
        "The image shows a domed landmark surrounded by dense city blocks. "
        "A waterfront and several roads frame the scene. "
        "An overhead view of"
    )

    cleaned, changed = _caption_trim_to_complete_sentences(raw, max_sentences=10)

    assert changed is True
    assert cleaned == (
        "The image shows a domed landmark surrounded by dense city blocks. "
        "A waterfront and several roads frame the scene."
    )
    assert _caption_needs_completion(raw) is True
    assert _caption_needs_completion(cleaned) is False


def test_caption_final_guard_respects_sentence_budget():
    raw = "One sentence. Two sentence. Three sentence."

    cleaned, changed = _caption_trim_to_complete_sentences(raw, max_sentences=2)

    assert changed is True
    assert cleaned == "One sentence. Two sentence."


def test_thinking_reasoning_leak_is_detected_as_meta_caption():
    raw = (
        "We need to produce a final caption. The user wants a single complete sentence. "
        'The draft caption is: "A tank burns in a muddy field." '
        "So we need to keep it grounded in the image."
    )

    cleaned = _sanitize_qwen_caption(raw)

    assert _thinking_caption_needs_cleanup(cleaned, raw) is True
    assert _caption_has_meta(cleaned) is True


def test_extract_caption_prefers_final_xml_marker():
    cleaned, marker_found = _extract_caption_from_text(
        "<think>reasoning</think><final>A tank burns in a muddy field.</final>"
    )

    assert marker_found is True
    assert cleaned == "A tank burns in a muddy field."


def test_extract_caption_marker_does_not_strip_plain_final_word():
    raw = "The final image shows a car parked near a road."

    cleaned, marker_found = _extract_caption_from_text(raw, marker="FINAL")

    assert marker_found is False
    assert cleaned == raw
    assert _sanitize_qwen_caption(raw) == raw
    assert _extract_caption_from_text("FINAL: A car is parked.", marker="FINAL") == (
        "A car is parked.",
        True,
    )


def test_caption_missing_labels_uses_word_boundaries():
    assert _caption_missing_labels("A cargo yard is visible.", {"car": 1}) == ["car"]
    assert _caption_missing_labels("A car is parked near a road.", {"car": 1}) == []


def test_caption_completion_allows_terminal_quote_after_punctuation():
    assert _caption_needs_completion('A road sign reads "STOP."') is False


def test_caption_decode_defaults_include_repeat_controls():
    decode = _resolve_qwen_caption_decode(_DecodePayload(), is_thinking=False)

    assert decode["do_sample"] is True
    assert decode["temperature"] == 0.7
    assert decode["top_p"] == 0.8
    assert decode["top_k"] == 20
    assert decode["repetition_penalty"] > 1.0
    assert decode["repetition_context_size"] >= 64
    assert decode["no_repeat_ngram_size"] >= 4


def test_caption_decode_uses_qwen_thinking_sampling_defaults():
    decode = _resolve_qwen_caption_decode(_DecodePayload(), is_thinking=True)

    assert decode["do_sample"] is True
    assert decode["temperature"] == 0.6
    assert decode["top_p"] == 0.95
    assert decode["top_k"] == 20
    assert decode["presence_penalty"] == 0.0
    assert decode["repetition_penalty"] == 1.0


def test_caption_decode_avoids_greedy_for_thinking_models():
    class Payload(_DecodePayload):
        use_sampling = False

    decode = _resolve_qwen_caption_decode(Payload(), is_thinking=True)

    assert decode["do_sample"] is True
    assert decode["temperature"] == 0.6
    assert decode["top_p"] == 0.95
    assert decode["top_k"] == 20


def test_caption_decode_clamps_invalid_sampling_parameters():
    class Payload(_DecodePayload):
        temperature = float("nan")
        top_p = 9
        top_k = -3
        presence_penalty = 99

    decode = _resolve_qwen_caption_decode(Payload(), is_thinking=False)

    assert decode["temperature"] == 0.7
    assert decode["top_p"] == 1.0
    assert decode["top_k"] == 1
    assert decode["presence_penalty"] == 2.0


def test_caption_decode_defaults_nonfinite_integer_sampling_parameters():
    class Payload(_DecodePayload):
        top_k = float("inf")

    decode = _resolve_qwen_caption_decode(Payload(), is_thinking=False)

    assert decode["top_k"] == 20


def test_caption_request_drops_nonfinite_numeric_controls():
    payload = QwenCaptionRequest(
        image_base64="data:image/png;base64,AA==",
        temperature=float("nan"),
        top_p=float("inf"),
        presence_penalty=float("-inf"),
        window_overlap=float("nan"),
        window_size=float("inf"),
        image_width=float("nan"),
        image_height=float("inf"),
        max_new_tokens=float("inf"),
    )

    assert payload.temperature is None
    assert payload.top_p is None
    assert payload.presence_penalty is None
    assert payload.window_overlap is None
    assert payload.window_size is None
    assert payload.image_width is None
    assert payload.image_height is None
    assert payload.max_new_tokens == 1000


def test_qwen_caption_window_helpers_fallback_for_nonfinite_values():
    assert _resolve_qwen_window_overlap(float("nan"), default_overlap=0.1) == 0.1
    assert _resolve_qwen_window_overlap(float("inf"), default_overlap=0.1) == 0.1

    window = _resolve_qwen_window_size(
        float("nan"),
        image_width=1600,
        image_height=900,
        overlap=float("nan"),
        default_size=672,
        default_overlap=0.1,
    )
    positions = _window_positions_impl(1600, window, float("nan"))

    assert window == 672
    assert positions[0] == 0
    assert positions[-1] == 1600 - window


def test_qwen_caption_window_grid_covers_wide_image_not_only_corners():
    window = _resolve_qwen_window_size(
        None,
        image_width=1920,
        image_height=1080,
        overlap=0.2,
        default_size=672,
        default_overlap=0.2,
    )
    x_positions = _window_positions_impl(1920, window, 0.2, force_two=True)
    y_positions = _window_positions_impl(1080, window, 0.2, force_two=True)

    assert window == 600
    assert x_positions == [0, 480, 960, 1320]
    assert y_positions == [0, 480]
    assert len(x_positions) * len(y_positions) == 8


def test_qwen_caption_window_grid_honors_smaller_requested_window_size():
    window = _resolve_qwen_window_size(
        384,
        image_width=1600,
        image_height=900,
        overlap=0.1,
        default_size=672,
        default_overlap=0.2,
    )
    x_positions = _window_positions_impl(1600, window, 0.1)
    y_positions = _window_positions_impl(900, window, 0.1)

    assert window == 384
    assert len(x_positions) > 2
    assert len(y_positions) > 2
    assert x_positions[0] == 0
    assert y_positions[0] == 0
    assert x_positions[-1] == 1600 - window
    assert y_positions[-1] == 900 - window
    assert all((x_positions[idx + 1] - x_positions[idx]) <= window for idx in range(len(x_positions) - 1))
    assert all((y_positions[idx + 1] - y_positions[idx]) <= window for idx in range(len(y_positions) - 1))


def test_qwen_caption_window_size_never_exceeds_tiny_image():
    window = _resolve_qwen_window_size(
        None,
        image_width=48,
        image_height=32,
        overlap=0.2,
        default_size=672,
        default_overlap=0.2,
    )
    x_positions = _window_positions_impl(48, window, 0.2)
    y_positions = _window_positions_impl(32, window, 0.2)

    assert window == 32
    assert x_positions == [0, 16]
    assert y_positions == [0]
    assert all(x + window <= 48 for x in x_positions)
    assert all(y + window <= 32 for y in y_positions)


def test_format_qwen_load_error_summarizes_missing_vision_tower():
    detail = _format_qwen_load_error_impl(
        RuntimeError(
            "Missing 393 parameters: vision_tower.blocks.0.attn.proj.bias, "
            "vision_tower.blocks.0.attn.proj.weight"
        )
    )

    assert detail.startswith("incompatible_checkpoint_missing_vision_tower")
    assert "vision tower weights" in detail
    assert "vision_tower.blocks.0.attn.proj.weight" not in detail


def test_caption_request_accepts_custom_system_prompt():
    payload = QwenCaptionRequest(
        image_base64="data:image/png;base64,AA==",
        caption_system_prompt="  Stop after a single paragraph.  ",
        caption_detection_context_prompt="  Detection policy.  ",
        caption_window_prompt="  Window policy.  ",
        caption_draft_refine_prompt="  Draft policy.  ",
        caption_merge_prompt="  Merge policy.  ",
        caption_cleanup_prompt="  Cleanup policy.  ",
    )

    assert payload.caption_system_prompt == "Stop after a single paragraph."
    assert payload.caption_detection_context_prompt == "Detection policy."
    assert payload.caption_window_prompt == "Window policy."
    assert payload.caption_draft_refine_prompt == "Draft policy."
    assert payload.caption_merge_prompt == "Merge policy."
    assert payload.caption_cleanup_prompt == "Cleanup policy."


def test_qwen_caption_prompt_preview_uses_caption_stack_without_loading_model(monkeypatch):
    monkeypatch.setattr(
        api,
        "_ensure_qwen_ready_for_caption",
        lambda *_args, **_kwargs: pytest.fail("prompt preview must not load a Qwen runtime"),
    )
    payload = QwenCaptionRequest(
        image_base64=_caption_test_image_data_url(),
        image_name="preview_image.png",
        image_width=96,
        image_height=96,
        user_prompt="Use the opening phrase: A top-down view shows",
        label_hints=[
            QwenCaptionHint(label="Boat", bbox=[8, 8, 44, 44], confidence=0.98),
            QwenCaptionHint(label="Person", bbox=[52, 52, 80, 86], confidence=0.91),
        ],
        include_counts=True,
        include_coords=True,
        max_boxes=10,
        model_id="Qwen/Qwen3-VL-4B-Thinking",
        model_variant="Thinking",
        two_stage_refine=True,
        final_answer_only=True,
        final_caption_max_sentences=2,
        caption_mode="windowed",
        window_size=64,
        caption_system_prompt="CUSTOM SYSTEM PROMPT",
        caption_detection_context_prompt="CUSTOM DETECTION CONTEXT PROMPT",
        caption_window_prompt="CUSTOM WINDOW PROMPT",
        caption_draft_refine_prompt="CUSTOM DRAFT REFINE PROMPT",
        caption_merge_prompt="CUSTOM MERGE PROMPT",
        caption_cleanup_prompt="CUSTOM CLEANUP PROMPT",
        labelmap_glossary="Boat: watercraft and vessels\nPerson: visible humans",
    )

    preview = qwen_caption_prompt_preview(payload)

    assert preview.used_counts == {"Boat": 1, "Person": 1}
    assert preview.used_boxes == 2
    assert preview.meta["planned_windows"] >= 1
    assert "Complete Qwen caption prompt flow preview" in preview.full_text
    assert "CUSTOM SYSTEM PROMPT" in preview.full_text
    assert "CUSTOM DETECTION CONTEXT PROMPT" in preview.full_text
    assert "CUSTOM WINDOW PROMPT" in preview.full_text
    assert "CUSTOM DRAFT REFINE PROMPT" in preview.full_text
    assert "CUSTOM MERGE PROMPT" in preview.full_text
    assert "CUSTOM CLEANUP PROMPT" in preview.full_text
    assert "Two-stage draft prompt" in preview.full_text
    assert "Two-stage refinement prompt template" in preview.full_text
    assert "Window merge prompt template" in preview.full_text
    assert "Conditional cleanup / guard prompt template" in preview.full_text
    assert "Only mention these classes if they appear:" in preview.full_text
    assert "watercraft" in preview.full_text
    assert "vessels" in preview.full_text
    assert "visible humans" in preview.full_text
    assert "Edit the draft with minimal changes. Do not introduce new objects or actions." in preview.full_text
    assert "First-stage model output context:" in preview.full_text
    assert "<generated caption from window" in preview.full_text
    assert "<generated draft caption>" in preview.full_text
    assert any(section.chat_messages for section in preview.sections)
    assert any(section.kind == "template" for section in preview.sections)


def test_qwen_caption_prompt_preview_hardens_windowed_evidence_contract(monkeypatch):
    monkeypatch.setattr(
        api,
        "_ensure_qwen_ready_for_caption",
        lambda *_args, **_kwargs: pytest.fail("prompt preview must not load a Qwen runtime"),
    )
    payload = QwenCaptionRequest(
        image_base64=_caption_test_image_data_url(width=100, height=100),
        image_name="caption_contract.png",
        image_width=100,
        image_height=100,
        user_prompt="Write a detailed caption describing the scene.",
        label_hints=[
            QwenCaptionHint(label="Building", bbox=[0, 0, 40, 40], confidence=0.9),
            QwenCaptionHint(label="Solarpanels", bbox=[50, 0, 90, 25], confidence=0.8),
            QwenCaptionHint(label="UPole", bbox=[10, 50, 20, 95], confidence=0.7),
        ],
        include_counts=True,
        include_coords=True,
        max_boxes=0,
        caption_mode="windowed",
        caption_all_windows=True,
        window_size=64,
        window_overlap=0.1,
        caption_window_min_sentences=2,
        caption_window_max_sentences=4,
        labelmap_glossary='Building: [\nSolarpanels: solar panels and rooftop arrays\nUPole: utility pole or antenna',
    )

    preview = qwen_caption_prompt_preview(payload)

    assert "Write 2-4 concrete sentences about this region only" in preview.full_text
    assert "Object reconciliation" in preview.full_text
    assert "Full-frame counts remain authoritative" in preview.full_text
    assert "Matched full-frame objects in this window" in preview.full_text
    assert "broad term \"[\"" not in preview.full_text
    assert "COUNTS (state exactly in final caption): [" not in preview.full_text
    assert '"label":"["' not in preview.full_text
    assert "Allowed classes: [" not in preview.full_text
    assert "if couts" not in preview.full_text
    assert "solar panels" in preview.full_text
    assert "utility pole" in preview.full_text
    assert preview.full_text.count("Overlap deduplication guidance") <= 2
    assert "global region middle-center" not in preview.full_text
    assert "Use the class meaning glossary above" in preview.full_text
    assert "Conditional cleanup / guard prompt template" in preview.full_text
    assert "Authoritative object counts" in preview.full_text


def test_qwen_caption_router_exposes_prompt_preview_endpoint():
    seen = {}

    def caption_fn(_payload):
        return QwenCaptionResponse(caption="caption", used_boxes=0, truncated=False)

    def preview_fn(payload):
        seen["payload"] = payload
        return QwenCaptionPromptPreviewResponse(full_text="preview text")

    app = FastAPI()
    app.include_router(
        build_qwen_caption_router(
            caption_fn=caption_fn,
            preview_fn=preview_fn,
            request_cls=QwenCaptionRequest,
            response_cls=QwenCaptionResponse,
            preview_response_cls=QwenCaptionPromptPreviewResponse,
        )
    )
    response = TestClient(app).post(
        "/qwen/caption/preview_prompt",
        json={
            "image_base64": _caption_test_image_data_url(),
            "image_name": "route_preview.png",
            "user_prompt": "Keep it concise.",
        },
    )

    assert response.status_code == 200
    assert response.json()["full_text"] == "preview text"
    assert isinstance(seen["payload"], QwenCaptionRequest)
    assert seen["payload"].image_name == "route_preview.png"
    assert seen["payload"].user_prompt == "Keep it concise."


def test_caption_request_normalizes_legacy_caption_mode_and_variant_values():
    hybrid_payload = QwenCaptionRequest(
        image_base64="data:image/png;base64,AA==",
        caption_mode="hybrid",
        model_variant="thinking",
    )
    invalid_payload = QwenCaptionRequest(
        image_base64="data:image/png;base64,AA==",
        caption_mode="unexpected",
        model_variant="unexpected",
    )

    assert hybrid_payload.caption_mode == "windowed"
    assert hybrid_payload.model_variant == "Thinking"
    assert invalid_payload.caption_mode == "full"
    assert invalid_payload.model_variant == "auto"


def test_prepass_request_normalizes_caption_variant_values():
    payload = QwenPrepassRequest(
        image_token=" token-1 ",
        image_name=" image.jpg ",
        model_variant="thinking",
        prepass_caption_variant="instruct",
        prepass_caption_model_id=" Qwen/Qwen3-VL-4B-Instruct ",
    )

    assert payload.image_token == "token-1"
    assert payload.image_name == "image.jpg"
    assert payload.model_variant == "Thinking"
    assert payload.prepass_caption_variant == "Instruct"
    assert payload.prepass_caption_model_id == "Qwen/Qwen3-VL-4B-Instruct"


def test_auto_label_request_normalizes_caption_planner_variant_values():
    payload = AutoLabelRequest(
        dataset_id="dataset-1",
        planner_model_variant="thinking",
        planner_model_id=" Qwen/Qwen3-VL-4B-Thinking ",
        yolo_id=" yolo-run ",
    )
    invalid_payload = AutoLabelRequest(
        dataset_id="dataset-1",
        planner_model_variant="unexpected",
    )

    assert payload.planner_model_variant == "Thinking"
    assert payload.planner_model_id == "Qwen/Qwen3-VL-4B-Thinking"
    assert payload.yolo_id == "yolo-run"
    assert invalid_payload.planner_model_variant == "auto"


def test_caption_hint_rejects_nonfinite_bbox_values():
    with pytest.raises(ValidationError):
        QwenCaptionHint(label="car", bbox=[float("nan"), 0, 10, 10])
    with pytest.raises(ValidationError):
        QwenCaptionHint(label="car", bbox=[0, 0, float("inf"), 10])


def test_caption_request_normalizes_refinement_model_id():
    same_payload = QwenCaptionRequest(
        image_base64="data:image/png;base64,AA==",
        refinement_model_id=" same ",
    )
    auto_payload = QwenCaptionRequest(
        image_base64="data:image/png;base64,AA==",
        refinement_model_id=" auto ",
    )
    explicit_payload = QwenCaptionRequest(
        image_base64="data:image/png;base64,AA==",
        refinement_model_id=" mlx-community/Qwen3-VL-4B-Instruct-4bit ",
    )

    assert same_payload.refinement_model_id == "same"
    assert auto_payload.refinement_model_id == "auto"
    assert explicit_payload.refinement_model_id == "mlx-community/Qwen3-VL-4B-Instruct-4bit"


def test_caption_request_normalizes_final_caption_max_sentences():
    default_payload = QwenCaptionRequest(
        image_base64="data:image/png;base64,AA==",
    )
    long_payload = QwenCaptionRequest(
        image_base64="data:image/png;base64,AA==",
        final_caption_max_sentences=40,
    )
    zero_payload = QwenCaptionRequest(
        image_base64="data:image/png;base64,AA==",
        final_caption_max_sentences=0,
    )

    assert default_payload.max_new_tokens == 1000
    assert default_payload.final_caption_max_sentences == 10
    assert default_payload.caption_all_windows is None
    assert long_payload.final_caption_max_sentences == 30
    assert zero_payload.final_caption_max_sentences == 10


def test_caption_request_allows_high_cap_caption_tokens():
    payload = QwenCaptionRequest(
        image_base64="data:image/png;base64,AA==",
        max_new_tokens=9000,
    )

    assert payload.max_new_tokens == 4096


def test_caption_request_normalizes_window_sentence_range():
    default_payload = QwenCaptionRequest(
        image_base64="data:image/png;base64,AA==",
    )
    custom_payload = QwenCaptionRequest(
        image_base64="data:image/png;base64,AA==",
        caption_window_min_sentences=2,
        caption_window_max_sentences=4,
    )
    inverted_payload = QwenCaptionRequest(
        image_base64="data:image/png;base64,AA==",
        caption_window_min_sentences=5,
        caption_window_max_sentences=2,
    )
    long_payload = QwenCaptionRequest(
        image_base64="data:image/png;base64,AA==",
        caption_window_min_sentences=40,
        caption_window_max_sentences=50,
    )

    assert default_payload.caption_window_min_sentences == 1
    assert default_payload.caption_window_max_sentences == 3
    assert custom_payload.caption_window_min_sentences == 2
    assert custom_payload.caption_window_max_sentences == 4
    assert inverted_payload.caption_window_min_sentences == 5
    assert inverted_payload.caption_window_max_sentences == 5
    assert long_payload.caption_window_min_sentences == 10
    assert long_payload.caption_window_max_sentences == 10


def test_resolve_qwen_caption_refinement_model_id_defaults_to_instruct_for_thinking_model():
    assert (
        _resolve_qwen_caption_refinement_model_id(
            None,
            desired_model_id="nightmedia/Huihui-Qwen3-VL-30B-A3B-Thinking-abliterated-qx86-hi-mlx",
            active_model_id="Qwen/Qwen3-VL-4B-Instruct",
        )
        == "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    )
    assert (
        _resolve_qwen_caption_refinement_model_id(
            "auto",
            desired_model_id="nightmedia/Huihui-Qwen3-VL-30B-A3B-Thinking-abliterated-qx86-hi-mlx",
            active_model_id="Qwen/Qwen3-VL-4B-Instruct",
        )
        == "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    )
    assert (
        _resolve_qwen_caption_refinement_model_id(
            "same",
            desired_model_id="nightmedia/Huihui-Qwen3-VL-30B-A3B-Thinking-abliterated-qx86-hi-mlx",
            active_model_id="Qwen/Qwen3-VL-4B-Instruct",
        )
        == "nightmedia/Huihui-Qwen3-VL-30B-A3B-Thinking-abliterated-qx86-hi-mlx"
    )
    assert (
        _resolve_qwen_caption_refinement_model_id(
            "active",
            desired_model_id="Qwen/Qwen3-VL-30B-A3B-Thinking",
            active_model_id="Qwen/Qwen3-VL-4B-Instruct",
        )
        == "Qwen/Qwen3-VL-4B-Instruct"
    )
    assert (
        _resolve_qwen_caption_refinement_model_id(
            "mlx-community/Qwen3-VL-4B-Instruct-4bit",
            desired_model_id="Qwen/Qwen3-VL-30B-A3B-Thinking",
            active_model_id="Qwen/Qwen3-VL-4B-Instruct",
        )
        == "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    )


def test_no_think_is_added_to_last_user_message_and_prefill_is_neutralized():
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Return final only."}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": object()},
                {"type": "text", "text": "Caption this image."},
            ],
        },
    ]

    updated = _qwen_messages_with_no_think(messages)
    rendered = _qwen_neutralize_thinking_prefill(
        "<|im_start|>assistant\n<think>\n",
        disable_thinking=True,
        model_id="Qwen/Qwen3-VL-30B-A3B-Thinking",
    )

    assert updated[1]["content"][1]["text"].endswith("/no_think")
    assert messages[1]["content"][1]["text"] == "Caption this image."
    assert rendered.endswith("<think>\n\n</think>\n\n")


def test_caption_merge_uses_refinement_model_override():
    calls = {}

    class ImageStub:
        width = 512
        height = 512

    def runtime_resolver(model_id):
        calls["model_id"] = model_id
        return ("runtime", None)

    def run_qwen_inference_fn(*_args, **kwargs):
        calls["runtime_override"] = kwargs.get("runtime_override")
        calls["chat_template_kwargs"] = kwargs.get("chat_template_kwargs")
        calls["prompt"] = _args[0]
        return "Merged caption.", None, None

    merged = _run_qwen_caption_merge(
        "Draft caption.",
        [(0, 0, 128, "Window detail in the bottom-right of the crop.")],
        pil_img=ImageStub(),
        base_model_id="Qwen/Qwen3-VL-30B-A3B-Thinking",
        runtime_resolver=runtime_resolver,
        max_new_tokens=64,
        model_id_override="mlx-community/Qwen3-VL-4B-Instruct-4bit",
        max_sentences=10,
        user_prompt="Can we infer the likely location of this scene?",
        overlap_guidance=(
            "Overlap deduplication guidance: repeated crop-edge descriptions of a car "
            "near the center should be merged into one object."
        ),
        merge_prompt_override=(
            "Frontend merge policy: preserve all window evidence. Use up to 10 complete sentences. "
            "preserve broad category terms; do not replace broad class terms with narrower subtypes."
        ),
        source_outputs=[("Full-image raw output", "Smoke rises near scattered wreckage.")],
        authoritative_counts_note=(
            "Authoritative object counts that must appear in the caption as natural scene facts: "
            "small vehicle: 251."
        ),
        chat_template_kwargs={"enable_thinking": False},
        run_qwen_inference_fn=run_qwen_inference_fn,
        resolve_variant_fn=lambda _base, _variant: "Qwen/Qwen3-VL-30B-A3B-Instruct",
        extract_caption_fn=lambda text, marker=None: (text, False),
        sanitize_caption_fn=lambda text: text.strip(),
    )

    assert merged == "Merged caption."
    assert calls["model_id"] == "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    assert calls["runtime_override"] == ("runtime", None)
    assert calls["chat_template_kwargs"] == {"enable_thinking": False}
    assert "up to 10 complete sentences" in calls["prompt"]
    assert "User caption request: Can we infer the likely location of this scene?" in calls["prompt"]
    assert "repeated crop-edge descriptions of a car" in calls["prompt"]
    assert "Spatial grounding" in calls["prompt"]
    assert "crop-relative" in calls["prompt"]
    assert "global upper-left section" in calls["prompt"]
    assert "bottom-right of the crop" in calls["prompt"]
    assert "Smoke rises near scattered wreckage." in calls["prompt"]
    assert "small vehicle: 251" in calls["prompt"]
    assert "Frontend merge policy: preserve all window evidence." in calls["prompt"]
    assert "preserve broad category terms" in calls["prompt"]
    assert "do not replace broad class terms with narrower subtypes" in calls["prompt"]


def test_caption_merge_rejects_degenerate_editor_output():
    def run_qwen_inference_fn(*_args, **_kwargs):
        return "!" * 160, None, None

    merged = _run_qwen_caption_merge(
        "Draft caption with useful details.",
        [(0, 0, 128, "Window detail in the bottom-right of the crop.")],
        pil_img=type("ImageStub", (), {"width": 512, "height": 512})(),
        base_model_id="Qwen/Qwen3-VL-4B-Instruct",
        runtime_resolver=lambda _model_id: ("runtime", None),
        max_new_tokens=64,
        run_qwen_inference_fn=run_qwen_inference_fn,
        resolve_variant_fn=lambda _base, _variant: "Qwen/Qwen3-VL-4B-Instruct",
        extract_caption_fn=lambda text, marker=None: (text, False),
        sanitize_caption_fn=lambda text: text.strip(),
    )

    assert merged == "Draft caption with useful details."


def test_caption_cleanup_forwards_no_thinking_template_kwargs():
    calls = {}

    def run_qwen_inference_fn(*_args, **kwargs):
        calls["chat_template_kwargs"] = kwargs.get("chat_template_kwargs")
        calls["prompt"] = _args[0]
        return "<final>A tank burns in a muddy field.</final>", None, None

    caption = _run_qwen_caption_cleanup(
        "We need to produce a final caption.",
        pil_img=object(),
        max_new_tokens=64,
        base_model_id="Qwen/Qwen3-VL-30B-A3B-Thinking",
        use_caption_cache=True,
        authoritative_counts_note=(
            "Authoritative object counts that must appear in the caption as natural scene facts: "
            "person: 2."
        ),
        cleanup_prompt_override="Frontend cleanup policy: remove loops and keep counts.",
        chat_template_kwargs={"enable_thinking": False},
        run_qwen_inference_fn=run_qwen_inference_fn,
        resolve_variant_fn=lambda _base, _variant: "Qwen/Qwen3-VL-30B-A3B-Instruct",
        extract_caption_fn=lambda text, marker=None: _extract_caption_from_text(text, marker),
        sanitize_caption_fn=_sanitize_qwen_caption,
    )

    assert caption == "A tank burns in a muddy field."
    assert calls["chat_template_kwargs"] == {"enable_thinking": False}
    assert "person: 2" in calls["prompt"]
    assert "Frontend cleanup policy: remove loops and keep counts." in calls["prompt"]


def test_caption_cleanup_rejects_degenerate_editor_output():
    def run_qwen_inference_fn(*_args, **_kwargs):
        return "!" * 160, None, None

    caption = _run_qwen_caption_cleanup(
        "A useful existing caption remains available.",
        pil_img=object(),
        max_new_tokens=64,
        base_model_id="Qwen/Qwen3-VL-4B-Instruct",
        use_caption_cache=True,
        run_qwen_inference_fn=run_qwen_inference_fn,
        resolve_variant_fn=lambda _base, _variant: "Qwen/Qwen3-VL-4B-Instruct",
        extract_caption_fn=lambda text, marker=None: (text, False),
        sanitize_caption_fn=lambda text: text.strip(),
    )

    assert caption == "A useful existing caption remains available."


def test_caption_cleanup_respects_long_final_caption_sentence_budget():
    calls = {}

    def run_qwen_inference_fn(*args, **_kwargs):
        calls["prompt"] = args[0]
        return "<final>A long caption can preserve more details.</final>", None, None

    caption = _run_qwen_caption_cleanup(
        "Draft caption.",
        pil_img=object(),
        max_new_tokens=512,
        base_model_id="Qwen/Qwen3-VL-30B-A3B-Thinking",
        use_caption_cache=True,
        strict=True,
        max_sentences=10,
        run_qwen_inference_fn=run_qwen_inference_fn,
        resolve_variant_fn=lambda _base, _variant: "Qwen/Qwen3-VL-30B-A3B-Instruct",
        extract_caption_fn=lambda text, marker=None: _extract_caption_from_text(text, marker),
        sanitize_caption_fn=_sanitize_qwen_caption,
    )

    assert caption == "A long caption can preserve more details."
    assert "up to 10 complete sentences" in calls["prompt"]
    assert "Return exactly one complete sentence" not in calls["prompt"]


def test_caption_cleanup_includes_raw_first_stage_source_context():
    calls = {}

    def run_qwen_inference_fn(*args, **_kwargs):
        calls["prompt"] = args[0]
        return "<final>A tank burns beside smoke and scattered wreckage.</final>", None, None

    caption = _run_qwen_caption_cleanup(
        "A tank burns.",
        pil_img=object(),
        max_new_tokens=512,
        base_model_id="Qwen/Qwen3-VL-30B-A3B-Thinking",
        use_caption_cache=True,
        strict=True,
        user_prompt="Can we infer the likely location of this scene?",
        source_output="Scattered wreckage lies near the burning tank.",
        run_qwen_inference_fn=run_qwen_inference_fn,
        resolve_variant_fn=lambda _base, _variant: "Qwen/Qwen3-VL-30B-A3B-Instruct",
        extract_caption_fn=lambda text, marker=None: _extract_caption_from_text(text, marker),
        sanitize_caption_fn=_sanitize_qwen_caption,
    )

    assert caption == "A tank burns beside smoke and scattered wreckage."
    assert "User caption request: Can we infer the likely location of this scene?" in calls["prompt"]
    assert "Scattered wreckage lies near the burning tank." in calls["prompt"]
