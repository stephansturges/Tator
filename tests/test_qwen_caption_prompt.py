from localinferenceapi import (
    QwenCaptionHint,
    _build_caption_overlap_guidance,
    _build_qwen_caption_prompt,
    _group_hints_by_window,
    _qwen_messages_with_no_think,
    _qwen_neutralize_thinking_prefill,
    _resolve_qwen_caption_refinement_model_id,
)
from models.schemas import QwenCaptionRequest
from services.qwen import (
    _caption_count_conflicts,
    _caption_demote_unstable_glossary_subtypes,
    _caption_is_degenerate_impl,
    _caption_has_meta,
    _caption_needs_english_rewrite,
    _caption_needs_completion,
    _caption_needs_refine,
    _caption_repetition_loop_detected,
    _caption_trim_to_complete_sentences,
    _clean_caption_source_context_text,
    _extract_caption_from_text,
    _format_caption_source_output_context,
    _format_qwen_load_error_impl,
    _resolve_qwen_caption_decode,
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
    assert "COUNTS (use exactly)" in prompt
    assert "User caption request: Test prompt" in prompt
    assert "Treat the user caption request as required guidance" in prompt
    assert "Labeled class inventory" in prompt
    assert '"label":"car"' in prompt or '"label":"person"' in prompt
    assert counts.get("car") == 1
    assert counts.get("person") == 1
    assert used == 1
    assert truncated is True


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
    glossary = {"UPole": ["utility pole"]}
    caption = (
        "A tall utility pole stands near the center-left. "
        "A few utility poles are visible throughout the neighborhood."
    )

    conflicts = _caption_count_conflicts(caption, {"UPole": 1}, glossary)
    needs_refine, missing = _caption_needs_refine(
        caption,
        {"UPole": 1},
        detailed_mode=True,
        include_counts=True,
        glossary_map=glossary,
    )

    assert conflicts == ["UPole"]
    assert needs_refine is True
    assert missing == ["UPole"]
    assert (
        _caption_count_conflicts(
            "A utility pole stands near the road, with another pole farther back.",
            {"UPole": 2},
            glossary,
        )
        == []
    )


def test_caption_english_rewrite_ignores_ascii_equivalent_punctuation():
    assert _caption_needs_english_rewrite("A top-down view shows a brown-roofed house.") is False
    assert _caption_needs_english_rewrite("A top\u2011down view shows a brown\u2011roofed house.") is False
    assert _caption_needs_english_rewrite("\u57ce\u5e02\u4e2d\u6709\u4e00\u8f86\u8eca\u3002") is True


def test_caption_demotes_disputed_glossary_subtype_to_broad_term():
    glossary = {
        "LightVehicle": [
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
        {"LightVehicle": 1},
        glossary,
        source_outputs=[
            ("Window 1", "A small vehicle is parked near the buildings."),
            ("Window 2", "A red vehicle sits on the dirt path."),
            ("Window 3", "A red pickup truck sits on the path."),
        ],
    )

    assert "pickup truck" not in cleaned
    assert "red vehicle" in cleaned


def test_caption_keeps_consistently_supported_glossary_subtype():
    glossary = {
        "LightVehicle": [
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
        {"LightVehicle": 1},
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
    assert "car or van" in guidance
    assert source_id not in guidance
    assert "object ID" not in guidance
    assert "bbox ID" not in guidance


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
    glossary = _default_agent_glossary_for_labelmap(["light_vehicle", "gas_tank"])
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
    assert "possible variants include light vehicle, car, van, pickup truck" in prompt
    assert 'gas_tank: broad term "storage tank"' in prompt
    assert "Glossary variants are possible members of a class, not assertions" in prompt
    assert "Do not choose a subtype from the glossary unless the image clearly supports that subtype" in prompt
    assert "COUNTS (use exactly): small vehicle: 1, storage tank: 1" in prompt
    assert '"label":"small vehicle"' in prompt
    assert '"label":"storage tank"' in prompt


def test_default_caption_glossary_matches_camelcase_labelmap_names():
    glossary = _default_agent_glossary_for_labelmap(["LightVehicle", "UPole", "GasTank"])

    assert '"LightVehicle": [\n    "small vehicle"' in glossary
    assert '"UPole": [\n    "utility pole"' in glossary
    assert '"GasTank": [\n    "storage tank"' in glossary


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

    assert "no more than 2 complete sentences" in prompt
    assert "Write a detailed caption" not in prompt
    assert "longer captions are acceptable" not in prompt
    assert "Be maximally descriptive" not in prompt


def test_sanitize_qwen_caption_removes_repeated_sentence_tail():
    repeated = (
        "A drone view shows a dry site with a white path and several buildings. "
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
        "An aerial view of"
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
    )

    assert payload.caption_system_prompt == "Stop after a single paragraph."


def test_caption_request_normalizes_refinement_model_id():
    same_payload = QwenCaptionRequest(
        image_base64="data:image/png;base64,AA==",
        refinement_model_id=" same ",
    )
    explicit_payload = QwenCaptionRequest(
        image_base64="data:image/png;base64,AA==",
        refinement_model_id=" mlx-community/Qwen3-VL-4B-Instruct-4bit ",
    )

    assert same_payload.refinement_model_id is None
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
        [(0, 0, 128, "Window detail.")],
        pil_img=object(),
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
        source_outputs=[("Full-image raw output", "Smoke rises near scattered wreckage.")],
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
    assert "Smoke rises near scattered wreckage." in calls["prompt"]
    assert "preserve broad category terms" in calls["prompt"]
    assert "do not change small vehicle into car" in calls["prompt"]


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
        chat_template_kwargs={"enable_thinking": False},
        run_qwen_inference_fn=run_qwen_inference_fn,
        resolve_variant_fn=lambda _base, _variant: "Qwen/Qwen3-VL-30B-A3B-Instruct",
        extract_caption_fn=lambda text, marker=None: _extract_caption_from_text(text, marker),
        sanitize_caption_fn=_sanitize_qwen_caption,
    )

    assert caption == "A tank burns in a muddy field."
    assert calls["chat_template_kwargs"] == {"enable_thinking": False}
    assert "preserve broad category terms" in calls["prompt"]


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
