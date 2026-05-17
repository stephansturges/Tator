from localinferenceapi import QwenCaptionHint, _build_qwen_caption_prompt
from models.schemas import QwenCaptionRequest
from services.qwen import (
    _format_qwen_load_error_impl,
    _resolve_qwen_caption_decode,
    _sanitize_qwen_caption,
)


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
    assert "Only mention these classes" in prompt
    assert '"label":"car"' in prompt or '"label":"person"' in prompt
    assert counts.get("car") == 1
    assert counts.get("person") == 1
    assert used == 1
    assert truncated is True


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


def test_caption_decode_defaults_include_repeat_controls():
    decode = _resolve_qwen_caption_decode(_DecodePayload(), is_thinking=False)

    assert decode["do_sample"] is True
    assert decode["repetition_penalty"] > 1.0
    assert decode["repetition_context_size"] >= 64
    assert decode["no_repeat_ngram_size"] >= 4


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
