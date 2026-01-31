from localinferenceapi import QwenCaptionHint, _build_qwen_caption_prompt


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
