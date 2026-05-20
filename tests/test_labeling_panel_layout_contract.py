import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = REPO_ROOT / "ybat-master" / "ybat.html"
CSS_PATH = REPO_ROOT / "ybat-master" / "ybat.css"
JS_PATH = REPO_ROOT / "ybat-master" / "ybat.js"


def _html() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


def _css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def _js() -> str:
    return JS_PATH.read_text(encoding="utf-8")


def _details_opening_tag(html: str, element_id: str) -> str:
    match = re.search(rf"<details\b[^>]*\bid=[\"']{re.escape(element_id)}[\"'][^>]*>", html)
    assert match, f"missing details#{element_id}"
    return match.group(0)


def test_labeling_tool_panels_default_closed_and_ordered():
    html = _html()
    panel_ids = [
        "qwenDetectionDetails",
        "qwenCaptionDetails",
        "qwenEdrDetails",
        "sam3TextPanel",
    ]

    positions = []
    for panel_id in panel_ids:
        tag = _details_opening_tag(html, panel_id)
        assert " open" not in tag, f"{panel_id} should default closed"
        positions.append(html.index(f'id="{panel_id}"'))

    assert positions == sorted(positions)
    assert "Qwen 3 detection engine (not great)" in html
    assert "EDR [wip]" in html
    assert "Ensemble Detection Recipe" in html


def test_yolo_import_and_export_controls_live_in_annotation_source_panel():
    html = _html()
    source_start = html.index('id="annotationSourcePanel"')
    source_end = html.index('id="labelingGpuLockNotice"')
    source_panel = html[source_start:source_end]

    for control_id in ("bboxes", "bboxesFolder", "bboxesSelectFolder", "saveBboxes"):
        assert f'id="{control_id}"' in source_panel

    assert source_start < html.index('id="bboxes"') < html.index('id="qwenDetectionDetails"')
    assert source_start < html.index('id="saveBboxes"') < html.index('id="qwenDetectionDetails"')


def test_caption_output_label_precedes_large_textarea():
    html = _html()
    output_start = html.index('class="qwen-caption-output"')
    output_end = html.index('id="qwenCaptionMeta"')
    output_block = html[output_start:output_end]

    label_pos = output_block.index('for="qwenCaptionOutput"')
    textarea_pos = output_block.index('id="qwenCaptionOutput"')

    assert label_pos < textarea_pos
    assert 'rows="9"' in output_block


def test_caption_prompt_controls_have_tooltips_and_roomy_textareas():
    html = _html()
    css = _css()

    assert "Caption style<span class=\"help-icon\"" in html
    assert "Style prompts (one per line)<span class=\"help-icon\"" in html
    assert "Opening phrases (one per line)<span class=\"help-icon\"" in html
    assert "Caption prompt stack<span class=\"help-icon\"" in html
    assert 'id="qwenCaptionPresetRandom" class="training-button secondary" title=' in html
    assert 'id="qwenCaptionStyleList" rows="4"' in html
    assert 'id="qwenCaptionOpeningList" rows="6"' in html
    assert 'id="qwenCaptionSystemPrompt" rows="12"' in html

    assert "#qwenCaptionStyleList,\n#qwenCaptionOpeningList" in css
    assert "min-height: 240px;" in css
    assert "max-height: 520px;" in css


def test_spacebar_image_navigation_shortcut_is_documented_and_guarded():
    html = _html()
    js = _js()

    assert "Space / Shift + Space" in html
    assert 'const isSpaceKey = key === 32' in js
    assert "event.shiftKey ? -1 : 1" in js
    assert "isInteractiveShortcutTarget" in js
    assert "input, textarea, select, button" in js
