import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = REPO_ROOT / "ybat-master" / "ybat.html"


def _html() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


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
