import pytest

from .helpers.ui import go_to_tab
from .helpers.usability import (
    assert_min_contrast,
    assert_no_horizontal_overflow,
    assert_text_not_clipped,
    open_tooltip_and_assert_readable,
    collect_soft_artifact,
)


pytestmark = [pytest.mark.ui, pytest.mark.ui_usability_full]


@pytest.mark.ui_usability_smoke
# CASE_ID: UX_TOOLTIP_READABILITY_PREPASS
def test_prepass_help_tooltip_renders_readable_content(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabPrepassBuilderButton", "#tabPrepassBuilder")
    selector = "#tabPrepassBuilder label[for='qwenAgentIou'] .help-icon[data-tooltip]"
    page.wait_for_selector(selector, timeout=10000)
    tooltip = open_tooltip_and_assert_readable(page, selector, min_chars=20)
    assert "IoU" in tooltip["content"] or "iou" in tooltip["content"].lower()
    collect_soft_artifact(page, "ux_tooltip_prepass_iou", payload=tooltip)


@pytest.mark.ui_usability_smoke
# CASE_ID: UX_STATUS_NOTICE_LEGIBILITY
def test_status_and_task_notice_text_are_legible(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabLabelingButton", "#tabLabeling")
    page.click("#detectorRunButton")
    page.wait_for_function(
        "(document.querySelector('#samStatus')?.textContent || '').trim().length > 0",
        timeout=8000,
    )
    assert_text_not_clipped(page, "#samStatus")
    assert_min_contrast(page, "#samStatus", threshold=4.5)
    page.wait_for_selector("#taskQueue.visible .task-queue__entry", timeout=10000)
    assert_no_horizontal_overflow(page, "#taskQueue")
    assert_text_not_clipped(page, "#taskQueue .task-queue__entry:first-child")
    collect_soft_artifact(page, "ux_status_task_notice")
