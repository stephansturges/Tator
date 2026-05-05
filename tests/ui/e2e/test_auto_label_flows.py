import uuid

import pytest

from .helpers.api import delete_dataset_if_exists
from .helpers.ui import ensure_local_mode, open_datasets_tab, open_transient_in_annotation


pytestmark = [pytest.mark.ui, pytest.mark.ui_smoke, pytest.mark.ui_auto_label]


def test_auto_label_panel_requires_linked_dataset(playwright_page):
    page, dataset_path = playwright_page
    ensure_local_mode(page)
    open_transient_in_annotation(page, dataset_path)
    page.wait_for_function(
        "document.querySelectorAll('#imageList option').length > 0",
        timeout=20000,
    )

    summary = page.text_content("#qwenAutoLabelModeSummary") or ""
    run_disabled = page.eval_on_selector("#qwenAutoLabelRun", "el => !!el.disabled")

    assert "requires a linked dataset annotation session" in summary.lower()
    assert run_disabled is True


def test_auto_label_panel_is_available_for_linked_dataset(playwright_page):
    page, dataset_path = playwright_page
    ensure_local_mode(page)
    linked_id = f"pw_auto_label_{uuid.uuid4().hex[:8]}"
    linked_label = f"Playwright Auto Label {linked_id}"
    try:
        open_datasets_tab(page)
        page.fill("#datasetPathInput", dataset_path)
        page.fill("#datasetPathId", linked_id)
        page.fill("#datasetPathLabel", linked_label)
        page.click("#datasetPathRegisterBtn")
        page.wait_for_function(
            """
(() => {
  const t = (document.querySelector('#datasetPathMessage')?.textContent || '').toLowerCase();
  return t.includes('registered') || t.includes('failed') || t.includes('error');
})()
""",
            timeout=120000,
        )
        page.click("#datasetListRefreshTop")
        linked_cards = page.locator(".training-history-item").filter(
            has=page.locator(".badge", has_text="LINKED")
        )
        linked_cards.first.wait_for(timeout=45000)
        linked_cards.first.get_by_role("button", name="Open in annotation").click()
        page.wait_for_function(
            "document.querySelector('#annotationSourceMode')?.textContent?.toLowerCase().includes('linked')",
            timeout=20000,
        )

        summary = page.text_content("#qwenAutoLabelModeSummary") or ""
        run_disabled = page.eval_on_selector("#qwenAutoLabelRun", "el => !!el.disabled")

        assert "writes directly into the current dataset overlay" in summary.lower()
        assert run_disabled is False
    finally:
        ensure_local_mode(page)
        delete_dataset_if_exists(linked_id)
