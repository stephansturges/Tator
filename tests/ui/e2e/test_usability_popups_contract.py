import pytest

from .helpers.ui import go_to_tab
from .helpers.usability import (
    assert_min_contrast,
    assert_text_not_clipped,
    assert_visible_in_viewport,
    collect_soft_artifact,
)


pytestmark = [pytest.mark.ui, pytest.mark.ui_usability_full]


@pytest.mark.ui_usability_smoke
# CASE_ID: UX_MODAL_BACKGROUND_LOAD_READABLE
def test_background_load_modal_is_legible_and_closable(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabLabelingButton", "#tabLabeling")
    try:
        page.evaluate(
            """() => {
                const modal = document.querySelector("#backgroundLoadModal");
                const msg = document.querySelector("#backgroundLoadMessage");
                if (!modal || !msg) return;
                msg.textContent = "Images are still loading in the background. You can continue once loading completes.";
                modal.classList.add("visible");
                modal.setAttribute("aria-hidden", "false");
            }"""
        )
        page.wait_for_function(
            "document.querySelector('#backgroundLoadModal')?.classList.contains('visible')",
            timeout=5000,
        )
        assert_visible_in_viewport(page, "#backgroundLoadModal .modal__dialog")
        assert_visible_in_viewport(page, "#backgroundLoadDismiss")
        assert_text_not_clipped(page, "#backgroundLoadTitle")
        assert_text_not_clipped(page, "#backgroundLoadMessage")
        assert_min_contrast(page, "#backgroundLoadTitle", threshold=4.5)
        assert_min_contrast(page, "#backgroundLoadMessage", threshold=4.5)
        collect_soft_artifact(page, "ux_modal_background_load")
    finally:
        page.evaluate(
            """() => {
                const modal = document.querySelector("#backgroundLoadModal");
                if (!modal) return;
                modal.classList.remove("visible");
                modal.setAttribute("aria-hidden", "true");
            }"""
        )


# CASE_ID: UX_MODAL_BATCH_TWEAK_READABLE
def test_batch_tweak_modal_layout_and_text_legibility(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabLabelingButton", "#tabLabeling")
    try:
        page.evaluate(
            """() => {
                const modal = document.querySelector("#batchTweakModal");
                const klass = document.querySelector("#batchTweakClass");
                if (!modal || !klass) return;
                klass.textContent = "vehicle (12)";
                modal.classList.add("visible");
                modal.setAttribute("aria-hidden", "false");
            }"""
        )
        page.wait_for_function(
            "document.querySelector('#batchTweakModal')?.classList.contains('visible')",
            timeout=5000,
        )
        assert_visible_in_viewport(page, "#batchTweakModal .modal__dialog")
        assert_visible_in_viewport(page, "#batchTweakConfirm")
        assert_visible_in_viewport(page, "#batchTweakCancel")
        assert_text_not_clipped(page, "#batchTweakTitle")
        assert_text_not_clipped(page, "#batchTweakMessage")
        assert_min_contrast(page, "#batchTweakTitle", threshold=4.5)
        assert_min_contrast(page, "#batchTweakMessage", threshold=4.5)
        collect_soft_artifact(page, "ux_modal_batch_tweak")
    finally:
        page.evaluate(
            """() => {
                const modal = document.querySelector("#batchTweakModal");
                if (!modal) return;
                modal.classList.remove("visible");
                modal.setAttribute("aria-hidden", "true");
            }"""
        )
