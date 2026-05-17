import pytest

from .helpers.ui import ensure_local_mode


pytestmark = [pytest.mark.ui, pytest.mark.ui_smoke, pytest.mark.ui_auto_label]


def test_auto_label_panel_is_not_shown_in_labeling_sidebar(playwright_page):
    page, _dataset_path = playwright_page
    ensure_local_mode(page)

    assert page.locator("#qwenAutoLabelRun").count() == 0
    assert page.locator("#qwenAutoLabelModeSummary").count() == 0
