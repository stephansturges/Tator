import pytest

from .helpers.env import backend_reachable, env, require_ui_env


pytestmark = pytest.mark.ui


@pytest.fixture(scope="module")
def playwright_page():
    if env("RUN_UI_E2E") != "1":
        pytest.skip("Set RUN_UI_E2E=1 to run browser E2E tests.")
    try:
        page_url, dataset_path = require_ui_env()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    if not backend_reachable():
        pytest.skip("Backend is unreachable for UI E2E checks.")
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.goto(page_url, wait_until="domcontentloaded")
        yield page, dataset_path
        context.close()
        browser.close()
