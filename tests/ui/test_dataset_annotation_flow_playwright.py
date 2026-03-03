import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
import uuid
from urllib.parse import urlparse

import pytest

pytestmark = pytest.mark.ui


def _env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default) or "").strip()


def _require_ui_env() -> tuple[str, str]:
    if _env("RUN_UI_E2E") != "1":
        pytest.skip("Set RUN_UI_E2E=1 to run browser E2E tests.")
    page_url = _env("UI_PAGE_URL")
    dataset_path = _env("UI_DATASET_PATH")
    if not page_url:
        pytest.skip("Set UI_PAGE_URL to the hosted ybat UI URL.")
    if not dataset_path:
        pytest.skip("Set UI_DATASET_PATH to a valid server-local dataset path.")
    return page_url, dataset_path


def _api_root() -> str:
    return _env("UI_API_ROOT", "http://127.0.0.1:8000").rstrip("/")


def _api_json(
    method: str,
    path: str,
    payload: dict | None = None,
    expected_statuses: tuple[int, ...] = (200,),
) -> object:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        url=f"{_api_root()}{path}",
        data=body,
        headers=headers,
        method=method.upper(),
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:  # nosec B310
            status = int(getattr(resp, "status", 200))
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        raw = exc.read().decode("utf-8", errors="replace")

    if status not in expected_statuses:
        raise AssertionError(f"Unexpected status {status} for {method} {path}: {raw}")
    try:
        return json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return {}


def _delete_dataset_if_exists(dataset_id: str) -> None:
    if not dataset_id:
        return
    try:
        _api_json("DELETE", f"/datasets/{dataset_id}", expected_statuses=(200, 404))
    except Exception:
        # Cleanup should not hide primary test failures.
        pass


def _backend_reachable() -> bool:
    health_url = _env("UI_HEALTH_URL", "http://127.0.0.1:8000/system/health_summary")
    parsed = urlparse(health_url)
    host = parsed.hostname or "127.0.0.1"
    if parsed.port:
        port = int(parsed.port)
    elif parsed.scheme == "https":
        port = 443
    else:
        port = 80
    try:
        with socket.create_connection((host, port), timeout=5):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def playwright_page():
    page_url, dataset_path = _require_ui_env()
    if not _backend_reachable():
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


def _open_datasets_tab(page) -> None:
    page.click("#tabDatasetsButton")
    page.wait_for_selector("#datasetPathInput", timeout=15000)


def _extract_transient_session_id(page) -> str:
    summary = page.text_content("#datasetPathSummary") or ""
    match = re.search(r"Transient session:\s*([^\s\u2022]+)", summary)
    if not match:
        raise AssertionError(f"Unable to parse transient session id from: {summary}")
    return match.group(1).strip()


def _open_transient_session(page, dataset_path: str) -> str:
    _open_datasets_tab(page)
    page.fill("#datasetPathInput", dataset_path)
    page.click("#datasetPathOpenBtn")
    page.wait_for_selector("#datasetPathAnnotateBtn:not([disabled])", timeout=15000)
    return _extract_transient_session_id(page)


def _ensure_local_mode(page) -> None:
    page.click("#tabLabelingButton")
    page.wait_for_selector("#annotationSourceMode", timeout=15000)
    mode_text = (page.text_content("#annotationSourceMode") or "").lower()
    if "local" in mode_text:
        return
    close_btn = page.locator("#annotationCloseBtn")
    close_btn.wait_for(state="visible", timeout=15000)
    close_btn.click(timeout=5000)
    page.wait_for_function(
        "document.querySelector('#annotationSourceMode')?.textContent?.toLowerCase().includes('local')",
        timeout=20000,
    )


def _open_transient_in_annotation(page, dataset_path: str) -> None:
    _open_transient_session(page, dataset_path)
    page.click("#datasetPathAnnotateBtn")
    page.wait_for_function(
        "document.querySelector('#annotationSourceMode')?.textContent?.toLowerCase().includes('transient')",
        timeout=20000,
    )


def test_dataset_manager_defaults_are_clear(playwright_page):
    page, _ = playwright_page
    _ensure_local_mode(page)
    _open_datasets_tab(page)
    summary = page.text_content("#datasetPathSummary") or ""
    assert "No transient server-path dataset is open" in summary
    save_disabled = page.eval_on_selector("#datasetPathSaveBtn", "el => !!el.disabled")
    annotate_disabled = page.eval_on_selector("#datasetPathAnnotateBtn", "el => !!el.disabled")
    assert save_disabled is True
    assert annotate_disabled is True
    source_summary = page.text_content("#annotationSourceSummary") or ""
    assert "open a dataset from Dataset Manager" in source_summary


def test_transient_path_opens_in_annotation(playwright_page):
    page, dataset_path = playwright_page
    _ensure_local_mode(page)
    _open_transient_in_annotation(page, dataset_path)
    page.wait_for_function(
        "document.querySelectorAll('#imageList option').length > 0",
        timeout=20000,
    )
    mode_text = page.text_content("#annotationSourceMode") or ""
    assert "transient" in mode_text.lower()
    option_count = page.evaluate(
        "(() => document.querySelectorAll('#imageList option').length)()"
    )
    assert int(option_count) > 0
    images_disabled = page.eval_on_selector("#images", "el => !!el.disabled")
    assert images_disabled is True
    _ensure_local_mode(page)


def test_register_path_creates_linked_card_and_opens_annotation(playwright_page):
    page, dataset_path = playwright_page
    _ensure_local_mode(page)
    linked_id = f"pw_linked_{uuid.uuid4().hex[:8]}"
    linked_label = f"Playwright Linked {linked_id}"
    try:
        _open_datasets_tab(page)
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
        message_raw = page.text_content("#datasetPathMessage") or ""
        message = message_raw.lower()
        assert "registered" in message, message_raw
        page.wait_for_function(
            "document.querySelector('#datasetListRefreshTop') && !document.querySelector('#datasetListRefreshTop').disabled",
            timeout=60000,
        )
        page.click("#datasetListRefreshTop")
        linked_cards = page.locator(".training-history-item").filter(
            has=page.locator(".badge", has_text="LINKED")
        )
        linked_cards.first.wait_for(timeout=45000)
        assert linked_cards.count() >= 1
        card = linked_cards.first

        open_btn = card.get_by_role("button", name="Open in annotation")
        title = open_btn.get_attribute("title") or ""
        assert "Label Images" in title
        open_btn.click()

        page.wait_for_function(
            "document.querySelector('#annotationSourceMode')?.textContent?.toLowerCase().includes('linked')",
            timeout=20000,
        )
        mode_text = page.text_content("#annotationSourceMode") or ""
        assert "linked" in mode_text.lower()
    finally:
        _ensure_local_mode(page)
        _delete_dataset_if_exists(linked_id)


def test_readonly_takeover_flow_for_transient_session(playwright_page):
    page, dataset_path = playwright_page
    _ensure_local_mode(page)
    session_id = _open_transient_session(page, dataset_path)
    _api_json(
        "POST",
        f"/datasets/transient/{session_id}/annotation/session/start",
        payload={
            "editor_name": "playwright-lock-holder",
            "session_id": f"pw-lock-{uuid.uuid4().hex[:8]}",
        },
    )

    page.click("#datasetPathAnnotateBtn")
    page.wait_for_function(
        "document.querySelector('#annotationSourceMode')?.textContent?.toLowerCase().includes('transient')",
        timeout=20000,
    )
    page.wait_for_function(
        "document.querySelector('#annotationSourceLock')?.textContent?.toLowerCase().includes('read-only')",
        timeout=20000,
    )
    lock_text = page.text_content("#annotationSourceLock") or ""
    assert "playwright-lock-holder" in lock_text

    takeover_enabled = page.eval_on_selector("#annotationTakeoverBtn", "el => !el.disabled")
    caption_disabled = page.eval_on_selector("#qwenCaptionOutput", "el => !!el.disabled")
    assert takeover_enabled is True
    assert caption_disabled is True

    page.click("#annotationTakeoverBtn")
    page.wait_for_function(
        "document.querySelector('#annotationSourceLock')?.textContent?.toLowerCase().includes('writable')",
        timeout=20000,
    )
    caption_disabled_after = page.eval_on_selector("#qwenCaptionOutput", "el => !!el.disabled")
    assert caption_disabled_after is False
    _ensure_local_mode(page)


def test_close_blocks_when_snapshot_save_fails_then_recovers(playwright_page):
    page, dataset_path = playwright_page
    _ensure_local_mode(page)
    _open_transient_in_annotation(page, dataset_path)
    page.wait_for_function(
        "document.querySelectorAll('#imageList option').length > 0",
        timeout=20000,
    )
    page.click("#imageList option:first-child")
    page.dispatch_event("#imageList", "change")
    page.wait_for_function(
        "(() => { const c = document.querySelector('#canvas'); return !!c && c.width > 0 && c.height > 0; })()",
        timeout=20000,
    )
    caption = page.locator("#qwenCaptionOutput")
    caption.fill(f"playwright dirty {uuid.uuid4().hex[:6]}")
    caption.press("Tab")

    failed = {"count": 0}

    def _fail_first_snapshot(route):
        if "/annotation/snapshot" in route.request.url and failed["count"] == 0:
            failed["count"] += 1
            route.fulfill(
                status=500,
                content_type="application/json",
                body=json.dumps({"detail": "forced_snapshot_failure"}),
            )
            return
        route.continue_()

    page.route("**/*annotation/snapshot*", _fail_first_snapshot)
    try:
        page.click("#annotationCloseBtn")
        page.wait_for_function(
            "document.querySelector('#annotationSourceSummary')?.textContent?.toLowerCase().includes('close blocked')",
            timeout=20000,
        )
        mode_text = page.text_content("#annotationSourceMode") or ""
        assert "transient" in mode_text.lower()
    finally:
        page.unroute("**/*annotation/snapshot*", _fail_first_snapshot)

    page.click("#annotationSaveNowBtn")
    page.wait_for_timeout(1200)
    _ensure_local_mode(page)


def test_close_during_delayed_manifest_does_not_resurrect_dataset(playwright_page):
    page, dataset_path = playwright_page
    _ensure_local_mode(page)
    _open_transient_session(page, dataset_path)

    delayed = {"used": False}

    def _delay_manifest(route):
        if (not delayed["used"]) and ("/annotation/manifest" in route.request.url):
            delayed["used"] = True
            time.sleep(1.0)
        route.continue_()

    page.route("**/*annotation/manifest*", _delay_manifest)
    try:
        page.click("#datasetPathAnnotateBtn")
        # Close immediately while manifest load is still in-flight.
        page.click("#annotationCloseBtn", timeout=5000)
        page.wait_for_function(
            "document.querySelector('#annotationSourceMode')?.textContent?.toLowerCase().includes('local')",
            timeout=20000,
        )
        mode_text = page.text_content("#annotationSourceMode") or ""
        assert "local" in mode_text.lower()
        images_disabled = page.eval_on_selector("#images", "el => !!el.disabled")
        assert images_disabled is False
    finally:
        page.unroute("**/*annotation/manifest*", _delay_manifest)
