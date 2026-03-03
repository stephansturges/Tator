import json
import time
import uuid

import pytest

from .helpers.api import api_json, delete_dataset_if_exists
from .helpers.ui import ensure_local_mode, open_datasets_tab, open_transient_in_annotation, open_transient_session


pytestmark = [pytest.mark.ui, pytest.mark.ui_smoke]


# CASE_ID: DATASET_DEFAULTS_CLEAR

def test_dataset_manager_defaults_are_clear(playwright_page):
    page, _ = playwright_page
    ensure_local_mode(page)
    open_datasets_tab(page)
    summary = page.text_content("#datasetPathSummary") or ""
    assert "No transient server-path dataset is open" in summary
    save_disabled = page.eval_on_selector("#datasetPathSaveBtn", "el => !!el.disabled")
    annotate_disabled = page.eval_on_selector("#datasetPathAnnotateBtn", "el => !!el.disabled")
    assert save_disabled is True
    assert annotate_disabled is True
    source_summary = page.text_content("#annotationSourceSummary") or ""
    assert "open a dataset from Dataset Manager" in source_summary


# CASE_ID: DATASET_TRANSIENT_OPEN_ANNOTATION

def test_transient_path_opens_in_annotation(playwright_page):
    page, dataset_path = playwright_page
    ensure_local_mode(page)
    open_transient_in_annotation(page, dataset_path)
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
    ensure_local_mode(page)


# CASE_ID: DATASET_REGISTER_LINKED_OPEN

def test_register_path_creates_linked_card_and_opens_annotation(playwright_page):
    page, dataset_path = playwright_page
    ensure_local_mode(page)
    linked_id = f"pw_linked_{uuid.uuid4().hex[:8]}"
    linked_label = f"Playwright Linked {linked_id}"
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
        ensure_local_mode(page)
        delete_dataset_if_exists(linked_id)


# CASE_ID: DATASET_READONLY_TAKEOVER

def test_readonly_takeover_flow_for_transient_session(playwright_page):
    page, dataset_path = playwright_page
    ensure_local_mode(page)
    session_id = open_transient_session(page, dataset_path)
    api_json(
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
    ensure_local_mode(page)


# CASE_ID: DATASET_CLOSE_BLOCKED_SAVE_FAILURE

def test_close_blocks_when_snapshot_save_fails_then_recovers(playwright_page):
    page, dataset_path = playwright_page
    ensure_local_mode(page)
    open_transient_in_annotation(page, dataset_path)
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
    ensure_local_mode(page)


# CASE_ID: DATASET_STALE_MANIFEST_NO_RESURRECT

def test_close_during_delayed_manifest_does_not_resurrect_dataset(playwright_page):
    page, dataset_path = playwright_page
    ensure_local_mode(page)
    open_transient_session(page, dataset_path)

    delayed = {"used": False}

    def _delay_manifest(route):
        if (not delayed["used"]) and ("/annotation/manifest" in route.request.url):
            delayed["used"] = True
            time.sleep(1.0)
        route.continue_()

    page.route("**/*annotation/manifest*", _delay_manifest)
    try:
        page.click("#datasetPathAnnotateBtn")
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
