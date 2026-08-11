import json
from pathlib import Path

import pytest


pytestmark = [pytest.mark.ui, pytest.mark.ui_smoke]


def test_openai_batch_manager_scan_path_and_collection_error_render_without_backend():
    sync_api = pytest.importorskip("playwright.sync_api")
    repo_root = Path(__file__).resolve().parents[3]
    page_url = (repo_root / "ybat-master" / "tator.html").as_uri()
    requests = []

    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        def route_backend(route):
            request = route.request
            requests.append((request.method, request.url, request.post_data or ""))
            if "/qwen/caption/openai_batches/scan" in request.url:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "artifacts": [
                                {
                                    "openai_batch_id": "batch_scan",
                                    "status": "adoptable",
                                    "reason": "ready_to_adopt",
                                    "artifact_dir": "/tmp/legacy",
                                    "adoptable": True,
                                }
                            ],
                            "truncated": False,
                        }
                    ),
                )
                return
            if request.url.endswith("/qwen/caption/openai_batches") or "/qwen/caption/openai_batches?" in request.url:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        [
                            {
                                "job_id": "ocap_partial",
                                "kind": "openai_caption_batch_collection",
                                "status": "partial_failed",
                                "dataset_id": "dataset_a",
                                "case_count": 100,
                                "request_counts": {"total": 100, "completed": 50, "failed": 0},
                                "shard_summary": {
                                    "total_shards": 2,
                                    "active_shards": 0,
                                    "problem_shards": 1,
                                    "collection_error_shards": 1,
                                    "request_counts": {"total": 100, "completed": 50, "failed": 0},
                                },
                                "message": "collection error test",
                            }
                        ]
                    ),
                )
                return
            route.fulfill(status=200, content_type="application/json", body="{}")

        page.route("http://localhost:8000/**", route_backend)
        page.route("http://127.0.0.1:8000/**", route_backend)
        try:
            page.goto(page_url, wait_until="domcontentloaded")
            page.wait_for_selector("#qwenCaptionOpenAiBatchJobs", state="attached", timeout=10000)
            page.evaluate("document.querySelector('#qwenCaptionAdoptOpenAiBatchPath').value = '/tmp/legacy'")
            page.evaluate("document.querySelector('#qwenCaptionScanOpenAiBatches').click()")
            page.wait_for_function(
                "document.querySelector('#qwenCaptionOpenAiBatchRecovery')?.textContent?.includes('batch_scan')",
                timeout=10000,
            )
            page.evaluate("document.querySelector('#qwenCaptionRefreshOpenAiBatches').click()")
            page.wait_for_function(
                "document.querySelector('#qwenCaptionOpenAiBatchJobs')?.textContent?.includes('collect error')",
                timeout=10000,
            )

            recovery_text = page.locator("#qwenCaptionOpenAiBatchRecovery").text_content() or ""
            jobs_text = page.locator("#qwenCaptionOpenAiBatchJobs").text_content() or ""
            assert "batch_scan" in recovery_text
            assert "adoptable" in recovery_text
            assert "ocap_partial" in jobs_text
            assert "1 collect error" in jobs_text
            assert "1 need attention" in jobs_text

            scan_payloads = [
                json.loads(payload)
                for _method, url, payload in requests
                if "/qwen/caption/openai_batches/scan" in url and payload
            ]
            assert scan_payloads
            assert scan_payloads[-1]["artifact_dirs"] == ["/tmp/legacy"]
            assert scan_payloads[-1]["roots"] == ["/tmp/legacy"]
        finally:
            context.close()
            browser.close()


def test_openai_batch_manager_requires_target_verification_before_import_without_backend():
    sync_api = pytest.importorskip("playwright.sync_api")
    repo_root = Path(__file__).resolve().parents[3]
    page_url = (repo_root / "ybat-master" / "tator.html").as_uri()

    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        def route_backend(route):
            request = route.request
            if request.url.endswith("/qwen/caption/openai_batches") or "/qwen/caption/openai_batches?" in request.url:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        [
                            {
                                "job_id": "ocap_collected",
                                "kind": "openai_caption_batch",
                                "status": "in_progress",
                                "dataset_id": "dataset_a",
                                "case_count": 10,
                                "request_counts": {"total": 10, "completed": 10, "failed": 0},
                                "message": "collected but not verified",
                            }
                        ]
                    ),
                )
                return
            route.fulfill(status=200, content_type="application/json", body="{}")

        page.route("http://localhost:8000/**", route_backend)
        page.route("http://127.0.0.1:8000/**", route_backend)
        try:
            page.goto(page_url, wait_until="domcontentloaded")
            page.wait_for_selector("#qwenCaptionOpenAiBatchJobs", state="attached", timeout=10000)
            page.evaluate(
                """
                const select = document.querySelector('#qwenCaptionDataset');
                select.innerHTML = '<option value="dataset_a">dataset_a</option>';
                select.value = 'dataset_a';
                document.querySelector('#qwenCaptionRefreshOpenAiBatches').click();
                """
            )
            page.wait_for_selector(
                "button[data-openai-batch-action='import'][data-job-id='ocap_collected']",
                state="attached",
                timeout=10000,
            )

            import_button = page.locator("button[data-openai-batch-action='import'][data-job-id='ocap_collected']")
            jobs_text = page.locator("#qwenCaptionOpenAiBatchJobs").text_content() or ""
            assert "ocap_collected" in jobs_text
            assert "not checked" in jobs_text
            assert import_button.is_disabled()
        finally:
            context.close()
            browser.close()


def test_openai_batch_manager_shows_incomplete_rows_and_posts_catchup_without_backend():
    sync_api = pytest.importorskip("playwright.sync_api")
    repo_root = Path(__file__).resolve().parents[3]
    page_url = (repo_root / "ybat-master" / "tator.html").as_uri()
    requests = []
    state = {"catchup_posted": False}

    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        def route_backend(route):
            request = route.request
            requests.append((request.method, request.url, request.post_data or ""))
            if request.url.endswith("/qwen/caption/openai_batches/ocap_incomplete/catchup"):
                state["catchup_posted"] = True
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "job_id": "ocap_catchup_1",
                            "kind": "openai_caption_batch",
                            "status": "preparing",
                            "dataset_id": "dataset_a",
                            "case_count": 2,
                            "catchup_for_job_id": "ocap_incomplete",
                            "message": "Preparing OpenAI Batch catch-up for 2 incomplete image(s).",
                        }
                    ),
                )
                return
            if request.url.endswith("/qwen/caption/openai_batches") or "/qwen/caption/openai_batches?" in request.url:
                jobs = [
                    {
                        "job_id": "ocap_incomplete",
                        "kind": "openai_caption_batch",
                        "status": "collected",
                        "dataset_id": "dataset_a",
                        "case_count": 10,
                        "match_status": "exact_dataset_match",
                        "warning_count": 1,
                        "request_counts": {"total": 10, "completed": 10, "failed": 0},
                        "output_summary": {
                            "total_cases": 10,
                            "caption_rows": 8,
                            "incomplete_caption_rows": 2,
                            "accepted_cases": 8,
                            "incomplete_cases": 2,
                        },
                        "message": "OpenAI Batch collected: 8/10 accepted caption row(s). 2 incomplete row(s) were held for catch-up.",
                    }
                ]
                if state["catchup_posted"]:
                    jobs[0]["last_catchup_job_id"] = "ocap_catchup_1"
                    jobs[0]["incomplete_resolution"] = {
                        "job_id": "ocap_catchup_1",
                        "status": "preparing",
                        "state": "active",
                        "active": True,
                        "message": "Preparing OpenAI Batch catch-up for 2 incomplete image(s).",
                    }
                    jobs.append(
                        {
                            "job_id": "ocap_catchup_1",
                            "kind": "openai_caption_batch",
                            "status": "preparing",
                            "dataset_id": "dataset_a",
                            "case_count": 2,
                            "catchup_for_job_id": "ocap_incomplete",
                            "message": "Preparing OpenAI Batch catch-up for 2 incomplete image(s).",
                        }
                    )
                route.fulfill(status=200, content_type="application/json", body=json.dumps(jobs))
                return
            route.fulfill(status=200, content_type="application/json", body="{}")

        page.route("http://localhost:8000/**", route_backend)
        page.route("http://127.0.0.1:8000/**", route_backend)
        try:
            page.goto(page_url, wait_until="domcontentloaded")
            page.wait_for_selector("#qwenCaptionOpenAiBatchJobs", state="attached", timeout=10000)
            page.evaluate(
                """
                const select = document.querySelector('#qwenCaptionDataset');
                select.innerHTML = '<option value="dataset_a">dataset_a</option>';
                select.value = 'dataset_a';
                document.querySelector('#qwenCaptionRefreshOpenAiBatches').click();
                """
            )
            page.wait_for_selector(
                "button[data-openai-batch-action='catchup'][data-job-id='ocap_incomplete']",
                state="attached",
                timeout=10000,
            )
            catchup_button = page.locator("button[data-openai-batch-action='catchup'][data-job-id='ocap_incomplete']")
            jobs_text = page.locator("#qwenCaptionOpenAiBatchJobs").text_content() or ""
            assert "2 incomplete QA rows held" in jobs_text
            assert "1 label warning" in jobs_text
            assert "3 label warning" not in jobs_text
            assert not catchup_button.is_disabled()

            page.evaluate(
                "document.querySelector(\"button[data-openai-batch-action='catchup'][data-job-id='ocap_incomplete']\").click()"
            )
            page.wait_for_function(
                "document.querySelector('#qwenCaptionOpenAiBatchStatus')?.textContent?.includes('catch-up')",
                timeout=10000,
            )

            assert state["catchup_posted"] is True
            assert any(url.endswith("/qwen/caption/openai_batches/ocap_incomplete/catchup") for _method, url, _payload in requests)
            page.wait_for_function(
                "document.querySelector(\"button[data-openai-batch-action='catchup'][data-job-id='ocap_incomplete']\")?.disabled === true",
                timeout=10000,
            )
            jobs_text_after = page.locator("#qwenCaptionOpenAiBatchJobs").text_content() or ""
            assert "catch-up ocap_catchup_1 active" in jobs_text_after
        finally:
            context.close()
            browser.close()


def test_openai_batch_manager_allows_retryable_missing_catchup_without_backend():
    sync_api = pytest.importorskip("playwright.sync_api")
    repo_root = Path(__file__).resolve().parents[3]
    page_url = (repo_root / "ybat-master" / "tator.html").as_uri()

    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        def route_backend(route):
            request = route.request
            if request.url.endswith("/qwen/caption/openai_batches") or "/qwen/caption/openai_batches?" in request.url:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        [
                            {
                                "job_id": "ocap_retry_missing",
                                "kind": "openai_caption_batch",
                                "status": "collected",
                                "dataset_id": "dataset_a",
                                "case_count": 4,
                                "match_status": "exact_dataset_match",
                                "output_summary": {
                                    "total_cases": 4,
                                    "caption_rows": 3,
                                    "incomplete_caption_rows": 1,
                                    "accepted_cases": 3,
                                    "incomplete_cases": 1,
                                },
                                "last_catchup_job_id": "ocap_missing_catchup",
                                "incomplete_resolution": {
                                    "job_id": "ocap_missing_catchup",
                                    "status": "missing",
                                    "state": "retryable",
                                    "message": "A catch-up job was queued but its local record is missing.",
                                },
                            }
                        ]
                    ),
                )
                return
            route.fulfill(status=200, content_type="application/json", body="{}")

        page.route("http://localhost:8000/**", route_backend)
        page.route("http://127.0.0.1:8000/**", route_backend)
        try:
            page.goto(page_url, wait_until="domcontentloaded")
            page.wait_for_selector("#qwenCaptionOpenAiBatchJobs", state="attached", timeout=10000)
            page.evaluate(
                """
                const select = document.querySelector('#qwenCaptionDataset');
                select.innerHTML = '<option value="dataset_a">dataset_a</option>';
                select.value = 'dataset_a';
                document.querySelector('#qwenCaptionRefreshOpenAiBatches').click();
                """
            )
            page.wait_for_selector(
                "button[data-openai-batch-action='catchup'][data-job-id='ocap_retry_missing']",
                state="attached",
                timeout=10000,
            )

            catchup_button = page.locator("button[data-openai-batch-action='catchup'][data-job-id='ocap_retry_missing']")
            jobs_text = page.locator("#qwenCaptionOpenAiBatchJobs").text_content() or ""
            assert "previous catch-up ocap_missing_catchup can be retried" in jobs_text
            assert not catchup_button.is_disabled()
        finally:
            context.close()
            browser.close()


def test_openai_batch_manager_confirmed_dataset_id_change_import_sends_safe_mode_without_backend():
    sync_api = pytest.importorskip("playwright.sync_api")
    repo_root = Path(__file__).resolve().parents[3]
    page_url = (repo_root / "ybat-master" / "tator.html").as_uri()
    requests = []

    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        def route_backend(route):
            request = route.request
            requests.append((request.method, request.url, request.post_data or ""))
            if request.url.endswith("/qwen/caption/openai_batches") or "/qwen/caption/openai_batches?" in request.url:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        [
                            {
                                "job_id": "ocap_renamed",
                                "kind": "openai_caption_batch",
                                "status": "collected",
                                "dataset_id": "old_dataset",
                                "case_count": 10,
                                "match_status": "same_images_different_dataset_id",
                                "target_verification": {
                                    "match_status": "same_images_different_dataset_id",
                                    "message": "Target image identities match but the dataset id differs.",
                                },
                            }
                        ]
                    ),
                )
                return
            if request.url.endswith("/qwen/caption/openai_batches/ocap_renamed/import"):
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "job_id": "ocap_renamed",
                            "kind": "openai_caption_batch",
                            "status": "imported",
                            "message": "Imported OpenAI Batch output.",
                            "import_report": {"saved_captions": 10, "saved_generated_qa_rows": 80},
                        }
                    ),
                )
                return
            route.fulfill(status=200, content_type="application/json", body="{}")

        page.route("http://localhost:8000/**", route_backend)
        page.route("http://127.0.0.1:8000/**", route_backend)
        try:
            page.goto(page_url, wait_until="domcontentloaded")
            page.wait_for_selector("#qwenCaptionOpenAiBatchJobs", state="attached", timeout=10000)
            page.evaluate(
                """
                const select = document.querySelector('#qwenCaptionDataset');
                select.innerHTML = '<option value="new_dataset">new_dataset</option>';
                select.value = 'new_dataset';
                document.querySelector('#qwenCaptionRefreshOpenAiBatches').click();
                """
            )
            page.wait_for_selector(
                "button[data-openai-batch-action='import'][data-job-id='ocap_renamed']",
                state="attached",
                timeout=10000,
            )

            page.once("dialog", lambda dialog: dialog.dismiss())
            page.evaluate("document.querySelector(\"button[data-openai-batch-action='import'][data-job-id='ocap_renamed']\").click()")
            dismissed_imports = [
                payload
                for _method, url, payload in requests
                if url.endswith("/qwen/caption/openai_batches/ocap_renamed/import") and payload
            ]
            assert dismissed_imports == []

            page.once("dialog", lambda dialog: dialog.accept())
            page.evaluate("document.querySelector(\"button[data-openai-batch-action='import'][data-job-id='ocap_renamed']\").click()")
            page.wait_for_function(
                "document.querySelector('#qwenCaptionOpenAiBatchStatus')?.textContent?.includes('Imported OpenAI Batch output')",
                timeout=10000,
            )

            import_payloads = [
                json.loads(payload)
                for _method, url, payload in requests
                if url.endswith("/qwen/caption/openai_batches/ocap_renamed/import") and payload
            ]
            assert len(import_payloads) == 1
            assert import_payloads[0] == {
                "dataset_id": "new_dataset",
                "import_mode": "allow_same_fingerprint_dataset_id_change",
            }
        finally:
            context.close()
            browser.close()


def test_openai_batch_manager_partial_parent_import_refreshes_caption_state_without_backend():
    sync_api = pytest.importorskip("playwright.sync_api")
    repo_root = Path(__file__).resolve().parents[3]
    page_url = (repo_root / "ybat-master" / "tator.html").as_uri()
    requests = []

    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        def route_backend(route):
            request = route.request
            requests.append((request.method, request.url, request.post_data or ""))
            if request.url.endswith("/qwen/caption/openai_batches") or "/qwen/caption/openai_batches?" in request.url:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        [
                            {
                                "job_id": "ocap_parent_partial",
                                "kind": "openai_caption_batch_collection",
                                "status": "collected",
                                "dataset_id": "dataset_a",
                                "case_count": 100,
                                "match_status": "exact_dataset_match",
                                "target_verification": {
                                    "match_status": "exact_dataset_match",
                                    "message": "Target dataset and image identities match.",
                                },
                                "shard_summary": {
                                    "total_shards": 2,
                                    "active_shards": 1,
                                    "status_counts": {"collected": 1, "in_progress": 1},
                                    "request_counts": {"total": 100, "completed": 50, "failed": 0},
                                },
                            }
                        ]
                    ),
                )
                return
            if request.url.endswith("/qwen/caption/openai_batches/ocap_parent_partial/import"):
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "job_id": "ocap_parent_partial",
                            "kind": "openai_caption_batch_collection",
                            "status": "in_progress",
                            "message": "Imported 1/2 OpenAI Batch shard(s): 50 caption(s), 400 generated QA row(s).",
                            "import_report": {
                                "saved_captions": 50,
                                "saved_generated_qa_rows": 400,
                                "imported_shards": 1,
                                "shards": 2,
                            },
                        }
                    ),
                )
                return
            if request.url.endswith("/datasets"):
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps([{"id": "dataset_a", "label": "dataset_a"}]),
                )
                return
            if "/datasets/dataset_a/captions/coverage" in request.url:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "image_count": 100,
                            "complete_image_count": 50,
                            "incomplete_image_count": 50,
                            "base_caption_count": 50,
                            "generated_qa_count": 400,
                            "targets": {"base_captions_per_image": 1, "generated_qa_per_image": 8},
                            "images": [],
                        }
                    ),
                )
                return
            route.fulfill(status=200, content_type="application/json", body="{}")

        page.route("http://localhost:8000/**", route_backend)
        page.route("http://127.0.0.1:8000/**", route_backend)
        try:
            page.goto(page_url, wait_until="domcontentloaded")
            page.wait_for_selector("#qwenCaptionOpenAiBatchJobs", state="attached", timeout=10000)
            page.evaluate(
                """
                const select = document.querySelector('#qwenCaptionDataset');
                select.innerHTML = '<option value="dataset_a">dataset_a</option>';
                select.value = 'dataset_a';
                document.querySelector('#qwenCaptionRefreshOpenAiBatches').click();
                """
            )
            page.wait_for_selector(
                "button[data-openai-batch-action='import'][data-job-id='ocap_parent_partial']",
                state="attached",
                timeout=10000,
            )
            page.evaluate(
                """
                const filter = document.querySelector('#qwenCaptionOpenAiBatchFilter');
                filter.value = 'collected';
                filter.dispatchEvent(new Event('change', { bubbles: true }));
                """
            )
            page.wait_for_selector(
                "button[data-openai-batch-action='import'][data-job-id='ocap_parent_partial']",
                state="attached",
                timeout=10000,
            )
            assert "1 ready to import" in (page.locator("#qwenCaptionOpenAiBatchJobs").text_content() or "")
            archive_button = page.locator("button[data-openai-batch-action='archive'][data-job-id='ocap_parent_partial']")
            cancel_button = page.locator("button[data-openai-batch-action='cancel'][data-job-id='ocap_parent_partial']")
            assert archive_button.is_disabled()
            assert not cancel_button.is_disabled()
            page.evaluate("document.querySelector(\"button[data-openai-batch-action='import'][data-job-id='ocap_parent_partial']\").click()")
            page.wait_for_function(
                "document.querySelector('#qwenCaptionOpenAiBatchStatus')?.textContent?.includes('Imported 1/2')",
                timeout=10000,
            )
            page.wait_for_function(
                "document.querySelector('#qwenCaptionCoverageStatus')?.textContent?.includes('50/100 complete')",
                timeout=10000,
            )

            dataset_refreshes = [url for _method, url, _payload in requests if url.endswith("/datasets")]
            coverage_refreshes = [url for _method, url, _payload in requests if "/datasets/dataset_a/captions/coverage" in url]
            assert dataset_refreshes
            assert coverage_refreshes
        finally:
            context.close()
            browser.close()


def test_openai_batch_manager_disables_parent_import_when_no_shard_is_ready_without_backend():
    sync_api = pytest.importorskip("playwright.sync_api")
    repo_root = Path(__file__).resolve().parents[3]
    page_url = (repo_root / "ybat-master" / "tator.html").as_uri()

    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        def route_backend(route):
            request = route.request
            if request.url.endswith("/qwen/caption/openai_batches") or "/qwen/caption/openai_batches?" in request.url:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        [
                            {
                                "job_id": "ocap_parent_waiting",
                                "kind": "openai_caption_batch_collection",
                                "status": "in_progress",
                                "dataset_id": "dataset_a",
                                "case_count": 100,
                                "match_status": "exact_dataset_match",
                                "target_verification": {
                                    "match_status": "exact_dataset_match",
                                    "message": "Target dataset and image identities match.",
                                },
                                "shard_summary": {
                                    "total_shards": 2,
                                    "active_shards": 2,
                                    "status_counts": {"in_progress": 2},
                                    "request_counts": {"total": 100, "completed": 0, "failed": 0},
                                },
                            }
                        ]
                    ),
                )
                return
            route.fulfill(status=200, content_type="application/json", body="{}")

        page.route("http://localhost:8000/**", route_backend)
        page.route("http://127.0.0.1:8000/**", route_backend)
        try:
            page.goto(page_url, wait_until="domcontentloaded")
            page.wait_for_selector("#qwenCaptionOpenAiBatchJobs", state="attached", timeout=10000)
            page.evaluate(
                """
                const select = document.querySelector('#qwenCaptionDataset');
                select.innerHTML = '<option value="dataset_a">dataset_a</option>';
                select.value = 'dataset_a';
                document.querySelector('#qwenCaptionRefreshOpenAiBatches').click();
                """
            )
            page.wait_for_selector(
                "button[data-openai-batch-action='import'][data-job-id='ocap_parent_waiting']",
                state="attached",
                timeout=10000,
            )

            import_button = page.locator("button[data-openai-batch-action='import'][data-job-id='ocap_parent_waiting']")
            jobs_text = page.locator("#qwenCaptionOpenAiBatchJobs").text_content() or ""
            assert "ocap_parent_waiting" in jobs_text
            assert "ready to import" not in jobs_text
            assert import_button.is_disabled()
            assert "No child shard has collected outputs ready" in (import_button.get_attribute("title") or "")
        finally:
            context.close()
            browser.close()


def test_openai_batch_manager_does_not_infer_parent_ready_from_degraded_status_without_backend():
    sync_api = pytest.importorskip("playwright.sync_api")
    repo_root = Path(__file__).resolve().parents[3]
    page_url = (repo_root / "ybat-master" / "tator.html").as_uri()

    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        def route_backend(route):
            request = route.request
            if request.url.endswith("/qwen/caption/openai_batches") or "/qwen/caption/openai_batches?" in request.url:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        [
                            {
                                "job_id": "ocap_parent_degraded",
                                "kind": "openai_caption_batch_collection",
                                "status": "partial_failed",
                                "dataset_id": "dataset_a",
                                "case_count": 100,
                                "match_status": "exact_dataset_match",
                                "target_verification": {
                                    "match_status": "exact_dataset_match",
                                    "message": "Target dataset and image identities match.",
                                },
                                "shard_summary": {
                                    "total_shards": 2,
                                    "active_shards": 0,
                                    "problem_shards": 1,
                                    "request_counts": {"total": 100, "completed": 50, "failed": 0},
                                },
                            }
                        ]
                    ),
                )
                return
            route.fulfill(status=200, content_type="application/json", body="{}")

        page.route("http://localhost:8000/**", route_backend)
        page.route("http://127.0.0.1:8000/**", route_backend)
        try:
            page.goto(page_url, wait_until="domcontentloaded")
            page.wait_for_selector("#qwenCaptionOpenAiBatchJobs", state="attached", timeout=10000)
            page.evaluate(
                """
                const select = document.querySelector('#qwenCaptionDataset');
                select.innerHTML = '<option value="dataset_a">dataset_a</option>';
                select.value = 'dataset_a';
                document.querySelector('#qwenCaptionRefreshOpenAiBatches').click();
                """
            )
            page.wait_for_selector(
                "button[data-openai-batch-action='import'][data-job-id='ocap_parent_degraded']",
                state="attached",
                timeout=10000,
            )

            import_button = page.locator("button[data-openai-batch-action='import'][data-job-id='ocap_parent_degraded']")
            jobs_text = page.locator("#qwenCaptionOpenAiBatchJobs").text_content() or ""
            assert "ocap_parent_degraded" in jobs_text
            assert "ready to import" not in jobs_text
            assert import_button.is_disabled()

            page.evaluate(
                """
                const filter = document.querySelector('#qwenCaptionOpenAiBatchFilter');
                filter.value = 'collected';
                filter.dispatchEvent(new Event('change', { bubbles: true }));
                """
            )
            page.wait_for_function(
                "!document.querySelector('#qwenCaptionOpenAiBatchJobs')?.textContent?.includes('ocap_parent_degraded')",
                timeout=10000,
            )
            filtered_text = page.locator("#qwenCaptionOpenAiBatchJobs").text_content() or ""
            assert "No OpenAI Batch or local artifact jobs match the selected filter." in filtered_text
        finally:
            context.close()
            browser.close()


def test_openai_batch_manager_allows_retry_import_for_failed_collection_with_import_blocked_child_without_backend():
    sync_api = pytest.importorskip("playwright.sync_api")
    repo_root = Path(__file__).resolve().parents[3]
    page_url = (repo_root / "ybat-master" / "tator.html").as_uri()

    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        def route_backend(route):
            request = route.request
            if request.url.endswith("/qwen/caption/openai_batches") or "/qwen/caption/openai_batches?" in request.url:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        [
                            {
                                "job_id": "ocap_parent_import_retry",
                                "kind": "openai_caption_batch_collection",
                                "status": "failed",
                                "dataset_id": "dataset_a",
                                "case_count": 100,
                                "match_status": "exact_dataset_match",
                                "target_verification": {
                                    "match_status": "exact_dataset_match",
                                    "message": "Target dataset and image identities match.",
                                },
                                "shard_summary": {
                                    "total_shards": 2,
                                    "active_shards": 0,
                                    "problem_shards": 2,
                                    "status_counts": {"import_blocked": 1, "failed": 1},
                                    "request_counts": {"total": 100, "completed": 50, "failed": 50},
                                },
                            }
                        ]
                    ),
                )
                return
            route.fulfill(status=200, content_type="application/json", body="{}")

        page.route("http://localhost:8000/**", route_backend)
        page.route("http://127.0.0.1:8000/**", route_backend)
        try:
            page.goto(page_url, wait_until="domcontentloaded")
            page.wait_for_selector("#qwenCaptionOpenAiBatchJobs", state="attached", timeout=10000)
            page.evaluate(
                """
                const select = document.querySelector('#qwenCaptionDataset');
                select.innerHTML = '<option value="dataset_a">dataset_a</option>';
                select.value = 'dataset_a';
                const filter = document.querySelector('#qwenCaptionOpenAiBatchFilter');
                filter.value = 'collected';
                document.querySelector('#qwenCaptionRefreshOpenAiBatches').click();
                """
            )
            page.wait_for_selector(
                "button[data-openai-batch-action='import'][data-job-id='ocap_parent_import_retry']",
                state="attached",
                timeout=10000,
            )

            jobs_text = page.locator("#qwenCaptionOpenAiBatchJobs").text_content() or ""
            import_button = page.locator("button[data-openai-batch-action='import'][data-job-id='ocap_parent_import_retry']")
            assert "1 ready to import" in jobs_text
            assert not import_button.is_disabled()
        finally:
            context.close()
            browser.close()


def test_openai_batch_manager_renders_local_materialization_as_artifact_job_without_backend():
    sync_api = pytest.importorskip("playwright.sync_api")
    repo_root = Path(__file__).resolve().parents[3]
    page_url = (repo_root / "ybat-master" / "tator.html").as_uri()

    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        local_job = {
            "job_id": "ocap_local_materialized",
            "kind": "openai_caption_batch",
            "status": "imported",
            "dataset_id": "dataset_a",
            "dataset_label": "Dataset A",
            "case_count": 0,
            "local_materialization": True,
            "match_status": "local_materialization",
            "target_verification": {
                "match_status": "local_materialization",
                "message": "No remote Batch import is required; instruction artifacts were materialized locally.",
            },
            "message": "No OpenAI Batch submission was needed; deterministic training artifacts were materialized locally.",
            "result": {
                "instruction_artifacts": {
                    "status": "ok",
                    "bundle_zip": "/tmp/local_bundle.zip",
                    "report_json": "/tmp/local_report.json",
                    "training_row_count": 4,
                }
            },
            "import_report": {
                "status": "imported",
                "saved_captions": 0,
                "saved_generated_qa_rows": 0,
                "instruction_artifacts": {
                    "status": "ok",
                    "bundle_zip": "/tmp/local_bundle.zip",
                    "report_json": "/tmp/local_report.json",
                    "training_row_count": 4,
                },
            },
        }

        def route_backend(route):
            request = route.request
            if "/qwen/caption/openai_batches/ocap_local_materialized" in request.url:
                route.fulfill(status=200, content_type="application/json", body=json.dumps(local_job))
                return
            if request.url.endswith("/qwen/caption/openai_batches") or "/qwen/caption/openai_batches?" in request.url:
                route.fulfill(status=200, content_type="application/json", body=json.dumps([local_job]))
                return
            route.fulfill(status=200, content_type="application/json", body="{}")

        page.route("http://localhost:8000/**", route_backend)
        page.route("http://127.0.0.1:8000/**", route_backend)
        try:
            page.goto(page_url, wait_until="domcontentloaded")
            page.wait_for_selector("#qwenCaptionOpenAiBatchJobs", state="attached", timeout=10000)
            page.evaluate(
                """
                const select = document.querySelector('#qwenCaptionDataset');
                select.innerHTML = '<option value="dataset_a">dataset_a</option>';
                select.value = 'dataset_a';
                document.querySelector('#qwenCaptionRefreshOpenAiBatches').click();
                """
            )
            page.wait_for_selector(
                "button[data-openai-batch-action='details'][data-job-id='ocap_local_materialized']",
                state="attached",
                timeout=10000,
            )

            jobs_text = page.locator("#qwenCaptionOpenAiBatchJobs").text_content() or ""
            import_button = page.locator(
                "button[data-openai-batch-action='import'][data-job-id='ocap_local_materialized']"
            )
            assert "local artifacts" in jobs_text
            assert "local artifacts ready" in jobs_text
            assert "No OpenAI Batch submission was needed" in jobs_text
            assert import_button.is_disabled()

            page.evaluate(
                "document.querySelector(\"button[data-openai-batch-action='details'][data-job-id='ocap_local_materialized']\").click()"
            )
            page.wait_for_function(
                "document.querySelector('#qwenCaptionOpenAiBatchDetails')?.textContent?.includes('/tmp/local_bundle.zip')",
                timeout=10000,
            )
            details_text = page.locator("#qwenCaptionOpenAiBatchDetails").text_content() or ""
            assert "Instruction artifacts" in details_text
            assert "/tmp/local_report.json" in details_text
        finally:
            context.close()
            browser.close()


def test_openai_batch_manager_details_render_output_samples_without_backend():
    sync_api = pytest.importorskip("playwright.sync_api")
    repo_root = Path(__file__).resolve().parents[3]
    page_url = (repo_root / "ybat-master" / "tator.html").as_uri()

    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        def route_backend(route):
            request = route.request
            if "/qwen/caption/openai_batches/ocap_details" in request.url:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "job_id": "ocap_details",
                            "kind": "openai_caption_batch",
                            "status": "collected",
                            "dataset_id": "dataset_a",
                            "case_count": 1,
                            "match_status": "exact_dataset_match",
                            "request_counts": {"total": 1, "completed": 1, "failed": 0},
                            "output_summary": {
                                "caption_rows": 1,
                                "incomplete_caption_rows": 0,
                                "missing_result_rows": 1,
                                "failed_cases": 2,
                            },
                            "result": {
                                "instruction_artifacts": {
                                    "status": "ok",
                                    "bundle_zip": "/tmp/caption_bundle.zip",
                                    "report_json": "/tmp/caption_report.json",
                                    "training_row_count": 9,
                                }
                            },
                            "import_report": {
                                "saved_captions": 1,
                                "saved_generated_qa_rows": 8,
                                "archived_generated_qa_rows": 1,
                                "message": "Imported one sampled job.",
                            },
                            "collected_output_samples": {
                                "accepted_caption_row_count": 1,
                                "incomplete_caption_row_count": 1,
                                "result_row_count": 1,
                                "accepted_caption_rows": [
                                    {
                                        "case_id": "case_001",
                                        "image_name": "sample.png",
                                        "final_status": "accepted",
                                        "caption": "A top-down view shows a pier beside calm water.",
                                        "generated_qa_pair_count": 8,
                                        "generated_qa_target_pair_count": 8,
                                        "qa_pairs": [
                                            {
                                                "question": "What structure is beside the water?",
                                                "answer": "A pier is beside the water.",
                                            }
                                        ],
                                    }
                                ],
                                "incomplete_caption_rows": [
                                    {
                                        "case_id": "case_002",
                                        "image_name": "missing_required.png",
                                        "final_status": "incomplete_qa",
                                        "failure_reason": "required_qa_missing",
                                        "caption": "A second row was held for review.",
                                        "generated_qa_pair_count": 8,
                                        "generated_qa_target_pair_count": 8,
                                        "missing_required_questions": ["What color is the roof?"],
                                    }
                                ],
                                "result_rows": [
                                    {
                                        "case_id": "case_001",
                                        "image_name": "sample.png",
                                        "final_status": "accepted",
                                        "generated_qa_pair_count": 8,
                                    }
                                ],
                            },
                            "events": [
                                {"timestamp": "2026-07-01T10:00:00Z", "message": "OpenAI Batch outputs collected"}
                            ],
                        }
                    ),
                )
                return
            if request.url.endswith("/qwen/caption/openai_batches") or "/qwen/caption/openai_batches?" in request.url:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        [
                            {
                                "job_id": "ocap_details",
                                "kind": "openai_caption_batch",
                                "status": "collected",
                                "dataset_id": "dataset_a",
                                "case_count": 1,
                                "match_status": "exact_dataset_match",
                                "request_counts": {"total": 1, "completed": 1, "failed": 0},
                                "output_summary": {
                                    "caption_rows": 1,
                                    "incomplete_caption_rows": 0,
                                    "missing_result_rows": 1,
                                    "failed_cases": 2,
                                },
                                "message": "collected detail sample",
                            }
                        ]
                    ),
                )
                return
            route.fulfill(status=200, content_type="application/json", body="{}")

        page.route("http://localhost:8000/**", route_backend)
        page.route("http://127.0.0.1:8000/**", route_backend)
        try:
            page.goto(page_url, wait_until="domcontentloaded")
            page.wait_for_selector("#qwenCaptionOpenAiBatchJobs", state="attached", timeout=10000)
            page.evaluate(
                """
                const select = document.querySelector('#qwenCaptionDataset');
                select.innerHTML = '<option value="dataset_a">dataset_a</option>';
                select.value = 'dataset_a';
                document.querySelector('#qwenCaptionRefreshOpenAiBatches').click();
                """
            )
            page.wait_for_selector(
                "button[data-openai-batch-action='details'][data-job-id='ocap_details']",
                state="attached",
                timeout=10000,
            )
            page.evaluate(
                "document.querySelector(\"button[data-openai-batch-action='details'][data-job-id='ocap_details']\").click()"
            )
            page.wait_for_function(
                "document.querySelector('#qwenCaptionOpenAiBatchDetails')?.textContent?.includes('Collected output samples')",
                timeout=10000,
            )

            details_text = page.locator("#qwenCaptionOpenAiBatchDetails").text_content() or ""
            jobs_text = page.locator("#qwenCaptionOpenAiBatchJobs").text_content() or ""
            assert "1 missing result row" in jobs_text
            assert "1 failed output" in jobs_text
            catchup_button = page.locator("button[data-openai-batch-action='catchup'][data-job-id='ocap_details']")
            assert not catchup_button.is_disabled()
            assert page.evaluate(
                """
                [...document.querySelectorAll('.qwen-caption-remote-batch-item')]
                  .some((item) => item.textContent.includes('ocap_details') && item.classList.contains('is-warn'))
                """
            )
            page.evaluate(
                """
                const filter = document.querySelector('#qwenCaptionOpenAiBatchFilter');
                filter.value = 'attention';
                filter.dispatchEvent(new Event('change', { bubbles: true }));
                """
            )
            assert "ocap_details" in (page.locator("#qwenCaptionOpenAiBatchJobs").text_content() or "")
            assert "Instruction artifacts" in details_text
            assert "/tmp/caption_bundle.zip" in details_text
            assert "Missing result rows" in details_text
            assert "Failed output rows" in details_text
            assert "Archived generated QA" in details_text
            assert "A top-down view shows a pier beside calm water." in details_text
            assert "What structure is beside the water?" in details_text
            assert "A pier is beside the water." in details_text
            assert "Missing required: What color is the roof?" in details_text
            assert "Raw detail JSON" in details_text
            assert "OpenAI Batch outputs collected" in details_text
        finally:
            context.close()
            browser.close()
