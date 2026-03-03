import json

import pytest

from .helpers.ui import go_to_tab


pytestmark = [pytest.mark.ui, pytest.mark.ui_full, pytest.mark.ui_gpu_lifecycle]


# CASE_ID: GPU_YOLO_LIFECYCLE_UI_CONTRACT

def test_yolo_training_lifecycle_render_contract(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabYoloTrainButton", "#tabYoloTrain")

    job_id = "pw_yolo_job_001"
    poll_count = {"value": 0}

    def _mock_yolo_lifecycle(route):
        url = route.request.url
        method = route.request.method.upper()

        if url.endswith("/datasets") and method == "GET":
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps([
                    {
                        "id": "playwright_dataset",
                        "label": "Playwright Dataset",
                        "type": "bbox",
                        "yolo_ready": True,
                        "yolo_seg_ready": False,
                        "storage_mode": "linked",
                        "format": "yolo"
                    }
                ]),
            )
            return

        if url.endswith("/yolo/train/jobs") and method == "POST":
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"job_id": job_id}),
            )
            return

        if url.endswith("/yolo/train/jobs") and method == "GET":
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps([
                    {
                        "job_id": job_id,
                        "status": "running",
                        "created_at": 1700000000,
                        "config": {"run_name": "playwright_yolo"}
                    }
                ]),
            )
            return

        if url.endswith(f"/yolo/train/jobs/{job_id}") and method == "GET":
            poll_count["value"] += 1
            status = "running" if poll_count["value"] < 2 else "cancelled"
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "job_id": job_id,
                        "status": status,
                        "progress": 0.31,
                        "message": "playwright lifecycle",
                        "logs": [{"timestamp": 1700000000, "message": "tick"}],
                        "metrics": [],
                    }
                ),
            )
            return

        if url.endswith(f"/yolo/train/jobs/{job_id}/cancel") and method == "POST":
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "cancel_requested", "job_id": job_id}),
            )
            return

        route.continue_()

    page.route("**/*", _mock_yolo_lifecycle)
    try:
        page.check("#yoloAcceptTos")
        page.click("#yoloDatasetRefresh")
        # <option> elements are not considered "visible"; assert population by count.
        page.wait_for_function(
            "document.querySelectorAll('#yoloDatasetSelect option').length > 0",
            timeout=15000,
        )

        page.click("#yoloTrainStartBtn")
        page.wait_for_function(
            "document.querySelector('#yoloTrainStatusText')?.textContent?.toLowerCase().includes('running')",
            timeout=20000,
        )

        cancel_enabled = page.eval_on_selector("#yoloTrainCancelBtn", "el => !el.disabled")
        assert cancel_enabled is True

        page.click("#yoloTrainCancelBtn")
        page.click("#yoloTrainRefreshBtn")
        page.wait_for_function(
            "document.querySelector('#yoloTrainStatusText')?.textContent?.toLowerCase().includes('cancelled')",
            timeout=20000,
        )
    finally:
        page.unroute("**/*", _mock_yolo_lifecycle)
