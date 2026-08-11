import json

import pytest

from .helpers.ui import go_to_tab
from .test_class_split_explorer_contract import (
    _mock_active_workspace_single_bbox_result,
    _mock_class_split_result,
)


pytestmark = [pytest.mark.ui, pytest.mark.ui_full]


def _two_image_local_bbox_result():
    result = _mock_class_split_result()
    result["summary"].update(
        {
            "source_mode": "active_workspace",
            "source_id": "pw_cross_layer_partial_export_workspace",
            "wrong_class_candidate_count": 2,
        }
    )
    fixtures = (
        (
            result["points"][1],
            "truck-2",
            "good.png",
            [100.0, 120.0, 180.0, 220.0],
        ),
        (
            result["points"][2],
            "person-1",
            "skipped.png",
            [200.0, 220.0, 270.0, 300.0],
        ),
    )
    candidates = []
    for point, point_id, image_name, bbox in fixtures:
        point.update(
            {
                "point_id": point_id,
                "class_name": "Truck",
                "source_mode": "active_workspace",
                "source_id": "pw_cross_layer_partial_export_workspace",
                "split": "train",
                "image_relpath": image_name,
                "frontend_image_key": image_name,
                "bbox_xyxy": bbox,
                "wrong_class_suspicion": 0.8,
                "is_wrong_class_candidate": True,
                "suggested_neighbor_class": "Person",
            }
        )
        candidates.append(
            {
                "point_id": point_id,
                "class_name": "Truck",
                "suggested_neighbor_class": "Person",
                "wrong_class_suspicion": 0.8,
                "image_relpath": image_name,
                "source_mode": "active_workspace",
                "source_id": "pw_cross_layer_partial_export_workspace",
            }
        )
    result["wrong_class_candidates"] = candidates
    return result


def _delete_receipt(job_id, request, count):
    return {
        "schema": "class-analysis-review-history-delete-v1",
        "status": "deleted",
        "job_id": job_id,
        "client_action_id": request.get("client_action_id", ""),
        "training_capture_requested": request.get("capture_training_data") is True,
        "requested_point_count": count,
        "resolved_review_key_count": count,
        "deleted_count": count,
        "absent_count": 0,
        "deleted_disposition_counts": {},
        "labels_changed": False,
        "annotations_changed": False,
        "training_actions_deleted": 0,
        "qwen_audits_deleted": 0,
    }


def _review_disposition_receipt(
    job_id,
    point_id,
    request,
    *,
    revision="rdr1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    reviewed_at=1_785_800_000.0,
    previous_disposition="",
):
    disposition = str(request.get("disposition") or "")
    cleared = disposition == "clear"
    receipt = {
        "schema": "class-analysis-review-disposition-v3",
        "status": "cleared" if cleared else "recorded",
        "job_id": job_id,
        "point_id": point_id,
        "review_object_key": "cro_" + ("a" * 64),
        "disposition": disposition,
        "client_action_id": request.get("client_action_id", ""),
        "training_capture_requested": request.get("capture_training_data") is True,
    }
    if cleared:
        receipt["previous_disposition"] = previous_disposition
    else:
        receipt.update(
            {
                "human_review_revision": revision,
                "human_reviewed_at": reviewed_at,
                "origin": "desktop",
            }
        )
    return receipt


def _set_verified_review_capabilities(page, **overrides):
    capabilities = {
        "review_disposition_api_version": 3,
        **overrides,
    }
    page.evaluate(
        """(value) => window.__TATOR_TEST_HOOKS__.classSplitSetCapabilities(value)""",
        capabilities,
    )


def _load_class_names(page, *class_names):
    page.locator("#classes").set_input_files(
        {
            "name": "classes.txt",
            "mimeType": "text/plain",
            "buffer": ("\n".join(class_names) + "\n").encode(),
        }
    )
    page.wait_for_function(
        "expected => document.querySelector('#classList')?.options.length === expected",
        arg=len(class_names),
    )


def test_review_history_restore_uses_saved_revision_and_requeues_candidate(
    playwright_page,
):
    page, _ = playwright_page
    job_id = "pw_cross_layer_restore_job"
    revision = "rdr1_0123456789abcdef0123456789abcdef"
    requests = []

    def clear_review(route):
        request = route.request.post_data_json or {}
        requests.append(request)
        if request.get("expected_revision") != revision:
            route.fulfill(
                status=400,
                content_type="application/json",
                body='{"detail":"review_disposition_clear_precondition_invalid"}',
            )
            return
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                _review_disposition_receipt(
                    job_id,
                    "truck-2",
                    request,
                    previous_disposition="skip",
                )
            ),
        )

    result = _mock_class_split_result()
    result["points"][1].update(
        {
            "human_review_disposition": "skip",
            "human_reviewed_at": "2026-08-03T20:00:00.123456Z",
            "human_review_revision": revision,
            "human_review_origin": "desktop",
            "human_review_persistence": "durable",
        }
    )
    page.route("**/review_disposition", clear_review)
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    _set_verified_review_capabilities(page)
    page.evaluate(
        """async ({result, jobId}) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            result,
            jobId
        )""",
        {"result": result, "jobId": job_id},
    )
    page.locator("#classSplitReviewedPanel").evaluate(
        "node => { node.open = true; }"
    )

    page.click(
        '#classSplitReviewedList [data-action="restore-review"]'
    )
    page.wait_for_function(
        "() => (document.querySelector('#classSplitReviewedSummary')?.textContent || '') === 'Review history (0)'",
        timeout=3000,
    )

    assert len(requests) == 1
    assert requests[0]["disposition"] == "clear"
    assert requests[0]["expected_revision"] == revision
    assert page.locator(
        '.class-split-wrong-item[data-point-id="truck-2"]'
    ).count() == 1


def test_restore_retry_accepts_already_clear_after_lost_success_response(
    playwright_page,
):
    page, _ = playwright_page
    job_id = "pw_cross_layer_restore_retry_job"
    revision = "rdr1_abcdefabcdefabcdefabcdefabcdefab"
    requests = []

    def clear_review(route):
        request = route.request.post_data_json or {}
        requests.append(request)
        if len(requests) == 1:
            # Model the transport failing after the server committed the clear.
            route.fulfill(
                status=503,
                content_type="application/json",
                body='{"detail":"response_lost_after_commit"}',
            )
            return
        receipt = _review_disposition_receipt(
            job_id,
            "truck-2",
            request,
            previous_disposition="",
        )
        receipt["status"] = "already_clear"
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(receipt),
        )

    result = _mock_class_split_result()
    result["points"][1].update(
        {
            "human_review_disposition": "skip",
            "human_reviewed_at": 1_785_800_050.0,
            "human_review_revision": revision,
            "human_review_origin": "desktop",
            "human_review_persistence": "durable",
        }
    )
    page.route("**/review_disposition", clear_review)
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    _set_verified_review_capabilities(page)
    page.evaluate(
        """async ({result, jobId}) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            result,
            jobId
        )""",
        {"result": result, "jobId": job_id},
    )
    panel = page.locator("#classSplitReviewedPanel")
    panel.evaluate("node => { node.open = true; }")

    page.click('#classSplitReviewedList [data-action="restore-review"]')
    page.wait_for_function(
        "() => (document.querySelector('#classSplitJobStatus')?.textContent || '').includes('response_lost_after_commit')",
        timeout=3000,
    )
    assert page.text_content("#classSplitReviewedSummary") == "Review history (1)"

    page.click('#classSplitReviewedList [data-action="restore-review"]')
    page.wait_for_function(
        "() => (document.querySelector('#classSplitReviewedSummary')?.textContent || '') === 'Review history (0)'",
        timeout=3000,
    )

    assert len(requests) == 2
    assert all(request["expected_revision"] == revision for request in requests)
    assert page.locator(
        '.class-split-wrong-item[data-point-id="truck-2"]'
    ).count() == 1


def test_malformed_success_receipt_restores_and_reuses_retry_token(playwright_page):
    page, _ = playwright_page
    requests = []

    def disposition_response(route):
        request = route.request.post_data_json or {}
        requests.append(request)
        if len(requests) == 1:
            route.fulfill(
                status=200,
                content_type="application/json",
                body="{}",
            )
            return
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                _review_disposition_receipt(
                    "pw_cross_layer_invalid_receipt_job",
                    "truck-2",
                    request,
                )
            ),
        )

    page.route("**/review_disposition", disposition_response)
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    _set_verified_review_capabilities(page)
    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            result,
            'pw_cross_layer_invalid_receipt_job'
        )""",
        _mock_class_split_result(),
    )
    card = page.locator(
        '.class-split-wrong-item[data-point-id="truck-2"]'
    )

    page.click(
        '.class-split-wrong-item[data-point-id="truck-2"] '
        '[data-action="skip-wrong"]'
    )
    page.wait_for_function(
        "() => (document.querySelector('#classSplitJobStatus')?.textContent || '').toLowerCase().includes('invalid')",
        timeout=3000,
    )

    assert card.count() == 1
    assert page.text_content("#classSplitReviewedSummary") == "Review history (0)"
    assert "invalid" in (page.text_content("#taskQueue") or "").lower()

    page.click(
        '.class-split-wrong-item[data-point-id="truck-2"] '
        '[data-action="skip-wrong"]'
    )
    page.wait_for_function(
        "() => (document.querySelector('#classSplitReviewedSummary')?.textContent || '') === 'Review history (1)'",
        timeout=3000,
    )
    assert len(requests) == 2
    assert requests[0]["client_action_id"] == requests[1]["client_action_id"]


def test_local_bbox_history_changes_from_pending_to_exported_after_download(
    playwright_page,
):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    _load_class_names(page, "Truck")
    page.evaluate(
        """(fixture) => window.__TATOR_TEST_HOOKS__.classSplitSeedActiveWorkspaceAnnotations(fixture)""",
        {
            "imageKey": "single.png",
            "width": 960,
            "height": 960,
            "boxes": {
                "Truck": [
                    {"x": 100, "y": 120, "width": 80, "height": 100},
                    {"x": 300, "y": 320, "width": 50, "height": 60},
                ],
            },
        },
    )
    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            result,
            'pw_cross_layer_bbox_export_job'
        )""",
        _mock_active_workspace_single_bbox_result(),
    )
    page.click(
        '.class-split-wrong-item[data-point-id="truck-2"] '
        '[data-action="delete-bbox"]'
    )
    page.locator("#classSplitReviewedPanel").evaluate(
        "node => { node.open = true; }"
    )
    page.wait_for_function(
        "() => (document.querySelector('#classSplitReviewedList')?.textContent || '').includes('press Shift+Y')",
        timeout=3000,
    )

    page.evaluate("() => document.querySelector('#saveBboxes').click()")
    page.wait_for_function(
        "() => (document.querySelector('#classSplitReviewedList')?.textContent || '').includes('Included in the latest YOLO export')",
        timeout=15000,
    )

    history_text = page.text_content("#classSplitReviewedList") or ""
    assert "press Shift+Y" not in history_text
    assert "Included in the latest YOLO export" in history_text


def test_partial_export_marks_only_history_whose_label_was_written(
    playwright_page,
):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.evaluate(
        """(fixture) => window.__TATOR_TEST_HOOKS__.classSplitSeedActiveWorkspaceAnnotations(fixture)""",
        {
            "imageKey": "good.png",
            "width": 960,
            "height": 960,
            "classNames": ["Truck"],
            "boxes": {
                "Truck": [
                    {"x": 100, "y": 120, "width": 80, "height": 100},
                    {"x": 300, "y": 320, "width": 50, "height": 60},
                ],
            },
        },
    )
    page.evaluate(
        """(fixture) => window.__TATOR_TEST_HOOKS__.classSplitSeedActiveWorkspaceAnnotations(fixture)""",
        {
            "imageKey": "skipped.png",
            "width": 960,
            "height": 960,
            "dimensionLoadFailure": True,
            "boxes": {
                "Truck": [
                    {"x": 200, "y": 220, "width": 70, "height": 80},
                    {"x": 400, "y": 420, "width": 55, "height": 65},
                ],
            },
        },
    )
    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            result,
            'pw_cross_layer_partial_export_job'
        )""",
        _two_image_local_bbox_result(),
    )

    for point_id in ("truck-2", "person-1"):
        page.click(
            f'.class-split-wrong-item[data-point-id="{point_id}"] '
            '[data-action="delete-bbox"]'
        )
        page.wait_for_function(
            "pointId => !document.querySelector(`.class-split-wrong-item[data-point-id=\"${pointId}\"]`)",
            arg=point_id,
            timeout=3000,
        )

    page.locator("#classSplitReviewedPanel").evaluate(
        "node => { node.open = true; }"
    )
    page.wait_for_function(
        "() => (document.querySelector('#classSplitReviewedSummary')?.textContent || '') === 'Review history (2)'",
        timeout=3000,
    )
    page.evaluate("() => document.querySelector('#saveBboxes').click()")
    page.wait_for_function(
        "() => (document.querySelector('#samStatus')?.textContent || '').includes('unreadable dimensions')",
        timeout=15000,
    )

    good_row = page.locator(".class-split-reviewed-item").filter(
        has_text="good.png"
    )
    skipped_row = page.locator(".class-split-reviewed-item").filter(
        has_text="skipped.png"
    )
    assert "Included in the latest YOLO export" in good_row.inner_text()
    assert "press Shift+Y" not in good_row.inner_text()
    assert "press Shift+Y" in skipped_row.inner_text()
    assert "Included in the latest YOLO export" not in skipped_row.inner_text()
