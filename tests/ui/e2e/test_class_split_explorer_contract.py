import pytest

from .helpers.ui import go_to_tab


pytestmark = [pytest.mark.ui, pytest.mark.ui_full]


def _mock_class_split_result():
    return {
        "summary": {
            "analysis_scope": "all_classes",
            "object_count": 4,
            "class_counts": {"Truck": 2, "Person": 2},
            "projection_mode": "class_balanced_pca",
            "projection_method": "pca",
            "wrong_class_candidate_count": 1,
        },
        "projection_options": {
            "selected": "class_balanced_pca",
            "available": ["global_pca", "class_balanced_pca", "between_class_pca", "within_filter_pca"],
            "coordinates": {
                "class_balanced_pca": [[-1.0, -0.8], [-0.8, -1.0], [0.9, 0.8], [1.0, 0.9]],
                "global_pca": [[-0.5, -0.3], [-0.4, -0.2], [0.4, 0.2], [0.5, 0.3]],
                "between_class_pca": [[-1.5, 0.0], [-1.3, 0.1], [1.3, -0.1], [1.5, 0.0]],
                "within_filter_pca": [[-0.2, -0.1], [0.2, 0.1], [-0.3, 0.2], [0.3, -0.2]],
            },
        },
        "points": [
            {
                "point_id": "truck-1",
                "class_name": "Truck",
                "image_relpath": "img_0.png",
                "projection": [-1.0, -0.8],
                "wrong_class_suspicion": 0.0,
                "is_wrong_class_candidate": False,
            },
            {
                "point_id": "truck-2",
                "class_name": "Truck",
                "image_relpath": "img_1.png",
                "projection": [-0.8, -1.0],
                "wrong_class_suspicion": 0.72,
                "is_wrong_class_candidate": True,
                "suggested_neighbor_class": "Person",
            },
            {
                "point_id": "person-1",
                "class_name": "Person",
                "image_relpath": "img_2.png",
                "projection": [0.9, 0.8],
                "wrong_class_suspicion": 0.0,
                "is_wrong_class_candidate": False,
            },
            {
                "point_id": "person-2",
                "class_name": "Person",
                "image_relpath": "img_3.png",
                "projection": [1.0, 0.9],
                "wrong_class_suspicion": 0.0,
                "is_wrong_class_candidate": False,
            },
        ],
        "wrong_class_candidates": [
            {
                "point_id": "truck-2",
                "class_name": "Truck",
                "suggested_neighbor_class": "Person",
                "wrong_class_suspicion": 0.72,
                "image_relpath": "img_1.png",
            }
        ],
        "clusters": {"clusters": []},
    }


def _mock_class_split_result_with_subclusters():
    result = _mock_class_split_result()
    result["summary"]["analysis_scope"] = "selected_class"
    result["summary"]["class_name"] = "Truck"
    result["summary"]["object_count"] = 4
    result["summary"]["class_counts"] = {"Truck": 4}
    result["projection_options"]["coordinates"] = {
        "class_balanced_pca": [
            [-1.0, -0.8],
            [-0.9, -0.7],
            [-0.15, 0.85],
            [-0.05, 0.92],
        ],
        "global_pca": [
            [-0.5, -0.3],
            [-0.45, -0.25],
            [-0.15, 0.35],
            [-0.1, 0.4],
        ],
        "between_class_pca": [
            [-1.5, 0.0],
            [-1.4, 0.05],
            [-1.2, 0.18],
            [-1.1, 0.25],
        ],
        "within_filter_pca": [
            [-0.8, -0.7],
            [-0.7, -0.8],
            [0.75, 0.7],
            [0.85, 0.8],
        ],
    }
    result["points"] = [
        {
            "point_id": "truck-1",
            "class_name": "Truck",
            "image_relpath": "img_0.png",
            "projection": [-1.0, -0.8],
            "wrong_class_suspicion": 0.0,
            "is_wrong_class_candidate": False,
        },
        {
            "point_id": "truck-2",
            "class_name": "Truck",
            "image_relpath": "img_1.png",
            "projection": [-0.9, -0.7],
            "wrong_class_suspicion": 0.72,
            "is_wrong_class_candidate": False,
            "suggested_neighbor_class": "Person",
        },
        {
            "point_id": "truck-3",
            "class_name": "Truck",
            "image_relpath": "img_2.png",
            "projection": [-0.15, 0.85],
            "wrong_class_suspicion": 0.0,
            "is_wrong_class_candidate": False,
        },
        {
            "point_id": "truck-4",
            "class_name": "Truck",
            "image_relpath": "img_3.png",
            "projection": [-0.05, 0.92],
            "wrong_class_suspicion": 0.0,
            "is_wrong_class_candidate": False,
        },
    ]
    result["wrong_class_candidates"] = []
    result["class_clusters"] = {}
    return result


def _mock_class_split_cluster_search_result():
    return {
        "summary": {
            "cluster_count": 2,
            "best_k": 2,
            "best_silhouette": 0.82,
            "sensitivity": "balanced",
        },
        "clusters": [
            {"cluster_id": 0, "size": 2, "medoid_point_id": "truck-1", "silhouette": 0.82},
            {"cluster_id": 1, "size": 2, "medoid_point_id": "truck-3", "silhouette": 0.82},
        ],
        "labels_by_point_id": {
            "truck-1": 0,
            "truck-2": 0,
            "truck-3": 1,
            "truck-4": 1,
        },
        "reason": "",
    }


def _mock_class_split_many_wrong_result(count=15):
    points = []
    candidates = []
    coords = []
    for idx in range(count):
        point_id = f"truck-wrong-{idx}"
        x = -1.0 + idx * 0.05
        y = -0.8 + idx * 0.03
        coords.append([x, y])
        points.append(
            {
                "point_id": point_id,
                "class_name": "Truck",
                "image_relpath": f"img_{idx}.png",
                "projection": [x, y],
                "wrong_class_suspicion": 0.9 - idx * 0.01,
                "is_wrong_class_candidate": True,
                "suggested_neighbor_class": "Person",
            }
        )
        candidates.append(
            {
                "point_id": point_id,
                "class_name": "Truck",
                "suggested_neighbor_class": "Person",
                "wrong_class_suspicion": 0.9 - idx * 0.01,
                "image_relpath": f"img_{idx}.png",
            }
        )
    return {
        "summary": {
            "analysis_scope": "all_classes",
            "object_count": count,
            "class_counts": {"Truck": count},
            "projection_mode": "class_balanced_pca",
            "projection_method": "pca",
            "wrong_class_candidate_count": count,
        },
        "projection_options": {
            "selected": "class_balanced_pca",
            "available": ["global_pca", "class_balanced_pca", "between_class_pca", "within_filter_pca"],
            "coordinates": {
                "class_balanced_pca": coords,
                "global_pca": coords,
                "between_class_pca": coords,
                "within_filter_pca": coords,
            },
        },
        "points": points,
        "wrong_class_candidates": candidates,
        "clusters": {"clusters": []},
    }


def test_class_split_initial_view_hides_result_toolbar_until_result(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")

    page.wait_for_selector("#classSplitProjection", timeout=15000)
    assert page.eval_on_selector("#classSplitProjection", "el => getComputedStyle(el).display !== 'none'") is True
    projection_box = page.locator("#classSplitProjection").bounding_box()
    viewport = page.viewport_size or {"width": 0}
    assert projection_box is not None
    assert projection_box["x"] >= 0
    assert projection_box["x"] + projection_box["width"] <= viewport["width"]
    assert page.eval_on_selector("#classSplitResults", "el => el.hidden") is True


def test_class_split_graph_controls_are_visible_and_coherent(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)

    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_controls_job')""",
        _mock_class_split_result(),
    )
    page.wait_for_selector("#classSplitColorMode", timeout=15000)
    page.wait_for_selector("#classSplitGraphProjection", timeout=15000)
    page.wait_for_selector("#classSplitFilterClass", timeout=15000)
    page.wait_for_selector("#classSplitDisplayMode", timeout=15000)
    page.wait_for_selector("#classSplitGraphStatus", timeout=15000)

    assert page.eval_on_selector("#classSplitColorMode", "el => el.value") == "class"
    assert page.eval_on_selector("#classSplitGraphProjection", "el => el.value") == "class_balanced_pca"
    assert page.eval_on_selector("#classSplitDisplayMode", "el => el.value") == "all"
    assert page.eval_on_selector("#classSplitScopeAll", "el => el.checked") is True
    assert page.eval_on_selector("#classSplitScopeSelected", "el => el.checked") is False
    assert page.eval_on_selector("#classSplitGraphStatus", "el => getComputedStyle(el).display !== 'none'") is True
    assert page.locator("#classSplitColorMode option[value='cluster']").count() == 0
    assert page.locator("#classSplitClusterOverlay").count() == 0
    assert page.eval_on_selector("#classSplitClusterRun", "el => el.disabled") is True


def test_class_split_running_state_hides_previous_graph_until_result(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)

    rendered = page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_previous_job')""",
        _mock_class_split_result(),
    )
    assert rendered["tracePointCounts"] == [2, 2]
    assert page.eval_on_selector("#classSplitResults", "el => el.hidden") is False

    running = page.evaluate("() => window.__TATOR_TEST_HOOKS__.classSplitEnterRunningState()")
    assert running["resultsHidden"] is True
    assert running["progressHidden"] is False
    assert running["traceCount"] == 0
    assert running["graphText"] == ""
    assert running["statusText"] == ""
    assert page.eval_on_selector("#classSplitResults", "el => el.hidden") is True


def test_class_split_failed_start_restores_previous_graph(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)

    rendered = page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_restore_job')""",
        _mock_class_split_result(),
    )
    assert rendered["traceNames"] == ["Person", "Truck"]
    assert rendered["tracePointCounts"] == [2, 2]
    assert rendered["resultsHidden"] is False

    restored = page.evaluate(
        """async () => window.__TATOR_TEST_HOOKS__.classSplitSimulateFailedStartAfterClear('upload failed')"""
    )
    assert restored["traceNames"] == ["Person", "Truck"]
    assert restored["tracePointCounts"] == [2, 2]
    assert restored["resultsHidden"] is False
    assert "4/4 objects shown" in restored["statusText"]
    assert "Failed: upload failed" in restored["jobStatus"]
    assert "Failed: upload failed" in restored["progressText"]
    assert "No points match" not in restored["graphText"]


def test_class_split_completed_result_keeps_graph_and_class_colors_after_click(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)

    snapshot = page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_job')""",
        _mock_class_split_result(),
    )
    assert snapshot["traceNames"] == ["Person", "Truck"]
    assert snapshot["tracePointCounts"] == [2, 2]
    assert "4/4 objects shown" in snapshot["statusText"]
    assert "2 classes" in snapshot["statusText"]
    assert "No points match" not in snapshot["graphText"]
    assert len({tuple(colors) for colors in snapshot["traceColors"]}) == 2

    clicked = page.evaluate(
        """async () => window.__TATOR_TEST_HOOKS__.classSplitEmitPointClick('truck-2')"""
    )
    assert clicked["selectedPointId"] == "truck-2"
    assert clicked["traceNames"] == ["Person", "Truck"]
    assert clicked["tracePointCounts"] == [2, 2]
    assert "No points match" not in clicked["graphText"]


def test_class_split_projection_filter_and_wrong_only_transitions_keep_graph_coherent(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)

    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_transition_job')""",
        _mock_class_split_result(),
    )
    page.select_option("#classSplitGraphProjection", "global_pca")
    page.wait_for_function(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot().statusText.includes('Global PCA')",
        timeout=15000,
    )
    global_snapshot = page.evaluate("() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot()")
    assert global_snapshot["traceNames"] == ["Person", "Truck"]
    assert global_snapshot["tracePointCounts"] == [2, 2]
    assert "4/4 objects shown" in global_snapshot["statusText"]

    page.select_option("#classSplitFilterClass", "Truck")
    page.wait_for_function(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot().statusText.includes('filter: Truck')",
        timeout=15000,
    )
    filtered_snapshot = page.evaluate("() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot()")
    assert filtered_snapshot["traceNames"] == ["Truck"]
    assert filtered_snapshot["tracePointCounts"] == [2]
    assert "2/4 objects shown" in filtered_snapshot["statusText"]
    assert "1 class" in filtered_snapshot["statusText"]
    assert "No points match" not in filtered_snapshot["graphText"]

    page.select_option("#classSplitDisplayMode", "wrong_only")
    page.wait_for_function(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot().statusText.includes('likely wrong only')",
        timeout=15000,
    )
    wrong_snapshot = page.evaluate("() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot()")
    assert wrong_snapshot["traceNames"] == ["Truck"]
    assert wrong_snapshot["tracePointCounts"] == [1]
    assert "1/4 objects shown" in wrong_snapshot["statusText"]
    assert "No points match" not in wrong_snapshot["graphText"]

    page.select_option("#classSplitFilterClass", "Person")
    page.wait_for_function(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot().graphText.includes('No likely wrong-class points')",
        timeout=15000,
    )
    empty_wrong_snapshot = page.evaluate("() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot()")
    assert empty_wrong_snapshot["traceCount"] == 0
    assert empty_wrong_snapshot["traceNames"] == []
    assert "No likely wrong-class points" in empty_wrong_snapshot["graphText"]

    page.select_option("#classSplitFilterClass", "Truck")
    page.wait_for_function(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot().traceNames.includes('Truck')",
        timeout=15000,
    )

    page.select_option("#classSplitGraphProjection", "within_filter_pca")
    page.wait_for_function(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot().statusText.includes('Within-filter PCA')",
        timeout=15000,
    )
    within_filtered = page.evaluate("() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot()")
    assert within_filtered["traceNames"] == ["Truck"]
    assert within_filtered["tracePointCounts"] == [1]
    assert "No points match" not in within_filtered["graphText"]

    page.select_option("#classSplitDisplayMode", "all")
    page.select_option("#classSplitFilterClass", "")
    page.wait_for_function(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot().graphText.includes('Choose a class filter')",
        timeout=15000,
    )
    unavailable = page.evaluate("() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot()")
    assert "Choose a class filter" in unavailable["graphText"]
    assert unavailable["traceCount"] == 0


def test_class_split_wrong_candidate_confirm_removes_vignette_without_breaking_plot(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)

    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_confirm_wrong_job')""",
        _mock_class_split_result(),
    )
    page.wait_for_selector('.class-split-wrong-item[data-point-id="truck-2"]', timeout=15000)
    page.click('.class-split-wrong-item[data-point-id="truck-2"] [data-action="correct-class"]')
    page.wait_for_function(
        "() => !document.querySelector('.class-split-wrong-item[data-point-id=\"truck-2\"]')",
        timeout=15000,
    )

    snapshot = page.evaluate("() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot()")
    assert snapshot["traceNames"] == ["Person", "Truck"]
    assert snapshot["tracePointCounts"] == [2, 2]
    assert "4/4 objects shown" in snapshot["statusText"]
    assert "No points match" not in snapshot["graphText"]
    assert "No likely wrong-class objects were flagged." in (page.text_content("#classSplitWrongList") or "")


def test_class_split_wrong_candidate_skip_removes_vignette_without_clearing_flag(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)

    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_skip_wrong_job')""",
        _mock_class_split_result(),
    )
    page.wait_for_selector('.class-split-wrong-item[data-point-id="truck-2"]', timeout=15000)
    page.click('.class-split-wrong-item[data-point-id="truck-2"] [data-action="skip-wrong"]')
    page.wait_for_function(
        "() => !document.querySelector('.class-split-wrong-item[data-point-id=\"truck-2\"]')",
        timeout=15000,
    )

    page.select_option("#classSplitDisplayMode", "wrong_only")
    page.wait_for_function(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot().tracePointCounts.some((count) => count === 1)",
        timeout=15000,
    )
    snapshot = page.evaluate("() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot()")
    assert snapshot["traceCount"] >= 1
    assert sum(snapshot["tracePointCounts"]) == 1
    assert "No likely wrong-class objects were flagged." in (page.text_content("#classSplitWrongList") or "")


def test_class_split_confirm_wrong_candidate_prunes_hidden_wrong_only_selection(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)

    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_confirm_wrong_only_job')""",
        _mock_class_split_result(),
    )
    page.evaluate("""async () => window.__TATOR_TEST_HOOKS__.classSplitEmitPointClick('truck-2')""")
    page.select_option("#classSplitDisplayMode", "wrong_only")
    page.wait_for_function(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot().selectedPointId === 'truck-2'",
        timeout=15000,
    )

    page.click('.class-split-wrong-item[data-point-id="truck-2"] [data-action="correct-class"]')
    page.wait_for_function(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot().graphText.includes('No likely wrong-class points')",
        timeout=15000,
    )
    snapshot = page.evaluate("() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot()")
    assert snapshot["selectedPointId"] == ""
    assert snapshot["traceCount"] == 0
    assert snapshot["traceNames"] == []
    assert "No likely wrong-class points" in snapshot["graphText"]
    assert "No likely wrong-class objects were flagged." in (page.text_content("#classSplitWrongList") or "")
    assert "Select a point to inspect its crop." in (page.text_content("#classSplitInspector") or "")


def test_class_split_all_class_subclusters_require_class_filter(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)

    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_all_class_cluster_disabled_job')""",
        _mock_class_split_result(),
    )
    assert page.locator("#classSplitClusterOverlay").count() == 0
    assert page.locator("#classSplitColorMode option[value='cluster']").count() == 0
    assert page.eval_on_selector("#classSplitClusterRun", "el => el.disabled") is True
    assert "disabled for all-class graphs" in (page.text_content("#classSplitClusterList") or "")

    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_subcluster_job')""",
        _mock_class_split_result_with_subclusters(),
    )
    page.wait_for_function(
        """() => {
            const state = window.__TATOR_TEST_HOOKS__?.classSplitClusterDebugState?.();
            return state && state.analysisScope === 'selected_class';
        }""",
        timeout=15000,
    )
    assert page.eval_on_selector("#classSplitClusterRun", "el => !el.disabled") is True
    assert "Find subclass clusters" in (page.text_content("#classSplitClusterList") or "")

    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyClusterResult(result, 'pw_cluster_search_job')""",
        _mock_class_split_cluster_search_result(),
    )
    page.wait_for_function(
        "() => (document.querySelector('#classSplitClusterList')?.textContent || '').includes('Subclass clusters')",
        timeout=15000,
    )
    cluster_state = page.evaluate("() => window.__TATOR_TEST_HOOKS__.classSplitClusterDebugState()")
    assert cluster_state["hullsAllowed"] is False, cluster_state
    assert cluster_state["overlayDisabled"] is True, cluster_state
    assert cluster_state["proposalsAllowed"] is True, cluster_state
    assert cluster_state["clusterKeys"] == ["0", "1"], cluster_state
    page.wait_for_function(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot().statusText.includes('4/4 objects shown')",
        timeout=15000,
    )

    snapshot = page.evaluate("() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot()")
    assert snapshot["traceNames"] == ["Truck"]
    assert "Subclass cluster 0" in (page.text_content("#classSplitClusterList") or "")
    assert "Subclass cluster 1" in (page.text_content("#classSplitClusterList") or "")
    assert "No points match" not in snapshot["graphText"]

    page.select_option("#classSplitDisplayMode", "wrong_only")
    page.wait_for_function(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot().statusText.includes('likely wrong only')",
        timeout=15000,
    )
    wrong_only_cluster_state = page.evaluate("() => window.__TATOR_TEST_HOOKS__.classSplitClusterDebugState()")
    assert wrong_only_cluster_state["filteredCount"] == 0, wrong_only_cluster_state
    assert wrong_only_cluster_state["hullsAllowed"] is False, wrong_only_cluster_state
    assert wrong_only_cluster_state["overlayDisabled"] is True, wrong_only_cluster_state
    assert "No likely wrong-class points match" in page.evaluate("() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot().graphText")


def test_class_split_wrong_candidate_queue_shows_twelve_and_refills(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)

    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_many_wrong_job')""",
        _mock_class_split_many_wrong_result(15),
    )
    page.wait_for_selector(".class-split-wrong-item", timeout=15000)
    assert page.locator(".class-split-wrong-item").count() == 12
    assert "Showing 12 of 15" in (page.text_content("#classSplitWrongQueueStatus") or "")

    first_id = page.eval_on_selector(".class-split-wrong-item", "el => el.getAttribute('data-point-id')")
    page.click(f'.class-split-wrong-item[data-point-id="{first_id}"] [data-action="skip-wrong"]')
    page.wait_for_function(
        """(pointId) => !document.querySelector(`.class-split-wrong-item[data-point-id="${pointId}"]`)""",
        arg=first_id,
        timeout=15000,
    )
    assert page.locator(".class-split-wrong-item").count() == 12
    assert "Showing 12 of 14" in (page.text_content("#classSplitWrongQueueStatus") or "")

    page.click("#classSplitWrongShuffle")
    page.wait_for_selector(".class-split-wrong-item", timeout=15000)
    assert page.locator(".class-split-wrong-item").count() == 12


def test_class_split_graph_survives_leaving_and_returning_to_tab(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)

    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_tab_return_job')""",
        _mock_class_split_result(),
    )
    page.select_option("#classSplitGraphProjection", "global_pca")
    page.select_option("#classSplitFilterClass", "Truck")
    page.select_option("#classSplitDisplayMode", "wrong_only")
    clicked = page.evaluate(
        """async () => window.__TATOR_TEST_HOOKS__.classSplitEmitPointClick('truck-2')"""
    )
    assert clicked["selectedPointId"] == "truck-2"
    assert clicked["traceNames"] == ["Truck"]
    assert clicked["tracePointCounts"] == [1]

    go_to_tab(page, "#tabDataIngestionButton", "#tabDataIngestion")
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot().statusText.includes('likely wrong only')",
        timeout=15000,
    )
    returned = page.evaluate("() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot()")
    assert returned["selectedPointId"] == "truck-2"
    assert returned["traceNames"] == ["Truck"]
    assert returned["tracePointCounts"] == [1]
    assert "1/4 objects shown" in returned["statusText"]
    assert "Global PCA" in returned["statusText"]
    assert "filter: Truck" in returned["statusText"]
    assert "No points match" not in returned["graphText"]


def test_class_split_legacy_pca_result_defaults_to_global_pca(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)

    legacy_result = _mock_class_split_result()
    legacy_result["summary"].pop("projection_mode", None)
    legacy_result["summary"]["projection"] = "pca"
    legacy_result["projection_options"] = {}

    snapshot = page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_legacy_job')""",
        legacy_result,
    )
    assert snapshot["traceNames"] == ["Person", "Truck"]
    assert snapshot["tracePointCounts"] == [2, 2]
    assert "Global PCA" in snapshot["statusText"]
    assert "Class-balanced PCA" not in snapshot["statusText"]
    assert "No points match" not in snapshot["graphText"]


def test_class_split_unannotated_legacy_result_defaults_to_global_pca(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)

    legacy_result = _mock_class_split_result()
    legacy_result["summary"].pop("projection_mode", None)
    legacy_result["summary"].pop("projection", None)
    legacy_result.pop("projection_options", None)

    snapshot = page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_unannotated_legacy_job')""",
        legacy_result,
    )
    assert snapshot["traceNames"] == ["Person", "Truck"]
    assert snapshot["tracePointCounts"] == [2, 2]
    assert "Global PCA" in snapshot["statusText"]
    assert "Class-balanced PCA" not in snapshot["statusText"]
    assert "No points match" not in snapshot["graphText"]


def test_class_split_legacy_all_class_result_still_plots_with_metric_color(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)

    legacy_result = _mock_class_split_result()
    legacy_result["summary"].pop("projection_mode", None)
    legacy_result["summary"]["projection"] = "pca"
    legacy_result["projection_options"] = None
    for idx, point in enumerate(legacy_result["points"], start=1):
        point["width"] = 10 * idx
        point["height"] = 12 * idx

    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_legacy_metric_job')""",
        legacy_result,
    )
    page.select_option("#classSplitColorMode", "area")
    page.wait_for_function(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot().statusText.includes('colored by box area')",
        timeout=15000,
    )
    snapshot = page.evaluate("() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot()")
    assert snapshot["traceNames"] == ["Objects", "Likely wrong class"]
    assert snapshot["tracePointCounts"] == [3, 1]
    assert "4/4 objects shown" in snapshot["statusText"]
    assert "Global PCA" in snapshot["statusText"]
    assert "No points match" not in snapshot["graphText"]


def test_class_split_pipboy_theme_keeps_plot_and_class_colors(playwright_page):
    page, _ = playwright_page
    page.evaluate(
        """() => {
            window.localStorage.setItem('tator.themeMode', 'pipboy');
            window.localStorage.setItem('tator.pipboyAccent', 'green');
        }"""
    )
    page.reload(wait_until="domcontentloaded")
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("document.documentElement.classList.contains('theme-pipboy')", timeout=15000)
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)

    snapshot = page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_pipboy_job')""",
        _mock_class_split_result(),
    )
    assert snapshot["traceNames"] == ["Person", "Truck"]
    assert snapshot["tracePointCounts"] == [2, 2]
    assert "4/4 objects shown" in snapshot["statusText"]
    assert "No points match" not in snapshot["graphText"]
    assert len({tuple(colors) for colors in snapshot["traceColors"]}) == 2
    assert page.eval_on_selector("#classSplitGraph", "el => getComputedStyle(el).display !== 'none'") is True

    page.evaluate(
        """() => {
            window.localStorage.setItem('tator.themeMode', 'light');
            window.localStorage.setItem('tator.darkMode', '0');
        }"""
    )
    page.reload(wait_until="domcontentloaded")
