import base64
import json
from pathlib import Path

import pytest

from .helpers.ui import go_to_tab


pytestmark = [pytest.mark.ui, pytest.mark.ui_full]

_SELECTOR_MODEL_DIGEST = json.loads(
    (Path(__file__).resolve().parents[3]
     / "models"
     / "class_analysis_selector_v6_default.json").read_text(encoding="utf-8")
)["model_digest"]


def _fulfill_review_disposition(route):
    payload = route.request.post_data_json or {}
    url_parts = route.request.url.split("/")
    job_id = url_parts[url_parts.index("jobs") + 1]
    point_id = url_parts[url_parts.index("points") + 1]
    disposition = str(payload.get("disposition") or "")
    route.fulfill(
        status=200,
        content_type="application/json",
        body=json.dumps(
            {
                "schema": "class-analysis-review-disposition-v3",
                "status": "recorded",
                "job_id": job_id,
                "point_id": point_id,
                "disposition": disposition,
                "client_action_id": payload.get("client_action_id", ""),
                "training_capture_requested": payload.get("capture_training_data") is True,
                "review_object_key": "cro_" + ("a" * 64),
                "human_reviewed_at": 1_785_800_000.0,
                "human_review_revision": "rdr1_" + ("1" * 32),
            }
        ),
    )


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


def _mock_active_workspace_dual_bbox_result():
    result = _mock_class_split_result()
    current_bbox = [557.00016, 720.0, 633.00048, 751.00032]
    other_bbox = [554.99952, 721.00032, 630.99984, 751.00032]
    conflict = {
        "enabled": True,
        "kind": "near_identical_cross_class_bbox",
        "review_mode": "dual_bbox_annotation_resolution",
        "point_id": "truck-2",
        "other_point_id": "person-1",
        "current_class": "Truck",
        "other_class_name": "LightVehicle",
        "split": "train",
        "image_relpath": "pair.png",
        "target_bbox_xyxy": current_bbox,
        "other_bbox_xyxy": other_bbox,
        "iou": 0.92,
        "target_area_covered": 0.94,
    }
    result["summary"].update(
        {
            "source_mode": "active_workspace",
            "source_id": "cas_playwright_active_workspace",
        }
    )
    result["points"][1].update(
        {
            "source_mode": "active_workspace",
            "source_id": "cas_playwright_active_workspace",
            "split": "train",
            "image_relpath": "pair.png",
            "frontend_image_key": "pair.png",
            "bbox_xyxy": current_bbox,
            "dual_bbox_conflict": conflict,
        }
    )
    result["points"][2].update(
        {
            "point_id": "person-1",
            "class_name": "LightVehicle",
            "source_mode": "active_workspace",
            "source_id": "cas_playwright_active_workspace",
            "split": "train",
            "image_relpath": "pair.png",
            "frontend_image_key": "pair.png",
            "bbox_xyxy": other_bbox,
        }
    )
    result["wrong_class_candidates"][0].update(
        {
            "source_mode": "active_workspace",
            "source_id": "cas_playwright_active_workspace",
            "split": "train",
            "image_relpath": "pair.png",
            "class_name": "Truck",
            "suggested_neighbor_class": "LightVehicle",
            "dual_bbox_conflict": conflict,
        }
    )
    return result


def _mock_active_workspace_single_bbox_result():
    result = _mock_class_split_result()
    target_bbox = [100.25, 120.5, 180.75, 220.75]
    result["summary"].update(
        {
            "source_mode": "active_workspace",
            "source_id": "cas_playwright_single_bbox",
        }
    )
    result["points"][1].update(
        {
            "source_mode": "active_workspace",
            "source_id": "cas_playwright_single_bbox",
            "split": "train",
            "image_relpath": "single.png",
            "frontend_image_key": "single.png",
            "bbox_xyxy": target_bbox,
        }
    )
    result["wrong_class_candidates"][0].update(
        {
            "source_mode": "active_workspace",
            "source_id": "cas_playwright_single_bbox",
            "split": "train",
            "image_relpath": "single.png",
            "bbox_xyxy": target_bbox,
        }
    )
    return result


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


def test_cancelled_refinement_loads_preserved_stage_one_but_early_cancel_does_not(
    playwright_page,
):
    page, _ = playwright_page
    preserved = _mock_class_split_result()
    preserved["summary"]["refinement"] = {
        "enabled": True,
        "status": "cancelled",
        "warnings": ["refinement_cancelled"],
    }
    preserved["refinement_summary"] = preserved["summary"]["refinement"]
    preserved["refinement_candidates"] = []
    preserved["vignette_candidates"] = []

    page.route(
        "**/class_analysis/jobs/cancel-with-fallback/result",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(preserved),
        ),
    )
    page.route(
        "**/class_analysis/jobs/cancel-with-fallback",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=(
                '{"job_id":"cancel-with-fallback","status":"cancelled",'
                '"summary":{"refinement":{"status":"cancelled"}}}'
            ),
        ),
    )
    page.route(
        "**/class_analysis/jobs/cancel-early/result",
        lambda route: route.fulfill(
            status=409,
            content_type="application/json",
            body='{"detail":"class_analysis_result_not_ready"}',
        ),
    )
    page.route(
        "**/class_analysis/jobs/cancel-early",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"job_id":"cancel-early","status":"cancelled"}',
        ),
    )
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function(
        "!!window.__TATOR_TEST_HOOKS__?.classSplitPollJob",
        timeout=15000,
    )

    preserved_snapshot = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPollJob('cancel-with-fallback')"
    )
    assert preserved_snapshot["traceCount"] == 2
    assert "preserved Stage-1 results" in page.locator(
        "#classSplitJobStatus"
    ).inner_text()

    page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitEnterRunningState()"
    )
    early_snapshot = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPollJob('cancel-early')"
    )
    assert early_snapshot["traceCount"] == 0
    assert page.locator("#classSplitJobStatus").inner_text() == "Cancelled"


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


def _mock_refined_class_split_result(*, selected_class=False):
    result = _mock_class_split_result()
    result["summary"].update(
        {
            "analysis_scope": "selected_class" if selected_class else "all_classes",
            "class_name": "Truck" if selected_class else "",
            "object_count": 5,
            "class_counts": {"Truck": 3, "Person": 2},
            "source_mode": "active_workspace",
            "refinement": {
                "enabled": True,
                "status": "completed",
                "quality_status": "actionable",
                "schema": "class-analysis-patch-refinement-v3",
                "decision_contract": "class-analysis-patch-decision-v4",
                "observability_contract": "class-analysis-refinement-observability-v3.1",
                "triage_semantics": "class-analysis-terminal-triage-v3.1",
                "rough_candidate_count": 5,
                "evaluated_candidate_count": 5,
                "primary_candidate_count": 1,
                "quality_gate": {
                    "passed": True,
                    "reasons": [],
                    "thresholds": {
                        "minimum_resolved_rate": 0.5,
                        "minimum_confirmation_eligible_pair_coverage": 0.75,
                    },
                    "metrics": {
                        "regular_evaluated_count": 4,
                        "resolved_count": 3,
                        "resolved_rate": 0.75,
                        "terminal_decisive_count": 2,
                        "terminal_decisive_rate": 0.5,
                        "unresolved_count": 1,
                        "unresolved_rate": 0.25,
                        "confirmation_eligible_pair_coverage": 0.75,
                    },
                },
                "triage_metrics": {
                    "regular_evaluated_count": 4,
                    "resolved_count": 3,
                    "resolved_rate": 0.75,
                    "terminal_decisive_count": 2,
                    "terminal_decisive_rate": 0.5,
                    "unresolved_count": 1,
                    "unresolved_rate": 0.25,
                    "pair_conflict_count": 1,
                },
                "pair_coverage": {
                    "confirmation_eligible_candidate_count": 3,
                    "confirmation_eligible_candidate_coverage": 0.75,
                },
                "queue_policy": {
                    "automatic_rough_fallback": False,
                    "fallback_reason": "",
                    "default_queue": "confirmed_outliers",
                    "confirmed_count": 1,
                    "rough_count": 5,
                },
            },
        }
    )
    extra_point = {
        "point_id": "truck-pair",
        "class_name": "Truck",
        "image_relpath": "img_pair.png",
        "projection": [0.2, -0.4],
        "wrong_class_suspicion": 0.81,
        "is_wrong_class_candidate": not selected_class,
        "is_rough_outlier_candidate": True,
        "candidate_kind": "pair_conflict",
        "refined_outlier": {
            "status": "pair_conflict",
            "current_support": 0.71,
            "alternative_class": "Person",
            "alternative_support": 0.74,
        },
        "dual_bbox_conflict": {
            "enabled": True,
            "current_class": "Truck",
            "other_class_name": "Person",
            "iou": 0.92,
            "target_area_covered": 0.94,
        },
    }
    result["points"].append(extra_point)
    for coordinates in result["projection_options"]["coordinates"].values():
        coordinates.append([0.2, -0.4])

    statuses = [
        ("truck-2", "confirmed_outlier", 0.18, 0.91),
        ("person-1", "explained_not_outlier", 0.93, 0.37),
        ("person-2", "mixed_or_composite", 0.84, 0.81),
        ("truck-1", "unresolved", 0.41, 0.44),
    ]
    candidates = []
    for sidecar_row, (
        point_id,
        status,
        current_support,
        alternative_support,
    ) in enumerate(statuses):
        point = next(point for point in result["points"] if point["point_id"] == point_id)
        point.update(
            {
                "is_rough_outlier_candidate": True,
                "candidate_kind": (
                    "within_class_anomaly" if selected_class else "wrong_class"
                ),
                "include_in_refined_vignettes": status == "confirmed_outlier",
                "refined_outlier": {
                    "status": status,
                    "schema": "class-analysis-patch-refinement-v3",
                    "decision_contract": "class-analysis-patch-decision-v4",
                    "intrinsic_current_support": current_support,
                    "alternative_class": "Person",
                    "intrinsic_alternative_support": alternative_support,
                    "directed_pair_raw_margin": alternative_support - current_support,
                    "directed_pair_margin": alternative_support - current_support,
                    "directed_pair_probe_score": alternative_support,
                    "directed_pair_probe_threshold": 0.6,
                    "directed_pair_probe_weights": [-0.7, 0.7],
                    "directed_pair_probe_contract": "source-cross-fitted-angle-grid-l2-sign-v1",
                    "directed_pair_probe_fold_count": 5,
                    "directed_pair_probe_fit_status": "fitted",
                    "directed_pair_probe_fold_digest": "ab" * 32,
                    "directed_pair_reliable": status != "unresolved",
                    "directed_pair_heldout_auroc": 0.82,
                    "reason_codes": [f"fixture_{status}"],
                    "sidecar_row": sidecar_row,
                },
            }
        )
        if not selected_class:
            point["is_wrong_class_candidate"] = True
        candidates.append(dict(point))
    candidates.append(dict(extra_point))
    result["refinement_candidates"] = candidates
    result["vignette_candidates"] = [dict(candidates[0])]
    result["wrong_class_candidates"] = [] if selected_class else list(candidates)
    result["within_class_outlier_candidates"] = list(candidates) if selected_class else []
    return result


def _mock_v6_refined_class_split_result(*, selected_class=False):
    result = _mock_refined_class_split_result(selected_class=selected_class)
    refinement = result["summary"]["refinement"]
    selector_contract = (
        "expected-review-utility-global-boosted-statistical-overlap-rerank-v7"
    )
    base_model_selector_contract = (
        "expected-review-utility-global-boosted-gated-dataset-overlap-v6"
    )
    feature_contract = (
        "raw-stage1-patch-same-image-and-gated-dataset-overlap-features-v3"
    )
    utility_policy_contract = (
        "actionability-times-75pct-base-plus-25pct-reviewability-v1"
    )
    dataset_overlap_application_contract = (
        "affirmative-current-localized-material-source-loo-v2"
    )
    dataset_overlap_diagnostic_contract = (
        "capture-loo-beta-wilson-shrunk-rank-only-v1"
    )
    model_digest = _SELECTOR_MODEL_DIGEST
    refinement.update(
        {
            "schema": "class-analysis-patch-refinement-v5",
            "decision_contract": "class-analysis-patch-decision-v9",
            "human_review_qualification_contract": (
                "class-analysis-qualified-human-review-v1"
            ),
            "human_review_rank_contract": (
                "confirmed-band-stage1-suspicion-probe-excess-v1"
            ),
            "selector_priority_contract": selector_contract,
            "selector_feature_contract": feature_contract,
            "selector_model_digest": model_digest,
            "selector_utility_policy_contract": utility_policy_contract,
            "selector_dataset_overlap_application_contract": (
                dataset_overlap_application_contract
            ),
            "selector_priority_candidate_count": 5,
            "qualified_human_review_candidate_count": 2,
            "observability_contract": (
                "class-analysis-refinement-observability-v3.3"
            ),
            "triage_semantics": "class-analysis-terminal-triage-v3.3",
            "queue_policy": {
                "automatic_rough_fallback": False,
                "fallback_reason": "",
                "mode": "selector_ranked_complete_stage1",
                "default_queue": "selector_ranked_stage1_candidates",
                "confirmed_count": 1,
                "pair_conflict_count": 1,
                "refined_review_candidate_count": 2,
                "effective_default_candidate_count": 2,
                "rough_count": 5,
            },
        }
    )

    selector_rows = {
        "truck-2": {
            "rank": 1,
            "actionability": 0.90,
            "reviewability": 0.80,
            "current_state": "absent",
            "alternative_state": "present",
            "overlap_state": "external",
            "dataset_overlap_applicable": False,
        },
        "truck-1": {
            "rank": 2,
            "actionability": 0.82,
            "reviewability": 0.70,
            "current_state": "indeterminate",
            "alternative_state": "present",
            "overlap_state": "none",
            "dataset_overlap_applicable": False,
        },
        "truck-pair": {
            "rank": 3,
            "actionability": 0.74,
            "reviewability": 0.60,
            "current_state": "present",
            "alternative_state": "present",
            "overlap_state": "duplicate_conflict",
            "dataset_overlap_applicable": False,
        },
        "person-2": {
            "rank": 4,
            "actionability": 0.52,
            "reviewability": 0.80,
            "current_state": "present",
            "alternative_state": "present",
            "overlap_state": "localized",
            "dataset_overlap_applicable": True,
        },
        "person-1": {
            "rank": 5,
            "actionability": 0.20,
            "reviewability": 0.90,
            "current_state": "present",
            "alternative_state": "indeterminate",
            "overlap_state": "localized",
            "dataset_overlap_applicable": False,
        },
    }
    qualified_ranks = {"truck-2": 1, "truck-1": 2}
    canonical_gates = {
        "directed_pair_reliable": True,
        "directed_pair_candidate_source_independent": True,
        "directed_pair_exact_calibration_contracts": True,
        "intrinsic_references_reliable": True,
        "positive_confirmation_pair_reliable": True,
        "positive_confirmation_pair_probe_auroc_sufficient": True,
        "positive_confirmation_pair_probe_lower_bound_sufficient": True,
        "source_resolution_sufficient": True,
        "current_absent": True,
        "alternative_strong": True,
        "directed_pair_dominates": True,
        "alternative_exclusive_component_corresponds": True,
        "view_consistent": True,
        "alternative_evidence_external_to_overlap": True,
    }
    split_digest = "cd" * 32
    for point in result["points"]:
        evidence = point.get("refined_outlier")
        selector_row = selector_rows.get(point["point_id"])
        if not isinstance(evidence, dict) or selector_row is None:
            continue

        point_id = point["point_id"]
        rank = selector_row["rank"]
        actionability = selector_row["actionability"]
        reviewability = selector_row["reviewability"]
        utility = actionability * (0.75 + 0.25 * reviewability)
        mislabeled = actionability * 0.70
        geometry = actionability - mislabeled
        dataset_overlap_applicable = selector_row[
            "dataset_overlap_applicable"
        ]
        directed_pair = (
            f"{point['class_name']}->{evidence.get('alternative_class') or 'Person'}"
        )
        evidence.update(
            {
                "schema": "class-analysis-patch-refinement-v5",
                "decision_contract": "class-analysis-patch-decision-v9",
                "selector_priority_contract": selector_contract,
                "selector_priority_base_rank": rank,
                "selector_priority_rank": rank,
                "selector_priority_base_score": utility,
                "selector_priority_semantic_overlap_adjustment": 0.0,
                "selector_priority_triage_frequency_adjustment": 0.0,
                "selector_priority_overlap_adjustment": 0.0,
                "selector_priority_score": utility,
                "selector_priority_status_band_index": 0,
                "selector_priority_status_band_name": (
                    "expected_review_utility"
                ),
                "selector_priority_band_base_rank": rank,
                "selector_priority_band_rank": rank,
                "selector_priority_band_candidate_count": 5,
                "selector_priority_base_components": {
                    "actionable_probability": actionability,
                    "reviewability_probability": reviewability,
                    "utility_reviewability_floor": 0.75,
                    "utility_reviewability_weight": 0.25,
                    "base_expected_review_utility": utility,
                },
                "selector_priority_reasons": [
                    "learned_expected_review_utility"
                ],
                "selector_v6": {
                    "selector_contract": selector_contract,
                    "base_model_selector_contract": (
                        base_model_selector_contract
                    ),
                    "feature_contract": feature_contract,
                    "model_digest": model_digest,
                    "utility_policy_contract": utility_policy_contract,
                    "dataset_overlap_application_contract": (
                        dataset_overlap_application_contract
                    ),
                    "dataset_overlap_diagnostic_contract": (
                        dataset_overlap_diagnostic_contract
                    ),
                    "dataset_overlap_scoring_effect_enabled": True,
                    "global_actionability_model_contract": (
                        "pair-blind-shallow-histogram-gradient-boosting-v1"
                    ),
                    "expected_review_utility": utility,
                    "base_expected_review_utility": utility,
                    "actionable_probability": actionability,
                    "reviewability_probability": reviewability,
                    "conditional_annotation_state": {
                        "mislabeled": mislabeled,
                        "actionable_geometry_or_composite": geometry,
                        "valid_or_harmless": 1.0 - actionability,
                    },
                    "insufficient_evidence_probability": (
                        1.0 - reviewability
                    ),
                    "current_evidence_state": (
                        selector_row["current_state"]
                    ),
                    "alternative_evidence_state": (
                        selector_row["alternative_state"]
                    ),
                    "overlap_evidence_state": (
                        selector_row["overlap_state"]
                    ),
                    "same_image_context": {
                        "available": True,
                        "image_object_count": 6,
                        "same_class_count": 3,
                        "trusted_same_class_anchor_count": 2,
                        "trusted_alternative_class_anchor_count": 2,
                        "bbox_width_norm": 0.18,
                        "bbox_height_norm": 0.25,
                        "bbox_area_fraction": 0.045,
                        "same_class_peer_width_median_norm": 0.20,
                        "same_class_peer_height_median_norm": 0.27,
                        "same_class_peer_area_median_fraction": 0.054,
                        "same_class_log_width_residual": -0.10,
                        "same_class_log_height_residual": -0.08,
                        "same_class_log_area_residual": -0.18,
                        "perspective_log_scale_residual": -0.12,
                    },
                    "dataset_overlap": {
                        "application_contract": (
                            dataset_overlap_application_contract
                        ),
                        "diagnostic_contract": (
                            dataset_overlap_diagnostic_contract
                        ),
                        "available": True,
                        "applicable": dataset_overlap_applicable,
                        "application_reason": (
                            "eligible_dataset_overlap_explanation"
                            if dataset_overlap_applicable
                            else "material_annotated_overlap_absent"
                        ),
                        "directed_pair": directed_pair,
                        "geometry_stratum": "material_nonduplicate",
                        "affirmative_current_evidence": (
                            selector_row["current_state"] == "present"
                        ),
                        "candidate_overlap": dataset_overlap_applicable,
                        "candidate_overlap_count": (
                            2 if dataset_overlap_applicable else 0
                        ),
                        "candidate_overlap_strength": (
                            0.74 if dataset_overlap_applicable else 0.0
                        ),
                        "scale_typicality": 0.68,
                        "fit_screening_adjustment_eligible": True,
                        "reliable": True,
                        "reliability_tier": "trusted",
                        "source_independence_verified": True,
                        "smoothed_capture_group_incidence": 0.42,
                        "conservative_strength": 0.28,
                        "capture_group_incidence_lower_bound": 0.19,
                        "eligible_capture_group_count": 40,
                        "overlap_capture_group_count": 16,
                        "model_values": {},
                        "model_available": False,
                        "scoring_effect_enabled": True,
                        "rank_only": True,
                        "uses_human_review_labels": False,
                        "applied": False,
                        "blend_policy": "diagnostic-only-v1",
                        "blend_weight": 0.0,
                        "overlap_expert_probability": actionability,
                        "counterfactual_actionable_probability": actionability,
                        "actionable_probability": actionability,
                        "probability_delta": 0.0,
                        "counterfactual_expected_review_utility": utility,
                        "base_expected_review_utility": utility,
                        "expected_review_utility": utility,
                        "utility_delta": 0.0,
                        "rank_discount_fraction": 0.0,
                        "maximum_rank_discount_fraction": 0.25,
                    },
                    "global_model": {
                        "contract": (
                            "pair-blind-shallow-histogram-gradient-boosting-v1"
                        ),
                        "raw_margin": 0.5,
                        "actionable_probability": actionability,
                    },
                },
            }
        )
        if evidence.get("status") == "pair_conflict":
            continue
        current = float(evidence.get("intrinsic_current_support") or 0.10)
        alternative = float(
            evidence.get("intrinsic_alternative_support") or 0.38284271
        )
        evidence.update(
            {
                "directed_pair_probe_score": (
                    -0.70710678 * current + 0.70710678 * alternative
                ),
                "directed_pair_probe_features": [current, alternative],
                "directed_pair_probe_feature_names": [
                    "current_patch_exclusive_support",
                    "alternative_patch_exclusive_support",
                ],
                "directed_pair_current_exclusive_support": current,
                "directed_pair_alternative_exclusive_support": alternative,
                "directed_pair_probe_threshold": 0.12,
                "directed_pair_probe_weights": [-0.70710678, 0.70710678],
                "directed_pair_probe_contract": (
                    "source-disjoint-exact-two-view-paired-exclusive-fit-"
                    "thresholds-angle-grid-l2-sign-v4"
                ),
                "directed_pair_probe_view_contract": (
                    "tight-context-weighted-target-mass-paired-exclusive-"
                    "mean-v1"
                ),
                "directed_pair_probe_lower_bound_contract": (
                    "hanley-mcneil-shrunk-two-sided-95-v1"
                ),
                "directed_pair_probe_fold_count": 1,
                "directed_pair_probe_fit_status": "ok",
                "directed_pair_probe_fold_digest": split_digest,
                "directed_pair_probe_fit_eval_split_digest": split_digest,
                "directed_pair_bank_reliable": True,
                "directed_pair_candidate_source_excluded": False,
                "directed_pair_candidate_source_fingerprint": "3" * 16,
                "directed_pair_candidate_source_membership_roles": [],
                "directed_pair_heldout_auroc": 0.82,
                "directed_pair_eval_auroc_lower_bound": 0.65,
                "directed_pair_probe_fit_current_source_count": 8,
                "directed_pair_probe_fit_alternative_source_count": 8,
                "directed_pair_probe_eval_current_source_count": 8,
                "directed_pair_probe_eval_alternative_source_count": 8,
                "directed_pair_probe_fit_balanced_accuracy": 0.75,
                "directed_pair_probe_eval_sensitivity": 0.75,
                "directed_pair_probe_eval_specificity": 0.75,
                "directed_pair_current_absence_eval_fraction": 0.75,
                "directed_pair_alternative_strong_eval_fraction": 0.75,
                "positive_confirmation_pair_probe_auroc_floor": 0.80,
                "positive_confirmation_pair_probe_auroc_lower_bound_floor": (
                    0.60
                ),
                "current_negative_threshold": 0.05,
                "current_support_threshold": 0.15,
                "current_strong_threshold": 0.25,
                "alternative_negative_threshold": 0.05,
                "alternative_support_threshold": 0.15,
                "alternative_strong_threshold": 0.25,
                "support_threshold_source": "fit_only_directed_pair",
                "human_review_qualification_contract": (
                    "class-analysis-qualified-human-review-v1"
                ),
                "human_review_rank_contract": (
                    "confirmed-band-stage1-suspicion-probe-excess-v1"
                ),
                "qualified_for_human_review": point_id in qualified_ranks,
                "human_review_rank": qualified_ranks.get(point_id),
                "decision_gates": {
                    **canonical_gates,
                    "qualified_for_human_review": (
                        point_id in qualified_ranks
                    ),
                },
            }
        )

    current_state_counts = {}
    for row in selector_rows.values():
        state = row["current_state"]
        current_state_counts[state] = current_state_counts.get(state, 0) + 1
    refinement["selector_priority"] = {
        "contract": selector_contract,
        "candidate_count": 5,
        "ranked_candidate_count": 5,
        "base_order_contract": (
            "expected-review-utility-before-statistical-overlap-then-point-id-v1"
        ),
        "base_score_contract": (
            "expected-review-utility-before-statistical-overlap-v1"
        ),
        "semantic_status_tiebreak_contract": (
            "none-global-expected-utility-order-v1"
        ),
        "status_band_score_gap": 0.0,
        "maximum_overlap_adjustment_bound": 0.25,
        "unique_contiguous_ranks": True,
        "higher_score_is_higher_priority": True,
        "status_band_partitioned": False,
        "cross_status_band_reordering": True,
        "status_band_order": ["expected_review_utility"],
        "status_band_counts": {"expected_review_utility": 5},
        "adjusted_candidate_count": 0,
        "semantic_adjusted_candidate_count": 0,
        "triage_adjusted_candidate_count": 0,
        "dataset_overlap_applied_candidate_count": 0,
        "prior_evaluation_failure_count": 0,
        "prior_evaluation_failures": [],
        "prior_evaluation_failure_digest": "0" * 64,
        "quality_gate": {"passed": True, "reasons": []},
        "changes_candidate_membership": False,
        "changes_semantic_status": False,
        "suppresses_candidates": False,
        "utility_model": {
            "contract": selector_contract,
            "base_model_selector_contract": base_model_selector_contract,
            "feature_contract": feature_contract,
            "model_schema": "class-analysis-selector-utility-model-v2",
            "model_digest": model_digest,
            "utility_policy_contract": utility_policy_contract,
            "dataset_overlap_application_contract": (
                dataset_overlap_application_contract
            ),
            "dataset_overlap_diagnostic_contract": (
                dataset_overlap_diagnostic_contract
            ),
            "dataset_overlap_scoring_effect_enabled": True,
            "global_actionability_model_contract": (
                "pair-blind-shallow-histogram-gradient-boosting-v1"
            ),
            "model_family": "pair-blind-global-hgb",
            "candidate_count": 5,
            "context_available_count": 5,
            "current_evidence_state_counts": current_state_counts,
            "dataset_overlap": {
                "application_contract": (
                    dataset_overlap_application_contract
                ),
                "diagnostic_contract": (
                    dataset_overlap_diagnostic_contract
                ),
                "scoring_effect_enabled": True,
                "rank_only": True,
                "uses_human_review_labels": False,
                "maximum_rank_discount_fraction": 0.25,
                "available_candidate_count": 5,
                "applicable_candidate_count": 1,
                "applied_candidate_count": 0,
                "effect_candidate_count": 0,
                "application_reason_counts": {
                    "eligible_dataset_overlap_explanation": 1,
                    "material_annotated_overlap_absent": 4,
                },
                "demoted_candidate_count": 0,
                "promoted_candidate_count": 0,
                "zero_effect_candidate_count": 5,
                "maximum_absolute_probability_effect": 0.0,
                "mean_absolute_probability_effect": 0.0,
                "maximum_absolute_utility_effect": 0.0,
                "mean_absolute_utility_effect": 0.0,
            },
            "changes_candidate_membership": False,
            "changes_semantic_status": False,
            "mutates_annotations": False,
        },
    }

    by_id = {point["point_id"]: point for point in result["points"]}
    for collection_name in (
        "refinement_candidates",
        "wrong_class_candidates",
        "within_class_outlier_candidates",
    ):
        for row in result.get(collection_name, []):
            source = by_id.get(row.get("point_id"), {})
            if isinstance(source.get("refined_outlier"), dict):
                row["refined_outlier"] = dict(source["refined_outlier"])
    result["vignette_candidates"] = [
        dict(row)
        for row in result["refinement_candidates"]
        if row.get("refined_outlier", {}).get("status")
        in {"confirmed_outlier", "pair_conflict"}
    ]
    return result


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


def test_class_split_accepted_replacement_terminal_failures_restore_previous_graph(playwright_page):
    page, _ = playwright_page

    page.route(
        "**/class_analysis/jobs/pw_replacement_failed",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"status":"failed","progress":1,"error":"synthetic replacement failure"}',
        ),
    )
    page.route(
        "**/class_analysis/jobs/pw_replacement_missing",
        lambda route: route.fulfill(
            status=404,
            content_type="application/json",
            body='{"detail":"class_analysis_job_not_found"}',
        ),
    )
    page.route(
        "**/class_analysis/jobs/pw_replacement_cancelled/result",
        lambda route: route.fulfill(
            status=404,
            content_type="application/json",
            body='{"detail":"class_analysis_result_not_found"}',
        ),
    )
    page.route(
        "**/class_analysis/jobs/pw_replacement_cancelled",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"status":"cancelled","progress":1,"message":"cancelled before Stage 1"}',
        ),
    )
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function(
        "!!window.__TATOR_TEST_HOOKS__?.classSplitPollReplacementJob",
        timeout=15000,
    )
    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            result,
            'pw_class_split_terminal_restore_source'
        )""",
        _mock_class_split_result(),
    )

    cases = [
        ("pw_replacement_failed", "Failed: synthetic replacement failure"),
        (
            "pw_replacement_missing",
            "Replacement analysis is unavailable; restored the previous completed analysis.",
        ),
        (
            "pw_replacement_cancelled",
            "Cancelled; restored the previous completed analysis.",
        ),
    ]
    for job_id, expected_status in cases:
        snapshot = page.evaluate(
            """async (jobId) => window.__TATOR_TEST_HOOKS__.classSplitPollReplacementJob(jobId)""",
            job_id,
        )
        assert snapshot["traceNames"] == ["Person", "Truck"]
        assert snapshot["tracePointCounts"] == [2, 2]
        assert snapshot["resultsHidden"] is False
        assert expected_status in snapshot["jobStatus"]
        assert "No points match" not in snapshot["graphText"]


def test_class_split_obsolete_memory_limiter_message_and_checkpoint_recovery(
    playwright_page,
):
    page, _ = playwright_page
    checkpoint = {
        "used": True,
        "checkpoint_schema": "class-analysis-embedding-checkpoint-v1",
        "checkpoint_sha256": "a" * 64,
        "record_count": 4,
    }
    page.route(
        "**/class_analysis/jobs/pw_checkpoint_recovery",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "running",
                    "progress": 0.73,
                    "message": "Projecting and scoring graph ...",
                    "runtime": {"embedding_resume": checkpoint},
                }
            ),
        ),
    )
    page.route(
        "**/class_analysis/jobs/pw_memory_checkpointed",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "failed",
                    "progress": 1.0,
                    "error": "class_analysis_memory_budget_exceeded",
                    "runtime": {
                        "embedding_resume": {**checkpoint, "used": False}
                    },
                }
            ),
        ),
    )
    page.route(
        "**/class_analysis/jobs/pw_memory_unknown_checkpoint",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "failed",
                    "progress": 1.0,
                    "error": "class_analysis_memory_budget_exceeded",
                }
            ),
        ),
    )
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function(
        "!!window.__TATOR_TEST_HOOKS__?.classSplitPollJob",
        timeout=15000,
    )

    recovered = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPollJob('pw_checkpoint_recovery')"
    )
    assert (
        "Recovered a validated adjusted-embedding checkpoint for 4 objects"
        in recovered["jobStatus"]
    )
    assert "Projecting and scoring graph" in recovered["jobStatus"]
    assert "Recovered a validated adjusted-embedding checkpoint" in recovered[
        "progressText"
    ]

    checkpointed_failure = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPollJob('pw_memory_checkpointed')"
    )
    assert "older analysis was stopped by a memory limiter" in checkpointed_failure["jobStatus"]
    assert "has now been removed" in checkpointed_failure["jobStatus"]
    assert "not terminated by a configured memory ceiling" in checkpointed_failure["jobStatus"]
    assert "No labels were changed" in checkpointed_failure["jobStatus"]
    assert (
        "Press Rerun with the same dataset and settings"
        in checkpointed_failure["jobStatus"]
    )
    assert "validated adjusted-embedding checkpoint" in checkpointed_failure["jobStatus"]
    assert "class_analysis_memory_budget_exceeded" not in checkpointed_failure[
        "jobStatus"
    ]

    unknown_checkpoint_failure = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPollJob('pw_memory_unknown_checkpoint')"
    )
    assert "older analysis was stopped by a memory limiter" in unknown_checkpoint_failure["jobStatus"]
    assert "Press Rerun to start this analysis with the current backend" in unknown_checkpoint_failure["jobStatus"]
    assert "checkpoint is available" not in unknown_checkpoint_failure["jobStatus"]


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
    page.route("**/review_disposition", _fulfill_review_disposition)
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)

    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_confirm_wrong_job')""",
        _mock_class_split_result(),
    )
    page.select_option("#classSplitDisplayMode", "all")
    page.wait_for_selector('.class-split-wrong-item[data-point-id="truck-2"]', timeout=15000)
    page.click('.class-split-wrong-item[data-point-id="truck-2"] [data-action="correct-class"]')
    page.wait_for_function(
        "() => !document.querySelector('.class-split-wrong-item[data-point-id=\"truck-2\"]')",
        timeout=15000,
    )

    snapshot = page.evaluate("() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot()")
    assert snapshot["traceNames"] == ["Person", "Truck"]
    assert snapshot["tracePointCounts"] == [2, 1]
    assert "3/4 objects shown" in snapshot["statusText"]
    assert "No points match" not in snapshot["graphText"]
    assert "No likely wrong-class objects were flagged." in (page.text_content("#classSplitWrongList") or "")


def test_class_split_wrong_candidate_skip_removes_vignette_without_clearing_flag(playwright_page):
    page, _ = playwright_page
    page.route("**/review_disposition", _fulfill_review_disposition)
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)

    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_skip_wrong_job')""",
        _mock_class_split_result(),
    )
    page.select_option("#classSplitDisplayMode", "all")
    page.wait_for_selector('.class-split-wrong-item[data-point-id="truck-2"]', timeout=15000)
    page.click('.class-split-wrong-item[data-point-id="truck-2"] [data-action="skip-wrong"]')
    page.wait_for_function(
        "() => !document.querySelector('.class-split-wrong-item[data-point-id=\"truck-2\"]')",
        timeout=15000,
    )

    page.select_option("#classSplitDisplayMode", "wrong_only")
    page.wait_for_function(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot().tracePointCounts.reduce((total, count) => total + count, 0) === 0",
        timeout=15000,
    )
    snapshot = page.evaluate("() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot()")
    assert sum(snapshot["tracePointCounts"]) == 0
    assert "No likely wrong-class objects were flagged." in (page.text_content("#classSplitWrongList") or "")


@pytest.mark.parametrize(
    "action,disposition",
    [
        ("correct-class", "confirm_current"),
        ("skip-wrong", "skip"),
    ],
)
def test_class_split_review_actions_acknowledge_and_hide_before_save_finishes(
    playwright_page,
    action,
    disposition,
):
    page, _ = playwright_page
    pending_routes = []

    def hold_review_disposition(route):
        pending_routes.append(route)

    page.route("**/review_disposition", hold_review_disposition)
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)
    page.evaluate(
        """(capabilities) => window.__TATOR_TEST_HOOKS__.classSplitSetCapabilities(capabilities)""",
        {"review_disposition_api_version": 3},
    )
    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_optimistic_review_job')""",
        _mock_class_split_result(),
    )
    selector = '.class-split-wrong-item[data-point-id="truck-2"]'
    button_selector = f'{selector} [data-action="{action}"]'
    page.wait_for_selector(button_selector, timeout=15000)
    page.evaluate(
        """({cardSelector, buttonSelector}) => {
            const card = document.querySelector(cardSelector);
            const button = document.querySelector(buttonSelector);
            window.__classSplitReviewTiming = {start: 0, acknowledged: 0, removed: 0};
            button.addEventListener('click', () => {
                window.__classSplitReviewTiming.start = performance.now();
            }, {capture: true, once: true});
            const observer = new MutationObserver(() => {
                const timing = window.__classSplitReviewTiming;
                if (!timing.acknowledged && button.classList.contains('is-acknowledged')) {
                    timing.acknowledged = performance.now();
                }
                if (!timing.removed && !document.documentElement.contains(card)) {
                    timing.removed = performance.now();
                    observer.disconnect();
                }
            });
            observer.observe(document.body, {attributes: true, childList: true, subtree: true});
        }""",
        {"cardSelector": selector, "buttonSelector": button_selector},
    )

    page.click(button_selector)
    page.wait_for_function(
        "() => window.__classSplitReviewTiming?.acknowledged > 0",
        timeout=1000,
    )
    page.wait_for_selector(selector, state="detached", timeout=1000)
    page.wait_for_function("() => window.__classSplitReviewTiming?.removed > 0", timeout=1000)
    timing = page.evaluate("() => window.__classSplitReviewTiming")

    assert timing["acknowledged"] - timing["start"] <= 150
    assert 250 <= timing["removed"] - timing["start"] <= 800
    assert len(pending_routes) == 1
    assert (pending_routes[0].request.post_data_json or {})["disposition"] == disposition

    _fulfill_review_disposition(pending_routes.pop())
    page.wait_for_function(
        "() => (document.querySelector('#taskQueue')?.textContent || '').toLowerCase().includes('saved')",
        timeout=3000,
    )


def test_class_split_review_disposition_failure_restores_optimistic_vignette(playwright_page):
    page, _ = playwright_page
    pending_routes = []

    def hold_review_disposition(route):
        pending_routes.append(route)

    page.route("**/review_disposition", hold_review_disposition)
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)
    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_failed_disposition_job')""",
        _mock_class_split_result(),
    )
    selector = '.class-split-wrong-item[data-point-id="truck-2"]'
    page.wait_for_selector(selector, timeout=15000)
    page.click(f'{selector} [data-action="skip-wrong"]')
    page.wait_for_selector(selector, state="detached", timeout=1000)
    assert len(pending_routes) == 1
    pending_routes.pop().fulfill(
        status=503,
        content_type="application/json",
        body='{"detail":"review_ledger_unavailable"}',
    )
    page.wait_for_function(
        "() => (document.querySelector('#classSplitJobStatus')?.textContent || '').includes('Could not save skip')",
        timeout=15000,
    )
    assert page.locator(selector).count() == 1
    assert page.locator(f'{selector} [data-action="skip-wrong"]').is_disabled()
    assert "review_ledger_unavailable" in (page.text_content("#classSplitJobStatus") or "")
    assert "review_ledger_unavailable" in (page.text_content("#taskQueue") or "")


def _mock_review_history_result():
    result = _mock_class_split_result()
    reviewed = [
        (0, "confirm_current", 1_000_000_000),
        (1, "skip", "2026-08-03T18:00:00Z"),
        (2, "confirm_current", 2_000_000_000),
    ]
    for index, disposition, reviewed_at in reviewed:
        result["points"][index].update(
            {
                "human_review_disposition": disposition,
                "human_reviewed_at": reviewed_at,
                "human_review_origin": "desktop",
                "human_review_persistence": "durable",
            }
        )
    return result


def _review_history_delete_receipt(
    job_id,
    *,
    client_action_id,
    requested=3,
    resolved=3,
    deleted=3,
    absent=0,
):
    return {
        "schema": "class-analysis-review-history-delete-v1",
        "status": "deleted" if deleted else "nothing_to_delete",
        "job_id": job_id,
        "client_action_id": client_action_id,
        "requested_point_count": requested,
        "resolved_review_key_count": resolved,
        "deleted_count": deleted,
        "absent_count": absent,
        "deleted_disposition_counts": {},
        "labels_changed": False,
        "annotations_changed": False,
        "training_actions_deleted": 0,
        "qwen_audits_deleted": 0,
    }


def test_class_split_review_history_is_below_vignettes_newest_first_and_bulk_deletable(
    playwright_page,
):
    page, _ = playwright_page
    delete_requests = []

    def delete_history(route):
        request = route.request.post_data_json or {}
        delete_requests.append(request)
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                _review_history_delete_receipt(
                    "pw_class_split_review_history_job",
                    client_action_id=request.get("client_action_id", ""),
                )
            ),
        )

    page.route("**/review_history/delete", delete_history)
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)
    page.evaluate(
        """(capabilities) => window.__TATOR_TEST_HOOKS__.classSplitSetCapabilities(capabilities)""",
        {
            "review_disposition_api_version": 3,
            "review_history_delete_api_version": 1,
        },
    )
    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_review_history_job')""",
        _mock_review_history_result(),
    )

    assert page.evaluate(
        """() => Boolean(
            document.querySelector('#classSplitWrongList').compareDocumentPosition(
                document.querySelector('#classSplitReviewedPanel')
            ) & Node.DOCUMENT_POSITION_FOLLOWING
        )"""
    )
    page.locator("#classSplitReviewedPanel").evaluate("node => { node.open = true; }")
    labels = page.locator("#classSplitReviewedList .class-split-reviewed-item strong").all_text_contents()
    assert labels == ["Confirmed: Person", "Skipped: Truck", "Confirmed: Truck"]
    assert page.text_content("#classSplitReviewedSummary") == "Review history (3)"
    assert page.locator("#classSplitReviewHistoryDelete").is_enabled()

    page.once("dialog", lambda dialog: dialog.accept())
    page.click("#classSplitReviewHistoryDelete")
    page.wait_for_function(
        "() => (document.querySelector('#classSplitReviewedSummary')?.textContent || '') === 'Review history (0)'",
        timeout=3000,
    )

    assert len(delete_requests) == 1
    assert delete_requests[0]["schema"] == "class-analysis-review-history-delete-v1"
    assert [entry["point_id"] for entry in delete_requests[0]["entries"]] == [
        "person-1",
        "truck-2",
        "truck-1",
    ]
    assert "No review history" in (page.text_content("#classSplitReviewedList") or "")


def test_class_split_review_history_delete_failure_keeps_history(playwright_page):
    page, _ = playwright_page

    def fail_delete_history(route):
        route.fulfill(
            status=503,
            content_type="application/json",
            body='{"detail":"review_history_storage_unavailable"}',
        )

    page.route("**/review_history/delete", fail_delete_history)
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.evaluate(
        """(capabilities) => window.__TATOR_TEST_HOOKS__.classSplitSetCapabilities(capabilities)""",
        {
            "review_disposition_api_version": 3,
            "review_history_delete_api_version": 1,
        },
    )
    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_review_history_failure_job')""",
        _mock_review_history_result(),
    )
    page.locator("#classSplitReviewedPanel").evaluate("node => { node.open = true; }")
    page.once("dialog", lambda dialog: dialog.accept())
    page.click("#classSplitReviewHistoryDelete")
    page.wait_for_function(
        "() => (document.querySelector('#classSplitJobStatus')?.textContent || '').includes('review_history_storage_unavailable')",
        timeout=3000,
    )

    assert page.text_content("#classSplitReviewedSummary") == "Review history (3)"
    assert page.locator("#classSplitReviewHistoryDelete").is_enabled()
    assert "review_history_storage_unavailable" in (page.text_content("#taskQueue") or "")


def test_class_split_review_history_rejects_invalid_success_receipt(playwright_page):
    page, _ = playwright_page

    page.route(
        "**/review_history/delete",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"status":"deleted","deleted_count":3}',
        ),
    )
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.evaluate(
        """(capabilities) => window.__TATOR_TEST_HOOKS__.classSplitSetCapabilities(capabilities)""",
        {
            "review_disposition_api_version": 3,
            "review_history_delete_api_version": 1,
        },
    )
    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_review_history_invalid_receipt_job')""",
        _mock_review_history_result(),
    )
    page.locator("#classSplitReviewedPanel").evaluate("node => { node.open = true; }")
    page.once("dialog", lambda dialog: dialog.accept())
    page.click("#classSplitReviewHistoryDelete")
    page.wait_for_function(
        "() => (document.querySelector('#classSplitJobStatus')?.textContent || '').includes('invalid review-history deletion receipt')",
        timeout=3000,
    )

    assert page.text_content("#classSplitReviewedSummary") == "Review history (3)"
    assert page.locator("#classSplitReviewHistoryDelete").is_enabled()
    assert "invalid review-history deletion receipt" in (
        page.text_content("#taskQueue") or ""
    )


def test_class_split_review_history_uses_saved_revision_precondition(playwright_page):
    page, _ = playwright_page
    delete_requests = []
    revision = "rdr1_0123456789abcdef0123456789abcdef"

    def save_review(route):
        payload = route.request.post_data_json or {}
        url_parts = route.request.url.split("/")
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "schema": "class-analysis-review-disposition-v3",
                    "status": "recorded",
                    "job_id": url_parts[url_parts.index("jobs") + 1],
                    "point_id": url_parts[url_parts.index("points") + 1],
                    "disposition": payload.get("disposition"),
                    "client_action_id": payload.get("client_action_id", ""),
                    "training_capture_requested": payload.get("capture_training_data") is True,
                    "review_object_key": "cro_" + ("b" * 64),
                    "human_reviewed_at": 1_785_800_003.0,
                    "human_review_revision": revision,
                }
            ),
        )

    def delete_history(route):
        request = route.request.post_data_json or {}
        delete_requests.append(request)
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                _review_history_delete_receipt(
                    "pw_class_split_review_history_revision_job",
                    client_action_id=request.get("client_action_id", ""),
                    requested=1,
                    resolved=1,
                    deleted=1,
                )
            ),
        )

    page.route("**/review_disposition", save_review)
    page.route("**/review_history/delete", delete_history)
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.evaluate(
        """(capabilities) => window.__TATOR_TEST_HOOKS__.classSplitSetCapabilities(capabilities)""",
        {
            "review_disposition_api_version": 3,
            "review_history_delete_api_version": 1,
        },
    )
    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_review_history_revision_job')""",
        _mock_class_split_result(),
    )

    page.click(
        '.class-split-wrong-item[data-point-id="truck-2"] [data-action="skip-wrong"]'
    )
    page.wait_for_function(
        "() => (document.querySelector('#classSplitReviewedSummary')?.textContent || '') === 'Review history (1)'",
        timeout=3000,
    )
    page.wait_for_function(
        "() => !document.querySelector('#classSplitReviewHistoryDelete')?.disabled",
        timeout=3000,
    )
    page.locator("#classSplitReviewedPanel").evaluate("node => { node.open = true; }")
    page.once("dialog", lambda dialog: dialog.accept())
    page.click("#classSplitReviewHistoryDelete")
    page.wait_for_function(
        "() => (document.querySelector('#classSplitReviewedSummary')?.textContent || '') === 'Review history (0)'",
        timeout=3000,
    )

    assert len(delete_requests) == 1
    entry = delete_requests[0]["entries"][0]
    assert entry["expected_revision"] == revision
    assert "expected_reviewed_at" not in entry


def test_class_split_review_history_missing_timestamp_fails_visibly_before_request(
    playwright_page,
):
    page, _ = playwright_page
    delete_requests = []

    def unexpected_delete(route):
        delete_requests.append(route.request.post_data_json or {})
        route.abort()

    page.route("**/review_history/delete", unexpected_delete)
    result = _mock_review_history_result()
    result["points"][0]["human_reviewed_at"] = ""
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.evaluate(
        """(capabilities) => window.__TATOR_TEST_HOOKS__.classSplitSetCapabilities(capabilities)""",
        {
            "review_disposition_api_version": 3,
            "review_history_delete_api_version": 1,
        },
    )
    page.evaluate(
        """async (result) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, 'pw_class_split_review_history_missing_timestamp_job')""",
        result,
    )
    page.locator("#classSplitReviewedPanel").evaluate("node => { node.open = true; }")
    page.click("#classSplitReviewHistoryDelete")
    page.wait_for_function(
        "() => (document.querySelector('#classSplitJobStatus')?.textContent || '').includes('has no durable revision or timestamp')",
        timeout=3000,
    )

    assert delete_requests == []
    assert page.text_content("#classSplitReviewedSummary") == "Review history (3)"
    assert "has no durable revision or timestamp" in (page.text_content("#taskQueue") or "")


def test_class_split_review_history_delete_blocks_then_allows_new_review(
    playwright_page,
):
    page, _ = playwright_page
    pending_delete = []

    def hold_delete(route):
        pending_delete.append(route)

    page.route("**/review_history/delete", hold_delete)
    page.route("**/review_disposition", _fulfill_review_disposition)
    result = _mock_review_history_result()
    for key in (
        "human_review_disposition",
        "human_reviewed_at",
        "human_review_origin",
        "human_review_persistence",
    ):
        result["points"][1].pop(key, None)
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.evaluate(
        """(capabilities) => window.__TATOR_TEST_HOOKS__.classSplitSetCapabilities(capabilities)""",
        {
            "review_disposition_api_version": 3,
            "review_history_delete_api_version": 1,
        },
    )
    job_id = "pw_class_split_review_history_concurrent_job"
    page.evaluate(
        """async ({result, jobId}) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, jobId)""",
        {"result": result, "jobId": job_id},
    )
    page.locator("#classSplitReviewedPanel").evaluate("node => { node.open = true; }")
    page.once("dialog", lambda dialog: dialog.accept())
    page.click("#classSplitReviewHistoryDelete")
    page.wait_for_timeout(100)
    assert len(pending_delete) == 1

    page.click(
        '.class-split-wrong-item[data-point-id="truck-2"] [data-action="skip-wrong"]'
    )
    page.wait_for_timeout(100)
    assert page.text_content("#classSplitReviewedSummary") == "Review history (2)"
    delete_route = pending_delete.pop()
    delete_request = delete_route.request.post_data_json or {}
    delete_route.fulfill(
        status=200,
        content_type="application/json",
        body=json.dumps(
            _review_history_delete_receipt(
                job_id,
                client_action_id=delete_request.get("client_action_id", ""),
                requested=2,
                resolved=2,
                deleted=2,
            )
        ),
    )
    page.wait_for_function(
        "() => (document.querySelector('#classSplitReviewedSummary')?.textContent || '') === 'Review history (0)'",
        timeout=3000,
    )

    page.click(
        '.class-split-wrong-item[data-point-id="truck-2"] [data-action="skip-wrong"]'
    )
    page.wait_for_function(
        "() => (document.querySelector('#classSplitReviewedSummary')?.textContent || '') === 'Review history (1)'",
        timeout=3000,
    )

    assert page.locator(
        "#classSplitReviewedList .class-split-reviewed-item strong"
    ).all_text_contents() == ["Skipped: Truck"]


def test_class_split_review_history_delete_tokens_survive_job_transition_and_suppress_old_failure(
    playwright_page,
):
    page, _ = playwright_page
    pending_deletes = []

    def hold_delete(route):
        request = route.request.post_data_json or {}
        pending_deletes.append(
            {
                "route": route,
                "job_id": route.request.url.split("/jobs/", 1)[1].split("/", 1)[0],
                "requested": len(request.get("entries") or []),
                "client_action_id": request.get("client_action_id", ""),
            }
        )

    page.route("**/review_history/delete", hold_delete)
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.evaluate(
        """(capabilities) => window.__TATOR_TEST_HOOKS__.classSplitSetCapabilities(capabilities)""",
        {
            "review_disposition_api_version": 3,
            "review_history_delete_api_version": 1,
        },
    )
    old_job = "pw_class_split_review_history_old_job"
    new_job = "pw_class_split_review_history_new_job"
    page.evaluate(
        """async ({result, jobId}) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, jobId)""",
        {"result": _mock_review_history_result(), "jobId": old_job},
    )
    page.once("dialog", lambda dialog: dialog.accept())
    page.click("#classSplitReviewHistoryDelete")
    page.wait_for_timeout(100)
    assert len(pending_deletes) == 1

    page.evaluate(
        """async ({result, jobId}) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(result, jobId)""",
        {"result": _mock_review_history_result(), "jobId": new_job},
    )
    assert page.locator("#classSplitReviewHistoryDelete").is_enabled()
    page.once("dialog", lambda dialog: dialog.accept())
    page.click("#classSplitReviewHistoryDelete")
    page.wait_for_timeout(100)
    assert len(pending_deletes) == 2
    assert page.text_content("#classSplitReviewHistoryDelete") == "Deleting …"

    first = pending_deletes[0]
    first["route"].fulfill(
        status=503,
        content_type="application/json",
        body='{"detail":"old_job_review_history_delete_failed"}',
    )
    page.wait_for_timeout(100)
    assert page.text_content("#classSplitReviewHistoryDelete") == "Deleting …"
    assert not page.locator("#classSplitReviewHistoryDelete").is_enabled()
    assert page.text_content("#classSplitReviewedSummary") == "Review history (3)"
    assert "old_job_review_history_delete_failed" not in (
        page.text_content("#classSplitJobStatus") or ""
    )
    assert "old_job_review_history_delete_failed" not in (
        page.text_content("#taskQueue") or ""
    )

    second = pending_deletes[1]
    second["route"].fulfill(
        status=200,
        content_type="application/json",
        body=json.dumps(
            _review_history_delete_receipt(
                second["job_id"],
                client_action_id=second["client_action_id"],
                requested=second["requested"],
                resolved=second["requested"],
                deleted=second["requested"],
            )
        ),
    )
    page.wait_for_function(
        "() => (document.querySelector('#classSplitReviewedSummary')?.textContent || '') === 'Review history (0)'",
        timeout=3000,
    )


def test_class_split_active_workspace_delete_bbox_is_immediate_and_exact(playwright_page):
    page, _ = playwright_page
    disposition_requests = []

    def unexpected_disposition(route):
        disposition_requests.append(route.request.post_data_json or {})
        route.fulfill(
            status=500,
            content_type="application/json",
            body='{"detail":"unexported_local_delete_must_not_be_durable"}',
        )

    page.route("**/review_disposition", unexpected_disposition)
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function(
        "!!window.__TATOR_TEST_HOOKS__?.classSplitSeedActiveWorkspaceAnnotations",
        timeout=15000,
    )
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
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_class_split_single_bbox_delete_job'
        )""",
        _mock_active_workspace_single_bbox_result(),
    )
    selector = '.class-split-wrong-item[data-point-id="truck-2"]'
    delete_button = page.locator(f'{selector} [data-action="delete-bbox"]')
    assert delete_button.is_enabled()
    delete_button.click()
    page.wait_for_selector(selector, state="detached", timeout=1000)
    snapshot = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitLocalAnnotationSnapshot('single.png')"
    )

    assert snapshot["classCounts"]["Truck"] == 1
    assert snapshot["boxes"]["Truck"] == [
        {"x": 300, "y": 320, "width": 50, "height": 60}
    ]
    assert disposition_requests == []
    assert "Shift+Y" in (page.text_content("#taskQueue") or "")


def test_class_split_switch_class_hides_before_optional_capture_finishes(playwright_page):
    page, _ = playwright_page
    pending_capture_routes = []

    def hold_training_capture(route):
        pending_capture_routes.append(route)

    page.route("**/training_actions", hold_training_capture)
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function(
        "!!window.__TATOR_TEST_HOOKS__?.classSplitSeedActiveWorkspaceAnnotations",
        timeout=15000,
    )
    page.evaluate(
        """(fixture) => window.__TATOR_TEST_HOOKS__.classSplitSeedActiveWorkspaceAnnotations(fixture)""",
        {
            "imageKey": "single.png",
            "width": 960,
            "height": 960,
            "classNames": ["Truck", "Person"],
            "boxes": {
                "Truck": [
                    {"x": 100, "y": 120, "width": 80, "height": 100},
                    {"x": 300, "y": 320, "width": 50, "height": 60},
                ],
            },
        },
    )
    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_class_split_fast_class_change_job'
        )""",
        _mock_active_workspace_single_bbox_result(),
    )
    page.check("#classSplitTrainingCapture")
    selector = '.class-split-wrong-item[data-point-id="truck-2"]'
    page.select_option(f'{selector} [data-action="target-class"]', "Person")
    page.click(f'{selector} [data-action="reassign-class"]')
    page.wait_for_selector(selector, state="detached", timeout=1000)
    assert len(pending_capture_routes) == 1

    snapshot = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitLocalAnnotationSnapshot('single.png')"
    )
    assert snapshot["classCounts"] == {"Truck": 1, "Person": 1}
    assert snapshot["boxes"]["Person"] == [
        {"x": 100, "y": 120, "width": 80, "height": 100}
    ]

    pending_capture_routes.pop().fulfill(
        status=200,
        content_type="application/json",
        body='{"status":"recorded","action_ids":["cta_fast_switch"]}',
    )
    page.wait_for_function(
        "() => (document.querySelector('#taskQueue')?.textContent || '').includes('Shift+Y')",
        timeout=3000,
    )


def test_class_split_dual_bbox_badge_uses_dark_theme_colors(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)
    result = _mock_class_split_result()
    conflict = {
        "enabled": True,
        "kind": "near_identical_cross_class_bbox",
        "review_mode": "dual_bbox_annotation_resolution",
        "point_id": "truck-2",
        "other_point_id": "person-1",
        "current_class": "Truck",
        "other_class_name": "Person",
        "split": "train",
        "image_relpath": "img_1.png",
        "target_bbox_xyxy": [10.0, 20.0, 50.0, 80.0],
        "other_bbox_xyxy": [10.2, 20.1, 50.1, 80.2],
        "iou": 0.91,
        "target_area_covered": 0.91,
    }
    result["points"][1]["split"] = "train"
    result["points"][1]["bbox_xyxy"] = [10.0, 20.0, 50.0, 80.0]
    result["points"][1]["dual_bbox_conflict"] = conflict
    result["wrong_class_candidates"][0]["dual_bbox_conflict"] = conflict
    page.evaluate(
        """() => {
            document.documentElement.classList.remove('theme-pipboy');
            document.documentElement.classList.add('theme-dark');
        }"""
    )
    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_class_split_dual_badge_job'
        )""",
        result,
    )

    style = page.eval_on_selector(
        ".class-split-wrong-item__badge--dual",
        """element => {
            const computed = getComputedStyle(element);
            return {
                backgroundColor: computed.backgroundColor,
                color: computed.color,
                display: computed.display,
            };
        }""",
    )

    assert style == {
        "backgroundColor": "rgba(127, 29, 29, 0.72)",
        "color": "rgb(254, 202, 202)",
        "display": "inline-block",
    }
    card = page.locator('.class-split-wrong-item[data-point-id="truck-2"]')
    assert card.get_by_role("button", name="Delete Truck box").count() == 1
    assert card.get_by_role("button", name="Delete Person box").count() == 1
    assert card.get_by_role("button", name="Keep both boxes").count() == 1
    assert card.get_by_role("button", name="Leave unresolved").count() == 1
    assert card.locator('[data-action="correct-class"]').count() == 0
    assert card.locator('[data-action="skip-wrong"]').count() == 0
    assert card.locator('[data-action="target-class"]').count() == 0
    assert card.locator('[data-action="reassign-class"]').count() == 0
    page.evaluate(
        """async () => window.__TATOR_TEST_HOOKS__.classSplitEmitPointClick('truck-2')"""
    )
    inspector = page.locator("#classSplitInspector")
    assert inspector.get_by_role("button", name="Delete Truck box").count() == 1
    assert inspector.get_by_role("button", name="Delete Person box").count() == 1
    assert inspector.locator('[data-action="class"]').count() == 0
    assert inspector.locator('[data-action="change"]').count() == 0


@pytest.mark.parametrize("disposition,button_name", [
    ("keep_both_boxes", "Keep both boxes"),
    ("unresolved", "Leave unresolved"),
])
def test_class_split_dual_bbox_review_only_actions_click_and_persist(
    playwright_page,
    disposition,
    button_name,
):
    page, _ = playwright_page
    requests = []

    def save_pair_review(route):
        requests.append(route.request.post_data_json or {})
        _fulfill_review_disposition(route)

    page.route("**/review_disposition", save_pair_review)
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function(
        "!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult",
        timeout=15000,
    )
    page.evaluate(
        """(capabilities) => window.__TATOR_TEST_HOOKS__.classSplitSetCapabilities(capabilities)""",
        {
            "review_disposition_api_version": 3,
            "dual_bbox_resolution_api_version": 1,
            "dual_bbox_annotation_transaction_api_version": 2,
        },
    )
    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_class_split_dual_review_action_job'
        )""",
        _mock_active_workspace_dual_bbox_result(),
    )
    button = page.locator(
        f'.class-split-wrong-item[data-point-id="truck-2"] button:has-text("{button_name}")'
    )
    assert button.is_enabled()
    button.click()
    page.wait_for_function(
        "() => !document.querySelector('.class-split-wrong-item[data-point-id=\"truck-2\"]')",
        timeout=15000,
    )

    assert len(requests) == 1
    assert requests[0]["disposition"] == disposition
    assert requests[0]["dual_bbox_conflict"]["point_id"] == "truck-2"
    assert requests[0]["dual_bbox_conflict"]["other_point_id"] == "person-1"
    page.unroute("**/review_disposition", save_pair_review)


def test_class_split_dual_bbox_active_workspace_delete_mutates_local_box(playwright_page):
    page, _ = playwright_page
    transaction_requests = []
    disposition_requests = []

    def unexpected_transaction(route):
        transaction_requests.append(route.request.post_data_json or {})
        route.fulfill(
            status=500,
            content_type="application/json",
            body='{"detail":"active_workspace_must_not_use_server_transaction"}',
        )

    def unexpected_disposition(route):
        disposition_requests.append(route.request.post_data_json or {})
        route.fulfill(
            status=500,
            content_type="application/json",
            body='{"detail":"unexported_local_delete_must_not_be_durable"}',
        )

    page.route("**/dual_bbox_annotation_transaction", unexpected_transaction)
    page.route("**/review_disposition", unexpected_disposition)
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function(
        "!!window.__TATOR_TEST_HOOKS__?.classSplitSeedActiveWorkspaceAnnotations",
        timeout=15000,
    )
    page.evaluate(
        """(capabilities) => window.__TATOR_TEST_HOOKS__.classSplitSetCapabilities(capabilities)""",
        {
            "review_disposition_api_version": 3,
            "dual_bbox_resolution_api_version": 1,
            "dual_bbox_annotation_transaction_api_version": 2,
        },
    )
    page.evaluate(
        """(fixture) => window.__TATOR_TEST_HOOKS__.classSplitSeedActiveWorkspaceAnnotations(fixture)""",
        {
            "imageKey": "pair.png",
            "width": 960,
            "height": 960,
            "boxes": {
                "Truck": [{"x": 557, "y": 720, "width": 76, "height": 31}],
                "LightVehicle": [{"x": 554, "y": 721, "width": 76, "height": 30}],
            },
        },
    )
    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_class_split_active_workspace_delete_job'
        )""",
        _mock_active_workspace_dual_bbox_result(),
    )
    page.evaluate(
        """async () => window.__TATOR_TEST_HOOKS__.classSplitEmitPointClick('truck-2')"""
    )
    delete_button = page.locator(
        '#classSplitInspector [data-disposition="delete_current_box"]'
    )
    assert delete_button.is_enabled()
    delete_button.click()
    page.wait_for_function(
        "() => (document.querySelector('#classSplitJobStatus')?.textContent || '').includes('Shift+Y')",
        timeout=15000,
    )
    snapshot = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitLocalAnnotationSnapshot('pair.png')"
    )

    assert snapshot["classCounts"].get("Truck", 0) == 0
    assert snapshot["classCounts"]["LightVehicle"] == 1
    assert "Shift+Y" in snapshot["jobStatus"]
    assert "Shift+Y" in snapshot["reviewedText"]
    assert transaction_requests == []
    assert disposition_requests == []
    page.unroute("**/dual_bbox_annotation_transaction", unexpected_transaction)
    page.unroute("**/review_disposition", unexpected_disposition)


def test_class_split_confirm_wrong_candidate_prunes_hidden_wrong_only_selection(playwright_page):
    page, _ = playwright_page
    page.route("**/review_disposition", _fulfill_review_disposition)
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
    assert "12 of 15 remaining" in (page.text_content("#classSplitWrongQueueStatus") or "")

    first_id = page.eval_on_selector(".class-split-wrong-item", "el => el.getAttribute('data-point-id')")
    page.click(f'.class-split-wrong-item[data-point-id="{first_id}"] [data-action="skip-wrong"]')
    page.wait_for_function(
        """(pointId) => !document.querySelector(`.class-split-wrong-item[data-point-id="${pointId}"]`)""",
        arg=first_id,
        timeout=15000,
    )
    assert page.locator(".class-split-wrong-item").count() == 12
    assert "12 of 14 remaining" in (page.text_content("#classSplitWrongQueueStatus") or "")

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
    assert snapshot["traceNames"] == ["Truck", "Person"]
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
    assert snapshot["traceNames"] == ["Truck", "Person"]
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


def test_class_split_quality_memory_queue_and_budget_controls(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_selector("#classSplitQualityMemoryPolicy", state="attached")

    initial = page.evaluate(
        """() => ({
            policy: document.querySelector('#classSplitQualityMemoryPolicy').value,
            budgetDisabled: document.querySelector('#classSplitQualityMemoryBudget').disabled,
            showAll: document.querySelector('#classSplitShowAllRough').checked,
            reviewPercent: document.querySelector('#classSplitQualityReviewFractionValue').textContent,
            guidance: document.querySelector('#classSplitEmbeddingGuidanceStatus').textContent,
            advancedOpen: document.querySelector('#classSplitAdvancedSetup').open,
            guidanceOpen: document.querySelector('#classSplitEmbeddingGuide').open,
            previewOpen: document.querySelector('.class-split-embedding-guide__preview').open,
            classFieldHidden: document.querySelector('#classSplitClassField').hidden,
        })"""
    )
    assert initial["policy"] == "auto"
    assert initial["budgetDisabled"] is True
    assert initial["showAll"] is False
    assert initial["reviewPercent"] == "5%"
    assert "Thorough quality" in initial["guidance"]
    assert "custom" not in initial["guidance"].lower()
    assert initial["advancedOpen"] is False
    assert initial["guidanceOpen"] is False
    assert initial["previewOpen"] is False
    assert initial["classFieldHidden"] is True

    page.locator("#classSplitAdvancedSetup > summary").click()

    page.select_option("#classSplitQualityMemoryPolicy", "full")
    full = page.locator("#classSplitQualityMemoryStatus").inner_text()
    assert "Maximum fidelity" in full
    assert page.locator("#classSplitQualityMemoryBudget").is_disabled()

    page.select_option("#classSplitQualityMemoryPolicy", "budgeted")
    page.fill("#classSplitQualityMemoryBudget", "3")
    page.dispatch_event("#classSplitQualityMemoryBudget", "change")
    bounded = page.locator("#classSplitQualityMemoryStatus").inner_text()
    assert "budgets 3 GiB" in bounded
    assert page.locator("#classSplitQualityMemoryBudget").is_enabled()

    page.fill("#classSplitQualityReviewFraction", "9")
    page.dispatch_event("#classSplitQualityReviewFraction", "input")
    assert page.locator("#classSplitQualityReviewFractionValue").inner_text() == "9%"


def test_class_split_adaptive_ranking_ignores_stale_job_response(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)

    old_result = _mock_class_split_result()
    old_result["summary"]["analysis_input_digest"] = "digest-old"
    old_result["quality_review_queue"] = {
        "priority_ids": ["truck-2"],
        "tiny_ids": [],
        "all_flagged_ids": ["truck-2"],
    }
    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_adaptive_old_job'
        )""",
        old_result,
    )

    held_routes = []
    page.route(
        "**/class_analysis/jobs/pw_adaptive_old_job/review-ranking/recalibrate",
        lambda route: held_routes.append(route),
    )
    page.check("#classSplitAdaptiveRanking")
    page.wait_for_timeout(100)
    assert len(held_routes) == 1
    request_payload = held_routes[0].request.post_data_json
    assert request_payload["analysis_digest"] == "digest-old"
    assert request_payload["ranking_revision"] == ""
    assert page.locator("#classSplitRunButton").is_disabled()
    assert page.locator("#classSplitAdaptiveRankingReset").is_disabled()

    new_result = _mock_class_split_result()
    new_result["summary"]["analysis_input_digest"] = "digest-new"
    new_result["points"] = new_result["points"][:2]
    new_result["summary"]["object_count"] = 2
    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_adaptive_new_job'
        )""",
        new_result,
    )
    held_routes[0].fulfill(
        status=200,
        content_type="application/json",
        body=json.dumps(
            {
                "enabled": True,
                "ready": True,
                "analysis_digest": "digest-old",
                "ranking_revision": "dqr1_old",
                "ordered_point_ids": ["truck-2"],
                "audit_point_ids": [],
                "ranking": [
                    {
                        "point_id": "truck-2",
                        "adaptive_review_score": 1.0,
                        "review_sampling_source": "priority",
                    }
                ],
                "readiness": {"receipt_count": 40},
            }
        ),
    )
    page.wait_for_timeout(100)
    assert "Calibrated from" not in page.locator("#classSplitAdaptiveRankingStatus").inner_text()
    snapshot = page.evaluate("() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot()")
    assert sum(snapshot["tracePointCounts"]) == 2


def test_class_split_deep_evidence_defaults_off_and_preserves_user_override(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitSetCapabilities", timeout=15000)
    supported_capabilities = {
        "default_encoder_type": "dinov3",
        "default_dinov3_model": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "projection_methods": ["pca"],
        "fine_grained_refinement": {
            "api_version": 5,
            "schema": "class-analysis-patch-refinement-v5",
            "decision_contract": "class-analysis-patch-decision-v9",
            "selector_priority_contract": (
                "expected-review-utility-global-boosted-statistical-overlap-rerank-v7"
            ),
            "selector_feature_contract": (
                "raw-stage1-patch-same-image-and-gated-dataset-overlap-features-v3"
            ),
            "selector_model_schema": (
                "class-analysis-selector-utility-model-v2"
            ),
            "selector_model_digest": _SELECTOR_MODEL_DIGEST,
            "selector_utility_policy_contract": (
                "actionability-times-75pct-base-plus-25pct-reviewability-v1"
            ),
            "selector_dataset_overlap_application_contract": (
                "affirmative-current-localized-material-source-loo-v2"
            ),
            "selector_dataset_overlap_diagnostic_contract": (
                "capture-loo-beta-wilson-shrunk-rank-only-v1"
            ),
            "selector_dataset_overlap_scoring_effect_enabled": True,
            "selector_global_actionability_model_contract": (
                "pair-blind-shallow-histogram-gradient-boosting-v1"
            ),
            "default_enabled": False,
            "precise_default_enabled": False,
            "experimental": True,
            "blocks_use": False,
            "supported_encoder": "dinov3",
            "supported_model_family": "vit",
        },
    }
    initial = page.evaluate(
        "(capabilities) => window.__TATOR_TEST_HOOKS__.classSplitSetCapabilities(capabilities)",
        supported_capabilities,
    )
    assert initial["checked"] is False
    assert "leaves detailed review ranking off unless you opt in" in initial["hint"]
    assert "Deep evidence uses its own DINOv3 spatial encoder" in initial["hint"]
    assert "does not change labels or hide candidates" in initial["hint"]

    fast = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitApplyRefinementPreset('fast_map_v1')"
    )
    assert fast["preference"] is False
    assert fast["checked"] is False

    overridden = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitApplyRefinementPreset('thorough_quality_v1', {userOverride: true})"
    )
    assert overridden["touched"] is True
    assert overridden["preference"] is True
    assert overridden["checked"] is True

    preserved = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitApplyRefinementPreset('fast_map_v1')"
    )
    assert preserved["preference"] is True
    assert preserved["checked"] is True

    unsupported = dict(supported_capabilities)
    unsupported["fine_grained_refinement"] = {"api_version": 0, "supported": False}
    unavailable = page.evaluate(
        "(capabilities) => window.__TATOR_TEST_HOOKS__.classSplitSetCapabilities(capabilities)",
        unsupported,
    )
    assert unavailable["disabled"] is True
    assert unavailable["checked"] is False
    assert "advertises an unsupported patch-refinement contract" in unavailable["hint"]

    stale_contract = dict(supported_capabilities)
    stale_contract["fine_grained_refinement"] = {
        **supported_capabilities["fine_grained_refinement"],
        "decision_contract": "class-analysis-patch-decision-v3",
    }
    stale = page.evaluate(
        "(capabilities) => window.__TATOR_TEST_HOOKS__.classSplitSetCapabilities(capabilities)",
        stale_contract,
    )
    assert stale["disabled"] is True
    assert stale["checked"] is False
    assert "unsupported patch-refinement contract" in stale["hint"]


def test_class_split_refined_vignette_categories_and_show_all_keep_analysis_totals(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)
    result = _mock_v6_refined_class_split_result()
    truck_one = next(point for point in result["points"] if point["point_id"] == "truck-1")
    truck_one["human_review_disposition"] = "skip"

    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_class_split_refined_categories_job'
        )""",
        result,
    )
    default_state = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitRefinementSnapshot()"
    )
    assert default_state["showAllRough"] is True
    assert default_state["visibleIds"] == [
        "truck-2",
        "truck-pair",
        "person-2",
        "person-1",
    ]
    assert page.eval_on_selector(
        "#classSplitRefinementQualityBanner", "element => element.hidden"
    ) is True

    priority = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitSetRefinementFilter('review_queue', false)"
    )
    assert priority["category"] == "review_queue"
    assert priority["visibleIds"] == ["truck-2", "truck-pair"]
    assert priority["immutableCounts"] == {
        "confirmed_outlier": 1,
        "explained_not_outlier": 1,
        "mixed_or_composite": 1,
        "unresolved": 1,
        "pair_conflict": 1,
        "review_queue": 3,
        "rough_total": 5,
    }
    assert "suggested review queue" in (
        page.text_content("#classSplitWrongQueueStatus") or ""
    )
    assert page.text_content("#classSplitWrongPanelSummary") == "Suggested review queue"
    assert page.locator(
        ".class-split-wrong-item__badge--priority-human-review"
    ).count() == 2

    confirmed = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitSetRefinementFilter('confirmed_outlier')"
    )
    assert confirmed["visibleIds"] == ["truck-2"]
    assert "likely wrong" in (page.text_content("#classSplitWrongQueueStatus") or "")
    assert page.locator(".class-split-wrong-item__badge--confirmed_outlier").count() == 1
    wrong_list_text = page.text_content("#classSplitWrongList") or ""
    assert "directed-pair probe score" in wrong_list_text
    assert "audit only; not the decision score" in wrong_list_text

    explained = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitSetRefinementFilter('explained_not_outlier')"
    )
    assert explained["visibleIds"] == ["person-1"]
    assert "explained by overlap/context" in (page.text_content("#classSplitWrongQueueStatus") or "")

    show_all = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitSetRefinementFilter('explained_not_outlier', true)"
    )
    assert show_all["visibleIds"] == ["truck-2", "truck-pair", "person-2", "person-1"]
    assert show_all["immutableCounts"]["rough_total"] == 5
    assert "truck-1" not in show_all["visibleIds"]

    page.select_option("#classSplitDisplayMode", "rough_only")
    page.wait_for_function(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot().statusText.includes('all rough candidates')",
        timeout=15000,
    )
    graph_snapshot = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitPlotSnapshot()"
    )
    assert sum(graph_snapshot["tracePointCounts"]) == 4


def test_class_split_v6_priority_review_routes_ranked_unresolved_row_to_qwen(playwright_page):
    page, _ = playwright_page
    qwen_requests = []
    job_id = "pw_class_split_v6_priority_review_job"

    def qwen_review(route):
        qwen_requests.append(route.request.post_data_json or {})
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "review_id": "priority-unresolved-review",
                    "parent_job_id": job_id,
                    "point_id": "truck-1",
                    "status": "completed",
                    "progress": 1,
                    "message": "Priority unresolved review completed",
                    "evidence": [],
                    "result": {
                        "decision": "skip_uncertain",
                        "target_class": "Truck",
                        "confidence": 0.4,
                    },
                }
            ),
        )

    page.route(
        f"**/class_analysis/jobs/{job_id}/points/truck-1/qwen_review",
        qwen_review,
    )
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function(
        "!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000
    )
    page.evaluate(
        """async ({payload, jobId}) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            jobId
        )""",
        {"payload": _mock_v6_refined_class_split_result(), "jobId": job_id},
    )

    default_snapshot = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitRefinementSnapshot()"
    )
    assert default_snapshot["showAllRough"] is True

    priority = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitSetRefinementFilter('review_queue', false)"
    )
    assert priority["visibleIds"] == ["truck-2", "truck-1", "truck-pair"]
    unresolved_card = page.locator(
        '.class-split-wrong-item[data-point-id="truck-1"]'
    )
    assert unresolved_card.get_attribute("data-selector-priority-rank") == "2"
    assert unresolved_card.locator(
        ".class-split-wrong-item__badge--unresolved"
    ).count() == 1
    assert "Review priority #2" in (unresolved_card.text_content() or "")
    assert "candidate remains unresolved" in (unresolved_card.text_content() or "")

    unresolved_card.locator('[data-action="qwen-review"]').click()
    page.wait_for_function(
        """() => window.__TATOR_TEST_HOOKS__
            .classSplitQwenReviewSnapshot('truck-1')?.status === 'completed'""",
        timeout=5000,
    )
    assert len(qwen_requests) == 1


def test_class_split_v6_utility_sort_and_concise_selector_cards(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function(
        "!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000
    )
    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_class_split_v6_sort_job'
        )""",
        _mock_v6_refined_class_split_result(),
    )
    page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitSetVignetteSort('priority')"
    )

    snapshot = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitRefinementSnapshot()"
    )
    assert snapshot["sort"] == "priority"
    assert snapshot["visibleIds"] == [
        "truck-2",
        "truck-1",
        "truck-pair",
        "person-2",
        "person-1",
    ]
    assert "review priority" in (
        page.text_content("#classSplitWrongQueueStatus") or ""
    )

    first_card = page.locator('.class-split-wrong-item[data-point-id="truck-2"]')
    assert first_card.get_attribute("data-selector-priority-rank") == "1"
    assert "Review priority #1" in (first_card.text_content() or "")
    assert first_card.locator(".class-split-wrong-item__summary--selector").count() == 0
    metric_chips = first_card.locator(".class-split-selector-metric")
    assert metric_chips.count() == 3
    assert metric_chips.nth(0).text_content() == "Review value 86/100"
    assert metric_chips.nth(1).text_content() == "Actionable 90%"
    assert metric_chips.nth(2).text_content() == "Reviewability 80%"
    assert "not the probability that this label is wrong" in (
        metric_chips.nth(0).get_attribute("data-tooltip") or ""
    )
    assert metric_chips.nth(0).get_attribute("tabindex") == "0"
    wrong_list = page.locator("#classSplitWrongList")
    assert wrong_list.locator(
        ".class-split-wrong-item__rank-adjustment--semantic"
    ).count() == 0
    assert wrong_list.locator(
        ".class-split-wrong-item__rank-adjustment--frequency"
    ).count() == 0
    technical = first_card.locator(".class-split-wrong-item__technical")
    assert technical.get_attribute("open") is None
    assert technical.locator(".class-split-wrong-item__technical-body").is_visible() is False
    technical_text = technical.text_content() or ""
    assert "Technical details" in technical_text
    assert "Review-ranking evidence" in technical_text
    assert "Patch and calibration evidence" in technical_text
    assert "Decision reasons" in technical_text
    assert "Expected review value" in technical_text
    assert "Visual evidence state" in technical_text
    assert "global HGB actionability model" in technical_text
    assert "dataset-overlap evidence" in technical_text
    assert "ranking contribution=enabled" in technical_text
    technical_body_style = technical.locator(
        ".class-split-wrong-item__technical-body"
    ).evaluate(
        "el => ({fontSize: getComputedStyle(el).fontSize, overflowWrap: getComputedStyle(el).overflowWrap, wordBreak: getComputedStyle(el).wordBreak})"
    )
    assert technical_body_style == {
        "fontSize": "12px",
        "overflowWrap": "anywhere",
        "wordBreak": "break-word",
    }
    overlap_chip = page.locator(
        '.class-split-wrong-item[data-point-id="person-2"] '
        ".class-split-wrong-item__dataset-overlap"
    )
    assert overlap_chip.text_content() == (
        "Common overlap pattern · 42% co-occurrence · used in ranking"
    )
    assert overlap_chip.get_attribute("tabindex") == "0"
    assert "contributes to review ranking" in (
        overlap_chip.get_attribute("data-tooltip") or ""
    )
    assert first_card.locator(
        ".class-split-wrong-item__dataset-overlap"
    ).count() == 0

    suspicion = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitSetVignetteSort('suspicion')"
    )
    assert suspicion["visibleIds"] == [
        "truck-2",
        "truck-1",
        "truck-pair",
        "person-2",
        "person-1",
    ]
    assert "most likely wrong first" in suspicion["statusText"]

    least_suspicious = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitSetVignetteSort('suspicion_ascending')"
    )
    assert least_suspicious["visibleIds"] == [
        "person-1",
        "person-2",
        "truck-pair",
        "truck-1",
        "truck-2",
    ]
    assert "least likely wrong first" in least_suspicious["statusText"]

    analysis_order = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitSetVignetteSort('analysis_order')"
    )
    assert analysis_order["visibleIds"] == [
        "truck-2",
        "person-1",
        "person-2",
        "truck-1",
        "truck-pair",
    ]
    assert "original analysis order" in analysis_order["statusText"]

    page.evaluate("document.documentElement.classList.add('theme-dark')")
    unresolved_badge = page.locator(
        '.class-split-wrong-item[data-point-id="truck-1"] '
        ".class-split-wrong-item__badge--unresolved"
    )
    dark_colors = unresolved_badge.evaluate(
        "el => ({color: getComputedStyle(el).color, background: getComputedStyle(el).backgroundColor})"
    )
    assert dark_colors == {
        "color": "rgb(241, 245, 249)",
        "background": "rgb(30, 41, 59)",
    }
    page.set_viewport_size({"width": 720, "height": 900})
    technical.locator("summary").click()
    mobile_technical_layout = technical.evaluate(
        "el => ({clientWidth: el.clientWidth, scrollWidth: el.scrollWidth, fontSize: getComputedStyle(el.querySelector('.class-split-wrong-item__technical-body')).fontSize})"
    )
    assert mobile_technical_layout["fontSize"] == "13px"
    assert mobile_technical_layout["scrollWidth"] <= mobile_technical_layout["clientWidth"] + 1


def test_class_split_completed_truncated_refinement_preserves_stage_one_membership(
    playwright_page,
):
    page, _ = playwright_page
    result = _mock_v6_refined_class_split_result()
    expected_stage_one_ids = [
        row["point_id"] for row in result["wrong_class_candidates"]
    ]
    result["refinement_candidates"] = result["refinement_candidates"][:2]

    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function(
        "!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000
    )
    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_class_split_truncated_completed_refinement_job'
        )""",
        result,
    )
    analysis_order = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitSetVignetteSort('analysis_order')"
    )

    assert analysis_order["visibleIds"] == expected_stage_one_ids
    assert "Review priority (unavailable)" in (
        page.locator('#classSplitVignetteSort option[value="priority"]')
        .text_content()
        or ""
    )
    assert page.locator("[data-selector-priority-rank]").count() == 0
    assert page.locator("#classSplitWrongList .class-split-wrong-item").count() == len(
        expected_stage_one_ids
    )


def test_class_split_unbound_priority_rank_falls_back_visibly_to_stage_one(playwright_page):
    page, _ = playwright_page
    result = _mock_v6_refined_class_split_result()
    for collection_name in (
        "points",
        "refinement_candidates",
        "wrong_class_candidates",
        "vignette_candidates",
    ):
        for row in result.get(collection_name, []):
            if row.get("point_id") == "person-1":
                row.get("refined_outlier", {})["selector_priority_contract"] = (
                    "stale-selector-priority-contract"
                )

    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function(
        "!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000
    )
    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_class_split_incomplete_priority_job'
        )""",
        result,
    )
    page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitSetVignetteSort('priority')"
    )

    snapshot = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitRefinementSnapshot()"
    )
    assert snapshot["sort"] == "suspicion"
    assert snapshot["visibleIds"] == [
        "truck-pair",
        "truck-2",
        "person-1",
        "person-2",
        "truck-1",
    ]
    assert page.locator(
        '#classSplitVignetteSort option[value="priority"]'
    ).text_content() == "Review priority (unavailable)"
    assert "most likely wrong first" in (
        page.text_content("#classSplitWrongQueueStatus") or ""
    )
    wrong_list_text = page.text_content("#classSplitWrongList") or ""
    assert page.locator("[data-selector-priority-rank]").count() == 0
    assert "Priority #" not in wrong_list_text
    assert "Semantic overlap" not in wrong_list_text
    assert "Triage frequency" not in wrong_list_text
    assert "saved priority data does not match every candidate" in (
        page.locator('#classSplitVignetteSort option[value="priority"]')
        .get_attribute("title")
        or ""
    )


def test_class_split_stale_selector_v5_payload_fails_closed(playwright_page):
    page, _ = playwright_page
    result = _mock_v6_refined_class_split_result()
    for collection_name in (
        "points",
        "refinement_candidates",
        "wrong_class_candidates",
        "vignette_candidates",
    ):
        for row in result.get(collection_name, []):
            evidence = row.get("refined_outlier", {})
            if "selector_v6" in evidence:
                evidence["selector_v5"] = evidence.pop("selector_v6")

    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function(
        "!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000
    )
    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_class_split_stale_v5_job'
        )""",
        result,
    )
    snapshot = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitRefinementSnapshot()"
    )
    assert snapshot["visibleIds"] == [
        "truck-pair",
        "truck-2",
        "person-1",
        "person-2",
        "truck-1",
    ]
    assert page.locator("[data-selector-priority-rank]").count() == 0
    wrong_list_text = page.text_content("#classSplitWrongList") or ""
    assert "Priority #" not in wrong_list_text
    assert "Review value" not in wrong_list_text
    assert "Review priority (unavailable)" in (
        page.locator('#classSplitVignetteSort option[value="priority"]')
        .text_content()
        or ""
    )


def test_class_split_failed_selector_runs_never_advertise_synthetic_priority(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function(
        "!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000
    )

    for technical_status in ("failed", "partial", "cancelled"):
        result = _mock_v6_refined_class_split_result()
        result["summary"]["refinement"]["status"] = technical_status
        result["summary"]["refinement"]["queue_policy"].update(
            {
                "automatic_rough_fallback": True,
                "fallback_reason": f"refinement_{technical_status}",
                "default_queue": "rough_candidates",
            }
        )
        page.evaluate(
            """async ({payload, status}) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
                payload,
                `pw_class_split_${status}_synthetic_priority_job`
            )""",
            {"payload": result, "status": technical_status},
        )
        page.evaluate(
            "() => window.__TATOR_TEST_HOOKS__.classSplitSetVignetteSort('priority')"
        )

        snapshot = page.evaluate(
            "() => window.__TATOR_TEST_HOOKS__.classSplitRefinementSnapshot()"
        )
        assert snapshot["visibleIds"] == [
            "truck-pair",
            "truck-2",
            "person-1",
            "person-2",
            "truck-1",
        ]
        expected_reason = f"refinement {technical_status}"
        assert page.locator(
            '#classSplitVignetteSort option[value="priority"]'
        ).text_content() == "Review priority (unavailable)"
        assert "most likely wrong first" in (
            page.text_content("#classSplitWrongQueueStatus") or ""
        )
        assert expected_reason in (
            page.locator('#classSplitVignetteSort option[value="priority"]')
            .get_attribute("title")
            or ""
        )
        assert page.locator("[data-selector-priority-rank]").count() == 0
        assert "Priority #" not in (
            page.text_content("#classSplitWrongList") or ""
        )


def test_class_split_v6_non_actionable_result_keeps_complete_ranked_queue(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)
    result = _mock_v6_refined_class_split_result()
    for candidate in result["refinement_candidates"]:
        if candidate.get("refined_outlier", {}).get("status") != "pair_conflict":
            candidate["refined_outlier"]["status"] = "unresolved"
            candidate["include_in_refined_vignettes"] = False
    result["vignette_candidates"] = []
    result["summary"]["refinement"].update(
        {
            "quality_status": "completed_non_actionable",
            "vignette_candidate_count": 0,
            "quality_gate": {
                "passed": False,
                "reasons": [
                    "resolved_rate_below_release_gate",
                    "confirmation_eligible_pair_coverage_below_release_gate",
                ],
                "thresholds": {
                    "minimum_resolved_rate": 0.5,
                    "minimum_confirmation_eligible_pair_coverage": 0.75,
                },
                "metrics": {
                    "regular_evaluated_count": 4,
                    "resolved_count": 0,
                    "resolved_rate": 0.0,
                    "terminal_decisive_count": 0,
                    "terminal_decisive_rate": 0.0,
                    "unresolved_count": 4,
                    "unresolved_rate": 1.0,
                    "confirmation_eligible_pair_coverage": 0.0,
                },
            },
            "queue_policy": {
                "mode": "selector_ranked_complete_stage1",
                "automatic_rough_fallback": False,
                "fallback_reason": "",
                "default_queue": "selector_ranked_stage1_candidates",
                "confirmed_count": 0,
                "pair_conflict_count": 1,
                "refined_review_candidate_count": 1,
                "effective_default_candidate_count": 5,
                "rough_count": 5,
            },
            "category_counts": {
                "confirmed_outlier": 0,
                "explained_not_outlier": 0,
                "mixed_or_composite": 0,
                "unresolved": 4,
                "pair_conflict": 1,
            },
        }
    )

    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_class_split_v6_non_actionable_job'
        )""",
        result,
    )

    snapshot = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitRefinementSnapshot()"
    )
    assert snapshot["showAllRough"] is True
    assert set(snapshot["visibleIds"]) == {
        "truck-2",
        "person-1",
        "person-2",
        "truck-1",
        "truck-pair",
    }
    assert page.is_checked("#classSplitShowAllRough") is True
    banner = page.text_content("#classSplitRefinementQualityBanner") or ""
    assert banner == ""
    assert page.eval_on_selector(
        "#classSplitRefinementQualityBanner", "element => element.hidden"
    ) is True


def test_class_split_disabled_refinement_keeps_legacy_vignette_queue(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)
    result = _mock_class_split_result()
    result["summary"]["refinement"] = {
        "enabled": False,
        "status": "disabled",
        "rough_candidate_count": len(result["wrong_class_candidates"]),
        "evaluated_count": 0,
        "vignette_candidate_count": 0,
    }
    result["within_class_outlier_candidates"] = []
    result["refinement_candidates"] = []
    result["vignette_candidates"] = []

    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_class_split_refinement_disabled_job'
        )""",
        result,
    )
    snapshot = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitRefinementSnapshot()"
    )
    assert snapshot["refined"] is False
    assert snapshot["visibleIds"] == ["truck-2"]
    assert snapshot["immutableCounts"]["confirmed_outlier"] == 1
    assert page.locator("#classSplitVignetteCategory").is_disabled()
    assert "pooled likely-wrong candidate" in (
        page.text_content("#classSplitVignetteCategoryCounts") or ""
    )


def test_class_split_partial_refinement_preserves_full_rough_queue(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)
    result = _mock_refined_class_split_result()
    result["summary"]["refinement"]["status"] = "partial"
    result["refinement_candidates"] = result["refinement_candidates"][:2]

    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_class_split_refinement_partial_job'
        )""",
        result,
    )
    page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitSetVignetteSort('priority')"
    )
    default_snapshot = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitRefinementSnapshot()"
    )
    assert default_snapshot["showAllRough"] is True
    assert default_snapshot["visibleIds"] == [
        "truck-pair",
        "truck-2",
        "person-1",
        "person-2",
        "truck-1",
    ]
    assert "Analysis totals (partial)" in (
        page.text_content("#classSplitVignetteCategoryCounts") or ""
    )

    rough_snapshot = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitSetRefinementFilter('confirmed_outlier', true)"
    )
    assert rough_snapshot["visibleIds"] == [
        "truck-pair",
        "truck-2",
        "person-1",
        "person-2",
        "truck-1",
    ]


def test_class_split_selected_refinement_uses_anomaly_language(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function("!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult", timeout=15000)
    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_class_split_selected_refinement_job'
        )""",
        _mock_refined_class_split_result(selected_class=True),
    )
    page.wait_for_selector('.class-split-wrong-item[data-point-id="truck-2"]', timeout=15000)

    assert "Potential issues" in (
        page.text_content("#classSplitWrongPanelSummary") or ""
    )
    page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitSetRefinementFilter('confirmed_outlier', false)"
    )
    assert "Likely anomalous objects" in (page.text_content("#classSplitWrongPanelSummary") or "")
    assert "visual anomaly review" in (page.text_content(".class-split-wrong-item") or "")
    assert "Diagnostic alternative: Person" in (page.text_content(".class-split-wrong-item") or "")
    assert page.eval_on_selector(
        '.class-split-wrong-item[data-point-id="truck-2"] [data-action="target-class"]',
        "element => element.value",
    ) == ""
    assert page.eval_on_selector(
        '#classSplitDisplayMode option[value="wrong_only"]',
        "element => element.textContent",
    ) == "Fine-grained anomaly subset"


def test_class_split_live_qwen_poll_preserves_card_disclosures_and_refinement_dom(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function(
        "!!window.__TATOR_TEST_HOOKS__?.classSplitApplyQwenReview",
        timeout=15000,
    )
    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_class_split_qwen_keyed_job'
        )""",
        _mock_refined_class_split_result(),
    )
    page.wait_for_selector(
        '.class-split-wrong-item[data-point-id="truck-2"]',
        timeout=15000,
    )
    page.evaluate(
        """() => {
            const card = document.querySelector('.class-split-wrong-item[data-point-id="truck-2"]');
            const refinement = card.querySelector('.class-split-refinement-evidence');
            const body = refinement.querySelector('.class-split-refinement-evidence__body');
            card.dataset.testIdentity = 'same-card';
            refinement.open = true;
            refinement.dataset.refinementLoaded = '1';
            body.dataset.testIdentity = 'same-refinement';
            body.innerHTML = '<img data-loaded-preview alt="loaded patch preview">';
        }"""
    )
    artifact = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ZQmcAAAAASUVORK5CYII="
    )
    page.evaluate(
        """() => {
            const toggle = document.querySelector('#classSplitQwenReviewTraceToggle');
            toggle.checked = true;
            toggle.dispatchEvent(new Event('change', { bubbles: true }));
        }"""
    )
    first_review = {
        "review_id": "qwen-keyed",
        "parent_job_id": "pw_class_split_qwen_keyed_job",
        "point_id": "truck-2",
        "status": "running",
        "progress": 0.35,
        "message": "Inspecting target evidence",
        "active_generation": {
            "phase": "specificity probe",
            "text": "first streamed fragment",
        },
        "evidence": [
            {
                "evidence_id": "target_detail_1",
                "kind": "target_detail",
                "title": "Target detail",
                "artifact_url": artifact,
            }
        ],
        "trace_events": [
            {
                "type": "model_output",
                "phase": f"probe_{index}",
                "text": f"raw output {index} " + ("evidence " * 35),
            }
            for index in range(8)
        ],
    }
    page.evaluate(
        "(review) => window.__TATOR_TEST_HOOKS__.classSplitApplyQwenReview(review)",
        first_review,
    )
    page.evaluate(
        """() => {
            const card = document.querySelector('.class-split-wrong-item[data-point-id="truck-2"]');
            card.querySelector('.class-split-qwen-review').dataset.testIdentity = 'same-qwen-root';
            card.querySelector('.class-split-qwen-review__audit').open = true;
            const traceDetails = document.querySelector(
                '#classSplitQwenReviewTraceBody details[data-disclosure-key^="trace-raw-outputs-"]'
            );
            if (traceDetails) traceDetails.open = true;
            const traceBody = document.querySelector('#classSplitQwenReviewTraceBody');
            traceBody.scrollTop = Math.min(40, Math.max(0, traceBody.scrollHeight - traceBody.clientHeight - 30));
            traceBody.dispatchEvent(new Event('scroll'));
        }"""
    )
    second_review = {
        **first_review,
        "progress": 0.62,
        "message": "Comparing overlap evidence",
        "active_generation": {
            "phase": "specificity probe",
            "text": "first streamed fragment plus second fragment",
        },
        "evidence": [
            *first_review["evidence"],
            {
                "evidence_id": "overlap_2",
                "kind": "overlap_decomposition",
                "title": "Overlap decomposition",
                "artifact_url": artifact,
            },
        ],
        "trace_events": [
            *first_review["trace_events"],
            {
                "type": "model_output",
                "phase": "probe_latest",
                "text": "latest raw streamed output",
            },
        ],
    }
    snapshot = page.evaluate(
        "(review) => window.__TATOR_TEST_HOOKS__.classSplitApplyQwenReview(review)",
        second_review,
    )
    assert snapshot["cardMarker"] == "same-card"
    assert snapshot["qwenMarker"] == "same-qwen-root"
    assert snapshot["refinementMarker"] == "same-refinement"
    assert snapshot["auditOpen"] is True
    assert snapshot["auditItems"] == 2
    assert snapshot["liveText"] == "first streamed fragment plus second fragment"
    assert page.eval_on_selector(
        '#classSplitQwenReviewTraceBody details[data-disclosure-key^="trace-raw-outputs-"]',
        "details => details.open",
    ) is True
    assert "first streamed fragment plus second fragment" in (
        page.text_content("#classSplitQwenReviewTraceBody") or ""
    )
    assert page.locator(
        '.class-split-wrong-item[data-point-id="truck-2"] [data-loaded-preview]'
    ).count() == 1

    completed_review = {
        **second_review,
        "status": "completed",
        "progress": 1,
        "active_generation": None,
        "result": {
            "decision": "confirm_current",
            "target_class": "Truck",
            "confidence": 0.91,
            "reviewed_by_model": "fixture",
        },
    }
    completed = page.evaluate(
        "(review) => window.__TATOR_TEST_HOOKS__.classSplitApplyQwenReview(review)",
        completed_review,
    )
    assert completed["cardMarker"] == "same-card"
    assert completed["refinementMarker"] == "same-refinement"
    assert completed["auditOpen"] is True
    assert page.locator(
        '.class-split-wrong-item[data-point-id="truck-2"] [data-loaded-preview]'
    ).count() == 1


def test_class_split_failed_replacement_restores_qwen_and_restore_cannot_erase_new_review(
    playwright_page,
):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function(
        "!!window.__TATOR_TEST_HOOKS__?.classSplitRestoreQwenReviews",
        timeout=15000,
    )
    job_id = "pw_class_split_qwen_restore_job"
    page.evaluate(
        """async ({payload, jobId}) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            jobId
        )""",
        {"payload": _mock_refined_class_split_result(), "jobId": job_id},
    )
    completed_review = {
        "review_id": "qwen-before-replacement",
        "parent_job_id": job_id,
        "point_id": "truck-2",
        "status": "completed",
        "progress": 1,
        "message": "Completed before replacement",
        "evidence": [],
        "result": {
            "decision": "confirm_current",
            "target_class": "Truck",
            "confidence": 0.9,
        },
    }
    page.evaluate(
        "(review) => window.__TATOR_TEST_HOOKS__.classSplitApplyQwenReview(review)",
        completed_review,
    )
    page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitSimulateFailedStartAfterClear('fixture')"
    )
    restored = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitQwenReviewSnapshot('truck-2')"
    )
    assert restored["review_id"] == "qwen-before-replacement"
    assert restored["result"]["decision"] == "confirm_current"
    assert "Confirm current class" in (
        page.text_content(
            '.class-split-wrong-item[data-point-id="truck-2"] [data-qwen-review-block]'
        )
        or ""
    )

    # Hold the server restore response, create a newer local review, then
    # release an older same-point record. The late restore must merge rather
    # than clear/overwrite the new review.
    page.evaluate(
        """(jobId) => {
            const originalFetch = window.fetch.bind(window);
            window.__qwenRestoreOriginalFetch = originalFetch;
            window.fetch = (input, init) => {
                const url = String(input || '');
                if (url.includes(`/class_analysis/jobs/${jobId}/qwen_reviews`)) {
                    return new Promise((resolve) => {
                        window.__releaseQwenRestore = (payload) => resolve(
                            new Response(JSON.stringify(payload), {
                                status: 200,
                                headers: {'Content-Type': 'application/json'},
                            })
                        );
                    });
                }
                return originalFetch(input, init);
            };
            window.__qwenRestorePromise = (
                window.__TATOR_TEST_HOOKS__.classSplitRestoreQwenReviews(jobId)
            );
        }""",
        job_id,
    )
    new_review = {
        **completed_review,
        "review_id": "qwen-new-concurrent-review",
        "status": "completed",
        "progress": 1,
        "message": "New concurrent review",
        "result": {
            "decision": "skip_uncertain",
            "target_class": "Truck",
            "confidence": 0.4,
        },
    }
    page.evaluate(
        "(review) => window.__TATOR_TEST_HOOKS__.classSplitApplyQwenReview(review)",
        new_review,
    )
    page.evaluate(
        """async (oldReview) => {
            window.__releaseQwenRestore({reviews: [oldReview]});
            await window.__qwenRestorePromise;
            window.fetch = window.__qwenRestoreOriginalFetch;
        }""",
        completed_review,
    )
    merged = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitQwenReviewSnapshot('truck-2')"
    )
    assert merged["review_id"] == "qwen-new-concurrent-review"
    assert merged["message"] == "New concurrent review"


def test_class_split_pipboy_theme_keeps_plot_and_class_colors(playwright_page):
    page, _ = playwright_page
    page.evaluate(
        """() => {
            window.localStorage.setItem('tator.themeMode', 'pipboy');
            window.localStorage.setItem('tator.pipboyAccent', 'green');
        }"""
    )
    _reload_class_split_test_page(page)

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
    _reload_class_split_test_page(page)


def _install_deferred_review_fetch(page):
    page.evaluate(
        r"""() => {
            const originalFetch = window.fetch.bind(window);
            const state = {
                originalFetch,
                pending: [],
                requests: [],
                active: 0,
                maxActive: 0,
                receiptSequence: 0,
            };
            window.__classSplitDeferredReviewFetch = state;
            window.fetch = (input, options = {}) => {
                const url = String(input?.url || input || '');
                if (!url.includes('/review_disposition')) {
                    return originalFetch(input, options);
                }
                const payload = JSON.parse(String(options?.body || '{}'));
                const match = url.match(/jobs\/([^/]+)\/points\/([^/]+)\/review_disposition/);
                const request = {
                    url,
                    payload,
                    jobId: decodeURIComponent(match?.[1] || ''),
                    pointId: decodeURIComponent(match?.[2] || ''),
                };
                state.requests.push(request);
                state.active += 1;
                state.maxActive = Math.max(state.maxActive, state.active);
                return new Promise((resolve) => {
                    state.pending.push({ ...request, resolve });
                });
            };
            window.__resolveNextClassSplitReview = ({ status = 200, code = '', pointId = '' } = {}) => {
                const pendingIndex = pointId
                    ? state.pending.findIndex(entry => entry.pointId === pointId)
                    : 0;
                const [entry] = pendingIndex >= 0
                    ? state.pending.splice(pendingIndex, 1)
                    : [];
                if (!entry) {
                    throw new Error('No deferred review request is pending.');
                }
                state.active = Math.max(0, state.active - 1);
                if (status >= 400) {
                    entry.resolve(new Response(JSON.stringify({
                        detail: { code: code || 'synthetic_review_failure' },
                    }), {
                        status,
                        headers: { 'Content-Type': 'application/json' },
                    }));
                    return entry.pointId;
                }
                state.receiptSequence += 1;
                const clear = entry.payload.disposition === 'clear';
                const receipt = {
                    schema: 'class-analysis-review-disposition-v3',
                    status: clear ? 'cleared' : 'recorded',
                    job_id: entry.jobId,
                    point_id: entry.pointId,
                    disposition: entry.payload.disposition,
                    client_action_id: entry.payload.client_action_id,
                    training_capture_requested: entry.payload.capture_training_data === true,
                    review_object_key: `cro_${'a'.repeat(64)}`,
                };
                if (clear) {
                    const prior = [...state.requests].reverse().find(request => (
                        request.pointId === entry.pointId
                        && request.payload.disposition !== 'clear'
                    ));
                    receipt.previous_disposition = prior?.payload.disposition || '';
                } else {
                    receipt.human_reviewed_at = 1785800000;
                    receipt.human_review_revision = `rdr1_${state.receiptSequence.toString(16).padStart(32, '0')}`;
                    receipt.origin = 'desktop';
                }
                entry.resolve(new Response(JSON.stringify(receipt), {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' },
                }));
                return entry.pointId;
            };
        }"""
    )


def _resolve_deferred_review(page, *, status=200, code="", point_id=""):
    if point_id:
        page.wait_for_function(
            """pointId => (window.__classSplitDeferredReviewFetch?.pending || [])
                .some(entry => entry.pointId === pointId)""",
            arg=point_id,
            timeout=3000,
        )
    else:
        page.wait_for_function(
            "() => (window.__classSplitDeferredReviewFetch?.pending.length || 0) > 0",
            timeout=3000,
        )
    return page.evaluate(
        "args => window.__resolveNextClassSplitReview(args)",
        {"status": status, "code": code, "pointId": point_id},
    )


def _restore_deferred_review_fetch(page):
    page.evaluate(
        """() => {
            const state = window.__classSplitDeferredReviewFetch;
            if (state?.originalFetch) {
                window.fetch = state.originalFetch;
            }
            delete window.__resolveNextClassSplitReview;
            delete window.__classSplitDeferredReviewFetch;
        }"""
    )


def _install_deferred_thumbnail_fetch(page):
    page.evaluate(
        r"""() => {
            const originalFetch = window.fetch.bind(window);
            const state = {
                originalFetch,
                pending: [],
                requests: [],
                active: 0,
                maxActive: 0,
                abortedSignals: 0,
            };
            window.__classSplitDeferredThumbnailFetch = state;
            window.fetch = (input, options = {}) => {
                const url = String(input?.url || input || '');
                const match = url.match(/\/thumbnail\/([^?]+)/);
                if (!match) {
                    return originalFetch(input, options);
                }
                const request = {
                    url,
                    pointId: decodeURIComponent(match[1]),
                    aborted: false,
                    networkActive: true,
                };
                state.requests.push(request);
                state.active += 1;
                state.maxActive = Math.max(state.maxActive, state.active);
                const releaseNetworkSlot = () => {
                    if (request.networkActive) {
                        request.networkActive = false;
                        state.active = Math.max(0, state.active - 1);
                    }
                };
                options.signal?.addEventListener('abort', () => {
                    request.aborted = true;
                    state.abortedSignals += 1;
                    releaseNetworkSlot();
                }, { once: true });
                return new Promise((resolve) => {
                    state.pending.push({
                        ...request,
                        resolve: () => {
                            releaseNetworkSlot();
                            const bytes = Uint8Array.from(
                                atob('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='),
                                value => value.charCodeAt(0)
                            );
                            resolve(new Response(bytes, {
                                status: 200,
                                headers: { 'Content-Type': 'image/png' },
                            }));
                        },
                    });
                });
            };
            window.__resolveClassSplitThumbnail = (pointId) => {
                const index = state.pending.findIndex(entry => entry.pointId === pointId);
                if (index < 0) {
                    throw new Error(`No deferred thumbnail request for ${pointId}.`);
                }
                const [entry] = state.pending.splice(index, 1);
                entry.resolve();
            };
        }"""
    )


def _resolve_deferred_thumbnail(page, point_id):
    page.wait_for_function(
        """pointId => (window.__classSplitDeferredThumbnailFetch?.pending || [])
            .some(entry => entry.pointId === pointId)""",
        arg=point_id,
        timeout=3000,
    )
    page.evaluate("pointId => window.__resolveClassSplitThumbnail(pointId)", point_id)


def _restore_deferred_thumbnail_fetch(page):
    page.evaluate(
        """() => {
            const state = window.__classSplitDeferredThumbnailFetch;
            if (state?.originalFetch) {
                window.fetch = state.originalFetch;
            }
            delete window.__resolveClassSplitThumbnail;
            delete window.__classSplitDeferredThumbnailFetch;
        }"""
    )


def _reload_class_split_test_page(page):
    page.reload(wait_until="load")
    page.wait_for_function(
        "!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult",
        timeout=15000,
    )


def test_class_split_multi_selection_panel_pages_and_has_one_live_region(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function(
        "!!window.__TATOR_TEST_HOOKS__?.classSplitEmitSelection",
        timeout=15000,
    )
    result = _mock_class_split_many_wrong_result(40)
    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_multi_selection_panel_job'
        )""",
        result,
    )
    point_ids = [point["point_id"] for point in result["points"]]
    snapshot = page.evaluate(
        "ids => window.__TATOR_TEST_HOOKS__.classSplitEmitSelection(ids)",
        point_ids,
    )

    assert snapshot["panelVisible"] is True
    assert snapshot["pointIds"] == point_ids
    assert snapshot["renderedCount"] == 36
    assert page.locator("#classSplitBulkPanel [role='status'][aria-live='polite']").count() == 1
    assert page.locator(".class-split-multi-selection__spinner[role='status']").count() == 0
    assert page.locator(".class-split-multi-selection__spinner[aria-hidden='true']").count() == 36

    page.locator(".class-split-multi-selection__tile").first.click()
    page.wait_for_timeout(100)
    removed_snapshot = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot()"
    )
    assert len(removed_snapshot["pointIds"]) == 39, removed_snapshot
    refill_diagnostics = {
        key: removed_snapshot[key]
        for key in (
            "renderedCount",
            "generation",
            "thumbnailBatchCurrent",
            "currentJobId",
            "thumbnailJobId",
            "panelRenderCount",
            "queuedCount",
            "activeLoads",
        )
    }
    assert 36 <= removed_snapshot["renderedCount"] <= 39, refill_diagnostics
    if removed_snapshot["renderedCount"] < 39:
        page.click("#classSplitBulkLoadMore")
    page.wait_for_function(
        "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot().renderedCount === 39"
    )
    loaded_snapshot = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot()"
    )
    assert loaded_snapshot["renderedCount"] == 39


def test_class_split_structured_progress_and_preview_preserve_visible_queue(
    playwright_page,
):
    page, _ = playwright_page
    preview_requests = []
    preview_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ZQmcAAAAASUVORK5CYII="
    )

    def refinement_preview(route):
        preview_requests.append(route.request.url)
        route.fulfill(
            status=200,
            content_type="image/png",
            body=preview_png,
        )

    page.route("**/refinement_preview", refinement_preview)
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.wait_for_function(
        "!!window.__TATOR_TEST_HOOKS__?.classSplitApplyResult",
        timeout=15000,
    )

    progress_text = page.evaluate(
        """() => window.__TATOR_TEST_HOOKS__.classSplitRenderProgress({
            progress: 0.91,
            stage: 'Refining patch evidence',
            stage_index: 5,
            stage_count: 6,
            stage_processed: 1248,
            stage_total: 5488,
            message: 'Scoring candidate patch grids'
        })"""
    )
    assert "Stage 5/6 · Refining patch evidence · 1,248/5,488" in progress_text
    assert "Scoring candidate patch grids" in progress_text

    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_class_split_refined_preview_job'
        )""",
        _mock_refined_class_split_result(),
    )
    page.click(
        '.class-split-wrong-item[data-point-id="truck-2"] '
        ".class-split-refinement-evidence > summary"
    )
    page.wait_for_selector(
        '.class-split-wrong-item[data-point-id="truck-2"] '
        ".class-split-refinement-evidence__body img",
        timeout=15000,
    )
    assert len(preview_requests) == 1
    page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitSetRefinementFilter('confirmed_outlier')"
    )
    assert page.locator(
        '.class-split-wrong-item[data-point-id="truck-2"] '
        ".class-split-refinement-evidence"
    ).get_attribute("open") == ""
    page.wait_for_selector(
        '.class-split-wrong-item[data-point-id="truck-2"] '
        ".class-split-refinement-evidence__body img",
        timeout=15000,
    )
    # Playwright routing disables the browser's private HTTP cache, so the
    # rerender may issue one extra request while preserving disclosure state.
    assert len(preview_requests) in {1, 2}
    assert page.locator(
        '.class-split-wrong-item[data-point-id="truck-2"]'
    ).count() == 1


def test_class_split_thumbnail_concurrency_and_stale_selection_responses(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    result = _mock_class_split_many_wrong_result(30)
    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_thumbnail_stale_job'
        )""",
        result,
    )
    first_ids = [point["point_id"] for point in result["points"][:12]]
    second_ids = [point["point_id"] for point in result["points"][18:30]]
    _install_deferred_thumbnail_fetch(page)
    try:
        page.evaluate(
            "ids => window.__TATOR_TEST_HOOKS__.classSplitEmitSelection(ids)",
            first_ids,
        )
        page.wait_for_function(
            "() => window.__classSplitDeferredThumbnailFetch.active === 6"
        )
        old_pending_ids = page.evaluate(
            "() => window.__classSplitDeferredThumbnailFetch.pending.map(entry => entry.pointId)"
        )
        assert len(old_pending_ids) == 6

        page.evaluate(
            "ids => window.__TATOR_TEST_HOOKS__.classSplitEmitSelection(ids)",
            second_ids,
        )
        page.wait_for_function(
            """() => window.__classSplitDeferredThumbnailFetch.abortedSignals === 6
                && window.__classSplitDeferredThumbnailFetch.active === 6"""
        )
        for point_id in old_pending_ids:
            _resolve_deferred_thumbnail(page, point_id)
        page.wait_for_timeout(50)
        current_tiles = page.locator(".class-split-multi-selection__tile")
        assert current_tiles.evaluate_all(
            "tiles => tiles.map(tile => tile.dataset.multiPointId)"
        ) == second_ids
        assert page.locator(".class-split-multi-selection__tile.is-loaded").count() == 0
        assert page.locator(".class-split-multi-selection__tile.is-error").count() == 0

        for point_id in second_ids:
            _resolve_deferred_thumbnail(page, point_id)
        page.wait_for_function(
            """() => document.querySelectorAll(
                '.class-split-multi-selection__tile.is-loaded'
            ).length === 12""",
            timeout=5000,
        )
        state = page.evaluate(
            """() => ({
                maxActive: window.__classSplitDeferredThumbnailFetch.maxActive,
                active: window.__classSplitDeferredThumbnailFetch.active,
                abortedSignals: window.__classSplitDeferredThumbnailFetch.abortedSignals,
            })"""
        )
        assert state == {"maxActive": 6, "active": 0, "abortedSignals": 6}
    finally:
        _restore_deferred_thumbnail_fetch(page)


def test_class_split_vignette_hover_click_scroll_and_keyboard_access(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.route(
        "**/thumbnail/**",
        lambda route: route.fulfill(
            status=200,
            content_type="image/svg+xml",
            body='<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2"><rect width="2" height="2" fill="white"/></svg>',
        ),
    )
    try:
        result = _mock_class_split_many_wrong_result(40)
        page.evaluate(
            """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
                payload,
                'pw_vignette_interaction_job'
            )""",
            result,
        )
        point_ids = [point["point_id"] for point in result["points"]]
        before = page.evaluate(
            "ids => window.__TATOR_TEST_HOOKS__.classSplitEmitSelection(ids)",
            point_ids,
        )
        page.wait_for_function(
            """() => document.querySelectorAll(
                '.class-split-multi-selection__tile.is-loaded'
            ).length >= 36""",
            timeout=5000,
        )
        rendered_before_removal = page.evaluate(
            "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot()"
        )
        expected_rendered = min(
            rendered_before_removal["renderedCount"],
            len(point_ids) - 1,
        )

        tile = page.locator(".class-split-multi-selection__tile").nth(12)
        tile.hover()
        page.wait_for_function(
            """() => {
                const preview = document.querySelector('#classSplitGraphHoverPreview');
                return preview && !preview.hidden
                    && preview.classList.contains('is-wide-context')
                    && preview.textContent.includes('Click vignette to dismiss from multi-selection');
            }""",
            timeout=3000,
        )
        tile.dispatch_event("pointerleave")

        removal = page.evaluate(
            """() => {
                const scroller = document.querySelector('#classSplitBulkScroller');
                scroller.scrollTop = Math.min(160, scroller.scrollHeight - scroller.clientHeight);
                const beforeScroll = scroller.scrollTop;
                const tile = document.querySelectorAll('.class-split-multi-selection__tile')[12];
                const pointId = tile.dataset.multiPointId;
                tile.click();
                return { pointId, beforeScroll };
            }"""
        )
        page.wait_for_function(
            """expected => {
                const snapshot = window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot();
                return snapshot.pointIds.length === 39
                    && snapshot.renderedCount === expected;
            }""",
            arg=expected_rendered,
        )
        after_click = page.evaluate(
            "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot()"
        )
        assert removal["pointId"] not in after_click["pointIds"]
        assert after_click["renderedCount"] == expected_rendered
        assert after_click["panelRenderCount"] == before["panelRenderCount"]
        assert page.eval_on_selector(
            "#classSplitBulkScroller",
            "el => el.scrollTop",
        ) == removal["beforeScroll"]

        keyboard_tile = page.locator(".class-split-multi-selection__tile").nth(10)
        keyboard_point_id = keyboard_tile.get_attribute("data-multi-point-id")
        keyboard_tile.focus()
        page.keyboard.press("Enter")
        page.wait_for_function(
            "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot().pointIds.length === 38"
        )
        assert keyboard_point_id not in page.evaluate(
            "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot().pointIds"
        )
    finally:
        page.unroute("**/thumbnail/**")


@pytest.mark.parametrize("drag_mode", ["select", "lasso"])
def test_class_split_physical_plotly_multi_selection_and_single_click(
    playwright_page,
    drag_mode,
):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    result = _mock_class_split_many_wrong_result(12)
    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_physical_selection_job'
        )""",
        result,
    )
    page.select_option("#classSplitDragMode", drag_mode)
    page.wait_for_function(
        "mode => document.querySelector('#classSplitGraph').layout.dragmode === mode",
        arg=drag_mode,
        timeout=5000,
    )
    drag_layer = page.locator("#classSplitGraph .nsewdrag")
    drag_layer.scroll_into_view_if_needed()
    page.wait_for_timeout(100)
    bounds = drag_layer.bounding_box()
    assert bounds is not None
    point_bounds = page.evaluate(
        """() => {
            const graph = document.querySelector('#classSplitGraph');
            const rect = graph.getBoundingClientRect();
            const xaxis = graph._fullLayout.xaxis;
            const yaxis = graph._fullLayout.yaxis;
            return graph.data.flatMap(trace => (trace.x || []).map((x, index) => {
                const clientX = rect.left + xaxis._offset + xaxis.l2p(x);
                const clientY = rect.top + yaxis._offset + yaxis.l2p(trace.y[index]);
                return {
                    left: clientX - 3,
                    right: clientX + 3,
                    top: clientY - 3,
                    bottom: clientY + 3,
                };
            }));
        }"""
    )
    assert len(point_bounds) == len(result["points"])
    left = max(bounds["x"] + 2, min(rect["left"] for rect in point_bounds) - 12)
    top = max(bounds["y"] + 2, min(rect["top"] for rect in point_bounds) - 12)
    right = min(
        bounds["x"] + bounds["width"] - 2,
        max(rect["right"] for rect in point_bounds) + 12,
    )
    bottom = min(
        bounds["y"] + bounds["height"] - 2,
        max(rect["bottom"] for rect in point_bounds) + 12,
    )
    page.evaluate(
        """() => {
            window.__physicalPlotlySelectionEvents = [];
            document.querySelector('#classSplitGraph').on('plotly_selected', event => {
                window.__physicalPlotlySelectionEvents.push(
                    (event?.points || []).map(point => point.customdata)
                );
            });
        }"""
    )
    hit_target = page.evaluate(
        """({x, y}) => {
            const target = document.elementFromPoint(x, y);
            return {
                tag: target?.tagName || '',
                className: String(target?.getAttribute?.('class') || ''),
            };
        }""",
        {"x": left, "y": top},
    )
    page.mouse.move(left, top)
    page.mouse.down()
    if drag_mode == "select":
        page.mouse.move(right, bottom, steps=12)
    else:
        page.mouse.move(right, top, steps=6)
        page.mouse.move(right, bottom, steps=6)
        page.mouse.move(left, bottom, steps=6)
        page.mouse.move(left, top, steps=6)
    page.mouse.up()
    page.wait_for_timeout(300)
    selected = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot()"
    )
    emitted = page.evaluate("() => window.__physicalPlotlySelectionEvents")
    assert selected["panelVisible"] is True and len(selected["pointIds"]) > 1, {
        "bounds": bounds,
        "selection_box": {"left": left, "top": top, "right": right, "bottom": bottom},
        "hit_target": hit_target,
        "emitted": emitted,
        "snapshot": selected,
    }

    page.click("#classSplitBulkClear")
    page.wait_for_function(
        "() => !window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot().panelVisible"
    )
    point_position = page.evaluate(
        """() => {
            const graph = document.querySelector('#classSplitGraph');
            const rect = graph.getBoundingClientRect();
            const trace = graph.data.find(candidate => candidate.x?.length);
            return {
                x: rect.left + graph._fullLayout.xaxis._offset
                    + graph._fullLayout.xaxis.l2p(trace.x[0]),
                y: rect.top + graph._fullLayout.yaxis._offset
                    + graph._fullLayout.yaxis.l2p(trace.y[0]),
            };
        }"""
    )
    page.mouse.click(point_position["x"], point_position["y"])
    page.wait_for_function(
        """() => document.querySelector('#classSplitInspector')?.textContent.includes('Truck')
            && !document.querySelector('#classSplitSingleInspectorSection')?.hidden""",
        timeout=3000,
    )
    assert page.locator("#classSplitBulkPanel").is_hidden()


@pytest.mark.parametrize("projection_method", ["pca", "umap"])
def test_class_split_batch_review_is_stable_and_undoes_as_one_action(
    playwright_page,
    projection_method,
):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    result = _mock_class_split_many_wrong_result(6)
    result["summary"]["projection_method"] = projection_method
    page.evaluate(
        "value => window.__TATOR_TEST_HOOKS__.classSplitSetCapabilities(value)",
        {"review_disposition_api_version": 3},
    )
    page.evaluate(
        """async ({payload, jobId}) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            jobId
        )""",
        {"payload": result, "jobId": f"pw_batch_{projection_method}_job"},
    )
    all_point_ids = [point["point_id"] for point in result["points"]]
    point_ids = all_point_ids[:4]
    retained_ids = all_point_ids[4:]
    before = page.evaluate(
        "ids => window.__TATOR_TEST_HOOKS__.classSplitEmitSelection(ids)",
        point_ids,
    )
    page.evaluate(
        """async () => {
            await Plotly.relayout(document.querySelector('#classSplitGraph'), {
                'xaxis.range': [-0.75, 0.25],
                'yaxis.range': [-0.5, 0.5],
            });
        }"""
    )
    ranges_before = page.evaluate(
        """() => ({
            x: [...document.querySelector('#classSplitGraph').layout.xaxis.range],
            y: [...document.querySelector('#classSplitGraph').layout.yaxis.range],
        })"""
    )
    retained_coordinates_before = page.evaluate(
        """ids => Object.fromEntries(document.querySelector('#classSplitGraph').data
            .flatMap(trace => (trace.customdata || []).map((pointId, index) => [
                pointId,
                [trace.x[index], trace.y[index]],
            ]))
            .filter(([pointId]) => ids.includes(pointId)))""",
        retained_ids,
    )
    _install_deferred_review_fetch(page)
    try:
        page.click("#classSplitBulkConfirm")
        page.wait_for_function(
            "() => (window.__classSplitDeferredReviewFetch?.pending.length || 0) === 3"
        )
        pending = page.evaluate(
            "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot()"
        )
        assert pending["actionInFlight"] is True
        assert pending["pointIds"] == point_ids
        assert pending["panelRenderCount"] == before["panelRenderCount"]
        assert pending["graphCommitCount"] == before["graphCommitCount"]
        assert pending["runDisabled"] is True
        assert pending["rerunDisabled"] is True
        assert pending["graphMutationLocked"] is True
        assert pending["graphProjectionDisabled"] is True
        assert pending["bulkControlsDisabled"] is True
        assert pending["disabledTileCount"] == len(point_ids)

        for _ in point_ids:
            pending_ids = page.evaluate(
                "() => window.__classSplitDeferredReviewFetch.pending.map(entry => entry.pointId)"
            )
            _resolve_deferred_review(page, point_id=pending_ids[-1])
        page.wait_for_function(
            "() => !window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot().actionInFlight",
            timeout=5000,
        )
        completed = page.evaluate(
            "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot()"
        )
        assert completed["pointIds"] == []
        assert completed["reviewedCount"] == len(point_ids)
        assert completed["lastReviewActionKind"] == "batch"
        assert completed["lastReviewActionCount"] == len(point_ids)
        assert completed["graphCommitCount"] == before["graphCommitCount"] + 1
        assert sum(completed["tracePointCounts"]) == len(retained_ids)
        assert page.evaluate(
            "() => window.__classSplitDeferredReviewFetch.maxActive"
        ) == 3
        assert page.evaluate(
            """() => ({
                x: [...document.querySelector('#classSplitGraph').layout.xaxis.range],
                y: [...document.querySelector('#classSplitGraph').layout.yaxis.range],
            })"""
        ) == ranges_before
        assert page.evaluate(
            """ids => Object.fromEntries(document.querySelector('#classSplitGraph').data
                .flatMap(trace => (trace.customdata || []).map((pointId, index) => [
                    pointId,
                    [trace.x[index], trace.y[index]],
                ]))
                .filter(([pointId]) => ids.includes(pointId)))""",
            retained_ids,
        ) == retained_coordinates_before

        page.click("#classSplitWrongUndoReview")
        for _ in point_ids:
            pending_ids = page.evaluate(
                "() => window.__classSplitDeferredReviewFetch.pending.map(entry => entry.pointId)"
            )
            _resolve_deferred_review(page, point_id=pending_ids[-1])
        page.wait_for_function(
            "() => !window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot().actionInFlight",
            timeout=5000,
        )
        undone = page.evaluate(
            "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot()"
        )
        assert undone["reviewedCount"] == 0
        assert undone["lastReviewActionKind"] == ""
        assert sum(undone["tracePointCounts"]) == len(all_point_ids)
        assert page.evaluate(
            """() => ({
                x: [...document.querySelector('#classSplitGraph').layout.xaxis.range],
                y: [...document.querySelector('#classSplitGraph').layout.yaxis.range],
            })"""
        ) == ranges_before
        clear_requests = page.evaluate(
            """() => window.__classSplitDeferredReviewFetch.requests
                .filter(request => request.payload.disposition === 'clear')
                .map(request => request.payload)"""
        )
        assert len(clear_requests) == len(point_ids)
        assert all(request.get("expected_revision", "").startswith("rdr1_") for request in clear_requests)
    finally:
        _restore_deferred_review_fetch(page)


def test_class_split_batch_lock_blocks_single_review_and_history_controls(
    playwright_page,
):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    result = _mock_class_split_many_wrong_result(6)
    reviewed_point = result["points"][-1]
    reviewed_point.update({
        "human_review_disposition": "skip",
        "human_reviewed_at": 1785800000,
        "human_review_revision": f"rdr1_{'f' * 32}",
        "human_review_origin": "desktop",
        "human_review_persistence": "durable",
    })
    page.evaluate(
        "value => window.__TATOR_TEST_HOOKS__.classSplitSetCapabilities(value)",
        {"review_disposition_api_version": 3},
    )
    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_batch_cross_action_lock_job'
        )""",
        result,
    )
    selected_ids = [point["point_id"] for point in result["points"][:4]]
    competing_point_id = result["points"][4]["point_id"]
    page.evaluate(
        "ids => window.__TATOR_TEST_HOOKS__.classSplitEmitSelection(ids)",
        selected_ids,
    )
    assert page.locator("#classSplitReviewHistoryDelete").is_enabled()
    _install_deferred_review_fetch(page)
    try:
        page.click("#classSplitBulkConfirm")
        page.wait_for_function(
            "() => (window.__classSplitDeferredReviewFetch?.pending.length || 0) === 3"
        )
        snapshot = page.evaluate(
            "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot()"
        )
        assert snapshot["reviewHistoryDeleteDisabled"] is True
        assert snapshot["bulkControlsDisabled"] is True
        assert snapshot["graphMutationLocked"] is True

        selector = (
            "#classSplitWrongList "
            f"[data-action='correct-class'][data-point-id='{competing_point_id}']"
        )
        page.wait_for_selector(selector)
        page.evaluate(
            """selector => {
                const button = document.querySelector(selector);
                button.disabled = false;
                button.click();
            }""",
            selector,
        )
        page.wait_for_timeout(350)
        assert page.evaluate(
            "() => window.__classSplitDeferredReviewFetch.requests.length"
        ) == 3

        for _ in selected_ids:
            _resolve_deferred_review(page)
        page.wait_for_function(
            "() => !window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot().actionInFlight",
            timeout=5000,
        )
        assert page.evaluate(
            "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot().reviewedCount"
        ) == 5
    finally:
        _restore_deferred_review_fetch(page)


def test_class_split_stale_poll_404_cannot_clear_replacement_job(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    result = _mock_class_split_many_wrong_result(4)
    page.evaluate(
        """() => {
            const originalFetch = window.fetch.bind(window);
            let resolvePoll;
            window.__classSplitStalePoll = {originalFetch};
            window.fetch = (input, options = {}) => {
                const url = String(input?.url || input || '');
                if (!url.includes('/class_analysis/jobs/pw_stale_poll_job')) {
                    return originalFetch(input, options);
                }
                return new Promise((resolve) => {
                    resolvePoll = resolve;
                    window.__classSplitStalePoll.pending = true;
                });
            };
            window.__classSplitStalePoll.resolve = () => resolvePoll(
                new Response(JSON.stringify({detail: 'gone'}), {
                    status: 404,
                    headers: {'Content-Type': 'application/json'},
                })
            );
            window.__classSplitStalePoll.promise =
                window.__TATOR_TEST_HOOKS__.classSplitPollJob('pw_stale_poll_job');
        }"""
    )
    page.wait_for_function("() => window.__classSplitStalePoll?.pending === true")
    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_replacement_after_stale_poll_job'
        )""",
        result,
    )
    page.evaluate("() => window.__classSplitStalePoll.resolve()")
    page.evaluate("() => window.__classSplitStalePoll.promise")
    snapshot = page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot()"
    )
    assert snapshot["currentJobId"] == "pw_replacement_after_stale_poll_job"
    assert sum(snapshot["tracePointCounts"]) == 4
    page.evaluate(
        """() => {
            window.fetch = window.__classSplitStalePoll.originalFetch;
            delete window.__classSplitStalePoll;
        }"""
    )


def test_class_split_batch_undo_partial_failure_is_retryable(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    result = _mock_class_split_many_wrong_result(5)
    point_ids = [point["point_id"] for point in result["points"]]
    page.evaluate(
        "value => window.__TATOR_TEST_HOOKS__.classSplitSetCapabilities(value)",
        {"review_disposition_api_version": 3},
    )
    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_batch_partial_undo_job'
        )""",
        result,
    )
    page.evaluate(
        "ids => window.__TATOR_TEST_HOOKS__.classSplitEmitSelection(ids)",
        point_ids,
    )
    _install_deferred_review_fetch(page)
    try:
        page.click("#classSplitBulkConfirm")
        for _ in point_ids:
            _resolve_deferred_review(page)
        page.wait_for_function(
            "() => !window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot().actionInFlight",
            timeout=5000,
        )

        page.click("#classSplitWrongUndoReview")
        conflict_id = _resolve_deferred_review(
            page,
            status=409,
            code="review_disposition_changed",
        )
        _resolve_deferred_review(page)
        uncertain_id = _resolve_deferred_review(
            page,
            status=503,
            code="response_lost_after_commit",
        )
        failed_ids = [conflict_id, uncertain_id]
        _resolve_deferred_review(page)
        _resolve_deferred_review(page)
        page.wait_for_function(
            "() => !window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot().actionInFlight",
            timeout=5000,
        )
        partial = page.evaluate(
            "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot()"
        )
        assert partial["reviewedCount"] == 2
        assert partial["lastReviewActionKind"] == "batch"
        assert partial["lastReviewActionCount"] == 2
        assert sum(partial["tracePointCounts"]) == 3
        assert "Restored 3 of 5; 2 could not be restored" in page.locator(
            "#classSplitJobStatus"
        ).inner_text()
        first_clear_tokens = page.evaluate(
            """() => Object.fromEntries(window.__classSplitDeferredReviewFetch.requests
                .filter(request => request.payload.disposition === 'clear')
                .map(request => [request.pointId, request.payload.client_action_id]))"""
        )
        assert all(
            request.get("expected_revision", "").startswith("rdr1_")
            for request in page.evaluate(
                """() => window.__classSplitDeferredReviewFetch.requests
                    .filter(request => request.payload.disposition === 'clear')
                    .map(request => request.payload)"""
            )
        )

        page.click("#classSplitWrongUndoReview")
        for point_id in failed_ids:
            _resolve_deferred_review(page, point_id=point_id)
        page.wait_for_function(
            "() => !window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot().actionInFlight",
            timeout=5000,
        )
        retried = page.evaluate(
            "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot()"
        )
        assert retried["reviewedCount"] == 0
        assert retried["lastReviewActionKind"] == ""
        assert sum(retried["tracePointCounts"]) == len(point_ids)
        retry_requests = page.evaluate(
            """() => window.__classSplitDeferredReviewFetch.requests
                .filter(request => request.payload.disposition === 'clear')
                .slice(-2)
                .map(request => ({
                    pointId: request.pointId,
                    token: request.payload.client_action_id,
                }))"""
        )
        assert all(
            first_clear_tokens[request["pointId"]] == request["token"]
            for request in retry_requests
        )
    finally:
        _restore_deferred_review_fetch(page)


def test_class_split_batch_review_retains_failures_and_reuses_retry_tokens(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    result = _mock_class_split_many_wrong_result(5)
    job_id = "pw_batch_partial_failure_job"
    page.evaluate(
        "value => window.__TATOR_TEST_HOOKS__.classSplitSetCapabilities(value)",
        {"review_disposition_api_version": 3},
    )
    page.evaluate(
        """async ({payload, jobId}) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            jobId
        )""",
        {"payload": result, "jobId": job_id},
    )
    point_ids = [point["point_id"] for point in result["points"]]
    page.evaluate(
        "ids => window.__TATOR_TEST_HOOKS__.classSplitEmitSelection(ids)",
        point_ids,
    )
    _install_deferred_review_fetch(page)
    try:
        page.click("#classSplitBulkSkip")
        resolved = [
            _resolve_deferred_review(page),
            _resolve_deferred_review(page, status=409, code="review_disposition_changed"),
            _resolve_deferred_review(page),
            _resolve_deferred_review(page, status=503, code="response_lost_after_commit"),
            _resolve_deferred_review(page),
        ]
        page.wait_for_function(
            "() => !window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot().actionInFlight",
            timeout=5000,
        )
        failed_ids = [resolved[1], resolved[3]]
        partial = page.evaluate(
            "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot()"
        )
        assert partial["pointIds"] == failed_ids
        assert partial["reviewedCount"] == 3
        assert partial["lastReviewActionCount"] == 3
        assert "being reconciled safely" in page.locator("#classSplitJobStatus").inner_text()
        first_tokens = page.evaluate(
            """() => Object.fromEntries(window.__classSplitDeferredReviewFetch.requests
                .filter(request => request.payload.disposition === 'skip')
                .map(request => [request.pointId, request.payload.client_action_id]))"""
        )

        page.click("#classSplitBulkSkip")
        for _ in failed_ids:
            _resolve_deferred_review(page)
        page.wait_for_function(
            "() => !window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot().actionInFlight",
            timeout=5000,
        )
        retried = page.evaluate(
            "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot()"
        )
        assert retried["pointIds"] == []
        assert retried["reviewedCount"] == 5
        retry_requests = page.evaluate(
            """() => window.__classSplitDeferredReviewFetch.requests
                .filter(request => request.payload.disposition === 'skip')
                .slice(-2)
                .map(request => ({
                    pointId: request.pointId,
                    token: request.payload.client_action_id,
                }))"""
        )
        assert all(first_tokens[request["pointId"]] == request["token"] for request in retry_requests)
    finally:
        _restore_deferred_review_fetch(page)


def test_class_split_stale_batch_cannot_mutate_a_new_job(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    old_result = _mock_class_split_many_wrong_result(4)
    old_ids = [point["point_id"] for point in old_result["points"]]
    page.evaluate(
        "value => window.__TATOR_TEST_HOOKS__.classSplitSetCapabilities(value)",
        {"review_disposition_api_version": 3},
    )
    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_stale_batch_old_job'
        )""",
        old_result,
    )
    page.evaluate(
        "ids => window.__TATOR_TEST_HOOKS__.classSplitEmitSelection(ids)",
        old_ids,
    )
    _install_deferred_review_fetch(page)
    try:
        page.click("#classSplitBulkConfirm")
        page.wait_for_function(
            "() => window.__classSplitDeferredReviewFetch.pending.length === 3"
        )

        new_result = _mock_class_split_many_wrong_result(3)
        for collection_name in ("points", "wrong_class_candidates"):
            for point in new_result[collection_name]:
                point["point_id"] = f"new-{point['point_id']}"
        new_ids = [point["point_id"] for point in new_result["points"]]
        page.evaluate(
            """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
                payload,
                'pw_stale_batch_new_job'
            )""",
            new_result,
        )
        new_snapshot = page.evaluate(
            "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot()"
        )
        new_status = page.locator("#classSplitJobStatus").inner_text()
        new_graph_ids = page.evaluate(
            """() => document.querySelector('#classSplitGraph').data
                .flatMap(trace => trace.customdata || [])"""
        )
        assert sorted(new_graph_ids) == sorted(new_ids)

        pending_ids = page.evaluate(
            "() => window.__classSplitDeferredReviewFetch.pending.map(entry => entry.pointId)"
        )
        for index, point_id in enumerate(reversed(pending_ids)):
            _resolve_deferred_review(
                page,
                point_id=point_id,
                status=503 if index == 0 else 200,
                code="response_lost_after_commit" if index == 0 else "",
            )
        page.wait_for_function(
            "() => !window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot().actionInFlight",
            timeout=5000,
        )
        settled = page.evaluate(
            "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot()"
        )
        settled_graph_ids = page.evaluate(
            """() => document.querySelector('#classSplitGraph').data
                .flatMap(trace => trace.customdata || [])"""
        )
        assert sorted(settled_graph_ids) == sorted(new_ids)
        assert settled["pointIds"] == new_snapshot["pointIds"]
        assert settled["reviewedCount"] == 0
        assert settled["reconciliationCount"] == 0
        assert settled["graphCommitCount"] == new_snapshot["graphCommitCount"]
        assert page.locator("#classSplitJobStatus").inner_text() == new_status
        assert page.evaluate(
            """() => window.__classSplitDeferredReviewFetch.requests.every(
                request => request.jobId === 'pw_stale_batch_old_job'
            )"""
        ) is True
    finally:
        _restore_deferred_review_fetch(page)


def test_class_split_stale_plotly_completion_cannot_touch_replacement_job(
    playwright_page,
):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    old_result = _mock_class_split_many_wrong_result(4)
    old_ids = [point["point_id"] for point in old_result["points"]]
    page.evaluate(
        "value => window.__TATOR_TEST_HOOKS__.classSplitSetCapabilities(value)",
        {"review_disposition_api_version": 3},
    )
    page.evaluate(
        """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            'pw_stale_plotly_old_job'
        )""",
        old_result,
    )
    page.evaluate(
        "ids => window.__TATOR_TEST_HOOKS__.classSplitEmitSelection(ids)",
        old_ids,
    )
    _install_deferred_review_fetch(page)
    page.evaluate(
        """() => {
            const original = Plotly.restyle.bind(Plotly);
            const state = { original, held: false, release: null };
            window.__classSplitHeldPointRemovalRestyle = state;
            Plotly.restyle = (graph, update, traceIndexes) => {
                const result = original(graph, update, traceIndexes);
                if (
                    !state.held
                    && Object.prototype.hasOwnProperty.call(update || {}, 'customdata')
                ) {
                    state.held = true;
                    let releaseGate;
                    const gate = new Promise(resolve => { releaseGate = resolve; });
                    return Promise.resolve(result).then(value => {
                        state.release = () => releaseGate(value);
                        return gate;
                    });
                }
                return result;
            };
        }"""
    )
    try:
        page.click("#classSplitBulkConfirm")
        for _ in old_ids:
            _resolve_deferred_review(page)
        page.wait_for_function(
            "() => typeof window.__classSplitHeldPointRemovalRestyle?.release === 'function'",
            timeout=5000,
        )

        new_result = _mock_class_split_many_wrong_result(3)
        for collection_name in ("points", "wrong_class_candidates"):
            for point in new_result[collection_name]:
                point["point_id"] = f"replacement-{point['point_id']}"
        new_ids = [point["point_id"] for point in new_result["points"]]
        page.evaluate(
            """async (payload) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
                payload,
                'pw_stale_plotly_new_job'
            )""",
            new_result,
        )
        page.evaluate(
            """async () => Plotly.relayout(document.querySelector('#classSplitGraph'), {
                'xaxis.range': [10, 12],
                'yaxis.range': [20, 22],
            })"""
        )
        replacement_snapshot = page.evaluate(
            "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot()"
        )
        replacement_status = page.locator("#classSplitJobStatus").inner_text()

        page.evaluate(
            "() => window.__classSplitHeldPointRemovalRestyle.release()"
        )
        page.wait_for_function(
            "() => !window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot().actionInFlight",
            timeout=5000,
        )
        assert sorted(page.evaluate(
            """() => document.querySelector('#classSplitGraph').data
                .flatMap(trace => trace.customdata || [])"""
        )) == sorted(new_ids)
        assert page.evaluate(
            """() => ({
                x: [...document.querySelector('#classSplitGraph').layout.xaxis.range],
                y: [...document.querySelector('#classSplitGraph').layout.yaxis.range],
            })"""
        ) == {"x": [10, 12], "y": [20, 22]}
        settled = page.evaluate(
            "() => window.__TATOR_TEST_HOOKS__.classSplitMultiSelectionSnapshot()"
        )
        assert settled["graphCommitCount"] == replacement_snapshot["graphCommitCount"]
        assert page.locator("#classSplitJobStatus").inner_text() == replacement_status
    finally:
        page.evaluate(
            """() => {
                const state = window.__classSplitHeldPointRemovalRestyle;
                if (state?.original) Plotly.restyle = state.original;
                delete window.__classSplitHeldPointRemovalRestyle;
            }"""
        )
        _restore_deferred_review_fetch(page)


@pytest.mark.parametrize("projection_method", ["pca", "umap"])
def test_class_split_bulk_relabel_keeps_protected_and_failed_bbox_selected(
    playwright_page,
    projection_method,
):
    page, _ = playwright_page
    go_to_tab(page, "#tabClassSplitButton", "#tabClassSplit")
    page.locator("#classes").set_input_files(
        {
            "name": "classes.txt",
            "mimeType": "text/plain",
            "buffer": b"Truck\nPerson\n",
        }
    )
    page.wait_for_function(
        "() => document.querySelector('#classList')?.options.length === 2"
    )
    page.evaluate(
        """fixture => window.__TATOR_TEST_HOOKS__.classSplitSeedActiveWorkspaceAnnotations(fixture)""",
        {
            "imageKey": "single.png",
            "width": 960,
            "height": 960,
            "classNames": ["Truck", "Person"],
            "boxes": {
                "Truck": [
                    {"x": 100.25, "y": 120.5, "width": 80.5, "height": 100.25},
                    {"x": 300.0, "y": 320.0, "width": 50.0, "height": 60.0},
                ],
                "Person": [
                    {"x": 102.0, "y": 122.0, "width": 80.0, "height": 100.0},
                ],
            },
        },
    )
    result = _mock_active_workspace_single_bbox_result()
    base = dict(result["points"][1])
    protected = {
        **base,
        "point_id": "truck-protected",
        "class_name": "Person",
        "bbox_xyxy": [102.0, 122.0, 182.0, 222.0],
        "projection": [-0.25, -0.25],
    }
    success = {
        **base,
        "point_id": "truck-success",
        "bbox_xyxy": [300.0, 320.0, 350.0, 380.0],
        "projection": [0.25, 0.25],
    }
    stale = {
        **base,
        "point_id": "truck-stale",
        "bbox_xyxy": [700.0, 700.0, 750.0, 760.0],
        "projection": [0.75, 0.75],
        "_resolved_frontend_bbox_uuid": "synthetic-missing-bbox-identity",
    }
    conflict = {
        "enabled": True,
        "kind": "near_identical_cross_class_bbox",
        "review_mode": "dual_bbox_annotation_resolution",
        "point_id": "truck-2",
        "other_point_id": "truck-protected",
        "current_class": "Truck",
        "other_class_name": "Person",
        "split": "train",
        "image_relpath": "single.png",
        "target_bbox_xyxy": base["bbox_xyxy"],
        "other_bbox_xyxy": protected["bbox_xyxy"],
        "iou": 0.9,
        "target_area_covered": 0.92,
    }
    base["projection"] = [-0.75, -0.75]
    base["dual_bbox_conflict"] = conflict
    result["points"] = [base, protected, success, stale]
    result["summary"].update(
        {
            "object_count": 4,
            "class_counts": {"Truck": 3, "Person": 1},
            "wrong_class_candidate_count": 4,
            "projection_method": projection_method,
        }
    )
    coordinates = [[-0.75, -0.75], [-0.25, -0.25], [0.25, 0.25], [0.75, 0.75]]
    result["projection_options"]["coordinates"] = {
        key: list(coordinates)
        for key in result["projection_options"]["available"]
    }
    result["wrong_class_candidates"] = [dict(point) for point in result["points"]]
    page.evaluate(
        """async ({payload, jobId}) => window.__TATOR_TEST_HOOKS__.classSplitApplyResult(
            payload,
            jobId
        )""",
        {"payload": result, "jobId": f"pw_bulk_relabel_{projection_method}_job"},
    )
    before = page.evaluate(
        """ids => window.__TATOR_TEST_HOOKS__.classSplitEmitSelection(ids)""",
        ["truck-2", "truck-protected", "truck-success", "truck-stale"],
    )
    page.evaluate(
        """async () => Plotly.relayout(document.querySelector('#classSplitGraph'), {
            'xaxis.range': [-1.0, 1.0],
            'yaxis.range': [-0.8, 0.8],
        })"""
    )
    ranges_before = page.evaluate(
        """() => ({
            x: [...document.querySelector('#classSplitGraph').layout.xaxis.range],
            y: [...document.querySelector('#classSplitGraph').layout.yaxis.range],
        })"""
    )
    changed = page.evaluate(
        """async () => window.__TATOR_TEST_HOOKS__.classSplitChangeSelectedClass('Person')"""
    )

    assert changed["pointIds"] == ["truck-2", "truck-protected", "truck-stale"]
    assert changed["graphCommitCount"] == before["graphCommitCount"] + 1
    assert page.evaluate(
        "() => window.__TATOR_TEST_HOOKS__.classSplitActiveWorkspaceClassCounts('single.png')"
    ) == {"Truck": 1, "Person": 2}
    assert "2 overlapping-box objects were left unchanged" in page.locator(
        "#samStatus"
    ).inner_text()
    assert "3 unresolved objects remain selected" in page.locator("#samStatus").inner_text()
    assert page.evaluate(
        """() => ({
            x: [...document.querySelector('#classSplitGraph').layout.xaxis.range],
            y: [...document.querySelector('#classSplitGraph').layout.yaxis.range],
        })"""
    ) == ranges_before
