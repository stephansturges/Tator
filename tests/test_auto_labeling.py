import pytest
import numpy as np

from services.auto_labeling import (
    AUTO_LABEL_TARGET_MODE_DETECTION,
    AUTO_LABEL_TARGET_MODE_SEGMENTATION,
    adjust_falcon_candidate_score,
    build_falcon_query_tiers,
    build_quadrant_windows,
    derive_mask_component_candidates,
    infer_dataset_annotation_mode,
    parse_yolo_label_line,
    score_falcon_candidate,
    serialize_bbox_polygon_label_line,
)


pytestmark = [pytest.mark.auto_label_smoke]


def _flat_falcon_query_rows(
    class_names,
    *,
    labelmap,
    glossary="",
    prompt_style="default",
    query_frame="term",
):
    tiers = build_falcon_query_tiers(
        class_names,
        labelmap=labelmap,
        glossary=glossary,
        prompt_style=prompt_style,
        query_frame=query_frame,
    )
    rows = []
    for class_name in class_names:
        for row in tiers.get(class_name, []):
            rows.append(
                {
                    "class_name": row["class_name"],
                    "query": row["query"],
                    "term": row["term"],
                }
            )
    return rows


def test_infer_dataset_annotation_mode_prefers_segmentation_when_polygons_exist() -> None:
    rows = [
        {"label_lines": ["0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3"]},
        {"label_lines": ["1 0.5 0.5 0.2 0.2"]},
    ]
    inferred = infer_dataset_annotation_mode(rows)
    assert inferred["mode"] == AUTO_LABEL_TARGET_MODE_SEGMENTATION
    assert inferred["polygon_count"] == 1
    assert inferred["bbox_count"] == 1


def test_infer_dataset_annotation_mode_returns_detection_for_box_only_rows() -> None:
    rows = [
        {"label_lines": ["0 0.5 0.5 0.2 0.2", "1 0.4 0.4 0.1 0.1"]},
    ]
    inferred = infer_dataset_annotation_mode(rows)
    assert inferred["mode"] == AUTO_LABEL_TARGET_MODE_DETECTION
    assert inferred["bbox_count"] == 2
    assert inferred["polygon_count"] == 0


def test_build_quadrant_windows_adds_overlap_and_bounds() -> None:
    windows = build_quadrant_windows(1000, 800, overlap_ratio=0.1)
    assert [window["id"] for window in windows] == ["Q1", "Q2", "Q3", "Q4"]
    assert windows[0]["xyxy"] == (0.0, 0.0, 550.0, 440.0)
    assert windows[3]["xyxy"] == (450.0, 360.0, 1000.0, 800.0)


def test_parse_yolo_label_line_parses_bbox_and_polygon() -> None:
    bbox = parse_yolo_label_line("2 0.5 0.5 0.2 0.4", width=200, height=100)
    assert bbox is not None
    assert bbox["class_id"] == 2
    assert bbox["kind"] == "bbox"
    assert bbox["bbox_xyxy"] == (80.0, 30.0, 120.0, 70.0)

    polygon = parse_yolo_label_line(
        "1 0.1 0.1 0.4 0.1 0.4 0.6 0.1 0.6",
        width=200,
        height=100,
    )
    assert polygon is not None
    assert polygon["class_id"] == 1
    assert polygon["kind"] == "polygon"
    assert polygon["bbox_xyxy"] == (20.0, 10.0, 80.0, 60.0)


def test_serialize_bbox_polygon_label_line_writes_polygon_coordinates() -> None:
    line = serialize_bbox_polygon_label_line(3, (20, 10, 80, 60), width=200, height=100)
    assert line == "3 0.100000 0.100000 0.400000 0.100000 0.400000 0.600000 0.100000 0.600000"


def test_build_falcon_query_tiers_uses_naturalized_labels_without_static_synonyms() -> None:
    rows = _flat_falcon_query_rows(
        ["light_vehicle", "utility_pole", "person"],
        labelmap=["light_vehicle", "utility_pole", "person"],
        glossary="",
    )
    assert rows == [
        {"class_name": "light_vehicle", "query": "light vehicle", "term": "light vehicle"},
        {"class_name": "utility_pole", "query": "utility pole", "term": "utility pole"},
        {"class_name": "person", "query": "person", "term": "person"},
    ]
    assert all("," not in row["query"] for row in rows)


def test_build_falcon_query_tiers_uses_canonical_label_when_no_glossary_exists() -> None:
    rows = _flat_falcon_query_rows(
        ["building", "truck"],
        labelmap=["building", "truck"],
        glossary="",
    )
    assert rows == [
        {"class_name": "building", "query": "building", "term": "building"},
        {"class_name": "truck", "query": "truck", "term": "truck"},
    ]


def test_build_falcon_query_tiers_uses_glossary_alternatives_as_separate_queries() -> None:
    rows = _flat_falcon_query_rows(
        ["light_vehicle"],
        labelmap=["light_vehicle"],
        glossary='{"light_vehicle":["car","van","sedan","pickup truck","automobile"]}',
    )
    assert {"class_name": "light_vehicle", "query": "car", "term": "car"} in rows
    assert {"class_name": "light_vehicle", "query": "van", "term": "van"} in rows
    assert {"class_name": "light_vehicle", "query": "sedan", "term": "sedan"} in rows
    assert {"class_name": "light_vehicle", "query": "pickup truck", "term": "pickup truck"} in rows
    assert {"class_name": "light_vehicle", "query": "automobile", "term": "automobile"} in rows


def test_build_falcon_query_tiers_glossary_only_prioritizes_first_glossary_term() -> None:
    tiers = build_falcon_query_tiers(
        ["light_vehicle"],
        labelmap=["light_vehicle"],
        glossary='{"light_vehicle":["car","sedan","automobile","SUV"]}',
        prompt_style="glossary_only",
    )
    rows = tiers["light_vehicle"]
    assert rows[0] == {
        "class_name": "light_vehicle",
        "query": "car",
        "term": "car",
        "tier": "A",
    }
    assert any(row["query"] == "SUV" and row["tier"] == "B" for row in rows)
    assert any(row["query"] == "automobile" and row["tier"] == "B" for row in rows)


def test_build_falcon_query_tiers_keeps_canonical_noun_in_tier_a_when_usable() -> None:
    tiers = build_falcon_query_tiers(
        ["boat", "building", "utility_pole"],
        labelmap=["boat", "building", "utility_pole"],
        glossary='{"boat":["canoe","ship"],"building":["house","warehouse"],"utility_pole":["streetlight","mast"]}',
    )
    assert tiers["boat"][0]["query"] == "boat"
    assert tiers["boat"][0]["tier"] == "A"
    assert any(row["query"] == "canoe" and row["tier"] == "C" for row in tiers["boat"])
    assert tiers["building"][0]["query"] == "building"
    assert tiers["utility_pole"][0]["query"] == "utility pole"


def test_build_falcon_query_tiers_can_force_canonical_only_style() -> None:
    tiers = build_falcon_query_tiers(
        ["light_vehicle", "utility_pole"],
        labelmap=["light_vehicle", "utility_pole"],
        glossary='{"light_vehicle":["car","SUV"],"utility_pole":["streetlight","mast"]}',
        prompt_style="canonical_only",
    )
    assert tiers["light_vehicle"] == [
        {
            "class_name": "light_vehicle",
            "query": "light vehicle",
            "term": "light vehicle",
            "tier": "A",
        }
    ]
    assert tiers["utility_pole"] == [
        {
            "class_name": "utility_pole",
            "query": "utility pole",
            "term": "utility pole",
            "tier": "A",
        }
    ]


def test_build_falcon_query_tiers_can_use_glossary_only_with_all_instances_frame() -> None:
    rows = _flat_falcon_query_rows(
        ["light_vehicle"],
        labelmap=["light_vehicle"],
        glossary='{"light_vehicle":["car","SUV","sedan"]}',
        prompt_style="glossary_only",
        query_frame="all_instances",
    )
    assert rows[:3] == [
        {
            "class_name": "light_vehicle",
            "query": "all instances of car",
            "term": "car",
        },
        {
            "class_name": "light_vehicle",
            "query": "all instances of SUV",
            "term": "SUV",
        },
        {
            "class_name": "light_vehicle",
            "query": "all instances of sedan",
            "term": "sedan",
        },
    ]


def test_build_falcon_query_tiers_can_use_priority_top1_style() -> None:
    rows = _flat_falcon_query_rows(
        ["truck", "building"],
        labelmap=["truck", "building"],
        glossary="",
        prompt_style="priority_top1",
    )
    assert rows == [
        {"class_name": "truck", "query": "truck", "term": "truck"},
        {"class_name": "building", "query": "building", "term": "building"},
    ]


def test_derive_mask_component_candidates_rejects_degenerate_and_keeps_local_components() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[:, :] = True
    mask[10:20, 10:20] = True
    mask[60:72, 58:70] = True
    summary = derive_mask_component_candidates(
        mask,
        crop_width=100,
        crop_height=100,
        mode="component_split",
    )
    assert summary["candidates"] == []
    assert any(item["drop_reason"] == "bbox_too_large" for item in summary["dropped"])

    local_mask = np.zeros((100, 100), dtype=bool)
    local_mask[10:20, 10:20] = True
    local_mask[60:72, 58:70] = True
    local_summary = derive_mask_component_candidates(
        local_mask,
        crop_width=100,
        crop_height=100,
        mode="component_cluster",
    )
    assert local_summary["candidates"]
    assert all(item["bbox_area_fraction"] < 0.85 for item in local_summary["candidates"])


def test_derive_mask_component_candidates_adds_edge_strip_clusters_for_fragmented_border_objects() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[26:34, 0:12] = True
    mask[42:78, 0:14] = True
    mask[78:86, 2:14] = True

    summary = derive_mask_component_candidates(
        mask,
        crop_width=100,
        crop_height=100,
        mode="component_split",
        max_components=32,
    )

    edge_clusters = [
        item for item in summary["candidates"] if str(item.get("derivation_mode") or "").startswith("edge_strip_cluster_left")
    ]
    assert edge_clusters
    best = max(edge_clusters, key=lambda item: float(item.get("bbox_area_fraction") or 0.0))
    assert best["component_count"] >= 2
    assert best["bbox_xyxy"] == (0.0, 26.0, 14.0, 86.0)


def test_score_falcon_candidate_prefers_object_scale_boxes_over_tiny_fragments() -> None:
    object_scale_score = score_falcon_candidate(
        bbox_area_fraction=0.03,
        border_touch_count=0,
        component_count=1,
        derivation_mode="component_split",
    )
    tiny_fragment_score = score_falcon_candidate(
        bbox_area_fraction=0.0002,
        border_touch_count=0,
        component_count=1,
        derivation_mode="component_split",
    )
    coarse_score = score_falcon_candidate(
        bbox_area_fraction=0.28,
        border_touch_count=1,
        component_count=1,
        derivation_mode="component_split",
    )
    edge_cluster_score = score_falcon_candidate(
        bbox_area_fraction=0.02,
        border_touch_count=1,
        component_count=2,
        derivation_mode="edge_strip_cluster_left",
    )
    assert object_scale_score > tiny_fragment_score
    assert object_scale_score > coarse_score
    assert edge_cluster_score > tiny_fragment_score


def test_adjust_falcon_candidate_score_keeps_geometry_score_dataset_agnostic() -> None:
    base_score = score_falcon_candidate(
        bbox_area_fraction=0.0032,
        border_touch_count=0,
        component_count=1,
        derivation_mode="component_split",
    )
    adjusted_a = adjust_falcon_candidate_score(
        class_name="class_a",
        query="first synonym",
        tier="A",
        base_score=base_score,
        bbox_xyxy=(58.0, 264.0, 126.0, 277.0),
        bbox_area_fraction=0.0032,
    )
    adjusted_b = adjust_falcon_candidate_score(
        class_name="class_b",
        query="other synonym",
        tier="C",
        base_score=base_score,
        bbox_xyxy=(193.0, 265.0, 224.0, 292.0),
        bbox_area_fraction=0.0030,
    )

    assert adjusted_a == pytest.approx(base_score)
    assert adjusted_b == pytest.approx(base_score)
