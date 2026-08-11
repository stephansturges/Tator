import math

import pytest

from services.class_analysis_context_features import (
    SAME_IMAGE_CONTEXT_FEATURE_CONTRACT,
    extract_same_image_context_features,
)


def _record(
    point_id,
    class_name,
    bbox,
    *,
    image="image.jpg",
    image_width=1000,
    image_height=1000,
    **extra,
):
    record = {
        "point_id": point_id,
        "split": "train",
        "image_relpath": image,
        "class_name": class_name,
        "bbox_xyxy": list(bbox),
        **extra,
    }
    if image_width is not None:
        record["_image_width"] = image_width
    if image_height is not None:
        record["_image_height"] = image_height
    return record


def _centered_box(cx, cy, width, height):
    return [
        (cx - width / 2.0) * 1000.0,
        (cy - height / 2.0) * 1000.0,
        (cx + width / 2.0) * 1000.0,
        (cy + height / 2.0) * 1000.0,
    ]


def _by_id(rows):
    return {row["point_id"]: row for row in rows}


def test_same_image_context_excludes_every_rough_candidate_from_anchor_statistics():
    records = [
        _record("target", "Bike", _centered_box(0.50, 0.50, 0.20, 0.20)),
        _record("anchor-1", "Bike", _centered_box(0.20, 0.20, 0.05, 0.05)),
        _record("anchor-2", "Bike", _centered_box(0.35, 0.35, 0.05, 0.05)),
        _record("anchor-3", "Bike", _centered_box(0.65, 0.65, 0.05, 0.05)),
        _record("anchor-4", "Bike", _centered_box(0.80, 0.80, 0.05, 0.05)),
        # This second rough candidate resembles the target. It must not pull
        # the target's expected scale toward the suspicious geometry.
        _record("rough-peer", "Bike", _centered_box(0.55, 0.55, 0.22, 0.22)),
        _record("person", "Person", _centered_box(0.52, 0.52, 0.06, 0.14)),
    ]

    rows = extract_same_image_context_features(
        records,
        excluded_anchor_point_ids={"target", "rough-peer"},
        target_point_ids={"target"},
    )

    assert len(rows) == 1
    target = rows[0]
    assert target["contract"] == SAME_IMAGE_CONTEXT_FEATURE_CONTRACT
    assert target["same_class_count"] == 6
    assert target["same_class_peer_count"] == 5
    assert target["trusted_same_class_anchor_count"] == 4
    assert target["excluded_same_class_peer_count"] == 1
    assert target["same_class_peer_width_median_norm"] == pytest.approx(0.05)
    assert target["same_class_log_width_residual"] == pytest.approx(math.log(4.0))
    assert target["same_class_area_robust_z"] == 20.0
    assert target["perspective_available"] is True
    assert target["perspective_peer_count"] == 4
    assert target["perspective_log_scale_residual"] > 1.0


def test_same_image_perspective_regression_is_leave_one_target_out_and_location_aware():
    records = []
    for index, y in enumerate((0.15, 0.30, 0.45, 0.60, 0.75)):
        # Exact log-linear perspective trend for both dimensions.
        size = math.exp(math.log(0.03) + 0.9 * y)
        records.append(
            _record(
                f"anchor-{index}",
                "Car",
                _centered_box(0.5, y, size, size * 0.6),
            )
        )
    target_y = 0.52
    expected_size = math.exp(math.log(0.03) + 0.9 * target_y)
    records.extend(
        [
            _record(
                "on-trend",
                "Car",
                _centered_box(0.5, target_y, expected_size, expected_size * 0.6),
            ),
            _record(
                "large-outlier",
                "Car",
                _centered_box(0.5, target_y, expected_size * 3.0, expected_size * 1.8),
                is_wrong_class_candidate=True,
            ),
        ]
    )

    rows = _by_id(
        extract_same_image_context_features(
            records,
            excluded_anchor_point_ids={"on-trend", "large-outlier"},
            target_point_ids={"on-trend", "large-outlier"},
            perspective_ridge=1e-8,
        )
    )

    assert rows["on-trend"]["perspective_available"] is True
    assert abs(rows["on-trend"]["perspective_log_scale_residual"]) < 1e-5
    assert rows["large-outlier"]["perspective_log_scale_residual"] == pytest.approx(
        math.log(3.0), abs=1e-5
    )
    assert rows["large-outlier"]["perspective_residual_magnitude"] > 1.5


def test_same_image_context_emits_border_density_and_overlap_features():
    records = [
        _record("target", "Pole", [0, 400, 100, 700]),
        _record("same-near", "Pole", [80, 420, 180, 720]),
        _record("building", "Building", [20, 390, 220, 750]),
        _record("far", "Person", [800, 50, 850, 150]),
    ]

    target = extract_same_image_context_features(
        records,
        excluded_anchor_point_ids={"target"},
        target_point_ids={"target"},
    )[0]

    assert target["bbox_touches_border"] is True
    assert target["bbox_border_distance_norm"] == 0.0
    assert target["local_object_count_r10"] == 2
    assert target["local_same_class_count_r10"] == 1
    assert target["overlapping_object_count"] == 2
    assert target["overlapping_same_class_count"] == 1
    assert target["overlapping_other_class_count"] == 1
    assert target["max_other_class_iou"] > 0.0
    assert target["max_target_coverage_by_other"] == pytest.approx(0.8)
    assert target["nearest_same_class_center_distance_norm"] < 0.10


def test_same_image_context_requires_source_dimensions_and_accepts_explicit_index():
    # ``width`` and ``height`` are bbox dimensions in public Class Analysis
    # points and must never be mistaken for source-image dimensions.
    public_record = _record(
        "target",
        "Boat",
        [100, 50, 300, 150],
        image_width=None,
        image_height=None,
        width=200,
        height=100,
    )
    missing = extract_same_image_context_features(
        [public_record],
        excluded_anchor_point_ids={"target"},
    )[0]
    assert missing == {
        "contract": SAME_IMAGE_CONTEXT_FEATURE_CONTRACT,
        "point_id": "target",
        "available": False,
        "unavailable_reason": "image_dimensions_missing",
    }

    explicit = extract_same_image_context_features(
        [public_record],
        excluded_anchor_point_ids={"target"},
        image_dimensions={("train", "image.jpg"): (1000, 500)},
    )[0]
    assert explicit["available"] is True
    assert explicit["bbox_width_norm"] == pytest.approx(0.2)
    assert explicit["bbox_height_norm"] == pytest.approx(0.2)
    assert explicit["bbox_center_y_norm"] == pytest.approx(0.2)


def test_same_image_context_fails_closed_on_conflicting_dimensions_and_bad_bbox():
    conflict_records = [
        _record("one", "Bike", [0, 0, 10, 10], image_width=100, image_height=100),
        _record("two", "Bike", [0, 0, 10, 10], image_width=200, image_height=100),
    ]
    conflict = extract_same_image_context_features(
        conflict_records,
        excluded_anchor_point_ids=set(),
    )
    assert {row["unavailable_reason"] for row in conflict} == {
        "image_dimensions_conflict"
    }

    malformed = extract_same_image_context_features(
        [_record("bad", "Bike", [10, 10, 5, 20])],
        excluded_anchor_point_ids=set(),
    )[0]
    assert malformed["available"] is False
    assert malformed["unavailable_reason"] == "bbox_degenerate"


def test_target_only_extraction_preserves_requested_input_order_and_full_anchor_pool():
    records = [
        _record("target-b", "TransitObject", [100, 100, 200, 200], image="b.jpg"),
        _record("anchor-b", "TransitObject", [300, 300, 400, 400], image="b.jpg"),
        _record("ignored", "TransitObject", [10, 10, 20, 20], image="a.jpg"),
        _record("target-a", "TransitObject", [30, 30, 50, 50], image="a.jpg"),
    ]

    rows = extract_same_image_context_features(
        records,
        excluded_anchor_point_ids={"target-a", "target-b"},
        target_point_ids={"target-a", "target-b"},
    )

    assert [row["point_id"] for row in rows] == ["target-b", "target-a"]
    assert [row["trusted_same_class_anchor_count"] for row in rows] == [1, 1]


def test_optional_alternative_scale_contrast_uses_trusted_alternative_anchors():
    records = [
        _record("target", "ElevatedFixture", _centered_box(0.5, 0.5, 0.20, 0.10)),
        _record("pole-1", "ElevatedFixture", _centered_box(0.2, 0.2, 0.02, 0.08)),
        _record("pole-2", "ElevatedFixture", _centered_box(0.8, 0.8, 0.02, 0.08)),
        _record("building-1", "Building", _centered_box(0.3, 0.3, 0.20, 0.10)),
        _record("building-2", "Building", _centered_box(0.7, 0.7, 0.20, 0.10)),
        # Explicitly excluded even though it is the proposed class.
        _record(
            "rough-building",
            "Building",
            _centered_box(0.6, 0.6, 0.03, 0.03),
            is_wrong_class_candidate=True,
        ),
    ]

    target = extract_same_image_context_features(
        records,
        excluded_anchor_point_ids={"target", "rough-building"},
        target_point_ids={"target"},
        alternative_class_by_point_id={"target": "Building"},
    )[0]

    assert target["alternative_class_count"] == 3
    assert target["trusted_alternative_class_anchor_count"] == 2
    assert target["excluded_alternative_class_anchor_count"] == 1
    assert target["scale_contrast_available"] is True
    assert target["alternative_class_log_area_residual"] == pytest.approx(0.0)
    assert target["current_minus_alternative_abs_scale_residual"] > 2.0
    assert target["current_minus_alternative_geometry_residual_magnitude"] > 1.5
