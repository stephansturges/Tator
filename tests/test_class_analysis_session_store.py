from __future__ import annotations

from pathlib import Path

import pytest

from services.class_analysis_session_store import (
    SessionStoreError,
    build_class_analysis_session_store,
    get_class_analysis_graph_payload,
    get_class_analysis_point_detail_payload,
    get_class_analysis_point_evidence_payload,
    get_class_analysis_review_queue_payload,
    validate_class_analysis_session_store,
)


def _points() -> list[dict]:
    return [
        {
            "point_id": f"point-{index}",
            "class_id": str(index % 2),
            "class_name": "alpha" if index % 2 == 0 else "beta",
            "image_relpath": f"images/{index}.jpg",
            "frontend_image_key": f"image-{index}",
            "bbox_xyxy": [index, index + 1, index + 10, index + 11],
            "width": 10,
            "height": 10,
            "projection": [float(index), float(index * 2)],
            "quality_review_candidate": index < 4,
            "is_wrong_class_candidate": index in {0, 2},
            "is_rough_outlier_candidate": index in {1, 3},
            "tiny_object": index == 3,
            "review_priority_score": 1.0 - index / 10,
            "proposed_class": "beta" if index == 0 else "",
            "refined_outlier": (
                {
                    "human_review_rank": index + 1,
                    "sidecar_row": index,
                    "token_scores": [0.1, 0.8],
                }
                if index in {1, 3}
                else None
            ),
        }
        for index in range(6)
    ]


def _store(tmp_path: Path) -> Path:
    path = tmp_path / "session.sqlite3"
    build_class_analysis_session_store(
        path,
        _points(),
        summary={"projection_mode": "umap"},
        request={"projection_mode": "umap"},
        projection_coordinates={
            "global_pca": [[index / 2, -index] for index in range(6)]
        },
        expected_point_count=6,
    )
    return path


def test_store_is_atomic_normalized_and_validated(tmp_path: Path) -> None:
    path = _store(tmp_path)
    result = validate_class_analysis_session_store(
        path, expected_point_count=6, expected_evidence_count=2
    )
    assert result == {
        "schema": "class-analysis-session-store-v2",
        "point_count": 6,
        "evidence_count": 2,
        "projection_count": 12,
    }
    assert list(tmp_path.glob("*.partial")) == []


def test_graph_is_bounded_filtered_and_projection_explicit(tmp_path: Path) -> None:
    path = _store(tmp_path)
    first = get_class_analysis_graph_payload(path, projection_mode="global_pca", limit=2)
    second = get_class_analysis_graph_payload(path, projection_mode="global_pca", limit=2)
    assert first == second
    assert first["returned"] == 2
    assert first["total_matching"] == 6
    assert first["truncated"] is True
    assert first["columns"]["x"] == [0.0, 0.5]
    filtered = get_class_analysis_graph_payload(
        path,
        projection_mode="umap",
        class_name="alpha",
        objects="wrong_class",
        limit=10,
    )
    assert filtered["columns"]["point_id"] == ["point-0", "point-2"]
    with pytest.raises(SessionStoreError, match="projection_mode_unavailable"):
        get_class_analysis_graph_payload(path, projection_mode="tsne")


def test_review_queue_uses_stable_cursor_pages(tmp_path: Path) -> None:
    path = _store(tmp_path)
    first = get_class_analysis_review_queue_payload(path, limit=2)
    second = get_class_analysis_review_queue_payload(
        path, limit=2, cursor=first["next_cursor"]
    )
    assert [item["point_id"] for item in first["items"]] == ["point-1", "point-3"]
    assert [item["point_id"] for item in second["items"]] == ["point-0", "point-2"]
    assert second["next_cursor"] is None


def test_detail_and_evidence_are_separate_bounded_payloads(tmp_path: Path) -> None:
    path = _store(tmp_path)
    detail = get_class_analysis_point_detail_payload(path, "point-1")
    assert "refined_outlier" not in detail["point"]
    evidence = get_class_analysis_point_evidence_payload(path, "point-1")
    assert evidence["evidence"]["token_scores"] == [0.1, 0.8]
    with pytest.raises(SessionStoreError, match="evidence_not_found"):
        get_class_analysis_point_evidence_payload(path, "point-0")
