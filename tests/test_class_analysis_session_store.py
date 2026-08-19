from __future__ import annotations

from pathlib import Path
import json
import sqlite3

import pytest

import services.class_analysis_session_store as session_store_module
from services.class_analysis_session_store import (
    SessionStoreError,
    build_class_analysis_session_store,
    get_class_analysis_graph_payload,
    get_class_analysis_point_detail_payload,
    get_class_analysis_point_evidence_payload,
    get_class_analysis_review_queue_payload,
    get_class_analysis_session_identity_summary,
    ensure_class_analysis_session_store_validated,
    upsert_class_analysis_review_state,
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
            "annotation_entity_id": f"ae1-{index}",
            "annotation_entity_revision": index + 1,
            "annotation_entity_record_revision": f"alr1-record-{index}",
            "annotation_source_identity": "asi1-source",
            "source_record_key": f"train:images/{index}.jpg",
            "identity_status": "ready",
            "annotation_attestation": f"attestation-{index}",
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
        "schema": "class-analysis-session-store-v3",
        "point_count": 6,
        "evidence_count": 2,
        "projection_count": 12,
        "ready_identity_count": 6,
        "identity_conflict_count": 0,
        "invalid_identity_count": 0,
    }
    assert list(tmp_path.glob("*.partial")) == []
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT value FROM session_meta "
            "WHERE key = 'store_validation_id'"
        ).fetchone()
    assert len(json.loads(str(row[0]))) == 32


def test_review_projection_reuses_immutable_store_admission(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = _store(tmp_path)
    calls = []
    original = session_store_module.validate_class_analysis_session_store
    with session_store_module._SESSION_STORE_VALIDATION_CACHE_LOCK:
        session_store_module._SESSION_STORE_VALIDATION_CACHE.clear()

    def counted(*args, **kwargs):
        calls.append(Path(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        session_store_module,
        "validate_class_analysis_session_store",
        counted,
    )
    ensure_class_analysis_session_store_validated(path)
    upsert_class_analysis_review_state(
        path,
        point_id="point-0",
        disposition="confirm_current",
    )
    ensure_class_analysis_session_store_validated(path)
    assert calls == [path]


def test_invalid_store_is_rejected_before_review_projection_write(
    tmp_path: Path,
) -> None:
    path = _store(tmp_path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE points_core SET annotation_entity_id = '' "
            "WHERE point_id = 'point-0'"
        )
    with session_store_module._SESSION_STORE_VALIDATION_CACHE_LOCK:
        session_store_module._SESSION_STORE_VALIDATION_CACHE.clear()

    with pytest.raises(
        SessionStoreError,
        match="class_analysis_session_point_identity_invalid",
    ):
        upsert_class_analysis_review_state(
            path,
            point_id="point-0",
            disposition="confirm_current",
        )
    with sqlite3.connect(path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM review_state"
        ).fetchone()[0]
    assert count == 0


def test_graph_is_bounded_filtered_and_projection_explicit(tmp_path: Path) -> None:
    path = _store(tmp_path)
    first = get_class_analysis_graph_payload(path, projection_mode="global_pca", limit=2)
    second = get_class_analysis_graph_payload(path, projection_mode="global_pca", limit=2)
    assert first == second
    assert first["returned"] == 2
    assert first["total_matching"] == 6
    assert first["truncated"] is True
    assert first["columns"]["x"] == [0.0, 0.5]
    assert first["columns"]["annotation_entity_id"] == ["ae1-0", "ae1-1"]
    assert first["columns"]["identity_status"] == ["ready", "ready"]
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
    assert [item["annotation_entity_id"] for item in first["items"]] == [
        "ae1-1",
        "ae1-3",
    ]
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


def test_v2_saved_session_is_rejected_instead_of_becoming_reviewable(
    tmp_path: Path,
) -> None:
    path = _store(tmp_path)
    with sqlite3.connect(path) as connection:
        for column in (
            "annotation_attestation",
            "identity_status",
            "source_record_key",
            "annotation_source_identity",
            "annotation_entity_record_revision",
            "annotation_entity_revision",
            "annotation_entity_id",
        ):
            connection.execute(f"ALTER TABLE points_core DROP COLUMN {column}")
        connection.execute(
            "UPDATE session_meta SET value = ? WHERE key = 'schema'",
            (json.dumps("class-analysis-session-store-v2"),),
        )
    with session_store_module._SESSION_STORE_VALIDATION_CACHE_LOCK:
        session_store_module._SESSION_STORE_VALIDATION_CACHE.clear()

    with pytest.raises(
        SessionStoreError,
        match="class_analysis_session_store_schema_unsupported",
    ):
        get_class_analysis_graph_payload(path, projection_mode="umap", limit=2)
    with pytest.raises(
        SessionStoreError,
        match="class_analysis_session_store_schema_unsupported",
    ):
        get_class_analysis_review_queue_payload(path, limit=1)


def test_malformed_v3_identity_is_rejected_by_production_readers(
    tmp_path: Path,
) -> None:
    path = _store(tmp_path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE points_core SET annotation_entity_id = '' WHERE point_id = 'point-0'"
        )
    with session_store_module._SESSION_STORE_VALIDATION_CACHE_LOCK:
        session_store_module._SESSION_STORE_VALIDATION_CACHE.clear()

    with pytest.raises(
        SessionStoreError,
        match="class_analysis_session_point_identity_invalid",
    ):
        get_class_analysis_graph_payload(path, projection_mode="umap", limit=2)
    with pytest.raises(
        SessionStoreError,
        match="class_analysis_session_point_identity_invalid",
    ):
        get_class_analysis_review_queue_payload(path, limit=1)


def test_identity_summary_preserves_frozen_conflicts(tmp_path: Path) -> None:
    points = _points()
    points[0].update(
        {
            "annotation_entity_id": "",
            "annotation_entity_revision": 0,
            "annotation_entity_record_revision": "",
            "identity_status": "identity_conflict",
            "annotation_attestation": "",
        }
    )
    path = tmp_path / "conflicted.sqlite3"
    build_class_analysis_session_store(
        path,
        points,
        summary={"projection_mode": "umap"},
        request={"projection_mode": "umap"},
        expected_point_count=6,
    )

    summary = get_class_analysis_session_identity_summary(path)

    assert summary["point_count"] == 6
    assert summary["ready_identity_count"] == 5
    assert summary["identity_conflict_count"] == 1
    assert summary["invalid_identity_count"] == 0
    assert summary["source_identities"] == ["asi1-source"]
