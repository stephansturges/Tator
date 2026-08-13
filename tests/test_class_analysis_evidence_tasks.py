from services.class_analysis_session_store import (
    build_class_analysis_session_store,
    claim_class_analysis_evidence_batch,
    class_analysis_evidence_progress,
    complete_class_analysis_evidence_batch,
    fail_class_analysis_evidence_batch,
    get_class_analysis_evidence_task,
    get_class_analysis_internal_evidence,
    get_class_analysis_point_evidence_payload,
    get_class_analysis_qwen_context,
    initialize_class_analysis_evidence_tasks,
    promote_class_analysis_evidence_task,
    release_class_analysis_evidence_leases,
)


def _build_store(tmp_path):
    path = tmp_path / "session.sqlite3"
    points = [
        {
            "point_id": "a",
            "image_relpath": "same.jpg",
            "class_name": "A",
            "bbox_xyxy": [0, 0, 10, 10],
            "review_priority_score": 4,
        },
        {
            "point_id": "b",
            "image_relpath": "same.jpg",
            "class_name": "B",
            "bbox_xyxy": [1, 1, 9, 9],
            "review_priority_score": 3,
        },
        {
            "point_id": "c",
            "image_relpath": "other.jpg",
            "class_name": "A",
            "bbox_xyxy": [2, 2, 8, 8],
            "review_priority_score": 2,
        },
    ]
    build_class_analysis_session_store(
        path,
        points,
        summary={},
        request={},
        projection_coordinates={},
        expected_point_count=len(points),
    )
    return path, points


def test_evidence_tasks_resume_in_source_groups_and_promote(tmp_path):
    path, points = _build_store(tmp_path)
    assert initialize_class_analysis_evidence_tasks(path, points) == {"pending": 3}

    first = claim_class_analysis_evidence_batch(path, limit=2)
    assert [item["point_id"] for item in first["items"]] == ["a", "b"]
    assert release_class_analysis_evidence_leases(path) == 2

    assert promote_class_analysis_evidence_task(path, "c") is True
    promoted = claim_class_analysis_evidence_batch(path, limit=1)
    assert [item["point_id"] for item in promoted["items"]] == ["c"]


def test_evidence_completion_is_private_but_available_to_vlm(tmp_path):
    path, points = _build_store(tmp_path)
    initialize_class_analysis_evidence_tasks(path, points)
    assert promote_class_analysis_evidence_task(path, "c") is True
    promoted = claim_class_analysis_evidence_batch(path, limit=1)
    evidence = {
        "status": "confirmed_outlier",
        "sidecar_row": 0,
        "_artifact": {
            "batch_relpath": "evidence_batches/example",
            "sidecar_file": "patch_refinement.npz",
            "manifest_file": "patch_refinement_manifest.json",
        },
    }
    assert complete_class_analysis_evidence_batch(
        path,
        lease_token=promoted["lease_token"],
        rows=[{"point_id": "c", "evidence": evidence}],
    ) == 1

    assert get_class_analysis_internal_evidence(path, "c")["_artifact"] == evidence["_artifact"]
    assert "_artifact" not in get_class_analysis_point_evidence_payload(path, "c")["evidence"]
    context = get_class_analysis_qwen_context(path, "c")
    target = next(point for point in context["points"] if point["point_id"] == "c")
    assert target["refined_outlier"]["sidecar_row"] == 0
    assert target["_evidence_artifact"] == evidence["_artifact"]
    assert get_class_analysis_evidence_task(path, "c")["status"] == "completed"

    remaining = claim_class_analysis_evidence_batch(path, limit=2)
    assert fail_class_analysis_evidence_batch(
        path,
        lease_token=remaining["lease_token"],
        error="transient",
        max_attempts=3,
    ) == 2
    progress = class_analysis_evidence_progress(path)
    assert progress["total"] == 3
    assert progress["completed"] == 1
    assert progress["failed"] == 0
    assert progress["pending"] == 2
    assert progress["processing"] == 0
    assert progress["counts"] == {"completed": 1, "retry": 2}
