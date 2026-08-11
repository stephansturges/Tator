import calendar
import hashlib
import json
import os
import re
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import pytest
from fastapi.testclient import TestClient

import localinferenceapi as api
from api.class_analysis import build_class_analysis_router


TRAINING_ACTION_SCHEMA = "vignette-training-action-v1"
TRAINING_ACTION_ROOT = Path("audit") / "vignette_training_actions"


def _review_object_key(
    *,
    image_sha256: str = "11" * 32,
    image_relpath: str = "nested/frame.jpg",
    class_name: str = "Boat",
    bbox_xyxy: Iterable[float] = (10, 20, 50, 80),
    source_key: str = "linked:aerial_demo",
) -> str:
    return api._class_analysis_review_object_key(
        source_key=source_key,
        image_sha256=image_sha256,
        split="train",
        image_relpath=image_relpath,
        class_name=class_name,
        geometry={"kind": "bbox", "bbox_xyxy": list(bbox_xyxy)},
        image_width=512,
        image_height=384,
    )


def _point(
    point_id: str,
    *,
    image_sha256: str = "11" * 32,
    image_relpath: str = "nested/frame.jpg",
    class_name: str = "Boat",
    bbox_xyxy: Iterable[float] = (10, 20, 50, 80),
    source_mode: str = "linked",
    source_id: str = "aerial_demo",
    source_key: str = "linked:aerial_demo",
) -> Dict[str, Any]:
    bbox = list(bbox_xyxy)
    return {
        "point_id": point_id,
        "review_object_key": _review_object_key(
            image_sha256=image_sha256,
            image_relpath=image_relpath,
            class_name=class_name,
            bbox_xyxy=bbox,
            source_key=source_key,
        ),
        "source_mode": source_mode,
        "source_id": source_id,
        "source_key": source_key,
        "split": "train",
        "image_relpath": image_relpath,
        "image_sha256": image_sha256,
        "class_name": class_name,
        "kind": "bbox",
        "bbox_xyxy": bbox,
        "wrong_class_suspicion": 0.93,
        "wrong_class_review_reason": "embedding_outlier",
        "embedding_wrong_class_suspicion": 0.88,
        "same_class_neighbor_ratio": 0.1,
        "top_other_neighbor_ratio": 0.9,
        "suggested_neighbor_class": "Building",
        "neighbor_class_counts": {"Boat": 1, "Building": 9},
        "review_signals": ["wrong_class", "dual_bbox_conflict"],
        "is_close_overlap_candidate": True,
        "close_overlap_matches": [
            {"point_id": "overlap", "class_name": "Building", "iou": 0.77}
        ],
        "is_dual_bbox_conflict": True,
        "dual_bbox_conflict": {
            "other_class": "Building",
            "iou": 0.77,
            "target_cover": 0.94,
        },
        "is_wrong_class_candidate": True,
    }


def _register_job(
    class_root: Path,
    *,
    job_id: str = "ca_vignette_capture",
    points: Optional[Iterable[Mapping[str, Any]]] = None,
) -> api.ClassAnalysisJob:
    resolved_points = [dict(point) for point in (points or [_point("p0")])]
    job_dir = class_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "summary": {
            "analysis_job_id": job_id,
            "source_mode": str(
                resolved_points[0].get("source_mode") or "linked"
            ),
            "source_id": str(
                resolved_points[0].get("source_id") or "aerial_demo"
            ),
            "source_key": str(
                resolved_points[0].get("source_key") or "linked:aerial_demo"
            ),
            "dataset_label": "Aerial Demo",
            "labelmap": ["Boat", "Building", "Person"],
        },
        "points": resolved_points,
        "wrong_class_candidates": [
            {
                "point_id": point["point_id"],
                "review_object_key": point["review_object_key"],
            }
            for point in resolved_points
        ],
    }
    job = api.ClassAnalysisJob(
        job_id=job_id,
        status="completed",
        result=result,
        summary=dict(result["summary"]),
        updated_at=time.time(),
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS[job_id] = job
    return job


@pytest.fixture
def capture_store(tmp_path, monkeypatch):
    class_root = tmp_path / "class_analysis"
    cache_root = tmp_path / "class_analysis_cache"
    monkeypatch.setattr(api, "CLASS_ANALYSIS_ROOT", class_root)
    monkeypatch.setattr(api, "CLASS_ANALYSIS_CACHE_ROOT", cache_root)

    with api.CLASS_ANALYSIS_JOBS_LOCK:
        previous_jobs = dict(api.CLASS_ANALYSIS_JOBS)
        api.CLASS_ANALYSIS_JOBS.clear()
    with api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS_LOCK:
        previous_reviews = dict(api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS)
        api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS.clear()
    with api.CLASS_ANALYSIS_TRAINING_ACTION_ID_INDEX_LOCK:
        previous_action_index = api.CLASS_ANALYSIS_TRAINING_ACTION_ID_INDEX.copy()
        api.CLASS_ANALYSIS_TRAINING_ACTION_ID_INDEX.clear()
    with api.CLASS_ANALYSIS_TRAINING_ARTIFACT_CACHE_LOCK:
        previous_artifact_cache = (
            api.CLASS_ANALYSIS_TRAINING_ARTIFACT_CACHE.copy()
        )
        api.CLASS_ANALYSIS_TRAINING_ARTIFACT_CACHE.clear()
    try:
        yield class_root
    finally:
        with api.CLASS_ANALYSIS_JOBS_LOCK:
            api.CLASS_ANALYSIS_JOBS.clear()
            api.CLASS_ANALYSIS_JOBS.update(previous_jobs)
        with api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS_LOCK:
            api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS.clear()
            api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS.update(previous_reviews)
        with api.CLASS_ANALYSIS_TRAINING_ACTION_ID_INDEX_LOCK:
            api.CLASS_ANALYSIS_TRAINING_ACTION_ID_INDEX.clear()
            api.CLASS_ANALYSIS_TRAINING_ACTION_ID_INDEX.update(
                previous_action_index
            )
        with api.CLASS_ANALYSIS_TRAINING_ARTIFACT_CACHE_LOCK:
            api.CLASS_ANALYSIS_TRAINING_ARTIFACT_CACHE.clear()
            api.CLASS_ANALYSIS_TRAINING_ARTIFACT_CACHE.update(
                previous_artifact_cache
            )


def _event_files(class_root: Path) -> list[Path]:
    root = class_root / TRAINING_ACTION_ROOT
    return sorted(root.rglob("*.jsonl")) if root.is_dir() else []


def _event_file_bytes(class_root: Path) -> Dict[Path, bytes]:
    return {
        path.relative_to(class_root): path.read_bytes()
        for path in _event_files(class_root)
    }


def _events(class_root: Path) -> list[Dict[str, Any]]:
    events: list[Dict[str, Any]] = []
    for path in _event_files(class_root):
        relative = path.relative_to(class_root / TRAINING_ACTION_ROOT)
        assert len(relative.parts) == 2
        assert (
            relative.parts[0] == "events"
            or re.fullmatch(
                r"\d{4}-\d{2}-\d{2}",
                relative.parts[0],
            )
        )
        assert re.fullmatch(r"[0-9a-f]{2}\.jsonl", relative.parts[1])
        for line in path.read_text(encoding="utf-8").splitlines():
            assert line.strip()
            event = json.loads(line)
            assert isinstance(event, dict)
            events.append(event)
    return sorted(
        events,
        key=lambda event: (
            float(event.get("recorded_at") or 0.0),
            str(event.get("action_id") or ""),
        ),
    )


def _install_fake_crop_snapshot(monkeypatch):
    artifacts: Dict[str, Dict[str, Any]] = {}

    def snapshot(_job, point):
        point_id = str(point.get("point_id") or "")
        artifact = artifacts.get(point_id)
        if artifact is None:
            artifact = api._class_analysis_store_vignette_training_blob(
                b"\xff\xd8fake-vignette-crop:" + point_id.encode("utf-8") + b"\xff\xd9",
                source_relpath=f"point/{point_id}/thumbnail.jpg",
                media_type="image/jpeg",
                role="object_crop",
            )
            artifacts[point_id] = artifact
        return [dict(artifact)]

    monkeypatch.setattr(
        api,
        "_class_analysis_snapshot_vignette_point_media",
        snapshot,
    )
    return artifacts


def _trust_captured_class_for_export(monkeypatch) -> None:
    """Model a linked dataset whose committed label matches the event."""

    monkeypatch.setattr(
        api,
        "_class_analysis_vignette_current_linked_class",
        lambda event, _cache: str(event.get("after_class") or ""),
    )


def _install_linked_annotation_state(
    class_root: Path,
    monkeypatch,
    *,
    current_class: str,
    source_id: str = "aerial_demo",
) -> tuple[Dict[str, Any], Dict[str, str]]:
    dataset_root = class_root.parent / f"linked-{source_id}"
    image_path = dataset_root / "images" / "nested" / "frame.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = api.Image.new("RGB", (512, 384), (31, 67, 103))
    image.save(image_path, format="JPEG", quality=95)
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    source_key = f"linked:{source_id}"
    point = _point(
        "p0",
        image_sha256=image_sha256,
        source_mode="linked",
        source_id=source_id,
        source_key=source_key,
    )
    point["image_width"] = 512
    point["image_height"] = 384
    labelmap = ["Boat", "Building", "Person"]
    class_id = labelmap.index(current_class)
    # bbox [10, 20, 50, 80] in a 512x384 image.
    label_line = (
        f"{class_id} {30 / 512:.10f} {50 / 384:.10f} "
        f"{40 / 512:.10f} {60 / 384:.10f}"
    )
    point["label_line"] = label_line
    point["label_line_index"] = 0
    entry = {
        "dataset_id": source_id,
        "dataset_root": str(dataset_root),
        "yolo_layout": "flat",
        "classes": labelmap,
    }

    def resolve_entry(requested_id):
        if str(requested_id) != source_id:
            raise api.HTTPException(status_code=404, detail="dataset_not_found")
        return entry

    monkeypatch.setattr(api, "_resolve_dataset_entry", resolve_entry)
    monkeypatch.setattr(
        api,
        "_dataset_effective_root_from_entry",
        lambda _entry: dataset_root,
    )
    monkeypatch.setattr(
        api,
        "_annotation_effective_label_lines",
        lambda *_args, **_kwargs: [label_line],
    )
    annotation_target = {
        "source_mode": "linked",
        "source_id": source_id,
        "split": "train",
        "image_relpath": "nested/frame.jpg",
    }
    return point, annotation_target


def _rehash_training_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    updated = json.loads(json.dumps(event))
    updated.pop("idempotency_payload_sha256", None)
    updated.pop("record_sha256", None)
    idempotency_payload = {
        key: value
        for key, value in updated.items()
        if key not in {"recorded_at", "recorded_at_iso"}
    }
    updated["idempotency_payload_sha256"] = hashlib.sha256(
        json.dumps(
            api._class_analysis_json_safe(idempotency_payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    updated["record_sha256"] = hashlib.sha256(
        json.dumps(
            api._class_analysis_json_safe(updated),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert api._class_analysis_vignette_training_event_hash_valid(updated)
    return updated


def _assert_commit_attestation(
    commit_event: Mapping[str, Any],
    source_event: Mapping[str, Any],
    *,
    expected_target: Mapping[str, Any],
    expected_method: str,
) -> Dict[str, Any]:
    attestation = commit_event.get("annotation_commit_attestation")
    assert isinstance(attestation, Mapping)
    assert (
        attestation["schema"]
        == "vignette-training-annotation-commit-attestation-v1"
    )
    assert attestation["verified"] is True
    assert attestation["annotation_target"] == dict(expected_target)
    assert attestation["image_sha256"] == source_event["point"]["image"]["sha256"]
    assert attestation["visual_object_key"] == source_event["visual_object_key"]
    assert attestation["geometry"] == source_event["point"]["bbox"]
    assert attestation["committed_class"] == source_event["after_class"]
    assert attestation["source_action_id"] == source_event["action_id"]
    assert (
        attestation["source_record_sha256"]
        == source_event["record_sha256"]
    )
    assert attestation["verification_method"] == expected_method
    assert attestation["verified_at"]
    digest = str(attestation.get("attestation_sha256") or "")
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    hashed_payload = {
        key: value
        for key, value in attestation.items()
        if key != "attestation_sha256"
    }
    assert digest == hashlib.sha256(
        json.dumps(
            api._class_analysis_json_safe(hashed_payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return dict(attestation)


def test_linked_reassignment_reader_requires_one_exact_valid_geometry(
    capture_store,
    monkeypatch,
):
    point, annotation_target = _install_linked_annotation_state(
        capture_store,
        monkeypatch,
        current_class="Building",
    )
    summary = {
        "source_mode": "linked",
        "source_id": "aerial_demo",
        "source_key": "linked:aerial_demo",
    }
    event = {
        "point": api._class_analysis_vignette_training_point_snapshot(
            point,
            summary,
        )
    }
    building_line = (
        f"1 {30 / 512:.10f} {50 / 384:.10f} "
        f"{40 / 512:.10f} {60 / 384:.10f}"
    )
    monkeypatch.setattr(
        api,
        "_annotation_effective_label_lines",
        lambda *_args, **_kwargs: [building_line],
    )
    assert api._class_analysis_vignette_current_annotation_target_class(
        event,
        annotation_target,
        {},
        require_unique=True,
    ) == "Building"

    monkeypatch.setattr(
        api,
        "_annotation_effective_label_lines",
        lambda *_args, **_kwargs: [
            building_line,
            building_line.replace("1 ", "999 ", 1),
        ],
    )
    assert api._class_analysis_vignette_current_annotation_target_class(
        event,
        annotation_target,
        {},
        require_unique=True,
    ) is None

    shifted_line = (
        f"1 {30.1 / 512:.10f} {50 / 384:.10f} "
        f"{40 / 512:.10f} {60 / 384:.10f}"
    )
    monkeypatch.setattr(
        api,
        "_annotation_effective_label_lines",
        lambda *_args, **_kwargs: [shifted_line],
    )
    assert api._class_analysis_vignette_current_annotation_target_class(
        event,
        annotation_target,
        {},
        require_unique=True,
    ) is None


def test_single_bbox_deletion_attestation_requires_exact_geometry_absence(
    capture_store,
    monkeypatch,
):
    point, annotation_target = _install_linked_annotation_state(
        capture_store,
        monkeypatch,
        current_class="Boat",
    )
    job = api.ClassAnalysisJob(
        job_id="ca_single_delete_attestation",
        status="completed",
        summary={
            "analysis_job_id": "ca_single_delete_attestation",
            "source_mode": "linked",
            "source_id": "aerial_demo",
            "source_key": "linked:aerial_demo",
            "labelmap": ["Boat", "Building", "Person"],
        },
    )
    dataset_root = capture_store.parent / "linked-aerial_demo"
    image_path = dataset_root / "images" / "nested" / "frame.jpg"
    source_identity = api._annotation_image_source_identity(
        source_mode="linked",
        source_id="aerial_demo",
        split="train",
        image_relpath=Path("nested/frame.jpg"),
        image_path=image_path,
        yolo_layout="flat",
    )
    boat_line = (
        f"0 {30 / 512:.10f} {50 / 384:.10f} "
        f"{40 / 512:.10f} {60 / 384:.10f}"
    )
    before_revision = api._annotation_image_label_revision([boat_line])
    committed_revision = api._annotation_image_label_revision([])
    monkeypatch.setattr(
        api,
        "_annotation_effective_label_lines",
        lambda *_args, **_kwargs: [],
    )
    attestation = api._class_analysis_validate_single_bbox_deletion_commit(
        job=job,
        point=point,
        summary=job.summary,
        annotation_target=annotation_target,
        before_revision=before_revision,
        committed_revision=committed_revision,
        expected_source_identity=source_identity,
    )
    assert attestation["committed"] is True
    assert attestation["committed_revision"] == committed_revision
    assert attestation["committed_label_count"] == 0
    assert attestation["verification_method"] == (
        api.CLASS_ANALYSIS_SINGLE_BBOX_DELETION_VERIFICATION_METHOD
    )
    assert attestation["deleted_label_line_index"] == 0
    assert re.fullmatch(
        r"[0-9a-f]{64}", attestation["deleted_label_line_sha256"]
    )
    assert attestation["attestation_sha256"] == (
        api._class_analysis_single_bbox_deletion_attestation_hash(
            attestation
        )
    )

    # A stale review card may observe a bbox that the user already deleted and
    # saved in the main editor.  That is a successful state reconciliation,
    # not evidence that this review click caused the transition.
    already_absent = (
        api._class_analysis_validate_single_bbox_deletion_commit(
            job=job,
            point=point,
            summary=job.summary,
            annotation_target=annotation_target,
            before_revision=committed_revision,
            committed_revision=committed_revision,
            expected_source_identity=source_identity,
            allow_already_absent=True,
        )
    )
    assert already_absent["committed"] is True
    assert already_absent["deletion_state"] == "already_absent"
    assert already_absent["before_revision"] == committed_revision
    assert already_absent["committed_revision"] == committed_revision
    assert already_absent["verification_method"] == (
        api.CLASS_ANALYSIS_SINGLE_BBOX_ALREADY_ABSENT_VERIFICATION_METHOD
    )
    assert "deleted_label_line_sha256" not in already_absent
    assert "deleted_label_line_index" not in already_absent
    assert already_absent["attestation_sha256"] == (
        api._class_analysis_single_bbox_deletion_attestation_hash(
            already_absent
        )
    )

    with pytest.raises(api.HTTPException) as wrong_mode_exc:
        api._class_analysis_validate_single_bbox_deletion_commit(
            job=job,
            point=point,
            summary=job.summary,
            annotation_target=annotation_target,
            before_revision=committed_revision,
            committed_revision=committed_revision,
            expected_source_identity=source_identity,
        )
    assert wrong_mode_exc.value.status_code == 409
    assert wrong_mode_exc.value.detail == (
        "single_bbox_deletion_commit_invalid"
    )

    # The browser serializes hydrated labels in class buckets with six-decimal
    # YOLO numbers. That can reorder otherwise untouched rows and normalize the
    # frozen target's textual representation. The causal transition proof must
    # still bind the same geometry and exactly one removed label.
    other_lines = [
        "1 0.2 0.2 0.1 0.1",
        "2 0.8 0.8 0.1 0.1",
    ]
    browser_boat_line = "0 0.058594 0.130208 0.078125 0.15625"
    browser_before_revision = api._annotation_image_label_revision(
        [other_lines[0], browser_boat_line, other_lines[1]]
    )
    browser_committed_revision = api._annotation_image_label_revision(
        other_lines
    )
    monkeypatch.setattr(
        api,
        "_annotation_effective_label_lines",
        lambda *_args, **_kwargs: list(other_lines),
    )
    browser_attestation = (
        api._class_analysis_validate_single_bbox_deletion_commit(
            job=job,
            point=point,
            summary=job.summary,
            annotation_target=annotation_target,
            before_revision=browser_before_revision,
            committed_revision=browser_committed_revision,
            expected_source_identity=source_identity,
        )
    )
    assert browser_attestation["deleted_label_line_index"] == 1
    assert browser_attestation["committed_label_count"] == 2

    monkeypatch.setattr(
        api,
        "_annotation_effective_label_lines",
        lambda *_args, **_kwargs: [boat_line],
    )
    with pytest.raises(api.HTTPException) as present_exc:
        api._class_analysis_validate_single_bbox_deletion_commit(
            job=job,
            point=point,
            summary=job.summary,
            annotation_target=annotation_target,
            before_revision=committed_revision,
            committed_revision=before_revision,
            expected_source_identity=source_identity,
        )
    assert present_exc.value.status_code == 409
    assert present_exc.value.detail["code"] == (
        "single_bbox_deletion_not_committed"
    )
    with pytest.raises(api.HTTPException) as present_reconcile_exc:
        api._class_analysis_validate_single_bbox_deletion_commit(
            job=job,
            point=point,
            summary=job.summary,
            annotation_target=annotation_target,
            before_revision=before_revision,
            committed_revision=before_revision,
            expected_source_identity=source_identity,
            allow_already_absent=True,
        )
    assert present_reconcile_exc.value.status_code == 409
    assert present_reconcile_exc.value.detail["code"] == (
        "single_bbox_deletion_not_committed"
    )

    monkeypatch.setattr(
        api,
        "_annotation_effective_label_lines",
        lambda *_args, **_kwargs: [],
    )
    with pytest.raises(api.HTTPException) as unrelated_before_exc:
        api._class_analysis_validate_single_bbox_deletion_commit(
            job=job,
            point=point,
            summary=job.summary,
            annotation_target=annotation_target,
            before_revision=f"alr1_{'ab' * 32}",
            committed_revision=committed_revision,
            expected_source_identity=source_identity,
        )
    assert unrelated_before_exc.value.status_code == 409
    assert unrelated_before_exc.value.detail["code"] == (
        "single_bbox_deletion_transition_unverified"
    )


def _read_export(response) -> tuple[Dict[str, Any], Dict[str, list[Dict[str, Any]]]]:
    output_path = Path(response.path)
    with zipfile.ZipFile(output_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        rows: Dict[str, list[Dict[str, Any]]] = {}
        for name in (
            "events",
            "classification",
            "preferences",
            "abstentions",
            "excluded",
        ):
            rows[name] = [
                json.loads(line)
                for line in archive.read(f"{name}.jsonl")
                .decode("utf-8")
                .splitlines()
                if line.strip()
            ]
    return manifest, rows


def _training_request(
    *,
    peer: str,
    host: str,
    origin: str = "",
    fetch_site: str = "",
    extra_headers: Optional[Mapping[str, str]] = None,
):
    headers = {"host": host}
    if origin:
        headers["origin"] = origin
    if fetch_site:
        headers["sec-fetch-site"] = fetch_site
    headers.update(
        {
            str(name).lower(): str(value)
            for name, value in (extra_headers or {}).items()
        }
    )
    return api.Request(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "scheme": "https" if origin.startswith("https://") else "http",
            "method": "POST",
            "path": "/class_analysis/jobs/job/training_actions",
            "raw_path": b"/class_analysis/jobs/job/training_actions",
            "query_string": b"",
            "headers": [
                (name.encode("latin-1"), value.encode("latin-1"))
                for name, value in headers.items()
            ],
            "client": (peer, 54321),
            "server": (host.split(":", 1)[0], 443),
        }
    )


def _capture_security_app(calls: Dict[str, Any]):
    def unused(*_args, **_kwargs):
        return {"status": "unused"}

    def disposition(job_id, point_id, payload):
        calls.setdefault("disposition", []).append(
            {
                "job_id": job_id,
                "point_id": point_id,
                "payload": dict(payload),
            }
        )
        return {"status": "recorded", "point_id": point_id}

    def training(job_id, payload):
        calls.setdefault("training", []).append(
            {"job_id": job_id, "payload": dict(payload)}
        )
        return {"status": "recorded"}

    def status():
        calls["status"] = int(calls.get("status") or 0) + 1
        return {"status": "ok"}

    def export():
        calls["export"] = int(calls.get("export") or 0) + 1
        return {"status": "exported"}

    router = build_class_analysis_router(
        capabilities_fn=unused,
        create_job_fn=unused,
        create_active_workspace_job_fn=unused,
        start_active_workspace_upload_fn=unused,
        batch_active_workspace_upload_fn=unused,
        finalize_active_workspace_upload_fn=unused,
        create_active_workspace_snapshot_job_fn=unused,
        cancel_active_workspace_upload_fn=unused,
        get_job_fn=unused,
        get_result_fn=unused,
        get_projection_fn=unused,
        get_thumbnail_fn=unused,
        record_review_disposition_fn=disposition,
        authorize_training_capture_request_fn=(
            api._class_analysis_authorize_training_capture_request
        ),
        record_training_action_fn=training,
        training_action_status_fn=status,
        export_training_actions_fn=export,
        create_cluster_search_fn=unused,
        get_cluster_search_fn=unused,
        cancel_cluster_search_fn=unused,
        cancel_job_fn=unused,
        create_qwen_review_fn=unused,
        list_qwen_reviews_fn=unused,
        get_qwen_review_fn=unused,
        cancel_qwen_review_fn=unused,
        get_qwen_review_evidence_fn=unused,
    )
    test_app = api.FastAPI()
    test_app.include_router(router)
    return test_app


def _nested_or_flat(
    payload: Mapping[str, Any],
    nested_name: str,
    *field_names: str,
) -> Any:
    nested = payload.get(nested_name)
    if isinstance(nested, Mapping):
        for field_name in field_names:
            if field_name in nested:
                return nested[field_name]
    for field_name in field_names:
        if field_name in payload:
            return payload[field_name]
    raise AssertionError(
        f"Expected {nested_name}.{field_names!r} or a flat equivalent in {payload!r}"
    )


def _assert_point_snapshot(event: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    captured = event["point"]
    assert isinstance(captured, Mapping)
    assert _nested_or_flat(captured, "identity", "id", "point_id") == source["point_id"]
    assert (
        _nested_or_flat(captured, "source", "mode", "source_mode")
        == source["source_mode"]
    )
    assert (
        _nested_or_flat(captured, "source", "id", "source_id")
        == source["source_id"]
    )
    assert (
        _nested_or_flat(captured, "source", "key", "source_key")
        == source["source_key"]
    )
    assert _nested_or_flat(captured, "image", "split") == source["split"]
    assert (
        _nested_or_flat(captured, "image", "relpath", "image_relpath")
        == source["image_relpath"]
    )
    assert (
        _nested_or_flat(captured, "class", "name", "class_name")
        == source["class_name"]
    )
    assert _nested_or_flat(captured, "bbox", "xyxy", "bbox_xyxy") == [
        float(value) for value in source["bbox_xyxy"]
    ]

    rails = captured.get("review_rails")
    if not isinstance(rails, Mapping):
        rails = captured.get("review")
    if not isinstance(rails, Mapping):
        rails = captured
    assert rails["wrong_class_suspicion"] == pytest.approx(0.93)
    assert rails["wrong_class_review_reason"] == "embedding_outlier"
    assert rails["suggested_neighbor_class"] == "Building"
    assert rails["neighbor_class_counts"] == {"Boat": 1, "Building": 9}
    assert rails["review_signals"] == ["wrong_class", "dual_bbox_conflict"]
    assert rails["is_dual_bbox_conflict"] is True


def _assert_common_event(
    event: Mapping[str, Any],
    *,
    action_type: str,
    point: Mapping[str, Any],
) -> None:
    assert event["schema"] == TRAINING_ACTION_SCHEMA
    assert re.fullmatch(r"[A-Za-z0-9_.:-]+", str(event["action_id"]))
    assert float(event["recorded_at"]) > 0
    assert isinstance(event["recorded_at_iso"], str)
    assert event["recorded_at_iso"]
    assert event["consent"] == {
        "explicit_opt_in": True,
        "control": "class_split_training_capture",
        "policy_version": "v1",
    }
    assert event["action_type"] == action_type
    assert event["analysis_job_id"] == "ca_vignette_capture"
    assert event["review_object_key"] == point["review_object_key"]
    _assert_point_snapshot(event, point)


@pytest.mark.parametrize("disposition", ["confirm_current", "skip"])
def test_explicit_capture_false_writes_no_training_action(
    capture_store,
    disposition,
):
    point = _point("p0")
    _register_job(capture_store, points=[point])

    response = api.record_class_analysis_review_disposition(
        "ca_vignette_capture",
        "p0",
        {
            "disposition": disposition,
            "origin": "desktop",
            "capture_training_data": False,
            "client_action_id": f"explicit-false-{disposition}",
        },
    )

    assert response["status"] == "recorded"
    assert _events(capture_store) == []
    assert not (capture_store / TRAINING_ACTION_ROOT).exists()


@pytest.mark.parametrize("disposition", ["confirm_current", "skip"])
def test_opted_in_review_dispositions_append_complete_immutable_events(
    capture_store,
    disposition,
):
    point = _point("p0")
    _register_job(capture_store, points=[point])

    response = api.record_class_analysis_review_disposition(
        "ca_vignette_capture",
        "p0",
        {
            "disposition": disposition,
            "origin": "desktop",
            "capture_training_data": True,
            "session_id": "browser-session-7",
            "group_id": "review-queue-3",
            "client_action_id": f"opted-in-{disposition}",
        },
    )
    events = _events(capture_store)

    assert response["status"] == "recorded"
    assert len(events) == 1
    event = events[0]
    _assert_common_event(event, action_type=disposition, point=point)
    assert event["origin"] == "desktop"
    assert event["session_id"] == "browser-session-7"
    assert event["group_id"] == "review-queue-3"
    assert event["before_class"] == "Boat"
    assert event["after_class"] == "Boat"
    assert event["model_review"] is None


def test_optional_capture_does_not_hold_global_review_mutation_lock(
    capture_store,
    monkeypatch,
):
    first_point = _point("p0", image_relpath="nested/first.jpg")
    second_point = _point("p1", image_relpath="nested/second.jpg")
    for point in (first_point, second_point):
        point["is_dual_bbox_conflict"] = False
        point["dual_bbox_conflict"] = None
    _register_job(capture_store, points=[first_point, second_point])

    capture_started = threading.Event()
    release_capture = threading.Event()
    second_finished = threading.Event()
    outcomes: Dict[str, Any] = {}

    def slow_capture(_job_id, payload):
        if str(payload.get("point_id") or "") == "p0":
            capture_started.set()
            assert release_capture.wait(timeout=3.0)
        return {"status": "recorded", "recorded_count": 1}

    monkeypatch.setattr(
        api,
        "record_class_analysis_vignette_training_action",
        slow_capture,
    )

    def record_first():
        try:
            outcomes["first"] = api.record_class_analysis_review_disposition(
                "ca_vignette_capture",
                "p0",
                {
                    "disposition": "confirm_current",
                    "origin": "desktop",
                    "capture_training_data": True,
                    "client_action_id": "nonblocking-capture-first",
                },
            )
        except Exception as exc:  # pragma: no cover - asserted below
            outcomes["first_error"] = exc

    def record_second():
        try:
            outcomes["second"] = api.record_class_analysis_review_disposition(
                "ca_vignette_capture",
                "p1",
                {
                    "disposition": "skip",
                    "origin": "desktop",
                    "capture_training_data": False,
                    "client_action_id": "nonblocking-capture-second",
                },
            )
        except Exception as exc:  # pragma: no cover - asserted below
            outcomes["second_error"] = exc
        finally:
            second_finished.set()

    first_thread = threading.Thread(target=record_first, daemon=True)
    first_thread.start()
    assert capture_started.wait(timeout=2.0)

    second_thread = threading.Thread(target=record_second, daemon=True)
    second_thread.start()
    assert second_finished.wait(timeout=1.0), (
        "an unrelated review action was serialized behind optional capture"
    )
    assert "second_error" not in outcomes
    assert outcomes["second"]["disposition"] == "skip"

    release_capture.set()
    first_thread.join(timeout=3.0)
    second_thread.join(timeout=3.0)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert "first_error" not in outcomes
    assert outcomes["first"]["training_capture"]["status"] == "recorded"


def test_training_capture_store_is_private_and_fsyncs_new_artifacts(
    capture_store,
    monkeypatch,
):
    point = _point("p0")
    _register_job(capture_store, points=[point])
    _install_fake_crop_snapshot(monkeypatch)
    fsynced = []
    real_fsync_directory = api._fsync_directory

    def observe_fsync(path):
        fsynced.append(Path(path))
        real_fsync_directory(path)

    monkeypatch.setattr(api, "_fsync_directory", observe_fsync)
    response = api.record_class_analysis_review_disposition(
        "ca_vignette_capture",
        "p0",
        {
            "disposition": "confirm_current",
            "origin": "desktop",
            "capture_training_data": True,
            "client_action_id": "private-capture-1",
        },
    )

    assert response["training_capture"]["status"] == "recorded"
    action_root = capture_store / TRAINING_ACTION_ROOT
    event_file = next((action_root / "events").glob("*.jsonl"))
    blob_file = next((action_root / "blobs" / "sha256").rglob("*.blob"))
    assert action_root.stat().st_mode & 0o777 == 0o700
    assert event_file.parent.stat().st_mode & 0o777 == 0o700
    assert blob_file.parent.stat().st_mode & 0o777 == 0o700
    assert event_file.stat().st_mode & 0o777 == 0o600
    assert blob_file.stat().st_mode & 0o777 == 0o600
    assert event_file.parent in fsynced
    assert blob_file.parent in fsynced


def test_repeated_action_and_undo_append_without_overwriting_prior_events(
    capture_store,
):
    point = _point("p0")
    _register_job(capture_store, points=[point])
    action_payload = {
        "disposition": "confirm_current",
        "origin": "desktop",
        "capture_training_data": True,
        "session_id": "browser-session-7",
        "group_id": "review-queue-3",
    }

    first = api.record_class_analysis_review_disposition(
        "ca_vignette_capture",
        "p0",
        {**action_payload, "client_action_id": "repeat-review-first"},
    )
    first_files = _event_file_bytes(capture_store)
    first_event = _events(capture_store)[0]
    first_clear = api.record_class_analysis_review_disposition(
        "ca_vignette_capture",
        "p0",
        {
            "disposition": "clear",
            "origin": "desktop",
            "capture_training_data": True,
            "session_id": "browser-session-7",
            "group_id": "review-queue-3",
            "client_action_id": "repeat-review-clear-first",
            "expected_revision": first["human_review_revision"],
        },
    )
    assert first_clear["status"] == "cleared"
    second = api.record_class_analysis_review_disposition(
        "ca_vignette_capture",
        "p0",
        {**action_payload, "client_action_id": "repeat-review-second"},
    )
    cleared = api.record_class_analysis_review_disposition(
        "ca_vignette_capture",
        "p0",
        {
            "disposition": "clear",
            "origin": "desktop",
            "capture_training_data": True,
            "session_id": "browser-session-7",
            "group_id": "review-queue-3",
            "client_action_id": "repeat-review-clear",
            "expected_revision": second["human_review_revision"],
        },
    )
    event_bytes_after_clear = _event_file_bytes(capture_store)
    retried_clear = api.record_class_analysis_review_disposition(
        "ca_vignette_capture",
        "p0",
        {
            "disposition": "clear",
            "origin": "desktop",
            "capture_training_data": True,
            "session_id": "browser-session-7",
            "group_id": "review-queue-3",
            "client_action_id": "repeat-review-clear",
            "expected_revision": second["human_review_revision"],
        },
    )
    assert cleared["status"] == "cleared"
    assert retried_clear["status"] == "already_clear"
    assert retried_clear["training_capture"] == {
        "status": "not_required",
        "reason": "review_already_clear",
    }
    assert _event_file_bytes(capture_store) == event_bytes_after_clear

    events = _events(capture_store)
    assert [event["action_type"] for event in events] == [
        "confirm_current",
        "undo_review",
        "confirm_current",
        "undo_review",
    ]
    assert len({event["action_id"] for event in events}) == 4
    assert events[1]["tombstone_of_action_id"] == events[0]["action_id"]
    assert events[3]["tombstone_of_action_id"] == events[2]["action_id"]
    assert events[0] == first_event
    for relative, original in first_files.items():
        current = (capture_store / relative).read_bytes()
        assert current.startswith(original)
    assert sum(
        len(content) for content in _event_file_bytes(capture_store).values()
    ) > sum(len(content) for content in first_files.values())


def test_class_change_records_before_and_after_classes(capture_store):
    point = _point("p0")
    _register_job(capture_store, points=[point])

    response = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "change_class",
            "point_id": "p0",
            "origin": "desktop",
            "session_id": "browser-session-8",
            "group_id": "review-queue-4",
            "before_class": "Boat",
            "after_class": "Building",
        },
    )

    assert isinstance(response, Mapping)
    event = _events(capture_store)[0]
    _assert_common_event(event, action_type="change_class", point=point)
    assert event["before_class"] == "Boat"
    assert event["after_class"] == "Building"


def test_single_bbox_delete_capture_is_bound_to_durable_proof_and_export(
    capture_store,
    monkeypatch,
):
    point = _point("p0")
    point["is_dual_bbox_conflict"] = False
    point["dual_bbox_conflict"] = None
    _register_job(capture_store, points=[point])
    _install_fake_crop_snapshot(monkeypatch)
    annotation_target = {
        "source_mode": "linked",
        "source_id": "aerial_demo",
        "split": "train",
        "image_relpath": "nested/frame.jpg",
    }
    attestation = {
        "schema": api.CLASS_ANALYSIS_SINGLE_BBOX_DELETION_ATTESTATION_SCHEMA,
        "committed": True,
        "analysis_job_id": "ca_vignette_capture",
        "point_id": "p0",
        "review_object_key": point["review_object_key"],
        "annotation_target": annotation_target,
        "source_identity": f"asi1_{'12' * 32}",
        "before_revision": f"alr1_{'34' * 32}",
        "committed_revision": f"alr1_{'56' * 32}",
        "image_sha256": point["image_sha256"],
        "committed_label_count": 0,
        "deleted_label_line_sha256": "ab" * 32,
        "deleted_label_line_index": 0,
        "verification_method": (
            api.CLASS_ANALYSIS_SINGLE_BBOX_DELETION_VERIFICATION_METHOD
        ),
        "verified_at": 100.0,
    }
    attestation["attestation_sha256"] = (
        api._class_analysis_single_bbox_deletion_attestation_hash(attestation)
    )
    api._class_analysis_record_review_disposition_entry(
        result={
            "summary": {"analysis_job_id": "ca_vignette_capture"}
        },
        point={
            **point,
            "single_bbox_deletion_attestation": attestation,
        },
        disposition="delete_bbox",
        origin="desktop",
        client_action_id="single-delete-capture",
        training_capture_requested=True,
    )
    payload = {
        "capture_training_data": True,
        "action_type": "delete_bbox",
        "point_id": "p0",
        "origin": "desktop",
        "client_action_id": "single-delete-capture",
        "label_commit_status": "committed",
        "annotation_target": annotation_target,
        "single_bbox_deletion_attestation": attestation,
    }

    first = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        payload,
    )
    replay = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        payload,
    )
    events = _events(capture_store)

    assert first["status"] == "recorded"
    assert replay["status"] == "already_recorded"
    assert len(events) == 1
    event = events[0]
    assert event["single_bbox_deletion_attestation"] == attestation
    rows = api._class_analysis_vignette_training_export_rows(events)
    assert [row["action_id"] for row in rows["geometry_decisions"]] == [
        event["action_id"]
    ]

    missing_proof = _rehash_training_event(
        {
            key: value
            for key, value in event.items()
            if key != "single_bbox_deletion_attestation"
        }
    )
    missing_rows = api._class_analysis_vignette_training_export_rows(
        [missing_proof]
    )
    assert missing_rows["geometry_decisions"] == []
    assert "single_bbox_deletion_attestation_invalid" in (
        missing_rows["excluded"][0]["reasons"]
    )

    wrong_job = _rehash_training_event(
        {**event, "analysis_job_id": "ca_other_job"}
    )
    wrong_job_rows = api._class_analysis_vignette_training_export_rows(
        [wrong_job]
    )
    assert wrong_job_rows["geometry_decisions"] == []
    assert "single_bbox_deletion_attestation_invalid" in (
        wrong_job_rows["excluded"][0]["reasons"]
    )

    with pytest.raises(api.HTTPException) as missing_exc_info:
        api.record_class_analysis_vignette_training_action(
            "ca_vignette_capture",
            {
                key: value
                for key, value in payload.items()
                if key != "single_bbox_deletion_attestation"
            },
        )
    assert missing_exc_info.value.status_code == 409
    assert missing_exc_info.value.detail == (
        "single_bbox_deletion_training_attestation_invalid"
    )


def test_already_absent_bbox_capture_is_saved_but_not_geometry_training(
    capture_store,
    monkeypatch,
):
    point = _point("p0")
    point["is_dual_bbox_conflict"] = False
    point["dual_bbox_conflict"] = None
    _register_job(capture_store, points=[point])
    _install_fake_crop_snapshot(monkeypatch)
    annotation_target = {
        "source_mode": "linked",
        "source_id": "aerial_demo",
        "split": "train",
        "image_relpath": "nested/frame.jpg",
    }
    current_revision = f"alr1_{'34' * 32}"
    attestation = {
        "schema": api.CLASS_ANALYSIS_SINGLE_BBOX_DELETION_ATTESTATION_SCHEMA,
        "committed": True,
        "deletion_state": "already_absent",
        "analysis_job_id": "ca_vignette_capture",
        "point_id": "p0",
        "review_object_key": point["review_object_key"],
        "annotation_target": annotation_target,
        "source_identity": f"asi1_{'12' * 32}",
        "before_revision": current_revision,
        "committed_revision": current_revision,
        "image_sha256": point["image_sha256"],
        "committed_label_count": 0,
        "verification_method": (
            api.CLASS_ANALYSIS_SINGLE_BBOX_ALREADY_ABSENT_VERIFICATION_METHOD
        ),
        "verified_at": 100.0,
    }
    attestation["attestation_sha256"] = (
        api._class_analysis_single_bbox_deletion_attestation_hash(
            attestation
        )
    )
    api._class_analysis_record_review_disposition_entry(
        result={
            "summary": {"analysis_job_id": "ca_vignette_capture"}
        },
        point={
            **point,
            "single_bbox_deletion_attestation": attestation,
        },
        disposition="delete_bbox",
        origin="desktop",
        client_action_id="single-absent-capture",
        training_capture_requested=True,
    )
    payload = {
        "capture_training_data": True,
        "action_type": "delete_bbox",
        "point_id": "p0",
        "origin": "desktop",
        "client_action_id": "single-absent-capture",
        "label_commit_status": "already_absent",
        "annotation_target": annotation_target,
        "single_bbox_deletion_attestation": attestation,
    }

    response = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        payload,
    )
    event = _events(capture_store)[0]
    rows = api._class_analysis_vignette_training_export_rows([event])

    assert response["status"] == "recorded"
    assert event["label_commit_status"] == "already_absent"
    assert event["single_bbox_deletion_attestation"] == attestation
    assert rows["geometry_decisions"] == []
    assert "single_bbox_deletion_attestation_invalid" in (
        rows["excluded"][0]["reasons"]
    )


def test_opt_in_capture_includes_compact_patch_refinement_rail(
    capture_store,
    monkeypatch,
):
    point = _point("p0")
    point["refined_outlier"] = {
        "schema": api.CLASS_ANALYSIS_REFINEMENT_SCHEMA,
        "decision_contract": (
            api.CLASS_ANALYSIS_REFINEMENT_DECISION_CONTRACT
        ),
        "status": "explained_not_outlier",
        "reason_codes": [
            "alternative_evidence_localized_to_overlap",
        ],
        "current_class": "Boat",
        "alternative_class": "Building",
        "current_support_score": 0.42,
        "alternative_support_score": 0.31,
        "intrinsic_current_support": 0.39,
        "intrinsic_alternative_support": 0.52,
        "directed_pair_raw_margin": 0.13,
        "directed_pair_probe_score": 0.264,
        "directed_pair_probe_features": [0.12, 0.42],
        "directed_pair_probe_feature_names": [
            "current_patch_exclusive_support",
            "alternative_patch_exclusive_support",
        ],
        "directed_pair_current_exclusive_support": 0.12,
        "directed_pair_alternative_exclusive_support": 0.42,
        "directed_pair_probe_threshold": 0.18,
        "directed_pair_probe_weights": [-0.6, 0.8],
        "directed_pair_probe_contract": (
            api.CLASS_ANALYSIS_REFINEMENT_V33_PAIR_PROBE_CONTRACT
        ),
        "directed_pair_probe_view_contract": (
            api.CLASS_ANALYSIS_REFINEMENT_V33_VIEW_FEATURE_CONTRACT
        ),
        "directed_pair_probe_lower_bound_contract": (
            api.CLASS_ANALYSIS_REFINEMENT_V33_LOWER_BOUND_CONTRACT
        ),
        "directed_pair_probe_fold_count": 1,
        "directed_pair_probe_fit_status": "ok",
        "directed_pair_probe_fold_digest": "cd" * 32,
        "directed_pair_probe_fit_eval_split_digest": "cd" * 32,
        "current_negative_threshold": 0.07,
        "current_support_threshold": 0.15,
        "current_strong_threshold": 0.25,
        "alternative_negative_threshold": 0.09,
        "alternative_support_threshold": 0.15,
        "alternative_strong_threshold": 0.25,
        "support_threshold_source": "fit_only_directed_pair",
        "directed_pair_reliable": True,
        "directed_pair_bank_reliable": True,
        "diagnostic_pair_reliability_contract": (
            api.CLASS_ANALYSIS_DIAGNOSTIC_PAIR_RELIABILITY_CONTRACT
        ),
        "diagnostic_pair_reliable": True,
        "diagnostic_pair_bank_reliable": True,
        "positive_confirmation_pair_reliable": True,
        "human_review_qualification_contract": (
            api.CLASS_ANALYSIS_HUMAN_REVIEW_QUALIFICATION_CONTRACT
        ),
        "human_review_rank_contract": (
            api.CLASS_ANALYSIS_HUMAN_REVIEW_RANK_CONTRACT
        ),
        "qualified_for_human_review": False,
        "human_review_rank": None,
        "directed_pair_candidate_source_excluded": False,
        "directed_pair_candidate_source_fingerprint": "12" * 8,
        "directed_pair_candidate_source_membership_roles": [],
        "directed_pair_heldout_auroc": 0.82,
        "directed_pair_eval_auroc_lower_bound": 0.68,
        "positive_confirmation_pair_probe_auroc_floor": 0.80,
        "positive_confirmation_pair_probe_auroc_lower_bound_floor": 0.60,
        "directed_pair_probe_fit_current_source_count": 12,
        "directed_pair_probe_fit_alternative_source_count": 13,
        "directed_pair_probe_eval_current_source_count": 9,
        "directed_pair_probe_eval_alternative_source_count": 10,
        "directed_pair_probe_fit_balanced_accuracy": 0.76,
        "directed_pair_probe_eval_sensitivity": 0.71,
        "directed_pair_probe_eval_specificity": 0.73,
        "directed_pair_current_absence_eval_fraction": 0.68,
        "directed_pair_alternative_strong_eval_fraction": 0.70,
        "decision_gates": {
            "directed_pair_reliable": True,
            "directed_pair_candidate_source_independent": True,
            "directed_pair_exact_calibration_contracts": True,
            "intrinsic_references_reliable": True,
            "diagnostic_pair_reliable": True,
            "positive_confirmation_pair_reliable": True,
            "qualified_for_human_review": False,
            "positive_confirmation_pair_probe_auroc_sufficient": True,
            "positive_confirmation_pair_probe_lower_bound_sufficient": True,
            "source_resolution_sufficient": True,
            "current_absent": False,
            "directed_pair_dominates": False,
            "alternative_strong": True,
            "alternative_exclusive_component_corresponds": True,
            "view_consistent": True,
            "alternative_evidence_external_to_overlap": False,
        },
        "alternative_evidence_inside_overlap_fraction": 0.91,
        "alternative_evidence_outside_overlap_fraction": 0.09,
        "current_evidence_outside_overlap_fraction": 0.73,
        "overlap_relation": "other_contains_target",
        "overlap_object_count": 1,
        "annotated_overlap_alternative_bbox_xyxy": [12, 22, 48, 78],
        "annotated_overlap_alternative_point_id": "overlap",
        "reference_reliable": True,
        "reference_distinct_source_count": 48,
        "current_reference_tier": "high",
        "alternative_reference_tier": "usable",
        "current_reference_heldout_auroc": 0.84,
        "alternative_reference_heldout_auroc": 0.76,
        "view_agreement": 0.88,
        "sidecar_row": 4,
        "source_image_sha256": "11" * 32,
        "_private_heatmaps": [[1.0, 2.0]],
    }
    api._class_analysis_assign_selector_priority_ranks([point])
    _register_job(capture_store, points=[point])
    _install_fake_crop_snapshot(monkeypatch)

    api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "change_class",
            "point_id": "p0",
            "origin": "desktop",
            "session_id": "browser-session-refinement",
            "group_id": "review-queue-refinement",
            "before_class": "Boat",
            "after_class": "Building",
        },
    )

    event = _events(capture_store)[0]
    rail = event["point"]["review_rails"]["patch_refinement"]
    assert rail["status"] == "explained_not_outlier"
    assert rail["reason_codes"] == [
        "alternative_evidence_localized_to_overlap"
    ]
    assert rail["current_support_score"] == pytest.approx(0.42)
    assert rail["alternative_support_score"] == pytest.approx(0.31)
    assert rail["overlap_relation"] == "other_contains_target"
    assert (
        rail["alternative_evidence_inside_overlap_fraction"]
        == pytest.approx(0.91)
    )
    assert rail["reference_reliable"] is True
    assert rail["sidecar_row"] == 4
    assert rail["advisory_only"] is True
    assert "cannot override" in rail["policy"]
    assert "_private_heatmaps" not in rail


def test_latest_completed_model_review_is_linked_with_artifact_locator(
    capture_store,
):
    point = _point("p0")
    _register_job(capture_store, points=[point])
    now = time.time()
    model_id = "MirilAI/Miril-DroneVLM-2B-2-MLX-4bit"
    old = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_old",
        parent_job_id="ca_vignette_capture",
        point_id="p0",
        status="completed",
        request={"model_id": model_id},
        result={"decision": "skip", "model_id": model_id},
        created_at=now - 30,
        updated_at=now - 20,
    )
    latest_result = {
        "decision": "confirm_current",
        "target_class": "Boat",
        "reason": "The target is unambiguously a small boat.",
        "model_id": "default",
        "reviewed_by_model": model_id,
        "model_invoked": True,
        "model_final_invoked": True,
        "model_final_completed": True,
        "model_final_validated": True,
        "model_provenance": {
            "schema": "qwen-review-model-provenance-v1",
            "requested_model_id": "default",
            "resolved_model_id": model_id,
            "runtime_platform": "mlx_vlm",
            "checkpoint_revision": "test-revision",
            "checkpoint_fingerprint": "a" * 64,
            "checkpoint_fingerprint_verified": False,
            "checkpoint_identity_verified": True,
            "fingerprint_strength": "revision_config_weight_manifest",
        },
    }
    latest = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_latest",
        parent_job_id="ca_vignette_capture",
        point_id="p0",
        status="completed",
        request={"model_id": "default"},
        result=latest_result,
        created_at=now - 15,
        updated_at=now - 10,
    )
    newer_failure = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_newer_failed",
        parent_job_id="ca_vignette_capture",
        point_id="p0",
        status="failed",
        request={"model_id": "broken/model"},
        error="load failed",
        created_at=now - 5,
        updated_at=now - 1,
    )
    latest_trace_path = None
    for review in (old, latest, newer_failure):
        review_dir = (
            capture_store
            / "ca_vignette_capture"
            / "qwen_reviews"
            / review.review_id
        )
        review_dir.mkdir(parents=True)
        (review_dir / "result.json").write_text(
            json.dumps(review.result or {"error": review.error}),
            encoding="utf-8",
        )
        if review.review_id == "cqr_latest":
            latest_trace_path = review_dir / "events.jsonl"
            latest_trace_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "model_input",
                                "phase": "specificity_probe",
                                "messages": [{"role": "user", "content": "probe"}],
                            }
                        ),
                        json.dumps(
                            {
                                "type": "model_partial_output",
                                "phase": "specificity_probe",
                                "text": "raw partial reasoning",
                            }
                        ),
                        json.dumps(
                            {
                                "type": "model_output",
                                "phase": "specificity_probe",
                                "text": "complete intermediate reasoning",
                            }
                        ),
                        json.dumps(
                            {
                                "type": "evidence_selected",
                                "phase": "controller",
                                "evidence_ids": ["target_detail_2"],
                            }
                        ),
                        json.dumps(
                            {
                                "type": "model_input",
                                "phase": "final_attempt_1",
                                "messages": [{"role": "user"}],
                            }
                        ),
                        json.dumps(
                            {
                                "type": "model_output",
                                "phase": "final_attempt_1",
                                "text": "{\"decision\":\"confirm_current\"}",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
    with api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS_LOCK:
        api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS.update(
            {review.review_id: review for review in (old, latest, newer_failure)}
        )

    api.record_class_analysis_review_disposition(
        "ca_vignette_capture",
        "p0",
        {
            "disposition": "confirm_current",
            "origin": "desktop",
            "capture_training_data": True,
            "client_action_id": "linked-review-capture",
        },
    )

    link = _events(capture_store)[0]["model_review"]
    assert link["review_id"] == "cqr_latest"
    assert link["model_id"] == model_id
    assert link["status"] == "completed"
    assert link["result"] == latest_result
    assert latest_trace_path is not None
    trace = link["vlm_trace"]
    assert trace == link["artifact_snapshot"]["vlm_trace"]
    assert trace["status"] == "complete"
    assert trace["event_count"] == 6
    assert trace["model_input_count"] == 2
    assert trace["model_output_count"] == 3
    assert trace["model_partial_output_count"] == 1
    assert trace["final_model_output_count"] == 1
    assert re.fullmatch(r"[0-9a-f]{64}", trace["sha256"])
    trace_blob = capture_store / trace["blob_relpath"]
    trace_bytes = trace_blob.read_bytes()
    assert trace_bytes == latest_trace_path.read_bytes()
    saved_trace_events = [
        json.loads(line)
        for line in trace_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    assert saved_trace_events[1]["text"] == "raw partial reasoning"
    assert saved_trace_events[2]["text"] == "complete intermediate reasoning"
    assert saved_trace_events[-1]["text"] == '{"decision":"confirm_current"}'
    assert "active_relpath" in link
    assert "archive_relpath" in link
    artifact_locators = [
        str(link.get(name) or "") for name in ("active_relpath", "archive_relpath")
    ]
    assert any(locator.endswith("qwen_reviews/cqr_latest") for locator in artifact_locators)

    explicit = api._class_analysis_vignette_training_model_review(
        job_id="ca_vignette_capture",
        point_id="p0",
        review_id="cqr_latest",
    )
    assert explicit["requested_model_id"] == "default"
    assert explicit["resolved_model_id"] == model_id
    assert explicit["checkpoint_fingerprint"] == "a" * 64
    assert explicit["checkpoint_fingerprint_verified"] is False
    assert explicit["checkpoint_identity_verified"] is True
    assert explicit["model_io_complete"] is True
    assert explicit["preference_train_eligible"] is True


@pytest.mark.parametrize(
    "job_id,payload,expected_status",
    [
        (
            "ca_vignette_capture",
            {
                "capture_training_data": True,
                "action_type": "not_a_review_action",
                "point_id": "p0",
            },
            400,
        ),
        (
            "ca_vignette_capture",
            {
                "capture_training_data": True,
                "action_type": "skip",
                "point_id": "missing",
            },
            404,
        ),
        (
            "missing_job",
            {
                "capture_training_data": True,
                "action_type": "skip",
                "point_id": "p0",
            },
            404,
        ),
        (
            "ca_vignette_capture",
            {
                "capture_training_data": True,
                "action_type": "change_class",
                "point_id": "p0",
                "before_class": "Boat",
            },
            400,
        ),
    ],
)
def test_malformed_or_unresolvable_actions_fail_without_corrupting_corpus(
    capture_store,
    job_id,
    payload,
    expected_status,
):
    _register_job(capture_store, points=[_point("p0")])
    api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "skip",
            "point_id": "p0",
            "origin": "desktop",
        },
    )
    before = _event_file_bytes(capture_store)

    with pytest.raises(api.HTTPException) as exc_info:
        api.record_class_analysis_vignette_training_action(job_id, payload)

    assert exc_info.value.status_code == expected_status
    assert _event_file_bytes(capture_store) == before
    assert len(_events(capture_store)) == 1


def test_startup_cleanup_preserves_vignette_training_action_store(
    capture_store,
    monkeypatch,
):
    _register_job(capture_store, points=[_point("p0")])
    api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "skip",
            "point_id": "p0",
            "origin": "desktop",
        },
    )
    before = _event_file_bytes(capture_store)
    state_before = dict(api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE)
    monkeypatch.setattr(api, "_class_analysis_delete_quarantined", lambda _paths: None)
    monkeypatch.setattr(
        api,
        "_class_analysis_enforce_cache_budget",
        lambda: {
            "files": 0,
            "bytes": 0,
            "removed_files": 0,
            "removed_bytes": 0,
        },
    )
    try:
        api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.update(
            {
                "status": "not_started",
                "started_at": None,
                "ready_at": None,
                "completed_at": None,
                "targets": 0,
                "error": None,
            }
        )

        api._class_analysis_startup_cleanup_worker()

        assert api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE["status"] == "completed"
        assert _event_file_bytes(capture_store) == before
        assert len(_events(capture_store)) == 1
    finally:
        api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.clear()
        api.CLASS_ANALYSIS_STARTUP_CLEANUP_STATE.update(state_before)


def test_vignette_training_action_route_is_exposed():
    paths = {getattr(route, "path", "") for route in api.app.routes}
    assert "/class_analysis/jobs/{job_id}/training_actions" in paths
    assert "/class_analysis/training_actions/status" in paths
    assert "/class_analysis/training_actions/export" in paths


def test_client_action_retry_is_idempotent(capture_store):
    point = _point("p0")
    _register_job(capture_store, points=[point])
    payload = {
        "capture_training_data": True,
        "action_type": "discard",
        "point_id": "p0",
        "origin": "desktop",
        "client_action_id": "browser-action-001",
    }

    first = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        payload,
    )
    second = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        payload,
    )

    assert first["status"] == "recorded"
    assert first["recorded_count"] == 1
    assert second["status"] == "already_recorded"
    assert second["recorded_count"] == 0
    assert second["duplicate_count"] == 1
    assert second["duplicate_action_ids"] == first["action_ids"]
    assert len(list(api._class_analysis_iter_vignette_training_actions())) == 1


@pytest.mark.parametrize("legacy_record_without_request_contract", [False, True])
def test_lost_response_retry_reuses_original_dynamic_review_snapshot(
    capture_store,
    monkeypatch,
    legacy_record_without_request_contract,
):
    point = _point("p0")
    _register_job(capture_store, points=[point])
    dynamic_state = {"generation": "first"}
    calls = {"model_review": 0, "point_media": 0}

    def snapshot_model_review(**_kwargs):
        calls["model_review"] += 1
        generation = dynamic_state["generation"]
        return {
            "review_id": f"review-{generation}",
            "linkage_method": "heuristic_latest_completed",
            "registry_validated": True,
            "result": {"decision": generation},
            "preference_train_eligible": False,
        }

    def snapshot_point_media(_job, _point_value):
        calls["point_media"] += 1
        generation = dynamic_state["generation"]
        return [
            {
                "role": "object_crop",
                "source_relpath": f"evidence/{generation}.jpg",
                "sha256": ("a" if generation == "first" else "b") * 64,
                "media_type": "image/jpeg",
                "size_bytes": 1,
            }
        ]

    monkeypatch.setattr(
        api,
        "_class_analysis_vignette_training_model_review",
        snapshot_model_review,
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_snapshot_vignette_point_media",
        snapshot_point_media,
    )
    payload = {
        "capture_training_data": True,
        "action_type": "confirm_current",
        "point_id": "p0",
        "origin": "desktop",
        "client_action_id": "lost-response-review-snapshot",
    }

    first = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        payload,
    )
    if legacy_record_without_request_contract:
        [event_path] = _event_files(capture_store)
        legacy_event = json.loads(event_path.read_text(encoding="utf-8"))
        legacy_event.pop("request_contract")
        idempotency_payload = {
            key: value
            for key, value in legacy_event.items()
            if key
            not in {
                "idempotency_payload_sha256",
                "record_sha256",
                "recorded_at",
                "recorded_at_iso",
            }
        }
        legacy_event["idempotency_payload_sha256"] = hashlib.sha256(
            json.dumps(
                idempotency_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        record_payload = {
            key: value
            for key, value in legacy_event.items()
            if key != "record_sha256"
        }
        legacy_event["record_sha256"] = hashlib.sha256(
            json.dumps(
                record_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        event_path.write_text(
            json.dumps(
                legacy_event,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        with api.CLASS_ANALYSIS_TRAINING_ACTION_ID_INDEX_LOCK:
            api.CLASS_ANALYSIS_TRAINING_ACTION_ID_INDEX.clear()
    dynamic_state["generation"] = "second"
    retry = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        payload,
    )

    assert first["status"] == "recorded"
    assert retry["status"] == "already_recorded"
    assert retry["duplicate_action_ids"] == first["action_ids"]
    assert calls == {"model_review": 1, "point_media": 1}
    events = list(api._class_analysis_iter_vignette_training_actions())
    assert len(events) == 1
    assert events[0]["model_review"]["review_id"] == "review-first"
    assert events[0]["artifacts"][0]["source_relpath"] == "evidence/first.jpg"
    if legacy_record_without_request_contract:
        assert "request_contract" not in events[0]
    else:
        assert (
            events[0]["request_contract"]["schema"]
            == api.CLASS_ANALYSIS_TRAINING_ACTION_REQUEST_CONTRACT_SCHEMA
        )


def test_dynamic_replay_shortcut_still_binds_request_authorization(
    capture_store,
    monkeypatch,
):
    point = _point("p0")
    _register_job(capture_store, points=[point])
    model_review_calls = 0

    def snapshot_model_review(**_kwargs):
        nonlocal model_review_calls
        model_review_calls += 1
        return None

    monkeypatch.setattr(
        api,
        "_class_analysis_vignette_training_model_review",
        snapshot_model_review,
    )
    payload = {
        "capture_training_data": True,
        "action_type": "confirm_current",
        "point_id": "p0",
        "origin": "desktop",
        "client_action_id": "authorization-bound-retry",
        "_training_authorization": {
            "source": "local_loopback",
            "actor": "local_user",
        },
    }
    api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        payload,
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.record_class_analysis_vignette_training_action(
            "ca_vignette_capture",
            {
                **payload,
                "_training_authorization": {
                    "source": "configured_token",
                    "actor": "token_holder",
                },
            },
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "training_action_idempotency_conflict"
    assert model_review_calls == 1
    assert len(list(api._class_analysis_iter_vignette_training_actions())) == 1


def test_client_action_retry_with_changed_payload_conflicts(capture_store):
    point = _point("p0")
    _register_job(capture_store, points=[point])
    common = {
        "capture_training_data": True,
        "point_id": "p0",
        "origin": "desktop",
        "client_action_id": "browser-action-002",
    }
    api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {**common, "action_type": "discard"},
    )
    before = _event_file_bytes(capture_store)

    with pytest.raises(api.HTTPException) as exc_info:
        api.record_class_analysis_vignette_training_action(
            "ca_vignette_capture",
            {**common, "action_type": "skip"},
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "training_action_idempotency_conflict"
    assert _event_file_bytes(capture_store) == before
    assert len(list(api._class_analysis_iter_vignette_training_actions())) == 1


def test_visual_object_key_is_stable_across_class_transition(capture_store):
    initial = _point("p0", class_name="Boat")
    initial["image_width"] = 512
    initial["image_height"] = 384
    initial["label_line"] = "0 0.058594 0.130208 0.078125 0.156250"
    job = _register_job(capture_store, points=[initial])

    api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "discard",
            "point_id": "p0",
            "client_action_id": "before-class-transition",
        },
    )
    transitioned = job.result["points"][0]
    transitioned["class_name"] = "Building"
    transitioned["label_line"] = "1 0.058594 0.130208 0.078125 0.156250"
    transitioned["review_object_key"] = _review_object_key(
        image_sha256=transitioned["image_sha256"],
        image_relpath=transitioned["image_relpath"],
        class_name="Building",
        bbox_xyxy=transitioned["bbox_xyxy"],
    )
    api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "discard",
            "point_id": "p0",
            "client_action_id": "after-class-transition",
        },
    )

    events = {
        event["client_action_id"]: event
        for event in api._class_analysis_iter_vignette_training_actions()
    }
    before = events["before-class-transition"]
    after = events["after-class-transition"]
    assert before["review_object_key"] != after["review_object_key"]
    assert before["visual_object_key"] == after["visual_object_key"]
    assert (
        before["point"]["identity"]["visual_object_key"]
        == after["point"]["identity"]["visual_object_key"]
    )
    assert before["point"]["class"]["name"] == "Boat"
    assert after["point"]["class"]["name"] == "Building"


def test_review_id_mismatch_does_not_roll_back_primary_disposition(
    capture_store,
):
    p0 = _point("p0")
    p1 = _point(
        "p1",
        image_sha256="22" * 32,
        image_relpath="nested/other.jpg",
    )
    _register_job(capture_store, points=[p0, p1])
    mismatched = api.ClassAnalysisQwenReviewJob(
        review_id="cqr_for_other_point",
        parent_job_id="ca_vignette_capture",
        point_id="p1",
        status="completed",
        result={"decision": "confirm_current"},
    )
    with api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS_LOCK:
        api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS[mismatched.review_id] = mismatched

    response = api.record_class_analysis_review_disposition(
        "ca_vignette_capture",
        "p0",
        {
            "disposition": "confirm_current",
            "capture_training_data": True,
            "review_id": mismatched.review_id,
            "origin": "desktop",
            "client_action_id": "mismatched-review-link",
        },
    )

    assert response["status"] == "recorded"
    assert response["training_capture"] == {
        "status": "failed",
        "detail": "training_action_review_mismatch",
    }
    persisted = api._class_analysis_lookup_review_dispositions(
        [p0["review_object_key"]]
    )
    assert persisted[p0["review_object_key"]]["disposition"] == "confirm_current"
    assert list(api._class_analysis_iter_vignette_training_actions()) == []


def test_multi_point_batch_is_atomic_and_iterator_flattens_it(capture_store):
    p0 = _point("p0")
    p1 = _point(
        "p1",
        image_sha256="22" * 32,
        image_relpath="nested/other.jpg",
    )
    _register_job(capture_store, points=[p0, p1])

    with pytest.raises(api.HTTPException) as exc_info:
        api.record_class_analysis_vignette_training_action(
            "ca_vignette_capture",
            {
                "capture_training_data": True,
                "action_type": "discard",
                "point_ids": ["p0", "missing"],
                "client_action_id": "invalid-batch",
            },
        )
    assert exc_info.value.status_code == 404
    assert _event_files(capture_store) == []

    response = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "discard",
            "point_ids": ["p0", "p1"],
            "client_action_id": "valid-batch",
            "group_id": "two-vignettes",
        },
    )

    assert response["status"] == "recorded"
    assert response["recorded_count"] == 2
    assert response["batch_id"].startswith("batch:")
    files = _event_files(capture_store)
    assert len(files) == 1
    physical_records = [
        json.loads(line)
        for line in files[0].read_text(encoding="utf-8").splitlines()
    ]
    assert len(physical_records) == 1
    assert physical_records[0]["schema"] == "vignette-training-action-batch-v1"
    assert physical_records[0]["expected_count"] == 2
    assert len(physical_records[0]["events"]) == 2
    flattened = list(api._class_analysis_iter_vignette_training_actions())
    assert {event["point"]["identity"]["id"] for event in flattened} == {
        "p0",
        "p1",
    }
    assert {event["action_type"] for event in flattened} == {"discard"}


def test_training_capture_status_counts_flattened_actions(
    capture_store,
    monkeypatch,
):
    p0 = _point("p0")
    p1 = _point(
        "p1",
        image_sha256="22" * 32,
        image_relpath="nested/one.jpg",
    )
    p2 = _point(
        "p2",
        image_sha256="33" * 32,
        image_relpath="nested/two.jpg",
    )
    _register_job(capture_store, points=[p0, p1, p2])
    monkeypatch.setattr(
        api,
        "_class_analysis_snapshot_vignette_point_media",
        lambda _job, _point: [],
    )
    api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "skip",
            "point_id": "p0",
            "client_action_id": "status-skip",
        },
    )
    api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "discard",
            "point_ids": ["p1", "p2"],
            "client_action_id": "status-navigation",
        },
    )

    status = api.get_class_analysis_vignette_training_status()

    assert status["schema"] == TRAINING_ACTION_SCHEMA
    assert status["event_count"] == 3
    assert status["valid_hash_count"] == 3
    assert status["action_counts"] == {"skip": 1, "discard": 2}
    assert status["event_bytes"] > 0
    assert status["blob_count"] == 0
    assert status["blob_bytes"] == 0
    assert status["total_bytes"] == status["event_bytes"]
    assert status["explicit_opt_in_required"] is True


def test_export_separates_labels_abstentions_and_nontraining_history(
    capture_store,
    monkeypatch,
):
    points = [
        _point(
            f"p{index}",
            image_sha256=f"{index + 1:02x}" * 32,
            image_relpath=f"nested/{index}.jpg",
        )
        for index in range(5)
    ]
    _register_job(capture_store, points=points)
    _install_fake_crop_snapshot(monkeypatch)
    _trust_captured_class_for_export(monkeypatch)

    classification = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "change_class",
            "point_id": "p0",
            "before_class": "Boat",
            "after_class": "Building",
            "client_action_id": "export-classification",
        },
    )
    classification_id = classification["action_ids"][0]
    classification_commit = (
        api.record_class_analysis_vignette_training_action(
            "ca_vignette_capture",
            {
                "capture_training_data": True,
                "action_type": "commit_class_change",
                "point_id": "p0",
                "before_class": "Boat",
                "after_class": "Building",
                "label_commit_status": "committed",
                "client_action_id": "export-classification:commit",
                "commits_action_id": classification_id,
            },
        )
    )
    abstention = api.record_class_analysis_review_disposition(
        "ca_vignette_capture",
        "p1",
        {
            "capture_training_data": True,
            "disposition": "skip",
            "client_action_id": "export-abstention",
        },
    )
    navigation = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "discard",
            "point_id": "p2",
            "client_action_id": "export-navigation",
        },
    )
    pending = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "change_class",
            "point_id": "p3",
            "before_class": "Boat",
            "after_class": "Person",
            "label_commit_status": "pending_desktop_sync",
            "client_action_id": "export-pending",
        },
    )
    confirmed = api.record_class_analysis_review_disposition(
        "ca_vignette_capture",
        "p4",
        {
            "disposition": "confirm_current",
            "capture_training_data": True,
            "client_action_id": "export-confirm-then-undo",
        },
    )
    confirmed_action_id = confirmed["training_capture"]["action_ids"][0]
    undone = api.record_class_analysis_review_disposition(
        "ca_vignette_capture",
        "p4",
        {
            "disposition": "clear",
            "capture_training_data": True,
            "client_action_id": "export-undo",
            "tombstone_of_action_id": confirmed_action_id,
            "expected_revision": confirmed["human_review_revision"],
        },
    )

    manifest, rows = _read_export(
        api.export_class_analysis_vignette_training_actions()
    )

    classification_commit_id = classification_commit["action_ids"][0]
    abstention_id = abstention["training_capture"]["action_ids"][0]
    navigation_id = navigation["action_ids"][0]
    pending_id = pending["action_ids"][0]
    undo_id = undone["training_capture"]["action_ids"][0]
    assert manifest["raw_event_count"] == 7
    assert manifest["classification_count"] == 1
    assert manifest["abstention_count"] == 1
    assert {
        row["action_id"] for row in rows["classification"]
    } == {classification_id}
    assert rows["classification"][0]["class_name"] == "Building"
    assert {row["action_id"] for row in rows["abstentions"]} == {
        abstention_id
    }
    excluded = {
        row["action_id"]: set(row["reasons"]) for row in rows["excluded"]
    }
    assert "navigation_only" in excluded[navigation_id]
    assert "annotation_not_committed" in excluded[pending_id]
    assert "explicitly_undone" in excluded[confirmed_action_id]
    assert "history_marker" in excluded[undo_id]
    assert "history_marker" in excluded[classification_commit_id]
    assert manifest["rules"]["skip_is_label"] is False
    assert manifest["rules"]["navigation_is_training_example"] is False
    assert manifest["rules"]["pending_changes_excluded"] is True
    assert manifest["rules"]["undone_or_superseded_excluded"] is True


def test_corrupt_crop_blob_is_excluded_from_export(
    capture_store,
    monkeypatch,
):
    point = _point("p0")
    _register_job(capture_store, points=[point])
    _install_fake_crop_snapshot(monkeypatch)
    _trust_captured_class_for_export(monkeypatch)
    recorded = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "change_class",
            "point_id": "p0",
            "before_class": "Boat",
            "after_class": "Building",
            "client_action_id": "corrupt-artifact",
        },
    )
    event = list(api._class_analysis_iter_vignette_training_actions())[0]
    artifact = event["artifacts"][0]
    blob_path = capture_store / artifact["blob_relpath"]
    original_size = int(artifact["bytes"])
    blob_path.write_bytes(b"x" * original_size)
    assert hashlib.sha256(blob_path.read_bytes()).hexdigest() != artifact["sha256"]

    manifest, rows = _read_export(
        api.export_class_analysis_vignette_training_actions()
    )

    action_id = recorded["action_ids"][0]
    assert manifest["classification_count"] == 0
    assert manifest["artifact_count"] == 0
    assert rows["classification"] == []
    excluded = {
        row["action_id"]: set(row["reasons"]) for row in rows["excluded"]
    }
    assert "object_crop_missing_or_corrupt" in excluded[action_id]


@pytest.mark.parametrize(
    ("disposition", "expected_reason"),
    [
        ("confirm_current", "current_review_state_not_confirmed"),
        ("skip", "current_review_state_not_skipped"),
    ],
)
def test_uncaptured_undo_still_invalidates_prior_captured_review(
    capture_store,
    monkeypatch,
    disposition,
    expected_reason,
):
    point = _point("p0")
    _register_job(capture_store, points=[point])
    _install_fake_crop_snapshot(monkeypatch)
    _trust_captured_class_for_export(monkeypatch)
    captured = api.record_class_analysis_review_disposition(
        "ca_vignette_capture",
        "p0",
        {
            "disposition": disposition,
            "capture_training_data": True,
            "client_action_id": "captured-confirm-before-opt-out",
        },
    )
    captured_action_id = captured["training_capture"]["action_ids"][0]

    cleared = api.record_class_analysis_review_disposition(
        "ca_vignette_capture",
        "p0",
        {
            "disposition": "clear",
            "capture_training_data": False,
            "client_action_id": "uncaptured-review-clear",
            "expected_revision": captured["human_review_revision"],
        },
    )

    assert cleared["status"] == "cleared"
    assert len(list(api._class_analysis_iter_vignette_training_actions())) == 1
    rows = api._class_analysis_vignette_training_export_rows(
        list(api._class_analysis_iter_vignette_training_actions())
    )
    assert rows["classification"] == []
    excluded = {
        row["action_id"]: set(row["reasons"]) for row in rows["excluded"]
    }
    assert expected_reason in excluded[captured_action_id]


def test_recorded_timestamp_is_covered_by_event_integrity(capture_store):
    point = _point("p0")
    _register_job(capture_store, points=[point])
    api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "discard",
            "point_id": "p0",
            "client_action_id": "timestamp-integrity",
        },
    )
    event = list(api._class_analysis_iter_vignette_training_actions())[0]
    assert api._class_analysis_vignette_training_event_hash_valid(event)
    tampered = json.loads(json.dumps(event))
    tampered["recorded_at"] = float(tampered["recorded_at"]) + 86_400
    tampered["recorded_at_iso"] = "2099-01-01T00:00:00Z"

    assert not api._class_analysis_vignette_training_event_hash_valid(tampered)
    rows = api._class_analysis_vignette_training_export_rows([tampered])
    assert rows["classification"] == []
    assert rows["abstentions"] == []
    assert rows["excluded"] == [
        {
            "action_id": event["action_id"],
            "reasons": ["event_hash_invalid", "navigation_only"],
        }
    ]


@pytest.mark.parametrize(
    "tamper",
    ["remove_child", "reorder_children", "replace_batch_hash"],
)
def test_tampered_batch_fails_closed_for_iteration_and_export(
    capture_store,
    tamper,
):
    points = [
        _point(
            f"p{index}",
            image_sha256=f"{index + 1:02x}" * 32,
            image_relpath=f"nested/batch-{index}.jpg",
        )
        for index in range(3)
    ]
    _register_job(capture_store, points=points)
    api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "discard",
            "point_ids": ["p0", "p1", "p2"],
            "client_action_id": f"batch-tamper-{tamper}",
        },
    )
    event_file = _event_files(capture_store)[0]
    record = json.loads(event_file.read_text(encoding="utf-8"))
    assert record["schema"] == "vignette-training-action-batch-v1"
    if tamper == "remove_child":
        record["events"].pop()
        # Updating the public count must not be enough to conceal removal.
        record["expected_count"] = len(record["events"])
    elif tamper == "reorder_children":
        record["events"] = list(reversed(record["events"]))
    else:
        record["idempotency_payload_sha256"] = "0" * 64
    event_file.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with api.CLASS_ANALYSIS_TRAINING_ACTION_ID_INDEX_LOCK:
        api.CLASS_ANALYSIS_TRAINING_ACTION_ID_INDEX.clear()

    with pytest.raises(api.HTTPException) as iter_error:
        list(api._class_analysis_iter_vignette_training_actions())
    assert iter_error.value.status_code == 500
    assert iter_error.value.detail == "vignette_training_action_store_corrupt"
    with pytest.raises(api.HTTPException) as export_error:
        api.export_class_analysis_vignette_training_actions()
    assert export_error.value.status_code == 500
    assert export_error.value.detail == "vignette_training_action_store_corrupt"


def test_client_action_retry_is_idempotent_across_utc_day_shards(
    capture_store,
    monkeypatch,
):
    point = _point("p0")
    _register_job(capture_store, points=[point])
    real_time = time

    class Clock:
        def __init__(self):
            self.now = float(
                calendar.timegm((2026, 7, 30, 23, 59, 59, 0, 0, 0))
            )

        def time(self):
            return self.now

        def __getattr__(self, name):
            return getattr(real_time, name)

    clock = Clock()
    monkeypatch.setattr(api, "time", clock)
    payload = {
        "capture_training_data": True,
        "action_type": "discard",
        "point_id": "p0",
        "client_action_id": "cross-midnight-action",
    }
    first = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        payload,
    )
    clock.now += 2
    second = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        payload,
    )

    assert first["status"] == "recorded"
    assert second["status"] == "already_recorded"
    assert second["duplicate_action_ids"] == first["action_ids"]
    assert len(_event_files(capture_store)) == 1
    assert len(list(api._class_analysis_iter_vignette_training_actions())) == 1


def test_reordered_batch_retry_is_one_idempotent_client_action(capture_store):
    candidates = [
        _point(
            f"candidate-{index}",
            image_sha256=f"{index + 1:02x}" * 32,
            image_relpath=f"nested/reorder-{index}.jpg",
        )
        for index in range(8)
    ]
    first = candidates[0]
    second = next(
        point
        for point in candidates[1:]
        if point["review_object_key"][4:6]
        != first["review_object_key"][4:6]
    )
    _register_job(capture_store, points=[first, second])
    common = {
        "capture_training_data": True,
        "action_type": "discard",
        "client_action_id": "reordered-batch-action",
    }
    original = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            **common,
            "point_ids": [first["point_id"], second["point_id"]],
        },
    )
    retry = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            **common,
            "point_ids": [second["point_id"], first["point_id"]],
        },
    )

    assert original["status"] == "recorded"
    assert retry["status"] == "already_recorded"
    assert set(retry["duplicate_action_ids"]) == set(original["action_ids"])
    assert len(_event_files(capture_store)) == 1
    flattened = list(api._class_analysis_iter_vignette_training_actions())
    assert len(flattened) == 2
    assert {event["point"]["identity"]["id"] for event in flattened} == {
        first["point_id"],
        second["point_id"],
    }


@pytest.mark.parametrize(
    "commit_case",
    ["unknown_action", "different_visual_object", "different_target_class"],
)
def test_invalid_commit_reference_cannot_promote_pending_label(
    capture_store,
    monkeypatch,
    commit_case,
):
    p0 = _point("p0")
    p1 = _point(
        "p1",
        image_sha256="22" * 32,
        image_relpath="nested/commit-other.jpg",
    )
    _register_job(capture_store, points=[p0, p1])
    _install_fake_crop_snapshot(monkeypatch)
    _trust_captured_class_for_export(monkeypatch)
    pending = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "change_class",
            "point_id": "p0",
            "before_class": "Boat",
            "after_class": "Building",
            "label_commit_status": "pending_desktop_sync",
            "client_action_id": f"pending-for-{commit_case}",
        },
    )
    pending_id = pending["action_ids"][0]
    commit_payload = {
        "capture_training_data": True,
        "action_type": "commit_class_change",
        "point_id": "p0",
        "before_class": "Boat",
        "after_class": "Building",
        "label_commit_status": "committed",
        "client_action_id": f"invalid-commit-{commit_case}",
        "commits_action_id": pending_id,
    }
    if commit_case == "unknown_action":
        commit_payload["commits_action_id"] = "client:does-not-exist"
    elif commit_case == "different_visual_object":
        commit_payload["point_id"] = "p1"
    else:
        commit_payload["after_class"] = "Person"

    with pytest.raises(api.HTTPException) as exc_info:
        api.record_class_analysis_vignette_training_action(
            "ca_vignette_capture",
            commit_payload,
        )

    assert exc_info.value.status_code == 409
    events = list(api._class_analysis_iter_vignette_training_actions())
    assert [event["action_id"] for event in events] == [pending_id]
    rows = api._class_analysis_vignette_training_export_rows(events)
    assert rows["classification"] == []
    assert rows["excluded"] == [
        {
            "action_id": pending_id,
            "reasons": ["annotation_not_committed"],
        }
    ]


def test_reader_never_observes_partial_append_as_corrupt(
    capture_store,
    monkeypatch,
):
    point = _point("p0")
    _register_job(capture_store, points=[point])
    api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "discard",
            "point_id": "p0",
            "client_action_id": "append-baseline",
        },
    )
    real_os = os
    halfway_written = threading.Event()
    allow_finish = threading.Event()
    split_once = threading.Event()

    class ControlledOs:
        def __getattr__(self, name):
            return getattr(real_os, name)

        @staticmethod
        def write(fd, data):
            if not split_once.is_set() and len(data) > 2:
                split_once.set()
                midpoint = len(data) // 2
                written = real_os.write(fd, data[:midpoint])
                halfway_written.set()
                allow_finish.wait(timeout=5)
                return written
            return real_os.write(fd, data)

    monkeypatch.setattr(api, "os", ControlledOs())
    writer_errors: list[BaseException] = []
    reader_errors: list[BaseException] = []
    reader_counts: list[int] = []

    def write_action():
        try:
            api.record_class_analysis_vignette_training_action(
                "ca_vignette_capture",
                {
                    "capture_training_data": True,
                    "action_type": "discard",
                    "point_id": "p0",
                    "client_action_id": "append-concurrent",
                },
            )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            writer_errors.append(exc)

    def read_actions():
        try:
            reader_counts.append(
                len(list(api._class_analysis_iter_vignette_training_actions()))
            )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            reader_errors.append(exc)

    writer = threading.Thread(target=write_action, daemon=True)
    writer.start()
    assert halfway_written.wait(timeout=5)
    reader = threading.Thread(target=read_actions, daemon=True)
    reader.start()
    # A locking reader may wait for the append. A snapshotting reader may
    # safely return only the prior complete line. Neither may report corruption.
    time.sleep(0.05)
    allow_finish.set()
    writer.join(timeout=5)
    reader.join(timeout=5)

    assert not writer.is_alive()
    assert not reader.is_alive()
    assert writer_errors == []
    assert reader_errors == []
    assert reader_counts and reader_counts[0] in {1, 2}
    assert (
        len(list(api._class_analysis_iter_vignette_training_actions()))
        == 2
    )


@pytest.mark.parametrize(
    "peer,host,origin",
    [
        ("127.0.0.1", "127.0.0.1:8000", ""),
        ("127.0.0.1", "localhost:8000", "http://localhost:8000"),
        ("::1", "[::1]:8000", "http://127.0.0.1:5173"),
    ],
)
def test_training_capture_authorizer_allows_loopback_cli_and_same_origin(
    monkeypatch,
    peer,
    host,
    origin,
):
    monkeypatch.delenv("TATOR_TRAINING_DATA_TOKEN", raising=False)
    monkeypatch.delenv(
        "TATOR_TRAINING_CAPTURE_TAILSCALE_USERS",
        raising=False,
    )
    monkeypatch.delenv(
        "TATOR_TRAINING_CAPTURE_TAILSCALE_CAPABILITY",
        raising=False,
    )

    authorization = api._class_analysis_authorize_training_capture_request(
        _training_request(peer=peer, host=host, origin=origin)
    )

    assert authorization == {
        "source": "local_loopback",
        "actor": "local_user",
    }


@pytest.mark.parametrize(
    "origin,fetch_site",
    [
        ("https://evil.example", ""),
        ("", "cross-site"),
        ("null", ""),
    ],
)
def test_training_capture_authorizer_denies_cross_site_browser_requests(
    monkeypatch,
    origin,
    fetch_site,
):
    monkeypatch.delenv("TATOR_TRAINING_DATA_TOKEN", raising=False)

    with pytest.raises(api.HTTPException) as exc_info:
        api._class_analysis_authorize_training_capture_request(
            _training_request(
                peer="127.0.0.1",
                host="localhost:8000",
                origin=origin,
                fetch_site=fetch_site,
            )
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "training_capture_forbidden"


def test_direct_tailnet_peer_cannot_spoof_tailscale_serve_identity(
    monkeypatch,
):
    monkeypatch.delenv("TATOR_TRAINING_DATA_TOKEN", raising=False)
    monkeypatch.setenv(
        "TATOR_TRAINING_CAPTURE_TAILSCALE_USERS",
        "owner@example.com",
    )
    monkeypatch.setenv(
        "TATOR_TRAINING_CAPTURE_TAILSCALE_CAPABILITY",
        "tator.training",
    )
    request = _training_request(
        peer="100.96.12.34",
        host="tator.example-tailnet.ts.net",
        origin="https://tator.example-tailnet.ts.net",
        fetch_site="same-origin",
        extra_headers={
            "tailscale-user-login": "owner@example.com",
            "tailscale-app-capabilities": json.dumps(
                {"tator.training": []}
            ),
        },
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api._class_analysis_authorize_training_capture_request(request)

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "training_capture_forbidden"


def test_loopback_tailscale_serve_requires_configured_login_and_same_origin(
    monkeypatch,
):
    monkeypatch.delenv("TATOR_TRAINING_DATA_TOKEN", raising=False)
    monkeypatch.setenv(
        "TATOR_TRAINING_CAPTURE_TAILSCALE_USERS",
        "owner@example.com",
    )
    host = "tator.example-tailnet.ts.net"
    allowed = api._class_analysis_authorize_training_capture_request(
        _training_request(
            peer="127.0.0.1",
            host=host,
            origin=f"https://{host}",
            fetch_site="same-origin",
            extra_headers={
                "tailscale-user-login": "owner@example.com",
            },
        )
    )
    assert allowed == {
        "source": "tailscale_identity",
        "actor": "owner@example.com",
    }

    with pytest.raises(api.HTTPException) as unconfigured_error:
        api._class_analysis_authorize_training_capture_request(
            _training_request(
                peer="127.0.0.1",
                host=host,
                origin=f"https://{host}",
                extra_headers={
                    "tailscale-user-login": "intruder@example.com",
                },
            )
        )
    assert unconfigured_error.value.status_code == 403

    with pytest.raises(api.HTTPException) as cross_origin_error:
        api._class_analysis_authorize_training_capture_request(
            _training_request(
                peer="127.0.0.1",
                host=host,
                origin="https://evil.example",
                extra_headers={
                    "tailscale-user-login": "owner@example.com",
                },
            )
        )
    assert cross_origin_error.value.status_code == 403


def test_loopback_tailscale_serve_accepts_configured_capability(
    monkeypatch,
):
    monkeypatch.delenv("TATOR_TRAINING_DATA_TOKEN", raising=False)
    monkeypatch.delenv(
        "TATOR_TRAINING_CAPTURE_TAILSCALE_USERS",
        raising=False,
    )
    monkeypatch.setenv(
        "TATOR_TRAINING_CAPTURE_TAILSCALE_CAPABILITY",
        "tator.training",
    )
    host = "tator.example-tailnet.ts.net"

    authorization = api._class_analysis_authorize_training_capture_request(
        _training_request(
            peer="::1",
            host=host,
            origin=f"https://{host}",
            fetch_site="same-origin",
            extra_headers={
                "tailscale-app-capabilities": json.dumps(
                    {"tator.training": [{"allow": True}]}
                ),
            },
        )
    )

    assert authorization == {
        "source": "tailscale_capability",
        "actor": "tailscale_capability",
    }


@pytest.mark.parametrize("origin", ["", "http://localhost:8000"])
def test_loopback_generic_training_route_invokes_backend_with_authorization(
    monkeypatch,
    origin,
):
    monkeypatch.delenv("TATOR_TRAINING_DATA_TOKEN", raising=False)
    calls: Dict[str, Any] = {}
    client = TestClient(
        _capture_security_app(calls),
        client=("127.0.0.1", 54321),
    )
    headers = {"host": "localhost:8000"}
    if origin:
        headers.update(
            {
                "origin": origin,
                "sec-fetch-site": "same-origin",
            }
        )

    response = client.post(
        "/class_analysis/jobs/job/training_actions",
        headers=headers,
        json={
            "capture_training_data": True,
            "action_type": "discard",
            "point_id": "p0",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "recorded"
    assert len(calls["training"]) == 1
    forwarded = calls["training"][0]["payload"]
    assert forwarded["capture_training_data"] is True
    assert forwarded["_training_authorization"] == {
        "source": "local_loopback",
        "actor": "local_user",
    }


@pytest.mark.parametrize(
    "method,path,counter",
    [
        (
            "post",
            "/class_analysis/jobs/job/training_actions",
            "training",
        ),
        ("get", "/class_analysis/training_actions/status", "status"),
        ("post", "/class_analysis/training_actions/export", "export"),
    ],
)
def test_denied_generic_training_routes_do_not_invoke_backend(
    monkeypatch,
    method,
    path,
    counter,
):
    monkeypatch.delenv("TATOR_TRAINING_DATA_TOKEN", raising=False)
    calls: Dict[str, Any] = {}
    client = TestClient(
        _capture_security_app(calls),
        client=("127.0.0.1", 54321),
    )
    request = getattr(client, method)
    kwargs: Dict[str, Any] = {
        "headers": {
            "host": "localhost:8000",
            "origin": "https://evil.example",
            "sec-fetch-site": "cross-site",
        },
    }
    if method == "post":
        kwargs["json"] = {
            "capture_training_data": True,
            "action_type": "discard",
            "point_id": "p0",
        }

    response = request(path, **kwargs)

    assert response.status_code == 403
    assert response.json() == {"detail": "training_capture_forbidden"}
    assert calls.get(counter) is None


def test_denied_disposition_capture_preserves_primary_action_and_forces_off(
    monkeypatch,
):
    monkeypatch.delenv("TATOR_TRAINING_DATA_TOKEN", raising=False)
    calls: Dict[str, Any] = {}
    client = TestClient(
        _capture_security_app(calls),
        client=("127.0.0.1", 54321),
    )

    response = client.post(
        "/class_analysis/jobs/job/points/p0/review_disposition",
        headers={
            "host": "localhost:8000",
            "origin": "https://evil.example",
            "sec-fetch-site": "cross-site",
        },
        json={
            "capture_training_data": True,
            "disposition": "confirm_current",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "recorded"
    assert payload["training_capture"] == {
        "status": "denied",
        "detail": "training_capture_forbidden",
    }
    assert len(calls["disposition"]) == 1
    forwarded = calls["disposition"][0]["payload"]
    assert forwarded["disposition"] == "confirm_current"
    assert forwarded["capture_training_data"] is False
    assert "_training_authorization" not in forwarded
    assert calls.get("training") is None


def test_default_cors_contract_allows_only_loopback_browser_origins(
    monkeypatch,
):
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)

    origins, origin_regex = api._configured_cors_origins()

    assert origins == []
    assert origin_regex
    assert re.fullmatch(origin_regex, "http://localhost:5173")
    assert re.fullmatch(origin_regex, "https://127.0.0.1:8443")
    assert re.fullmatch(origin_regex, "http://[::1]:3000")
    assert not re.fullmatch(origin_regex, "https://evil.example")
    assert not re.fullmatch(
        origin_regex,
        "https://tator.example-tailnet.ts.net",
    )


def test_linked_current_label_is_attested_and_exports(
    capture_store,
    monkeypatch,
):
    point, expected_target = _install_linked_annotation_state(
        capture_store,
        monkeypatch,
        current_class="Building",
    )
    _register_job(capture_store, points=[point])
    _install_fake_crop_snapshot(monkeypatch)
    pending = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "change_class",
            "point_id": "p0",
            "before_class": "Boat",
            "after_class": "Building",
            "label_commit_status": "pending_desktop_sync",
            "client_action_id": "linked-attestation-pending",
        },
    )
    pending_id = pending["action_ids"][0]
    # Commit queues outlive disposable analysis jobs. Model the normal backend
    # restart cleanup by removing the live job before the label commit marker.
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS.pop("ca_vignette_capture", None)
    commit = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "commit_class_change",
            "point_id": "p0",
            "before_class": "Boat",
            "after_class": "Building",
            "label_commit_status": "committed",
            "commits_action_id": pending_id,
            "client_action_id": "linked-attestation-commit",
        },
    )
    retry = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "commit_class_change",
            "point_id": "p0",
            "before_class": "Boat",
            "after_class": "Building",
            "label_commit_status": "committed",
            "commits_action_id": pending_id,
            "client_action_id": "linked-attestation-commit",
        },
    )
    assert retry["status"] == "already_recorded"
    assert retry["duplicate_action_ids"] == commit["action_ids"]
    events = {
        event["action_id"]: event
        for event in api._class_analysis_iter_vignette_training_actions()
    }
    source_event = events[pending_id]
    commit_event = events[commit["action_ids"][0]]

    _assert_commit_attestation(
        commit_event,
        source_event,
        expected_target=expected_target,
        expected_method="current_annotation_geometry_iou_v1",
    )
    rows = api._class_analysis_vignette_training_export_rows(
        list(events.values())
    )
    assert {row["action_id"] for row in rows["classification"]} == {
        pending_id
    }


def test_linked_current_class_mismatch_rejects_commit(
    capture_store,
    monkeypatch,
):
    point, _target = _install_linked_annotation_state(
        capture_store,
        monkeypatch,
        current_class="Person",
    )
    _register_job(capture_store, points=[point])
    _install_fake_crop_snapshot(monkeypatch)
    pending = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "change_class",
            "point_id": "p0",
            "before_class": "Boat",
            "after_class": "Building",
            "label_commit_status": "pending_desktop_sync",
            "client_action_id": "linked-mismatch-pending",
        },
    )
    pending_id = pending["action_ids"][0]

    with pytest.raises(api.HTTPException) as exc_info:
        api.record_class_analysis_vignette_training_action(
            "ca_vignette_capture",
            {
                "capture_training_data": True,
                "action_type": "commit_class_change",
                "point_id": "p0",
                "before_class": "Boat",
                "after_class": "Building",
                "label_commit_status": "committed",
                "commits_action_id": pending_id,
                "client_action_id": "linked-mismatch-commit",
            },
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "training_commit_annotation_unverified"
    events = list(api._class_analysis_iter_vignette_training_actions())
    assert [event["action_id"] for event in events] == [pending_id]
    assert api._class_analysis_vignette_training_export_rows(events)[
        "classification"
    ] == []


@pytest.mark.parametrize("source_mode", ["active_workspace"])
def test_nonlinked_commit_without_explicit_annotation_target_is_rejected(
    capture_store,
    monkeypatch,
    source_mode,
):
    source_id = (
        "workspace-snapshot"
        if source_mode == "active_workspace"
        else "transient-source"
    )
    source_key = (
        "active:snapshot-1"
        if source_mode == "active_workspace"
        else "transient:source-session"
    )
    point = _point(
        "p0",
        source_mode=source_mode,
        source_id=source_id,
        source_key=source_key,
    )
    _register_job(capture_store, points=[point])
    _install_fake_crop_snapshot(monkeypatch)
    pending = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "change_class",
            "point_id": "p0",
            "before_class": "Boat",
            "after_class": "Building",
            "label_commit_status": "pending_desktop_sync",
            "client_action_id": f"{source_mode}-missing-target-pending",
        },
    )
    pending_id = pending["action_ids"][0]

    with pytest.raises(api.HTTPException) as exc_info:
        api.record_class_analysis_vignette_training_action(
            "ca_vignette_capture",
            {
                "capture_training_data": True,
                "action_type": "commit_class_change",
                "point_id": "p0",
                "before_class": "Boat",
                "after_class": "Building",
                "label_commit_status": "committed",
                "commits_action_id": pending_id,
                "client_action_id": f"{source_mode}-missing-target-commit",
            },
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "training_commit_annotation_target_required"
    events = list(api._class_analysis_iter_vignette_training_actions())
    assert [event["action_id"] for event in events] == [pending_id]
    assert api._class_analysis_vignette_training_export_rows(events)[
        "classification"
    ] == []


@pytest.mark.parametrize("source_mode", ["active_workspace"])
def test_matching_explicit_annotation_target_is_attested(
    capture_store,
    monkeypatch,
    source_mode,
):
    linked_point, annotation_target = _install_linked_annotation_state(
        capture_store,
        monkeypatch,
        current_class="Building",
        source_id="committed-target",
    )
    source_id = (
        "workspace-snapshot"
        if source_mode == "active_workspace"
        else "transient-source"
    )
    source_key = (
        "active:snapshot-1"
        if source_mode == "active_workspace"
        else "transient:source-session"
    )
    point = _point(
        "p0",
        image_sha256=linked_point["image_sha256"],
        image_relpath=linked_point["image_relpath"],
        bbox_xyxy=linked_point["bbox_xyxy"],
        source_mode=source_mode,
        source_id=source_id,
        source_key=source_key,
    )
    point["image_width"] = 512
    point["image_height"] = 384
    _register_job(capture_store, points=[point])
    _install_fake_crop_snapshot(monkeypatch)
    pending = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "change_class",
            "point_id": "p0",
            "before_class": "Boat",
            "after_class": "Building",
            "label_commit_status": "pending_desktop_sync",
            "annotation_target": annotation_target,
            "client_action_id": f"{source_mode}-explicit-target-pending",
        },
    )
    pending_id = pending["action_ids"][0]
    commit = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "commit_class_change",
            "point_id": "p0",
            "before_class": "Boat",
            "after_class": "Building",
            "label_commit_status": "committed",
            "commits_action_id": pending_id,
            "annotation_target": annotation_target,
            "client_action_id": f"{source_mode}-explicit-target-commit",
        },
    )
    events = {
        event["action_id"]: event
        for event in api._class_analysis_iter_vignette_training_actions()
    }
    source_event = events[pending_id]
    commit_event = events[commit["action_ids"][0]]

    _assert_commit_attestation(
        commit_event,
        source_event,
        expected_target=annotation_target,
        expected_method="current_annotation_geometry_iou_v1",
    )
    rows = api._class_analysis_vignette_training_export_rows(
        list(events.values())
    )
    assert {row["action_id"] for row in rows["classification"]} == {
        pending_id
    }


def test_transient_commit_remains_exportable_after_session_expiry(
    capture_store,
    monkeypatch,
):
    point = _point(
        "p0",
        source_mode="transient",
        source_id="transient-source",
        source_key="transient:source-session",
    )
    _register_job(capture_store, points=[point])
    _install_fake_crop_snapshot(monkeypatch)
    monkeypatch.setattr(
        api,
        "_class_analysis_vignette_current_annotation_target_class",
        lambda event, _target, _cache: str(
            event.get("after_class") or ""
        ),
    )
    pending = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "change_class",
            "point_id": "p0",
            "before_class": "Boat",
            "after_class": "Building",
            "label_commit_status": "pending_desktop_sync",
            "client_action_id": "transient-durable-pending",
        },
    )
    pending_id = pending["action_ids"][0]
    commit = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "commit_class_change",
            "point_id": "p0",
            "before_class": "Boat",
            "after_class": "Building",
            "label_commit_status": "committed",
            "commits_action_id": pending_id,
            "client_action_id": "transient-durable-commit",
        },
    )
    events = list(
        api._class_analysis_iter_vignette_training_actions()
    )
    commit_event = next(
        event
        for event in events
        if event["action_id"] == commit["action_ids"][0]
    )
    assert commit_event["annotation_target"] == {
        "source_mode": "transient",
        "source_id": "transient-source",
        "split": "train",
        "image_relpath": "nested/frame.jpg",
    }

    monkeypatch.setattr(
        api,
        "_class_analysis_vignette_current_annotation_target_class",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_vignette_transient_target_unavailable",
        lambda _target: True,
    )
    rows = api._class_analysis_vignette_training_export_rows(events)

    assert {row["action_id"] for row in rows["classification"]} == {
        pending_id
    }


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_attestation",
        "tampered_attestation",
        "current_annotation_changed",
    ],
)
def test_invalid_commit_attestation_cannot_promote_at_export(
    capture_store,
    monkeypatch,
    tamper,
):
    point, annotation_target = _install_linked_annotation_state(
        capture_store,
        monkeypatch,
        current_class="Building",
    )
    _register_job(capture_store, points=[point])
    _install_fake_crop_snapshot(monkeypatch)
    pending = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "change_class",
            "point_id": "p0",
            "before_class": "Boat",
            "after_class": "Building",
            "label_commit_status": "pending_desktop_sync",
            "client_action_id": f"attestation-export-{tamper}-pending",
        },
    )
    pending_id = pending["action_ids"][0]
    commit = api.record_class_analysis_vignette_training_action(
        "ca_vignette_capture",
        {
            "capture_training_data": True,
            "action_type": "commit_class_change",
            "point_id": "p0",
            "before_class": "Boat",
            "after_class": "Building",
            "label_commit_status": "committed",
            "commits_action_id": pending_id,
            "annotation_target": annotation_target,
            "client_action_id": f"attestation-export-{tamper}-commit",
        },
    )
    commit_id = commit["action_ids"][0]
    events = {
        event["action_id"]: event
        for event in api._class_analysis_iter_vignette_training_actions()
    }
    if tamper == "missing_attestation":
        altered_commit = json.loads(json.dumps(events[commit_id]))
        altered_commit.pop("annotation_commit_attestation", None)
        events[commit_id] = _rehash_training_event(altered_commit)
    elif tamper == "tampered_attestation":
        altered_commit = json.loads(json.dumps(events[commit_id]))
        altered_commit["annotation_commit_attestation"][
            "committed_class"
        ] = "Person"
        events[commit_id] = _rehash_training_event(altered_commit)
    else:
        person_line = (
            f"2 {30 / 512:.10f} {50 / 384:.10f} "
            f"{40 / 512:.10f} {60 / 384:.10f}"
        )
        monkeypatch.setattr(
            api,
            "_annotation_effective_label_lines",
            lambda *_args, **_kwargs: [person_line],
        )

    rows = api._class_analysis_vignette_training_export_rows(
        list(events.values())
    )

    assert rows["classification"] == []
    excluded = {
        row["action_id"]: set(row["reasons"])
        for row in rows["excluded"]
    }
    assert "annotation_not_committed" in excluded[pending_id]
    assert "commit_attestation_invalid" in excluded[commit_id]
