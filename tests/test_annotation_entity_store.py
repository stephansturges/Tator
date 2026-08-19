from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from services.annotation_entity_store import AnnotationEntityStore


def test_operation_journal_is_monotonic_scoped_and_idempotent(tmp_path: Path) -> None:
    store = AnnotationEntityStore(tmp_path / "entities.sqlite3")
    request = {"client_action_id": "annotation:test", "records": [{"x": 1}]}
    request_hash = store.request_hash(request)

    prepared = store.begin_operation(
        operation_id="annotation:test",
        request_hash=request_hash,
        request=request,
        job_id="job-a",
        source_identity="source-a",
        response={"plan": {"records": []}},
    )
    assert prepared["state"] == "prepared"
    assert store.get_receipt("annotation:test", request_hash) is None
    assert [item["operation_id"] for item in store.list_operations(job_id="job-a")] == [
        "annotation:test"
    ]
    assert store.list_operations(job_id="job-b") == []

    store.advance_operation("annotation:test", "annotation_committed")
    store.advance_operation("annotation:test", "entity_committed")
    store.advance_operation("annotation:test", "review_committed")
    complete = store.advance_operation(
        "annotation:test", "complete", response={"status": "complete"}
    )
    assert complete["state"] == "complete"
    assert store.get_receipt("annotation:test", request_hash) == {"status": "complete"}
    assert store.list_operations(job_id="job-a") == []
    replay = store.begin_operation(
        operation_id="annotation:test",
        request_hash=request_hash,
        request=request,
        job_id="job-a",
        source_identity="source-a",
    )
    assert replay["state"] == "complete"
    with pytest.raises(ValueError, match="reused"):
        store.begin_operation(
            operation_id="annotation:test",
            request_hash=store.request_hash({"different": True}),
            request={"different": True},
            job_id="job-a",
            source_identity="source-a",
        )


def test_existing_receipt_schema_migrates_as_complete(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE annotation_entity_transactions ("
            "operation_id TEXT PRIMARY KEY, request_hash TEXT NOT NULL, "
            "response_json TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        connection.execute(
            "INSERT INTO annotation_entity_transactions VALUES (?, ?, ?, ?)",
            ("legacy-op", "hash", json.dumps({"legacy": True}), 1.0),
        )

    store = AnnotationEntityStore(path)

    assert store.get_receipt("legacy-op", "hash") == {"legacy": True}
    assert store.get_operation("legacy-op")["state"] == "complete"


def test_exact_legacy_point_binding_is_persistent(tmp_path: Path) -> None:
    store = AnnotationEntityStore(tmp_path / "entities.sqlite3")
    store.put_point_binding(
        job_id="job-a",
        point_id="point-a",
        source_identity="source-a",
        image_key="train:image.jpg",
        annotation_entity_id="ae1-a",
        entity_revision=3,
        record_revision="alr1-a",
        status="legacy_exact",
        attestation="proof",
    )

    binding = store.get_point_binding("job-a", "point-a")

    assert binding is not None
    assert binding["annotation_entity_id"] == "ae1-a"
    assert binding["entity_revision"] == 3
    assert store.get_point_binding("job-b", "point-a") is None


def test_recovery_required_cannot_regress_to_annotation_commit(tmp_path: Path) -> None:
    store = AnnotationEntityStore(tmp_path / "entities.sqlite3")
    request = {"job_id": "job-a", "records": []}
    operation_id = "operation-recovery"
    store.begin_operation(
        operation_id=operation_id,
        request_hash=store.request_hash(request),
        request=request,
        job_id="job-a",
        source_identity="source-a",
    )
    store.advance_operation(operation_id, "recovery_required")

    with pytest.raises(ValueError, match="state_regression"):
        store.advance_operation(operation_id, "annotation_committed")


def test_job_source_bindings_scope_conflicts_without_an_operation(tmp_path: Path) -> None:
    store = AnnotationEntityStore(tmp_path / "entities.sqlite3")
    store.bind_job_sources("job-a", ["source-b", "source-a", "source-a"])
    store.mark_conflict("source-a", "train:image.jpg", "duplicate exact bbox")

    assert store.list_job_sources("job-a") == ["source-a", "source-b"]
    assert store.list_job_sources("job-b") == []
    assert store.list_conflicts(store.list_job_sources("job-a")[0])[0][
        "image_key"
    ] == "train:image.jpg"


def test_operation_journal_strips_transport_credentials_and_rehashes_semantics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "entities.sqlite3"
    store = AnnotationEntityStore(path)
    request = {
        "client_action_id": "annotation:credential-test",
        "records": [{"x": 1}],
        "annotation_target": {"source_mode": "transient", "session_id": "source-session"},
        "annotation_save": {
            "session_id": "editor",
            "lock_session_id": "lease",
            "lock_token": "secret",
        },
        "_training_authorization": {"token": "training-secret"},
    }
    semantic = {
        "client_action_id": "annotation:credential-test",
        "records": [{"x": 1}],
        "annotation_target": {"source_mode": "transient", "session_id": "source-session"},
    }
    store.begin_operation(
        operation_id="annotation:credential-test",
        request_hash=store.request_hash(request),
        request=request,
        job_id="job-a",
        source_identity="source-a",
    )

    operation = AnnotationEntityStore(path).get_operation(
        "annotation:credential-test",
        request_hash=store.request_hash(semantic),
    )

    assert operation is not None
    assert operation["request"] == semantic
    serialized = json.dumps(operation["request"])
    assert "secret" not in serialized
    assert '"session_id": "source-session"' in serialized
    assert "lease" not in serialized


def test_terminal_conflict_remains_authoritative_after_record_rewrite(
    tmp_path: Path,
) -> None:
    store = AnnotationEntityStore(tmp_path / "entities.sqlite3")
    request = {"client_action_id": "annotation:conflict", "records": []}
    store.begin_operation(
        operation_id="annotation:conflict",
        request_hash=store.request_hash(request),
        request=request,
        job_id="job-a",
        source_identity="source-a",
        response={"annotation_snapshot": {"status": "saved"}, "committed": True},
    )
    store.advance_operation("annotation:conflict", "annotation_committed")
    store.advance_operation("annotation:conflict", "entity_committed")
    store.mark_operation_conflict(
        operation_id="annotation:conflict",
        source_identity="source-a",
        image_key="train:image.jpg",
        detail={
            "code": "review_revision_conflict",
            "message": "Review state changed after labels committed.",
            "rerun_required": True,
            "mutation_committed": True,
        },
    )
    store.put_record("source-a", "train:image.jpg", {"record_revision": "new"})

    assert store.list_conflicts("source-a") == []
    operation = store.list_operations(job_id="job-a", include_terminal=True)[0]
    assert operation["state"] == "conflict"
    assert operation["response"]["rerun_required"] is True
    assert operation["error"]["detail"]["mutation_committed"] is True


def test_source_generation_and_batch_manifest_are_atomic(tmp_path: Path) -> None:
    store = AnnotationEntityStore(tmp_path / "entities.sqlite3")
    binding = store.claim_source_job(
        "transient",
        "source-a",
        "job-a",
        source_descriptor={"kind": "transient", "id": "source-a"},
    )
    generation = int(binding["binding_generation"])
    items = [
        {
            "sequence": 0,
            "point_id": "point-a",
            "payload": {
                "split": "train",
                "image_relpath": "one.jpg",
                "actions": [{"point_id": "point-a", "action": "delete"}],
            },
        },
        {
            "sequence": 1,
            "point_id": "point-b",
            "payload": {
                "split": "train",
                "image_relpath": "two.jpg",
                "actions": [{"point_id": "point-b", "action": "delete"}],
            },
        },
    ]
    store.create_batch(
        batch_id="batch-a",
        job_id="job-a",
        source_mode="transient",
        source_id="source-a",
        binding_generation=generation,
        action="delete",
        target_class_name="",
        declared_count=len(items),
        manifest_hash=store.request_hash(items),
    )
    store.append_batch_items("batch-a", items[:1])
    store.append_batch_items("batch-a", items[1:])
    started = store.start_batch("batch-a")

    assert started["state"] == "ready"
    assert [item["payload"] for item in store.get_batch_items("batch-a")] == [
        item["payload"] for item in items
    ]
    with pytest.raises(ValueError, match="annotation_source_has_uncheckpointed_analysis"):
        store.claim_source_job("transient", "source-a", "job-b")


def test_annotation_source_job_binding_is_durable_and_replaceable(tmp_path: Path) -> None:
    store = AnnotationEntityStore(tmp_path / "entities.sqlite3")

    store.bind_source_job("linked", "dataset-a", "job-a")
    assert store.get_source_job_binding("linked", "dataset-a")["job_id"] == "job-a"

    store.bind_source_job("linked", "dataset-a", "job-b")
    assert store.get_source_job_binding("linked", "dataset-a")["job_id"] == "job-b"
    assert store.get_source_job_binding("linked", "dataset-b") is None
