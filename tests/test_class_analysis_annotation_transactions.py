from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
import hashlib
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import localinferenceapi as api
from services.annotation_entities import (
    apply_annotation_entity_actions,
    live_label_lines,
    migrate_legacy_annotation_record,
)
from services.annotation_entity_store import AnnotationEntityStore


JOB_ID = "job-annotation-transaction"
SOURCE_ID = "transient-source"
SOURCE_IDENTITY = "asi1_test_annotation_source"
LABELMAP = ["First", "Second", "Third"]


def _attestation(image_key: str, entity: dict, record_revision: str) -> str:
    return hashlib.sha256(
        "\0".join(
            [
                SOURCE_IDENTITY,
                image_key,
                str(entity["annotation_entity_id"]),
                str(entity["entity_revision"]),
                record_revision,
            ]
        ).encode("utf-8")
    ).hexdigest()


def _refresh_manifest_entity_revisions(manifest, store):
    for row in manifest["images"]:
        image_key = f"{row['split']}:{row['image_relpath']}"
        record = store.get_record(SOURCE_IDENTITY, image_key)
        if record is not None and live_label_lines(record) == row["label_lines"]:
            row["annotation_entity_record_revision"] = record["record_revision"]
    return manifest


def _install_transaction_harness(monkeypatch, tmp_path, image_lines):
    store = AnnotationEntityStore(tmp_path / "annotation-entities.sqlite3")
    manifest = {
        "labelmap": list(LABELMAP),
        "session_revision": 7,
        "images": [],
    }
    points = {}
    records = []
    save_calls = []
    review_calls = []
    state = {"save_hook": None, "review_hook": None}

    for image_index, (relpath, lines) in enumerate(image_lines.items()):
        image_key = f"train:{relpath}"
        record = migrate_legacy_annotation_record(
            source_identity=SOURCE_IDENTITY,
            source_lines=lines,
            current_lines=lines,
        )
        store.put_record(SOURCE_IDENTITY, image_key, record)
        row = {
            "split": "train",
            "image_relpath": relpath,
            "label_lines": list(lines),
            "annotation_record_revision": f"labels-{image_index}",
            "annotation_source_identity": SOURCE_IDENTITY,
            "annotation_entity_record_revision": record["record_revision"],
        }
        manifest["images"].append(row)
        actions = []
        for line_index, (line, entity) in enumerate(zip(lines, record["entities"])):
            point_id = f"point-{image_index}-{line_index}"
            points[point_id] = {
                "point_id": point_id,
                "review_object_key": f"review-{point_id}",
                "split": "train",
                "image_relpath": relpath,
                "label_line_index": line_index,
                "label_line": line,
                "class_name": LABELMAP[int(line.split()[0])],
                "annotation_entity_id": entity["annotation_entity_id"],
                "annotation_entity_revision": entity["entity_revision"],
                "annotation_entity_record_revision": record["record_revision"],
                "annotation_source_identity": SOURCE_IDENTITY,
                "source_record_key": image_key,
                "identity_status": "ready",
                "annotation_attestation": _attestation(
                    image_key, entity, record["record_revision"]
                ),
            }
            actions.append(
                {
                    "point_id": point_id,
                    "annotation_entity_id": entity["annotation_entity_id"],
                    "expected_entity_revision": entity["entity_revision"],
                    "action": "delete",
                }
            )
        records.append(
            {
                "split": "train",
                "image_relpath": relpath,
                "annotation_source_identity": SOURCE_IDENTITY,
                "expected_annotation_record_revision": row[
                    "annotation_record_revision"
                ],
                "expected_record_revision": record["record_revision"],
                "actions": actions,
            }
        )

    def apply_saved_records(payload):
        saved_records = []
        for saved in payload.get("records") or []:
            row = next(
                item
                for item in manifest["images"]
                if item["split"] == saved["split"]
                and item["image_relpath"] == saved["image_relpath"]
            )
            row["label_lines"] = list(saved.get("label_lines") or [])
            row["annotation_record_revision"] = (
                f"labels-saved-{manifest['session_revision'] + 1}"
            )
            saved_records.append(
                {
                    "split": saved["split"],
                    "image_relpath": saved["image_relpath"],
                    "annotation_source_identity": SOURCE_IDENTITY,
                    "annotation_record_revision": row[
                        "annotation_record_revision"
                    ],
                }
            )
        manifest["session_revision"] += 1
        return {
            "status": "saved",
            "session_revision": manifest["session_revision"],
            "records": saved_records,
        }

    def save_snapshot(source_kind, source_id, payload):
        assert source_kind == "transient"
        assert source_id == SOURCE_ID
        save_calls.append(deepcopy(payload))
        hook = state["save_hook"]
        if hook is not None:
            return hook(payload, apply_saved_records)
        return apply_saved_records(payload)

    def record_review(**kwargs):
        review_calls.append(dict(kwargs))
        hook = state["review_hook"]
        if hook is not None:
            return hook(kwargs)
        return {
            "status": "recorded",
            "point_id": kwargs["point_id"],
            "client_action_id": kwargs["client_action_id"],
        }

    job = SimpleNamespace(job_id=JOB_ID, summary={"labelmap": list(LABELMAP)})

    def targeted_snapshot(source_kind, source_id, requested_records):
        assert source_kind == "transient"
        assert source_id == SOURCE_ID
        refreshed = _refresh_manifest_entity_revisions(manifest, store)
        requested_keys = {
            api._class_analysis_annotation_image_key(dict(record))
            for record in requested_records
            if isinstance(record, dict)
        }
        return {
            "labelmap": list(refreshed["labelmap"]),
            "session_revision": int(refreshed["session_revision"]),
            "rows_by_key": {
                f"{row['split']}:{row['image_relpath']}": deepcopy(row)
                for row in refreshed["images"]
                if f"{row['split']}:{row['image_relpath']}" in requested_keys
            },
        }

    monkeypatch.setattr(api, "_class_analysis_annotation_entity_store", lambda: store)
    monkeypatch.setattr(api, "_get_class_analysis_job", lambda job_id: job)
    monkeypatch.setattr(
        api,
        "_class_analysis_annotation_transaction_snapshot",
        targeted_snapshot,
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_annotation_transaction_manifest",
        lambda *_args, **_kwargs: pytest.fail("transaction scanned the full manifest"),
    )
    monkeypatch.setattr(api, "_class_analysis_annotation_transaction_save", save_snapshot)
    monkeypatch.setattr(
        api,
        "_class_analysis_review_point_for_mutation",
        lambda job_id, point_id: (job, points[point_id]),
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_record_entity_transaction_review",
        record_review,
    )
    payload = {
        "client_action_id": "operation-1",
        "annotation_source": {"kind": "transient", "id": SOURCE_ID},
        "annotation_save": {
            "lock_session_id": "lease-old",
            "session_id": "lease-old",
            "lock_token": "token-old",
            "expected_session_revision": 7,
        },
        "records": records,
    }
    return SimpleNamespace(
        store=store,
        manifest=manifest,
        points=points,
        payload=payload,
        save_calls=save_calls,
        review_calls=review_calls,
        state=state,
    )


@pytest.mark.parametrize("source_kind", ["transient", "dataset"])
def test_annotation_transaction_snapshot_captures_one_locked_live_row_per_image(
    monkeypatch, source_kind
) -> None:
    calls = []

    def record(_source_kind, _source_id, split, image_relpath):
        calls.append((split, image_relpath))
        return {
            "split": split,
            "image_relpath": image_relpath,
            "label_lines": ["0 0.5 0.5 0.2 0.3"],
        }

    monkeypatch.setattr(
        api, "_class_analysis_annotation_transaction_record", record
    )
    if source_kind == "transient":
        monkeypatch.setattr(
            api,
            "_resolve_transient_session",
            lambda _source_id: {"classes": ["One", "Two"], "_state_revision": 17},
        )
    else:
        entry = {"classes": ["One", "Two"]}
        monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _source_id: entry)
        monkeypatch.setattr(
            api, "_dataset_annotation_mutation_lock", lambda _entry: nullcontext()
        )
        monkeypatch.setattr(
            api,
            "_annotation_load_or_create_meta",
            lambda _entry: (None, {"annotation_revision": 17}),
        )

    snapshot = api._class_analysis_annotation_transaction_snapshot(
        source_kind,
        "source",
        [
            {"split": "train", "image_relpath": "one.jpg"},
            {"split": "train", "image_relpath": "one.jpg"},
            {"split": "val", "image_relpath": "two.jpg"},
        ],
    )

    assert snapshot["labelmap"] == ["One", "Two"]
    assert snapshot["session_revision"] == 17
    assert set(snapshot["rows_by_key"]) == {"train:one.jpg", "val:two.jpg"}
    assert calls == [("train", "one.jpg"), ("val", "two.jpg")]


def test_annotation_transaction_conflict_detail_isolates_exact_points() -> None:
    detail = api._class_analysis_annotation_transaction_conflict_detail(
        [
            {"image_key": "train:one.jpg"},
            {"image_key": "train:two.jpg"},
        ],
        ["conflict", "before"],
        [
            {
                "point_id": "point-one",
                "split": "train",
                "image_relpath": "one.jpg",
            },
            {
                "point_id": "point-two",
                "split": "train",
                "image_relpath": "two.jpg",
            },
        ],
    )

    assert detail["code"] == "annotation_transaction_snapshot_state_conflict"
    assert detail["image_keys"] == ["train:one.jpg"]
    assert detail["point_ids"] == ["point-one"]


def test_transaction_commits_once_and_replays_with_refreshed_transport_credentials(
    monkeypatch, tmp_path
) -> None:
    harness = _install_transaction_harness(
        monkeypatch,
        tmp_path,
        {"one.jpg": ["0 0.5 0.5 0.2 0.3"]},
    )

    first = api.commit_class_analysis_annotation_transaction(
        JOB_ID, deepcopy(harness.payload)
    )
    replay_payload = deepcopy(harness.payload)
    replay_payload["annotation_save"]["lock_token"] = "token-new"
    replay = api.commit_class_analysis_annotation_transaction(JOB_ID, replay_payload)

    assert first["status"] == "complete"
    assert replay["status"] == "complete"
    assert "plan" not in first
    assert len(harness.save_calls) == 1
    assert len(harness.review_calls) == 1
    assert harness.manifest["images"][0]["label_lines"] == []
    stored = harness.store.get_record(SOURCE_IDENTITY, "train:one.jpg")
    assert stored is not None
    assert live_label_lines(stored) == []


def test_resumable_batch_processes_bounded_manifest(monkeypatch, tmp_path) -> None:
    harness = _install_transaction_harness(
        monkeypatch,
        tmp_path,
        {
            "one.jpg": ["0 0.5 0.5 0.2 0.3"],
            "two.jpg": ["1 0.4 0.4 0.1 0.2"],
        },
    )
    items = [
        {
            "sequence": index,
            "point_id": str(record["actions"][0]["point_id"]),
            "payload": deepcopy(record),
        }
        for index, record in enumerate(harness.payload["records"])
    ]
    manifest_hash = harness.store.request_hash(items)
    created = api.create_class_analysis_annotation_batch(
        JOB_ID,
        {
            "batch_id": "batch-1",
            "annotation_source": {"kind": "transient", "id": SOURCE_ID},
            "action": "delete",
            "declared_count": len(items),
            "manifest_hash": manifest_hash,
        },
    )
    assert created["state"] == "draft"
    api.append_class_analysis_annotation_batch_items(
        JOB_ID, "batch-1", {"items": items}
    )
    api.start_class_analysis_annotation_batch(JOB_ID, "batch-1")
    completed = api.process_class_analysis_annotation_batch(
        JOB_ID,
        "batch-1",
        {"annotation_save": deepcopy(harness.payload["annotation_save"])},
    )
    results = api.get_class_analysis_annotation_batch_results(
        JOB_ID, "batch-1", -1, 500
    )

    assert completed["state"] == "complete"
    assert completed["completed_count"] == 2
    assert completed["settled_count"] == 2
    assert completed["succeeded_count"] == 2
    assert completed["conflict_count"] == 0
    assert [item["state"] for item in results["items"]] == ["complete", "complete"]
    assert len(harness.save_calls) == 1
    assert len(harness.review_calls) == 2


def test_batch_isolates_one_structured_conflict_and_commits_the_other_record(
    monkeypatch, tmp_path
) -> None:
    harness = _install_transaction_harness(
        monkeypatch,
        tmp_path,
        {
            "one.jpg": ["0 0.5 0.5 0.2 0.3"],
            "two.jpg": ["1 0.4 0.4 0.1 0.2"],
        },
    )
    items = [
        {
            "sequence": index,
            "point_id": str(record["actions"][0]["point_id"]),
            "payload": deepcopy(record),
        }
        for index, record in enumerate(harness.payload["records"])
    ]
    manifest_hash = harness.store.request_hash(items)
    api.create_class_analysis_annotation_batch(
        JOB_ID,
        {
            "batch_id": "batch-conflict",
            "annotation_source": {"kind": "transient", "id": SOURCE_ID},
            "action": "delete",
            "declared_count": len(items),
            "manifest_hash": manifest_hash,
        },
    )
    api.append_class_analysis_annotation_batch_items(
        JOB_ID, "batch-conflict", {"items": items}
    )
    api.start_class_analysis_annotation_batch(JOB_ID, "batch-conflict")
    original_commit = api.commit_class_analysis_annotation_transaction
    blocked = {"raised": False}

    def conflict_once(job_id, payload):
        if not blocked["raised"]:
            blocked["raised"] = True
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "annotation_entity_sidecar_out_of_sync",
                    "message": "The first row changed.",
                    "image_keys": ["train:one.jpg"],
                    "point_ids": ["point-0-0"],
                },
            )
        return original_commit(job_id, payload)

    monkeypatch.setattr(
        api,
        "commit_class_analysis_annotation_transaction",
        conflict_once,
    )
    first = api.process_class_analysis_annotation_batch(
        JOB_ID,
        "batch-conflict",
        {"annotation_save": deepcopy(harness.payload["annotation_save"])},
    )
    final = api.process_class_analysis_annotation_batch(
        JOB_ID,
        "batch-conflict",
        {"annotation_save": deepcopy(harness.payload["annotation_save"])},
    )
    results = api.get_class_analysis_annotation_batch_results(
        JOB_ID, "batch-conflict", -1, 500
    )

    assert first["conflict_count"] == 1
    assert first["succeeded_count"] == 0
    assert final["state"] == "partial"
    assert final["settled_count"] == 2
    assert final["succeeded_count"] == 1
    assert final["conflict_count"] == 1
    assert [item["state"] for item in results["items"]] == [
        "conflict",
        "complete",
    ]
    assert results["items"][0]["error"]["image_keys"] == ["train:one.jpg"]
    assert len(harness.save_calls) == 1


def test_relabel_never_falls_back_to_the_frozen_analysis_labelmap(
    monkeypatch, tmp_path
) -> None:
    harness = _install_transaction_harness(
        monkeypatch,
        tmp_path,
        {"one.jpg": ["0 0.5 0.5 0.2 0.3"]},
    )
    harness.manifest["labelmap"] = []
    payload = deepcopy(harness.payload)
    payload["client_action_id"] = "operation-missing-source-labelmap"
    payload["records"][0]["actions"][0].update(
        {"action": "relabel", "target_class_id": 1}
    )

    with pytest.raises(HTTPException) as exc_info:
        api.commit_class_analysis_annotation_transaction(JOB_ID, payload)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == "annotation_source_labelmap_unavailable"
    assert harness.save_calls == []


def test_recovery_resumes_only_unsaved_records_with_a_fresh_lock(monkeypatch, tmp_path) -> None:
    harness = _install_transaction_harness(
        monkeypatch,
        tmp_path,
        {
            "one.jpg": ["0 0.5 0.5 0.2 0.3"],
            "two.jpg": ["1 0.4 0.4 0.1 0.1"],
        },
    )

    def interrupt_after_first_record(payload, apply_saved_records):
        partial = dict(payload)
        partial["records"] = [payload["records"][0]]
        apply_saved_records(partial)
        harness.state["save_hook"] = None
        raise RuntimeError("simulated interruption after first record")

    harness.state["save_hook"] = interrupt_after_first_record
    with pytest.raises(RuntimeError, match="simulated interruption"):
        api.commit_class_analysis_annotation_transaction(
            JOB_ID, deepcopy(harness.payload)
        )

    operation = harness.store.get_operation("operation-1")
    assert operation is not None
    assert operation["state"] == "prepared"
    recovery = api.recover_class_analysis_annotation_session(
        JOB_ID,
        {
            "mode": "retry_transactions",
            "operation_ids": ["operation-1"],
            "annotation_save": {
                "lock_session_id": "lease-new",
                "session_id": "lease-new",
                "lock_token": "token-new",
            },
        },
    )

    assert recovery["status"] == "ready"
    assert recovery["failures"] == []
    assert len(harness.save_calls) == 2
    assert harness.save_calls[1]["lock_token"] == "token-new"
    assert [item["image_relpath"] for item in harness.save_calls[1]["records"]] == [
        "two.jpg"
    ]
    assert {
        item["image_key"]
        for item in recovery["results"][0]["records"]
        if item["annotation_record_revision"]
    } == {"train:one.jpg", "train:two.jpg"}
    assert all(row["label_lines"] == [] for row in harness.manifest["images"])


def test_recovery_refuses_to_replay_without_a_fresh_annotation_lease(
    monkeypatch, tmp_path
) -> None:
    harness = _install_transaction_harness(
        monkeypatch,
        tmp_path,
        {"one.jpg": ["0 0.5 0.5 0.2 0.3"]},
    )

    def interrupt(_payload, _apply_saved_records):
        raise RuntimeError("connection lost before save acknowledgement")

    harness.state["save_hook"] = interrupt
    with pytest.raises(RuntimeError, match="connection lost"):
        api.commit_class_analysis_annotation_transaction(
            JOB_ID, deepcopy(harness.payload)
        )

    with pytest.raises(HTTPException) as exc_info:
        api.recover_class_analysis_annotation_session(
            JOB_ID,
            {"mode": "retry_transactions", "operation_ids": ["operation-1"]},
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "annotation_recovery_fresh_lease_required"
    assert harness.store.get_operation("operation-1")["state"] == "prepared"


def test_partial_review_receipts_retry_with_stable_action_ids(monkeypatch, tmp_path) -> None:
    harness = _install_transaction_harness(
        monkeypatch,
        tmp_path,
        {
            "one.jpg": [
                "0 0.3 0.3 0.1 0.1",
                "1 0.7 0.7 0.1 0.1",
            ]
        },
    )
    failed_point = "point-0-1"
    attempts = {}

    def fail_second_receipt_once(kwargs):
        point_id = kwargs["point_id"]
        attempts[point_id] = attempts.get(point_id, 0) + 1
        if point_id == failed_point and attempts[point_id] == 1:
            raise HTTPException(status_code=503, detail="review ledger unavailable")
        return {
            "status": "recorded",
            "point_id": point_id,
            "client_action_id": kwargs["client_action_id"],
        }

    harness.state["review_hook"] = fail_second_receipt_once
    first = api.commit_class_analysis_annotation_transaction(
        JOB_ID, deepcopy(harness.payload)
    )
    recovery = api.recover_class_analysis_annotation_session(
        JOB_ID,
        {
            "mode": "retry_transactions",
            "operation_ids": ["operation-1"],
            "annotation_save": {
                "lock_session_id": "lease-new",
                "session_id": "lease-new",
                "lock_token": "token-new",
            },
        },
    )

    assert first["status"] == "recovery_required"
    assert recovery["status"] == "ready"
    assert len(harness.save_calls) == 1
    action_ids = {}
    for call in harness.review_calls:
        action_ids.setdefault(call["point_id"], set()).add(call["client_action_id"])
    assert all(len(values) == 1 for values in action_ids.values())
    assert attempts == {"point-0-0": 2, "point-0-1": 2}


def test_untouched_sibling_entity_can_commit_after_prior_graph_edit(
    monkeypatch, tmp_path
) -> None:
    harness = _install_transaction_harness(
        monkeypatch,
        tmp_path,
        {
            "one.jpg": [
                "0 0.3 0.3 0.1 0.1",
                "1 0.7 0.7 0.1 0.1",
            ]
        },
    )
    first = deepcopy(harness.payload)
    first["records"][0]["actions"] = [first["records"][0]["actions"][0]]
    assert api.commit_class_analysis_annotation_transaction(JOB_ID, first)["status"] == "complete"

    current_record = harness.store.get_record(SOURCE_IDENTITY, "train:one.jpg")
    assert current_record is not None
    second = deepcopy(harness.payload)
    second["client_action_id"] = "operation-2"
    second["records"][0]["actions"] = [second["records"][0]["actions"][1]]
    second["records"][0]["expected_annotation_record_revision"] = harness.manifest[
        "images"
    ][0]["annotation_record_revision"]
    second["records"][0]["expected_record_revision"] = current_record[
        "record_revision"
    ]

    result = api.commit_class_analysis_annotation_transaction(JOB_ID, second)

    assert result["status"] == "complete"
    assert harness.manifest["images"][0]["label_lines"] == []
    assert len(harness.save_calls) == 2


def test_transaction_rejects_ambiguous_or_invalid_requests_before_saving(
    monkeypatch, tmp_path
) -> None:
    harness = _install_transaction_harness(
        monkeypatch,
        tmp_path,
        {"one.jpg": ["0 0.5 0.5 0.2 0.3"]},
    )
    point_id = "point-0-0"
    harness.points[point_id]["annotation_attestation"] = ""
    with pytest.raises(HTTPException, match="annotation_entity_attestation_required"):
        api.commit_class_analysis_annotation_transaction(
            JOB_ID, deepcopy(harness.payload)
        )
    assert harness.save_calls == []

    stored = harness.store.get_record(SOURCE_IDENTITY, "train:one.jpg")
    assert stored is not None
    harness.points[point_id]["annotation_attestation"] = _attestation(
        "train:one.jpg",
        stored["entities"][0],
        harness.payload["records"][0]["expected_record_revision"],
    )
    duplicate = deepcopy(harness.payload)
    duplicate["records"].append(deepcopy(duplicate["records"][0]))
    with pytest.raises(HTTPException, match="annotation_transaction_duplicate_record"):
        api.commit_class_analysis_annotation_transaction(JOB_ID, duplicate)

    invalid_target = deepcopy(harness.payload)
    invalid_target["client_action_id"] = "operation-invalid-target"
    invalid_target["records"][0]["actions"][0].update(
        {"action": "relabel", "target_class_id": 99}
    )
    with pytest.raises(HTTPException, match="annotation_transaction_target_class_invalid"):
        api.commit_class_analysis_annotation_transaction(JOB_ID, invalid_target)

    manual_edit = deepcopy(harness.payload)
    manual_edit["client_action_id"] = "operation-sidecar-stale"
    harness.manifest["images"][0]["label_lines"] = ["2 0.5 0.5 0.2 0.3"]
    harness.manifest["images"][0]["annotation_entity_record_revision"] = "alr1_manual_edit"
    with pytest.raises(HTTPException, match="annotation_entity_sidecar_out_of_sync"):
        api.commit_class_analysis_annotation_transaction(JOB_ID, manual_edit)
    assert harness.save_calls == []


def test_confirm_current_requires_the_same_live_entity_revision_and_class(
    monkeypatch, tmp_path
) -> None:
    harness = _install_transaction_harness(
        monkeypatch,
        tmp_path,
        {"one.jpg": ["0 0.5 0.5 0.2 0.3"]},
    )
    point = harness.points["point-0-0"]
    row = harness.manifest["images"][0]
    job = api._get_class_analysis_job(JOB_ID)
    monkeypatch.setattr(
        api,
        "_class_analysis_annotation_transaction_record",
        lambda source_kind, source_id, split, image_relpath: deepcopy(row),
    )
    precondition = {
        "annotation_source": {"kind": "transient", "id": SOURCE_ID},
        "split": "train",
        "image_relpath": "one.jpg",
        "annotation_source_identity": SOURCE_IDENTITY,
        "annotation_entity_id": point["annotation_entity_id"],
        "expected_entity_revision": point["annotation_entity_revision"],
        "expected_record_revision": point["annotation_entity_record_revision"],
        "expected_annotation_record_revision": row["annotation_record_revision"],
        "expected_class_id": 0,
        "expected_class_name": "First",
    }

    attestation = api._class_analysis_validate_confirm_current_annotation(
        job=job,
        point=point,
        payload={"annotation_precondition": precondition},
    )

    assert attestation["annotation_entity_id"] == point["annotation_entity_id"]
    assert attestation["class_name"] == "First"
    assert attestation["verification_method"] == "stable_entity_live_class_cas_v1"

    stored = harness.store.get_record(SOURCE_IDENTITY, "train:one.jpg")
    assert stored is not None
    changed, _results = apply_annotation_entity_actions(
        stored,
        [
            {
                "annotation_entity_id": point["annotation_entity_id"],
                "expected_entity_revision": point["annotation_entity_revision"],
                "action": "relabel",
                "target_class_id": 1,
            }
        ],
        expected_record_revision=stored["record_revision"],
    )
    harness.store.put_record(SOURCE_IDENTITY, "train:one.jpg", changed)
    row["label_lines"] = live_label_lines(changed)
    row["annotation_entity_record_revision"] = changed["record_revision"]
    row["annotation_record_revision"] = "labels-changed"
    stale_precondition = deepcopy(precondition)
    stale_precondition["expected_record_revision"] = changed["record_revision"]
    stale_precondition["expected_annotation_record_revision"] = "labels-changed"

    with pytest.raises(HTTPException) as exc_info:
        api._class_analysis_validate_confirm_current_annotation(
            job=job,
            point=point,
            payload={"annotation_precondition": stale_precondition},
        )
    assert "annotation_review_entity_changed" in str(exc_info.value.detail)


def test_confirm_current_without_identity_precondition_never_records(
    monkeypatch, tmp_path
) -> None:
    harness = _install_transaction_harness(
        monkeypatch,
        tmp_path,
        {"one.jpg": ["0 0.5 0.5 0.2 0.3"]},
    )
    point = deepcopy(harness.points["point-0-0"])
    point["annotation_entity_id"] = ""
    monkeypatch.setattr(api, "_class_analysis_lookup_review_dispositions", lambda keys: {})
    monkeypatch.setattr(
        api,
        "_class_analysis_record_review_disposition_entry",
        lambda **kwargs: pytest.fail("invalid confirm reached the durable writer"),
    )

    with pytest.raises(HTTPException) as exc_info:
        api._class_analysis_record_confirm_current_disposition(
            job=api._get_class_analysis_job(JOB_ID),
            result={"summary": {"analysis_job_id": JOB_ID}},
            point=point,
            payload={},
            origin="desktop",
            client_action_id="confirm-without-identity",
            training_capture_requested=False,
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail["code"] == "annotation_precondition_required"


def test_recovery_surfaces_bound_source_conflict_without_an_operation(
    monkeypatch, tmp_path
) -> None:
    harness = _install_transaction_harness(
        monkeypatch,
        tmp_path,
        {"one.jpg": ["0 0.5 0.5 0.2 0.3"]},
    )
    harness.store.bind_job_sources(JOB_ID, [SOURCE_IDENTITY])
    harness.store.mark_conflict(
        SOURCE_IDENTITY,
        "train:one.jpg",
        "duplicate exact bbox",
    )
    monkeypatch.setattr(
        api,
        "_class_analysis_session_store_path",
        lambda _job_id: tmp_path / "session.sqlite3",
    )
    monkeypatch.setattr(
        api,
        "get_class_analysis_session_identity_summary",
        lambda _path: {
            "schema": "class-analysis-session-store-v3",
            "identity_conflict_count": 0,
            "invalid_identity_count": 0,
            "source_identities": [SOURCE_IDENTITY],
        },
    )

    recovery = api.get_class_analysis_annotation_recovery(JOB_ID)

    assert recovery["status"] == "rerun_required"
    assert recovery["rerun_required"] is True
    assert recovery["identity_conflict_count"] == 1
    assert recovery["can_save_session"] is False
    assert recovery["can_export_dataset"] is False


def test_recovery_cannot_clear_a_frozen_conflict_with_a_mutable_record(
    monkeypatch, tmp_path
) -> None:
    harness = _install_transaction_harness(
        monkeypatch,
        tmp_path,
        {"one.jpg": ["0 0.5 0.5 0.2 0.3"]},
    )
    harness.store.mark_conflict(
        SOURCE_IDENTITY,
        "train:one.jpg",
        "duplicate exact bbox",
    )
    harness.store.put_record(SOURCE_IDENTITY, "train:one.jpg", {"entities": []})
    assert harness.store.list_conflicts(SOURCE_IDENTITY) == []
    monkeypatch.setattr(
        api,
        "_class_analysis_session_store_path",
        lambda _job_id: tmp_path / "session.sqlite3",
    )
    monkeypatch.setattr(
        api,
        "get_class_analysis_session_identity_summary",
        lambda _path: {
            "schema": "class-analysis-session-store-v3",
            "identity_conflict_count": 1,
            "invalid_identity_count": 0,
            "source_identities": [SOURCE_IDENTITY],
        },
    )

    recovery = api.get_class_analysis_annotation_recovery(JOB_ID)

    assert recovery["status"] == "rerun_required"
    assert recovery["frozen_identity_conflict_count"] == 1
    assert recovery["frozen_identity_invalid_count"] == 0
    assert recovery["reason_codes"] == ["frozen_annotation_identity_conflict"]
    assert recovery["can_save_session"] is False
    assert recovery["can_export_dataset"] is False
