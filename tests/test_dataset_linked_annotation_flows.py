from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import localinferenceapi as api


def test_delete_linked_dataset_only_removes_registry_record(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "keep.txt").write_text("source", encoding="utf-8")

    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    record_root = registry_root / "ds_linked"
    record_root.mkdir(parents=True, exist_ok=True)
    (record_root / api.DATASET_META_NAME).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
        },
    )

    out = api.delete_dataset_entry("ds_linked")

    assert out["status"] == "deleted"
    assert out["storage_mode"] == "linked"
    assert source_root.exists(), "linked source must never be deleted"
    assert not record_root.exists(), "registry record should be removed"


def test_delete_linked_dataset_unlinks_registry_symlink_without_target_delete(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    target_record = registry_root / "target_record"
    target_record.mkdir(parents=True, exist_ok=True)
    (target_record / "payload.bin").write_bytes(b"target")
    record_link = registry_root / "ds_linked"
    try:
        record_link.symlink_to(target_record, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_link),
            "storage_mode": "linked",
            "linked_root": str(source_root),
        },
    )

    out = api.delete_dataset_entry("ds_linked")

    assert out["status"] == "deleted"
    assert not record_link.exists()
    assert not record_link.is_symlink()
    assert (target_record / "payload.bin").read_bytes() == b"target"
    assert source_root.exists()


def test_delete_linked_dataset_rejects_symlinked_registry_parent_without_target_delete(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    target_parent = registry_root / "target_parent"
    target_parent.mkdir(parents=True, exist_ok=True)
    target_record = target_parent / "ds_linked"
    target_record.mkdir()
    (target_record / "payload.bin").write_bytes(b"target")
    parent_link = registry_root / "linked_parent"
    try:
        parent_link.symlink_to(target_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(parent_link / "ds_linked"),
            "storage_mode": "linked",
            "linked_root": str(source_root),
        },
    )

    with pytest.raises(api.HTTPException) as exc:
        api.delete_dataset_entry("ds_linked")

    assert exc.value.status_code == 400
    assert exc.value.detail == "dataset_delete_forbidden"
    assert (target_record / "payload.bin").read_bytes() == b"target"
    assert source_root.exists()


def test_delete_managed_dataset_rejects_symlinked_registry_parent_without_target_delete(
    tmp_path, monkeypatch
) -> None:
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    target_parent = registry_root / "target_parent"
    target_parent.mkdir(parents=True, exist_ok=True)
    target_root = target_parent / "managed_ds"
    target_root.mkdir()
    (target_root / "payload.bin").write_bytes(b"target")
    parent_link = registry_root / "linked_parent"
    try:
        parent_link.symlink_to(target_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", tmp_path / "sam3")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", tmp_path / "qwen")
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "managed_ds",
            "dataset_root": str(parent_link / "managed_ds"),
            "registry_root": str(parent_link / "managed_ds"),
            "storage_mode": "managed",
        },
    )

    with pytest.raises(api.HTTPException) as exc:
        api.delete_dataset_entry("managed_ds")

    assert exc.value.status_code == 400
    assert exc.value.detail == "dataset_delete_forbidden"
    assert (target_root / "payload.bin").read_bytes() == b"target"


def test_delete_managed_dataset_moves_to_trash_and_restores(
    tmp_path, monkeypatch
) -> None:
    registry_root = tmp_path / "registry"
    dataset_root = registry_root / "managed_ds"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "payload.bin").write_bytes(b"dataset")
    (dataset_root / api.DATASET_META_NAME).write_text(
        json.dumps({"id": "managed_ds", "label": "Managed DS"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", tmp_path / "sam3")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", tmp_path / "qwen")
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "managed_ds",
            "label": "Managed DS",
            "dataset_root": str(dataset_root),
            "registry_root": str(dataset_root),
            "storage_mode": "managed",
            "source": "registry",
        },
    )

    out = api.delete_dataset_entry("managed_ds")

    assert out["status"] == "trashed"
    assert out["restore_available"] is True
    assert not dataset_root.exists()
    trash_entries = api.list_dataset_trash_entries()
    assert [entry["trash_id"] for entry in trash_entries] == [out["trash_id"]]
    assert trash_entries[0]["original_id"] == "managed_ds"
    trashed_payload = Path(trash_entries[0]["dataset_path"]) / "payload.bin"
    assert trashed_payload.read_bytes() == b"dataset"

    restored = api.restore_dataset_trash_entry(out["trash_id"])

    assert restored["status"] == "restored"
    assert restored["id"] == "managed_ds"
    assert dataset_root.exists()
    assert (dataset_root / "payload.bin").read_bytes() == b"dataset"
    restored_meta = json.loads((dataset_root / api.DATASET_META_NAME).read_text(encoding="utf-8"))
    assert restored_meta["id"] == "managed_ds"
    assert restored_meta["restored_from_trash_id"] == out["trash_id"]
    assert api.list_dataset_trash_entries() == []


def test_restore_managed_dataset_uses_unique_id_when_original_exists(
    tmp_path, monkeypatch
) -> None:
    registry_root = tmp_path / "registry"
    dataset_root = registry_root / "managed_ds"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "payload.bin").write_bytes(b"dataset")
    (dataset_root / api.DATASET_META_NAME).write_text(
        json.dumps({"id": "managed_ds", "label": "Managed DS"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", tmp_path / "sam3")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", tmp_path / "qwen")
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "managed_ds",
            "label": "Managed DS",
            "dataset_root": str(dataset_root),
            "registry_root": str(dataset_root),
            "storage_mode": "managed",
        },
    )
    trashed = api.delete_dataset_entry("managed_ds")
    replacement = registry_root / "managed_ds"
    replacement.mkdir()
    (replacement / api.DATASET_META_NAME).write_text(
        json.dumps({"id": "managed_ds", "label": "Replacement"}),
        encoding="utf-8",
    )

    restored = api.restore_dataset_trash_entry(trashed["trash_id"])

    assert restored["id"] == "managed_ds_1"
    restored_root = registry_root / "managed_ds_1"
    assert restored_root.exists()
    assert (restored_root / "payload.bin").read_bytes() == b"dataset"
    restored_meta = json.loads((restored_root / api.DATASET_META_NAME).read_text(encoding="utf-8"))
    assert restored_meta["id"] == "managed_ds_1"
    assert (replacement / api.DATASET_META_NAME).exists()


def test_restore_managed_dataset_rolls_back_when_metadata_write_fails(
    tmp_path, monkeypatch
) -> None:
    registry_root = tmp_path / "registry"
    dataset_root = registry_root / "managed_ds"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "payload.bin").write_bytes(b"dataset")
    (dataset_root / api.DATASET_META_NAME).write_text(
        json.dumps({"id": "managed_ds", "label": "Managed DS"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", tmp_path / "sam3")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", tmp_path / "qwen")
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "managed_ds",
            "label": "Managed DS",
            "dataset_root": str(dataset_root),
            "registry_root": str(dataset_root),
            "storage_mode": "managed",
        },
    )
    trashed = api.delete_dataset_entry("managed_ds")
    trash_entry = api.list_dataset_trash_entries()[0]
    trash_dataset_root = Path(trash_entry["dataset_path"])

    def fail_metadata_write(*_args, **_kwargs):
        raise api.HTTPException(status_code=400, detail="metadata_write_failed")

    monkeypatch.setattr(api, "_write_dataset_metadata_json", fail_metadata_write)

    with pytest.raises(api.HTTPException) as exc:
        api.restore_dataset_trash_entry(trashed["trash_id"])

    assert exc.value.detail == "metadata_write_failed"
    assert not dataset_root.exists()
    assert trash_dataset_root.exists()
    assert (trash_dataset_root / "payload.bin").read_bytes() == b"dataset"


def test_restore_managed_dataset_rejects_symlinked_trash_entry_without_target_write(
    tmp_path, monkeypatch
) -> None:
    registry_root = tmp_path / "registry"
    trash_root = registry_root / api.DATASET_TRASH_DIRNAME
    outside = tmp_path / "outside"
    outside.mkdir(parents=True)
    trash_root.mkdir(parents=True)
    try:
        (trash_root / "linked-trash").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    marker = outside / "payload.bin"
    marker.write_bytes(b"outside")
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", tmp_path / "sam3")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", tmp_path / "qwen")

    with pytest.raises(api.HTTPException) as exc:
        api.restore_dataset_trash_entry("linked-trash")

    assert exc.value.status_code == 404
    assert exc.value.detail == "dataset_trash_entry_not_found"
    assert marker.read_bytes() == b"outside"


def test_delete_linked_dataset_blocks_active_annotation_lock(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)

    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    record_root = registry_root / "ds_linked"
    record_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": "ds_linked",
        "annotation_lock": {
            "holder": "annotator",
            "session_id": "sess-active",
            "expires_at": time.time() + 300.0,
        },
    }
    (record_root / api.DATASET_META_NAME).write_text(json.dumps(meta), encoding="utf-8")

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
        },
    )

    with pytest.raises(api.HTTPException) as exc:
        api.delete_dataset_entry("ds_linked")

    assert exc.value.status_code == 409
    assert exc.value.detail == "dataset_delete_blocked_annotation_lock"
    assert record_root.exists()
    assert source_root.exists()


def test_delete_linked_dataset_blocks_active_job_reference(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)

    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    record_root = registry_root / "ds_linked"
    record_root.mkdir(parents=True, exist_ok=True)
    (record_root / api.DATASET_META_NAME).write_text(
        json.dumps({"id": "ds_linked"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
        },
    )
    with api.DATA_INGESTION_JOBS_LOCK:
        original_jobs = dict(api.DATA_INGESTION_JOBS)
        api.DATA_INGESTION_JOBS.clear()
        api.DATA_INGESTION_JOBS["di_active"] = api.DataIngestionJob(
            job_id="di_active",
            status="running",
            request={"reference_dataset_id": "ds_linked"},
        )

    try:
        with pytest.raises(api.HTTPException) as exc:
            api.delete_dataset_entry("ds_linked")
    finally:
        with api.DATA_INGESTION_JOBS_LOCK:
            api.DATA_INGESTION_JOBS.clear()
            api.DATA_INGESTION_JOBS.update(original_jobs)

    assert exc.value.status_code == 409
    assert exc.value.detail == "dataset_delete_blocked_active_jobs:data_ingestion"
    assert record_root.exists()
    assert source_root.exists()


def test_delete_managed_dataset_blocks_active_annotation_lock(tmp_path, monkeypatch) -> None:
    registry_root = tmp_path / "registry"
    dataset_root = registry_root / "managed_ds"
    dataset_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": "managed_ds",
        "annotation_lock": {
            "holder": "annotator",
            "session_id": "sess-active",
            "expires_at": time.time() + 300.0,
        },
    }
    (dataset_root / api.DATASET_META_NAME).write_text(json.dumps(meta), encoding="utf-8")

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", tmp_path / "sam3")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", tmp_path / "qwen")
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "managed_ds",
            "dataset_root": str(dataset_root),
            "registry_root": str(dataset_root),
            "storage_mode": "managed",
        },
    )

    with pytest.raises(api.HTTPException) as exc:
        api.delete_dataset_entry("managed_ds")

    assert exc.value.status_code == 409
    assert exc.value.detail == "dataset_delete_blocked_annotation_lock"
    assert dataset_root.exists()


def test_delete_managed_dataset_ignores_completed_job_reference(
    tmp_path, monkeypatch
) -> None:
    registry_root = tmp_path / "registry"
    dataset_root = registry_root / "managed_ds"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "payload.bin").write_bytes(b"dataset")
    (dataset_root / api.DATASET_META_NAME).write_text(
        json.dumps({"id": "managed_ds", "label": "Managed DS"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", tmp_path / "sam3")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", tmp_path / "qwen")
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "managed_ds",
            "label": "Managed DS",
            "dataset_root": str(dataset_root),
            "registry_root": str(dataset_root),
            "storage_mode": "managed",
            "source": "registry",
        },
    )
    with api.DATA_INGESTION_JOBS_LOCK:
        original_jobs = dict(api.DATA_INGESTION_JOBS)
        api.DATA_INGESTION_JOBS.clear()
        api.DATA_INGESTION_JOBS["di_done"] = api.DataIngestionJob(
            job_id="di_done",
            status="completed",
            request={
                "reference_dataset_id": "managed_ds",
                "reference_uploads": [{"path": str(dataset_root / "payload.bin")}],
            },
        )

    try:
        out = api.delete_dataset_entry("managed_ds")
    finally:
        with api.DATA_INGESTION_JOBS_LOCK:
            api.DATA_INGESTION_JOBS.clear()
            api.DATA_INGESTION_JOBS.update(original_jobs)

    assert out["status"] == "trashed"
    assert not dataset_root.exists()
    trash_entries = api.list_dataset_trash_entries()
    assert len(trash_entries) == 1
    assert trash_entries[0]["original_id"] == "managed_ds"


def test_save_transient_dataset_persists_overlay_content(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)

    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "_validate_linked_dataset_path", lambda _p: source_root)

    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS.clear()
        api.DATASET_TRANSIENT_SESSIONS["sess1"] = {
            "session_id": "sess1",
            "dataset_root": str(source_root),
            "label": "Transient DS",
            "classes": ["car"],
            "yolo_layout": "flat",
            "annotation_status": "in_progress",
            "annotation_notes": "half done",
            "annotation_cursor": {"split": "train", "image_relpath": "img1.jpg"},
            "annotation_progress": {"images_total": 1},
            "overlay_labels": {"train:img1.jpg": ["0 0.5 0.5 0.2 0.2"]},
            "overlay_text": {"train:img1.jpg": "car in frame"},
        }

    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry_impl",
        lambda dataset_id, **_kwargs: {
            "id": dataset_id,
            "dataset_root": str(source_root),
            "registry_root": str(registry_root / dataset_id),
            "storage_mode": "linked",
            "linked_root": str(source_root),
            "yolo_layout": "flat",
            "classes": ["car"],
        },
    )

    out = api.save_transient_dataset(
        "sess1",
        dataset_id="saved_linked",
        label="Saved Linked",
        context="context",
        notes="notes",
    )

    record_root = registry_root / "saved_linked"
    overlay_label = (
        record_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME / "labels" / "train" / "img1.txt"
    )
    overlay_text = record_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME / "text_labels" / "img1.txt"
    registry_labelmap = record_root / "labelmap.txt"
    meta_path = record_root / api.DATASET_META_NAME

    assert out["id"] == "saved_linked"
    assert not (source_root / "labelmap.txt").exists()
    assert registry_labelmap.read_text(encoding="utf-8") == "car\n"
    assert overlay_label.exists()
    assert overlay_label.read_text(encoding="utf-8").strip() == "0 0.5 0.5 0.2 0.2"
    assert overlay_text.exists()
    assert overlay_text.read_text(encoding="utf-8").strip() == "car in frame"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["annotation_status"] == "in_progress"
    assert meta["annotation_notes"] == "notes"
    assert meta["classes"] == ["car"]
    assert meta["yolo_labelmap_path"] == str(registry_labelmap)
    assert meta["labelmap_source"] == "registry_overlay"


def test_save_transient_dataset_rejects_symlinked_registry_root(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside_registry"
    outside.mkdir()
    registry_root = tmp_path / "registry"
    try:
        registry_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "_validate_linked_dataset_path", lambda _p: source_root)

    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS.clear()
        api.DATASET_TRANSIENT_SESSIONS["sess1"] = {
            "session_id": "sess1",
            "dataset_root": str(source_root),
            "label": "Transient DS",
            "classes": ["car"],
            "yolo_layout": "flat",
            "annotation_status": "in_progress",
            "overlay_labels": {"train:img1.jpg": ["0 0.5 0.5 0.2 0.2"]},
            "overlay_text": {"train:img1.jpg": "car in frame"},
        }

    with pytest.raises(api.HTTPException) as exc_info:
        api.save_transient_dataset(
            "sess1",
            dataset_id="saved_linked",
            label="Saved Linked",
            context="context",
            notes="notes",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_register_target_invalid"
    assert list(outside.iterdir()) == []


def test_save_transient_dataset_rolls_back_when_metadata_write_fails(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(api, "_validate_linked_dataset_path", lambda _p: source_root)

    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS.clear()
        api.DATASET_TRANSIENT_SESSIONS["sess1"] = {
            "session_id": "sess1",
            "dataset_root": str(source_root),
            "label": "Transient DS",
            "classes": ["car"],
            "yolo_layout": "flat",
            "annotation_status": "in_progress",
            "overlay_labels": {"train:img1.jpg": ["0 0.5 0.5 0.2 0.2"]},
            "overlay_text": {"train:img1.jpg": "car in frame"},
        }

    def fail_metadata_write(*_args, **_kwargs):
        raise api.HTTPException(status_code=400, detail="metadata_write_failed")

    monkeypatch.setattr(api, "_write_dataset_metadata_json", fail_metadata_write)

    with pytest.raises(api.HTTPException) as exc_info:
        api.save_transient_dataset(
            "sess1",
            dataset_id="saved_linked",
            label="Saved Linked",
            context="context",
            notes="notes",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "metadata_write_failed"
    assert not (registry_root / "saved_linked").exists()
    assert not (source_root / "labelmap.txt").exists()


def test_download_dataset_entry_applies_overlay_files(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "linked_source"
    (source_root / "images").mkdir(parents=True, exist_ok=True)
    (source_root / "labels").mkdir(parents=True, exist_ok=True)
    (source_root / "text_labels").mkdir(parents=True, exist_ok=True)
    (source_root / "labels" / "img1.txt").write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")
    (source_root / "text_labels" / "img1.txt").write_text("old", encoding="utf-8")

    record_root = tmp_path / "registry" / "ds_linked"
    overlay_root = record_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (overlay_root / "text_labels").mkdir(parents=True, exist_ok=True)
    (overlay_root / "labels" / "train" / "img1.txt").write_text(
        "0 0.9 0.9 0.2 0.2\n", encoding="utf-8"
    )
    (overlay_root / "text_labels" / "img1.txt").write_text("new", encoding="utf-8")

    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
            "yolo_layout": "flat",
        },
    )

    response = api.download_dataset_entry("ds_linked")

    zip_path = Path(response.path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        label_name = f"{source_root.name}/labels/img1.txt"
        text_name = f"{source_root.name}/text_labels/img1.txt"
        assert label_name in names
        assert text_name in names
        assert (
            f"{source_root.name}/{api.DATASET_ANNOTATION_OVERLAY_DIRNAME}/labels/train/img1.txt"
            not in names
        )
        assert zf.read(label_name).decode("utf-8").strip() == "0 0.9 0.9 0.2 0.2"
        assert zf.read(text_name).decode("utf-8").strip() == "new"


def test_download_linked_dataset_applies_registry_labelmap_override(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "labelmap.txt").write_text("old\n", encoding="utf-8")

    record_root = tmp_path / "registry" / "ds_linked"
    record_root.mkdir(parents=True, exist_ok=True)
    registry_labelmap = record_root / "labelmap.txt"
    registry_labelmap.write_text("old\nnew\n", encoding="utf-8")

    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
            "yolo_layout": "flat",
            "yolo_labelmap_path": str(registry_labelmap),
            "labelmap_source": "registry_overlay",
        },
    )

    response = api.download_dataset_entry("ds_linked")

    with zipfile.ZipFile(Path(response.path), "r") as zf:
        labelmap_name = f"{source_root.name}/labelmap.txt"
        assert labelmap_name in set(zf.namelist())
        assert zf.read(labelmap_name).decode("utf-8") == "old\nnew\n"


def test_download_split_dataset_applies_split_scoped_text_overlays(
    tmp_path, monkeypatch
) -> None:
    entry = _split_entry_for_annotation(tmp_path)
    source_root = Path(entry["dataset_root"])
    for split, old_text, new_text in (
        ("train", "old train", "new train"),
        ("val", "old val", "new val"),
    ):
        source_text = source_root / split / "text_labels"
        source_text.mkdir(parents=True, exist_ok=True)
        (source_text / "shared.txt").write_text(old_text, encoding="utf-8")
        overlay_text = (
            Path(entry["registry_root"])
            / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
            / "text_labels"
            / split
        )
        overlay_text.mkdir(parents=True, exist_ok=True)
        (overlay_text / "shared.txt").write_text(new_text, encoding="utf-8")

    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)

    response = api.download_dataset_entry("ds")

    with zipfile.ZipFile(Path(response.path), "r") as zf:
        names = set(zf.namelist())
        train_name = f"{source_root.name}/train/text_labels/shared.txt"
        val_name = f"{source_root.name}/val/text_labels/shared.txt"
        assert train_name in names
        assert val_name in names
        assert zf.read(train_name).decode("utf-8") == "new train"
        assert zf.read(val_name).decode("utf-8") == "new val"
        assert f"{source_root.name}/text_labels/train/shared.txt" not in names


def test_download_dataset_entry_skips_dataset_symlink_escape(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "linked_source"
    (source_root / "images").mkdir(parents=True, exist_ok=True)
    (source_root / "images" / "ok.txt").write_text("ok", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (source_root / "images" / "escape.txt").symlink_to(outside)
    record_root = tmp_path / "registry" / "ds_linked"
    record_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
            "yolo_layout": "flat",
        },
    )

    response = api.download_dataset_entry("ds_linked")

    with zipfile.ZipFile(Path(response.path), "r") as zf:
        names = set(zf.namelist())
        assert f"{source_root.name}/images/ok.txt" in names
        assert f"{source_root.name}/images/escape.txt" not in names


def test_download_dataset_entry_skips_overlay_symlink_escape(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    record_root = tmp_path / "registry" / "ds_linked"
    overlay_root = record_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (overlay_root / "labels" / "train" / "escape.txt").symlink_to(outside)

    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
            "yolo_layout": "flat",
        },
    )

    response = api.download_dataset_entry("ds_linked")

    with zipfile.ZipFile(Path(response.path), "r") as zf:
        assert f"{source_root.name}/labels/escape.txt" not in set(zf.namelist())


def test_download_linked_dataset_rejects_symlinked_registry_labelmap(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "labelmap.txt").write_text("old\n", encoding="utf-8")
    record_root = tmp_path / "registry" / "ds_linked"
    record_root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside_labelmap.txt"
    outside.write_text("secret\n", encoding="utf-8")
    registry_labelmap = record_root / "labelmap.txt"
    try:
        registry_labelmap.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
            "yolo_layout": "flat",
            "yolo_labelmap_path": str(registry_labelmap),
            "labelmap_source": "registry_overlay",
        },
    )

    with pytest.raises(api.HTTPException) as exc:
        api.download_dataset_entry("ds_linked")

    assert exc.value.status_code == 400
    assert exc.value.detail == "labelmap_path_forbidden"


def test_set_dataset_glossary_for_linked_dataset_writes_registry_meta(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)

    record_root = tmp_path / "registry" / "ds_linked"
    record_root.mkdir(parents=True, exist_ok=True)
    (record_root / api.DATASET_META_NAME).write_text(
        json.dumps({"id": "ds_linked", "label": "ds_linked"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
            "classes": ["gas_tank"],
        },
    )

    default = api.get_dataset_glossary("ds_linked")

    assert default["classes"] == ["gas_tank"]
    assert "storage tank" in default["default_glossary"]

    out = api.set_dataset_glossary("ds_linked", '{"car": ["vehicle"]}')

    assert out["dataset_id"] == "ds_linked"
    meta = json.loads((record_root / api.DATASET_META_NAME).read_text(encoding="utf-8"))
    assert "labelmap_glossary" in meta
    assert not (source_root / api.DATASET_META_NAME).exists()


def test_set_dataset_glossary_rejects_symlinked_registry_parent_without_write(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    outside_registry = tmp_path / "outside_registry"
    outside_registry.mkdir()
    linked_registry = tmp_path / "linked_registry"
    try:
        linked_registry.symlink_to(outside_registry, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(linked_registry / "ds_linked"),
            "storage_mode": "linked",
            "linked_root": str(source_root),
            "classes": ["gas_tank"],
        },
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.set_dataset_glossary("ds_linked", '{"car": ["vehicle"]}')

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_metadata_path_forbidden"
    assert list(outside_registry.iterdir()) == []


def test_set_dataset_glossary_rejects_symlinked_metadata_without_target_write(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "linked_source"
    source_root.mkdir(parents=True, exist_ok=True)
    record_root = tmp_path / "registry" / "ds_linked"
    record_root.mkdir(parents=True, exist_ok=True)
    outside_meta = tmp_path / "outside_meta.json"
    outside_meta.write_text('{"secret": true}', encoding="utf-8")
    try:
        (record_root / api.DATASET_META_NAME).symlink_to(outside_meta)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds_linked",
            "dataset_root": str(source_root),
            "registry_root": str(record_root),
            "storage_mode": "linked",
            "linked_root": str(source_root),
            "classes": ["gas_tank"],
        },
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.set_dataset_glossary("ds_linked", '{"car": ["vehicle"]}')

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_metadata_path_forbidden"
    assert outside_meta.read_text(encoding="utf-8") == '{"secret": true}'


def _entry_for_annotation(tmp_path: Path) -> dict:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    for relpath in (
        "img.jpg",
        "img_ok.jpg",
        "nested/img.jpg",
        "sub/img.jpg",
        "sub_a/img.jpg",
        "sub_b/img.png",
    ):
        _write_test_image(dataset_root / "images" / relpath)
    registry_root = tmp_path / "registry" / "ds"
    registry_root.mkdir(parents=True, exist_ok=True)
    return {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "registry_root": str(registry_root),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "flat",
        "classes": ["car"],
    }


def _split_entry_for_annotation(tmp_path: Path) -> dict:
    dataset_root = tmp_path / "dataset"
    for split in ("train", "val"):
        (dataset_root / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_root / split / "labels").mkdir(parents=True, exist_ok=True)
        _write_test_image(dataset_root / split / "images" / "shared.jpg")
    registry_root = tmp_path / "registry" / "ds"
    registry_root.mkdir(parents=True, exist_ok=True)
    return {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "registry_root": str(registry_root),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "split",
        "classes": ["car"],
    }


def _active_lock(session_id: str = "sess-1") -> dict:
    return {
        "holder": "annotator",
        "session_id": session_id,
        "expires_at": time.time() + 300.0,
    }


def _write_test_image(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color=(20, 40, 80))
    image.save(path)


def test_manifest_does_not_expose_meta_path(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    manifest = api.get_dataset_annotation_manifest("ds")
    assert "meta_path" not in manifest


def test_persistent_snapshot_requires_active_lock(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_dataset_effective_root_from_entry", lambda _entry: Path(_entry["dataset_root"])
    )
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: {"progress": {}})
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    with pytest.raises(api.HTTPException) as exc:
        api.save_dataset_annotation_snapshot(
            "ds",
            {"records": [{"split": "train", "image_relpath": "img.jpg", "label_lines": []}]},
        )
    assert exc.value.status_code == 409
    assert exc.value.detail == "annotation_lock_session_required"

    with pytest.raises(api.HTTPException) as exc2:
        api.save_dataset_annotation_snapshot(
            "ds",
            {
                "session_id": "wrong",
                "records": [{"split": "train", "image_relpath": "img.jpg", "label_lines": []}],
            },
        )
    assert exc2.value.status_code == 409
    assert exc2.value.detail == "annotation_lock_active"


def test_persistent_meta_patch_requires_lock_owner(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    with pytest.raises(api.HTTPException) as exc:
        api.patch_dataset_annotation_meta("ds", {"status": "in_progress"})
    assert exc.value.status_code == 409
    assert exc.value.detail == "annotation_lock_session_required"


def test_persistent_snapshot_validates_status_before_overlay_writes(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_dataset_effective_root_from_entry", lambda _entry: Path(_entry["dataset_root"])
    )
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: {"progress": {}})
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    with pytest.raises(api.HTTPException) as exc:
        api.save_dataset_annotation_snapshot(
            "ds",
            {
                "session_id": "sess-lock",
                "status": "not_a_status",
                "records": [
                    {
                        "split": "train",
                        "image_relpath": "img.jpg",
                        "label_lines": ["0 0.5 0.5 0.1 0.1"],
                        "text_label": "caption",
                    }
                ],
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "invalid_annotation_status"
    overlay_root = Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    assert not (overlay_root / "labels" / "train" / "img.txt").exists()
    assert not (overlay_root / "text_labels" / "img.txt").exists()
    assert "annotation_status" not in meta


def test_persistent_snapshot_normalises_all_records_before_overlay_writes(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_dataset_effective_root_from_entry", lambda _entry: Path(_entry["dataset_root"])
    )
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: {"progress": {}})
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    with pytest.raises(api.HTTPException) as exc:
        api.save_dataset_annotation_snapshot(
            "ds",
            {
                "session_id": "sess-lock",
                "records": [
                    {
                        "split": "train",
                        "image_relpath": "img_ok.jpg",
                        "label_lines": ["0 0.5 0.5 0.1 0.1"],
                    },
                    {
                        "split": "train",
                        "image_relpath": "../bad.jpg",
                        "label_lines": [],
                    },
                ],
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "invalid_relative_path"
    overlay_root = Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    assert not (overlay_root / "labels" / "train" / "img_ok.txt").exists()


def test_persistent_snapshot_rejects_records_for_missing_images(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: {"progress": {}})
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    with pytest.raises(api.HTTPException) as exc:
        api.save_dataset_annotation_snapshot(
            "ds",
            {
                "session_id": "sess-lock",
                "records": [
                    {
                        "split": "train",
                        "image_relpath": "missing.jpg",
                        "label_lines": ["0 0.5 0.5 0.1 0.1"],
                    }
                ],
            },
        )

    assert exc.value.status_code == 404
    assert exc.value.detail == "annotation_image_not_found"
    overlay_root = Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    assert not (overlay_root / "labels" / "train" / "missing.txt").exists()


def test_persistent_snapshot_text_only_update_preserves_existing_label_overlay(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    overlay_root = Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    label_path = overlay_root / "labels" / "train" / "img.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: {"progress": {}})
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    out = api.save_dataset_annotation_snapshot(
        "ds",
        {
            "session_id": "sess-lock",
            "records": [
                {
                    "split": "train",
                    "image_relpath": "img.jpg",
                    "text_label": "new caption",
                }
            ],
        },
    )

    assert out["status"] == "saved"
    assert label_path.read_text(encoding="utf-8") == "0 0.5 0.5 0.2 0.2\n"
    text_path = overlay_root / "text_labels" / "img.txt"
    assert text_path.read_text(encoding="utf-8") == "new caption"


def test_split_snapshot_text_overlays_are_split_scoped(tmp_path, monkeypatch) -> None:
    entry = _split_entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    out = api.save_dataset_annotation_snapshot(
        "ds",
        {
            "session_id": "sess-lock",
            "records": [
                {
                    "split": "train",
                    "image_relpath": "shared.jpg",
                    "text_label": "train caption",
                },
                {
                    "split": "val",
                    "image_relpath": "shared.jpg",
                    "text_label": "val caption",
                },
            ],
        },
    )

    assert out["status"] == "saved"
    overlay_root = Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    assert (
        overlay_root / "text_labels" / "train" / "shared.txt"
    ).read_text(encoding="utf-8") == "train caption"
    assert (
        overlay_root / "text_labels" / "val" / "shared.txt"
    ).read_text(encoding="utf-8") == "val caption"
    assert not (overlay_root / "text_labels" / "shared.txt").exists()

    manifest = api.get_dataset_annotation_manifest("ds")
    captions = {
        (row["split"], row["image_relpath"]): row["text_label"]
        for row in manifest["images"]
    }
    assert captions[("train", "shared.jpg")] == "train caption"
    assert captions[("val", "shared.jpg")] == "val caption"


def test_persistent_snapshot_replaces_overlay_file_symlink_without_touching_target(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    overlay_root = Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    label_dir = overlay_root / "labels" / "train"
    label_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (label_dir / "img.txt").symlink_to(outside)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_dataset_effective_root_from_entry", lambda _entry: Path(_entry["dataset_root"])
    )
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: {"progress": {}})
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    out = api.save_dataset_annotation_snapshot(
        "ds",
        {
            "session_id": "sess-lock",
            "records": [
                {
                    "split": "train",
                    "image_relpath": "img.jpg",
                    "label_lines": ["0 0.5 0.5 0.1 0.1"],
                }
            ],
        },
    )

    assert out["status"] == "saved"
    assert outside.read_text(encoding="utf-8") == "secret"
    label_path = label_dir / "img.txt"
    assert not label_path.is_symlink()
    assert label_path.read_text(encoding="utf-8").strip() == "0 0.5 0.5 0.1 0.1"


def test_persistent_snapshot_rejects_overlay_parent_symlink_escape(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    overlay_root = Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    label_parent = overlay_root / "labels"
    label_parent.mkdir(parents=True, exist_ok=True)
    outside_dir = tmp_path / "outside_labels"
    outside_dir.mkdir()
    (label_parent / "train").symlink_to(outside_dir, target_is_directory=True)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_dataset_effective_root_from_entry", lambda _entry: Path(_entry["dataset_root"])
    )
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )

    with pytest.raises(api.HTTPException) as exc:
        api.save_dataset_annotation_snapshot(
            "ds",
            {
                "session_id": "sess-lock",
                "records": [
                    {
                        "split": "train",
                        "image_relpath": "img.jpg",
                        "label_lines": ["0 0.5 0.5 0.1 0.1"],
                    }
                ],
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "annotation_overlay_path_forbidden"
    assert not (outside_dir / "img.txt").exists()


def test_annotation_overlay_text_write_is_atomic_over_symlink_leaves(
    tmp_path, monkeypatch
) -> None:
    class FixedUUID:
        hex = "deadbeef000000000000000000000000"

    overlay_root = tmp_path / "overlay"
    overlay_root.mkdir()
    label_path = overlay_root / "labels" / "train" / "img.txt"
    label_path.parent.mkdir(parents=True)
    tmp_path_link = label_path.with_suffix(f"{label_path.suffix}.{FixedUUID.hex}.tmp")
    outside_tmp = tmp_path / "outside_tmp.txt"
    outside_final = tmp_path / "outside_final.txt"
    outside_tmp.write_text("external tmp", encoding="utf-8")
    outside_final.write_text("external final", encoding="utf-8")
    try:
        tmp_path_link.symlink_to(outside_tmp)
        label_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api.uuid, "uuid4", lambda: FixedUUID())

    api._annotation_write_text_within_root(
        label_path,
        overlay_root,
        "0 0.5 0.5 0.1 0.1\n",
    )

    assert not tmp_path_link.exists()
    assert not label_path.is_symlink()
    assert label_path.read_text(encoding="utf-8") == "0 0.5 0.5 0.1 0.1\n"
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    assert outside_final.read_text(encoding="utf-8") == "external final"


def test_persistent_snapshot_rejects_overlay_ancestor_symlink_escape(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    overlay_root = Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    overlay_root.mkdir(parents=True, exist_ok=True)
    outside_dir = tmp_path / "outside_labels"
    outside_dir.mkdir()
    try:
        (overlay_root / "labels").symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_dataset_effective_root_from_entry", lambda _entry: Path(_entry["dataset_root"])
    )
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )

    with pytest.raises(api.HTTPException) as exc:
        api.save_dataset_annotation_snapshot(
            "ds",
            {
                "session_id": "sess-lock",
                "records": [
                    {
                        "split": "train",
                        "image_relpath": "nested/img.jpg",
                        "label_lines": ["0 0.5 0.5 0.1 0.1"],
                    }
                ],
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "annotation_overlay_path_forbidden"
    assert list(outside_dir.iterdir()) == []


def test_persistent_snapshot_rejects_symlinked_registry_parent_before_overlay_creation(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    outside_registry = tmp_path / "outside_registry"
    outside_registry.mkdir()
    linked_parent = tmp_path / "linked_registry"
    try:
        linked_parent.symlink_to(outside_registry, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    entry["registry_root"] = str(linked_parent / "nested" / "ds")
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_dataset_effective_root_from_entry", lambda _entry: Path(_entry["dataset_root"])
    )
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )

    with pytest.raises(api.HTTPException) as exc:
        api.save_dataset_annotation_snapshot(
            "ds",
            {
                "session_id": "sess-lock",
                "records": [
                    {
                        "split": "train",
                        "image_relpath": "nested/img.jpg",
                        "label_lines": ["0 0.5 0.5 0.1 0.1"],
                    }
                ],
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "annotation_overlay_path_forbidden"
    assert not (outside_registry / "nested").exists()


def test_annotation_manifest_ignores_overlay_and_source_symlink_escapes(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    _write_test_image(Path(entry["dataset_root"]) / "images" / "img.jpg")
    outside_label = tmp_path / "outside_label.txt"
    outside_label.write_text("0 0.9 0.9 0.1 0.1\n", encoding="utf-8")
    outside_text = tmp_path / "outside_text.txt"
    outside_text.write_text("secret caption", encoding="utf-8")
    overlay_root = Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    overlay_label_dir = overlay_root / "labels" / "train"
    overlay_label_dir.mkdir(parents=True, exist_ok=True)
    (overlay_label_dir / "img.txt").symlink_to(outside_label)
    text_dir = Path(entry["dataset_root"]) / "text_labels"
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / "img.txt").symlink_to(outside_text)
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)

    manifest = api.get_dataset_annotation_manifest("ds")

    assert manifest["images"][0]["label_lines"] == []
    assert manifest["images"][0]["text_label"] == ""


def test_resolve_annotation_image_path_rejects_symlink_escape(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    images_root = dataset_root / "images"
    images_root.mkdir(parents=True)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"not really an image")
    (images_root / "escape.jpg").symlink_to(outside)

    with pytest.raises(api.HTTPException) as exc:
        api._resolve_annotation_image_path(
            dataset_root,
            "flat",
            "train",
            Path("escape.jpg"),
        )

    assert exc.value.status_code == 404
    assert exc.value.detail == "annotation_image_not_found"


def test_persistent_meta_patch_validates_status_before_labelmap_write(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    with pytest.raises(api.HTTPException) as exc:
        api.patch_dataset_annotation_meta(
            "ds",
            {
                "session_id": "sess-lock",
                "status": "not_a_status",
                "labelmap": ["car", "truck"],
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "invalid_annotation_status"
    assert not (Path(entry["dataset_root"]) / "labelmap.txt").exists()
    assert "annotation_status" not in meta


def test_persistent_meta_patch_rejects_labelmap_path_outside_dataset_roots(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    entry["storage_mode"] = "managed"
    entry["linked_root"] = None
    entry["registry_root"] = None
    outside = tmp_path / "outside" / "labelmap.txt"
    entry["yolo_labelmap_path"] = str(outside)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    with pytest.raises(api.HTTPException) as exc:
        api.patch_dataset_annotation_meta(
            "ds",
            {
                "session_id": "sess-lock",
                "labelmap": ["car", "truck"],
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "labelmap_path_forbidden"
    assert not outside.exists()
    assert set(meta) == {"annotation_lock"}


def test_persistent_meta_patch_linked_labelmap_stays_in_registry(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    dataset_root = Path(entry["dataset_root"])
    registry_root = Path(entry["registry_root"])
    source_labelmap = dataset_root / "labelmap.txt"
    source_labelmap.write_text("car\n", encoding="utf-8")
    meta = {
        "id": "ds",
        "label": "ds",
        "annotation_lock": _active_lock("sess-lock"),
        "classes": ["car"],
    }
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api,
        "_annotation_load_or_create_meta",
        lambda _entry: (registry_root / api.DATASET_META_NAME, meta),
    )

    out = api.patch_dataset_annotation_meta(
        "ds",
        {
            "session_id": "sess-lock",
            "labelmap": ["car", "truck"],
        },
    )

    registry_labelmap = registry_root / "labelmap.txt"
    saved_meta = json.loads((registry_root / api.DATASET_META_NAME).read_text(encoding="utf-8"))
    assert out["labelmap"] == ["car", "truck"]
    assert source_labelmap.read_text(encoding="utf-8") == "car\n"
    assert registry_labelmap.read_text(encoding="utf-8") == "car\ntruck\n"
    assert saved_meta["classes"] == ["car", "truck"]
    assert saved_meta["yolo_labelmap_path"] == str(registry_labelmap)
    assert saved_meta["labelmap_source"] == "registry_overlay"


def test_annotation_labelmap_write_is_atomic_over_symlink_leaves(
    tmp_path, monkeypatch
) -> None:
    class FixedUUID:
        hex = "deadbeef000000000000000000000000"

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    labelmap_path = dataset_root / "labelmap.txt"
    tmp_path_link = labelmap_path.with_suffix(f"{labelmap_path.suffix}.{FixedUUID.hex}.tmp")
    outside_tmp = tmp_path / "outside_tmp.txt"
    outside_final = tmp_path / "outside_final.txt"
    outside_tmp.write_text("external tmp", encoding="utf-8")
    outside_final.write_text("external final", encoding="utf-8")
    try:
        tmp_path_link.symlink_to(outside_tmp)
        labelmap_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api.uuid, "uuid4", lambda: FixedUUID())

    api._annotation_write_labelmap_file(labelmap_path, [dataset_root.resolve()], ["car", "truck"])

    assert not tmp_path_link.exists()
    assert not labelmap_path.is_symlink()
    assert labelmap_path.read_text(encoding="utf-8") == "car\ntruck\n"
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    assert outside_final.read_text(encoding="utf-8") == "external final"


def test_persistent_meta_patch_rejects_symlinked_registry_parent_before_metadata_write(
    tmp_path, monkeypatch
) -> None:
    entry = _entry_for_annotation(tmp_path)
    outside_registry = tmp_path / "outside_registry"
    outside_registry.mkdir()
    linked_parent = tmp_path / "linked_registry"
    try:
        linked_parent.symlink_to(outside_registry, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    entry["registry_root"] = str(linked_parent / "nested" / "ds")
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )

    with pytest.raises(api.HTTPException) as exc:
        api.patch_dataset_annotation_meta(
            "ds",
            {
                "session_id": "sess-lock",
                "notes": "blocked",
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "dataset_metadata_path_forbidden"
    assert not (outside_registry / "nested").exists()


def test_stop_session_requires_matching_session_or_force(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    with pytest.raises(api.HTTPException) as exc:
        api.stop_dataset_annotation_session("ds", {})
    assert exc.value.status_code == 409
    assert exc.value.detail == "annotation_lock_session_mismatch"

    out = api.stop_dataset_annotation_session("ds", {"force": True})
    assert out["status"] == "ok"
    assert meta.get("annotation_lock") == {}


def test_start_session_rejects_same_holder_name_with_different_session(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {
        "annotation_lock": {
            "holder": "webui:tab-a",
            "session_id": "sess-a",
            "expires_at": time.time() + 300.0,
        }
    }
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_dataset_effective_root_from_entry", lambda _entry: Path(_entry["dataset_root"])
    )
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: {"progress": {}})
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    out = api.start_dataset_annotation_session(
        "ds",
        {"session_id": "sess-b", "editor_name": "webui:tab-a"},
    )

    assert out["status"] == "warning"
    assert out["warning"] == "annotation_lock_active"
    assert meta["annotation_lock"]["session_id"] == "sess-a"


def test_transient_snapshot_and_meta_patch_require_lock(monkeypatch) -> None:
    session_id = "transient-lock"
    now = time.time()
    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS[session_id] = {
            "session_id": session_id,
            "dataset_root": "/tmp/path",
            "overlay_labels": {},
            "overlay_text": {},
            "annotation_lock": _active_lock("sess-lock"),
            "expires_at": now + 300.0,
        }

    with pytest.raises(api.HTTPException) as exc:
        api.save_transient_annotation_snapshot(session_id, {"records": []})
    assert exc.value.status_code == 409
    assert exc.value.detail == "annotation_lock_session_required"

    with pytest.raises(api.HTTPException) as exc2:
        api.patch_transient_annotation_meta(session_id, {"session_id": "bad", "notes": "x"})
    assert exc2.value.status_code == 409
    assert exc2.value.detail == "annotation_lock_active"

    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS.pop(session_id, None)


def test_transient_snapshot_validates_status_before_overlay_mutation(monkeypatch) -> None:
    session_id = "transient-invalid-status"
    now = time.time()
    overlay_labels = {}
    overlay_text = {}
    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS[session_id] = {
            "session_id": session_id,
            "dataset_root": "/tmp/path",
            "overlay_labels": overlay_labels,
            "overlay_text": overlay_text,
            "annotation_lock": _active_lock("sess-lock"),
            "expires_at": now + 300.0,
        }

    with pytest.raises(api.HTTPException) as exc:
        api.save_transient_annotation_snapshot(
            session_id,
            {
                "session_id": "sess-lock",
                "status": "not_a_status",
                "records": [
                    {
                        "split": "train",
                        "image_relpath": "img.jpg",
                        "label_lines": ["0 0.5 0.5 0.1 0.1"],
                        "text_label": "caption",
                    }
                ],
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "invalid_annotation_status"
    with api.DATASET_TRANSIENT_LOCK:
        session = api.DATASET_TRANSIENT_SESSIONS[session_id]
        assert session["overlay_labels"] == {}
        assert session["overlay_text"] == {}
        assert "annotation_status" not in session
        api.DATASET_TRANSIENT_SESSIONS.pop(session_id, None)


def test_transient_snapshot_rejects_records_for_missing_images(tmp_path, monkeypatch) -> None:
    session_id = "transient-missing-image"
    source_root = tmp_path / "source"
    (source_root / "images").mkdir(parents=True, exist_ok=True)
    now = time.time()
    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS[session_id] = {
            "session_id": session_id,
            "dataset_root": str(source_root),
            "yolo_layout": "flat",
            "overlay_labels": {},
            "overlay_text": {},
            "annotation_lock": _active_lock("sess-lock"),
            "expires_at": now + 300.0,
        }
    monkeypatch.setattr(api, "_validate_linked_dataset_path", lambda _path: source_root)

    with pytest.raises(api.HTTPException) as exc:
        api.save_transient_annotation_snapshot(
            session_id,
            {
                "session_id": "sess-lock",
                "records": [
                    {
                        "split": "train",
                        "image_relpath": "missing.jpg",
                        "label_lines": ["0 0.5 0.5 0.1 0.1"],
                    }
                ],
            },
        )

    assert exc.value.status_code == 404
    assert exc.value.detail == "annotation_image_not_found"
    with api.DATASET_TRANSIENT_LOCK:
        session = api.DATASET_TRANSIENT_SESSIONS[session_id]
        assert session["overlay_labels"] == {}
        assert session["overlay_text"] == {}
        api.DATASET_TRANSIENT_SESSIONS.pop(session_id, None)


def test_transient_meta_patch_validates_status_before_labelmap_write(
    tmp_path, monkeypatch
) -> None:
    session_id = "transient-invalid-meta"
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    now = time.time()
    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS[session_id] = {
            "session_id": session_id,
            "dataset_root": str(source_root),
            "classes": ["old"],
            "annotation_lock": _active_lock("sess-lock"),
            "expires_at": now + 300.0,
        }

    with pytest.raises(api.HTTPException) as exc:
        api.patch_transient_annotation_meta(
            session_id,
            {
                "session_id": "sess-lock",
                "status": "not_a_status",
                "labelmap": ["car", "truck"],
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "invalid_annotation_status"
    assert not (source_root / "labelmap.txt").exists()
    with api.DATASET_TRANSIENT_LOCK:
        session = api.DATASET_TRANSIENT_SESSIONS[session_id]
        assert session["classes"] == ["old"]
        assert "annotation_status" not in session
        api.DATASET_TRANSIENT_SESSIONS.pop(session_id, None)


def test_transient_meta_patch_does_not_touch_forbidden_source_root(
    tmp_path, monkeypatch
) -> None:
    session_id = "transient-forbidden-root"
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    now = time.time()
    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS[session_id] = {
            "session_id": session_id,
            "dataset_root": str(source_root),
            "classes": ["old"],
            "annotation_lock": _active_lock("sess-lock"),
            "expires_at": now + 300.0,
        }

    def reject_linked_path(_path: str) -> Path:
        raise api.HTTPException(status_code=400, detail="dataset_path_not_allowlisted")

    monkeypatch.setattr(api, "_validate_linked_dataset_path", reject_linked_path)

    out = api.patch_transient_annotation_meta(
        session_id,
        {
            "session_id": "sess-lock",
            "labelmap": ["car", "truck"],
        },
    )

    assert out["labelmap"] == ["car", "truck"]
    assert not (source_root / "labelmap.txt").exists()
    with api.DATASET_TRANSIENT_LOCK:
        session = api.DATASET_TRANSIENT_SESSIONS[session_id]
        assert session["classes"] == ["car", "truck"]
        assert session["labelmap_source"] == "transient_session"
        api.DATASET_TRANSIENT_SESSIONS.pop(session_id, None)


def test_transient_meta_patch_keeps_labelmap_in_session_memory(
    tmp_path, monkeypatch
) -> None:
    session_id = "transient-allowed-root"
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    now = time.time()
    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS[session_id] = {
            "session_id": session_id,
            "dataset_root": str(source_root),
            "classes": ["old"],
            "annotation_lock": _active_lock("sess-lock"),
            "expires_at": now + 300.0,
        }
    out = api.patch_transient_annotation_meta(
        session_id,
        {
            "session_id": "sess-lock",
            "labelmap": ["car", "truck"],
        },
    )

    assert out["labelmap"] == ["car", "truck"]
    assert not (source_root / "labelmap.txt").exists()
    with api.DATASET_TRANSIENT_LOCK:
        session = api.DATASET_TRANSIENT_SESSIONS[session_id]
        assert session["classes"] == ["car", "truck"]
        assert session["labelmap_source"] == "transient_session"
        api.DATASET_TRANSIENT_SESSIONS.pop(session_id, None)


def test_transient_meta_patch_never_replaces_source_labelmap_symlink(
    tmp_path, monkeypatch
) -> None:
    session_id = "transient-labelmap-symlink"
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    outside_final = tmp_path / "outside_labelmap.txt"
    outside_tmp = tmp_path / "outside_tmp.txt"
    outside_final.write_text("external final", encoding="utf-8")
    outside_tmp.write_text("external tmp", encoding="utf-8")
    labelmap_path = source_root / "labelmap.txt"
    tmp_link = labelmap_path.with_suffix(f"{labelmap_path.suffix}.feedface.tmp")
    try:
        labelmap_path.symlink_to(outside_final)
        tmp_link.symlink_to(outside_tmp)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    now = time.time()
    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS[session_id] = {
            "session_id": session_id,
            "dataset_root": str(source_root),
            "classes": ["old"],
            "annotation_lock": _active_lock("sess-lock"),
            "expires_at": now + 300.0,
        }
    out = api.patch_transient_annotation_meta(
        session_id,
        {
            "session_id": "sess-lock",
            "labelmap": ["car", "truck"],
        },
    )

    assert out["labelmap"] == ["car", "truck"]
    assert labelmap_path.is_symlink()
    assert tmp_link.is_symlink()
    assert outside_final.read_text(encoding="utf-8") == "external final"
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    with api.DATASET_TRANSIENT_LOCK:
        session = api.DATASET_TRANSIENT_SESSIONS[session_id]
        assert session["classes"] == ["car", "truck"]
        api.DATASET_TRANSIENT_SESSIONS.pop(session_id, None)


def test_transient_delete_and_expiry(monkeypatch) -> None:
    active_id = "transient-active"
    expired_id = "transient-expired"
    now = time.time()
    with api.DATASET_TRANSIENT_LOCK:
        api.DATASET_TRANSIENT_SESSIONS[active_id] = {
            "session_id": active_id,
            "expires_at": now + 300.0,
        }
        api.DATASET_TRANSIENT_SESSIONS[expired_id] = {
            "session_id": expired_id,
            "expires_at": now - 1.0,
        }

    with pytest.raises(api.HTTPException) as exc:
        api.delete_transient_dataset(expired_id)
    assert exc.value.status_code == 410
    assert exc.value.detail == "transient_session_expired"
    out = api.delete_transient_dataset(active_id)
    assert out["status"] == "deleted"

    with api.DATASET_TRANSIENT_LOCK:
        assert active_id not in api.DATASET_TRANSIENT_SESSIONS
        assert expired_id not in api.DATASET_TRANSIENT_SESSIONS


def test_text_overlay_paths_keep_nested_relpaths(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {"annotation_lock": _active_lock("sess-lock")}
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)
    monkeypatch.setattr(
        api, "_dataset_effective_root_from_entry", lambda _entry: Path(_entry["dataset_root"])
    )
    monkeypatch.setattr(
        api, "_annotation_load_or_create_meta", lambda _entry: (Path("/tmp/meta.json"), meta)
    )
    monkeypatch.setattr(api, "_annotation_manifest_for_entry", lambda _entry: {"progress": {}})
    monkeypatch.setattr(api, "_annotation_persist_meta", lambda _entry, _meta: None)

    api.save_dataset_annotation_snapshot(
        "ds",
        {
            "session_id": "sess-lock",
            "records": [
                {
                    "split": "train",
                    "image_relpath": "sub_a/img.jpg",
                    "label_lines": [],
                    "text_label": "A",
                },
                {
                    "split": "train",
                    "image_relpath": "sub_b/img.png",
                    "label_lines": [],
                    "text_label": "B",
                },
            ],
        },
    )

    overlay_root = (
        Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME / "text_labels"
    )
    assert (overlay_root / "sub_a" / "img.txt").read_text(encoding="utf-8") == "A"
    assert (overlay_root / "sub_b" / "img.txt").read_text(encoding="utf-8") == "B"
    assert not (overlay_root / "img.txt").exists()


def test_effective_text_label_falls_back_to_legacy_flat_text_labels(tmp_path) -> None:
    entry = _entry_for_annotation(tmp_path)
    dataset_root = Path(entry["dataset_root"])
    text_root = dataset_root / "text_labels"
    text_root.mkdir(parents=True, exist_ok=True)
    (text_root / "img.txt").write_text("legacy", encoding="utf-8")
    value = api._annotation_effective_text_label(entry, Path("sub_a/img.jpg"))
    assert value == "legacy"


def test_get_text_labels_batch_reads_overlay_source_and_legacy(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    dataset_root = Path(entry["dataset_root"])
    source_text = dataset_root / "text_labels" / "sub_a"
    source_text.mkdir(parents=True, exist_ok=True)
    (source_text / "img.txt").write_text("source caption", encoding="utf-8")
    legacy_text = dataset_root / "text_labels"
    legacy_text.mkdir(parents=True, exist_ok=True)
    (legacy_text / "legacy.txt").write_text("legacy caption", encoding="utf-8")
    overlay_root = (
        Path(entry["registry_root"]) / api.DATASET_ANNOTATION_OVERLAY_DIRNAME / "text_labels"
    )
    (overlay_root / "sub_b").mkdir(parents=True, exist_ok=True)
    (overlay_root / "sub_b" / "img.txt").write_text("overlay caption", encoding="utf-8")
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)

    result = api.get_text_labels(
        "ds",
        ["sub_a/img.jpg", "sub_b/img.png", "nested/legacy.jpg", "missing.jpg"],
    )

    assert result["captions"] == {
        "sub_a/img.jpg": "source caption",
        "sub_b/img.png": "overlay caption",
        "nested/legacy.jpg": "legacy caption",
    }
    assert result["missing"] == ["missing.jpg"]


def test_text_label_endpoints_read_split_prefixed_source_captions(
    tmp_path, monkeypatch
) -> None:
    entry = _split_entry_for_annotation(tmp_path)
    dataset_root = Path(entry["dataset_root"])
    train_text = dataset_root / "train" / "text_labels"
    val_text = dataset_root / "val" / "text_labels"
    train_text.mkdir(parents=True, exist_ok=True)
    val_text.mkdir(parents=True, exist_ok=True)
    (train_text / "shared.txt").write_text("train source caption", encoding="utf-8")
    (val_text / "shared.txt").write_text("val source caption", encoding="utf-8")
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)

    assert api.get_text_label("ds", "train/shared.jpg") == {
        "caption": "train source caption",
    }
    result = api.get_text_labels(
        "ds",
        ["train/shared.jpg", "val/shared.jpg", "shared.jpg"],
    )

    assert result["captions"] == {
        "train/shared.jpg": "train source caption",
        "val/shared.jpg": "val source caption",
    }
    assert result["missing"] == ["shared.jpg"]


def test_text_label_routes_accept_encoded_split_prefixed_image_names(
    tmp_path, monkeypatch
) -> None:
    entry = _split_entry_for_annotation(tmp_path)
    dataset_root = Path(entry["dataset_root"])
    train_text = dataset_root / "train" / "text_labels"
    train_text.mkdir(parents=True, exist_ok=True)
    (train_text / "shared.txt").write_text("train source caption", encoding="utf-8")
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)

    client = TestClient(api.app)
    response = client.get("/datasets/ds/text_labels/train%2Fshared.jpg")

    assert response.status_code == 200
    assert response.json() == {"caption": "train source caption"}

    response = client.post(
        "/datasets/ds/text_labels/val%2Fshared.jpg",
        json={"caption": "new val caption"},
    )

    assert response.status_code == 200
    assert response.json()["caption"] == "new val caption"
    assert (
        Path(entry["registry_root"])
        / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
        / "text_labels"
        / "val"
        / "shared.txt"
    ).read_text(encoding="utf-8") == "new val caption"


def test_set_text_label_requires_active_lock_owner_when_locked(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta = {
        "id": "ds",
        "annotation_lock": _active_lock("sess-lock"),
    }
    meta_path = Path(entry["registry_root"]) / api.DATASET_META_NAME
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)

    with pytest.raises(api.HTTPException) as exc:
        api.set_text_label("ds", "sub/img.jpg", {"caption": "blocked"})
    assert exc.value.status_code == 409
    assert exc.value.detail == "annotation_lock_session_required"

    with pytest.raises(api.HTTPException) as exc2:
        api.set_text_label("ds", "sub/img.jpg", {"session_id": "wrong", "caption": "blocked"})
    assert exc2.value.status_code == 409
    assert exc2.value.detail == "annotation_lock_active"

    out = api.set_text_label(
        "ds",
        "sub/img.jpg",
        {"session_id": "sess-lock", "caption": "saved"},
    )

    assert out == {"status": "saved", "caption": "saved"}
    overlay_text = (
        Path(entry["registry_root"])
        / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
        / "text_labels"
        / "sub"
        / "img.txt"
    )
    assert overlay_text.read_text(encoding="utf-8") == "saved"


def test_set_text_label_allows_direct_write_without_active_lock(tmp_path, monkeypatch) -> None:
    entry = _entry_for_annotation(tmp_path)
    meta_path = Path(entry["registry_root"]) / api.DATASET_META_NAME
    meta_path.write_text(json.dumps({"id": "ds", "annotation_lock": {}}), encoding="utf-8")
    monkeypatch.setattr(api, "_resolve_dataset_entry", lambda _dataset_id: entry)

    out = api.set_text_label("ds", "img.jpg", {"caption": "direct caption"})

    assert out == {"status": "saved", "caption": "direct caption"}
    overlay_text = (
        Path(entry["registry_root"])
        / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
        / "text_labels"
        / "img.txt"
    )
    assert overlay_text.read_text(encoding="utf-8") == "direct caption"


def test_build_qwen_dataset_from_yolo_uses_flat_annotation_overlay_and_registry_glossary(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "nested" / "img.jpg")
    (dataset_root / "labels" / "nested").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "nested" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("old\nnew\n", encoding="utf-8")

    registry_root = tmp_path / "registry" / "ds"
    overlay_root = registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train" / "nested").mkdir(parents=True, exist_ok=True)
    (overlay_root / "labels" / "train" / "nested" / "img.txt").write_text(
        "1 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    registry_root.mkdir(parents=True, exist_ok=True)
    (registry_root / api.DATASET_META_NAME).write_text(
        json.dumps({"id": "ds", "labelmap_glossary": "new: overlay class"}),
        encoding="utf-8",
    )

    qwen_root = tmp_path / "qwen"
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", qwen_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds",
            "label": "ds",
            "dataset_root": str(dataset_root),
            "registry_root": str(registry_root),
            "storage_mode": "linked",
            "linked_root": str(dataset_root),
            "yolo_layout": "flat",
            "yolo_ready": True,
            "yolo_labelmap_path": str(dataset_root / "labelmap.txt"),
            "classes": ["old", "new"],
            "context": "site imagery",
        },
    )

    out = api.build_qwen_dataset_from_yolo("ds")

    assert out["train_count"] == 1
    assert out["val_count"] == 0
    assert out["labelmap_glossary"] == "new: overlay class"
    ann_path = qwen_root / out["id"] / "train" / "annotations.jsonl"
    annotation = json.loads(ann_path.read_text(encoding="utf-8").strip())
    assert annotation["image"] == "nested/img.jpg"
    assert annotation["detections"][0]["label"] == "new"
    assert annotation["detections"][0]["bbox"] == [30, 30, 70, 70]
    assert (qwen_root / out["id"] / "train" / "nested" / "img.jpg").exists()


def test_qwen_dataset_text_write_is_atomic_over_symlink_leaves(
    tmp_path, monkeypatch
) -> None:
    class FixedUUID:
        hex = "deadbeef000000000000000000000000"

    qwen_root = tmp_path / "qwen_dataset"
    annotations_path = qwen_root / "train" / "annotations.jsonl"
    annotations_path.parent.mkdir(parents=True)
    tmp_path_link = annotations_path.with_suffix(
        f"{annotations_path.suffix}.{FixedUUID.hex}.tmp"
    )
    outside_tmp = tmp_path / "outside_tmp.jsonl"
    outside_final = tmp_path / "outside_final.jsonl"
    outside_tmp.write_text("external tmp\n", encoding="utf-8")
    outside_final.write_text("external final\n", encoding="utf-8")
    try:
        tmp_path_link.symlink_to(outside_tmp)
        annotations_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api.uuid, "uuid4", lambda: FixedUUID())

    api._qwen_dataset_write_text_within_root(
        annotations_path,
        qwen_root,
        '{"image":"nested/img.jpg"}\n',
    )

    assert not tmp_path_link.exists()
    assert not annotations_path.is_symlink()
    assert annotations_path.read_text(encoding="utf-8") == '{"image":"nested/img.jpg"}\n'
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp\n"
    assert outside_final.read_text(encoding="utf-8") == "external final\n"


def test_build_qwen_dataset_from_yolo_rejects_symlinked_qwen_root_before_write(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "img.jpg")
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("building\n", encoding="utf-8")
    outside = tmp_path / "outside_qwen"
    outside.mkdir()
    qwen_root = tmp_path / "qwen"
    try:
        qwen_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", qwen_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds",
            "label": "ds",
            "dataset_root": str(dataset_root),
            "yolo_layout": "flat",
            "yolo_ready": True,
            "yolo_labelmap_path": str(dataset_root / "labelmap.txt"),
            "classes": ["building"],
        },
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.build_qwen_dataset_from_yolo("ds")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "qwen_dataset_target_invalid"
    assert list(outside.iterdir()) == []


def test_build_qwen_dataset_from_yolo_rejects_symlinked_qwen_root_parent_before_write(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "img.jpg")
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("building\n", encoding="utf-8")
    outside = tmp_path / "outside_qwen"
    outside.mkdir()
    link_parent = tmp_path / "linked_parent"
    try:
        link_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", link_parent / "qwen")
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds",
            "label": "ds",
            "dataset_root": str(dataset_root),
            "yolo_layout": "flat",
            "yolo_ready": True,
            "yolo_labelmap_path": str(dataset_root / "labelmap.txt"),
            "classes": ["building"],
        },
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.build_qwen_dataset_from_yolo("ds")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "qwen_dataset_target_invalid"
    assert list(outside.iterdir()) == []


def test_build_qwen_dataset_from_yolo_rejects_labelmap_path_outside_dataset_roots(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "img.jpg")
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    outside_labelmap = tmp_path / "outside" / "labelmap.txt"
    outside_labelmap.parent.mkdir(parents=True, exist_ok=True)
    outside_labelmap.write_text("secret\n", encoding="utf-8")
    qwen_root = tmp_path / "qwen"
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", qwen_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds",
            "label": "ds",
            "dataset_root": str(dataset_root),
            "yolo_layout": "flat",
            "yolo_ready": True,
            "yolo_labelmap_path": str(outside_labelmap),
            "classes": ["building"],
        },
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.build_qwen_dataset_from_yolo("ds")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "labelmap_path_forbidden"
    assert not qwen_root.exists()


def test_build_qwen_dataset_from_yolo_rejects_symlinked_labelmap_before_write(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "img.jpg")
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    outside_labelmap = tmp_path / "outside_labelmap.txt"
    outside_labelmap.write_text("secret\n", encoding="utf-8")
    try:
        (dataset_root / "labelmap.txt").symlink_to(outside_labelmap)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    qwen_root = tmp_path / "qwen"
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", qwen_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds",
            "label": "ds",
            "dataset_root": str(dataset_root),
            "yolo_layout": "flat",
            "yolo_ready": True,
            "yolo_labelmap_path": str(dataset_root / "labelmap.txt"),
            "classes": ["building"],
        },
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.build_qwen_dataset_from_yolo("ds")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "labelmap_path_forbidden"
    assert not qwen_root.exists()


def test_build_qwen_dataset_from_yolo_rejects_symlinked_registry_root_before_read(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "img.jpg")
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("building\n", encoding="utf-8")
    outside_registry = tmp_path / "outside_registry"
    outside_registry.mkdir()
    linked_registry = tmp_path / "linked_registry"
    try:
        linked_registry.symlink_to(outside_registry, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    qwen_root = tmp_path / "qwen"
    monkeypatch.setattr(api, "QWEN_DATASET_ROOT", qwen_root)
    monkeypatch.setattr(
        api,
        "_resolve_dataset_entry",
        lambda _dataset_id: {
            "id": "ds",
            "label": "ds",
            "dataset_root": str(dataset_root),
            "registry_root": str(linked_registry),
            "yolo_layout": "flat",
            "yolo_ready": True,
            "yolo_labelmap_path": str(dataset_root / "labelmap.txt"),
            "classes": ["building"],
        },
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.build_qwen_dataset_from_yolo("ds")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_metadata_path_forbidden"
    assert not qwen_root.exists()
    assert list(outside_registry.iterdir()) == []


def test_resolve_sam3_dataset_meta_materializes_annotation_overlay_for_linked_flat_yolo(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "nested" / "img.jpg")
    (dataset_root / "labels" / "nested").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "nested" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("old\nnew\n", encoding="utf-8")

    registry_root = tmp_path / "registry" / "ds"
    overlay_root = registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train" / "nested").mkdir(parents=True, exist_ok=True)
    (overlay_root / "labels" / "train" / "nested" / "img.txt").write_text(
        "1 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    registry_root.mkdir(parents=True, exist_ok=True)
    (registry_root / api.DATASET_META_NAME).write_text(json.dumps({"id": "ds"}), encoding="utf-8")

    entry = {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "registry_root": str(registry_root),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "flat",
        "yolo_ready": True,
        "classes": ["old", "new"],
    }
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [entry])
    monkeypatch.setattr(api, "_resolve_sam3_or_qwen_dataset", lambda _dataset_id: dataset_root)

    meta = api._resolve_sam3_dataset_meta("ds")

    assert meta["source"] == "annotation_overlay"
    assert meta["source_dataset_root"] == str(dataset_root.resolve())
    materialized_root = Path(meta["dataset_root"])
    assert materialized_root == (
        registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME / "sam3_materialized"
    ).resolve()
    assert not (dataset_root / "train").exists()
    coco_train = json.loads(Path(meta["coco_train_json"]).read_text(encoding="utf-8"))
    assert coco_train["categories"][1]["name"] == "new"
    assert coco_train["images"][0]["file_name"] == "images/nested/img.jpg"
    assert coco_train["annotations"][0]["category_id"] == 2
    assert coco_train["annotations"][0]["bbox"] == [30.0, 30.0, 40.0, 40.0]
    assert (materialized_root / "train" / "images" / "nested" / "img.jpg").exists()


def test_resolve_sam3_dataset_meta_rejects_symlinked_materialized_root_without_target_delete(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "img.jpg")
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("building\n", encoding="utf-8")

    registry_root = tmp_path / "registry" / "ds"
    overlay_root = registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (overlay_root / "labels" / "train" / "img.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    victim = overlay_root / "victim"
    victim.mkdir(parents=True)
    marker = victim / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    try:
        (overlay_root / "sam3_materialized").symlink_to(victim, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    registry_root.mkdir(parents=True, exist_ok=True)
    (registry_root / api.DATASET_META_NAME).write_text(json.dumps({"id": "ds"}), encoding="utf-8")

    entry = {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "registry_root": str(registry_root),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "flat",
        "yolo_ready": True,
        "classes": ["building"],
    }
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [entry])
    monkeypatch.setattr(api, "_resolve_sam3_or_qwen_dataset", lambda _dataset_id: dataset_root)

    with pytest.raises(api.HTTPException) as exc_info:
        api._resolve_sam3_dataset_meta("ds")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "sam3_materialize_path_invalid"
    assert marker.read_text(encoding="utf-8") == "keep"


def test_materialized_dataset_text_write_is_atomic_over_symlink_leaves(
    tmp_path, monkeypatch
) -> None:
    class FixedUUID:
        hex = "deadbeef000000000000000000000000"

    materialized_root = tmp_path / "materialized"
    label_path = materialized_root / "train" / "labels" / "img.txt"
    label_path.parent.mkdir(parents=True)
    tmp_path_link = label_path.with_suffix(f"{label_path.suffix}.{FixedUUID.hex}.tmp")
    outside_tmp = tmp_path / "outside_tmp.txt"
    outside_final = tmp_path / "outside_final.txt"
    outside_tmp.write_text("external tmp", encoding="utf-8")
    outside_final.write_text("external final", encoding="utf-8")
    try:
        tmp_path_link.symlink_to(outside_tmp)
        label_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api.uuid, "uuid4", lambda: FixedUUID())

    api._materialized_dataset_write_text_within_root(
        label_path,
        materialized_root,
        "0 0.5 0.5 0.2 0.2\n",
        detail="sam3_materialize_path_invalid",
    )

    assert not tmp_path_link.exists()
    assert not label_path.is_symlink()
    assert label_path.read_text(encoding="utf-8") == "0 0.5 0.5 0.2 0.2\n"
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    assert outside_final.read_text(encoding="utf-8") == "external final"


def test_resolve_sam3_dataset_meta_rejects_symlinked_registry_root_before_materialize(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "img.jpg")
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("building\n", encoding="utf-8")
    outside_registry = tmp_path / "outside_registry"
    outside_registry.mkdir()
    linked_registry = tmp_path / "linked_registry"
    try:
        linked_registry.symlink_to(outside_registry, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    entry = {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "registry_root": str(linked_registry),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "flat",
        "yolo_ready": True,
        "classes": ["building"],
    }
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [entry])
    monkeypatch.setattr(api, "_resolve_sam3_or_qwen_dataset", lambda _dataset_id: dataset_root)

    with pytest.raises(api.HTTPException) as exc_info:
        api._resolve_sam3_dataset_meta("ds")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "sam3_materialize_path_invalid"
    assert list(outside_registry.iterdir()) == []


def test_reset_materialized_dataset_root_rejects_nested_symlinked_allowed_root_before_mkdir(
    tmp_path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(api.HTTPException) as exc_info:
        api._reset_materialized_dataset_root(
            linked_parent / "nested" / "cache" / "materialized",
            linked_parent / "nested" / "cache",
            detail="yolo_cache_path_invalid",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "yolo_cache_path_invalid"
    assert list(outside.iterdir()) == []


def test_reset_materialized_dataset_root_rejects_nested_symlinked_target_before_mkdir(
    tmp_path,
) -> None:
    allowed_root = tmp_path / "allowed"
    outside = tmp_path / "outside_target"
    allowed_root.mkdir()
    outside.mkdir()
    linked_child = allowed_root / "linked_child"
    try:
        linked_child.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(api.HTTPException) as exc_info:
        api._reset_materialized_dataset_root(
            linked_child / "nested" / "materialized",
            allowed_root,
            detail="sam3_materialize_path_invalid",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "sam3_materialize_path_invalid"
    assert list(outside.iterdir()) == []


def test_resolve_yolo_training_dataset_materializes_annotation_overlay_for_linked_flat_yolo(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "nested" / "img.jpg")
    (dataset_root / "labels" / "nested").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "nested" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("old\nnew\n", encoding="utf-8")

    registry_root = tmp_path / "registry" / "ds"
    overlay_root = registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train" / "nested").mkdir(parents=True, exist_ok=True)
    (overlay_root / "labels" / "train" / "nested" / "img.txt").write_text(
        "1 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    registry_root.mkdir(parents=True, exist_ok=True)
    (registry_root / api.DATASET_META_NAME).write_text(json.dumps({"id": "ds"}), encoding="utf-8")

    entry = {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "registry_root": str(registry_root),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "flat",
        "yolo_ready": True,
        "classes": ["old", "new"],
    }
    cache_root = tmp_path / "yolo_cache"
    prepared_root = cache_root / "ds_cache"
    monkeypatch.setattr(api, "YOLO_DATASET_CACHE_ROOT", cache_root)
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [entry])
    monkeypatch.setattr(
        api,
        "_yolo_training_dataset_base_resolver",
        lambda _payload: {
            "dataset_id": "ds",
            "dataset_root": str(dataset_root),
            "prepared_root": str(dataset_root),
            "cache_root": str(prepared_root),
            "yolo_ready": True,
            "yolo_layout": "flat",
            "yolo_labelmap_path": str(dataset_root / "labelmap.txt"),
            "source": "registry",
        },
    )

    out = api._resolve_yolo_training_dataset(api.YoloTrainRequest(dataset_id="ds", accept_tos=True))

    assert out["source"] == "annotation_overlay_cache"
    assert out["prepared_root"] == str(prepared_root.resolve())
    assert out["yolo_layout"] == "split"
    assert Path(out["yolo_labelmap_path"]).read_text(encoding="utf-8") == "old\nnew\n"
    label_path = prepared_root / "train" / "labels" / "nested" / "img.txt"
    assert label_path.read_text(encoding="utf-8") == "1 0.5 0.5 0.4 0.4\n"
    assert (prepared_root / "train" / "images" / "nested" / "img.jpg").exists()
    assert not (dataset_root / "train").exists()


def test_resolve_yolo_training_dataset_rejects_symlinked_cache_materialization_target(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "img.jpg")
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("building\n", encoding="utf-8")

    registry_root = tmp_path / "registry" / "ds"
    overlay_root = registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (overlay_root / "labels" / "train" / "img.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    registry_root.mkdir(parents=True, exist_ok=True)
    (registry_root / api.DATASET_META_NAME).write_text(json.dumps({"id": "ds"}), encoding="utf-8")

    entry = {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "registry_root": str(registry_root),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "flat",
        "yolo_ready": True,
        "classes": ["building"],
    }
    cache_root = tmp_path / "yolo_cache"
    victim = cache_root / "victim"
    victim.mkdir(parents=True)
    marker = victim / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    prepared_root = cache_root / "ds_cache"
    try:
        prepared_root.symlink_to(victim, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "YOLO_DATASET_CACHE_ROOT", cache_root)
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [entry])
    monkeypatch.setattr(
        api,
        "_yolo_training_dataset_base_resolver",
        lambda _payload: {
            "dataset_id": "ds",
            "dataset_root": str(dataset_root),
            "prepared_root": str(dataset_root),
            "cache_root": str(prepared_root),
            "yolo_ready": True,
            "yolo_layout": "flat",
            "yolo_labelmap_path": str(dataset_root / "labelmap.txt"),
            "source": "registry",
        },
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api._resolve_yolo_training_dataset(api.YoloTrainRequest(dataset_id="ds", accept_tos=True))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "yolo_cache_path_invalid"
    assert marker.read_text(encoding="utf-8") == "keep"


def test_resolve_rfdetr_training_dataset_materializes_annotation_overlay_for_linked_flat_yolo(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "nested" / "img.jpg")
    (dataset_root / "labels" / "nested").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "nested" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("old\nnew\n", encoding="utf-8")

    registry_root = tmp_path / "registry" / "ds"
    overlay_root = registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train" / "nested").mkdir(parents=True, exist_ok=True)
    (overlay_root / "labels" / "train" / "nested" / "img.txt").write_text(
        "1 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    registry_root.mkdir(parents=True, exist_ok=True)
    (registry_root / api.DATASET_META_NAME).write_text(json.dumps({"id": "ds"}), encoding="utf-8")

    entry = {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "registry_root": str(registry_root),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "flat",
        "yolo_ready": True,
        "classes": ["old", "new"],
    }
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [entry])

    out = api._resolve_rfdetr_training_dataset(
        api.RfDetrTrainRequest(dataset_id="ds", accept_tos=True)
    )

    assert out["source"] == "annotation_overlay"
    materialized_root = Path(out["dataset_root"])
    assert materialized_root == (
        registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME / "sam3_materialized"
    ).resolve()
    coco_train = json.loads(Path(out["coco_train_json"]).read_text(encoding="utf-8"))
    assert coco_train["annotations"][0]["category_id"] == 2
    assert coco_train["annotations"][0]["bbox"] == [30.0, 30.0, 40.0, 40.0]
    assert not (dataset_root / "train").exists()


def test_prompt_helper_suggest_materializes_annotation_overlay_for_linked_flat_yolo(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "nested" / "img.jpg")
    (dataset_root / "labels" / "nested").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "nested" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("old\nnew\n", encoding="utf-8")

    registry_root = tmp_path / "registry" / "ds"
    overlay_root = registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train" / "nested").mkdir(parents=True, exist_ok=True)
    (overlay_root / "labels" / "train" / "nested" / "img.txt").write_text(
        "1 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    registry_root.mkdir(parents=True, exist_ok=True)
    (registry_root / api.DATASET_META_NAME).write_text(json.dumps({"id": "ds"}), encoding="utf-8")

    entry = {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "registry_root": str(registry_root),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "flat",
        "yolo_ready": True,
        "classes": ["old", "new"],
    }
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [entry])

    out = api.prompt_helper_suggest(
        api.PromptHelperSuggestRequest(dataset_id="ds", max_synonyms=0, use_qwen=False)
    )

    assert any(row["class_id"] == 2 and row["class_name"] == "new" for row in out["classes"])
    materialized_root = (
        registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME / "sam3_materialized"
    )
    coco_train = json.loads(
        (materialized_root / "train" / "_annotations.coco.json").read_text(encoding="utf-8")
    )
    assert coco_train["annotations"][0]["category_id"] == 2
    assert coco_train["annotations"][0]["bbox"] == [30.0, 30.0, 40.0, 40.0]
    assert not (dataset_root / "train").exists()


def test_plan_segmentation_build_materializes_annotation_overlay_for_linked_flat_yolo(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_source"
    _write_test_image(dataset_root / "images" / "nested" / "img.jpg")
    (dataset_root / "labels" / "nested").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "nested" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("old\nnew\n", encoding="utf-8")

    registry_root = tmp_path / "registry" / "ds"
    overlay_root = registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME
    (overlay_root / "labels" / "train" / "nested").mkdir(parents=True, exist_ok=True)
    (overlay_root / "labels" / "train" / "nested" / "img.txt").write_text(
        "1 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    registry_root.mkdir(parents=True, exist_ok=True)
    (registry_root / api.DATASET_META_NAME).write_text(json.dumps({"id": "ds"}), encoding="utf-8")

    entry = {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "registry_root": str(registry_root),
        "storage_mode": "linked",
        "linked_root": str(dataset_root),
        "yolo_layout": "flat",
        "yolo_ready": True,
        "classes": ["old", "new"],
    }
    sam3_root = tmp_path / "sam3_outputs"
    seg_root = tmp_path / "seg_jobs"
    sam3_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [entry])
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", sam3_root)
    monkeypatch.setattr(api, "SEG_BUILDER_ROOT", seg_root)

    planned_meta, planned_layout = api._plan_segmentation_build(
        api.SegmentationBuildRequest(source_dataset_id="ds", output_name="ds_seg")
    )

    materialized_root = (
        registry_root / api.DATASET_ANNOTATION_OVERLAY_DIRNAME / "sam3_materialized"
    ).resolve()
    assert planned_meta["source_dataset_root"] == str(materialized_root)
    assert planned_meta["source_dataset_id"] == "ds"
    assert planned_meta["classes"] == ["old", "new"]
    assert Path(planned_layout["dataset_root"]) == (sam3_root / "ds_seg").resolve()
    assert (
        materialized_root / "train" / "labels" / "nested" / "img.txt"
    ).read_text(encoding="utf-8").strip() == "1 0.5 0.5 0.4 0.4"
    assert not (dataset_root / "train").exists()


def test_plan_segmentation_build_rejects_symlinked_sam3_output_parent(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "source_split"
    _write_test_image(dataset_root / "train" / "images" / "img.jpg")
    (dataset_root / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "train" / "labels" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "labelmap.txt").write_text("building\n", encoding="utf-8")
    outside = tmp_path / "outside_sam3_outputs"
    outside.mkdir()
    link_parent = tmp_path / "linked_parent"
    try:
        link_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    entry = {
        "id": "ds",
        "label": "ds",
        "dataset_root": str(dataset_root),
        "storage_mode": "managed",
        "yolo_layout": "split",
        "yolo_ready": True,
        "classes": ["building"],
    }
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [entry])
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", link_parent / "sam3_outputs")

    with pytest.raises(api.HTTPException) as exc_info:
        api._plan_segmentation_build(
            api.SegmentationBuildRequest(source_dataset_id="ds", output_name="ds_seg")
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "segmentation_output_path_invalid"
    assert list(outside.iterdir()) == []


def test_prepare_segmentation_output_root_rejects_symlink_swap(
    tmp_path, monkeypatch
) -> None:
    sam3_root = tmp_path / "sam3_outputs"
    sam3_root.mkdir()
    outside = tmp_path / "outside_output"
    outside.mkdir()
    try:
        (sam3_root / "ds_seg").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "SAM3_DATASET_ROOT", sam3_root)

    with pytest.raises(api.HTTPException) as exc_info:
        api._prepare_segmentation_output_root(
            {"id": "ds_seg"},
            {"dataset_root": str(sam3_root / "ds_seg")},
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "segmentation_output_path_invalid"
    assert list(outside.iterdir()) == []


def test_write_segmentation_output_labelmap_skips_source_symlink_escape(
    tmp_path,
) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    dataset_root = tmp_path / "source"
    dataset_root.mkdir()
    outside = tmp_path / "outside_labelmap.txt"
    try:
        (dataset_root / "labelmap.txt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    api._write_segmentation_output_labelmap(
        output_root=output_root,
        dataset_root=dataset_root,
        classes=["building", "vehicle"],
    )

    assert (output_root / "labelmap.txt").read_text(encoding="utf-8") == "building\nvehicle\n"
    assert not outside.exists()


def test_segmentation_output_text_write_is_atomic_over_symlink_leaves(
    tmp_path, monkeypatch
) -> None:
    class FixedUUID:
        hex = "deadbeef000000000000000000000000"

    output_root = tmp_path / "output"
    label_path = output_root / "train" / "labels" / "img.txt"
    label_path.parent.mkdir(parents=True)
    tmp_path_link = label_path.with_suffix(f"{label_path.suffix}.{FixedUUID.hex}.tmp")
    outside_tmp = tmp_path / "outside_tmp.txt"
    outside_final = tmp_path / "outside_final.txt"
    outside_tmp.write_text("external tmp", encoding="utf-8")
    outside_final.write_text("external final", encoding="utf-8")
    try:
        tmp_path_link.symlink_to(outside_tmp)
        label_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api.uuid, "uuid4", lambda: FixedUUID())

    api._segmentation_write_text_within_root(
        label_path,
        output_root,
        "0 0.1 0.1 0.2 0.1 0.2 0.2\n",
    )

    assert not tmp_path_link.exists()
    assert not label_path.is_symlink()
    assert label_path.read_text(encoding="utf-8") == "0 0.1 0.1 0.2 0.1 0.2 0.2\n"
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    assert outside_final.read_text(encoding="utf-8") == "external final"


def test_register_path_dedupes_existing_linked_entry(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "linked_ds"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labelmap.txt").write_text("car\n", encoding="utf-8")

    monkeypatch.setattr(api, "DATASET_LINK_ROOTS", [tmp_path.resolve()])
    existing = {
        "id": "existing",
        "label": "Existing",
        "storage_mode": "linked",
        "linked_root": str(dataset_root.resolve()),
        "signature": api._compute_dir_signature_impl(dataset_root),
    }
    monkeypatch.setattr(api, "_list_all_datasets", lambda: [existing])

    out = api.register_dataset_path(
        str(dataset_root), None, None, None, None, force_new=False, strict=True
    )
    assert out["id"] == "existing"


def test_register_path_rejects_symlinked_registry_root(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "linked_ds"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labelmap.txt").write_text("car\n", encoding="utf-8")
    outside = tmp_path / "outside_registry"
    outside.mkdir()
    registry_root = tmp_path / "registry"
    try:
        registry_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    monkeypatch.setattr(api, "DATASET_LINK_ROOTS", [tmp_path.resolve()])
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)

    with pytest.raises(api.HTTPException) as exc_info:
        api.register_dataset_path(
            str(dataset_root),
            "linked_ds",
            None,
            None,
            None,
            force_new=True,
            strict=True,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_register_target_invalid"
    assert list(outside.iterdir()) == []


def test_register_path_rejects_symlinked_labelmap_before_registry_write(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_ds"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside_labelmap.txt"
    outside.write_text("secret\n", encoding="utf-8")
    try:
        (dataset_root / "labelmap.txt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    registry_root = tmp_path / "registry"
    monkeypatch.setattr(api, "DATASET_LINK_ROOTS", [tmp_path.resolve()])
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)

    with pytest.raises(api.HTTPException) as exc_info:
        api.register_dataset_path(
            str(dataset_root),
            "linked_ds",
            None,
            None,
            None,
            force_new=True,
            strict=True,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "labelmap_path_forbidden"
    assert not registry_root.exists()
    assert outside.read_text(encoding="utf-8") == "secret\n"


def test_register_path_rejects_target_symlink_without_target_write(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_ds"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labelmap.txt").write_text("car\n", encoding="utf-8")
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    outside = tmp_path / "outside_registry"
    outside.mkdir()
    outside_target = outside / "ghost"
    try:
        (registry_root / "linked_ds").symlink_to(outside_target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    monkeypatch.setattr(api, "DATASET_LINK_ROOTS", [tmp_path.resolve()])
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)

    with pytest.raises(api.HTTPException) as exc_info:
        api.register_dataset_path(
            str(dataset_root),
            "linked_ds",
            None,
            None,
            None,
            force_new=True,
            strict=True,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "dataset_register_target_invalid"
    assert not outside_target.exists()


def test_register_path_rolls_back_when_metadata_write_fails(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "linked_ds"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labelmap.txt").write_text("car\n", encoding="utf-8")
    registry_root = tmp_path / "registry"
    monkeypatch.setattr(api, "DATASET_LINK_ROOTS", [tmp_path.resolve()])
    monkeypatch.setattr(api, "DATASET_REGISTRY_ROOT", registry_root)

    def fail_metadata_write(*_args, **_kwargs):
        raise api.HTTPException(status_code=400, detail="metadata_write_failed")

    monkeypatch.setattr(api, "_write_dataset_metadata_json", fail_metadata_write)

    with pytest.raises(api.HTTPException) as exc_info:
        api.register_dataset_path(
            str(dataset_root),
            "linked_ds",
            None,
            None,
            None,
            force_new=True,
            strict=True,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "metadata_write_failed"
    assert not (registry_root / "linked_ds").exists()
    assert dataset_root.exists()


def test_open_path_strict_requires_labelmap(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "bad_ds"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "DATASET_LINK_ROOTS", [tmp_path.resolve()])

    with pytest.raises(api.HTTPException) as exc:
        api.open_dataset_path(str(dataset_root), strict=True)
    assert exc.value.status_code == 400
    assert exc.value.detail == "dataset_shape_missing_labelmap"


def test_open_path_strict_rejects_symlinked_labelmap(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "bad_ds"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside_labelmap.txt"
    outside.write_text("secret\n", encoding="utf-8")
    try:
        (dataset_root / "labelmap.txt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "DATASET_LINK_ROOTS", [tmp_path.resolve()])

    with pytest.raises(api.HTTPException) as exc:
        api.open_dataset_path(str(dataset_root), strict=True)

    assert exc.value.status_code == 400
    assert exc.value.detail == "labelmap_path_forbidden"
