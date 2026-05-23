from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import pytest
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
    meta_path = record_root / api.DATASET_META_NAME

    assert out["id"] == "saved_linked"
    assert overlay_label.exists()
    assert overlay_label.read_text(encoding="utf-8").strip() == "0 0.5 0.5 0.2 0.2"
    assert overlay_text.exists()
    assert overlay_text.read_text(encoding="utf-8").strip() == "car in frame"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["annotation_status"] == "in_progress"
    assert meta["annotation_notes"] == "notes"


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


def _entry_for_annotation(tmp_path: Path) -> dict:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
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


def test_transient_meta_patch_revalidates_dataset_root_before_labelmap_write(
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

    with pytest.raises(api.HTTPException) as exc:
        api.patch_transient_annotation_meta(
            session_id,
            {
                "session_id": "sess-lock",
                "labelmap": ["car", "truck"],
            },
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "dataset_path_not_allowlisted"
    assert not (source_root / "labelmap.txt").exists()
    with api.DATASET_TRANSIENT_LOCK:
        session = api.DATASET_TRANSIENT_SESSIONS[session_id]
        assert session["classes"] == ["old"]
        api.DATASET_TRANSIENT_SESSIONS.pop(session_id, None)


def test_transient_meta_patch_writes_labelmap_after_revalidation(
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
    monkeypatch.setattr(api, "_validate_linked_dataset_path", lambda _path: source_root)

    out = api.patch_transient_annotation_meta(
        session_id,
        {
            "session_id": "sess-lock",
            "labelmap": ["car", "truck"],
        },
    )

    assert out["labelmap"] == ["car", "truck"]
    assert (source_root / "labelmap.txt").read_text(encoding="utf-8") == "car\ntruck\n"
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


def test_open_path_strict_requires_labelmap(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "bad_ds"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "DATASET_LINK_ROOTS", [tmp_path.resolve()])

    with pytest.raises(api.HTTPException) as exc:
        api.open_dataset_path(str(dataset_root), strict=True)
    assert exc.value.status_code == 400
    assert exc.value.detail == "dataset_shape_missing_labelmap"
