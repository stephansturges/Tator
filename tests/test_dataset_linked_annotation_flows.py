from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import pytest

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
        },
    )

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
