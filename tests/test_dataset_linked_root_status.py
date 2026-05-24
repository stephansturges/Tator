from __future__ import annotations

import json
from pathlib import Path

import pytest

from services import datasets as datasets_service


def _load_registry_meta(path: Path):
    meta_path = path / "dataset_meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _list_entries(registry_root: Path):
    return datasets_service._list_all_datasets_impl(
        prefer_registry=True,
        dataset_registry_root=registry_root,
        sam3_dataset_root=registry_root / "_empty_sam3",
        qwen_dataset_root=registry_root / "_empty_qwen",
        load_registry_meta_fn=_load_registry_meta,
        load_sam3_meta_fn=lambda _path: None,
        load_qwen_meta_fn=lambda _path: None,
        coerce_meta_fn=lambda path, raw_meta, source: datasets_service._coerce_dataset_metadata_impl(
            path,
            raw_meta,
            source,
        ),
        yolo_labels_have_polygons_fn=lambda _path: False,
        convert_qwen_dataset_to_coco_fn=lambda _path: (_ for _ in ()).throw(
            RuntimeError("unexpected_convert_qwen")
        ),
        convert_coco_dataset_to_yolo_fn=lambda _path: (_ for _ in ()).throw(
            RuntimeError("unexpected_convert_coco")
        ),
        load_dataset_glossary_fn=lambda _path: "",
        glossary_preview_fn=lambda glossary, _labels: glossary,
        count_caption_labels_fn=lambda _path: (0, False),
        count_dataset_images_fn=lambda _path: 0,
        linked_root_allowed_fn=lambda _path: True,
    )


def test_linked_root_status_missing_for_broken_link(tmp_path) -> None:
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = registry_root / "linked_missing"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    missing_root = tmp_path / "does_not_exist"
    (dataset_dir / "dataset_meta.json").write_text(
        json.dumps(
            {
                "id": "linked_missing",
                "label": "linked_missing",
                "storage_mode": "linked",
                "linked_root": str(missing_root),
            }
        ),
        encoding="utf-8",
    )

    entries = _list_entries(registry_root)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["id"] == "linked_missing"
    assert entry["storage_mode"] == "linked"
    assert entry["linked_root_status"] == "missing"
    assert entry["linked_root"] == str(missing_root)


def test_linked_root_status_ok_for_available_link(tmp_path) -> None:
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    linked_source = tmp_path / "linked_src"
    (linked_source / "images").mkdir(parents=True, exist_ok=True)
    (linked_source / "labels").mkdir(parents=True, exist_ok=True)
    (linked_source / "labelmap.txt").write_text("car\n", encoding="utf-8")

    dataset_dir = registry_root / "linked_ok"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset_meta.json").write_text(
        json.dumps(
            {
                "id": "linked_ok",
                "label": "linked_ok",
                "storage_mode": "linked",
                "linked_root": str(linked_source),
                "source": "registry",
            }
        ),
        encoding="utf-8",
    )

    entries = _list_entries(registry_root)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["id"] == "linked_ok"
    assert entry["storage_mode"] == "linked"
    assert entry["linked_root_status"] == "ok"
    assert entry["linked_root"] == str(linked_source.resolve())
    assert entry["dataset_root"] == str(linked_source.resolve())
    assert entry["yolo_ready"] is True


def test_linked_root_status_not_allowlisted_does_not_inspect_source(tmp_path) -> None:
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    linked_source = tmp_path / "outside_linked_src"
    (linked_source / "images").mkdir(parents=True, exist_ok=True)
    (linked_source / "labels").mkdir(parents=True, exist_ok=True)
    (linked_source / "labelmap.txt").write_text("car\n", encoding="utf-8")

    dataset_dir = registry_root / "linked_blocked"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset_meta.json").write_text(
        json.dumps(
            {
                "id": "linked_blocked",
                "label": "linked_blocked",
                "storage_mode": "linked",
                "linked_root": str(linked_source),
                "source": "registry",
            }
        ),
        encoding="utf-8",
    )

    entries = datasets_service._list_all_datasets_impl(
        prefer_registry=True,
        dataset_registry_root=registry_root,
        sam3_dataset_root=registry_root / "_empty_sam3",
        qwen_dataset_root=registry_root / "_empty_qwen",
        load_registry_meta_fn=_load_registry_meta,
        load_sam3_meta_fn=lambda _path: None,
        load_qwen_meta_fn=lambda _path: None,
        coerce_meta_fn=lambda path, raw_meta, source: datasets_service._coerce_dataset_metadata_impl(
            path,
            raw_meta,
            source,
        ),
        yolo_labels_have_polygons_fn=lambda _path: False,
        convert_qwen_dataset_to_coco_fn=lambda _path: (_ for _ in ()).throw(
            RuntimeError("blocked_source_must_not_convert_qwen")
        ),
        convert_coco_dataset_to_yolo_fn=lambda _path: (_ for _ in ()).throw(
            RuntimeError("blocked_source_must_not_convert_coco")
        ),
        load_dataset_glossary_fn=lambda _path: "",
        glossary_preview_fn=lambda glossary, _labels: glossary,
        count_caption_labels_fn=lambda path: (_ for _ in ()).throw(
            RuntimeError(f"blocked_source_must_not_count_captions:{path}")
        )
        if path == linked_source
        else (0, False),
        count_dataset_images_fn=lambda path: (_ for _ in ()).throw(
            RuntimeError(f"blocked_source_must_not_count_images:{path}")
        )
        if path == linked_source
        else 0,
        linked_root_allowed_fn=lambda _path: False,
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry["id"] == "linked_blocked"
    assert entry["storage_mode"] == "linked"
    assert entry["linked_root_status"] == "not_allowlisted"
    assert entry["linked_root"] == str(linked_source.resolve())
    assert entry["dataset_root"] != str(linked_source.resolve())
    assert entry["yolo_ready"] is False


def test_linked_registry_labelmap_overrides_source_labelmap(tmp_path) -> None:
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    linked_source = tmp_path / "linked_src"
    (linked_source / "images").mkdir(parents=True, exist_ok=True)
    (linked_source / "labels").mkdir(parents=True, exist_ok=True)
    (linked_source / "labelmap.txt").write_text("car\n", encoding="utf-8")

    dataset_dir = registry_root / "linked_overlay"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    registry_labelmap = dataset_dir / "labelmap.txt"
    registry_labelmap.write_text("car\ntruck\n", encoding="utf-8")
    (dataset_dir / "dataset_meta.json").write_text(
        json.dumps(
            {
                "id": "linked_overlay",
                "label": "linked_overlay",
                "storage_mode": "linked",
                "linked_root": str(linked_source),
                "source": "registry",
                "classes": ["car", "truck"],
                "yolo_labelmap_path": str(registry_labelmap),
            }
        ),
        encoding="utf-8",
    )

    entries = _list_entries(registry_root)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["classes"] == ["car", "truck"]
    assert entry["yolo_labelmap_path"] == str(registry_labelmap)
    assert entry["yolo_ready"] is True


def test_linked_listing_does_not_auto_convert_or_backfill_source(tmp_path) -> None:
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    linked_source = tmp_path / "linked_qwen_source"
    (linked_source / "train").mkdir(parents=True, exist_ok=True)
    (linked_source / "val").mkdir(parents=True, exist_ok=True)
    (linked_source / "metadata.json").write_text(
        json.dumps({"id": "source_qwen", "classes": ["car"]}),
        encoding="utf-8",
    )
    (linked_source / "train" / "annotations.jsonl").write_text("", encoding="utf-8")
    (linked_source / "val" / "annotations.jsonl").write_text("", encoding="utf-8")

    dataset_dir = registry_root / "linked_qwen"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset_meta.json").write_text(
        json.dumps(
            {
                "id": "linked_qwen",
                "label": "linked_qwen",
                "storage_mode": "linked",
                "linked_root": str(linked_source),
                "source": "registry",
                "classes": ["car"],
                "labelmap_glossary": "car: vehicle",
            }
        ),
        encoding="utf-8",
    )

    source_sam3_reads: list[Path] = []

    def load_sam3_meta(path: Path):
        source_sam3_reads.append(path)
        return None

    entries = datasets_service._list_all_datasets_impl(
        prefer_registry=True,
        dataset_registry_root=registry_root,
        sam3_dataset_root=registry_root / "_empty_sam3",
        qwen_dataset_root=registry_root / "_empty_qwen",
        load_registry_meta_fn=_load_registry_meta,
        load_sam3_meta_fn=load_sam3_meta,
        load_qwen_meta_fn=lambda path: json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if (path / "metadata.json").exists()
        else None,
        coerce_meta_fn=lambda path, raw_meta, source: datasets_service._coerce_dataset_metadata_impl(
            path,
            raw_meta,
            source,
        ),
        yolo_labels_have_polygons_fn=lambda _path: False,
        convert_qwen_dataset_to_coco_fn=lambda _path: (_ for _ in ()).throw(
            RuntimeError("linked_source_must_not_convert_qwen")
        ),
        convert_coco_dataset_to_yolo_fn=lambda _path: (_ for _ in ()).throw(
            RuntimeError("linked_source_must_not_convert_coco")
        ),
        load_dataset_glossary_fn=lambda _path: (_ for _ in ()).throw(
            RuntimeError("linked_source_must_not_backfill_glossary")
        ),
        glossary_preview_fn=lambda glossary, _labels: glossary,
        count_caption_labels_fn=lambda _path: (0, False),
        count_dataset_images_fn=lambda _path: 0,
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry["id"] == "linked_qwen"
    assert entry["storage_mode"] == "linked"
    assert entry["linked_root_status"] == "ok"
    assert entry["qwen_ready"] is False
    assert entry["coco_ready"] is False
    assert entry["glossary_preview"] == "car: vehicle"
    assert source_sam3_reads == []
    assert not (linked_source / "labelmap.txt").exists()
    assert not (linked_source / "sam3_dataset.json").exists()
    assert not (linked_source / "train" / "_annotations.coco.json").exists()
    assert not (linked_source / "train" / "labels").exists()


def test_dataset_listing_skips_symlinked_registry_record(tmp_path) -> None:
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    outside_record = tmp_path / "outside_record"
    outside_record.mkdir(parents=True, exist_ok=True)
    (outside_record / "dataset_meta.json").write_text(
        json.dumps(
            {
                "id": "escaped",
                "label": "escaped",
                "storage_mode": "managed",
            }
        ),
        encoding="utf-8",
    )
    try:
        (registry_root / "linked_record").symlink_to(outside_record, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    assert _list_entries(registry_root) == []
