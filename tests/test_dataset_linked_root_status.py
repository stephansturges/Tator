from __future__ import annotations

import json
from pathlib import Path

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
