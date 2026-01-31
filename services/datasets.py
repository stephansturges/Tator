from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple

from utils.glossary import _normalize_labelmap_glossary


def _load_qwen_labelmap(dataset_root: Path, *, load_qwen_meta, collect_labels) -> list[str]:
    meta = load_qwen_meta(dataset_root) or {}
    classes = [str(cls).strip() for cls in meta.get("classes", []) if str(cls).strip()]
    if classes:
        return classes
    labels = set()
    for split in ("train", "val"):
        labels.update(collect_labels(dataset_root / split / "annotations.jsonl"))
    return sorted(labels)


def _load_dataset_glossary(dataset_root: Path, *, load_sam3_meta, load_qwen_meta) -> str:
    sam_meta = load_sam3_meta(dataset_root) or {}
    qwen_meta = load_qwen_meta(dataset_root) or {}
    raw = sam_meta.get("labelmap_glossary") or qwen_meta.get("labelmap_glossary")
    return _normalize_labelmap_glossary(raw)


def _agent_load_labelmap_meta(
    dataset_id: Optional[str],
    *,
    active_label_list: Optional[Sequence[str]] = None,
    resolve_dataset,
    discover_yolo_labelmap,
    load_qwen_labelmap,
    load_sam3_meta,
    load_qwen_meta,
    normalize_glossary,
    default_glossary_fn,
    collect_labels,
) -> Tuple[list[str], str]:
    labelmap: list[str] = []
    glossary = ""
    if dataset_id:
        dataset_root = resolve_dataset(dataset_id)
        yolo_labels = discover_yolo_labelmap(dataset_root)
        if yolo_labels:
            labelmap = list(yolo_labels)
        else:
            labelmap = load_qwen_labelmap(
                dataset_root,
                load_qwen_meta=load_qwen_meta,
                collect_labels=collect_labels,
            )
        meta = load_sam3_meta(dataset_root) or load_qwen_meta(dataset_root) or {}
        glossary = normalize_glossary(
            meta.get("labelmap_glossary") or meta.get("glossary") or meta.get("labelmap_ontology")
        )
        if not glossary and labelmap:
            glossary = default_glossary_fn(labelmap)
        return labelmap, glossary
    if active_label_list:
        labelmap = [str(lbl) for lbl in active_label_list]
    if not glossary and labelmap:
        glossary = default_glossary_fn(labelmap)
    return labelmap, glossary


def _detect_yolo_layout_impl(dataset_root: Path) -> dict:
    labelmap_path = dataset_root / "labelmap.txt"
    train_images = dataset_root / "train" / "images"
    train_labels = dataset_root / "train" / "labels"
    root_images = dataset_root / "images"
    root_labels = dataset_root / "labels"
    yolo_images_dir: Optional[str] = None
    yolo_labels_dir: Optional[str] = None
    yolo_layout: Optional[str] = None
    if labelmap_path.exists():
        if train_images.exists() and train_labels.exists():
            yolo_images_dir = str(train_images)
            yolo_labels_dir = str(train_labels)
            yolo_layout = "split"
        elif root_images.exists() and root_labels.exists():
            yolo_images_dir = str(root_images)
            yolo_labels_dir = str(root_labels)
            yolo_layout = "flat"
    yolo_ready = bool(labelmap_path.exists() and yolo_images_dir and yolo_labels_dir)
    return {
        "yolo_ready": yolo_ready,
        "yolo_images_dir": yolo_images_dir,
        "yolo_labels_dir": yolo_labels_dir,
        "yolo_labelmap_path": str(labelmap_path) if labelmap_path.exists() else None,
        "yolo_layout": yolo_layout,
    }


def _yolo_labels_have_polygons_impl(
    labels_dir: Optional[Path],
    *,
    max_files: int = 200,
    max_lines: int = 2000,
) -> bool:
    if not labels_dir or not labels_dir.exists():
        return False
    checked_files = 0
    checked_lines = 0
    for label_path in labels_dir.rglob("*.txt"):
        checked_files += 1
        if checked_files > max_files:
            break
        try:
            with label_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if checked_lines >= max_lines:
                        return False
                    stripped = line.strip()
                    if not stripped:
                        continue
                    checked_lines += 1
                    parts = stripped.split()
                    # class + 4 bbox coords is 5 tokens; polygons add more.
                    if len(parts) > 5:
                        return True
        except Exception:
            continue
    return False
