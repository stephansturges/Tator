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
