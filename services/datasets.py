from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, List

from fastapi import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR
from PIL import Image

from utils.glossary import _normalize_labelmap_glossary
from utils.io import _compute_dir_signature
from utils.labels import _load_labelmap_file
from utils.coco import (
    _coco_has_invalid_image_refs_impl,
    _coco_missing_segmentation_impl,
    _ensure_coco_info_fields_impl,
    _write_coco_annotations_impl,
)
from utils.image import (
    _image_path_for_label_impl,
    _resolve_coco_image_path_impl,
    _label_relpath_for_image_impl,
)

logger = logging.getLogger(__name__)


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


def _ensure_qwen_dataset_signature_impl(
    dataset_dir: Path,
    metadata: Dict[str, Any],
    *,
    compute_dir_signature_fn,
    persist_metadata_fn,
) -> Tuple[Dict[str, Any], str]:
    signature = metadata.get("signature")
    if signature:
        return metadata, str(signature)
    signature = compute_dir_signature_fn(dataset_dir)
    metadata["signature"] = signature
    persist_metadata_fn(dataset_dir, metadata)
    return metadata, signature


def _load_registry_dataset_metadata_impl(dataset_dir: Path, *, load_json_metadata_fn, meta_name: str) -> Optional[Dict[str, Any]]:
    return load_json_metadata_fn(dataset_dir / meta_name)


def _persist_dataset_metadata_impl(
    dataset_dir: Path,
    metadata: Dict[str, Any],
    *,
    meta_name: str,
    logger=None,
) -> None:
    meta_path = dataset_dir / meta_name
    try:
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
    except Exception as exc:
        if logger is not None:
            logger.warning("Failed to write dataset metadata for %s: %s", dataset_dir, exc)


def _coerce_dataset_metadata_impl(
    dataset_dir: Path,
    raw_meta: Optional[Dict[str, Any]],
    source: str,
    *,
    dataset_context_key: str = "dataset_context",
    compute_dir_signature_fn=None,
    persist_metadata_fn=None,
) -> Dict[str, Any]:
    meta = dict(raw_meta or {})
    updated = False
    if "id" not in meta:
        meta["id"] = dataset_dir.name
        updated = True
    if "label" not in meta:
        meta["label"] = meta["id"]
        updated = True
    dataset_type = meta.get("type") or meta.get("dataset_type") or "bbox"
    meta["type"] = dataset_type
    if "classes" not in meta:
        meta["classes"] = []
        updated = True
    if "context" not in meta and meta.get(dataset_context_key):
        meta["context"] = meta.get(dataset_context_key) or ""
        updated = True
    if "created_at" not in meta:
        meta["created_at"] = dataset_dir.stat().st_mtime
        updated = True
    if "source" not in meta:
        meta["source"] = source
        updated = True
    signature = meta.get("signature")
    if not signature and compute_dir_signature_fn is not None:
        signature = compute_dir_signature_fn(dataset_dir)
        meta["signature"] = signature
        updated = True
    if source == "registry" and updated and persist_metadata_fn is not None:
        persist_metadata_fn(dataset_dir, meta)
    return meta


def _load_qwen_dataset_metadata_impl(
    dataset_dir: Path,
    *,
    meta_name: str,
    load_json_metadata_fn,
) -> Optional[Dict[str, Any]]:
    return load_json_metadata_fn(dataset_dir / meta_name)


def _persist_qwen_dataset_metadata_impl(
    dataset_dir: Path,
    metadata: Dict[str, Any],
    *,
    meta_name: str,
    write_qwen_metadata_fn,
) -> None:
    write_qwen_metadata_fn(dataset_dir / meta_name, metadata)


def _load_sam3_dataset_metadata_impl(
    dataset_dir: Path,
    *,
    meta_name: str,
    load_json_metadata_fn,
    persist_metadata_fn,
) -> Optional[Dict[str, Any]]:
    meta_path = dataset_dir / meta_name
    data = load_json_metadata_fn(meta_path)
    if not data:
        return None
    updated = False
    if "id" not in data:
        data["id"] = dataset_dir.name
        updated = True
    if "type" not in data:
        data["type"] = "bbox"
        updated = True
    if updated:
        persist_metadata_fn(dataset_dir, data)
    return data


def _persist_sam3_dataset_metadata_impl(
    dataset_dir: Path,
    metadata: Dict[str, Any],
    *,
    meta_name: str,
    logger=None,
) -> None:
    meta_path = dataset_dir / meta_name
    try:
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
    except Exception as exc:
        if logger is not None:
            logger.warning("Failed to write SAM3 dataset metadata for %s: %s", dataset_dir, exc)


def _count_dataset_images_impl(dataset_root: Path, *, iter_images_fn) -> int:
    train_dir = dataset_root / "train" / "images"
    val_dir = dataset_root / "val" / "images"
    if train_dir.exists() or val_dir.exists():
        return len(iter_images_fn(train_dir)) + len(iter_images_fn(val_dir))
    root_images = dataset_root / "images"
    if root_images.exists():
        return len(iter_images_fn(root_images))
    return len(iter_images_fn(dataset_root))


def _count_caption_labels_impl(
    dataset_root: Path,
    *,
    label_dirs: Sequence[str] = ("text_labels", "captions", "caption_labels"),
    allowed_exts: Sequence[str] = (".txt", ".json", ".jsonl"),
) -> Tuple[int, bool]:
    total = 0
    present = False
    allowed = {ext.lower() for ext in allowed_exts}
    for name in label_dirs:
        path = dataset_root / name
        if not path.exists():
            continue
        present = True
        if path.is_dir():
            for entry in path.rglob("*"):
                if not entry.is_file():
                    continue
                if allowed and entry.suffix.lower() not in allowed:
                    continue
                total += 1
        elif path.is_file():
            if not allowed or path.suffix.lower() in allowed:
                total += 1
    jsonl_path = dataset_root / "captions.jsonl"
    if jsonl_path.exists():
        present = True
        try:
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for _ in handle:
                    total += 1
        except Exception:
            pass
    return total, present


def _list_all_datasets_impl(
    *,
    prefer_registry: bool,
    dataset_registry_root: Path,
    sam3_dataset_root: Path,
    qwen_dataset_root: Path,
    load_registry_meta_fn,
    load_sam3_meta_fn,
    load_qwen_meta_fn,
    coerce_meta_fn,
    yolo_labels_have_polygons_fn,
    convert_qwen_dataset_to_coco_fn,
    convert_coco_dataset_to_yolo_fn,
    load_dataset_glossary_fn,
    glossary_preview_fn,
    count_caption_labels_fn,
    count_dataset_images_fn,
    logger=None,
) -> list[Dict[str, Any]]:
    entries: list[Dict[str, Any]] = []
    seen: Dict[str, Tuple[int, str]] = {}
    sources = [
        ("registry", dataset_registry_root, load_registry_meta_fn),
        ("sam3", sam3_dataset_root, load_sam3_meta_fn),
        ("qwen", qwen_dataset_root, load_qwen_meta_fn),
    ]
    for source, root, loader in sources:
        if not root.exists():
            continue
        for path in root.iterdir():
            if not path.is_dir():
                continue
            raw_meta = loader(path)
            if not raw_meta and source == "registry":
                raw_meta = load_sam3_meta_fn(path) or load_qwen_meta_fn(path)
            if not raw_meta:
                continue
            meta = coerce_meta_fn(path, raw_meta, source)
            sam3_meta = load_sam3_meta_fn(path) if source != "sam3" else meta
            coco_train = None
            coco_val = None
            coco_ready = False
            if sam3_meta:
                coco_train = sam3_meta.get("coco_train_json")
                coco_val = sam3_meta.get("coco_val_json")
                coco_ready = bool(coco_train and coco_val)
            labelmap_path = path / "labelmap.txt"
            train_images = path / "train" / "images"
            train_labels = path / "train" / "labels"
            root_images = path / "images"
            root_labels = path / "labels"
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
            qwen_meta = load_qwen_meta_fn(path)
            qwen_ready = bool(
                qwen_meta
                and (path / "train" / "annotations.jsonl").exists()
                and (path / "val" / "annotations.jsonl").exists()
            )
            if not yolo_ready:
                try:
                    coco_exists = (path / "train" / "_annotations.coco.json").exists() or (
                        path / "val" / "_annotations.coco.json"
                    ).exists()
                    if not coco_exists and qwen_ready:
                        convert_qwen_dataset_to_coco_fn(path)
                        coco_exists = True
                    if coco_exists:
                        convert_coco_dataset_to_yolo_fn(path)
                        labelmap_path = path / "labelmap.txt"
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
                            if yolo_ready and not meta.get("classes"):
                                try:
                                    with labelmap_path.open("r", encoding="utf-8") as handle:
                                        meta["classes"] = [line.strip() for line in handle if line.strip()]
                                except Exception:
                                    pass
                except Exception as exc:
                    if logger is not None:
                        logger.warning("Failed to auto-convert COCO to YOLO for %s: %s", path, exc)
            dataset_format = "unknown"
            dataset_type = meta.get("type", "bbox")
            if yolo_ready:
                dataset_format = "yolo"
            elif qwen_ready:
                dataset_format = "qwen"
            yolo_seg_ready = False
            if yolo_ready and yolo_labels_dir:
                yolo_seg_ready = yolo_labels_have_polygons_fn(Path(yolo_labels_dir))
                if yolo_seg_ready and dataset_type != "seg":
                    dataset_type = "seg"
            labelmap = list(meta.get("classes") or [])
            if not labelmap and labelmap_path.exists():
                try:
                    with labelmap_path.open("r", encoding="utf-8") as handle:
                        labelmap = [line.strip() for line in handle if line.strip()]
                except Exception:
                    labelmap = []
            glossary_text = load_dataset_glossary_fn(path)
            glossary_preview = glossary_preview_fn(glossary_text, labelmap)
            signature = meta.get("signature") or ""
            caption_count, caption_dir_present = count_caption_labels_fn(path)
            image_total = meta.get("image_count")
            if not image_total:
                image_total = count_dataset_images_fn(path)
            caption_percent = None
            if image_total and image_total > 0:
                caption_percent = (caption_count / image_total) * 100.0
            key = signature or meta["id"]
            entry = {
                "id": meta.get("id") or path.name,
                "label": meta.get("label") or path.name,
                "dataset_root": str(path),
                "created_at": meta.get("created_at") or path.stat().st_mtime,
                "image_count": meta.get("image_count"),
                "train_count": meta.get("train_count"),
                "val_count": meta.get("val_count"),
                "classes": meta.get("classes", []),
                "context": meta.get("context", "") or meta.get("dataset_context", ""),
                "signature": signature,
                "source": meta.get("source") or source,
                "type": dataset_type,
                "coco_ready": coco_ready,
                "coco_seg_ready": bool(coco_ready and dataset_type == "seg"),
                "coco_train_json": coco_train,
                "coco_val_json": coco_val,
                "format": dataset_format,
                "yolo_ready": yolo_ready,
                "yolo_seg_ready": yolo_seg_ready,
                "yolo_images_dir": yolo_images_dir,
                "yolo_labels_dir": yolo_labels_dir,
                "yolo_labelmap_path": str(labelmap_path) if labelmap_path.exists() else None,
                "yolo_layout": yolo_layout,
                "qwen_ready": qwen_ready,
                "qwen_train_count": qwen_meta.get("train_count") if qwen_meta else None,
                "qwen_val_count": qwen_meta.get("val_count") if qwen_meta else None,
                "caption_count": caption_count,
                "caption_dir": caption_dir_present,
                "caption_percent": caption_percent,
                "caption_total": image_total,
                "glossary_present": bool(str(glossary_text or "").strip()),
                "glossary_preview": glossary_preview,
            }
            existing = seen.get(key)
            if existing is not None:
                existing_idx, existing_origin = existing
                if prefer_registry:
                    if existing_origin == "registry":
                        continue
                    if source == "registry":
                        entries[existing_idx] = entry
                        seen[key] = (existing_idx, source)
                        continue
                continue
            seen[key] = (len(entries), source)
            entries.append(entry)
    entries.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return entries


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


def _extract_qwen_detections_from_payload_impl(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    detections = payload.get("detections")
    if isinstance(detections, list):
        return [d for d in detections if isinstance(d, dict)]
    conversations = payload.get("conversations")
    if not isinstance(conversations, list):
        return []
    for message in reversed(conversations):
        if not isinstance(message, dict):
            continue
        if message.get("from") != "gpt":
            continue
        value = message.get("value")
        if not isinstance(value, str):
            continue
        try:
            parsed = json.loads(value)
        except Exception:
            continue
        detections = parsed.get("detections")
        if isinstance(detections, list):
            return [det for det in detections if isinstance(det, dict)]
    return []


def _collect_labels_from_qwen_jsonl_impl(
    jsonl_path: Path,
    *,
    extract_detections_fn=_extract_qwen_detections_from_payload_impl,
) -> List[str]:
    labels: set[str] = set()
    if not jsonl_path.exists():
        return []
    try:
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                detections = extract_detections_fn(payload)
                for det in detections:
                    label = str(det.get("label", "")).strip()
                    if label:
                        labels.add(label)
    except Exception:
        return sorted(labels)
    return sorted(labels)


def _discover_yolo_labelmap_impl(
    dataset_root: Path,
    *,
    load_labelmap_file_fn,
) -> List[str]:
    for name in ("labelmap.txt", "classes.txt", "labels.txt"):
        candidate = dataset_root / name
        classes = load_labelmap_file_fn(candidate)
        if classes:
            return classes
    return []


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


def _resolve_dataset_entry_impl(dataset_id: str, *, list_all_datasets_fn) -> Optional[Dict[str, Any]]:
    cleaned = (dataset_id or "").strip()
    if not cleaned:
        return None
    for entry in list_all_datasets_fn():
        if cleaned in (entry.get("id"), entry.get("signature")):
            return entry
    return None


def _resolve_dataset_legacy_impl(
    dataset_id: str,
    *,
    qwen_root: Path,
    sam3_root: Path,
    registry_root: Path,
    http_exception_cls,
) -> Path:
    cleaned = (dataset_id or "").strip().replace("\\", "/")
    safe = re.sub(r"[^A-Za-z0-9._/-]", "_", cleaned)
    candidate_qwen = (qwen_root / safe).resolve()
    if str(candidate_qwen).startswith(str(qwen_root.resolve())) and candidate_qwen.exists():
        return candidate_qwen
    candidate_sam3 = (sam3_root / safe).resolve()
    if str(candidate_sam3).startswith(str(sam3_root.resolve())) and candidate_sam3.exists():
        return candidate_sam3
    candidate_registry = (registry_root / safe).resolve()
    if str(candidate_registry).startswith(str(registry_root.resolve())) and candidate_registry.exists():
        return candidate_registry
    raise http_exception_cls(status_code=404, detail="sam3_dataset_not_found")


def _resolve_sam3_or_qwen_dataset_impl(
    dataset_id: str,
    *,
    list_all_datasets_fn,
    resolve_dataset_legacy_fn,
) -> Path:
    cleaned = (dataset_id or "").strip()
    for entry in list_all_datasets_fn():
        if cleaned in (entry.get("id"), entry.get("signature")):
            path = Path(entry["dataset_root"]).resolve()
            if path.exists():
                return path
    return resolve_dataset_legacy_fn(dataset_id)


def _yolo_resolve_split_paths_impl(dataset_root: Path, layout: Optional[str]) -> Tuple[str, str]:
    if layout == "split":
        train_images = dataset_root / "train" / "images"
        val_images = dataset_root / "val" / "images"
        train_rel = str(train_images.relative_to(dataset_root))
        if val_images.exists():
            val_rel = str(val_images.relative_to(dataset_root))
        else:
            val_rel = train_rel
        return train_rel, val_rel
    images = dataset_root / "images"
    train_rel = str(images.relative_to(dataset_root))
    val_images = dataset_root / "val" / "images"
    if val_images.exists():
        val_rel = str(val_images.relative_to(dataset_root))
    else:
        val_rel = train_rel
    return train_rel, val_rel


def _compute_labelmap_hash_impl(categories: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    names: List[Tuple[int, str]] = []
    for idx, cat in enumerate(categories):
        try:
            cid = int(cat.get("id", idx))
        except Exception:
            cid = idx
        names.append((cid, str(cat.get("name", f"class_{cid}"))))
    names.sort(key=lambda c: c[0])
    labels = [name for _, name in names]
    try:
        digest = hashlib.sha256("|".join(labels).encode("utf-8")).hexdigest()[:12]
    except Exception:
        digest = "unknown"
    return digest, labels


def _compute_dataset_signature_impl(
    dataset_id: str,
    dataset_root: Path,
    images: Dict[int, Dict[str, Any]],
    categories: List[Dict[str, Any]],
) -> str:
    """
    Create a location-agnostic signature for portability:
    - dataset_id
    - counts of images/categories
    - hashes of category names (sorted)
    - hashes of image file names (sorted)
    """
    try:
        cat_names = [str(c.get("name", f"class_{idx}")) for idx, c in enumerate(categories)]
        cat_hash = hashlib.sha256("|".join(sorted(cat_names)).encode("utf-8")).hexdigest()[:12]
        file_names = [Path(info.get("file_name") or "").name for info in images.values() if info.get("file_name")]
        file_hash = hashlib.sha256("|".join(sorted(file_names)).encode("utf-8")).hexdigest()[:12]
        payload = f"{dataset_id}|{len(images)}|{len(categories)}|{cat_hash}|{file_hash}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    except Exception:
        return "unknown"


def _resolve_yolo_training_dataset_impl(
    payload,
    *,
    resolve_dataset_entry_fn,
    resolve_sam3_or_qwen_dataset_fn,
    compute_dir_signature_fn,
    sanitize_yolo_run_id_fn,
    detect_yolo_layout_fn,
    yolo_labels_have_polygons_fn,
    stable_hash_fn,
    yolo_cache_root: Path,
    http_exception_cls,
) -> Dict[str, Any]:
    task = (getattr(payload, "task", None) or "detect").lower().strip()
    dataset_id = (getattr(payload, "dataset_id", None) or "").strip()
    dataset_root: Optional[Path] = None
    entry: Optional[Dict[str, Any]] = None
    if dataset_id:
        entry = resolve_dataset_entry_fn(dataset_id)
        if entry and entry.get("dataset_root"):
            dataset_root = Path(entry["dataset_root"]).resolve()
        else:
            dataset_root = resolve_sam3_or_qwen_dataset_fn(dataset_id)
    elif getattr(payload, "dataset_root", None):
        dataset_root = Path(getattr(payload, "dataset_root")).expanduser().resolve()
    if not dataset_root or not dataset_root.exists():
        raise http_exception_cls(status_code=404, detail="dataset_not_found")
    dataset_signature = None
    if entry and entry.get("signature"):
        dataset_signature = str(entry.get("signature"))
    if not dataset_signature:
        dataset_signature = compute_dir_signature_fn(dataset_root)
    safe_name = sanitize_yolo_run_id_fn(entry.get("id") if entry else dataset_root.name)
    cache_key = stable_hash_fn([safe_name, dataset_signature, task])[:12]
    cache_root = (yolo_cache_root / f"{safe_name}_{cache_key}").resolve()
    cache_layout = detect_yolo_layout_fn(cache_root) if cache_root.exists() else None
    layout = detect_yolo_layout_fn(dataset_root)
    if cache_layout and cache_layout.get("yolo_ready"):
        prepared_root = cache_root
        yolo_ready = True
        yolo_images_dir = cache_layout.get("yolo_images_dir")
        yolo_labels_dir = cache_layout.get("yolo_labels_dir")
        yolo_labelmap_path = cache_layout.get("yolo_labelmap_path")
        yolo_layout = cache_layout.get("yolo_layout")
        source = "cache"
    else:
        prepared_root = dataset_root
        yolo_ready = bool(layout.get("yolo_ready"))
        yolo_images_dir = layout.get("yolo_images_dir")
        yolo_labels_dir = layout.get("yolo_labels_dir")
        yolo_labelmap_path = layout.get("yolo_labelmap_path")
        yolo_layout = layout.get("yolo_layout")
        source = "registry" if entry else "custom"
    yolo_seg_ready = False
    if yolo_ready and yolo_labels_dir:
        yolo_seg_ready = yolo_labels_have_polygons_fn(Path(yolo_labels_dir))
    return {
        "dataset_id": entry.get("id") if entry else dataset_root.name,
        "dataset_root": str(dataset_root),
        "prepared_root": str(prepared_root),
        "signature": dataset_signature,
        "task": task,
        "yolo_ready": yolo_ready,
        "yolo_seg_ready": yolo_seg_ready,
        "yolo_images_dir": yolo_images_dir,
        "yolo_labels_dir": yolo_labels_dir,
        "yolo_labelmap_path": yolo_labelmap_path,
        "yolo_layout": yolo_layout,
        "cache_root": str(cache_root),
        "cache_key": cache_key,
        "source": source,
    }


def _resolve_rfdetr_training_dataset_impl(
    payload,
    *,
    resolve_dataset_entry_fn,
    resolve_sam3_or_qwen_dataset_fn,
    load_sam3_meta_fn,
    detect_yolo_layout_fn,
    yolo_labels_have_polygons_fn,
    convert_yolo_dataset_to_coco_fn,
    convert_qwen_dataset_to_coco_fn,
    load_qwen_dataset_metadata_fn,
    ensure_coco_supercategory_fn,
    http_exception_cls,
) -> Dict[str, Any]:
    task = (getattr(payload, "task", None) or "detect").lower().strip()
    dataset_id = (getattr(payload, "dataset_id", None) or "").strip()
    dataset_root: Optional[Path] = None
    entry: Optional[Dict[str, Any]] = None
    if dataset_id:
        entry = resolve_dataset_entry_fn(dataset_id)
        if entry and entry.get("dataset_root"):
            dataset_root = Path(entry["dataset_root"]).resolve()
        else:
            dataset_root = resolve_sam3_or_qwen_dataset_fn(dataset_id)
    elif getattr(payload, "dataset_root", None):
        dataset_root = Path(getattr(payload, "dataset_root")).expanduser().resolve()
    if not dataset_root or not dataset_root.exists():
        raise http_exception_cls(status_code=404, detail="dataset_not_found")
    meta = load_sam3_meta_fn(dataset_root) or {}
    coco_train = meta.get("coco_train_json")
    coco_val = meta.get("coco_val_json")
    coco_ready = bool(coco_train and coco_val)
    dataset_type = (entry.get("type") if entry else None) or meta.get("type", "bbox")
    yolo_layout = detect_yolo_layout_fn(dataset_root)
    yolo_seg_ready = False
    if yolo_layout.get("yolo_ready") and yolo_layout.get("yolo_labels_dir"):
        yolo_seg_ready = yolo_labels_have_polygons_fn(Path(yolo_layout["yolo_labels_dir"]))
        if yolo_seg_ready and dataset_type != "seg":
            dataset_type = "seg"
    if task == "segment" and dataset_type != "seg":
        raise http_exception_cls(status_code=400, detail="rfdetr_seg_requires_polygons")
    if not coco_ready:
        if entry and entry.get("yolo_ready"):
            meta = convert_yolo_dataset_to_coco_fn(dataset_root)
        elif entry and entry.get("qwen_ready"):
            meta = convert_qwen_dataset_to_coco_fn(dataset_root)
        else:
            layout = detect_yolo_layout_fn(dataset_root)
            if layout.get("yolo_ready"):
                meta = convert_yolo_dataset_to_coco_fn(dataset_root)
            elif load_qwen_dataset_metadata_fn(dataset_root):
                meta = convert_qwen_dataset_to_coco_fn(dataset_root)
            else:
                raise http_exception_cls(status_code=400, detail="dataset_not_ready")
        coco_train = meta.get("coco_train_json")
        coco_val = meta.get("coco_val_json")
        coco_ready = bool(coco_train and coco_val)
        dataset_type = meta.get("type", dataset_type)
    if not coco_ready:
        raise http_exception_cls(status_code=400, detail="dataset_not_ready")
    if coco_train:
        ensure_coco_supercategory_fn(Path(coco_train))
    if coco_val:
        ensure_coco_supercategory_fn(Path(coco_val))
    dataset_label = entry.get("label") if entry else meta.get("label") or dataset_root.name
    return {
        "dataset_id": entry.get("id") if entry else dataset_root.name,
        "dataset_root": str(dataset_root),
        "dataset_label": dataset_label,
        "task": task,
        "coco_train_json": coco_train,
        "coco_val_json": coco_val,
        "type": dataset_type,
    }


def _convert_yolo_dataset_to_coco_impl(dataset_root: Path) -> Dict[str, Any]:
    dataset_root = dataset_root.resolve()
    train_images = dataset_root / "train" / "images"
    train_labels = dataset_root / "train" / "labels"
    val_images = dataset_root / "val" / "images"
    val_labels = dataset_root / "val" / "labels"
    for path in (train_images, train_labels):
        if not path.exists():
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_yolo_split_missing")
    has_val_images = val_images.exists()
    has_val_labels = val_labels.exists()
    if has_val_images != has_val_labels:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_yolo_val_split_incomplete")
    if not has_val_images:
        val_images.mkdir(parents=True, exist_ok=True)
        val_labels.mkdir(parents=True, exist_ok=True)

    labelmap = _discover_yolo_labelmap_impl(dataset_root, load_labelmap_file_fn=_load_labelmap_file)
    if not labelmap:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_labelmap_missing")
    label_to_id = {label: idx + 1 for idx, label in enumerate(labelmap)}
    categories = [{"id": cid, "name": name, "supercategory": "object"} for name, cid in label_to_id.items()]
    signature = _compute_dir_signature(dataset_root)
    existing_meta = _load_sam3_dataset_metadata_impl(dataset_root)
    dataset_type = (existing_meta or {}).get("type", "bbox")
    dataset_label = (existing_meta or {}).get("label", dataset_root.name)
    dataset_source = (existing_meta or {}).get("source", "yolo")
    if (
        existing_meta
        and existing_meta.get("signature") == signature
        and existing_meta.get("coco_train_json")
        and existing_meta.get("coco_val_json")
    ):
        coco_train_path = Path(existing_meta["coco_train_json"])
        coco_val_path = Path(existing_meta["coco_val_json"])
        rebuild = _coco_has_invalid_image_refs_impl(coco_train_path) or _coco_has_invalid_image_refs_impl(coco_val_path)
        if dataset_type == "seg":
            rebuild = rebuild or _coco_missing_segmentation_impl(coco_train_path) or _coco_missing_segmentation_impl(coco_val_path)
        if not rebuild:
            _ensure_coco_info_fields_impl(coco_train_path, dataset_root.name, categories)
            _ensure_coco_info_fields_impl(coco_val_path, dataset_root.name, categories)
            return existing_meta

    dataset_type = "bbox"
    image_id_counter = 1
    annotation_id = 1
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def _convert_split(split_images: Path, split_labels: Path, split_name: str) -> str:
        nonlocal image_id_counter, annotation_id, dataset_type
        images: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []
        images_lookup: Dict[str, int] = {}
        image_sizes: Dict[str, Tuple[int, int]] = {}

        def _clamp01(val: float) -> float:
            return max(0.0, min(1.0, val))

        def _bbox_xyxy_from_cxcywh(
            cx: float, cy: float, w: float, h: float
        ) -> Optional[Tuple[float, float, float, float]]:
            if w <= 0 or h <= 0:
                return None
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            x1 = _clamp01(x1)
            y1 = _clamp01(y1)
            x2 = _clamp01(x2)
            y2 = _clamp01(y2)
            if x2 <= x1 or y2 <= y1:
                return None
            return (x1, y1, x2, y2)

        def _bbox_xyxy_from_polygon(coords: List[float]) -> Optional[Tuple[float, float, float, float]]:
            if len(coords) < 6 or len(coords) % 2 != 0:
                return None
            xs = coords[0::2]
            ys = coords[1::2]
            if not xs or not ys:
                return None
            min_x = _clamp01(min(xs))
            max_x = _clamp01(max(xs))
            min_y = _clamp01(min(ys))
            max_y = _clamp01(max(ys))
            if max_x <= min_x or max_y <= min_y:
                return None
            return (min_x, min_y, max_x, max_y)

        def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
            inter_area = inter_w * inter_h
            if inter_area <= 0:
                return 0.0
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            denom = area_a + area_b - inter_area
            return inter_area / denom if denom > 0 else 0.0

        def _polygon_to_coco_segmentation(
            coords: List[float], width: int, height: int
        ) -> Optional[List[List[float]]]:
            if len(coords) < 6 or len(coords) % 2 != 0:
                return None
            out: List[float] = []
            for idx in range(0, len(coords), 2):
                x = _clamp01(coords[idx]) * width
                y = _clamp01(coords[idx + 1]) * height
                out.extend([x, y])
            if len(out) < 6:
                return None
            return [out]

        for label_file in sorted(split_labels.rglob("*.txt")):
            image_path = _image_path_for_label_impl(split_labels, split_images, label_file, image_exts)
            if image_path is None:
                logger.warning("No matching image for label file %s", label_file)
                continue
            image_rel = str(image_path.relative_to(split_images.parent))
            if image_rel not in images_lookup:
                try:
                    with Image.open(image_path) as im:
                        width, height = im.size
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to read image %s: %s", image_path, exc)
                    continue
                images_lookup[image_rel] = image_id_counter
                image_sizes[image_rel] = (width, height)
                images.append(
                    {
                        "id": image_id_counter,
                        "file_name": image_rel,
                        "width": width,
                        "height": height,
                    }
                )
                image_id_counter += 1
            image_id = images_lookup[image_rel]
            width, height = image_sizes.get(image_rel, (None, None))
            try:
                with label_file.open("r", encoding="utf-8") as handle:
                    lines = [ln.strip() for ln in handle if ln.strip()]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read YOLO labels from %s: %s", label_file, exc)
                continue
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    class_idx = int(float(parts[0]))
                except (TypeError, ValueError):
                    continue
                if class_idx < 0 or class_idx >= len(labelmap):
                    continue
                if width is None or height is None:
                    continue
                raw_vals: List[float] = []
                for token in parts[1:]:
                    try:
                        raw_vals.append(float(token))
                    except (TypeError, ValueError):
                        raw_vals = []
                        break
                if not raw_vals:
                    continue
                bbox_xyxy: Optional[Tuple[float, float, float, float]] = None
                segmentation: Optional[List[List[float]]] = None
                if len(raw_vals) == 4:
                    cx, cy, w, h = raw_vals
                    bbox_xyxy = _bbox_xyxy_from_cxcywh(cx, cy, w, h)
                else:
                    poly_only = raw_vals if len(raw_vals) >= 6 and len(raw_vals) % 2 == 0 else None
                    bbox_plus_poly = None
                    if len(raw_vals) > 4 and (len(raw_vals) - 4) >= 6 and (len(raw_vals) - 4) % 2 == 0:
                        bbox_plus_poly = (raw_vals[:4], raw_vals[4:])
                    chosen_poly = None
                    if poly_only is not None and bbox_plus_poly is not None:
                        bbox_fields, poly_fields = bbox_plus_poly
                        bbox_from_fields = _bbox_xyxy_from_cxcywh(*bbox_fields)
                        bbox_from_poly = _bbox_xyxy_from_polygon(poly_fields)
                        if bbox_from_fields is not None and bbox_from_poly is not None and _bbox_iou(bbox_from_fields, bbox_from_poly) >= 0.9:
                            chosen_poly = poly_fields
                            bbox_xyxy = bbox_from_poly
                        else:
                            chosen_poly = poly_only
                            bbox_xyxy = _bbox_xyxy_from_polygon(poly_only)
                    elif bbox_plus_poly is not None:
                        _, poly_fields = bbox_plus_poly
                        chosen_poly = poly_fields
                        bbox_xyxy = _bbox_xyxy_from_polygon(poly_fields)
                    elif poly_only is not None:
                        chosen_poly = poly_only
                        bbox_xyxy = _bbox_xyxy_from_polygon(poly_only)
                    else:
                        bbox_xyxy = _bbox_xyxy_from_cxcywh(*raw_vals[:4])
                    if chosen_poly is not None and bbox_xyxy is not None:
                        segmentation = _polygon_to_coco_segmentation(chosen_poly, int(width), int(height))
                        dataset_type = "seg"

                if bbox_xyxy is None:
                    continue
                x1_n, y1_n, x2_n, y2_n = bbox_xyxy
                x1 = x1_n * width
                y1 = y1_n * height
                abs_w = (x2_n - x1_n) * width
                abs_h = (y2_n - y1_n) * height
                if abs_w <= 0 or abs_h <= 0:
                    continue
                ann = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_idx + 1,
                    "bbox": [x1, y1, abs_w, abs_h],
                    "area": abs_w * abs_h,
                    "iscrowd": 0,
                }
                if segmentation is not None:
                    ann["segmentation"] = segmentation
                annotations.append(ann)
                annotation_id += 1
        output_path = dataset_root / split_name / "_annotations.coco.json"
        try:
            _write_coco_annotations_impl(
                output_path,
                dataset_id=dataset_root.name,
                categories=categories,
                images=images,
                annotations=annotations,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"sam3_coco_write_failed:{exc}") from exc
        return str(output_path)

    coco_train = _convert_split(train_images, train_labels, "train")
    coco_val = _convert_split(val_images, val_labels, "val")
    sam3_meta = {
        "id": dataset_root.name,
        "label": dataset_label,
        "source": dataset_source,
        "type": dataset_type,
        "dataset_root": str(dataset_root),
        "signature": signature,
        "classes": labelmap,
        "context": "",
        "image_count": None,
        "train_count": None,
        "val_count": None,
        "coco_train_json": coco_train,
        "coco_val_json": coco_val,
        "converted_at": time.time(),
    }
    _persist_sam3_dataset_metadata_impl(dataset_root, sam3_meta)
    return sam3_meta


def _convert_qwen_dataset_to_coco_impl(dataset_root: Path) -> Dict[str, Any]:
    dataset_root = dataset_root.resolve()
    metadata = _load_qwen_dataset_metadata_impl(dataset_root) or {}
    metadata, signature = _ensure_qwen_dataset_signature_impl(
        dataset_root,
        metadata,
        compute_dir_signature_fn=_compute_dir_signature,
        persist_metadata_fn=_persist_qwen_dataset_metadata_impl,
    )
    if "type" not in metadata:
        metadata["type"] = "bbox"
        _persist_qwen_dataset_metadata_impl(dataset_root, metadata)
    dataset_id = metadata.get("id") or dataset_root.name
    labelmap = _load_qwen_labelmap(
        dataset_root,
        load_qwen_meta=_load_qwen_dataset_metadata_impl,
        collect_labels=_collect_labels_from_qwen_jsonl_impl,
    )
    if not labelmap:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_labelmap_missing")
    label_to_id = {label: idx + 1 for idx, label in enumerate(labelmap)}
    categories = [{"id": cid, "name": name, "supercategory": "object"} for name, cid in label_to_id.items()]
    existing_meta = _load_sam3_dataset_metadata_impl(dataset_root)
    if (
        existing_meta
        and existing_meta.get("signature") == signature
        and existing_meta.get("coco_train_json")
        and existing_meta.get("coco_val_json")
    ):
        _ensure_coco_info_fields_impl(Path(existing_meta["coco_train_json"]), dataset_id, categories)
        _ensure_coco_info_fields_impl(Path(existing_meta["coco_val_json"]), dataset_id, categories)
        return existing_meta

    annotation_id = 1
    images_lookup: Dict[str, int] = {}
    image_sizes: Dict[str, Tuple[int, int]] = {}

    def _convert_split(split: str) -> str:
        nonlocal annotation_id
        jsonl_path = dataset_root / split / "annotations.jsonl"
        if not jsonl_path.exists():
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"sam3_annotations_missing:{split}")
        images: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []
        try:
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    image_rel = payload.get("image")
                    if not isinstance(image_rel, str):
                        continue
                    if image_rel not in images_lookup:
                        image_path = dataset_root / split / image_rel
                        if not image_path.exists():
                            logger.warning("Missing image referenced in %s: %s", jsonl_path, image_path)
                            continue
                        try:
                            with Image.open(image_path) as im:
                                width, height = im.size
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Failed to read image %s: %s", image_path, exc)
                            continue
                        images_lookup[image_rel] = len(images_lookup) + 1
                        image_sizes[image_rel] = (width, height)
                        images.append(
                            {
                                "id": images_lookup[image_rel],
                                "file_name": image_rel,
                                "width": width,
                                "height": height,
                            }
                        )
                    image_id = images_lookup[image_rel]
                    width, height = image_sizes.get(image_rel, (None, None))
                    detections = _extract_qwen_detections_from_payload_impl(payload)
                    for det in detections:
                        label = str(det.get("label", "")).strip()
                        if not label or label not in label_to_id:
                            continue
                        bbox = det.get("bbox")
                        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                            try:
                                x1 = float(bbox[0])
                                y1 = float(bbox[1])
                                x2 = float(bbox[2])
                                y2 = float(bbox[3])
                            except (TypeError, ValueError):
                                continue
                            if width is not None and height is not None:
                                x1 = max(0.0, min(x1, width))
                                x2 = max(0.0, min(x2, width))
                                y1 = max(0.0, min(y1, height))
                                y2 = max(0.0, min(y2, height))
                            w = max(0.0, x2 - x1)
                            h = max(0.0, y2 - y1)
                            if w <= 0 or h <= 0:
                                continue
                            coco_bbox = [x1, y1, w, h]
                        else:
                            point = det.get("point")
                            if not (isinstance(point, (list, tuple)) and len(point) >= 2):
                                continue
                            try:
                                cx = float(point[0])
                                cy = float(point[1])
                            except (TypeError, ValueError):
                                continue
                            size = 2.0
                            x1 = cx - size / 2.0
                            y1 = cy - size / 2.0
                            coco_bbox = [x1, y1, size, size]
                        area = coco_bbox[2] * coco_bbox[3]
                        if area <= 0:
                            continue
                        annotations.append(
                            {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": label_to_id[label],
                                "bbox": coco_bbox,
                                "area": area,
                                "iscrowd": 0,
                            }
                        )
                        annotation_id += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to convert %s to COCO: %s", jsonl_path, exc)
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"sam3_coco_conversion_failed:{split}")
        output_path = dataset_root / split / "_annotations.coco.json"
        try:
            _write_coco_annotations_impl(
                output_path,
                dataset_id=dataset_id,
                categories=categories,
                images=images,
                annotations=annotations,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"sam3_coco_write_failed:{exc}") from exc
        return str(output_path)

    coco_train = _convert_split("train")
    coco_val = _convert_split("val")
    sam3_meta = {
        "id": metadata.get("id") or dataset_root.name,
        "label": metadata.get("label") or metadata.get("id") or dataset_root.name,
        "source": "qwen",
        "type": metadata.get("type", "bbox"),
        "dataset_root": str(dataset_root),
        "signature": signature,
        "classes": labelmap,
        "context": metadata.get("context", ""),
        "image_count": metadata.get("image_count"),
        "train_count": metadata.get("train_count"),
        "val_count": metadata.get("val_count"),
        "coco_train_json": coco_train,
        "coco_val_json": coco_val,
        "converted_at": time.time(),
    }
    _persist_sam3_dataset_metadata_impl(dataset_root, sam3_meta)
    return sam3_meta


def _find_coco_split_impl(dataset_root: Path) -> Tuple[Path, Path]:
    candidates = list(dataset_root.rglob("_annotations.coco.json"))
    if not candidates:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="coco_annotations_missing")
    candidates.sort(key=lambda p: len(p.parts))
    ann_path = candidates[0]
    images_dir = ann_path.parent / "images"
    if not images_dir.exists():
        images_dir = ann_path.parent
    return ann_path, images_dir


def _convert_coco_dataset_to_yolo_impl(dataset_root: Path) -> Dict[str, Any]:
    dataset_root = dataset_root.resolve()
    ann_paths: List[Tuple[str, Path, Path]] = []
    for split in ("train", "val"):
        ann_path = dataset_root / split / "_annotations.coco.json"
        if not ann_path.exists():
            continue
        images_dir = ann_path.parent / "images"
        if not images_dir.exists():
            images_dir = ann_path.parent
        ann_paths.append((split, ann_path, images_dir))
    if not ann_paths:
        ann_path, images_dir = _find_coco_split_impl(dataset_root)
        ann_paths = [("train", ann_path, images_dir)]

    category_map: Dict[int, str] = {}
    has_segmentation = False
    for _, ann_path, _ in ann_paths:
        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"coco_load_failed:{exc}") from exc
        for cat in data.get("categories", []) or []:
            try:
                cid = int(cat.get("id"))
            except Exception:
                continue
            name = str(cat.get("name") or f"class_{cid}")
            category_map.setdefault(cid, name)
        if not category_map:
            for ann in data.get("annotations", []) or []:
                try:
                    cid = int(ann.get("category_id"))
                except Exception:
                    continue
                category_map.setdefault(cid, f"class_{cid}")
        if not has_segmentation:
            for ann in data.get("annotations", []) or []:
                seg = ann.get("segmentation")
                if seg:
                    has_segmentation = True
                    break

    if not category_map:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="coco_categories_missing")

    sorted_ids = sorted(category_map.keys())
    labelmap = [category_map[cid] for cid in sorted_ids]
    labelmap_path = dataset_root / "labelmap.txt"
    labelmap_path.write_text("\n".join(labelmap) + "\n", encoding="utf-8")
    cat_id_to_idx = {cid: idx for idx, cid in enumerate(sorted_ids)}

    dataset_type = "seg" if has_segmentation else "bbox"
    for split_name, ann_path, images_dir in ann_paths:
        labels_dir = dataset_root / split_name / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"coco_load_failed:{exc}") from exc
        images = data.get("images", []) or []
        annotations = data.get("annotations", []) or []
        ann_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in annotations:
            try:
                img_id = int(ann.get("image_id"))
            except Exception:
                continue
            ann_by_image.setdefault(img_id, []).append(ann)
        for img in images:
            try:
                img_id = int(img.get("id"))
            except Exception:
                continue
            file_name = str(img.get("file_name") or "")
            img_path = _resolve_coco_image_path_impl(file_name, images_dir, split_name, dataset_root)
            if img_path is None:
                logger.warning("COCO->YOLO: missing image for %s in %s", file_name, dataset_root)
                continue
            width = img.get("width")
            height = img.get("height")
            if not width or not height:
                try:
                    with Image.open(img_path) as im:
                        width, height = im.size
                except Exception as exc:  # noqa: BLE001
                    logger.warning("COCO->YOLO: failed to read image size for %s: %s", img_path, exc)
                    continue
            label_rel = _label_relpath_for_image_impl(file_name)
            label_path = labels_dir / label_rel
            label_path.parent.mkdir(parents=True, exist_ok=True)
            lines: List[str] = []
            for ann in ann_by_image.get(img_id, []):
                try:
                    cat_id = int(ann.get("category_id"))
                except Exception:
                    continue
                if cat_id not in cat_id_to_idx:
                    continue
                class_idx = cat_id_to_idx[cat_id]
                bbox = ann.get("bbox") or []
                if len(bbox) < 4:
                    continue
                x, y, w, h = map(float, bbox[:4])
                if w <= 0 or h <= 0:
                    continue
                cx = (x + w / 2.0) / float(width)
                cy = (y + h / 2.0) / float(height)
                bw = w / float(width)
                bh = h / float(height)
                if dataset_type == "seg":
                    seg = ann.get("segmentation")
                    poly = None
                    if isinstance(seg, list):
                        for candidate in seg:
                            if isinstance(candidate, list) and len(candidate) >= 6:
                                poly = candidate
                                break
                    if poly is not None:
                        coords: List[str] = []
                        for idx in range(0, len(poly), 2):
                            px = float(poly[idx]) / float(width)
                            py = float(poly[idx + 1]) / float(height)
                            coords.append(f"{max(0.0, min(1.0, px)):.6f}")
                            coords.append(f"{max(0.0, min(1.0, py)):.6f}")
                        if len(coords) >= 6:
                            lines.append(f"{class_idx} " + " ".join(coords))
                            continue
                lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if lines:
                label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = _load_sam3_dataset_metadata_impl(dataset_root) or {}
    meta.setdefault("id", dataset_root.name)
    meta.setdefault("label", dataset_root.name)
    meta.setdefault("source", meta.get("source") or "coco")
    meta["classes"] = labelmap
    meta["type"] = dataset_type
    meta["dataset_root"] = str(dataset_root)
    meta["signature"] = _compute_dir_signature(dataset_root)
    meta["yolo_converted_at"] = time.time()
    _persist_sam3_dataset_metadata_impl(dataset_root, meta)
    return meta


def _resolve_sam3_dataset_meta_impl(dataset_id: str) -> Dict[str, Any]:
    dataset_root = _resolve_sam3_or_qwen_dataset_impl(dataset_id)
    annotations_path = dataset_root / "train" / "annotations.jsonl"
    train_images = dataset_root / "train" / "images"
    train_labels = dataset_root / "train" / "labels"
    if annotations_path.exists():
        meta = _convert_qwen_dataset_to_coco_impl(dataset_root)
    elif train_images.exists() and train_labels.exists():
        meta = _convert_yolo_dataset_to_coco_impl(dataset_root)
    else:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_dataset_type_unsupported")
    meta["dataset_root"] = str(dataset_root)
    return meta


def _load_coco_index_impl(dataset_root: Path) -> Tuple[Dict[str, Any], Dict[int, Dict[int, List[List[float]]]], Dict[int, Dict[str, Any]]]:
    dataset_root = dataset_root.resolve()
    coco_paths: List[Path] = []
    meta = _load_sam3_dataset_metadata_impl(dataset_root) or {}
    coco_train = meta.get("coco_train_json")
    coco_val = meta.get("coco_val_json")
    if coco_train:
        coco_paths.append(Path(coco_train))
    if coco_val:
        coco_paths.append(Path(coco_val))
    if not coco_paths:
        annotations_path = dataset_root / "train" / "annotations.jsonl"
        train_images = dataset_root / "train" / "images"
        train_labels = dataset_root / "train" / "labels"
        if annotations_path.exists():
            meta = _convert_qwen_dataset_to_coco_impl(dataset_root)
        elif train_images.exists() and train_labels.exists():
            meta = _convert_yolo_dataset_to_coco_impl(dataset_root)
        else:
            ann_path, _images_dir = _find_coco_split_impl(dataset_root)
            meta = {"coco_train_json": str(ann_path)}
        coco_train = meta.get("coco_train_json")
        coco_val = meta.get("coco_val_json")
        if coco_train:
            coco_paths.append(Path(coco_train))
        if coco_val:
            coco_paths.append(Path(coco_val))
    if not coco_paths:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="coco_annotations_missing")

    combined: Dict[str, Any] = {"images": [], "annotations": [], "categories": []}
    images_by_id: Dict[int, Dict[str, Any]] = {}
    gt_by_image_cat: Dict[int, Dict[int, List[List[float]]]] = {}
    used_image_ids: set[int] = set()
    used_ann_ids: set[int] = set()
    next_image_id = 1
    next_ann_id = 1

    for coco_path in coco_paths:
        if not coco_path.exists():
            continue
        try:
            data = json.loads(coco_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"coco_load_failed:{exc}") from exc
        categories = data.get("categories")
        if isinstance(categories, list) and categories:
            if not combined["categories"]:
                combined["categories"] = categories
        images = data.get("images", []) or []
        annotations = data.get("annotations", []) or []
        images_dir = coco_path.parent / "images"
        if not images_dir.exists():
            images_dir = coco_path.parent
        split_name = images_dir.parent.name if images_dir.name == "images" else ""
        id_map: Dict[int, int] = {}
        for img in images:
            if not isinstance(img, dict):
                continue
            try:
                old_id = int(img.get("id", next_image_id))
            except Exception:
                old_id = next_image_id
            new_id = old_id
            if new_id in used_image_ids:
                new_id = next_image_id
            next_image_id = max(next_image_id, new_id + 1)
            used_image_ids.add(new_id)
            id_map[old_id] = new_id
            file_name = str(img.get("file_name") or "")
            width = img.get("width")
            height = img.get("height")
            resolved_path = _resolve_coco_image_path_impl(file_name, images_dir, split_name, dataset_root)
            entry = {
                "id": new_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "path": str(resolved_path) if resolved_path else None,
            }
            images_by_id[new_id] = entry
            combined["images"].append({k: v for k, v in entry.items() if k != "path"})
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            try:
                old_ann_id = int(ann.get("id", next_ann_id))
            except Exception:
                old_ann_id = next_ann_id
            ann_id = old_ann_id
            if ann_id in used_ann_ids:
                ann_id = next_ann_id
            next_ann_id = max(next_ann_id, ann_id + 1)
            used_ann_ids.add(ann_id)
            try:
                old_img_id = int(ann.get("image_id"))
            except Exception:
                continue
            new_img_id = id_map.get(old_img_id, old_img_id)
            try:
                cat_id = int(ann.get("category_id"))
            except Exception:
                continue
            bbox = ann.get("bbox") or []
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                try:
                    x, y, w, h = map(float, bbox[:4])
                except (TypeError, ValueError):
                    x = y = w = h = None
                if x is not None and w is not None:
                    gt_by_image_cat.setdefault(new_img_id, {}).setdefault(cat_id, []).append([x, y, x + w, y + h])
            ann_copy = dict(ann)
            ann_copy["id"] = ann_id
            ann_copy["image_id"] = new_img_id
            combined["annotations"].append(ann_copy)

    return combined, gt_by_image_cat, images_by_id
