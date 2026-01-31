from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, List

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


def _purge_dataset_artifacts_impl(
    dataset_id: str,
    *,
    normalise_relative_path_fn,
    agent_mining_meta_root: Path,
    agent_mining_det_cache_root: Path,
    prompt_helper_preset_root: Path,
) -> None:
    """Remove per-dataset agent/prompt-helper artifacts."""
    safe_dataset = normalise_relative_path_fn(dataset_id)
    for derived_root in (
        agent_mining_meta_root / safe_dataset,
        agent_mining_det_cache_root / safe_dataset,
    ):
        try:
            shutil.rmtree(derived_root, ignore_errors=True)
        except Exception:
            pass
    try:
        for preset_path in prompt_helper_preset_root.glob("*.json"):
            try:
                with preset_path.open("r", encoding="utf-8") as handle:
                    preset_data = json.load(handle)
                if preset_data.get("dataset_id") in {dataset_id, safe_dataset}:
                    preset_path.unlink(missing_ok=True)
            except Exception:
                continue
    except Exception:
        pass


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


def _find_any_file_impl(root: Path) -> Optional[Path]:
    if not root.exists() or not root.is_dir():
        return None
    try:
        for entry in root.iterdir():
            if entry.is_file():
                return entry
    except Exception:
        return None
    return None


def _count_dir_files_impl(root: Path) -> Optional[int]:
    if not root.exists() or not root.is_dir():
        return None
    try:
        return sum(1 for entry in root.iterdir() if entry.is_file())
    except Exception:
        return None


def _dataset_integrity_report_impl(
    dataset_root: Path,
    *,
    find_any_file_fn,
    count_dir_files_fn,
    load_qwen_labelmap_fn,
    load_qwen_meta_fn,
    collect_labels_fn,
    discover_yolo_labelmap_fn,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "type": "unknown",
        "ok": True,
        "issues": [],
    }
    annotations_path = dataset_root / "train" / "annotations.jsonl"
    train_images = dataset_root / "train" / "images"
    train_labels = dataset_root / "train" / "labels"
    if annotations_path.exists():
        report["type"] = "qwen"
        if not train_images.exists():
            report["issues"].append("missing_train_images")
        if not annotations_path.is_file() or annotations_path.stat().st_size == 0:
            report["issues"].append("missing_annotations")
        if train_images.exists():
            if not find_any_file_fn(train_images):
                report["issues"].append("no_images_found")
        report["image_count"] = count_dir_files_fn(train_images)
        report["annotation_bytes"] = annotations_path.stat().st_size if annotations_path.exists() else None
        labelmap = load_qwen_labelmap_fn(
            dataset_root,
            load_qwen_meta=load_qwen_meta_fn,
            collect_labels=collect_labels_fn,
        )
        if not labelmap:
            report["issues"].append("missing_labelmap")
    elif train_images.exists() and train_labels.exists():
        report["type"] = "yolo"
        if not find_any_file_fn(train_images):
            report["issues"].append("no_images_found")
        if not find_any_file_fn(train_labels):
            report["issues"].append("no_labels_found")
        report["image_count"] = count_dir_files_fn(train_images)
        report["label_count"] = count_dir_files_fn(train_labels)
        labelmap = discover_yolo_labelmap_fn(dataset_root)
        if not labelmap:
            report["issues"].append("missing_labelmap")
    else:
        report["issues"].append("dataset_layout_unrecognized")
    if report["issues"]:
        report["ok"] = False
    return report


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
