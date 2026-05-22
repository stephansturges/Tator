"""Classifier utility helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from utils.labels import _normalize_class_name_for_match


def _is_background_class_name(name: Optional[str]) -> bool:
    try:
        label = str(name or "").strip().lower()
    except Exception:
        return False
    return label.startswith("__bg_")


def _clip_head_background_indices(classes: Sequence[str]) -> List[int]:
    return [idx for idx, label in enumerate(classes) if _is_background_class_name(label)]


def _classifier_classes_list(classes_raw: Any) -> List[str]:
    if classes_raw is None:
        return []
    if hasattr(classes_raw, "tolist"):
        try:
            classes_raw = classes_raw.tolist()
        except Exception:
            return []
    if isinstance(classes_raw, str):
        return [str(classes_raw)]
    try:
        return [str(c) for c in list(classes_raw)]
    except Exception:
        return []


def _clip_head_classes(head: Optional[Mapping[str, Any]]) -> List[str]:
    if not isinstance(head, Mapping):
        return []
    return _classifier_classes_list(head.get("classes"))


def _agent_background_classes_from_head(head: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(head, dict):
        return []
    classes = _clip_head_classes(head)
    indices = _clip_head_background_indices(classes)
    return [classes[idx] for idx in indices if 0 <= idx < len(classes)]


def _find_clip_head_target_index(classes: Sequence[str], class_name: Optional[str]) -> Optional[int]:
    target = _normalize_class_name_for_match(class_name)
    if not target:
        return None
    for idx, c in enumerate(classes):
        if _normalize_class_name_for_match(c) == target:
            return int(idx)
    return None
