from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from utils.labels import _normalize_class_name_for_match


def _is_background_class_name(name: Optional[str]) -> bool:
    try:
        label = str(name or "").strip().lower()
    except Exception:
        return False
    return label.startswith("__bg_")


def _clip_head_background_indices(classes: Sequence[str]) -> List[int]:
    return [idx for idx, label in enumerate(classes) if _is_background_class_name(label)]


def _agent_background_classes_from_head(head: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(head, dict):
        return []
    classes = [str(c) for c in list(head.get("classes") or [])]
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
