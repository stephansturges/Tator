from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence


def _agent_classifier_classes_for_path(path: Path, *, load_model_fn) -> List[str]:
    classes: List[str] = []
    try:
        obj = load_model_fn(str(path))
        if isinstance(obj, dict):
            raw = obj.get("classes")
        else:
            raw = getattr(obj, "classes_", None)
        if raw is not None:
            try:
                classes = [str(c) for c in list(raw)]
            except Exception:
                classes = [str(raw)]
    except Exception:
        classes = []
    return classes


def _agent_classifier_matches_labelmap(
    path: Path,
    labelmap: Sequence[str],
    *,
    load_model_fn,
    normalize_label_fn,
    bg_indices_fn,
) -> bool:
    if not labelmap:
        return False
    classes = _agent_classifier_classes_for_path(path, load_model_fn=load_model_fn)
    if not classes:
        return False
    bg_indices = bg_indices_fn(classes)
    filtered = [cls for idx, cls in enumerate(classes) if idx not in bg_indices]
    label_norm = {normalize_label_fn(n) for n in labelmap if n}
    clf_norm = {normalize_label_fn(n) for n in filtered if n}
    return bool(label_norm.intersection(clf_norm))


def _agent_default_classifier_for_dataset(
    dataset_id: Optional[str],
    *,
    load_labelmap_fn,
    classifier_matches_fn,
    root_dir: Path,
) -> Optional[str]:
    if not dataset_id:
        return None
    labelmap = load_labelmap_fn(dataset_id)
    if not labelmap:
        return None
    candidates = [
        "DinoV3_best_model_large.pkl",
        f"{dataset_id}_clip_vitl14_bg5.pkl",
        f"{dataset_id}_clip_vitb16_bg5.pkl",
        f"{dataset_id}_clip_vitb32_bg5.pkl",
    ]
    for name in candidates:
        path = root_dir / name
        if path.exists() and classifier_matches_fn(path, labelmap):
            return str(path)
    return None
