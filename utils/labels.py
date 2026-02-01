from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Union
import re
import colorsys

import joblib
from fastapi import HTTPException
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_412_PRECONDITION_FAILED,
)

logger = logging.getLogger(__name__)


def _read_labelmap_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        lines = [line.strip() for line in path.read_text().splitlines()]
        return [line for line in lines if line]
    except Exception:
        return []


def _normalize_class_name_for_match(name: Optional[str]) -> str:
    if not name:
        return ""
    try:
        s = str(name).strip().lower()
    except Exception:
        return ""
    return re.sub(r"[^a-z0-9]+", "", s)


def _normalize_labelmap_entries(values: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for raw in values or []:
        text = str(raw or "").strip()
        if not text:
            normalized.append("")
            continue
        norm = _normalize_class_name_for_match(text)
        normalized.append(norm or text.lower())
    return normalized


def _apply_expected_labelmap_warnings(
    expected: Optional[Sequence[str]],
    labelmap: Sequence[str],
    warnings: List[str],
) -> None:
    if expected and not labelmap:
        warnings.append("labelmap_missing")
    elif expected and labelmap and list(expected) != list(labelmap):
        warnings.append("labelmap_mismatch")


def _labelmaps_match(expected: Sequence[str], actual: Sequence[str]) -> bool:
    if not expected or not actual:
        return True
    exp_norm = _normalize_labelmap_entries(expected)
    act_norm = _normalize_labelmap_entries(actual)
    if len(exp_norm) != len(act_norm):
        return False
    for exp, act in zip(exp_norm, act_norm):
        if exp != act:
            return False
    return True


def _raise_on_labelmap_mismatch(
    *,
    expected: Optional[Sequence[str]],
    actual: Optional[Sequence[str]],
    context: str,
) -> None:
    if not expected or not actual:
        return
    if not _labelmaps_match(expected, actual):
        raise HTTPException(
            status_code=HTTP_412_PRECONDITION_FAILED,
            detail=f"detector_labelmap_mismatch:{context}",
        )


def _agent_label_prefix_candidates(label: str) -> List[str]:
    if not label:
        return []
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", str(label)).strip()
    tokens = [tok for tok in cleaned.split() if tok]
    candidates: List[str] = []
    if tokens:
        if len(tokens) >= 2:
            candidates.append(tokens[0][0] + tokens[1][0])
            if len(tokens[1]) >= 2:
                candidates.append(tokens[0][0] + tokens[1][:2])
        if len(tokens[0]) >= 2:
            candidates.append(tokens[0][:2])
        if len(tokens[0]) >= 3:
            candidates.append(tokens[0][:3])
    else:
        flat = re.sub(r"[^A-Za-z0-9]+", "", str(label))
        if len(flat) >= 2:
            candidates.append(flat[:2])
        if len(flat) >= 3:
            candidates.append(flat[:3])
    seen: set[str] = set()
    uniq: List[str] = []
    for cand in candidates:
        cand = re.sub(r"[^A-Za-z0-9]+", "", cand).upper()
        if len(cand) < 2 or cand in seen:
            continue
        uniq.append(cand)
        seen.add(cand)
    return uniq


def _agent_label_color_map(labelmap: Sequence[str]) -> Dict[str, str]:
    colors: Dict[str, str] = {}
    if not labelmap:
        return colors
    golden = 0.61803398875
    for idx, label in enumerate(labelmap):
        name = str(label).strip()
        if not name:
            continue
        hue = (idx * golden) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.9)
        colors[name] = f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"
    return colors


def _agent_label_prefix_map(labels: Sequence[str]) -> Dict[str, str]:
    prefixes: Dict[str, str] = {}
    used: Set[str] = set()
    for label in labels:
        label_str = str(label or "").strip()
        if not label_str:
            continue
        candidates = _agent_label_prefix_candidates(label_str)
        chosen = None
        for cand in candidates:
            if cand not in used:
                chosen = cand
                break
        if chosen is None:
            base = candidates[0] if candidates else "X"
            base = re.sub(r"[^A-Za-z0-9]+", "", base).upper() or "X"
            idx = 1
            chosen = f"{base}{idx}"
            while chosen in used:
                idx += 1
                chosen = f"{base}{idx}"
        prefixes[label_str] = chosen
        used.add(chosen)
    return prefixes


def _agent_overlay_key_text(
    label_colors: Mapping[str, str],
    label_prefixes: Optional[Mapping[str, str]] = None,
) -> str:
    if not label_colors:
        return ""
    lines: List[str] = []
    for label, color in label_colors.items():
        prefix = label_prefixes.get(label) if label_prefixes else None
        if prefix:
            lines.append(f"{label} ({prefix}) -> {color}")
        else:
            lines.append(f"{label} -> {color}")
    return "\n".join(lines)


def _agent_fuzzy_align_label(label: Optional[str], labelmap: Sequence[str]) -> Optional[str]:
    if not label:
        return None
    label_norm = _normalize_class_name_for_match(label)
    if not label_norm:
        return None
    norm_map: Dict[str, str] = {}
    for lbl in labelmap:
        norm = _normalize_class_name_for_match(lbl)
        if norm:
            norm_map.setdefault(norm, lbl)
    if label_norm in norm_map:
        return norm_map[label_norm]
    try:
        import difflib
    except Exception:
        return None
    # Fuzzy matching is deliberately strict to avoid re-mapping to the wrong class.
    # Only allow near-exact matches with small edit distance and no ambiguity.
    keys = list(norm_map.keys())
    if not keys:
        return None
    best = None
    best_score = 0.0
    second = 0.0
    for key in keys:
        score = difflib.SequenceMatcher(None, label_norm, key).ratio()
        if score > best_score:
            second = best_score
            best_score = score
            best = key
        elif score > second:
            second = score
    if best is None:
        return None
    if best_score < 0.92:
        return None
    if abs(len(best) - len(label_norm)) > 2:
        return None
    if best[0] != label_norm[0]:
        return None
    if second and abs(best_score - second) <= 0.02:
        return None
    return norm_map.get(best)


def _load_labelmap_file(path: Optional[Union[str, Path]], *, strict: bool = False) -> List[str]:
    if path is None:
        return []
    if isinstance(path, Path):
        path_str = str(path)
        lower = path.name.lower()
    else:
        path_str = str(path)
        lower = Path(path_str).name.lower()
    if not path_str.strip():
        return []
    try:
        if lower.endswith(".pkl"):
            data = joblib.load(path_str)
            if isinstance(data, list):
                return [str(item) for item in data]
            raise ValueError("labelmap_pickle_invalid")
        entries: List[str] = []
        with open(path_str, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    entries.append(stripped)
        if not entries:
            raise ValueError("labelmap_empty")
        return entries
    except FileNotFoundError as exc:
        if strict:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="labelmap_not_found") from exc
        return []
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"labelmap_load_failed:{exc}") from exc
        logger.warning("Failed to load labelmap from %s: %s", path_str, exc)
        return []
