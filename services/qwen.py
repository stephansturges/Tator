"""Qwen captioning + prompt expansion helpers."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from utils.glossary import _normalize_labelmap_glossary, _parse_glossary_mapping


_CAPTION_BAD_DISPLAY_TERMS = {"[", "]", "{", "}", "[]", "{}"}
_FULL_IMAGE_AUTO_BOX_LIMIT = 80
_CAPTION_RESTRICTED_BLOCKED_TERMS: Dict[str, Tuple[str, ...]] = {
    "Crane": ("crane", "cranes", "gantry crane", "gantry cranes"),
}


def _caption_display_term(term: Any, *, max_len: int = 80) -> str:
    cleaned = _collapse_whitespace(str(term or "").replace("_", " ").strip(" \t\r\n\"'`"))
    cleaned = cleaned.strip(" .,:;")
    if not cleaned:
        return ""
    if cleaned in _CAPTION_BAD_DISPLAY_TERMS:
        return ""
    if any(ch in cleaned for ch in "[]{}"):
        return ""
    if not re.search(r"[A-Za-z0-9]", cleaned):
        return ""
    if len(cleaned) > max_len:
        return ""
    return cleaned


def _caption_natural_label(label: Any) -> str:
    raw = str(label or "").strip()
    if not raw:
        return ""
    spaced = raw.replace("_", " ")
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", spaced)
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)
    return _caption_display_term(spaced) or raw


def _caption_sanitize_glossary_map(
    glossary_map: Optional[Mapping[str, Sequence[Any]]],
) -> Dict[str, List[str]]:
    cleaned: Dict[str, List[str]] = {}
    for label, terms in (glossary_map or {}).items():
        label_key = str(label or "").strip()
        if not label_key:
            continue
        if isinstance(terms, str):
            term_iter: Sequence[Any] = [terms]
        else:
            term_iter = list(terms or [])
        display_terms: List[str] = []
        seen: set[str] = set()
        for term in term_iter:
            display = _caption_display_term(term)
            if not display:
                continue
            key = display.lower()
            if key in seen:
                continue
            seen.add(key)
            display_terms.append(display)
        if display_terms:
            cleaned[label_key] = display_terms
    return cleaned


def _extract_balanced_json(text: str, start_char: str, end_char: str) -> Optional[str]:
    start = text.find(start_char)
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == start_char:
            depth += 1
        elif ch == end_char:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _generate_qwen_text(
    prompt: str,
    *,
    max_new_tokens: int,
    use_system_prompt: bool,
    system_prompt: Optional[str],
    ensure_qwen_ready_fn: Callable[[], Tuple[Any, Any]],
    resolve_qwen_device_fn: Callable[[], Any],
) -> str:
    model, processor = ensure_qwen_ready_fn()
    messages: List[Dict[str, Any]] = []
    if use_system_prompt:
        sys_prompt = system_prompt
        if sys_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": sys_prompt}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt")
    device = resolve_qwen_device_fn()
    inputs = inputs.to(device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0, top_p=1.0)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]
    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return decoded.strip()


def _caption_glossary_map(labelmap_glossary: Optional[str], labels: Sequence[str]) -> Dict[str, List[str]]:
    if not labelmap_glossary:
        return {}
    parsed = _parse_glossary_mapping(_normalize_labelmap_glossary(labelmap_glossary), list(labels))
    return _caption_sanitize_glossary_map(parsed)


def _sanitize_prompts_impl(prompts: List[str]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for p in prompts:
        if not isinstance(p, str):
            continue
        val = p.strip()
        if not val:
            continue
        words = val.split()
        if not (1 <= len(words) <= 4):
            continue
        if any(len(w) <= 1 for w in words):
            continue
        key = val.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(val)
    return cleaned


def _caption_preferred_label(label: str, glossary_map: Optional[Dict[str, List[str]]] = None) -> str:
    label = str(label or "").strip()
    if not label:
        return ""
    terms = _caption_sanitize_glossary_map(glossary_map).get(label) or []
    for term in terms:
        display = _caption_display_term(term)
        if display:
            return display
    return _caption_natural_label(label)


def _caption_hint_value(hint: Any, key: str, default: Any = None) -> Any:
    if isinstance(hint, Mapping):
        return hint.get(key, default)
    return getattr(hint, key, default)


def _caption_finite_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return num if math.isfinite(num) else None


def _caption_box_subset_score(entry: Mapping[str, Any], image_area: float) -> Tuple[float, float, int]:
    confidence = _caption_finite_float(entry.get("confidence"))
    area = _caption_finite_float(entry.get("area")) or 0.0
    try:
        index = int(entry.get("index") or 0)
    except (TypeError, ValueError, OverflowError):
        index = 0
    return (
        confidence if confidence is not None else 0.0,
        min(1.0, max(0.0, area) / max(1.0, image_area)),
        -index,
    )


def _caption_box_center(entry: Mapping[str, Any], image_width: int, image_height: int) -> Tuple[float, float]:
    bbox = entry.get("bbox")
    if isinstance(bbox, Sequence) and len(bbox) == 4:
        coords = [_caption_finite_float(value) for value in bbox]
        if all(value is not None for value in coords):
            x1, y1, x2, y2 = coords
            assert x1 is not None and y1 is not None and x2 is not None and y2 is not None
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return (float(image_width) / 2.0, float(image_height) / 2.0)


def _caption_spatial_order_key(entry: Mapping[str, Any], image_width: int, image_height: int) -> Tuple[int, int, int]:
    cx, cy = _caption_box_center(entry, image_width, image_height)
    try:
        index = int(entry.get("index") or 0)
    except (TypeError, ValueError, OverflowError):
        index = 0
    return (int(round(cy * 1000.0)), int(round(cx * 1000.0)), index)


def _caption_select_representative_box_subset(
    hints: Sequence[Mapping[str, Any]],
    counts: Mapping[str, int],
    *,
    limit: int,
    image_width: int,
    image_height: int,
) -> List[Mapping[str, Any]]:
    """Choose a deterministic class-aware and spatially spread box subset."""

    clean_limit = max(0, int(limit or 0))
    if clean_limit <= 0 or len(hints) <= clean_limit:
        return list(hints)

    image_area = float(max(1, image_width) * max(1, image_height))

    def entry_key(entry: Mapping[str, Any]) -> int:
        try:
            return int(entry.get("index") or 0)
        except (TypeError, ValueError, OverflowError):
            return id(entry)

    def add(entry: Mapping[str, Any]) -> None:
        key = entry_key(entry)
        if key in selected_keys or len(selected) >= clean_limit:
            return
        selected.append(entry)
        selected_keys.add(key)

    selected: List[Mapping[str, Any]] = []
    selected_keys: set[int] = set()
    ranked = sorted(
        hints,
        key=lambda entry: _caption_box_subset_score(entry, image_area),
        reverse=True,
    )

    best_by_label: Dict[str, Mapping[str, Any]] = {}
    for entry in ranked:
        label = str(entry.get("label") or "").strip()
        if label and label not in best_by_label:
            best_by_label[label] = entry

    # First preserve class coverage when the cap is large enough for it.
    if len(best_by_label) <= clean_limit:
        for label in sorted(best_by_label, key=lambda key: (-int(counts.get(key, 0) or 0), key)):
            add(best_by_label[label])

    # Then add one high-scoring item from each occupied grid cell, walking the
    # image in reading order. This makes "representative" mean spatially spread,
    # not merely the highest confidence boxes clustered in one corner.
    aspect = max(0.25, min(4.0, float(max(1, image_width)) / float(max(1, image_height))))
    grid_cols = max(1, int(math.ceil(math.sqrt(clean_limit * aspect))))
    grid_rows = max(1, int(math.ceil(clean_limit / grid_cols)))
    cell_best: Dict[Tuple[int, int], Mapping[str, Any]] = {}
    for entry in ranked:
        cx, cy = _caption_box_center(entry, image_width, image_height)
        col = min(grid_cols - 1, max(0, int((cx / max(1.0, float(image_width))) * grid_cols)))
        row = min(grid_rows - 1, max(0, int((cy / max(1.0, float(image_height))) * grid_rows)))
        cell_best.setdefault((row, col), entry)
    for cell in sorted(cell_best):
        add(cell_best[cell])
        if len(selected) >= clean_limit:
            break

    for entry in ranked:
        add(entry)
        if len(selected) >= clean_limit:
            break

    return sorted(
        selected[:clean_limit],
        key=lambda entry: _caption_spatial_order_key(entry, image_width, image_height),
    )


def _resolve_caption_all_windows(
    caption_mode: Optional[str],
    caption_all_windows: Optional[bool],
    *,
    has_label_hints: bool,
    restrict_to_labels: bool,
) -> bool:
    if str(caption_mode or "full") != "windowed":
        return bool(caption_all_windows)
    if caption_all_windows is not None:
        return bool(caption_all_windows)
    return not (bool(has_label_hints) and bool(restrict_to_labels))


def _format_caption_glossary_instruction(
    glossary_map: Dict[str, List[str]],
    labels: Sequence[str],
) -> str:
    glossary_map = _caption_sanitize_glossary_map(glossary_map)
    if not glossary_map:
        return ""
    pieces: List[str] = []
    seen: set[str] = set()
    for raw_label in labels:
        label = str(raw_label or "").strip()
        if not label or label in seen:
            continue
        seen.add(label)
        terms = [
            _caption_display_term(term)
            for term in (glossary_map.get(label) or [])
            if _caption_display_term(term)
        ]
        if not terms:
            continue
        natural_terms: List[str] = []
        term_seen: set[str] = set()
        for term in terms:
            key = term.lower()
            if key in term_seen:
                continue
            term_seen.add(key)
            natural_terms.append(term)
            if len(natural_terms) >= 6:
                break
        if natural_terms:
            preferred = _caption_preferred_label(label, glossary_map)
            variants = [term for term in natural_terms if term.lower() != preferred.lower()]
            if variants:
                pieces.append(
                    f"{label}: broad term \"{preferred}\"; possible variants include "
                    f"{', '.join(variants)}"
                )
            else:
                pieces.append(f"{label}: broad term \"{preferred}\"")
    if not pieces:
        return ""
    return (
        "Class meaning glossary: "
        + "; ".join(pieces)
        + ". Use glossary entries only to understand the broad class meaning of label hints. "
        "Glossary variants are possible members of a class, not assertions that those variants appear in this image. "
        "Do not choose a subtype from the glossary unless the image clearly supports that subtype; "
        "when uncertain, use the broad term."
    )


_CAPTION_EDITOR_PRESERVE_BROAD_TERMS = (
    "Editor rule: preserve broad category terms from the draft and window observations. "
    "Do not replace a broad class term with a more specific subtype from the glossary unless that "
    "exact subtype is used consistently by the draft and relevant window/source observations. "
    "If observations disagree, or only one crop uses a subtype while other observations use the "
    "broad term, keep the broad term during merge, cleanup, or refinement."
)


def _caption_editor_preserve_broad_terms_instruction() -> str:
    return _CAPTION_EDITOR_PRESERVE_BROAD_TERMS


_SHORT_CAPTION_REQUEST_RE = re.compile(
    r"(?:"
    r"\b(?:short|brief|concise|compact)\b|"
    r"\b(?:single|one|1)\s+(?:complete\s+)?sentence\b|"
    r"\b(?:1|one)\s*(?:-|to|or)\s*(?:2|two)\s+(?:complete\s+)?sentences?\b|"
    r"\b(?:at most|no more than|maximum of|max)\s+(?:1|2|one|two)\s+(?:complete\s+)?sentences?\b"
    r")",
    re.IGNORECASE,
)


def _caption_requested_sentence_limit(
    user_prompt: Optional[str],
    max_sentences: Optional[int] = None,
) -> Optional[int]:
    try:
        explicit = int(max_sentences) if max_sentences is not None else None
    except (TypeError, ValueError):
        explicit = None
    cleaned = str(user_prompt or "")
    lowered = cleaned.lower()
    prompt_limit: Optional[int] = None
    if re.search(r"\b(?:single|one|1)\s+(?:complete\s+)?sentence\b", lowered):
        prompt_limit = 1
    elif re.search(r"\b(?:1|one)\s*(?:-|to|or)\s*(?:2|two)\s+(?:complete\s+)?sentences?\b", lowered):
        prompt_limit = 2
    else:
        match = re.search(
            r"\b(?:at most|no more than|maximum of|max)\s+(1|2|one|two)\s+(?:complete\s+)?sentences?\b",
            lowered,
        )
        if match:
            val = match.group(1)
            prompt_limit = 1 if val in {"1", "one"} else 2
    if explicit is not None and explicit > 0 and prompt_limit is not None:
        return min(explicit, prompt_limit)
    if prompt_limit is not None:
        return prompt_limit
    if explicit is not None and explicit > 0:
        return explicit
    return None


def _caption_user_requested_short(
    user_prompt: Optional[str],
    max_sentences: Optional[int] = None,
) -> bool:
    limit = _caption_requested_sentence_limit(user_prompt, max_sentences)
    if limit is not None and limit <= 2:
        return True
    return _SHORT_CAPTION_REQUEST_RE.search(str(user_prompt or "")) is not None


def _build_qwen_caption_prompt(
    user_prompt: str,
    label_hints: Sequence[Any],
    image_width: int,
    image_height: int,
    include_counts: bool,
    include_coords: bool,
    max_boxes: int,
    detailed_mode: bool,
    restrict_to_labels: bool = True,
    labelmap_glossary: Optional[str] = None,
    max_sentences: Optional[int] = None,
    context_prompt: Optional[str] = None,
) -> Tuple[str, Dict[str, int], int, bool]:
    safe_width = max(1, int(image_width))
    safe_height = max(1, int(image_height))
    counts: Dict[str, int] = dict(
        Counter(
            label
            for label in (
                str(_caption_hint_value(hint, "label", "") or "").strip()
                for hint in label_hints
            )
            if label
        )
    )

    def _bbox_to_qwen_2d(bbox: Sequence[float]) -> List[int]:
        x1, y1, x2, y2 = bbox
        scale = 1000.0
        nx1 = int(round((x1 / safe_width) * scale))
        ny1 = int(round((y1 / safe_height) * scale))
        nx2 = int(round((x2 / safe_width) * scale))
        ny2 = int(round((y2 / safe_height) * scale))
        return [
            max(0, min(1000, nx1)),
            max(0, min(1000, ny1)),
            max(0, min(1000, nx2)),
            max(0, min(1000, ny2)),
        ]

    hints_payload = []
    for hint_index, hint in enumerate(label_hints):
        bbox = _caption_hint_value(hint, "bbox", None) or []
        label = str(_caption_hint_value(hint, "label", "") or "").strip()
        if not label:
            continue
        confidence = _caption_finite_float(_caption_hint_value(hint, "confidence", None))
        if len(bbox) == 4:
            coords = [_caption_finite_float(value) for value in bbox]
            if any(value is None for value in coords):
                x1 = y1 = x2 = y2 = None
            else:
                x1_raw, y1_raw, x2_raw, y2_raw = coords
                assert (
                    x1_raw is not None
                    and y1_raw is not None
                    and x2_raw is not None
                    and y2_raw is not None
                )
                x1 = max(0.0, min(x1_raw, safe_width))
                y1 = max(0.0, min(y1_raw, safe_height))
                x2 = max(0.0, min(x2_raw, safe_width))
                y2 = max(0.0, min(y2_raw, safe_height))
            if (x1 is not None and x2 <= x1) or (y1 is not None and y2 <= y1):
                continue
        else:
            x1 = y1 = x2 = y2 = None
        hints_payload.append(
            {
                "index": hint_index,
                "label": label,
                "bbox": [x1, y1, x2, y2] if x1 is not None else None,
                "bbox_2d": _bbox_to_qwen_2d([x1, y1, x2, y2]) if x1 is not None else None,
                "confidence": confidence if confidence is not None else None,
                "area": (x2 - x1) * (y2 - y1) if x1 is not None else 0.0,
            }
        )
    sorted_hints = sorted(
        hints_payload,
        key=lambda entry: (
            -(entry["confidence"] if entry["confidence"] is not None else 0.0),
            -entry["area"],
        ),
    )
    if max_boxes <= 0:
        auto_limit = _FULL_IMAGE_AUTO_BOX_LIMIT
        if auto_limit > 0 and len(sorted_hints) > auto_limit:
            selected = list(
                _caption_select_representative_box_subset(
                    sorted_hints,
                    counts,
                    limit=auto_limit,
                    image_width=safe_width,
                    image_height=safe_height,
                )
            )
            truncated = True
        else:
            selected = sorted_hints
            truncated = False
    else:
        selected = list(
            _caption_select_representative_box_subset(
                sorted_hints,
                counts,
                limit=max_boxes,
                image_width=safe_width,
                image_height=safe_height,
            )
        )
        truncated = len(sorted_hints) > len(selected)
    custom_context_prompt = _collapse_whitespace(context_prompt or "")
    has_custom_context_prompt = bool(custom_context_prompt)
    lines: List[str] = []
    if user_prompt:
        lines.append(f"User caption request: {user_prompt}")
        if not has_custom_context_prompt and "style inspirations" in user_prompt.lower():
            lines.append(
                "Style guidance: use inspirations for tone/angle only. Rephrase, do not copy wording."
            )
    lines.append(f"Image size: {safe_width}x{safe_height} pixels.")
    glossary_map = _caption_glossary_map(
        labelmap_glossary,
        list(counts.keys()) or [entry["label"] for entry in hints_payload if entry.get("label")],
    )
    glossary_instruction = _format_caption_glossary_instruction(
        glossary_map,
        list(counts.keys()) or [entry["label"] for entry in hints_payload if entry.get("label")],
    )
    if glossary_instruction:
        lines.append(glossary_instruction)

    # Build a forbidden token list for labelmap tags that should not appear verbatim.
    forbidden_labels: List[str] = []
    for lbl in sorted(set([entry.get("label") for entry in hints_payload if entry.get("label")])):
        preferred = _caption_preferred_label(lbl, glossary_map)
        if "_" in lbl or (preferred and preferred.lower() != str(lbl).lower()):
            forbidden_labels.append(str(lbl))

    if include_counts and counts:
        counts_text = ", ".join(
            f"{_caption_preferred_label(label, glossary_map)}: {count}" for label, count in counts.items()
        )
        if restrict_to_labels:
            lines.append(f"COUNTS (state exactly in final caption): {counts_text}.")
            if not has_custom_context_prompt:
                lines.append(
                    "State these counts as ordinary image facts in the final caption, without qualifiers "
                    "(avoid words like 'visible', 'roughly', or 'approximately')."
                )
        else:
            lines.append(f"COUNTS (use as hints; may be incomplete): {counts_text}.")
    elif counts:
        lines.append("Use the label hints to mention the main objects you see.")
    if counts and restrict_to_labels:
        allowed = ", ".join(sorted(_caption_preferred_label(lbl, glossary_map) for lbl in counts.keys()))
        if allowed:
            if has_custom_context_prompt:
                lines.append(f"Labeled class inventory: {allowed}.")
            else:
                lines.append(
                    f"Labeled class inventory: {allowed}. Treat these classes and counts as authoritative. "
                    "Mention other visible scene/background context only generically; do not name additional object "
                    "types outside this inventory, do not add extra counted object lists outside this inventory, "
                    "and do not pluralize a class whose count is 1."
                )
    elif counts and not restrict_to_labels:
        lines.append("Label hints are suggestions; you may mention other visible objects too.")
    if selected:
        if truncated:
            lines.append(
                f"Box list policy: this prompt lists a representative spatial subset of {len(selected)} boxes out of "
                f"{len(hints_payload)} total boxes, selected for class coverage and spatial spread. "
                "The listed boxes are layout examples, not the full object inventory; authoritative counts still reflect all label hints."
            )
        if include_coords:
            lines.append(
                "Labeled boxes (bbox_2d=[x1,y1,x2,y2], coords 0–1000 relative to this image/window):"
            )
            compact = [
                {"label": _caption_preferred_label(entry["label"], glossary_map), "bbox_2d": entry["bbox_2d"]}
                for entry in selected
                if entry["bbox_2d"] is not None
            ]
            if compact:
                lines.append(json.dumps(compact, separators=(",", ":")))
            if not has_custom_context_prompt:
                lines.append("Use relative positions (e.g., top-left, center) when describing layout.")
        else:
            labels_only = ", ".join(_caption_preferred_label(entry["label"], glossary_map) for entry in selected)
            lines.append(f"Labeled objects (one per box): {labels_only}.")
    if forbidden_labels:
        lines.append(
            "Forbidden label tokens (do NOT output these exact strings): "
            + ", ".join(forbidden_labels)
            + ". Use natural synonyms instead."
        )
    elif hints_payload and not include_counts:
        lines.append("Labels provided but box details omitted.")
    if has_custom_context_prompt:
        lines.append("Caption policy:")
        lines.append(custom_context_prompt)
    else:
        lines.append("Caption policy:")
        lines.append(
            "Treat the user caption request as required guidance. If it asks a question or asks for an inference such as likely location, scene type, event, or time, answer it when the image supports a grounded answer; otherwise state uncertainty briefly.\n"
            "Do not mention that a request or hint was provided.\n"
            "Follow the combined user request assembled from the editable caption style text and optional opening guidance; rephrase opening wording instead of copying exact phrases.\n"
            "When authoritative counts of objects or people are present, state those counts as ordinary image facts in the final caption using digits and without qualifiers such as visible, roughly, or approximately.\n"
            "Use relative positions such as top-left, center, or lower-right when box layout matters, but never mention coordinates.\n"
            "Write a detailed caption. Use the image as truth and incorporate label hints; if hints conflict with the image, mention the uncertainty briefly.\n"
            "Describe what the main objects or people are doing or how they are arranged when visible.\n"
            "Mention important concrete details without adding unsupported objects or actions.\n"
            "When authoritative counts are present, state them as ordinary image facts using digits.\n"
            "When box layout matters, convert box positions into natural relative layout words; never mention coordinates.\n"
            "Use glossary entries as semantic class meanings, not as forced words to copy.\n"
            "Final caption length: The final caption may be long when the image contains many visible details; use up to 10 complete sentences when needed to preserve concrete detail."
        )
    return "\n".join(lines), counts, len(selected), truncated


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _extract_caption_from_text(text: str, marker: Optional[str] = None) -> Tuple[str, bool]:
    cleaned = text.strip()
    marker_found = False
    final_match = re.search(r"<final>(.*?)</final>", cleaned, re.IGNORECASE | re.DOTALL)
    if final_match:
        cleaned = final_match.group(1)
        marker_found = True
    if marker:
        marker_pattern = re.escape(marker)
        match = re.search(
            rf"(?:^|\n)\s*{marker_pattern}(?:\s+(?:ANSWER|CAPTION))?\s*:?\s*(.+)",
            cleaned,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            cleaned = match.group(1)
            marker_found = True
    if not marker_found:
        match = re.search(
            r"(?:^|\n)\s*FINAL(?:\s+(?:ANSWER|CAPTION))?\s*:?\s*(.+)",
            cleaned,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            cleaned = match.group(1)
            marker_found = True
    cleaned = _collapse_whitespace(cleaned) if cleaned else text.strip()
    return cleaned, marker_found


def _caption_sentence_key(sentence: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()


def _caption_sentence_units(text: str) -> List[str]:
    if not text:
        return []
    units = re.findall(r"[^.!?]+[.!?]+(?:[\"')\]]+)?|[^.!?]+$", text)
    return [unit.strip() for unit in units if unit and unit.strip()]


def _caption_repetition_loop_detected(text: str) -> bool:
    if not text:
        return False
    if _caption_surface_loop_reason(text):
        return True
    compact = re.sub(r"\s+", "", text)
    if re.search(r"([A-Za-z0-9])\1{30,}", compact):
        return True
    units = _caption_sentence_units(text)
    keys = []
    for unit in units:
        key = _caption_sentence_key(unit)
        if len(key.split()) >= 3:
            keys.append(key)
    if len(keys) >= 3:
        counts = Counter(keys)
        if any(count >= 3 for count in counts.values()):
            return True
    if len(keys) >= 4:
        for width in range(1, min(4, len(keys) // 2) + 1):
            if keys[-width:] == keys[-2 * width : -width]:
                if len(keys) >= 3 * width and keys[-width:] == keys[-3 * width : -2 * width]:
                    return True
    words = re.findall(r"[a-z0-9']+", text.lower())
    if len(words) >= 24:
        for size in (4, 5, 6, 8):
            if len(words) < size * 3:
                continue
            ngrams = [tuple(words[idx : idx + size]) for idx in range(0, len(words) - size + 1)]
            if any(count >= 4 for count in Counter(ngrams).values()):
                return True
    return False


def _caption_surface_loop_reason(text: str) -> Optional[str]:
    if not text:
        return None
    compact = re.sub(r"\s+", "", str(text))
    if not compact:
        return None
    if re.search(r"([A-Za-z0-9])\1{30,}", compact):
        return "alnum_char_loop"
    if re.search(r"([^A-Za-z0-9\s])\1{20,}", str(text)):
        return "punctuation_loop"
    if len(compact) > 10 and re.fullmatch(r"[^A-Za-z0-9]+", compact):
        return "punctuation_only"
    alnum = re.findall(r"[A-Za-z0-9]", compact)
    if len(compact) >= 20 and len(alnum) / max(1, len(compact)) < 0.2:
        return "low_alnum"
    if len(compact) >= 80:
        most_common_ratio = max((compact.count(ch) for ch in set(compact)), default=0) / max(1, len(compact))
        if most_common_ratio >= 0.80:
            return "char_dominance"
        for width in range(2, 17):
            if len(compact) < width * 12:
                continue
            chunk = compact[:width]
            repeated = chunk * (len(compact) // width)
            if chunk and repeated == compact[: len(repeated)]:
                return "repeated_chunk"
        if re.search(r"(.{2,16})\1{12,}", compact):
            return "repeated_chunk"
    return None


def _truncate_repeated_caption_loop(text: str) -> str:
    if not text:
        return text
    char_loop = re.search(r"([A-Za-z0-9])\1{30,}", text)
    if char_loop:
        return text[: max(0, char_loop.start() + 1)].strip()
    punctuation_loop = re.search(r"([^A-Za-z0-9\s])\1{20,}", text)
    if punctuation_loop:
        return text[: max(0, punctuation_loop.start() + 1)].strip()
    units = _caption_sentence_units(text)
    if len(units) < 3:
        return text
    kept: List[str] = []
    keys: List[str] = []
    counts: Counter[str] = Counter()
    for unit in units:
        key = _caption_sentence_key(unit)
        if len(key.split()) >= 3:
            counts[key] += 1
            if counts[key] >= 3:
                break
        kept.append(unit.strip())
        if len(key.split()) >= 3:
            keys.append(key)
            for width in range(1, min(4, len(keys) // 2) + 1):
                if keys[-width:] == keys[-2 * width : -width]:
                    # Keep the first cycle and drop the duplicated cycle that revealed the loop.
                    del kept[-width:]
                    del keys[-width:]
                    return _collapse_whitespace(" ".join(kept))
    if not kept:
        return ""
    return _collapse_whitespace(" ".join(kept))


def _trim_repeated_caption_sentences(text: str) -> str:
    if not text:
        return text
    units = _caption_sentence_units(text)
    if len(units) < 2:
        return text
    seen: set[str] = set()
    kept: List[str] = []
    for unit in units:
        sentence = unit.strip()
        if not sentence:
            continue
        key = _caption_sentence_key(sentence)
        words = key.split()
        if len(words) >= 5:
            is_complete = bool(re.search(r"[.!?][\"')\]]*$", sentence))
            if key in seen:
                continue
            if not is_complete and any(prev.startswith(key) for prev in seen if len(key) >= 28):
                continue
            seen.add(key)
        kept.append(sentence)
    if not kept:
        return ""
    return _collapse_whitespace(" ".join(kept))


_CAPTION_NON_ENGLISH_SCRIPT_RE = re.compile(
    r"[\u0400-\u052F\u0590-\u05FF\u0600-\u06FF\u0900-\u097F"
    r"\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]"
)


def _caption_needs_english_rewrite(text: str) -> bool:
    return bool(_CAPTION_NON_ENGLISH_SCRIPT_RE.search(str(text or "")))


_CAPTION_GENERIC_OPENERS = (
    "an overhead view",
    "overhead view",
    "from a high angle",
    "a high-angle image",
    "a bird's-eye view",
    "overhead view",
)


def _caption_starts_generic(text: str) -> bool:
    lowered = text.strip().lower()
    return any(lowered.startswith(prefix) for prefix in _CAPTION_GENERIC_OPENERS)


def _caption_label_terms(
    label: str,
    glossary_map: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    terms = [str(label or "")]
    if "_" in str(label or ""):
        terms.append(str(label).replace("_", " "))
    clean_glossary = _caption_sanitize_glossary_map(glossary_map)
    if clean_glossary and clean_glossary.get(label):
        terms.extend(clean_glossary[label])
    preferred = _caption_preferred_label(label, glossary_map)
    if preferred:
        terms.append(preferred)
    cleaned: List[str] = []
    seen: set[str] = set()
    for term in terms:
        val = _collapse_whitespace(str(term or "").strip()).lower()
        if not val or val in seen:
            continue
        seen.add(val)
        cleaned.append(val)
    return cleaned


def _caption_plural_variants(term: str) -> List[str]:
    term = _collapse_whitespace(str(term or "").strip()).lower()
    if not term:
        return []
    words = term.split()
    last = words[-1]
    variants = {term}
    if last == "person":
        people_words = list(words)
        people_words[-1] = "people"
        variants.add(" ".join(people_words))
    if last.endswith("s") and not last.endswith(("ss", "us")):
        plural = last
    elif last.endswith(("s", "x", "ch", "sh")):
        plural = f"{last}es"
    elif last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        plural = f"{last[:-1]}ies"
    else:
        plural = f"{last}s"
    words[-1] = plural
    variants.add(" ".join(words))
    return sorted(variants, key=len, reverse=True)


def _caption_term_pattern(term: str) -> str:
    return r"\s+".join(re.escape(part) for part in _collapse_whitespace(term).split())


def _caption_count_number_pattern(count: int) -> str:
    return re.escape(str(count))


_CAPTION_INEXACT_COUNT_QUALIFIER_PATTERN = (
    r"(?:at\s+least|at\s+most|about|around|roughly|approximately|approx\.?|"
    r"nearly|almost|over|more\s+than|fewer\s+than|less\s+than|"
    r"no\s+more\s+than|no\s+fewer\s+than|up\s+to|as\s+many\s+as|"
    r"estimated(?:\s+at)?)"
)


def _caption_count_match_has_inexact_qualifier(text: str, match_start: int) -> bool:
    if match_start <= 0:
        return False
    prefix = str(text or "")[max(0, match_start - 80) : match_start].lower()
    return bool(
        re.search(
            rf"(?:^|[\s,;:]){_CAPTION_INEXACT_COUNT_QUALIFIER_PATTERN}\s+$",
            prefix,
            flags=re.IGNORECASE,
        )
    )


def _caption_term_present(text: str, term: str) -> bool:
    text = str(text or "")
    term = _collapse_whitespace(str(term or "").strip()).lower()
    if not text or not term:
        return False
    patterns = [
        _caption_term_pattern(variant)
        for variant in _caption_plural_variants(term)
        if variant
    ]
    if not patterns:
        return False
    return bool(re.search(rf"\b(?:{'|'.join(patterns)})\b", text, flags=re.IGNORECASE))


def _caption_plural_term(term: str) -> str:
    base = _collapse_whitespace(str(term or "").strip()).lower()
    words = base.split()
    if words and words[-1] == "person":
        words[-1] = "people"
        return " ".join(words)
    for variant in _caption_plural_variants(base):
        if variant != base:
            return variant
    return base


def _caption_replace_term(text: str, term: str, replacement: str) -> str:
    output = str(text or "")
    term = _collapse_whitespace(str(term or "").strip()).lower()
    replacement = _collapse_whitespace(str(replacement or "").strip()).lower()
    if not output or not term or not replacement:
        return output
    replacement_plural = _caption_plural_term(replacement)
    forms = []
    for variant in _caption_plural_variants(term):
        if variant:
            forms.append((variant, variant != term))
    forms.sort(key=lambda item: len(item[0]), reverse=True)
    for form, is_plural in forms:
        pattern = _caption_term_pattern(form)
        repl_text = replacement_plural if is_plural else replacement

        def _repl(match: re.Match[str]) -> str:
            matched = match.group(0)
            if matched[:1].isupper():
                return repl_text[:1].upper() + repl_text[1:]
            return repl_text

        output = re.sub(rf"\b{pattern}\b", _repl, output, flags=re.IGNORECASE)
    return output


def _caption_is_broad_glossary_alias(term: str, preferred: str, label: str) -> bool:
    val = _collapse_whitespace(str(term or "").strip()).lower()
    pref = _collapse_whitespace(str(preferred or "").strip()).lower()
    label_text = _collapse_whitespace(str(label or "").replace("_", " ").strip()).lower()
    if not val:
        return True
    if val in {pref, label_text}:
        return True
    return False


def _caption_broad_source_terms(label: str, preferred: str) -> List[str]:
    terms = [
        preferred,
        str(label or "").replace("_", " "),
        str(label or ""),
    ]
    cleaned: List[str] = []
    seen: set[str] = set()
    for term in terms:
        val = _collapse_whitespace(str(term or "").strip())
        key = val.lower()
        if not val or key in seen or "_" in val:
            continue
        seen.add(key)
        cleaned.append(val)
    return cleaned


def _caption_demote_unstable_glossary_subtypes(
    caption: str,
    counts: Optional[Dict[str, int]],
    glossary_map: Optional[Dict[str, List[str]]] = None,
    source_outputs: Optional[Sequence[Tuple[str, str]]] = None,
) -> str:
    """Prefer broad class wording when subtype evidence is inconsistent."""

    output = str(caption or "")
    if not output or not glossary_map:
        return output
    labels = list((counts or {}).keys()) or list(glossary_map.keys())
    source_texts = [str(text or "") for _label, text in (source_outputs or []) if str(text or "").strip()]
    for label in labels:
        raw_terms = glossary_map.get(label) or []
        if not raw_terms:
            continue
        preferred = _caption_preferred_label(label, glossary_map)
        if not preferred:
            continue
        replacement = preferred
        broad_terms = _caption_broad_source_terms(label, preferred)
        variants: List[str] = []
        seen_variants: set[str] = set()
        for term in raw_terms:
            val = _collapse_whitespace(str(term or "").strip())
            key = val.lower()
            if not val or key in seen_variants or "_" in val:
                continue
            if _caption_is_broad_glossary_alias(val, preferred, label):
                continue
            seen_variants.add(key)
            variants.append(val)
        variants.sort(key=len, reverse=True)
        for variant in variants:
            if not _caption_term_present(output, variant):
                continue
            variant_sources = {
                idx for idx, text in enumerate(source_texts) if _caption_term_present(text, variant)
            }
            broad_sources = {
                idx
                for idx, text in enumerate(source_texts)
                if any(_caption_term_present(text, broad) for broad in broad_terms)
            }
            broad_only_sources = broad_sources - variant_sources
            if variant_sources and not broad_only_sources:
                continue
            output = _caption_replace_term(output, variant, replacement)
    return _collapse_whitespace(output)


def _caption_count_conflicts(
    text: str,
    counts: Dict[str, int],
    glossary_map: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    if not text or not counts:
        return []
    lowered = text.lower()
    conflicts: List[str] = []
    quantified_many = r"(?:two|three|four|five|six|seven|eight|nine|ten|several|multiple|many|numerous|various|a\s+few|few)"
    singular_quantifier = r"(?:single|only\s+one|just\s+one)"
    for label, count in counts.items():
        try:
            count_int = int(count)
        except (TypeError, ValueError):
            continue
        if count_int <= 0:
            continue
        label_conflict = False
        for term in _caption_label_terms(label, glossary_map):
            variants = _caption_plural_variants(term)
            if not variants:
                continue
            singular_pattern = _caption_term_pattern(term)
            plural_patterns = [_caption_term_pattern(variant) for variant in variants if variant != term]
            plural_pattern = "|".join(plural_patterns) or singular_pattern
            if count_int == 1:
                if re.search(rf"\b{quantified_many}\s+(?:\w+\s+){{0,3}}(?:{plural_pattern})\b", lowered):
                    label_conflict = True
                if re.search(rf"\b(?:[2-9]|\d{{2,}})\s+(?:\w+\s+){{0,3}}(?:{plural_pattern})\b", lowered):
                    label_conflict = True
                if plural_patterns and re.search(rf"\b(?:{plural_pattern})\s+(?:are|were|stand|sit|rest|appear|line|cluster|lie)\b", lowered):
                    label_conflict = True
            else:
                if re.search(rf"\b(?:{singular_quantifier})\s+(?:\w+\s+){{0,3}}{singular_pattern}\b", lowered):
                    label_conflict = True
                if re.search(rf"\b(?:only\s+)?1\s+(?:\w+\s+){{0,3}}{singular_pattern}\b", lowered):
                    label_conflict = True
            if label_conflict:
                conflicts.append(label)
                break
    return conflicts


def _caption_missing_exact_counts(
    text: str,
    counts: Dict[str, int],
    glossary_map: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    if not text or not counts:
        return []
    missing: List[str] = []
    for label, count in counts.items():
        try:
            count_int = int(count)
        except (TypeError, ValueError):
            continue
        if count_int <= 0:
            continue
        number_pattern = _caption_count_number_pattern(count_int)
        if not number_pattern:
            continue
        exact_present = False
        for term in _caption_label_terms(label, glossary_map):
            term_patterns = [
                _caption_term_pattern(variant)
                for variant in _caption_plural_variants(term)
                if variant
            ]
            if not term_patterns:
                continue
            term_pattern = "|".join(term_patterns)
            front_count_pattern = (
                rf"\b(?:a\s+total\s+of\s+|total\s+of\s+|exactly\s+)?(?:{number_pattern})\s+"
                rf"(?:[\w'-]+\s+){{0,4}}(?:{term_pattern})\b"
            )
            for match in re.finditer(front_count_pattern, text, flags=re.IGNORECASE):
                if _caption_count_match_has_inexact_qualifier(text, match.start()):
                    continue
                exact_present = True
                break
            if exact_present:
                break
            trailing_count_pattern = (
                rf"\b(?:{term_pattern})\s+(?:[\w'-]+\s+){{0,6}}"
                rf"(?:total|in\s+total|overall|altogether)\s+(?:is|are|of\s+)?(?:{number_pattern})\b"
            )
            if re.search(trailing_count_pattern, text, flags=re.IGNORECASE):
                exact_present = True
                break
        if not exact_present:
            missing.append(label)
    return missing


def _caption_count_term_for_sentence(
    label: str,
    count: int,
    glossary_map: Optional[Dict[str, List[str]]] = None,
) -> str:
    term = _caption_preferred_label(label, glossary_map).strip() or str(label or "").strip()
    if not term:
        return ""
    if int(count) == 1:
        return " ".join(
            token if token.isupper() and len(token) <= 3 else token.lower()
            for token in term.split()
        )
    plural = _caption_plural_term(term)
    original_tokens = term.split()
    plural_tokens = plural.split()
    if (
        original_tokens
        and plural_tokens
        and original_tokens[0].isupper()
        and len(original_tokens[0]) <= 3
    ):
        plural_tokens[0] = original_tokens[0]
    return " ".join(plural_tokens)


def _caption_join_count_phrases(pieces: Sequence[str]) -> str:
    cleaned = [piece for piece in pieces if piece]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _caption_split_sentences(text: str) -> List[str]:
    cleaned = _sanitize_qwen_caption(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def _caption_sentence_contradicts_positive_count(
    sentence: str,
    counts: Dict[str, int],
    glossary_map: Optional[Dict[str, List[str]]] = None,
) -> bool:
    if not sentence or not counts:
        return False
    lowered = sentence.lower()
    for label, count in counts.items():
        try:
            count_int = int(count)
        except (TypeError, ValueError):
            continue
        if count_int <= 0:
            continue
        for term in _caption_label_terms(label, glossary_map):
            term_patterns = [
                _caption_term_pattern(variant)
                for variant in _caption_plural_variants(term)
                if variant
            ]
            if not term_patterns:
                continue
            term_pattern = rf"(?:{'|'.join(term_patterns)})"
            negative_patterns = [
                rf"\bno\b(?:\s+[\w'-]+){{0,8}}\s+{term_pattern}\b",
                rf"\bwithout\b(?:\s+[\w'-]+){{0,8}}\s+{term_pattern}\b",
                rf"\bnot\s+any\b(?:\s+[\w'-]+){{0,8}}\s+{term_pattern}\b",
            ]
            if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in negative_patterns):
                return True
            if _caption_sentence_has_mismatched_quantity_for_term(
                lowered,
                term_pattern,
                count_int,
            ):
                return True
    return False


_CAPTION_QUANTITY_WORDS: Dict[str, int] = {
    "zero": 0,
    "one": 1,
    "single": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def _caption_parse_quantity_token(value: str) -> Optional[int]:
    token = str(value or "").strip().lower()
    if not token:
        return None
    if token.isdigit():
        try:
            return int(token)
        except (TypeError, ValueError):
            return None
    return _CAPTION_QUANTITY_WORDS.get(token)


def _caption_sentence_has_mismatched_quantity_for_term(
    lowered_sentence: str,
    term_pattern: str,
    authoritative_count: int,
) -> bool:
    quantity_pattern = (
        r"(?P<qty>\d+|zero|one|single|two|three|four|five|six|seven|eight|nine|"
        r"ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
        r"eighteen|nineteen|twenty)"
    )
    patterns = [
        rf"\bthere\s+(?:is|are)\s+(?:only\s+|just\s+)?{quantity_pattern}\s+(?:[\w'-]+\s+){{0,3}}{term_pattern}\b",
        rf"^\s*(?:only\s+|just\s+)?{quantity_pattern}\s+{term_pattern}\s+(?:is|are|was|were|can\s+be|could\s+be)\s+"
        rf"(?:seen|visible|present|shown|spotted|parked|standing|located|resting|grouped|arranged)\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, lowered_sentence, flags=re.IGNORECASE):
            parsed = _caption_parse_quantity_token(match.group("qty"))
            if parsed is not None and parsed != authoritative_count:
                return True
    return False


def _caption_remove_count_contradiction_sentences(
    text: str,
    counts: Dict[str, int],
    glossary_map: Optional[Dict[str, List[str]]] = None,
) -> str:
    sentences = _caption_split_sentences(text)
    if not sentences:
        return _sanitize_qwen_caption(text)
    kept: List[str] = []
    changed = False
    for sentence in sentences:
        if not _caption_sentence_contradicts_positive_count(sentence, counts, glossary_map):
            kept.append(sentence)
            continue
        changed = True
        salvage_match = re.search(r"\b(?:however|but|yet)\b,?\s+", sentence, flags=re.IGNORECASE)
        if not salvage_match:
            continue
        salvage = sentence[salvage_match.end() :].strip(" ;,")
        if not salvage or _caption_sentence_contradicts_positive_count(salvage, counts, glossary_map):
            continue
        kept.append(salvage[:1].upper() + salvage[1:])
    if not kept or not changed:
        return _sanitize_qwen_caption(text)
    return _collapse_whitespace(" ".join(kept))


def _caption_sentence_is_raw_count_inventory(
    sentence: str,
    counts: Dict[str, int],
    glossary_map: Optional[Dict[str, List[str]]] = None,
) -> bool:
    if not sentence or not counts or ":" not in sentence:
        return False
    lowered = sentence.lower()
    hits = 0
    for label, count in counts.items():
        try:
            count_int = int(count)
        except (TypeError, ValueError):
            continue
        if count_int <= 0:
            continue
        for term in _caption_label_terms(label, glossary_map):
            term_patterns = [
                _caption_term_pattern(variant)
                for variant in _caption_plural_variants(term)
                if variant
            ]
            if not term_patterns:
                continue
            if re.search(
                rf"\b(?:{'|'.join(term_patterns)})\s*:\s*{re.escape(str(count_int))}\b",
                lowered,
                flags=re.IGNORECASE,
            ):
                hits += 1
                break
    if hits <= 0:
        return False
    return len(sentence) <= 180 and not re.search(
        r"\b(?:contains|shows|captures|includes|arranged|standing|parked|moving)\b",
        lowered,
    )


def _caption_strip_count_inventory_sentences(
    text: str,
    counts: Dict[str, int],
    glossary_map: Optional[Dict[str, List[str]]] = None,
) -> str:
    sentences = _caption_split_sentences(text)
    if not sentences:
        return _sanitize_qwen_caption(text)
    kept = [
        sentence
        for sentence in sentences
        if not _caption_sentence_is_raw_count_inventory(sentence, counts, glossary_map)
    ]
    if not kept or len(kept) == len(sentences):
        return _sanitize_qwen_caption(text)
    return _collapse_whitespace(" ".join(kept))


def _caption_remove_unsupported_specific_terms(
    text: str,
    counts: Dict[str, int],
    glossary_map: Optional[Dict[str, List[str]]] = None,
) -> str:
    sentences = _caption_split_sentences(text)
    if not sentences or not counts:
        return _sanitize_qwen_caption(text)
    allowed_terms: set[str] = set()
    for label, count in counts.items():
        try:
            count_int = int(count)
        except (TypeError, ValueError):
            continue
        if count_int <= 0:
            continue
        for term in _caption_label_terms(label, glossary_map):
            for variant in _caption_plural_variants(term):
                compact = _collapse_whitespace(variant).lower()
                if compact:
                    allowed_terms.add(compact)
    blocked_patterns: List[str] = []
    for canonical, variants in _CAPTION_RESTRICTED_BLOCKED_TERMS.items():
        canonical_allowed = _collapse_whitespace(canonical).lower() in allowed_terms
        variant_allowed = any(_collapse_whitespace(variant).lower() in allowed_terms for variant in variants)
        if canonical_allowed or variant_allowed:
            continue
        blocked_patterns.extend(
            rf"\b{re.escape(_collapse_whitespace(variant).lower())}\b"
            for variant in variants
            if _collapse_whitespace(variant)
        )
    if not blocked_patterns:
        return _sanitize_qwen_caption(text)
    kept = [
        sentence
        for sentence in sentences
        if not any(
            re.search(pattern, sentence.lower(), flags=re.IGNORECASE)
            for pattern in blocked_patterns
        )
    ]
    if not kept or len(kept) == len(sentences):
        return _sanitize_qwen_caption(text)
    return _collapse_whitespace(" ".join(kept))


def _caption_ensure_exact_count_sentence(
    text: str,
    counts: Dict[str, int],
    glossary_map: Optional[Dict[str, List[str]]] = None,
) -> str:
    if not text or not counts:
        return _sanitize_qwen_caption(text)
    missing = _caption_missing_exact_counts(text, counts, glossary_map)
    if not missing:
        return _sanitize_qwen_caption(text)
    pieces: List[str] = []
    for label in counts.keys():
        try:
            count_int = int(counts.get(label, 0))
        except (TypeError, ValueError):
            continue
        if count_int <= 0:
            continue
        term = _caption_count_term_for_sentence(label, count_int, glossary_map)
        if term:
            pieces.append(f"{count_int} {term}")
    joined = _caption_join_count_phrases(pieces)
    cleaned = _sanitize_qwen_caption(text)
    if not joined:
        return cleaned
    count_sentence = f"The scene contains {joined}."
    if not cleaned:
        return count_sentence
    return _collapse_whitespace(f"{count_sentence} {cleaned}")


def _caption_normalize_inexact_count_qualifiers(
    text: str,
    counts: Dict[str, int],
    glossary_map: Optional[Dict[str, List[str]]] = None,
) -> str:
    output = _sanitize_qwen_caption(text)
    if not output or not counts:
        return output
    for label, count in counts.items():
        try:
            count_int = int(count)
        except (TypeError, ValueError):
            continue
        if count_int <= 0:
            continue
        number_pattern = _caption_count_number_pattern(count_int)
        if not number_pattern:
            continue
        for term in _caption_label_terms(label, glossary_map):
            term_patterns = [
                _caption_term_pattern(variant)
                for variant in _caption_plural_variants(term)
                if variant
            ]
            if not term_patterns:
                continue
            term_pattern = "|".join(term_patterns)
            pattern = (
                rf"\b(?P<qual>{_CAPTION_INEXACT_COUNT_QUALIFIER_PATTERN})\s+"
                rf"(?P<count>{number_pattern})\s+"
                rf"(?P<middle>(?:[\w'-]+\s+){{0,4}})"
                rf"(?P<term>{term_pattern})\b"
            )

            def repl(match: re.Match[str]) -> str:
                middle = match.group("middle") or ""
                return f"{match.group('count')} {middle}{match.group('term')}"

            output = re.sub(pattern, repl, output, flags=re.IGNORECASE)
    return _collapse_whitespace(output)


def _caption_repair_count_text_artifacts(
    text: str,
    counts: Dict[str, int],
    glossary_map: Optional[Dict[str, List[str]]] = None,
) -> str:
    cleaned = _caption_remove_count_contradiction_sentences(text, counts, glossary_map)
    cleaned = _caption_strip_count_inventory_sentences(cleaned, counts, glossary_map)
    cleaned = _caption_remove_unsupported_specific_terms(cleaned, counts, glossary_map)
    cleaned = _caption_normalize_inexact_count_qualifiers(cleaned, counts, glossary_map)
    return _caption_ensure_exact_count_sentence(cleaned, counts, glossary_map)


def _caption_missing_labels(
    text: str,
    counts: Dict[str, int],
    glossary_map: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    if not text:
        return list(counts.keys())
    lowered = text.lower()
    missing = []
    for label, count in counts.items():
        if count <= 0:
            continue
        label_terms = _caption_label_terms(label, glossary_map)
        if not any(_caption_term_present(lowered, term) for term in label_terms):
            missing.append(label)
    return missing


def _caption_needs_refine(
    caption: str,
    counts: Dict[str, int],
    detailed_mode: bool,
    include_counts: bool,
    glossary_map: Optional[Dict[str, List[str]]] = None,
) -> Tuple[bool, List[str]]:
    words = caption.split() if caption else []
    min_words = 12 if detailed_mode else 8
    if len(words) < min_words:
        return True, []
    missing = _caption_missing_labels(caption, counts, glossary_map) if include_counts else []
    count_conflicts = _caption_count_conflicts(caption, counts, glossary_map) if include_counts else []
    missing_exact_counts = (
        _caption_missing_exact_counts(caption, counts, glossary_map) if include_counts else []
    )
    if missing or count_conflicts or missing_exact_counts:
        return True, sorted(set(missing + count_conflicts + missing_exact_counts))
    if _caption_starts_generic(caption) and detailed_mode:
        return True, []
    return False, []


def _sanitize_qwen_caption(text: str) -> str:
    if not text:
        return text
    cleaned = text.strip()
    final_match = re.search(r"<final>(.*?)</final>", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if final_match:
        cleaned = final_match.group(1).strip()
    if re.search(r"</think>", cleaned, flags=re.IGNORECASE):
        parts = re.split(r"</think>", cleaned, flags=re.IGNORECASE)
        cleaned = parts[-1].strip()
    if re.search(
        r"(?:^|\n)\s*FINAL(?:\s+(?:ANSWER|CAPTION))?\s*:",
        cleaned,
        flags=re.IGNORECASE,
    ):
        cleaned, _ = _extract_caption_from_text(cleaned, marker="FINAL")
    cleaned = cleaned.strip()
    if cleaned.startswith(":"):
        cleaned = cleaned.lstrip(":").strip()
    cleaned = _QWEN_BBOX_ID_RE.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    cleaned = _truncate_repeated_caption_loop(cleaned)
    cleaned = _trim_repeated_caption_sentences(cleaned)
    return cleaned


_QWEN_THINKING_REASONING_RE = re.compile(
    r"(?:"
    r"\bgot it\b|\blet'?s\b|\bwait\b|\bactually\b|\bfirst\b|\bsecond\b|\bthird\b|\bstep\b|"
    r"\bi need\b|\bwe need\b|\bwe should\b|\bso we need\b|\bnow\b|"
    r"\bwe can mention\b|\bcan mention\b|\bwe can describe\b|\bwe can include\b|"
    r"\bwe should mention\b|\bwe can say\b|\bwe'?ll produce\b|\bwe will produce\b|"
    r"\bthe task\b|\bthe user wants\b|\bthe user says\b|\bthe user request\b|"
    r"\bthe prompt asks\b|\bthe instruction(?:s)?(?: say| says)?\b|\bstyle guidance\b|"
    r"\bpreferred opening\b|\bdraft caption\b|\bthe draft\b|"
    r"\bwindow observations\b|\bwindow region\b|\bregion of interest\b|"
    r"\bfinal answer only\b|\bwe need to produce\b|\bthis is ambiguous\b"
    r")",
    re.IGNORECASE,
)
_QWEN_CAPTION_META_RE = re.compile(
    r"("
    r"authoritative|as indicated|label hint|bounding box|bbox|coordinates|hinted|"
    r"counts are provided|draft caption|window observations|window region|region of interest|"
    r"user request|the user wants|the user says|instruction says|instructions say|"
    r"style guidance|style inspiration|preferred opening|first-stage|raw output|"
    r"object id|object ids|bbox id|bbox ids|source id|source ids|"
    r"we need to|we can mention|can mention|we can describe|we can include|"
    r"we should mention|we can say|the prompt asks|final answer only|this is ambiguous"
    r")",
    re.IGNORECASE,
)
_QWEN_COORDINATE_CONTEXT_RE = re.compile(
    r"(?:\[\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?)?\s*\]|"
    r"\b(?:from|to)\s+\d+\s*,\s*\d+\b)",
    re.IGNORECASE,
)
_QWEN_BBOX_ID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|"
    r"\b[0-9a-f]{24,64}\b",
    re.IGNORECASE,
)


def _thinking_caption_needs_cleanup(cleaned: str, raw: Optional[str]) -> bool:
    if not cleaned:
        return True
    if len(cleaned.split()) < 6:
        return True
    if raw and not re.search(r"<final>|\bFINAL\b", raw, re.IGNORECASE):
        return True
    if _QWEN_THINKING_REASONING_RE.search(cleaned):
        return True
    return False


def _caption_needs_completion(caption: str) -> bool:
    if not caption:
        return True
    trimmed = caption.strip()
    if not trimmed:
        return True
    terminal = re.sub(r"[\"')\]]+$", "", trimmed).rstrip()
    if not terminal or terminal[-1] not in ".!?":
        return True
    return _caption_has_dangling_tail(trimmed)


_CAPTION_DANGLING_TAIL_RE = re.compile(
    r"\b(?:"
    r"a|an|the|and|or|of|with|without|to|from|in|on|at|by|for|into|onto|"
    r"over|under|through|between|beside|near|along|across|including|featuring|"
    r"showing|shows|captures|contains|visible|surrounded|bordered"
    r")\s*$",
    re.IGNORECASE,
)


def _caption_has_dangling_tail(caption: str) -> bool:
    trimmed = re.sub(r"[.!?\"')\]]+$", "", str(caption or "").strip()).strip()
    if not trimmed:
        return True
    return _CAPTION_DANGLING_TAIL_RE.search(trimmed) is not None


def _caption_complete_sentence_units(text: str) -> List[str]:
    units = _caption_sentence_units(text)
    complete: List[str] = []
    for unit in units:
        sentence = unit.strip()
        if not sentence:
            continue
        if re.search(r"[.!?][\"')\]]*$", sentence) and not _caption_has_dangling_tail(sentence):
            complete.append(sentence)
    return complete


def _caption_trim_to_complete_sentences(
    caption: str,
    max_sentences: Optional[int] = None,
) -> Tuple[str, bool]:
    cleaned = _sanitize_qwen_caption(str(caption or ""))
    if not cleaned:
        return "", bool(caption)
    complete = _caption_complete_sentence_units(cleaned)
    if not complete:
        stripped = re.sub(r"[,;:]+$", "", cleaned).strip()
        if stripped and not _caption_has_dangling_tail(stripped):
            completed = f"{stripped}."
            return completed, completed != cleaned
        return "", True
    try:
        limit = int(max_sentences) if max_sentences is not None else None
    except (TypeError, ValueError):
        limit = None
    if limit is not None and limit > 0:
        complete = complete[:limit]
    finalized = _collapse_whitespace(" ".join(complete))
    return finalized, finalized != cleaned


def _caption_has_meta(caption: str) -> bool:
    if not caption:
        return False
    return (
        _QWEN_CAPTION_META_RE.search(caption) is not None
        or _QWEN_THINKING_REASONING_RE.search(caption) is not None
    )


def _strip_caption_meta_sentences(text: str) -> str:
    if not text:
        return ""
    kept: List[str] = []
    for unit in _caption_sentence_units(text):
        if _caption_has_meta(unit):
            continue
        kept.append(unit.strip())
    if not kept:
        return ""
    return _collapse_whitespace(" ".join(kept))


def _caption_source_context_has_meta(text: str) -> bool:
    if not text:
        return False
    return (
        _caption_has_meta(text)
        or _QWEN_COORDINATE_CONTEXT_RE.search(text) is not None
    )


def _strip_caption_source_context_meta_sentences(text: str) -> str:
    if not text:
        return ""
    kept: List[str] = []
    for unit in _caption_sentence_units(text):
        if _caption_source_context_has_meta(unit):
            continue
        kept.append(unit.strip())
    if not kept:
        return ""
    return _collapse_whitespace(" ".join(kept))


def _caption_needs_short_form(caption: str, max_words: int = 80, max_sentences: int = 2) -> bool:
    if not caption:
        return False
    words = caption.split()
    if len(words) > max_words:
        return True
    sentences = [s.strip() for s in re.split(r"[.!?]+", caption) if s.strip()]
    return len(sentences) > max_sentences


def _allowed_caption_labels_impl(label_hints: Sequence[Any]) -> List[str]:
    labels = []
    for entry in label_hints or []:
        label = str(_caption_hint_value(entry, "label", "") or "").strip()
        if not label:
            continue
        labels.append(label)
    return sorted(set(labels))


def _caption_degenerate_reason(caption: str, *, allow_short_caption: bool = False) -> Optional[str]:
    if not caption:
        return "empty"
    trimmed = caption.strip()
    if not trimmed:
        return "empty"
    surface_reason = _caption_surface_loop_reason(trimmed)
    if surface_reason:
        return surface_reason
    if _QWEN_THINKING_REASONING_RE.search(trimmed):
        return "thinking_reasoning"
    if _caption_repetition_loop_detected(trimmed):
        return "repetition_loop"
    words = caption.split()
    if not allow_short_caption and len(words) < 8:
        return "too_short"
    sentences = [s.strip().lower() for s in re.split(r"[.!?]+", caption) if s.strip()]
    if sentences:
        counts = Counter(sentences)
        most_common = counts.most_common(1)[0][1]
        if most_common >= 3:
            return "repeated_sentence"
        if len(sentences) >= 3 and most_common >= 2 and most_common / len(sentences) > 0.45:
            return "repeated_sentence"
    if len(words) > 40:
        tokens = [w.lower() for w in words]
        bigrams = list(zip(tokens, tokens[1:], strict=False))
        if bigrams:
            unique_ratio = len(set(bigrams)) / len(bigrams)
            if unique_ratio < 0.55:
                return "low_bigram_diversity"
    return None


def _caption_is_degenerate_impl(caption: str) -> bool:
    return _caption_degenerate_reason(caption) is not None


def _resolve_qwen_caption_decode(payload: Any, is_thinking: bool) -> Dict[str, Any]:
    def _float_param(name: str, default: float, minimum: float, maximum: float) -> float:
        value = _caption_finite_float(getattr(payload, name, None))
        if value is None:
            return default
        return max(minimum, min(value, maximum))

    def _int_param(name: str, default: int, minimum: int, maximum: int) -> int:
        try:
            raw = getattr(payload, name, None)
            raw_float = float(raw)
            if not math.isfinite(raw_float):
                return default
            value = int(raw_float)
        except (TypeError, ValueError, OverflowError):
            return default
        return max(minimum, min(value, maximum))

    use_sampling = payload.use_sampling if getattr(payload, "use_sampling", None) is not None else True
    if not use_sampling:
        if is_thinking:
            return {
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0,
                "repetition_context_size": 128,
                "no_repeat_ngram_size": 8,
            }
        return {
            "do_sample": False,
            "repetition_penalty": 1.08,
            "repetition_context_size": 128,
            "no_repeat_ngram_size": 8,
        }
    defaults = {
        "temperature": 0.6 if is_thinking else 0.7,
        "top_p": 0.95 if is_thinking else 0.8,
        "top_k": 20,
        "presence_penalty": 0.0 if is_thinking else 1.5,
        "repetition_penalty": 1.0 if is_thinking else 1.10,
        "repetition_context_size": 128,
        "no_repeat_ngram_size": 8,
    }
    temperature = _float_param("temperature", defaults["temperature"], 0.01, 2.0)
    top_p = _float_param("top_p", defaults["top_p"], 0.01, 1.0)
    top_k = _int_param("top_k", defaults["top_k"], 1, 1000)
    presence_penalty = _float_param("presence_penalty", defaults["presence_penalty"], -2.0, 2.0)
    return {
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
        "repetition_penalty": defaults["repetition_penalty"],
        "repetition_context_size": defaults["repetition_context_size"],
        "no_repeat_ngram_size": defaults["no_repeat_ngram_size"],
    }


def _adjust_prompt_for_thinking(prompt_text: str) -> str:
    return prompt_text


def _caption_length_instruction(max_sentences: Optional[int], *, final: bool = True) -> str:
    if max_sentences is None:
        return ""
    try:
        sentence_count = int(max_sentences)
    except (TypeError, ValueError):
        return ""
    if sentence_count <= 0:
        return ""
    if sentence_count == 1:
        return "Return exactly one complete sentence. "
    if sentence_count <= 2:
        subject = "final caption" if final else "caption"
        return f"Use at most {sentence_count} complete sentences in the {subject}; keep it concise. "
    subject = "final caption" if final else "caption"
    return (
        f"The {subject} may be long when the image contains many visible details; "
        f"use up to {sentence_count} complete sentences when needed to preserve concrete detail. "
        "Do not collapse distinct visible objects, actions, or attributes merely for brevity. "
    )


def _caption_user_request_instruction(user_prompt: Optional[str]) -> str:
    cleaned = _collapse_whitespace(str(user_prompt or ""))
    if not cleaned:
        return ""
    return (
        f"User caption request: {cleaned}\n"
        "Preserve the user's requested angle in the final caption. If the request asks a question "
        "or asks for an inference such as likely location, scene type, event, or time, answer it "
        "when the image supports a grounded answer; otherwise state uncertainty briefly. "
        "Do not mention that a request or hint was provided.\n"
    )


def _clean_caption_source_context_text(text: str) -> str:
    cleaned = _sanitize_qwen_caption(str(text or ""))
    cleaned = _strip_caption_source_context_meta_sentences(cleaned)
    if _caption_source_context_has_meta(cleaned):
        return ""
    if _caption_repetition_loop_detected(cleaned):
        return ""
    if len(cleaned.split()) < 3:
        return ""
    return cleaned


def _format_caption_source_output_context(
    source_output: Optional[str] = None,
    source_outputs: Optional[Sequence[Tuple[str, str]]] = None,
) -> str:
    sections: List[Tuple[str, str]] = []
    if source_output:
        sections.append(("Raw first-stage output", str(source_output)))
    for label, text in source_outputs or []:
        sections.append((str(label or "First-stage output"), str(text or "")))
    cleaned_sections: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for label, text in sections:
        cleaned = _clean_caption_source_context_text(text)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        cleaned_sections.append((label.strip() or "First-stage output", cleaned))
    if not cleaned_sections:
        return ""
    lines = [
        "First-stage model output context for detail preservation. "
        "Use this only to recover concrete visible details; ignore reasoning, instructions, labels, hints, counts, or coordinates."
    ]
    for label, text in cleaned_sections:
        lines.append(f"{label}:\n{text}")
    return "\n\n".join(lines)


def _caption_window_global_region(
    x0: int,
    y0: int,
    size: int,
    image_width: Optional[int],
    image_height: Optional[int],
    *,
    row_index: Optional[int] = None,
    col_index: Optional[int] = None,
    row_count: Optional[int] = None,
    col_count: Optional[int] = None,
) -> str:
    try:
        safe_width = max(1, int(image_width or 0))
        safe_height = max(1, int(image_height or 0))
    except (TypeError, ValueError):
        safe_width = safe_height = 0
    try:
        left = max(0, int(round(float(x0))))
        top = max(0, int(round(float(y0))))
        crop_size = max(1, int(round(float(size))))
    except (TypeError, ValueError):
        return "global crop location unavailable"
    if safe_width <= 0 or safe_height <= 0:
        return "global crop location unavailable"
    right = min(safe_width, left + crop_size)
    bottom = min(safe_height, top + crop_size)
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0

    def _axis_word(start: int, end: int, full: int, center: float, low: str, mid: str, high: str) -> str:
        if start <= 0 and end >= full:
            return f"full-{low}-{high}"
        if start <= 0:
            return low
        if end >= full:
            return high
        if center < full / 3.0:
            return low
        if center > full * 2 / 3.0:
            return high
        return mid

    horiz = _axis_word(left, right, safe_width, center_x, "left", "center", "right")
    vert = _axis_word(top, bottom, safe_height, center_y, "upper", "middle", "lower")
    x0_pct = int(round((left / safe_width) * 100))
    x1_pct = int(round((right / safe_width) * 100))
    y0_pct = int(round((top / safe_height) * 100))
    y1_pct = int(round((bottom / safe_height) * 100))
    grid_note = ""
    if (
        row_index is not None
        and col_index is not None
        and row_count is not None
        and col_count is not None
        and row_count > 1
        and col_count > 1
    ):
        row_word = {
            0: "first-row",
            row_count - 1: "last-row",
        }.get(row_index, f"row {row_index + 1}")
        col_word = {
            0: "first-column",
            col_count - 1: "last-column",
        }.get(col_index, f"column {col_index + 1}")
        grid_note = f"{row_word}, {col_word} "
    overlap_note = ""
    if (right - left) > safe_width / 2.0 or (bottom - top) > safe_height / 2.0:
        overlap_note = ", overlapping toward the image center"
    return (
        f"{grid_note}global {vert}-{horiz} section{overlap_note}; "
        f"covers x {x0_pct}-{x1_pct}% and y {y0_pct}-{y1_pct}% of the full image"
    )


def _caption_bbox_tuple(bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    if not bbox or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _caption_bbox_area(box: Optional[Tuple[float, float, float, float]]) -> float:
    if not box:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _caption_bbox_intersection_area(
    a: Optional[Tuple[float, float, float, float]],
    b: Optional[Tuple[float, float, float, float]],
) -> float:
    if not a or not b:
        return 0.0
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _caption_bbox_iou(
    a: Optional[Tuple[float, float, float, float]],
    b: Optional[Tuple[float, float, float, float]],
) -> float:
    inter = _caption_bbox_intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = _caption_bbox_area(a) + _caption_bbox_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _caption_bbox_center_inside(
    inner_center_box: Optional[Tuple[float, float, float, float]],
    outer_box: Optional[Tuple[float, float, float, float]],
) -> bool:
    if not inner_center_box or not outer_box:
        return False
    cx = (inner_center_box[0] + inner_center_box[2]) * 0.5
    cy = (inner_center_box[1] + inner_center_box[3]) * 0.5
    return outer_box[0] <= cx <= outer_box[2] and outer_box[1] <= cy <= outer_box[3]


def _caption_position_phrase_for_bbox(
    bbox: Sequence[float],
    image_width: Optional[int],
    image_height: Optional[int],
) -> str:
    box = _caption_bbox_tuple(bbox)
    if not box:
        return ""
    safe_width = max(1.0, float(image_width or 1))
    safe_height = max(1.0, float(image_height or 1))
    cx = max(0.0, min(safe_width, (box[0] + box[2]) * 0.5))
    cy = max(0.0, min(safe_height, (box[1] + box[3]) * 0.5))
    horiz = "left" if cx < safe_width / 3.0 else "right" if cx > safe_width * 2.0 / 3.0 else "center"
    vert = "upper" if cy < safe_height / 3.0 else "lower" if cy > safe_height * 2.0 / 3.0 else "middle"
    if horiz == "center" and vert == "middle":
        return "near the center"
    if horiz == "center":
        return f"in the {vert} center"
    if vert == "middle":
        return f"near the {horiz} side"
    return f"in the {vert}-{horiz}"


def _caption_full_object_refs(
    full_label_hints: Optional[Sequence[Any]],
    glossary_map: Optional[Mapping[str, Sequence[Any]]],
    image_width: Optional[int],
    image_height: Optional[int],
) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for index, hint in enumerate(full_label_hints or [], start=1):
        raw_label = str(_caption_hint_value(hint, "label", "") or "").strip()
        bbox = _caption_bbox_tuple(_caption_hint_value(hint, "bbox", None))
        if not raw_label or not bbox:
            continue
        display = _caption_preferred_label(raw_label, glossary_map)
        refs.append(
            {
                "id": f"object_{index:03d}",
                "label": raw_label,
                "display": display or _caption_natural_label(raw_label),
                "bbox": bbox,
                "source_id": str(_caption_hint_value(hint, "source_id", "") or "").strip(),
                "position": _caption_position_phrase_for_bbox(bbox, image_width, image_height),
            }
        )
    return refs


def _caption_window_hint_full_bbox(
    hint: Any,
    x0: int,
    y0: int,
    image_width: Optional[int],
    image_height: Optional[int],
) -> Optional[Tuple[float, float, float, float]]:
    local_box = _caption_bbox_tuple(_caption_hint_value(hint, "bbox", None))
    if not local_box:
        return None
    safe_width = max(1.0, float(image_width or 1))
    safe_height = max(1.0, float(image_height or 1))
    left = max(0.0, min(safe_width, float(x0) + local_box[0]))
    top = max(0.0, min(safe_height, float(y0) + local_box[1]))
    right = max(0.0, min(safe_width, float(x0) + local_box[2]))
    bottom = max(0.0, min(safe_height, float(y0) + local_box[3]))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _caption_labels_compatible(raw_label: str, ref: Mapping[str, Any], display: str) -> bool:
    raw_key = str(raw_label or "").strip().lower()
    if not raw_key:
        return False
    if raw_key == str(ref.get("label") or "").strip().lower():
        return True
    return bool(display and display.lower() == str(ref.get("display") or "").strip().lower())


def _caption_match_window_hint_to_full_ref(
    hint: Any,
    full_box: Optional[Tuple[float, float, float, float]],
    full_refs: Sequence[Mapping[str, Any]],
    glossary_map: Optional[Mapping[str, Sequence[Any]]],
) -> Tuple[str, Optional[Mapping[str, Any]], str]:
    raw_label = str(_caption_hint_value(hint, "label", "") or "").strip()
    display = _caption_preferred_label(raw_label, glossary_map)
    source_id = str(_caption_hint_value(hint, "source_id", "") or "").strip()
    if not raw_label or not full_box:
        return "unmatched", None, "missing label or bbox"

    candidates: List[Tuple[float, Mapping[str, Any], str]] = []
    window_area = max(_caption_bbox_area(full_box), 1e-6)
    for ref in full_refs:
        if not _caption_labels_compatible(raw_label, ref, display):
            continue
        ref_box = ref.get("bbox")
        ref_source_id = str(ref.get("source_id") or "").strip()
        if source_id and ref_source_id and source_id == ref_source_id:
            candidates.append((100.0, ref, "same source id"))
            continue
        inter = _caption_bbox_intersection_area(full_box, ref_box)
        if inter <= 0:
            continue
        iou = _caption_bbox_iou(full_box, ref_box)
        window_coverage = inter / window_area
        ref_area = max(_caption_bbox_area(ref_box), 1e-6)
        ref_coverage = inter / ref_area
        center_inside = _caption_bbox_center_inside(full_box, ref_box)
        if iou < 0.05 and window_coverage < 0.5 and not center_inside:
            continue
        score = iou * 3.0 + window_coverage * 2.0 + ref_coverage
        if center_inside:
            score += 0.75
        reason = (
            f"IoU {iou:.2f}, window coverage {window_coverage:.2f}, "
            f"object coverage {ref_coverage:.2f}"
        )
        candidates.append((score, ref, reason))
    if not candidates:
        return "unmatched", None, "no compatible full-frame object"
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] < 100 and abs(candidates[0][0] - candidates[1][0]) < 0.05:
        return "ambiguous", None, "multiple compatible full-frame objects"
    return "matched", candidates[0][1], candidates[0][2]


def _format_caption_reconciled_window_evidence(
    x0: int,
    y0: int,
    window_hints: Sequence[Any],
    full_refs: Sequence[Mapping[str, Any]],
    glossary_map: Optional[Mapping[str, Sequence[Any]]],
    image_width: Optional[int],
    image_height: Optional[int],
    *,
    max_items: int = 60,
) -> List[str]:
    if not full_refs and not window_hints:
        return []
    matched: List[str] = []
    unmatched: List[str] = []
    ambiguous: List[str] = []
    seen_refs: set[str] = set()
    for hint in window_hints or []:
        raw_label = str(_caption_hint_value(hint, "label", "") or "").strip()
        display = _caption_preferred_label(raw_label, glossary_map)
        full_box = _caption_window_hint_full_bbox(hint, x0, y0, image_width, image_height)
        status, ref, reason = _caption_match_window_hint_to_full_ref(
            hint,
            full_box,
            full_refs,
            glossary_map,
        )
        position = _caption_position_phrase_for_bbox(full_box or [], image_width, image_height)
        if status == "matched" and ref:
            ref_id = str(ref.get("id") or "")
            if ref_id and ref_id in seen_refs:
                continue
            seen_refs.add(ref_id)
            label = str(ref.get("display") or display or raw_label)
            ref_position = str(ref.get("position") or position or "").strip()
            matched.append(
                " ".join(
                    part
                    for part in (
                        ref_id,
                        label,
                        ref_position,
                        f"({reason})" if reason else "",
                    )
                    if part
                )
            )
        elif status == "ambiguous":
            ambiguous.append(" ".join(part for part in (display or raw_label, position, f"({reason})") if part))
        else:
            unmatched.append(" ".join(part for part in (display or raw_label, position, f"({reason})") if part))

    lines: List[str] = []
    if matched:
        shown = matched[:max_items]
        suffix = f"; {len(matched) - len(shown)} more matched objects not listed" if len(shown) < len(matched) else ""
        lines.append("Matched full-frame objects in this window: " + "; ".join(shown) + suffix + ".")
    if ambiguous:
        shown = ambiguous[:max_items]
        suffix = f"; {len(ambiguous) - len(shown)} more ambiguous items not listed" if len(shown) < len(ambiguous) else ""
        lines.append(
            "Ambiguous window evidence: "
            + "; ".join(shown)
            + suffix
            + ". Use as visual detail only, not as a new count."
        )
    if unmatched:
        shown = unmatched[:max_items]
        suffix = f"; {len(unmatched) - len(shown)} more unmatched items not listed" if len(shown) < len(unmatched) else ""
        lines.append(
            "Window-only evidence not matched to the full-frame inventory: "
            + "; ".join(shown)
            + suffix
            + ". Use as scene context only, not as extra object inventory."
        )
    if window_hints and not lines:
        lines.append("Window detections could not be reconciled; use this observation as visual context only.")
    return lines


def _format_caption_window_observation_lines(
    windowed_captions: Sequence[Tuple[int, int, int, str]],
    *,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    full_label_hints: Optional[Sequence[Any]] = None,
    window_hints_by_window: Optional[Mapping[Tuple[int, int], Sequence[Any]]] = None,
    glossary_map: Optional[Mapping[str, Sequence[Any]]] = None,
) -> List[str]:
    lines = [
        "Window observations (reconciled close-up evidence; do NOT invent objects):",
        (
            "Object reconciliation: window-local detection boxes are matched back to full-image annotation objects when possible. "
            "Object reference IDs are internal merge aids; never output them. "
            "Full-frame counts remain authoritative; matched window evidence only adds visual details."
        ),
        (
            "Spatial grounding: each observation is only one crop of the full image. "
            "Absolute local words in a crop caption such as top, bottom, left, right, center, or corner are crop-relative. "
            "Use the crop global region and percent bounds below to restate positions in full-image terms, and do not copy local spatial wording when it would be misleading globally. "
            "The percent bounds are prompt-only layout aids; do not mention crop coordinates or percentages in the final caption."
        ),
    ]
    full_refs = _caption_full_object_refs(full_label_hints, glossary_map, image_width, image_height)
    x_order = sorted({int(x0) for x0, _y0, _size, _caption in windowed_captions})
    y_order = sorted({int(y0) for _x0, y0, _size, _caption in windowed_captions})
    x_to_col = {value: idx for idx, value in enumerate(x_order)}
    y_to_row = {value: idx for idx, value in enumerate(y_order)}
    for index, (x0, y0, size, caption) in enumerate(windowed_captions, start=1):
        text = _collapse_whitespace(str(caption or ""))
        if not text:
            continue
        region = _caption_window_global_region(
            x0,
            y0,
            size,
            image_width,
            image_height,
            row_index=y_to_row.get(int(y0)),
            col_index=x_to_col.get(int(x0)),
            row_count=len(y_order),
            col_count=len(x_order),
        )
        evidence_lines = _format_caption_reconciled_window_evidence(
            x0,
            y0,
            (window_hints_by_window or {}).get((x0, y0), []),
            full_refs,
            glossary_map,
            image_width,
            image_height,
        )
        evidence_text = (" " + " ".join(evidence_lines)) if evidence_lines else ""
        lines.append(f"- Window {index} ({region}): {text}")
        if evidence_text:
            lines[-1] = f"{lines[-1]}{evidence_text}"
    return lines


def _run_qwen_caption_cleanup(
    prompt: str,
    pil_img: Any,
    max_new_tokens: int,
    base_model_id: str,
    use_caption_cache: bool,
    *,
    model_id_override: Optional[str] = None,
    runtime_override: Optional[Tuple[Any, Any]] = None,
    allowed_labels: Optional[List[str]] = None,
    strict: bool = False,
    minimal_edit: bool = False,
    max_sentences: Optional[int] = None,
    user_prompt: Optional[str] = None,
    source_output: Optional[str] = None,
    source_outputs: Optional[Sequence[Tuple[str, str]]] = None,
    authoritative_counts_note: Optional[str] = None,
    glossary_line: Optional[str] = None,
    cleanup_prompt_override: Optional[str] = None,
    cleanup_system_prompt_override: Optional[str] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    run_qwen_inference_fn: Callable[..., Tuple[str, Any, Any]],
    run_qwen_text_inference_fn: Optional[Callable[..., Tuple[str, Any, Any]]] = None,
    resolve_variant_fn: Callable[[str, str], str],
    extract_caption_fn: Callable[[str, Optional[str]], Tuple[str, bool]],
    sanitize_caption_fn: Callable[[str], str],
) -> str:
    allowed_note = ""
    if allowed_labels:
        allowed_note = (
            f"Only mention these classes if they appear: {', '.join(sorted(set(allowed_labels)))}. "
            "Do not introduce any other entity types. "
        )
    strict_note = _caption_length_instruction(max_sentences) if strict else ""
    if strict and not strict_note:
        strict_note = "Return exactly one complete sentence. "
    minimal_note = (
        "Edit the draft with minimal changes. Do not introduce new objects or actions. "
        if minimal_edit
        else ""
    )
    cleanup_system = (
        _collapse_whitespace(cleanup_system_prompt_override or "")
        or (
            "You are a captioning assistant. Respond in English only. "
            "Return only <final>...</final> and nothing else."
        )
    )
    user_request_note = _caption_user_request_instruction(user_prompt)
    source_context = _format_caption_source_output_context(source_output, source_outputs)
    cleanup_policy = (
        _collapse_whitespace(cleanup_prompt_override or "")
        or (
            "Remove repetition, avoid coordinates, and remove any mention of labels, hints, or counts being provided. "
            "Never copy planning phrases such as 'we can mention', 'we need to', or 'the user wants'. "
            f"{_CAPTION_EDITOR_PRESERVE_BROAD_TERMS} "
            "If the draft is mostly reasoning or planning text, ignore that draft and write a fresh image-grounded caption. "
            "Keep the caption grounded in the image."
        )
    )
    cleanup_prompt = (
        f"{strict_note}{allowed_note}{minimal_note}"
        f"{user_request_note}"
        f"{_collapse_whitespace(glossary_line or '') + ' ' if glossary_line else ''}"
        f"{_collapse_whitespace(authoritative_counts_note or '') + ' ' if authoritative_counts_note else ''}"
        f"{cleanup_policy}\n"
        f"Draft caption: {prompt}"
    )
    if source_context:
        cleanup_prompt = f"{cleanup_prompt}\n\n{source_context}"
    cleanup_model = model_id_override or resolve_variant_fn(base_model_id, "Instruct")
    cleanup_decode = (
        {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.0,
            "repetition_context_size": 128,
            "no_repeat_ngram_size": 8,
        }
        if "thinking" in str(cleanup_model or "").lower()
        else {"do_sample": False}
    )
    if run_qwen_text_inference_fn is not None:
        qwen_text, _, _ = run_qwen_text_inference_fn(
            cleanup_prompt,
            max_new_tokens=max_new_tokens,
            system_prompt_override=cleanup_system,
            model_id_override=cleanup_model if use_caption_cache and runtime_override is None else None,
            runtime_override=runtime_override,
            decode_override=cleanup_decode,
            chat_template_kwargs=chat_template_kwargs,
        )
    else:
        qwen_text, _, _ = run_qwen_inference_fn(
            cleanup_prompt,
            pil_img,
            max_new_tokens=max_new_tokens,
            system_prompt_override=cleanup_system,
            model_id_override=cleanup_model if use_caption_cache and runtime_override is None else None,
            runtime_override=runtime_override,
            decode_override=cleanup_decode,
            chat_template_kwargs=chat_template_kwargs,
        )
    caption_text, _ = extract_caption_fn(qwen_text, marker=None)
    cleaned = sanitize_caption_fn(caption_text)
    if _caption_degenerate_reason(cleaned, allow_short_caption=True):
        return sanitize_caption_fn(prompt)
    return cleaned


def _run_qwen_caption_merge(
    draft_caption: str,
    windowed_captions: Sequence[Tuple[int, int, int, str]],
    *,
    pil_img: Any,
    base_model_id: str,
    runtime_resolver: Callable[[str], Tuple[Any, Any]],
    max_new_tokens: int,
    glossary_line: Optional[str] = None,
    model_id_override: Optional[str] = None,
    max_sentences: Optional[int] = None,
    user_prompt: Optional[str] = None,
    overlap_guidance: Optional[str] = None,
    source_outputs: Optional[Sequence[Tuple[str, str]]] = None,
    authoritative_counts_note: Optional[str] = None,
    full_label_hints: Optional[Sequence[Any]] = None,
    window_hints_by_window: Optional[Mapping[Tuple[int, int], Sequence[Any]]] = None,
    glossary_map: Optional[Mapping[str, Sequence[Any]]] = None,
    merge_prompt_override: Optional[str] = None,
    merge_system_prompt_override: Optional[str] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    run_qwen_inference_fn: Callable[..., Tuple[str, Any, Any]],
    run_qwen_text_inference_fn: Optional[Callable[..., Tuple[str, Any, Any]]] = None,
    resolve_variant_fn: Callable[[str, str], str],
    extract_caption_fn: Callable[[str, Optional[str]], Tuple[str, bool]],
    sanitize_caption_fn: Callable[[str], str],
) -> str:
    if not draft_caption or not windowed_captions:
        return draft_caption
    image_width = getattr(pil_img, "width", None)
    image_height = getattr(pil_img, "height", None)
    window_lines = _format_caption_window_observation_lines(
        windowed_captions,
        image_width=image_width,
        image_height=image_height,
        full_label_hints=full_label_hints,
        window_hints_by_window=window_hints_by_window,
        glossary_map=glossary_map,
    )
    if len(window_lines) <= 2:
        return draft_caption
    length_note = _caption_length_instruction(max_sentences)
    short_requested = _caption_user_requested_short(user_prompt, max_sentences)
    user_request_note = _caption_user_request_instruction(user_prompt)
    source_context = _format_caption_source_output_context(source_outputs=source_outputs)
    detail_policy = (
        "Keep the revised caption concise. Fold in only the most important missing visible details from the windows. "
        if short_requested
        else (
            "Use multiple sentences if needed. Preserve specific counts, actions, and notable attributes from the windows; "
            "treat window text as supporting evidence, not a complete object inventory. Ignore extra object lists "
            "or quantities that conflict with the full image or authoritative counts. "
        )
    )
    merge_policy = (
        _collapse_whitespace(merge_prompt_override or "")
        or (
            "Revise the draft caption so it includes all distinct object details "
            "from the window observations that are missing in the draft. "
            "Do not invent new objects, and do not turn background window descriptions into extra counted object categories. "
            f"{_CAPTION_EDITOR_PRESERVE_BROAD_TERMS} "
            f"{detail_policy}"
            f"{length_note}"
            "Do not mention labels, hints, or coordinates."
        )
    )
    merge_prompt = (
        f"{user_request_note}"
        f"{_collapse_whitespace(glossary_line or '') + ' ' if glossary_line else ''}"
        f"{_collapse_whitespace(authoritative_counts_note or '') + ' ' if authoritative_counts_note else ''}"
        f"{merge_policy}\n"
        f"Draft caption: {draft_caption}\n"
        + "\n".join(window_lines)
    )
    if overlap_guidance:
        merge_prompt = f"{merge_prompt}\n\n{overlap_guidance}"
    if source_context:
        merge_prompt = f"{merge_prompt}\n\n{source_context}"
    merge_system = (
        _collapse_whitespace(merge_system_prompt_override or "")
        or "You are a caption editor. Return only the revised caption in English."
    )
    merge_model = model_id_override or resolve_variant_fn(base_model_id, "Instruct")
    merge_decode = (
        {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.0,
            "repetition_context_size": 128,
            "no_repeat_ngram_size": 8,
        }
        if "thinking" in str(merge_model or "").lower()
        else {"do_sample": False}
    )
    run_editor_inference = run_qwen_text_inference_fn or run_qwen_inference_fn
    if run_qwen_text_inference_fn is not None:
        qwen_text, _, _ = run_editor_inference(
            merge_prompt,
            max_new_tokens=max_new_tokens,
            system_prompt_override=merge_system,
            model_id_override=merge_model,
            runtime_override=runtime_resolver(merge_model),
            decode_override=merge_decode,
            chat_template_kwargs=chat_template_kwargs,
        )
    else:
        qwen_text, _, _ = run_editor_inference(
            merge_prompt,
            pil_img,
            max_new_tokens=max_new_tokens,
            system_prompt_override=merge_system,
            model_id_override=merge_model,
            runtime_override=runtime_resolver(merge_model),
            decode_override=merge_decode,
            chat_template_kwargs=chat_template_kwargs,
        )
    merged, _ = extract_caption_fn(qwen_text, marker=None)
    merged = sanitize_caption_fn(merged)
    if _caption_degenerate_reason(merged, allow_short_caption=True):
        return draft_caption
    return merged or draft_caption


def _resolve_qwen_window_size(
    requested: Optional[int],
    image_width: int,
    image_height: int,
    *,
    overlap: Optional[float] = None,
    default_size: int = 672,
    default_overlap: float = 0.2,
) -> int:
    try:
        safe_width = max(1, int(image_width))
        safe_height = max(1, int(image_height))
    except (TypeError, ValueError, OverflowError):
        safe_width = safe_height = max(1, int(default_size or 672))
    if requested is None:
        overlap_val = _resolve_qwen_window_overlap(overlap, default_overlap=default_overlap)
        base_dim = max(1, min(safe_width, safe_height))
        # Default to two overlapping windows on the shorter axis; the longer axis
        # may need more positions to avoid corner-only coverage.
        base = base_dim / max(1.0, 2.0 - overlap_val)
    else:
        base = requested
    try:
        base_float = float(base)
        if not math.isfinite(base_float):
            raise ValueError("nonfinite_window_size")
        base = int(base_float)
    except (TypeError, ValueError, OverflowError):
        base = default_size
    base = max(128, min(base, 4096))
    return max(1, min(base, safe_width, safe_height))


def _resolve_qwen_window_overlap(requested: Optional[float], *, default_overlap: float = 0.2) -> float:
    try:
        overlap = float(requested) if requested is not None else default_overlap
    except (TypeError, ValueError, OverflowError):
        overlap = default_overlap
    if not math.isfinite(overlap):
        overlap = default_overlap
    try:
        default = float(default_overlap)
    except (TypeError, ValueError, OverflowError):
        default = 0.2
    if not math.isfinite(default):
        default = 0.2
    overlap = overlap if math.isfinite(overlap) else default
    return max(0.0, min(overlap, 0.2))


def _resolve_qwen_variant_model_id_impl(base_model_id: str, variant: Optional[str]) -> str:
    if not variant or variant == "auto":
        return base_model_id
    if "Thinking" in base_model_id:
        if variant == "Thinking":
            return base_model_id
        if variant == "Instruct":
            return base_model_id.replace("Thinking", "Instruct")
    if "Instruct" in base_model_id:
        if variant == "Instruct":
            return base_model_id
        if variant == "Thinking":
            return base_model_id.replace("Instruct", "Thinking")
    return base_model_id


def _strip_qwen_model_suffix_impl(model_id: str) -> Optional[str]:
    base = str(model_id or "")
    if not base:
        return None
    if base.endswith("-2507"):
        return base[: -len("-2507")]
    for suffix in ("-GPTQ-Int4", "-GPTQ-Int8", "-GGUF", "-AWQ", "-INT4", "-INT8"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    if ":" in base:
        return base.split(":", 1)[0]
    if "@" in base:
        return base.split("@", 1)[0]
    return None


def _format_qwen_load_error_impl(exc: Exception, *, torch_module: Any) -> str:
    msg = str(exc)
    if "Missing" in msg and "vision_tower." in msg:
        return (
            "incompatible_checkpoint_missing_vision_tower: selected checkpoint is missing "
            "vision tower weights and cannot run Qwen3-VL image inference. Pick a full "
            "Qwen3-VL checkpoint for captioning or detection."
        )
    if "FP8" in msg and "compute capability" in msg:
        cc = None
        if torch_module.cuda.is_available():
            try:
                major, minor = torch_module.cuda.get_device_capability(torch_module.cuda.current_device())
                cc = f"{major}.{minor}"
            except Exception:
                cc = None
        cc_note = f" Current GPU compute capability: {cc}." if cc else ""
        return (
            f"{msg} FP8 models require GPU compute capability >= 8.9 (e.g., 4090/H100)."
            f"{cc_note} Use a non-FP8 model on lower-capability GPUs."
        )
    return msg


def _get_qwen_prompt_config_impl(config: Any, lock: Any) -> Any:
    with lock:
        return config.copy(deep=True)


def _render_qwen_prompt_impl(
    prompt_type: str,
    *,
    items: Optional[str],
    image_type: Optional[str],
    extra_context: Optional[str],
    get_config_fn: Any,
    http_exception_cls: Any,
    http_422: int,
) -> str:
    if not items:
        raise http_exception_cls(status_code=http_422, detail="qwen_items_required")
    config = get_config_fn()
    section_name = "bbox" if prompt_type in {"bbox", "bbox_sam"} else prompt_type
    section = getattr(config, section_name)
    template = (section.base_prompt or "{items}").strip()
    image_value = (image_type or section.default_image_type or "image").strip() or "image"
    extra_value = extra_context if extra_context is not None and extra_context.strip() else section.default_extra_context
    formatted = template.format(
        image_type=image_value,
        items=items.strip(),
        extra_context=(extra_value or "").strip(),
    )
    return formatted.strip()


def _extract_qwen_json_block_impl(text: str) -> tuple[str, list]:
    def _attempt_parse(raw: str):
        snippet = (raw or "").strip()
        if not snippet:
            return None
        snippet = snippet.strip("`").strip()

        parsed = None
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError:
            parsed = None

        if parsed is None:
            for start_char, end_char in (("{", "}"), ("[", "]")):
                start = snippet.find(start_char)
                end = snippet.rfind(end_char)
                if start < 0 or end < 0 or end <= start:
                    continue
                candidate = snippet[start : end + 1]
                try:
                    parsed = json.loads(candidate)
                    snippet = candidate
                    break
                except json.JSONDecodeError:
                    parsed = None

        if parsed is None:
            return None

        if isinstance(parsed, dict):
            if "detections" in parsed and isinstance(parsed["detections"], list):
                return snippet, [item for item in parsed["detections"] if isinstance(item, dict)]
            return snippet, [parsed]
        if isinstance(parsed, list):
            return snippet, [item for item in parsed if isinstance(item, dict)]
        return None

    fenced = re.findall(r"```(?:[a-zA-Z0-9_-]+)?\\s*(.*?)```", text or "", flags=re.DOTALL)
    for raw in [*fenced, text]:
        parsed = _attempt_parse(raw)
        if parsed is not None:
            return parsed

    # Strip any <final> style wrapper content
    cleaned = re.sub(r"<\\s*/?final\\s*>", "", text or "", flags=re.IGNORECASE).strip()
    parsed = _attempt_parse(cleaned)
    if parsed is not None:
        return parsed

    return "", []


def _strip_qwen_model_suffix_impl(model_id: str) -> Optional[str]:
    base = str(model_id or "")
    if not base:
        return None
    if base.endswith("-2507"):
        return base[: -len("-2507")]
    for suffix in ("-FP8", "-GPTQ-Int4", "-GPTQ-Int8", "-GGUF", "-AWQ", "-INT4", "-INT8"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    if ":" in base:
        return base.split(":", 1)[0]
    if "@" in base:
        return base.split("@", 1)[0]
    return None


def _format_qwen_load_error_impl(exc: Exception, *, torch_module: Any = None) -> str:
    text = str(exc or "")
    if not text:
        return "unknown"
    if "Missing" in text and "vision_tower." in text:
        return (
            "incompatible_checkpoint_missing_vision_tower: selected checkpoint is missing "
            "vision tower weights and cannot run Qwen3-VL image inference. Pick a full "
            "Qwen3-VL checkpoint for captioning or detection."
        )
    if "FP8" in text and "compute capability" in text:
        cc = None
        if torch_module is not None and torch_module.cuda.is_available():
            try:
                major, minor = torch_module.cuda.get_device_capability(torch_module.cuda.current_device())
                cc = f"{major}.{minor}"
            except Exception:
                cc = None
        cc_note = f" Current GPU compute capability: {cc}." if cc else ""
        return (
            f"{text} FP8 models require GPU compute capability >= 8.9 (e.g., 4090/H100)."
            f"{cc_note} Use an AWQ, GPTQ, or BF16 Qwen3-VL model on lower-capability GPUs."
        )
    for needle in (
        "FileNotFoundError",
        "No such file",
        "not found",
        "RepositoryNotFoundError",
        "GatedRepoError",
    ):
        if needle in text:
            return "missing_weights"
    for needle in ("CUDA out of memory", "out of memory"):
        if needle in text:
            return "oom"
    return text[:200]


def _window_positions_impl(
    total: int,
    window: int,
    overlap: float,
    *,
    force_two: bool = False,
) -> List[int]:
    try:
        safe_total = max(1, int(total))
    except (TypeError, ValueError, OverflowError):
        safe_total = 1
    try:
        safe_window = max(1, int(window))
    except (TypeError, ValueError, OverflowError):
        safe_window = safe_total
    try:
        safe_overlap = float(overlap)
    except (TypeError, ValueError, OverflowError):
        safe_overlap = 0.0
    if not math.isfinite(safe_overlap):
        safe_overlap = 0.0
    safe_overlap = max(0.0, min(safe_overlap, 0.95))
    if safe_total <= safe_window:
        return [0]
    step = max(1, int(round(safe_window * (1.0 - safe_overlap))))
    positions = list(range(0, max(1, safe_total - safe_window + 1), step))
    last = safe_total - safe_window
    if positions[-1] != last:
        positions.append(last)
    return sorted(set(positions))
