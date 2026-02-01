from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from utils.glossary import _normalize_labelmap_glossary, _parse_glossary_mapping


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


def _parse_prompt_candidates(raw: str, seen: set[str], limit: int) -> List[str]:
    """Parse and validate a comma/list output into cleaned candidates; returns [] if invalid."""
    if not raw:
        return []
    parts = re.split(r"[,;\n]+", raw)
    parsed: List[str] = []
    for part in parts:
        cand = part.strip().strip('"').strip("'")
        cand = re.sub(r"(?i)^assistant\\s+final[:\\s]+", "", cand)
        if not cand:
            continue
        if cand.upper() == "STOP":
            break
        # Must be letters/spaces/hyphens only.
        if re.search(r"[^A-Za-z\\s\\-]", cand):
            continue
        words = cand.split()
        if not (1 <= len(words) <= 4):
            continue
        if any(len(w) < 2 for w in words):
            continue
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        parsed.append(cand)
        if limit and len(parsed) >= limit:
            break
    return parsed


def _generate_prompt_text(
    prompt: str,
    *,
    max_new_tokens: int,
    generate_text_fn: Callable[[str, int], str],
) -> str:
    """
    Text-only helper for prompt brainstorming/critique.
    Uses Qwen (text-only) and returns empty string on failure.
    """
    try:
        helper_prompt = (
            "You generate short noun-phrase candidates for open-vocabulary detection. "
            "Respond with a comma-separated list only (no prose). "
            "Each candidate: 1-3 words, letters/spaces/hyphens only, no numbers, no quotes, no JSON. "
            "If no valid candidates, return an empty list.\n\n"
            f"Task: {prompt}"
        )
        text = generate_text_fn(helper_prompt, max_new_tokens)
        return text.strip()
    except Exception:
        pass
    return ""


def _caption_glossary_map(labelmap_glossary: Optional[str], labels: Sequence[str]) -> Dict[str, List[str]]:
    if not labelmap_glossary:
        return {}
    return _parse_glossary_mapping(_normalize_labelmap_glossary(labelmap_glossary), list(labels))


def _humanize_class_name_impl(name: str) -> str:
    return re.sub(r"[\\-_]+", " ", name).strip()


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


def _generate_prompt_variants_for_class_impl(
    class_name: str,
    max_synonyms: int,
    use_qwen: bool,
    *,
    generate_prompt_text_fn: Callable[[str, int], str],
    parse_prompt_candidates_fn: Callable[[str, set[str], int], List[str]],
    sanitize_prompts_fn: Callable[[List[str]], List[str]],
    humanize_class_name_fn: Callable[[str], str],
    log_fn: Optional[Callable[[str], None]] = None,
) -> List[str]:
    base: List[str] = []
    cleaned = class_name.strip()
    if cleaned:
        base.append(cleaned)
    human = humanize_class_name_fn(cleaned)
    if human and human.lower() != cleaned.lower():
        base.append(human)
    base_lower = {b.lower() for b in base if b}
    base_words = []
    for entry in base_lower:
        base_words.extend(re.split(r"[\\s_\\-]+", entry))
    base_words = [w for w in base_words if w]
    variants: List[str] = []
    if use_qwen and max_synonyms > 0:
        try:
            text = generate_prompt_text_fn(
                (
                    f"Generate up to {max_synonyms} alternative, common English labels for the object class "
                    f"'{human or cleaned}'. Each label must be 1-3 full words, each word at least 3 letters. "
                    "No abbreviations, no partial/truncated words, no numbering, no JSON. Avoid repeating the original name. "
                    "Use labels typical of object-detection datasets (e.g., car -> car, automobile, sedan; "
                    "person -> person, human, individual; utility pole -> utility pole, telephone pole, power pole). "
                    "Return a single comma-separated list."
                ),
                96,
            )
            raw_parts = re.split(r"[\\n;,]+", text)
            for part in raw_parts:
                normalized = part.strip().strip('"').strip("'")
                if not normalized:
                    continue
                if any(ch in normalized for ch in "{}[]:\""):
                    continue
                alpha_chars = re.sub(r"[^A-Za-z]", "", normalized)
                if len(alpha_chars) < 3:
                    continue
                if len(normalized) > 40:
                    continue
                if not re.search(r"[A-Za-z]", normalized):
                    continue
                words = normalized.split()
                if len(words) > 4:
                    continue
                if any(len(w) < 3 for w in words):
                    continue
                lowered = normalized.lower()
                fragment_of_base = any(
                    (lowered != b and lowered.startswith(b[: max(1, len(b) - 2)]))
                    or (b.startswith(lowered) and (len(b) - len(lowered) <= 2))
                    for b in base_lower
                )
                if fragment_of_base:
                    continue
                truncated_token = False
                for w in words:
                    lw = w.lower()
                    for bw in base_words:
                        if not bw:
                            continue
                        if bw.startswith(lw) and len(bw) - len(lw) >= 2:
                            truncated_token = True
                            break
                        if lw.startswith(bw) and len(lw) - len(bw) >= 3:
                            truncated_token = True
                            break
                    if truncated_token:
                        break
                if truncated_token:
                    continue
                variants.append(normalized)
        except Exception as exc:
            if log_fn:
                try:
                    log_fn(f"Prompt helper: Qwen generation failed for {class_name}: {exc}")
                except Exception:
                    pass
            variants = []
    seen = set()
    ordered: List[str] = []
    for item in [*base, *variants]:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    if max_synonyms <= 0:
        return ordered[:1]
    head = ordered[:1]
    tail = ordered[1 : 1 + max_synonyms]
    return head + tail


def _expand_prompts_with_prompt_llm_impl(
    class_name: str,
    base_prompts: List[str],
    max_new: int,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
    max_new_tokens: int = 128,
    generate_prompt_text_fn: Callable[[str, int], str],
    parse_prompt_candidates_fn: Callable[[str, set[str], int], List[str]],
    sanitize_prompts_fn: Callable[[List[str]], List[str]],
    humanize_class_name_fn: Callable[[str], str],
) -> List[str]:
    """Use Qwen text-only generation to brainstorm additional prompt variants for a class."""
    cleaned_base = sanitize_prompts_fn(base_prompts)
    if max_new <= 0 or not cleaned_base:
        return []

    seen = {p.lower() for p in cleaned_base}
    suggestions: List[str] = []
    try:
        known_list_str = ", ".join(cleaned_base)

        def _log(msg: str) -> None:
            if log_fn:
                try:
                    log_fn(msg)
                except Exception:
                    pass

        def _run_brainstorm_with_retries(remaining: int, round_idx: int) -> List[str]:
            """Try up to 3 times (initial + 2 critiques) to get a clean list."""
            base_prompt = [
                "Generate diverse noun-phrase prompts for open-vocabulary object detection with SAM3.",
                f"Target class: '{humanize_class_name_fn(class_name)}'.",
                f"Known good prompts: {known_list_str}.",
            ]
            base_prompt.extend(
                [
                    f"Propose up to {remaining} NEW, concrete object names (1-3 words) that strictly describe this class.",
                    "Rules: letters/spaces/hyphens only; no numbers; no punctuation beyond commas between items; no adjectives alone; avoid repeats.",
                    "Return ONLY a comma-separated list. Example: thing one, thing two, thing three",
                ]
            )
            last_text = ""
            for attempt in range(3):
                prompt_lines = list(base_prompt)
                if attempt > 0:
                    prompt_lines.extend(
                        [
                            f"Previous output was invalid: {last_text or '(empty)'}",
                            f"Try again. Respond ONLY with up to {remaining} comma-separated noun phrases (1-3 words, letters/spaces/hyphens).",
                            "No commentary.",
                        ]
                    )
                prompt_text = "\n".join(prompt_lines)
                text = generate_prompt_text_fn(prompt_text, max_new_tokens)
                last_text = text
                if not text:
                    _log(f"Qwen prompt expansion (class={class_name}, round {round_idx + 1}, attempt {attempt + 1}) returned empty/invalid")
                    continue
                parsed = parse_prompt_candidates_fn(text, seen, remaining)
                if parsed and len(parsed) > remaining:
                    parsed = parsed[:remaining]
                _log(
                    f"Qwen prompt expansion (class={class_name}, round {round_idx + 1}, attempt {attempt + 1}"
                    "): "
                    f"{', '.join(parsed) if parsed else text}"
                )
                if parsed:
                    return parsed
                dup_check = parse_prompt_candidates_fn(text, set(), remaining)
                if dup_check:
                    _log(
                        f"Qwen prompt expansion (class={class_name}, round {round_idx + 1}, attempt {attempt + 1}) "
                        "produced only duplicates; keeping existing list."
                    )
                    return []
                _log(f"Qwen prompt expansion (class={class_name}, round {round_idx + 1}, attempt {attempt + 1}) yielded no valid candidates")
            _log(f"Qwen prompt expansion (class={class_name}, round {round_idx + 1}) failed after 3 attempts")
            return []

        for round_idx in range(3):
            if len(suggestions) >= max_new:
                break
            remaining = max_new - len(suggestions)
            parsed = _run_brainstorm_with_retries(remaining, round_idx)
            if not parsed:
                continue
            suggestions.extend(parsed)
            if len(suggestions) >= max_new:
                break
    except Exception as exc:
        if log_fn:
            try:
                log_fn(f"Prompt helper: Qwen expansion failed for {class_name}: {exc}")
            except Exception:
                pass
        suggestions = []
    reviewed = sanitize_prompts_fn([*cleaned_base, *suggestions])
    reviewed_lower = {p.lower() for p in cleaned_base}
    final_new: List[str] = []
    for p in reviewed:
        low = p.lower()
        if low in reviewed_lower:
            continue
        reviewed_lower.add(low)
        final_new.append(p)
        if len(final_new) >= max_new:
            break
    if not final_new:
        if log_fn:
            try:
                log_fn(f"Qwen prompts fell back to base for {class_name}")
            except Exception:
                pass
        return cleaned_base
    return final_new


def _refine_prompts_with_qwen_impl(
    prompts: List[str],
    *,
    generate_prompt_text_fn: Callable[[str, int], str],
    sanitize_prompts_fn: Callable[[List[str]], List[str]],
) -> List[str]:
    prompts = sanitize_prompts_fn(prompts)
    if not prompts:
        return prompts
    try:
        prompt_text = (
            "You are validating candidate noun phrases for open-vocabulary object detection. "
            "Keep only entries that are concrete object-like noun phrases (1-4 words, nouns included). "
            "Reject fragments, verbs, partial words, or unrelated terms. "
            "Respond ONLY as a comma-separated list, no numbering, no explanations, ending with STOP.\n"
            f"Candidates: {', '.join(prompts)}"
        )
        text = generate_prompt_text_fn(prompt_text, 160)
        if not text:
            return prompts
        parts = [t.strip() for t in re.split(r"[,\\n]+", text) if t.strip() and t.strip().upper() != "STOP"]
        cleaned = sanitize_prompts_fn(parts)
        return cleaned or prompts
    except Exception:
        return prompts


def _qwen_self_filter_prompts_impl(
    class_name: str,
    prompts: List[str],
    *,
    generate_prompt_text_fn: Callable[[str, int], str],
    sanitize_prompts_fn: Callable[[List[str]], List[str]],
    humanize_class_name_fn: Callable[[str], str],
) -> List[str]:
    """Ask Qwen to self-critique the candidate prompts against the target class and return only credible entries."""
    prompts = sanitize_prompts_fn(prompts)
    if not prompts:
        return prompts
    try:
        prompt_text = (
            "You are double-checking candidate noun phrases for object detection. "
            f"Target class: '{humanize_class_name_fn(class_name)}'. "
            "From the list, keep ONLY phrases that clearly describe that class (synonyms or sub-types). "
            "Drop anything ambiguous, misspelled, or unrelated. "
            "Return ONLY a comma-separated list, no explanations, ending with STOP.\n"
            f"Candidates: {', '.join(prompts)}"
        )
        text = generate_prompt_text_fn(prompt_text, 160)
        if not text:
            return prompts
        parts = [t.strip() for t in re.split(r"[,\\n]+", text) if t.strip() and t.strip().upper() != "STOP"]
        cleaned = sanitize_prompts_fn(parts)
        return cleaned or prompts
    except Exception:
        return prompts


def _caption_preferred_label(label: str, glossary_map: Optional[Dict[str, List[str]]] = None) -> str:
    label = str(label or "").strip()
    if not label:
        return ""
    terms = (glossary_map or {}).get(label) or []
    for term in terms:
        if term and "_" not in term:
            return str(term)
    if "_" in label:
        return label.replace("_", " ")
    return label


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
) -> Tuple[str, Dict[str, int], int, bool]:
    safe_width = max(1, int(image_width))
    safe_height = max(1, int(image_height))
    counts: Dict[str, int] = dict(Counter([getattr(hint, "label", "") for hint in label_hints if getattr(hint, "label", "")]))

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
    for hint in label_hints:
        bbox = getattr(hint, "bbox", None) or []
        label = getattr(hint, "label", None)
        confidence = getattr(hint, "confidence", None)
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            x1 = max(0.0, min(float(x1), safe_width))
            y1 = max(0.0, min(float(y1), safe_height))
            x2 = max(0.0, min(float(x2), safe_width))
            y2 = max(0.0, min(float(y2), safe_height))
            if x2 <= x1 or y2 <= y1:
                continue
        else:
            x1 = y1 = x2 = y2 = None
        hints_payload.append(
            {
                "label": label,
                "bbox": [x1, y1, x2, y2] if x1 is not None else None,
                "bbox_2d": _bbox_to_qwen_2d([x1, y1, x2, y2]) if x1 is not None else None,
                "confidence": confidence if confidence is not None else None,
                "area": (x2 - x1) * (y2 - y1) if x1 is not None else 0.0,
            }
        )
    if max_boxes <= 0:
        selected = sorted(
            hints_payload,
            key=lambda entry: (
                -(entry["confidence"] if entry["confidence"] is not None else 0.0),
                -entry["area"],
            ),
        )
        truncated = False
    else:
        sorted_hints = sorted(
            hints_payload,
            key=lambda entry: (
                -(entry["confidence"] if entry["confidence"] is not None else 0.0),
                -entry["area"],
            ),
        )
        selected = sorted_hints[:max_boxes]
        truncated = len(sorted_hints) > len(selected)
    lines: List[str] = []
    if user_prompt:
        lines.append(f"User hint: {user_prompt}")
        if "style inspirations" in user_prompt.lower():
            lines.append(
                "Style guidance: use inspirations for tone/angle only. Rephrase, do not copy wording."
            )
    lines.append(f"Image size: {safe_width}x{safe_height} pixels.")
    glossary_map = _caption_glossary_map(
        labelmap_glossary,
        list(counts.keys()) or [entry["label"] for entry in hints_payload if entry.get("label")],
    )

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
            lines.append(f"COUNTS (use exactly): {counts_text}.")
            lines.append(
                "State these counts without qualifiers (avoid words like 'visible', 'roughly', or 'approximately')."
            )
        else:
            lines.append(f"COUNTS (use as hints; may be incomplete): {counts_text}.")
    elif counts:
        lines.append("Use the label hints to mention the main objects you see.")
    if counts and restrict_to_labels:
        allowed = ", ".join(sorted(_caption_preferred_label(lbl, glossary_map) for lbl in counts.keys()))
        if allowed:
            lines.append(
                f"Only mention these classes if they appear: {allowed}. Do not invent other entity types."
            )
    elif counts and not restrict_to_labels:
        lines.append("Label hints are suggestions; you may mention other visible objects too.")
    if selected:
        if include_coords:
            lines.append(
                "Labeled boxes (bbox_2d=[x1,y1,x2,y2], coords 0–1000 relative to this image/window):"
            )
            compact = [
                {"label": entry["label"], "bbox_2d": entry["bbox_2d"]}
                for entry in selected
                if entry["bbox_2d"] is not None
            ]
            if compact:
                lines.append(json.dumps(compact, separators=(",", ":")))
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
    if truncated:
        lines.append("Note: Only a subset of boxes is shown; counts reflect all hints.")
    lines.append(
        "Write a detailed caption. Use the image as truth and incorporate the label hints; "
        "if hints conflict with the image, mention the uncertainty briefly."
    )
    lines.append("Describe what the main objects are doing or how they are arranged when it is visible.")
    lines.append(
        "Be maximally descriptive: longer captions are acceptable when there is a lot to see. "
        "The labeled boxes are especially important and should be mentioned explicitly unless counts are overwhelming "
        "(e.g., summarize many cars as a parking lot)."
    )
    return "\n".join(lines), counts, len(selected), truncated


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _extract_caption_from_text(text: str, marker: Optional[str] = None) -> Tuple[str, bool]:
    cleaned = text.strip()
    marker_found = False
    if marker:
        match = re.search(rf"{marker}\\s*:?\\s*(.+)", cleaned, re.IGNORECASE | re.DOTALL)
        if match:
            cleaned = match.group(1)
            marker_found = True
    if not marker_found:
        match = re.search(r"FINAL\\s*:?\\s*(.+)", cleaned, re.IGNORECASE | re.DOTALL)
        if match:
            cleaned = match.group(1)
            marker_found = True
    cleaned = _collapse_whitespace(cleaned) if cleaned else text.strip()
    return cleaned, marker_found


def _caption_needs_english_rewrite(text: str) -> bool:
    return bool(re.search(r"[^\x00-\x7F]", text))


_CAPTION_GENERIC_OPENERS = (
    "an aerial view",
    "aerial view",
    "from a high angle",
    "a drone image",
    "a bird's-eye view",
    "overhead view",
)


def _caption_starts_generic(text: str) -> bool:
    lowered = text.strip().lower()
    return any(lowered.startswith(prefix) for prefix in _CAPTION_GENERIC_OPENERS)


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
        label_terms = [str(label)]
        if "_" in label:
            label_terms.append(label.replace("_", " "))
        if glossary_map and glossary_map.get(label):
            label_terms.extend(glossary_map[label])
        label_terms = [term.strip() for term in label_terms if term and term.strip()]
        if not any(term.lower() in lowered for term in label_terms):
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
    if missing:
        return True, missing
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
    if re.search(r"\bFINAL\b", cleaned, flags=re.IGNORECASE):
        cleaned, _ = _extract_caption_from_text(cleaned, marker="FINAL")
    cleaned = cleaned.strip()
    if cleaned.startswith(":"):
        cleaned = cleaned.lstrip(":").strip()
    return cleaned


_QWEN_THINKING_REASONING_RE = re.compile(
    r"(?:\bgot it\b|\blet'?s\b|\bfirst\b|\bsecond\b|\bthird\b|\bstep\b|\bi need\b|\bnow\b|\bthe task\b)",
    re.IGNORECASE,
)
_QWEN_CAPTION_META_RE = re.compile(
    r"(authoritative|as indicated|label hint|bounding box|bbox|coordinates|hinted|counts are provided)",
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
    return trimmed[-1] not in ".!?"


def _caption_has_meta(caption: str) -> bool:
    if not caption:
        return False
    return _QWEN_CAPTION_META_RE.search(caption) is not None


def _caption_needs_short_form(caption: str, max_words: int = 80, max_sentences: int = 2) -> bool:
    if not caption:
        return False
    words = caption.split()
    if len(words) > max_words:
        return True
    sentences = [s.strip() for s in re.split(r"[.!?]+", caption) if s.strip()]
    return len(sentences) > max_sentences


def _allowed_caption_labels_impl(label_hints: Sequence[Dict[str, Any]]) -> List[str]:
    labels = []
    for entry in label_hints or []:
        try:
            label = str(entry.get("label") or "").strip()
        except Exception:
            label = ""
        if not label:
            continue
        labels.append(label)
    return sorted(set(labels))


def _caption_is_degenerate_impl(caption: str) -> bool:
    if not caption:
        return True
    trimmed = caption.strip()
    if not trimmed:
        return True
    if _QWEN_THINKING_REASONING_RE.search(trimmed):
        return True
    compact = re.sub(r"\\s+", "", trimmed)
    if compact:
        alnum = re.findall(r"[A-Za-z0-9]", compact)
        if not alnum and len(compact) > 10:
            return True
        if len(compact) >= 20:
            ratio = len(alnum) / max(1, len(compact))
            if ratio < 0.2:
                return True
        if re.fullmatch(r"[^A-Za-z0-9]+", compact):
            return True
        if re.search(r"([!?.<>\\-_=])\\1{20,}", compact):
            return True
    words = caption.split()
    if len(words) < 8:
        return True
    sentences = [s.strip().lower() for s in re.split(r"[.!?]+", caption) if s.strip()]
    if sentences:
        counts = Counter(sentences)
        most_common = counts.most_common(1)[0][1]
        if most_common >= 3:
            return True
        if most_common / max(1, len(sentences)) > 0.45:
            return True
    if len(words) > 40:
        tokens = [w.lower() for w in words]
        bigrams = list(zip(tokens, tokens[1:]))
        if bigrams:
            unique_ratio = len(set(bigrams)) / len(bigrams)
            if unique_ratio < 0.55:
                return True
    return False


def _resolve_qwen_caption_decode(payload: Any, is_thinking: bool) -> Dict[str, Any]:
    use_sampling = payload.use_sampling if getattr(payload, "use_sampling", None) is not None else True
    if not use_sampling:
        return {"do_sample": False}
    defaults = {
        "temperature": 1.0 if is_thinking else 0.7,
        "top_p": 0.95 if is_thinking else 0.8,
        "top_k": 20,
        "presence_penalty": 0.0 if is_thinking else 1.5,
    }
    temperature = payload.temperature if getattr(payload, "temperature", None) is not None else defaults["temperature"]
    top_p = payload.top_p if getattr(payload, "top_p", None) is not None else defaults["top_p"]
    top_k = payload.top_k if getattr(payload, "top_k", None) is not None else defaults["top_k"]
    presence_penalty = (
        payload.presence_penalty if getattr(payload, "presence_penalty", None) is not None else defaults["presence_penalty"]
    )
    return {
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
    }


def _adjust_prompt_for_thinking(prompt_text: str) -> str:
    if not prompt_text:
        return prompt_text
    lines = prompt_text.splitlines()
    filtered = [line for line in lines if not line.startswith("Write a concise caption")]
    return "\n".join(filtered)


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
    run_qwen_inference_fn: Callable[..., Tuple[str, Any, Any]],
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
    strict_note = "Return exactly one complete sentence. " if strict else ""
    minimal_note = (
        "Edit the draft with minimal changes. Do not introduce new objects or actions. "
        if minimal_edit
        else ""
    )
    cleanup_system = (
        "You are a captioning assistant. Respond in English only. "
        "Return only <final>...</final> and nothing else."
    )
    cleanup_prompt = (
        f"{strict_note}{allowed_note}{minimal_note}"
        "Remove repetition, avoid coordinates, and remove any mention of labels, hints, or counts being provided. "
        "Keep the caption grounded in the image.\n"
        f"Draft caption: {prompt}"
    )
    cleanup_model = model_id_override or resolve_variant_fn(base_model_id, "Instruct")
    qwen_text, _, _ = run_qwen_inference_fn(
        cleanup_prompt,
        pil_img,
        max_new_tokens=max_new_tokens,
        system_prompt_override=cleanup_system,
        model_id_override=cleanup_model if use_caption_cache and runtime_override is None else None,
        runtime_override=runtime_override,
        decode_override={"do_sample": False},
    )
    caption_text, _ = extract_caption_fn(qwen_text, marker=None)
    return sanitize_caption_fn(caption_text)


def _run_qwen_caption_merge(
    draft_caption: str,
    windowed_captions: Sequence[Tuple[int, int, int, str]],
    *,
    pil_img: Any,
    base_model_id: str,
    runtime_resolver: Callable[[str], Tuple[Any, Any]],
    max_new_tokens: int,
    glossary_line: Optional[str] = None,
    run_qwen_inference_fn: Callable[..., Tuple[str, Any, Any]],
    resolve_variant_fn: Callable[[str, str], str],
    extract_caption_fn: Callable[[str, Optional[str]], Tuple[str, bool]],
    sanitize_caption_fn: Callable[[str], str],
) -> str:
    if not draft_caption or not windowed_captions:
        return draft_caption
    window_lines = ["Window observations (do NOT invent objects):"]
    for x0, y0, size, caption in windowed_captions:
        if caption:
            window_lines.append(f"- {caption}")
    if len(window_lines) == 1:
        return draft_caption
    merge_prompt = (
        "Revise the draft caption so it includes all distinct object details "
        "from the window observations that are missing in the draft. "
        "Do not invent new objects. Use multiple sentences if needed. "
        "Preserve specific counts, actions, and notable attributes from the windows; "
        "do not drop any concrete window detail unless it clearly conflicts with the full image. "
        "Do not mention labels, hints, or coordinates.\n"
        f"Draft caption: {draft_caption}\n"
        + "\n".join(window_lines)
    )
    if glossary_line:
        merge_prompt = f"{merge_prompt}\n{glossary_line}"
    merge_system = (
        "You are a caption editor. Return only the revised caption in English."
    )
    merge_model = resolve_variant_fn(base_model_id, "Instruct")
    qwen_text, _, _ = run_qwen_inference_fn(
        merge_prompt,
        pil_img,
        max_new_tokens=max_new_tokens,
        system_prompt_override=merge_system,
        runtime_override=runtime_resolver(merge_model),
        decode_override={"do_sample": False},
    )
    merged, _ = extract_caption_fn(qwen_text, marker=None)
    merged = sanitize_caption_fn(merged)
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
    if requested is None:
        overlap_val = _resolve_qwen_window_overlap(overlap, default_overlap=default_overlap)
        base_dim = max(1, min(int(image_width), int(image_height)))
        # 2x2 grid with overlap -> window = dim / (2 - overlap)
        base = base_dim / max(1.0, 2.0 - overlap_val)
    else:
        base = requested
    try:
        base = int(base)
    except (TypeError, ValueError):
        base = default_size
    base = max(128, min(base, 4096))
    return max(64, min(base, int(image_width), int(image_height)))


def _resolve_qwen_window_overlap(requested: Optional[float], *, default_overlap: float = 0.2) -> float:
    try:
        overlap = float(requested) if requested is not None else default_overlap
    except (TypeError, ValueError):
        overlap = default_overlap
    return max(0.0, min(overlap, 0.2))
