#!/usr/bin/env python3
"""Audit Class Split Qwen review benchmark runs.

The review agent is intentionally conservative and controller-gated. This script
turns the ad hoc safety checks used during development into a repeatable report
so future prompt, model, or evidence changes can be compared against the same
invariants.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CLASS_CHANGE_DECISIONS = {"accept_suggested", "change_to_other"}
ACTIONABLE_DECISIONS = CLASS_CHANGE_DECISIONS | {"confirm_current"}
BAD_CLASS_CHANGE_OVERLAPS = {
    "duplicate_like",
    "target_contains_other",
    "other_contains_target",
    "unclear",
}


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_sanitize(item) for item in value)
    return value


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _records(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = data.get("records")
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def _text_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(record.get("model_compact_arguments") or {})
    payload.setdefault("rationale_short", record.get("rationale_short") or "")
    payload.setdefault("counter_evidence", record.get("counter_evidence") or "")
    return payload


def _text_parts(payload: Dict[str, Any]) -> List[str]:
    return [
        str(payload.get(key) or "").strip()
        for key in ("rationale_short", "counter_evidence")
        if str(payload.get(key) or "").strip()
    ]


def _text_rebuts_overlap_contamination(payload: Dict[str, Any], *, target_class: str = "") -> bool:
    text_lower = " ".join(_text_parts(payload)).lower()
    if not text_lower:
        return False
    patterns: List[str] = [
        r"\boverlap(?:ping)?\b[^.?!]{0,180}\b(?:does\s+not|doesn't|do\s+not|not)\b[^.?!]{0,180}\b(?:explain|account\s+for)\b[^.?!]{0,180}\b(?:target|object|own|features?)\b",
        r"\b(?:target|object)\b[^.?!]{0,180}\b(?:own|intrinsic|visible)\b[^.?!]{0,180}\bfeatures?\b[^.?!]{0,180}\b(?:not|do\s+not|does\s+not|doesn't)\b[^.?!]{0,180}\b(?:explained|caused|accounted\s+for)\b[^.?!]{0,180}\boverlap\b",
        r"\boverlap\b[^.?!]{0,80}\b(?:minor|minimal|small|low|weak|slight)\b",
        r"\b(?:minor|minimal|small|low|weak|slight)\b[^.?!]{0,80}\boverlap\b",
        r"\boverlapping\b[^.?!]{0,180}\b(?:adjacent|nearby|separate)\b[^.?!]{0,120}\b(?:not|not\s+the|not\s+part\s+of)\b[^.?!]{0,80}\b(?:target|object|itself)\b",
        r"\boverlapping\b[^.?!]{0,180}\b(?:are|is)\b[^.?!]{0,80}\b(?:adjacent|nearby|separate)\b[^.?!]{0,120}\b(?:not|rather\s+than)\b[^.?!]{0,80}\b(?:target|object|itself)\b",
        r"\boverlap\b[^.?!]{0,120}\bbackground\b[^.?!]{0,120}\bnot\b[^.?!]{0,80}\b(?:target|object|primary|main)\b",
        r"\boverlap\b[^.?!]{0,180}\b(?:minor|background)\b[^.?!]{0,80}\b(?:element|object|structure)\b",
        r"\b(?:background|minor)\b[^.?!]{0,80}\b(?:element|object|structure)\b[^.?!]{0,180}\bnot\b[^.?!]{0,80}\b(?:target|object|primary|main)\b",
    ]
    alias_patterns = [
        re.escape(alias).replace(r"\ ", r"\s+")
        for alias in _label_aliases(target_class)
        if str(alias or "").strip()
    ]
    if alias_patterns:
        alias_expr = "|".join(alias_patterns[:8])
        patterns.append(
            rf"\boverlap\b[^.?!]{{0,120}}\bbackground\b[^.?!]{{0,120}}\bnot\b[^.?!]{{0,80}}\b(?:{alias_expr})s?\b"
        )
    return any(re.search(pattern, text_lower) for pattern in patterns)


def _normalize_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _label_aliases(value: Any) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text).replace("_", " ").replace("-", " ").lower()
    aliases = {text.lower(), spaced}
    generic_tokens = {"class", "label", "object", "target", "small", "large", "light", "heavy", "other", "misc"}
    for token in re.findall(r"[a-z0-9]{4,}", spaced):
        if token not in generic_tokens:
            aliases.add(token)
    return sorted((alias for alias in aliases if len(alias) >= 3), key=len, reverse=True)


def _record_text(record: Dict[str, Any]) -> str:
    payload = record.get("model_compact_arguments") if isinstance(record.get("model_compact_arguments"), dict) else {}
    parts: List[str] = []
    for source in (record, payload):
        for key in ("rationale_short", "counter_evidence", "reason", "rationale"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value)
        for key in ("visible_target_cues", "target_observations", "visible_cues"):
            value = source.get(key)
            if isinstance(value, (list, tuple)):
                parts.extend(str(item) for item in value if str(item or "").strip())
            elif isinstance(value, str) and value.strip():
                parts.append(value)
    # Keep separate model fields as separate sentences. Joining raw fields with a
    # plain space can synthesize false negations such as "no overlap contamination
    # flat roof ... building features".
    return ". ".join(parts).strip().lower()


def _record_rejects_target_alias(record: Dict[str, Any]) -> Optional[str]:
    text = _record_text(record)
    if not text:
        return None
    def _context_only_negation(match: re.Match[str]) -> bool:
        sentence_start = max(text.rfind(".", 0, match.start()), text.rfind("?", 0, match.start()), text.rfind("!", 0, match.start()), text.rfind(";", 0, match.start()))
        sentence = text[sentence_start + 1 : match.end() + 80]
        prefix = sentence[: max(0, match.start() - sentence_start - 1)]
        has_target_marker = bool(re.search(r"\b(?:target|object|crop|target\s+crop)\b", prefix[-140:]))
        has_context_marker = bool(re.search(r"\b(?:background|overlap|overlapping|nearby|adjacent|context|road|markings)\b", prefix[-140:]))
        return bool(has_context_marker and not has_target_marker)

    for alias in _label_aliases(record.get("target_class")):
        alias_expr = re.escape(alias)
        patterns = (
            rf"\b(?:target|object|crop|target\s+crop)\b[^.?!]{{0,160}}\bnot\s+(?:a\s+|an\s+|the\s+)?{alias_expr}s?\b",
            rf"\bnot\s+(?:a\s+|an\s+|the\s+)?{alias_expr}s?\b",
            rf"\bno\b[^.?!]{{0,80}}\b{alias_expr}s?\b[^.?!]{{0,80}}\b(?:features|cues|evidence|support)\b",
            rf"\b{alias_expr}s?\b[^.?!]{{0,80}}\b(?:not\s+visible|not\s+present|absent|missing)\b",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                if _context_only_negation(match):
                    continue
                return alias
    return None


def _record_best_class_material_overlap(record: Dict[str, Any], class_name: Any) -> Optional[Dict[str, Any]]:
    class_norm = _normalize_label(class_name)
    if not class_norm:
        return None
    best: Optional[Dict[str, Any]] = None
    for evidence in record.get("evidence") or []:
        if not isinstance(evidence, dict) or evidence.get("kind") != "overlap_decomposition":
            continue
        metadata = evidence.get("metadata") if isinstance(evidence.get("metadata"), dict) else {}
        for item in metadata.get("overlaps") or []:
            if not isinstance(item, dict):
                continue
            if _normalize_label(item.get("class_name")) != class_norm:
                continue
            relation = str(item.get("relation") or "").strip()
            if relation in {"", "none"}:
                continue
            try:
                target_cover = float(item.get("target_area_covered") or 0.0)
                other_cover = float(item.get("other_area_covered") or 0.0)
                iou = float(item.get("iou") or 0.0)
            except Exception:
                continue
            if max(target_cover, other_cover, iou) <= 0.0:
                continue
            candidate = dict(item)
            candidate["_material_score"] = max(target_cover, other_cover, iou)
            if best is None or float(candidate["_material_score"]) > float(best.get("_material_score") or 0.0):
                best = candidate
    return best


def _normalize_visible_cues(
    value: Any,
    *,
    current_class: str = "",
    suggested_class: str = "",
    target_class: str = "",
) -> List[str]:
    """Normalize explicit visible-cue ledgers without dataset-specific terms."""

    if isinstance(value, (list, tuple)):
        raw_items = list(value)
    elif value is None:
        raw_items = []
    else:
        raw_text = str(value or "").strip()
        raw_items = re.split(r"[;\n|]+", raw_text) if raw_text else []

    class_terms = {
        _normalize_label(item)
        for item in (current_class, suggested_class, target_class)
        if str(item or "").strip()
    }
    generic_terms = {
        _normalize_label(item)
        for item in (
            "matches class",
            "matches target class",
            "matches suggested class",
            "fits class",
            "fits target class",
            "fits suggested class",
            "target is clear",
            "clear target",
            "visible target",
            "visible object",
            "object visible",
            "same as anchors",
            "looks like class",
            "class specific cues",
            "class-specific cues",
        )
    }
    boilerplate_tokens = (
        "target",
        "object",
        "class",
        "specific",
        "cue",
        "cues",
        "visible",
        "clear",
        "matches",
        "match",
        "fits",
        "fit",
        "suggested",
        "current",
    )
    context_only_tokens = {
        "aerial",
        "adjacent",
        "background",
        "beside",
        "clear",
        "crop",
        "down",
        "ground",
        "image",
        "inside",
        "located",
        "location",
        "near",
        "nearby",
        "on",
        "outside",
        "overhead",
        "parked",
        "partial",
        "pavement",
        "perspective",
        "road",
        "scene",
        "shadow",
        "shadows",
        "sitting",
        "standing",
        "top",
        "view",
        "visible",
        "water",
    }
    connector_tokens = {"a", "an", "and", "at", "by", "from", "in", "of", "the", "to", "with"}

    def _context_only_cue(text_value: str) -> bool:
        lowered = str(text_value or "").strip().lower()
        if not lowered:
            return True
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", lowered)
            if token not in connector_tokens and token not in boilerplate_tokens
        ]
        if not tokens:
            return True
        if all(token in context_only_tokens for token in tokens):
            return True
        if (
            re.search(r"\b(?:top[-\s]?down|overhead|aerial)\b", lowered)
            and len([token for token in tokens if token not in context_only_tokens]) == 0
        ):
            return True
        return False

    cues: List[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        text = re.sub(r"\s+", " ", str(raw_item or "")).strip(" .,:;-")
        if not text:
            continue
        text = text[:140]
        norm = _normalize_label(text)
        if not norm or norm in seen:
            continue
        if norm in class_terms or norm in generic_terms:
            continue
        if not re.search(r"[a-zA-Z]", text):
            continue
        if _context_only_cue(text):
            continue
        stripped_norm = norm
        for class_term in class_terms:
            if class_term:
                stripped_norm = stripped_norm.replace(class_term, "")
        for token in boilerplate_tokens:
            stripped_norm = re.sub(rf"\b{re.escape(token)}\b", "", stripped_norm)
        if len(stripped_norm.strip()) < 3:
            continue
        cues.append(text)
        seen.add(norm)
        if len(cues) >= 6:
            break
    return cues


def _explicit_visible_cues_from_record(record: Dict[str, Any]) -> List[str]:
    payload = record.get("model_compact_arguments") if isinstance(record.get("model_compact_arguments"), dict) else {}
    raw_value = record.get("visible_target_cues")
    if raw_value is None:
        raw_value = payload.get("visible_target_cues")
    if raw_value is None:
        raw_value = payload.get("target_observations")
    if raw_value is None:
        raw_value = payload.get("visible_cues")
    return _normalize_visible_cues(
        raw_value,
        current_class=str(record.get("current_class") or ""),
        suggested_class=str(record.get("suggested_neighbor_class") or ""),
        target_class=str(record.get("target_class") or ""),
    )


def _record_has_clean_visual_ledger(record: Dict[str, Any]) -> bool:
    ledger = record.get("evidence_ledger") if isinstance(record.get("evidence_ledger"), dict) else {}
    clean_ids = [
        str(item or "").strip()
        for item in (ledger.get("clean_visual_evidence_ids") or [])
        if str(item or "").strip()
    ]
    reference_ids = [
        str(item or "").strip()
        for item in (ledger.get("clean_visual_reference_evidence_ids") or [])
        if str(item or "").strip()
    ]
    return bool(clean_ids) and bool(clean_ids or reference_ids)


def _clean_visual_ids_from_record(record: Dict[str, Any]) -> List[str]:
    ledger = record.get("evidence_ledger") if isinstance(record.get("evidence_ledger"), dict) else {}
    rows = [row for row in (ledger.get("rows") or []) if isinstance(row, dict)]
    clean_ids = [
        str(item or "").strip()
        for item in (ledger.get("clean_visual_evidence_ids") or [])
        if str(item or "").strip()
    ]
    if rows:
        row_clean_ids = [
            str(row.get("evidence_id") or "").strip()
            for row in rows
            if str(row.get("use") or "").strip() == "clean_visual"
            and str(row.get("evidence_id") or "").strip()
        ]
        if row_clean_ids:
            clean_ids = row_clean_ids
    seen = set()
    ordered = []
    for evidence_id in clean_ids:
        if evidence_id not in seen:
            ordered.append(evidence_id)
            seen.add(evidence_id)
    return ordered


def _target_source_clean_ids_from_record(record: Dict[str, Any]) -> List[str]:
    ledger = record.get("evidence_ledger") if isinstance(record.get("evidence_ledger"), dict) else {}
    rows = [row for row in (ledger.get("rows") or []) if isinstance(row, dict)]
    explicit_ids = [
        str(item or "").strip()
        for item in (ledger.get("clean_target_source_evidence_ids") or [])
        if str(item or "").strip()
    ]
    if explicit_ids or "clean_target_source_evidence_ids" in ledger:
        return explicit_ids
    if not rows:
        return _clean_visual_ids_from_record(record)
    clean_ids = [
        str(row.get("evidence_id") or "").strip()
        for row in rows
        if str(row.get("use") or "").strip() == "clean_visual"
        and str(row.get("kind") or "").strip() != "class_context_pack"
        and str(row.get("evidence_id") or "").strip()
    ]
    return clean_ids


def _supporting_clean_evidence_ids_from_record(record: Dict[str, Any]) -> List[str]:
    payload = record.get("model_compact_arguments") if isinstance(record.get("model_compact_arguments"), dict) else {}
    raw_value = record.get("supporting_clean_evidence_ids")
    if raw_value is None:
        raw_value = payload.get("supporting_clean_evidence_ids")
    if raw_value is None:
        raw_value = payload.get("visible_cue_evidence_ids")
    if raw_value is None:
        raw_value = payload.get("clean_evidence_ids")
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        raw_items: Iterable[Any] = re.split(r"[,;\s]+", raw_value)
    elif isinstance(raw_value, (list, tuple, set)):
        raw_items = raw_value
    else:
        raw_items = [raw_value]
    ids: List[str] = []
    seen = set()
    for raw_item in raw_items:
        evidence_id = str(raw_item or "").strip()
        if evidence_id and evidence_id not in seen:
            ids.append(evidence_id)
            seen.add(evidence_id)
    return ids


def _record_key(record: Dict[str, Any], fallback_index: int) -> str:
    point_id = str(record.get("point_id") or "").strip()
    if point_id:
        return f"point:{point_id}"
    review_id = str(record.get("review_id") or "").strip()
    if review_id:
        return f"review:{review_id}"
    return f"ordinal:{record.get('ordinal') or fallback_index}"


def _short_record(record: Dict[str, Any]) -> Dict[str, Any]:
    payload = record.get("model_compact_arguments") if isinstance(record.get("model_compact_arguments"), dict) else {}
    def _field(name: str) -> Any:
        value = record.get(name)
        return value if value not in (None, "") else payload.get(name)

    return {
        "ordinal": record.get("ordinal"),
        "point_id": record.get("point_id"),
        "decision": record.get("decision"),
        "current_class": record.get("current_class"),
        "suggested_neighbor_class": record.get("suggested_neighbor_class"),
        "target_class": record.get("target_class"),
        "confidence": record.get("confidence"),
        "backend_tier": record.get("backend_tier"),
        "visual_quality": record.get("visual_quality"),
        "object_visibility": record.get("object_visibility"),
        "overlap_assessment": _field("overlap_assessment"),
        "current_evidence": _field("current_evidence"),
        "suggested_evidence": _field("suggested_evidence"),
        "target_evidence": _field("target_evidence"),
        "same_image_scale_evidence": _field("same_image_scale_evidence"),
        "same_image_embedding_evidence": _field("same_image_embedding_evidence"),
        "same_image_scale_report_signal": record.get("same_image_scale_report_signal"),
        "same_image_embedding_report_signal": record.get("same_image_embedding_report_signal"),
        "visible_target_cues": _explicit_visible_cues_from_record(record),
        "supporting_clean_evidence_ids": _supporting_clean_evidence_ids_from_record(record),
        "rationale_short": record.get("rationale_short") or "",
        "guardrail_reasons": record.get("guardrail_reasons") or [],
        "advisory_reasons": record.get("advisory_reasons") or [],
        "guarded_recommendation": record.get("guarded_recommendation"),
    }


def _add_issue(issues: Dict[str, List[Dict[str, Any]]], name: str, record: Dict[str, Any], reason: str) -> None:
    item = _short_record(record)
    item["reason"] = reason
    issues[name].append(item)


def audit_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Return machine-checkable safety and usefulness diagnostics."""

    issues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    guarded_clear_target_candidates: List[Dict[str, Any]] = []
    guarded_recommendations: List[Dict[str, Any]] = []
    actionables: List[Dict[str, Any]] = []

    for record in records:
        decision = str(record.get("decision") or "").strip()
        backend_tier = str(record.get("backend_tier") or "").strip().lower()
        visual_quality = str(record.get("visual_quality") or "").strip().lower()
        object_visibility = str(record.get("object_visibility") or "").strip().lower()
        payload = record.get("model_compact_arguments") if isinstance(record.get("model_compact_arguments"), dict) else {}
        current_evidence = str(record.get("current_evidence") or payload.get("current_evidence") or "").strip().lower()
        suggested_evidence = str(record.get("suggested_evidence") or payload.get("suggested_evidence") or "").strip().lower()
        target_evidence = str(record.get("target_evidence") or payload.get("target_evidence") or "").strip().lower()
        anchor_evidence_suggested = str(
            record.get("anchor_evidence_suggested") or payload.get("anchor_evidence_suggested") or ""
        ).strip().lower()
        overlap_assessment = str(record.get("overlap_assessment") or payload.get("overlap_assessment") or "").strip().lower()
        guardrails = [str(item) for item in (record.get("guardrail_reasons") or [])]
        guarded = record.get("guarded_recommendation")
        if isinstance(guarded, dict) and guarded.get("blocked"):
            guarded_recommendations.append(_short_record(record))

        if decision in ACTIONABLE_DECISIONS:
            actionables.append(_short_record(record))
            if backend_tier != "clear" or visual_quality != "clear" or object_visibility != "clear":
                _add_issue(
                    issues,
                    "non_skip_low_quality",
                    record,
                    f"tier={backend_tier}, visual={visual_quality}, visibility={object_visibility}",
                )
            if guardrails:
                _add_issue(issues, "non_skip_with_guardrails", record, "; ".join(guardrails))

        if decision in CLASS_CHANGE_DECISIONS:
            visible_target_cues = _explicit_visible_cues_from_record(record)
            if len(visible_target_cues) < 2:
                _add_issue(
                    issues,
                    "class_change_missing_visible_target_cues",
                    record,
                    f"visible_target_cues={len(visible_target_cues)}",
                )
            if not _record_has_clean_visual_ledger(record):
                _add_issue(
                    issues,
                    "class_change_missing_clean_visual_ledger",
                    record,
                    "missing controller evidence ledger with clean visual evidence ids",
                )
            supporting_clean_ids = set(_supporting_clean_evidence_ids_from_record(record))
            clean_visual_ids = set(_clean_visual_ids_from_record(record))
            target_source_clean_ids = set(_target_source_clean_ids_from_record(record))
            if not supporting_clean_ids:
                _add_issue(
                    issues,
                    "class_change_missing_supporting_clean_evidence",
                    record,
                    "missing supporting_clean_evidence_ids for visible cues",
                )
            elif not target_source_clean_ids:
                _add_issue(
                    issues,
                    "class_change_missing_supporting_clean_evidence",
                    record,
                    "controller ledger has no clean target/source evidence ids",
                )
            elif not supporting_clean_ids.intersection(clean_visual_ids):
                _add_issue(
                    issues,
                    "class_change_missing_supporting_clean_evidence",
                    record,
                    "supporting_clean_evidence_ids do not reference clean visual evidence",
                )
            elif target_source_clean_ids and not supporting_clean_ids.intersection(target_source_clean_ids):
                _add_issue(
                    issues,
                    "class_change_missing_supporting_clean_evidence",
                    record,
                    "supporting_clean_evidence_ids cite only reference context, not target/source clean evidence",
                )
            if backend_tier != "clear":
                _add_issue(issues, "class_change_low_backend_quality", record, f"backend_tier={backend_tier}")
            if current_evidence == "strong":
                _add_issue(issues, "class_change_overrides_strong_current", record, "current_evidence=strong")
            if suggested_evidence != "strong" and decision == "accept_suggested":
                _add_issue(issues, "accept_without_strong_suggested", record, f"suggested_evidence={suggested_evidence}")
            if decision == "accept_suggested" and anchor_evidence_suggested != "strong":
                _add_issue(
                    issues,
                    "accept_without_strong_suggested_anchor",
                    record,
                    f"anchor_evidence_suggested={anchor_evidence_suggested or 'missing'}",
                )
            if overlap_assessment in BAD_CLASS_CHANGE_OVERLAPS:
                _add_issue(issues, "class_change_bad_overlap", record, f"overlap_assessment={overlap_assessment}")
            current_overlap = _record_best_class_material_overlap(record, record.get("current_class"))
            target_overlap = _record_best_class_material_overlap(record, record.get("target_class"))
            current_cover = float(current_overlap.get("target_area_covered") or 0.0) if current_overlap else 0.0
            target_cover = float(target_overlap.get("target_area_covered") or 0.0) if target_overlap else 0.0
            if current_overlap and current_cover >= 0.50 and target_cover <= min(0.25, current_cover * 0.50):
                _add_issue(
                    issues,
                    "class_change_dominant_current_overlap",
                    record,
                    (
                        "current class dominates the target bbox in overlap decomposition "
                        f"({current_overlap.get('relation')}, current_cover={current_cover:.2f}, "
                        f"target_class_cover={target_cover:.2f})"
                    ),
                )
            rejected_alias = _record_rejects_target_alias(record)
            if rejected_alias:
                _add_issue(
                    issues,
                    "class_change_text_rejects_target",
                    record,
                    f"model text rejects target-class cue `{rejected_alias}`",
                )
            if overlap_assessment == "partial_contamination" and not _text_rebuts_overlap_contamination(
                _text_payload(record),
                target_class=str(record.get("target_class") or ""),
            ):
                _add_issue(
                    issues,
                    "partial_overlap_without_explicit_rebuttal",
                    record,
                    "partial_contamination class change lacks textual rebuttal",
                )

        if decision == "confirm_current" and suggested_evidence == "strong":
            _add_issue(issues, "confirm_overrides_strong_suggested", record, "suggested_evidence=strong")

        if (
            decision == "skip_uncertain"
            and backend_tier == "clear"
            and visual_quality == "clear"
            and object_visibility == "clear"
            and target_evidence == "strong"
            and (current_evidence in {"weak", "none"} or suggested_evidence == "strong")
        ):
            guarded_clear_target_candidates.append(_short_record(record))

    issue_counts = {name: len(items) for name, items in sorted(issues.items())}
    return {
        "records": len(records),
        "decision_counts": dict(Counter(str(record.get("decision") or "missing") for record in records)),
        "backend_tier_counts": dict(Counter(str(record.get("backend_tier") or "unknown") for record in records)),
        "actionable_count": len(actionables),
        "issue_counts": issue_counts,
        "unsafe_issue_count": sum(issue_counts.values()),
        "issues": dict(sorted(issues.items())),
        "guarded_clear_target_candidate_count": len(guarded_clear_target_candidates),
        "guarded_clear_target_candidates": guarded_clear_target_candidates,
        "guarded_recommendation_count": len(guarded_recommendations),
        "guarded_recommendations": guarded_recommendations,
        "actionables": actionables,
    }


def compare_runs(current: Sequence[Dict[str, Any]], previous: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    previous_by_key = {_record_key(record, idx): record for idx, record in enumerate(previous, start=1)}
    changes: List[Dict[str, Any]] = []
    missing_previous = 0
    for idx, record in enumerate(current, start=1):
        key = _record_key(record, idx)
        prev = previous_by_key.get(key)
        if not prev:
            missing_previous += 1
            continue
        fields = ("decision", "target_class", "confidence")
        changed = {
            field: {"previous": prev.get(field), "current": record.get(field)}
            for field in fields
            if prev.get(field) != record.get(field)
        }
        if changed:
            changes.append(
                {
                    "key": key,
                    "ordinal": record.get("ordinal"),
                    "point_id": record.get("point_id"),
                    "current_class": record.get("current_class"),
                    "suggested_neighbor_class": record.get("suggested_neighbor_class"),
                    "changed": changed,
                    "current_rationale": record.get("rationale_short") or "",
                    "previous_rationale": prev.get("rationale_short") or "",
                }
            )
    return {
        "matched_current_records": len(current) - missing_previous,
        "missing_previous_records": missing_previous,
        "changed_count": len(changes),
        "changes": changes,
    }


def _print_section(title: str, rows: Iterable[Dict[str, Any]], *, limit: int) -> None:
    rows = list(rows)
    print(f"\n{title}: {len(rows)}")
    for row in rows[: max(0, limit)]:
        ordinal = row.get("ordinal")
        decision = row.get("decision")
        current = row.get("current_class")
        suggested = row.get("suggested_neighbor_class")
        target = row.get("target_class")
        guarded = row.get("guarded_recommendation") if isinstance(row.get("guarded_recommendation"), dict) else None
        guarded_text = ""
        if guarded and guarded.get("blocked"):
            guarded_text = (
                f" guarded={guarded.get('decision')} target={guarded.get('target_class')}"
                f" conf={guarded.get('confidence')}"
            )
        reason = row.get("reason") or row.get("rationale_short") or ""
        print(f"  #{ordinal} {decision} {current}->{suggested} target={target}{guarded_text}: {reason[:180]}")
    if len(rows) > limit:
        print(f"  ... {len(rows) - limit} more")


def print_report(audit: Dict[str, Any], comparison: Optional[Dict[str, Any]] = None, *, limit: int = 20) -> None:
    print("Class Split Qwen review benchmark audit")
    print(f"Records: {audit['records']}")
    print(f"Decisions: {json.dumps(audit['decision_counts'], sort_keys=True)}")
    print(f"Backend tiers: {json.dumps(audit['backend_tier_counts'], sort_keys=True)}")
    print(f"Actionable recommendations: {audit['actionable_count']}")
    print(f"Guarded recommendations: {audit.get('guarded_recommendation_count', 0)}")
    print(f"Unsafe issue count: {audit['unsafe_issue_count']}")
    print(f"Issue counts: {json.dumps(audit['issue_counts'], sort_keys=True)}")
    for issue_name, rows in audit["issues"].items():
        _print_section(f"Issue: {issue_name}", rows, limit=limit)
    _print_section("Guarded clear-target candidates", audit["guarded_clear_target_candidates"], limit=limit)
    _print_section("Guarded recommendations", audit.get("guarded_recommendations") or [], limit=limit)
    _print_section("Actionable recommendations", audit["actionables"], limit=limit)
    if comparison is not None:
        print("\nComparison")
        print(f"Matched current records: {comparison['matched_current_records']}")
        print(f"Missing previous records: {comparison['missing_previous_records']}")
        print(f"Changed decisions/targets/confidence: {comparison['changed_count']}")
        for change in comparison["changes"][: max(0, limit)]:
            changed = ", ".join(
                f"{field}: {values['previous']} -> {values['current']}"
                for field, values in change["changed"].items()
            )
            print(
                f"  #{change.get('ordinal')} {change.get('current_class')}->"
                f"{change.get('suggested_neighbor_class')}: {changed}"
            )
        if comparison["changed_count"] > limit:
            print(f"  ... {comparison['changed_count'] - limit} more")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a Class Split Qwen review benchmark JSON file.")
    parser.add_argument("run_json", type=Path)
    parser.add_argument("--compare-run", type=Path, default=None)
    parser.add_argument("--write-json", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--fail-on-unsafe", action="store_true")
    args = parser.parse_args()

    run_path = args.run_json
    data = _load_json(run_path)
    records = _records(data)
    audit = audit_records(records)
    comparison = None
    if args.compare_run:
        comparison = compare_runs(records, _records(_load_json(args.compare_run)))
    print_report(audit, comparison, limit=int(args.limit or 20))

    output_payload = {"audit": audit}
    if comparison is not None:
        output_payload["comparison"] = comparison
    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(_json_sanitize(output_payload), indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote audit JSON: {args.write_json}")

    if args.fail_on_unsafe and int(audit["unsafe_issue_count"] or 0) > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
