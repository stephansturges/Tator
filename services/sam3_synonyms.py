from __future__ import annotations

import hashlib
import json
from collections import deque
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from utils.glossary import _parse_glossary_mapping, _normalize_synonym_list, _dedupe_synonyms


_SAM3_SYNONYM_CACHE: Dict[str, Dict[str, List[str]]] = {}
_SAM3_SYNONYM_CACHE_ORDER: deque[str] = deque()
_SAM3_SYNONYM_CACHE_LIMIT = 32


def _sam3_synonym_cache_key(labelmap: Sequence[str], glossary: str, max_synonyms: Optional[int]) -> str:
    joined = ",".join([str(lbl).strip() for lbl in labelmap if str(lbl).strip()])
    limit = "none" if max_synonyms is None else str(int(max_synonyms))
    payload = f"{joined}\n{glossary or ''}\nmax={limit}".encode("utf-8", errors="ignore")
    return hashlib.md5(payload).hexdigest()


def _get_cached_sam3_synonyms(cache_key: str) -> Optional[Dict[str, List[str]]]:
    cached = _SAM3_SYNONYM_CACHE.get(cache_key)
    if cached is None:
        return None
    try:
        _SAM3_SYNONYM_CACHE_ORDER.remove(cache_key)
    except ValueError:
        pass
    _SAM3_SYNONYM_CACHE_ORDER.append(cache_key)
    return cached


def _set_cached_sam3_synonyms(cache_key: str, mapping: Dict[str, List[str]]) -> None:
    _SAM3_SYNONYM_CACHE[cache_key] = mapping
    try:
        _SAM3_SYNONYM_CACHE_ORDER.remove(cache_key)
    except ValueError:
        pass
    _SAM3_SYNONYM_CACHE_ORDER.append(cache_key)
    while len(_SAM3_SYNONYM_CACHE_ORDER) > _SAM3_SYNONYM_CACHE_LIMIT:
        evict = _SAM3_SYNONYM_CACHE_ORDER.popleft()
        _SAM3_SYNONYM_CACHE.pop(evict, None)


def _agent_generate_sam3_synonyms(
    labelmap: Sequence[str],
    glossary: str,
    *,
    max_synonyms: Optional[int] = 10,
    generate_text_fn,
    extract_json_fn,
    default_synonyms: Mapping[str, List[str]],
    label_key_fn,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
    labels = [str(lbl).strip() for lbl in labelmap if str(lbl).strip()]
    if not labels:
        return {}, {}
    glossary_terms = _parse_glossary_mapping(glossary, labels)
    if max_synonyms is None:
        limit: Optional[int] = None
    else:
        limit = max(0, int(max_synonyms))
    mapping: Dict[str, List[str]] = {}
    if limit is None or limit > 0:
        cache_key = _sam3_synonym_cache_key(labels, glossary or "", limit)
        cached = _get_cached_sam3_synonyms(cache_key)
        if cached is None:
            if limit is None:
                limit_text = "as many short noun phrases (1-3 words) as are useful"
            else:
                limit_text = f"up to {limit} short noun phrases (1-3 words)"
            prompt = (
                "You generate short text prompts for a segmentation model. "
                f"Return ONLY JSON. For each label, provide {limit_text}. "
                "Use lowercase. No extra text, no markdown, no explanation. "
                "Avoid filler like 'including' or 'all kinds of'. "
                "If unsure, return an empty list for that label.\n"
                f"Labelmap: {', '.join(labels)}\n"
                f"Glossary hints:\n{glossary or 'none'}\n"
                "JSON example:\n"
                "{\"light_vehicle\": [\"car\", \"van\", \"pickup truck\", \"suv\", \"sedan\"]}"
            )
            raw = ""
            try:
                raw = generate_text_fn(prompt, max_new_tokens=384, use_system_prompt=False)
            except Exception:
                raw = ""
            data: Dict[str, Any] = {}
            json_text = extract_json_fn(raw, "{", "}")
            if json_text:
                try:
                    data = json.loads(json_text)
                except Exception:
                    data = {}
            if not data and raw:
                for line in raw.splitlines():
                    if ":" not in line:
                        continue
                    key, rest = line.split(":", 1)
                    data[key.strip()] = [item.strip() for item in rest.split(",") if item.strip()]
            norm_to_label = {label_key_fn(lbl): lbl for lbl in labels}
            for key, value in (data or {}).items():
                label = norm_to_label.get(label_key_fn(key))
                if not label:
                    continue
                items = []
                if isinstance(value, (list, tuple)):
                    items = [str(v).strip() for v in value if str(v).strip()]
                elif isinstance(value, str):
                    items = [item.strip() for item in value.split(",") if item.strip()]
                mapping[label] = _normalize_synonym_list(items)
            _set_cached_sam3_synonyms(cache_key, mapping)
        else:
            mapping = cached
    term_meta: Dict[str, Dict[str, List[str]]] = {}
    for label in labels:
        base_terms = _normalize_synonym_list([label, str(label).replace("_", " ")])
        glossary_list = _normalize_synonym_list(glossary_terms.get(label, []))
        expanded_terms = _normalize_synonym_list(mapping.get(label, []))
        expanded_terms = _dedupe_synonyms(glossary_list + expanded_terms)
        if not expanded_terms:
            fallback = default_synonyms.get(label_key_fn(label), [])
            expanded_terms = _normalize_synonym_list(fallback)
        term_meta[label] = {
            "base_terms": base_terms,
            "expanded_terms": expanded_terms,
        }
        mapping[label] = expanded_terms
    return mapping, term_meta
