#!/usr/bin/env python3
"""Run an evidence-capturing benchmark for the Qwen caption flow.

The parent process selects dense/sparse YOLO-labeled images and launches one
worker process per caption case. The worker calls the real `qwen_caption`
function so prompt building, MLX generation, guards, recovery, and trace logs
are exercised exactly as the app uses them, while a Metal abort is isolated to
that worker process.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.error
import urllib.request
import uuid

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.glossary import _normalize_labelmap_glossary, _parse_glossary_mapping
DEFAULT_DATASET = REPO_ROOT / "uploads/datasets/data_ingestion_reference_current_label_images_dataset_9526"
DEFAULT_MODEL = "mlx-community/Qwen3-VL-2B-Instruct-4bit"
CAPTION_PROVIDER_LOCAL = "local_qwen"
CAPTION_PROVIDER_OPENAI = "openai"
CAPTION_PROVIDERS = (CAPTION_PROVIDER_LOCAL, CAPTION_PROVIDER_OPENAI)
DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_OPENAI_IMAGE_DETAIL = "original"
OPENAI_IMAGE_DETAIL_CHOICES = ("original", "auto", "high", "low")
DEFAULT_OPENAI_API_KEY_FILE = "openAI_API_KEY_DoNotCommit"
DEFAULT_OPENAI_SERVICE_TIER = "standard"
OPENAI_SERVICE_TIER_CHOICES = ("standard", "batch")
DEFAULT_OPENAI_TIMEOUT_SECONDS = 120.0
DEFAULT_OPENAI_MAX_RETRIES = 5
DEFAULT_OPENAI_REASONING_EFFORT = "medium"
OPENAI_REASONING_EFFORT_CHOICES = ("low", "medium", "high", "xhigh")
DEFAULT_PROMPT = (
    "Write a detailed caption describing the scene, setting, visible objects, "
    "spatial relationships, and notable details. Prefer a high-angle or "
    "top-down wording when it fits the image."
)
LABEL_ALIASES = {
    "LightVehicle": "Light Vehicle",
    "Solarpanels": "Solar Panels",
    "Gastank": "Gas Tank",
}
QA_VERIFIER_STATUS = "machine_validated"
QA_VERIFIER_MAX_NEW_TOKENS = 2048
QA_RAW_LABEL_LEAK_PREFIX = "raw_label_term:"
QA_STRICT_SPECULATION_PATTERN = re.compile(
    (
        r"\b("
        r"likely|probably|maybe|possibly|presumably|potential|seems|appears|"
        r"suggest|suggests|suggesting|hint|hints|hinting|"
        r"indicate|indicates|indicating|infer|inferred|inference|"
        r"imply|implies|implied|could|might|"
        r"unclear|unknown|cannot determine|not enough information|"
        r"not visible|not shown|cannot be seen"
        r")\b"
    ),
    flags=re.IGNORECASE,
)
UNSUPPORTED_SPECIFIC_TERMS = {
    "Crane": ("crane", "cranes", "gantry crane", "gantry cranes"),
}
RUNNER_LOCK_NAME = ".runner.lock"
RUNNER_RESTART_REQUEST_NAME = "restart_requested.json"
RUNNER_RESTART_ACK_NAME = "restart_acknowledged.json"
RUNNER_CAPABILITY_GRACEFUL_RESTART = "graceful_restart_request"
RUNNER_CAPABILITY_PARENT_DETERMINISTIC_RECOVERY = "parent_deterministic_recovery"
RUNNER_CAPABILITY_CAPTION_IO_EVENT_SUMMARY = "caption_io_event_summary"
RUNNER_CAPABILITY_WORKER_PROGRESS_HEARTBEAT = "worker_progress_heartbeat"
RUNNER_CAPABILITY_ADAPTIVE_RETRY_PROFILE = "adaptive_retry_profile"
RUNNER_CAPABILITY_INSTRUCTION_QA = "instruction_qa_generation"
RUNNER_CAPABILITY_PARALLEL_CASES = "parallel_case_workers"
RUNNER_CAPABILITIES = (
    RUNNER_CAPABILITY_GRACEFUL_RESTART,
    RUNNER_CAPABILITY_PARENT_DETERMINISTIC_RECOVERY,
    RUNNER_CAPABILITY_CAPTION_IO_EVENT_SUMMARY,
    RUNNER_CAPABILITY_WORKER_PROGRESS_HEARTBEAT,
    RUNNER_CAPABILITY_ADAPTIVE_RETRY_PROFILE,
    RUNNER_CAPABILITY_INSTRUCTION_QA,
    RUNNER_CAPABILITY_PARALLEL_CASES,
)
WORKER_PROGRESS_JSON = "worker_progress.json"
WORKER_PROGRESS_JSONL = "worker_progress.jsonl"
QWEN_CAPTION_IO_SUMMARY_JSON = "qwen_caption_io_summary.json"
RUNNER_LOCK_STALE_SECONDS = 21600.0
RUNNER_LOCK_POLL_SECONDS = 5.0
DEFAULT_ARTIFACT_LOG_BYTES = 0
DEFAULT_COOLDOWN_BACKOFF_MULTIPLIER = 2.0
DEFAULT_MAX_COOLDOWN_AFTER_CRASH = 60.0
DEFAULT_RETRY_IMAGE_SIDE_SCALE = 0.75
DEFAULT_MIN_RETRY_IMAGE_SIDE = 256
DEFAULT_SUMMARY_ROW_LIMIT = 250
DEFAULT_INSTRUCTION_QA_MAX_TOPUP_ATTEMPTS = 6
MAX_INSTRUCTION_QA_TOPUP_ATTEMPTS = 12
SUMMARY_OK_STATUSES = {"ok", "preview_only", "skipped_completed", "skipped_existing_caption"}
RUN_SETTINGS_SCHEMA_VERSION = 1
IMAGE_SPECIFIC_REQUEST_KEYS = (
    "image_base64",
    "image_token",
    "image_name",
    "label_hints",
    "image_width",
    "image_height",
)
RUN_SETTINGS_ARG_DEFAULTS: dict[str, Any] = {
    "caption_provider": CAPTION_PROVIDER_LOCAL,
    "openai_model": DEFAULT_OPENAI_MODEL,
    "openai_image_detail": DEFAULT_OPENAI_IMAGE_DETAIL,
    "openai_api_key_file": DEFAULT_OPENAI_API_KEY_FILE,
    "openai_service_tier": DEFAULT_OPENAI_SERVICE_TIER,
    "openai_timeout": DEFAULT_OPENAI_TIMEOUT_SECONDS,
    "openai_max_retries": DEFAULT_OPENAI_MAX_RETRIES,
    "openai_reasoning_effort": DEFAULT_OPENAI_REASONING_EFFORT,
    "model_id": DEFAULT_MODEL,
    "refinement_model_id": "same",
    "fallback_model_id": "auto",
    "loop_recovery": "safe_retry_fallback",
    "caption_mode": "full",
    "windowed_full_image_strategy": "visual",
    "max_boxes": 0,
    "max_new_tokens": None,
    "final_sentences": 8,
    "window_size": 672,
    "window_overlap": 0.1,
    "mlx_max_image_side": 512,
    "retry_image_side_scale": DEFAULT_RETRY_IMAGE_SIDE_SCALE,
    "min_retry_image_side": DEFAULT_MIN_RETRY_IMAGE_SIDE,
    "cooldown_after_success": 0.0,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 20,
    "prompt": DEFAULT_PROMPT,
    "preview_only": False,
    "use_sampling": False,
    "instruction_dataset": False,
    "subcaptions_per_image": 0,
    "include_caption0_in_training": True,
    "include_generated_qa_in_training": True,
    "include_deterministic_metadata_qa": False,
    "instruction_qa_imposed_questions": [],
    "instruction_max_new_tokens": None,
    "instruction_qa_max_topup_attempts": DEFAULT_INSTRUCTION_QA_MAX_TOPUP_ATTEMPTS,
    "instruction_qa_restrict_speculative_language": False,
    "include_source_annotations_in_generator_context": True,
    "strict_grounding": True,
    "qa_mix": "balanced",
    "answer_format": "natural",
    "parallel_cases": 1,
}


def quality_label_variants(label: str) -> list[str]:
    term = re.sub(r"\s+", " ", str(label or "").strip().lower())
    if not term:
        return []
    words = term.split()
    variants = {term}
    last = words[-1]
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
    plural_words = list(words)
    plural_words[-1] = plural
    variants.add(" ".join(plural_words))
    return sorted(variants, key=len, reverse=True)


def quality_term_pattern(term: str) -> str:
    return r"\s+".join(re.escape(part) for part in re.sub(r"\s+", " ", term).split())


def _quality_glossary_label_terms(
    label: str,
    glossary_map: Mapping[str, Sequence[Any]] | None = None,
) -> list[str]:
    clean_label = str(label or "").strip()
    terms: list[str] = []
    seen: set[str] = set()

    def add(raw: Any) -> None:
        term = _display_term(raw)
        key = _label_key(term)
        if term and key and key not in seen:
            seen.add(key)
            terms.append(term)

    for raw_term in (glossary_map or {}).get(clean_label, []) or []:
        add(raw_term)
    add(_natural_label(clean_label))
    add(clean_label)
    return terms


def quality_label_present(
    text: str,
    label: str,
    glossary_map: Mapping[str, Sequence[Any]] | None = None,
) -> bool:
    lowered = str(text or "").lower()
    compact = lowered.replace(" ", "")
    for term in _quality_glossary_label_terms(label, glossary_map):
        for variant in quality_label_variants(term):
            if re.search(rf"\b{quality_term_pattern(variant)}\b", lowered, flags=re.IGNORECASE):
                return True
            if variant.replace(" ", "") in compact:
                return True
    return False


def quality_negative_mentions_label(
    sentence: str,
    label: str,
    glossary_map: Mapping[str, Sequence[Any]] | None = None,
) -> bool:
    lowered = str(sentence or "").lower()
    for term in _quality_glossary_label_terms(label, glossary_map):
        for variant in quality_label_variants(term):
            term_pattern = quality_term_pattern(variant)
            negative_patterns = [
                rf"\bno\b(?:\s+[\w'-]+){{0,8}}\s+{term_pattern}\b",
                rf"\bwithout\b(?:\s+[\w'-]+){{0,8}}\s+{term_pattern}\b",
                rf"\bnot\s+any\b(?:\s+[\w'-]+){{0,8}}\s+{term_pattern}\b",
                rf"\b(?:absent|missing)\b(?:\s+[\w'-]+){{0,4}}\s+{term_pattern}\b",
            ]
            if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in negative_patterns):
                return True
    return False
SAMPLE_STRATEGY_STRESS_PLUS_RANDOM = "stress_plus_random"


@dataclass(frozen=True)
class DatasetItem:
    stem: str
    image_path: Path
    label_path: Path
    label_count: int
    class_counts: Counter[str]

    @property
    def dominant_class_count(self) -> int:
        return max(self.class_counts.values(), default=0)

    @property
    def class_count(self) -> int:
        return len(self.class_counts)


def load_labelmap(dataset_root: Path) -> list[str]:
    labelmap = dataset_root / "labelmap.txt"
    if not labelmap.exists():
        raise SystemExit(f"Missing labelmap: {labelmap}")
    return [line.strip() for line in labelmap.read_text().splitlines() if line.strip()]


def class_name(raw: str) -> str:
    raw = str(raw or "").strip()
    return LABEL_ALIASES.get(raw, raw)


def _label_key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").strip().lower())


def _natural_label(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    aliased = LABEL_ALIASES.get(text)
    if aliased:
        return aliased
    spaced = text.replace("_", " ")
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", spaced)
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)
    return re.sub(r"\s+", " ", spaced).strip()


def _display_term(raw: Any) -> str:
    text = re.sub(r"\s+", " ", str(raw or "").strip())
    text = text.strip(" \t\r\n\"'`.,;:")
    if not text or any(ch in text for ch in "[]{}"):
        return ""
    if not re.search(r"[A-Za-z0-9]", text):
        return ""
    return text[:80]


def _request_template_value(args: argparse.Namespace, key: str, default: Any = None) -> Any:
    try:
        template = load_request_template(getattr(args, "request_json", None))
    except Exception:
        return default
    return template.get(key, default) if isinstance(template, Mapping) else default


def _case_label_names(case: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for raw_label, raw_count in _case_class_counts(case).items():
        label = str(raw_label or "").strip()
        if not label:
            continue
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError, OverflowError):
            count = 0
        if count <= 0:
            continue
        key = _label_key(label)
        if key and key not in seen:
            seen.add(key)
            names.append(label)
    return names


def _case_raw_label_variants(label: str) -> list[str]:
    label_text = str(label or "").strip()
    variants = {label_text}
    for raw_label, alias in LABEL_ALIASES.items():
        if _label_key(alias) == _label_key(label_text):
            variants.add(raw_label)
            variants.add(_natural_label(raw_label))
    return [term for term in (_display_term(item) for item in variants) if term]


def _case_glossary_map(case: Mapping[str, Any], args: argparse.Namespace) -> dict[str, list[str]]:
    raw_glossary = _request_template_value(args, "labelmap_glossary", None)
    if not raw_glossary:
        return {}
    labels = _case_label_names(case)
    parse_labels: list[str] = []
    seen: set[str] = set()
    for label in labels:
        for term in [label, *_case_raw_label_variants(label)]:
            key = _label_key(term)
            if key and key not in seen:
                seen.add(key)
                parse_labels.append(term)
    try:
        parsed = _parse_glossary_mapping(_normalize_labelmap_glossary(raw_glossary), parse_labels)
    except Exception:
        return {}
    alias_to_case_label: dict[str, str] = {}
    for label in labels:
        for term in [label, *_case_raw_label_variants(label)]:
            key = _label_key(term)
            if key:
                alias_to_case_label[key] = label
    out: dict[str, list[str]] = {}
    for raw_label, raw_terms in parsed.items():
        case_label = alias_to_case_label.get(_label_key(raw_label), str(raw_label or "").strip())
        if not case_label:
            continue
        terms: list[str] = []
        seen_terms: set[str] = set()
        for raw_term in raw_terms or []:
            term = _display_term(raw_term)
            key = term.lower()
            if term and key not in seen_terms:
                seen_terms.add(key)
                terms.append(term)
        if terms:
            out[case_label] = terms
    return out


def _case_preferred_label(label: str, glossary_map: Mapping[str, Sequence[Any]] | None = None) -> str:
    clean_label = str(label or "").strip()
    for term in (glossary_map or {}).get(clean_label, []) or []:
        display = _display_term(term)
        if display:
            return display
    return _natural_label(clean_label)


def _case_canonical_class_counts(case: Mapping[str, Any], args: argparse.Namespace) -> dict[str, int]:
    glossary_map = _case_glossary_map(case, args)
    counts: dict[str, int] = {}
    for raw_label, raw_count in _case_class_counts(case).items():
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError, OverflowError):
            continue
        if count <= 0:
            continue
        label = _case_preferred_label(str(raw_label or "").strip(), glossary_map)
        if not label:
            continue
        counts[label] = counts.get(label, 0) + count
    return counts


def _case_glossary_context(case: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    glossary_map = _case_glossary_map(case, args)
    entries: list[dict[str, Any]] = []
    for label in _case_label_names(case):
        preferred = _case_preferred_label(label, glossary_map)
        raw_terms = [
            term
            for term in _case_raw_label_variants(label)
            if _label_key(term) != _label_key(preferred)
        ]
        entry: dict[str, Any] = {
            "source_class": label,
            "canonical_term": preferred,
        }
        if raw_terms:
            entry["raw_terms_to_avoid"] = sorted(set(raw_terms), key=str.lower)
        variants = [
            _display_term(term)
            for term in (glossary_map.get(label) or [])
            if _display_term(term) and _label_key(term) != _label_key(preferred)
        ]
        if variants:
            entry["glossary_variants"] = variants[:6]
        entries.append(entry)
    return {
        "policy": (
            "Use canonical_term for labeled classes. Never output raw_terms_to_avoid. "
            "Use glossary_variants only when the image clearly supports the narrower variant."
        ),
        "classes": entries,
    }


def read_yolo_counts(label_path: Path, names: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not label_path.exists():
        return counts
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_idx = int(float(parts[0]))
        except ValueError:
            continue
        label = names[cls_idx] if 0 <= cls_idx < len(names) else f"class_{cls_idx}"
        counts[class_name(label)] += 1
    return counts


def discover_items(dataset_root: Path) -> list[DatasetItem]:
    names = load_labelmap(dataset_root)
    image_dir = dataset_root / "train" / "images"
    label_dir = dataset_root / "train" / "labels"
    if not image_dir.exists():
        image_dir = dataset_root / "images"
    if not label_dir.exists():
        label_dir = dataset_root / "labels"
    images = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    }
    labels = {path.stem: path for path in label_dir.glob("*.txt")} if label_dir.exists() else {}
    items: list[DatasetItem] = []
    for stem in sorted(images):
        label_path = labels.get(stem) or (label_dir / f"{stem}.txt")
        counts = read_yolo_counts(label_path, names)
        items.append(
            DatasetItem(
                stem=stem,
                image_path=images[stem],
                label_path=label_path,
                label_count=sum(counts.values()),
                class_counts=counts,
            )
        )
    if not items:
        raise SystemExit(f"No image items found under {dataset_root}")
    return items


def first_matching(items: list[DatasetItem], predicate: Any, key: Any) -> DatasetItem | None:
    matches = [item for item in items if predicate(item)]
    if not matches:
        return None
    return sorted(matches, key=key)[0]


def select_cases(items: list[DatasetItem]) -> list[dict[str, Any]]:
    selected: list[tuple[str, DatasetItem, str]] = []
    dense = max(items, key=lambda item: item.label_count)
    selected.append(("dense_max", dense, "full"))
    mixed_dense = first_matching(
        items,
        lambda item: item.label_count >= 80 and item.class_count >= 3,
        key=lambda item: (-item.class_count, -item.label_count, item.stem),
    )
    if mixed_dense and mixed_dense.stem != dense.stem:
        selected.append(("dense_mixed", mixed_dense, "full"))
    medium = first_matching(
        items,
        lambda item: 15 <= item.label_count <= 45,
        key=lambda item: (abs(item.label_count - 25), -item.class_count, item.stem),
    )
    if medium:
        selected.append(("medium", medium, "full"))
    sparse = first_matching(
        items,
        lambda item: 1 <= item.label_count <= 4,
        key=lambda item: (item.label_count, -item.class_count, item.stem),
    )
    if sparse:
        selected.append(("sparse_labeled", sparse, "full"))
    empty = first_matching(
        items,
        lambda item: item.label_count == 0,
        key=lambda item: item.stem,
    )
    if empty:
        selected.append(("sparse_empty", empty, "full"))
    if dense:
        selected.append(("dense_windowed", dense, "windowed"))

    cases: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for name, item, caption_mode in selected:
        if name in seen_names:
            continue
        seen_names.add(name)
        cases.append(
            {
                "name": name,
                "stem": item.stem,
                "image_path": str(item.image_path),
                "label_path": str(item.label_path),
                "label_count": item.label_count,
                "class_counts": dict(item.class_counts),
                "caption_mode": caption_mode,
            }
        )
    return cases


def select_all_image_cases(items: list[DatasetItem], *, caption_mode: str) -> list[dict[str, Any]]:
    mode = caption_mode if caption_mode in {"full", "windowed"} else "full"
    return [
        {
            "case_id": f"image:{item.stem}:{mode}",
            "name": f"image_{index:06d}",
            "stem": item.stem,
            "image_path": str(item.image_path),
            "label_path": str(item.label_path),
            "label_count": item.label_count,
            "class_counts": dict(item.class_counts),
            "caption_mode": mode,
        }
        for index, item in enumerate(items, start=1)
    ]


def case_key(case: Mapping[str, Any]) -> str:
    explicit = str(case.get("case_id") or "").strip()
    if explicit:
        return explicit
    return f"{case.get('name') or 'case'}:{case.get('stem') or ''}:{case.get('caption_mode') or 'full'}"


def _case_label_count(case: Mapping[str, Any]) -> int:
    try:
        return max(0, int(case.get("label_count") or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _case_class_counts(case: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = case.get("class_counts")
    return raw if isinstance(raw, Mapping) else {}


def _case_class_count(case: Mapping[str, Any]) -> int:
    return len([key for key, value in _case_class_counts(case).items() if key and value])


def _case_dominant_class_count(case: Mapping[str, Any]) -> int:
    counts = []
    for value in _case_class_counts(case).values():
        try:
            counts.append(max(0, int(value or 0)))
        except (TypeError, ValueError, OverflowError):
            continue
    return max(counts, default=0)


def _case_stable_sort_key(case: Mapping[str, Any], original_index: Mapping[str, int]) -> tuple[int, str]:
    return (original_index.get(case_key(case), 0), case_key(case))


def sample_cases_with_meta(
    cases: Sequence[Mapping[str, Any]],
    *,
    sample_size: int,
    sample_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_cases = [dict(case) for case in cases if isinstance(case, Mapping)]
    requested = max(0, int(sample_size or 0))
    meta: dict[str, Any] = {
        "strategy": "all" if requested <= 0 or requested >= len(source_cases) else SAMPLE_STRATEGY_STRESS_PLUS_RANDOM,
        "source_cases": len(source_cases),
        "selected_cases": len(source_cases),
        "requested_sample_size": requested,
        "sample_seed": int(sample_seed or 0),
        "stress_case_keys": [],
        "random_fill_case_keys": [],
    }
    if requested <= 0 or requested >= len(source_cases):
        return source_cases, meta

    original_index = {case_key(case): index for index, case in enumerate(source_cases)}
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    stress_keys: list[str] = []

    def add_stress(case: Mapping[str, Any] | None) -> None:
        if case is None or len(selected) >= requested:
            return
        key = case_key(case)
        if key in selected_keys:
            return
        selected.append(dict(case))
        selected_keys.add(key)
        stress_keys.append(key)

    def max_case(key: Any, pool: Sequence[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        candidates = list(pool if pool is not None else source_cases)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda case: (*key(case), -original_index.get(case_key(case), 0)),
        )

    add_stress(max_case(lambda case: (_case_label_count(case), _case_class_count(case), _case_dominant_class_count(case))))
    add_stress(max_case(lambda case: (_case_class_count(case), _case_label_count(case), _case_dominant_class_count(case))))
    add_stress(max_case(lambda case: (_case_dominant_class_count(case), _case_label_count(case), _case_class_count(case))))

    labeled_cases = [case for case in source_cases if _case_label_count(case) > 0]
    if labeled_cases:
        add_stress(
            min(
                labeled_cases,
                key=lambda case: (
                    _case_label_count(case),
                    -_case_class_count(case),
                    original_index.get(case_key(case), 0),
                ),
            )
        )
    empty_cases = [case for case in source_cases if _case_label_count(case) == 0]
    if empty_cases:
        add_stress(min(empty_cases, key=lambda case: _case_stable_sort_key(case, original_index)))

    modes = sorted({str(case.get("caption_mode") or "full") for case in source_cases})
    for mode in modes:
        mode_cases = [case for case in source_cases if str(case.get("caption_mode") or "full") == mode]
        add_stress(
            max_case(
                lambda case: (_case_label_count(case), _case_class_count(case), _case_dominant_class_count(case)),
                pool=mode_cases,
            )
        )

    remaining = [case for case in source_cases if case_key(case) not in selected_keys]
    rng = random.Random(sample_seed)
    fill_count = max(0, min(requested - len(selected), len(remaining)))
    random_fill = rng.sample(remaining, k=fill_count) if fill_count else []
    selected.extend(dict(case) for case in random_fill)

    meta.update(
        {
            "selected_cases": len(selected),
            "stress_case_keys": stress_keys,
            "random_fill_case_keys": [case_key(case) for case in random_fill],
        }
    )
    return selected, meta


def sample_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    sample_size: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    sampled, _meta = sample_cases_with_meta(cases, sample_size=sample_size, sample_seed=sample_seed)
    return sampled


def case_dir_name(index: int, case: Mapping[str, Any]) -> str:
    raw = case_key(case)
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case.get("name") or "case")).strip("._-") or "case"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case.get("stem") or "image")).strip("._-") or "image"
    return f"{index:06d}_{name[:32]}_{stem[:32]}_{digest}"


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp_path.replace(path)


def _read_runner_restart_request(output_root: Path) -> dict[str, Any] | None:
    path = output_root / RUNNER_RESTART_REQUEST_NAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        payload = {}
    request = dict(payload) if isinstance(payload, Mapping) else {}
    request.setdefault("reason", "restart_requested")
    request.setdefault("path", str(path))
    return request


def _consume_runner_restart_request(
    output_root: Path,
    *,
    processed: int,
    total_cases: int,
) -> dict[str, Any] | None:
    request = _read_runner_restart_request(output_root)
    if request is None:
        return None
    request_path = output_root / RUNNER_RESTART_REQUEST_NAME
    ack = {
        "event": "runner_restart_request_acknowledged",
        "acknowledged_at": datetime.now(timezone.utc).isoformat(),
        "acknowledged_epoch": time.time(),
        "processed": processed,
        "total_cases": total_cases,
        "request": request,
    }
    atomic_write_json(output_root / RUNNER_RESTART_ACK_NAME, ack)
    try:
        request_path.unlink()
    except FileNotFoundError:
        pass
    return ack


def normalize_artifact_log_bytes(value: Any) -> int:
    try:
        limit = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_ARTIFACT_LOG_BYTES
    return max(0, min(limit, 1_073_741_824))


def return_signal_fields(return_code: Any) -> dict[str, Any]:
    if not isinstance(return_code, int) or return_code >= 0:
        return {"return_signal": None, "return_signal_name": None}
    signal_number = abs(return_code)
    try:
        signal_name = signal.Signals(signal_number).name
    except ValueError:
        signal_name = f"SIG{signal_number}"
    return {
        "return_signal": signal_number,
        "return_signal_name": signal_name,
    }


def attempt_failure_kind(*, return_code: Any, row_status: str, timeout: bool) -> str:
    if timeout:
        return "timeout"
    if isinstance(return_code, int) and return_code < 0:
        return "signal_exit"
    if isinstance(return_code, int) and return_code != 0:
        return "nonzero_exit"
    if str(row_status or "") in {"exception", "missing_result", "error"}:
        return "worker_error"
    return "none"


def attempt_cooldown_seconds(args: argparse.Namespace, *, failed_attempt: int, failure_kind: str) -> float:
    base = max(0.0, float(getattr(args, "cooldown_after_crash", 0.0) or 0.0))
    if base <= 0:
        return 0.0
    multiplier = max(1.0, float(
        getattr(args, "cooldown_backoff_multiplier", DEFAULT_COOLDOWN_BACKOFF_MULTIPLIER)
        or DEFAULT_COOLDOWN_BACKOFF_MULTIPLIER
    ))
    cap = max(base, float(
        getattr(args, "max_cooldown_after_crash", DEFAULT_MAX_COOLDOWN_AFTER_CRASH)
        or DEFAULT_MAX_COOLDOWN_AFTER_CRASH
    ))
    hard_failures = {"signal_exit", "timeout", "nonzero_exit"}
    exponent = max(0, int(failed_attempt) - 1) if failure_kind in hard_failures else 0
    return min(cap, base * (multiplier ** exponent))


def retry_mlx_image_side(
    base_image_side: Any,
    *,
    attempt: int,
    scale: Any = DEFAULT_RETRY_IMAGE_SIDE_SCALE,
    min_image_side: Any = DEFAULT_MIN_RETRY_IMAGE_SIDE,
    previous_failure_kind: str | None = None,
) -> int:
    """Return the MLX image side for a retry attempt.

    The first attempt always uses the requested side. Later attempts may reduce
    only the generation tensor size; source image dimensions and boxes are not
    altered. A prior native signal exit jumps straight to the configured floor
    so set-and-forget runs do not repeat the same GPU-pressure fault.
    """

    try:
        base = int(base_image_side)
    except (TypeError, ValueError, OverflowError):
        base = 512
    base = max(1, base)
    try:
        attempt_number = max(1, int(attempt))
    except (TypeError, ValueError, OverflowError):
        attempt_number = 1
    try:
        retry_scale = float(scale)
    except (TypeError, ValueError, OverflowError):
        retry_scale = DEFAULT_RETRY_IMAGE_SIDE_SCALE
    if attempt_number <= 1 or retry_scale <= 0.0 or retry_scale >= 1.0:
        return base
    try:
        minimum = int(min_image_side)
    except (TypeError, ValueError, OverflowError):
        minimum = DEFAULT_MIN_RETRY_IMAGE_SIDE
    minimum = max(1, min(base, minimum))
    if str(previous_failure_kind or "") == "signal_exit":
        return minimum
    adapted = int(round(float(base) * (retry_scale ** (attempt_number - 1))))
    return min(base, max(minimum, adapted))


def attempt_generation_profile(
    args: argparse.Namespace,
    *,
    attempt: int,
    previous_failure_kind: str | None = None,
) -> dict[str, Any]:
    base_side = getattr(args, "mlx_max_image_side", RUN_SETTINGS_ARG_DEFAULTS["mlx_max_image_side"])
    scale = getattr(args, "retry_image_side_scale", DEFAULT_RETRY_IMAGE_SIDE_SCALE)
    minimum = getattr(args, "min_retry_image_side", DEFAULT_MIN_RETRY_IMAGE_SIDE)
    effective_side = retry_mlx_image_side(
        base_side,
        attempt=attempt,
        scale=scale,
        min_image_side=minimum,
        previous_failure_kind=previous_failure_kind,
    )
    try:
        base_side_int = int(base_side)
    except (TypeError, ValueError, OverflowError):
        base_side_int = int(RUN_SETTINGS_ARG_DEFAULTS["mlx_max_image_side"])
    try:
        scale_float = float(scale)
    except (TypeError, ValueError, OverflowError):
        scale_float = DEFAULT_RETRY_IMAGE_SIDE_SCALE
    try:
        minimum_int = int(minimum)
    except (TypeError, ValueError, OverflowError):
        minimum_int = DEFAULT_MIN_RETRY_IMAGE_SIDE
    attempt_number = max(1, int(attempt))
    if attempt_number <= 1:
        retry_reason = "initial_attempt"
    elif scale_float <= 0.0 or scale_float >= 1.0:
        retry_reason = "adaptive_retry_disabled"
    elif str(previous_failure_kind or "") == "signal_exit":
        retry_reason = "previous_signal_exit_min_side"
    else:
        retry_reason = "scaled_retry"
    return {
        "attempt": attempt_number,
        "mlx_max_image_side": effective_side,
        "base_mlx_max_image_side": max(1, base_side_int),
        "retry_image_side_scale": scale_float,
        "min_retry_image_side": max(1, minimum_int),
        "adaptive_retry_image_side": effective_side < max(1, base_side_int),
        "adaptive_retry_reason": retry_reason,
        "previous_attempt_failure_kind": previous_failure_kind,
    }


def _artifact_truncation_marker(
    *,
    original_bytes: int,
    max_bytes: int,
    source: str,
    jsonl: bool,
) -> bytes:
    if jsonl:
        return (
            json.dumps(
                {
                    "event": "artifact_truncated",
                    "original_bytes": original_bytes,
                    "max_bytes": max_bytes,
                    "source": source,
                    "note": "set --max-artifact-log-bytes 0 to keep full raw logs",
                },
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    return (
        f"[tator artifact truncated: source={source} original_bytes={original_bytes} "
        f"max_bytes={max_bytes}; set --max-artifact-log-bytes 0 to keep full raw logs]\n"
    ).encode("utf-8")


def _bounded_artifact_bytes(
    data: bytes,
    *,
    max_bytes: int,
    source: str,
    jsonl: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    original_bytes = len(data)
    normalized_limit = normalize_artifact_log_bytes(max_bytes)
    if normalized_limit <= 0 or original_bytes <= normalized_limit:
        return data, {
            "source": source,
            "original_bytes": original_bytes,
            "written_bytes": original_bytes,
            "max_bytes": normalized_limit,
            "truncated": False,
        }
    marker = _artifact_truncation_marker(
        original_bytes=original_bytes,
        max_bytes=normalized_limit,
        source=source,
        jsonl=jsonl,
    )
    if len(marker) >= normalized_limit:
        marker = (b'{"event":"artifact_truncated"}\n' if jsonl else b"[artifact truncated]\n")
    if len(marker) >= normalized_limit:
        marker = marker[:normalized_limit]
    tail_budget = max(0, normalized_limit - len(marker))
    tail = data[-tail_budget:] if tail_budget else b""
    if tail:
        newline_index = tail.find(b"\n")
        if 0 <= newline_index < len(tail) - 1:
            tail = tail[newline_index + 1 :]
    bounded = marker + tail
    if len(bounded) > normalized_limit:
        bounded = (marker + tail[-max(0, normalized_limit - len(marker)) :])[:normalized_limit]
    return bounded, {
        "source": source,
        "original_bytes": original_bytes,
        "written_bytes": len(bounded),
        "max_bytes": normalized_limit,
        "truncated": True,
    }


def write_text_artifact(path: Path, text: str, *, max_bytes: int, source: str) -> dict[str, Any]:
    payload, meta = _bounded_artifact_bytes(
        str(text or "").encode("utf-8", errors="replace"),
        max_bytes=max_bytes,
        source=source,
        jsonl=path.suffix.lower() == ".jsonl",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {**meta, "path": str(path)}


def write_bytes_artifact(path: Path, data: bytes, *, max_bytes: int, source: str) -> dict[str, Any]:
    payload, meta = _bounded_artifact_bytes(
        data,
        max_bytes=max_bytes,
        source=source,
        jsonl=path.suffix.lower() == ".jsonl",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {**meta, "path": str(path)}


def copy_text_artifact(src: Path, dst: Path, *, max_bytes: int, source: str) -> dict[str, Any]:
    payload, meta = _bounded_artifact_bytes(
        src.read_bytes(),
        max_bytes=max_bytes,
        source=source,
        jsonl=dst.suffix.lower() == ".jsonl",
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(payload)
    return {**meta, "path": str(dst)}


def summarize_qwen_caption_io_event_lines(lines: Sequence[str], *, path: str = "") -> dict[str, Any]:
    event_counts: Counter[str] = Counter()
    invalid_rows = 0
    rows = 0
    max_prompt_tokens = 0
    max_input_tokens = 0
    effective_max_new_tokens: list[int] = []
    requested_max_new_tokens: list[int] = []
    prompt_budget_adapted_events = 0

    def _json_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            invalid_rows += 1
            continue
        if not isinstance(payload, Mapping):
            invalid_rows += 1
            continue
        event = str(payload.get("event") or "").strip()
        if not event:
            continue
        rows += 1
        event_counts[event] += 1
        if event == "prompt_budget":
            prompt_tokens = _json_int(payload.get("prompt_tokens"))
            input_tokens = _json_int(payload.get("input_tokens"))
            requested_tokens = _json_int(payload.get("requested_max_new_tokens"))
            effective_tokens = _json_int(payload.get("effective_max_new_tokens"))
            if prompt_tokens is not None:
                max_prompt_tokens = max(max_prompt_tokens, max(0, prompt_tokens))
            if input_tokens is not None:
                max_input_tokens = max(max_input_tokens, max(0, input_tokens))
            if requested_tokens is not None:
                requested_max_new_tokens.append(max(0, requested_tokens))
            if effective_tokens is not None:
                effective_max_new_tokens.append(max(0, effective_tokens))
            if (
                requested_tokens is not None
                and effective_tokens is not None
                and effective_tokens < requested_tokens
            ):
                prompt_budget_adapted_events += 1
    summary = {
        "path": str(path),
        "rows": rows,
        "invalid_rows": invalid_rows,
        "event_counts": dict(sorted(event_counts.items())),
        "stream_loop_detected_events": int(event_counts.get("stream_loop_detected") or 0),
        "loop_trim_events": int(event_counts.get("loop_trim") or 0),
        "prompt_budget_events": int(event_counts.get("prompt_budget") or 0),
        "prompt_budget_adapted_events": prompt_budget_adapted_events,
        "max_prompt_tokens": max_prompt_tokens,
        "max_input_tokens": max_input_tokens,
        "recovery_events": int(event_counts.get("recovery") or 0),
    }
    if effective_max_new_tokens:
        summary["effective_max_new_tokens_min"] = min(effective_max_new_tokens)
        summary["effective_max_new_tokens_max"] = max(effective_max_new_tokens)
    if requested_max_new_tokens:
        summary["requested_max_new_tokens_max"] = max(requested_max_new_tokens)
    summary["loop_guard_events"] = (
        int(summary["stream_loop_detected_events"]) + int(summary["loop_trim_events"])
    )
    return summary


def summarize_qwen_caption_io_event_bytes(data: bytes, *, path: str = "") -> dict[str, Any]:
    text = data.decode("utf-8", errors="replace")
    return summarize_qwen_caption_io_event_lines(text.splitlines(), path=path)


def summarize_qwen_caption_io_events(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return summarize_qwen_caption_io_event_lines(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            path=str(path),
        )
    except OSError:
        return {}


def read_qwen_caption_io_summary(attempt_dir: Path) -> dict[str, Any]:
    summary_path = attempt_dir / QWEN_CAPTION_IO_SUMMARY_JSON
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            payload.setdefault("path", str(attempt_dir / "qwen_caption_io.jsonl"))
            return payload
    return summarize_qwen_caption_io_events(attempt_dir / "qwen_caption_io.jsonl")


def _qwen_caption_io_safe_run_id(run_id: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(run_id or "manual"))[:80] or "manual"


def _qwen_caption_io_run_paths(run_id: Any) -> tuple[Path, Path]:
    safe_run_id = _qwen_caption_io_safe_run_id(run_id)
    run_root = REPO_ROOT / "logs" / "qwen_caption_io"
    return run_root / f"{safe_run_id}.jsonl", run_root / f"{safe_run_id}.log"


def _collect_qwen_run_id(run_ids: list[str], payload: Mapping[str, Any]) -> None:
    run_id = str(payload.get("run_id") or "").strip()
    if run_id and run_id not in run_ids:
        run_ids.append(run_id)


def collect_worker_progress_run_ids(output_dir: Path) -> list[str]:
    run_ids: list[str] = []
    progress_jsonl = output_dir / WORKER_PROGRESS_JSONL
    if progress_jsonl.exists():
        try:
            lines = progress_jsonl.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        for line in lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                _collect_qwen_run_id(run_ids, payload)
    progress_json = output_dir / WORKER_PROGRESS_JSON
    if progress_json.exists():
        payload = _read_json_dict(progress_json)
        if payload:
            _collect_qwen_run_id(run_ids, payload)
    return run_ids


def _concatenate_artifact_sources(paths: Sequence[Path]) -> bytes:
    chunks: list[bytes] = []
    for path in paths:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if not data:
            continue
        if chunks and not chunks[-1].endswith(b"\n"):
            chunks.append(b"\n")
        chunks.append(data)
        if not data.endswith(b"\n"):
            chunks.append(b"\n")
    return b"".join(chunks)


def _qwen_caption_io_missing_summary(
    output_dir: Path,
    *,
    run_ids: Sequence[str],
    source_records: Sequence[Mapping[str, str]],
    reason: str,
) -> dict[str, Any]:
    return {
        "path": str(output_dir / "qwen_caption_io.jsonl"),
        "source": "worker_progress_run_ids",
        "source_run_ids": list(run_ids),
        "source_files": [],
        "source_records": [dict(record) for record in source_records],
        "rows": 0,
        "invalid_rows": 0,
        "event_counts": {},
        "stream_loop_detected_events": 0,
        "loop_trim_events": 0,
        "prompt_budget_events": 0,
        "prompt_budget_adapted_events": 0,
        "max_prompt_tokens": 0,
        "max_input_tokens": 0,
        "recovery_events": 0,
        "loop_guard_events": 0,
        "missing_trace": True,
        "missing_trace_reason": reason,
        "fallback_skipped": "qwen_caption_io_latest",
        "unattended_evidence": False,
    }


def _write_qwen_caption_io_summary(
    output_dir: Path,
    summary: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    summary_path = output_dir / QWEN_CAPTION_IO_SUMMARY_JSON
    summary_path.write_text(
        json.dumps(dict(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "path": str(summary_path),
        "source": source,
        "source_run_ids": list(summary.get("source_run_ids") or []),
        "source_files": list(summary.get("source_files") or []),
        "truncated": False,
    }


def copy_worker_qwen_caption_io_artifacts(
    output_dir: Path,
    *,
    max_bytes: int,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    artifact_files: dict[str, Any] = {}
    artifact_errors: list[dict[str, str]] = []
    run_ids = collect_worker_progress_run_ids(output_dir)
    source_records: list[dict[str, str]] = []
    jsonl_sources: list[Path] = []
    log_sources: list[Path] = []
    for run_id in run_ids:
        jsonl_path, log_path = _qwen_caption_io_run_paths(run_id)
        record = {"run_id": run_id, "jsonl": str(jsonl_path), "log": str(log_path)}
        source_records.append(record)
        if jsonl_path.exists():
            jsonl_sources.append(jsonl_path)
        if log_path.exists():
            log_sources.append(log_path)

    copied_from_per_run = bool(jsonl_sources or log_sources)
    if copied_from_per_run:
        if jsonl_sources:
            try:
                jsonl_data = _concatenate_artifact_sources(jsonl_sources)
                summary = summarize_qwen_caption_io_event_bytes(
                    jsonl_data,
                    path=str(output_dir / "qwen_caption_io.jsonl"),
                )
                summary["source"] = "qwen_caption_io_per_run"
                summary["source_run_ids"] = list(run_ids)
                summary["source_files"] = [str(path) for path in jsonl_sources]
                artifact_files["qwen_caption_io_summary"] = _write_qwen_caption_io_summary(
                    output_dir,
                    summary,
                    source="qwen_caption_io_per_run_summary",
                )
                meta = write_bytes_artifact(
                    output_dir / "qwen_caption_io.jsonl",
                    jsonl_data,
                    max_bytes=max_bytes,
                    source="qwen_caption_io_per_run",
                )
                meta["source_run_ids"] = list(run_ids)
                meta["source_files"] = [str(path) for path in jsonl_sources]
                artifact_files["qwen_caption_io_jsonl"] = meta
            except Exception as exc:  # noqa: BLE001
                artifact_errors.append({
                    "artifact": "qwen_caption_io_jsonl",
                    "type": type(exc).__name__,
                    "message": str(exc),
                })
        if log_sources:
            try:
                log_data = _concatenate_artifact_sources(log_sources)
                meta = write_bytes_artifact(
                    output_dir / "qwen_caption_io.log",
                    log_data,
                    max_bytes=max_bytes,
                    source="qwen_caption_io_per_run",
                )
                meta["source_run_ids"] = list(run_ids)
                meta["source_files"] = [str(path) for path in log_sources]
                artifact_files["qwen_caption_io_log"] = meta
            except Exception as exc:  # noqa: BLE001
                artifact_errors.append({
                    "artifact": "qwen_caption_io_log",
                    "type": type(exc).__name__,
                    "message": str(exc),
                })
        if source_records:
            try:
                source_meta = {
                    "source": "worker_progress_run_ids",
                    "run_ids": list(run_ids),
                    "files": source_records,
                }
                artifact_files["qwen_caption_io_sources"] = write_text_artifact(
                    output_dir / "qwen_caption_io_sources.json",
                    json.dumps(source_meta, indent=2, sort_keys=True),
                    max_bytes=max_bytes,
                    source="qwen_caption_io_sources",
                )
            except Exception as exc:  # noqa: BLE001
                artifact_errors.append({
                    "artifact": "qwen_caption_io_sources",
                    "type": type(exc).__name__,
                    "message": str(exc),
                })
        return artifact_files, artifact_errors

    missing_reason = (
        "worker progress did not expose a qwen caption run id"
        if not run_ids
        else "worker qwen caption run ids had no matching per-run trace files"
    )
    try:
        summary = _qwen_caption_io_missing_summary(
            output_dir,
            run_ids=run_ids,
            source_records=source_records,
            reason=missing_reason,
        )
        artifact_files["qwen_caption_io_summary"] = _write_qwen_caption_io_summary(
            output_dir,
            summary,
            source="qwen_caption_io_missing_summary",
        )
    except Exception as exc:  # noqa: BLE001
        artifact_errors.append({
            "artifact": "qwen_caption_io_summary",
            "type": type(exc).__name__,
            "message": str(exc),
        })
    if source_records:
        try:
            source_meta = {
                "source": "worker_progress_run_ids",
                "run_ids": list(run_ids),
                "files": source_records,
            }
            artifact_files["qwen_caption_io_sources"] = write_text_artifact(
                output_dir / "qwen_caption_io_sources.json",
                json.dumps(source_meta, indent=2, sort_keys=True),
                max_bytes=max_bytes,
                source="qwen_caption_io_sources",
            )
        except Exception as exc:  # noqa: BLE001
            artifact_errors.append({
                "artifact": "qwen_caption_io_sources",
                "type": type(exc).__name__,
                "message": str(exc),
            })
    return artifact_files, artifact_errors


def write_heartbeat(output_root: Path, payload: Mapping[str, Any]) -> None:
    heartbeat = dict(payload)
    heartbeat["heartbeat_at"] = datetime.now(timezone.utc).isoformat()
    heartbeat["heartbeat_epoch"] = time.time()
    atomic_write_json(output_root / "heartbeat.json", heartbeat)


def _read_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _pid_is_alive(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError, OverflowError):
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class ArtifactRunnerLock:
    def __init__(
        self,
        output_root: Path,
        *,
        wait_timeout_seconds: float = 0.0,
        stale_seconds: float = RUNNER_LOCK_STALE_SECONDS,
        poll_seconds: float = RUNNER_LOCK_POLL_SECONDS,
    ) -> None:
        self.output_root = output_root
        self.lock_path = output_root / RUNNER_LOCK_NAME
        self.wait_timeout_seconds = max(0.0, float(wait_timeout_seconds or 0.0))
        self.stale_seconds = max(0.0, float(stale_seconds or 0.0))
        self.poll_seconds = max(0.05, float(poll_seconds or RUNNER_LOCK_POLL_SECONDS))
        self.runner_id = uuid.uuid4().hex
        self.started_epoch = time.time()
        self.acquired = False

    def _payload(self, phase: str, **fields: Any) -> dict[str, Any]:
        now = time.time()
        return {
            "runner_id": self.runner_id,
            "pid": os.getpid(),
            "output_dir": str(self.output_root),
            "phase": phase,
            "started_at": datetime.fromtimestamp(self.started_epoch, tz=timezone.utc).isoformat(),
            "started_epoch": self.started_epoch,
            "heartbeat_at": datetime.now(timezone.utc).isoformat(),
            "heartbeat_epoch": now,
            **fields,
        }

    def _try_create(self, phase: str) -> bool:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            fd = os.open(str(self.lock_path), flags, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(self._payload(phase), handle, indent=2, sort_keys=True)
        self.acquired = True
        return True

    def _existing_lock_can_takeover(self, payload: Mapping[str, Any]) -> bool:
        if not payload:
            return True
        raw_pid = payload.get("pid")
        has_pid = str(raw_pid or "").strip() != ""
        if has_pid:
            return not _pid_is_alive(raw_pid)
        if self.stale_seconds <= 0:
            return False
        try:
            heartbeat_epoch = float(payload.get("heartbeat_epoch") or 0.0)
        except (TypeError, ValueError, OverflowError):
            return True
        return heartbeat_epoch <= 0 or (time.time() - heartbeat_epoch) > self.stale_seconds

    def acquire(self) -> None:
        deadline = None
        if self.wait_timeout_seconds > 0:
            deadline = time.monotonic() + self.wait_timeout_seconds
        last_log_at = 0.0
        while True:
            if self._try_create("acquired"):
                return
            existing = _read_json_dict(self.lock_path)
            if self._existing_lock_can_takeover(existing):
                try:
                    self.lock_path.unlink()
                except FileNotFoundError:
                    pass
                except Exception as exc:  # noqa: BLE001
                    raise SystemExit(f"artifact_lock_remove_failed:{exc}") from exc
                continue
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                holder = existing.get("runner_id") or existing.get("pid") or "unknown"
                raise SystemExit(f"artifact_lock_active:{self.lock_path}:{holder}")
            if now - last_log_at >= self.poll_seconds:
                holder = existing.get("runner_id") or existing.get("pid") or "unknown"
                print(f"[artifact_lock_wait] {self.lock_path} held by {holder}", flush=True)
                last_log_at = now
            sleep_seconds = self.poll_seconds
            if deadline is not None:
                sleep_seconds = min(sleep_seconds, max(0.05, deadline - now))
            time.sleep(sleep_seconds)

    def refresh(self, phase: str, **fields: Any) -> None:
        if not self.acquired:
            return
        atomic_write_json(self.lock_path, self._payload(phase, **fields))

    def release(self) -> None:
        if not self.acquired:
            return
        current = _read_json_dict(self.lock_path)
        if str(current.get("runner_id") or "") == self.runner_id:
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass
        self.acquired = False


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _worker_progress_text_tail(value: Any, *, limit: int = 240) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[-limit:]


def _worker_progress_compact_step_plan(value: Any, *, limit: int = 80) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    compact: list[dict[str, str]] = []
    for index, raw_entry in enumerate(value[:limit]):
        if not isinstance(raw_entry, Mapping):
            continue
        step_id = str(raw_entry.get("id") or f"step_{index + 1}").strip() or f"step_{index + 1}"
        entry = {
            "id": step_id,
            "label": str(raw_entry.get("label") or step_id).strip() or step_id,
        }
        detail = str(raw_entry.get("detail") or "").strip()
        if detail:
            entry["detail"] = _worker_progress_text_tail(detail, limit=280)
        compact.append(entry)
    return compact


def _worker_progress_compact_io_events(value: Any, *, limit: int = 8) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    compact: list[dict[str, str]] = []
    for raw_event in value[-limit:]:
        if not isinstance(raw_event, Mapping):
            continue
        event = str(raw_event.get("event") or "event").strip() or "event"
        kind = str(raw_event.get("kind") or "").strip()
        title = str(raw_event.get("title") or event).strip() or event
        entry = {
            "event": event,
            "title": _worker_progress_text_tail(title, limit=240),
            "text": _worker_progress_text_tail(raw_event.get("text"), limit=2000),
        }
        if kind:
            entry["kind"] = kind
        run_id = str(raw_event.get("run_id") or "").strip()
        if run_id:
            entry["run_id"] = run_id
        compact.append(entry)
    return compact


def worker_progress_snapshot(snapshot: Mapping[str, Any], *, seq: int) -> dict[str, Any]:
    io_events = snapshot.get("io_events")
    last_io_event: dict[str, Any] | None = None
    if isinstance(io_events, list) and io_events:
        raw_event = io_events[-1]
        if isinstance(raw_event, Mapping):
            last_io_event = {
                "event": str(raw_event.get("event") or ""),
                "source": str(raw_event.get("source") or ""),
                "run_id": str(raw_event.get("run_id") or ""),
                "step_label": str(raw_event.get("step_label") or ""),
                "text_tail": _worker_progress_text_tail(raw_event.get("text")),
            }
    token_preview = str(snapshot.get("token_preview") or "")
    live_output = str(snapshot.get("live_output") or "")
    payload: dict[str, Any] = {
        "seq": int(seq),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "captured_epoch": time.time(),
        "active": bool(snapshot.get("active")),
        "run_id": str(snapshot.get("run_id") or ""),
        "phase": str(snapshot.get("phase") or ""),
        "phase_label": str(snapshot.get("phase_label") or ""),
        "progress": snapshot.get("progress"),
        "message": str(snapshot.get("message") or ""),
        "model_id": str(snapshot.get("model_id") or ""),
        "platform": str(snapshot.get("platform") or ""),
        "step_id": str(snapshot.get("step_id") or ""),
        "step_index": snapshot.get("step_index"),
        "step_total": snapshot.get("step_total"),
        "step_label": str(snapshot.get("step_label") or ""),
        "step_detail": str(snapshot.get("step_detail") or ""),
        "generated_tokens": snapshot.get("generated_tokens"),
        "max_new_tokens": snapshot.get("max_new_tokens"),
        "updated_at": snapshot.get("updated_at"),
        "token_preview_tail": _worker_progress_text_tail(token_preview),
        "token_preview_chars": len(token_preview),
        "live_output_chars": len(live_output),
        "io_event_count": len(io_events) if isinstance(io_events, list) else 0,
    }
    if last_io_event is not None:
        payload["last_io_event"] = last_io_event
    compact_step_plan = _worker_progress_compact_step_plan(snapshot.get("step_plan"))
    if compact_step_plan:
        payload["step_plan"] = compact_step_plan
    compact_io_events = _worker_progress_compact_io_events(io_events)
    if compact_io_events:
        payload["io_events"] = compact_io_events
    instruction_qa_accumulator = snapshot.get("instruction_qa_accumulator")
    if isinstance(instruction_qa_accumulator, Mapping):
        payload["instruction_qa_accumulator"] = dict(instruction_qa_accumulator)
    return payload


def start_worker_progress_mirror(
    *,
    api: Any,
    output_dir: Path,
    stop_event: threading.Event,
    poll_seconds: float = 0.5,
) -> threading.Thread:
    progress_json = output_dir / WORKER_PROGRESS_JSON
    progress_jsonl = output_dir / WORKER_PROGRESS_JSONL

    def _mirror() -> None:
        seq = 0
        last_fingerprint = ""
        while not stop_event.wait(max(0.05, float(poll_seconds or 0.5))):
            try:
                snapshot = api.qwen_progress()
            except Exception:
                continue
            if not isinstance(snapshot, Mapping):
                continue
            payload = worker_progress_snapshot(snapshot, seq=seq + 1)
            fingerprint_source = {
                key: payload.get(key)
                for key in (
                    "active",
                    "run_id",
                    "phase",
                    "message",
                    "step_id",
                    "step_index",
                    "generated_tokens",
                    "io_event_count",
                    "updated_at",
                    "token_preview_chars",
                    "live_output_chars",
                )
            }
            fingerprint = json.dumps(fingerprint_source, sort_keys=True, default=str)
            if fingerprint == last_fingerprint:
                continue
            seq += 1
            payload["seq"] = seq
            last_fingerprint = fingerprint
            atomic_write_json(progress_json, payload)
            append_jsonl(progress_jsonl, payload)

    thread = threading.Thread(
        target=_mirror,
        name="qwen-caption-worker-progress",
        daemon=True,
    )
    thread.start()
    return thread


def read_worker_progress(attempt_dir: Path, *, last_mtime: float = 0.0) -> tuple[float, dict[str, Any] | None]:
    progress_path = attempt_dir / WORKER_PROGRESS_JSON
    try:
        stat_result = progress_path.stat()
    except FileNotFoundError:
        return last_mtime, None
    except Exception:
        return last_mtime, None
    mtime = float(stat_result.st_mtime or 0.0)
    if mtime <= last_mtime:
        return last_mtime, None
    payload = _read_json_dict(progress_path)
    if not payload:
        return mtime, None
    return mtime, payload


class JsonlObjectRowsError(ValueError):
    def __init__(self, path: Path, errors: Sequence[Mapping[str, Any]], *, artifact_name: str):
        self.path = path
        self.errors = list(errors)
        self.artifact_name = artifact_name
        detail = "; ".join(
            f"line {error.get('line')}: {error.get('error')}"
            for error in self.errors[:5]
        )
        if len(self.errors) > 5:
            detail = f"{detail}; and {len(self.errors) - 5} more"
        super().__init__(detail or f"invalid {artifact_name}")


class ResultsJsonlError(JsonlObjectRowsError):
    def __init__(self, path: Path, errors: Sequence[Mapping[str, Any]]):
        super().__init__(path, errors, artifact_name="results.jsonl")


class CaptionsJsonlError(JsonlObjectRowsError):
    def __init__(self, path: Path, errors: Sequence[Mapping[str, Any]]):
        super().__init__(path, errors, artifact_name="captions.jsonl")


def read_jsonl_object_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows, []
    errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            errors.append(
                {
                    "line": line_number,
                    "error": str(exc),
                    "preview": stripped[:160],
                }
            )
            continue
        if not isinstance(row, dict):
            errors.append(
                {
                    "line": line_number,
                    "error": "JSONL row is not an object",
                    "preview": stripped[:160],
                }
            )
            continue
        rows.append(row)
    return rows, errors


def validate_captions_jsonl(captions_jsonl: Path) -> list[dict[str, Any]]:
    rows, errors = read_jsonl_object_rows(captions_jsonl)
    if errors:
        raise CaptionsJsonlError(captions_jsonl, errors)
    return rows


def load_latest_rows(results_jsonl: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    rows, errors = read_jsonl_object_rows(results_jsonl)
    if errors:
        raise ResultsJsonlError(results_jsonl, errors)
    for row in rows:
        key = str(row.get("case_id") or "").strip()
        if key:
            latest[key] = row
    return latest


def row_succeeded(row: Mapping[str, Any], *, ignore_quality_failures: bool) -> bool:
    if row.get("exit_code") != 0:
        return False
    if row.get("status") not in {"ok", "preview_only"}:
        return False
    return ignore_quality_failures or not row.get("quality_failures")


def row_has_recovery_events(row: Mapping[str, Any]) -> bool:
    events = row.get("recovery_events")
    return isinstance(events, list) and bool(events)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def row_generated_qa_warning(row: Mapping[str, Any]) -> bool:
    warning = row.get("generated_qa_warning") or row.get("instruction_qa_warning")
    if isinstance(warning, Mapping):
        return True
    status = str(row.get("instruction_qa_status") or "").strip().lower()
    if status in {"error", "underfilled", "empty", "instruction_qa_failed"}:
        return True
    target = _safe_int(row.get("generated_qa_target_pair_count"), 0)
    accepted = _safe_int(row.get("generated_qa_pair_count"), 0)
    return target > 0 and accepted < target


def row_generated_qa_error(row: Mapping[str, Any]) -> bool:
    status = str(row.get("instruction_qa_status") or "").strip().lower()
    if status in {"error", "instruction_qa_failed"}:
        return True
    warning = row.get("generated_qa_warning") or row.get("instruction_qa_warning")
    if isinstance(warning, Mapping):
        warning_type = str(warning.get("type") or "").strip()
        return warning_type == "InstructionQaGenerationWarning"
    return False


def write_summary(
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    row_limit: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    latest: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = str(row.get("case_id") or "").strip()
        if key:
            latest[key] = row
    ordered = list(latest.values())
    totals = Counter(str(row.get("final_status") or "unknown") for row in ordered)
    effective_limit = DEFAULT_SUMMARY_ROW_LIMIT if row_limit is None else int(row_limit)
    if effective_limit < 0:
        sampled_rows = ordered
        sample_policy = "all"
    elif effective_limit == 0:
        sampled_rows = []
        sample_policy = "none"
    else:
        sampled_rows = ordered[-effective_limit:]
        sample_policy = "latest"
    rows_truncated = len(sampled_rows) < len(ordered)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(ordered),
        "failed_cases": sum(
            count
            for status, count in totals.items()
            if status not in SUMMARY_OK_STATUSES
        ),
        "quality_failed_cases": sum(1 for row in ordered if row.get("quality_failures")),
        "generated_qa_warning_cases": sum(1 for row in ordered if row_generated_qa_warning(row)),
        "generated_qa_error_cases": sum(1 for row in ordered if row_generated_qa_error(row)),
        "generated_qa_underfilled_cases": sum(
            1
            for row in ordered
            if str(row.get("instruction_qa_status") or "").strip().lower() == "underfilled"
        ),
        "generated_qa_zero_pair_cases": sum(
            1
            for row in ordered
            if _safe_int(row.get("generated_qa_target_pair_count"), 0) > 0
            and _safe_int(row.get("generated_qa_pair_count"), 0) <= 0
        ),
        "totals": dict(totals),
        "row_count": len(ordered),
        "row_limit": effective_limit,
        "rows_truncated": rows_truncated,
        "rows_omitted": len(ordered) - len(sampled_rows),
        "rows_sample_policy": sample_policy if rows_truncated else "all",
        "rows": sampled_rows,
    }
    if extra:
        payload.update(dict(extra))
    atomic_write_json(output_root / "summary.json", payload)


def dataset_text_label_path(dataset_root: Path, image_path: Path) -> Path:
    try:
        rel = image_path.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        return dataset_root / "text_labels" / image_path.with_suffix(".txt").name
    parts = rel.parts
    if len(parts) >= 3 and parts[1] == "images":
        return dataset_root / parts[0] / "text_labels" / Path(*parts[2:]).with_suffix(".txt")
    if len(parts) >= 2 and parts[0] == "images":
        return dataset_root / "text_labels" / Path(*parts[1:]).with_suffix(".txt")
    return dataset_root / "text_labels" / rel.with_suffix(".txt")


def caption_exists_for_case(dataset_root: Path, case: Mapping[str, Any]) -> bool:
    path = dataset_text_label_path(dataset_root, Path(str(case.get("image_path") or "")))
    return path.exists() and bool(path.read_text().strip())


def save_caption_for_case(dataset_root: Path, case: Mapping[str, Any], caption: str) -> Path:
    path = dataset_text_label_path(dataset_root, Path(str(case.get("image_path") or "")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(caption or "").strip() + "\n")
    return path


def yolo_hints(label_path: Path, image_width: int, image_height: int, names: list[str]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    if not label_path.exists():
        return hints
    for index, line in enumerate(label_path.read_text().splitlines(), start=1):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_idx = int(float(parts[0]))
            xc, yc, width, height = [float(value) for value in parts[1:5]]
        except ValueError:
            continue
        label = names[cls_idx] if 0 <= cls_idx < len(names) else f"class_{cls_idx}"
        x1 = max(0.0, (xc - width / 2.0) * image_width)
        y1 = max(0.0, (yc - height / 2.0) * image_height)
        x2 = min(float(image_width), (xc + width / 2.0) * image_width)
        y2 = min(float(image_height), (yc + height / 2.0) * image_height)
        if x2 <= x1 or y2 <= y1:
            continue
        confidence = None
        if len(parts) >= 6:
            try:
                confidence = float(parts[5])
            except ValueError:
                confidence = None
        hints.append(
            {
                "label": class_name(label),
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "source_id": f"{label_path.stem}_{index:04d}",
            }
        )
    return hints


def image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _normalize_caption_provider(value: Any) -> str:
    provider = str(value or CAPTION_PROVIDER_LOCAL).strip().lower().replace("-", "_")
    if provider in {"qwen", "local", "localqwen", "local_qwen"}:
        return CAPTION_PROVIDER_LOCAL
    if provider in {"openai", "openai_api", "remote"}:
        return CAPTION_PROVIDER_OPENAI
    return CAPTION_PROVIDER_LOCAL


def _normalize_openai_image_detail(value: Any) -> str:
    detail = str(value or DEFAULT_OPENAI_IMAGE_DETAIL).strip().lower()
    return detail if detail in OPENAI_IMAGE_DETAIL_CHOICES else DEFAULT_OPENAI_IMAGE_DETAIL


def _resolve_openai_key_file(path_value: Any) -> Path | None:
    raw = str(path_value or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _extract_openai_response_text(data: Mapping[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    chunks: list[str] = []
    for item in data.get("output") if isinstance(data.get("output"), list) else []:
        if not isinstance(item, Mapping):
            continue
        for content in item.get("content") if isinstance(item.get("content"), list) else []:
            if not isinstance(content, Mapping):
                continue
            text = content.get("text") or content.get("output_text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(part for part in chunks if part).strip()


def _openai_relevant_headers(headers: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        items = list(headers.items())
    except Exception:
        items = []
    for raw_key, raw_value in items:
        key = str(raw_key or "").strip().lower()
        if not key:
            continue
        if (
            key.startswith("x-ratelimit-")
            or key in {"retry-after", "openai-processing-ms", "x-request-id"}
        ):
            out[key] = str(raw_value or "")
    return out


def _openai_error_payload(text: str) -> dict[str, Any]:
    try:
        data = json.loads(str(text or ""))
    except Exception:
        return {}
    error = data.get("error") if isinstance(data, Mapping) else None
    return dict(error) if isinstance(error, Mapping) else {}


def _openai_error_is_retryable(*, status_code: int, error_payload: Mapping[str, Any]) -> bool:
    code = str(error_payload.get("code") or "").strip().lower()
    error_type = str(error_payload.get("type") or "").strip().lower()
    if code in {"insufficient_quota", "billing_hard_limit_reached"}:
        return False
    if error_type in {"insufficient_quota"}:
        return False
    return status_code in {408, 409, 429, 500, 502, 503, 504}


class OpenAICaptionApiAdapter:
    """Provider adapter that preserves the local caption orchestration.

    The runner still calls localinferenceapi.qwen_caption and generated-QA
    helpers. Only the model-call primitives are swapped, so prompt preview,
    caption guards, QA top-ups, artifacts, resume behavior, and exported rows
    stay on the same code path as the local Qwen provider.
    """

    def __init__(self, local_api: Any, args: argparse.Namespace):
        self.local_api = local_api
        self.model = str(getattr(args, "openai_model", DEFAULT_OPENAI_MODEL) or DEFAULT_OPENAI_MODEL).strip()
        if not self.model:
            self.model = DEFAULT_OPENAI_MODEL
        self.image_detail = _normalize_openai_image_detail(getattr(args, "openai_image_detail", DEFAULT_OPENAI_IMAGE_DETAIL))
        reasoning_effort = str(
            getattr(args, "openai_reasoning_effort", DEFAULT_OPENAI_REASONING_EFFORT)
            or DEFAULT_OPENAI_REASONING_EFFORT
        ).strip().lower()
        self.reasoning_effort = (
            reasoning_effort
            if reasoning_effort in OPENAI_REASONING_EFFORT_CHOICES
            else DEFAULT_OPENAI_REASONING_EFFORT
        )
        self.api_key_file = _resolve_openai_key_file(getattr(args, "openai_api_key_file", DEFAULT_OPENAI_API_KEY_FILE))
        try:
            timeout = float(getattr(args, "openai_timeout", DEFAULT_OPENAI_TIMEOUT_SECONDS) or DEFAULT_OPENAI_TIMEOUT_SECONDS)
        except (TypeError, ValueError, OverflowError):
            timeout = DEFAULT_OPENAI_TIMEOUT_SECONDS
        self.timeout = max(5.0, min(timeout, 600.0))
        try:
            retries = int(getattr(args, "openai_max_retries", DEFAULT_OPENAI_MAX_RETRIES) or DEFAULT_OPENAI_MAX_RETRIES)
        except (TypeError, ValueError, OverflowError):
            retries = DEFAULT_OPENAI_MAX_RETRIES
        self.max_retries = max(0, min(retries, 12))
        self._progress_lock = threading.Lock()
        self._progress: dict[str, Any] = {
            "active": False,
            "phase": "openai_caption_provider",
            "step_label": "OpenAI provider idle",
            "step_detail": f"model={self.model}, image_detail={self.image_detail}",
            "model_id": self.model,
            "progress": 0.0,
            "io_events": [],
        }

    def _api_key(self) -> str:
        env_key = str(os.environ.get("OPENAI_API_KEY") or "").strip()
        if env_key:
            return env_key
        if self.api_key_file and self.api_key_file.exists():
            key = self.api_key_file.read_text().strip()
            if key:
                return key
        raise RuntimeError(
            "openai_api_key_not_configured: set OPENAI_API_KEY or configure an existing OpenAI API key file"
        )

    def _set_progress(self, *, active: bool, step_label: str, step_detail: str = "", live_text: str = "", event: Mapping[str, Any] | None = None) -> None:
        with self._progress_lock:
            events = list(self._progress.get("io_events") or [])
            if event:
                events.append(dict(event))
                events = events[-8:]
            self._progress.update(
                {
                    "active": active,
                    "phase": "openai_caption_provider",
                    "step_label": step_label,
                    "step_detail": step_detail,
                    "model_id": self.model,
                    "progress": 0.5 if active else 1.0,
                    "live_text": live_text,
                    "io_events": events,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    def qwen_progress(self) -> dict[str, Any]:
        with self._progress_lock:
            snapshot = dict(self._progress)
            snapshot["io_events"] = list(self._progress.get("io_events") or [])
        snapshot["loaded"] = True
        snapshot["memory"] = {}
        snapshot["vram"] = {}
        return snapshot

    def qwen_caption_prompt_preview(self, payload: Any) -> Any:
        return self.local_api.qwen_caption_prompt_preview(payload)

    def qwen_caption(self, payload: Any) -> Any:
        original_visual = self.local_api._run_qwen_inference
        original_text = self.local_api._run_qwen_text_inference
        self.local_api._run_qwen_inference = self._run_qwen_inference
        self.local_api._run_qwen_text_inference = self._run_qwen_text_inference
        try:
            return self.local_api.qwen_caption(payload)
        finally:
            self.local_api._run_qwen_inference = original_visual
            self.local_api._run_qwen_text_inference = original_text
            self._set_progress(active=False, step_label="OpenAI caption call complete")

    def _pil_image_data_url(self, pil_img: Image.Image) -> str:
        buf = io.BytesIO()
        image = pil_img.convert("RGB") if pil_img.mode not in {"RGB", "L"} else pil_img
        image.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

    def _request_responses_api(
        self,
        *,
        prompt: str,
        system_prompt: str = "",
        pil_img: Image.Image | None = None,
        max_new_tokens: int | None = None,
        call_kind: str,
    ) -> str:
        content: list[dict[str, Any]] = []
        text_parts = []
        if system_prompt:
            text_parts.append(system_prompt)
        text_parts.append(prompt)
        content.append({"type": "input_text", "text": "\n\n".join(part for part in text_parts if part)})
        if pil_img is not None:
            content.append(
                {
                    "type": "input_image",
                    "image_url": self._pil_image_data_url(pil_img),
                    "detail": self.image_detail,
                }
            )
        body: dict[str, Any] = {
            "model": self.model,
            "input": [{"role": "user", "content": content}],
            "store": False,
            "reasoning": {"effort": self.reasoning_effort},
        }
        if max_new_tokens is not None:
            body["max_output_tokens"] = max(1, int(max_new_tokens))
        payload = json.dumps(body).encode("utf-8")
        self._set_progress(
            active=True,
            step_label="Waiting for OpenAI output",
            step_detail=f"{call_kind}; model={self.model}; detail={self.image_detail}",
            event={
                "kind": "prompt",
                "title": f"OpenAI {call_kind} request",
                "text": f"model={self.model}\nreasoning_effort={self.reasoning_effort}\ndetail={self.image_detail}\nmax_output_tokens={body.get('max_output_tokens', 'auto')}\nprompt_chars={len(prompt)}",
            },
        )
        response_body = ""
        for request_attempt in range(1, self.max_retries + 2):
            req = urllib.request.Request(
                "https://api.openai.com/v1/responses",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key()}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    response_headers = _openai_relevant_headers(getattr(resp, "headers", None))
                    response_body = resp.read().decode("utf-8", errors="replace")
                if response_headers:
                    self._set_progress(
                        active=True,
                        step_label="OpenAI response headers received",
                        step_detail=f"{call_kind}; {json.dumps(response_headers, sort_keys=True)}",
                    )
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                error_payload = _openai_error_payload(detail)
                headers = _openai_relevant_headers(exc.headers)
                retryable = _openai_error_is_retryable(
                    status_code=int(exc.code),
                    error_payload=error_payload,
                )
                should_retry = retryable and request_attempt <= self.max_retries
                if not should_retry:
                    diagnostic = {
                        "status_code": int(exc.code),
                        "error": error_payload,
                        "headers": headers,
                        "retryable": retryable,
                    }
                    raise RuntimeError(
                        f"openai_responses_http_error:{exc.code}:{json.dumps(diagnostic, sort_keys=True)}:{detail}"
                    ) from exc
                retry_after = 0.0
                try:
                    retry_after = float(exc.headers.get("Retry-After") or 0.0)
                except (TypeError, ValueError, OverflowError):
                    retry_after = 0.0
                delay = max(retry_after, min(60.0, 1.5 * (2 ** (request_attempt - 1)) + random.random()))
                self._set_progress(
                    active=True,
                    step_label="Retrying OpenAI output",
                    step_detail=(
                        f"{call_kind}; HTTP {exc.code}; retry {request_attempt}/{self.max_retries}; "
                        f"sleeping {delay:.1f}s; headers={json.dumps(headers, sort_keys=True)}"
                    ),
                )
                time.sleep(delay)
            except urllib.error.URLError as exc:
                should_retry = request_attempt <= self.max_retries
                if not should_retry:
                    raise RuntimeError(f"openai_responses_request_error:{exc.reason}") from exc
                delay = min(60.0, 1.5 * (2 ** (request_attempt - 1)) + random.random())
                self._set_progress(
                    active=True,
                    step_label="Retrying OpenAI output",
                    step_detail=f"{call_kind}; network error; retry {request_attempt}/{self.max_retries}; sleeping {delay:.1f}s",
                )
                time.sleep(delay)
        data = json.loads(response_body)
        text = _extract_openai_response_text(data)
        if not text:
            raise RuntimeError("openai_responses_empty_output")
        self._set_progress(
            active=False,
            step_label="OpenAI output received",
            live_text=text,
            event={
                "kind": "output",
                "title": f"OpenAI {call_kind} output",
                "text": text,
            },
        )
        return text

    def _run_qwen_inference(
        self,
        prompt: str,
        pil_img: Image.Image,
        max_new_tokens: int | None = None,
        system_prompt_override: str | None = None,
        model_id_override: str | None = None,
        runtime_override: Any = None,
        decode_override: Mapping[str, Any] | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[str, int, int]:
        del model_id_override, runtime_override, decode_override, chat_template_kwargs
        call_id = uuid.uuid4().hex[:12]
        if hasattr(self.local_api, "_qwen_caption_io_input"):
            self.local_api._qwen_caption_io_input(
                call_id=call_id,
                source="openai_inference",
                runtime_platform="openai_responses",
                model_id=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": "<base64 image omitted>", "detail": self.image_detail},
                        ],
                    }
                ],
                prompt_text=prompt,
                system_prompt=str(system_prompt_override or ""),
                user_prompt=prompt,
                image_width=int(pil_img.width),
                image_height=int(pil_img.height),
                max_new_tokens=max_new_tokens,
            )
        output_text = self._request_responses_api(
            prompt=prompt,
            system_prompt=str(system_prompt_override or ""),
            pil_img=pil_img,
            max_new_tokens=max_new_tokens,
            call_kind="visual",
        )
        if hasattr(self.local_api, "_qwen_caption_io_output"):
            self.local_api._qwen_caption_io_output(
                call_id=call_id,
                source="openai_inference",
                runtime_platform="openai_responses",
                model_id=self.model,
                output_text=output_text,
                loop_detected=False,
                degenerate_reason=None,
            )
        return output_text, int(pil_img.width), int(pil_img.height)

    def _run_qwen_text_inference(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        system_prompt_override: str | None = None,
        model_id_override: str | None = None,
        runtime_override: Any = None,
        decode_override: Mapping[str, Any] | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[str, int, int]:
        del model_id_override, runtime_override, decode_override, chat_template_kwargs
        call_id = uuid.uuid4().hex[:12]
        if hasattr(self.local_api, "_qwen_caption_io_input"):
            self.local_api._qwen_caption_io_input(
                call_id=call_id,
                source="openai_text_inference",
                runtime_platform="openai_responses",
                model_id=self.model,
                messages=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                prompt_text=prompt,
                system_prompt=str(system_prompt_override or ""),
                user_prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
        output_text = self._request_responses_api(
            prompt=prompt,
            system_prompt=str(system_prompt_override or ""),
            pil_img=None,
            max_new_tokens=max_new_tokens,
            call_kind="text",
        )
        if hasattr(self.local_api, "_qwen_caption_io_output"):
            self.local_api._qwen_caption_io_output(
                call_id=call_id,
                source="openai_text_inference",
                runtime_platform="openai_responses",
                model_id=self.model,
                output_text=output_text,
                loop_detected=False,
                degenerate_reason=None,
            )
        return output_text, 0, 0


def caption_api_for_args(local_api: Any, args: argparse.Namespace) -> Any:
    provider = _normalize_caption_provider(getattr(args, "caption_provider", CAPTION_PROVIDER_LOCAL))
    if provider == CAPTION_PROVIDER_OPENAI:
        return OpenAICaptionApiAdapter(local_api, args)
    return local_api


def load_request_template(request_json: Any) -> dict[str, Any]:
    if not request_json:
        return {}
    request_path = Path(request_json)
    if not request_path.exists():
        raise FileNotFoundError(f"request template not found: {request_path}")
    raw_template = json.loads(request_path.read_text())
    if not isinstance(raw_template, dict):
        raise ValueError("request template must be a JSON object")
    template = dict(raw_template)
    for image_specific_key in IMAGE_SPECIFIC_REQUEST_KEYS:
        template.pop(image_specific_key, None)
    return template


def run_settings_payload(
    args: argparse.Namespace,
    *,
    request_template: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    template = dict(request_template) if request_template is not None else load_request_template(getattr(args, "request_json", None))
    arg_settings = {
        key: getattr(args, key, default)
        for key, default in RUN_SETTINGS_ARG_DEFAULTS.items()
    }
    payload = {
        "schema_version": RUN_SETTINGS_SCHEMA_VERSION,
        "caption_args": arg_settings,
        "request_template": template,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": RUN_SETTINGS_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        **payload,
    }


def manifest_run_settings_status(
    manifest: Mapping[str, Any],
    current_settings: Mapping[str, Any],
) -> tuple[str, str]:
    previous = manifest.get("run_settings") if isinstance(manifest.get("run_settings"), Mapping) else None
    if previous is None:
        return "legacy", "existing manifest has no run settings fingerprint"
    previous_fingerprint = str(previous.get("fingerprint") or "").strip()
    current_fingerprint = str(current_settings.get("fingerprint") or "").strip()
    if previous_fingerprint and current_fingerprint and previous_fingerprint == current_fingerprint:
        return "ok", "run settings fingerprint matches existing manifest"
    return "mismatch", "existing manifest run settings do not match requested caption settings"


def case_payload(case: dict[str, Any], dataset_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    names = load_labelmap(dataset_root)
    image_path = Path(case["image_path"])
    label_path = Path(case["label_path"])
    with Image.open(image_path) as image:
        width, height = image.size
    base_payload = {
        "image_base64": image_base64(image_path),
        "image_name": str(case.get("image_name") or image_path.name),
        "user_prompt": args.prompt,
        "label_hints": yolo_hints(label_path, width, height, names),
        "image_width": width,
        "image_height": height,
        "include_counts": True,
        "include_coords": True,
        "max_boxes": args.max_boxes,
        "max_new_tokens": args.max_new_tokens,
        "model_id": args.model_id,
        "model_variant": "Instruct",
        "refinement_model_id": args.refinement_model_id,
        "caption_loop_recovery_mode": args.loop_recovery,
        "caption_fallback_model_id": args.fallback_model_id,
        "caption_loop_cooldown": True,
        "caption_mode": case["caption_mode"],
        "caption_windowed_full_image_strategy": getattr(args, "windowed_full_image_strategy", "visual") or "visual",
        "caption_all_windows": True if case["caption_mode"] == "windowed" else None,
        "window_size": args.window_size,
        "window_overlap": args.window_overlap,
        "final_caption_max_sentences": args.final_sentences,
        "caption_window_min_sentences": 1,
        "caption_window_max_sentences": 3,
        "restrict_to_labels": True,
        "unload_others": False,
        "force_unload": True,
        "multi_model_cache": False,
        "fast_mode": False,
        "use_sampling": args.use_sampling,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    template = load_request_template(getattr(args, "request_json", None))
    return {**base_payload, **template, **{key: base_payload[key] for key in (
        "image_base64",
        "image_name",
        "label_hints",
        "image_width",
        "image_height",
    )}}


def summarize_caption(
    text: str,
    expected_counts: dict[str, int],
    glossary_map: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    lowered = text.lower()
    missing_labels = [
        label
        for label in expected_counts
        if not quality_label_present(lowered, label, glossary_map)
    ]
    missing_counts = [
        f"{count} {label}"
        for label, count in expected_counts.items()
        if count > 0 and str(count) not in lowered
    ]
    words = [word.strip(".,;:!?()[]{}\"'").lower() for word in text.split()]
    repeated_runs = 0
    last = None
    run = 0
    for word in words:
        if word and word == last:
            run += 1
        else:
            run = 1
        last = word
        repeated_runs = max(repeated_runs, run)
    raw_inventory_sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if re.fullmatch(r"(?:[A-Za-z][A-Za-z ]*:\s*\d+\s*,?\s*)+[.!]?", sentence.strip())
    ]
    count_contradictions: list[str] = []
    for label, count in expected_counts.items():
        if count <= 0:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if quality_negative_mentions_label(sentence, label, glossary_map):
                count_contradictions.append(sentence.strip())
                break
    unsupported_specific_terms = []
    allowed_compact = {
        str(label or "").lower().replace(" ", "")
        for label in expected_counts
    }
    for canonical, variants in UNSUPPORTED_SPECIFIC_TERMS.items():
        if canonical.lower().replace(" ", "") in allowed_compact:
            continue
        if any(re.search(rf"\b{re.escape(variant)}\b", lowered) for variant in variants):
            unsupported_specific_terms.append(canonical)
    return {
        "characters": len(text),
        "sentences": sum(text.count(mark) for mark in ".!?"),
        "missing_label_mentions": missing_labels,
        "missing_count_digits": missing_counts,
        "raw_count_inventory_sentences": raw_inventory_sentences,
        "count_contradiction_sentences": count_contradictions,
        "unsupported_specific_terms": unsupported_specific_terms,
        "max_same_word_run": repeated_runs,
        "contains_prompt_leak": any(
            needle in lowered
            for needle in (
                "bounding box",
                "bbox",
                "label hint",
                "coordinates",
                "counts were provided",
                "provided counts",
            )
        ),
        "contains_internal_underscore": "_" in text,
    }


def caption_quality_failures(quality: Any) -> list[str]:
    if not isinstance(quality, dict):
        return []
    failures: list[str] = []
    list_fields = {
        "missing_label_mentions": "missing labels",
        "missing_count_digits": "missing counts",
        "raw_count_inventory_sentences": "raw count inventory",
        "count_contradiction_sentences": "count contradiction",
        "unsupported_specific_terms": "unsupported terms",
    }
    for field, label in list_fields.items():
        values = quality.get(field)
        if isinstance(values, list) and values:
            failures.append(f"{label}: {', '.join(str(value) for value in values[:3])}")
    if quality.get("contains_prompt_leak"):
        failures.append("prompt leak")
    if quality.get("contains_internal_underscore"):
        failures.append("internal underscore")
    try:
        same_word_run = int(quality.get("max_same_word_run") or 0)
    except (TypeError, ValueError):
        same_word_run = 0
    if same_word_run >= 8:
        failures.append(f"same word repeated {same_word_run} times")
    return failures


def _fallback_label_term(label: str, count: int) -> str:
    term = re.sub(r"[_\s]+", " ", str(label or "").strip())
    if not term:
        return "object" if count == 1 else "objects"
    words = [
        word if (len(word) == 1 and word.isupper()) else word.lower()
        for word in term.split()
    ]
    if count == 1:
        return " ".join(words)
    last = words[-1]
    lower = last.lower()
    if lower == "person":
        words[-1] = "people"
    elif lower.endswith("s"):
        words[-1] = last
    elif lower.endswith(("x", "ch", "sh")):
        words[-1] = f"{last}es"
    elif lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
        words[-1] = f"{last[:-1]}ies"
    else:
        words[-1] = f"{last}s"
    return " ".join(words)


def _join_count_phrases(phrases: Sequence[str]) -> str:
    cleaned = [str(phrase).strip() for phrase in phrases if str(phrase).strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def parent_deterministic_recovery_caption(case: Mapping[str, Any]) -> tuple[str, dict[str, int]] | None:
    counts: dict[str, int] = {}
    for raw_label, raw_count in _case_class_counts(case).items():
        label = str(raw_label or "").strip()
        if not label:
            continue
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError, OverflowError):
            continue
        if count > 0:
            counts[label] = count
    if not counts:
        return None
    phrases = [
        f"{count} {_fallback_label_term(label, count)}"
        for label, count in sorted(counts.items(), key=lambda item: item[0].lower())
    ]
    caption = (
        f"From a high angle, the scene contains {_join_count_phrases(phrases)} "
        "arranged across the image."
    )
    return caption, counts


def parent_deterministic_recovery_eligibility(row: Mapping[str, Any]) -> tuple[bool, str]:
    failure_kind = str(row.get("attempt_failure_kind") or "").strip()
    status = str(row.get("status") or "").strip()
    exception = row.get("exception") if isinstance(row.get("exception"), Mapping) else {}
    exception_type = str(exception.get("type") or "").strip()
    exception_message = str(exception.get("message") or "").strip().lower()
    nonrecoverable_exception_types = {
        "AttributeError",
        "ImportError",
        "KeyError",
        "ModuleNotFoundError",
        "SyntaxError",
        "TypeError",
        "ValidationError",
        "ValueError",
    }
    if exception_type in nonrecoverable_exception_types:
        return False, f"nonrecoverable_exception:{exception_type}"
    if "validation error for qwencaptionrequest" in exception_message:
        return False, "nonrecoverable_request_validation"
    if failure_kind in {"signal_exit", "timeout"}:
        return True, failure_kind
    if failure_kind == "nonzero_exit" and exception_type in {"RuntimeError", "LoopDetectedError"}:
        return True, f"{failure_kind}:{exception_type}"
    qwen_io = row.get("qwen_caption_io") if isinstance(row.get("qwen_caption_io"), Mapping) else {}
    try:
        loop_events = int(qwen_io.get("loop_guard_events") or 0) + int(qwen_io.get("loop_trim_events") or 0)
    except (TypeError, ValueError, OverflowError):
        loop_events = 0
    if loop_events > 0:
        return True, "loop_detected"
    if status in {"loop_detected", "stream_loop_detected"}:
        return True, status
    if failure_kind in {"worker_error"}:
        return True, failure_kind
    return False, f"ineligible_failure:{failure_kind or status or 'unknown'}"


def _qa_json_candidate_texts(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    candidates = [raw]
    seen = {raw}

    def add_candidate(value: str) -> None:
        value = str(value or "").strip()
        if value and value not in seen:
            seen.add(value)
            candidates.append(value)

    def repair_extra_quote_before_key(value: str) -> str:
        # Qwen often emits otherwise-valid QA JSON with an extra quote before
        # optional metadata keys, e.g. `,""row_type":...`. Salvage that without
        # changing quoted question or answer content.
        return re.sub(
            r'(?<=[{,])\s*""(?=[A-Za-z_][A-Za-z0-9_]*"\s*:)',
            '"',
            value,
        )

    def repair_missing_key_close_quote(value: str) -> str:
        # Larger thinking/abliterated models sometimes emit `"answer:"Yes`
        # instead of `"answer":"Yes`. Repair object-key quotes only at member
        # boundaries so quoted question/answer content is left alone.
        return re.sub(
            r'(?<=[{,])(\s*)"([A-Za-z_][A-Za-z0-9_]*)\s*:(?=\s*(?:"|\{|\[|-?\d|true\b|false\b|null\b))',
            r'\1"\2":',
            value,
        )

    def repair_trailing_commas(value: str) -> str:
        return re.sub(r",\s*([}\]])", r"\1", value)

    repairs = (
        repair_extra_quote_before_key,
        repair_missing_key_close_quote,
        repair_trailing_commas,
    )
    index = 0
    while index < len(candidates):
        current = candidates[index]
        for repair in repairs:
            repaired = repair(current)
            if repaired != current:
                add_candidate(repaired)
        index += 1
    return candidates


def _extract_json_payload(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        return None
    for candidate in _qa_json_candidate_texts(raw):
        try:
            return json.loads(candidate)
        except Exception:
            pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        for candidate in _qa_json_candidate_texts(fenced.group(1)):
            try:
                return json.loads(candidate)
            except Exception:
                pass
    starts = [idx for idx in (raw.find("{"), raw.find("[")) if idx >= 0]
    if not starts:
        return None
    start = min(starts)
    end = max(raw.rfind("}"), raw.rfind("]"))
    if end <= start:
        return None
    for candidate in _qa_json_candidate_texts(raw[start : end + 1]):
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None


def _extract_partial_qa_payload(text: str) -> dict[str, list[dict[str, Any]]] | None:
    candidates = _qa_json_candidate_texts(text)
    if not candidates:
        return None
    raw = candidates[-1]
    match = re.search(r'"qa_pairs"\s*:\s*\[', raw)
    if match:
        position = match.end()
    else:
        array_start = raw.find("[")
        if array_start < 0:
            return None
        position = array_start + 1
    decoder = json.JSONDecoder()
    pairs: list[dict[str, Any]] = []
    while position < len(raw):
        while position < len(raw) and raw[position] in " \t\r\n,":
            position += 1
        if position >= len(raw) or raw[position] == "]":
            break
        if raw[position] != "{":
            break
        try:
            item, end_position = decoder.raw_decode(raw, position)
        except json.JSONDecodeError:
            break
        if isinstance(item, dict):
            pairs.append(item)
        position = end_position
    return {"qa_pairs": pairs} if pairs else None


def _normalize_generated_qa_pairs(
    payload: Any,
    *,
    requested: int,
    case: Mapping[str, Any],
    answer_format: str = "natural",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(payload, Mapping):
        raw_pairs = payload.get("qa_pairs")
        if raw_pairs is None:
            raw_pairs = payload.get("questions")
    else:
        raw_pairs = payload
    if not isinstance(raw_pairs, list):
        return [], [{"reason": "qa_json_missing_list"}]
    pairs: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    case_id = case_key(case)
    for index, raw_pair in enumerate(raw_pairs, start=1):
        if len(pairs) >= requested:
            break
        if not isinstance(raw_pair, Mapping):
            rejections.append({"index": index, "reason": "pair_not_object"})
            continue
        question = re.sub(r"\s+", " ", str(raw_pair.get("question") or "").strip())
        raw_answer = raw_pair.get("answer")
        if isinstance(raw_answer, (dict, list)):
            answer = json.dumps(raw_answer, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        else:
            answer = re.sub(r"\s+", " ", str(raw_answer or "").strip())
        if not question or not answer:
            rejections.append({"index": index, "reason": "missing_question_or_answer"})
            continue
        question_key = question.lower()
        if question_key in seen_questions:
            rejections.append({"index": index, "reason": "duplicate_question", "question": question})
            continue
        seen_questions.add(question_key)
        row_answer_format = str(raw_pair.get("answer_format") or answer_format or "natural").strip().lower()
        if row_answer_format not in {"natural", "json"}:
            row_answer_format = "natural"
        if row_answer_format == "json":
            try:
                json.loads(answer)
            except Exception:
                rejections.append({"index": index, "reason": "invalid_json_answer", "question": question})
                continue
        pairs.append(
            {
                "qa_id": f"{case_id}__generated_qa_{len(pairs) + 1:04d}",
                "question": question,
                "answer": answer,
                "row_type": str(raw_pair.get("row_type") or "generated_qa").strip() or "generated_qa",
                "answer_format": row_answer_format,
                "answer_source": "vlm_generated",
                "validated_against": ["image", "language_annotations.caption0", "source_annotations"],
                "validation_status": "candidate",
                "review_status": "unreviewed",
                "metadata": {
                    "case_id": case_id,
                    "generator": "qwen_caption_instruction_pass",
                },
                "validation": {
                    "status": "candidate",
                    "strict_grounding_checked_by_prompt": True,
                },
            }
        )
    return pairs, rejections


def _instruction_qa_max_new_tokens(args: argparse.Namespace, requested: int) -> int:
    explicit = getattr(args, "instruction_max_new_tokens", None)
    if explicit is not None:
        try:
            parsed = int(explicit)
        except (TypeError, ValueError, OverflowError):
            parsed = 0
        if parsed > 0:
            return max(128, min(parsed, 8192))
    return max(512, min(4096, 256 + max(1, requested) * 192))


def _case_source_annotation_context(case: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    counts = _case_canonical_class_counts(case, args)
    return {
        "source": "dataset_label_counts",
        "label_count": _case_label_count(case),
        "object_counts": counts,
        "note": "These counts are read-only source annotations; generated QA must not change them.",
    }


def _normalize_instruction_qa_imposed_questions(raw: Any) -> list[str]:
    if raw is None:
        return []
    values: list[Any]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            values = parsed
        else:
            values = text.splitlines()
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        values = list(raw)
    else:
        values = [raw]
    questions: list[str] = []
    seen: set[str] = set()
    for value in values:
        question = re.sub(r"\s+", " ", str(value or "").strip())
        question = question.strip(" \t\r\n\"'`,;")
        if not question:
            continue
        if not question.endswith("?"):
            question = f"{question}?"
        key = question.lower()
        if key in seen:
            continue
        seen.add(key)
        questions.append(question[:500])
        if len(questions) >= 20:
            break
    return questions


def _instruction_qa_imposed_questions(args: argparse.Namespace) -> list[str]:
    raw = getattr(args, "instruction_qa_imposed_questions", None)
    if raw:
        return _normalize_instruction_qa_imposed_questions(raw)
    return _normalize_instruction_qa_imposed_questions(
        _request_template_value(args, "instruction_qa_imposed_questions", None)
    )


def _instruction_qa_requested_count(args: argparse.Namespace) -> int:
    try:
        requested = int(getattr(args, "subcaptions_per_image", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        requested = 0
    imposed = _instruction_qa_imposed_questions(args)
    return max(0, min(max(requested, len(imposed)), 20))


def _generated_qa_required_for_worker_success(args: argparse.Namespace) -> bool:
    if not bool(getattr(args, "instruction_dataset", False)):
        return False
    if _instruction_qa_requested_count(args) <= 0:
        return False
    include_generated = bool(getattr(args, "include_generated_qa_in_training", True))
    include_caption0 = bool(getattr(args, "include_caption0_in_training", True))
    return include_generated and not include_caption0


def _generated_qa_blocks_parent_deterministic_recovery(args: argparse.Namespace) -> bool:
    if not bool(getattr(args, "instruction_dataset", False)):
        return False
    if _instruction_qa_requested_count(args) <= 0:
        return False
    return bool(getattr(args, "include_generated_qa_in_training", True))


def build_instruction_qa_prompt(
    case: Mapping[str, Any],
    caption: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    imposed_questions = _instruction_qa_imposed_questions(args)
    requested = _instruction_qa_requested_count(args)
    glossary_context = _case_glossary_context(case, args)
    source_context = (
        _case_source_annotation_context(case, args)
        if bool(getattr(args, "include_source_annotations_in_generator_context", True))
        else {"source": "disabled_by_request"}
    )
    restrict_speculative_language = _instruction_qa_restrict_speculative_language(args)
    speculation_text = _instruction_qa_speculation_policy_text(restrict_speculative_language)
    strict_text = (
        "Use strict grounding: each answer must be supported by the image, the completed caption, or the read-only source annotation context."
        if bool(getattr(args, "strict_grounding", True))
        else "Prefer visual grounding and avoid unsupported facts."
    )
    qa_mix = str(getattr(args, "qa_mix", "balanced") or "balanced").strip().lower()
    if qa_mix not in {"balanced", "scene", "object", "caption"}:
        qa_mix = "balanced"
    mix_text = {
        "balanced": "Use a balanced mix of scene-level, object-focused, and caption-rephrasing questions.",
        "scene": "Prefer scene-level questions about visible setting, layout, and conditions.",
        "object": "Prefer object-focused questions whose answers are grounded in visible objects and read-only context.",
        "caption": "Prefer caption-rephrasing questions that produce grounded alternate descriptions.",
    }[qa_mix]
    answer_format = str(getattr(args, "answer_format", "natural") or "natural").strip().lower()
    if answer_format not in {"natural", "json"}:
        answer_format = "natural"
    answer_shape = (
        '{"qa_pairs":[{"question":"...","answer":{"answer":"..."},"row_type":"generated_qa","answer_format":"json"}]}'
        if answer_format == "json"
        else '{"qa_pairs":[{"question":"...","answer":"...","row_type":"generated_qa","answer_format":"natural"}]}'
    )
    format_text = (
        'Answers must be valid JSON strings. For general visual QA, use {"answer":"..."}; '
        "do not invent structured count or class-list JSON."
        if answer_format == "json"
        else "Answers must be concise natural-language facts, not JSON strings."
    )
    imposed_text = ""
    if imposed_questions:
        imposed_answer_policy = (
            "If a required question cannot be answered directly from the image, completed caption, glossary context, or read-only source annotations, omit it rather than inventing an answer or using unavailable-information language. "
            if restrict_speculative_language
            else "If a required question cannot be answered from the image, completed caption, glossary context, or read-only source annotations, answer that it is not visible or cannot be determined instead of inventing an answer. "
        )
        imposed_text = (
            "Required user questions:\n"
            f"{json.dumps(imposed_questions, ensure_ascii=False)}\n"
            "Answer these questions first and keep each question text exactly as provided when the answer is grounded. "
            f"{imposed_answer_policy}"
            "After grounded required questions, add generated questions only if more rows are needed.\n"
        )
    prompt = (
        f"Create up to {requested} diverse visual instruction question/answer pairs for this image.\n"
        f"Return only valid JSON with this shape: {answer_shape}.\n"
        "Questions must be image-specific and useful for training a vision-language model.\n"
        f"{imposed_text}"
        f"{mix_text}\n"
        f"{format_text}\n"
        "Do not mention prompts, labels, bounding boxes, coordinates, source annotations, or that counts were provided.\n"
        "When referring to labeled classes, use the canonical object terms from the glossary context. "
        "If a class has no glossary entry, use the natural English class term from the context. "
        "Never output raw labelmap names or odd internal spellings from the labelmap.\n"
        "Do not ask about an object that is absent or only implied by missing labels.\n"
        f"{speculation_text}\n"
        f"{strict_text}\n\n"
        f"Completed broad caption:\n{caption}\n\n"
        f"Glossary context:\n{json.dumps(glossary_context, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Read-only source annotation context:\n{json.dumps(source_context, ensure_ascii=False, sort_keys=True)}"
    )
    system_prompt = (
        "You generate grounded image question/answer training rows. "
        "Use the image as truth and return only the requested JSON."
    )
    max_new_tokens = _instruction_qa_max_new_tokens(args, requested)
    return {
        "requested": requested,
        "source_context": source_context,
        "glossary_context": glossary_context,
        "imposed_questions": imposed_questions,
        "qa_mix": qa_mix,
        "answer_format": answer_format,
        "restrict_speculative_language": restrict_speculative_language,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
    }


def _instruction_qa_recovery_prompt(
    *,
    case: Mapping[str, Any],
    caption: str,
    requested: int,
    answer_format: str,
    source_context: Mapping[str, Any],
    glossary_context: Mapping[str, Any],
    imposed_questions: Sequence[str] | None = None,
    restrict_speculative_language: bool = False,
) -> str:
    caption_text = re.sub(r"\s+", " ", str(caption or "").strip())
    if len(caption_text) > 2400:
        caption_text = caption_text[:2400].rsplit(" ", 1)[0].strip() + "..."
    answer_shape = (
        '{"qa_pairs":[{"question":"...","answer":{"answer":"..."},"row_type":"generated_qa","answer_format":"json"}]}'
        if answer_format == "json"
        else '{"qa_pairs":[{"question":"...","answer":"...","row_type":"generated_qa","answer_format":"natural"}]}'
    )
    format_text = (
        "Each answer must be a JSON object encoded as a JSON value."
        if answer_format == "json"
        else "Each answer must be one concise natural-language sentence."
    )
    count_text = json.dumps(source_context.get("object_counts") if isinstance(source_context, Mapping) else {}, ensure_ascii=False, sort_keys=True)
    imposed = _normalize_instruction_qa_imposed_questions(imposed_questions or [])
    imposed_text = ""
    if imposed:
        imposed_answer_policy = (
            "If a required answer is not directly findable in the image, caption, or source counts, omit that question; do not answer with unknown, not visible, or cannot determine.\n"
            if restrict_speculative_language
            else "If a required answer is not findable in the image, caption, or source counts, say that directly instead of inventing it.\n"
        )
        imposed_text = (
            f"Required user questions: {json.dumps(imposed, ensure_ascii=False)}\n"
            "Keep required question text exactly as provided. "
            f"{imposed_answer_policy}"
        )
    uncertainty_policy = (
        "Do not invent time, weather, purpose, activity, or hidden object function. Omit questions whose answer is not directly findable; do not use unavailable-information answers.\n"
        if restrict_speculative_language
        else "Do not invent time, weather, purpose, activity, or hidden object function; if such a detail is not findable, answer that it is not visible or cannot be determined.\n"
    )
    return (
        "Visual QA recovery top-up. Return only valid JSON and no prose outside JSON.\n"
        f"Return up to {requested} independent image-grounded question/answer pairs with this shape: {answer_shape}.\n"
        "Every question must end with a question mark.\n"
        f"{imposed_text}"
        "Use only facts supported by the image, the caption, or the source counts below.\n"
        "Use canonical object terms from the glossary context when referring to labeled classes.\n"
        "Never output raw labelmap names or odd internal spellings from the labelmap.\n"
        f"{uncertainty_policy}"
        f"{_instruction_qa_speculation_policy_text(restrict_speculative_language)}\n"
        "Do not mention prompts, labels, bounding boxes, coordinates, or source annotations.\n"
        f"{format_text}\n\n"
        f"Caption: {caption_text}\n"
        f"Glossary context: {json.dumps(glossary_context, ensure_ascii=False, sort_keys=True)}\n"
        f"Source counts: {count_text}\n"
        f"Grounding context: {json.dumps(source_context, ensure_ascii=False, sort_keys=True)}"
    )


def _instruction_qa_fallback_max_new_tokens(primary_max_new_tokens: int, requested: int) -> int:
    compact_budget = 192 + max(1, requested) * 160
    return max(128, min(int(primary_max_new_tokens), max(512, compact_budget)))


def _instruction_qa_max_topup_attempts(args: argparse.Namespace) -> int:
    try:
        value = int(getattr(args, "instruction_qa_max_topup_attempts", DEFAULT_INSTRUCTION_QA_MAX_TOPUP_ATTEMPTS))
    except (TypeError, ValueError, OverflowError):
        value = DEFAULT_INSTRUCTION_QA_MAX_TOPUP_ATTEMPTS
    return max(0, min(value, MAX_INSTRUCTION_QA_TOPUP_ATTEMPTS))


def _instruction_qa_restrict_speculative_language(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "instruction_qa_restrict_speculative_language", False))


def _instruction_qa_speculation_policy_text(strict: bool) -> str:
    if strict:
        return (
            "Restrict speculative language: do not use likely, probably, appears, suggests, indicates, could, might, unknown, not visible, or cannot determine. "
            "Ask only questions with directly findable answers and omit unavailable-information questions rather than answering them."
        )
    return (
        "Unavailable-information answers are allowed: if a useful question asks for a detail that is not findable, say that it is not visible or cannot be determined instead of inventing it. "
        "Mild visual inference wording such as likely, appears, suggests, or indicates is allowed when it is grounded in visible evidence."
    )


def _instruction_qa_profile_specs(
    *,
    max_new_tokens: int,
    requested: int,
    max_topup_attempts: int,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "profile": "primary",
            "max_new_tokens": max_new_tokens,
            "decode_override": {
                "do_sample": False,
                "repetition_penalty": 1.08,
                "repetition_context_size": 128,
                "no_repeat_ngram_size": 8,
            },
            "detail": f"Creating up to {requested} generated QA rows",
            "output_section": "Generated QA output",
        },
    ]
    topup_templates = [
        {
            "profile": "caption_grounded_fallback",
            "max_new_tokens": _instruction_qa_fallback_max_new_tokens(max_new_tokens, requested),
            "decode_override": {
                "do_sample": False,
                "repetition_penalty": 1.18,
                "repetition_context_size": 256,
                "no_repeat_ngram_size": 8,
            },
            "detail": "Filling missing generated QA rows from caption-grounded visual context",
            "output_section": "Generated QA caption-grounded fallback output",
        },
        {
            "profile": "sparse_scene_fallback",
            "max_new_tokens": _instruction_qa_fallback_max_new_tokens(max_new_tokens, requested),
            "decode_override": {
                "do_sample": False,
                "repetition_penalty": 1.2,
                "repetition_context_size": 256,
                "no_repeat_ngram_size": 8,
            },
            "detail": "Filling missing generated QA rows with sparse-scene visual prompts",
            "output_section": "Generated QA sparse-scene fallback output",
        },
    ]
    for index in range(max(0, int(max_topup_attempts))):
        spec = dict(topup_templates[index % len(topup_templates)])
        spec["topup_attempt"] = index + 1
        specs.append(spec)
    return specs


def _instruction_qa_profile_label(profile: str) -> str:
    labels = {
        "primary": "Primary prompt",
        "caption_grounded_fallback": "Caption-grounded fallback",
        "sparse_scene_fallback": "Sparse-scene fallback",
        "qa_verifier_rewrite": "Verifier/rewrite",
    }
    return labels.get(str(profile or "").strip(), str(profile or "").replace("_", " ").strip().title() or "QA prompt")


def _question_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _instruction_qa_topup_context(
    *,
    accepted_pairs: Sequence[Mapping[str, Any]],
    rejected_pairs: Sequence[Mapping[str, Any]],
    max_items: int = 12,
) -> dict[str, Any]:
    accepted_questions = [
        str(pair.get("question") or "").strip()
        for pair in accepted_pairs
        if str(pair.get("question") or "").strip()
    ]
    rejected_questions = [
        str(pair.get("question") or "").strip()
        for pair in rejected_pairs
        if str(pair.get("question") or "").strip()
    ]
    return {
        "accepted_questions": accepted_questions[-max_items:],
        "rejected_questions": rejected_questions[-max_items:],
    }


def _instruction_qa_topup_prompt(
    *,
    profile: str,
    case: Mapping[str, Any],
    caption: str,
    requested: int,
    missing: int,
    answer_format: str,
    source_context: Mapping[str, Any],
    glossary_context: Mapping[str, Any],
    accepted_pairs: Sequence[Mapping[str, Any]],
    rejected_pairs: Sequence[Mapping[str, Any]],
    imposed_questions: Sequence[str] | None = None,
    restrict_speculative_language: bool = False,
) -> str:
    caption_text = re.sub(r"\s+", " ", str(caption or "").strip())
    if len(caption_text) > 2400:
        caption_text = caption_text[:2400].rsplit(" ", 1)[0].strip() + "..."
    answer_shape = (
        '{"qa_pairs":[{"question":"...","answer":{"answer":"..."},"row_type":"generated_qa","answer_format":"json"}]}'
        if answer_format == "json"
        else '{"qa_pairs":[{"question":"...","answer":"...","row_type":"generated_qa","answer_format":"natural"}]}'
    )
    format_text = (
        "Each answer must be a JSON object encoded as a JSON value."
        if answer_format == "json"
        else "Each answer must be one concise natural-language sentence."
    )
    imposed = _normalize_instruction_qa_imposed_questions(imposed_questions or [])
    context = _instruction_qa_topup_context(
        accepted_pairs=accepted_pairs,
        rejected_pairs=rejected_pairs,
    )
    count_text = json.dumps(
        source_context.get("object_counts") if isinstance(source_context, Mapping) else {},
        ensure_ascii=False,
        sort_keys=True,
    )
    common = (
        f"Return only valid JSON with this shape: {answer_shape}.\n"
        f"Need {missing} additional accepted image-grounded QA row(s) out of the original target of {requested}.\n"
        "Every question must end with a question mark.\n"
        "Do not repeat any accepted question exactly. Avoid rejected questions unless you can make them clearly grounded and useful.\n"
        "Use canonical object terms from the glossary context when referring to labeled classes. "
        "If a class has no glossary entry, use the natural English class term from the context. "
        "Never output raw labelmap names or odd internal spellings from the labelmap.\n"
        "Do not mention prompts, labels, bounding boxes, coordinates, source annotations, or that counts were provided.\n"
        f"{format_text}\n\n"
        f"Completed broad caption: {caption_text}\n"
        f"Glossary context: {json.dumps(glossary_context, ensure_ascii=False, sort_keys=True)}\n"
        f"Source counts: {count_text}\n"
        f"Accepted/rejected question history: {json.dumps(context, ensure_ascii=False, sort_keys=True)}\n"
    )
    if imposed:
        imposed_answer_policy = (
            "If its answer is not directly findable in the image, caption, or source counts, omit it; do not answer with unknown, not visible, or cannot determine.\n"
            if restrict_speculative_language
            else "If its answer is not findable in the image, caption, or source counts, keep the question and say that directly instead of inventing an answer.\n"
        )
        common += (
            f"Required user questions: {json.dumps(imposed, ensure_ascii=False)}\n"
            "If a required question is still unanswered and grounded, answer it first with the exact question text. "
            f"{imposed_answer_policy}"
        )
    common += f"{_instruction_qa_speculation_policy_text(restrict_speculative_language)}\n"
    if profile == "caption_grounded_fallback":
        return (
            "Caption-grounded visual QA top-up. You can see the image; use it as truth.\n"
            "Prefer straightforward questions that are answerable from the visible scene and the completed caption.\n"
            "Useful fallback questions may ask about the main setting, the main visible object groups, count facts from source counts, visible layout, colors, or spatial relationships.\n"
            "Do not invent activity, intent, time, location, or hidden object function.\n"
            f"{common}"
        )
    if profile == "sparse_scene_fallback":
        sparse_policy = (
            "In strict mode, use only directly answerable sparse-scene questions and omit unavailable-information questions.\n"
            if restrict_speculative_language
            else "If a useful question asks for a detail that is not findable, answer that it is not visible or cannot be determined instead of inventing it.\n"
        )
        return (
            "Sparse-scene visual QA top-up. You can see the image; use it as truth.\n"
            "It is acceptable to produce simple, similar-but-not-identical questions when the scene has little going on.\n"
            "Favor grounded count, presence, arrangement, viewpoint, surface, color, and broad-scene questions. "
            f"{sparse_policy}"
            f"{common}"
        )
    return _instruction_qa_recovery_prompt(
        case=case,
        caption=caption,
        requested=missing,
        answer_format=answer_format,
        source_context=source_context,
        glossary_context=glossary_context,
        imposed_questions=imposed,
        restrict_speculative_language=restrict_speculative_language,
    )


def _instruction_qa_accumulator_payload(
    *,
    case: Mapping[str, Any],
    requested: int,
    accepted_pairs: Sequence[Mapping[str, Any]],
    rejected_pairs: Sequence[Mapping[str, Any]],
    attempt_summary: Sequence[Mapping[str, Any]],
    status: str,
) -> dict[str, Any]:
    accepted_count = len(accepted_pairs)
    clean_status = str(status or "").strip() or ("ok" if accepted_count >= requested else "underfilled")
    final_status = clean_status not in {"generating", "pending"}
    underfilled = bool(final_status and requested > 0 and accepted_count < requested)
    return {
        "case_id": case_key(case),
        "image_name": str(case.get("image_name") or case.get("name") or case.get("stem") or "").strip(),
        "caption0_status": "complete",
        "target_pair_count": int(requested),
        "accepted_pair_count": int(accepted_count),
        "rejected_pair_count": int(len(rejected_pairs)),
        "status": clean_status,
        "underfilled": underfilled,
        "continuing_with_pair_count": int(accepted_count) if underfilled else None,
        "attempt_summary": [dict(item) for item in attempt_summary],
        "pairs": [dict(pair) for pair in accepted_pairs[:20]],
    }


def _instruction_qa_progress_detail(
    payload: Mapping[str, Any],
    *,
    active_profile: str | None = None,
) -> str:
    accepted = int(payload.get("accepted_pair_count") or 0)
    target = int(payload.get("target_pair_count") or 0)
    parts = [
        "Caption0 complete",
        f"Generated Q&A {accepted}/{target} accepted" if target > 0 else "Generated Q&A skipped",
    ]
    for item in payload.get("attempt_summary") or []:
        if not isinstance(item, Mapping):
            continue
        label = str(item.get("label") or _instruction_qa_profile_label(str(item.get("profile") or ""))).strip()
        accepted_count = int(item.get("accepted_count") or 0)
        rejected_count = int(item.get("rejected_count") or 0)
        suffix = f"{accepted_count} accepted"
        if rejected_count:
            suffix = f"{suffix}, {rejected_count} rejected"
        parts.append(f"{label}: {suffix}")
    if payload.get("underfilled"):
        parts.append(f"Continuing with {accepted}/{target}")
    elif active_profile:
        parts.append(f"{_instruction_qa_profile_label(active_profile)} running")
    return " • ".join(parts)


def _phrase_present(text: str, phrase: str) -> bool:
    phrase = str(phrase or "").strip()
    if not phrase:
        return False
    pattern = r"(?<![A-Za-z0-9])" + re.escape(phrase) + r"(?![A-Za-z0-9])"
    return bool(re.search(pattern, str(text or ""), flags=re.IGNORECASE))


def _qa_raw_terms_to_avoid(case: Mapping[str, Any], args: argparse.Namespace) -> dict[str, list[str]]:
    glossary_context = _case_glossary_context(case, args)
    out: dict[str, list[str]] = {}
    for entry in glossary_context.get("classes", []) if isinstance(glossary_context, Mapping) else []:
        if not isinstance(entry, Mapping):
            continue
        canonical = str(entry.get("canonical_term") or "").strip()
        raw_terms = [
            str(term).strip()
            for term in (entry.get("raw_terms_to_avoid") or [])
            if str(term).strip()
        ]
        if canonical and raw_terms:
            out[canonical] = raw_terms
    return out


def _qa_pair_rejection_reasons(
    pair: Mapping[str, Any],
    case: Mapping[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    reasons: list[str] = []
    question = str(pair.get("question") or "").strip()
    answer = str(pair.get("answer") or "").strip()
    combined = f"{question}\n{answer}"
    if not question.endswith("?"):
        reasons.append("question_not_question")
    if len(answer.split()) < 2:
        reasons.append("answer_too_short")
    for _canonical, raw_terms in _qa_raw_terms_to_avoid(case, args).items():
        for raw_term in raw_terms:
            if _phrase_present(combined, raw_term):
                reasons.append(f"{QA_RAW_LABEL_LEAK_PREFIX}{raw_term}")
    if _instruction_qa_restrict_speculative_language(args) and QA_STRICT_SPECULATION_PATTERN.search(combined):
        reasons.append("speculative_or_unavailable_language")
    return sorted(set(reasons))


def _qa_machine_validated_pair(
    pair: Mapping[str, Any],
    *,
    verifier_decision: str,
    verifier_reasons: Sequence[str] | None = None,
    question: str | None = None,
    answer: str | None = None,
) -> dict[str, Any]:
    out = dict(pair)
    original_question = str(pair.get("question") or "").strip()
    original_answer = str(pair.get("answer") or "").strip()
    if question is not None:
        out["question"] = re.sub(r"\s+", " ", str(question or "").strip())
    if answer is not None:
        if isinstance(answer, (dict, list)):
            out["answer"] = json.dumps(answer, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        else:
            out["answer"] = re.sub(r"\s+", " ", str(answer or "").strip())
    out["validation_status"] = QA_VERIFIER_STATUS
    out["review_status"] = QA_VERIFIER_STATUS
    out["answer_source"] = str(out.get("answer_source") or "vlm_generated").strip() or "vlm_generated"
    out["validated_against"] = ["image", "language_annotations.caption0", "source_annotations", "qa_verifier"]
    metadata = dict(out.get("metadata")) if isinstance(out.get("metadata"), Mapping) else {}
    metadata["qa_verifier"] = {
        "status": QA_VERIFIER_STATUS,
        "decision": verifier_decision,
        "reasons": list(verifier_reasons or []),
    }
    if verifier_decision == "rewrite":
        metadata["original_question"] = original_question
        metadata["original_answer"] = original_answer
    out["metadata"] = metadata
    validation = dict(out.get("validation")) if isinstance(out.get("validation"), Mapping) else {}
    validation.update(
        {
            "status": QA_VERIFIER_STATUS,
            "qa_verifier_passed": True,
            "qa_verifier_decision": verifier_decision,
        }
    )
    out["validation"] = validation
    out.pop("rejection_reasons", None)
    return out


def _qa_rejected_pair(pair: Mapping[str, Any], reasons: Sequence[str]) -> dict[str, Any]:
    out = dict(pair)
    clean_reasons = sorted({str(reason).strip() for reason in reasons if str(reason).strip()})
    out["validation_status"] = "rejected"
    out["review_status"] = "rejected"
    out["rejection_reasons"] = clean_reasons or ["qa_verifier_rejected"]
    metadata = dict(out.get("metadata")) if isinstance(out.get("metadata"), Mapping) else {}
    metadata["qa_verifier"] = {
        "status": "rejected",
        "decision": "reject",
        "reasons": out["rejection_reasons"],
    }
    out["metadata"] = metadata
    validation = dict(out.get("validation")) if isinstance(out.get("validation"), Mapping) else {}
    validation.update(
        {
            "status": "rejected",
            "qa_verifier_passed": False,
            "qa_verifier_decision": "reject",
            "rejection_reasons": out["rejection_reasons"],
        }
    )
    out["validation"] = validation
    return out


def _instruction_qa_verifier_prompt(
    *,
    candidates: Sequence[Mapping[str, Any]],
    case: Mapping[str, Any],
    caption: str,
    args: argparse.Namespace,
    source_context: Mapping[str, Any],
    glossary_context: Mapping[str, Any],
) -> str:
    caption_text = re.sub(r"\s+", " ", str(caption or "").strip())[:2400]
    candidate_rows = []
    for pair in candidates:
        candidate_rows.append(
            {
                "qa_id": str(pair.get("qa_id") or pair.get("id") or "").strip(),
                "question": str(pair.get("question") or "").strip(),
                "answer": str(pair.get("answer") or "").strip(),
                "deterministic_reasons": _qa_pair_rejection_reasons(pair, case, args),
            }
        )
    if _instruction_qa_restrict_speculative_language(args):
        speculation_rule = (
            "Strict speculative language is enabled: reject pairs that use speculative or unavailable-information wording such as likely, probably, appears, suggests, could, might, unknown, not visible, or cannot determine."
        )
    else:
        speculation_rule = (
            "Answers that say a detail is not visible, unknown, or cannot be determined are acceptable when that is the grounded answer; do not reject them only for being unavailable-information answers. Mild inference wording is acceptable when grounded in visible evidence."
        )
    return (
        "Verify and, when possible, rewrite generated image QA pairs for training.\n"
        "Return only valid JSON with this shape: "
        '{"qa_pairs":[{"qa_id":"...","decision":"accept|rewrite|reject","question":"...","answer":"...","rejection_reasons":[]}]}\n'
        "Use rewrite when a pair is useful but contains raw label names or a malformed question.\n"
        "Reject pairs that are duplicate, empty, contradictory, or cannot be fixed from the provided context.\n"
        f"{speculation_rule}\n"
        "Every accepted or rewritten question must end with a question mark.\n"
        "Use canonical object terms from the glossary context for labeled classes. Never output raw labelmap names or odd internal spellings from the labelmap.\n"
        "Do not mention prompts, labels, bounding boxes, coordinates, source annotations, or that counts were provided.\n\n"
        f"Caption: {caption_text}\n"
        f"Glossary context: {json.dumps(glossary_context, ensure_ascii=False, sort_keys=True)}\n"
        f"Source context: {json.dumps(source_context, ensure_ascii=False, sort_keys=True)}\n"
        f"Candidates: {json.dumps(candidate_rows, ensure_ascii=False, sort_keys=True)}"
    )


def _verifier_candidate_from_payload(payload: Any) -> dict[str, Mapping[str, Any]]:
    raw_pairs = payload.get("qa_pairs") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_pairs, list):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for raw in raw_pairs:
        if not isinstance(raw, Mapping):
            continue
        qa_id = str(raw.get("qa_id") or raw.get("id") or "").strip()
        if qa_id:
            out[qa_id] = raw
    return out


def _verify_generated_qa_pairs(
    api: Any,
    pairs: Sequence[Mapping[str, Any]],
    *,
    case: Mapping[str, Any],
    caption: str,
    args: argparse.Namespace,
    source_context: Mapping[str, Any],
    glossary_context: Mapping[str, Any],
) -> dict[str, Any]:
    verified: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    verifier_attempts: list[dict[str, Any]] = []
    flagged: list[Mapping[str, Any]] = []
    for pair in pairs:
        reasons = _qa_pair_rejection_reasons(pair, case, args)
        if reasons:
            flagged.append(pair)
        else:
            verified.append(
                _qa_machine_validated_pair(
                    pair,
                    verifier_decision="accept",
                    verifier_reasons=[],
                )
            )
    if not flagged:
        return {
            "pairs": verified,
            "rejected_pairs": rejected,
            "attempts": verifier_attempts,
            "status": "ok",
        }

    if not hasattr(api, "_run_qwen_text_inference"):
        for pair in flagged:
            rejected.append(_qa_rejected_pair(pair, _qa_pair_rejection_reasons(pair, case, args)))
        return {
            "pairs": verified,
            "rejected_pairs": rejected,
            "attempts": [{"status": "skipped", "reason": "text_inference_unavailable"}],
            "status": "ok" if verified else "empty",
        }

    prompt = _instruction_qa_verifier_prompt(
        candidates=flagged,
        case=case,
        caption=caption,
        args=args,
        source_context=source_context,
        glossary_context=glossary_context,
    )
    attempt_started = time.time()
    attempt_record: dict[str, Any] = {
        "profile": "qa_verifier_rewrite",
        "candidate_count": len(flagged),
        "max_new_tokens": QA_VERIFIER_MAX_NEW_TOKENS,
    }
    try:
        if hasattr(api, "_qwen_progress_update"):
            api._qwen_progress_update(
                phase="generate",
                phase_label="Verifying QA",
                progress=0.95,
                message="Verifying generated QA rows",
                step_id="instruction_qa_verify",
                step_label="Verify instruction QA",
                step_detail=f"Rewriting or rejecting {len(flagged)} generated QA row(s)",
                token_preview="",
                live_output_reset=True,
            )
        raw, _width, _height = api._run_qwen_text_inference(
            prompt,
            max_new_tokens=QA_VERIFIER_MAX_NEW_TOKENS,
            system_prompt_override="You are a strict visual QA verifier. Return only valid JSON.",
            model_id_override=getattr(args, "model_id", None),
            decode_override={
                "do_sample": False,
                "repetition_penalty": 1.18,
                "repetition_context_size": 256,
                "no_repeat_ngram_size": 8,
            },
            chat_template_kwargs={"enable_thinking": False},
        )
        parsed = _extract_json_payload(raw)
        decisions = _verifier_candidate_from_payload(parsed)
        attempt_record.update(
            {
                "status": "ok" if decisions else "empty",
                "raw_output": raw,
                "decision_count": len(decisions),
                "elapsed_seconds": round(time.time() - attempt_started, 3),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        decisions = {}
        attempt_record.update(
            {
                "status": "error",
                "type": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": round(time.time() - attempt_started, 3),
            }
        )
    verifier_attempts.append(attempt_record)

    for pair in flagged:
        qa_id = str(pair.get("qa_id") or pair.get("id") or "").strip()
        decision = decisions.get(qa_id)
        if not isinstance(decision, Mapping):
            rejected.append(
                _qa_rejected_pair(
                    pair,
                    [*_qa_pair_rejection_reasons(pair, case, args), "qa_verifier_missing_decision"],
                )
            )
            continue
        action = re.sub(r"[\s-]+", "_", str(decision.get("decision") or "").strip().lower())
        if action not in {"accept", "rewrite", "reject"}:
            action = "reject"
        if action == "reject":
            reasons = decision.get("rejection_reasons") if isinstance(decision.get("rejection_reasons"), list) else []
            rejected.append(_qa_rejected_pair(pair, [str(reason) for reason in reasons] or ["qa_verifier_rejected"]))
            continue
        question = str(decision.get("question") or pair.get("question") or "").strip()
        answer_value = decision.get("answer") if "answer" in decision else pair.get("answer")
        candidate = _qa_machine_validated_pair(
            pair,
            verifier_decision=action,
            verifier_reasons=_qa_pair_rejection_reasons(pair, case, args),
            question=question,
            answer=answer_value,
        )
        post_reasons = _qa_pair_rejection_reasons(candidate, case, args)
        if post_reasons:
            rejected.append(_qa_rejected_pair(candidate, [*post_reasons, "qa_verifier_postcheck_failed"]))
        else:
            verified.append(candidate)

    return {
        "pairs": verified,
        "rejected_pairs": rejected,
        "attempts": verifier_attempts,
        "status": "ok" if verified else "empty",
    }


def generate_instruction_qa_pairs(
    api: Any,
    case: Mapping[str, Any],
    image_path: Path,
    caption: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt_spec = build_instruction_qa_prompt(case, caption, args)
    requested = int(prompt_spec.get("requested") or 0)
    if requested <= 0:
        return {"requested": 0, "pairs": [], "rejections": [], "status": "skipped"}
    answer_format = str(prompt_spec.get("answer_format") or "natural")
    system_prompt = str(prompt_spec.get("system_prompt") or "")
    max_new_tokens = int(prompt_spec.get("max_new_tokens") or _instruction_qa_max_new_tokens(args, requested))
    restrict_speculative_language = bool(prompt_spec.get("restrict_speculative_language"))
    started = time.time()
    attempts: list[dict[str, Any]] = []
    attempt_summary: list[dict[str, Any]] = []
    accepted_pairs: list[dict[str, Any]] = []
    rejected_pairs: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    accepted_question_keys: set[str] = set()
    recovered_by: str | None = None
    source_context = (
        prompt_spec.get("source_context")
        if isinstance(prompt_spec.get("source_context"), Mapping)
        else {}
    )
    glossary_context = (
        prompt_spec.get("glossary_context")
        if isinstance(prompt_spec.get("glossary_context"), Mapping)
        else {}
    )
    imposed_questions = (
        prompt_spec.get("imposed_questions")
        if isinstance(prompt_spec.get("imposed_questions"), Sequence)
        else []
    )
    max_topup_attempts = _instruction_qa_max_topup_attempts(args)
    profile_specs = _instruction_qa_profile_specs(
        max_new_tokens=max_new_tokens,
        requested=requested,
        max_topup_attempts=max_topup_attempts,
    )
    for attempt_index, attempt_spec in enumerate(profile_specs, start=1):
        missing = max(0, requested - len(accepted_pairs))
        if missing <= 0:
            break
        attempt_started = time.time()
        profile = str(attempt_spec["profile"])
        profile_label = _instruction_qa_profile_label(profile)
        attempt_max_new_tokens = int(attempt_spec["max_new_tokens"])
        if profile == "primary":
            attempt_prompt = str(prompt_spec.get("prompt") or "")
            attempt_requested = requested
        else:
            attempt_requested = missing
            attempt_prompt = _instruction_qa_topup_prompt(
                profile=profile,
                case=case,
                caption=caption,
                requested=requested,
                missing=missing,
                answer_format=answer_format,
                source_context=source_context,
                glossary_context=glossary_context,
                accepted_pairs=accepted_pairs,
                rejected_pairs=rejected_pairs,
                imposed_questions=imposed_questions,
                restrict_speculative_language=restrict_speculative_language,
            )
        attempt_record: dict[str, Any] = {
            "attempt": attempt_index,
            "profile": profile,
            "label": profile_label,
            "call_kind": "visual",
            "requested": attempt_requested,
            "max_new_tokens": attempt_max_new_tokens,
        }
        if attempt_spec.get("topup_attempt"):
            attempt_record["topup_attempt"] = int(attempt_spec["topup_attempt"])
        summary_record: dict[str, Any] = {
            "profile": profile,
            "label": profile_label,
            "call_kind": "visual",
            "requested": attempt_requested,
            "accepted_count": 0,
            "rejected_count": 0,
            "raw_pair_count": 0,
            "status": "pending",
            "total_accepted": len(accepted_pairs),
        }
        if attempt_spec.get("topup_attempt"):
            summary_record["topup_attempt"] = int(attempt_spec["topup_attempt"])
        try:
            if hasattr(api, "_qwen_progress_update"):
                accumulator = _instruction_qa_accumulator_payload(
                    case=case,
                    requested=requested,
                    accepted_pairs=accepted_pairs,
                    rejected_pairs=rejected_pairs,
                    attempt_summary=attempt_summary,
                    status="generating",
                )
                api._qwen_progress_update(
                    phase="generate",
                    phase_label="Generating QA",
                    progress=0.9,
                    message="Generating instruction QA rows",
                    step_id="instruction_qa",
                    step_label="Generate instruction QA",
                    step_detail=_instruction_qa_progress_detail(accumulator, active_profile=profile),
                    token_preview="",
                    live_output_reset=True,
                    instruction_qa_accumulator=accumulator,
                )
            if hasattr(api, "_qwen_progress_begin_output_section"):
                api._qwen_progress_begin_output_section(str(attempt_spec["output_section"]))
            with Image.open(image_path) as source_image:
                pil_img = source_image.convert("RGB")
            raw, _width, _height = api._run_qwen_inference(
                attempt_prompt,
                pil_img,
                max_new_tokens=attempt_max_new_tokens,
                system_prompt_override=system_prompt,
                model_id_override=getattr(args, "model_id", None),
                decode_override=attempt_spec["decode_override"],
                chat_template_kwargs={"enable_thinking": False},
            )
            parsed = _extract_json_payload(raw)
            parse_rejections: list[dict[str, Any]] = []
            if parsed is None:
                parsed = _extract_partial_qa_payload(raw)
                if parsed is not None:
                    parse_rejections.append({"reason": "qa_json_partial_payload_salvaged"})
                else:
                    parse_rejections.append({"reason": "qa_json_parse_failed"})
            pairs, normalize_rejections = _normalize_generated_qa_pairs(
                parsed,
                requested=attempt_requested,
                case=case,
                answer_format=answer_format,
            )
            normalize_rejections = [*parse_rejections, *normalize_rejections]
            status = "ok" if pairs else "empty"
            attempt_record.update(
                {
                    "status": status,
                    "pair_count": len(pairs),
                    "rejections": normalize_rejections,
                    "raw_output": raw,
                    "elapsed_seconds": round(time.time() - attempt_started, 3),
                }
            )
            attempts.append(attempt_record)
            rejections.extend(normalize_rejections)
            summary_record["raw_pair_count"] = len(pairs)
            if pairs:
                verifier_result = _verify_generated_qa_pairs(
                    api,
                    pairs,
                    case=case,
                    caption=caption,
                    args=args,
                    source_context=source_context,
                    glossary_context=glossary_context,
                )
                verified_pairs = list(verifier_result.get("pairs") or [])
                attempt_rejected_pairs = list(verifier_result.get("rejected_pairs") or [])
                for rejected_pair in attempt_rejected_pairs:
                    rejected_pairs.append(dict(rejected_pair))
                    rejections.append(
                        {
                            "reason": "qa_verifier_rejected",
                            "qa_id": str(rejected_pair.get("qa_id") or rejected_pair.get("id") or "").strip(),
                            "question": str(rejected_pair.get("question") or "").strip(),
                            "rejection_reasons": list(rejected_pair.get("rejection_reasons") or []),
                        }
                    )
                attempts.extend(list(verifier_result.get("attempts") or []))
                attempt_accepted = 0
                for pair in verified_pairs:
                    key = _question_key(pair.get("question"))
                    if key in accepted_question_keys:
                        duplicate = _qa_rejected_pair(pair, ["duplicate_question_across_qa_profiles"])
                        rejected_pairs.append(duplicate)
                        rejections.append(
                            {
                                "reason": "duplicate_question_across_qa_profiles",
                                "qa_id": str(pair.get("qa_id") or pair.get("id") or "").strip(),
                                "question": str(pair.get("question") or "").strip(),
                            }
                        )
                        continue
                    accepted_question_keys.add(key)
                    accepted = dict(pair)
                    accepted["qa_id"] = f"{case_key(case)}__generated_qa_{len(accepted_pairs) + 1:04d}"
                    metadata = dict(accepted.get("metadata")) if isinstance(accepted.get("metadata"), Mapping) else {}
                    metadata["qa_profile"] = profile
                    metadata["qa_profile_label"] = profile_label
                    accepted["metadata"] = metadata
                    accepted_pairs.append(accepted)
                    attempt_accepted += 1
                    if profile != "primary" and recovered_by is None:
                        recovered_by = profile
                    if len(accepted_pairs) >= requested:
                        break
                summary_record["accepted_count"] = attempt_accepted
                summary_record["rejected_count"] = (
                    len(normalize_rejections)
                    + len(attempt_rejected_pairs)
                    + max(0, len(verified_pairs) - attempt_accepted)
                )
                summary_record["status"] = "ok" if attempt_accepted else "empty"
                summary_record["total_accepted"] = len(accepted_pairs)
            else:
                summary_record["rejected_count"] = len(normalize_rejections)
                summary_record["status"] = "empty"
        except BaseException as exc:  # noqa: BLE001
            loop_diagnostic: dict[str, Any] | None = None
            diagnostic_fn = getattr(exc, "diagnostic_payload", None)
            if callable(diagnostic_fn):
                try:
                    payload = diagnostic_fn(max_chars=6000)
                except TypeError:
                    payload = diagnostic_fn()
                except Exception:
                    payload = None
                if isinstance(payload, Mapping):
                    loop_diagnostic = dict(payload)
            salvaged_pairs: list[dict[str, Any]] = []
            salvage_normalize_rejections: list[dict[str, Any]] = []
            salvage_rejected_pairs: list[dict[str, Any]] = []
            salvage_accepted = 0
            salvage_error: dict[str, Any] | None = None
            if loop_diagnostic:
                for diagnostic_key in ("trimmed_output", "raw_output_head", "raw_output_tail"):
                    partial_payload = _extract_partial_qa_payload(
                        str(loop_diagnostic.get(diagnostic_key) or "")
                    )
                    if partial_payload:
                        salvaged_pairs, salvage_normalize_rejections = _normalize_generated_qa_pairs(
                            partial_payload,
                            requested=attempt_requested,
                            case=case,
                            answer_format=answer_format,
                        )
                        if salvaged_pairs:
                            attempt_record["loop_salvage_source"] = diagnostic_key
                            break
                if salvaged_pairs:
                    try:
                        verifier_result = _verify_generated_qa_pairs(
                            api,
                            salvaged_pairs,
                            case=case,
                            caption=caption,
                            args=args,
                            source_context=source_context,
                            glossary_context=glossary_context,
                        )
                        verified_pairs = list(verifier_result.get("pairs") or [])
                        salvage_rejected_pairs = list(verifier_result.get("rejected_pairs") or [])
                        for rejected_pair in salvage_rejected_pairs:
                            rejected_pairs.append(dict(rejected_pair))
                            rejections.append(
                                {
                                    "reason": "qa_verifier_rejected",
                                    "qa_id": str(rejected_pair.get("qa_id") or rejected_pair.get("id") or "").strip(),
                                    "question": str(rejected_pair.get("question") or "").strip(),
                                    "rejection_reasons": list(rejected_pair.get("rejection_reasons") or []),
                                    "salvaged_from_loop": True,
                                }
                            )
                        attempts.extend(list(verifier_result.get("attempts") or []))
                        for pair in verified_pairs:
                            key = _question_key(pair.get("question"))
                            if key in accepted_question_keys:
                                duplicate = _qa_rejected_pair(pair, ["duplicate_question_across_qa_profiles"])
                                rejected_pairs.append(duplicate)
                                rejections.append(
                                    {
                                        "reason": "duplicate_question_across_qa_profiles",
                                        "qa_id": str(pair.get("qa_id") or pair.get("id") or "").strip(),
                                        "question": str(pair.get("question") or "").strip(),
                                        "salvaged_from_loop": True,
                                    }
                                )
                                continue
                            accepted_question_keys.add(key)
                            accepted = dict(pair)
                            accepted["qa_id"] = f"{case_key(case)}__generated_qa_{len(accepted_pairs) + 1:04d}"
                            metadata = (
                                dict(accepted.get("metadata"))
                                if isinstance(accepted.get("metadata"), Mapping)
                                else {}
                            )
                            metadata["qa_profile"] = profile
                            metadata["qa_profile_label"] = profile_label
                            metadata["salvaged_from_loop"] = True
                            accepted["metadata"] = metadata
                            accepted_pairs.append(accepted)
                            salvage_accepted += 1
                            if profile != "primary" and recovered_by is None:
                                recovered_by = profile
                            if len(accepted_pairs) >= requested:
                                break
                    except Exception as salvage_exc:  # noqa: BLE001
                        salvage_error = {
                            "reason": "loop_salvage_failed",
                            "type": type(salvage_exc).__name__,
                            "message": str(salvage_exc),
                        }
            rejection: dict[str, Any] = {
                "reason": "generator_exception",
                "type": type(exc).__name__,
                "message": str(exc),
            }
            if loop_diagnostic:
                rejection["loop_diagnostic"] = loop_diagnostic
            if salvaged_pairs:
                rejection["salvaged_pair_count"] = len(salvaged_pairs)
                rejection["salvaged_accepted_count"] = salvage_accepted
            attempt_record.update(
                {
                    "status": "partial_loop_recovered" if salvage_accepted else "error",
                    "pair_count": len(salvaged_pairs),
                    "rejections": [
                        rejection,
                        *salvage_normalize_rejections,
                        *([salvage_error] if salvage_error else []),
                    ],
                    "elapsed_seconds": round(time.time() - attempt_started, 3),
                }
            )
            if loop_diagnostic:
                attempt_record["loop_diagnostic"] = loop_diagnostic
            if salvaged_pairs:
                attempt_record["salvaged_pair_count"] = len(salvaged_pairs)
                attempt_record["salvaged_accepted_count"] = salvage_accepted
            attempts.append(attempt_record)
            rejections.append(rejection)
            rejections.extend(salvage_normalize_rejections)
            if salvage_error:
                rejections.append(salvage_error)
            summary_record.update(
                {
                    "status": "partial_loop_recovered" if salvage_accepted else "error",
                    "accepted_count": salvage_accepted,
                    "rejected_count": (
                        1
                        + len(salvage_normalize_rejections)
                        + len(salvage_rejected_pairs)
                        + max(0, len(salvaged_pairs) - salvage_accepted - len(salvage_rejected_pairs))
                    ),
                    "raw_pair_count": len(salvaged_pairs),
                    "total_accepted": len(accepted_pairs),
                    "elapsed_seconds": round(time.time() - attempt_started, 3),
                    "error_type": type(exc).__name__,
                }
            )
            if salvaged_pairs:
                summary_record["salvaged_pair_count"] = len(salvaged_pairs)
                summary_record["salvaged_accepted_count"] = salvage_accepted
        if "elapsed_seconds" not in summary_record:
            summary_record["elapsed_seconds"] = round(time.time() - attempt_started, 3)
        summary_record["total_accepted"] = len(accepted_pairs)
        attempt_summary.append(summary_record)
        if hasattr(api, "_qwen_progress_update"):
            status = "ok" if len(accepted_pairs) >= requested else "generating"
            accumulator = _instruction_qa_accumulator_payload(
                case=case,
                requested=requested,
                accepted_pairs=accepted_pairs,
                rejected_pairs=rejected_pairs,
                attempt_summary=attempt_summary,
                status=status,
            )
            api._qwen_progress_update(
                phase="generate",
                phase_label="Generating QA",
                progress=0.9,
                message="Generating instruction QA rows",
                step_id="instruction_qa",
                step_label="Generate instruction QA",
                step_detail=_instruction_qa_progress_detail(accumulator),
                instruction_qa_accumulator=accumulator,
            )
    final_status = "ok" if len(accepted_pairs) >= requested else ("underfilled" if accepted_pairs else "empty")
    if not accepted_pairs and any(str(attempt.get("status") or "") == "error" for attempt in attempts):
        final_status = "error"
    accumulator = _instruction_qa_accumulator_payload(
        case=case,
        requested=requested,
        accepted_pairs=accepted_pairs,
        rejected_pairs=rejected_pairs,
        attempt_summary=attempt_summary,
        status=final_status,
    )
    if hasattr(api, "_qwen_progress_update"):
        api._qwen_progress_update(
            phase="generate",
            phase_label="Generating QA",
            progress=0.92,
            message="Generated instruction QA rows"
            if accepted_pairs
            else "No accepted generated instruction QA rows",
            step_id="instruction_qa",
            step_label="Generate instruction QA",
            step_detail=_instruction_qa_progress_detail(accumulator),
            instruction_qa_accumulator=accumulator,
        )
    verifier_rejected = [
        pair
        for pair in rejected_pairs
        if isinstance(pair.get("metadata"), Mapping)
        and isinstance(pair.get("metadata", {}).get("qa_verifier"), Mapping)
    ]
    return {
        "status": final_status,
        "requested": requested,
        "target_pair_count": requested,
        "pair_count": len(accepted_pairs),
        "accepted_pair_count": len(accepted_pairs),
        "rejected_pair_count": len(rejected_pairs),
        "pairs": accepted_pairs,
        "rejected_pairs": rejected_pairs,
        "rejections": rejections,
        "max_new_tokens": max_new_tokens,
        "max_topup_attempts": max_topup_attempts,
        "restrict_speculative_language": restrict_speculative_language,
        "elapsed_seconds": round(time.time() - started, 3),
        "attempts": attempts,
        "attempt_summary": attempt_summary,
        "accumulator": accumulator,
        "underfilled": bool(requested > 0 and len(accepted_pairs) < requested),
        "continuing_with_pair_count": len(accepted_pairs)
        if requested > 0 and len(accepted_pairs) < requested
        else None,
        "verifier": {
            "validated_count": len(accepted_pairs),
            "rejected_count": len(verifier_rejected),
        },
        "recovered_by": recovered_by,
    }


def run_worker(case_path: Path, output_dir: Path, dataset_root: Path, args: argparse.Namespace) -> int:
    case = json.loads(case_path.read_text())
    payload_data = case_payload(case, dataset_root, args)
    image_path = Path(str(case.get("image_path") or ""))
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_log_bytes = normalize_artifact_log_bytes(
        getattr(args, "max_artifact_log_bytes", DEFAULT_ARTIFACT_LOG_BYTES)
    )
    payload_out = output_dir / "payload.json"
    payload_copy = dict(payload_data)
    payload_copy["image_base64"] = f"<{len(payload_data['image_base64'])} base64 chars>"
    payload_out.write_text(json.dumps(payload_copy, indent=2, sort_keys=True))

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("QWEN_CAPTION_CACHE_LIMIT", "0")
    os.environ.setdefault("TATOR_QWEN_CAPTION_PROGRESS_STALE_SECONDS", "180")
    os.environ.setdefault("QWEN_CAPTION_MLX_MAX_IMAGE_SIDE", str(args.mlx_max_image_side))

    started = time.time()
    result: dict[str, Any] = {
        "case": case,
        "model_id": args.model_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "error",
    }
    progress_stop = threading.Event()
    progress_thread: threading.Thread | None = None
    api_module: Any = None
    try:
        import localinferenceapi as local_api
        from models.schemas import QwenCaptionRequest

        api = caption_api_for_args(local_api, args)
        api_module = api
        progress_thread = start_worker_progress_mirror(
            api=api,
            output_dir=output_dir,
            stop_event=progress_stop,
            poll_seconds=0.5,
        )
        request = QwenCaptionRequest(**payload_data)
        preview = api.qwen_caption_prompt_preview(request)
        result["preview"] = preview.dict() if hasattr(preview, "dict") else dict(preview)
        if args.preview_only:
            result["status"] = "preview_only"
        else:
            response = api.qwen_caption(request)
            response_data = response.dict() if hasattr(response, "dict") else dict(response)
            caption_text = str(response_data.get("caption") or "").strip()
            requested_subcaptions = _instruction_qa_requested_count(args)
            if bool(getattr(args, "instruction_dataset", False)) and requested_subcaptions > 0:
                instruction_qa = generate_instruction_qa_pairs(
                    api,
                    case,
                    image_path,
                    caption_text,
                    args,
                )
                result["instruction_qa"] = instruction_qa
                response_data["generated_qa_pairs"] = list(instruction_qa.get("pairs") or [])
                response_data["generated_qa_rejected_pairs"] = list(
                    instruction_qa.get("rejected_pairs") or []
                )
                response_data["generated_qa_pair_count"] = int(instruction_qa.get("pair_count") or 0)
                response_data["generated_qa_rejected_pair_count"] = int(
                    instruction_qa.get("rejected_pair_count") or 0
                )
                response_data["generated_qa_target_pair_count"] = int(
                    instruction_qa.get("target_pair_count")
                    or instruction_qa.get("requested")
                    or requested_subcaptions
                    or 0
                )
                response_data["generated_qa_attempt_summary"] = list(
                    instruction_qa.get("attempt_summary") or []
                )
                if isinstance(instruction_qa.get("accumulator"), Mapping):
                    response_data["generated_qa_accumulator"] = dict(instruction_qa["accumulator"])
                if bool(instruction_qa.get("underfilled")) and int(instruction_qa.get("pair_count") or 0) > 0:
                    warning = {
                        "type": "InstructionQaUnderfilledWarning",
                        "message": (
                            "generated QA was requested, but fewer accepted QA pairs "
                            f"were produced than requested; accepted={int(instruction_qa.get('pair_count') or 0)} "
                            f"target={int(response_data.get('generated_qa_target_pair_count') or 0)}"
                        ),
                        "status": instruction_qa.get("status"),
                        "accepted_pair_count": int(instruction_qa.get("pair_count") or 0),
                        "target_pair_count": int(response_data.get("generated_qa_target_pair_count") or 0),
                        "attempt_summary": list(instruction_qa.get("attempt_summary") or []),
                    }
                    result["instruction_qa_warning"] = warning
                    response_data["generated_qa_warning"] = warning
            result["response"] = response_data
            result["caption_quality"] = summarize_caption(
                caption_text,
                {str(k): int(v) for k, v in response_data.get("used_counts", {}).items()},
                _case_glossary_map(case, args),
            )
            if (
                bool(getattr(args, "instruction_dataset", False))
                and requested_subcaptions > 0
                and int(response_data.get("generated_qa_pair_count") or 0) <= 0
            ):
                instruction_status = (
                    result.get("instruction_qa", {}).get("status")
                    if isinstance(result.get("instruction_qa"), Mapping)
                    else "missing"
                )
                warning = {
                    "type": "InstructionQaGenerationWarning",
                    "message": (
                        "generated QA was requested, but no valid generated QA pairs "
                        f"were produced; status={instruction_status}"
                    ),
                    "status": instruction_status,
                    "qa_only_required": _generated_qa_required_for_worker_success(args),
                }
                if _generated_qa_required_for_worker_success(args):
                    result["status"] = "instruction_qa_failed"
                    result["exception"] = {
                        "type": "InstructionQaGenerationError",
                        "message": warning["message"],
                    }
                    return 1
                result["instruction_qa_warning"] = warning
                response_data["generated_qa_warning"] = warning
            result["status"] = "ok"
    except BaseException as exc:  # noqa: BLE001
        result["status"] = "exception"
        result["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        progress_stop.set()
        if api_module is not None:
            try:
                previous_progress = _read_json_dict(output_dir / WORKER_PROGRESS_JSON)
                previous_seq = int(previous_progress.get("seq") or 0)
            except Exception:
                previous_seq = 0
            try:
                final_progress = worker_progress_snapshot(
                    api_module.qwen_progress(),
                    seq=previous_seq + 1,
                )
                final_progress["final_snapshot"] = True
                atomic_write_json(output_dir / WORKER_PROGRESS_JSON, final_progress)
                append_jsonl(output_dir / WORKER_PROGRESS_JSONL, final_progress)
            except Exception:
                pass
        if progress_thread is not None:
            progress_thread.join(timeout=1.0)
        result["elapsed_seconds"] = round(time.time() - started, 3)
        artifact_files, artifact_errors = copy_worker_qwen_caption_io_artifacts(
            output_dir,
            max_bytes=artifact_log_bytes,
        )
        result["artifact_limits"] = {
            "max_artifact_log_bytes": artifact_log_bytes,
            "files": artifact_files,
        }
        if artifact_errors:
            result["artifact_errors"] = artifact_errors
        (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"ok", "preview_only"} else 1


def _select_parent_cases(
    args: argparse.Namespace,
    dataset_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.cases_json:
        loaded_cases = json.loads(Path(args.cases_json).read_text())
        if not isinstance(loaded_cases, list):
            raise SystemExit("--cases-json must contain a JSON list")
        cases = [dict(case) for case in loaded_cases if isinstance(case, dict)]
    else:
        items = discover_items(dataset_root)
        if args.all_images:
            cases = select_all_image_cases(items, caption_mode=args.caption_mode)
        else:
            cases = select_cases(items)
    if not cases:
        raise SystemExit("No caption cases selected")
    sample_meta: dict[str, Any] = {
        "strategy": "all",
        "source_cases": len(cases),
        "selected_cases": len(cases),
        "requested_sample_size": max(0, int(args.sample_size or 0)),
        "sample_seed": int(args.sample_seed or 0),
        "stress_case_keys": [],
        "random_fill_case_keys": [],
    }
    if args.sample_size and args.sample_size > 0:
        cases, sample_meta = sample_cases_with_meta(
            cases,
            sample_size=int(args.sample_size),
            sample_seed=int(args.sample_seed),
        )
    if args.case:
        wanted = set(args.case)
        cases = [
            case
            for case in cases
            if case["name"] in wanted or case["stem"] in wanted or case_key(case) in wanted
        ]
    if args.limit:
        cases = cases[: args.limit]
    return cases, sample_meta


def _worker_attempt_command(
    args: argparse.Namespace,
    *,
    dataset_root: Path,
    attempt_case_path: Path,
    attempt_dir: Path,
    attempt_profile: Mapping[str, Any],
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--case-json",
        str(attempt_case_path),
        "--output-dir",
        str(attempt_dir),
        "--dataset-root",
        str(dataset_root),
        "--caption-provider",
        str(getattr(args, "caption_provider", CAPTION_PROVIDER_LOCAL) or CAPTION_PROVIDER_LOCAL),
        "--openai-model",
        str(getattr(args, "openai_model", DEFAULT_OPENAI_MODEL) or DEFAULT_OPENAI_MODEL),
        "--openai-image-detail",
        str(getattr(args, "openai_image_detail", DEFAULT_OPENAI_IMAGE_DETAIL) or DEFAULT_OPENAI_IMAGE_DETAIL),
        "--openai-reasoning-effort",
        str(getattr(args, "openai_reasoning_effort", DEFAULT_OPENAI_REASONING_EFFORT) or DEFAULT_OPENAI_REASONING_EFFORT),
        "--openai-api-key-file",
        str(getattr(args, "openai_api_key_file", DEFAULT_OPENAI_API_KEY_FILE) or DEFAULT_OPENAI_API_KEY_FILE),
        "--openai-service-tier",
        str(getattr(args, "openai_service_tier", DEFAULT_OPENAI_SERVICE_TIER) or DEFAULT_OPENAI_SERVICE_TIER),
        "--openai-timeout",
        str(getattr(args, "openai_timeout", DEFAULT_OPENAI_TIMEOUT_SECONDS) or DEFAULT_OPENAI_TIMEOUT_SECONDS),
        "--openai-max-retries",
        str(getattr(args, "openai_max_retries", DEFAULT_OPENAI_MAX_RETRIES) or DEFAULT_OPENAI_MAX_RETRIES),
        "--model-id",
        args.model_id,
        "--refinement-model-id",
        args.refinement_model_id,
        "--fallback-model-id",
        args.fallback_model_id,
        "--loop-recovery",
        args.loop_recovery,
        "--windowed-full-image-strategy",
        str(getattr(args, "windowed_full_image_strategy", "visual") or "visual"),
        "--max-boxes",
        str(args.max_boxes),
        "--final-sentences",
        str(args.final_sentences),
        "--window-size",
        str(args.window_size),
        "--window-overlap",
        str(args.window_overlap),
        "--mlx-max-image-side",
        str(attempt_profile["mlx_max_image_side"]),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--prompt",
        args.prompt,
    ]
    if args.request_json:
        cmd.extend(["--request-json", str(args.request_json)])
    artifact_log_bytes = normalize_artifact_log_bytes(
        getattr(args, "max_artifact_log_bytes", DEFAULT_ARTIFACT_LOG_BYTES)
    )
    cmd.extend(["--max-artifact-log-bytes", str(artifact_log_bytes)])
    if getattr(args, "instruction_dataset", False):
        cmd.append("--instruction-dataset")
    if int(getattr(args, "subcaptions_per_image", 0) or 0) > 0:
        cmd.extend(["--subcaptions-per-image", str(args.subcaptions_per_image)])
    if not bool(getattr(args, "include_caption0_in_training", True)):
        cmd.append("--no-include-caption0-in-training")
    if not bool(getattr(args, "include_generated_qa_in_training", True)):
        cmd.append("--no-include-generated-qa-in-training")
    if bool(getattr(args, "include_deterministic_metadata_qa", False)):
        cmd.append("--include-deterministic-metadata-qa")
    for question in _instruction_qa_imposed_questions(args):
        cmd.extend(["--instruction-qa-imposed-question", question])
    if getattr(args, "instruction_max_new_tokens", None) is not None:
        cmd.extend(["--instruction-max-new-tokens", str(args.instruction_max_new_tokens)])
    cmd.extend(
        [
            "--instruction-qa-max-topup-attempts",
            str(_instruction_qa_max_topup_attempts(args)),
        ]
    )
    if _instruction_qa_restrict_speculative_language(args):
        cmd.append("--restrict-speculative-qa-language")
    cmd.extend(["--qa-mix", str(getattr(args, "qa_mix", "balanced") or "balanced")])
    cmd.extend(["--answer-format", str(getattr(args, "answer_format", "natural") or "natural")])
    if not bool(getattr(args, "include_source_annotations_in_generator_context", True)):
        cmd.append("--no-source-annotations-in-generator-context")
    if not bool(getattr(args, "strict_grounding", True)):
        cmd.append("--relaxed-instruction-grounding")
    if args.max_new_tokens is not None:
        cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
    if args.preview_only:
        cmd.append("--preview-only")
    if args.use_sampling:
        cmd.append("--use-sampling")
    return cmd


def _execute_worker_attempt(
    args: argparse.Namespace,
    *,
    dataset_root: Path,
    case: Mapping[str, Any],
    case_index: int,
    key: str,
    image_name: str,
    attempt: int,
    total_attempts: int,
    attempt_profile: Mapping[str, Any],
    attempt_case_path: Path,
    attempt_dir: Path,
    heartbeat_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cmd = _worker_attempt_command(
        args,
        dataset_root=dataset_root,
        attempt_case_path=attempt_case_path,
        attempt_dir=attempt_dir,
        attempt_profile=attempt_profile,
    )
    artifact_log_bytes = normalize_artifact_log_bytes(
        getattr(args, "max_artifact_log_bytes", DEFAULT_ARTIFACT_LOG_BYTES)
    )
    started = time.time()
    heartbeat_interval = max(0.0, float(getattr(args, "heartbeat_interval", 30.0) or 0.0))
    heartbeat_stop = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    if heartbeat_callback is not None and heartbeat_interval > 0:
        worker_progress_mtime = 0.0

        def _beat() -> None:
            nonlocal worker_progress_mtime
            while not heartbeat_stop.wait(heartbeat_interval):
                worker_progress_mtime, worker_progress = read_worker_progress(
                    attempt_dir,
                    last_mtime=worker_progress_mtime,
                )
                fields: dict[str, Any] = {
                    "case_id": key,
                    "case": case.get("name"),
                    "image_name": image_name,
                    "stem": case.get("stem"),
                    "case_index": case_index,
                    "attempt": attempt,
                    "total_attempts": total_attempts,
                    "attempt_started_epoch": started,
                    "attempt_timeout_seconds": float(args.timeout),
                    "artifact_dir": str(attempt_dir),
                    "attempt_profile": dict(attempt_profile),
                    "attempt_mlx_max_image_side": attempt_profile["mlx_max_image_side"],
                }
                if worker_progress:
                    fields.update(
                        {
                            "worker_progress": worker_progress,
                            "worker_progress_seq": worker_progress.get("seq"),
                            "worker_phase": worker_progress.get("phase"),
                            "worker_step_id": worker_progress.get("step_id"),
                            "worker_step_label": worker_progress.get("step_label"),
                            "worker_message": worker_progress.get("message"),
                            "worker_generated_tokens": worker_progress.get("generated_tokens"),
                            "worker_max_new_tokens": worker_progress.get("max_new_tokens"),
                        }
                    )
                heartbeat_callback(fields)

        heartbeat_thread = threading.Thread(
            target=_beat,
            name=f"qwen-caption-parallel-heartbeat-{case_index}-{attempt}",
            daemon=True,
        )
        heartbeat_thread.start()
    timeout = False
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
            check=False,
        )
        returncode: Any = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timeout = True
        returncode = "timeout"
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)
    stdout_meta = write_text_artifact(
        attempt_dir / "stdout.txt",
        stdout or "",
        max_bytes=artifact_log_bytes,
        source="child_stdout",
    )
    stderr_meta = write_text_artifact(
        attempt_dir / "stderr.txt",
        stderr or "",
        max_bytes=artifact_log_bytes,
        source="child_stderr",
    )
    result_path = attempt_dir / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    response_data = result.get("response") or {}
    preview_data = result.get("preview") or {}
    worker_progress = _read_json_dict(attempt_dir / WORKER_PROGRESS_JSON)
    preview_meta = preview_data.get("meta") if isinstance(preview_data, dict) else {}
    preview_prompt_budget = (
        preview_meta.get("prompt_budget")
        if isinstance(preview_meta, dict)
        else None
    )
    quality_failures = caption_quality_failures(result.get("caption_quality"))
    row_status = "timeout" if timeout else (result.get("status") or "missing_result")
    failure_kind = attempt_failure_kind(
        return_code=returncode,
        row_status=str(row_status),
        timeout=timeout,
    )
    signal_fields = return_signal_fields(returncode)
    qwen_caption_io = read_qwen_caption_io_summary(attempt_dir)
    row = {
        "case_id": key,
        "case": case["name"],
        "image_name": image_name,
        "stem": case["stem"],
        "caption_mode": case["caption_mode"],
        "label_count": case["label_count"],
        "class_counts": case["class_counts"],
        "attempt": attempt,
        "attempt_profile": dict(attempt_profile),
        "exit_code": returncode,
        **signal_fields,
        "elapsed_seconds": round(time.time() - started, 3),
        "status": row_status,
        "attempt_failure_kind": failure_kind,
        "used_boxes": response_data.get("used_boxes", preview_data.get("used_boxes")),
        "truncated": response_data.get("truncated", preview_data.get("truncated")),
        "recovery_events": response_data.get("recovery_events", []),
        "preview_full_text_chars": len(str(preview_data.get("full_text") or "")),
        "preview_prompt_budget": preview_prompt_budget,
        "caption_quality": result.get("caption_quality"),
        "generated_qa_pair_count": int(
            ((result.get("instruction_qa") or {}) if isinstance(result.get("instruction_qa"), Mapping) else {}).get("pair_count")
            or len(response_data.get("generated_qa_pairs") or [])
            or 0
        ),
        "generated_qa_target_pair_count": int(
            response_data.get("generated_qa_target_pair_count")
            or ((result.get("instruction_qa") or {}) if isinstance(result.get("instruction_qa"), Mapping) else {}).get("target_pair_count")
            or ((result.get("instruction_qa") or {}) if isinstance(result.get("instruction_qa"), Mapping) else {}).get("requested")
            or 0
        ),
        "generated_qa_rejected_pair_count": int(
            response_data.get("generated_qa_rejected_pair_count")
            or ((result.get("instruction_qa") or {}) if isinstance(result.get("instruction_qa"), Mapping) else {}).get("rejected_pair_count")
            or len(response_data.get("generated_qa_rejected_pairs") or [])
            or 0
        ),
        "instruction_qa_status": (
            ((result.get("instruction_qa") or {}) if isinstance(result.get("instruction_qa"), Mapping) else {}).get("status")
        ),
        "generated_qa_warning": (
            response_data.get("generated_qa_warning")
            if isinstance(response_data.get("generated_qa_warning"), Mapping)
            else result.get("instruction_qa_warning")
            if isinstance(result.get("instruction_qa_warning"), Mapping)
            else None
        ),
        "quality_failures": quality_failures,
        "exception": result.get("exception"),
        "qwen_caption_io": qwen_caption_io,
        "worker_progress": worker_progress,
        "artifact_dir": str(attempt_dir),
        "artifact_limits": {
            "max_artifact_log_bytes": artifact_log_bytes,
            "stdout": stdout_meta,
            "stderr": stderr_meta,
            "worker": result.get("artifact_limits") or {},
        },
    }
    succeeded = row_succeeded(
        row,
        ignore_quality_failures=args.continue_on_quality_failures,
    )
    if not succeeded and failure_kind == "none":
        failure_kind = "quality_or_policy_failure"
        row["attempt_failure_kind"] = failure_kind
    row["final_status"] = "ok" if succeeded else "failed_attempt"
    return row, result if isinstance(result, dict) else {}, response_data if isinstance(response_data, dict) else {}


def _caption_record_from_response(
    *,
    key: str,
    image_name: str,
    case: Mapping[str, Any],
    response_data: Mapping[str, Any],
) -> dict[str, Any] | None:
    caption = str(response_data.get("caption") or "").strip()
    if not caption:
        return None
    caption_record: dict[str, Any] = {
        "case_id": key,
        "image_name": image_name,
        "image_path": case.get("image_path"),
        "caption": caption,
        "used_counts": response_data.get("used_counts") or {},
        "recovery_events": response_data.get("recovery_events") or [],
        "generated_qa_pairs": response_data.get("generated_qa_pairs") or [],
        "generated_qa_pair_count": int(response_data.get("generated_qa_pair_count") or 0),
        "generated_qa_target_pair_count": int(response_data.get("generated_qa_target_pair_count") or 0),
        "generated_qa_rejected_pair_count": int(response_data.get("generated_qa_rejected_pair_count") or 0),
        "generated_qa_attempt_summary": response_data.get("generated_qa_attempt_summary") or [],
    }
    if isinstance(response_data.get("generated_qa_accumulator"), Mapping):
        caption_record["generated_qa_accumulator"] = response_data["generated_qa_accumulator"]
    if isinstance(response_data.get("generated_qa_warning"), Mapping):
        caption_record["generated_qa_warning"] = response_data["generated_qa_warning"]
    return caption_record


def _run_parent_locked(args: argparse.Namespace, runner_lock: ArtifactRunnerLock) -> int:
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases, sample_meta = _select_parent_cases(args, dataset_root)
    results_jsonl = output_root / "results.jsonl"
    captions_jsonl = output_root / "captions.jsonl"
    parent_started = time.time()
    run_settings = run_settings_payload(args)
    existing_manifest_path = output_root / "manifest.json"
    if args.resume and existing_manifest_path.exists():
        existing_manifest = _read_json_dict(existing_manifest_path)
        settings_status, settings_detail = manifest_run_settings_status(existing_manifest, run_settings)
        if settings_status == "mismatch":
            raise SystemExit(f"resume_settings_mismatch:{settings_detail}")
    if not args.resume:
        for path in (results_jsonl, captions_jsonl):
            if path.exists():
                path.unlink()
    try:
        latest_rows = load_latest_rows(results_jsonl)
    except ResultsJsonlError as exc:
        raise SystemExit(f"resume_results_jsonl_invalid:{exc}") from exc
    try:
        validate_captions_jsonl(captions_jsonl)
    except CaptionsJsonlError as exc:
        raise SystemExit(f"resume_captions_jsonl_invalid:{exc}") from exc
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "model_id": args.model_id,
        "runner_capabilities": list(RUNNER_CAPABILITIES),
        "preview_only": args.preview_only,
        "all_images": args.all_images,
        "cases_json": str(args.cases_json or ""),
        "request_json": str(args.request_json or ""),
        "resume": args.resume,
        "attempts": args.attempts,
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "sample_selection": sample_meta,
        "max_boxes": args.max_boxes,
        "max_new_tokens": args.max_new_tokens,
        "summary_row_limit": getattr(args, "summary_row_limit", DEFAULT_SUMMARY_ROW_LIMIT),
        "retry_image_side_scale": getattr(args, "retry_image_side_scale", DEFAULT_RETRY_IMAGE_SIDE_SCALE),
        "min_retry_image_side": getattr(args, "min_retry_image_side", DEFAULT_MIN_RETRY_IMAGE_SIDE),
        "run_settings": run_settings,
        "cases": cases,
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    summary: list[dict[str, Any]] = list(latest_rows.values()) if args.resume else []
    summary_row_limit = int(getattr(args, "summary_row_limit", DEFAULT_SUMMARY_ROW_LIMIT))
    if args.resume:
        write_summary(output_root, summary, row_limit=summary_row_limit)
    exit_code = 0
    failed_cases = 0
    heartbeat_seq = 0
    heartbeat_lock = threading.Lock()
    heartbeat_interval = max(0.0, float(getattr(args, "heartbeat_interval", 30.0) or 0.0))
    restart_ack: dict[str, Any] | None = None

    def emit_heartbeat(phase: str, **fields: Any) -> dict[str, Any]:
        nonlocal heartbeat_seq
        with heartbeat_lock:
            heartbeat_seq += 1
            payload = {
                "seq": heartbeat_seq,
                "status": "running",
                "phase": phase,
                "runner_capabilities": list(RUNNER_CAPABILITIES),
                "processed": len({case_key(row) for row in summary if isinstance(row, Mapping)}),
                "total_cases": len(cases),
                "failed_cases": failed_cases,
                "generated_qa_warning_cases": sum(
                    1 for row in summary if isinstance(row, Mapping) and row_generated_qa_warning(row)
                ),
                "generated_qa_error_cases": sum(
                    1 for row in summary if isinstance(row, Mapping) and row_generated_qa_error(row)
                ),
                **fields,
            }
            write_heartbeat(output_root, payload)
            refresh_lock_from_heartbeat(phase, payload)
            return payload

    def refresh_lock_from_heartbeat(phase: str, heartbeat: Mapping[str, Any]) -> None:
        lock_fields = dict(heartbeat)
        lock_fields.pop("phase", None)
        runner_lock.refresh(phase, **lock_fields)

    emit_heartbeat("started")
    for index, case in enumerate(cases, start=1):
        restart_ack = _consume_runner_restart_request(
            output_root,
            processed=len({case_key(row) for row in summary if isinstance(row, Mapping)}),
            total_cases=len(cases),
        )
        if restart_ack is not None:
            break
        key = case_key(case)
        image_name = Path(str(case.get("image_path") or case.get("name") or "")).name
        previous = latest_rows.get(key)
        reprocess_recovery = bool(getattr(args, "resume_reprocess_recovery_events", False))
        previous_has_recovery = bool(previous and row_has_recovery_events(previous))
        if (
            args.resume
            and previous
            and row_succeeded(previous, ignore_quality_failures=args.continue_on_quality_failures)
            and not (reprocess_recovery and previous_has_recovery)
        ):
            if bool(getattr(args, "record_resume_skips", False)):
                skipped = dict(previous)
                skipped["resumed_skip"] = True
                skipped["final_status"] = "skipped_completed"
                append_jsonl(results_jsonl, skipped)
                summary.append(skipped)
                write_summary(output_root, summary, row_limit=summary_row_limit)
                emit_heartbeat(
                    "case_skipped_completed",
                    case_id=key,
                    case=case.get("name"),
                    image_name=image_name,
                    stem=case.get("stem"),
                    case_index=index,
                    recorded=True,
                )
                print(json.dumps(skipped, sort_keys=True), flush=True)
            continue
        emit_heartbeat(
            "case_start",
            case_id=key,
            case=case.get("name"),
            image_name=image_name,
            stem=case.get("stem"),
            case_index=index,
        )
        if args.skip_existing_captions and caption_exists_for_case(dataset_root, case):
            skipped = {
                "case_id": key,
                "case": case["name"],
                "image_name": image_name,
                "stem": case["stem"],
                "caption_mode": case["caption_mode"],
                "label_count": case["label_count"],
                "class_counts": case["class_counts"],
                "exit_code": 0,
                "status": "skipped_existing_caption",
                "final_status": "skipped_existing_caption",
                "quality_failures": [],
                "artifact_dir": None,
            }
            append_jsonl(results_jsonl, skipped)
            summary.append(skipped)
            write_summary(output_root, summary, row_limit=summary_row_limit)
            emit_heartbeat(
                "case_skipped_existing_caption",
                case_id=key,
                case=case.get("name"),
                image_name=image_name,
                stem=case.get("stem"),
                case_index=index,
            )
            print(json.dumps(skipped, sort_keys=True), flush=True)
            continue
        case_dir = output_root / case_dir_name(index, case)
        case_dir.mkdir(parents=True, exist_ok=True)
        case_path = case_dir / "case.json"
        case_payload_data = dict(case)
        case_payload_data["case_id"] = key
        case_path.write_text(json.dumps(case_payload_data, indent=2, sort_keys=True))
        final_row: dict[str, Any] | None = None
        total_attempts = max(1, int(args.attempts))
        previous_failure_kind: str | None = None
        for attempt in range(1, total_attempts + 1):
            attempt_profile = attempt_generation_profile(
                args,
                attempt=attempt,
                previous_failure_kind=previous_failure_kind,
            )
            attempt_dir = case_dir / f"attempt_{attempt:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            attempt_case_path = attempt_dir / "case.json"
            attempt_case_path.write_text(json.dumps(case_payload_data, indent=2, sort_keys=True))
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--case-json",
                str(attempt_case_path),
                "--output-dir",
                str(attempt_dir),
                "--dataset-root",
                str(dataset_root),
                "--caption-provider",
                str(getattr(args, "caption_provider", CAPTION_PROVIDER_LOCAL) or CAPTION_PROVIDER_LOCAL),
                "--openai-model",
                str(getattr(args, "openai_model", DEFAULT_OPENAI_MODEL) or DEFAULT_OPENAI_MODEL),
                "--openai-image-detail",
                str(getattr(args, "openai_image_detail", DEFAULT_OPENAI_IMAGE_DETAIL) or DEFAULT_OPENAI_IMAGE_DETAIL),
                "--openai-reasoning-effort",
                str(getattr(args, "openai_reasoning_effort", DEFAULT_OPENAI_REASONING_EFFORT) or DEFAULT_OPENAI_REASONING_EFFORT),
                "--openai-api-key-file",
                str(getattr(args, "openai_api_key_file", DEFAULT_OPENAI_API_KEY_FILE) or DEFAULT_OPENAI_API_KEY_FILE),
                "--openai-service-tier",
                str(getattr(args, "openai_service_tier", DEFAULT_OPENAI_SERVICE_TIER) or DEFAULT_OPENAI_SERVICE_TIER),
                "--openai-timeout",
                str(getattr(args, "openai_timeout", DEFAULT_OPENAI_TIMEOUT_SECONDS) or DEFAULT_OPENAI_TIMEOUT_SECONDS),
                "--openai-max-retries",
                str(getattr(args, "openai_max_retries", DEFAULT_OPENAI_MAX_RETRIES) or DEFAULT_OPENAI_MAX_RETRIES),
                "--model-id",
                args.model_id,
                "--refinement-model-id",
                args.refinement_model_id,
                "--fallback-model-id",
                args.fallback_model_id,
                "--loop-recovery",
                args.loop_recovery,
                "--windowed-full-image-strategy",
                str(getattr(args, "windowed_full_image_strategy", "visual") or "visual"),
                "--max-boxes",
                str(args.max_boxes),
                "--final-sentences",
                str(args.final_sentences),
                "--window-size",
                str(args.window_size),
                "--window-overlap",
                str(args.window_overlap),
                "--mlx-max-image-side",
                str(attempt_profile["mlx_max_image_side"]),
                "--temperature",
                str(args.temperature),
                "--top-p",
                str(args.top_p),
                "--top-k",
                str(args.top_k),
                "--prompt",
                args.prompt,
            ]
            if args.request_json:
                cmd.extend(["--request-json", str(args.request_json)])
            artifact_log_bytes = normalize_artifact_log_bytes(
                getattr(args, "max_artifact_log_bytes", DEFAULT_ARTIFACT_LOG_BYTES)
            )
            cmd.extend(["--max-artifact-log-bytes", str(artifact_log_bytes)])
            if getattr(args, "instruction_dataset", False):
                cmd.append("--instruction-dataset")
            if int(getattr(args, "subcaptions_per_image", 0) or 0) > 0:
                cmd.extend(["--subcaptions-per-image", str(args.subcaptions_per_image)])
            if not bool(getattr(args, "include_caption0_in_training", True)):
                cmd.append("--no-include-caption0-in-training")
            if not bool(getattr(args, "include_generated_qa_in_training", True)):
                cmd.append("--no-include-generated-qa-in-training")
            if bool(getattr(args, "include_deterministic_metadata_qa", False)):
                cmd.append("--include-deterministic-metadata-qa")
            for question in _instruction_qa_imposed_questions(args):
                cmd.extend(["--instruction-qa-imposed-question", question])
            if getattr(args, "instruction_max_new_tokens", None) is not None:
                cmd.extend(["--instruction-max-new-tokens", str(args.instruction_max_new_tokens)])
            cmd.extend(
                [
                    "--instruction-qa-max-topup-attempts",
                    str(_instruction_qa_max_topup_attempts(args)),
                ]
            )
            if _instruction_qa_restrict_speculative_language(args):
                cmd.append("--restrict-speculative-qa-language")
            cmd.extend(["--qa-mix", str(getattr(args, "qa_mix", "balanced") or "balanced")])
            cmd.extend(["--answer-format", str(getattr(args, "answer_format", "natural") or "natural")])
            if not bool(getattr(args, "include_source_annotations_in_generator_context", True)):
                cmd.append("--no-source-annotations-in-generator-context")
            if not bool(getattr(args, "strict_grounding", True)):
                cmd.append("--relaxed-instruction-grounding")
            if args.max_new_tokens is not None:
                cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
            if args.preview_only:
                cmd.append("--preview-only")
            if args.use_sampling:
                cmd.append("--use-sampling")
            started = time.time()
            attempt_heartbeat = emit_heartbeat(
                "attempt_running",
                case_id=key,
                case=case.get("name"),
                image_name=image_name,
                stem=case.get("stem"),
                case_index=index,
                attempt=attempt,
                total_attempts=total_attempts,
                attempt_started_epoch=started,
                attempt_timeout_seconds=float(args.timeout),
                artifact_dir=str(attempt_dir),
                attempt_profile=attempt_profile,
                attempt_mlx_max_image_side=attempt_profile["mlx_max_image_side"],
            )
            heartbeat_stop = threading.Event()
            heartbeat_thread: threading.Thread | None = None
            if heartbeat_interval > 0:
                worker_progress_mtime = 0.0

                def _beat() -> None:
                    nonlocal attempt_heartbeat, worker_progress_mtime
                    while not heartbeat_stop.wait(heartbeat_interval):
                        worker_progress_mtime, worker_progress = read_worker_progress(
                            attempt_dir,
                            last_mtime=worker_progress_mtime,
                        )
                        if worker_progress:
                            attempt_heartbeat = emit_heartbeat(
                                "attempt_running",
                                case_id=key,
                                case=case.get("name"),
                                image_name=image_name,
                                stem=case.get("stem"),
                                case_index=index,
                                attempt=attempt,
                                total_attempts=total_attempts,
                                attempt_started_epoch=started,
                                attempt_timeout_seconds=float(args.timeout),
                                artifact_dir=str(attempt_dir),
                                attempt_profile=attempt_profile,
                                attempt_mlx_max_image_side=attempt_profile["mlx_max_image_side"],
                                worker_progress=worker_progress,
                                worker_progress_seq=worker_progress.get("seq"),
                                worker_phase=worker_progress.get("phase"),
                                worker_step_id=worker_progress.get("step_id"),
                                worker_step_label=worker_progress.get("step_label"),
                                worker_message=worker_progress.get("message"),
                                worker_generated_tokens=worker_progress.get("generated_tokens"),
                                worker_max_new_tokens=worker_progress.get("max_new_tokens"),
                            )
                        else:
                            write_heartbeat(output_root, attempt_heartbeat)
                            refresh_lock_from_heartbeat("attempt_running", attempt_heartbeat)

                heartbeat_thread = threading.Thread(
                    target=_beat,
                    name=f"qwen-caption-benchmark-heartbeat-{index}-{attempt}",
                    daemon=True,
                )
                heartbeat_thread.start()
            timeout = False
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=str(REPO_ROOT),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=args.timeout,
                    check=False,
                )
                returncode: Any = completed.returncode
                stdout = completed.stdout
                stderr = completed.stderr
            except subprocess.TimeoutExpired as exc:
                timeout = True
                returncode = "timeout"
                stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
                stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
            finally:
                heartbeat_stop.set()
                if heartbeat_thread is not None:
                    heartbeat_thread.join(timeout=1.0)
            stdout_meta = write_text_artifact(
                attempt_dir / "stdout.txt",
                stdout or "",
                max_bytes=artifact_log_bytes,
                source="child_stdout",
            )
            stderr_meta = write_text_artifact(
                attempt_dir / "stderr.txt",
                stderr or "",
                max_bytes=artifact_log_bytes,
                source="child_stderr",
            )
            result_path = attempt_dir / "result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            response_data = result.get("response") or {}
            preview_data = result.get("preview") or {}
            worker_progress = _read_json_dict(attempt_dir / WORKER_PROGRESS_JSON)
            preview_meta = preview_data.get("meta") if isinstance(preview_data, dict) else {}
            preview_prompt_budget = (
                preview_meta.get("prompt_budget")
                if isinstance(preview_meta, dict)
                else None
            )
            quality_failures = caption_quality_failures(result.get("caption_quality"))
            row_status = "timeout" if timeout else (result.get("status") or "missing_result")
            failure_kind = attempt_failure_kind(
                return_code=returncode,
                row_status=str(row_status),
                timeout=timeout,
            )
            signal_fields = return_signal_fields(returncode)
            qwen_caption_io = read_qwen_caption_io_summary(attempt_dir)
            row = {
                "case_id": key,
                "case": case["name"],
                "image_name": image_name,
                "stem": case["stem"],
                "caption_mode": case["caption_mode"],
                "label_count": case["label_count"],
                "class_counts": case["class_counts"],
                "attempt": attempt,
                "attempt_profile": attempt_profile,
                "exit_code": returncode,
                **signal_fields,
                "elapsed_seconds": round(time.time() - started, 3),
                "status": row_status,
                "attempt_failure_kind": failure_kind,
                "used_boxes": response_data.get("used_boxes", preview_data.get("used_boxes")),
                "truncated": response_data.get("truncated", preview_data.get("truncated")),
                "recovery_events": response_data.get("recovery_events", []),
                "preview_full_text_chars": len(str(preview_data.get("full_text") or "")),
                "preview_prompt_budget": preview_prompt_budget,
                "caption_quality": result.get("caption_quality"),
                "generated_qa_pair_count": int(
                    ((result.get("instruction_qa") or {}) if isinstance(result.get("instruction_qa"), Mapping) else {}).get("pair_count")
                    or len(response_data.get("generated_qa_pairs") or [])
                    or 0
                ),
                "generated_qa_target_pair_count": int(
                    response_data.get("generated_qa_target_pair_count")
                    or ((result.get("instruction_qa") or {}) if isinstance(result.get("instruction_qa"), Mapping) else {}).get("target_pair_count")
                    or ((result.get("instruction_qa") or {}) if isinstance(result.get("instruction_qa"), Mapping) else {}).get("requested")
                    or 0
                ),
                "generated_qa_rejected_pair_count": int(
                    response_data.get("generated_qa_rejected_pair_count")
                    or ((result.get("instruction_qa") or {}) if isinstance(result.get("instruction_qa"), Mapping) else {}).get("rejected_pair_count")
                    or len(response_data.get("generated_qa_rejected_pairs") or [])
                    or 0
                ),
                "instruction_qa_status": (
                    ((result.get("instruction_qa") or {}) if isinstance(result.get("instruction_qa"), Mapping) else {}).get("status")
                ),
                "generated_qa_warning": (
                    response_data.get("generated_qa_warning")
                    if isinstance(response_data.get("generated_qa_warning"), Mapping)
                    else result.get("instruction_qa_warning")
                    if isinstance(result.get("instruction_qa_warning"), Mapping)
                    else None
                ),
                "quality_failures": quality_failures,
                "exception": result.get("exception"),
                "qwen_caption_io": qwen_caption_io,
                "worker_progress": worker_progress,
                "artifact_dir": str(attempt_dir),
                "artifact_limits": {
                    "max_artifact_log_bytes": artifact_log_bytes,
                    "stdout": stdout_meta,
                    "stderr": stderr_meta,
                    "worker": result.get("artifact_limits") or {},
                },
            }
            succeeded = row_succeeded(
                row,
                ignore_quality_failures=args.continue_on_quality_failures,
            )
            if not succeeded and failure_kind == "none":
                failure_kind = "quality_or_policy_failure"
                row["attempt_failure_kind"] = failure_kind
            row["final_status"] = "ok" if succeeded else "failed_attempt"
            next_attempt_cooldown = 0.0
            if not succeeded and attempt < total_attempts:
                next_attempt_cooldown = attempt_cooldown_seconds(
                    args,
                    failed_attempt=attempt,
                    failure_kind=failure_kind,
                )
                row["next_attempt_cooldown_seconds"] = next_attempt_cooldown
            if succeeded:
                caption = str(response_data.get("caption") or "").strip()
                if caption:
                    if args.save_dataset_text_labels:
                        saved_path = save_caption_for_case(dataset_root, case, caption)
                        row["saved_text_label"] = str(saved_path)
                    caption_record = {
                        "case_id": key,
                        "image_name": image_name,
                        "image_path": case.get("image_path"),
                        "caption": caption,
                        "used_counts": response_data.get("used_counts") or {},
                        "recovery_events": response_data.get("recovery_events") or [],
                        "generated_qa_pairs": response_data.get("generated_qa_pairs") or [],
                        "generated_qa_pair_count": int(
                            response_data.get("generated_qa_pair_count") or 0
                        ),
                        "generated_qa_target_pair_count": int(
                            response_data.get("generated_qa_target_pair_count") or 0
                        ),
                        "generated_qa_rejected_pair_count": int(
                            response_data.get("generated_qa_rejected_pair_count") or 0
                        ),
                        "generated_qa_attempt_summary": response_data.get(
                            "generated_qa_attempt_summary"
                        )
                        or [],
                    }
                    if isinstance(response_data.get("generated_qa_accumulator"), Mapping):
                        caption_record["generated_qa_accumulator"] = response_data[
                            "generated_qa_accumulator"
                        ]
                    if isinstance(response_data.get("generated_qa_warning"), Mapping):
                        caption_record["generated_qa_warning"] = response_data["generated_qa_warning"]
                    append_jsonl(captions_jsonl, caption_record)
            append_jsonl(results_jsonl, row)
            print(json.dumps(row, sort_keys=True), flush=True)
            final_row = row
            if succeeded:
                break
            previous_failure_kind = failure_kind
            if next_attempt_cooldown > 0:
                emit_heartbeat(
                    "attempt_cooldown",
                    case_id=key,
                    case=case.get("name"),
                    image_name=image_name,
                    stem=case.get("stem"),
                    case_index=index,
                    attempt=attempt,
                    next_attempt=attempt + 1,
                    total_attempts=total_attempts,
                    cooldown_seconds=next_attempt_cooldown,
                    attempt_failure_kind=failure_kind,
                    return_signal=signal_fields.get("return_signal"),
                    return_signal_name=signal_fields.get("return_signal_name"),
                )
                time.sleep(next_attempt_cooldown)
        if final_row is None:
            continue
        if final_row.get("final_status") != "ok":
            generated_qa_is_required = _generated_qa_blocks_parent_deterministic_recovery(args)
            recovery_eligible, recovery_eligibility_reason = parent_deterministic_recovery_eligibility(final_row)
            allow_parent_deterministic_recovery = bool(
                not generated_qa_is_required and recovery_eligible
            )
            deterministic_recovery = (
                parent_deterministic_recovery_caption(case)
                if allow_parent_deterministic_recovery
                else None
            )
            if deterministic_recovery is None:
                final_row = dict(final_row)
                final_row["final_status"] = "failed"
                final_row["terminal_failure"] = True
                if generated_qa_is_required:
                    final_row["parent_deterministic_recovery_skipped"] = {
                        "reason": "generated_qa_required",
                        "detail": (
                            "Parent deterministic count/layout recovery cannot satisfy "
                            "a generated-QA instruction dataset case."
                        ),
                    }
                elif not recovery_eligible:
                    final_row["parent_deterministic_recovery_skipped"] = {
                        "reason": recovery_eligibility_reason,
                        "detail": (
                            "Parent deterministic count/layout recovery is only for "
                            "model/runtime failures; request validation and code/config "
                            "errors must remain visible."
                        ),
                    }
                append_jsonl(results_jsonl, final_row)
                print(json.dumps(final_row, sort_keys=True), flush=True)
                failed_cases += 1
                if args.max_failures:
                    exit_code = 1
            else:
                caption, recovered_counts = deterministic_recovery
                recovery_event = {
                    "action": "deterministic_recovery_succeeded",
                    "attempt": "parent_deterministic_recovery",
                    "call_kind": "deterministic",
                    "stage_label": "Parent exhausted-attempt recovery",
                    "message": (
                        "Recovered case with deterministic count/layout fallback after "
                        "child attempts were exhausted."
                    ),
                    "detail": (
                        "The fallback used authoritative case counts so unattended "
                        "runs can continue without manual image recovery."
                    ),
                }
                quality = summarize_caption(caption, recovered_counts)
                quality_failures = caption_quality_failures(quality)
                recovered_row = dict(final_row)
                recovered_row.update(
                    {
                        "source_attempt_exit_code": final_row.get("exit_code"),
                        "source_attempt_status": final_row.get("status"),
                        "source_attempt_failure_kind": final_row.get("attempt_failure_kind"),
                        "source_attempt_exception": final_row.get("exception"),
                        "exit_code": 0,
                        "return_signal": None,
                        "return_signal_name": None,
                        "status": "ok",
                        "attempt_failure_kind": "parent_deterministic_recovery",
                        "final_status": "ok",
                        "terminal_failure": False,
                        "parent_deterministic_recovery": True,
                        "caption_quality": quality,
                        "quality_failures": quality_failures,
                        "recovery_events": list(final_row.get("recovery_events") or [])
                        + [recovery_event],
                    }
                )
                if not quality_failures or args.continue_on_quality_failures:
                    if args.save_dataset_text_labels:
                        saved_path = save_caption_for_case(dataset_root, case, caption)
                        recovered_row["saved_text_label"] = str(saved_path)
                    append_jsonl(results_jsonl, recovered_row)
                    print(json.dumps(recovered_row, sort_keys=True), flush=True)
                    append_jsonl(
                        captions_jsonl,
                        {
                            "case_id": key,
                            "image_name": image_name,
                            "image_path": case.get("image_path"),
                            "caption": caption,
                            "used_counts": recovered_counts,
                            "recovery_events": recovered_row["recovery_events"],
                        },
                    )
                    final_row = recovered_row
                else:
                    final_row = dict(final_row)
                    final_row["final_status"] = "failed"
                    final_row["terminal_failure"] = True
                    final_row["parent_deterministic_recovery_rejected"] = quality_failures
                    append_jsonl(results_jsonl, final_row)
                    print(json.dumps(final_row, sort_keys=True), flush=True)
                    failed_cases += 1
                    if args.max_failures:
                        exit_code = 1
        summary.append(final_row)
        write_summary(output_root, summary, row_limit=summary_row_limit)
        emit_heartbeat(
            "case_complete",
            case_id=key,
            case=case.get("name"),
            image_name=image_name,
            stem=case.get("stem"),
            case_index=index,
            final_status=final_row.get("final_status"),
        )
        if args.max_failures and failed_cases >= args.max_failures:
            exit_code = 1
            break
        success_cooldown = max(0.0, float(getattr(args, "cooldown_after_success", 0.0) or 0.0))
        if (
            success_cooldown > 0
            and str(final_row.get("final_status") or "").lower() == "ok"
            and index < len(cases)
        ):
            emit_heartbeat(
                "case_success_cooldown",
                case_id=key,
                case=case.get("name"),
                image_name=image_name,
                stem=case.get("stem"),
                case_index=index,
                next_case_index=index + 1,
                cooldown_seconds=success_cooldown,
            )
            time.sleep(success_cooldown)
    final_heartbeat = {
        "seq": heartbeat_seq + 1,
        "status": "restart_requested" if restart_ack is not None else "completed" if exit_code == 0 else "failed",
        "phase": "restart_requested" if restart_ack is not None else "finished",
        "runner_capabilities": list(RUNNER_CAPABILITIES),
        "processed": len({case_key(row) for row in summary if isinstance(row, Mapping)}),
        "total_cases": len(cases),
        "failed_cases": failed_cases,
        "generated_qa_warning_cases": sum(
            1 for row in summary if isinstance(row, Mapping) and row_generated_qa_warning(row)
        ),
        "generated_qa_error_cases": sum(
            1 for row in summary if isinstance(row, Mapping) and row_generated_qa_error(row)
        ),
        "exit_code": exit_code,
    }
    if restart_ack is not None:
        final_heartbeat["restart_request"] = restart_ack
    write_summary(
        output_root,
        summary,
        row_limit=summary_row_limit,
        extra={
            "status": final_heartbeat["status"],
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.time() - parent_started, 3),
            "exit_code": exit_code,
        },
    )
    write_heartbeat(output_root, final_heartbeat)
    lock_fields = dict(final_heartbeat)
    lock_fields.pop("phase", None)
    runner_lock.refresh("finished", **lock_fields)
    return exit_code


def _run_parent_parallel_locked(args: argparse.Namespace, runner_lock: ArtifactRunnerLock) -> int:
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases, sample_meta = _select_parent_cases(args, dataset_root)
    parallel_cases = max(1, min(int(getattr(args, "parallel_cases", 1) or 1), len(cases)))
    results_jsonl = output_root / "results.jsonl"
    captions_jsonl = output_root / "captions.jsonl"
    parent_started = time.time()
    run_settings = run_settings_payload(args)
    existing_manifest_path = output_root / "manifest.json"
    if args.resume and existing_manifest_path.exists():
        existing_manifest = _read_json_dict(existing_manifest_path)
        settings_status, settings_detail = manifest_run_settings_status(existing_manifest, run_settings)
        if settings_status == "mismatch":
            raise SystemExit(f"resume_settings_mismatch:{settings_detail}")
    if not args.resume:
        for path in (results_jsonl, captions_jsonl):
            if path.exists():
                path.unlink()
    try:
        latest_rows = load_latest_rows(results_jsonl)
    except ResultsJsonlError as exc:
        raise SystemExit(f"resume_results_jsonl_invalid:{exc}") from exc
    try:
        validate_captions_jsonl(captions_jsonl)
    except CaptionsJsonlError as exc:
        raise SystemExit(f"resume_captions_jsonl_invalid:{exc}") from exc
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "model_id": args.model_id,
        "runner_capabilities": list(RUNNER_CAPABILITIES),
        "preview_only": args.preview_only,
        "all_images": args.all_images,
        "cases_json": str(args.cases_json or ""),
        "request_json": str(args.request_json or ""),
        "resume": args.resume,
        "attempts": args.attempts,
        "parallel_cases": parallel_cases,
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "sample_selection": sample_meta,
        "max_boxes": args.max_boxes,
        "max_new_tokens": args.max_new_tokens,
        "summary_row_limit": getattr(args, "summary_row_limit", DEFAULT_SUMMARY_ROW_LIMIT),
        "retry_image_side_scale": getattr(args, "retry_image_side_scale", DEFAULT_RETRY_IMAGE_SIDE_SCALE),
        "min_retry_image_side": getattr(args, "min_retry_image_side", DEFAULT_MIN_RETRY_IMAGE_SIDE),
        "run_settings": run_settings,
        "cases": cases,
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    summary: list[dict[str, Any]] = list(latest_rows.values()) if args.resume else []
    summary_by_key: dict[str, dict[str, Any]] = {
        key: row for key, row in latest_rows.items()
    } if args.resume else {}
    summary_row_limit = int(getattr(args, "summary_row_limit", DEFAULT_SUMMARY_ROW_LIMIT))
    if args.resume:
        write_summary(output_root, summary, row_limit=summary_row_limit)

    heartbeat_seq = 0
    exit_code = 0
    restart_ack: dict[str, Any] | None = None
    state_lock = threading.RLock()
    heartbeat_lock = threading.Lock()
    print_lock = threading.Lock()
    active_cases: dict[str, dict[str, Any]] = {}
    state = {"failed_cases": 0, "submitted_cases": 0}

    def _summary_processed_count() -> int:
        return len({case_key(row) for row in summary if isinstance(row, Mapping)})

    def emit_heartbeat(phase: str, **fields: Any) -> dict[str, Any]:
        nonlocal heartbeat_seq
        with heartbeat_lock:
            with state_lock:
                active_snapshot = list(active_cases.values())
                processed = _summary_processed_count()
                failed_cases = int(state["failed_cases"])
                submitted_cases = int(state["submitted_cases"])
            heartbeat_seq += 1
            payload = {
                "seq": heartbeat_seq,
                "status": "running",
                "phase": phase,
                "runner_capabilities": list(RUNNER_CAPABILITIES),
                "parallel_cases": parallel_cases,
                "submitted_cases": submitted_cases,
                "in_flight_cases": len(active_snapshot),
                "active_cases": active_snapshot[: min(20, parallel_cases)],
                "processed": processed,
                "total_cases": len(cases),
                "failed_cases": failed_cases,
                "generated_qa_warning_cases": sum(
                    1 for row in summary if isinstance(row, Mapping) and row_generated_qa_warning(row)
                ),
                "generated_qa_error_cases": sum(
                    1 for row in summary if isinstance(row, Mapping) and row_generated_qa_error(row)
                ),
                **fields,
            }
            write_heartbeat(output_root, payload)
            lock_fields = dict(payload)
            lock_fields.pop("phase", None)
            runner_lock.refresh(phase, **lock_fields)
            return payload

    def _set_active(key: str, payload: Mapping[str, Any]) -> None:
        with state_lock:
            active_cases[key] = {
                "case_id": key,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **dict(payload),
            }

    def _clear_active(key: str) -> None:
        with state_lock:
            active_cases.pop(key, None)

    def _append_result_row(row: Mapping[str, Any]) -> None:
        with state_lock:
            append_jsonl(results_jsonl, row)
        with print_lock:
            print(json.dumps(dict(row), sort_keys=True), flush=True)

    def _append_caption_record(record: Mapping[str, Any]) -> None:
        with state_lock:
            append_jsonl(captions_jsonl, record)

    def _record_final_row(row: Mapping[str, Any]) -> None:
        with state_lock:
            final_row = dict(row)
            key = str(final_row.get("case_id") or "").strip()
            if key:
                summary_by_key[key] = final_row
            summary.append(final_row)
            write_summary(output_root, summary, row_limit=summary_row_limit)

    def _process_case(index: int, case: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal exit_code
        key = case_key(case)
        image_name = Path(str(case.get("image_path") or case.get("name") or "")).name
        _set_active(
            key,
            {
                "case": case.get("name"),
                "image_name": image_name,
                "stem": case.get("stem"),
                "case_index": index,
                "phase": "case_start",
            },
        )
        emit_heartbeat(
            "case_start",
            case_id=key,
            case=case.get("name"),
            image_name=image_name,
            stem=case.get("stem"),
            case_index=index,
        )
        try:
            previous = latest_rows.get(key)
            reprocess_recovery = bool(getattr(args, "resume_reprocess_recovery_events", False))
            previous_has_recovery = bool(previous and row_has_recovery_events(previous))
            if (
                args.resume
                and previous
                and row_succeeded(previous, ignore_quality_failures=args.continue_on_quality_failures)
                and not (reprocess_recovery and previous_has_recovery)
            ):
                if bool(getattr(args, "record_resume_skips", False)):
                    skipped = dict(previous)
                    skipped["resumed_skip"] = True
                    skipped["final_status"] = "skipped_completed"
                    _append_result_row(skipped)
                    _record_final_row(skipped)
                    emit_heartbeat(
                        "case_skipped_completed",
                        case_id=key,
                        case=case.get("name"),
                        image_name=image_name,
                        stem=case.get("stem"),
                        case_index=index,
                        recorded=True,
                    )
                    return skipped
                return None
            if args.skip_existing_captions and caption_exists_for_case(dataset_root, case):
                skipped = {
                    "case_id": key,
                    "case": case["name"],
                    "image_name": image_name,
                    "stem": case["stem"],
                    "caption_mode": case["caption_mode"],
                    "label_count": case["label_count"],
                    "class_counts": case["class_counts"],
                    "exit_code": 0,
                    "status": "skipped_existing_caption",
                    "final_status": "skipped_existing_caption",
                    "quality_failures": [],
                    "artifact_dir": None,
                }
                _append_result_row(skipped)
                _record_final_row(skipped)
                emit_heartbeat(
                    "case_skipped_existing_caption",
                    case_id=key,
                    case=case.get("name"),
                    image_name=image_name,
                    stem=case.get("stem"),
                    case_index=index,
                )
                return skipped
            case_dir = output_root / case_dir_name(index, case)
            case_dir.mkdir(parents=True, exist_ok=True)
            case_payload_data = dict(case)
            case_payload_data["case_id"] = key
            (case_dir / "case.json").write_text(json.dumps(case_payload_data, indent=2, sort_keys=True))
            final_row: dict[str, Any] | None = None
            final_response_data: dict[str, Any] = {}
            total_attempts = max(1, int(args.attempts))
            previous_failure_kind: str | None = None
            for attempt in range(1, total_attempts + 1):
                attempt_profile = attempt_generation_profile(
                    args,
                    attempt=attempt,
                    previous_failure_kind=previous_failure_kind,
                )
                attempt_dir = case_dir / f"attempt_{attempt:02d}"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                attempt_case_path = attempt_dir / "case.json"
                attempt_case_path.write_text(json.dumps(case_payload_data, indent=2, sort_keys=True))
                started = time.time()
                _set_active(
                    key,
                    {
                        "case": case.get("name"),
                        "image_name": image_name,
                        "stem": case.get("stem"),
                        "case_index": index,
                        "phase": "attempt_running",
                        "attempt": attempt,
                        "total_attempts": total_attempts,
                        "attempt_started_epoch": started,
                        "artifact_dir": str(attempt_dir),
                    },
                )
                emit_heartbeat(
                    "attempt_running",
                    case_id=key,
                    case=case.get("name"),
                    image_name=image_name,
                    stem=case.get("stem"),
                    case_index=index,
                    attempt=attempt,
                    total_attempts=total_attempts,
                    attempt_started_epoch=started,
                    attempt_timeout_seconds=float(args.timeout),
                    artifact_dir=str(attempt_dir),
                    attempt_profile=attempt_profile,
                    attempt_mlx_max_image_side=attempt_profile["mlx_max_image_side"],
                )

                def _attempt_progress(fields: Mapping[str, Any]) -> None:
                    _set_active(
                        key,
                        {
                            "case": case.get("name"),
                            "image_name": image_name,
                            "stem": case.get("stem"),
                            "case_index": index,
                            "phase": "attempt_running",
                            "attempt": attempt,
                            "total_attempts": total_attempts,
                            "artifact_dir": str(attempt_dir),
                            "worker_step_label": fields.get("worker_step_label"),
                            "worker_message": fields.get("worker_message"),
                        },
                    )
                    emit_heartbeat("attempt_running", **dict(fields))

                row, _result, response_data = _execute_worker_attempt(
                    args,
                    dataset_root=dataset_root,
                    case=case,
                    case_index=index,
                    key=key,
                    image_name=image_name,
                    attempt=attempt,
                    total_attempts=total_attempts,
                    attempt_profile=attempt_profile,
                    attempt_case_path=attempt_case_path,
                    attempt_dir=attempt_dir,
                    heartbeat_callback=_attempt_progress,
                )
                succeeded = row_succeeded(
                    row,
                    ignore_quality_failures=args.continue_on_quality_failures,
                )
                next_attempt_cooldown = 0.0
                if not succeeded and attempt < total_attempts:
                    next_attempt_cooldown = attempt_cooldown_seconds(
                        args,
                        failed_attempt=attempt,
                        failure_kind=str(row.get("attempt_failure_kind") or "none"),
                    )
                    row["next_attempt_cooldown_seconds"] = next_attempt_cooldown
                _append_result_row(row)
                final_row = dict(row)
                final_response_data = dict(response_data)
                if succeeded:
                    caption = str(response_data.get("caption") or "").strip()
                    if caption:
                        if args.save_dataset_text_labels:
                            saved_path = save_caption_for_case(dataset_root, case, caption)
                            final_row["saved_text_label"] = str(saved_path)
                        caption_record = _caption_record_from_response(
                            key=key,
                            image_name=image_name,
                            case=case,
                            response_data=response_data,
                        )
                        if caption_record is not None:
                            _append_caption_record(caption_record)
                    break
                previous_failure_kind = str(row.get("attempt_failure_kind") or "none")
                if next_attempt_cooldown > 0:
                    emit_heartbeat(
                        "attempt_cooldown",
                        case_id=key,
                        case=case.get("name"),
                        image_name=image_name,
                        stem=case.get("stem"),
                        case_index=index,
                        attempt=attempt,
                        next_attempt=attempt + 1,
                        total_attempts=total_attempts,
                        cooldown_seconds=next_attempt_cooldown,
                        attempt_failure_kind=previous_failure_kind,
                        return_signal=row.get("return_signal"),
                        return_signal_name=row.get("return_signal_name"),
                    )
                    time.sleep(next_attempt_cooldown)
            if final_row is None:
                return None
            if final_row.get("final_status") != "ok":
                generated_qa_is_required = _generated_qa_blocks_parent_deterministic_recovery(args)
                recovery_eligible, recovery_eligibility_reason = parent_deterministic_recovery_eligibility(final_row)
                allow_parent_deterministic_recovery = bool(
                    not generated_qa_is_required and recovery_eligible
                )
                deterministic_recovery = (
                    parent_deterministic_recovery_caption(case)
                    if allow_parent_deterministic_recovery
                    else None
                )
                if deterministic_recovery is None:
                    final_row = dict(final_row)
                    final_row["final_status"] = "failed"
                    final_row["terminal_failure"] = True
                    if generated_qa_is_required:
                        final_row["parent_deterministic_recovery_skipped"] = {
                            "reason": "generated_qa_required",
                            "detail": (
                                "Parent deterministic count/layout recovery cannot satisfy "
                                "a generated-QA instruction dataset case."
                            ),
                        }
                    elif not recovery_eligible:
                        final_row["parent_deterministic_recovery_skipped"] = {
                            "reason": recovery_eligibility_reason,
                            "detail": (
                                "Parent deterministic count/layout recovery is only for "
                                "model/runtime failures; request validation and code/config "
                                "errors must remain visible."
                            ),
                        }
                    _append_result_row(final_row)
                    with state_lock:
                        state["failed_cases"] = int(state["failed_cases"]) + 1
                    if args.max_failures:
                        exit_code = 1
                else:
                    caption, recovered_counts = deterministic_recovery
                    recovery_event = {
                        "action": "deterministic_recovery_succeeded",
                        "attempt": "parent_deterministic_recovery",
                        "call_kind": "deterministic",
                        "stage_label": "Parent exhausted-attempt recovery",
                        "message": (
                            "Recovered case with deterministic count/layout fallback after "
                            "child attempts were exhausted."
                        ),
                        "detail": (
                            "The fallback used authoritative case counts so unattended "
                            "runs can continue without manual image recovery."
                        ),
                    }
                    quality = summarize_caption(caption, recovered_counts)
                    quality_failures = caption_quality_failures(quality)
                    recovered_row = dict(final_row)
                    recovered_row.update(
                        {
                            "source_attempt_exit_code": final_row.get("exit_code"),
                            "source_attempt_status": final_row.get("status"),
                            "source_attempt_failure_kind": final_row.get("attempt_failure_kind"),
                            "source_attempt_exception": final_row.get("exception"),
                            "exit_code": 0,
                            "return_signal": None,
                            "return_signal_name": None,
                            "status": "ok",
                            "attempt_failure_kind": "parent_deterministic_recovery",
                            "final_status": "ok",
                            "terminal_failure": False,
                            "parent_deterministic_recovery": True,
                            "caption_quality": quality,
                            "quality_failures": quality_failures,
                            "recovery_events": list(final_row.get("recovery_events") or [])
                            + [recovery_event],
                        }
                    )
                    if not quality_failures or args.continue_on_quality_failures:
                        if args.save_dataset_text_labels:
                            saved_path = save_caption_for_case(dataset_root, case, caption)
                            recovered_row["saved_text_label"] = str(saved_path)
                        _append_result_row(recovered_row)
                        _append_caption_record(
                            {
                                "case_id": key,
                                "image_name": image_name,
                                "image_path": case.get("image_path"),
                                "caption": caption,
                                "used_counts": recovered_counts,
                                "recovery_events": recovered_row["recovery_events"],
                            }
                        )
                        final_row = recovered_row
                    else:
                        final_row = dict(final_row)
                        final_row["final_status"] = "failed"
                        final_row["terminal_failure"] = True
                        final_row["parent_deterministic_recovery_rejected"] = quality_failures
                        _append_result_row(final_row)
                        with state_lock:
                            state["failed_cases"] = int(state["failed_cases"]) + 1
                        if args.max_failures:
                            exit_code = 1
            _record_final_row(final_row)
            emit_heartbeat(
                "case_complete",
                case_id=key,
                case=case.get("name"),
                image_name=image_name,
                stem=case.get("stem"),
                case_index=index,
                final_status=final_row.get("final_status"),
                generated_qa_pair_count=final_row.get("generated_qa_pair_count"),
                generated_qa_target_pair_count=final_row.get("generated_qa_target_pair_count"),
            )
            return final_row
        except BaseException as exc:  # noqa: BLE001
            failed = {
                "case_id": key,
                "case": case.get("name"),
                "image_name": image_name,
                "stem": case.get("stem"),
                "caption_mode": case.get("caption_mode"),
                "label_count": case.get("label_count"),
                "class_counts": case.get("class_counts"),
                "exit_code": 1,
                "status": "parent_exception",
                "final_status": "failed",
                "terminal_failure": True,
                "exception": {"type": type(exc).__name__, "message": str(exc)},
                "quality_failures": [],
                "artifact_dir": None,
            }
            _append_result_row(failed)
            _record_final_row(failed)
            with state_lock:
                state["failed_cases"] = int(state["failed_cases"]) + 1
            if args.max_failures:
                exit_code = 1
            emit_heartbeat(
                "case_parent_exception",
                case_id=key,
                case=case.get("name"),
                image_name=image_name,
                stem=case.get("stem"),
                case_index=index,
                exception=failed["exception"],
            )
            return failed
        finally:
            _clear_active(key)

    emit_heartbeat("started")
    pending_index = 0
    stop_submitting = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_cases) as executor:
        futures: dict[concurrent.futures.Future[dict[str, Any] | None], tuple[int, dict[str, Any]]] = {}

        def _submit_available() -> None:
            nonlocal pending_index, restart_ack, stop_submitting
            while len(futures) < parallel_cases and pending_index < len(cases) and not stop_submitting:
                with state_lock:
                    processed = _summary_processed_count()
                restart_ack = _consume_runner_restart_request(
                    output_root,
                    processed=processed,
                    total_cases=len(cases),
                )
                if restart_ack is not None:
                    stop_submitting = True
                    break
                if args.max_failures and int(state["failed_cases"]) >= int(args.max_failures):
                    stop_submitting = True
                    break
                pending_index += 1
                case = cases[pending_index - 1]
                with state_lock:
                    state["submitted_cases"] = int(state["submitted_cases"]) + 1
                future = executor.submit(_process_case, pending_index, case)
                futures[future] = (pending_index, case)

        while pending_index < len(cases) or futures:
            _submit_available()
            if not futures:
                break
            done, _pending = concurrent.futures.wait(
                futures,
                timeout=1.0,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                emit_heartbeat("parallel_running")
                continue
            for future in done:
                futures.pop(future, None)
                try:
                    future.result()
                except BaseException as exc:  # noqa: BLE001
                    # _process_case catches case-specific exceptions; this guard
                    # keeps an unexpected scheduler error visible without killing
                    # already-running workers.
                    emit_heartbeat(
                        "parallel_worker_exception",
                        exception={"type": type(exc).__name__, "message": str(exc)},
                    )
                    with state_lock:
                        state["failed_cases"] = int(state["failed_cases"]) + 1
                    if args.max_failures:
                        exit_code = 1
                if args.max_failures and int(state["failed_cases"]) >= int(args.max_failures):
                    stop_submitting = True
    final_heartbeat = {
        "seq": heartbeat_seq + 1,
        "status": "restart_requested" if restart_ack is not None else "completed" if exit_code == 0 else "failed",
        "phase": "restart_requested" if restart_ack is not None else "finished",
        "runner_capabilities": list(RUNNER_CAPABILITIES),
        "parallel_cases": parallel_cases,
        "submitted_cases": int(state["submitted_cases"]),
        "processed": _summary_processed_count(),
        "total_cases": len(cases),
        "failed_cases": int(state["failed_cases"]),
        "generated_qa_warning_cases": sum(
            1 for row in summary if isinstance(row, Mapping) and row_generated_qa_warning(row)
        ),
        "generated_qa_error_cases": sum(
            1 for row in summary if isinstance(row, Mapping) and row_generated_qa_error(row)
        ),
        "exit_code": exit_code,
    }
    if restart_ack is not None:
        final_heartbeat["restart_request"] = restart_ack
    write_summary(
        output_root,
        summary,
        row_limit=summary_row_limit,
        extra={
            "status": final_heartbeat["status"],
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.time() - parent_started, 3),
            "exit_code": exit_code,
            "parallel_cases": parallel_cases,
        },
    )
    write_heartbeat(output_root, final_heartbeat)
    lock_fields = dict(final_heartbeat)
    lock_fields.pop("phase", None)
    runner_lock.refresh("finished", **lock_fields)
    return exit_code


def run_parent(args: argparse.Namespace) -> int:
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runner_lock = ArtifactRunnerLock(
        output_root,
        wait_timeout_seconds=float(getattr(args, "artifact_lock_timeout", 0.0) or 0.0),
        stale_seconds=float(
            getattr(args, "artifact_lock_stale_seconds", RUNNER_LOCK_STALE_SECONDS)
            or RUNNER_LOCK_STALE_SECONDS
        ),
        poll_seconds=float(
            getattr(args, "artifact_lock_poll_seconds", RUNNER_LOCK_POLL_SECONDS)
            or RUNNER_LOCK_POLL_SECONDS
        ),
    )
    runner_lock.acquire()
    try:
        if int(getattr(args, "parallel_cases", 1) or 1) > 1:
            return _run_parent_parallel_locked(args, runner_lock)
        return _run_parent_locked(args, runner_lock)
    finally:
        runner_lock.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--cases-json",
        type=Path,
        default=None,
        help="Optional explicit case list. Used by backend dataset caption jobs.",
    )
    parser.add_argument(
        "--request-json",
        type=Path,
        default=None,
        help="Optional QwenCaptionRequest field template merged into every image payload.",
    )
    parser.add_argument(
        "--caption-provider",
        choices=CAPTION_PROVIDERS,
        default=CAPTION_PROVIDER_LOCAL,
        help="Model-call provider for caption and generated-QA stages.",
    )
    parser.add_argument(
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI Responses API model used when --caption-provider=openai.",
    )
    parser.add_argument(
        "--openai-image-detail",
        choices=OPENAI_IMAGE_DETAIL_CHOICES,
        default=DEFAULT_OPENAI_IMAGE_DETAIL,
        help="OpenAI image detail for visual caption and QA calls. Original keeps full-resolution image handling by default.",
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=OPENAI_REASONING_EFFORT_CHOICES,
        default=DEFAULT_OPENAI_REASONING_EFFORT,
        help="OpenAI reasoning effort for remote caption and generated-QA calls.",
    )
    parser.add_argument(
        "--openai-api-key-file",
        default=DEFAULT_OPENAI_API_KEY_FILE,
        help="Optional backend-local API key file. OPENAI_API_KEY environment variable takes precedence.",
    )
    parser.add_argument(
        "--openai-service-tier",
        choices=OPENAI_SERVICE_TIER_CHOICES,
        default=DEFAULT_OPENAI_SERVICE_TIER,
        help="Remote pricing tier used for estimates and provenance. Direct runner calls use the Responses API; Batch is an estimate tier, not a direct synchronous service tier.",
    )
    parser.add_argument(
        "--openai-timeout",
        type=float,
        default=DEFAULT_OPENAI_TIMEOUT_SECONDS,
        help="Seconds to wait for each OpenAI Responses API call.",
    )
    parser.add_argument(
        "--openai-max-retries",
        type=int,
        default=DEFAULT_OPENAI_MAX_RETRIES,
        help="Retries per OpenAI Responses API call for 429, 5xx, and transient network failures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "tmp"
        / "qwen_caption_benchmark"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--refinement-model-id", default="same")
    parser.add_argument("--fallback-model-id", default="auto")
    parser.add_argument("--loop-recovery", default="safe_retry_fallback")
    parser.add_argument(
        "--all-images",
        action="store_true",
        help="Run every discovered dataset image instead of the small dense/sparse benchmark selection.",
    )
    parser.add_argument(
        "--caption-mode",
        choices=["full", "windowed"],
        default="full",
        help="Caption mode to use with --all-images.",
    )
    parser.add_argument(
        "--windowed-full-image-strategy",
        choices=["visual", "text_only"],
        default="visual",
        help=(
            "In windowed mode, choose whether final full-image composition sends the full image again "
            "('visual') or composes from completed window observations with a text-only editor pass ('text_only')."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help=(
            "Run this many selected cases. The sample always keeps representative "
            "stress cases such as dense, diverse, sparse, and empty examples, then "
            "fills the remainder with a deterministic random sample."
        ),
    )
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--resume", action="store_true", help="Resume from results.jsonl in the output directory.")
    parser.add_argument(
        "--record-resume-skips",
        action="store_true",
        help=(
            "Append skipped_completed rows for already completed cases during --resume. "
            "Off by default so repeated unattended restarts do not bloat results.jsonl."
        ),
    )
    parser.add_argument(
        "--resume-reprocess-recovery-events",
        action="store_true",
        help=(
            "When resuming, reprocess otherwise successful cases whose latest row "
            "contains caption recovery events."
        ),
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Attempts per image. Each attempt runs in a fresh child process.",
    )
    parser.add_argument(
        "--cooldown-after-crash",
        type=float,
        default=5.0,
        help="Base seconds to wait between failed child attempts.",
    )
    parser.add_argument(
        "--cooldown-after-success",
        type=float,
        default=0.0,
        help="Seconds to wait between successful cases before launching the next child process.",
    )
    parser.add_argument(
        "--cooldown-backoff-multiplier",
        type=float,
        default=DEFAULT_COOLDOWN_BACKOFF_MULTIPLIER,
        help="Multiplier applied to repeated hard child failures for the same image.",
    )
    parser.add_argument(
        "--max-cooldown-after-crash",
        type=float,
        default=DEFAULT_MAX_COOLDOWN_AFTER_CRASH,
        help="Maximum seconds to wait between repeated failed child attempts.",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=0,
        help="Stop after this many failed images. 0 means keep going.",
    )
    parser.add_argument(
        "--continue-on-quality-failures",
        action="store_true",
        help="Record quality warnings but still treat the image as processed.",
    )
    parser.add_argument(
        "--skip-existing-captions",
        action="store_true",
        help="Skip images whose dataset text_labels caption already exists.",
    )
    parser.add_argument(
        "--save-dataset-text-labels",
        action="store_true",
        help="Write successful captions back to the dataset text_labels tree.",
    )
    parser.add_argument("--max-boxes", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--instruction-dataset",
        action="store_true",
        help="Generate instruction-dataset artifacts in addition to broad captions.",
    )
    parser.add_argument(
        "--subcaptions-per-image",
        type=int,
        default=0,
        help="Number of image-grounded generated QA pairs to request per image, clamped to 0-20.",
    )
    parser.add_argument(
        "--include-caption0-in-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether caption0 rows are selected for instruction training artifacts.",
    )
    parser.add_argument(
        "--include-generated-qa-in-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether generated QA rows are selected for instruction training artifacts.",
    )
    parser.add_argument(
        "--include-deterministic-metadata-qa",
        action="store_true",
        help="Whether deterministic metadata QA rows are selected for instruction training artifacts.",
    )
    parser.add_argument(
        "--instruction-qa-imposed-question",
        dest="instruction_qa_imposed_questions",
        action="append",
        default=None,
        help="Required user-supplied question for generated QA. May be passed more than once.",
    )
    parser.add_argument(
        "--instruction-max-new-tokens",
        type=int,
        default=None,
        help="Optional token cap for the generated QA pass. Auto scales with requested QA count.",
    )
    parser.add_argument(
        "--instruction-qa-max-topup-attempts",
        type=int,
        default=DEFAULT_INSTRUCTION_QA_MAX_TOPUP_ATTEMPTS,
        help=(
            "Maximum additional visual generated-QA top-up attempts after the primary prompt. "
            f"Clamped to 0-{MAX_INSTRUCTION_QA_TOPUP_ATTEMPTS}."
        ),
    )
    parser.add_argument(
        "--restrict-speculative-qa-language",
        dest="instruction_qa_restrict_speculative_language",
        action="store_true",
        default=False,
        help=(
            "Reject generated QA rows that use speculative or unavailable-information language. "
            "Default allows grounded inference and unknown/not-visible answers."
        ),
    )
    parser.add_argument(
        "--qa-mix",
        choices=("balanced", "scene", "object", "caption"),
        default="balanced",
        help="Generated QA mix requested from the instruction pass.",
    )
    parser.add_argument(
        "--answer-format",
        choices=("natural", "json"),
        default="natural",
        help="Generated QA answer format requested from the instruction pass.",
    )
    parser.add_argument(
        "--no-source-annotations-in-generator-context",
        dest="include_source_annotations_in_generator_context",
        action="store_false",
        help="Do not include read-only source annotation counts in the generated QA prompt.",
    )
    parser.add_argument(
        "--relaxed-instruction-grounding",
        dest="strict_grounding",
        action="store_false",
        help="Relax the generated QA prompt grounding language.",
    )
    parser.set_defaults(include_source_annotations_in_generator_context=True, strict_grounding=True)
    parser.add_argument("--final-sentences", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=672)
    parser.add_argument("--window-overlap", type=float, default=0.1)
    parser.add_argument("--mlx-max-image-side", type=int, default=512)
    parser.add_argument(
        "--retry-image-side-scale",
        type=float,
        default=DEFAULT_RETRY_IMAGE_SIDE_SCALE,
        help=(
            "Scale MLX caption image side on retry attempts after a failed child attempt. "
            "Attempt 1 uses --mlx-max-image-side; later attempts use side * scale^(attempt-1). "
            "After a native signal exit, the next attempt jumps to --min-retry-image-side. "
            "Use 1.0 to disable adaptive retry downshifting."
        ),
    )
    parser.add_argument(
        "--min-retry-image-side",
        type=int,
        default=DEFAULT_MIN_RETRY_IMAGE_SIDE,
        help="Minimum MLX caption image side used by adaptive retry downshifting.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--use-sampling", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--parallel-cases",
        type=int,
        default=1,
        help=(
            "Maximum image workers to run concurrently under one output manifest. "
            "Use >1 for remote API runs; local MLX defaults to one worker for GPU stability."
        ),
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=30.0,
        help="Seconds between parent-runner heartbeat.json refreshes while a child attempt is running. 0 disables periodic refresh.",
    )
    parser.add_argument(
        "--artifact-lock-timeout",
        type=float,
        default=0.0,
        help="Seconds to wait for another parent runner using the same output directory. 0 waits indefinitely.",
    )
    parser.add_argument(
        "--artifact-lock-stale-seconds",
        type=float,
        default=RUNNER_LOCK_STALE_SECONDS,
        help=(
            "Seconds before an ownerless artifact lock may be treated as stale. "
            "Locks with a live owner pid are never overtaken automatically."
        ),
    )
    parser.add_argument(
        "--artifact-lock-poll-seconds",
        type=float,
        default=RUNNER_LOCK_POLL_SECONDS,
        help="Seconds between artifact-lock wait messages.",
    )
    parser.add_argument(
        "--max-artifact-log-bytes",
        type=int,
        default=DEFAULT_ARTIFACT_LOG_BYTES,
        help=(
            "Maximum bytes to keep for each raw per-attempt text artifact "
            "(child stdout/stderr and copied qwen_caption_io logs). 0 keeps full logs."
        ),
    )
    parser.add_argument(
        "--summary-row-limit",
        type=int,
        default=DEFAULT_SUMMARY_ROW_LIMIT,
        help=(
            "Maximum latest result rows to copy into summary.json. "
            "-1 keeps all rows for manual diagnostics; 0 writes totals only. "
            "results.jsonl remains the authoritative full ledger."
        ),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-json", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.caption_provider = _normalize_caption_provider(args.caption_provider)
        args.openai_image_detail = _normalize_openai_image_detail(args.openai_image_detail)
        args.openai_reasoning_effort = str(
            getattr(args, "openai_reasoning_effort", DEFAULT_OPENAI_REASONING_EFFORT)
            or DEFAULT_OPENAI_REASONING_EFFORT
        ).strip().lower()
        if args.openai_reasoning_effort not in OPENAI_REASONING_EFFORT_CHOICES:
            args.openai_reasoning_effort = DEFAULT_OPENAI_REASONING_EFFORT
        try:
            args.openai_timeout = max(5.0, min(float(args.openai_timeout), 600.0))
        except (TypeError, ValueError, OverflowError):
            args.openai_timeout = DEFAULT_OPENAI_TIMEOUT_SECONDS
        try:
            args.openai_max_retries = max(0, min(int(args.openai_max_retries or 0), 12))
        except (TypeError, ValueError, OverflowError):
            args.openai_max_retries = DEFAULT_OPENAI_MAX_RETRIES
        args.subcaptions_per_image = max(0, min(int(args.subcaptions_per_image or 0), 20))
    except (TypeError, ValueError, OverflowError):
        args.subcaptions_per_image = 0
    args.instruction_qa_imposed_questions = _normalize_instruction_qa_imposed_questions(
        getattr(args, "instruction_qa_imposed_questions", None)
    )
    args.instruction_qa_max_topup_attempts = _instruction_qa_max_topup_attempts(args)
    args.instruction_qa_restrict_speculative_language = _instruction_qa_restrict_speculative_language(args)
    args.qa_mix = str(getattr(args, "qa_mix", "balanced") or "balanced").strip().lower()
    if args.qa_mix not in {"balanced", "scene", "object", "caption"}:
        args.qa_mix = "balanced"
    args.answer_format = str(getattr(args, "answer_format", "natural") or "natural").strip().lower()
    if args.answer_format not in {"natural", "json"}:
        args.answer_format = "natural"
    try:
        args.parallel_cases = max(1, min(int(args.parallel_cases or 1), 256))
    except (TypeError, ValueError, OverflowError):
        args.parallel_cases = 1
    if args.worker:
        if not args.case_json:
            raise SystemExit("--worker requires --case-json")
        return run_worker(args.case_json.resolve(), args.output_dir.resolve(), args.dataset_root.resolve(), args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
