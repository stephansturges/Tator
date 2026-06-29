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
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
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
from typing import Any, Mapping, Sequence
import uuid

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_DATASET = REPO_ROOT / "uploads/datasets/data_ingestion_reference_current_label_images_dataset_9526"
DEFAULT_MODEL = "mlx-community/Qwen3-VL-2B-Instruct-4bit"
DEFAULT_PROMPT = (
    "Write a detailed caption describing the scene, setting, visible objects, "
    "spatial relationships, and notable details. Prefer a high-angle or "
    "top-down wording when it fits the image."
)
LABEL_ALIASES = {
    "LightVehicle": "Light Vehicle",
    "UPole": "U Pole",
    "Solarpanels": "Solar Panels",
    "Gastank": "Gas Tank",
}
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
RUNNER_CAPABILITIES = (
    RUNNER_CAPABILITY_GRACEFUL_RESTART,
    RUNNER_CAPABILITY_PARENT_DETERMINISTIC_RECOVERY,
    RUNNER_CAPABILITY_CAPTION_IO_EVENT_SUMMARY,
    RUNNER_CAPABILITY_WORKER_PROGRESS_HEARTBEAT,
    RUNNER_CAPABILITY_ADAPTIVE_RETRY_PROFILE,
    RUNNER_CAPABILITY_INSTRUCTION_QA,
)
WORKER_PROGRESS_JSON = "worker_progress.json"
WORKER_PROGRESS_JSONL = "worker_progress.jsonl"
QWEN_CAPTION_IO_SUMMARY_JSON = "qwen_caption_io_summary.json"
RUNNER_LOCK_STALE_SECONDS = 21600.0
RUNNER_LOCK_POLL_SECONDS = 5.0
DEFAULT_ARTIFACT_LOG_BYTES = 1_048_576
DEFAULT_COOLDOWN_BACKOFF_MULTIPLIER = 2.0
DEFAULT_MAX_COOLDOWN_AFTER_CRASH = 60.0
DEFAULT_RETRY_IMAGE_SIDE_SCALE = 0.75
DEFAULT_MIN_RETRY_IMAGE_SIDE = 256
DEFAULT_SUMMARY_ROW_LIMIT = 250
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
    "instruction_max_new_tokens": None,
    "include_source_annotations_in_generator_context": True,
    "strict_grounding": True,
    "qa_mix": "balanced",
    "answer_format": "natural",
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


def quality_label_present(text: str, label: str) -> bool:
    lowered = str(text or "").lower()
    compact = lowered.replace(" ", "")
    for variant in quality_label_variants(label):
        if re.search(rf"\b{quality_term_pattern(variant)}\b", lowered, flags=re.IGNORECASE):
            return True
        if variant.replace(" ", "") in compact:
            return True
    return False


def quality_negative_mentions_label(sentence: str, label: str) -> bool:
    lowered = str(sentence or "").lower()
    for variant in quality_label_variants(label):
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


def write_summary(
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    row_limit: int | None = None,
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
        "totals": dict(totals),
        "row_count": len(ordered),
        "row_limit": effective_limit,
        "rows_truncated": rows_truncated,
        "rows_omitted": len(ordered) - len(sampled_rows),
        "rows_sample_policy": sample_policy if rows_truncated else "all",
        "rows": sampled_rows,
    }
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


def summarize_caption(text: str, expected_counts: dict[str, int]) -> dict[str, Any]:
    lowered = text.lower()
    missing_labels = [
        label
        for label in expected_counts
        if not quality_label_present(lowered, label)
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
            if quality_negative_mentions_label(sentence, label):
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


def _extract_json_payload(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except Exception:
            pass
    starts = [idx for idx in (raw.find("{"), raw.find("[")) if idx >= 0]
    if not starts:
        return None
    start = min(starts)
    end = max(raw.rfind("}"), raw.rfind("]"))
    if end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except Exception:
        return None


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


def _case_source_annotation_context(case: Mapping[str, Any]) -> dict[str, Any]:
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
    return {
        "source": "dataset_label_counts",
        "label_count": _case_label_count(case),
        "object_counts": counts,
        "note": "These counts are read-only source annotations; generated QA must not change them.",
    }


def generate_instruction_qa_pairs(
    api: Any,
    case: Mapping[str, Any],
    image_path: Path,
    caption: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    requested = max(0, min(int(getattr(args, "subcaptions_per_image", 0) or 0), 20))
    if requested <= 0:
        return {"requested": 0, "pairs": [], "rejections": [], "status": "skipped"}
    source_context = (
        _case_source_annotation_context(case)
        if bool(getattr(args, "include_source_annotations_in_generator_context", True))
        else {"source": "disabled_by_request"}
    )
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
    prompt = (
        f"Create up to {requested} diverse visual instruction question/answer pairs for this image.\n"
        f"Return only valid JSON with this shape: {answer_shape}.\n"
        "Questions must be image-specific and useful for training a vision-language model.\n"
        f"{mix_text}\n"
        f"{format_text}\n"
        "Do not mention prompts, labels, bounding boxes, coordinates, source annotations, or that counts were provided.\n"
        "Do not ask about an object that is absent or only implied by missing labels.\n"
        f"{strict_text}\n\n"
        f"Completed broad caption:\n{caption}\n\n"
        f"Read-only source annotation context:\n{json.dumps(source_context, ensure_ascii=False, sort_keys=True)}"
    )
    system_prompt = (
        "You generate grounded image question/answer training rows. "
        "Use the image as truth and return only the requested JSON."
    )
    max_new_tokens = _instruction_qa_max_new_tokens(args, requested)
    started = time.time()
    try:
        if hasattr(api, "_qwen_progress_update"):
            api._qwen_progress_update(
                phase="generate",
                phase_label="Generating QA",
                progress=0.9,
                message="Generating instruction QA rows",
                step_id="instruction_qa",
                step_label="Generate instruction QA",
                step_detail=f"Creating up to {requested} generated QA rows",
                token_preview="",
                live_output_reset=True,
            )
        if hasattr(api, "_qwen_progress_begin_output_section"):
            api._qwen_progress_begin_output_section("Generated QA output")
        with Image.open(image_path) as source_image:
            pil_img = source_image.convert("RGB")
        raw, _width, _height = api._run_qwen_inference(
            prompt,
            pil_img,
            max_new_tokens=max_new_tokens,
            system_prompt_override=system_prompt,
            model_id_override=getattr(args, "model_id", None),
            decode_override={
                "do_sample": False,
                "repetition_penalty": 1.08,
                "repetition_context_size": 128,
                "no_repeat_ngram_size": 8,
            },
            chat_template_kwargs={"enable_thinking": False},
        )
        parsed = _extract_json_payload(raw)
        pairs, rejections = _normalize_generated_qa_pairs(
            parsed,
            requested=requested,
            case=case,
            answer_format=answer_format,
        )
        status = "ok" if pairs else "empty"
        return {
            "status": status,
            "requested": requested,
            "pair_count": len(pairs),
            "pairs": pairs,
            "rejections": rejections,
            "raw_output": raw,
            "max_new_tokens": max_new_tokens,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    except BaseException as exc:  # noqa: BLE001
        return {
            "status": "error",
            "requested": requested,
            "pair_count": 0,
            "pairs": [],
            "rejections": [{"reason": "generator_exception", "type": type(exc).__name__, "message": str(exc)}],
            "max_new_tokens": max_new_tokens,
            "elapsed_seconds": round(time.time() - started, 3),
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
        import localinferenceapi as api
        from models.schemas import QwenCaptionRequest

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
            requested_subcaptions = int(getattr(args, "subcaptions_per_image", 0) or 0)
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
                response_data["generated_qa_pair_count"] = int(instruction_qa.get("pair_count") or 0)
            result["response"] = response_data
            result["caption_quality"] = summarize_caption(
                caption_text,
                {str(k): int(v) for k, v in response_data.get("used_counts", {}).items()},
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
                result["status"] = "instruction_qa_failed"
                result["exception"] = {
                    "type": "InstructionQaGenerationError",
                    "message": (
                        "generated QA was requested, but no valid generated QA pairs "
                        f"were produced; status={instruction_status}"
                    ),
                }
                return 1
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


def _run_parent_locked(args: argparse.Namespace, runner_lock: ArtifactRunnerLock) -> int:
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
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
    results_jsonl = output_root / "results.jsonl"
    captions_jsonl = output_root / "captions.jsonl"
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
            if getattr(args, "instruction_max_new_tokens", None) is not None:
                cmd.extend(["--instruction-max-new-tokens", str(args.instruction_max_new_tokens)])
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
                "instruction_qa_status": (
                    ((result.get("instruction_qa") or {}) if isinstance(result.get("instruction_qa"), Mapping) else {}).get("status")
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
                    }
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
            allow_parent_deterministic_recovery = not (
                bool(getattr(args, "instruction_dataset", False))
                and int(getattr(args, "subcaptions_per_image", 0) or 0) > 0
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
                if not allow_parent_deterministic_recovery:
                    final_row["parent_deterministic_recovery_skipped"] = {
                        "reason": "generated_qa_required",
                        "detail": (
                            "Parent deterministic count/layout recovery cannot satisfy "
                            "a generated-QA instruction dataset case."
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
        "exit_code": exit_code,
    }
    if restart_ack is not None:
        final_heartbeat["restart_request"] = restart_ack
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
        "--instruction-max-new-tokens",
        type=int,
        default=None,
        help="Optional token cap for the generated QA pass. Auto scales with requested QA count.",
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
        args.subcaptions_per_image = max(0, min(int(args.subcaptions_per_image or 0), 20))
    except (TypeError, ValueError, OverflowError):
        args.subcaptions_per_image = 0
    args.qa_mix = str(getattr(args, "qa_mix", "balanced") or "balanced").strip().lower()
    if args.qa_mix not in {"balanced", "scene", "object", "caption"}:
        args.qa_mix = "balanced"
    args.answer_format = str(getattr(args, "answer_format", "natural") or "natural").strip().lower()
    if args.answer_format not in {"natural", "json"}:
        args.answer_format = "natural"
    if args.worker:
        if not args.case_json:
            raise SystemExit("--worker requires --case-json")
        return run_worker(args.case_json.resolve(), args.output_dir.resolve(), args.dataset_root.resolve(), args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
