#!/usr/bin/env python3
"""Audit Qwen caption soak-run artifacts for unattended-run health."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence


FINAL_OK_STATUSES = {"ok", "preview_only", "skipped_completed", "skipped_existing_caption"}
CAPTION_ROW_REQUIRED_FINAL_STATUSES = {"ok", "skipped_completed"}
CAPTION_ROW_EXEMPT_ROW_STATUSES = {"preview_only"}
RUNNER_LOCK_NAME = ".runner.lock"
DEFAULT_MAX_RECOVERY_EVENT_CASE_RATE = 0.25
DEFAULT_MAX_LOOP_RECOVERY_CASE_RATE = 0.0
DEFAULT_SET_AND_FORGET_MAX_LOOP_RECOVERY_CASE_RATE = 0.05
DEFAULT_MAX_LOOP_GUARD_CASE_RATE = 0.0
DEFAULT_SET_AND_FORGET_MAX_LOOP_GUARD_CASE_RATE = 0.05
DEFAULT_MAX_DETERMINISTIC_RECOVERY_CASE_RATE = 0.0
DEFAULT_SET_AND_FORGET_MAX_DETERMINISTIC_RECOVERY_CASE_RATE = 0.01
DEFAULT_MAX_SIGNAL_EXIT_ATTEMPT_ROW_RATE = 0.0
DEFAULT_SET_AND_FORGET_MAX_SIGNAL_EXIT_ATTEMPT_ROW_RATE = 0.05
DEFAULT_DEGRADED_RATE_CAUTION_RATIO = 0.80
MIN_LOOP_RECOVERY_RATE_EVENTS = 3
MIN_DETERMINISTIC_RECOVERY_RATE_EVENTS = 2


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _finite_threshold(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if parsed != parsed:
        return default
    return parsed


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed


def resolve_loop_recovery_threshold(value: Any, *, set_and_forget: bool = False) -> float:
    default = (
        DEFAULT_SET_AND_FORGET_MAX_LOOP_RECOVERY_CASE_RATE
        if set_and_forget
        else DEFAULT_MAX_LOOP_RECOVERY_CASE_RATE
    )
    return _finite_threshold(value, default)


def resolve_loop_guard_threshold(value: Any, *, set_and_forget: bool = False) -> float:
    default = (
        DEFAULT_SET_AND_FORGET_MAX_LOOP_GUARD_CASE_RATE
        if set_and_forget
        else DEFAULT_MAX_LOOP_GUARD_CASE_RATE
    )
    return _finite_threshold(value, default)


def resolve_deterministic_recovery_threshold(value: Any, *, set_and_forget: bool = False) -> float:
    default = (
        DEFAULT_SET_AND_FORGET_MAX_DETERMINISTIC_RECOVERY_CASE_RATE
        if set_and_forget
        else DEFAULT_MAX_DETERMINISTIC_RECOVERY_CASE_RATE
    )
    return _finite_threshold(value, default)


def resolve_signal_exit_attempt_threshold(value: Any, *, set_and_forget: bool = False) -> float:
    default = (
        DEFAULT_SET_AND_FORGET_MAX_SIGNAL_EXIT_ATTEMPT_ROW_RATE
        if set_and_forget
        else DEFAULT_MAX_SIGNAL_EXIT_ATTEMPT_ROW_RATE
    )
    return _finite_threshold(value, default)


def _min_loop_recovery_rate_cases(threshold: float, min_rate_cases: int) -> int:
    """Return the sample floor for low-rate loop recovery gates on live runs."""
    base = max(1, int(min_rate_cases or 0))
    if threshold <= 0 or threshold >= 0.1:
        return base
    return max(base, int(math.ceil(MIN_LOOP_RECOVERY_RATE_EVENTS / threshold)))


def _min_deterministic_recovery_rate_cases(threshold: float, min_rate_cases: int) -> int:
    base = max(1, int(min_rate_cases or 0))
    if threshold <= 0 or threshold >= 0.1:
        return base
    return max(base, int(math.ceil(MIN_DETERMINISTIC_RECOVERY_RATE_EVENTS / threshold)))


def _row_recovery_events(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = row.get("recovery_events")
    if not isinstance(raw, list):
        return []
    return [event for event in raw if isinstance(event, Mapping)]


def _row_has_loop_recovery(row: Mapping[str, Any]) -> bool:
    for event in _row_recovery_events(row):
        action = str(event.get("action") or "").lower()
        message = str(event.get("message") or "").lower()
        detail = str(event.get("detail") or "").lower()
        if "loop" in action or "loop" in message or "loop" in detail:
            return True
    return False


def _row_has_deterministic_recovery(row: Mapping[str, Any]) -> bool:
    if bool(row.get("parent_deterministic_recovery")):
        return True
    for event in _row_recovery_events(row):
        action = str(event.get("action") or "").lower()
        attempt = str(event.get("attempt") or "").lower()
        call_kind = str(event.get("call_kind") or "").lower()
        if (
            action == "deterministic_recovery_succeeded"
            or attempt in {"deterministic_recovery", "parent_deterministic_recovery"}
            or call_kind == "deterministic"
        ):
            return True
    return False


def _row_prompt_budget(row: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = row.get("preview_prompt_budget")
    budget = dict(raw) if isinstance(raw, Mapping) else {}
    raw_io = row.get("qwen_caption_io")
    io_summary = raw_io if isinstance(raw_io, Mapping) else {}
    runtime_prompt_tokens = max(0, _safe_int(io_summary.get("max_prompt_tokens"), 0))
    runtime_input_tokens = max(0, _safe_int(io_summary.get("max_input_tokens"), 0))
    runtime_events = max(0, _safe_int(io_summary.get("prompt_budget_events"), 0))
    runtime_adapted_events = max(
        0,
        _safe_int(io_summary.get("prompt_budget_adapted_events"), 0),
    )
    if runtime_prompt_tokens > 0:
        preview_prompt_tokens = max(0, _safe_int(budget.get("max_prompt_tokens"), 0))
        budget["preview_max_prompt_tokens"] = preview_prompt_tokens
        budget["runtime_max_prompt_tokens"] = runtime_prompt_tokens
        budget["max_prompt_tokens"] = max(preview_prompt_tokens, runtime_prompt_tokens)
    if runtime_input_tokens > 0:
        budget["runtime_max_input_tokens"] = runtime_input_tokens
    if runtime_events > 0:
        budget["runtime_prompt_budget_events"] = runtime_events
    if runtime_adapted_events > 0:
        budget["runtime_prompt_budget_adapted_events"] = runtime_adapted_events
    return budget


def _row_has_adapted_prompt_budget(row: Mapping[str, Any]) -> bool:
    budget = _row_prompt_budget(row)
    try:
        return (
            int(budget.get("adapted_sections") or 0) > 0
            or int(budget.get("runtime_prompt_budget_adapted_events") or 0) > 0
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _row_max_prompt_tokens(row: Mapping[str, Any]) -> int:
    budget = _row_prompt_budget(row)
    try:
        return max(0, int(budget.get("max_prompt_tokens") or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _row_io_event_count(row: Mapping[str, Any], event_name: str) -> int:
    raw_io = row.get("qwen_caption_io")
    if not isinstance(raw_io, Mapping):
        return 0
    event_counts = raw_io.get("event_counts")
    if isinstance(event_counts, Mapping):
        try:
            return max(0, int(event_counts.get(event_name) or 0))
        except (TypeError, ValueError, OverflowError):
            return 0
    direct_key = f"{event_name}_events"
    try:
        return max(0, int(raw_io.get(direct_key) or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _row_has_loop_guard(row: Mapping[str, Any]) -> bool:
    return (
        _row_io_event_count(row, "stream_loop_detected") > 0
        or _row_io_event_count(row, "loop_trim") > 0
    )


def _count_failed_attempt_rows(rows: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for row in rows:
        final_status = str(row.get("final_status") or "").strip().lower()
        status = str(row.get("status") or "").strip().lower()
        if final_status == "failed_attempt" or status in {"timeout", "exception", "missing_result"}:
            count += 1
    return count


def _count_signal_exit_attempt_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[int, dict[str, int]]:
    count = 0
    signal_names: Counter[str] = Counter()
    for row in rows:
        failure_kind = str(row.get("attempt_failure_kind") or "").strip().lower()
        return_signal = row.get("return_signal")
        exit_code = row.get("exit_code")
        has_negative_exit = isinstance(exit_code, int) and exit_code < 0
        if failure_kind == "signal_exit" or return_signal is not None or has_negative_exit:
            count += 1
            signal_name = str(row.get("return_signal_name") or "").strip() or "unknown"
            signal_names[signal_name] += 1
    return count, dict(signal_names)


def _allowed_count_for_threshold(threshold: float, denominator: int) -> int:
    if threshold < 0 or denominator <= 0:
        return -1
    return int(math.floor(float(threshold) * float(denominator) + 1e-12))


def _rate_headroom_item(
    *,
    rate_key: str,
    numerator: int,
    denominator: int,
    threshold: float,
    terminal_denominator: int | None = None,
) -> dict[str, Any] | None:
    if threshold < 0:
        return None
    denominator = max(0, int(denominator or 0))
    numerator = max(0, int(numerator or 0))
    actual = _rate(numerator, denominator)
    current_allowed = _allowed_count_for_threshold(threshold, denominator)
    item: dict[str, Any] = {
        "rate": rate_key,
        "numerator": numerator,
        "denominator": denominator,
        "actual": actual,
        "threshold": threshold,
        "allowed_count_at_current_denominator": current_allowed,
        "remaining_count_at_current_denominator": current_allowed - numerator,
        "caution_ratio": DEFAULT_DEGRADED_RATE_CAUTION_RATIO,
        "near_threshold": bool(
            threshold > 0
            and actual + 1e-12 >= threshold * DEFAULT_DEGRADED_RATE_CAUTION_RATIO
        ),
    }
    if threshold > 0:
        item["threshold_fraction_used"] = actual / threshold
    if terminal_denominator is not None and terminal_denominator > denominator:
        terminal_allowed = _allowed_count_for_threshold(threshold, terminal_denominator)
        item.update({
            "terminal_denominator": terminal_denominator,
            "allowed_count_at_terminal_denominator": terminal_allowed,
            "remaining_count_at_terminal_denominator": terminal_allowed - numerator,
            "terminal_rate_if_no_more_events": _rate(numerator, terminal_denominator),
        })
    return item


def _degraded_rate_checks(
    *,
    latest_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    expected_cases: int,
    failed_cases: int,
    pending_failed_attempt_cases: int,
    quality_failed: int,
    heartbeat_status: str,
    max_failed_case_rate: float,
    max_quality_failure_rate: float,
    max_recovery_event_case_rate: float,
    max_loop_recovery_case_rate: float,
    max_loop_guard_case_rate: float,
    max_deterministic_recovery_case_rate: float,
    max_failed_attempt_row_rate: float,
    max_signal_exit_attempt_row_rate: float,
    min_rate_cases: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    processed = len(latest_rows)
    attempt_rows = len(rows)
    failed_attempt_rows = _count_failed_attempt_rows(rows)
    signal_exit_attempt_rows, signal_exit_names = _count_signal_exit_attempt_rows(rows)
    recovery_event_cases = sum(1 for row in latest_rows if _row_recovery_events(row))
    recovery_events = sum(len(_row_recovery_events(row)) for row in latest_rows)
    loop_recovery_cases = sum(1 for row in latest_rows if _row_has_loop_recovery(row))
    deterministic_recovery_cases = sum(1 for row in latest_rows if _row_has_deterministic_recovery(row))
    prompt_budget_rows = sum(1 for row in latest_rows if _row_prompt_budget(row))
    prompt_budget_adapted_cases = sum(1 for row in latest_rows if _row_has_adapted_prompt_budget(row))
    max_prompt_tokens = max((_row_max_prompt_tokens(row) for row in latest_rows), default=0)
    stream_loop_detected_cases = sum(
        1 for row in latest_rows if _row_io_event_count(row, "stream_loop_detected") > 0
    )
    stream_loop_detected_events = sum(
        _row_io_event_count(row, "stream_loop_detected") for row in latest_rows
    )
    loop_trim_cases = sum(1 for row in latest_rows if _row_io_event_count(row, "loop_trim") > 0)
    loop_trim_events = sum(_row_io_event_count(row, "loop_trim") for row in latest_rows)
    loop_guard_cases = sum(1 for row in latest_rows if _row_has_loop_guard(row))
    completed = heartbeat_status in {"completed", "failed"}
    expected_cases = max(0, int(expected_cases or 0))
    min_rate_cases = max(1, int(min_rate_cases or 0))
    active = completed or processed >= min_rate_cases
    min_loop_recovery_rate_cases = _min_loop_recovery_rate_cases(
        max_loop_recovery_case_rate,
        min_rate_cases,
    )
    min_loop_guard_rate_cases = _min_loop_recovery_rate_cases(
        max_loop_guard_case_rate,
        min_rate_cases,
    )
    min_deterministic_recovery_rate_cases = _min_deterministic_recovery_rate_cases(
        max_deterministic_recovery_case_rate,
        min_rate_cases,
    )
    rates = {
        "processed_cases": processed,
        "attempt_rows": attempt_rows,
        "failed_cases": failed_cases,
        "pending_failed_attempt_cases": pending_failed_attempt_cases,
        "quality_failed_cases": quality_failed,
        "failed_attempt_rows": failed_attempt_rows,
        "signal_exit_attempt_rows": signal_exit_attempt_rows,
        "signal_exit_names": signal_exit_names,
        "recovery_event_cases": recovery_event_cases,
        "recovery_events": recovery_events,
        "loop_recovery_cases": loop_recovery_cases,
        "loop_guard_cases": loop_guard_cases,
        "deterministic_recovery_cases": deterministic_recovery_cases,
        "prompt_budget_rows": prompt_budget_rows,
        "prompt_budget_adapted_cases": prompt_budget_adapted_cases,
        "max_prompt_tokens": max_prompt_tokens,
        "stream_loop_detected_cases": stream_loop_detected_cases,
        "stream_loop_detected_events": stream_loop_detected_events,
        "loop_trim_cases": loop_trim_cases,
        "loop_trim_events": loop_trim_events,
        "failed_case_rate": _rate(failed_cases, processed),
        "quality_failure_rate": _rate(quality_failed, processed),
        "recovery_event_case_rate": _rate(recovery_event_cases, processed),
        "loop_recovery_case_rate": _rate(loop_recovery_cases, processed),
        "loop_guard_case_rate": _rate(loop_guard_cases, processed),
        "deterministic_recovery_case_rate": _rate(deterministic_recovery_cases, processed),
        "prompt_budget_coverage_rate": _rate(prompt_budget_rows, processed),
        "prompt_budget_adapted_case_rate": _rate(prompt_budget_adapted_cases, processed),
        "stream_loop_detected_case_rate": _rate(stream_loop_detected_cases, processed),
        "loop_trim_case_rate": _rate(loop_trim_cases, processed),
        "failed_attempt_row_rate": _rate(failed_attempt_rows, attempt_rows),
        "signal_exit_attempt_row_rate": _rate(signal_exit_attempt_rows, attempt_rows),
        "min_rate_cases": min_rate_cases,
        "min_loop_recovery_rate_cases": min_loop_recovery_rate_cases,
        "min_loop_guard_rate_cases": min_loop_guard_rate_cases,
        "min_deterministic_recovery_rate_cases": min_deterministic_recovery_rate_cases,
        "active": active,
    }
    thresholds = {
        "max_failed_case_rate": max_failed_case_rate,
        "max_quality_failure_rate": max_quality_failure_rate,
        "max_recovery_event_case_rate": max_recovery_event_case_rate,
        "max_loop_recovery_case_rate": max_loop_recovery_case_rate,
        "max_loop_guard_case_rate": max_loop_guard_case_rate,
        "max_deterministic_recovery_case_rate": max_deterministic_recovery_case_rate,
        "max_failed_attempt_row_rate": max_failed_attempt_row_rate,
        "max_signal_exit_attempt_row_rate": max_signal_exit_attempt_row_rate,
    }
    terminal_case_denominator = (
        expected_cases
        if expected_cases > processed and not completed
        else None
    )
    headroom_candidates = [
        _rate_headroom_item(
            rate_key="failed_case_rate",
            numerator=failed_cases,
            denominator=processed,
            threshold=max_failed_case_rate,
            terminal_denominator=terminal_case_denominator,
        ),
        _rate_headroom_item(
            rate_key="quality_failure_rate",
            numerator=quality_failed,
            denominator=processed,
            threshold=max_quality_failure_rate,
            terminal_denominator=terminal_case_denominator,
        ),
        _rate_headroom_item(
            rate_key="recovery_event_case_rate",
            numerator=recovery_event_cases,
            denominator=processed,
            threshold=max_recovery_event_case_rate,
            terminal_denominator=terminal_case_denominator,
        ),
        _rate_headroom_item(
            rate_key="loop_recovery_case_rate",
            numerator=loop_recovery_cases,
            denominator=processed,
            threshold=max_loop_recovery_case_rate,
            terminal_denominator=terminal_case_denominator,
        ),
        _rate_headroom_item(
            rate_key="loop_guard_case_rate",
            numerator=loop_guard_cases,
            denominator=processed,
            threshold=max_loop_guard_case_rate,
            terminal_denominator=terminal_case_denominator,
        ),
        _rate_headroom_item(
            rate_key="deterministic_recovery_case_rate",
            numerator=deterministic_recovery_cases,
            denominator=processed,
            threshold=max_deterministic_recovery_case_rate,
            terminal_denominator=terminal_case_denominator,
        ),
        _rate_headroom_item(
            rate_key="failed_attempt_row_rate",
            numerator=failed_attempt_rows,
            denominator=attempt_rows,
            threshold=max_failed_attempt_row_rate,
        ),
        _rate_headroom_item(
            rate_key="signal_exit_attempt_row_rate",
            numerator=signal_exit_attempt_rows,
            denominator=attempt_rows,
            threshold=max_signal_exit_attempt_row_rate,
        ),
    ]
    rate_headroom = [item for item in headroom_candidates if item is not None]
    caution_rates = [
        item
        for item in rate_headroom
        if item.get("near_threshold")
    ]
    violations: list[dict[str, Any]] = []
    active_violations: list[dict[str, Any]] = []
    sample_floor_by_rate = {
        "loop_recovery_case_rate": min_loop_recovery_rate_cases,
        "loop_guard_case_rate": min_loop_guard_rate_cases,
        "deterministic_recovery_case_rate": min_deterministic_recovery_rate_cases,
    }
    for threshold_key, rate_key in (
        ("max_failed_case_rate", "failed_case_rate"),
        ("max_quality_failure_rate", "quality_failure_rate"),
        ("max_recovery_event_case_rate", "recovery_event_case_rate"),
        ("max_loop_recovery_case_rate", "loop_recovery_case_rate"),
        ("max_loop_guard_case_rate", "loop_guard_case_rate"),
        ("max_deterministic_recovery_case_rate", "deterministic_recovery_case_rate"),
        ("max_failed_attempt_row_rate", "failed_attempt_row_rate"),
        ("max_signal_exit_attempt_row_rate", "signal_exit_attempt_row_rate"),
    ):
        threshold = thresholds[threshold_key]
        if threshold < 0:
            continue
        actual = float(rates[rate_key])
        if actual > threshold:
            violation = {
                "rate": rate_key,
                "actual": actual,
                "threshold": threshold,
            }
            if (
                rate_key in sample_floor_by_rate
                and not completed
                and threshold > 0
                and processed < sample_floor_by_rate[rate_key]
            ):
                floor = sample_floor_by_rate[rate_key]
                violation["active"] = False
                violation["deferred_by_sample_floor"] = True
                violation["detail"] = (
                    f"{rate_key} is above threshold, but the live sample is below "
                    f"the rate activation floor of {floor} cases"
                )
            elif rate_key == "signal_exit_attempt_row_rate" and not completed and threshold <= 0:
                violation["active"] = False
                violation["deferred_until_terminal"] = True
                violation["detail"] = (
                    "signal-exit attempts are tracked during live recovery and enforced by terminal audit"
                )
            else:
                violation["active"] = bool(
                    active
                    or (
                        rate_key == "failed_case_rate"
                        and threshold <= 0
                        and failed_cases > 0
                    )
                )
            violations.append(violation)
            if violation["active"]:
                active_violations.append(violation)
    return (
        {
            **rates,
            "thresholds": thresholds,
            "rate_headroom": rate_headroom,
            "caution_rates": caution_rates,
            "violations": violations,
            "active_violations": active_violations,
        },
        active_violations,
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_jsonl_with_errors(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not path.exists():
        return rows, errors
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            errors.append(
                {
                    "line": line_number,
                    "error": str(exc),
                    "preview": stripped[:160],
                }
            )
            continue
        if isinstance(payload, dict):
            rows.append(payload)
        else:
            errors.append(
                {
                    "line": line_number,
                    "error": "JSONL row is not an object",
                    "preview": stripped[:160],
                }
            )
    return rows, errors


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows, _errors = _read_jsonl_with_errors(path)
    return rows


def _case_key(row: Mapping[str, Any]) -> str:
    return str(row.get("case_id") or row.get("case") or row.get("stem") or "").strip()


def _latest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _case_key(row)
        if key:
            latest[key] = row
    return list(latest.values())


def _final_status(row: Mapping[str, Any]) -> str:
    return str(row.get("final_status") or "").strip().lower()


def _row_status(row: Mapping[str, Any]) -> str:
    return str(row.get("status") or "").strip().lower()


def _sample_case(row: Mapping[str, Any]) -> dict[str, Any]:
    sample: dict[str, Any] = {"case_id": _case_key(row)}
    for field in ("case", "stem", "image_name", "final_status", "status"):
        value = row.get(field)
        if value not in (None, ""):
            sample[field] = value
    return sample


def _caption_coverage_check(
    *,
    latest_rows: Sequence[Mapping[str, Any]],
    caption_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Check that generated and resumed-completed successful rows have caption records."""
    required_rows = [
        row
        for row in latest_rows
        if _final_status(row) in CAPTION_ROW_REQUIRED_FINAL_STATUSES
        and _row_status(row) not in CAPTION_ROW_EXEMPT_ROW_STATUSES
    ]
    valid_caption_case_ids: set[str] = set()
    invalid_caption_rows: list[dict[str, Any]] = []
    for line_number, row in enumerate(caption_rows, start=1):
        key = _case_key(row)
        caption = str(row.get("caption") or "").strip()
        if not key or not caption:
            invalid_caption_rows.append(
                {
                    "line": line_number,
                    "case_id": key,
                    "missing_case_id": not bool(key),
                    "empty_caption": not bool(caption),
                }
            )
            continue
        valid_caption_case_ids.add(key)
    missing_rows = [
        row
        for row in required_rows
        if _case_key(row) not in valid_caption_case_ids
    ]
    covered = len(required_rows) - len(missing_rows)
    status = "ok"
    if missing_rows or invalid_caption_rows:
        status = "error"
    detail = (
        f"{covered}/{len(required_rows)} latest successful generated/resumed cases "
        "have non-empty caption rows"
    )
    if invalid_caption_rows:
        detail += f"; {len(invalid_caption_rows)} caption row(s) are missing a case id or caption"
    return {
        "status": status,
        "detail": detail,
        "required_generated_successes": len(required_rows),
        "covered_generated_successes": covered,
        "caption_rows": len(caption_rows),
        "missing_caption_rows": len(missing_rows),
        "missing_cases": [_sample_case(row) for row in missing_rows[:10]],
        "invalid_caption_rows": invalid_caption_rows[:10],
    }


def _resolve_saved_text_label_path(output_dir: Path, value: Any) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    return output_dir / candidate


def _saved_text_label_coverage_check(
    *,
    output_dir: Path,
    latest_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Check that generated and resumed-completed successful rows have saved label files."""
    required_rows = [
        row
        for row in latest_rows
        if _final_status(row) in CAPTION_ROW_REQUIRED_FINAL_STATUSES
        and _row_status(row) not in CAPTION_ROW_EXEMPT_ROW_STATUSES
    ]
    missing_rows: list[dict[str, Any]] = []
    missing_files: list[dict[str, Any]] = []
    empty_files: list[dict[str, Any]] = []
    unreadable_files: list[dict[str, Any]] = []
    for row in required_rows:
        raw_path = str(row.get("saved_text_label") or "").strip()
        sample = _sample_case(row)
        if raw_path:
            sample["saved_text_label"] = raw_path
        saved_path = _resolve_saved_text_label_path(output_dir, raw_path)
        if saved_path is None:
            missing_rows.append(sample)
            continue
        if not saved_path.exists() or not saved_path.is_file():
            sample["resolved_path"] = str(saved_path)
            missing_files.append(sample)
            continue
        try:
            saved_text = saved_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as exc:  # noqa: BLE001
            sample["resolved_path"] = str(saved_path)
            sample["error"] = str(exc)
            unreadable_files.append(sample)
            continue
        if not saved_text:
            sample["resolved_path"] = str(saved_path)
            empty_files.append(sample)

    failed_rows = len(missing_rows) + len(missing_files) + len(empty_files) + len(unreadable_files)
    covered = len(required_rows) - failed_rows
    status = "ok" if failed_rows == 0 else "error"
    detail = (
        f"{covered}/{len(required_rows)} latest successful generated/resumed cases "
        "have saved dataset text-label files"
    )
    if missing_rows:
        detail += f"; {len(missing_rows)} row(s) are missing saved_text_label"
    if missing_files:
        detail += f"; {len(missing_files)} saved text-label file(s) are missing"
    if empty_files:
        detail += f"; {len(empty_files)} saved text-label file(s) are empty"
    if unreadable_files:
        detail += f"; {len(unreadable_files)} saved text-label file(s) are unreadable"
    return {
        "status": status,
        "detail": detail,
        "required_saved_text_labels": len(required_rows),
        "covered_saved_text_labels": covered,
        "missing_saved_text_label_rows": len(missing_rows),
        "missing_saved_text_label_files": len(missing_files),
        "empty_saved_text_label_files": len(empty_files),
        "unreadable_saved_text_label_files": len(unreadable_files),
        "missing_cases": missing_rows[:10],
        "missing_files": missing_files[:10],
        "empty_files": empty_files[:10],
        "unreadable_files": unreadable_files[:10],
    }


def _is_live_retryable_failed_attempt(
    row: Mapping[str, Any],
    *,
    allow_running_incomplete: bool,
    heartbeat_status: str,
) -> bool:
    has_next_attempt = "next_attempt_cooldown_seconds" in row
    return (
        allow_running_incomplete
        and heartbeat_status == "running"
        and _final_status(row) == "failed_attempt"
        and has_next_attempt
    )


def _status_rank(status: str) -> int:
    return {"ok": 0, "warn": 1, "error": 2}.get(status, 2)


def _add_check(checks: list[dict[str, Any]], name: str, status: str, detail: str = "", **extra: Any) -> None:
    checks.append({"name": name, "status": status, "detail": detail, **extra})


def _format_bytes(value: Any) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError, OverflowError):
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


def _disk_reserve_check(output_dir: Path, min_free_gb: float) -> dict[str, Any] | None:
    min_free_gb = max(0.0, _finite_threshold(min_free_gb, 0.0))
    if min_free_gb <= 0:
        return None
    min_free_bytes = int(min_free_gb * 1024 * 1024 * 1024)
    try:
        usage = shutil.disk_usage(output_dir)
    except Exception as exc:  # noqa: BLE001
        return {
            "name": "disk_reserve",
            "status": "error",
            "detail": f"could not inspect free disk for {output_dir}: {exc}",
            "min_free_gb": min_free_gb,
            "min_free_bytes": min_free_bytes,
        }
    status = "ok" if int(usage.free) >= min_free_bytes else "error"
    return {
        "name": "disk_reserve",
        "status": status,
        "detail": (
            f"free disk {_format_bytes(usage.free)} meets live reserve {_format_bytes(min_free_bytes)}"
            if status == "ok"
            else f"free disk {_format_bytes(usage.free)} is below live reserve {_format_bytes(min_free_bytes)}"
        ),
        "free_bytes": int(usage.free),
        "free_human": _format_bytes(usage.free),
        "min_free_gb": min_free_gb,
        "min_free_bytes": min_free_bytes,
        "min_free_human": _format_bytes(min_free_bytes),
    }


def _heartbeat_age_seconds(heartbeat: Mapping[str, Any], heartbeat_path: Path) -> float:
    try:
        epoch = float(heartbeat.get("heartbeat_epoch") or 0.0)
    except (TypeError, ValueError, OverflowError):
        epoch = 0.0
    if epoch > 0:
        return max(0.0, time.time() - epoch)
    try:
        return max(0.0, time.time() - heartbeat_path.stat().st_mtime)
    except Exception:
        return 0.0


def _active_attempt_runtime(heartbeat: Mapping[str, Any]) -> dict[str, Any] | None:
    status = str(heartbeat.get("status") or "").strip().lower()
    phase = str(heartbeat.get("phase") or "").strip().lower()
    if status != "running" or phase != "attempt_running":
        return None
    started = _finite_threshold(heartbeat.get("attempt_started_epoch"), 0.0)
    timeout = _finite_threshold(heartbeat.get("attempt_timeout_seconds"), 0.0)
    if started <= 0 or timeout <= 0:
        return None
    runtime = max(0.0, time.time() - started)
    active_attempt: dict[str, Any] = {
        "status": status,
        "phase": phase,
        "attempt_started_epoch": started,
        "runtime_seconds": runtime,
        "attempt_timeout_seconds": timeout,
    }
    for key in ("case", "case_id", "image_name", "case_index", "attempt"):
        value = heartbeat.get(key)
        if value is not None and value != "":
            active_attempt[key] = value
    worker_progress = heartbeat.get("worker_progress")
    if isinstance(worker_progress, Mapping):
        active_attempt["worker_progress"] = dict(worker_progress)
    for key in (
        "worker_progress_seq",
        "worker_phase",
        "worker_step_id",
        "worker_step_label",
        "worker_message",
        "worker_generated_tokens",
        "worker_max_new_tokens",
    ):
        value = heartbeat.get(key)
        if value is not None and value != "":
            active_attempt[key] = value
    return active_attempt


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


def _epoch_age_seconds(value: Any) -> float:
    try:
        epoch = float(value or 0.0)
    except (TypeError, ValueError, OverflowError):
        epoch = 0.0
    if epoch <= 0:
        return 0.0
    return max(0.0, time.time() - epoch)


def _positive_epoch(value: Any) -> float:
    try:
        epoch = float(value or 0.0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(epoch) or epoch <= 0:
        return 0.0
    return epoch


def _first_supervisor_epoch(output_dir: Path) -> tuple[float, str]:
    supervisor_log = output_dir / "supervisor.jsonl"
    if not supervisor_log.exists():
        return 0.0, ""
    try:
        with supervisor_log.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, Mapping):
                    continue
                epoch = _positive_epoch(payload.get("time"))
                if epoch > 0:
                    event = str(payload.get("event") or "supervisor").strip() or "supervisor"
                    return epoch, f"supervisor:{event}"
    except OSError:
        return 0.0, ""
    return 0.0, ""


def _run_start_epoch(
    output_dir: Path,
    *,
    heartbeat: Mapping[str, Any],
    runner_lock: Mapping[str, Any],
) -> tuple[float, str]:
    supervisor_epoch, supervisor_source = _first_supervisor_epoch(output_dir)
    candidates: list[tuple[float, str]] = []
    if supervisor_epoch > 0:
        candidates.append((supervisor_epoch, supervisor_source))
    for source, payload in (("heartbeat", heartbeat), ("runner_lock", runner_lock)):
        for key in ("run_started_epoch", "started_epoch"):
            epoch = _positive_epoch(payload.get(key))
            if epoch > 0:
                candidates.append((epoch, f"{source}:{key}"))
    if not candidates:
        return 0.0, ""
    return min(candidates, key=lambda item: item[0])


def _projected_wall_time(
    output_dir: Path,
    *,
    heartbeat: Mapping[str, Any],
    runner_lock: Mapping[str, Any],
    heartbeat_status: str,
    expected_cases: int,
    processed_cases: int,
    max_projected_duration_hours: float,
    min_rate_cases: int,
) -> dict[str, Any] | None:
    duration_limit = _finite_threshold(max_projected_duration_hours, 0.0)
    if duration_limit <= 0:
        return None
    active_floor = max(1, int(min_rate_cases or 0))
    nonterminal = heartbeat_status not in {"completed", "failed"}
    if nonterminal and processed_cases < active_floor:
        return {
            "name": "projected_wall_time",
            "status": "ok",
            "detail": (
                f"runtime projection waits for {active_floor} processed cases; "
                f"{processed_cases} case(s) are complete"
            ),
            "processed_cases": processed_cases,
            "expected_cases": expected_cases,
            "min_rate_cases": active_floor,
            "max_projected_duration_hours": duration_limit,
            "active": False,
        }
    if expected_cases <= 0 or processed_cases <= 0:
        return {
            "name": "projected_wall_time",
            "status": "error",
            "detail": "cannot project wall time without expected and processed case counts",
            "processed_cases": processed_cases,
            "expected_cases": expected_cases,
            "max_projected_duration_hours": duration_limit,
            "active": True,
        }
    start_epoch, start_source = _run_start_epoch(
        output_dir,
        heartbeat=heartbeat,
        runner_lock=runner_lock,
    )
    if start_epoch <= 0:
        return {
            "name": "projected_wall_time",
            "status": "error",
            "detail": "cannot project wall time without run start telemetry",
            "processed_cases": processed_cases,
            "expected_cases": expected_cases,
            "max_projected_duration_hours": duration_limit,
            "active": True,
        }
    elapsed_seconds = max(0.001, time.time() - start_epoch)
    cases_per_hour = processed_cases / (elapsed_seconds / 3600.0)
    projected_duration_hours = (elapsed_seconds / processed_cases) * expected_cases / 3600.0
    remaining_cases = max(0, expected_cases - processed_cases)
    remaining_hours = remaining_cases / cases_per_hour if cases_per_hour > 0 else math.inf
    status = "ok" if projected_duration_hours <= duration_limit else "error"
    detail = (
        f"{expected_cases} cases project to {projected_duration_hours:.2f}h at current wall-clock throughput"
        if status == "ok"
        else (
            f"{expected_cases} cases project to {projected_duration_hours:.2f}h at current wall-clock throughput, "
            f"above {duration_limit:.2f}h"
        )
    )
    return {
        "name": "projected_wall_time",
        "status": status,
        "detail": detail,
        "processed_cases": processed_cases,
        "expected_cases": expected_cases,
        "remaining_cases": remaining_cases,
        "elapsed_seconds": elapsed_seconds,
        "cases_per_hour": cases_per_hour,
        "projected_duration_hours": projected_duration_hours,
        "remaining_hours": remaining_hours,
        "max_projected_duration_hours": duration_limit,
        "run_start_epoch": start_epoch,
        "run_start_source": start_source,
        "active": True,
    }


def audit_soak(
    output_dir: Path,
    *,
    max_heartbeat_age_seconds: float = 600.0,
    allow_running_incomplete: bool = False,
    max_failed_case_rate: float = 0.0,
    max_quality_failure_rate: float = 0.0,
    max_recovery_event_case_rate: float = DEFAULT_MAX_RECOVERY_EVENT_CASE_RATE,
    max_loop_recovery_case_rate: float | None = None,
    max_loop_guard_case_rate: float | None = None,
    max_deterministic_recovery_case_rate: float | None = None,
    max_failed_attempt_row_rate: float = 0.25,
    max_signal_exit_attempt_row_rate: float | None = None,
    max_attempt_overrun_seconds: float = 60.0,
    max_projected_duration_hours: float = 0.0,
    min_free_gb: float = 0.0,
    min_rate_cases: int = 20,
    set_and_forget: bool = False,
    require_saved_text_labels: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve(strict=False)
    checks: list[dict[str, Any]] = []
    if not output_dir.exists() or not output_dir.is_dir():
        _add_check(checks, "artifact_directory", "error", f"missing directory: {output_dir}")
        return {
            "status": "error",
            "output_dir": str(output_dir),
            "checks": checks,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    _add_check(checks, "artifact_directory", "ok", "directory exists")
    disk_reserve = _disk_reserve_check(output_dir, min_free_gb)
    if disk_reserve is not None:
        checks.append(disk_reserve)

    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    expected_cases = 0
    if manifest_path.exists():
        try:
            raw_manifest = _read_json(manifest_path)
            if isinstance(raw_manifest, dict):
                manifest = raw_manifest
            expected_cases = len(manifest.get("cases") or [])
            _add_check(checks, "manifest", "ok", f"{expected_cases} expected cases")
        except Exception as exc:  # noqa: BLE001
            _add_check(checks, "manifest", "error", f"invalid manifest: {exc}")
    else:
        _add_check(checks, "manifest", "warn", "manifest.json is missing")

    heartbeat_path = output_dir / "heartbeat.json"
    heartbeat: dict[str, Any] = {}
    heartbeat_status = ""
    heartbeat_load_error = ""
    if heartbeat_path.exists():
        try:
            raw_heartbeat = _read_json(heartbeat_path)
            if isinstance(raw_heartbeat, dict):
                heartbeat = raw_heartbeat
            heartbeat_status = str(heartbeat.get("status") or "").lower()
        except Exception as exc:  # noqa: BLE001
            heartbeat_load_error = str(exc)

    results_path = output_dir / "results.jsonl"
    rows, result_jsonl_errors = _read_jsonl_with_errors(results_path)
    latest_rows = _latest_rows(rows)
    processed = len(latest_rows)
    if result_jsonl_errors:
        _add_check(
            checks,
            "results_jsonl",
            "error",
            f"results.jsonl has {len(result_jsonl_errors)} invalid row(s)",
            invalid_rows=result_jsonl_errors[:10],
        )
    elif rows:
        _add_check(checks, "results_jsonl", "ok", f"{len(rows)} rows, {processed} latest cases")
    elif allow_running_incomplete and heartbeat_status == "running":
        _add_check(checks, "results_jsonl", "ok", "running run has not written result rows yet")
    else:
        _add_check(checks, "results_jsonl", "error", "results.jsonl is missing or empty")

    captions_path = output_dir / "captions.jsonl"
    caption_rows, caption_jsonl_errors = _read_jsonl_with_errors(captions_path)
    caption_coverage: dict[str, Any] | None = None
    saved_text_label_coverage: dict[str, Any] | None = None
    if caption_jsonl_errors:
        _add_check(
            checks,
            "captions_jsonl",
            "error",
            f"captions.jsonl has {len(caption_jsonl_errors)} invalid row(s)",
            invalid_rows=caption_jsonl_errors[:10],
        )
    elif captions_path.exists():
        _add_check(checks, "captions_jsonl", "ok", f"{len(caption_rows)} caption rows")

    if not result_jsonl_errors and not caption_jsonl_errors:
        caption_coverage = _caption_coverage_check(
            latest_rows=latest_rows,
            caption_rows=caption_rows,
        )
        _add_check(
            checks,
            "caption_coverage",
            str(caption_coverage["status"]),
            str(caption_coverage["detail"]),
            required_generated_successes=caption_coverage["required_generated_successes"],
            covered_generated_successes=caption_coverage["covered_generated_successes"],
            caption_rows=caption_coverage["caption_rows"],
            missing_caption_rows=caption_coverage["missing_caption_rows"],
            missing_cases=caption_coverage["missing_cases"],
            invalid_caption_rows=caption_coverage["invalid_caption_rows"],
        )
    if require_saved_text_labels and not result_jsonl_errors:
        saved_text_label_coverage = _saved_text_label_coverage_check(
            output_dir=output_dir,
            latest_rows=latest_rows,
        )
        _add_check(
            checks,
            "saved_text_label_coverage",
            str(saved_text_label_coverage["status"]),
            str(saved_text_label_coverage["detail"]),
            required_saved_text_labels=saved_text_label_coverage["required_saved_text_labels"],
            covered_saved_text_labels=saved_text_label_coverage["covered_saved_text_labels"],
            missing_saved_text_label_rows=saved_text_label_coverage["missing_saved_text_label_rows"],
            missing_saved_text_label_files=saved_text_label_coverage["missing_saved_text_label_files"],
            empty_saved_text_label_files=saved_text_label_coverage["empty_saved_text_label_files"],
            unreadable_saved_text_label_files=saved_text_label_coverage["unreadable_saved_text_label_files"],
            missing_cases=saved_text_label_coverage["missing_cases"],
            missing_files=saved_text_label_coverage["missing_files"],
            empty_files=saved_text_label_coverage["empty_files"],
            unreadable_files=saved_text_label_coverage["unreadable_files"],
        )

    totals = Counter(str(row.get("final_status") or row.get("status") or "unknown") for row in latest_rows)
    pending_failed_attempt_cases = sum(
        1
        for row in latest_rows
        if _is_live_retryable_failed_attempt(
            row,
            allow_running_incomplete=allow_running_incomplete,
            heartbeat_status=heartbeat_status,
        )
    )
    failed_cases = sum(
        1
        for row in latest_rows
        if _final_status(row) not in FINAL_OK_STATUSES
        and not _is_live_retryable_failed_attempt(
            row,
            allow_running_incomplete=allow_running_incomplete,
            heartbeat_status=heartbeat_status,
        )
    )
    quality_failed = sum(1 for row in latest_rows if row.get("quality_failures"))
    if failed_cases:
        _add_check(checks, "failed_cases", "warn", f"{failed_cases} latest cases need review")
    elif pending_failed_attempt_cases:
        _add_check(
            checks,
            "failed_cases",
            "ok",
            f"{pending_failed_attempt_cases} retryable failed attempt(s) are still pending in a running job",
        )
    else:
        _add_check(checks, "failed_cases", "ok", "no latest failed cases")
    if quality_failed:
        _add_check(checks, "quality_failures", "warn", f"{quality_failed} latest cases have quality warnings")
    else:
        _add_check(checks, "quality_failures", "ok", "no latest quality warnings")

    degraded_rates, active_degraded_violations = _degraded_rate_checks(
        latest_rows=latest_rows,
        rows=rows,
        expected_cases=expected_cases,
        failed_cases=failed_cases,
        pending_failed_attempt_cases=pending_failed_attempt_cases,
        quality_failed=quality_failed,
        heartbeat_status=heartbeat_status,
        max_failed_case_rate=_finite_threshold(max_failed_case_rate, 0.0),
        max_quality_failure_rate=_finite_threshold(max_quality_failure_rate, 0.0),
        max_recovery_event_case_rate=_finite_threshold(
            max_recovery_event_case_rate,
            DEFAULT_MAX_RECOVERY_EVENT_CASE_RATE,
        ),
        max_loop_recovery_case_rate=resolve_loop_recovery_threshold(
            max_loop_recovery_case_rate,
            set_and_forget=set_and_forget,
        ),
        max_loop_guard_case_rate=resolve_loop_guard_threshold(
            max_loop_guard_case_rate,
            set_and_forget=set_and_forget,
        ),
        max_deterministic_recovery_case_rate=resolve_deterministic_recovery_threshold(
            max_deterministic_recovery_case_rate,
            set_and_forget=set_and_forget,
        ),
        max_failed_attempt_row_rate=_finite_threshold(max_failed_attempt_row_rate, 0.25),
        max_signal_exit_attempt_row_rate=resolve_signal_exit_attempt_threshold(
            max_signal_exit_attempt_row_rate,
            set_and_forget=set_and_forget,
        ),
        min_rate_cases=max(1, int(min_rate_cases or 0)),
    )
    if active_degraded_violations:
        detail = ", ".join(
            f"{item['rate']}={item['actual']:.3f}>{item['threshold']:.3f}"
            for item in active_degraded_violations
        )
        _add_check(
            checks,
            "degraded_case_rates",
            "error",
            f"degraded output rates exceed thresholds: {detail}",
            rates=degraded_rates,
        )
    elif degraded_rates["violations"]:
        deferred_violations = [
            violation
            for violation in degraded_rates["violations"]
            if isinstance(violation, Mapping)
            and (violation.get("deferred_by_sample_floor") or violation.get("deferred_until_terminal"))
        ]
        if deferred_violations and len(deferred_violations) == len(degraded_rates["violations"]):
            all_sample_floor_deferred = all(
                bool(violation.get("deferred_by_sample_floor"))
                for violation in deferred_violations
            )
            all_terminal_deferred = all(
                bool(violation.get("deferred_until_terminal"))
                for violation in deferred_violations
            )
            if all_sample_floor_deferred:
                detail = "degraded output rates are above thresholds but below its activation floor"
            elif all_terminal_deferred:
                detail = "signal-exit attempts are tracked during live recovery and enforced by terminal audit"
            else:
                detail = "degraded output rates are being monitored below their live activation conditions"
            _add_check(
                checks,
                "degraded_case_rates",
                "ok",
                detail,
                rates=degraded_rates,
            )
        else:
            _add_check(
                checks,
                "degraded_case_rates",
                "warn",
                "degraded output rates exceed thresholds but sample is below the active gate",
                rates=degraded_rates,
            )
    else:
        _add_check(
            checks,
            "degraded_case_rates",
            "ok",
            "degraded output rates are within thresholds",
            rates=degraded_rates,
        )
    caution_rates = [
        item
        for item in (degraded_rates.get("caution_rates") or [])
        if isinstance(item, Mapping)
    ]
    if caution_rates:
        caution_detail = ", ".join(
            (
                f"{item.get('rate')} at "
                f"{float(item.get('threshold_fraction_used') or 0.0):.1%} of threshold"
                + (
                    f", {int(item.get('remaining_count_at_terminal_denominator'))} terminal-count budget remains"
                    if item.get("remaining_count_at_terminal_denominator") is not None
                    else ""
                )
            )
            for item in caution_rates[:5]
        )
        _add_check(
            checks,
            "degraded_rate_headroom",
            "ok",
            f"degraded rates are within thresholds but near caution band: {caution_detail}",
            caution_rates=caution_rates,
            rate_headroom=degraded_rates.get("rate_headroom") or [],
        )
    else:
        _add_check(
            checks,
            "degraded_rate_headroom",
            "ok",
            "degraded rates have threshold headroom",
            rate_headroom=degraded_rates.get("rate_headroom") or [],
        )

    summary_path = output_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            raw_summary = _read_json(summary_path)
            if isinstance(raw_summary, dict):
                summary = raw_summary
            summary_totals = dict(summary.get("totals") or {})
            summary_matches = summary_totals == dict(totals) and int(summary.get("total_cases") or 0) == processed
            if summary_matches:
                _add_check(checks, "summary_consistency", "ok", "summary matches latest results")
            elif allow_running_incomplete and heartbeat_status == "running":
                _add_check(
                    checks,
                    "summary_consistency",
                    "ok",
                    "summary does not yet match latest results during running job",
                    summary_totals=summary_totals,
                    latest_totals=dict(totals),
                    summary_total_cases=summary.get("total_cases"),
                    latest_total_cases=processed,
                )
            else:
                _add_check(
                    checks,
                    "summary_consistency",
                    "error",
                    "summary does not match latest results",
                    summary_totals=summary_totals,
                    latest_totals=dict(totals),
                    summary_total_cases=summary.get("total_cases"),
                    latest_total_cases=processed,
                )
        except Exception as exc:  # noqa: BLE001
            _add_check(checks, "summary_consistency", "error", f"invalid summary: {exc}")
    elif allow_running_incomplete and heartbeat_status == "running" and not rows:
        _add_check(checks, "summary_consistency", "ok", "running run has not written a summary yet")
    else:
        _add_check(checks, "summary_consistency", "warn", "summary.json is missing")

    if heartbeat_path.exists():
        if heartbeat_load_error:
            _add_check(checks, "heartbeat", "error", f"invalid heartbeat: {heartbeat_load_error}")
        else:
            heartbeat_status = str(heartbeat.get("status") or "").lower()
            heartbeat_age = _heartbeat_age_seconds(heartbeat, heartbeat_path)
            if heartbeat_status == "running" and max_heartbeat_age_seconds > 0 and heartbeat_age > max_heartbeat_age_seconds:
                _add_check(
                    checks,
                    "heartbeat",
                    "error",
                    f"running heartbeat is stale by {heartbeat_age:.1f}s",
                    age_seconds=heartbeat_age,
                )
            else:
                _add_check(
                    checks,
                    "heartbeat",
                    "ok",
                    f"status={heartbeat_status or 'unknown'} age={heartbeat_age:.1f}s",
                    age_seconds=heartbeat_age,
                )
    else:
        _add_check(checks, "heartbeat", "warn", "heartbeat.json is missing")

    attempt_grace = _finite_threshold(max_attempt_overrun_seconds, 60.0)
    active_attempt = _active_attempt_runtime(heartbeat)
    if attempt_grace >= 0 and active_attempt:
        runtime = float(active_attempt["runtime_seconds"])
        timeout = float(active_attempt["attempt_timeout_seconds"])
        allowed = timeout + attempt_grace
        if runtime > allowed:
            _add_check(
                checks,
                "attempt_timeout_overrun",
                "error",
                f"active attempt has run {runtime:.1f}s, exceeding timeout {timeout:.1f}s plus grace {attempt_grace:.1f}s",
                runtime_seconds=runtime,
                attempt_timeout_seconds=timeout,
                grace_seconds=attempt_grace,
                overrun_seconds=runtime - allowed,
            )
        else:
            _add_check(
                checks,
                "attempt_timeout_overrun",
                "ok",
                f"active attempt runtime {runtime:.1f}s is within timeout {timeout:.1f}s plus grace {attempt_grace:.1f}s",
                runtime_seconds=runtime,
                attempt_timeout_seconds=timeout,
                grace_seconds=attempt_grace,
            )
    elif attempt_grace >= 0:
        _add_check(
            checks,
            "attempt_timeout_overrun",
            "ok",
            "no active attempt with timeout metadata is overdue",
            grace_seconds=attempt_grace,
        )

    lock_path = output_dir / RUNNER_LOCK_NAME
    runner_lock: dict[str, Any] = {}
    if lock_path.exists():
        try:
            raw_lock = _read_json(lock_path)
            if isinstance(raw_lock, dict):
                runner_lock = raw_lock
            lock_pid_alive = _pid_is_alive(runner_lock.get("pid"))
            lock_age = _epoch_age_seconds(runner_lock.get("heartbeat_epoch"))
            lock_phase = str(runner_lock.get("phase") or "").strip() or "unknown"
            if heartbeat_status in {"completed", "failed"}:
                _add_check(
                    checks,
                    "runner_lock",
                    "warn",
                    "runner lock remains after terminal heartbeat",
                    age_seconds=lock_age,
                    pid_alive=lock_pid_alive,
                    phase=lock_phase,
                )
            elif not lock_pid_alive:
                _add_check(
                    checks,
                    "runner_lock",
                    "error",
                    "running heartbeat has a runner lock whose owner pid is not alive",
                    age_seconds=lock_age,
                    pid_alive=lock_pid_alive,
                    phase=lock_phase,
                )
            elif max_heartbeat_age_seconds > 0 and lock_age > max_heartbeat_age_seconds:
                detail = (
                    f"runner lock owner is alive but stale by {lock_age:.1f}s"
                    if lock_pid_alive
                    else f"runner lock is stale by {lock_age:.1f}s"
                )
                _add_check(
                    checks,
                    "runner_lock",
                    "error",
                    detail,
                    age_seconds=lock_age,
                    pid_alive=lock_pid_alive,
                    phase=lock_phase,
                )
            else:
                _add_check(
                    checks,
                    "runner_lock",
                    "ok",
                    f"runner lock phase={lock_phase} age={lock_age:.1f}s",
                    age_seconds=lock_age,
                    pid_alive=lock_pid_alive,
                    phase=lock_phase,
                )
        except Exception as exc:  # noqa: BLE001
            _add_check(checks, "runner_lock", "error", f"invalid runner lock: {exc}")
    elif heartbeat_status == "running":
        _add_check(checks, "runner_lock", "warn", "running heartbeat has no runner lock")
    else:
        _add_check(checks, "runner_lock", "ok", "no active runner lock")

    incomplete_cases = max(0, int(expected_cases or 0) - processed)
    heartbeat_status = str(heartbeat.get("status") or "").lower()
    runtime_projection = _projected_wall_time(
        output_dir,
        heartbeat=heartbeat,
        runner_lock=runner_lock,
        heartbeat_status=heartbeat_status,
        expected_cases=expected_cases,
        processed_cases=processed,
        max_projected_duration_hours=max_projected_duration_hours,
        min_rate_cases=min_rate_cases,
    )
    if runtime_projection is not None:
        checks.append(runtime_projection)
    if expected_cases and incomplete_cases:
        if allow_running_incomplete and heartbeat_status == "running":
            status = "ok"
        else:
            status = "error" if heartbeat_status in {"completed", "failed"} else "warn"
        _add_check(
            checks,
            "case_coverage",
            status,
            f"{processed}/{expected_cases} cases have latest result rows",
            incomplete_cases=incomplete_cases,
        )
    elif expected_cases:
        _add_check(checks, "case_coverage", "ok", f"{processed}/{expected_cases} cases have latest result rows")

    resumable = bool(
        manifest_path.exists()
        and (results_path.exists() or (allow_running_incomplete and heartbeat_status == "running"))
    )
    resumable_detail = "manifest and results are present"
    if resumable and not results_path.exists():
        resumable_detail = "manifest is present; resume can start from zero result rows"
    _add_check(
        checks,
        "resumability",
        "ok" if resumable else "warn",
        resumable_detail if resumable else "resume needs manifest.json and results.jsonl",
    )

    status = max((check["status"] for check in checks), key=_status_rank)
    return {
        "status": status,
        "output_dir": str(output_dir),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "expected_cases": expected_cases,
        "processed_cases": processed,
        "incomplete_cases": incomplete_cases,
        "failed_cases": failed_cases,
        "pending_failed_attempt_cases": pending_failed_attempt_cases,
        "quality_failed_cases": quality_failed,
        "degraded_rates": degraded_rates,
        "runtime_projection": runtime_projection,
        "disk_reserve": disk_reserve,
        "caption_coverage": caption_coverage,
        "saved_text_label_coverage": saved_text_label_coverage,
        "totals": dict(totals),
        "resumable": resumable,
        "heartbeat": heartbeat,
        "active_attempt": active_attempt,
        "runner_lock": runner_lock,
        "checks": checks,
    }


def _compact_percent(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return "n/a"
    if parsed != parsed:
        return "n/a"
    return f"{parsed * 100:.2f}%"


def _compact_hours(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return "n/a"
    if parsed != parsed:
        return "n/a"
    return f"{parsed:.2f}h"


def _compact_seconds(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return "n/a"
    if parsed != parsed:
        return "n/a"
    return f"{parsed:.1f}s"


def _compact_count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _compact_headroom(
    rates: Mapping[str, Any],
    rate_name: str,
) -> Mapping[str, Any]:
    raw_headroom = rates.get("rate_headroom")
    if not isinstance(raw_headroom, list):
        return {}
    for item in raw_headroom:
        if isinstance(item, Mapping) and item.get("rate") == rate_name:
            return item
    return {}


def _compact_rate_line(
    *,
    label: str,
    rates: Mapping[str, Any],
    numerator_key: str,
    rate_key: str,
    threshold_key: str,
    headroom_key: str,
) -> str:
    numerator = _compact_count(rates.get(numerator_key))
    rate = _compact_percent(rates.get(rate_key))
    thresholds = rates.get("thresholds") if isinstance(rates.get("thresholds"), Mapping) else {}
    threshold = _compact_percent(thresholds.get(threshold_key))
    headroom = _compact_headroom(rates, headroom_key)
    terminal_remaining = headroom.get("remaining_count_at_terminal_denominator")
    current_remaining = headroom.get("remaining_count_at_current_denominator")
    if terminal_remaining is not None:
        budget = f", terminal budget remaining {terminal_remaining}"
    elif current_remaining is not None:
        budget = f", current budget remaining {current_remaining}"
    else:
        budget = ""
    return f"{label}: {numerator} ({rate}; cap {threshold}{budget})"


def _compact_check_counts(report: Mapping[str, Any]) -> dict[str, int]:
    counts = {"ok": 0, "warn": 0, "error": 0}
    checks = report.get("checks")
    if not isinstance(checks, list):
        return counts
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        status = str(check.get("status") or "").strip().lower()
        if status in counts:
            counts[status] += 1
    return counts


def format_compact_report(report: Mapping[str, Any]) -> str:
    """Return a concise operator summary for a soak audit report."""
    status = str(report.get("status") or "unknown").upper()
    output_dir = str(report.get("output_dir") or "")
    expected = _compact_count(report.get("expected_cases"))
    processed = _compact_count(report.get("processed_cases"))
    incomplete = _compact_count(report.get("incomplete_cases"))
    heartbeat = report.get("heartbeat") if isinstance(report.get("heartbeat"), Mapping) else {}
    rates = report.get("degraded_rates") if isinstance(report.get("degraded_rates"), Mapping) else {}
    projection = (
        report.get("runtime_projection")
        if isinstance(report.get("runtime_projection"), Mapping)
        else {}
    )
    signal_names = rates.get("signal_exit_names") if isinstance(rates.get("signal_exit_names"), Mapping) else {}
    signal_suffix = ""
    if signal_names:
        signal_suffix = ", " + ", ".join(
            f"{name}={count}" for name, count in sorted(signal_names.items())
        )
    check_counts = _compact_check_counts(report)
    active_attempt_raw = report.get("active_attempt")
    active_attempt = (
        active_attempt_raw
        if isinstance(active_attempt_raw, Mapping)
        else _active_attempt_runtime(heartbeat)
    )
    caption_coverage = (
        report.get("caption_coverage")
        if isinstance(report.get("caption_coverage"), Mapping)
        else {}
    )
    saved_text_label_coverage = (
        report.get("saved_text_label_coverage")
        if isinstance(report.get("saved_text_label_coverage"), Mapping)
        else {}
    )
    active_case = (
        str(active_attempt.get("case") or "").strip()
        if active_attempt is not None
        else str(heartbeat.get("case") or "").strip()
    )
    active_attempt_number = (
        active_attempt.get("attempt")
        if active_attempt is not None and "attempt" in active_attempt
        else heartbeat.get("attempt")
    )
    active_case_index = (
        active_attempt.get("case_index")
        if active_attempt is not None and "case_index" in active_attempt
        else heartbeat.get("case_index")
    )
    disk_reserve = report.get("disk_reserve") if isinstance(report.get("disk_reserve"), Mapping) else {}
    lines = [
        f"Qwen caption soak: {status}",
        f"Output: {output_dir}",
        f"Progress: {processed}/{expected} cases complete ({incomplete} remaining)",
        (
            "Caption rows: "
            f"{_compact_count(caption_coverage.get('covered_generated_successes'))}/"
            f"{_compact_count(caption_coverage.get('required_generated_successes'))} "
            "generated/resumed successes covered"
        ),
        (
            "Heartbeat: "
            f"{heartbeat.get('status') or 'unknown'} / {heartbeat.get('phase') or 'unknown'}"
        ),
        (
            "Failures: "
            f"{_compact_count(report.get('failed_cases'))} latest failed, "
            f"{_compact_count(report.get('quality_failed_cases'))} quality warnings, "
            f"{_compact_count(report.get('pending_failed_attempt_cases'))} pending failed attempts"
        ),
        _compact_rate_line(
            label="Recovery events",
            rates=rates,
            numerator_key="recovery_event_cases",
            rate_key="recovery_event_case_rate",
            threshold_key="max_recovery_event_case_rate",
            headroom_key="recovery_event_case_rate",
        ),
        _compact_rate_line(
            label="Loop recovery",
            rates=rates,
            numerator_key="loop_recovery_cases",
            rate_key="loop_recovery_case_rate",
            threshold_key="max_loop_recovery_case_rate",
            headroom_key="loop_recovery_case_rate",
        ),
        _compact_rate_line(
            label="Loop guards",
            rates=rates,
            numerator_key="loop_guard_cases",
            rate_key="loop_guard_case_rate",
            threshold_key="max_loop_guard_case_rate",
            headroom_key="loop_guard_case_rate",
        ),
        _compact_rate_line(
            label="Deterministic recovery",
            rates=rates,
            numerator_key="deterministic_recovery_cases",
            rate_key="deterministic_recovery_case_rate",
            threshold_key="max_deterministic_recovery_case_rate",
            headroom_key="deterministic_recovery_case_rate",
        ),
        (
            "Prompt budget: "
            f"{_compact_count(rates.get('max_prompt_tokens'))} max prompt tokens, "
            f"{_compact_count(rates.get('prompt_budget_rows'))}/{processed} rows with telemetry, "
            f"{_compact_count(rates.get('prompt_budget_adapted_cases'))} adapted "
            f"({_compact_percent(rates.get('prompt_budget_adapted_case_rate'))})"
        ),
        (
            _compact_rate_line(
                label="Signal exits",
                rates=rates,
                numerator_key="signal_exit_attempt_rows",
                rate_key="signal_exit_attempt_row_rate",
                threshold_key="max_signal_exit_attempt_row_rate",
                headroom_key="signal_exit_attempt_row_rate",
            )
            + signal_suffix
        ),
    ]
    if saved_text_label_coverage:
        lines.insert(
            4,
            (
                "Saved text labels: "
                f"{_compact_count(saved_text_label_coverage.get('covered_saved_text_labels'))}/"
                f"{_compact_count(saved_text_label_coverage.get('required_saved_text_labels'))} "
                "generated/resumed successes saved"
            ),
        )
    if active_attempt is not None:
        case_bits = []
        if active_case:
            case_bits.append(active_case)
        if active_case_index is not None:
            case_bits.append(f"index {active_case_index}")
        if active_attempt_number is not None:
            case_bits.append(f"attempt {active_attempt_number}")
        case_text = ", ".join(case_bits) if case_bits else "current case"
        lines.insert(
            4,
            (
                "Active attempt: "
                f"{case_text}, runtime {_compact_seconds(active_attempt.get('runtime_seconds'))} / "
                f"timeout {_compact_seconds(active_attempt.get('attempt_timeout_seconds'))}"
            ),
        )
    if disk_reserve:
        lines.append(
            "Disk reserve: "
            f"{disk_reserve.get('status') or 'unknown'}; "
            f"{disk_reserve.get('free_human') or 'unknown'} free / "
            f"{disk_reserve.get('min_free_human') or 'unknown'} required"
        )
    if projection:
        lines.append(
            "Runtime projection: "
            f"{_compact_hours(projection.get('projected_duration_hours'))} total, "
            f"{_compact_hours(projection.get('remaining_hours'))} remaining "
            f"(cap {_compact_hours(projection.get('max_projected_duration_hours'))})"
        )
    lines.append(
        "Checks: "
        f"{check_counts['ok']} ok, {check_counts['warn']} warn, {check_counts['error']} error"
    )
    caution_rates = rates.get("caution_rates")
    if isinstance(caution_rates, list) and caution_rates:
        detail_parts = []
        for item in caution_rates:
            if not isinstance(item, Mapping):
                continue
            rate_name = str(item.get("rate") or "unknown")
            fraction = item.get("threshold_fraction_used")
            if fraction is None:
                detail_parts.append(rate_name)
            else:
                detail_parts.append(f"{rate_name} at {_compact_percent(fraction)} of cap")
        if detail_parts:
            lines.append("Attention: " + "; ".join(detail_parts))
    active_violations = rates.get("active_violations")
    if isinstance(active_violations, list) and active_violations:
        violation_names = [
            str(item.get("rate") or item.get("name") or "unknown")
            for item in active_violations
            if isinstance(item, Mapping)
        ]
        if violation_names:
            lines.append("Violations: " + ", ".join(violation_names))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-heartbeat-age", type=float, default=600.0)
    parser.add_argument(
        "--max-failed-case-rate",
        type=float,
        default=0.0,
        help="Maximum latest-case failure rate before the audit errors. Use -1 to disable.",
    )
    parser.add_argument(
        "--max-quality-failure-rate",
        type=float,
        default=0.0,
        help="Maximum latest-case quality-warning rate before the audit errors. Use -1 to disable.",
    )
    parser.add_argument(
        "--max-recovery-event-case-rate",
        type=float,
        default=DEFAULT_MAX_RECOVERY_EVENT_CASE_RATE,
        help="Maximum latest-case recovery-event rate before the audit errors. Use -1 to disable.",
    )
    parser.add_argument(
        "--max-loop-recovery-case-rate",
        type=float,
        default=None,
        help=(
            "Maximum latest-case loop-recovery rate before the audit errors. Use -1 to disable. "
            f"Omit for {DEFAULT_MAX_LOOP_RECOVERY_CASE_RATE:g}, or "
            f"{DEFAULT_SET_AND_FORGET_MAX_LOOP_RECOVERY_CASE_RATE:g} with --set-and-forget."
        ),
    )
    parser.add_argument(
        "--max-loop-guard-case-rate",
        type=float,
        default=None,
        help=(
            "Maximum latest-case stream-loop or loop-trim guard rate before the audit errors. "
            "Use -1 to disable. Omit for "
            f"{DEFAULT_MAX_LOOP_GUARD_CASE_RATE:g}, or "
            f"{DEFAULT_SET_AND_FORGET_MAX_LOOP_GUARD_CASE_RATE:g} with --set-and-forget."
        ),
    )
    parser.add_argument(
        "--max-deterministic-recovery-case-rate",
        type=float,
        default=None,
        help=(
            "Maximum latest-case deterministic count/layout recovery rate before the audit errors. "
            "Use -1 to disable. Omit for "
            f"{DEFAULT_MAX_DETERMINISTIC_RECOVERY_CASE_RATE:g}, or "
            f"{DEFAULT_SET_AND_FORGET_MAX_DETERMINISTIC_RECOVERY_CASE_RATE:g} with --set-and-forget."
        ),
    )
    parser.add_argument(
        "--max-failed-attempt-row-rate",
        type=float,
        default=0.25,
        help="Maximum failed attempt-row rate before the audit errors. Use -1 to disable.",
    )
    parser.add_argument(
        "--max-signal-exit-attempt-row-rate",
        type=float,
        default=None,
        help=(
            "Maximum signal-exit attempt-row rate before the audit errors. Use -1 to disable. "
            f"Omit for {DEFAULT_MAX_SIGNAL_EXIT_ATTEMPT_ROW_RATE:g}, or "
            f"{DEFAULT_SET_AND_FORGET_MAX_SIGNAL_EXIT_ATTEMPT_ROW_RATE:g} with --set-and-forget."
        ),
    )
    parser.add_argument(
        "--max-attempt-overrun",
        type=float,
        default=60.0,
        help=(
            "Grace seconds beyond the active attempt timeout before a running heartbeat is unhealthy. "
            "Use -1 to disable."
        ),
    )
    parser.add_argument(
        "--max-projected-duration-hours",
        type=float,
        default=0.0,
        help=(
            "Maximum projected all-case wall-clock duration at the current observed throughput. "
            "0 disables this gate."
        ),
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=0.0,
        help="Minimum free disk reserve required during audit/watchdog checks. 0 disables this gate.",
    )
    parser.add_argument(
        "--min-rate-cases",
        type=int,
        default=20,
        help="Minimum processed cases before rate gates are active for a running job. Terminal jobs are always gated.",
    )
    parser.add_argument(
        "--allow-running-incomplete",
        action="store_true",
        help="Treat a fresh running job with incomplete case coverage as healthy.",
    )
    parser.add_argument(
        "--set-and-forget",
        action="store_true",
        help=(
            "Use unattended-run health defaults. This keeps failed and quality rates strict, "
            "but allows small bounded loop-recovery and deterministic-recovery rates when "
            "not overridden."
        ),
    )
    parser.add_argument(
        "--require-saved-text-labels",
        action="store_true",
        help=(
            "Require every generated or resumed-completed successful row to reference an existing "
            "non-empty dataset text-label file."
        ),
    )
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print a concise operator summary instead of the full JSON report.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = audit_soak(
        args.output_dir,
        max_heartbeat_age_seconds=args.max_heartbeat_age,
        allow_running_incomplete=args.allow_running_incomplete,
        max_failed_case_rate=args.max_failed_case_rate,
        max_quality_failure_rate=args.max_quality_failure_rate,
        max_recovery_event_case_rate=args.max_recovery_event_case_rate,
        max_loop_recovery_case_rate=args.max_loop_recovery_case_rate,
        max_loop_guard_case_rate=args.max_loop_guard_case_rate,
        max_deterministic_recovery_case_rate=args.max_deterministic_recovery_case_rate,
        max_failed_attempt_row_rate=args.max_failed_attempt_row_rate,
        max_signal_exit_attempt_row_rate=args.max_signal_exit_attempt_row_rate,
        max_attempt_overrun_seconds=args.max_attempt_overrun,
        max_projected_duration_hours=args.max_projected_duration_hours,
        min_free_gb=args.min_free_gb,
        min_rate_cases=args.min_rate_cases,
        set_and_forget=args.set_and_forget,
        require_saved_text_labels=args.require_saved_text_labels,
    )
    if args.compact:
        print(format_compact_report(report))
    else:
        print(json.dumps(report, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
