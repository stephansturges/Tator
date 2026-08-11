#!/usr/bin/env python3
"""Certify a pilot Qwen caption soak before scaling to a long unattended run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import audit_qwen_caption_soak as audit  # noqa: E402
from tools import run_qwen_caption_flow_benchmark as runner  # noqa: E402


DEFAULT_TARGET_CASES = 10_000
DEFAULT_MAX_DURATION_HOURS = 24 * 14
DEFAULT_MIN_PILOT_CASES = 50
DEFAULT_DURATION_SAFETY_FACTOR = 1.25
DEFAULT_MAX_PROMPT_BUDGET_ADAPTED_CASE_RATE = 1.0
DEFAULT_DETERMINISTIC_RECOVERY_CONFIDENCE = 0.95
SAMPLE_STRATEGY_ALL = "all"
SAMPLE_STRATEGY_STRESS_PLUS_RANDOM = "stress_plus_random"
SKIPPED_STATUSES = {"skipped_completed", "skipped_existing_caption"}
GENERATED_SUCCESS_STATUSES = {"ok", "preview_only"}
STATUS_RANK = {"ok": 0, "warn": 1, "error": 2}
QWEN_CAPTION_IO_ACCEPTED_RUNTIME_SOURCES = {"qwen_caption_io_per_run"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_rank(status: str) -> int:
    return STATUS_RANK.get(str(status or "error"), 2)


def _add_check(checks: list[dict[str, Any]], name: str, status: str, detail: str, **fields: Any) -> None:
    checks.append({"name": name, "status": status, "detail": detail, **fields})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        amount = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if amount != amount or amount in {float("inf"), float("-inf")}:
        return default
    return amount


def _bounded_confidence(value: Any) -> float:
    confidence = _safe_float(value, DEFAULT_DETERMINISTIC_RECOVERY_CONFIDENCE)
    if confidence <= 0:
        return 0.0
    if confidence >= 1:
        return DEFAULT_DETERMINISTIC_RECOVERY_CONFIDENCE
    return max(0.5, confidence)


def _one_sided_normal_z(confidence: float) -> float:
    return statistics.NormalDist().inv_cdf(confidence)


def _wilson_upper_bound(numerator: int, denominator: int, *, confidence: float) -> float:
    if denominator <= 0:
        return 1.0
    numerator = max(0, min(int(numerator), int(denominator)))
    denominator = int(denominator)
    z = _one_sided_normal_z(confidence)
    z2 = z * z
    phat = float(numerator) / float(denominator)
    base = phat + z2 / (2.0 * denominator)
    radius = z * math.sqrt(
        phat * (1.0 - phat) / float(denominator)
        + z2 / (4.0 * denominator * denominator)
    )
    return max(0.0, min(1.0, (base + radius) / (1.0 + z2 / denominator)))


def _zero_event_cases_for_wilson_limit(limit: float, *, confidence: float) -> int:
    if limit <= 0:
        return 0
    z = _one_sided_normal_z(confidence)
    required = (z * z) * (1.0 - limit) / limit
    return max(1, int(math.ceil(required)))


def _case_key(row: Mapping[str, Any], index: int) -> str:
    return str(
        row.get("case_id")
        or row.get("case")
        or row.get("stem")
        or f"row_{index}"
    ).strip()


def _latest_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        key = _case_key(row, index)
        if key:
            latest[key] = row
    return latest


def _row_recovery_events(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = row.get("recovery_events")
    if not isinstance(raw, list):
        return []
    return [event for event in raw if isinstance(event, Mapping)]


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


def _deterministic_recovery_reliability_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: float,
    confidence: float,
    enabled: bool,
) -> dict[str, Any]:
    row_count = len(rows)
    deterministic_rows = [
        _case_key(row, index)
        for index, row in enumerate(rows)
        if _row_has_deterministic_recovery(row)
    ]
    recovery_cases = len(deterministic_rows)
    observed_rate = float(recovery_cases) / float(row_count) if row_count else 0.0
    upper_bound_rate = (
        _wilson_upper_bound(recovery_cases, row_count, confidence=confidence)
        if enabled and confidence > 0
        else observed_rate
    )
    return {
        "enabled": enabled,
        "confidence": confidence,
        "limit": limit,
        "generated_cases": row_count,
        "deterministic_recovery_cases": recovery_cases,
        "observed_rate": observed_rate,
        "upper_bound_rate": upper_bound_rate,
        "required_zero_recovery_cases": (
            _zero_event_cases_for_wilson_limit(limit, confidence=confidence)
            if enabled and confidence > 0 and limit > 0
            else 0
        ),
        "deterministic_recovery_case_keys_sample": sorted(deterministic_rows)[:20],
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    clean = sorted(value for value in values if value >= 0)
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * max(0.0, min(float(percentile), 100.0)) / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(clean) - 1)
    fraction = rank - lower
    return clean[lower] * (1.0 - fraction) + clean[upper] * fraction


def _case_runtime_seconds(
    rows: Sequence[Mapping[str, Any]],
    *,
    generated_case_keys: set[str] | None = None,
) -> dict[str, float]:
    runtimes: dict[str, float] = {}
    for index, row in enumerate(rows):
        status = str(row.get("final_status") or row.get("status") or "").strip()
        if status in SKIPPED_STATUSES:
            continue
        key = _case_key(row, index)
        if generated_case_keys is not None and key not in generated_case_keys:
            continue
        elapsed = _safe_float(row.get("elapsed_seconds"), 0.0)
        if elapsed <= 0:
            continue
        if key:
            runtimes[key] = runtimes.get(key, 0.0) + elapsed
    return runtimes


def _generated_case_evidence_summary(
    rows: Sequence[Mapping[str, Any]],
    latest: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    generated_by_key: dict[str, Mapping[str, Any]] = {}
    generated_status_counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        status = str(row.get("final_status") or row.get("status") or "").strip()
        if status in SKIPPED_STATUSES or status not in GENERATED_SUCCESS_STATUSES:
            continue
        key = _case_key(row, index)
        if not key:
            continue
        generated_by_key[key] = row
        generated_status_counts[status] = generated_status_counts.get(status, 0) + 1

    skipped_latest_keys: list[str] = []
    missing_generated_keys: list[str] = []
    for key, row in latest.items():
        status = str(row.get("final_status") or row.get("status") or "").strip()
        if status in SKIPPED_STATUSES:
            skipped_latest_keys.append(key)
        if key not in generated_by_key:
            missing_generated_keys.append(key)

    summary = {
        "latest_cases": len(latest),
        "generated_cases": len(generated_by_key),
        "generated_success_status_counts": dict(sorted(generated_status_counts.items())),
        "latest_skipped_cases": len(skipped_latest_keys),
        "latest_skipped_cases_with_prior_generated_success": sum(
            1 for key in skipped_latest_keys if key in generated_by_key
        ),
        "latest_cases_missing_generated_success": len(missing_generated_keys),
        "missing_generated_case_keys_sample": sorted(missing_generated_keys)[:20],
    }
    return generated_by_key, summary


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


def _prompt_budget_summary(latest_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows_with_budget = [row for row in latest_rows if _row_prompt_budget(row)]
    adapted_cases = sum(1 for row in latest_rows if _row_has_adapted_prompt_budget(row))
    prompt_tokens = [_row_max_prompt_tokens(row) for row in latest_rows]
    max_prompt_tokens = max(prompt_tokens, default=0)
    return {
        "pilot_cases": len(latest_rows),
        "rows_with_prompt_budget": len(rows_with_budget),
        "adapted_cases": adapted_cases,
        "adapted_case_rate": (
            float(adapted_cases) / float(len(latest_rows))
            if latest_rows
            else 0.0
        ),
        "max_prompt_tokens": max_prompt_tokens,
    }


def _qwen_caption_io_source_summary(latest_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required_rows = 0
    runtime_prompt_budget_rows = 0
    valid_runtime_prompt_budget_rows = 0
    missing_runtime_rows: list[dict[str, Any]] = []
    invalid_runtime_rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    missing_trace_rows = 0

    for index, row in enumerate(latest_rows):
        status = str(row.get("final_status") or row.get("status") or "").strip()
        if status in SKIPPED_STATUSES:
            continue
        required_rows += 1
        key = _case_key(row, index)
        raw_io = row.get("qwen_caption_io")
        if not isinstance(raw_io, Mapping) or not raw_io:
            missing_runtime_rows.append({"case_key": key, "reason": "missing_qwen_caption_io"})
            continue
        source = str(raw_io.get("source") or "").strip()
        source_key = source or "missing_source"
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        prompt_budget_events = max(0, _safe_int(raw_io.get("prompt_budget_events"), 0))
        missing_trace = bool(raw_io.get("missing_trace"))
        if prompt_budget_events > 0:
            runtime_prompt_budget_rows += 1
        if missing_trace:
            missing_trace_rows += 1
        if source not in QWEN_CAPTION_IO_ACCEPTED_RUNTIME_SOURCES or missing_trace:
            invalid_runtime_rows.append(
                {
                    "case_key": key,
                    "source": source_key,
                    "missing_trace": missing_trace,
                    "prompt_budget_events": prompt_budget_events,
                    "reason": (
                        str(raw_io.get("missing_trace_reason") or "").strip()
                        if missing_trace
                        else "unbound_or_unsupported_qwen_caption_io_source"
                    ),
                }
            )
            continue
        if prompt_budget_events > 0:
            valid_runtime_prompt_budget_rows += 1

    return {
        "required_rows": required_rows,
        "runtime_prompt_budget_rows": runtime_prompt_budget_rows,
        "valid_runtime_prompt_budget_rows": valid_runtime_prompt_budget_rows,
        "missing_runtime_rows": missing_runtime_rows,
        "missing_runtime_rows_count": len(missing_runtime_rows),
        "invalid_runtime_rows": invalid_runtime_rows,
        "invalid_runtime_rows_count": len(invalid_runtime_rows),
        "missing_trace_rows": missing_trace_rows,
        "source_counts": dict(sorted(source_counts.items())),
        "accepted_sources": sorted(QWEN_CAPTION_IO_ACCEPTED_RUNTIME_SOURCES),
    }


def _normalized_capabilities(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        capability = str(item or "").strip()
        if capability and capability not in seen:
            seen.add(capability)
            normalized.append(capability)
    return normalized


def _runner_capability_summary(
    manifest: Mapping[str, Any],
    heartbeat: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_capabilities = _normalized_capabilities(manifest.get("runner_capabilities"))
    heartbeat_capabilities = _normalized_capabilities(heartbeat.get("runner_capabilities"))
    capabilities = manifest_capabilities or heartbeat_capabilities
    required = list(runner.RUNNER_CAPABILITIES)
    capability_set = set(capabilities)
    return {
        "runner_capabilities": capabilities,
        "required_capabilities": required,
        "missing_capabilities": [
            capability
            for capability in required
            if capability not in capability_set
        ],
        "manifest_runner_capabilities": manifest_capabilities,
        "heartbeat_runner_capabilities": heartbeat_capabilities,
        "capability_sources": [
            source
            for source, values in (
                ("manifest", manifest_capabilities),
                ("heartbeat", heartbeat_capabilities),
            )
            if values
        ],
    }


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _manifest_sample_selection_summary(
    manifest: Mapping[str, Any],
    *,
    pilot_cases: int,
) -> dict[str, Any]:
    sample_size = max(0, _safe_int(manifest.get("sample_size"), 0))
    raw_selection = manifest.get("sample_selection")
    selection = raw_selection if isinstance(raw_selection, Mapping) else {}
    strategy = str(selection.get("strategy") or "").strip()
    stress_case_keys = [
        str(key).strip()
        for key in (selection.get("stress_case_keys") or [])
        if str(key).strip()
    ]
    random_fill_case_keys = [
        str(key).strip()
        for key in (selection.get("random_fill_case_keys") or [])
        if str(key).strip()
    ]
    source_cases = max(0, _safe_int(selection.get("source_cases"), 0))
    selected_cases = max(0, _safe_int(selection.get("selected_cases"), 0))
    return {
        "sample_size": sample_size,
        "sampled": sample_size > 0,
        "has_sample_selection": bool(selection),
        "strategy": strategy,
        "source_cases": source_cases,
        "selected_cases": selected_cases,
        "pilot_cases": pilot_cases,
        "stress_case_keys": stress_case_keys,
        "stress_case_count": len(stress_case_keys),
        "random_fill_case_keys": random_fill_case_keys,
        "random_fill_case_count": len(random_fill_case_keys),
    }


def _timing_summary(case_seconds: Mapping[str, float]) -> dict[str, Any]:
    values = [float(value) for value in case_seconds.values() if float(value) > 0]
    if not values:
        return {
            "timed_cases": 0,
            "mean_case_seconds": 0.0,
            "median_case_seconds": 0.0,
            "p95_case_seconds": 0.0,
            "max_case_seconds": 0.0,
            "cases_per_hour": 0.0,
        }
    mean_seconds = statistics.fmean(values)
    return {
        "timed_cases": len(values),
        "mean_case_seconds": mean_seconds,
        "median_case_seconds": statistics.median(values),
        "p95_case_seconds": _percentile(values, 95),
        "max_case_seconds": max(values),
        "cases_per_hour": 3600.0 / mean_seconds if mean_seconds > 0 else 0.0,
    }


def certify_soak(
    output_dir: Path,
    *,
    target_cases: int = DEFAULT_TARGET_CASES,
    max_duration_hours: float = DEFAULT_MAX_DURATION_HOURS,
    max_p95_duration_hours: float | None = None,
    min_pilot_cases: int = DEFAULT_MIN_PILOT_CASES,
    duration_safety_factor: float = DEFAULT_DURATION_SAFETY_FACTOR,
    max_heartbeat_age_seconds: float = 600.0,
    max_failed_case_rate: float = 0.0,
    max_quality_failure_rate: float = 0.0,
    max_recovery_event_case_rate: float = audit.DEFAULT_MAX_RECOVERY_EVENT_CASE_RATE,
    max_loop_recovery_case_rate: float | None = None,
    max_loop_guard_case_rate: float | None = None,
    max_deterministic_recovery_case_rate: float | None = None,
    max_failed_attempt_row_rate: float = 0.25,
    max_signal_exit_attempt_row_rate: float | None = None,
    max_attempt_overrun_seconds: float = 60.0,
    min_rate_cases: int = 20,
    set_and_forget: bool = False,
    expected_run_settings_fingerprint: str | None = None,
    require_prompt_budget_data: bool = True,
    require_runner_capabilities: bool = True,
    max_prompt_tokens: int = 0,
    max_prompt_budget_adapted_case_rate: float = DEFAULT_MAX_PROMPT_BUDGET_ADAPTED_CASE_RATE,
    deterministic_recovery_confidence: float = DEFAULT_DETERMINISTIC_RECOVERY_CONFIDENCE,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve(strict=False)
    rows = _read_jsonl(output_dir / "results.jsonl")
    latest = _latest_rows(rows)
    latest_rows = list(latest.values())
    generated_rows_by_key, generated_case_evidence = _generated_case_evidence_summary(rows, latest)
    generated_rows = list(generated_rows_by_key.values())
    generated_case_keys = set(generated_rows_by_key)
    case_seconds = _case_runtime_seconds(rows, generated_case_keys=generated_case_keys)
    timing = _timing_summary(case_seconds)
    prompt_budget = _prompt_budget_summary(generated_rows)
    qwen_caption_io_sources = _qwen_caption_io_source_summary(generated_rows)
    manifest_path = output_dir / "manifest.json"
    manifest = _read_json_dict(manifest_path)
    heartbeat = _read_json_dict(output_dir / "heartbeat.json")
    runner_capabilities = _runner_capability_summary(manifest, heartbeat)
    sample_selection = _manifest_sample_selection_summary(manifest, pilot_cases=len(latest))
    manifest_run_settings = (
        manifest.get("run_settings")
        if isinstance(manifest.get("run_settings"), Mapping)
        else {}
    )
    pilot_fingerprint = str(manifest_run_settings.get("fingerprint") or "").strip()
    expected_fingerprint = str(expected_run_settings_fingerprint or "").strip()
    target_cases = max(1, int(target_cases or DEFAULT_TARGET_CASES))
    max_duration_hours = max(0.01, float(max_duration_hours or DEFAULT_MAX_DURATION_HOURS))
    p95_duration_gate = (
        max_duration_hours
        if max_p95_duration_hours is None
        else float(max_p95_duration_hours)
    )
    min_pilot_cases = max(1, int(min_pilot_cases or DEFAULT_MIN_PILOT_CASES))
    duration_safety_factor = max(1.0, float(duration_safety_factor or DEFAULT_DURATION_SAFETY_FACTOR))
    deterministic_recovery_confidence = _bounded_confidence(deterministic_recovery_confidence)
    deterministic_recovery_limit = audit.resolve_deterministic_recovery_threshold(
        max_deterministic_recovery_case_rate,
        set_and_forget=set_and_forget,
    )
    deterministic_recovery_reliability = _deterministic_recovery_reliability_summary(
        generated_rows,
        limit=deterministic_recovery_limit,
        confidence=deterministic_recovery_confidence,
        enabled=bool(
            set_and_forget
            and deterministic_recovery_confidence > 0
            and deterministic_recovery_limit >= 0
        ),
    )
    mean_projected_hours = (
        float(timing["mean_case_seconds"]) * target_cases / 3600.0
        if timing["timed_cases"]
        else 0.0
    )
    safety_projected_hours = mean_projected_hours * duration_safety_factor
    p95_all_cases_hours = (
        float(timing["p95_case_seconds"]) * target_cases / 3600.0
        if timing["timed_cases"]
        else 0.0
    )
    audit_report = audit.audit_soak(
        output_dir,
        max_heartbeat_age_seconds=max_heartbeat_age_seconds,
        allow_running_incomplete=False,
        max_failed_case_rate=max_failed_case_rate,
        max_quality_failure_rate=max_quality_failure_rate,
        max_recovery_event_case_rate=max_recovery_event_case_rate,
        max_loop_recovery_case_rate=max_loop_recovery_case_rate,
        max_loop_guard_case_rate=max_loop_guard_case_rate,
        max_deterministic_recovery_case_rate=max_deterministic_recovery_case_rate,
        max_failed_attempt_row_rate=max_failed_attempt_row_rate,
        max_signal_exit_attempt_row_rate=max_signal_exit_attempt_row_rate,
        max_attempt_overrun_seconds=max_attempt_overrun_seconds,
        min_rate_cases=min_rate_cases,
        set_and_forget=set_and_forget,
    )

    checks: list[dict[str, Any]] = []
    audit_status = str(audit_report.get("status") or "error")
    _add_check(
        checks,
        "artifact_audit",
        "ok" if audit_status == "ok" else "error",
        f"strict artifact audit status is {audit_status}",
        audit_status=audit_status,
    )
    if timing["timed_cases"]:
        _add_check(
            checks,
            "timing_data",
            "ok",
            f"{timing['timed_cases']} cases have positive elapsed_seconds",
        )
    else:
        _add_check(checks, "timing_data", "error", "no positive elapsed_seconds rows found")
    generated_count = int(generated_case_evidence["generated_cases"])
    if generated_count >= min_pilot_cases:
        _add_check(
            checks,
            "generated_case_evidence",
            "ok",
            f"{generated_count} cases have prior successful generated rows",
            generated_cases=generated_count,
            min_pilot_cases=min_pilot_cases,
            latest_cases=generated_case_evidence["latest_cases"],
            latest_skipped_cases=generated_case_evidence["latest_skipped_cases"],
            latest_skipped_cases_with_prior_generated_success=generated_case_evidence[
                "latest_skipped_cases_with_prior_generated_success"
            ],
        )
    else:
        _add_check(
            checks,
            "generated_case_evidence",
            "error",
            f"{generated_count} generated cases is below minimum {min_pilot_cases}",
            generated_cases=generated_count,
            min_pilot_cases=min_pilot_cases,
            latest_cases=generated_case_evidence["latest_cases"],
            latest_skipped_cases=generated_case_evidence["latest_skipped_cases"],
            latest_skipped_cases_with_prior_generated_success=generated_case_evidence[
                "latest_skipped_cases_with_prior_generated_success"
            ],
            missing_generated_case_keys=generated_case_evidence["missing_generated_case_keys_sample"],
        )
    if int(timing["timed_cases"]) >= min_pilot_cases:
        _add_check(
            checks,
            "pilot_sample_size",
            "ok",
            f"{timing['timed_cases']} generated timed cases meets minimum {min_pilot_cases}",
        )
    else:
        _add_check(
            checks,
            "pilot_sample_size",
            "error",
            f"{timing['timed_cases']} generated timed cases is below minimum {min_pilot_cases}",
            timed_cases=timing["timed_cases"],
            min_pilot_cases=min_pilot_cases,
        )
    if sample_selection["sampled"]:
        strategy = str(sample_selection.get("strategy") or "")
        if not sample_selection["has_sample_selection"]:
            _add_check(
                checks,
                "sample_selection",
                "error",
                "sampled pilot manifest is missing stress-sample selection metadata",
                manifest_path=str(manifest_path),
                sample_size=sample_selection["sample_size"],
            )
        elif strategy == SAMPLE_STRATEGY_STRESS_PLUS_RANDOM:
            if int(sample_selection["stress_case_count"]) > 0:
                _add_check(
                    checks,
                    "sample_selection",
                    "ok",
                    "sampled pilot used stress-biased representative case selection",
                    **sample_selection,
                )
            else:
                _add_check(
                    checks,
                    "sample_selection",
                    "error",
                    "stress-biased sampled pilot did not record any stress cases",
                    **sample_selection,
                )
        elif strategy == SAMPLE_STRATEGY_ALL:
            if int(sample_selection["selected_cases"]) == int(sample_selection["source_cases"]) == len(latest):
                _add_check(
                    checks,
                    "sample_selection",
                    "ok",
                    "sample-size covered every available pilot case",
                    **sample_selection,
                )
            else:
                _add_check(
                    checks,
                    "sample_selection",
                    "error",
                    "sample selection metadata says all cases but counts do not match pilot rows",
                    **sample_selection,
                )
        else:
            _add_check(
                checks,
                "sample_selection",
                "error",
                "sampled pilot used an unrecognized sample selection strategy",
                **sample_selection,
            )
    else:
        _add_check(
            checks,
            "sample_selection",
            "ok",
            "pilot manifest did not request case sampling",
            **sample_selection,
        )
    if safety_projected_hours <= max_duration_hours:
        _add_check(
            checks,
            "projected_duration",
            "ok",
            f"{target_cases} cases projected at {safety_projected_hours:.2f}h including safety factor",
            projected_hours=safety_projected_hours,
            max_duration_hours=max_duration_hours,
        )
    else:
        _add_check(
            checks,
            "projected_duration",
            "error",
            f"{target_cases} cases project to {safety_projected_hours:.2f}h, above {max_duration_hours:.2f}h",
            projected_hours=safety_projected_hours,
            max_duration_hours=max_duration_hours,
        )
    if p95_duration_gate < 0:
        _add_check(
            checks,
            "p95_projected_duration",
            "ok",
            "p95 projected-duration gate disabled",
            projected_hours=p95_all_cases_hours,
            max_p95_duration_hours=p95_duration_gate,
        )
    elif p95_all_cases_hours <= p95_duration_gate:
        _add_check(
            checks,
            "p95_projected_duration",
            "ok",
            f"{target_cases} cases at p95 case time project to {p95_all_cases_hours:.2f}h",
            projected_hours=p95_all_cases_hours,
            max_p95_duration_hours=p95_duration_gate,
        )
    else:
        _add_check(
            checks,
            "p95_projected_duration",
            "error",
            (
                f"{target_cases} cases at p95 case time project to {p95_all_cases_hours:.2f}h, "
                f"above {p95_duration_gate:.2f}h"
            ),
            projected_hours=p95_all_cases_hours,
            max_p95_duration_hours=p95_duration_gate,
        )
    median = float(timing.get("median_case_seconds") or 0.0)
    p95 = float(timing.get("p95_case_seconds") or 0.0)
    if median > 0 and p95 > median * 5.0:
        _add_check(
            checks,
            "runtime_tail",
            "ok",
            f"runtime-tail advisory: p95 case time {p95:.2f}s is more than 5x median {median:.2f}s",
            advisory=True,
            median_case_seconds=median,
            p95_case_seconds=p95,
        )
    else:
        _add_check(checks, "runtime_tail", "ok", "runtime tail is within the warning bound")
    if expected_fingerprint:
        if not pilot_fingerprint:
            _add_check(
                checks,
                "run_settings_fingerprint",
                "error",
                "pilot manifest does not include a run settings fingerprint",
                manifest_path=str(manifest_path),
                expected_fingerprint=expected_fingerprint,
            )
        elif pilot_fingerprint == expected_fingerprint:
            _add_check(
                checks,
                "run_settings_fingerprint",
                "ok",
                "pilot run settings match the requested launch settings",
                pilot_fingerprint=pilot_fingerprint,
                expected_fingerprint=expected_fingerprint,
            )
        else:
            _add_check(
                checks,
                "run_settings_fingerprint",
                "error",
                "pilot run settings do not match the requested launch settings",
                pilot_fingerprint=pilot_fingerprint,
                expected_fingerprint=expected_fingerprint,
            )
    if require_prompt_budget_data:
        if prompt_budget["pilot_cases"] and prompt_budget["rows_with_prompt_budget"] == prompt_budget["pilot_cases"]:
            _add_check(
                checks,
                "prompt_budget_data",
                "ok",
                f"{prompt_budget['rows_with_prompt_budget']} pilot cases include prompt-budget telemetry",
            )
        else:
            _add_check(
                checks,
                "prompt_budget_data",
                "error",
                "pilot rows are missing prompt-budget telemetry from the current runner",
                rows_with_prompt_budget=prompt_budget["rows_with_prompt_budget"],
                pilot_cases=prompt_budget["pilot_cases"],
            )
    else:
        _add_check(
            checks,
            "prompt_budget_data",
            "ok",
            "prompt-budget telemetry requirement disabled",
            rows_with_prompt_budget=prompt_budget["rows_with_prompt_budget"],
            pilot_cases=prompt_budget["pilot_cases"],
        )
    invalid_runtime_rows = list(qwen_caption_io_sources.get("invalid_runtime_rows") or [])
    if invalid_runtime_rows:
        _add_check(
            checks,
            "qwen_caption_io_source",
            "error",
            "pilot contains unbound or missing qwen_caption_io runtime evidence",
            invalid_runtime_rows=invalid_runtime_rows[:20],
            invalid_runtime_rows_count=qwen_caption_io_sources["invalid_runtime_rows_count"],
            missing_runtime_rows_count=qwen_caption_io_sources["missing_runtime_rows_count"],
            source_counts=qwen_caption_io_sources["source_counts"],
            accepted_sources=qwen_caption_io_sources["accepted_sources"],
        )
    elif set_and_forget and require_prompt_budget_data and (
        int(qwen_caption_io_sources["valid_runtime_prompt_budget_rows"])
        < int(qwen_caption_io_sources["required_rows"])
    ):
        _add_check(
            checks,
            "qwen_caption_io_source",
            "error",
            "set-and-forget pilot rows must include run-bound qwen_caption_io runtime prompt-budget evidence",
            valid_runtime_prompt_budget_rows=qwen_caption_io_sources["valid_runtime_prompt_budget_rows"],
            required_rows=qwen_caption_io_sources["required_rows"],
            runtime_prompt_budget_rows=qwen_caption_io_sources["runtime_prompt_budget_rows"],
            missing_runtime_rows=qwen_caption_io_sources["missing_runtime_rows"][:20],
            missing_runtime_rows_count=qwen_caption_io_sources["missing_runtime_rows_count"],
            source_counts=qwen_caption_io_sources["source_counts"],
            accepted_sources=qwen_caption_io_sources["accepted_sources"],
        )
    else:
        detail = (
            "pilot runtime prompt-budget evidence is run-bound"
            if int(qwen_caption_io_sources["valid_runtime_prompt_budget_rows"]) > 0
            else "no runtime qwen_caption_io source evidence required for this certification mode"
        )
        _add_check(
            checks,
            "qwen_caption_io_source",
            "ok",
            detail,
            valid_runtime_prompt_budget_rows=qwen_caption_io_sources["valid_runtime_prompt_budget_rows"],
            required_rows=qwen_caption_io_sources["required_rows"],
            runtime_prompt_budget_rows=qwen_caption_io_sources["runtime_prompt_budget_rows"],
            missing_runtime_rows_count=qwen_caption_io_sources["missing_runtime_rows_count"],
            source_counts=qwen_caption_io_sources["source_counts"],
            accepted_sources=qwen_caption_io_sources["accepted_sources"],
        )
    if require_runner_capabilities:
        missing_capabilities = list(runner_capabilities.get("missing_capabilities") or [])
        if missing_capabilities:
            _add_check(
                checks,
                "runner_capabilities",
                "error",
                "pilot manifest or heartbeat lacks current unattended runner recovery and stream-loop telemetry capabilities",
                missing_capabilities=missing_capabilities,
                runner_capabilities=runner_capabilities.get("runner_capabilities") or [],
                capability_sources=runner_capabilities.get("capability_sources") or [],
            )
        else:
            _add_check(
                checks,
                "runner_capabilities",
                "ok",
                "pilot was produced by a runner advertising current unattended recovery and stream-loop telemetry capabilities",
                runner_capabilities=runner_capabilities.get("runner_capabilities") or [],
                capability_sources=runner_capabilities.get("capability_sources") or [],
            )
    else:
        _add_check(
            checks,
            "runner_capabilities",
            "ok",
            "runner-capability certification gate disabled",
            runner_capabilities=runner_capabilities.get("runner_capabilities") or [],
            missing_capabilities=runner_capabilities.get("missing_capabilities") or [],
        )
    max_prompt_tokens = max(0, int(max_prompt_tokens or 0))
    if max_prompt_tokens > 0 and int(prompt_budget["max_prompt_tokens"]) > max_prompt_tokens:
        _add_check(
            checks,
            "max_prompt_tokens",
            "error",
            f"pilot max prompt size {prompt_budget['max_prompt_tokens']} exceeds limit {max_prompt_tokens}",
            max_prompt_tokens=prompt_budget["max_prompt_tokens"],
            limit=max_prompt_tokens,
        )
    elif max_prompt_tokens > 0:
        _add_check(
            checks,
            "max_prompt_tokens",
            "ok",
            f"pilot max prompt size {prompt_budget['max_prompt_tokens']} is within limit {max_prompt_tokens}",
            max_prompt_tokens=prompt_budget["max_prompt_tokens"],
            limit=max_prompt_tokens,
        )
    else:
        _add_check(
            checks,
            "max_prompt_tokens",
            "ok",
            "max prompt-token certification gate disabled",
            max_prompt_tokens=prompt_budget["max_prompt_tokens"],
            limit=max_prompt_tokens,
        )
    adapted_rate_threshold = _safe_float(
        max_prompt_budget_adapted_case_rate,
        DEFAULT_MAX_PROMPT_BUDGET_ADAPTED_CASE_RATE,
    )
    if adapted_rate_threshold >= 0 and float(prompt_budget["adapted_case_rate"]) > adapted_rate_threshold:
        _add_check(
            checks,
            "prompt_budget_adapted_case_rate",
            "error",
            (
                "prompt-budget adaptation rate "
                f"{prompt_budget['adapted_case_rate']:.3f} exceeds limit {adapted_rate_threshold:.3f}"
            ),
            adapted_case_rate=prompt_budget["adapted_case_rate"],
            limit=adapted_rate_threshold,
            adapted_cases=prompt_budget["adapted_cases"],
        )
    else:
        detail = (
            "prompt-budget adaptation gate disabled"
            if adapted_rate_threshold < 0
            else (
                "prompt-budget adaptation rate "
                f"{prompt_budget['adapted_case_rate']:.3f} is within limit {adapted_rate_threshold:.3f}"
            )
        )
        _add_check(
            checks,
            "prompt_budget_adapted_case_rate",
            "ok",
            detail,
            adapted_case_rate=prompt_budget["adapted_case_rate"],
            limit=adapted_rate_threshold,
            adapted_cases=prompt_budget["adapted_cases"],
        )

    if deterministic_recovery_reliability["enabled"]:
        upper_bound_rate = float(deterministic_recovery_reliability["upper_bound_rate"])
        if not generated_rows:
            _add_check(
                checks,
                "deterministic_recovery_confidence",
                "error",
                "set-and-forget deterministic-recovery confidence gate has no generated rows to evaluate",
                **deterministic_recovery_reliability,
            )
        elif upper_bound_rate <= deterministic_recovery_limit:
            _add_check(
                checks,
                "deterministic_recovery_confidence",
                "ok",
                (
                    f"{deterministic_recovery_confidence:.0%} one-sided deterministic-recovery "
                    f"upper bound {upper_bound_rate:.4f} is within limit {deterministic_recovery_limit:.4f}"
                ),
                **deterministic_recovery_reliability,
            )
        else:
            _add_check(
                checks,
                "deterministic_recovery_confidence",
                "error",
                (
                    f"{deterministic_recovery_confidence:.0%} one-sided deterministic-recovery "
                    f"upper bound {upper_bound_rate:.4f} exceeds limit {deterministic_recovery_limit:.4f}"
                ),
                **deterministic_recovery_reliability,
            )
    else:
        _add_check(
            checks,
            "deterministic_recovery_confidence",
            "ok",
            (
                "deterministic-recovery confidence gate disabled"
                if deterministic_recovery_confidence <= 0 or deterministic_recovery_limit < 0
                else "deterministic-recovery confidence gate applies only to set-and-forget certification"
            ),
            **deterministic_recovery_reliability,
        )

    status = max((str(check["status"]) for check in checks), key=_status_rank)
    return {
        "status": status,
        "output_dir": str(output_dir),
        "updated_at": _now_iso(),
        "target_cases": target_cases,
        "pilot_cases": len(latest),
        "generated_pilot_cases": generated_count,
        "attempt_rows": len(rows),
        "timing": timing,
        "generated_case_evidence": generated_case_evidence,
        "prompt_budget": prompt_budget,
        "qwen_caption_io_sources": qwen_caption_io_sources,
        "runner_capabilities": runner_capabilities,
        "deterministic_recovery_reliability": deterministic_recovery_reliability,
        "sample_selection": sample_selection,
        "projection": {
            "mean_projected_hours": mean_projected_hours,
            "duration_safety_factor": duration_safety_factor,
            "safety_projected_hours": safety_projected_hours,
            "p95_all_cases_hours": p95_all_cases_hours,
            "max_duration_hours": max_duration_hours,
            "max_p95_duration_hours": p95_duration_gate,
        },
        "run_settings": {
            "manifest_path": str(manifest_path),
            "pilot_fingerprint": pilot_fingerprint,
            "expected_fingerprint": expected_fingerprint,
            "match_required": bool(expected_fingerprint),
        },
        "audit": audit_report,
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--target-cases", type=int, default=DEFAULT_TARGET_CASES)
    parser.add_argument("--max-duration-hours", type=float, default=DEFAULT_MAX_DURATION_HOURS)
    parser.add_argument(
        "--max-p95-duration-hours",
        type=float,
        default=None,
        help=(
            "Maximum projected duration if all target cases ran at the pilot p95 case time. "
            "Defaults to --max-duration-hours. Use -1 to disable."
        ),
    )
    parser.add_argument("--min-pilot-cases", type=int, default=DEFAULT_MIN_PILOT_CASES)
    parser.add_argument("--duration-safety-factor", type=float, default=DEFAULT_DURATION_SAFETY_FACTOR)
    parser.add_argument("--max-heartbeat-age", type=float, default=600.0)
    parser.add_argument("--max-failed-case-rate", type=float, default=0.0)
    parser.add_argument("--max-quality-failure-rate", type=float, default=0.0)
    parser.add_argument("--max-recovery-event-case-rate", type=float, default=audit.DEFAULT_MAX_RECOVERY_EVENT_CASE_RATE)
    parser.add_argument("--max-loop-recovery-case-rate", type=float, default=None)
    parser.add_argument(
        "--max-loop-guard-case-rate",
        type=float,
        default=None,
        help=(
            "Maximum latest-case stream-loop or loop-trim guard rate before certification fails. "
            "Use -1 to disable. Omit for "
            f"{audit.DEFAULT_MAX_LOOP_GUARD_CASE_RATE:g}, or "
            f"{audit.DEFAULT_SET_AND_FORGET_MAX_LOOP_GUARD_CASE_RATE:g} with --set-and-forget."
        ),
    )
    parser.add_argument("--max-deterministic-recovery-case-rate", type=float, default=None)
    parser.add_argument("--max-failed-attempt-row-rate", type=float, default=0.25)
    parser.add_argument(
        "--max-signal-exit-attempt-row-rate",
        type=float,
        default=None,
        help=(
            "Maximum signal-exit attempt-row rate before certification fails. Use -1 to disable. "
            f"Omit for {audit.DEFAULT_MAX_SIGNAL_EXIT_ATTEMPT_ROW_RATE:g}, or "
            f"{audit.DEFAULT_SET_AND_FORGET_MAX_SIGNAL_EXIT_ATTEMPT_ROW_RATE:g} with --set-and-forget."
        ),
    )
    parser.add_argument("--max-attempt-overrun", type=float, default=60.0)
    parser.add_argument("--min-rate-cases", type=int, default=20)
    parser.add_argument(
        "--no-require-prompt-budget-data",
        action="store_true",
        help="Allow legacy pilot artifacts without per-row prompt-budget telemetry.",
    )
    parser.add_argument(
        "--no-require-runner-capabilities",
        action="store_true",
        help="Allow legacy pilot artifacts without current runner capability markers.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=0,
        help="Maximum observed per-case prompt token estimate allowed in the pilot. 0 disables this gate.",
    )
    parser.add_argument(
        "--max-prompt-budget-adapted-case-rate",
        type=float,
        default=DEFAULT_MAX_PROMPT_BUDGET_ADAPTED_CASE_RATE,
        help="Maximum fraction of pilot cases whose prompt budget required adaptation. Use -1 to disable.",
    )
    parser.add_argument(
        "--deterministic-recovery-confidence",
        type=float,
        default=DEFAULT_DETERMINISTIC_RECOVERY_CONFIDENCE,
        help=(
            "One-sided confidence used to bound deterministic count/layout recovery rate in "
            "--set-and-forget certification. Use 0 to disable this confidence gate for diagnostics."
        ),
    )
    parser.add_argument(
        "--set-and-forget",
        action="store_true",
        help=(
            "Use unattended-run certification defaults. This keeps failed and quality rates strict, "
            "but allows small bounded loop-recovery and recovered signal-exit rates unless explicitly overridden."
        ),
    )
    parser.add_argument(
        "--expected-run-settings-fingerprint",
        default=None,
        help="Require the pilot manifest run_settings fingerprint to match this requested launch fingerprint.",
    )
    parser.add_argument(
        "--write-json",
        type=Path,
        default=None,
        help="Write the certification report to this path. Defaults to <output-dir>/certification.json.",
    )
    parser.add_argument("--no-write-json", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = certify_soak(
        args.output_dir,
        target_cases=args.target_cases,
        max_duration_hours=args.max_duration_hours,
        max_p95_duration_hours=args.max_p95_duration_hours,
        min_pilot_cases=args.min_pilot_cases,
        duration_safety_factor=args.duration_safety_factor,
        max_heartbeat_age_seconds=args.max_heartbeat_age,
        max_failed_case_rate=args.max_failed_case_rate,
        max_quality_failure_rate=args.max_quality_failure_rate,
        max_recovery_event_case_rate=args.max_recovery_event_case_rate,
        max_loop_recovery_case_rate=args.max_loop_recovery_case_rate,
        max_loop_guard_case_rate=args.max_loop_guard_case_rate,
        max_deterministic_recovery_case_rate=args.max_deterministic_recovery_case_rate,
        max_failed_attempt_row_rate=args.max_failed_attempt_row_rate,
        max_signal_exit_attempt_row_rate=args.max_signal_exit_attempt_row_rate,
        max_attempt_overrun_seconds=args.max_attempt_overrun,
        min_rate_cases=args.min_rate_cases,
        set_and_forget=args.set_and_forget,
        expected_run_settings_fingerprint=args.expected_run_settings_fingerprint,
        require_prompt_budget_data=not args.no_require_prompt_budget_data,
        require_runner_capabilities=not args.no_require_runner_capabilities,
        max_prompt_tokens=args.max_prompt_tokens,
        max_prompt_budget_adapted_case_rate=args.max_prompt_budget_adapted_case_rate,
        deterministic_recovery_confidence=args.deterministic_recovery_confidence,
    )
    if not args.no_write_json:
        output_dir = args.output_dir.expanduser().resolve(strict=False)
        write_path = args.write_json.expanduser().resolve(strict=False) if args.write_json else output_dir / "certification.json"
        write_path.parent.mkdir(parents=True, exist_ok=True)
        write_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
