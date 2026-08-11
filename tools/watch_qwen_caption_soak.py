#!/usr/bin/env python3
"""Watch Qwen caption soak artifacts and record health snapshots."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import audit_qwen_caption_soak as audit
from tools import run_qwen_caption_flow_benchmark as runner


TERMINAL_HEARTBEAT_STATUSES = {"completed", "failed"}
DEFAULT_MAX_NO_PROGRESS_SECONDS = 3600.0
LATEST_STATUS_NAME = "watchdog_latest.json"
WATCHDOG_STATE_NAME = "watchdog_state.json"
COMPACT_DEGRADED_RATE_KEYS = (
    "failed_case_rate",
    "quality_failure_rate",
    "recovery_event_case_rate",
    "loop_recovery_case_rate",
    "loop_guard_case_rate",
    "deterministic_recovery_case_rate",
    "failed_attempt_row_rate",
    "signal_exit_attempt_row_rate",
    "failed_cases",
    "quality_failed_cases",
    "recovery_event_cases",
    "loop_recovery_cases",
    "loop_guard_cases",
    "deterministic_recovery_cases",
    "failed_attempt_rows",
    "signal_exit_attempt_rows",
    "pending_failed_attempt_cases",
    "processed_cases",
    "attempt_rows",
    "stream_loop_detected_cases",
    "stream_loop_detected_events",
    "stream_loop_detected_case_rate",
    "loop_trim_cases",
    "loop_trim_events",
    "loop_trim_case_rate",
    "prompt_budget_rows",
    "prompt_budget_coverage_rate",
    "prompt_budget_adapted_cases",
    "prompt_budget_adapted_case_rate",
    "max_prompt_tokens",
)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _read_json_object_status(path: Path) -> tuple[dict[str, Any], bool, str | None]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, False, None
    except Exception as exc:
        return {}, True, f"{type(exc).__name__}: {exc}"
    if not isinstance(raw, dict):
        return {}, True, f"JSON root is {type(raw).__name__}, expected object"
    return raw, True, None


def _latest_watchdog_state_from_log(log_path: Path) -> dict[str, Any]:
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    latest: dict[str, Any] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        state = event.get("watchdog_state")
        if isinstance(state, dict):
            latest = dict(state)
    return latest


def _load_watchdog_state(state_path: Path, log_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    state, state_file_exists, state_error = _read_json_object_status(state_path)
    if state:
        return state, {
            "loaded_state_source": "state_json",
            "state_file_exists": state_file_exists,
            "state_load_error": None,
            "state_recovered_from_log": False,
        }
    if state_error is None and state_file_exists:
        return state, {
            "loaded_state_source": "state_json",
            "state_file_exists": state_file_exists,
            "state_load_error": None,
            "state_recovered_from_log": False,
        }
    recovered_state = _latest_watchdog_state_from_log(log_path)
    if recovered_state:
        return recovered_state, {
            "loaded_state_source": "watchdog_log",
            "state_file_exists": state_file_exists,
            "state_load_error": state_error,
            "state_recovered_from_log": True,
        }
    return {}, {
        "loaded_state_source": "default",
        "state_file_exists": state_file_exists,
        "state_load_error": state_error,
        "state_recovered_from_log": False,
    }


def _optional_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed


def _finite_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return default
    return parsed


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _runner_restart_request_path(output_dir: Path) -> Path:
    return output_dir / runner.RUNNER_RESTART_REQUEST_NAME


def _runner_supports_graceful_restart(heartbeat: Mapping[str, Any]) -> bool:
    capabilities = heartbeat.get("runner_capabilities")
    if not isinstance(capabilities, list):
        return False
    normalized = {str(item).strip() for item in capabilities}
    return runner.RUNNER_CAPABILITY_GRACEFUL_RESTART in normalized


def _write_runner_restart_request(
    output_dir: Path,
    *,
    reason: str,
    check_index: int,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    path = _runner_restart_request_path(output_dir)
    existing = _read_json_object(path)
    if existing:
        requested_epoch = _finite_float(existing.get("requested_epoch"), 0.0)
        return {
            "action": "graceful_restart_pending",
            "status": "ok",
            "path": str(path),
            "requested_epoch": requested_epoch,
            "age_seconds": max(0.0, time.time() - requested_epoch) if requested_epoch else None,
            "detail": "runner restart request already exists",
        }
    payload = {
        "reason": reason,
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "requested_epoch": time.time(),
        "check_index": check_index,
        "audit_status": report.get("status"),
        "failed_cases": report.get("failed_cases"),
        "quality_failed_cases": report.get("quality_failed_cases"),
        "processed_cases": report.get("processed_cases"),
        "expected_cases": report.get("expected_cases"),
        "degraded_rates": _compact_degraded_rates(report.get("degraded_rates")),
    }
    _write_json_atomic(path, payload)
    return {
        "action": "graceful_restart_request",
        "status": "ok",
        "path": str(path),
        "requested_epoch": payload["requested_epoch"],
        "detail": "requested runner restart after the active case completes",
    }


def _compact_check(check: Mapping[str, Any]) -> dict[str, Any]:
    compact = {
        "name": check.get("name"),
        "status": check.get("status"),
        "detail": check.get("detail"),
    }
    for key in (
        "age_seconds",
        "runtime_seconds",
        "attempt_timeout_seconds",
        "grace_seconds",
        "processed_cases",
        "expected_cases",
        "incomplete_cases",
        "seconds_since_progress",
        "max_no_progress_seconds",
        "free_human",
        "min_free_human",
        "min_free_gb",
        "active_violations",
        "caution_rates",
        "violations",
        "progress_source",
    ):
        if key in check:
            compact[key] = check.get(key)
    return {key: value for key, value in compact.items() if value is not None}


def _compact_degraded_rates(rates: Any) -> dict[str, Any]:
    if not isinstance(rates, Mapping):
        return {}
    compact = {
        key: rates.get(key)
        for key in COMPACT_DEGRADED_RATE_KEYS
        if key in rates
    }
    for key in ("thresholds", "signal_exit_names", "active_violations", "violations", "caution_rates"):
        if key in rates:
            compact[key] = rates.get(key)
    return compact


def _compact_runtime_projection(projection: Any) -> dict[str, Any] | None:
    if not isinstance(projection, Mapping):
        return None
    keys = (
        "status",
        "detail",
        "active",
        "processed_cases",
        "expected_cases",
        "remaining_cases",
        "cases_per_hour",
        "projected_duration_hours",
        "remaining_hours",
        "max_projected_duration_hours",
        "run_start_epoch",
        "run_start_source",
        "min_rate_cases",
    )
    return {
        key: projection.get(key)
        for key in keys
        if key in projection and projection.get(key) is not None
    }


def _check_counts(checks: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"ok": 0, "warn": 0, "error": 0}
    for check in checks:
        status = str(check.get("status") or "").strip().lower()
        if status in counts:
            counts[status] += 1
    return counts


def _active_attempt_progress_signal(report: Mapping[str, Any]) -> dict[str, Any]:
    active_attempt = report.get("active_attempt")
    if not isinstance(active_attempt, Mapping):
        return {"source": "case", "signal": ""}
    worker_progress = active_attempt.get("worker_progress")
    if not isinstance(worker_progress, Mapping):
        return {"source": "case", "signal": ""}
    signal_fields = {
        "case_id": active_attempt.get("case_id") or active_attempt.get("case"),
        "attempt": active_attempt.get("attempt"),
        "run_id": worker_progress.get("run_id"),
        "seq": worker_progress.get("seq"),
        "phase": worker_progress.get("phase"),
        "step_id": worker_progress.get("step_id"),
        "generated_tokens": worker_progress.get("generated_tokens"),
        "live_output_chars": worker_progress.get("live_output_chars"),
        "token_preview_chars": worker_progress.get("token_preview_chars"),
        "io_event_count": worker_progress.get("io_event_count"),
        "updated_at": worker_progress.get("updated_at"),
    }
    if not any(value not in (None, "") for value in signal_fields.values()):
        return {"source": "case", "signal": ""}
    return {
        "source": "worker",
        "signal": json.dumps(signal_fields, sort_keys=True, default=str),
        "worker_progress": {
            key: worker_progress.get(key)
            for key in (
                "run_id",
                "seq",
                "phase",
                "step_id",
                "step_label",
                "generated_tokens",
                "max_new_tokens",
                "live_output_chars",
                "token_preview_chars",
                "io_event_count",
                "updated_at",
            )
            if key in worker_progress
        },
    }


def _watchdog_event(
    *,
    output_dir: Path,
    report: Mapping[str, Any],
    check_index: int,
    consecutive_unhealthy: int,
    terminal: bool,
    strict_completion: bool,
    progress_watch: Mapping[str, Any],
    effective_status: str,
    watchdog_state: Mapping[str, Any],
    remediation: Mapping[str, Any] | None = None,
    compact: bool = True,
    checked_epoch: float | None = None,
) -> dict[str, Any]:
    resolved_checked_epoch = float(checked_epoch if checked_epoch is not None else time.time())
    checked_at = datetime.fromtimestamp(resolved_checked_epoch, timezone.utc).isoformat()
    heartbeat = report.get("heartbeat") if isinstance(report.get("heartbeat"), Mapping) else {}
    active_attempt = report.get("active_attempt") if isinstance(report.get("active_attempt"), Mapping) else None
    checks = report.get("checks") if isinstance(report.get("checks"), list) else []
    progress_check = progress_watch.get("check") if isinstance(progress_watch.get("check"), Mapping) else None
    event_checks = [*checks, progress_check] if progress_check else checks
    clean_checks = [check for check in event_checks if isinstance(check, Mapping)]
    event_degraded_rates = (
        _compact_degraded_rates(report.get("degraded_rates"))
        if compact
        else report.get("degraded_rates")
    )
    runtime_projection = report.get("runtime_projection")
    event_runtime_projection = (
        _compact_runtime_projection(runtime_projection)
        if compact
        else dict(runtime_projection) if isinstance(runtime_projection, Mapping) else None
    )
    disk_reserve = report.get("disk_reserve") if isinstance(report.get("disk_reserve"), Mapping) else None
    event = {
        "event": "qwen_caption_soak_watchdog",
        "event_detail": "compact" if compact else "full",
        "checked_at": checked_at,
        "checked_epoch": resolved_checked_epoch,
        "time": resolved_checked_epoch,
        "check_index": check_index,
        "output_dir": str(output_dir),
        "status": effective_status,
        "audit_status": report.get("status"),
        "heartbeat_status": heartbeat.get("status"),
        "heartbeat_phase": heartbeat.get("phase"),
        "processed_cases": report.get("processed_cases"),
        "expected_cases": report.get("expected_cases"),
        "failed_cases": report.get("failed_cases"),
        "quality_failed_cases": report.get("quality_failed_cases"),
        "degraded_rates": event_degraded_rates,
        "runtime_projection": event_runtime_projection,
        "disk_reserve": (
            {
                "status": disk_reserve.get("status"),
                "detail": disk_reserve.get("detail"),
                "free_human": disk_reserve.get("free_human"),
                "min_free_human": disk_reserve.get("min_free_human"),
                "min_free_gb": disk_reserve.get("min_free_gb"),
            }
            if compact and disk_reserve is not None
            else dict(disk_reserve) if disk_reserve is not None else None
        ),
        "active_attempt": dict(active_attempt) if active_attempt is not None else None,
        "incomplete_cases": report.get("incomplete_cases"),
        "resumable": report.get("resumable"),
        "progress_watch": dict(progress_watch),
        "watchdog_state": dict(watchdog_state),
        "consecutive_unhealthy": consecutive_unhealthy,
        "terminal": terminal,
        "strict_completion": strict_completion,
        "check_counts": _check_counts(clean_checks),
        "checks": [_compact_check(check) for check in clean_checks] if compact else clean_checks,
    }
    if remediation is not None:
        event["remediation"] = dict(remediation)
    return event


def _launchd_remediation(
    *,
    launchctl_bin: str,
    domain: str,
    label: str,
    plist_path: Path | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    target = f"{domain}/{label}" if domain else str(label or "")
    launchctl = str(launchctl_bin or "launchctl")
    timeout = max(1.0, float(timeout_seconds or 30.0))
    command = [launchctl, "kickstart", "-k", target]
    payload: dict[str, Any] = {
        "action": "launchd_kickstart",
        "target": target,
        "command": command,
    }
    if not target or "/" not in target:
        payload.update({"status": "error", "detail": "launchd remediation target is missing a domain or label"})
        return payload

    def run_step(step_command: list[str]) -> dict[str, Any]:
        step_payload: dict[str, Any] = {"command": step_command}
        try:
            completed = subprocess.run(
                step_command,
                capture_output=True,
                check=False,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            step_payload.update({
                "status": "error",
                "error_type": type(exc).__name__,
                "stdout": str(exc.stdout or "")[-4000:],
                "stderr": str(exc.stderr or "")[-4000:],
                "timeout_seconds": timeout,
            })
            return step_payload
        step_payload.update({
            "status": "ok" if completed.returncode == 0 else "error",
            "returncode": completed.returncode,
            "stdout": str(completed.stdout or "")[-4000:],
            "stderr": str(completed.stderr or "")[-4000:],
        })
        return step_payload

    kickstart = run_step(command)
    payload.update(kickstart)
    if kickstart.get("status") == "ok":
        return payload

    if plist_path is None:
        return payload

    supervisor_plist = plist_path.expanduser().resolve(strict=False)
    if not supervisor_plist.exists():
        payload.update({
            "status": "error",
            "detail": f"launchd remediation plist is missing: {supervisor_plist}",
            "plist_path": str(supervisor_plist),
        })
        return payload

    bootout_command = [launchctl, "bootout", target]
    bootstrap_command = [launchctl, "bootstrap", domain, str(supervisor_plist)]
    bootout = run_step(bootout_command)
    bootstrap = run_step(bootstrap_command)
    payload.update({
        "action": "launchd_rebootstrap",
        "plist_path": str(supervisor_plist),
        "command": bootstrap_command,
        "kickstart": kickstart,
        "bootout": bootout,
        "bootstrap": bootstrap,
        "status": "ok" if bootstrap.get("status") == "ok" else "error",
        "returncode": bootstrap.get("returncode"),
        "stdout": str(bootstrap.get("stdout") or "")[-4000:],
        "stderr": str(bootstrap.get("stderr") or "")[-4000:],
    })
    if bootstrap.get("status") == "ok" and bootout.get("status") != "ok":
        payload["detail"] = "kickstart failed; bootout failed or target was not loaded; bootstrap succeeded"
    elif bootstrap.get("status") == "ok":
        payload["detail"] = "kickstart failed; bootout and bootstrap succeeded"
    return payload


def watch_soak(
    output_dir: Path,
    *,
    log_jsonl: Path | None = None,
    latest_json: Path | None = None,
    state_json: Path | None = None,
    interval_seconds: float = 60.0,
    max_heartbeat_age_seconds: float = 600.0,
    max_consecutive_unhealthy: int = 3,
    max_checks: int = 0,
    max_failed_case_rate: float = 0.0,
    max_quality_failure_rate: float = 0.0,
    max_recovery_event_case_rate: float = audit.DEFAULT_MAX_RECOVERY_EVENT_CASE_RATE,
    max_loop_recovery_case_rate: float | None = None,
    max_loop_guard_case_rate: float | None = None,
    max_deterministic_recovery_case_rate: float | None = None,
    max_failed_attempt_row_rate: float = 0.25,
    max_signal_exit_attempt_row_rate: float | None = None,
    max_attempt_overrun_seconds: float = 60.0,
    max_projected_duration_hours: float = 0.0,
    min_free_gb: float = 0.0,
    max_no_progress_seconds: float = DEFAULT_MAX_NO_PROGRESS_SECONDS,
    min_rate_cases: int = 20,
    set_and_forget: bool = False,
    require_saved_text_labels: bool = False,
    remediate_launchd_label: str | None = None,
    remediate_launchd_domain: str | None = None,
    remediate_launchd_plist: Path | None = None,
    remediate_launchctl: str = "launchctl",
    max_remediations: int = 0,
    remediation_cooldown_seconds: float = 300.0,
    remediation_timeout_seconds: float = 30.0,
    graceful_restart_timeout_seconds: float = 300.0,
) -> int:
    output_dir = output_dir.expanduser().resolve(strict=False)
    log_path = (log_jsonl or (output_dir / "watchdog.jsonl")).expanduser().resolve(strict=False)
    latest_path = (latest_json or (output_dir / LATEST_STATUS_NAME)).expanduser().resolve(strict=False)
    state_path = (state_json or (output_dir / WATCHDOG_STATE_NAME)).expanduser().resolve(strict=False)
    interval_seconds = max(0.0, float(interval_seconds or 0.0))
    max_heartbeat_age_seconds = max(0.0, float(max_heartbeat_age_seconds or 0.0))
    max_no_progress_seconds = float(max_no_progress_seconds or 0.0)
    max_consecutive_unhealthy = max(0, int(max_consecutive_unhealthy or 0))
    max_checks = max(0, int(max_checks or 0))
    max_remediations = max(0, int(max_remediations or 0))
    remediation_cooldown_seconds = max(0.0, float(remediation_cooldown_seconds or 0.0))
    remediation_timeout_seconds = max(1.0, float(remediation_timeout_seconds or 30.0))
    graceful_restart_timeout_seconds = max(0.0, float(graceful_restart_timeout_seconds or 0.0))
    persisted_state, state_load = _load_watchdog_state(state_path, log_path)
    consecutive_unhealthy = max(0, _optional_int(persisted_state.get("consecutive_unhealthy")) or 0)
    check_index = 0
    remediation_count = max(0, _optional_int(persisted_state.get("remediation_count")) or 0)
    next_remediation_at = max(0.0, _finite_float(persisted_state.get("next_remediation_epoch"), 0.0))
    last_progress_cases = _optional_int(persisted_state.get("last_progress_cases"))
    last_progress_at = _finite_float(persisted_state.get("last_progress_epoch"), time.time())
    last_progress_signal = str(persisted_state.get("last_progress_signal") or "")
    last_progress_source = str(persisted_state.get("last_progress_source") or "case")
    while True:
        check_index += 1
        live_report = audit.audit_soak(
            output_dir,
            max_heartbeat_age_seconds=max_heartbeat_age_seconds,
            allow_running_incomplete=True,
            max_failed_case_rate=max_failed_case_rate,
            max_quality_failure_rate=max_quality_failure_rate,
            max_recovery_event_case_rate=max_recovery_event_case_rate,
            max_loop_recovery_case_rate=max_loop_recovery_case_rate,
            max_loop_guard_case_rate=max_loop_guard_case_rate,
            max_deterministic_recovery_case_rate=max_deterministic_recovery_case_rate,
            max_failed_attempt_row_rate=max_failed_attempt_row_rate,
            max_signal_exit_attempt_row_rate=max_signal_exit_attempt_row_rate,
            max_attempt_overrun_seconds=max_attempt_overrun_seconds,
            max_projected_duration_hours=max_projected_duration_hours,
            min_free_gb=min_free_gb,
            min_rate_cases=min_rate_cases,
            set_and_forget=set_and_forget,
            require_saved_text_labels=require_saved_text_labels,
        )
        heartbeat = live_report.get("heartbeat") if isinstance(live_report.get("heartbeat"), Mapping) else {}
        heartbeat_status = str(heartbeat.get("status") or "").strip().lower()
        terminal = heartbeat_status in TERMINAL_HEARTBEAT_STATUSES
        report = live_report
        strict_completion = False
        if terminal:
            strict_completion = True
            report = audit.audit_soak(
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
                max_projected_duration_hours=max_projected_duration_hours,
                min_free_gb=min_free_gb,
                min_rate_cases=min_rate_cases,
                set_and_forget=set_and_forget,
                require_saved_text_labels=require_saved_text_labels,
            )
        now = time.time()
        try:
            processed_cases = int(report.get("processed_cases") or 0)
        except (TypeError, ValueError, OverflowError):
            processed_cases = 0
        try:
            expected_cases = int(report.get("expected_cases") or 0)
        except (TypeError, ValueError, OverflowError):
            expected_cases = 0
        active_signal = _active_attempt_progress_signal(report)
        progress_source = str(active_signal.get("source") or "case")
        progress_signal = str(active_signal.get("signal") or "")
        progress_changed = False
        if last_progress_cases is None or processed_cases != last_progress_cases:
            last_progress_cases = processed_cases
            last_progress_at = now
            last_progress_signal = progress_signal
            last_progress_source = progress_source if progress_signal else "case"
            progress_changed = True
        elif progress_signal and progress_signal != last_progress_signal:
            last_progress_at = now
            last_progress_signal = progress_signal
            last_progress_source = progress_source
            progress_changed = True
        seconds_since_progress = max(0.0, now - last_progress_at)
        incomplete = expected_cases <= 0 or processed_cases < expected_cases
        progress_stalled = (
            not terminal
            and max_no_progress_seconds > 0
            and incomplete
            and seconds_since_progress > max_no_progress_seconds
        )
        progress_status = "error" if progress_stalled else "ok"
        progress_watch = {
            "processed_cases": processed_cases,
            "expected_cases": expected_cases,
            "last_progress_cases": last_progress_cases,
            "seconds_since_progress": seconds_since_progress,
            "max_no_progress_seconds": max_no_progress_seconds,
            "progress_source": progress_source if progress_signal else "case",
            "last_progress_source": last_progress_source,
            "progress_changed": progress_changed,
            "stalled": progress_stalled,
            "check": {
                "name": "watchdog_case_progress",
                "status": progress_status,
                "detail": (
                    f"no active worker or completed-case progress for {seconds_since_progress:.1f}s"
                    if progress_stalled and progress_signal
                    else f"no completed-case progress for {seconds_since_progress:.1f}s"
                    if progress_stalled
                    else f"active worker progress age {seconds_since_progress:.1f}s"
                    if progress_signal
                    else f"completed-case progress age {seconds_since_progress:.1f}s"
                ),
                "seconds_since_progress": seconds_since_progress,
                "max_no_progress_seconds": max_no_progress_seconds,
                "processed_cases": processed_cases,
                "expected_cases": expected_cases,
                "progress_source": progress_source if progress_signal else "case",
            },
        }
        if active_signal.get("worker_progress"):
            progress_watch["worker_progress"] = active_signal["worker_progress"]
        healthy = str(report.get("status") or "") == "ok" and not progress_stalled
        effective_status = "ok" if healthy else "error"
        consecutive_unhealthy = 0 if healthy else consecutive_unhealthy + 1
        remediation: Mapping[str, Any] | None = None
        unhealthy_threshold_reached = bool(
            max_consecutive_unhealthy and consecutive_unhealthy >= max_consecutive_unhealthy
        )
        restart_request = _read_json_object(_runner_restart_request_path(output_dir))
        restart_request_epoch = _finite_float(restart_request.get("requested_epoch"), 0.0)
        restart_request_age = max(0.0, now - restart_request_epoch) if restart_request_epoch else None
        runner_supports_graceful_restart = _runner_supports_graceful_restart(heartbeat)
        graceful_restart_enabled = bool(
            set_and_forget
            and runner_supports_graceful_restart
            and graceful_restart_timeout_seconds > 0
        )
        graceful_restart_pending = bool(
            restart_request
            and graceful_restart_enabled
            and restart_request_age is not None
            and restart_request_age < graceful_restart_timeout_seconds
        )
        can_request_graceful_restart = (
            not terminal
            and unhealthy_threshold_reached
            and graceful_restart_enabled
            and not graceful_restart_pending
            and not restart_request
        )
        can_remediate = (
            not terminal
            and unhealthy_threshold_reached
            and not graceful_restart_pending
            and not can_request_graceful_restart
            and bool(str(remediate_launchd_label or "").strip())
            and remediation_count < max_remediations
            and now >= next_remediation_at
        )
        if can_request_graceful_restart:
            remediation = _write_runner_restart_request(
                output_dir,
                reason="watchdog_unhealthy_threshold",
                check_index=check_index,
                report=report,
            )
        elif graceful_restart_pending:
            remediation = {
                "action": "graceful_restart_pending",
                "status": "ok",
                "path": str(_runner_restart_request_path(output_dir)),
                "requested_epoch": restart_request_epoch,
                "age_seconds": restart_request_age,
                "timeout_seconds": graceful_restart_timeout_seconds,
                "detail": "waiting for runner to stop between cases",
            }
        elif can_remediate:
            remediation_count += 1
            remediation = _launchd_remediation(
                launchctl_bin=remediate_launchctl,
                domain=str(remediate_launchd_domain or "").strip(),
                label=str(remediate_launchd_label or "").strip(),
                plist_path=remediate_launchd_plist,
                timeout_seconds=remediation_timeout_seconds,
            )
            remediation = {
                **dict(remediation),
                "remediation_index": remediation_count,
                "max_remediations": max_remediations,
            }
        remediation_succeeded = remediation is not None and remediation.get("status") == "ok"
        if remediation_succeeded:
            if remediation.get("action") in {"graceful_restart_request", "graceful_restart_pending"}:
                restart_request_epoch = _finite_float(remediation.get("requested_epoch"), restart_request_epoch)
                restart_request_age = remediation.get("age_seconds")
            consecutive_unhealthy = 0
            last_progress_at = time.time()
            next_remediation_at = last_progress_at + remediation_cooldown_seconds
        watchdog_state = {
            "state_version": 1,
            "state_json": str(state_path),
            **state_load,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "last_progress_cases": last_progress_cases,
            "last_progress_epoch": last_progress_at,
            "last_progress_signal": last_progress_signal,
            "last_progress_source": last_progress_source,
            "remediation_count": remediation_count,
            "max_remediations": max_remediations,
            "next_remediation_epoch": next_remediation_at,
            "consecutive_unhealthy": consecutive_unhealthy,
            "graceful_restart_request_epoch": restart_request_epoch or None,
            "graceful_restart_request_age_seconds": restart_request_age,
            "graceful_restart_timeout_seconds": graceful_restart_timeout_seconds,
            "runner_supports_graceful_restart": runner_supports_graceful_restart,
        }
        event_checked_epoch = time.time()
        history_event = _watchdog_event(
            output_dir=output_dir,
            report=report,
            check_index=check_index,
            consecutive_unhealthy=consecutive_unhealthy,
            terminal=terminal,
            strict_completion=strict_completion,
            progress_watch=progress_watch,
            effective_status=effective_status,
            watchdog_state=watchdog_state,
            remediation=remediation,
            compact=True,
            checked_epoch=event_checked_epoch,
        )
        latest_event = _watchdog_event(
            output_dir=output_dir,
            report=report,
            check_index=check_index,
            consecutive_unhealthy=consecutive_unhealthy,
            terminal=terminal,
            strict_completion=strict_completion,
            progress_watch=progress_watch,
            effective_status=effective_status,
            watchdog_state=watchdog_state,
            remediation=remediation,
            compact=False,
            checked_epoch=event_checked_epoch,
        )
        _append_jsonl(log_path, history_event)
        _write_json_atomic(state_path, watchdog_state)
        _write_json_atomic(latest_path, latest_event)
        print(json.dumps(history_event, sort_keys=True), flush=True)
        if terminal:
            return 0 if healthy else 1
        if max_checks and check_index >= max_checks:
            return 0 if consecutive_unhealthy == 0 else 2
        if remediation_succeeded:
            time.sleep(interval_seconds)
            continue
        if unhealthy_threshold_reached:
            return 2
        time.sleep(interval_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--log-jsonl", type=Path, default=None)
    parser.add_argument(
        "--latest-json",
        type=Path,
        default=None,
        help=(
            f"Write the latest watchdog health snapshot to this JSON file. "
            f"Defaults to {LATEST_STATUS_NAME} in the output directory."
        ),
    )
    parser.add_argument(
        "--state-json",
        type=Path,
        default=None,
        help=(
            "Persist watchdog progress and remediation counters to this JSON file. "
            f"Defaults to {WATCHDOG_STATE_NAME} in the output directory."
        ),
    )
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--max-heartbeat-age", type=float, default=600.0)
    parser.add_argument("--max-failed-case-rate", type=float, default=0.0)
    parser.add_argument("--max-quality-failure-rate", type=float, default=0.0)
    parser.add_argument("--max-recovery-event-case-rate", type=float, default=audit.DEFAULT_MAX_RECOVERY_EVENT_CASE_RATE)
    parser.add_argument("--max-loop-recovery-case-rate", type=float, default=None)
    parser.add_argument("--max-loop-guard-case-rate", type=float, default=None)
    parser.add_argument("--max-deterministic-recovery-case-rate", type=float, default=None)
    parser.add_argument("--max-failed-attempt-row-rate", type=float, default=0.25)
    parser.add_argument("--max-signal-exit-attempt-row-rate", type=float, default=None)
    parser.add_argument("--max-attempt-overrun", type=float, default=60.0)
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
        help="Minimum free disk reserve required during watchdog health checks. 0 disables this gate.",
    )
    parser.add_argument(
        "--max-no-progress",
        type=float,
        default=DEFAULT_MAX_NO_PROGRESS_SECONDS,
        help=(
            "Maximum seconds a nonterminal watch may go without completed-case progress "
            "or active worker progress when worker_progress is available. 0 disables."
        ),
    )
    parser.add_argument("--min-rate-cases", type=int, default=20)
    parser.add_argument("--max-consecutive-unhealthy", type=int, default=3)
    parser.add_argument("--set-and-forget", action="store_true")
    parser.add_argument(
        "--require-saved-text-labels",
        action="store_true",
        help=(
            "Require live and terminal audits to prove generated/resumed successes have "
            "existing non-empty saved dataset text-label files."
        ),
    )
    parser.add_argument(
        "--remediate-launchd-label",
        default=None,
        help="LaunchAgent label to kickstart when a nonterminal run stays unhealthy. Disabled when omitted.",
    )
    parser.add_argument(
        "--remediate-launchd-domain",
        default=None,
        help="launchctl domain for --remediate-launchd-label, for example gui/501.",
    )
    parser.add_argument(
        "--remediate-launchd-plist",
        type=Path,
        default=None,
        help="Supervisor LaunchAgent plist to bootout/bootstrap if kickstart fails or hangs.",
    )
    parser.add_argument(
        "--remediate-launchctl",
        default="launchctl",
        help="launchctl executable used for remediation kickstarts.",
    )
    parser.add_argument(
        "--max-remediations",
        type=int,
        default=0,
        help="Maximum launchd kickstarts before the watchdog exits unhealthy. 0 disables remediation.",
    )
    parser.add_argument(
        "--remediation-cooldown",
        type=float,
        default=300.0,
        help="Minimum seconds between watchdog launchd kickstarts.",
    )
    parser.add_argument(
        "--remediation-timeout",
        type=float,
        default=30.0,
        help="Maximum seconds to wait for each launchctl remediation command.",
    )
    parser.add_argument(
        "--graceful-restart-timeout",
        type=float,
        default=300.0,
        help=(
            "Seconds to wait after writing restart_requested.json before launchd "
            "escalation is allowed. 0 disables graceful restart requests."
        ),
    )
    parser.add_argument(
        "--max-checks",
        type=int,
        default=0,
        help="Stop after this many checks. 0 means watch until terminal or unhealthy threshold.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return watch_soak(
        args.output_dir,
        log_jsonl=args.log_jsonl,
        latest_json=args.latest_json,
        state_json=args.state_json,
        interval_seconds=args.interval,
        max_heartbeat_age_seconds=args.max_heartbeat_age,
        max_consecutive_unhealthy=args.max_consecutive_unhealthy,
        max_checks=args.max_checks,
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
        max_no_progress_seconds=args.max_no_progress,
        min_rate_cases=args.min_rate_cases,
        set_and_forget=args.set_and_forget,
        require_saved_text_labels=args.require_saved_text_labels,
        remediate_launchd_label=args.remediate_launchd_label,
        remediate_launchd_domain=args.remediate_launchd_domain,
        remediate_launchd_plist=args.remediate_launchd_plist,
        remediate_launchctl=args.remediate_launchctl,
        max_remediations=args.max_remediations,
        remediation_cooldown_seconds=args.remediation_cooldown,
        remediation_timeout_seconds=args.remediation_timeout,
        graceful_restart_timeout_seconds=args.graceful_restart_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
