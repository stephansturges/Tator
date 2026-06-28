#!/usr/bin/env python3
"""Audit the live unattended Qwen caption operation envelope."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import plistlib
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import audit_qwen_caption_soak as artifact_audit
from tools import run_qwen_caption_unattended as unattended


STATUS_RANK = {"ok": 0, "warn": 1, "error": 2}
ACCEPTED_LAUNCHD_STATES = {"running", "xpcproxy"}
TRANSIENT_LAUNCHD_STATES = {"spawn scheduled"}
DEFAULT_LAUNCHD_SETTLE_SECONDS = 10.0
DEFAULT_LAUNCHD_SETTLE_INTERVAL_SECONDS = 0.5
SET_AND_FORGET_DEGRADED_RATE_FLAGS = (
    "--max-failed-case-rate",
    "--max-quality-failure-rate",
    "--max-recovery-event-case-rate",
    "--max-loop-recovery-case-rate",
    "--max-loop-guard-case-rate",
    "--max-deterministic-recovery-case-rate",
    "--max-failed-attempt-row-rate",
    "--max-signal-exit-attempt-row-rate",
    "--max-attempt-overrun",
    "--min-rate-cases",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_check(checks: list[dict[str, Any]], name: str, status: str, detail: str, **extra: Any) -> None:
    payload = {"name": name, "status": status, "detail": detail}
    payload.update(extra)
    checks.append(payload)


def _status_rank(status: str) -> int:
    return STATUS_RANK.get(str(status or "").lower(), STATUS_RANK["error"])


def _report_status(checks: Sequence[Mapping[str, Any]]) -> str:
    if not checks:
        return "error"
    return max((str(check.get("status") or "error") for check in checks), key=_status_rank)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def resolve_runbook_path(path: Path) -> Path:
    """Resolve a run root, run artifact dir, or explicit runbook path."""
    candidate = path.expanduser()
    if candidate.is_file():
        return candidate.resolve(strict=False)
    candidates = [
        candidate / unattended.RUNBOOK_NAME,
        candidate / "run" / unattended.RUNBOOK_NAME,
        candidate.parent / unattended.RUNBOOK_NAME,
    ]
    for item in candidates:
        if item.exists():
            return item.resolve(strict=False)
    return (candidate / unattended.RUNBOOK_NAME).resolve(strict=False)


def _path_from_runbook(runbook: Mapping[str, Any], key: str) -> Path | None:
    paths = runbook.get("paths") if isinstance(runbook.get("paths"), Mapping) else {}
    raw = paths.get(key)
    if not raw:
        return None
    return Path(str(raw)).expanduser().resolve(strict=False)


def _readiness_from_runbook(runbook: Mapping[str, Any]) -> dict[str, Any]:
    embedded = runbook.get("readiness") if isinstance(runbook.get("readiness"), Mapping) else {}
    if embedded:
        return dict(embedded)
    readiness_path = _path_from_runbook(runbook, "readiness_json")
    if readiness_path is None or not readiness_path.exists():
        return {}
    return _read_json_object(readiness_path)


def _command_flag_value(command: Sequence[Any], flag: str) -> str | None:
    parts = [str(item) for item in command]
    try:
        index = parts.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(parts):
        return None
    return parts[index + 1]


def _command_flag_float(command: Sequence[Any], flag: str, default: float | None = None) -> float | None:
    raw = _command_flag_value(command, flag)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError, OverflowError):
        return default


def _command_flag_int(command: Sequence[Any], flag: str, default: int) -> int:
    raw = _command_flag_value(command, flag)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError, OverflowError):
        return default


def _runbook_command(runbook: Mapping[str, Any], name: str) -> list[str]:
    commands = runbook.get("commands") if isinstance(runbook.get("commands"), Mapping) else {}
    raw = commands.get(name)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(item) for item in raw]
    return []


def _command_has_flag(command: Sequence[Any], flag: str) -> bool:
    return flag in [str(item) for item in command]


def _command_positive_float(command: Sequence[Any], flag: str) -> float | None:
    value = _command_flag_float(command, flag, None)
    if value is None or value <= 0:
        return None
    return value


def _resolved_path_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return str(Path(text).expanduser().resolve(strict=False))


def _iso_epoch(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _status_snapshot_epoch(payload: Mapping[str, Any]) -> float | None:
    for key in ("checked_epoch", "time"):
        try:
            epoch = float(payload.get(key) or 0.0)
        except (TypeError, ValueError, OverflowError):
            epoch = 0.0
        if epoch > 0:
            return epoch
    return _iso_epoch(payload.get("checked_at"))


def _run_command(
    command_runner: Callable[..., Any],
    command: Sequence[str],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        completed = command_runner(
            [str(item) for item in command],
            capture_output=True,
            check=False,
            text=True,
            timeout=max(1.0, float(timeout_seconds or 1.0)),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "error",
            "command": [str(item) for item in command],
            "error_type": type(exc).__name__,
            "stdout": str(exc.stdout or "")[-4000:],
            "stderr": str(exc.stderr or "")[-4000:],
            "timeout_seconds": timeout_seconds,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "command": [str(item) for item in command],
            "error_type": type(exc).__name__,
            "detail": str(exc),
            "timeout_seconds": timeout_seconds,
        }
    return {
        "status": "ok" if int(getattr(completed, "returncode", 1) or 0) == 0 else "error",
        "command": [str(item) for item in command],
        "returncode": int(getattr(completed, "returncode", 1) or 0),
        "stdout": str(getattr(completed, "stdout", "") or "")[-8000:],
        "stderr": str(getattr(completed, "stderr", "") or "")[-8000:],
        "timeout_seconds": timeout_seconds,
    }


def _launchd_role(runbook: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    launchd = runbook.get("launchd_install") if isinstance(runbook.get("launchd_install"), Mapping) else {}
    roles = launchd.get("roles") if isinstance(launchd.get("roles"), Mapping) else {}
    raw = roles.get(role)
    return raw if isinstance(raw, Mapping) else {}


def _launchd_domain(runbook: Mapping[str, Any]) -> str:
    launchd = runbook.get("launchd_install") if isinstance(runbook.get("launchd_install"), Mapping) else {}
    domain = str(launchd.get("domain") or "").strip()
    if domain:
        return domain
    readiness = _readiness_from_runbook(runbook)
    install = readiness.get("launchd_install_result") if isinstance(readiness.get("launchd_install_result"), Mapping) else {}
    return str(install.get("domain") or "").strip()


def _launchd_timeout(runbook: Mapping[str, Any], default: float) -> float:
    launchd = runbook.get("launchd_install") if isinstance(runbook.get("launchd_install"), Mapping) else {}
    try:
        timeout = float(launchd.get("timeout_seconds") or default)
    except (TypeError, ValueError, OverflowError):
        timeout = default
    return max(1.0, timeout)


def _extract_launchd_state(stdout: str) -> str:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("state = "):
            return stripped.split("=", 1)[1].strip()
    return ""


def _extract_launchd_arguments(stdout: str) -> list[str]:
    lines = stdout.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "arguments = {":
            continue
        args: list[str] = []
        for raw in lines[index + 1:]:
            stripped = raw.strip()
            if stripped == "}":
                return args
            if stripped:
                args.append(stripped)
        return args
    return []


def _plist_program_arguments(plist_path: str) -> tuple[list[str], str]:
    payload, error = _plist_payload(plist_path)
    if error:
        return [], error
    raw_args = payload.get("ProgramArguments")
    if not isinstance(raw_args, Sequence) or isinstance(raw_args, (str, bytes)):
        return [], "plist ProgramArguments is missing or invalid"
    return [str(item) for item in raw_args], ""


def _plist_payload(plist_path: str) -> tuple[dict[str, Any], str]:
    if not plist_path:
        return {}, "plist path is missing"
    path = Path(plist_path).expanduser().resolve(strict=False)
    if not path.exists():
        return {}, f"plist is missing: {path}"
    try:
        payload = plistlib.loads(path.read_bytes())
    except Exception as exc:  # noqa: BLE001
        return {}, f"plist is invalid: {exc}"
    if not isinstance(payload, Mapping):
        return {}, "plist payload is not a dictionary"
    return dict(payload), ""


def _launchd_policy_violation(role: str, role_meta: Mapping[str, Any], output_dir: Path | None) -> dict[str, Any] | None:
    plist_path = str(role_meta.get("plist_path") or "").strip()
    payload, error = _plist_payload(plist_path)
    if error:
        return {"role": role, "plist_path": plist_path, "error": error}
    keep_alive = payload.get("KeepAlive")
    stdout_path = str(payload.get("StandardOutPath") or "").strip()
    stderr_path = str(payload.get("StandardErrorPath") or "").strip()
    working_directory = str(payload.get("WorkingDirectory") or "").strip()
    violations: list[str] = []
    if payload.get("RunAtLoad") is not True:
        violations.append("RunAtLoad must be true")
    if not isinstance(keep_alive, Mapping) or keep_alive.get("SuccessfulExit") is not False:
        violations.append("KeepAlive.SuccessfulExit must be false")
    try:
        throttle_interval = int(payload.get("ThrottleInterval"))
    except (TypeError, ValueError, OverflowError):
        throttle_interval = 0
    if throttle_interval < 10:
        violations.append("ThrottleInterval must be at least 10 seconds")
    if working_directory != str(REPO_ROOT):
        violations.append("WorkingDirectory must be the repository root")
    if not stdout_path:
        violations.append("StandardOutPath is missing")
    if not stderr_path:
        violations.append("StandardErrorPath is missing")
    if output_dir is not None:
        resolved_output = output_dir.expanduser().resolve(strict=False)
        for field_name, raw_path in (("StandardOutPath", stdout_path), ("StandardErrorPath", stderr_path)):
            if not raw_path:
                continue
            resolved_path = Path(raw_path).expanduser().resolve(strict=False)
            try:
                resolved_path.relative_to(resolved_output)
            except ValueError:
                violations.append(f"{field_name} must be under the run output directory")
    if not violations:
        return None
    return {
        "role": role,
        "plist_path": plist_path,
        "violations": violations,
        "run_at_load": payload.get("RunAtLoad"),
        "keep_alive": dict(keep_alive) if isinstance(keep_alive, Mapping) else keep_alive,
        "throttle_interval": payload.get("ThrottleInterval"),
        "working_directory": working_directory,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
    }


def _runbook_program_argument_variants(runbook: Mapping[str, Any], role: str) -> list[list[str]]:
    command = _runbook_command(runbook, role)
    if not command:
        return []
    variants = [command]
    caffeinated = unattended._launchd_program_arguments(command, caffeinate=True)
    if caffeinated != command:
        variants.append(caffeinated)
    return variants


def _argument_difference(actual: Sequence[str], expected: Sequence[str]) -> dict[str, Any]:
    for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
        if actual_item != expected_item:
            return {
                "index": index,
                "actual": actual_item,
                "expected": expected_item,
            }
    if len(actual) != len(expected):
        return {
            "index": min(len(actual), len(expected)),
            "actual": actual[min(len(actual), len(expected)):] or None,
            "expected": expected[min(len(actual), len(expected)):] or None,
        }
    return {}


def _canonical_command_arguments(arguments: Sequence[str]) -> list[tuple[str, str | None]]:
    """Canonicalize CLI args enough to ignore harmless flag-order differences."""
    canonical: list[tuple[str, str | None]] = []
    options_started = False
    index = 0
    while index < len(arguments):
        item = str(arguments[index])
        if item.startswith("--"):
            options_started = True
            value: str | None = None
            if index + 1 < len(arguments) and not str(arguments[index + 1]).startswith("--"):
                value = str(arguments[index + 1])
                index += 1
            canonical.append((item, value))
        elif options_started:
            canonical.append(("__positional_after_options__", item))
        else:
            canonical.append((f"__prefix_{index}__", item))
        index += 1
    prefix = [item for item in canonical if item[0].startswith("__prefix_")]
    rest = sorted(item for item in canonical if not item[0].startswith("__prefix_"))
    return [*prefix, *rest]


def _arguments_equivalent(actual: Sequence[str], expected: Sequence[str]) -> bool:
    return (
        [str(item) for item in actual] == [str(item) for item in expected]
        or _canonical_command_arguments(actual) == _canonical_command_arguments(expected)
    )


def _launchd_argument_check(
    *,
    checks: list[dict[str, Any]],
    runbook: Mapping[str, Any],
    role: str,
    stdout: str,
    plist_path: str,
) -> None:
    loaded_args = _extract_launchd_arguments(stdout)
    if not loaded_args:
        _add_check(
            checks,
            f"{role}_launchd_arguments",
            "error",
            f"{role} LaunchAgent loaded arguments could not be parsed",
            plist_path=plist_path,
        )
        return

    plist_args, plist_error = _plist_program_arguments(plist_path)
    if plist_error:
        _add_check(
            checks,
            f"{role}_launchd_arguments",
            "error",
            f"{role} LaunchAgent plist arguments could not be inspected: {plist_error}",
            loaded_argument_count=len(loaded_args),
            plist_path=plist_path,
        )
        return

    if loaded_args != plist_args:
        _add_check(
            checks,
            f"{role}_launchd_arguments",
            "error",
            f"{role} LaunchAgent loaded arguments do not match the on-disk plist",
            loaded_argument_count=len(loaded_args),
            plist_argument_count=len(plist_args),
            first_difference=_argument_difference(loaded_args, plist_args),
            plist_path=plist_path,
        )
        return

    variants = _runbook_program_argument_variants(runbook, role)
    if not variants:
        _add_check(
            checks,
            f"{role}_launchd_arguments",
            "error",
            f"{role} runbook command is missing",
            loaded_argument_count=len(loaded_args),
            plist_path=plist_path,
        )
        return
    if not any(_arguments_equivalent(plist_args, variant) for variant in variants):
        closest = min(variants, key=lambda item: abs(len(item) - len(plist_args)))
        _add_check(
            checks,
            f"{role}_launchd_arguments",
            "error",
            f"{role} LaunchAgent plist arguments do not match the runbook command",
            plist_argument_count=len(plist_args),
            runbook_argument_counts=[len(variant) for variant in variants],
            first_difference=_argument_difference(plist_args, closest),
            plist_path=plist_path,
        )
        return

    _add_check(
        checks,
        f"{role}_launchd_arguments",
        "ok",
        f"{role} LaunchAgent loaded arguments match the plist and runbook",
        argument_count=len(loaded_args),
        plist_path=plist_path,
    )


def _launchd_print_check(
    *,
    checks: list[dict[str, Any]],
    runbook: Mapping[str, Any],
    role: str,
    command_runner: Callable[..., Any],
    launchctl_bin: str,
    timeout_seconds: float,
    settle_seconds: float,
    settle_interval_seconds: float,
    require_caffeinate: bool = False,
) -> None:
    domain = _launchd_domain(runbook)
    role_meta = _launchd_role(runbook, role)
    label = str(role_meta.get("label") or "").strip()
    plist_path = str(role_meta.get("plist_path") or "").strip()
    if not domain or not label:
        _add_check(
            checks,
            f"{role}_launchd",
            "error",
            f"{role} LaunchAgent domain or label is missing from the runbook",
            domain=domain,
            label=label,
        )
        return

    target = f"{domain}/{label}"
    result = _run_command(command_runner, [launchctl_bin, "print", target], timeout_seconds=timeout_seconds)
    state = _extract_launchd_state(str(result.get("stdout") or ""))
    state_history = [state or "unknown"]
    settle_attempts = 1
    first_transient_state = state if state.lower() in TRANSIENT_LAUNCHD_STATES else ""
    if result.get("status") == "ok" and first_transient_state:
        deadline = time.monotonic() + max(0.0, float(settle_seconds or 0.0))
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            interval = min(max(0.0, float(settle_interval_seconds or 0.0)), remaining)
            if interval > 0:
                time.sleep(interval)
            result = _run_command(command_runner, [launchctl_bin, "print", target], timeout_seconds=timeout_seconds)
            settle_attempts += 1
            state = _extract_launchd_state(str(result.get("stdout") or ""))
            state_history.append(state or "unknown")
            if result.get("status") != "ok" or state.lower() not in TRANSIENT_LAUNCHD_STATES:
                break
            if interval <= 0:
                break
    output = str(result.get("stdout") or "")
    expected_path_ok = bool(not plist_path or plist_path in output)
    if result.get("status") != "ok":
        _add_check(
            checks,
            f"{role}_launchd",
            "error",
            f"{role} LaunchAgent is not loaded",
            target=target,
            command=result,
        )
        return
    state_key = state.lower()
    if state and state_key not in ACCEPTED_LAUNCHD_STATES:
        _add_check(
            checks,
            f"{role}_launchd",
            "error",
            (
                f"{role} LaunchAgent state is {state} after a {settle_seconds:g}s settle window"
                if state_key in TRANSIENT_LAUNCHD_STATES
                else f"{role} LaunchAgent state is {state}"
            ),
            target=target,
            state=state,
            state_history=state_history,
            settle_attempts=settle_attempts,
            settle_seconds=settle_seconds,
            command=result,
        )
        return
    if not expected_path_ok:
        _add_check(
            checks,
            f"{role}_launchd",
            "error",
            f"{role} LaunchAgent is loaded from an unexpected plist path",
            target=target,
            expected_plist_path=plist_path,
            state=state,
            command=result,
        )
        return
    if require_caffeinate and unattended.CAFFEINATE_BIN not in output:
        _add_check(
            checks,
            f"{role}_launchd",
            "error",
            f"{role} LaunchAgent is not caffeinate-wrapped",
            target=target,
            expected_program=unattended.CAFFEINATE_BIN,
            state=state,
            command=result,
        )
        return
    detail = f"{role} LaunchAgent is loaded"
    extra: dict[str, Any] = {}
    if first_transient_state:
        detail = f"{role} LaunchAgent is loaded after transient {first_transient_state} state settled"
        extra.update(
            transient_state=first_transient_state,
            state_history=state_history,
            settle_attempts=settle_attempts,
            settle_seconds=settle_seconds,
        )
    _add_check(
        checks,
        f"{role}_launchd",
        "ok",
        detail,
        target=target,
        state=state or "unknown",
        plist_path=plist_path,
        **extra,
    )
    _launchd_argument_check(
        checks=checks,
        runbook=runbook,
        role=role,
        stdout=output,
        plist_path=plist_path,
    )


def _audit_threshold_args(runbook: Mapping[str, Any]) -> dict[str, Any]:
    command = _runbook_command(runbook, "live_status") or _runbook_command(runbook, "final_audit")
    set_and_forget = bool((runbook.get("set_and_forget_gate") or {}).get("tenk_mode")) if isinstance(runbook.get("set_and_forget_gate"), Mapping) else False
    return {
        "max_heartbeat_age_seconds": _command_flag_float(command, "--max-heartbeat-age", 600.0) or 600.0,
        "max_failed_case_rate": _command_flag_float(command, "--max-failed-case-rate", 0.0) or 0.0,
        "max_quality_failure_rate": _command_flag_float(command, "--max-quality-failure-rate", 0.0) or 0.0,
        "max_recovery_event_case_rate": _command_flag_float(
            command,
            "--max-recovery-event-case-rate",
            artifact_audit.DEFAULT_MAX_RECOVERY_EVENT_CASE_RATE,
        )
        or artifact_audit.DEFAULT_MAX_RECOVERY_EVENT_CASE_RATE,
        "max_loop_recovery_case_rate": _command_flag_float(command, "--max-loop-recovery-case-rate", None),
        "max_loop_guard_case_rate": _command_flag_float(command, "--max-loop-guard-case-rate", None),
        "max_deterministic_recovery_case_rate": _command_flag_float(
            command,
            "--max-deterministic-recovery-case-rate",
            None,
        ),
        "max_failed_attempt_row_rate": _command_flag_float(command, "--max-failed-attempt-row-rate", 0.25) or 0.25,
        "max_signal_exit_attempt_row_rate": _command_flag_float(command, "--max-signal-exit-attempt-row-rate", None),
        "max_attempt_overrun_seconds": _command_flag_float(command, "--max-attempt-overrun", 60.0) or 60.0,
        "max_projected_duration_hours": _command_flag_float(command, "--max-projected-duration-hours", 0.0) or 0.0,
        "min_free_gb": _command_flag_float(command, "--min-free-gb", 0.0) or 0.0,
        "min_rate_cases": _command_flag_int(command, "--min-rate-cases", 20),
        "set_and_forget": set_and_forget,
    }


def _audit_artifacts(
    *,
    checks: list[dict[str, Any]],
    runbook: Mapping[str, Any],
    allow_running_incomplete: bool,
) -> dict[str, Any]:
    output_dir_raw = str(runbook.get("output_dir") or "").strip()
    output_dir = Path(output_dir_raw).expanduser().resolve(strict=False) if output_dir_raw else None
    if output_dir is None:
        _add_check(checks, "artifact_audit", "error", "runbook output_dir is missing")
        return {}
    thresholds = _audit_threshold_args(runbook)
    report = artifact_audit.audit_soak(
        output_dir,
        allow_running_incomplete=allow_running_incomplete,
        **thresholds,
    )
    status = str(report.get("status") or "error").lower()
    _add_check(
        checks,
        "artifact_audit",
        "ok" if status == "ok" else "error",
        f"caption artifact audit status is {status}",
        output_dir=str(output_dir),
        processed_cases=report.get("processed_cases"),
        expected_cases=report.get("expected_cases"),
        failed_cases=report.get("failed_cases"),
        quality_failed_cases=report.get("quality_failed_cases"),
    )
    return report


def _audit_watchdog_files(
    *,
    checks: list[dict[str, Any]],
    runbook: Mapping[str, Any],
    max_status_age_seconds: float,
    now_epoch: float,
    strict_set_and_forget: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    latest_path = _path_from_runbook(runbook, "watchdog_latest_json")
    state_path = _path_from_runbook(runbook, "watchdog_state_json")
    output_dir_expected = _resolved_path_text(runbook.get("output_dir"))
    state_path_expected = str(state_path) if state_path is not None else ""
    latest: dict[str, Any] = {}
    state: dict[str, Any] = {}
    if latest_path is None or not latest_path.exists():
        _add_check(
            checks,
            "watchdog_latest",
            "error",
            "watchdog latest-status JSON is missing",
            path=str(latest_path) if latest_path else None,
        )
    else:
        latest = _read_json_object(latest_path)
        checked_epoch = _status_snapshot_epoch(latest)
        age = None if checked_epoch is None else max(0.0, now_epoch - checked_epoch)
        status = str(latest.get("status") or latest.get("audit_status") or "error").lower()
        if not latest:
            _add_check(checks, "watchdog_latest", "error", "watchdog latest-status JSON is invalid", path=str(latest_path))
        elif age is None:
            _add_check(checks, "watchdog_latest", "error", "watchdog latest-status timestamp is missing or invalid", path=str(latest_path))
        elif age > max_status_age_seconds:
            _add_check(
                checks,
                "watchdog_latest",
                "error",
                f"watchdog latest-status is stale ({age:.1f}s old)",
                path=str(latest_path),
                age_seconds=age,
                max_age_seconds=max_status_age_seconds,
                watchdog_status=status,
            )
        elif status != "ok":
            _add_check(
                checks,
                "watchdog_latest",
                "error",
                f"watchdog latest-status is {status}",
                path=str(latest_path),
                age_seconds=age,
                watchdog_status=status,
            )
        else:
            _add_check(
                checks,
                "watchdog_latest",
                "ok",
                "watchdog latest-status is fresh and healthy",
                path=str(latest_path),
                age_seconds=age,
                processed_cases=latest.get("processed_cases"),
                active_attempt=latest.get("active_attempt"),
            )
        if latest and strict_set_and_forget:
            latest_output_dir = _resolved_path_text(latest.get("output_dir"))
            if not latest_output_dir:
                _add_check(
                    checks,
                    "watchdog_latest_run_binding",
                    "error",
                    "strict set-and-forget requires watchdog latest-status to record its output directory",
                    path=str(latest_path),
                    expected_output_dir=output_dir_expected,
                )
            elif output_dir_expected and latest_output_dir != output_dir_expected:
                _add_check(
                    checks,
                    "watchdog_latest_run_binding",
                    "error",
                    "watchdog latest-status belongs to a different output directory",
                    path=str(latest_path),
                    expected_output_dir=output_dir_expected,
                    actual_output_dir=latest_output_dir,
                )
            else:
                _add_check(
                    checks,
                    "watchdog_latest_run_binding",
                    "ok",
                    "watchdog latest-status is bound to the runbook output directory",
                    path=str(latest_path),
                    output_dir=latest_output_dir,
                )
    if state_path is None or not state_path.exists():
        _add_check(
            checks,
            "watchdog_state",
            "error",
            "watchdog restart-stable state JSON is missing",
            path=str(state_path) if state_path else None,
        )
    else:
        state = _read_json_object(state_path)
        remediation_count = state.get("remediation_count")
        max_remediations = state.get("max_remediations")
        unhealthy = state.get("consecutive_unhealthy")
        if not state:
            _add_check(checks, "watchdog_state", "error", "watchdog restart-stable state JSON is invalid", path=str(state_path))
        elif (
            isinstance(remediation_count, int)
            and isinstance(max_remediations, int)
            and max_remediations >= 0
            and remediation_count > max_remediations
        ):
            _add_check(
                checks,
                "watchdog_state",
                "error",
                "watchdog remediation budget is exhausted",
                path=str(state_path),
                remediation_count=remediation_count,
                max_remediations=max_remediations,
            )
        elif isinstance(unhealthy, int) and unhealthy > 0:
            _add_check(
                checks,
                "watchdog_state",
                "warn",
                "watchdog has consecutive unhealthy checks recorded",
                path=str(state_path),
                consecutive_unhealthy=unhealthy,
                remediation_count=remediation_count,
                max_remediations=max_remediations,
            )
        else:
            _add_check(
                checks,
                "watchdog_state",
                "ok",
                "watchdog restart-stable state is present",
                path=str(state_path),
                last_progress_cases=state.get("last_progress_cases"),
                remediation_count=remediation_count,
                max_remediations=max_remediations,
            )
        if state and strict_set_and_forget:
            state_json = _resolved_path_text(state.get("state_json"))
            latest_watchdog_state = latest.get("watchdog_state") if isinstance(latest.get("watchdog_state"), Mapping) else {}
            latest_state_json = _resolved_path_text(latest_watchdog_state.get("state_json"))
            mismatches: list[str] = []
            if not state_json:
                mismatches.append("state file does not record state_json")
            elif state_path_expected and state_json != state_path_expected:
                mismatches.append("state file state_json does not match the runbook path")
            if not latest_state_json:
                mismatches.append("latest status does not embed watchdog_state.state_json")
            elif state_path_expected and latest_state_json != state_path_expected:
                mismatches.append("latest status watchdog_state.state_json does not match the runbook path")
            if mismatches:
                _add_check(
                    checks,
                    "watchdog_state_run_binding",
                    "error",
                    "strict set-and-forget requires watchdog state to be bound to the runbook state path",
                    path=str(state_path),
                    expected_state_json=state_path_expected,
                    state_json=state_json or None,
                    latest_state_json=latest_state_json or None,
                    mismatches=mismatches,
                )
            else:
                _add_check(
                    checks,
                    "watchdog_state_run_binding",
                    "ok",
                    "watchdog state is bound to the runbook state path",
                    path=str(state_path),
                    state_json=state_json,
                )
    return latest, state


def _audit_readiness_file(checks: list[dict[str, Any]], runbook: Mapping[str, Any]) -> None:
    readiness_path = _path_from_runbook(runbook, "readiness_json")
    if readiness_path is None or not readiness_path.exists():
        _add_check(checks, "readiness_artifact", "error", "readiness JSON is missing", path=str(readiness_path) if readiness_path else None)
        return
    readiness = _read_json_object(readiness_path)
    status = str(readiness.get("status") or "error").lower()
    ready = bool(readiness.get("ready_for_10k_set_and_forget"))
    if status != "ok" or not ready:
        _add_check(
            checks,
            "readiness_artifact",
            "error",
            "readiness JSON does not prove 10k set-and-forget readiness",
            path=str(readiness_path),
            readiness_status=status,
            ready_for_10k_set_and_forget=ready,
        )
    else:
        _add_check(
            checks,
            "readiness_artifact",
            "ok",
            "readiness JSON proves 10k set-and-forget readiness",
            path=str(readiness_path),
        )


def _audit_pmset(
    *,
    checks: list[dict[str, Any]],
    runbook: Mapping[str, Any],
    command_runner: Callable[..., Any],
    pmset_bin: str,
    timeout_seconds: float,
) -> None:
    power = runbook.get("launchd_power_assertion") if isinstance(runbook.get("launchd_power_assertion"), Mapping) else {}
    enabled = bool(power.get("enabled"))
    if not enabled:
        _add_check(checks, "sleep_assertion", "warn", "runbook does not request launchd caffeinate sleep prevention")
        return
    result = _run_command(command_runner, [pmset_bin, "-g", "assertions"], timeout_seconds=timeout_seconds)
    output = f"{result.get('stdout') or ''}\n{result.get('stderr') or ''}"
    if result.get("status") != "ok":
        _add_check(checks, "sleep_assertion", "error", "could not inspect macOS sleep assertions", command=result)
    elif "caffeinate" not in output or "PreventSystemSleep" not in output:
        _add_check(
            checks,
            "sleep_assertion",
            "error",
            "macOS sleep assertions do not show an active caffeinate PreventSystemSleep assertion",
            command=result,
        )
    else:
        _add_check(
            checks,
            "sleep_assertion",
            "ok",
            "macOS sleep assertions show active caffeinate protection",
        )


def _audit_strict_set_and_forget_contract(
    *,
    checks: list[dict[str, Any]],
    runbook: Mapping[str, Any],
) -> None:
    readiness = _readiness_from_runbook(runbook)
    output_dir_raw = str(runbook.get("output_dir") or "").strip()
    output_dir = Path(output_dir_raw).expanduser().resolve(strict=False) if output_dir_raw else None
    gate = runbook.get("set_and_forget_gate") if isinstance(runbook.get("set_and_forget_gate"), Mapping) else {}
    readiness_ok = str(readiness.get("status") or "").lower() == "ok"
    ready = bool(readiness.get("ready_for_10k_set_and_forget"))
    if readiness_ok and ready:
        _add_check(
            checks,
            "set_and_forget_readiness",
            "ok",
            "runbook readiness report approves set-and-forget handoff",
            tenk_mode=bool(gate.get("tenk_mode")),
            require_readiness_ok=bool(gate.get("require_readiness_ok")),
        )
    else:
        _add_check(
            checks,
            "set_and_forget_readiness",
            "error",
            "strict set-and-forget audit requires an ok readiness report",
            readiness_status=readiness.get("status"),
            ready_for_10k_set_and_forget=ready,
            tenk_mode=bool(gate.get("tenk_mode")),
            require_readiness_ok=bool(gate.get("require_readiness_ok")),
        )

    readiness_summary = readiness.get("summary") if isinstance(readiness.get("summary"), Mapping) else {}
    live_adoption = (
        runbook.get("live_adoption_certification")
        if isinstance(runbook.get("live_adoption_certification"), Mapping)
        else {}
    )
    live_adoption_requested = bool(readiness_summary.get("live_adoption_requested")) or bool(live_adoption)
    if live_adoption_requested:
        adoption_checks = (
            live_adoption.get("checks")
            if isinstance(live_adoption.get("checks"), Sequence)
            and not isinstance(live_adoption.get("checks"), (str, bytes))
            else []
        )
        restart_capability = next(
            (
                check
                for check in adoption_checks
                if isinstance(check, Mapping)
                and str(check.get("name") or "") == "live_runner_restart_capability"
            ),
            {},
        )
        adoption_status = str(live_adoption.get("status") or "missing").lower()
        capability_status = str(restart_capability.get("status") or "missing").lower()
        if adoption_status == "ok" and capability_status == "ok":
            _add_check(
                checks,
                "set_and_forget_live_adoption_capability",
                "ok",
                "live adoption certification proves the active runner supports cooperative restart",
                adoption_status=adoption_status,
                runner_capabilities=restart_capability.get("runner_capabilities") or [],
            )
        else:
            _add_check(
                checks,
                "set_and_forget_live_adoption_capability",
                "error",
                "strict live-run adoption requires active-runner cooperative restart capability evidence",
                adoption_status=adoption_status,
                restart_capability_status=capability_status,
                runner_capabilities=restart_capability.get("runner_capabilities") or [],
            )

    required_commands = ("supervisor", "watchdog", "live_status", "final_audit")
    commands = runbook.get("commands") if isinstance(runbook.get("commands"), Mapping) else {}
    missing_commands = [
        name
        for name in required_commands
        if not isinstance(commands.get(name), Sequence) or isinstance(commands.get(name), (str, bytes))
    ]
    operational_audit = _runbook_command(runbook, "operational_audit")
    missing_operational_flags = [
        flag
        for flag in ("--allow-running-incomplete", "--compact", "--write-json", "--strict-set-and-forget")
        if not _command_has_flag(operational_audit, flag)
    ]
    operational_audit_recorded = not missing_operational_flags
    if missing_commands or missing_operational_flags:
        _add_check(
            checks,
            "set_and_forget_commands",
            "error",
            (
                "strict set-and-forget audit requires recorded recovery, watchdog, status, "
                "final-audit, and strict operation-audit commands"
            ),
            missing_commands=missing_commands,
            operational_audit_recorded=operational_audit_recorded,
            missing_operational_audit_flags=missing_operational_flags,
        )
    else:
        _add_check(
            checks,
            "set_and_forget_commands",
            "ok",
            "runbook records recovery, watchdog, status, and final-audit commands",
            operational_audit_recorded=operational_audit_recorded,
            missing_operational_audit_flags=missing_operational_flags,
        )

    supervisor_command = _runbook_command(runbook, "supervisor")
    watchdog_command = _runbook_command(runbook, "watchdog")
    live_status_command = _runbook_command(runbook, "live_status")
    final_audit_command = _runbook_command(runbook, "final_audit")
    saved_text_label_gate_required = any(
        _command_has_flag(supervisor_command, flag)
        for flag in ("--save-dataset-text-labels", "--require-saved-text-labels")
    )
    saved_text_label_gate_commands = {
        "watchdog": _command_has_flag(watchdog_command, "--require-saved-text-labels"),
        "live_status": _command_has_flag(live_status_command, "--require-saved-text-labels"),
        "final_audit": _command_has_flag(final_audit_command, "--require-saved-text-labels"),
    }
    missing_saved_text_label_gates = [
        name
        for name, present in saved_text_label_gate_commands.items()
        if saved_text_label_gate_required and not present
    ]
    if missing_saved_text_label_gates:
        _add_check(
            checks,
            "set_and_forget_saved_text_label_gates",
            "error",
            "strict set-and-forget audit requires saved-label audit gates when the supervisor saves dataset text labels",
            required=saved_text_label_gate_required,
            command_gates=saved_text_label_gate_commands,
            missing_commands=missing_saved_text_label_gates,
        )
    else:
        _add_check(
            checks,
            "set_and_forget_saved_text_label_gates",
            "ok",
            (
                "saved-label audit gates are present for dataset text-label saving"
                if saved_text_label_gate_required
                else "dataset text-label saving is not requested by the supervisor command"
            ),
            required=saved_text_label_gate_required,
            command_gates=saved_text_label_gate_commands,
            missing_commands=[],
        )

    degraded_rate_gate_commands = {
        "watchdog": watchdog_command,
        "live_status": live_status_command,
        "final_audit": final_audit_command,
    }
    pilot_certification_command = _runbook_command(runbook, "pilot_certification")
    if pilot_certification_command:
        degraded_rate_gate_commands["pilot_certification"] = pilot_certification_command
    degraded_rate_gate_command_names = [
        name.replace("_", "-") for name in degraded_rate_gate_commands
    ]
    degraded_rate_gate_command_detail = ", ".join(degraded_rate_gate_command_names)
    missing_degraded_rate_flags = {
        name: [
            flag
            for flag in SET_AND_FORGET_DEGRADED_RATE_FLAGS
            if _command_flag_value(command, flag) is None
        ]
        for name, command in degraded_rate_gate_commands.items()
    }
    missing_degraded_rate_flags = {
        name: flags
        for name, flags in missing_degraded_rate_flags.items()
        if flags
    }
    if missing_degraded_rate_flags:
        _add_check(
            checks,
            "set_and_forget_degraded_rate_gates",
            "error",
            f"strict set-and-forget audit requires degraded-rate gates on {degraded_rate_gate_command_detail} commands",
            missing_flags=missing_degraded_rate_flags,
            required_flags=list(SET_AND_FORGET_DEGRADED_RATE_FLAGS),
        )
    else:
        _add_check(
            checks,
            "set_and_forget_degraded_rate_gates",
            "ok",
            f"{degraded_rate_gate_command_detail} commands carry degraded-rate gates",
            required_flags=list(SET_AND_FORGET_DEGRADED_RATE_FLAGS),
            command_names=list(degraded_rate_gate_commands),
        )
    command_gate_values = {
        "supervisor_min_free_gb": _command_positive_float(supervisor_command, "--min-free-gb"),
        "supervisor_max_projected_duration_hours": _command_positive_float(
            supervisor_command,
            "--max-projected-duration-hours",
        ),
        "watchdog_min_free_gb": _command_positive_float(watchdog_command, "--min-free-gb"),
        "live_status_min_free_gb": _command_positive_float(live_status_command, "--min-free-gb"),
        "final_audit_min_free_gb": _command_positive_float(final_audit_command, "--min-free-gb"),
        "watchdog_max_projected_duration_hours": _command_positive_float(
            watchdog_command,
            "--max-projected-duration-hours",
        ),
        "final_audit_max_projected_duration_hours": _command_positive_float(
            final_audit_command,
            "--max-projected-duration-hours",
        ),
    }
    missing_or_disabled_gates = [
        name
        for name, value in command_gate_values.items()
        if value is None
    ]
    if missing_or_disabled_gates:
        _add_check(
            checks,
            "set_and_forget_live_gates",
            "error",
            "strict set-and-forget audit requires supervisor, watchdog, live-status, and final-audit live gates",
            missing_or_disabled=missing_or_disabled_gates,
            gate_values=command_gate_values,
        )
    else:
        _add_check(
            checks,
            "set_and_forget_live_gates",
            "ok",
            "supervisor, watchdog, live-status, and final-audit commands carry live durability gates",
            gate_values=command_gate_values,
        )

    launchd_install = runbook.get("launchd_install") if isinstance(runbook.get("launchd_install"), Mapping) else {}
    install_result = (
        runbook.get("launchd_install_result")
        if isinstance(runbook.get("launchd_install_result"), Mapping)
        else {}
    )
    roles = launchd_install.get("roles") if isinstance(launchd_install.get("roles"), Mapping) else {}
    missing_launchd_roles = [
        role
        for role in ("supervisor", "watchdog")
        if not isinstance(roles.get(role), Mapping)
        or not str((roles.get(role) or {}).get("label") or "").strip()
        or not str((roles.get(role) or {}).get("plist_path") or "").strip()
    ]
    install_status = str(install_result.get("status") or "").lower()
    if not bool(launchd_install.get("requested")) or missing_launchd_roles or install_status != "ok":
        _add_check(
            checks,
            "set_and_forget_launchd_install",
            "error",
            "strict set-and-forget audit requires installed supervisor and watchdog LaunchAgents",
            launchd_requested=bool(launchd_install.get("requested")),
            missing_roles=missing_launchd_roles,
            launchd_install_status=install_status or None,
        )
    else:
        _add_check(
            checks,
            "set_and_forget_launchd_install",
            "ok",
            "runbook records installed supervisor and watchdog LaunchAgents",
            domain=str(launchd_install.get("domain") or ""),
        )

    non_persistent_roles = [
        {
            "role": role,
            "plist_path": str((roles.get(role) or {}).get("plist_path") or ""),
        }
        for role in ("supervisor", "watchdog")
        if not unattended._is_launchagents_plist_path((roles.get(role) or {}).get("plist_path"))
    ]
    if non_persistent_roles:
        _add_check(
            checks,
            "set_and_forget_launchd_persistence",
            "error",
            "strict set-and-forget audit requires LaunchAgent plists under a Library/LaunchAgents directory",
            non_persistent_roles=non_persistent_roles,
        )
    else:
        _add_check(
            checks,
            "set_and_forget_launchd_persistence",
            "ok",
            "supervisor and watchdog plist paths are under a Library/LaunchAgents directory",
        )

    policy_violations = [
        violation
        for role in ("supervisor", "watchdog")
        if (violation := _launchd_policy_violation(role, roles.get(role) or {}, output_dir)) is not None
    ]
    if policy_violations:
        _add_check(
            checks,
            "set_and_forget_launchd_policy",
            "error",
            "strict set-and-forget audit requires restart-safe LaunchAgent policies",
            policy_violations=policy_violations,
        )
    else:
        _add_check(
            checks,
            "set_and_forget_launchd_policy",
            "ok",
            "supervisor and watchdog LaunchAgent plists have restart-safe policies",
        )

    remediation_flags = (
        "--remediate-launchd-label",
        "--remediate-launchd-domain",
        "--remediate-launchd-plist",
        "--max-remediations",
    )
    missing_remediation_flags = [
        flag
        for flag in remediation_flags
        if not _command_has_flag(watchdog_command, flag)
    ]
    max_remediations = _command_flag_int(watchdog_command, "--max-remediations", 0)
    if missing_remediation_flags or max_remediations <= 0:
        _add_check(
            checks,
            "set_and_forget_watchdog_remediation",
            "error",
            "strict set-and-forget audit requires bounded watchdog remediation of the supervisor LaunchAgent",
            missing_flags=missing_remediation_flags,
            max_remediations=max_remediations,
        )
    else:
        _add_check(
            checks,
            "set_and_forget_watchdog_remediation",
            "ok",
            "watchdog command can boundedly remediate the supervisor LaunchAgent",
            max_remediations=max_remediations,
        )

    graceful_restart_timeout = _command_flag_float(watchdog_command, "--graceful-restart-timeout")
    if graceful_restart_timeout is not None and graceful_restart_timeout > 0:
        _add_check(
            checks,
            "set_and_forget_watchdog_graceful_restart",
            "ok",
            "watchdog command enables cooperative runner restart requests before launchd escalation",
            graceful_restart_timeout_seconds=graceful_restart_timeout,
        )
    else:
        _add_check(
            checks,
            "set_and_forget_watchdog_graceful_restart",
            "error",
            "strict set-and-forget audit requires positive watchdog graceful-restart timeout",
            graceful_restart_timeout_seconds=graceful_restart_timeout,
        )

    power = runbook.get("launchd_power_assertion") if isinstance(runbook.get("launchd_power_assertion"), Mapping) else {}
    if bool(power.get("enabled")) and str(power.get("program") or unattended.CAFFEINATE_BIN) == unattended.CAFFEINATE_BIN:
        _add_check(
            checks,
            "set_and_forget_sleep_prevention",
            "ok",
            "runbook requests caffeinate-wrapped LaunchAgents for sleep prevention",
            program=str(power.get("program") or unattended.CAFFEINATE_BIN),
        )
    else:
        _add_check(
            checks,
            "set_and_forget_sleep_prevention",
            "error",
            "strict set-and-forget audit requires caffeinate-wrapped LaunchAgents",
            enabled=bool(power.get("enabled")),
            program=str(power.get("program") or ""),
        )


def audit_operation(
    runbook_or_dir: Path,
    *,
    allow_running_incomplete: bool = False,
    strict_set_and_forget: bool = False,
    max_watchdog_status_age_seconds: float = 300.0,
    launchctl_bin: str = "launchctl",
    pmset_bin: str = "pmset",
    command_timeout_seconds: float = 30.0,
    launchd_settle_seconds: float = DEFAULT_LAUNCHD_SETTLE_SECONDS,
    launchd_settle_interval_seconds: float = DEFAULT_LAUNCHD_SETTLE_INTERVAL_SECONDS,
    command_runner: Callable[..., Any] = subprocess.run,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    runbook_path = resolve_runbook_path(runbook_or_dir)
    runbook = _read_json_object(runbook_path)
    if not runbook:
        _add_check(checks, "runbook", "error", f"runbook is missing or invalid: {runbook_path}")
        return {
            "schema_version": 1,
            "status": "error",
            "checked_at": _now_iso(),
            "runbook_path": str(runbook_path),
            "checks": checks,
        }
    _add_check(checks, "runbook", "ok", "runbook is readable", path=str(runbook_path))

    now = float(now_epoch if now_epoch is not None else time.time())
    artifact_report = _audit_artifacts(
        checks=checks,
        runbook=runbook,
        allow_running_incomplete=allow_running_incomplete,
    )
    _audit_watchdog_files(
        checks=checks,
        runbook=runbook,
        max_status_age_seconds=max_watchdog_status_age_seconds,
        now_epoch=now,
        strict_set_and_forget=bool(strict_set_and_forget),
    )
    _audit_readiness_file(checks, runbook)
    if strict_set_and_forget:
        _audit_strict_set_and_forget_contract(checks=checks, runbook=runbook)

    launchd = runbook.get("launchd_install") if isinstance(runbook.get("launchd_install"), Mapping) else {}
    launchctl = str(launchd.get("launchctl") or launchctl_bin or "launchctl")
    timeout = _launchd_timeout(runbook, command_timeout_seconds)
    power = runbook.get("launchd_power_assertion") if isinstance(runbook.get("launchd_power_assertion"), Mapping) else {}
    _launchd_print_check(
        checks=checks,
        runbook=runbook,
        role="supervisor",
        command_runner=command_runner,
        launchctl_bin=launchctl,
        timeout_seconds=timeout,
        settle_seconds=launchd_settle_seconds,
        settle_interval_seconds=launchd_settle_interval_seconds,
    )
    _launchd_print_check(
        checks=checks,
        runbook=runbook,
        role="watchdog",
        command_runner=command_runner,
        launchctl_bin=launchctl,
        timeout_seconds=timeout,
        settle_seconds=launchd_settle_seconds,
        settle_interval_seconds=launchd_settle_interval_seconds,
        require_caffeinate=bool(power.get("enabled")),
    )
    _audit_pmset(
        checks=checks,
        runbook=runbook,
        command_runner=command_runner,
        pmset_bin=pmset_bin,
        timeout_seconds=command_timeout_seconds,
    )

    status = _report_status(checks)
    return {
        "schema_version": 1,
        "status": status,
        "checked_at": _now_iso(),
        "runbook_path": str(runbook_path),
        "output_dir": runbook.get("output_dir"),
        "strict_set_and_forget": bool(strict_set_and_forget),
        "processed_cases": artifact_report.get("processed_cases"),
        "expected_cases": artifact_report.get("expected_cases"),
        "heartbeat": artifact_report.get("heartbeat"),
        "checks": checks,
    }


def _check_counts(report: Mapping[str, Any]) -> dict[str, int]:
    counts = {"ok": 0, "warn": 0, "error": 0}
    for check in report.get("checks") if isinstance(report.get("checks"), list) else []:
        if not isinstance(check, Mapping):
            continue
        status = str(check.get("status") or "").lower()
        if status in counts:
            counts[status] += 1
    return counts


def format_compact_report(report: Mapping[str, Any]) -> str:
    status = str(report.get("status") or "unknown").upper()
    mode = " strict set-and-forget" if bool(report.get("strict_set_and_forget")) else ""
    processed = report.get("processed_cases")
    expected = report.get("expected_cases")
    heartbeat = report.get("heartbeat") if isinstance(report.get("heartbeat"), Mapping) else {}
    counts = _check_counts(report)
    lines = [
        f"Qwen unattended operation{mode}: {status}",
        f"Runbook: {report.get('runbook_path') or ''}",
        f"Progress: {processed or 0}/{expected or 0} cases complete",
        f"Heartbeat: {heartbeat.get('status') or 'unknown'} / {heartbeat.get('phase') or 'unknown'}",
        f"Checks: {counts['ok']} ok, {counts['warn']} warn, {counts['error']} error",
    ]
    problems = [
        f"{check.get('name')}: {check.get('detail')}"
        for check in report.get("checks", [])
        if isinstance(check, Mapping) and str(check.get("status") or "").lower() != "ok"
    ]
    if problems:
        lines.append("Attention: " + "; ".join(problems[:4]))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runbook_or_dir", type=Path, help="unattended_run.json, run root, or run artifact directory")
    parser.add_argument("--allow-running-incomplete", action="store_true")
    parser.add_argument(
        "--strict-set-and-forget",
        action="store_true",
        help=(
            "Require the full set-and-forget handoff envelope: ok readiness, installed supervisor and watchdog "
            "LaunchAgents, watchdog remediation, sleep prevention, live durability gates, and a recorded strict "
            "operation-audit command."
        ),
    )
    parser.add_argument("--max-watchdog-status-age", type=float, default=300.0)
    parser.add_argument("--launchctl", default="launchctl")
    parser.add_argument("--pmset", default="pmset")
    parser.add_argument("--command-timeout", type=float, default=30.0)
    parser.add_argument(
        "--launchd-settle-seconds",
        type=float,
        default=DEFAULT_LAUNCHD_SETTLE_SECONDS,
        help=(
            "Seconds to wait for transient launchd restart states, such as 'spawn scheduled', "
            "before treating a loaded LaunchAgent as unhealthy."
        ),
    )
    parser.add_argument(
        "--launchd-settle-interval",
        type=float,
        default=DEFAULT_LAUNCHD_SETTLE_INTERVAL_SECONDS,
        help="Polling interval while waiting for a transient launchd state to settle.",
    )
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--write-json", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = audit_operation(
        args.runbook_or_dir,
        allow_running_incomplete=args.allow_running_incomplete,
        strict_set_and_forget=args.strict_set_and_forget,
        max_watchdog_status_age_seconds=args.max_watchdog_status_age,
        launchctl_bin=args.launchctl,
        pmset_bin=args.pmset,
        command_timeout_seconds=args.command_timeout,
        launchd_settle_seconds=args.launchd_settle_seconds,
        launchd_settle_interval_seconds=args.launchd_settle_interval,
    )
    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.compact:
        print(format_compact_report(report))
    else:
        print(json.dumps(report, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
