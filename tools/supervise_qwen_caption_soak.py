#!/usr/bin/env python3
"""Supervise a resumable Qwen caption soak until strict completion audit passes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import audit_qwen_caption_soak as audit  # noqa: E402
from tools import preflight_qwen_caption_soak as preflight  # noqa: E402
from tools import run_qwen_caption_flow_benchmark as runner  # noqa: E402


TERMINAL_SUCCESS = 0
TERMINAL_PRECHECK_FAILED = 1
TERMINAL_RESTART_LIMIT = 2
TERMINAL_STRICT_AUDIT_FAILED = 3
DEFAULT_SET_AND_FORGET_MLX_MAX_IMAGE_SIDE = 224
DEFAULT_SET_AND_FORGET_MIN_RETRY_IMAGE_SIDE = 192
DEFAULT_SET_AND_FORGET_COOLDOWN_AFTER_SUCCESS_SECONDS = 5.0
DEFAULT_SET_AND_FORGET_ATTEMPTS = 3


def _argv_has_flag(argv: Sequence[str], flag: str) -> bool:
    return any(str(item) == flag or str(item).startswith(f"{flag}=") for item in argv)


def apply_set_and_forget_runtime_defaults(
    args: argparse.Namespace,
    raw_argv: Sequence[str],
) -> bool:
    """Use lower-risk runtime defaults for unattended MLX caption runs."""

    if not bool(getattr(args, "set_and_forget", False)):
        return False
    changed = False
    if _argv_has_flag(raw_argv, "--mlx-max-image-side"):
        pass
    else:
        args.mlx_max_image_side = DEFAULT_SET_AND_FORGET_MLX_MAX_IMAGE_SIDE
        changed = True
    if not _argv_has_flag(raw_argv, "--min-retry-image-side"):
        args.min_retry_image_side = DEFAULT_SET_AND_FORGET_MIN_RETRY_IMAGE_SIDE
        changed = True
    if not _argv_has_flag(raw_argv, "--cooldown-after-success"):
        args.cooldown_after_success = DEFAULT_SET_AND_FORGET_COOLDOWN_AFTER_SUCCESS_SECONDS
        changed = True
    if not _argv_has_flag(raw_argv, "--attempts"):
        args.attempts = DEFAULT_SET_AND_FORGET_ATTEMPTS
        changed = True
    return changed


def _check_by_name(report: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    checks = report.get("checks")
    if not isinstance(checks, list):
        return None
    for check in checks:
        if isinstance(check, Mapping) and check.get("name") == name:
            return dict(check)
    return None


def _live_runner_lock_check(report: Mapping[str, Any]) -> dict[str, Any] | None:
    check = _check_by_name(report, "runner_lock")
    if not check or check.get("status") != "error":
        return None
    detail = str(check.get("detail") or "").lower()
    return check if "live runner" in detail else None


def _has_existing_run_artifacts(output_dir: Path) -> bool:
    return any((output_dir / name).exists() for name in ("manifest.json", "results.jsonl", "heartbeat.json", "summary.json"))


def _strict_audit(args: argparse.Namespace) -> dict[str, Any]:
    return audit.audit_soak(
        args.output_dir,
        max_heartbeat_age_seconds=args.max_heartbeat_age,
        allow_running_incomplete=False,
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
        set_and_forget=bool(getattr(args, "set_and_forget", False)),
        require_saved_text_labels=bool(
            getattr(args, "require_saved_text_labels", False)
            or getattr(args, "save_dataset_text_labels", False)
        ),
    )


def _emit_strict_audit(log_path: Path, *, run_index: int, strict_report: Mapping[str, Any], pre_run: bool) -> str:
    strict_status = str(strict_report.get("status") or "error")
    _event(
        log_path,
        "strict_audit",
        run_index=run_index,
        pre_run=pre_run,
        status=strict_status,
        checks=strict_report.get("checks"),
        processed_cases=strict_report.get("processed_cases"),
        expected_cases=strict_report.get("expected_cases"),
        failed_cases=strict_report.get("failed_cases"),
        incomplete_cases=strict_report.get("incomplete_cases"),
        degraded_rates=strict_report.get("degraded_rates"),
    )
    return strict_status


def _report_int_field(report: Mapping[str, Any], name: str) -> int | None:
    if name not in report:
        return None
    try:
        value = int(report.get(name) or 0)
    except (TypeError, ValueError, OverflowError):
        return None
    return max(0, value)


def _strict_audit_recoverable_by_resume(report: Mapping[str, Any]) -> bool:
    """Return whether another resume can plausibly improve a failed audit."""
    if not isinstance(report, Mapping):
        return True

    expected = _report_int_field(report, "expected_cases")
    processed = _report_int_field(report, "processed_cases")
    incomplete = _report_int_field(report, "incomplete_cases")
    failed = _report_int_field(report, "failed_cases")
    pending = _report_int_field(report, "pending_failed_attempt_cases")
    quality_failed = _report_int_field(report, "quality_failed_cases")

    if any(value is None for value in (expected, processed, incomplete, failed, pending, quality_failed)):
        return True
    if not expected and not processed:
        return True
    if incomplete or failed or pending or quality_failed:
        return True
    return False


def _strict_audit_terminal_stop_event(
    log_path: Path,
    *,
    run_index: int,
    strict_report: Mapping[str, Any],
    strict_status: str,
    last_runner_status: str | None = None,
    last_return_code: int | None = None,
    pre_run: bool = False,
) -> int:
    _event(
        log_path,
        "supervisor_stop",
        reason="strict_audit_failed_nonrecoverable",
        run_index=run_index,
        pre_run=pre_run,
        last_runner_status=last_runner_status,
        last_return_code=last_return_code,
        strict_audit_status=strict_status,
        expected_cases=strict_report.get("expected_cases"),
        processed_cases=strict_report.get("processed_cases"),
        failed_cases=strict_report.get("failed_cases"),
        incomplete_cases=strict_report.get("incomplete_cases"),
        pending_failed_attempt_cases=strict_report.get("pending_failed_attempt_cases"),
        quality_failed_cases=strict_report.get("quality_failed_cases"),
        degraded_rates=strict_report.get("degraded_rates"),
    )
    return TERMINAL_STRICT_AUDIT_FAILED


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _event(log_path: Path, event: str, **fields: Any) -> dict[str, Any]:
    payload = {
        "event": event,
        "time": time.time(),
        **fields,
    }
    _append_jsonl(log_path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def _heartbeat_age(output_dir: Path, *, min_mtime: float | None = None) -> float | None:
    heartbeat_path = output_dir / "heartbeat.json"
    if not heartbeat_path.exists():
        return None
    try:
        stat = heartbeat_path.stat()
    except OSError:
        return None
    if min_mtime is not None and stat.st_mtime < min_mtime:
        return None
    try:
        heartbeat = json.loads(heartbeat_path.read_text())
        if not isinstance(heartbeat, Mapping):
            return None
        epoch = float(heartbeat.get("heartbeat_epoch") or 0.0)
    except Exception:
        return None
    if epoch <= 0:
        return max(0.0, time.time() - stat.st_mtime)
    return max(0.0, time.time() - epoch)


def _process_group_id(process: subprocess.Popen[str]) -> int | None:
    try:
        return os.getpgid(process.pid)
    except Exception:
        return None


def _process_group_alive(pgid: int | None) -> bool:
    if pgid is None:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def _signal_process_tree(process: subprocess.Popen[str], sig: int, *, pgid: int | None = None) -> None:
    target_pgid = pgid if pgid is not None else _process_group_id(process)
    if target_pgid is not None:
        try:
            os.killpg(target_pgid, sig)
            return
        except ProcessLookupError:
            return
        except Exception:
            pass
    try:
        process.send_signal(sig)
    except Exception:
        pass


def _wait_process_tree_exit(
    process: subprocess.Popen[str],
    *,
    pgid: int | None,
    timeout_seconds: float,
) -> bool:
    deadline = time.time() + max(0.0, float(timeout_seconds or 0.0))
    while True:
        parent_running = process.poll() is None
        group_running = _process_group_alive(pgid)
        if not parent_running and not group_running:
            return True
        remaining = deadline - time.time()
        if remaining <= 0:
            return False
        if parent_running:
            try:
                process.wait(timeout=min(0.05, remaining))
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass
        else:
            time.sleep(min(0.05, remaining))


def _terminate_process(
    process: subprocess.Popen[str],
    *,
    kill_timeout: float = 10.0,
    pgid: int | None = None,
) -> None:
    resolved_pgid = pgid if pgid is not None else _process_group_id(process)
    if process.poll() is not None and not _process_group_alive(resolved_pgid):
        return
    _signal_process_tree(process, signal.SIGTERM, pgid=resolved_pgid)
    if _wait_process_tree_exit(process, pgid=resolved_pgid, timeout_seconds=max(0.1, float(kill_timeout or 10.0))):
        return
    _signal_process_tree(process, signal.SIGKILL, pgid=resolved_pgid)
    _wait_process_tree_exit(process, pgid=resolved_pgid, timeout_seconds=1.0)


def _read_stdout_thread(process: subprocess.Popen[str], output_queue: "queue.Queue[str | None]") -> None:
    stdout = process.stdout
    if stdout is None:
        output_queue.put(None)
        return
    try:
        for line in stdout:
            output_queue.put(line)
    finally:
        output_queue.put(None)


def _return_signal_fields(return_code: int | None) -> dict[str, Any]:
    if return_code is None or return_code >= 0:
        return {"return_signal": None, "return_signal_name": None}
    signal_number = abs(int(return_code))
    try:
        signal_name = signal.Signals(signal_number).name
    except ValueError:
        signal_name = f"SIG{signal_number}"
    return {
        "return_signal": signal_number,
        "return_signal_name": signal_name,
    }


def _preflight_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_root=args.dataset_root,
        cases_json=args.cases_json,
        request_json=args.request_json,
        output_dir=args.output_dir,
        all_images=args.all_images,
        caption_mode=args.caption_mode,
        windowed_full_image_strategy=args.windowed_full_image_strategy,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        case=list(args.case or []),
        limit=args.limit,
        resume=True,
        save_dataset_text_labels=args.save_dataset_text_labels,
        attempts=args.attempts,
        max_artifact_log_bytes=args.max_artifact_log_bytes,
        min_free_gb=args.min_free_gb,
        disk_safety_factor=args.disk_safety_factor,
        max_heartbeat_age=args.max_heartbeat_age,
        model_id=args.model_id,
        model_variant="Instruct",
        refinement_model_id=args.refinement_model_id,
        fallback_model_id=args.fallback_model_id,
        loop_recovery=args.loop_recovery,
        allow_model_download=args.allow_model_download,
        preview_only=args.preview_only,
        max_boxes=args.max_boxes,
        max_new_tokens=args.max_new_tokens,
        final_sentences=args.final_sentences,
        window_size=args.window_size,
        window_overlap=args.window_overlap,
        mlx_max_image_side=args.mlx_max_image_side,
        retry_image_side_scale=args.retry_image_side_scale,
        min_retry_image_side=args.min_retry_image_side,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        use_sampling=args.use_sampling,
        prompt=args.prompt,
    )


def _runner_command(args: argparse.Namespace, runner_extra_args: Sequence[str]) -> list[str]:
    script = args.runner_script or (REPO_ROOT / "tools" / "run_qwen_caption_flow_benchmark.py")
    cmd = [
        sys.executable,
        str(script),
        "--dataset-root",
        str(args.dataset_root),
        "--output-dir",
        str(args.output_dir),
        "--attempts",
        str(args.attempts),
        "--timeout",
        str(args.timeout),
        "--cooldown-after-crash",
        str(args.cooldown_after_crash),
        "--cooldown-after-success",
        str(args.cooldown_after_success),
        "--cooldown-backoff-multiplier",
        str(args.cooldown_backoff_multiplier),
        "--max-cooldown-after-crash",
        str(args.max_cooldown_after_crash),
        "--max-failures",
        str(args.max_failures),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--max-artifact-log-bytes",
        str(args.max_artifact_log_bytes),
        "--model-id",
        str(args.model_id),
        "--refinement-model-id",
        str(args.refinement_model_id),
        "--fallback-model-id",
        str(args.fallback_model_id),
        "--loop-recovery",
        str(args.loop_recovery),
        "--caption-mode",
        str(args.caption_mode),
        "--windowed-full-image-strategy",
        str(args.windowed_full_image_strategy),
        "--sample-size",
        str(args.sample_size),
        "--sample-seed",
        str(args.sample_seed),
        "--max-boxes",
        str(args.max_boxes),
        "--final-sentences",
        str(args.final_sentences),
        "--window-size",
        str(args.window_size),
        "--window-overlap",
        str(args.window_overlap),
        "--mlx-max-image-side",
        str(args.mlx_max_image_side),
        "--retry-image-side-scale",
        str(args.retry_image_side_scale),
        "--min-retry-image-side",
        str(args.min_retry_image_side),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--prompt",
        str(args.prompt),
        "--resume",
    ]
    if args.cases_json:
        cmd.extend(["--cases-json", str(args.cases_json)])
    if args.request_json:
        cmd.extend(["--request-json", str(args.request_json)])
    if args.all_images:
        cmd.append("--all-images")
    if args.max_new_tokens is not None:
        cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    for case in args.case or []:
        cmd.extend(["--case", str(case)])
    if args.preview_only:
        cmd.append("--preview-only")
    if args.use_sampling:
        cmd.append("--use-sampling")
    if args.continue_on_quality_failures:
        cmd.append("--continue-on-quality-failures")
    if args.skip_existing_captions:
        cmd.append("--skip-existing-captions")
    if args.save_dataset_text_labels:
        cmd.append("--save-dataset-text-labels")
    cmd.extend(runner_extra_args)
    return cmd


def _run_child_once(
    args: argparse.Namespace,
    *,
    log_path: Path,
    runner_extra_args: Sequence[str],
    run_index: int,
) -> tuple[str, int | str]:
    cmd = _runner_command(args, runner_extra_args)
    _event(log_path, "runner_start", run_index=run_index, command=" ".join(cmd))
    env = os.environ.copy()
    env.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    process = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    process_pgid = _process_group_id(process)
    output_queue: "queue.Queue[str | None]" = queue.Queue()
    reader = threading.Thread(
        target=_read_stdout_thread,
        args=(process, output_queue),
        name=f"qwen-caption-soak-supervisor-stdout-{run_index}",
        daemon=True,
    )
    reader.start()
    stale = False
    missing_heartbeat = False
    stdout_closed = False
    start = time.time()
    last_output_at = start
    monitor_interval = max(0.01, float(args.monitor_interval or 1.0))
    max_heartbeat_age = max(0.0, float(args.max_heartbeat_age or 0.0))
    heartbeat_startup_grace = max(
        max_heartbeat_age,
        max(0.0, float(getattr(args, "heartbeat_startup_grace", 0.0) or 0.0)),
    )
    while process.poll() is None:
        while True:
            try:
                item = output_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                stdout_closed = True
                continue
            last_output_at = time.time()
            if args.echo_runner_output:
                print(item, end="", flush=True)
        age = _heartbeat_age(args.output_dir, min_mtime=start - 0.001)
        if age is not None and max_heartbeat_age > 0 and age > max_heartbeat_age:
            stale = True
            _event(
                log_path,
                "runner_stale_heartbeat",
                run_index=run_index,
                heartbeat_age_seconds=age,
                max_heartbeat_age_seconds=max_heartbeat_age,
                last_output_age_seconds=max(0.0, time.time() - last_output_at),
            )
            _terminate_process(process, kill_timeout=args.kill_timeout, pgid=process_pgid)
            break
        if age is None and max_heartbeat_age > 0 and time.time() - start > heartbeat_startup_grace:
            missing_heartbeat = True
            _event(
                log_path,
                "runner_missing_heartbeat",
                run_index=run_index,
                max_heartbeat_age_seconds=max_heartbeat_age,
                heartbeat_startup_grace_seconds=heartbeat_startup_grace,
                elapsed_seconds=round(time.time() - start, 3),
                last_output_age_seconds=max(0.0, time.time() - last_output_at),
            )
            _terminate_process(process, kill_timeout=args.kill_timeout, pgid=process_pgid)
            break
        time.sleep(monitor_interval)
    return_code = process.poll()
    if return_code is None:
        _terminate_process(process, kill_timeout=args.kill_timeout, pgid=process_pgid)
        return_code = process.poll()
    while not stdout_closed:
        try:
            item = output_queue.get_nowait()
        except queue.Empty:
            break
        if item is None:
            stdout_closed = True
        elif args.echo_runner_output:
            print(item, end="", flush=True)
    reader.join(timeout=1.0)
    if missing_heartbeat:
        status = "missing_heartbeat"
    elif stale:
        status = "stale_heartbeat"
    else:
        if return_code == 0:
            status = "ok"
        elif return_code is not None and return_code < 0:
            status = "signal_exit"
        else:
            status = "nonzero_exit"
    _event(
        log_path,
        "runner_exit",
        run_index=run_index,
        status=status,
        return_code=return_code,
        **_return_signal_fields(return_code),
        elapsed_seconds=round(time.time() - start, 3),
    )
    return status, return_code if return_code is not None else "unknown"


def resolve_supervisor_set_and_forget_thresholds(args: argparse.Namespace) -> argparse.Namespace:
    args.max_loop_recovery_case_rate = audit.resolve_loop_recovery_threshold(
        getattr(args, "max_loop_recovery_case_rate", None),
        set_and_forget=bool(getattr(args, "set_and_forget", False)),
    )
    args.max_loop_guard_case_rate = audit.resolve_loop_guard_threshold(
        getattr(args, "max_loop_guard_case_rate", None),
        set_and_forget=bool(getattr(args, "set_and_forget", False)),
    )
    args.max_deterministic_recovery_case_rate = audit.resolve_deterministic_recovery_threshold(
        getattr(args, "max_deterministic_recovery_case_rate", None),
        set_and_forget=bool(getattr(args, "set_and_forget", False)),
    )
    args.max_signal_exit_attempt_row_rate = audit.resolve_signal_exit_attempt_threshold(
        getattr(args, "max_signal_exit_attempt_row_rate", None),
        set_and_forget=bool(getattr(args, "set_and_forget", False)),
    )
    return args


def supervise_soak(args: argparse.Namespace, runner_extra_args: Sequence[str] | None = None) -> int:
    runner_extra_args = list(runner_extra_args or [])
    resolve_supervisor_set_and_forget_thresholds(args)
    args.output_dir = args.output_dir.expanduser().resolve(strict=False)
    log_path = (args.log_jsonl or (args.output_dir / "supervisor.jsonl")).expanduser().resolve(strict=False)
    max_restarts = max(0, int(args.max_runner_restarts or 0))
    restart_delay = max(0.0, float(args.restart_delay or 0.0))
    live_lock_wait_timeout = max(0.0, float(args.live_lock_wait_timeout or 0.0))
    live_lock_wait_started: float | None = None
    restart_count = 0
    run_index = 0
    while True:
        preflight_report = preflight.preflight_soak(_preflight_args(args))
        _event(
            log_path,
            "preflight",
            run_index=run_index + 1,
            status=preflight_report.get("status"),
            checks=preflight_report.get("checks"),
            resume=preflight_report.get("resume"),
            disk=preflight_report.get("disk"),
            model_cache=preflight_report.get("model_cache"),
        )
        preflight_status = str(preflight_report.get("status") or "error")
        live_runner_lock = _live_runner_lock_check(preflight_report)
        if preflight_status == "error" and live_runner_lock is not None:
            now = time.monotonic()
            if live_lock_wait_started is None:
                live_lock_wait_started = now
            elapsed = now - live_lock_wait_started
            if live_lock_wait_timeout > 0 and elapsed >= live_lock_wait_timeout:
                _event(
                    log_path,
                    "supervisor_stop",
                    reason="live_runner_lock_timeout",
                    status=preflight_status,
                    elapsed_seconds=round(elapsed, 3),
                    timeout_seconds=live_lock_wait_timeout,
                    runner_lock=live_runner_lock,
                )
                return TERMINAL_PRECHECK_FAILED
            _event(
                log_path,
                "preflight_live_runner_lock_wait",
                run_index=run_index + 1,
                elapsed_seconds=round(elapsed, 3),
                timeout_seconds=live_lock_wait_timeout,
                retry_after_seconds=max(0.01, float(args.monitor_interval or 1.0)),
                runner_lock=live_runner_lock,
            )
            time.sleep(max(0.01, float(args.monitor_interval or 1.0)))
            continue
        live_lock_wait_started = None
        if preflight_status == "error" or (args.fail_on_warn and preflight_status == "warn"):
            _event(log_path, "supervisor_stop", reason="preflight_failed", status=preflight_status)
            return TERMINAL_PRECHECK_FAILED
        if _has_existing_run_artifacts(args.output_dir):
            pre_run_strict_report = _strict_audit(args)
            pre_run_strict_status = _emit_strict_audit(
                log_path,
                run_index=run_index,
                strict_report=pre_run_strict_report,
                pre_run=True,
            )
            if pre_run_strict_status == "ok":
                _event(
                    log_path,
                    "supervisor_complete",
                    run_index=run_index,
                    restarts=restart_count,
                    already_complete=True,
                )
                return TERMINAL_SUCCESS
            if not _strict_audit_recoverable_by_resume(pre_run_strict_report):
                return _strict_audit_terminal_stop_event(
                    log_path,
                    run_index=run_index,
                    strict_report=pre_run_strict_report,
                    strict_status=pre_run_strict_status,
                    pre_run=True,
                )
        run_index += 1
        status, return_code = _run_child_once(
            args,
            log_path=log_path,
            runner_extra_args=runner_extra_args,
            run_index=run_index,
        )
        strict_report = _strict_audit(args)
        strict_status = _emit_strict_audit(
            log_path,
            run_index=run_index,
            strict_report=strict_report,
            pre_run=False,
        )
        if return_code == 0 and strict_status == "ok":
            _event(log_path, "supervisor_complete", run_index=run_index, restarts=restart_count)
            return TERMINAL_SUCCESS
        if strict_status != "ok" and not _strict_audit_recoverable_by_resume(strict_report):
            return _strict_audit_terminal_stop_event(
                log_path,
                run_index=run_index,
                strict_report=strict_report,
                strict_status=strict_status,
                last_runner_status=status,
                last_return_code=return_code,
            )
        if restart_count >= max_restarts:
            _event(
                log_path,
                "supervisor_stop",
                reason="restart_limit",
                run_index=run_index,
                restarts=restart_count,
                last_runner_status=status,
                last_return_code=return_code,
                strict_audit_status=strict_status,
            )
            return TERMINAL_RESTART_LIMIT
        restart_count += 1
        _event(
            log_path,
            "supervisor_restart",
            next_run_index=run_index + 1,
            restarts=restart_count,
            max_runner_restarts=max_restarts,
            delay_seconds=restart_delay,
            last_runner_status=status,
            last_return_code=return_code,
            strict_audit_status=strict_status,
        )
        if restart_delay:
            time.sleep(restart_delay)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=runner.DEFAULT_DATASET)
    parser.add_argument("--cases-json", type=Path, default=None)
    parser.add_argument("--request-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=preflight.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--all-images", action="store_true")
    parser.add_argument("--caption-mode", choices=["full", "windowed"], default="full")
    parser.add_argument("--windowed-full-image-strategy", choices=["visual", "text_only"], default="visual")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--cooldown-after-crash", type=float, default=5.0)
    parser.add_argument("--cooldown-after-success", type=float, default=0.0)
    parser.add_argument("--cooldown-backoff-multiplier", type=float, default=runner.DEFAULT_COOLDOWN_BACKOFF_MULTIPLIER)
    parser.add_argument("--max-cooldown-after-crash", type=float, default=runner.DEFAULT_MAX_COOLDOWN_AFTER_CRASH)
    parser.add_argument("--max-failures", type=int, default=0)
    parser.add_argument("--continue-on-quality-failures", action="store_true")
    parser.add_argument("--skip-existing-captions", action="store_true")
    parser.add_argument("--save-dataset-text-labels", action="store_true")
    parser.add_argument(
        "--require-saved-text-labels",
        action="store_true",
        help=(
            "Require terminal strict audits to prove each generated/resumed success has an existing "
            "non-empty saved dataset text-label file. This is implied by --save-dataset-text-labels."
        ),
    )
    parser.add_argument("--max-artifact-log-bytes", type=int, default=runner.DEFAULT_ARTIFACT_LOG_BYTES)
    parser.add_argument("--min-free-gb", type=float, default=preflight.DEFAULT_MIN_FREE_GB)
    parser.add_argument("--disk-safety-factor", type=float, default=preflight.DEFAULT_DISK_SAFETY_FACTOR)
    parser.add_argument("--max-heartbeat-age", type=float, default=900.0)
    parser.add_argument(
        "--heartbeat-startup-grace",
        type=float,
        default=120.0,
        help=(
            "Seconds to wait for the runner's first current heartbeat before treating it as missing. "
            "Existing current heartbeats are still judged by --max-heartbeat-age."
        ),
    )
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
    parser.add_argument("--min-rate-cases", type=int, default=20)
    parser.add_argument("--monitor-interval", type=float, default=5.0)
    parser.add_argument("--kill-timeout", type=float, default=10.0)
    parser.add_argument(
        "--live-lock-wait-timeout",
        type=float,
        default=0.0,
        help="Maximum seconds to wait when preflight finds a live runner lock. 0 waits indefinitely.",
    )
    parser.add_argument("--max-runner-restarts", type=int, default=25)
    parser.add_argument("--restart-delay", type=float, default=30.0)
    parser.add_argument("--fail-on-warn", action="store_true")
    parser.add_argument(
        "--set-and-forget",
        action="store_true",
        help=(
            "Use unattended-run health defaults. This keeps failed and quality rates strict, "
            "but allows small bounded loop-recovery and deterministic-recovery rates unless "
            "explicitly overridden."
        ),
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow the supervised run to start even when a selected concrete model must be downloaded.",
    )
    parser.add_argument("--log-jsonl", type=Path, default=None)
    parser.add_argument("--runner-script", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--echo-runner-output", action="store_true")
    parser.add_argument("--model-id", default=runner.DEFAULT_MODEL)
    parser.add_argument("--refinement-model-id", default="same")
    parser.add_argument("--fallback-model-id", default="auto")
    parser.add_argument("--loop-recovery", default="safe_retry_fallback")
    parser.add_argument("--max-boxes", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--final-sentences", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=672)
    parser.add_argument("--window-overlap", type=float, default=0.1)
    parser.add_argument("--mlx-max-image-side", type=int, default=512)
    parser.add_argument("--retry-image-side-scale", type=float, default=runner.DEFAULT_RETRY_IMAGE_SIDE_SCALE)
    parser.add_argument("--min-retry-image-side", type=int, default=runner.DEFAULT_MIN_RETRY_IMAGE_SIDE)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--use-sampling", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--heartbeat-interval", type=float, default=30.0)
    parser.add_argument("--prompt", default=runner.DEFAULT_PROMPT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args, runner_extra_args = parser.parse_known_args(raw_argv)
    apply_set_and_forget_runtime_defaults(args, raw_argv)
    if (
        bool(getattr(args, "set_and_forget", False))
        and str(getattr(args, "caption_mode", "full") or "full") == "windowed"
        and not _argv_has_flag(raw_argv, "--windowed-full-image-strategy")
    ):
        args.windowed_full_image_strategy = "text_only"
    return supervise_soak(args, runner_extra_args=runner_extra_args)


if __name__ == "__main__":
    raise SystemExit(main())
