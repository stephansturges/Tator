#!/usr/bin/env python3
"""Prepare and optionally run a launchd-friendly unattended Qwen caption soak."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import plistlib
import subprocess
import shlex
import sys
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import certify_qwen_caption_soak as certify  # noqa: E402
from tools import preflight_qwen_caption_soak as preflight  # noqa: E402
from tools import run_qwen_caption_flow_benchmark as runner  # noqa: E402
from tools import run_qwen_caption_soak_drill as soak_drill  # noqa: E402
from tools import supervise_qwen_caption_soak as supervise  # noqa: E402


DEFAULT_LAUNCHD_LABEL = "com.tator.qwen-caption-soak"
DEFAULT_WATCHDOG_LAUNCHD_LABEL = "com.tator.qwen-caption-soak.watchdog"
RUNBOOK_NAME = "unattended_run.json"
READINESS_NAME = "readiness.json"
LAUNCHD_STDOUT_NAME = "launchd_stdout.log"
LAUNCHD_STDERR_NAME = "launchd_stderr.log"
WATCHDOG_LAUNCHD_STDOUT_NAME = "watchdog_launchd_stdout.log"
WATCHDOG_LAUNCHD_STDERR_NAME = "watchdog_launchd_stderr.log"
WATCHDOG_LATEST_NAME = "watchdog_latest.json"
WATCHDOG_STATE_NAME = "watchdog_state.json"
SET_AND_FORGET_DRILL_CASES = 10_000
DEFAULT_SUPERVISOR_DRILL_CHUNK_SIZE = 1000
DEFAULT_LAUNCHCTL_TIMEOUT_SECONDS = 30.0
DEFAULT_POST_INSTALL_OPERATION_AUDIT_TIMEOUT_SECONDS = 300.0
DEFAULT_POST_INSTALL_OPERATION_AUDIT_INTERVAL_SECONDS = 5.0
CAFFEINATE_BIN = "/usr/bin/caffeinate"
CAFFEINATE_ARGS = ["-dimsu"]
STATUS_RANK = {"ok": 0, "warn": 1, "error": 2}
DEFAULT_TENK_PILOT_MAX_PROMPT_TOKENS = 9000
DEFAULT_TENK_MIN_PILOT_CASES = 300
DEFAULT_TENK_PILOT_SUBDIR = "pilot"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _quote_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preflight and write artifacts without starting the supervisor.",
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Write artifacts and run without the wrapper-level preflight. The supervisor still preflights.",
    )
    parser.add_argument(
        "--runbook-json",
        type=Path,
        default=None,
        help="Runbook path. Defaults to <output-dir>/unattended_run.json.",
    )
    parser.add_argument(
        "--write-launchd-plist",
        type=Path,
        default=None,
        help="Write a macOS LaunchAgent plist that runs the supervisor command.",
    )
    parser.add_argument(
        "--launchd-label",
        default=DEFAULT_LAUNCHD_LABEL,
        help="LaunchAgent label used when --write-launchd-plist is set.",
    )
    parser.add_argument(
        "--launchd-throttle-interval",
        type=int,
        default=300,
        help="LaunchAgent restart throttle interval in seconds.",
    )
    parser.add_argument(
        "--launchd-caffeinate",
        dest="launchd_caffeinate",
        action="store_true",
        default=None,
        help="Wrap generated LaunchAgent commands with caffeinate to prevent macOS sleep.",
    )
    parser.add_argument(
        "--no-launchd-caffeinate",
        dest="launchd_caffeinate",
        action="store_false",
        help="Do not wrap generated LaunchAgent commands with caffeinate, even in --tenk-set-and-forget mode.",
    )
    parser.add_argument(
        "--write-watchdog-launchd-plist",
        type=Path,
        default=None,
        help="Write a second macOS LaunchAgent plist that runs the independent artifact watchdog.",
    )
    parser.add_argument(
        "--install-launchd-plists",
        action="store_true",
        help=(
            "Install and start the supervisor and watchdog LaunchAgents after all launch gates pass. "
            "When plist paths are omitted, defaults to ~/Library/LaunchAgents/<label>.plist for both roles. "
            "In --dry-run mode this only records the launchctl plan."
        ),
    )
    parser.add_argument(
        "--adopt-live-run",
        action="store_true",
        help=(
            "When wrapper preflight finds a live runner lock, treat this as an intentional launchd handoff "
            "for an already-running soak. The generated supervisor waits on the live lock instead of "
            "starting a competing runner. Strict readiness requires --install-launchd-plists and a clean "
            "live adoption audit."
        ),
    )
    parser.add_argument(
        "--launchctl",
        default="launchctl",
        help="launchctl executable used by --install-launchd-plists.",
    )
    parser.add_argument(
        "--launchctl-domain",
        default=None,
        help="launchctl domain used by --install-launchd-plists. Defaults to gui/<uid>.",
    )
    parser.add_argument(
        "--launchctl-timeout",
        type=float,
        default=DEFAULT_LAUNCHCTL_TIMEOUT_SECONDS,
        help="Maximum seconds to wait for each launchctl install/verification command.",
    )
    parser.add_argument(
        "--skip-post-install-operation-audit",
        action="store_true",
        help=(
            "After --install-launchd-plists, do not wait for the strict live operation audit to pass. "
            "This is for manual diagnostics; --tenk-set-and-forget rejects it."
        ),
    )
    parser.add_argument(
        "--post-install-operation-audit-timeout",
        type=float,
        default=DEFAULT_POST_INSTALL_OPERATION_AUDIT_TIMEOUT_SECONDS,
        help=(
            "Maximum seconds to poll the live operation audit after installing LaunchAgents in strict readiness mode."
        ),
    )
    parser.add_argument(
        "--post-install-operation-audit-interval",
        type=float,
        default=DEFAULT_POST_INSTALL_OPERATION_AUDIT_INTERVAL_SECONDS,
        help="Polling interval for the post-install live operation audit.",
    )
    parser.add_argument(
        "--watchdog-launchd-label",
        default=DEFAULT_WATCHDOG_LAUNCHD_LABEL,
        help="LaunchAgent label used when --write-watchdog-launchd-plist is set.",
    )
    parser.add_argument(
        "--watchdog-launchd-throttle-interval",
        type=int,
        default=300,
        help="Watchdog LaunchAgent restart throttle interval in seconds.",
    )
    parser.add_argument(
        "--watchdog-max-no-progress",
        type=float,
        default=3600.0,
        help="Maximum seconds the watchdog may observe no completed-case progress before exiting unhealthy. 0 disables.",
    )
    parser.add_argument(
        "--watchdog-max-remediations",
        type=int,
        default=25,
        help=(
            "Maximum supervisor LaunchAgent kickstarts the watchdog may attempt during a launchd-backed "
            "set-and-forget run. 0 disables automatic remediation."
        ),
    )
    parser.add_argument(
        "--watchdog-remediation-cooldown",
        type=float,
        default=300.0,
        help="Minimum seconds between watchdog supervisor kickstarts.",
    )
    parser.add_argument(
        "--watchdog-remediation-timeout",
        type=float,
        default=DEFAULT_LAUNCHCTL_TIMEOUT_SECONDS,
        help="Maximum seconds to wait for each watchdog launchctl kickstart command.",
    )
    parser.add_argument(
        "--watchdog-graceful-restart-timeout",
        type=float,
        default=300.0,
        help=(
            "Seconds the watchdog waits for a capable runner to acknowledge restart_requested.json "
            "before launchd escalation is allowed. 0 disables cooperative restart requests."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to record in supervisor/watch/audit commands.",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the generated runbook JSON.",
    )
    parser.add_argument(
        "--readiness-json",
        type=Path,
        default=None,
        help="Readiness report path. Defaults to <output-dir>/readiness.json.",
    )
    parser.add_argument(
        "--require-readiness-ok",
        action="store_true",
        help="Block launch unless the consolidated 10k set-and-forget readiness report is ok.",
    )
    parser.add_argument(
        "--tenk-set-and-forget",
        action="store_true",
        help=(
            "Fail closed for a true 10k unattended launch. This requires the same consolidated readiness "
            "report as --require-readiness-ok, including pilot certification and supervisor/watchdog LaunchAgents."
        ),
    )
    parser.add_argument(
        "--require-pilot-certification",
        type=Path,
        default=None,
        metavar="PILOT_OUTPUT_DIR",
        help="Require a clean certification report from this pilot artifact directory before starting.",
    )
    parser.add_argument(
        "--create-pilot-output-dir",
        type=Path,
        default=None,
        metavar="PILOT_OUTPUT_DIR",
        help="Run a supervised pilot sample into this directory before certifying and launching the large run.",
    )
    parser.add_argument(
        "--pilot-sample-size",
        type=int,
        default=0,
        help="Sample size for --create-pilot-output-dir. 0 uses --pilot-min-cases.",
    )
    parser.add_argument(
        "--skip-supervisor-drill",
        action="store_true",
        help="Skip the no-GPU supervisor interruption drill before launch.",
    )
    parser.add_argument(
        "--skip-watchdog-drill",
        action="store_true",
        help="Skip the no-GPU watchdog launchd-remediation drill before launch.",
    )
    parser.add_argument(
        "--supervisor-drill-output-dir",
        type=Path,
        default=None,
        help="Directory for the launch-gate supervisor drill. Defaults to <output-dir>/supervisor_drill.",
    )
    parser.add_argument(
        "--watchdog-drill-output-dir",
        type=Path,
        default=None,
        help="Directory for the launch-gate watchdog drill. Defaults to <output-dir>/watchdog_drill.",
    )
    parser.add_argument(
        "--supervisor-drill-cases",
        type=int,
        default=SET_AND_FORGET_DRILL_CASES,
        help="Synthetic cases used by the no-GPU supervisor launch drill. Use 1 for the minimal drill.",
    )
    parser.add_argument(
        "--supervisor-drill-chunk-size",
        type=int,
        default=DEFAULT_SUPERVISOR_DRILL_CHUNK_SIZE,
        help="Synthetic cases completed per fake-runner invocation when supervisor drill cases > 1.",
    )
    parser.add_argument(
        "--pilot-target-cases",
        type=int,
        default=certify.DEFAULT_TARGET_CASES,
        help="Target case count used when certifying --require-pilot-certification.",
    )
    parser.add_argument(
        "--pilot-max-duration-hours",
        type=float,
        default=certify.DEFAULT_MAX_DURATION_HOURS,
        help="Maximum projected target runtime used when certifying --require-pilot-certification.",
    )
    parser.add_argument(
        "--pilot-max-p95-duration-hours",
        type=float,
        default=None,
        help=(
            "Maximum target runtime if all cases ran at the pilot p95 case time. "
            "Defaults to --pilot-max-duration-hours. Use -1 to disable."
        ),
    )
    parser.add_argument(
        "--pilot-min-cases",
        type=int,
        default=certify.DEFAULT_MIN_PILOT_CASES,
        help="Minimum timed pilot cases required when certifying --require-pilot-certification.",
    )
    parser.add_argument(
        "--adopt-min-completed-cases",
        type=int,
        default=certify.DEFAULT_MIN_PILOT_CASES,
        help="Minimum completed cases required before --adopt-live-run can substitute the active run for a pilot.",
    )
    parser.add_argument(
        "--adopt-min-prompt-budget-coverage",
        type=float,
        default=0.95,
        help=(
            "Minimum fraction of latest completed cases that must carry prompt-budget telemetry before "
            "--adopt-live-run can satisfy the live adoption evidence gate."
        ),
    )
    parser.add_argument(
        "--pilot-duration-safety-factor",
        type=float,
        default=certify.DEFAULT_DURATION_SAFETY_FACTOR,
        help="Safety multiplier applied to observed pilot runtime projection.",
    )
    parser.add_argument(
        "--no-pilot-prompt-budget-required",
        action="store_true",
        help="Allow legacy pilot artifacts without per-row prompt-budget telemetry.",
    )
    parser.add_argument(
        "--pilot-max-prompt-tokens",
        type=int,
        default=None,
        help=(
            "Maximum observed pilot prompt-token estimate allowed. "
            "Omitted means auto in --tenk-set-and-forget. "
            "0 disables this gate for manual diagnostics; --tenk-set-and-forget rejects explicit 0."
        ),
    )
    parser.add_argument(
        "--pilot-max-prompt-budget-adapted-case-rate",
        type=float,
        default=certify.DEFAULT_MAX_PROMPT_BUDGET_ADAPTED_CASE_RATE,
        help="Maximum fraction of pilot cases whose prompt budget required adaptation. Use -1 to disable.",
    )
    parser.add_argument(
        "--pilot-deterministic-recovery-confidence",
        type=float,
        default=certify.DEFAULT_DETERMINISTIC_RECOVERY_CONFIDENCE,
        help=(
            "One-sided confidence used by pilot certification to bound deterministic count/layout "
            "recovery in set-and-forget mode. Use 0 only for manual diagnostics."
        ),
    )
    parser.epilog = (
        "All unrecognized arguments are passed to tools/supervise_qwen_caption_soak.py. "
        "Example: tools/run_qwen_caption_unattended.py --dry-run -- --dataset-root /data --all-images"
    )
    return parser


def parse_wrapper_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = _wrapper_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args, supervisor_argv = parser.parse_known_args(raw_argv)
    args.pilot_min_cases_explicit = "--pilot-min-cases" in raw_argv
    args.pilot_sample_size_explicit = "--pilot-sample-size" in raw_argv
    supervisor_argv = [str(item) for item in supervisor_argv if str(item) != "--"]
    if bool(getattr(args, "tenk_set_and_forget", False)):
        args.require_readiness_ok = True
        if getattr(args, "launchd_caffeinate", None) is None:
            args.launchd_caffeinate = True
        if "--set-and-forget" not in supervisor_argv:
            supervisor_argv.append("--set-and-forget")
        if "--max-projected-duration-hours" not in supervisor_argv:
            supervisor_argv.extend([
                "--max-projected-duration-hours",
                _cli_number(certify.DEFAULT_MAX_DURATION_HOURS),
            ])
    if bool(getattr(args, "install_launchd_plists", False)):
        launch_agents = Path.home() / "Library" / "LaunchAgents"
        if not args.write_launchd_plist:
            args.write_launchd_plist = launch_agents / f"{args.launchd_label}.plist"
        if not args.write_watchdog_launchd_plist:
            args.write_watchdog_launchd_plist = launch_agents / f"{args.watchdog_launchd_label}.plist"
    if getattr(args, "launchd_caffeinate", None) is None:
        args.launchd_caffeinate = False
    return args, supervisor_argv


def parse_supervisor_args(supervisor_argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    supervisor_parser = supervise.build_parser()
    return supervisor_parser.parse_known_args(list(supervisor_argv))


def _threshold_args(args: argparse.Namespace, *, include_projected_duration: bool = True) -> list[str]:
    set_and_forget = bool(getattr(args, "set_and_forget", False))
    max_loop_recovery_case_rate = supervise.audit.resolve_loop_recovery_threshold(
        getattr(args, "max_loop_recovery_case_rate", None),
        set_and_forget=set_and_forget,
    )
    max_loop_guard_case_rate = supervise.audit.resolve_loop_guard_threshold(
        getattr(args, "max_loop_guard_case_rate", None),
        set_and_forget=set_and_forget,
    )
    max_deterministic_recovery_case_rate = supervise.audit.resolve_deterministic_recovery_threshold(
        getattr(args, "max_deterministic_recovery_case_rate", None),
        set_and_forget=set_and_forget,
    )
    max_signal_exit_attempt_row_rate = supervise.audit.resolve_signal_exit_attempt_threshold(
        getattr(args, "max_signal_exit_attempt_row_rate", None),
        set_and_forget=set_and_forget,
    )
    threshold_args = [
        "--max-failed-case-rate",
        str(args.max_failed_case_rate),
        "--max-quality-failure-rate",
        str(args.max_quality_failure_rate),
        "--max-recovery-event-case-rate",
        str(args.max_recovery_event_case_rate),
        "--max-loop-recovery-case-rate",
        str(max_loop_recovery_case_rate),
        "--max-loop-guard-case-rate",
        str(max_loop_guard_case_rate),
        "--max-deterministic-recovery-case-rate",
        str(max_deterministic_recovery_case_rate),
        "--max-failed-attempt-row-rate",
        str(args.max_failed_attempt_row_rate),
        "--max-signal-exit-attempt-row-rate",
        str(max_signal_exit_attempt_row_rate),
        "--max-attempt-overrun",
        str(args.max_attempt_overrun),
        "--min-rate-cases",
        str(args.min_rate_cases),
    ]
    if include_projected_duration:
        threshold_args.extend([
            "--max-projected-duration-hours",
            str(args.max_projected_duration_hours),
        ])
    return threshold_args


def _disk_reserve_args(args: argparse.Namespace) -> list[str]:
    return ["--min-free-gb", str(getattr(args, "min_free_gb", 0.0))]


def _saved_text_label_audit_args(args: argparse.Namespace) -> list[str]:
    if bool(
        getattr(args, "require_saved_text_labels", False)
        or getattr(args, "save_dataset_text_labels", False)
    ):
        return ["--require-saved-text-labels"]
    return []


def _ensure_cli_flag_value(command: list[str], flag: str, value: Any) -> None:
    if flag not in command:
        command.extend([flag, str(value)])


def _same_resolved_path(left: Any, right: Any) -> bool:
    try:
        left_path = Path(str(left)).expanduser().resolve(strict=False)
        right_path = Path(str(right)).expanduser().resolve(strict=False)
    except (TypeError, ValueError, OSError):
        return False
    return left_path == right_path


def _pilot_output_dir(wrapper_args: argparse.Namespace, output_dir: Path) -> Path:
    raw = wrapper_args.create_pilot_output_dir or wrapper_args.require_pilot_certification
    if raw:
        return raw.expanduser().resolve(strict=False)
    return output_dir


def _pilot_certification_required(wrapper_args: argparse.Namespace) -> bool:
    return bool(wrapper_args.require_pilot_certification or wrapper_args.create_pilot_output_dir)


def _tenk_auto_pilot_case_count(wrapper_args: argparse.Namespace) -> int:
    target_cases = max(1, int(getattr(wrapper_args, "pilot_target_cases", certify.DEFAULT_TARGET_CASES) or 1))
    return max(
        certify.DEFAULT_MIN_PILOT_CASES,
        DEFAULT_TENK_MIN_PILOT_CASES,
        int(math.ceil(math.sqrt(float(target_cases)))),
    )


def apply_tenk_set_and_forget_defaults(
    wrapper_args: argparse.Namespace,
    supervisor_args: argparse.Namespace,
) -> None:
    """Make --tenk-set-and-forget a real turnkey launch while preserving overrides."""
    if not bool(getattr(wrapper_args, "tenk_set_and_forget", False)):
        return
    if getattr(wrapper_args, "pilot_max_prompt_tokens", None) is None:
        wrapper_args.pilot_max_prompt_tokens = DEFAULT_TENK_PILOT_MAX_PROMPT_TOKENS
        wrapper_args.auto_pilot_max_prompt_tokens = True
    elif not hasattr(wrapper_args, "auto_pilot_max_prompt_tokens"):
        wrapper_args.auto_pilot_max_prompt_tokens = False

    has_explicit_evidence = bool(
        getattr(wrapper_args, "require_pilot_certification", None)
        or getattr(wrapper_args, "create_pilot_output_dir", None)
        or getattr(wrapper_args, "adopt_live_run", False)
    )
    if not has_explicit_evidence:
        output_dir = supervisor_args.output_dir.expanduser().resolve(strict=False)
        wrapper_args.create_pilot_output_dir = output_dir / DEFAULT_TENK_PILOT_SUBDIR
        wrapper_args.auto_create_pilot_output_dir = True
    elif not hasattr(wrapper_args, "auto_create_pilot_output_dir"):
        wrapper_args.auto_create_pilot_output_dir = False

    auto_pilot_cases = _tenk_auto_pilot_case_count(wrapper_args)
    if not bool(getattr(wrapper_args, "pilot_min_cases_explicit", False)):
        wrapper_args.pilot_min_cases = auto_pilot_cases
        wrapper_args.auto_pilot_min_cases = True
    elif not hasattr(wrapper_args, "auto_pilot_min_cases"):
        wrapper_args.auto_pilot_min_cases = False

    if (
        bool(getattr(wrapper_args, "create_pilot_output_dir", None))
        and not bool(getattr(wrapper_args, "pilot_sample_size_explicit", False))
        and int(getattr(wrapper_args, "pilot_sample_size", 0) or 0) <= 0
    ):
        wrapper_args.pilot_sample_size = auto_pilot_cases
        wrapper_args.auto_pilot_sample_size = True
    elif not hasattr(wrapper_args, "auto_pilot_sample_size"):
        wrapper_args.auto_pilot_sample_size = False


def _argv_has_flag(argv: Sequence[str], flag: str) -> bool:
    return any(str(item) == flag or str(item).startswith(f"{flag}=") for item in argv)


def apply_set_and_forget_windowed_strategy_default(
    supervisor_args: argparse.Namespace,
    supervisor_argv: Sequence[str],
) -> bool:
    """Default unattended windowed set-and-forget to the lower-risk composition path."""
    if (
        not bool(getattr(supervisor_args, "set_and_forget", False))
        or str(getattr(supervisor_args, "caption_mode", "full") or "full") != "windowed"
        or _argv_has_flag(supervisor_argv, "--windowed-full-image-strategy")
    ):
        return False
    supervisor_args.windowed_full_image_strategy = "text_only"
    return True


def apply_set_and_forget_runtime_defaults(
    supervisor_args: argparse.Namespace,
    supervisor_argv: Sequence[str],
) -> bool:
    """Keep launchd/wrapper set-and-forget runs aligned with supervisor defaults."""

    if not bool(getattr(supervisor_args, "set_and_forget", False)):
        return False
    changed = False
    if not _argv_has_flag(supervisor_argv, "--mlx-max-image-side"):
        supervisor_args.mlx_max_image_side = supervise.DEFAULT_SET_AND_FORGET_MLX_MAX_IMAGE_SIDE
        changed = True
    if not _argv_has_flag(supervisor_argv, "--min-retry-image-side"):
        supervisor_args.min_retry_image_side = supervise.DEFAULT_SET_AND_FORGET_MIN_RETRY_IMAGE_SIDE
        changed = True
    if not _argv_has_flag(supervisor_argv, "--cooldown-after-success"):
        supervisor_args.cooldown_after_success = supervise.DEFAULT_SET_AND_FORGET_COOLDOWN_AFTER_SUCCESS_SECONDS
        changed = True
    if not _argv_has_flag(supervisor_argv, "--attempts"):
        supervisor_args.attempts = supervise.DEFAULT_SET_AND_FORGET_ATTEMPTS
        changed = True
    return changed


def _command_paths(output_dir: Path) -> dict[str, str]:
    supervisor_drill_dir = output_dir / "supervisor_drill"
    watchdog_drill_dir = output_dir / "watchdog_drill"
    return {
        "runbook_json": str(output_dir / RUNBOOK_NAME),
        "supervisor_jsonl": str(output_dir / "supervisor.jsonl"),
        "watchdog_jsonl": str(output_dir / "watchdog.jsonl"),
        "watchdog_latest_json": str(output_dir / WATCHDOG_LATEST_NAME),
        "watchdog_state_json": str(output_dir / WATCHDOG_STATE_NAME),
        "launchd_stdout": str(output_dir / LAUNCHD_STDOUT_NAME),
        "launchd_stderr": str(output_dir / LAUNCHD_STDERR_NAME),
        "watchdog_launchd_stdout": str(output_dir / WATCHDOG_LAUNCHD_STDOUT_NAME),
        "watchdog_launchd_stderr": str(output_dir / WATCHDOG_LAUNCHD_STDERR_NAME),
        "readiness_json": str(output_dir / READINESS_NAME),
        "final_audit_json": str(output_dir / "final_audit.json"),
        "operational_audit_json": str(output_dir / "operational_audit.json"),
        "required_pilot_certification_json": str(output_dir / "required_pilot_certification.json"),
        "created_pilot_generation_json": str(output_dir / "created_pilot_generation.json"),
        "supervisor_drill_dir": str(supervisor_drill_dir),
        "supervisor_drill_json": str(supervisor_drill_dir / "drill_report.json"),
        "watchdog_drill_dir": str(watchdog_drill_dir),
        "watchdog_drill_json": str(watchdog_drill_dir / "watchdog_drill_report.json"),
    }


def _launchctl_domain(wrapper_args: argparse.Namespace) -> str:
    raw = str(getattr(wrapper_args, "launchctl_domain", "") or "").strip()
    if raw:
        return raw
    return f"gui/{os.getuid()}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _status_rank(status: str) -> int:
    return STATUS_RANK.get(str(status or "error").lower(), 2)


def _readiness_status(checks: Sequence[Mapping[str, Any]]) -> str:
    if not checks:
        return "error"
    return max((str(check.get("status") or "error") for check in checks), key=_status_rank)


def _check_status(ok: bool) -> str:
    return "ok" if ok else "error"


def _is_launchagents_plist_path(path: Any) -> bool:
    try:
        parsed = Path(str(path or "")).expanduser().resolve(strict=False)
    except (TypeError, ValueError, OSError):
        return False
    parts = parsed.parts
    return (
        parsed.suffix == ".plist"
        and len(parts) >= 3
        and parts[-2] == "LaunchAgents"
        and parts[-3] == "Library"
    )


def _watchdog_remediation_args(wrapper_args: argparse.Namespace) -> list[str]:
    if not getattr(wrapper_args, "write_launchd_plist", None):
        return []
    max_remediations = max(0, int(getattr(wrapper_args, "watchdog_max_remediations", 0) or 0))
    if max_remediations <= 0:
        return []
    return [
        "--remediate-launchd-label",
        str(wrapper_args.launchd_label),
        "--remediate-launchd-domain",
        _launchctl_domain(wrapper_args),
        "--remediate-launchd-plist",
        str(wrapper_args.write_launchd_plist.expanduser().resolve(strict=False)),
        "--remediate-launchctl",
        str(getattr(wrapper_args, "launchctl", "launchctl") or "launchctl"),
        "--max-remediations",
        str(max_remediations),
        "--remediation-cooldown",
        str(max(0.0, float(getattr(wrapper_args, "watchdog_remediation_cooldown", 300.0) or 0.0))),
        "--remediation-timeout",
        str(
            max(
                1.0,
                float(
                    getattr(
                        wrapper_args,
                        "watchdog_remediation_timeout",
                        DEFAULT_LAUNCHCTL_TIMEOUT_SECONDS,
                    )
                    or DEFAULT_LAUNCHCTL_TIMEOUT_SECONDS
                ),
            )
        ),
    ]


def _add_readiness_check(
    checks: list[dict[str, Any]],
    name: str,
    status: str,
    detail: str,
    **fields: Any,
) -> None:
    checks.append({"name": name, "status": status, "detail": detail, **fields})


def _command_flag_float(command: Sequence[Any], flag: str) -> float | None:
    if flag not in command:
        return None
    try:
        return float(command[command.index(flag) + 1])
    except (IndexError, TypeError, ValueError, OverflowError):
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _certification_qwen_caption_io_source_check_fields(
    certification_report: Mapping[str, Any],
) -> dict[str, Any]:
    source_report = certification_report.get("qwen_caption_io_sources")
    source_report = source_report if isinstance(source_report, Mapping) else {}
    source_counts_raw = source_report.get("source_counts")
    source_counts = source_counts_raw if isinstance(source_counts_raw, Mapping) else {}
    normalized_source_counts = {
        str(source): _safe_int(count, 0)
        for source, count in source_counts.items()
        if str(source)
    }
    accepted_runtime_sources = set(certify.QWEN_CAPTION_IO_ACCEPTED_RUNTIME_SOURCES)
    accepted_observed_sources = [
        source
        for source, count in sorted(normalized_source_counts.items())
        if source in accepted_runtime_sources and count > 0
    ]
    unsupported_sources = [
        source
        for source, count in sorted(normalized_source_counts.items())
        if source not in accepted_runtime_sources and count > 0
    ]
    required_rows = _safe_int(source_report.get("required_rows"), 0)
    return {
        "has_source_report": bool(source_report),
        "required_rows": required_rows,
        "runtime_prompt_budget_rows": _safe_int(source_report.get("runtime_prompt_budget_rows"), 0),
        "valid_runtime_prompt_budget_rows": _safe_int(
            source_report.get("valid_runtime_prompt_budget_rows"),
            0,
        ),
        "invalid_runtime_rows_count": _safe_int(source_report.get("invalid_runtime_rows_count"), 0),
        "missing_runtime_rows_count": _safe_int(source_report.get("missing_runtime_rows_count"), 0),
        "source_counts": normalized_source_counts,
        "accepted_sources": [
            str(source)
            for source in (source_report.get("accepted_sources") or [])
            if str(source)
        ],
        "accepted_observed_sources": accepted_observed_sources,
        "unsupported_sources": unsupported_sources,
    }


def _certification_generated_evidence_fields(
    certification_report: Mapping[str, Any],
    *,
    min_pilot_cases: int,
) -> dict[str, Any]:
    evidence = certification_report.get("generated_case_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    generated_cases = _safe_int(
        certification_report.get("generated_pilot_cases") or evidence.get("generated_cases"),
        0,
    )
    required_cases = max(1, int(min_pilot_cases or certify.DEFAULT_MIN_PILOT_CASES))
    return {
        "generated_pilot_cases": generated_cases,
        "min_pilot_cases": required_cases,
        "has_generated_case_evidence": bool(evidence),
        "latest_cases": _safe_int(evidence.get("latest_cases"), 0),
        "latest_skipped_cases": _safe_int(evidence.get("latest_skipped_cases"), 0),
        "latest_skipped_cases_with_prior_generated_success": _safe_int(
            evidence.get("latest_skipped_cases_with_prior_generated_success"),
            0,
        ),
    }


def _certification_deterministic_reliability_fields(
    certification_report: Mapping[str, Any],
) -> dict[str, Any]:
    reliability = certification_report.get("deterministic_recovery_reliability")
    reliability = reliability if isinstance(reliability, Mapping) else {}
    return {
        "has_reliability_report": bool(reliability),
        "enabled": bool(reliability.get("enabled")),
        "confidence": _safe_float(
            reliability.get("confidence"),
            certify.DEFAULT_DETERMINISTIC_RECOVERY_CONFIDENCE,
        ),
        "limit": _safe_float(reliability.get("limit"), 0.0),
        "generated_cases": _safe_int(reliability.get("generated_cases"), 0),
        "deterministic_recovery_cases": _safe_int(
            reliability.get("deterministic_recovery_cases"),
            0,
        ),
        "observed_rate": _safe_float(reliability.get("observed_rate"), 0.0),
        "upper_bound_rate": _safe_float(reliability.get("upper_bound_rate"), 1.0),
        "required_zero_recovery_cases": _safe_int(
            reliability.get("required_zero_recovery_cases"),
            0,
        ),
    }


def _post_install_operation_audit_timeout(wrapper_args: argparse.Namespace) -> float:
    return max(
        0.0,
        _safe_float(
            getattr(
                wrapper_args,
                "post_install_operation_audit_timeout",
                DEFAULT_POST_INSTALL_OPERATION_AUDIT_TIMEOUT_SECONDS,
            ),
            DEFAULT_POST_INSTALL_OPERATION_AUDIT_TIMEOUT_SECONDS,
        ),
    )


def _post_install_operation_audit_interval(wrapper_args: argparse.Namespace) -> float:
    return max(
        0.1,
        _safe_float(
            getattr(
                wrapper_args,
                "post_install_operation_audit_interval",
                DEFAULT_POST_INSTALL_OPERATION_AUDIT_INTERVAL_SECONDS,
            ),
            DEFAULT_POST_INSTALL_OPERATION_AUDIT_INTERVAL_SECONDS,
        ),
    )


def _post_install_operation_audit_required(wrapper_args: argparse.Namespace) -> bool:
    return (
        bool(getattr(wrapper_args, "install_launchd_plists", False))
        and bool(getattr(wrapper_args, "require_readiness_ok", False))
        and not bool(getattr(wrapper_args, "skip_post_install_operation_audit", False))
    )


def _command_flag_value(command: Sequence[Any], flag: str) -> str | None:
    if flag not in command:
        return None
    try:
        return str(command[command.index(flag) + 1])
    except (IndexError, TypeError, ValueError):
        return None


def _launchd_program_arguments(command: Sequence[str], *, caffeinate: bool = False) -> list[str]:
    base_command = [str(part) for part in command]
    if not caffeinate:
        return base_command
    return [CAFFEINATE_BIN, *CAFFEINATE_ARGS, *base_command]


def _checks_by_name(report: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    checks = (report or {}).get("checks")
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
        return {}
    by_name: dict[str, Mapping[str, Any]] = {}
    for item in checks:
        if isinstance(item, Mapping):
            name = str(item.get("name") or "").strip()
            if name:
                by_name[name] = item
    return by_name


def build_readiness_report(
    *,
    plan: Mapping[str, Any],
    wrapper_args: argparse.Namespace,
    preflight_report: Mapping[str, Any] | None,
    supervisor_drill_report: Mapping[str, Any] | None,
    pilot_generation_report: Mapping[str, Any] | None,
    certification_report: Mapping[str, Any] | None,
    launch_blocked: bool,
    watchdog_drill_report: Mapping[str, Any] | None = None,
    live_adoption_report: Mapping[str, Any] | None = None,
    static_gate_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    output_dir = str(plan.get("output_dir") or "")
    commands = plan.get("commands") if isinstance(plan.get("commands"), Mapping) else {}
    paths = plan.get("paths") if isinstance(plan.get("paths"), Mapping) else {}
    launchd_install = plan.get("launchd_install") if isinstance(plan.get("launchd_install"), Mapping) else {}
    launchd_roles = launchd_install.get("roles") if isinstance(launchd_install.get("roles"), Mapping) else {}
    drill_gate = plan.get("supervisor_drill_gate") if isinstance(plan.get("supervisor_drill_gate"), Mapping) else {}
    watchdog_drill_gate = (
        plan.get("watchdog_drill_gate") if isinstance(plan.get("watchdog_drill_gate"), Mapping) else {}
    )
    set_and_forget_gate = (
        plan.get("set_and_forget_gate") if isinstance(plan.get("set_and_forget_gate"), Mapping) else {}
    )
    power_assertion = (
        plan.get("launchd_power_assertion")
        if isinstance(plan.get("launchd_power_assertion"), Mapping)
        else {}
    )
    tenk_mode = bool(getattr(wrapper_args, "tenk_set_and_forget", False) or set_and_forget_gate.get("tenk_mode"))
    pilot_generation_gate = (
        plan.get("pilot_generation_gate") if isinstance(plan.get("pilot_generation_gate"), Mapping) else {}
    )
    cert_gate = plan.get("certification_gate") if isinstance(plan.get("certification_gate"), Mapping) else {}
    live_adoption_status = str((live_adoption_report or {}).get("status") or "missing").lower()
    live_adoption_ok = live_adoption_status == "ok"
    static_gate_status = str((static_gate_report or {}).get("status") or "ok").lower()

    if static_gate_report is not None:
        _add_readiness_check(
            checks,
            "tenk_static_launch_gate",
            "ok" if static_gate_status == "ok" else "error",
            (
                "static 10k set-and-forget launch gates passed"
                if static_gate_status == "ok"
                else "static 10k set-and-forget launch gates blocked before GPU work"
            ),
        )

    if preflight_report is None:
        _add_readiness_check(
            checks,
            "preflight",
            "error" if tenk_mode else "warn",
            (
                "10k set-and-forget readiness requires wrapper preflight before handoff"
                if tenk_mode
                else "wrapper preflight was skipped; only supervisor-time preflight remains"
            ),
        )
    else:
        preflight_status = str(preflight_report.get("status") or "error").lower()
        if preflight_status == "error":
            if _preflight_error_can_defer_to_supervisor(preflight_report):
                preflight_deferred_status = "ok" if live_adoption_ok else "warn"
                _add_readiness_check(
                    checks,
                    "preflight",
                    preflight_deferred_status,
                    (
                        "wrapper preflight found a live runner lock; launchd supervisor will adopt and wait"
                        if live_adoption_ok
                        else "wrapper preflight found a live runner lock and deferred waiting to the supervisor"
                    ),
                )
            else:
                _add_readiness_check(
                    checks,
                    "preflight",
                    "error",
                    "wrapper preflight has a blocking error",
                )
        elif preflight_status == "warn":
            _add_readiness_check(checks, "preflight", "warn", "wrapper preflight completed with warnings")
        else:
            _add_readiness_check(checks, "preflight", "ok", "wrapper preflight completed cleanly")

        if bool(getattr(wrapper_args, "adopt_live_run", False)):
            if live_adoption_report is None:
                _add_readiness_check(
                    checks,
                    "live_adoption",
                    "warn",
                    "--adopt-live-run was requested but wrapper preflight did not find a live runner lock",
                )
            elif live_adoption_ok:
                _add_readiness_check(
                    checks,
                    "live_adoption",
                    "ok",
                    "active run passed live adoption certification for launchd handoff",
                )
            else:
                _add_readiness_check(
                    checks,
                    "live_adoption",
                    "error",
                    f"active run adoption certification status is {live_adoption_status}",
                )

        preflight_checks = _checks_by_name(preflight_report)
        disk_check = preflight_checks.get("disk_budget")
        if disk_check:
            _add_readiness_check(
                checks,
                "disk_budget",
                str(disk_check.get("status") or "error"),
                str(disk_check.get("detail") or "disk budget check completed"),
            )
        model_cache = preflight_report.get("model_cache")
        if isinstance(model_cache, Mapping):
            models = model_cache.get("models") if isinstance(model_cache.get("models"), Sequence) else []
            missing = [
                item
                for item in models
                if isinstance(item, Mapping) and bool(item.get("needs_download"))
            ]
            if missing:
                _add_readiness_check(
                    checks,
                    "model_downloads",
                    "error" if tenk_mode else "warn",
                    (
                        "10k set-and-forget readiness requires selected caption models to be downloaded before handoff"
                        if tenk_mode
                        else "one or more selected caption models still need downloading"
                    ),
                    models=[str(item.get("model_id") or "") for item in missing],
                )
            else:
                _add_readiness_check(
                    checks,
                    "model_downloads",
                    "ok",
                    "selected concrete caption models are already cached or no concrete model was required",
                )

    drill_status = str((supervisor_drill_report or {}).get("status") or "missing").lower()
    drill_required = bool(drill_gate.get("required", True))
    if supervisor_drill_report is None:
        _add_readiness_check(checks, "supervisor_drill", "error", "supervisor drill report is missing")
    elif drill_status == "ok":
        drill_checks = supervisor_drill_report.get("checks")
        false_checks = [
            key
            for key, value in (drill_checks.items() if isinstance(drill_checks, Mapping) else [])
            if not bool(value)
        ]
        required_checks = {
            "supervisor_success",
            "saw_nonzero_exit",
            "saw_signal_exit",
            "saw_stale_heartbeat",
            "saw_missing_heartbeat",
            "caption_io_retention_ok",
            "summary_totals_cover_all_cases",
            "summary_snapshot_bounded",
            "summary_truncated_when_over_limit",
            "summary_omits_rows_when_over_limit",
        }
        missing_checks = (
            sorted(required_checks - {str(key) for key in drill_checks.keys()})
            if isinstance(drill_checks, Mapping)
            else sorted(required_checks)
        )
        if false_checks:
            _add_readiness_check(
                checks,
                "supervisor_drill",
                "error",
                "supervisor drill status is ok but one or more internal checks failed",
                false_checks=false_checks,
            )
        elif missing_checks:
            _add_readiness_check(
                checks,
                "supervisor_drill",
                "error",
                "supervisor drill status is ok but required restart checks are missing",
                missing_checks=missing_checks,
            )
        else:
            _add_readiness_check(checks, "supervisor_drill", "ok", "supervisor restart drill passed")
    elif drill_status == "skipped" and not drill_required:
        _add_readiness_check(
            checks,
            "supervisor_drill",
            "error" if tenk_mode else "warn",
            (
                "10k set-and-forget readiness requires the supervisor restart drill"
                if tenk_mode
                else "supervisor drill was explicitly skipped; restart behavior is not proven for this launch"
            ),
        )
    else:
        _add_readiness_check(
            checks,
            "supervisor_drill",
            "error",
            f"supervisor drill status is {drill_status}",
        )

    drill_case_count = int(
        (supervisor_drill_report or {}).get("case_count")
        or drill_gate.get("case_count")
        or 0
    )
    if drill_status == "ok" and drill_case_count >= SET_AND_FORGET_DRILL_CASES:
        _add_readiness_check(
            checks,
            "supervisor_drill_scale",
            "ok",
            f"supervisor drill covered {drill_case_count} synthetic cases",
            case_count=drill_case_count,
        )
    elif drill_status == "ok":
        _add_readiness_check(
            checks,
            "supervisor_drill_scale",
            "error" if tenk_mode else "warn",
            (
                f"supervisor drill covered {drill_case_count} synthetic cases; "
                f"{SET_AND_FORGET_DRILL_CASES} is the default set-and-forget gate"
            ),
            case_count=drill_case_count,
            recommended_case_count=SET_AND_FORGET_DRILL_CASES,
        )

    watchdog_drill_status = str((watchdog_drill_report or {}).get("status") or "missing").lower()
    watchdog_drill_required = bool(watchdog_drill_gate.get("required", True))
    if watchdog_drill_report is None:
        _add_readiness_check(checks, "watchdog_drill", "error", "watchdog remediation drill report is missing")
    elif watchdog_drill_status == "ok":
        watchdog_checks = watchdog_drill_report.get("checks")
        false_checks = [
            key
            for key, value in (watchdog_checks.items() if isinstance(watchdog_checks, Mapping) else [])
            if not bool(value)
        ]
        required_watchdog_checks = {
            "watchdog_success",
            "saw_unhealthy_status",
            "saw_launchd_rebootstrap",
            "saw_bootstrap_success",
            "saw_restored_health",
            "latest_status_ok",
            "state_persisted",
        }
        missing_watchdog_checks = (
            sorted(required_watchdog_checks - {str(key) for key in watchdog_checks.keys()})
            if isinstance(watchdog_checks, Mapping)
            else sorted(required_watchdog_checks)
        )
        if false_checks:
            _add_readiness_check(
                checks,
                "watchdog_drill",
                "error",
                "watchdog drill status is ok but one or more internal checks failed",
                false_checks=false_checks,
            )
        elif missing_watchdog_checks:
            _add_readiness_check(
                checks,
                "watchdog_drill",
                "error",
                "watchdog drill status is ok but required remediation checks are missing",
                missing_checks=missing_watchdog_checks,
            )
        else:
            _add_readiness_check(
                checks,
                "watchdog_drill",
                "ok",
                "watchdog launchd-remediation drill passed",
            )
    elif watchdog_drill_status == "skipped" and not watchdog_drill_required:
        _add_readiness_check(
            checks,
            "watchdog_drill",
            "error" if tenk_mode else "warn",
            (
                "10k set-and-forget readiness requires the watchdog launchd-remediation drill"
                if tenk_mode
                else "watchdog drill was explicitly skipped; automatic launchd repair is not proven for this launch"
            ),
        )
    else:
        _add_readiness_check(
            checks,
            "watchdog_drill",
            "error",
            f"watchdog drill status is {watchdog_drill_status}",
        )

    pilot_generation_required = bool(pilot_generation_gate.get("required"))
    pilot_generation_status = str((pilot_generation_report or {}).get("status") or "missing").lower()
    if pilot_generation_required:
        if pilot_generation_report is None:
            _add_readiness_check(checks, "pilot_generation", "error", "required pilot generation report is missing")
        elif pilot_generation_status == "ok":
            _add_readiness_check(checks, "pilot_generation", "ok", "pilot sample generation completed")
        elif pilot_generation_status == "skipped":
            _add_readiness_check(
                checks,
                "pilot_generation",
                "error" if tenk_mode else "warn",
                (
                    "10k set-and-forget readiness cannot use a dry-run-skipped generated pilot"
                    if tenk_mode
                    else "pilot generation was skipped; dry-run cannot prove generated pilot readiness"
                ),
            )
        else:
            _add_readiness_check(
                checks,
                "pilot_generation",
                "error",
                f"pilot generation status is {pilot_generation_status}",
            )

    cert_required = bool(cert_gate.get("required"))
    cert_status = str((certification_report or {}).get("status") or "missing").lower()
    if not cert_required:
        if live_adoption_ok:
            live_audit = (
                live_adoption_report.get("live_audit")
                if isinstance(live_adoption_report.get("live_audit"), Mapping)
                else {}
            )
            _add_readiness_check(
                checks,
                "pilot_certification",
                "ok",
                "live adoption certification supplies runtime and prompt-budget evidence from the active run",
                processed_cases=live_audit.get("processed_cases"),
                expected_cases=live_audit.get("expected_cases"),
            )
        else:
            _add_readiness_check(
                checks,
                "pilot_certification",
                "error" if tenk_mode else "warn",
                (
                    "10k set-and-forget readiness requires pilot certification or clean live-run adoption evidence"
                    if tenk_mode
                    else "pilot certification was not required; 10k runtime and prompt-budget readiness are unproven"
                ),
            )
    elif certification_report is None:
        _add_readiness_check(checks, "pilot_certification", "error", "required pilot certification report is missing")
    elif cert_status == "ok":
        _add_readiness_check(checks, "pilot_certification", "ok", "required pilot certification passed")
    else:
        _add_readiness_check(
            checks,
            "pilot_certification",
            "error",
            f"required pilot certification status is {cert_status}",
        )

    if cert_required:
        min_pilot_cases = max(1, _safe_int(cert_gate.get("min_pilot_cases"), certify.DEFAULT_MIN_PILOT_CASES))
        if certification_report is None:
            _add_readiness_check(
                checks,
                "pilot_generated_evidence",
                "error",
                "required pilot certification report is missing generated model evidence",
                min_pilot_cases=min_pilot_cases,
            )
        elif cert_status != "ok":
            _add_readiness_check(
                checks,
                "pilot_generated_evidence",
                "error",
                "required pilot certification did not pass generated model evidence checks",
                min_pilot_cases=min_pilot_cases,
            )
        else:
            generated_evidence = _certification_generated_evidence_fields(
                certification_report,
                min_pilot_cases=min_pilot_cases,
            )
            if int(generated_evidence["generated_pilot_cases"]) >= int(
                generated_evidence["min_pilot_cases"]
            ):
                _add_readiness_check(
                    checks,
                    "pilot_generated_evidence",
                    "ok",
                    "passed pilot certification includes enough generated model-success evidence",
                    **generated_evidence,
                )
            else:
                _add_readiness_check(
                    checks,
                    "pilot_generated_evidence",
                    "error",
                    "passed pilot certification lacks enough generated model-success evidence",
                    **generated_evidence,
                )

        if tenk_mode:
            if certification_report is None:
                _add_readiness_check(
                    checks,
                    "pilot_deterministic_recovery_confidence",
                    "error",
                    "required pilot certification report is missing deterministic-recovery reliability evidence",
                )
            elif cert_status != "ok":
                _add_readiness_check(
                    checks,
                    "pilot_deterministic_recovery_confidence",
                    "error",
                    "required pilot certification did not pass deterministic-recovery confidence checks",
                )
            else:
                reliability = _certification_deterministic_reliability_fields(certification_report)
                confidence_ok = (
                    bool(reliability["has_reliability_report"])
                    and bool(reliability["enabled"])
                    and float(reliability["upper_bound_rate"]) <= float(reliability["limit"])
                )
                if confidence_ok:
                    _add_readiness_check(
                        checks,
                        "pilot_deterministic_recovery_confidence",
                        "ok",
                        "passed pilot certification bounds deterministic-recovery rate for set-and-forget",
                        **reliability,
                    )
                else:
                    _add_readiness_check(
                        checks,
                        "pilot_deterministic_recovery_confidence",
                        "error",
                        "passed pilot certification does not bound deterministic-recovery rate tightly enough",
                        **reliability,
                    )

        required_capabilities = [
            str(item)
            for item in (cert_gate.get("required_runner_capabilities") or runner.RUNNER_CAPABILITIES)
            if str(item)
        ]
        if not bool(cert_gate.get("require_runner_capabilities", True)):
            _add_readiness_check(
                checks,
                "pilot_runner_capabilities",
                "error" if tenk_mode else "warn",
                (
                    "10k set-and-forget readiness requires current runner capability evidence"
                    if tenk_mode
                    else "pilot runner-capability requirement is disabled"
                ),
                required_capabilities=required_capabilities,
            )
        elif certification_report is None:
            _add_readiness_check(
                checks,
                "pilot_runner_capabilities",
                "error",
                "required pilot certification report is missing runner-capability evidence",
                required_capabilities=required_capabilities,
            )
        elif cert_status != "ok":
            _add_readiness_check(
                checks,
                "pilot_runner_capabilities",
                "error",
                "required pilot certification did not pass runner-capability checks",
                required_capabilities=required_capabilities,
            )
        else:
            capability_report = certification_report.get("runner_capabilities")
            capability_report = capability_report if isinstance(capability_report, Mapping) else {}
            pilot_capabilities = [
                str(item)
                for item in (capability_report.get("runner_capabilities") or [])
                if str(item)
            ]
            reported_missing = [
                str(item)
                for item in (capability_report.get("missing_capabilities") or [])
                if str(item)
            ]
            effective_missing = [
                capability
                for capability in required_capabilities
                if capability not in set(pilot_capabilities)
            ]
            missing_capabilities = reported_missing or effective_missing
            if missing_capabilities:
                _add_readiness_check(
                    checks,
                    "pilot_runner_capabilities",
                    "error",
                    "passed pilot certification lacks current runner recovery and stream-loop telemetry capability proof",
                    required_capabilities=required_capabilities,
                    runner_capabilities=pilot_capabilities,
                    missing_capabilities=missing_capabilities,
                )
            else:
                _add_readiness_check(
                    checks,
                    "pilot_runner_capabilities",
                    "ok",
                    "passed pilot certification includes current runner recovery and stream-loop telemetry capability proof",
                    required_capabilities=required_capabilities,
                    runner_capabilities=pilot_capabilities,
                    capability_sources=capability_report.get("capability_sources") or [],
                )

    if cert_required:
        prompt_budget_required = bool(cert_gate.get("require_prompt_budget_data", True))
        if not prompt_budget_required:
            _add_readiness_check(
                checks,
                "pilot_prompt_budget_evidence",
                "error" if tenk_mode else "warn",
                (
                    "10k set-and-forget readiness requires pilot prompt-budget telemetry"
                    if tenk_mode
                    else "pilot prompt-budget telemetry requirement is disabled"
                ),
            )
        elif certification_report is None:
            _add_readiness_check(
                checks,
                "pilot_prompt_budget_evidence",
                "error",
                "required pilot certification report is missing prompt-budget evidence",
            )
        elif cert_status != "ok":
            _add_readiness_check(
                checks,
                "pilot_prompt_budget_evidence",
                "error",
                "required pilot certification did not pass prompt-budget checks",
            )
        else:
            prompt_budget_report = certification_report.get("prompt_budget")
            prompt_budget_report = prompt_budget_report if isinstance(prompt_budget_report, Mapping) else {}
            pilot_cases = _safe_int(
                prompt_budget_report.get("pilot_cases") or certification_report.get("pilot_cases"),
                0,
            )
            rows_with_budget = _safe_int(prompt_budget_report.get("rows_with_prompt_budget"), 0)
            observed_max_prompt_tokens = _safe_int(prompt_budget_report.get("max_prompt_tokens"), 0)
            adapted_case_rate = _safe_float(prompt_budget_report.get("adapted_case_rate"), 0.0)
            prompt_token_limit = max(0, _safe_int(cert_gate.get("max_prompt_tokens"), 0))
            adapted_rate_limit = _safe_float(
                cert_gate.get(
                    "max_prompt_budget_adapted_case_rate",
                    certify.DEFAULT_MAX_PROMPT_BUDGET_ADAPTED_CASE_RATE,
                ),
                certify.DEFAULT_MAX_PROMPT_BUDGET_ADAPTED_CASE_RATE,
            )
            if pilot_cases <= 0 or rows_with_budget != pilot_cases:
                _add_readiness_check(
                    checks,
                    "pilot_prompt_budget_evidence",
                    "error",
                    "passed pilot certification lacks complete prompt-budget telemetry",
                    pilot_cases=pilot_cases,
                    rows_with_prompt_budget=rows_with_budget,
                )
            elif prompt_token_limit > 0 and observed_max_prompt_tokens > prompt_token_limit:
                _add_readiness_check(
                    checks,
                    "pilot_prompt_budget_evidence",
                    "error",
                    "passed pilot certification exceeds the configured prompt-size ceiling",
                    max_prompt_tokens=observed_max_prompt_tokens,
                    limit=prompt_token_limit,
                    pilot_cases=pilot_cases,
                    rows_with_prompt_budget=rows_with_budget,
                )
            elif adapted_rate_limit >= 0 and adapted_case_rate > adapted_rate_limit:
                _add_readiness_check(
                    checks,
                    "pilot_prompt_budget_evidence",
                    "error",
                    "passed pilot certification exceeds the configured prompt-budget adaptation-rate ceiling",
                    adapted_case_rate=adapted_case_rate,
                    limit=adapted_rate_limit,
                    pilot_cases=pilot_cases,
                    rows_with_prompt_budget=rows_with_budget,
                )
            else:
                _add_readiness_check(
                    checks,
                    "pilot_prompt_budget_evidence",
                    "ok",
                    "passed pilot certification includes complete prompt-budget telemetry within launch gates",
                    pilot_cases=pilot_cases,
                    rows_with_prompt_budget=rows_with_budget,
                    max_prompt_tokens=observed_max_prompt_tokens,
                    prompt_token_limit=prompt_token_limit,
                    adapted_case_rate=adapted_case_rate,
                    adapted_rate_limit=adapted_rate_limit,
                )

        if prompt_budget_required:
            if certification_report is None:
                _add_readiness_check(
                    checks,
                    "pilot_qwen_caption_io_source",
                    "error",
                    "required pilot certification report is missing run-bound Qwen caption IO source evidence",
                )
            elif cert_status != "ok":
                _add_readiness_check(
                    checks,
                    "pilot_qwen_caption_io_source",
                    "error",
                    "required pilot certification did not pass run-bound Qwen caption IO source checks",
                )
            else:
                qwen_io_source = _certification_qwen_caption_io_source_check_fields(certification_report)
                source_rows_ok = (
                    bool(qwen_io_source["has_source_report"])
                    and int(qwen_io_source["required_rows"]) > 0
                    and int(qwen_io_source["valid_runtime_prompt_budget_rows"])
                    >= int(qwen_io_source["required_rows"])
                    and int(qwen_io_source["invalid_runtime_rows_count"]) == 0
                    and int(qwen_io_source["missing_runtime_rows_count"]) == 0
                    and not qwen_io_source["unsupported_sources"]
                    and bool(qwen_io_source["accepted_observed_sources"])
                )
                if source_rows_ok:
                    _add_readiness_check(
                        checks,
                        "pilot_qwen_caption_io_source",
                        "ok",
                        "passed pilot certification uses run-bound Qwen caption IO prompt-budget evidence",
                        **qwen_io_source,
                    )
                else:
                    _add_readiness_check(
                        checks,
                        "pilot_qwen_caption_io_source",
                        "error",
                        "passed pilot certification lacks complete run-bound Qwen caption IO prompt-budget evidence",
                        **qwen_io_source,
                    )

    target_cases = int(cert_gate.get("target_cases") or 0)
    if cert_required and target_cases >= certify.DEFAULT_TARGET_CASES:
        _add_readiness_check(
            checks,
            "target_scale",
            "ok",
            f"pilot certification targets {target_cases} cases",
            target_cases=target_cases,
        )
    elif cert_required:
        _add_readiness_check(
            checks,
            "target_scale",
            "error" if tenk_mode else "warn",
            f"pilot certification target {target_cases} is below the 10k readiness target",
            target_cases=target_cases,
            recommended_target_cases=certify.DEFAULT_TARGET_CASES,
        )

    if bool(cert_gate.get("require_prompt_budget_data", True)):
        _add_readiness_check(checks, "prompt_budget_telemetry", "ok", "pilot certification requires prompt-budget telemetry")
    else:
        _add_readiness_check(
            checks,
            "prompt_budget_telemetry",
            "error" if tenk_mode else "warn",
            (
                "10k set-and-forget readiness requires pilot prompt-budget telemetry"
                if tenk_mode
                else "pilot prompt-budget telemetry requirement is disabled"
            ),
        )
    if int(cert_gate.get("max_prompt_tokens") or 0) > 0:
        _add_readiness_check(
            checks,
            "prompt_size_ceiling",
            "ok",
            "pilot certification has a maximum prompt-size gate",
            max_prompt_tokens=int(cert_gate.get("max_prompt_tokens") or 0),
        )
    else:
        _add_readiness_check(
            checks,
            "prompt_size_ceiling",
            "error" if tenk_mode else "warn",
            (
                "10k set-and-forget readiness requires a positive pilot prompt-size ceiling"
                if tenk_mode
                else "pilot certification max prompt-size gate is disabled"
            ),
        )

    required_commands = [
        "supervisor",
        "watchdog",
        "live_status",
        "operational_audit",
        "final_audit",
        "supervisor_drill",
        "watchdog_drill",
        "pilot_certification",
    ]
    if pilot_generation_required:
        required_commands.append("pilot_generation")
    missing_commands = [name for name in required_commands if not commands.get(name)]
    if missing_commands:
        _add_readiness_check(
            checks,
            "recovery_commands",
            "error",
            "runbook is missing one or more recovery commands",
            missing_commands=missing_commands,
        )
    else:
        _add_readiness_check(
            checks,
            "recovery_commands",
            "ok",
            "runbook includes supervisor, watchdog, live status, operational audit, final audit, drills, and pilot certification commands",
        )

    raw_watchdog_command = commands.get("watchdog")
    watchdog_command = (
        raw_watchdog_command
        if isinstance(raw_watchdog_command, Sequence) and not isinstance(raw_watchdog_command, (str, bytes))
        else []
    )
    raw_final_audit_command = commands.get("final_audit")
    final_audit_command = (
        raw_final_audit_command
        if isinstance(raw_final_audit_command, Sequence) and not isinstance(raw_final_audit_command, (str, bytes))
        else []
    )
    raw_live_status_command = commands.get("live_status")
    live_status_command = (
        raw_live_status_command
        if isinstance(raw_live_status_command, Sequence) and not isinstance(raw_live_status_command, (str, bytes))
        else []
    )
    watchdog_latest_path = paths.get("watchdog_latest_json")
    watchdog_state_path = paths.get("watchdog_state_json")
    watchdog_latest_arg = _command_flag_value(watchdog_command, "--latest-json")
    watchdog_state_arg = _command_flag_value(watchdog_command, "--state-json")
    missing_watchdog_artifacts = []
    if not watchdog_latest_path:
        missing_watchdog_artifacts.append("watchdog_latest_json")
    if not watchdog_state_path:
        missing_watchdog_artifacts.append("watchdog_state_json")
    missing_watchdog_flags = []
    if not watchdog_latest_arg:
        missing_watchdog_flags.append("--latest-json")
    if not watchdog_state_arg:
        missing_watchdog_flags.append("--state-json")
    mismatched_watchdog_flags = []
    if watchdog_latest_arg and watchdog_latest_path and str(watchdog_latest_arg) != str(watchdog_latest_path):
        mismatched_watchdog_flags.append("--latest-json")
    if watchdog_state_arg and watchdog_state_path and str(watchdog_state_arg) != str(watchdog_state_path):
        mismatched_watchdog_flags.append("--state-json")
    if missing_watchdog_artifacts or missing_watchdog_flags or mismatched_watchdog_flags:
        _add_readiness_check(
            checks,
            "watchdog_status_artifacts",
            "error",
            "watchdog command must record explicit latest-status and durable state artifacts",
            missing_paths=missing_watchdog_artifacts,
            missing_flags=missing_watchdog_flags,
            mismatched_flags=mismatched_watchdog_flags,
            watchdog_latest_json=watchdog_latest_path,
            watchdog_state_json=watchdog_state_path,
        )
    else:
        _add_readiness_check(
            checks,
            "watchdog_status_artifacts",
            "ok",
            "watchdog command records latest-status and restart-stable state artifacts",
            watchdog_latest_json=str(watchdog_latest_path),
            watchdog_state_json=str(watchdog_state_path),
        )
    watchdog_projected_hours = _command_flag_float(watchdog_command, "--max-projected-duration-hours")
    final_audit_projected_hours = _command_flag_float(final_audit_command, "--max-projected-duration-hours")
    graceful_restart_timeout = _command_flag_float(watchdog_command, "--graceful-restart-timeout")
    if graceful_restart_timeout is not None and graceful_restart_timeout > 0:
        _add_readiness_check(
            checks,
            "watchdog_graceful_restart",
            "ok",
            "watchdog command explicitly enables cooperative runner restart requests before launchd escalation",
            graceful_restart_timeout_seconds=graceful_restart_timeout,
        )
    else:
        _add_readiness_check(
            checks,
            "watchdog_graceful_restart",
            "error" if tenk_mode else "warn",
            "watchdog cooperative runner restart requests are disabled or not recorded",
            graceful_restart_timeout_seconds=graceful_restart_timeout,
        )
    if tenk_mode:
        missing_or_disabled = []
        if watchdog_projected_hours is None or watchdog_projected_hours <= 0:
            missing_or_disabled.append("watchdog")
        if final_audit_projected_hours is None or final_audit_projected_hours <= 0:
            missing_or_disabled.append("final_audit")
        if missing_or_disabled:
            _add_readiness_check(
                checks,
                "projected_duration_gate",
                "error",
                "10k set-and-forget readiness requires a positive projected-duration gate",
                missing_or_disabled=missing_or_disabled,
                watchdog_max_projected_duration_hours=watchdog_projected_hours,
                final_audit_max_projected_duration_hours=final_audit_projected_hours,
            )
        else:
            _add_readiness_check(
                checks,
                "projected_duration_gate",
                "ok",
                "watchdog and final audit enforce projected all-case wall-clock duration",
                watchdog_max_projected_duration_hours=watchdog_projected_hours,
                final_audit_max_projected_duration_hours=final_audit_projected_hours,
            )

    supervisor_plan_args = (
        plan.get("supervisor_args") if isinstance(plan.get("supervisor_args"), Mapping) else {}
    )
    configured_min_free_gb = _command_flag_float(watchdog_command, "--min-free-gb")
    try:
        requested_min_free_gb = float(supervisor_plan_args.get("min_free_gb") or 0.0)
    except (TypeError, ValueError, OverflowError):
        requested_min_free_gb = 0.0
    disk_gate_values = {
        "watchdog": configured_min_free_gb,
        "final_audit": _command_flag_float(final_audit_command, "--min-free-gb"),
        "live_status": _command_flag_float(live_status_command, "--min-free-gb"),
    }
    missing_disk_gate = [name for name, value in disk_gate_values.items() if value is None]
    disabled_disk_gate = [
        name for name, value in disk_gate_values.items()
        if value is not None and value <= 0
    ]
    mismatched_disk_gate = [
        name for name, value in disk_gate_values.items()
        if value is not None and requested_min_free_gb > 0 and abs(value - requested_min_free_gb) > 1e-9
    ]
    if requested_min_free_gb <= 0 and tenk_mode:
        _add_readiness_check(
            checks,
            "disk_reserve_gate",
            "error",
            "10k set-and-forget readiness should keep a positive live disk-reserve gate",
            requested_min_free_gb=requested_min_free_gb,
            command_min_free_gb=disk_gate_values,
        )
    elif missing_disk_gate or mismatched_disk_gate or (tenk_mode and disabled_disk_gate):
        _add_readiness_check(
            checks,
            "disk_reserve_gate",
            "error" if tenk_mode else "warn",
            "watchdog, live status, and final audit must share the configured live disk-reserve gate",
            requested_min_free_gb=requested_min_free_gb,
            command_min_free_gb=disk_gate_values,
            missing_commands=missing_disk_gate,
            disabled_commands=disabled_disk_gate,
            mismatched_commands=mismatched_disk_gate,
        )
    elif requested_min_free_gb > 0:
        _add_readiness_check(
            checks,
            "disk_reserve_gate",
            "ok",
            "watchdog, live status, and final audit enforce the live disk-reserve gate",
            requested_min_free_gb=requested_min_free_gb,
            command_min_free_gb=disk_gate_values,
        )
    else:
        _add_readiness_check(
            checks,
            "disk_reserve_gate",
            "ok",
            "live disk-reserve gate is optional outside strict 10k set-and-forget mode",
            requested_min_free_gb=requested_min_free_gb,
            command_min_free_gb=disk_gate_values,
        )

    if wrapper_args.write_launchd_plist:
        _add_readiness_check(
            checks,
            "launchd_restart",
            "ok",
            "LaunchAgent plist output was requested for unattended restart by macOS",
            plist_path=str(wrapper_args.write_launchd_plist.expanduser().resolve(strict=False)),
        )
    else:
        _add_readiness_check(
            checks,
            "launchd_restart",
            "error" if tenk_mode else "warn",
            (
                "10k set-and-forget readiness requires a supervisor LaunchAgent plist"
                if tenk_mode
                else "no LaunchAgent plist was requested; direct shell launch is less suitable for a two-week unattended run"
            ),
        )
    power_assertion_enabled = bool(power_assertion.get("enabled"))
    if tenk_mode and power_assertion_enabled:
        _add_readiness_check(
            checks,
            "launchd_power_assertion",
            "ok",
            "generated LaunchAgent commands prevent macOS sleep with caffeinate",
            program=str(power_assertion.get("program") or CAFFEINATE_BIN),
            arguments=list(power_assertion.get("arguments") or []),
        )
    elif tenk_mode:
        _add_readiness_check(
            checks,
            "launchd_power_assertion",
            "error",
            "10k set-and-forget readiness requires macOS sleep prevention while launchd agents run",
            program=str(power_assertion.get("program") or CAFFEINATE_BIN),
            arguments=list(power_assertion.get("arguments") or CAFFEINATE_ARGS),
        )
    elif power_assertion_enabled:
        _add_readiness_check(
            checks,
            "launchd_power_assertion",
            "ok",
            "generated LaunchAgent commands prevent macOS sleep with caffeinate",
            program=str(power_assertion.get("program") or CAFFEINATE_BIN),
            arguments=list(power_assertion.get("arguments") or []),
        )
    else:
        _add_readiness_check(
            checks,
            "launchd_power_assertion",
            "ok",
            "macOS sleep prevention is optional outside strict 10k set-and-forget mode",
        )
    if wrapper_args.write_watchdog_launchd_plist:
        _add_readiness_check(
            checks,
            "watchdog_launchd",
            "ok",
            "watchdog LaunchAgent plist output was requested for independent artifact health monitoring",
            plist_path=str(wrapper_args.write_watchdog_launchd_plist.expanduser().resolve(strict=False)),
        )
    else:
        _add_readiness_check(
            checks,
            "watchdog_launchd",
            "error" if tenk_mode else "warn",
            (
                "10k set-and-forget readiness requires a watchdog LaunchAgent plist"
                if tenk_mode
                else "no watchdog LaunchAgent plist was requested; health monitoring remains a manual side process"
            ),
        )

    has_remediation = (
        "--remediate-launchd-label" in watchdog_command
        and "--remediate-launchd-plist" in watchdog_command
        and "--max-remediations" in watchdog_command
    )
    if wrapper_args.write_launchd_plist and wrapper_args.write_watchdog_launchd_plist and has_remediation:
        _add_readiness_check(
            checks,
            "watchdog_remediation",
            "ok",
            "watchdog command can kickstart or rebootstrap the supervisor LaunchAgent after repeated unhealthy checks",
            supervisor_label=str(wrapper_args.launchd_label),
            domain=_launchctl_domain(wrapper_args),
            plist_path=str(wrapper_args.write_launchd_plist.expanduser().resolve(strict=False)),
            max_remediations=int(getattr(wrapper_args, "watchdog_max_remediations", 0) or 0),
        )
    elif wrapper_args.write_launchd_plist and wrapper_args.write_watchdog_launchd_plist:
        _add_readiness_check(
            checks,
            "watchdog_remediation",
            "error" if tenk_mode else "warn",
            (
                "10k set-and-forget readiness requires watchdog automatic supervisor kickstart"
                if tenk_mode
                else "watchdog automatic supervisor kickstart is disabled"
            ),
            supervisor_label=str(wrapper_args.launchd_label),
            max_remediations=int(getattr(wrapper_args, "watchdog_max_remediations", 0) or 0),
        )
    else:
        _add_readiness_check(
            checks,
            "watchdog_remediation",
            "error" if tenk_mode else "warn",
            (
                "10k set-and-forget readiness requires watchdog remediation with both LaunchAgent plists"
                if tenk_mode
                else "watchdog cannot automatically kickstart the supervisor without both LaunchAgent plists"
            ),
        )

    install_requested = bool(launchd_install.get("requested"))
    missing_install_roles = [
        role
        for role in ("supervisor", "watchdog")
        if not isinstance(launchd_roles.get(role), Mapping)
    ]
    if install_requested and not missing_install_roles:
        _add_readiness_check(
            checks,
            "launchd_install",
            "ok",
            "launchd installation was requested for both supervisor and watchdog agents",
            domain=str(launchd_install.get("domain") or ""),
        )
        non_persistent_roles = [
            {
                "role": role,
                "plist_path": str((launchd_roles.get(role) or {}).get("plist_path") or ""),
            }
            for role in ("supervisor", "watchdog")
            if not _is_launchagents_plist_path((launchd_roles.get(role) or {}).get("plist_path"))
        ]
        if non_persistent_roles:
            _add_readiness_check(
                checks,
                "launchd_persistence",
                "error",
                "installed LaunchAgent plist paths must be under a Library/LaunchAgents directory",
                non_persistent_roles=non_persistent_roles,
            )
        else:
            _add_readiness_check(
                checks,
                "launchd_persistence",
                "ok",
                "installed LaunchAgent plist paths are in a Library/LaunchAgents directory",
            )
    elif install_requested:
        _add_readiness_check(
            checks,
            "launchd_install",
            "error",
            "launchd installation was requested but one or more agent plist paths are missing",
            missing_roles=missing_install_roles,
        )
    else:
        _add_readiness_check(
            checks,
            "launchd_install",
            "error" if tenk_mode else "warn",
            (
                "10k set-and-forget readiness requires the wrapper to install supervisor and watchdog LaunchAgents"
                if tenk_mode
                else "LaunchAgent plists will not be installed by the wrapper; this is a manual launch plan"
            ),
        )

    if paths.get("readiness_json"):
        _add_readiness_check(
            checks,
            "readiness_artifact",
            "ok",
            "readiness report path is recorded in the runbook",
            path=str(paths.get("readiness_json")),
        )
    else:
        _add_readiness_check(checks, "readiness_artifact", "error", "readiness report path is missing from the runbook")

    status = _readiness_status(checks)
    return {
        "schema_version": 1,
        "status": status,
        "ready_for_10k_set_and_forget": status == "ok",
        "launch_blocked": bool(launch_blocked),
        "requires_readiness_ok": bool(getattr(wrapper_args, "require_readiness_ok", False)),
        "tenk_set_and_forget": bool(
            tenk_mode
        ),
        "checked_at": _now_iso(),
        "output_dir": output_dir,
        "checks": checks,
        "summary": {
            "tenk_set_and_forget": bool(
                tenk_mode
            ),
            "preflight_status": str((preflight_report or {}).get("status") or "skipped"),
            "supervisor_drill_status": drill_status,
            "supervisor_drill_cases": drill_case_count,
            "watchdog_drill_status": watchdog_drill_status,
            "pilot_certification_required": cert_required,
            "pilot_certification_status": cert_status,
            "live_adoption_requested": bool(getattr(wrapper_args, "adopt_live_run", False)),
            "live_adoption_status": live_adoption_status,
            "pilot_generation_required": pilot_generation_required,
            "pilot_generation_status": pilot_generation_status,
            "launchd_plist_requested": bool(wrapper_args.write_launchd_plist),
            "watchdog_launchd_plist_requested": bool(wrapper_args.write_watchdog_launchd_plist),
            "launchd_install_requested": install_requested,
        },
    }


def _cli_number(value: float | int) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.15g}"


def _certification_command(
    *,
    python_exe: str,
    output_dir: Path,
    supervisor_args: argparse.Namespace,
    target_cases: int,
    max_duration_hours: float,
    max_p95_duration_hours: float,
    min_pilot_cases: int,
    duration_safety_factor: float,
    require_prompt_budget_data: bool = True,
    max_prompt_tokens: int = 0,
    max_prompt_budget_adapted_case_rate: float = certify.DEFAULT_MAX_PROMPT_BUDGET_ADAPTED_CASE_RATE,
    deterministic_recovery_confidence: float = certify.DEFAULT_DETERMINISTIC_RECOVERY_CONFIDENCE,
    write_json: Path | None = None,
    expected_run_settings_fingerprint: str | None = None,
) -> list[str]:
    certify_script = REPO_ROOT / "tools" / "certify_qwen_caption_soak.py"
    command = [
        python_exe,
        str(certify_script),
        str(output_dir),
        "--target-cases",
        str(target_cases),
        "--max-duration-hours",
        _cli_number(max_duration_hours),
        "--max-p95-duration-hours",
        _cli_number(max_p95_duration_hours),
        "--min-pilot-cases",
        str(min_pilot_cases),
        "--duration-safety-factor",
        str(duration_safety_factor),
        "--max-prompt-tokens",
        str(max(0, int(max_prompt_tokens or 0))),
        "--max-prompt-budget-adapted-case-rate",
        str(max_prompt_budget_adapted_case_rate),
        "--deterministic-recovery-confidence",
        _cli_number(deterministic_recovery_confidence),
        "--max-heartbeat-age",
        str(supervisor_args.max_heartbeat_age),
        *_threshold_args(supervisor_args, include_projected_duration=False),
        "--pretty",
    ]
    if not require_prompt_budget_data:
        command.append("--no-require-prompt-budget-data")
    if bool(getattr(supervisor_args, "set_and_forget", False)):
        command.append("--set-and-forget")
    if expected_run_settings_fingerprint:
        command.extend(["--expected-run-settings-fingerprint", expected_run_settings_fingerprint])
    if write_json is not None:
        command.extend(["--write-json", str(write_json)])
    return command


def _pilot_max_p95_duration_hours(wrapper_args: argparse.Namespace) -> float:
    raw = getattr(wrapper_args, "pilot_max_p95_duration_hours", None)
    if raw is None:
        raw = getattr(wrapper_args, "pilot_max_duration_hours", certify.DEFAULT_MAX_DURATION_HOURS)
    try:
        parsed = float(raw)
    except (TypeError, ValueError, OverflowError):
        return float(getattr(wrapper_args, "pilot_max_duration_hours", certify.DEFAULT_MAX_DURATION_HOURS))
    return parsed


def _supervisor_drill_output_dir(wrapper_args: argparse.Namespace, output_dir: Path) -> Path:
    raw = getattr(wrapper_args, "supervisor_drill_output_dir", None)
    if raw:
        return raw.expanduser().resolve(strict=False)
    return output_dir / "supervisor_drill"


def _watchdog_drill_output_dir(wrapper_args: argparse.Namespace, output_dir: Path) -> Path:
    raw = getattr(wrapper_args, "watchdog_drill_output_dir", None)
    if raw:
        return raw.expanduser().resolve(strict=False)
    return output_dir / "watchdog_drill"


def _supervisor_drill_command(
    *,
    python_exe: str,
    output_dir: Path,
    write_json: Path,
    case_count: int,
    chunk_size: int,
) -> list[str]:
    command = [
        python_exe,
        str(REPO_ROOT / "tools" / "run_qwen_caption_soak_drill.py"),
        "--output-dir",
        str(output_dir),
        "--force",
        "--write-json",
        str(write_json),
    ]
    if int(case_count or 0) > 1:
        command.extend([
            "--endurance-cases",
            str(max(1, int(case_count or 1))),
            "--endurance-chunk-size",
            str(max(1, int(chunk_size or 1))),
        ])
    return command


def _watchdog_drill_command(
    *,
    python_exe: str,
    output_dir: Path,
    write_json: Path,
) -> list[str]:
    return [
        python_exe,
        str(REPO_ROOT / "tools" / "run_qwen_caption_soak_drill.py"),
        "--output-dir",
        str(output_dir),
        "--force",
        "--watchdog-remediation",
        "--write-json",
        str(write_json),
    ]


def _launchd_role_install_commands(
    *,
    launchctl_bin: str,
    domain: str,
    label: str,
    plist_path: Path,
) -> list[dict[str, Any]]:
    service_target = f"{domain}/{label}"
    plist = str(plist_path.expanduser().resolve(strict=False))
    return [
        {
            "name": "bootout_existing_service",
            "required": False,
            "command": [launchctl_bin, "bootout", service_target],
        },
        {
            "name": "bootout_existing",
            "required": False,
            "command": [launchctl_bin, "bootout", domain, plist],
        },
        {
            "name": "bootstrap",
            "required": True,
            "command": [launchctl_bin, "bootstrap", domain, plist],
        },
        {
            "name": "print_status",
            "required": True,
            "command": [launchctl_bin, "print", service_target],
        },
    ]


def _launchd_install_plan(wrapper_args: argparse.Namespace) -> dict[str, Any]:
    requested = bool(getattr(wrapper_args, "install_launchd_plists", False))
    domain = _launchctl_domain(wrapper_args)
    launchctl_bin = str(getattr(wrapper_args, "launchctl", "launchctl") or "launchctl")
    roles: dict[str, Any] = {}
    if wrapper_args.write_launchd_plist:
        roles["supervisor"] = {
            "label": str(wrapper_args.launchd_label),
            "plist_path": str(wrapper_args.write_launchd_plist.expanduser().resolve(strict=False)),
            "commands": _launchd_role_install_commands(
                launchctl_bin=launchctl_bin,
                domain=domain,
                label=str(wrapper_args.launchd_label),
                plist_path=wrapper_args.write_launchd_plist,
            ),
        }
    if wrapper_args.write_watchdog_launchd_plist:
        roles["watchdog"] = {
            "label": str(wrapper_args.watchdog_launchd_label),
            "plist_path": str(wrapper_args.write_watchdog_launchd_plist.expanduser().resolve(strict=False)),
            "commands": _launchd_role_install_commands(
                launchctl_bin=launchctl_bin,
                domain=domain,
                label=str(wrapper_args.watchdog_launchd_label),
                plist_path=wrapper_args.write_watchdog_launchd_plist,
            ),
        }
    return {
        "requested": requested,
        "domain": domain,
        "launchctl": launchctl_bin,
        "timeout_seconds": max(1.0, float(getattr(wrapper_args, "launchctl_timeout", DEFAULT_LAUNCHCTL_TIMEOUT_SECONDS) or DEFAULT_LAUNCHCTL_TIMEOUT_SECONDS)),
        "roles": roles,
    }


def _expected_run_settings_fingerprint(supervisor_args: argparse.Namespace) -> str:
    return str(runner.run_settings_payload(supervisor_args).get("fingerprint") or "")


def _run_required_pilot_certification(
    *,
    wrapper_args: argparse.Namespace,
    supervisor_args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any] | None:
    if not _pilot_certification_required(wrapper_args):
        return None
    pilot_output_dir = _pilot_output_dir(wrapper_args, output_dir)
    expected_fingerprint = _expected_run_settings_fingerprint(supervisor_args)
    report = certify.certify_soak(
        pilot_output_dir,
        target_cases=wrapper_args.pilot_target_cases,
        max_duration_hours=wrapper_args.pilot_max_duration_hours,
        max_p95_duration_hours=_pilot_max_p95_duration_hours(wrapper_args),
        min_pilot_cases=wrapper_args.pilot_min_cases,
        duration_safety_factor=wrapper_args.pilot_duration_safety_factor,
        require_prompt_budget_data=not bool(wrapper_args.no_pilot_prompt_budget_required),
        max_prompt_tokens=wrapper_args.pilot_max_prompt_tokens,
        max_prompt_budget_adapted_case_rate=wrapper_args.pilot_max_prompt_budget_adapted_case_rate,
        deterministic_recovery_confidence=wrapper_args.pilot_deterministic_recovery_confidence,
        max_heartbeat_age_seconds=supervisor_args.max_heartbeat_age,
        max_failed_case_rate=supervisor_args.max_failed_case_rate,
        max_quality_failure_rate=supervisor_args.max_quality_failure_rate,
        max_recovery_event_case_rate=supervisor_args.max_recovery_event_case_rate,
        max_loop_recovery_case_rate=supervisor_args.max_loop_recovery_case_rate,
        max_loop_guard_case_rate=supervisor_args.max_loop_guard_case_rate,
        max_deterministic_recovery_case_rate=supervisor_args.max_deterministic_recovery_case_rate,
        max_failed_attempt_row_rate=supervisor_args.max_failed_attempt_row_rate,
        max_signal_exit_attempt_row_rate=supervisor_args.max_signal_exit_attempt_row_rate,
        max_attempt_overrun_seconds=supervisor_args.max_attempt_overrun,
        min_rate_cases=supervisor_args.min_rate_cases,
        set_and_forget=bool(getattr(supervisor_args, "set_and_forget", False)),
        expected_run_settings_fingerprint=expected_fingerprint,
    )
    write_path = output_dir / "required_pilot_certification.json"
    write_path.parent.mkdir(parents=True, exist_ok=True)
    write_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _run_supervisor_drill(
    *,
    wrapper_args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    drill_output_dir = _supervisor_drill_output_dir(wrapper_args, output_dir)
    report_path = drill_output_dir / "drill_report.json"
    if bool(getattr(wrapper_args, "skip_supervisor_drill", False)):
        report = {
            "status": "skipped",
            "detail": "supervisor drill skipped by --skip-supervisor-drill",
            "output_dir": str(drill_output_dir),
            "report_json": str(report_path),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report
    try:
        case_count = max(
            1,
            int(
                getattr(wrapper_args, "supervisor_drill_cases", SET_AND_FORGET_DRILL_CASES)
                or SET_AND_FORGET_DRILL_CASES
            ),
        )
        chunk_size = max(
            1,
            int(
                getattr(
                    wrapper_args,
                    "supervisor_drill_chunk_size",
                    DEFAULT_SUPERVISOR_DRILL_CHUNK_SIZE,
                )
                or DEFAULT_SUPERVISOR_DRILL_CHUNK_SIZE
            ),
        )
        if case_count > 1:
            return soak_drill.run_endurance_drill(
                drill_output_dir,
                force=True,
                case_count=case_count,
                chunk_size=chunk_size,
                write_json=report_path,
            )
        return soak_drill.run_drill(drill_output_dir, force=True, write_json=report_path)
    except Exception as exc:  # noqa: BLE001
        report = {
            "status": "error",
            "detail": str(exc),
            "error_type": type(exc).__name__,
            "output_dir": str(drill_output_dir),
            "report_json": str(report_path),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report


def _run_watchdog_drill(
    *,
    wrapper_args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    drill_output_dir = _watchdog_drill_output_dir(wrapper_args, output_dir)
    report_path = drill_output_dir / "watchdog_drill_report.json"
    if bool(getattr(wrapper_args, "skip_watchdog_drill", False)):
        report = {
            "status": "skipped",
            "detail": "watchdog drill skipped by --skip-watchdog-drill",
            "output_dir": str(drill_output_dir),
            "report_json": str(report_path),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report
    try:
        return soak_drill.run_watchdog_remediation_drill(
            drill_output_dir,
            force=True,
            write_json=report_path,
        )
    except Exception as exc:  # noqa: BLE001
        report = {
            "status": "error",
            "detail": str(exc),
            "error_type": type(exc).__name__,
            "output_dir": str(drill_output_dir),
            "report_json": str(report_path),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report


def _pilot_sample_size(wrapper_args: argparse.Namespace) -> int:
    requested = int(getattr(wrapper_args, "pilot_sample_size", 0) or 0)
    if requested > 0:
        return requested
    return max(1, int(getattr(wrapper_args, "pilot_min_cases", 50) or 50))


def _pilot_supervisor_args(
    *,
    wrapper_args: argparse.Namespace,
    supervisor_args: argparse.Namespace,
    output_dir: Path,
) -> argparse.Namespace:
    pilot_args = argparse.Namespace(**vars(supervisor_args))
    pilot_args.output_dir = output_dir
    pilot_args.sample_size = _pilot_sample_size(wrapper_args)
    pilot_args.save_dataset_text_labels = False
    return pilot_args


def _pilot_generation_command(
    *,
    python_exe: str,
    wrapper_args: argparse.Namespace,
    supervisor_args: argparse.Namespace,
    supervisor_extra_args: Sequence[str],
    output_dir: Path,
) -> list[str]:
    pilot_args = _pilot_supervisor_args(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        output_dir=output_dir,
    )
    command = [
        python_exe,
        str(REPO_ROOT / "tools" / "supervise_qwen_caption_soak.py"),
        "--dataset-root",
        str(pilot_args.dataset_root),
        "--output-dir",
        str(pilot_args.output_dir),
        "--sample-size",
        str(pilot_args.sample_size),
        "--sample-seed",
        str(pilot_args.sample_seed),
        "--caption-mode",
        str(pilot_args.caption_mode),
        "--windowed-full-image-strategy",
        str(pilot_args.windowed_full_image_strategy),
        "--attempts",
        str(pilot_args.attempts),
        "--timeout",
        str(pilot_args.timeout),
        "--cooldown-after-crash",
        str(pilot_args.cooldown_after_crash),
        "--cooldown-after-success",
        str(pilot_args.cooldown_after_success),
        "--cooldown-backoff-multiplier",
        str(pilot_args.cooldown_backoff_multiplier),
        "--max-cooldown-after-crash",
        str(pilot_args.max_cooldown_after_crash),
        "--max-failures",
        str(pilot_args.max_failures),
        "--heartbeat-interval",
        str(pilot_args.heartbeat_interval),
        "--max-heartbeat-age",
        str(pilot_args.max_heartbeat_age),
        "--max-runner-restarts",
        str(pilot_args.max_runner_restarts),
        "--restart-delay",
        str(pilot_args.restart_delay),
        "--max-artifact-log-bytes",
        str(pilot_args.max_artifact_log_bytes),
        "--min-free-gb",
        str(pilot_args.min_free_gb),
        "--disk-safety-factor",
        str(pilot_args.disk_safety_factor),
        "--model-id",
        str(pilot_args.model_id),
        "--refinement-model-id",
        str(pilot_args.refinement_model_id),
        "--fallback-model-id",
        str(pilot_args.fallback_model_id),
        "--loop-recovery",
        str(pilot_args.loop_recovery),
        "--max-boxes",
        str(pilot_args.max_boxes),
        "--final-sentences",
        str(pilot_args.final_sentences),
        "--window-size",
        str(pilot_args.window_size),
        "--window-overlap",
        str(pilot_args.window_overlap),
        "--mlx-max-image-side",
        str(pilot_args.mlx_max_image_side),
        "--min-retry-image-side",
        str(pilot_args.min_retry_image_side),
        "--temperature",
        str(pilot_args.temperature),
        "--top-p",
        str(pilot_args.top_p),
        "--top-k",
        str(pilot_args.top_k),
        "--prompt",
        str(pilot_args.prompt),
        *_threshold_args(pilot_args),
    ]
    if pilot_args.cases_json:
        command.extend(["--cases-json", str(pilot_args.cases_json)])
    if pilot_args.request_json:
        command.extend(["--request-json", str(pilot_args.request_json)])
    if pilot_args.all_images:
        command.append("--all-images")
    if pilot_args.max_new_tokens is not None:
        command.extend(["--max-new-tokens", str(pilot_args.max_new_tokens)])
    if pilot_args.limit:
        command.extend(["--limit", str(pilot_args.limit)])
    for case in pilot_args.case or []:
        command.extend(["--case", str(case)])
    if pilot_args.preview_only:
        command.append("--preview-only")
    if pilot_args.use_sampling:
        command.append("--use-sampling")
    if pilot_args.continue_on_quality_failures:
        command.append("--continue-on-quality-failures")
    if pilot_args.skip_existing_captions:
        command.append("--skip-existing-captions")
    if pilot_args.save_dataset_text_labels:
        command.append("--save-dataset-text-labels")
    if pilot_args.allow_model_download:
        command.append("--allow-model-download")
    command.extend(str(item) for item in supervisor_extra_args)
    return command


def _run_created_pilot(
    *,
    wrapper_args: argparse.Namespace,
    supervisor_args: argparse.Namespace,
    supervisor_extra_args: Sequence[str],
    output_dir: Path,
) -> dict[str, Any] | None:
    if not wrapper_args.create_pilot_output_dir:
        return None
    pilot_output_dir = _pilot_output_dir(wrapper_args, output_dir)
    report_path = output_dir / "created_pilot_generation.json"
    if _same_resolved_path(pilot_output_dir, output_dir):
        report = {
            "status": "error",
            "detail": "pilot output directory must be separate from the large-run output directory",
            "output_dir": str(pilot_output_dir),
            "target_output_dir": str(output_dir),
            "sample_size": _pilot_sample_size(wrapper_args),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report
    if bool(getattr(wrapper_args, "dry_run", False)):
        report = {
            "status": "skipped",
            "detail": "pilot generation skipped by --dry-run",
            "output_dir": str(pilot_output_dir),
            "sample_size": _pilot_sample_size(wrapper_args),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report
    pilot_args = _pilot_supervisor_args(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        output_dir=pilot_output_dir,
    )
    return_code = supervise.supervise_soak(pilot_args, runner_extra_args=supervisor_extra_args)
    report = {
        "status": "ok" if return_code == supervise.TERMINAL_SUCCESS else "error",
        "return_code": return_code,
        "output_dir": str(pilot_output_dir),
        "sample_size": pilot_args.sample_size,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def build_unattended_plan(
    *,
    wrapper_args: argparse.Namespace,
    supervisor_args: argparse.Namespace,
    supervisor_extra_args: Sequence[str],
    supervisor_argv: Sequence[str],
) -> dict[str, Any]:
    supervise.resolve_supervisor_set_and_forget_thresholds(supervisor_args)
    apply_tenk_set_and_forget_defaults(wrapper_args, supervisor_args)
    auto_runtime_defaults = apply_set_and_forget_runtime_defaults(
        supervisor_args,
        supervisor_argv,
    )
    auto_mlx_max_image_side = bool(
        getattr(supervisor_args, "set_and_forget", False)
        and not _argv_has_flag(supervisor_argv, "--mlx-max-image-side")
    )
    auto_min_retry_image_side = bool(
        getattr(supervisor_args, "set_and_forget", False)
        and not _argv_has_flag(supervisor_argv, "--min-retry-image-side")
    )
    auto_cooldown_after_success = bool(
        getattr(supervisor_args, "set_and_forget", False)
        and not _argv_has_flag(supervisor_argv, "--cooldown-after-success")
    )
    auto_attempts = bool(
        getattr(supervisor_args, "set_and_forget", False)
        and not _argv_has_flag(supervisor_argv, "--attempts")
    )
    auto_windowed_full_image_strategy = apply_set_and_forget_windowed_strategy_default(
        supervisor_args,
        supervisor_argv,
    )
    output_dir = supervisor_args.output_dir.expanduser().resolve(strict=False)
    python_exe = str(Path(str(wrapper_args.python)).expanduser())
    supervisor_drill_dir = _supervisor_drill_output_dir(wrapper_args, output_dir)
    supervisor_drill_json = supervisor_drill_dir / "drill_report.json"
    watchdog_drill_dir = _watchdog_drill_output_dir(wrapper_args, output_dir)
    watchdog_drill_json = watchdog_drill_dir / "watchdog_drill_report.json"
    pilot_output_dir = _pilot_output_dir(wrapper_args, output_dir)
    supervisor_script = REPO_ROOT / "tools" / "supervise_qwen_caption_soak.py"
    watch_script = REPO_ROOT / "tools" / "watch_qwen_caption_soak.py"
    audit_script = REPO_ROOT / "tools" / "audit_qwen_caption_soak.py"
    operation_audit_script = REPO_ROOT / "tools" / "audit_qwen_caption_operation.py"
    paths = _command_paths(output_dir)
    runbook_json = (
        wrapper_args.runbook_json.expanduser().resolve(strict=False)
        if wrapper_args.runbook_json
        else output_dir / RUNBOOK_NAME
    )
    paths["runbook_json"] = str(runbook_json)
    supervisor_command_args = [str(item) for item in supervisor_argv]
    if auto_runtime_defaults:
        if auto_mlx_max_image_side:
            _ensure_cli_flag_value(
                supervisor_command_args,
                "--mlx-max-image-side",
                supervisor_args.mlx_max_image_side,
            )
        if auto_min_retry_image_side:
            _ensure_cli_flag_value(
                supervisor_command_args,
                "--min-retry-image-side",
                supervisor_args.min_retry_image_side,
            )
        if auto_cooldown_after_success:
            _ensure_cli_flag_value(
                supervisor_command_args,
                "--cooldown-after-success",
                supervisor_args.cooldown_after_success,
            )
        if auto_attempts:
            _ensure_cli_flag_value(
                supervisor_command_args,
                "--attempts",
                supervisor_args.attempts,
            )
    if auto_windowed_full_image_strategy:
        _ensure_cli_flag_value(
            supervisor_command_args,
            "--windowed-full-image-strategy",
            supervisor_args.windowed_full_image_strategy,
        )
    if bool(getattr(wrapper_args, "require_readiness_ok", False)):
        _ensure_cli_flag_value(supervisor_command_args, "--min-free-gb", supervisor_args.min_free_gb)
        if float(getattr(supervisor_args, "max_projected_duration_hours", 0.0) or 0.0) > 0:
            _ensure_cli_flag_value(
                supervisor_command_args,
                "--max-projected-duration-hours",
                supervisor_args.max_projected_duration_hours,
            )
    supervisor_command = [
        python_exe,
        str(supervisor_script),
        *supervisor_command_args,
    ]
    watch_command = [
        python_exe,
        str(watch_script),
        str(output_dir),
        "--interval",
        "60",
        "--max-heartbeat-age",
        str(supervisor_args.max_heartbeat_age),
        "--max-no-progress",
        str(wrapper_args.watchdog_max_no_progress),
        "--graceful-restart-timeout",
        str(wrapper_args.watchdog_graceful_restart_timeout),
        "--latest-json",
        str(paths["watchdog_latest_json"]),
        "--state-json",
        str(paths["watchdog_state_json"]),
        *_threshold_args(supervisor_args),
        *_disk_reserve_args(supervisor_args),
        *_saved_text_label_audit_args(supervisor_args),
        *_watchdog_remediation_args(wrapper_args),
    ]
    audit_command = [
        python_exe,
        str(audit_script),
        str(output_dir),
        "--max-heartbeat-age",
        str(supervisor_args.max_heartbeat_age),
        *_threshold_args(supervisor_args),
        *_disk_reserve_args(supervisor_args),
        *_saved_text_label_audit_args(supervisor_args),
        "--pretty",
    ]
    live_status_command = [
        python_exe,
        str(audit_script),
        str(output_dir),
        "--max-heartbeat-age",
        str(supervisor_args.max_heartbeat_age),
        *_threshold_args(supervisor_args),
        *_disk_reserve_args(supervisor_args),
        *_saved_text_label_audit_args(supervisor_args),
        "--allow-running-incomplete",
        "--compact",
    ]
    operational_audit_command = [
        python_exe,
        str(operation_audit_script),
        str(paths["runbook_json"]),
        "--allow-running-incomplete",
        "--compact",
        "--write-json",
        str(paths["operational_audit_json"]),
    ]
    if bool(getattr(wrapper_args, "require_readiness_ok", False)):
        operational_audit_command.append("--strict-set-and-forget")
    paths["supervisor_drill_dir"] = str(supervisor_drill_dir)
    paths["supervisor_drill_json"] = str(supervisor_drill_json)
    paths["watchdog_drill_dir"] = str(watchdog_drill_dir)
    paths["watchdog_drill_json"] = str(watchdog_drill_json)
    readiness_json = (
        wrapper_args.readiness_json.expanduser().resolve(strict=False)
        if wrapper_args.readiness_json
        else output_dir / READINESS_NAME
    )
    paths["readiness_json"] = str(readiness_json)
    certification_output_dir = pilot_output_dir
    certification_write_path = (
        output_dir / "required_pilot_certification.json"
        if _pilot_certification_required(wrapper_args)
        else output_dir / "certification.json"
    )
    expected_fingerprint = (
        _expected_run_settings_fingerprint(supervisor_args)
        if _pilot_certification_required(wrapper_args)
        else ""
    )
    certification_command = _certification_command(
        python_exe=python_exe,
        output_dir=certification_output_dir,
        supervisor_args=supervisor_args,
        target_cases=wrapper_args.pilot_target_cases,
        max_duration_hours=wrapper_args.pilot_max_duration_hours,
        max_p95_duration_hours=_pilot_max_p95_duration_hours(wrapper_args),
        min_pilot_cases=wrapper_args.pilot_min_cases,
        duration_safety_factor=wrapper_args.pilot_duration_safety_factor,
        require_prompt_budget_data=not bool(wrapper_args.no_pilot_prompt_budget_required),
        max_prompt_tokens=wrapper_args.pilot_max_prompt_tokens,
        max_prompt_budget_adapted_case_rate=wrapper_args.pilot_max_prompt_budget_adapted_case_rate,
        deterministic_recovery_confidence=wrapper_args.pilot_deterministic_recovery_confidence,
        write_json=certification_write_path,
        expected_run_settings_fingerprint=expected_fingerprint,
    )
    supervisor_drill_command = _supervisor_drill_command(
        python_exe=python_exe,
        output_dir=supervisor_drill_dir,
        write_json=supervisor_drill_json,
        case_count=wrapper_args.supervisor_drill_cases,
        chunk_size=wrapper_args.supervisor_drill_chunk_size,
    )
    watchdog_drill_command = _watchdog_drill_command(
        python_exe=python_exe,
        output_dir=watchdog_drill_dir,
        write_json=watchdog_drill_json,
    )
    pilot_generation_command = _pilot_generation_command(
        python_exe=python_exe,
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        output_dir=pilot_output_dir,
    )
    return {
        "schema_version": 1,
        "created_at": _now_iso(),
        "repo_root": str(REPO_ROOT),
        "output_dir": str(output_dir),
        "paths": paths,
        "supervisor_args": _jsonable(vars(supervisor_args)),
        "supervisor_extra_args": list(supervisor_extra_args),
        "commands": {
            "supervisor": supervisor_command,
            "supervisor_shell": _quote_command(supervisor_command),
            "watchdog": watch_command,
            "watchdog_shell": _quote_command(watch_command),
            "final_audit": audit_command,
            "final_audit_shell": _quote_command(audit_command),
            "live_status": live_status_command,
            "live_status_shell": _quote_command(live_status_command),
            "operational_audit": operational_audit_command,
            "operational_audit_shell": _quote_command(operational_audit_command),
            "pilot_certification": certification_command,
            "pilot_certification_shell": _quote_command(certification_command),
            "supervisor_drill": supervisor_drill_command,
            "supervisor_drill_shell": _quote_command(supervisor_drill_command),
            "watchdog_drill": watchdog_drill_command,
            "watchdog_drill_shell": _quote_command(watchdog_drill_command),
            "pilot_generation": pilot_generation_command,
            "pilot_generation_shell": _quote_command(pilot_generation_command),
        },
        "supervisor_drill_gate": {
            "required": not bool(wrapper_args.skip_supervisor_drill),
            "output_dir": str(supervisor_drill_dir),
            "report_json": str(supervisor_drill_json),
            "case_count": max(1, int(wrapper_args.supervisor_drill_cases or 1)),
            "chunk_size": max(1, int(wrapper_args.supervisor_drill_chunk_size or 1)),
        },
        "watchdog_drill_gate": {
            "required": not bool(wrapper_args.skip_watchdog_drill),
            "output_dir": str(watchdog_drill_dir),
            "report_json": str(watchdog_drill_json),
        },
        "set_and_forget_gate": {
            "tenk_mode": bool(getattr(wrapper_args, "tenk_set_and_forget", False)),
            "require_readiness_ok": bool(getattr(wrapper_args, "require_readiness_ok", False)),
            "auto_pilot_generation": bool(getattr(wrapper_args, "auto_create_pilot_output_dir", False)),
            "auto_prompt_size_ceiling": bool(getattr(wrapper_args, "auto_pilot_max_prompt_tokens", False)),
            "auto_pilot_min_cases": bool(getattr(wrapper_args, "auto_pilot_min_cases", False)),
            "auto_pilot_sample_size": bool(getattr(wrapper_args, "auto_pilot_sample_size", False)),
            "auto_runtime_defaults": bool(auto_runtime_defaults),
            "pilot_deterministic_recovery_confidence": float(
                getattr(
                    wrapper_args,
                    "pilot_deterministic_recovery_confidence",
                    certify.DEFAULT_DETERMINISTIC_RECOVERY_CONFIDENCE,
                )
                or 0.0
            ),
            "auto_mlx_max_image_side": bool(auto_mlx_max_image_side),
            "auto_min_retry_image_side": bool(auto_min_retry_image_side),
            "auto_cooldown_after_success": bool(auto_cooldown_after_success),
            "auto_attempts": bool(auto_attempts),
            "mlx_max_image_side": int(getattr(supervisor_args, "mlx_max_image_side", 0) or 0),
            "min_retry_image_side": int(getattr(supervisor_args, "min_retry_image_side", 0) or 0),
            "attempts": int(getattr(supervisor_args, "attempts", 0) or 0),
            "cooldown_after_success_seconds": float(
                getattr(supervisor_args, "cooldown_after_success", 0.0) or 0.0
            ),
            "windowed_full_image_strategy": str(
                getattr(supervisor_args, "windowed_full_image_strategy", "visual") or "visual"
            ),
            "auto_windowed_full_image_strategy": bool(auto_windowed_full_image_strategy),
        },
        "launchd_power_assertion": {
            "enabled": bool(getattr(wrapper_args, "launchd_caffeinate", False)),
            "program": CAFFEINATE_BIN,
            "arguments": [*CAFFEINATE_ARGS],
            "detail": (
                "LaunchAgent commands are wrapped with caffeinate to prevent macOS sleep"
                if bool(getattr(wrapper_args, "launchd_caffeinate", False))
                else "LaunchAgent commands are not wrapped with caffeinate"
            ),
        },
        "post_install_operation_audit_gate": {
            "required": _post_install_operation_audit_required(wrapper_args),
            "skipped": bool(getattr(wrapper_args, "skip_post_install_operation_audit", False)),
            "timeout_seconds": _post_install_operation_audit_timeout(wrapper_args),
            "interval_seconds": _post_install_operation_audit_interval(wrapper_args),
            "detail": (
                "Strict launchd handoff waits for the live operation audit to pass after installing LaunchAgents"
                if _post_install_operation_audit_required(wrapper_args)
                else "Post-install operation audit is not required for this launch plan"
            ),
        },
        "launchd_install": _launchd_install_plan(wrapper_args),
        "certification_gate": {
            "required": _pilot_certification_required(wrapper_args),
            "pilot_output_dir": str(certification_output_dir),
            "auto_prompt_size_ceiling": bool(getattr(wrapper_args, "auto_pilot_max_prompt_tokens", False)),
            "expected_run_settings_fingerprint": expected_fingerprint,
            "target_cases": wrapper_args.pilot_target_cases,
            "max_duration_hours": wrapper_args.pilot_max_duration_hours,
            "max_p95_duration_hours": _pilot_max_p95_duration_hours(wrapper_args),
            "min_pilot_cases": wrapper_args.pilot_min_cases,
            "auto_min_pilot_cases": bool(getattr(wrapper_args, "auto_pilot_min_cases", False)),
            "duration_safety_factor": wrapper_args.pilot_duration_safety_factor,
            "require_prompt_budget_data": not bool(wrapper_args.no_pilot_prompt_budget_required),
            "require_runner_capabilities": True,
            "required_runner_capabilities": list(runner.RUNNER_CAPABILITIES),
            "max_prompt_tokens": max(0, int(wrapper_args.pilot_max_prompt_tokens or 0)),
            "max_prompt_budget_adapted_case_rate": wrapper_args.pilot_max_prompt_budget_adapted_case_rate,
            "deterministic_recovery_confidence": wrapper_args.pilot_deterministic_recovery_confidence,
        },
        "pilot_generation_gate": {
            "required": bool(wrapper_args.create_pilot_output_dir),
            "output_dir": str(pilot_output_dir),
            "target_output_dir": str(output_dir),
            "separate_from_target_output_dir": not _same_resolved_path(pilot_output_dir, output_dir),
            "sample_size": _pilot_sample_size(wrapper_args),
            "auto_configured": bool(getattr(wrapper_args, "auto_create_pilot_output_dir", False)),
            "auto_sample_size": bool(getattr(wrapper_args, "auto_pilot_sample_size", False)),
            "read_only": True,
            "large_run_save_dataset_text_labels": bool(supervisor_args.save_dataset_text_labels),
        },
        "notes": [
            "Run the supervisor command directly, or install the generated supervisor LaunchAgent plist with launchctl.",
            "For set-and-forget runs, also install the generated watchdog LaunchAgent plist so health audits continue independently.",
            "A launchd-backed watchdog can boundedly kickstart the supervisor LaunchAgent after repeated unhealthy checks.",
            "Use --install-launchd-plists for a one-command launchd-backed set-and-forget start after gates pass.",
            "The LaunchAgent KeepAlive policy restarts each role only after unsuccessful exits.",
            "The supervisor drill is a no-GPU launch gate that proves restart behavior before a set-and-forget run starts.",
            "The watchdog drill is a no-GPU launch gate that proves automatic launchd remediation before a set-and-forget run starts.",
            "When pilot generation is enabled, the wrapper runs the pilot-generation command before certification and the large supervisor.",
            "Use --tenk-set-and-forget for a fail-closed 10k launch; it enables strict readiness automatically.",
            "Use --require-pilot-certification for 10k-scale launches that must not start until a pilot run certifies clean.",
            "Use the live_status command for a compact health summary while the run is still incomplete.",
            "Use the operational_audit command to verify launchd services, watchdog state, sleep assertions, and live artifacts after handoff.",
            "A final audit status of ok is required before trusting a completed two-week run.",
        ],
    }


def build_static_tenk_launch_gate_report(
    *,
    wrapper_args: argparse.Namespace,
    supervisor_args: argparse.Namespace,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject statically impossible 10k plans before pilot/GPU work starts."""
    tenk_mode = bool(getattr(wrapper_args, "tenk_set_and_forget", False))
    checks: list[dict[str, Any]] = []
    if not tenk_mode:
        return {
            "schema_version": 1,
            "required": False,
            "status": "ok",
            "checked_at": _now_iso(),
            "checks": checks,
        }

    launchd_install = plan.get("launchd_install") if isinstance(plan.get("launchd_install"), Mapping) else {}
    launchd_roles = launchd_install.get("roles") if isinstance(launchd_install.get("roles"), Mapping) else {}

    def add(name: str, ok: bool, ok_detail: str, error_detail: str, **fields: Any) -> None:
        _add_readiness_check(
            checks,
            name,
            _check_status(ok),
            ok_detail if ok else error_detail,
            **fields,
        )

    add(
        "wrapper_preflight_enabled",
        not bool(getattr(wrapper_args, "no_preflight", False)),
        "wrapper preflight will run before handoff",
        "10k set-and-forget requires wrapper preflight before any pilot or handoff work",
    )
    add(
        "supervisor_drill_enabled",
        not bool(getattr(wrapper_args, "skip_supervisor_drill", False)),
        "supervisor restart drill is enabled",
        "10k set-and-forget requires the supervisor restart drill before pilot or handoff work",
    )
    add(
        "watchdog_drill_enabled",
        not bool(getattr(wrapper_args, "skip_watchdog_drill", False)),
        "watchdog remediation drill is enabled",
        "10k set-and-forget requires the watchdog remediation drill before pilot or handoff work",
    )
    drill_cases = max(1, int(getattr(wrapper_args, "supervisor_drill_cases", 1) or 1))
    add(
        "supervisor_drill_scale",
        drill_cases >= SET_AND_FORGET_DRILL_CASES,
        "supervisor restart drill scale meets the set-and-forget gate",
        "10k set-and-forget requires the default supervisor restart drill scale before pilot work",
        case_count=drill_cases,
        required_case_count=SET_AND_FORGET_DRILL_CASES,
    )
    has_pilot_or_adoption = bool(
        getattr(wrapper_args, "require_pilot_certification", None)
        or getattr(wrapper_args, "create_pilot_output_dir", None)
        or getattr(wrapper_args, "adopt_live_run", False)
    )
    add(
        "pilot_or_live_adoption_evidence",
        has_pilot_or_adoption,
        "pilot certification, pilot generation, or live-run adoption evidence is configured",
        "10k set-and-forget requires pilot certification, generated pilot certification, or live-run adoption evidence",
    )
    configured_pilot_dir = bool(
        getattr(wrapper_args, "require_pilot_certification", None)
        or getattr(wrapper_args, "create_pilot_output_dir", None)
    )
    target_output_dir = supervisor_args.output_dir.expanduser().resolve(strict=False)
    pilot_output_dir = _pilot_output_dir(wrapper_args, target_output_dir)
    add(
        "pilot_output_dir_isolated",
        (not configured_pilot_dir) or not _same_resolved_path(pilot_output_dir, target_output_dir),
        "pilot or certification artifacts are isolated from the large-run output directory",
        "10k set-and-forget requires pilot or certification artifacts outside the large-run output directory",
        pilot_output_dir=str(pilot_output_dir) if configured_pilot_dir else "",
        target_output_dir=str(target_output_dir),
    )
    add(
        "prompt_size_ceiling",
        int(getattr(wrapper_args, "pilot_max_prompt_tokens", 0) or 0) > 0,
        "positive prompt-size ceiling is configured",
        "10k set-and-forget requires a positive prompt-size ceiling before pilot or handoff work",
        max_prompt_tokens=max(0, int(getattr(wrapper_args, "pilot_max_prompt_tokens", 0) or 0)),
    )
    add(
        "prompt_budget_telemetry_required",
        not bool(getattr(wrapper_args, "no_pilot_prompt_budget_required", False)),
        "pilot prompt-budget telemetry is required",
        "10k set-and-forget requires pilot prompt-budget telemetry",
    )
    add(
        "prompt_budget_adaptation_gate",
        float(getattr(wrapper_args, "pilot_max_prompt_budget_adapted_case_rate", 0.0) or 0.0) >= 0.0,
        "prompt-budget adaptation-rate gate is configured",
        "10k set-and-forget requires a prompt-budget adaptation-rate gate",
        max_prompt_budget_adapted_case_rate=float(
            getattr(wrapper_args, "pilot_max_prompt_budget_adapted_case_rate", 0.0) or 0.0
        ),
    )
    required_pilot_cases = _tenk_auto_pilot_case_count(wrapper_args)
    configured_min_pilot_cases = max(0, int(getattr(wrapper_args, "pilot_min_cases", 0) or 0))
    configured_pilot_sample_size = _pilot_sample_size(wrapper_args)
    add(
        "pilot_confidence_sample_size",
        (
            (not configured_pilot_dir)
            or (
                configured_min_pilot_cases >= required_pilot_cases
                and (
                    not bool(getattr(wrapper_args, "create_pilot_output_dir", None))
                    or configured_pilot_sample_size >= required_pilot_cases
                )
            )
        ),
        "pilot sample size is large enough for the deterministic-recovery confidence gate",
        "10k set-and-forget requires a confidence-backed pilot sample before pilot or handoff work",
        min_pilot_cases=configured_min_pilot_cases,
        sample_size=configured_pilot_sample_size if bool(getattr(wrapper_args, "create_pilot_output_dir", None)) else 0,
        required_pilot_cases=required_pilot_cases,
    )
    confidence = float(
        getattr(
            wrapper_args,
            "pilot_deterministic_recovery_confidence",
            certify.DEFAULT_DETERMINISTIC_RECOVERY_CONFIDENCE,
        )
        or 0.0
    )
    add(
        "pilot_deterministic_recovery_confidence",
        confidence > 0.0,
        "deterministic-recovery confidence gate is enabled",
        "10k set-and-forget requires deterministic-recovery confidence certification; use manual mode for diagnostics",
        confidence=confidence,
    )
    add(
        "model_downloads_disabled",
        not bool(getattr(supervisor_args, "allow_model_download", False)),
        "selected caption models must already be local before handoff",
        "10k set-and-forget rejects --allow-model-download; pre-download models before pilot or handoff work",
        allow_model_download=bool(getattr(supervisor_args, "allow_model_download", False)),
    )
    supervisor_attempts = max(0, int(getattr(supervisor_args, "attempts", 0) or 0))
    add(
        "vlm_retry_attempts",
        supervisor_attempts >= 3,
        "supervisor will retry failed VLM cases before degraded fallback",
        "10k set-and-forget requires at least 3 attempts so signal exits retry on the VLM path before degraded fallback",
        attempts=supervisor_attempts,
        required_attempts=3,
    )
    add(
        "launchd_install_requested",
        bool(getattr(wrapper_args, "install_launchd_plists", False)),
        "wrapper will install supervisor and watchdog LaunchAgents",
        "10k set-and-forget requires --install-launchd-plists before pilot or handoff work",
    )
    add(
        "post_install_operation_audit_enabled",
        not bool(getattr(wrapper_args, "skip_post_install_operation_audit", False)),
        "post-install live operation audit is enabled",
        "10k set-and-forget requires the launcher to prove the live operation audit after launchd handoff",
    )
    add(
        "post_install_operation_audit_timeout",
        _post_install_operation_audit_timeout(wrapper_args) > 0.0,
        "post-install live operation audit has a positive timeout",
        "10k set-and-forget requires a positive post-install operation-audit timeout",
        timeout_seconds=_post_install_operation_audit_timeout(wrapper_args),
        interval_seconds=_post_install_operation_audit_interval(wrapper_args),
    )
    add(
        "supervisor_launchagent_configured",
        bool(getattr(wrapper_args, "write_launchd_plist", None)),
        "supervisor LaunchAgent plist path is configured",
        "10k set-and-forget requires a supervisor LaunchAgent plist path",
    )
    add(
        "watchdog_launchagent_configured",
        bool(getattr(wrapper_args, "write_watchdog_launchd_plist", None)),
        "watchdog LaunchAgent plist path is configured",
        "10k set-and-forget requires a watchdog LaunchAgent plist path",
    )
    add(
        "launchd_roles_configured",
        all(isinstance(launchd_roles.get(role), Mapping) for role in ("supervisor", "watchdog")),
        "launchd install plan includes both supervisor and watchdog roles",
        "10k set-and-forget requires launchd install roles for both supervisor and watchdog",
    )
    add(
        "launchd_sleep_prevention",
        bool(getattr(wrapper_args, "launchd_caffeinate", False)),
        "launchd commands will run under caffeinate",
        "10k set-and-forget requires launchd sleep prevention before pilot or handoff work",
    )
    add(
        "watchdog_remediation_enabled",
        int(getattr(wrapper_args, "watchdog_max_remediations", 0) or 0) > 0,
        "watchdog launchd remediation is enabled",
        "10k set-and-forget requires watchdog launchd remediation before pilot or handoff work",
        max_remediations=max(0, int(getattr(wrapper_args, "watchdog_max_remediations", 0) or 0)),
    )
    add(
        "watchdog_graceful_restart_enabled",
        float(getattr(wrapper_args, "watchdog_graceful_restart_timeout", 0.0) or 0.0) > 0,
        "watchdog cooperative restart requests are enabled",
        "10k set-and-forget requires cooperative restart requests before launchd escalation",
        graceful_restart_timeout_seconds=max(
            0.0,
            float(getattr(wrapper_args, "watchdog_graceful_restart_timeout", 0.0) or 0.0),
        ),
    )
    add(
        "projected_duration_gate",
        float(getattr(supervisor_args, "max_projected_duration_hours", 0.0) or 0.0) > 0,
        "projected-duration gate is configured",
        "10k set-and-forget requires a positive projected-duration gate before pilot or handoff work",
        max_projected_duration_hours=float(
            getattr(supervisor_args, "max_projected_duration_hours", 0.0) or 0.0
        ),
    )
    add(
        "disk_reserve_gate",
        float(getattr(supervisor_args, "min_free_gb", 0.0) or 0.0) > 0,
        "live disk-reserve gate is configured",
        "10k set-and-forget requires a positive live disk-reserve gate before pilot or handoff work",
        min_free_gb=float(getattr(supervisor_args, "min_free_gb", 0.0) or 0.0),
    )

    status = _readiness_status(checks)
    return {
        "schema_version": 1,
        "required": True,
        "status": status,
        "checked_at": _now_iso(),
        "checks": checks,
    }


def write_runbook(
    path: Path,
    plan: Mapping[str, Any],
    preflight_report: Mapping[str, Any] | None,
    static_gate_report: Mapping[str, Any] | None = None,
    supervisor_drill_report: Mapping[str, Any] | None = None,
    watchdog_drill_report: Mapping[str, Any] | None = None,
    pilot_generation_report: Mapping[str, Any] | None = None,
    certification_report: Mapping[str, Any] | None = None,
    live_adoption_report: Mapping[str, Any] | None = None,
    readiness_report: Mapping[str, Any] | None = None,
    launchd_install_report: Mapping[str, Any] | None = None,
    post_install_operation_audit_report: Mapping[str, Any] | None = None,
) -> None:
    payload = dict(plan)
    if preflight_report is not None:
        payload["preflight"] = dict(preflight_report)
    if static_gate_report is not None:
        payload["tenk_static_launch_gate"] = dict(static_gate_report)
    if supervisor_drill_report is not None:
        payload["supervisor_drill"] = dict(supervisor_drill_report)
    if watchdog_drill_report is not None:
        payload["watchdog_drill"] = dict(watchdog_drill_report)
    if pilot_generation_report is not None:
        payload["created_pilot_generation"] = dict(pilot_generation_report)
    if certification_report is not None:
        payload["required_pilot_certification"] = dict(certification_report)
    if live_adoption_report is not None:
        payload["live_adoption_certification"] = dict(live_adoption_report)
    if readiness_report is not None:
        payload["readiness"] = dict(readiness_report)
    if launchd_install_report is not None:
        payload["launchd_install_result"] = dict(launchd_install_report)
    if post_install_operation_audit_report is not None:
        payload["post_install_operation_audit"] = dict(post_install_operation_audit_report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_readiness_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True), encoding="utf-8")


def _preflight_error_can_defer_to_supervisor(preflight_report: Mapping[str, Any] | None) -> bool:
    if not preflight_report:
        return False
    status = str(preflight_report.get("status") or "").lower()
    if status != "error" or supervise._live_runner_lock_check(preflight_report) is None:
        return False
    checks = preflight_report.get("checks")
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
        return False
    error_check_names = [
        str(check.get("name") or "")
        for check in checks
        if isinstance(check, Mapping) and str(check.get("status") or "").lower() == "error"
    ]
    return bool(error_check_names) and set(error_check_names) <= {"runner_lock"}


def _live_adoption_requested(wrapper_args: argparse.Namespace, preflight_report: Mapping[str, Any] | None) -> bool:
    return bool(getattr(wrapper_args, "adopt_live_run", False)) and _preflight_error_can_defer_to_supervisor(
        preflight_report
    )


def _live_adoption_certification_required(
    wrapper_args: argparse.Namespace,
    preflight_report: Mapping[str, Any] | None,
) -> bool:
    return _live_adoption_requested(wrapper_args, preflight_report)


def _run_live_adoption_certification(
    *,
    wrapper_args: argparse.Namespace,
    supervisor_args: argparse.Namespace,
    preflight_report: Mapping[str, Any] | None,
    output_dir: Path,
) -> dict[str, Any] | None:
    if not _live_adoption_certification_required(wrapper_args, preflight_report):
        return None
    tenk_mode = bool(getattr(wrapper_args, "tenk_set_and_forget", False))
    checks: list[dict[str, Any]] = []
    runner_lock = supervise._live_runner_lock_check(preflight_report or {})
    if runner_lock is None:
        _add_readiness_check(
            checks,
            "live_runner_lock",
            "error",
            "--adopt-live-run requires wrapper preflight to find a live runner lock",
        )
    else:
        _add_readiness_check(
            checks,
            "live_runner_lock",
            "ok",
            "wrapper preflight found a live runner lock that the launchd supervisor can wait on",
            runner_lock=runner_lock,
        )
        if bool(runner_lock.get("runner_supports_graceful_restart")):
            _add_readiness_check(
                checks,
                "live_runner_restart_capability",
                "ok",
                "active runner advertises cooperative restart support for unattended handoff",
                runner_capabilities=runner_lock.get("runner_capabilities") or [],
                runner_capability_sources=runner_lock.get("runner_capability_sources") or [],
            )
        else:
            _add_readiness_check(
                checks,
                "live_runner_restart_capability",
                "error",
                (
                    "active runner does not advertise cooperative restart support; "
                    "restart it under the current runner before adopting as set-and-forget"
                ),
                runner_capabilities=runner_lock.get("runner_capabilities") or [],
                runner_capability_sources=runner_lock.get("runner_capability_sources") or [],
            )
    if bool(getattr(wrapper_args, "install_launchd_plists", False)):
        _add_readiness_check(
            checks,
            "launchd_install_requested",
            "ok",
            "live-run adoption will install supervisor and watchdog LaunchAgents",
        )
    else:
        _add_readiness_check(
            checks,
            "launchd_install_requested",
            "error",
            "--adopt-live-run requires --install-launchd-plists for a true walk-away handoff",
        )
    live_audit = supervise.audit.audit_soak(
        output_dir,
        max_heartbeat_age_seconds=supervisor_args.max_heartbeat_age,
        allow_running_incomplete=True,
        max_failed_case_rate=supervisor_args.max_failed_case_rate,
        max_quality_failure_rate=supervisor_args.max_quality_failure_rate,
        max_recovery_event_case_rate=supervisor_args.max_recovery_event_case_rate,
        max_loop_recovery_case_rate=supervisor_args.max_loop_recovery_case_rate,
        max_loop_guard_case_rate=supervisor_args.max_loop_guard_case_rate,
        max_deterministic_recovery_case_rate=supervisor_args.max_deterministic_recovery_case_rate,
        max_failed_attempt_row_rate=supervisor_args.max_failed_attempt_row_rate,
        max_signal_exit_attempt_row_rate=supervisor_args.max_signal_exit_attempt_row_rate,
        max_attempt_overrun_seconds=supervisor_args.max_attempt_overrun,
        max_projected_duration_hours=supervisor_args.max_projected_duration_hours,
        min_free_gb=supervisor_args.min_free_gb,
        min_rate_cases=supervisor_args.min_rate_cases,
        set_and_forget=bool(getattr(supervisor_args, "set_and_forget", False)),
        require_saved_text_labels=bool(
            getattr(supervisor_args, "require_saved_text_labels", False)
            or getattr(supervisor_args, "save_dataset_text_labels", False)
        ),
    )
    live_audit_status = str(live_audit.get("status") or "error")
    _add_readiness_check(
        checks,
        "live_artifact_audit",
        "ok" if live_audit_status == "ok" else "error",
        f"live set-and-forget artifact audit status is {live_audit_status}",
        audit_status=live_audit_status,
    )
    processed_cases = max(0, int(live_audit.get("processed_cases") or 0))
    min_completed_cases = max(1, int(getattr(wrapper_args, "adopt_min_completed_cases", 1) or 1))
    if processed_cases >= min_completed_cases:
        _add_readiness_check(
            checks,
            "completed_case_evidence",
            "ok",
            f"{processed_cases} completed cases meet adoption minimum {min_completed_cases}",
            processed_cases=processed_cases,
            min_completed_cases=min_completed_cases,
        )
    else:
        _add_readiness_check(
            checks,
            "completed_case_evidence",
            "error",
            f"{processed_cases} completed cases is below adoption minimum {min_completed_cases}",
            processed_cases=processed_cases,
            min_completed_cases=min_completed_cases,
        )
    degraded_rates = (
        live_audit.get("degraded_rates")
        if isinstance(live_audit.get("degraded_rates"), Mapping)
        else {}
    )
    prompt_coverage = float(degraded_rates.get("prompt_budget_coverage_rate") or 0.0)
    min_prompt_coverage = max(
        0.0,
        min(1.0, float(getattr(wrapper_args, "adopt_min_prompt_budget_coverage", 0.0) or 0.0)),
    )
    if prompt_coverage >= min_prompt_coverage:
        _add_readiness_check(
            checks,
            "prompt_budget_coverage",
            "ok",
            f"prompt-budget telemetry coverage {prompt_coverage:.3f} meets adoption minimum {min_prompt_coverage:.3f}",
            prompt_budget_coverage_rate=prompt_coverage,
            min_prompt_budget_coverage_rate=min_prompt_coverage,
        )
    else:
        _add_readiness_check(
            checks,
            "prompt_budget_coverage",
            "error",
            f"prompt-budget telemetry coverage {prompt_coverage:.3f} is below adoption minimum {min_prompt_coverage:.3f}",
            prompt_budget_coverage_rate=prompt_coverage,
            min_prompt_budget_coverage_rate=min_prompt_coverage,
        )
    max_prompt_tokens_gate = max(0, int(getattr(wrapper_args, "pilot_max_prompt_tokens", 0) or 0))
    observed_max_prompt_tokens = max(0, int(degraded_rates.get("max_prompt_tokens") or 0))
    if max_prompt_tokens_gate > 0 and observed_max_prompt_tokens > max_prompt_tokens_gate:
        _add_readiness_check(
            checks,
            "max_prompt_tokens",
            "error",
            f"live max prompt size {observed_max_prompt_tokens} exceeds adoption gate {max_prompt_tokens_gate}",
            max_prompt_tokens=observed_max_prompt_tokens,
            max_prompt_tokens_gate=max_prompt_tokens_gate,
        )
    elif max_prompt_tokens_gate > 0:
        _add_readiness_check(
            checks,
            "max_prompt_tokens",
            "ok",
            f"live max prompt size {observed_max_prompt_tokens} is within adoption gate {max_prompt_tokens_gate}",
            max_prompt_tokens=observed_max_prompt_tokens,
            max_prompt_tokens_gate=max_prompt_tokens_gate,
        )
    else:
        _add_readiness_check(
            checks,
            "max_prompt_tokens",
            "error" if tenk_mode else "ok",
            (
                "10k set-and-forget live adoption requires a positive prompt-size gate"
                if tenk_mode
                else "live adoption prompt-size gate is disabled"
            ),
            max_prompt_tokens=observed_max_prompt_tokens,
            max_prompt_tokens_gate=max_prompt_tokens_gate,
        )
    status = _readiness_status(checks)
    return {
        "schema_version": 1,
        "status": status,
        "checked_at": _now_iso(),
        "output_dir": str(output_dir),
        "checks": checks,
        "live_audit": {
            "status": live_audit_status,
            "processed_cases": processed_cases,
            "expected_cases": live_audit.get("expected_cases"),
            "degraded_rates": degraded_rates,
            "runtime_projection": live_audit.get("runtime_projection"),
            "active_attempt": live_audit.get("active_attempt"),
            "disk_reserve": live_audit.get("disk_reserve"),
        },
    }


def launchd_plist_payload(
    *,
    label: str,
    command: Sequence[str],
    working_directory: Path,
    stdout_path: Path,
    stderr_path: Path,
    throttle_interval: int,
    caffeinate: bool = False,
) -> dict[str, Any]:
    return {
        "Label": str(label),
        "ProgramArguments": _launchd_program_arguments(command, caffeinate=caffeinate),
        "WorkingDirectory": str(working_directory),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": max(10, int(throttle_interval or 300)),
        "StandardOutPath": str(stdout_path),
        "StandardErrorPath": str(stderr_path),
    }


def _plan_launchd_caffeinate(plan: Mapping[str, Any]) -> bool:
    power_assertion = (
        plan.get("launchd_power_assertion")
        if isinstance(plan.get("launchd_power_assertion"), Mapping)
        else {}
    )
    return bool(power_assertion.get("enabled"))


def write_launchd_plist(path: Path, *, plan: Mapping[str, Any], label: str, throttle_interval: int) -> dict[str, Any]:
    output_dir = Path(str(plan["output_dir"]))
    paths = plan.get("paths") if isinstance(plan.get("paths"), Mapping) else {}
    payload = launchd_plist_payload(
        label=label,
        command=plan["commands"]["supervisor"],
        working_directory=REPO_ROOT,
        stdout_path=Path(str(paths.get("launchd_stdout") or output_dir / LAUNCHD_STDOUT_NAME)),
        stderr_path=Path(str(paths.get("launchd_stderr") or output_dir / LAUNCHD_STDERR_NAME)),
        throttle_interval=throttle_interval,
        caffeinate=_plan_launchd_caffeinate(plan),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        plistlib.dump(payload, handle, sort_keys=True)
    return payload


def write_watchdog_launchd_plist(
    path: Path,
    *,
    plan: Mapping[str, Any],
    label: str,
    throttle_interval: int,
) -> dict[str, Any]:
    output_dir = Path(str(plan["output_dir"]))
    paths = plan.get("paths") if isinstance(plan.get("paths"), Mapping) else {}
    payload = launchd_plist_payload(
        label=label,
        command=plan["commands"]["watchdog"],
        working_directory=REPO_ROOT,
        stdout_path=Path(str(paths.get("watchdog_launchd_stdout") or output_dir / WATCHDOG_LAUNCHD_STDOUT_NAME)),
        stderr_path=Path(str(paths.get("watchdog_launchd_stderr") or output_dir / WATCHDOG_LAUNCHD_STDERR_NAME)),
        throttle_interval=throttle_interval,
        caffeinate=_plan_launchd_caffeinate(plan),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        plistlib.dump(payload, handle, sort_keys=True)
    return payload


def _run_launchctl_command(
    command: Sequence[str],
    *,
    timeout_seconds: float = DEFAULT_LAUNCHCTL_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in command],
        capture_output=True,
        check=False,
        text=True,
        timeout=max(1.0, float(timeout_seconds or DEFAULT_LAUNCHCTL_TIMEOUT_SECONDS)),
    )


def install_launchd_plists(plan: Mapping[str, Any]) -> dict[str, Any]:
    launchd_install = plan.get("launchd_install") if isinstance(plan.get("launchd_install"), Mapping) else {}
    roles = launchd_install.get("roles") if isinstance(launchd_install.get("roles"), Mapping) else {}
    report: dict[str, Any] = {
        "status": "ok",
        "requested": bool(launchd_install.get("requested")),
        "domain": str(launchd_install.get("domain") or ""),
        "installed_at": _now_iso(),
        "steps": [],
    }
    try:
        launchctl_timeout = max(
            1.0,
            float(launchd_install.get("timeout_seconds") or DEFAULT_LAUNCHCTL_TIMEOUT_SECONDS),
        )
    except (TypeError, ValueError):
        launchctl_timeout = DEFAULT_LAUNCHCTL_TIMEOUT_SECONDS
    report["timeout_seconds"] = launchctl_timeout
    if not report["requested"]:
        report.update({"status": "skipped", "detail": "launchd installation was not requested"})
        return report
    # Stop the old watchdog first so it cannot rebootstrap an old supervisor
    # plist while a launchd-backed adoption replaces the supervisor service.
    for role_name in ("watchdog", "supervisor"):
        role = roles.get(role_name)
        if not isinstance(role, Mapping):
            report["status"] = "error"
            report.setdefault("missing_roles", []).append(role_name)
            continue
        commands = role.get("commands") if isinstance(role.get("commands"), Sequence) else []
        for step in commands:
            if not isinstance(step, Mapping):
                continue
            command = [str(part) for part in (step.get("command") or [])]
            required = bool(step.get("required"))
            step_payload: dict[str, Any] = {
                "role": role_name,
                "name": str(step.get("name") or ""),
                "required": required,
                "command": command,
                "timeout_seconds": launchctl_timeout,
            }
            try:
                completed = _run_launchctl_command(command, timeout_seconds=launchctl_timeout)
            except subprocess.TimeoutExpired as exc:
                step_payload.update({
                    "returncode": None,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "stdout": str(exc.stdout or "")[-4000:],
                    "stderr": str(exc.stderr or "")[-4000:],
                    "timeout_seconds": launchctl_timeout,
                })
                if required:
                    report["status"] = "error"
                report["steps"].append(step_payload)
                if required:
                    break
                continue
            except Exception as exc:  # noqa: BLE001
                step_payload.update({
                    "returncode": None,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "stderr": str(exc),
                })
                if required:
                    report["status"] = "error"
                report["steps"].append(step_payload)
                if required:
                    break
                continue
            step_payload.update({
                "returncode": completed.returncode,
                "stdout": (completed.stdout or "")[-4000:],
                "stderr": (completed.stderr or "")[-4000:],
                "status": "ok" if completed.returncode == 0 or not required else "error",
            })
            if completed.returncode != 0 and required:
                report["status"] = "error"
            report["steps"].append(step_payload)
            if completed.returncode != 0 and required:
                break
    if report["status"] == "ok":
        report["detail"] = "supervisor and watchdog LaunchAgents were bootstrapped and verified"
    else:
        report["detail"] = "one or more required launchctl steps failed"
    return report


def _readiness_with_launchd_install_result(
    readiness_report: Mapping[str, Any],
    install_report: Mapping[str, Any],
) -> dict[str, Any]:
    updated = dict(readiness_report)
    checks = [
        dict(check)
        for check in (updated.get("checks") or [])
        if isinstance(check, Mapping)
    ]
    install_status = str(install_report.get("status") or "error")
    _add_readiness_check(
        checks,
        "launchd_install_result",
        "ok" if install_status == "ok" else "error",
        str(install_report.get("detail") or f"launchd installation status is {install_status}"),
    )
    status = _readiness_status(checks)
    updated["checks"] = checks
    updated["status"] = status
    updated["ready_for_10k_set_and_forget"] = status == "ok"
    updated["launchd_install_result"] = dict(install_report)
    summary = dict(updated.get("summary") or {})
    summary["launchd_install_status"] = install_status
    updated["summary"] = summary
    return updated


def _readiness_with_post_install_operation_audit_result(
    readiness_report: Mapping[str, Any],
    audit_report: Mapping[str, Any],
    *,
    required: bool,
) -> dict[str, Any]:
    updated = dict(readiness_report)
    checks = [
        dict(check)
        for check in (updated.get("checks") or [])
        if isinstance(check, Mapping)
    ]
    audit_status = str(audit_report.get("status") or "error").lower()
    _add_readiness_check(
        checks,
        "post_install_operation_audit",
        "ok" if audit_status == "ok" else ("error" if required else "warn"),
        (
            "post-install live operation audit passed"
            if audit_status == "ok"
            else f"post-install live operation audit status is {audit_status}"
        ),
        audit_status=audit_status,
        strict_set_and_forget=bool(audit_report.get("strict_set_and_forget")),
        poll=audit_report.get("post_install_poll") if isinstance(audit_report.get("post_install_poll"), Mapping) else {},
    )
    status = _readiness_status(checks)
    updated["checks"] = checks
    updated["status"] = status
    updated["ready_for_10k_set_and_forget"] = status == "ok"
    updated["post_install_operation_audit"] = dict(audit_report)
    summary = dict(updated.get("summary") or {})
    summary["post_install_operation_audit_status"] = audit_status
    updated["summary"] = summary
    return updated


def run_post_install_operation_audit(
    *,
    runbook_path: Path,
    plan: Mapping[str, Any],
    wrapper_args: argparse.Namespace,
) -> dict[str, Any]:
    """Poll the strict live operation audit after launchd handoff."""
    from tools import audit_qwen_caption_operation as operation_audit  # noqa: PLC0415

    timeout_seconds = _post_install_operation_audit_timeout(wrapper_args)
    interval_seconds = _post_install_operation_audit_interval(wrapper_args)
    launchctl_timeout = max(
        1.0,
        _safe_float(
            getattr(wrapper_args, "launchctl_timeout", DEFAULT_LAUNCHCTL_TIMEOUT_SECONDS),
            DEFAULT_LAUNCHCTL_TIMEOUT_SECONDS,
        ),
    )
    paths = plan.get("paths") if isinstance(plan.get("paths"), Mapping) else {}
    output_raw = paths.get("operational_audit_json")
    output_path = Path(str(output_raw)) if output_raw else None
    started_monotonic = time.monotonic()
    deadline = started_monotonic + timeout_seconds
    attempt = 0
    last_report: dict[str, Any] = {}
    while True:
        attempt += 1
        report = operation_audit.audit_operation(
            runbook_path,
            allow_running_incomplete=True,
            strict_set_and_forget=True,
            launchctl_bin=str(getattr(wrapper_args, "launchctl", "launchctl") or "launchctl"),
            command_timeout_seconds=launchctl_timeout,
        )
        elapsed = time.monotonic() - started_monotonic
        report = dict(report)
        report["post_install_poll"] = {
            "status": "ok" if str(report.get("status") or "").lower() == "ok" else "polling",
            "attempts": attempt,
            "elapsed_seconds": elapsed,
            "timeout_seconds": timeout_seconds,
            "interval_seconds": interval_seconds,
        }
        last_report = report
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        if str(report.get("status") or "").lower() == "ok":
            return report
        if time.monotonic() >= deadline:
            break
        time.sleep(min(interval_seconds, max(0.0, deadline - time.monotonic())))

    last_report = dict(last_report or {
        "schema_version": 1,
        "status": "error",
        "checked_at": _now_iso(),
        "runbook_path": str(runbook_path),
        "checks": [],
    })
    poll = dict(last_report.get("post_install_poll") if isinstance(last_report.get("post_install_poll"), Mapping) else {})
    poll.update({
        "status": "timeout",
        "attempts": attempt,
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "timeout_seconds": timeout_seconds,
        "interval_seconds": interval_seconds,
    })
    last_report["post_install_poll"] = poll
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(last_report, indent=2, sort_keys=True), encoding="utf-8")
    return last_report


def main(argv: Sequence[str] | None = None) -> int:
    wrapper_args, supervisor_argv = parse_wrapper_args(argv)
    supervisor_args, supervisor_extra_args = parse_supervisor_args(supervisor_argv)
    if bool(getattr(wrapper_args, "tenk_set_and_forget", False)):
        supervisor_args.set_and_forget = True
    supervise.resolve_supervisor_set_and_forget_thresholds(supervisor_args)
    supervisor_args.output_dir = supervisor_args.output_dir.expanduser().resolve(strict=False)
    plan = build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )
    preflight_report: Mapping[str, Any] | None = None
    if not wrapper_args.no_preflight:
        preflight_report = preflight.preflight_soak(supervise._preflight_args(supervisor_args))
        if _preflight_error_can_defer_to_supervisor(preflight_report):
            preflight_report = {
                **dict(preflight_report),
                "deferred_to_supervisor": True,
                "deferred_reason": "live_runner_lock",
            }
    preflight_status = str((preflight_report or {}).get("status") or "ok").lower()
    preflight_deferred_to_supervisor = _preflight_error_can_defer_to_supervisor(preflight_report)
    preflight_blocks_launch = preflight_status == "error" and not preflight_deferred_to_supervisor
    static_gate_report = build_static_tenk_launch_gate_report(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        plan=plan,
    )
    static_gate_status = str(static_gate_report.get("status") or "ok").lower()
    static_blocks_launch = (
        bool(getattr(wrapper_args, "tenk_set_and_forget", False))
        and static_gate_status != "ok"
    )
    supervisor_drill_report: Mapping[str, Any] | None = None
    watchdog_drill_report: Mapping[str, Any] | None = None
    pilot_generation_report: Mapping[str, Any] | None = None
    certification_report: Mapping[str, Any] | None = None
    live_adoption_report: Mapping[str, Any] | None = None
    if not preflight_blocks_launch and not static_blocks_launch:
        supervisor_drill_report = _run_supervisor_drill(
            wrapper_args=wrapper_args,
            output_dir=supervisor_args.output_dir,
        )
        drill_status = str((supervisor_drill_report or {}).get("status") or "error").lower()
        if drill_status in {"ok", "skipped"}:
            watchdog_drill_report = _run_watchdog_drill(
                wrapper_args=wrapper_args,
                output_dir=supervisor_args.output_dir,
            )
        watchdog_drill_status = str((watchdog_drill_report or {}).get("status") or "error").lower()
        if drill_status in {"ok", "skipped"} and watchdog_drill_status in {"ok", "skipped"}:
            pilot_generation_report = _run_created_pilot(
                wrapper_args=wrapper_args,
                supervisor_args=supervisor_args,
                supervisor_extra_args=supervisor_extra_args,
                output_dir=supervisor_args.output_dir,
            )
            pilot_generation_status = str((pilot_generation_report or {}).get("status") or "ok").lower()
            if pilot_generation_status == "ok" or (
                pilot_generation_report is None and not wrapper_args.create_pilot_output_dir
            ):
                certification_report = _run_required_pilot_certification(
                    wrapper_args=wrapper_args,
                    supervisor_args=supervisor_args,
                    output_dir=supervisor_args.output_dir,
                )
        live_adoption_report = _run_live_adoption_certification(
            wrapper_args=wrapper_args,
            supervisor_args=supervisor_args,
            preflight_report=preflight_report,
            output_dir=supervisor_args.output_dir,
        )
    drill_status = str((supervisor_drill_report or {}).get("status") or "ok").lower()
    watchdog_drill_status = str((watchdog_drill_report or {}).get("status") or "ok").lower()
    pilot_generation_status = str((pilot_generation_report or {}).get("status") or "ok").lower()
    certification_status = str((certification_report or {}).get("status") or "ok").lower()
    live_adoption_status = str((live_adoption_report or {}).get("status") or "ok").lower()
    live_adoption_required = _live_adoption_certification_required(wrapper_args, preflight_report)
    launch_blocked = (
        preflight_blocks_launch
        or static_blocks_launch
        or drill_status not in {"ok", "skipped"}
        or watchdog_drill_status not in {"ok", "skipped"}
        or pilot_generation_status not in {"ok", "skipped"}
        or certification_status != "ok"
        or (live_adoption_required and live_adoption_status != "ok")
    )
    readiness_report = build_readiness_report(
        plan=plan,
        wrapper_args=wrapper_args,
        preflight_report=preflight_report,
        supervisor_drill_report=supervisor_drill_report,
        watchdog_drill_report=watchdog_drill_report,
        pilot_generation_report=pilot_generation_report,
        certification_report=certification_report,
        live_adoption_report=live_adoption_report,
        static_gate_report=static_gate_report,
        launch_blocked=launch_blocked,
    )
    readiness_path = Path(str(plan["paths"]["readiness_json"]))
    write_readiness_report(readiness_path, readiness_report)
    runbook_path = (
        wrapper_args.runbook_json.expanduser().resolve(strict=False)
        if wrapper_args.runbook_json
        else supervisor_args.output_dir / RUNBOOK_NAME
    )
    write_runbook(
        path=runbook_path,
        plan=plan,
        preflight_report=preflight_report,
        static_gate_report=static_gate_report,
        supervisor_drill_report=supervisor_drill_report,
        watchdog_drill_report=watchdog_drill_report,
        pilot_generation_report=pilot_generation_report,
        certification_report=certification_report,
        live_adoption_report=live_adoption_report,
        readiness_report=readiness_report,
    )
    if wrapper_args.write_launchd_plist:
        write_launchd_plist(
            wrapper_args.write_launchd_plist.expanduser().resolve(strict=False),
            plan=plan,
            label=wrapper_args.launchd_label,
            throttle_interval=wrapper_args.launchd_throttle_interval,
        )
    if wrapper_args.write_watchdog_launchd_plist:
        write_watchdog_launchd_plist(
            wrapper_args.write_watchdog_launchd_plist.expanduser().resolve(strict=False),
            plan=plan,
            label=wrapper_args.watchdog_launchd_label,
            throttle_interval=wrapper_args.watchdog_launchd_throttle_interval,
        )
    if wrapper_args.print_plan:
        printable = dict(plan)
        if preflight_report is not None:
            printable["preflight"] = dict(preflight_report)
        printable["tenk_static_launch_gate"] = dict(static_gate_report)
        if supervisor_drill_report is not None:
            printable["supervisor_drill"] = dict(supervisor_drill_report)
        if watchdog_drill_report is not None:
            printable["watchdog_drill"] = dict(watchdog_drill_report)
        if certification_report is not None:
            printable["required_pilot_certification"] = dict(certification_report)
        if live_adoption_report is not None:
            printable["live_adoption_certification"] = dict(live_adoption_report)
        printable["readiness"] = dict(readiness_report)
        print(json.dumps(printable, indent=2, sort_keys=True), flush=True)
    if preflight_blocks_launch:
        return supervise.TERMINAL_PRECHECK_FAILED
    if static_blocks_launch:
        return supervise.TERMINAL_PRECHECK_FAILED
    if drill_status not in {"ok", "skipped"}:
        return supervise.TERMINAL_PRECHECK_FAILED
    if watchdog_drill_status not in {"ok", "skipped"}:
        return supervise.TERMINAL_PRECHECK_FAILED
    if pilot_generation_status not in {"ok", "skipped"}:
        return supervise.TERMINAL_PRECHECK_FAILED
    if certification_status != "ok":
        return supervise.TERMINAL_PRECHECK_FAILED
    if live_adoption_required and live_adoption_status != "ok":
        return supervise.TERMINAL_PRECHECK_FAILED
    if bool(getattr(wrapper_args, "require_readiness_ok", False)) and str(readiness_report.get("status") or "error") != "ok":
        return supervise.TERMINAL_PRECHECK_FAILED
    if wrapper_args.dry_run:
        return 0
    if bool(getattr(wrapper_args, "install_launchd_plists", False)):
        launchd_install_report = install_launchd_plists(plan)
        readiness_report = _readiness_with_launchd_install_result(
            readiness_report,
            launchd_install_report,
        )
        write_readiness_report(readiness_path, readiness_report)
        write_runbook(
            path=runbook_path,
            plan=plan,
            preflight_report=preflight_report,
            static_gate_report=static_gate_report,
            supervisor_drill_report=supervisor_drill_report,
            watchdog_drill_report=watchdog_drill_report,
            pilot_generation_report=pilot_generation_report,
            certification_report=certification_report,
            live_adoption_report=live_adoption_report,
            readiness_report=readiness_report,
            launchd_install_report=launchd_install_report,
        )
        if str(launchd_install_report.get("status") or "error") != "ok":
            return supervise.TERMINAL_PRECHECK_FAILED
        post_install_operation_audit_report: Mapping[str, Any] | None = None
        if _post_install_operation_audit_required(wrapper_args):
            post_install_operation_audit_report = run_post_install_operation_audit(
                runbook_path=runbook_path,
                plan=plan,
                wrapper_args=wrapper_args,
            )
            readiness_report = _readiness_with_post_install_operation_audit_result(
                readiness_report,
                post_install_operation_audit_report,
                required=True,
            )
            write_readiness_report(readiness_path, readiness_report)
            write_runbook(
                path=runbook_path,
                plan=plan,
                preflight_report=preflight_report,
                static_gate_report=static_gate_report,
                supervisor_drill_report=supervisor_drill_report,
                watchdog_drill_report=watchdog_drill_report,
                pilot_generation_report=pilot_generation_report,
                certification_report=certification_report,
                live_adoption_report=live_adoption_report,
                readiness_report=readiness_report,
                launchd_install_report=launchd_install_report,
                post_install_operation_audit_report=post_install_operation_audit_report,
            )
            if str(post_install_operation_audit_report.get("status") or "error") != "ok":
                return supervise.TERMINAL_PRECHECK_FAILED
        return 0
    return supervise.supervise_soak(supervisor_args, runner_extra_args=supervisor_extra_args)


if __name__ == "__main__":
    raise SystemExit(main())
