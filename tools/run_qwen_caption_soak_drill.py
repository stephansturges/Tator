#!/usr/bin/env python3
"""Run a deterministic no-GPU drill for the Qwen caption soak supervisor."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import audit_qwen_caption_soak as audit  # noqa: E402
from tools import supervise_qwen_caption_soak as supervise  # noqa: E402
from tools import watch_qwen_caption_soak as watch  # noqa: E402


DEFAULT_BASE_DIR = REPO_ROOT / "tmp" / "qwen_caption_benchmark" / "soak_drill"


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _write_sized_file(path: Path, text: str, *, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))


def _case_payload(root: Path, case_count: int = 1) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index in range(max(1, int(case_count or 1))):
        stem = f"frame_{index + 1:06d}"
        cases.append(
            {
                "name": f"image_{index + 1:06d}",
                "stem": stem,
                "caption_mode": "full",
                "image_path": str(root / f"{stem}.jpg"),
                "label_count": 0,
                "class_counts": {},
            }
        )
    return cases


def _write_fake_runner(path: Path, state_path: Path, *, include_missing_heartbeat: bool) -> None:
    missing_branch = """
if run_number == 4:
    (out / "missing_heartbeat_started.txt").write_text("started")
    time.sleep(30)
    sys.exit(88)
"""
    success_run_number = 5 if include_missing_heartbeat else 4
    path.write_text(
        f"""
import argparse
import json
import os
import pathlib
import signal
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir")
parser.add_argument("--cases-json")
args, _ = parser.parse_known_args()
out = pathlib.Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
state_path = pathlib.Path({str(state_path)!r})
state_path.parent.mkdir(parents=True, exist_ok=True)
state = {{}}
if state_path.exists():
    state = json.loads(state_path.read_text())
run_number = int(state.get("run_number") or 0) + 1
state["run_number"] = run_number
state_path.write_text(json.dumps(state))
cases = json.loads(pathlib.Path(args.cases_json).read_text())

if run_number == 1:
    sys.exit(7)

if run_number == 2:
    os.kill(os.getpid(), signal.SIGABRT)

(out / "manifest.json").write_text(json.dumps({{"cases": cases}}))

if run_number == 3:
    (out / "heartbeat.json").write_text(json.dumps({{
        "status": "running",
        "phase": "attempt_running",
        "heartbeat_epoch": time.time() - 1000
    }}))
    time.sleep(30)
    sys.exit(99)

{missing_branch if include_missing_heartbeat else ""}

if run_number < {success_run_number}:
    sys.exit(89)

(out / "results.jsonl").write_text(json.dumps({{
    "case_id": "image_000001:frame:full",
    "status": "preview_only",
    "final_status": "ok",
    "quality_failures": [],
    "elapsed_seconds": 0.01
}}) + "\\n")
(out / "summary.json").write_text(json.dumps({{"total_cases": 1, "totals": {{"ok": 1}}}}))
(out / "heartbeat.json").write_text(json.dumps({{
    "status": "completed",
    "phase": "finished",
    "heartbeat_epoch": time.time()
}}))
sys.exit(0)
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_endurance_fake_runner(
    path: Path,
    state_path: Path,
    *,
    chunk_size: int,
    include_missing_heartbeat: bool,
    summary_row_limit: int,
) -> None:
    missing_branch = """
if run_number == 4:
    (out / "missing_heartbeat_started.txt").write_text("started")
    time.sleep(30)
    sys.exit(88)
"""
    path.write_text(
        f"""
import argparse
import collections
import json
import os
import pathlib
import signal
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir")
parser.add_argument("--cases-json")
args, _ = parser.parse_known_args()
out = pathlib.Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
state_path = pathlib.Path({str(state_path)!r})
state_path.parent.mkdir(parents=True, exist_ok=True)
state = {{}}
if state_path.exists():
    state = json.loads(state_path.read_text())
run_number = int(state.get("run_number") or 0) + 1
state["run_number"] = run_number
state_path.write_text(json.dumps(state))
cases = json.loads(pathlib.Path(args.cases_json).read_text())

(out / "manifest.json").write_text(json.dumps({{"cases": cases}}))

if run_number == 2:
    os.kill(os.getpid(), signal.SIGABRT)

if run_number == 3:
    (out / "heartbeat.json").write_text(json.dumps({{
        "status": "running",
        "phase": "attempt_running",
        "heartbeat_epoch": time.time() - 1000
    }}))
    time.sleep(30)
    sys.exit(99)

{missing_branch if include_missing_heartbeat else ""}

results_path = out / "results.jsonl"

def write_heartbeat(processed, *, status="running", phase="attempt_running"):
    (out / "heartbeat.json").write_text(json.dumps({{
        "status": status,
        "phase": phase,
        "processed": int(processed),
        "total_cases": len(cases),
        "heartbeat_epoch": time.time()
    }}))

completed = set()
if results_path.exists():
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("final_status") in {{"ok", "preview_only", "skipped_completed", "skipped_existing_caption"}}:
            key = str(row.get("case_id") or "")
            if key:
                completed.add(key)
write_heartbeat(len(completed))

chunk = max(1, int({int(max(1, chunk_size))!r}))
remaining = []
for case in cases:
    case_id = f"{{case.get('name')}}:{{case.get('stem')}}:{{case.get('caption_mode') or 'full'}}"
    if case_id not in completed:
        remaining.append((case_id, case))

batch = remaining[:chunk]
processed_now = len(completed)
with results_path.open("a") as handle:
    for offset, (case_id, case) in enumerate(batch, start=1):
        handle.write(json.dumps({{
            "case_id": case_id,
            "case": case.get("name"),
            "stem": case.get("stem"),
            "status": "preview_only",
            "final_status": "ok",
            "quality_failures": [],
            "elapsed_seconds": 0.01,
            "preview_prompt_budget": {{
                "max_prompt_tokens": 1200,
                "adapted_sections": 0
            }}
        }}) + "\\n")
        processed_now += 1
        if offset % 100 == 0:
            write_heartbeat(processed_now)
write_heartbeat(processed_now)

rows = []
if results_path.exists():
    for line_index, line in enumerate(results_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if line_index % 1000 == 0:
            write_heartbeat(processed_now)
latest = {{}}
for row in rows:
    latest[str(row.get("case_id") or "")] = row
ordered = list(latest.values())
totals = collections.Counter(str(row.get("final_status") or row.get("status") or "unknown") for row in ordered)
summary_limit = int({int(summary_row_limit)!r})
if summary_limit < 0:
    summary_rows = ordered
    summary_policy = "all"
elif summary_limit == 0:
    summary_rows = []
    summary_policy = "none"
else:
    summary_rows = ordered[-summary_limit:]
    summary_policy = "latest"
summary_truncated = len(summary_rows) < len(ordered)
(out / "summary.json").write_text(json.dumps({{
    "total_cases": len(ordered),
    "failed_cases": sum(count for status, count in totals.items() if status not in {{"ok", "preview_only", "skipped_completed", "skipped_existing_caption"}}),
    "quality_failed_cases": sum(1 for row in ordered if row.get("quality_failures")),
    "totals": dict(totals),
    "row_count": len(ordered),
    "row_limit": summary_limit,
    "rows_truncated": summary_truncated,
    "rows_omitted": len(ordered) - len(summary_rows),
    "rows_sample_policy": summary_policy if summary_truncated else "all",
    "rows": summary_rows
}}))

done = len(ordered) >= len(cases)
(out / "heartbeat.json").write_text(json.dumps({{
    "status": "completed" if done else "running",
    "phase": "finished" if done else "attempt_running",
    "processed": len(ordered),
    "total_cases": len(cases),
    "heartbeat_epoch": time.time()
}}))
if run_number == 1 and not done:
    sys.exit(7)
sys.exit(0 if done else 42)
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_fake_launchctl(path: Path, artifact_dir: Path) -> None:
    path.write_text(
        f"""
#!/usr/bin/env python3
import json
import os
import pathlib
import sys
import time

artifact_dir = pathlib.Path({str(artifact_dir)!r})
artifact_dir.mkdir(parents=True, exist_ok=True)
log_path = artifact_dir / "fake_launchctl.jsonl"
command = sys.argv[1:]
with log_path.open("a") as handle:
    handle.write(json.dumps({{"command": command, "time": time.time()}}) + "\\n")

if command and command[0] == "kickstart":
    sys.stderr.write("fake kickstart failed; exercise rebootstrap path\\n")
    sys.exit(78)

if command and command[0] == "bootout":
    sys.stderr.write("fake bootout target was not loaded\\n")
    sys.exit(3)

if command and command[0] == "bootstrap":
    now = time.time()
    runner_pid = int(os.environ.get("TATOR_FAKE_RUNNER_PID") or os.getppid())
    (artifact_dir / "heartbeat.json").write_text(json.dumps({{
        "status": "running",
        "phase": "attempt_running",
        "heartbeat_epoch": now,
        "case": "watchdog_drill_restored",
        "case_index": 0,
        "attempt": 1,
        "attempt_started_epoch": now,
        "attempt_timeout_seconds": 120.0
    }}))
    (artifact_dir / ".runner.lock").write_text(json.dumps({{
        "runner_id": "watchdog-drill-restored",
        "pid": runner_pid,
        "phase": "attempt_running",
        "heartbeat_epoch": now
    }}))
    sys.stdout.write("fake bootstrap restored heartbeat\\n")
    sys.exit(0)

sys.exit(0)
""".lstrip(),
        encoding="utf-8",
    )
    path.chmod(0o755)


def _event_seen(events: list[Mapping[str, Any]], event_name: str) -> bool:
    return any(str(event.get("event") or "") == event_name for event in events)


def run_caption_io_retention_drill(
    output_dir: Path,
    *,
    force: bool = False,
    write_json: Path | None = None,
    max_files: int = 8,
    max_bytes: int = 560,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve(strict=False)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"caption IO retention drill output directory is not empty: {output_dir}; "
                "pass --force to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = write_json or (output_dir / "caption_io_retention_report.json")
    log_root = output_dir / "repo_logs"
    run_root = log_root / "qwen_caption_io"
    run_root.mkdir(parents=True, exist_ok=True)
    active_run_id = "active-retention-run"
    outside_target = output_dir / "outside_symlink_target.txt"
    outside_target.write_text("do not touch\n", encoding="utf-8")
    symlink_path = run_root / "outside.log"
    symlink_created = False
    try:
        symlink_path.symlink_to(outside_target)
        symlink_created = True
    except (AttributeError, NotImplementedError, OSError):
        symlink_created = False

    backend_import_stdout = io.StringIO()
    with contextlib.redirect_stdout(backend_import_stdout):
        qwen_api = importlib.import_module("localinferenceapi")
    previous_values = {
        "LOG_ROOT": getattr(qwen_api, "LOG_ROOT", None),
        "QWEN_CAPTION_IO_RUN_LOG_MAX_FILES": getattr(qwen_api, "QWEN_CAPTION_IO_RUN_LOG_MAX_FILES", None),
        "QWEN_CAPTION_IO_RUN_LOG_MAX_BYTES": getattr(qwen_api, "QWEN_CAPTION_IO_RUN_LOG_MAX_BYTES", None),
    }
    try:
        qwen_api.LOG_ROOT = log_root
        qwen_api.QWEN_CAPTION_IO_RUN_LOG_MAX_FILES = max(0, int(max_files or 0))
        qwen_api.QWEN_CAPTION_IO_RUN_LOG_MAX_BYTES = max(0, int(max_bytes or 0))
        active_jsonl, active_log, latest_jsonl, latest_log = qwen_api._qwen_caption_io_paths(active_run_id)
        now = time.time()
        _write_sized_file(active_jsonl, "active-jsonl\n" + ("A" * 95), mtime=now + 100)
        _write_sized_file(active_log, "active-log\n" + ("B" * 95), mtime=now + 101)
        for index in range(12):
            stamp = now - (100 - index)
            payload = f"inactive {index:02d}\n" + ("x" * 105)
            _write_sized_file(run_root / f"inactive_{index:02d}.jsonl", payload, mtime=stamp)
            _write_sized_file(run_root / f"inactive_{index:02d}.log", payload, mtime=stamp + 0.1)
        direct_before = [
            path
            for path in run_root.iterdir()
            if path.is_file() and not path.is_symlink() and path.suffix in {".jsonl", ".log"}
        ]
        qwen_api._qwen_caption_io_reset_latest(active_run_id)
    finally:
        for name, value in previous_values.items():
            if value is None and hasattr(qwen_api, name):
                delattr(qwen_api, name)
            elif value is not None:
                setattr(qwen_api, name, value)

    direct_after = [
        path
        for path in run_root.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix in {".jsonl", ".log"}
    ]
    total_after_bytes = sum(path.stat().st_size for path in direct_after)
    latest_jsonl_empty = latest_jsonl.exists() and latest_jsonl.read_text(encoding="utf-8") == ""
    latest_log_empty = latest_log.exists() and latest_log.read_text(encoding="utf-8") == ""
    symlink_target_unchanged = outside_target.read_text(encoding="utf-8") == "do not touch\n"
    checks = {
        "active_jsonl_kept": active_jsonl.exists(),
        "active_log_kept": active_log.exists(),
        "old_run_logs_pruned": len(direct_after) < len(direct_before),
        "file_cap_respected": len(direct_after) <= max(0, int(max_files or 0)),
        "byte_cap_respected": total_after_bytes <= max(0, int(max_bytes or 0)),
        "latest_jsonl_reset": latest_jsonl_empty,
        "latest_log_reset": latest_log_empty,
        "symlink_not_followed": symlink_target_unchanged and (not symlink_created or symlink_path.is_symlink()),
    }
    status = "ok" if all(checks.values()) else "error"
    report: dict[str, Any] = {
        "status": status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "log_root": str(log_root),
        "run_root": str(run_root),
        "active_run_id": active_run_id,
        "max_files": max(0, int(max_files or 0)),
        "max_bytes": max(0, int(max_bytes or 0)),
        "direct_files_before": len(direct_before),
        "direct_files_after": len(direct_after),
        "direct_bytes_after": total_after_bytes,
        "remaining_files": sorted(path.name for path in direct_after),
        "active_jsonl": str(active_jsonl),
        "active_log": str(active_log),
        "latest_jsonl": str(latest_jsonl),
        "latest_log": str(latest_log),
        "symlink_created": symlink_created,
        "symlink_path": str(symlink_path),
        "outside_target": str(outside_target),
        "backend_import_stdout": backend_import_stdout.getvalue(),
        "checks": checks,
    }
    _write_json(report_path, report)
    report["report_json"] = str(report_path)
    return report


def run_drill(
    output_dir: Path,
    *,
    force: bool = False,
    include_missing_heartbeat: bool = True,
    max_heartbeat_age: float = 0.05,
    heartbeat_startup_grace: float = 0.05,
    max_runner_restarts: int = 5,
    include_caption_io_retention: bool = True,
    write_json: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve(strict=False)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"drill output directory is not empty: {output_dir}; pass --force to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / "artifacts"
    cases_json = output_dir / "cases.json"
    fake_runner = output_dir / "fake_caption_runner.py"
    state_path = output_dir / "runner_state.json"
    log_path = output_dir / "supervisor.jsonl"
    supervisor_stdout_path = output_dir / "supervisor_stdout.log"
    report_path = write_json or (output_dir / "drill_report.json")
    _write_json(cases_json, _case_payload(output_dir))
    _write_fake_runner(fake_runner, state_path, include_missing_heartbeat=include_missing_heartbeat)

    parser = supervise.build_parser()
    supervise_args = parser.parse_args(
        [
            "--dataset-root",
            str(output_dir),
            "--cases-json",
            str(cases_json),
            "--output-dir",
            str(artifact_dir),
            "--runner-script",
            str(fake_runner),
            "--min-free-gb",
            "0",
            "--restart-delay",
            "0",
            "--monitor-interval",
            "0.01",
            "--max-heartbeat-age",
            str(max_heartbeat_age),
            "--heartbeat-startup-grace",
            str(heartbeat_startup_grace),
            "--max-runner-restarts",
            str(max_runner_restarts),
            "--kill-timeout",
            "0.2",
            "--log-jsonl",
            str(log_path),
            "--preview-only",
        ]
    )
    started_at = time.time()
    supervisor_stdout = io.StringIO()
    with contextlib.redirect_stdout(supervisor_stdout):
        supervisor_return_code = supervise.supervise_soak(supervise_args)
    supervisor_stdout_path.write_text(supervisor_stdout.getvalue(), encoding="utf-8")
    elapsed_seconds = time.time() - started_at
    events = _read_jsonl(log_path)
    final_audit = audit.audit_soak(
        artifact_dir,
        max_heartbeat_age_seconds=max_heartbeat_age,
        allow_running_incomplete=False,
    )
    checks = {
        "supervisor_success": supervisor_return_code == supervise.TERMINAL_SUCCESS,
        "saw_nonzero_exit": any(
            event.get("event") == "runner_exit" and event.get("status") == "nonzero_exit"
            for event in events
        ),
        "saw_signal_exit": any(
            event.get("event") == "runner_exit" and event.get("status") == "signal_exit"
            for event in events
        ),
        "saw_stale_heartbeat": _event_seen(events, "runner_stale_heartbeat"),
        "saw_missing_heartbeat": _event_seen(events, "runner_missing_heartbeat"),
        "saw_supervisor_restart": _event_seen(events, "supervisor_restart"),
        "saw_supervisor_complete": _event_seen(events, "supervisor_complete"),
        "final_audit_ok": str(final_audit.get("status") or "") == "ok",
    }
    if not include_missing_heartbeat:
        checks["saw_missing_heartbeat"] = True
    caption_io_retention_report = None
    if include_caption_io_retention:
        caption_io_retention_report = run_caption_io_retention_drill(
            output_dir / "caption_io_retention",
            force=True,
        )
        checks["caption_io_retention_ok"] = caption_io_retention_report.get("status") == "ok"
    else:
        checks["caption_io_retention_ok"] = True
    status = "ok" if all(checks.values()) else "error"
    report: dict[str, Any] = {
        "status": status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "output_dir": str(output_dir),
        "artifact_dir": str(artifact_dir),
        "supervisor_log": str(log_path),
        "supervisor_stdout": str(supervisor_stdout_path),
        "fake_runner": str(fake_runner),
        "cases_json": str(cases_json),
        "supervisor_return_code": supervisor_return_code,
        "final_audit_status": final_audit.get("status"),
        "checks": checks,
        "event_counts": {
            event_name: sum(1 for event in events if event.get("event") == event_name)
            for event_name in sorted({str(event.get("event") or "") for event in events})
        },
        "final_audit": final_audit,
        "caption_io_retention": caption_io_retention_report,
    }
    _write_json(report_path, report)
    report["report_json"] = str(report_path)
    return report


def run_endurance_drill(
    output_dir: Path,
    *,
    force: bool = False,
    case_count: int = 200,
    chunk_size: int = 25,
    include_missing_heartbeat: bool = True,
    max_heartbeat_age: float = 0.05,
    heartbeat_startup_grace: float = 0.05,
    summary_row_limit: int = 250,
    max_runner_restarts: int = 25,
    include_caption_io_retention: bool = True,
    write_json: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve(strict=False)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"drill output directory is not empty: {output_dir}; pass --force to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / "artifacts"
    cases_json = output_dir / "cases.json"
    fake_runner = output_dir / "fake_caption_runner.py"
    state_path = output_dir / "runner_state.json"
    log_path = output_dir / "supervisor.jsonl"
    supervisor_stdout_path = output_dir / "supervisor_stdout.log"
    report_path = write_json or (output_dir / "endurance_report.json")
    case_count = max(1, int(case_count or 1))
    chunk_size = max(1, int(chunk_size or 1))
    _write_json(cases_json, _case_payload(output_dir, case_count))
    _write_endurance_fake_runner(
        fake_runner,
        state_path,
        chunk_size=chunk_size,
        include_missing_heartbeat=include_missing_heartbeat,
        summary_row_limit=summary_row_limit,
    )

    expected_processing_runs = (case_count + chunk_size - 1) // chunk_size
    expected_hazard_runs = 4 if include_missing_heartbeat else 3
    required_restarts = expected_processing_runs + expected_hazard_runs - 1
    max_runner_restarts = max(int(max_runner_restarts or 0), required_restarts + 1)
    parser = supervise.build_parser()
    supervise_args = parser.parse_args(
        [
            "--dataset-root",
            str(output_dir),
            "--cases-json",
            str(cases_json),
            "--output-dir",
            str(artifact_dir),
            "--runner-script",
            str(fake_runner),
            "--min-free-gb",
            "0",
            "--restart-delay",
            "0",
            "--monitor-interval",
            "0.01",
            "--max-heartbeat-age",
            str(max_heartbeat_age),
            "--heartbeat-startup-grace",
            str(heartbeat_startup_grace),
            "--max-runner-restarts",
            str(max_runner_restarts),
            "--kill-timeout",
            "0.2",
            "--log-jsonl",
            str(log_path),
            "--preview-only",
        ]
    )
    started_at = time.time()
    supervisor_stdout = io.StringIO()
    with contextlib.redirect_stdout(supervisor_stdout):
        supervisor_return_code = supervise.supervise_soak(supervise_args)
    supervisor_stdout_path.write_text(supervisor_stdout.getvalue(), encoding="utf-8")
    elapsed_seconds = time.time() - started_at
    events = _read_jsonl(log_path)
    final_audit = audit.audit_soak(
        artifact_dir,
        max_heartbeat_age_seconds=max_heartbeat_age,
        allow_running_incomplete=False,
    )
    results_path = artifact_dir / "results.jsonl"
    summary_path = artifact_dir / "summary.json"
    summary_payload: dict[str, Any] = {}
    if summary_path.exists():
        try:
            loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded_summary, dict):
                summary_payload = loaded_summary
        except json.JSONDecodeError:
            summary_payload = {}
    summary_rows = summary_payload.get("rows") if isinstance(summary_payload.get("rows"), list) else []
    summary_row_count = int(summary_payload.get("row_count") or summary_payload.get("total_cases") or 0)
    summary_rows_omitted = int(summary_payload.get("rows_omitted") or 0)
    result_rows_written = 0
    if results_path.exists():
        result_rows_written = sum(1 for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip())
    result_bytes = results_path.stat().st_size if results_path.exists() else 0
    summary_bytes = summary_path.stat().st_size if summary_path.exists() else 0
    summary_metrics = {
        "summary_path": str(summary_path),
        "results_path": str(results_path),
        "summary_bytes": summary_bytes,
        "results_bytes": result_bytes,
        "summary_to_results_size_ratio": round(summary_bytes / result_bytes, 6) if result_bytes else 0.0,
        "result_rows_written": result_rows_written,
        "summary_row_count": summary_row_count,
        "summary_rows_in_snapshot": len(summary_rows),
        "summary_row_limit": summary_payload.get("row_limit"),
        "summary_rows_truncated": bool(summary_payload.get("rows_truncated")),
        "summary_rows_omitted": summary_rows_omitted,
        "summary_rows_sample_policy": summary_payload.get("rows_sample_policy"),
    }
    summary_limit_value = int(summary_row_limit)
    expected_summary_snapshot_limit = (
        case_count if summary_limit_value < 0 else max(0, summary_limit_value)
    )
    expected_summary_omitted = max(0, case_count - expected_summary_snapshot_limit)
    processed_cases = int(final_audit.get("processed_cases") or 0)
    expected_cases = int(final_audit.get("expected_cases") or 0)
    event_counts = {
        event_name: sum(1 for event in events if event.get("event") == event_name)
        for event_name in sorted({str(event.get("event") or "") for event in events})
    }
    checks = {
        "supervisor_success": supervisor_return_code == supervise.TERMINAL_SUCCESS,
        "saw_nonzero_exit": any(
            event.get("event") == "runner_exit" and event.get("status") == "nonzero_exit"
            for event in events
        ),
        "saw_signal_exit": any(
            event.get("event") == "runner_exit" and event.get("status") == "signal_exit"
            for event in events
        ),
        "saw_stale_heartbeat": _event_seen(events, "runner_stale_heartbeat"),
        "saw_missing_heartbeat": _event_seen(events, "runner_missing_heartbeat"),
        "saw_multiple_restarts": int(event_counts.get("supervisor_restart") or 0) >= 3,
        "final_audit_ok": str(final_audit.get("status") or "") == "ok",
        "all_cases_processed": processed_cases == case_count and expected_cases == case_count,
        "all_latest_rows_ok": dict(final_audit.get("totals") or {}).get("ok") == case_count,
        "prompt_budget_recorded": int((final_audit.get("degraded_rates") or {}).get("max_prompt_tokens") or 0) > 0,
        "summary_totals_cover_all_cases": summary_row_count == case_count
        and dict(summary_payload.get("totals") or {}).get("ok") == case_count,
        "summary_snapshot_bounded": len(summary_rows) <= expected_summary_snapshot_limit,
        "summary_truncated_when_over_limit": (
            expected_summary_omitted == 0
            or bool(summary_payload.get("rows_truncated"))
        ),
        "summary_omits_rows_when_over_limit": (
            summary_rows_omitted == expected_summary_omitted
        ),
    }
    if not include_missing_heartbeat:
        checks["saw_missing_heartbeat"] = True
    caption_io_retention_report = None
    if include_caption_io_retention:
        caption_io_retention_report = run_caption_io_retention_drill(
            output_dir / "caption_io_retention",
            force=True,
        )
        checks["caption_io_retention_ok"] = caption_io_retention_report.get("status") == "ok"
    else:
        checks["caption_io_retention_ok"] = True
    status = "ok" if all(checks.values()) else "error"
    report: dict[str, Any] = {
        "status": status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "output_dir": str(output_dir),
        "artifact_dir": str(artifact_dir),
        "supervisor_log": str(log_path),
        "supervisor_stdout": str(supervisor_stdout_path),
        "fake_runner": str(fake_runner),
        "cases_json": str(cases_json),
        "case_count": case_count,
        "chunk_size": chunk_size,
        "expected_processing_runs": expected_processing_runs,
        "required_restarts": required_restarts,
        "max_runner_restarts": max_runner_restarts,
        "summary_metrics": summary_metrics,
        "supervisor_return_code": supervisor_return_code,
        "final_audit_status": final_audit.get("status"),
        "checks": checks,
        "event_counts": event_counts,
        "final_audit": final_audit,
        "caption_io_retention": caption_io_retention_report,
    }
    _write_json(report_path, report)
    report["report_json"] = str(report_path)
    return report


def run_watchdog_remediation_drill(
    output_dir: Path,
    *,
    force: bool = False,
    write_json: Path | None = None,
    max_heartbeat_age: float = 0.05,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve(strict=False)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"drill output directory is not empty: {output_dir}; pass --force to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = write_json or (output_dir / "watchdog_remediation_report.json")
    watchdog_log_path = output_dir / "watchdog.jsonl"
    watchdog_stdout_path = output_dir / "watchdog_stdout.log"
    latest_path = output_dir / "watchdog_latest.json"
    state_path = output_dir / "watchdog_state.json"
    fake_launchctl = output_dir / "fake_launchctl.py"
    supervisor_plist = output_dir / "com.example.supervisor.plist"
    supervisor_plist.write_text("fake supervisor plist\n", encoding="utf-8")
    _write_json(
        artifact_dir / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    stale_epoch = time.time() - max(10.0, float(max_heartbeat_age or 0.05) * 100.0)
    _write_json(
        artifact_dir / "heartbeat.json",
        {
            "status": "running",
            "phase": "attempt_running",
            "heartbeat_epoch": stale_epoch,
            "case": "watchdog_drill_stale",
            "case_index": 0,
            "attempt": 1,
            "attempt_started_epoch": stale_epoch,
            "attempt_timeout_seconds": 120.0,
        },
    )
    _write_json(
        artifact_dir / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "watchdog-drill-stale",
            "pid": os.getpid(),
            "phase": "attempt_running",
            "heartbeat_epoch": stale_epoch,
        },
    )
    _write_fake_launchctl(fake_launchctl, artifact_dir)

    previous_runner_pid = os.environ.get("TATOR_FAKE_RUNNER_PID")
    os.environ["TATOR_FAKE_RUNNER_PID"] = str(os.getpid())
    started_at = time.time()
    watchdog_stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(watchdog_stdout):
            watchdog_return_code = watch.watch_soak(
                artifact_dir,
                log_jsonl=watchdog_log_path,
                latest_json=latest_path,
                state_json=state_path,
                interval_seconds=0,
                max_heartbeat_age_seconds=max_heartbeat_age,
                max_consecutive_unhealthy=1,
                max_checks=2,
                remediate_launchd_label="com.example.supervisor",
                remediate_launchd_domain="gui/501",
                remediate_launchd_plist=supervisor_plist,
                remediate_launchctl=str(fake_launchctl),
                max_remediations=1,
                remediation_cooldown_seconds=0,
            )
    finally:
        if previous_runner_pid is None:
            os.environ.pop("TATOR_FAKE_RUNNER_PID", None)
        else:
            os.environ["TATOR_FAKE_RUNNER_PID"] = previous_runner_pid
    watchdog_stdout_path.write_text(watchdog_stdout.getvalue(), encoding="utf-8")
    elapsed_seconds = time.time() - started_at
    events = _read_jsonl(watchdog_log_path)
    fake_launchctl_events = _read_jsonl(artifact_dir / "fake_launchctl.jsonl")
    latest = {}
    if latest_path.exists():
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            latest = {}
    state = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    remediation = events[0].get("remediation") if events else {}
    remediation = remediation if isinstance(remediation, Mapping) else {}
    checks = {
        "watchdog_success": watchdog_return_code == 0,
        "saw_unhealthy_status": bool(events) and str(events[0].get("status") or "") == "error",
        "saw_launchd_rebootstrap": str(remediation.get("action") or "") == "launchd_rebootstrap",
        "saw_kickstart_failure": (remediation.get("kickstart") or {}).get("returncode") == 78
        if isinstance(remediation.get("kickstart"), Mapping)
        else False,
        "saw_bootstrap_success": (remediation.get("bootstrap") or {}).get("returncode") == 0
        if isinstance(remediation.get("bootstrap"), Mapping)
        else False,
        "saw_restored_health": len(events) >= 2 and str(events[-1].get("status") or "") == "ok",
        "history_is_compact": bool(events) and all(str(event.get("event_detail") or "") == "compact" for event in events),
        "latest_is_full": str(latest.get("event_detail") or "") == "full",
        "latest_status_ok": str(latest.get("status") or "") == "ok",
        "state_persisted": int(state.get("remediation_count") or 0) == 1
        and int(state.get("consecutive_unhealthy") or 0) == 0,
    }
    status = "ok" if all(checks.values()) else "error"
    report: dict[str, Any] = {
        "status": status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "output_dir": str(output_dir),
        "artifact_dir": str(artifact_dir),
        "watchdog_log": str(watchdog_log_path),
        "watchdog_stdout": str(watchdog_stdout_path),
        "watchdog_latest": str(latest_path),
        "watchdog_state": str(state_path),
        "fake_launchctl": str(fake_launchctl),
        "fake_launchctl_log": str(artifact_dir / "fake_launchctl.jsonl"),
        "supervisor_plist": str(supervisor_plist),
        "watchdog_return_code": watchdog_return_code,
        "checks": checks,
        "event_counts": {
            str(event.get("event") or ""): sum(
                1 for candidate in events if candidate.get("event") == event.get("event")
            )
            for event in events
        },
        "launchctl_commands": [event.get("command") for event in fake_launchctl_events],
        "latest_status": latest,
        "watchdog_state_payload": state,
    }
    _write_json(report_path, report)
    report["report_json"] = str(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for drill artifacts. Defaults to a timestamped tmp/qwen_caption_benchmark/soak_drill child.",
    )
    parser.add_argument("--force", action="store_true", help="Replace a non-empty drill output directory.")
    parser.add_argument(
        "--skip-missing-heartbeat-phase",
        action="store_true",
        help="Exercise nonzero and stale-heartbeat restarts only.",
    )
    parser.add_argument("--max-heartbeat-age", type=float, default=0.05)
    parser.add_argument(
        "--heartbeat-startup-grace",
        type=float,
        default=0.05,
        help="Accelerated first-heartbeat grace used by the synthetic supervisor drill.",
    )
    parser.add_argument("--max-runner-restarts", type=int, default=5)
    parser.add_argument(
        "--endurance-cases",
        type=int,
        default=0,
        help="Run the multi-case no-GPU endurance drill with this many synthetic cases.",
    )
    parser.add_argument(
        "--endurance-chunk-size",
        type=int,
        default=25,
        help="Synthetic cases completed per fake-runner invocation in endurance mode.",
    )
    parser.add_argument(
        "--summary-row-limit",
        type=int,
        default=250,
        help="Maximum latest rows copied into the synthetic endurance summary.json; -1 keeps all rows.",
    )
    parser.add_argument(
        "--watchdog-remediation",
        action="store_true",
        help="Run the no-GPU watchdog launchd-remediation drill instead of the supervisor drill.",
    )
    parser.add_argument(
        "--caption-io-retention-only",
        action="store_true",
        help="Run only the global qwen_caption_io retention drill.",
    )
    parser.add_argument(
        "--skip-caption-io-retention-drill",
        action="store_true",
        help="Do not run the qwen_caption_io retention check after supervisor or endurance drills.",
    )
    parser.add_argument("--write-json", type=Path, default=None)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir or (DEFAULT_BASE_DIR / _now_stamp())
    try:
        if bool(args.caption_io_retention_only):
            report = run_caption_io_retention_drill(
                output_dir,
                force=bool(args.force),
                write_json=args.write_json,
            )
        elif bool(args.watchdog_remediation):
            report = run_watchdog_remediation_drill(
                output_dir,
                force=bool(args.force),
                max_heartbeat_age=float(args.max_heartbeat_age),
                write_json=args.write_json,
            )
        elif int(args.endurance_cases or 0) > 0:
            report = run_endurance_drill(
                output_dir,
                force=bool(args.force),
                case_count=int(args.endurance_cases),
                chunk_size=int(args.endurance_chunk_size),
                include_missing_heartbeat=not bool(args.skip_missing_heartbeat_phase),
                max_heartbeat_age=float(args.max_heartbeat_age),
                heartbeat_startup_grace=float(args.heartbeat_startup_grace),
                summary_row_limit=int(args.summary_row_limit),
                max_runner_restarts=int(args.max_runner_restarts),
                include_caption_io_retention=not bool(args.skip_caption_io_retention_drill),
                write_json=args.write_json,
            )
        else:
            report = run_drill(
                output_dir,
                force=bool(args.force),
                include_missing_heartbeat=not bool(args.skip_missing_heartbeat_phase),
                max_heartbeat_age=float(args.max_heartbeat_age),
                heartbeat_startup_grace=float(args.heartbeat_startup_grace),
                max_runner_restarts=int(args.max_runner_restarts),
                include_caption_io_retention=not bool(args.skip_caption_io_retention_drill),
                write_json=args.write_json,
            )
    except Exception as exc:  # noqa: BLE001
        report = {
            "status": "error",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        if args.write_json:
            _write_json(args.write_json, report)
        print(json.dumps(report, indent=2 if args.pretty else None, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
