from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from tools import supervise_qwen_caption_soak as supervise


ROOT = Path(__file__).resolve().parents[1]


def _cases_json(tmp_path: Path) -> Path:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps([
            {
                "name": "image_000001",
                "stem": "frame",
                "caption_mode": "full",
                "image_path": str(tmp_path / "frame.jpg"),
                "label_count": 0,
                "class_counts": {},
            }
        ])
    )
    return cases_path


def _args(
    tmp_path: Path,
    script: Path,
    *,
    max_restarts: int = 1,
    max_heartbeat_age: float = 60.0,
    heartbeat_startup_grace: float = 1.0,
    request_json: Path | None = None,
    extra: list[str] | None = None,
):
    parser = supervise.build_parser()
    argv = [
        "--dataset-root",
        str(tmp_path),
        "--cases-json",
        str(_cases_json(tmp_path)),
        "--output-dir",
        str(tmp_path / "run"),
        "--runner-script",
        str(script),
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
        str(max_restarts),
        "--log-jsonl",
        str(tmp_path / "supervisor.jsonl"),
        "--preview-only",
    ]
    if request_json:
        argv.extend(["--request-json", str(request_json)])
    if extra:
        argv.extend(extra)
    return parser.parse_args(argv)


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_completed_artifacts(output_dir: Path, cases_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = json.loads(cases_path.read_text())
    (output_dir / "manifest.json").write_text(json.dumps({"cases": cases}))
    (output_dir / "results.jsonl").write_text(
        json.dumps({
            "case_id": "image_000001:frame:full",
            "status": "preview_only",
            "final_status": "ok",
            "quality_failures": [],
        })
        + "\n"
    )
    (output_dir / "summary.json").write_text(json.dumps({"total_cases": 1, "totals": {"ok": 1}}))
    (output_dir / "heartbeat.json").write_text(
        json.dumps({
            "status": "completed",
            "phase": "finished",
            "heartbeat_epoch": time.time(),
        })
    )


def test_supervisor_restarts_nonzero_runner_and_completes_after_resume(tmp_path: Path) -> None:
    script = tmp_path / "fake_runner.py"
    script.write_text(
        """
import argparse, json, pathlib, sys, time
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir")
parser.add_argument("--cases-json")
args, _ = parser.parse_known_args()
out = pathlib.Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
state = out / "state.txt"
count = int(state.read_text()) if state.exists() else 0
state.write_text(str(count + 1))
if count == 0:
    sys.exit(5)
cases = json.loads(pathlib.Path(args.cases_json).read_text())
(out / "manifest.json").write_text(json.dumps({"cases": cases}))
(out / "results.jsonl").write_text(json.dumps({
    "case_id": "image_000001:frame:full",
    "status": "preview_only",
    "final_status": "ok",
    "quality_failures": []
}) + "\\n")
(out / "summary.json").write_text(json.dumps({"total_cases": 1, "totals": {"ok": 1}}))
(out / "heartbeat.json").write_text(json.dumps({
    "status": "completed",
    "phase": "finished",
    "heartbeat_epoch": time.time()
}))
sys.exit(0)
""".strip()
    )
    args = _args(tmp_path, script, max_restarts=2)

    assert supervise.supervise_soak(args) == supervise.TERMINAL_SUCCESS

    events = _events(tmp_path / "supervisor.jsonl")
    assert any(event["event"] == "supervisor_restart" for event in events)
    assert events[-1]["event"] == "supervisor_complete"
    assert (tmp_path / "run" / "state.txt").read_text() == "2"


def test_supervisor_restarts_signal_abort_runner_and_records_signal(tmp_path: Path) -> None:
    script = tmp_path / "abort_then_success_runner.py"
    script.write_text(
        """
import argparse, json, os, pathlib, signal, sys, time
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir")
parser.add_argument("--cases-json")
args, _ = parser.parse_known_args()
out = pathlib.Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
state = out / "state.txt"
count = int(state.read_text()) if state.exists() else 0
state.write_text(str(count + 1))
if count == 0:
    os.kill(os.getpid(), signal.SIGABRT)
cases = json.loads(pathlib.Path(args.cases_json).read_text())
(out / "manifest.json").write_text(json.dumps({"cases": cases}))
(out / "results.jsonl").write_text(json.dumps({
    "case_id": "image_000001:frame:full",
    "status": "preview_only",
    "final_status": "ok",
    "quality_failures": []
}) + "\\n")
(out / "summary.json").write_text(json.dumps({"total_cases": 1, "totals": {"ok": 1}}))
(out / "heartbeat.json").write_text(json.dumps({
    "status": "completed",
    "phase": "finished",
    "heartbeat_epoch": time.time()
}))
sys.exit(0)
""".strip()
    )
    args = _args(tmp_path, script, max_restarts=2)

    assert supervise.supervise_soak(args) == supervise.TERMINAL_SUCCESS

    events = _events(tmp_path / "supervisor.jsonl")
    signal_exit = [
        event
        for event in events
        if event["event"] == "runner_exit" and event["status"] == "signal_exit"
    ][0]
    assert signal_exit["return_code"] == -6
    assert signal_exit["return_signal"] == 6
    assert signal_exit["return_signal_name"] == "SIGABRT"
    assert any(event["event"] == "supervisor_restart" for event in events)
    assert events[-1]["event"] == "supervisor_complete"
    assert (tmp_path / "run" / "state.txt").read_text() == "2"


def test_supervisor_kills_stale_heartbeat_and_stops_at_restart_limit(tmp_path: Path) -> None:
    script = tmp_path / "stale_runner.py"
    script.write_text(
        """
import argparse, json, pathlib, time
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir")
parser.add_argument("--cases-json")
args, _ = parser.parse_known_args()
out = pathlib.Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
cases = json.loads(pathlib.Path(args.cases_json).read_text())
(out / "manifest.json").write_text(json.dumps({"cases": cases}))
(out / "heartbeat.json").write_text(json.dumps({
    "status": "running",
    "phase": "attempt_running",
    "heartbeat_epoch": time.time() - 1000
}))
time.sleep(30)
""".strip()
    )
    args = _args(tmp_path, script, max_restarts=0, max_heartbeat_age=0.05)

    assert supervise.supervise_soak(args) == supervise.TERMINAL_RESTART_LIMIT

    events = _events(tmp_path / "supervisor.jsonl")
    assert any(event["event"] == "runner_stale_heartbeat" for event in events)
    assert events[-1]["event"] == "supervisor_stop"
    assert events[-1]["reason"] == "restart_limit"


def test_supervisor_stale_heartbeat_terminates_runner_process_group(tmp_path: Path) -> None:
    child_script = tmp_path / "sleeping_worker_child.py"
    child_script.write_text(
        """
import pathlib, signal, sys, time
marker = pathlib.Path(sys.argv[1])
ready = pathlib.Path(sys.argv[2])
def handle_term(_signal_number, _frame):
    marker.write_text("terminated")
    raise SystemExit(0)
signal.signal(signal.SIGTERM, handle_term)
ready.write_text("ready")
while True:
    time.sleep(0.1)
""".strip()
    )
    script = tmp_path / "stale_runner_with_child.py"
    script.write_text(
        f"""
import argparse, json, pathlib, subprocess, sys, time
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir")
parser.add_argument("--cases-json")
args, _ = parser.parse_known_args()
out = pathlib.Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
marker = out / "child_terminated.txt"
ready = out / "child_ready.txt"
child = subprocess.Popen([sys.executable, {str(child_script)!r}, str(marker), str(ready)])
(out / "child_pid.txt").write_text(str(child.pid))
deadline = time.time() + 5
while not ready.exists() and time.time() < deadline:
    time.sleep(0.01)
if not ready.exists():
    raise SystemExit("child did not become ready")
cases = json.loads(pathlib.Path(args.cases_json).read_text())
(out / "manifest.json").write_text(json.dumps({{"cases": cases}}))
(out / "heartbeat.json").write_text(json.dumps({{
    "status": "running",
    "phase": "attempt_running",
    "heartbeat_epoch": time.time() - 1000
}}))
time.sleep(30)
""".strip()
    )
    args = _args(tmp_path, script, max_restarts=0, max_heartbeat_age=0.05)

    assert supervise.supervise_soak(args) == supervise.TERMINAL_RESTART_LIMIT

    marker = tmp_path / "run" / "child_terminated.txt"
    deadline = time.time() + 2.0
    while time.time() < deadline and not marker.exists():
        time.sleep(0.05)
    assert marker.read_text() == "terminated"
    events = _events(tmp_path / "supervisor.jsonl")
    assert any(event["event"] == "runner_stale_heartbeat" for event in events)


def test_supervisor_sigkills_term_resistant_runner_process_group(tmp_path: Path) -> None:
    child_script = tmp_path / "term_resistant_worker_child.py"
    child_script.write_text(
        """
import pathlib, signal, sys, time
ready = pathlib.Path(sys.argv[1])
signal.signal(signal.SIGTERM, signal.SIG_IGN)
ready.write_text("ready")
while True:
    time.sleep(0.1)
""".strip()
    )
    script = tmp_path / "stale_runner_with_term_resistant_child.py"
    script.write_text(
        f"""
import argparse, json, pathlib, subprocess, sys, time
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir")
parser.add_argument("--cases-json")
args, _ = parser.parse_known_args()
out = pathlib.Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
ready = out / "term_resistant_child_ready.txt"
child = subprocess.Popen([sys.executable, {str(child_script)!r}, str(ready)])
(out / "term_resistant_child_pid.txt").write_text(str(child.pid))
deadline = time.time() + 5
while not ready.exists() and time.time() < deadline:
    time.sleep(0.01)
if not ready.exists():
    raise SystemExit("child did not become ready")
cases = json.loads(pathlib.Path(args.cases_json).read_text())
(out / "manifest.json").write_text(json.dumps({{"cases": cases}}))
(out / "heartbeat.json").write_text(json.dumps({{
    "status": "running",
    "phase": "attempt_running",
    "heartbeat_epoch": time.time() - 1000
}}))
time.sleep(30)
""".strip()
    )
    args = _args(
        tmp_path,
        script,
        max_restarts=0,
        max_heartbeat_age=0.05,
        extra=["--kill-timeout", "0.1"],
    )

    assert supervise.supervise_soak(args) == supervise.TERMINAL_RESTART_LIMIT

    child_pid = int((tmp_path / "run" / "term_resistant_child_pid.txt").read_text())
    deadline = time.time() + 2.0
    while time.time() < deadline and _pid_alive(child_pid):
        time.sleep(0.05)
    if _pid_alive(child_pid):
        os.kill(child_pid, signal.SIGKILL)
    assert not _pid_alive(child_pid)
    events = _events(tmp_path / "supervisor.jsonl")
    assert any(event["event"] == "runner_stale_heartbeat" for event in events)


def test_supervisor_restarts_after_stale_heartbeat_artifacts_and_completes(tmp_path: Path) -> None:
    script = tmp_path / "stale_then_success_runner.py"
    script.write_text(
        """
import argparse, json, pathlib, sys, time
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir")
parser.add_argument("--cases-json")
args, _ = parser.parse_known_args()
out = pathlib.Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
state = out / "state.txt"
count = int(state.read_text()) if state.exists() else 0
state.write_text(str(count + 1))
cases = json.loads(pathlib.Path(args.cases_json).read_text())
(out / "manifest.json").write_text(json.dumps({"cases": cases}))
if count == 0:
    (out / "heartbeat.json").write_text(json.dumps({
        "status": "running",
        "phase": "attempt_running",
        "heartbeat_epoch": time.time() - 1000
    }))
    time.sleep(30)
    sys.exit(99)
(out / "results.jsonl").write_text(json.dumps({
    "case_id": "image_000001:frame:full",
    "status": "preview_only",
    "final_status": "ok",
    "quality_failures": []
}) + "\\n")
(out / "summary.json").write_text(json.dumps({"total_cases": 1, "totals": {"ok": 1}}))
(out / "heartbeat.json").write_text(json.dumps({
    "status": "completed",
    "phase": "finished",
    "heartbeat_epoch": time.time()
}))
sys.exit(0)
""".strip()
    )
    args = _args(tmp_path, script, max_restarts=2, max_heartbeat_age=0.05)

    assert supervise.supervise_soak(args) == supervise.TERMINAL_SUCCESS

    events = _events(tmp_path / "supervisor.jsonl")
    assert any(event["event"] == "runner_stale_heartbeat" for event in events)
    assert any(event["event"] == "supervisor_restart" for event in events)
    preflight_events = [event for event in events if event["event"] == "preflight"]
    assert any(event["status"] == "warn" for event in preflight_events)
    assert events[-1]["event"] == "supervisor_complete"
    assert (tmp_path / "run" / "state.txt").read_text() == "2"


def test_supervisor_kills_runner_that_never_writes_heartbeat(tmp_path: Path) -> None:
    script = tmp_path / "missing_heartbeat_runner.py"
    script.write_text(
        """
import argparse, pathlib, time
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir")
args, _ = parser.parse_known_args()
out = pathlib.Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
(out / "started.txt").write_text("started")
time.sleep(30)
""".strip()
    )
    args = _args(tmp_path, script, max_restarts=0, max_heartbeat_age=0.05)

    assert supervise.supervise_soak(args) == supervise.TERMINAL_RESTART_LIMIT

    events = _events(tmp_path / "supervisor.jsonl")
    assert any(event["event"] == "runner_missing_heartbeat" for event in events)
    missing_event = [event for event in events if event["event"] == "runner_missing_heartbeat"][-1]
    assert missing_event["heartbeat_startup_grace_seconds"] == 1.0
    runner_exit = [event for event in events if event["event"] == "runner_exit"][-1]
    assert runner_exit["status"] == "missing_heartbeat"
    assert events[-1]["event"] == "supervisor_stop"
    assert events[-1]["last_runner_status"] == "missing_heartbeat"


def test_supervisor_heartbeat_fuse_ignores_chattery_stdout(tmp_path: Path) -> None:
    script = tmp_path / "chatty_stale_runner.py"
    script.write_text(
        """
import argparse, json, pathlib, sys, time
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir")
parser.add_argument("--cases-json")
args, _ = parser.parse_known_args()
out = pathlib.Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
cases = json.loads(pathlib.Path(args.cases_json).read_text())
(out / "manifest.json").write_text(json.dumps({"cases": cases}))
(out / "heartbeat.json").write_text(json.dumps({
    "status": "running",
    "phase": "attempt_running",
    "heartbeat_epoch": time.time() - 1000
}))
while True:
    print("still writing stdout", flush=True)
    time.sleep(0.005)
""".strip()
    )
    args = _args(tmp_path, script, max_restarts=0, max_heartbeat_age=0.05)

    assert supervise.supervise_soak(args) == supervise.TERMINAL_RESTART_LIMIT

    events = _events(tmp_path / "supervisor.jsonl")
    assert any(event["event"] == "runner_stale_heartbeat" for event in events)
    runner_exit = [event for event in events if event["event"] == "runner_exit"][-1]
    assert runner_exit["status"] == "stale_heartbeat"


def test_supervisor_missing_heartbeat_fuse_ignores_chattery_stdout(tmp_path: Path) -> None:
    script = tmp_path / "chatty_missing_heartbeat_runner.py"
    script.write_text(
        """
import argparse, pathlib, time
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir")
args, _ = parser.parse_known_args()
out = pathlib.Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
while True:
    print("still writing stdout", flush=True)
    time.sleep(0.005)
""".strip()
    )
    args = _args(tmp_path, script, max_restarts=0, max_heartbeat_age=0.05)

    assert supervise.supervise_soak(args) == supervise.TERMINAL_RESTART_LIMIT

    events = _events(tmp_path / "supervisor.jsonl")
    assert any(event["event"] == "runner_missing_heartbeat" for event in events)
    runner_exit = [event for event in events if event["event"] == "runner_exit"][-1]
    assert runner_exit["status"] == "missing_heartbeat"


def test_supervisor_non_lock_preflight_error_blocks_runner_launch(tmp_path: Path) -> None:
    script = tmp_path / "should_not_run.py"
    marker = tmp_path / "started.txt"
    script.write_text(f"from pathlib import Path\\nPath({str(marker)!r}).write_text('started')\\n")
    args = _args(
        tmp_path,
        script,
        max_restarts=1,
        request_json=tmp_path / "missing-request.json",
    )

    assert supervise.supervise_soak(args) == supervise.TERMINAL_PRECHECK_FAILED

    assert not marker.exists()
    events = _events(tmp_path / "supervisor.jsonl")
    assert events[-1]["reason"] == "preflight_failed"


def test_supervisor_waits_for_live_runner_lock_before_launch(tmp_path: Path) -> None:
    script = tmp_path / "success_runner.py"
    marker = tmp_path / "started.txt"
    script.write_text(
        f"""
import argparse, json, pathlib, time
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir")
parser.add_argument("--cases-json")
args, _ = parser.parse_known_args()
out = pathlib.Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
pathlib.Path({str(marker)!r}).write_text("started")
cases = json.loads(pathlib.Path(args.cases_json).read_text())
(out / "manifest.json").write_text(json.dumps({{"cases": cases}}))
(out / "results.jsonl").write_text(json.dumps({{
    "case_id": "image_000001:frame:full",
    "status": "preview_only",
    "final_status": "ok",
    "quality_failures": []
}}) + "\\n")
(out / "summary.json").write_text(json.dumps({{"total_cases": 1, "totals": {{"ok": 1}}}}))
(out / "heartbeat.json").write_text(json.dumps({{
    "status": "completed",
    "phase": "finished",
    "heartbeat_epoch": time.time()
}}))
""".strip()
    )
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    lock_path = output_dir / supervise.runner.RUNNER_LOCK_NAME
    lock_path.write_text(
        json.dumps({
            "runner_id": "external-live",
            "pid": os.getpid(),
            "heartbeat_epoch": supervise.time.time(),
        })
    )

    def release_lock() -> None:
        time.sleep(0.05)
        lock_path.unlink(missing_ok=True)

    thread = threading.Thread(target=release_lock)
    thread.start()
    args = _args(
        tmp_path,
        script,
        max_restarts=0,
        extra=["--live-lock-wait-timeout", "1"],
    )

    try:
        assert supervise.supervise_soak(args) == supervise.TERMINAL_SUCCESS
    finally:
        thread.join(timeout=1.0)

    assert marker.read_text() == "started"
    events = _events(tmp_path / "supervisor.jsonl")
    assert any(event["event"] == "preflight_live_runner_lock_wait" for event in events)
    assert events[-1]["event"] == "supervisor_complete"


def test_supervisor_completes_without_launching_runner_when_artifacts_already_pass(tmp_path: Path) -> None:
    script = tmp_path / "should_not_run.py"
    marker = tmp_path / "started.txt"
    script.write_text(f"from pathlib import Path\\nPath({str(marker)!r}).write_text('started')\\n")
    cases_path = _cases_json(tmp_path)
    _write_completed_artifacts(tmp_path / "run", cases_path)
    args = _args(tmp_path, script, max_restarts=0)

    assert supervise.supervise_soak(args) == supervise.TERMINAL_SUCCESS

    assert not marker.exists()
    events = _events(tmp_path / "supervisor.jsonl")
    assert any(event["event"] == "strict_audit" and event["pre_run"] is True for event in events)
    assert events[-1]["event"] == "supervisor_complete"
    assert events[-1]["already_complete"] is True


def test_supervisor_stops_before_launch_when_completed_artifacts_fail_nonrecoverable_audit(
    tmp_path: Path,
) -> None:
    script = tmp_path / "should_not_run.py"
    marker = tmp_path / "started.txt"
    script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('started')\n")
    cases_path = _cases_json(tmp_path)
    output_dir = tmp_path / "run"
    _write_completed_artifacts(output_dir, cases_path)
    row = json.loads((output_dir / "results.jsonl").read_text())
    row["recovery_events"] = [{"action": "loop_detected"}]
    (output_dir / "results.jsonl").write_text(json.dumps(row) + "\n")
    args = _args(
        tmp_path,
        script,
        max_restarts=1,
        extra=["--max-loop-recovery-case-rate", "0"],
    )

    assert supervise.supervise_soak(args) == supervise.TERMINAL_STRICT_AUDIT_FAILED

    assert not marker.exists()
    events = _events(tmp_path / "supervisor.jsonl")
    assert any(event["event"] == "strict_audit" and event["pre_run"] is True for event in events)
    assert not any(event["event"] == "supervisor_restart" for event in events)
    assert events[-1]["event"] == "supervisor_stop"
    assert events[-1]["reason"] == "strict_audit_failed_nonrecoverable"
    assert events[-1]["pre_run"] is True


def test_supervisor_preflight_uses_request_json_template(tmp_path: Path) -> None:
    script = tmp_path / "runner.py"
    request_json = tmp_path / "request.json"
    request_json.write_text(json.dumps({"model_id": "example/template-model"}))
    args = _args(tmp_path, script, request_json=request_json)

    preflight_args = supervise._preflight_args(args)

    assert preflight_args.request_json == request_json


def test_supervisor_preflight_args_match_caption_runner_settings(tmp_path: Path) -> None:
    script = tmp_path / "runner.py"
    args = _args(
        tmp_path,
        script,
        extra=[
            "--max-boxes",
            "24",
            "--max-new-tokens",
            "1800",
            "--final-sentences",
            "5",
            "--window-size",
            "512",
            "--window-overlap",
            "0.25",
            "--mlx-max-image-side",
            "384",
            "--retry-image-side-scale",
            "0.5",
            "--min-retry-image-side",
            "192",
            "--temperature",
            "0.35",
            "--top-p",
            "0.7",
            "--top-k",
            "10",
            "--use-sampling",
            "--save-dataset-text-labels",
            "--prompt",
            "Custom caption prompt.",
        ],
    )

    preflight_args = supervise._preflight_args(args)

    assert preflight_args.max_boxes == 24
    assert preflight_args.max_new_tokens == 1800
    assert preflight_args.final_sentences == 5
    assert preflight_args.window_size == 512
    assert preflight_args.window_overlap == 0.25
    assert preflight_args.mlx_max_image_side == 384
    assert preflight_args.retry_image_side_scale == 0.5
    assert preflight_args.min_retry_image_side == 192
    assert preflight_args.temperature == 0.35
    assert preflight_args.top_p == 0.7
    assert preflight_args.top_k == 10
    assert preflight_args.use_sampling is True
    assert preflight_args.save_dataset_text_labels is True
    assert preflight_args.prompt == "Custom caption prompt."


def test_supervisor_set_and_forget_defaults_to_safer_runtime_profile(tmp_path: Path) -> None:
    script = tmp_path / "runner.py"
    parser = supervise.build_parser()
    argv = [
        "--dataset-root",
        str(tmp_path),
        "--cases-json",
        str(_cases_json(tmp_path)),
        "--output-dir",
        str(tmp_path / "run"),
        "--runner-script",
        str(script),
        "--set-and-forget",
    ]
    args = parser.parse_args(argv)

    changed = supervise.apply_set_and_forget_runtime_defaults(args, argv)

    assert changed is True
    assert args.mlx_max_image_side == supervise.DEFAULT_SET_AND_FORGET_MLX_MAX_IMAGE_SIDE
    assert args.min_retry_image_side == supervise.DEFAULT_SET_AND_FORGET_MIN_RETRY_IMAGE_SIDE
    assert args.cooldown_after_success == supervise.DEFAULT_SET_AND_FORGET_COOLDOWN_AFTER_SUCCESS_SECONDS
    assert args.attempts == supervise.DEFAULT_SET_AND_FORGET_ATTEMPTS


def test_supervisor_set_and_forget_preserves_explicit_runtime_profile(tmp_path: Path) -> None:
    script = tmp_path / "runner.py"
    parser = supervise.build_parser()
    argv = [
        "--dataset-root",
        str(tmp_path),
        "--cases-json",
        str(_cases_json(tmp_path)),
        "--output-dir",
        str(tmp_path / "run"),
        "--runner-script",
        str(script),
        "--set-and-forget",
        "--mlx-max-image-side",
        "384",
        "--min-retry-image-side",
        "192",
        "--cooldown-after-success",
        "0",
        "--attempts",
        "2",
    ]
    args = parser.parse_args(argv)

    changed = supervise.apply_set_and_forget_runtime_defaults(args, argv)

    assert changed is False
    assert args.mlx_max_image_side == 384
    assert args.min_retry_image_side == 192
    assert args.cooldown_after_success == 0
    assert args.attempts == 2


def test_supervisor_strict_audit_requires_saved_text_labels_when_saving(
    monkeypatch,
    tmp_path: Path,
) -> None:
    script = tmp_path / "runner.py"
    captured = {}

    def fake_audit_soak(output_dir, **kwargs):
        captured["output_dir"] = output_dir
        captured["kwargs"] = kwargs
        return {"status": "ok", "checks": []}

    monkeypatch.setattr(supervise.audit, "audit_soak", fake_audit_soak)
    args = _args(
        tmp_path,
        script,
        extra=["--save-dataset-text-labels"],
    )

    report = supervise._strict_audit(args)

    assert report["status"] == "ok"
    assert captured["output_dir"] == args.output_dir
    assert captured["kwargs"]["require_saved_text_labels"] is True
    assert captured["kwargs"]["set_and_forget"] is False


def test_supervisor_strict_audit_can_require_saved_text_labels_explicitly(
    monkeypatch,
    tmp_path: Path,
) -> None:
    script = tmp_path / "runner.py"
    captured = {}

    def fake_audit_soak(output_dir, **kwargs):
        captured["kwargs"] = kwargs
        return {"status": "ok", "checks": []}

    monkeypatch.setattr(supervise.audit, "audit_soak", fake_audit_soak)
    args = _args(
        tmp_path,
        script,
        extra=["--require-saved-text-labels"],
    )

    supervise._strict_audit(args)

    assert captured["kwargs"]["require_saved_text_labels"] is True


def test_supervisor_windowed_set_and_forget_defaults_to_text_only_full_compose(
    monkeypatch,
    tmp_path: Path,
) -> None:
    script = tmp_path / "runner.py"
    captured = {}

    def fake_supervise_soak(args, runner_extra_args=None):
        captured["args"] = args
        captured["runner_extra_args"] = list(runner_extra_args or [])
        return 0

    monkeypatch.setattr(supervise, "supervise_soak", fake_supervise_soak)

    result = supervise.main([
        "--dataset-root",
        str(tmp_path),
        "--cases-json",
        str(_cases_json(tmp_path)),
        "--output-dir",
        str(tmp_path / "run"),
        "--runner-script",
        str(script),
        "--set-and-forget",
        "--caption-mode",
        "windowed",
    ])

    assert result == 0
    assert captured["args"].windowed_full_image_strategy == "text_only"


def test_supervisor_windowed_set_and_forget_preserves_explicit_visual_full_compose(
    monkeypatch,
    tmp_path: Path,
) -> None:
    script = tmp_path / "runner.py"
    captured = {}

    def fake_supervise_soak(args, runner_extra_args=None):
        captured["args"] = args
        captured["runner_extra_args"] = list(runner_extra_args or [])
        return 0

    monkeypatch.setattr(supervise, "supervise_soak", fake_supervise_soak)

    result = supervise.main([
        "--dataset-root",
        str(tmp_path),
        "--cases-json",
        str(_cases_json(tmp_path)),
        "--output-dir",
        str(tmp_path / "run"),
        "--runner-script",
        str(script),
        "--set-and-forget",
        "--caption-mode",
        "windowed",
        "--windowed-full-image-strategy",
        "visual",
    ])

    assert result == 0
    assert captured["args"].windowed_full_image_strategy == "visual"


def test_supervisor_stops_when_terminal_audit_fails_nonrecoverable_degraded_rates(tmp_path: Path) -> None:
    script = tmp_path / "degraded_runner.py"
    script.write_text(
        """
import argparse, json, pathlib, sys, time
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir")
parser.add_argument("--cases-json")
args, _ = parser.parse_known_args()
out = pathlib.Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
state = out / "state.txt"
count = int(state.read_text()) if state.exists() else 0
state.write_text(str(count + 1))
cases = json.loads(pathlib.Path(args.cases_json).read_text())
(out / "manifest.json").write_text(json.dumps({"cases": cases}))
(out / "results.jsonl").write_text(json.dumps({
    "case_id": "image_000001:frame:full",
    "status": "preview_only",
    "final_status": "ok",
    "quality_failures": [],
    "recovery_events": [{"action": "loop_detected"}]
}) + "\\n")
(out / "summary.json").write_text(json.dumps({"total_cases": 1, "totals": {"ok": 1}}))
(out / "heartbeat.json").write_text(json.dumps({
    "status": "completed",
    "phase": "finished",
    "heartbeat_epoch": time.time()
}))
sys.exit(0)
""".strip()
    )
    args = _args(
        tmp_path,
        script,
        max_restarts=0,
        extra=["--max-loop-recovery-case-rate", "0"],
    )

    assert supervise.supervise_soak(args) == supervise.TERMINAL_STRICT_AUDIT_FAILED

    events = _events(tmp_path / "supervisor.jsonl")
    strict_events = [event for event in events if event["event"] == "strict_audit"]
    assert strict_events
    assert strict_events[-1]["status"] == "error"
    assert strict_events[-1]["degraded_rates"]["loop_recovery_case_rate"] == 1.0
    assert not any(event["event"] == "supervisor_restart" for event in events)
    assert events[-1]["reason"] == "strict_audit_failed_nonrecoverable"
    assert events[-1]["last_runner_status"] == "ok"
    assert events[-1]["last_return_code"] == 0
    assert (tmp_path / "run" / "state.txt").read_text() == "1"


def test_supervisor_cli_runs_directly_with_fake_runner(tmp_path: Path) -> None:
    script = tmp_path / "success_runner.py"
    cases_path = _cases_json(tmp_path)
    output_dir = tmp_path / "run"
    log_path = tmp_path / "supervisor.jsonl"
    script.write_text(
        """
import argparse, json, pathlib, time
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir")
parser.add_argument("--cases-json")
args, _ = parser.parse_known_args()
out = pathlib.Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
cases = json.loads(pathlib.Path(args.cases_json).read_text())
(out / "manifest.json").write_text(json.dumps({"cases": cases}))
(out / "results.jsonl").write_text(json.dumps({
    "case_id": "image_000001:frame:full",
    "status": "preview_only",
    "final_status": "ok",
    "quality_failures": []
}) + "\\n")
(out / "summary.json").write_text(json.dumps({"total_cases": 1, "totals": {"ok": 1}}))
(out / "heartbeat.json").write_text(json.dumps({
    "status": "completed",
    "phase": "finished",
    "heartbeat_epoch": time.time()
}))
""".strip()
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "supervise_qwen_caption_soak.py"),
            "--dataset-root",
            str(tmp_path),
            "--cases-json",
            str(cases_path),
            "--output-dir",
            str(output_dir),
            "--runner-script",
            str(script),
            "--min-free-gb",
            "0",
            "--restart-delay",
            "0",
            "--monitor-interval",
            "0.01",
            "--max-runner-restarts",
            "0",
            "--log-jsonl",
            str(log_path),
            "--preview-only",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert _events(log_path)[-1]["event"] == "supervisor_complete"
