from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import plistlib
import subprocess

from tools import audit_qwen_caption_operation as operation


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _launchd_stdout(
    *,
    plist_path: Path,
    arguments: list[str],
    caffeinate: bool = False,
    state: str = "running",
) -> str:
    program = "/usr/bin/caffeinate" if caffeinate else arguments[0]
    argument_block = "\n".join(f"\t\t{item}" for item in arguments)
    return (
        f"state = {state}\n"
        f"path = {plist_path}\n"
        f"program = {program}\n"
        "arguments = {\n"
        f"{argument_block}\n"
        "}\n"
    )


def _fixture_plist(tmp_path: Path, role: str) -> Path:
    return tmp_path / "Library" / "LaunchAgents" / f"{role}.plist"


def _plist_payload(
    *,
    command: list[str],
    output_dir: Path,
    stdout_name: str,
    stderr_name: str,
    caffeinate: bool = False,
) -> dict:
    program_arguments = ["/usr/bin/caffeinate", "-dimsu", *command] if caffeinate else command
    return {
        "ProgramArguments": program_arguments,
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 300,
        "WorkingDirectory": str(operation.REPO_ROOT),
        "StandardOutPath": str(output_dir / stdout_name),
        "StandardErrorPath": str(output_dir / stderr_name),
    }


def _write_operation_fixture(
    tmp_path: Path,
    *,
    latest_checked_epoch: float = 950.0,
    latest_timestamp_field: str = "checked_at",
) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    readiness_path = tmp_path / "readiness.json"
    latest_path = run_dir / "watchdog_latest.json"
    state_path = run_dir / "watchdog_state.json"
    operational_audit_path = run_dir / "operational_audit.json"
    launch_agents_dir = tmp_path / "Library" / "LaunchAgents"
    supervisor_plist = launch_agents_dir / "supervisor.plist"
    watchdog_plist = launch_agents_dir / "watchdog.plist"
    run_dir.mkdir(parents=True)
    launch_agents_dir.mkdir(parents=True)
    degraded_rate_args = [
        "--max-failed-case-rate",
        "0",
        "--max-quality-failure-rate",
        "0",
        "--max-recovery-event-case-rate",
        "0.25",
        "--max-loop-recovery-case-rate",
        "0.05",
        "--max-loop-guard-case-rate",
        "0.05",
        "--max-deterministic-recovery-case-rate",
        "0.01",
        "--max-failed-attempt-row-rate",
        "0.25",
        "--max-signal-exit-attempt-row-rate",
        "0.01",
        "--max-attempt-overrun",
        "60",
        "--min-rate-cases",
        "20",
    ]
    supervisor_command = [
        "/bin/python3",
        "tools/supervise_qwen_caption_soak.py",
        "--output-dir",
        str(run_dir),
        "--set-and-forget",
        "--max-runner-restarts",
        "25",
        "--max-heartbeat-age",
        "900",
        "--min-free-gb",
        "1",
        "--max-projected-duration-hours",
        "336",
    ]
    watchdog_command = [
        "/bin/python3",
        "tools/watch_qwen_caption_soak.py",
        str(run_dir),
        *degraded_rate_args,
        "--min-free-gb",
        "1",
        "--max-projected-duration-hours",
        "336",
        "--graceful-restart-timeout",
        "300",
        "--remediate-launchd-label",
        "com.example.caption",
        "--remediate-launchd-domain",
        "gui/501",
        "--remediate-launchd-plist",
        str(supervisor_plist),
        "--max-remediations",
        "25",
    ]
    supervisor_plist.write_bytes(
        plistlib.dumps(
            _plist_payload(
                command=supervisor_command,
                output_dir=run_dir,
                stdout_name="launchd_stdout.log",
                stderr_name="launchd_stderr.log",
            )
        )
    )
    watchdog_plist.write_bytes(
        plistlib.dumps(
            _plist_payload(
                command=watchdog_command,
                output_dir=run_dir,
                stdout_name="watchdog_launchd_stdout.log",
                stderr_name="watchdog_launchd_stderr.log",
                caffeinate=True,
            )
        )
    )

    readiness_path.write_text(
        json.dumps({"status": "ok", "ready_for_10k_set_and_forget": True}),
        encoding="utf-8",
    )
    latest_payload = {
        "status": "ok",
        "audit_status": "ok",
        "output_dir": str(run_dir),
        "processed_cases": 12,
        "active_attempt": {"case": "image_000013", "attempt": 1},
    }
    if latest_timestamp_field == "checked_epoch":
        latest_payload["checked_epoch"] = latest_checked_epoch
    elif latest_timestamp_field == "time":
        latest_payload["time"] = latest_checked_epoch
    else:
        latest_payload["checked_at"] = _iso(latest_checked_epoch)
    state_payload = {
        "consecutive_unhealthy": 0,
        "last_progress_cases": 12,
        "max_remediations": 25,
        "remediation_count": 0,
        "state_json": str(state_path),
        "state_version": 1,
    }
    latest_payload["watchdog_state"] = dict(state_payload)
    latest_path.write_text(json.dumps(latest_payload), encoding="utf-8")
    state_path.write_text(json.dumps(state_payload), encoding="utf-8")

    runbook = {
        "schema_version": 1,
        "output_dir": str(run_dir),
        "paths": {
            "readiness_json": str(readiness_path),
            "watchdog_latest_json": str(latest_path),
            "watchdog_state_json": str(state_path),
            "operational_audit_json": str(operational_audit_path),
        },
        "commands": {
            "supervisor": supervisor_command,
            "watchdog": watchdog_command,
            "live_status": [
                "/bin/python3",
                "tools/audit_qwen_caption_soak.py",
                str(run_dir),
                "--max-heartbeat-age",
                "900",
                *degraded_rate_args,
                "--min-free-gb",
                "1",
                "--max-projected-duration-hours",
                "336",
            ],
            "final_audit": [
                "/bin/python3",
                "tools/audit_qwen_caption_soak.py",
                str(run_dir),
                *degraded_rate_args,
                "--max-projected-duration-hours",
                "336",
                "--min-free-gb",
                "1",
            ],
            "pilot_certification": [
                "/bin/python3",
                "tools/certify_qwen_caption_soak.py",
                str(run_dir),
                *degraded_rate_args,
            ],
            "operational_audit": [
                "/bin/python3",
                "tools/audit_qwen_caption_operation.py",
                str(tmp_path / "unattended_run.json"),
                "--allow-running-incomplete",
                "--compact",
                "--write-json",
                str(operational_audit_path),
                "--strict-set-and-forget",
            ],
        },
        "launchd_install": {
            "domain": "gui/501",
            "launchctl": "launchctl",
            "requested": True,
            "timeout_seconds": 3,
            "roles": {
                "supervisor": {
                    "label": "com.example.caption",
                    "plist_path": str(supervisor_plist),
                },
                "watchdog": {
                    "label": "com.example.caption.watchdog",
                    "plist_path": str(watchdog_plist),
                },
            },
        },
        "launchd_power_assertion": {
            "enabled": True,
            "program": "/usr/bin/caffeinate",
            "arguments": ["-dimsu"],
        },
        "set_and_forget_gate": {"tenk_mode": True},
        "launchd_install_result": {
            "status": "ok",
            "requested": True,
            "domain": "gui/501",
        },
    }
    runbook_path = tmp_path / "unattended_run.json"
    runbook_path.write_text(json.dumps(runbook), encoding="utf-8")
    return runbook_path, run_dir


def _fake_artifact_report():
    return {
        "status": "ok",
        "processed_cases": 12,
        "expected_cases": 100,
        "failed_cases": 0,
        "quality_failed_cases": 0,
        "heartbeat": {"status": "running", "phase": "attempt_running"},
    }


def _healthy_command_runner(tmp_path: Path):
    def fake_run(command, **_kwargs):
        if command[:2] == ["launchctl", "print"]:
            plist = (
                tmp_path
                / "Library"
                / "LaunchAgents"
                / ("watchdog.plist" if command[-1].endswith("watchdog") else "supervisor.plist")
            )
            arguments = plistlib.loads(plist.read_bytes())["ProgramArguments"]
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_launchd_stdout(
                    plist_path=plist,
                    arguments=arguments,
                    caffeinate=command[-1].endswith("watchdog"),
                ),
                stderr="",
            )
        if command[:3] == ["pmset", "-g", "assertions"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="PreventSystemSleep 1\npid 123(caffeinate): PreventSystemSleep\n",
                stderr="",
            )
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="unexpected")

    return fake_run


def test_operation_audit_checks_launchd_watchdog_sleep_and_artifacts(monkeypatch, tmp_path: Path) -> None:
    runbook_path, run_dir = _write_operation_fixture(tmp_path)
    calls: dict[str, object] = {}

    def fake_audit(output_dir: Path, **kwargs):
        calls["output_dir"] = output_dir
        calls["kwargs"] = kwargs
        return _fake_artifact_report()

    def fake_run(command, **_kwargs):
        target = command[-1]
        if command[:2] == ["launchctl", "print"] and target.endswith("com.example.caption.watchdog"):
            plist_path = _fixture_plist(tmp_path, "watchdog")
            arguments = plistlib.loads(plist_path.read_bytes())["ProgramArguments"]
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_launchd_stdout(
                    plist_path=plist_path,
                    arguments=arguments,
                    caffeinate=True,
                ),
                stderr="",
            )
        if command[:2] == ["launchctl", "print"] and target.endswith("com.example.caption"):
            plist_path = _fixture_plist(tmp_path, "supervisor")
            arguments = plistlib.loads(plist_path.read_bytes())["ProgramArguments"]
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_launchd_stdout(
                    plist_path=plist_path,
                    arguments=arguments,
                ),
                stderr="",
            )
        if command[:3] == ["pmset", "-g", "assertions"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="PreventSystemSleep 1\npid 123(caffeinate): PreventSystemSleep\n",
                stderr="",
            )
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="unexpected")

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", fake_audit)
    report = operation.audit_operation(
        run_dir,
        allow_running_incomplete=True,
        command_runner=fake_run,
        now_epoch=1000.0,
    )

    assert report["status"] == "ok"
    assert calls["output_dir"] == run_dir.resolve(strict=False)
    assert calls["kwargs"]["allow_running_incomplete"] is True
    assert calls["kwargs"]["set_and_forget"] is True
    assert calls["kwargs"]["min_free_gb"] == 1.0
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["artifact_audit"]["status"] == "ok"
    assert checks["watchdog_latest"]["status"] == "ok"
    assert checks["watchdog_state"]["status"] == "ok"
    assert checks["readiness_artifact"]["status"] == "ok"
    assert checks["supervisor_launchd"]["status"] == "ok"
    assert checks["supervisor_launchd_arguments"]["status"] == "ok"
    assert checks["watchdog_launchd"]["status"] == "ok"
    assert checks["watchdog_launchd_arguments"]["status"] == "ok"
    assert checks["sleep_assertion"]["status"] == "ok"


def test_operation_audit_waits_for_transient_launchd_spawn_state(monkeypatch, tmp_path: Path) -> None:
    runbook_path, _run_dir = _write_operation_fixture(tmp_path)
    launchd_targets: list[str] = []

    def fake_run(command, **_kwargs):
        if command[:2] == ["launchctl", "print"]:
            role = "watchdog" if command[-1].endswith("watchdog") else "supervisor"
            launchd_targets.append(role)
            plist = _fixture_plist(tmp_path, role)
            arguments = plistlib.loads(plist.read_bytes())["ProgramArguments"]
            watchdog_first_probe = role == "watchdog" and launchd_targets.count("watchdog") == 1
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_launchd_stdout(
                    plist_path=plist,
                    arguments=arguments,
                    caffeinate=role == "watchdog",
                    state="spawn scheduled" if watchdog_first_probe else "running",
                ),
                stderr="",
            )
        if command[:3] == ["pmset", "-g", "assertions"]:
            return subprocess.CompletedProcess(command, 0, stdout="PreventSystemSleep 1\ncaffeinate\n", stderr="")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="unexpected")

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", lambda *_args, **_kwargs: _fake_artifact_report())
    report = operation.audit_operation(
        runbook_path,
        command_runner=fake_run,
        now_epoch=1000.0,
        launchd_settle_seconds=1.0,
        launchd_settle_interval_seconds=0.0,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "ok"
    assert launchd_targets.count("watchdog") == 2
    assert checks["watchdog_launchd"]["status"] == "ok"
    assert checks["watchdog_launchd"]["transient_state"] == "spawn scheduled"
    assert checks["watchdog_launchd"]["state_history"] == ["spawn scheduled", "running"]
    assert checks["watchdog_launchd"]["settle_attempts"] == 2
    assert checks["watchdog_launchd_arguments"]["status"] == "ok"


def test_operation_audit_fails_when_transient_launchd_state_does_not_settle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runbook_path, _run_dir = _write_operation_fixture(tmp_path)
    launchd_targets: list[str] = []

    def fake_run(command, **_kwargs):
        if command[:2] == ["launchctl", "print"]:
            role = "watchdog" if command[-1].endswith("watchdog") else "supervisor"
            launchd_targets.append(role)
            plist = _fixture_plist(tmp_path, role)
            arguments = plistlib.loads(plist.read_bytes())["ProgramArguments"]
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_launchd_stdout(
                    plist_path=plist,
                    arguments=arguments,
                    caffeinate=role == "watchdog",
                    state="spawn scheduled" if role == "watchdog" else "running",
                ),
                stderr="",
            )
        if command[:3] == ["pmset", "-g", "assertions"]:
            return subprocess.CompletedProcess(command, 0, stdout="PreventSystemSleep 1\ncaffeinate\n", stderr="")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="unexpected")

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", lambda *_args, **_kwargs: _fake_artifact_report())
    report = operation.audit_operation(
        runbook_path,
        command_runner=fake_run,
        now_epoch=1000.0,
        launchd_settle_seconds=1.0,
        launchd_settle_interval_seconds=0.0,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert launchd_targets.count("watchdog") == 2
    assert checks["watchdog_launchd"]["status"] == "error"
    assert checks["watchdog_launchd"]["state"] == "spawn scheduled"
    assert checks["watchdog_launchd"]["state_history"] == ["spawn scheduled", "spawn scheduled"]
    assert "settle window" in checks["watchdog_launchd"]["detail"]
    assert "watchdog_launchd_arguments" not in checks


def test_operation_audit_accepts_epoch_watchdog_status_timestamp(monkeypatch, tmp_path: Path) -> None:
    current_fixture_root = {"path": tmp_path}

    def fake_run(command, **_kwargs):
        if command[:2] == ["launchctl", "print"]:
            base = current_fixture_root["path"] / "Library" / "LaunchAgents"
            plist = base / f"{'watchdog' if command[-1].endswith('watchdog') else 'supervisor'}.plist"
            arguments = plistlib.loads(plist.read_bytes())["ProgramArguments"]
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_launchd_stdout(
                    plist_path=plist,
                    arguments=arguments,
                    caffeinate=command[-1].endswith("watchdog"),
                ),
                stderr="",
            )
        if command[:3] == ["pmset", "-g", "assertions"]:
            return subprocess.CompletedProcess(command, 0, stdout="PreventSystemSleep 1\ncaffeinate\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", lambda *_args, **_kwargs: _fake_artifact_report())
    for timestamp_field in ("checked_epoch", "time"):
        current_fixture_root["path"] = tmp_path / timestamp_field
        runbook_path, _run_dir = _write_operation_fixture(
            current_fixture_root["path"],
            latest_checked_epoch=950.0,
            latest_timestamp_field=timestamp_field,
        )
        report = operation.audit_operation(runbook_path, command_runner=fake_run, now_epoch=1000.0)

        checks = {check["name"]: check for check in report["checks"]}
        assert checks["watchdog_latest"]["status"] == "ok"


def test_operation_audit_strict_set_and_forget_checks_handoff_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runbook_path, _run_dir = _write_operation_fixture(tmp_path)

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", lambda *_args, **_kwargs: _fake_artifact_report())
    report = operation.audit_operation(
        runbook_path,
        allow_running_incomplete=True,
        strict_set_and_forget=True,
        command_runner=_healthy_command_runner(tmp_path),
        now_epoch=1000.0,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "ok"
    assert report["strict_set_and_forget"] is True
    assert checks["set_and_forget_readiness"]["status"] == "ok"
    assert checks["set_and_forget_commands"]["status"] == "ok"
    assert checks["set_and_forget_live_gates"]["status"] == "ok"
    assert checks["set_and_forget_degraded_rate_gates"]["status"] == "ok"
    assert "--max-loop-guard-case-rate" in checks["set_and_forget_degraded_rate_gates"]["required_flags"]
    assert checks["set_and_forget_saved_text_label_gates"]["status"] == "ok"
    assert checks["set_and_forget_saved_text_label_gates"]["required"] is False
    assert checks["set_and_forget_launchd_install"]["status"] == "ok"
    assert checks["set_and_forget_launchd_persistence"]["status"] == "ok"
    assert checks["set_and_forget_launchd_policy"]["status"] == "ok"
    assert checks["set_and_forget_watchdog_remediation"]["status"] == "ok"
    assert checks["set_and_forget_watchdog_graceful_restart"]["status"] == "ok"
    assert checks["set_and_forget_sleep_prevention"]["status"] == "ok"
    assert checks["watchdog_latest_run_binding"]["status"] == "ok"
    assert checks["watchdog_state_run_binding"]["status"] == "ok"


def test_operation_audit_strict_set_and_forget_accepts_saved_text_label_gates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runbook_path, run_dir = _write_operation_fixture(tmp_path)
    payload = json.loads(runbook_path.read_text(encoding="utf-8"))
    payload["commands"]["supervisor"].append("--save-dataset-text-labels")
    for command_name in ("watchdog", "live_status", "final_audit"):
        payload["commands"][command_name].append("--require-saved-text-labels")
    _fixture_plist(tmp_path, "supervisor").write_bytes(
        plistlib.dumps(
            _plist_payload(
                command=payload["commands"]["supervisor"],
                output_dir=run_dir,
                stdout_name="launchd_stdout.log",
                stderr_name="launchd_stderr.log",
            )
        )
    )
    _fixture_plist(tmp_path, "watchdog").write_bytes(
        plistlib.dumps(
            _plist_payload(
                command=payload["commands"]["watchdog"],
                output_dir=run_dir,
                stdout_name="watchdog_launchd_stdout.log",
                stderr_name="watchdog_launchd_stderr.log",
                caffeinate=True,
            )
        )
    )
    runbook_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", lambda *_args, **_kwargs: _fake_artifact_report())
    report = operation.audit_operation(
        runbook_path,
        allow_running_incomplete=True,
        strict_set_and_forget=True,
        command_runner=_healthy_command_runner(tmp_path),
        now_epoch=1000.0,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "ok"
    assert checks["set_and_forget_saved_text_label_gates"]["status"] == "ok"
    assert checks["set_and_forget_saved_text_label_gates"]["required"] is True
    assert checks["set_and_forget_saved_text_label_gates"]["command_gates"] == {
        "watchdog": True,
        "live_status": True,
        "final_audit": True,
    }


def test_operation_audit_strict_set_and_forget_rejects_missing_saved_text_label_gates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runbook_path, run_dir = _write_operation_fixture(tmp_path)
    payload = json.loads(runbook_path.read_text(encoding="utf-8"))
    payload["commands"]["supervisor"].append("--save-dataset-text-labels")
    payload["commands"]["watchdog"].append("--require-saved-text-labels")
    _fixture_plist(tmp_path, "supervisor").write_bytes(
        plistlib.dumps(
            _plist_payload(
                command=payload["commands"]["supervisor"],
                output_dir=run_dir,
                stdout_name="launchd_stdout.log",
                stderr_name="launchd_stderr.log",
            )
        )
    )
    _fixture_plist(tmp_path, "watchdog").write_bytes(
        plistlib.dumps(
            _plist_payload(
                command=payload["commands"]["watchdog"],
                output_dir=run_dir,
                stdout_name="watchdog_launchd_stdout.log",
                stderr_name="watchdog_launchd_stderr.log",
                caffeinate=True,
            )
        )
    )
    runbook_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", lambda *_args, **_kwargs: _fake_artifact_report())
    report = operation.audit_operation(
        runbook_path,
        allow_running_incomplete=True,
        strict_set_and_forget=True,
        command_runner=_healthy_command_runner(tmp_path),
        now_epoch=1000.0,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["set_and_forget_saved_text_label_gates"]["status"] == "error"
    assert checks["set_and_forget_saved_text_label_gates"]["required"] is True
    assert checks["set_and_forget_saved_text_label_gates"]["missing_commands"] == [
        "live_status",
        "final_audit",
    ]


def test_operation_audit_strict_set_and_forget_rejects_missing_degraded_rate_gates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runbook_path, _run_dir = _write_operation_fixture(tmp_path)
    payload = json.loads(runbook_path.read_text(encoding="utf-8"))

    def remove_flag(command: list[str], flag: str) -> None:
        index = command.index(flag)
        del command[index : index + 2]

    remove_flag(payload["commands"]["live_status"], "--max-loop-guard-case-rate")
    remove_flag(payload["commands"]["final_audit"], "--max-loop-guard-case-rate")
    remove_flag(payload["commands"]["pilot_certification"], "--max-loop-guard-case-rate")
    runbook_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", lambda *_args, **_kwargs: _fake_artifact_report())
    report = operation.audit_operation(
        runbook_path,
        allow_running_incomplete=True,
        strict_set_and_forget=True,
        command_runner=_healthy_command_runner(tmp_path),
        now_epoch=1000.0,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["set_and_forget_degraded_rate_gates"]["status"] == "error"
    assert checks["set_and_forget_degraded_rate_gates"]["missing_flags"] == {
        "live_status": ["--max-loop-guard-case-rate"],
        "final_audit": ["--max-loop-guard-case-rate"],
        "pilot_certification": ["--max-loop-guard-case-rate"],
    }


def test_operation_audit_strict_set_and_forget_rejects_cross_run_watchdog_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runbook_path, _run_dir = _write_operation_fixture(tmp_path)
    payload = json.loads(runbook_path.read_text(encoding="utf-8"))
    latest_path = Path(payload["paths"]["watchdog_latest_json"])
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["output_dir"] = str(tmp_path / "other-run")
    latest_path.write_text(json.dumps(latest), encoding="utf-8")

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", lambda *_args, **_kwargs: _fake_artifact_report())
    report = operation.audit_operation(
        runbook_path,
        allow_running_incomplete=True,
        strict_set_and_forget=True,
        command_runner=_healthy_command_runner(tmp_path),
        now_epoch=1000.0,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["watchdog_latest_run_binding"]["status"] == "error"
    assert checks["watchdog_latest_run_binding"]["actual_output_dir"] == str((tmp_path / "other-run").resolve())


def test_operation_audit_strict_set_and_forget_rejects_cross_run_watchdog_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runbook_path, _run_dir = _write_operation_fixture(tmp_path)
    payload = json.loads(runbook_path.read_text(encoding="utf-8"))
    state_path = Path(payload["paths"]["watchdog_state_json"])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["state_json"] = str(tmp_path / "other-run" / "watchdog_state.json")
    state_path.write_text(json.dumps(state), encoding="utf-8")

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", lambda *_args, **_kwargs: _fake_artifact_report())
    report = operation.audit_operation(
        runbook_path,
        allow_running_incomplete=True,
        strict_set_and_forget=True,
        command_runner=_healthy_command_runner(tmp_path),
        now_epoch=1000.0,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["watchdog_state_run_binding"]["status"] == "error"
    assert "state file state_json does not match the runbook path" in checks["watchdog_state_run_binding"]["mismatches"]


def test_operation_audit_strict_set_and_forget_accepts_live_adoption_capability(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runbook_path, _run_dir = _write_operation_fixture(tmp_path)
    payload = json.loads(runbook_path.read_text(encoding="utf-8"))
    readiness_path = Path(payload["paths"]["readiness_json"])
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["summary"] = {"live_adoption_requested": True}
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    payload["live_adoption_certification"] = {
        "status": "ok",
        "checks": [
            {
                "name": "live_runner_restart_capability",
                "status": "ok",
                "runner_capabilities": ["graceful_restart_request"],
            }
        ],
    }
    runbook_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", lambda *_args, **_kwargs: _fake_artifact_report())
    report = operation.audit_operation(
        runbook_path,
        allow_running_incomplete=True,
        strict_set_and_forget=True,
        command_runner=_healthy_command_runner(tmp_path),
        now_epoch=1000.0,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "ok"
    assert checks["set_and_forget_live_adoption_capability"]["status"] == "ok"


def test_operation_audit_strict_set_and_forget_rejects_live_adoption_without_capability(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runbook_path, _run_dir = _write_operation_fixture(tmp_path)
    payload = json.loads(runbook_path.read_text(encoding="utf-8"))
    readiness_path = Path(payload["paths"]["readiness_json"])
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["summary"] = {"live_adoption_requested": True}
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    payload["live_adoption_certification"] = {
        "status": "ok",
        "checks": [
            {
                "name": "live_runner_restart_capability",
                "status": "error",
                "runner_capabilities": [],
            }
        ],
    }
    runbook_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", lambda *_args, **_kwargs: _fake_artifact_report())
    report = operation.audit_operation(
        runbook_path,
        allow_running_incomplete=True,
        strict_set_and_forget=True,
        command_runner=_healthy_command_runner(tmp_path),
        now_epoch=1000.0,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["set_and_forget_live_adoption_capability"]["status"] == "error"


def test_operation_audit_strict_set_and_forget_rejects_manual_handoff_gaps(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runbook_path, _run_dir = _write_operation_fixture(tmp_path)
    payload = json.loads(runbook_path.read_text(encoding="utf-8"))
    payload["commands"].pop("operational_audit")
    payload["commands"]["watchdog"] = [
        item
        for item in payload["commands"]["watchdog"]
        if item
        not in {
            "--remediate-launchd-label",
            "com.example.caption",
            "--remediate-launchd-domain",
            "gui/501",
            "--remediate-launchd-plist",
            str(tmp_path / "Library" / "LaunchAgents" / "supervisor.plist"),
            "--max-remediations",
            "25",
        }
    ]
    payload["launchd_power_assertion"]["enabled"] = False
    payload.pop("launchd_install_result")
    runbook_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", lambda *_args, **_kwargs: _fake_artifact_report())
    report = operation.audit_operation(
        runbook_path,
        allow_running_incomplete=True,
        strict_set_and_forget=True,
        command_runner=_healthy_command_runner(tmp_path),
        now_epoch=1000.0,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["set_and_forget_commands"]["status"] == "error"
    assert checks["set_and_forget_commands"]["operational_audit_recorded"] is False
    assert checks["set_and_forget_launchd_install"]["status"] == "error"
    assert checks["set_and_forget_launchd_persistence"]["status"] == "ok"
    assert checks["set_and_forget_launchd_policy"]["status"] == "ok"
    assert checks["set_and_forget_watchdog_remediation"]["status"] == "error"
    assert checks["set_and_forget_sleep_prevention"]["status"] == "error"


def test_operation_audit_strict_set_and_forget_rejects_non_strict_operation_audit_command(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runbook_path, _run_dir = _write_operation_fixture(tmp_path)
    payload = json.loads(runbook_path.read_text(encoding="utf-8"))
    command = payload["commands"]["operational_audit"]
    command.remove("--strict-set-and-forget")
    runbook_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", lambda *_args, **_kwargs: _fake_artifact_report())
    report = operation.audit_operation(
        runbook_path,
        allow_running_incomplete=True,
        strict_set_and_forget=True,
        command_runner=_healthy_command_runner(tmp_path),
        now_epoch=1000.0,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["set_and_forget_commands"]["status"] == "error"
    assert checks["set_and_forget_commands"]["operational_audit_recorded"] is False
    assert checks["set_and_forget_commands"]["missing_operational_audit_flags"] == ["--strict-set-and-forget"]


def test_operation_audit_strict_set_and_forget_rejects_missing_supervisor_live_gate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runbook_path, _run_dir = _write_operation_fixture(tmp_path)
    payload = json.loads(runbook_path.read_text(encoding="utf-8"))
    supervisor_command = payload["commands"]["supervisor"]
    projected_index = supervisor_command.index("--max-projected-duration-hours")
    del supervisor_command[projected_index : projected_index + 2]
    _fixture_plist(tmp_path, "supervisor").write_bytes(
        plistlib.dumps(
            _plist_payload(
                command=supervisor_command,
                output_dir=_run_dir,
                stdout_name="launchd_stdout.log",
                stderr_name="launchd_stderr.log",
            )
        )
    )
    runbook_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", lambda *_args, **_kwargs: _fake_artifact_report())
    report = operation.audit_operation(
        runbook_path,
        allow_running_incomplete=True,
        strict_set_and_forget=True,
        command_runner=_healthy_command_runner(tmp_path),
        now_epoch=1000.0,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["set_and_forget_live_gates"]["status"] == "error"
    assert checks["set_and_forget_live_gates"]["missing_or_disabled"] == [
        "supervisor_max_projected_duration_hours"
    ]


def test_operation_audit_strict_set_and_forget_rejects_disabled_graceful_restart(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runbook_path, _run_dir = _write_operation_fixture(tmp_path)
    payload = json.loads(runbook_path.read_text(encoding="utf-8"))
    command = payload["commands"]["watchdog"]
    command[command.index("--graceful-restart-timeout") + 1] = "0"
    runbook_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", lambda *_args, **_kwargs: _fake_artifact_report())
    report = operation.audit_operation(
        runbook_path,
        allow_running_incomplete=True,
        strict_set_and_forget=True,
        command_runner=_healthy_command_runner(tmp_path),
        now_epoch=1000.0,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["set_and_forget_watchdog_graceful_restart"]["status"] == "error"


def test_operation_audit_strict_set_and_forget_rejects_unsafe_launchagent_policy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runbook_path, run_dir = _write_operation_fixture(tmp_path)
    payload = json.loads(runbook_path.read_text(encoding="utf-8"))
    supervisor_command = payload["commands"]["supervisor"]
    _fixture_plist(tmp_path, "supervisor").write_bytes(
        plistlib.dumps(
            {
                **_plist_payload(
                    command=supervisor_command,
                    output_dir=run_dir,
                    stdout_name="launchd_stdout.log",
                    stderr_name="launchd_stderr.log",
                ),
                "RunAtLoad": False,
                "KeepAlive": {"SuccessfulExit": True},
                "ThrottleInterval": 0,
            }
        )
    )

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", lambda *_args, **_kwargs: _fake_artifact_report())
    report = operation.audit_operation(
        runbook_path,
        allow_running_incomplete=True,
        strict_set_and_forget=True,
        command_runner=_healthy_command_runner(tmp_path),
        now_epoch=1000.0,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["set_and_forget_launchd_policy"]["status"] == "error"
    violation = checks["set_and_forget_launchd_policy"]["policy_violations"][0]
    assert violation["role"] == "supervisor"
    assert "RunAtLoad must be true" in violation["violations"]
    assert "KeepAlive.SuccessfulExit must be false" in violation["violations"]
    assert "ThrottleInterval must be at least 10 seconds" in violation["violations"]


def test_operation_audit_strict_set_and_forget_rejects_nonpersistent_launchagent_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runbook_path, _run_dir = _write_operation_fixture(tmp_path)
    payload = json.loads(runbook_path.read_text(encoding="utf-8"))
    supervisor_plist = tmp_path / "run-local-supervisor.plist"
    watchdog_plist = tmp_path / "run-local-watchdog.plist"
    for source, target in (
        (tmp_path / "Library" / "LaunchAgents" / "supervisor.plist", supervisor_plist),
        (tmp_path / "Library" / "LaunchAgents" / "watchdog.plist", watchdog_plist),
    ):
        target.write_bytes(source.read_bytes())
    payload["launchd_install"]["roles"]["supervisor"]["plist_path"] = str(supervisor_plist)
    payload["launchd_install"]["roles"]["watchdog"]["plist_path"] = str(watchdog_plist)
    payload["commands"]["watchdog"] = [
        str(supervisor_plist) if item == str(tmp_path / "Library" / "LaunchAgents" / "supervisor.plist") else item
        for item in payload["commands"]["watchdog"]
    ]
    supervisor_plist.write_bytes(
        plistlib.dumps(
            _plist_payload(
                command=payload["commands"]["supervisor"],
                output_dir=_run_dir,
                stdout_name="launchd_stdout.log",
                stderr_name="launchd_stderr.log",
            )
        )
    )
    watchdog_plist.write_bytes(
        plistlib.dumps(
            _plist_payload(
                command=payload["commands"]["watchdog"],
                output_dir=_run_dir,
                stdout_name="watchdog_launchd_stdout.log",
                stderr_name="watchdog_launchd_stderr.log",
                caffeinate=True,
            )
        )
    )
    runbook_path.write_text(json.dumps(payload), encoding="utf-8")

    def fake_run(command, **_kwargs):
        if command[:2] == ["launchctl", "print"]:
            plist = watchdog_plist if command[-1].endswith("watchdog") else supervisor_plist
            arguments = plistlib.loads(plist.read_bytes())["ProgramArguments"]
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_launchd_stdout(
                    plist_path=plist,
                    arguments=arguments,
                    caffeinate=command[-1].endswith("watchdog"),
                ),
                stderr="",
            )
        if command[:3] == ["pmset", "-g", "assertions"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="PreventSystemSleep 1\npid 123(caffeinate): PreventSystemSleep\n",
                stderr="",
            )
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="unexpected")

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", lambda *_args, **_kwargs: _fake_artifact_report())
    report = operation.audit_operation(
        runbook_path,
        allow_running_incomplete=True,
        strict_set_and_forget=True,
        command_runner=fake_run,
        now_epoch=1000.0,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["set_and_forget_launchd_install"]["status"] == "ok"
    assert checks["set_and_forget_launchd_persistence"]["status"] == "error"
    assert checks["set_and_forget_launchd_policy"]["status"] == "ok"
    assert {item["role"] for item in checks["set_and_forget_launchd_persistence"]["non_persistent_roles"]} == {
        "supervisor",
        "watchdog",
    }


def test_operation_audit_fails_stale_watchdog_status(monkeypatch, tmp_path: Path) -> None:
    runbook_path, _run_dir = _write_operation_fixture(tmp_path, latest_checked_epoch=100.0)

    def fake_run(command, **_kwargs):
        if command[:2] == ["launchctl", "print"]:
            plist = _fixture_plist(tmp_path, "watchdog" if command[-1].endswith("watchdog") else "supervisor")
            arguments = plistlib.loads(plist.read_bytes())["ProgramArguments"]
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_launchd_stdout(
                    plist_path=plist,
                    arguments=arguments,
                    caffeinate=command[-1].endswith("watchdog"),
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="PreventSystemSleep 1\ncaffeinate\n", stderr="")

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", lambda *_args, **_kwargs: _fake_artifact_report())
    report = operation.audit_operation(runbook_path, command_runner=fake_run, now_epoch=1000.0)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["watchdog_latest"]["status"] == "error"
    assert "stale" in checks["watchdog_latest"]["detail"]


def test_operation_audit_fails_when_watchdog_is_not_caffeinate_wrapped(monkeypatch, tmp_path: Path) -> None:
    runbook_path, _run_dir = _write_operation_fixture(tmp_path)

    def fake_run(command, **_kwargs):
        if command[:2] == ["launchctl", "print"]:
            plist = _fixture_plist(tmp_path, "watchdog" if command[-1].endswith("watchdog") else "supervisor")
            arguments = plistlib.loads(plist.read_bytes())["ProgramArguments"]
            if command[-1].endswith("watchdog"):
                arguments = arguments[2:]
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_launchd_stdout(plist_path=plist, arguments=arguments),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="PreventSystemSleep 1\n", stderr="")

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", lambda *_args, **_kwargs: _fake_artifact_report())
    report = operation.audit_operation(runbook_path, command_runner=fake_run, now_epoch=1000.0)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["watchdog_launchd"]["status"] == "error"
    assert checks["sleep_assertion"]["status"] == "error"


def test_operation_audit_fails_when_loaded_watchdog_arguments_drift_from_plist(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runbook_path, _run_dir = _write_operation_fixture(tmp_path)

    def fake_run(command, **_kwargs):
        if command[:2] == ["launchctl", "print"]:
            plist = _fixture_plist(tmp_path, "watchdog" if command[-1].endswith("watchdog") else "supervisor")
            arguments = plistlib.loads(plist.read_bytes())["ProgramArguments"]
            if command[-1].endswith("watchdog"):
                arguments = [item for item in arguments if item not in {"--min-free-gb", "1"}]
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_launchd_stdout(
                    plist_path=plist,
                    arguments=arguments,
                    caffeinate=command[-1].endswith("watchdog"),
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="PreventSystemSleep 1\ncaffeinate\n", stderr="")

    monkeypatch.setattr(operation.artifact_audit, "audit_soak", lambda *_args, **_kwargs: _fake_artifact_report())
    report = operation.audit_operation(runbook_path, command_runner=fake_run, now_epoch=1000.0)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["watchdog_launchd"]["status"] == "ok"
    assert checks["watchdog_launchd_arguments"]["status"] == "error"
    assert "loaded arguments do not match" in checks["watchdog_launchd_arguments"]["detail"]
