from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from tools import run_qwen_caption_soak_drill as drill


ROOT = Path(__file__).resolve().parents[1]


def test_soak_drill_exercises_supervisor_restart_path(tmp_path: Path) -> None:
    report = drill.run_drill(tmp_path / "drill")

    assert report["status"] == "ok"
    checks = report["checks"]
    assert checks["supervisor_success"] is True
    assert checks["saw_nonzero_exit"] is True
    assert checks["saw_signal_exit"] is True
    assert checks["saw_stale_heartbeat"] is True
    assert checks["saw_missing_heartbeat"] is True
    assert checks["saw_supervisor_restart"] is True
    assert checks["saw_supervisor_complete"] is True
    assert checks["final_audit_ok"] is True
    assert checks["caption_io_retention_ok"] is True
    assert report["caption_io_retention"]["checks"]["active_jsonl_kept"] is True
    assert Path(report["report_json"]).exists()


def test_soak_drill_cli_writes_json_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "drill"
    report_path = tmp_path / "drill_report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_qwen_caption_soak_drill.py",
            "--output-dir",
            str(output_dir),
            "--write-json",
            str(report_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    stdout_report = json.loads(completed.stdout)
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert stdout_report["status"] == "ok"
    assert saved_report["status"] == "ok"
    assert stdout_report["checks"]["saw_stale_heartbeat"] is True
    assert saved_report["checks"]["saw_missing_heartbeat"] is True
    assert saved_report["checks"]["caption_io_retention_ok"] is True


def test_soak_endurance_drill_processes_many_cases_across_restarts(tmp_path: Path) -> None:
    report = drill.run_endurance_drill(
        tmp_path / "endurance",
        case_count=37,
        chunk_size=8,
        summary_row_limit=5,
        max_runner_restarts=12,
    )

    assert report["status"] == "ok"
    checks = report["checks"]
    assert checks["supervisor_success"] is True
    assert checks["saw_nonzero_exit"] is True
    assert checks["saw_signal_exit"] is True
    assert checks["saw_stale_heartbeat"] is True
    assert checks["saw_missing_heartbeat"] is True
    assert checks["saw_multiple_restarts"] is True
    assert checks["final_audit_ok"] is True
    assert checks["all_cases_processed"] is True
    assert checks["all_latest_rows_ok"] is True
    assert checks["prompt_budget_recorded"] is True
    assert checks["summary_totals_cover_all_cases"] is True
    assert checks["summary_snapshot_bounded"] is True
    assert checks["summary_truncated_when_over_limit"] is True
    assert checks["summary_omits_rows_when_over_limit"] is True
    assert checks["caption_io_retention_ok"] is True
    assert report["summary_metrics"]["summary_row_count"] == 37
    assert report["summary_metrics"]["summary_rows_in_snapshot"] == 5
    assert report["summary_metrics"]["summary_rows_omitted"] == 32
    assert report["summary_metrics"]["summary_rows_truncated"] is True
    assert report["final_audit"]["processed_cases"] == 37
    assert report["final_audit"]["expected_cases"] == 37
    assert report["final_audit"]["totals"] == {"ok": 37}
    assert report["event_counts"]["supervisor_restart"] >= 3


def test_soak_endurance_drill_cli_writes_json_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "endurance"
    report_path = tmp_path / "endurance_report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_qwen_caption_soak_drill.py",
            "--output-dir",
            str(output_dir),
            "--endurance-cases",
            "31",
            "--endurance-chunk-size",
            "10",
            "--summary-row-limit",
            "7",
            "--write-json",
            str(report_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    stdout_report = json.loads(completed.stdout)
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert stdout_report["status"] == "ok"
    assert saved_report["status"] == "ok"
    assert saved_report["summary_metrics"]["summary_rows_in_snapshot"] == 7
    assert saved_report["summary_metrics"]["summary_rows_omitted"] == 24
    assert saved_report["case_count"] == 31
    assert saved_report["checks"]["all_cases_processed"] is True
    assert saved_report["checks"]["caption_io_retention_ok"] is True


def test_caption_io_retention_drill_prunes_global_trace_cache(tmp_path: Path) -> None:
    report = drill.run_caption_io_retention_drill(tmp_path / "caption_io_retention")

    assert report["status"] == "ok"
    checks = report["checks"]
    assert checks["active_jsonl_kept"] is True
    assert checks["active_log_kept"] is True
    assert checks["old_run_logs_pruned"] is True
    assert checks["file_cap_respected"] is True
    assert checks["byte_cap_respected"] is True
    assert checks["latest_jsonl_reset"] is True
    assert checks["latest_log_reset"] is True
    assert checks["symlink_not_followed"] is True
    assert report["direct_files_after"] <= report["max_files"]
    assert report["direct_bytes_after"] <= report["max_bytes"]
    assert Path(report["report_json"]).exists()


def test_caption_io_retention_drill_cli_writes_json_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "caption_io_retention"
    report_path = tmp_path / "caption_io_retention_report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_qwen_caption_soak_drill.py",
            "--output-dir",
            str(output_dir),
            "--caption-io-retention-only",
            "--write-json",
            str(report_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    stdout_report = json.loads(completed.stdout)
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert stdout_report["status"] == "ok"
    assert saved_report["status"] == "ok"
    assert saved_report["checks"]["old_run_logs_pruned"] is True


def test_watchdog_remediation_drill_exercises_rebootstrap_path(tmp_path: Path) -> None:
    report = drill.run_watchdog_remediation_drill(tmp_path / "watchdog")

    assert report["status"] == "ok"
    checks = report["checks"]
    assert checks["watchdog_success"] is True
    assert checks["saw_unhealthy_status"] is True
    assert checks["saw_launchd_rebootstrap"] is True
    assert checks["saw_kickstart_failure"] is True
    assert checks["saw_bootstrap_success"] is True
    assert checks["saw_restored_health"] is True
    assert checks["history_is_compact"] is True
    assert checks["latest_is_full"] is True
    assert checks["latest_status_ok"] is True
    assert checks["state_persisted"] is True
    assert [command[0] for command in report["launchctl_commands"]] == [
        "kickstart",
        "bootout",
        "bootstrap",
    ]
    assert Path(report["report_json"]).exists()


def test_watchdog_remediation_drill_cli_writes_json_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "watchdog"
    report_path = tmp_path / "watchdog_report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_qwen_caption_soak_drill.py",
            "--output-dir",
            str(output_dir),
            "--watchdog-remediation",
            "--write-json",
            str(report_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    stdout_report = json.loads(completed.stdout)
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert stdout_report["status"] == "ok"
    assert saved_report["status"] == "ok"
    assert saved_report["checks"]["saw_launchd_rebootstrap"] is True
