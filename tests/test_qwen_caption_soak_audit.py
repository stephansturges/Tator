from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from tools import audit_qwen_caption_soak as audit


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload))


def _append_jsonl(path: Path, payload: object) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")
    if path.name != "results.jsonl" or not isinstance(payload, dict):
        return
    if payload.get("final_status") != "ok" or payload.get("status") == "preview_only":
        return
    case_id = str(payload.get("case_id") or "").strip()
    if not case_id:
        return
    with path.with_name("captions.jsonl").open("a") as handle:
        handle.write(json.dumps({"case_id": case_id, "caption": f"Caption for {case_id}."}) + "\n")


def _write_completed_soak(output_dir: Path) -> None:
    _write_json(
        output_dir / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    _append_jsonl(
        output_dir / "results.jsonl",
        {"case_id": "image:a:full", "final_status": "ok", "quality_failures": []},
    )
    _write_json(output_dir / "summary.json", {"total_cases": 1, "totals": {"ok": 1}})
    _write_json(
        output_dir / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )


def test_soak_audit_ok_for_consistent_completed_run(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {
            "cases": [
                {"case_id": "image:a:full", "image_name": "a.jpg"},
                {"case_id": "image:b:full", "image_name": "b.jpg"},
            ]
        },
    )
    _append_jsonl(
        tmp_path / "results.jsonl",
        {"case_id": "image:a:full", "final_status": "ok", "quality_failures": []},
    )
    _append_jsonl(
        tmp_path / "results.jsonl",
        {"case_id": "image:b:full", "final_status": "skipped_existing_caption", "quality_failures": []},
    )
    _write_json(
        tmp_path / "summary.json",
        {"total_cases": 2, "totals": {"ok": 1, "skipped_existing_caption": 1}},
    )
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )

    report = audit.audit_soak(tmp_path, max_heartbeat_age_seconds=60)

    assert report["status"] == "ok"
    assert report["expected_cases"] == 2
    assert report["processed_cases"] == 2
    assert report["failed_cases"] == 0
    assert report["quality_failed_cases"] == 0
    assert report["resumable"] is True
    checks = {check["name"]: check["status"] for check in report["checks"]}
    assert checks["case_coverage"] == "ok"
    assert checks["runner_lock"] == "ok"


def test_soak_audit_checks_live_disk_reserve(monkeypatch, tmp_path: Path) -> None:
    _write_completed_soak(tmp_path)
    monkeypatch.setattr(
        audit.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=200, used=80, free=120 * 1024 * 1024 * 1024),
    )

    report = audit.audit_soak(tmp_path, max_heartbeat_age_seconds=60, min_free_gb=5)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "ok"
    assert checks["disk_reserve"]["status"] == "ok"
    assert report["disk_reserve"]["min_free_gb"] == 5.0


def test_soak_audit_errors_when_live_disk_reserve_is_below_floor(monkeypatch, tmp_path: Path) -> None:
    _write_completed_soak(tmp_path)
    monkeypatch.setattr(
        audit.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=200, used=199, free=1 * 1024 * 1024 * 1024),
    )

    report = audit.audit_soak(tmp_path, max_heartbeat_age_seconds=60, min_free_gb=5)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["disk_reserve"]["status"] == "error"
    assert checks["disk_reserve"]["free_bytes"] == 1 * 1024 * 1024 * 1024


def test_soak_audit_set_and_forget_allows_rare_loop_recovery(tmp_path: Path) -> None:
    cases = [
        {"case_id": f"image:{index}:full", "image_name": f"{index}.jpg"}
        for index in range(50)
    ]
    _write_json(tmp_path / "manifest.json", {"cases": cases})
    for index, case in enumerate(cases):
        _append_jsonl(
            tmp_path / "results.jsonl",
            {
                "case_id": case["case_id"],
                "final_status": "ok",
                "quality_failures": [],
                "recovery_events": (
                    [{"action": "loop_detected"}, {"action": "safe_retry_succeeded"}]
                    if index == 0
                    else []
                ),
            },
        )
    _write_json(tmp_path / "summary.json", {"total_cases": 50, "totals": {"ok": 50}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )

    strict_report = audit.audit_soak(tmp_path, max_heartbeat_age_seconds=60)
    set_and_forget_report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        set_and_forget=True,
    )

    assert strict_report["status"] == "error"
    assert set_and_forget_report["status"] == "ok"
    assert set_and_forget_report["degraded_rates"]["loop_recovery_case_rate"] == 0.02
    assert (
        set_and_forget_report["degraded_rates"]["thresholds"]["max_loop_recovery_case_rate"]
        == 0.05
    )


def test_soak_audit_reports_stream_loop_guard_evidence(tmp_path: Path) -> None:
    cases = [
        {"case_id": f"image:{index}:full", "image_name": f"{index}.jpg"}
        for index in range(4)
    ]
    _write_json(tmp_path / "manifest.json", {"cases": cases})
    for index, case in enumerate(cases):
        _append_jsonl(
            tmp_path / "results.jsonl",
            {
                "case_id": case["case_id"],
                "final_status": "ok",
                "quality_failures": [],
                "recovery_events": (
                    [{"action": "loop_detected"}, {"action": "safe_retry_succeeded"}]
                    if index == 1
                    else []
                ),
                "qwen_caption_io": (
                    {
                        "event_counts": {
                            "stream_loop_detected": 2,
                            "loop_trim": 1,
                            "prompt_budget": 4,
                        }
                    }
                    if index == 1
                    else {"event_counts": {"prompt_budget": 3}}
                ),
            },
        )
    _write_json(tmp_path / "summary.json", {"total_cases": 4, "totals": {"ok": 4}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )

    report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        set_and_forget=True,
    )

    rates = report["degraded_rates"]
    assert report["status"] == "error"
    assert rates["stream_loop_detected_cases"] == 1
    assert rates["stream_loop_detected_events"] == 2
    assert rates["loop_trim_cases"] == 1
    assert rates["loop_trim_events"] == 1
    assert rates["loop_guard_cases"] == 1
    assert rates["stream_loop_detected_case_rate"] == 0.25
    assert rates["loop_trim_case_rate"] == 0.25
    assert rates["loop_guard_case_rate"] == 0.25
    assert rates["thresholds"]["max_loop_guard_case_rate"] == 0.05
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["degraded_case_rates"]["status"] == "error"
    assert "loop_guard_case_rate=0.250" in checks["degraded_case_rates"]["detail"]


def test_soak_audit_set_and_forget_allows_rare_loop_guard(tmp_path: Path) -> None:
    cases = [
        {"case_id": f"image:{index}:full", "image_name": f"{index}.jpg"}
        for index in range(50)
    ]
    _write_json(tmp_path / "manifest.json", {"cases": cases})
    for index, case in enumerate(cases):
        _append_jsonl(
            tmp_path / "results.jsonl",
            {
                "case_id": case["case_id"],
                "final_status": "ok",
                "quality_failures": [],
                "qwen_caption_io": (
                    {"event_counts": {"stream_loop_detected": 1, "loop_trim": 1}}
                    if index == 0
                    else {"event_counts": {"prompt_budget": 1}}
                ),
            },
        )
    _write_json(tmp_path / "summary.json", {"total_cases": 50, "totals": {"ok": 50}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )

    strict_report = audit.audit_soak(tmp_path, max_heartbeat_age_seconds=60)
    set_and_forget_report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        set_and_forget=True,
    )

    assert strict_report["status"] == "error"
    assert set_and_forget_report["status"] == "ok"
    assert set_and_forget_report["degraded_rates"]["loop_guard_case_rate"] == 0.02
    assert (
        set_and_forget_report["degraded_rates"]["thresholds"]["max_loop_guard_case_rate"]
        == 0.05
    )


def test_soak_audit_set_and_forget_allows_bounded_recovered_signal_exit(tmp_path: Path) -> None:
    cases = [
        {"case_id": f"image:{index}:full", "image_name": f"{index}.jpg"}
        for index in range(200)
    ]
    _write_json(tmp_path / "manifest.json", {"cases": cases})
    _append_jsonl(
        tmp_path / "results.jsonl",
        {
            "case_id": cases[0]["case_id"],
            "final_status": "failed_attempt",
            "status": "missing_result",
            "attempt_failure_kind": "signal_exit",
            "exit_code": -6,
            "return_signal": 6,
            "return_signal_name": "SIGABRT",
            "quality_failures": [],
        },
    )
    for case in cases:
        _append_jsonl(
            tmp_path / "results.jsonl",
            {
                "case_id": case["case_id"],
                "final_status": "ok",
                "status": "ok",
                "quality_failures": [],
            },
        )
    _write_json(tmp_path / "summary.json", {"total_cases": 200, "totals": {"ok": 200}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )

    strict_report = audit.audit_soak(tmp_path, max_heartbeat_age_seconds=60)
    set_and_forget_report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        set_and_forget=True,
    )
    explicit_zero_report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        set_and_forget=True,
        max_signal_exit_attempt_row_rate=0.0,
    )

    assert strict_report["status"] == "error"
    assert set_and_forget_report["status"] == "ok"
    assert explicit_zero_report["status"] == "error"
    rates = set_and_forget_report["degraded_rates"]
    assert rates["failed_cases"] == 0
    assert rates["signal_exit_attempt_rows"] == 1
    assert rates["signal_exit_names"] == {"SIGABRT": 1}
    assert rates["thresholds"]["max_signal_exit_attempt_row_rate"] == 0.05
    assert rates["signal_exit_attempt_row_rate"] < 0.05


def test_soak_audit_live_set_and_forget_delays_low_sample_loop_rate(tmp_path: Path) -> None:
    cases = [
        {"case_id": f"image:{index}:full", "image_name": f"{index}.jpg"}
        for index in range(100)
    ]
    _write_json(tmp_path / "manifest.json", {"cases": cases})
    for index in range(28):
        _append_jsonl(
            tmp_path / "results.jsonl",
            {
                "case_id": cases[index]["case_id"],
                "final_status": "ok",
                "quality_failures": [],
                "recovery_events": (
                    [{"action": "loop_detected"}, {"action": "safe_retry_succeeded"}]
                    if index in {3, 17}
                    else []
                ),
            },
        )
    _write_json(tmp_path / "summary.json", {"total_cases": 28, "totals": {"ok": 28}})
    _write_json(
        tmp_path / "heartbeat.json",
        {
            "status": "running",
            "phase": "attempt_running",
            "heartbeat_epoch": time.time(),
            "case": "image_live",
            "case_index": 99,
            "attempt": 2,
            "attempt_started_epoch": time.time() - 15,
            "attempt_timeout_seconds": 120,
            "worker_progress": {
                "run_id": "caption-run",
                "seq": 7,
                "step_id": "window_2",
                "generated_tokens": 42,
            },
            "worker_progress_seq": 7,
            "worker_step_id": "window_2",
            "worker_generated_tokens": 42,
        },
    )
    _write_json(
        tmp_path / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "live",
            "pid": os.getpid(),
            "phase": "attempt_running",
            "heartbeat_epoch": time.time(),
        },
    )

    report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        allow_running_incomplete=True,
        set_and_forget=True,
        min_rate_cases=20,
    )

    assert report["status"] == "ok"
    rates = report["degraded_rates"]
    assert rates["active"] is True
    assert rates["loop_recovery_case_rate"] == 2 / 28
    assert rates["min_loop_recovery_rate_cases"] == 60
    assert rates["violations"][0]["rate"] == "loop_recovery_case_rate"
    assert rates["violations"][0]["active"] is False
    assert rates["active_violations"] == []
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["degraded_case_rates"]["status"] == "ok"
    assert "below its activation floor" in checks["degraded_case_rates"]["detail"]


def test_soak_audit_reports_live_degraded_rate_headroom(tmp_path: Path) -> None:
    cases = [
        {"case_id": f"image:{index}:full", "image_name": f"{index}.jpg"}
        for index in range(1000)
    ]
    _write_json(tmp_path / "manifest.json", {"cases": cases})
    for index, case in enumerate(cases[:100]):
        _append_jsonl(
            tmp_path / "results.jsonl",
            {
                "case_id": case["case_id"],
                "final_status": "ok",
                "quality_failures": [],
                "preview_prompt_budget": {
                    "max_prompt_tokens": 1200 + index,
                    "adapted_sections": 1 if index in {4, 18} else 0,
                },
                "recovery_events": (
                    [{"action": "loop_detected"}, {"action": "safe_retry_succeeded"}]
                    if index in {3, 17, 39, 72}
                    else []
                ),
            },
        )
    _write_json(tmp_path / "summary.json", {"total_cases": 100, "totals": {"ok": 100}})
    _write_json(
        tmp_path / "heartbeat.json",
        {
            "status": "running",
            "phase": "attempt_running",
            "heartbeat_epoch": time.time(),
            "case": "image_live",
            "case_index": 99,
            "attempt": 2,
            "attempt_started_epoch": time.time() - 15,
            "attempt_timeout_seconds": 120,
            "worker_progress": {
                "run_id": "caption-run",
                "seq": 7,
                "step_id": "window_2",
                "generated_tokens": 42,
            },
            "worker_progress_seq": 7,
            "worker_step_id": "window_2",
            "worker_generated_tokens": 42,
        },
    )
    _write_json(
        tmp_path / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "live",
            "pid": os.getpid(),
            "phase": "attempt_running",
            "heartbeat_epoch": time.time(),
        },
    )

    report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        allow_running_incomplete=True,
        set_and_forget=True,
        min_rate_cases=20,
    )

    assert report["status"] == "ok"
    rates = report["degraded_rates"]
    loop_headroom = [
        item for item in rates["rate_headroom"] if item["rate"] == "loop_recovery_case_rate"
    ][0]
    assert loop_headroom["near_threshold"] is True
    assert loop_headroom["allowed_count_at_terminal_denominator"] == 50
    assert loop_headroom["remaining_count_at_terminal_denominator"] == 46
    assert loop_headroom["terminal_rate_if_no_more_events"] == 0.004
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["degraded_rate_headroom"]["status"] == "ok"
    assert "loop_recovery_case_rate" in checks["degraded_rate_headroom"]["detail"]


def test_soak_audit_compact_report_summarizes_operator_status(tmp_path: Path) -> None:
    cases = [
        {"case_id": f"image:{index}:full", "image_name": f"{index}.jpg"}
        for index in range(1000)
    ]
    _write_json(tmp_path / "manifest.json", {"cases": cases})
    _append_jsonl(
        tmp_path / "results.jsonl",
        {
            "case_id": cases[0]["case_id"],
            "final_status": "failed_attempt",
            "status": "missing_result",
            "attempt_failure_kind": "signal_exit",
            "exit_code": -6,
            "return_signal": 6,
            "return_signal_name": "SIGABRT",
            "quality_failures": [],
        },
    )
    for index, case in enumerate(cases[:100]):
        _append_jsonl(
            tmp_path / "results.jsonl",
            {
                "case_id": case["case_id"],
                "final_status": "ok",
                "quality_failures": [],
                "preview_prompt_budget": {
                    "max_prompt_tokens": 1200 + index,
                    "adapted_sections": 1 if index in {4, 18} else 0,
                },
                "recovery_events": (
                    [{"action": "loop_detected"}, {"action": "safe_retry_succeeded"}]
                    if index in {3, 17, 39, 72}
                    else []
                ),
            },
        )
    _write_json(tmp_path / "summary.json", {"total_cases": 100, "totals": {"ok": 100}})
    _write_json(
        tmp_path / "heartbeat.json",
        {
            "status": "running",
            "phase": "attempt_running",
            "heartbeat_epoch": time.time(),
            "case": "image_live",
            "case_index": 99,
            "attempt": 2,
            "attempt_started_epoch": time.time() - 15,
            "attempt_timeout_seconds": 120,
            "worker_progress": {
                "run_id": "caption-run",
                "seq": 7,
                "step_id": "window_2",
                "generated_tokens": 42,
            },
            "worker_progress_seq": 7,
            "worker_step_id": "window_2",
            "worker_generated_tokens": 42,
        },
    )
    _write_json(
        tmp_path / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "live",
            "pid": os.getpid(),
            "phase": "attempt_running",
            "heartbeat_epoch": time.time(),
        },
    )

    report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        allow_running_incomplete=True,
        set_and_forget=True,
        min_rate_cases=20,
    )

    assert report["active_attempt"]["case"] == "image_live"
    assert report["active_attempt"]["case_index"] == 99
    assert report["active_attempt"]["attempt"] == 2
    assert report["active_attempt"]["attempt_timeout_seconds"] == 120
    assert report["active_attempt"]["worker_progress"]["run_id"] == "caption-run"
    assert report["active_attempt"]["worker_progress_seq"] == 7
    assert report["active_attempt"]["worker_step_id"] == "window_2"
    text = audit.format_compact_report(report)

    assert "Qwen caption soak: OK" in text
    assert "Progress: 100/1000 cases complete (900 remaining)" in text
    assert "Caption rows: 100/100 generated/resumed successes covered" in text
    assert "Active attempt: image_live, index 99, attempt 2, runtime " in text
    assert " / timeout 120.0s" in text
    assert "Failures: 0 latest failed, 0 quality warnings, 0 pending failed attempts" in text
    assert "Loop recovery: 4 (4.00%; cap 5.00%, terminal budget remaining 46)" in text
    assert "Prompt budget: 1299 max prompt tokens, 100/100 rows with telemetry, 2 adapted (2.00%)" in text
    assert "Signal exits: 1 (0.99%; cap 5.00%, current budget remaining 4), SIGABRT=1" in text
    assert "Attention: loop_recovery_case_rate at 80.00% of cap" in text


def test_soak_audit_compact_cli_outputs_summary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    _append_jsonl(
        tmp_path / "results.jsonl",
        {"case_id": "image:a:full", "final_status": "ok", "quality_failures": []},
    )
    _write_json(tmp_path / "summary.json", {"total_cases": 1, "totals": {"ok": 1}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )

    monkeypatch.setattr(sys, "argv", ["audit_qwen_caption_soak.py", str(tmp_path), "--compact"])

    assert audit.main() == 0
    output = capsys.readouterr().out
    assert output.startswith("Qwen caption soak: OK\n")
    assert "Progress: 1/1 cases complete (0 remaining)" in output
    assert not output.lstrip().startswith("{")


def test_soak_audit_flags_projected_wall_time_over_budget(tmp_path: Path) -> None:
    cases = [
        {"case_id": f"image:{index}:full", "image_name": f"{index}.jpg"}
        for index in range(100)
    ]
    _write_json(tmp_path / "manifest.json", {"cases": cases})
    for case in cases[:10]:
        _append_jsonl(
            tmp_path / "results.jsonl",
            {"case_id": case["case_id"], "final_status": "ok", "quality_failures": []},
        )
    _write_json(tmp_path / "summary.json", {"total_cases": 10, "totals": {"ok": 10}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "running", "phase": "attempt_running", "heartbeat_epoch": time.time()},
    )
    _write_json(
        tmp_path / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "live",
            "pid": os.getpid(),
            "phase": "attempt_running",
            "heartbeat_epoch": time.time(),
            "started_epoch": time.time() - 3600,
        },
    )

    report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        allow_running_incomplete=True,
        max_projected_duration_hours=5,
        min_rate_cases=5,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["projected_wall_time"]["status"] == "error"
    assert checks["projected_wall_time"]["projected_duration_hours"] > 9
    assert report["runtime_projection"]["max_projected_duration_hours"] == 5


def test_soak_audit_delays_projected_wall_time_until_live_sample_floor(tmp_path: Path) -> None:
    cases = [
        {"case_id": f"image:{index}:full", "image_name": f"{index}.jpg"}
        for index in range(100)
    ]
    _write_json(tmp_path / "manifest.json", {"cases": cases})
    _append_jsonl(
        tmp_path / "results.jsonl",
        {"case_id": cases[0]["case_id"], "final_status": "ok", "quality_failures": []},
    )
    _write_json(tmp_path / "summary.json", {"total_cases": 1, "totals": {"ok": 1}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "running", "phase": "attempt_running", "heartbeat_epoch": time.time()},
    )
    _write_json(
        tmp_path / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "live",
            "pid": os.getpid(),
            "phase": "attempt_running",
            "heartbeat_epoch": time.time(),
            "started_epoch": time.time() - 3600,
        },
    )

    report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        allow_running_incomplete=True,
        max_projected_duration_hours=5,
        min_rate_cases=5,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "ok"
    assert checks["projected_wall_time"]["status"] == "ok"
    assert checks["projected_wall_time"]["active"] is False


def test_soak_audit_terminal_set_and_forget_enforces_loop_rate(tmp_path: Path) -> None:
    cases = [
        {"case_id": f"image:{index}:full", "image_name": f"{index}.jpg"}
        for index in range(28)
    ]
    _write_json(tmp_path / "manifest.json", {"cases": cases})
    for index, case in enumerate(cases):
        _append_jsonl(
            tmp_path / "results.jsonl",
            {
                "case_id": case["case_id"],
                "final_status": "ok",
                "quality_failures": [],
                "recovery_events": (
                    [{"action": "loop_detected"}, {"action": "safe_retry_succeeded"}]
                    if index in {3, 17}
                    else []
                ),
            },
        )
    _write_json(tmp_path / "summary.json", {"total_cases": 28, "totals": {"ok": 28}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )

    report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        set_and_forget=True,
        min_rate_cases=20,
    )

    assert report["status"] == "error"
    rates = report["degraded_rates"]
    assert rates["loop_recovery_case_rate"] == 2 / 28
    assert rates["violations"][0]["active"] is True
    assert rates["active_violations"][0]["rate"] == "loop_recovery_case_rate"


def test_soak_audit_reports_and_limits_deterministic_recovery(tmp_path: Path) -> None:
    cases = [
        {"case_id": f"image:{index}:full", "image_name": f"{index}.jpg"}
        for index in range(100)
    ]
    _write_json(tmp_path / "manifest.json", {"cases": cases})
    for index, case in enumerate(cases):
        recovery_events = []
        if index in {3, 17}:
            recovery_events = [
                {"action": "loop_detected"},
                {
                    "action": "deterministic_recovery_succeeded",
                    "attempt": "deterministic_recovery",
                    "call_kind": "deterministic",
                },
            ]
        _append_jsonl(
            tmp_path / "results.jsonl",
            {
                "case_id": case["case_id"],
                "final_status": "ok",
                "quality_failures": [],
                "recovery_events": recovery_events,
            },
        )
    _write_json(tmp_path / "summary.json", {"total_cases": 100, "totals": {"ok": 100}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )

    report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        set_and_forget=True,
    )

    assert report["status"] == "error"
    rates = report["degraded_rates"]
    assert rates["deterministic_recovery_cases"] == 2
    assert rates["deterministic_recovery_case_rate"] == 0.02
    assert rates["thresholds"]["max_deterministic_recovery_case_rate"] == 0.01
    assert rates["active_violations"][0]["rate"] == "deterministic_recovery_case_rate"


def test_soak_audit_flags_stale_running_heartbeat_and_summary_mismatch(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {
            "cases": [
                {"case_id": "image:a:full", "image_name": "a.jpg"},
                {"case_id": "image:b:full", "image_name": "b.jpg"},
            ]
        },
    )
    _append_jsonl(
        tmp_path / "results.jsonl",
        {
            "case_id": "image:a:full",
            "final_status": "failed",
            "status": "exception",
            "quality_failures": ["missing count: Boat"],
        },
    )
    _write_json(tmp_path / "summary.json", {"total_cases": 2, "totals": {"ok": 2}})
    _write_json(
        tmp_path / "heartbeat.json",
        {
            "status": "running",
            "phase": "attempt_running",
            "heartbeat_epoch": time.time() - 3600,
        },
    )

    report = audit.audit_soak(tmp_path, max_heartbeat_age_seconds=60)

    assert report["status"] == "error"
    assert report["processed_cases"] == 1
    assert report["incomplete_cases"] == 1
    assert report["failed_cases"] == 1
    assert report["quality_failed_cases"] == 1
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["heartbeat"]["status"] == "error"
    assert checks["summary_consistency"]["status"] == "error"
    assert checks["case_coverage"]["status"] == "warn"
    assert checks["failed_cases"]["status"] == "warn"
    assert checks["degraded_case_rates"]["status"] == "error"


def test_soak_audit_errors_on_invalid_results_jsonl_rows(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    (tmp_path / "results.jsonl").write_text(
        json.dumps({"case_id": "image:a:full", "final_status": "ok", "quality_failures": []})
        + "\n"
        + "{not valid json}\n"
        + json.dumps(["not", "an", "object"])
        + "\n"
    )
    _write_json(tmp_path / "summary.json", {"total_cases": 1, "totals": {"ok": 1}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )

    report = audit.audit_soak(tmp_path, max_heartbeat_age_seconds=60)

    assert report["status"] == "error"
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["results_jsonl"]["status"] == "error"
    assert "2 invalid row" in checks["results_jsonl"]["detail"]
    invalid_rows = checks["results_jsonl"]["invalid_rows"]
    assert invalid_rows[0]["line"] == 2
    assert "not valid json" in invalid_rows[0]["preview"]
    assert invalid_rows[1]["line"] == 3
    assert "not an object" in invalid_rows[1]["error"]
    assert report["processed_cases"] == 1


def test_soak_audit_errors_on_invalid_captions_jsonl_rows(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    _append_jsonl(
        tmp_path / "results.jsonl",
        {"case_id": "image:a:full", "final_status": "ok", "quality_failures": []},
    )
    (tmp_path / "captions.jsonl").write_text(
        json.dumps({"case_id": "image:a:full", "caption": "A concise caption."})
        + "\n"
        + "{bad caption row}\n"
        + json.dumps(["caption", "row"])
        + "\n"
    )
    _write_json(tmp_path / "summary.json", {"total_cases": 1, "totals": {"ok": 1}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )

    report = audit.audit_soak(tmp_path, max_heartbeat_age_seconds=60)

    assert report["status"] == "error"
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["captions_jsonl"]["status"] == "error"
    assert "2 invalid row" in checks["captions_jsonl"]["detail"]
    invalid_rows = checks["captions_jsonl"]["invalid_rows"]
    assert invalid_rows[0]["line"] == 2
    assert "bad caption row" in invalid_rows[0]["preview"]
    assert invalid_rows[1]["line"] == 3
    assert "not an object" in invalid_rows[1]["error"]
    assert checks["results_jsonl"]["status"] == "ok"


def test_soak_audit_requires_caption_rows_for_successful_generated_cases(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {
            "cases": [
                {"case_id": "image:a:full", "image_name": "a.jpg"},
                {"case_id": "image:b:full", "image_name": "b.jpg"},
            ]
        },
    )
    (tmp_path / "results.jsonl").write_text(
        json.dumps({"case_id": "image:a:full", "final_status": "ok", "status": "ok", "quality_failures": []})
        + "\n"
        + json.dumps({"case_id": "image:b:full", "final_status": "ok", "status": "ok", "quality_failures": []})
        + "\n"
    )
    (tmp_path / "captions.jsonl").write_text(
        json.dumps({"case_id": "image:a:full", "caption": "A caption for the first image."})
        + "\n"
    )
    _write_json(tmp_path / "summary.json", {"total_cases": 2, "totals": {"ok": 2}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )

    report = audit.audit_soak(tmp_path, max_heartbeat_age_seconds=60)

    assert report["status"] == "error"
    assert report["caption_coverage"]["required_generated_successes"] == 2
    assert report["caption_coverage"]["covered_generated_successes"] == 1
    assert report["caption_coverage"]["missing_cases"] == [
        {
            "case_id": "image:b:full",
            "final_status": "ok",
            "status": "ok",
        }
    ]
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["caption_coverage"]["status"] == "error"
    assert "1/2 latest successful generated/resumed cases" in checks["caption_coverage"]["detail"]


def test_soak_audit_requires_saved_text_label_rows_when_enabled(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    (tmp_path / "results.jsonl").write_text(
        json.dumps({"case_id": "image:a:full", "final_status": "ok", "status": "ok", "quality_failures": []})
        + "\n"
    )
    (tmp_path / "captions.jsonl").write_text(
        json.dumps({"case_id": "image:a:full", "caption": "A caption for the image."})
        + "\n"
    )
    _write_json(tmp_path / "summary.json", {"total_cases": 1, "totals": {"ok": 1}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )

    report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        require_saved_text_labels=True,
    )

    assert report["status"] == "error"
    coverage = report["saved_text_label_coverage"]
    assert coverage["required_saved_text_labels"] == 1
    assert coverage["covered_saved_text_labels"] == 0
    assert coverage["missing_saved_text_label_rows"] == 1
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["saved_text_label_coverage"]["status"] == "error"
    assert "0/1 latest successful generated/resumed cases" in checks["saved_text_label_coverage"]["detail"]
    assert "missing saved_text_label" in checks["saved_text_label_coverage"]["detail"]


def test_soak_audit_requires_saved_text_label_files_when_enabled(tmp_path: Path) -> None:
    empty_label = tmp_path / "labels" / "empty.txt"
    ok_label = tmp_path / "labels" / "ok.txt"
    empty_label.parent.mkdir(parents=True)
    empty_label.write_text("  \n")
    ok_label.write_text("A durable saved caption.\n")
    cases = [
        {"case_id": "image:a:full", "image_name": "a.jpg"},
        {"case_id": "image:b:full", "image_name": "b.jpg"},
        {"case_id": "image:c:full", "image_name": "c.jpg"},
    ]
    _write_json(tmp_path / "manifest.json", {"cases": cases})
    (tmp_path / "results.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case_id": "image:a:full",
                        "final_status": "ok",
                        "status": "ok",
                        "quality_failures": [],
                        "saved_text_label": "labels/missing.txt",
                    }
                ),
                json.dumps(
                    {
                        "case_id": "image:b:full",
                        "final_status": "ok",
                        "status": "ok",
                        "quality_failures": [],
                        "saved_text_label": str(empty_label),
                    }
                ),
                json.dumps(
                    {
                        "case_id": "image:c:full",
                        "final_status": "ok",
                        "status": "ok",
                        "quality_failures": [],
                        "saved_text_label": str(ok_label),
                    }
                ),
            ]
        )
        + "\n"
    )
    (tmp_path / "captions.jsonl").write_text(
        "\n".join(
            json.dumps({"case_id": case["case_id"], "caption": f"Caption for {case['image_name']}."})
            for case in cases
        )
        + "\n"
    )
    _write_json(tmp_path / "summary.json", {"total_cases": 3, "totals": {"ok": 3}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )

    report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        require_saved_text_labels=True,
    )

    assert report["status"] == "error"
    coverage = report["saved_text_label_coverage"]
    assert coverage["required_saved_text_labels"] == 3
    assert coverage["covered_saved_text_labels"] == 1
    assert coverage["missing_saved_text_label_files"] == 1
    assert coverage["empty_saved_text_label_files"] == 1
    assert coverage["missing_files"][0]["resolved_path"] == str(tmp_path / "labels" / "missing.txt")
    assert coverage["empty_files"][0]["resolved_path"] == str(empty_label)


def test_soak_audit_accepts_saved_text_label_files_when_enabled(tmp_path: Path) -> None:
    saved_label = tmp_path / "labels" / "a.txt"
    saved_label.parent.mkdir(parents=True)
    saved_label.write_text("A durable saved caption.\n")
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    (tmp_path / "results.jsonl").write_text(
        json.dumps(
            {
                "case_id": "image:a:full",
                "final_status": "ok",
                "status": "ok",
                "quality_failures": [],
                "saved_text_label": str(saved_label),
            }
        )
        + "\n"
    )
    (tmp_path / "captions.jsonl").write_text(
        json.dumps({"case_id": "image:a:full", "caption": "A caption for the image."})
        + "\n"
    )
    _write_json(tmp_path / "summary.json", {"total_cases": 1, "totals": {"ok": 1}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )

    report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        require_saved_text_labels=True,
    )

    assert report["status"] == "ok"
    coverage = report["saved_text_label_coverage"]
    assert coverage["required_saved_text_labels"] == 1
    assert coverage["covered_saved_text_labels"] == 1
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["saved_text_label_coverage"]["status"] == "ok"


def test_soak_audit_exempts_preview_and_existing_caption_skips_from_caption_rows(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {
            "cases": [
                {"case_id": "image:a:full", "image_name": "a.jpg"},
                {"case_id": "image:b:full", "image_name": "b.jpg"},
                {"case_id": "image:c:full", "image_name": "c.jpg"},
                {"case_id": "image:d:full", "image_name": "d.jpg"},
            ]
        },
    )
    (tmp_path / "results.jsonl").write_text(
        json.dumps(
            {
                "case_id": "image:a:full",
                "final_status": "ok",
                "status": "preview_only",
                "quality_failures": [],
            }
        )
        + "\n"
        + json.dumps(
            {
                "case_id": "image:b:full",
                "final_status": "skipped_existing_caption",
                "status": "skipped_existing_caption",
                "quality_failures": [],
            }
        )
        + "\n"
        + json.dumps(
            {
                "case_id": "image:c:full",
                "final_status": "skipped_completed",
                "status": "ok",
                "quality_failures": [],
            }
        )
        + "\n"
        + json.dumps(
            {
                "case_id": "image:d:full",
                "final_status": "skipped_completed",
                "status": "ok",
                "quality_failures": [],
            }
        )
        + "\n"
    )
    (tmp_path / "captions.jsonl").write_text(
        json.dumps({"case_id": "image:c:full", "caption": "Existing resumed caption."})
        + "\n"
        + json.dumps({"case_id": "image:d:full", "caption": "Another resumed caption."})
        + "\n"
    )
    _write_json(
        tmp_path / "summary.json",
        {
            "total_cases": 4,
            "totals": {
                "ok": 1,
                "skipped_existing_caption": 1,
                "skipped_completed": 2,
            },
        },
    )
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )

    report = audit.audit_soak(tmp_path, max_heartbeat_age_seconds=60)

    assert report["status"] == "ok"
    assert report["caption_coverage"]["required_generated_successes"] == 2
    assert report["caption_coverage"]["covered_generated_successes"] == 2
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["caption_coverage"]["status"] == "ok"


def test_soak_audit_requires_caption_rows_for_resume_completed_cases(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    (tmp_path / "results.jsonl").write_text(
        json.dumps(
            {
                "case_id": "image:a:full",
                "final_status": "skipped_completed",
                "status": "ok",
                "quality_failures": [],
            }
        )
        + "\n"
    )
    _write_json(tmp_path / "summary.json", {"total_cases": 1, "totals": {"skipped_completed": 1}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )

    report = audit.audit_soak(tmp_path, max_heartbeat_age_seconds=60)

    assert report["status"] == "error"
    assert report["caption_coverage"]["required_generated_successes"] == 1
    assert report["caption_coverage"]["covered_generated_successes"] == 0
    assert report["caption_coverage"]["missing_cases"] == [
        {
            "case_id": "image:a:full",
            "final_status": "skipped_completed",
            "status": "ok",
        }
    ]
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["caption_coverage"]["status"] == "error"
    assert "0/1 latest successful generated/resumed cases" in checks["caption_coverage"]["detail"]


def test_soak_audit_errors_on_empty_caption_rows_for_successful_cases(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    (tmp_path / "results.jsonl").write_text(
        json.dumps({"case_id": "image:a:full", "final_status": "ok", "status": "ok", "quality_failures": []})
        + "\n"
    )
    (tmp_path / "captions.jsonl").write_text(
        json.dumps({"case_id": "image:a:full", "caption": "   "})
        + "\n"
    )
    _write_json(tmp_path / "summary.json", {"total_cases": 1, "totals": {"ok": 1}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )

    report = audit.audit_soak(tmp_path, max_heartbeat_age_seconds=60)

    assert report["status"] == "error"
    assert report["caption_coverage"]["invalid_caption_rows"] == [
        {
            "line": 1,
            "case_id": "image:a:full",
            "missing_case_id": False,
            "empty_caption": True,
        }
    ]
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["caption_coverage"]["status"] == "error"
    assert "caption row(s) are missing" in checks["caption_coverage"]["detail"]


def test_soak_audit_health_mode_allows_fresh_running_incomplete_run(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {
            "cases": [
                {"case_id": "image:a:full", "image_name": "a.jpg"},
                {"case_id": "image:b:full", "image_name": "b.jpg"},
            ]
        },
    )
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "running", "phase": "attempt_running", "heartbeat_epoch": time.time()},
    )
    _write_json(
        tmp_path / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "live",
            "pid": os.getpid(),
            "phase": "attempt_running",
            "heartbeat_epoch": time.time(),
        },
    )

    strict_report = audit.audit_soak(tmp_path, max_heartbeat_age_seconds=60)
    health_report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        allow_running_incomplete=True,
    )

    assert strict_report["status"] == "error"
    assert health_report["status"] == "ok"
    assert health_report["processed_cases"] == 0
    assert health_report["incomplete_cases"] == 2
    checks = {check["name"]: check for check in health_report["checks"]}
    assert checks["results_jsonl"]["status"] == "ok"
    assert checks["summary_consistency"]["status"] == "ok"
    assert checks["runner_lock"]["status"] == "ok"
    assert checks["case_coverage"]["status"] == "ok"


def test_soak_audit_errors_when_active_attempt_exceeds_timeout_grace(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {
            "cases": [
                {"case_id": "image:a:full", "image_name": "a.jpg"},
                {"case_id": "image:b:full", "image_name": "b.jpg"},
            ]
        },
    )
    _append_jsonl(
        tmp_path / "results.jsonl",
        {"case_id": "image:a:full", "final_status": "ok", "quality_failures": []},
    )
    _write_json(
        tmp_path / "heartbeat.json",
        {
            "status": "running",
            "phase": "attempt_running",
            "heartbeat_epoch": time.time(),
            "case": "image_b",
            "case_index": 1,
            "attempt": 3,
            "attempt_started_epoch": time.time() - 200,
            "attempt_timeout_seconds": 120,
        },
    )

    report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        allow_running_incomplete=True,
        max_attempt_overrun_seconds=30,
    )

    assert report["status"] == "error"
    assert report["active_attempt"]["case"] == "image_b"
    assert report["active_attempt"]["attempt"] == 3
    assert report["active_attempt"]["runtime_seconds"] > 190
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["heartbeat"]["status"] == "ok"
    assert checks["attempt_timeout_overrun"]["status"] == "error"
    assert checks["attempt_timeout_overrun"]["overrun_seconds"] > 40


def test_soak_audit_allows_active_attempt_within_timeout_grace(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    _write_json(
        tmp_path / "heartbeat.json",
        {
            "status": "running",
            "phase": "attempt_running",
            "heartbeat_epoch": time.time(),
            "case": "image_a",
            "case_index": 0,
            "attempt": 1,
            "attempt_started_epoch": time.time() - 90,
            "attempt_timeout_seconds": 120,
        },
    )
    _write_json(
        tmp_path / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "live",
            "pid": os.getpid(),
            "phase": "attempt_running",
            "heartbeat_epoch": time.time(),
        },
    )

    report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        allow_running_incomplete=True,
        max_attempt_overrun_seconds=30,
    )

    assert report["status"] == "ok"
    assert report["active_attempt"]["case"] == "image_a"
    assert report["active_attempt"]["attempt_timeout_seconds"] == 120
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["attempt_timeout_overrun"]["status"] == "ok"


def test_soak_audit_warns_when_terminal_run_leaves_runner_lock(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    _append_jsonl(
        tmp_path / "results.jsonl",
        {"case_id": "image:a:full", "final_status": "ok", "quality_failures": []},
    )
    _write_json(tmp_path / "summary.json", {"total_cases": 1, "totals": {"ok": 1}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )
    _write_json(
        tmp_path / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "leftover",
            "pid": os.getpid(),
            "phase": "finished",
            "heartbeat_epoch": time.time(),
        },
    )

    report = audit.audit_soak(tmp_path, max_heartbeat_age_seconds=60)

    assert report["status"] == "warn"
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["runner_lock"]["status"] == "warn"
    assert "terminal heartbeat" in checks["runner_lock"]["detail"]


def test_soak_audit_errors_on_stale_live_runner_lock(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "running", "phase": "attempt_running", "heartbeat_epoch": time.time()},
    )
    _write_json(
        tmp_path / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "stale-live",
            "pid": os.getpid(),
            "phase": "attempt_running",
            "heartbeat_epoch": time.time() - 3600,
        },
    )

    report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        allow_running_incomplete=True,
    )

    assert report["status"] == "error"
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["runner_lock"]["status"] == "error"
    assert "owner is alive but stale" in checks["runner_lock"]["detail"]


def test_soak_audit_errors_on_live_runner_lock_with_dead_pid(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "running", "phase": "attempt_running", "heartbeat_epoch": time.time()},
    )
    _write_json(
        tmp_path / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "dead-live",
            "pid": -1,
            "phase": "attempt_running",
            "heartbeat_epoch": time.time(),
        },
    )

    report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        allow_running_incomplete=True,
    )

    assert report["status"] == "error"
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["runner_lock"]["status"] == "error"
    assert "owner pid is not alive" in checks["runner_lock"]["detail"]


def test_soak_audit_errors_when_terminal_degraded_rates_exceed_thresholds(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {
            "cases": [
                {"case_id": "image:a:full", "image_name": "a.jpg"},
                {"case_id": "image:b:full", "image_name": "b.jpg"},
            ]
        },
    )
    _append_jsonl(
        tmp_path / "results.jsonl",
        {
            "case_id": "image:a:full",
            "final_status": "failed_attempt",
            "status": "exception",
            "quality_failures": [],
            "recovery_events": [],
        },
    )
    _append_jsonl(
        tmp_path / "results.jsonl",
        {
            "case_id": "image:a:full",
            "final_status": "failed",
            "status": "exception",
            "quality_failures": ["missing counts: Boat"],
            "recovery_events": [{"action": "loop_detected", "stage": "full"}],
        },
    )
    _append_jsonl(
        tmp_path / "results.jsonl",
        {
            "case_id": "image:b:full",
            "final_status": "ok",
            "status": "ok",
            "quality_failures": [],
            "recovery_events": [],
            "preview_prompt_budget": {"adapted_sections": 1, "max_prompt_tokens": 9000},
            "qwen_caption_io": {
                "event_counts": {"stream_loop_detected": 1},
                "prompt_budget_events": 4,
                "prompt_budget_adapted_events": 1,
                "max_prompt_tokens": 12000,
            },
        },
    )
    _write_json(tmp_path / "summary.json", {"total_cases": 2, "totals": {"failed": 1, "ok": 1}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )

    report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        max_recovery_event_case_rate=0.1,
        max_failed_attempt_row_rate=0.1,
    )

    assert report["status"] == "error"
    assert report["degraded_rates"]["failed_case_rate"] == 0.5
    assert report["degraded_rates"]["quality_failure_rate"] == 0.5
    assert report["degraded_rates"]["recovery_event_case_rate"] == 0.5
    assert report["degraded_rates"]["loop_recovery_case_rate"] == 0.5
    assert report["degraded_rates"]["loop_guard_case_rate"] == 0.5
    assert report["degraded_rates"]["failed_attempt_row_rate"] == 2 / 3
    assert report["degraded_rates"]["prompt_budget_rows"] == 1
    assert report["degraded_rates"]["prompt_budget_coverage_rate"] == 0.5
    assert report["degraded_rates"]["prompt_budget_adapted_cases"] == 1
    assert report["degraded_rates"]["prompt_budget_adapted_case_rate"] == 0.5
    assert report["degraded_rates"]["max_prompt_tokens"] == 12000
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["degraded_case_rates"]["status"] == "error"
    assert "failed_case_rate=0.500" in checks["degraded_case_rates"]["detail"]
    assert "loop_recovery_case_rate=0.500" in checks["degraded_case_rates"]["detail"]
    assert "loop_guard_case_rate=0.500" in checks["degraded_case_rates"]["detail"]


def test_soak_audit_treats_live_failed_attempt_as_pending_retry(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {
            "cases": [
                {"case_id": "image:a:full", "image_name": "a.jpg"},
                {"case_id": "image:b:full", "image_name": "b.jpg"},
            ]
        },
    )
    _append_jsonl(
        tmp_path / "results.jsonl",
        {
            "case_id": "image:a:full",
            "final_status": "ok",
            "status": "ok",
            "quality_failures": [],
        },
    )
    _append_jsonl(
        tmp_path / "results.jsonl",
        {
            "case_id": "image:b:full",
            "final_status": "failed_attempt",
            "status": "exception",
            "attempt": 1,
            "next_attempt_cooldown_seconds": 5,
            "quality_failures": [],
        },
    )
    _write_json(tmp_path / "summary.json", {"total_cases": 1, "totals": {"ok": 1}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "running", "phase": "case_start", "heartbeat_epoch": time.time()},
    )
    _write_json(
        tmp_path / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "live",
            "pid": os.getpid(),
            "phase": "case_start",
            "heartbeat_epoch": time.time(),
        },
    )

    report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        allow_running_incomplete=True,
        max_failed_case_rate=0.0,
        max_failed_attempt_row_rate=1.0,
    )

    assert report["status"] == "ok"
    assert report["failed_cases"] == 0
    assert report["pending_failed_attempt_cases"] == 1
    assert report["degraded_rates"]["failed_case_rate"] == 0.0
    assert report["degraded_rates"]["failed_attempt_row_rate"] == 0.5
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["failed_cases"]["status"] == "ok"
    assert "retryable failed attempt" in checks["failed_cases"]["detail"]


def test_soak_audit_treats_exhausted_live_failed_attempt_as_failed_case(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {
            "cases": [
                {"case_id": "image:a:full", "image_name": "a.jpg"},
                {"case_id": "image:b:full", "image_name": "b.jpg"},
            ]
        },
    )
    _append_jsonl(
        tmp_path / "results.jsonl",
        {
            "case_id": "image:a:full",
            "final_status": "ok",
            "status": "ok",
            "quality_failures": [],
        },
    )
    _append_jsonl(
        tmp_path / "results.jsonl",
        {
            "case_id": "image:b:full",
            "final_status": "failed_attempt",
            "status": "exception",
            "attempt": 2,
            "quality_failures": [],
        },
    )
    _write_json(tmp_path / "summary.json", {"total_cases": 1, "totals": {"ok": 1}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "running", "phase": "case_start", "heartbeat_epoch": time.time()},
    )
    _write_json(
        tmp_path / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "live",
            "pid": os.getpid(),
            "phase": "case_start",
            "heartbeat_epoch": time.time(),
        },
    )

    report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        allow_running_incomplete=True,
        max_failed_case_rate=0.0,
        max_failed_attempt_row_rate=1.0,
    )

    assert report["status"] == "error"
    assert report["failed_cases"] == 1
    assert report["pending_failed_attempt_cases"] == 0
    assert report["degraded_rates"]["failed_case_rate"] == 0.5
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["failed_cases"]["status"] == "warn"
    assert checks["degraded_case_rates"]["status"] == "error"
    assert "failed_case_rate=0.500" in checks["degraded_case_rates"]["detail"]


def test_soak_audit_tracks_signal_exit_attempt_rate_even_when_latest_case_ok(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    _append_jsonl(
        tmp_path / "results.jsonl",
        {
            "case_id": "image:a:full",
            "final_status": "failed_attempt",
            "status": "missing_result",
            "attempt_failure_kind": "signal_exit",
            "exit_code": -6,
            "return_signal": 6,
            "return_signal_name": "SIGABRT",
            "quality_failures": [],
        },
    )
    _append_jsonl(
        tmp_path / "results.jsonl",
        {
            "case_id": "image:a:full",
            "final_status": "ok",
            "status": "ok",
            "quality_failures": [],
        },
    )
    _write_json(tmp_path / "summary.json", {"total_cases": 1, "totals": {"ok": 1}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )

    report = audit.audit_soak(tmp_path, max_heartbeat_age_seconds=60)

    assert report["status"] == "error"
    rates = report["degraded_rates"]
    assert rates["failed_cases"] == 0
    assert rates["signal_exit_attempt_rows"] == 1
    assert rates["signal_exit_attempt_row_rate"] == 0.5
    assert rates["signal_exit_names"] == {"SIGABRT": 1}
    checks = {check["name"]: check for check in report["checks"]}
    assert "signal_exit_attempt_row_rate=0.500" in checks["degraded_case_rates"]["detail"]

    relaxed = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        max_failed_attempt_row_rate=1.0,
        max_signal_exit_attempt_row_rate=1.0,
    )
    assert relaxed["status"] == "ok"


def test_soak_audit_defers_live_signal_exit_attempts_until_terminal(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    _append_jsonl(
        tmp_path / "results.jsonl",
        {
            "case_id": "image:a:full",
            "final_status": "failed_attempt",
            "status": "missing_result",
            "attempt_failure_kind": "signal_exit",
            "exit_code": -6,
            "return_signal": 6,
            "return_signal_name": "SIGABRT",
            "quality_failures": [],
        },
    )
    _append_jsonl(
        tmp_path / "results.jsonl",
        {
            "case_id": "image:a:full",
            "final_status": "ok",
            "status": "ok",
            "quality_failures": [],
        },
    )
    _write_json(tmp_path / "summary.json", {"total_cases": 1, "totals": {"ok": 1}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "running", "phase": "case_start", "heartbeat_epoch": time.time()},
    )
    _write_json(
        tmp_path / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "live",
            "pid": os.getpid(),
            "phase": "case_start",
            "heartbeat_epoch": time.time(),
        },
    )

    report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        allow_running_incomplete=True,
        max_failed_attempt_row_rate=1.0,
        max_signal_exit_attempt_row_rate=0.0,
        min_rate_cases=1,
    )

    assert report["status"] == "ok"
    rates = report["degraded_rates"]
    assert rates["active"] is True
    assert rates["failed_cases"] == 0
    assert rates["signal_exit_attempt_rows"] == 1
    assert rates["signal_exit_attempt_row_rate"] == 0.5
    assert rates["active_violations"] == []
    assert rates["violations"][0]["rate"] == "signal_exit_attempt_row_rate"
    assert rates["violations"][0]["active"] is False
    assert rates["violations"][0]["deferred_until_terminal"] is True
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["degraded_case_rates"]["status"] == "ok"
    assert "enforced by terminal audit" in checks["degraded_case_rates"]["detail"]


def test_soak_audit_rate_gate_waits_for_minimum_running_sample(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {
            "cases": [
                {"case_id": "image:a:full", "image_name": "a.jpg"},
                {"case_id": "image:b:full", "image_name": "b.jpg"},
            ]
        },
    )
    _append_jsonl(
        tmp_path / "results.jsonl",
        {
            "case_id": "image:a:full",
            "final_status": "ok",
            "status": "ok",
            "quality_failures": [],
            "recovery_events": [{"action": "text_recovery_succeeded"}],
        },
    )
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "running", "phase": "case_start", "heartbeat_epoch": time.time()},
    )
    _write_json(
        tmp_path / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "live",
            "pid": os.getpid(),
            "phase": "case_start",
            "heartbeat_epoch": time.time(),
        },
    )

    report = audit.audit_soak(
        tmp_path,
        max_heartbeat_age_seconds=60,
        allow_running_incomplete=True,
        max_recovery_event_case_rate=0.0,
        min_rate_cases=20,
    )

    assert report["status"] == "warn"
    assert report["degraded_rates"]["active"] is False
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["degraded_case_rates"]["status"] == "warn"
    assert "below the active gate" in checks["degraded_case_rates"]["detail"]
