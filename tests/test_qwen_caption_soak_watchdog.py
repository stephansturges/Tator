from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from tools import audit_qwen_caption_soak as audit
from tools import watch_qwen_caption_soak as watch


ROOT = Path(__file__).resolve().parents[1]


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


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_watchdog_exits_zero_for_clean_completed_run(tmp_path: Path) -> None:
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
    log_path = tmp_path / "watchdog.jsonl"
    latest_path = tmp_path / watch.LATEST_STATUS_NAME

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        interval_seconds=0,
        max_heartbeat_age_seconds=60,
    )

    assert exit_code == 0
    rows = _read_jsonl(log_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["terminal"] is True
    assert rows[0]["strict_completion"] is True


def test_watchdog_requires_saved_text_labels_in_live_and_terminal_audits(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_audit_soak(output_dir, **kwargs):
        calls.append((output_dir, kwargs))
        return {
            "status": "ok",
            "output_dir": str(output_dir),
            "heartbeat": {"status": "completed", "phase": "finished"},
            "processed_cases": 1,
            "expected_cases": 1,
            "checks": [],
            "degraded_rates": {},
        }

    monkeypatch.setattr(watch.audit, "audit_soak", fake_audit_soak)

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=tmp_path / "watchdog.jsonl",
        interval_seconds=0,
        max_checks=1,
        require_saved_text_labels=True,
    )

    assert exit_code == 0
    assert len(calls) == 2
    assert calls[0][1]["allow_running_incomplete"] is True
    assert calls[0][1]["require_saved_text_labels"] is True
    assert calls[1][1]["allow_running_incomplete"] is False
    assert calls[1][1]["require_saved_text_labels"] is True


def test_watchdog_records_healthy_running_checks(tmp_path: Path) -> None:
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
        {
            "status": "running",
            "phase": "attempt_running",
            "heartbeat_epoch": time.time(),
            "case": "image_b",
            "case_index": 1,
            "attempt": 2,
            "attempt_started_epoch": time.time() - 5,
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
    log_path = tmp_path / "watchdog.jsonl"
    latest_path = tmp_path / watch.LATEST_STATUS_NAME

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        interval_seconds=0,
        max_heartbeat_age_seconds=60,
        max_consecutive_unhealthy=1,
        max_checks=2,
    )

    assert exit_code == 0
    rows = _read_jsonl(log_path)
    assert [row["check_index"] for row in rows] == [1, 2]
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["event_detail"] == "compact" for row in rows)
    assert all(row["terminal"] is False for row in rows)
    assert all(row["strict_completion"] is False for row in rows)
    assert all("check_counts" in row for row in rows)
    assert all(isinstance(row.get("checked_at"), str) and row["checked_at"] for row in rows)
    assert all(isinstance(row.get("checked_epoch"), float) and row["checked_epoch"] > 0 for row in rows)
    assert all(row["time"] == row["checked_epoch"] for row in rows)
    assert all(
        "rate_headroom" not in (row.get("degraded_rates") or {})
        for row in rows
    )
    latest = json.loads(latest_path.read_text())
    assert latest["event_detail"] == "full"
    assert latest["check_index"] == 2
    assert latest["status"] == "ok"
    assert latest["processed_cases"] == 0
    assert latest["active_attempt"]["case"] == "image_b"
    assert latest["active_attempt"]["case_index"] == 1
    assert latest["active_attempt"]["attempt"] == 2
    assert latest["active_attempt"]["attempt_timeout_seconds"] == 120
    assert "rate_headroom" in latest.get("degraded_rates", {})
    state = json.loads((tmp_path / watch.WATCHDOG_STATE_NAME).read_text())
    assert state["last_progress_cases"] == 0
    assert state["remediation_count"] == 0


def test_watchdog_writes_explicit_latest_status_snapshot(tmp_path: Path) -> None:
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
    log_path = tmp_path / "events" / "watchdog.jsonl"
    latest_path = tmp_path / "status" / "latest.json"

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        latest_json=latest_path,
        interval_seconds=0,
        max_heartbeat_age_seconds=60,
    )

    assert exit_code == 0
    rows = _read_jsonl(log_path)
    latest = json.loads(latest_path.read_text())
    assert rows[-1]["event_detail"] == "compact"
    assert latest["event_detail"] == "full"
    assert latest["check_index"] == rows[-1]["check_index"]
    assert latest["status"] == rows[-1]["status"]
    assert latest["checked_at"] == rows[-1]["checked_at"]
    assert latest["checked_epoch"] == rows[-1]["checked_epoch"]
    assert latest["time"] == rows[-1]["time"]
    assert latest["processed_cases"] == rows[-1]["processed_cases"]
    assert latest["expected_cases"] == rows[-1]["expected_cases"]
    assert latest["watchdog_state"] == rows[-1]["watchdog_state"]
    assert "rate_headroom" not in (rows[-1].get("degraded_rates") or {})
    assert "rate_headroom" in latest.get("degraded_rates", {})
    assert latest["terminal"] is True
    assert latest["strict_completion"] is True


def test_watchdog_exits_after_unhealthy_threshold(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    _write_json(
        tmp_path / "heartbeat.json",
        {
            "status": "running",
            "phase": "attempt_running",
            "heartbeat_epoch": time.time() - 3600,
        },
    )
    log_path = tmp_path / "watchdog.jsonl"

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        interval_seconds=0,
        max_heartbeat_age_seconds=60,
        max_consecutive_unhealthy=2,
        max_checks=10,
    )

    assert exit_code == 2
    rows = _read_jsonl(log_path)
    assert len(rows) == 2
    assert [row["consecutive_unhealthy"] for row in rows] == [1, 2]
    assert all(row["status"] == "error" for row in rows)


def test_watchdog_exits_when_active_attempt_exceeds_timeout_grace(tmp_path: Path) -> None:
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
            "attempt": 2,
            "attempt_started_epoch": time.time() - 200,
            "attempt_timeout_seconds": 120,
        },
    )
    log_path = tmp_path / "watchdog.jsonl"

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        interval_seconds=0,
        max_heartbeat_age_seconds=60,
        max_attempt_overrun_seconds=30,
        max_consecutive_unhealthy=1,
        max_checks=10,
    )

    assert exit_code == 2
    rows = _read_jsonl(log_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["active_attempt"]["case"] == "image_b"
    assert rows[0]["active_attempt"]["attempt"] == 2
    checks = {check["name"]: check for check in rows[0]["checks"]}
    assert checks["heartbeat"]["status"] == "ok"
    assert checks["attempt_timeout_overrun"]["status"] == "error"


def test_watchdog_exits_when_completed_case_progress_stalls(monkeypatch, tmp_path: Path) -> None:
    fake_now = {"value": 100.0}

    monkeypatch.setattr(watch.time, "time", lambda: fake_now["value"])
    monkeypatch.setattr(
        watch.time,
        "sleep",
        lambda _seconds: fake_now.__setitem__("value", fake_now["value"] + 10.0),
    )
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
    log_path = tmp_path / "watchdog.jsonl"

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        interval_seconds=1,
        max_heartbeat_age_seconds=60,
        max_no_progress_seconds=5,
        max_consecutive_unhealthy=1,
        max_checks=10,
    )

    assert exit_code == 2
    rows = _read_jsonl(log_path)
    assert [row["status"] for row in rows] == ["ok", "error"]
    assert rows[1]["progress_watch"]["stalled"] is True
    checks = {check["name"]: check for check in rows[1]["checks"]}
    assert checks["watchdog_case_progress"]["status"] == "error"
    state = json.loads((tmp_path / watch.WATCHDOG_STATE_NAME).read_text())
    assert state["last_progress_cases"] == 1
    assert state["consecutive_unhealthy"] == 1


def test_watchdog_progress_timer_resets_when_case_count_advances(monkeypatch, tmp_path: Path) -> None:
    fake_now = {"value": 100.0}
    appended = {"done": False}

    def fake_sleep(_seconds: float) -> None:
        fake_now["value"] += 10.0
        if not appended["done"]:
            _append_jsonl(
                tmp_path / "results.jsonl",
                {"case_id": "image:b:full", "final_status": "ok", "quality_failures": []},
            )
            appended["done"] = True

    monkeypatch.setattr(watch.time, "time", lambda: fake_now["value"])
    monkeypatch.setattr(watch.time, "sleep", fake_sleep)
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
    log_path = tmp_path / "watchdog.jsonl"

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        interval_seconds=1,
        max_heartbeat_age_seconds=60,
        max_no_progress_seconds=5,
        max_consecutive_unhealthy=1,
        max_checks=2,
    )

    assert exit_code == 0
    rows = _read_jsonl(log_path)
    assert [row["status"] for row in rows] == ["ok", "ok"]
    assert rows[1]["processed_cases"] == 2
    assert rows[1]["progress_watch"]["seconds_since_progress"] == 0.0
    state = json.loads((tmp_path / watch.WATCHDOG_STATE_NAME).read_text())
    assert state["last_progress_cases"] == 2
    assert state["last_progress_epoch"] == 110.0


def test_watchdog_progress_timer_resets_when_worker_progress_advances(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fake_now = {"value": 100.0}
    real_now = time.time()
    advanced = {"done": False}

    def write_active_heartbeat(seq: int, generated_tokens: int) -> None:
        _write_json(
            tmp_path / "heartbeat.json",
            {
                "status": "running",
                "phase": "attempt_running",
                "heartbeat_epoch": time.time(),
                "case": "image_a",
                "case_id": "image:a:windowed",
                "case_index": 1,
                "attempt": 1,
                "attempt_started_epoch": real_now - 10,
                "attempt_timeout_seconds": 120,
                "worker_progress": {
                    "run_id": "caption-run",
                    "seq": seq,
                    "phase": "generate",
                    "step_id": "window_1",
                    "step_label": "Window observation 1/4",
                    "generated_tokens": generated_tokens,
                    "live_output_chars": generated_tokens * 12,
                    "token_preview_chars": generated_tokens * 4,
                    "io_event_count": 8,
                    "updated_at": fake_now["value"],
                },
            },
        )

    def fake_sleep(_seconds: float) -> None:
        fake_now["value"] += 10.0
        if not advanced["done"]:
            write_active_heartbeat(seq=2, generated_tokens=25)
            advanced["done"] = True

    monkeypatch.setattr(watch.time, "time", lambda: fake_now["value"])
    monkeypatch.setattr(watch.time, "sleep", fake_sleep)
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:windowed", "image_name": "a.jpg"}]},
    )
    _write_json(tmp_path / "summary.json", {"total_cases": 0, "totals": {}})
    write_active_heartbeat(seq=1, generated_tokens=10)
    _write_json(
        tmp_path / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "live",
            "pid": os.getpid(),
            "phase": "attempt_running",
            "heartbeat_epoch": time.time(),
        },
    )
    log_path = tmp_path / "watchdog.jsonl"

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        interval_seconds=1,
        max_heartbeat_age_seconds=60,
        max_no_progress_seconds=5,
        max_consecutive_unhealthy=1,
        max_checks=2,
    )

    assert exit_code == 0
    rows = _read_jsonl(log_path)
    assert [row["status"] for row in rows] == ["ok", "ok"]
    assert rows[1]["progress_watch"]["progress_source"] == "worker"
    assert rows[1]["progress_watch"]["progress_changed"] is True
    assert rows[1]["progress_watch"]["seconds_since_progress"] == 0.0
    assert rows[1]["progress_watch"]["worker_progress"]["seq"] == 2
    state = json.loads((tmp_path / watch.WATCHDOG_STATE_NAME).read_text())
    assert state["last_progress_cases"] == 0
    assert state["last_progress_epoch"] == 110.0
    assert state["last_progress_source"] == "worker"


def test_watchdog_exits_when_worker_progress_is_static(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fake_now = {"value": 100.0}
    real_now = time.time()

    monkeypatch.setattr(watch.time, "time", lambda: fake_now["value"])
    monkeypatch.setattr(
        watch.time,
        "sleep",
        lambda _seconds: fake_now.__setitem__("value", fake_now["value"] + 10.0),
    )
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:windowed", "image_name": "a.jpg"}]},
    )
    _write_json(tmp_path / "summary.json", {"total_cases": 0, "totals": {}})
    _write_json(
        tmp_path / "heartbeat.json",
        {
            "status": "running",
            "phase": "attempt_running",
            "heartbeat_epoch": time.time(),
            "case": "image_a",
            "case_id": "image:a:windowed",
            "case_index": 1,
            "attempt": 1,
            "attempt_started_epoch": real_now - 10,
            "attempt_timeout_seconds": 120,
            "worker_progress": {
                "run_id": "caption-run",
                "seq": 1,
                "phase": "generate",
                "step_id": "window_1",
                "generated_tokens": 10,
                "live_output_chars": 120,
                "token_preview_chars": 40,
                "io_event_count": 8,
                "updated_at": 100.0,
            },
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
    log_path = tmp_path / "watchdog.jsonl"

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        interval_seconds=1,
        max_heartbeat_age_seconds=60,
        max_no_progress_seconds=5,
        max_consecutive_unhealthy=1,
        max_checks=3,
    )

    assert exit_code == 2
    rows = _read_jsonl(log_path)
    assert [row["status"] for row in rows] == ["ok", "error"]
    assert rows[1]["progress_watch"]["progress_source"] == "worker"
    assert rows[1]["progress_watch"]["stalled"] is True
    checks = {check["name"]: check for check in rows[1]["checks"]}
    assert checks["watchdog_case_progress"]["status"] == "error"
    assert checks["watchdog_case_progress"]["progress_source"] == "worker"


def test_watchdog_uses_persisted_no_progress_state_after_restart(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fake_now = {"value": 120.0}
    monkeypatch.setattr(watch.time, "time", lambda: fake_now["value"])
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
    _write_json(tmp_path / "summary.json", {"total_cases": 1, "totals": {"ok": 1}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "running", "phase": "case_start", "heartbeat_epoch": fake_now["value"]},
    )
    _write_json(
        tmp_path / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "live",
            "pid": os.getpid(),
            "phase": "case_start",
            "heartbeat_epoch": fake_now["value"],
        },
    )
    state_path = tmp_path / "saved_watchdog_state.json"
    _write_json(
        state_path,
        {
            "state_version": 1,
            "last_progress_cases": 1,
            "last_progress_epoch": 100.0,
            "remediation_count": 0,
            "next_remediation_epoch": 0.0,
            "consecutive_unhealthy": 0,
        },
    )
    log_path = tmp_path / "watchdog.jsonl"

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        state_json=state_path,
        interval_seconds=0,
        max_heartbeat_age_seconds=60,
        max_no_progress_seconds=5,
        max_consecutive_unhealthy=1,
        max_checks=10,
    )

    assert exit_code == 2
    rows = _read_jsonl(log_path)
    assert len(rows) == 1
    assert rows[0]["progress_watch"]["seconds_since_progress"] == 20.0
    assert rows[0]["progress_watch"]["stalled"] is True
    state = json.loads(state_path.read_text())
    assert state["last_progress_cases"] == 1
    assert state["last_progress_epoch"] == 100.0
    assert state["consecutive_unhealthy"] == 1


def test_watchdog_exits_when_projected_wall_time_exceeds_budget(tmp_path: Path) -> None:
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
    log_path = tmp_path / "watchdog.jsonl"

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        interval_seconds=0,
        max_heartbeat_age_seconds=60,
        max_projected_duration_hours=5,
        max_consecutive_unhealthy=1,
        min_rate_cases=5,
        max_checks=10,
    )

    assert exit_code == 2
    rows = _read_jsonl(log_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["runtime_projection"]["status"] == "error"
    assert rows[0]["runtime_projection"]["active"] is True
    assert rows[0]["runtime_projection"]["processed_cases"] == 10
    assert rows[0]["runtime_projection"]["expected_cases"] == 100
    assert rows[0]["runtime_projection"]["projected_duration_hours"] > 5
    latest = json.loads((tmp_path / watch.LATEST_STATUS_NAME).read_text())
    assert latest["runtime_projection"]["status"] == "error"
    assert latest["runtime_projection"]["remaining_cases"] == 90
    checks = {check["name"]: check for check in rows[0]["checks"]}
    assert checks["projected_wall_time"]["status"] == "error"


def test_watchdog_exits_when_live_disk_reserve_is_below_floor(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        audit.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=200, used=199, free=1 * 1024 * 1024 * 1024),
    )
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
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
    log_path = tmp_path / "watchdog.jsonl"

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        interval_seconds=0,
        max_heartbeat_age_seconds=60,
        max_consecutive_unhealthy=1,
        min_free_gb=5,
        max_checks=10,
    )

    assert exit_code == 2
    rows = _read_jsonl(log_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    checks = {check["name"]: check for check in rows[0]["checks"]}
    assert checks["disk_reserve"]["status"] == "error"
    latest = json.loads((tmp_path / watch.LATEST_STATUS_NAME).read_text())
    assert latest["disk_reserve"]["min_free_gb"] == 5.0


def test_watchdog_records_degraded_rate_metrics_for_terminal_run(tmp_path: Path) -> None:
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
            "quality_failures": [],
            "recovery_events": [{"action": "loop_detected"}],
        },
    )
    _append_jsonl(
        tmp_path / "results.jsonl",
        {
            "case_id": "image:b:full",
            "final_status": "ok",
            "quality_failures": [],
            "recovery_events": [],
        },
    )
    _write_json(tmp_path / "summary.json", {"total_cases": 2, "totals": {"ok": 2}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )
    log_path = tmp_path / "watchdog.jsonl"

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        interval_seconds=0,
        max_heartbeat_age_seconds=60,
        max_loop_recovery_case_rate=0.0,
    )

    assert exit_code == 1
    rows = _read_jsonl(log_path)
    assert len(rows) == 1
    assert rows[0]["terminal"] is True
    assert rows[0]["degraded_rates"]["loop_recovery_case_rate"] == 0.5
    checks = {check["name"]: check for check in rows[0]["checks"]}
    assert checks["degraded_case_rates"]["status"] == "error"


def test_watchdog_compact_snapshot_preserves_loop_guard_evidence(tmp_path: Path) -> None:
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
                "preview_prompt_budget": (
                    {"max_prompt_tokens": 9000, "adapted_sections": 1}
                    if index == 2
                    else {}
                ),
                "qwen_caption_io": (
                    {
                        "event_counts": {
                            "stream_loop_detected": 2,
                            "loop_trim": 1,
                        }
                    }
                    if index == 1
                    else {"event_counts": {}}
                ),
            },
        )
    _write_json(tmp_path / "summary.json", {"total_cases": 4, "totals": {"ok": 4}})
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "completed", "phase": "finished", "heartbeat_epoch": time.time()},
    )
    log_path = tmp_path / "watchdog.jsonl"
    latest_path = tmp_path / watch.LATEST_STATUS_NAME

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        interval_seconds=0,
        max_heartbeat_age_seconds=60,
        set_and_forget=True,
    )

    assert exit_code == 1
    rows = _read_jsonl(log_path)
    compact_rates = rows[0]["degraded_rates"]
    assert "rate_headroom" not in compact_rates
    assert compact_rates["loop_guard_cases"] == 1
    assert compact_rates["loop_guard_case_rate"] == 0.25
    assert compact_rates["stream_loop_detected_cases"] == 1
    assert compact_rates["stream_loop_detected_events"] == 2
    assert compact_rates["stream_loop_detected_case_rate"] == 0.25
    assert compact_rates["loop_trim_cases"] == 1
    assert compact_rates["loop_trim_events"] == 1
    assert compact_rates["loop_trim_case_rate"] == 0.25
    assert compact_rates["prompt_budget_rows"] == 1
    assert compact_rates["prompt_budget_coverage_rate"] == 0.25
    assert compact_rates["prompt_budget_adapted_cases"] == 1
    assert compact_rates["prompt_budget_adapted_case_rate"] == 0.25
    assert compact_rates["max_prompt_tokens"] == 9000
    latest = json.loads(latest_path.read_text())
    assert "rate_headroom" in latest["degraded_rates"]


def test_watchdog_launchd_remediation_can_restore_running_health(monkeypatch, tmp_path: Path) -> None:
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
        {
            "status": "running",
            "phase": "attempt_running",
            "heartbeat_epoch": time.time() - 3600,
            "runner_capabilities": [watch.runner.RUNNER_CAPABILITY_GRACEFUL_RESTART],
        },
    )
    _write_json(
        tmp_path / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "stale",
            "pid": os.getpid(),
            "phase": "attempt_running",
            "heartbeat_epoch": time.time() - 3600,
        },
    )
    launchctl_commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        launchctl_commands.append([str(part) for part in command])
        now = time.time()
        _write_json(
            tmp_path / "heartbeat.json",
            {"status": "running", "phase": "attempt_running", "heartbeat_epoch": now},
        )
        _write_json(
            tmp_path / audit.RUNNER_LOCK_NAME,
            {
                "runner_id": "restored",
                "pid": os.getpid(),
                "phase": "attempt_running",
                "heartbeat_epoch": now,
            },
        )
        return subprocess.CompletedProcess(command, 0, stdout="started", stderr="")

    monkeypatch.setattr(watch.subprocess, "run", fake_run)
    log_path = tmp_path / "watchdog.jsonl"

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        interval_seconds=0,
        max_heartbeat_age_seconds=60,
        max_consecutive_unhealthy=1,
        max_checks=2,
        remediate_launchd_label="com.example.supervisor",
        remediate_launchd_domain="gui/501",
        max_remediations=1,
        remediation_cooldown_seconds=0,
    )

    assert exit_code == 0
    assert launchctl_commands == [["launchctl", "kickstart", "-k", "gui/501/com.example.supervisor"]]
    rows = _read_jsonl(log_path)
    assert [row["status"] for row in rows] == ["error", "ok"]
    assert rows[0]["remediation"]["status"] == "ok"
    assert rows[0]["remediation"]["remediation_index"] == 1
    assert rows[1].get("remediation") is None


def test_watchdog_set_and_forget_requests_graceful_restart_before_launchd(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    _write_json(
        tmp_path / "heartbeat.json",
        {
            "status": "running",
            "phase": "attempt_running",
            "heartbeat_epoch": time.time() - 3600,
            "runner_capabilities": [watch.runner.RUNNER_CAPABILITY_GRACEFUL_RESTART],
        },
    )
    _write_json(
        tmp_path / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "stale",
            "pid": os.getpid(),
            "phase": "attempt_running",
            "heartbeat_epoch": time.time() - 3600,
        },
    )
    launchctl_commands: list[list[str]] = []
    monkeypatch.setattr(
        watch.subprocess,
        "run",
        lambda command, **_kwargs: launchctl_commands.append([str(part) for part in command]),
    )
    log_path = tmp_path / "watchdog.jsonl"

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        interval_seconds=0,
        max_heartbeat_age_seconds=60,
        max_consecutive_unhealthy=1,
        max_checks=1,
        set_and_forget=True,
        remediate_launchd_label="com.example.supervisor",
        remediate_launchd_domain="gui/501",
        max_remediations=1,
        remediation_cooldown_seconds=0,
    )

    assert exit_code == 0
    assert launchctl_commands == []
    request_path = tmp_path / watch.runner.RUNNER_RESTART_REQUEST_NAME
    request = json.loads(request_path.read_text())
    assert request["reason"] == "watchdog_unhealthy_threshold"
    rows = _read_jsonl(log_path)
    assert rows[0]["status"] == "error"
    assert rows[0]["remediation"]["action"] == "graceful_restart_request"
    assert rows[0]["remediation"]["status"] == "ok"
    state = json.loads((tmp_path / watch.WATCHDOG_STATE_NAME).read_text())
    assert state["consecutive_unhealthy"] == 0
    assert state["graceful_restart_request_epoch"] == request["requested_epoch"]
    assert state["runner_supports_graceful_restart"] is True


def test_watchdog_set_and_forget_old_runner_escalates_without_graceful_wait(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "running", "phase": "attempt_running", "heartbeat_epoch": time.time() - 3600},
    )
    _write_json(
        tmp_path / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "old-runner",
            "pid": os.getpid(),
            "phase": "attempt_running",
            "heartbeat_epoch": time.time() - 3600,
        },
    )
    launchctl_commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        launchctl_commands.append([str(part) for part in command])
        return subprocess.CompletedProcess(command, 0, stdout="started", stderr="")

    monkeypatch.setattr(watch.subprocess, "run", fake_run)
    log_path = tmp_path / "watchdog.jsonl"

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        interval_seconds=0,
        max_heartbeat_age_seconds=60,
        max_consecutive_unhealthy=1,
        max_checks=1,
        set_and_forget=True,
        remediate_launchd_label="com.example.supervisor",
        remediate_launchd_domain="gui/501",
        max_remediations=1,
        remediation_cooldown_seconds=0,
    )

    assert exit_code == 0
    assert launchctl_commands == [["launchctl", "kickstart", "-k", "gui/501/com.example.supervisor"]]
    assert not (tmp_path / watch.runner.RUNNER_RESTART_REQUEST_NAME).exists()
    rows = _read_jsonl(log_path)
    assert rows[0]["remediation"]["action"] == "launchd_kickstart"
    state = json.loads((tmp_path / watch.WATCHDOG_STATE_NAME).read_text())
    assert state["runner_supports_graceful_restart"] is False


def test_watchdog_escalates_launchd_after_stale_graceful_restart_request(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "running", "phase": "attempt_running", "heartbeat_epoch": time.time() - 3600},
    )
    _write_json(
        tmp_path / audit.RUNNER_LOCK_NAME,
        {
            "runner_id": "stale",
            "pid": os.getpid(),
            "phase": "attempt_running",
            "heartbeat_epoch": time.time() - 3600,
        },
    )
    _write_json(
        tmp_path / watch.runner.RUNNER_RESTART_REQUEST_NAME,
        {"reason": "old_request", "requested_epoch": time.time() - 10},
    )
    launchctl_commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        launchctl_commands.append([str(part) for part in command])
        now = time.time()
        _write_json(
            tmp_path / "heartbeat.json",
            {"status": "running", "phase": "attempt_running", "heartbeat_epoch": now},
        )
        _write_json(
            tmp_path / audit.RUNNER_LOCK_NAME,
            {
                "runner_id": "restored",
                "pid": os.getpid(),
                "phase": "attempt_running",
                "heartbeat_epoch": now,
            },
        )
        return subprocess.CompletedProcess(command, 0, stdout="started", stderr="")

    monkeypatch.setattr(watch.subprocess, "run", fake_run)
    log_path = tmp_path / "watchdog.jsonl"

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        interval_seconds=0,
        max_heartbeat_age_seconds=60,
        max_consecutive_unhealthy=1,
        max_checks=2,
        set_and_forget=True,
        graceful_restart_timeout_seconds=1,
        remediate_launchd_label="com.example.supervisor",
        remediate_launchd_domain="gui/501",
        max_remediations=1,
        remediation_cooldown_seconds=0,
    )

    assert exit_code == 0
    assert launchctl_commands == [["launchctl", "kickstart", "-k", "gui/501/com.example.supervisor"]]
    rows = _read_jsonl(log_path)
    assert [row["status"] for row in rows] == ["error", "ok"]
    assert rows[0]["remediation"]["action"] == "launchd_kickstart"


def test_watchdog_launchd_remediation_failure_exits_unhealthy(monkeypatch, tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "running", "phase": "attempt_running", "heartbeat_epoch": time.time() - 3600},
    )
    launchctl_commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        launchctl_commands.append([str(part) for part in command])
        return subprocess.CompletedProcess(command, 78, stdout="", stderr="not loaded")

    monkeypatch.setattr(watch.subprocess, "run", fake_run)
    log_path = tmp_path / "watchdog.jsonl"

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        interval_seconds=0,
        max_heartbeat_age_seconds=60,
        max_consecutive_unhealthy=1,
        max_checks=10,
        remediate_launchd_label="com.example.supervisor",
        remediate_launchd_domain="gui/501",
        max_remediations=1,
        remediation_cooldown_seconds=0,
    )

    assert exit_code == 2
    assert launchctl_commands == [["launchctl", "kickstart", "-k", "gui/501/com.example.supervisor"]]
    rows = _read_jsonl(log_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["remediation"]["status"] == "error"
    assert rows[0]["remediation"]["returncode"] == 78


def test_watchdog_uses_persisted_remediation_budget_after_restart(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "running", "phase": "attempt_running", "heartbeat_epoch": time.time() - 3600},
    )
    launchctl_commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        launchctl_commands.append([str(part) for part in command])
        return subprocess.CompletedProcess(command, 0, stdout="started", stderr="")

    monkeypatch.setattr(watch.subprocess, "run", fake_run)
    state_path = tmp_path / "saved_watchdog_state.json"
    _write_json(
        state_path,
        {
            "state_version": 1,
            "last_progress_cases": 0,
            "last_progress_epoch": time.time(),
            "remediation_count": 1,
            "next_remediation_epoch": 0.0,
            "consecutive_unhealthy": 0,
        },
    )
    log_path = tmp_path / "watchdog.jsonl"

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        state_json=state_path,
        interval_seconds=0,
        max_heartbeat_age_seconds=60,
        max_consecutive_unhealthy=1,
        max_checks=10,
        remediate_launchd_label="com.example.supervisor",
        remediate_launchd_domain="gui/501",
        max_remediations=1,
        remediation_cooldown_seconds=0,
    )

    assert exit_code == 2
    assert launchctl_commands == []
    rows = _read_jsonl(log_path)
    assert len(rows) == 1
    assert rows[0].get("remediation") is None
    state = json.loads(state_path.read_text())
    assert state["remediation_count"] == 1
    assert state["consecutive_unhealthy"] == 1


def test_watchdog_recovers_remediation_budget_from_log_when_state_file_is_corrupt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"cases": [{"case_id": "image:a:full", "image_name": "a.jpg"}]},
    )
    _write_json(
        tmp_path / "heartbeat.json",
        {"status": "running", "phase": "attempt_running", "heartbeat_epoch": time.time() - 3600},
    )
    launchctl_commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        launchctl_commands.append([str(part) for part in command])
        return subprocess.CompletedProcess(command, 0, stdout="started", stderr="")

    monkeypatch.setattr(watch.subprocess, "run", fake_run)
    state_path = tmp_path / "saved_watchdog_state.json"
    state_path.write_text("{not-json")
    log_path = tmp_path / "watchdog.jsonl"
    recovered_epoch = time.time() - 120
    _append_jsonl(
        log_path,
        {
            "event": "qwen_caption_soak_watchdog",
            "watchdog_state": {
                "state_version": 1,
                "last_progress_cases": 0,
                "last_progress_epoch": recovered_epoch,
                "last_progress_signal": "",
                "last_progress_source": "case",
                "remediation_count": 1,
                "next_remediation_epoch": 0.0,
                "consecutive_unhealthy": 0,
            },
        },
    )

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        state_json=state_path,
        interval_seconds=0,
        max_heartbeat_age_seconds=60,
        max_consecutive_unhealthy=1,
        max_checks=10,
        remediate_launchd_label="com.example.supervisor",
        remediate_launchd_domain="gui/501",
        max_remediations=1,
        remediation_cooldown_seconds=0,
    )

    assert exit_code == 2
    assert launchctl_commands == []
    rows = _read_jsonl(log_path)
    assert rows[-1].get("remediation") is None
    state = json.loads(state_path.read_text())
    assert state["remediation_count"] == 1
    assert state["loaded_state_source"] == "watchdog_log"
    assert state["state_file_exists"] is True
    assert state["state_load_error"]
    assert state["state_recovered_from_log"] is True
    assert state["consecutive_unhealthy"] == 1


def test_watchdog_launchd_remediation_rebootstraps_when_kickstart_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
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
            "runner_id": "dead",
            "pid": -1,
            "phase": "attempt_running",
            "heartbeat_epoch": time.time(),
        },
    )
    supervisor_plist = tmp_path / "com.example.supervisor.plist"
    supervisor_plist.write_text("plist")
    launchctl_commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        command = [str(part) for part in command]
        launchctl_commands.append(command)
        if command[1] == "bootstrap":
            now = time.time()
            _write_json(
                tmp_path / "heartbeat.json",
                {"status": "running", "phase": "attempt_running", "heartbeat_epoch": now},
            )
            _write_json(
                tmp_path / audit.RUNNER_LOCK_NAME,
                {
                    "runner_id": "restored",
                    "pid": os.getpid(),
                    "phase": "attempt_running",
                    "heartbeat_epoch": now,
                },
            )
            return subprocess.CompletedProcess(command, 0, stdout="bootstrapped", stderr="")
        if command[1] == "bootout":
            return subprocess.CompletedProcess(command, 3, stdout="", stderr="not loaded")
        return subprocess.CompletedProcess(command, 78, stdout="", stderr="kickstart failed")

    monkeypatch.setattr(watch.subprocess, "run", fake_run)
    log_path = tmp_path / "watchdog.jsonl"

    exit_code = watch.watch_soak(
        tmp_path,
        log_jsonl=log_path,
        interval_seconds=0,
        max_heartbeat_age_seconds=60,
        max_consecutive_unhealthy=1,
        max_checks=2,
        remediate_launchd_label="com.example.supervisor",
        remediate_launchd_domain="gui/501",
        remediate_launchd_plist=supervisor_plist,
        max_remediations=1,
        remediation_cooldown_seconds=0,
    )

    assert exit_code == 0
    assert launchctl_commands == [
        ["launchctl", "kickstart", "-k", "gui/501/com.example.supervisor"],
        ["launchctl", "bootout", "gui/501/com.example.supervisor"],
        ["launchctl", "bootstrap", "gui/501", str(supervisor_plist.resolve(strict=False))],
    ]
    rows = _read_jsonl(log_path)
    assert [row["status"] for row in rows] == ["error", "ok"]
    remediation = rows[0]["remediation"]
    assert remediation["action"] == "launchd_rebootstrap"
    assert remediation["status"] == "ok"
    assert remediation["kickstart"]["returncode"] == 78
    assert remediation["bootout"]["returncode"] == 3
    assert remediation["bootstrap"]["returncode"] == 0


def test_watchdog_cli_runs_as_script(tmp_path: Path) -> None:
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
    log_path = tmp_path / "cli_watchdog.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "watch_qwen_caption_soak.py"),
            str(tmp_path),
            "--log-jsonl",
            str(log_path),
            "--interval",
            "0",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0
    rows = _read_jsonl(log_path)
    assert len(rows) == 1
    assert rows[0]["terminal"] is True
