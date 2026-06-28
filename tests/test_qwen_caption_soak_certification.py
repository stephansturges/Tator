from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from tools import certify_qwen_caption_soak as certify


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _write_completed_soak(
    output_dir: Path,
    *,
    case_count: int,
    elapsed_seconds: float,
    recovery_every: int = 0,
    run_settings_fingerprint: str | None = None,
    include_prompt_budget: bool = True,
    max_prompt_tokens: int = 1200,
    runtime_max_prompt_tokens: int = 0,
    adapted_every: int = 0,
    loop_guard_every: int = 0,
    deterministic_every: int = 0,
    signal_exit_before_success: bool = False,
    sample_size: int = 0,
    sample_selection: dict[str, object] | None = None,
    runner_capabilities: list[str] | None = None,
) -> None:
    cases = [{"case_id": f"image:{index}:full", "image_name": f"{index}.jpg"} for index in range(case_count)]
    if runner_capabilities is None:
        runner_capabilities = list(certify.runner.RUNNER_CAPABILITIES)
    manifest = {"cases": cases, "runner_capabilities": runner_capabilities}
    if sample_size:
        manifest["sample_size"] = sample_size
    if sample_selection is not None:
        manifest["sample_selection"] = sample_selection
    if run_settings_fingerprint is not None:
        manifest["run_settings"] = {"fingerprint": run_settings_fingerprint}
    _write_json(output_dir / "manifest.json", manifest)
    for index, case in enumerate(cases):
        recovery_events = []
        if recovery_every and index % recovery_every == 0:
            recovery_events = [{"action": "text_recovery_succeeded"}]
        parent_deterministic_recovery = bool(deterministic_every and index % deterministic_every == 0)
        if parent_deterministic_recovery:
            recovery_events.append(
                {
                    "action": "deterministic_recovery_succeeded",
                    "attempt": "parent_deterministic_recovery",
                    "call_kind": "deterministic",
                }
            )
        prompt_budget = (
            {
                "max_prompt_tokens": max_prompt_tokens,
                "adapted_sections": 1 if adapted_every and index % adapted_every == 0 else 0,
            }
            if include_prompt_budget
            else None
        )
        row = {
            "case_id": case["case_id"],
            "status": "ok",
            "final_status": "ok",
            "quality_failures": [],
            "recovery_events": recovery_events,
            "elapsed_seconds": elapsed_seconds,
        }
        if parent_deterministic_recovery:
            row["parent_deterministic_recovery"] = True
        if prompt_budget is not None:
            event_counts = {"prompt_budget": 1}
            if loop_guard_every and index % loop_guard_every == 0:
                event_counts.update({"stream_loop_detected": 1, "loop_trim": 1})
            row["preview_prompt_budget"] = prompt_budget
            row["qwen_caption_io"] = {
                "source": "qwen_caption_io_per_run",
                "source_run_ids": [f"run-{index}"],
                "event_counts": event_counts,
                "prompt_budget_events": 1,
                "prompt_budget_adapted_events": (
                    1 if adapted_every and index % adapted_every == 0 else 0
                ),
                "max_prompt_tokens": runtime_max_prompt_tokens or max_prompt_tokens,
            }
        if signal_exit_before_success and index == 0:
            _append_jsonl(
                output_dir / "results.jsonl",
                {
                    "case_id": case["case_id"],
                    "status": "missing_result",
                    "final_status": "failed_attempt",
                    "attempt_failure_kind": "signal_exit",
                    "exit_code": -6,
                    "return_signal": 6,
                    "return_signal_name": "SIGABRT",
                    "quality_failures": [],
                    "elapsed_seconds": elapsed_seconds,
                    "preview_prompt_budget": prompt_budget,
                },
            )
        _append_jsonl(
            output_dir / "results.jsonl",
            row,
        )
        _append_jsonl(
            output_dir / "captions.jsonl",
            {
                "case_id": case["case_id"],
                "caption": f"Caption for {case['case_id']}.",
            },
        )
    _write_json(output_dir / "summary.json", {"total_cases": case_count, "totals": {"ok": case_count}})
    _write_json(
        output_dir / "heartbeat.json",
        {
            "status": "completed",
            "phase": "finished",
            "heartbeat_epoch": time.time(),
            "runner_capabilities": runner_capabilities,
        },
    )


def _append_resume_skip_rows(output_dir: Path, *, case_count: int) -> None:
    for index in range(case_count):
        _append_jsonl(
            output_dir / "results.jsonl",
            {
                "case_id": f"image:{index}:full",
                "status": "skipped_completed",
                "final_status": "skipped_completed",
                "resumed_skip": True,
                "quality_failures": [],
                "elapsed_seconds": 0.01,
            },
        )
    _write_json(
        output_dir / "summary.json",
        {"total_cases": case_count, "totals": {"skipped_completed": case_count}},
    )


def _write_skipped_only_soak(output_dir: Path, *, case_count: int) -> None:
    cases = [{"case_id": f"image:{index}:full", "image_name": f"{index}.jpg"} for index in range(case_count)]
    _write_json(
        output_dir / "manifest.json",
        {
            "cases": cases,
            "runner_capabilities": list(certify.runner.RUNNER_CAPABILITIES),
        },
    )
    for case in cases:
        _append_jsonl(
            output_dir / "results.jsonl",
            {
                "case_id": case["case_id"],
                "status": "skipped_existing_caption",
                "final_status": "skipped_existing_caption",
                "quality_failures": [],
                "elapsed_seconds": 0.01,
            },
        )
    _write_json(
        output_dir / "summary.json",
        {"total_cases": case_count, "totals": {"skipped_existing_caption": case_count}},
    )
    _write_json(
        output_dir / "heartbeat.json",
        {
            "status": "completed",
            "phase": "finished",
            "heartbeat_epoch": time.time(),
            "runner_capabilities": list(certify.runner.RUNNER_CAPABILITIES),
        },
    )


def _checks_by_name(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(check["name"]): check for check in report["checks"]}  # type: ignore[index]


def test_certification_passes_pilot_with_ok_audit_and_duration_projection(tmp_path: Path) -> None:
    _write_completed_soak(tmp_path, case_count=4, elapsed_seconds=10.0)

    report = certify.certify_soak(
        tmp_path,
        target_cases=10_000,
        max_duration_hours=40,
        min_pilot_cases=4,
        duration_safety_factor=1.0,
        max_heartbeat_age_seconds=60,
    )

    assert report["status"] == "ok"
    assert report["pilot_cases"] == 4
    assert report["timing"]["timed_cases"] == 4  # type: ignore[index]
    assert report["timing"]["cases_per_hour"] == 360.0  # type: ignore[index]
    assert report["prompt_budget"]["rows_with_prompt_budget"] == 4  # type: ignore[index]
    assert report["projection"]["safety_projected_hours"] == 10.0 * 10_000 / 3600.0  # type: ignore[index]
    assert report["projection"]["p95_all_cases_hours"] == 10.0 * 10_000 / 3600.0  # type: ignore[index]
    assert report["projection"]["max_p95_duration_hours"] == 40  # type: ignore[index]
    checks = _checks_by_name(report)
    assert checks["artifact_audit"]["status"] == "ok"
    assert checks["generated_case_evidence"]["status"] == "ok"
    assert report["generated_pilot_cases"] == 4
    assert checks["pilot_sample_size"]["status"] == "ok"
    assert checks["projected_duration"]["status"] == "ok"
    assert checks["p95_projected_duration"]["status"] == "ok"
    assert checks["prompt_budget_data"]["status"] == "ok"
    assert checks["qwen_caption_io_source"]["status"] == "ok"
    assert checks["runner_capabilities"]["status"] == "ok"
    assert report["runner_capabilities"]["missing_capabilities"] == []  # type: ignore[index]
    assert (
        certify.runner.RUNNER_CAPABILITY_CAPTION_IO_EVENT_SUMMARY
        in report["runner_capabilities"]["runner_capabilities"]  # type: ignore[index]
    )
    assert (
        certify.runner.RUNNER_CAPABILITY_WORKER_PROGRESS_HEARTBEAT
        in report["runner_capabilities"]["runner_capabilities"]  # type: ignore[index]
    )


def test_certification_fails_when_pilot_too_small_or_projection_too_slow(tmp_path: Path) -> None:
    _write_completed_soak(tmp_path, case_count=2, elapsed_seconds=1000.0)

    report = certify.certify_soak(
        tmp_path,
        target_cases=10_000,
        max_duration_hours=10,
        min_pilot_cases=3,
        duration_safety_factor=1.0,
        max_heartbeat_age_seconds=60,
    )

    assert report["status"] == "error"
    checks = _checks_by_name(report)
    assert checks["generated_case_evidence"]["status"] == "error"
    assert checks["pilot_sample_size"]["status"] == "error"
    assert checks["projected_duration"]["status"] == "error"
    assert checks["p95_projected_duration"]["status"] == "error"


def test_certification_accepts_resume_skips_only_when_prior_generated_rows_exist(tmp_path: Path) -> None:
    _write_completed_soak(tmp_path, case_count=4, elapsed_seconds=10.0)
    _append_resume_skip_rows(tmp_path, case_count=4)

    report = certify.certify_soak(
        tmp_path,
        target_cases=1_000,
        max_duration_hours=40,
        min_pilot_cases=4,
        duration_safety_factor=1.0,
        max_heartbeat_age_seconds=60,
        set_and_forget=True,
        deterministic_recovery_confidence=0,
    )

    assert report["status"] == "ok"
    assert report["pilot_cases"] == 4
    assert report["generated_pilot_cases"] == 4
    assert report["generated_case_evidence"]["latest_skipped_cases"] == 4  # type: ignore[index]
    assert report["generated_case_evidence"]["latest_skipped_cases_with_prior_generated_success"] == 4  # type: ignore[index]
    checks = _checks_by_name(report)
    assert checks["generated_case_evidence"]["status"] == "ok"
    assert checks["qwen_caption_io_source"]["required_rows"] == 4
    assert checks["qwen_caption_io_source"]["valid_runtime_prompt_budget_rows"] == 4


def test_certification_rejects_skipped_only_pilot_without_generated_evidence(tmp_path: Path) -> None:
    _write_skipped_only_soak(tmp_path, case_count=4)

    report = certify.certify_soak(
        tmp_path,
        target_cases=10_000,
        max_duration_hours=40,
        min_pilot_cases=4,
        duration_safety_factor=1.0,
        max_heartbeat_age_seconds=60,
        set_and_forget=True,
        deterministic_recovery_confidence=0,
    )

    assert report["status"] == "error"
    assert report["pilot_cases"] == 4
    assert report["generated_pilot_cases"] == 0
    assert report["generated_case_evidence"]["latest_cases_missing_generated_success"] == 4  # type: ignore[index]
    checks = _checks_by_name(report)
    assert checks["generated_case_evidence"]["status"] == "error"
    assert checks["pilot_sample_size"]["status"] == "error"
    assert checks["qwen_caption_io_source"]["required_rows"] == 0


def test_certification_fails_on_degraded_recovery_rates(tmp_path: Path) -> None:
    _write_completed_soak(tmp_path, case_count=4, elapsed_seconds=5.0, recovery_every=2)

    report = certify.certify_soak(
        tmp_path,
        target_cases=10_000,
        max_duration_hours=40,
        min_pilot_cases=4,
        duration_safety_factor=1.0,
        max_heartbeat_age_seconds=60,
        max_recovery_event_case_rate=0.0,
        min_rate_cases=4,
    )

    assert report["status"] == "error"
    checks = _checks_by_name(report)
    assert checks["artifact_audit"]["status"] == "error"
    assert report["audit"]["degraded_rates"]["recovery_event_case_rate"] == 0.5  # type: ignore[index]


def test_certification_set_and_forget_allows_rare_recovered_loop(tmp_path: Path) -> None:
    _write_completed_soak(tmp_path, case_count=50, elapsed_seconds=1.0)
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    rows[0]["recovery_events"] = [
        {"action": "loop_detected", "stage": "Compose full-image caption"},
        {"action": "safe_retry_succeeded", "stage": "Compose full-image caption"},
    ]
    with (tmp_path / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    report = certify.certify_soak(
        tmp_path,
        target_cases=1_000,
        max_duration_hours=40,
        min_pilot_cases=50,
        duration_safety_factor=1.0,
        max_heartbeat_age_seconds=60,
        set_and_forget=True,
        deterministic_recovery_confidence=0,
    )

    assert report["status"] == "ok"
    assert report["audit"]["degraded_rates"]["loop_recovery_case_rate"] == 0.02  # type: ignore[index]
    assert report["audit"]["degraded_rates"]["thresholds"]["max_loop_recovery_case_rate"] == 0.05  # type: ignore[index]


def test_certification_set_and_forget_allows_rare_loop_guard(tmp_path: Path) -> None:
    _write_completed_soak(
        tmp_path,
        case_count=50,
        elapsed_seconds=1.0,
        loop_guard_every=50,
    )

    report = certify.certify_soak(
        tmp_path,
        target_cases=1_000,
        max_duration_hours=40,
        min_pilot_cases=50,
        duration_safety_factor=1.0,
        max_heartbeat_age_seconds=60,
        set_and_forget=True,
        deterministic_recovery_confidence=0,
    )

    assert report["status"] == "ok"
    assert report["audit"]["degraded_rates"]["loop_guard_case_rate"] == 0.02  # type: ignore[index]
    assert report["audit"]["degraded_rates"]["thresholds"]["max_loop_guard_case_rate"] == 0.05  # type: ignore[index]


def test_certification_set_and_forget_rejects_small_clean_10k_pilot_confidence(
    tmp_path: Path,
) -> None:
    _write_completed_soak(tmp_path, case_count=100, elapsed_seconds=1.0)

    report = certify.certify_soak(
        tmp_path,
        target_cases=10_000,
        max_duration_hours=40,
        min_pilot_cases=100,
        duration_safety_factor=1.0,
        max_heartbeat_age_seconds=60,
        set_and_forget=True,
    )

    assert report["status"] == "error"
    checks = _checks_by_name(report)
    confidence_check = checks["deterministic_recovery_confidence"]
    reliability = report["deterministic_recovery_reliability"]  # type: ignore[index]
    assert checks["artifact_audit"]["status"] == "ok"
    assert confidence_check["status"] == "error"
    assert reliability["deterministic_recovery_cases"] == 0
    assert reliability["upper_bound_rate"] > reliability["limit"]
    assert reliability["required_zero_recovery_cases"] > 100


def test_certification_set_and_forget_accepts_confidence_sized_clean_10k_pilot(
    tmp_path: Path,
) -> None:
    _write_completed_soak(tmp_path, case_count=300, elapsed_seconds=1.0)

    report = certify.certify_soak(
        tmp_path,
        target_cases=10_000,
        max_duration_hours=40,
        min_pilot_cases=300,
        duration_safety_factor=1.0,
        max_heartbeat_age_seconds=60,
        set_and_forget=True,
    )

    assert report["status"] == "ok"
    checks = _checks_by_name(report)
    reliability = report["deterministic_recovery_reliability"]  # type: ignore[index]
    assert checks["deterministic_recovery_confidence"]["status"] == "ok"
    assert reliability["generated_cases"] == 300
    assert reliability["deterministic_recovery_cases"] == 0
    assert reliability["upper_bound_rate"] <= reliability["limit"]


def test_certification_fails_on_excess_loop_guard_rate(tmp_path: Path) -> None:
    _write_completed_soak(
        tmp_path,
        case_count=20,
        elapsed_seconds=1.0,
        loop_guard_every=4,
    )

    report = certify.certify_soak(
        tmp_path,
        target_cases=10_000,
        max_duration_hours=40,
        min_pilot_cases=20,
        duration_safety_factor=1.0,
        max_heartbeat_age_seconds=60,
        set_and_forget=True,
    )

    assert report["status"] == "error"
    checks = _checks_by_name(report)
    assert checks["artifact_audit"]["status"] == "error"
    assert report["audit"]["degraded_rates"]["loop_guard_case_rate"] == 0.25  # type: ignore[index]
    assert report["audit"]["degraded_rates"]["active_violations"][0]["rate"] == "loop_guard_case_rate"  # type: ignore[index]


def test_certification_fails_when_p95_projection_exceeds_duration_budget(tmp_path: Path) -> None:
    _write_completed_soak(tmp_path, case_count=20, elapsed_seconds=1.0)
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    rows[-1]["elapsed_seconds"] = 20.0
    rows[-2]["elapsed_seconds"] = 20.0
    with (tmp_path / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    report = certify.certify_soak(
        tmp_path,
        target_cases=10_000,
        max_duration_hours=40,
        min_pilot_cases=20,
        duration_safety_factor=1.0,
        max_heartbeat_age_seconds=60,
    )
    checks = _checks_by_name(report)

    assert report["status"] == "error"
    assert checks["projected_duration"]["status"] == "ok"
    assert checks["p95_projected_duration"]["status"] == "error"
    assert checks["runtime_tail"]["status"] == "ok"
    assert checks["runtime_tail"]["advisory"] is True
    assert report["projection"]["safety_projected_hours"] < 40  # type: ignore[index]
    assert report["projection"]["p95_all_cases_hours"] > 40  # type: ignore[index]


def test_certification_can_disable_p95_projection_gate_for_manual_diagnostics(tmp_path: Path) -> None:
    _write_completed_soak(tmp_path, case_count=20, elapsed_seconds=1.0)
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    rows[-1]["elapsed_seconds"] = 20.0
    rows[-2]["elapsed_seconds"] = 20.0
    with (tmp_path / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    report = certify.certify_soak(
        tmp_path,
        target_cases=10_000,
        max_duration_hours=40,
        max_p95_duration_hours=-1,
        min_pilot_cases=20,
        duration_safety_factor=1.0,
        max_heartbeat_age_seconds=60,
    )

    checks = _checks_by_name(report)
    assert report["status"] == "ok"
    assert checks["p95_projected_duration"]["status"] == "ok"
    assert "disabled" in checks["p95_projected_duration"]["detail"]


def test_certification_fails_on_recovered_signal_exit_by_default(tmp_path: Path) -> None:
    _write_completed_soak(
        tmp_path,
        case_count=4,
        elapsed_seconds=5.0,
        signal_exit_before_success=True,
    )

    report = certify.certify_soak(
        tmp_path,
        target_cases=10_000,
        max_duration_hours=40,
        min_pilot_cases=4,
        duration_safety_factor=1.0,
        max_heartbeat_age_seconds=60,
    )

    assert report["status"] == "error"
    assert report["audit"]["degraded_rates"]["signal_exit_attempt_rows"] == 1  # type: ignore[index]
    assert report["audit"]["degraded_rates"]["signal_exit_names"] == {"SIGABRT": 1}  # type: ignore[index]
    checks = _checks_by_name(report)
    assert checks["artifact_audit"]["status"] == "error"

    relaxed = certify.certify_soak(
        tmp_path,
        target_cases=10_000,
        max_duration_hours=40,
        min_pilot_cases=4,
        duration_safety_factor=1.0,
        max_heartbeat_age_seconds=60,
        max_failed_attempt_row_rate=1.0,
        max_signal_exit_attempt_row_rate=1.0,
    )
    assert relaxed["status"] == "ok"


def test_certification_set_and_forget_allows_bounded_recovered_signal_exit(tmp_path: Path) -> None:
    _write_completed_soak(
        tmp_path,
        case_count=300,
        elapsed_seconds=1.0,
        signal_exit_before_success=True,
    )

    report = certify.certify_soak(
        tmp_path,
        target_cases=10_000,
        max_duration_hours=40,
        min_pilot_cases=300,
        duration_safety_factor=1.0,
        max_heartbeat_age_seconds=60,
        set_and_forget=True,
    )
    explicit_zero = certify.certify_soak(
        tmp_path,
        target_cases=10_000,
        max_duration_hours=40,
        min_pilot_cases=300,
        duration_safety_factor=1.0,
        max_heartbeat_age_seconds=60,
        set_and_forget=True,
        max_signal_exit_attempt_row_rate=0.0,
    )

    assert report["status"] == "ok"
    assert explicit_zero["status"] == "error"
    rates = report["audit"]["degraded_rates"]  # type: ignore[index]
    assert rates["signal_exit_attempt_rows"] == 1
    assert rates["thresholds"]["max_signal_exit_attempt_row_rate"] == 0.05
    assert rates["signal_exit_attempt_row_rate"] < 0.05


def test_certification_requires_matching_run_settings_fingerprint(tmp_path: Path) -> None:
    _write_completed_soak(
        tmp_path,
        case_count=4,
        elapsed_seconds=5.0,
        run_settings_fingerprint="pilot-fingerprint",
    )

    matching = certify.certify_soak(
        tmp_path,
        target_cases=10,
        max_duration_hours=1,
        min_pilot_cases=1,
        max_heartbeat_age_seconds=60,
        expected_run_settings_fingerprint="pilot-fingerprint",
    )
    assert matching["status"] == "ok"
    assert matching["run_settings"]["pilot_fingerprint"] == "pilot-fingerprint"  # type: ignore[index]
    assert matching["run_settings"]["expected_fingerprint"] == "pilot-fingerprint"  # type: ignore[index]
    assert _checks_by_name(matching)["run_settings_fingerprint"]["status"] == "ok"

    mismatched = certify.certify_soak(
        tmp_path,
        target_cases=10,
        max_duration_hours=1,
        min_pilot_cases=1,
        max_heartbeat_age_seconds=60,
        expected_run_settings_fingerprint="large-run-fingerprint",
    )
    assert mismatched["status"] == "error"
    mismatch_check = _checks_by_name(mismatched)["run_settings_fingerprint"]
    assert mismatch_check["status"] == "error"
    assert mismatch_check["pilot_fingerprint"] == "pilot-fingerprint"
    assert mismatch_check["expected_fingerprint"] == "large-run-fingerprint"


def test_certification_fails_without_prompt_budget_telemetry_by_default(tmp_path: Path) -> None:
    _write_completed_soak(tmp_path, case_count=4, elapsed_seconds=5.0, include_prompt_budget=False)

    report = certify.certify_soak(
        tmp_path,
        target_cases=10,
        max_duration_hours=1,
        min_pilot_cases=1,
        max_heartbeat_age_seconds=60,
    )

    assert report["status"] == "error"
    checks = _checks_by_name(report)
    assert checks["prompt_budget_data"]["status"] == "error"
    assert report["prompt_budget"]["rows_with_prompt_budget"] == 0  # type: ignore[index]

    legacy_allowed = certify.certify_soak(
        tmp_path,
        target_cases=10,
        max_duration_hours=1,
        min_pilot_cases=1,
        max_heartbeat_age_seconds=60,
        require_prompt_budget_data=False,
    )
    assert legacy_allowed["status"] == "ok"
    assert _checks_by_name(legacy_allowed)["prompt_budget_data"]["status"] == "ok"


def test_certification_rejects_unbound_latest_qwen_caption_io_source(tmp_path: Path) -> None:
    _write_completed_soak(tmp_path, case_count=4, elapsed_seconds=5.0)
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    for row in rows:
        row["qwen_caption_io"]["source"] = "qwen_caption_io_latest"
        row["qwen_caption_io"]["source_run_ids"] = []
    with (tmp_path / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    report = certify.certify_soak(
        tmp_path,
        target_cases=10,
        max_duration_hours=1,
        min_pilot_cases=1,
        max_heartbeat_age_seconds=60,
        set_and_forget=True,
    )

    assert report["status"] == "error"
    checks = _checks_by_name(report)
    assert checks["qwen_caption_io_source"]["status"] == "error"
    assert checks["qwen_caption_io_source"]["source_counts"] == {"qwen_caption_io_latest": 4}
    assert report["qwen_caption_io_sources"]["invalid_runtime_rows_count"] == 4  # type: ignore[index]


def test_certification_set_and_forget_requires_runtime_qwen_caption_io(
    tmp_path: Path,
) -> None:
    _write_completed_soak(tmp_path, case_count=4, elapsed_seconds=5.0)
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    rows[0].pop("qwen_caption_io")
    with (tmp_path / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    report = certify.certify_soak(
        tmp_path,
        target_cases=10,
        max_duration_hours=1,
        min_pilot_cases=1,
        max_heartbeat_age_seconds=60,
        set_and_forget=True,
    )

    assert report["status"] == "error"
    checks = _checks_by_name(report)
    assert checks["prompt_budget_data"]["status"] == "ok"
    assert checks["qwen_caption_io_source"]["status"] == "error"
    assert checks["qwen_caption_io_source"]["valid_runtime_prompt_budget_rows"] == 3
    assert checks["qwen_caption_io_source"]["required_rows"] == 4
    assert checks["qwen_caption_io_source"]["missing_runtime_rows_count"] == 1


def test_certification_fails_without_runner_capabilities_by_default(tmp_path: Path) -> None:
    _write_completed_soak(
        tmp_path,
        case_count=4,
        elapsed_seconds=5.0,
        runner_capabilities=[],
    )

    report = certify.certify_soak(
        tmp_path,
        target_cases=10,
        max_duration_hours=1,
        min_pilot_cases=1,
        max_heartbeat_age_seconds=60,
    )

    assert report["status"] == "error"
    checks = _checks_by_name(report)
    assert checks["runner_capabilities"]["status"] == "error"
    assert (
        certify.runner.RUNNER_CAPABILITY_GRACEFUL_RESTART
        in checks["runner_capabilities"]["missing_capabilities"]
    )
    assert (
        certify.runner.RUNNER_CAPABILITY_CAPTION_IO_EVENT_SUMMARY
        in checks["runner_capabilities"]["missing_capabilities"]
    )
    assert (
        certify.runner.RUNNER_CAPABILITY_WORKER_PROGRESS_HEARTBEAT
        in checks["runner_capabilities"]["missing_capabilities"]
    )

    legacy_allowed = certify.certify_soak(
        tmp_path,
        target_cases=10,
        max_duration_hours=1,
        min_pilot_cases=1,
        max_heartbeat_age_seconds=60,
        require_runner_capabilities=False,
    )
    assert legacy_allowed["status"] == "ok"
    assert _checks_by_name(legacy_allowed)["runner_capabilities"]["status"] == "ok"


def test_certification_rejects_stale_runner_without_current_progress_capabilities(tmp_path: Path) -> None:
    stale_capabilities = [
        certify.runner.RUNNER_CAPABILITY_GRACEFUL_RESTART,
        certify.runner.RUNNER_CAPABILITY_PARENT_DETERMINISTIC_RECOVERY,
    ]
    _write_completed_soak(
        tmp_path,
        case_count=4,
        elapsed_seconds=5.0,
        runner_capabilities=stale_capabilities,
    )

    report = certify.certify_soak(
        tmp_path,
        target_cases=10,
        max_duration_hours=1,
        min_pilot_cases=1,
        max_heartbeat_age_seconds=60,
    )

    assert report["status"] == "error"
    checks = _checks_by_name(report)
    assert checks["runner_capabilities"]["status"] == "error"
    assert checks["runner_capabilities"]["runner_capabilities"] == stale_capabilities
    assert checks["runner_capabilities"]["missing_capabilities"] == [
        certify.runner.RUNNER_CAPABILITY_CAPTION_IO_EVENT_SUMMARY,
        certify.runner.RUNNER_CAPABILITY_WORKER_PROGRESS_HEARTBEAT,
        certify.runner.RUNNER_CAPABILITY_ADAPTIVE_RETRY_PROFILE,
        certify.runner.RUNNER_CAPABILITY_INSTRUCTION_QA,
    ]


def test_certification_fails_sampled_pilot_without_sample_selection_metadata(tmp_path: Path) -> None:
    _write_completed_soak(
        tmp_path,
        case_count=4,
        elapsed_seconds=5.0,
        sample_size=4,
    )

    report = certify.certify_soak(
        tmp_path,
        target_cases=10,
        max_duration_hours=1,
        min_pilot_cases=1,
        max_heartbeat_age_seconds=60,
    )

    assert report["status"] == "error"
    checks = _checks_by_name(report)
    assert checks["sample_selection"]["status"] == "error"
    assert "missing stress-sample selection metadata" in checks["sample_selection"]["detail"]
    assert report["sample_selection"]["sampled"] is True  # type: ignore[index]
    assert report["sample_selection"]["has_sample_selection"] is False  # type: ignore[index]


def test_certification_accepts_stress_biased_sample_selection_metadata(tmp_path: Path) -> None:
    _write_completed_soak(
        tmp_path,
        case_count=4,
        elapsed_seconds=5.0,
        sample_size=4,
        sample_selection={
            "strategy": "stress_plus_random",
            "source_cases": 10000,
            "selected_cases": 4,
            "requested_sample_size": 4,
            "sample_seed": 13,
            "stress_case_keys": ["image:0:full", "image:1:full"],
            "random_fill_case_keys": ["image:2:full", "image:3:full"],
        },
    )

    report = certify.certify_soak(
        tmp_path,
        target_cases=10,
        max_duration_hours=1,
        min_pilot_cases=1,
        max_heartbeat_age_seconds=60,
    )

    assert report["status"] == "ok"
    checks = _checks_by_name(report)
    assert checks["sample_selection"]["status"] == "ok"
    assert report["sample_selection"]["strategy"] == "stress_plus_random"  # type: ignore[index]
    assert report["sample_selection"]["stress_case_count"] == 2  # type: ignore[index]
    assert report["sample_selection"]["random_fill_case_count"] == 2  # type: ignore[index]


def test_certification_can_gate_prompt_size_and_adaptation_rate(tmp_path: Path) -> None:
    _write_completed_soak(
        tmp_path,
        case_count=4,
        elapsed_seconds=5.0,
        max_prompt_tokens=2400,
        runtime_max_prompt_tokens=3200,
        adapted_every=2,
    )

    report = certify.certify_soak(
        tmp_path,
        target_cases=10,
        max_duration_hours=1,
        min_pilot_cases=1,
        max_heartbeat_age_seconds=60,
        max_prompt_tokens=2000,
        max_prompt_budget_adapted_case_rate=0.25,
    )

    assert report["status"] == "error"
    checks = _checks_by_name(report)
    assert checks["max_prompt_tokens"]["status"] == "error"
    assert checks["prompt_budget_adapted_case_rate"]["status"] == "error"
    assert report["prompt_budget"]["max_prompt_tokens"] == 3200  # type: ignore[index]
    assert report["prompt_budget"]["adapted_case_rate"] == 0.5  # type: ignore[index]


def test_certification_cli_writes_json_report(tmp_path: Path) -> None:
    _write_completed_soak(tmp_path, case_count=1, elapsed_seconds=2.0)
    output_path = tmp_path / "cert.json"

    result = subprocess.run(
        [
            sys.executable,
            "tools/certify_qwen_caption_soak.py",
            str(tmp_path),
            "--target-cases",
            "10",
            "--max-duration-hours",
            "1",
            "--min-pilot-cases",
            "1",
            "--max-heartbeat-age",
            "60",
            "--write-json",
            str(output_path),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    stdout_report = json.loads(result.stdout)
    assert report["status"] == "ok"
    assert stdout_report["status"] == "ok"
    assert report["output_dir"] == str(tmp_path.resolve())
