from __future__ import annotations

import json
from pathlib import Path
import plistlib

from tools import run_qwen_caption_unattended as unattended
from tools import supervise_qwen_caption_soak as supervise


def _ok_drill_checks() -> dict[str, bool]:
    return {
        "supervisor_success": True,
        "saw_nonzero_exit": True,
        "saw_signal_exit": True,
        "saw_stale_heartbeat": True,
        "saw_missing_heartbeat": True,
        "final_audit_ok": True,
        "caption_io_retention_ok": True,
        "summary_totals_cover_all_cases": True,
        "summary_snapshot_bounded": True,
        "summary_truncated_when_over_limit": True,
        "summary_omits_rows_when_over_limit": True,
    }


def _ok_watchdog_drill_checks() -> dict[str, bool]:
    return {
        "watchdog_success": True,
        "saw_unhealthy_status": True,
        "saw_launchd_rebootstrap": True,
        "saw_bootstrap_success": True,
        "saw_restored_health": True,
        "latest_status_ok": True,
        "state_persisted": True,
    }


def _arg_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _ok_certification_report(pilot_output_dir: object, *, target_cases: int = 10_000) -> dict[str, object]:
    capabilities = list(unattended.runner.RUNNER_CAPABILITIES)
    pilot_cases = unattended.DEFAULT_TENK_MIN_PILOT_CASES
    return {
        "status": "ok",
        "output_dir": str(pilot_output_dir),
        "target_cases": target_cases,
        "checks": [
            {"name": "artifact_audit", "status": "ok"},
            {"name": "runner_capabilities", "status": "ok"},
            {"name": "deterministic_recovery_confidence", "status": "ok"},
        ],
        "runner_capabilities": {
            "runner_capabilities": capabilities,
            "required_capabilities": capabilities,
            "missing_capabilities": [],
            "capability_sources": ["manifest"],
        },
        "pilot_cases": pilot_cases,
        "generated_pilot_cases": pilot_cases,
        "generated_case_evidence": {
            "latest_cases": pilot_cases,
            "generated_cases": pilot_cases,
            "latest_skipped_cases": 0,
            "latest_skipped_cases_with_prior_generated_success": 0,
        },
        "prompt_budget": {
            "pilot_cases": pilot_cases,
            "rows_with_prompt_budget": pilot_cases,
            "adapted_cases": 0,
            "adapted_case_rate": 0.0,
            "max_prompt_tokens": 1200,
        },
        "qwen_caption_io_sources": {
            "required_rows": pilot_cases,
            "runtime_prompt_budget_rows": pilot_cases,
            "valid_runtime_prompt_budget_rows": pilot_cases,
            "invalid_runtime_rows_count": 0,
            "missing_runtime_rows_count": 0,
            "missing_trace_rows": 0,
            "source_counts": {"qwen_caption_io_per_run": pilot_cases},
            "accepted_sources": ["qwen_caption_io_per_run"],
            "missing_runtime_rows": [],
            "invalid_runtime_rows": [],
        },
        "deterministic_recovery_reliability": {
            "enabled": True,
            "confidence": unattended.certify.DEFAULT_DETERMINISTIC_RECOVERY_CONFIDENCE,
            "limit": 0.01,
            "generated_cases": pilot_cases,
            "deterministic_recovery_cases": 0,
            "observed_rate": 0.0,
            "upper_bound_rate": 0.009,
            "required_zero_recovery_cases": 268,
            "deterministic_recovery_case_keys_sample": [],
        },
    }


def _ok_post_install_operation_audit_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "ok",
        "strict_set_and_forget": True,
        "checks": [
            {"name": "runbook", "status": "ok"},
            {"name": "supervisor_launchd", "status": "ok"},
            {"name": "watchdog_launchd", "status": "ok"},
            {"name": "pmset_sleep_assertion", "status": "ok"},
        ],
        "post_install_poll": {
            "status": "ok",
            "attempts": 2,
            "timeout_seconds": 300.0,
            "interval_seconds": 5.0,
        },
    }


def test_unattended_plan_contains_supervisor_watchdog_and_audit_commands(tmp_path: Path) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--python",
        "/bin/python3",
        "--no-preflight",
        "--output-dir",
        str(tmp_path / "soak"),
        "--all-images",
        "--max-heartbeat-age",
        "123",
        "--max-recovery-event-case-rate",
        "0.1",
        "--max-attempt-overrun",
        "45",
        "--watchdog-max-no-progress",
        "1200",
        "--pilot-max-prompt-tokens",
        "9000",
        "--pilot-max-prompt-budget-adapted-case-rate",
        "0.2",
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)

    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )

    supervisor_cmd = plan["commands"]["supervisor"]
    watchdog_cmd = plan["commands"]["watchdog"]
    audit_cmd = plan["commands"]["final_audit"]
    live_status_cmd = plan["commands"]["live_status"]
    operational_audit_cmd = plan["commands"]["operational_audit"]
    certification_cmd = plan["commands"]["pilot_certification"]
    drill_cmd = plan["commands"]["supervisor_drill"]
    watchdog_drill_cmd = plan["commands"]["watchdog_drill"]
    assert supervisor_cmd[:2] == [
        "/bin/python3",
        str(unattended.REPO_ROOT / "tools" / "supervise_qwen_caption_soak.py"),
    ]
    assert "--all-images" in supervisor_cmd
    assert str(tmp_path / "soak") in supervisor_cmd
    assert watchdog_cmd[0] == "/bin/python3"
    assert "--max-heartbeat-age" in watchdog_cmd
    assert watchdog_cmd[watchdog_cmd.index("--max-heartbeat-age") + 1] == "123.0"
    assert _arg_value(watchdog_cmd, "--graceful-restart-timeout") == "300.0"
    assert "--max-recovery-event-case-rate" in watchdog_cmd
    assert _arg_value(watchdog_cmd, "--max-recovery-event-case-rate") == "0.1"
    assert _arg_value(watchdog_cmd, "--max-loop-recovery-case-rate") == "0.0"
    assert _arg_value(watchdog_cmd, "--max-loop-guard-case-rate") == "0.0"
    assert _arg_value(watchdog_cmd, "--max-deterministic-recovery-case-rate") == "0.0"
    assert "--max-attempt-overrun" in watchdog_cmd
    assert _arg_value(watchdog_cmd, "--max-attempt-overrun") == "45.0"
    assert _arg_value(watchdog_cmd, "--max-projected-duration-hours") == "0.0"
    assert _arg_value(watchdog_cmd, "--min-free-gb") == "5.0"
    assert "--max-no-progress" in watchdog_cmd
    assert _arg_value(watchdog_cmd, "--max-no-progress") == "1200.0"
    assert _arg_value(watchdog_cmd, "--latest-json") == plan["paths"]["watchdog_latest_json"]
    assert plan["paths"]["watchdog_latest_json"].endswith("watchdog_latest.json")
    assert _arg_value(watchdog_cmd, "--state-json") == plan["paths"]["watchdog_state_json"]
    assert plan["paths"]["watchdog_state_json"].endswith("watchdog_state.json")
    assert audit_cmd[-1] == "--pretty"
    assert "--max-attempt-overrun" in audit_cmd
    assert _arg_value(audit_cmd, "--max-attempt-overrun") == "45.0"
    assert _arg_value(audit_cmd, "--max-projected-duration-hours") == "0.0"
    assert _arg_value(audit_cmd, "--min-free-gb") == "5.0"
    assert _arg_value(audit_cmd, "--max-loop-recovery-case-rate") == "0.0"
    assert _arg_value(audit_cmd, "--max-loop-guard-case-rate") == "0.0"
    assert _arg_value(audit_cmd, "--max-deterministic-recovery-case-rate") == "0.0"
    assert live_status_cmd[:2] == [
        "/bin/python3",
        str(unattended.REPO_ROOT / "tools" / "audit_qwen_caption_soak.py"),
    ]
    assert "--allow-running-incomplete" in live_status_cmd
    assert "--compact" in live_status_cmd
    assert "--pretty" not in live_status_cmd
    assert _arg_value(live_status_cmd, "--max-heartbeat-age") == "123.0"
    assert _arg_value(live_status_cmd, "--max-attempt-overrun") == "45.0"
    assert _arg_value(live_status_cmd, "--min-free-gb") == "5.0"
    assert operational_audit_cmd[:2] == [
        "/bin/python3",
        str(unattended.REPO_ROOT / "tools" / "audit_qwen_caption_operation.py"),
    ]
    assert str(tmp_path / "soak" / unattended.RUNBOOK_NAME) in operational_audit_cmd
    assert "--allow-running-incomplete" in operational_audit_cmd
    assert "--compact" in operational_audit_cmd
    assert "--write-json" in operational_audit_cmd
    assert "--strict-set-and-forget" not in operational_audit_cmd
    assert _arg_value(operational_audit_cmd, "--write-json").endswith("operational_audit.json")
    assert certification_cmd[:2] == [
        "/bin/python3",
        str(unattended.REPO_ROOT / "tools" / "certify_qwen_caption_soak.py"),
    ]
    assert "--target-cases" in certification_cmd
    assert "--write-json" in certification_cmd
    assert "--max-prompt-tokens" in certification_cmd
    assert _arg_value(certification_cmd, "--max-prompt-tokens") == "9000"
    assert "--max-prompt-budget-adapted-case-rate" in certification_cmd
    assert _arg_value(certification_cmd, "--max-prompt-budget-adapted-case-rate") == "0.2"
    assert _arg_value(certification_cmd, "--max-loop-recovery-case-rate") == "0.0"
    assert _arg_value(certification_cmd, "--max-loop-guard-case-rate") == "0.0"
    assert _arg_value(certification_cmd, "--max-deterministic-recovery-case-rate") == "0.0"
    assert "--max-attempt-overrun" in certification_cmd
    assert certification_cmd[certification_cmd.index("--max-attempt-overrun") + 1] == "45.0"
    assert drill_cmd[:2] == [
        "/bin/python3",
        str(unattended.REPO_ROOT / "tools" / "run_qwen_caption_soak_drill.py"),
    ]
    assert "--force" in drill_cmd
    assert "--endurance-cases" in drill_cmd
    assert drill_cmd[drill_cmd.index("--endurance-cases") + 1] == str(unattended.SET_AND_FORGET_DRILL_CASES)
    assert "--endurance-chunk-size" in drill_cmd
    assert drill_cmd[drill_cmd.index("--endurance-chunk-size") + 1] == str(unattended.DEFAULT_SUPERVISOR_DRILL_CHUNK_SIZE)
    assert watchdog_drill_cmd[:2] == [
        "/bin/python3",
        str(unattended.REPO_ROOT / "tools" / "run_qwen_caption_soak_drill.py"),
    ]
    assert "--watchdog-remediation" in watchdog_drill_cmd
    assert "--write-json" in watchdog_drill_cmd
    assert plan["paths"]["runbook_json"].endswith("unattended_run.json")
    assert plan["paths"]["operational_audit_json"].endswith("operational_audit.json")
    assert plan["paths"]["readiness_json"].endswith("readiness.json")
    assert plan["paths"]["supervisor_drill_json"].endswith("supervisor_drill/drill_report.json")
    assert plan["paths"]["watchdog_drill_json"].endswith("watchdog_drill/watchdog_drill_report.json")
    assert plan["supervisor_args"]["output_dir"] == str(tmp_path / "soak")
    assert plan["supervisor_drill_gate"]["required"] is True
    assert plan["supervisor_drill_gate"]["case_count"] == unattended.SET_AND_FORGET_DRILL_CASES
    assert plan["supervisor_drill_gate"]["chunk_size"] == unattended.DEFAULT_SUPERVISOR_DRILL_CHUNK_SIZE
    assert plan["watchdog_drill_gate"]["required"] is True
    assert plan["certification_gate"]["required"] is False
    assert plan["certification_gate"]["require_prompt_budget_data"] is True
    assert plan["certification_gate"]["require_runner_capabilities"] is True
    assert plan["certification_gate"]["required_runner_capabilities"] == list(unattended.runner.RUNNER_CAPABILITIES)
    assert plan["certification_gate"]["max_prompt_tokens"] == 9000
    assert plan["certification_gate"]["max_prompt_budget_adapted_case_rate"] == 0.2


def test_unattended_plan_requires_saved_text_label_audits_when_saving(tmp_path: Path) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--python",
        "/bin/python3",
        "--output-dir",
        str(tmp_path / "soak"),
        "--all-images",
        "--save-dataset-text-labels",
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)

    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )

    assert "--save-dataset-text-labels" in plan["commands"]["supervisor"]
    assert "--require-saved-text-labels" in plan["commands"]["watchdog"]
    assert "--require-saved-text-labels" in plan["commands"]["final_audit"]
    assert "--require-saved-text-labels" in plan["commands"]["live_status"]


def test_unattended_parser_normalizes_separator_and_does_not_duplicate_extra_args(tmp_path: Path) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--python",
        "/bin/python3",
        "--output-dir",
        str(tmp_path / "soak"),
        "--",
        "--dataset-root",
        str(tmp_path / "dataset"),
        "--all-images",
        "--runner-extra-flag",
        "value",
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)

    assert "--" not in supervisor_argv
    assert supervisor_args.output_dir == tmp_path / "soak"
    assert supervisor_args.dataset_root == tmp_path / "dataset"
    assert supervisor_args.all_images is True
    assert supervisor_extra_args == ["--runner-extra-flag", "value"]

    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )
    supervisor_cmd = plan["commands"]["supervisor"]
    assert supervisor_cmd.count("--dataset-root") == 1
    assert supervisor_cmd.count("--runner-extra-flag") == 1
    assert "--" not in supervisor_cmd


def test_unattended_set_and_forget_plan_uses_bounded_recovery_defaults(tmp_path: Path) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--python",
        "/bin/python3",
        "--tenk-set-and-forget",
        "--output-dir",
        str(tmp_path / "soak"),
        "--all-images",
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)

    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )

    assert supervisor_args.set_and_forget is True
    assert supervisor_args.attempts == 3
    assert supervisor_args.max_projected_duration_hours == 336.0
    for command_name in ("watchdog", "final_audit", "pilot_certification"):
        command = plan["commands"][command_name]
        assert _arg_value(command, "--max-loop-recovery-case-rate") == "0.05"
        assert _arg_value(command, "--max-loop-guard-case-rate") == "0.05"
        assert _arg_value(command, "--max-deterministic-recovery-case-rate") == "0.01"
        assert _arg_value(command, "--max-signal-exit-attempt-row-rate") == "0.05"
    assert "--set-and-forget" in plan["commands"]["pilot_certification"]
    assert _arg_value(
        plan["commands"]["pilot_certification"],
        "--deterministic-recovery-confidence",
    ) == "0.95"
    assert _arg_value(plan["commands"]["supervisor"], "--attempts") == "3"
    assert _arg_value(plan["commands"]["pilot_generation"], "--attempts") == "3"
    assert _arg_value(plan["commands"]["watchdog"], "--max-projected-duration-hours") == "336.0"
    assert float(_arg_value(plan["commands"]["supervisor"], "--max-projected-duration-hours")) == 336.0
    assert _arg_value(plan["commands"]["final_audit"], "--max-projected-duration-hours") == "336.0"
    assert _arg_value(plan["commands"]["supervisor"], "--min-free-gb") == "5.0"
    assert _arg_value(plan["commands"]["watchdog"], "--min-free-gb") == "5.0"
    assert _arg_value(plan["commands"]["final_audit"], "--min-free-gb") == "5.0"
    assert _arg_value(plan["commands"]["live_status"], "--min-free-gb") == "5.0"
    assert "--strict-set-and-forget" in plan["commands"]["operational_audit"]
    assert plan["set_and_forget_gate"]["auto_pilot_generation"] is True
    assert plan["set_and_forget_gate"]["auto_prompt_size_ceiling"] is True
    assert plan["set_and_forget_gate"]["auto_pilot_min_cases"] is True
    assert plan["set_and_forget_gate"]["auto_pilot_sample_size"] is True
    assert plan["set_and_forget_gate"]["auto_attempts"] is True
    assert plan["set_and_forget_gate"]["attempts"] == 3
    assert plan["pilot_generation_gate"]["required"] is True
    assert plan["pilot_generation_gate"]["auto_configured"] is True
    assert plan["pilot_generation_gate"]["auto_sample_size"] is True
    assert plan["pilot_generation_gate"]["sample_size"] == unattended.DEFAULT_TENK_MIN_PILOT_CASES
    assert plan["pilot_generation_gate"]["output_dir"] == str((tmp_path / "soak" / "pilot").resolve(strict=False))
    assert plan["certification_gate"]["required"] is True
    assert plan["certification_gate"]["pilot_output_dir"] == str((tmp_path / "soak" / "pilot").resolve(strict=False))
    assert plan["certification_gate"]["max_prompt_tokens"] == unattended.DEFAULT_TENK_PILOT_MAX_PROMPT_TOKENS
    assert plan["certification_gate"]["auto_min_pilot_cases"] is True
    assert plan["certification_gate"]["min_pilot_cases"] == unattended.DEFAULT_TENK_MIN_PILOT_CASES
    assert plan["certification_gate"]["deterministic_recovery_confidence"] == 0.95
    assert plan["certification_gate"]["auto_prompt_size_ceiling"] is True
    assert "--min-free-gb" not in plan["commands"]["pilot_certification"]
    assert "--max-projected-duration-hours" not in plan["commands"]["pilot_certification"]


def test_unattended_windowed_set_and_forget_defaults_to_text_only_full_compose(tmp_path: Path) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--python",
        "/bin/python3",
        "--tenk-set-and-forget",
        "--output-dir",
        str(tmp_path / "soak"),
        "--all-images",
        "--caption-mode",
        "windowed",
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)

    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )

    assert supervisor_args.windowed_full_image_strategy == "text_only"
    assert _arg_value(plan["commands"]["supervisor"], "--windowed-full-image-strategy") == "text_only"
    assert _arg_value(plan["commands"]["pilot_generation"], "--windowed-full-image-strategy") == "text_only"
    assert plan["set_and_forget_gate"]["windowed_full_image_strategy"] == "text_only"
    assert plan["set_and_forget_gate"]["auto_windowed_full_image_strategy"] is True


def test_unattended_set_and_forget_defaults_to_safer_mlx_image_side(tmp_path: Path) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--python",
        "/bin/python3",
        "--tenk-set-and-forget",
        "--output-dir",
        str(tmp_path / "soak"),
        "--all-images",
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)

    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )

    expected = str(supervise.DEFAULT_SET_AND_FORGET_MLX_MAX_IMAGE_SIDE)
    expected_floor = str(supervise.DEFAULT_SET_AND_FORGET_MIN_RETRY_IMAGE_SIDE)
    expected_success_cooldown = str(supervise.DEFAULT_SET_AND_FORGET_COOLDOWN_AFTER_SUCCESS_SECONDS)
    assert supervisor_args.mlx_max_image_side == supervise.DEFAULT_SET_AND_FORGET_MLX_MAX_IMAGE_SIDE
    assert supervisor_args.min_retry_image_side == supervise.DEFAULT_SET_AND_FORGET_MIN_RETRY_IMAGE_SIDE
    assert supervisor_args.cooldown_after_success == supervise.DEFAULT_SET_AND_FORGET_COOLDOWN_AFTER_SUCCESS_SECONDS
    assert _arg_value(plan["commands"]["supervisor"], "--mlx-max-image-side") == expected
    assert _arg_value(plan["commands"]["supervisor"], "--min-retry-image-side") == expected_floor
    assert _arg_value(plan["commands"]["supervisor"], "--cooldown-after-success") == expected_success_cooldown
    assert _arg_value(plan["commands"]["pilot_generation"], "--mlx-max-image-side") == expected
    assert _arg_value(plan["commands"]["pilot_generation"], "--min-retry-image-side") == expected_floor
    assert _arg_value(plan["commands"]["pilot_generation"], "--cooldown-after-success") == expected_success_cooldown
    assert plan["set_and_forget_gate"]["auto_runtime_defaults"] is True
    assert plan["set_and_forget_gate"]["auto_mlx_max_image_side"] is True
    assert plan["set_and_forget_gate"]["auto_min_retry_image_side"] is True
    assert plan["set_and_forget_gate"]["auto_cooldown_after_success"] is True
    assert plan["set_and_forget_gate"]["mlx_max_image_side"] == supervise.DEFAULT_SET_AND_FORGET_MLX_MAX_IMAGE_SIDE
    assert plan["set_and_forget_gate"]["min_retry_image_side"] == supervise.DEFAULT_SET_AND_FORGET_MIN_RETRY_IMAGE_SIDE
    assert (
        plan["set_and_forget_gate"]["cooldown_after_success_seconds"]
        == supervise.DEFAULT_SET_AND_FORGET_COOLDOWN_AFTER_SUCCESS_SECONDS
    )


def test_unattended_set_and_forget_preserves_explicit_mlx_retry_sides(tmp_path: Path) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--python",
        "/bin/python3",
        "--tenk-set-and-forget",
        "--output-dir",
        str(tmp_path / "soak"),
        "--all-images",
        "--mlx-max-image-side",
        "384",
        "--min-retry-image-side",
        "192",
        "--cooldown-after-success",
        "0",
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)

    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )

    assert supervisor_args.mlx_max_image_side == 384
    assert supervisor_args.min_retry_image_side == 192
    assert supervisor_args.cooldown_after_success == 0
    assert _arg_value(plan["commands"]["supervisor"], "--mlx-max-image-side") == "384"
    assert _arg_value(plan["commands"]["supervisor"], "--min-retry-image-side") == "192"
    assert _arg_value(plan["commands"]["supervisor"], "--cooldown-after-success") == "0"
    assert _arg_value(plan["commands"]["pilot_generation"], "--mlx-max-image-side") == "384"
    assert _arg_value(plan["commands"]["pilot_generation"], "--min-retry-image-side") == "192"
    assert _arg_value(plan["commands"]["pilot_generation"], "--cooldown-after-success") == "0.0"
    assert _arg_value(plan["commands"]["supervisor"], "--attempts") == "3"
    assert _arg_value(plan["commands"]["pilot_generation"], "--attempts") == "3"
    assert plan["set_and_forget_gate"]["auto_runtime_defaults"] is True
    assert plan["set_and_forget_gate"]["auto_mlx_max_image_side"] is False
    assert plan["set_and_forget_gate"]["auto_min_retry_image_side"] is False
    assert plan["set_and_forget_gate"]["auto_cooldown_after_success"] is False
    assert plan["set_and_forget_gate"]["auto_attempts"] is True
    assert plan["set_and_forget_gate"]["mlx_max_image_side"] == 384
    assert plan["set_and_forget_gate"]["min_retry_image_side"] == 192
    assert plan["set_and_forget_gate"]["attempts"] == 3
    assert plan["set_and_forget_gate"]["cooldown_after_success_seconds"] == 0


def test_unattended_windowed_set_and_forget_preserves_explicit_visual_full_compose(tmp_path: Path) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--python",
        "/bin/python3",
        "--tenk-set-and-forget",
        "--output-dir",
        str(tmp_path / "soak"),
        "--all-images",
        "--caption-mode",
        "windowed",
        "--windowed-full-image-strategy",
        "visual",
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)

    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )

    assert supervisor_args.windowed_full_image_strategy == "visual"
    assert _arg_value(plan["commands"]["supervisor"], "--windowed-full-image-strategy") == "visual"
    assert _arg_value(plan["commands"]["pilot_generation"], "--windowed-full-image-strategy") == "visual"
    assert plan["set_and_forget_gate"]["windowed_full_image_strategy"] == "visual"
    assert plan["set_and_forget_gate"]["auto_windowed_full_image_strategy"] is False


def test_unattended_set_and_forget_honors_explicit_zero_signal_exit_rate(tmp_path: Path) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--python",
        "/bin/python3",
        "--tenk-set-and-forget",
        "--output-dir",
        str(tmp_path / "soak"),
        "--all-images",
        "--max-signal-exit-attempt-row-rate",
        "0",
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)

    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )

    for command_name in ("watchdog", "final_audit", "pilot_certification"):
        assert _arg_value(
            plan["commands"][command_name],
            "--max-signal-exit-attempt-row-rate",
        ) == "0.0"


def test_unattended_set_and_forget_preserves_explicit_pilot_size_overrides(tmp_path: Path) -> None:
    pilot_dir = tmp_path / "custom-pilot"
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--python",
        "/bin/python3",
        "--tenk-set-and-forget",
        "--create-pilot-output-dir",
        str(pilot_dir),
        "--pilot-sample-size",
        "12",
        "--pilot-min-cases",
        "12",
        "--pilot-max-prompt-tokens",
        "8000",
        "--output-dir",
        str(tmp_path / "soak"),
        "--all-images",
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)

    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )

    assert plan["set_and_forget_gate"]["auto_pilot_generation"] is False
    assert plan["set_and_forget_gate"]["auto_prompt_size_ceiling"] is False
    assert plan["set_and_forget_gate"]["auto_pilot_min_cases"] is False
    assert plan["set_and_forget_gate"]["auto_pilot_sample_size"] is False
    assert plan["pilot_generation_gate"]["output_dir"] == str(pilot_dir.resolve(strict=False))
    assert plan["pilot_generation_gate"]["sample_size"] == 12
    assert plan["pilot_generation_gate"]["auto_sample_size"] is False
    assert plan["certification_gate"]["pilot_output_dir"] == str(pilot_dir.resolve(strict=False))
    assert plan["certification_gate"]["min_pilot_cases"] == 12
    assert plan["certification_gate"]["auto_min_pilot_cases"] is False
    assert plan["certification_gate"]["max_prompt_tokens"] == 8000
    assert plan["certification_gate"]["auto_prompt_size_ceiling"] is False


def test_unattended_tenk_static_gate_rejects_diagnostic_pilot_size_overrides(tmp_path: Path) -> None:
    pilot_dir = tmp_path / "custom-pilot"
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--python",
        "/bin/python3",
        "--tenk-set-and-forget",
        "--create-pilot-output-dir",
        str(pilot_dir),
        "--pilot-sample-size",
        "12",
        "--pilot-min-cases",
        "12",
        "--pilot-max-prompt-tokens",
        "8000",
        "--pilot-deterministic-recovery-confidence",
        "0",
        "--output-dir",
        str(tmp_path / "soak"),
        "--all-images",
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)
    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )

    static_gate = unattended.build_static_tenk_launch_gate_report(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        plan=plan,
    )

    checks = {check["name"]: check for check in static_gate["checks"]}
    assert static_gate["status"] == "error"
    assert checks["pilot_confidence_sample_size"]["status"] == "error"
    assert checks["pilot_confidence_sample_size"]["required_pilot_cases"] == unattended.DEFAULT_TENK_MIN_PILOT_CASES
    assert checks["pilot_confidence_sample_size"]["min_pilot_cases"] == 12
    assert checks["pilot_confidence_sample_size"]["sample_size"] == 12
    assert checks["pilot_deterministic_recovery_confidence"]["status"] == "error"


def test_unattended_launchd_plist_restarts_only_unsuccessful_exit(tmp_path: Path) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--python",
        "/bin/python3",
        "--no-preflight",
        "--output-dir",
        str(tmp_path / "soak"),
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)
    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )
    plist_path = tmp_path / "com.example.qwen-caption.plist"

    payload = unattended.write_launchd_plist(
        plist_path,
        plan=plan,
        label="com.example.qwen-caption",
        throttle_interval=77,
    )

    loaded = plistlib.loads(plist_path.read_bytes())
    assert loaded == payload
    assert loaded["Label"] == "com.example.qwen-caption"
    assert loaded["RunAtLoad"] is True
    assert loaded["KeepAlive"] == {"SuccessfulExit": False}
    assert loaded["ThrottleInterval"] == 77
    assert loaded["ProgramArguments"] == plan["commands"]["supervisor"]
    assert loaded["StandardOutPath"].endswith("launchd_stdout.log")
    assert loaded["StandardErrorPath"].endswith("launchd_stderr.log")


def test_unattended_watchdog_launchd_plist_runs_independent_watchdog(tmp_path: Path) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--python",
        "/bin/python3",
        "--no-preflight",
        "--output-dir",
        str(tmp_path / "soak"),
        "--max-heartbeat-age",
        "321",
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)
    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )
    plist_path = tmp_path / "com.example.qwen-caption-watchdog.plist"

    payload = unattended.write_watchdog_launchd_plist(
        plist_path,
        plan=plan,
        label="com.example.qwen-caption-watchdog",
        throttle_interval=88,
    )

    loaded = plistlib.loads(plist_path.read_bytes())
    assert loaded == payload
    assert loaded["Label"] == "com.example.qwen-caption-watchdog"
    assert loaded["RunAtLoad"] is True
    assert loaded["KeepAlive"] == {"SuccessfulExit": False}
    assert loaded["ThrottleInterval"] == 88
    assert loaded["ProgramArguments"] == plan["commands"]["watchdog"]
    assert loaded["ProgramArguments"][1].endswith("watch_qwen_caption_soak.py")
    assert "--max-heartbeat-age" in loaded["ProgramArguments"]
    assert loaded["ProgramArguments"][loaded["ProgramArguments"].index("--max-heartbeat-age") + 1] == "321.0"
    assert loaded["StandardOutPath"].endswith("watchdog_launchd_stdout.log")
    assert loaded["StandardErrorPath"].endswith("watchdog_launchd_stderr.log")


def test_unattended_install_launchd_defaults_both_plist_paths() -> None:
    wrapper_args, _supervisor_argv = unattended.parse_wrapper_args([
        "--install-launchd-plists",
    ])

    assert wrapper_args.install_launchd_plists is True
    assert wrapper_args.write_launchd_plist.name == "com.tator.qwen-caption-soak.plist"
    assert wrapper_args.write_watchdog_launchd_plist.name == "com.tator.qwen-caption-soak.watchdog.plist"
    assert wrapper_args.write_launchd_plist.parent.name == "LaunchAgents"
    assert wrapper_args.write_watchdog_launchd_plist.parent.name == "LaunchAgents"


def test_unattended_dry_run_writes_runbook_and_stops_on_preflight_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {"status": "error", "checks": [{"name": "model_cache", "status": "error"}]},
    )
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("supervisor should not start")),
    )
    runbook = tmp_path / "runbook.json"

    exit_code = unattended.main([
        "--dry-run",
        "--skip-supervisor-drill",
        "--runbook-json",
        str(runbook),
        "--output-dir",
        str(tmp_path / "soak"),
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads(runbook.read_text())
    assert payload["preflight"]["status"] == "error"
    assert payload["commands"]["supervisor"][1].endswith("supervise_qwen_caption_soak.py")


def test_unattended_defers_live_lock_preflight_error_to_supervisor(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "error",
            "checks": [
                {
                    "name": "runner_lock",
                    "status": "error",
                    "detail": "output directory is currently owned by a live runner",
                    "pid": 123,
                }
            ],
        },
    )
    captured = {}

    def fake_supervise(supervisor_args, **kwargs):
        captured["output_dir"] = supervisor_args.output_dir
        captured["kwargs"] = kwargs
        return supervise.TERMINAL_SUCCESS

    monkeypatch.setattr(unattended.supervise, "supervise_soak", fake_supervise)
    runbook = tmp_path / "runbook.json"

    exit_code = unattended.main([
        "--runbook-json",
        str(runbook),
        "--skip-supervisor-drill",
        "--output-dir",
        str(tmp_path / "soak"),
    ])

    assert exit_code == supervise.TERMINAL_SUCCESS
    assert captured["output_dir"] == (tmp_path / "soak").resolve(strict=False)
    payload = json.loads(runbook.read_text())
    assert payload["preflight"]["status"] == "error"
    assert payload["preflight"]["deferred_to_supervisor"] is True
    assert payload["preflight"]["deferred_reason"] == "live_runner_lock"


def test_unattended_does_not_defer_live_lock_when_other_preflight_errors_exist(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "error",
            "checks": [
                {
                    "name": "runner_lock",
                    "status": "error",
                    "detail": "output directory is currently owned by a live runner",
                },
                {
                    "name": "resume_settings",
                    "status": "error",
                    "detail": "requested settings do not match existing run",
                },
            ],
        },
    )
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("mixed preflight errors should block")),
    )
    runbook = tmp_path / "runbook.json"

    exit_code = unattended.main([
        "--runbook-json",
        str(runbook),
        "--skip-supervisor-drill",
        "--output-dir",
        str(tmp_path / "soak"),
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads(runbook.read_text())
    assert payload["preflight"]["status"] == "error"
    assert "deferred_to_supervisor" not in payload["preflight"]


def test_unattended_dry_run_allows_live_lock_preflight_for_launchd_plan(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "error",
            "checks": [
                {
                    "name": "runner_lock",
                    "status": "error",
                    "detail": "output directory is currently owned by a live runner",
                }
            ],
        },
    )
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dry-run should not start supervisor")),
    )
    output_dir = tmp_path / "soak"

    exit_code = unattended.main([
        "--dry-run",
        "--skip-supervisor-drill",
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == 0
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    assert payload["preflight"]["status"] == "error"
    assert payload["preflight"]["deferred_to_supervisor"] is True
    assert payload["commands"]["supervisor"][1].endswith("supervise_qwen_caption_soak.py")


def test_unattended_required_pilot_certification_blocks_launch(monkeypatch, tmp_path: Path) -> None:
    pilot_dir = tmp_path / "pilot"
    output_dir = tmp_path / "soak"
    runbook = tmp_path / "runbook.json"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {"status": "ok", "checks": [{"name": "artifact_directory", "status": "ok"}]},
    )
    captured = {}

    def fake_certify(*_args, **kwargs):
        captured["kwargs"] = kwargs
        return {
            "status": "error",
            "checks": [{"name": "projected_duration", "status": "error"}],
        }

    monkeypatch.setattr(unattended.certify, "certify_soak", fake_certify)
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("supervisor should not start")),
    )

    exit_code = unattended.main([
        "--dry-run",
        "--skip-supervisor-drill",
        "--require-pilot-certification",
        str(pilot_dir),
        "--no-pilot-prompt-budget-required",
        "--pilot-max-prompt-tokens",
        "7500",
        "--pilot-max-prompt-budget-adapted-case-rate",
        "0.1",
        "--pilot-max-p95-duration-hours",
        "72",
        "--runbook-json",
        str(runbook),
        "--output-dir",
        str(output_dir),
        "--all-images",
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads(runbook.read_text())
    assert payload["preflight"]["status"] == "ok"
    assert payload["required_pilot_certification"]["status"] == "error"
    assert payload["certification_gate"]["required"] is True
    assert payload["certification_gate"]["pilot_output_dir"] == str(pilot_dir.resolve(strict=False))
    assert payload["certification_gate"]["require_prompt_budget_data"] is False
    assert payload["certification_gate"]["max_prompt_tokens"] == 7500
    assert payload["certification_gate"]["max_prompt_budget_adapted_case_rate"] == 0.1
    assert payload["certification_gate"]["max_p95_duration_hours"] == 72
    assert payload["certification_gate"]["require_runner_capabilities"] is True
    assert "--no-require-prompt-budget-data" in payload["commands"]["pilot_certification"]
    assert payload["certification_gate"]["expected_run_settings_fingerprint"]
    assert (
        captured["kwargs"]["expected_run_settings_fingerprint"]
        == payload["certification_gate"]["expected_run_settings_fingerprint"]
    )
    assert captured["kwargs"]["require_prompt_budget_data"] is False
    assert captured["kwargs"]["max_prompt_tokens"] == 7500
    assert captured["kwargs"]["max_prompt_budget_adapted_case_rate"] == 0.1
    assert captured["kwargs"]["max_p95_duration_hours"] == 72
    certification_report = json.loads((output_dir / "required_pilot_certification.json").read_text())
    assert certification_report["status"] == "error"


def test_unattended_required_pilot_certification_allows_clean_dry_run(monkeypatch, tmp_path: Path) -> None:
    pilot_dir = tmp_path / "pilot"
    output_dir = tmp_path / "soak"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {"status": "ok", "checks": [{"name": "artifact_directory", "status": "ok"}]},
    )
    captured = {}

    def fake_certify(pilot_output_dir, **kwargs):
        captured["kwargs"] = kwargs
        report = _ok_certification_report(pilot_output_dir, target_cases=kwargs["target_cases"])
        report["run_settings"] = {
            "expected_fingerprint": kwargs["expected_run_settings_fingerprint"],
            "match_required": True,
        }
        return report

    monkeypatch.setattr(unattended.certify, "certify_soak", fake_certify)
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dry-run should not start supervisor")),
    )

    exit_code = unattended.main([
        "--dry-run",
        "--skip-supervisor-drill",
        "--require-pilot-certification",
        str(pilot_dir),
        "--pilot-target-cases",
        "10000",
        "--pilot-max-duration-hours",
        "48",
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == 0
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    assert payload["required_pilot_certification"]["status"] == "ok"
    assert payload["required_pilot_certification"]["target_cases"] == unattended.SET_AND_FORGET_DRILL_CASES
    assert payload["certification_gate"]["max_p95_duration_hours"] == 48
    assert payload["certification_gate"]["expected_run_settings_fingerprint"]
    assert (
        captured["kwargs"]["expected_run_settings_fingerprint"]
        == payload["certification_gate"]["expected_run_settings_fingerprint"]
    )
    assert captured["kwargs"]["max_p95_duration_hours"] == 48


def test_unattended_plan_contains_generated_pilot_command(tmp_path: Path) -> None:
    pilot_dir = tmp_path / "pilot"
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--python",
        "/bin/python3",
        "--no-preflight",
        "--create-pilot-output-dir",
        str(pilot_dir),
        "--pilot-sample-size",
        "77",
        "--output-dir",
        str(tmp_path / "soak"),
        "--all-images",
        "--save-dataset-text-labels",
        "--model-id",
        "example/model",
        "--artifact-lock-timeout",
        "12",
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)

    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )

    pilot_cmd = plan["commands"]["pilot_generation"]
    certification_cmd = plan["commands"]["pilot_certification"]
    assert plan["pilot_generation_gate"]["required"] is True
    assert plan["pilot_generation_gate"]["output_dir"] == str(pilot_dir.resolve(strict=False))
    assert plan["pilot_generation_gate"]["sample_size"] == 77
    assert plan["pilot_generation_gate"]["read_only"] is True
    assert plan["pilot_generation_gate"]["large_run_save_dataset_text_labels"] is True
    assert plan["certification_gate"]["required"] is True
    assert plan["certification_gate"]["pilot_output_dir"] == str(pilot_dir.resolve(strict=False))
    assert pilot_cmd[:2] == [
        "/bin/python3",
        str(unattended.REPO_ROOT / "tools" / "supervise_qwen_caption_soak.py"),
    ]
    assert "--output-dir" in pilot_cmd
    assert pilot_cmd[pilot_cmd.index("--output-dir") + 1] == str(pilot_dir.resolve(strict=False))
    assert "--sample-size" in pilot_cmd
    assert pilot_cmd[pilot_cmd.index("--sample-size") + 1] == "77"
    assert "--model-id" in pilot_cmd
    assert pilot_cmd[pilot_cmd.index("--model-id") + 1] == "example/model"
    assert "--all-images" in pilot_cmd
    assert "--save-dataset-text-labels" in plan["commands"]["supervisor"]
    assert "--save-dataset-text-labels" not in pilot_cmd
    assert "--artifact-lock-timeout" in plan["commands"]["supervisor"]
    assert plan["commands"]["supervisor"][plan["commands"]["supervisor"].index("--artifact-lock-timeout") + 1] == "12"
    assert "--artifact-lock-timeout" in pilot_cmd
    assert pilot_cmd[pilot_cmd.index("--artifact-lock-timeout") + 1] == "12"
    assert "--write-json" in certification_cmd
    assert "--max-p95-duration-hours" in certification_cmd
    assert certification_cmd[certification_cmd.index("--max-p95-duration-hours") + 1] == str(
        unattended.certify.DEFAULT_MAX_DURATION_HOURS
    )
    assert certification_cmd[certification_cmd.index("--write-json") + 1].endswith(
        "required_pilot_certification.json"
    )
    pilot_args = unattended._pilot_supervisor_args(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        output_dir=pilot_dir.resolve(strict=False),
    )
    assert pilot_args.save_dataset_text_labels is False
    assert unattended._expected_run_settings_fingerprint(pilot_args) == unattended._expected_run_settings_fingerprint(
        supervisor_args
    )


def test_unattended_generated_pilot_runs_before_large_supervisor(monkeypatch, tmp_path: Path) -> None:
    pilot_dir = tmp_path / "pilot"
    output_dir = tmp_path / "soak"
    calls = []
    captured_cert = {}

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {"status": "ok", "checks": [{"name": "artifact_directory", "status": "ok"}]},
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
    )

    def fake_supervise(supervisor_args, **kwargs):
        calls.append(
            {
                "output_dir": supervisor_args.output_dir,
                "sample_size": supervisor_args.sample_size,
                "all_images": supervisor_args.all_images,
                "save_dataset_text_labels": supervisor_args.save_dataset_text_labels,
                "kwargs": kwargs,
            }
        )
        return supervise.TERMINAL_SUCCESS

    def fake_certify(pilot_output_dir, **kwargs):
        captured_cert["pilot_output_dir"] = pilot_output_dir
        captured_cert["kwargs"] = kwargs
        return _ok_certification_report(pilot_output_dir, target_cases=kwargs["target_cases"])

    monkeypatch.setattr(unattended.supervise, "supervise_soak", fake_supervise)
    monkeypatch.setattr(unattended.certify, "certify_soak", fake_certify)

    exit_code = unattended.main([
        "--create-pilot-output-dir",
        str(pilot_dir),
        "--pilot-sample-size",
        "5",
        "--pilot-min-cases",
        "5",
        "--output-dir",
        str(output_dir),
        "--all-images",
        "--save-dataset-text-labels",
    ])

    assert exit_code == supervise.TERMINAL_SUCCESS
    assert [call["output_dir"] for call in calls] == [
        pilot_dir.resolve(strict=False),
        output_dir.resolve(strict=False),
    ]
    assert calls[0]["sample_size"] == 5
    assert calls[0]["all_images"] is True
    assert calls[0]["save_dataset_text_labels"] is False
    assert calls[1]["all_images"] is True
    assert calls[1]["save_dataset_text_labels"] is True
    assert captured_cert["pilot_output_dir"] == pilot_dir.resolve(strict=False)
    assert captured_cert["kwargs"]["min_pilot_cases"] == 5
    assert captured_cert["kwargs"]["max_p95_duration_hours"] == unattended.certify.DEFAULT_MAX_DURATION_HOURS
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    assert payload["created_pilot_generation"]["status"] == "ok"
    assert payload["required_pilot_certification"]["status"] == "ok"
    assert payload["readiness"]["summary"]["pilot_generation_required"] is True
    assert payload["readiness"]["summary"]["pilot_generation_status"] == "ok"


def test_unattended_strict_readiness_blocks_warning_only_plan(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "soak"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "ok",
            "checks": [
                {"name": "artifact_directory", "status": "ok"},
                {"name": "disk_budget", "status": "ok", "detail": "disk is fine"},
            ],
            "model_cache": {
                "models": [{"role": "caption", "model_id": "example/model", "needs_download": False}],
            },
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("strict dry-run should not start supervisor")),
    )

    exit_code = unattended.main([
        "--dry-run",
        "--require-readiness-ok",
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    readiness = payload["readiness"]
    assert readiness["status"] == "warn"
    assert readiness["ready_for_10k_set_and_forget"] is False
    check_by_name = {check["name"]: check for check in readiness["checks"]}
    assert check_by_name["pilot_certification"]["status"] == "warn"
    assert check_by_name["launchd_restart"]["status"] == "warn"
    assert check_by_name["watchdog_launchd"]["status"] == "warn"


def test_unattended_readiness_rejects_stale_supervisor_drill_without_retention_gate(tmp_path: Path) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--output-dir",
        str(tmp_path / "soak"),
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)
    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )
    stale_checks = _ok_drill_checks()
    stale_checks.pop("caption_io_retention_ok")

    report = unattended.build_readiness_report(
        plan=plan,
        wrapper_args=wrapper_args,
        preflight_report={
            "status": "ok",
            "checks": [{"name": "artifact_directory", "status": "ok"}],
            "model_cache": {"models": []},
        },
        supervisor_drill_report={
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": stale_checks,
        },
        watchdog_drill_report={
            "status": "ok",
            "checks": _ok_watchdog_drill_checks(),
        },
        pilot_generation_report=None,
        certification_report=None,
        launch_blocked=False,
    )

    check_by_name = {check["name"]: check for check in report["checks"]}
    assert check_by_name["supervisor_drill"]["status"] == "error"
    assert check_by_name["supervisor_drill"]["missing_checks"] == ["caption_io_retention_ok"]


def test_unattended_tenk_readiness_rejects_disabled_projected_duration_gate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "soak"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "ok",
            "checks": [{"name": "artifact_directory", "status": "ok"}],
            "model_cache": {
                "models": [{"role": "caption", "model_id": "example/model", "needs_download": False}],
            },
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tenk dry-run should not start supervisor")),
    )

    exit_code = unattended.main([
        "--dry-run",
        "--tenk-set-and-forget",
        "--output-dir",
        str(output_dir),
        "--max-projected-duration-hours",
        "0",
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    readiness = payload["readiness"]
    check_by_name = {check["name"]: check for check in readiness["checks"]}
    assert readiness["status"] == "error"
    assert check_by_name["projected_duration_gate"]["status"] == "error"
    assert check_by_name["projected_duration_gate"]["missing_or_disabled"] == ["watchdog", "final_audit"]
    assert _arg_value(payload["commands"]["watchdog"], "--max-projected-duration-hours") == "0.0"
    readiness_json = json.loads((output_dir / unattended.READINESS_NAME).read_text())
    assert readiness_json == readiness


def test_unattended_tenk_readiness_rejects_disabled_graceful_restart(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "soak"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "ok",
            "checks": [{"name": "artifact_directory", "status": "ok"}],
            "model_cache": {
                "models": [{"role": "caption", "model_id": "example/model", "needs_download": False}],
            },
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_watchdog_remediation_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "checks": _ok_watchdog_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tenk dry-run should not start supervisor")),
    )

    exit_code = unattended.main([
        "--dry-run",
        "--tenk-set-and-forget",
        "--watchdog-graceful-restart-timeout",
        "0",
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    readiness = payload["readiness"]
    check_by_name = {check["name"]: check for check in readiness["checks"]}
    assert readiness["status"] == "error"
    assert check_by_name["watchdog_graceful_restart"]["status"] == "error"
    assert _arg_value(payload["commands"]["watchdog"], "--graceful-restart-timeout") == "0.0"


def test_unattended_tenk_readiness_rejects_disabled_prompt_size_ceiling(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pilot_dir = tmp_path / "pilot"
    output_dir = tmp_path / "soak"
    launch_agents_dir = tmp_path / "Library" / "LaunchAgents"
    plist_path = launch_agents_dir / "com.example.qwen-caption.plist"
    watchdog_plist_path = launch_agents_dir / "com.example.qwen-caption-watchdog.plist"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "ok",
            "checks": [{"name": "artifact_directory", "status": "ok"}],
            "model_cache": {
                "models": [{"role": "caption", "model_id": "example/model", "needs_download": False}],
            },
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_watchdog_remediation_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "checks": _ok_watchdog_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.certify,
        "certify_soak",
        lambda pilot_output_dir, **kwargs: _ok_certification_report(
            pilot_output_dir,
            target_cases=kwargs["target_cases"],
        ),
    )
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tenk dry-run should not start supervisor")),
    )

    exit_code = unattended.main([
        "--dry-run",
        "--tenk-set-and-forget",
        "--require-pilot-certification",
        str(pilot_dir),
        "--pilot-max-prompt-tokens",
        "0",
        "--install-launchd-plists",
        "--write-launchd-plist",
        str(plist_path),
        "--write-watchdog-launchd-plist",
        str(watchdog_plist_path),
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    readiness = payload["readiness"]
    check_by_name = {check["name"]: check for check in readiness["checks"]}
    static_checks = {
        check["name"]: check
        for check in payload["tenk_static_launch_gate"]["checks"]
    }
    assert readiness["status"] == "error"
    assert payload["tenk_static_launch_gate"]["status"] == "error"
    assert check_by_name["tenk_static_launch_gate"]["status"] == "error"
    assert check_by_name["prompt_size_ceiling"]["status"] == "error"
    assert static_checks["prompt_size_ceiling"]["status"] == "error"
    assert check_by_name["pilot_prompt_budget_evidence"]["status"] == "error"
    assert "required_pilot_certification" not in payload
    assert payload["certification_gate"]["max_prompt_tokens"] == 0


def test_unattended_tenk_static_gate_blocks_created_pilot_before_gpu_work(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pilot_dir = tmp_path / "pilot"
    output_dir = tmp_path / "soak"
    launch_agents_dir = tmp_path / "Library" / "LaunchAgents"
    plist_path = launch_agents_dir / "com.example.qwen-caption.plist"
    watchdog_plist_path = launch_agents_dir / "com.example.qwen-caption-watchdog.plist"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "ok",
            "checks": [{"name": "artifact_directory", "status": "ok"}],
            "model_cache": {
                "models": [{"role": "caption", "model_id": "example/model", "needs_download": False}],
            },
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("static gate should block before drills")),
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_watchdog_remediation_drill",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("static gate should block before watchdog drill")),
    )
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("static gate should block before pilot/supervisor")),
    )
    monkeypatch.setattr(
        unattended.certify,
        "certify_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("static gate should block before certification")),
    )

    exit_code = unattended.main([
        "--tenk-set-and-forget",
        "--create-pilot-output-dir",
        str(pilot_dir),
        "--pilot-max-prompt-tokens",
        "0",
        "--install-launchd-plists",
        "--write-launchd-plist",
        str(plist_path),
        "--write-watchdog-launchd-plist",
        str(watchdog_plist_path),
        "--output-dir",
        str(output_dir),
        "--all-images",
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    static_checks = {
        check["name"]: check
        for check in payload["tenk_static_launch_gate"]["checks"]
    }
    readiness_checks = {check["name"]: check for check in payload["readiness"]["checks"]}
    assert payload["tenk_static_launch_gate"]["status"] == "error"
    assert static_checks["prompt_size_ceiling"]["status"] == "error"
    assert readiness_checks["tenk_static_launch_gate"]["status"] == "error"
    assert "supervisor_drill" not in payload
    assert "watchdog_drill" not in payload
    assert "created_pilot_generation" not in payload
    assert "required_pilot_certification" not in payload


def test_unattended_tenk_static_gate_rejects_model_downloads_before_gpu_work(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pilot_dir = tmp_path / "pilot"
    output_dir = tmp_path / "soak"
    launch_agents_dir = tmp_path / "Library" / "LaunchAgents"
    plist_path = launch_agents_dir / "com.example.qwen-caption.plist"
    watchdog_plist_path = launch_agents_dir / "com.example.qwen-caption-watchdog.plist"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "ok",
            "checks": [{"name": "artifact_directory", "status": "ok"}],
            "model_cache": {
                "models": [{"role": "caption", "model_id": "example/model", "needs_download": False}],
            },
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("static gate should block before drills")),
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_watchdog_remediation_drill",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("static gate should block before watchdog drill")),
    )
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("static gate should block before pilot/supervisor")),
    )
    monkeypatch.setattr(
        unattended.certify,
        "certify_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("static gate should block before certification")),
    )

    exit_code = unattended.main([
        "--tenk-set-and-forget",
        "--create-pilot-output-dir",
        str(pilot_dir),
        "--allow-model-download",
        "--install-launchd-plists",
        "--write-launchd-plist",
        str(plist_path),
        "--write-watchdog-launchd-plist",
        str(watchdog_plist_path),
        "--output-dir",
        str(output_dir),
        "--all-images",
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    static_checks = {
        check["name"]: check
        for check in payload["tenk_static_launch_gate"]["checks"]
    }
    readiness_checks = {check["name"]: check for check in payload["readiness"]["checks"]}
    assert payload["tenk_static_launch_gate"]["status"] == "error"
    assert static_checks["model_downloads_disabled"]["status"] == "error"
    assert static_checks["model_downloads_disabled"]["allow_model_download"] is True
    assert readiness_checks["tenk_static_launch_gate"]["status"] == "error"
    assert "supervisor_drill" not in payload
    assert "watchdog_drill" not in payload
    assert "created_pilot_generation" not in payload
    assert "required_pilot_certification" not in payload


def test_unattended_tenk_static_gate_rejects_single_attempt_before_gpu_work(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pilot_dir = tmp_path / "pilot"
    output_dir = tmp_path / "soak"
    launch_agents_dir = tmp_path / "Library" / "LaunchAgents"
    plist_path = launch_agents_dir / "com.example.qwen-caption.plist"
    watchdog_plist_path = launch_agents_dir / "com.example.qwen-caption-watchdog.plist"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "ok",
            "checks": [{"name": "artifact_directory", "status": "ok"}],
            "model_cache": {
                "models": [{"role": "caption", "model_id": "example/model", "needs_download": False}],
            },
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("static gate should block before drills")),
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_watchdog_remediation_drill",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("static gate should block before watchdog drill")),
    )
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("static gate should block before pilot/supervisor")),
    )
    monkeypatch.setattr(
        unattended.certify,
        "certify_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("static gate should block before certification")),
    )

    exit_code = unattended.main([
        "--tenk-set-and-forget",
        "--create-pilot-output-dir",
        str(pilot_dir),
        "--install-launchd-plists",
        "--write-launchd-plist",
        str(plist_path),
        "--write-watchdog-launchd-plist",
        str(watchdog_plist_path),
        "--output-dir",
        str(output_dir),
        "--all-images",
        "--attempts",
        "1",
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    static_checks = {
        check["name"]: check
        for check in payload["tenk_static_launch_gate"]["checks"]
    }
    readiness_checks = {check["name"]: check for check in payload["readiness"]["checks"]}
    assert payload["tenk_static_launch_gate"]["status"] == "error"
    assert static_checks["vlm_retry_attempts"]["status"] == "error"
    assert static_checks["vlm_retry_attempts"]["attempts"] == 1
    assert static_checks["vlm_retry_attempts"]["required_attempts"] == 3
    assert readiness_checks["tenk_static_launch_gate"]["status"] == "error"
    assert "supervisor_drill" not in payload
    assert "watchdog_drill" not in payload
    assert "created_pilot_generation" not in payload
    assert "required_pilot_certification" not in payload


def test_unattended_tenk_static_gate_rejects_pilot_dir_equal_to_target_before_gpu_work(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "soak"
    launch_agents_dir = tmp_path / "Library" / "LaunchAgents"
    plist_path = launch_agents_dir / "com.example.qwen-caption.plist"
    watchdog_plist_path = launch_agents_dir / "com.example.qwen-caption-watchdog.plist"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "ok",
            "checks": [
                {"name": "artifact_directory", "status": "ok"},
                {"name": "disk_budget", "status": "ok", "detail": "disk is fine"},
            ],
            "model_cache": {
                "models": [{"role": "caption", "model_id": "example/model", "needs_download": False}],
            },
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("static gate should block before drills")),
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_watchdog_remediation_drill",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("static gate should block before watchdog drill")),
    )
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("static gate should block before pilot/supervisor")),
    )
    monkeypatch.setattr(
        unattended.certify,
        "certify_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("static gate should block before certification")),
    )

    exit_code = unattended.main([
        "--tenk-set-and-forget",
        "--create-pilot-output-dir",
        str(output_dir),
        "--install-launchd-plists",
        "--write-launchd-plist",
        str(plist_path),
        "--write-watchdog-launchd-plist",
        str(watchdog_plist_path),
        "--output-dir",
        str(output_dir),
        "--all-images",
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    static_checks = {
        check["name"]: check
        for check in payload["tenk_static_launch_gate"]["checks"]
    }
    readiness_checks = {check["name"]: check for check in payload["readiness"]["checks"]}
    assert payload["tenk_static_launch_gate"]["status"] == "error"
    assert static_checks["pilot_output_dir_isolated"]["status"] == "error"
    assert static_checks["pilot_output_dir_isolated"]["pilot_output_dir"] == str(output_dir.resolve(strict=False))
    assert static_checks["pilot_output_dir_isolated"]["target_output_dir"] == str(output_dir.resolve(strict=False))
    assert readiness_checks["tenk_static_launch_gate"]["status"] == "error"
    assert "supervisor_drill" not in payload
    assert "watchdog_drill" not in payload
    assert "created_pilot_generation" not in payload
    assert "required_pilot_certification" not in payload


def test_unattended_tenk_mode_enables_strict_readiness(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "soak"

    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--tenk-set-and-forget",
        "--output-dir",
        str(output_dir),
    ])
    assert wrapper_args.tenk_set_and_forget is True
    assert wrapper_args.require_readiness_ok is True
    assert wrapper_args.launchd_caffeinate is True
    assert "--set-and-forget" in supervisor_argv

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "ok",
            "checks": [{"name": "artifact_directory", "status": "ok"}],
            "model_cache": {
                "models": [{"role": "caption", "model_id": "example/model", "needs_download": False}],
            },
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tenk dry-run should not start supervisor")),
    )

    exit_code = unattended.main([
        "--dry-run",
        "--tenk-set-and-forget",
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    readiness = payload["readiness"]
    assert payload["set_and_forget_gate"]["tenk_mode"] is True
    assert payload["set_and_forget_gate"]["require_readiness_ok"] is True
    assert readiness["tenk_set_and_forget"] is True
    assert readiness["requires_readiness_ok"] is True
    assert readiness["summary"]["tenk_set_and_forget"] is True
    assert "--set-and-forget" in payload["commands"]["supervisor"]
    assert readiness["ready_for_10k_set_and_forget"] is False
    check_by_name = {check["name"]: check for check in readiness["checks"]}
    assert readiness["status"] == "error"
    assert check_by_name["pilot_certification"]["status"] == "error"
    assert check_by_name["launchd_restart"]["status"] == "error"
    assert check_by_name["watchdog_launchd"]["status"] == "error"
    assert check_by_name["watchdog_remediation"]["status"] == "error"
    assert check_by_name["launchd_install"]["status"] == "error"
    assert check_by_name["projected_duration_gate"]["status"] == "ok"
    assert check_by_name["projected_duration_gate"]["watchdog_max_projected_duration_hours"] == 336.0
    assert check_by_name["disk_reserve_gate"]["status"] == "ok"
    assert check_by_name["disk_reserve_gate"]["requested_min_free_gb"] == 5.0
    assert check_by_name["launchd_power_assertion"]["status"] == "ok"


def test_unattended_tenk_adopts_live_run_with_launchd_handoff(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "soak"
    launch_agents = tmp_path / "Library" / "LaunchAgents"
    supervisor_plist = launch_agents / "com.example.qwen-caption.plist"
    watchdog_plist = launch_agents / "com.example.qwen-caption.watchdog.plist"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "error",
            "checks": [
                {
                    "name": "runner_lock",
                    "status": "error",
                    "detail": "output directory is currently owned by a live runner",
                    "pid": 1234,
                    "runner_supports_graceful_restart": True,
                    "runner_capabilities": [unattended.runner.RUNNER_CAPABILITY_GRACEFUL_RESTART],
                    "runner_capability_sources": ["runner_lock"],
                }
            ],
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_watchdog_remediation_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "checks": _ok_watchdog_drill_checks(),
        },
    )
    live_audit_calls = []

    def fake_live_audit(*_args, **kwargs):
        live_audit_calls.append(kwargs)
        return {
            "status": "ok",
            "processed_cases": 120,
            "expected_cases": 10000,
            "degraded_rates": {
                "prompt_budget_coverage_rate": 1.0,
                "max_prompt_tokens": 2200,
            },
            "runtime_projection": {"status": "ok", "projected_duration_hours": 42.0},
            "active_attempt": {"case": "image_00121"},
            "disk_reserve": {"status": "ok"},
        }

    monkeypatch.setattr(unattended.supervise.audit, "audit_soak", fake_live_audit)
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("launchd install should not run foreground supervisor")),
    )
    monkeypatch.setattr(
        unattended,
        "install_launchd_plists",
        lambda _plan: {
            "status": "ok",
            "requested": True,
            "detail": "supervisor and watchdog LaunchAgents were bootstrapped and verified",
            "steps": [],
        },
    )
    monkeypatch.setattr(
        unattended,
        "run_post_install_operation_audit",
        lambda **_kwargs: _ok_post_install_operation_audit_report(),
    )

    exit_code = unattended.main([
        "--tenk-set-and-forget",
        "--install-launchd-plists",
        "--adopt-live-run",
        "--pilot-max-prompt-tokens",
        "9000",
        "--write-launchd-plist",
        str(supervisor_plist),
        "--write-watchdog-launchd-plist",
        str(watchdog_plist),
        "--output-dir",
        str(output_dir),
        "--all-images",
        "--save-dataset-text-labels",
    ])

    assert exit_code == supervise.TERMINAL_SUCCESS
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    readiness = payload["readiness"]
    checks = {check["name"]: check for check in readiness["checks"]}
    assert payload["preflight"]["deferred_to_supervisor"] is True
    assert payload["live_adoption_certification"]["status"] == "ok"
    assert live_audit_calls[0]["require_saved_text_labels"] is True
    assert readiness["status"] == "ok"
    assert readiness["ready_for_10k_set_and_forget"] is True
    assert checks["preflight"]["status"] == "ok"
    assert checks["live_adoption"]["status"] == "ok"
    adoption_checks = {
        check["name"]: check
        for check in payload["live_adoption_certification"]["checks"]
    }
    assert adoption_checks["live_runner_restart_capability"]["status"] == "ok"
    assert checks["pilot_certification"]["status"] == "ok"
    assert checks["prompt_size_ceiling"]["status"] == "ok"
    assert checks["launchd_install_result"]["status"] == "ok"
    assert checks["post_install_operation_audit"]["status"] == "ok"
    assert payload["post_install_operation_audit"]["status"] == "ok"
    assert "--max-projected-duration-hours" in payload["commands"]["supervisor"]
    assert float(_arg_value(payload["commands"]["supervisor"], "--max-projected-duration-hours")) == 336.0
    supervisor_payload = plistlib.loads(supervisor_plist.read_bytes())
    watchdog_payload = plistlib.loads(watchdog_plist.read_bytes())
    assert supervisor_payload["ProgramArguments"][:2] == [
        unattended.CAFFEINATE_BIN,
        unattended.CAFFEINATE_ARGS[0],
    ]
    assert watchdog_payload["ProgramArguments"][:2] == [
        unattended.CAFFEINATE_BIN,
        unattended.CAFFEINATE_ARGS[0],
    ]
    assert supervisor_payload["KeepAlive"] == {"SuccessfulExit": False}
    assert watchdog_payload["KeepAlive"] == {"SuccessfulExit": False}


def test_unattended_tenk_adoption_rejects_disabled_prompt_size_gate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "soak"
    launch_agents = tmp_path / "Library" / "LaunchAgents"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "error",
            "checks": [
                {
                    "name": "runner_lock",
                    "status": "error",
                    "detail": "output directory is currently owned by a live runner",
                    "pid": 1234,
                    "runner_supports_graceful_restart": True,
                    "runner_capabilities": [unattended.runner.RUNNER_CAPABILITY_GRACEFUL_RESTART],
                    "runner_capability_sources": ["runner_lock"],
                }
            ],
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_watchdog_remediation_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "checks": _ok_watchdog_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.supervise.audit,
        "audit_soak",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "processed_cases": 120,
            "expected_cases": 10000,
            "degraded_rates": {
                "prompt_budget_coverage_rate": 1.0,
                "max_prompt_tokens": 2200,
            },
            "runtime_projection": {"status": "ok", "projected_duration_hours": 42.0},
            "disk_reserve": {"status": "ok"},
        },
    )
    monkeypatch.setattr(
        unattended,
        "install_launchd_plists",
        lambda _plan: (_ for _ in ()).throw(AssertionError("failed adoption should not install launchd")),
    )

    exit_code = unattended.main([
        "--tenk-set-and-forget",
        "--install-launchd-plists",
        "--adopt-live-run",
        "--pilot-max-prompt-tokens",
        "0",
        "--write-launchd-plist",
        str(launch_agents / "com.example.qwen-caption.plist"),
        "--write-watchdog-launchd-plist",
        str(launch_agents / "com.example.qwen-caption.watchdog.plist"),
        "--output-dir",
        str(output_dir),
        "--all-images",
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    static_checks = {
        check["name"]: check
        for check in payload["tenk_static_launch_gate"]["checks"]
    }
    readiness_checks = {check["name"]: check for check in payload["readiness"]["checks"]}
    assert payload["tenk_static_launch_gate"]["status"] == "error"
    assert static_checks["prompt_size_ceiling"]["status"] == "error"
    assert readiness_checks["tenk_static_launch_gate"]["status"] == "error"
    assert "live_adoption_certification" not in payload


def test_unattended_adopt_live_run_blocks_without_launchd_install(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "soak"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "error",
            "checks": [
                {
                    "name": "runner_lock",
                    "status": "error",
                    "detail": "output directory is currently owned by a live runner",
                }
            ],
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_watchdog_remediation_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "checks": _ok_watchdog_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.supervise.audit,
        "audit_soak",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "processed_cases": 120,
            "expected_cases": 10000,
            "degraded_rates": {
                "prompt_budget_coverage_rate": 1.0,
                "max_prompt_tokens": 2200,
            },
        },
    )
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("blocked adoption should not start supervisor")),
    )

    exit_code = unattended.main([
        "--tenk-set-and-forget",
        "--adopt-live-run",
        "--pilot-max-prompt-tokens",
        "9000",
        "--output-dir",
        str(output_dir),
        "--all-images",
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    checks = {check["name"]: check for check in payload["readiness"]["checks"]}
    static_checks = {
        check["name"]: check
        for check in payload["tenk_static_launch_gate"]["checks"]
    }
    assert payload["tenk_static_launch_gate"]["status"] == "error"
    assert static_checks["launchd_install_requested"]["status"] == "error"
    assert checks["tenk_static_launch_gate"]["status"] == "error"
    assert "live_adoption_certification" not in payload


def test_unattended_adopt_live_run_blocks_old_runner_without_restart_capability(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "soak"
    launch_agents = tmp_path / "Library" / "LaunchAgents"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "error",
            "checks": [
                {
                    "name": "runner_lock",
                    "status": "error",
                    "detail": "output directory is currently owned by a live runner",
                    "pid": 1234,
                    "runner_supports_graceful_restart": False,
                    "runner_capabilities": [],
                    "runner_capability_sources": [],
                }
            ],
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_watchdog_remediation_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "checks": _ok_watchdog_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.supervise.audit,
        "audit_soak",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "processed_cases": 120,
            "expected_cases": 10000,
            "degraded_rates": {
                "prompt_budget_coverage_rate": 1.0,
                "max_prompt_tokens": 2200,
            },
        },
    )
    monkeypatch.setattr(
        unattended,
        "install_launchd_plists",
        lambda _plan: (_ for _ in ()).throw(AssertionError("old live runner should not be adopted")),
    )

    exit_code = unattended.main([
        "--tenk-set-and-forget",
        "--install-launchd-plists",
        "--adopt-live-run",
        "--pilot-max-prompt-tokens",
        "9000",
        "--write-launchd-plist",
        str(launch_agents / "com.example.qwen-caption.plist"),
        "--write-watchdog-launchd-plist",
        str(launch_agents / "com.example.qwen-caption.watchdog.plist"),
        "--output-dir",
        str(output_dir),
        "--all-images",
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    adoption_checks = {
        check["name"]: check
        for check in payload["live_adoption_certification"]["checks"]
    }
    readiness_checks = {check["name"]: check for check in payload["readiness"]["checks"]}
    assert payload["live_adoption_certification"]["status"] == "error"
    assert adoption_checks["live_runner_restart_capability"]["status"] == "error"
    assert readiness_checks["live_adoption"]["status"] == "error"
    assert payload["readiness"]["ready_for_10k_set_and_forget"] is False


def test_unattended_adopt_live_run_blocks_unhealthy_live_audit(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "soak"
    launch_agents = tmp_path / "Library" / "LaunchAgents"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "error",
            "checks": [
                {
                    "name": "runner_lock",
                    "status": "error",
                    "detail": "output directory is currently owned by a live runner",
                }
            ],
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_watchdog_remediation_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "checks": _ok_watchdog_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.supervise.audit,
        "audit_soak",
        lambda *_args, **_kwargs: {
            "status": "error",
            "processed_cases": 10,
            "expected_cases": 10000,
            "degraded_rates": {
                "prompt_budget_coverage_rate": 0.5,
                "max_prompt_tokens": 12000,
            },
        },
    )
    monkeypatch.setattr(
        unattended,
        "install_launchd_plists",
        lambda _plan: (_ for _ in ()).throw(AssertionError("unhealthy adoption should not install launchd")),
    )

    exit_code = unattended.main([
        "--tenk-set-and-forget",
        "--install-launchd-plists",
        "--adopt-live-run",
        "--pilot-max-prompt-tokens",
        "9000",
        "--write-launchd-plist",
        str(launch_agents / "com.example.qwen-caption.plist"),
        "--write-watchdog-launchd-plist",
        str(launch_agents / "com.example.qwen-caption.watchdog.plist"),
        "--output-dir",
        str(output_dir),
        "--all-images",
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    adoption_checks = {
        check["name"]: check
        for check in payload["live_adoption_certification"]["checks"]
    }
    assert payload["live_adoption_certification"]["status"] == "error"
    assert adoption_checks["live_artifact_audit"]["status"] == "error"
    assert adoption_checks["completed_case_evidence"]["status"] == "error"
    assert adoption_checks["prompt_budget_coverage"]["status"] == "error"
    assert adoption_checks["max_prompt_tokens"]["status"] == "error"


def test_unattended_tenk_readiness_rejects_disabled_disk_reserve(tmp_path: Path) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--tenk-set-and-forget",
        "--no-preflight",
        "--min-free-gb",
        "0",
        "--output-dir",
        str(tmp_path / "soak"),
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)
    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )

    report = unattended.build_readiness_report(
        plan=plan,
        wrapper_args=wrapper_args,
        preflight_report={"status": "ok", "checks": []},
        supervisor_drill_report={
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
        watchdog_drill_report={
            "status": "ok",
            "checks": _ok_watchdog_drill_checks(),
        },
        pilot_generation_report=None,
        certification_report=None,
        launch_blocked=False,
    )

    check_by_name = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert check_by_name["disk_reserve_gate"]["status"] == "error"
    assert check_by_name["disk_reserve_gate"]["requested_min_free_gb"] == 0.0
    assert _arg_value(plan["commands"]["watchdog"], "--min-free-gb") == "0.0"
    assert _arg_value(plan["commands"]["final_audit"], "--min-free-gb") == "0.0"
    assert _arg_value(plan["commands"]["live_status"], "--min-free-gb") == "0.0"


def test_unattended_tenk_readiness_rejects_disabled_caffeinate(tmp_path: Path) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--tenk-set-and-forget",
        "--no-launchd-caffeinate",
        "--no-preflight",
        "--output-dir",
        str(tmp_path / "soak"),
    ])
    assert wrapper_args.launchd_caffeinate is False
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)
    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )

    report = unattended.build_readiness_report(
        plan=plan,
        wrapper_args=wrapper_args,
        preflight_report={"status": "ok", "checks": []},
        supervisor_drill_report={
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
        watchdog_drill_report={
            "status": "ok",
            "checks": _ok_watchdog_drill_checks(),
        },
        pilot_generation_report=None,
        certification_report=None,
        launch_blocked=False,
    )

    check_by_name = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert check_by_name["launchd_power_assertion"]["status"] == "error"


def test_readiness_rejects_ok_certification_without_runner_capability_evidence(tmp_path: Path) -> None:
    pilot_dir = tmp_path / "pilot"
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--require-readiness-ok",
        "--require-pilot-certification",
        str(pilot_dir),
        "--pilot-max-prompt-tokens",
        "9000",
        "--no-preflight",
        "--output-dir",
        str(tmp_path / "soak"),
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)
    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )

    report = unattended.build_readiness_report(
        plan=plan,
        wrapper_args=wrapper_args,
        preflight_report={"status": "ok", "checks": []},
        supervisor_drill_report={
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
        watchdog_drill_report={
            "status": "ok",
            "checks": _ok_watchdog_drill_checks(),
        },
        pilot_generation_report=None,
        certification_report={
            "status": "ok",
            "target_cases": 10000,
            "checks": [{"name": "artifact_audit", "status": "ok"}],
        },
        launch_blocked=False,
    )

    check_by_name = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert check_by_name["pilot_certification"]["status"] == "ok"
    assert check_by_name["pilot_runner_capabilities"]["status"] == "error"
    assert unattended.runner.RUNNER_CAPABILITY_GRACEFUL_RESTART in check_by_name[
        "pilot_runner_capabilities"
    ]["missing_capabilities"]
    assert unattended.runner.RUNNER_CAPABILITY_CAPTION_IO_EVENT_SUMMARY in check_by_name[
        "pilot_runner_capabilities"
    ]["missing_capabilities"]


def test_readiness_rejects_ok_certification_without_generated_pilot_evidence(tmp_path: Path) -> None:
    pilot_dir = tmp_path / "pilot"
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--require-readiness-ok",
        "--require-pilot-certification",
        str(pilot_dir),
        "--pilot-min-cases",
        "50",
        "--pilot-max-prompt-tokens",
        "9000",
        "--no-preflight",
        "--output-dir",
        str(tmp_path / "soak"),
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)
    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )
    certification_report = _ok_certification_report(pilot_dir, target_cases=10_000)
    certification_report.pop("generated_pilot_cases")
    certification_report.pop("generated_case_evidence")

    report = unattended.build_readiness_report(
        plan=plan,
        wrapper_args=wrapper_args,
        preflight_report={"status": "ok", "checks": []},
        supervisor_drill_report={
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
        watchdog_drill_report={
            "status": "ok",
            "checks": _ok_watchdog_drill_checks(),
        },
        pilot_generation_report=None,
        certification_report=certification_report,
        launch_blocked=False,
    )

    check_by_name = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert check_by_name["pilot_certification"]["status"] == "ok"
    assert check_by_name["pilot_generated_evidence"]["status"] == "error"
    assert check_by_name["pilot_generated_evidence"]["generated_pilot_cases"] == 0
    assert check_by_name["pilot_generated_evidence"]["min_pilot_cases"] == 50
    assert check_by_name["pilot_runner_capabilities"]["status"] == "ok"


def test_readiness_rejects_stale_ok_certification_without_current_progress_capabilities(tmp_path: Path) -> None:
    pilot_dir = tmp_path / "pilot"
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--require-readiness-ok",
        "--require-pilot-certification",
        str(pilot_dir),
        "--pilot-max-prompt-tokens",
        "9000",
        "--no-preflight",
        "--output-dir",
        str(tmp_path / "soak"),
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)
    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )
    certification_report = _ok_certification_report(pilot_dir, target_cases=10_000)
    stale_capabilities = [
        unattended.runner.RUNNER_CAPABILITY_GRACEFUL_RESTART,
        unattended.runner.RUNNER_CAPABILITY_PARENT_DETERMINISTIC_RECOVERY,
    ]
    certification_report["runner_capabilities"] = {
        "runner_capabilities": stale_capabilities,
        "required_capabilities": stale_capabilities,
        "missing_capabilities": [],
        "capability_sources": ["manifest"],
    }

    report = unattended.build_readiness_report(
        plan=plan,
        wrapper_args=wrapper_args,
        preflight_report={"status": "ok", "checks": []},
        supervisor_drill_report={
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
        watchdog_drill_report={
            "status": "ok",
            "checks": _ok_watchdog_drill_checks(),
        },
        pilot_generation_report=None,
        certification_report=certification_report,
        launch_blocked=False,
    )

    check_by_name = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert check_by_name["pilot_certification"]["status"] == "ok"
    assert check_by_name["pilot_runner_capabilities"]["status"] == "error"
    assert check_by_name["pilot_runner_capabilities"]["runner_capabilities"] == stale_capabilities
    assert check_by_name["pilot_runner_capabilities"]["missing_capabilities"] == [
        unattended.runner.RUNNER_CAPABILITY_CAPTION_IO_EVENT_SUMMARY,
        unattended.runner.RUNNER_CAPABILITY_WORKER_PROGRESS_HEARTBEAT,
        unattended.runner.RUNNER_CAPABILITY_ADAPTIVE_RETRY_PROFILE,
        unattended.runner.RUNNER_CAPABILITY_INSTRUCTION_QA,
    ]


def test_readiness_rejects_ok_certification_without_prompt_budget_evidence(tmp_path: Path) -> None:
    pilot_dir = tmp_path / "pilot"
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--require-readiness-ok",
        "--require-pilot-certification",
        str(pilot_dir),
        "--pilot-max-prompt-tokens",
        "9000",
        "--no-preflight",
        "--output-dir",
        str(tmp_path / "soak"),
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)
    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )
    certification_report = _ok_certification_report(pilot_dir, target_cases=10_000)
    certification_report.pop("prompt_budget")

    report = unattended.build_readiness_report(
        plan=plan,
        wrapper_args=wrapper_args,
        preflight_report={"status": "ok", "checks": []},
        supervisor_drill_report={
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
        watchdog_drill_report={
            "status": "ok",
            "checks": _ok_watchdog_drill_checks(),
        },
        pilot_generation_report=None,
        certification_report=certification_report,
        launch_blocked=False,
    )

    check_by_name = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert check_by_name["pilot_runner_capabilities"]["status"] == "ok"
    assert check_by_name["pilot_prompt_budget_evidence"]["status"] == "error"
    assert check_by_name["pilot_prompt_budget_evidence"]["rows_with_prompt_budget"] == 0


def test_readiness_rejects_ok_certification_without_qwen_caption_io_source_evidence(
    tmp_path: Path,
) -> None:
    pilot_dir = tmp_path / "pilot"
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--require-readiness-ok",
        "--require-pilot-certification",
        str(pilot_dir),
        "--pilot-max-prompt-tokens",
        "9000",
        "--no-preflight",
        "--output-dir",
        str(tmp_path / "soak"),
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)
    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )
    certification_report = _ok_certification_report(pilot_dir, target_cases=10_000)
    certification_report.pop("qwen_caption_io_sources")

    report = unattended.build_readiness_report(
        plan=plan,
        wrapper_args=wrapper_args,
        preflight_report={"status": "ok", "checks": []},
        supervisor_drill_report={
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
        watchdog_drill_report={
            "status": "ok",
            "checks": _ok_watchdog_drill_checks(),
        },
        pilot_generation_report=None,
        certification_report=certification_report,
        launch_blocked=False,
    )

    check_by_name = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert check_by_name["pilot_prompt_budget_evidence"]["status"] == "ok"
    assert check_by_name["pilot_qwen_caption_io_source"]["status"] == "error"
    assert check_by_name["pilot_qwen_caption_io_source"]["has_source_report"] is False


def test_readiness_rejects_unbound_latest_qwen_caption_io_source_evidence(
    tmp_path: Path,
) -> None:
    pilot_dir = tmp_path / "pilot"
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--require-readiness-ok",
        "--require-pilot-certification",
        str(pilot_dir),
        "--pilot-max-prompt-tokens",
        "9000",
        "--no-preflight",
        "--output-dir",
        str(tmp_path / "soak"),
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)
    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )
    certification_report = _ok_certification_report(pilot_dir, target_cases=10_000)
    certification_report["qwen_caption_io_sources"] = {
        "required_rows": 100,
        "runtime_prompt_budget_rows": 100,
        "valid_runtime_prompt_budget_rows": 100,
        "invalid_runtime_rows_count": 0,
        "missing_runtime_rows_count": 0,
        "source_counts": {"qwen_caption_io_latest": 100},
        "accepted_sources": ["qwen_caption_io_per_run"],
    }

    report = unattended.build_readiness_report(
        plan=plan,
        wrapper_args=wrapper_args,
        preflight_report={"status": "ok", "checks": []},
        supervisor_drill_report={
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
        watchdog_drill_report={
            "status": "ok",
            "checks": _ok_watchdog_drill_checks(),
        },
        pilot_generation_report=None,
        certification_report=certification_report,
        launch_blocked=False,
    )

    check_by_name = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert check_by_name["pilot_prompt_budget_evidence"]["status"] == "ok"
    assert check_by_name["pilot_qwen_caption_io_source"]["status"] == "error"
    assert check_by_name["pilot_qwen_caption_io_source"]["unsupported_sources"] == [
        "qwen_caption_io_latest"
    ]
    assert check_by_name["pilot_qwen_caption_io_source"]["accepted_observed_sources"] == []


def test_readiness_rejects_ok_certification_above_prompt_size_gate(tmp_path: Path) -> None:
    pilot_dir = tmp_path / "pilot"
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--require-readiness-ok",
        "--require-pilot-certification",
        str(pilot_dir),
        "--pilot-max-prompt-tokens",
        "9000",
        "--no-preflight",
        "--output-dir",
        str(tmp_path / "soak"),
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)
    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )
    certification_report = _ok_certification_report(pilot_dir, target_cases=10_000)
    certification_report["prompt_budget"] = {
        "pilot_cases": 100,
        "rows_with_prompt_budget": 100,
        "adapted_cases": 0,
        "adapted_case_rate": 0.0,
        "max_prompt_tokens": 9500,
    }

    report = unattended.build_readiness_report(
        plan=plan,
        wrapper_args=wrapper_args,
        preflight_report={"status": "ok", "checks": []},
        supervisor_drill_report={
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
        watchdog_drill_report={
            "status": "ok",
            "checks": _ok_watchdog_drill_checks(),
        },
        pilot_generation_report=None,
        certification_report=certification_report,
        launch_blocked=False,
    )

    check_by_name = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert check_by_name["pilot_prompt_budget_evidence"]["status"] == "error"
    assert check_by_name["pilot_prompt_budget_evidence"]["max_prompt_tokens"] == 9500
    assert check_by_name["pilot_prompt_budget_evidence"]["limit"] == 9000


def test_unattended_strict_readiness_allows_clean_10k_plan(monkeypatch, tmp_path: Path) -> None:
    pilot_dir = tmp_path / "pilot"
    output_dir = tmp_path / "soak"
    launch_agents_dir = tmp_path / "Library" / "LaunchAgents"
    plist_path = launch_agents_dir / "com.example.qwen-caption.plist"
    watchdog_plist_path = launch_agents_dir / "com.example.qwen-caption-watchdog.plist"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "ok",
            "checks": [
                {"name": "artifact_directory", "status": "ok"},
                {"name": "disk_budget", "status": "ok", "detail": "disk is fine"},
            ],
            "model_cache": {
                "models": [{"role": "caption", "model_id": "example/model", "needs_download": False}],
            },
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
    )

    def fake_certify(pilot_output_dir, **kwargs):
        return _ok_certification_report(pilot_output_dir, target_cases=kwargs["target_cases"])

    monkeypatch.setattr(unattended.certify, "certify_soak", fake_certify)
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dry-run should not start supervisor")),
    )

    exit_code = unattended.main([
        "--dry-run",
        "--require-readiness-ok",
        "--require-pilot-certification",
        str(pilot_dir),
        "--pilot-max-prompt-tokens",
        "9000",
        "--install-launchd-plists",
        "--write-launchd-plist",
        str(plist_path),
        "--write-watchdog-launchd-plist",
        str(watchdog_plist_path),
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == 0
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    readiness = payload["readiness"]
    assert readiness["status"] == "ok"
    assert readiness["ready_for_10k_set_and_forget"] is True
    assert readiness["summary"]["tenk_set_and_forget"] is False
    assert readiness["summary"]["pilot_certification_required"] is True
    assert readiness["summary"]["launchd_plist_requested"] is True
    assert readiness["summary"]["watchdog_launchd_plist_requested"] is True
    assert readiness["summary"]["launchd_install_requested"] is True
    check_by_name = {check["name"]: check for check in readiness["checks"]}
    assert check_by_name["launchd_persistence"]["status"] == "ok"
    assert check_by_name["watchdog_remediation"]["status"] == "ok"
    assert check_by_name["watchdog_status_artifacts"]["status"] == "ok"
    assert check_by_name["disk_reserve_gate"]["status"] == "ok"
    assert check_by_name["disk_reserve_gate"]["requested_min_free_gb"] == 5.0
    assert "--strict-set-and-forget" in payload["commands"]["operational_audit"]
    watchdog_command = payload["commands"]["watchdog"]
    assert _arg_value(watchdog_command, "--remediate-launchd-label") == "com.tator.qwen-caption-soak"
    assert _arg_value(watchdog_command, "--remediate-launchd-domain") == f"gui/{unattended.os.getuid()}"
    assert _arg_value(watchdog_command, "--remediate-launchd-plist") == str(plist_path.resolve(strict=False))
    assert _arg_value(watchdog_command, "--max-remediations") == "25"
    assert json.loads((output_dir / unattended.READINESS_NAME).read_text()) == readiness
    assert plist_path.exists()
    assert watchdog_plist_path.exists()
    watchdog_plist = plistlib.loads(watchdog_plist_path.read_bytes())
    assert watchdog_plist["ProgramArguments"] == watchdog_command


def test_unattended_strict_readiness_rejects_disabled_watchdog_remediation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pilot_dir = tmp_path / "pilot"
    output_dir = tmp_path / "soak"
    launch_agents_dir = tmp_path / "Library" / "LaunchAgents"
    plist_path = launch_agents_dir / "com.example.qwen-caption.plist"
    watchdog_plist_path = launch_agents_dir / "com.example.qwen-caption-watchdog.plist"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "ok",
            "checks": [
                {"name": "artifact_directory", "status": "ok"},
                {"name": "disk_budget", "status": "ok", "detail": "disk is fine"},
            ],
            "model_cache": {
                "models": [{"role": "caption", "model_id": "example/model", "needs_download": False}],
            },
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.certify,
        "certify_soak",
        lambda pilot_output_dir, **kwargs: _ok_certification_report(
            pilot_output_dir,
            target_cases=kwargs["target_cases"],
        ),
    )
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dry-run should not start supervisor")),
    )

    exit_code = unattended.main([
        "--dry-run",
        "--require-readiness-ok",
        "--require-pilot-certification",
        str(pilot_dir),
        "--pilot-max-prompt-tokens",
        "9000",
        "--install-launchd-plists",
        "--watchdog-max-remediations",
        "0",
        "--write-launchd-plist",
        str(plist_path),
        "--write-watchdog-launchd-plist",
        str(watchdog_plist_path),
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    readiness = payload["readiness"]
    check_by_name = {check["name"]: check for check in readiness["checks"]}
    assert readiness["status"] == "warn"
    assert check_by_name["watchdog_remediation"]["status"] == "warn"
    assert "--remediate-launchd-label" not in payload["commands"]["watchdog"]


def test_unattended_readiness_rejects_installed_plists_outside_launchagents(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pilot_dir = tmp_path / "pilot"
    output_dir = tmp_path / "soak"
    plist_path = tmp_path / "run-local-supervisor.plist"
    watchdog_plist_path = tmp_path / "run-local-watchdog.plist"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "ok",
            "checks": [
                {"name": "artifact_directory", "status": "ok"},
                {"name": "disk_budget", "status": "ok", "detail": "disk is fine"},
            ],
            "model_cache": {
                "models": [{"role": "caption", "model_id": "example/model", "needs_download": False}],
            },
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.certify,
        "certify_soak",
        lambda pilot_output_dir, **kwargs: _ok_certification_report(
            pilot_output_dir,
            target_cases=kwargs["target_cases"],
        ),
    )
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dry-run should not start supervisor")),
    )

    exit_code = unattended.main([
        "--dry-run",
        "--require-readiness-ok",
        "--require-pilot-certification",
        str(pilot_dir),
        "--pilot-max-prompt-tokens",
        "9000",
        "--install-launchd-plists",
        "--write-launchd-plist",
        str(plist_path),
        "--write-watchdog-launchd-plist",
        str(watchdog_plist_path),
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    readiness = payload["readiness"]
    assert readiness["status"] == "error"
    check_by_name = {check["name"]: check for check in readiness["checks"]}
    assert check_by_name["launchd_install"]["status"] == "ok"
    assert check_by_name["launchd_persistence"]["status"] == "error"
    assert {item["role"] for item in check_by_name["launchd_persistence"]["non_persistent_roles"]} == {
        "supervisor",
        "watchdog",
    }


def test_unattended_tenk_mode_allows_clean_10k_plan(monkeypatch, tmp_path: Path) -> None:
    pilot_dir = tmp_path / "pilot"
    output_dir = tmp_path / "soak"
    launch_agents_dir = tmp_path / "Library" / "LaunchAgents"
    plist_path = launch_agents_dir / "com.example.qwen-caption.plist"
    watchdog_plist_path = launch_agents_dir / "com.example.qwen-caption-watchdog.plist"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "ok",
            "checks": [
                {"name": "artifact_directory", "status": "ok"},
                {"name": "disk_budget", "status": "ok", "detail": "disk is fine"},
            ],
            "model_cache": {
                "models": [{"role": "caption", "model_id": "example/model", "needs_download": False}],
            },
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.certify,
        "certify_soak",
        lambda pilot_output_dir, **kwargs: _ok_certification_report(
            pilot_output_dir,
            target_cases=kwargs["target_cases"],
        ),
    )
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dry-run should not start supervisor")),
    )

    exit_code = unattended.main([
        "--dry-run",
        "--tenk-set-and-forget",
        "--require-pilot-certification",
        str(pilot_dir),
        "--pilot-max-prompt-tokens",
        "9000",
        "--install-launchd-plists",
        "--write-launchd-plist",
        str(plist_path),
        "--write-watchdog-launchd-plist",
        str(watchdog_plist_path),
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == 0
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    readiness = payload["readiness"]
    assert payload["set_and_forget_gate"]["tenk_mode"] is True
    assert readiness["status"] == "ok"
    assert readiness["ready_for_10k_set_and_forget"] is True
    assert readiness["tenk_set_and_forget"] is True
    assert readiness["summary"]["tenk_set_and_forget"] is True
    assert "--set-and-forget" in payload["commands"]["supervisor"]
    assert readiness["summary"]["pilot_certification_required"] is True
    assert readiness["summary"]["launchd_plist_requested"] is True
    assert readiness["summary"]["watchdog_launchd_plist_requested"] is True
    assert readiness["summary"]["launchd_install_requested"] is True
    check_by_name = {check["name"]: check for check in readiness["checks"]}
    assert check_by_name["launchd_persistence"]["status"] == "ok"
    assert check_by_name["watchdog_remediation"]["status"] == "ok"
    assert check_by_name["launchd_power_assertion"]["status"] == "ok"
    assert payload["launchd_power_assertion"]["enabled"] is True
    assert "--strict-set-and-forget" in payload["commands"]["operational_audit"]
    assert _arg_value(payload["commands"]["watchdog"], "--remediate-launchd-label") == "com.tator.qwen-caption-soak"
    assert _arg_value(payload["commands"]["watchdog"], "--remediate-launchd-plist") == str(
        plist_path.resolve(strict=False)
    )
    assert plist_path.exists()
    assert watchdog_plist_path.exists()
    supervisor_plist = plistlib.loads(plist_path.read_bytes())
    watchdog_plist = plistlib.loads(watchdog_plist_path.read_bytes())
    expected_prefix = [unattended.CAFFEINATE_BIN, *unattended.CAFFEINATE_ARGS]
    assert supervisor_plist["ProgramArguments"][: len(expected_prefix)] == expected_prefix
    assert supervisor_plist["ProgramArguments"][len(expected_prefix) :] == payload["commands"]["supervisor"]
    assert watchdog_plist["ProgramArguments"][: len(expected_prefix)] == expected_prefix
    assert watchdog_plist["ProgramArguments"][len(expected_prefix) :] == payload["commands"]["watchdog"]


def test_unattended_tenk_mode_auto_generates_pilot_before_launchd_handoff(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "soak"
    auto_pilot_dir = output_dir / unattended.DEFAULT_TENK_PILOT_SUBDIR
    launch_agents_dir = tmp_path / "Library" / "LaunchAgents"
    plist_path = launch_agents_dir / "com.example.qwen-caption.plist"
    watchdog_plist_path = launch_agents_dir / "com.example.qwen-caption-watchdog.plist"
    supervised_dirs: list[Path] = []

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "ok",
            "checks": [
                {"name": "artifact_directory", "status": "ok"},
                {"name": "disk_budget", "status": "ok", "detail": "disk is fine"},
            ],
            "model_cache": {
                "models": [{"role": "caption", "model_id": "example/model", "needs_download": False}],
            },
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_watchdog_remediation_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "checks": _ok_watchdog_drill_checks(),
        },
    )

    def fake_supervise(supervisor_args, **_kwargs):
        assert supervisor_args.sample_size == unattended.DEFAULT_TENK_MIN_PILOT_CASES
        supervised_dirs.append(supervisor_args.output_dir)
        return supervise.TERMINAL_SUCCESS

    def fake_certify(pilot_output_dir, **kwargs):
        assert pilot_output_dir == auto_pilot_dir.resolve(strict=False)
        assert kwargs["max_prompt_tokens"] == unattended.DEFAULT_TENK_PILOT_MAX_PROMPT_TOKENS
        assert kwargs["min_pilot_cases"] == unattended.DEFAULT_TENK_MIN_PILOT_CASES
        return _ok_certification_report(pilot_output_dir, target_cases=kwargs["target_cases"])

    monkeypatch.setattr(unattended.supervise, "supervise_soak", fake_supervise)
    monkeypatch.setattr(unattended.certify, "certify_soak", fake_certify)
    monkeypatch.setattr(
        unattended,
        "install_launchd_plists",
        lambda _plan: {
            "status": "ok",
            "requested": True,
            "detail": "supervisor and watchdog LaunchAgents were bootstrapped and verified",
            "steps": [],
        },
    )
    monkeypatch.setattr(
        unattended,
        "run_post_install_operation_audit",
        lambda **_kwargs: _ok_post_install_operation_audit_report(),
    )

    exit_code = unattended.main([
        "--tenk-set-and-forget",
        "--install-launchd-plists",
        "--write-launchd-plist",
        str(plist_path),
        "--write-watchdog-launchd-plist",
        str(watchdog_plist_path),
        "--output-dir",
        str(output_dir),
        "--all-images",
    ])

    assert exit_code == supervise.TERMINAL_SUCCESS
    assert supervised_dirs == [auto_pilot_dir.resolve(strict=False)]
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    checks = {check["name"]: check for check in payload["readiness"]["checks"]}
    static_checks = {
        check["name"]: check
        for check in payload["tenk_static_launch_gate"]["checks"]
    }
    assert payload["set_and_forget_gate"]["auto_pilot_generation"] is True
    assert payload["set_and_forget_gate"]["auto_prompt_size_ceiling"] is True
    assert payload["set_and_forget_gate"]["auto_pilot_min_cases"] is True
    assert payload["set_and_forget_gate"]["auto_pilot_sample_size"] is True
    assert payload["pilot_generation_gate"]["auto_configured"] is True
    assert payload["pilot_generation_gate"]["auto_sample_size"] is True
    assert payload["pilot_generation_gate"]["sample_size"] == unattended.DEFAULT_TENK_MIN_PILOT_CASES
    assert payload["pilot_generation_gate"]["output_dir"] == str(auto_pilot_dir.resolve(strict=False))
    assert payload["certification_gate"]["max_prompt_tokens"] == unattended.DEFAULT_TENK_PILOT_MAX_PROMPT_TOKENS
    assert payload["certification_gate"]["min_pilot_cases"] == unattended.DEFAULT_TENK_MIN_PILOT_CASES
    assert payload["created_pilot_generation"]["status"] == "ok"
    assert payload["required_pilot_certification"]["status"] == "ok"
    assert payload["tenk_static_launch_gate"]["status"] == "ok"
    assert static_checks["pilot_or_live_adoption_evidence"]["status"] == "ok"
    assert static_checks["prompt_size_ceiling"]["status"] == "ok"
    assert checks["pilot_generation"]["status"] == "ok"
    assert checks["pilot_certification"]["status"] == "ok"
    assert checks["launchd_install_result"]["status"] == "ok"
    assert checks["post_install_operation_audit"]["status"] == "ok"
    assert payload["post_install_operation_audit"]["status"] == "ok"
    assert payload["readiness"]["ready_for_10k_set_and_forget"] is True
    assert plist_path.exists()
    assert watchdog_plist_path.exists()


def test_unattended_tenk_rejects_skipped_post_install_operation_audit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "soak"
    launch_agents = tmp_path / "Library" / "LaunchAgents"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "ok",
            "checks": [
                {"name": "artifact_directory", "status": "ok"},
                {"name": "disk_budget", "status": "ok", "detail": "disk is fine"},
            ],
            "model_cache": {
                "models": [{"role": "caption", "model_id": "example/model", "needs_download": False}],
            },
        },
    )
    monkeypatch.setattr(
        unattended,
        "install_launchd_plists",
        lambda _plan: (_ for _ in ()).throw(AssertionError("static gate should block launchd install")),
    )

    exit_code = unattended.main([
        "--dry-run",
        "--tenk-set-and-forget",
        "--install-launchd-plists",
        "--skip-post-install-operation-audit",
        "--write-launchd-plist",
        str(launch_agents / "com.example.qwen-caption.plist"),
        "--write-watchdog-launchd-plist",
        str(launch_agents / "com.example.qwen-caption.watchdog.plist"),
        "--output-dir",
        str(output_dir),
        "--all-images",
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    static_checks = {
        check["name"]: check
        for check in payload["tenk_static_launch_gate"]["checks"]
    }
    readiness_checks = {check["name"]: check for check in payload["readiness"]["checks"]}
    assert payload["tenk_static_launch_gate"]["status"] == "error"
    assert static_checks["post_install_operation_audit_enabled"]["status"] == "error"
    assert static_checks["post_install_operation_audit_timeout"]["status"] == "ok"
    assert readiness_checks["tenk_static_launch_gate"]["status"] == "error"
    assert "supervisor_drill" not in payload
    assert "created_pilot_generation" not in payload
    assert "required_pilot_certification" not in payload


def test_unattended_tenk_launchd_handoff_blocks_failed_post_install_operation_audit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "soak"
    auto_pilot_dir = output_dir / unattended.DEFAULT_TENK_PILOT_SUBDIR
    launch_agents = tmp_path / "Library" / "LaunchAgents"
    plist_path = launch_agents / "com.example.qwen-caption.plist"
    watchdog_plist_path = launch_agents / "com.example.qwen-caption.watchdog.plist"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "ok",
            "checks": [
                {"name": "artifact_directory", "status": "ok"},
                {"name": "disk_budget", "status": "ok", "detail": "disk is fine"},
            ],
            "model_cache": {
                "models": [{"role": "caption", "model_id": "example/model", "needs_download": False}],
            },
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_watchdog_remediation_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "checks": _ok_watchdog_drill_checks(),
        },
    )

    def fake_supervise(supervisor_args, **_kwargs):
        assert supervisor_args.output_dir == auto_pilot_dir.resolve(strict=False)
        return supervise.TERMINAL_SUCCESS

    monkeypatch.setattr(unattended.supervise, "supervise_soak", fake_supervise)
    monkeypatch.setattr(
        unattended.certify,
        "certify_soak",
        lambda pilot_output_dir, **kwargs: _ok_certification_report(
            pilot_output_dir,
            target_cases=kwargs["target_cases"],
        ),
    )
    monkeypatch.setattr(
        unattended,
        "install_launchd_plists",
        lambda _plan: {
            "status": "ok",
            "requested": True,
            "detail": "supervisor and watchdog LaunchAgents were bootstrapped and verified",
            "steps": [],
        },
    )
    monkeypatch.setattr(
        unattended,
        "run_post_install_operation_audit",
        lambda **_kwargs: {
            "schema_version": 1,
            "status": "error",
            "strict_set_and_forget": True,
            "checks": [
                {
                    "name": "watchdog_latest",
                    "status": "error",
                    "detail": "watchdog latest status is missing",
                }
            ],
            "post_install_poll": {
                "status": "timeout",
                "attempts": 3,
                "timeout_seconds": 300.0,
                "interval_seconds": 5.0,
            },
        },
    )

    exit_code = unattended.main([
        "--tenk-set-and-forget",
        "--install-launchd-plists",
        "--write-launchd-plist",
        str(plist_path),
        "--write-watchdog-launchd-plist",
        str(watchdog_plist_path),
        "--output-dir",
        str(output_dir),
        "--all-images",
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    readiness = payload["readiness"]
    checks = {check["name"]: check for check in readiness["checks"]}
    assert payload["launchd_install_result"]["status"] == "ok"
    assert payload["post_install_operation_audit"]["status"] == "error"
    assert checks["launchd_install_result"]["status"] == "ok"
    assert checks["post_install_operation_audit"]["status"] == "error"
    assert checks["post_install_operation_audit"]["poll"]["status"] == "timeout"
    assert readiness["ready_for_10k_set_and_forget"] is False
    assert json.loads((output_dir / unattended.READINESS_NAME).read_text()) == readiness


def test_unattended_non_dry_run_installs_launchd_and_does_not_start_foreground_supervisor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "soak"
    launch_agents_dir = tmp_path / "Library" / "LaunchAgents"
    plist_path = launch_agents_dir / "com.example.qwen-caption.plist"
    watchdog_plist_path = launch_agents_dir / "com.example.qwen-caption-watchdog.plist"
    launchctl_commands: list[list[str]] = []

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {
            "status": "ok",
            "checks": [
                {"name": "artifact_directory", "status": "ok"},
                {"name": "disk_budget", "status": "ok", "detail": "disk is fine"},
            ],
            "model_cache": {
                "models": [{"role": "caption", "model_id": "example/model", "needs_download": False}],
            },
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("launchd install should own startup")),
    )

    def fake_launchctl(command, *, timeout_seconds=30.0):
        launchctl_commands.append([str(part) for part in command])
        return unattended.subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(unattended, "_run_launchctl_command", fake_launchctl)

    exit_code = unattended.main([
        "--install-launchd-plists",
        "--launchctl-domain",
        "gui/501",
        "--write-launchd-plist",
        str(plist_path),
        "--write-watchdog-launchd-plist",
        str(watchdog_plist_path),
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == 0
    assert plist_path.exists()
    assert watchdog_plist_path.exists()
    assert [command[1] for command in launchctl_commands] == [
        "bootout",
        "bootout",
        "bootstrap",
        "print",
        "bootout",
        "bootout",
        "bootstrap",
        "print",
    ]
    assert launchctl_commands[0][2] == "gui/501/com.tator.qwen-caption-soak.watchdog"
    assert launchctl_commands[3][2] == "gui/501/com.tator.qwen-caption-soak.watchdog"
    assert launchctl_commands[4][2] == "gui/501/com.tator.qwen-caption-soak"
    assert launchctl_commands[7][2] == "gui/501/com.tator.qwen-caption-soak"
    assert all(
        command[2] == "gui/501"
        or command[2].startswith("gui/501/com.tator.qwen-caption-soak")
        for command in launchctl_commands
    )
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    install_result = payload["launchd_install_result"]
    assert install_result["status"] == "ok"
    assert len(install_result["steps"]) == 8
    assert install_result["timeout_seconds"] == 30.0
    readiness = json.loads((output_dir / unattended.READINESS_NAME).read_text())
    assert readiness["launchd_install_result"]["status"] == "ok"


def test_launchd_install_times_out_hung_launchctl_step(tmp_path: Path, monkeypatch) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--install-launchd-plists",
        "--launchctl-domain",
        "gui/501",
        "--launchctl-timeout",
        "2",
        "--write-launchd-plist",
        str(tmp_path / "com.example.qwen-caption.plist"),
        "--output-dir",
        str(tmp_path / "soak"),
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)
    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )

    def fake_launchctl(command, *, timeout_seconds=30.0):
        if "bootstrap" in command:
            raise unattended.subprocess.TimeoutExpired(command, timeout_seconds)
        return unattended.subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(unattended, "_run_launchctl_command", fake_launchctl)

    report = unattended.install_launchd_plists(plan)

    assert report["status"] == "error"
    assert report["timeout_seconds"] == 2.0
    assert report["steps"][2]["name"] == "bootstrap"
    assert report["steps"][2]["error_type"] == "TimeoutExpired"
    assert report["steps"][2]["timeout_seconds"] == 2.0


def test_readiness_rejects_drill_report_without_signal_abort_evidence(tmp_path: Path) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--no-preflight",
        "--output-dir",
        str(tmp_path / "soak"),
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)
    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )

    report = unattended.build_readiness_report(
        plan=plan,
        wrapper_args=wrapper_args,
        preflight_report={"status": "ok", "checks": []},
        supervisor_drill_report={
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": {
                "supervisor_success": True,
                "saw_nonzero_exit": True,
                "saw_stale_heartbeat": True,
                "saw_missing_heartbeat": True,
                "caption_io_retention_ok": True,
                "summary_totals_cover_all_cases": True,
                "summary_snapshot_bounded": True,
                "summary_truncated_when_over_limit": True,
                "summary_omits_rows_when_over_limit": True,
            },
        },
        pilot_generation_report=None,
        certification_report=None,
        launch_blocked=False,
    )

    check_by_name = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert check_by_name["supervisor_drill"]["status"] == "error"
    assert check_by_name["supervisor_drill"]["missing_checks"] == ["saw_signal_exit"]


def test_readiness_rejects_runbook_without_live_status_command(tmp_path: Path) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--no-preflight",
        "--output-dir",
        str(tmp_path / "soak"),
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)
    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )
    plan["commands"].pop("live_status")
    plan["commands"].pop("live_status_shell")

    report = unattended.build_readiness_report(
        plan=plan,
        wrapper_args=wrapper_args,
        preflight_report={"status": "ok", "checks": []},
        supervisor_drill_report={
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
        pilot_generation_report=None,
        certification_report=None,
        launch_blocked=False,
    )

    check_by_name = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert check_by_name["recovery_commands"]["status"] == "error"
    assert check_by_name["recovery_commands"]["missing_commands"] == ["live_status"]


def test_readiness_rejects_runbook_without_operational_audit_command(tmp_path: Path) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--no-preflight",
        "--output-dir",
        str(tmp_path / "soak"),
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)
    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )
    plan["commands"].pop("operational_audit")
    plan["commands"].pop("operational_audit_shell")

    report = unattended.build_readiness_report(
        plan=plan,
        wrapper_args=wrapper_args,
        preflight_report={"status": "ok", "checks": []},
        supervisor_drill_report={
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
        pilot_generation_report=None,
        certification_report=None,
        launch_blocked=False,
    )

    check_by_name = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert check_by_name["recovery_commands"]["status"] == "error"
    assert check_by_name["recovery_commands"]["missing_commands"] == ["operational_audit"]


def test_readiness_rejects_watchdog_without_restart_stable_state(tmp_path: Path) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--no-preflight",
        "--output-dir",
        str(tmp_path / "soak"),
    ])
    supervisor_args, supervisor_extra_args = unattended.parse_supervisor_args(supervisor_argv)
    plan = unattended.build_unattended_plan(
        wrapper_args=wrapper_args,
        supervisor_args=supervisor_args,
        supervisor_extra_args=supervisor_extra_args,
        supervisor_argv=supervisor_argv,
    )
    watchdog_command = plan["commands"]["watchdog"]
    state_index = watchdog_command.index("--state-json")
    del watchdog_command[state_index : state_index + 2]
    plan["paths"].pop("watchdog_state_json")

    report = unattended.build_readiness_report(
        plan=plan,
        wrapper_args=wrapper_args,
        preflight_report={"status": "ok", "checks": []},
        supervisor_drill_report={
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
        pilot_generation_report=None,
        certification_report=None,
        launch_blocked=False,
    )

    check_by_name = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert check_by_name["watchdog_status_artifacts"]["status"] == "error"
    assert check_by_name["watchdog_status_artifacts"]["missing_paths"] == ["watchdog_state_json"]
    assert check_by_name["watchdog_status_artifacts"]["missing_flags"] == ["--state-json"]


def test_unattended_runs_supervisor_drill_before_launch(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "soak"
    captured = {}

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {"status": "ok", "checks": [{"name": "artifact_directory", "status": "ok"}]},
    )

    def fake_run_endurance_drill(drill_output_dir, **kwargs):
        captured["drill_output_dir"] = drill_output_dir
        captured["drill_kwargs"] = kwargs
        return {
            "status": "ok",
            "output_dir": str(drill_output_dir),
            "report_json": str(kwargs["write_json"]),
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        }

    monkeypatch.setattr(unattended.soak_drill, "run_endurance_drill", fake_run_endurance_drill)
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: supervise.TERMINAL_SUCCESS,
    )

    exit_code = unattended.main([
        "--dry-run",
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == 0
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    assert payload["supervisor_drill"]["status"] == "ok"
    assert payload["supervisor_drill_gate"]["required"] is True
    assert captured["drill_output_dir"] == output_dir.resolve(strict=False) / "supervisor_drill"
    assert captured["drill_kwargs"]["force"] is True
    assert captured["drill_kwargs"]["case_count"] == unattended.SET_AND_FORGET_DRILL_CASES
    assert captured["drill_kwargs"]["chunk_size"] == unattended.DEFAULT_SUPERVISOR_DRILL_CHUNK_SIZE
    assert captured["drill_kwargs"]["write_json"] == output_dir.resolve(strict=False) / "supervisor_drill" / "drill_report.json"


def test_unattended_runs_watchdog_drill_before_launch(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "soak"
    captured = {}

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {"status": "ok", "checks": [{"name": "artifact_directory", "status": "ok"}]},
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda drill_output_dir, **kwargs: {
            "status": "ok",
            "output_dir": str(drill_output_dir),
            "report_json": str(kwargs["write_json"]),
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
    )

    def fake_run_watchdog_remediation_drill(drill_output_dir, **kwargs):
        captured["watchdog_drill_output_dir"] = drill_output_dir
        captured["watchdog_drill_kwargs"] = kwargs
        return {
            "status": "ok",
            "output_dir": str(drill_output_dir),
            "report_json": str(kwargs["write_json"]),
            "checks": _ok_watchdog_drill_checks(),
        }

    monkeypatch.setattr(
        unattended.soak_drill,
        "run_watchdog_remediation_drill",
        fake_run_watchdog_remediation_drill,
    )
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: supervise.TERMINAL_SUCCESS,
    )

    exit_code = unattended.main([
        "--dry-run",
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == 0
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    assert payload["watchdog_drill"]["status"] == "ok"
    assert payload["watchdog_drill_gate"]["required"] is True
    assert captured["watchdog_drill_output_dir"] == output_dir.resolve(strict=False) / "watchdog_drill"
    assert captured["watchdog_drill_kwargs"]["force"] is True
    assert captured["watchdog_drill_kwargs"]["write_json"] == (
        output_dir.resolve(strict=False) / "watchdog_drill" / "watchdog_drill_report.json"
    )


def test_unattended_blocks_when_supervisor_drill_fails(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "soak"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {"status": "ok", "checks": [{"name": "artifact_directory", "status": "ok"}]},
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: {"status": "error", "detail": "scripted drill failure"},
    )
    monkeypatch.setattr(
        unattended.certify,
        "certify_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("certification should not run")),
    )
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("supervisor should not start")),
    )

    exit_code = unattended.main([
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    assert payload["supervisor_drill"]["status"] == "error"
    assert payload["supervisor_drill"]["detail"] == "scripted drill failure"


def test_unattended_blocks_when_watchdog_drill_fails(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "soak"

    monkeypatch.setattr(
        unattended.preflight,
        "preflight_soak",
        lambda _args: {"status": "ok", "checks": [{"name": "artifact_directory", "status": "ok"}]},
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_endurance_drill",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "case_count": unattended.SET_AND_FORGET_DRILL_CASES,
            "checks": _ok_drill_checks(),
        },
    )
    monkeypatch.setattr(
        unattended.soak_drill,
        "run_watchdog_remediation_drill",
        lambda *_args, **_kwargs: {"status": "error", "detail": "scripted watchdog drill failure"},
    )
    monkeypatch.setattr(
        unattended.certify,
        "certify_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("certification should not run")),
    )
    monkeypatch.setattr(
        unattended.supervise,
        "supervise_soak",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("supervisor should not start")),
    )

    exit_code = unattended.main([
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == supervise.TERMINAL_PRECHECK_FAILED
    payload = json.loads((output_dir / unattended.RUNBOOK_NAME).read_text())
    assert payload["watchdog_drill"]["status"] == "error"
    assert payload["watchdog_drill"]["detail"] == "scripted watchdog drill failure"


def test_unattended_skip_supervisor_drill_records_skipped_report(tmp_path: Path) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--skip-supervisor-drill",
        "--output-dir",
        str(tmp_path / "soak"),
    ])
    supervisor_args, _extra = unattended.parse_supervisor_args(supervisor_argv)

    report = unattended._run_supervisor_drill(
        wrapper_args=wrapper_args,
        output_dir=supervisor_args.output_dir.resolve(strict=False),
    )

    assert report["status"] == "skipped"
    assert "skip-supervisor-drill" in report["detail"]
    assert Path(report["report_json"]).exists()


def test_unattended_skip_watchdog_drill_records_skipped_report(tmp_path: Path) -> None:
    wrapper_args, supervisor_argv = unattended.parse_wrapper_args([
        "--skip-watchdog-drill",
        "--output-dir",
        str(tmp_path / "soak"),
    ])
    supervisor_args, _extra = unattended.parse_supervisor_args(supervisor_argv)

    report = unattended._run_watchdog_drill(
        wrapper_args=wrapper_args,
        output_dir=supervisor_args.output_dir.resolve(strict=False),
    )

    assert report["status"] == "skipped"
    assert "skip-watchdog-drill" in report["detail"]
    assert Path(report["report_json"]).exists()
