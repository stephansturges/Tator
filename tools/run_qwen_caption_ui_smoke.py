#!/usr/bin/env python3
"""Rendered smoke check for the Qwen captioning panel.

This intentionally checks the real browser surface instead of only static HTML.
It is meant to catch missing controls, wrong defaults, hardcoded API origins,
and console/network errors before loading a large caption dataset for testing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_OUT_JSON = REPO_ROOT / "tmp" / "qwen_caption_ui_smoke_report.json"
DEFAULT_SCREENSHOT = REPO_ROOT / "tmp" / "qwen_caption_ui_smoke.png"

CRITICAL_CONTROLS = (
    "qwenCaptionDetails",
    "qwenCaptionDataset",
    "qwenCaptionDatasetRefresh",
    "qwenCaptionRunButton",
    "qwenCaptionBatchRun",
    "qwenCaptionBatchRunAll",
    "qwenCaptionSetAndForget",
    "qwenCaptionMaxAutoResumes",
    "qwenCaptionSaveText",
    "qwenCaptionGeneratedPrimary",
    "qwenCaptionAlternates",
    "qwenCaptionArchiveStatus",
    "qwenCaptionSaveAlternate",
    "qwenCaptionUpdateSelected",
    "qwenCaptionSetPrimary",
    "qwenCaptionDeleteSelected",
    "qwenCaptionDownloadJsonl",
    "qwenCaptionDownloadGroupedJson",
    "qwenCaptionDownloadVlmJsonl",
    "qwenCaptionSubcaptionsPerImage",
    "qwenCaptionQaMix",
    "qwenCaptionAnswerFormat",
    "qwenCaptionIncludeCaption0Training",
    "qwenCaptionIncludeGeneratedQaTraining",
    "qwenCaptionIncludeDeterministicMetadataQa",
    "qwenCaptionIncludeSourceAnnotationsContext",
    "qwenCaptionStrictGrounding",
    "qwenCaptionRequireReadyInstructionExport",
    "qwenCaptionBuildInstructionDataset",
    "qwenCaptionDownloadInstructionJsonl",
    "qwenCaptionDownloadInstructionArchive",
    "qwenCaptionDownloadInstructionReview",
    "qwenCaptionImportInstructionReview",
    "qwenCaptionDownloadInstructionReport",
    "qwenCaptionExportHealth",
    "qwenCaptionReadinessRun",
    "qwenCaptionReadinessStatus",
    "qwenCaptionReadinessResults",
    "qwenCaptionBackendJobStatus",
    "settingsApiRoot",
)

ALLOWED_STATUS_TEXT_SNIPPETS = (
    "auto-attaches immediately and periodically",
    "auto-resume is armed up to",
    "not advertising crash-restart supervision",
)

IGNORED_NETWORK_SUFFIXES = (
    "/favicon.ico",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a rendered Qwen caption UI smoke check against a live backend.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Backend/UI origin, for example http://127.0.0.1:8000.")
    parser.add_argument("--timeout-ms", type=int, default=30_000, help="Playwright navigation and selector timeout.")
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON), help="Where to write the structured smoke report.")
    parser.add_argument("--screenshot", default=str(DEFAULT_SCREENSHOT), help="Where to write a screenshot; use empty string to disable.")
    parser.add_argument(
        "--fail-on-unsupervised-backend",
        action="store_true",
        help="Fail if the UI reports that backend crash-restart supervision is not advertised.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_check(report: dict[str, Any], name: str, ok: bool, detail: str, **fields: Any) -> None:
    report.setdefault("checks", []).append({"name": name, "ok": ok, "detail": detail, **fields})
    if not ok:
        report["ok"] = False


def normalize_base_url(value: str) -> str:
    return str(value or "").strip().rstrip("/")


def is_ignored_url(url: str) -> bool:
    return any(str(url).endswith(suffix) for suffix in IGNORED_NETWORK_SUFFIXES)


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "generated_at": now_iso(),
            "error": "python Playwright is required. Install it in the active environment or run from .venv-macos.",
            "error_type": type(exc).__name__,
            "error_detail": str(exc),
        }

    base_url = normalize_base_url(args.base_url)
    page_url = f"{base_url}/tator.html"
    screenshot_path = Path(args.screenshot).expanduser() if args.screenshot else None
    report: dict[str, Any] = {
        "ok": True,
        "generated_at": now_iso(),
        "base_url": base_url,
        "page_url": page_url,
        "screenshot": str(screenshot_path) if screenshot_path else None,
        "checks": [],
    }

    console_errors: list[str] = []
    failed_requests: list[str] = []
    bad_responses: list[dict[str, Any]] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("requestfailed", lambda request: failed_requests.append(request.url))
        page.on(
            "response",
            lambda response: bad_responses.append({"url": response.url, "status": response.status})
            if response.status >= 400 and not is_ignored_url(response.url)
            else None,
        )

        page.goto(page_url, wait_until="domcontentloaded", timeout=args.timeout_ms)
        page.wait_for_selector("#qwenCaptionDetails", timeout=args.timeout_ms)
        page.evaluate(
            """
            () => {
                const captionDetails = document.querySelector("#qwenCaptionDetails");
                if (captionDetails) captionDetails.open = true;
                const settingsTab = document.querySelector("#tabSettingsButton");
                if (settingsTab) settingsTab.click();
            }
            """
        )
        page.wait_for_timeout(250)
        settings_placeholder = page.locator("#settingsApiRoot").get_attribute("placeholder") or ""
        settings_value = page.locator("#settingsApiRoot").input_value()
        page.evaluate(
            """
            () => {
                const labelingTab = document.querySelector("#tabLabelingButton");
                if (labelingTab) labelingTab.click();
                const captionDetails = document.querySelector("#qwenCaptionDetails");
                if (captionDetails) captionDetails.open = true;
                const caption = document.querySelector("#qwenCaptionOutput");
                if (caption) caption.scrollIntoView({ block: "center" });
            }
            """
        )
        page.wait_for_timeout(750)

        controls: dict[str, dict[str, Any]] = {}
        for control_id in CRITICAL_CONTROLS:
            locator = page.locator(f"#{control_id}")
            count = locator.count()
            controls[control_id] = {
                "count": count,
                "visible": locator.is_visible() if count == 1 else False,
                "enabled": locator.is_enabled() if count == 1 else False,
            }

        set_and_forget_checked = page.locator("#qwenCaptionSetAndForget").is_checked()
        save_generated_checked = page.locator("#qwenCaptionSaveText").is_checked()
        generated_primary_checked = page.locator("#qwenCaptionGeneratedPrimary").is_checked()
        backend_status = page.locator("#qwenCaptionBackendJobStatus").inner_text(timeout=args.timeout_ms)
        alternate_status = page.locator("#qwenCaptionAlternateStatus").inner_text(timeout=args.timeout_ms)
        archive_status = page.locator("#qwenCaptionArchiveStatus").inner_text(timeout=args.timeout_ms)
        export_health = page.locator("#qwenCaptionExportHealth").inner_text(timeout=args.timeout_ms)
        alternates_text = page.locator("#qwenCaptionAlternates").inner_text(timeout=args.timeout_ms)
        grouped_export_count = page.locator("text=Download grouped JSON").count()
        vlm_export_count = page.locator("text=Download VLM JSONL").count()
        instruction_build_count = page.locator("text=Create VLM training dataset").count()
        instruction_jsonl_count = page.locator("text=Download instruction JSONL").count()
        instruction_archive_count = page.locator("text=Download instruction archive").count()
        instruction_review_count = page.locator("text=Download review JSONL").count()
        instruction_import_count = page.locator("text=Import reviewed JSONL").count()
        instruction_report_count = page.locator("text=Download instruction report").count()
        instruction_help = page.locator(".qwen-caption-instruction-panel .training-help").inner_text(timeout=args.timeout_ms)
        action_button_metrics = page.evaluate(
            """
            () => Array.from(document.querySelectorAll(
                ".qwen-caption-actions-row .training-button, "
                + ".qwen-caption-status-row .training-button, "
                + ".qwen-caption-resume-row .training-button"
            )).map((button) => {
                const rect = button.getBoundingClientRect();
                return {
                    id: button.id || "",
                    text: button.textContent.trim(),
                    width: Math.round(rect.width),
                    height: Math.round(rect.height),
                    clientWidth: button.clientWidth,
                    scrollWidth: button.scrollWidth,
                    clientHeight: button.clientHeight,
                    scrollHeight: button.scrollHeight,
                };
            })
            """
        )
        overflowing_action_buttons = [
            metric
            for metric in action_button_metrics
            if metric.get("scrollWidth", 0) > metric.get("clientWidth", 0) + 1
            or metric.get("scrollHeight", 0) > metric.get("clientHeight", 0) + 1
        ]
        subcaptions_value = page.locator("#qwenCaptionSubcaptionsPerImage").input_value()
        subcaptions_min = page.locator("#qwenCaptionSubcaptionsPerImage").get_attribute("min") or ""
        subcaptions_max = page.locator("#qwenCaptionSubcaptionsPerImage").get_attribute("max") or ""
        qa_mix_value = page.locator("#qwenCaptionQaMix").input_value()
        answer_format_value = page.locator("#qwenCaptionAnswerFormat").input_value()
        include_caption0_checked = page.locator("#qwenCaptionIncludeCaption0Training").is_checked()
        include_generated_qa_checked = page.locator("#qwenCaptionIncludeGeneratedQaTraining").is_checked()
        include_deterministic_checked = page.locator("#qwenCaptionIncludeDeterministicMetadataQa").is_checked()
        include_source_context_checked = page.locator("#qwenCaptionIncludeSourceAnnotationsContext").is_checked()
        strict_grounding_checked = page.locator("#qwenCaptionStrictGrounding").is_checked()
        require_ready_instruction_export_checked = page.locator("#qwenCaptionRequireReadyInstructionExport").is_checked()
        page.locator("#qwenCaptionReadinessRun").click(timeout=args.timeout_ms)
        page.wait_for_function(
            """
            () => {
                const status = document.querySelector("#qwenCaptionReadinessStatus");
                const results = document.querySelectorAll("#qwenCaptionReadinessResults li");
                return status && status.textContent.includes("Caption readiness:") && results.length > 0;
            }
            """,
            timeout=args.timeout_ms,
        )
        readiness_status = page.locator("#qwenCaptionReadinessStatus").inner_text(timeout=args.timeout_ms)
        readiness_result_count = page.locator("#qwenCaptionReadinessResults li").count()

        if screenshot_path:
            page.evaluate(
                """
                () => {
                    const panel = document.querySelector(".qwen-caption-instruction-panel");
                    if (panel) panel.scrollIntoView({ block: "center" });
                }
                """
            )
            page.wait_for_timeout(250)
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshot_path), full_page=False)
        browser.close()

    missing = [control_id for control_id, state in controls.items() if state["count"] != 1]
    visibility_exempt_controls = {"qwenCaptionDetails", "qwenCaptionReadinessResults", "settingsApiRoot"}
    hidden = [
        control_id
        for control_id, state in controls.items()
        if state["count"] == 1 and not state["visible"] and control_id not in visibility_exempt_controls
    ]
    app_console_errors = [message for message in console_errors if "favicon" not in message.lower()]
    app_failed_requests = [url for url in failed_requests if not is_ignored_url(url)]
    unsupervised_warning = "not advertising crash-restart supervision" in backend_status

    report.update(
        {
            "controls": controls,
            "set_and_forget_checked": set_and_forget_checked,
            "save_generated_checked": save_generated_checked,
            "generated_primary_checked": generated_primary_checked,
            "backend_status": backend_status,
            "alternate_status": alternate_status,
            "archive_status": archive_status,
            "export_health": export_health,
            "alternates_text": alternates_text,
            "settings_placeholder": settings_placeholder,
            "settings_value": settings_value,
            "grouped_export_button_count": grouped_export_count,
            "vlm_export_button_count": vlm_export_count,
            "instruction_build_button_count": instruction_build_count,
            "instruction_jsonl_button_count": instruction_jsonl_count,
            "instruction_archive_button_count": instruction_archive_count,
            "instruction_review_button_count": instruction_review_count,
            "instruction_import_button_count": instruction_import_count,
            "instruction_report_button_count": instruction_report_count,
            "instruction_help": instruction_help,
            "action_button_metrics": action_button_metrics,
            "overflowing_action_buttons": overflowing_action_buttons,
            "subcaptions_value": subcaptions_value,
            "subcaptions_min": subcaptions_min,
            "subcaptions_max": subcaptions_max,
            "qa_mix_value": qa_mix_value,
            "answer_format_value": answer_format_value,
            "include_caption0_checked": include_caption0_checked,
            "include_generated_qa_checked": include_generated_qa_checked,
            "include_deterministic_checked": include_deterministic_checked,
            "include_source_context_checked": include_source_context_checked,
            "strict_grounding_checked": strict_grounding_checked,
            "require_ready_instruction_export_checked": require_ready_instruction_export_checked,
            "readiness_status": readiness_status,
            "readiness_result_count": readiness_result_count,
            "console_errors": app_console_errors,
            "failed_requests": app_failed_requests,
            "bad_responses": bad_responses,
            "unsupervised_backend_warning": unsupervised_warning,
        }
    )

    add_check(report, "critical_controls_present", not missing, "All critical caption controls are present.", missing=missing)
    add_check(report, "critical_controls_visible", not hidden, "All critical caption controls are visible.", hidden=hidden)
    add_check(report, "set_and_forget_default", set_and_forget_checked, "Set-and-forget is checked by default.")
    add_check(report, "save_generated_default", save_generated_checked, "Generated captions are saved by default.")
    add_check(
        report,
        "generated_primary_default_off",
        not generated_primary_checked,
        "Generated captions append as alternates by default instead of replacing the selected primary.",
    )
    add_check(report, "grouped_export_visible", grouped_export_count == 1, "Grouped JSON export button is visible.", count=grouped_export_count)
    add_check(report, "vlm_export_visible", vlm_export_count == 1, "VLM JSONL export button is visible.", count=vlm_export_count)
    add_check(
        report,
        "instruction_dataset_controls_visible",
        instruction_build_count == 1
        and instruction_jsonl_count == 1
        and instruction_archive_count == 1
        and instruction_review_count == 1
        and instruction_import_count == 1
        and instruction_report_count == 1,
        "Instruction dataset action and export buttons are visible.",
        build_count=instruction_build_count,
        jsonl_count=instruction_jsonl_count,
        archive_count=instruction_archive_count,
        review_count=instruction_review_count,
        import_count=instruction_import_count,
        report_count=instruction_report_count,
    )
    add_check(
        report,
        "instruction_dataset_defaults",
        (
            subcaptions_value == "8"
            and subcaptions_min == "0"
            and subcaptions_max == "20"
            and qa_mix_value == "balanced"
            and answer_format_value == "natural"
            and include_caption0_checked
            and include_generated_qa_checked
            and not include_deterministic_checked
            and include_source_context_checked
            and strict_grounding_checked
            and require_ready_instruction_export_checked
        ),
        "Instruction dataset defaults are set for caption0 plus generated QA with deterministic metadata off and ready-gated trainer export on.",
        subcaptions_value=subcaptions_value,
        subcaptions_min=subcaptions_min,
        subcaptions_max=subcaptions_max,
        qa_mix_value=qa_mix_value,
        answer_format_value=answer_format_value,
        include_caption0_checked=include_caption0_checked,
        include_generated_qa_checked=include_generated_qa_checked,
        include_deterministic_checked=include_deterministic_checked,
        include_source_context_checked=include_source_context_checked,
        strict_grounding_checked=strict_grounding_checked,
        require_ready_instruction_export_checked=require_ready_instruction_export_checked,
    )
    add_check(
        report,
        "instruction_dataset_help_explains_separation",
        "Generated QA never becomes source annotations" in instruction_help
        and "deterministic metadata QA is included only when explicitly enabled" in instruction_help
        and "imported to apply accepted, rejected, or needs-revision decisions before training" in instruction_help,
        "Instruction dataset help explains generated QA/source annotation separation and review import.",
        help_text=instruction_help,
    )
    add_check(
        report,
        "caption_action_buttons_do_not_clip",
        not overflowing_action_buttons,
        "Caption action buttons render without clipped text.",
        overflowing=overflowing_action_buttons,
        metrics=action_button_metrics,
    )
    add_check(
        report,
        "readiness_check_renders_results",
        "Caption readiness:" in readiness_status and readiness_result_count >= 10,
        "In-app caption readiness check renders concrete results.",
        status=readiness_status,
        result_count=readiness_result_count,
    )
    add_check(
        report,
        "alternate_caption_empty_state",
        "Load an image to see saved captions" in alternates_text,
        "Alternate caption list has a useful empty state.",
    )
    add_check(
        report,
        "alternate_caption_status",
        "Primary text label plus saved alternates" in alternate_status,
        "Alternate caption status explains primary plus alternates.",
    )
    add_check(
        report,
        "caption_archive_status",
        "No upper limit is enforced" in archive_status,
        "Caption archive status states that per-image alternate captions are uncapped.",
        status=archive_status,
    )
    add_check(
        report,
        "caption_export_health_status",
        "VLM export validation has not run yet" in export_health,
        "VLM export validation status is visible before any export.",
        status=export_health,
    )
    add_check(
        report,
        "backend_status_explains_set_and_forget",
        any(snippet in backend_status for snippet in ALLOWED_STATUS_TEXT_SNIPPETS),
        "Backend status explains auto-attach or crash-supervision state.",
    )
    add_check(
        report,
        "backend_supervision_required",
        not (args.fail_on_unsupervised_backend and unsupervised_warning),
        "Backend advertises crash-restart supervision when strict mode is requested.",
    )
    add_check(
        report,
        "api_root_defaults_to_serving_origin",
        settings_placeholder == "Current backend origin" and settings_value == base_url,
        "Backend config defaults to the serving origin while preserving manual override.",
        expected=base_url,
        actual=settings_value,
        placeholder=settings_placeholder,
    )
    add_check(report, "no_console_errors", not app_console_errors, "No app console errors occurred.", errors=app_console_errors)
    add_check(report, "no_failed_requests", not app_failed_requests, "No app network requests failed.", failed_requests=app_failed_requests)
    add_check(report, "no_bad_responses", not bad_responses, "No app HTTP responses were 400 or higher.", bad_responses=bad_responses)

    return report


def main() -> int:
    args = parse_args()
    report = run_smoke(args)
    write_report(args.out_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
