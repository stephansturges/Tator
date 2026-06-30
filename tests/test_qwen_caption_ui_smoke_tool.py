from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "run_qwen_caption_ui_smoke.py"
README_PATH = ROOT / "tools" / "README.md"


def _tool_text() -> str:
    return TOOL_PATH.read_text(encoding="utf-8")


def test_qwen_caption_ui_smoke_tool_covers_critical_controls() -> None:
    text = _tool_text()

    for control_id in [
        "qwenCaptionDetails",
        "qwenCaptionDataset",
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
        "qwenCaptionBuildInstructionDataset",
        "qwenCaptionPreviewInstructionProcess",
        "qwenCaptionDownloadInstructionBundle",
        "qwenCaptionDownloadInstructionJsonl",
        "qwenCaptionDownloadInstructionArchive",
        "qwenCaptionDownloadInstructionReview",
        "qwenCaptionImportInstructionReview",
        "qwenCaptionDownloadInstructionReport",
        "qwenCaptionInstructionAdvanced",
        "qwenCaptionInstructionDatasetStatus",
        "qwenCaptionInstructionModelStatus",
        "qwenCaptionInstructionJobStatus",
        "qwenCaptionInstructionReadinessStatus",
        "qwenCaptionInstructionActionReason",
        "qwenCaptionExportHealth",
        "qwenCaptionReadinessRun",
        "qwenCaptionReadinessStatus",
        "qwenCaptionReadinessResults",
        "qwenCaptionBackendJobStatus",
        "settingsApiRoot",
    ]:
        assert control_id in text

    assert "Current backend origin" in text
    assert "auto-attaches immediately and periodically" in text
    assert "auto-resume is armed up to" in text
    assert "not advertising crash-restart supervision" in text
    assert "Primary text label plus saved alternates" in text
    assert "generated_primary_default_off" in text
    assert "Load an image to see saved captions" in text
    assert "No upper limit is enforced" in text
    assert "VLM export validation has not run yet" in text
    assert "Download grouped JSON" in text
    assert "Download VLM JSONL" in text
    assert "Create training dataset" in text
    assert "Preview dataset prompts" in text
    assert "Download training bundle" in text
    assert "Download trainer JSONL" in text
    assert "Download construction archive" in text
    assert "Import review decisions" in text
    assert "Download readiness report" in text
    assert "Instruction dataset defaults are set for caption0 plus generated QA" in text
    assert "Generated Q&A never becomes source annotations" in text
    assert "All-image caption and training-dataset jobs start with the selected image" in text
    assert "all_image_order_help_visible" in text
    assert "vlm_export_button_count" in text
    assert "instruction_build_button_count" in text
    assert "instruction_bundle_button_count" in text
    assert "instruction_preview_button_count" in text
    assert "instruction_preview_before_dataset_create" in text
    assert "instruction_advanced_exports_collapsed_by_default" in text
    assert "instruction_status_strip_visible" in text
    assert "instruction_status_strip_updates_after_readiness" in text
    assert "instruction_preview_index" in text
    assert "instruction_build_index" in text
    assert "instruction_review_button_count" in text
    assert "instruction_import_button_count" in text
    assert "instruction_report_button_count" in text
    assert "instruction_dataset_help_explains_separation" in text
    assert "caption_action_buttons_do_not_clip" in text
    assert "overflowing_action_buttons" in text
    assert "Caption readiness:" in text
    assert "readiness_result_count" in text
    assert "readiness_check_renders_results" in text
    assert "requestfailed" in text
    assert "console_errors" in text
    assert "bad_responses" in text
    assert "--fail-on-unsupervised-backend" in text


def test_qwen_caption_ui_smoke_tool_help_and_readme_are_wired() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL_PATH), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--base-url" in result.stdout
    assert "--fail-on-unsupervised-backend" in result.stdout
    readme = README_PATH.read_text(encoding="utf-8")
    assert "python tools/run_qwen_caption_ui_smoke.py --base-url http://127.0.0.1:8000" in readme
    assert "--fail-on-unsupervised-backend" in readme
