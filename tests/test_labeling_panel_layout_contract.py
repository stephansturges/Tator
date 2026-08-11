from html.parser import HTMLParser
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = REPO_ROOT / "ybat-master" / "tator.html"
CSS_PATH = REPO_ROOT / "ybat-master" / "ybat.css"
JS_PATH = REPO_ROOT / "ybat-master" / "ybat.js"
STATIC_CONTROL_FIELD_CLASSES = {
    "training-field",
    "sam3-text-field",
    "data-ingestion-field",
    "class-split-field",
    "class-split-cluster-controls__field",
    "qwen-caption-row",
    "shortcut-settings-row",
}
VOID_HTML_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
DYNAMIC_JS_CREATED_IDS = {
    "dataIngestionHoverPreview",
    "classSplitGraphHoverPreview",
    "classSplitDatasetAnalysisHoverPreview",
}


def _html() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


def _css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def _js() -> str:
    return JS_PATH.read_text(encoding="utf-8")


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def _extract_js_function(source: str, name: str) -> str:
    start = source.index(f"function {name}")
    parameter_start = source.index("(", start)
    parameter_depth = 0
    parameter_end = None
    for index in range(parameter_start, len(source)):
        char = source[index]
        if char == "(":
            parameter_depth += 1
        elif char == ")":
            parameter_depth -= 1
            if parameter_depth == 0:
                parameter_end = index
                break
    if parameter_end is None:
        raise AssertionError(f"Could not extract JS function signature {name}")
    brace_start = source.index("{", parameter_end)
    depth = 0
    for index in range(brace_start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"Could not extract JS function {name}")


def _extract_js_function_before(source: str, name: str, next_marker: str) -> str:
    start = source.index(f"function {name}")
    end = source.index(next_marker, start)
    return source[start:end].rstrip()


def _extract_js_block(source: str, marker: str) -> str:
    start = source.index(marker)
    brace_start = source.index("{", start + len(marker))
    depth = 0
    for index in range(brace_start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"Could not extract JS block {marker}")


def test_backend_api_root_defaults_to_serving_origin_with_manual_override():
    html = _html()
    js = _js()

    assert 'const FALLBACK_API_ROOT = "http://localhost:8000";' in js
    assert "function resolveDefaultApiRoot()" in js
    assert "window.location.origin" in js
    assert 'const DEFAULT_API_ROOT = resolveDefaultApiRoot();' in js
    assert "const normalized = normalizeApiRoot(saved)" in js
    assert "return normalized || DEFAULT_API_ROOT" in js
    assert 'placeholder="Current backend origin"' in html
    assert "By default, the UI uses the same backend origin that served this page." in html
    assert "Override this only for tunnels or split frontend/backend setups" in html


class _HtmlNode:
    def __init__(self, tag: str, attrs: dict[str, str | None], parent=None, position=(0, 0)):
        self.tag = tag
        self.attrs = attrs
        self.parent = parent
        self.position = position
        self.children: list[_HtmlNode] = []
        self.text_parts: list[str] = []

    @property
    def classes(self) -> set[str]:
        return set(str(self.attrs.get("class") or "").split())

    def text_content(self) -> str:
        text = "".join(self.text_parts)
        for child in self.children:
            text += child.text_content()
        return re.sub(r"\s+", " ", text).strip()

    def ancestors(self):
        node = self.parent
        while node is not None:
            yield node
            node = node.parent

    def descendants(self, tag: str | None = None):
        for child in self.children:
            if tag is None or child.tag == tag:
                yield child
            yield from child.descendants(tag)


class _StaticHtmlParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.root = _HtmlNode("document", {})
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        node = _HtmlNode(tag, dict(attrs), self.stack[-1], self.getpos())
        self.stack[-1].children.append(node)
        if tag not in VOID_HTML_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        tag = tag.lower()
        node = _HtmlNode(tag, dict(attrs), self.stack[-1], self.getpos())
        self.stack[-1].children.append(node)

    def handle_endtag(self, tag):
        tag = tag.lower()
        while len(self.stack) > 1:
            node = self.stack.pop()
            if node.tag == tag:
                break

    def handle_data(self, data):
        self.stack[-1].text_parts.append(data)


def _parse_static_html() -> _HtmlNode:
    parser = _StaticHtmlParser()
    parser.feed(_html())
    return parser.root


def _static_html_ids(html: str) -> set[str]:
    matches = re.findall(r"""\bid=(?:"([^"]+)"|'([^']+)')""", html)
    return {first or second for first, second in matches if first or second}


def _static_get_element_by_id_refs(js: str) -> set[str]:
    matches = re.findall(r"""document\.getElementById\(\s*(?:"([^"]+)"|'([^']+)')\s*\)""", js)
    return {first or second for first, second in matches if first or second}


def _nodes_by_tag(root: _HtmlNode, tag: str) -> list[_HtmlNode]:
    return list(root.descendants(tag))


def _control_override_id_list(js: str) -> list[str]:
    match = re.search(r"const CONTROL_TOOLTIP_OVERRIDES = Object\.freeze\(\{(.*?)\n\s*\}\);", js, re.S)
    assert match, "missing CONTROL_TOOLTIP_OVERRIDES"
    override_block = match.group(1)
    return re.findall(r"^\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*:", override_block, re.M)


def _control_override_ids(js: str) -> set[str]:
    return set(_control_override_id_list(js))


def _label_text_for_field(control: _HtmlNode) -> str:
    for ancestor in control.ancestors():
        if ancestor.classes & STATIC_CONTROL_FIELD_CLASSES:
            labels = [child for child in ancestor.children if child.tag == "label"]
            labels.extend(ancestor.descendants("label"))
            for label in labels:
                text = label.text_content()
                if text:
                    return text
        if ancestor.tag == "details":
            summary = next(ancestor.descendants("summary"), None)
            if summary:
                return summary.text_content()
    return ""


def _control_has_accessible_static_or_runtime_tooltip(
    control: _HtmlNode,
    labels_by_for: dict[str, str],
    override_ids: set[str],
) -> bool:
    if str(control.attrs.get("title") or "").strip():
        return True
    if control.tag == "button" and control.text_content():
        return True
    if control.tag == "input" and str(control.attrs.get("type") or "").lower() in {"button", "submit", "reset"}:
        if str(control.attrs.get("value") or "").strip():
            return True

    control_id = str(control.attrs.get("id") or "").strip()
    if control_id:
        if control_id in override_ids:
            return True
        if labels_by_for.get(control_id):
            return True

    if any(ancestor.tag == "label" and ancestor.text_content() for ancestor in control.ancestors()):
        return True

    return bool(_label_text_for_field(control))


def _describe_control(control: _HtmlNode) -> str:
    attrs = []
    for key in ("id", "name", "type", "class"):
        value = control.attrs.get(key)
        if value:
            attrs.append(f'{key}="{value}"')
    line, column = control.position
    return f"<{control.tag} {' '.join(attrs)}> at {line}:{column}"


def _details_opening_tag(html: str, element_id: str) -> str:
    match = re.search(rf"<details\b[^>]*\bid=[\"']{re.escape(element_id)}[\"'][^>]*>", html)
    assert match, f"missing details#{element_id}"
    return match.group(0)


def test_labeling_tool_panels_default_closed_and_ordered():
    html = _html()
    panel_ids = [
        "qwenDetectionDetails",
        "qwenCaptionDetails",
        "qwenEdrDetails",
        "sam3TextPanel",
    ]

    positions = []
    for panel_id in panel_ids:
        tag = _details_opening_tag(html, panel_id)
        assert " open" not in tag, f"{panel_id} should default closed"
        positions.append(html.index(f'id="{panel_id}"'))

    assert positions == sorted(positions)
    assert "Qwen 3 object detection" in html
    assert "not great" not in html.lower()
    assert "Detection Recipe" in html
    assert "Ensemble Detection Recipe" in html
    assert "[wip]" not in html.lower()


def test_qwen_caption_all_advertises_resumable_backend_job():
    html = _html()
    js = _js()

    assert "Set-and-forget is the default dataset-backed path for Caption image, next-N, and Caption all" in html
    assert "uses persisted backend jobs, isolated retries, health gates, and a bounded auto-resume limit after backend restarts" in html
    assert "Direct single-image captioning is only for deliberate non-set-and-forget diagnostics" in html
    assert "Saved generated captions are appended as alternate caption records" in html
    assert "qwenCaptionAlternates" in html
    assert "qwenCaptionArchiveStatus" in html
    assert "qwenCaptionGeneratedPrimary" in html
    assert "Make generated caption primary" in html
    assert "Off by default for alternate-caption training" in html
    assert "qwenCaptionSaveAlternate" in html
    assert "qwenCaptionUpdateSelected" in html
    assert "qwenCaptionSetPrimary" in html
    assert "qwenCaptionDeleteSelected" in html
    assert "qwenCaptionDownloadJsonl" in html
    assert "qwenCaptionDownloadGroupedJson" in html
    assert "qwenCaptionDownloadVlmJsonl" in html
    assert "qwenCaptionBatchFollowActive" in html
    assert "Follow backend image" in html
    assert "qwenCaptionImposedQuestions" in html
    assert "Imposed questions" in html
    assert "qwenCaptionGeneratedQaOutput" in html
    assert "Latest generated Q&amp;A output" in html
    assert "qwenCaptionSubcaptionsPerImage" in html
    assert 'id="qwenCaptionSubcaptionsPerImage" min="0" max="20" step="1" value="8"' in html
    assert "qwenCaptionQaMix" in html
    assert '<option value="balanced" selected>Balanced</option>' in html
    assert "qwenCaptionAnswerFormat" in html
    assert '<option value="natural" selected>Natural text</option>' in html
    assert "qwenCaptionIncludeCaption0Training" in html
    assert "qwenCaptionIncludeGeneratedQaTraining" in html
    assert "qwenCaptionIncludeDeterministicMetadataQa" in html
    assert "qwenCaptionIncludeSourceAnnotationsContext" in html
    assert "qwenCaptionStrictGrounding" in html
    assert "qwenCaptionRestrictSpeculativeQaLanguage" in html
    assert "Restrict speculative Q&amp;A language" in html
    assert "qwenCaptionRequireReadyInstructionExport" in html
    assert 'id="qwenCaptionRequireReadyInstructionExport" checked' in html
    assert "qwenCaptionBuildInstructionDataset" in html
    assert "qwenCaptionDownloadInstructionBundle" in html
    assert "qwenCaptionDownloadInstructionJsonl" in html
    assert "qwenCaptionDownloadInstructionArchive" in html
    assert "qwenCaptionDownloadInstructionReview" in html
    assert "qwenCaptionImportInstructionReview" in html
    assert "qwenCaptionImportInstructionReviewFile" in html
    assert "qwenCaptionDownloadInstructionReport" in html
    assert "qwenCaptionInstructionAdvanced" in html
    assert "qwenCaptionInstructionDatasetStatus" in html
    assert "qwenCaptionInstructionModelStatus" in html
    assert "qwenCaptionInstructionJobStatus" in html
    assert "qwenCaptionInstructionReadinessStatus" in html
    assert "qwenCaptionInstructionActionReason" in html
    assert "Caption archive" in html
    assert "Training dataset" in html
    assert "Preview prompts" in html
    assert "Create training dataset" in html
    assert "Download training bundle" in html
    assert "Advanced exports and review" in html
    assert "Download trainer JSONL" in html
    assert "Download construction archive" in html
    assert "Download review file" in html
    assert "Import review decisions" in html
    assert "Download readiness report" in html
    assert "Generated Q&amp;A never becomes source annotations" in html
    assert "The training bundle is the handoff" in html
    assert "copied images, effective labels, trainer JSONL, construction archive, review file, readiness report, and checksums" in html
    assert "Trainer JSONL and bundle downloads require a ready report by default" in html
    assert "advanced exports are for review and diagnostics" in html
    assert "qwenCaptionExportHealth" in html
    assert "qwenCaptionReadinessRun" in html
    assert "qwenCaptionInstructionReadinessRun" in html
    assert "qwenCaptionReadinessStatus" in html
    assert "qwenCaptionReadinessResults" in html
    assert "Check caption readiness" in html
    assert html.count("Check caption readiness") >= 2
    assert "Download grouped JSON" in html
    assert "Download VLM JSONL" in html
    assert "Flat JSONL is the audit record stream" in html
    assert "grouped JSON keeps each image with all of its ordered alternate captions" in html
    assert "VLM JSONL emits one normal image/question/answer training row per caption" in html
    assert "Caption archive status will appear here" in html
    assert "VLM export validation has not run yet" in html
    assert "backend launcher or another process supervisor is running" in html
    assert "tools/run_macos_backend.sh or another process supervisor" in html
    assert "Requires a caption dataset so the request runs as a persisted backend job" in html
    assert "/qwen/caption/jobs" in js
    assert "qwenCaptionBatchBackendJobId" in js
    assert "function getCaptionInstructionDatasetSettings" in js
    assert "function validateCaptionInstructionLaunchSettings" in js
    assert "function describeCaptionInstructionLaunchSettings" in js
    assert "subcaptions_per_image: subcaptions" in js
    assert "Enable at least one instruction training row family" in js
    assert "Generated Q&A is disabled" in js
    assert "generated QA disabled" in js
    assert "instructionDataset: true" in js
    assert "function validateCaptionInstructionTrainingRows" in js
    assert "function validateCaptionInstructionArchiveRows" in js
    assert "function validateCaptionInstructionReviewRows" in js
    assert "function parseCaptionInstructionReviewRowsText" in js
    assert "async function importCaptionInstructionReviewFile" in js
    assert "CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_BYTES" in js
    assert "browser import safety limit" in js
    assert "normalizeCaptionInstructionReviewDecision" in js
    assert "function validateCaptionInstructionReport" in js
    assert "function validateCaptionInstructionArtifactConsistency" in js
    assert "corpus_quality_metrics" in js
    assert "training_readiness" in js
    assert "instruction_export_validation" in js
    assert "artifact consistency failed" in js
    assert "backend artifact consistency failed" in js
    assert "instruction_artifact_consistency objects disagree" in js
    assert "does not match report selected row count" in js
    assert "function downloadCaptionInstructionBundle" in js
    assert "function downloadCaptionInstructionJsonl" in js
    assert "function downloadCaptionInstructionArchive" in js
    assert "function downloadCaptionInstructionReview" in js
    assert "function importCaptionInstructionReviewFile" in js
    assert "function downloadCaptionInstructionReport" in js
    assert "function formatCaptionInstructionExportApiError" in js
    assert "requireReadyInstructionExport" in js
    assert "instruction_export_not_ready" in js
    assert "persistableRows" in js
    assert "decisions only for deterministic rows" in js
    assert "rows: persistableRows" in js
    assert "captionInstructionReadinessSummary" in js
    assert "require_ready_instruction_export" in js
    assert 'saveBlobToDisk(blob, "caption_instruction_training.jsonl")' in js
    assert 'saveBlobToDisk(blob, "caption_instruction_training_bundle.zip")' in js
    assert 'saveBlobToDisk(blob, "caption_instruction_archive.jsonl")' in js
    assert 'saveBlobToDisk(blob, "caption_instruction_review.jsonl")' in js
    assert 'saveBlobToDisk(blob, "caption_instruction_report.json")' in js
    assert "/captions/instruction_review" in js
    assert "/captions/instruction_bundle" in js
    assert "function formatCaptionInstructionBundleApiError" in js
    assert "formatCaptionInstructionBundleApiError(parseApiError" in js
    assert "async function applyQwenCaptionBackendJobCaptions" not in js
    assert "function applyQwenCaptionBackendJobCaptions" in js
    assert "result.latest_caption" in js
    assert "/datasets/${encodeURIComponent(datasetId)}/captions/export" in js
    export_start = js.index("async function downloadCaptionInstructionJsonl")
    archive_start = js.index("async function downloadCaptionInstructionArchive", export_start)
    export_fn = js[export_start:archive_start]
    assert "requireReadyInstructionExport: settings.require_ready_instruction_export === true" in export_fn
    assert "return;\n            return;" not in export_fn
    archive_end = js.index("async function downloadCaptionInstructionReview", archive_start)
    archive_fn = js[archive_start:archive_end]
    assert "requireReadyInstructionExport" not in archive_fn
    review_end = js.index("async function importCaptionInstructionReviewFile", archive_end)
    review_fn = js[archive_end:review_end]
    assert "requireReadyInstructionExport" not in review_fn
    report_start = js.index("async function downloadCaptionInstructionReport", review_end)
    report_end = js.index("function buildSam3TextSnapshot", report_start)
    report_fn = js[report_start:report_end]
    assert "requireReadyInstructionExport" not in report_fn


def test_qwen_caption_export_preserves_saved_alternates_and_primary_rows():
    js = _js()

    ensure_start = js.index("async function ensureCaptionsForExport")
    export_start = js.index("async function loadCaptionExportRecords", ensure_start)
    ensure_helper = js[ensure_start:export_start]
    export_end = js.index("async function loadCaptionForCurrentImage", export_start)
    export_helper = js[export_start:export_end]
    load_end = js.index("async function captionExistsForImage", export_end)
    load_helper = js[export_end:load_end]

    assert "if (!missingImageNames.length)" in ensure_helper
    assert "await loadCaptionExportRecords(datasetId).catch" in ensure_helper
    assert "const hasCaptionExportContent = (record)" in export_helper
    assert "if (!hasCaptionExportContent(record))" in export_helper
    assert "const backendContentKeys = new Set(backendExportRecords.map(exportContentKey))" in export_helper
    assert "!backendContentKeys.has(key) && !seenContentKeys.has(key)" in export_helper
    assert "const perImageCaptionCounts = {}" in export_helper
    assert "record.caption_index = perImageCaptionCounts[imageName]" in export_helper
    assert 'caption_index: Number.parseInt(source.caption_index || "0", 10) || 0' in js
    assert "async function prepareCaptionExportRecords" in js
    assert "function buildGroupedCaptionExport" in js
    assert 'format: "tator_caption_grouped_v1"' in js
    assert "caption_count: imagesOut.reduce" in js
    assert 'saveBlobToDisk(blob, "captions_grouped.json")' in js
    assert "function buildCaptionVlmTrainingRows" in js
    assert 'answer: JSON.stringify({ caption })' in js
    assert "getCaptionVlmTrainingQuestion(captionIndex)" in js
    assert "function validateCaptionVlmTrainingRows" in js
    assert "function validateCaptionInstructionTrainingRows" in js
    assert "function validateCaptionInstructionArchiveRows" in js
    assert "function validateCaptionInstructionReviewRows" in js
    assert "function parseCaptionInstructionReviewRowsText" in js
    assert "async function importCaptionInstructionReviewFile" in js
    assert "function validateCaptionInstructionReport" in js
    instruction_validator_start = js.index("function validateCaptionInstructionTrainingRows")
    instruction_validator_end = js.index("function describeCaptionInstructionValidation", instruction_validator_start)
    instruction_validator = js[instruction_validator_start:instruction_validator_end]
    assert "const metadata = row?.metadata" in instruction_validator
    assert "metadata missing qa_id" in instruction_validator
    assert "metadata missing row_type" in instruction_validator
    assert "metadata missing answer_source" in instruction_validator
    assert "metadata missing source_archive" in instruction_validator
    assert "metadata missing answer_format" in instruction_validator
    assert "metadata missing validation_status" in instruction_validator
    assert "metadata missing review_status" in instruction_validator
    assert "metadata validation_status is unsupported" in instruction_validator
    assert "metadata review_status is unsupported" in instruction_validator
    assert "const rowType = String(metadata.row_type" in instruction_validator
    assert "const sourceArchive = String(metadata.source_archive" in instruction_validator
    assert "const answerFormat = String(metadata.answer_format" in instruction_validator
    assert "const validationStatus = String(metadata.validation_status" in instruction_validator
    assert "[metadata.review_status, metadata.review_decision]" in instruction_validator
    assert "reviewStatuses.some" in instruction_validator
    assert "has non-trainable review status" in instruction_validator
    report_validator_start = js.index("function validateCaptionInstructionReport")
    report_validator_end = js.index("async function downloadCaptionJsonl", report_validator_start)
    report_validator = js[report_validator_start:report_validator_end]
    assert "corpus_quality_metrics" in report_validator
    assert "generated_qa_question_diversity_ratio" in report_validator
    assert "source_class_coverage_rate" in report_validator
    assert "training_answer_format_distribution" in report_validator
    assert "report missing training_readiness" in report_validator
    assert "training_readiness.status is invalid" in report_validator
    assert "training_readiness.ready_for_training must be boolean" in report_validator
    assert "training_readiness.ready_for_training must be true when status is ready" in report_validator
    assert "training_readiness.ready_for_training must be false unless status is ready" in report_validator
    assert "training_readiness ready status cannot include quality_warnings" in report_validator
    assert "training_readiness blocked status requires blocking_reasons" in report_validator
    assert "training_readiness.thresholds is missing" in report_validator
    assert "report missing instruction_export_validation" in report_validator
    assert "instruction_export_validation contains training-row errors" in report_validator
    assert "report selected_flattened_row_count is missing or invalid" in report_validator
    assert "corpus_quality_metrics.selected_flattened_row_count does not match report selected_flattened_row_count" in report_validator
    assert "instruction_export_validation.row_count does not match selected_flattened_row_count" in report_validator
    assert "report instruction_review_row_count is missing or invalid" in report_validator
    assert "report manual_review_required_count is missing or invalid" in report_validator
    assert "Training readiness blocked" in js
    assert "Training readiness needs review" in js
    assert "Instruction JSONL export blocked: " in js
    assert "artifact consistency failed" in js
    assert "Disable Require ready report only for deliberate review-pending diagnostics" in js
    review_validator_start = js.index("function validateCaptionInstructionReviewRows")
    review_validator_end = js.index("function describeCaptionInstructionReviewValidation", review_validator_start)
    review_validator = js[review_validator_start:review_validator_end]
    assert "tator_caption_instruction_review_rows_v1" in review_validator
    assert "selected_for_training must be boolean" in review_validator
    assert "requires_manual_review must be boolean" in review_validator
    assert "missing review_decision field" in review_validator
    assert "missing review_notes field" in review_validator
    assert "unsupported review_decision" in review_validator
    assert "missing dataset_id for persisted language review row" in review_validator
    assert "unsupported actionable row_origin" in review_validator
    assert "duplicate actionable review target" in review_validator
    assert "conflicting duplicate actionable review target" in review_validator
    assert "normalizeCaptionInstructionReviewDecision" in js
    assert "formatCaptionInstructionReviewImportApiError" in js
    assert "accepted, rejected, or needs-revision decisions" in js
    assert "Use accepted, rejected, needs-revision, or leave the decision blank" in js
    assert "review_rows_no_actionable_decisions" in js
    assert "no accepted, rejected, or needs-revision caption0 or generated-QA decisions" in js
    assert "captionMutationPayload({ rows: persistableRows })" in js
    assert "Export a fresh review JSONL" in js
    assert "duplicate image_path + question" in js
    assert "function setCaptionExportHealth" in js
    assert "VLM JSONL export blocked" in js
    assert "VLM JSONL validated:" in js
    assert "Instruction JSONL export blocked" in js
    assert "Instruction JSONL validated:" in js
    assert "generated_make_primary" in js
    assert "generated_make_primary: !!qwenElements.captionGeneratedPrimary?.checked" in js
    assert "qwenElements.captionGeneratedPrimary?.checked === true" in js
    assert "Generated captions append by default" in js
    assert 'saveBlobToDisk(blob, "captions_vlm_training.jsonl")' in js
    assert 'saveBlobToDisk(blob, "caption_instruction_training.jsonl")' in js
    assert 'saveBlobToDisk(blob, "caption_instruction_archive.jsonl")' in js
    assert 'saveBlobToDisk(blob, "caption_instruction_review.jsonl")' in js
    assert 'saveBlobToDisk(blob, "caption_instruction_report.json")' in js
    assert "downloadCaptionGroupedJson().catch" in js
    assert "downloadCaptionVlmJsonl().catch" in js
    assert "downloadCaptionInstructionJsonl().catch" in js
    assert "downloadCaptionInstructionArchive().catch" in js
    assert "downloadCaptionInstructionReview().catch" in js
    assert "importCaptionInstructionReviewFile(file).catch" in js
    assert "async function runQwenCaptionReadinessCheck" in js
    assert "function collectQwenCaptionReadinessChecks" in js
    assert "renderQwenCaptionReadinessChecks" in js
    assert "Caption readiness:" in js
    assert "Unlimited per-image captions" in js
    assert "captionReadinessRun.addEventListener" in js
    assert "captionInstructionReadinessRun.addEventListener" in js
    assert '["captionInstructionReadinessRun", "Training dataset readiness control"]' in js
    assert "if (updated?.is_primary)" in js
    assert "captionAutoSaveState.lastSaved.set(imageName, updated.caption || caption)" in js
    init_start = js.index("function initQwenPanel")
    init_end = js.index("function refreshQwenStatus", init_start)
    init_helper = js[init_start:init_end]
    assert init_helper.count("renderCaptionAlternatesForCurrentImage();") >= 2
    update_caption_start = js.index("function updateQwenCaptionButton")
    update_caption_end = js.index("function getCaptionPresetText", update_caption_start)
    update_caption_helper = js[update_caption_start:update_caption_end]
    assert "const hasCaptionDataset = !!getCaptionDatasetId();" in update_caption_helper
    assert "function qwenCaptionArchiveMutationActive" in js
    assert "function updateCaptionRunConfigurationControls" in js
    assert "function updateCaptionInstructionDatasetOptionControls" in js
    assert "function updateCaptionInstructionUxStatus" in js
    assert "function setCaptionInstructionStatusChip" in js
    assert "function updateCaptionArchiveActionControls" in js
    assert "const busy = qwenCaptionArchiveMutationActive();" in update_caption_helper
    assert "const captionExportDisabled = busy;" in update_caption_helper
    assert "qwenElements.captionDownloadJsonl.disabled = captionExportDisabled" in update_caption_helper
    assert "qwenElements.captionDownloadGroupedJson.disabled = captionExportDisabled" in update_caption_helper
    assert "qwenElements.captionDownloadVlmJsonl.disabled = captionExportDisabled" in update_caption_helper
    assert "const instructionExportDisabled = !hasCaptionDataset || busy;" in update_caption_helper
    assert "qwenElements.captionDownloadInstructionBundle.disabled = instructionExportDisabled" in update_caption_helper
    assert "qwenElements.captionDownloadInstructionJsonl.disabled = instructionExportDisabled" in update_caption_helper
    assert "qwenElements.captionDownloadInstructionArchive.disabled = instructionExportDisabled" in update_caption_helper
    assert "qwenElements.captionDownloadInstructionReview.disabled = instructionExportDisabled" in update_caption_helper
    assert "qwenElements.captionImportInstructionReview.disabled = instructionExportDisabled" in update_caption_helper
    assert "qwenElements.captionDownloadInstructionReport.disabled = instructionExportDisabled" in update_caption_helper
    assert "qwenElements.captionRecipeLoad.disabled = busy" in update_caption_helper
    assert "qwenElements.captionRecipeUploadButton.disabled = busy" in update_caption_helper
    assert "qwenElements.captionRecipeUpload.disabled = busy" in update_caption_helper
    assert "updateCaptionRunConfigurationControls();" in update_caption_helper
    assert "updateCaptionInstructionDatasetOptionControls();" in update_caption_helper
    assert "updateCaptionGlossaryControls();" in update_caption_helper
    assert "updateCaptionArchiveActionControls();" in update_caption_helper
    assert "updateCaptionInstructionUxStatus({ locked, busy, hasCaptionDataset });" in update_caption_helper
    assert 'typeof updateCaptionInstructionUxStatus === "function"' in update_caption_helper
    run_config_helper = _extract_js_function(js, "updateCaptionRunConfigurationControls")
    assert "captionRunConfigurationElements().forEach" in run_config_helper
    assert "el.disabled = busy;" in run_config_helper
    assert "qwenElements.captionStyleText" in js
    assert "qwenElements.captionModel" in js
    assert "qwenElements.captionSetAndForget" in js
    assert "qwenElements.captionBatchCount" in js
    glossary_helper = _extract_js_function(js, "updateCaptionGlossaryControls")
    assert "const busy = qwenCaptionArchiveMutationActive();" in glossary_helper
    assert "qwenElements.captionGlossary.disabled = locked;" in glossary_helper
    assert "qwenElements.captionGlossaryReset.disabled = locked;" in glossary_helper
    assert "qwenElements.captionGlossarySave.disabled = locked || !datasetId;" in glossary_helper
    assert "function updateCaptionOutputEditControl" in js
    assert "function captionOutputEditingBlocked" in js
    assert 'guardQwenCaptionArchiveIdle("editing caption text")' in js
    assert 'guardQwenCaptionArchiveIdle("saving caption text edits")' in js
    assert 'guardQwenCaptionArchiveIdle("editing caption prompt settings")' in js
    assert 'guardQwenCaptionArchiveIdle("editing caption run settings")' in js
    assert 'guardQwenCaptionArchiveIdle("editing the caption glossary")' in js
    assert 'guardQwenCaptionArchiveIdle("resetting the caption glossary")' in js
    assert 'guardQwenCaptionArchiveIdle("saving the caption glossary")' in js
    instruction_option_helper = _extract_js_function(js, "updateCaptionInstructionDatasetOptionControls")
    assert "qwenElements.captionSubcaptionsPerImage" in instruction_option_helper
    assert "qwenElements.captionQaMix" in instruction_option_helper
    assert "qwenElements.captionAnswerFormat" in instruction_option_helper
    assert "qwenElements.captionIncludeCaption0Training" in instruction_option_helper
    assert "qwenElements.captionIncludeGeneratedQaTraining" in instruction_option_helper
    assert "qwenElements.captionIncludeDeterministicMetadataQa" in instruction_option_helper
    assert "qwenElements.captionIncludeSourceAnnotationsContext" in instruction_option_helper
    assert "qwenElements.captionStrictGrounding" in instruction_option_helper
    assert "qwenElements.captionRestrictSpeculativeQaLanguage" in instruction_option_helper
    render_alternates_helper = _extract_js_function(js, "renderCaptionAlternatesForCurrentImage")
    assert "updateCaptionArchiveActionControls();" in render_alternates_helper
    assert "qwenElements.captionSaveAlternate.disabled = busy || !imageName || !caption" in js
    assert "qwenElements.captionUpdateSelected.disabled = busy || !imageName || !selected || !caption" in js
    assert "qwenElements.captionSetPrimary.disabled = busy || !imageName || !storedAlternate" in js
    assert "qwenElements.captionDeleteSelected.disabled = busy || !imageName || !storedAlternate" in js
    assert "function deferCaptionArchiveReadWhileBusy" in js
    assert "async function loadCaptionForCurrentImage(options = {})" in js
    assert "const allowDuringActive = options.allowDuringActive === true;" in load_helper
    assert 'const busyActionLabel = options.actionLabel || "loading caption archive";' in load_helper
    assert load_helper.count("deferCaptionArchiveReadWhileBusy(busyActionLabel);") >= 2
    assert "if (!allowDuringActive && qwenCaptionArchiveMutationActive())" in load_helper
    assert "return false;" in load_helper
    assert "return true;" in load_helper
    assert 'actionLabel: "loading completed caption job output"' in js
    assert 'actionLabel: "loading completed backend caption output"' in js
    assert "const datasetId = getCaptionRecordDatasetId();" in load_helper
    assert "isAnnotationDatasetModeActive()" not in load_helper
    assert "function captionInstructionArtifactBusyMessage" in js
    assert "the instruction archive is changing" in js
    instruction_artifact_actions = [
        ("downloadCaptionInstructionBundle", "exporting the training bundle"),
        ("downloadCaptionInstructionJsonl", "exporting instruction trainer JSONL"),
        ("downloadCaptionInstructionArchive", "exporting the instruction archive"),
        ("downloadCaptionInstructionReview", "exporting instruction review rows"),
        ("importCaptionInstructionReviewFile", "importing reviewed instruction rows"),
        ("downloadCaptionInstructionReport", "exporting the instruction report"),
    ]
    for function_name, action_label in instruction_artifact_actions:
        action_helper = _extract_js_function(js, function_name)
        assert f'captionInstructionArtifactBusyMessage("{action_label}")' in action_helper
        assert 'setCaptionExportHealth(busyMessage, "warn")' in action_helper
        assert 'setSamStatus(busyMessage, { variant: "warn", duration: 5000 })' in action_helper
    caption_export_actions = [
        ("downloadCaptionJsonl", "exporting caption audit JSONL"),
        ("downloadCaptionGroupedJson", "exporting grouped captions"),
        ("downloadCaptionVlmJsonl", "exporting VLM caption rows"),
    ]
    for function_name, action_label in caption_export_actions:
        action_helper = _extract_js_function(js, function_name)
        assert f'captionArchiveExportBusyMessage("{action_label}")' in action_helper
        assert 'setCaptionExportHealth(busyMessage, "warn")' in action_helper
        assert 'setSamStatus(busyMessage, { variant: "warn", duration: 5000 })' in action_helper


def test_qwen_caption_instruction_artifacts_block_while_backend_job_id_is_active():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "let qwenCaptionActive = false;",
            "let qwenCaptionBatchActive = false;",
            "let qwenCaptionBatchBackendJobId = 'job-1';",
            "let qwenCaptionCancelRequested = false;",
            "let qwenCaptionBatchCancel = false;",
            "let qwenAvailable = true;",
            "let currentImage = { name: 'frame.jpg' };",
            "const selectedCaption = { id: 'alt-1', is_primary: false, caption: 'caption text' };",
            "function getCaptionDatasetId() { return 'ds'; }",
            "function isGpuHeavyLockActive() { return false; }",
            "function isAnnotationDatasetModeActive() { return false; }",
            "function isAnnotationMutationBlocked() { return false; }",
            "function getSelectedCaptionRecord(imageName) { return imageName ? selectedCaption : null; }",
            "function syncQwenCaptionDatasetControls() {}",
            "function button() { return { disabled: false, textContent: '' }; }",
            "const qwenElements = {",
            "  captionRunButton: button(),",
            "  captionCancelButton: button(),",
            "  captionBatchRun: button(),",
            "  captionBatchRunAll: button(),",
            "  captionDownloadJsonl: button(),",
            "  captionDownloadGroupedJson: button(),",
            "  captionDownloadVlmJsonl: button(),",
            "  captionBuildInstructionDataset: button(),",
            "  captionDownloadInstructionBundle: button(),",
            "  captionDownloadInstructionJsonl: button(),",
            "  captionDownloadInstructionArchive: button(),",
            "  captionDownloadInstructionReview: button(),",
            "  captionImportInstructionReview: button(),",
            "  captionDownloadInstructionReport: button(),",
            "  captionBatchCancel: button(),",
            "  captionResumeBackendJob: button(),",
            "  captionRecipeLoad: button(),",
            "  captionRecipeUploadButton: button(),",
            "  captionRecipeUpload: button(),",
            "  captionPromptUser: button(),",
            "  captionStyleText: button(),",
            "  captionMode: button(),",
            "  captionModel: button(),",
            "  captionMaxTokens: button(),",
            "  captionSetAndForget: button(),",
            "  captionBatchCount: button(),",
            "  captionBatchOverwrite: button(),",
            "  captionGlossary: { disabled: false, value: 'stable glossary' },",
            "  captionGlossaryReset: button(),",
            "  captionGlossarySave: button(),",
            "  captionSubcaptionsPerImage: button(),",
            "  captionQaMix: button(),",
            "  captionAnswerFormat: button(),",
            "  captionIncludeCaption0Training: button(),",
            "  captionIncludeGeneratedQaTraining: button(),",
            "  captionIncludeDeterministicMetadataQa: button(),",
            "  captionIncludeSourceAnnotationsContext: button(),",
            "  captionStrictGrounding: button(),",
            "  captionRestrictSpeculativeQaLanguage: button(),",
            "  captionOutput: { value: 'caption text' },",
            "  captionSaveAlternate: button(),",
            "  captionUpdateSelected: button(),",
            "  captionSetPrimary: button(),",
            "  captionDeleteSelected: button(),",
            "};",
            "const qwenCaptionGlossaryState = { saveInFlight: false };",
            "function getCaptionGlossaryDatasetId() { return 'ds'; }",
            _extract_js_function(js, "qwenCaptionArchiveMutationActive"),
            _extract_js_function(js, "captionInstructionArtifactBusyMessage"),
            _extract_js_function(js, "getCaptionPromptStackEditors"),
            _extract_js_function(js, "captionRunConfigurationElements"),
            _extract_js_function(js, "updateCaptionRunConfigurationControls"),
            _extract_js_function(js, "captionOutputEditingBlocked"),
            _extract_js_function(js, "updateCaptionOutputEditControl"),
            _extract_js_function(js, "updateCaptionGlossaryControls"),
            _extract_js_function(js, "updateCaptionInstructionDatasetOptionControls"),
            _extract_js_function(js, "updateCaptionArchiveActionControls"),
            _extract_js_function(js, "updateQwenCaptionButton"),
            "assert.strictEqual(qwenCaptionArchiveMutationActive(), true);",
            "assert(captionInstructionArtifactBusyMessage('exporting instruction rows').includes('instruction archive is changing'));",
            "updateQwenCaptionButton();",
            "assert.strictEqual(qwenElements.captionRunButton.disabled, true);",
            "assert.strictEqual(qwenElements.captionDownloadJsonl.disabled, true);",
            "assert.strictEqual(qwenElements.captionDownloadGroupedJson.disabled, true);",
            "assert.strictEqual(qwenElements.captionDownloadVlmJsonl.disabled, true);",
            "assert.strictEqual(qwenElements.captionBuildInstructionDataset.disabled, true);",
            "assert.strictEqual(qwenElements.captionDownloadInstructionBundle.disabled, true);",
            "assert.strictEqual(qwenElements.captionDownloadInstructionJsonl.disabled, true);",
            "assert.strictEqual(qwenElements.captionDownloadInstructionArchive.disabled, true);",
            "assert.strictEqual(qwenElements.captionDownloadInstructionReview.disabled, true);",
            "assert.strictEqual(qwenElements.captionImportInstructionReview.disabled, true);",
            "assert.strictEqual(qwenElements.captionDownloadInstructionReport.disabled, true);",
            "assert.strictEqual(qwenElements.captionResumeBackendJob.disabled, true);",
            "assert.strictEqual(qwenElements.captionCancelButton.disabled, false);",
            "assert.strictEqual(qwenElements.captionBatchCancel.disabled, false);",
            "assert.strictEqual(qwenElements.captionOutput.disabled, true);",
            "assert.strictEqual(qwenElements.captionSaveAlternate.disabled, true);",
            "assert.strictEqual(qwenElements.captionUpdateSelected.disabled, true);",
            "assert.strictEqual(qwenElements.captionSetPrimary.disabled, true);",
            "assert.strictEqual(qwenElements.captionDeleteSelected.disabled, true);",
            "assert.strictEqual(qwenElements.captionSubcaptionsPerImage.disabled, true);",
            "assert.strictEqual(qwenElements.captionRecipeLoad.disabled, true);",
            "assert.strictEqual(qwenElements.captionRecipeUploadButton.disabled, true);",
            "assert.strictEqual(qwenElements.captionRecipeUpload.disabled, true);",
            "assert.strictEqual(qwenElements.captionPromptUser.disabled, true);",
            "assert.strictEqual(qwenElements.captionStyleText.disabled, true);",
            "assert.strictEqual(qwenElements.captionMode.disabled, true);",
            "assert.strictEqual(qwenElements.captionModel.disabled, true);",
            "assert.strictEqual(qwenElements.captionMaxTokens.disabled, true);",
            "assert.strictEqual(qwenElements.captionSetAndForget.disabled, true);",
            "assert.strictEqual(qwenElements.captionBatchCount.disabled, true);",
            "assert.strictEqual(qwenElements.captionBatchOverwrite.disabled, true);",
            "assert.strictEqual(qwenElements.captionGlossary.disabled, true);",
            "assert.strictEqual(qwenElements.captionGlossaryReset.disabled, true);",
            "assert.strictEqual(qwenElements.captionGlossarySave.disabled, true);",
            "assert.strictEqual(qwenElements.captionQaMix.disabled, true);",
            "assert.strictEqual(qwenElements.captionAnswerFormat.disabled, true);",
            "assert.strictEqual(qwenElements.captionIncludeCaption0Training.disabled, true);",
            "assert.strictEqual(qwenElements.captionIncludeGeneratedQaTraining.disabled, true);",
            "assert.strictEqual(qwenElements.captionIncludeDeterministicMetadataQa.disabled, true);",
            "assert.strictEqual(qwenElements.captionIncludeSourceAnnotationsContext.disabled, true);",
            "assert.strictEqual(qwenElements.captionStrictGrounding.disabled, true);",
            "assert.strictEqual(qwenElements.captionRestrictSpeculativeQaLanguage.disabled, true);",
            "qwenCaptionBatchBackendJobId = '';",
            "assert.strictEqual(qwenCaptionArchiveMutationActive(), false);",
            "assert.strictEqual(captionInstructionArtifactBusyMessage('exporting instruction rows'), '');",
            "updateQwenCaptionButton();",
            "assert.strictEqual(qwenElements.captionRunButton.disabled, false);",
            "assert.strictEqual(qwenElements.captionDownloadJsonl.disabled, false);",
            "assert.strictEqual(qwenElements.captionDownloadGroupedJson.disabled, false);",
            "assert.strictEqual(qwenElements.captionDownloadVlmJsonl.disabled, false);",
            "assert.strictEqual(qwenElements.captionBuildInstructionDataset.disabled, false);",
            "assert.strictEqual(qwenElements.captionDownloadInstructionBundle.disabled, false);",
            "assert.strictEqual(qwenElements.captionDownloadInstructionJsonl.disabled, false);",
            "assert.strictEqual(qwenElements.captionDownloadInstructionArchive.disabled, false);",
            "assert.strictEqual(qwenElements.captionDownloadInstructionReview.disabled, false);",
            "assert.strictEqual(qwenElements.captionImportInstructionReview.disabled, false);",
            "assert.strictEqual(qwenElements.captionDownloadInstructionReport.disabled, false);",
            "assert.strictEqual(qwenElements.captionResumeBackendJob.disabled, false);",
            "assert.strictEqual(qwenElements.captionCancelButton.disabled, true);",
            "assert.strictEqual(qwenElements.captionBatchCancel.disabled, true);",
            "assert.strictEqual(qwenElements.captionOutput.disabled, false);",
            "assert.strictEqual(qwenElements.captionSaveAlternate.disabled, false);",
            "assert.strictEqual(qwenElements.captionUpdateSelected.disabled, false);",
            "assert.strictEqual(qwenElements.captionSetPrimary.disabled, false);",
            "assert.strictEqual(qwenElements.captionDeleteSelected.disabled, false);",
            "assert.strictEqual(qwenElements.captionSubcaptionsPerImage.disabled, false);",
            "assert.strictEqual(qwenElements.captionRecipeLoad.disabled, false);",
            "assert.strictEqual(qwenElements.captionRecipeUploadButton.disabled, false);",
            "assert.strictEqual(qwenElements.captionRecipeUpload.disabled, false);",
            "assert.strictEqual(qwenElements.captionPromptUser.disabled, false);",
            "assert.strictEqual(qwenElements.captionStyleText.disabled, false);",
            "assert.strictEqual(qwenElements.captionMode.disabled, false);",
            "assert.strictEqual(qwenElements.captionModel.disabled, false);",
            "assert.strictEqual(qwenElements.captionMaxTokens.disabled, false);",
            "assert.strictEqual(qwenElements.captionSetAndForget.disabled, false);",
            "assert.strictEqual(qwenElements.captionBatchCount.disabled, false);",
            "assert.strictEqual(qwenElements.captionBatchOverwrite.disabled, false);",
            "assert.strictEqual(qwenElements.captionGlossary.disabled, false);",
            "assert.strictEqual(qwenElements.captionGlossaryReset.disabled, false);",
            "assert.strictEqual(qwenElements.captionGlossarySave.disabled, false);",
            "assert.strictEqual(qwenElements.captionQaMix.disabled, false);",
            "assert.strictEqual(qwenElements.captionAnswerFormat.disabled, false);",
            "assert.strictEqual(qwenElements.captionIncludeCaption0Training.disabled, false);",
            "assert.strictEqual(qwenElements.captionIncludeGeneratedQaTraining.disabled, false);",
            "assert.strictEqual(qwenElements.captionIncludeDeterministicMetadataQa.disabled, false);",
            "assert.strictEqual(qwenElements.captionIncludeSourceAnnotationsContext.disabled, false);",
            "assert.strictEqual(qwenElements.captionStrictGrounding.disabled, false);",
            "assert.strictEqual(qwenElements.captionRestrictSpeculativeQaLanguage.disabled, false);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_dual_bbox_cards_use_human_controlled_annotation_actions():
    js = JS_PATH.read_text(encoding="utf-8")
    css = CSS_PATH.read_text(encoding="utf-8")
    local_pair_review_at = js.index(
        "function applyClassSplitDualBBoxReviewLocally"
    )
    local_pair_review = js[
        local_pair_review_at:js.index(
            "function applyClassSplitOptimisticDualBBoxReview",
            local_pair_review_at,
        )
    ]

    assert "CLASS_SPLIT_DUAL_BBOX_DISPOSITIONS" in js
    assert '"delete_current_box"' in js
    assert '"delete_overlapping_box"' in js
    assert '"keep_both_boxes"' in js
    assert '"unresolved"' in js
    assert "function getClassSplitDualBBoxContract" in js
    assert "rawConflict.point_id || safePointId" in js
    assert "rawConflict.current_class || point.class_name" in js
    assert "rawConflict.target_bbox_xyxy || point.bbox_xyxy" in js
    assert "function ensureClassSplitDualBBoxExactMatch" in js
    assert "matches.length !== 1" in js
    assert "function ensureClassSplitDualBBoxPairExactMatches" in js
    assert "function classSplitDualBBoxLocalImportCoordinates" in js
    assert "function getClassSplitDualBBoxDeletionMode" in js
    assert 'deletionMode === "local_workspace"' in js
    assert "Press Shift+Y to export the updated labels" in js
    assert "current.imageKey !== other.imageKey" in js
    assert "current.match.bbox === other.match.bbox" in js
    assert "function commitClassSplitDualBBoxDeletionTransaction" in js
    assert "function resolveClassSplitDualBBox" in js
    assert "function dismissClassSplitStaleDualBBoxPair" in js
    assert "function markClassSplitDualBBoxResolved" in js
    assert 'enqueueTaskNotice("BBox not found."' in js
    assert "expected_record_revision: expectedRecordRevision" in js
    assert "expected_source_identity: expectedSourceIdentity" in js
    assert "annotation_commit_attestation" in js
    assert "Save this image first." in js
    assert "dualBBoxTransactionInFlight: false" in js
    assert "|| classSplitState.dualBBoxTransactionInFlight" in js
    assert "Deletion commit state unknown; reload required." in js
    assert "stopAnnotationTimers();" in js
    assert "function getClassSplitDualBBoxParticipantIds" in js
    assert "const blockedDualIds = selectedIds.filter" in js
    assert "Resolve each pair with its dedicated box controls." in js
    assert "This object belongs to an overlapping-box pair." in js
    assert "excludeBbox: exactTarget.match.bbox" in js
    assert "const liveIndex = bucket.indexOf(match.bbox)" in js
    assert "requestPayload.dual_bbox_conflict" in js
    assert "selectedBboxes.delete(bboxUuid)" in js
    assert "negativeBboxes.delete(bboxUuid)" in js
    assert "Delete ${escapeHtml(currentClass" in js
    assert "Delete ${escapeHtml(dualOtherClass" in js
    assert "Keep both boxes" in js
    assert "Leave unresolved" in js
    assert "Mark resolved" in js
    assert 'data-action="resolve-dual-bbox"' in js
    assert js.count('data-action="mark-dual-bbox-resolved"') >= 2
    assert js.count(
        'data-pair-point-id="${escapeHtml(dualOtherPointId)}"'
    ) >= 8
    assert "runClassSplitAcknowledgedAction(button, () => (\n                    resolveClassSplitDualBBox" in js
    assert "only you can apply a deletion" in js
    assert "A saved box deletion cannot be restored" in js
    assert "Possible duplicate boxes:" in js
    assert "VLM advisory:" in js
    assert "Advisory only — no box was changed" in js
    assert "class-split-dual-bbox-actions" in css
    assert ".training-button:disabled" in css
    assert "class-split-dual-bbox-action--recommended" in css
    assert "renderClassSplitReviewedList();" in local_pair_review
    resolver = js[
        js.index("async function resolveClassSplitDualBBox"):
        js.index("function updateClassSplitSummaryClassCounts")
    ]
    assert "window.confirm" not in resolver
    assert "from this open workspace?" not in resolver
    assert "flushAnnotationSnapshot" not in resolver
    assert resolver.index("findClassSplitExactGeometryMatches(target)") < resolver.index(
        'annotationEditableGuard("Deleting overlapping annotation")'
    )
    assert resolver.index("commitClassSplitDualBBoxDeletionTransaction") < resolver.index(
        "applyClassSplitDualBBoxTransactionLocally"
    )
    barrier_on = resolver.index("classSplitState.dualBBoxTransactionInFlight = true;")
    disposition_save = resolver.index("saveClassSplitReviewDisposition(", barrier_on)
    barrier_off = resolver.index(
        "classSplitState.dualBBoxTransactionInFlight = false;",
        disposition_save,
    )
    assert barrier_on < disposition_save < barrier_off
    assert resolver.index("syncLabelingSourceControls();", barrier_on) < disposition_save
    assert resolver.index("syncLabelingSourceControls();", barrier_off) > barrier_off


def test_data_quality_vignette_actions_acknowledge_before_background_persistence():
    js = JS_PATH.read_text(encoding="utf-8")
    css = CSS_PATH.read_text(encoding="utf-8")
    save_disposition = _extract_js_function(js, "saveClassSplitReviewDisposition")
    class_change = js[
        js.index("async function changeClassSplitPointClass"):
        js.index("function initClassSplitExplorer")
    ]
    acknowledgement = js[
        js.index("function runClassSplitAcknowledgedAction"):
        js.index("function scheduleClassSplitBackgroundReviewRefresh")
    ]

    assert "if (!deferUi)" in save_disposition
    assert "refreshClassSplitControls" in save_disposition
    assert "renderClassSplitWrongList" not in save_disposition
    assert 'button.classList.add("class-split-review-action", "is-acknowledged")' in acknowledgement
    assert "Promise.resolve()" in acknowledgement
    assert "}, 300);" not in acknowledgement
    assert "applyClassSplitOptimisticReview" in js
    assert "restoreClassSplitOptimisticReview" in js
    assert "scheduleClassSplitBackgroundReviewRefresh" in js
    assert "enqueueTaskNotice" in class_change
    assert class_change.index("renderClassSplitWrongList();") < class_change.index(
        "await captureClassSplitTrainingAction("
    )
    assert "refreshClassSplitControls" not in class_change
    assert 'data-action="delete-bbox"' in js
    assert "async function deleteClassSplitPointBbox" in js
    assert "findClassSplitExactGeometryMatches(point)" in js
    assert "ensureClassSplitDualBBoxPairExactMatches(contract)" in js
    assert ".class-split-review-action.is-acknowledged" in css
    assert "position: fixed;" in css[css.index(".task-queue {"):css.index(".task-queue.visible")]


def test_graph_review_dismissal_and_sparse_pair_contract_are_immediate():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert').strict;",
            _extract_js_function(js, "classSplitDualBBoxCoordinates"),
            _extract_js_function(js, "classSplitDualBBoxCoordinatesMatch"),
            "assert.equal(classSplitDualBBoxCoordinatesMatch([921, 918, 946, 958], [921, 917.99952, 946.00032, 957.99984]), true);",
            "assert.equal(classSplitDualBBoxCoordinatesMatch([921, 918, 946, 958], [921, 917.98, 946, 958]), false);",
            "const point = {point_id: 'p', class_name: 'LightVehicle', bbox_xyxy: [1, 2, 11, 22], split: 'train', image_relpath: 'image.jpg', frontend_image_key: 'train:image.jpg'};",
            "const rawConflict = {enabled: true, review_mode: 'dual_bbox_annotation_resolution', other_point_id: 'q', other_class_name: 'Bike', other_bbox_xyxy: [2, 3, 12, 23], split: 'train', image_relpath: 'image.jpg', pair_review_key: 'crp_' + '1'.repeat(64), current_review_object_key: 'cro_' + '2'.repeat(64), other_review_object_key: 'cro_' + '3'.repeat(64)};",
            "function getClassSplitPointById(pointId) { return pointId === 'p' ? point : null; }",
            "function getClassSplitCandidateByPointId() { return {}; }",
            "function getClassSplitDualBBoxConflict() { return rawConflict; }",
            _extract_js_function(js, "getClassSplitDualBBoxContract"),
            "const contract = getClassSplitDualBBoxContract('p');",
            "assert.ok(contract);",
            "assert.equal(contract.conflict.point_id, 'p');",
            "assert.equal(contract.conflict.current_class, 'LightVehicle');",
            "assert.deepEqual(contract.conflict.target_bbox_xyxy, point.bbox_xyxy);",
            "assert.equal(contract.conflict.pair_review_key, rawConflict.pair_review_key);",
            "const classSplitState = {dismissedWrongIds: new Set(), result: {points: [{point_id: 'p'}, {point_id: 'q'}, {point_id: 'keep'}]}};",
            "const classSplitElements = {filterClass: null};",
            "const removed = []; const notices = []; let persisted = 0;",
            "function removeClassSplitPointFromActiveReviewGraph(pointId, options) { removed.push([pointId, options.force]); }",
            "function syncClassSplitWrongCandidateSummaryCount() {}",
            "function renderClassSplitWrongList() {}",
            "function persistDataQualityExplorerSession() { persisted += 1; return true; }",
            "function enqueueTaskNotice(message) { notices.push(message); }",
            "function classSplitReviewToastKey() { return 'toast'; }",
            "function setClassSplitJobStatus() {}",
            "function scheduleClassSplitBackgroundReviewRefresh() {}",
            _extract_js_function(js, "dismissClassSplitStaleDualBBoxPair"),
            _extract_js_function(js, "markClassSplitDualBBoxResolved"),
            _extract_js_function(js, "getClassSplitFilteredPoints"),
            "dismissClassSplitStaleDualBBoxPair(contract);",
            "assert.deepEqual([...classSplitState.dismissedWrongIds].sort(), ['p', 'q']);",
            "assert.deepEqual(removed, [['p', true], ['q', true]]);",
            "assert.deepEqual(notices, ['BBox not found.']);",
            "assert.equal(persisted, 1);",
            "assert.deepEqual(getClassSplitFilteredPoints().map((item) => item.point_id), ['keep']);",
            "classSplitState.dismissedWrongIds.clear(); removed.length = 0; notices.length = 0;",
            "markClassSplitDualBBoxResolved('p');",
            "assert.deepEqual([...classSplitState.dismissedWrongIds].sort(), ['p', 'q']);",
            "assert.deepEqual(removed, [['p', true], ['q', true]]);",
            "assert.deepEqual(notices, ['Marked resolved.']);",
            "assert.equal(persisted, 2);",
            "assert.deepEqual(getClassSplitFilteredPoints().map((item) => item.point_id), ['keep']);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_class_split_projection_setting_diffs_compare_live_controls_to_run_metadata():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert').strict;",
            "const classSplitState = {",
            "  capabilities: {",
            "    default_projection_metric: 'cosine', default_projection_spread: 1,",
            "    default_projection_preprocess: 'none',",
            "    umap_projection_metrics: ['cosine', 'euclidean'],",
            "    projection_preprocess_modes: ['none', 'center', 'zscore'],",
            "  },",
            "  lastRequest: {},",
            "  result: {summary: {projection_metadata: {requested: {",
            "    projection_preprocess: 'none', projection_metric: 'cosine',",
            "    projection_neighbor_k: 50, projection_min_dist: 0.08, projection_spread: 1,",
            "  }}}},",
            "};",
            "const classSplitElements = {",
            "  projectionPreprocess: {value: 'none'}, projectionMetric: {value: 'cosine'},",
            "  projectionNeighborK: {value: '50'}, projectionMinDist: {value: '0.08'},",
            "  projectionSpread: {value: '1'},",
            "};",
            _extract_js_function(js, "getClassSplitUmapProjectionMetrics"),
            _extract_js_function(js, "getClassSplitProjectionPreprocessModes"),
            _extract_js_function(js, "normalizeClassSplitProjectionMetric"),
            _extract_js_function(js, "normalizeClassSplitProjectionPreprocess"),
            _extract_js_function(js, "getClassSplitDefaultProjectionMetric"),
            _extract_js_function(js, "getClassSplitDefaultProjectionSpread"),
            _extract_js_function(js, "getClassSplitDefaultProjectionPreprocess"),
            _extract_js_function(js, "normalizeClassSplitProjectionChoice"),
            _extract_js_function(js, "classSplitProjectionSettingValueMatch"),
            _extract_js_function(js, "getClassSplitProjectionSettingDiffs"),
            "assert.deepEqual(getClassSplitProjectionSettingDiffs('umap'), []);",
            "classSplitElements.projectionSpread.value = '2';",
            "assert.deepEqual(getClassSplitProjectionSettingDiffs('umap'), ['spread']);",
            "classSplitElements.projectionSpread.value = '1';",
            "classSplitElements.projectionNeighborK.value = '0';",
            "assert.deepEqual(getClassSplitProjectionSettingDiffs('umap'), ['neighbors']);",
            "classSplitElements.projectionNeighborK.value = '50';",
            "classSplitElements.projectionPreprocess.value = 'center';",
            "assert.deepEqual(getClassSplitProjectionSettingDiffs('global_pca'), ['projection preprocessing']);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_data_quality_acknowledged_actions_reserve_mirrors_and_dual_pair():
    js = JS_PATH.read_text(encoding="utf-8")
    acknowledged = _extract_js_function(js, "runClassSplitAcknowledgedAction")
    script = "\n".join(
        [
            "const assert = require('assert').strict;",
            "class FakeClassList {",
            "  constructor() { this.values = new Set(); }",
            "  add(...tokens) { tokens.forEach((token) => this.values.add(token)); }",
            "  remove(...tokens) { tokens.forEach((token) => this.values.delete(token)); }",
            "  contains(token) { return this.values.has(token); }",
            "}",
            "class FakeButton {",
            "  constructor(pointId, pairPointId) {",
            "    this.attributes = new Map([['data-point-id', pointId], ['data-pair-point-id', pairPointId]]);",
            "    this.disabled = false;",
            "    this.textContent = 'Apply';",
            "    this.dataset = {};",
            "    this.classList = new FakeClassList();",
            "    this.isConnected = true;",
            "  }",
            "  getAttribute(name) { return this.attributes.get(name) || null; }",
            "  closest() { return null; }",
            "}",
            "const HTMLButtonElement = FakeButton;",
            "const scheduled = [];",
            "const window = {",
            "  setTimeout(callback, delay) { scheduled.push({callback, delay}); return scheduled.length; },",
            "};",
            "const classSplitState = {",
            "  currentJobId: 'job-1',",
            "  analysisGeneration: 7,",
            "  reviewActionPendingPointIds: new Set(),",
            "  selectedPointId: 'pair-b',",
            "};",
            "function classSplitAsyncRequestIsCurrent(generation, jobId) {",
            "  return generation === 7 && jobId === 'job-1';",
            "}",
            "function classSplitMutationIsBusy() { return false; }",
            "let historyDeleteActive = false;",
            "function classSplitReviewHistoryDeleteOperation() { return historyDeleteActive ? {} : null; }",
            "let recoveryRenders = 0;",
            "let inspectorRenders = 0;",
            "let reviewedListRenders = 0;",
            "let controlRefreshes = 0;",
            "function renderClassSplitPendingReviewRecovery() { recoveryRenders += 1; }",
            "function renderClassSplitWrongList() { throw new Error('unexpected list render'); }",
            "function renderClassSplitInspector() { inspectorRenders += 1; }",
            "function renderClassSplitReviewedList() { reviewedListRenders += 1; }",
            "function refreshClassSplitControls() { controlRefreshes += 1; }",
            "function updateClassSplitMultiSelectionControls() {}",
            "function enqueueTaskNotice() {}",
            "function setSamStatus() {}",
            acknowledged,
            "(async () => {",
            "  const first = new FakeButton('pair-a', 'pair-b');",
            "  const mirror = new FakeButton('pair-b', 'pair-a');",
            "  const calls = [];",
            "  runClassSplitAcknowledgedAction(first, () => { calls.push('first'); });",
            "  assert.equal(first.disabled, true);",
            "  assert.equal(first.classList.contains('is-acknowledged'), true);",
            "  assert.deepEqual([...classSplitState.reviewActionPendingPointIds].sort(), ['pair-a', 'pair-b']);",
            "  assert.equal(scheduled.length, 0);",
            "  assert.deepEqual(calls, []);",
            "  runClassSplitAcknowledgedAction(mirror, () => { calls.push('mirror-raced'); });",
            "  assert.equal(scheduled.length, 0);",
            "  assert.deepEqual(calls, []);",
            "  assert.equal(recoveryRenders, 0);",
            "  assert.equal(inspectorRenders, 0);",
            "  await new Promise((resolve) => setImmediate(resolve));",
            "  assert.equal(scheduled.length, 1);",
            "  assert.equal(scheduled[0].delay, 280);",
            "  scheduled.shift().callback();",
            "  await new Promise((resolve) => setImmediate(resolve));",
            "  assert.deepEqual(calls, ['first']);",
            "  assert.deepEqual([...classSplitState.reviewActionPendingPointIds], []);",
            "  assert.equal(recoveryRenders, 1);",
            "  assert.equal(inspectorRenders, 1);",
            "  assert.equal(reviewedListRenders, 2);",
            "  assert.equal(controlRefreshes, 2);",
            "  assert.equal(first.disabled, false);",
            "  runClassSplitAcknowledgedAction(mirror, () => { calls.push('mirror-after'); });",
            "  await new Promise((resolve) => setImmediate(resolve));",
            "  assert.equal(scheduled.length, 1);",
            "  scheduled.shift().callback();",
            "  await new Promise((resolve) => setImmediate(resolve));",
            "  assert.deepEqual(calls, ['first', 'mirror-after']);",
            "  assert.deepEqual([...classSplitState.reviewActionPendingPointIds], []);",
            "  assert.equal(reviewedListRenders, 4);",
            "  assert.equal(controlRefreshes, 4);",
            "  historyDeleteActive = true;",
            "  runClassSplitAcknowledgedAction(first, () => { calls.push('during-delete'); });",
            "  assert.equal(scheduled.length, 0);",
            "  assert.deepEqual(calls, ['first', 'mirror-after']);",
            "})().catch((error) => { console.error(error); process.exitCode = 1; });",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_annotation_snapshot_receipts_refresh_image_cas_contracts():
    js = JS_PATH.read_text(encoding="utf-8")
    snapshot = js[
        js.index("async function flushAnnotationSnapshot"):
        js.index("async function annotationHeartbeatTick")
    ]

    assert "const savedContractByKey = new Map();" in snapshot
    assert "savedContract.annotation_record_revision" in snapshot
    assert "savedContract.annotation_source_identity" in snapshot


def test_dual_bbox_pair_binding_requires_two_distinct_exact_annotations():
    js = JS_PATH.read_text(encoding="utf-8")
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const current = {uuid: 'current', x: 10, y: 20, width: 40, height: 60};",
            "const other = {uuid: 'other', x: 10.2, y: 20.1, width: 39.9, height: 60.1};",
            "const images = {image: {sourceType: 'dataset'}};",
            "const bboxes = {image: {Truck: [current], Person: [other]}};",
            "const annotationSourceState = {hydratedKeys: new Set(['image'])};",
            "function getClassSplitPointImageKey() { return 'image'; }",
            "function isDatasetBackedImageRecord(record) { return record?.sourceType === 'dataset'; }",
            "function hydrateDatasetBboxesForImage() { return true; }",
            "function getBboxBounds(bbox) { return {left: bbox.x, top: bbox.y, right: bbox.x + bbox.width, bottom: bbox.y + bbox.height}; }",
            _extract_js_function(js, "classSplitDualBBoxCoordinates"),
            _extract_js_function(js, "classSplitDualBBoxCoordinatesMatch"),
            _extract_js_function(js, "findClassSplitExactGeometryMatches"),
            _extract_js_function(js, "ensureClassSplitDualBBoxExactMatch"),
            _extract_js_function(js, "ensureClassSplitDualBBoxPairExactMatches"),
            "const contract = {",
            "  currentTarget: {class_name: 'Truck', bbox_xyxy: [10, 20, 50, 80]},",
            "  otherTarget: {class_name: 'Person', bbox_xyxy: [10.2, 20.1, 50.1, 80.2]},",
            "};",
            "const pair = ensureClassSplitDualBBoxPairExactMatches(contract);",
            "assert.strictEqual(pair.current.match.bbox, current);",
            "assert.strictEqual(pair.other.match.bbox, other);",
            "bboxes.image.Person = [];",
            "assert.throws(() => ensureClassSplitDualBBoxPairExactMatches(contract), /no longer exists|no exact box matches/);",
            "bboxes.image.Person = [current];",
            "contract.otherTarget.bbox_xyxy = [10, 20, 50, 80];",
            "assert.throws(() => ensureClassSplitDualBBoxPairExactMatches(contract), /same annotation/);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_dual_bbox_local_workspace_binding_uses_yolo_import_quantization():
    js = JS_PATH.read_text(encoding="utf-8")
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const current = {uuid: 'current', x: 557, y: 720, width: 76, height: 31};",
            "const other = {uuid: 'other', x: 554, y: 721, width: 76, height: 30};",
            "const images = {image: {sourceType: 'local_file'}};",
            "const bboxes = {image: {Truck: [current], LightVehicle: [other]}};",
            "const annotationSourceState = {hydratedKeys: new Set()};",
            "function getClassSplitPointImageKey() { return 'image'; }",
            "function isDatasetBackedImageRecord(record) { return record?.sourceType === 'dataset'; }",
            "function hydrateDatasetBboxesForImage() { return true; }",
            "function getBboxBounds(bbox) { return {left: bbox.x, top: bbox.y, right: bbox.x + bbox.width, bottom: bbox.y + bbox.height}; }",
            _extract_js_function(js, "classSplitDualBBoxCoordinates"),
            _extract_js_function(js, "classSplitDualBBoxCoordinatesMatch"),
            _extract_js_function(js, "classSplitDualBBoxLocalImportCoordinates"),
            _extract_js_function(js, "findClassSplitExactGeometryMatches"),
            _extract_js_function(js, "ensureClassSplitDualBBoxExactMatch"),
            _extract_js_function(js, "ensureClassSplitDualBBoxPairExactMatches"),
            "const contract = {",
            "  currentTarget: {class_name: 'Truck', bbox_xyxy: [557.00016, 720, 633.00048, 751.00032]},",
            "  otherTarget: {class_name: 'LightVehicle', bbox_xyxy: [554.99952, 721.00032, 630.99984, 751.00032]},",
            "};",
            "const pair = ensureClassSplitDualBBoxPairExactMatches(contract);",
            "assert.strictEqual(pair.current.match.bbox, current);",
            "assert.strictEqual(pair.other.match.bbox, other);",
            "bboxes.image.Truck.push({...current, uuid: 'ambiguous'});",
            "assert.throws(() => ensureClassSplitDualBBoxPairExactMatches(contract), /more than one exact box matches/);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_vignette_bbox_uuid_binding_survives_geometry_edit_and_relabel():
    js = JS_PATH.read_text(encoding="utf-8")
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const CLASS_SPLIT_BBOX_GEOMETRY_EDIT_SCHEMA = 'class-analysis-bbox-geometry-edit-v1';",
            "const live = {uuid: 'stable-box', x: 10, y: 20, width: 40, height: 60, class: 'Truck'};",
            "const images = {image: {sourceType: 'dataset'}};",
            "const bboxes = {image: {Truck: [live], Building: [], Person: []}};",
            "const annotationSourceState = {hydratedKeys: new Set(['image'])};",
            "const loadedClassList = ['Truck', 'Building', 'Person'];",
            "const classSplitState = {selectedClusterId: ''};",
            "let currentImage = {name: 'image'};",
            "let currentBbox = {bbox: live};",
            "let currentClass = 'Truck';",
            "function getClassSplitPointImageKey() { return 'image'; }",
            "function resolveClassSplitPointImageKey() { return 'image'; }",
            "function isDatasetBackedImageRecord(record) { return record?.sourceType === 'dataset'; }",
            "function hydrateDatasetBboxesForImage() { return true; }",
            "function getBboxBounds(bbox) { return {left: bbox.x, top: bbox.y, right: bbox.x + bbox.width, bottom: bbox.y + bbox.height}; }",
            "function generateUUID() { return 'generated-box'; }",
            "function getClassSplitDualBBoxParticipantIds() { return new Set(); }",
            "function clearClassSplitWrongCandidate() {}",
            "function updateClassSplitSummaryClassCounts() {}",
            "function selectClassListOptionByName() {}",
            "function syncQwenClassToCurrent() {}",
            "function updateSam3ClassOptions() {}",
            "function syncClassSplitCurrentClassSelection() {}",
            "function markOnlyBboxRecord() {}",
            "function captureAnnotationDirtyStateForImage() {}",
            "function scheduleAnnotationDiversityMetricRefresh() {}",
            "function clearClassSplitDatasetAnalysis() {}",
            _extract_js_function(js, "classSplitDualBBoxCoordinates"),
            _extract_js_function(js, "classSplitDualBBoxCoordinatesMatch"),
            _extract_js_function(js, "findClassSplitBboxMatch"),
            _extract_js_function(js, "bindClassSplitPointBboxIdentity"),
            _extract_js_function(js, "ensureClassSplitPointBboxMatch"),
            _extract_js_function(js, "buildClassSplitBboxGeometryEdit"),
            _extract_js_function(js, "applyClassSplitPointClassLocally"),
            "const point = {point_id: 'p1', class_name: 'Truck', bbox_xyxy: [10, 20, 50, 80]};",
            "const initial = ensureClassSplitPointBboxMatch(point);",
            "assert.strictEqual(initial.match.bbox, live);",
            "assert.strictEqual(point._resolved_frontend_bbox_uuid, 'stable-box');",
            "assert.strictEqual(point._resolved_frontend_bbox_image_key, 'image');",
            "assert.deepStrictEqual(point._resolved_frontend_bbox_original_xyxy, [10, 20, 50, 80]);",
            "live.x = 100; live.y = 40; live.width = 70; live.height = 55;",
            "const moved = findClassSplitBboxMatch(point);",
            "assert.strictEqual(moved.identity, 'uuid');",
            "assert.strictEqual(moved.bbox, live);",
            "const edit = buildClassSplitBboxGeometryEdit(point, moved.bbox);",
            "assert.deepStrictEqual(edit.analysis_bbox_xyxy, [10, 20, 50, 80]);",
            "assert.deepStrictEqual(edit.edited_bbox_xyxy, [100, 40, 170, 95]);",
            "assert.strictEqual(edit.changed, true);",
            "assert.strictEqual(applyClassSplitPointClassLocally(point, 'Building', {resolvedMatch: {imageKey: 'image', match: moved}}), true);",
            "assert.deepStrictEqual(bboxes.image.Truck, []);",
            "assert.deepStrictEqual(bboxes.image.Building, [live]);",
            "assert.strictEqual(live.class, 'Building');",
            "assert.strictEqual(point.class_name, 'Building');",
            "assert.strictEqual(findClassSplitBboxMatch(point).bbox, live);",
            "bboxes.image.Person.push({uuid: 'stable-box', x: 1, y: 1, width: 5, height: 5});",
            "assert.strictEqual(findClassSplitBboxMatch(point), null);",
            "bboxes.image.Person = [];",
            "bboxes.image.Building = [];",
            "bboxes.image.Truck = [{uuid: 'replacement', x: 10, y: 20, width: 40, height: 60}];",
            "assert.strictEqual(findClassSplitBboxMatch(point), null);",
            "const ambiguous = {point_id: 'p2', class_name: 'Truck', bbox_xyxy: [10, 20, 50, 80]};",
            "bboxes.image.Truck = [",
            "  {uuid: 'a', x: 10, y: 20, width: 40, height: 60},",
            "  {uuid: 'b', x: 10.5, y: 20, width: 40, height: 60},",
            "];",
            "assert.strictEqual(findClassSplitBboxMatch(ambiguous), null);",
            "point._resolved_frontend_bbox_image_key = 'other-image';",
            "assert.strictEqual(findClassSplitBboxMatch(point), null);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_pending_training_commit_recovers_from_edited_geometry_after_reload():
    js = JS_PATH.read_text(encoding="utf-8")
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const CLASS_SPLIT_BBOX_GEOMETRY_EDIT_SCHEMA = 'class-analysis-bbox-geometry-edit-v1';",
            "const CLASS_SPLIT_ANNOTATION_REVISION_PATTERN = /^alr1_[0-9a-f]{64}$/;",
            "const CLASS_SPLIT_ANNOTATION_SOURCE_IDENTITY_PATTERN = /^asi1_[0-9a-f]{64}$/;",
            "const beforeRevision = 'alr1_' + '1'.repeat(64);",
            "const committedRevision = 'alr1_' + '2'.repeat(64);",
            "const sourceIdentity = 'asi1_' + '3'.repeat(64);",
            "const edited = {uuid: 'rehydrated', x: 100, y: 40, width: 70, height: 55, class: 'Building'};",
            "const frozenReplacement = {uuid: 'wrong-object', x: 10, y: 20, width: 40, height: 60, class: 'Building'};",
            "const images = {image: {sourceType: 'dataset'}};",
            "const bboxes = {image: {Truck: [], Building: [edited, frozenReplacement]}};",
            "const annotationSourceState = {",
            "  dirtyRecordsByKey: new Set(),",
            "  hydratedKeys: new Set(['image']),",
            "  imageRowsByKey: new Map([['image', {annotation_record_revision: committedRevision, annotation_source_identity: sourceIdentity}]])",
            "};",
            "const classSplitState = {currentJobId: 'job'};",
            "const point = {point_id: 'p1', class_name: 'Truck', bbox_xyxy: [10, 20, 50, 80]};",
            "function getClassSplitPointById() { return point; }",
            "function isAnnotationDatasetModeActive() { return true; }",
            "function resolveClassSplitPointImageKey() { return 'image'; }",
            "function getClassSplitPointImageKey() { return 'image'; }",
            "function isDatasetBackedImageRecord() { return true; }",
            "function hydrateDatasetBboxesForImage() { return true; }",
            "function classSplitDualBBoxLocalImportCoordinates() { return null; }",
            "function getBboxBounds(bbox) { return {left: bbox.x, top: bbox.y, right: bbox.x + bbox.width, bottom: bbox.y + bbox.height}; }",
            "function ensureClassSplitPointBboxMatch() { throw new Error('must use edited geometry'); }",
            _extract_js_function(js, "classSplitDualBBoxCoordinates"),
            _extract_js_function(js, "classSplitDualBBoxCoordinatesMatch"),
            _extract_js_function(js, "normalizeClassSplitBboxGeometryEdit"),
            _extract_js_function(js, "findClassSplitExactGeometryMatches"),
            _extract_js_function(js, "isClassSplitPendingTrainingLabelPersisted"),
            "const entry = {",
            "  jobId: 'job', pointId: 'p1', afterClass: 'Building', labelPersisted: false,",
            "  commitPayload: {",
            "    annotation_before_revision: beforeRevision,",
            "    geometry_edit: {schema: CLASS_SPLIT_BBOX_GEOMETRY_EDIT_SCHEMA, analysis_bbox_xyxy: [10, 20, 50, 80], edited_bbox_xyxy: [100, 40, 170, 95]}",
            "  }",
            "};",
            "assert.strictEqual(isClassSplitPendingTrainingLabelPersisted(entry), true);",
            "assert.strictEqual(entry.commitPayload.annotation_commit_revision, committedRevision);",
            "assert.strictEqual(entry.commitPayload.annotation_source_identity, sourceIdentity);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_legacy_dual_bbox_diagnostic_keeps_ordinary_review_controls():
    js = JS_PATH.read_text(encoding="utf-8")
    card_render = js[
        js.index("const normalReviewActions = ["):
        js.index("const thumbUrl =", js.index("const normalReviewActions = ["))
    ]
    card_layout = js[
        js.index('`<div class="class-split-wrong-item__actions">`'):
        js.index('`</div>`', js.index('`<div class="class-split-wrong-item__actions">`'))
    ]
    assert "Confirm current class" in card_render
    assert "Skip" in card_render
    assert "Mark resolved" in card_render
    assert 'data-action="mark-dual-bbox-resolved"' in card_render
    assert 'data-action="reassign-class"' in card_render
    assert "dualConflict ? dualReviewActions : normalReviewActions" in card_layout

    script = "\n".join(
        [
            "const assert = require('assert');",
            _extract_js_function(js, "classSplitDualBBoxCoordinates"),
            _extract_js_function(js, "isClassSplitDualBBoxResolutionTask"),
            _extract_js_function(js, "getClassSplitDualBBoxConflict"),
            "const legacyConflict = {",
            "  enabled: true,",
            "  review_mode: 'dual_bbox_class_resolution',",
            "  point_id: 'current',",
            "  current_class: 'Bike',",
            "  other_point_id: 'other',",
            "  other_class_name: 'Person',",
            "};",
            "const legacyCandidate = {point_id: 'current', class_name: 'Bike', dual_bbox_conflict: legacyConflict};",
            "const canonicalConflict = {",
            "  ...legacyConflict,",
            "  review_mode: 'dual_bbox_annotation_resolution',",
            "  target_bbox_xyxy: [10, 20, 50, 80],",
            "  other_bbox_xyxy: [10, 20, 50, 80],",
            "};",
            "const canonicalCandidate = {point_id: 'current', class_name: 'Bike', dual_bbox_conflict: canonicalConflict};",
            "assert.strictEqual(getClassSplitDualBBoxConflict(legacyCandidate, null), null);",
            "assert.strictEqual(getClassSplitDualBBoxConflict(canonicalCandidate, null), canonicalConflict);",
            "const rawCandidates = [legacyCandidate];",
            "const classSplitState = {pointsById: new Map()};",
            "function getClassSplitRawCandidates() { return rawCandidates; }",
            _extract_js_function(js, "getClassSplitDualBBoxParticipantIds"),
            "assert.deepStrictEqual([...getClassSplitDualBBoxParticipantIds()], []);",
            "rawCandidates.splice(0, 1, canonicalCandidate);",
            "assert.deepStrictEqual([...getClassSplitDualBBoxParticipantIds()].sort(), ['current', 'other']);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_caption_archive_loads_defer_while_backend_job_mutates_archive():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "let qwenCaptionActive = false;",
            "let qwenCaptionBatchActive = false;",
            "let qwenCaptionBatchBackendJobId = 'job-1';",
            "let currentImage = { name: 'frame.jpg' };",
            "let captionStatus = '';",
            "let backendStatus = '';",
            "let renderCount = 0;",
            "let buttonUpdates = 0;",
            "let ensureCalls = 0;",
            "let bundleCalls = 0;",
            "let textLabels = { 'frame.jpg': 'old caption' };",
            "let captionRecordsByImage = { 'frame.jpg': [{ id: 'old', caption: 'old alternate' }] };",
            "const captionAutoSaveState = { lastSaved: new Map() };",
            "const qwenElements = { captionOutput: { value: 'stable caption' } };",
            "function setQwenCaptionStatus(message) { captionStatus = message; }",
            "function setQwenCaptionBackendJobStatus(message) { backendStatus = message; }",
            "function updateQwenCaptionButton() { buttonUpdates += 1; }",
            "function renderCaptionAlternatesForCurrentImage() { renderCount += 1; }",
            "function getCaptionRecordDatasetId() { return 'ds'; }",
            "function ensureCaptionLabelStoreForDataset() { ensureCalls += 1; }",
            "function normalizeCaptionRecord(record) { return record || {}; }",
            "function getSelectedCaptionRecord() { return { caption: 'selected caption' }; }",
            "function setCaptionOutputValue() { throw new Error('caption output should not be replaced by a stale archive read'); }",
            "async function loadCaptionBundleForImage() { bundleCalls += 1; return { primary_caption: 'new caption', captions: [] }; }",
            _extract_js_function(js, "qwenCaptionArchiveMutationActive"),
            _extract_js_function(js, "captionArchiveMutationBusyMessage"),
            _extract_js_function(js, "deferCaptionArchiveReadWhileBusy"),
            "async " + _extract_js_function_before(
                js,
                "loadCaptionForCurrentImage",
                "\n    async function captionExistsForImage",
            ),
            "let loaded = await loadCaptionForCurrentImage();",
            "assert.strictEqual(loaded, false);",
            "assert.strictEqual(bundleCalls, 0);",
            "assert.strictEqual(ensureCalls, 0);",
            "assert.strictEqual(qwenElements.captionOutput.value, 'stable caption');",
            "assert.strictEqual(captionStatus, 'Caption archive busy');",
            "assert(backendStatus.includes('loading caption archive'));",
            "assert(backendStatus.includes('caption archive is changing'));",
            "assert.strictEqual(renderCount, 1);",
            "assert.strictEqual(buttonUpdates, 1);",
            "qwenCaptionBatchBackendJobId = '';",
            "captionStatus = '';",
            "backendStatus = '';",
            "renderCount = 0;",
            "buttonUpdates = 0;",
            "ensureCalls = 0;",
            "bundleCalls = 0;",
            "loadCaptionBundleForImage = async function() {",
            "  bundleCalls += 1;",
            "  qwenCaptionBatchBackendJobId = 'job-2';",
            "  return { primary_caption: 'new caption', captions: [] };",
            "};",
            "loaded = await loadCaptionForCurrentImage();",
            "assert.strictEqual(loaded, false);",
            "assert.strictEqual(bundleCalls, 1);",
            "assert.strictEqual(ensureCalls, 1);",
            "assert.strictEqual(qwenElements.captionOutput.value, 'stable caption');",
            "assert.strictEqual(textLabels['frame.jpg'], 'old caption');",
            "assert.deepStrictEqual(captionRecordsByImage['frame.jpg'], [{ id: 'old', caption: 'old alternate' }]);",
            "assert.strictEqual(captionStatus, 'Caption archive busy');",
            "assert(backendStatus.includes('loading caption archive'));",
            "assert(backendStatus.includes('caption archive is changing'));",
            "assert.strictEqual(renderCount, 1);",
            "assert.strictEqual(buttonUpdates, 1);",
        ]
    )
    subprocess.run(
        [
            "node",
            "-e",
            f"(async () => {{\n{script}\n}})().catch((error) => {{ console.error(error); process.exit(1); }});",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_qwen_caption_exports_block_while_backend_job_id_is_active():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "let qwenCaptionActive = false;",
            "let qwenCaptionBatchActive = false;",
            "let qwenCaptionBatchBackendJobId = 'job-1';",
            "let exportHealth = null;",
            "let status = null;",
            "let prepared = 0;",
            "let saved = 0;",
            "function setCaptionExportHealth(message, variant) { exportHealth = { message, variant }; }",
            "function setSamStatus(message, options) { status = { message, options }; }",
            "async function prepareCaptionExportRecords() { prepared += 1; throw new Error('prepare should not run while archive is mutating'); }",
            "function saveBlobToDisk() { saved += 1; throw new Error('save should not run while archive is mutating'); }",
            _extract_js_function(js, "qwenCaptionArchiveMutationActive"),
            _extract_js_function(js, "captionArchiveExportBusyMessage"),
            "async " + _extract_js_function(js, "downloadCaptionJsonl"),
            "async " + _extract_js_function(js, "downloadCaptionGroupedJson"),
            "async " + _extract_js_function(js, "downloadCaptionVlmJsonl"),
            "await downloadCaptionJsonl();",
            "assert.strictEqual(prepared, 0);",
            "assert.strictEqual(saved, 0);",
            "assert.strictEqual(exportHealth.variant, 'warn');",
            "assert(exportHealth.message.includes('caption archive is changing'));",
            "assert.strictEqual(status.options.variant, 'warn');",
            "await downloadCaptionGroupedJson();",
            "assert.strictEqual(prepared, 0);",
            "assert.strictEqual(saved, 0);",
            "assert(exportHealth.message.includes('caption archive is changing'));",
            "await downloadCaptionVlmJsonl();",
            "assert.strictEqual(prepared, 0);",
            "assert.strictEqual(saved, 0);",
            "assert(exportHealth.message.includes('caption archive is changing'));",
        ]
    )
    subprocess.run(
        [
            "node",
            "-e",
            f"(async () => {{\n{script}\n}})().catch((error) => {{ console.error(error); process.exit(1); }});",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_dataset_manager_download_uses_fetch_and_surfaces_server_errors():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const API_ROOT = 'http://backend.test';",
            "const datasetManagerState = { actionInFlight: new Set(), datasets: [] };",
            "let renderCount = 0;",
            "let fetchCalls = [];",
            "let messages = [];",
            "let saved = [];",
            "function datasetActionKey(datasetId, action) { return `${datasetId}:${action}`; }",
            "function renderDatasetList() { renderCount += 1; }",
            "function setDatasetUploadMessage(text, tone) { messages.push({ text, tone }); }",
            "function filenameFromResponse(_resp, _fallback) { return 'dataset_export.zip'; }",
            "function saveBlobToDisk(blob, filename) { saved.push({ blob, filename }); }",
            _extract_js_function(js, "parseApiError"),
            "async " + _extract_js_function(js, "handleDatasetDownload"),
            "global.fetch = async (url) => {",
            "  fetchCalls.push(url);",
            "  return {",
            "    ok: false,",
            "    status: 409,",
            "    text: async () => JSON.stringify({ detail: 'dataset_download_busy:qcap_busy:running' }),",
            "  };",
            "};",
            "await handleDatasetDownload({ id: 'ds', label: 'Demo dataset' });",
            "assert.strictEqual(saved.length, 0);",
            "assert.strictEqual(datasetManagerState.actionInFlight.size, 0);",
            "const downloadError = messages.find((entry) => entry.tone === 'error');",
            "assert(downloadError);",
            "assert(downloadError.text.includes('Dataset download is blocked while caption dataset job qcap_busy is running.'));",
            "assert(!downloadError.text.includes('dataset_download_busy:qcap_busy:running'));",
            "assert(!downloadError.text.includes('{\"detail\"'));",
            "messages = [];",
            "global.fetch = async (url) => {",
            "  fetchCalls.push(url);",
            "  return {",
            "    ok: true,",
            "    status: 200,",
            "    headers: { get: () => 'attachment; filename=\"dataset.zip\"' },",
            "    blob: async () => ({ bytes: 3 }),",
            "  };",
            "};",
            "await handleDatasetDownload({ id: 'ds', label: 'Demo dataset' });",
            "assert.strictEqual(saved.length, 1);",
            "assert.strictEqual(saved[0].filename, 'dataset_export.zip');",
            "assert(messages.some((entry) => entry.tone === 'success' && entry.text.includes('Downloaded Demo dataset.')));",
            "assert.strictEqual(datasetManagerState.actionInFlight.size, 0);",
            "assert(fetchCalls.every((url) => url === 'http://backend.test/datasets/ds/download'));",
            "assert(renderCount >= 4);",
        ]
    )
    subprocess.run(
        [
            "node",
            "-e",
            f"(async () => {{\n{script}\n}})().catch((error) => {{ console.error(error); process.exit(1); }});",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_parse_api_error_formats_caption_job_guards_for_operators():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            _extract_js_function(js, "parseApiError"),
            "const cases = [",
            "  ['caption_export_busy:qcap_1:running', 'Caption export is blocked while caption dataset job qcap_1 is running. Wait for that job to finish, then retry.'],",
            "  ['caption_review_import_busy:qcap_2:queued', 'Review import is blocked while caption dataset job qcap_2 is queued. Wait for that job to finish, then retry.'],",
            "  ['caption_mutation_busy:qcap_3:cancelling', 'Caption and text-label edits are blocked while caption dataset job qcap_3 is cancelling. Wait for that job to finish, then retry.'],",
            "  ['caption_read_busy:qcap_4:running', 'Caption and text-label reads are blocked while caption dataset job qcap_4 is running. Wait for that job to finish, then retry.'],",
            "  ['dataset_download_busy:qcap_5:running', 'Dataset download is blocked while caption dataset job qcap_5 is running. Wait for that job to finish, then retry.'],",
            "  ['caption_metadata_busy:qcap_6:running', 'Dataset glossary changes are blocked while caption dataset job qcap_6 is running. Wait for that job to finish, then retry.'],",
            "  ['qwen_caption_dataset_job_active:qcap_7:queued', 'A caption dataset job is already active while caption dataset job qcap_7 is queued. Wait for that job to finish, then retry.'],",
            "];",
            "for (const [detail, expected] of cases) {",
            "  assert.strictEqual(parseApiError(JSON.stringify({ detail }), 'fallback'), expected);",
            "}",
            "assert.strictEqual(",
            "  parseApiError(JSON.stringify({ detail: 'annotation_lock_session_required' }), 'fallback'),",
            "  'This dataset is locked by an active annotation session. Reopen that annotation session or wait for the lock to expire before starting a write-owning job.'",
            ");",
            "assert.strictEqual(",
            "  parseApiError(JSON.stringify({ detail: 'annotation_lock_active' }), 'fallback'),",
            "  'This dataset is locked by another annotation session. Use the matching session or wait for the lock to expire.'",
            ");",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_dataset_manager_glossary_save_formats_caption_metadata_busy_error():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const API_ROOT = 'http://backend.test';",
            "const datasetManagerState = { glossaryDatasetSaveInFlight: false, glossaryDatasetLoadInFlight: false, glossaryDatasetSaveAsInFlight: false, glossaryLibraryLoadInFlight: false, glossaryLibrarySaveInFlight: false, glossaryLibraryDeleteInFlight: false };",
            "const glossaryLibraryState = { inFlight: false };",
            "let messages = [];",
            "let buttonUpdates = 0;",
            "const datasetManagerElements = {",
            "  glossaryDatasetSelect: { value: 'ds' },",
            "  glossaryDatasetEditor: { value: '{\"car\":[\"vehicle\"]}' },",
            "  glossaryDatasetMessage: {},",
            "  glossaryDatasetLoad: {},",
            "  glossaryDatasetSave: {},",
            "  glossaryDatasetSaveAs: {},",
            "  glossaryLibrarySelect: { value: '' },",
            "  glossaryLibraryName: { value: '' },",
            "  glossaryLibraryRefresh: {},",
            "  glossaryLibrarySave: {},",
            "  glossaryLibraryDelete: {},",
            "  glossaryLibraryDownload: {},",
            "};",
            "function setGlossaryMessage(_element, text, tone) { messages.push({ text, tone }); }",
            "function updateGlossaryDatasetSummary() {}",
            "function updateGlossaryLibrarySelect() {}",
            _extract_js_function(js, "parseApiError"),
            _extract_js_function(js, "updateGlossaryActionButtons").replace(
                "function updateGlossaryActionButtons",
                "function originalUpdateGlossaryActionButtons",
            ),
            "function updateGlossaryActionButtons() { buttonUpdates += 1; originalUpdateGlossaryActionButtons(); }",
            "async " + _extract_js_function(js, "saveDatasetGlossary"),
            "global.fetch = async (url, options) => {",
            "  assert.strictEqual(url, 'http://backend.test/datasets/ds/glossary');",
            "  assert.strictEqual(options.method, 'POST');",
            "  return {",
            "    ok: false,",
            "    status: 409,",
            "    statusText: 'Conflict',",
            "    text: async () => JSON.stringify({ detail: 'caption_metadata_busy:qcap_busy:running' }),",
            "  };",
            "};",
            "await saveDatasetGlossary();",
            "assert.strictEqual(datasetManagerState.glossaryDatasetSaveInFlight, false);",
            "assert(buttonUpdates >= 2);",
            "assert(messages.some((entry) => entry.tone === 'error' && entry.text.includes('Dataset glossary changes are blocked while caption dataset job qcap_busy is running.')));",
            "assert(!messages.some((entry) => entry.text.includes('{\"detail\"')));",
        ]
    )
    subprocess.run(
        [
            "node",
            "-e",
            f"(async () => {{\n{script}\n}})().catch((error) => {{ console.error(error); process.exit(1); }});",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_qwen_caption_text_label_save_formats_caption_mutation_busy_error():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const API_ROOT = 'http://backend.test';",
            "let textLabels = {};",
            "const annotationSourceState = { dirtyRecordsByKey: new Map() };",
            "const captionAutoSaveState = { timerId: null, pendingImage: null, lastAttempted: new Map(), lastSaved: new Map() };",
            "const statusMessages = [];",
            "const samMessages = [];",
            "function resolveCaptionPersistenceContext(datasetIdOverride = null) { return { mode: 'dataset', datasetId: datasetIdOverride || 'ds' }; }",
            "function isAnnotationMutationBlocked() { return false; }",
            "function isAnnotationDatasetModeActive() { return false; }",
            "function annotationEditableGuard() { return true; }",
            "function guardQwenCaptionArchiveIdle() { return true; }",
            "function captureAnnotationDirtyStateForImage() {}",
            "async function flushAnnotationSnapshot() { return true; }",
            "function ensureCaptionLabelStoreForDataset() {}",
            "function storeCaptionRecord() { throw new Error('storeCaptionRecord should not run after failed save'); }",
            "function setQwenCaptionStatus(message) { statusMessages.push(message); }",
            "function setSamStatus(message, options) { samMessages.push({ message, options }); }",
            _extract_js_function(js, "parseApiError"),
            _extract_js_function(js, "formatBackendFetchError"),
            "async " + _extract_js_function(js, "persistCaptionLabel"),
            "async " + _extract_js_function_before(
                js,
                "saveCaptionImmediate",
                "\n    function scheduleCaptionAutosave",
            ),
            "global.fetch = async (url, options) => {",
            "  assert.strictEqual(url, 'http://backend.test/datasets/ds/text_labels/frame.jpg');",
            "  assert.strictEqual(options.method, 'POST');",
            "  assert.deepStrictEqual(JSON.parse(options.body), { caption: 'caption text' });",
            "  return {",
            "    ok: false,",
            "    status: 409,",
            "    statusText: 'Conflict',",
            "    text: async () => JSON.stringify({ detail: 'caption_mutation_busy:qcap_busy:running' }),",
            "  };",
            "};",
            "const saved = await saveCaptionImmediate('frame.jpg', 'caption text', { datasetId: 'ds' });",
            "assert.strictEqual(saved, false);",
            "assert(statusMessages.some((message) => message.includes('Caption and text-label edits are blocked while caption dataset job qcap_busy is running.')));",
            "assert(samMessages.some((entry) => entry.message.includes('Caption save failed: Caption and text-label edits are blocked while caption dataset job qcap_busy is running.')));",
            "assert(!statusMessages.some((message) => message.includes('{\"detail\"')));",
            "assert(!samMessages.some((entry) => entry.message.includes('caption_mutation_busy:qcap_busy:running')));",
        ]
    )
    subprocess.run(
        [
            "node",
            "-e",
            f"(async () => {{\n{script}\n}})().catch((error) => {{ console.error(error); process.exit(1); }});",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_qwen_caption_archive_action_failures_are_formatted_for_operator_status():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const qwenElements = {",
            "  captionStatus: { textContent: '' },",
            "  captionExportHealth: { textContent: '', classes: [], classList: { remove(...names) { this.owner.classes = this.owner.classes.filter((name) => !names.includes(name)); }, add(name) { this.owner.classes.push(name); } } },",
            "};",
            "qwenElements.captionExportHealth.classList.owner = qwenElements.captionExportHealth;",
            "const samMessages = [];",
            "function setSamStatus(message, options) { samMessages.push({ message, options }); }",
            _extract_js_function(js, "setQwenCaptionStatus"),
            _extract_js_function(js, "setCaptionExportHealth"),
            _extract_js_function(js, "captionArchiveActionFailureMessage"),
            _extract_js_function(js, "reportCaptionArchiveActionFailure"),
            _extract_js_function(js, "reportCaptionArchiveExportFailure"),
            "assert.strictEqual(",
            "  captionArchiveActionFailureMessage('Caption update', new Error('Caption update failed: stale row')),",
            "  'Caption update failed: stale row'",
            ");",
            "assert.strictEqual(",
            "  captionArchiveActionFailureMessage('Alternate caption delete', new Error('Caption and text-label edits are blocked while caption dataset job qcap_1 is running.')),",
            "  'Caption and text-label edits are blocked while caption dataset job qcap_1 is running.'",
            ");",
            "const reported = reportCaptionArchiveActionFailure('Primary caption update', new Error('backend unavailable'), 1234);",
            "assert.strictEqual(reported, 'Primary caption update failed: backend unavailable');",
            "assert.strictEqual(qwenElements.captionStatus.textContent, reported);",
            "assert.strictEqual(samMessages[0].message, reported);",
            "assert.strictEqual(samMessages[0].options.duration, 1234);",
            "const blocked = reportCaptionArchiveExportFailure('Caption JSONL export', new Error('Caption export is blocked while caption dataset job qcap_1 is running.'), 2345);",
            "assert.strictEqual(blocked, 'Caption export is blocked while caption dataset job qcap_1 is running.');",
            "assert.strictEqual(qwenElements.captionExportHealth.textContent, blocked);",
            "assert(qwenElements.captionExportHealth.classes.includes('is-fail'));",
            "assert.strictEqual(samMessages[1].message, blocked);",
            "assert.strictEqual(samMessages[1].options.duration, 2345);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_caption_archive_action_listeners_do_not_report_noop_success():
    js = _js()
    listener_start = js.index("if (qwenElements.captionSaveAlternate)")
    listener_end = js.index("if (qwenElements.captionCopyButton)", listener_start)
    listener_block = js[listener_start:listener_end]
    assert ".then((record) => {" in listener_block
    assert ".then((updated) => {" in listener_block
    assert ".then((deleted) => {" in listener_block
    assert "if (record) {" in listener_block
    assert "if (updated) {" in listener_block
    assert "if (deleted) {" in listener_block
    assert 'reportCaptionArchiveActionFailure("Alternate caption save", error' in listener_block
    assert 'reportCaptionArchiveActionFailure("Caption update", error' in listener_block
    assert 'reportCaptionArchiveActionFailure("Primary caption update", error' in listener_block
    assert 'reportCaptionArchiveActionFailure("Alternate caption delete", error' in listener_block
    assert "Alternate caption delete failed: ${error.message || error}" not in listener_block
    update_fn = _extract_js_function_before(
        js,
        "updateSelectedCaptionFromTextarea",
        "\n    async function setSelectedCaptionAsPrimary",
    )
    assert "const saved = await saveCaptionImmediate(imageName, caption);" in update_fn
    assert "if (!saved)" in update_fn
    assert update_fn.count("return false;") >= 2


def test_qwen_caption_instruction_review_import_parser_accepts_reviewer_file_shapes():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_ID_CHARS = 512;",
            "const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_PATH_CHARS = 4096;",
            "const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_QUESTION_CHARS = 4096;",
            "const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_ANSWER_CHARS = 65536;",
            "const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_NOTES_CHARS = 8192;",
            _extract_js_function(js, "normalizeCaptionInstructionReviewDecision"),
            _extract_js_function_before(
                js,
                "parseCaptionInstructionReviewRowsText",
                "\n    async function importCaptionInstructionReviewFile",
            ),
            _extract_js_function(js, "validateCaptionInstructionReviewRows"),
            _extract_js_function(js, "captionInstructionReviewDatasetMismatches"),
            "const row = {",
            "  format: 'tator_caption_instruction_review_rows_v1',",
            "  image_path: 'train/frame.jpg',",
            "  qa_id: 'qa-1',",
            "  review_decision: 'needs-revision'",
            "};",
            "assert.strictEqual(normalizeCaptionInstructionReviewDecision('needs-revision'), 'needs_revision');",
            "assert.strictEqual(normalizeCaptionInstructionReviewDecision('needs review'), 'needs_revision');",
            "assert.strictEqual(normalizeCaptionInstructionReviewDecision('Needs-Rewrite'), 'needs_revision');",
            "assert.strictEqual(parseCaptionInstructionReviewRowsText(JSON.stringify([row], null, 2))[0].qa_id, 'qa-1');",
            "assert.strictEqual(parseCaptionInstructionReviewRowsText(JSON.stringify({ instruction_review_rows: [row] }, null, 2))[0].qa_id, 'qa-1');",
            "assert.strictEqual(parseCaptionInstructionReviewRowsText(JSON.stringify(row, null, 2))[0].qa_id, 'qa-1');",
            "assert.strictEqual(parseCaptionInstructionReviewRowsText(JSON.stringify({ rows: [], instruction_review_rows: [row] }, null, 2)).length, 0);",
            "assert.throws(() => parseCaptionInstructionReviewRowsText(JSON.stringify({ rows: null, instruction_review_rows: [row] }, null, 2)), /rows must be an array/);",
            "const jsonl = JSON.stringify(row) + '\\n' + JSON.stringify({ ...row, qa_id: 'qa-2', review_decision: 'accepted' });",
            "assert.strictEqual(parseCaptionInstructionReviewRowsText(jsonl).length, 2);",
            "assert.deepStrictEqual(captionInstructionReviewDatasetMismatches([{ ...row, dataset_id: 'ds' }], 'ds'), []);",
            "assert.deepStrictEqual(captionInstructionReviewDatasetMismatches([{ ...row, dataset_id: 'other' }], 'ds'), ['other']);",
            "const actionableRow = { ...row, row_origin: 'generated_qa', question: 'What is shown?', candidate_answer: 'A scene.', training_answer: 'A scene.', validation_status: 'accepted', selected_for_training: true, requires_manual_review: true, source_summary: {}, rejection_reasons: [], review_notes: '' };",
            "assert(validateCaptionInstructionReviewRows([actionableRow]).errors.some((error) => error.includes('missing dataset_id for persisted language review row')));",
            "assert(!validateCaptionInstructionReviewRows([{ ...actionableRow, dataset_id: 'ds' }]).errors.some((error) => error.includes('missing dataset_id')));",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_caption_instruction_review_import_rejects_oversized_file_before_read():
    js = _js()
    constant_match = re.search(r"const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_BYTES = [^;]+;", js)
    assert constant_match
    script = "\n".join(
        [
            "const assert = require('assert');",
            constant_match.group(0),
            "let health = null;",
            "let status = null;",
            "let readCalled = false;",
            "let qwenCaptionActive = false;",
            "let qwenCaptionBatchActive = false;",
            "let qwenCaptionBatchBackendJobId = '';",
            "function getCaptionDatasetId() { return 'dataset-a'; }",
            "function setCaptionExportHealth(message, severity) { health = { message, severity }; }",
            "function setSamStatus(message, options) { status = { message, options }; }",
            _extract_js_function(js, "formatBytesLabel"),
            _extract_js_function(js, "qwenCaptionArchiveMutationActive"),
            _extract_js_function(js, "captionInstructionArtifactBusyMessage"),
            "async " + _extract_js_function(js, "importCaptionInstructionReviewFile"),
            "const hugeFile = {",
            "  size: CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_BYTES + 1,",
            "  text: async () => { readCalled = true; throw new Error('file body should not be read'); }",
            "};",
            "await importCaptionInstructionReviewFile(hugeFile);",
            "assert.strictEqual(readCalled, false);",
            "assert.strictEqual(health.severity, 'fail');",
            "assert(health.message.includes('browser import safety limit'));",
            "assert(health.message.includes('Split the review packet'));",
            "assert.strictEqual(status.options.variant, 'error');",
            "assert(status.message.includes('smaller review JSONL'));",
        ]
    )
    subprocess.run(
        [
            "node",
            "-e",
            f"(async () => {{\n{script}\n}})().catch((error) => {{ console.error(error); process.exit(1); }});",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_qwen_caption_instruction_review_import_formats_backend_failures():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            _extract_js_function(js, "formatCaptionInstructionReviewImportApiError"),
            "const staleQa = formatCaptionInstructionReviewImportApiError('review_rows_generated_qa_not_found:row_3');",
            "assert(staleQa.includes('blocked at row 3'));",
            "assert(staleQa.includes('no longer matches a saved generated-QA record'));",
            "assert(staleQa.includes('Export a fresh review JSONL'));",
            "const staleQaTraining = formatCaptionInstructionReviewImportApiError('review_rows_generated_qa_training_answer_stale:row_5');",
            "assert(staleQaTraining.includes('blocked at row 5'));",
            "assert(staleQaTraining.includes('training answer no longer matches'));",
            "assert(staleQaTraining.includes('fresh review JSONL'));",
            "const missingQaText = formatCaptionInstructionReviewImportApiError('review_rows_generated_qa_text_missing:row_8');",
            "assert(missingQaText.includes('blocked at row 8'));",
            "assert(missingQaText.includes('missing the reviewed question or answer text'));",
            "const invalidSelectedFlag = formatCaptionInstructionReviewImportApiError('review_rows_selected_for_training_invalid:row_2');",
            "assert(invalidSelectedFlag.includes('blocked at row 2'));",
            "assert(invalidSelectedFlag.includes('selected_for_training must be a boolean'));",
            "const missingReviewNotes = formatCaptionInstructionReviewImportApiError('review_rows_review_notes_missing:row_9');",
            "assert(missingReviewNotes.includes('blocked at row 9'));",
            "assert(missingReviewNotes.includes('review_notes column'));",
            "const staleCaption = formatCaptionInstructionReviewImportApiError('review_rows_caption0_not_found:row_4');",
            "assert(staleCaption.includes('caption0 row no longer matches the saved caption text'));",
            "const staleCaptionTraining = formatCaptionInstructionReviewImportApiError('review_rows_caption0_training_answer_stale:row_6');",
            "assert(staleCaptionTraining.includes('blocked at row 6'));",
            "assert(staleCaptionTraining.includes('training answer no longer matches'));",
            "const mismatch = formatCaptionInstructionReviewImportApiError('review_rows_dataset_id_mismatch:row_2:other-ds!=current-ds');",
            "assert(mismatch.includes('blocked at row 2'));",
            "assert(mismatch.includes('other-ds'));",
            "assert(mismatch.includes('current-ds'));",
            "const missingDataset = formatCaptionInstructionReviewImportApiError('review_rows_dataset_id_missing:row_9');",
            "assert(missingDataset.includes('blocked at row 9'));",
            "assert(missingDataset.includes('missing the embedded dataset id'));",
            "const missingQaId = formatCaptionInstructionReviewImportApiError('review_rows_qa_id_missing:row_10');",
            "assert(missingQaId.includes('blocked at row 10'));",
            "assert(missingQaId.includes('missing the stable QA id'));",
            "const duplicate = formatCaptionInstructionReviewImportApiError('review_rows_conflicting_duplicate_target:row_1:row_5');",
            "assert(duplicate.includes('conflicting duplicate decisions'));",
            "assert(duplicate.includes('rows 1 and 5'));",
            "assert(duplicate.includes('same actionable review target'));",
            "const resolvedDuplicate = formatCaptionInstructionReviewImportApiError('review_rows_duplicate_resolved_target:row_2:row_6');",
            "assert(resolvedDuplicate.includes('duplicate decisions'));",
            "assert(resolvedDuplicate.includes('same saved caption or generated-QA record'));",
            "const unsupported = formatCaptionInstructionReviewImportApiError('review_rows_unsupported_row_origin:row_6:freeform_review');",
            "assert(unsupported.includes('freeform_review is not a persisted review row type'));",
            "const tooLong = formatCaptionInstructionReviewImportApiError('review_rows_field_too_long:row_2:review_notes:8192');",
            "assert(tooLong.includes('blocked at row 2'));",
            "assert(tooLong.includes('review_notes exceeds 8192 characters'));",
            "const invalidText = formatCaptionInstructionReviewImportApiError('review_rows_review_notes_invalid:row_3');",
            "assert(invalidText.includes('blocked at row 3'));",
            "assert(invalidText.includes('review_notes must be a text field'));",
            "const invalidBundleHash = formatCaptionInstructionReviewImportApiError('review_rows_bundle_image_sha256_invalid:row_11');",
            "assert(invalidBundleHash.includes('blocked at row 11'));",
            "assert(invalidBundleHash.includes('64-character SHA-256 digest'));",
            "const missingBundleOriginal = formatCaptionInstructionReviewImportApiError('review_rows_original_image_path_missing_for_bundle:row_12');",
            "assert(missingBundleOriginal.includes('blocked at row 12'));",
            "assert(missingBundleOriginal.includes('copied bundle image'));",
            "assert(missingBundleOriginal.includes('original image path'));",
            "const bundleImageMissing = formatCaptionInstructionReviewImportApiError('review_rows_bundle_image_not_found:row_13');",
            "assert(bundleImageMissing.includes('blocked at row 13'));",
            "assert(bundleImageMissing.includes('reviewed bundle image cannot be resolved'));",
            "assert(bundleImageMissing.includes('Export a fresh review JSONL'));",
            "const bundleImageHashFailed = formatCaptionInstructionReviewImportApiError('review_rows_bundle_image_hash_failed:row_14');",
            "assert(bundleImageHashFailed.includes('blocked at row 14'));",
            "assert(bundleImageHashFailed.includes('could not be hashed'));",
            "const bundleImageMismatch = formatCaptionInstructionReviewImportApiError('review_rows_bundle_image_sha256_mismatch:row_15');",
            "assert(bundleImageMismatch.includes('blocked at row 15'));",
            "assert(bundleImageMismatch.includes('no longer match the image bytes'));",
            "assert(bundleImageMismatch.includes('Export a fresh review JSONL'));",
            "const noActionable = formatCaptionInstructionReviewImportApiError('review_rows_no_actionable_decisions');",
            "assert(noActionable.includes('no accepted, rejected, or needs-revision caption0 or generated-QA decisions'));",
            "assert(noActionable.includes('Fill review_decision'));",
            "const blockedCreate = formatCaptionInstructionReviewImportApiError('review_rows_caption0_creation_not_allowed:row_7');",
            "assert(blockedCreate.includes('blocked at row 7'));",
            "assert(blockedCreate.includes('selected dataset, resolved image key, and current text-label caption'));",
            "assert(formatCaptionInstructionReviewImportApiError('plain backend error').includes('plain backend error'));",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_caption_instruction_export_query_uses_backend_ready_gate_only_when_requested():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            _extract_js_function(js, "captionInstructionExportQuery"),
            "const settings = {",
            "  include_caption0_in_training: true,",
            "  include_generated_qa_in_training: false,",
            "  include_deterministic_metadata_qa: true,",
            "  instruction_qa_restrict_speculative_language: true,",
            "  qa_mix: 'object',",
            "  answer_format: 'json',",
            "};",
            "const strictParams = new URLSearchParams(captionInstructionExportQuery(settings, { requireReadyInstructionExport: true }));",
            "assert.strictEqual(strictParams.get('include_caption0_in_training'), 'true');",
            "assert.strictEqual(strictParams.get('include_generated_qa_in_training'), 'false');",
            "assert.strictEqual(strictParams.get('include_deterministic_metadata_qa'), 'true');",
            "assert.strictEqual(strictParams.get('instruction_qa_restrict_speculative_language'), 'true');",
            "assert.strictEqual(strictParams.get('qa_mix'), 'object');",
            "assert.strictEqual(strictParams.get('answer_format'), 'json');",
            "assert.strictEqual(strictParams.get('require_ready_instruction_export'), 'true');",
            "const diagnosticParams = new URLSearchParams(captionInstructionExportQuery(settings));",
            "assert.strictEqual(diagnosticParams.has('require_ready_instruction_export'), false);",
            "const explicitDiagnosticParams = new URLSearchParams(captionInstructionExportQuery(settings, { requireReadyInstructionExport: false }));",
            "assert.strictEqual(explicitDiagnosticParams.get('require_ready_instruction_export'), 'false');",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_caption_instruction_export_formats_backend_readiness_failure():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            _extract_js_function(js, "formatCaptionInstructionExportApiError"),
            "const message = formatCaptionInstructionExportApiError('instruction_export_not_ready:needs_review');",
            "assert(message.includes('Instruction JSONL export blocked'));",
            "assert(message.includes('training readiness is needs_review'));",
            "assert(message.includes('disable Require ready report only for deliberate review-pending diagnostics'));",
            "assert.strictEqual(formatCaptionInstructionExportApiError('plain backend error'), 'plain backend error');",
            "assert.strictEqual(formatCaptionInstructionExportApiError(''), 'Instruction export failed.');",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_caption_instruction_bundle_formats_backend_failures():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            _extract_js_function(js, "formatCaptionInstructionBundleApiError"),
            "const ready = formatCaptionInstructionBundleApiError('instruction_export_not_ready:needs_review');",
            "assert(ready.includes('Training bundle export blocked'));",
            "assert(ready.includes('training readiness is needs_review'));",
            "assert(ready.includes('disable Require ready report only for deliberate review-pending diagnostics'));",
            "const missingImage = formatCaptionInstructionBundleApiError('caption_instruction_bundle_image_unavailable:train/frame.jpg');",
            "assert(missingImage.includes('copied image source is unavailable'));",
            "assert(missingImage.includes('train/frame.jpg'));",
            "const missingArchive = formatCaptionInstructionBundleApiError('caption_instruction_bundle_archive_image_missing:train/stale.jpg');",
            "assert(missingArchive.includes('archive image train/stale.jpg was not copied'));",
            "const missingReview = formatCaptionInstructionBundleApiError('caption_instruction_bundle_review_image_missing:train/stale.jpg');",
            "assert(missingReview.includes('review image train/stale.jpg was not copied'));",
            "const inconsistent = formatCaptionInstructionBundleApiError('caption_instruction_bundle_artifacts_inconsistent:selected_review_row_count_mismatch');",
            "assert(inconsistent.includes('artifacts are inconsistent'));",
            "assert(inconsistent.includes('selected_review_row_count_mismatch'));",
            "const manifest = formatCaptionInstructionBundleApiError('caption_instruction_bundle_manifest_invalid:manifest_file_sha256_mismatch:images/train/frame.jpg');",
            "assert(manifest.includes('does not match its manifest checksum'));",
            "assert(manifest.includes('images/train/frame.jpg'));",
            "const rowCount = formatCaptionInstructionBundleApiError('caption_instruction_bundle_manifest_invalid:manifest_row_count_mismatch:training_rows');",
            "assert(rowCount.includes('manifest training rows count does not match'));",
            "const missingManifestImage = formatCaptionInstructionBundleApiError('caption_instruction_bundle_manifest_invalid:manifest_image_file_missing:images/train/missing.jpg');",
            "assert(missingManifestImage.includes('manifest references copied image images/train/missing.jpg'));",
            "const copiedLabelSha = formatCaptionInstructionBundleApiError('caption_instruction_bundle_manifest_invalid:manifest_label_sha256_mismatch:labels/train/frame.txt');",
            "assert(copiedLabelSha.includes('copied label labels/train/frame.txt has a manifest checksum mismatch'));",
            "const badArtifact = formatCaptionInstructionBundleApiError('caption_instruction_bundle_manifest_invalid:manifest_artifact_path_invalid:training_jsonl');",
            "assert(badArtifact.includes('manifest points training_jsonl at the wrong artifact path'));",
            "const artifactDrift = formatCaptionInstructionBundleApiError('caption_instruction_bundle_manifest_invalid:manifest_artifacts_inconsistent:training row count mismatch');",
            "assert(artifactDrift.includes('bundled artifacts do not agree with each other'));",
            "assert(artifactDrift.includes('training row count mismatch'));",
            "const trainerRowsInvalid = formatCaptionInstructionBundleApiError('caption_instruction_bundle_manifest_invalid:manifest_training_rows_invalid:row 1 metadata missing source_archive');",
            "assert(trainerRowsInvalid.includes('bundled trainer rows are invalid'));",
            "assert(trainerRowsInvalid.includes('row 1 metadata missing source_archive'));",
            "const badJsonl = formatCaptionInstructionBundleApiError('caption_instruction_bundle_manifest_invalid:manifest_jsonl_row_invalid:caption_instruction_training.jsonl:3');",
            "assert(badJsonl.includes('caption_instruction_training.jsonl has a malformed JSONL row at line 3'));",
            "const duplicateZipMember = formatCaptionInstructionBundleApiError('caption_instruction_bundle_manifest_invalid:manifest_zip_duplicate_member');",
            "assert(duplicateZipMember.includes('ZIP contains duplicate member names'));",
            "const noRows = formatCaptionInstructionBundleApiError('caption_instruction_bundle_no_archive_rows');",
            "assert(noRows.includes('Create a training dataset first'));",
            "assert.strictEqual(formatCaptionInstructionBundleApiError('plain backend error'), 'plain backend error');",
            "assert.strictEqual(formatCaptionInstructionBundleApiError(''), 'Training bundle export failed.');",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_caption_instruction_action_failures_update_export_health_without_double_prefix():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const qwenElements = { captionExportHealth: { textContent: '', classes: [], classList: { remove(...names) { this.owner.classes = this.owner.classes.filter((name) => !names.includes(name)); }, add(name) { this.owner.classes.push(name); } } } };",
            "qwenElements.captionExportHealth.classList.owner = qwenElements.captionExportHealth;",
            "const samMessages = [];",
            "function setSamStatus(message, options) { samMessages.push({ message, options }); }",
            _extract_js_function(js, "setCaptionExportHealth"),
            _extract_js_function(js, "captionInstructionActionFailureMessage"),
            _extract_js_function(js, "reportCaptionInstructionActionFailure"),
            "assert.strictEqual(",
            "  captionInstructionActionFailureMessage('Instruction review import', new Error('Instruction review import blocked: stale row.')),",
            "  'Instruction review import blocked: stale row.'",
            ");",
            "assert.strictEqual(",
            "  captionInstructionActionFailureMessage('Instruction archive export', new Error('Caption export is blocked while caption dataset job q1 is running.')),",
            "  'Caption export is blocked while caption dataset job q1 is running.'",
            ");",
            "const reported = reportCaptionInstructionActionFailure('Instruction report export', new Error('backend unavailable'), 1234);",
            "assert.strictEqual(reported, 'Instruction report export failed: backend unavailable');",
            "assert.strictEqual(qwenElements.captionExportHealth.textContent, reported);",
            "assert(qwenElements.captionExportHealth.classes.includes('is-fail'));",
            "assert.strictEqual(samMessages[0].message, reported);",
            "assert.strictEqual(samMessages[0].options.duration, 1234);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_caption_instruction_action_listeners_share_failure_reporter():
    js = _js()
    listener_start = js.index("if (qwenElements.captionDownloadInstructionBundle)")
    listener_end = js.index("renderCaptionAlternatesForCurrentImage();", listener_start)
    listener_block = js[listener_start:listener_end]
    expected_actions = [
        "Training bundle export",
        "Instruction JSONL export",
        "Instruction archive export",
        "Instruction review export",
        "Instruction review import",
        "Instruction report export",
    ]
    for action in expected_actions:
        assert f'reportCaptionInstructionActionFailure("{action}", error' in listener_block
    assert "Instruction review import failed: ${error.message || error}" not in listener_block
    assert "Instruction archive export failed: ${error.message || error}" not in listener_block
    assert "Instruction report export failed: ${error.message || error}" not in listener_block


def test_qwen_caption_instruction_review_import_click_blocks_busy_file_picker():
    js = _js()
    listener_start = js.index("if (qwenElements.captionImportInstructionReview && qwenElements.captionImportInstructionReviewFile)")
    listener_end = js.index("if (qwenElements.captionDownloadInstructionReport)", listener_start)
    listener_block = js[listener_start:listener_end]
    click_call = "qwenElements.captionImportInstructionReviewFile.click();"
    busy_guard = 'captionInstructionArtifactBusyMessage("selecting reviewed instruction rows")'
    assert busy_guard in listener_block
    assert 'setCaptionExportHealth(busyMessage, "warn")' in listener_block
    assert 'setSamStatus(busyMessage, { variant: "warn", duration: 5000 })' in listener_block
    assert listener_block.index(busy_guard) < listener_block.index(click_call)
    assert "return;" in listener_block[: listener_block.index(click_call)]


def test_qwen_caption_export_action_listeners_share_failure_reporter():
    js = _js()
    listener_start = js.index("if (qwenElements.captionDownloadJsonl)")
    listener_end = js.index("if (qwenElements.captionDownloadInstructionJsonl)", listener_start)
    listener_block = js[listener_start:listener_end]
    expected_actions = [
        "Caption JSONL export",
        "Grouped caption export",
        "VLM caption export",
    ]
    for action in expected_actions:
        assert f'reportCaptionArchiveExportFailure("{action}", error' in listener_block
    assert "Caption JSONL export failed: ${error.message || error}" not in listener_block
    assert "Grouped caption export failed: ${error.message || error}" not in listener_block
    assert "VLM caption export failed: ${error.message || error}" not in listener_block


def test_qwen_caption_instruction_export_actions_preserve_malformed_payload_errors():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_ID_CHARS = 512;",
            "const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_PATH_CHARS = 4096;",
            "const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_QUESTION_CHARS = 4096;",
            "const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_ANSWER_CHARS = 65536;",
            "const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_NOTES_CHARS = 8192;",
            _extract_js_function(js, "normalizeCaptionInstructionReviewDecision"),
            _extract_js_function(js, "validateCaptionInstructionTrainingRows"),
            _extract_js_function(js, "describeCaptionInstructionValidation"),
            _extract_js_function(js, "validateCaptionInstructionArchiveRows"),
            _extract_js_function(js, "validateCaptionInstructionReviewRows"),
            _extract_js_function(js, "describeCaptionInstructionReviewValidation"),
            _extract_js_function(js, "validateCaptionInstructionReport"),
            _extract_js_function(js, "validateCaptionInstructionArtifactConsistency"),
            _extract_js_function(js, "downloadCaptionInstructionJsonl").replace(
                "function downloadCaptionInstructionJsonl",
                "async function downloadCaptionInstructionJsonl",
                1,
            ),
            _extract_js_function(js, "downloadCaptionInstructionArchive").replace(
                "function downloadCaptionInstructionArchive",
                "async function downloadCaptionInstructionArchive",
                1,
            ),
            _extract_js_function(js, "downloadCaptionInstructionReview").replace(
                "function downloadCaptionInstructionReview",
                "async function downloadCaptionInstructionReview",
                1,
            ),
            _extract_js_function(js, "downloadCaptionInstructionReport").replace(
                "function downloadCaptionInstructionReport",
                "async function downloadCaptionInstructionReport",
                1,
            ),
            "let currentPayload = {};",
            "let exportHealth = null;",
            "let saveCount = 0;",
            "function getCaptionDatasetId() { return 'ds'; }",
            "function captionInstructionArtifactBusyMessage() { return ''; }",
            "function getCaptionInstructionDatasetSettings() { return { require_ready_instruction_export: false }; }",
            "async function loadCaptionExportPayload() { return currentPayload; }",
            "function setCaptionExportHealth(message, variant) { exportHealth = { message, variant }; }",
            "function setSamStatus() {}",
            "function saveBlobToDisk() { saveCount += 1; throw new Error('save should not be reached'); }",
            "const consistencyOk = {",
            "  format: 'tator_caption_instruction_artifact_consistency_v1',",
            "  ok: true,",
            "  error_count: 0,",
            "  errors: [],",
            "};",
            "const instructionSettings = {",
            "  include_caption0_in_training: true,",
            "  include_generated_qa_in_training: true,",
            "  include_deterministic_metadata_qa: false,",
            "  instruction_qa_restrict_speculative_language: false,",
            "  qa_mix: 'balanced',",
            "  answer_format: 'natural',",
            "};",
            "const instructionSettingsFingerprint = 'instruction-settings-fingerprint';",
            "const validReport = {",
            "  format: 'tator_caption_instruction_report_v1',",
            "  image_count: 1,",
            "  selected_flattened_row_count: 1,",
            "  instruction_review_row_count: 1,",
            "  manual_review_required_count: 0,",
            "  corpus_quality_metrics: {",
            "    image_count: 1,",
            "    selected_flattened_row_count: 1,",
            "    rejected_training_row_count: 0,",
            "    generated_qa_candidate_count: 1,",
            "    accepted_generated_qa_count: 1,",
            "    rejected_generated_qa_count: 0,",
            "    generated_qa_question_diversity_ratio: 1,",
            "    generated_qa_acceptance_rate: 1,",
            "    generated_qa_rejection_rate: 0,",
            "    structured_rewrite_rate: 0,",
            "    source_validated_training_row_rate: 1,",
            "    source_class_coverage_rate: 1,",
            "    source_classes: ['Building'],",
            "    source_classes_covered_by_training_rows: ['Building'],",
            "    training_answer_format_distribution: { natural: 1 },",
            "  },",
            "  training_readiness: {",
            "    status: 'ready',",
            "    ready_for_training: true,",
            "    blocking_reasons: [],",
            "    required_actions: [],",
            "    quality_warnings: [],",
            "    thresholds: {},",
            "    selected_training_row_count: 1,",
            "    selected_review_row_count: 1,",
            "    selected_manual_review_row_count: 0,",
            "    accepted_manual_review_row_count: 0,",
            "    pending_manual_review_row_count: 0,",
            "    rejected_manual_review_row_count: 0,",
            "    needs_revision_manual_review_row_count: 0,",
            "    instruction_export_validation_error_count: 0,",
            "  },",
            "  instruction_export_validation: { ok: true, error_count: 0, errors: [], row_count: 1 },",
            "  instruction_artifact_consistency: consistencyOk,",
            "  instruction_settings: instructionSettings,",
            "  instruction_settings_fingerprint: instructionSettingsFingerprint,",
            "};",
            "const validTrainingRow = {",
            "  image_path: 'frame.jpg',",
            "  question: 'What is shown?',",
            "  answer: 'A building.',",
            "  metadata: {",
            "    qa_id: 'qa-1',",
            "    row_type: 'generated_qa',",
            "    answer_source: 'generated_qa_record',",
            "    source_archive: 'tator_caption_instruction_archive_v1',",
            "    answer_format: 'natural',",
            "    validation_status: 'accepted',",
            "    review_status: 'accepted',",
            "  },",
            "};",
            "(async () => {",
            "  currentPayload = { instruction_training_rows: { bad: true } };",
            "  await downloadCaptionInstructionJsonl();",
            "  assert.strictEqual(saveCount, 0);",
            "  assert.strictEqual(exportHealth.variant, 'fail');",
            "  assert(exportHealth.message.includes('instruction rows must be an array'));",
            "  currentPayload = { instruction_archive_rows: { bad: true } };",
            "  await downloadCaptionInstructionArchive();",
            "  assert.strictEqual(saveCount, 0);",
            "  assert.strictEqual(exportHealth.variant, 'fail');",
            "  assert(exportHealth.message.includes('instruction archive rows must be an array'));",
            "  currentPayload = { instruction_review_rows: { bad: true } };",
            "  await downloadCaptionInstructionReview();",
            "  assert.strictEqual(saveCount, 0);",
            "  assert.strictEqual(exportHealth.variant, 'fail');",
            "  assert(exportHealth.message.includes('instruction review rows must be an array'));",
            "  currentPayload = { instruction_report: validReport, instruction_training_rows: { bad: true } };",
            "  await downloadCaptionInstructionReport();",
            "  assert.strictEqual(saveCount, 0);",
            "  assert.strictEqual(exportHealth.variant, 'fail');",
            "  assert(exportHealth.message.includes('Instruction report export blocked'));",
            "  assert(exportHealth.message.includes('instruction rows must be an array'));",
            "  assert(!exportHealth.message.includes('Instruction JSONL export blocked'));",
            "  currentPayload = {",
            "    instruction_report: validReport,",
            "    instruction_settings: instructionSettings,",
            "    instruction_settings_fingerprint: instructionSettingsFingerprint,",
            "    instruction_archive: { settings: instructionSettings, settings_fingerprint: instructionSettingsFingerprint, instruction_artifact_consistency: consistencyOk },",
            "    instruction_training_rows: [validTrainingRow],",
            "    instruction_archive_rows: { bad: true },",
            "    instruction_review_rows: [],",
            "  };",
            "  await downloadCaptionInstructionReport();",
            "  assert.strictEqual(saveCount, 0);",
            "  assert.strictEqual(exportHealth.variant, 'fail');",
            "  assert(exportHealth.message.includes('Instruction report export blocked'));",
            "  assert(exportHealth.message.includes('instruction archive rows must be an array'));",
            "  currentPayload = {",
            "    instruction_report: validReport,",
            "    instruction_settings: instructionSettings,",
            "    instruction_settings_fingerprint: instructionSettingsFingerprint,",
            "    instruction_archive: { settings: instructionSettings, settings_fingerprint: instructionSettingsFingerprint, instruction_artifact_consistency: consistencyOk },",
            "    instruction_training_rows: [validTrainingRow],",
            "    instruction_archive_rows: [],",
            "    instruction_review_rows: { bad: true },",
            "  };",
            "  await downloadCaptionInstructionReport();",
            "  assert.strictEqual(saveCount, 0);",
            "  assert.strictEqual(exportHealth.variant, 'fail');",
            "  assert(exportHealth.message.includes('Instruction report export blocked'));",
            "  assert(exportHealth.message.includes('instruction review rows must be an array'));",
            "  currentPayload = {",
            "    instruction_report: validReport,",
            "    instruction_settings: instructionSettings,",
            "    instruction_settings_fingerprint: instructionSettingsFingerprint,",
            "    instruction_archive: { settings: instructionSettings, settings_fingerprint: instructionSettingsFingerprint, instruction_artifact_consistency: consistencyOk },",
            "    instruction_training_rows: [validTrainingRow],",
            "    instruction_review_rows: [],",
            "    instruction_archive_rows: [],",
            "  };",
            "  await downloadCaptionInstructionReport();",
            "  assert.strictEqual(saveCount, 0);",
            "  assert.strictEqual(exportHealth.variant, 'fail');",
            "  assert(exportHealth.message.includes('Instruction report export blocked'));",
            "  assert(exportHealth.message.includes('missing from selected review rows'));",
            "})().catch((error) => { console.error(error); process.exit(1); });",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_caption_vlm_training_validator_rejects_canonical_image_path_duplicates():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            _extract_js_function(js, "validateCaptionVlmTrainingRows"),
            "const base = {",
            "  image_path: './train//frame.jpg',",
            "  question: 'Caption 0.',",
            "  answer: JSON.stringify({ caption: 'A grounded caption.' }),",
            "  metadata: {",
            "    row_type: 'caption',",
            "    answer_format: 'json',",
            "    validation_status: 'accepted',",
            "  },",
            "};",
            "const duplicate = validateCaptionVlmTrainingRows([base, { ...base, image_path: 'train/frame.jpg' }]);",
            "assert.strictEqual(duplicate.ok, false);",
            "assert.strictEqual(duplicate.imageCount, 1);",
            "assert(duplicate.errors.some((error) => error.includes('duplicate image_path + question')));",
            "const missingRows = validateCaptionVlmTrainingRows(undefined);",
            "assert.strictEqual(missingRows.ok, false);",
            "assert.strictEqual(missingRows.rowCount, 0);",
            "assert(missingRows.errors.some((error) => error.includes('VLM rows must be an array')));",
            "assert(missingRows.warnings.some((warning) => warning.includes('no VLM rows')));",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_caption_instruction_training_validator_blocks_non_trainable_rows():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            _extract_js_function(js, "normalizeCaptionInstructionReviewDecision"),
            _extract_js_function(js, "validateCaptionInstructionTrainingRows"),
            "const base = {",
            "  image_path: 'train/frame.jpg',",
            "  question: 'Describe the image.',",
            "  answer: 'A grounded answer.',",
            "  metadata: {",
            "    qa_id: 'qa-1',",
            "    row_type: 'generated_qa',",
            "    answer_source: 'vlm_generated',",
            "    answer_format: 'natural',",
            "    source_archive: 'tator_caption_instruction_archive_v1',",
            "    validation_status: 'accepted',",
            "    review_status: 'unreviewed',",
            "  },",
            "};",
            "assert.strictEqual(validateCaptionInstructionTrainingRows([base]).ok, true);",
            "const rejected = validateCaptionInstructionTrainingRows([{ ...base, metadata: { ...base.metadata, validation_status: 'invalid' } }]);",
            "assert.strictEqual(rejected.ok, false);",
            "assert(rejected.errors.some((error) => error.includes('rejected by archive validation')));",
            "const needsRevision = validateCaptionInstructionTrainingRows([{ ...base, metadata: { ...base.metadata, review_decision: 'needs-revision' } }]);",
            "assert.strictEqual(needsRevision.ok, false);",
            "assert(needsRevision.errors.some((error) => error.includes('non-trainable review status')));",
            "const missingMetadata = validateCaptionInstructionTrainingRows([{ image_path: 'train/frame.jpg', question: 'Q?', answer: 'A.' }]);",
            "assert.strictEqual(missingMetadata.ok, false);",
            "assert(missingMetadata.errors.some((error) => error.includes('metadata missing qa_id')));",
            "assert(missingMetadata.errors.some((error) => error.includes('metadata missing row_type')));",
            "assert(missingMetadata.errors.some((error) => error.includes('metadata missing answer_source')));",
            "assert(missingMetadata.errors.some((error) => error.includes('metadata missing source_archive')));",
            "assert(missingMetadata.errors.some((error) => error.includes('metadata missing answer_format')));",
            "assert(missingMetadata.errors.some((error) => error.includes('metadata missing validation_status')));",
            "assert(missingMetadata.errors.some((error) => error.includes('metadata missing review_status')));",
            "const unknownValidation = validateCaptionInstructionTrainingRows([{ ...base, metadata: { ...base.metadata, validation_status: 'maybe' } }]);",
            "assert.strictEqual(unknownValidation.ok, false);",
            "assert(unknownValidation.errors.some((error) => error.includes('validation_status is unsupported')));",
            "const unknownReview = validateCaptionInstructionTrainingRows([{ ...base, metadata: { ...base.metadata, review_status: 'maybe' } }]);",
            "assert.strictEqual(unknownReview.ok, false);",
            "assert(unknownReview.errors.some((error) => error.includes('review_status is unsupported')));",
            "const normalizedDuplicate = validateCaptionInstructionTrainingRows([base, { ...base, question: ' describe   THE image. ', metadata: { ...base.metadata, qa_id: 'qa-2' } }]);",
            "assert.strictEqual(normalizedDuplicate.ok, false);",
            "assert(normalizedDuplicate.errors.some((error) => error.includes('duplicate image_path + question')));",
            "const canonicalImageDuplicate = validateCaptionInstructionTrainingRows([{ ...base, image_path: './train//frame.jpg' }, { ...base, image_path: 'train/frame.jpg', metadata: { ...base.metadata, qa_id: 'qa-3' } }]);",
            "assert.strictEqual(canonicalImageDuplicate.ok, false);",
            "assert.strictEqual(canonicalImageDuplicate.imageCount, 1);",
            "assert(canonicalImageDuplicate.errors.some((error) => error.includes('duplicate image_path + question')));",
            "const missingRows = validateCaptionInstructionTrainingRows(null);",
            "assert.strictEqual(missingRows.ok, false);",
            "assert.strictEqual(missingRows.rowCount, 0);",
            "assert(missingRows.errors.some((error) => error.includes('instruction rows must be an array')));",
            "assert(missingRows.warnings.some((warning) => warning.includes('no instruction rows')));",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_caption_instruction_launch_settings_block_empty_training_family():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            _extract_js_function(js, "validateCaptionInstructionLaunchSettings"),
            _extract_js_function(js, "describeCaptionInstructionLaunchSettings"),
            "const empty = validateCaptionInstructionLaunchSettings({",
            "  instruction_dataset: true,",
            "  include_caption0_in_training: false,",
            "  include_generated_qa_in_training: false,",
            "  include_deterministic_metadata_qa: false,",
            "  subcaptions_per_image: 8,",
            "});",
            "assert.strictEqual(empty.ok, false);",
            "assert(empty.errors.some((error) => error.includes('Enable at least one instruction training row family')));",
            "const archiveOnlyQa = validateCaptionInstructionLaunchSettings({",
            "  instruction_dataset: true,",
            "  include_caption0_in_training: true,",
            "  include_generated_qa_in_training: false,",
            "  include_deterministic_metadata_qa: false,",
            "  subcaptions_per_image: 3,",
            "});",
            "assert.strictEqual(archiveOnlyQa.ok, true);",
            "assert(archiveOnlyQa.warnings.some((warning) => warning.includes('Generated Q&A is disabled')));",
            "const tooMany = validateCaptionInstructionLaunchSettings({",
            "  instruction_dataset: true,",
            "  include_caption0_in_training: true,",
            "  include_generated_qa_in_training: true,",
            "  include_deterministic_metadata_qa: false,",
            "  instruction_qa_restrict_speculative_language: true,",
            "  subcaptions_per_image: 99,",
            "});",
            "assert.strictEqual(tooMany.ok, true);",
            "assert(tooMany.warnings.some((warning) => warning.includes('adjusted from 99 to 20')));",
            "assert(tooMany.warnings.some((warning) => warning.includes('Restrict speculative Q&A language is on')));",
            "const negative = validateCaptionInstructionLaunchSettings({",
            "  instruction_dataset: true,",
            "  include_caption0_in_training: true,",
            "  include_generated_qa_in_training: true,",
            "  include_deterministic_metadata_qa: false,",
            "  subcaptions_per_image_requested: -5,",
            "  subcaptions_per_image: 0,",
            "});",
            "assert.strictEqual(negative.ok, true);",
            "assert(negative.warnings.some((warning) => warning.includes('adjusted from -5 to 0')));",
            "const summary = describeCaptionInstructionLaunchSettings({",
            "  instruction_dataset: true,",
            "  include_caption0_in_training: false,",
            "  include_generated_qa_in_training: false,",
            "  include_deterministic_metadata_qa: true,",
            "  instruction_qa_restrict_speculative_language: true,",
            "  subcaptions_per_image: 3,",
            "});",
            "assert(summary.includes('generated QA disabled'));",
            "assert(summary.includes('deterministic metadata QA rows'));",
            "assert(summary.includes('strict speculative-language filter'));",
            "const batchValidation = validateCaptionInstructionLaunchSettings({",
            "  instruction_dataset: true,",
            "  include_caption0_in_training: true,",
            "  include_generated_qa_in_training: true,",
            "  include_deterministic_metadata_qa: false,",
            "  subcaptions_per_image: 4,",
            "  target_generated_qa_per_image: 8,",
            "  instruction_qa_max_topup_attempts: 6,",
            "}, { provider: 'openai', serviceTier: 'batch' });",
            "assert.strictEqual(batchValidation.ok, true);",
            "assert(batchValidation.warnings.some((warning) => warning.includes('one combined visual request per image')));",
            "assert(batchValidation.warnings.some((warning) => warning.includes('toward a final target of 8')));",
            "const batchSummary = describeCaptionInstructionLaunchSettings({",
            "  instruction_dataset: true,",
            "  include_caption0_in_training: true,",
            "  include_generated_qa_in_training: true,",
            "  include_deterministic_metadata_qa: false,",
            "  subcaptions_per_image: 4,",
            "  instruction_qa_max_topup_attempts: 6,",
            "}, { provider: 'openai', serviceTier: 'batch' });",
            "assert(batchSummary.includes('one Batch request'));",
            "assert(batchSummary.includes('Catch up QA later'));",
            "const existingOnly = describeCaptionInstructionLaunchSettings({",
            "  instruction_dataset: true,",
            "  include_caption0_in_training: false,",
            "  include_generated_qa_in_training: true,",
            "  include_deterministic_metadata_qa: false,",
            "  subcaptions_per_image: 0,",
            "});",
            "assert(existingOnly.includes('existing generated QA rows only'));",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_caption_generated_qa_backend_failures_stop_by_default():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const DEFAULT_CAPTION_INSTRUCTION_MAX_FAILURES = 1;",
            "let qwenElements = { captionMaxFailures: { value: '' } };",
            _extract_js_function(js, "getCaptionBackendMaxFailures"),
            "assert.strictEqual(getCaptionBackendMaxFailures(true, { instruction_dataset: true, subcaptions_per_image: 8 }), 1);",
            "assert.strictEqual(getCaptionBackendMaxFailures(false, { instruction_dataset: true, subcaptions_per_image: 8 }), 0);",
            "assert.strictEqual(getCaptionBackendMaxFailures(true, { instruction_dataset: true, subcaptions_per_image: 0 }), 0);",
            "assert.strictEqual(getCaptionBackendMaxFailures(true, { instruction_dataset: false, subcaptions_per_image: 8 }), 0);",
            "qwenElements.captionMaxFailures.value = '3';",
            "assert.strictEqual(getCaptionBackendMaxFailures(true, { instruction_dataset: true, subcaptions_per_image: 8 }), 3);",
            "qwenElements.captionMaxFailures.value = '0';",
            "assert.strictEqual(getCaptionBackendMaxFailures(true, { instruction_dataset: true, subcaptions_per_image: 8 }), 0);",
            "qwenElements.captionMaxFailures.value = '10000000';",
            "assert.strictEqual(getCaptionBackendMaxFailures(true, { instruction_dataset: true, subcaptions_per_image: 8 }), 1000000);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_caption_instruction_artifact_consistency_blocks_mismatched_exports():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_ID_CHARS = 512;",
            "const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_PATH_CHARS = 4096;",
            "const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_QUESTION_CHARS = 4096;",
            "const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_ANSWER_CHARS = 65536;",
            "const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_NOTES_CHARS = 8192;",
            _extract_js_function(js, "normalizeCaptionInstructionReviewDecision"),
            _extract_js_function(js, "validateCaptionInstructionArchiveRows"),
            _extract_js_function(js, "validateCaptionInstructionReviewRows"),
            _extract_js_function(js, "validateCaptionInstructionReport"),
            _extract_js_function(js, "validateCaptionInstructionArtifactConsistency"),
            "const instructionSettings = {",
            "  include_caption0_in_training: true,",
            "  include_generated_qa_in_training: true,",
            "  include_deterministic_metadata_qa: false,",
            "  instruction_qa_restrict_speculative_language: false,",
            "  qa_mix: 'balanced',",
            "  answer_format: 'natural',",
            "};",
            "const instructionSettingsFingerprint = 'instruction-settings-fingerprint';",
            "const report = {",
            "  format: 'tator_caption_instruction_report_v1',",
            "  image_count: 1,",
            "  selected_flattened_row_count: 1,",
            "  instruction_review_row_count: 1,",
            "  manual_review_required_count: 1,",
            "  corpus_quality_metrics: {",
            "    image_count: 1,",
            "    selected_flattened_row_count: 1,",
            "    rejected_training_row_count: 0,",
            "    generated_qa_candidate_count: 1,",
            "    accepted_generated_qa_count: 1,",
            "    rejected_generated_qa_count: 0,",
            "    generated_qa_question_diversity_ratio: 1,",
            "    generated_qa_acceptance_rate: 1,",
            "    generated_qa_rejection_rate: 0,",
            "    structured_rewrite_rate: 0,",
            "    source_validated_training_row_rate: 1,",
            "    source_class_coverage_rate: 1,",
            "    source_classes: ['Building'],",
            "    source_classes_covered_by_training_rows: ['Building'],",
            "    training_answer_format_distribution: { natural: 1 },",
            "  },",
            "  training_readiness: {",
            "    status: 'ready',",
            "    ready_for_training: true,",
            "    blocking_reasons: [],",
            "    required_actions: [],",
            "    quality_warnings: [],",
            "    thresholds: {},",
            "    selected_training_row_count: 1,",
            "    selected_review_row_count: 1,",
            "    selected_manual_review_row_count: 1,",
            "    accepted_manual_review_row_count: 1,",
            "    pending_manual_review_row_count: 0,",
            "    rejected_manual_review_row_count: 0,",
            "    needs_revision_manual_review_row_count: 0,",
            "    instruction_export_validation_error_count: 0,",
            "  },",
            "  instruction_export_validation: { ok: true, error_count: 0, errors: [], row_count: 1 },",
            "  instruction_artifact_consistency: {",
            "    format: 'tator_caption_instruction_artifact_consistency_v1',",
            "    ok: true,",
            "    error_count: 0,",
            "    errors: [],",
            "  },",
            "  instruction_settings: instructionSettings,",
            "  instruction_settings_fingerprint: instructionSettingsFingerprint,",
            "};",
            "const consistencyOk = report.instruction_artifact_consistency;",
            "assert.strictEqual(validateCaptionInstructionReport(report).ok, true);",
            "const missingReportConsistency = validateCaptionInstructionReport({ ...report, instruction_artifact_consistency: undefined });",
            "assert.strictEqual(missingReportConsistency.ok, false);",
            "assert(missingReportConsistency.errors.some((error) => error.includes('report missing instruction_artifact_consistency')));",
            "const mismatchedReportSelectedCount = validateCaptionInstructionReport({ ...report, selected_flattened_row_count: 2 });",
            "assert.strictEqual(mismatchedReportSelectedCount.ok, false);",
            "assert(mismatchedReportSelectedCount.errors.some((error) => error.includes('corpus_quality_metrics.selected_flattened_row_count does not match report selected_flattened_row_count')));",
            "const readyFlagMismatch = validateCaptionInstructionReport({ ...report, training_readiness: { ...report.training_readiness, ready_for_training: false } });",
            "assert.strictEqual(readyFlagMismatch.ok, false);",
            "assert(readyFlagMismatch.errors.some((error) => error.includes('training_readiness.ready_for_training must be true when status is ready')));",
            "const readyWithWarnings = validateCaptionInstructionReport({ ...report, training_readiness: { ...report.training_readiness, quality_warnings: ['needs review'] } });",
            "assert.strictEqual(readyWithWarnings.ok, false);",
            "assert(readyWithWarnings.errors.some((error) => error.includes('training_readiness ready status cannot include quality_warnings')));",
            "const readyWithPendingReview = validateCaptionInstructionReport({ ...report, training_readiness: { ...report.training_readiness, accepted_manual_review_row_count: 0, pending_manual_review_row_count: 1 } });",
            "assert.strictEqual(readyWithPendingReview.ok, false);",
            "assert(readyWithPendingReview.errors.some((error) => error.includes('training_readiness ready status cannot include unresolved manual review rows')));",
            "const staleReadinessCount = validateCaptionInstructionReport({ ...report, training_readiness: { ...report.training_readiness, selected_training_row_count: 2 } });",
            "assert.strictEqual(staleReadinessCount.ok, false);",
            "assert(staleReadinessCount.errors.some((error) => error.includes('training_readiness.selected_training_row_count does not match report selected_flattened_row_count')));",
            "const blockedFlagMismatch = validateCaptionInstructionReport({ ...report, training_readiness: { ...report.training_readiness, status: 'blocked', ready_for_training: true, blocking_reasons: ['no_selected_training_rows'] } });",
            "assert.strictEqual(blockedFlagMismatch.ok, false);",
            "assert(blockedFlagMismatch.errors.some((error) => error.includes('training_readiness.ready_for_training must be false unless status is ready')));",
            "const blockedWithoutReasons = validateCaptionInstructionReport({ ...report, training_readiness: { ...report.training_readiness, status: 'blocked', ready_for_training: false, blocking_reasons: [] } });",
            "assert.strictEqual(blockedWithoutReasons.ok, false);",
            "assert(blockedWithoutReasons.errors.some((error) => error.includes('training_readiness blocked status requires blocking_reasons')));",
            "const invalidReportConsistency = validateCaptionInstructionReport({ ...report, instruction_artifact_consistency: { format: 'wrong', ok: true, error_count: 0, errors: [] } });",
            "assert.strictEqual(invalidReportConsistency.ok, false);",
            "assert(invalidReportConsistency.errors.some((error) => error.includes('instruction_artifact_consistency format is invalid')));",
            "const failedReportConsistency = validateCaptionInstructionReport({ ...report, instruction_artifact_consistency: { format: 'tator_caption_instruction_artifact_consistency_v1', ok: false, error_count: 1, errors: ['server mismatch'] } });",
            "assert.strictEqual(failedReportConsistency.ok, false);",
            "assert(failedReportConsistency.errors.some((error) => error.includes('instruction_artifact_consistency is not ok')));",
            "const missingReportSettings = validateCaptionInstructionReport({ ...report, instruction_settings: undefined });",
            "assert.strictEqual(missingReportSettings.ok, false);",
            "assert(missingReportSettings.errors.some((error) => error.includes('report missing instruction_settings')));",
            "const trainingRow = {",
            "  image_path: 'frame.jpg',",
            "  question: 'What is shown?',",
            "  answer: 'A building.',",
            "  metadata: {",
            "    qa_id: 'qa-1',",
            "    row_type: 'generated_qa',",
            "    answer_source: 'vlm_generated',",
            "    source_archive: 'tator_caption_instruction_archive_v1',",
            "    answer_format: 'natural',",
            "    validation_status: 'accepted',",
            "    review_status: 'accepted',",
            "  },",
            "};",
            "const archiveRow = {",
            "  image_path: 'frame.jpg',",
            "  source_annotations: {},",
            "  language_annotations: { generated_qa_pairs: [{ qa_id: 'qa-1', question: 'What is shown?', answer: 'A building.' }] },",
            "  deterministic_metadata_qa_pairs: [],",
            "  export_metadata: {",
            "    selected_training_row_count: 1,",
            "    settings: instructionSettings,",
            "    settings_fingerprint: instructionSettingsFingerprint,",
            "  },",
            "};",
            "const reviewRow = {",
            "  format: 'tator_caption_instruction_review_rows_v1',",
            "  dataset_id: 'ds',",
            "  image_path: 'frame.jpg',",
            "  qa_id: 'qa-1',",
            "  row_origin: 'generated_qa',",
            "  question: 'What is shown?',",
            "  candidate_answer: 'A building.',",
            "  training_answer: 'A building.',",
            "  validation_status: 'accepted',",
            "  selected_for_training: true,",
            "  requires_manual_review: true,",
            "  source_summary: {},",
            "  rejection_reasons: [],",
            "  review_decision: 'accepted',",
            "  review_notes: '',",
            "};",
            "const completePayload = {",
            "  instruction_report: report,",
            "  instruction_settings: instructionSettings,",
            "  instruction_settings_fingerprint: instructionSettingsFingerprint,",
            "  instruction_artifact_consistency: consistencyOk,",
            "  instruction_archive: {",
            "    image_count: 1,",
            "    settings: instructionSettings,",
            "    settings_fingerprint: instructionSettingsFingerprint,",
            "    instruction_artifact_consistency: consistencyOk,",
            "  },",
            "  instruction_training_rows: [trainingRow],",
            "  instruction_archive_rows: [archiveRow],",
            "  instruction_review_rows: [reviewRow],",
            "};",
            "const archiveValidation = validateCaptionInstructionArchiveRows([archiveRow]);",
            "assert.strictEqual(archiveValidation.ok, true);",
            "assert.strictEqual(validateCaptionInstructionArtifactConsistency(completePayload, 'archive', archiveValidation).ok, true);",
            "const staleReadinessArtifact = validateCaptionInstructionArtifactConsistency({ ...completePayload, instruction_report: { ...report, training_readiness: { ...report.training_readiness, selected_review_row_count: 0 } } }, 'archive', archiveValidation);",
            "assert.strictEqual(staleReadinessArtifact.ok, false);",
            "assert(staleReadinessArtifact.errors.some((error) => error.includes('training_readiness.selected_review_row_count 0 does not match actual count 1')));",
            "const staleSettings = validateCaptionInstructionArtifactConsistency({ ...completePayload, instruction_settings: { ...instructionSettings, include_generated_qa_in_training: false } }, 'archive', archiveValidation);",
            "assert.strictEqual(staleSettings.ok, false);",
            "assert(staleSettings.errors.some((error) => error.includes('instruction settings disagree between payload and report')));",
            "const duplicateArchive = validateCaptionInstructionArchiveRows([archiveRow, archiveRow]);",
            "assert.strictEqual(duplicateArchive.ok, false);",
            "assert(duplicateArchive.errors.some((error) => error.includes('duplicate archive image_path')));",
            "const duplicateArchiveAlias = validateCaptionInstructionArchiveRows([archiveRow, { ...archiveRow, image_path: './frame.jpg' }]);",
            "assert.strictEqual(duplicateArchiveAlias.ok, false);",
            "assert.strictEqual(duplicateArchiveAlias.imageCount, 1);",
            "assert(duplicateArchiveAlias.errors.some((error) => error.includes('duplicate archive image_path')));",
            "const missingArchiveRows = validateCaptionInstructionArchiveRows(undefined);",
            "assert.strictEqual(missingArchiveRows.ok, false);",
            "assert.strictEqual(missingArchiveRows.rowCount, 0);",
            "assert(missingArchiveRows.errors.some((error) => error.includes('instruction archive rows must be an array')));",
            "assert(missingArchiveRows.warnings.some((warning) => warning.includes('no instruction archive rows')));",
            "const canonicalAliasPayload = {",
            "  ...completePayload,",
            "  instruction_training_rows: [{ ...trainingRow, image_path: './frame.jpg' }],",
            "  instruction_archive_rows: [{ ...archiveRow, image_path: 'frame.jpg' }],",
            "  instruction_review_rows: [{ ...reviewRow, image_path: 'frame.jpg' }],",
            "};",
            "assert.strictEqual(validateCaptionInstructionArtifactConsistency(canonicalAliasPayload, 'training', { rowCount: 1 }).ok, true);",
            "const archiveMismatch = validateCaptionInstructionArtifactConsistency({ ...completePayload, instruction_report: { ...report, image_count: 2, corpus_quality_metrics: { ...report.corpus_quality_metrics, image_count: 2 } }, instruction_archive: { ...completePayload.instruction_archive, image_count: 2 } }, 'archive', archiveValidation);",
            "assert.strictEqual(archiveMismatch.ok, false);",
            "assert(archiveMismatch.errors.some((error) => error.includes('archive row count 1 does not match report image count 2')));",
            "const reviewValidation = validateCaptionInstructionReviewRows([reviewRow]);",
            "assert.strictEqual(reviewValidation.ok, true);",
            "assert.strictEqual(validateCaptionInstructionArtifactConsistency(completePayload, 'review', reviewValidation).ok, true);",
            "const missingReviewRows = validateCaptionInstructionReviewRows({ rows: [reviewRow] });",
            "assert.strictEqual(missingReviewRows.ok, false);",
            "assert.strictEqual(missingReviewRows.rowCount, 0);",
            "assert(missingReviewRows.errors.some((error) => error.includes('instruction review rows must be an array')));",
            "assert(missingReviewRows.warnings.some((warning) => warning.includes('no instruction review rows')));",
            "const reviewMismatch = validateCaptionInstructionArtifactConsistency({ ...completePayload, instruction_report: { ...report, instruction_review_row_count: 2 } }, 'review', reviewValidation);",
            "assert.strictEqual(reviewMismatch.ok, false);",
            "assert(reviewMismatch.errors.some((error) => error.includes('review row count 1 does not match report review row count 2')));",
            "const trainingMismatch = validateCaptionInstructionArtifactConsistency({ ...completePayload, instruction_report: { ...report, selected_flattened_row_count: 2, corpus_quality_metrics: { ...report.corpus_quality_metrics, selected_flattened_row_count: 2 }, instruction_export_validation: { ok: true, error_count: 0, errors: [], row_count: 2 }, training_readiness: { ...report.training_readiness, selected_training_row_count: 2 } } }, 'training', { rowCount: 1 });",
            "assert.strictEqual(trainingMismatch.ok, false);",
            "assert(trainingMismatch.errors.some((error) => error.includes('training row count 1 does not match report selected row count 2')));",
            "const identityMismatch = validateCaptionInstructionArtifactConsistency({ ...completePayload, instruction_review_rows: [{ ...reviewRow, qa_id: 'qa-other' }] }, 'training', { rowCount: 1 });",
            "assert.strictEqual(identityMismatch.ok, false);",
            "assert(identityMismatch.errors.some((error) => error.includes('training row qa_id qa-1 image frame.jpg question \"what is shown?\" is missing from selected review rows')));",
            "assert(identityMismatch.errors.some((error) => error.includes('selected review row qa_id qa-other image frame.jpg question \"what is shown?\" is missing from training rows')));",
            "const staleArchive = validateCaptionInstructionArtifactConsistency({ ...completePayload, instruction_archive_rows: [{ ...archiveRow, language_annotations: { generated_qa_pairs: [{ qa_id: 'qa-1', question: 'What is shown?', answer: 'A stale answer.' }] } }] }, 'training', { rowCount: 1 });",
            "assert.strictEqual(staleArchive.ok, false);",
            "assert(staleArchive.errors.some((error) => error.includes('archive candidate qa_id qa-1 image frame.jpg question \"what is shown?\" answer does not match training row answer')));",
            "const archiveConsistencyMismatch = validateCaptionInstructionArtifactConsistency({ ...completePayload, instruction_archive: { ...completePayload.instruction_archive, instruction_artifact_consistency: { ...consistencyOk, counts: { archive_row_count: 99 } } } }, 'training', { rowCount: 1 });",
            "assert.strictEqual(archiveConsistencyMismatch.ok, false);",
            "assert(archiveConsistencyMismatch.errors.some((error) => error.includes('instruction_artifact_consistency objects disagree between payload and archive')));",
            "const staleCountsConsistency = { ...consistencyOk, counts: { training_row_count: 2, archive_row_count: 1, review_row_count: 1 } };",
            "const staleConsistencyCounts = validateCaptionInstructionArtifactConsistency({ ...completePayload, instruction_artifact_consistency: staleCountsConsistency, instruction_report: { ...report, instruction_artifact_consistency: staleCountsConsistency }, instruction_archive: { ...completePayload.instruction_archive, instruction_artifact_consistency: staleCountsConsistency } }, 'training', { rowCount: 1 });",
            "assert.strictEqual(staleConsistencyCounts.ok, false);",
            "assert(staleConsistencyCounts.errors.some((error) => error.includes('payload instruction_artifact_consistency.counts.training_row_count 2 does not match actual count 1')));",
            "const backendMismatch = validateCaptionInstructionArtifactConsistency({ ...completePayload, instruction_artifact_consistency: { format: 'tator_caption_instruction_artifact_consistency_v1', ok: false, error_count: 1, errors: ['server mismatch'] } }, 'training', { rowCount: 1 });",
            "assert.strictEqual(backendMismatch.ok, false);",
            "assert(backendMismatch.errors.some((error) => error.includes('backend artifact consistency failed: server mismatch')));",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_caption_instruction_review_validator_blocks_bad_actionable_rows():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_ID_CHARS = 512;",
            "const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_PATH_CHARS = 4096;",
            "const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_QUESTION_CHARS = 4096;",
            "const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_ANSWER_CHARS = 65536;",
            "const CAPTION_INSTRUCTION_REVIEW_IMPORT_MAX_NOTES_CHARS = 8192;",
            _extract_js_function(js, "normalizeCaptionInstructionReviewDecision"),
            _extract_js_function(js, "validateCaptionInstructionReviewRows"),
            "const base = {",
            "  format: 'tator_caption_instruction_review_rows_v1',",
            "  dataset_id: 'ds',",
            "  image_path: 'train/frame.jpg',",
            "  split: 'train',",
            "  row_origin: 'generated_qa',",
            "  qa_id: 'qa-1',",
            "  row_type: 'generated_qa',",
            "  question: 'What is shown?',",
            "  candidate_answer: 'A waterfront area.',",
            "  training_answer: 'A waterfront area.',",
            "  validation_status: 'accepted',",
            "  selected_for_training: true,",
            "  requires_manual_review: true,",
            "  review_decision: 'accepted',",
            "  review_notes: '',",
            "  rejection_reasons: [],",
            "  source_summary: { status: 'ok' },",
            "};",
            "const unsupported = validateCaptionInstructionReviewRows([{ ...base, row_origin: 'freeform_review' }]);",
            "assert.strictEqual(unsupported.ok, false);",
            "assert(unsupported.errors.some((error) => error.includes('unsupported actionable row_origin')));",
            "const typoDecision = validateCaptionInstructionReviewRows([{ ...base, review_decision: 'acceppted' }]);",
            "assert.strictEqual(typoDecision.ok, false);",
            "assert(typoDecision.errors.some((error) => error.includes('unsupported review_decision')));",
            "const blankDecision = validateCaptionInstructionReviewRows([{ ...base, review_decision: '' }]);",
            "assert.strictEqual(blankDecision.ok, true);",
            "const blankWithoutDataset = validateCaptionInstructionReviewRows([{ ...base, dataset_id: '', review_decision: '' }]);",
            "assert.strictEqual(blankWithoutDataset.ok, false);",
            "assert(blankWithoutDataset.errors.some((error) => error.includes('missing dataset_id for persisted language review row')));",
            "const duplicate = validateCaptionInstructionReviewRows([base, { ...base, row_type: 'external_edit' }]);",
            "assert.strictEqual(duplicate.ok, false);",
            "assert(duplicate.errors.some((error) => error.includes('duplicate actionable review target')));",
            "const aliasDuplicate = validateCaptionInstructionReviewRows([base, { ...base, image_path: './train//frame.jpg', split: '', row_type: 'external_edit' }]);",
            "assert.strictEqual(aliasDuplicate.ok, false);",
            "assert.strictEqual(aliasDuplicate.imageCount, 1);",
            "assert(aliasDuplicate.errors.some((error) => error.includes('duplicate actionable review target')));",
            "assert(aliasDuplicate.errors.some((error) => error.includes('duplicate image_path + qa_id')));",
            "const conflicting = validateCaptionInstructionReviewRows([base, { ...base, review_decision: 'rejected' }]);",
            "assert.strictEqual(conflicting.ok, false);",
            "assert(conflicting.errors.some((error) => error.includes('conflicting duplicate actionable review target')));",
            "const invalidNotes = validateCaptionInstructionReviewRows([{ ...base, review_notes: 123 }]);",
            "assert.strictEqual(invalidNotes.ok, false);",
            "assert(invalidNotes.errors.some((error) => error.includes('review_notes must be text')));",
            "const longNotes = validateCaptionInstructionReviewRows([{ ...base, review_notes: 'x'.repeat(8193) }]);",
            "assert.strictEqual(longNotes.ok, false);",
            "assert(longNotes.errors.some((error) => error.includes('review_notes exceeds 8192 characters')));",
            "const whitespaceLongNotes = validateCaptionInstructionReviewRows([{ ...base, review_notes: ' '.repeat(8193) }]);",
            "assert.strictEqual(whitespaceLongNotes.ok, false);",
            "assert(whitespaceLongNotes.errors.some((error) => error.includes('review_notes exceeds 8192 characters')));",
            "const longQuestion = validateCaptionInstructionReviewRows([{ ...base, question: 'x'.repeat(4097) }]);",
            "assert.strictEqual(longQuestion.ok, false);",
            "assert(longQuestion.errors.some((error) => error.includes('question exceeds 4096 characters')));",
            "const whitespaceLongQuestion = validateCaptionInstructionReviewRows([{ ...base, question: ' '.repeat(4097) }]);",
            "assert.strictEqual(whitespaceLongQuestion.ok, false);",
            "assert(whitespaceLongQuestion.errors.some((error) => error.includes('question exceeds 4096 characters')));",
            "const missingBundleOriginal = validateCaptionInstructionReviewRows([{ ...base, image_path: 'images/train/frame.jpg', bundle_image_sha256: 'a'.repeat(64) }]);",
            "assert.strictEqual(missingBundleOriginal.ok, false);",
            "assert(missingBundleOriginal.errors.some((error) => error.includes('copied bundle image')));",
            "const invalidBundleHash = validateCaptionInstructionReviewRows([{ ...base, original_image_path: 'frame.jpg', bundle_image_sha256: 'not-a-sha' }]);",
            "assert.strictEqual(invalidBundleHash.ok, false);",
            "assert(invalidBundleHash.errors.some((error) => error.includes('64-character SHA-256 digest')));",
            "const bundleAlias = validateCaptionInstructionReviewRows([{ ...base, image_path: 'images/train/frame.jpg', original_image_path: 'frame.jpg', bundle_image_sha256: 'a'.repeat(64) }]);",
            "assert.strictEqual(bundleAlias.ok, true);",
            "const deterministic = validateCaptionInstructionReviewRows([{ ...base, row_origin: 'deterministic_metadata_qa', qa_id: 'meta-1', row_type: 'deterministic_count', selected_for_training: false, requires_manual_review: false }]);",
            "assert.strictEqual(deterministic.ok, true);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_caption_instruction_readiness_summary_formats_blockers_for_operators():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            _extract_js_function(js, "captionInstructionReadinessLabel"),
            _extract_js_function_before(
                js,
                "captionInstructionReadinessSummary",
                "\n    async function downloadCaptionJsonl",
            ),
            "const summary = captionInstructionReadinessSummary({ training_readiness: {",
            "  status: 'blocked',",
            "  ready_for_training: false,",
            "  blocking_reasons: ['selected_row_needs_revision_by_manual_review'],",
            "  required_actions: ['revise_selected_language_rows'],",
            "  quality_warnings: [],",
            "} });",
            "assert.strictEqual(summary.blocked, true);",
            "assert.strictEqual(summary.status, 'blocked');",
            "assert.strictEqual(summary.severity, 'fail');",
            "assert(summary.message.includes('a selected row needs revision'));",
            "assert(!summary.message.includes('selected_row_needs_revision_by_manual_review'));",
            "const needsReview = captionInstructionReadinessSummary({ training_readiness: {",
            "  status: 'needs_review',",
            "  ready_for_training: false,",
            "  pending_manual_review_row_count: 2,",
            "  blocking_reasons: [],",
            "  required_actions: ['review_selected_language_rows'],",
            "  quality_warnings: [],",
            "} });",
            "assert.strictEqual(needsReview.status, 'needs_review');",
            "assert.strictEqual(needsReview.blocked, false);",
            "assert(needsReview.message.includes('2 selected language rows pending review'));",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_caption_instruction_readiness_chip_does_not_treat_zero_fail_as_blocked():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const qwenElements = { captionReadinessStatus: { textContent: '' } };",
            _extract_js_function(js, "captionInstructionReadinessChip"),
            "function chip(text) { qwenElements.captionReadinessStatus.textContent = text; return captionInstructionReadinessChip(); }",
            "assert.deepStrictEqual(chip('Readiness not checked.'), { text: 'Readiness not checked', status: 'pending' });",
            "assert.deepStrictEqual(chip('Caption readiness: 52 pass, 0 warnings, 0 fail.'), { text: 'Readiness passed', status: 'pass' });",
            "assert.deepStrictEqual(chip('Caption readiness: 51 pass, 1 warning, 0 fail.'), { text: 'Readiness needs review', status: 'warn' });",
            "assert.deepStrictEqual(chip('Caption readiness: 49 pass, 0 warnings, 2 fail.'), { text: 'Readiness blocked', status: 'fail' });",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_single_caption_uses_isolated_backend_job_when_dataset_backed():
    js = _js()

    assert "async function runQwenCaptionSingleBackendJob" in js
    helper_start = js.index("async function runQwenCaptionSingleBackendJob")
    helper_end = js.index("async function runQwenCaptionBackendBatch", helper_start)
    helper = js[helper_start:helper_end]
    assert "/qwen/caption/jobs" in helper
    assert "Set-and-forget captioning requires a selected caption dataset" in helper
    assert "image_names: [imageName]" in helper
    assert "save_text_labels: qwenElements.captionSaveText?.checked !== false" in helper
    assert "set_and_forget: setAndForget" in helper
    assert "runner_heartbeat_interval_seconds: DEFAULT_CAPTION_BACKEND_HEARTBEAT_SECONDS" in helper
    assert "qwenBackendCrashSupervisionMessage()" in helper
    assert "Isolated caption job auto-resumed as ${autoResumeJobId}" in helper
    assert "latest_caption" in helper
    assert "caption backend job completed without a caption" in helper

    handle_start = js.index("async function handleQwenCaption")
    handle_end = js.index("function getCaptionImageList", handle_start)
    handle = js[handle_start:handle_end]
    assert "guardQwenCaptionArchiveIdle(\"starting another caption job\")" in handle
    assert "runQwenCaptionSingleBackendJob(" in handle
    assert "datasetBackedResult || await invokeQwenCaption" not in handle
    assert "if (!result)" in handle
    assert "qwenElements.captionSetAndForget?.checked !== false" in handle
    assert "Running one-off direct caption request; set-and-forget is disabled." in handle
    assert "await invokeQwenCaption" in handle


def test_qwen_next_n_caption_prefers_resumable_backend_job():
    js = _js()

    listener_start = js.index("qwenElements.captionBatchRun.addEventListener")
    listener_end = js.index("if (qwenElements.captionBatchRunAll)", listener_start)
    listener = js[listener_start:listener_end]
    click_guard = 'guardQwenCaptionArchiveIdle("starting a caption batch job")'
    assert click_guard in listener
    assert listener.index(click_guard) < listener.index("const includeCurrent")
    assert listener.index(click_guard) < listener.index("runQwenCaptionBatch(batch")
    assert "runQwenCaptionBatch(batch" in listener
    assert "backend: true" in listener

    batch_start = js.index("async function runQwenCaptionBatch")
    batch_end = js.index("function setQwenAgentStatus", batch_start)
    batch = js[batch_start:batch_end]
    assert "Backend dataset required" in batch
    assert "batch captioning uses isolated backend jobs so Metal crashes cannot take down" in batch
    assert "validateCaptionInstructionLaunchSettings(getCaptionInstructionDatasetSettings(true), getCaptionProviderSettings())" in batch
    assert "guardQwenCaptionArchiveIdle(" in batch
    assert "starting a training dataset job" in batch
    assert "starting another caption batch" in batch
    assert "Instruction dataset not started" in batch
    assert "try {" in batch
    assert "runQwenCaptionBackendBatch(imageNames, { ...options, backend: true })" in batch
    assert "formatBackendFetchError(error" in batch
    assert "training dataset" in batch
    assert "failed to start" in batch
    assert "setQwenCaptionBackendJobStatus(message)" in batch
    assert "renderQwenCaptionBackendJobProgress(currentJob" in js
    assert "renderQwenCaptionBackendJobProgress(job" in js
    assert "isQwenCaptionBackendJobHardFailure" in js
    assert "hasQwenCaptionBackendResumeEvidence" in js
    assert "Backend job is not resumable" in js
    assert "!isQwenCaptionBackendJobRecoverable(job)" in js
    assert "caption_runner_preflight_failed" in js
    assert "Caption all images" in _html()
    assert "invokeQwenCaptionForImage(" not in batch


def test_qwen_all_image_caption_and_instruction_runs_start_with_selected_image():
    js = _js()
    html = _html()

    run_all_start = js.index("qwenElements.captionBatchRunAll.addEventListener")
    run_all_end = js.index("if (qwenElements.captionBuildInstructionDataset)", run_all_start)
    run_all_listener = js[run_all_start:run_all_end]
    instruction_start = run_all_end
    instruction_end = js.index("if (qwenElements.captionBatchCancel)", instruction_start)
    instruction_listener = js[instruction_start:instruction_end]

    assert "getCaptionImageList({ startAtCurrent: true })" in run_all_listener
    assert "Caption all ${imageNames.length} images starting with ${firstImage}" in run_all_listener
    assert 'ensureAutomationAvailable("training dataset creation")' in instruction_listener
    assert "Qwen backend is unavailable" in instruction_listener
    assert "Select or register a caption dataset before creating a training dataset." in instruction_listener
    assert "getCaptionImageList({ startAtCurrent: true })" in instruction_listener
    assert "const runImageCount = settings.max_images" in instruction_listener
    assert "Create a training dataset for ${runImageCount} image${runImageCount === 1 ? \"\" : \"s\"} starting with ${firstImage}" in instruction_listener
    assert instruction_listener.index("Select or register a caption dataset before creating a training dataset.") < instruction_listener.index("confirm(")
    assert "All-image caption and training-dataset jobs start with the selected image" in html
    assert "viewer advances to each backend case" in html

    script = "\n".join(
        [
            "const assert = require('assert');",
            _extract_js_function(js, "rotateCaptionImageNamesFromIndex"),
            "assert.deepStrictEqual(rotateCaptionImageNamesFromIndex(['a.jpg', 'b.jpg', 'c.jpg'], 1), ['b.jpg', 'c.jpg', 'a.jpg']);",
            "assert.deepStrictEqual(rotateCaptionImageNamesFromIndex(['a.jpg', 'b.jpg', 'c.jpg'], 0), ['a.jpg', 'b.jpg', 'c.jpg']);",
            "assert.deepStrictEqual(rotateCaptionImageNamesFromIndex(['a.jpg', 'b.jpg', 'c.jpg'], 5), ['a.jpg', 'b.jpg', 'c.jpg']);",
            "assert.deepStrictEqual(rotateCaptionImageNamesFromIndex([], 2), []);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_caption_launches_block_while_archive_is_mutating():
    js = _js()
    handle_start = js.index("async function handleQwenCaption()")
    handle_end = js.index("function getCaptionImageList", handle_start)
    batch_start = js.index("async function runQwenCaptionBatch")
    batch_end = js.index("function setQwenAgentStatus", batch_start)
    script = "\n".join(
        [
            "const assert = require('assert');",
            "let qwenCaptionActive = false;",
            "let qwenCaptionBatchActive = false;",
            "let qwenCaptionBatchBackendJobId = 'job-1';",
            "let qwenAvailable = true;",
            "let automationChecked = 0;",
            "let backendLaunches = 0;",
            "let updateCalls = 0;",
            "const captionStatuses = [];",
            "const backendStatuses = [];",
            "const samStatuses = [];",
            "function setQwenCaptionStatus(message) { captionStatuses.push(message); }",
            "function setQwenCaptionBackendJobStatus(message) { backendStatuses.push(message); }",
            "function setSamStatus(message, options) { samStatuses.push({ message, options }); }",
            "function updateQwenCaptionButton() { updateCalls += 1; }",
            "function ensureAutomationAvailable() { automationChecked += 1; return true; }",
            "function getCaptionDatasetId() { return 'ds'; }",
            "async function runQwenCaptionBackendBatch() { backendLaunches += 1; throw new Error('backend launch should be blocked'); }",
            _extract_js_function(js, "qwenCaptionArchiveMutationActive"),
            _extract_js_function(js, "captionArchiveMutationBusyMessage"),
            _extract_js_function(js, "guardQwenCaptionArchiveIdle"),
            js[handle_start:handle_end],
            js[batch_start:batch_end],
            "await handleQwenCaption();",
            "assert.strictEqual(automationChecked, 0);",
            "assert.strictEqual(backendLaunches, 0);",
            "assert(captionStatuses.includes('Caption archive busy'));",
            "assert(backendStatuses.some((message) => message.includes('starting another caption job')));",
            "assert(samStatuses.some((entry) => entry.message.includes('caption archive is changing')));",
            "await runQwenCaptionBatch(['frame.jpg'], { backend: true });",
            "assert.strictEqual(backendLaunches, 0);",
            "assert(backendStatuses.some((message) => message.includes('starting another caption batch')));",
            "await runQwenCaptionBatch(['frame.jpg'], { backend: true, instructionDataset: true });",
            "assert.strictEqual(backendLaunches, 0);",
            "assert(backendStatuses.some((message) => message.includes('starting a training dataset job')));",
            "assert(updateCalls >= 3);",
        ]
    )
    subprocess.run(
        [
            "node",
            "-e",
            f"(async () => {{\n{script}\n}})().catch((error) => {{ console.error(error); process.exit(1); }});",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_qwen_caption_cancel_handles_detached_backend_jobs():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const API_ROOT = 'http://api.test';",
            "let qwenCaptionActive = false;",
            "let qwenCaptionBatchActive = false;",
            "let qwenCaptionBatchBackendJobId = 'qcap_child';",
            "let qwenCaptionCancelRequested = false;",
            "let qwenCaptionBatchCancel = false;",
            "let qwenCaptionAbortController = null;",
            "let qwenCaptionBatchAbortController = null;",
            "let qwenProgressActiveContext = { kind: 'caption' };",
            "let qwenCaptionBackendActiveJobs = [",
            "  { job_id: 'qcap_child', status: 'running' },",
            "  { job_id: 'qcap_source', status: 'interrupted' },",
            "  { job_id: 'qcap_done', status: 'completed' },",
            "];",
            "const fetchCalls = [];",
            "let refreshCalls = 0;",
            "let updateCalls = 0;",
            "global.fetch = async (url, options = {}) => { fetchCalls.push({ url: String(url), method: options.method || 'GET' }); return { ok: true, json: async () => [] }; };",
            "function updateQwenCaptionButton() { updateCalls += 1; }",
            "function renderQwenCaptionLocalProgress() {}",
            "function setQwenCaptionStatus() {}",
            "function setSamStatus() {}",
            "function renderQwenCaptionProgressState() {}",
            "function hideQwenCaptionLiveToast() {}",
            "function getQwenCaptionLiveToastTerminalSignature() { return 'cancelled'; }",
            "function stopQwenProgressPolling() {}",
            "async function refreshQwenCaptionBackendJobsStatus() { refreshCalls += 1; return []; }",
            _extract_js_function(js, "isQwenCaptionBackendJobActive"),
            _extract_js_function(js, "qwenCaptionArchiveMutationActive"),
            _extract_js_function(js, "qwenCaptionKnownBackendJobIds"),
            _extract_js_function(js, "makeCaptionAbortError"),
            "async " + _extract_js_function_before(
                js,
                "requestQwenCaptionCancel",
                "\n    function formatQwenMemory",
            ),
            "assert.strictEqual(qwenCaptionArchiveMutationActive(), true);",
            "assert.deepStrictEqual(qwenCaptionKnownBackendJobIds(), ['qcap_child']);",
            "await requestQwenCaptionCancel({ force: false });",
            "assert.deepStrictEqual(fetchCalls.map((call) => call.url), [",
            "  'http://api.test/qwen/caption/jobs/qcap_child/cancel',",
            "]);",
            "assert(fetchCalls.every((call) => call.method === 'POST'));",
            "assert.strictEqual(qwenCaptionBatchCancel, false);",
            "assert.strictEqual(qwenCaptionCancelRequested, false);",
            "assert.strictEqual(qwenCaptionBatchBackendJobId, null);",
            "assert.deepStrictEqual(qwenCaptionBackendActiveJobs, []);",
            "assert.strictEqual(refreshCalls, 1);",
            "assert(updateCalls >= 2);",
        ]
    )
    subprocess.run(
        [
            "node",
            "-e",
            f"(async () => {{\n{script}\n}})().catch((error) => {{ console.error(error); process.exit(1); }});",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_qwen_caption_interrupted_backend_jobs_do_not_block_dataset_controls():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "let qwenCaptionActive = false;",
            "let qwenCaptionBatchActive = false;",
            "let qwenCaptionBatchBackendJobId = null;",
            "let qwenCaptionDatasetRefreshInFlight = false;",
            "let qwenCaptionBackendActiveJobs = [",
            "  { job_id: 'qcap_interrupted', status: 'interrupted' },",
            "];",
            "const qwenElements = {",
            "  captionDatasetSelect: { disabled: true },",
            "  captionDatasetRefresh: { disabled: true },",
            "};",
            "function isAnnotationDatasetModeActive() { return false; }",
            _extract_js_function(js, "isQwenCaptionBackendJobActive"),
            _extract_js_function(js, "qwenCaptionArchiveMutationActive"),
            _extract_js_function(js, "syncQwenCaptionDatasetControls"),
            "assert.strictEqual(isQwenCaptionBackendJobActive(qwenCaptionBackendActiveJobs[0]), false);",
            "assert.strictEqual(qwenCaptionArchiveMutationActive(), false);",
            "syncQwenCaptionDatasetControls();",
            "assert.strictEqual(qwenElements.captionDatasetSelect.disabled, false);",
            "assert.strictEqual(qwenElements.captionDatasetRefresh.disabled, false);",
        ]
    )
    subprocess.run(
        [
            "node",
            "-e",
            f"(async () => {{\n{script}\n}})().catch((error) => {{ console.error(error); process.exit(1); }});",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_qwen_caption_cancel_backend_jobs_clears_local_backend_job_lock():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const API_ROOT = 'http://api.test';",
            "let qwenCaptionActive = false;",
            "let qwenCaptionBatchActive = true;",
            "let qwenCaptionBatchBackendJobId = 'qcap_child';",
            "let qwenCaptionCancelRequested = true;",
            "let qwenCaptionBatchCancel = false;",
            "let qwenCaptionBackendActiveJobs = [];",
            "const activeJobs = [{ job_id: 'qcap_child', status: 'running' }];",
            "const fetchCalls = [];",
            "let refreshCalls = 0;",
            "let updateCalls = 0;",
            "global.window = { confirm: () => true };",
            "global.fetch = async (url, options = {}) => {",
            "  fetchCalls.push({ url: String(url), method: options.method || 'GET' });",
            "  return { ok: true, text: async () => '', json: async () => [] };",
            "};",
            "async function fetchQwenCaptionBackendJobs() { return activeJobs; }",
            "function summarizeQwenCaptionBackendJobs() { return { activeJobs, message: 'active' }; }",
            "function updateQwenCaptionButton() { updateCalls += 1; }",
            "function qwenCaptionBackendJobShortLabel(job) { return `${job.job_id} (${job.status})`; }",
            "function setQwenCaptionBackendJobStatus() {}",
            "function setSamStatus() {}",
            "function parseApiError(detail, fallback) { return detail || fallback; }",
            "async function refreshQwenCaptionBackendJobsStatus() { refreshCalls += 1; qwenCaptionBackendActiveJobs = []; return []; }",
            _extract_js_function(js, "isQwenCaptionBackendJobActive"),
            _extract_js_function(js, "qwenCaptionArchiveMutationActive"),
            "async " + _extract_js_function(js, "cancelQwenCaptionBackendActiveJobs"),
            "assert.strictEqual(qwenCaptionArchiveMutationActive(), true);",
            "await cancelQwenCaptionBackendActiveJobs();",
            "assert.deepStrictEqual(fetchCalls, [{ url: 'http://api.test/qwen/caption/jobs/qcap_child/cancel', method: 'POST' }]);",
            "assert.strictEqual(qwenCaptionBatchActive, false);",
            "assert.strictEqual(qwenCaptionBatchBackendJobId, null);",
            "assert.strictEqual(qwenCaptionCancelRequested, false);",
            "assert.strictEqual(qwenCaptionBatchCancel, false);",
            "assert.strictEqual(refreshCalls, 1);",
            "assert.strictEqual(qwenCaptionArchiveMutationActive(), false);",
            "assert(updateCalls >= 2);",
        ]
    )
    subprocess.run(
        [
            "node",
            "-e",
            f"(async () => {{\n{script}\n}})().catch((error) => {{ console.error(error); process.exit(1); }});",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_qwen_caption_recipe_load_and_upload_block_while_archive_is_mutating():
    js = _js()
    apply_start = js.index("function applyCaptionRecipeToUi")
    apply_end = js.index("function saveCurrentCaptionRecipe", apply_start)
    script = "\n".join(
        [
            "const assert = require('assert');",
            "let qwenCaptionActive = false;",
            "let qwenCaptionBatchActive = false;",
            "let qwenCaptionBatchBackendJobId = 'job-1';",
            "let updateCalls = 0;",
            "let fileReads = 0;",
            "let recipeStatus = '';",
            "const captionStatuses = [];",
            "const backendStatuses = [];",
            "const samStatuses = [];",
            "const qwenElements = { captionRecipeName: { value: 'unchanged' } };",
            "function setCaptionRecipeStatus(message) { recipeStatus = message; }",
            "function setQwenCaptionStatus(message) { captionStatuses.push(message); }",
            "function setQwenCaptionBackendJobStatus(message) { backendStatuses.push(message); }",
            "function setSamStatus(message, options) { samStatuses.push({ message, options }); }",
            "function updateQwenCaptionButton() { updateCalls += 1; }",
            "function readFileAsTextPromise() { fileReads += 1; throw new Error('recipe upload should be blocked before file read'); }",
            _extract_js_function(js, "qwenCaptionArchiveMutationActive"),
            _extract_js_function(js, "captionArchiveMutationBusyMessage"),
            _extract_js_function(js, "guardQwenCaptionArchiveIdle"),
            js[apply_start:apply_end],
            "async " + _extract_js_function(js, "uploadCaptionRecipeFromFile"),
            "const applied = applyCaptionRecipeToUi({ recipe: { name: 'new recipe' } }, { actionLabel: 'loading a caption recipe' });",
            "assert.strictEqual(applied, false);",
            "assert.strictEqual(qwenElements.captionRecipeName.value, 'unchanged');",
            "assert(recipeStatus.includes('loading a caption recipe'));",
            "assert(recipeStatus.includes('caption archive is changing'));",
            "assert(captionStatuses.includes('Caption archive busy'));",
            "await uploadCaptionRecipeFromFile({ name: 'blocked.caption-recipe.json' });",
            "assert.strictEqual(fileReads, 0);",
            "assert(recipeStatus.includes('uploading a caption recipe'));",
            "assert(backendStatuses.some((message) => message.includes('caption archive is changing')));",
            "assert(samStatuses.some((entry) => entry.message.includes('caption archive is changing')));",
            "assert(updateCalls >= 2);",
        ]
    )
    subprocess.run(
        [
            "node",
            "-e",
            f"(async () => {{\n{script}\n}})().catch((error) => {{ console.error(error); process.exit(1); }});",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_qwen_caption_glossary_actions_block_while_archive_is_mutating():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "let qwenCaptionActive = false;",
            "let qwenCaptionBatchActive = false;",
            "let qwenCaptionBatchBackendJobId = 'job-1';",
            "let updateCalls = 0;",
            "let fetchCalls = 0;",
            "const captionStatuses = [];",
            "const backendStatuses = [];",
            "const samStatuses = [];",
            "const qwenCaptionGlossaryState = { datasetId: 'ds', dirty: false, loadRequestId: 0, saveInFlight: false, source: 'dataset', text: 'stable glossary' };",
            "const qwenElements = {",
            "  captionGlossary: { value: 'stable glossary', disabled: false },",
            "  captionGlossaryStatus: { textContent: '' },",
            "  captionGlossaryReset: { disabled: false },",
            "  captionGlossarySave: { disabled: false, textContent: '' },",
            "};",
            "function getCaptionGlossaryDatasetId() { return 'ds'; }",
            "function getCaptionGlossaryLabelmap() { return ['Boat']; }",
            "function buildDefaultCaptionGlossary() { return '{\"Boat\":[\"boat\"]}'; }",
            "function setQwenCaptionStatus(message) { captionStatuses.push(message); }",
            "function setQwenCaptionBackendJobStatus(message) { backendStatuses.push(message); }",
            "function setSamStatus(message, options) { samStatuses.push({ message, options }); }",
            "function updateQwenCaptionButton() { updateCalls += 1; }",
            "function updateQwenCaptionPromptStack() { throw new Error('prompt stack should not update while blocked'); }",
            "function naturalizeCaptionGlossaryLabel(label) { return label; }",
            "function dedupeCaptionGlossaryTerms(terms) { return terms; }",
            "function parseApiError(detail) { return detail; }",
            "global.fetch = async () => { fetchCalls += 1; throw new Error('fetch should be blocked'); };",
            _extract_js_function(js, "qwenCaptionArchiveMutationActive"),
            _extract_js_function(js, "captionArchiveMutationBusyMessage"),
            _extract_js_function(js, "guardQwenCaptionArchiveIdle"),
            _extract_js_function(js, "setCaptionGlossaryStatus"),
            _extract_js_function(js, "updateCaptionGlossaryControls"),
            _extract_js_function(js, "resetCaptionGlossaryFromClasses"),
            "async " + _extract_js_function(js, "saveCaptionGlossaryToDataset"),
            "const resetResult = resetCaptionGlossaryFromClasses();",
            "assert.strictEqual(resetResult, false);",
            "assert.strictEqual(qwenElements.captionGlossary.value, 'stable glossary');",
            "assert(captionStatuses.includes('Caption archive busy'));",
            "assert(backendStatuses.some((message) => message.includes('resetting the caption glossary')));",
            "assert(samStatuses.some((entry) => entry.message.includes('caption archive is changing')));",
            "const saveResult = await saveCaptionGlossaryToDataset();",
            "assert.strictEqual(saveResult, false);",
            "assert.strictEqual(fetchCalls, 0);",
            "assert(backendStatuses.some((message) => message.includes('saving the caption glossary')));",
            "assert(updateCalls >= 2);",
        ]
    )
    subprocess.run(
        [
            "node",
            "-e",
            f"(async () => {{\n{script}\n}})().catch((error) => {{ console.error(error); process.exit(1); }});",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_qwen_caption_text_autosave_blocks_while_archive_is_mutating():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "let qwenCaptionActive = false;",
            "let qwenCaptionBatchActive = false;",
            "let qwenCaptionBatchBackendJobId = 'job-1';",
            "let updateCalls = 0;",
            "let fetchCalls = 0;",
            "const captionStatuses = [];",
            "const backendStatuses = [];",
            "const samStatuses = [];",
            "const qwenElements = { captionSaveText: { checked: true } };",
            "const captionAutoSaveState = { timerId: null, pendingImage: null, lastSaved: new Map(), lastAttempted: new Map() };",
            "function setQwenCaptionStatus(message) { captionStatuses.push(message); }",
            "function setQwenCaptionBackendJobStatus(message) { backendStatuses.push(message); }",
            "function setSamStatus(message, options) { samStatuses.push({ message, options }); }",
            "function updateQwenCaptionButton() { updateCalls += 1; }",
            "function formatBackendFetchError(error) { return error?.message || String(error); }",
            "global.fetch = async () => { fetchCalls += 1; throw new Error('fetch should be blocked'); };",
            _extract_js_function(js, "qwenCaptionArchiveMutationActive"),
            _extract_js_function(js, "captionArchiveMutationBusyMessage"),
            _extract_js_function(js, "guardQwenCaptionArchiveIdle"),
            "async " + _extract_js_function_before(
                js,
                "saveCaptionImmediate",
                "\n    function scheduleCaptionAutosave",
            ),
            _extract_js_function(js, "scheduleCaptionAutosave"),
            "const saved = await saveCaptionImmediate('frame.jpg', 'edited caption');",
            "assert.strictEqual(saved, false);",
            "assert.strictEqual(fetchCalls, 0);",
            "assert(captionStatuses.includes('Caption archive busy'));",
            "assert(backendStatuses.some((message) => message.includes('saving caption text edits')));",
            "scheduleCaptionAutosave('frame.jpg', 'edited caption');",
            "assert.strictEqual(captionAutoSaveState.timerId, null);",
            "assert.strictEqual(fetchCalls, 0);",
            "assert(updateCalls >= 2);",
        ]
    )
    subprocess.run(
        [
            "node",
            "-e",
            f"(async () => {{\n{script}\n}})().catch((error) => {{ console.error(error); process.exit(1); }});",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_top_navigation_tabs_have_tooltips():
    html = _html()
    css = _css()
    js = _js()
    tab_buttons = re.findall(r"<button\b[^>]*\bclass=\"[^\"]*\btab-button\b[^\"]*\"[^>]*>", html)

    assert tab_buttons
    missing_titles = [tag for tag in tab_buttons if 'title="' not in tag]
    assert not missing_titles
    assert "Open the main annotation workspace" in html
    assert "Score new images and videos against a reference dataset" in html
    assert "Embed labeled objects, inspect likely wrong classes" in html
    assert "overflow-x: auto;" in css
    assert "scrollbar-gutter: stable;" in css
    utility_block = css[css.index(".tab-bar__utility"):css.index(".theme-toggle-button")]
    assert "margin-left: auto;" not in utility_block
    assert "TOP_TAB_BASE_METRICS" not in js
    assert "adaptiveTopTabs" not in js
    assert "setAdaptiveTopTabsScale" not in js
    assert "scheduleAdaptiveTopTabsUpdate" not in js
    assert "measureAdaptiveTopTabsWidth" not in js
    assert "availableWidth / naturalWidth" not in js
    assert "button.dataset.automationUnlockedTitle = button.getAttribute(\"title\") || \"\";" in js
    assert "button.title = restoredTitle;" in js
    assert 'button.title = "";' not in js
    assert 'activeElements.clipSelect.removeAttribute("title");' in js
    assert "refreshUiTooltips(activeElements.clipSelect);" in js
    assert 'agentElements.stepsPromptPrefilter.removeAttribute("title");' in js
    assert "refreshUiTooltips(agentElements.stepsPromptPrefilter);" in js
    assert 'activeElements.clipSelect.title = "";' not in js
    assert 'agentElements.stepsPromptPrefilter.title = "";' not in js


def test_static_get_element_by_id_bindings_exist_in_tator_html():
    html_ids = _static_html_ids(_html())
    js_refs = _static_get_element_by_id_refs(_js())
    missing = sorted(js_refs - html_ids - DYNAMIC_JS_CREATED_IDS)

    assert missing == []


def test_all_tator_buttons_declare_type_to_avoid_form_submit_fallbacks():
    root = _parse_static_html()
    missing = [
        _describe_control(button)
        for button in _nodes_by_tag(root, "button")
        if not str(button.attrs.get("type") or "").strip()
    ]

    assert missing == []


def test_ui_only_forms_prevent_browser_submit_navigation():
    js = _js()

    assert "function preventUiOnlyFormSubmits()" in js
    assert 'form.addEventListener("submit", (event) => {' in js
    assert "event.preventDefault();" in js
    assert "preventUiOnlyFormSubmits();" in js


def test_yolo_import_and_export_controls_live_in_annotation_source_panel():
    html = _html()
    source_start = html.index('id="annotationSourcePanel"')
    source_end = html.index('id="labelingGpuLockNotice"')
    source_panel = html[source_start:source_end]

    for control_id in ("bboxes", "bboxesFolder", "bboxesSelectFolder", "saveBboxes"):
        assert f'id="{control_id}"' in source_panel

    assert source_start < html.index('id="bboxes"') < html.index('id="qwenDetectionDetails"')
    assert source_start < html.index('id="saveBboxes"') < html.index('id="qwenDetectionDetails"')


def test_dataset_yolo_export_reuses_manifest_rows_without_loading_every_image():
    js = _js()
    open_start = js.index("async function openDatasetInAnnotationMode")
    open_end = js.index("async function openDatasetEntryInAnnotation", open_start)
    open_body = js[open_start:open_end]

    assert "const imageWidth = Number(row.image_width);" in open_body
    assert "const imageHeight = Number(row.image_height);" in open_body
    assert "width: Number.isFinite(imageWidth) && imageWidth > 0 ? imageWidth : 0" in open_body
    assert "height: Number.isFinite(imageHeight) && imageHeight > 0 ? imageHeight : 0" in open_body

    script = "\n".join(
        [
            "const assert = require('assert');",
            "let ensureDimensionCalls = 0;",
            "const rawRows = new Map([",
            "  ['train/frame001.jpg', { split: 'train', image_relpath: 'frame001.jpg', text_label: 'manifest caption' }],",
            "  ['val/nested/frame.002.png', { split: 'val', image_relpath: 'nested/frame.002.png', text_label: '' }],",
            "]);",
            "const annotationSourceState = { imageRowsByKey: rawRows };",
            "const textLabels = { 'train/frame001.jpg': 'edited caption' };",
            "const textLabelRecords = [];",
            "const bboxes = { 'train/frame001.jpg': {}, 'val/nested/frame.002.png': {} };",
            "const images = {};",
            "const classes = { car: 0 };",
            "const datasetType = 'bbox';",
            "function annotationImageKey(split, rel) { return `${split}/${rel}`; }",
            "function getAnnotationRecordLabelLines(imageKey) {",
            "  return imageKey === 'train/frame001.jpg'",
            "    ? ['4 0.5 0.5 0.2 0.3']",
            "    : ['7 0.1 0.2 0.03 0.04'];",
            "}",
            "function getCaptionDatasetId() { return ''; }",
            "async function ensureCaptionsForExport() { throw new Error('caption preload should not run'); }",
            "function isAnnotationDatasetModeActive() { return true; }",
            "async function ensureImageDimensions() { ensureDimensionCalls += 1; }",
            "class FakeZip {",
            "  constructor() { this.files = {}; }",
            "  folder(prefix) {",
            "    return { file: (name, value) => { this.files[`${prefix}/${name}`] = value; } };",
            "  }",
            "  file(name, value) { this.files[name] = value; }",
            "  async generateAsync() { return this.files; }",
            "}",
            "function ensureJsZipAvailable() { return FakeZip; }",
            _extract_js_function(js, "yoloExportTextFilename"),
            _extract_js_function(js, "yoloExportLabelmapText"),
            _extract_js_function(js, "writeDatasetAnnotationLabelsToYoloExport"),
            "async " + _extract_js_function(js, "buildYoloCaptionsExportZip"),
            "const result = await buildYoloCaptionsExportZip();",
            "assert.strictEqual(ensureDimensionCalls, 0);",
            "assert.strictEqual(result.skippedDimensionErrors, 0);",
            "assert.deepStrictEqual(result.exportedLabelIdentities, [",
            "  {image_key: 'train/frame001.jpg', label_name: 'train/frame001.txt'},",
            "  {image_key: 'val/nested/frame.002.png', label_name: 'val/nested/frame.002.txt'},",
            "]);",
            "assert.deepStrictEqual(result.skippedLabelImageKeys, []);",
            "assert.strictEqual(result.blob['train/frame001.txt'], '4 0.5 0.5 0.2 0.3');",
            "assert.strictEqual(result.blob['val/nested/frame.002.txt'], '7 0.1 0.2 0.03 0.04');",
            "assert.strictEqual(result.blob['text_labels/train/frame001.txt'], 'edited caption');",
            "assert.strictEqual(result.blob['text_labels/val/nested/frame.002.txt'], undefined);",
            "assert.strictEqual(result.blob['labelmap.txt'], 'car');",
        ]
    )
    subprocess.run(
        [
            "node",
            "-e",
            f"(async () => {{\n{script}\n}})().catch((error) => {{ console.error(error); process.exit(1); }});",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_dataset_yolo_export_uses_backend_download_instead_of_browser_zip():
    js = _js()
    export_runner = _extract_js_function(js, "runYoloCaptionsExport")
    backend_export = _extract_js_function(js, "runBackendAnnotationExport")

    assert "if (isAnnotationDatasetModeActive())" in export_runner
    assert export_runner.index("if (isAnnotationDatasetModeActive())") < export_runner.index(
        "const validation = validateGeometryForSave();"
    )
    assert "buildYoloCaptionsExportZip" not in export_runner[
        export_runner.index("if (isAnnotationDatasetModeActive())"):
        export_runner.index("const validation = validateGeometryForSave();")
    ]
    assert 'await ensureClassSplitSnapshotClean("YOLO export")' in backend_export
    assert 'fetch(`${exportUrl}/prepare`' in backend_export
    assert "expected_revision: expectedRevision" in backend_export
    assert 'link.href = `${exportUrl}?token=${encodeURIComponent(token)}`' in backend_export
    assert "link.click();" in backend_export
    assert "resp.blob" not in backend_export


def test_graph_inspector_has_the_same_review_actions_and_live_vlm_block_as_vignettes():
    js = _js()
    inspector = _extract_js_function(js, "renderClassSplitInspector")
    optimistic = _extract_js_function(js, "applyClassSplitOptimisticReview")

    for action in (
        'data-action="correct-class"',
        'data-action="skip-wrong"',
        'data-action="mark-dual-bbox-resolved"',
        'data-action="delete-bbox"',
        'data-action="qwen-review"',
        'data-action="change"',
    ):
        assert action in inspector
    assert "renderClassSplitQwenReviewBlock(point.point_id)" in inspector
    assert "bindClassSplitQwenTraceLoaders(inspector)" in inspector
    assert "markClassSplitWrongCandidateCorrect(point.point_id)" in inspector
    assert "skipClassSplitWrongCandidate(point.point_id)" in inspector
    assert "removeClassSplitPointFromActiveReviewGraph(safeId, { force: true })" in optimistic
    filtered_points = _extract_js_function(js, "getClassSplitFilteredPoints")
    assert "point?.annotation_deleted !== true" in filtered_points


def test_graph_review_removes_only_the_reviewed_point_from_the_live_trace():
    js = _js()
    capture_viewport = _extract_js_function(
        js, "captureClassSplitGraphViewport"
    )
    restore_viewport = "async " + _extract_js_function(
        js, "restoreClassSplitGraphViewport"
    )
    remove_points = _extract_js_function(
        js, "removeClassSplitPointsFromActiveReviewGraph"
    )
    remove_point = _extract_js_function(
        js, "removeClassSplitPointFromActiveReviewGraph"
    )
    script = "\n".join(
        [
            "const assert = require('assert').strict;",
            "const updates = [];",
            "const graph = { data: [{",
            "  x: [1, 2, 3], y: [4, 5, 6],",
            "  customdata: ['keep-a', 'remove-me', 'keep-b'],",
            "  text: ['a', 'remove', 'b'], selectedpoints: [1, 2],",
            "  marker: {size: [8, 10, 8], color: ['a', 'r', 'b'], line: {color: ['x', 'y', 'z'], width: [1, 2, 1]}},",
            "}]};",
            "const classSplitElements = {graph, displayMode: {value: 'wrong_only'}};",
            "const classSplitState = {selectedPointId: 'remove-me', lassoPointIds: new Set(['remove-me']), selectionRevision: 0, multiSelectionSignature: '', multiSelectionGraphCommitCount: 0};",
            "let inspectorRenders = 0; let bulkRenders = 0; let flashStops = 0;",
            "function hideClassSplitGraphHoverPreview() {}",
            "function stopClassSplitPlotFlash() { flashStops += 1; }",
            "function renderClassSplitInspector() { inspectorRenders += 1; }",
                "function renderClassSplitBulkPanel() { bulkRenders += 1; }",
                "function renderClassSplitPlot() {}",
                "function captureClassSplitGraphMutationContext() { return {}; }",
                "function classSplitGraphMutationContextIsCurrent() { return true; }",
                "const window = {Plotly: {restyle(_graph, update, traces) { updates.push({update, traces}); return Promise.resolve(); }}};",
                capture_viewport,
                restore_viewport,
                remove_points,
            remove_point,
            "assert.equal(removeClassSplitPointFromActiveReviewGraph('remove-me'), true);",
            "assert.equal(classSplitState.selectedPointId, '');",
            "assert.equal(inspectorRenders, 1); assert.equal(bulkRenders, 1); assert.equal(flashStops, 1);",
            "assert.deepEqual(updates[0].update.customdata[0], ['keep-a', 'keep-b']);",
            "assert.deepEqual(updates[0].update.selectedpoints[0], [1]);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_class_split_qwen_http_routes_remain_available():
    router = _read("api/class_analysis.py")

    assert '@router.post("/class_analysis/jobs/{job_id}/points/{point_id}/qwen_review")' in router
    assert '@router.get("/class_analysis/jobs/{job_id}/qwen_reviews")' in router
    assert '@router.get("/class_analysis/qwen_review/{review_id}")' in router
    assert '@router.post("/class_analysis/qwen_review/{review_id}/cancel")' in router
    assert '@router.get("/class_analysis/qwen_review/{review_id}/evidence/{evidence_id}")' in router


def test_class_split_qwen_guarded_review_is_prominent_and_actionable():
    js = _js()

    assert "Guarded suggestion: confirm current class" in js
    assert "Guarded suggestion: switch class to ${guardedTarget}" in js
    assert "Model confidence ${Number.isFinite(guardedConfidence)" in js
    assert "qwenGuarded?.blocked && qwenGuardedDecision !== \"confirm_current\"" in js
    assert "preferredTargetClass ? `Switch class to ${preferredTargetClass}` : \"Reassign\"" in js


def test_local_image_selection_shows_first_image_before_dimension_scan():
    js = _js()
    ingest_start = js.index("async function ingestImageFiles")
    ingest_end = js.index("function startImageDimensionScan", ingest_start)
    ingest_body = js[ingest_start:ingest_end]

    assert "document.createDocumentFragment()" in ingest_body
    assert "await readImageDimensions(file)" not in ingest_body
    assert "setCurrentImage(images[firstName]);" in ingest_body
    assert "startImageDimensionScan(stagedFiles, scanToken);" in ingest_body
    assert ingest_body.index("setCurrentImage(images[firstName]);") < ingest_body.index(
        "startImageDimensionScan(stagedFiles, scanToken);"
    )

    set_current_start = js.index("function setCurrentImage")
    set_current_end = js.index("const fitZoom", set_current_start)
    set_current_body = js[set_current_start:set_current_end]
    assert "decodeImageFromBlob(image.meta)" in set_current_body
    assert "reader.readAsDataURL(image.meta)" not in set_current_body

    load_object_start = js.index("function loadImageObject")
    load_object_end = js.index("function showProgressModal", load_object_start)
    load_object_body = js[load_object_start:load_object_end]
    assert "decodeImageFromBlob(imgData.meta)" in load_object_body


def test_annotation_diversity_metric_control_contract():
    html = _html()
    css = _css()
    js = _js()
    helper = _read("ybat-master/annotation_diversity.js")

    assert "annotation_diversity.js" in html
    assert html.index('src="annotation_diversity.js') < html.index('src="ybat.js')
    assert 'id="showAnnotationDiversityMetric"' in html
    assert "Show annotation class-balance score" in html
    assert "does not use pixels, embeddings, Data Quality Explorer analysis, Data Ingestion, or a reference profile" in html
    assert 'id="annotationDiversityMetric"' in html
    assert 'data-testid="status.annotation.diversity_metric"' in html
    assert ".annotation-diversity-metric" in css
    assert "ANNOTATION_DIVERSITY_METRIC_STORAGE_KEY" in js
    assert "initAnnotationDiversityControls();" in js
    assert "scheduleAnnotationDiversityMetricRefresh();" in js
    assert "It is not visual diversity and does not use a reference profile" in js
    assert "computeImageDiversityMetric" in helper
    assert "countBoxesByClassFromYoloLines" in helper
    assert "Class-balance score" in helper


def test_caption_output_label_precedes_large_textarea():
    html = _html()
    output_start = html.index('class="qwen-caption-output"')
    output_end = html.index('id="qwenCaptionMeta"')
    output_block = html[output_start:output_end]

    label_pos = output_block.index('for="qwenCaptionOutput"')
    textarea_pos = output_block.index('id="qwenCaptionOutput"')

    assert label_pos < textarea_pos
    assert 'rows="9"' in output_block


def test_caption_prompt_controls_have_tooltips_and_roomy_textareas():
    html = _html()
    css = _css()

    assert "Caption style preset<span class=\"help-icon\"" in html
    assert "Caption style text<span class=\"help-icon\"" in html
    assert "Opening phrase options (one per line)<span class=\"help-icon\"" in html
    assert "Final user request = Caption style text + optional opening phrase guidance." in html
    assert "Combined user request prompt<span class=\"help-icon\"" in html
    assert "Caption prompt stack<span class=\"help-icon\"" in html
    assert "qwen-caption-settings__section" in html
    assert "Caption scope" in html
    assert "Generation and guards" in html
    assert "Auto editor model (compact/Instruct)" in html
    assert "Same is literal: it uses the selected caption model." in html
    assert 'id="qwenCaptionPresetRandom" class="training-button secondary" title=' in html
    assert 'id="qwenCaptionStyleText" rows="5"' in html
    assert 'id="qwenCaptionOpeningList" rows="6"' in html
    assert 'id="qwenCaptionSystemPrompt" rows="12"' in html
    assert "Style prompts (one per line)" not in html
    assert "qwenCaptionStyleInspiration" not in html

    assert "#qwenCaptionStyleText,\n#qwenCaptionOpeningList" in css
    assert "min-height: 240px;" in css
    assert "max-height: 520px;" in css


def test_qwen_caption_recipes_are_portable_and_cover_prompt_stack():
    html = _html()
    js = _js()
    css = _css()

    for control_id in [
        "qwenCaptionPromptEditorSystem",
        "qwenCaptionPromptCoverage",
        "qwenCaptionPromptLanguageRewrite",
        "qwenCaptionRecipeSelect",
        "qwenCaptionRecipeName",
        "qwenCaptionRecipeSave",
        "qwenCaptionRecipeLoad",
        "qwenCaptionRecipeDelete",
        "qwenCaptionRecipeDownload",
        "qwenCaptionRecipeUploadButton",
        "qwenCaptionRecipeUpload",
        "qwenCaptionRecipeStatus",
        "qwenCaptionFallbackModel",
        "qwenCaptionLoopRecovery",
        "qwenCaptionLoopCooldown",
    ]:
        assert f'id="{control_id}"' in html
        assert control_id in js

    assert "Caption recipes" in html
    assert "Advanced guard/editor prompts" in html
    assert "Loop recovery" in html
    assert "Auto stable fallback" in html
    assert "never image pixels, per-image boxes, image tokens, or generated captions" in html
    assert "complete prompt-flow preview" in html
    assert "CAPTION_RECIPE_KIND" in js
    assert "tator.caption_recipe" in js
    assert "CAPTION_RECIPE_STORAGE_KEY" in js
    assert "collectCaptionRecipeFromUi" in js
    assert "applyCaptionRecipeToUi" in js
    assert "uploadCaptionRecipeFromFile" in js
    assert "downloadCaptionRecipe" in js
    assert "readFileAsTextPromise(file)" in js
    assert "saveBlobToDisk(blob, filename)" in js
    assert "caption_editor_system_prompt" in js
    assert "caption_coverage_prompt" in js
    assert "caption_language_rewrite_prompt" in js
    assert "caption_loop_recovery_mode" in js
    assert "caption_fallback_model_id" in js
    assert "caption_loop_cooldown" in js
    assert "recovery_events" in js

    collect_start = js.index("function collectCaptionRecipeFromUi")
    collect_end = js.index("function buildCaptionRecipeExportItem", collect_start)
    collect_block = js[collect_start:collect_end]
    for reusable_key in [
        "style",
        "prompt_stack",
        "detection_context",
        "draft_refine",
        "merge",
        "cleanup",
        "editor_system",
        "coverage",
        "language_rewrite",
        "scope",
        "models",
        "generation",
        "glossary_text",
    ]:
        assert reusable_key in collect_block
    for per_image_key in [
        "image_base64",
        "image_token",
        "label_hints",
        "used_boxes",
        "used_counts",
    ]:
        assert per_image_key not in collect_block

    assert ".qwen-caption-recipe" in css
    assert ".qwen-caption-recipe__actions button" in css


def test_help_tooltips_are_keyboard_accessible_app_wide():
    js = _js()
    css = _css()

    assert 'tooltipElements(root, ".help-icon").forEach' in js
    assert 'el.dataset.tooltip = tooltip;' in js
    assert 'el.removeAttribute("title");' in js
    assert "el.tabIndex = 0;" in js
    assert 'el.setAttribute("aria-label", `Help: ${tooltip}`);' in js
    assert "function ensureUiTooltipElement()" in js
    assert "function positionUiTooltip()" in js
    assert "function initializeViewportHelpTooltips()" in js
    assert 'tooltip.id = "uiViewportTooltip";' in js
    assert 'document.addEventListener("pointerover"' in js
    assert 'document.addEventListener("focusin"' in js
    assert 'document.addEventListener("scroll", () => positionUiTooltip(), true);' in js
    assert "anchor.setAttribute(\"aria-describedby\", tooltip.id);" in js
    assert ".help-icon[data-tooltip]:focus-visible" in css
    assert ".ui-tooltip" in css
    assert "position: fixed;" in css
    assert "max-width: min(360px, calc(100vw - 24px));" in css
    assert ".ui-tooltip::before" in css
    assert "var(--ui-tooltip-arrow-x, 50%)" in css
    assert ".help-icon[data-tooltip]::after" not in css
    assert "content: attr(data-tooltip);" not in css
    assert '[data-action="selector-help"][data-tooltip]' in js


def test_runtime_control_tooltips_cover_core_workflows():
    js = _js()
    override_ids = _control_override_id_list(js)

    assert "const CONTROL_TOOLTIP_OVERRIDES = Object.freeze({" in js
    assert len(override_ids) == len(set(override_ids))
    assert "const CONTROL_FIELD_LABEL_SELECTOR = [" in js
    assert '".data-ingestion-field"' in js
    assert '".class-split-field"' in js
    assert '".sam3-text-field"' in js
    assert "function initControlTooltips(root = document)" in js
    assert "function tooltipElements(root, selector)" in js
    assert "root.nodeType === 1" in js
    assert "root.querySelectorAll(selector).forEach" in js
    assert 'tooltipElements(root, "button, input, select, textarea").forEach' in js
    assert "function deriveControlTooltip(el)" in js
    assert "function cssEscapeIdentifier(value)" in js
    assert 'return raw.replace(/[^A-Za-z0-9_-]/g, "\\\\$&");' in js
    assert "function normalizeTooltipLabelText(text)" in js
    assert "function labelTextFromElement(label)" in js
    assert "function associatedControlLabelText(el)" in js
    assert 'document.querySelector(`label[for="${cssEscapeIdentifier(id)}"]`)' in js
    assert "el.closest(CONTROL_FIELD_LABEL_SELECTOR)" in js
    assert 'child?.tagName?.toLowerCase() === "label"' in js
    assert 'details?.querySelector("summary")?.textContent' in js
    assert "function initializeUiTooltipObserver()" in js
    assert "new MutationObserver" in js
    assert "function scheduleUiTooltipRefresh(root = document)" in js
    assert "scheduleUiTooltipRefresh(node);" in js
    assert "const uiTooltipRefreshRoots = new Set();" in js
    assert "uiTooltipRefreshRoots.add(root);" in js
    assert "const roots = uiTooltipRefreshRoots.size ? Array.from(uiTooltipRefreshRoots) : [document];" in js
    assert "uiTooltipRefreshRoots.clear();" in js
    assert 'const existingTitle = String(el.getAttribute("title") || "").trim();' in js
    assert "const tooltip = existingTitle || String(deriveControlTooltip(el) || \"\").trim();" in js
    assert "if (!existingTitle) {" in js
    assert "initializeUiTooltipObserver();" in js
    assert "initControlTooltips(root);" in js
    assert 'if (lower === "refresh") return "Refresh this list or status panel.";' in js
    assert "Open this Qwen training job and refresh its status when it is still active." in js
    assert "Show this Qwen training job's status, logs, and result metadata." in js
    assert "Show this YOLO training job's status, logs, and result metadata." in js
    assert "Show this RF-DETR training job's status, logs, and result metadata." in js
    assert "Show this head-graft job's status, logs, and result metadata." in js
    assert 'title="Remove this SAM3 text cascade step."' in js
    for control_id in [
        "saveBboxes",
        "detectorRunButton",
        "qwenRunButton",
        "qwenCaptionPromptUser",
        "qwenCaptionPromptCleanup",
        "sam3RunButton",
        "dataIngestionAnalyzeButton",
        "dataIngestionDownloadAcceptedButton",
        "classSplitRunButton",
        "classSplitBulkClass",
        "classSplitWrongShuffle",
        "qwenAgentRecipeImportFile",
        "datasetUploadCurrentBtn",
        "datasetPathRegisterBtn",
        "datasetGlossarySave",
        "trainDatasetRefresh",
        "startTrainingBtn",
        "qwenTrainStartBtn",
        "sam3StartBtn",
        "sam3TrendSmooth",
        "yoloTrainStartBtn",
        "rfdetrTrainStartBtn",
        "detectorYoloRunActivate",
        "activeClassifierUse",
        "activeClassifierUpload",
        "activeLabelmapUpload",
        "qwenModelRefreshBtn",
        "sam3PromptActivate",
        "settingsTest",
        "runBackendFuzzer",
    ]:
        assert f"{control_id}:" in js


def test_static_visible_controls_have_tooltips_or_discoverable_labels():
    root = _parse_static_html()
    js = _js()
    labels_by_for = {
        str(label.attrs.get("for")).strip(): label.text_content()
        for label in _nodes_by_tag(root, "label")
        if str(label.attrs.get("for") or "").strip()
    }
    override_ids = _control_override_ids(js)
    controls = [
        node
        for tag in ("button", "input", "select", "textarea")
        for node in _nodes_by_tag(root, tag)
        if str(node.attrs.get("type") or "").lower() != "hidden"
    ]

    missing = [
        _describe_control(control)
        for control in controls
        if not _control_has_accessible_static_or_runtime_tooltip(control, labels_by_for, override_ids)
    ]

    assert not missing, "static controls without title, text, label, or runtime tooltip override:\n" + "\n".join(
        missing[:40]
    )


def test_caption_dataset_picker_is_locked_to_annotation_dataset():
    js = _js()

    assert "function syncQwenCaptionDatasetControls()" in js
    assert "qwenCaptionDatasetRefreshInFlight || isAnnotationDatasetModeActive() || busy" in js
    assert "qwenElements.captionDatasetRefresh.disabled = qwenCaptionDatasetRefreshInFlight || busy" in js
    assert "guardQwenCaptionArchiveIdle(\"refreshing caption datasets\")" in js
    assert "guardQwenCaptionArchiveIdle(\"changing caption datasets\")" in js
    assert "const allowDuringActive = options.allowDuringActive === true;" in js
    assert "if (!allowDuringActive && qwenCaptionArchiveMutationActive())" in js
    assert "await refreshQwenCaptionDatasets({\n                silent: true,\n                allowDuringActive: true," in js
    assert "qwenElements.captionDatasetSelect.value = stableDatasetId;" in js
    assert "if (isAnnotationDatasetModeActive()) {\n            return getActiveAnnotationDatasetIdForCaption();\n        }" in js
    assert "if (isAnnotationDatasetModeActive()) {\n                    syncCaptionDatasetSelectionWithAnnotationDataset();" in js
    assert "if (isAnnotationDatasetModeActive()) {\n                syncCaptionDatasetSelectionWithAnnotationDataset();\n            } else {" in js
    assert "qwenElements.captionDatasetSelect.disabled = false;" not in js
    assert "updateQwenCaptionDatasetRefreshButton" not in js


def test_caption_dataset_controls_block_refresh_while_archive_is_mutating():
    js = _js()
    refresh_start = js.index("async function refreshQwenCaptionDatasets")
    refresh_end = js.index("function getQwenAgentDatasetId", refresh_start)
    script = "\n".join(
        [
            "const assert = require('assert');",
            "let qwenCaptionActive = false;",
            "let qwenCaptionBatchActive = false;",
            "let qwenCaptionBatchBackendJobId = 'job-1';",
            "let qwenCaptionDatasetRefreshInFlight = false;",
            "let qwenCaptionDatasetRefreshNeedsRefresh = false;",
            "let fetchCalls = 0;",
            "let updateCalls = 0;",
            "const captionStatuses = [];",
            "const backendStatuses = [];",
            "const samStatuses = [];",
            "const qwenElements = {",
            "  captionDatasetSelect: { disabled: false, value: 'ds-current' },",
            "  captionDatasetRefresh: { disabled: false },",
            "};",
            "function isAnnotationDatasetModeActive() { return false; }",
            "function setQwenCaptionStatus(message) { captionStatuses.push(message); }",
            "function setQwenCaptionBackendJobStatus(message) { backendStatuses.push(message); }",
            "function setSamStatus(message, options) { samStatuses.push({ message, options }); }",
            "function updateQwenCaptionButton() { updateCalls += 1; syncQwenCaptionDatasetControls(); }",
            "async function fetch() { fetchCalls += 1; throw new Error('refresh should be blocked before fetch'); }",
            _extract_js_function(js, "qwenCaptionArchiveMutationActive"),
            _extract_js_function(js, "captionArchiveMutationBusyMessage"),
            _extract_js_function(js, "guardQwenCaptionArchiveIdle"),
            _extract_js_function(js, "syncQwenCaptionDatasetControls"),
            js[refresh_start:refresh_end],
            "syncQwenCaptionDatasetControls();",
            "assert.strictEqual(qwenElements.captionDatasetSelect.disabled, true);",
            "assert.strictEqual(qwenElements.captionDatasetRefresh.disabled, true);",
            "await refreshQwenCaptionDatasets();",
            "assert.strictEqual(fetchCalls, 0);",
            "assert(captionStatuses.includes('Caption archive busy'));",
            "assert(backendStatuses.some((message) => message.includes('refreshing caption datasets')));",
            "assert(samStatuses.some((entry) => entry.message.includes('caption archive is changing')));",
            "qwenCaptionBatchBackendJobId = '';",
            "syncQwenCaptionDatasetControls();",
            "assert.strictEqual(qwenElements.captionDatasetSelect.disabled, false);",
            "assert.strictEqual(qwenElements.captionDatasetRefresh.disabled, false);",
            "assert(updateCalls >= 1);",
        ]
    )
    subprocess.run(
        [
            "node",
            "-e",
            f"(async () => {{\n{script}\n}})().catch((error) => {{ console.error(error); process.exit(1); }});",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_annotation_snapshot_save_preserves_edits_made_during_inflight_save():
    js = _js()

    assert "saveQueued: false" in js
    assert "const sentSnapshotByKey = new Map();" in js
    assert "sentSnapshotByKey.set(key, serializeAnnotationRecord(record));" in js
    assert "const currentRecord = buildAnnotationRecord(key);" in js
    assert "const currentSnapshot = serializeAnnotationRecord(currentRecord || record);" in js
    assert "if (currentSnapshot === sentSnapshot) {" in js
    assert "annotationSourceState.dirtyRecordsByKey.set(key, currentRecord);" in js
    assert "annotationSourceState.saveQueued = true;" in js
    assert "Queued annotation snapshot flush failed" in js


def test_loaded_edr_recipe_prepass_caption_flag_is_honored():
    js = _js()

    assert "const prepassCaptionEnabled = usePackageRuntime\n            ? false\n            : (useRecipeConfig ? getConfigEnabled(\"prepass_caption\", true) : true);" in js
    assert "prepass_caption: prepassCaptionEnabled," in js
    assert "prepass_caption: usePackageRuntime ? false : true," not in js
    assert "const inheritedPrepassCaption = activeRecipeConfig\n            && Object.prototype.hasOwnProperty.call(activeRecipeConfig, \"prepass_caption\")\n            ? activeRecipeConfig.prepass_caption !== false\n            : true;" in js
    assert "const prepassCaptionEnabled = edrPackageId ? false : inheritedPrepassCaption;" in js
    assert "prepass_caption: edrPackageId ? false : true," not in js


def test_keyboard_image_navigation_shortcuts_are_documented_and_guarded():
    html = _html()
    js = _js()

    assert 'id="shortcutHelpList"' in html
    assert 'id="shortcutSettingsPanel"' in html
    assert "Customize keyboard shortcuts" in html
    assert 'id="shortcutResetAll"' in html
    assert 'id="shortcutExportConfig"' in html
    assert 'id="shortcutImportConfigButton"' in html
    assert 'id="shortcutImportConfig"' in html
    assert 'accept=".json,application/json"' in html
    assert 'id="shortcutSettingsList"' in html
    assert 'class="shortcut-help-list"' in html
    assert "SHORTCUT_STORAGE_KEY" in js
    assert '"tator.annotation.shortcuts.v1"' in js
    assert "SHORTCUT_CLASS_ID_COUNT" not in js
    assert "CLASS_SHORTCUT_ID_PATTERN" in js
    assert "function getShortcutClassNames()" in js
    assert "function makeClassShortcutAction(index, className)" in js
    assert 'id: `class_id_${index}`' in js
    assert "load a labelmap or dataset first to configure direct class shortcuts" in js
    assert "direct class IDs 0-19" not in js
    assert "shortcutClassListObserver.observe(classList, { childList: true });" in js
    assert 'id: "image_next"' in js
    assert 'id: "image_previous"' in js
    assert 'id: "drawing_start"' in js
    assert 'id: "drawing_finish"' in js
    assert 'id: "delete_selected_current"' in js
    assert "renderShortcutHelp()" in js
    assert "renderShortcutSettings()" in js
    assert "assignShortcutBinding(actionId, binding)" in js
    assert "removeShortcutBindingConflicts(actionId, bindings)" in js
    assert "eventToBinding(event)" in js
    assert "readFileAsTextPromise(file)" in js
    assert 'saveBlobToDisk(blob, "tator-shortcuts.json")' in js
    assert "run: () => navigateImage(1)" in js
    assert "run: () => navigateImage(-1)" in js
    assert "action.run(event)" in js
    assert 'window.addEventListener("keydown"' in js
    assert '}, true);' in js
    assert "annotationWorkspaceHotkeysActive" in js
    assert "__tatorImageNavigationHandled" in js
    assert "canvas.element.focus({ preventScroll: true })" in js
    assert "const requestedMaxItems = Number.isFinite(options.maxVisibleItems)" in js
    assert "function scheduleClassSplitControlsRefresh" in js
    assert "function scheduleSam3TextWorkflowRefresh" in js
    assert "syncSam3ClassToCurrent();" in js
    assert "scheduleClassSplitControlsRefresh();" not in js
    assert "if (!isClassSplitTabActive()) {\n            return;\n        }" in js
    assert "scheduleClassSplitControlsRefresh({ delay: 0, preferCurrentClass: true });" in js
    assert "const nextIndex = (currentIndex + delta + total) % total;\n            Array.from(classList.options).forEach" not in js


def test_local_salad_is_data_ingestion_only_in_ui():
    html = _html()
    js = _js()

    assert 'OT-SALAD [experimental]' not in html
    assert 'id="dataIngestionSaladHead"' in html
    assert '<option value="local_salad"' not in html
    assert '<option value="sam3_mask_salad_fusion_v1"' in html
    for control_id in (
        "classSplitSaladPreset",
        "classSplitSaladWeight",
        "classSplitSaladTrainMax",
        "classSplitSaladEpochs",
        "classSplitSaladTokenBudget",
    ):
        assert f'id="{control_id}"' in html
    assert "Big Salad (8,448-D, experimental)" in html
    assert "Maximum label-free training objects, not an analysis cap" in html
    assert 'id="classSplitSaladHead"' not in html
    assert 'id="trainEmbeddingAggregation"' not in html
    assert 'id="trainSaladHead"' not in html
    assert '<option value="local_salad">Local SALAD separation</option>' not in html
    assert '<option value="local_salad">Local SALAD head</option>' not in html
    assert "Local SALAD requires a trained local head" not in html
    assert "Reference profiles are built from your own dataset images only" in html
    assert "classSplitElements.saladHead" not in js
    assert "function updateClassSplitSaladControls()" in js
    for request_field in (
        "salad_preset",
        "salad_weight",
        "salad_max_train_objects",
        "salad_epochs",
        "salad_token_budget_mb",
    ):
        assert request_field in js
    assert "trainingElements.embeddingAggregationSelect" not in js
    assert "trainingElements.saladHeadSelect" not in js
    assert 'formData.append("embedding_aggregation"' not in js
    assert 'formData.append("embedding_salad_head_id"' not in js
    assert "SALAD head: ${escapeHtml(art.embedding_salad_head_id)}" not in js
    assert "Aggregation: ${escapeHtml(art.embedding_aggregation" not in js
    assert "startLocalSaladTraining" in js
    assert "canvas.element.focus();" in js
    assert "function isTextEditingTarget(target)" in js
    assert 'targetElement.closest(".shortcut-settings-panel")' in js
    assert 'targetTag === "textarea"' in js


def test_local_salad_benchmark_does_not_recommend_crop_workflows():
    diversity_benchmark = _read("tools/benchmark_salad_diversity.py")
    class_benchmark = _read("tools/benchmark_salad_class_separation.py")
    clip_training_cli = _read("tools/train_clip_regression_from_YOLO.py")

    assert '"embedding_aggregation": "local_salad"' not in diversity_benchmark
    assert '"embedding_aggregation": "local_salad"' not in class_benchmark
    assert 'choices=["pooled", "local_salad"]' not in clip_training_cli
    assert "--train-local-salad" not in class_benchmark
    assert "crop-token aggregator" not in diversity_benchmark
    assert "Data Quality Explorer or auto-class training" not in diversity_benchmark
    assert "Data Ingestion diversity scoring" in diversity_benchmark
    assert "crop_level_local_salad" in class_benchmark


def test_qwen_training_fallback_catalog_covers_mlx_and_abliterated_paths():
    js = _js()
    start = js.index("const QWEN_TRAINING_MODEL_FALLBACKS = [")
    end = js.index("function inferQwenModelSize", start)
    fallback_block = js[start:end]
    fallback_ids = re.findall(r'qwenTrainingFallback\("([^"]+)"', fallback_block)

    assert fallback_ids
    assert len(fallback_ids) == len(set(fallback_ids))
    assert "qwenTrainingFallback(id, label, metadata = {})" in js
    assert "mlx-community/Qwen3-VL-4B-Instruct-4bit" in fallback_ids
    assert "mlx-community/Qwen3-VL-4B-Thinking-4bit" in fallback_ids
    assert "EZCon/Huihui-Qwen3-VL-4B-Instruct-abliterated-4bit-mlx" in fallback_ids
    assert "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated" in fallback_ids
    assert "nicklas373/Huihui-Qwen3-VL-8B-Thinking-abliterated-AWQ" in fallback_ids
    assert 'runtime_platform: "mlx_vlm"' in fallback_block
    assert "abliterated: true" in fallback_block
    assert 'training_model_id: "huihui-ai/Huihui-Qwen3-VL-8B-Thinking-abliterated"' in fallback_block
    assert '"27B": 24.0' in js
    assert '["235B", "35B", "30B", "32B", "27B", "8B", "4B", "2B"]' in js


def test_qwen_injected_runtime_options_use_shared_mlx_resolver():
    js = _js()
    start = js.index("function ensureQwenSelectOption")
    end = js.index("function populateQwenRuntimeModelSelects", start)
    helper = js[start:end]

    assert "option.dataset.runtimePlatform = inferQwenRuntimePlatform(optionValue);" in helper
    assert 'optionValue.startsWith("mlx-community/")' not in helper
    assert 'lowered.includes("-mlx-")' in js
    assert 'lowered.endsWith("-mlx")' in js
    assert "goekdeniz-guelmez/josiefied-qwen3-vl-" in js


def test_qwen_runtime_selects_keep_full_model_refresh_authoritative():
    js = _js()
    start = js.index("function populateQwenMlxModelSelect")
    end = js.index("async function refreshQwenSettings", start)
    settings_block = js[start:end]

    assert "populateQwenRuntimeModelSelects(items);" not in settings_block
    assert "populateQwenRuntimeModelSelects(qwenModelState.models);" in js
    assert "applyQwenModelAvailabilityStyle(option, entry);" in settings_block


def test_qwen_model_select_options_color_downloads_red_and_local_white():
    js = _js()
    css = _css()
    start = js.index("function applyQwenModelAvailabilityStyle")
    end = js.index("function styleQwenModelSelectOptions", start)
    block = js[start:end]
    select_start = js.index("function applyQwenModelSelectAvailabilityState")
    select_end = js.index("function styleQwenCaptionModelSelects", select_start)
    select_block = js[select_start:select_end]

    assert "availability.needs_download" in block
    assert 'option.style.color = "#ff4d4f";' in block
    assert 'option.style.color = "#ffffff";' in block
    assert "qwen-model-option--download" in block
    assert "qwen-model-option--local" in block
    assert "applyQwenModelSelectAvailabilityState(select);" in js
    assert "qwen-model-select--download" in select_block
    assert "qwen-model-select--local" in select_block
    assert ".qwen-model-select--download" in css
    assert "color: #ff4d4f;" in css
    assert ".qwen-model-select--local" in css
    assert "color: #ffffff;" in css


def test_qwen_caption_and_agent_selects_share_workable_vlm_catalog():
    js = _js()
    start = js.index("function appendQwenMlxOptionsToSelect")
    end = js.index("function populateQwenMlxModelSelect", start)
    block = js[start:end]

    assert "caption_supported" not in block
    assert "qwenElements.captionModel" in block
    assert "qwenElements.captionRefinementModel" in block
    assert "qwenElements.agentModel" in block
    assert "qwenElements.agentCaptionModel" in block
    assert "appendQwenMlxOptionsToSelect(select, items);" in block


def test_class_split_qwen_review_selector_keeps_broad_workable_vlm_catalog():
    js = _js()
    start = js.index("function renderClassSplitQwenReviewModelOptions")
    end = js.index("async function refreshClassSplitQwenReviewModels", start)
    block = js[start:end]

    assert "metadata.inference_supported === false" in block
    assert "metadata.vision_inference_supported === false" in block
    assert "caption_supported" not in block
    assert "classSplitState.qwenReviewDefaultModelId" in block
    assert "reviewDefaultCompatible" in block
    refresh_end = js.index("function clearClassSplitQwenReviewPolls", end)
    refresh_block = js[end:refresh_end]
    assert "data.review_default" in refresh_block


def test_qwen_caption_cancel_does_not_force_backend_restart():
    js = _js()

    assert "requestQwenCaptionCancel({ force: true })" not in js
    assert "requestQwenCaptionCancel({ force: false })" in js
    assert 'const url = `${API_ROOT}/qwen/caption/cancel?force=${force ? "1" : "0"}`;' in js
    assert "hideQwenCaptionLiveToast(0" in js
    assert 'phase: "cancelled"' in js


def test_qwen_caption_toast_shows_prompt_output_trace_blocks():
    js = _js()
    css = _css()

    assert "function renderQwenCaptionLiveToastBody" in js
    assert "function maybeToastQwenCaptionGeneratedQaPairs" in js
    assert "qwenCaptionSeenGeneratedQaToasts.add(key)" in js
    assert "maybeToastQwenCaptionGeneratedQaPairs(liveSnapshot)" in js
    assert "progress?.io_events" in js
    start = js.index("function renderQwenCaptionLiveToastBody")
    end = js.index("function hideQwenCaptionLiveToast", start)
    assert "progress?.io_events.slice" not in js[start:end]
    assert "qwen-caption-live-toast__trace--${kind}" in js
    assert ".qwen-caption-live-toast__trace--prompt" in css
    assert ".qwen-caption-live-toast__trace--output" in css
    assert "let qwenCaptionLiveToastHovered = false;" in js
    assert 'el.addEventListener("mouseenter", () => {' in js
    assert "if (qwenCaptionLiveToastHovered)" in js
    assert "width: min(710px, calc(100vw - 36px));" in css
    assert "max-height: min(88vh, 960px);" in css
    assert "max-height: min(76vh, 820px);" in css
    assert ".qwen-caption-live-toast__trace-text" in css
    trace_text_start = css.index(".qwen-caption-live-toast__trace-text")
    trace_text_end = css.index(".qwen-caption-live-toast__trace--prompt", trace_text_start)
    assert "max-height:" not in css[trace_text_start:trace_text_end]
    assert ".left textarea," in css
    assert ".qwen-caption-output textarea" in css
    assert "box-sizing: border-box;" in css


def test_qwen_caption_workflow_can_preview_complete_prompt_flow():
    html = _html()
    js = _js()
    css = _css()

    assert "preview complete prompt flow on image" in js
    assert "handleQwenCaptionPromptPreview" in js
    assert "/qwen/caption/preview_prompt" in js
    assert "invokeQwenCaptionPromptPreview" in js
    assert "buildQwenCaptionRequestFields(requestImageName)" in js
    assert "Preview dataset prompts" in html
    assert "qwenCaptionPreviewInstructionProcess" in html
    assert html.index("qwenCaptionPreviewInstructionProcess") < html.index("qwenCaptionBuildInstructionDataset")
    assert "handleQwenCaptionInstructionProcessPreview" in js
    assert "/qwen/caption/jobs/preview_process" in js
    assert "buildQwenCaptionDatasetJobRequestPayload" in js
    assert "generated-QA prompt template" in js
    assert "Pilot certification is a launch-only gate" in js
    assert "will still certify the selected pilot artifact directory" in js
    assert "max prompt ~" in js
    assert "output ${effectiveMin === effectiveMax" in js
    assert ".qwen-caption-prompt-preview-toast" in css
    assert ".qwen-caption-prompt-preview-toast__body" in css
    assert "max-height: calc(86vh - 118px);" in css


def test_qwen_caption_max_boxes_explains_auto_representative_subset():
    html = _html()
    js = _js()

    assert "Auto keeps full counts but sends representative spatial boxes when scenes are dense" in html
    assert "omitted boxes are not absent objects" in html
    assert "Auto estimates prompt size and adapts output tokens at runtime" in html
    assert "Max boxes is set to Auto" in js
    assert "representative spatial subset of boxes" in js
    assert "omitted boxes are not absent objects" in js


def test_qwen_caption_remote_provider_controls_and_cost_estimator_are_wired():
    html = _html()
    js = _js()
    css = _css()

    assert 'id="qwenCaptionProvider"' in html
    assert "OpenAI API" in html
    assert 'id="qwenCaptionOpenAiModel"' in html
    assert 'value="gpt-5.5"' in html
    assert 'id="qwenCaptionOpenAiImageDetail"' in html
    assert 'value="original"' in html
    assert 'id="qwenCaptionOpenAiServiceTier"' in html
    assert "Batch durable async" in html
    assert '<option value="standard">Standard</option>' in html
    assert '<option value="batch" selected>Batch durable async</option>' in html
    assert 'id="qwenCaptionOpenAiBatchShardSize"' in html
    assert 'value="100"' in html
    assert 'id="qwenCaptionOpenAiShardPlan"' in html
    assert 'id="qwenCaptionOpenAiKeyPath"' in html
    assert 'value="openAI_API_KEY_DoNotCommit"' in html
    assert 'id="qwenCaptionOpenAiCostEstimate"' in html
    assert 'id="qwenCaptionRefreshOpenAiBatches"' in html
    assert 'id="qwenCaptionScanOpenAiBatches"' in html
    assert 'id="qwenCaptionOpenAiBatchFilter"' in html
    assert 'id="qwenCaptionRefreshOpenAiSpend"' in html
    assert 'id="qwenCaptionOpenAiAdminKeyPath"' in html
    assert 'value="openAI_ADMIN_KEY_DoNotCommit"' in html
    assert 'id="qwenCaptionOpenAiSpendDays"' in html
    assert 'id="qwenCaptionOpenAiSpendStatus"' in html
    assert 'id="qwenCaptionOpenAiSpendSummary"' in html
    assert 'id="qwenCaptionAdoptOpenAiBatchPath"' in html
    assert 'id="qwenCaptionAdoptOpenAiBatch"' in html
    assert 'id="qwenCaptionOpenAiBatchIdentity"' in html
    assert 'id="qwenCaptionOpenAiBatchStatus"' in html
    assert 'id="qwenCaptionOpenAiBatchRecovery"' in html
    assert 'id="qwenCaptionOpenAiBatchJobs"' in html
    assert 'id="qwenCaptionOpenAiBatchDetails"' in html
    assert "Run kind to Test run" in html
    assert "Max images to the desired count" in html
    assert "const DEFAULT_OPENAI_CAPTION_MODEL = \"gpt-5.5\"" in js
    assert "const DEFAULT_OPENAI_CAPTION_DETAIL = \"original\"" in js
    assert "const DEFAULT_OPENAI_ADMIN_KEY_FILE = \"openAI_ADMIN_KEY_DoNotCommit\"" in js
    assert "OPENAI_CAPTION_PRICING_PER_MILLION" in js
    assert "const DEFAULT_OPENAI_CAPTION_REASONING_EFFORT = \"high\"" in js
    assert "const DEFAULT_OPENAI_CAPTION_SERVICE_TIER = \"batch\"" in js
    assert "function getCaptionProviderSettings" in js
    assert "function updateOpenAiCaptionCostEstimate" in js
    coverage_target_helper = _extract_js_function(js, "effectiveCaptionCoverageBaseTarget")
    assert "writePolicy.startsWith(\"qa_only\")" in coverage_target_helper
    assert "resolved.include_caption0_in_training === false" in coverage_target_helper
    assert "completionMode === \"incremental\" && incrementBase <= 0" in coverage_target_helper
    assert "return 0;" in coverage_target_helper
    coverage_qa_helper = _extract_js_function(js, "effectiveCaptionCoverageGeneratedQaTarget")
    assert "resolved.include_generated_qa_in_training === false" in coverage_qa_helper
    assert "return 0;" in coverage_qa_helper
    assert "params.set(\"target_base_captions_per_image\", String(effectiveBaseTarget));" in js
    assert "params.set(\"target_generated_qa_per_image\", String(effectiveGeneratedTarget));" in js
    assert "function openAiCaptionPricingTable" in js
    assert "function openAiCaptionGranularCostEstimate" in js
    assert "const billableCaptionUnits = Math.max(captions, qa > 0 ? 1 : 0);" in js
    assert "billable_caption_units_per_image: billableCaptionUnits" in js
    assert "const includeGeneratedQa = qwenElements.captionIncludeGeneratedQaTraining?.checked !== false;" in js
    assert "const rawRequestedQa = completionMode === \"incremental\"" in js
    assert "const requestedQa = includeGeneratedQa ? rawRequestedQa : 0;" in js
    assert "const qaOnlyWritePolicy = writePolicy.startsWith(\"qa_only\");" in js
    assert "caption/grounding unit" in js
    assert "one caption/grounding unit is billed even when base captions are not saved" in js
    assert "const baseCaptionCalls = writePolicy.startsWith(\"qa_only\") ? 0 : imageCount;" not in js
    assert "/qwen/caption/openai_metadata" in js
    assert "function refreshOpenAiCaptionSpend" in js
    assert "function renderOpenAiCaptionSpendSummary" in js
    assert "/qwen/caption/openai_spend" in js
    assert "function runOpenAiCaptionBatchDatasetJob" in js
    assert "/qwen/caption/openai_batches" in js
    assert "/qwen/caption/openai_batches/scan" in js
    assert "function scanOpenAiCaptionBatchArtifacts" in js
    assert "function showOpenAiCaptionBatchDetails" in js
    assert "function renderOpenAiCaptionBatchRecovery" in js
    assert "same_images_different_dataset_id" in js
    assert "exact_dataset_match_label_warning" in js
    assert "local_materialization" in js
    assert "local artifacts ready" in js
    assert "Instruction artifacts materialized locally" in js
    assert "No OpenAI Batch or local artifact jobs are registered." in js
    assert "safe: image match, label warning" in js
    assert "openAiCaptionBatchImportAllowedForMatch" in js
    assert 'return ["exact_dataset_match", "exact_dataset_match_label_warning", "same_images_different_dataset_id"].includes(value);' in js
    assert "Select a caption dataset before importing remote Batch outputs." in js
    assert "Remote Batch import needs target verification first." in js
    assert "Import is blocked because the Batch target does not match the active caption dataset." in js
    assert "Scan limit reached; enter a narrower backend-local artifact folder to search more deeply." in js
    assert "Scan limit reached; narrow the search path for a deeper scan." in js
    assert "payload.artifact_dirs = [artifactDir]" in js
    assert "collection_error_shards" in js
    assert "providerSettings.provider === CAPTION_PROVIDER_OPENAI && providerSettings.serviceTier === \"batch\"" in js
    assert "function refreshOpenAiCaptionBatchJobs" in js
    assert "function handleOpenAiCaptionBatchAction" in js
    assert "function adoptOpenAiCaptionBatchArtifact" in js
    assert "Retry failed rows" in js
    assert "Accepted rows are not rerun." in js
    assert "hasCollectableRemoteFiles" in js
    assert 'addAction("restore", "Restore")' in js
    assert 'cleanAction === "archive" || cleanAction === "restore"' in js
    assert "label_snapshot_mismatch" in js
    assert "caption_provider: providerSettings.provider" in js
    assert "openai_model: providerSettings.model" in js
    assert "openai_image_detail: providerSettings.detail" in js
    assert "openai_reasoning_effort: providerSettings.reasoningEffort" in js
    assert "openai_api_key_path: providerSettings.keyPath" in js
    assert "openai_service_tier: providerSettings.serviceTier" in js
    assert "openai_batch_shard_size: providerSettings.batchShardSize" in js
    assert "captionOpenAiBatchShardSize" in js
    assert "captionOpenAiShardPlan" in js
    assert "captionOpenAiAdminKeyPath" in js
    assert "captionOpenAiSpendSummary" in js
    assert "Caption provider control" in js
    assert "OpenAI reasoning effort control" in js
    assert "OpenAI cost estimate display" in js
    assert "Pricing verified ${qwenCaptionOpenAiMetadata.pricing_last_verified}" in js
    assert "provider.model}, ${provider.detail}, ${provider.reasoningEffort}, ${provider.serviceTier}" in js
    assert "event.ts || event.timestamp || event.time" in js
    assert "Remote provider calls use the same prompt stack and output-token budget" in js
    assert "OpenAI captioning uses the persisted backend job path" in js
    assert "OpenAI provider needs a model and backend-local key path or OPENAI_API_KEY" in js
    assert ".qwen-caption-remote-cost" in css
    assert ".qwen-caption-remote-cost.is-warn" in css
    assert ".qwen-caption-remote-batch-panel" in css
    assert ".qwen-caption-remote-batch-item" in css
    assert ".qwen-caption-remote-batch-badge" in css
    assert ".qwen-caption-remote-batch-details" in css
    assert ".qwen-caption-remote-batch-artifact" in css
    assert ".qwen-caption-remote-spend-panel" in css


def test_qwen_caption_windowed_full_image_compose_is_set_and_forget_aware():
    html = _html()
    js = _js()

    assert "qwenCaptionWindowFullImageStrategy" in html
    assert "Windowed full-image compose" in html
    assert "Auto: set-and-forget text-only" in html
    assert "Text-only from windows" in html
    assert "Visual full-image pass" in html
    assert "function resolveCaptionWindowedFullImageStrategy" in js
    assert 'return backendSetAndForget ? "text_only" : "visual";' in js
    assert 'caption_windowed_full_image_strategy: captionMode === "windowed" ? windowedFullImageStrategy : "visual"' in js
    assert "Text-only full-image composition" in js
    assert "windowed_full_image_strategy" in js


def test_qwen_caption_set_and_forget_controls_are_exposed():
    html = _html()
    js = _js()

    assert "qwenCaptionSetAndForget" in html
    assert "qwenCaptionPilotDeterministicRecoveryConfidence" in html
    assert "qwenCaptionAllowModelDownload" in html
    assert "qwen-model-option--download" in js
    assert "qwenCaptionBackendJobAutoResumeId" in js
    assert "qwenCaptionRefreshBackendJobs" in html
    assert "qwenCaptionCancelBackendJobs" in html
    assert "qwenCaptionBackendJobsSummary" in html
    assert "Refresh backend jobs" in html
    assert "Cancel active backend jobs" in html
    assert "Attach / recover selected dataset" in html
    assert "qwenElements.captionRefreshBackendJobs" in js
    assert "qwenElements.captionCancelBackendJobs" in js
    assert "qwenElements.captionBackendJobsSummary" in js
    assert "function summarizeQwenCaptionBackendJobs" in js
    assert "function cancelQwenCaptionBackendActiveJobs" in js
    assert "await refreshQwenCaptionBackendJobsStatus({ silent: true })" in js
def test_qwen_caption_coverage_status_tracks_imposed_question_gaps():
    js = _js()
    start = js.index("async function refreshCaptionCoverageStatus")
    end = js.index("function describeCaptionInstructionLaunchSettings", start)
    helper = js[start:end]

    assert "const effectiveGeneratedTarget = effectiveCaptionCoverageGeneratedQaTarget(settings);" in helper
    assert 'params.set("target_generated_qa_per_image", String(effectiveGeneratedTarget));' in helper
    assert 'params.set("instruction_qa_imposed_questions", JSON.stringify(imposedQuestions));' in helper
    assert "missing_imposed_question_count" in helper
    assert "missing imposed question" in helper
    assert "imposed missing" in helper


def test_qwen_caption_backend_batch_uses_visible_failure_stop_gate():
    html = _html()
    js = _js()

    single_start = js.index("async function runQwenCaptionSingleBackendJob")
    builder_start = js.index("function buildQwenCaptionDatasetJobRequestPayload", single_start)
    single_end = builder_start
    single_helper = js[single_start:single_end]
    builder_end = js.index("async function runQwenCaptionBackendBatch", builder_start)
    batch_request_builder = js[builder_start:builder_end]
    batch_start = builder_end
    batch_end = js.index("function setQwenAgentStatus", batch_start)
    batch_helper = js[batch_start:batch_end]
    request_fields_start = js.index("function buildQwenCaptionRequestFields")
    request_fields_end = js.index("function getCaptionInstructionDatasetSettings", request_fields_start)
    request_fields_helper = js[request_fields_start:request_fields_end]
    finish_start = js.index("async function finishQwenCaptionBackendJob")
    finish_end = js.index("async function monitorQwenCaptionBackendJob", finish_start)
    finish_helper = js[finish_start:finish_end]

    assert "const { requestFields } = buildQwenCaptionRequestFields(templateImageName);" in batch_request_builder
    assert "caption_request: requestFields" in batch_request_builder
    assert "instruction_qa_imposed_questions: imposedQuestions" in js
    assert "parseCaptionImposedQuestionsText" in js
    assert "captionGeneratedQaOutput" in js
    assert "labelmap_glossary: labelmapGlossary || null" in request_fields_helper
    assert "max_failures: getCaptionBackendMaxFailures(setAndForget, instructionSettings)" in single_helper
    assert "max_failures: getCaptionBackendMaxFailures(setAndForget, instructionSettings)" in batch_request_builder
    assert "runner_heartbeat_interval_seconds: DEFAULT_CAPTION_BACKEND_HEARTBEAT_SECONDS" in single_helper
    assert "runner_heartbeat_interval_seconds: DEFAULT_CAPTION_BACKEND_HEARTBEAT_SECONDS" in batch_request_builder
    assert "Backend batch complete • ${finalFailed} failed" in finish_helper
    assert "Backend caption batch complete with ${finalFailed} failed image" in finish_helper
    assert "instruction_artifacts" in finish_helper
    assert "bundle saved" in finish_helper
    assert "training bundle saved" in finish_helper
    assert "qwenCaptionSetAndForget" in html
    assert "Set-and-forget backend run" in html
    assert "qwenCaptionBatchFollowActive" in html
    assert "Follow backend image" in html
    assert "qwenCaptionAllowModelDownload" in html
    assert "Allow model downloads" in html
    assert "qwenCaptionAttempts" in html
    assert "VLM attempts" in html
    assert "Auto: set-and-forget uses 3 attempts" in html
    assert "qwenCaptionArtifactLogMb" in html
    assert "Attempt log cap (MB)" in html
    assert 'id="qwenCaptionArtifactLogMb" min="0" max="1024" step="0.25" value="0"' in html
    assert "Default 0 keeps full raw logs" in html
    assert "const DEFAULT_CAPTION_ARTIFACT_LOG_MB = 0" in js
    assert "qwenCaptionMaxRecoveryRate" in html
    assert "Max recovery rate" in html
    assert "qwenCaptionMaxFailures" in html
    assert "Stop after failed cases" in html
    assert "Generated-QA training jobs do not use this fallback" in html
    assert "qwenCaptionMaxLoopRecoveryRate" in html
    assert "Max loop recovery rate" in html
    assert "qwenCaptionMaxDeterministicRecoveryRate" in html
    assert "Max deterministic fallback rate" in html
    assert "qwenCaptionMaxFailedAttemptRate" in html
    assert "Max failed attempt rate" in html
    assert "qwenCaptionMaxSignalExitRate" in html
    assert "Max native signal-exit rate" in html
    assert "qwenCaptionMinRateCases" in html
    assert "Min live-rate cases" in html
    assert "qwenCaptionRequirePilotCertification" in html
    assert "Require certified pilot" in html
    assert "qwenCaptionPilotOutputDir" in html
    assert "Certified pilot artifact dir" in html
    assert "Required for 10k-scale set-and-forget backend launches" in html
    assert "qwenCaptionPilotTargetCases" in html
    assert "Pilot target cases" in html
    assert "qwenCaptionPilotMaxDurationHours" in html
    assert "Pilot max hours" in html
    assert "Pilot p95 max hours" in html
    assert "qwenCaptionPilotMinCases" in html
    assert "Pilot min cases" in html
    assert 'id="qwenCaptionPilotMinCases" min="1" max="1000000" step="1" value="300"' in html
    assert "default is 300 for 10k-scale deterministic-recovery confidence" in html
    assert "qwenCaptionPilotSafetyFactor" in html
    assert "Pilot safety factor" in html
    assert "qwenCaptionPilotDeterministicRecoveryConfidence" in html
    assert "Pilot deterministic confidence" in html
    assert 'id="qwenCaptionPilotDeterministicRecoveryConfidence" min="0" max="0.999999" step="0.01" value="0.95"' in html
    assert "qwenCaptionPilotRequirePromptBudget" in html
    assert "Require prompt-budget telemetry" in html
    assert "qwenCaptionPilotMaxPromptTokens" in html
    assert "Pilot max prompt tokens" in html
    assert 'id="qwenCaptionPilotMaxPromptTokens" min="0" max="1000000" step="100" value="9000"' in html
    assert "10k set-and-forget requires a positive ceiling" in html
    assert "qwenCaptionPilotPromptAdaptedRate" in html
    assert "Pilot max prompt adaptation rate" in html
    assert "const DEFAULT_CAPTION_PILOT_MIN_CASES = 300" in js
    assert "const DEFAULT_CAPTION_PILOT_DETERMINISTIC_RECOVERY_CONFIDENCE = 0.95" in js
    assert "const DEFAULT_CAPTION_PILOT_MAX_PROMPT_TOKENS = 9000" in js
    assert "generation.pilot_max_prompt_tokens ?? DEFAULT_CAPTION_PILOT_MAX_PROMPT_TOKENS" in js
    assert "const setAndForget = qwenElements.captionSetAndForget?.checked !== false" in single_helper
    assert "const healthGates = getCaptionHealthGateSettings()" in single_helper
    assert "const pilotCertification = getCaptionPilotCertificationSettings(setAndForget)" in single_helper
    assert "const providerSettings = getCaptionProviderSettings()" in single_helper
    assert "caption_provider: providerSettings.provider" in single_helper
    assert "openai_model: providerSettings.model" in single_helper
    assert "openai_image_detail: providerSettings.detail" in single_helper
    assert "openai_reasoning_effort: providerSettings.reasoningEffort" in single_helper
    assert "openai_api_key_path: providerSettings.keyPath" in single_helper
    assert "openai_service_tier: providerSettings.serviceTier" in single_helper
    assert "allowMissingForPreview" not in single_helper
    assert "save_text_labels: qwenElements.captionSaveText?.checked !== false" in single_helper
    assert "set_and_forget: setAndForget" in single_helper
    assert "allow_model_download: !!qwenElements.captionAllowModelDownload?.checked" in single_helper
    assert "runner_artifact_log_bytes: getCaptionRunnerArtifactLogBytes()" in single_helper
    assert "...healthGates" in single_helper
    assert "...pilotCertification" in single_helper
    assert "Isolated caption job auto-resumed as ${autoResumeJobId}" in single_helper
    assert "qwenCaptionBackendJobAutoResumeId(job)" in single_helper
    assert "const setAndForget = qwenElements.captionSetAndForget?.checked !== false" in batch_request_builder
    assert "const providerSettings = getCaptionProviderSettings()" in batch_request_builder
    assert "caption_provider: providerSettings.provider" in batch_request_builder
    assert "openai_model: providerSettings.model" in batch_request_builder
    assert "openai_image_detail: providerSettings.detail" in batch_request_builder
    assert "openai_reasoning_effort: providerSettings.reasoningEffort" in batch_request_builder
    assert "openai_api_key_path: providerSettings.keyPath" in batch_request_builder
    assert "openai_service_tier: providerSettings.serviceTier" in batch_request_builder
    assert "allowMissingForPreview: !!options.previewOnly" in batch_request_builder
    assert "set_and_forget: setAndForget" in batch_request_builder
    assert "allow_model_download: !!qwenElements.captionAllowModelDownload?.checked" in batch_request_builder
    assert "runner_artifact_log_bytes: getCaptionRunnerArtifactLogBytes()" in batch_request_builder
    assert "const healthGates = getCaptionHealthGateSettings()" in batch_request_builder
    assert "...healthGates" in batch_request_builder
    assert "...pilotCertification" in batch_request_builder
    assert "body: JSON.stringify(requestPayload)" in batch_helper
    assert "if (!captionProviderReadyForBackend())" in batch_helper
    assert "set_and_forget_backend" in js
    assert "require_pilot_certification" in js
    assert "pilot_output_dir" in js
    assert "pilot_target_cases" in js
    assert "pilot_require_prompt_budget_data" in js
    assert "pilot_max_prompt_tokens" in js
    assert "pilot_max_prompt_budget_adapted_case_rate" in js
    assert "pilot_deterministic_recovery_confidence" in js
    assert "pilot_max_duration_hours" in js
    assert "pilot_max_p95_duration_hours" in js
    assert "pilot_min_cases" in js
    assert "pilot_duration_safety_factor" in js
    assert "allow_model_download_backend" in js
    assert "let qwenBackendSupervision = null" in js
    assert "function qwenBackendCrashSupervisionMessage()" in js
    assert "progress.supervision" in js
    assert "status.supervision" in js
    assert "set_and_forget_ready" in js
    assert "restart_policy" in js
    assert "restart policy is not large-run ready" in js
    assert "not advertising crash-restart supervision" in js
    assert "large-run-ready crash supervision" in html
    assert "updateQwenCaptionSetAndForgetSupervisionStatus({ force: true })" in js
    assert "max_recovery_event_case_rate" in js
    assert "max_auto_resumes" in js
    assert "max_loop_recovery_case_rate" in js
    assert "max_deterministic_recovery_case_rate" in js
    assert "max_failed_attempt_row_rate" in js
    assert "max_signal_exit_attempt_row_rate" in js
    assert "min_rate_cases" in js
    assert "DEFAULT_CAPTION_HEALTH_MAX_LOOP_RECOVERY_RATE = 0.05" in js
    assert "DEFAULT_CAPTION_HEALTH_MAX_DETERMINISTIC_RECOVERY_RATE = 0.01" in js
    assert "DEFAULT_CAPTION_HEALTH_MAX_SIGNAL_EXIT_RATE = 0.05" in js
    assert "DEFAULT_CAPTION_SET_AND_FORGET_ATTEMPTS = 3" in js
    assert "DEFAULT_CAPTION_MAX_AUTO_RESUMES = 2" in js
    assert "function getCaptionBackendAttempts" in js
    assert "function getCaptionMaxAutoResumes" in js
    assert "attempts: getCaptionBackendAttempts(setAndForget)" in js
    assert "max_auto_resumes: getCaptionMaxAutoResumes(setAndForget)" in js
    assert 'id="qwenCaptionMaxAutoResumes" min="0" max="25" step="1" value="2"' in html
    assert "Default 2 prevents an unstable first image or GPU fault from looping indefinitely" in html
    assert 'id="qwenCaptionMaxLoopRecoveryRate" min="-1" max="1" step="0.01" value="0.05"' in html
    assert "enter 0 to require zero loop recoveries" in html
    assert 'id="qwenCaptionMaxDeterministicRecoveryRate" min="-1" max="1" step="0.01" value="0.01"' in html
    assert 'id="qwenCaptionMaxSignalExitRate" min="-1" max="1" step="0.01" value="0.05"' in html
    assert "function getCaptionRunnerArtifactLogBytes" in js
    assert "function getCaptionHealthGateSettings" in js
    assert "function getCaptionPilotCertificationSettings" in js
    assert "Enter a certified pilot artifact directory" in js
    assert "Pilot certification" in js
    assert "deterministic-recovery confidence" in js
    assert "function qwenCaptionBackendJobAutoResumeId" in js
    assert "function updateQwenSetAndForgetAutoAttachWatcher" in js
    assert "function qwenCaptionBackendJobDisplayError" in js
    assert "function qwenCaptionCheckReportFirstError" in js
    assert "friendlyByCode" in js
    assert "auto_resumed_job_id" in js
    assert "Backend batch auto-resumed as ${autoResumeJobId}" in js
    assert "qwenCaptionResumeBackendJob" in html
    assert "Attach / recover selected dataset" in html
    assert "this page auto-attaches immediately and periodically when backend state is available" in html
    assert "async function recoverLatestQwenCaptionBackendJob" in js
    assert "recoverLatestQwenCaptionBackendJob({ auto: true })" in js
    assert "/qwen/caption/jobs/${encodeURIComponent(job.job_id)}/resume" in js
    assert "selectRecoverableQwenCaptionBackendJob(jobs, datasetId, { auto })" in js
    assert '["queued", "running", "interrupted", "failed"].includes(status)' in js
    assert "options.auto ? null : candidates[0] || null" in js
    assert 'status === "cancelled"' in js
    assert "Cancelled caption jobs stay cancelled" in js
    assert "function runQwenSetAndForgetAutoAttachCheck" in js
    assert "function scheduleQwenSetAndForgetAutoAttachCheck" in js
    assert "window.setInterval(runQwenSetAndForgetAutoAttachCheck, 5000)" in js
    assert "scheduleQwenSetAndForgetAutoAttachCheck();" in js


def test_qwen_caption_backend_job_display_error_formats_structured_failures():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            _extract_js_function(js, "qwenCaptionCheckReportFirstError"),
            _extract_js_function(js, "qwenCaptionBackendJobDisplayError"),
            "let message = qwenCaptionBackendJobDisplayError({",
            "  status: 'failed',",
            "  error: 'caption_runner_pilot_required',",
            "  result: {",
            "    required_pilot_certification: {",
            "      status: 'error',",
            "      checks: [{ name: 'set_and_forget_pilot_required', status: 'error', detail: 'certified pilot is required before starting a set-and-forget caption job with 10000 cases' }],",
            "    },",
            "  },",
            "});",
            "assert(message.includes('Pilot certification failed: certified pilot is required before starting a set-and-forget caption job with 10000 cases'));",
            "assert(!message.includes('caption_runner_pilot_required'));",
            "message = qwenCaptionBackendJobDisplayError({",
            "  status: 'failed',",
            "  error: 'caption_runner_backend_supervision_required',",
            "  result: {",
            "    backend_supervision: {",
            "      status: 'error',",
            "      checks: [{ name: 'backend_crash_supervision', status: 'error', detail: 'large set-and-forget caption jobs require backend crash-restart supervision' }],",
            "    },",
            "  },",
            "});",
            "assert(message.includes('Backend supervision failed: large set-and-forget caption jobs require backend crash-restart supervision'));",
            "assert(!message.includes('caption_runner_backend_supervision_required'));",
            "message = qwenCaptionBackendJobDisplayError({",
            "  status: 'failed',",
            "  error: 'caption_runner_preflight_failed',",
            "  result: {",
            "    preflight: {",
            "      status: 'error',",
            "      checks: [{ name: 'model_available', status: 'error', detail: 'selected caption model is not local' }],",
            "    },",
            "  },",
            "});",
            "assert(message.includes('Caption runner preflight failed: selected caption model is not local'));",
            "assert(!message.includes('caption_runner_preflight_failed'));",
            "message = qwenCaptionBackendJobDisplayError({ status: 'failed', error: 'caption_runner_pilot_certification_failed' });",
            "assert(message.includes('Pilot certification failed. Check the certified pilot artifact directory'));",
            "assert(!message.includes('caption_runner_pilot_certification_failed'));",
            "assert.strictEqual(",
            "  qwenCaptionBackendJobDisplayError({ status: 'failed', message: 'Custom backend message', error: 'caption_runner_pilot_required' }),",
            "  'Custom backend message'",
            ");",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_caption_backend_live_progress_snapshot_normalizes_job_telemetry():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            _extract_js_function(js, "qwenCaptionCheckReportFirstError"),
            _extract_js_function(js, "qwenCaptionBackendJobDisplayError"),
            _extract_js_function(js, "qwenCaptionBackendProgressFallbackPlan"),
            _extract_js_function(js, "qwenCaptionBackendTerminalPhase"),
            _extract_js_function(js, "qwenCaptionBackendLiveProgressSnapshot"),
            "let snapshot = qwenCaptionBackendLiveProgressSnapshot({",
            "  job_id: 'job-1',",
            "  status: 'running',",
            "  message: 'Caption runner attempt running',",
            "  live_progress: {",
            "    phase: 'complete',",
            "    phase_label: 'Complete',",
            "    progress: 0.25,",
            "    step_id: 'caption',",
            "    step_index: 3,",
            "    step_total: 4,",
            "    step_label: 'Compose full-image caption',",
            "    step_plan: [{ id: 'prepare', label: 'Prepare image and prompts' }, { id: 'prompt_stack', label: 'Build prompt stack' }, { id: 'caption', label: 'Compose full-image caption' }],",
            "    token_preview: 'The scene contains a vehicle.',",
            "    latest_generated_qa_pairs: [{ id: 'qa-1', question: 'What is shown?', answer: 'A vehicle.' }],",
            "    latest_generated_qa_pair_count: 1,",
            "    latest_generated_qa_target_count: 8,",
            "    latest_generated_qa_accepted_count: 5,",
            "    latest_generated_qa_rejected_pair_count: 3,",
            "    latest_generated_qa_status: 'underfilled',",
            "    latest_generated_qa_attempt_summary: [",
            "      { profile: 'primary', label: 'Primary prompt', accepted_count: 2, rejected_count: 4 },",
            "      { profile: 'caption_grounded_fallback', label: 'Caption-grounded fallback', accepted_count: 2, rejected_count: 1 },",
            "      { profile: 'sparse_scene_fallback', label: 'Sparse-scene fallback', accepted_count: 1, rejected_count: 0 },",
            "    ],",
            "    latest_generated_qa_image_name: 'frame002.jpg',",
            "    latest_generated_qa_case_id: 'case-2',",
            "    io_events: [{ event: 'output', kind: 'output', title: 'output', text: 'The scene contains a vehicle.' }],",
            "    caption_dataset_progress: { processed: 2, total_cases: 10, case_index: 3, case: 'image_000003', image_name: 'frame003.jpg', stem: 'frame003', failed: 1, saved_text_labels: 2 },",
            "  },",
            "});",
            "assert.strictEqual(snapshot.phase, 'running');",
            "assert.strictEqual(snapshot.phase_label, 'Captioning');",
            "assert.strictEqual(snapshot.active, true);",
            "assert.strictEqual(snapshot.step_plan[1].label, 'Build prompt stack');",
            "assert.strictEqual(snapshot.caption_dataset_progress.case, 'image_000003');",
            "assert.strictEqual(snapshot.caption_dataset_progress.image_name, 'frame003.jpg');",
            "assert.strictEqual(snapshot.caption_dataset_progress.stem, 'frame003');",
            "assert(snapshot.step_detail.includes('frame003.jpg'));",
            "assert.strictEqual(snapshot.io_events[0].kind, 'output');",
            "assert.strictEqual(snapshot.latest_generated_qa_pairs[0].question, 'What is shown?');",
            "assert.strictEqual(snapshot.latest_generated_qa_pair_count, 1);",
            "assert.strictEqual(snapshot.latest_generated_qa_target_count, 8);",
            "assert.strictEqual(snapshot.latest_generated_qa_accepted_count, 5);",
            "assert.strictEqual(snapshot.latest_generated_qa_rejected_pair_count, 3);",
            "assert.strictEqual(snapshot.latest_generated_qa_status, 'underfilled');",
            "assert.strictEqual(snapshot.latest_generated_qa_attempt_summary[1].label, 'Caption-grounded fallback');",
            "assert.strictEqual(snapshot.latest_generated_qa_image_name, 'frame002.jpg');",
            "assert.strictEqual(snapshot.latest_generated_qa_case_id, 'case-2');",
            "snapshot = qwenCaptionBackendLiveProgressSnapshot({",
            "  job_id: 'job-2',",
            "  status: 'running',",
            "  progress: 0.5,",
            "  result: { processed: 4, total_cases: 8 },",
            "});",
            "assert.strictEqual(snapshot.progress, 0.5);",
            "assert.strictEqual(snapshot.step_plan[1].label, 'Build prompt stack');",
            "assert.strictEqual(snapshot.caption_dataset_progress.total_cases, 8);",
            "snapshot = qwenCaptionBackendLiveProgressSnapshot({",
            "  job_id: 'job-3',",
            "  status: 'failed',",
            "  error: 'caption_runner_preflight_failed',",
            "  live_progress: { phase: 'error', error: 'caption_runner_preflight_failed' },",
            "  result: { preflight: { status: 'error', checks: [{ name: 'model_available', status: 'error', detail: 'selected caption model is not local' }] } },",
            "});",
            "assert.strictEqual(snapshot.phase, 'error');",
            "assert(snapshot.error.includes('Caption runner preflight failed: selected caption model is not local'));",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_caption_generated_qa_accumulator_renders_compact_status():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            _extract_js_function(js, "qwenCaptionGeneratedQaAttemptLine"),
            _extract_js_function(js, "updateQwenCaptionGeneratedQaAccumulator"),
            "let qwenElements = { captionGeneratedQaAccumulator: { textContent: '', hidden: true } };",
            "updateQwenCaptionGeneratedQaAccumulator({",
            "  latest_generated_qa_image_name: 'frame132.jpg',",
            "  latest_generated_qa_target_count: 8,",
            "  latest_generated_qa_accepted_count: 5,",
            "  latest_generated_qa_rejected_pair_count: 5,",
            "  latest_generated_qa_status: 'underfilled',",
            "  latest_generated_qa_attempt_summary: [",
            "    { label: 'Primary prompt', accepted_count: 2, rejected_count: 4 },",
            "    { label: 'Caption-grounded fallback', accepted_count: 2, rejected_count: 1 },",
            "    { label: 'Sparse-scene fallback', accepted_count: 1, rejected_count: 0 },",
            "  ],",
            "  caption_dataset_progress: { case_index: 132, total_cases: 258, image_name: 'frame132.jpg' },",
            "});",
            "const text = qwenElements.captionGeneratedQaAccumulator.textContent;",
            "assert(text.includes('Image 132/258: frame132.jpg'));",
            "assert(text.includes('Caption0: complete'));",
            "assert(text.includes('Generated Q&A: 5/8 accepted, 5 rejected'));",
            "assert(text.includes('Primary prompt: 2 accepted, 4 rejected'));",
            "assert(text.includes('Caption-grounded fallback: 2 accepted, 1 rejected'));",
            "assert(text.includes('Sparse-scene fallback: 1 accepted'));",
            "assert(text.includes('Continuing with 5/8'));",
            "assert.strictEqual(qwenElements.captionGeneratedQaAccumulator.hidden, false);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_qwen_caption_model_defaults_to_active_runtime():
    html = _html()
    select_match = re.search(r'<select id="qwenCaptionModel">(.*?)</select>', html, re.S)

    assert select_match
    options = select_match.group(1)
    assert '<option value="active" selected>Auto caption default</option>' in options
    assert 'value="Qwen/Qwen3-VL-30B-A3B-Thinking" selected' not in options
    assert options.count(" selected") == 1


def test_data_ingestion_panel_contract():
    html = _html()
    css = _css()
    js = _js()
    router = _read("api/data_ingestion.py")

    assert 'id="tabDataIngestionButton"' in html
    assert 'data-tab="data-ingestion"' in html
    assert 'id="tabDataIngestion" data-tab-panel="data-ingestion"' in html
    assert "Reference dataset profile" in html
    assert "Candidate review" in html
    assert "Build a reference profile from an existing dataset" in html
    assert "Reference profiles are built from your own dataset images only" in html
    assert "Reference profiles train local SALAD" not in html
    assert "local SALAD reference profile" not in html
    assert "SALAD base encoder" not in html
    assert "value=\"reference_profile\"" in html
    assert 'id="dataIngestionFiles"' in html
    assert 'id="dataIngestionRecipe"' not in html
    assert 'id="dataIngestionTrainFiles"' not in html
    assert 'id="dataIngestionTrainActiveButton"' not in html
    assert 'id="dataIngestionReferenceActive"' in html
    assert 'id="dataIngestionReferenceBackend"' in html
    assert 'id="dataIngestionReferenceDataset"' in html
    assert 'id="dataIngestionActiveUploadName"' in html
    assert "Current upload dataset name" in html
    assert "delete it later from Dataset Management" in html
    assert 'id="dataIngestionBuildProfileButton"' in html
    assert 'id="dataIngestionProfileDownload"' in html
    assert 'id="dataIngestionProfileUpload"' in html
    assert 'id="dataIngestionMaxTrainImages" min="0" max="1000000" step="1" value="0"' in html
    assert 'id="dataIngestionReferenceCap"' not in html
    assert 'id="dataIngestionKeepFraction"' in html
    assert 'id="dataIngestionLocalVendiEnabled"' in html
    assert 'id="dataIngestionLocalVendiWeight" min="0" max="0.5" step="0.05" value="0.2"' in html
    assert "whole upload batch, not 20% of each file or video" in html
    assert "Vendi-style effective rank" in html
    assert 'id="dataIngestionUseActiveReference"' not in html
    assert 'id="dataIngestionSaladHead"' in html
    assert 'id="dataIngestionReportTitle">Ingestion report</strong>' in html
    assert 'id="dataIngestionListTitle">Candidate priority ranking</strong>' in html
    assert 'id="dataIngestionDistribution"' in html
    assert 'id="dataIngestionDistributionButton"' in html
    assert 'id="dataIngestionOpenDatasetAnalysisButton"' in html
    assert "Open Data Quality Explorer setup" in html
    assert 'id="dataIngestionDistributionGraph"' in html
    assert 'id="dataIngestionDistributionDetails"' in html
    assert 'id="dataIngestionAcceptance"' in html
    assert 'id="dataIngestionOutputMode"' in html
    assert 'value="tile" selected>Tile into crops</option>' in html
    assert 'id="dataIngestionTargetWidth" min="1" max="8192" step="1" value="960"' in html
    assert 'id="dataIngestionTargetHeight" min="1" max="8192" step="1" value="960"' in html
    assert 'id="dataIngestionTileEdgePolicy"' in html
    assert 'value="cover_no_padding" selected>Cover no padding</option>' in html
    assert 'id="dataIngestionPreviewAcceptedButton"' in html
    assert 'id="dataIngestionDownloadAcceptedButton"' in html
    assert "Split later by original source to avoid train/val leakage" in html
    assert "Use 0 for the backend safety cap" in html
    assert "Encoder guide" in html
    assert "Data Ingestion" in html
    assert 'id="dataIngestionCradioModel"' not in html
    assert 'id="dataIngestionCradioPooling"' not in html
    assert 'id="dataIngestionTrainEncoder"' in html
    assert 'id="dataIngestionTrainCradioModel"' in html

    assert '.tab-panel[data-tab-panel="data-ingestion"]' in css
    assert ".data-ingestion-workspace" in css
    assert ".data-ingestion-results" in css
    assert ".data-ingestion-acceptance" in css
    assert ".data-ingestion-distribution" in css
    assert ".data-ingestion-distribution-graph" in css
    assert ".data-ingestion-distribution-details__actions" in css
    assert ".data-ingestion-tile-preview" in css
    assert ".data-ingestion-list-controls" in css
    assert ".data-ingestion-list__metrics" in css
    assert ".embedding-benchmark-note__grid" in css
    assert "html.theme-dark .data-ingestion-panel" in css
    assert "html.theme-dark .embedding-benchmark-note" in css
    assert "html.theme-pipboy .data-ingestion-panel" in css
    assert "html.theme-pipboy .embedding-benchmark-note" in css

    assert 'const TAB_DATA_INGESTION = "data-ingestion";' in js
    assert "tabElements.dataIngestionButton = document.getElementById(\"tabDataIngestionButton\")" in js
    assert "function initDataIngestionUi" in js
    assert "function startDataIngestionAnalysis" in js
    assert "function startLocalSaladTraining" in js
    assert "function getDataIngestionDatasetEntryImageCount" in js
    assert "function downloadDataIngestionReferenceProfile" in js
    assert "function uploadDataIngestionReferenceProfile" in js
    assert "function previewDataIngestionAcceptedOutputs" in js
    assert "function downloadDataIngestionAcceptedZip" in js
    assert "function setDataIngestionItemAccepted" in js
    assert "function loadDataIngestionDistribution" in js
    assert "function renderDataIngestionDistributionGraph" in js
    assert "function openClassSplitDatasetAnalysisFromIngestion" in js
    assert 'document.getElementById("classSplitTitle")?.closest(".class-split-panel")' in js
    assert "classSplitElements.runButton.focus({ preventScroll: true })" in js
    assert "function openDatasetEntryInDataIngestion" in js
    assert js.count('activeCount > 0 && backendCount !== null && backendCount !== activeCount') >= 2
    assert 'cachedCount > 0 && backendCount !== null && backendCount !== cachedCount' in js
    assert "const backendCount = getDataIngestionDatasetImageCount(headDatasetId);" in js
    assert "appendActiveWorkspaceTrainingFiles" in js
    assert "activeDatasetSaladHeadName" in js
    assert "function getDataIngestionActiveReferenceUploadName" in js
    assert "dataIngestionElements.activeUploadName" in js
    assert 'const uploadCacheKey = `${signature}|name:${uploadName}`;' in js
    assert "getDataIngestionNumber(dataIngestionElements.maxTrainImages, 0" in js
    assert 'startLocalSaladTraining("active_dataset")' not in js
    assert "preferredSaladHeadId" in js
    assert 'setSelectValueIfPresent(dataIngestionElements.recipe, "local_salad_top20")' not in js
    assert 'fetch(`${API_ROOT}/data_ingestion/jobs`' in js
    assert 'fetch(`${API_ROOT}/data_ingestion/salad_train_jobs`' in js
    assert 'fetch(`${API_ROOT}/data_ingestion/reference_profiles/${encodeURIComponent(profileId)}/export`' in js
    assert 'fetch(`${API_ROOT}/data_ingestion/reference_profiles/import`' in js
    assert 'accepted_export/preview`' in js
    assert 'accepted_export/download`' in js
    assert 'fetch(`${API_ROOT}/data_ingestion/jobs/${encodeURIComponent(dataIngestionState.activeJobId)}/distribution`' in js
    assert "candidate_thumbnail" in router
    assert "reference_thumbnail" in router
    assert "data-data-ingestion-item-id" in js
    assert "data-data-ingestion-detail-toggle" in js
    assert "Keep candidate" in js
    assert "Discard candidate" in js
    assert "Show all candidates" in js
    assert "data-data-ingestion-output-id" not in js
    assert "acceptedOutputFilterActive" not in js
    assert "function formatDataIngestionCoverageRank" in js
    assert "Selection priority #" in js
    assert "priority score" in js
    assert "selection_score_description" in js
    assert "Coverage after cutoff" in js
    assert "Reference novelty p" in js
    assert "Ordered by selection priority across the whole pooled upload batch" in js
    assert "formatDataIngestionLocalVendi" in js
    assert "local_vendi_enabled" in js
    assert "Reference novelty" in js
    assert "dataIngestionHoverPreview" in js
    assert "existing reference data" in js
    assert "Data Quality Explorer Dataset Analysis is not required for this map" in js
    assert "This does not require Data Quality Explorer Dataset Analysis" in html
    assert "dataIngestionDistributionStatus" in js
    assert "width: min(400px, calc(100vw - 24px));" in css
    assert "Tile overlap creates near-duplicates" in js
    assert 'fetch(`${API_ROOT}/datasets`' in js
    assert "appendActiveWorkspaceReferenceFiles" in js
    assert "dataIngestionRecipeValues" not in js
    assert "dataIngestionElements.referenceDataset" in js
    assert "reference_dataset_id" in js
    assert "reference_source" in js
    assert "function dataIngestionBackendDatasetExists" in js
    assert "function dataIngestionActiveTransientReferenceId" in js
    assert "function getDataIngestionServerReferenceHandle" in js
    assert "function shouldDataIngestionUseBackendReferenceDataset" in js
    assert "function shouldDataIngestionUseServerReferenceDataset" in js
    assert "use_backend_reference_dataset" in js
    assert "use_server_reference_dataset" in js
    assert "reference_session_id" in js
    assert "reference_open_path" in js
    assert "reference_dataset_kind" in js
    assert "Too many reference files were uploaded" in js
    assert "activeCount > 500 && dataIngestionBackendDatasetExists(selectedDatasetId)" not in js
    assert "function dataIngestionHeadMatchesReference" in js
    assert "const activeHandle = getDataIngestionServerReferenceHandle(\"active_label_images\");" in js
    assert "const selectedHeadReferenceHandle = dataIngestionHandleFromSelectedHead(selectedHead);" in js
    assert "activeReferenceCount > 0 || !!selectedHeadReferenceHandle" in js
    assert "if (!dataIngestionBackendDatasetExists(headDatasetId)) return null;" in js
    assert "imageCount: activeCount || backendCount || headActiveCount || 0" in js
    assert "const headLabel = String(head.reference_label || head.reference_dataset_label || \"\").trim();" in js
    assert "const activeLabel = getDataIngestionReferenceLabel();" in js
    assert 'if (!headReferenceSource && !headDatasetId && !headSessionId) {\n            return false;' in js
    assert "function handleDataIngestionReferenceChange" in js
    assert "dataIngestionElements.reportTitle" in js
    assert "dataIngestionElements.listTitle" in js
    assert "Reference profile report" in js
    assert "Profile ready for candidate review" in js
    assert "heads.filter((head) => dataIngestionHeadMatchesReference(head))" in js
    assert "Choose matching reference profile" in js
    assert "No reference profiles built" in js
    assert "No local SALAD heads trained" not in js
    assert "No profiles for selected reference" in js
    assert "Use for ingestion" in js
    assert "action.datasets.card.use_for_ingestion" in js
    assert "Selected reference profile does not match the chosen reference dataset" in js
    assert "Open at least two images in Label Images before building a reference profile" in js
    assert "Reference profile build needs at least two usable images or frames." in js
    assert "Reference profile build did not return a job id." in js
    assert "Profile policy" in js
    assert "SALAD policy" not in js
    assert "function formatDataIngestionProfilePolicy" in js
    assert "encoder: \"local_salad\"" in js
    assert "dataIngestionElements.trainEncoder" in js
    assert "dataIngestionElements.trainCradioModel" in js
    assert "local_salad" in js

    assert '"/data_ingestion/capabilities"' in router
    assert '"/data_ingestion/salad_train_jobs"' in router
    assert '"/data_ingestion/reference_profiles/{profile_id}/export"' in router
    assert '"/data_ingestion/reference_profiles/import"' in router
    assert '"/data_ingestion/jobs/{job_id}/accepted_export/preview"' in router
    assert '"/data_ingestion/jobs/{job_id}/distribution"' in router
    assert '"/data_ingestion/jobs/{job_id}/reference_thumbnail/{point_id}"' in router
    assert '"/data_ingestion/jobs/{job_id}/accepted_export/{preview_id}/thumbnail/{output_id}"' in router
    assert '"/data_ingestion/jobs/{job_id}/accepted_export/download"' in router
    assert 'max_part_size=1024 * 1024 * 1024' in router


def test_dataset_manager_delete_trash_restore_contract():
    html = _html()
    js = _js()
    router = _read("api/datasets.py")

    assert 'id="datasetTrashRefresh"' in html
    assert 'id="datasetTrashList"' in html
    assert 'id="datasetUploadSessionsRefresh"' in html
    assert 'id="datasetUploadSessionsList"' in html
    assert 'data-testid="list.datasets.trash"' in html
    assert 'data-testid="list.datasets.upload_sessions"' in html
    assert "Deleted managed datasets" in html
    assert "Managed dataset deletes move here first" in html
    assert "Staged upload sessions" in html
    assert "Cancelling removes only temporary upload chunks" in html

    assert "trashRefresh: null" in js
    assert "trashList: null" in js
    assert "uploadSessionsRefresh: null" in js
    assert "uploadSessionsList: null" in js
    assert "trashRefreshInFlight: false" in js
    assert "uploadSessionRefreshInFlight: false" in js
    assert "function renderDatasetUploadSessions(list)" in js
    assert "async function handleDatasetUploadSessionCancel(entry)" in js
    assert "async function refreshDatasetUploadSessions()" in js
    assert "fetch(`${API_ROOT}/datasets/upload_sessions`)" in js
    assert "fetch(`${API_ROOT}/datasets/upload_session/${encodeURIComponent(sessionId)}/cancel`" in js
    assert "function renderDatasetTrashList(list)" in js
    assert "async function handleDatasetTrashRestore(entry)" in js
    assert "async function refreshDatasetTrashList()" in js
    assert "fetch(`${API_ROOT}/datasets/trash`)" in js
    assert "fetch(`${API_ROOT}/datasets/trash/${encodeURIComponent(entry.trash_id)}/restore`" in js
    assert 'Move managed dataset "${entry.label || entry.id}" to deleted datasets?' in js
    assert "function isDatasetEntryLinkedUnavailable(entry)" in js
    assert "Linked root is unavailable; fix or re-register the dataset path before using source-dependent actions." in js
    assert "switched annotation to the persistent linked dataset" in js
    assert "Reopen it from the dataset card for persistent edits." not in js
    delete_start = js.index("async function handleDatasetDelete")
    delete_end = js.index("async function handleDatasetConvert")
    assert "This cannot be undone." not in js[delete_start:delete_end]

    assert '@router.get("/datasets/upload_sessions")' in router
    assert '@router.post("/datasets/upload_session/{session_id}/cancel")' in router
    assert '@router.get("/datasets/trash")' in router
    assert '@router.post("/datasets/trash/{trash_id}/restore")' in router


def test_class_scroll_contrast_and_double_w_selected_scope_contract():
    css = _css()
    js = _js()

    assert "function getTextColorForClassToastBackground" in js
    assert 'return relativeLuminance(backgroundHex) > 0.7 ? "#111827" : "#f8fafc";' in js
    assert "textOnStrongBg = getTextColorForClassToastBackground(strongBg)" in js
    assert "for (let offset = -aboveCount; offset <= belowCount; offset++)" in js
    assert "--bubble-font-size" in js
    assert "--bubble-scale" in js
    assert "max-height: calc(100vh - 32px);" in css
    assert "-webkit-text-fill-color: var(--class-text, #f8fafc);" in css
    assert "--bubble-font-size: 23.8px;" in css
    assert "function getBatchTweakSelectionTarget" in js
    assert "getSelectedBboxRecords({ negative: false })" in js
    assert "runBatchTweakForRecords(selectionTarget.records, selectionTarget.className" in js
    assert 'scopeLabel: "selected"' in js
    assert "async function runBatchTweakForRecords" in js
    assert "async function runBatchTweakForClass" in js
    assert "function handleMagicTweakTapHotkey" in js
    assert "let magicTweakHotkeyTimeoutId = null" in js
    assert 'id: "magic_tweak"' in js
    assert 'defaultBindings: [makeBinding("KeyW", "W")]' in js
    assert 'id: "delete_selected_current"' in js
    assert 'makeBinding("KeyX", "X")' in js


def test_sam3_text_panel_has_dark_theme_coverage():
    css = _css()

    assert "html.theme-dark .tool-panel[open] > summary" in css
    assert "html.theme-dark .sam3-labelmap-extension" in css
    assert "html.theme-dark .sam3-text-batch" in css
    assert "html.theme-dark .sam3-text-cascade__dedupe" in css
    assert "html.theme-dark .sam3-text-panel label" in css
    assert "html.theme-pipboy .tool-panel[open] > summary" in css
    assert "html.theme-pipboy .sam3-labelmap-extension" in css
    assert "html.theme-pipboy .sam3-text-batch" in css
    assert "html.theme-pipboy .sam3-text-cascade__dedupe" in css
    assert "html.theme-pipboy .sam3-text-panel label" in css


def test_sam3_text_panel_controls_use_aligned_field_layout():
    html = _html()
    css = _css()
    js = _js()

    assert 'id="sam3TextWorkflow" class="qwen-caption-workflow sam3-text-workflow"' in html
    batch_tag = _details_opening_tag(html, "sam3TextBatchPanel")
    assert " open" not in batch_tag
    assert "<summary>Apply to next N images</summary>" in html
    assert 'class="sam3-text-batch__body"' in html
    assert 'id="sam3BatchModeSingle"' in html
    assert 'id="sam3BatchModeCascade"' in html
    assert "Current single prompt" in html
    assert "Text prompt cascade" in html
    assert '<label for="sam3BatchCount">Images to process</label>' in html
    assert 'id="sam3TextCascadeDedupeAssigned"' in html
    assert 'id="sam3TextCascadeDedupeIou"' in html
    assert "Dedupe assigned classes after cascade" in html
    assert 'class="sam3-text-field sam3-text-field--wide"' in html
    assert 'class="sam3-text-field sam3-text-field--checkbox"' in html
    assert '<label for="sam3Threshold">Score threshold</label>' in html
    assert '<label for="sam3ClassSelect">Assign detections to class</label>' in html
    assert "sam3-text-cascade__max-points" in js
    assert "sam3-text-cascade__windowed" in js
    assert "sam3-text-cascade__window-size" in js
    assert "sam3-text-cascade__window-overlap" in js
    assert "updateSam3TextCascadeStepWindowControls" in js
    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in css
    assert ".sam3-text-batch > summary" in css
    assert ".sam3-text-batch__body" in css
    assert ".sam3-text-batch__mode" in css
    assert ".sam3-text-cascade__dedupe" in css
    assert ".sam3-text-field.is-disabled" in css
    assert ".sam3-text-cascade__step-grid > div" in css
    assert ".sam3-text-live-toast" in css
    assert "batchModeSingleRadio" in js
    assert "batchModeCascadeRadio" in js
    assert "getSam3TextBatchMode" in js
    assert "Open the cascade editor and add at least one cascade step" in js
    assert "cascadeDedupeToggle" in js
    assert "getSam3TextCascadePostDedupeConfig" in js
    assert "dedupeSam3AssignedClassesForCurrentImage" in js
    assert "getSam3TextCascadeAssignedClasses" in js
    assert "maxPointsPerPolygon: maxPoints" in js
    assert 'class="sam3-text-field">\n                    <label>Score threshold</label>' in js
    assert "function updateSam3TextWorkflow" in js
    assert "function startSam3TextWindowOverlay" in js
    assert "drawSam3TextRegionOverlay(context)" in js


def test_class_split_explorer_panel_contract():
    html = _html()
    css = _css()
    js = _js()
    router = _read("api/class_analysis.py")

    assert 'plotly-2.35.2.min.js' in html
    assert 'id="tabClassSplitButton"' in html
    assert 'data-tab="class-split"' in html
    assert 'id="tabClassSplit" data-tab-panel="class-split"' in html
    assert 'class="class-split-workspace"' in html
    assert 'class="class-split-panel class-split-panel--workspace"' in html
    assert 'id="classSplitDetails"' not in html
    guided_ids = (
        "classSplitGuidedSetup",
        "classSplitScopeSelected",
        "classSplitScopeAll",
        "classSplitClassSelect",
        "classSplitProjection",
        "classSplitFeatureMode",
        "classSplitFeatureTuning",
        "classSplitUseEl2n",
        "classSplitCompactWeight",
        "classSplitQualityMemoryPolicy",
        "classSplitRefineOutliers",
        "classSplitResolvedSummary",
        "classSplitRunButton",
        "classSplitProgressPhase",
        "classSplitProgressPercent",
        "classSplitProgressSteps",
    )
    for element_id in guided_ids:
        assert html.count(f'id="{element_id}"') == 1
    step_titles = (
        "Choose what to inspect",
        "Choose the map",
        "Choose the feature system",
        "Choose the memory policy",
        "Add spatial evidence",
        "Review the recipe and run",
    )
    assert [html.index(title) for title in step_titles] == sorted(
        html.index(title) for title in step_titles
    )
    assert '<option value="thorough_quality_v1" selected>Thorough multi-backbone (recommended)</option>' in html
    assert '<option value="precise_compact_v1">Balanced compact fusion</option>' in html
    assert '<option value="fast_map_v1">Fast single-backbone</option>' in html
    assert '<option value="local_salad">Local SALAD separation</option>' not in html
    assert '<option value="class_balanced_pca">Class-balanced PCA</option>' in html
    assert '<option value="global_pca">Global PCA</option>' in html
    assert '<option value="between_class_pca">Between-class PCA</option>' in html
    assert '<option value="within_filter_pca">Within-filter PCA</option>' in html
    assert '<option value="umap" selected>UMAP local neighborhoods</option>' in html
    assert 'id="classSplitProjectionHint"' in html
    assert "Map tuning" in html
    assert 'id="classSplitProjectionMetric"' in html
    assert '<option value="cosine" selected>cosine</option>' in html
    assert '<option value="euclidean">euclidean</option>' in html
    assert 'id="classSplitRefineOutliers" checked' not in html
    assert "Spatial evidence refinement" in html
    assert "without asking a VLM to judge labels" in html
    assert "VLM evidence pass" not in html
    assert html.count('id="classSplitRefinementPreview"') == 1
    assert 'id="classSplitRefinementPreview" open' not in html
    assert "It works with PCA or UMAP" in html
    assert "this analysis pass does not make the final judgment" in html
    assert "never creates synthetic review items" in html
    assert 'id="classSplitRecipeExplanation"' in html
    assert 'id="classSplitPreprocessMode"' in html
    assert 'id="classSplitSizeBiasMode"' in html
    assert 'id="classSplitAdvancedSetup"' not in html
    assert 'id="classSplitEmbeddingGuide"' not in html
    assert "classSplitAdvancedSetup" not in js
    assert 'thorough_quality_v1: { mode: "multi_backbone_fusion"' in js
    assert 'precise_compact_v1: { mode: "compact_fusion"' in js
    assert 'fast_map_v1: { mode: "single_backbone"' in js
    assert "renderClassSplitGuidedRecipeExplanation" in js
    assert "control.disabled = setupLocked" in js
    assert html.count('id="classSplitProgressText"') == 1
    assert 'id="classSplitProgressText" class="training-help" role="status" aria-live="polite"' in html
    assert "progressState.progress" in js
    assert "renderClassSplitProgressPhases(job, progress, stageIndex, stageLabel);" in js
    assert 'id="classSplitGraph" class="class-split-graph"' in html
    assert 'id="classSplitDisplayMode"' in html
    assert 'id="classSplitGraphProjection"' in html
    assert 'id="classSplitDragMode"' in html
    assert '<option value="class_balanced_pca">Balanced class overview</option>' in html
    assert '<option value="global_pca">Overall visual similarity</option>' in html
    assert '<option value="between_class_pca">Emphasize class differences</option>' in html
    assert '<option value="within_filter_pca">Detail within selected class</option>' in html
    assert '<option value="umap" selected>Local similarity groups (UMAP)</option>' in html
    assert "Suggested graph view (layout fidelity; likely-wrong ranking is unchanged)" in js
    assert '<option value="wrong">Likelihood the label is wrong</option>' in html
    assert '<option value="outlier">How unusual the object looks</option>' in html
    projection_options = _extract_js_function(js, "updateClassSplitProjectionOptions")
    assert 'select === classSplitElements.graphProjection' in projection_options
    assert 'class_balanced_pca: "Balanced class overview"' in projection_options
    assert 'global_pca: "Overall visual similarity"' in projection_options
    assert 'umap: availableProjectionMethods.includes("umap")' in projection_options
    assert '<option value="pan">Move map</option>' in html
    assert '<option value="wrong_only">Suggested review queue</option>' in html
    assert '<option value="rough_only">All similarity-based flags</option>' in html
    assert '<option value="cluster">Cluster</option>' not in html
    assert 'id="classSplitClusterOverlay"' not in html
    assert 'id="classSplitClusterSource"' in html
    assert '<option value="umap_islands" selected>UMAP island proposals</option>' in html
    assert '<option value="embedding_kmeans">Strict embedding clusters</option>' in html
    assert 'id="classSplitClusterSensitivity"' in html
    assert 'id="classSplitClusterMaxClusters"' in html
    assert 'id="classSplitClusterMinSize"' in html
    assert 'id="classSplitClusterUmapNeighbors" min="0" max="5000" value="15"' in html
    assert 'id="classSplitClusterUmapMinDist" min="0" max="0.99" step="0.01" value="0.02"' in html
    assert 'id="classSplitClusterRun"' in html
    assert "UMAP island proposals search the selected class in a local UMAP map" in html
    assert 'id="classSplitCradioPooling"' in html
    assert 'id="classSplitReport" class="class-split-report"' in html
    assert 'id="classSplitBulkPanel" class="class-split-bulk-panel class-split-multi-selection"' in html
    assert 'id="classSplitGraphStatus" class="class-split-graph-status" aria-live="polite"' in html
    assert 'id="classSplitClusterPanel" class="class-split-review-section class-split-cluster-panel" open' in html
    assert 'id="classSplitClusterList" class="class-split-cluster-list"' in html
    assert 'id="classSplitWrongPanel" class="class-split-review-section class-split-wrong-panel class-split-wrong-panel--wide" open' in html
    assert 'id="classSplitWrongQueueStatus"' in html
    assert 'id="classSplitVignetteCategory"' in html
    assert '<option value="review_queue" selected>Suggested review queue</option>' in html
    assert "Candidates to review" in html
    assert "Choose the suggested queue or a specific explanation group" in html
    assert "classSplitElements.vignetteCategory.disabled = !refined || showAllRough;" in js
    assert '<option value="confirmed_outlier">Likely wrong / anomalous</option>' in html
    assert '<option value="explained_not_outlier">Explained by overlap or context</option>' in html
    assert '<option value="mixed_or_composite">Mixed or composite (human triage)</option>' in html
    assert '<option value="unresolved">Unresolved</option>' in html
    assert '<option value="pair_conflict">Dual-box conflicts</option>' in html
    assert 'id="classSplitVignetteSort"' in html
    assert '<option value="priority" selected>Review priority</option>' in html
    assert '<option value="suspicion">Most likely wrong first</option>' in html
    assert '<option value="suspicion_ascending">Least likely wrong first</option>' in html
    assert '<option value="analysis_order">Original analysis order</option>' in html
    assert "Choose which candidates appear first" in html
    assert 'id="classSplitShowAllRough" checked' not in html
    assert "Show all flagged objects" in html
    assert "Priority demotes size-only evidence" in html
    assert '<option value="low_detail">Tiny / low-detail boxes</option>' in html
    assert 'id="classSplitLimitPlotPoints"' not in html
    assert 'id="classSplitVignetteOrderHelp"' in html
    assert "Sorting changes display order only" in html
    assert 'class="class-split-vignette-counts-details"' in html
    assert 'id="classSplitVignetteCategoryCounts"' in html
    assert 'id="classSplitWrongDiscardCount" min="1" max="12" step="1" value="12"' in html
    assert 'id="classSplitWrongDiscardFirst"' in html
    assert 'id="classSplitWrongShuffle"' in html
    assert 'id="classSplitQwenReviewMechanism"' in html
    assert "How VLM vignette review works" in html
    assert 'id="classSplitQwenReviewTraceToggle"' in html
    assert 'id="classSplitQwenReviewTraceToast"' in html
    assert 'id="classSplitQwenReviewTraceBody"' in html
    assert 'id="classSplitQwenReviewTraceClose"' in html
    assert 'id="classSplitQwenKeepLoaded" checked' in html
    assert "Keep VLM loaded" in html
    assert "keep_vlm_loaded:" in js
    assert "classSplitState.qwenKeepLoaded" in js
    assert "runtime_retention" in js
    assert "Review with VLM" in js
    assert "VLM reviewing ..." in js
    assert "Cancel VLM review" in js
    assert "Review with Qwen" not in js
    assert "Qwen reviewing ..." not in js
    assert 'id="classSplitWrongList"' in html
    assert ".class-split-wrong-item__technical" in css
    assert ".class-split-selector-metrics" in css
    assert ".class-split-selector-metric--review-value" in css
    assert '.class-split-wrong-item__body [data-action="selector-help"][data-tooltip]:focus-visible' in css
    assert ".class-split-selector-explanation" in css
    assert ".class-split-wrong-item__overlap-prior" in css
    assert ".class-split-wrong-item__rank-adjustment--semantic" in css
    assert ".class-split-wrong-item__rank-adjustment--frequency" in css
    assert ".class-split-wrong-item__badge--unresolved" in css
    assert re.search(
        r"html\.theme-dark \.class-split-wrong-item__body "
        r"\.class-split-wrong-item__badge--unresolved\s*\{[^}]*"
        r"background:\s*#1e293b;[^}]*color:\s*#f1f5f9;",
        css,
        re.DOTALL,
    )
    assert re.search(
        r"html\.theme-dark \.class-split-wrong-item__body "
        r"\.class-split-wrong-item__rank-adjustment--semantic\s*\{[^}]*"
        r"background:\s*#0c4a6e;[^}]*color:\s*#e0f2fe;",
        css,
        re.DOTALL,
    )
    assert re.search(
        r"html\.theme-dark \.class-split-refinement-quality-banner,\s*"
        r"html\.theme-pipboy \.class-split-refinement-quality-banner\s*\{[^}]*"
        r"background:\s*#422006;[^}]*color:\s*#fef3c7;",
        css,
        re.DOTALL,
    )
    assert 'id="classSplitInspector"' in html
    assert '<option value="image_value">Image review value</option>' in html
    assert 'id="classSplitDatasetAnalysisPanel" class="class-split-panel class-split-dataset-analysis"' in html
    assert 'id="classSplitDatasetAnalysisPanel" class="class-split-panel class-split-dataset-analysis" open' not in html
    assert "Optional image-value review" in html
    assert 'id="classSplitDatasetAnalysisRun"' in html
    assert 'id="classSplitDatasetAnalysisGraph" class="class-split-dataset-graph"' in html
    assert 'id="classSplitDatasetAnalysisList" class="class-split-dataset-list"' in html
    assert "Selected crop" in html
    assert "Likely wrong class" in html

    assert '.tab-panel[data-tab-panel="class-split"]' in css
    assert ".class-split-workspace" in css
    assert "grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));" in css
    assert ".class-split-field--projection" in css
    assert ".class-split-panel--workspace .class-split-results" in css
    assert ".class-split-graph-footer" in css
    assert ".class-split-bulk-panel" in css
    assert ".class-split-graph-status" in css
    assert ".class-split-graph" in css
    assert ".class-split-report" in css
    assert ".class-split-cluster-list" in css
    assert ".class-split-cluster-item" in css
    assert ".class-split-wrong-panel--wide" in css
    assert ".class-split-wrong-toolbar__discard" in css
    assert ".class-split-refinement-control" in css
    assert ".class-split-vignette-filters" in css
    assert ".class-split-refinement-evidence" in css
    assert ".class-split-qwen-mechanism" in css
    assert ".class-split-qwen-trace-toast" in css
    assert ".class-split-qwen-trace-toast__body" in css
    assert "width: min(780px, calc(100vw - 36px));" in css
    assert "--class-split-qwen-review-bg" in css
    assert "html.theme-pipboy .class-split-qwen-review" in css
    assert ".class-split-wrong-item__preview" in css
    assert ".class-split-wrong-item__badge--dual" in css
    assert (
        "html.theme-dark .class-split-wrong-item__body "
        ".class-split-wrong-item__badge--dual"
    ) in css
    assert "background: rgba(127, 29, 29, 0.72);" in css
    assert "grid-template-columns: 232px minmax(0, 1fr);" in css
    assert "width: 232px;" in css
    assert "height: 192px;" in css
    assert ".class-split-results[hidden]" in css
    assert ".data-ingestion-results[hidden]" in css
    assert ".class-split-graph-hover-preview" in css
    assert ".class-split-graph.class-split-graph--pan" not in css
    assert ".class-split-review-section--inspector" in css
    assert ".class-split-hover-card" not in css
    assert "--class-split-crop-scale" in css
    assert "max-width: none;" in css
    assert "transition: width 0.08s ease-out, height 0.08s ease-out;" in css
    assert ".embedding-recipe-note__text" in css
    assert ".class-split-review" in css
    assert ".class-split-dataset-analysis" in css
    assert ".class-split-dataset-hover-preview" in css
    assert '"dataset dataset"' not in css
    assert "html.theme-pipboy .class-split-dataset-analysis" in css
    assert "height: calc(100vh - 330px);" in css
    assert "html.theme-dark .class-split-panel" in css
    assert "raw VLM / controller output" in js
    assert "review.trace_events" in js
    assert "html.theme-dark .class-split-workspace__header" in css
    assert "html.theme-pipboy .class-split-panel" in css
    assert '"toolbar toolbar"\n        "status status"\n        "graph review"\n        "footer footer"\n        "wrong wrong"' in css

    assert 'const TAB_CLASS_SPLIT = "class-split";' in js
    assert "const TOP_TAB_KEYS = new Set([" in js
    assert "tabElements.classSplitButton = document.getElementById(\"tabClassSplitButton\")" in js
    assert "setActiveTab(TAB_CLASS_SPLIT)" in js
    assert "function handleTopTabNavigationClick(event)" in js
    assert 'event.target.closest(".tab-button[data-tab]")' in js
    assert "document.addEventListener(\"click\", handleTopTabNavigationClick, true);" in js
    assert "let tabNavigationInitialized = false;" in js
    assert "if (tabNavigationInitialized) {\n            setActiveTab(activeTab);\n            return;\n        }\n        tabNavigationInitialized = true;" in js
    assert "initializeThemeToggle();\n        setupTabNavigation();\n        applyPlaywrightTestIds();" in js
    assert "document.readyState !== \"complete\"" in js
    assert "const classSplitElements = {" in js
    assert "const classSplitState = {" in js
    assert "function initClassSplitExplorer" in js
    assert "function startClassSplitAnalysis" in js
    assert "function getClassSplitSampleCap" in js
    assert "request.sample_cap = sampleCap" in js
    assert "function getClassSplitProjectionRequestParts" in js
    assert "function inferClassSplitResultSelectedProjection" in js
    assert "function getClassSplitPointProjection" in js
    assert "function ensureClassSplitProjectionCoordinates" in js
    assert "function getClassSplitGraphViewModel" in js
    assert "function getClassSplitVisibleClassNames" in js
    assert "function buildClassSplitClassTraces" in js
    assert 'const pointTraces = colorMode === "class"' in js
    assert "getClassSplitClassColorTokens(className, points).stroke" in js
    assert "const CLASS_SPLIT_MAX_PLOT_POINTS = 50000;" in js
    assert "function sampleClassSplitGraphPoints" in js
    assert "plot thinned at ${view.plotCap} to keep the browser responsive" in js
    assert "const markerLine = getClassSplitPointMarkerLine(point, { suspiciousTrace });" in js
    assert "markerLineWidths.push(markerLine.width)" in js
    assert "function updateClassSplitGraphStatus" in js
    assert "function hideClassSplitResultUiUntilReady" in js
    assert "function syncClassSplitSetupControlsFromResult" in js
    assert "function applyClassSplitResultPayload" in js
    assert "function installTatorTestHooks" in js
    assert "classSplitApplyResult" in js
    assert "classSplitEmitPointClick" in js
    assert "classSplitEnterRunningState" in js
    assert "classSplitElements.graphProjection" in js
    assert "classSplitElements.graphStatus" in js
    assert "classSplitElements.qwenReviewTraceToggle" in js
    assert "function renderClassSplitQwenReviewTraceToast" in js
    assert "function buildClassSplitQwenReviewTraceHtml" in js
    assert "function buildClassSplitQwenReviewTraceText" in js
    assert "Audit trail intermediate outputs" in js
    assert "setClassSplitQwenReviewTraceEnabled" in js
    assert "plotRenderToken" in js
    assert "projection_mode: projectionParts.projectionMode" in js
    assert "projection_neighbor_k: projectionNeighborK" in js
    assert "projection_min_dist: projectionMinDist" in js
    assert "projection_metric: projectionMetric" in js
    assert "projection_spread: projectionSpread" in js
    assert "refine_outliers: refineOutliers" in js
    assert "refinement_schema: refinementCompatibility.schema" in js
    assert 'CLASS_SPLIT_REFINEMENT_API_VERSION = 5' in js
    assert 'CLASS_SPLIT_REFINEMENT_SCHEMA = "class-analysis-patch-refinement-v5"' in js
    assert 'CLASS_SPLIT_REFINEMENT_DECISION_CONTRACT = "class-analysis-patch-decision-v9"' in js
    assert "directed_pair_bank_reliable" in js
    assert "directed_pair_candidate_source_excluded" in js
    assert "directed_pair_candidate_source_fingerprint" in js
    assert "directed_pair_candidate_source_membership_roles" in js
    assert "directed_pair_candidate_source_independent" in js
    assert "positive pair confirmation blocked" in js
    assert "function getClassSplitRefinementCompatibility" in js
    assert "function refreshClassSplitRefinementControl" in js
    assert "proposal_source:" in js
    assert "umap_neighbors:" in js
    assert "umap_min_dist:" in js
    assert "cradio_pooling:" in js
    assert "classSplitElements.cradioPooling" in js
    assert "function updateClassSplitEmbeddingRecipeExplanation" in js
    assert "function updateTrainingEmbeddingRecipeExplanation" in js
    assert "regresses out log bbox area, log crop area, bbox aspect, and crop aspect" in js
    assert "Why fixed canonical crops" in js
    assert "Mean-color padding avoids adding artificial black borders" in js
    assert "Why not full whitening" in js
    assert "full covariance whitening/PCA would rotate every embedding dimension" in js
    assert 'applyEmbeddingRecipePresetToClassSplit(classSplitElements.recipePreset?.value || "precise")' in js
    assert 'preprocess_mode: String(classSplitElements.preprocessMode?.value || "canonical")' in js
    assert 'embedding_adjustment: String(classSplitElements.sizeBiasMode?.value || "remove_size_bias")' in js
    assert 'if (trainingElements.preprocessModeSelect) {\n            formData.append("preprocess_mode", trainingElements.preprocessModeSelect.value || "canonical");\n        }' in js
    assert 'if (trainingElements.embeddingAdjustmentSelect) {\n            formData.append("embedding_adjustment", trainingElements.embeddingAdjustmentSelect.value || "remove_size_bias");\n        }' in js
    assert 'Encoder type<span class="help-icon"' in html
    assert 'C-RADIOv4 Backbone<span class="help-icon"' in html
    assert "C-RADIOv4 uses the shared backend: local MLX on Mac" in html
    assert 'Embedding preset<span class="help-icon"' in html
    assert 'id="trainEmbeddingRecipeExplanation"' in html
    assert 'id="trainPreprocessMode"' in html
    assert 'id="trainEmbeddingAdjustment"' in html
    assert "Auto-class crop preprocessing" in html
    assert "diagonal standardization, not full PCA/ZCA whitening" in html
    assert "function getTrainingEmbeddingDimMultiplier" in js
    assert 'if (encoderType === "cradio")' in js
    assert 'if (lower.includes("so400m")) return 1152;' in js
    assert 'if (lower.includes("c-radiov4-h") || lower.endsWith("/h")) return 1280;' in js
    assert 'pooling === "summary_spatial_concat"' in js
    assert 'pooling === "cls_patch_concat"' in js
    assert "baseDim * getTrainingEmbeddingDimMultiplier(encoderType)" in js
    assert 'Crop geometry<span class="help-icon"' in html
    assert 'Background<span class="help-icon"' in html
    assert 'Embedding views<span class="help-icon"' in html
    assert 'DINOv3 pooling<span class="help-icon"' in html
    assert 'C-RADIOv4 pooling<span class="help-icon"' in html
    assert "Active Label Images workspace" in js
    assert "function buildClassSplitActiveWorkspaceForm" in js
    assert "function getClassSplitPointImageKey" in js
    assert "function renderClassSplitReport" in js
    assert "function runClassSplitDatasetAnalysis" in js
    assert "function getClassSplitGraphPoints" in js
    assert "function classSplitPointMatchesActiveGraphView" in js
    assert "function refreshClassSplitFilteredReviewUi" in js
    assert 'displayMode === "wrong_only"' in js
    assert 'displayMode === "rough_only"' in js
    assert "function classSplitPointIsPrimaryCandidate" in js
    assert "function classSplitPointIsRoughCandidate" in js
    primary_candidate = _extract_js_function(js, "classSplitPointIsPrimaryCandidate")
    assert '"confirm_current"' in primary_candidate
    assert '"reassign_class"' in primary_candidate
    assert '"delete_bbox"' in primary_candidate
    assert '"delete_current_box"' in primary_candidate
    assert '"keep_both_boxes"' in primary_candidate
    assert "function showClassSplitGraphHoverPreview" in js
    assert "function bindClassSplitGraphHoverPreviewMovement" in js
    assert "classSplitGraphHoverPreview" in js
    assert "selectedpoints" in js
    assert "selectionrevision" in js
    assert 'dragmode: String(classSplitElements.dragMode?.value || "lasso")' in js
    assert "function renderClassSplitClusterList" in js
    assert "function startClassSplitClusterSearch" in js
    assert "function pollClassSplitClusterSearch" in js
    assert "function selectClassSplitCluster" in js
    assert "function classSplitClusterProposalsAllowed" in js
    assert "function formatClassSplitClusterReport" in js
    assert "point.class_cluster_id" not in js
    assert "classSplitState.clusterSearchResult" in js
    assert "Subclass clustering is disabled for all-class graphs" in js
    assert "Click Find subclass clusters" in js
    assert "UMAP island mode follows local visual density" in js
    assert "Strict embedding KMeans proposals" in js
    assert "defaultClassSplitProjectionForScope" in js
    assert 'classSplitUmapAvailable() ? "umap" : "within_filter_pca"' in js
    assert ': (classSplitUmapAvailable() ? "umap" : "class_balanced_pca");' in js
    assert "projectionNeighborK.disabled = !available || classSplitState.active || mutationBusy || projectionChoice !== \"umap\"" in js
    assert 'classSplitElements.displayMode.value = "all";' in js
    assert "function buildClassSplitClusterHullTraces" in js
    assert "computeDatasetImageValueAnalysis(points)" in js
    assert "function isClassSplitDatasetAnalysisPanelOpen" in js
    assert "function renderClassSplitDatasetAnalysisGraph" in js
    assert 'classSplitElements.datasetAnalysisPanel.addEventListener("toggle"' in js
    assert "renderClassSplitDatasetAnalysisGraph(classSplitState.datasetAnalysis, { force: true })" in js
    assert "classSplitDatasetAnalysisHoverPreview" in js
    assert 'graphEl.on("plotly_hover", (event) => {' in js
    assert "showClassSplitGraphHoverPreview(event.event, previewUrl, point)" in js
    assert "showClassSplitDatasetHoverPreview(event.event, previewUrl" in js
    assert "dataset_image_value_score" in js
    assert "scheduleAnnotationDiversityMetricRefresh();" in js
    assert "Projection neighbors" in js
    assert "Projection metric" in js
    assert "Projection spread" in js
    assert "Projection trustworthiness" in js
    assert "Projection class separation" in js
    assert "Projection silhouette" in js
    assert "Projection separation score" in js
    assert "Projection overview score" in js
    assert "Projection recommendation" in js
    assert "Suggested graph view (layout fidelity; likely-wrong ranking is unchanged)" in js
    assert 'if (value == null || value === "")' in js
    assert "projectionTrustworthiness !== null" in js
    assert "Graph projection" in js
    assert "Within-filter PCA" in js
    assert "Size-axis check" in js
    assert "Crop cache" in js
    assert "Embedding cache" in js
    assert '["Aggregation",' not in js
    assert '["SALAD head",' not in js
    assert "Hold Shift over this tab to switch the graph" not in js
    assert "function isClassSplitGraphPointerActive" in js
    assert "function getClassSplitGraphDragMode" in js
    assert "function setClassSplitGraphDragMode" in js
    assert "function toggleClassSplitGraphPanMode" in js
    assert 'event.key !== "f"' in js
    assert "handleClassSplitGraphPanShortcut(event)" in js
    assert "function panClassSplitPlotWithWheel" not in js
    assert 'dragmode: String(classSplitElements.dragMode?.value || "lasso")' in js
    assert "scrollZoom: true" in js
    assert "__classSplitShiftWheelGuard" not in js
    assert "function suppressClassSplitShiftWheel" in js
    assert "__classSplitShiftWheelSuppressor" in js
    assert 'graphEl.addEventListener("wheel", graphEl.__classSplitShiftWheelSuppressor, { passive: false, capture: true })' in js
    assert "function rememberClassSplitSelectionFromPlot" in js
    assert "function changeClassSplitSelectedPointsClass" in js
    assert "function markClassSplitWrongCandidateCorrect" in js
    assert "async function saveClassSplitReviewDisposition" in js
    assert "/review_disposition" in js
    assert "review_disposition_api_version" in js
    assert "Click to reconnect durable review actions." in js
    assert "Restart the backend to load verified durable review actions." in js
    assert 'disposition: safeDisposition' in js
    assert "const requestKey = `${jobId}:${safeId}`;" in js
    assert "reviewDispositionRetryTokens: new Map()" in js
    assert "classSplitState.reviewDispositionRetryTokens.get(" in js
    assert "classSplitState.reviewDispositionRetryTokens.delete(retryTokenKey);" in js
    assert "training_capture_requested" in js
    assert "String(classSplitState.currentJobId || \"\").trim() !== saved.jobId" in js
    assert "function undoLastClassSplitReviewDisposition" in js
    assert "function restoreClassSplitReviewDisposition" in js
    assert "function renderClassSplitReviewedList" in js
    assert "function deleteClassSplitReviewHistory" in js
    assert "function getClassSplitReviewHistoryPoints" in js
    assert "reviewedPointsById: new Map()" in js
    assert "Array.from(classSplitState.reviewedPointsById.values())" in js
    assert "right.reviewedAt - left.reviewedAt" in js
    assert "Changed class: ${point.human_review_before_class" in js
    assert "annotationMutationDisposition" in js
    assert "A saved class change cannot be undone from Review history" in js
    assert 'saveClassSplitReviewDisposition(safeId, "clear", {' in js
    assert "class-analysis-review-disposition-v3" in js
    assert "CLASS_SPLIT_REVIEW_OBJECT_KEY_PATTERN" in js
    assert "classSplitReviewDispositionClearPrecondition" in js
    assert 'id="classSplitWrongUndoReview"' in html
    assert 'id="classSplitReviewedPanel"' in html
    assert 'id="classSplitReviewHistoryDelete"' in html
    assert 'id="classSplitPendingReviewRecovery"' in html
    assert 'id="classSplitPendingReviewRetry"' in html
    assert 'id="classSplitPendingReviewDiscard"' in html
    assert "Review history (0)" in html
    assert html.index('id="classSplitWrongList"') < html.index('id="classSplitReviewedPanel"')
    assert "Saved choices survive restarts" in html
    assert "Local bbox-edit history stays on this page" in html
    assert "/review_history/delete" in js
    assert "class-analysis-review-history-delete-v1" in js
    assert "review_history_delete_api_version" in js
    assert "function markClassSplitLocalReviewHistoryExported" in js
    assert js.count("markClassSplitLocalReviewHistoryExported(exportResult);") == 2
    assert "const pendingReviewAction = Array.from(" in js
    assert "includeReconciliation" in js
    assert "scheduleClassSplitPendingTrainingCommitDrain();" in js
    assert "pendingReviewDispositionCommits: new Map()" in js
    assert "function drainClassSplitPendingReviewCommits" in js
    assert "function getClassSplitUnverifiedPendingReviewCommits" in js
    assert "function discardClassSplitUnverifiedPendingReviewCommits" in js
    assert "Discarding removes only the local retry intent" in js
    assert "fetchClassSplitReviewRequest(" in js
    assert "CLASS_SPLIT_REVIEW_REQUEST_TIMEOUT_MS = 12000" in js
    assert "scheduleClassSplitPendingReviewCommitDrain();" in js
    assert "queueClassSplitPendingReviewCommit({" in js
    assert '"local_workspace_exported"' in js
    assert 'data-action="restore-review"' in js
    assert 'point.human_review_persistence = "annotation_committed_review_unsaved"' in js
    assert '"annotation_committed_review_unsaved",' in js
    assert "Skipped object and saved that choice for future analyses." in js
    assert "Confirmed current class and saved it for future analyses." in js
    assert "point.wrong_class_suspicion = 0" not in js
    assert "const focusPromise = window.Plotly.relayout" in js
    assert "focusPromise\n            .then(() => {" in js
    assert "return window.Plotly.react(graphEl, traces, layout, config).then(async () => {" in js
    assert "const plotRender = renderClassSplitPlot();" in js
    assert "Promise.resolve(plotRender).then(() => {" in js
    assert "classSplitState.selectedPointId === safeId" in js
    assert 'focusPlot\n            && classSplitElements.displayMode' in js
    assert 'String(classSplitElements.displayMode.value || "all") === "wrong_only"' in js
    assert "!classSplitPointIsPrimaryCandidate(point)" in js
    assert "Suggested by neighbors: ${escapeHtml(point.suggested_neighbor_class)}" in js
    assert 'classSplitElements.filterClass.addEventListener("input", handleFilterClassChange);' in js
    assert 'classSplitElements.filterClass.addEventListener("change", handleFilterClassChange);' in js
    assert "classSplitState.selectedClusterId = \"\";\n                classSplitState.wrongQueueIds = [];" in js
    assert 'classSplitElements.displayMode.addEventListener("change", () => {' in js
    assert "refreshClassSplitFilteredReviewUi();" in js
    assert "if (filterChanged) {\n            renderClassSplitBulkPanel();\n            renderClassSplitClusterList();" in js
    assert "function showClassSplitHoverCard" not in js
    assert "cropPreview.naturalWidth" in js
    assert "Math.min(shellWidth / naturalWidth, shellHeight / naturalHeight)" in js
    assert "cropPreview.style.width = `${Math.max(1, Math.round(naturalWidth * nextScale))}px`" in js
    assert "cropPreview.style.height = `${Math.max(1, Math.round(naturalHeight * nextScale))}px`" in js
    assert "Math.max(0.2, Math.min(16, cropZoom * factor))" in js
    assert "new ResizeObserver(updateCropFitScale)" in js
    assert 'if (/^https?:\\/\\//i.test(String(thumbPath))) {' in js
    assert "transition: width 0.08s ease-out, height 0.08s ease-out;" in css
    assert "function focusClassSplitPlotOnPoint" in js
    assert "classSplitElements.bulkPanel" in js
    assert "panClassSplitPlotWithWheel" not in js
    assert "Confirm current class" in js
    assert "Skip" in js
    assert 'data-action="skip-wrong"' in js
    assert "function skipClassSplitWrongCandidate" in js
    assert "function discardFirstClassSplitWrongCandidates" in js
    assert "function getClassSplitWrongReviewOrder" in js
    assert "function getClassSplitWrongDiscardCount" in js
    assert 'id="classSplitWrongDiscardCount" min="1" max="12" step="1" value="12"' in html
    assert "Dismiss only cards currently shown in this 12-item queue" in html
    review_order_start = js.index("function getClassSplitWrongReviewOrder")
    review_order_end = js.index("function getClassSplitWrongDiscardCount", review_order_start)
    review_order = js[review_order_start:review_order_end]
    assert ".map((pointId) => candidateById.get" in review_order
    assert "ordered.push" not in review_order
    assert "Discarded ${discardedIds.length} ${discardLabel} vignette" in js
    assert 'getClassSplitVignetteCategory() === "review_queue"' in js
    assert "Objects with overlapping boxes" in html
    assert "classSplitOverlapPairMode" in html
    assert "is_close_overlap_candidate" in js
    assert "classSplitPointMatchesOverlapPairFilter" in js
    assert "close_overlap_candidates" in _read("localinferenceapi.py")
    assert "function reconcileClassSplitWrongQueue" in js
    assert "function getClassSplitRawCandidates" in js
    assert "function getClassSplitRefinementStatus" in js
    assert "function getClassSplitPrimaryCandidates" in js
    assert "function getClassSplitImmutableVignetteCounts" in js
    assert 'summaryRefinement.enabled === false || status === "disabled"' in js
    assert "classSplitResultHasRefinement(result)" in js
    assert "&& Array.isArray(result.refinement_candidates)" in js
    assert '["failed", "partial", "cancelled"].includes(refinementStatus)' in js
    assert "function loadClassSplitRefinementPreview" in js
    assert "function clearClassSplitRefinementPreviewCache" in js
    assert 'data-disclosure-key="refinement-${escapeHtml(String(classSplitState.currentJobId || "job"))}-${escapeHtml(pointId)}"' in js
    assert "/refinement_preview" in js
    assert 'link.target = "_blank"' in js
    assert 'link.rel = "noopener"' in js
    assert 'class-split-refinement-evidence__open' in js
    assert ".class-split-refinement-evidence__open" in css
    assert "function getClassSplitWideContextUrl" in js
    assert "function bindClassSplitVignetteWideContextPreviews" in js
    assert "?context=wide" in js
    assert ".class-split-graph-hover-preview.is-wide-context" in css
    assert "function shuffleClassSplitWrongQueue" in js
    assert "const cap = 12;" in js
    assert "Reassign" in js
    assert "Switch class to ${suggestedClass}" in js
    assert ">Choose class</option>" in js
    assert "Review overlapping boxes with VLM" in js
    assert "dual_bbox_action" in js
    assert "Dual-box conflict" in js
    assert "function getClassSplitContextCropUrl" in js
    assert "const maxPreviewDim = 1400;" in js
    assert 'alt="Object context crop"' in js
    assert "point.cluster_id = null;" in js
    assert "imageKeys: activeKeys" in js
    assert "imageKeys," in js
    assert "const finalMessage = `Changed class to ${targetClass} · ${saveStatus}" in js
    assert "Save pending; use Save labels if it does not clear." in js
    assert "Changed class to ${targetClass}; rerunning analysis." not in js
    assert "function drawClassSplitInstancePulse" in js
    assert "startClassSplitInstancePulse(match.bbox" in js
    assert "function getClassSplitServerSourceHandle" in js
    assert "function getClassSplitServerAnalysisSourceHandle" in js
    assert "classSplitElements.uploadDatasetName" not in js
    assert "CLASS_SPLIT_ACTIVE_WORKSPACE_UPLOADS_STORAGE_KEY" not in js
    assert "Active workspace upload did not return a backend dataset." not in js
    assert "function uploadClassSplitActiveWorkspaceSource" not in js
    assert "function buildClassSplitActiveWorkspaceUploadPlan" not in js
    assert 'return getClassSplitServerSourceHandle();' in js
    assert "function buildClassSplitActiveWorkspaceSnapshot" in js
    assert "function postClassSplitActiveWorkspaceChunked" in js
    assert "class_analysis/jobs/active_workspace/upload_session/start" in js
    assert "class_analysis/jobs/active_workspace/upload_session/${encodeURIComponent(sessionId)}/batch" in js
    assert "class_analysis/jobs/active_workspace/upload_session/${encodeURIComponent(sessionId)}/finalize" in js
    assert "class_analysis/jobs/active_workspace/upload_session/${encodeURIComponent(sessionId)}/cancel" in js
    assert "publicClassSplitActiveWorkspaceRows" in js
    assert "frontend_image_key" in js
    assert 'transport: "chunked"' not in js[js.index("function buildClassSplitActiveWorkspaceSnapshot"):js.index("function getClassSplitServerSourceHandle")]
    assert "const uploadAbortController = new AbortController();" in js
    assert "signal: uploadAbortController.signal" in js
    assert "function postClassSplitActiveWorkspaceForm" in js
    assert "xhr.upload.onprogress" in js
    assert "Snapshot-uploading ${Math.round((event.loaded / event.total) * 100)}%" in js
    assert "Snapshot-packaging ${processed}/${totalImages} active images" in js
    assert "const includeRefinementContext = (" in js
    assert "scope === \"selected_class\"" in js
    assert "&& Boolean(request.refine_outliers)" in js
    assert "const allLabelLines = getClassSplitActiveLabelLines(imageKey);" in js
    assert "const labelLines = includeRefinementContext" in js
    assert "query_object_count: queryObjectCount" in js
    assert "context_object_count: contextObjectCount" in js
    assert "activeUploadSessionId" in js
    assert 'fetch(`${API_ROOT}/datasets/upload_session/${encodeURIComponent(uploadSessionId)}/cancel`' in js
    assert 'source_mode: sourceHandle.sourceMode' in js
    assert "payload.dataset_id = sourceHandle.datasetId" in js
    assert 'fetch(`${API_ROOT}/class_analysis/jobs`, {' in js
    assert 'headers: { "Content-Type": "application/json" }' in js
    assert 'xhr.open("POST", `${API_ROOT}/class_analysis/jobs/active_workspace`)' in js
    assert "window.Plotly.react" in js
    assert "function jumpToClassSplitPoint" in js
    assert "setActiveTab(TAB_LABELING)" in js
    assert "See instance" in js
    assert 'data-action="jump-instance"' in js
    assert 'listEl.querySelectorAll(\'[data-action="jump-instance"]\')' in js
    assert "Data Quality Explorer vignette jump failed" in js
    assert "function changeClassSplitPointClass" in js
    assert "const previousClass = String(match.className || graphClass).trim();" in js
    assert "captureAnnotationDirtyStateForImage(imageKey)" in js
    assert "async function ensureClassSplitSnapshotClean" in js
    assert "captureCurrentAnnotationDirtyState();" in js
    assert 'await ensureClassSplitSnapshotClean("Data Quality Explorer analysis")' in js
    assert "Data Quality Explorer analysis is running. The graph will appear when the backend finishes embedding and projection." in js
    assert "startClassSplitAnalysis({ reuseLast: true })" in js
    assert "classSplitState.pollRequestId += 1;" in js[js.index("async function startClassSplitAnalysis"):js.index("function stopClassSplitPoll")]
    assert "initClassSplitExplorer();" in js
    assert "function renderClassSplitQwenReviewModelOptions" in js
    assert "metadata.inference_supported === false || metadata.vision_inference_supported === false" in js
    assert "metadata.display_name || metadata.label || metadata.name || entry.label" in js
    assert "max_files=float(\"inf\")" in router
    assert "max_part_size=512 * 1024 * 1024" in router


def test_data_quality_explorer_performance_and_state_recovery_contract():
    html = _html()
    css = _css()
    js = _js()

    assert "Data Quality Explorer" in html
    assert 'src="ybat.js?v=20260811c"' in html
    assert 'href="ybat.css?v=20260808a"' in html
    assert 'id="classSplitLimitPlotPoints"' not in html
    assert 'type: "scattergl"' in js
    assert "showing every point; this large graph may use substantial browser memory" in js
    assert 'id="classSplitRecipeExplanation"' in html
    assert "What this preset actually uses" in html
    assert "2,944-D: DINOv3 tight/context 2,048-D + SAM3 mask 256-D + balanced SALAD 640-D." in js
    assert "Percentages are similarity weights, not shares of dimensions." in js
    assert "class_split_embedding_presets_preview.svg" not in html
    assert "DATA_QUALITY_SESSION_KEY" in js
    assert "persistDataQualityExplorerSession" in js
    assert "restoreDataQualityExplorerSession" in js
    assert "active_workspace/snapshots/${encodeURIComponent(classSplitState.snapshotId)}" in js
    assert "snapshotSignature" in js
    assert "classSplitSnapshotSourceIdentity" in js
    assert "classSplitState.snapshotSignature !== workspaceSourceSignature" in js
    assert "/qwen_reviews" in js
    assert "restoreClassSplitQwenReviews" in js
    assert "Connection interrupted; retrying analysis status" in js
    assert "Connection interrupted; retrying VLM review status" in js
    assert "runtime.objects_per_second" in js
    assert "memory.backend_rss_bytes" in js
    assert "memory.worker_rss_bytes" in js
    assert "installTatorBackGestureGuard" in js
    assert 'window.addEventListener("wheel"' in js
    assert "event.preventDefault()" in js
    assert "history.pushState" in js
    assert "overscroll-behavior-x: none" in css


def test_data_quality_explorer_replacement_and_async_identity_contract():
    js = _js()
    start_at = js.index("async function startClassSplitAnalysis")
    start = js[start_at:js.index("function stopClassSplitPoll", start_at)]
    poll = _extract_js_function(js, "pollClassSplitJob")
    apply_result = _extract_js_function(js, "applyClassSplitResultPayload")
    projection = _extract_js_function(js, "ensureClassSplitProjectionCoordinates")
    qwen_start = _extract_js_function(js, "startClassSplitQwenReview")
    qwen_poll = _extract_js_function(js, "scheduleClassSplitQwenReviewPoll")
    controls_at = js.index("function refreshClassSplitControls")
    controls = js[controls_at:js.index("function buildClassSplitRequest", controls_at)]
    bulk_relabel = _extract_js_function(js, "changeClassSplitSelectedPointsClass")
    point_relabel_at = js.index("async function changeClassSplitPointClass")
    point_relabel = js[
        point_relabel_at:js.index("function initClassSplitExplorer", point_relabel_at)
    ]
    snapshot_state = _extract_js_function(js, "snapshotClassSplitReviewState")
    restore_state = _extract_js_function(js, "restoreClassSplitReviewState")
    restore_qwen = _extract_js_function(js, "restoreClassSplitQwenReviews")
    startup_ready = _extract_js_function(js, "classSplitStartupCleanupIsReady")
    wait_startup_at = js.index("async function waitForClassSplitStartupReady")
    wait_startup = js[
        wait_startup_at:js.index("function snapshotClassSplitReviewState", wait_startup_at)
    ]

    snapshot_at = start.index("previousReviewState = snapshotClassSplitReviewState();")
    retain_at = start.index(
        "classSplitState.pendingPreviousReviewState = previousReviewState;",
        snapshot_at,
    )
    clear_at = start.index("classSplitState.result = null;", retain_at)
    assert snapshot_at < retain_at < clear_at
    assert "restorePendingClassSplitReviewState()" in start
    assert "if (reuseLast)" in start
    assert "request.force_new_run = true;" in start
    assert "delete request.force_new_run;" in start
    assert "delete classSplitState.lastRequest.force_new_run;" in start
    assert "classSplitState.pendingPreviousReviewState = null;" in apply_result
    assert poll.count("restorePendingClassSplitReviewState()") >= 3
    assert poll.index("if (requestId !== classSplitState.pollRequestId)") < poll.index(
        "if (!resp.ok)"
    )
    assert "Replacement analysis is unavailable; restored the previous completed analysis." in poll
    assert "Cancelled; restored the previous completed analysis." in poll
    assert "qwenReviewJobs: new Map(classSplitState.qwenReviewJobs || [])" in snapshot_state
    assert "classSplitState.qwenReviewJobs = new Map(snapshot.qwenReviewJobs || [])" in restore_state
    assert "resumeClassSplitQwenReviewPolls();" in restore_state
    assert "clearClassSplitQwenReviewPolls" not in restore_qwen
    assert "existingId !== incomingId" in restore_qwen
    assert "cancelActiveClassSplitQwenReviewsForReplacement();" in start
    assert start.index("cancelActiveClassSplitQwenReviewsForReplacement();") < snapshot_at
    assert "classSplitMutationIsBusy({" in start
    assert "includeDirtyAnnotations: true" in start
    assert "const mutationBusy = classSplitMutationIsBusy();" in controls
    assert 'if (!cleanup || typeof cleanup !== "object")' in startup_ready
    assert "return false;" in startup_ready
    assert '"deleting"' in startup_ready
    assert "Boolean(cleanup.ready_at)" in startup_ready
    assert "await waitForClassSplitStartupReady();" in start
    assert "classSplitState.startupOperation = {" in start
    assert start.index("classSplitState.startupOperation = {") < start.index(
        "await preflightClassSplitRequest(request);"
    )
    mutation_busy = _extract_js_function(js, "classSplitMutationIsBusy")
    assert "startupOperationBusy" in mutation_busy
    assert "ignoreStartupOperationToken" in mutation_busy
    assert "ignoreStartupOperationToken: startupOperationToken" in _extract_js_function(
        js, "buildClassSplitRequest"
    )
    assert "buildClassSplitRequest({ startupOperationToken: startupToken })" in start
    assert "Backend did not report Data Quality Explorer startup readiness" in wait_startup
    assert "cachedCleanup" not in wait_startup
    assert 'cache: "no-store"' in wait_startup
    assert wait_startup.index("fetch(") < wait_startup.index(
        "classSplitStartupCleanupIsReady(cleanup)"
    )

    for request in (projection, qwen_start, qwen_poll):
        assert "classSplitAsyncRequestIsCurrent(" in request
        assert "AbortController" in request or "controller.signal" in request
    assert "signal: controller.signal" in projection
    assert "signal: controller.signal" in qwen_start
    assert "signal: controller.signal" in qwen_poll
    assert "mutationBusy" in controls
    in_flight_at = bulk_relabel.index("classSplitState.relabelInFlight = true;")
    disable_at = bulk_relabel.index("refreshClassSplitControls();", in_flight_at)
    clear_in_flight_at = bulk_relabel.index(
        "classSplitState.relabelInFlight = false;",
        disable_at,
    )
    enable_at = bulk_relabel.index("refreshClassSplitControls();", clear_in_flight_at)
    assert in_flight_at < disable_at < clear_in_flight_at < enable_at
    assert "refreshClassSplitControls();" not in point_relabel
    assert point_relabel.index("renderClassSplitWrongList();") < point_relabel.index(
        "await captureClassSplitTrainingAction("
    )


def test_review_history_frontend_receipts_concurrency_and_selective_export_contract():
    js = _js()
    validator_start = js.index(
        "function validateClassSplitReviewDispositionReceipt"
    )
    validator = js[
        validator_start:js.index(
            "function scheduleClassSplitReviewDispositionHydration",
            validator_start,
        )
    ]
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const CLASS_SPLIT_REVIEW_DISPOSITION_RECEIPT_SCHEMA = 'class-analysis-review-disposition-v3';",
            "const CLASS_SPLIT_REVIEW_REVISION_PATTERN = /^rdr1_[0-9a-f]{32}$/;",
            "const CLASS_SPLIT_REVIEW_OBJECT_KEY_PATTERN = /^(cro|crp|crj)_[0-9a-f]{64}$/;",
            "const CLASS_SPLIT_REVIEW_DISPOSITIONS = ['confirm_current', 'skip', 'reassign_class', 'delete_bbox', 'delete_current_box', 'delete_overlapping_box', 'keep_both_boxes', 'unresolved'];",
            _extract_js_function(js, "classSplitReviewHistoryTimestampSeconds"),
            _extract_js_function(js, "classSplitCanonicalJson"),
            validator,
            "const record = {schema: CLASS_SPLIT_REVIEW_DISPOSITION_RECEIPT_SCHEMA, status: 'recorded', job_id: 'j', point_id: 'p', disposition: 'skip', client_action_id: 'client-123', review_object_key: 'cro_' + 'a'.repeat(64), training_capture_requested: false, human_reviewed_at: 200, human_review_revision: 'rdr1_' + '1'.repeat(32)};",
            "assert.strictEqual(validateClassSplitReviewDispositionReceipt(record, {jobId: 'j', pointId: 'p', disposition: 'skip', clientActionId: 'client-123'}), record);",
            "const deniedCapture = {...record, training_capture: {status: 'denied'}};",
            "assert.strictEqual(validateClassSplitReviewDispositionReceipt(deniedCapture, {jobId: 'j', pointId: 'p', disposition: 'skip', clientActionId: 'client-123', captureTrainingData: true}), deniedCapture);",
            "assert.throws(() => validateClassSplitReviewDispositionReceipt({...record, training_capture_requested: true}, {jobId: 'j', pointId: 'p', disposition: 'skip', clientActionId: 'client-123'}), /invalid review-disposition receipt/);",
            "assert.throws(() => validateClassSplitReviewDispositionReceipt({...record, review_object_key: 'bad'}, {jobId: 'j', pointId: 'p', disposition: 'skip', clientActionId: 'client-123'}), /invalid review-disposition receipt/);",
            "const reassigned = {...record, disposition: 'reassign_class', review_object_key: 'crj_' + 'b'.repeat(64), target_class: 'Building'};",
            "assert.strictEqual(validateClassSplitReviewDispositionReceipt(reassigned, {jobId: 'j', pointId: 'p', disposition: 'reassign_class', clientActionId: 'client-123', expectedTargetClass: 'Building'}), reassigned);",
            "assert.throws(() => validateClassSplitReviewDispositionReceipt({...reassigned, target_class: 'Person'}, {jobId: 'j', pointId: 'p', disposition: 'reassign_class', clientActionId: 'client-123', expectedTargetClass: 'Building'}), /invalid review-disposition receipt/);",
            "assert.throws(() => validateClassSplitReviewDispositionReceipt({...record, review_object_key: 'crj_' + 'c'.repeat(64)}, {jobId: 'j', pointId: 'p', disposition: 'skip', clientActionId: 'client-123'}), /invalid review-disposition receipt/);",
            "const annotationTarget = {source_mode: 'linked', source_id: 'dataset', split: 'train', image_relpath: 'image.jpg'};",
            "const committedRevision = 'alr1_' + '4'.repeat(64);",
            "const sourceIdentity = 'asi1_' + '5'.repeat(64);",
            "const alreadyAbsentAttestation = {schema: 'class-analysis-single-bbox-deletion-attestation-v1', committed: true, deletion_state: 'already_absent', analysis_job_id: 'j', point_id: 'p', review_object_key: record.review_object_key, annotation_target: annotationTarget, source_identity: sourceIdentity, before_revision: committedRevision, committed_revision: committedRevision, verification_method: 'exact_frozen_geometry_already_absent_v1', attestation_sha256: '6'.repeat(64)};",
            "const alreadyAbsent = {...record, status: 'already_absent', annotation_state: 'already_absent', disposition: 'delete_bbox', annotation_commit_attestation: alreadyAbsentAttestation};",
            "const alreadyAbsentExpected = {labelCommitStatus: 'already_absent', annotationTarget, beforeRevision: committedRevision, committedRevision, sourceIdentity};",
            "assert.strictEqual(validateClassSplitReviewDispositionReceipt(alreadyAbsent, {jobId: 'j', pointId: 'p', disposition: 'delete_bbox', clientActionId: 'client-123', expectedDeletionCommit: alreadyAbsentExpected}), alreadyAbsent);",
            "assert.throws(() => validateClassSplitReviewDispositionReceipt({...alreadyAbsent, status: 'recorded'}, {jobId: 'j', pointId: 'p', disposition: 'delete_bbox', clientActionId: 'client-123', expectedDeletionCommit: alreadyAbsentExpected}), /invalid saved-review receipt/);",
            "assert.throws(() => validateClassSplitReviewDispositionReceipt({...alreadyAbsent, annotation_commit_attestation: {...alreadyAbsentAttestation, verification_method: 'exact_single_label_transition_v2'}}, {jobId: 'j', pointId: 'p', disposition: 'delete_bbox', clientActionId: 'client-123', expectedDeletionCommit: alreadyAbsentExpected}), /did not prove/);",
            "const cleared = {schema: CLASS_SPLIT_REVIEW_DISPOSITION_RECEIPT_SCHEMA, status: 'cleared', job_id: 'j', point_id: 'p', disposition: 'clear', client_action_id: 'client-456', review_object_key: 'cro_' + 'a'.repeat(64), training_capture_requested: false, previous_disposition: 'skip'};",
            "assert.strictEqual(validateClassSplitReviewDispositionReceipt(cleared, {jobId: 'j', pointId: 'p', disposition: 'clear', clientActionId: 'client-456', previousDisposition: 'skip'}), cleared);",
            "const alreadyClear = {...cleared, status: 'already_clear', previous_disposition: ''};",
            "assert.strictEqual(validateClassSplitReviewDispositionReceipt(alreadyClear, {jobId: 'j', pointId: 'p', disposition: 'clear', clientActionId: 'client-456', previousDisposition: 'skip'}), alreadyClear);",
            "assert.throws(() => validateClassSplitReviewDispositionReceipt({...alreadyClear, previous_disposition: 'skip'}, {jobId: 'j', pointId: 'p', disposition: 'clear', clientActionId: 'client-456', previousDisposition: 'skip'}), /invalid review-clear receipt/);",
            "const exported = {point_id: 'p1', class_name: 'Person', frontend_image_key: 'good.png', human_review_persistence: 'local_workspace_pending_export'};",
            "const skipped = {point_id: 'p2', frontend_image_key: 'skipped.png', human_review_persistence: 'local_workspace_pending_export'};",
            "let renderCount = 0;",
            "let persistCount = 0;",
            "let drainScheduleCount = 0;",
            "const pendingCommit = {jobId: 'j', pointId: 'p1', afterClass: 'Person', labelPersisted: false};",
            "const classSplitState = {reviewedPointsById: new Map([['p1', exported], ['p2', skipped]]), pendingTrainingClassCommits: new Map([['pending', pendingCommit]]), currentJobId: 'j', analysisGeneration: 1, reviewDispositionInFlight: new Set()};",
            "function resolveClassSplitPointImageKey(point) { return point.frontend_image_key; }",
            "function getClassSplitPointById(pointId) { return classSplitState.reviewedPointsById.get(pointId); }",
            "function ensureClassSplitPointBboxMatch(point) { return {match: {className: point.class_name}}; }",
            "function persistDataQualityExplorerSession() { persistCount += 1; }",
            "function scheduleClassSplitPendingTrainingCommitDrain() { drainScheduleCount += 1; }",
            "function renderClassSplitReviewedList() { renderCount += 1; }",
            _extract_js_function(js, "markClassSplitLocalReviewHistoryExported"),
            "assert.strictEqual(markClassSplitLocalReviewHistoryExported({exportedLabelIdentities: [{image_key: 'good.png', label_name: 'good.txt'}]}), 1);",
            "assert.strictEqual(exported.human_review_persistence, 'local_workspace_exported');",
            "assert.strictEqual(skipped.human_review_persistence, 'local_workspace_pending_export');",
            "assert.strictEqual(pendingCommit.labelPersisted, true);",
            "assert.strictEqual(persistCount, 1);",
            "assert.strictEqual(drainScheduleCount, 1);",
            "assert.strictEqual(renderCount, 1);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_review_actions_are_job_scoped_conflict_safe_and_target_hydration_only():
    js = _js()
    hydration_at = js.index("function scheduleClassSplitReviewDispositionHydration")
    hydration = js[
        hydration_at:js.index(
            "async function saveClassSplitReviewDisposition",
            hydration_at,
        )
    ]
    acknowledged_at = js.index("function runClassSplitAcknowledgedAction")
    acknowledged = js[
        acknowledged_at:js.index(
            "function scheduleClassSplitBackgroundReviewRefresh",
            acknowledged_at,
        )
    ]
    single_delete = _extract_js_function(js, "deleteClassSplitPointBbox")
    dual_resolve = _extract_js_function(js, "resolveClassSplitDualBBox")
    point_relabel_at = js.index("async function changeClassSplitPointClass")
    point_relabel = js[
        point_relabel_at:js.index(
            "function initClassSplitExplorer",
            point_relabel_at,
        )
    ]
    review_save_at = js.index("async function saveClassSplitReviewDisposition")
    review_save = js[
        review_save_at:js.index(
            "async function skipClassSplitWrongCandidate",
            review_save_at,
        )
    ]
    capability_refresh = _extract_js_function(
        js,
        "ensureClassSplitReviewDispositionCapabilities",
    )
    review_receipt_at = js.index(
        "function validateClassSplitReviewDispositionReceipt"
    )
    review_receipt = js[
        review_receipt_at:review_save_at
    ]
    review_history_delete = _extract_js_function(
        js,
        "deleteClassSplitReviewHistory",
    )
    reviewed_list_render = _extract_js_function(
        js,
        "renderClassSplitReviewedList",
    )
    dual_commit_at = js.index(
        "async function commitClassSplitDualBBoxDeletionTransaction"
    )
    dual_commit = js[
        dual_commit_at:js.index(
            "function applyClassSplitDualBBoxTransactionLocally",
            dual_commit_at,
        )
    ]
    dual_receipt_at = js.index(
        "async function validateClassSplitDualBBoxDeletionReceipt"
    )
    dual_receipt = js[
        dual_receipt_at:dual_commit_at
    ]

    assert "reviewDispositionHydrationTargets" in hydration
    assert "hydrationTargets.forEach" in hydration
    assert "classSplitState.pointsById.forEach" not in hydration
    assert 'cache: "no-store"' in hydration
    assert "classSplitReviewPersistenceIsLocalOnly(point)" in hydration
    assert "classSplitReviewHistoryEntryMatchesSnapshot(" in hydration
    assert "target.clearPrecondition" in hydration
    assert "serverReviewedAt <= currentReviewedAt" in hydration

    assert "actionGeneration" in acknowledged
    assert "actionJobId" in acknowledged
    assert "classSplitAsyncRequestIsCurrent(" in acknowledged
    assert "actionReservationPointIds" in acknowledged
    assert "reviewActionPendingPointIds.has(pointId)" in acknowledged
    assert "reviewActionPendingPointIds.add(pointId)" in acknowledged
    assert "reviewActionPendingPointIds.delete(" in acknowledged
    assert "renderClassSplitInspector()" in acknowledged
    before_acknowledgement = acknowledged[
        :acknowledged.index(
            'button.classList.add("class-split-review-action", "is-acknowledged")'
        )
    ]
    assert "renderClassSplitWrongList()" not in before_acknowledgement
    assert "renderClassSplitInspector()" not in before_acknowledgement
    assert "stagedPendingEntries" in review_history_delete
    assert "browser recovery storage could not be updated safely" in review_history_delete
    assert "classSplitReviewHistoryCommitUnknown" in review_history_delete
    assert "classSplitMutationIsBusy({" in review_history_delete
    assert "includeReconciliation: true" in review_history_delete
    assert "includeHydration: true" in review_history_delete
    assert review_history_delete.index(
        "!persistDataQualityExplorerSession()"
    ) < review_history_delete.index("fetchClassSplitReviewRequest(")
    assert "classSplitMutationIsBusy({" in reviewed_list_render
    assert "sessionStorage.getItem(DATA_QUALITY_LIMIT_PLOT_SESSION_KEY)" not in js
    assert "sessionStorage.setItem(DATA_QUALITY_LIMIT_PLOT_SESSION_KEY" not in js
    assert "classSplitReviewHistoryNeedsDurableClear(point)" in single_delete
    assert "humanReviewRevision" in single_delete
    assert "queueClassSplitPendingReviewCommit(" in single_delete
    assert "drainClassSplitPendingReviewCommits()" in single_delete
    assert "annotationTarget: operation.annotationTarget" in single_delete
    assert "beforeRevision: operation.beforeAnnotationRevision" in single_delete
    assert '"delete_bbox"' in single_delete
    assert '"annotation_committed_review_unsaved"' in single_delete
    assert "classSplitPointReviewMutationBlocked(safePointId)" in single_delete
    assert single_delete.count("classSplitState.relabelInFlight = true;") == 1
    assert "waitForClassSplitAnnotationImageSave(imageKey)" in single_delete
    assert single_delete.index("classSplitState.relabelInFlight = false;") < single_delete.index(
        'flushAnnotationSnapshot({ manual: false })'
    )
    local_delete_branch = single_delete[
        single_delete.index('if (deletionState.mode === "local_workspace")'):
        single_delete.index(
            'enqueueTaskNotice("BBox deleted · saving labels in background …"',
        )
    ]
    assert "renderClassSplitReviewedList();" in local_delete_branch
    assert "classSplitReviewHistoryNeedsDurableClear" in dual_resolve
    assert "classSplitDualBBoxOperationIsCurrent(operation)" in dual_resolve
    assert "applyClassSplitOptimisticDualBBoxReview(contract)" in dual_resolve
    assert "restoreClassSplitOptimisticDualBBoxReview(" in dual_resolve
    assert "jobId: operation.jobId" in dual_resolve
    assert "deletionBarrierToken" in dual_resolve
    assert "annotationSourceState.readOnly = true" in dual_resolve

    assert "classSplitPointReviewMutationBlocked(pointId)" in point_relabel
    durable_queue_at = point_relabel.index(
        "queueClassSplitPendingReviewCommit("
    )
    label_save_at = point_relabel.index(
        "flushAnnotationSnapshot({ manual: false })"
    )
    durable_drain_at = point_relabel.index(
        "drainClassSplitPendingReviewCommits()",
        label_save_at,
    )
    optional_capture_at = point_relabel.index(
        "captureClassSplitTrainingAction(",
        durable_drain_at,
    )
    assert durable_queue_at < label_save_at < durable_drain_at < optional_capture_at
    assert "waitForClassSplitAnnotationImageSave(imageKey)" in point_relabel
    assert 'point.human_review_disposition = "reassign_class";' in point_relabel
    assert 'point.human_review_persistence = "annotation_committed_review_unsaved";' in point_relabel
    assert "await ensureClassSplitReviewDispositionCapabilities(safeDisposition)" in review_save
    assert "await loadClassSplitCapabilities()" in capability_refresh
    assert "review_class_reassignment_api_version" in capability_refresh
    assert "review_single_bbox_deletion_api_version" in capability_refresh
    assert "requestPayload.annotation_before_revision = beforeRevision" in review_save
    assert "requestPayload.annotation_commit_revision = committedRevision" in review_save
    assert "requestPayload.annotation_source_identity = sourceIdentity" in review_save
    assert '"exact_single_label_transition_v2"' in review_receipt
    assert "attestation.deleted_label_line_sha256" in review_receipt
    assert "attestation.deleted_label_line_index" in review_receipt
    assert "expectedTargetClass: targetClass" in review_save
    assert "options.captureTrainingData === undefined" in review_save
    assert 'origin: String(options.origin || "desktop")' in review_save

    assert "requestDispatched = true" in review_save
    assert "reviewDispositionCommitUnknown" in review_save
    assert "reviewDispositionReconciliationPointIds.add(" in review_save
    assert "scheduleClassSplitReviewDispositionHydration(jobId, [safeId])" in review_save
    assert "validateClassSplitDualBBoxDeletionReceipt(" in dual_commit
    assert "validateClassSplitReviewDispositionReceipt(" in dual_commit
    assert "review_client_action_id" in dual_commit
    assert "savedReview" in dual_commit
    assert "annotationTransactionCommitted = null" in dual_commit
    assert "commitStateUnknown" in dual_commit
    for bound_field in (
        "analysis_job_id",
        "point_id",
        "pair_review_key",
        "deleted_point_id",
        "surviving_point_id",
        "source_identity",
        "before_revision",
        "committed_revision",
        "operation_id",
        "attestation_sha256",
    ):
        assert bound_field in dual_receipt


def test_pending_review_queue_removal_rolls_back_exact_entry_when_storage_fails():
    js = _js()
    start = js.index("function removeClassSplitPendingReviewCommit")
    end = js.index("function classSplitPendingReviewCommitCountForJob", start)
    remove_fn = js[start:end]
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const original = {queueKey: 'job:p0:delete_bbox', pointId: 'p0'};",
            "const classSplitState = {pendingReviewDispositionCommits: new Map([[original.queueKey, original]])};",
            "let persistOk = false;",
            "function persistDataQualityExplorerSession() { return persistOk; }",
            remove_fn,
            "assert.strictEqual(removeClassSplitPendingReviewCommit(original.queueKey), false);",
            "assert.strictEqual(classSplitState.pendingReviewDispositionCommits.get(original.queueKey), original);",
            "assert.strictEqual(removeClassSplitPendingReviewCommit(original.queueKey, {rollbackOnPersistenceFailure: false}), true);",
            "assert.strictEqual(classSplitState.pendingReviewDispositionCommits.has(original.queueKey), false);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_already_absent_bbox_reconciliation_saves_existing_manual_edit_once():
    js = _js()
    reconcile = _extract_js_function(
        js,
        "reconcileClassSplitAlreadyAbsentBbox",
    ).replace(
        "function reconcileClassSplitAlreadyAbsentBbox",
        "async function reconcileClassSplitAlreadyAbsentBbox",
        1,
    )
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const revision = 'alr1_' + '4'.repeat(64);",
            "const sourceIdentity = 'asi1_' + '5'.repeat(64);",
            "const point = {point_id: 'p0'};",
            "const operation = {jobId: 'job', generation: 1, pointId: 'p0', point, imageKey: 'train:image.jpg', annotationTarget: {source_mode: 'linked', source_id: 'dataset', split: 'train', image_relpath: 'image.jpg'}, clientActionId: 'client-action'};",
            "const annotationSourceState = {dirtyRecordsByKey: new Map(), imageRowsByKey: new Map([[operation.imageKey, {annotation_record_revision: 'alr1_' + '3'.repeat(64), annotation_source_identity: sourceIdentity}]])};",
            "const classSplitState = {reviewDispositionReconciliationPointIds: new Set()};",
            "const notices = []; let flushes = 0; let savedOptions = null; let applied = 0; let cleared = 0;",
            "function classSplitReviewToastKey() { return 'toast'; }",
            "async function ensureClassSplitAlreadyAbsentDeletionCapability() {}",
            "function applyClassSplitOptimisticReview() { return {pointId: 'p0'}; }",
            "function enqueueTaskNotice(message) { notices.push(message); }",
            "function captureAnnotationDirtyStateForImage(imageKey) { annotationSourceState.dirtyRecordsByKey.set(imageKey, {}); }",
            "async function flushAnnotationSnapshot() { flushes += 1; annotationSourceState.dirtyRecordsByKey.delete(operation.imageKey); annotationSourceState.imageRowsByKey.get(operation.imageKey).annotation_record_revision = revision; return true; }",
            "async function waitForClassSplitAnnotationImageSave() { throw new Error('save completed synchronously'); }",
            "async function saveClassSplitReviewDisposition(_pointId, _disposition, options) { savedOptions = options; return {jobId: 'job', payload: {status: 'already_absent'}}; }",
            "function classSplitAsyncRequestIsCurrent() { return true; }",
            "function getClassSplitPointById() { return point; }",
            "function clearClassSplitPendingReviewCommitsForPoint() { cleared += 1; }",
            "function applyClassSplitAlreadyAbsentReview() { applied += 1; }",
            "function persistDataQualityExplorerSession() { return true; }",
            "function setClassSplitJobStatus() {}",
            "function scheduleClassSplitBackgroundReviewRefresh() {}",
            "function restoreClassSplitOptimisticReview() { throw new Error('must not restore'); }",
            "function renderClassSplitPendingReviewRecovery() {}",
            "function renderClassSplitReviewedList() {}",
            "const CLASS_SPLIT_ANNOTATION_REVISION_PATTERN = /^alr1_[0-9a-f]{64}$/;",
            "const CLASS_SPLIT_ANNOTATION_SOURCE_IDENTITY_PATTERN = /^asi1_[0-9a-f]{64}$/;",
            reconcile,
            "(async () => {",
            "  await reconcileClassSplitAlreadyAbsentBbox(operation, {mode: 'server'});",
            "  assert.strictEqual(flushes, 1);",
            "  assert.strictEqual(savedOptions.labelCommitStatus, 'already_absent');",
            "  assert.strictEqual(savedOptions.beforeRevision, revision);",
            "  assert.strictEqual(savedOptions.annotationCommitRevision, revision);",
            "  assert.strictEqual(savedOptions.annotationSourceIdentity, sourceIdentity);",
            "  assert.strictEqual(savedOptions.clientActionId, 'client-action');",
            "  assert.strictEqual(applied, 1);",
            "  assert.strictEqual(cleared, 1);",
            "  assert.ok(notices.some((message) => message.includes('already deleted')));",
            "})().catch((error) => { console.error(error); process.exit(1); });",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_main_view_bbox_delete_paths_capture_annotation_dirty_state_immediately():
    js = _js()
    selected_delete = _extract_js_function(js, "deleteSelectedBboxes")
    shortcuts = js[
        js.index("function deleteSelectedOrCurrentBboxShortcut"):
        js.index("function shouldHandleShortcut", js.index("function deleteSelectedOrCurrentBboxShortcut"))
    ]

    assert "captureAnnotationDirtyStateForImage(currentImage.name);" in selected_delete
    assert shortcuts.count(
        "captureAnnotationDirtyStateForImage(currentImage.name);"
    ) == 2


def test_pending_review_mutation_state_rejects_a_different_annotation_source():
    js = _js()
    mutation_state = _extract_js_function(
        js,
        "classSplitPendingReviewMutationState",
    )
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const classSplitState = {currentJobId: 'job', pendingReviewDispositionCommits: new Map()};",
            "const annotationSourceState = {dirtyRecordsByKey: new Set(), imageRowsByKey: new Map()};",
            "function isAnnotationDatasetModeActive() { return true; }",
            "function getClassSplitPointById() { return {point_id: 'p0'}; }",
            "function getClassSplitAnnotationTarget() { return {source_mode: 'linked', source_id: 'other', split: 'train', image_relpath: 'same.jpg'}; }",
            "function classSplitCanonicalJson(value) { return JSON.stringify(value, Object.keys(value || {}).sort()); }",
            "function findClassSplitExactGeometryMatches() { throw new Error('geometry must not run for another source'); }",
            "const CLASS_SPLIT_ANNOTATION_REVISION_PATTERN = /^alr1_[0-9a-f]{64}$/;",
            "const CLASS_SPLIT_ANNOTATION_SOURCE_IDENTITY_PATTERN = /^asi1_[0-9a-f]{64}$/;",
            mutation_state,
            "const state = classSplitPendingReviewMutationState({jobId: 'job', pointId: 'p0', imageKey: 'train:same.jpg', beforeRevision: 'alr1_' + '1'.repeat(64), annotationTarget: {source_mode: 'linked', source_id: 'expected', split: 'train', image_relpath: 'same.jpg'}});",
            "assert.strictEqual(state, 'source_mismatch');",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_annotation_load_reconciles_saved_unsaved_and_wrong_source_reviews():
    js = _js()
    assert (
        "startAnnotationTimers();\n"
        "            reconcileClassSplitPendingReviewCommitsAfterAnnotationLoad();"
    ) in js
    reconcile = _extract_js_function(
        js,
        "reconcileClassSplitPendingReviewCommitsAfterAnnotationLoad",
    )
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const entries = [",
            "  {queueKey: 'job:p0:delete_bbox', jobId: 'job', pointId: 'p0', state: 'not_committed'},",
            "  {queueKey: 'job:p1:delete_bbox', jobId: 'job', pointId: 'p1', state: 'committed'},",
            "  {queueKey: 'job:p2:delete_bbox', jobId: 'job', pointId: 'p2', state: 'source_mismatch'},",
            "];",
            "const classSplitState = {currentJobId: 'job', pendingReviewDispositionCommits: new Map(entries.map((entry) => [entry.queueKey, entry]))};",
            "let scheduled = 0; let rendered = 0; const notices = []; const applied = [];",
            "function isAnnotationDatasetModeActive() { return true; }",
            "function classSplitPendingReviewMutationState(entry) { return entry.state; }",
            "function removeClassSplitPendingReviewCommit(key) { return classSplitState.pendingReviewDispositionCommits.delete(key); }",
            "function applyClassSplitCommittedPendingReview(entry) { applied.push(entry.pointId); return {}; }",
            "function persistDataQualityExplorerSession() { return true; }",
            "function syncClassSplitWrongCandidateSummaryCount() {}",
            "function renderClassSplitWrongList() { rendered += 1; }",
            "function renderClassSplitReport() {}",
            "function renderClassSplitInspector() {}",
            "function refreshClassSplitControls() {}",
            "function scheduleClassSplitPendingReviewCommitDrain() { scheduled += 1; }",
            "function enqueueTaskNotice(message) { notices.push(message); }",
            reconcile,
            "reconcileClassSplitPendingReviewCommitsAfterAnnotationLoad();",
            "assert.deepStrictEqual(applied, ['p1']);",
            "assert.strictEqual(classSplitState.pendingReviewDispositionCommits.has('job:p0:delete_bbox'), false);",
            "assert.strictEqual(classSplitState.pendingReviewDispositionCommits.has('job:p1:delete_bbox'), true);",
            "assert.strictEqual(classSplitState.pendingReviewDispositionCommits.has('job:p2:delete_bbox'), true);",
            "assert.strictEqual(scheduled, 1);",
            "assert.strictEqual(rendered, 1);",
            "assert.ok(notices.some((message) => message.includes('different annotation source')));",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_pending_review_drain_resnapshots_entries_queued_while_active():
    js = _js()
    drain = _extract_js_function(
        js,
        "drainClassSplitPendingReviewCommits",
    ).replace(
        "function drainClassSplitPendingReviewCommits",
        "async function drainClassSplitPendingReviewCommits",
        1,
    )
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const first = {queueKey: 'job:p0:delete_bbox', jobId: 'job', pointId: 'p0', disposition: 'delete_bbox', persisted: true};",
            "const second = {queueKey: 'job:p1:delete_bbox', jobId: 'job', pointId: 'p1', disposition: 'delete_bbox', persisted: true};",
            "const points = new Map([['p0', {}], ['p1', {}]]);",
            "const classSplitState = {currentJobId: 'job', pendingReviewDispositionCommits: new Map([[first.queueKey, first]]), reviewCommitDrainPromise: null, reviewCommitDrainRequested: false, dismissedWrongIds: new Set(), reviewedPointsById: new Map(), reviewDispositionReconciliationPointIds: new Set()};",
            "const savedIds = [];",
            "function classSplitPendingReviewMutationState(entry) { return entry.persisted ? 'committed' : 'unknown'; }",
            "function classSplitPendingReviewLabelIsPersisted(entry) { return classSplitPendingReviewMutationState(entry) === 'committed'; }",
            "function getClassSplitPointById(id) { return points.get(id); }",
            "async function saveClassSplitReviewDisposition(id) { savedIds.push(id); if (id === 'p0') { classSplitState.pendingReviewDispositionCommits.set(second.queueKey, second); classSplitState.reviewCommitDrainRequested = true; } return {payload: {human_reviewed_at: 'now', human_review_revision: 'rdr1_' + 'a'.repeat(32)}}; }",
            "function applyClassSplitCommittedPendingReview() {}",
            "function removeClassSplitPendingReviewCommit(key) { return classSplitState.pendingReviewDispositionCommits.delete(key); }",
            "function classSplitPendingReviewCommitCountForJob() { return classSplitState.pendingReviewDispositionCommits.size; }",
            "function persistDataQualityExplorerSession() { return true; }",
            "function renderClassSplitReviewedList() {}",
            "function renderClassSplitPendingReviewRecovery() {}",
            "function refreshClassSplitControls() {}",
            "function scheduleClassSplitReviewDispositionHydration() {}",
            "function classSplitReviewHistoryDeleteOperation() { return null; }",
            drain,
            "(async () => { const complete = await drainClassSplitPendingReviewCommits(); assert.strictEqual(complete, true); assert.deepStrictEqual(savedIds, ['p0', 'p1']); assert.strictEqual(classSplitState.pendingReviewDispositionCommits.size, 0); })().catch((error) => { console.error(error); process.exit(1); });",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_pending_review_drain_drops_a_history_tombstoned_stale_retry():
    drain = _extract_js_function(
        _js(),
        "drainClassSplitPendingReviewCommits",
    ).replace(
        "function drainClassSplitPendingReviewCommits",
        "async function drainClassSplitPendingReviewCommits",
        1,
    )
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const entry = {queueKey: 'job:p0:delete_bbox', jobId: 'job', pointId: 'p0', disposition: 'delete_bbox', persisted: true};",
            "const point = {human_review_disposition: 'delete_bbox', human_reviewed_at: 'now', human_review_revision: 'rdr1_' + 'a'.repeat(32), human_review_origin: 'desktop', human_review_persistence: 'annotation_committed_review_unsaved', human_review_before_class: 'A', human_review_target_class: 'B'};",
            "const classSplitState = {currentJobId: 'job', pendingReviewDispositionCommits: new Map([[entry.queueKey, entry]]), reviewCommitDrainPromise: null, reviewCommitDrainRequested: false, dismissedWrongIds: new Set(), reviewedPointsById: new Map([['p0', point]]), reviewDispositionReconciliationPointIds: new Set()};",
            "function classSplitPendingReviewMutationState() { return 'committed'; }",
            "function classSplitPendingReviewLabelIsPersisted() { return true; }",
            "function getClassSplitPointById() { return point; }",
            "async function saveClassSplitReviewDisposition() { const error = new Error('deleted'); error.httpStatus = 409; error.apiCode = 'review_disposition_action_deleted'; throw error; }",
            "function applyClassSplitCommittedPendingReview() { throw new Error('must not apply'); }",
            "function removeClassSplitPendingReviewCommit(key) { return classSplitState.pendingReviewDispositionCommits.delete(key); }",
            "function classSplitPendingReviewCommitCountForJob() { return classSplitState.pendingReviewDispositionCommits.size; }",
            "function persistDataQualityExplorerSession() { return true; }",
            "function renderClassSplitReviewedList() {}",
            "function renderClassSplitPendingReviewRecovery() {}",
            "function refreshClassSplitControls() {}",
            "function scheduleClassSplitReviewDispositionHydration() {}",
            "function classSplitReviewHistoryDeleteOperation() { return null; }",
            drain,
            "(async () => {",
            "  const complete = await drainClassSplitPendingReviewCommits();",
            "  assert.strictEqual(complete, true);",
            "  assert.strictEqual(classSplitState.pendingReviewDispositionCommits.size, 0);",
            "  assert.strictEqual(classSplitState.reviewedPointsById.has('p0'), false);",
            "  assert.strictEqual(classSplitState.dismissedWrongIds.has('p0'), true);",
            "  assert.strictEqual(Object.prototype.hasOwnProperty.call(point, 'human_review_disposition'), false);",
            "})().catch((error) => { console.error(error); process.exit(1); });",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_data_quality_explorer_context_preview_cache_is_bounded_and_generation_safe():
    js = _js()
    crop = _extract_js_function(js, "getClassSplitContextCropUrl")
    clear = _extract_js_function(js, "clearClassSplitContextPreviewCache")
    trim = _extract_js_function(js, "trimClassSplitContextPreviewCache")

    assert "const CLASS_SPLIT_CONTEXT_PREVIEW_CACHE_LIMIT = 36;" in js
    assert "while (classSplitContextPreviewCache.size > CLASS_SPLIT_CONTEXT_PREVIEW_CACHE_LIMIT)" in trim
    assert "classSplitContextPreviewGeneration += 1;" in clear
    assert "classSplitContextPreviewCache.clear();" in clear
    assert "classSplitContextPreviewLoads.clear();" in clear
    assert "classSplitContextPreviewSourceLoads.clear();" in clear
    assert "classSplitContextPreviewCache.forEach(revokeClassSplitContextPreviewEntry);" in clear
    assert "imageRecord.object = null" not in js
    assert "generation !== classSplitContextPreviewGeneration" in crop
    assert "if (classSplitContextPreviewLoads.get(cacheKey) === loadPromise)" in crop
    assert 'canvasEl.toBlob(resolve, "image/jpeg", 0.9);' in crop
    assert "URL.revokeObjectURL(url);" in crop
    assert "classSplitContextPreviewCache.set(cacheKey, { url, imageKey, objectUrl: true });" in crop
    assert "trimClassSplitContextPreviewCache();" in crop


def test_data_quality_explorer_multi_selection_vignette_session_contract():
    html = _html()
    css = _css()
    js = _js()
    selection = _extract_js_function(js, "rememberClassSplitSelectionFromPlot")
    panel = _extract_js_function(js, "renderClassSplitBulkPanel")
    loader = _extract_js_function(js, "pumpClassSplitMultiSelectionThumbnailQueue")
    cancel = _extract_js_function(js, "cancelClassSplitMultiSelectionLoads")
    remove = _extract_js_function(js, "removeClassSplitPointFromMultiSelection")
    disposition = _extract_js_function(js, "applyClassSplitMultiSelectionDisposition")
    mutation_busy = _extract_js_function(js, "classSplitMutationIsBusy")
    graph_remove = _extract_js_function(
        js, "removeClassSplitPointsFromActiveReviewGraph"
    )
    save_disposition = _extract_js_function(js, "saveClassSplitReviewDisposition")
    hover = _extract_js_function(js, "showClassSplitGraphHoverPreview")
    refresh = _extract_js_function(js, "refreshClassSplitControls")

    for element_id in (
        "classSplitBulkPanel",
        "classSplitBulkGrid",
        "classSplitBulkConfirm",
        "classSplitBulkSkip",
        "classSplitBulkLoadMore",
        "classSplitBulkActionStatus",
        "classSplitSingleInspectorSection",
    ):
        assert f'id="{element_id}"' in html
    assert html.index('id="classSplitBulkPanel"') > html.index('<div class="class-split-review">')
    assert 'class-split-multi-selection__spinner" aria-hidden="true"' in js
    assert "Click vignette to dismiss from multi-selection" in js
    assert "context=wide" in js
    assert "CLASS_SPLIT_MULTI_SELECTION_PAGE_SIZE = 36" in js
    assert "CLASS_SPLIT_MULTI_SELECTION_LOAD_CONCURRENCY = 6" in js
    assert "CLASS_SPLIT_MULTI_SELECTION_ACTION_CONCURRENCY = 3" in js
    assert "new AbortController()" in loader
    assert 'cache: "force-cache"' in loader
    assert "classSplitMultiSelectionBatchIsCurrent(batch)" in loader
    assert "batch.controllers.forEach((controller) => controller.abort())" in cancel
    assert "batch.objectUrls.forEach((url) => URL.revokeObjectURL(url))" in cancel
    assert "classSplitMutationIsBusy()" in selection
    normal_selection = selection[selection.index("const ids = new Set();"):]
    assert normal_selection.index("renderClassSplitBulkPanel();") < normal_selection.index("renderClassSplitPlot();")
    assert "selectedIds.length > 1" in panel
    assert "classSplitState.lassoPointIds.delete(safePointId)" in remove
    assert "renderClassSplitPlot();" in remove
    assert "const selectedIds = getClassSplitMultiSelectionIds();" in disposition
    assert "saveClassSplitReviewDisposition(" in disposition
    assert "classSplitMutationIsBusy()" in disposition
    assert "deferUi: true" in disposition
    assert "removeClassSplitPointFromActiveReviewGraph(" not in disposition
    assert "classSplitState.lassoPointIds = new Set(failedIds);" in disposition
    assert "classSplitMutationIsBusy()" in refresh
    assert "classSplitReviewHistoryDeleteOperation(safeJobId)" in mutation_busy
    assert "classSplitPendingReviewCommitCountForJob(safeJobId)" in mutation_busy
    assert "ignoreReviewActionPointIds" in mutation_busy
    assert "renderedPointIds" in js
    assert "classSplitGraphMutationContextIsCurrent(mutationContext)" in graph_remove
    assert "keepIndexByOriginalIndex" in graph_remove
    assert "keepIndices.indexOf(index)" not in graph_remove
    assert "syncAfterExternalRestyle" in graph_remove
    assert "classSplitAsyncRequestIsCurrent(requestGeneration, jobId)" in save_disposition
    bulk_html = html[
        html.index('id="classSplitBulkPanel"'):
        html.index('id="classSplitSingleInspectorSection"')
    ]
    assert 'role="status" aria-live="polite"' in bulk_html
    assert bulk_html.count('role="status" aria-live="polite"') == 1
    assert "requestKey === classSplitGraphHoverState.requestKey" in hover
    assert "classSplitEmitSelection" in js
    assert "classSplitMultiSelectionSnapshot" in js
    assert ".class-split-multi-selection__grid" in css
    assert ".class-split-multi-selection__tile.is-loaded" in css
    assert ".class-split-multi-selection__tile.is-error" in css


def test_annotation_reopen_retry_and_training_commit_drain_do_not_block_saved_labels():
    js = _js()
    reopen_at = js.index("async function handleAnnotationTransientExpiry")
    reopen = js[reopen_at:js.index("async function flushAnnotationSnapshot", reopen_at)]
    flush_at = js.index("async function flushAnnotationSnapshot")
    flush = js[flush_at:js.index("async function annotationHeartbeatTick", flush_at)]
    unavailable = _extract_js_function(js, "isTransientSessionUnavailableResponse")
    heartbeat_at = js.index("async function annotationHeartbeatTick")
    heartbeat = js[heartbeat_at:js.index("function startAnnotationTimers", heartbeat_at)]
    schedule_drain = _extract_js_function(
        js, "scheduleClassSplitPendingTrainingCommitDrain"
    )

    assert "{ retrySnapshot = true }" in reopen
    assert "if (retrySnapshot)" in reopen
    assert "return true;" in reopen
    assert "handleAnnotationTransientExpiry({ retrySnapshot: false })" in flush
    assert "isTransientSessionUnavailableResponse(resp.status, detail)" in flush
    assert 'String(payload?.detail || "").trim() === "transient_session_not_found"' in unavailable
    assert "numericStatus !== 404" in unavailable
    assert "isTransientSessionUnavailableResponse(resp.status, detail)" in heartbeat
    assert 'annotationSourceState.lastFailedSnapshotSignature = "";' in heartbeat
    assert "flushAnnotationSnapshot({ manual: false, background: true })" in heartbeat
    release_at = flush.index("annotationSourceState.saveInFlight = false;")
    retry_at = flush.index("const retried = await flushAnnotationSnapshot({")
    assert release_at < retry_at
    assert "await retryClassSplitPendingTrainingCommitsAfterAnnotationSave();" not in flush
    assert flush.count("scheduleClassSplitPendingTrainingCommitDrain();") >= 2
    assert "window.setTimeout" in schedule_drain
    assert "retryClassSplitPendingTrainingCommitsAfterAnnotationSave()" in schedule_drain


def test_annotation_source_resume_descriptor_is_minimal_same_tab_state():
    js = _js()
    helpers_start = js.index("function normalizeAnnotationResumeText")
    helpers_end = js.index("const glossaryLibraryState", helpers_start)
    helpers = js[helpers_start:helpers_end]
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const ANNOTATION_SOURCE_RESUME_STORAGE_KEY = 'tator.annotationSource.resume.v1';",
            "const ANNOTATION_SOURCE_RESUME_SCHEMA_VERSION = 1;",
            "const values = new Map();",
            "const sessionStorage = {",
            "  getItem: (key) => values.has(key) ? values.get(key) : null,",
            "  setItem: (key, value) => values.set(key, String(value)),",
            "  removeItem: (key) => values.delete(key),",
            "};",
            "const window = { sessionStorage };",
            helpers,
            "const linked = {",
            "  mode: 'linked', datasetId: 'dataset-123', datasetLabel: 'not persisted',",
            "  lockSessionId: 'old-lock', serverDatasets: ['/secret/a', '/secret/b'],",
            "};",
            "assert.strictEqual(persistAnnotationSourceResumeDescriptor(linked), true);",
            "assert.deepStrictEqual(JSON.parse(values.get(ANNOTATION_SOURCE_RESUME_STORAGE_KEY)), {",
            "  version: 1, mode: 'linked', datasetId: 'dataset-123',",
            "});",
            "assert.deepStrictEqual(readAnnotationSourceResumeDescriptor(), {",
            "  version: 1, mode: 'linked', datasetId: 'dataset-123',",
            "});",
            "assert.strictEqual(clearAnnotationSourceResumeDescriptor({ linkedDatasetId: 'other' }), false);",
            "assert.strictEqual(clearAnnotationSourceResumeDescriptor({ linkedDatasetId: 'dataset-123' }), true);",
            "assert.strictEqual(values.has(ANNOTATION_SOURCE_RESUME_STORAGE_KEY), false);",
            "const transient = {",
            "  mode: 'transient', sessionId: 'opaque-session',",
            "  transientOpenPath: '/already/known/dataset', datasetLabel: 'My transient set',",
            "  lockSessionId: 'must-not-survive', serverDatasetList: ['/other/path'],",
            "};",
            "assert.strictEqual(persistAnnotationSourceResumeDescriptor(transient), true);",
            "const storedTransient = JSON.parse(values.get(ANNOTATION_SOURCE_RESUME_STORAGE_KEY));",
            "assert.deepStrictEqual(storedTransient, {",
            "  version: 1, mode: 'transient', sessionId: 'opaque-session',",
            "  transientOpenPath: '/already/known/dataset', datasetLabel: 'My transient set',",
            "});",
            "assert.strictEqual('lockSessionId' in storedTransient, false);",
            "assert.strictEqual('serverDatasetList' in storedTransient, false);",
            "assert.strictEqual(clearAnnotationSourceResumeDescriptor({ transientSessionId: 'wrong' }), false);",
            "assert(values.has(ANNOTATION_SOURCE_RESUME_STORAGE_KEY));",
            "assert.strictEqual(clearAnnotationSourceResumeDescriptor({ transientSessionId: 'opaque-session' }), true);",
            "assert.strictEqual(values.has(ANNOTATION_SOURCE_RESUME_STORAGE_KEY), false);",
            "values.set(ANNOTATION_SOURCE_RESUME_STORAGE_KEY, '{broken');",
            "assert.strictEqual(readAnnotationSourceResumeDescriptor(), null);",
            "assert.strictEqual(values.has(ANNOTATION_SOURCE_RESUME_STORAGE_KEY), false);",
            "values.set(ANNOTATION_SOURCE_RESUME_STORAGE_KEY, 'x'.repeat(8193));",
            "assert.strictEqual(readAnnotationSourceResumeDescriptor(), null);",
            "assert.strictEqual(values.has(ANNOTATION_SOURCE_RESUME_STORAGE_KEY), false);",
            "values.set(ANNOTATION_SOURCE_RESUME_STORAGE_KEY, JSON.stringify({ version: 2, mode: 'linked', datasetId: 'stale' }));",
            "assert.strictEqual(readAnnotationSourceResumeDescriptor(), null);",
            "assert.strictEqual(values.has(ANNOTATION_SOURCE_RESUME_STORAGE_KEY), false);",
            "values.set(ANNOTATION_SOURCE_RESUME_STORAGE_KEY, JSON.stringify({ mode: 'linked', datasetId: 'unversioned' }));",
            "assert.strictEqual(readAnnotationSourceResumeDescriptor(), null);",
            "assert.strictEqual(values.has(ANNOTATION_SOURCE_RESUME_STORAGE_KEY), false);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_annotation_source_page_restore_uses_normal_open_and_preserves_retry_pointer():
    js = _js()
    resume_start = js.index("async function resumeAnnotationSourceAfterPageRestore")
    resume_end = js.index("function parseJsonObjectSafe", resume_start)
    resume_helpers = js[resume_start:resume_end]
    open_start = js.index("async function openDatasetInAnnotationMode")
    open_end = js.index("async function openDatasetEntryInAnnotation", open_start)
    open_body = js[open_start:open_end]
    close_start = js.index("async function closeAnnotationDataset")
    close_end = js.index("async function startAnnotationSession", close_start)
    close_body = js[close_start:close_end]
    dom_ready = js[js.index('document.addEventListener("DOMContentLoaded"'):]

    assert "clearResumeDescriptor: false" in open_body
    assert "annotationSourceState.lockSessionId = generateUUID();" in open_body
    assert "persistAnnotationSourceResumeDescriptor(annotationSourceState);" in open_body
    assert open_body.index("const manifestResp = await fetch") < open_body.index(
        "persistAnnotationSourceResumeDescriptor(annotationSourceState);"
    )
    assert "if (activateLabeling)" in open_body
    assert "clearAnnotationSourceResumeDescriptor();" in close_body
    assert close_body.index("await stopAnnotationSession();") < close_body.index(
        "clearAnnotationSourceResumeDescriptor();"
    )
    assert dom_ready.index("initClassSplitExplorer();") < dom_ready.index(
        "scheduleAnnotationSourceResumeAfterPageRestore();"
    )
    assert 'window.addEventListener("pageshow"' in dom_ready
    transient_reopen = js[
        js.index("async function handleAnnotationTransientExpiry") : js.index(
            "function isTransientSessionUnavailableResponse"
        )
    ]
    assert transient_reopen.index("await startAnnotationSession({ force: false });") < transient_reopen.index(
        "persistAnnotationSourceResumeDescriptor(annotationSourceState);"
    )
    dataset_delete = js[
        js.index("async function handleDatasetDelete") : js.index(
            "async function handleDatasetConvert"
        )
    ]
    assert "clearAnnotationSourceResumeDescriptor({" in dataset_delete
    assert "linkedDatasetId: String(entry.id || \"\").trim()" in dataset_delete

    script = "\n".join(
        [
            "const assert = require('assert');",
            "let annotationSourceResumeInFlight = null;",
            "let annotationSourceResumeScheduled = false;",
            "let active = false;",
            "let shouldFail = true;",
            "let openCalls = [];",
            "let statuses = [];",
            "const descriptor = {",
            "  version: 1, mode: 'transient', sessionId: 'session-a',",
            "  transientOpenPath: '/known/path', datasetLabel: 'Known set',",
            "};",
            "function isAnnotationDatasetModeActive() { return active; }",
            "function readAnnotationSourceResumeDescriptor() { return descriptor; }",
            "async function openDatasetInAnnotationMode(options) {",
            "  openCalls.push(options);",
            "  if (shouldFail) throw new Error('backend restarting');",
            "  active = true;",
            "}",
            "function isAnnotationReadOnly() { return false; }",
            "function annotationSourceLabel() { return 'transient dataset'; }",
            "function setSamStatus(message, options) { statuses.push({ message, options }); }",
            "const window = {",
            "  setTimeout: (callback) => { callback(); return 1; },",
            "  addEventListener: () => {},",
            "};",
            "const document = { readyState: 'complete' };",
            resume_helpers,
            "assert.strictEqual(await resumeAnnotationSourceAfterPageRestore(), false);",
            "assert.strictEqual(openCalls.length, 1);",
            "assert.deepStrictEqual(openCalls[0], {",
            "  mode: 'transient', datasetId: '', sessionId: 'session-a',",
            "  datasetLabel: 'Known set', transientOpenPath: '/known/path',",
            "  activateLabeling: false,",
            "});",
            "assert(statuses.at(-1).message.includes('resume pointer was kept for retry'));",
            "shouldFail = false;",
            "assert.strictEqual(await resumeAnnotationSourceAfterPageRestore(), true);",
            "assert.strictEqual(openCalls.length, 2);",
            "assert(statuses.at(-1).message.includes('fresh editor lock'));",
        ]
    )
    subprocess.run(
        [
            "node",
            "-e",
            f"(async () => {{\n{script}\n}})().catch((error) => {{ console.error(error); process.exit(1); }});",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_transient_dataset_help_describes_restart_and_same_tab_lock_recovery():
    html = _html()

    assert "lost on backend restart" not in html
    assert "retained across ordinary backend restarts" in html
    assert "retained across ordinary backend restarts and lease expiry until explicitly deleted" in html
    assert "safely requests a new editor lock after reload or back navigation" in html
    assert 'src="ybat.js?v=20260811b"' in html
    assert 'href="ybat.css?v=20260808a"' in html


def test_transient_library_save_quiesces_editor_and_uses_revision_gate():
    js = _js()
    ensure_at = js.index("async function ensureTransientAnnotationSavedBeforeLibrarySave")
    ensure = js[ensure_at:js.index("async function saveTransientDatasetPath", ensure_at)]
    save_at = js.index("async function saveTransientDatasetPath")
    save = js[save_at:js.index("async function uploadDatasetZip", save_at)]

    assert "allowLibrarySave: true" in ensure
    assert "reopen the path and retry" in ensure
    quiesce_at = save.index("annotationSourceState.librarySaveInFlight = true;")
    flush_at = save.index("await ensureTransientAnnotationSavedBeforeLibrarySave(sessionId)")
    request_at = save.index("expected_revision: expectedRevision")
    switch_at = save.index("await openDatasetEntryInAnnotation(data)")
    cleanup_at = save.index("method: \"DELETE\"")
    clear_at = save.index("clearDatasetPathTransientState();", cleanup_at)
    assert quiesce_at < flush_at < request_at < switch_at < cleanup_at < clear_at
    assert "The retained transient session was not deleted." in save
    assert "transientCleanupWarning" in save


def test_transient_restart_recovery_matches_only_exact_missing_session_and_retries_autosave():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const API_ROOT = 'http://backend.test';",
            "const annotationSourceState = {",
            "  mode: 'transient',",
            "  lock: {},",
            "  statusMessage: '',",
            "  lastFailedSnapshotSignature: 'failed-signature',",
            "  dirtyRecordsByKey: new Map([['train/img.jpg', {}]]),",
            "  saveInFlight: false,",
            "};",
            "let response = null;",
            "let reopenCalls = 0;",
            "let flushCalls = 0;",
            "const scheduled = [];",
            "const window = { setTimeout: (callback) => { scheduled.push(callback); return scheduled.length; } };",
            "function isAnnotationDatasetModeActive() { return true; }",
            "function isAnnotationReadOnly() { return false; }",
            "function getAnnotationBasePath() { return API_ROOT + '/datasets/transient/old/annotation'; }",
            "function annotationSessionPayload() { return { session_id: 'lock' }; }",
            "function stopAnnotationTimers() {}",
            "function syncLabelingSourceControls() {}",
            "function parseApiError(detail, fallback) {",
            "  try { return JSON.parse(detail).detail || fallback; } catch { return fallback; }",
            "}",
            "async function handleAnnotationTransientExpiry() { reopenCalls += 1; return true; }",
            "async function flushAnnotationSnapshot() { flushCalls += 1; return true; }",
            # The source helper has an object-literal default argument.  The
            # lightweight function extractor intentionally does not parse JS
            # parameter lists deeply, so use an equivalent harness helper here
            # instead of truncating the function at the first ``{``.
            "function parseJsonObjectSafe(rawText, fallback = {}) {",
            "  try { const parsed = JSON.parse(rawText); return parsed && typeof parsed === 'object' ? parsed : fallback; } catch { return fallback; }",
            "}",
            _extract_js_function(js, "isTransientSessionUnavailableResponse"),
            "async " + _extract_js_function(js, "annotationHeartbeatTick"),
            "global.fetch = async () => response;",
            "response = { ok: false, status: 404, text: async () => JSON.stringify({ detail: 'other_missing' }) };",
            "await assert.rejects(annotationHeartbeatTick(), /other_missing/);",
            "assert.strictEqual(reopenCalls, 0);",
            "response = { ok: false, status: 404, text: async () => JSON.stringify({ detail: 'transient_session_not_found' }) };",
            "await annotationHeartbeatTick();",
            "assert.strictEqual(reopenCalls, 1);",
            "response = { ok: true, status: 200, text: async () => JSON.stringify({ status: 'ok', lock: {} }) };",
            "await annotationHeartbeatTick();",
            "assert.strictEqual(annotationSourceState.lastFailedSnapshotSignature, '');",
            "assert.strictEqual(scheduled.length, 1);",
            "scheduled[0]();",
            "assert.strictEqual(flushCalls, 1);",
            "assert.strictEqual(isTransientSessionUnavailableResponse(404, JSON.stringify({ message: 'transient_session_not_found' })), false);",
            "assert.strictEqual(isTransientSessionUnavailableResponse(500, JSON.stringify({ detail: 'transient_session_not_found' })), false);",
        ]
    )
    subprocess.run(
        [
            "node",
            "-e",
            f"(async () => {{\n{script}\n}})().catch((error) => {{ console.error(error); process.exit(1); }});",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_data_quality_explorer_live_qwen_updates_preserve_keyed_vignette_dom():
    js = _js()
    update_job = _extract_js_function(js, "updateClassSplitQwenReviewJob")
    patch_card = _extract_js_function(js, "patchClassSplitQwenReviewCard")
    patch_wrapper = _extract_js_function(js, "patchClassSplitQwenReviewWrapper")
    restore_disclosures = _extract_js_function(
        js, "restoreClassSplitDisclosureState"
    )
    qwen_poll = _extract_js_function(js, "scheduleClassSplitQwenReviewPoll")
    qwen_start = _extract_js_function(js, "startClassSplitQwenReview")

    assert "patchClassSplitQwenReviewCard(pointId);" in update_job
    assert "renderClassSplitWrongList();" not in qwen_poll
    assert "renderClassSplitWrongList();" not in qwen_start
    assert "patchClassSplitQwenReviewWrapper(" in patch_card
    assert "safePointId\n            ) || patched" in patch_card
    assert "wrappers.forEach((wrapper)" in patch_card
    assert 'wrapper.querySelector(":scope > .class-split-qwen-review")' in patch_wrapper
    assert "if (!currentRoot || !nextRoot || !active)" in patch_wrapper
    assert "currentText.textContent = nextText.textContent;" in patch_wrapper
    assert "reconcileClassSplitQwenEvidenceItems(" in patch_wrapper
    assert "currentSummary.innerHTML = nextSummary.innerHTML;" in patch_wrapper
    assert "window.scrollTo(pageScrollX, pageScrollY);" in patch_wrapper
    assert 'data-qwen-review-block="${escapeHtml(pointId)}"' in js
    assert 'data-evidence-id="${escapeHtml(evidenceId || url)}"' in js
    assert 'details.dataset.disclosureBound !== "1"' in restore_disclosures
    trace = _extract_js_function(js, "renderClassSplitQwenReviewTraceToast")
    assert trace.index("rememberClassSplitDisclosureState(body);") < trace.index(
        "classSplitElements.qwenReviewTraceBody.innerHTML"
    )
    assert trace.index("restoreClassSplitDisclosureState(body);") < trace.index(
        "body.scrollTop = followBottom ? body.scrollHeight : previousScrollTop;"
    )


def test_data_quality_explorer_custom_backbone_survives_async_catalog_refresh():
    js = _js()
    update = _extract_js_function(js, "updateClassSplitBackboneOptions")
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const DINOV3_BACKBONES = ['dino-default', 'dino-custom'];",
            "const CRADIO_BACKBONES = ['cradio-default'];",
            "const select = { value: '', options: [] };",
            "const classSplitElements = { backbone: select, encoderType: { value: 'clip' } };",
            "const classSplitState = {",
            "  pendingBackboneSelection: 'clip-custom',",
            "  clipBackbones: [],",
            "  capabilities: { default_clip_model: 'clip-default' },",
            "};",
            "function fillSelectOptions(target, options, preferred) {",
            "  const values = options.map((entry) => typeof entry === 'object' ? entry.value : entry);",
            "  target.options = values;",
            "  target.value = values.includes(preferred) ? preferred : (values[0] || '');",
            "}",
            update,
            "updateClassSplitBackboneOptions();",
            "assert.strictEqual(select.value, 'clip-default');",
            "assert.strictEqual(classSplitState.pendingBackboneSelection, 'clip-custom');",
            "classSplitState.clipBackbones = ['clip-default', 'clip-custom'];",
            "updateClassSplitBackboneOptions();",
            "assert.strictEqual(select.value, 'clip-custom');",
            "assert.strictEqual(classSplitState.pendingBackboneSelection, '');",
            "classSplitState.capabilities.default_clip_model = 'clip-new-default';",
            "classSplitState.clipBackbones = ['clip-new-default', 'clip-custom'];",
            "updateClassSplitBackboneOptions();",
            "assert.strictEqual(select.value, 'clip-custom');",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_class_split_refinement_candidate_compatibility_contract():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const CLASS_SPLIT_REFINEMENT_CATEGORIES = Object.freeze([",
            "  'confirmed_outlier', 'explained_not_outlier', 'mixed_or_composite', 'unresolved', 'pair_conflict',",
            "]);",
            "const CLASS_SPLIT_REFINEMENT_SCHEMA = 'class-analysis-patch-refinement-v5';",
            "const CLASS_SPLIT_REFINEMENT_DECISION_CONTRACT = 'class-analysis-patch-decision-v9';",
            "const CLASS_SPLIT_HUMAN_REVIEW_QUALIFICATION_CONTRACT = 'class-analysis-qualified-human-review-v1';",
            "const CLASS_SPLIT_HUMAN_REVIEW_RANK_CONTRACT = 'confirmed-band-stage1-suspicion-probe-excess-v1';",
            "const CLASS_SPLIT_SELECTOR_PRIORITY_CONTRACT = 'expected-review-utility-global-boosted-statistical-overlap-rerank-v7';",
            "const CLASS_SPLIT_SELECTOR_BASE_MODEL_CONTRACT = 'expected-review-utility-global-boosted-gated-dataset-overlap-v6';",
            "const classSplitState = { result: null, pointsById: new Map() };",
            "function getClassSplitPointById(pointId) { return classSplitState.pointsById.get(String(pointId || '')); }",
            "function getClassSplitDualBBoxConflict(candidate) { return candidate?.dual_bbox_conflict || null; }",
            _extract_js_function(js, "classSplitResultHasRefinement"),
            _extract_js_function(js, "normalizeClassSplitRefinementStatus"),
            _extract_js_function(js, "mergeClassSplitCandidateWithPoint"),
            _extract_js_function(js, "dedupeClassSplitCandidates"),
            _extract_js_function(js, "getClassSplitRawCandidates"),
            _extract_js_function(js, "getClassSplitRefinementStatus"),
            _extract_js_function(js, "classSplitResultSupportsPriorityHumanReview"),
            _extract_js_function(js, "getClassSplitCandidatePriorityHumanReviewRank"),
            _extract_js_function(js, "getClassSplitCandidateSelectorPriority"),
            _extract_js_function(js, "sortClassSplitPriorityHumanReviewCandidates"),
            _extract_js_function(js, "getClassSplitPrimaryCandidates"),
            "const rawP1 = { point_id: 'p1', class_name: 'Bike' };",
            "const rawP2 = { point_id: 'p2', class_name: 'Bike' };",
            "classSplitState.result = {",
            "  summary: { analysis_scope: 'all_classes', refinement: { enabled: false, status: 'disabled' } },",
            "  wrong_class_candidates: [rawP1], refinement_candidates: [], vignette_candidates: [],",
            "};",
            "assert.strictEqual(classSplitResultHasRefinement(), false);",
            "assert.deepStrictEqual(getClassSplitRawCandidates().map((row) => row.point_id), ['p1']);",
                "assert.deepStrictEqual(getClassSplitPrimaryCandidates().map((row) => row.point_id), ['p1']);",
            "const refinedP1 = { point_id: 'p1', refined_outlier: { status: 'confirmed_outlier' } };",
            "const pair = { point_id: 'pair', candidate_kind: 'pair_conflict' };",
            "classSplitState.result = {",
            "  summary: { analysis_scope: 'all_classes', refinement: { enabled: true, status: 'partial' } },",
            "  wrong_class_candidates: [rawP1, rawP2, pair],",
            "  refinement_candidates: [refinedP1],",
            "  vignette_candidates: [refinedP1, pair],",
            "};",
            "assert.strictEqual(classSplitResultHasRefinement(), true);",
            "assert.deepStrictEqual(getClassSplitRawCandidates().map((row) => row.point_id), ['p1', 'p2', 'pair']);",
                "assert.deepStrictEqual(getClassSplitPrimaryCandidates().map((row) => row.point_id), ['p1', 'pair']);",
            "classSplitState.result = {",
            "  summary: { analysis_scope: 'all_classes', refinement: { enabled: true, status: 'failed' } },",
            "  wrong_class_candidates: [rawP1, rawP2],",
            "  refinement_candidates: [{ point_id: 'p1', refined_outlier: { status: 'unresolved' } }],",
            "  vignette_candidates: [],",
            "};",
            "assert.deepStrictEqual(getClassSplitRawCandidates().map((row) => row.point_id), ['p1', 'p2']);",
            "assert.deepStrictEqual(getClassSplitPrimaryCandidates(), []);",
            "classSplitState.result = {",
            "  summary: { analysis_scope: 'all_classes', refinement: { enabled: true, status: 'completed' } },",
            "  wrong_class_candidates: [rawP1, rawP2],",
            "  refinement_candidates: [{ point_id: 'p1', refined_outlier: { status: 'confirmed_outlier', marker: 'refined' } }],",
            "  vignette_candidates: [],",
            "};",
            "const completedTruncated = getClassSplitRawCandidates();",
            "assert.deepStrictEqual(completedTruncated.map((row) => row.point_id), ['p1', 'p2']);",
            "assert.strictEqual(completedTruncated[0].refined_outlier.marker, 'refined');",
            "assert.strictEqual(completedTruncated[1].class_name, 'Bike');",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_class_split_v9_priority_human_review_union_and_rank_contract():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const CLASS_SPLIT_REFINEMENT_CATEGORIES = Object.freeze([",
            "  'confirmed_outlier', 'explained_not_outlier', 'mixed_or_composite', 'unresolved', 'pair_conflict',",
            "]);",
            "const CLASS_SPLIT_REFINEMENT_SCHEMA = 'class-analysis-patch-refinement-v5';",
            "const CLASS_SPLIT_REFINEMENT_DECISION_CONTRACT = 'class-analysis-patch-decision-v9';",
            "const CLASS_SPLIT_HUMAN_REVIEW_QUALIFICATION_CONTRACT = 'class-analysis-qualified-human-review-v1';",
            "const CLASS_SPLIT_HUMAN_REVIEW_RANK_CONTRACT = 'confirmed-band-stage1-suspicion-probe-excess-v1';",
            "const CLASS_SPLIT_SELECTOR_PRIORITY_CONTRACT = 'expected-review-utility-global-boosted-gated-dataset-overlap-v6';",
            "const classSplitState = { result: null, pointsById: new Map() };",
            "function getClassSplitPointById(pointId) { return classSplitState.pointsById.get(String(pointId || '')); }",
            "function getClassSplitDualBBoxConflict(candidate) { return candidate?.dual_bbox_conflict || null; }",
            _extract_js_function(js, "classSplitResultHasRefinement"),
            _extract_js_function(js, "normalizeClassSplitRefinementStatus"),
            _extract_js_function(js, "mergeClassSplitCandidateWithPoint"),
            _extract_js_function(js, "dedupeClassSplitCandidates"),
            _extract_js_function(js, "getClassSplitRawCandidates"),
            _extract_js_function(js, "classSplitResultSupportsPriorityHumanReview"),
            _extract_js_function(js, "getClassSplitCandidatePriorityHumanReviewRank"),
            _extract_js_function(js, "getClassSplitCandidateSelectorPriority"),
            _extract_js_function(js, "sortClassSplitPriorityHumanReviewCandidates"),
            _extract_js_function(js, "getClassSplitPrimaryCandidates"),
            "const evidence = (status, qualified, rank, contract = CLASS_SPLIT_HUMAN_REVIEW_QUALIFICATION_CONTRACT) => ({",
            "  schema: 'class-analysis-patch-refinement-v5',",
            "  decision_contract: CLASS_SPLIT_REFINEMENT_DECISION_CONTRACT,",
            "  status, human_review_qualification_contract: contract,",
            "  human_review_rank_contract: CLASS_SPLIT_HUMAN_REVIEW_RANK_CONTRACT,",
            "  qualified_for_human_review: qualified, human_review_rank: rank,",
            "});",
            "const rankedConfirmed = { point_id: 'ranked-confirmed', wrong_class_suspicion: 0.8, refined_outlier: evidence('confirmed_outlier', true, 1) };",
            "const rankedUnresolved = { point_id: 'ranked-unresolved', wrong_class_suspicion: 0.9, refined_outlier: evidence('unresolved', true, 2) };",
            "const serializedConfirmed = { point_id: 'serialized-confirmed', wrong_class_suspicion: 0.7, refined_outlier: evidence('confirmed_outlier', false, null) };",
            "const serializedPair = { point_id: 'serialized-pair', wrong_class_suspicion: 0.95, refined_outlier: evidence('pair_conflict', false, null) };",
            "const invalidQualified = { point_id: 'invalid-qualified', wrong_class_suspicion: 1, refined_outlier: evidence('unresolved', true, 3, 'stale-contract') };",
            "const points = [rankedConfirmed, rankedUnresolved, serializedConfirmed, serializedPair, invalidQualified];",
            "classSplitState.pointsById = new Map(points.map((row) => [row.point_id, row]));",
            "classSplitState.result = {",
            "  summary: { analysis_scope: 'all_classes', refinement: {",
            "    enabled: true, status: 'completed', schema: CLASS_SPLIT_REFINEMENT_SCHEMA, decision_contract: CLASS_SPLIT_REFINEMENT_DECISION_CONTRACT,",
            "    human_review_qualification_contract: CLASS_SPLIT_HUMAN_REVIEW_QUALIFICATION_CONTRACT,",
            "    human_review_rank_contract: CLASS_SPLIT_HUMAN_REVIEW_RANK_CONTRACT,",
            "  } },",
            "  points, refinement_candidates: [rankedUnresolved, invalidQualified, rankedConfirmed, serializedConfirmed, serializedPair],",
            "  vignette_candidates: [serializedPair, serializedConfirmed, rankedConfirmed], wrong_class_candidates: points,",
            "};",
            "assert.strictEqual(classSplitResultSupportsPriorityHumanReview(), true);",
            "assert.deepStrictEqual(classSplitState.result.vignette_candidates.map((row) => row.point_id), [",
            "  'serialized-pair', 'serialized-confirmed', 'ranked-confirmed',",
            "]);",
            "assert.deepStrictEqual(getClassSplitPrimaryCandidates().map((row) => row.point_id), [",
            "  'ranked-confirmed', 'serialized-confirmed', 'serialized-pair', 'ranked-unresolved',",
            "]);",
            "assert.strictEqual(getClassSplitCandidatePriorityHumanReviewRank(rankedUnresolved), 2);",
            "assert.strictEqual(getClassSplitCandidatePriorityHumanReviewRank(invalidQualified), null);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_class_split_refinement_queue_order_and_stability_contract():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const classSplitElements = {",
            "  filterClass: { value: '' },",
            "  vignetteCategory: { value: 'confirmed_outlier' },",
            "  showAllRough: { checked: false },",
            "};",
            "const classSplitState = {",
            "  currentJobId: 'job-1', vignetteCategory: 'confirmed_outlier',",
            "  showAllRough: false, wrongQueueIds: [], wrongQueueSignature: '',",
            "};",
            "function getClassSplitVignetteCategory() { return classSplitState.vignetteCategory; }",
            "function getClassSplitVignetteSort() { return 'priority'; }",
            _extract_js_function(js, "getClassSplitWrongQueueSignature"),
            _extract_js_function_before(
                js,
                "reconcileClassSplitWrongQueue",
                "function orderClassSplitCandidatesByQueue",
            ),
            _extract_js_function(js, "orderClassSplitCandidatesByQueue"),
            "const candidates = Array.from({ length: 16 }, (_, index) => ({ point_id: `p${index}` }));",
            "const originalRandom = Math.random;",
            "Math.random = () => 0;",
            "const shuffled = reconcileClassSplitWrongQueue(candidates, { shuffle: true });",
            "Math.random = originalRandom;",
            "assert.strictEqual(shuffled.length, 12);",
            "assert.notDeepStrictEqual(shuffled, candidates.slice(0, 12).map((row) => row.point_id));",
            "assert.deepStrictEqual(",
            "  orderClassSplitCandidatesByQueue(candidates, shuffled).map((row) => row.point_id),",
            "  shuffled,",
            ");",
            "const removedId = shuffled[3];",
            "const remainingCandidates = candidates.filter((row) => row.point_id !== removedId);",
            "const nextQueue = reconcileClassSplitWrongQueue(remainingCandidates);",
            "assert.deepStrictEqual(",
            "  nextQueue.slice(0, 11),",
            "  shuffled.filter((pointId) => pointId !== removedId),",
            ");",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_class_split_refinement_preset_session_and_defaults_contract():
    js = _js()
    script = "\n".join(
            [
                "const assert = require('assert');",
                "const CLASS_SPLIT_REFINEMENT_API_VERSION = 5;",
                "const CLASS_SPLIT_REFINEMENT_SCHEMA = 'class-analysis-patch-refinement-v5';",
                "const CLASS_SPLIT_REFINEMENT_DECISION_CONTRACT = 'class-analysis-patch-decision-v9';",
                "const CLASS_SPLIT_SELECTOR_PRIORITY_CONTRACT = 'expected-review-utility-global-boosted-statistical-overlap-rerank-v7';",
                "const CLASS_SPLIT_SELECTOR_FEATURE_CONTRACT = 'raw-stage1-patch-same-image-and-gated-dataset-overlap-features-v3';",
                "const CLASS_SPLIT_SELECTOR_MODEL_SCHEMA = 'class-analysis-selector-utility-model-v2';",
                "const CLASS_SPLIT_SELECTOR_UTILITY_POLICY_CONTRACT = 'actionability-times-75pct-base-plus-25pct-reviewability-v1';",
                "const CLASS_SPLIT_SELECTOR_DATASET_OVERLAP_APPLICATION_CONTRACT = 'affirmative-current-localized-material-source-loo-v2';",
                "const CLASS_SPLIT_SELECTOR_DATASET_OVERLAP_DIAGNOSTIC_CONTRACT = 'capture-loo-beta-wilson-shrunk-rank-only-v1';",
                "const CLASS_SPLIT_SELECTOR_GLOBAL_ACTIONABILITY_MODEL_CONTRACT = 'pair-blind-shallow-histogram-gradient-boosting-v1';",
                "const classSplitState = {capabilities: {fine_grained_refinement: {api_version: 5, schema: CLASS_SPLIT_REFINEMENT_SCHEMA, decision_contract: CLASS_SPLIT_REFINEMENT_DECISION_CONTRACT, selector_priority_contract: CLASS_SPLIT_SELECTOR_PRIORITY_CONTRACT, selector_feature_contract: CLASS_SPLIT_SELECTOR_FEATURE_CONTRACT, selector_model_schema: CLASS_SPLIT_SELECTOR_MODEL_SCHEMA, selector_model_digest: 'a'.repeat(64), selector_utility_policy_contract: CLASS_SPLIT_SELECTOR_UTILITY_POLICY_CONTRACT, selector_dataset_overlap_application_contract: CLASS_SPLIT_SELECTOR_DATASET_OVERLAP_APPLICATION_CONTRACT, selector_dataset_overlap_diagnostic_contract: CLASS_SPLIT_SELECTOR_DATASET_OVERLAP_DIAGNOSTIC_CONTRACT, selector_dataset_overlap_scoring_effect_enabled: true, selector_global_actionability_model_contract: CLASS_SPLIT_SELECTOR_GLOBAL_ACTIONABILITY_MODEL_CONTRACT, supported: true, default_enabled: false, precise_default_enabled: true, experimental: true, blocks_use: false}}};",
                _extract_js_function(js, "embeddingRecipePresetValues"),
                _extract_js_function(js, "getClassSplitRefinementCapability"),
                _extract_js_function(js, "classSplitRefinementDefaultForPreset"),
                _extract_js_function_before(
                    js,
                    "resolveClassSplitRefinementPreferenceForPreset",
                "function applyClassSplitRefinementPresetRecommendation",
            ),
            "assert.strictEqual(embeddingRecipePresetValues('precise').encoderType, 'dinov3');",
            "assert.strictEqual(embeddingRecipePresetValues('balanced').encoderType, 'dinov3');",
            "assert.strictEqual(embeddingRecipePresetValues('fast').encoderType, 'dinov3');",
            "assert.strictEqual(embeddingRecipePresetValues('cradio').encoderType, 'cradio');",
                "assert.strictEqual(embeddingRecipePresetValues('thorough_quality_v1').embeddingAggregation, 'sam3_mask_salad_fusion_v1');",
                "assert.strictEqual(embeddingRecipePresetValues('precise_compact_v1').embeddingAggregation, 'sam3_mask_salad_fusion_v1');",
                "assert.strictEqual(embeddingRecipePresetValues('fast_map_v1').embeddingAggregation, 'pooled');",
                "assert.strictEqual(resolveClassSplitRefinementPreferenceForPreset('precise'), false);",
                "assert.strictEqual(resolveClassSplitRefinementPreferenceForPreset('thorough_quality_v1'), false);",
                "assert.strictEqual(resolveClassSplitRefinementPreferenceForPreset('balanced'), false);",
            "assert.strictEqual(resolveClassSplitRefinementPreferenceForPreset('fast'), false);",
            "assert.strictEqual(resolveClassSplitRefinementPreferenceForPreset('cradio'), false);",
                "assert.strictEqual(resolveClassSplitRefinementPreferenceForPreset('custom', { preference: true }), true);",
                "assert.strictEqual(resolveClassSplitRefinementPreferenceForPreset('balanced', { touched: true, preference: true }), true);",
                "classSplitState.capabilities.fine_grained_refinement.precise_default_enabled = false;",
                "assert.strictEqual(resolveClassSplitRefinementPreferenceForPreset('precise'), false);",
                "classSplitState.capabilities.fine_grained_refinement.api_version = 2;",
                "assert.strictEqual(getClassSplitRefinementCapability(), null);",
                "delete classSplitState.capabilities.fine_grained_refinement.api_version;",
                "classSplitState.capabilities.fine_grained_refinement.version = 3;",
                "assert.strictEqual(getClassSplitRefinementCapability(), null);",
                "classSplitState.capabilities.fine_grained_refinement.api_version = 4;",
                "classSplitState.capabilities.fine_grained_refinement.schema = 'class-analysis-patch-refinement-v2';",
                "assert.strictEqual(getClassSplitRefinementCapability(), null);",
                "classSplitState.capabilities.fine_grained_refinement.schema = CLASS_SPLIT_REFINEMENT_SCHEMA;",
                "classSplitState.capabilities.fine_grained_refinement.decision_contract = 'class-analysis-patch-decision-v3';",
                "assert.strictEqual(getClassSplitRefinementCapability(), null);",
            ]
        )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)

    assert "recipePreset: explorerInitialized" in js
    assert "recipeValues: explorerInitialized" in js
    restore = _extract_js_function(js, "restoreDataQualityExplorerSession")
    assert "applyEmbeddingRecipePresetToClassSplit(recipePreset)" in restore
    assert "restoreClassSplitSessionRecipeValues(saved.recipeValues)" in restore
    assert "resolveClassSplitRefinementPreferenceForPreset(" in restore


def test_class_split_v33_probe_evidence_is_labeled_without_raw_margin_overclaim():
    js = _js()
    css = _css()
    wrong_list = _extract_js_function(js, "renderClassSplitWrongList")
    selector_priority = _extract_js_function(
        js, "getClassSplitCandidateSelectorPriority"
    )
    selector_priority_validation = _extract_js_function(
        js, "classSplitResultHasCompleteSelectorPriorityRanks"
    )
    quality_banner = _extract_js_function(
        js, "renderClassSplitRefinementQualityBanner"
    )

    for field in (
        "intrinsic_current_support",
        "intrinsic_alternative_support",
        "directed_pair_raw_margin",
        "directed_pair_probe_score",
        "directed_pair_probe_features",
        "directed_pair_probe_feature_names",
        "directed_pair_current_exclusive_support",
        "directed_pair_alternative_exclusive_support",
        "directed_pair_probe_threshold",
        "directed_pair_probe_weights",
        "directed_pair_probe_contract",
        "directed_pair_probe_fold_count",
        "directed_pair_probe_fit_status",
        "directed_pair_probe_fold_digest",
        "directed_pair_heldout_auroc",
        "directed_pair_eval_auroc_lower_bound",
        "positive_confirmation_pair_probe_auroc_floor",
        "positive_confirmation_pair_probe_auroc_lower_bound_floor",
        "directed_pair_probe_view_contract",
        "directed_pair_probe_lower_bound_contract",
        "current_negative_threshold",
        "current_support_threshold",
        "current_strong_threshold",
        "alternative_negative_threshold",
        "alternative_support_threshold",
        "alternative_strong_threshold",
        "support_threshold_source",
        "current_exclusive_component_corresponds",
        "alternative_exclusive_component_corresponds",
        "exclusive_components_spatially_separated",
        "selector_priority_score",
        "selector_priority_semantic_overlap_adjustment",
        "selector_priority_triage_frequency_adjustment",
        "selector_priority_overlap_adjustment",
        "frequent_overlap_prior",
    ):
        assert field in wrong_list
    assert "selector_priority_rank" in selector_priority
    assert "selector_v6" in selector_priority
    assert "expected_review_utility" in selector_priority_validation
    assert "actionable_probability" in selector_priority_validation
    assert "reviewability_probability" in selector_priority_validation
    assert "conditional_annotation_state" in selector_priority_validation
    assert "current_evidence_state" in selector_priority_validation
    assert "alternative_evidence_state" in selector_priority_validation
    assert "overlap_evidence_state" in selector_priority_validation
    assert "global_model" in selector_priority_validation
    assert "dataset_overlap" in selector_priority_validation
    assert "scoring_effect_enabled" in selector_priority_validation
    for stale_field in (
        "directed_pair_probe_eval_auroc_lower_bound",
        "alternative_component_view_correspondence",
        "current_exclusive_component_view_correspondence",
        "mixed_spatially_separated",
        "current_exclusive_evidence_observable",
    ):
        assert stale_field not in wrong_list
    assert "audit only; not the decision score" in wrong_list
    assert "directed-pair probe score" in wrong_list
    assert "paired exclusive probe inputs" in wrong_list
    assert "distinct from raw intrinsic supports" in wrong_list
    assert "positive-confirmation AUROC floor" in wrong_list
    assert "held-out eval AUROC" in wrong_list
    assert "held-out AUROC lower bound" in wrong_list
    assert "one source-disjoint fit/eval split" in wrong_list
    assert "fit-only pair thresholds" in wrong_list
    assert "intrinsic spatial thresholds are calibrated separately on held-out sources" in wrong_list
    assert "exclusive-component geometry" in wrong_list
    assert "fit eligibility=" in wrong_list
    assert "prior contract=" in wrong_list
    assert "selectorV6Active" in wrong_list
    assert "Review value" in wrong_list
    assert "actionable" in wrong_list
    assert "reviewability" in wrong_list
    assert "Expected review value" in wrong_list
    assert "Visual evidence state" in wrong_list
    assert "global HGB actionability model" in wrong_list
    assert "dataset-overlap evidence" in wrong_list
    assert "ranking contribution=" in wrong_list
    assert "&& !selectorV6Active" in wrong_list
    assert "Technical details" in wrong_list
    assert "Review-ranking evidence" in wrong_list
    assert "Patch and calibration evidence" in wrong_list
    assert "Decision reasons" in wrong_list
    assert "Review value combines estimated actionability" in wrong_list
    assert "not the probability that this label is wrong" in wrong_list
    assert "Common overlap pattern" in wrong_list
    assert "used in ranking" in wrong_list
    assert "overflow-wrap: anywhere;" in css
    assert "word-break: break-word;" in css
    assert ".class-split-wrong-item__technical > summary" in css
    assert "font-size: 13px;" in css
    assert ".class-split-wrong-item__dataset-overlap" in css
    assert (
        "html.theme-dark .class-split-wrong-item__body > "
        ".class-split-wrong-item__dataset-overlap"
    ) in css
    assert "qualityMetrics.resolved_rate" not in quality_banner
    assert "qualityMetrics.confirmation_eligible_pair_coverage" not in quality_banner
    assert "Dataset-overlap evidence was unavailable" in quality_banner
    assert "fresh locked human audit" not in quality_banner
    assert "failed its usefulness gate" not in quality_banner


def test_class_split_selector_priority_requires_a_completed_bound_run():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const CLASS_SPLIT_REFINEMENT_SCHEMA = 'class-analysis-patch-refinement-v5';",
            "const CLASS_SPLIT_REFINEMENT_DECISION_CONTRACT = 'class-analysis-patch-decision-v9';",
            "const CLASS_SPLIT_SELECTOR_PRIORITY_CONTRACT = 'expected-review-utility-global-boosted-statistical-overlap-rerank-v7';",
            "const CLASS_SPLIT_SELECTOR_BASE_MODEL_CONTRACT = 'expected-review-utility-global-boosted-gated-dataset-overlap-v6';",
            "const CLASS_SPLIT_SELECTOR_FEATURE_CONTRACT = 'raw-stage1-patch-same-image-and-gated-dataset-overlap-features-v3';",
            "const CLASS_SPLIT_SELECTOR_MODEL_SCHEMA = 'class-analysis-selector-utility-model-v2';",
            "const CLASS_SPLIT_SELECTOR_UTILITY_POLICY_CONTRACT = 'actionability-times-75pct-base-plus-25pct-reviewability-v1';",
            "const CLASS_SPLIT_SELECTOR_DATASET_OVERLAP_APPLICATION_CONTRACT = 'affirmative-current-localized-material-source-loo-v2';",
            "const CLASS_SPLIT_SELECTOR_DATASET_OVERLAP_DIAGNOSTIC_CONTRACT = 'capture-loo-beta-wilson-shrunk-rank-only-v1';",
            "const CLASS_SPLIT_SELECTOR_MAXIMUM_RANK_DISCOUNT_FRACTION = 0.25;",
            "const CLASS_SPLIT_SELECTOR_GLOBAL_ACTIONABILITY_MODEL_CONTRACT = 'pair-blind-shallow-histogram-gradient-boosting-v1';",
            "const CLASS_SPLIT_SELECTOR_BASE_ORDER_CONTRACT = 'expected-review-utility-before-statistical-overlap-then-point-id-v1';",
            "const CLASS_SPLIT_SELECTOR_BASE_SCORE_CONTRACT = 'expected-review-utility-before-statistical-overlap-v1';",
            "const CLASS_SPLIT_SELECTOR_SEMANTIC_TIEBREAK_CONTRACT = 'none-global-expected-utility-order-v1';",
            "const CLASS_SPLIT_SELECTOR_CURRENT_EVIDENCE_STATES = Object.freeze(['present', 'absent', 'indeterminate', 'unavailable']);",
            "const CLASS_SPLIT_SELECTOR_OVERLAP_EVIDENCE_STATES = Object.freeze(['none', 'localized', 'external', 'uncertain', 'duplicate_conflict']);",
            "const CLASS_SPLIT_SELECTOR_DATASET_OVERLAP_APPLICATION_REASONS = Object.freeze(['eligible_dataset_overlap_explanation', 'material_annotated_overlap_absent']);",
            "const modelDigest = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';",
            "const makeCandidate = ({pointId, rank, actionability, reviewability, currentState, overlapApplicable}) => {",
            "  const utility = actionability * (0.75 + 0.25 * reviewability);",
            "  const mislabeled = actionability * 0.625;",
            "  const geometry = actionability - mislabeled;",
            "  return {point_id: pointId, refined_outlier: {",
            "    schema: CLASS_SPLIT_REFINEMENT_SCHEMA,",
            "    decision_contract: CLASS_SPLIT_REFINEMENT_DECISION_CONTRACT,",
            "    selector_priority_contract: CLASS_SPLIT_SELECTOR_PRIORITY_CONTRACT,",
            "    selector_priority_rank: rank, selector_priority_base_rank: rank,",
            "    selector_priority_score: utility, selector_priority_base_score: utility,",
            "    selector_priority_overlap_adjustment: 0,",
            "    selector_priority_semantic_overlap_adjustment: 0,",
            "    selector_priority_triage_frequency_adjustment: 0,",
            "    selector_priority_status_band_index: 0,",
            "    selector_priority_status_band_name: 'expected_review_utility',",
            "    selector_priority_band_base_rank: rank, selector_priority_band_rank: rank,",
            "    selector_priority_band_candidate_count: 2,",
            "    selector_priority_base_components: {",
            "      actionable_probability: actionability, reviewability_probability: reviewability,",
            "      utility_reviewability_floor: 0.75, utility_reviewability_weight: 0.25,",
            "      base_expected_review_utility: utility,",
            "    },",
            "    selector_priority_reasons: ['ranked_by_learned_expected_human_review_utility'],",
            "    selector_v6: {",
            "      selector_contract: CLASS_SPLIT_SELECTOR_PRIORITY_CONTRACT,",
            "      base_model_selector_contract: CLASS_SPLIT_SELECTOR_BASE_MODEL_CONTRACT,",
            "      feature_contract: CLASS_SPLIT_SELECTOR_FEATURE_CONTRACT,",
            "      model_digest: modelDigest,",
            "      utility_policy_contract: CLASS_SPLIT_SELECTOR_UTILITY_POLICY_CONTRACT,",
            "      dataset_overlap_application_contract: CLASS_SPLIT_SELECTOR_DATASET_OVERLAP_APPLICATION_CONTRACT,",
            "      dataset_overlap_diagnostic_contract: CLASS_SPLIT_SELECTOR_DATASET_OVERLAP_DIAGNOSTIC_CONTRACT,",
            "      dataset_overlap_scoring_effect_enabled: true,",
            "      global_actionability_model_contract: CLASS_SPLIT_SELECTOR_GLOBAL_ACTIONABILITY_MODEL_CONTRACT,",
            "      base_expected_review_utility: utility, expected_review_utility: utility, actionable_probability: actionability,",
            "      reviewability_probability: reviewability,",
            "      insufficient_evidence_probability: 1 - reviewability,",
            "      conditional_annotation_state: {",
            "        mislabeled, actionable_geometry_or_composite: geometry,",
            "        valid_or_harmless: 1 - actionability,",
            "      },",
            "      current_evidence_state: currentState, alternative_evidence_state: 'present',",
            "      overlap_evidence_state: 'localized',",
            "      same_image_context: {available: true, image_object_count: 4, same_class_count: 2},",
            "      global_model: {contract: CLASS_SPLIT_SELECTOR_GLOBAL_ACTIONABILITY_MODEL_CONTRACT, raw_margin: 0.4, actionable_probability: actionability},",
            "      dataset_overlap: {",
            "        application_contract: CLASS_SPLIT_SELECTOR_DATASET_OVERLAP_APPLICATION_CONTRACT,",
            "        diagnostic_contract: CLASS_SPLIT_SELECTOR_DATASET_OVERLAP_DIAGNOSTIC_CONTRACT,",
            "        available: true, applicable: overlapApplicable,",
            "        application_reason: overlapApplicable ? 'eligible_dataset_overlap_explanation' : 'material_annotated_overlap_absent',",
            "        scoring_effect_enabled: true, rank_only: true, uses_human_review_labels: false, applied: false,",
            "        counterfactual_actionable_probability: actionability, actionable_probability: actionability, probability_delta: 0,",
            "        base_expected_review_utility: utility, counterfactual_expected_review_utility: utility, expected_review_utility: utility, utility_delta: 0,",
            "        maximum_rank_discount_fraction: 0.25, rank_discount_fraction: 0,",
            "      },",
            "    },",
            "  }};",
            "};",
            "const candidates = [",
            "  makeCandidate({pointId: 'point-1', rank: 1, actionability: 0.8, reviewability: 0.5, currentState: 'present', overlapApplicable: true}),",
            "  makeCandidate({pointId: 'point-2', rank: 2, actionability: 0.6, reviewability: 1, currentState: 'indeterminate', overlapApplicable: false}),",
            "];",
            "const refinement = {",
            "  enabled: true, status: 'completed', schema: CLASS_SPLIT_REFINEMENT_SCHEMA,",
            "  decision_contract: CLASS_SPLIT_REFINEMENT_DECISION_CONTRACT,",
            "  selector_priority_contract: CLASS_SPLIT_SELECTOR_PRIORITY_CONTRACT,",
            "  selector_priority_candidate_count: 2,",
            "  selector_priority: {",
            "    contract: CLASS_SPLIT_SELECTOR_PRIORITY_CONTRACT, candidate_count: 2, ranked_candidate_count: 2,",
            "    unique_contiguous_ranks: true, higher_score_is_higher_priority: true,",
            "    changes_candidate_membership: false, changes_semantic_status: false, suppresses_candidates: false,",
            "    status_band_partitioned: false, cross_status_band_reordering: true,",
            "    base_order_contract: CLASS_SPLIT_SELECTOR_BASE_ORDER_CONTRACT,",
            "    base_score_contract: CLASS_SPLIT_SELECTOR_BASE_SCORE_CONTRACT,",
            "    semantic_status_tiebreak_contract: CLASS_SPLIT_SELECTOR_SEMANTIC_TIEBREAK_CONTRACT,",
            "    status_band_score_gap: 0, maximum_overlap_adjustment_bound: 0.25,",
            "    status_band_order: ['expected_review_utility'],",
            "    status_band_counts: {expected_review_utility: 2},",
            "    dataset_overlap_applied_candidate_count: 0,",
            "    utility_model: {",
            "      contract: CLASS_SPLIT_SELECTOR_PRIORITY_CONTRACT,",
            "      base_model_selector_contract: CLASS_SPLIT_SELECTOR_BASE_MODEL_CONTRACT,",
            "      feature_contract: CLASS_SPLIT_SELECTOR_FEATURE_CONTRACT,",
            "      model_schema: CLASS_SPLIT_SELECTOR_MODEL_SCHEMA, model_digest: modelDigest,",
            "      utility_policy_contract: CLASS_SPLIT_SELECTOR_UTILITY_POLICY_CONTRACT,",
            "      dataset_overlap_application_contract: CLASS_SPLIT_SELECTOR_DATASET_OVERLAP_APPLICATION_CONTRACT,",
            "      dataset_overlap_diagnostic_contract: CLASS_SPLIT_SELECTOR_DATASET_OVERLAP_DIAGNOSTIC_CONTRACT,",
            "      dataset_overlap_scoring_effect_enabled: true,",
            "      global_actionability_model_contract: CLASS_SPLIT_SELECTOR_GLOBAL_ACTIONABILITY_MODEL_CONTRACT,",
            "      candidate_count: 2, context_available_count: 2,",
            "      current_evidence_state_counts: {present: 1, indeterminate: 1},",
            "      dataset_overlap: {",
            "        application_contract: CLASS_SPLIT_SELECTOR_DATASET_OVERLAP_APPLICATION_CONTRACT, diagnostic_contract: CLASS_SPLIT_SELECTOR_DATASET_OVERLAP_DIAGNOSTIC_CONTRACT, scoring_effect_enabled: true, rank_only: true, uses_human_review_labels: false, maximum_rank_discount_fraction: 0.25,",
            "        available_candidate_count: 2, applicable_candidate_count: 1, applied_candidate_count: 0, effect_candidate_count: 0,",
            "        application_reason_counts: {eligible_dataset_overlap_explanation: 1, material_annotated_overlap_absent: 1},",
            "        demoted_candidate_count: 0, promoted_candidate_count: 0, zero_effect_candidate_count: 2,",
            "        maximum_absolute_probability_effect: 0, mean_absolute_probability_effect: 0,",
            "        maximum_absolute_utility_effect: 0, mean_absolute_utility_effect: 0,",
            "      },",
            "      changes_candidate_membership: false, changes_semantic_status: false, mutates_annotations: false,",
            "    },",
            "  },",
            "};",
            "const classSplitState = {result: {summary: {refinement}}};",
            "function getClassSplitPointById(pointId) { return candidates.find((row) => row.point_id === pointId) || null; }",
            "function getClassSplitRawCandidates() { return candidates; }",
            "function getClassSplitCandidatePriorityHumanReviewRank() { return 99; }",
            _extract_js_function(js, "getClassSplitCandidateSelectorPriority"),
            _extract_js_function(js, "classSplitResultHasCompleteSelectorPriorityRanks"),
            _extract_js_function(js, "getClassSplitSelectorPriorityUnavailableReason"),
            "assert.strictEqual(classSplitResultHasCompleteSelectorPriorityRanks(), true);",
            "classSplitState.result.summary.analysis_scope = 'all_classes';",
            "classSplitState.result.wrong_class_candidates = candidates;",
            "classSplitState.result.refinement_candidates = [candidates[0]];",
            "assert.strictEqual(classSplitResultHasCompleteSelectorPriorityRanks(), false);",
            "classSplitState.result.refinement_candidates = [...candidates];",
            "assert.strictEqual(classSplitResultHasCompleteSelectorPriorityRanks(), true);",
            "assert.strictEqual(getClassSplitCandidateSelectorPriority(candidates[0]).rank, 1);",
            "assert.strictEqual(getClassSplitCandidateSelectorPriority(candidates[0]).source, 'selector');",
            "assert.ok(Math.abs(getClassSplitCandidateSelectorPriority(candidates[0]).expectedReviewUtility - 0.7) < 1e-12);",
            "for (const status of ['failed', 'partial', 'cancelled']) {",
            "  refinement.status = status;",
            "  assert.strictEqual(classSplitResultHasCompleteSelectorPriorityRanks(), false);",
            "  assert.strictEqual(getClassSplitCandidateSelectorPriority(candidates[0]), null);",
            "  assert.strictEqual(getClassSplitSelectorPriorityUnavailableReason(), `refinement ${status}`);",
            "}",
            "refinement.status = 'completed';",
            "refinement.selector_priority.suppresses_candidates = true;",
            "assert.strictEqual(classSplitResultHasCompleteSelectorPriorityRanks(), false);",
            "refinement.selector_priority.suppresses_candidates = false;",
            "candidates[0].refined_outlier.selector_priority_rank = 2;",
            "candidates[0].refined_outlier.selector_priority_band_rank = 2;",
            "candidates[1].refined_outlier.selector_priority_rank = 1;",
            "candidates[1].refined_outlier.selector_priority_band_rank = 1;",
            "assert.strictEqual(classSplitResultHasCompleteSelectorPriorityRanks(), false);",
            "assert.strictEqual(getClassSplitSelectorPriorityUnavailableReason(), 'the saved priority data does not match every candidate in this analysis');",
            "candidates[0].refined_outlier.selector_priority_rank = 1;",
            "candidates[0].refined_outlier.selector_priority_band_rank = 1;",
            "candidates[1].refined_outlier.selector_priority_rank = 2;",
            "candidates[1].refined_outlier.selector_priority_band_rank = 2;",
            "candidates[1].refined_outlier.selector_v6.expected_review_utility = 0.59;",
            "assert.strictEqual(classSplitResultHasCompleteSelectorPriorityRanks(), false);",
            "candidates[1].refined_outlier.selector_v6.expected_review_utility = 0.6;",
            "candidates[1].refined_outlier.selector_v6.current_evidence_state = 'maybe';",
            "assert.strictEqual(classSplitResultHasCompleteSelectorPriorityRanks(), false);",
            "candidates[1].refined_outlier.selector_v6.current_evidence_state = 'indeterminate';",
            "candidates[1].refined_outlier.selector_v6.model_digest = 'b'.repeat(64);",
            "assert.strictEqual(classSplitResultHasCompleteSelectorPriorityRanks(), false);",
            "candidates[1].refined_outlier.selector_v6.model_digest = modelDigest;",
            "candidates[1].refined_outlier.selector_v6.dataset_overlap.scoring_effect_enabled = false;",
            "assert.strictEqual(classSplitResultHasCompleteSelectorPriorityRanks(), false);",
            "candidates[1].refined_outlier.selector_v6.dataset_overlap.scoring_effect_enabled = true;",
            "const staleV5 = candidates[1].refined_outlier.selector_v6;",
            "candidates[1].refined_outlier.selector_v5 = staleV5;",
            "delete candidates[1].refined_outlier.selector_v6;",
            "assert.strictEqual(classSplitResultHasCompleteSelectorPriorityRanks(), false);",
            "candidates[1].refined_outlier.selector_v6 = staleV5;",
            "delete candidates[1].refined_outlier.selector_v5;",
            "candidates[1].refined_outlier.selector_priority_contract = 'stale-selector';",
            "assert.strictEqual(classSplitResultHasCompleteSelectorPriorityRanks(), false);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_class_split_overlap_prior_copy_respects_capture_group_provenance():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const CLASS_SPLIT_CAPTURE_GROUP_CONTRACT = 'explicit-exporter-sequence-perceptual-cluster-tiered-independence-v2';",
            "const CLASS_SPLIT_FREQUENT_OVERLAP_PRIOR_CONTRACT = 'capture-aware-directed-class-overlap-trusted-label-stratified-beta-smoothed-loo-v4';",
            "const CLASS_SPLIT_FREQUENT_OVERLAP_TRIAGE_CONTRACT = 'review-unresolved-annotated-overlap-frequency-rank-only-v2';",
            "const CLASS_SPLIT_FREQUENT_OVERLAP_FIT_ELIGIBILITY_CONTRACT = 'exclude-stage1-rough-labels-both-directed-roles-v1';",
            _extract_js_function(js, "getClassSplitOverlapPriorPresentation"),
            _extract_js_function(js, "getClassSplitTriageFrequencyPresentation"),
            "const base = {contract: CLASS_SPLIT_FREQUENT_OVERLAP_PRIOR_CONTRACT, capture_group_contract: CLASS_SPLIT_CAPTURE_GROUP_CONTRACT, fit_eligibility_contract: CLASS_SPLIT_FREQUENT_OVERLAP_FIT_ELIGIBILITY_CONTRACT};",
            "const strong = getClassSplitOverlapPriorPresentation({...base, candidate_capture_group_tier: 'provisional_unlineaged', reliability_tier: 'strong', source_independence_verified: true, provisional: false}, 0.615);",
            "assert.strictEqual(strong.tier, 'strong');",
            "assert.strictEqual(strong.evidence, 'observed across 62% of independent capture groups');",
            "assert.ok(strong.title.includes('source-independent capture groups'));",
            "assert.ok(strong.title.includes(\"does not establish the candidate image's lineage\"));",
            "assert.ok(strong.title.includes('excludes Stage-1 rough candidates from both annotation roles'));",
            "const perceptual = getClassSplitOverlapPriorPresentation({...base, candidate_capture_group_tier: 'lower_confidence', reliability_tier: 'lower_confidence', source_independence_verified: false, provisional: false}, 0.4);",
            "assert.strictEqual(perceptual.tier, 'lower_confidence');",
            "assert.strictEqual(perceptual.evidence, 'observed across 40% of lower-confidence visual groups (perceptual)');",
            "assert.ok(perceptual.title.includes('source independence is not established'));",
            "assert.ok(!perceptual.title.includes('source-independent'));",
            "const provisional = getClassSplitOverlapPriorPresentation({...base, candidate_capture_group_tier: 'provisional_unlineaged', reliability_tier: 'strong', source_independence_verified: true, provisional: true}, 0.25);",
            "assert.strictEqual(provisional.tier, 'provisional_unlineaged');",
            "assert.strictEqual(provisional.evidence, 'observed across 25% of provisional images (heuristic)');",
            "assert.ok(!provisional.evidence.includes('independent'));",
            "assert.ok(!provisional.title.includes('source-independent'));",
            "assert.ok(provisional.title.includes('source independence is not established'));",
            "const inconsistent = getClassSplitOverlapPriorPresentation({...base, reliability_tier: 'strong', source_independence_verified: false, provisional: false}, 0.5);",
            "assert.strictEqual(inconsistent.tier, 'unresolved_provenance');",
            "assert.ok(!inconsistent.evidence.includes('independent'));",
            "const staleFit = getClassSplitOverlapPriorPresentation({...base, fit_eligibility_contract: 'legacy-fit', reliability_tier: 'strong', source_independence_verified: true, provisional: false}, 0.5);",
            "assert.strictEqual(staleFit.tier, 'unresolved_provenance');",
            "assert.ok(staleFit.title.includes('contracts or capture provenance'));",
            "assert.ok(!inconsistent.title.includes('source-independent'));",
            "const legacy = getClassSplitOverlapPriorPresentation({reliability_tier: 'strong', source_independence_verified: true}, 0.5);",
            "assert.strictEqual(legacy.tier, 'unresolved_provenance');",
            "assert.ok(!legacy.evidence.includes('independent'));",
            "assert.ok(legacy.title.includes('source independence is not established'));",
            "const triageStrong = getClassSplitTriageFrequencyPresentation({...base, triage_contract: CLASS_SPLIT_FREQUENT_OVERLAP_TRIAGE_CONTRACT, reliability_tier: 'none', source_independence_verified: false, provisional: true, triage_reliability_tier: 'strong', triage_source_independence_verified: true, triage_provisional: false, triage_smoothed_capture_group_incidence: 0.615});",
            "assert.strictEqual(triageStrong.tier, 'strong');",
            "assert.strictEqual(triageStrong.evidence, 'observed across 62% of independent capture groups');",
            "assert.ok(triageStrong.title.includes('shared human-review band only'));",
            "const triageSeparate = getClassSplitTriageFrequencyPresentation({...base, triage_contract: CLASS_SPLIT_FREQUENT_OVERLAP_TRIAGE_CONTRACT, reliability_tier: 'strong', source_independence_verified: true, provisional: false, triage_reliability_tier: 'lower_confidence', triage_source_independence_verified: false, triage_provisional: false, triage_smoothed_capture_group_incidence: 0.4});",
            "assert.strictEqual(triageSeparate.tier, 'lower_confidence');",
            "assert.strictEqual(triageSeparate.evidence, 'observed across 40% of lower-confidence visual groups (perceptual)');",
            "const staleTriage = getClassSplitTriageFrequencyPresentation({...base, triage_contract: 'legacy-triage', triage_reliability_tier: 'strong', triage_source_independence_verified: true, triage_provisional: false, triage_smoothed_capture_group_incidence: 0.5});",
            "assert.strictEqual(staleTriage.tier, 'unresolved_provenance');",
            "assert.ok(!staleTriage.evidence.includes('independent'));",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)

    wrong_list = _extract_js_function(js, "renderClassSplitWrongList")
    assert "smoothed_capture_group_incidence" in wrong_list
    assert "raw_capture_group_incidence" in wrong_list
    assert "capture_group_incidence_wilson_lower_bound" in wrong_list
    assert "eligible_capture_group_count" in wrong_list
    assert "overlap_capture_group_count" in wrong_list
    assert "% of independent sources" not in wrong_list


def test_class_split_priority_sort_orders_the_complete_visible_rough_queue():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const candidates = [",
            "  {point_id: 'analysis-first', wrong_class_suspicion: 0.90, rank: 3},",
            "  {point_id: 'priority-first', wrong_class_suspicion: 0.20, rank: 1},",
            "  {point_id: 'priority-second', wrong_class_suspicion: 0.50, rank: 2},",
            "];",
            "const classSplitElements = {filterClass: {value: ''}};",
            "const classSplitState = {dismissedWrongIds: new Set()};",
            "let complete = true;",
            "function getClassSplitVignetteCandidatePool() { return candidates; }",
            "function getClassSplitPointById() { return null; }",
            "function getClassSplitVignetteSort() { return 'priority'; }",
            "function getClassSplitRawCandidates() { return candidates; }",
            "function classSplitResultHasCompleteSelectorPriorityRanks() { return complete; }",
            "function getClassSplitCandidateSelectorPriority(candidate) { return {rank: candidate.rank, source: 'selector'}; }",
            "function normalizeClassSplitRefinementStatus() { return 'unresolved'; }",
            _extract_js_function(js, "getClassSplitVisibleWrongCandidates"),
            _extract_js_function(js, "sortClassSplitPriorityHumanReviewCandidates"),
            "assert.deepStrictEqual(getClassSplitVisibleWrongCandidates().map((row) => row.point_id), [",
            "  'priority-first', 'priority-second', 'analysis-first',",
            "]);",
            "assert.deepStrictEqual(sortClassSplitPriorityHumanReviewCandidates(candidates).map((row) => row.point_id), [",
            "  'priority-first', 'priority-second', 'analysis-first',",
            "]);",
            "complete = false;",
            "assert.deepStrictEqual(getClassSplitVisibleWrongCandidates().map((row) => row.point_id), [",
            "  'analysis-first', 'priority-second', 'priority-first',",
            "]);",
            "assert.deepStrictEqual(sortClassSplitPriorityHumanReviewCandidates(candidates).map((row) => row.point_id), [",
            "  'analysis-first', 'priority-second', 'priority-first',",
            "]);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)

    wrong_list = _extract_js_function(js, "renderClassSplitWrongList")
    assert 'selectorPriorityAvailable\n                && candidatePriority?.source === "selector"' in wrong_list
    assert "frequentOverlapPrior && selectorPriorityAvailable" in wrong_list
    assert "Detailed ranking data is incomplete for this analysis" in wrong_list


def test_class_split_likely_wrong_sorts_are_explicit_and_bidirectional():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const candidates = [",
            "  {point_id: 'middle', wrong_class_suspicion: 0.99, refined_outlier: {selector_v6: {conditional_annotation_state: {mislabeled: 0.50}}}},",
            "  {point_id: 'least', wrong_class_suspicion: 0.90, refined_outlier: {selector_v6: {conditional_annotation_state: {mislabeled: 0.10}}}},",
            "  {point_id: 'most', wrong_class_suspicion: 0.01, refined_outlier: {selector_v6: {conditional_annotation_state: {mislabeled: 0.90}}}},",
            "];",
            "const classSplitElements = {filterClass: {value: ''}};",
            "const classSplitState = {dismissedWrongIds: new Set()};",
            "let sortMode = 'suspicion';",
            "function getClassSplitVignetteCandidatePool() { return candidates; }",
            "function getClassSplitPointById() { return null; }",
            "function getClassSplitVignetteSort() { return sortMode; }",
            "function getClassSplitRawCandidates() { return candidates; }",
            "let selectorArtifactComplete = true;",
            "function classSplitResultHasCompleteSelectorPriorityRanks() { return selectorArtifactComplete; }",
            "function getClassSplitCandidateSelectorPriority() { return null; }",
            _extract_js_function(js, "getClassSplitVisibleWrongCandidates"),
            "assert.deepStrictEqual(getClassSplitVisibleWrongCandidates().map((row) => row.point_id), [",
            "  'most', 'middle', 'least',",
            "]);",
            "sortMode = 'suspicion_ascending';",
            "assert.deepStrictEqual(getClassSplitVisibleWrongCandidates().map((row) => row.point_id), [",
            "  'least', 'middle', 'most',",
            "]);",
            "selectorArtifactComplete = false;",
            "sortMode = 'suspicion';",
            "assert.deepStrictEqual(getClassSplitVisibleWrongCandidates().map((row) => row.point_id), [",
            "  'middle', 'least', 'most',",
            "]);",
            "sortMode = 'suspicion_ascending';",
            "assert.deepStrictEqual(getClassSplitVisibleWrongCandidates().map((row) => row.point_id), [",
            "  'most', 'least', 'middle',",
            "]);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_class_split_completed_ranking_does_not_show_obsolete_quality_gate_warning():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const CLASS_SPLIT_REFINEMENT_DECISION_CONTRACT = 'class-analysis-patch-decision-v9';",
            "const classSplitElements = { refinementQualityBanner: { hidden: true, innerHTML: '' } };",
            "const classSplitState = { result: { summary: { refinement: {",
            "  enabled: true, status: 'completed', quality_status: 'completed_non_actionable',",
            "  rough_candidate_count: 100, vignette_candidate_count: 0,",
            "  quality_gate: { passed: false, reasons: ['resolved_rate_below_release_gate', 'confirmation_eligible_pair_coverage_below_release_gate'], metrics: { resolved_rate: 0.42, confirmation_eligible_pair_coverage: 0.68 } },",
            "  queue_policy: { automatic_rough_fallback: true, rough_count: 100, confirmed_count: 0 },",
            "} } } };",
            _extract_js_function(js, "classSplitResultHasRefinement"),
            _extract_js_function(js, "getClassSplitRefinementQualityStatus"),
            _extract_js_function(js, "escapeHtml"),
            _extract_js_function(js, "renderClassSplitRefinementQualityBanner"),
            "renderClassSplitRefinementQualityBanner();",
            "assert.strictEqual(classSplitElements.refinementQualityBanner.hidden, true);",
            "const text = classSplitElements.refinementQualityBanner.innerHTML;",
            "assert.strictEqual(text, '');",
            "assert.ok(!text.toLowerCase().includes('release gate'));",
            "assert.ok(!text.includes('failed its usefulness gate'));",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_class_split_refinement_warning_surfaces_prior_failures_and_scope_ineligibility():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const CLASS_SPLIT_REFINEMENT_DECISION_CONTRACT = 'class-analysis-patch-decision-v9';",
            "const classSplitElements = { refinementQualityBanner: { hidden: true, innerHTML: '', textContent: '' } };",
            "const prior = {fit_screening_scope: 'all_classes', fit_screening_exhaustive: true, fit_screening_adjustment_eligible: true, fit_screening_quality_gate: {passed: true, reason: 'exhaustive_all_classes', ordering_adjustments_enabled: true}};",
            "const refinement = {enabled: true, status: 'completed', quality_status: 'actionable', decision_contract: CLASS_SPLIT_REFINEMENT_DECISION_CONTRACT, queue_policy: {mode: 'selector_ranked_complete_stage1', rough_count: 10, confirmed_count: 1}, selector_priority: {prior_evaluation_failure_count: 2, frequent_overlap_prior: prior}};",
            "const classSplitState = {result: {summary: {refinement}}};",
            "function getClassSplitRefinementQualityStatus() { return refinement.quality_status; }",
            _extract_js_function(js, "escapeHtml"),
            _extract_js_function(js, "renderClassSplitRefinementQualityBanner"),
            "renderClassSplitRefinementQualityBanner();",
            "assert.strictEqual(classSplitElements.refinementQualityBanner.hidden, false);",
            "assert.ok(classSplitElements.refinementQualityBanner.innerHTML.includes('Dataset-overlap evidence was unavailable for 2 candidates'));",
            "assert.ok(classSplitElements.refinementQualityBanner.innerHTML.includes('ranked with the remaining available signals'));",
            "assert.ok(classSplitElements.refinementQualityBanner.innerHTML.includes('Overlap-context feature failures: 2'));",
            "assert.ok(classSplitElements.refinementQualityBanner.innerHTML.includes('<details class=\"class-split-refinement-quality-banner__technical\">'));",
            "assert.ok(!classSplitElements.refinementQualityBanner.innerHTML.includes('<details open'));",
            "refinement.selector_priority.prior_evaluation_failure_count = 0;",
            "Object.assign(prior, {fit_screening_scope: 'selected_class', fit_screening_exhaustive: true, fit_screening_adjustment_eligible: false, fit_screening_quality_gate: {passed: false, reason: 'screening_scope_ineligible', ordering_adjustments_enabled: false}});",
            "renderClassSplitRefinementQualityBanner();",
            "assert.ok(classSplitElements.refinementQualityBanner.innerHTML.includes('require an exhaustive all-class run'));",
            "assert.ok(classSplitElements.refinementQualityBanner.innerHTML.includes('Scope = All classes and leave Sample cap blank'));",
            "Object.assign(prior, {fit_screening_scope: 'all_classes', fit_screening_exhaustive: false, fit_screening_adjustment_eligible: false, fit_screening_quality_gate: {passed: false, reason: 'screening_scope_ineligible', ordering_adjustments_enabled: false}});",
            "renderClassSplitRefinementQualityBanner();",
            "assert.ok(classSplitElements.refinementQualityBanner.innerHTML.includes('all-class run was sample-capped'));",
            "assert.ok(classSplitElements.refinementQualityBanner.innerHTML.includes('leave Sample cap blank'));",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_class_split_v6_normal_completion_has_no_release_gate_banner():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const CLASS_SPLIT_REFINEMENT_DECISION_CONTRACT = 'class-analysis-patch-decision-v9';",
            "const classSplitElements = { refinementQualityBanner: { hidden: true, innerHTML: '' } };",
            "const classSplitState = { result: { summary: { refinement: {",
            "  enabled: true, status: 'completed', quality_status: 'actionable',",
            "  decision_contract: CLASS_SPLIT_REFINEMENT_DECISION_CONTRACT,",
            "  rough_candidate_count: 100, vignette_candidate_count: 20,",
            "  queue_policy: { mode: 'selector_ranked_complete_stage1', rough_count: 100, confirmed_count: 12 },",
            "} } } };",
            _extract_js_function(js, "classSplitResultHasRefinement"),
            _extract_js_function(js, "getClassSplitRefinementQualityStatus"),
            _extract_js_function(js, "escapeHtml"),
            _extract_js_function(js, "renderClassSplitRefinementQualityBanner"),
            "renderClassSplitRefinementQualityBanner();",
            "assert.strictEqual(classSplitElements.refinementQualityBanner.hidden, true);",
            "const text = classSplitElements.refinementQualityBanner.innerHTML;",
            "assert.strictEqual(text, '');",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_class_split_authoritative_analysis_totals_contract():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const CLASS_SPLIT_REFINEMENT_CATEGORIES = [",
            "  'confirmed_outlier', 'explained_not_outlier', 'mixed_or_composite', 'unresolved', 'pair_conflict',",
            "];",
            "const raw = [",
            "  { point_id: 'visible-1', refined_outlier: { status: 'confirmed_outlier' } },",
            "  { point_id: 'visible-2', refined_outlier: { status: 'unresolved' } },",
            "];",
            "let primary = [raw[0]];",
            "const classSplitState = { result: { summary: {",
            "  vignette_candidate_count_before_human_review: 9,",
            "  wrong_class_candidate_count_before_human_review: 41,",
            "  refinement: {",
            "    rough_candidate_count: 42,",
            "    category_counts: {",
            "      confirmed_outlier: 9, explained_not_outlier: 8, mixed_or_composite: 7, unresolved: 6, pair_conflict: 5,",
            "    },",
            "  },",
            "} } };",
            "function getClassSplitRawCandidates() { return raw; }",
            "function getClassSplitPrimaryCandidates() { return primary; }",
            "function classSplitCandidateRefinementStatus(candidate) { return candidate.refined_outlier.status; }",
            "function classSplitResultSupportsPriorityHumanReview() { return false; }",
            _extract_js_function(js, "getClassSplitImmutableVignetteCounts"),
            "assert.deepStrictEqual(getClassSplitImmutableVignetteCounts(), {",
            "  confirmed_outlier: 9, explained_not_outlier: 8, mixed_or_composite: 7, unresolved: 6, pair_conflict: 5, review_queue: 9, rough_total: 42,",
            "});",
            "classSplitState.result = { summary: {} };",
            "assert.deepStrictEqual(getClassSplitImmutableVignetteCounts(), {",
            "  confirmed_outlier: 1, explained_not_outlier: 0, mixed_or_composite: 0, unresolved: 1, pair_conflict: 0, review_queue: 1, rough_total: 2,",
            "});",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_class_split_refinement_disclosure_preview_and_failure_fallback_contract():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const CLASS_SPLIT_REFINEMENT_DECISION_CONTRACT = 'class-analysis-patch-decision-v9';",
            _extract_js_function(js, "classSplitRefinementNeedsRoughFallback"),
            "assert.strictEqual(classSplitRefinementNeedsRoughFallback({ summary: { refinement: { status: 'failed' } } }), true);",
            "assert.strictEqual(classSplitRefinementNeedsRoughFallback({ summary: { refinement: { status: 'partial' } } }), true);",
            "assert.strictEqual(classSplitRefinementNeedsRoughFallback({ summary: { refinement: { status: 'cancelled' } } }), true);",
            "assert.strictEqual(classSplitRefinementNeedsRoughFallback({ summary: { refinement: { status: 'completed' } } }), false);",
            "assert.strictEqual(classSplitRefinementNeedsRoughFallback({ summary: { refinement: { status: 'completed', quality_status: 'completed_non_actionable', rough_candidate_count: 10, vignette_candidate_count: 3 } } }), false);",
            "assert.strictEqual(classSplitRefinementNeedsRoughFallback({ summary: { refinement: { status: 'completed', quality_status: 'actionable', rough_candidate_count: 10, vignette_candidate_count: 0 } } }), false);",
            "assert.strictEqual(classSplitRefinementNeedsRoughFallback({ summary: { refinement: { status: 'completed', quality_status: 'actionable', rough_candidate_count: 10, vignette_candidate_count: 2 } } }), false);",
            "assert.strictEqual(classSplitRefinementNeedsRoughFallback({ summary: { refinement: { status: 'completed', quality_status: 'actionable', queue_policy: { mode: 'selector_ranked_complete_stage1' } } } }), false);",
            "assert.strictEqual(classSplitRefinementNeedsRoughFallback({ summary: { refinement: { status: 'completed', queue_policy: { automatic_rough_fallback: true, fallback_reason: 'refinement_failed' } } } }), true);",
            "assert.strictEqual(classSplitRefinementNeedsRoughFallback({ summary: { refinement: { status: 'completed', queue_policy: { automatic_rough_fallback: false, fallback_reason: '' } } } }), false);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)

    apply_result = _extract_js_function(js, "applyClassSplitResultPayload")
    assert "if (classSplitRefinementNeedsRoughFallback(result))" in apply_result
    assert "classSplitState.showAllRough = true;" in apply_result
    assert "classSplitElements.showAllRough.checked = true;" in apply_result
    assert "classSplitState.disclosureState.clear();" in apply_result
    assert "disclosureState: new Map(classSplitState.disclosureState || [])" in js
    assert "classSplitState.disclosureState = new Map(snapshot.disclosureState || [])" in js
    assert 'trace-event-output-${escapeHtml(disclosureKey)}-${eventIndex}' in js
    assert 'card-audit-${escapeHtml(String(review.review_id || review.job_id || pointId))}' in js
    assert 'refinement-${escapeHtml(String(classSplitState.currentJobId || "job"))}-${escapeHtml(pointId)}' in js

    preview = _extract_js_function(js, "loadClassSplitRefinementPreview")
    clear_preview = _extract_js_function(js, "clearClassSplitRefinementPreviewCache")
    assert 'document.createElement("img")' in preview
    assert 'preview.addEventListener("load"' in preview
    assert 'preview.addEventListener("error"' in preview
    assert "details.isConnected" in preview
    assert "body.isConnected" in preview
    assert "preview.src = previewUrl" in preview
    assert "URL.createObjectURL" not in preview
    assert "fetch(" not in preview
    assert "classSplitState.refinementPreviewGeneration += 1;" in clear_preview


def test_class_split_result_reload_keeps_terminal_reviews_out_of_active_graph():
    js = _js()
    apply_result = _extract_js_function(js, "applyClassSplitResultPayload")
    initialize_hidden = "classSplitState.dismissedWrongIds = new Set();"
    add_reviewed = "classSplitState.dismissedWrongIds.add(pointId);"

    assert apply_result.count(initialize_hidden) == 1
    assert apply_result.count(add_reviewed) == 2
    assert apply_result.index(initialize_hidden) < apply_result.index(
        "(Array.isArray(result.points) ? result.points : []).forEach"
    )
    assert apply_result.index(add_reviewed) > apply_result.index(
        '"keep_both_boxes"'
    )
    assert apply_result.index(add_reviewed) > apply_result.index(
        '"unresolved"'
    )
    assert "const reviewedPairKeys = new Set(" in apply_result
    assert "point?.human_review_pair_key" in apply_result
    assert "point?.dual_bbox_conflict?.pair_review_key" in apply_result
    assert "classSplitState.dismissedWrongIds.add(pointId);" in apply_result


def test_selected_class_refinement_snapshot_includes_full_anchor_context():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const images = {",
            "  a: { displayName: 'a.jpg', meta: { name: 'a.jpg' } },",
            "  b: { displayName: 'b.jpg', meta: { name: 'b.jpg' } },",
            "  c: { displayName: 'c.jpg', meta: { name: 'c.jpg' } },",
            "};",
            "const lines = {",
            "  a: ['0 0.5 0.5 0.2 0.2', '1 0.2 0.2 0.1 0.1'],",
            "  b: ['1 0.7 0.7 0.1 0.1'],",
            "  c: [],",
            "};",
            "const textLabels = {};",
            "const annotationSourceState = { datasetLabel: 'fixture' };",
            "function captureCurrentAnnotationDirtyState() {}",
            "function getClassSplitLabelmapEntries() { return [{ id: 0, name: 'Target' }, { id: 1, name: 'Other' }]; }",
            "function getClassSplitImageKeys() { return ['a', 'b', 'c']; }",
            "function getClassSplitActiveLabelLines(key) { return [...lines[key]]; }",
            "function classSplitLineMatchesScope(line, scope) { return scope === 'all_classes' || String(line).startsWith('0 '); }",
            "function makeClassSplitUploadName(key) { return `${key}.jpg`; }",
            "function classSplitSnapshotSourceIdentity(key) { return key; }",
            "function classSplitHashValues(values) { return values.join('|'); }",
            _extract_js_function_before(
                js,
                "buildClassSplitActiveWorkspaceSnapshot",
                "async function buildClassSplitActiveWorkspaceFormFromSnapshot",
            ),
            "const progress = [];",
            "const refined = buildClassSplitActiveWorkspaceSnapshot({",
            "  analysis_scope: 'selected_class', class_name: 'Target', refine_outliers: true,",
            "}, { onProgress: (event) => progress.push(event) });",
            "assert.strictEqual(refined.imageCount, 2);",
            "assert.strictEqual(refined.objectCount, 1);",
            "assert.strictEqual(refined.queryObjectCount, 1);",
            "assert.strictEqual(refined.contextObjectCount, 3);",
            "assert.deepStrictEqual(refined.rows.map((row) => row.label_lines), [lines.a, lines.b]);",
            "assert.strictEqual(refined.manifest.query_object_count, 1);",
            "assert.strictEqual(refined.manifest.context_object_count, 3);",
            "assert(progress.at(-1).message.includes('1 query objects'));",
            "assert(progress.at(-1).message.includes('3 context annotations'));",
            "const pooled = buildClassSplitActiveWorkspaceSnapshot({",
            "  analysis_scope: 'selected_class', class_name: 'Target', refine_outliers: false,",
            "});",
            "assert.strictEqual(pooled.imageCount, 1);",
            "assert.strictEqual(pooled.objectCount, 1);",
            "assert.strictEqual(pooled.contextObjectCount, 1);",
            "assert.deepStrictEqual(pooled.rows[0].label_lines, [lines.a[0]]);",
            "const allClasses = buildClassSplitActiveWorkspaceSnapshot({ analysis_scope: 'all_classes', refine_outliers: true });",
            "assert.strictEqual(allClasses.imageCount, 2);",
            "assert.strictEqual(allClasses.objectCount, 3);",
            "assert.strictEqual(allClasses.contextObjectCount, 3);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_back_gesture_guard_preserves_canvas_shift_wheel_propagation():
    js = _js()
    guard_start = js.index("function installTatorBackGestureGuard")
    guard_end = js.index("function formatBytesLabel", guard_start)
    guard_body = js[guard_start:guard_end]

    assert "event.preventDefault();" in guard_body
    assert "isTatorCanvasWheelTarget(event.target)" in guard_body

    script = "\n".join(
        [
            "const assert = require('assert');",
            "const TATOR_HISTORY_GUARD_KEY = 'test_guard';",
            "class MockElement {",
            "  constructor({ id = '', parentElement = null, overflowX = 'visible', scrollLeft = 0, clientWidth = 100, scrollWidth = 100 } = {}) {",
            "    Object.assign(this, { id, parentElement, overflowX, scrollLeft, clientWidth, scrollWidth });",
            "  }",
            "  closest(selector) {",
            "    let node = this;",
            "    while (node) {",
            "      if (selector === '#canvas' && node.id === 'canvas') return node;",
            "      node = node.parentElement;",
            "    }",
            "    return null;",
            "  }",
            "}",
            "const Element = MockElement;",
            "const document = { body: new MockElement({ id: 'body' }) };",
            "const listeners = {};",
            "const window = {",
            "  location: { href: 'http://127.0.0.1/tator.html' },",
            "  getComputedStyle: (node) => ({ overflowX: node.overflowX }),",
            "  addEventListener: (type, handler, options) => { listeners[type] = { handler, options }; },",
            "};",
            "const history = {",
            "  state: {},",
            "  replaceState(state) { this.state = state; },",
            "  pushState(state) { this.state = state; },",
            "};",
            _extract_js_function(js, "findHorizontalScrollContainer"),
            _extract_js_function(js, "isTatorCanvasWheelTarget"),
            _extract_js_function(js, "installTatorBackGestureGuard"),
            "installTatorBackGestureGuard();",
            "assert.strictEqual(listeners.wheel.options.capture, true);",
            "assert.strictEqual(listeners.wheel.options.passive, false);",
            "function dispatchWheel(target, deltaX, deltaY, shiftKey = true) {",
            "  const result = { prevented: 0, stopped: 0, downstream: 0 };",
            "  const event = {",
            "    target, deltaX, deltaY, shiftKey, cancelable: true,",
            "    preventDefault() { result.prevented += 1; },",
            "    stopPropagation() { result.stopped += 1; },",
            "  };",
            "  listeners.wheel.handler(event);",
            "  if (!result.stopped) result.downstream += 1;",
            "  return result;",
            "}",
            "const canvas = new MockElement({ id: 'canvas', parentElement: document.body });",
            "assert.deepStrictEqual(dispatchWheel(canvas, 18, 1), { prevented: 1, stopped: 0, downstream: 1 });",
            "assert.deepStrictEqual(dispatchWheel(canvas, 18, 1, false), { prevented: 1, stopped: 1, downstream: 0 });",
            "const background = new MockElement({ id: 'background', parentElement: document.body });",
            "assert.deepStrictEqual(dispatchWheel(background, 18, 1), { prevented: 1, stopped: 1, downstream: 0 });",
            "const scroller = new MockElement({ parentElement: document.body, overflowX: 'auto', scrollLeft: 20, clientWidth: 100, scrollWidth: 300 });",
            "assert.deepStrictEqual(dispatchWheel(scroller, -18, 1), { prevented: 0, stopped: 0, downstream: 1 });",
            "history.state = { [TATOR_HISTORY_GUARD_KEY]: 'base' };",
            "listeners.popstate.handler();",
            "assert.strictEqual(history.state[TATOR_HISTORY_GUARD_KEY], 'sentinel');",
        ]
    )
    subprocess.run(
        [
            "node",
            "-e",
            script,
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_data_quality_graph_hover_preview_shows_loader_until_current_image_loads():
    js = _js()
    css = _css()
    move_start = js.index("function moveClassSplitGraphHoverPreview")
    move_end = js.index("function formatClassSplitGraphHoverLabel", move_start)
    move_block = js[move_start:move_end]
    hover_start = js.index("function resetClassSplitGraphHoverPreviewElement")
    hover_end = js.index("function classSplitClusterKey", hover_start)
    hover_block = js[hover_start:hover_end]
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const CLASS_SPLIT_GRAPH_HOVER_DEBOUNCE_MS = 90;",
            "const classSplitGraphHoverState = { timer: null, token: 0, pointId: '', anchor: null, loading: false };",
            "const classSplitState = { currentJobId: 'job-1' };",
            "const API_ROOT = 'http://api';",
            "let nextTimerId = 1;",
            "const timers = new Map();",
            "const window = {",
            "  setTimeout(callback) { const id = nextTimerId++; timers.set(id, callback); return id; },",
            "  clearTimeout(id) { timers.delete(id); },",
            "};",
            "const classes = new Set();",
            "const imgAttributes = new Map();",
            "const img = {",
            "  naturalWidth: 0,",
            "  naturalHeight: 0,",
            "  onload: null,",
            "  onerror: null,",
            "  setAttribute(name, value) { imgAttributes.set(name, String(value)); },",
            "  getAttribute(name) { return imgAttributes.has(name) ? imgAttributes.get(name) : null; },",
            "  removeAttribute(name) { imgAttributes.delete(name); },",
            "};",
            "const caption = { textContent: '' };",
            "const preview = {",
            "  hidden: true,",
            "  style: {},",
            "  getBoundingClientRect() { return { width: 100, height: 100 }; },",
                "  classList: {",
                "    add(name) { classes.add(name); },",
                "    remove(name) { classes.delete(name); },",
                "    contains(name) { return classes.has(name); },",
                "    toggle(name, force) {",
                "      const enabled = force === undefined ? !classes.has(name) : Boolean(force);",
                "      if (enabled) classes.add(name); else classes.delete(name);",
                "      return enabled;",
                "    },",
                "  },",
            "  querySelector(selector) {",
            "    if (selector === 'img') return img;",
            "    if (selector === '.class-split-graph-hover-caption') return caption;",
            "    return null;",
            "  },",
            "};",
            "const document = { getElementById(id) { return id === 'classSplitGraphHoverPreview' ? preview : null; } };",
            "window.innerWidth = 1000;",
            "window.innerHeight = 800;",
            "function getClassSplitGraphHoverPreview() { return preview; }",
            "function formatClassSplitGraphHoverLabel(point) { return `point ${point.point_id}`; }",
            move_block,
            hover_block,
            "async function runTimers() {",
            "  const callbacks = Array.from(timers.values());",
            "  timers.clear();",
            "  for (const callback of callbacks) await callback();",
            "}",
            "(async () => {",
            "  showClassSplitGraphHoverPreview({ clientX: 1, clientY: 2 }, '/thumb/a', { point_id: 'a' });",
            "  assert.strictEqual(preview.hidden, true);",
            "  assert.strictEqual(preview.classList.contains('is-loading'), false);",
            "  assert.strictEqual(classSplitGraphHoverState.loading, true);",
            "  const firstTimerId = Array.from(timers.keys())[0];",
            "  showClassSplitGraphHoverPreview({ clientX: 3, clientY: 4 }, '/thumb/a', { point_id: 'a' });",
            "  assert.strictEqual(Array.from(timers.keys())[0], firstTimerId);",
            "  assert.strictEqual(timers.size, 1);",
            "  moveClassSplitGraphHoverPreview({ clientX: 9, clientY: 10 });",
            "  assert.deepStrictEqual(classSplitGraphHoverState.anchor, { clientX: 9, clientY: 10 });",
            "  await runTimers();",
            "  assert.strictEqual(img.getAttribute('src'), '/thumb/a');",
            "  assert.strictEqual(preview.hidden, false);",
            "  assert.strictEqual(preview.classList.contains('is-loading'), true);",
            "  img.naturalWidth = 80;",
            "  img.naturalHeight = 40;",
            "  img.onload();",
            "  assert.strictEqual(preview.hidden, false);",
            "  assert.strictEqual(preview.classList.contains('is-loading'), false);",
            "  assert.strictEqual(preview.classList.contains('is-loaded'), true);",
            "  assert.strictEqual(classSplitGraphHoverState.loading, false);",
            "  assert.strictEqual(preview.style.left, '27px');",
            "  assert.strictEqual(preview.style.top, '28px');",
            "  hideClassSplitGraphHoverPreview();",
            "",
            "  showClassSplitGraphHoverPreview({ clientX: 1, clientY: 2 }, '/thumb/a', { point_id: 'a' });",
            "  showClassSplitGraphHoverPreview({ clientX: 3, clientY: 4 }, '/thumb/b', { point_id: 'b' });",
            "  assert.strictEqual(timers.size, 1);",
            "  assert.strictEqual(preview.hidden, true);",
            "  assert.strictEqual(preview.classList.contains('is-loading'), false);",
            "  assert.strictEqual(img.getAttribute('src'), null);",
            "  await runTimers();",
            "  assert.strictEqual(img.getAttribute('src'), '/thumb/b');",
            "  assert.strictEqual(preview.hidden, false);",
            "  assert.strictEqual(preview.classList.contains('is-loading'), true);",
            "  assert.strictEqual(preview.classList.contains('is-loaded'), false);",
            "  img.naturalWidth = 80;",
            "  img.naturalHeight = 40;",
            "  img.onload();",
            "  assert.strictEqual(preview.hidden, false);",
            "  assert.strictEqual(preview.classList.contains('is-loaded'), true);",
            "  assert.deepStrictEqual(classSplitGraphHoverState.anchor, { clientX: 3, clientY: 4 });",
            "  hideClassSplitGraphHoverPreview();",
            "  assert.strictEqual(preview.hidden, true);",
            "  assert.strictEqual(img.getAttribute('src'), null);",
            "  assert.strictEqual(caption.textContent, '');",
            "",
            "  showClassSplitGraphHoverPreview({ clientX: 5, clientY: 6 }, '/thumb/c', { point_id: 'c' });",
            "  await runTimers();",
            "  assert.strictEqual(img.getAttribute('src'), '/thumb/c');",
            "  img.onerror();",
            "  assert.strictEqual(preview.hidden, false);",
            "  assert.strictEqual(preview.classList.contains('is-loading'), true);",
            "  assert.strictEqual(img.getAttribute('src'), 'http://api/class_analysis/jobs/job-1/thumbnail/c');",
            "  img.onerror();",
            "  assert.strictEqual(preview.hidden, true);",
            "  assert.strictEqual(preview.classList.contains('is-loading'), false);",
            "  assert.strictEqual(preview.classList.contains('is-loaded'), false);",
            "  assert.strictEqual(classSplitGraphHoverState.loading, false);",
            "  assert.strictEqual(img.getAttribute('src'), null);",
            "  showClassSplitGraphHoverPreview({ clientX: 5, clientY: 6 }, '/thumb/c', { point_id: 'c' });",
            "  await runTimers();",
            "  assert.strictEqual(img.getAttribute('src'), '/thumb/c');",
            "",
            "  const staleLoad = img.onload;",
            "  showClassSplitGraphHoverPreview({ clientX: 7, clientY: 8 }, '/thumb/d', { point_id: 'd' });",
            "  img.naturalWidth = 80;",
            "  img.naturalHeight = 40;",
            "  staleLoad();",
            "  assert.strictEqual(preview.hidden, true);",
            "  assert.strictEqual(preview.classList.contains('is-loading'), false);",
            "  assert.strictEqual(img.getAttribute('src'), null);",
            "  hideClassSplitGraphHoverPreview();",
            "  assert.strictEqual(classSplitGraphHoverState.loading, false);",
            "  await runTimers();",
            "  assert.strictEqual(preview.hidden, true);",
            "  assert.strictEqual(img.getAttribute('src'), null);",
            "})().catch((error) => { console.error(error); process.exit(1); });",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)

    assert ".class-split-graph-hover-preview.is-loaded img" in css
    assert ".class-split-graph-hover-preview.is-loading .class-split-graph-hover-loading" in css
    assert "class-split-hover-preview-spin 700ms linear infinite" in css
    assert 'class="class-split-graph-hover-loading" role="status"' in js
    assert ".class-split-graph-hover-preview img {\n    min-height: 0;\n    opacity: 0;" in css
    assert "getClassSplitContextCropUrl(point, { cache: false })" not in hover_block
    assert "URL.createObjectURL" not in hover_block


def test_qwen_review_disclosures_and_live_generation_contract():
    html = _html()
    css = _css()
    js = _js()

    assert "rememberClassSplitDisclosureState" in js
    assert "restoreClassSplitDisclosureState" in js
    assert "data-disclosure-key" in js
    assert "active_generation" in js
    assert "class-split-qwen-live-output" in css
    assert "Live model output — unverified until complete" in js
    assert "Complete VLM reasoning trace" in js
    assert "complete_trace" in js
    assert "Prompt and selected model inputs" in js
    assert "Audit trail intermediate outputs" in js
    assert "(please wait)" in js
    assert "currentSummary.innerHTML = nextSummary.innerHTML" in js
    assert "function loadClassSplitQwenCompleteTrace" in js
    assert "function bindClassSplitQwenTraceLoaders" in js
    assert "data-qwen-trace-review-id" in js
    assert ".class-split-qwen-review__trace" in css
    assert ".class-split-qwen-review__wait" in css
    assert 'id="qwenCachePreviewBtn"' in html
    assert "recommended cleanup candidate" in js
    assert "Shared-cache detached revisions are reported but excluded" in js


def test_qwen_card_disclosures_do_not_select_the_card_and_complete_trace_renders_in_place():
    js = _js()
    trace_render = _extract_js_function(js, "renderClassSplitQwenCompleteTrace")
    trace_load = _extract_js_function(js, "loadClassSplitQwenCompleteTrace")
    trace_load_executable = trace_load.replace(
        "function loadClassSplitQwenCompleteTrace",
        "async function loadClassSplitQwenCompleteTrace",
        1,
    )

    assert '"[data-action], details, a, button, select, input, label"' in js
    assert "renderClassSplitQwenCompleteTrace(details, review)" in trace_load
    assert "updateClassSplitQwenReviewJob(review)" not in trace_load

    script = "\n".join(
        [
            "const assert = require('assert');",
            "class FakeDetails {}",
            "global.HTMLDetailsElement = FakeDetails;",
            "const summary = {textContent: ''};",
            "const body = {innerHTML: '', textContent: ''};",
            "const details = new FakeDetails();",
            "details.dataset = {};",
            "details.open = false;",
            "details.getAttribute = (name) => ({",
            "  'data-qwen-trace-review-id': 'review-1',",
            "  'data-qwen-trace-point-id': 'point-1',",
            "}[name] || '');",
            "details.querySelector = (selector) => selector === ':scope > summary' ? summary : body;",
            "const classSplitState = {qwenReviewJobs: new Map()};",
            "const API_ROOT = '';",
            "let toastReview = null;",
            "let refreshedPoint = '';",
            "function restoreClassSplitDisclosureState() {}",
            "function renderClassSplitQwenTraceEvent(event, index) { return `<li>${index}:${event.type}:${event.text || ''}</li>`; }",
            "function parseApiError(detail, fallback) { return detail || fallback; }",
            "function parseJsonObjectSafe(detail, fallback) { try { return JSON.parse(detail); } catch (_) { return fallback; } }",
            "function renderClassSplitQwenReviewTraceToast(review) { toastReview = review; }",
            "function refreshClassSplitQwenActionControls(pointId) { refreshedPoint = pointId; }",
            "const responseReview = {",
            "  review_id: 'review-1', point_id: 'point-1',",
            "  complete_trace: {included: true, model_output_count: 1, events: [{type: 'model_output', text: 'answer'}]},",
            "};",
            "global.fetch = async () => ({ok: true, status: 200, text: async () => JSON.stringify(responseReview)});",
            trace_render,
            trace_load_executable,
            "(async () => {",
            "  await loadClassSplitQwenCompleteTrace(details);",
            "  assert.strictEqual(details.dataset.qwenTraceLoaded, '1');",
            "  assert.strictEqual(details.open, true);",
            "  assert.ok(body.innerHTML.includes('model_output:answer'));",
            "  assert.strictEqual(summary.textContent, 'Complete VLM reasoning trace (1 model outputs, 1 events)');",
            "  assert.deepStrictEqual(classSplitState.qwenReviewJobs.get('point-1'), responseReview);",
            "  assert.strictEqual(toastReview.review_id, responseReview.review_id);",
            "  assert.strictEqual(refreshedPoint, 'point-1');",
            "})().catch((error) => { console.error(error); process.exit(1); });",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_review_controls_refresh_stale_capabilities_on_click_without_weakening_v3_gate():
    js = _js()
    helper = _extract_js_function(
        js,
        "ensureClassSplitReviewDispositionCapabilities",
    )
    helper_executable = helper.replace(
        "function ensureClassSplitReviewDispositionCapabilities",
        "async function ensureClassSplitReviewDispositionCapabilities",
        1,
    )
    wrong_list = _extract_js_function(js, "renderClassSplitWrongList")

    assert "const dispositionDisabled = dispositionBusy;" in wrong_list
    assert "!reviewDispositionApiAvailable" not in wrong_list[
        wrong_list.index("const dispositionDisabled"):
        wrong_list.index("const dualActionState")
    ]

    script = "\n".join(
        [
            "const assert = require('assert');",
            "const CLASS_SPLIT_DUAL_BBOX_DISPOSITIONS = ['keep_both_boxes'];",
            "const classSplitState = {capabilities: {}};",
            "let loads = 0;",
            "let makeAvailable = true;",
            "async function loadClassSplitCapabilities() {",
            "  loads += 1;",
            "  classSplitState.capabilities = makeAvailable ? {review_disposition_api_version: 3} : {};",
            "}",
            "function classSplitDualBBoxResolutionApiAvailable() { return true; }",
            helper_executable,
            "(async () => {",
            "  await ensureClassSplitReviewDispositionCapabilities('skip');",
            "  assert.strictEqual(loads, 1);",
            "  await ensureClassSplitReviewDispositionCapabilities('confirm_current');",
            "  assert.strictEqual(loads, 1);",
            "  makeAvailable = false;",
            "  classSplitState.capabilities = {};",
            "  await assert.rejects(",
            "    ensureClassSplitReviewDispositionCapabilities('skip'),",
            "    /Restart the backend/",
            "  );",
            "})().catch((error) => { console.error(error); process.exit(1); });",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_cancelled_refinement_restores_persisted_stage_one_result_contract():
    js = _js()
    poll_function = _extract_js_function(js, "pollClassSplitJob")
    cancelled_block = _extract_js_block(
        poll_function, 'if (status === "cancelled")'
    )

    assert "await loadClassSplitResult(jobId)" in cancelled_block
    assert "[404, 409].includes(Number(error?.httpStatus))" in cancelled_block
    assert "const restoredPreviousResult = !restoredStageOne" in cancelled_block
    assert "&& restorePendingClassSplitReviewState()" in cancelled_block
    assert "preserved Stage-1 results are available" in cancelled_block
    assert "restored the previous completed analysis" in cancelled_block
    assert cancelled_block.index("await loadClassSplitResult(jobId)") < cancelled_block.index(
        "restorePendingClassSplitReviewState()"
    )
    load_start = js.index("async function loadClassSplitResult(jobId)")
    load_end = js.index("function classSplitResultHasRefinement", load_start)
    assert "error.httpStatus = resp.status" in js[load_start:load_end]

    script = "\n".join(
        [
            "const assert = require('assert');",
            "const API_ROOT = '';",
            "let classSplitState;",
            "let loadMode;",
            "let fallbackResult;",
            "let statusCalls;",
            "let restoreCalls;",
            "let renderCalls;",
            "let timerCalls;",
            "const stopClassSplitPoll = () => {};",
            "const persistDataQualityExplorerSession = () => {};",
            "const parseApiError = (_detail, fallback) => fallback;",
            "const parseJsonObjectSafe = (value) => JSON.parse(value);",
            "const formatClassSplitJobFailure = () => '';",
            "const formatClassSplitEmbeddingRecovery = () => '';",
            "const renderClassSplitProgress = () => {};",
            "const refreshClassSplitControls = () => {};",
            "const setClassSplitJobStatus = (...args) => statusCalls.push(args);",
            "const renderClassSplitPlot = () => { renderCalls += 1; };",
            "const restorePendingClassSplitReviewState = () => { restoreCalls += 1; return fallbackResult; };",
            "const loadClassSplitResult = async () => {",
            "  if (loadMode === 'success') return;",
            "  const error = new Error(`load ${loadMode}`);",
            "  error.httpStatus = Number(loadMode);",
            "  throw error;",
            "};",
            "const fetch = async () => ({ ok: true, status: 200, text: async () => JSON.stringify({ status: 'cancelled' }) });",
            "const window = { setTimeout: (...args) => { timerCalls.push(args); return 1; } };",
            poll_function,
            "async function runScenario(mode, fallback) {",
            "  classSplitState = { active: true, pollFailureCount: 9, pollRequestId: 0, pollTimer: null };",
            "  loadMode = mode; fallbackResult = fallback; statusCalls = []; restoreCalls = 0; renderCalls = 0; timerCalls = [];",
            "  await pollClassSplitJob('cancelled-job');",
            "  return { active: classSplitState.active, statusCalls, restoreCalls, renderCalls, timerCalls };",
            "}",
            "(async () => {",
            "  const persisted = await runScenario('success', true);",
            "  assert.strictEqual(persisted.active, false);",
            "  assert.strictEqual(persisted.restoreCalls, 0);",
            "  assert.strictEqual(persisted.renderCalls, 0);",
            "  assert.strictEqual(persisted.timerCalls.length, 0);",
            "  assert.deepStrictEqual(persisted.statusCalls.at(-1), ['Cancelled; preserved Stage-1 results are available.', 'warn']);",
            "  for (const status of [404, 409]) {",
            "    const fallback = await runScenario(String(status), true);",
            "    assert.strictEqual(fallback.active, false);",
            "    assert.strictEqual(fallback.restoreCalls, 1);",
            "    assert.strictEqual(fallback.renderCalls, 1);",
            "    assert.strictEqual(fallback.timerCalls.length, 0);",
            "    assert.deepStrictEqual(fallback.statusCalls.at(-1), ['Cancelled; restored the previous completed analysis.', 'warn']);",
            "  }",
            "})().catch((error) => { console.error(error); process.exit(1); });",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_review_request_timeout_releases_a_stalled_local_backend_call():
    helper = "async " + _extract_js_function(
        _js(),
        "fetchClassSplitReviewRequest",
    )
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const CLASS_SPLIT_REVIEW_REQUEST_TIMEOUT_MS = 12000;",
            "const window = {setTimeout, clearTimeout};",
            "let stallPhase = 'headers';",
            "const stalled = (signal) => new Promise((_resolve, reject) => {",
            "  signal.addEventListener('abort', () => {",
            "    const error = new Error('aborted'); error.name = 'AbortError'; reject(error);",
            "  });",
            "});",
            "const fetch = (_url, options) => stallPhase === 'headers'",
            "  ? stalled(options.signal)",
            "  : Promise.resolve({ok: true, status: 200, text: () => stalled(options.signal)});",
            helper,
            "(async () => {",
            "  const started = Date.now();",
            "  await assert.rejects(",
            "    fetchClassSplitReviewRequest('/stalled', {}, {timeoutMs: 5, timeoutMessage: 'bounded'}),",
            "    (error) => error.name === 'ClassSplitRequestTimeoutError' && error.classSplitRequestTimedOut === true && error.message === 'bounded'",
            "  );",
            "  stallPhase = 'body';",
            "  await assert.rejects(",
            "    fetchClassSplitReviewRequest('/stalled-body', {}, {timeoutMs: 5, timeoutMessage: 'body bounded'}),",
            "    (error) => error.name === 'ClassSplitRequestTimeoutError' && error.classSplitRequestTimedOut === true && error.message === 'body bounded'",
            "  );",
            "  assert.ok(Date.now() - started < 1000);",
            "})().catch((error) => { console.error(error); process.exit(1); });",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_unverified_review_recovery_discards_only_after_session_state_persists():
    js = _js()
    pending = _extract_js_function(
        js,
        "getClassSplitUnverifiedPendingReviewCommits",
    )
    discard = _extract_js_function(
        js,
        "discardClassSplitUnverifiedPendingReviewCommits",
    )
    script = "\n".join(
        [
            "const assert = require('assert');",
            "const first = {queueKey: 'job:p1:delete_bbox', jobId: 'job', pointId: 'p1', state: 'unknown'};",
            "const second = {queueKey: 'other:p2:delete_bbox', jobId: 'other', pointId: 'p2', state: 'unknown'};",
            "const classSplitState = {currentJobId: 'job', pendingReviewDispositionCommits: new Map([[first.queueKey, first], [second.queueKey, second]]), reviewDispositionReconciliationPointIds: new Set(['p1'])};",
            "const window = {confirm: () => true};",
            "let persistOk = false; let wrongRenders = 0; let historyRenders = 0; let refreshes = 0; let notices = 0;",
            "const classSplitPendingReviewMutationState = (entry) => entry.state;",
            "const persistDataQualityExplorerSession = () => persistOk;",
            "const renderClassSplitPendingReviewRecovery = () => {};",
            "const renderClassSplitWrongList = () => { wrongRenders += 1; };",
            "const renderClassSplitReviewedList = () => { historyRenders += 1; };",
            "const refreshClassSplitControls = () => { refreshes += 1; };",
            "const enqueueTaskNotice = () => { notices += 1; };",
            "const classSplitMutationIsBusy = () => false;",
            pending,
            discard,
            "assert.throws(() => discardClassSplitUnverifiedPendingReviewCommits(), /were kept/);",
            "assert.strictEqual(classSplitState.pendingReviewDispositionCommits.has(first.queueKey), true);",
            "assert.strictEqual(classSplitState.pendingReviewDispositionCommits.has(second.queueKey), true);",
            "persistOk = true;",
            "discardClassSplitUnverifiedPendingReviewCommits();",
            "assert.strictEqual(classSplitState.pendingReviewDispositionCommits.has(first.queueKey), false);",
            "assert.strictEqual(classSplitState.pendingReviewDispositionCommits.has(second.queueKey), true);",
            "assert.strictEqual(classSplitState.reviewDispositionReconciliationPointIds.has('p1'), false);",
            "assert.strictEqual(wrongRenders, 1); assert.strictEqual(historyRenders, 1); assert.strictEqual(refreshes, 1); assert.strictEqual(notices, 1);",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)


def test_review_history_delete_restores_retry_only_for_known_rejection():
    delete_history = "async " + _extract_js_function(
        _js(),
        "deleteClassSplitReviewHistory",
    )
    script = "\n".join(
        [
            "const assert = require('node:assert/strict');",
            "const point = {point_id: 'p1', human_review_disposition: 'skip', human_review_revision: 'rdr1_' + 'a'.repeat(32)};",
            "const pending = {queueKey: 'job-1:p1:delete_bbox', jobId: 'job-1', pointId: 'p1'};",
            "let failureMode = 'known'; let persisted = [];",
            "const classSplitState = {currentJobId: 'job-1', relabelInFlight: false, reviewActionPendingPointIds: new Set(), reviewDispositionReconciliationPointIds: new Set(), reviewDispositionHydrationTimer: null, reviewDispositionHydrationTargets: new Map(), reviewDispositionHydrationInFlight: new Set(), reviewCommitDrainPromise: null, reviewCommitDrainRequested: false, reviewedPointsById: new Map([['p1', point]]), pendingReviewDispositionCommits: new Map(), capabilities: {review_history_delete_api_version: 1}, reviewHistoryDeleteOperations: new Map(), reviewHistoryDeleteSequence: 0, reviewedChoiceLimit: 250, lastReviewDisposition: null};",
            "function reset() { classSplitState.pendingReviewDispositionCommits = new Map([[pending.queueKey, pending]]); classSplitState.reviewHistoryDeleteOperations.clear(); classSplitState.reviewCommitDrainRequested = false; persisted = []; }",
            "function classSplitReviewHistoryDeleteOperation(jobId) { return classSplitState.reviewHistoryDeleteOperations.get(jobId) || null; }",
            "function classSplitReviewDispositionInFlightForJob() { return false; }",
            "function classSplitMutationIsBusy() { return false; }",
            "function getClassSplitReviewHistoryPoints() { return [point]; }",
            "function buildClassSplitReviewHistoryDeleteEntries() { return [{point_id: 'p1'}]; }",
            "function snapshotClassSplitReviewHistory() { return [{pointId: 'p1', revision: point.human_review_revision}]; }",
            "function createClassSplitTrainingClientActionId() { return 'client-action-123'; }",
            "function persistDataQualityExplorerSession() { persisted.push([...classSplitState.pendingReviewDispositionCommits.keys()]); return true; }",
            "function renderClassSplitReviewedList() {} function refreshClassSplitControls() {} function refreshClassSplitVignetteControls() {}",
            "function enqueueTaskNotice() {} function setClassSplitJobStatus() {} function scheduleClassSplitPendingReviewCommitDrain() { throw new Error('must not drain staged entry'); }",
            "const window = {confirm: () => true}; const API_ROOT = '';",
            "async function fetchClassSplitReviewRequest() { const error = new Error(failureMode); if (failureMode === 'known') error.classSplitReviewHistoryCommitUnknown = false; throw error; }",
            delete_history,
            "(async () => {",
            "  reset(); await assert.rejects(deleteClassSplitReviewHistory(), /known/);",
            "  assert.deepEqual([...classSplitState.pendingReviewDispositionCommits.keys()], [pending.queueKey]);",
            "  assert.deepEqual(persisted.at(-1), [pending.queueKey]);",
            "  reset(); failureMode = 'unknown';",
            "  await assert.rejects(deleteClassSplitReviewHistory(), /outcome is unclear/);",
            "  assert.equal(classSplitState.pendingReviewDispositionCommits.size, 0);",
            "  assert.deepEqual(persisted.at(-1), []);",
            "})().catch((error) => { console.error(error); process.exit(1); });",
        ]
    )
    subprocess.run(["node", "-e", script], cwd=REPO_ROOT, check=True)
