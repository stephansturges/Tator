from html.parser import HTMLParser
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = REPO_ROOT / "ybat-master" / "tator.html"
CSS_PATH = REPO_ROOT / "ybat-master" / "ybat.css"
JS_PATH = REPO_ROOT / "ybat-master" / "ybat.js"
MOBILE_REVIEW_PATH = REPO_ROOT / "ybat-master" / "mobile_review.html"
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
    brace_start = source.index("{", start)
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


def _mobile_review_html() -> str:
    return MOBILE_REVIEW_PATH.read_text(encoding="utf-8")


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
    assert "uses persisted backend jobs, isolated retries, health gates, and auto-resume after backend restarts" in html
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
    assert "qwenCaptionRequireReadyInstructionExport" in html
    assert 'id="qwenCaptionRequireReadyInstructionExport" checked' in html
    assert "qwenCaptionBuildInstructionDataset" in html
    assert "qwenCaptionDownloadInstructionJsonl" in html
    assert "qwenCaptionDownloadInstructionArchive" in html
    assert "qwenCaptionDownloadInstructionReview" in html
    assert "qwenCaptionImportInstructionReview" in html
    assert "qwenCaptionImportInstructionReviewFile" in html
    assert "qwenCaptionDownloadInstructionReport" in html
    assert "Create VLM training dataset" in html
    assert "Import reviewed JSONL" in html
    assert "Generated QA never becomes source annotations" in html
    assert "review JSONL can be exported for audit then imported" in html
    assert "deterministic metadata QA is included only when explicitly enabled" in html
    assert "Trainer JSONL download requires a ready instruction report by default" in html
    assert "qwenCaptionExportHealth" in html
    assert "qwenCaptionReadinessRun" in html
    assert "qwenCaptionReadinessStatus" in html
    assert "qwenCaptionReadinessResults" in html
    assert "Check caption readiness" in html
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
    assert "excluded from trainer JSONL" in js
    assert "instructionDataset: true" in js
    assert "function validateCaptionInstructionTrainingRows" in js
    assert "function validateCaptionInstructionArchiveRows" in js
    assert "function validateCaptionInstructionReviewRows" in js
    assert "function parseCaptionInstructionReviewRowsText" in js
    assert "async function importCaptionInstructionReviewFile" in js
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
    assert 'saveBlobToDisk(blob, "caption_instruction_archive.jsonl")' in js
    assert 'saveBlobToDisk(blob, "caption_instruction_review.jsonl")' in js
    assert 'saveBlobToDisk(blob, "caption_instruction_report.json")' in js
    assert "/captions/instruction_review" in js
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
    assert "if (updated?.is_primary)" in js
    assert "captionAutoSaveState.lastSaved.set(imageName, updated.caption || caption)" in js
    init_start = js.index("function initQwenPanel")
    init_end = js.index("function refreshQwenStatus", init_start)
    init_helper = js[init_start:init_end]
    assert init_helper.count("renderCaptionAlternatesForCurrentImage();") >= 2
    assert "const datasetId = getCaptionRecordDatasetId();" in load_helper
    assert "isAnnotationDatasetModeActive()" not in load_helper


def test_qwen_caption_instruction_review_import_parser_accepts_reviewer_file_shapes():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
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
            "  qa_mix: 'object',",
            "  answer_format: 'json',",
            "};",
            "const strictParams = new URLSearchParams(captionInstructionExportQuery(settings, { requireReadyInstructionExport: true }));",
            "assert.strictEqual(strictParams.get('include_caption0_in_training'), 'true');",
            "assert.strictEqual(strictParams.get('include_generated_qa_in_training'), 'false');",
            "assert.strictEqual(strictParams.get('include_deterministic_metadata_qa'), 'true');",
            "assert.strictEqual(strictParams.get('qa_mix'), 'object');",
            "assert.strictEqual(strictParams.get('answer_format'), 'json');",
            "assert.strictEqual(strictParams.get('require_ready_instruction_export'), 'true');",
            "const diagnosticParams = new URLSearchParams(captionInstructionExportQuery(settings));",
            "assert.strictEqual(diagnosticParams.has('require_ready_instruction_export'), false);",
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
            "assert(archiveOnlyQa.warnings.some((warning) => warning.includes('excluded from trainer JSONL')));",
            "const summary = describeCaptionInstructionLaunchSettings({",
            "  instruction_dataset: true,",
            "  include_caption0_in_training: false,",
            "  include_generated_qa_in_training: false,",
            "  include_deterministic_metadata_qa: true,",
            "  subcaptions_per_image: 3,",
            "});",
            "assert(summary.includes('generated QA candidates per image for archive/review only'));",
            "assert(summary.includes('deterministic metadata QA rows'));",
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


def test_qwen_caption_instruction_artifact_consistency_blocks_mismatched_exports():
    js = _js()
    script = "\n".join(
        [
            "const assert = require('assert');",
            _extract_js_function(js, "normalizeCaptionInstructionReviewDecision"),
            _extract_js_function(js, "validateCaptionInstructionArchiveRows"),
            _extract_js_function(js, "validateCaptionInstructionReviewRows"),
            _extract_js_function(js, "validateCaptionInstructionReport"),
            _extract_js_function(js, "validateCaptionInstructionArtifactConsistency"),
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
            "  },",
            "  instruction_export_validation: { ok: true, error_count: 0, errors: [], row_count: 1 },",
            "  instruction_artifact_consistency: {",
            "    format: 'tator_caption_instruction_artifact_consistency_v1',",
            "    ok: true,",
            "    error_count: 0,",
            "    errors: [],",
            "  },",
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
            "  export_metadata: { selected_training_row_count: 1 },",
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
            "  review_decision: '',",
            "  review_notes: '',",
            "};",
            "const completePayload = {",
            "  instruction_report: report,",
            "  instruction_artifact_consistency: consistencyOk,",
            "  instruction_archive: { image_count: 1, instruction_artifact_consistency: consistencyOk },",
            "  instruction_training_rows: [trainingRow],",
            "  instruction_archive_rows: [archiveRow],",
            "  instruction_review_rows: [reviewRow],",
            "};",
            "const archiveValidation = validateCaptionInstructionArchiveRows([archiveRow]);",
            "assert.strictEqual(archiveValidation.ok, true);",
            "assert.strictEqual(validateCaptionInstructionArtifactConsistency(completePayload, 'archive', archiveValidation).ok, true);",
            "const duplicateArchive = validateCaptionInstructionArchiveRows([archiveRow, archiveRow]);",
            "assert.strictEqual(duplicateArchive.ok, false);",
            "assert(duplicateArchive.errors.some((error) => error.includes('duplicate archive image_path')));",
            "const duplicateArchiveAlias = validateCaptionInstructionArchiveRows([archiveRow, { ...archiveRow, image_path: './frame.jpg' }]);",
            "assert.strictEqual(duplicateArchiveAlias.ok, false);",
            "assert.strictEqual(duplicateArchiveAlias.imageCount, 1);",
            "assert(duplicateArchiveAlias.errors.some((error) => error.includes('duplicate archive image_path')));",
            "const canonicalAliasPayload = {",
            "  ...completePayload,",
            "  instruction_training_rows: [{ ...trainingRow, image_path: './frame.jpg' }],",
            "  instruction_archive_rows: [{ ...archiveRow, image_path: 'frame.jpg' }],",
            "  instruction_review_rows: [{ ...reviewRow, image_path: 'frame.jpg' }],",
            "};",
            "assert.strictEqual(validateCaptionInstructionArtifactConsistency(canonicalAliasPayload, 'training', { rowCount: 1 }).ok, true);",
            "const archiveMismatch = validateCaptionInstructionArtifactConsistency({ instruction_report: { ...report, image_count: 2, corpus_quality_metrics: { ...report.corpus_quality_metrics, image_count: 2 } }, instruction_archive: { image_count: 2 } }, 'archive', archiveValidation);",
            "assert.strictEqual(archiveMismatch.ok, false);",
            "assert(archiveMismatch.errors.some((error) => error.includes('archive row count 1 does not match report image count 2')));",
            "const reviewValidation = validateCaptionInstructionReviewRows([reviewRow]);",
            "assert.strictEqual(reviewValidation.ok, true);",
            "assert.strictEqual(validateCaptionInstructionArtifactConsistency({ instruction_report: report }, 'review', reviewValidation).ok, true);",
            "const reviewMismatch = validateCaptionInstructionArtifactConsistency({ instruction_report: { ...report, instruction_review_row_count: 2 } }, 'review', reviewValidation);",
            "assert.strictEqual(reviewMismatch.ok, false);",
            "assert(reviewMismatch.errors.some((error) => error.includes('review row count 1 does not match report review row count 2')));",
            "const trainingMismatch = validateCaptionInstructionArtifactConsistency({ instruction_report: { ...report, selected_flattened_row_count: 2, corpus_quality_metrics: { ...report.corpus_quality_metrics, selected_flattened_row_count: 2 }, instruction_export_validation: { ok: true, error_count: 0, errors: [], row_count: 2 } } }, 'training', { rowCount: 1 });",
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
            "const backendMismatch = validateCaptionInstructionArtifactConsistency({ instruction_report: report, instruction_artifact_consistency: { format: 'tator_caption_instruction_artifact_consistency_v1', ok: false, error_count: 1, errors: ['server mismatch'] } }, 'training', { rowCount: 1 });",
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
    assert "qwenBackendCrashSupervisionMessage()" in helper
    assert "Isolated caption job auto-resumed as ${autoResumeJobId}" in helper
    assert "latest_caption" in helper
    assert "caption backend job completed without a caption" in helper

    handle_start = js.index("async function handleQwenCaption")
    handle_end = js.index("function getCaptionImageList", handle_start)
    handle = js[handle_start:handle_end]
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
    assert "runQwenCaptionBatch(batch" in listener
    assert "backend: true" in listener

    batch_start = js.index("async function runQwenCaptionBatch")
    batch_end = js.index("function setQwenAgentStatus", batch_start)
    batch = js[batch_start:batch_end]
    assert "Backend dataset required" in batch
    assert "batch captioning uses isolated backend jobs so Metal crashes cannot take down" in batch
    assert "validateCaptionInstructionLaunchSettings(getCaptionInstructionDatasetSettings(true))" in batch
    assert "Instruction dataset not started" in batch
    assert "try {" in batch
    assert "runQwenCaptionBackendBatch(imageNames, { ...options, backend: true })" in batch
    assert "formatBackendFetchError(error" in batch
    assert "VLM training dataset" in batch
    assert "failed to start" in batch
    assert "setQwenCaptionBackendJobStatus(message)" in batch
    assert "invokeQwenCaptionForImage(" not in batch


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


def test_mobile_review_page_contract():
    html = _mobile_review_html()

    assert "<title>Tator Mobile Review</title>" in html
    assert 'id="sessionInput"' in html
    assert 'placeholder="Review session id"' in html
    assert 'id="preview" alt="Object context crop"' in html
    assert 'id="confirmButton" class="primary">Confirm current class</button>' in html
    assert 'id="skipButton" class="secondary">Skip</button>' in html
    assert 'id="targetClass"' in html
    assert 'id="changeButton" class="warn">Change class</button>' in html
    assert 'id="skipCount" type="number" min="1" max="1000" step="1" value="20"' in html
    assert 'id="skipNextButton" class="secondary">Skip next</button>' in html
    assert "/class_analysis/mobile_review/${encodeURIComponent(sessionId)}" in html
    assert 'action: "confirm"' in html
    assert 'action: "skip"' in html
    assert 'action: "change_class"' in html
    assert 'action: "skip_next"' in html
    assert "Change class to ${els.targetClass.value}" in html


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
    assert "does not use pixels, embeddings, Class Split analysis, Data Ingestion, or a reference profile" in html
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
    assert ".help-icon[data-tooltip]:focus-visible" in css
    assert ".help-icon[data-tooltip]:focus-visible::before" in css
    assert ".help-icon[data-tooltip]:focus-visible::after" in css
    assert "content: attr(data-tooltip);" in css


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
        "classSplitMobilePush",
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
    assert "qwenElements.captionDatasetSelect.disabled =\n                qwenCaptionDatasetRefreshInFlight || isAnnotationDatasetModeActive();" in js
    assert "if (isAnnotationDatasetModeActive()) {\n            return getActiveAnnotationDatasetIdForCaption();\n        }" in js
    assert "if (isAnnotationDatasetModeActive()) {\n                    syncCaptionDatasetSelectionWithAnnotationDataset();" in js
    assert "if (isAnnotationDatasetModeActive()) {\n                syncCaptionDatasetSelectionWithAnnotationDataset();\n            } else {" in js
    assert "qwenElements.captionDatasetSelect.disabled = false;" not in js
    assert "updateQwenCaptionDatasetRefreshButton" not in js


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
    assert 'id="classSplitEmbeddingAggregation"' not in html
    assert 'id="classSplitSaladHead"' not in html
    assert 'id="trainEmbeddingAggregation"' not in html
    assert 'id="trainSaladHead"' not in html
    assert '<option value="local_salad">Local SALAD separation</option>' not in html
    assert '<option value="local_salad">Local SALAD head</option>' not in html
    assert "Local SALAD requires a trained local head" not in html
    assert "Reference profiles are built from your own dataset images only" in html
    assert "classSplitElements.embeddingAggregation" not in js
    assert "classSplitElements.saladHead" not in js
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
    assert "Class Split or auto-class training" not in diversity_benchmark
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
    js = _js()
    css = _css()

    assert "preview complete prompt flow on image" in js
    assert "handleQwenCaptionPromptPreview" in js
    assert "/qwen/caption/preview_prompt" in js
    assert "invokeQwenCaptionPromptPreview" in js
    assert "buildQwenCaptionRequestFields(requestImageName)" in js
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


def test_qwen_caption_ui_scenarios_document_set_and_forget_workflows():
    scenarios = _read("docs/qwen_caption_ui_scenarios.md")
    html = _html()
    js = _js()

    for index, title in enumerate(
        [
            "Caption The Current Image With Defaults",
            "Run A Direct Diagnostic Caption",
            "Caption The Next N Images",
            "Caption All Images As A Walk-Away Job",
            "Launch A 10k-Scale Certified Run",
            "Reuse A Saved Caption Recipe",
            "Select A Missing Model Intentionally",
            "Use A Thinking Model Safely",
            "Caption Dense Label Scenes",
            "Auto-Recover Or Attach To Existing Work",
            "Export Alternate Captions For Training",
            "Create A VLM Instruction Dataset",
        ],
        start=1,
    ):
        assert f"## {index}. {title}" in scenarios

    assert "set-and-forget" in scenarios
    assert "Check caption readiness" in scenarios
    assert "direct diagnostic" in scenarios
    assert "Pilot min cases" in scenarios
    assert "300 or higher" in scenarios
    assert "deterministic-recovery confidence" in scenarios
    assert "Download-needed model options are red" in scenarios
    assert "local\nmodel options are white" in scenarios
    assert "Max output tokens" in scenarios
    assert "Max boxes" in scenarios
    assert "Attach / recover now" in scenarios
    assert "Download grouped JSON" in scenarios
    assert "Download VLM JSONL" in scenarios
    assert "Create VLM training dataset" in scenarios
    assert "Generated QA per image" in scenarios
    assert "0-20" in scenarios
    assert "Download instruction JSONL" in scenarios
    assert "Download instruction archive" in scenarios
    assert "Download review JSONL" in scenarios
    assert "Download instruction report" in scenarios
    assert "training-readiness block" in scenarios
    assert "`ready` / `needs_review` / `blocked`" in scenarios
    assert "Require\nready report for trainer JSONL" in scenarios
    assert "review-pending diagnostics" in scenarios
    assert "blank review decision/note fields" in scenarios
    assert "duplicate-question/diversity metrics" in scenarios
    assert "source-class coverage" in scenarios
    assert "generated QA never\nbecomes source annotations" in scenarios
    assert "duplicate canonical image-path/question pairs" in scenarios
    assert "VLM export validation status" in scenarios
    assert "No export imposes a per-image caption limit" in scenarios.replace("\n", " ")
    assert "bad\nrows are blocked instead of downloaded" in scenarios
    assert "Make generated caption primary" in scenarios
    assert "Generated caption jobs append variants by default" in scenarios
    assert "primary-first order" in scenarios
    assert "generated captions append as saved alternate caption records" in scenarios
    assert "qwenCaptionSetAndForget" in html
    assert "qwenCaptionPilotDeterministicRecoveryConfidence" in html
    assert "qwenCaptionAllowModelDownload" in html
    assert "qwen-model-option--download" in js
    assert "qwenCaptionBackendJobAutoResumeId" in js


def test_qwen_caption_backend_batch_explicitly_keeps_going_after_failures():
    html = _html()
    js = _js()

    single_start = js.index("async function runQwenCaptionSingleBackendJob")
    single_end = js.index("async function runQwenCaptionBackendBatch", single_start)
    single_helper = js[single_start:single_end]
    batch_start = single_end
    batch_end = js.index("function setQwenAgentStatus", batch_start)
    batch_helper = js[batch_start:batch_end]
    finish_start = js.index("async function finishQwenCaptionBackendJob")
    finish_end = js.index("async function monitorQwenCaptionBackendJob", finish_start)
    finish_helper = js[finish_start:finish_end]

    assert "max_failures: 0" in single_helper
    assert "max_failures: 0" in batch_helper
    assert "Backend batch complete • ${finalFailed} failed" in finish_helper
    assert "Backend caption batch complete with ${finalFailed} failed image" in finish_helper
    assert "qwenCaptionSetAndForget" in html
    assert "Set-and-forget backend run" in html
    assert "qwenCaptionAllowModelDownload" in html
    assert "Allow model downloads" in html
    assert "qwenCaptionAttempts" in html
    assert "VLM attempts" in html
    assert "Auto: set-and-forget uses 3 attempts" in html
    assert "qwenCaptionArtifactLogMb" in html
    assert "Attempt log cap (MB)" in html
    assert "qwenCaptionMaxRecoveryRate" in html
    assert "Max recovery rate" in html
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
    assert "save_text_labels: qwenElements.captionSaveText?.checked !== false" in single_helper
    assert "set_and_forget: setAndForget" in single_helper
    assert "allow_model_download: !!qwenElements.captionAllowModelDownload?.checked" in single_helper
    assert "runner_artifact_log_bytes: getCaptionRunnerArtifactLogBytes()" in single_helper
    assert "...healthGates" in single_helper
    assert "...pilotCertification" in single_helper
    assert "Isolated caption job auto-resumed as ${autoResumeJobId}" in single_helper
    assert "qwenCaptionBackendJobAutoResumeId(job)" in single_helper
    assert "const setAndForget = qwenElements.captionSetAndForget?.checked !== false" in batch_helper
    assert "const pilotCertification = getCaptionPilotCertificationSettings(setAndForget)" in batch_helper
    assert "set_and_forget: setAndForget" in batch_helper
    assert "allow_model_download: !!qwenElements.captionAllowModelDownload?.checked" in batch_helper
    assert "runner_artifact_log_bytes: getCaptionRunnerArtifactLogBytes()" in batch_helper
    assert "const healthGates = getCaptionHealthGateSettings()" in batch_helper
    assert "...healthGates" in batch_helper
    assert "...pilotCertification" in batch_helper
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
    assert "max_loop_recovery_case_rate" in js
    assert "max_deterministic_recovery_case_rate" in js
    assert "max_failed_attempt_row_rate" in js
    assert "max_signal_exit_attempt_row_rate" in js
    assert "min_rate_cases" in js
    assert "DEFAULT_CAPTION_HEALTH_MAX_LOOP_RECOVERY_RATE = 0.05" in js
    assert "DEFAULT_CAPTION_HEALTH_MAX_DETERMINISTIC_RECOVERY_RATE = 0.01" in js
    assert "DEFAULT_CAPTION_HEALTH_MAX_SIGNAL_EXIT_RATE = 0.05" in js
    assert "DEFAULT_CAPTION_SET_AND_FORGET_ATTEMPTS = 3" in js
    assert "function getCaptionBackendAttempts" in js
    assert "attempts: getCaptionBackendAttempts(setAndForget)" in js
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
    assert "job?.message || job?.error" in js
    assert "auto_resumed_job_id" in js
    assert "Backend batch auto-resumed as ${autoResumeJobId}" in js
    assert "qwenCaptionResumeBackendJob" in html
    assert "Attach / recover now" in html
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
    assert "Open Class Split setup" in html
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
    assert "Class Split Dataset Analysis is not required for this map" in js
    assert "This does not require Class Split Dataset Analysis" in html
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
    assert "Reopen it from the dataset card for persistent edits." in js
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
    assert 'id="classSplitScopeSelected"' in html
    assert 'id="classSplitScopeAll"' in html
    assert 'id="classSplitEncoderType"' in html
    assert 'id="classSplitBackbone"' in html
    assert '<option value="precise" selected>Precise best</option>' in html
    assert '<option value="cradio">C-RADIOv4 summary</option>' in html
    assert '<option value="local_salad">Local SALAD separation</option>' not in html
    assert 'Graph projection<span class="help-icon"' in html
    assert 'class="class-split-field class-split-field--projection"' in html
    assert html.index('id="classSplitProjection"') < html.index('id="classSplitRecipePreset"')
    assert '<option value="class_balanced_pca" selected>Class-balanced PCA</option>' in html
    assert '<option value="global_pca">Global PCA</option>' in html
    assert '<option value="between_class_pca">Between-class PCA</option>' in html
    assert '<option value="within_filter_pca">Within-filter PCA</option>' in html
    assert '<option value="umap">UMAP if available</option>' in html
    assert 'id="classSplitProjectionHint"' in html
    assert "Projection guide" in html
    assert "UMAP</strong><span>Best for selected-class subclass islands" in html
    assert "Class-balanced PCA</strong><span>Best all-class default" in html
    assert 'Scope<span class="help-icon"' in html
    assert 'Class<span class="help-icon"' in html
    assert 'Encoder<span class="help-icon"' in html
    assert 'Backbone<span class="help-icon"' in html
    assert "DINOv3 Precise is the stable default" in html
    assert "C-RADIOv4 is a slower opt-in comparison path" in html
    assert 'class="embedding-benchmark-note" open' not in html
    assert "Neighbor agreement is a clustering signal, not classifier accuracy" in html
    assert "Useful when you want to audit whether another visual backbone separates your dataset better" in html
    assert "on Mac it uses local MLX when available" in html
    assert 'UMAP neighbors<span class="help-icon"' in html
    assert 'UMAP min distance<span class="help-icon"' in html
    assert 'id="classSplitProjectionMinDist" min="0" max="0.99" step="0.01" value="0.08"' in html
    assert 'id="classSplitProjectionNeighborK" min="0" max="5000" value="50"' in html
    assert 'id="classSplitSampleCap" min="0" max="50000" placeholder="All objects"' in html
    assert 'Crop padding<span class="help-icon"' in html
    assert '<option value="tight_context" selected>Tight + context</option>' in html
    assert 'id="classSplitRecipeExplanation"' in html
    assert 'id="classSplitPreprocessMode"' in html
    assert 'id="classSplitSizeBiasMode"' in html
    assert "Canonical square resize" in html
    assert "Remove size/aspect bias" in html
    assert "It is not full whitening" in html
    assert 'Scoring neighbors<span class="help-icon"' in html
    assert 'value="5000"' not in html
    assert 'id="classSplitGraph" class="class-split-graph"' in html
    assert 'id="classSplitDisplayMode"' in html
    assert 'id="classSplitGraphProjection"' in html
    assert 'id="classSplitDragMode"' in html
    assert '<option value="pan">Pan</option>' in html
    assert '<option value="wrong_only">Likely wrong class only</option>' in html
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
    assert "benchmark carefully before promoting any C-RADIO pooling mode" in html
    assert 'id="classSplitReport" class="class-split-report"' in html
    assert 'id="classSplitBulkPanel" class="class-split-bulk-panel" hidden' in html
    assert 'id="classSplitGraphStatus" class="class-split-graph-status" aria-live="polite"' in html
    assert 'id="classSplitClusterPanel" class="class-split-review-section class-split-cluster-panel" open' in html
    assert 'id="classSplitClusterList" class="class-split-cluster-list"' in html
    assert 'id="classSplitWrongPanel" class="class-split-review-section class-split-wrong-panel class-split-wrong-panel--wide" open' in html
    assert 'id="classSplitWrongQueueStatus"' in html
    assert 'id="classSplitWrongDiscardCount" min="1" max="1000" step="1" value="20"' in html
    assert 'id="classSplitWrongDiscardFirst"' in html
    assert 'id="classSplitWrongShuffle"' in html
    assert 'id="classSplitQwenReviewMechanism"' in html
    assert "How VLM vignette review works" in html
    assert 'id="classSplitQwenReviewTraceToggle"' in html
    assert 'id="classSplitQwenReviewTraceToast"' in html
    assert 'id="classSplitQwenReviewTraceBody"' in html
    assert 'id="classSplitQwenReviewTraceClose"' in html
    assert 'id="classSplitWrongList"' in html
    assert 'id="classSplitInspector"' in html
    assert '<option value="image_value">Image value</option>' in html
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
    assert ".class-split-qwen-mechanism" in css
    assert ".class-split-qwen-trace-toast" in css
    assert ".class-split-qwen-trace-toast__body" in css
    assert "width: min(780px, calc(100vw - 36px));" in css
    assert "--class-split-qwen-review-bg" in css
    assert "html.theme-pipboy .class-split-qwen-review" in css
    assert ".class-split-wrong-item__preview" in css
    assert ".class-split-wrong-item__badge--dual" in css
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
    assert '"bulk bulk"\n        "graph review"\n        "footer footer"\n        "wrong wrong"' in css

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
    assert "projectionNeighborK.disabled = !available || classSplitState.active || projectionChoice !== \"umap\"" in js
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
    assert "Graph projection" in js
    assert "Within-filter PCA" in js
    assert "Size-axis check" in js
    assert "Crop cache" in js
    assert "Embedding cache" in js
    assert '["Aggregation",' not in js
    assert '["SALAD head",' not in js
    assert "Hold Shift over this tab to switch the graph" not in js
    assert "function setClassSplitGraphPanMode" not in js
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
    assert "const focusPromise = window.Plotly.relayout" in js
    assert "focusPromise\n            .then(() => {" in js
    assert "return window.Plotly.react(graphEl, traces, layout, config).then(() => {" in js
    assert "const plotRender = renderClassSplitPlot();" in js
    assert "Promise.resolve(plotRender).then(() => {" in js
    assert "classSplitState.selectedPointId === safeId" in js
    assert 'focusPlot\n            && classSplitElements.displayMode' in js
    assert 'String(classSplitElements.displayMode.value || "all") === "wrong_only"' in js
    assert "!point.is_wrong_class_candidate" in js
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
    assert "Discarded ${discardedIds.length} likely-wrong vignette" in js
    assert "function pushClassSplitReviewToMobile" in js
    assert "function syncClassSplitMobileReviewEdits" in js
    assert "annotation_session_id: annotationSourceState.lockSessionId" in js
    push_start = js.index("async function pushClassSplitReviewToMobile")
    push_end = js.index("async function syncClassSplitMobileReviewEdits", push_start)
    push_body = js[push_start:push_end]
    assert "target_mode:" not in push_body
    assert '["linked", "transient", "active_workspace"].includes(sourceMode)' in push_body
    assert "Backend-backed edits save directly." not in js
    assert "saves to the backend dataset directly" not in js
    assert "classSplitState.mobileReviewTargetMode" in js
    assert "classSplitElements.mobileSync.addEventListener" in js
    assert "Sync mobile edits when you return." in js
    assert "applyClassSplitPointClassLocally(point, targetClass)" in js
    assert "action.action_id" in js
    assert "mobile_review" in js
    assert "classSplitMobilePush" in html
    assert "classSplitMobileSync" in html
    assert "Push to mobile" in html
    assert "Sync mobile edits" in html
    assert "mobile_review.html" in router or "mobile_review.html" in _read("localinferenceapi.py")
    assert "Close overlaps only" in html
    assert "classSplitOverlapPairMode" in html
    assert "is_close_overlap_candidate" in js
    assert "classSplitPointMatchesOverlapPairFilter" in js
    assert "close_overlap_candidates" in _read("localinferenceapi.py")
    assert "function reconcileClassSplitWrongQueue" in js
    assert "function shuffleClassSplitWrongQueue" in js
    assert "const cap = 12;" in js
    assert "Reassign" in js
    assert "Switch class to ${suggestedClass}" in js
    assert ">Choose class</option>" in js
    assert "Review dual bbox with Qwen" in js
    assert "dual_bbox_resolution" in js
    assert "Dual-box conflict" in js
    assert "function getClassSplitContextCropUrl" in js
    assert "const maxPreviewDim = 1400;" in js
    assert 'alt="Object context crop"' in js
    assert "point.cluster_id = null;" in js
    assert "imageKeys: activeKeys" in js
    assert "imageKeys," in js
    assert "Changed class to ${targetClass}. ${saveStatus}" in js
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
    assert "Class Split vignette jump failed" in js
    assert "function changeClassSplitPointClass" in js
    assert "const previousClass = String(match.className || graphClass).trim();" in js
    assert "captureAnnotationDirtyStateForImage(imageKey)" in js
    assert "async function ensureClassSplitSnapshotClean" in js
    assert "captureCurrentAnnotationDirtyState();" in js
    assert 'await ensureClassSplitSnapshotClean("Class Split analysis")' in js
    assert "Class Split analysis is running. The graph will appear when the backend finishes embedding and projection." in js
    assert "startClassSplitAnalysis({ reuseLast: true })" in js
    assert "classSplitState.pollRequestId += 1;" in js[js.index("async function startClassSplitAnalysis"):js.index("function stopClassSplitPoll")]
    assert "initClassSplitExplorer();" in js
    assert "function renderClassSplitQwenReviewModelOptions" in js
    assert "metadata.inference_supported === false || metadata.vision_inference_supported === false" in js
    assert "metadata.display_name || metadata.label || metadata.name || entry.label" in js
    assert "max_files=float(\"inf\")" in router
    assert "max_part_size=512 * 1024 * 1024" in router
