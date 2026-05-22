import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = REPO_ROOT / "ybat-master" / "ybat.html"
CSS_PATH = REPO_ROOT / "ybat-master" / "ybat.css"
JS_PATH = REPO_ROOT / "ybat-master" / "ybat.js"


def _html() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


def _css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def _js() -> str:
    return JS_PATH.read_text(encoding="utf-8")


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


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
    assert "Qwen 3 detection engine (not great)" in html
    assert "EDR [wip]" in html
    assert "Ensemble Detection Recipe" in html


def test_yolo_import_and_export_controls_live_in_annotation_source_panel():
    html = _html()
    source_start = html.index('id="annotationSourcePanel"')
    source_end = html.index('id="labelingGpuLockNotice"')
    source_panel = html[source_start:source_end]

    for control_id in ("bboxes", "bboxesFolder", "bboxesSelectFolder", "saveBboxes"):
        assert f'id="{control_id}"' in source_panel

    assert source_start < html.index('id="bboxes"') < html.index('id="qwenDetectionDetails"')
    assert source_start < html.index('id="saveBboxes"') < html.index('id="qwenDetectionDetails"')


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

    assert "Caption style<span class=\"help-icon\"" in html
    assert "Style prompts (one per line)<span class=\"help-icon\"" in html
    assert "Opening phrases (one per line)<span class=\"help-icon\"" in html
    assert "Caption prompt stack<span class=\"help-icon\"" in html
    assert 'id="qwenCaptionPresetRandom" class="training-button secondary" title=' in html
    assert 'id="qwenCaptionStyleList" rows="4"' in html
    assert 'id="qwenCaptionOpeningList" rows="6"' in html
    assert 'id="qwenCaptionSystemPrompt" rows="12"' in html

    assert "#qwenCaptionStyleList,\n#qwenCaptionOpeningList" in css
    assert "min-height: 240px;" in css
    assert "max-height: 520px;" in css


def test_keyboard_image_navigation_shortcuts_are_documented_and_guarded():
    html = _html()
    js = _js()

    assert "<strong>Images:</strong> Space = next image; Tab = previous image" in html
    assert "<strong>Focus:</strong> V toggles image-only focus mode" in html
    assert "<strong>Classes:</strong> ↓ / R = next class; ↑ / E = previous class" in html
    assert "Delete / Backspace / X removes selected or current bboxes" in html
    assert "W runs magic tweak; double W tweaks selected boxes first, otherwise the carousel class" in html
    assert "Shift + R + drag runs YOLO/RF-DETR region detect" in html
    assert "Shift + Y requests Save YOLO + captions" in html
    assert "const imageNavigationKey = (event) =>" in js
    assert "eventKey === 32" in js
    assert "eventKey === 9" in js
    assert 'event.key === "Tab"' in js
    assert "return 1;" in js
    assert "return -1;" in js
    assert "navigateImage(direction)" in js
    assert 'window.addEventListener("keydown"' in js
    assert '}, true);' in js
    assert "annotationWorkspaceHotkeysActive" in js
    assert "__tatorImageNavigationHandled" in js
    assert "canvas.element.focus({ preventScroll: true })" in js


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
    assert "const isTextEditingTarget = (target) =>" in js
    assert 'targetTag === "textarea"' in js


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
    assert "Reference profiles train local SALAD from your own dataset images only" in html
    assert 'id="dataIngestionFiles"' in html
    assert 'id="dataIngestionRecipe"' not in html
    assert 'id="dataIngestionTrainFiles"' not in html
    assert 'id="dataIngestionTrainActiveButton"' not in html
    assert 'id="dataIngestionReferenceActive"' in html
    assert 'id="dataIngestionReferenceBackend"' in html
    assert 'id="dataIngestionReferenceDataset"' in html
    assert 'id="dataIngestionBuildProfileButton"' in html
    assert 'id="dataIngestionMaxTrainImages" min="0" max="1000000" step="1" value="0"' in html
    assert 'id="dataIngestionReferenceCap"' not in html
    assert 'id="dataIngestionKeepFraction"' in html
    assert 'id="dataIngestionUseActiveReference"' not in html
    assert 'id="dataIngestionSaladHead"' in html
    assert "WALDO encoder benchmark comparison" in html
    assert "Data Ingestion" in html
    assert 'id="dataIngestionCradioModel"' not in html
    assert 'id="dataIngestionCradioPooling"' not in html
    assert 'id="dataIngestionTrainEncoder"' in html
    assert 'id="dataIngestionTrainCradioModel"' in html

    assert '.tab-panel[data-tab-panel="data-ingestion"]' in css
    assert ".data-ingestion-workspace" in css
    assert ".data-ingestion-results" in css
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
    assert "appendActiveWorkspaceTrainingFiles" in js
    assert "activeDatasetSaladHeadName" in js
    assert "getDataIngestionNumber(dataIngestionElements.maxTrainImages, 0" in js
    assert 'startLocalSaladTraining("active_dataset")' not in js
    assert "preferredSaladHeadId" in js
    assert 'setSelectValueIfPresent(dataIngestionElements.recipe, "local_salad_top20")' not in js
    assert 'fetch(`${API_ROOT}/data_ingestion/jobs`' in js
    assert 'fetch(`${API_ROOT}/data_ingestion/salad_train_jobs`' in js
    assert 'fetch(`${API_ROOT}/datasets`' in js
    assert "appendActiveWorkspaceReferenceFiles" in js
    assert "dataIngestionRecipeValues" not in js
    assert "dataIngestionElements.referenceDataset" in js
    assert "reference_dataset_id" in js
    assert "reference_source" in js
    assert "function dataIngestionHeadMatchesReference" in js
    assert "function handleDataIngestionReferenceChange" in js
    assert "heads.filter((head) => dataIngestionHeadMatchesReference(head))" in js
    assert "Choose matching reference profile" in js
    assert "No profiles for selected reference" in js
    assert "Selected reference profile does not match the chosen reference dataset" in js
    assert "encoder: \"local_salad\"" in js
    assert "dataIngestionElements.trainEncoder" in js
    assert "dataIngestionElements.trainCradioModel" in js
    assert "local_salad" in js

    assert '"/data_ingestion/capabilities"' in router
    assert '"/data_ingestion/salad_train_jobs"' in router
    assert 'max_part_size=1024 * 1024 * 1024' in router


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
    assert '(key === 87 || event.key === "w" || event.key === "W")' in js
    assert '(key === 88 || event.key === "x" || event.key === "X")' in js


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
    assert 'Projection<span class="help-icon"' in html
    assert 'Scope<span class="help-icon"' in html
    assert 'Class<span class="help-icon"' in html
    assert 'Encoder<span class="help-icon"' in html
    assert 'Backbone<span class="help-icon"' in html
    assert "Full WALDO tests keep DINOv3 Precise as the stable default" in html
    assert "C-RADIOv4 improves NN purity only in a very slow opt-in audit path" in html
    assert "its slowest recipe improved NN purity in full WALDO tests" in html
    assert "The first NN number is object-weighted; the second balances classes" in html
    assert "the visible C-RADIO preset starts from summary mode" in html
    assert "On Mac it now uses local MLX when available" in html
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
    assert 'id="classSplitCradioPooling"' in html
    assert "benchmark carefully before promoting any C-RADIO pooling mode" in html
    assert 'id="classSplitReport" class="class-split-report"' in html
    assert 'id="classSplitBulkPanel" class="class-split-bulk-panel" hidden' in html
    assert 'id="classSplitWrongPanel" class="class-split-review-section class-split-wrong-panel" open' in html
    assert 'id="classSplitWrongList"' in html
    assert 'id="classSplitInspector"' in html
    assert "Selected crop" in html
    assert "Likely wrong class" in html

    assert '.tab-panel[data-tab-panel="class-split"]' in css
    assert ".class-split-workspace" in css
    assert ".class-split-panel--workspace .class-split-results" in css
    assert ".class-split-bulk-panel" in css
    assert ".class-split-graph" in css
    assert ".class-split-report" in css
    assert ".class-split-graph.class-split-graph--pan" not in css
    assert ".class-split-review-section--inspector" in css
    assert ".class-split-hover-card" not in css
    assert "--class-split-crop-scale" in css
    assert "max-width: none;" in css
    assert "transform: scale(var(--class-split-crop-scale, 1));" in css
    assert ".embedding-recipe-note__text" in css
    assert ".class-split-review" in css
    assert "height: calc(100vh - 330px);" in css
    assert "html.theme-dark .class-split-panel" in css
    assert "html.theme-dark .class-split-workspace__header" in css
    assert "html.theme-pipboy .class-split-panel" in css

    assert 'const TAB_CLASS_SPLIT = "class-split";' in js
    assert "tabElements.classSplitButton = document.getElementById(\"tabClassSplitButton\")" in js
    assert "setActiveTab(TAB_CLASS_SPLIT)" in js
    assert "const classSplitElements = {" in js
    assert "const classSplitState = {" in js
    assert "function initClassSplitExplorer" in js
    assert "function startClassSplitAnalysis" in js
    assert "function getClassSplitSampleCap" in js
    assert "request.sample_cap = sampleCap" in js
    assert "projection_neighbor_k: projectionNeighborK" in js
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
    assert 'Crop geometry<span class="help-icon"' in html
    assert 'Background<span class="help-icon"' in html
    assert 'Embedding views<span class="help-icon"' in html
    assert 'DINOv3 pooling<span class="help-icon"' in html
    assert 'C-RADIOv4 pooling<span class="help-icon"' in html
    assert "Active Label Images workspace" in js
    assert "function buildClassSplitActiveWorkspaceForm" in js
    assert "function getClassSplitPointImageKey" in js
    assert "function renderClassSplitReport" in js
    assert "Projection neighbors" in js
    assert "Size-axis check" in js
    assert "Crop cache" in js
    assert "Embedding cache" in js
    assert '["Aggregation",' not in js
    assert '["SALAD head",' not in js
    assert "Hold Shift over this tab to switch the graph" not in js
    assert "function setClassSplitGraphPanMode" not in js
    assert "function panClassSplitPlotWithWheel" not in js
    assert 'dragmode: "lasso"' in js
    assert "__classSplitShiftWheelGuard" in js
    assert "event.stopImmediatePropagation();" in js
    assert "function rememberClassSplitSelectionFromPlot" in js
    assert "function changeClassSplitSelectedPointsClass" in js
    assert "function markClassSplitWrongCandidateCorrect" in js
    assert "function showClassSplitHoverCard" not in js
    assert "cropPreview.naturalWidth" in js
    assert "Math.min(shellWidth / naturalWidth, shellHeight / naturalHeight)" in js
    assert "Math.max(0.2, Math.min(16, cropZoom * factor))" in js
    assert "new ResizeObserver(updateCropFitScale)" in js
    assert "function focusClassSplitPlotOnPoint" in js
    assert "classSplitElements.bulkPanel" in js
    assert "panClassSplitPlotWithWheel" not in js
    assert "Correct class" in js
    assert "function drawClassSplitInstancePulse" in js
    assert "startClassSplitInstancePulse(match.bbox" in js
    assert 'fetch(`${API_ROOT}/class_analysis/jobs/active_workspace`' in js
    assert "window.Plotly.react" in js
    assert "function jumpToClassSplitPoint" in js
    assert "setActiveTab(TAB_LABELING)" in js
    assert "See instance" in js
    assert "function changeClassSplitPointClass" in js
    assert "captureAnnotationDirtyStateForImage(imageKey)" in js
    assert "async function ensureClassSplitSnapshotClean" in js
    assert "captureCurrentAnnotationDirtyState();" in js
    assert "startClassSplitAnalysis({ reuseLast: true })" in js
    assert "initClassSplitExplorer();" in js
    assert "max_files=float(\"inf\")" in router
    assert "max_part_size=512 * 1024 * 1024" in router
