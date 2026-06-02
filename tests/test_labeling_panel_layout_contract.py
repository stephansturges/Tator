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
    assert "Show diversity metric / image value" in html
    assert 'id="annotationDiversityMetric"' in html
    assert 'data-testid="status.annotation.diversity_metric"' in html
    assert ".annotation-diversity-metric" in css
    assert "ANNOTATION_DIVERSITY_METRIC_STORAGE_KEY" in js
    assert "initAnnotationDiversityControls();" in js
    assert "scheduleAnnotationDiversityMetricRefresh();" in js
    assert "computeImageDiversityMetric" in helper
    assert "countBoxesByClassFromYoloLines" in helper


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

    assert "<strong>Images:</strong> Space = next image; Tab = previous image" in html
    assert "<strong>Focus:</strong> V toggles image-only focus mode" in html
    assert "<strong>Classes:</strong> ↓ / R = next class; ↑ / E = previous class" in html
    assert "Delete / Backspace / X removes selected or current bboxes" in html
    assert "S toggles SAM; W runs magic tweak; double W tweaks selected boxes first, otherwise the carousel class" in html
    assert "<strong>Point prompts:</strong> D toggles SAM point mode; M toggles multi-point" in html
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
    assert "const requestedMaxItems = Number.isFinite(options.maxVisibleItems)" in js
    assert "function scheduleClassSplitControlsRefresh" in js
    assert "syncSam3ClassToCurrent();" in js
    assert "scheduleClassSplitControlsRefresh();" in js
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
    assert "const isTextEditingTarget = (target) =>" in js
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


def test_qwen_caption_model_defaults_to_active_runtime():
    html = _html()
    select_match = re.search(r'<select id="qwenCaptionModel">(.*?)</select>', html, re.S)

    assert select_match
    options = select_match.group(1)
    assert '<option value="active" selected>Use active model</option>' in options
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
    assert "WALDO encoder benchmark comparison" in html
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
    assert 'id="classSplitUploadDatasetName"' in html
    assert "Workspace upload name" in html
    assert 'id="classSplitEncoderType"' in html
    assert 'id="classSplitBackbone"' in html
    assert '<option value="precise" selected>Precise best</option>' in html
    assert '<option value="cradio">C-RADIOv4 summary</option>' in html
    assert '<option value="local_salad">Local SALAD separation</option>' not in html
    assert 'Graph projection<span class="help-icon"' in html
    assert 'class="class-split-field class-split-field--projection"' in html
    assert html.index('id="classSplitProjection"') < html.index('id="classSplitRecipePreset"')
    assert html.index('id="classSplitProjection"') < html.index('id="classSplitUploadDatasetName"')
    assert '<option value="class_balanced_pca" selected>Class-balanced PCA</option>' in html
    assert '<option value="global_pca">Global PCA</option>' in html
    assert '<option value="between_class_pca">Between-class PCA</option>' in html
    assert '<option value="within_filter_pca">Within-filter PCA</option>' in html
    assert 'Scope<span class="help-icon"' in html
    assert 'Class<span class="help-icon"' in html
    assert 'Encoder<span class="help-icon"' in html
    assert 'Backbone<span class="help-icon"' in html
    assert "Full WALDO tests keep DINOv3 Precise as the stable default" in html
    assert "C-RADIOv4 improves NN purity only in a very slow opt-in audit path" in html
    assert 'class="embedding-benchmark-note" open' not in html
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
    assert 'id="classSplitDisplayMode"' in html
    assert 'id="classSplitGraphProjection"' in html
    assert 'id="classSplitDragMode"' in html
    assert '<option value="pan">Pan</option>' in html
    assert '<option value="wrong_only">Likely wrong class only</option>' in html
    assert '<option value="cluster">Cluster</option>' not in html
    assert 'id="classSplitClusterOverlay"' not in html
    assert 'id="classSplitClusterSensitivity"' in html
    assert 'id="classSplitClusterMaxClusters"' in html
    assert 'id="classSplitClusterMinSize"' in html
    assert 'id="classSplitClusterRun"' in html
    assert 'id="classSplitCradioPooling"' in html
    assert "benchmark carefully before promoting any C-RADIO pooling mode" in html
    assert 'id="classSplitReport" class="class-split-report"' in html
    assert 'id="classSplitBulkPanel" class="class-split-bulk-panel" hidden' in html
    assert 'id="classSplitGraphStatus" class="class-split-graph-status" aria-live="polite"' in html
    assert 'id="classSplitClusterPanel" class="class-split-review-section class-split-cluster-panel" open' in html
    assert 'id="classSplitClusterList" class="class-split-cluster-list"' in html
    assert 'id="classSplitWrongPanel" class="class-split-review-section class-split-wrong-panel class-split-wrong-panel--wide" open' in html
    assert 'id="classSplitWrongQueueStatus"' in html
    assert 'id="classSplitWrongShuffle"' in html
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
    assert ".class-split-wrong-item__preview" in css
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
    assert "initializeAdaptiveTopTabs();\n        setupTabNavigation();\n        applyPlaywrightTestIds();" in js
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
    assert "plotRenderToken" in js
    assert "projection_mode: projectionParts.projectionMode" in js
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
    assert "function reconcileClassSplitWrongQueue" in js
    assert "function shuffleClassSplitWrongQueue" in js
    assert "const cap = 12;" in js
    assert "Reassign" in js
    assert "Switch class to ${suggestedClass}" in js
    assert ">Choose class</option>" in js
    assert "function getClassSplitContextCropUrl" in js
    assert "const maxPreviewDim = 1400;" in js
    assert 'alt="Object context crop"' in js
    assert "point.cluster_id = null;" in js
    assert "imageKeys: activeKeys" in js
    assert "imageKeys," in js
    assert "Changed class to ${targetClass}. Save labels when ready." in js
    assert "Changed class to ${targetClass}; rerunning analysis." not in js
    assert "function drawClassSplitInstancePulse" in js
    assert "startClassSplitInstancePulse(match.bbox" in js
    assert "function getClassSplitServerSourceHandle" in js
    assert "function ensureClassSplitUploadedWorkspaceSourceHandle" in js
    assert "classSplitElements.uploadDatasetName" in js
    assert 'const uploadCacheKey = `${signature}|name:${uploadName}`;' in js
    assert "CLASS_SPLIT_ACTIVE_WORKSPACE_UPLOADS_STORAGE_KEY" in js
    assert "transport: \"chunked\"" in js
    assert "labelLinesForImage: (imageKey) => getClassSplitActiveLabelLines(imageKey)" in js
    assert "activeUploadSessionId" in js
    assert 'fetch(`${API_ROOT}/datasets/upload_session/${encodeURIComponent(uploadSessionId)}/cancel`' in js
    assert 'source_mode: sourceHandle.sourceMode' in js
    assert "payload.dataset_id = sourceHandle.datasetId" in js
    assert 'fetch(`${API_ROOT}/class_analysis/jobs`, {' in js
    assert 'headers: { "Content-Type": "application/json" }' in js
    assert 'fetch(`${API_ROOT}/class_analysis/jobs/active_workspace`' in js
    assert "window.Plotly.react" in js
    assert "function jumpToClassSplitPoint" in js
    assert "setActiveTab(TAB_LABELING)" in js
    assert "See instance" in js
    assert 'data-action="jump-instance"' in js
    assert 'listEl.querySelectorAll(\'[data-action="jump-instance"]\')' in js
    assert "Class Split vignette jump failed" in js
    assert "function changeClassSplitPointClass" in js
    assert "captureAnnotationDirtyStateForImage(imageKey)" in js
    assert "async function ensureClassSplitSnapshotClean" in js
    assert "captureCurrentAnnotationDirtyState();" in js
    assert 'await ensureClassSplitSnapshotClean("Class Split analysis")' in js
    assert "Class Split analysis is running. The graph will appear when the backend finishes embedding and projection." in js
    assert "startClassSplitAnalysis({ reuseLast: true })" in js
    assert "initClassSplitExplorer();" in js
    assert "max_files=float(\"inf\")" in router
    assert "max_part_size=512 * 1024 * 1024" in router
