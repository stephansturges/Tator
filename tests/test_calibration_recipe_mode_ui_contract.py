from pathlib import Path


def test_calibration_recipe_mode_ui_controls_exist() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    html_text = (repo_root / "ybat-master" / "tator.html").read_text(encoding="utf-8")
    js_text = (repo_root / "ybat-master" / "ybat.js").read_text(encoding="utf-8")

    for control_id in (
        "qwenCalibrationRecipeMode",
        "qwenCalibrationLaneSelection",
        "qwenCalibrationRecipeInfo",
    ):
        assert f'id="{control_id}"' in html_text
        assert control_id in js_text

    assert "recipe_mode: recipeMode || \"auto\"" in js_text
    assert "lane_selection: laneSelection || \"window\"" in js_text
    assert "EDR mode" in html_text
    assert "EDR Builder" in html_text
    assert 'id="qwenCanonicalRecipeSelect"' in html_text
    assert 'id="qwenCanonicalRecipeLoad"' in html_text
    assert 'id="qwenCanonicalRecipeUse"' in html_text
    assert "Canonical EDRs" in html_text
    assert 'id="qwenAutoLabelRun"' not in html_text
    assert 'id="qwenAutoLabelCancel"' not in html_text
    assert 'id="qwenAutoLabelWindowMode"' not in html_text
    assert 'id="qwenAutoLabelModeSummary"' not in html_text
    assert "canonicalRecipeSelect" in js_text
    assert "loadCanonicalRecipeIntoBuilder" in js_text
    assert "useCanonicalRecipeForInference" in js_text
    assert "startAutoLabelJob" in js_text
    assert "cancelAutoLabelJob" in js_text
    assert "buildAutoLabelPayload" in js_text
    assert "updateAutoLabelModeSummary" in js_text
    assert "chooseAutoLabelTargetMode" in js_text
    assert "inferAutoLabelModeFromManifest" in js_text
    assert "/auto_label/jobs" in js_text
    assert "annotation_session_id: String(annotationSourceState.lockSessionId || \"\").trim() || null" in js_text
    assert 'fetchAutomationJobs(`${API_ROOT}/auto_label/jobs`)' in js_text
    assert 'pushActive(autoLabelJobs, "Auto Label")' in js_text
    assert "resolveCaptionPersistenceContext" in js_text
    assert "captureAnnotationDirtyStateForImage" in js_text
    assert "Automatic Labeling is running on this annotation session" in js_text
    assert "hydrationStateByKey: new Map()" in js_text
    assert "Annotations loading for current image." in js_text
    assert "annotations are still loading for this image" in js_text
    assert "tator.annotationEditorId" in js_text
    assert "buildSavedEdrRecipeConfig" in js_text
    assert "qwenActiveInferenceRecipe" in js_text
    assert "recipe_source_dataset_id" in js_text
    assert "edr_package_id" in js_text
    assert "getQwenAgentInferenceDatasetId" in js_text
    assert "const datasetId = getQwenAgentInferenceDatasetId(imageRecord);" in js_text
    assert 'datasetId: String(annotationSourceState.datasetId || "").trim()' in js_text
    assert "qwenAutoLabelModeSummary" not in html_text
    assert "qwenAutoLabelUnlabeledOnly" not in html_text
    assert "canonical bundle" in js_text
    assert "source_weighted" in html_text
    assert "edr_runtime_mode" in js_text
    assert "getSelectedAgentEnsembleJobId" in js_text
    assert "const genericRecipeItems = recipeItems.filter((item) => !isCanonicalPrepassRecipeItem(item));" in js_text
    assert 'setSamStatus("Canonical EDR is now active for Label Images."' in js_text
    assert "recipe_kind" in js_text
    assert "qwenCalibrationPolicyLayerVariant" not in html_text
    assert "qwenCalibrationPolicyLayerVariant" not in js_text
    set_current_image_idx = js_text.index("function setCurrentImage(image) {")
    deferred_idx = js_text.index("const runDeferredCurrentImageSelection = (expectedName) => {", set_current_image_idx)
    prepare_idx = js_text.index('prepareSamForCurrentImage({ messagePrefix, immediate: true })', deferred_idx)
    caption_idx = js_text.index("loadCaptionForCurrentImage().catch((error) => {", deferred_idx)
    assert prepare_idx < caption_idx
