import pytest

from .helpers.ui import go_to_tab
from .helpers.usability import (
    assert_min_contrast,
    assert_min_font_size,
    assert_no_horizontal_overflow,
    assert_text_not_clipped,
    collect_soft_artifact,
)


pytestmark = [pytest.mark.ui, pytest.mark.ui_usability_full]


# CASE_ID: UX_TAB_PANELS_NO_OVERFLOW
def test_visible_tab_panels_do_not_overflow_horizontally(playwright_page):
    page, _ = playwright_page
    tab_map = [
        ("#tabLabelingButton", "#tabLabeling", "#imageList"),
        ("#tabPrepassBuilderButton", "#tabPrepassBuilder", "#qwenAgentRecipeSave"),
        ("#tabTrainingButton", "#tabTraining", "#startTrainingBtn"),
        ("#tabQwenTrainButton", "#tabQwenTrain", "#qwenTrainStartBtn"),
        ("#tabSam3TrainButton", "#tabSam3Train", "#sam3StartBtn"),
        ("#tabYoloTrainButton", "#tabYoloTrain", "#yoloTrainStartBtn"),
        ("#tabRfDetrTrainButton", "#tabRfDetrTrain", "#rfdetrTrainStartBtn"),
        ("#tabAgentMiningButton", "#tabAgentMining", "#agentRunBtn"),
        ("#tabPromptHelperButton", "#tabPromptHelper", "#promptHelperGenerateBtn"),
        ("#tabDatasetsButton", "#tabDatasets", "#datasetPathOpenBtn"),
        ("#tabSam3PromptModelsButton", "#tabSam3PromptModels", "#sam3PromptRefresh"),
        ("#tabDetectorsButton", "#tabDetectors", "#detectorDefaultSave"),
        ("#tabActiveButton", "#tabActive", "#activeClassifierUse"),
        ("#tabQwenButton", "#tabQwen", "#qwenSettingsApply"),
        ("#tabPredictorsButton", "#tabPredictors", "#predictorApply"),
        ("#tabSettingsButton", "#tabSettings", "#settingsApply"),
    ]
    for tab_button, panel, key_control in tab_map:
        go_to_tab(page, tab_button, panel)
        page.wait_for_selector(key_control, timeout=15000)
        assert_no_horizontal_overflow(page, panel)


@pytest.mark.ui_usability_smoke
# CASE_ID: UX_CRITICAL_TEXT_CONTRAST
def test_critical_status_text_has_readable_contrast_and_size(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabLabelingButton", "#tabLabeling")
    assert_min_font_size(page, "#annotationSourceSummary", min_px=12)
    assert_text_not_clipped(page, "#annotationSourceSummary")
    assert_min_contrast(page, "#annotationSourceSummary", threshold=4.3)

    page.evaluate(
        """() => {
            const banner = document.querySelector("#automationLockBanner");
            if (!banner) return;
            banner.textContent = "Training and prepass jobs are active. GPU-heavy tabs are temporarily disabled.";
            banner.style.display = "block";
        }"""
    )
    assert_text_not_clipped(page, "#automationLockBanner")
    assert_min_font_size(page, "#automationLockBanner", min_px=12)
    assert_min_contrast(page, "#automationLockBanner", threshold=4.5)
    collect_soft_artifact(page, "ux_critical_text_contrast")
