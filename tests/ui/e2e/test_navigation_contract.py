import pytest

from .helpers.ui import go_to_tab


pytestmark = [pytest.mark.ui, pytest.mark.ui_full]


# CASE_ID: NAV_ALL_TABS_RENDER

def test_all_tab_buttons_open_expected_panels(playwright_page):
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
        ("#tabDataIngestionButton", "#tabDataIngestion", "#dataIngestionBuildProfileButton"),
        ("#tabClassSplitButton", "#tabClassSplit", "#classSplitRunButton"),
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
        panel_visible = page.eval_on_selector(panel, "el => !!el && getComputedStyle(el).display !== 'none'")
        assert panel_visible is True

    # Return to labeling as default resting state.
    go_to_tab(page, "#tabLabelingButton", "#tabLabeling")


def test_enter_in_labeling_search_does_not_submit_or_reload_app(playwright_page):
    page, _ = playwright_page
    go_to_tab(page, "#tabLabelingButton", "#tabLabeling")
    page.wait_for_selector("#imageSearch", timeout=15000)

    page.evaluate("() => { window.__tatorNoSubmitMarker = 'still-here'; }")
    page.fill("#imageSearch", "img")
    page.press("#imageSearch", "Enter")
    page.wait_for_timeout(250)

    assert page.evaluate("() => window.__tatorNoSubmitMarker") == "still-here"
    assert page.eval_on_selector("#tabLabeling", "el => !!el && getComputedStyle(el).display !== 'none'") is True
