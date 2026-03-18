from pathlib import Path


def test_calibration_recipe_mode_ui_controls_exist() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    html_text = (repo_root / "ybat-master" / "ybat.html").read_text(encoding="utf-8")
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
    assert "qwenCalibrationPolicyLayerVariant" not in html_text
    assert "qwenCalibrationPolicyLayerVariant" not in js_text
