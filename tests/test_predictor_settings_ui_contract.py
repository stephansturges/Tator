from pathlib import Path


def test_training_gpu_refresh_reads_predictor_settings_snake_case_fields():
    repo_root = Path(__file__).resolve().parents[1]
    js_text = (repo_root / "ybat-master" / "ybat.js").read_text(encoding="utf-8")

    assert "function readPredictorSettingsNumber" in js_text
    assert 'readPredictorSettingsNumber(data, "gpu_total_mb", "gpuTotalMb")' in js_text
    assert 'readPredictorSettingsNumber(data, "gpu_device_count", "gpuDeviceCount")' in js_text
    assert 'typeof data.gpuTotalMb === "number"' not in js_text
    assert 'typeof data.gpuDeviceCount === "number"' not in js_text
