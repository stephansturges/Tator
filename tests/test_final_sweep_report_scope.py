from pathlib import Path

from tools.run_final_calibration_sweep import _final_sweep_report_analysis_glob


def test_final_sweep_report_analysis_glob_is_limited_to_final_matrix():
    run_root = Path("/tmp/example_run")
    pattern = _final_sweep_report_analysis_glob(run_root)
    assert "final_matrix" in pattern
    assert pattern.endswith("*.analysis.json")
    assert "/**/" in pattern
