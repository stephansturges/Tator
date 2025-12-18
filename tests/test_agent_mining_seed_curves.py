from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localinferenceapi import (  # noqa: E402
    _build_seed_threshold_sweep_grid,
    _compute_seed_threshold_curve,
    _select_seed_threshold_operating_point,
)


def test_seed_threshold_grid_includes_base_and_zero() -> None:
    grid = _build_seed_threshold_sweep_grid(base_seed_threshold=0.05, observed_scores=[0.9, 0.1, 0.02], limit=64)
    assert isinstance(grid, list)
    assert grid == sorted(grid)
    assert all(0.0 <= t <= 1.0 for t in grid)
    assert 0.0 in grid
    assert any(abs(t - 0.05) < 1e-9 for t in grid)


def test_seed_threshold_curve_monotonic_counts() -> None:
    gt_best_scores = {"a": 0.9, "b": 0.6, "c": 0.55}
    fp_scores = [0.8, 0.4, 0.2]
    thresholds = [0.0, 0.5, 0.6, 0.9]
    curve = _compute_seed_threshold_curve(gt_best_scores=gt_best_scores, fp_scores=fp_scores, thresholds=thresholds)
    assert [p["threshold"] for p in curve] == sorted(thresholds)
    matches = [p["matches"] for p in curve]
    fps = [p["fps"] for p in curve]
    assert matches == sorted(matches, reverse=True)
    assert fps == sorted(fps, reverse=True)
    assert curve[0]["matches"] == 3
    assert curve[1]["matches"] == 3
    assert curve[2]["matches"] == 2
    assert curve[3]["matches"] == 1


def test_select_operating_point_min_precision_prefers_max_matches() -> None:
    gt_best_scores = {"a": 0.9, "b": 0.6, "c": 0.55}
    fp_scores = [0.8, 0.4, 0.2]
    curve = _compute_seed_threshold_curve(
        gt_best_scores=gt_best_scores,
        fp_scores=fp_scores,
        thresholds=[0.0, 0.5, 0.6, 0.9],
    )
    point = _select_seed_threshold_operating_point(curve, min_precision=0.7)
    assert point is not None
    assert abs(point["threshold"] - 0.5) < 1e-9
    assert point["matches"] == 3
    assert point["fps"] == 1
    assert point["precision"] >= 0.7


def test_select_operating_point_max_fps_can_force_high_threshold() -> None:
    gt_best_scores = {"a": 0.9, "b": 0.6, "c": 0.55}
    fp_scores = [0.8, 0.4, 0.2]
    curve = _compute_seed_threshold_curve(
        gt_best_scores=gt_best_scores,
        fp_scores=fp_scores,
        thresholds=[0.0, 0.5, 0.6, 0.9],
    )
    point = _select_seed_threshold_operating_point(curve, max_fps=0)
    assert point is not None
    assert abs(point["threshold"] - 0.9) < 1e-9
    assert point["fps"] == 0


def test_select_operating_point_fallback_when_unreachable_constraints() -> None:
    gt_best_scores = {"a": 0.6, "b": 0.6}
    fp_scores = [0.9, 0.9]
    curve = _compute_seed_threshold_curve(
        gt_best_scores=gt_best_scores,
        fp_scores=fp_scores,
        thresholds=[0.0, 0.5, 0.9],
    )
    point = _select_seed_threshold_operating_point(curve, min_precision=0.9)
    assert point is not None
    assert abs(point["threshold"] - 0.0) < 1e-9
    assert point["precision"] == 0.5
