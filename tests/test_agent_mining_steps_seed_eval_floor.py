from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localinferenceapi import (  # noqa: E402
    AgentMiningRequest,
    _compute_steps_seed_eval_max_results,
    _compute_steps_seed_eval_threshold,
)


def test_steps_seed_eval_threshold_defaults_to_base() -> None:
    payload = AgentMiningRequest(dataset_id="d", seed_threshold=0.05)
    assert abs(_compute_steps_seed_eval_threshold(payload) - 0.05) < 1e-9


def test_steps_seed_eval_threshold_honors_floor_when_lower() -> None:
    payload = AgentMiningRequest(dataset_id="d", seed_threshold=0.05, steps_seed_eval_floor=0.0)
    assert abs(_compute_steps_seed_eval_threshold(payload) - 0.0) < 1e-9


def test_steps_seed_eval_threshold_ignores_floor_when_higher() -> None:
    payload = AgentMiningRequest(dataset_id="d", seed_threshold=0.05, steps_seed_eval_floor=0.2)
    assert abs(_compute_steps_seed_eval_threshold(payload) - 0.05) < 1e-9


def test_steps_seed_eval_max_results_defaults_to_base() -> None:
    payload = AgentMiningRequest(dataset_id="d", max_results=123)
    assert _compute_steps_seed_eval_max_results(payload) == 123


def test_steps_seed_eval_max_results_honors_override() -> None:
    payload = AgentMiningRequest(dataset_id="d", max_results=123, steps_seed_eval_max_results=50)
    assert _compute_steps_seed_eval_max_results(payload) == 50

