from __future__ import annotations

import json
import sys
from pathlib import Path

from tools import run_canonical_prepass_discovery as runner


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_canonical_prepass_discovery_composes_recipe_from_decision_summaries(
    monkeypatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"

    _write_json(
        run_root / "results_ranked.json",
        {
            "views": {
                "intersection": {
                    "ranked_lanes": [
                        {"lane": "window", "mean_f1": 0.82},
                        {"lane": "nonwindow", "mean_f1": 0.81},
                    ]
                }
            },
            "winner": {"lane": "window"},
        },
    )
    _write_json(
        run_root / "results_raw.json",
        {
            "search_results": {
                "best_stack": {
                    "window": {
                        "hp": {"max_depth": 8, "n_estimators": 900},
                        "scenario": {"split_head": False, "sam_quality": True, "alpha": 0.5},
                        "policy": {
                            "threshold_by_class_override": {"person": 0.9},
                            "logit_bias_by_source_class": {
                                "sam3_text": {"__default__": -1.0},
                                "sam3_similarity": {"__default__": -0.8},
                            },
                            "sam_only_min_prob_default": 0.15,
                            "consensus_iou_default": 0.7,
                            "consensus_class_aware": True,
                        },
                    },
                    "nonwindow": {
                        "hp": {"max_depth": 12, "n_estimators": 600},
                        "scenario": {"split_head": True, "sam_quality": True, "alpha": 0.5},
                        "policy": {
                            "threshold_by_class_override": {"person": 0.92},
                            "logit_bias_by_source_class": {
                                "sam3_text": {"__default__": -1.0},
                                "sam3_similarity": {"__default__": -0.8},
                            },
                            "sam_only_min_prob_default": 0.15,
                            "consensus_iou_default": 0.7,
                            "consensus_class_aware": True,
                        },
                    },
                }
            }
        },
    )
    _write_json(
        run_root / "final_default_recipe.json",
        {"winner_lane": "window"},
    )
    _write_json(
        run_root / "postrun_sam_bias_magnitude_sweep" / "results_ranked.json",
        {
            "full": [{"tag": "text_m1p4__sim_m1p2"}],
            "pilot": [],
        },
    )
    _write_json(
        run_root / "postrun_alpha_extension" / "decision_summary.json",
        {
            "promoted_config": {"sam3_text_quality_alpha": 0.8},
        },
    )
    _write_json(
        run_root / "postrun_sam_bias_scope_ablation" / "decision_summary.json",
        {
            "promoted_config": {"sam_bias_scope": "sam_only"},
        },
    )
    _write_json(
        run_root / "postrun_sam_bias_magnitude_sweep" / "decision_summary.json",
        {
            "status": "promoted",
            "promoted_config": {
                "sam3_text_bias_default": -1.4,
                "sam3_similarity_bias_default": -1.2,
            },
            "full_winner": {"mean_f1": 0.8424, "mean_delta_vs_baseline_f1": 0.0021},
        },
    )
    _write_json(
        run_root / "postrun_similarity_quality_full_window_eval" / "decision_summary.json",
        {
            "status": "promoted",
            "winner_metrics": {"mean_f1": 0.8436, "mean_delta_f1": 0.0011},
            "promoted_config": {
                "train_sam3_similarity_quality": True,
                "sam3_similarity_quality_alpha": 0.5,
            },
        },
    )
    _write_json(
        run_root / "postrun_nonwindow_policy_confirmation" / "decision_summary.json",
        {
            "nonwindow_lane": "nonwindow",
            "refined_policy_status": "promoted",
            "similarity_quality_status": "rejected",
            "canonical_recipe": {
                "winner_lane": "nonwindow",
                "scenario": {
                    "split_head": True,
                    "train_sam3_text_quality": True,
                    "sam3_text_quality_alpha": 0.5,
                    "train_sam3_similarity_quality": False,
                    "sam3_similarity_quality_alpha": None,
                },
                "policy": {"sam_bias_scope": "sam_only"},
                "xgb_hparams": {"max_depth": 12, "n_estimators": 600},
                "expected_metrics": {"full_mean_f1": 0.8211},
            },
        },
    )

    calls = []

    class Result:
        def __init__(self) -> None:
            self.stdout = ""

    def fake_run(cmd, cwd=None, check=None):
        calls.append(list(cmd))
        return Result()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_canonical_prepass_discovery.py",
            "--run-root",
            str(run_root),
            "--lane-selection",
            "compare_both",
        ],
    )

    runner.main()

    payload = json.loads((run_root / "canonical_edr.json").read_text())
    progress = json.loads((run_root / runner.PROGRESS_FILENAME).read_text())
    assert (run_root / "canonical_prepass_recipe.json").exists()
    assert payload["promotion_status"]["windowed_similarity_quality"] == "promoted"
    assert payload["canonical_windowed_recipe"]["scenario"]["sam3_text_quality_alpha"] == 0.8
    assert payload["canonical_windowed_recipe"]["scenario"]["sam3_similarity_quality_alpha"] == 0.5
    assert payload["canonical_windowed_recipe"]["policy"]["sam_bias_scope"] == "sam_only"
    assert payload["canonical_windowed_recipe"]["policy"]["logit_bias_by_source_class"]["sam3_text"]["__default__"] == -1.4
    assert "second_stage_policy_layer" not in payload["canonical_windowed_recipe"]
    assert "second_stage_policy_layer" not in payload["canonical_nonwindowed_recipe"]
    assert payload["canonical_nonwindowed_recipe"]["winner_lane"] == "nonwindow"
    assert payload["canonical_nonwindowed_recipe"]["scenario"]["sam3_text_quality_alpha"] == 0.5
    assert progress["stage_key"] == "write_canonical_recipe"
    assert progress["status"] == "completed"
    assert progress["stage_total"] == 7


def test_canonical_prepass_discovery_respects_window_only_lane_selection(
    monkeypatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    _write_json(
        run_root / "results_ranked.json",
        {
            "views": {
                "intersection": {
                    "ranked_lanes": [
                        {"lane": "window", "mean_f1": 0.82},
                        {"lane": "nonwindow", "mean_f1": 0.81},
                    ]
                }
            },
            "winner": {"lane": "window"},
        },
    )
    _write_json(
        run_root / "results_raw.json",
        {
            "search_results": {
                "best_stack": {
                    "window": {
                        "hp": {"max_depth": 8, "n_estimators": 900},
                        "scenario": {"split_head": False, "sam_quality": True, "alpha": 0.5},
                        "policy": {},
                    },
                }
            }
        },
    )
    _write_json(run_root / "final_default_recipe.json", {"winner_lane": "window"})
    _write_json(run_root / "postrun_sam_bias_magnitude_sweep" / "results_ranked.json", {"full": [{"tag": "text_m1p4__sim_m1p2"}]})
    _write_json(run_root / "postrun_alpha_extension" / "decision_summary.json", {"promoted_config": {"sam3_text_quality_alpha": 0.8}})
    _write_json(run_root / "postrun_sam_bias_scope_ablation" / "decision_summary.json", {"promoted_config": {"sam_bias_scope": "sam_only"}})
    _write_json(
        run_root / "postrun_sam_bias_magnitude_sweep" / "decision_summary.json",
        {"status": "promoted", "promoted_config": {}, "full_winner": {"mean_f1": 0.84, "mean_delta_vs_baseline_f1": 0.0}},
    )
    _write_json(
        run_root / "postrun_similarity_quality_full_window_eval" / "decision_summary.json",
        {"status": "rejected", "winner_metrics": {"mean_f1": 0.84, "mean_delta_f1": 0.0}, "promoted_config": {}},
    )
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_canonical_prepass_discovery.py", "--run-root", str(run_root), "--lane-selection", "window"],
    )

    runner.main()

    payload = json.loads((run_root / "canonical_edr.json").read_text())
    progress = json.loads((run_root / runner.PROGRESS_FILENAME).read_text())
    assert (run_root / "canonical_prepass_recipe.json").exists()
    assert payload["canonical_windowed_recipe"]["winner_lane"] == "window"
    assert "second_stage_policy_layer" not in payload["canonical_windowed_recipe"]
    assert payload["canonical_nonwindowed_recipe"] == {}
    assert progress["stage_key"] == "write_canonical_recipe"
    assert progress["stage_total"] == 6
    assert progress["status"] == "completed"


def test_canonical_prepass_discovery_compare_both_nonwindow_uses_actual_stage_total(
    monkeypatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    _write_json(
        run_root / "results_ranked.json",
        {
            "views": {
                "intersection": {
                    "ranked_lanes": [
                        {"lane": "nonwindow", "mean_f1": 0.82},
                        {"lane": "window", "mean_f1": 0.81},
                    ]
                }
            },
            "winner": {"lane": "nonwindow"},
        },
    )
    _write_json(
        run_root / "results_raw.json",
        {
            "search_results": {
                "best_stack": {
                    "window": {
                        "hp": {"max_depth": 8, "n_estimators": 900},
                        "scenario": {"split_head": False, "sam_quality": True, "alpha": 0.5},
                        "policy": {},
                    },
                    "nonwindow": {
                        "hp": {"max_depth": 12, "n_estimators": 600},
                        "scenario": {"split_head": True, "sam_quality": True, "alpha": 0.5},
                        "policy": {},
                    },
                }
            }
        },
    )
    _write_json(run_root / "final_default_recipe.json", {"winner_lane": "nonwindow"})
    _write_json(
        run_root / "postrun_nonwindow_policy_confirmation" / "decision_summary.json",
        {
            "nonwindow_lane": "nonwindow",
            "refined_policy_status": "promoted",
            "similarity_quality_status": "rejected",
            "canonical_recipe": {
                "winner_lane": "nonwindow",
                "scenario": {
                    "split_head": True,
                    "train_sam3_text_quality": True,
                    "sam3_text_quality_alpha": 0.5,
                    "train_sam3_similarity_quality": False,
                    "sam3_similarity_quality_alpha": None,
                },
                "policy": {"sam_bias_scope": "sam_only"},
                "xgb_hparams": {"max_depth": 12, "n_estimators": 600},
                "expected_metrics": {"full_mean_f1": 0.8211},
            },
        },
    )

    calls = []

    def fake_run(cmd, cwd=None, check=None):
        calls.append(list(cmd))
        return None

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_canonical_prepass_discovery.py", "--run-root", str(run_root), "--lane-selection", "compare_both"],
    )

    runner.main()

    progress = json.loads((run_root / runner.PROGRESS_FILENAME).read_text())
    assert progress["stage_key"] == "write_canonical_recipe"
    assert progress["stage_total"] == 3
    assert [cmd[1] for cmd in calls] == [
        "tools/run_final_calibration_sweep.py",
        "tools/run_postrun_nonwindow_policy_confirmation.py",
    ]


def test_stage_output_reuse_requires_matching_sidecar_metadata(tmp_path: Path) -> None:
    output_path = tmp_path / "postrun_alpha_extension" / "decision_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("{}", encoding="utf-8")
    cmd = [sys.executable, "tools/run_postrun_alpha_extension.py", "--run-root", "/tmp/run"]

    assert not runner._is_stage_output_reusable(output_path, "alpha_extension", cmd)

    runner._write_stage_reuse_meta(output_path, "alpha_extension", cmd)

    assert runner._is_stage_output_reusable(output_path, "alpha_extension", cmd)
    assert not runner._is_stage_output_reusable(
        output_path,
        "alpha_extension",
        [sys.executable, "tools/run_postrun_alpha_extension.py", "--run-root", "/tmp/other"],
    )
