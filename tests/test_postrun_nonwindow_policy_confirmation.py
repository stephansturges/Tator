from __future__ import annotations

import json
import sys
from pathlib import Path

from tools import run_postrun_nonwindow_policy_confirmation as runner


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_nonwindow_policy_confirmation_promotes_refined_policy(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    lane = "nonwindow"
    seed = 42

    _write_json(
        run_root / "results_ranked.json",
        {
            "views": {
                "intersection": {
                    "ranked_lanes": [
                        {"lane": "window", "mean_f1": 0.82},
                        {"lane": lane, "mean_f1": 0.81},
                    ]
                }
            }
        },
    )
    _write_json(
        run_root / "results_raw.json",
        {
            "search_results": {
                "best_stack": {
                    lane: {
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
                    }
                }
            }
        },
    )
    _write_json(
        run_root / "lane_manifest.json",
        {
            "lanes": {
                lane: {
                    "variant": "nonwindow",
                    "labeled": str((run_root / "lanes" / lane / "labeled.npz").resolve()),
                    "prepass_jsonl": str((run_root / "prepass" / "nonwindow.jsonl").resolve()),
                }
            }
        },
    )
    labeled_npz = run_root / "lanes" / lane / "labeled.npz"
    labeled_npz.parent.mkdir(parents=True, exist_ok=True)
    labeled_npz.write_bytes(b"stub")
    prepass_jsonl = run_root / "prepass" / "nonwindow.jsonl"
    prepass_jsonl.parent.mkdir(parents=True, exist_ok=True)
    prepass_jsonl.write_text("", encoding="utf-8")

    seed_dir = run_root / "final_matrix" / lane / "full" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "model_foo.json").write_text("{}", encoding="utf-8")
    _write_json(
        seed_dir / "model_foo.meta.json",
        {
            "xgb_params": {"max_depth": 12, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1.0, "gamma": 0.0, "lambda": 1.0, "alpha": 0.0, "tree_method": "hist", "max_bin": 256},
            "n_estimators": 600,
            "split_head": {"enabled": True},
            "sam3_text_quality": {"enabled": True, "alpha": 0.5},
            "split_val_images": ["img1.jpg", "img2.jpg"],
        },
    )
    _write_json(seed_dir / "policy_foo.json", {"sam_only_min_prob_default": 0.15})
    baseline_eval = {
        "tp": 100,
        "fp": 10,
        "fn": 20,
        "precision": 0.91,
        "recall": 0.83,
        "f1": 0.868,
        "metric_tiers": {
            "post_xgb": {"accepted_all": {"precision": 0.91, "recall": 0.83, "f1": 0.868}},
            "post_cluster": {"source_attributed": {"yolo_rfdetr_union": {"f1": 0.80}}},
        },
        "coverage_upper_bound": {"candidate_all": {"recall_upper_bound": 0.95}},
    }
    _write_json(seed_dir / "eval_foo.json", baseline_eval)

    class Result:
        def __init__(self, stdout: str = "") -> None:
            self.stdout = stdout

    commands = []

    def fake_run(cmd, cwd=None, check=None, capture_output=False, text=False):
        commands.append(list(cmd))
        cmd_str = " ".join(str(part) for part in cmd)
        if "tools/tune_ensemble_thresholds_xgb.py" in cmd_str:
            return Result()
        if "tools/train_ensemble_xgb.py" in cmd_str:
            out_idx = cmd.index("--output") + 1
            prefix = Path(cmd[out_idx])
            prefix.with_suffix(".json").write_text("{}", encoding="utf-8")
            prefix.with_suffix(".meta.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            return Result()
        if "tools/eval_ensemble_xgb_dedupe.py" in cmd_str:
            model_arg = cmd[cmd.index("--model") + 1]
            if model_arg.endswith("model_foo.json"):
                payload = {
                    "tp": 101,
                    "fp": 9,
                    "fn": 19,
                    "precision": 0.918,
                    "recall": 0.841,
                    "f1": 0.878,
                    "metric_tiers": {
                        "post_xgb": {"accepted_all": {"precision": 0.918, "recall": 0.841, "f1": 0.878}},
                        "post_cluster": {"source_attributed": {"yolo_rfdetr_union": {"f1": 0.80}}},
                    },
                    "coverage_upper_bound": {"candidate_all": {"recall_upper_bound": 0.95}},
                }
            elif "a0p2" in model_arg:
                payload = {
                    "tp": 100,
                    "fp": 11,
                    "fn": 20,
                    "precision": 0.901,
                    "recall": 0.833,
                    "f1": 0.866,
                    "metric_tiers": {
                        "post_xgb": {"accepted_all": {"precision": 0.901, "recall": 0.833, "f1": 0.866}},
                        "post_cluster": {"source_attributed": {"yolo_rfdetr_union": {"f1": 0.80}}},
                    },
                    "coverage_upper_bound": {"candidate_all": {"recall_upper_bound": 0.95}},
                }
            else:
                payload = {
                    "tp": 102,
                    "fp": 9,
                    "fn": 18,
                    "precision": 0.919,
                    "recall": 0.85,
                    "f1": 0.883,
                    "metric_tiers": {
                        "post_xgb": {"accepted_all": {"precision": 0.919, "recall": 0.85, "f1": 0.883}},
                        "post_cluster": {"source_attributed": {"yolo_rfdetr_union": {"f1": 0.80}}},
                    },
                    "coverage_upper_bound": {"candidate_all": {"recall_upper_bound": 0.95}},
                }
            if "--analysis-json" in cmd:
                Path(cmd[cmd.index("--analysis-json") + 1]).write_text("{}", encoding="utf-8")
            return Result(stdout=json.dumps(payload))
        if "tools/build_calibration_report_bundle.py" in cmd_str:
            Path(cmd[cmd.index("--output-json") + 1]).write_text("{}", encoding="utf-8")
            Path(cmd[cmd.index("--output-md") + 1]).write_text("# report\n", encoding="utf-8")
            return Result()
        raise AssertionError(cmd)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_postrun_nonwindow_policy_confirmation.py",
            "--run-root",
            str(run_root),
            "--similarity-alphas",
            "0.2,0.5",
            "--output-dir",
            str(run_root / "out"),
        ],
    )

    runner.main()

    decision = json.loads((run_root / "out" / "decision_summary.json").read_text())
    assert decision["refined_policy_status"] == "promoted"
    assert decision["similarity_quality_status"] == "promoted"
    assert decision["canonical_recipe"]["winner_lane"] == lane
    assert decision["canonical_recipe"]["scenario"]["sam3_similarity_quality_alpha"] == 0.5
    assert decision["canonical_recipe"]["policy"]["sam_bias_scope"] == "sam_only"
    assert any("tools/build_calibration_report_bundle.py" in " ".join(cmd) for cmd in commands)
