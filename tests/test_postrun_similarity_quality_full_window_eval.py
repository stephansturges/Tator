from __future__ import annotations

import json
import sys
from pathlib import Path

from tools import run_postrun_similarity_quality_full_window_eval as runner


def test_full_window_similarity_runner_uses_external_eval_surface(
    monkeypatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    lane = "window"
    seed = "42"
    refined_tag = "text_m1p4__sim_m1p2"

    labeled_npz = run_root / "lanes" / lane / "labeled.npz"
    labeled_npz.parent.mkdir(parents=True, exist_ok=True)
    labeled_npz.write_bytes(b"stub")

    prepass_jsonl = run_root / "prepass" / "window.jsonl"
    prepass_jsonl.parent.mkdir(parents=True, exist_ok=True)
    prepass_jsonl.write_text("", encoding="utf-8")

    base_meta = {
        "xgb_params": {
            "max_depth": 8,
            "eta": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "lambda": 1.0,
            "alpha": 0.0,
            "tree_method": "hist",
            "max_bin": 256,
        },
        "n_estimators": 900,
        "split_val_images": ["img1.jpg", "img2.jpg"],
        "split_head": {"enabled": False, "models": {}},
    }
    base_meta_path = (
        run_root / "final_matrix" / lane / "full" / f"seed_{seed}" / "model_dynamic.meta.json"
    )
    base_meta_path.parent.mkdir(parents=True, exist_ok=True)
    base_meta_path.write_text(json.dumps(base_meta), encoding="utf-8")

    baseline_dir = (
        run_root / "postrun_sam_bias_magnitude_sweep" / "full" / refined_tag / "full" / f"seed_{seed}"
    )
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_payload = {
        "tp": 100,
        "fp": 10,
        "fn": 20,
        "precision": 0.9,
        "recall": 0.8,
        "f1": 0.847,
        "metric_tiers": {
            "post_xgb": {"accepted_all": {"precision": 0.9, "recall": 0.8, "f1": 0.847}},
            "post_cluster": {"source_attributed": {"yolo_rfdetr_union": {"f1": 0.75}}},
        },
        "coverage_upper_bound": {"candidate_all": {"recall_upper_bound": 0.95}},
    }
    (baseline_dir / "eval.json").write_text(json.dumps(baseline_payload), encoding="utf-8")
    (baseline_dir / "policy.json").write_text(json.dumps({"sam_bias_scope": "sam_only"}), encoding="utf-8")

    commands = []

    class Result:
        def __init__(self, stdout: str = "") -> None:
            self.stdout = stdout

    def fake_run(cmd, cwd=None, check=None, capture_output=False, text=False):
        commands.append(list(cmd))
        if "tools/train_ensemble_xgb.py" in cmd:
            out_idx = cmd.index("--output") + 1
            prefix = Path(cmd[out_idx])
            prefix.with_suffix(".json").write_text("{}", encoding="utf-8")
            prefix.with_suffix(".meta.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            return Result()
        if "tools/tune_ensemble_thresholds_xgb.py" in cmd:
            return Result()
        if "tools/eval_ensemble_xgb_dedupe.py" in cmd:
            payload = {
                "tp": 101,
                "fp": 9,
                "fn": 19,
                "precision": 0.91,
                "recall": 0.81,
                "f1": 0.857,
                "metric_tiers": {
                    "post_xgb": {"accepted_all": {"precision": 0.91, "recall": 0.81, "f1": 0.857}},
                    "post_cluster": {"source_attributed": {"yolo_rfdetr_union": {"f1": 0.75}}},
                },
                "coverage_upper_bound": {"candidate_all": {"recall_upper_bound": 0.95}},
            }
            return Result(stdout=json.dumps(payload))
        if "tools/build_calibration_report_bundle.py" in cmd:
            Path(cmd[cmd.index("--output-json") + 1]).write_text("{}", encoding="utf-8")
            Path(cmd[cmd.index("--output-md") + 1]).write_text("# report\n", encoding="utf-8")
            return Result()
        raise AssertionError(cmd)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_postrun_similarity_quality_full_window_eval.py",
            "--run-root",
            str(run_root),
            "--refined-tag",
            refined_tag,
            "--seeds",
            seed,
            "--alphas",
            "0.5",
            "--output-dir",
            str(run_root / "out"),
        ],
    )

    runner.main()

    tune_cmd = next(cmd for cmd in commands if "tools/tune_ensemble_thresholds_xgb.py" in cmd)
    eval_cmd = next(cmd for cmd in commands if "tools/eval_ensemble_xgb_dedupe.py" in cmd)
    assert "--use-val-split" in tune_cmd
    assert "--use-val-split" not in eval_cmd

    summary = json.loads((run_root / "out" / "results_summary.json").read_text())
    decision = json.loads((run_root / "out" / "decision_summary.json").read_text())
    assert summary["ranked"][0]["tag"] == "a0p5"
    assert summary["ranked"][0]["mean_delta_f1"] > 0.0
    assert decision["status"] == "promoted"
    assert decision["promoted_alpha"] == 0.5
