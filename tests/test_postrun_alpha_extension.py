from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tools import run_postrun_alpha_extension as alpha_ext


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fake_eval_payload(f1: float, precision: float, recall: float) -> dict:
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "metric_tiers": {
            "post_cluster": {
                "source_attributed": {
                    "yolo_rfdetr_union": {
                        "f1": 0.75,
                    }
                }
            }
        },
        "coverage_upper_bound": {
            "candidate_all": {
                "recall_upper_bound": 0.9,
            }
        },
    }


def test_winner_contexts_and_summary(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_json(
        run_root / "final_default_recipe.json",
        {
            "winner_lane": "window",
            "lane_settings": {
                "scenario": {
                    "sam_quality": True,
                    "alpha": 0.5,
                }
            },
        },
    )
    _write_json(
        run_root / "lane_manifest.json",
        {
            "lanes": {
                "window": {
                    "variant": "window",
                    "labeled": str(run_root / "lanes" / "window" / "labeled.npz"),
                    "prepass_jsonl": str(run_root / "prepass" / "window.jsonl"),
                }
            },
            "intersection_labeled": {
                "window": {
                    "path": str(run_root / "views" / "window_intersection.labeled.npz"),
                }
            },
            "intersection_prepass_jsonl": {
                "window": str(run_root / "views" / "window_intersection.jsonl"),
            },
        },
    )
    for path in [
        run_root / "lanes" / "window" / "labeled.npz",
        run_root / "prepass" / "window.jsonl",
        run_root / "views" / "window_intersection.labeled.npz",
        run_root / "views" / "window_intersection.jsonl",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    for view, seed, f1 in (("full", 42, 0.84), ("intersection", 1337, 0.83)):
        seed_dir = run_root / "final_matrix" / "window" / view / f"seed_{seed}"
        model_json = seed_dir / "model_demo.json"
        meta_json = seed_dir / "model_demo.meta.json"
        policy_json = seed_dir / "policy_demo.json"
        eval_json = seed_dir / "eval_demo.json"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model_json.write_text("{}", encoding="utf-8")
        _write_json(
            meta_json,
            {
                "sam3_text_quality": {
                    "enabled": True,
                    "model_path": str(seed_dir / "model_demo.sam3_text_quality.json"),
                    "alpha": 0.5,
                },
                "calibration_objective_params": {
                    "optimize": "f1",
                    "target_fp_ratio": 0.2,
                    "min_recall": 0.6,
                    "steps": 300,
                    "eval_iou": 0.5,
                    "dedupe_iou": 0.75,
                    "scoreless_iou": 0.0,
                    "use_val_split": True,
                },
            },
        )
        policy_json.write_text("{}", encoding="utf-8")
        _write_json(eval_json, _fake_eval_payload(f1=f1, precision=0.9, recall=0.8))
        _write_json(seed_dir / "eval_demo.analysis.json", {"overall_metrics": {"f1": f1}})

    contexts = alpha_ext._winner_contexts(run_root, "window")
    assert len(contexts) == 2
    baseline_rows = alpha_ext._baseline_rows(contexts, baseline_alpha=0.5)
    assert {row["view"] for row in baseline_rows} == {"full", "intersection"}
    summary = alpha_ext._summarize(baseline_rows, selection_view="intersection")
    assert summary["winner_alpha"] == 0.5
    assert summary["views"]["intersection"][0]["mean_f1"] == 0.83
    assert all(not str(ctx.baseline_eval_json).endswith(".analysis.json") for ctx in contexts)


def test_run_alpha_eval_reuses_existing_models(tmp_path: Path, monkeypatch) -> None:
    seed_dir = tmp_path / "seed"
    model_json = seed_dir / "model_demo.json"
    meta_json = seed_dir / "model_demo.meta.json"
    policy_json = seed_dir / "policy_demo.json"
    baseline_eval = seed_dir / "eval_demo.json"
    labeled_npz = tmp_path / "labeled.npz"
    prepass_jsonl = tmp_path / "prepass.jsonl"
    for path in [model_json, policy_json, labeled_npz, prepass_jsonl]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    _write_json(
        meta_json,
        {
            "sam3_text_quality": {
                "enabled": True,
                "model_path": str(seed_dir / "model_demo.sam3_text_quality.json"),
                "alpha": 0.5,
            },
            "calibration_objective_params": {
                "optimize": "f1",
                "target_fp_ratio": 0.2,
                "min_recall": 0.6,
                "steps": 300,
                "eval_iou": 0.5,
                "dedupe_iou": 0.75,
                "scoreless_iou": 0.0,
                "use_val_split": True,
            },
        },
    )
    _write_json(baseline_eval, _fake_eval_payload(f1=0.84, precision=0.9, recall=0.8))

    calls = []

    def fake_run(cmd, *, capture=False):
        calls.append(list(cmd))
        if any("eval_ensemble_xgb_dedupe.py" in part for part in cmd):
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=json.dumps(_fake_eval_payload(f1=0.845, precision=0.91, recall=0.79)),
                stderr="",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(alpha_ext, "_run", fake_run)

    ctx = alpha_ext.EvalContext(
        lane="window",
        view="intersection",
        seed=42,
        variant="window",
        model_json=model_json,
        meta_json=meta_json,
        policy_json=policy_json,
        labeled_npz=labeled_npz,
        prepass_jsonl=prepass_jsonl,
        baseline_eval_json=baseline_eval,
    )
    row = alpha_ext._run_alpha_eval(ctx, alpha=0.7, output_root=tmp_path / "alpha")
    assert row["alpha"] == 0.7
    assert row["f1"] == 0.845
    copied_meta = Path(row["meta_json"])
    meta_payload = json.loads(copied_meta.read_text(encoding="utf-8"))
    assert meta_payload["sam3_text_quality"]["alpha"] == 0.7
    assert any("tune_ensemble_thresholds_xgb.py" in " ".join(cmd) for cmd in calls)
    assert any("eval_ensemble_xgb_dedupe.py" in " ".join(cmd) for cmd in calls)
