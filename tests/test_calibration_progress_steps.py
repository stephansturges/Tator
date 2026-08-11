import json
from pathlib import Path

from services.calibration import (
    CalibrationJob,
    _build_calibration_step_plan,
    _normalize_classifier_id_for_fingerprint,
    _serialize_calibration_job,
)


def test_build_calibration_step_plan_includes_recipe_discovery_and_policy_for_xgb_auto() -> None:
    plan = _build_calibration_step_plan(
        recipe_mode="auto",
        calibration_model="xgb",
        policy_layer_variant="bakeoff",
    )

    phases = [step["phase"] for step in plan]

    assert phases[:4] == ["select_images", "fingerprint", "recipe_lookup", "recipe_discovery"]
    assert "prepass" in phases
    assert "features" in phases
    assert "labeling" in phases
    assert "train" in phases
    assert "relax" in phases
    assert "objective" in phases
    assert "policy" in phases
    assert phases[-2:] == ["eval", "report"]


def test_serialize_calibration_job_includes_step_fields() -> None:
    job = CalibrationJob(job_id="cal-steps")
    job.step_current = 4
    job.step_total = 12
    job.step_label = "Discover canonical EDR"
    job.substep_current = 2
    job.substep_total = 7
    job.substep_label = "SAM-bias magnitude sweep"

    payload = _serialize_calibration_job(job)

    assert payload["step_current"] == 4
    assert payload["step_total"] == 12
    assert payload["step_label"] == "Discover canonical EDR"
    assert payload["substep_current"] == 2
    assert payload["substep_total"] == 7
    assert payload["substep_label"] == "SAM-bias magnitude sweep"


def test_serialize_calibration_job_prefers_reference_iou_metrics(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "tp": 1813,
                "fp": 28,
                "fn": 437,
                "precision": 0.9848,
                "recall": 0.8058,
                "f1": 0.8863,
                "reference_iou": {
                    "dedupe_iou": 0.75,
                    "eval_iou": 0.5,
                    "xgb_ensemble": {
                        "tp": 1814,
                        "fp": 152,
                        "fn": 436,
                        "precision": 0.9227,
                        "recall": 0.8062,
                        "f1": 0.8605,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    job = CalibrationJob(job_id="cal-metrics")
    job.result = {"eval": str(eval_path)}

    payload = _serialize_calibration_job(job)
    metrics = payload["result"]["metrics"]

    assert metrics["tp"] == 1814
    assert metrics["fp"] == 152
    assert metrics["fn"] == 436
    assert metrics["precision"] == 0.9227
    assert metrics["recall"] == 0.8062
    assert metrics["f1"] == 0.8605
    assert metrics["reported_operating_point"] == {
        "kind": "reference_iou",
        "dedupe_iou": 0.75,
        "eval_iou": 0.5,
    }
    assert metrics["best_sweep_metrics"] == {
        "tp": 1813,
        "fp": 28,
        "fn": 437,
        "precision": 0.9848,
        "recall": 0.8058,
        "f1": 0.8863,
    }


def test_normalize_classifier_id_for_fingerprint_resolves_classifier_root_paths(tmp_path: Path) -> None:
    classifier_root = tmp_path / "uploads" / "classifiers"
    classifier_root.mkdir(parents=True, exist_ok=True)
    classifier_path = classifier_root / "demo.pkl"
    classifier_path.write_text("model", encoding="utf-8")

    normalized_basename = _normalize_classifier_id_for_fingerprint("demo.pkl", root_dir=tmp_path)
    normalized_path = _normalize_classifier_id_for_fingerprint(str(classifier_path), root_dir=tmp_path)

    assert normalized_basename == str(classifier_path.resolve())
    assert normalized_path == str(classifier_path.resolve())
