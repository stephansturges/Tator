from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

import services.canonical_edr_completion as canonical_completion
from services.calibration_recipe_registry import register_promoted_recipe
from services.canonical_edr_completion import (
    CANONICAL_COMPLETION_CONTEXT_JSON_NAME,
    CANONICAL_COMPLETION_SUMMARY_JSON_NAME,
    get_canonical_deployment_job,
    list_canonical_deployment_jobs,
    materialize_canonical_deployment_bundle,
    persist_canonical_edr_completion,
    repair_persisted_canonical_completion,
    rewrite_canonical_deployment_bundle_metadata,
    _copy2_if_different,
    _write_json_atomic,
)
from tools import run_canonical_prepass_discovery as runner


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_canonical_write_json_atomic_replaces_symlink_targets_without_target_write(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "job" / "summary.json"
    json_path.parent.mkdir()
    outside_tmp = tmp_path / "outside_tmp.json"
    outside_final = tmp_path / "outside_final.json"
    outside_tmp.write_text("external tmp", encoding="utf-8")
    outside_final.write_text("external final", encoding="utf-8")
    tmp_link = json_path.with_suffix(json_path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp_link.symlink_to(outside_tmp)
        json_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _write_json_atomic(json_path, {"status": "ok"})

    assert not tmp_link.exists()
    assert not json_path.is_symlink()
    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "ok"
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    assert outside_final.read_text(encoding="utf-8") == "external final"


def test_canonical_write_json_atomic_rejects_nested_symlinked_parent_before_mkdir(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(ValueError, match="canonical_json_parent_symlink"):
        _write_json_atomic(
            linked_parent / "nested" / "canonical_jobs" / "summary.json",
            {"status": "ok"},
        )

    assert list(outside.iterdir()) == []


def test_canonical_copy_rejects_nested_symlinked_parent_before_mkdir(
    tmp_path: Path,
) -> None:
    src = tmp_path / "source.json"
    src.write_text("source", encoding="utf-8")
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(ValueError, match="canonical_copy_parent_symlink"):
        _copy2_if_different(src, linked_parent / "nested" / "canonical_jobs" / "source.json")

    assert list(outside.iterdir()) == []


def _write_deployment_source_bundle(
    source_dir: Path,
    *,
    f1: float = 0.8,
    fp: float = 100.0,
    precision: float = 0.9,
    recall: float = 0.7,
) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    _write_json(source_dir / "model.json", {"kind": "xgb"})
    _write_json(
        source_dir / "model.meta.json",
        {
            "ensemble_policy": {},
            "sam3_text_quality": {
                "enabled": True,
                "alpha": 0.5,
                "model_path": "model.sam3_text_quality.json",
            },
            "sam3_similarity_quality": {
                "enabled": True,
                "alpha": 0.5,
                "model_path": "model.sam3_similarity_quality.json",
            },
        },
    )
    _write_json(source_dir / "model.sam3_text_quality.json", {"kind": "quality"})
    _write_json(source_dir / "model.sam3_similarity_quality.json", {"kind": "quality"})
    _write_json(
        source_dir / "eval.json",
        {
            "f1": f1,
            "fp": fp,
            "precision": precision,
            "recall": recall,
        },
    )


def _write_materialize_fixture(tmp_path: Path):
    run_root = tmp_path / "calibration_cache" / "discovery_runs" / "run"
    _write_json(
        run_root / "postrun_similarity_quality_full_window_eval" / "decision_summary.json",
        {
            "status": "promoted",
            "winner_tag": "a0p8",
        },
    )
    _write_deployment_source_bundle(
        run_root / "postrun_similarity_quality_full_window_eval" / "a0p8" / "seed_42",
        f1=0.8239,
    )
    canonical_recipe_json = run_root / "canonical_edr.json"
    canonical_recipe_payload = {
        "dataset": "qwen_dataset",
        "discovered_winner_lane": "window",
        "canonical_windowed_recipe": {
            "scenario": {
                "train_sam3_similarity_quality": True,
                "sam3_similarity_quality_alpha": 0.8,
            },
            "policy": {},
            "source_decisions": {
                "similarity_quality": str(
                    (run_root / "postrun_similarity_quality_full_window_eval" / "decision_summary.json").resolve()
                )
            },
        },
    }
    _write_json(canonical_recipe_json, canonical_recipe_payload)
    return run_root, canonical_recipe_json, canonical_recipe_payload


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
    completion = json.loads((run_root / CANONICAL_COMPLETION_SUMMARY_JSON_NAME).read_text())
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
    assert completion["persistence_status"] == "artifact_only_missing_context"
    assert completion["saved_prepass_recipe_id"] is None


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
    completion = json.loads((run_root / CANONICAL_COMPLETION_SUMMARY_JSON_NAME).read_text())
    assert (run_root / "canonical_prepass_recipe.json").exists()
    assert payload["canonical_windowed_recipe"]["winner_lane"] == "window"
    assert "second_stage_policy_layer" not in payload["canonical_windowed_recipe"]
    assert payload["canonical_nonwindowed_recipe"] == {}
    assert progress["stage_key"] == "write_canonical_recipe"
    assert progress["stage_total"] == 6
    assert progress["status"] == "completed"
    assert completion["persistence_status"] == "artifact_only_missing_context"


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


def test_canonical_prepass_discovery_persists_registry_and_saved_recipe_when_context_present(
    monkeypatch, tmp_path: Path
) -> None:
    cache_root = tmp_path / "calibration_cache"
    run_root = cache_root / "discovery_runs" / "8a922d9945b17c16f4ed9dc39f50f5e66b28f614"

    _write_json(
        run_root / "results_ranked.json",
        {
            "views": {
                "intersection": {
                    "ranked_lanes": [
                        {"lane": "window", "mean_f1": 0.82},
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
                        "hp": {"max_depth": 8},
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
        {
            "status": "promoted",
            "winner_metrics": {"mean_f1": 0.841, "mean_delta_f1": 0.001},
            "winner_tag": "a0p8",
            "promoted_config": {
                "train_sam3_similarity_quality": True,
                "sam3_similarity_quality_alpha": 0.8,
            },
        },
    )
    _write_deployment_source_bundle(
        run_root / "postrun_similarity_quality_full_window_eval" / "a0p8" / "seed_42"
    )
    _write_json(
        run_root / CANONICAL_COMPLETION_CONTEXT_JSON_NAME,
        {
            "schema_version": 1,
            "dataset_id": "qwen_dataset",
            "recipe_fingerprint": "8a922d9945b17c16f4ed9dc39f50f5e66b28f614",
            "recipe_fingerprint_payload": {
                "dataset_id": "qwen_dataset",
                "labelmap_hash": "labelhash",
                "glossary_hash": "glossaryhash",
                "classifier_id": "uploads/classifiers/DinoV3_best_model_large.pkl",
                "lane_selection": "window",
                "prepass_config": {"window": {"sam3_text_window_extension": True}},
                "selected_hash": "selhash",
                "selected_count": 9526,
                "selection_seed": 42,
                "requested_max_images": 9526,
                "support_iou": 0.5,
                "context_radius": 0.075,
                "label_iou": 0.5,
                "eval_iou": 0.5,
                "feature_version": 7,
                "recipe_defaults_version": 1,
            },
            "calibration_request": {
                "dataset_id": "qwen_dataset",
                "lane_selection": "window",
                "recipe_mode": "force_rediscover",
                "enable_yolo": True,
                "enable_rfdetr": True,
                "classifier_id": "uploads/classifiers/DinoV3_best_model_large.pkl",
                "threshold_steps": 200,
                "base_fp_ratio": 0.2,
                "relax_fp_ratio": 0.2,
                "support_iou": 0.5,
                "label_iou": 0.5,
                "eval_iou": 0.5,
                "dedupe_iou": 0.75,
            },
            "resolved_classifier_id": "uploads/classifiers/DinoV3_best_model_large.pkl",
            "glossary_text": "{\"person\":[\"person\"]}",
        },
    )
    classifier_path = tmp_path / "uploads" / "classifiers" / "DinoV3_best_model_large.pkl"
    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    classifier_path.write_bytes(b"classifier")

    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(canonical_completion, "_resolve_local_sam3_checkpoint", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_canonical_prepass_discovery.py", "--run-root", str(run_root), "--lane-selection", "window"],
    )

    runner.main()

    completion = json.loads((run_root / CANONICAL_COMPLETION_SUMMARY_JSON_NAME).read_text())
    assert completion["persistence_status"] == "registered_and_saved"
    assert completion["saved_prepass_recipe_id"] == "canonical_edr_qwen_dataset_8a922d9945b1"
    assert completion["canonical_deployment_job_id"] == "canonical_edr_qwen_dataset_8a922d9945b1"
    assert completion["edr_package_id"] == "canonical_edr_pkg_qwen_dataset_8a922d9945b1"
    assert Path(completion["recipe_registry_entry"]["canonical_recipe_json"]).exists()
    assert Path(completion["recipe_registry_entry"]["fingerprint_json"]).exists()
    assert Path(completion["edr_package"]["package_zip"]).exists()
    assert (
        tmp_path
        / "prepass_recipes"
        / "canonical_edr_qwen_dataset_8a922d9945b1"
        / "prepass.meta.json"
    ).exists()
    deployment_dir = (
        tmp_path / "calibration_jobs" / "canonical_edr_qwen_dataset_8a922d9945b1"
    )
    assert (deployment_dir / "ensemble_xgb.json").exists()
    assert (deployment_dir / "ensemble_xgb.meta.json").exists()
    package_dir = tmp_path / "edr_packages" / "canonical_edr_pkg_qwen_dataset_8a922d9945b1"
    assert (package_dir / "package.edr.zip").exists()
    assert (package_dir / "payload" / "edr_manifest.json").exists()


def test_canonical_prepass_discovery_fails_on_invalid_completion_context(
    monkeypatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "calibration_cache" / "discovery_runs" / "badctx"
    _write_json(
        run_root / "results_ranked.json",
        {
            "views": {"intersection": {"ranked_lanes": [{"lane": "window", "mean_f1": 0.82}]}},
            "winner": {"lane": "window"},
        },
    )
    _write_json(
        run_root / "results_raw.json",
        {
            "search_results": {
                "best_stack": {
                    "window": {
                        "hp": {"max_depth": 8},
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
    _write_json(
        run_root / CANONICAL_COMPLETION_CONTEXT_JSON_NAME,
        {
            "schema_version": 1,
            "dataset_id": "qwen_dataset",
            "recipe_fingerprint": "badctx",
            "calibration_request": {},
        },
    )
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_canonical_prepass_discovery.py", "--run-root", str(run_root), "--lane-selection", "window"],
    )

    with pytest.raises(RuntimeError, match="canonical_completion_context_invalid"):
        runner.main()


def test_materialize_canonical_deployment_bundle_uses_best_promoted_seed(tmp_path: Path) -> None:
    run_root = tmp_path / "calibration_cache" / "discovery_runs" / "run"
    _write_json(
        run_root / "postrun_similarity_quality_full_window_eval" / "decision_summary.json",
        {
            "status": "promoted",
            "winner_tag": "a0p8",
        },
    )
    _write_deployment_source_bundle(
        run_root / "postrun_similarity_quality_full_window_eval" / "a0p8" / "seed_42",
        f1=0.8239,
        fp=24000,
        precision=0.8789,
        recall=0.7750,
    )
    _write_deployment_source_bundle(
        run_root / "postrun_similarity_quality_full_window_eval" / "a0p8" / "seed_1337",
        f1=0.8246,
        fp=24345,
        precision=0.8769,
        recall=0.7756,
    )
    _write_deployment_source_bundle(
        run_root / "postrun_similarity_quality_full_window_eval" / "a0p8" / "seed_2025",
        f1=0.8241,
        fp=23903,
        precision=0.8788,
        recall=0.7757,
    )
    canonical_recipe_json = run_root / "canonical_edr.json"
    canonical_recipe_json.parent.mkdir(parents=True, exist_ok=True)
    canonical_recipe_json.write_text(
        json.dumps(
            {
                "dataset": "qwen_dataset",
                "discovered_winner_lane": "window",
                "canonical_windowed_recipe": {
                    "scenario": {
                        "train_sam3_similarity_quality": True,
                        "sam3_similarity_quality_alpha": 0.8,
                    },
                    "policy": {},
                    "source_decisions": {
                        "similarity_quality": str(
                            (run_root / "postrun_similarity_quality_full_window_eval" / "decision_summary.json").resolve()
                        )
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    deployment = materialize_canonical_deployment_bundle(
        calibration_jobs_root=tmp_path / "calibration_jobs",
        run_root=run_root,
        dataset_id="qwen_dataset",
        recipe_fingerprint="8a922d9945b17c16f4ed9dc39f50f5e66b28f614",
        canonical_recipe_payload=json.loads(canonical_recipe_json.read_text(encoding="utf-8")),
        canonical_recipe_json=canonical_recipe_json,
        report_bundle_json=None,
    )

    assert deployment["source_seed"] == 1337
    assert deployment["metrics"]["f1"] == pytest.approx(0.8246)


def test_materialize_canonical_deployment_bundle_rejects_symlinked_jobs_parent_without_target_write(
    tmp_path: Path,
) -> None:
    run_root, canonical_recipe_json, canonical_recipe_payload = _write_materialize_fixture(tmp_path)
    outside = tmp_path / "outside_jobs_parent"
    outside.mkdir()
    jobs_parent = tmp_path / "linked_parent"
    try:
        jobs_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(ValueError, match="canonical_jobs_root_symlink"):
        materialize_canonical_deployment_bundle(
            calibration_jobs_root=jobs_parent / "calibration_jobs",
            run_root=run_root,
            dataset_id="qwen_dataset",
            recipe_fingerprint="8a922d9945b17c16f4ed9dc39f50f5e66b28f614",
            canonical_recipe_payload=canonical_recipe_payload,
            canonical_recipe_json=canonical_recipe_json,
            report_bundle_json=None,
        )

    assert list(outside.iterdir()) == []


def test_materialize_canonical_deployment_bundle_rejects_nested_symlinked_jobs_parent_without_target_write(
    tmp_path: Path,
) -> None:
    run_root, canonical_recipe_json, canonical_recipe_payload = _write_materialize_fixture(tmp_path)
    outside = tmp_path / "outside_jobs_parent"
    outside.mkdir()
    jobs_parent = tmp_path / "linked_parent"
    try:
        jobs_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(ValueError, match="canonical_jobs_root_symlink"):
        materialize_canonical_deployment_bundle(
            calibration_jobs_root=jobs_parent / "nested" / "calibration_jobs",
            run_root=run_root,
            dataset_id="qwen_dataset",
            recipe_fingerprint="8a922d9945b17c16f4ed9dc39f50f5e66b28f614",
            canonical_recipe_payload=canonical_recipe_payload,
            canonical_recipe_json=canonical_recipe_json,
            report_bundle_json=None,
        )

    assert list(outside.iterdir()) == []


def test_materialize_canonical_deployment_bundle_replaces_symlinked_final_dir_without_target_write(
    tmp_path: Path,
) -> None:
    run_root, canonical_recipe_json, canonical_recipe_payload = _write_materialize_fixture(tmp_path)
    jobs_root = tmp_path / "calibration_jobs"
    jobs_root.mkdir()
    outside = tmp_path / "outside_final"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    job_id = canonical_completion.canonical_deployment_job_id(
        "qwen_dataset",
        "8a922d9945b17c16f4ed9dc39f50f5e66b28f614",
    )
    try:
        (jobs_root / job_id).symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    deployment = materialize_canonical_deployment_bundle(
        calibration_jobs_root=jobs_root,
        run_root=run_root,
        dataset_id="qwen_dataset",
        recipe_fingerprint="8a922d9945b17c16f4ed9dc39f50f5e66b28f614",
        canonical_recipe_payload=canonical_recipe_payload,
        canonical_recipe_json=canonical_recipe_json,
        report_bundle_json=None,
    )

    final_dir = jobs_root / job_id
    assert deployment["job_id"] == job_id
    assert not final_dir.is_symlink()
    assert (final_dir / "ensemble_xgb.json").exists()
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_rewrite_canonical_deployment_bundle_metadata_rewrites_local_paths(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "calibration_jobs" / "canonical_edr_example"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        bundle_dir / "ensemble_xgb.meta.json",
        {
            "model_path": "ensemble_xgb.json",
            "canonical_recipe_json": "/old/machine/canonical_edr.json",
            "canonical_deployment_source_dir": "/old/machine/source_dir",
        },
    )
    _write_json(
        bundle_dir / "canonical_deployment.json",
        {
            "job_id": "old_job",
            "dataset_id": "old_dataset",
            "source_dir": "/old/machine/source_dir",
            "source_stage": "postrun_similarity_quality_full_window_eval",
            "source_seed": 42,
            "canonical_recipe_json": "/old/machine/canonical_edr.json",
        },
    )
    canonical_json = tmp_path / "recipes" / "canonical_edr.json"
    canonical_md = tmp_path / "recipes" / "canonical_edr.md"
    report_json = tmp_path / "recipes" / "report_bundle.json"
    for path in (canonical_json, canonical_md, report_json):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    rewrite_canonical_deployment_bundle_metadata(
        bundle_dir,
        job_id="canonical_edr_example",
        dataset_id="qwen_dataset",
        canonical_recipe_json=canonical_json,
        canonical_recipe_md=canonical_md,
        canonical_report_bundle_json=report_json,
        source_stage="imported_bundle",
        source_seed=1337,
        source_dir=bundle_dir,
    )

    meta_payload = json.loads((bundle_dir / "ensemble_xgb.meta.json").read_text(encoding="utf-8"))
    deployment_payload = json.loads((bundle_dir / "canonical_deployment.json").read_text(encoding="utf-8"))
    assert meta_payload["canonical_recipe_json"] == str(canonical_json.resolve())
    assert meta_payload["original_canonical_recipe_json"] == "/old/machine/canonical_edr.json"
    assert meta_payload["canonical_deployment_job_id"] == "canonical_edr_example"
    assert deployment_payload["job_id"] == "canonical_edr_example"
    assert deployment_payload["job_dir"] == str(bundle_dir.resolve())
    assert deployment_payload["source_dir"] == str(bundle_dir.resolve())
    assert deployment_payload["original_source_dir"] == "/old/machine/source_dir"
    assert deployment_payload["canonical_recipe_json"] == str(canonical_json.resolve())


def test_list_canonical_deployment_jobs_exposes_persistent_bundle_entries(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "calibration_jobs" / "canonical_edr_qwen_dataset_8a922d9945b1"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _write_json(bundle_dir / "ensemble_xgb.json", {"kind": "xgb"})
    _write_json(bundle_dir / "ensemble_xgb.meta.json", {"model_path": "ensemble_xgb.json"})
    _write_json(
        bundle_dir / "canonical_deployment.json",
        {
            "job_id": "canonical_edr_qwen_dataset_8a922d9945b1",
            "dataset_id": "qwen_dataset",
            "winner_lane": "window",
            "source_stage": "postrun_similarity_quality_full_window_eval",
            "source_seed": 1337,
            "source_dir": str(bundle_dir.resolve()),
            "canonical_recipe_json": str((tmp_path / "canonical_edr.json").resolve()),
        },
    )

    jobs = list_canonical_deployment_jobs(tmp_path / "calibration_jobs")
    assert len(jobs) == 1
    assert jobs[0]["job_kind"] == "canonical_bundle"
    assert jobs[0]["persistent_bundle"] is True
    assert jobs[0]["result"]["canonical_deployment_job_id"] == "canonical_edr_qwen_dataset_8a922d9945b1"
    assert get_canonical_deployment_job(
        tmp_path / "calibration_jobs",
        "canonical_edr_qwen_dataset_8a922d9945b1",
    )["request"]["dataset_id"] == "qwen_dataset"


def test_repair_persisted_canonical_completion_backfills_existing_bundle(
    monkeypatch, tmp_path: Path
) -> None:
    cache_root = tmp_path / "calibration_cache"
    run_root = cache_root / "discovery_runs" / "8a922d9945b17c16f4ed9dc39f50f5e66b28f614"
    fingerprint_json = (
        cache_root / "recipe_registry" / "8a922d9945b17c16f4ed9dc39f50f5e66b28f614" / "fingerprint.json"
    )
    _write_json(
        run_root / "canonical_edr.json",
        {
            "dataset": "qwen_dataset",
            "discovered_winner_lane": "window",
            "canonical_windowed_recipe": {
                "scenario": {
                    "train_sam3_similarity_quality": True,
                    "sam3_similarity_quality_alpha": 0.8,
                },
                "policy": {},
                "source_decisions": {
                    "similarity_quality": str(
                        (run_root / "postrun_similarity_quality_full_window_eval" / "decision_summary.json").resolve()
                    )
                },
            },
        },
    )
    _write_json(run_root / "canonical_edr.md", {"md": True})
    _write_json(run_root / "report_bundle.json", {"report": True})
    _write_json(
        run_root / "postrun_similarity_quality_full_window_eval" / "decision_summary.json",
        {"status": "promoted", "winner_tag": "a0p8"},
    )
    _write_deployment_source_bundle(
        run_root / "postrun_similarity_quality_full_window_eval" / "a0p8" / "seed_42",
        f1=0.8239,
    )
    _write_deployment_source_bundle(
        run_root / "postrun_similarity_quality_full_window_eval" / "a0p8" / "seed_1337",
        f1=0.8246,
    )
    _write_json(
        fingerprint_json,
        {
            "dataset_id": "qwen_dataset",
            "classifier_id": "uploads/classifiers/DinoV3_best_model_large.pkl",
            "lane_selection": "window",
            "requested_max_images": 9526,
            "support_iou": 0.5,
            "label_iou": 0.5,
            "eval_iou": 0.5,
            "prepass_config": {
                "window": {
                    "enable_yolo": True,
                    "enable_rfdetr": True,
                    "dedupe_iou": 0.75,
                }
            },
        },
    )
    _write_json(
        run_root / CANONICAL_COMPLETION_SUMMARY_JSON_NAME,
        {
            "recipe_registry_entry": {
                "fingerprint": "8a922d9945b17c16f4ed9dc39f50f5e66b28f614",
                "dataset_id": "qwen_dataset",
                "lane_selection": "window",
                "requested_max_images": 9526,
                "classifier_id": "uploads/classifiers/DinoV3_best_model_large.pkl",
                "fingerprint_json": str(fingerprint_json.resolve()),
                "report_bundle_json": str((run_root / "report_bundle.json").resolve()),
            },
            "saved_prepass_recipe": {
                "id": "canonical_edr_qwen_dataset_8a922d9945b1",
                "config": {
                    "dataset_id": "qwen_dataset",
                    "lane_selection": "window",
                    "resolved_classifier_id": "uploads/classifiers/DinoV3_best_model_large.pkl",
                    "recipe_registry_fingerprint": "8a922d9945b17c16f4ed9dc39f50f5e66b28f614",
                },
                "glossary": "{\"person\":[\"person\"]}",
            },
        },
    )
    classifier_path = tmp_path / "uploads" / "classifiers" / "DinoV3_best_model_large.pkl"
    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    classifier_path.write_bytes(b"classifier")
    monkeypatch.setattr(canonical_completion, "_resolve_local_sam3_checkpoint", lambda: None)

    summary = repair_persisted_canonical_completion(
        calibration_cache_root=cache_root,
        run_root=run_root,
    )

    assert summary["canonical_deployment_job"]["source_seed"] == 1337
    assert summary["edr_package_id"] == "canonical_edr_pkg_qwen_dataset_8a922d9945b1"
    assert (run_root / CANONICAL_COMPLETION_CONTEXT_JSON_NAME).exists()
    assert (tmp_path / "edr_packages" / "canonical_edr_pkg_qwen_dataset_8a922d9945b1" / "package.edr.zip").exists()
    saved_meta = json.loads(
        (
            tmp_path
            / "prepass_recipes"
            / "canonical_edr_qwen_dataset_8a922d9945b1"
            / "prepass.meta.json"
        ).read_text(encoding="utf-8")
    )
    assert saved_meta["config"]["canonical_deployment_job_id"] == "canonical_edr_qwen_dataset_8a922d9945b1"


def test_persist_canonical_completion_reuses_existing_imported_canonical_without_snapshot_drift(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "calibration_cache"
    recipes_root = tmp_path / "prepass_recipes"
    calibration_jobs_root = tmp_path / "calibration_jobs"
    canonical_recipe_json = cache_root / "recipe_registry" / "fp123" / "canonical_edr.json"
    canonical_recipe_json.parent.mkdir(parents=True, exist_ok=True)
    canonical_recipe_json.write_text(
        json.dumps(
            {
                "dataset": "qwen_dataset",
                "lane_selection": "window",
                "canonical_windowed_recipe": {
                    "scenario": {},
                    "policy": {},
                },
            }
        ),
        encoding="utf-8",
    )
    canonical_recipe_md = canonical_recipe_json.with_name("canonical_edr.md")
    canonical_recipe_md.write_text("# canonical", encoding="utf-8")

    bundle_dir = calibration_jobs_root / "canonical_edr_qwen_dataset_fp123"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _write_json(bundle_dir / "ensemble_xgb.json", {"kind": "xgb"})
    _write_json(bundle_dir / "ensemble_xgb.meta.json", {"model_path": "ensemble_xgb.json"})
    _write_json(
        bundle_dir / "canonical_deployment.json",
        {
            "job_id": "canonical_edr_qwen_dataset_fp123",
            "dataset_id": "qwen_dataset",
            "winner_lane": "window",
            "source_stage": "imported_bundle",
            "source_seed": 1337,
            "source_dir": str(bundle_dir.resolve()),
            "canonical_recipe_json": str(canonical_recipe_json.resolve()),
        },
    )

    saved_recipe_dir = recipes_root / "canonical_edr_qwen_dataset_fp123"
    saved_recipe_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        saved_recipe_dir / "prepass.meta.json",
        {
            "id": "canonical_edr_qwen_dataset_fp123",
            "schema_version": 1,
            "name": "Canonical EDR",
            "description": "",
            "config": {
                "recipe_kind": "canonical_edr",
                "dataset_id": "qwen_dataset",
                "recipe_fingerprint": "fp123",
                "recipe_registry_fingerprint": "fp123",
                "base_fp_ratio": 0.2,
                "threshold_steps": 200,
                "canonical_deployment_job_id": "canonical_edr_qwen_dataset_fp123",
                "canonical_deployment_job_dir": str(bundle_dir.resolve()),
                "canonical_edr_json": str(canonical_recipe_json.resolve()),
            },
            "glossary": "",
            "created_at": 1.0,
            "updated_at": 1.0,
        },
    )

    registry_entry = register_promoted_recipe(
        cache_root,
        fingerprint="fp123",
        fingerprint_payload={
            "dataset_id": "qwen_dataset",
            "labelmap_hash": "labelhash",
            "glossary_hash": "glossaryhash",
            "classifier_id": "clf.pkl",
            "lane_selection": "window",
            "prepass_config": {"window": {"enable_yolo": True}},
            "selected_hash": "selhash",
            "selected_count": 100,
            "selection_seed": 42,
            "requested_max_images": 100,
            "support_iou": 0.5,
            "context_radius": 0.075,
            "label_iou": 0.5,
            "eval_iou": 0.5,
            "feature_version": 7,
        },
        dataset_id="qwen_dataset",
        canonical_recipe_json=canonical_recipe_json,
        canonical_recipe_md=canonical_recipe_md,
        report_bundle_json=None,
        discovery_run_root=None,
        origin_kind="imported_portable",
        canonical_deployment={
            "job_id": "canonical_edr_qwen_dataset_fp123",
            "job_dir": str(bundle_dir.resolve()),
            "source_stage": "imported_bundle",
            "source_seed": 1337,
            "source_dir": str(bundle_dir.resolve()),
        },
    )
    classifier_path = tmp_path / "uploads" / "classifiers" / "clf.pkl"
    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    classifier_path.write_bytes(b"classifier")
    monkeypatch.setattr(canonical_completion, "_resolve_local_sam3_checkpoint", lambda: None)

    summary = persist_canonical_edr_completion(
        calibration_cache_root=cache_root,
        run_root=None,
        canonical_recipe_json=canonical_recipe_json,
        canonical_recipe_md=canonical_recipe_md,
        canonical_recipe_payload=json.loads(canonical_recipe_json.read_text(encoding="utf-8")),
        completion_context={
            "dataset_id": "qwen_dataset",
            "recipe_fingerprint": "fp123",
            "recipe_fingerprint_payload": {"dataset_id": "qwen_dataset"},
            "calibration_request": {
                "dataset_id": "qwen_dataset",
                "base_fp_ratio": 0.9,
                "threshold_steps": 999,
            },
            "resolved_classifier_id": "clf.pkl",
            "glossary_text": "",
        },
        existing_registry_entry=registry_entry,
        report_bundle_json=None,
        write_summary=False,
    )

    assert summary["persistence_status"] == "reused_existing"
    assert summary["canonical_deployment_job"]["job_id"] == "canonical_edr_qwen_dataset_fp123"
    assert summary["saved_prepass_recipe"]["config"]["base_fp_ratio"] == 0.2
    assert summary["saved_prepass_recipe"]["config"]["threshold_steps"] == 200
    assert summary["saved_prepass_recipe"]["config"]["edr_package_id"] == "canonical_edr_pkg_qwen_dataset_fp123"
    assert summary["recipe_registry_entry"]["edr_package_id"] == "canonical_edr_pkg_qwen_dataset_fp123"
    assert (tmp_path / "edr_packages" / "canonical_edr_pkg_qwen_dataset_fp123" / "package.edr.zip").exists()
