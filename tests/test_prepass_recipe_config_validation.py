import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from services.prepass_recipes import (
    _collect_recipe_assets_impl,
    _delete_agent_recipe_impl,
    _delete_prepass_recipe_impl,
    _ensure_recipe_zip_impl,
    _export_prepass_recipe_impl,
    _import_prepass_recipe_from_zip_impl,
    _list_agent_recipes_impl,
    _list_prepass_recipes_impl,
    _load_prepass_recipe_meta,
    _persist_agent_recipe_impl,
    _save_prepass_recipe_impl,
    _validate_prepass_recipe_config_impl,
    _write_prepass_recipe_meta,
    upsert_canonical_edr_saved_recipe_impl,
)


def _within_root(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def test_prepass_recipe_config_rejects_legacy_cross_iou():
    with pytest.raises(HTTPException) as exc_info:
        _validate_prepass_recipe_config_impl({"cross_iou": 0.8, "enable_yolo": True})
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "prepass_recipe_legacy_cross_iou_unsupported"


def test_prepass_recipe_config_accepts_current_cross_class_dedupe_fields():
    config = {
        "cross_class_dedupe_enabled": True,
        "cross_class_dedupe_iou": 0.8,
        "enable_yolo": True,
    }
    normalized = _validate_prepass_recipe_config_impl(config)
    assert normalized == config
    assert normalized is not config


def test_prepass_recipe_config_rejects_non_dict_payload():
    with pytest.raises(HTTPException) as exc_info:
        _validate_prepass_recipe_config_impl(["bad", "config"])
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "prepass_recipe_invalid_config"
    with pytest.raises(HTTPException) as exc_info_2:
        _validate_prepass_recipe_config_impl([])
    assert exc_info_2.value.status_code == 400
    assert exc_info_2.value.detail == "prepass_recipe_invalid_config"


def test_prepass_recipe_config_allows_missing_config_as_empty():
    assert _validate_prepass_recipe_config_impl(None) == {}


def test_save_prepass_recipe_preserves_created_at(tmp_path):
    payload = {
        "name": "Recipe A",
        "description": "first",
        "config": {"enable_yolo": True},
        "glossary": "",
    }
    first = _save_prepass_recipe_impl(
        payload,
        recipe_id="recipe_a",
        prepass_schema_version=1,
        recipes_root=tmp_path,
        sanitize_run_id_fn=lambda value: value,
        normalize_glossary_fn=lambda value: value or "",
        write_meta_fn=_write_prepass_recipe_meta,
    )
    updated = _save_prepass_recipe_impl(
        {**payload, "description": "second"},
        recipe_id="recipe_a",
        prepass_schema_version=1,
        recipes_root=tmp_path,
        sanitize_run_id_fn=lambda value: value,
        normalize_glossary_fn=lambda value: value or "",
        write_meta_fn=_write_prepass_recipe_meta,
    )
    assert updated["created_at"] == first["created_at"]
    assert updated["updated_at"] >= first["updated_at"]


def test_save_prepass_recipe_rejects_invalid_config_before_directory_create(tmp_path):
    with pytest.raises(HTTPException) as exc_info:
        _save_prepass_recipe_impl(
            {
                "name": "Invalid",
                "description": "",
                "config": {"cross_iou": 0.8, "enable_yolo": True},
                "glossary": "",
            },
            recipe_id="invalid_recipe",
            prepass_schema_version=1,
            recipes_root=tmp_path,
            sanitize_run_id_fn=lambda value: value,
            normalize_glossary_fn=lambda value: value or "",
            write_meta_fn=_write_prepass_recipe_meta,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "prepass_recipe_legacy_cross_iou_unsupported"
    assert not (tmp_path / "invalid_recipe").exists()


def test_persist_agent_recipe_preserves_numpy_clip_head_classes(tmp_path):
    saved = _persist_agent_recipe_impl(
        dataset_id=None,
        class_id=None,
        class_name="person",
        label="person recipe",
        recipe={
            "text_prompts": ["person"],
            "_clip_head_classifier_path": "classifier.pkl",
        },
        meta_overrides={
            "dataset_signature": "dataset-signature",
            "labelmap_hash": "labelmap-hash",
            "labelmap": ["person", "vehicle"],
        },
        recipes_root=tmp_path,
        max_clip_head_bytes=1024 * 1024,
        max_crops=10,
        max_crop_bytes=1024 * 1024,
        resolve_dataset_fn=lambda _dataset_id: tmp_path,
        load_coco_index_fn=lambda _root: ({"categories": []}, {}, {}),
        compute_dataset_signature_fn=lambda *_args: "dataset-signature",
        compute_labelmap_hash_fn=lambda _categories: ("labelmap-hash", ["person", "vehicle"]),
        resolve_clip_classifier_fn=lambda _path: tmp_path / "classifier.pkl",
        load_clip_head_fn=lambda _path: {
            "classes": np.asarray(["person", "vehicle"], dtype=object),
            "clip_model": "ViT-B/32",
            "proba_mode": "softmax",
        },
        save_clip_head_artifacts_fn=lambda **_kwargs: None,
        load_clip_head_artifacts_fn=lambda **_kwargs: {
            "classes": np.asarray(["person", "vehicle"], dtype=object),
            "clip_model": "ViT-B/32",
            "proba_mode": "softmax",
            "min_prob": 0.6,
            "margin": 0.1,
        },
        save_exemplar_crop_fn=lambda **_kwargs: None,
        sanitize_prompts_fn=lambda prompts: prompts,
        path_is_within_root_fn=lambda path, root: Path(path).resolve().is_relative_to(Path(root).resolve()),
    )

    assert saved["recipe"]["clip_head"]["classes"] == ["person", "vehicle"]
    assert saved["recipe"]["clip_head"]["min_prob"] == 0.6


def test_load_prepass_recipe_meta_supports_legacy_recipe_json(tmp_path):
    recipe_dir = tmp_path / "legacy_recipe"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    (recipe_dir / "recipe.json").write_text(
        '{"id":"legacy_recipe","name":"Legacy","description":"x","created_at":1.0,"updated_at":2.0}',
        encoding="utf-8",
    )
    loaded = _load_prepass_recipe_meta(recipe_dir)
    assert loaded.get("id") == "legacy_recipe"
    assert loaded.get("name") == "Legacy"


def test_list_prepass_recipes_supports_legacy_recipe_json(tmp_path):
    recipe_dir = tmp_path / "legacy_recipe"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    (recipe_dir / "recipe.json").write_text(
        '{"id":"legacy_recipe","name":"Legacy","description":"x","created_at":1.0,"updated_at":2.0}',
        encoding="utf-8",
    )
    listed = _list_prepass_recipes_impl(recipes_root=tmp_path, meta_filename="prepass.meta.json")
    assert listed
    assert listed[0]["id"] == "legacy_recipe"
    assert listed[0]["name"] == "Legacy"


def test_list_prepass_recipes_handles_mixed_timestamp_types(tmp_path):
    first = tmp_path / "first"
    first.mkdir(parents=True, exist_ok=True)
    (first / "prepass.meta.json").write_text(
        '{"id":"first","name":"First","description":"","created_at":1.0,"updated_at":1.0}',
        encoding="utf-8",
    )

    second = tmp_path / "second"
    second.mkdir(parents=True, exist_ok=True)
    (second / "prepass.meta.json").write_text(
        '{"id":"second","name":"Second","description":"","created_at":"2026-02-09T00:00:00Z","updated_at":"2026-02-09T01:00:00Z"}',
        encoding="utf-8",
    )

    third = tmp_path / "third"
    third.mkdir(parents=True, exist_ok=True)
    (third / "prepass.meta.json").write_text(
        '{"id":"third","name":"Third","description":"","created_at":"bad","updated_at":"bad"}',
        encoding="utf-8",
    )

    listed = _list_prepass_recipes_impl(recipes_root=tmp_path, meta_filename="prepass.meta.json")
    assert [item["id"] for item in listed] == ["second", "first", "third"]


def test_upsert_canonical_edr_saved_recipe_persists_ui_compatible_snapshot(tmp_path):
    canonical_json = tmp_path / "canonical_edr.json"
    canonical_json.write_text(
        json.dumps(
            {
                "dataset": "qwen_dataset",
                "lane_selection": "window",
                "discovered_winner_lane": "window",
                "canonical_windowed_recipe": {
                    "scenario": {
                        "split_head": False,
                        "train_sam3_text_quality": True,
                        "sam3_text_quality_alpha": 0.8,
                        "train_sam3_similarity_quality": True,
                        "sam3_similarity_quality_alpha": 0.8,
                    },
                    "policy": {
                        "sam_bias_scope": "sam_only",
                        "sam_only_min_prob_default": 0.15,
                        "consensus_iou_default": 0.7,
                        "consensus_iou_by_source_class": {"sam3_text": {"__default__": 0.7}},
                        "threshold_by_class_override": {"person": 0.9},
                        "logit_bias_by_source_class": {
                            "sam3_text": {"__default__": -1.4},
                            "sam3_similarity": {"__default__": -1.2},
                        },
                    },
                    "expected_metrics": {"full_mean_f1": 0.8242},
                },
            }
        ),
        encoding="utf-8",
    )
    report_bundle = tmp_path / "report_bundle.json"
    report_bundle.write_text("{}", encoding="utf-8")

    saved = upsert_canonical_edr_saved_recipe_impl(
        recipes_root=tmp_path / "prepass_recipes",
        dataset_id="qwen_dataset",
        calibration_request={
            "enable_yolo": True,
            "enable_rfdetr": True,
            "classifier_id": "uploads/classifiers/DinoV3_best_model_large.pkl",
            "recipe_mode": "force_rediscover",
            "lane_selection": "window",
            "base_fp_ratio": 0.2,
            "relax_fp_ratio": 0.2,
            "threshold_steps": 200,
            "support_iou": 0.5,
            "label_iou": 0.5,
            "eval_iou": 0.5,
            "eval_iou_grid": "0.5,0.75",
            "dedupe_iou": 0.75,
            "dedupe_iou_grid": "0.5,0.75",
            "max_images": 9526,
        },
        classifier_id="uploads/classifiers/DinoV3_best_model_large.pkl",
        recipe_fingerprint="8a922d9945b17c16f4ed9dc39f50f5e66b28f614",
        canonical_recipe=json.loads(canonical_json.read_text(encoding="utf-8")),
        canonical_recipe_json=canonical_json,
        canonical_recipe_md=None,
        report_bundle_json=report_bundle,
        recipe_registry_entry={"fingerprint": "8a922d9945b17c16f4ed9dc39f50f5e66b28f614", "recipe_root": "/tmp/registry/8a922"},
        glossary_text='{"person":["person"]}',
        prepass_schema_version=1,
        canonical_deployment={
            "job_id": "canonical_edr_qwen_dataset_8a922d9945b1",
            "job_dir": "/tmp/calibration_jobs/canonical_edr_qwen_dataset_8a922d9945b1",
            "source_stage": "postrun_similarity_quality_full_window_eval",
            "source_seed": 42,
        },
        edr_package={
            "id": "canonical_edr_pkg_qwen_dataset_8a922d9945b1",
            "package_root": "/tmp/edr_packages/canonical_edr_pkg_qwen_dataset_8a922d9945b1",
            "package_zip": "/tmp/edr_packages/canonical_edr_pkg_qwen_dataset_8a922d9945b1/package.edr.zip",
            "package_sha256": "sha256",
        },
    )

    assert saved["id"] == "canonical_edr_qwen_dataset_8a922d9945b1"
    meta = _load_prepass_recipe_meta((tmp_path / "prepass_recipes") / saved["id"])
    assert meta["config"]["recipe_mode"] == "reuse_only"
    assert meta["config"]["lane_selection"] == "window"
    assert meta["config"]["ensemble_job_id"] == "canonical_edr_qwen_dataset_8a922d9945b1"
    assert meta["config"]["apply_default_ensemble_policy"] is False
    assert meta["config"]["calibration_max_images"] == 9526
    assert meta["config"]["eval_iou_grid"] == "0.5,0.75"
    assert meta["config"]["dedupe_iou_grid"] == "0.5,0.75"
    assert meta["config"]["train_sam3_text_quality"] is True
    assert meta["config"]["train_sam3_similarity_quality"] is True
    assert meta["config"]["sam3_text_quality_alpha"] == 0.8
    assert meta["config"]["sam3_similarity_quality_alpha"] == 0.8
    assert meta["config"]["sam_bias_scope"] == "sam_only"
    assert meta["config"]["sam3_text_bias_default"] == -1.4
    assert meta["config"]["sam3_similarity_bias_default"] == -1.2
    assert meta["config"]["canonical_deployment_job_id"] == "canonical_edr_qwen_dataset_8a922d9945b1"
    assert meta["config"]["edr_package_id"] == "canonical_edr_pkg_qwen_dataset_8a922d9945b1"
    assert meta["config"]["edr_runtime_mode"] == "package"
    assert meta["config"]["prepass_caption"] is False
    assert meta["config"]["sam3_text_synonym_budget"] == 0
    assert meta["config"]["canonical_edr_json"] == str(canonical_json.resolve())
    assert meta["config"]["recipe_registry_fingerprint"] == "8a922d9945b17c16f4ed9dc39f50f5e66b28f614"
    assert meta["glossary"] == '{"person":["person"]}'


def test_list_prepass_recipes_exposes_canonical_edr_metadata(tmp_path):
    canonical_json = tmp_path / "canonical_edr.json"
    canonical_json.write_text(
        '{"dataset":"qwen_dataset","lane_selection":"window","canonical_windowed_recipe":{"expected_metrics":{"full_mean_f1":0.8242}}}',
        encoding="utf-8",
    )
    saved = upsert_canonical_edr_saved_recipe_impl(
        recipes_root=tmp_path / "prepass_recipes",
        dataset_id="qwen_dataset",
        calibration_request={
            "lane_selection": "window",
            "classifier_id": "uploads/classifiers/DinoV3_best_model_large.pkl",
        },
        classifier_id="uploads/classifiers/DinoV3_best_model_large.pkl",
        recipe_fingerprint="8a922d9945b17c16f4ed9dc39f50f5e66b28f614",
        canonical_recipe={
            "dataset": "qwen_dataset",
            "lane_selection": "window",
            "canonical_windowed_recipe": {"expected_metrics": {"full_mean_f1": 0.8242}},
        },
        canonical_recipe_json=canonical_json,
        canonical_recipe_md=None,
        report_bundle_json=None,
        recipe_registry_entry=None,
        glossary_text="",
        prepass_schema_version=1,
    )

    listed = _list_prepass_recipes_impl(
        recipes_root=tmp_path / "prepass_recipes",
        meta_filename="prepass.meta.json",
    )

    assert listed[0]["id"] == saved["id"]
    assert listed[0]["recipe_kind"] == "canonical_edr"
    assert listed[0]["edr_saved_source"] == "canonical_discovery"
    assert listed[0]["dataset_id"] == "qwen_dataset"
    assert listed[0]["lane_selection"] == "window"
    assert listed[0]["recipe_fingerprint"] == "8a922d9945b17c16f4ed9dc39f50f5e66b28f614"
    assert listed[0]["expected_mean_f1"] == 0.8242


def test_delete_prepass_recipe_rejects_canonical_edr(tmp_path):
    canonical_json = tmp_path / "canonical_edr.json"
    canonical_json.write_text(
        '{"dataset":"qwen_dataset","lane_selection":"window","canonical_windowed_recipe":{"expected_metrics":{"full_mean_f1":0.8242}}}',
        encoding="utf-8",
    )
    saved = upsert_canonical_edr_saved_recipe_impl(
        recipes_root=tmp_path / "prepass_recipes",
        dataset_id="qwen_dataset",
        calibration_request={"lane_selection": "window"},
        classifier_id=None,
        recipe_fingerprint="8a922d9945b17c16f4ed9dc39f50f5e66b28f614",
        canonical_recipe={
            "dataset": "qwen_dataset",
            "lane_selection": "window",
            "canonical_windowed_recipe": {"expected_metrics": {"full_mean_f1": 0.8242}},
        },
        canonical_recipe_json=canonical_json,
        canonical_recipe_md=None,
        report_bundle_json=None,
        recipe_registry_entry=None,
        glossary_text="",
        prepass_schema_version=1,
    )

    with pytest.raises(HTTPException) as exc_info:
        _delete_prepass_recipe_impl(
            saved["id"],
            recipes_root=tmp_path / "prepass_recipes",
            sanitize_run_id_fn=lambda value: value,
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "prepass_recipe_canonical_delete_forbidden"


def test_collect_recipe_assets_includes_canonical_artifacts_and_bundle(tmp_path):
    export_root = tmp_path / "export"
    export_root.mkdir(parents=True, exist_ok=True)
    registry_root = tmp_path / "calibration_cache" / "recipe_registry" / "8a922d9945b17c16f4ed9dc39f50f5e66b28f614"
    registry_root.mkdir(parents=True, exist_ok=True)
    (registry_root / "registry_entry.json").write_text(
        json.dumps(
            {
                "fingerprint": "8a922d9945b17c16f4ed9dc39f50f5e66b28f614",
                "dataset_id": "qwen_dataset",
                "origin_kind": "discovery_backed",
            }
        ),
        encoding="utf-8",
    )
    (registry_root / "fingerprint.json").write_text(
        json.dumps(
            {
                "dataset_id": "qwen_dataset",
                "labelmap_hash": "labelhash",
                "glossary_hash": "glossaryhash",
                "classifier_id": "DinoV3_best_model_large.pkl",
                "lane_selection": "window",
                "selected_count": 9526,
            }
        ),
        encoding="utf-8",
    )
    recipe_meta = {
        "config": {
            "recipe_kind": "canonical_edr",
            "dataset_id": "qwen_dataset",
            "classifier_id": "DinoV3_best_model_large.pkl",
            "ensemble_job_id": "canonical_edr_qwen_dataset_8a922d9945b1",
            "canonical_edr_json": str((tmp_path / "canonical_edr.json").resolve()),
            "canonical_edr_md": str((tmp_path / "canonical_edr.md").resolve()),
            "canonical_report_bundle_json": str((tmp_path / "report_bundle.json").resolve()),
            "recipe_fingerprint": "8a922d9945b17c16f4ed9dc39f50f5e66b28f614",
            "recipe_registry_fingerprint": "8a922d9945b17c16f4ed9dc39f50f5e66b28f614",
            "recipe_registry_root": str(registry_root.resolve()),
            "labelmap": ["person"],
        },
        "glossary": '{"person":["person"]}',
    }
    (tmp_path / "canonical_edr.json").write_text("{}", encoding="utf-8")
    (tmp_path / "canonical_edr.md").write_text("# canonical", encoding="utf-8")
    (tmp_path / "report_bundle.json").write_text("{}", encoding="utf-8")
    classifier_dir = tmp_path / "uploads" / "classifiers"
    classifier_dir.mkdir(parents=True, exist_ok=True)
    (classifier_dir / "DinoV3_best_model_large.pkl").write_bytes(b"classifier")
    calibration_root = tmp_path / "uploads" / "calibration_jobs"
    job_dir = calibration_root / "canonical_edr_qwen_dataset_8a922d9945b1"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "ensemble_xgb.json").write_text("{}", encoding="utf-8")
    (job_dir / "ensemble_xgb.meta.json").write_text("{}", encoding="utf-8")

    def copy_tree_filtered(src, dest, keep_files=None):
        copied = []
        dest.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            if not item.is_file():
                continue
            if keep_files is not None and item.name not in keep_files:
                continue
            target = dest / item.name
            target.write_bytes(item.read_bytes())
            copied.append(
                {
                    "path": str(target.relative_to(export_root)),
                    "size": target.stat().st_size,
                    "sha256": "sha",
                }
            )
        return copied

    assets = _collect_recipe_assets_impl(
        recipe_meta,
        export_root,
        read_labelmap_lines_fn=lambda path: [],
        load_labelmap_meta_fn=lambda dataset_id: ([], None),
        active_labelmap_path=None,
        sanitize_run_id_fn=lambda value: value,
        copy_tree_filtered_fn=copy_tree_filtered,
        sha256_fn=lambda path: "sha",
        get_qwen_model_entry_fn=lambda model_id: None,
        resolve_classifier_path_fn=lambda classifier_id: classifier_dir / classifier_id,
        yolo_job_root=tmp_path / "uploads" / "yolo",
        rfdetr_job_root=tmp_path / "uploads" / "rfdetr",
        rfdetr_keep_files=None,
        qwen_metadata_filename="metadata.json",
        qwen_job_root=tmp_path / "uploads" / "qwen",
        upload_root=tmp_path / "uploads",
        calibration_root=calibration_root,
    )

    copied_paths = {entry["path"] for entry in assets["copied"]}
    assert "canonical/canonical_edr.json" in copied_paths
    assert "canonical/canonical_edr.md" in copied_paths
    assert "canonical/report_bundle.json" in copied_paths
    assert "canonical/registry_entry.json" in copied_paths
    assert "canonical/fingerprint.json" in copied_paths
    assert "models/calibration_jobs/canonical_edr_qwen_dataset_8a922d9945b1/ensemble_xgb.json" in copied_paths


def test_import_canonical_prepass_recipe_restores_local_canonical_assets_and_registry(tmp_path):
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "manifest.json").write_text(json.dumps({"schema_version": 1, "assets": []}), encoding="utf-8")
    (archive_root / "prepass.meta.json").write_text(
        json.dumps(
            {
                "id": "canonical_old",
                "name": "Canonical EDR",
                "description": "portable canonical",
                "config": {
                    "recipe_kind": "canonical_edr",
                    "dataset_id": "qwen_dataset",
                    "recipe_fingerprint": "8a922d9945b17c16f4ed9dc39f50f5e66b28f614",
                    "recipe_registry_fingerprint": "8a922d9945b17c16f4ed9dc39f50f5e66b28f614",
                    "ensemble_job_id": "canonical_old_job",
                    "classifier_id": "DinoV3_best_model_large.pkl",
                    "labelmap": ["person"],
                },
                "glossary": '{"person":["person"]}',
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    canonical_dir = archive_root / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    (canonical_dir / "canonical_edr.json").write_text("{}", encoding="utf-8")
    (canonical_dir / "canonical_edr.md").write_text("# canonical", encoding="utf-8")
    (canonical_dir / "report_bundle.json").write_text("{}", encoding="utf-8")
    (canonical_dir / "registry_entry.json").write_text(
        json.dumps(
            {
                "fingerprint": "8a922d9945b17c16f4ed9dc39f50f5e66b28f614",
                "dataset_id": "qwen_dataset",
                "origin_kind": "discovery_backed",
                "canonical_deployment_source_stage": "postrun_similarity_quality_full_window_eval",
                "canonical_deployment_source_seed": 42,
            }
        ),
        encoding="utf-8",
    )
    (canonical_dir / "fingerprint.json").write_text(
        json.dumps(
            {
                "dataset_id": "qwen_dataset",
                "labelmap_hash": "labelhash",
                "glossary_hash": "glossaryhash",
                "classifier_id": "DinoV3_best_model_large.pkl",
                "lane_selection": "window",
                "selected_hash": "selhash",
                "selected_count": 9526,
                "selection_seed": 42,
                "requested_max_images": 9526,
                "support_iou": 0.5,
                "context_radius": 0.075,
                "label_iou": 0.5,
                "eval_iou": 0.5,
                "feature_version": 7,
            }
        ),
        encoding="utf-8",
    )
    models_dir = archive_root / "models"
    (models_dir / "classifiers").mkdir(parents=True, exist_ok=True)
    (models_dir / "classifiers" / "DinoV3_best_model_large.pkl").write_bytes(b"classifier")
    calib_job_dir = models_dir / "calibration_jobs" / "canonical_old_job"
    calib_job_dir.mkdir(parents=True, exist_ok=True)
    (calib_job_dir / "ensemble_xgb.json").write_text("{}", encoding="utf-8")
    (calib_job_dir / "ensemble_xgb.meta.json").write_text(
        json.dumps(
            {
                "canonical_recipe_json": "/exporter/machine/canonical_edr.json",
                "canonical_deployment_source_dir": "/exporter/machine/source_dir",
            }
        ),
        encoding="utf-8",
    )
    (calib_job_dir / "canonical_deployment.json").write_text(
        json.dumps(
            {
                "job_id": "canonical_old_job",
                "dataset_id": "qwen_dataset",
                "source_dir": "/exporter/machine/source_dir",
                "source_stage": "postrun_similarity_quality_full_window_eval",
                "source_seed": 42,
                "canonical_recipe_json": "/exporter/machine/canonical_edr.json",
            }
        ),
        encoding="utf-8",
    )
    zip_path = tmp_path / "canonical_recipe.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in archive_root.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(archive_root))

    response = _import_prepass_recipe_from_zip_impl(
        zip_path,
        prepass_recipe_meta="prepass.meta.json",
        prepass_schema_version=1,
        prepass_recipe_root=tmp_path / "prepass_recipes",
        prepass_tmp_root=tmp_path / "tmp",
        yolo_job_root=tmp_path / "uploads" / "yolo",
        rfdetr_job_root=tmp_path / "uploads" / "rfdetr",
        rfdetr_keep_files=None,
        qwen_job_root=tmp_path / "uploads" / "qwen",
        qwen_metadata_filename="metadata.json",
        upload_root=tmp_path / "uploads",
        calibration_root=tmp_path / "uploads" / "calibration_jobs",
        read_labelmap_lines_fn=lambda path: [],
        validate_manifest_fn=lambda manifest, extract_dir: None,
        unique_name_fn=lambda name: (name, None),
        normalize_glossary_fn=lambda value: str(value or "").strip(),
        write_meta_fn=_write_prepass_recipe_meta,
        sanitize_run_id_fn=lambda value: value,
    )

    assert response["id"] == "canonical_edr_qwen_dataset_8a922d9945b1"
    assert response["config"]["dataset_id"] == "qwen_dataset"
    assert Path(response["config"]["canonical_edr_json"]).exists()
    assert Path(response["config"]["canonical_edr_md"]).exists()
    assert Path(response["config"]["canonical_report_bundle_json"]).exists()
    assert response["config"]["recipe_registry_fingerprint"] == "8a922d9945b17c16f4ed9dc39f50f5e66b28f614"
    assert response["config"]["recipe_registry_root"]
    copied_job = (tmp_path / "uploads" / "calibration_jobs" / response["config"]["ensemble_job_id"])
    assert copied_job.exists()
    assert (copied_job / "ensemble_xgb.json").exists()
    assert response["config"]["canonical_deployment_job_id"] == response["config"]["ensemble_job_id"]
    assert response["config"]["canonical_deployment_job_dir"] == str(copied_job.resolve())
    registry_root = Path(response["config"]["recipe_registry_root"])
    registry_entry = json.loads((registry_root / "registry_entry.json").read_text(encoding="utf-8"))
    fingerprint_payload = json.loads((registry_root / "fingerprint.json").read_text(encoding="utf-8"))
    assert registry_entry["origin_kind"] == "imported_portable"
    assert registry_entry["discovery_run_root"] is None
    assert registry_entry["canonical_deployment_job_id"] == response["config"]["ensemble_job_id"]
    assert fingerprint_payload["selected_count"] == 9526
    assert fingerprint_payload["feature_version"] == 7
    copied_meta = json.loads((copied_job / "ensemble_xgb.meta.json").read_text(encoding="utf-8"))
    copied_bundle = json.loads((copied_job / "canonical_deployment.json").read_text(encoding="utf-8"))
    assert copied_meta["canonical_recipe_json"] == str(Path(response["config"]["canonical_edr_json"]).resolve())
    assert copied_meta["original_canonical_recipe_json"] == "/exporter/machine/canonical_edr.json"
    assert copied_bundle["source_dir"] == str(copied_job.resolve())
    assert copied_bundle["original_source_dir"] == "/exporter/machine/source_dir"
    assert copied_bundle["job_id"] == response["config"]["ensemble_job_id"]


def test_export_keeps_dataset_id_for_canonical_recipe(tmp_path):
    recipe_root = tmp_path / "prepass_recipes"
    export_root = tmp_path / "exports"
    export_root.mkdir(parents=True, exist_ok=True)
    recipe_dir = recipe_root / "canonical_edr_qwen_dataset_8a922d9945b1"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    _write_prepass_recipe_meta(
        recipe_dir,
        {
            "id": "canonical_edr_qwen_dataset_8a922d9945b1",
            "schema_version": 1,
            "name": "Canonical EDR",
            "description": "portable canonical",
            "config": {
                "recipe_kind": "canonical_edr",
                "dataset_id": "qwen_dataset",
                "recipe_fingerprint": "8a922d9945b17c16f4ed9dc39f50f5e66b28f614",
            },
            "glossary": "",
            "created_at": 1.0,
            "updated_at": 1.0,
        },
    )

    zip_path = _export_prepass_recipe_impl(
        "canonical_edr_qwen_dataset_8a922d9945b1",
        prepass_recipe_meta="prepass.meta.json",
        prepass_schema_version=1,
        prepass_recipe_export_root=export_root,
        prepass_recipe_root=recipe_root,
        sanitize_run_id_fn=lambda value: value,
        load_meta_fn=_load_prepass_recipe_meta,
        collect_assets_fn=lambda meta, temp_dir: {"copied": [], "missing": []},
    )

    with zipfile.ZipFile(zip_path) as zf:
        exported_meta = json.loads(zf.read("prepass.meta.json").decode("utf-8"))
    assert exported_meta["config"]["dataset_id"] == "qwen_dataset"


def test_list_prepass_recipes_skips_symlinked_recipe_dir_escape(tmp_path: Path) -> None:
    recipes_root = tmp_path / "recipes"
    outside = tmp_path / "outside_recipe"
    recipes_root.mkdir()
    outside.mkdir()
    _write_prepass_recipe_meta(
        outside,
        {
            "id": "escaped",
            "schema_version": 1,
            "name": "escaped",
            "config": {},
            "created_at": 1,
            "updated_at": 1,
        },
    )
    try:
        (recipes_root / "escaped").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    recipes = _list_prepass_recipes_impl(
        recipes_root=recipes_root,
        meta_filename="prepass.meta.json",
    )

    assert recipes == []


def test_delete_agent_recipe_unlinks_symlinked_recipe_dir_without_touching_target(
    tmp_path: Path,
) -> None:
    recipes_root = tmp_path / "recipes"
    outside = tmp_path / "outside_recipe"
    recipes_root.mkdir()
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    link_path = recipes_root / "agent_recipe"
    try:
        link_path.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _delete_agent_recipe_impl(
        "agent_recipe",
        recipes_root=recipes_root,
        path_is_within_root_fn=_within_root,
        http_exception_cls=HTTPException,
    )

    assert not link_path.is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_list_agent_recipes_skips_symlinked_json_escape(tmp_path: Path) -> None:
    recipes_root = tmp_path / "recipes"
    outside = tmp_path / "outside.json"
    recipes_root.mkdir()
    outside.write_text(json.dumps({"id": "outside", "created_at": 1}), encoding="utf-8")
    try:
        (recipes_root / "outside.json").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    recipes = _list_agent_recipes_impl(recipes_root=recipes_root)

    assert recipes == []


def test_ensure_recipe_zip_rejects_symlinked_recipe_dir_escape(tmp_path: Path) -> None:
    recipes_root = tmp_path / "recipes"
    outside = tmp_path / "outside_recipe"
    recipes_root.mkdir()
    outside.mkdir()
    (outside / "crops").mkdir()
    (outside / "crops" / "secret.png").write_bytes(b"secret")
    try:
        (recipes_root / "recipe_a").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(HTTPException) as exc_info:
        _ensure_recipe_zip_impl({"id": "recipe_a"}, recipes_root=recipes_root)

    assert exc_info.value.detail == "agent_recipe_path_invalid"
