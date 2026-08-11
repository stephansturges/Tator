from __future__ import annotations

import pytest

import localinferenceapi as api
from models.schemas import QwenPrepassRequest


def test_agent_apply_edr_package_runtime_falls_back_to_source_dataset_labelmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = QwenPrepassRequest(
        dataset_id="linked_qwen_clone",
        edr_package_id="pkg1",
        image_base64="abc",
    )

    monkeypatch.setattr(
        api,
        "_resolve_edr_package_runtime_impl",
        lambda **_kwargs: {
            "labelmap": [],
            "runtime_config": {
                "dataset_id": "qwen_dataset",
                "recipe_source_dataset_id": "qwen_dataset",
                "resolved_classifier_id": "/tmp/classifier.pkl",
                "enable_yolo": True,
                "enable_rfdetr": True,
            },
            "staged_ensemble_job_id": "bundle1",
            "glossary_text": "",
            "staged_yolo_id": None,
            "staged_rfdetr_id": None,
            "sam3_checkpoint_path": None,
            "staged_classifier_id": None,
            "feature_contract": {},
        },
    )

    def _load_labelmap(dataset_id: str):
        if dataset_id in {"qwen_dataset", "linked_qwen_clone"}:
            return (["light_vehicle", "person"], "")
        raise AssertionError(dataset_id)

    monkeypatch.setattr(api, "_agent_load_labelmap_meta", _load_labelmap)

    updated, runtime = api._agent_apply_edr_package_runtime(payload)

    assert updated.labelmap == ["light_vehicle", "person"]
    assert updated.dataset_id == "linked_qwen_clone"
    assert updated.recipe_source_dataset_id == "qwen_dataset"
    assert updated.ensemble_job_id == "bundle1"
    assert runtime["staged_ensemble_job_id"] == "bundle1"
