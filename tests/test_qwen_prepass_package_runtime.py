from __future__ import annotations

import localinferenceapi as api
from models.schemas import QwenPrepassRequest


def test_edr_package_runtime_can_apply_prepass_without_ensemble(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "_resolve_edr_package_runtime_impl",
        lambda **_kwargs: {
            "labelmap": ["car"],
            "runtime_config": {
                "dataset_id": "ds1",
                "enable_yolo": True,
                "enable_rfdetr": True,
                "enable_sam3_text": True,
                "enable_sam3_similarity": True,
            },
            "staged_ensemble_job_id": "bundle_1",
            "staged_classifier_id": "clf_1.pkl",
            "glossary_text": "car",
        },
    )
    monkeypatch.setattr(api, "_agent_load_labelmap_meta", lambda _dataset_id: (["car"], "car"))

    req = QwenPrepassRequest(
        dataset_id="ds1",
        edr_package_id="pkg1",
        edr_package_apply_ensemble=False,
        ensemble_enabled=False,
    )

    updated, runtime = api._agent_apply_edr_package_runtime(req)

    assert runtime is not None
    assert updated.enable_yolo is True
    assert updated.enable_rfdetr is True
    assert updated.enable_sam3_text is True
    assert updated.enable_sam3_similarity is True
    assert updated.ensemble_enabled is False
    assert updated.ensemble_job_id is None
    assert updated.classifier_id == "clf_1.pkl"
