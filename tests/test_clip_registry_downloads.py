from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.responses import FileResponse

import localinferenceapi


def test_download_clip_classifier_returns_file_response(tmp_path, monkeypatch) -> None:
    classifier_path = tmp_path / "head.pkl"
    classifier_path.write_bytes(b"model")
    monkeypatch.setattr(
        localinferenceapi,
        "_resolve_agent_clip_classifier_path_impl",
        lambda *args, **kwargs: classifier_path,
    )

    response = localinferenceapi.download_clip_classifier(rel_path="head.pkl")
    assert isinstance(response, FileResponse)
    assert Path(response.path) == classifier_path
    assert response.filename == "head.pkl"


def test_download_clip_classifier_not_found(monkeypatch) -> None:
    monkeypatch.setattr(
        localinferenceapi,
        "_resolve_agent_clip_classifier_path_impl",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(HTTPException) as exc:
        localinferenceapi.download_clip_classifier(rel_path="missing.pkl")
    assert exc.value.status_code == 404
    assert exc.value.detail == "classifier_not_found"


def test_delete_active_clip_classifier_clears_active_state(tmp_path, monkeypatch) -> None:
    classifier_path = tmp_path / "head.pkl"
    meta_path = tmp_path / "head.meta.pkl"
    labelmap_path = tmp_path / "labels.pkl"
    classifier_path.write_bytes(b"model")
    meta_path.write_bytes(b"meta")
    labelmap_path.write_bytes(b"labels")
    stale_head = {"classes": ["car"]}
    monkeypatch.setattr(
        localinferenceapi,
        "_resolve_agent_clip_classifier_path_impl",
        lambda *args, **kwargs: classifier_path,
    )
    monkeypatch.setattr(localinferenceapi, "clf", object())
    monkeypatch.setattr(localinferenceapi, "active_classifier_path", str(classifier_path))
    monkeypatch.setattr(localinferenceapi, "active_labelmap_path", str(labelmap_path))
    monkeypatch.setattr(localinferenceapi, "active_label_list", ["car"])
    monkeypatch.setattr(localinferenceapi, "active_classifier_meta", {"encoder_type": "clip"})
    monkeypatch.setattr(localinferenceapi, "active_head_normalize_embeddings", False)
    monkeypatch.setattr(localinferenceapi, "active_classifier_head", stale_head)
    monkeypatch.setattr(localinferenceapi, "clip_last_error", None)

    response = localinferenceapi.delete_clip_classifier(rel_path="head.pkl")

    assert response == {"status": "deleted", "rel_path": "head.pkl"}
    assert not classifier_path.exists()
    assert not meta_path.exists()
    assert localinferenceapi.clf is None
    assert localinferenceapi.active_classifier_path is None
    assert localinferenceapi.active_labelmap_path is None
    assert localinferenceapi.active_label_list == []
    assert localinferenceapi.active_classifier_meta == {}
    assert localinferenceapi.active_head_normalize_embeddings is True
    assert localinferenceapi.active_classifier_head is None
    assert localinferenceapi.clip_last_error == "classifier_deleted"


def test_active_classifier_head_for_inference_drops_missing_cached_head(tmp_path, monkeypatch) -> None:
    missing_path = tmp_path / "missing.pkl"
    stale_head = {"classes": ["car"]}
    monkeypatch.setattr(localinferenceapi, "clf", object())
    monkeypatch.setattr(localinferenceapi, "active_classifier_path", str(missing_path))
    monkeypatch.setattr(localinferenceapi, "active_labelmap_path", str(tmp_path / "labels.pkl"))
    monkeypatch.setattr(localinferenceapi, "active_label_list", ["car"])
    monkeypatch.setattr(localinferenceapi, "active_classifier_meta", {"encoder_type": "clip"})
    monkeypatch.setattr(localinferenceapi, "active_head_normalize_embeddings", False)
    monkeypatch.setattr(localinferenceapi, "active_classifier_head", stale_head)
    monkeypatch.setattr(localinferenceapi, "clip_last_error", None)

    assert localinferenceapi._active_classifier_head_for_inference() is None
    assert localinferenceapi.clf is None
    assert localinferenceapi.active_classifier_path is None
    assert localinferenceapi.active_classifier_head is None
    assert localinferenceapi.clip_last_error == "classifier_not_found"


def test_active_encoder_ready_clears_missing_classifier_before_inference(
    tmp_path, monkeypatch
) -> None:
    missing_path = tmp_path / "missing.pkl"
    stale_head = {"classes": ["car"]}
    monkeypatch.setattr(localinferenceapi, "clf", object())
    monkeypatch.setattr(localinferenceapi, "active_classifier_path", str(missing_path))
    monkeypatch.setattr(localinferenceapi, "active_labelmap_path", str(tmp_path / "labels.pkl"))
    monkeypatch.setattr(localinferenceapi, "active_label_list", ["car"])
    monkeypatch.setattr(localinferenceapi, "active_classifier_meta", {"encoder_type": "clip"})
    monkeypatch.setattr(localinferenceapi, "active_head_normalize_embeddings", False)
    monkeypatch.setattr(localinferenceapi, "active_classifier_head", stale_head)
    monkeypatch.setattr(localinferenceapi, "active_encoder_type", "clip")
    monkeypatch.setattr(localinferenceapi, "clip_initialized", True)
    monkeypatch.setattr(localinferenceapi, "clip_model", object())
    monkeypatch.setattr(localinferenceapi, "clip_preprocess", object())
    monkeypatch.setattr(localinferenceapi, "clip_last_error", None)

    assert localinferenceapi._active_encoder_ready() is False
    assert localinferenceapi.clf is None
    assert localinferenceapi.active_classifier_path is None
    assert localinferenceapi.active_classifier_head is None
    assert localinferenceapi.clip_last_error == "classifier_not_found"


def test_current_active_payload_clears_missing_classifier_before_readiness(tmp_path, monkeypatch) -> None:
    missing_path = tmp_path / "missing.pkl"
    stale_head = {"classes": ["car"]}
    monkeypatch.setattr(localinferenceapi, "clf", object())
    monkeypatch.setattr(localinferenceapi, "active_classifier_path", str(missing_path))
    monkeypatch.setattr(localinferenceapi, "active_labelmap_path", str(tmp_path / "labels.pkl"))
    monkeypatch.setattr(localinferenceapi, "active_label_list", ["car"])
    monkeypatch.setattr(localinferenceapi, "active_classifier_meta", {"encoder_type": "clip"})
    monkeypatch.setattr(localinferenceapi, "active_head_normalize_embeddings", False)
    monkeypatch.setattr(localinferenceapi, "active_classifier_head", stale_head)
    monkeypatch.setattr(localinferenceapi, "active_encoder_type", "clip")
    monkeypatch.setattr(localinferenceapi, "clip_initialized", True)
    monkeypatch.setattr(localinferenceapi, "clip_model", object())
    monkeypatch.setattr(localinferenceapi, "clip_preprocess", object())
    monkeypatch.setattr(localinferenceapi, "clip_last_error", None)

    payload = localinferenceapi._current_active_payload()

    assert payload["clip_ready"] is False
    assert payload["classifier_path"] is None
    assert payload["labelmap_path"] is None
    assert payload["labelmap_entries"] == []
    assert payload["clip_error"] == "classifier_not_found"


def test_active_model_response_preserves_readiness_error_fields() -> None:
    response = localinferenceapi.ActiveModelResponse(
        clip_model="ViT-B/32",
        encoder_type="clip",
        encoder_model="ViT-B/32",
        classifier_path=None,
        labelmap_path=None,
        clip_ready=False,
        clip_error="classifier_not_found",
        encoder_ready=False,
        encoder_error="classifier_not_found",
        labelmap_entries=[],
    )
    payload = response.model_dump() if hasattr(response, "model_dump") else response.dict()

    assert payload["clip_error"] == "classifier_not_found"
    assert payload["encoder_ready"] is False
    assert payload["encoder_error"] == "classifier_not_found"


def test_predict_base64_skips_decode_when_classifier_path_disappeared(
    tmp_path, monkeypatch
) -> None:
    missing_path = tmp_path / "missing.pkl"
    monkeypatch.setattr(localinferenceapi, "clf", object())
    monkeypatch.setattr(localinferenceapi, "active_classifier_path", str(missing_path))
    monkeypatch.setattr(localinferenceapi, "active_labelmap_path", str(tmp_path / "labels.pkl"))
    monkeypatch.setattr(localinferenceapi, "active_label_list", ["car"])
    monkeypatch.setattr(localinferenceapi, "active_classifier_meta", {"encoder_type": "clip"})
    monkeypatch.setattr(localinferenceapi, "active_head_normalize_embeddings", False)
    monkeypatch.setattr(localinferenceapi, "active_classifier_head", {"classes": ["car"]})
    monkeypatch.setattr(localinferenceapi, "active_encoder_type", "clip")
    monkeypatch.setattr(localinferenceapi, "clip_initialized", True)
    monkeypatch.setattr(localinferenceapi, "clip_model", object())
    monkeypatch.setattr(localinferenceapi, "clip_preprocess", object())
    monkeypatch.setattr(localinferenceapi, "clip_last_error", None)

    def fail_decode(*args, **kwargs):
        raise AssertionError("missing classifier should be rejected before decoding image data")

    monkeypatch.setattr(localinferenceapi, "_resolve_detector_image_impl", fail_decode)

    response = localinferenceapi.predict_base64(
        localinferenceapi.Base64Payload(image_base64="ignored", uuid="stale")
    )

    assert response.error == "clip_unavailable"
    assert localinferenceapi.clf is None
    assert localinferenceapi.clip_last_error == "classifier_not_found"


def test_download_clip_labelmap_returns_file_response(tmp_path, monkeypatch) -> None:
    labelmap_path = tmp_path / "labelmap.txt"
    labelmap_path.write_text("car\n", encoding="utf-8")
    monkeypatch.setattr(
        localinferenceapi,
        "_resolve_clip_labelmap_path_impl",
        lambda *args, **kwargs: labelmap_path,
    )

    response = localinferenceapi.download_clip_labelmap(rel_path="labelmap.txt", root=None)
    assert isinstance(response, FileResponse)
    assert Path(response.path) == labelmap_path
    assert response.filename == "labelmap.txt"


def test_download_clip_labelmap_not_found(monkeypatch) -> None:
    monkeypatch.setattr(
        localinferenceapi,
        "_resolve_clip_labelmap_path_impl",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(HTTPException) as exc:
        localinferenceapi.download_clip_labelmap(rel_path="missing.txt", root=None)
    assert exc.value.status_code == 404
    assert exc.value.detail == "labelmap_not_found"


def test_delete_active_clip_labelmap_clears_active_label_state(tmp_path, monkeypatch) -> None:
    labelmap_path = tmp_path / "labelmap.pkl"
    labelmap_path.write_bytes(b"labels")
    monkeypatch.setattr(
        localinferenceapi,
        "_resolve_clip_labelmap_path_impl",
        lambda *args, **kwargs: labelmap_path,
    )
    monkeypatch.setattr(localinferenceapi, "active_labelmap_path", str(labelmap_path))
    monkeypatch.setattr(localinferenceapi, "active_label_list", ["car"])

    response = localinferenceapi.delete_clip_labelmap(rel_path="labelmap.pkl", root=None)

    assert response == {"status": "deleted", "rel_path": "labelmap.pkl"}
    assert not labelmap_path.exists()
    assert localinferenceapi.active_labelmap_path is None
    assert localinferenceapi.active_label_list == []
