from __future__ import annotations

import asyncio
import io
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sklearn.exceptions import InconsistentVersionWarning
from starlette.datastructures import UploadFile
from starlette.responses import FileResponse

import localinferenceapi


async def _stream_body(response) -> bytes:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    return b"".join(chunks)


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


def test_download_clip_classifier_zip_skips_symlink_meta_escape(tmp_path, monkeypatch) -> None:
    upload_root = tmp_path / "uploads"
    classifiers_root = upload_root / "classifiers"
    classifiers_root.mkdir(parents=True, exist_ok=True)
    classifier_path = classifiers_root / "head.pkl"
    classifier_path.write_bytes(b"model")
    outside = tmp_path / "outside.meta.pkl"
    outside.write_bytes(b"secret")
    try:
        (classifiers_root / "head.meta.pkl").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(localinferenceapi, "UPLOAD_ROOT", upload_root)
    monkeypatch.setattr(
        localinferenceapi,
        "_resolve_agent_clip_classifier_path_impl",
        lambda *args, **kwargs: classifier_path,
    )
    monkeypatch.setattr(
        localinferenceapi,
        "_find_labelmap_for_classifier_impl",
        lambda *args, **kwargs: None,
    )

    response = localinferenceapi.download_clip_classifier_zip(rel_path="head.pkl")
    raw = asyncio.run(_stream_body(response))

    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
        names = set(zf.namelist())
        payloads = {name: zf.read(name) for name in names}
    assert names == {"head.pkl"}
    assert payloads["head.pkl"] == b"model"


def test_load_clip_head_skips_symlink_meta_escape(tmp_path) -> None:
    classifier_path = tmp_path / "head.pkl"
    classifier_path.write_bytes(b"model")
    outside = tmp_path / "outside.meta.pkl"
    outside.write_bytes(b"secret")
    try:
        (tmp_path / "head.meta.pkl").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    classifier = SimpleNamespace(
        classes_=["car", "boat"],
        coef_=[[0.0] * 512, [0.0] * 512],
        intercept_=[0.0, 0.0],
        solver="lbfgs",
        multi_class="auto",
    )
    loaded_paths = []

    def fake_load(path):
        loaded_paths.append(Path(path))
        if Path(path) == classifier_path:
            return classifier
        raise AssertionError(f"unexpected metadata load: {path}")

    head = localinferenceapi._load_clip_head_from_classifier_impl(
        classifier_path,
        joblib_load_fn=fake_load,
        http_exception_cls=HTTPException,
        clip_head_background_indices_fn=lambda _classes: [],
        resolve_head_normalize_embeddings_fn=lambda _head, default=True: default,
        infer_clip_model_fn=lambda _dim, active_name=None: active_name,
        active_clip_model_name="ViT-B/32",
        default_clip_model="ViT-B/32",
        logger=localinferenceapi.logger,
    )

    assert head["classes"] == ["car", "boat"]
    assert loaded_paths == [classifier_path]


def test_find_labelmap_for_classifier_skips_symlink_meta_escape(tmp_path) -> None:
    upload_root = tmp_path / "uploads"
    classifiers_root = upload_root / "classifiers"
    classifiers_root.mkdir(parents=True)
    classifier_path = classifiers_root / "head.pkl"
    classifier_path.write_bytes(b"model")
    outside = tmp_path / "outside.meta.pkl"
    outside.write_bytes(b"secret")
    try:
        (classifiers_root / "head.meta.pkl").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    def fail_load(path):
        raise AssertionError(f"unsafe metadata should not be loaded: {path}")

    found = localinferenceapi._find_labelmap_for_classifier_impl(
        classifier_path,
        upload_root=upload_root,
        labelmap_exts=localinferenceapi.LABELMAP_ALLOWED_EXTS,
        path_is_within_root_fn=localinferenceapi._path_is_within_root_impl,
        joblib_load_fn=fail_load,
        resolve_clip_labelmap_path_fn=lambda _path, _root_hint=None: None,
    )

    assert found is None


def test_list_clip_classifiers_skips_symlink_meta_escape(tmp_path) -> None:
    upload_root = tmp_path / "uploads"
    classifiers_root = upload_root / "classifiers"
    classifiers_root.mkdir(parents=True)
    classifier_path = classifiers_root / "head.pkl"
    classifier_path.write_bytes(b"model")
    outside = tmp_path / "outside.meta.pkl"
    outside.write_bytes(b"secret")
    try:
        (classifiers_root / "head.meta.pkl").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    loaded_paths = []

    def fake_load(path):
        loaded_paths.append(Path(path))
        if Path(path).name == "head.pkl":
            return {"classes": ["car", "boat"], "classifier_type": "logreg"}
        raise AssertionError(f"unsafe metadata should not be loaded: {path}")

    entries = localinferenceapi._list_clip_classifiers_impl(
        upload_root=upload_root,
        classifier_exts=localinferenceapi.CLASSIFIER_ALLOWED_EXTS,
        labelmap_exts=localinferenceapi.LABELMAP_ALLOWED_EXTS,
        path_is_within_root_fn=localinferenceapi._path_is_within_root_impl,
        joblib_load_fn=fake_load,
        resolve_clip_labelmap_path_fn=lambda _path, _root_hint=None: None,
    )

    assert len(entries) == 1
    assert entries[0]["rel_path"] == "head.pkl"
    assert entries[0]["n_classes"] == 2
    assert loaded_paths == [classifier_path]


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


def test_delete_clip_classifier_unlinks_broken_meta_symlink(tmp_path, monkeypatch) -> None:
    classifier_path = tmp_path / "head.pkl"
    classifier_path.write_bytes(b"model")
    meta_path = tmp_path / "head.meta.pkl"
    try:
        meta_path.symlink_to(tmp_path / "missing.meta.pkl")
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(
        localinferenceapi,
        "_resolve_agent_clip_classifier_path_impl",
        lambda *args, **kwargs: classifier_path,
    )
    monkeypatch.setattr(localinferenceapi, "active_classifier_path", None)

    response = localinferenceapi.delete_clip_classifier(rel_path="head.pkl")

    assert response == {"status": "deleted", "rel_path": "head.pkl"}
    assert not classifier_path.exists()
    assert not meta_path.is_symlink()


def test_delete_clip_classifier_rejects_symlink_alias_without_target_unlink(
    tmp_path, monkeypatch
) -> None:
    upload_root = tmp_path / "uploads"
    classifiers_root = upload_root / "classifiers"
    classifiers_root.mkdir(parents=True)
    target = classifiers_root / "target.pkl"
    target.write_bytes(b"model")
    alias = classifiers_root / "alias.pkl"
    try:
        alias.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(localinferenceapi, "UPLOAD_ROOT", upload_root)

    with pytest.raises(HTTPException) as exc_info:
        localinferenceapi.delete_clip_classifier(rel_path="alias.pkl")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "agent_clip_classifier_path_not_allowed"
    assert target.read_bytes() == b"model"
    assert alias.is_symlink()


def test_rename_clip_classifier_moves_file_meta_and_active_state(tmp_path, monkeypatch) -> None:
    upload_root = tmp_path / "uploads"
    classifiers_root = upload_root / "classifiers"
    classifiers_root.mkdir(parents=True)
    classifier_path = classifiers_root / "head.pkl"
    meta_path = classifiers_root / "head.meta.pkl"
    classifier_path.write_bytes(b"model")
    meta_path.write_bytes(b"meta")
    monkeypatch.setattr(localinferenceapi, "UPLOAD_ROOT", upload_root)
    monkeypatch.setattr(localinferenceapi, "active_classifier_path", str(classifier_path))

    response = localinferenceapi.rename_clip_classifier(
        rel_path="head.pkl",
        new_name="renamed",
    )

    renamed_path = classifiers_root / "renamed.pkl"
    renamed_meta = classifiers_root / "renamed.meta.pkl"
    assert response["status"] == "renamed"
    assert response["old_rel_path"] == "head.pkl"
    assert response["new_rel_path"] == "renamed.pkl"
    assert renamed_path.read_bytes() == b"model"
    assert renamed_meta.read_bytes() == b"meta"
    assert not classifier_path.exists()
    assert not meta_path.exists()
    assert localinferenceapi.active_classifier_path == str(renamed_path)


def test_rename_clip_classifier_does_not_follow_existing_target_symlink(
    tmp_path, monkeypatch
) -> None:
    upload_root = tmp_path / "uploads"
    classifiers_root = upload_root / "classifiers"
    classifiers_root.mkdir(parents=True)
    classifier_path = classifiers_root / "head.pkl"
    classifier_path.write_bytes(b"model")
    symlink_target = classifiers_root / "hidden.pkl"
    alias = classifiers_root / "alias.pkl"
    try:
        alias.symlink_to(symlink_target)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(localinferenceapi, "UPLOAD_ROOT", upload_root)

    response = localinferenceapi.rename_clip_classifier(
        rel_path="head.pkl",
        new_name="alias.pkl",
    )

    safe_target = classifiers_root / "alias_1.pkl"
    assert response["status"] == "renamed"
    assert response["new_rel_path"] == "alias_1.pkl"
    assert safe_target.read_bytes() == b"model"
    assert alias.is_symlink()
    assert not symlink_target.exists()
    assert not classifier_path.exists()


def test_resolve_clip_classifier_rejects_nested_symlinked_registry_parent_without_write(
    tmp_path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(HTTPException) as exc_info:
        localinferenceapi._resolve_agent_clip_classifier_path_impl(
            "head.pkl",
            allowed_root=linked_parent / "nested" / "classifiers",
            allowed_exts=(".pkl",),
            path_is_within_root_fn=localinferenceapi._path_is_within_root_impl,
            http_exception_cls=HTTPException,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "agent_clip_classifier_path_not_allowed"
    assert list(outside.iterdir()) == []


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
        clip_warnings=["legacy sklearn artifact"],
        encoder_ready=False,
        encoder_error="classifier_not_found",
        labelmap_entries=[],
    )
    payload = response.model_dump() if hasattr(response, "model_dump") else response.dict()

    assert payload["clip_error"] == "classifier_not_found"
    assert payload["clip_warnings"] == ["legacy sklearn artifact"]
    assert payload["encoder_ready"] is False
    assert payload["encoder_error"] == "classifier_not_found"


def test_joblib_load_records_sklearn_version_warning(monkeypatch) -> None:
    localinferenceapi.classifier_artifact_warnings.clear()
    localinferenceapi._classifier_artifact_warning_keys.clear()

    def fake_load(_path):
        warnings.warn(
            InconsistentVersionWarning(
                estimator_name="LogisticRegression",
                current_sklearn_version="1.8.0",
                original_sklearn_version="1.7.2",
            )
        )
        return {"ok": True}

    monkeypatch.setattr(localinferenceapi.joblib, "load", fake_load)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        loaded = localinferenceapi._joblib_load("legacy.pkl")

    assert loaded == {"ok": True}
    assert not captured
    assert localinferenceapi.classifier_artifact_warnings
    assert "legacy.pkl" in localinferenceapi.classifier_artifact_warnings[-1]


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


def test_list_clip_labelmaps_skips_symlink_escape(tmp_path) -> None:
    upload_root = tmp_path / "uploads"
    labelmaps_root = upload_root / "labelmaps"
    labelmaps_root.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    (labelmaps_root / "escape.txt").symlink_to(outside)

    entries = localinferenceapi._list_clip_labelmaps_impl(
        upload_root=upload_root,
        labelmap_exts=localinferenceapi.LABELMAP_ALLOWED_EXTS,
        load_labelmap_file_fn=lambda _path: ["secret"],
        path_is_within_root_fn=localinferenceapi._path_is_within_root_impl,
    )

    assert entries == []


def test_delete_clip_labelmap_rejects_symlink_alias_without_target_unlink(
    tmp_path, monkeypatch
) -> None:
    upload_root = tmp_path / "uploads"
    labelmaps_root = upload_root / "labelmaps"
    labelmaps_root.mkdir(parents=True)
    target = labelmaps_root / "target.txt"
    target.write_text("car\n", encoding="utf-8")
    alias = labelmaps_root / "alias.txt"
    try:
        alias.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(localinferenceapi, "UPLOAD_ROOT", upload_root)

    with pytest.raises(HTTPException) as exc_info:
        localinferenceapi.delete_clip_labelmap(rel_path="alias.txt", root="labelmaps")

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "labelmap_not_found"
    assert target.read_text(encoding="utf-8") == "car\n"
    assert alias.is_symlink()


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


def test_upload_classifier_rejects_symlinked_registry_root_without_write(
    tmp_path, monkeypatch
) -> None:
    upload_root = tmp_path / "uploads"
    outside = tmp_path / "outside_classifiers"
    upload_root.mkdir()
    outside.mkdir()
    try:
        (upload_root / "classifiers").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(localinferenceapi, "UPLOAD_ROOT", upload_root)
    upload = UploadFile(filename="head.pkl", file=io.BytesIO(b"model"))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(localinferenceapi.upload_classifier(file=upload))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "classifiers_root_invalid"
    assert list(outside.iterdir()) == []


def test_upload_classifier_rejects_symlinked_upload_root_parent_without_write(
    tmp_path, monkeypatch
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(localinferenceapi, "UPLOAD_ROOT", linked_parent / "uploads")
    upload = UploadFile(filename="head.pkl", file=io.BytesIO(b"model"))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(localinferenceapi.upload_classifier(file=upload))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "classifiers_root_invalid"
    assert list(outside.iterdir()) == []


def test_save_upload_file_rejects_symlinked_write_root_parent_without_write(
    tmp_path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    upload = UploadFile(filename="head.pkl", file=io.BytesIO(b"model"))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(localinferenceapi._save_upload_file(upload, linked_parent / "classifiers"))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_relative_path"
    assert list(outside.iterdir()) == []
