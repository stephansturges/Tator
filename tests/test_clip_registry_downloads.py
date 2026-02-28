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
