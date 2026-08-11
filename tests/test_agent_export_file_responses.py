from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.responses import FileResponse

import localinferenceapi


def test_agent_mining_export_recipe_returns_file_response(tmp_path, monkeypatch) -> None:
    zip_path = tmp_path / "recipe.zip"
    zip_path.write_bytes(b"zip")
    monkeypatch.setattr(localinferenceapi, "_load_agent_recipe_impl", lambda *args, **kwargs: {"id": "r1"})
    monkeypatch.setattr(localinferenceapi, "_ensure_recipe_zip", lambda _recipe: zip_path)

    response = localinferenceapi.agent_mining_export_recipe("recipe_1")
    assert isinstance(response, FileResponse)
    assert Path(response.path) == zip_path
    assert response.filename == "recipe_1.zip"


def test_agent_mining_export_recipe_wraps_file_errors(monkeypatch) -> None:
    monkeypatch.setattr(localinferenceapi, "_load_agent_recipe_impl", lambda *args, **kwargs: {"id": "r1"})
    monkeypatch.setattr(localinferenceapi, "_ensure_recipe_zip", lambda _recipe: Path("/definitely/missing.zip"))

    with pytest.raises(HTTPException) as exc:
        localinferenceapi.agent_mining_export_recipe("recipe_1")
    assert exc.value.status_code == 500
    assert str(exc.value.detail).startswith("agent_recipe_export_failed:")


def test_agent_mining_export_cascade_returns_file_response(tmp_path, monkeypatch) -> None:
    zip_path = tmp_path / "cascade.zip"
    zip_path.write_bytes(b"zip")
    monkeypatch.setattr(localinferenceapi, "_load_agent_cascade_impl", lambda *args, **kwargs: {"id": "c1"})
    monkeypatch.setattr(localinferenceapi, "_ensure_cascade_zip", lambda _cascade: zip_path)

    response = localinferenceapi.agent_mining_export_cascade("cascade_1")
    assert isinstance(response, FileResponse)
    assert Path(response.path) == zip_path
    assert response.filename == "cascade_1.zip"


def test_agent_mining_export_cascade_wraps_file_errors(monkeypatch) -> None:
    monkeypatch.setattr(localinferenceapi, "_load_agent_cascade_impl", lambda *args, **kwargs: {"id": "c1"})
    monkeypatch.setattr(localinferenceapi, "_ensure_cascade_zip", lambda _cascade: Path("/definitely/missing.zip"))

    with pytest.raises(HTTPException) as exc:
        localinferenceapi.agent_mining_export_cascade("cascade_1")
    assert exc.value.status_code == 500
    assert str(exc.value.detail).startswith("agent_cascade_export_failed:")


def test_export_prepass_recipe_edr_wrapper_maps_invalid_package(tmp_path, monkeypatch) -> None:
    recipe_dir = tmp_path / "recipe"
    recipe_dir.mkdir()
    monkeypatch.setattr(localinferenceapi, "_prepass_recipe_dir_impl", lambda *args, **kwargs: recipe_dir)
    monkeypatch.setattr(
        localinferenceapi,
        "_load_prepass_recipe_meta",
        lambda _recipe_dir: {"config": {"edr_package_id": "pkg1"}},
    )

    def fail_export(*_args, **_kwargs):
        raise RuntimeError("edr_package_invalid_zip")

    monkeypatch.setattr(localinferenceapi, "_export_edr_package_impl", fail_export)

    with pytest.raises(HTTPException) as exc:
        localinferenceapi.export_prepass_recipe("recipe_1")

    assert exc.value.status_code == 412
    assert exc.value.detail == "edr_package_invalid_zip"
