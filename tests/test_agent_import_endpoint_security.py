from __future__ import annotations

import asyncio
from io import BytesIO

import pytest
from fastapi import UploadFile

import localinferenceapi as api


def test_agent_recipe_import_rejects_symlinked_endpoint_root_before_staging(
    tmp_path,
    monkeypatch,
) -> None:
    outside = tmp_path / "outside_recipes"
    outside.mkdir()
    recipes_root = tmp_path / "recipes"
    try:
        recipes_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "AGENT_MINING_RECIPES_ROOT", recipes_root)
    monkeypatch.setattr(
        api,
        "_import_agent_recipe_zip_file",
        lambda _path: pytest.fail("importer should not run"),
    )
    upload = UploadFile(filename="recipe.zip", file=BytesIO(b"zip-bytes"))

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(api.agent_mining_import_recipe(upload))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "agent_recipe_import_invalid_path"
    assert list(outside.iterdir()) == []
    assert upload.file.closed


def test_agent_cascade_import_rejects_symlinked_endpoint_root_before_staging(
    tmp_path,
    monkeypatch,
) -> None:
    outside = tmp_path / "outside_cascades"
    outside.mkdir()
    cascades_root = tmp_path / "cascades"
    try:
        cascades_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")
    monkeypatch.setattr(api, "AGENT_MINING_CASCADES_ROOT", cascades_root)
    monkeypatch.setattr(
        api,
        "_import_agent_cascade_zip_file",
        lambda _path: pytest.fail("importer should not run"),
    )
    upload = UploadFile(filename="cascade.zip", file=BytesIO(b"zip-bytes"))

    with pytest.raises(api.HTTPException) as exc_info:
        asyncio.run(api.agent_mining_import_cascade(upload))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "agent_cascade_import_invalid_path"
    assert list(outside.iterdir()) == []
    assert upload.file.closed
