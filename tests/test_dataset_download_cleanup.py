from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException

import localinferenceapi


def test_download_dataset_entry_sets_background_cleanup(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "sample.txt").write_text("hello", encoding="utf-8")

    monkeypatch.setattr(
        localinferenceapi,
        "_resolve_dataset_entry",
        lambda dataset_id: {"dataset_root": str(dataset_root), "id": dataset_id},
    )

    response = localinferenceapi.download_dataset_entry("demo_dataset")
    zip_path = Path(response.path)
    assert zip_path.exists()
    assert response.background is not None

    asyncio.run(response.background())
    assert not zip_path.parent.exists()


def test_download_dataset_entry_cleans_tmp_dir_on_zip_error(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "sample.txt").write_text("hello", encoding="utf-8")
    export_dir = tmp_path / "dataset_export_tmp"

    monkeypatch.setattr(
        localinferenceapi,
        "_resolve_dataset_entry",
        lambda dataset_id: {"dataset_root": str(dataset_root), "id": dataset_id},
    )
    monkeypatch.setattr(localinferenceapi.tempfile, "mkdtemp", lambda prefix: str(export_dir))

    def _boom_zipfile(*_args, **_kwargs):
        raise RuntimeError("zip_boom")

    monkeypatch.setattr(localinferenceapi.zipfile, "ZipFile", _boom_zipfile)

    with pytest.raises(HTTPException) as exc:
        localinferenceapi.download_dataset_entry("demo_dataset")
    assert exc.value.status_code == 500
    assert exc.value.detail.startswith("dataset_export_failed:")
    assert not export_dir.exists()
