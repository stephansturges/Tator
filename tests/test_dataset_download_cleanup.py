from __future__ import annotations

import asyncio
import zipfile
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


def test_download_dataset_entry_rejects_incomplete_zip_before_serving(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "sample.txt").write_text("hello", encoding="utf-8")
    export_parent = tmp_path / "exports"
    export_parent.mkdir()

    monkeypatch.setattr(
        localinferenceapi,
        "_resolve_dataset_entry",
        lambda dataset_id: {"dataset_root": str(dataset_root), "id": dataset_id},
    )

    real_zipfile = zipfile.ZipFile

    class DroppingZipFile:
        def __init__(self, *args, **kwargs):
            self._mode = str(args[1] if len(args) > 1 else kwargs.get("mode", "r"))
            self._inner = real_zipfile(*args, **kwargs)

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._inner.__exit__(exc_type, exc, tb)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def write(self, filename, arcname=None, *args, **kwargs):
            if "w" in self._mode and str(arcname) == "dataset/sample.txt":
                return None
            return self._inner.write(filename, arcname=arcname, *args, **kwargs)

    def fixed_mkdtemp(prefix):
        export_dir = export_parent / f"{prefix}unit"
        export_dir.mkdir()
        return str(export_dir)

    monkeypatch.setattr(localinferenceapi.zipfile, "ZipFile", DroppingZipFile)
    monkeypatch.setattr(localinferenceapi.tempfile, "mkdtemp", fixed_mkdtemp)

    with pytest.raises(HTTPException) as exc:
        localinferenceapi.download_dataset_entry("demo_dataset")

    assert exc.value.status_code == 500
    assert exc.value.detail == "dataset_export_zip_missing:dataset/sample.txt"
    assert list(export_parent.iterdir()) == []
