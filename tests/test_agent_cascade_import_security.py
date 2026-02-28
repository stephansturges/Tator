from __future__ import annotations

import json
import stat
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi import HTTPException

from services.agent_cascades import _import_agent_cascade_zip_bytes_impl


def _make_zip(entries: Dict[str, bytes]) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)
    return buf.getvalue()


def _call_import(zip_bytes: bytes, tmp_path: Path, **kwargs: Any) -> Dict[str, Any]:
    def _import_recipe(_payload: bytes) -> tuple[str, Dict[str, Any]]:
        return "r1", {"id": "recipe_new"}

    def _import_recipe_file(_path: Path) -> tuple[str, Dict[str, Any]]:
        return "r1", {"id": "recipe_new"}

    def _persist(label: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"label": label, **payload}

    max_entry_bytes = kwargs.pop("max_entry_bytes", 1024 * 1024)
    import_recipe_fn = kwargs.pop("import_recipe_fn", _import_recipe)
    import_recipe_file_fn = kwargs.pop("import_recipe_file_fn", None)
    return _import_agent_cascade_zip_bytes_impl(
        zip_bytes,
        cascades_root=tmp_path / "cascades",
        classifiers_root=tmp_path / "classifiers",
        max_json_bytes=1024 * 1024,
        max_entry_bytes=max_entry_bytes,
        classifier_allowed_exts=[".pkl", ".joblib"],
        path_is_within_root_fn=lambda path, root: str(path.resolve()).startswith(str(root.resolve())),
        import_recipe_fn=import_recipe_fn,
        import_recipe_file_fn=import_recipe_file_fn,
        persist_cascade_fn=_persist,
        **kwargs,
    )


def test_agent_cascade_import_basic_success(tmp_path: Path) -> None:
    cascade = {
        "label": "demo",
        "steps": [{"recipe_id": "r1", "enabled": True}],
        "dedupe": {},
    }
    payload = _make_zip(
        {
            "cascade.json": json.dumps(cascade).encode("utf-8"),
            "recipes/r1.zip": b"recipe-bytes",
        }
    )

    out = _call_import(payload, tmp_path)

    assert out["label"] == "demo"
    assert out["steps"][0]["recipe_id"] == "recipe_new"


def test_agent_cascade_import_rejects_symlink_entries(tmp_path: Path) -> None:
    cascade = {"label": "demo", "steps": [{"recipe_id": "r1"}], "dedupe": {}}
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("cascade.json", json.dumps(cascade))
        zf.writestr("recipes/r1.zip", b"recipe-bytes")
        info = zipfile.ZipInfo("classifiers/evil.pkl")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, b"../../outside")

    with pytest.raises(HTTPException) as exc_info:
        _call_import(buf.getvalue(), tmp_path)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "agent_cascade_import_symlink_unsupported"


def test_agent_cascade_import_rejects_oversize_entries(tmp_path: Path) -> None:
    cascade = {
        "label": "demo",
        "steps": [{"recipe_id": "r1"}],
        "dedupe": {},
        "padding": "x" * 4000,
    }
    payload = _make_zip(
        {
            "cascade.json": json.dumps(cascade).encode("utf-8"),
            "recipes/r1.zip": b"recipe-bytes",
        }
    )

    with pytest.raises(HTTPException) as exc_info:
        _call_import(payload, tmp_path, max_entry_bytes=256)

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == "agent_cascade_import_entry_too_large"


def test_agent_cascade_import_uses_file_recipe_importer_when_available(tmp_path: Path) -> None:
    cascade = {"label": "demo", "steps": [{"recipe_id": "r1"}], "dedupe": {}}
    payload = _make_zip(
        {
            "cascade.json": json.dumps(cascade).encode("utf-8"),
            "recipes/r1.zip": b"recipe-bytes",
        }
    )
    seen: Dict[str, Any] = {"called": False}

    def _import_recipe_file(path: Path) -> tuple[str, Dict[str, Any]]:
        seen["called"] = path.exists() and path.is_file()
        return "r1", {"id": "recipe_new"}

    def _import_recipe_bytes(_payload: bytes) -> tuple[str, Dict[str, Any]]:
        raise AssertionError("bytes importer should not be used when file importer is provided")

    out = _call_import(
        payload,
        tmp_path,
        import_recipe_fn=_import_recipe_bytes,
        import_recipe_file_fn=_import_recipe_file,
    )

    assert seen["called"] is True
    assert out["steps"][0]["recipe_id"] == "recipe_new"


def test_agent_cascade_import_rejects_oversize_nested_recipe_zip(tmp_path: Path) -> None:
    cascade = {"label": "demo", "steps": [{"recipe_id": "r1"}], "dedupe": {}}
    payload = _make_zip(
        {
            "cascade.json": json.dumps(cascade).encode("utf-8"),
            "recipes/r1.zip": b"x" * 2048,
        }
    )

    with pytest.raises(HTTPException) as exc_info:
        _call_import(payload, tmp_path, max_entry_bytes=8192, max_recipe_zip_bytes=512)

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == "agent_cascade_import_recipe_too_large"
