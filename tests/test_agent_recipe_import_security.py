from __future__ import annotations

import json
import stat
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi import HTTPException

from services.prepass_recipes import _import_agent_recipe_zip_bytes_impl


def _make_zip(entries: Dict[str, bytes]) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)
    return buf.getvalue()


def _call_import(zip_bytes: bytes, tmp_path: Path, **kwargs: Any):
    def _persist(
        dataset_id,
        class_id,
        class_name,
        label,
        recipe,
        *,
        crop_overrides=None,
        clip_head_overrides=None,
        meta_overrides=None,
    ):
        return {
            "id": "persisted_recipe",
            "dataset_id": dataset_id,
            "class_id": class_id,
            "class_name": class_name,
            "label": label,
            "recipe": recipe,
            "crops": dict(crop_overrides or {}),
            "clip_head": dict(clip_head_overrides or {}),
            "meta": dict(meta_overrides or {}),
        }

    max_entry_bytes = kwargs.pop("max_entry_bytes", 1024 * 1024)
    return _import_agent_recipe_zip_bytes_impl(
        zip_bytes,
        recipes_root=tmp_path / "recipes",
        max_json_bytes=1024 * 1024,
        max_clip_head_bytes=1024 * 1024,
        max_crops=100,
        max_crop_bytes=1024 * 1024,
        max_entry_bytes=max_entry_bytes,
        persist_recipe_fn=_persist,
        **kwargs,
    )


def test_agent_recipe_import_basic_success(tmp_path: Path) -> None:
    recipe = {"id": "r1", "label": "demo", "dataset_id": "ds1"}
    payload = _make_zip(
        {
            "recipe.json": json.dumps(recipe).encode("utf-8"),
            "crops/sample.png": b"\x89PNG\r\n\x1a\n",
        }
    )

    old_id, persisted = _call_import(payload, tmp_path)

    assert old_id == "r1"
    assert persisted["id"] == "persisted_recipe"
    assert persisted["label"] == "demo"
    assert "crops/sample.png" in persisted["crops"]


def test_agent_recipe_import_rejects_symlink_entry(tmp_path: Path) -> None:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("recipe.json", json.dumps({"id": "r1", "label": "demo"}))
        info = zipfile.ZipInfo("crops/evil.png")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, b"../../outside")

    with pytest.raises(HTTPException) as exc_info:
        _call_import(buf.getvalue(), tmp_path)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "agent_recipe_import_symlink_unsupported"


def test_agent_recipe_import_rejects_oversize_entry(tmp_path: Path) -> None:
    payload = _make_zip(
        {
            "recipe.json": json.dumps({"id": "r1", "label": "demo"}).encode("utf-8"),
            "crops/huge.png": b"x" * 4096,
        }
    )

    with pytest.raises(HTTPException) as exc_info:
        _call_import(payload, tmp_path, max_entry_bytes=256)

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == "agent_recipe_import_entry_too_large"
