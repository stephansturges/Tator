from __future__ import annotations

import json
import stat
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest
from fastapi import HTTPException

from services.prepass_recipes import _import_prepass_recipe_from_zip_impl


def _write_zip(path: Path, entries: Dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel, content in entries.items():
            zf.writestr(rel, content)


def _write_recipe_meta(recipe_dir: Path, payload: Dict[str, Any]) -> None:
    recipe_dir.mkdir(parents=True, exist_ok=True)
    (recipe_dir / "prepass.meta.json").write_text(json.dumps(payload), encoding="utf-8")


def _unique_name(name: str) -> Tuple[str, Optional[str]]:
    return name, None


def _call_import(
    zip_path: Path,
    tmp_path: Path,
    *,
    max_zip_bytes: Optional[int] = None,
    max_extract_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    return _import_prepass_recipe_from_zip_impl(
        zip_path,
        prepass_recipe_meta="prepass.meta.json",
        prepass_schema_version=2,
        prepass_recipe_root=tmp_path / "prepass_recipes",
        prepass_tmp_root=tmp_path / "tmp_prepass",
        yolo_job_root=tmp_path / "yolo_runs",
        rfdetr_job_root=tmp_path / "rfdetr_runs",
        rfdetr_keep_files=None,
        qwen_job_root=tmp_path / "qwen_runs",
        qwen_metadata_filename="metadata.json",
        upload_root=tmp_path / "uploads",
        calibration_root=tmp_path / "calibration",
        read_labelmap_lines_fn=lambda _path: [],
        validate_manifest_fn=lambda _manifest, _extract: None,
        unique_name_fn=_unique_name,
        normalize_glossary_fn=lambda glossary: glossary,
        write_meta_fn=_write_recipe_meta,
        sanitize_run_id_fn=lambda value: value,
        max_zip_bytes=max_zip_bytes,
        max_extract_bytes=max_extract_bytes,
    )


def test_import_prepass_recipe_rejects_archive_path_traversal(tmp_path: Path) -> None:
    zip_path = tmp_path / "bad_recipe.zip"
    _write_zip(
        zip_path,
        {
            "manifest.json": json.dumps({"schema_version": 2, "assets": []}),
            "prepass.meta.json": json.dumps({"name": "bad", "config": {}}),
            "../escape.txt": "nope",
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        _call_import(zip_path, tmp_path)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "prepass_recipe_archive_path_traversal"


def test_import_prepass_recipe_accepts_basic_archive(tmp_path: Path) -> None:
    zip_path = tmp_path / "ok_recipe.zip"
    _write_zip(
        zip_path,
        {
            "manifest.json": json.dumps({"schema_version": 2, "assets": []}),
            "prepass.meta.json": json.dumps(
                {
                    "name": "ok_recipe",
                    "description": "import smoke",
                    "config": {"sam3_score_thr": 0.2},
                    "glossary": "",
                }
            ),
        },
    )

    imported = _call_import(zip_path, tmp_path)

    assert imported["name"] == "ok_recipe"
    assert imported["config"]["sam3_score_thr"] == 0.2


def test_import_prepass_recipe_rejects_symlink_member(tmp_path: Path) -> None:
    zip_path = tmp_path / "symlink_recipe.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps({"schema_version": 2, "assets": []}))
        zf.writestr("prepass.meta.json", json.dumps({"name": "bad", "config": {}}))
        info = zipfile.ZipInfo("models/link")
        info.create_system = 3  # Unix metadata semantics.
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "../outside")

    with pytest.raises(HTTPException) as exc_info:
        _call_import(zip_path, tmp_path)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "prepass_recipe_archive_symlink_unsupported"


def test_import_prepass_recipe_rejects_oversize_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "oversize_recipe.zip"
    _write_zip(
        zip_path,
        {
            "manifest.json": json.dumps({"schema_version": 2, "assets": []}),
            "prepass.meta.json": json.dumps({"name": "big", "config": {}}),
        },
    )
    zip_size = zip_path.stat().st_size
    assert zip_size > 0

    with pytest.raises(HTTPException) as exc_info:
        _call_import(zip_path, tmp_path, max_zip_bytes=max(1, zip_size - 1))

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == "prepass_recipe_import_zip_too_large"


def test_import_prepass_recipe_rejects_oversize_uncompressed_total(tmp_path: Path) -> None:
    zip_path = tmp_path / "oversize_uncompressed_recipe.zip"
    # Small compressed archive can still expand to large uncompressed payloads.
    _write_zip(
        zip_path,
        {
            "manifest.json": json.dumps({"schema_version": 2, "assets": []}),
            "prepass.meta.json": json.dumps({"name": "big_u", "config": {}}),
            "models/blob.bin": "x" * 2048,
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        _call_import(zip_path, tmp_path, max_extract_bytes=512)

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == "prepass_recipe_import_uncompressed_too_large"
