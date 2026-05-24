from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi import HTTPException

from services.prepass_recipes import (
    _copy2_if_different,
    _copy_tree_filtered_impl,
    _ensure_recipe_zip_impl,
    _prepare_output_file,
)


def test_ensure_recipe_zip_rebuilds_corrupt_existing_zip(tmp_path: Path) -> None:
    recipes_root = tmp_path / "recipes"
    recipes_root.mkdir(parents=True, exist_ok=True)
    corrupt_zip = recipes_root / "r1.zip"
    corrupt_zip.write_bytes(b"corrupt")

    recipe: Dict[str, Any] = {"id": "r1", "label": "demo", "config": {"x": 1}}
    zip_path = _ensure_recipe_zip_impl(recipe, recipes_root=recipes_root)

    assert zip_path == corrupt_zip
    with zipfile.ZipFile(zip_path, "r") as zf:
        assert "recipe.json" in zf.namelist()
        payload = json.loads(zf.read("recipe.json").decode("utf-8"))
    assert payload["id"] == "r1"


def test_ensure_recipe_zip_skips_symlink_artifact_escape(tmp_path: Path) -> None:
    recipes_root = tmp_path / "recipes"
    recipe_dir = recipes_root / "r2"
    crops_dir = recipe_dir / "crops"
    clip_head_dir = recipe_dir / "clip_head"
    crops_dir.mkdir(parents=True, exist_ok=True)
    clip_head_dir.mkdir(parents=True, exist_ok=True)
    outside_crop = tmp_path / "outside.png"
    outside_crop.write_bytes(b"not-a-crop")
    outside_meta = tmp_path / "outside_meta.json"
    outside_meta.write_text("{}", encoding="utf-8")
    try:
        (crops_dir / "escape.png").symlink_to(outside_crop)
        (clip_head_dir / "meta.json").symlink_to(outside_meta)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    zip_path = _ensure_recipe_zip_impl(
        {"id": "r2", "label": "demo", "config": {"x": 1}},
        recipes_root=recipes_root,
    )

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
    assert "recipe.json" in names
    assert "crops/escape.png" not in names
    assert "clip_head/meta.json" not in names


def test_ensure_recipe_zip_replaces_symlinked_existing_zip_without_target_write(
    tmp_path: Path,
) -> None:
    recipes_root = tmp_path / "recipes"
    recipes_root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.zip"
    outside.write_bytes(b"external")
    zip_path = recipes_root / "r3.zip"
    try:
        zip_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    result = _ensure_recipe_zip_impl(
        {"id": "r3", "label": "demo", "config": {"x": 1}},
        recipes_root=recipes_root,
    )

    assert result == zip_path
    assert not zip_path.is_symlink()
    assert outside.read_bytes() == b"external"
    with zipfile.ZipFile(zip_path, "r") as zf:
        assert "recipe.json" in set(zf.namelist())


def test_ensure_recipe_zip_rejects_symlinked_recipe_root_without_write(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_recipes"
    outside.mkdir()
    recipes_root = tmp_path / "recipes"
    try:
        recipes_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(HTTPException) as exc_info:
        _ensure_recipe_zip_impl(
            {"id": "r4", "label": "demo", "config": {"x": 1}},
            recipes_root=recipes_root,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "agent_recipe_path_invalid"
    assert list(outside.iterdir()) == []


def test_ensure_recipe_zip_rejects_symlinked_recipe_parent_without_write(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    recipes_parent = tmp_path / "linked_parent"
    try:
        recipes_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(HTTPException) as exc_info:
        _ensure_recipe_zip_impl(
            {"id": "r5", "label": "demo", "config": {"x": 1}},
            recipes_root=recipes_parent / "recipes",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "agent_recipe_path_invalid"
    assert list(outside.iterdir()) == []


def test_ensure_recipe_zip_rejects_nested_symlinked_recipe_parent_without_write(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    recipes_parent = tmp_path / "linked_parent"
    try:
        recipes_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(HTTPException) as exc_info:
        _ensure_recipe_zip_impl(
            {"id": "r6", "label": "demo", "config": {"x": 1}},
            recipes_root=recipes_parent / "nested" / "recipes",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "agent_recipe_path_invalid"
    assert list(outside.iterdir()) == []


def test_prepass_prepare_output_file_rejects_nested_symlinked_parent_before_mkdir(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(HTTPException) as exc_info:
        _prepare_output_file(linked_parent / "nested" / "recipes" / "r1.zip")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "prepass_recipe_path_invalid"
    assert list(outside.iterdir()) == []


def test_prepass_copy2_if_different_replaces_symlink_to_source(tmp_path: Path) -> None:
    src = tmp_path / "source.bin"
    src.write_text("source", encoding="utf-8")
    dest = tmp_path / "dest.bin"
    try:
        dest.symlink_to(src)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _copy2_if_different(src, dest)

    assert not dest.is_symlink()
    assert dest.read_text(encoding="utf-8") == "source"


def test_copy_tree_filtered_skips_symlink_escape(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.mkdir()
    (src / "safe.txt").write_text("ok", encoding="utf-8")
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret", encoding="utf-8")
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    (outside_dir / "nested.txt").write_text("secret", encoding="utf-8")
    try:
        (src / "escape.txt").symlink_to(outside_file)
        (src / "escape_dir").symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    copied = _copy_tree_filtered_impl(src, dest)

    assert (dest / "safe.txt").read_text(encoding="utf-8") == "ok"
    assert not (dest / "escape.txt").exists()
    assert not (dest / "escape_dir").exists()
    assert [entry["path"] for entry in copied] == ["dest/safe.txt"]


def test_copy_tree_filtered_rejects_nested_symlinked_destination_parent_before_mkdir(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "safe.txt").write_text("ok", encoding="utf-8")
    outside = tmp_path / "outside_parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked_parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(HTTPException) as exc_info:
        _copy_tree_filtered_impl(src, linked_parent / "nested" / "dest")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "prepass_recipe_path_invalid"
    assert list(outside.iterdir()) == []
