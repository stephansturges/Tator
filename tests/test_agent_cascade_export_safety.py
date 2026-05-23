from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi import HTTPException

from services.agent_cascades import (
    _delete_agent_cascade_impl,
    _ensure_cascade_zip_impl,
    _list_agent_cascades_impl,
    _load_agent_cascade_impl,
    _persist_agent_cascade_impl,
)


def _within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _write_recipe_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("recipe.json", json.dumps({"id": "r1"}))


def test_ensure_cascade_zip_skips_unsafe_classifier_archive_path(tmp_path: Path) -> None:
    cascades_root = tmp_path / "cascades"
    recipes_root = tmp_path / "recipes"
    classifiers_root = tmp_path / "classifiers"
    cascades_root.mkdir(parents=True, exist_ok=True)
    recipes_root.mkdir(parents=True, exist_ok=True)
    classifiers_root.mkdir(parents=True, exist_ok=True)

    recipe_zip = recipes_root / "r1.zip"
    _write_recipe_zip(recipe_zip)
    classifier_path = classifiers_root / "safe.pkl"
    classifier_path.write_bytes(b"classifier")
    classifier_path.with_suffix(".pkl.meta.pkl").write_bytes(b"meta")

    cascade: Dict[str, Any] = {
        "id": "ac_test",
        "label": "demo",
        "steps": [{"recipe_id": "r1", "extra_clip_classifier_path": "../evil.pkl"}],
        "dedupe": {},
    }
    zip_path = _ensure_cascade_zip_impl(
        cascade,
        cascades_root=cascades_root,
        recipes_root=recipes_root,
        classifiers_root=classifiers_root,
        path_is_within_root_fn=_within_root,
        ensure_recipe_zip_fn=lambda _recipe: recipe_zip,
        load_recipe_fn=lambda _rid: {"id": "r1"},
        resolve_classifier_fn=lambda _rel: str(classifier_path),
    )

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    assert "cascade.json" in names
    assert "recipes/r1.zip" in names
    # Unsafe classifier relpaths must not be emitted as archive members.
    assert not any(name.startswith("classifiers/") for name in names)
    assert all(".." not in Path(name).parts for name in names)


def test_ensure_cascade_zip_skips_unsafe_recipe_archive_path(tmp_path: Path) -> None:
    cascades_root = tmp_path / "cascades"
    recipes_root = tmp_path / "recipes"
    classifiers_root = tmp_path / "classifiers"
    cascades_root.mkdir(parents=True, exist_ok=True)
    recipes_root.mkdir(parents=True, exist_ok=True)
    classifiers_root.mkdir(parents=True, exist_ok=True)

    recipe_zip = recipes_root / "r1.zip"
    _write_recipe_zip(recipe_zip)

    cascade: Dict[str, Any] = {
        "id": "ac_test_2",
        "label": "demo",
        "steps": [{"recipe_id": "../r1"}],
        "dedupe": {},
    }
    zip_path = _ensure_cascade_zip_impl(
        cascade,
        cascades_root=cascades_root,
        recipes_root=recipes_root,
        classifiers_root=classifiers_root,
        path_is_within_root_fn=_within_root,
        ensure_recipe_zip_fn=lambda _recipe: recipe_zip,
        load_recipe_fn=lambda _rid: {"id": "r1"},
        resolve_classifier_fn=lambda _rel: None,
    )

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    assert "cascade.json" in names
    # Unsafe recipe ids must not be emitted into zip paths.
    assert not any(name.startswith("recipes/") for name in names)
    assert all(".." not in Path(name).parts for name in names)


def test_ensure_cascade_zip_rejects_symlinked_cascade_root_without_write(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_cascades"
    outside.mkdir()
    cascades_root = tmp_path / "cascades"
    recipes_root = tmp_path / "recipes"
    classifiers_root = tmp_path / "classifiers"
    recipes_root.mkdir(parents=True, exist_ok=True)
    classifiers_root.mkdir(parents=True, exist_ok=True)
    try:
        cascades_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    cascade: Dict[str, Any] = {
        "id": "ac_test",
        "label": "demo",
        "steps": [{"recipe_id": "r1"}],
        "dedupe": {},
    }

    with pytest.raises(HTTPException) as exc_info:
        _ensure_cascade_zip_impl(
            cascade,
            cascades_root=cascades_root,
            recipes_root=recipes_root,
            classifiers_root=classifiers_root,
            path_is_within_root_fn=_within_root,
            ensure_recipe_zip_fn=lambda _recipe: recipes_root / "r1.zip",
            load_recipe_fn=lambda _rid: {"id": "r1"},
            resolve_classifier_fn=lambda _rel: None,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "agent_cascade_path_invalid"
    assert list(outside.iterdir()) == []


def test_ensure_cascade_zip_skips_symlink_classifier_meta_escape(tmp_path: Path) -> None:
    cascades_root = tmp_path / "cascades"
    recipes_root = tmp_path / "recipes"
    classifiers_root = tmp_path / "classifiers"
    cascades_root.mkdir(parents=True, exist_ok=True)
    recipes_root.mkdir(parents=True, exist_ok=True)
    classifiers_root.mkdir(parents=True, exist_ok=True)

    recipe_zip = recipes_root / "r1.zip"
    _write_recipe_zip(recipe_zip)
    classifier_path = classifiers_root / "safe.pkl"
    classifier_path.write_bytes(b"classifier")
    outside_meta = tmp_path / "outside.meta.pkl"
    outside_meta.write_bytes(b"secret")
    try:
        classifier_path.with_suffix(".pkl.meta.pkl").symlink_to(outside_meta)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    cascade: Dict[str, Any] = {
        "id": "ac_meta_escape",
        "label": "demo",
        "steps": [{"recipe_id": "r1", "extra_clip_classifier_path": "safe.pkl"}],
        "dedupe": {},
    }
    zip_path = _ensure_cascade_zip_impl(
        cascade,
        cascades_root=cascades_root,
        recipes_root=recipes_root,
        classifiers_root=classifiers_root,
        path_is_within_root_fn=_within_root,
        ensure_recipe_zip_fn=lambda _recipe: recipe_zip,
        load_recipe_fn=lambda _rid: {"id": "r1"},
        resolve_classifier_fn=lambda _rel: str(classifier_path),
    )

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
    assert "classifiers/safe.pkl" in names
    assert "classifiers/safe.pkl.meta.pkl" not in names


def test_ensure_cascade_zip_rebuilds_corrupt_existing_zip(tmp_path: Path) -> None:
    cascades_root = tmp_path / "cascades"
    recipes_root = tmp_path / "recipes"
    classifiers_root = tmp_path / "classifiers"
    cascades_root.mkdir(parents=True, exist_ok=True)
    recipes_root.mkdir(parents=True, exist_ok=True)
    classifiers_root.mkdir(parents=True, exist_ok=True)

    corrupt_zip = cascades_root / "ac_rebuild.zip"
    corrupt_zip.write_bytes(b"not-a-zip")

    cascade: Dict[str, Any] = {"id": "ac_rebuild", "label": "demo", "steps": [], "dedupe": {}}
    zip_path = _ensure_cascade_zip_impl(
        cascade,
        cascades_root=cascades_root,
        recipes_root=recipes_root,
        classifiers_root=classifiers_root,
        path_is_within_root_fn=_within_root,
        ensure_recipe_zip_fn=lambda _recipe: recipes_root / "noop.zip",
        load_recipe_fn=lambda _rid: {"id": _rid},
        resolve_classifier_fn=lambda _rel: None,
    )

    assert zip_path == corrupt_zip
    with zipfile.ZipFile(zip_path, "r") as zf:
        assert "cascade.json" in zf.namelist()


def test_delete_agent_cascade_unlinks_symlinked_zip_without_touching_target(
    tmp_path: Path,
) -> None:
    cascades_root = tmp_path / "cascades"
    outside = tmp_path / "outside.zip"
    cascades_root.mkdir()
    outside.write_bytes(b"keep")
    link_path = cascades_root / "ac_link.zip"
    try:
        link_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    _delete_agent_cascade_impl(
        "ac_link",
        cascades_root=cascades_root,
        path_is_within_root_fn=_within_root,
    )

    assert not link_path.is_symlink()
    assert outside.read_bytes() == b"keep"


def test_list_agent_cascades_skips_symlinked_json_escape(tmp_path: Path) -> None:
    cascades_root = tmp_path / "cascades"
    outside = tmp_path / "outside.json"
    cascades_root.mkdir()
    outside.write_text(json.dumps({"id": "outside", "created_at": 1}), encoding="utf-8")
    try:
        (cascades_root / "outside.json").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    cascades = _list_agent_cascades_impl(cascades_root=cascades_root)

    assert cascades == []


def test_ensure_cascade_zip_replaces_symlinked_existing_zip(tmp_path: Path) -> None:
    cascades_root = tmp_path / "cascades"
    recipes_root = tmp_path / "recipes"
    classifiers_root = tmp_path / "classifiers"
    outside = tmp_path / "outside.zip"
    cascades_root.mkdir()
    recipes_root.mkdir()
    classifiers_root.mkdir()
    outside.write_bytes(b"keep")
    link_path = cascades_root / "ac_symlink.zip"
    try:
        link_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    zip_path = _ensure_cascade_zip_impl(
        {"id": "ac_symlink", "label": "demo", "steps": [], "dedupe": {}},
        cascades_root=cascades_root,
        recipes_root=recipes_root,
        classifiers_root=classifiers_root,
        path_is_within_root_fn=_within_root,
        ensure_recipe_zip_fn=lambda _recipe: recipes_root / "noop.zip",
        load_recipe_fn=lambda _rid: {"id": _rid},
        resolve_classifier_fn=lambda _rel: None,
    )

    assert zip_path == link_path
    assert outside.read_bytes() == b"keep"
    with zipfile.ZipFile(zip_path, "r") as zf:
        assert "cascade.json" in zf.namelist()


def test_persist_agent_cascade_replaces_symlink_targets_without_target_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cascades_root = tmp_path / "cascades"
    cascades_root.mkdir()
    cascade_id = "ac_deadbeef"
    json_path = cascades_root / f"{cascade_id}.json"
    outside_tmp = tmp_path / "outside_tmp.json"
    outside_final = tmp_path / "outside_final.json"
    outside_tmp.write_text("external tmp", encoding="utf-8")
    outside_final.write_text("external final", encoding="utf-8")
    tmp_link = json_path.with_suffix(json_path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp_link.symlink_to(outside_tmp)
        json_path.symlink_to(outside_final)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    class FixedUUID:
        hex = "deadbeef000000000000000000000000"

    monkeypatch.setattr("services.agent_cascades.uuid.uuid4", lambda: FixedUUID())

    record = _persist_agent_cascade_impl(
        "demo",
        {"steps": [{"recipe_id": "r1"}], "dedupe": {}},
        cascades_root=cascades_root,
        path_is_within_root_fn=_within_root,
    )

    assert record["id"] == cascade_id
    assert not tmp_link.exists()
    assert not json_path.is_symlink()
    assert json.loads(json_path.read_text(encoding="utf-8"))["id"] == cascade_id
    assert outside_tmp.read_text(encoding="utf-8") == "external tmp"
    assert outside_final.read_text(encoding="utf-8") == "external final"


def test_load_agent_cascade_rejects_symlinked_json_escape(tmp_path: Path) -> None:
    cascades_root = tmp_path / "cascades"
    cascades_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"id": "ac_link", "steps": []}), encoding="utf-8")
    try:
        (cascades_root / "ac_link.json").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(HTTPException) as exc_info:
        _load_agent_cascade_impl(
            "ac_link",
            cascades_root=cascades_root,
            path_is_within_root_fn=_within_root,
        )

    assert getattr(exc_info.value, "detail", None) == "agent_cascade_not_found"
