from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict

from services.agent_cascades import _ensure_cascade_zip_impl


def _within_root(path: Path, root: Path) -> bool:
    return str(path.resolve()).startswith(str(root.resolve()))


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
