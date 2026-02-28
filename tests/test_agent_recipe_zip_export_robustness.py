from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict

from services.prepass_recipes import _ensure_recipe_zip_impl


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
