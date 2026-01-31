from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import HTTPException
from starlette.status import HTTP_404_NOT_FOUND


def _write_prepass_recipe_meta(recipe_dir: Path, payload: Dict[str, Any]) -> None:
    meta_path = recipe_dir / "prepass.meta.json"
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_prepass_recipe_meta(recipe_dir: Path) -> Dict[str, Any]:
    meta_path = recipe_dir / "prepass.meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="prepass_recipe_not_found")
    return json.loads(meta_path.read_text())
