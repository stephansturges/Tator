from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND, HTTP_500_INTERNAL_SERVER_ERROR


def _list_prompt_helper_presets_impl(*, presets_root: Path) -> List[Dict[str, Any]]:
    presets: List[Dict[str, Any]] = []
    for path in presets_root.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            presets.append(data)
        except Exception:
            continue
    presets.sort(key=lambda p: p.get("created_at", 0), reverse=True)
    return presets


def _load_prompt_helper_preset_impl(
    preset_id: str,
    *,
    presets_root: Path,
    path_is_within_root_fn,
) -> Dict[str, Any]:
    path = (presets_root / f"{preset_id}.json").resolve()
    if not path_is_within_root_fn(path, presets_root.resolve()) or not path.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="prompt_helper_preset_not_found")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"prompt_helper_preset_load_failed:{exc}") from exc


def _save_prompt_helper_preset_impl(
    label: str,
    dataset_id: str,
    prompts_by_class: Dict[int, List[str]],
    *,
    presets_root: Path,
    path_is_within_root_fn,
) -> Dict[str, Any]:
    preset_id = f"phset_{uuid.uuid4().hex[:8]}"
    created_at = time.time()
    payload = {
        "id": preset_id,
        "label": label or preset_id,
        "dataset_id": dataset_id,
        "created_at": created_at,
        "prompts_by_class": prompts_by_class,
    }
    path = (presets_root / f"{preset_id}.json").resolve()
    if not path_is_within_root_fn(path, presets_root.resolve()):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="prompt_helper_preset_path_invalid")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload
