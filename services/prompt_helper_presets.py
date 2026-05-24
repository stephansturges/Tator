"""Prompt helper preset utilities."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND, HTTP_500_INTERNAL_SERVER_ERROR


def _path_has_symlink_component(path: Path) -> bool:
    candidate = path if path.is_absolute() else path.absolute()
    checks = [candidate]
    checks.extend(candidate.parents)
    for component in checks:
        if component == component.parent:
            continue
        if component.is_symlink():
            return True
    return False


def _safe_presets_root(presets_root: Path, *, create: bool = False) -> Path | None:
    try:
        if _path_has_symlink_component(presets_root):
            return None
        if create:
            presets_root.mkdir(parents=True, exist_ok=True)
            if _path_has_symlink_component(presets_root):
                return None
        if presets_root.exists() and not presets_root.is_dir():
            return None
        if _path_has_symlink_component(presets_root):
            return None
        return presets_root.resolve(strict=False)
    except Exception:
        return None


def _list_prompt_helper_presets_impl(*, presets_root: Path) -> List[Dict[str, Any]]:
    presets: List[Dict[str, Any]] = []
    root = _safe_presets_root(presets_root)
    if root is None or not root.exists():
        return presets
    for path in root.glob("*.json"):
        try:
            if path.is_symlink():
                continue
            resolved = path.resolve(strict=True)
            if not _path_within_root(resolved, root) or not resolved.is_file():
                continue
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            presets.append(data)
        except Exception:
            continue
    presets.sort(key=lambda p: p.get("created_at", 0), reverse=True)
    return presets


def _path_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def _prepare_preset_output_file(path: Path, root: Path, *, path_is_within_root_fn) -> Path:
    try:
        if _path_has_symlink_component(root) or _path_has_symlink_component(path.parent):
            raise ValueError("prompt helper preset path has a symlink component")
        if path.is_symlink() or (path.exists() and path.is_dir()):
            raise ValueError("prompt helper preset target is not writable")
        target = path.resolve(strict=False)
        if not path_is_within_root_fn(target, root):
            raise ValueError("prompt helper preset target escapes root")
        return path
    except Exception as exc:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="prompt_helper_preset_path_invalid",
        ) from exc


def _load_prompt_helper_preset_impl(
    preset_id: str,
    *,
    presets_root: Path,
    path_is_within_root_fn,
) -> Dict[str, Any]:
    root = _safe_presets_root(presets_root)
    if root is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="prompt_helper_preset_not_found")
    raw_path = root / f"{preset_id}.json"
    if raw_path.is_symlink():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="prompt_helper_preset_not_found")
    path = raw_path.resolve(strict=False)
    if not path_is_within_root_fn(path, root) or not path.exists() or not path.is_file():
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
    root = _safe_presets_root(presets_root, create=True)
    if root is None:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="prompt_helper_preset_path_invalid")
    raw_path = root / f"{preset_id}.json"
    path = _prepare_preset_output_file(
        raw_path, root, path_is_within_root_fn=path_is_within_root_fn
    )
    tmp_path = root / f".{preset_id}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    tmp_path = _prepare_preset_output_file(
        tmp_path, root, path_is_within_root_fn=path_is_within_root_fn
    )
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"prompt_helper_preset_save_failed:{exc}",
        ) from exc
    finally:
        if tmp_path.exists() or tmp_path.is_symlink():
            tmp_path.unlink(missing_ok=True)
    return payload
