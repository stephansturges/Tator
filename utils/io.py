from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST

logger = logging.getLogger(__name__)


def _ensure_directory(path: str) -> str:
    abs_path = os.path.abspath(path or ".")
    if not os.path.isdir(abs_path):
        try:
            Path(abs_path).mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"output_dir_missing:{abs_path}") from exc
    return abs_path


def _write_qwen_metadata(meta_path: Path, metadata: Dict[str, Any]) -> None:
    try:
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write Qwen metadata for %s: %s", meta_path.parent, exc)


def _load_json_metadata(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read metadata file %s: %s", path, exc)
    return None
