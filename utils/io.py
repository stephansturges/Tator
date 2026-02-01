from __future__ import annotations

import csv
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, List

from fastapi import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST

from utils.hashing import _stable_hash

logger = logging.getLogger(__name__)


def _ensure_directory(path: str) -> str:
    abs_path = os.path.abspath(path or ".")
    if not os.path.isdir(abs_path):
        try:
            Path(abs_path).mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"output_dir_missing:{abs_path}") from exc
    return abs_path


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



def _sanitize_yolo_run_id(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", (raw or "").strip()).strip("-_.")
    if cleaned:
        return cleaned
    return uuid.uuid4().hex[:12]


def _compute_dir_signature(root: Path, *, allowed_exts: Optional[set[str]] = None) -> str:
    """Return a stable signature for all files under ``root``."""
    entries: List[str] = []
    if not root.exists():
        return ""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if allowed_exts is not None and path.suffix.lower() not in allowed_exts:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = path.relative_to(root)
        entries.append(f"{rel}:{stat.st_mtime_ns}:{stat.st_size}")
    return _stable_hash(entries)


def _path_is_within_root_impl(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except Exception:
                continue
    return total
def _read_csv_last_row(path: Path) -> Optional[Dict[str, str]]:
    if not path.exists():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            last_row = None
            for row in reader:
                last_row = row
            return last_row
    except Exception:
        return None
