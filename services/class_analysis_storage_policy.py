"""Persistent generated-data budget and recoverable-session inventory."""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping


STORAGE_POLICY_SCHEMA = "class-analysis-storage-policy-v1"
DEFAULT_GENERATED_DATA_BUDGET_BYTES = 50 * 1024**3
MIN_GENERATED_DATA_BUDGET_BYTES = 1 * 1024**3
MAX_GENERATED_DATA_BUDGET_BYTES = 4096 * 1024**3
_SESSION_PATTERN = re.compile(r"ca_[A-Za-z0-9]+")


class StoragePolicyError(RuntimeError):
    """Invalid or unsafe generated-data policy state."""


def normalize_generated_data_budget(value: Any) -> int:
    if isinstance(value, bool):
        raise StoragePolicyError("class_analysis_storage_budget_invalid")
    try:
        budget = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StoragePolicyError("class_analysis_storage_budget_invalid") from exc
    if not MIN_GENERATED_DATA_BUDGET_BYTES <= budget <= MAX_GENERATED_DATA_BUDGET_BYTES:
        raise StoragePolicyError("class_analysis_storage_budget_out_of_range")
    return budget


def read_generated_data_budget(path: Path | str, *, default_bytes: int) -> int:
    policy_path = Path(path)
    default = normalize_generated_data_budget(default_bytes)
    if not policy_path.exists():
        return default
    if policy_path.is_symlink() or not policy_path.is_file():
        raise StoragePolicyError("class_analysis_storage_policy_path_unsafe")
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StoragePolicyError("class_analysis_storage_policy_invalid") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != STORAGE_POLICY_SCHEMA:
        raise StoragePolicyError("class_analysis_storage_policy_invalid")
    return normalize_generated_data_budget(payload.get("max_bytes"))


def write_generated_data_budget(path: Path | str, max_bytes: Any) -> dict[str, Any]:
    policy_path = Path(path)
    budget = normalize_generated_data_budget(max_bytes)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    if policy_path.parent.is_symlink() or policy_path.is_symlink():
        raise StoragePolicyError("class_analysis_storage_policy_path_unsafe")
    payload = {"schema": STORAGE_POLICY_SCHEMA, "max_bytes": budget}
    temporary = policy_path.with_name(f".{policy_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, policy_path)
        descriptor = os.open(str(policy_path.parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def _tree_bytes(path: Path) -> tuple[int, int]:
    total = 0
    files = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += max(0, int(entry.stat(follow_symlinks=False).st_size))
                    files += 1
            except OSError:
                continue
    return total, files


def recoverable_session_inventory(
    root: Path | str,
    *,
    pinned_job_ids: Iterable[str] = (),
) -> dict[str, Any]:
    analysis_root = Path(root)
    pins = {str(job_id or "").strip() for job_id in pinned_job_ids if str(job_id or "").strip()}
    rows: list[dict[str, Any]] = []
    if analysis_root.is_dir() and not analysis_root.is_symlink():
        for child in analysis_root.iterdir():
            if (
                child.is_symlink()
                or not child.is_dir()
                or _SESSION_PATTERN.fullmatch(child.name) is None
            ):
                continue
            size, files = _tree_bytes(child)
            try:
                modified_at = float(child.stat().st_mtime)
            except OSError:
                modified_at = 0.0
            rows.append(
                {
                    "job_id": child.name,
                    "path": str(child),
                    "bytes": size,
                    "files": files,
                    "modified_at": modified_at,
                    "pinned": child.name in pins,
                }
            )
    rows.sort(key=lambda row: (float(row["modified_at"]), str(row["job_id"])))
    total = sum(int(row["bytes"]) for row in rows)
    pinned = sum(int(row["bytes"]) for row in rows if row["pinned"])
    return {
        "sessions": rows,
        "session_count": len(rows),
        "total_bytes": total,
        "pinned_bytes": pinned,
        "evictable_bytes": max(0, total - pinned),
    }


def choose_unpinned_session_evictions(
    sessions: Iterable[Mapping[str, Any]],
    *,
    cache_bytes: int,
    max_bytes: int,
) -> list[dict[str, Any]]:
    budget = normalize_generated_data_budget(max_bytes)
    rows = [dict(row) for row in sessions]
    remaining = max(0, int(cache_bytes)) + sum(max(0, int(row.get("bytes") or 0)) for row in rows)
    victims: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (float(item.get("modified_at") or 0), str(item.get("job_id") or ""))):
        if remaining <= budget:
            break
        if bool(row.get("pinned")):
            continue
        victims.append(row)
        remaining -= max(0, int(row.get("bytes") or 0))
    return victims
