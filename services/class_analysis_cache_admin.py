"""Safe inventory and targeted purge helpers for class-analysis caches."""

from __future__ import annotations

import os
import stat
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

CACHE_CATEGORIES = (
    "image_packs",
    "resume_embeddings",
    "patch_reference_banks",
)
PURGEABLE_CATEGORIES = (
    "image_packs",
    "resume_embeddings",
)

_ADMIN_LOCK = threading.RLock()


class CacheAdminError(RuntimeError):
    """Base class for cache administration failures."""


class CacheRootUnsafeError(CacheAdminError):
    """Raised when the configured cache root is unsafe to inspect."""


class CacheBusyError(CacheAdminError):
    """Raised when active jobs make cache deletion unsafe."""

    def __init__(self, active_users: Iterable[Mapping[str, Any]]):
        self.active_users = [dict(row) for row in active_users]
        super().__init__("class_analysis_cache_busy")


def _absolute_without_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _safe_cache_root(root: Path | str, *, create: bool) -> Path:
    candidate = _absolute_without_resolution(Path(root))
    if candidate == Path(candidate.anchor) or len(candidate.parts) < 3:
        raise CacheRootUnsafeError("class_analysis_cache_root_too_broad")

    cursor = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        cursor = cursor / part
        try:
            if cursor.is_symlink():
                raise CacheRootUnsafeError(
                    "class_analysis_cache_root_contains_symlink"
                )
        except OSError as exc:
            raise CacheRootUnsafeError(
                "class_analysis_cache_root_unreadable"
            ) from exc

    if create:
        candidate.mkdir(parents=True, exist_ok=True)
    if not candidate.is_dir():
        raise CacheRootUnsafeError("class_analysis_cache_root_not_directory")
    return candidate


def _scan_tree(path: Path) -> dict[str, int]:
    totals = {"bytes": 0, "files": 0, "directories": 0, "symlinks": 0}
    if not path.exists():
        return totals
    if path.is_symlink():
        totals["symlinks"] = 1
        return totals
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            mode = entry_stat.st_mode
            if stat.S_ISLNK(mode):
                totals["symlinks"] += 1
            elif stat.S_ISDIR(mode):
                totals["directories"] += 1
                stack.append(Path(entry.path))
            elif stat.S_ISREG(mode):
                totals["files"] += 1
                totals["bytes"] += max(0, int(entry_stat.st_size))
    return totals


def _scan_other(root: Path) -> dict[str, int]:
    totals = {"bytes": 0, "files": 0, "directories": 0, "symlinks": 0}
    known = set(CACHE_CATEGORIES)
    try:
        entries = list(os.scandir(root))
    except OSError:
        return totals
    for entry in entries:
        if entry.name in known or entry.name.startswith(".purge-"):
            continue
        entry_path = Path(entry.path)
        try:
            entry_stat = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        if stat.S_ISLNK(entry_stat.st_mode):
            totals["symlinks"] += 1
        elif stat.S_ISDIR(entry_stat.st_mode):
            totals["directories"] += 1
            scanned = _scan_tree(entry_path)
            for key in totals:
                totals[key] += int(scanned[key])
        elif stat.S_ISREG(entry_stat.st_mode):
            totals["files"] += 1
            totals["bytes"] += max(0, int(entry_stat.st_size))
    return totals


def cache_inventory(
    root: Path | str,
    *,
    max_bytes: int = 0,
    active_users: Optional[Iterable[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    with _ADMIN_LOCK:
        safe_root = _safe_cache_root(root, create=True)
        categories: dict[str, dict[str, Any]] = {}
        total_bytes = 0
        total_files = 0
        total_symlinks = 0
        for name in CACHE_CATEGORIES:
            stats = _scan_tree(safe_root / name)
            row = {
                **stats,
                "purgeable": name in PURGEABLE_CATEGORIES,
            }
            categories[name] = row
            total_bytes += int(row["bytes"])
            total_files += int(row["files"])
            total_symlinks += int(row["symlinks"])
        other = _scan_other(safe_root)
        categories["other"] = {**other, "purgeable": False}
        total_bytes += int(other["bytes"])
        total_files += int(other["files"])
        total_symlinks += int(other["symlinks"])
        budget = max(0, int(max_bytes))
        purgeable_bytes = sum(
            int(categories[name]["bytes"])
            for name in PURGEABLE_CATEGORIES
        )
        protected_bytes = max(0, total_bytes - purgeable_bytes)
        return {
            "status": "ready",
            "root": str(safe_root),
            "max_bytes": budget,
            "total_bytes": total_bytes,
            "managed_bytes": purgeable_bytes,
            "purgeable_bytes": purgeable_bytes,
            "protected_bytes": protected_bytes,
            "over_budget_bytes": (
                max(0, purgeable_bytes - budget) if budget > 0 else 0
            ),
            "total_files": total_files,
            "symlink_entries": total_symlinks,
            "usage_fraction": (
                purgeable_bytes / budget if budget > 0 else None
            ),
            "budget_scope": list(PURGEABLE_CATEGORIES),
            "categories": categories,
            "purgeable_categories": list(PURGEABLE_CATEGORIES),
            "active_users": [dict(row) for row in (active_users or [])],
        }


def _remove_tree_without_following_symlinks(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
        return
    if not path.exists():
        return
    with os.scandir(path) as entries:
        for entry in entries:
            child = Path(entry.path)
            entry_stat = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(entry_stat.st_mode):
                _remove_tree_without_following_symlinks(child)
            else:
                child.unlink()
    path.rmdir()


def purge_cache(
    root: Path | str,
    *,
    categories: Optional[Iterable[str]] = None,
    max_bytes: int = 0,
    active_users_fn: Optional[
        Callable[[], Iterable[Mapping[str, Any]]]
    ] = None,
) -> dict[str, Any]:
    requested = tuple(
        dict.fromkeys(
            str(name or "").strip()
            for name in (categories or PURGEABLE_CATEGORIES)
            if str(name or "").strip()
        )
    )
    if not requested or any(
        name not in PURGEABLE_CATEGORIES for name in requested
    ):
        raise CacheAdminError(
            "class_analysis_cache_category_not_purgeable"
        )

    with _ADMIN_LOCK:
        active_users = (
            list(active_users_fn() or []) if active_users_fn else []
        )
        if active_users:
            raise CacheBusyError(active_users)
        safe_root = _safe_cache_root(root, create=True)
        before = cache_inventory(safe_root, max_bytes=max_bytes)
        quarantined: list[Path] = []
        for name in requested:
            target = safe_root / name
            if target.is_symlink():
                raise CacheRootUnsafeError(
                    f"class_analysis_cache_category_symlink:{name}"
                )
            if target.exists() and not target.is_dir():
                raise CacheRootUnsafeError(
                    f"class_analysis_cache_category_not_directory:{name}"
                )
            if target.exists():
                quarantine = (
                    safe_root / f".purge-{name}-{uuid.uuid4().hex}"
                )
                os.replace(target, quarantine)
                quarantined.append(quarantine)
            target.mkdir(mode=0o755, exist_ok=True)
        for quarantine in quarantined:
            _remove_tree_without_following_symlinks(quarantine)
        after = cache_inventory(safe_root, max_bytes=max_bytes)
        before_bytes = sum(
            int(before["categories"][name]["bytes"])
            for name in requested
        )
        after_bytes = sum(
            int(after["categories"][name]["bytes"])
            for name in requested
        )
        return {
            "status": "cleared",
            "cleared_categories": list(requested),
            "bytes_reclaimed": max(0, before_bytes - after_bytes),
            "before": before,
            "after": after,
        }
