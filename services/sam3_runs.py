"""SAM3 run registry helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.io import _path_is_within_root_impl


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


def _safe_path_mtime_impl(path: Path) -> float:
    if path.is_symlink():
        return 0.0
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _active_run_paths_for_variant_impl(
    *,
    variant: str,
    jobs_lock: Any,
    jobs: Dict[str, Any],
) -> set[Path]:
    paths: set[Path] = set()
    with jobs_lock:
        job_list = list(jobs.values())
    for job in job_list:
        if getattr(job, "status", None) not in {"running", "queued", "cancelling"}:
            continue
        exp_dir = None
        try:
            exp_dir = job.config.get("paths", {}).get("experiment_log_dir")
        except Exception:
            exp_dir = None
        if exp_dir:
            try:
                paths.add(Path(exp_dir).resolve())
            except Exception:
                continue
    return paths


def _describe_run_dir_impl(
    *,
    run_dir: Path,
    variant: str,
    active_paths: set[Path],
    dir_size_fn: Callable[[Path], int],
) -> Dict[str, Any]:
    run_root = run_dir.resolve()
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    tensorboard_dir = run_dir / "tensorboard"
    dumps_dir = run_dir / "dumps"
    marker_path = run_dir / ".promoted"
    promoted = False
    promoted_at: Optional[float] = None
    if marker_path.exists() and not marker_path.is_symlink():
        promoted = True
        try:
            marker_resolved = marker_path.resolve(strict=True)
            if not _path_is_within_root_impl(marker_resolved, run_root):
                raise ValueError("promoted marker escaped run root")
            meta = json.loads(marker_path.read_text())
            promoted_at = meta.get("timestamp")
        except Exception:
            promoted_at = None
    checkpoints: List[Dict[str, Any]] = []
    if checkpoints_dir.exists() and not checkpoints_dir.is_symlink():
        checkpoints_root = checkpoints_dir.resolve()
        if not _path_is_within_root_impl(checkpoints_root, run_root):
            checkpoints_root = None
    else:
        checkpoints_root = None
    if checkpoints_root is not None:
        for ckpt in sorted(
            checkpoints_dir.iterdir(),
            key=_safe_path_mtime_impl,
            reverse=True,
        ):
            if ckpt.is_symlink():
                continue
            if ckpt.is_file():
                try:
                    if not _path_is_within_root_impl(ckpt.resolve(strict=True), checkpoints_root):
                        continue
                    stat = ckpt.stat()
                    checkpoints.append(
                        {
                            "file": ckpt.name,
                            "path": str(ckpt),
                            "size_bytes": stat.st_size,
                            "updated_at": stat.st_mtime,
                        }
                    )
                except Exception:
                    continue
    try:
        dir_stat = run_dir.stat()
        created_at = dir_stat.st_ctime
        updated_at = dir_stat.st_mtime
    except Exception:
        created_at = None
        updated_at = created_at
    entry = {
        "id": run_dir.name,
        "variant": variant,
        "path": str(run_dir),
        "created_at": created_at,
        "updated_at": updated_at,
        "size_bytes": dir_size_fn(run_dir),
        "checkpoints_size_bytes": dir_size_fn(checkpoints_dir),
        "logs_size_bytes": dir_size_fn(logs_dir),
        "tensorboard_size_bytes": dir_size_fn(tensorboard_dir),
        "dumps_size_bytes": dir_size_fn(dumps_dir),
        "checkpoints": checkpoints,
        "active": run_dir.resolve() in active_paths,
        "promoted": promoted,
        "promoted_at": promoted_at,
    }
    return entry


def _list_sam3_runs_impl(
    *,
    variant: str,
    job_root: Path,
    dataset_root: Path,
    active_paths_fn: Callable[[str], set[Path]],
    describe_fn: Callable[[Path, str, set[Path]], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    root = job_root
    if _path_has_symlink_component(root):
        return []
    if not root.exists():
        return []
    try:
        root_resolved = root.resolve()
    except Exception:
        return []
    dataset_root_resolved = dataset_root.resolve(strict=False)
    active_paths = active_paths_fn(variant)
    runs: List[Dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=_safe_path_mtime_impl, reverse=True):
        if child.is_symlink():
            continue
        try:
            run_dir = child.resolve()
        except Exception:
            continue
        if not _path_is_within_root_impl(run_dir, root_resolved):
            continue
        if not run_dir.is_dir():
            continue
        if variant == "sam3" and run_dir == dataset_root_resolved:
            continue
        if run_dir.name.lower() == "datasets":
            continue
        try:
            runs.append(describe_fn(run_dir, variant, active_paths))
        except Exception:
            continue
    return runs


def _run_dir_for_request_impl(
    *,
    run_id: str,
    variant: str,
    job_root: Path,
    http_exception_cls: Any,
    http_400: int,
    http_404: int,
) -> Path:
    root = job_root
    raw_id = str(run_id or "").strip()
    if _path_has_symlink_component(root):
        raise http_exception_cls(status_code=http_400, detail="invalid_run_id")
    if (
        not raw_id
        or raw_id in {".", ".."}
        or "/" in raw_id
        or "\\" in raw_id
        or Path(raw_id).is_absolute()
    ):
        raise http_exception_cls(status_code=http_400, detail="invalid_run_id")
    requested = root / raw_id
    if requested.is_symlink():
        raise http_exception_cls(status_code=http_400, detail="invalid_run_id")
    root_resolved = root.resolve()
    candidate = requested.resolve()
    if not _path_is_within_root_impl(candidate, root_resolved):
        raise http_exception_cls(status_code=http_400, detail="invalid_run_id")
    if not candidate.exists() or not candidate.is_dir():
        raise http_exception_cls(status_code=http_404, detail="sam3_run_not_found")
    return candidate


def _delete_run_scope_impl(
    *,
    run_dir: Path,
    scope: str,
    dir_size_fn: Callable[[Path], int],
    rmtree_fn: Callable[[Path], None],
) -> Tuple[List[str], int]:
    targets: List[Path] = []
    if scope == "all":
        targets.append(run_dir)
    else:
        mapping = {
            "checkpoints": run_dir / "checkpoints",
            "logs": run_dir / "logs",
            "tensorboard": run_dir / "tensorboard",
            "dumps": run_dir / "dumps",
        }
        target = mapping.get(scope)
        if target:
            targets.append(target)
    deleted: List[str] = []
    freed = 0
    run_root = run_dir.resolve()
    for target in targets:
        if not target.exists():
            continue
        if target.is_symlink():
            try:
                target.unlink(missing_ok=True)
                deleted.append(str(target))
            except Exception:
                continue
            continue
        try:
            target_resolved = target.resolve()
        except Exception:
            continue
        if not _path_is_within_root_impl(target_resolved, run_root):
            continue
        freed += dir_size_fn(target)
        try:
            rmtree_fn(target)
        except Exception:
            continue
        deleted.append(str(target))
    return deleted, freed
