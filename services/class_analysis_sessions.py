"""Small, restart-safe manifests for persisted Data Quality Explorer sessions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional


SESSION_MANIFEST_SCHEMA = "class-analysis-session-manifest-v1"
LATEST_SESSION_SCHEMA = "class-analysis-latest-session-v1"
SESSION_MANIFEST_FILENAME = "session_manifest.json"
LATEST_SESSION_FILENAME = "latest_session.json"
SESSION_MANIFEST_MAX_BYTES = 512 * 1024
_JOB_ID_PATTERN = re.compile(r"ca_[A-Za-z0-9_-]{1,120}\Z")
_TERMINAL_STATUSES = {"completed", "cancelled"}


class SessionManifestError(RuntimeError):
    """Raised when a persisted session manifest is unsafe or invalid."""


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _safe_root(root: Path | str, *, create: bool = False) -> Path:
    candidate = _absolute(root)
    if candidate == Path(candidate.anchor) or len(candidate.parts) < 3:
        raise SessionManifestError("class_analysis_session_root_too_broad")
    cursor = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise SessionManifestError("class_analysis_session_root_contains_symlink")
    if create:
        candidate.mkdir(parents=True, exist_ok=True)
    if not candidate.is_dir():
        raise SessionManifestError("class_analysis_session_root_not_directory")
    return candidate


def _safe_job_id(job_id: Any) -> str:
    value = str(job_id or "").strip()
    if _JOB_ID_PATTERN.fullmatch(value) is None:
        raise SessionManifestError("class_analysis_session_job_id_invalid")
    return value


def _safe_job_dir(root: Path, job_id: str, *, create: bool = False) -> Path:
    path = root / _safe_job_id(job_id)
    if path.is_symlink():
        raise SessionManifestError("class_analysis_session_job_dir_symlink")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir() or path.parent != root:
        raise SessionManifestError("class_analysis_session_job_dir_invalid")
    return path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, Any], *, max_bytes: int) -> int:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > max_bytes:
        raise SessionManifestError("class_analysis_session_manifest_too_large")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return len(encoded)


def _artifact_rows(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    prefixes = {
        "result": "result",
        "config": "config",
        "metadata": "metadata",
        "points": "points",
        "projection": "projection",
        "sidecar": "sidecar",
        "refinement_manifest": "refinement_manifest",
        "session_store": "session_store",
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for label, prefix in prefixes.items():
        filename = str(state.get(f"{prefix}_file") or "").strip()
        sha256 = str(state.get(f"{prefix}_sha256") or "").strip().lower()
        size = state.get(f"{prefix}_bytes")
        if not filename:
            continue
        row: dict[str, Any] = {"file": filename}
        if re.fullmatch(r"[0-9a-f]{64}", sha256):
            row["sha256"] = sha256
        if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
            row["bytes"] = size
        artifacts[label] = row
    return artifacts


def _compact_recipe(summary: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    resolved = summary.get("resolved_recipe")
    if not isinstance(resolved, Mapping):
        resolved = request.get("resolved_recipe")
    resolved = resolved if isinstance(resolved, Mapping) else {}
    return {
        "id": str(
            resolved.get("recipe_id")
            or summary.get("quality_recipe")
            or request.get("quality_recipe")
            or ""
        ),
        "label": str(resolved.get("label") or ""),
        "feature_mode": str(
            resolved.get("feature_mode")
            or summary.get("feature_mode")
            or request.get("feature_mode")
            or ""
        ),
        "projection": str(
            summary.get("projection") or request.get("projection") or ""
        ),
        "deep_evidence": bool(
            request.get("deep_evidence_pass")
            or request.get("refine_outliers")
        ),
    }


def build_session_manifest(
    *,
    job_id: str,
    status: str,
    summary: Mapping[str, Any],
    request: Mapping[str, Any],
    state: Mapping[str, Any],
    created_at: Optional[float] = None,
    updated_at: Optional[float] = None,
) -> dict[str, Any]:
    safe_job_id = _safe_job_id(job_id)
    clean_status = str(status or "").strip().lower()
    if clean_status not in _TERMINAL_STATUSES:
        raise SessionManifestError("class_analysis_session_status_invalid")
    refinement = summary.get("refinement")
    if not isinstance(refinement, Mapping):
        refinement = {}
    runtime = summary.get("runtime")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    refinement_runtime = runtime.get("refinement")
    refinement_runtime = (
        refinement_runtime if isinstance(refinement_runtime, Mapping) else {}
    )
    created = float(created_at or state.get("created_at") or time.time())
    updated = float(updated_at or state.get("updated_at") or time.time())
    manifest = {
        "schema": SESSION_MANIFEST_SCHEMA,
        "job_id": safe_job_id,
        "status": clean_status,
        "created_at": created,
        "updated_at": updated,
        "completed_at": updated if clean_status == "completed" else None,
        "source": {
            "mode": str(summary.get("source_mode") or request.get("source_mode") or ""),
            "id": str(summary.get("source_id") or request.get("source_id") or ""),
            "key": str(summary.get("source_key") or ""),
            "snapshot_id": str(
                summary.get("snapshot_id") or request.get("snapshot_id") or ""
            ),
            "snapshot_signature": str(request.get("snapshot_signature") or ""),
            "analysis_input_digest": str(
                summary.get("analysis_input_digest")
                or state.get("analysis_input_digest")
                or ""
            ),
            "dataset_label": str(summary.get("dataset_label") or ""),
            "image_count": max(0, int(summary.get("image_count") or 0)),
            "object_count": max(0, int(summary.get("object_count") or 0)),
        },
        "recipe": _compact_recipe(summary, request),
        "projection_modes": list(
            dict.fromkeys(
                str(mode)
                for mode in (
                    summary.get("projection_modes")
                    or summary.get("projection_coordinates_available")
                    or []
                )
                if str(mode)
            )
        ),
        "evidence": {
            "status": str(refinement.get("status") or "disabled"),
            "processed": max(0, int(refinement_runtime.get("processed") or 0)),
            "total": max(0, int(refinement_runtime.get("total") or 0)),
        },
        "artifacts": _artifact_rows(state),
    }
    # Validate size before any filesystem mutation.
    encoded = json.dumps(manifest, ensure_ascii=True, separators=(",", ":")).encode(
        "utf-8"
    )
    if len(encoded) > SESSION_MANIFEST_MAX_BYTES:
        raise SessionManifestError("class_analysis_session_manifest_too_large")
    return manifest


def validate_session_manifest(payload: Any, *, expected_job_id: str = "") -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise SessionManifestError("class_analysis_session_manifest_invalid")
    manifest = dict(payload)
    if manifest.get("schema") != SESSION_MANIFEST_SCHEMA:
        raise SessionManifestError("class_analysis_session_manifest_schema_invalid")
    job_id = _safe_job_id(manifest.get("job_id"))
    if expected_job_id and job_id != _safe_job_id(expected_job_id):
        raise SessionManifestError("class_analysis_session_manifest_job_mismatch")
    if str(manifest.get("status") or "") not in _TERMINAL_STATUSES:
        raise SessionManifestError("class_analysis_session_manifest_status_invalid")
    for field in ("source", "recipe", "evidence", "artifacts"):
        if not isinstance(manifest.get(field), Mapping):
            raise SessionManifestError(
                f"class_analysis_session_manifest_{field}_invalid"
            )
    return manifest


def read_session_manifest(root: Path | str, job_id: str) -> dict[str, Any]:
    safe_root = _safe_root(root)
    safe_job_id = _safe_job_id(job_id)
    job_dir = _safe_job_dir(safe_root, safe_job_id)
    path = job_dir / SESSION_MANIFEST_FILENAME
    if path.is_symlink() or not path.is_file():
        raise SessionManifestError("class_analysis_session_manifest_not_found")
    stat_result = path.stat()
    if stat_result.st_size <= 0 or stat_result.st_size > SESSION_MANIFEST_MAX_BYTES:
        raise SessionManifestError("class_analysis_session_manifest_size_invalid")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise SessionManifestError("class_analysis_session_manifest_json_invalid") from exc
    return validate_session_manifest(payload, expected_job_id=safe_job_id)


def write_session_manifest(
    root: Path | str,
    manifest: Mapping[str, Any],
    *,
    publish_latest: bool = True,
) -> dict[str, Any]:
    safe_root = _safe_root(root, create=True)
    validated = validate_session_manifest(manifest)
    job_id = str(validated["job_id"])
    job_dir = _safe_job_dir(safe_root, job_id, create=True)
    _atomic_json(
        job_dir / SESSION_MANIFEST_FILENAME,
        validated,
        max_bytes=SESSION_MANIFEST_MAX_BYTES,
    )
    if publish_latest and validated.get("status") == "completed":
        manifest_digest = hashlib.sha256(
            json.dumps(
                validated,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        pointer = {
            "schema": LATEST_SESSION_SCHEMA,
            "job_id": job_id,
            "manifest_sha256": manifest_digest,
            "updated_at": float(validated.get("updated_at") or time.time()),
        }
        _atomic_json(
            safe_root / LATEST_SESSION_FILENAME,
            pointer,
            max_bytes=16 * 1024,
        )
    return validated


def _scan_latest(safe_root: Path) -> Optional[dict[str, Any]]:
    latest: Optional[dict[str, Any]] = None
    for candidate in safe_root.iterdir():
        if not candidate.is_dir() or candidate.is_symlink():
            continue
        if _JOB_ID_PATTERN.fullmatch(candidate.name) is None:
            continue
        try:
            manifest = read_session_manifest(safe_root, candidate.name)
        except SessionManifestError:
            continue
        if manifest.get("status") != "completed":
            continue
        if latest is None or (
            float(manifest.get("updated_at") or 0.0), str(manifest.get("job_id"))
        ) > (
            float(latest.get("updated_at") or 0.0), str(latest.get("job_id"))
        ):
            latest = manifest
    return latest


def latest_session(root: Path | str) -> Optional[dict[str, Any]]:
    safe_root = _safe_root(root, create=True)
    pointer_path = safe_root / LATEST_SESSION_FILENAME
    if pointer_path.is_file() and not pointer_path.is_symlink():
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            if (
                isinstance(pointer, Mapping)
                and pointer.get("schema") == LATEST_SESSION_SCHEMA
            ):
                manifest = read_session_manifest(safe_root, pointer.get("job_id"))
                if manifest.get("status") == "completed":
                    return manifest
        except (OSError, UnicodeDecodeError, ValueError, SessionManifestError):
            pass
    manifest = _scan_latest(safe_root)
    if manifest is not None:
        write_session_manifest(safe_root, manifest, publish_latest=True)
    return manifest
