from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional


def _serialize_clip_job_impl(job) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "logs": job.logs,
        "metrics": job.metrics,
        "artifacts": job.artifacts,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def _clip_job_log_impl(
    job,
    message: str,
    *,
    max_logs: int,
    logger: Optional[logging.Logger] = None,
) -> None:
    entry = {"timestamp": time.time(), "message": message}
    job.logs.append(entry)
    if len(job.logs) > max_logs:
        job.logs[:] = job.logs[-max_logs:]
    job.updated_at = time.time()
    if logger is None:
        logger = logging.getLogger(__name__)
    try:
        logger.info("[clip-train %s] %s", job.job_id[:8], message)
    except Exception:  # noqa: BLE001 - logging failures should never break workflow
        pass


def _clip_job_update_impl(
    job,
    *,
    status: Optional[str] = None,
    message: Optional[str] = None,
    progress: Optional[float] = None,
    error: Optional[str] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    max_logs: int,
    logger: Optional[logging.Logger] = None,
) -> None:
    if status is not None:
        job.status = status
    if message is not None:
        if message != job.message:
            job.message = message
            _clip_job_log_impl(job, message, max_logs=max_logs, logger=logger)
        else:
            job.message = message
    if progress is not None:
        job.progress = max(0.0, min(1.0, progress))
    if error is not None:
        job.error = error
    if artifacts is not None:
        job.artifacts = artifacts
