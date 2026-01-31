from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional


def _serialize_sam3_job_impl(job) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "logs": job.logs,
        "metrics": job.metrics,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def _sam3_job_log_impl(
    job,
    message: str,
    *,
    max_logs: int,
    logger: Optional[logging.Logger] = None,
) -> None:
    job.log_seq += 1
    entry = {"timestamp": time.time(), "message": message, "seq": job.log_seq}
    job.logs.append(entry)
    if len(job.logs) > max_logs:
        job.logs[:] = job.logs[-max_logs:]
    job.updated_at = time.time()
    if logger is None:
        logger = logging.getLogger(__name__)
    try:
        logger.info("[sam3-train %s] %s", job.job_id[:8], message)
    except Exception:
        pass


def _sam3_job_append_metric_impl(job, metric: Dict[str, Any], *, max_points: Optional[int]) -> None:
    if not metric:
        return
    job.metrics.append(metric)
    if max_points and len(job.metrics) > max_points:
        job.metrics[:] = job.metrics[-max_points:]
    job.updated_at = time.time()


def _sam3_job_update_impl(
    job,
    *,
    status: Optional[str] = None,
    message: Optional[str] = None,
    progress: Optional[float] = None,
    error: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
    log_message: bool = True,
    max_logs: int,
    logger: Optional[logging.Logger] = None,
) -> None:
    if status is not None:
        job.status = status
    if message is not None:
        if message != job.message:
            job.message = message
            if log_message:
                _sam3_job_log_impl(job, message, max_logs=max_logs, logger=logger)
        else:
            job.message = message
    if progress is not None:
        job.progress = max(0.0, min(1.0, progress))
    if error is not None:
        job.error = error
    if result is not None:
        job.result = result
    job.updated_at = time.time()
