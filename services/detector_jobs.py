from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional


def _serialize_yolo_job_impl(job) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "config": job.config,
        "logs": job.logs,
        "metrics": job.metrics,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def _serialize_yolo_head_graft_job_impl(job) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "config": job.config,
        "logs": job.logs,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def _serialize_rfdetr_job_impl(job) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "config": job.config,
        "logs": job.logs,
        "metrics": job.metrics,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def _yolo_job_update_impl(
    job,
    *,
    status: Optional[str] = None,
    message: Optional[str] = None,
    progress: Optional[float] = None,
    error: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
) -> None:
    if status is not None:
        job.status = status
    if message is not None:
        job.message = message
    if progress is not None:
        job.progress = max(0.0, min(1.0, progress))
    if error is not None:
        job.error = error
    if result is not None:
        job.result = result
    job.updated_at = time.time()


def _yolo_job_log_impl(
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
        logger.info("[yolo-train %s] %s", job.job_id[:8], message)
    except Exception:
        pass


def _yolo_job_append_metric_impl(job, metric: Dict[str, Any], *, max_points: int) -> None:
    if not metric:
        return
    job.metrics.append(metric)
    if len(job.metrics) > max_points:
        job.metrics[:] = job.metrics[-max_points:]
    job.updated_at = time.time()


def _rfdetr_job_update_impl(
    job,
    *,
    status: Optional[str] = None,
    message: Optional[str] = None,
    progress: Optional[float] = None,
    error: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
) -> None:
    if status is not None:
        job.status = status
    if message is not None:
        job.message = message
    if progress is not None:
        job.progress = max(0.0, min(1.0, progress))
    if error is not None:
        job.error = error
    if result is not None:
        job.result = result
    job.updated_at = time.time()


def _rfdetr_job_log_impl(
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
        logger.info("[rfdetr-train %s] %s", job.job_id[:8], message)
    except Exception:
        pass


def _rfdetr_job_append_metric_impl(job, metric: Dict[str, Any], *, max_points: int) -> None:
    if not metric:
        return
    job.metrics.append(metric)
    if len(job.metrics) > max_points:
        job.metrics[:] = job.metrics[-max_points:]
    job.updated_at = time.time()
