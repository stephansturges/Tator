from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional, Sequence


def _serialize_qwen_job_impl(job) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "logs": job.logs,
        "config": job.config,
        "metrics": job.metrics,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def _log_qwen_get_request_impl(endpoint: str, jobs: Sequence, logger: Optional[logging.Logger] = None) -> None:
    logger = logger or logging.getLogger(__name__)
    try:
        if not jobs:
            logger.info("[qwen-train] GET %s -> 0 jobs", endpoint)
            return
        for job in jobs:
            config = job.config or {}
            tracked_fields = {
                "accelerator": config.get("accelerator"),
                "devices": config.get("devices"),
                "batch_size": config.get("batch_size"),
                "accumulate_grad_batches": config.get("accumulate_grad_batches"),
            }
            logger.info(
                "[qwen-train %s] GET %s -> status=%s message=%s config=%s",
                job.job_id[:8],
                endpoint,
                job.status,
                job.message,
                json.dumps(tracked_fields, ensure_ascii=False),
            )
    except Exception:
        pass


def _qwen_job_log_impl(
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
        logger.info("[qwen-train %s] %s", job.job_id[:8], message)
    except Exception:
        pass


def _qwen_job_update_impl(
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
                _qwen_job_log_impl(job, message, max_logs=max_logs, logger=logger)
        else:
            job.message = message
    if progress is not None:
        job.progress = max(0.0, min(1.0, progress))
    if error is not None:
        job.error = error
    if result is not None:
        job.result = result
    job.updated_at = time.time()


def _qwen_job_append_metric_impl(job, metric: Dict[str, Any], *, max_points: Optional[int]) -> None:
    if not metric:
        return
    job.metrics.append(metric)
    if max_points and len(job.metrics) > max_points:
        job.metrics[:] = job.metrics[-max_points:]
    job.updated_at = time.time()
