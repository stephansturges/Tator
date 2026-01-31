from __future__ import annotations

import json
import logging
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
