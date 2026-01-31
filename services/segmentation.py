from __future__ import annotations

from typing import Any, Dict


def _serialize_seg_job_impl(job) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "result": job.result,
        "logs": job.logs,
        "error": job.error,
        "config": job.config,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
