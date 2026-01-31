from __future__ import annotations

from typing import Any, Dict


def _serialize_prompt_helper_job_impl(job) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "message": job.message,
        "progress": job.progress,
        "total_steps": job.total_steps,
        "completed_steps": job.completed_steps,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "request": job.request,
        "result": job.result,
        "logs": job.logs,
        "error": job.error,
    }
