"""Prompt helper job utilities."""

from __future__ import annotations

from typing import Any, Dict

from services.job_payloads import json_sanitize


def _serialize_prompt_helper_job_impl(job) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "message": job.message,
        "progress": json_sanitize(job.progress),
        "total_steps": job.total_steps,
        "completed_steps": job.completed_steps,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "request": json_sanitize(job.request),
        "result": json_sanitize(job.result),
        "logs": json_sanitize(job.logs),
        "error": json_sanitize(job.error),
    }
