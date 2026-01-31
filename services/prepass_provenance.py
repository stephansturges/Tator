from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


def _agent_attach_provenance(
    detections: Sequence[Dict[str, Any]],
    *,
    source: str,
    source_primary: Optional[str] = None,
    source_prompt: Optional[str] = None,
    source_exemplar_handles: Optional[Sequence[str]] = None,
    source_detector_run_id: Optional[str] = None,
) -> None:
    for det in detections:
        if not isinstance(det, dict):
            continue
        if source_primary:
            det.setdefault("source_primary", source_primary)
        else:
            det.setdefault("source_primary", source)
        if source_prompt:
            det.setdefault("source_prompt", source_prompt)
        if source_exemplar_handles:
            det.setdefault("source_exemplar_handles", list(source_exemplar_handles))
        if source_detector_run_id:
            det.setdefault("source_detector_run_id", source_detector_run_id)
        sources = det.get("source_list")
        if not isinstance(sources, list):
            sources = []
        if source and source not in sources:
            sources.append(source)
        det["source_list"] = sources


def _agent_finalize_provenance(detections: Sequence[Dict[str, Any]]) -> None:
    for det in detections:
        if not isinstance(det, dict):
            continue
        primary = det.get("source_primary")
        if not primary:
            primary = det.get("source") or det.get("score_source") or "unknown"
            det["source_primary"] = primary
        sources = det.get("source_list")
        if not isinstance(sources, list):
            sources = []
        if primary and primary not in sources:
            sources.append(primary)
        det["source_list"] = sources
