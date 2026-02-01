"""Error helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _agent_error_payload(code: str, message: Optional[str] = None, fix_hint: Optional[str] = None) -> Dict[str, Any]:
    return {
        "error": {
            "code": str(code),
            "message": str(message or code),
            "fix_hint": str(fix_hint) if fix_hint else None,
        }
    }


def _agent_error_from_detail(detail: str, tool_name: Optional[str] = None) -> Tuple[str, str, Optional[str]]:
    text = str(detail or "").strip()
    lower = text.lower()
    if "grid_cell_required" in lower or "grid_cell_invalid" in lower:
        return "missing_grid_cell", text, "Provide a valid grid_cell (e.g., C2)."
    if "grid_unavailable" in lower:
        return "missing_grid_cell", text, "Enable grid overlay or provide window_bbox_2d."
    if "sam3_similarity_label_required" in lower or "missing_label" in lower:
        return "missing_label", text, "Provide a canonical labelmap class."
    if "sam3_similarity_exemplar_required" in lower or "invalid_exemplar" in lower:
        return "invalid_exemplar", text, "Provide exemplar handles from list_candidates/get_tile_context."
    if "classifier_unavailable" in lower:
        return "classifier_unavailable", text, "Classifier unavailable; skip verification."
    if "prompt_type_invalid" in lower or "items_required" in lower or "agent_detector_mode_invalid" in lower:
        return "invalid_prompt", text, "Provide a valid prompt or items list."
    if "bbox_required" in lower or "invalid_bbox" in lower or "window_bbox_required" in lower:
        return "invalid_prompt", text, "Provide a valid bbox or grid_cell."
    if "context_handle_required" in lower or "context_chunk_index_required" in lower:
        return "invalid_prompt", text, "Provide context_handle and chunk_index."
    if "context_handle_missing" in lower or "context_chunk_index_invalid" in lower:
        return "invalid_prompt", text, "Use a valid context_handle and chunk_index."
    if "cluster_not_found" in lower:
        return "invalid_prompt", text, "Provide a valid cluster_id from list_candidates."
    if "tool_not_found" in lower:
        return "tool_failed", text, "Use a supported tool name."
    return "tool_failed", text, "Check tool arguments and retry once."
