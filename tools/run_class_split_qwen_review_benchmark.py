#!/usr/bin/env python3
"""Benchmark the Class Split Qwen likely-wrong review agent.

The script runs the backend review implementation directly against an existing
Class Analysis result. It is deliberately in-process so state-machine changes can
be tested without a browser or a running HTTP server.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import localinferenceapi as api


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _record_current_class_plausible(record: Dict[str, Any]) -> bool:
    if _coerce_bool(record.get("current_class_plausible")):
        return True
    verifier = record.get("cue_verifier")
    if isinstance(verifier, dict) and _coerce_bool(verifier.get("current_class_plausible")):
        return True
    return False


def _safe_read_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not path.is_file():
        return events
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _normalize_filter_values(values: Sequence[str]) -> set[str]:
    normalized: set[str] = set()
    for value in values or []:
        for part in str(value or "").split(","):
            item = part.strip().lower()
            if item:
                normalized.add(item)
    return normalized


def _source_point_ids(
    source_run: Optional[Path],
    count: int,
    start: int,
    *,
    backend_tiers: Optional[set[str]] = None,
    decisions: Optional[set[str]] = None,
    dispositions: Optional[set[str]] = None,
    disposition_signals: Optional[set[str]] = None,
    guarded_only: bool = False,
    reviewable_only: bool = False,
) -> List[str]:
    if not source_run:
        return []
    data = _load_json(source_run)
    records = data.get("records") if isinstance(data.get("records"), list) else []
    point_ids: List[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        point_id = str(record.get("point_id") or "").strip()
        if not point_id:
            continue
        disposition = record.get("review_disposition") if isinstance(record.get("review_disposition"), dict) else {}
        guarded = record.get("guarded_recommendation") if isinstance(record.get("guarded_recommendation"), dict) else None
        if backend_tiers and str(record.get("backend_tier") or "").strip().lower() not in backend_tiers:
            continue
        if decisions and str(record.get("decision") or "").strip().lower() not in decisions:
            continue
        if dispositions and str(disposition.get("disposition") or "").strip().lower() not in dispositions:
            continue
        if disposition_signals and str(disposition.get("signal") or "").strip().lower() not in disposition_signals:
            continue
        if guarded_only and not (guarded and guarded.get("blocked")):
            continue
        if reviewable_only:
            signal = str(disposition.get("signal") or "").strip().lower()
            if signal not in {"actionable", "guarded_human_triage", "useful_negative"}:
                continue
        point_ids.append(point_id)
    start_idx = max(0, int(start or 0))
    return point_ids[start_idx : start_idx + max(0, int(count or 0))]


def _fallback_point_ids(result: Dict[str, Any], count: int, start: int) -> List[str]:
    candidates = result.get("wrong_class_candidates")
    if not isinstance(candidates, list):
        candidates = [
            point
            for point in result.get("points") or []
            if isinstance(point, dict) and point.get("is_wrong_class_candidate")
        ]
    point_ids = [
        str(point.get("point_id") or "").strip()
        for point in candidates
        if isinstance(point, dict) and str(point.get("point_id") or "").strip()
    ]
    start_idx = max(0, int(start or 0))
    return point_ids[start_idx : start_idx + max(0, int(count or 0))]


def _register_parent_job(job_id: str, result: Dict[str, Any], result_path: Path) -> None:
    parent = api.ClassAnalysisJob(
        job_id=job_id,
        status="completed",
        progress=1.0,
        message="Loaded benchmark parent result.",
        result=result,
        result_path=str(result_path.resolve()),
    )
    with api.CLASS_ANALYSIS_JOBS_LOCK:
        api.CLASS_ANALYSIS_JOBS[job_id] = parent


def _review_dir(parent_job_id: str, review_id: str) -> Path:
    return Path("uploads/class_analysis") / parent_job_id / "qwen_reviews" / review_id


def _event_counts(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    phase_counts = Counter(str(event.get("phase") or "") for event in events if event.get("type") == "model_input")
    controller_calls = [
        event for event in events if event.get("type") == "controller_tool_call"
    ]
    tool_results = [event for event in events if event.get("type") == "tool_result"]
    router_events = [event for event in events if event.get("type") == "router_decision"]
    final_errors = [event for event in events if event.get("type") == "final_validation_error"]
    cue_verifier_errors = [event for event in events if event.get("type") == "cue_verifier_parse_error"]

    def _schema_label(event: Dict[str, Any]) -> str:
        schema = event.get("tool_schema") if isinstance(event.get("tool_schema"), list) else []
        name = ((schema or [{}])[0] or {}).get("name")
        if name:
            return str(name)
        phase = str(event.get("phase") or "").strip()
        if phase.startswith("concept_brief_"):
            return "concept_brief"
        if phase.startswith("concept_pair_"):
            return "concept_pair_contrast"
        return phase or "plain_model_call"

    return {
        "model_input_events": sum(phase_counts.values()),
        "model_output_events": sum(1 for event in events if event.get("type") == "model_output"),
        "model_phase_counts": dict(phase_counts),
        "concept_brief_model_calls": sum(
            count for phase, count in phase_counts.items() if str(phase).startswith("concept_brief_")
        ),
        "pair_contrast_model_calls": sum(
            count for phase, count in phase_counts.items() if str(phase).startswith("concept_pair_")
        ),
        "controller_tool_calls": len(controller_calls),
        "tool_result_events": len(tool_results),
        "router_decisions": len(router_events),
        "final_validation_errors": len(final_errors),
        "cue_verifier_parse_errors": len(cue_verifier_errors),
        "local_consensus_controller_calls": sum(
            1 for event in controller_calls if event.get("tool") == "inspect_local_consensus_context"
        ),
        "active_tool_schemas": [_schema_label(event) for event in events if event.get("type") == "model_input"],
        "router": (router_events[-1].get("router") if router_events else None),
    }


def _record_from_review(
    ordinal: int,
    review: api.ClassAnalysisQwenReviewJob,
    started_at: float,
) -> Dict[str, Any]:
    final = review.result or {}
    events_path = _review_dir(review.parent_job_id, review.review_id) / "events.jsonl"
    events = _safe_read_events(events_path)
    event_counts = _event_counts(events)
    backend_quality = final.get("backend_visual_quality") if isinstance(final.get("backend_visual_quality"), dict) else {}
    deterministic_context = final.get("deterministic_context") if isinstance(final.get("deterministic_context"), dict) else {}
    scale_context = deterministic_context.get("scale") if isinstance(deterministic_context.get("scale"), dict) else {}
    embedding_context = deterministic_context.get("embedding") if isinstance(deterministic_context.get("embedding"), dict) else {}
    cue_verifier = final.get("cue_verifier") if isinstance(final.get("cue_verifier"), dict) else {}
    final_current_plausible = _coerce_bool(final.get("current_class_plausible"))
    cue_current_plausible = _coerce_bool(cue_verifier.get("current_class_plausible"))
    current_plausible = final_current_plausible or cue_current_plausible
    current_plausibility_reason = str(
        final.get("current_class_plausibility_reason")
        or cue_verifier.get("current_class_plausibility_reason")
        or ""
    )
    return {
        "ordinal": ordinal,
        "review_id": review.review_id,
        "point_id": review.point_id,
        "status": review.status,
        "error": review.error or "",
        "elapsed_seconds": time.time() - started_at,
        "decision": final.get("decision"),
        "target_class": final.get("target_class"),
        "current_class": final.get("current_class"),
        "suggested_neighbor_class": final.get("suggested_neighbor_class"),
        "confidence": final.get("confidence"),
        "visual_quality": final.get("visual_quality"),
        "object_visibility": final.get("object_visibility"),
        "current_evidence": final.get("current_evidence"),
        "suggested_evidence": final.get("suggested_evidence"),
        "target_evidence": final.get("target_evidence"),
        "overlap_assessment": final.get("overlap_assessment"),
        "overlap_explains_candidate_similarity": final.get("overlap_explains_candidate_similarity"),
        "overlap_adjudication_verified": final.get("overlap_adjudication_verified"),
        "anchor_adjudication_verified": final.get("anchor_adjudication_verified"),
        "anchor_adjudication_reason": final.get("anchor_adjudication_reason") or "",
        "current_class_plausible": current_plausible,
        "current_class_plausibility_reason": current_plausibility_reason,
        "cue_verifier_current_class_plausible": cue_current_plausible,
        "dual_bbox_resolution": final.get("dual_bbox_resolution"),
        "dual_bbox_conflict": final.get("dual_bbox_conflict"),
        "specificity_alignment": final.get("specificity_alignment"),
        "target_background_contrast": final.get("target_background_contrast"),
        "target_identity_summary": final.get("target_identity_summary") or "",
        "target_identity_uncertainty": final.get("target_identity_uncertainty") or "",
        "target_identity_evidence_ids": final.get("target_identity_evidence_ids") or [],
        "whole_target_extent_supported": final.get("whole_target_extent_supported"),
        "whole_target_extent_reason": final.get("whole_target_extent_reason") or "",
        "visible_target_cues": final.get("visible_target_cues") or [],
        "supporting_clean_evidence_ids": final.get("supporting_clean_evidence_ids") or [],
        "anchor_evidence_current": final.get("anchor_evidence_current"),
        "anchor_evidence_suggested": final.get("anchor_evidence_suggested"),
        "local_context_evidence": final.get("local_context_evidence"),
        "local_consensus_evidence": final.get("local_consensus_evidence"),
        "global_context_evidence": final.get("global_context_evidence"),
        "same_image_scale_evidence": final.get("same_image_scale_evidence"),
        "same_image_embedding_evidence": final.get("same_image_embedding_evidence"),
        "same_image_scale_report_signal": scale_context.get("signal"),
        "same_image_embedding_report_signal": embedding_context.get("signal"),
        "same_image_scale_anchor_count": scale_context.get("same_image_anchor_count"),
        "same_image_embedding_anchor_count": embedding_context.get("same_image_anchor_count"),
        "deterministic_context": deterministic_context,
        "backend_tier": backend_quality.get("tier"),
        "guardrail_reasons": final.get("guardrail_reasons") or [],
        "advisory_reasons": final.get("advisory_reasons") or [],
        "guarded_recommendation": final.get("guarded_recommendation"),
        "cue_verifier": cue_verifier,
        "review_disposition": final.get("review_disposition") or {},
        "evidence_ledger": final.get("evidence_ledger") or {},
        "class_concept_briefs": final.get("class_concept_briefs") or {},
        "rationale_short": final.get("rationale_short") or "",
        "model_compact_arguments": final.get("model_compact_arguments") or {},
        "controller_preflight": final.get("controller_preflight") or {},
        "controller_reconciliation": final.get("controller_reconciliation") or {"applied": False},
        "review_agent_controller": final.get("review_agent_controller") or "",
        "router": final.get("router"),
        "evidence": review.evidence,
        "events_path": str(Path("qwen_reviews") / review.review_id / "events.jsonl"),
        **event_counts,
    }


def _summarize(records: Sequence[Dict[str, Any]], *, run_id: str, job_id: str, model_id: str, started_at: float) -> Dict[str, Any]:
    decision_counts = Counter(str(record.get("decision") or "missing") for record in records)
    status_counts = Counter(str(record.get("status") or "missing") for record in records)
    backend_tiers = Counter(str(record.get("backend_tier") or "unknown") for record in records)
    schema_sequences = Counter(
        "->".join(str(item or "") for item in record.get("active_tool_schemas") or [])
        for record in records
    )
    router_actions = Counter(
        str((record.get("router") or {}).get("action") or "none")
        for record in records
    )
    same_image_scale = Counter(str(record.get("same_image_scale_evidence") or "missing") for record in records)
    same_image_embedding = Counter(str(record.get("same_image_embedding_evidence") or "missing") for record in records)
    same_image_scale_report = Counter(str(record.get("same_image_scale_report_signal") or "missing") for record in records)
    same_image_embedding_report = Counter(str(record.get("same_image_embedding_report_signal") or "missing") for record in records)
    specificity_alignment = Counter(str(record.get("specificity_alignment") or "missing") for record in records)
    target_background_contrast = Counter(str(record.get("target_background_contrast") or "missing") for record in records)
    target_identity_uncertainty = Counter(str(record.get("target_identity_uncertainty") or "missing") for record in records)
    whole_target_extent = Counter(
        "supported" if record.get("whole_target_extent_supported") is True
        else "unsupported" if record.get("whole_target_extent_supported") is False
        else "missing"
        for record in records
    )
    dual_bbox_resolution = Counter(str(record.get("dual_bbox_resolution") or "missing") for record in records)
    overlap_adjudication_verified = sum(1 for record in records if bool(record.get("overlap_adjudication_verified")))
    anchor_adjudication_verified = sum(1 for record in records if bool(record.get("anchor_adjudication_verified")))
    anchor_support_basis = Counter(
        str(record["cue_verifier"].get("anchor_support_basis") or "missing")
        for record in records
        if isinstance(record.get("cue_verifier"), dict) and record["cue_verifier"]
    )
    anchor_support_verified = sum(
        1
        for record in records
        if isinstance(record.get("cue_verifier"), dict)
        and bool(record["cue_verifier"].get("anchor_support_verified"))
    )
    cue_verifier_contrastive_support = sum(
        1
        for record in records
        if isinstance(record.get("cue_verifier"), dict)
        and bool(record["cue_verifier"].get("contrastively_supported_target"))
    )
    cue_verifier_missing_current_cues = sum(
        1
        for record in records
        if isinstance(record.get("cue_verifier"), dict)
        and bool(record["cue_verifier"].get("current_class_missing_or_inconsistent_cues"))
    )
    current_class_plausible = sum(1 for record in records if _record_current_class_plausible(record))
    disposition_counts = Counter(
        str((record.get("review_disposition") or {}).get("disposition") or "missing")
        for record in records
    )
    disposition_signal_counts = Counter(
        str((record.get("review_disposition") or {}).get("signal") or "missing")
        for record in records
    )
    return {
        "run_id": run_id,
        "job_id": job_id,
        "model_id": model_id,
        "sample_size": len(records),
        "completed": sum(1 for record in records if record.get("status") == "completed"),
        "failed": sum(1 for record in records if record.get("status") == "failed"),
        "elapsed_seconds": time.time() - started_at,
        "decision_counts": dict(decision_counts),
        "status_counts": dict(status_counts),
        "backend_tier_counts": dict(backend_tiers),
        "schema_sequence_counts": dict(schema_sequences),
        "router_action_counts": dict(router_actions),
        "same_image_scale_evidence_counts": dict(same_image_scale),
        "same_image_embedding_evidence_counts": dict(same_image_embedding),
        "same_image_scale_report_signal_counts": dict(same_image_scale_report),
        "same_image_embedding_report_signal_counts": dict(same_image_embedding_report),
        "specificity_alignment_counts": dict(specificity_alignment),
        "target_background_contrast_counts": dict(target_background_contrast),
        "target_identity_uncertainty_counts": dict(target_identity_uncertainty),
        "whole_target_extent_counts": dict(whole_target_extent),
        "dual_bbox_resolution_counts": dict(dual_bbox_resolution),
        "overlap_adjudication_verified_count": overlap_adjudication_verified,
        "anchor_adjudication_verified_count": anchor_adjudication_verified,
        "anchor_support_basis_counts": dict(anchor_support_basis),
        "anchor_support_verified_count": anchor_support_verified,
        "cue_verifier_contrastive_support_count": cue_verifier_contrastive_support,
        "cue_verifier_missing_current_cue_count": cue_verifier_missing_current_cues,
        "current_class_plausible_count": current_class_plausible,
        "review_disposition_counts": dict(disposition_counts),
        "review_disposition_signal_counts": dict(disposition_signal_counts),
        "effective_human_signal_count": sum(
            1
            for record in records
            if str((record.get("review_disposition") or {}).get("signal") or "") in {
                "actionable",
                "guarded_human_triage",
                "useful_negative",
            }
        ),
        "guarded_human_triage_count": sum(
            1
            for record in records
            if str((record.get("review_disposition") or {}).get("signal") or "") == "guarded_human_triage"
        ),
        "state_machine_v2_count": sum(
            1 for record in records if record.get("review_agent_controller") == "state_machine_v2"
        ),
        "final_validation_error_count": sum(int(record.get("final_validation_errors") or 0) for record in records),
        "local_consensus_controller_calls": sum(
            int(record.get("local_consensus_controller_calls") or 0) for record in records
        ),
        "concept_brief_model_calls": sum(int(record.get("concept_brief_model_calls") or 0) for record in records),
        "pair_contrast_model_calls": sum(int(record.get("pair_contrast_model_calls") or 0) for record in records),
        "concept_brief_enabled_count": sum(
            1
            for record in records
            if isinstance(record.get("class_concept_briefs"), dict)
            and record["class_concept_briefs"].get("enabled")
        ),
        "concept_brief_cache_hit_count": sum(
            1
            for record in records
            if isinstance(record.get("class_concept_briefs"), dict)
            for hit in (record["class_concept_briefs"].get("cache_hits") or [])
            if hit
        ),
        "pair_contrast_cache_hit_count": sum(
            1
            for record in records
            if isinstance(record.get("class_concept_briefs"), dict)
            for hit in (record["class_concept_briefs"].get("pair_cache_hits") or [])
            if hit
        ),
        "controller_reconciled_count": sum(
            1
            for record in records
            if isinstance(record.get("controller_reconciliation"), dict)
            and record["controller_reconciliation"].get("applied")
        ),
        "guarded_recommendation_count": sum(
            1
            for record in records
            if isinstance(record.get("guarded_recommendation"), dict)
            and record["guarded_recommendation"].get("blocked")
        ),
        "cue_verifier_count": sum(
            1
            for record in records
            if isinstance(record.get("cue_verifier"), dict) and record["cue_verifier"]
        ),
        "cue_verifier_promoted_count": sum(
            1
            for record in records
            if isinstance(record.get("cue_verifier"), dict)
            and record["cue_verifier"]
            and record["cue_verifier"].get("promoted_from_guarded_recommendation")
        ),
        "cue_verifier_parse_error_count": sum(int(record.get("cue_verifier_parse_errors") or 0) for record in records),
        "records_written": len(records),
    }


def _evidence_path(parent_job_id: str, review_id: str, evidence: Dict[str, Any]) -> Optional[Path]:
    filename = str(evidence.get("filename") or "").strip()
    if not filename:
        return None
    path = _review_dir(parent_job_id, review_id) / "evidence" / filename
    return path if path.is_file() else None


def _make_visual_sheet(
    records: Sequence[Dict[str, Any]],
    *,
    parent_job_id: str,
    output_path: Path,
    title: str,
    decisions: Optional[Iterable[str]] = None,
    guarded_only: bool = False,
    limit: int = 12,
) -> None:
    selected_decisions = {str(item) for item in decisions or []}
    rows = [
        record
        for record in records
        if not selected_decisions or str(record.get("decision") or "") in selected_decisions
        if not guarded_only
        or (
            isinstance(record.get("guarded_recommendation"), dict)
            and record["guarded_recommendation"].get("blocked")
        )
    ][: max(1, int(limit or 1))]
    if not rows:
        return
    cell_w, cell_h = 420, 500
    cols = 3
    sheet = Image.new("RGB", (cell_w * cols, 44 + cell_h * math.ceil(len(rows) / cols)), (8, 20, 10))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), title[:180], fill=(142, 255, 102))
    for idx, record in enumerate(rows):
        x = (idx % cols) * cell_w
        y = 44 + (idx // cols) * cell_h
        evidence_list = record.get("evidence") if isinstance(record.get("evidence"), list) else []
        preferred = None
        for kind in ("target_detail", "target_context", "zoom_region", "local_consensus_context", "class_context_pack"):
            preferred = next((ev for ev in evidence_list if isinstance(ev, dict) and ev.get("kind") == kind), None)
            if preferred:
                break
        if preferred:
            path = _evidence_path(parent_job_id, str(record.get("review_id") or ""), preferred)
            if path:
                try:
                    with Image.open(path) as loaded:
                        image = loaded.convert("RGB")
                    image.thumbnail((cell_w - 24, 310), Image.Resampling.LANCZOS)
                    sheet.paste(image, (x + (cell_w - image.width) // 2, y + 8))
                except Exception:
                    pass
        text_y = y + 326
        lines = [
            f"#{record.get('ordinal')} {record.get('decision')} conf {float(record.get('confidence') or 0):.2f}",
            f"{record.get('current_class')} -> {record.get('suggested_neighbor_class')} target {record.get('target_class')}",
            f"tier {record.get('backend_tier')} vis {record.get('visual_quality')}/{record.get('object_visibility')}",
            f"specificity {record.get('specificity_alignment')} contrast {record.get('target_background_contrast')}",
            f"router {(record.get('router') or {}).get('action')}",
            str(record.get("rationale_short") or "")[:90],
        ]
        guarded = record.get("guarded_recommendation")
        if isinstance(guarded, dict) and guarded.get("blocked"):
            lines.insert(
                3,
                (
                    f"guarded {guarded.get('decision')} target {guarded.get('target_class')} "
                    f"conf {float(guarded.get('confidence') or 0):.2f}"
                ),
            )
        disposition = record.get("review_disposition") if isinstance(record.get("review_disposition"), dict) else {}
        if disposition:
            lines.insert(4, f"signal {disposition.get('signal')} / {disposition.get('disposition')}")
        for line in lines:
            draw.text((x + 10, text_y), line[:92], fill=(210, 255, 190))
            text_y += 24
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def _write_benchmark_outputs(
    args: argparse.Namespace,
    *,
    root: Path,
    job_id: str,
    run_id: str,
    model_id: str,
    records: Sequence[Dict[str, Any]],
    started_at: float,
) -> Path:
    summary = _summarize(records, run_id=run_id, job_id=job_id, model_id=model_id or "default", started_at=started_at)
    output = root / "qwen_reviews" / f"{run_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "records": list(records)}
    output.write_text(json.dumps(api.json_sanitize(payload), indent=2, sort_keys=True), encoding="utf-8")
    visual_limit = int(args.visual_limit or len(records) or 1)
    _make_visual_sheet(
        records,
        parent_job_id=job_id,
        output_path=output.with_name(f"{run_id}_visual_non_skip.jpg"),
        title=f"{run_id} non-skip reviews",
        decisions={"accept_suggested", "confirm_current", "change_to_other"},
        limit=visual_limit,
    )
    _make_visual_sheet(
        records,
        parent_job_id=job_id,
        output_path=output.with_name(f"{run_id}_visual_guarded.jpg"),
        title=f"{run_id} guarded review opinions",
        guarded_only=True,
        limit=visual_limit,
    )
    _make_visual_sheet(
        records,
        parent_job_id=job_id,
        output_path=output.with_name(f"{run_id}_visual_all.jpg"),
        title=f"{run_id} all reviews",
        limit=visual_limit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {output}")
    if args.audit:
        audit_output = output.with_name(f"{run_id}_audit.json")
        cmd = [
            sys.executable,
            str(ROOT / "tools" / "analyze_class_split_qwen_review_benchmark.py"),
            str(output),
            "--write-json",
            str(audit_output),
        ]
        if args.compare_run:
            cmd.extend(["--compare-run", str(args.compare_run)])
        if args.fail_on_unsafe:
            cmd.append("--fail-on-unsafe")
        subprocess.run(cmd, check=True)
    return output


def _run_subprocess_review(
    args: argparse.Namespace,
    *,
    child_run_id: str,
    point_id: str,
    ordinal: int,
    total: int,
    root: Path,
) -> Dict[str, Any]:
    child_output = root / "qwen_reviews" / f"{child_run_id}.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--job-id",
        str(args.job_id),
        "--point-id",
        str(point_id),
        "--run-id",
        child_run_id,
        "--count",
        "1",
        "--max-turns",
        str(int(args.max_turns or 10)),
        "--visual-limit",
        "1",
    ]
    if args.model_id:
        cmd.extend(["--model-id", str(args.model_id)])
    if args.enable_local_consensus:
        cmd.append("--enable-local-consensus")
    if args.enable_class_concept_briefs:
        cmd.append("--enable-class-concept-briefs")
    if args.skip_limited_final_review:
        cmd.append("--skip-limited-final-review")
    if args.allow_poor_final_review:
        cmd.append("--allow-poor-final-review")
    if args.skip_poor_final_review:
        cmd.append("--skip-poor-final-review")
    if args.skip_cue_verifier:
        cmd.append("--skip-cue-verifier")
    if int(args.mlx_reset_every) >= 0:
        cmd.extend(["--mlx-reset-every", str(max(0, int(args.mlx_reset_every)))])
    if args.reset_qwen_after_review:
        cmd.append("--reset-qwen-after-review")
    for value in args.source_backend_tier or []:
        cmd.extend(["--source-backend-tier", str(value)])
    for value in args.source_decision or []:
        cmd.extend(["--source-decision", str(value)])
    for value in args.source_disposition or []:
        cmd.extend(["--source-disposition", str(value)])
    for value in args.source_disposition_signal or []:
        cmd.extend(["--source-disposition-signal", str(value)])
    if args.source_guarded_only:
        cmd.append("--source-guarded-only")
    if args.source_reviewable_only:
        cmd.append("--source-reviewable-only")
    started = time.time()
    print(f"[{ordinal}/{total}] {point_id} -> subprocess {child_run_id}")
    timeout_seconds = max(1, int(args.review_timeout_seconds or 1))
    try:
        completed = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        if exc.stdout:
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout)
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        if exc.stderr:
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr)
            print(stderr, end="" if stderr.endswith("\n") else "\n")
        return {
            "ordinal": ordinal,
            "review_id": child_run_id,
            "point_id": point_id,
            "status": "failed",
            "error": f"review_timeout_after_{timeout_seconds}s",
            "elapsed_seconds": time.time() - started,
            "decision": "missing",
            "subprocess_run_id": child_run_id,
            "subprocess_returncode": "timeout",
        }
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n")
    if completed.returncode == 0 and child_output.is_file():
        child_payload = _load_json(child_output)
        child_records = child_payload.get("records") if isinstance(child_payload.get("records"), list) else []
        if child_records and isinstance(child_records[0], dict):
            record = dict(child_records[0])
            record["ordinal"] = ordinal
            record["subprocess_run_id"] = child_run_id
            record["subprocess_returncode"] = completed.returncode
            return record
    return {
        "ordinal": ordinal,
        "review_id": child_run_id,
        "point_id": point_id,
        "status": "failed",
        "error": f"subprocess_returncode_{completed.returncode}",
        "elapsed_seconds": time.time() - started,
        "decision": "missing",
        "subprocess_run_id": child_run_id,
        "subprocess_returncode": completed.returncode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Class Split Qwen review benchmark.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--source-run",
        default="",
        help="Optional previous benchmark JSON whose point order should be replayed. "
        "When omitted, the runner samples the current parent job's likely-wrong candidates.",
    )
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--source-backend-tier",
        action="append",
        default=[],
        help="Filter --source-run records by backend_tier before start/count slicing. Repeat or comma-separate values.",
    )
    parser.add_argument(
        "--source-decision",
        action="append",
        default=[],
        help="Filter --source-run records by final decision before start/count slicing. Repeat or comma-separate values.",
    )
    parser.add_argument(
        "--source-disposition",
        action="append",
        default=[],
        help="Filter --source-run records by review_disposition.disposition before start/count slicing.",
    )
    parser.add_argument(
        "--source-disposition-signal",
        action="append",
        default=[],
        help="Filter --source-run records by review_disposition.signal before start/count slicing.",
    )
    parser.add_argument(
        "--source-guarded-only",
        action="store_true",
        help="Filter --source-run records to rows with a blocked guarded_recommendation.",
    )
    parser.add_argument(
        "--source-reviewable-only",
        action="store_true",
        help="Filter --source-run records to actionable, guarded_human_triage, or useful_negative rows.",
    )
    parser.add_argument("--run-label", default="state_machine")
    parser.add_argument("--run-id", default="", help="Exact run id to write; bypasses timestamped run-label construction.")
    parser.add_argument("--point-id", action="append", default=[], help="Run exact point id; may be repeated.")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--visual-limit", type=int, default=0)
    parser.add_argument("--enable-local-consensus", action="store_true")
    parser.add_argument("--enable-class-concept-briefs", action="store_true")
    parser.add_argument(
        "--skip-limited-final-review",
        action="store_true",
        help="Preserve the older controller behavior that skips Qwen finalization for limited-quality targets.",
    )
    parser.add_argument(
        "--skip-poor-final-review",
        action="store_true",
        help="Preserve the older controller behavior that skips Qwen finalization for poor-quality targets.",
    )
    parser.add_argument(
        "--allow-poor-final-review",
        action="store_true",
        help="Experimental: let poor-quality targets reach Qwen for guarded human-triage opinions.",
    )
    parser.add_argument(
        "--skip-cue-verifier",
        action="store_true",
        help="Disable the bounded second-pass verifier for guarded clear-target class changes missing positive cues.",
    )
    parser.add_argument(
        "--mlx-reset-every",
        type=int,
        default=-1,
        help="Override Class Split Qwen MLX runtime reset cadence; 0 disables, 1 resets after each model call.",
    )
    parser.add_argument(
        "--reset-qwen-after-review",
        action="store_true",
        help="Unload the Qwen runtime after every vignette review for long-run MLX stability testing.",
    )
    parser.add_argument(
        "--per-review-subprocess",
        action="store_true",
        help="Run each selected vignette in its own Python process so a Metal fault cannot kill the whole benchmark.",
    )
    parser.add_argument(
        "--review-timeout-seconds",
        type=int,
        default=240,
        help="Per-vignette timeout used with --per-review-subprocess; timed-out children become failed records.",
    )
    parser.add_argument("--audit", action="store_true", help="Run the benchmark audit report after writing results.")
    parser.add_argument("--compare-run", default="", help="Optional previous benchmark JSON for audit comparison.")
    parser.add_argument("--fail-on-unsafe", action="store_true", help="Exit non-zero if the audit finds unsafe recommendations.")
    args = parser.parse_args()

    job_id = str(args.job_id or "").strip()
    if not job_id:
        raise SystemExit("job-id is required")
    root = Path("uploads/class_analysis") / job_id
    result_path = root / "result.json"
    if not result_path.is_file():
        raise SystemExit(f"Missing result file: {result_path}")
    print(f"Loading parent result: {result_path}")
    result = _load_json(result_path)
    _register_parent_job(job_id, result, result_path)

    requested_point_ids = [str(item or "").strip() for item in (args.point_id or []) if str(item or "").strip()]
    source_run = Path(args.source_run) if args.source_run else None
    point_ids = requested_point_ids or _source_point_ids(
        source_run,
        args.count,
        args.start,
        backend_tiers=_normalize_filter_values(args.source_backend_tier),
        decisions=_normalize_filter_values(args.source_decision),
        dispositions=_normalize_filter_values(args.source_disposition),
        disposition_signals=_normalize_filter_values(args.source_disposition_signal),
        guarded_only=bool(args.source_guarded_only),
        reviewable_only=bool(args.source_reviewable_only),
    )
    if not point_ids and not requested_point_ids:
        point_ids = _fallback_point_ids(result, args.count, args.start)
    if not point_ids:
        raise SystemExit("No point ids selected.")

    timestamp = int(time.time())
    run_id = str(args.run_id or "").strip() or f"{args.run_label}_{len(point_ids)}_{timestamp}"
    model_id = str(args.model_id or "").strip()
    records: List[Dict[str, Any]] = []
    started_at = time.time()
    print(f"Benchmark {run_id}: {len(point_ids)} reviews on {job_id}")
    if args.per_review_subprocess:
        for ordinal, point_id in enumerate(point_ids, start=1):
            records.append(
                _run_subprocess_review(
                    args,
                    child_run_id=f"{run_id}_child_{ordinal:03d}",
                    point_id=point_id,
                    ordinal=ordinal,
                    total=len(point_ids),
                    root=root,
                )
            )
        _write_benchmark_outputs(
            args,
            root=root,
            job_id=job_id,
            run_id=run_id,
            model_id=model_id,
            records=records,
            started_at=started_at,
        )
        return 0

    for ordinal, point_id in enumerate(point_ids, start=1):
        review_id = f"cqr_{run_id}_{ordinal:03d}"
        review = api.ClassAnalysisQwenReviewJob(
            review_id=review_id,
            parent_job_id=job_id,
            point_id=point_id,
            request={
                "max_turns": int(args.max_turns or 10),
                "model_id": model_id or None,
                "enable_local_consensus_context": bool(args.enable_local_consensus),
                "enable_class_concept_briefs": bool(args.enable_class_concept_briefs),
                "allow_limited_final_review": not bool(args.skip_limited_final_review),
                "allow_poor_final_review": bool(args.allow_poor_final_review) and not bool(args.skip_poor_final_review),
                "enable_cue_verifier": not bool(args.skip_cue_verifier),
                "reset_qwen_runtime_after_review": bool(args.reset_qwen_after_review),
                "benchmark_run_id": run_id,
            },
        )
        if int(args.mlx_reset_every) >= 0:
            review.request["mlx_reset_every"] = max(0, int(args.mlx_reset_every))
        with api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS_LOCK:
            api.CLASS_ANALYSIS_QWEN_REVIEW_JOBS[review_id] = review
        per_started = time.time()
        print(f"[{ordinal}/{len(point_ids)}] {point_id} -> {review_id}")
        api._run_class_analysis_qwen_review_job(review)
        record = _record_from_review(ordinal, review, per_started)
        records.append(record)
        print(
            f"  {record.get('status')} {record.get('decision')} conf={record.get('confidence')} "
            f"schemas={record.get('active_tool_schemas')} router={(record.get('router') or {}).get('action')}"
        )

    _write_benchmark_outputs(
        args,
        root=root,
        job_id=job_id,
        run_id=run_id,
        model_id=model_id,
        records=records,
        started_at=started_at,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
