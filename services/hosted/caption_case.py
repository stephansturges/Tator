"""One real caption/QA attempt using the existing benchmark's scoring helpers."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict, Field


class CaptionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)
    attempts: int = Field(default=2, ge=1, le=100)
    per_image_timeout_seconds: float = Field(default=900, gt=0, le=86400)
    cooldown_after_crash_seconds: float = Field(default=5, ge=0, le=300)
    continue_on_quality_failures: bool = False
    set_and_forget: bool = True
    max_auto_resumes: int = Field(default=2, ge=0, le=100)
    preview_only: bool = False
    max_failed_case_rate: float = Field(default=0, ge=-1, le=1)
    max_quality_failure_rate: float = Field(default=0, ge=-1, le=1)
    max_recovery_event_case_rate: float = Field(default=0.25, ge=-1, le=1)
    max_loop_recovery_case_rate: float = Field(default=0, ge=-1, le=1)
    max_loop_guard_case_rate: float = Field(default=0, ge=-1, le=1)
    max_deterministic_recovery_case_rate: float = Field(default=0, ge=-1, le=1)
    max_failed_attempt_row_rate: float = Field(default=0.25, ge=-1, le=1)
    max_signal_exit_attempt_row_rate: float = Field(default=0, ge=-1, le=1)
    min_rate_cases: int = Field(default=20, ge=1, le=1000000)
    instruction_dataset: bool = False
    subcaptions_per_image: int = Field(default=0, ge=0, le=20)
    include_caption0_in_training: bool = True
    include_generated_qa_in_training: bool = True
    include_source_annotations_in_generator_context: bool = True
    strict_grounding: bool = True
    qa_mix: str = Field(default="balanced", max_length=80)
    answer_format: str = Field(default="natural", max_length=80)
    instruction_qa_imposed_questions: list[str] = Field(default_factory=list, max_length=20)
    instruction_qa_max_topup_attempts: int = Field(default=6, ge=0, le=12)
    instruction_qa_restrict_speculative_language: bool = False


def execute_caption_case(runtime, image, payload, context):
    """Keep preview, generation, QA verification and evidence in one worker.

    The control service supplies the immutable image path and canonical case;
    no benchmark filesystem label writer or legacy API import is called.
    """
    from tools import run_qwen_caption_flow_benchmark as runner
    settings = CaptionSettings(**payload["settings"])
    caption_request = dict(payload["caption_request"])
    case = payload["case"]
    if settings.include_source_annotations_in_generator_context:
        caption_request["label_hints"] = case["label_hints"]
    preview = runtime.execute("qwen_caption_preview", image, caption_request, context)
    if settings.preview_only:
        return {"status": "preview_only", "preview": preview}
    response = runtime.execute("qwen_caption", image, caption_request, context)
    text = str(response.get("caption") or "").strip()
    result = {"status": "ok", "preview": preview, "response": response}
    args = SimpleNamespace(**{**runner.RUN_SETTINGS_ARG_DEFAULTS, **settings.model_dump(),
        "model_id": caption_request.get("model_id") or runtime.default_model})
    if args.model_id in {"active", "auto"}:
        args.model_id = runtime.default_model
    # Existing helpers read glossary/request fields from a file. This private
    # temporary transport contains only the already validated request.
    with tempfile.TemporaryDirectory(prefix="hosted-caption-case-") as temporary:
        request_path = Path(temporary) / "request.json"
        request_path.write_text(json.dumps(caption_request, allow_nan=False))
        args.request_json = request_path
        result["caption_quality"] = runner.summarize_caption(text,
            {str(key): int(value) for key, value in response.get("used_counts", {}).items()},
            runner._case_glossary_map(case, args))
        if settings.instruction_dataset and runner._instruction_qa_requested_count(args) > 0:
            with runtime.execution_lock:
                runtime.image, runtime.operation = image, "qwen_caption"
                try:
                    qa = runner.generate_instruction_qa_pairs(SimpleNamespace(**runtime.pipeline),
                        case, Path(payload["media_path"]), text, args)
                finally:
                    runtime.image, runtime.operation = None, None
            result["instruction_qa"] = qa
            response.update(generated_qa_pairs=list(qa.get("pairs") or []),
                generated_qa_rejected_pairs=list(qa.get("rejected_pairs") or []),
                generated_qa_pair_count=int(qa.get("pair_count") or 0),
                generated_qa_target_pair_count=int(qa.get("target_pair_count") or qa.get("requested") or 0),
                generated_qa_attempt_summary=list(qa.get("attempt_summary") or []))
            if not response["generated_qa_pair_count"] and runner._generated_qa_required_for_worker_success(args):
                result["status"] = "instruction_qa_failed"
            elif qa.get("underfilled"):
                response["generated_qa_warning"] = {"type": "InstructionQaUnderfilledWarning",
                    "accepted_pair_count": response["generated_qa_pair_count"],
                    "target_pair_count": response["generated_qa_target_pair_count"]}
    response["hosted_evidence"] = {"model_io": runtime.io.copy(), "progress": runtime.events.copy(),
        "progress_events_dropped": runtime.events_dropped, "models": list(runtime.loaded)}
    return result
