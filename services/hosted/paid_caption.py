"""Native caption/QA orchestration over the trusted, metered provider gateway.

Only visual/text generation changes. The existing caption pipeline still owns
window selection, merge/editor passes, grounding checks, recovery and QA topups.
No credentials, HTTP client or local model loader live in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from services.hosted.errors import DomainError
from services.hosted.qwen_adapters import QwenRuntime


class PaidCaptionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1, max_length=160)
    detail: Literal["low", "high", "auto", "original"] = "high"
    max_output_tokens: int = Field(default=2048, ge=16, le=1000000)
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    service_tier: Literal["default", "standard"] = "default"


@dataclass(frozen=True)
class PaidHandle:
    model_id: str
    platform: str = "openai_responses"


class PaidCaptionRuntime(QwenRuntime):
    """A request-private CPU orchestrator; handles describe provider models."""
    def __init__(self, gateway, actor, authority, *, source_media_hash, options,
                 stage_prefix, check=lambda: None, dispatch_guard=None):
        self.gateway, self.actor, self.authority = gateway, actor, authority
        self.source_media_hash, self.options = source_media_hash, options
        self.stage_prefix, self.stage_number, self.check = stage_prefix, 0, check
        self.dispatch_guard = dispatch_guard
        self.fatal = None
        super().__init__({"actor_id": actor.user_id, "default_model_id": options.model,
                          "models": {options.model: {"provider": "openai"}}}, "cpu")
        self.pipeline.update({
            "_resolve_qwen_runtime_platform": lambda *a, **k: "openai_responses",
            "_qwen_runtime_platform_from_runtime": lambda handle: handle.platform,
            "_qwen_runtime_model_id": lambda handle: handle.model_id,
            "_qwen_model_local_state": lambda *a, **k: {"local": False, "partial": False},
            "_resolve_qwen_caption_default_model_id": lambda *a, **k: self.default_model,
            "_qwen_caption_default_refinement_model_id": lambda *a, **k: self.default_model,
        })

    def _load(self, model_id):
        model_id = self._model_id(model_id)
        self.loaded.setdefault(model_id, PaidHandle(model_id))
        return self.loaded[model_id]

    def _check_cancel(self):
        if self.fatal is not None:
            raise self.fatal
        self.check()
        super()._check_cancel()

    def _stage(self, prompt, image, max_new_tokens, system_prompt, model_id, runtime):
        self._check_cancel()
        selected = model_id or getattr(runtime, "model_id", None) or self.default_model
        self._model_id(selected)
        self.stage_number += 1
        stage_id = "caption-stage-" + hashlib.sha256(f"{self.stage_prefix}:{self.stage_number}".encode()).hexdigest()
        image_hash = None
        if image is not None:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_hash = hashlib.sha256(buffer.getvalue()).hexdigest()
        limit = min(int(max_new_tokens or self.options.max_output_tokens), self.options.max_output_tokens)
        self._record({"event": "input", "call_id": stage_id, "runtime_platform": "openai_responses",
            "model_id": selected, "prompt_text": prompt, "system_prompt": system_prompt or "",
            "image_width": image.width if image else None, "image_height": image.height if image else None,
            "source_media_sha256": self.source_media_hash, "derived_image_sha256": image_hash,
            "max_new_tokens": limit})
        try:
            while True:
                self._check_cancel()
                try:
                    result = self.gateway.stage(self.actor, self.authority, stage_id=stage_id, model=selected,
                        user_prompt=prompt, system_prompt=system_prompt or "", image=image,
                        source_media_hash=self.source_media_hash, max_output_tokens=limit,
                        detail=self.options.detail, reasoning_effort=self.options.reasoning_effort,
                        service_tier=self.options.service_tier, dispatch_guard=self.dispatch_guard)
                    break
                except DomainError as exc:
                    if exc.code != "runtime_automation_paused":
                        raise
                    # The gateway keeps this undispatched reservation. Reuse
                    # its exact stage ID after the controller's pause wait.
        except Exception as exc:
            # Native recovery may catch generation exceptions. A transport or
            # authority failure cannot become another paid stage automatically.
            self.fatal = exc
            # A pause can hold an undispatched reservation. If cancellation or
            # expired authority aborts the wait, release only that reservation;
            # dispatched/unknown/settled calls retain their accounting state.
            if hasattr(self.gateway, "store"):
                with self.gateway.store.read_connection() as con:
                    pending = con.execute("SELECT id FROM hosted_provider_calls WHERE actor_id=? AND operation_id=? AND state='reserved'",
                        (self.actor.user_id, stage_id)).fetchone()
                if pending is not None:
                    self.gateway.accounts.release_reserved(pending["id"], getattr(exc, "code", "caption_stage_aborted"))
            raise
        self._record({"event": "output", "call_id": stage_id, "source": "openai_inference" if image else "openai_text_inference",
            "runtime_platform": "openai_responses", "model_id": selected,
            "output_text": result["caption"], "usage": result.get("usage"),
            "provider_call_id": result["call_id"], "response_id": result.get("provider_response_id"),
            "truncated": result.get("truncated", False)})
        self._check_cancel()
        return result["caption"], image.width if image else 0, image.height if image else 0

    def _visual(self, prompt, pil_img, max_new_tokens=None, system_prompt_override=None,
                model_id_override=None, runtime_override=None, decode_override=None, chat_template_kwargs=None):
        return self._stage(prompt, pil_img, max_new_tokens, system_prompt_override, model_id_override, runtime_override)

    def _text(self, prompt, *, max_new_tokens=None, system_prompt_override=None,
              model_id_override=None, runtime_override=None, decode_override=None, chat_template_kwargs=None):
        return self._stage(prompt, None, max_new_tokens, system_prompt_override, model_id_override, runtime_override)

    def execute(self, operation, image, payload, context):
        try:
            result = super().execute(operation, image, payload, context)
            if self.fatal is not None:
                raise self.fatal
            return result
        except Exception:
            if self.fatal is not None:
                raise self.fatal
            raise


def validate_paid_request(gateway, actor, dataset_id, caption_request, options):
    """Validate configured transport and preserve explicitly selected models."""
    if gateway is None:
        raise DomainError("paid_provider_not_configured", 503)
    model = next((item for item in gateway.catalog(actor, dataset_id)["models"] if item["id"] == options.model), None)
    if not model or options.max_output_tokens > model["max_output_tokens"]:
        raise DomainError("paid_provider_model_or_limit_not_allowed", 403)
    if options.detail not in model.get("image_details", ["low", "high", "auto"]):
        raise DomainError("paid_provider_image_detail_not_allowed", 422)
    for field in ("model_id", "refinement_model_id", "caption_fallback_model_id"):
        if caption_request.get(field) not in {None, "", "active", "auto", "same", "none", options.model}:
            raise DomainError("paid_caption_local_model_selected", 422)
    explicit = caption_request.get("max_new_tokens")
    if explicit is not None and explicit > options.max_output_tokens:
        raise DomainError("paid_caption_output_limit", 422)
    if any(caption_request.get(field) is not None for field in ("temperature", "top_p", "top_k", "presence_penalty")):
        raise DomainError("paid_caption_local_sampling_unsupported", 422)
    identity = {"provider": "openai", "options": options.model_dump(), "pricing": gateway.pricing.get(options.model)}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
