"""Private local Qwen worker; no legacy application or provider credentials.

The static pipeline preserves first-pane sequencing. This module supplies its
explicit runtime, image, progress and model-selection dependencies. Model I/O
is local to an immutable admin-registered snapshot, never a request path.
"""
from __future__ import annotations

from contextvars import ContextVar
import hashlib
import json
from pathlib import Path, PurePosixPath
import threading
import time


def _verify_snapshot(directory, files):
    """Check a complete, regular-file-only snapshot without following links."""
    from services.hosted.artifacts import open_regular, validate_digest
    root = Path(directory)
    if root.is_symlink() or not root.is_dir() or not isinstance(files, dict) or not files:
        raise ValueError("hosted_qwen_snapshot_invalid")
    root = root.resolve(strict=True)
    actual_paths = set()
    for path in root.rglob("*"):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ValueError("hosted_qwen_snapshot_link_invalid")
        if path.is_file():
            actual_paths.add(path.relative_to(root).as_posix())
    if actual_paths != set(files):
        raise ValueError("hosted_qwen_snapshot_inventory_changed")
    for relative, expected in files.items():
        path = PurePosixPath(relative)
        if not isinstance(relative, str) or path.is_absolute() or path.as_posix() != relative or ".." in path.parts or "\\" in relative:
            raise ValueError("hosted_qwen_snapshot_path_invalid")
        expected = validate_digest(expected)
        digest = hashlib.sha256()
        with open_regular(root / relative) as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("hosted_qwen_snapshot_changed")
    return root


def _load_native(definition, model_id, device, min_pixels, max_pixels):
    """Load the registered architecture locally on the assigned device only."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from services.qwen_model_catalog import qwen_transformers_load_kwargs
    from services.hosted.qwen_caption_pipeline import QwenRuntime as ModelHandle
    path = _verify_snapshot(definition["model_path"], definition["files"])
    adapter = None
    if definition.get("adapter_path"):
        adapter = _verify_snapshot(definition["adapter_path"], definition["adapter_files"])
    options = qwen_transformers_load_kwargs(model_id, device=str(device), device_pref=str(device), torch_module=torch)
    model = AutoModelForImageTextToText.from_pretrained(str(path), local_files_only=True, trust_remote_code=False, **options)
    if not options.get("device_map"):
        model.to(device)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True, is_trainable=False)
    model.eval()
    processor = AutoProcessor.from_pretrained(str(path), min_pixels=min_pixels, max_pixels=max_pixels,
                                             local_files_only=True, trust_remote_code=False)
    return ModelHandle(model=model, processor=processor, model_id=model_id)


def _vision_info(messages):
    from qwen_vl_utils import process_vision_info
    return process_vision_info(messages)


def _default_prompts():
    from models.schemas import QwenPromptConfig, QwenPromptSection
    return QwenPromptConfig(
        bbox=QwenPromptSection(base_prompt=(
            "Output a JSON formatted list of very tight bounding boxes for detections in this {image_type}. "
            "Each entry must contain bbox_2d coordinates in format [x1,y1,x2,y2]. "
            "Use 0-1000 coordinates relative to this image, not pixel coordinates. "
            "Make a single bounding box for each unique instance of the things we want to detect. "
            "The objects we want to detect are: {items}. {extra_context}"),
            default_image_type="image", default_extra_context="Return only JSON, no additional text."),
        point=QwenPromptSection(base_prompt=(
            "Output a JSON formatted list of positive click points with point_2d coordinates in format [x, y]. "
            "Use 0-1000 coordinates relative to this image, not pixel coordinates, for detections in this {image_type}. "
            "Each entry must contain \"point_2d\": [x, y] centered on the object so Segment Anything can turn it into a mask/bbox. "
            "Make one point per object. The objects we want to detect are: {items}. {extra_context}"),
            default_image_type="image", default_extra_context="Respond with JSON only."))


class QwenRuntime:
    def __init__(self, config, device, *, sam_predict=None):
        from services.hosted import qwen_caption_pipeline as pipeline
        from services.qwen_model_catalog import is_qwen_mlx_model_id
        self.config = json.loads(json.dumps(config, allow_nan=False))
        self.device, self.sam_predict = device, sam_predict
        self.models = self.config.get("models")
        self.default_model = self.config.get("default_model_id")
        if (not isinstance(self.models, dict) or not self.models or self.default_model not in self.models
                or len(self.models) > 16 or any(is_qwen_mlx_model_id(name) for name in self.models)):
            raise ValueError("hosted_qwen_models_invalid")
        for key in ("refinement_model_id", "fallback_model_id"):
            if self.config.get(key) and self.config[key] not in self.models:
                raise ValueError("hosted_qwen_model_not_registered")
        self.min_pixels = int(self.config.get("min_pixels", 256 * 28 * 28))
        self.max_pixels = int(self.config.get("max_pixels", 1280 * 28 * 28))
        if not 1 <= self.min_pixels <= self.max_pixels <= 64_000_000:
            raise ValueError("hosted_qwen_pixel_limits_invalid")
        self.loaded, self.cooldowns = {}, {}
        self.progress, self.events, self.io = {}, [], []
        self.io_bytes = self.events_dropped = 0
        self.image = None
        self.closed = False
        self.cancelled = threading.Event()
        self.lock = threading.RLock()
        self.execution_lock = threading.Lock()
        self.window_hook = ContextVar("hosted_qwen_caption_window", default=None)
        self.call_id = ContextVar("hosted_qwen_call_id", default=None)
        bindings = {
            "QWEN_DEVICE_PREF": str(device), "QWEN_MODEL_NAME": self.default_model,
            "_CAPTION_WINDOW_HOOK": self.window_hook,
            "_agent_full_trace_write": self._record,
            "_default_variant": lambda value: "hosted",
            "_effective_qwen_model_id_for_platform": lambda name, platform: self._model_id(name),
            "_ensure_qwen_ready": lambda: self._load(self.default_model),
            "_ensure_qwen_ready_for_caption": self._load,
            "_predict_with_cache": self._predict_sam,
            "_qwen_cancel_requested": self.cancelled.is_set,
            "_qwen_caption_io_input": lambda **values: self._record({"event": "input", **values}),
            "_qwen_caption_io_output": lambda **values: self._record({"event": "output", **values}),
            "_qwen_caption_io_prompt_budget": lambda **values: self._record({"event": "prompt_budget", **values}),
            "_qwen_caption_io_record": self._record,
            "_qwen_caption_loop_cooldown_remaining": self._cooldown_remaining,
            "_qwen_mlx_post_generation_cleanup": self._unsupported_mlx,
            "_qwen_model_local_state": lambda name, platform: {"local": name in self.models, "partial": False},
            "_qwen_progress_begin_output_section": lambda label: self._progress(output_section=label),
            "_qwen_progress_cancelled": lambda message: self._progress(message=message, active=False, phase="cancelled"),
            "_qwen_progress_error": lambda message: self._progress(message=message, active=False, phase="failed"),
            "_qwen_progress_finish": lambda message, **values: self._progress(message=message, active=False, phase="complete", **values),
            "_qwen_progress_start": lambda **values: self._progress(active=True, **values),
            "_qwen_progress_token": lambda text, **values: self._progress(token=text, **values),
            "_qwen_progress_update": self._progress,
            "_raise_if_qwen_cancelled": self._check_cancel,
            "_resolve_qwen_runtime_platform": lambda *args, **kwargs: "transformers",
            "_run_qwen_chat_mlx": self._unsupported_mlx,
            "_run_qwen_inference_mlx": self._unsupported_mlx,
            "_suspend_clip_backbone": self._forbidden_global_unload,
            "_unload_detector_inference": self._forbidden_global_unload,
            "_unload_dinov3_backbone": self._forbidden_global_unload,
            "_unload_non_qwen_runtimes_impl": self._forbidden_global_unload,
            "_unload_qwen_runtime": self._unload,
            "_unload_sam3_text_runtime": self._forbidden_global_unload,
            "active_qwen_metadata": {"model_id": self.default_model},
            "active_qwen_model_id": self.default_model, "active_qwen_model_path": None,
            "predictor_manager": None, "process_vision_info": _vision_info,
            "qwen_caption_loop_cooldowns": self.cooldowns, "qwen_config_lock": self.lock,
            "qwen_device": device, "qwen_llm_call_id_var": self.call_id,
            "qwen_progress_lock": self.lock, "qwen_progress_state": self.progress,
            "qwen_prompt_config": _default_prompts(), "resolve_image_payload": self._resolve_image,
        }
        self.pipeline = pipeline.bind_pipeline(bindings)
        self.native_visual = self.pipeline["_run_qwen_inference"]
        self.native_text = self.pipeline["_run_qwen_text_inference"]
        self.pipeline["_run_qwen_inference"] = self._visual
        self.pipeline["_run_qwen_text_inference"] = self._text
        self.operation = None

    def _generation(self, generate, *args, **kwargs):
        from services.hosted.qwen_caption_pipeline import QwenCaptionLoopDetected
        try:
            return generate(*args, **kwargs)
        except QwenCaptionLoopDetected as exc:
            self._record({"event": "generation_loop_failure", "raw_output": exc.raw_output,
                          "trimmed_output": exc.trimmed_output, "reason": exc.degenerate_reason})
            raise

    def _visual(self, *args, **kwargs):
        runtime = kwargs.get("runtime_override")
        if runtime is not None and not kwargs.get("model_id_override"):
            # Caption windows pass a cached runtime without repeating its ID.
            # Bind provenance and model-specific prompting to that exact model,
            # rather than the worker's unrelated default selection.
            kwargs["model_id_override"] = self._model_id(runtime.model_id)
        text, width, height = self._generation(self.native_visual, *args, **kwargs)
        # Qwen's detection prompt explicitly specifies the 0–1000 protocol.
        # The legacy parser instead received processor pixels, clipping boxes
        # on small images. Keep decoded pixel dimensions for captions only.
        return (text, 1000, 1000) if self.operation == "qwen_infer" else (text, width, height)

    def _text(self, *args, **kwargs):
        runtime = kwargs.get("runtime_override")
        if runtime is not None and not kwargs.get("model_id_override"):
            kwargs["model_id_override"] = self._model_id(runtime.model_id)
        return self._generation(self.native_text, *args, **kwargs)

    def _model_id(self, value):
        if value not in self.models:
            raise ValueError("hosted_qwen_model_not_registered")
        return value

    def _load(self, model_id):
        model_id = self._model_id(model_id)
        self._check_cancel()
        if model_id not in self.loaded:
            self.loaded[model_id] = _load_native(self.models[model_id], model_id, self.device, self.min_pixels, self.max_pixels)
        return self.loaded[model_id]

    def _unload(self):
        # Only this user's Qwen references. No global environment, SAM,
        # classifier, detector or other worker cache reset.
        self.loaded.clear()

    @staticmethod
    def _unsupported_mlx(*args, **kwargs):
        raise ValueError("hosted_qwen_requires_registered_transformers_runtime")

    @staticmethod
    def _forbidden_global_unload(*args, **kwargs):
        raise RuntimeError("hosted_qwen_global_unload_forbidden")

    def _check_cancel(self):
        if self.cancelled.is_set():
            from services.hosted.qwen_caption_pipeline import QwenCancellationRequested
            raise QwenCancellationRequested("qwen_cancelled")

    def _cooldown_remaining(self, key):
        expiry = self.cooldowns.get(key, 0)
        return max(0, float(expiry) - time.time())

    def _resolve_image(self, *args, **kwargs):
        import numpy as np
        if self.image is None:
            raise ValueError("hosted_qwen_image_context_missing")
        return self.image, np.asarray(self.image), "hosted-current-image"

    def _predict_sam(self, array, token, variant, *, image_name=None, **options):
        if self.sam_predict is None or self.image is None:
            raise ValueError("hosted_qwen_sam_not_configured")
        return self.sam_predict(self.image, **options)

    @staticmethod
    def _json_value(value):
        from PIL import Image
        if isinstance(value, Image.Image):
            return {"type": "immutable_image", "width": value.width, "height": value.height}
        if isinstance(value, dict):
            return {str(key): QwenRuntime._json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [QwenRuntime._json_value(item) for item in value]
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        return str(value)

    def _record(self, value):
        clean = self._json_value(value)
        encoded = json.dumps(clean, allow_nan=False)
        if self.io_bytes + len(encoded.encode()) > 8 * 1024 * 1024:
            raise ValueError("hosted_qwen_evidence_limit_exceeded")
        self.io_bytes += len(encoded.encode())
        self.io.append(clean)

    def _progress(self, **values):
        clean = self._json_value(values)
        with self.lock:
            self.progress.update(clean)
            self.events.append(clean)
            if len(self.events) > 256:
                del self.events[0]
                self.events_dropped += 1

    def execute(self, operation, image, payload, context):
        from models.schemas import QwenCaptionRequest, QwenInferenceRequest
        if self.closed or context.get("actor_id") != self.config.get("actor_id"):
            raise ValueError("hosted_qwen_owner_invalid")
        if operation not in {"qwen_caption", "qwen_caption_preview", "qwen_infer"}:
            raise ValueError("hosted_qwen_operation_invalid")
        caption_operation = operation in {"qwen_caption", "qwen_caption_preview"}
        schema = QwenCaptionRequest if caption_operation else QwenInferenceRequest
        fields = set(schema.model_fields)
        # The controller supplies provenance beside the verified working hash.
        # It is never a model request field and browser parameters cannot set it.
        server_fields = {"media_path", "media_sha256", "source_media_hash", "request_id"}
        if 'source_media_hash' in payload:
            from services.hosted.artifacts import validate_digest
            validate_digest(payload['source_media_hash'])
        if set(payload) - fields - server_fields:
            raise ValueError("hosted_qwen_request_fields_unknown")
        values = {key: value for key, value in payload.items() if key in fields}
        values.update(image_base64=None, image_token="hosted-current-image")
        if caption_operation:
            values.update(image_width=image.width, image_height=image.height, unload_others=False)
            for field, configured in (("refinement_model_id", "refinement_model_id"), ("caption_fallback_model_id", "fallback_model_id")):
                if not values.get(field) or values[field] == "auto":
                    values[field] = self.config.get(configured) or self.default_model
            for field in ("model_id", "refinement_model_id", "caption_fallback_model_id"):
                name = values.get(field)
                if name and name not in {"active", "same", "none"}:
                    self._model_id(name)
        with self.execution_lock:
            self.events.clear()
            self.io.clear()
            self.progress.clear()
            self.io_bytes = self.events_dropped = 0
            self.cancelled.clear()
            self.image = image
            self.operation = operation
            outer_window_hook = self.window_hook.get()
            def record_window(x, y, size, caption):
                self._record({"event": "caption_window", "x": x, "y": y, "size": size, "caption": caption})
                if outer_window_hook is not None:
                    outer_window_hook(x, y, size, caption)
            token = self.window_hook.set(record_window)
            try:
                request = schema(**values)
                entrypoint = "qwen_caption_prompt_preview" if operation == "qwen_caption_preview" else operation
                response = self.pipeline[entrypoint](request)
                result = response.model_dump(mode="json")
                result["hosted_evidence"] = {"model_io": self.io.copy(), "progress": self.events.copy(),
                    "progress_events_dropped": self.events_dropped, "models": list(self.loaded)}
                return result
            finally:
                self.window_hook.reset(token)
                self.image = None
                self.operation = None

    def close(self):
        self.cancelled.set()
        self._unload()
        self.closed = True
