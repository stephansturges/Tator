from __future__ import annotations

import base64, hashlib, io, zipfile, math, uuid, os, tempfile, shutil, time, logging, subprocess, sys, json, re
from copy import deepcopy
from pathlib import Path
import numpy as np
from typing import Optional, List, Dict, Tuple, Any, Literal, Sequence
import torch, clip, joblib
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, root_validator, Field
import psutil
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_412_PRECONDITION_FAILED,
    HTTP_404_NOT_FOUND,
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_428_PRECONDITION_REQUIRED,
    HTTP_409_CONFLICT,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)
from collections import OrderedDict
from segment_anything import sam_model_registry, SamPredictor
import threading
import queue
import itertools
from dataclasses import dataclass, field, asdict

# Compatibility shim for older pickled sklearn artifacts that reference numpy._core
if "numpy._core" not in sys.modules:  # pragma: no cover - import guard
    try:
        import numpy.core as _np_core
        sys.modules["numpy._core"] = _np_core
    except Exception:
        pass

from tools.clip_training import train_clip_from_yolo, TrainingError, TrainingArtifacts
try:
    from tools.qwen_training import (
        QwenTrainingConfig,
        QwenTrainingResult,
        train_qwen_model,
        TrainingError as QwenTrainingError,
        DEFAULT_SYSTEM_PROMPT,
    )
except Exception as exc:  # noqa: BLE001
    QWEN_TRAINING_IMPORT_ERROR = exc
    QwenTrainingConfig = None  # type: ignore[assignment]
    QwenTrainingResult = None  # type: ignore[assignment]
    train_qwen_model = None  # type: ignore[assignment]
    QwenTrainingError = TrainingError  # type: ignore[assignment]
    DEFAULT_SYSTEM_PROMPT = (
        "You are an annotation assistant that only returns JSON objects shaped like {\"detections\":[{\"label\":\"class\","
        "\"bbox\":[x1,y1,x2,y2]} or {\"label\":\"class\",\"point\":[x,y]}]}"
    )
else:
    QWEN_TRAINING_IMPORT_ERROR = None

try:
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        Qwen2_5_VLProcessor,
    )
    from qwen_vl_utils import process_vision_info
except Exception as exc:  # noqa: BLE001
    QWEN_IMPORT_ERROR = exc
    Qwen2_5_VLForConditionalGeneration = None  # type: ignore[assignment]
    AutoProcessor = None  # type: ignore[assignment]
    Qwen2_5_VLProcessor = None  # type: ignore[assignment]
    process_vision_info = None  # type: ignore[assignment]
else:
    QWEN_IMPORT_ERROR = None

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor as Sam3ImageProcessor
except Exception as exc:  # noqa: BLE001
    SAM3_NATIVE_IMAGE_IMPORT_ERROR = exc
    build_sam3_image_model = None  # type: ignore[assignment]
    Sam3ImageProcessor = None  # type: ignore[assignment]
else:
    SAM3_NATIVE_IMAGE_IMPORT_ERROR = None

try:
    from peft import PeftModel
except Exception as exc:  # noqa: BLE001
    PEFT_IMPORT_ERROR = exc
    PeftModel = None  # type: ignore[assignment]
else:
    PEFT_IMPORT_ERROR = None

MAX_PREDICTOR_SLOTS = 3


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


QWEN_MODEL_NAME = os.environ.get("QWEN_MODEL_NAME", "Qwen/Qwen2.5-VL-3B-Instruct")
QWEN_MIN_PIXELS = _env_int("QWEN_MIN_PIXELS", 256 * 28 * 28)
QWEN_MAX_PIXELS = _env_int("QWEN_MAX_PIXELS", 1280 * 28 * 28)
QWEN_MAX_NEW_TOKENS = _env_int("QWEN_MAX_NEW_TOKENS", 1024)
QWEN_DO_SAMPLE = _env_bool("QWEN_DO_SAMPLE", False)
QWEN_TEMPERATURE = _env_float("QWEN_TEMPERATURE", 0.2)
QWEN_TOP_P = _env_float("QWEN_TOP_P", 0.9)
QWEN_DEVICE_PREF = os.environ.get("QWEN_DEVICE", "auto").strip().lower()

qwen_model = None
qwen_processor = None
qwen_device: Optional[str] = None
qwen_last_error: Optional[str] = None
qwen_lock = threading.RLock()
qwen_config_lock = threading.RLock()

QWEN_METADATA_FILENAME = "metadata.json"


def _default_qwen_metadata() -> Dict[str, Any]:
    return {
        "id": "default",
        "label": "Base Qwen 2.5",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "dataset_context": "",
        "classes": [],
        "model_id": QWEN_MODEL_NAME,
        "source": "huggingface",
        "max_image_dim": 1024,
        "max_detections_per_sample": 200,
    }


active_qwen_model_id = "default"
active_qwen_model_path: Optional[Path] = None
active_qwen_metadata: Dict[str, Any] = _default_qwen_metadata()
loaded_qwen_model_id: Optional[str] = None


def _reset_qwen_runtime() -> None:
    global qwen_model, qwen_processor, qwen_last_error, loaded_qwen_model_id, qwen_device
    qwen_model = None
    qwen_processor = None
    qwen_device = None
    loaded_qwen_model_id = None
    qwen_last_error = None
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


sam3_text_model = None
sam3_text_processor = None
sam3_text_device: Optional[torch.device] = None
sam3_text_lock = threading.RLock()


def _set_active_qwen_model_default() -> None:
    global active_qwen_model_id, active_qwen_model_path, active_qwen_metadata
    active_qwen_model_id = "default"
    active_qwen_model_path = None
    active_qwen_metadata = _default_qwen_metadata()
    _reset_qwen_runtime()


def _set_active_qwen_model_custom(model_id: str, ckpt_path: Path, metadata: Dict[str, Any]) -> None:
    global active_qwen_model_id, active_qwen_model_path, active_qwen_metadata
    active_qwen_model_id = model_id
    active_qwen_model_path = ckpt_path
    active_qwen_metadata = metadata or {}
    active_qwen_metadata.setdefault("id", model_id)
    _reset_qwen_runtime()


def _prepare_for_qwen_training() -> None:
    try:
        predictor_manager.unload_all()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to unload SAM predictors before training: %s", exc)
    _reset_qwen_runtime()
    _suspend_clip_backbone()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _finalize_qwen_training_environment() -> None:
    _resume_clip_backbone()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _bytes_to_mb(value: int) -> float:
    return round(value / (1024 * 1024), 2)

# ----------------------------------------------------------------
# 1) Define a global error message and a global load-flag for CLIP
ERROR_MESSAGE = 0 # messy hack, making this an int because of the way we parse it later... the message has actually just been moved to the JS and appears when bbox uuid is None
clip_initialized = True
# ----------------------------------------------------------------

# 2) Attempt to load the logistic regression model (.pkl)
MODEL_PATH = "./my_logreg_model.pkl"
clf = None
if os.path.exists(MODEL_PATH):
    try:
        print("Loading logistic regression...")
        clf = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load logistic regression model: {e}")
        clip_initialized = False
else:
    print(f"File {MODEL_PATH} not found.")
    clip_initialized = False

LABELMAP_DEFAULT_PATH = "./my_label_list.pkl"
active_classifier_path: Optional[str] = MODEL_PATH if clf is not None else None
active_labelmap_path: Optional[str] = LABELMAP_DEFAULT_PATH if os.path.exists(LABELMAP_DEFAULT_PATH) else None
active_label_list: List[str] = []
if active_labelmap_path:
    try:
        if active_labelmap_path.lower().endswith(".pkl"):
            loaded = joblib.load(active_labelmap_path)
            if isinstance(loaded, list):
                active_label_list = [str(item) for item in loaded]
            else:
                active_labelmap_path = None
        else:
            with open(active_labelmap_path, "r", encoding="utf-8") as handle:
                active_label_list = [line.strip() for line in handle if line.strip()]
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load labelmap {active_labelmap_path}: {exc}")
        active_labelmap_path = None
        active_label_list = []

# 3) Attempt to load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to enable TF32: %s", exc)
SUPPORTED_CLIP_MODELS = [
    "ViT-B/32",
    "ViT-B/16",
    "ViT-L/14",
]
DEFAULT_CLIP_MODEL = SUPPORTED_CLIP_MODELS[0]

clip_model = None
clip_preprocess = None
clip_model_name: Optional[str] = None
_clip_reload_needed = False
try:
    print("Loading CLIP model...")
    clip_model, clip_preprocess = clip.load(DEFAULT_CLIP_MODEL, device=device)
    clip_model_name = DEFAULT_CLIP_MODEL
except Exception as e:
    print(f"Failed to load CLIP model: {e}")
    clip_initialized = False
    clip_model_name = None

clip_lock = threading.Lock()
if clip_model is None or clf is None:
    clip_initialized = False


def _suspend_clip_backbone() -> None:
    global clip_model, clip_preprocess, clip_initialized, _clip_reload_needed
    with clip_lock:
        if clip_model is None:
            return
        logger.info("Suspending CLIP backbone to free GPU memory for training.")
        clip_model = None
        clip_preprocess = None
        clip_initialized = False
        _clip_reload_needed = True


def _resume_clip_backbone() -> None:
    global clip_model, clip_preprocess, clip_initialized, _clip_reload_needed
    if not _clip_reload_needed:
        return
    with clip_lock:
        if clip_model is not None:
            _clip_reload_needed = False
            clip_initialized = True
            return
        clip_name = clip_model_name or DEFAULT_CLIP_MODEL
        try:
            clip_model, clip_preprocess = clip.load(clip_name, device=device)
            clip_initialized = bool(clf is not None and clip_model is not None)
            logger.info("Reloaded CLIP backbone %s after training.", clip_name)
        except Exception as exc:  # noqa: BLE001
            clip_model = None
            clip_preprocess = None
            clip_initialized = False
            logger.warning("Failed to reload CLIP backbone %s: %s", clip_name, exc)
        finally:
            _clip_reload_needed = False

# 4) Load the SAM model (segment-anything) as normal:
MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "vit_h")
CHECKPOINT_PATH = os.environ.get("SAM_CHECKPOINT_PATH", "./sam_vit_h_4b8939.pth")
SAM3_MODEL_ID = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")
SAM3_PROCESSOR_ID = os.environ.get("SAM3_PROCESSOR_ID", SAM3_MODEL_ID)
SAM3_CHECKPOINT_PATH = os.environ.get("SAM3_CHECKPOINT_PATH")
SAM3_DEVICE_PREF = os.environ.get("SAM3_DEVICE", "auto").strip().lower()


def _resolve_sam3_device() -> torch.device:
    if SAM3_DEVICE_PREF in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(SAM3_DEVICE_PREF)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"invalid_sam3_device:{SAM3_DEVICE_PREF}:{exc}") from exc


class _Sam1Backend:
    def __init__(self):
        self.predictor = SamPredictor(sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH))

    def set_image(self, np_img: np.ndarray) -> None:
        self.predictor.set_image(np_img)

    def predict(self, **kwargs):
        return self.predictor.predict(**kwargs)

    def unload(self) -> None:
        try:
            del self.predictor
        except Exception:  # noqa: BLE001
            pass
        self.predictor = None


class _Sam3Backend:
    def __init__(self):
        if SAM3_NATIVE_IMAGE_IMPORT_ERROR is not None or build_sam3_image_model is None:
            raise RuntimeError(f"sam3_unavailable:{SAM3_NATIVE_IMAGE_IMPORT_ERROR}")
        self.device = _resolve_sam3_device()
        device_str = "cuda" if self.device.type == "cuda" else "cpu"
        try:
            model = build_sam3_image_model(
                device=device_str,
                checkpoint_path=SAM3_CHECKPOINT_PATH,
                load_from_HF=SAM3_CHECKPOINT_PATH is None,
                enable_inst_interactivity=True,
            )
            if self.device:
                model = model.to(self.device)
            predictor = getattr(model, "inst_interactive_predictor", None)
            if predictor is None:
                raise RuntimeError("sam3_interactive_predictor_missing")
            tracker = getattr(predictor, "model", None)
            if tracker is None:
                raise RuntimeError("sam3_tracker_missing")
            if getattr(tracker, "backbone", None) is None:
                tracker.backbone = model.backbone
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"sam3_load_failed:{exc}") from exc
        self.model = model
        self.predictor = predictor

    def set_image(self, np_img: np.ndarray) -> None:
        arr = np.ascontiguousarray(np_img)
        self.predictor.set_image(arr)

    def predict(self, **kwargs):
        point_coords = kwargs.get("point_coords")
        point_labels = kwargs.get("point_labels")
        box = kwargs.get("box")
        mask_input = kwargs.get("mask_input")
        multimask_output = kwargs.get("multimask_output", True)
        return self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
        )

    def unload(self) -> None:
        try:
            del self.predictor
        except Exception:  # noqa: BLE001
            pass
        try:
            del self.model
        except Exception:  # noqa: BLE001
            pass
        self.predictor = None
        self.model = None
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass


def _build_backend_for_variant(variant: str):
    normalized = (variant or "sam1").lower()
    if normalized == "sam3":
        return _Sam3Backend()
    # default to classic SAM1 backend
    return _Sam1Backend()


def _ensure_sam3_text_runtime():
    global sam3_text_model, sam3_text_processor, sam3_text_device
    with sam3_text_lock:
        if sam3_text_model is not None and sam3_text_processor is not None and sam3_text_device is not None:
            return sam3_text_model, sam3_text_processor, sam3_text_device
        device = _resolve_sam3_device()
        if SAM3_NATIVE_IMAGE_IMPORT_ERROR is not None or build_sam3_image_model is None or Sam3ImageProcessor is None:
            detail = f"sam3_text_unavailable:{SAM3_NATIVE_IMAGE_IMPORT_ERROR}"
            raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
        try:
            device_str = "cuda" if device.type == "cuda" else "cpu"
            if SAM3_CHECKPOINT_PATH:
                model = build_sam3_image_model(
                    checkpoint_path=SAM3_CHECKPOINT_PATH,
                    device=device_str,
                    load_from_HF=False,
                ).to(device)
            else:
                model = build_sam3_image_model(device=device_str).to(device)
            processor = Sam3ImageProcessor(model)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"sam3_text_load_failed:{exc}") from exc
        sam3_text_model = model
        sam3_text_processor = processor
        sam3_text_device = device
        return sam3_text_model, sam3_text_processor, sam3_text_device


class PredictorSlot:
    def __init__(self, name: str):
        self.name = name
        self.backends: Dict[str, Any] = {}
        self.token: Optional[str] = None
        self.variant: Optional[str] = None
        self.image_shape: Optional[Tuple[int, int, int]] = None
        self.image_name: Optional[str] = None
        self.last_loaded: float = 0.0
        self.lock = threading.RLock()
        self._busy = threading.Event()
        self.image_memory_bytes: int = 0

    def set_image(self, np_img: np.ndarray, token: Optional[str], variant: Optional[str], image_name: Optional[str]) -> None:
        variant_name = (variant or "sam1").lower()
        with self.lock:
            self._busy.set()
            try:
                backend = self._ensure_backend(variant_name)
                backend.set_image(np_img)
                self.token = token
                self.variant = variant_name
                self.image_shape = np_img.shape
                self.image_name = image_name
                self.last_loaded = time.time()
                self.image_memory_bytes = int(np_img.nbytes)
            finally:
                self._busy.clear()

    def predict(self, **kwargs):
        with self.lock:
            self._busy.set()
            try:
                backend = self._ensure_backend((self.variant or "sam1").lower())
                return backend.predict(**kwargs)
            finally:
                self._busy.clear()

    def is_busy(self) -> bool:
        return self._busy.is_set()

    def clear(self) -> None:
        with self.lock:
            self.token = None
            self.variant = None
            self.image_shape = None
            self.image_name = None
            self.last_loaded = 0.0
            self.image_memory_bytes = 0

    def unload(self) -> None:
        with self.lock:
            self.clear()
            for backend in self.backends.values():
                try:
                    backend.unload()
                except Exception:  # noqa: BLE001
                    pass
            self.backends.clear()

    def _ensure_backend(self, variant: str):
        backend = self.backends.get(variant)
        if backend is None:
            backend = _build_backend_for_variant(variant)
            self.backends[variant] = backend
        return backend


class PredictorManager:
    def __init__(self):
        self.slots: Dict[str, PredictorSlot] = {
            "current": PredictorSlot("current"),
            "next": PredictorSlot("next"),
            "previous": PredictorSlot("previous"),
        }
        self.slot_order: List[str] = ["current", "next", "previous"]
        self.capacity_lock = threading.RLock()
        self.capacity: int = min(MAX_PREDICTOR_SLOTS, len(self.slot_order))
        self.enabled_slots: set[str] = set(self.slot_order[: self.capacity])
        self.token_index: Dict[Tuple[str, str], PredictorSlot] = {}
        self.image_index: Dict[Tuple[str, str], PredictorSlot] = {}
        self.queue: "queue.Queue[Tuple[str, Dict[str, Any]]]" = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker, name="predictor-preload-worker", daemon=True)
        self.worker.start()

    def _slot_key(self, token: Optional[str], variant: Optional[str]) -> Optional[Tuple[str, str]]:
        if not token or not variant:
            return None
        return (token, variant)

    def _image_key(self, image_name: Optional[str], variant: Optional[str]) -> Optional[Tuple[str, str]]:
        if not image_name or not variant:
            return None
        return (variant, image_name)

    def is_slot_enabled(self, slot_name: str) -> bool:
        return slot_name in self.enabled_slots

    def resolve_slot(self, slot_name: Optional[str], *, allow_disabled_fallback: bool = True) -> str:
        """Return a normalised slot name.

        When ``allow_disabled_fallback`` is False we fail fast if the requested
        slot is currently disabled instead of silently falling back to the
        "current" slot. This prevents background preloads from clobbering the
        user's active predictor when the capacity shrinks.
        """

        candidate = (slot_name or "current").lower()
        if candidate not in self.slots:
            return "current"
        if self.is_slot_enabled(candidate):
            return candidate
        if allow_disabled_fallback:
            return "current"
        raise ValueError(f"slot_disabled:{candidate}")

    def capacity_limits(self) -> Tuple[int, int]:
        return (1, min(MAX_PREDICTOR_SLOTS, len(self.slot_order)))

    def get_capacity(self) -> int:
        with self.capacity_lock:
            return self.capacity

    def set_capacity(self, capacity: int) -> None:
        minimum, maximum = self.capacity_limits()
        normalized = max(minimum, min(maximum, capacity))
        with self.capacity_lock:
            if normalized == self.capacity:
                return
            self.capacity = normalized
            new_enabled = set(self.slot_order[: normalized])
            disabled = self.enabled_slots - new_enabled
            self.enabled_slots = new_enabled
            for slot_name in disabled:
                slot = self.slots.get(slot_name)
                if slot:
                    self._clear_slot_refs(slot)
                    slot.clear()

    def active_slot_count(self) -> int:
        return len(self.enabled_slots)

    def loaded_slot_count(self) -> int:
        return sum(1 for name, slot in self.slots.items() if name in self.enabled_slots and slot.token)

    def total_image_memory_bytes(self) -> int:
        return sum(slot.image_memory_bytes for name, slot in self.slots.items() if name in self.enabled_slots)

    def _clear_slot_refs(self, slot: PredictorSlot) -> None:
        remove_keys = [key for key, value in self.token_index.items() if value is slot]
        for key in remove_keys:
            self.token_index.pop(key, None)
        remove_image_keys = [key for key, value in self.image_index.items() if value is slot]
        for key in remove_image_keys:
            self.image_index.pop(key, None)

    def unload_all(self) -> None:
        with self.capacity_lock:
            for slot in self.slots.values():
                self._clear_slot_refs(slot)
                slot.unload()

    def set_slot(self, slot_name: str, np_img: np.ndarray, token: Optional[str], variant: Optional[str], image_name: Optional[str]) -> None:
        slot_name = self.resolve_slot(slot_name, allow_disabled_fallback=False)
        slot = self.slots[slot_name]
        self._clear_slot_refs(slot)
        slot.set_image(np_img, token, variant, image_name)
        key = self._slot_key(token, variant)
        if key:
            self.token_index[key] = slot
        image_key = self._image_key(image_name, variant)
        if image_key:
            self.image_index[image_key] = slot

    def ensure_current(self, np_img: np.ndarray, token: Optional[str], variant: Optional[str], image_name: Optional[str]) -> PredictorSlot:
        slot = self.token_index.get(self._slot_key(token, variant)) if token and variant else None
        if slot and slot.name == "current":
            return slot
        self.set_slot("current", np_img, token, variant, image_name)
        return self.slots["current"]

    def get_slot_for_token(self, token: Optional[str], variant: Optional[str]) -> Optional[PredictorSlot]:
        key = self._slot_key(token, variant)
        if key is None:
            return None
        return self.token_index.get(key)

    def get_slot_for_image(self, image_name: Optional[str], variant: Optional[str]) -> Optional[PredictorSlot]:
        key = self._image_key(image_name, variant)
        if key is None:
            return None
        return self.image_index.get(key)

    def promote_slot(self, slot_name: str) -> bool:
        if slot_name not in self.slots or slot_name == "current" or not self.is_slot_enabled(slot_name):
            return False
        if slot_name == "next":
            prev_slot = self.slots["previous"]
            curr_slot = self.slots["current"]
            next_slot = self.slots["next"]
            self.slots["previous"] = curr_slot
            self.slots["current"] = next_slot
            self.slots["next"] = prev_slot
        elif slot_name == "previous":
            prev_slot = self.slots["previous"]
            curr_slot = self.slots["current"]
            next_slot = self.slots["next"]
            self.slots["next"] = curr_slot
            self.slots["current"] = prev_slot
            self.slots["previous"] = next_slot
        else:
            return False
        self.slots["previous"].name = "previous"
        self.slots["current"].name = "current"
        self.slots["next"].name = "next"
        return True

    def predict(self, np_img: np.ndarray, token: Optional[str], variant: Optional[str], image_name: Optional[str], **predict_kwargs):
        slot = self.get_slot_for_token(token, variant)
        if slot is None:
            slot = self.ensure_current(np_img, token, variant, image_name)
        return slot.predict(**predict_kwargs)

    def set_slot_with_wait(self, slot_name: str, np_img: np.ndarray, token: Optional[str], variant: Optional[str], image_name: Optional[str]) -> None:
        slot_name = self.resolve_slot(slot_name, allow_disabled_fallback=False)
        if slot_name != "current":
            waited = 0.0
            # Give the "current" slot a brief head start so the active image always begins loading first,
            # but do not block background slots for the full duration of set_image.
            while (
                not self.stop_event.is_set()
                and waited < 0.2
                and not self.slots["current"].is_busy()
                and not self.slots["current"].token
            ):
                time.sleep(0.01)
                waited += 0.01
        self.set_slot(slot_name, np_img, token, variant, image_name)

    def stop(self) -> None:
        self.stop_event.set()
        self.worker.join(timeout=1.0)

    def schedule_slot(self, slot_name: str, payload: Dict[str, Any]) -> None:
        self.queue.put((slot_name, payload))

    def status(self) -> List[Dict[str, Any]]:
        info = []
        for name, slot in self.slots.items():
            entry: Dict[str, Any] = {
                "slot": name,
                "token": slot.token,
                "variant": slot.variant,
                "image_name": slot.image_name,
                "last_loaded": slot.last_loaded,
                "busy": slot.is_busy(),
                "enabled": self.is_slot_enabled(name),
                "memory_bytes": slot.image_memory_bytes,
            }
            if slot.image_shape:
                entry["height"] = slot.image_shape[0]
                entry["width"] = slot.image_shape[1]
            info.append(entry)
        return info

    def _materialize(self, payload: Dict[str, Any]) -> Tuple[np.ndarray, str, str, Optional[str]]:
        variant = _default_variant(payload.get("sam_variant"))
        image_name = payload.get("image_name")
        token = payload.get("image_token")
        if token:
            cached = _fetch_preloaded_image(token, variant)
            if cached is not None:
                return cached, token, variant, image_name
        base64_data = payload.get("image_base64")
        if not base64_data:
            raise HTTPException(status_code=HTTP_428_PRECONDITION_REQUIRED, detail="image_payload_missing")
        data = base64.b64decode(base64_data)
        pil_img = Image.open(BytesIO(data)).convert("RGB")
        np_img = np.array(pil_img)
        token = hashlib.md5(np_img.tobytes()).hexdigest()
        _store_preloaded_image(token, np_img, variant)
        return np_img, token, variant, image_name

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                slot_name, payload = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                np_img, token, variant, image_name = self._materialize(payload)
                try:
                    self.set_slot_with_wait(slot_name, np_img, token, variant, image_name)
                except ValueError:
                    # Slot was disabled while this job was in flight; skip.
                    continue
            except Exception as exc:  # noqa: BLE001
                print(f"predictor preload failed: {exc}")


predictor_manager = PredictorManager()

# 5) Threading lock for SAM usage:
sam_lock = threading.Lock()

job_store: Dict[str, List["CropImage"]] = {}

app = FastAPI(title="Local Inference API (Multi-Predictor)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("localinferenceapi")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.propagate = False

# Cache for repeated calls
SAM_CACHE_LIMIT = 8
sam_cache_lock = threading.Lock()
sam_preload_cache: "OrderedDict[str, Tuple[np.ndarray, str]]" = OrderedDict()


def _store_preloaded_image(token: str, np_img: np.ndarray, variant: str) -> None:
    arr = np.ascontiguousarray(np_img)
    with sam_cache_lock:
        sam_preload_cache[token] = (arr, variant)
        sam_preload_cache.move_to_end(token)
        while len(sam_preload_cache) > SAM_CACHE_LIMIT:
            sam_preload_cache.popitem(last=False)


def _fetch_preloaded_image(token: str, variant: str) -> Optional[np.ndarray]:
    with sam_cache_lock:
        item = sam_preload_cache.get(token)
        if not item:
            return None
        arr, stored_variant = item
        if stored_variant != variant:
            return None
        sam_preload_cache.move_to_end(token)
        return arr


_job_id_counter = itertools.count(1)


@dataclass
class SamPreloadJob:
    request_id: int
    variant: str
    slot: str
    generation: Optional[int]
    image_token: Optional[str]
    image_base64: Optional[str]
    image_name: Optional[str]
    event: threading.Event
    result: Optional[SamPreloadResponse] = None
    error: Optional[Exception] = None


class SamPreloadManager:
    def __init__(self):
        self.queue: "queue.Queue[SamPreloadJob]" = queue.Queue()
        self.lock = threading.Lock()
        self.latest_request_id: Dict[Tuple[str, str], int] = {}
        self.latest_generation: Dict[Tuple[str, str], int] = {}
        self.worker = threading.Thread(target=self._worker, name="sam-preload-worker", daemon=True)
        self.worker.start()

    def submit(
        self,
        *,
        variant: str,
        slot: str,
        generation: Optional[int],
        image_token: Optional[str],
        image_base64: Optional[str],
        image_name: Optional[str],
    ) -> SamPreloadResponse:
        job = SamPreloadJob(
            request_id=next(_job_id_counter),
            variant=variant,
            slot=slot,
            generation=generation,
            image_token=image_token,
            image_base64=image_base64,
            image_name=image_name,
            event=threading.Event(),
        )
        key = (variant, slot)
        with self.lock:
            self.latest_request_id[key] = job.request_id
            if generation is not None:
                prev = self.latest_generation.get(key)
                if prev is None or generation > prev:
                    self.latest_generation[key] = generation
        self.queue.put(job)
        job.event.wait()
        if job.error:
            raise job.error
        return job.result  # type: ignore[return-value]

    def _worker(self) -> None:
        while True:
            job = self.queue.get()
            try:
                if self._is_superseded(job):
                    job.result = self._superseded_response(job)
                else:
                    job.result = self._process_job(job)
            except Exception as exc:  # noqa: BLE001
                job.error = exc
            finally:
                job.event.set()
                self.queue.task_done()

    def _key(self, job: SamPreloadJob) -> Tuple[str, str]:
        return (job.variant, job.slot)

    def _is_superseded(self, job: SamPreloadJob) -> bool:
        with self.lock:
            latest_id = self.latest_request_id.get(self._key(job))
            latest_generation = self.latest_generation.get(self._key(job))
        if latest_id is not None and job.request_id < latest_id:
            return True
        if job.generation is not None and latest_generation is not None and job.generation < latest_generation:
            return True
        return False

    def _superseded_response(self, job: SamPreloadJob) -> SamPreloadResponse:
        width = 0
        height = 0
        if job.image_token:
            cached = _fetch_preloaded_image(job.image_token, job.variant)
            if cached is not None:
                height, width = cached.shape[:2]
        return SamPreloadResponse(status="superseded", width=int(width), height=int(height), token=job.image_token or "")

    def _process_job(self, job: SamPreloadJob) -> SamPreloadResponse:
        variant = job.variant
        slot = job.slot
        image_name = job.image_name

        if job.image_token:
            cached = _fetch_preloaded_image(job.image_token, variant)
            if cached is not None:
                if self._is_superseded(job):
                    height, width = cached.shape[:2]
                    return SamPreloadResponse(
                        status="superseded",
                        width=int(width),
                        height=int(height),
                        token=job.image_token,
                    )
                predictor_manager.set_slot_with_wait(slot, cached, job.image_token, variant, image_name)
                height, width = cached.shape[:2]
                return SamPreloadResponse(status="ready", width=int(width), height=int(height), token=job.image_token)
            if not job.image_base64:
                raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="image_token_not_found")

        if not job.image_base64:
            raise HTTPException(status_code=HTTP_428_PRECONDITION_REQUIRED, detail="image_base64_required")

        np_img = self._decode_base64(job.image_base64)
        token = hashlib.md5(np_img.tobytes()).hexdigest()
        _store_preloaded_image(token, np_img, variant)

        if self._is_superseded(job):
            height, width = np_img.shape[:2]
            return SamPreloadResponse(status="superseded", width=int(width), height=int(height), token=token)

        predictor_manager.set_slot_with_wait(slot, np_img, token, variant, image_name)
        height, width = np_img.shape[:2]
        return SamPreloadResponse(status="ready", width=int(width), height=int(height), token=token)

    @staticmethod
    def _decode_base64(image_base64: str) -> np.ndarray:
        try:
            data = base64.b64decode(image_base64)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"invalid_base64:{exc}") from exc
        try:
            pil_img = Image.open(BytesIO(data)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"invalid_image:{exc}") from exc
        return np.array(pil_img)


sam_preload_manager = SamPreloadManager()


def _predict_with_cache(
    np_img: np.ndarray,
    token: Optional[str],
    variant: str,
    *,
    image_name: Optional[str] = None,
    **predict_kwargs: Any,
):
    return predictor_manager.predict(np_img, token, variant, image_name=image_name, **predict_kwargs)


def _default_variant(value: Optional[str]) -> str:
    return (value or "sam1").lower()


_job_id_counter = itertools.count(1)


@dataclass
class SamPreloadJob:
    request_id: int
    variant: str
    generation: Optional[int]
    image_token: Optional[str]
    image_base64: Optional[str]
    image_name: Optional[str]
    slot: str
    event: threading.Event
    result: Optional['SamPreloadResponse'] = None
    error: Optional[Exception] = None


class SamPreloadManager:
    def __init__(self):
        self.queue: "queue.Queue[SamPreloadJob]" = queue.Queue()
        self.lock = threading.Lock()
        self.latest_request_id: Dict[str, int] = {}
        self.latest_generation: Dict[str, int] = {}
        self.worker = threading.Thread(target=self._worker, name="sam-preload-worker", daemon=True)
        self.worker.start()

    def submit(
        self,
        *,
        variant: str,
        generation: Optional[int],
        image_token: Optional[str],
        image_base64: Optional[str],
        image_name: Optional[str],
        slot: str,
    ) -> 'SamPreloadResponse':
        job = SamPreloadJob(
            request_id=next(_job_id_counter),
            variant=variant,
            generation=generation,
            image_token=image_token,
            image_base64=image_base64,
            image_name=image_name,
            slot=slot,
            event=threading.Event(),
        )
        with self.lock:
            self.latest_request_id[variant] = job.request_id
            if generation is not None:
                prev = self.latest_generation.get(variant)
                if prev is None or generation > prev:
                    self.latest_generation[variant] = generation
        self.queue.put(job)
        job.event.wait()
        if job.error:
            raise job.error
        return job.result  # type: ignore[return-value]

    def _worker(self) -> None:
        while True:
            job = self.queue.get()
            try:
                if self._is_superseded(job):
                    job.result = SamPreloadResponse(status="superseded", width=0, height=0, token=job.image_token or "")
                else:
                    job.result = self._process_job(job)
            except Exception as exc:  # noqa: BLE001 - propagate to caller
                job.error = exc
            finally:
                job.event.set()
                self.queue.task_done()

    def _is_superseded(self, job: SamPreloadJob) -> bool:
        with self.lock:
            latest_id = self.latest_request_id.get(job.variant)
            latest_generation = self.latest_generation.get(job.variant)
        if latest_id is not None and job.request_id < latest_id:
            return True
        if job.generation is not None and latest_generation is not None and job.generation < latest_generation:
            return True
        return False

    def _process_job(self, job: SamPreloadJob) -> 'SamPreloadResponse':
        variant = job.variant
        try:
            slot_name = predictor_manager.resolve_slot(job.slot, allow_disabled_fallback=False)
        except ValueError:
            return SamPreloadResponse(status="slot_disabled", width=0, height=0, token=job.image_token or "")
        image_name = job.image_name

        if job.image_token:
            cached = _fetch_preloaded_image(job.image_token, variant)
            if cached is not None:
                if self._is_superseded(job):
                    return SamPreloadResponse(
                        status="superseded",
                        width=int(cached.shape[1]),
                        height=int(cached.shape[0]),
                        token=job.image_token,
                    )
                predictor_manager.set_slot_with_wait(slot_name, cached, job.image_token, variant, image_name)
                height, width = cached.shape[:2]
                return SamPreloadResponse(status="ready", width=int(width), height=int(height), token=job.image_token)
            if not job.image_base64:
                raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="image_token_not_found")

        if not job.image_base64:
            raise HTTPException(status_code=HTTP_428_PRECONDITION_REQUIRED, detail="image_base64_required")

        np_img = self._decode_base64(job.image_base64)
        token = hashlib.md5(np_img.tobytes()).hexdigest()
        _store_preloaded_image(token, np_img, variant)

        if self._is_superseded(job):
            return SamPreloadResponse(status="superseded", width=int(np_img.shape[1]), height=int(np_img.shape[0]), token=token)

        predictor_manager.set_slot_with_wait(slot_name, np_img, token, variant, image_name)
        height, width = np_img.shape[:2]
        return SamPreloadResponse(status="ready", width=int(width), height=int(height), token=token)

    @staticmethod
    def _decode_base64(image_base64: str) -> np.ndarray:
        try:
            data = base64.b64decode(image_base64)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"invalid_base64:{exc}") from exc
        try:
            pil_img = Image.open(BytesIO(data)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"invalid_image:{exc}") from exc
        return np.array(pil_img)


def _resolve_qwen_device() -> str:
    if QWEN_DEVICE_PREF and QWEN_DEVICE_PREF != "auto":
        if QWEN_DEVICE_PREF.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("cuda_requested_but_unavailable")
        if QWEN_DEVICE_PREF.startswith("mps"):
            mps_backend = getattr(torch.backends, "mps", None)
            if not mps_backend or not mps_backend.is_available():  # type: ignore[attr-defined]
                raise RuntimeError("mps_requested_but_unavailable")
        return QWEN_DEVICE_PREF
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def _get_qwen_prompt_config() -> QwenPromptConfig:
    with qwen_config_lock:
        return qwen_prompt_config.copy(deep=True)


def _set_qwen_prompt_config(config: QwenPromptConfig) -> None:
    global qwen_prompt_config
    with qwen_config_lock:
        qwen_prompt_config = config.copy(deep=True)


def _render_qwen_prompt(
    prompt_type: str,
    *,
    items: Optional[str],
    image_type: Optional[str],
    extra_context: Optional[str],
) -> str:
    if not items:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="qwen_items_required")
    config = _get_qwen_prompt_config()
    section_name = "bbox" if prompt_type in {"bbox", "bbox_sam"} else prompt_type
    section = getattr(config, section_name)
    template = (section.base_prompt or "{items}").strip()
    image_value = (image_type or section.default_image_type or "image").strip() or "image"
    extra_value = extra_context if extra_context is not None and extra_context.strip() else section.default_extra_context
    formatted = template.format(
        image_type=image_value,
        items=items.strip(),
        extra_context=(extra_value or "").strip(),
    )
    return formatted.strip()


def _extract_qwen_json_block(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    candidates = re.findall(r"```json\s*(.*?)```", text, flags=re.DOTALL)
    search_space = candidates or [text]
    for raw in search_space:
        snippet = raw.strip()
        if not snippet:
            continue
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            if "detections" in parsed and isinstance(parsed["detections"], list):
                return snippet, [item for item in parsed["detections"] if isinstance(item, dict)]
            return snippet, [parsed]
        if isinstance(parsed, list):
            return snippet, [item for item in parsed if isinstance(item, dict)]
    raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="qwen_parse_error:no_json_block_found")


def _extract_numeric_sequence(value: Any, *, length: int) -> Optional[List[float]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, (list, tuple)) or len(value) < length:
        return None
    numbers: List[float] = []
    for idx in range(length):
        try:
            numbers.append(float(value[idx]))
        except (TypeError, ValueError):
            return None
    return numbers


def _scale_coord(value: float, src: int, dst: int) -> float:
    if src <= 0:
        return float(value)
    return float(value) * (float(dst) / float(src))


def _scale_bbox_to_image(
    bbox: List[float],
    proc_w: int,
    proc_h: int,
    full_w: int,
    full_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    if len(bbox) < 4:
        return None
    left = _scale_coord(bbox[0], proc_w, full_w)
    top = _scale_coord(bbox[1], proc_h, full_h)
    right = _scale_coord(bbox[2], proc_w, full_w)
    bottom = _scale_coord(bbox[3], proc_h, full_h)
    left_i = max(0, min(full_w, int(round(left))))
    top_i = max(0, min(full_h, int(round(top))))
    right_i = max(0, min(full_w, int(round(right))))
    bottom_i = max(0, min(full_h, int(round(bottom))))
    if right_i <= left_i or bottom_i <= top_i:
        return None
    return left_i, top_i, right_i, bottom_i


def _scale_point_to_image(
    point: List[float],
    proc_w: int,
    proc_h: int,
    full_w: int,
    full_h: int,
) -> Optional[Tuple[float, float]]:
    if len(point) < 2:
        return None
    x = _scale_coord(point[0], proc_w, full_w)
    y = _scale_coord(point[1], proc_h, full_h)
    x = float(min(max(x, 0.0), float(full_w)))
    y = float(min(max(y, 0.0), float(full_h)))
    return x, y


def _qwen_items_from_payload(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [item for item in items if isinstance(item, dict)]


def _qwen_bbox_results(
    items: List[Dict[str, Any]],
    proc_w: int,
    proc_h: int,
    full_w: int,
    full_h: int,
    *,
    limit: int,
) -> List['QwenDetection']:
    results: List[QwenDetection] = []
    for item in items:
        bbox = (
            _extract_numeric_sequence(item.get("bbox_2d"), length=4)
            or _extract_numeric_sequence(item.get("bbox"), length=4)
            or _extract_numeric_sequence(item.get("box"), length=4)
        )
        if not bbox:
            continue
        scaled = _scale_bbox_to_image(bbox, proc_w, proc_h, full_w, full_h)
        if not scaled:
            continue
        left, top, right, bottom = scaled
        yolo_box = to_yolo(full_w, full_h, left, top, right, bottom)
        label = item.get("label") or item.get("class") or item.get("name")
        results.append(QwenDetection(bbox=yolo_box, qwen_label=str(label) if label else None, source="bbox"))
        if len(results) >= limit:
            break
    return results


def _qwen_bbox_sam_results(
    items: List[Dict[str, Any]],
    proc_w: int,
    proc_h: int,
    pil_img: Image.Image,
    np_img: np.ndarray,
    token: Optional[str],
    variant: str,
    *,
    image_name: Optional[str],
    limit: int,
) -> List['QwenDetection']:
    results: List[QwenDetection] = []
    for item in items:
        bbox = (
            _extract_numeric_sequence(item.get("bbox_2d"), length=4)
            or _extract_numeric_sequence(item.get("bbox"), length=4)
            or _extract_numeric_sequence(item.get("box"), length=4)
        )
        if not bbox:
            continue
        scaled = _scale_bbox_to_image(bbox, proc_w, proc_h, pil_img.width, pil_img.height)
        if not scaled:
            continue
        sub_box = np.array(list(scaled), dtype=np.float32)
        masks, _, _ = _predict_with_cache(
            np_img,
            token,
            variant,
            image_name=image_name,
            box=sub_box,
            multimask_output=False,
        )
        mask = masks[0]
        left, top, right, bottom = mask_to_bounding_box(mask)
        if right <= left or bottom <= top:
            continue
        yolo_box = to_yolo(pil_img.width, pil_img.height, left, top, right, bottom)
        label = item.get("label") or item.get("class") or item.get("name")
        results.append(QwenDetection(bbox=yolo_box, qwen_label=str(label) if label else None, source="bbox_sam"))
        if len(results) >= limit:
            break
    return results


def _qwen_point_results(
    items: List[Dict[str, Any]],
    proc_w: int,
    proc_h: int,
    pil_img: Image.Image,
    np_img: np.ndarray,
    token: Optional[str],
    variant: str,
    *,
    image_name: Optional[str],
    limit: int,
) -> List['QwenDetection']:
    results: List[QwenDetection] = []
    for item in items:
        point = _extract_numeric_sequence(item.get("point_2d") or item.get("point"), length=2)
        if not point:
            continue
        scaled_point = _scale_point_to_image(point, proc_w, proc_h, pil_img.width, pil_img.height)
        if not scaled_point:
            continue
        coords = np.array([[scaled_point[0], scaled_point[1]]], dtype=np.float32)
        labels = np.array([1], dtype=np.int64)
        masks, _, _ = _predict_with_cache(
            np_img,
            token,
            variant,
            image_name=image_name,
            point_coords=coords,
            point_labels=labels,
            multimask_output=False,
        )
        mask = masks[0]
        left, top, right, bottom = mask_to_bounding_box(mask)
        if right <= left or bottom <= top:
            continue
        yolo_box = to_yolo(pil_img.width, pil_img.height, left, top, right, bottom)
        label = item.get("label") or item.get("class") or item.get("name")
        results.append(QwenDetection(bbox=yolo_box, qwen_label=str(label) if label else None, source="point"))
        if len(results) >= limit:
            break
    return results


def _sam3_text_detections(
    pil_img: Image.Image,
    payload: Dict[str, Any],
    text_prompt: str,
    limit: Optional[int],
    *,
    min_score: Optional[float] = None,
) -> List[QwenDetection]:
    width, height = pil_img.width, pil_img.height
    boxes_source = payload.get("boxes")
    scores_source = payload.get("scores")
    masks = payload.get("masks")
    if isinstance(boxes_source, torch.Tensor):
        boxes_iter: Sequence[Any] = boxes_source.cpu().numpy()
    elif boxes_source is None:
        boxes_iter = []
    else:
        boxes_iter = boxes_source
    if isinstance(scores_source, torch.Tensor):
        scores_iter: Sequence[Any] = scores_source.cpu().numpy().tolist()
    elif scores_source is None:
        scores_iter = []
    else:
        scores_iter = scores_source
    masks_arr: Optional[np.ndarray] = None
    if masks is not None:
        if isinstance(masks, torch.Tensor):
            masks_arr = masks.cpu().numpy()
        else:
            masks_arr = np.asarray(masks)
    detections: List[QwenDetection] = []
    numeric_limit = limit if limit and limit > 0 else None
    for idx, box in enumerate(boxes_iter):
        coords = np.asarray(box, dtype=np.float32).tolist()
        if len(coords) < 4:
            continue
        x_min, y_min, x_max, y_max = coords[:4]
        if x_max <= x_min or y_max <= y_min:
            continue
        yolo_box = to_yolo(width, height, x_min, y_min, x_max, y_max)
        score_val = None
        if idx < len(scores_iter):
            try:
                score_val = float(scores_iter[idx])
            except (TypeError, ValueError):
                score_val = None
        if min_score is not None and score_val is not None and score_val < min_score:
            continue
        detections.append(QwenDetection(bbox=yolo_box, qwen_label=text_prompt, source="sam3_text", score=score_val))
        if numeric_limit and len(detections) >= numeric_limit:
            break
    if detections or masks_arr is None:
        return detections
    for idx, mask in enumerate(masks_arr):
        x_min, y_min, x_max, y_max = mask_to_bounding_box(mask)
        if x_max <= x_min or y_max <= y_min:
            continue
        yolo_box = to_yolo(width, height, x_min, y_min, x_max, y_max)
        score_val = None
        if idx < len(scores_iter):
            try:
                score_val = float(scores_iter[idx])
            except (TypeError, ValueError):
                score_val = None
        if min_score is not None and score_val is not None and score_val < min_score:
            continue
        detections.append(QwenDetection(bbox=yolo_box, qwen_label=text_prompt, source="sam3_text", score=score_val))
        if numeric_limit and len(detections) >= numeric_limit:
            break
    return detections


def _run_sam3_text_inference(
    pil_img: Image.Image,
    text_prompt: str,
    threshold: float,
    mask_threshold: float,
    limit: Optional[int],
) -> List[QwenDetection]:
    _, processor, _ = _ensure_sam3_text_runtime()
    normalized_limit: Optional[int]
    if limit is None:
        normalized_limit = None
    else:
        try:
            normalized_limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            normalized_limit = None
    state = processor.set_image(pil_img)
    output = processor.set_text_prompt(state=state, prompt=text_prompt)
    return _sam3_text_detections(pil_img, output, text_prompt, normalized_limit, min_score=float(threshold))


def _ensure_qwen_ready():
    global qwen_model, qwen_processor, qwen_device, qwen_last_error, loaded_qwen_model_id
    if QWEN_IMPORT_ERROR is not None or Qwen2_5_VLForConditionalGeneration is None or AutoProcessor is None or process_vision_info is None:
        detail = f"qwen_dependencies_missing:{QWEN_IMPORT_ERROR}"
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
    if (
        qwen_model is not None
        and qwen_processor is not None
        and loaded_qwen_model_id == active_qwen_model_id
    ):
        return qwen_model, qwen_processor
    with qwen_lock:
        if (
            qwen_model is not None
            and qwen_processor is not None
            and loaded_qwen_model_id == active_qwen_model_id
        ):
            return qwen_model, qwen_processor
        try:
            device = _resolve_qwen_device()
        except RuntimeError as exc:  # noqa: BLE001
            qwen_last_error = str(exc)
            raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=f"qwen_device_unavailable:{exc}") from exc
        use_auto_map = QWEN_DEVICE_PREF == "auto" and device.startswith("cuda") and torch.cuda.is_available()
        load_kwargs: Dict[str, Any]
        if use_auto_map:
            load_kwargs = {
                "torch_dtype": "auto",
                "device_map": "auto",
            }
        else:
            dtype = torch.float16 if device.startswith(("cuda", "mps")) else torch.float32
            load_kwargs = {
                "torch_dtype": dtype,
                "low_cpu_mem_usage": True,
            }
        adapter_path = active_qwen_model_path
        metadata = active_qwen_metadata or {}
        base_model_id = metadata.get("model_id") or QWEN_MODEL_NAME
        if adapter_path and PeftModel is None:
            detail = "qwen_peft_missing"
            if PEFT_IMPORT_ERROR is not None:
                detail = f"{detail}:{PEFT_IMPORT_ERROR}"
            raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(base_model_id),
                **load_kwargs,
            )
            if adapter_path:
                model = PeftModel.from_pretrained(model, str(adapter_path))
            if not load_kwargs.get("device_map"):
                model.to(device)
            model.eval()
            processor_source = str(adapter_path) if adapter_path else str(base_model_id)
            if adapter_path and Qwen2_5_VLProcessor is not None:
                processor = Qwen2_5_VLProcessor.from_pretrained(
                    processor_source,
                    min_pixels=QWEN_MIN_PIXELS,
                    max_pixels=QWEN_MAX_PIXELS,
                )
            else:
                processor = AutoProcessor.from_pretrained(
                    processor_source,
                    min_pixels=QWEN_MIN_PIXELS,
                    max_pixels=QWEN_MAX_PIXELS,
                )
        except Exception as exc:  # noqa: BLE001
            qwen_last_error = str(exc)
            raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=f"qwen_load_failed:{exc}") from exc
        qwen_model = model
        qwen_processor = processor
        qwen_device = device
        qwen_last_error = None
        loaded_qwen_model_id = active_qwen_model_id
        return model, processor


def _run_qwen_inference(prompt: str, pil_img: Image.Image) -> Tuple[str, int, int]:
    """Execute a Qwen 2.5 VL inference following the reference blog recipe."""
    model, processor = _ensure_qwen_ready()
    messages: List[Dict[str, Any]] = []
    sys_prompt = (active_qwen_metadata or {}).get("system_prompt")
    if sys_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_prompt}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt},
            ],
        }
    )
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = qwen_device or _resolve_qwen_device()
    inputs = inputs.to(device)
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": QWEN_MAX_NEW_TOKENS,
    }
    if QWEN_DO_SAMPLE:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": QWEN_TEMPERATURE,
                "top_p": QWEN_TOP_P,
            }
        )
    else:
        gen_kwargs["do_sample"] = False
    with torch.inference_mode():
        try:
            generated_ids = model.generate(**inputs, **gen_kwargs)
        except RuntimeError as exc:
            if QWEN_DO_SAMPLE and "probability tensor" in str(exc).lower():
                fallback_kwargs = {**gen_kwargs}
                fallback_kwargs["do_sample"] = False
                fallback_kwargs.pop("temperature", None)
                fallback_kwargs.pop("top_p", None)
                generated_ids = model.generate(**inputs, **fallback_kwargs)
            else:
                raise
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    grid = inputs.get("image_grid_thw")
    if grid is not None:
        grid_values = grid[0]
        input_height = int(grid_values[1].item() * 14)
        input_width = int(grid_values[2].item() * 14)
    else:
        input_height = pil_img.height
        input_width = pil_img.width
    return output_text, input_width, input_height


def resolve_image_payload(
    image_base64: Optional[str],
    image_token: Optional[str],
    sam_variant: Optional[str],
) -> Tuple[Image.Image, np.ndarray, str]:
    variant = _default_variant(sam_variant)
    if image_token:
        cached = _fetch_preloaded_image(image_token, variant)
        if cached is None:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="image_token_not_found")
        pil_img = Image.fromarray(cached)
        return pil_img, cached, image_token
    if not image_base64:
        raise HTTPException(status_code=HTTP_428_PRECONDITION_REQUIRED, detail="image_payload_missing")
    data = base64.b64decode(image_base64)
    pil_img = Image.open(BytesIO(data)).convert("RGB")
    np_img = np.array(pil_img)
    token = hashlib.md5(np_img.tobytes()).hexdigest()
    _store_preloaded_image(token, np_img, variant)
    return pil_img, np_img, token


class Base64Payload(BaseModel):
    image_base64: str
    uuid: Optional[str] = None


class PredictResponse(BaseModel):
    prediction: str
    uuid: Optional[str] = None


class BboxModel(BaseModel):
    className: str
    x: float
    y: float
    width: float
    height: float


class CropImage(BaseModel):
    image_base64: str
    originalName: str
    bboxes: List[BboxModel]


class CropZipRequest(BaseModel):
    images: List[CropImage]


class PointPrompt(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    point_x: float
    point_y: float
    uuid: Optional[str] = None
    sam_variant: Optional[str] = None
    image_name: Optional[str] = None

    @root_validator
    def _ensure_point_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        return values


class BboxPrompt(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    bbox_left: float
    bbox_top: float
    bbox_width: float
    bbox_height: float
    uuid: Optional[str] = None
    sam_variant: Optional[str] = None
    image_name: Optional[str] = None

    @root_validator
    def _ensure_bbox_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        return values


class SamPreloadRequest(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    sam_variant: Optional[str] = None
    preload_generation: Optional[int] = None
    image_name: Optional[str] = None
    slot: Optional[str] = "current"

    @root_validator
    def _ensure_preload_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        if values.get("slot") and values.get("slot") != "current" and not values.get("image_name"):
            raise ValueError("image_name_required_for_slot")
        return values


class SamPreloadResponse(BaseModel):
    status: str = "ready"
    width: int
    height: int
    token: str


sam_preload_manager = SamPreloadManager()


class SamSlotStatus(BaseModel):
    slot: str
    image_name: Optional[str]
    token: Optional[str]
    variant: Optional[str]
    width: Optional[int]
    height: Optional[int]
    busy: bool
    last_loaded: float
    enabled: bool = True
    memory_bytes: Optional[int] = None


class SamActivateRequest(BaseModel):
    image_name: str
    sam_variant: Optional[str] = None


class SamActivateResponse(BaseModel):
    status: str
    slot: Optional[str] = None
    token: Optional[str] = None


class PredictorSettings(BaseModel):
    max_predictors: int
    min_predictors: int
    max_supported_predictors: int
    active_predictors: int
    loaded_predictors: int
    process_ram_mb: float
    total_ram_mb: float
    available_ram_mb: float
    image_ram_mb: float


class PredictorSettingsUpdate(BaseModel):
    max_predictors: int


class MultiPointPrompt(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    positive_points: List[List[float]] = []
    negative_points: List[List[float]] = []
    uuid: Optional[str] = None
    sam_variant: Optional[str] = None
    image_name: Optional[str] = None

    @root_validator
    def _ensure_multi_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        return values


class YoloBboxOutput(BaseModel):
    class_id: str
    bbox: List[float]
    uuid: Optional[str] = None
    image_token: Optional[str] = None


class YoloBboxClassOutput(BaseModel):
    class_id: int
    bbox: List[float]
    uuid: Optional[str] = None
    image_token: Optional[str] = None


class Sam3TextPrompt(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    text_prompt: str
    threshold: float = 0.5
    mask_threshold: float = 0.5
    sam_variant: Optional[str] = None
    image_name: Optional[str] = None
    max_results: Optional[int] = None

    @root_validator
    def _ensure_text_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        if not values.get("text_prompt"):
            raise ValueError("text_prompt_required")
        return values


class SamPointAutoResponse(BaseModel):
    prediction: Optional[str] = None
    proba: Optional[float] = None
    bbox: List[float]
    uuid: Optional[str] = None
    error: Optional[str] = None
    image_token: Optional[str] = None
    score: Optional[float] = None


class QwenDetection(BaseModel):
    bbox: List[float]
    qwen_label: Optional[str] = None
    source: Literal["bbox", "point", "bbox_sam", "sam3_text"]
    score: Optional[float] = None


class QwenInferenceRequest(BaseModel):
    prompt: Optional[str] = None
    item_list: Optional[str] = None
    image_type: Optional[str] = None
    extra_context: Optional[str] = None
    prompt_type: Literal["bbox", "point", "bbox_sam"] = "bbox"
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    sam_variant: Optional[str] = None
    image_name: Optional[str] = None
    max_results: Optional[int] = 8

    @root_validator
    def _validate_qwen_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        prompt = (values.get("prompt") or "").strip()
        items = (values.get("item_list") or "").strip()
        if prompt:
            values["prompt"] = prompt
        elif items:
            values["item_list"] = items
        else:
            raise ValueError("prompt_or_items_required")
        max_results = values.get("max_results")
        if max_results is not None:
            try:
                max_int = int(max_results)
            except (TypeError, ValueError):
                max_int = 8
            values["max_results"] = max(1, min(max_int, 50))
        else:
            values["max_results"] = 8
        return values


class QwenInferenceResponse(BaseModel):
    boxes: List[QwenDetection] = Field(default_factory=list)
    raw_response: str
    prompt: str
    prompt_type: Literal["bbox", "point", "bbox_sam"]
    warnings: List[str] = Field(default_factory=list)
    image_token: Optional[str] = None


class Sam3TextPromptResponse(BaseModel):
    detections: List[QwenDetection] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    image_token: Optional[str] = None


class Sam3TextPromptAutoResponse(BaseModel):
    detections: List[SamPointAutoResponse] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    image_token: Optional[str] = None


class QwenPromptSection(BaseModel):
    base_prompt: str
    default_image_type: str = "image"
    default_extra_context: str = ""

    @root_validator
    def _validate_qwen_section(cls, values):  # noqa: N805
        template = values.get("base_prompt") or ""
        if "{items}" not in template:
            raise ValueError("base_prompt_missing_items_placeholder")
        if "{image_type}" not in template:
            raise ValueError("base_prompt_missing_image_type_placeholder")
        if "{extra_context}" not in template:
            raise ValueError("base_prompt_missing_extra_context_placeholder")
        return values


class QwenPromptConfig(BaseModel):
    bbox: QwenPromptSection
    point: QwenPromptSection


DEFAULT_QWEN_PROMPT_CONFIG = QwenPromptConfig(
    bbox=QwenPromptSection(
        base_prompt=(
            "Output a JSON formatted list of very tight bounding boxes with coordinates in format (x1,y1,x2,y2) "
            "of detections in this {image_type}. Make a single bounding box for each unique instance of the things we want to detect. "
            "The objects we want to detect are: {items}. {extra_context}"
        ),
        default_image_type="image",
        default_extra_context="Return only JSON, no additional text.",
    ),
    point=QwenPromptSection(
        base_prompt=(
            "Output a JSON formatted list of positive click points with coordinates in format (x,y) for detections in this {image_type}. "
            "Each entry must contain \"point_2d\": [x, y] centered on the object so Segment Anything can turn it into a mask/bbox. "
            "Make one point per object. The objects we want to detect are: {items}. {extra_context}"
        ),
        default_image_type="image",
        default_extra_context="Respond with JSON only.",
    ),
)

qwen_prompt_config = DEFAULT_QWEN_PROMPT_CONFIG.copy(deep=True)


class QwenTrainRequest(BaseModel):
    dataset_root: str
    run_name: Optional[str] = None
    model_id: Optional[str] = None
    system_prompt: Optional[str] = None
    system_prompt_noise: Optional[float] = None
    batch_size: Optional[int] = None
    max_epochs: Optional[int] = None
    lr: Optional[float] = None
    accumulate_grad_batches: Optional[int] = None
    check_val_every_n_epoch: Optional[int] = None
    gradient_clip_val: Optional[float] = None
    warmup_steps: Optional[int] = None
    num_workers: Optional[int] = None
    use_qlora: Optional[bool] = None
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    patience: Optional[int] = None
    accelerator: Optional[str] = None
    devices: Optional[List[int]] = None
    device_map: Optional[Any] = None
    seed: Optional[int] = None
    max_detections_per_sample: Optional[int] = None
    max_image_dim: Optional[int] = None


class QwenModelActivateRequest(BaseModel):
    model_id: str

class ActiveModelRequest(BaseModel):
    classifier_path: Optional[str] = None
    labelmap_path: Optional[str] = None
    clip_model: Optional[str] = None


class ActiveModelResponse(BaseModel):
    clip_model: Optional[str]
    classifier_path: Optional[str]
    labelmap_path: Optional[str]
    clip_ready: bool
    labelmap_entries: List[str] = []


@dataclass
class ClipTrainingJob:
    job_id: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    logs: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    temp_dir: Optional[str] = None
    images_dir: Optional[str] = None
    labels_dir: Optional[str] = None
    labelmap_path: Optional[str] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class ClipDatasetUploadJob:
    job_id: str
    root_dir: Path
    images_dir: Path
    labels_dir: Path
    created_at: float = field(default_factory=time.time)
    image_count: int = 0
    label_count: int = 0
    completed: bool = False


@dataclass
class QwenDatasetUploadJob:
    job_id: str
    root_dir: Path
    train_dir: Path
    val_dir: Path
    train_annotations: Path
    val_annotations: Path
    created_at: float = field(default_factory=time.time)
    run_name: Optional[str] = None
    train_count: int = 0
    val_count: int = 0
    completed: bool = False


TRAINING_JOBS: Dict[str, ClipTrainingJob] = {}
TRAINING_JOBS_LOCK = threading.Lock()

QWEN_JOB_ROOT = Path(os.environ.get("QWEN_TRAINING_ROOT", "./uploads/qwen_runs"))
QWEN_JOB_ROOT.mkdir(parents=True, exist_ok=True)
QWEN_DATASET_ROOT = QWEN_JOB_ROOT / "datasets"
QWEN_DATASET_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class QwenTrainingJob:
    job_id: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    config: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_event: threading.Event = field(default_factory=threading.Event)


QWEN_TRAINING_JOBS: Dict[str, QwenTrainingJob] = {}
QWEN_TRAINING_JOBS_LOCK = threading.Lock()
UPLOAD_ROOT = Path("uploads")
UPLOAD_ROOT.mkdir(exist_ok=True)
CLIP_DATASET_UPLOAD_ROOT = UPLOAD_ROOT / "clip_dataset_uploads"
CLIP_DATASET_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
CLIP_DATASET_JOBS: Dict[str, ClipDatasetUploadJob] = {}
CLIP_DATASET_JOBS_LOCK = threading.Lock()
QWEN_DATASET_JOBS: Dict[str, QwenDatasetUploadJob] = {}
QWEN_DATASET_JOBS_LOCK = threading.Lock()

MAX_JOB_LOGS = 250
MAX_QWEN_METRIC_POINTS: Optional[int] = None


def _job_log(job: ClipTrainingJob, message: str) -> None:
    entry = {"timestamp": time.time(), "message": message}
    job.logs.append(entry)
    if len(job.logs) > MAX_JOB_LOGS:
        job.logs[:] = job.logs[-MAX_JOB_LOGS:]
    job.updated_at = time.time()
    try:
        logger.info("[clip-train %s] %s", job.job_id[:8], message)
    except Exception:  # noqa: BLE001 - logging failures should never break workflow
        pass


def _job_update(job: ClipTrainingJob, *, status: Optional[str] = None, message: Optional[str] = None,
                progress: Optional[float] = None, error: Optional[str] = None,
                artifacts: Optional[Dict[str, Any]] = None) -> None:
    if status is not None:
        job.status = status
    if message is not None:
        if message != job.message:
            job.message = message
            _job_log(job, message)
        else:
            job.message = message
    if progress is not None:
        job.progress = max(0.0, min(1.0, progress))
    if error is not None:
        job.error = error
    if artifacts is not None:
        job.artifacts = artifacts


def _qwen_job_log(job: QwenTrainingJob, message: str) -> None:
    entry = {"timestamp": time.time(), "message": message}
    job.logs.append(entry)
    if len(job.logs) > MAX_JOB_LOGS:
        job.logs[:] = job.logs[-MAX_JOB_LOGS:]
    job.updated_at = time.time()
    try:
        logger.info("[qwen-train %s] %s", job.job_id[:8], message)
    except Exception:
        pass


def _qwen_job_update(
    job: QwenTrainingJob,
    *,
    status: Optional[str] = None,
    message: Optional[str] = None,
    progress: Optional[float] = None,
    error: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
    log_message: bool = True,
) -> None:
    if status is not None:
        job.status = status
    if message is not None:
        if message != job.message:
            job.message = message
            if log_message:
                _qwen_job_log(job, message)
        else:
            job.message = message
    if progress is not None:
        job.progress = max(0.0, min(1.0, progress))
    if error is not None:
        job.error = error
    if result is not None:
        job.result = result
    job.updated_at = time.time()


def _serialize_job(job: ClipTrainingJob) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "logs": job.logs,
        "artifacts": job.artifacts,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def _serialize_qwen_job(job: QwenTrainingJob) -> Dict[str, Any]:
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


def _log_qwen_get_request(endpoint: str, jobs: Sequence[QwenTrainingJob]) -> None:
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
    except Exception:  # noqa: BLE001
        logger.exception("Failed to log Qwen GET request for %s", endpoint)


def _coerce_metric_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _coerce_metric_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_metric_value(item) for item in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _qwen_job_append_metric(job: QwenTrainingJob, metric: Dict[str, Any]) -> None:
    if not metric:
        return
    sanitized = {str(key): _coerce_metric_value(val) for key, val in metric.items()}
    job.metrics.append(sanitized)
    limit = MAX_QWEN_METRIC_POINTS
    if isinstance(limit, int) and limit > 0 and len(job.metrics) > limit:
        job.metrics[:] = job.metrics[-limit:]
    job.updated_at = time.time()


def _summarize_qwen_metric(metric: Dict[str, Any]) -> str:
    phase = (metric.get("phase") or "").lower()
    epoch = metric.get("epoch")
    total_epochs = metric.get("total_epochs")
    parts: List[str] = []
    if isinstance(epoch, (int, float)):
        if isinstance(total_epochs, (int, float)) and total_epochs:
            parts.append(f"Epoch {int(epoch)}/{int(total_epochs)}")
        else:
            parts.append(f"Epoch {int(epoch)}")
    if phase == "train":
        batch = metric.get("batch")
        batches_per_epoch = metric.get("batches_per_epoch")
        if isinstance(batch, (int, float)) and isinstance(batches_per_epoch, (int, float)) and batches_per_epoch:
            parts.append(f"Batch {int(batch)}/{int(batches_per_epoch)}")
        train_loss = metric.get("train_loss")
        if isinstance(train_loss, (int, float)):
            parts.append(f"Loss {float(train_loss):.4f}")
    elif phase == "val":
        value = metric.get("value")
        metric_name = metric.get("metric") or "validation"
        if isinstance(value, (int, float)):
            parts.append(f"{metric_name} {float(value):.4f}")
    if not parts:
        return "Training in progress ..."
    return " • ".join(parts)


def _write_qwen_metadata(meta_path: Path, metadata: Dict[str, Any]) -> None:
    try:
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write Qwen metadata for %s: %s", meta_path.parent, exc)


def _persist_qwen_run_metadata(
    result_path: Path,
    config: QwenTrainingConfig,
    training_result: QwenTrainingResult,
) -> Dict[str, Any]:
    dataset_meta = training_result.metadata or {}
    metadata = {
        "id": config.run_name or result_path.name,
        "label": config.run_name or result_path.name,
        "system_prompt": config.system_prompt,
        "system_prompt_noise": config.system_prompt_noise,
        "dataset_context": dataset_meta.get("context", ""),
        "classes": dataset_meta.get("classes", []) or [],
        "model_id": config.model_id,
        "use_qlora": config.use_qlora,
        "max_image_dim": config.max_image_dim,
        "max_detections_per_sample": config.max_detections_per_sample,
        "created_at": time.time(),
        "latest_checkpoint": training_result.latest_checkpoint,
        "source_dataset": config.dataset_root,
    }
    _write_qwen_metadata(result_path / QWEN_METADATA_FILENAME, metadata)
    return metadata


def _persist_qwen_dataset_metadata(dataset_root: Path, metadata: Dict[str, Any]) -> None:
    meta_path = dataset_root / QWEN_METADATA_FILENAME
    try:
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write Qwen dataset metadata for %s: %s", dataset_root, exc)


def _list_qwen_dataset_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not QWEN_DATASET_ROOT.exists():
        return entries
    for path in QWEN_DATASET_ROOT.iterdir():
        if not path.is_dir():
            continue
        metadata = _load_qwen_dataset_metadata(path)
        if not metadata:
            continue
        entry = {
            "id": metadata.get("id") or path.name,
            "label": metadata.get("label") or path.name,
            "dataset_root": str(path),
            "created_at": metadata.get("created_at"),
            "image_count": metadata.get("image_count"),
            "train_count": metadata.get("train_count"),
            "val_count": metadata.get("val_count"),
            "classes": metadata.get("classes", []),
            "context": metadata.get("context", ""),
        }
        entries.append(entry)
    entries.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return entries


def _load_qwen_run_metadata(run_dir: Path) -> Optional[Dict[str, Any]]:
    meta_path = run_dir / QWEN_METADATA_FILENAME
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                data.setdefault("id", run_dir.name)
                return data
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read Qwen metadata from %s: %s", meta_path, exc)
    return None


def _infer_qwen_run_metadata_from_artifacts(run_dir: Path) -> Optional[Dict[str, Any]]:
    latest_dir = run_dir / "latest"
    if not latest_dir.exists():
        return None
    dataset_dir = QWEN_DATASET_ROOT / run_dir.name
    dataset_meta = _load_qwen_dataset_metadata(dataset_dir) or {}
    adapter_config_path = latest_dir / "adapter_config.json"
    adapter_meta: Dict[str, Any] = {}
    if adapter_config_path.exists():
        try:
            with adapter_config_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, dict):
                    adapter_meta = data
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read adapter config for %s: %s", adapter_config_path, exc)
    base_model_id = (
        adapter_meta.get("base_model_name_or_path")
        or dataset_meta.get("model_id")
        or QWEN_MODEL_NAME
    )
    metadata = {
        "id": run_dir.name,
        "label": dataset_meta.get("label") or dataset_meta.get("id") or run_dir.name,
        "system_prompt": dataset_meta.get("system_prompt", ""),
        "system_prompt_noise": dataset_meta.get("system_prompt_noise", 0.05),
        "dataset_context": dataset_meta.get("context", ""),
        "classes": dataset_meta.get("classes", []) or [],
        "model_id": base_model_id,
        "use_qlora": dataset_meta.get("use_qlora"),
        "created_at": dataset_meta.get("created_at") or run_dir.stat().st_mtime,
        "latest_checkpoint": str(latest_dir),
        "source_dataset": str(dataset_dir) if dataset_dir.exists() else None,
    }
    return metadata


def _load_or_repair_qwen_run_metadata(run_dir: Path) -> Optional[Dict[str, Any]]:
    metadata = _load_qwen_run_metadata(run_dir)
    if metadata:
        return metadata
    inferred = _infer_qwen_run_metadata_from_artifacts(run_dir)
    if inferred:
        _write_qwen_metadata(run_dir / QWEN_METADATA_FILENAME, inferred)
    return inferred


def _load_qwen_dataset_metadata(dataset_dir: Path) -> Optional[Dict[str, Any]]:
    meta_path = dataset_dir / QWEN_METADATA_FILENAME
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                data.setdefault("id", dataset_dir.name)
                return data
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read Qwen dataset metadata from %s: %s", meta_path, exc)
    return None


def _list_qwen_model_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for path in QWEN_JOB_ROOT.iterdir():
        if not path.is_dir() or path.name == QWEN_DATASET_ROOT.name:
            continue
        latest = path / "latest"
        if not latest.exists():
            continue
        metadata = _load_or_repair_qwen_run_metadata(path)
        if not metadata:
            continue
        entries.append(
            {
                "id": metadata.get("id") or path.name,
                "label": metadata.get("label") or metadata.get("run_name") or path.name,
                "path": str(latest),
                "created_at": metadata.get("created_at"),
                "metadata": metadata,
                "type": "trained",
            }
        )
    entries.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return entries


def _get_qwen_model_entry(model_id: str) -> Optional[Dict[str, Any]]:
    for entry in _list_qwen_model_entries():
        if entry.get("id") == model_id:
            return entry
    return None


def _build_qwen_config(payload: QwenTrainRequest, job_id: str) -> QwenTrainingConfig:
    if QWEN_TRAINING_IMPORT_ERROR is not None or QwenTrainingConfig is None:
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"qwen_training_unavailable:{QWEN_TRAINING_IMPORT_ERROR}",
        )
    dataset_root = os.path.abspath(payload.dataset_root)
    if not os.path.isdir(dataset_root):
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="dataset_root_not_found")
    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="dataset_missing_train_val")
    run_name = payload.run_name or f"qwen_run_{job_id}"
    result_path = (QWEN_JOB_ROOT / run_name).resolve()
    system_prompt = (payload.system_prompt or DEFAULT_SYSTEM_PROMPT).strip() or DEFAULT_SYSTEM_PROMPT
    cfg_kwargs: Dict[str, Any] = {
        "dataset_root": dataset_root,
        "result_path": str(result_path),
        "model_id": payload.model_id or "Qwen/Qwen2.5-VL-3B-Instruct",
        "run_name": run_name,
        "system_prompt": system_prompt,
    }
    defaults = {
        "batch_size": payload.batch_size,
        "max_epochs": payload.max_epochs,
        "lr": payload.lr,
        "accumulate_grad_batches": payload.accumulate_grad_batches,
        "check_val_every_n_epoch": payload.check_val_every_n_epoch,
        "gradient_clip_val": payload.gradient_clip_val,
        "warmup_steps": payload.warmup_steps,
        "num_workers": payload.num_workers,
        "lora_rank": payload.lora_rank,
        "lora_alpha": payload.lora_alpha,
        "lora_dropout": payload.lora_dropout,
        "patience": payload.patience,
        "accelerator": payload.accelerator,
        "device_map": payload.device_map,
        "seed": payload.seed,
    }
    for key, value in defaults.items():
        if value is not None:
            cfg_kwargs[key] = value
    if payload.devices is not None:
        cfg_kwargs["devices"] = payload.devices
    if payload.use_qlora is not None:
        cfg_kwargs["use_qlora"] = payload.use_qlora
    if payload.system_prompt_noise is not None:
        try:
            noise_val = float(payload.system_prompt_noise)
        except (TypeError, ValueError):
            noise_val = 0.05
        cfg_kwargs["system_prompt_noise"] = max(0.0, min(noise_val, 0.3))
    if payload.max_detections_per_sample is not None:
        try:
            max_dets = int(payload.max_detections_per_sample)
        except (TypeError, ValueError):
            max_dets = 200
        cfg_kwargs["max_detections_per_sample"] = max(1, min(max_dets, 200))
    if payload.max_image_dim is not None:
        try:
            max_dim = int(payload.max_image_dim)
        except (TypeError, ValueError):
            max_dim = 1024
        cfg_kwargs["max_image_dim"] = max(64, min(max_dim, 4096))
    return QwenTrainingConfig(**cfg_kwargs)


def _normalise_relative_path(name: Optional[str]) -> Path:
    candidate = (name or "").replace("\\", "/")
    path = Path(candidate)
    parts = []
    for part in path.parts:
        if part in ("", ".", ".."):
            continue
        if part.endswith(":"):
            continue
        parts.append(part)
    if not parts:
        fallback = Path(candidate).name or f"file_{uuid.uuid4().hex}"
        parts = [fallback]
    return Path(*parts)


def _get_clip_dataset_job(job_id: str) -> ClipDatasetUploadJob:
    with CLIP_DATASET_JOBS_LOCK:
        job = CLIP_DATASET_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="clip_dataset_job_not_found")
    return job


def _pop_clip_dataset_job(job_id: str) -> ClipDatasetUploadJob:
    with CLIP_DATASET_JOBS_LOCK:
        job = CLIP_DATASET_JOBS.pop(job_id, None)
    if not job:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="clip_dataset_job_not_found")
    return job


def _get_qwen_dataset_job(job_id: str) -> QwenDatasetUploadJob:
    with QWEN_DATASET_JOBS_LOCK:
        job = QWEN_DATASET_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="qwen_dataset_job_not_found")
    return job


def _pop_qwen_dataset_job(job_id: str) -> QwenDatasetUploadJob:
    with QWEN_DATASET_JOBS_LOCK:
        job = QWEN_DATASET_JOBS.pop(job_id, None)
    if not job:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="qwen_dataset_job_not_found")
    return job


def _get_qwen_job(job_id: str) -> QwenTrainingJob:
    with QWEN_TRAINING_JOBS_LOCK:
        job = QWEN_TRAINING_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="qwen_job_not_found")
        return job


@app.post("/clip/dataset/init")
def clip_dataset_init():
    job_id = uuid.uuid4().hex
    root = (CLIP_DATASET_UPLOAD_ROOT / job_id).resolve()
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    job = ClipDatasetUploadJob(job_id=job_id, root_dir=root, images_dir=images_dir, labels_dir=labels_dir)
    with CLIP_DATASET_JOBS_LOCK:
        CLIP_DATASET_JOBS[job_id] = job
    return {"job_id": job_id}


@app.post("/clip/dataset/chunk")
async def clip_dataset_chunk(
    job_id: str = Form(...),
    kind: str = Form(...),
    relative_path: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    job = _get_clip_dataset_job(job_id)
    if job.completed:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="clip_dataset_job_finalized")
    kind_lower = kind.strip().lower()
    if kind_lower not in {"image", "label"}:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="invalid_chunk_kind")
    filename = relative_path or file.filename or f"{kind_lower}_{uuid.uuid4().hex}"
    normalised = _normalise_relative_path(filename)
    target_dir = job.images_dir if kind_lower == "image" else job.labels_dir
    dest_path = (target_dir / normalised).resolve()
    if not str(dest_path).startswith(str(job.root_dir)):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="invalid_relative_path")
    await _write_upload_file(file, dest_path)
    with CLIP_DATASET_JOBS_LOCK:
        if kind_lower == "image":
            job.image_count += 1
        else:
            job.label_count += 1
    return {"status": "ok", "images": job.image_count, "labels": job.label_count}


@app.post("/clip/dataset/finalize")
def clip_dataset_finalize(job_id: str = Form(...)):
    job = _pop_clip_dataset_job(job_id)
    job.completed = True
    if job.image_count == 0:
        shutil.rmtree(job.root_dir, ignore_errors=True)
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="dataset_images_missing")
    return {
        "images_path": str(job.images_dir),
        "labels_path": str(job.labels_dir),
        "temp_dir": str(job.root_dir),
        "images": job.image_count,
        "labels": job.label_count,
    }


@app.post("/clip/dataset/cancel")
def clip_dataset_cancel(job_id: str = Form(...)):
    job = None
    with CLIP_DATASET_JOBS_LOCK:
        job = CLIP_DATASET_JOBS.pop(job_id, None)
    if job:
        shutil.rmtree(job.root_dir, ignore_errors=True)
    return {"status": "cancelled"}


@app.post("/qwen/dataset/init")
def qwen_dataset_init(run_name: Optional[str] = Form(None)):
    job_id = uuid.uuid4().hex
    staging_dir = (QWEN_DATASET_ROOT / f"staging_{job_id}").resolve()
    train_dir = staging_dir / "train"
    val_dir = staging_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    train_annotations = train_dir / "annotations.jsonl"
    val_annotations = val_dir / "annotations.jsonl"
    train_annotations.touch()
    val_annotations.touch()
    job = QwenDatasetUploadJob(
        job_id=job_id,
        root_dir=staging_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        train_annotations=train_annotations,
        val_annotations=val_annotations,
        run_name=run_name,
    )
    with QWEN_DATASET_JOBS_LOCK:
        QWEN_DATASET_JOBS[job_id] = job
    logger.info("[qwen-dataset %s] init run_name=%s root=%s", job_id[:8], run_name or "", staging_dir)
    return {"job_id": job_id}


@app.post("/qwen/dataset/chunk")
async def qwen_dataset_chunk(
    job_id: str = Form(...),
    split: str = Form(...),
    image_name: Optional[str] = Form(None),
    annotation_line: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    job = _get_qwen_dataset_job(job_id)
    if job.completed:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="qwen_dataset_job_finalized")
    split_lower = (split or "").strip().lower()
    if split_lower not in {"train", "val"}:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="invalid_split")
    target_dir = job.train_dir if split_lower == "train" else job.val_dir
    target_annotations = job.train_annotations if split_lower == "train" else job.val_annotations
    name = image_name or file.filename or f"{split_lower}_{uuid.uuid4().hex}"
    normalised = _normalise_relative_path(name)
    dest_path = (target_dir / normalised).resolve()
    if not str(dest_path).startswith(str(job.root_dir)):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="invalid_relative_path")
    await _write_upload_file(file, dest_path)
    line = (annotation_line or "").strip()
    if not line:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="annotation_required")
    with target_annotations.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip("\n") + "\n")
    with QWEN_DATASET_JOBS_LOCK:
        if split_lower == "train":
            job.train_count += 1
        else:
            job.val_count += 1
        train_count = job.train_count
        val_count = job.val_count
    size_bytes = None
    try:
        size_bytes = dest_path.stat().st_size
    except OSError:
        size_bytes = None
    logger.info(
        "[qwen-dataset %s] chunk split=%s image=%s size=%sB train=%d val=%d",
        job_id[:8],
        split_lower,
        normalised,
        size_bytes if size_bytes is not None else "unknown",
        train_count,
        val_count,
    )
    return {"status": "ok", "train": train_count, "val": val_count}


@app.post("/qwen/dataset/finalize")
def qwen_dataset_finalize(
    job_id: str = Form(...),
    metadata: str = Form(...),
    run_name: Optional[str] = Form(None),
):
    job = _pop_qwen_dataset_job(job_id)
    try:
        meta_obj = json.loads(metadata)
        if not isinstance(meta_obj, dict):
            raise ValueError("metadata_not_dict")
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(job.root_dir, ignore_errors=True)
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"metadata_invalid:{exc}") from exc
    meta_path = job.root_dir / "dataset_meta.json"
    try:
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta_obj, handle, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(job.root_dir, ignore_errors=True)
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"metadata_write_failed:{exc}") from exc
    desired_name = run_name or job.run_name or f"dataset_{job_id}"
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", desired_name).strip("_") or f"dataset_{job_id}"
    dest_dir = (QWEN_DATASET_ROOT / safe_name).resolve()
    if dest_dir.exists():
        shutil.rmtree(dest_dir, ignore_errors=True)
    shutil.move(str(job.root_dir), str(dest_dir))
    job.completed = True
    dataset_meta = {
        "id": safe_name,
        "label": meta_obj.get("label") or safe_name,
        "classes": meta_obj.get("classes") or [],
        "context": meta_obj.get("context") or "",
        "created_at": time.time(),
        "image_count": job.train_count + job.val_count,
        "train_count": job.train_count,
        "val_count": job.val_count,
    }
    _persist_qwen_dataset_metadata(dest_dir, dataset_meta)
    logger.info(
        "[qwen-dataset %s] finalized dataset=%s train=%d val=%d meta=%s",
        job_id[:8],
        safe_name,
        job.train_count,
        job.val_count,
        json.dumps(meta_obj, ensure_ascii=False),
    )
    return {"dataset_root": str(dest_dir), "run_name": safe_name, "metadata": dataset_meta}


@app.post("/qwen/dataset/cancel")
def qwen_dataset_cancel(job_id: str = Form(...)):
    job = None
    with QWEN_DATASET_JOBS_LOCK:
        job = QWEN_DATASET_JOBS.pop(job_id, None)
    if job:
        shutil.rmtree(job.root_dir, ignore_errors=True)
        logger.info("[qwen-dataset %s] cancelled and cleaned up %s", job_id[:8], job.root_dir)
    return {"status": "cancelled"}


@app.get("/qwen/datasets")
def list_qwen_datasets():
    return _list_qwen_dataset_entries()


@app.delete("/qwen/datasets/{dataset_id}")
def delete_qwen_dataset(dataset_id: str):
    target = (QWEN_DATASET_ROOT / dataset_id).resolve()
    if not str(target).startswith(str(QWEN_DATASET_ROOT.resolve())) or not target.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="qwen_dataset_not_found")
    shutil.rmtree(target, ignore_errors=True)
    return {"status": "deleted"}


@app.post("/qwen/train/dataset/upload")
async def upload_qwen_dataset(file: UploadFile = File(...), run_name: Optional[str] = Form(None)):
    if not file.filename:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="dataset_file_required")
    dataset_token = run_name or f"dataset_{uuid.uuid4().hex}"
    safe_token = re.sub(r"[^A-Za-z0-9._-]", "_", dataset_token)
    dest_dir = QWEN_DATASET_ROOT / safe_token
    if dest_dir.exists():
        shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        contents = await file.read()
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"dataset_read_failed:{exc}") from exc
    try:
        with zipfile.ZipFile(io.BytesIO(contents)) as archive:
            archive.extractall(dest_dir)
    except zipfile.BadZipFile as exc:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"dataset_invalid_zip:{exc}") from exc
    for split in ("train", "val"):
        annotations = dest_dir / split / "annotations.jsonl"
        if not annotations.exists():
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"dataset_missing_annotations:{split}")
    return {"dataset_root": str(dest_dir), "run_name": safe_token}


async def _save_upload_file(upload: UploadFile, root: Path) -> Path:
    rel_path = _normalise_relative_path(upload.filename)
    dest = root / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    await upload.close()
    return dest


async def _save_asset(upload: UploadFile, *, subdir: str) -> str:
    dest_dir = UPLOAD_ROOT / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    rel_name = Path(upload.filename or f"asset_{uuid.uuid4().hex}").name
    dest_path = dest_dir / rel_name
    with dest_path.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    await upload.close()
    return str(dest_path.resolve())


async def _write_upload_file(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    await upload.close()


def _artifacts_to_payload(artifacts: TrainingArtifacts) -> Dict[str, Any]:
    data = asdict(artifacts)
    return data


def _cleanup_job(job: ClipTrainingJob) -> None:
    if job.temp_dir and os.path.isdir(job.temp_dir):
        shutil.rmtree(job.temp_dir, ignore_errors=True)


def _load_labelmap_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    lower = path.lower()
    try:
        if lower.endswith(".pkl"):
            data = joblib.load(path)
            if isinstance(data, list):
                return [str(item) for item in data]
            raise ValueError("labelmap_pickle_invalid")
        entries: List[str] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    entries.append(stripped)
        if not entries:
            raise ValueError("labelmap_empty")
        return entries
    except FileNotFoundError as exc:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="labelmap_not_found") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"labelmap_load_failed:{exc}") from exc


def _current_active_payload() -> Dict[str, Any]:
    return {
        "clip_model": clip_model_name,
        "classifier_path": active_classifier_path,
        "labelmap_path": active_labelmap_path,
        "clip_ready": bool(clip_initialized and clf is not None and clip_model is not None),
        "labelmap_entries": list(active_label_list),
    }


def _select_directory_with_os(initial: Optional[str]) -> Optional[str]:
    if sys.platform == "darwin":
        return _select_directory_macos(initial)
    if sys.platform.startswith("linux"):
        return _select_directory_linux(initial)
    if os.name == "nt":
        return _select_directory_windows(initial)
    return None


def _select_directory_macos(initial: Optional[str]) -> Optional[str]:
    default = initial or os.path.expanduser("~")
    default_path = Path(default)
    if not default_path.exists():
        default_path = Path.home()
    default_str = str(default_path).replace('"', '\\"')
    script = "\n".join([
        f'set startingFolder to POSIX file "{default_str}"',
        'set chosenFolder to choose folder with prompt "Select folder" default location startingFolder',
        'POSIX path of chosenFolder',
    ])
    try:
        result = subprocess.run(["osascript", "-e", script], check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    path = result.stdout.strip()
    return path or None


def _select_directory_linux(initial: Optional[str]) -> Optional[str]:
    default = initial or os.path.expanduser("~")
    candidates = [
        ["zenity", "--file-selection", "--directory", "--title", "Select folder", "--filename", f"{default.rstrip('/')}/"],
        ["kdialog", "--getexistingdirectory", default],
    ]
    for cmd in candidates:
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            path = result.stdout.strip()
            if path:
                return path
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return None


def _select_directory_windows(initial: Optional[str]) -> Optional[str]:
    ps_script = (
        "$initial = '{initial}'.Trim();"
        "if (-not (Test-Path $initial)) { $initial = [Environment]::GetFolderPath('Desktop') }"
        "$app = New-Object -ComObject Shell.Application;"
        "$folder = $app.BrowseForFolder(0, 'Select folder', 0, $initial);"
        "if ($folder) { $folder.Self.Path }"
    ).format(initial=(initial or "").replace("'", "''"))
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    path = result.stdout.strip()
    return path or None

def mask_to_bounding_box(mask: np.ndarray) -> tuple[int,int,int,int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return (0,0,0,0)
    y_min,y_max = np.where(rows)[0][[0,-1]]
    x_min,x_max = np.where(cols)[0][[0,-1]]
    return (int(x_min), int(y_min), int(x_max), int(y_max))

def to_yolo(w: int, h: int, left: int, top: int, right: int, bottom: int) -> List[float]:
    w_abs = float(right - left)
    h_abs = float(bottom - top)
    cx_abs = left + w_abs/2
    cy_abs = top + h_abs/2
    cx = cx_abs / w
    cy = cy_abs / h
    ww = w_abs / w
    hh = h_abs / h
    return [cx, cy, ww, hh]


def yolo_to_corners(box: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    if len(box) < 4:
        return (0, 0, 0, 0)
    cx, cy, ww, hh = box[:4]
    w_abs = max(0.0, float(ww) * w)
    h_abs = max(0.0, float(hh) * h)
    cx_abs = float(cx) * w
    cy_abs = float(cy) * h
    left = int(round(cx_abs - w_abs / 2))
    top = int(round(cy_abs - h_abs / 2))
    right = int(round(cx_abs + w_abs / 2))
    bottom = int(round(cy_abs + h_abs / 2))
    left = max(0, min(w, left))
    top = max(0, min(h, top))
    right = max(left, min(w, right))
    bottom = max(top, min(h, bottom))
    return left, top, right, bottom

@app.post("/predict_base64", response_model=PredictResponse)
def predict_base64(payload: Base64Payload):
    # If CLIP/logreg not loaded, return error message in "prediction"
    if not clip_initialized:
        return PredictResponse(prediction=str(ERROR_MESSAGE), uuid=None) # messy ... returning the error message int as str. Crap logic needs cleanup

    data = base64.b64decode(payload.image_base64)
    pil_img = Image.open(BytesIO(data)).convert("RGB")
    inp = clip_preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = clip_model.encode_image(inp)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    feats_np = feats.squeeze(0).cpu().numpy().reshape(1, -1)
    pred_cls = clf.predict(feats_np)[0]
    return PredictResponse(prediction=pred_cls, uuid=payload.uuid)


@app.get("/clip/backbones")
def list_clip_backbones():
    return {
        "available": SUPPORTED_CLIP_MODELS,
        "active": clip_model_name,
    }


@app.post("/fs/upload_classifier")
async def upload_classifier(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="filename_required")
    saved_path = await _save_asset(file, subdir="classifiers")
    return {"path": saved_path}


@app.post("/fs/upload_labelmap")
async def upload_labelmap(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="filename_required")
    saved_path = await _save_asset(file, subdir="labelmaps")
    return {"path": saved_path}


@app.get("/fs/select_directory")
def select_directory(initial: str = "."):
    selected = _select_directory_with_os(initial or ".")
    return {"path": selected}


def _validate_job_exists(job_id: str) -> ClipTrainingJob:
    with TRAINING_JOBS_LOCK:
        job = TRAINING_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="job_not_found")
        return job


def _start_training_worker(job: ClipTrainingJob, *, images_dir: str, labels_dir: str, labelmap_path: Optional[str],
                           clip_name: str, output_dir: str, model_filename: str, labelmap_filename: str,
                           test_size: float, random_seed: int, batch_size: int, max_iter: int,
                           min_per_class: int, class_weight: str, C: float, device_override: Optional[str],
                           solver: str, reuse_embeddings: bool, hard_example_mining: bool,
                           hard_mining_misclassified_weight: float,
                           hard_mining_low_conf_weight: float,
                           hard_mining_low_conf_threshold: float,
                           hard_mining_margin_threshold: float,
                           convergence_tol: float,
                           cancel_event: threading.Event) -> None:

    def progress_cb(value: float, message: str) -> None:
        with TRAINING_JOBS_LOCK:
            if cancel_event.is_set() and job.status not in {"cancelled", "failed"}:
                _job_update(job, status="cancelling", message="Cancellation requested ...", progress=value)
                return
            _job_update(job, status="running", progress=value, message=message)

    def worker() -> None:
        try:
            with TRAINING_JOBS_LOCK:
                if cancel_event.is_set():
                    _job_update(job, status="cancelled", progress=job.progress, message="Training cancelled before start.")
                    return
                _job_update(job, status="running", progress=0.01, message="Preparing training job ...")
            artifacts = train_clip_from_yolo(
                images_path=images_dir,
                labels_path=labels_dir,
                model_output=os.path.join(output_dir, model_filename),
                labelmap_output=os.path.join(output_dir, labelmap_filename),
                clip_model=clip_name,
                input_labelmap=labelmap_path,
                test_size=test_size,
                random_seed=random_seed,
                batch_size=batch_size,
                max_iter=max_iter,
                min_per_class=min_per_class,
                class_weight=class_weight,
                C=C,
                solver=solver,
                reuse_embeddings=reuse_embeddings,
                hard_example_mining=hard_example_mining,
                hard_mining_misclassified_weight=hard_mining_misclassified_weight,
                hard_mining_low_conf_weight=hard_mining_low_conf_weight,
                hard_mining_low_conf_threshold=hard_mining_low_conf_threshold,
                hard_mining_margin_threshold=hard_mining_margin_threshold,
                convergence_tol=convergence_tol,
                device=device_override,
                progress_cb=progress_cb,
                should_cancel=cancel_event.is_set,
            )
            payload = _artifacts_to_payload(artifacts)
            with TRAINING_JOBS_LOCK:
                _job_update(job, status="succeeded", progress=1.0, message="Training completed.", artifacts=payload)
        except TrainingError as exc:
            with TRAINING_JOBS_LOCK:
                if str(exc) == "cancelled":
                    _job_update(job, status="cancelled", message="Training cancelled by user.")
                    logger.info("[clip-train %s] Training cancelled", job.job_id[:8])
                else:
                    _job_update(job, status="failed", message=str(exc), error=str(exc))
                    logger.warning("[clip-train %s] Training failed: %s", job.job_id[:8], exc)
        except Exception as exc:  # noqa: BLE001
            with TRAINING_JOBS_LOCK:
                _job_update(job, status="failed", message="Training crashed.", error=str(exc))
            logger.exception("[clip-train %s] Training crashed", job.job_id[:8])
        finally:
            _cleanup_job(job)

    threading.Thread(target=worker, name=f"clip-train-{job.job_id[:8]}", daemon=True).start()


def _ensure_directory(path: str) -> str:
    abs_path = os.path.abspath(path or ".")
    if not os.path.isdir(abs_path):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"output_dir_missing:{abs_path}")
    return abs_path


def _coerce_int(value: Any, fallback: int, *, minimum: Optional[int] = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = fallback
    if minimum is not None and result < minimum:
        result = minimum
    return result


def _coerce_float(value: Any, fallback: float, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = fallback
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _normalise_optional_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    return trimmed


def _parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@app.post("/clip/train")
async def start_clip_training(
    images: Optional[List[UploadFile]] = File(None),
    labels: Optional[List[UploadFile]] = File(None),
    labelmap: Optional[UploadFile] = File(None),
    clip_model_name: str = Form(DEFAULT_CLIP_MODEL),
    output_dir: str = Form("."),
    model_filename: str = Form("my_logreg_model.pkl"),
    labelmap_filename: str = Form("my_label_list.pkl"),
    test_size: float = Form(0.2),
    random_seed: int = Form(42),
    batch_size: int = Form(64),
    max_iter: int = Form(1000),
    min_per_class: int = Form(2),
    class_weight: str = Form("none"),
    C: float = Form(1.0),
    device_override: Optional[str] = Form(None),
    images_path_native: Optional[str] = Form(None),
    labels_path_native: Optional[str] = Form(None),
    labelmap_path_native: Optional[str] = Form(None),
    solver: str = Form("saga"),
    reuse_embeddings: Optional[str] = Form(None),
    hard_example_mining: Optional[str] = Form(None),
    hard_mis_weight: float = Form(3.0),
    hard_low_conf_weight: float = Form(2.0),
    hard_low_conf_threshold: float = Form(0.65),
    hard_margin_threshold: float = Form(0.15),
    convergence_tol: float = Form(1e-4),
    staged_temp_dir: Optional[str] = Form(None),
):
    images_path_native = _normalise_optional_path(images_path_native)
    labels_path_native = _normalise_optional_path(labels_path_native)
    labelmap_path_native = _normalise_optional_path(labelmap_path_native)

    solver_name = (solver or "saga").strip().lower()
    if solver_name not in {"saga", "sag", "lbfgs", "liblinear", "newton-cg"}:
        solver_name = "saga"
    reuse_embeddings_flag = _parse_bool(reuse_embeddings)
    hard_example_flag = _parse_bool(hard_example_mining)

    use_native_paths = bool(images_path_native and labels_path_native)
    if use_native_paths and (images or labels):
        logger.info("Ignoring uploaded files; using native dataset paths provided.")
    if reuse_embeddings_flag and not use_native_paths:
        logger.info("Embedding cache reuse requested but dataset is staged upload; disabling reuse for job %s", images_path_native or "<staged>")
        reuse_embeddings_flag = False

    if not use_native_paths:
        if not images:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="images_required")
        if not labels:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="labels_required")

    if clip_model_name not in SUPPORTED_CLIP_MODELS:
        SUPPORTED_CLIP_MODELS.append(clip_model_name)

    output_dir_abs = _ensure_directory(output_dir)

    temp_root: Optional[str] = None
    images_dir: Optional[str] = None
    labels_dir: Optional[str] = None

    if use_native_paths:
        images_dir = _ensure_directory(images_path_native)
        labels_dir = _ensure_directory(labels_path_native)
    else:
        temp_root = tempfile.mkdtemp(prefix="clip_train_")
        images_dir = os.path.join(temp_root, "images")
        labels_dir = os.path.join(temp_root, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for upload in images or []:
            await _save_upload_file(upload, Path(images_dir))

        for upload in labels or []:
            await _save_upload_file(upload, Path(labels_dir))

    labelmap_path = None
    if labelmap_path_native:
        labelmap_path = os.path.abspath(labelmap_path_native)
        if not os.path.isfile(labelmap_path):
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="labelmap_not_found")
    elif labelmap is not None:
        if temp_root is None:
            temp_root = tempfile.mkdtemp(prefix="clip_train_")
        labelmap_path = str(await _save_upload_file(labelmap, Path(temp_root)))

    job_id = uuid.uuid4().hex
    if images_dir is None or labels_dir is None:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="dataset_paths_unresolved")
    logger.info("Starting training job %s (clip=%s, native_paths=%s)", job_id[:8], clip_model_name, use_native_paths)
    if staged_temp_dir:
        temp_root = os.path.abspath(staged_temp_dir)
    job = ClipTrainingJob(job_id=job_id, temp_dir=temp_root, images_dir=images_dir, labels_dir=labels_dir, labelmap_path=labelmap_path)
    job_message = "Job queued (native paths)" if use_native_paths else "Job queued (upload staging)"
    extras = [solver_name]
    if reuse_embeddings_flag:
        extras.append("cache")
    if hard_example_flag:
        extras.append(f"hard({hard_mis_weight_f:.1f}/{hard_low_conf_weight_f:.1f})")
    job_message += f" [{', '.join(extras)}]"
    _job_log(job, job_message)

    test_size_f = _coerce_float(test_size, 0.2, minimum=0.0, maximum=0.9)
    random_seed_i = _coerce_int(random_seed, 42)
    batch_size_i = _coerce_int(batch_size, 64, minimum=1)
    max_iter_i = _coerce_int(max_iter, 1000, minimum=1)
    min_per_class_i = _coerce_int(min_per_class, 2, minimum=1)
    class_weight_norm = (class_weight or "none").lower()
    if class_weight_norm not in {"balanced", "none"}:
        class_weight_norm = "none"
    C_f = _coerce_float(C, 1.0, minimum=0.0001)
    device_override_clean = (device_override or None)
    hard_mis_weight_f = _coerce_float(hard_mis_weight, 3.0, minimum=1.0)
    hard_low_conf_weight_f = _coerce_float(hard_low_conf_weight, 2.0, minimum=1.0)
    hard_low_conf_threshold_f = _coerce_float(hard_low_conf_threshold, 0.65, minimum=0.0, maximum=0.9999)
    hard_margin_threshold_f = _coerce_float(hard_margin_threshold, 0.15, minimum=0.0)
    convergence_tol_f = _coerce_float(convergence_tol, 1e-4, minimum=1e-8)

    with TRAINING_JOBS_LOCK:
        TRAINING_JOBS[job_id] = job

    _start_training_worker(
        job,
        images_dir=images_dir,
        labels_dir=labels_dir,
        labelmap_path=labelmap_path,
        clip_name=clip_model_name,
        output_dir=output_dir_abs,
        model_filename=model_filename,
        labelmap_filename=labelmap_filename,
        test_size=test_size_f,
        random_seed=random_seed_i,
        batch_size=batch_size_i,
        max_iter=max_iter_i,
        min_per_class=min_per_class_i,
        class_weight=class_weight_norm,
        C=C_f,
        device_override=device_override_clean,
        solver=solver_name,
        reuse_embeddings=reuse_embeddings_flag,
        hard_example_mining=hard_example_flag,
        hard_mining_misclassified_weight=hard_mis_weight_f,
        hard_mining_low_conf_weight=hard_low_conf_weight_f,
        hard_mining_low_conf_threshold=hard_low_conf_threshold_f,
        hard_mining_margin_threshold=hard_margin_threshold_f,
        convergence_tol=convergence_tol_f,
        cancel_event=job.cancel_event,
    )

    return {"job_id": job_id}


def _start_qwen_training_worker(job: QwenTrainingJob, config: QwenTrainingConfig) -> None:
    result_path = Path(config.result_path)

    def progress_cb(value: float, message: str) -> None:
        with QWEN_TRAINING_JOBS_LOCK:
            if job.cancel_event.is_set() and job.status not in {"cancelled", "failed"}:
                _qwen_job_update(job, status="cancelling", message="Cancelling ...", progress=value)
                return
            _qwen_job_update(job, status="running", message=message, progress=value)

    def metrics_cb(payload: Dict[str, Any]) -> None:
        if not payload:
            return
        with QWEN_TRAINING_JOBS_LOCK:
            _qwen_job_append_metric(job, payload)
            progress_val = payload.get("progress")
            progress = None
            if isinstance(progress_val, (int, float)):
                progress = max(0.0, min(float(progress_val), 0.999))
            message = _summarize_qwen_metric(payload)
            _qwen_job_update(job, status="running", message=message, progress=progress, log_message=False)

    def cancel_cb() -> bool:
        return job.cancel_event.is_set()

    def worker() -> None:
        try:
            _prepare_for_qwen_training()
            with QWEN_TRAINING_JOBS_LOCK:
                if job.cancel_event.is_set():
                    _qwen_job_update(job, status="cancelled", message="Cancelled before start.")
                    return
                _qwen_job_update(job, status="running", progress=0.01, message="Preparing Qwen training job ...")
            result = train_qwen_model(config, progress_cb=progress_cb, cancel_cb=cancel_cb, metrics_cb=metrics_cb)
            run_metadata = _persist_qwen_run_metadata(result_path, config, result)
            payload = {
                "checkpoints": result.checkpoints,
                "latest": result.latest_checkpoint,
                "epochs_ran": result.epochs_ran,
                "metadata": run_metadata,
            }
            with QWEN_TRAINING_JOBS_LOCK:
                _qwen_job_update(job, status="succeeded", progress=1.0, message="Training complete", result=payload)
        except QwenTrainingError as exc:
            with QWEN_TRAINING_JOBS_LOCK:
                status = "cancelled" if job.cancel_event.is_set() else "failed"
                _qwen_job_update(job, status=status, message=str(exc), error=str(exc))
        except Exception as exc:  # noqa: BLE001
            with QWEN_TRAINING_JOBS_LOCK:
                _qwen_job_update(job, status="failed", message="Unexpected error", error=str(exc))
        finally:
            _finalize_qwen_training_environment()

    thread = threading.Thread(target=worker, name=f"qwen-train-{job.job_id}", daemon=True)
    thread.start()



@app.get("/clip/train")
def list_training_jobs():
    with TRAINING_JOBS_LOCK:
        jobs = sorted(TRAINING_JOBS.values(), key=lambda job: job.created_at, reverse=True)
        return [{"job_id": job.job_id, "status": job.status, "created_at": job.created_at} for job in jobs]


@app.get("/clip/train/{job_id}")
def get_training_job(job_id: str):
    job = _validate_job_exists(job_id)
    return _serialize_job(job)


@app.post("/clip/train/{job_id}/cancel")
def cancel_training_job(job_id: str):
    job = _validate_job_exists(job_id)
    next_status = job.status
    with TRAINING_JOBS_LOCK:
        if job.status in {"succeeded", "failed", "cancelled"}:
            raise HTTPException(status_code=HTTP_428_PRECONDITION_REQUIRED, detail="job_not_cancellable")
        if job.cancel_event.is_set():
            return {"status": job.status}
        job.cancel_event.set()
        next_status = job.status if job.status not in {"running", "queued"} else "cancelling"
        _job_update(job, status=next_status, message="Cancellation requested ...")
    return {"status": next_status}


@app.post("/qwen/train/jobs")
def create_qwen_training_job(payload: QwenTrainRequest):
    if QWEN_TRAINING_IMPORT_ERROR is not None or train_qwen_model is None:
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"qwen_training_unavailable:{QWEN_TRAINING_IMPORT_ERROR}",
        )
    if not payload.dataset_root:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="dataset_root_required")
    job_id = uuid.uuid4().hex
    config = _build_qwen_config(payload, job_id)
    config_dict = asdict(config)
    job = QwenTrainingJob(job_id=job_id, config=config_dict)
    logger.info(
        "[qwen-train %s] create job accelerator=%s devices=%s dataset=%s",
        job_id[:8],
        payload.accelerator or config_dict.get("accelerator"),
        payload.devices or config_dict.get("devices"),
        payload.dataset_root,
    )
    with QWEN_TRAINING_JOBS_LOCK:
        QWEN_TRAINING_JOBS[job_id] = job
        _qwen_job_log(job, "Job queued")
    _start_qwen_training_worker(job, config)
    return {"job_id": job_id}


@app.get("/qwen/train/jobs")
def list_qwen_training_jobs(request: Request):
    with QWEN_TRAINING_JOBS_LOCK:
        jobs = sorted(QWEN_TRAINING_JOBS.values(), key=lambda job: job.created_at, reverse=True)
        _log_qwen_get_request(str(request.url.path), jobs)
        return [_serialize_qwen_job(job) for job in jobs]


@app.get("/qwen/train/jobs/{job_id}")
def get_qwen_training_job(job_id: str, request: Request):
    job = _get_qwen_job(job_id)
    _log_qwen_get_request(str(request.url.path), [job])
    return _serialize_qwen_job(job)


@app.post("/qwen/train/jobs/{job_id}/cancel")
def cancel_qwen_training_job(job_id: str):
    job = _get_qwen_job(job_id)
    with QWEN_TRAINING_JOBS_LOCK:
        if job.status in {"succeeded", "failed", "cancelled"}:
            raise HTTPException(status_code=HTTP_428_PRECONDITION_REQUIRED, detail="job_not_cancellable")
        if job.cancel_event.is_set():
            return {"status": job.status}
        job.cancel_event.set()
        next_status = job.status if job.status not in {"running", "queued"} else "cancelling"
        _qwen_job_update(job, status=next_status, message="Cancellation requested ...")
        return {"status": next_status}


@app.get("/qwen/models")
def list_qwen_models():
    default_entry = {
        "id": "default",
        "label": "Base Qwen 2.5",
        "type": "builtin",
        "metadata": _default_qwen_metadata(),
        "path": None,
        "created_at": None,
        "active": active_qwen_model_id == "default",
    }
    entries = _list_qwen_model_entries()
    data = [default_entry]
    for entry in entries:
        entry["active"] = entry.get("id") == active_qwen_model_id
        data.append(entry)
    return {
        "active": active_qwen_model_id,
        "models": data,
    }


@app.post("/qwen/models/activate")
def activate_qwen_model(payload: QwenModelActivateRequest):
    model_id = (payload.model_id or "").strip()
    if not model_id:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="model_id_required")
    if model_id == "default":
        _set_active_qwen_model_default()
    else:
        entry = _get_qwen_model_entry(model_id)
        if not entry:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="qwen_model_not_found")
        latest = entry.get("path")
        if not latest or not Path(latest).exists():
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="qwen_model_missing_checkpoint")
        _set_active_qwen_model_custom(model_id, Path(latest), entry.get("metadata") or {})
    return {
        "active": active_qwen_model_id,
        "metadata": active_qwen_metadata,
    }


@app.get("/clip/active_model", response_model=ActiveModelResponse)
def get_active_model():
    return _current_active_payload()


@app.post("/clip/active_model", response_model=ActiveModelResponse)
def set_active_model(payload: ActiveModelRequest):
    global clf, clip_model, clip_preprocess, clip_model_name, clip_initialized
    global active_classifier_path, active_labelmap_path, active_label_list

    classifier_path = _normalise_optional_path(payload.classifier_path) or active_classifier_path
    labelmap_path = _normalise_optional_path(payload.labelmap_path)
    labelmap_provided = "labelmap_path" in payload.__fields_set__
    clip_name = _normalise_optional_path(payload.clip_model) or clip_model_name or DEFAULT_CLIP_MODEL

    if not classifier_path:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="classifier_required")
    classifier_path_abs = os.path.abspath(classifier_path)
    if not os.path.isfile(classifier_path_abs):
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="classifier_not_found")

    try:
        new_clf = joblib.load(classifier_path_abs)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"classifier_load_failed:{exc}") from exc

    meta_clip_model = None
    meta_path = os.path.splitext(classifier_path_abs)[0] + ".meta.pkl"
    if os.path.exists(meta_path):
        try:
            meta_obj = joblib.load(meta_path)
            if isinstance(meta_obj, dict):
                meta_clip_model = meta_obj.get("clip_model")
        except Exception:
            meta_clip_model = None
    if meta_clip_model and not payload.clip_model:
        clip_name = str(meta_clip_model)

    if clip_name not in SUPPORTED_CLIP_MODELS:
        SUPPORTED_CLIP_MODELS.append(clip_name)

    need_new_clip = clip_model is None or clip_model_name != clip_name
    if need_new_clip:
        try:
            new_clip_model, new_preprocess = clip.load(clip_name, device=device)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"clip_load_failed:{exc}") from exc
    else:
        new_clip_model = clip_model
        new_preprocess = clip_preprocess

    embed_dim = None
    try:
        coef = getattr(new_clf, "coef_", None)
        if coef is not None:
            embed_dim = coef.shape[1]
    except Exception:
        embed_dim = None

    clip_dim = getattr(getattr(new_clip_model, "visual", None), "output_dim", None)
    if embed_dim is not None and clip_dim is not None and embed_dim != clip_dim:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"dimension_mismatch:{embed_dim}!={clip_dim}")

    labelmap_path_abs = None
    labelmap_entries: List[str] = []
    if labelmap_path is not None:
        labelmap_path_abs = os.path.abspath(labelmap_path)
        labelmap_entries = _load_labelmap_file(labelmap_path_abs)
    elif not labelmap_provided and active_labelmap_path:
        labelmap_path_abs = active_labelmap_path
        labelmap_entries = list(active_label_list)

    with clip_lock:
        clf = new_clf
        clip_model = new_clip_model
        clip_preprocess = new_preprocess
        clip_model_name = clip_name
        clip_initialized = True
        active_classifier_path = classifier_path_abs
        active_labelmap_path = labelmap_path_abs
        active_label_list = labelmap_entries

    return _current_active_payload()


# note this one is actually not used. For a while I thought it would be cool to send a smaller crop to SAM but I'm not sure it makes sense since
# now I'm caching / checking the file that is currently loaded in the predictor and not updating on every call so it's actually waaaay faster and we have the whole image
# ---------------------------------------------------------------------------
# SAM preload endpoint
# ---------------------------------------------------------------------------

@app.post("/sam_preload", response_model=SamPreloadResponse)
def sam_preload(payload: SamPreloadRequest):
    variant = _default_variant(payload.sam_variant)
    try:
        slot_name = predictor_manager.resolve_slot(payload.slot, allow_disabled_fallback=False)
        return sam_preload_manager.submit(
            variant=variant,
            generation=payload.preload_generation,
            image_token=payload.image_token,
            image_base64=payload.image_base64,
            image_name=payload.image_name,
            slot=slot_name,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"sam_preload_failed:{exc}") from exc


@app.get("/sam_slots", response_model=List[SamSlotStatus])
def sam_slots():
    return predictor_manager.status()


@app.post("/sam_activate_slot", response_model=SamActivateResponse)
def sam_activate_slot(payload: SamActivateRequest):
    variant = _default_variant(payload.sam_variant)
    slot = predictor_manager.get_slot_for_image(payload.image_name, variant)
    if slot is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="slot_not_found")
    promoted = predictor_manager.promote_slot(slot.name)
    if not promoted and slot.name != "current":
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="slot_busy")
    return SamActivateResponse(status="promoted", slot="current", token=slot.token)


def _predictor_settings_payload() -> PredictorSettings:
    min_cap, max_cap = predictor_manager.capacity_limits()
    current_cap = predictor_manager.get_capacity()
    active = predictor_manager.active_slot_count()
    loaded = predictor_manager.loaded_slot_count()
    image_memory = predictor_manager.total_image_memory_bytes()
    vm = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    process_mb = _bytes_to_mb(process.memory_info().rss)
    total_mb = _bytes_to_mb(int(vm.total))
    available_mb = _bytes_to_mb(int(vm.available))
    image_mb = _bytes_to_mb(image_memory)
    return PredictorSettings(
        max_predictors=current_cap,
        min_predictors=min_cap,
        max_supported_predictors=max_cap,
        active_predictors=active,
        loaded_predictors=loaded,
        process_ram_mb=process_mb,
        total_ram_mb=total_mb,
        available_ram_mb=available_mb,
        image_ram_mb=image_mb,
    )


@app.get("/predictor_settings", response_model=PredictorSettings)
def get_predictor_settings():
    return _predictor_settings_payload()


@app.post("/predictor_settings", response_model=PredictorSettings)
def update_predictor_settings(payload: PredictorSettingsUpdate):
    predictor_manager.set_capacity(payload.max_predictors)
    return _predictor_settings_payload()


@app.post("/predict_crop", response_model=PredictResponse)
def predict_crop(file: UploadFile = File(...),
                 x: int = Form(...),
                 y: int = Form(...),
                 w: int = Form(...),
                 h: int = Form(...)):

    if not clip_initialized:
        # Return the error message in the "prediction"
        return PredictResponse(prediction=str(ERROR_MESSAGE), uuid=None) # messy ... returning the error message int as str. Crap logic needs cleanup

    image_bytes = file.file.read()
    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    cropped = pil_img.crop((x, y, x+w, y+h))
    inp = clip_preprocess(cropped).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = clip_model.encode_image(inp)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    feats_np = feats.squeeze(0).cpu().numpy().reshape(1, -1)
    pred_cls = clf.predict(feats_np)[0]
    return PredictResponse(prediction=pred_cls, uuid=None)


@app.get("/qwen/status")
def qwen_status():
    dependency_error = str(QWEN_IMPORT_ERROR) if QWEN_IMPORT_ERROR else None
    device_guess = qwen_device
    pending_error = qwen_last_error
    if not device_guess and not dependency_error:
        try:
            device_guess = _resolve_qwen_device()
        except RuntimeError as exc:  # noqa: BLE001
            pending_error = str(exc)
            device_guess = None
    return {
        "available": dependency_error is None,
        "loaded": qwen_model is not None,
        "model_name": QWEN_MODEL_NAME,
        "device": device_guess,
        "max_new_tokens": QWEN_MAX_NEW_TOKENS,
        "min_pixels": QWEN_MIN_PIXELS,
        "max_pixels": QWEN_MAX_PIXELS,
        "last_error": pending_error,
        "dependency_error": dependency_error,
        "active_model": active_qwen_model_id,
        "active_metadata": active_qwen_metadata,
    }


@app.get("/qwen/config", response_model=QwenPromptConfig)
def qwen_get_config():
    return _get_qwen_prompt_config()


@app.post("/qwen/config", response_model=QwenPromptConfig)
def qwen_update_config(payload: QwenPromptConfig):
    _set_qwen_prompt_config(payload)
    return _get_qwen_prompt_config()


@app.post("/qwen/config/reset", response_model=QwenPromptConfig)
def qwen_reset_config():
    _set_qwen_prompt_config(DEFAULT_QWEN_PROMPT_CONFIG.copy(deep=True))
    return _get_qwen_prompt_config()


@app.post("/qwen/infer", response_model=QwenInferenceResponse)
def qwen_infer(payload: QwenInferenceRequest):
    prompt_type = payload.prompt_type.lower()
    if prompt_type not in {"bbox", "point", "bbox_sam"}:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="invalid_prompt_type")
    pil_img, np_img, token = resolve_image_payload(
        payload.image_base64,
        payload.image_token,
        getattr(payload, "sam_variant", None),
    )
    manual_prompt = (payload.prompt or "").strip()
    if manual_prompt:
        final_prompt = manual_prompt
    else:
        item_list = (payload.item_list or "").strip()
        final_prompt = _render_qwen_prompt(
            prompt_type,
            items=item_list,
            image_type=(payload.image_type or "").strip() or None,
            extra_context=(payload.extra_context or "").strip() or None,
        )
    try:
        qwen_text, proc_w, proc_h = _run_qwen_inference(final_prompt, pil_img)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=f"qwen_inference_failed:{exc}") from exc
    print("[Qwen prompt]", final_prompt)
    print("[Qwen raw output]", qwen_text)
    try:
        _, items = _extract_qwen_json_block(qwen_text)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        print(f"[Qwen parse error] {detail}; raw text follows:\n{qwen_text}")
        raise
    normalized_items = _qwen_items_from_payload(items)
    if not normalized_items:
        print("[Qwen parsed but empty list]", qwen_text)
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="qwen_empty_payload")
    variant = _default_variant(getattr(payload, "sam_variant", None))
    limit = payload.max_results or 8
    image_name = getattr(payload, "image_name", None)
    if prompt_type == "bbox":
        boxes = _qwen_bbox_results(normalized_items, proc_w, proc_h, pil_img.width, pil_img.height, limit=limit)
    elif prompt_type == "bbox_sam":
        boxes = _qwen_bbox_sam_results(
            normalized_items,
            proc_w,
            proc_h,
            pil_img,
            np_img,
            token,
            variant,
            image_name=image_name,
            limit=limit,
        )
    else:
        boxes = _qwen_point_results(
            normalized_items,
            proc_w,
            proc_h,
            pil_img,
            np_img,
            token,
            variant,
            image_name=image_name,
            limit=limit,
        )
    warnings: List[str] = []
    if not boxes:
        warnings.append("no_results")
    return QwenInferenceResponse(
        boxes=boxes,
        raw_response=qwen_text,
        prompt=final_prompt,
        prompt_type=prompt_type,  # type: ignore[arg-type]
        warnings=warnings,
        image_token=token,
    )


@app.post("/sam3/text_prompt", response_model=Sam3TextPromptResponse)
def sam3_text_prompt(payload: Sam3TextPrompt):
    variant = _default_variant(payload.sam_variant or "sam3")
    if variant != "sam3":
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_text_requires_sam3")
    pil_img, np_img, token = resolve_image_payload(payload.image_base64, payload.image_token, variant)
    effective_limit = payload.max_results if payload.max_results is not None else 20
    detections = _run_sam3_text_inference(
        pil_img,
        payload.text_prompt,
        payload.threshold,
        payload.mask_threshold,
        effective_limit,
    )
    warnings: List[str] = []
    if not detections:
        warnings.append("no_results")
    return Sam3TextPromptResponse(detections=detections, warnings=warnings, image_token=token)


@app.post("/sam3/text_prompt_auto", response_model=Sam3TextPromptAutoResponse)
def sam3_text_prompt_auto(payload: Sam3TextPrompt):
    variant = _default_variant(payload.sam_variant or "sam3")
    if variant != "sam3":
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_text_requires_sam3")
    if not clip_initialized or clf is None or clip_model is None or clip_preprocess is None:
        return Sam3TextPromptAutoResponse(
            detections=[],
            warnings=["clip_unavailable"],
            image_token=None,
        )
    pil_img, np_img, token = resolve_image_payload(payload.image_base64, payload.image_token, variant)
    effective_limit = payload.max_results if payload.max_results is not None else 20
    detections = _run_sam3_text_inference(
        pil_img,
        payload.text_prompt,
        payload.threshold,
        payload.mask_threshold,
        effective_limit,
    )
    responses: List[SamPointAutoResponse] = []
    warnings: List[str] = []
    if not detections:
        warnings.append("no_results")
    for det in detections:
        x_min, y_min, x_max, y_max = yolo_to_corners(det.bbox, pil_img.width, pil_img.height)
        li = max(0, int(x_min))
        ti = max(0, int(y_min))
        ri = min(pil_img.width, int(x_max))
        bi = min(pil_img.height, int(y_max))
        if ri <= li or bi <= ti:
            responses.append(
                SamPointAutoResponse(
                    prediction="unknown",
                    bbox=det.bbox,
                    uuid=str(uuid.uuid4()),
                    error="empty_mask",
                    image_token=token,
                    score=det.score,
                )
            )
            continue
        subarr = np_img[ti:bi, li:ri, :]
        final_pil = Image.fromarray(subarr)
        inp = clip_preprocess(final_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = clip_model.encode_image(inp)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_np = feats.squeeze(0).cpu().numpy().reshape(1, -1)
        try:
            pred_cls = clf.predict(feats_np)[0]
            prediction = str(pred_cls)
        except Exception as exc:  # noqa: BLE001
            prediction = "unknown"
            warnings.append(f"classifier_error:{exc}")
        responses.append(
            SamPointAutoResponse(
                prediction=prediction,
                bbox=det.bbox,
                uuid=str(uuid.uuid4()),
                image_token=token,
                score=det.score,
            )
        )
    return Sam3TextPromptAutoResponse(detections=responses, warnings=warnings, image_token=token)


@app.post("/sam_point", response_model=YoloBboxOutput)
def sam_point(prompt: PointPrompt):
    pil_img, np_img, token = resolve_image_payload(
        prompt.image_base64,
        getattr(prompt, "image_token", None),
        getattr(prompt, "sam_variant", None),
    )
    coords = np.array([[prompt.point_x, prompt.point_y]])
    labels = np.array([1])
    variant = _default_variant(getattr(prompt, "sam_variant", None))
    masks, _, _ = _predict_with_cache(
        np_img,
        token,
        variant,
        image_name=getattr(prompt, "image_name", None),
        point_coords=coords,
        point_labels=labels,
        multimask_output=False,
    )
    mask = masks[0]
    left, top, right, bottom = mask_to_bounding_box(mask)
    yolo_box = to_yolo(pil_img.width, pil_img.height, left, top, right, bottom)
    return YoloBboxOutput(class_id="0", bbox=yolo_box, uuid=prompt.uuid, image_token=token)


@app.post("/sam_bbox_auto", response_model=SamPointAutoResponse)
def sam_bbox_auto(prompt: BboxPrompt):
    if not clip_initialized:
        return SamPointAutoResponse(prediction=ERROR_MESSAGE, bbox=[], uuid=prompt.uuid)

    pil_img, np_img, token = resolve_image_payload(
        prompt.image_base64,
        getattr(prompt, "image_token", None),
        getattr(prompt, "sam_variant", None),
    )
    full_h, full_w = pil_img.height, pil_img.width
    left = max(0, prompt.bbox_left)
    top = max(0, prompt.bbox_top)
    right = min(full_w, left + prompt.bbox_width)
    bottom = min(full_h, top + prompt.bbox_height)
    if right <= left or bottom <= top:
        return SamPointAutoResponse(
            prediction="unknown",
            bbox=[0, 0, 0, 0],
            uuid=prompt.uuid,
            error="invalid_bbox",
            image_token=token,
        )
    sub_box = np.array([left, top, right, bottom], dtype=np.float32)
    variant = _default_variant(getattr(prompt, "sam_variant", None))
    masks, _, _ = _predict_with_cache(
        np_img,
        token,
        variant,
        image_name=getattr(prompt, "image_name", None),
        box=sub_box,
        multimask_output=False,
    )
    mask = masks[0]
    x_min, y_min, x_max, y_max = mask_to_bounding_box(mask)
    yolo_box = to_yolo(full_w, full_h, x_min, y_min, x_max, y_max)
    gx_min_i = max(0, int(x_min))
    gy_min_i = max(0, int(y_min))
    gx_max_i = min(full_w, int(x_max))
    gy_max_i = min(full_h, int(y_max))
    if gx_max_i <= gx_min_i or gy_max_i <= gy_min_i:
        return SamPointAutoResponse(
            prediction="unknown",
            bbox=yolo_box,
            uuid=prompt.uuid,
            error="empty_mask",
            image_token=token,
        )
    subarr = np_img[gy_min_i:gy_max_i, gx_min_i:gx_max_i, :]
    final_pil = Image.fromarray(subarr)
    inp = clip_preprocess(final_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = clip_model.encode_image(inp)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    feats_np = feats.squeeze(0).cpu().numpy().reshape(1, -1)
    pred_cls = clf.predict(feats_np)[0]
    return SamPointAutoResponse(
        prediction=str(pred_cls),
        bbox=yolo_box,
        uuid=prompt.uuid,
        image_token=token,
    )


@app.post("/sam_point_auto", response_model=SamPointAutoResponse)
def sam_point_auto(prompt: PointPrompt):
    if not clip_initialized:
        return SamPointAutoResponse(prediction=ERROR_MESSAGE, bbox=[], uuid=prompt.uuid)

    pil_img, np_img, token = resolve_image_payload(
        prompt.image_base64,
        getattr(prompt, "image_token", None),
        getattr(prompt, "sam_variant", None),
    )
    coords = np.array([[prompt.point_x, prompt.point_y]])
    labels = np.array([1])
    variant = _default_variant(getattr(prompt, "sam_variant", None))
    masks, _, _ = _predict_with_cache(
        np_img,
        token,
        variant,
        image_name=getattr(prompt, "image_name", None),
        point_coords=coords,
        point_labels=labels,
        multimask_output=False,
    )
    mask = masks[0]
    left, top, right, bottom = mask_to_bounding_box(mask)
    yolo_box = to_yolo(pil_img.width, pil_img.height, left, top, right, bottom)
    li = max(0, int(left))
    ti = max(0, int(top))
    ri = min(pil_img.width, int(right))
    bi = min(pil_img.height, int(bottom))
    if ri <= li or bi <= ti:
        return SamPointAutoResponse(prediction="unknown", bbox=yolo_box, uuid=prompt.uuid, error="empty_mask", image_token=token)
    subarr = np_img[ti:bi, li:ri, :]
    final_pil = Image.fromarray(subarr)
    inp = clip_preprocess(final_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = clip_model.encode_image(inp)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    feats_np = feats.squeeze(0).cpu().numpy().reshape(1, -1)
    pred_cls = clf.predict(feats_np)[0]
    return SamPointAutoResponse(prediction=str(pred_cls), bbox=yolo_box, uuid=prompt.uuid, image_token=token)


@app.post("/sam_point_multi", response_model=YoloBboxOutput)
def sam_point_multi(prompt: MultiPointPrompt):
    positive = prompt.positive_points or []
    negative = prompt.negative_points or []
    if len(positive) == 0:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="positive_points_required")

    pil_img, np_img, token = resolve_image_payload(
        prompt.image_base64,
        getattr(prompt, "image_token", None),
        getattr(prompt, "sam_variant", None),
    )
    coords = np.array(positive + negative, dtype=np.float32)
    labels = np.array([1] * len(positive) + [0] * len(negative), dtype=np.int64)
    variant = _default_variant(getattr(prompt, "sam_variant", None))
    masks, _, _ = _predict_with_cache(
        np_img,
        token,
        variant,
        image_name=getattr(prompt, "image_name", None),
        point_coords=coords,
        point_labels=labels,
        multimask_output=False,
    )
    mask = masks[0]
    left, top, right, bottom = mask_to_bounding_box(mask)
    yolo_box = to_yolo(pil_img.width, pil_img.height, left, top, right, bottom)
    return YoloBboxOutput(class_id="0", bbox=yolo_box, uuid=prompt.uuid, image_token=token)


@app.post("/sam_point_multi_auto", response_model=SamPointAutoResponse)
def sam_point_multi_auto(prompt: MultiPointPrompt):
    if not clip_initialized:
        return SamPointAutoResponse(prediction=ERROR_MESSAGE, bbox=[], uuid=prompt.uuid)

    positive = prompt.positive_points or []
    negative = prompt.negative_points or []
    if len(positive) == 0:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="positive_points_required")

    pil_img, np_img, token = resolve_image_payload(
        prompt.image_base64,
        getattr(prompt, "image_token", None),
        getattr(prompt, "sam_variant", None),
    )
    coords = np.array(positive + negative, dtype=np.float32)
    labels = np.array([1] * len(positive) + [0] * len(negative), dtype=np.int64)
    variant = _default_variant(getattr(prompt, "sam_variant", None))
    masks, _, _ = _predict_with_cache(
        np_img,
        token,
        variant,
        image_name=getattr(prompt, "image_name", None),
        point_coords=coords,
        point_labels=labels,
        multimask_output=False,
    )
    mask = masks[0]
    left, top, right, bottom = mask_to_bounding_box(mask)
    yolo_box = to_yolo(pil_img.width, pil_img.height, left, top, right, bottom)
    li = max(0, int(left))
    ti = max(0, int(top))
    ri = min(pil_img.width, int(right))
    bi = min(pil_img.height, int(bottom))
    if ri <= li or bi <= ti:
        return SamPointAutoResponse(prediction="unknown", bbox=yolo_box, uuid=prompt.uuid, error="empty_mask", image_token=token)
    subarr = np_img[ti:bi, li:ri, :]
    final_pil = Image.fromarray(subarr)
    inp = clip_preprocess(final_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = clip_model.encode_image(inp)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    feats_np = feats.squeeze(0).cpu().numpy().reshape(1, -1)
    pred_cls = clf.predict(feats_np)[0]
    return SamPointAutoResponse(prediction=str(pred_cls), bbox=yolo_box, uuid=prompt.uuid, image_token=token)


@app.post("/sam_bbox", response_model=YoloBboxOutput)
def sam_bbox(prompt: BboxPrompt):
    pil_img, np_img, token = resolve_image_payload(
        prompt.image_base64,
        getattr(prompt, "image_token", None),
        getattr(prompt, "sam_variant", None),
    )
    full_h, full_w = pil_img.height, pil_img.width
    left = max(0, prompt.bbox_left)
    top = max(0, prompt.bbox_top)
    right = min(full_w, left + prompt.bbox_width)
    bottom = min(full_h, top + prompt.bbox_height)
    if right <= left or bottom <= top:
        return YoloBboxOutput(
            class_id="0",
            bbox=[0, 0, 0, 0],
            uuid=prompt.uuid
        )
    sub_box = np.array([left, top, right, bottom], dtype=np.float32)
    variant = _default_variant(getattr(prompt, "sam_variant", None))
    masks, _, _ = _predict_with_cache(
        np_img,
        token,
        variant,
        image_name=getattr(prompt, "image_name", None),
        box=sub_box,
        multimask_output=False,
    )
    mask = masks[0]
    x_min, y_min, x_max, y_max = mask_to_bounding_box(mask)
    yolo_box = to_yolo(full_w, full_h, x_min, y_min, x_max, y_max)
    gx_min_i = max(0, int(x_min))
    gy_min_i = max(0, int(y_min))
    gx_max_i = min(full_w, int(x_max))
    gy_max_i = min(full_h, int(y_max))
    if gx_max_i <= gx_min_i or gy_max_i <= gy_min_i:
        return YoloBboxOutput(
            class_id="0",
            bbox=yolo_box,
            uuid=prompt.uuid,
            image_token=token,
        )
    return YoloBboxOutput(
        class_id="0",
        bbox=yolo_box,
        uuid=prompt.uuid,
        image_token=token,
    )


@app.post("/sam_bbox_auto_class", response_model=YoloBboxClassOutput)
def sam_bbox_auto_class(prompt: BboxPrompt):
    if not clip_initialized:
        return YoloBboxClassOutput(class_id=ERROR_MESSAGE, bbox=[], uuid=None)

    class_map = {"unknown": 0}

    pil_img, np_img, token = resolve_image_payload(
        prompt.image_base64,
        getattr(prompt, "image_token", None),
        getattr(prompt, "sam_variant", None),
    )

    full_h, full_w = pil_img.height, pil_img.width
    left = max(0, prompt.bbox_left)
    top = max(0, prompt.bbox_top)
    right = min(full_w, left + prompt.bbox_width)
    bottom = min(full_h, top + prompt.bbox_height)
    if right <= left or bottom <= top:
        return YoloBboxClassOutput(
            class_id=class_map.get("unknown", 0),
            bbox=[0, 0, 0, 0],
            uuid=prompt.uuid,
            image_token=token,
        )
    sub_box = np.array([left, top, right, bottom], dtype=np.float32)
    variant = _default_variant(getattr(prompt, "sam_variant", None))
    masks, _, _ = _predict_with_cache(
        np_img,
        token,
        variant,
        image_name=getattr(prompt, "image_name", None),
        box=sub_box,
        multimask_output=False,
    )
    mask = masks[0]
    x_min, y_min, x_max, y_max = mask_to_bounding_box(mask)
    yolo_box = to_yolo(full_w, full_h, x_min, y_min, x_max, y_max)
    gx_min_i = max(0, int(x_min))
    gy_min_i = max(0, int(y_min))
    gx_max_i = min(full_w, int(x_max))
    gy_max_i = min(full_h, int(y_max))
    if gx_max_i <= gx_min_i or gy_max_i <= gy_min_i:
        return YoloBboxClassOutput(
            class_id=class_map.get("unknown", 0),
            bbox=yolo_box,
            uuid=prompt.uuid,
            image_token=token,
        )
    subarr = np_img[gy_min_i:gy_max_i, gx_min_i:gx_max_i, :]
    final_pil = Image.fromarray(subarr)
    inp = clip_preprocess(final_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = clip_model.encode_image(inp)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    feats_np = feats.squeeze(0).cpu().numpy().reshape(1, -1)
    pred_label = clf.predict(feats_np)[0]
    class_id = class_map.get(pred_label, 0)
    return YoloBboxClassOutput(
        class_id=class_id,
        bbox=yolo_box,
        uuid=prompt.uuid,
        image_token=token,
    )

@app.post("/crop_zip_init")
def crop_zip_init():
    jobId = str(uuid.uuid4())
    job_store[jobId] = []
    return {"jobId": jobId}

@app.post("/crop_zip_chunk")
def crop_zip_chunk(request: CropZipRequest, jobId: str = Query(...)):
    if jobId not in job_store:
        raise HTTPException(status_code=400, detail="Invalid jobId")
    job_store[jobId].extend(request.images)
    return {"status": "ok", "count": len(request.images)}

@app.get("/crop_zip_finalize")
def crop_zip_finalize(jobId: str):
    if jobId not in job_store:
        raise HTTPException(status_code=400, detail="Invalid jobId")
    all_images = job_store[jobId]
    if len(all_images) == 0:
        empty_buffer = io.BytesIO()
        with zipfile.ZipFile(empty_buffer, mode="w") as zf:
            pass
        empty_buffer.seek(0)
        del job_store[jobId]
        return StreamingResponse(
            empty_buffer,
            media_type="application/x-zip-compressed",
            headers={"Content-Disposition": "attachment; filename=crops.zip"}
        )
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, cropImage in enumerate(all_images):
            img_data = base64.b64decode(cropImage.image_base64)
            pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
            for bindex, bbox in enumerate(cropImage.bboxes):
                left = bbox.x
                top = bbox.y
                right = left + bbox.width
                bottom = top + bbox.height
                left = max(0, min(left, pil_img.width))
                right = max(0, min(right, pil_img.width))
                top = max(0, min(top, pil_img.height))
                bottom = max(0, min(bottom, pil_img.height))
                if right <= left or bottom <= top:
                    continue
                sub_img = pil_img.crop((left, top, right, bottom))
                stem = cropImage.originalName.rsplit(".",1)[0]
                out_name = f"{stem}-{bbox.className}-{bindex}.jpg"
                crop_buffer = io.BytesIO()
                sub_img.save(crop_buffer, format="JPEG")
                crop_buffer.seek(0)
                zf.writestr(out_name, crop_buffer.read())
    zip_buffer.seek(0)
    del job_store[jobId]
    return StreamingResponse(
        zip_buffer,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": "attachment; filename=crops.zip"}
    )
