from __future__ import annotations

import base64, hashlib, io, zipfile, math, uuid, os, tempfile, shutil, time, logging, subprocess, sys, json, re, signal, random, gzip
from copy import deepcopy
from pathlib import Path
import numpy as np
import yaml
from typing import Optional, List, Dict, Tuple, Any, Literal, Sequence, Mapping, Callable
from collections import deque
import torch, clip, joblib, tiktoken
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, root_validator, Field
from omegaconf import OmegaConf
import psutil
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_403_FORBIDDEN,
    HTTP_412_PRECONDITION_FAILED,
    HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    HTTP_404_NOT_FOUND,
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_428_PRECONDITION_REQUIRED,
    HTTP_409_CONFLICT,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)
from collections import OrderedDict
try:
    from scipy.spatial import ConvexHull
except Exception:  # noqa: BLE001
    ConvexHull = None
from segment_anything import sam_model_registry, SamPredictor
import threading
import queue
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict

# Ensure we import the bundled SAM3 package (sam3/sam3) rather than shadowing it
# with the repo root folder name (sam3/). Without this, sam3 becomes a namespace
# that lacks the train.data modules needed for text prompting.
SAM3_SRC_ROOT = (Path(__file__).resolve().parent / "sam3").resolve()
if SAM3_SRC_ROOT.exists():
    sys.path.insert(0, str(SAM3_SRC_ROOT))

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

# Optional text-only LLM for prompt expansion (preferred over Qwen when available).
GPT_OSS_MODEL_ID = os.environ.get("PROMPT_LLM_MODEL_ID", "openai/gpt-oss-20b")
GPT_OSS_PIPELINE = None
GPT_OSS_PIPELINE_ERROR: Optional[Exception] = None
_GPT_OSS_PIPELINE_LOCK = threading.Lock()
_HARMONY_ENCODING = tiktoken.get_encoding("o200k_harmony")
_HARMONY_START = _HARMONY_ENCODING.decode([_HARMONY_ENCODING.encode("<|start|>", allowed_special="all")[0]])
_HARMONY_END = _HARMONY_ENCODING.decode([_HARMONY_ENCODING.encode("<|end|>", allowed_special="all")[0]])
_HARMONY_MESSAGE = _HARMONY_ENCODING.decode([_HARMONY_ENCODING.encode("<|message|>", allowed_special="all")[0]])
_HARMONY_CHANNEL = _HARMONY_ENCODING.decode([_HARMONY_ENCODING.encode("<|channel|>", allowed_special="all")[0]])
_HARMONY_RETURN_ID = _HARMONY_ENCODING.encode("<|return|>", allowed_special="all")[0]
_HARMONY_CALL_ID = _HARMONY_ENCODING.encode("<|call|>", allowed_special="all")[0]
_HARMONY_CONSTRAIN_ID = _HARMONY_ENCODING.encode("<|constrain|>", allowed_special="all")[0]
_HARMONY_STOP_IDS = [_HARMONY_RETURN_ID, _HARMONY_CALL_ID]
_HARMONY_SPECIAL_IDS = {
    _HARMONY_ENCODING.encode("<|start|>", allowed_special="all")[0],
    _HARMONY_ENCODING.encode("<|end|>", allowed_special="all")[0],
    _HARMONY_ENCODING.encode("<|message|>", allowed_special="all")[0],
    _HARMONY_ENCODING.encode("<|channel|>", allowed_special="all")[0],
    _HARMONY_RETURN_ID,
    _HARMONY_CALL_ID,
    _HARMONY_CONSTRAIN_ID,
}

BASE64_IMAGE_MAX_BYTES = int(os.environ.get("IMAGE_MAX_BYTES", str(15 * 1024 * 1024)))
BASE64_IMAGE_MAX_DIM = int(os.environ.get("IMAGE_MAX_DIM", "4096"))

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


MAX_PREDICTOR_SLOTS = 3
DATASET_ZIP_MAX_BYTES = _env_int("DATASET_ZIP_MAX_BYTES", 100 * 1024 * 1024 * 1024)
DATASET_ZIP_ENTRY_MAX_BYTES = _env_int("DATASET_ZIP_ENTRY_MAX_BYTES", 50 * 1024 * 1024 * 1024)
CLIP_DATASET_CHUNK_MAX_BYTES = _env_int("CLIP_DATASET_CHUNK_MAX_BYTES", 10 * 1024 * 1024 * 1024)
CLIP_DATASET_UPLOAD_QUOTA_BYTES = _env_int("CLIP_DATASET_UPLOAD_QUOTA_BYTES", 100 * 1024 * 1024 * 1024)
QWEN_DATASET_CHUNK_MAX_BYTES = _env_int("QWEN_DATASET_CHUNK_MAX_BYTES", 10 * 1024 * 1024 * 1024)
QWEN_DATASET_UPLOAD_QUOTA_BYTES = _env_int("QWEN_DATASET_UPLOAD_QUOTA_BYTES", 100 * 1024 * 1024 * 1024)
QWEN_DATASET_ZIP_MAX_BYTES = _env_int("QWEN_DATASET_ZIP_MAX_BYTES", 100 * 1024 * 1024 * 1024)
ASSET_MAX_BYTES = _env_int("ASSET_MAX_BYTES", 10 * 1024 * 1024 * 1024)
ASSET_UPLOAD_QUOTA_BYTES = _env_int("ASSET_UPLOAD_QUOTA_BYTES", 100 * 1024 * 1024 * 1024)
CLASSIFIER_ALLOWED_EXTS = {".pkl", ".joblib"}
LABELMAP_ALLOWED_EXTS = {".txt", ".pkl"}

SAM_PRELOAD_MAX_BYTES = _env_int("SAM_PRELOAD_MAX_BYTES", 2 * 1024 * 1024 * 1024)
MAX_RESPONSE_DETECTIONS = _env_int("MAX_RESPONSE_DETECTIONS", 5000)
MAX_RESPONSE_MASKS = _env_int("MAX_RESPONSE_MASKS", 2000)
MASK_ENCODE_MAX_BYTES = _env_int("MASK_ENCODE_MAX_BYTES", 64 * 1024 * 1024)
AGENT_MINING_CACHE_MAX_BYTES = _env_int("AGENT_MINING_CACHE_MAX_BYTES", 80 * 1024 * 1024 * 1024)
AGENT_MINING_CACHE_TTL_HOURS = _env_int("AGENT_MINING_CACHE_TTL_HOURS", 0)  # 0 = no TTL purge by default
AGENT_RECIPE_MAX_CROPS = _env_int("AGENT_RECIPE_MAX_CROPS", 1000)
AGENT_RECIPE_MAX_CROP_BYTES = _env_int("AGENT_RECIPE_MAX_CROP_BYTES", 512 * 1024 * 1024)
AGENT_RECIPE_MAX_CLIP_HEAD_BYTES = _env_int("AGENT_RECIPE_MAX_CLIP_HEAD_BYTES", 256 * 1024 * 1024)
AGENT_RECIPE_MAX_JSON_BYTES = _env_int("AGENT_RECIPE_MAX_JSON_BYTES", 10 * 1024 * 1024)
AGENT_RECIPE_MAX_BYTES = _env_int("AGENT_RECIPE_MAX_BYTES", 2 * 1024 * 1024 * 1024)
CLIP_TRAIN_UPLOAD_MAX_BYTES = _env_int("CLIP_TRAIN_UPLOAD_MAX_BYTES", 10 * 1024 * 1024 * 1024)
CLIP_TRAIN_UPLOAD_QUOTA_BYTES = _env_int("CLIP_TRAIN_UPLOAD_QUOTA_BYTES", 100 * 1024 * 1024 * 1024)
FS_DIALOG_ENABLED = _env_bool("FS_DIALOG_ENABLED", True)
FS_DIALOG_ALLOW_REMOTE = _env_bool("FS_DIALOG_ALLOW_REMOTE", False)

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
clip_last_error: Optional[str] = None
# ----------------------------------------------------------------

# 2) Attempt to load the logistic regression model (.pkl)
MODEL_PATH = "./my_logreg_model.pkl"
clf = None
if os.path.exists(MODEL_PATH):
    try:
        print("Loading logistic regression...")
        clf = joblib.load(MODEL_PATH)
        clip_last_error = None
    except Exception as e:
        print(f"Failed to load logistic regression model: {e}")
        clip_initialized = False
        clip_last_error = str(e)
else:
    print(f"File {MODEL_PATH} not found.")
    clip_initialized = False
    clip_last_error = "classifier_not_found"

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

# Keep default CLIP artifacts usable with path allowlists by mirroring them into uploads/.
# This preserves older workflows that write my_logreg_model.pkl/my_label_list.pkl at repo root.
try:
    _uploads_root_early = Path("uploads")
    _uploads_root_early.mkdir(exist_ok=True)
    _classifiers_root_early = (_uploads_root_early / "classifiers").resolve()
    _labelmaps_root_early = (_uploads_root_early / "labelmaps").resolve()
    _classifiers_root_early.mkdir(parents=True, exist_ok=True)
    _labelmaps_root_early.mkdir(parents=True, exist_ok=True)

    if active_classifier_path and os.path.isfile(active_classifier_path):
        src = Path(active_classifier_path).resolve()
        if not str(src).startswith(str(_classifiers_root_early)):
            dst = _classifiers_root_early / src.name
            try:
                if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime or dst.stat().st_size != src.stat().st_size:
                    shutil.copy2(src, dst)
                active_classifier_path = str(dst)
            except Exception:
                pass

    if active_labelmap_path and os.path.isfile(active_labelmap_path):
        src = Path(active_labelmap_path).resolve()
        if not str(src).startswith(str(_labelmaps_root_early)):
            dst = _labelmaps_root_early / src.name
            try:
                if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime or dst.stat().st_size != src.stat().st_size:
                    shutil.copy2(src, dst)
                active_labelmap_path = str(dst)
            except Exception:
                pass
except Exception:
    pass

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


def _ensure_clip_backbone_for_mining() -> Tuple[Optional[Any], Optional[Any]]:
    """Ensure a CLIP backbone is available for exemplar embedding/fp guard (raw CLIP, no classifier required)."""
    global clip_model, clip_preprocess, clip_model_name, clip_initialized
    if clip is None:
        return None, None
    with clip_lock:
        if clip_model is None or clip_preprocess is None:
            clip_name = clip_model_name or DEFAULT_CLIP_MODEL
            try:
                clip_model, clip_preprocess = clip.load(clip_name, device=device)
                clip_model_name = clip_name
                clip_initialized = bool(clip_model is not None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Agent mining could not load CLIP backbone: %s", exc)
                clip_model = None
                clip_preprocess = None
                clip_initialized = False
                return None, None
    return clip_model, clip_preprocess

# 4) Load the SAM model (segment-anything) as normal:
MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "vit_h")
CHECKPOINT_PATH = os.environ.get("SAM_CHECKPOINT_PATH", "./sam_vit_h_4b8939.pth")
SAM3_MODEL_ID = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")
SAM3_PROCESSOR_ID = os.environ.get("SAM3_PROCESSOR_ID", SAM3_MODEL_ID)
SAM3_CHECKPOINT_PATH = os.environ.get("SAM3_CHECKPOINT_PATH")
SAM3_DEVICE_PREF = os.environ.get("SAM3_DEVICE", "auto").strip().lower()
active_sam3_model_id = "default"
active_sam3_checkpoint = SAM3_CHECKPOINT_PATH
active_sam3_enable_segmentation = True
active_sam3_metadata: Dict[str, Any] = {
    "id": "default",
    "label": "Base SAM3",
    "checkpoint": SAM3_CHECKPOINT_PATH,
    "source": "env",
    "enable_segmentation": True,
}


def _resolve_sam3_device() -> torch.device:
    if SAM3_DEVICE_PREF in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(SAM3_DEVICE_PREF)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"invalid_sam3_device:{SAM3_DEVICE_PREF}:{exc}") from exc


def _resolve_sam3_mining_devices() -> List[torch.device]:
    """
    Resolve the list of devices to use for agent mining. If SAM3_DEVICE specifies an explicit device
    (or comma-separated list), honor it; otherwise fan out across all available CUDA devices, falling
    back to CPU when needed.
    """
    devices: List[torch.device] = []
    if SAM3_DEVICE_PREF not in {"", "auto"}:
        for part in SAM3_DEVICE_PREF.split(","):
            name = part.strip()
            if not name:
                continue
            try:
                devices.append(torch.device(name))
            except Exception:
                logger.warning("Invalid SAM3 device in SAM3_DEVICE=%s", name)
    if not devices and torch.cuda.is_available():
        try:
            for idx in range(torch.cuda.device_count()):
                devices.append(torch.device(f"cuda:{idx}"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to enumerate CUDA devices for mining: %s", exc)
            devices = []
    if not devices:
        devices = [torch.device("cpu")]
    return devices


def _reset_sam3_runtime() -> None:
    global sam3_text_model, sam3_text_processor, sam3_text_device
    sam3_text_model = None
    sam3_text_processor = None
    sam3_text_device = None
    try:
        predictor_manager.unload_all()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _resolve_sam1_devices() -> List[torch.device]:
    devices: List[torch.device] = []
    if torch.cuda.is_available():
        try:
            for idx in range(torch.cuda.device_count()):
                devices.append(torch.device(f"cuda:{idx}"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to enumerate CUDA devices for SAM1: %s", exc)
            devices = []
    if not devices:
        devices = [torch.device("cpu")]
    return devices


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
        source = active_sam3_metadata.get("source") if isinstance(active_sam3_metadata, dict) else None
        try:
            model = build_sam3_image_model(
                device=device_str,
                checkpoint_path=active_sam3_checkpoint,
                load_from_HF=active_sam3_checkpoint is None,
                enable_inst_interactivity=True,
                enable_segmentation=active_sam3_enable_segmentation,
                bpe_path=str(SAM3_BPE_PATH),
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
            # Force segmentation head on for text prompting so pred_masks and related keys exist.
            enable_seg = True
            if active_sam3_checkpoint:
                model = build_sam3_image_model(
                    checkpoint_path=active_sam3_checkpoint,
                    device=device_str,
                    load_from_HF=False,
                    enable_segmentation=enable_seg,
                    bpe_path=str(SAM3_BPE_PATH),
                ).to(device)
            else:
                model = build_sam3_image_model(
                    device=device_str,
                    enable_segmentation=enable_seg,
                    bpe_path=str(SAM3_BPE_PATH),
                ).to(device)
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
        _, np_img = _decode_image_base64(base64_data)
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
sam_preload_cache_bytes = 0


def _store_preloaded_image(token: str, np_img: np.ndarray, variant: str) -> None:
    global sam_preload_cache_bytes
    arr = np.ascontiguousarray(np_img)
    arr_bytes = arr.nbytes
    if SAM_PRELOAD_MAX_BYTES > 0 and arr_bytes > SAM_PRELOAD_MAX_BYTES:
        logger.warning("Skipping preload store: image too large (%d bytes > %d)", arr_bytes, SAM_PRELOAD_MAX_BYTES)
        return
    with sam_cache_lock:
        # Remove existing entry bytes
        if token in sam_preload_cache:
            old_arr, _ = sam_preload_cache[token]
            sam_preload_cache_bytes -= getattr(old_arr, "nbytes", 0)
        sam_preload_cache[token] = (arr, variant)
        sam_preload_cache.move_to_end(token)
        sam_preload_cache_bytes += arr_bytes
        while len(sam_preload_cache) > SAM_CACHE_LIMIT or (
            SAM_PRELOAD_MAX_BYTES > 0 and sam_preload_cache_bytes > SAM_PRELOAD_MAX_BYTES
        ):
            _, (evicted_arr, _) = sam_preload_cache.popitem(last=False)
            sam_preload_cache_bytes -= getattr(evicted_arr, "nbytes", 0)


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
        _, np_img = _decode_image_base64(image_base64)
        return np_img


sam_preload_manager = SamPreloadManager()


def _predict_with_cache(
    np_img: np.ndarray,
    token: Optional[str],
    variant: str,
    *,
    image_name: Optional[str] = None,
    **predict_kwargs: Any,
):
    normalized = _default_variant(variant)
    if normalized == "sam3" and not active_sam3_enable_segmentation:
        height, width = np_img.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        box = predict_kwargs.get("box")
        point_coords = predict_kwargs.get("point_coords")
        if box is not None and len(box) >= 4:
            try:
                x1 = int(round(float(box[0])))
                y1 = int(round(float(box[1])))
                x2 = int(round(float(box[2])))
                y2 = int(round(float(box[3])))
            except (TypeError, ValueError):
                x1 = y1 = x2 = y2 = 0
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1
        elif point_coords is not None:
            try:
                px = int(round(float(point_coords[0][0])))
                py = int(round(float(point_coords[0][1])))
            except Exception:
                px = py = 0
            px = max(0, min(px, width - 1))
            py = max(0, min(py, height - 1))
            size = 2
            x1 = max(0, px - size)
            x2 = min(width, px + size)
            y1 = max(0, py - size)
            y2 = min(height, py + size)
            mask[y1:y2, x1:x2] = 1
        masks = np.asarray([mask], dtype=np.uint8)
        return masks, None, None
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
        _, np_img = _decode_image_base64(image_base64)
        return np_img


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
    def _attempt_parse(raw: str) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
        snippet = (raw or "").strip()
        if not snippet:
            return None
        snippet = snippet.strip("`").strip()

        parsed: Any = None
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError:
            parsed = None

        if parsed is None:
            for start_char, end_char in (("{", "}"), ("[", "]")):
                start = snippet.find(start_char)
                end = snippet.rfind(end_char)
                if start < 0 or end < 0 or end <= start:
                    continue
                candidate = snippet[start : end + 1]
                try:
                    parsed = json.loads(candidate)
                    snippet = candidate
                    break
                except json.JSONDecodeError:
                    parsed = None

        if parsed is None:
            return None

        if isinstance(parsed, dict):
            if "detections" in parsed and isinstance(parsed["detections"], list):
                return snippet, [item for item in parsed["detections"] if isinstance(item, dict)]
            return snippet, [parsed]
        if isinstance(parsed, list):
            return snippet, [item for item in parsed if isinstance(item, dict)]
        return None

    fenced = re.findall(r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)```", text, flags=re.DOTALL)
    for raw in [*fenced, text]:
        parsed = _attempt_parse(raw)
        if parsed is not None:
            return parsed

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
    masks_arr: Optional[np.ndarray] = None,
    min_size: Optional[float] = None,
    simplify_epsilon: Optional[float] = None,
    collected_masks: Optional[List[np.ndarray]] = None,
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
    if masks_arr is None and masks is not None:
        if isinstance(masks, torch.Tensor):
            masks_arr = masks.cpu().numpy()
        else:
            masks_arr = np.asarray(masks)
    detections: List[QwenDetection] = []
    if limit is None:
        numeric_limit: Optional[int] = None
    else:
        try:
            numeric_limit = int(limit)
        except (TypeError, ValueError):
            numeric_limit = None
        else:
            if numeric_limit <= 0:
                numeric_limit = None
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
        area = max(0.0, (x_max - x_min) * (y_max - y_min))
        if masks_arr is not None and idx < len(masks_arr):
            try:
                area = float(np.count_nonzero(masks_arr[idx]))
            except Exception:
                area = area
        if min_size is not None:
            try:
                if area < float(min_size):
                    continue
            except Exception:
                pass
        mask_payload = None
        mask_value = None
        if masks_arr is not None and idx < len(masks_arr):
            mask_value = masks_arr[idx]
            mask_payload = encode_binary_mask(mask_value)
        if collected_masks is not None:
            collected_masks.append(mask_value)
        detections.append(
            QwenDetection(
                bbox=yolo_box,
                qwen_label=text_prompt,
                source="sam3_text",
                score=score_val,
                mask=mask_payload,
                simplify_epsilon=simplify_epsilon,
            )
        )
        if numeric_limit is not None and len(detections) >= numeric_limit:
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
        area = max(0.0, (x_max - x_min) * (y_max - y_min))
        try:
            area = float(np.count_nonzero(mask))
        except Exception:
            area = area
        if min_size is not None:
            try:
                if area < float(min_size):
                    continue
            except Exception:
                pass
        detections.append(
            QwenDetection(
                bbox=yolo_box,
                qwen_label=text_prompt,
                source="sam3_text",
                score=score_val,
                mask=encode_binary_mask(mask),
                simplify_epsilon=simplify_epsilon,
            )
        )
        if collected_masks is not None:
            collected_masks.append(mask)
        if numeric_limit is not None and len(detections) >= numeric_limit:
            break
    return detections


def _run_sam3_text_inference(
    pil_img: Image.Image,
    text_prompt: str,
    threshold: float,
    mask_threshold: float,
    limit: Optional[int],
    *,
    return_masks: bool = False,
    min_size: Optional[float] = None,
    simplify_epsilon: Optional[float] = None,
    processor_override: Optional[Any] = None,
    state: Optional[Any] = None,
) -> List[QwenDetection] | Tuple[List[QwenDetection], Optional[List[np.ndarray]]]:
    """
    Run SAM3 text inference. By default returns detections list; callers that need masks should
    inspect the second element when `return_masks=True`.
    """
    if processor_override is not None:
        processor = processor_override
    else:
        _, processor, _ = _ensure_sam3_text_runtime()
    try:
        processor.set_confidence_threshold(float(threshold))
    except Exception:
        # If the processor refuses the threshold, continue with its default.
        pass
    normalized_limit: Optional[int]
    if limit is None:
        normalized_limit = None
    else:
        try:
            normalized_limit = max(1, int(limit))
        except (TypeError, ValueError):
            normalized_limit = None
    img_state = state if state is not None else processor.set_image(pil_img)
    masks_arr: Optional[np.ndarray] = None
    try:
        output = processor.set_text_prompt(state=img_state, prompt=text_prompt)
    except KeyError:
        # Box-only checkpoints (enable_segmentation=False) do not emit pred_masks.
        # Fall back to raw model output and extract boxes/scores manually.
        try:
            raw = processor.model.forward_grounding(
                backbone_out=img_state.get("backbone_out", {}),
                find_input=processor.find_stage,
                find_target=None,
                geometric_prompt=img_state.get("geometric_prompt", processor.model._get_dummy_prompt()),
            )
            boxes_xyxy = raw.get("pred_boxes_xyxy")
            if boxes_xyxy is None:
                boxes_xyxy = raw.get("pred_boxes")
            scores = None
            logits = raw.get("pred_logits")
            if logits is not None:
                try:
                    scores = torch.sigmoid(logits.squeeze(-1))
                except Exception:  # noqa: BLE001
                    try:
                        scores = torch.sigmoid(logits)
                    except Exception:  # noqa: BLE001
                        scores = None
            output = {
                "boxes": boxes_xyxy,
                "scores": scores,
                # no masks for box-only checkpoints
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("SAM3 box-only text prompt fallback failed: %s", exc)
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"sam3_text_grounding_failed:{exc}") from exc
    try:
        if output is not None and hasattr(output, "pred_masks"):
            masks = output.pred_masks
            if masks is not None:
                try:
                    masks_arr = masks.cpu().numpy()
                except Exception:
                    try:
                        masks_arr = np.asarray(masks)
                    except Exception:
                        masks_arr = None
        if masks_arr is not None:
            try:
                masks_arr = (masks_arr >= float(mask_threshold)).astype(np.uint8)
            except Exception:
                # If thresholding fails, keep the raw masks.
                pass
    except Exception:
        masks_arr = None
    collected_masks: Optional[List[np.ndarray]] = [] if return_masks else None
    preds = _sam3_text_detections(
        pil_img,
        output,
        text_prompt,
        normalized_limit,
        min_score=float(threshold),
        masks_arr=masks_arr,
        min_size=min_size,
        simplify_epsilon=simplify_epsilon,
        collected_masks=collected_masks,
    )
    aligned_masks: Optional[List[np.ndarray]]
    if collected_masks is None:
        aligned_masks = None
    else:
        aligned_masks = collected_masks
    return (preds, aligned_masks) if return_masks else preds


def _run_sam3_visual_inference(
    pil_img: Image.Image,
    bbox_xywh: Tuple[float, float, float, float],
    threshold: float,
    mask_threshold: float,
    limit: Optional[int],
    *,
    return_masks: bool = False,
    min_size: Optional[float] = None,
    simplify_epsilon: Optional[float] = None,
    processor_override: Optional[Any] = None,
    state: Optional[Any] = None,
) -> List[QwenDetection] | Tuple[List[QwenDetection], Optional[List[np.ndarray]]]:
    """
    Run SAM3 with a single positive visual (box) prompt. By default returns detections list;
    callers that need masks should inspect the second element when `return_masks=True`.
    """
    if processor_override is not None:
        processor = processor_override
    else:
        _, processor, _ = _ensure_sam3_text_runtime()
    try:
        processor.set_confidence_threshold(float(threshold))
    except Exception:
        pass
    normalized_limit: Optional[int]
    if limit is None:
        normalized_limit = None
    else:
        try:
            normalized_limit = max(1, int(limit))
        except (TypeError, ValueError):
            normalized_limit = None
    img_state = state if state is not None else processor.set_image(pil_img)
    img_w, img_h = float(pil_img.width), float(pil_img.height)
    x, y, w, h = bbox_xywh
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    try:
        output = processor.add_geometric_prompt([cx, cy, w_norm, h_norm], True, state=img_state)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"sam3_visual_prompt_failed:{exc}") from exc
    masks_arr: Optional[np.ndarray] = None
    mask_logits = None
    if isinstance(output, Mapping):
        if "masks_logits" in output and output.get("masks_logits") is not None:
            mask_logits = output.get("masks_logits")
        elif "masks" in output and output.get("masks") is not None:
            mask_logits = output.get("masks")
    if mask_logits is None and isinstance(img_state, Mapping):
        if "masks_logits" in img_state and img_state.get("masks_logits") is not None:
            mask_logits = img_state.get("masks_logits")
        elif "masks" in img_state and img_state.get("masks") is not None:
            mask_logits = img_state.get("masks")
    try:
        threshold_val = float(mask_threshold)
    except Exception:
        threshold_val = 0.5
    threshold_val = max(0.0, min(1.0, threshold_val))
    # Normalize mask logits into a numpy array before thresholding.
    try:
        def _sigmoid_np(arr: np.ndarray) -> np.ndarray:
            try:
                return 1.0 / (1.0 + np.exp(-np.clip(arr, -50, 50)))
            except Exception:
                return 1.0 / (1.0 + np.exp(-arr))

        if isinstance(mask_logits, (list, tuple)):
            if any(isinstance(m, torch.Tensor) for m in mask_logits):
                stacked = [m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else np.asarray(m) for m in mask_logits]
                mask_logits = np.stack(stacked)
            else:
                mask_logits = np.asarray(mask_logits)
        if isinstance(mask_logits, torch.Tensor):
            try:
                probs = mask_logits
                try:
                    min_v = float(probs.min())
                    max_v = float(probs.max())
                    if not (0.0 <= min_v <= 1.0 and 0.0 <= max_v <= 1.0):
                        probs = torch.sigmoid(probs)
                except Exception:
                    probs = torch.sigmoid(probs)
                masks_arr = (probs > threshold_val).cpu().numpy()
            except Exception:
                masks_arr = mask_logits.detach().cpu().numpy()
        elif mask_logits is not None:
            masks_np = np.asarray(mask_logits)
            if masks_np.dtype == bool or (
                np.issubdtype(masks_np.dtype, np.floating)
                and np.nanmin(masks_np) >= 0.0
                and np.nanmax(masks_np) <= 1.0
            ):
                probs_np = masks_np
            else:
                probs_np = _sigmoid_np(masks_np)
            masks_arr = probs_np > threshold_val
        # Normalize mask shape to (N, H, W) where possible
        if masks_arr is not None:
            masks_arr = np.asarray(masks_arr)
            if masks_arr.dtype == object:
                flattened = [np.asarray(m) for m in masks_arr]
                masks_arr = np.stack(flattened)
            if masks_arr.ndim == 2:
                masks_arr = masks_arr[None, ...]
            elif masks_arr.ndim == 4 and masks_arr.shape[1] == 1:
                masks_arr = masks_arr[:, 0, ...]
            elif masks_arr.ndim == 4 and masks_arr.shape[-1] == 1:
                masks_arr = masks_arr[..., 0]
    except Exception:
        masks_arr = None
    def _to_numpy_safe(val: Any) -> Optional[np.ndarray]:
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            try:
                return val.detach().cpu().numpy()
            except Exception:
                return None
        try:
            return np.asarray(val)
        except Exception:
            return None

    payload_for_detection: Dict[str, Any] = {}
    if isinstance(output, Mapping):
        boxes_val = _to_numpy_safe(output.get("boxes"))
        scores_val = _to_numpy_safe(output.get("scores"))
        masks_val = _to_numpy_safe(output.get("masks"))
        if boxes_val is not None:
            payload_for_detection["boxes"] = boxes_val
        if scores_val is not None:
            payload_for_detection["scores"] = scores_val
        if masks_val is not None:
            payload_for_detection["masks"] = masks_val
    collected_masks: Optional[List[np.ndarray]] = [] if return_masks else None
    detections = _sam3_text_detections(
        pil_img,
        payload_for_detection,
        "visual",
        normalized_limit,
        min_score=float(threshold),
        masks_arr=masks_arr,
        min_size=min_size,
        simplify_epsilon=simplify_epsilon,
        collected_masks=collected_masks,
    )
    # Drop the seed box if SAM returns it again (dedupe by IoU against the input box).
    seed_xyxy = (bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3])
    def _iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    aligned_masks: Optional[List[np.ndarray]]
    if collected_masks is None:
        aligned_masks = None
    else:
        aligned_masks = collected_masks
    if detections:
        filtered_dets: List[QwenDetection] = []
        filtered_masks: List[np.ndarray] = []
        for det_idx, det in enumerate(detections):
            bbox = det.bbox or []
            if len(bbox) < 4:
                continue
            det_xyxy = yolo_to_corners(bbox, pil_img.width, pil_img.height)
            if _iou(seed_xyxy, det_xyxy) > 0.9:
                continue
            filtered_dets.append(det)
            if aligned_masks is not None and det_idx < len(aligned_masks):
                filtered_masks.append(aligned_masks[det_idx])
        detections = filtered_dets
        if aligned_masks is not None:
            aligned_masks = filtered_masks
    return (detections, aligned_masks) if return_masks else detections


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


def _unload_qwen_runtime() -> None:
    """Release Qwen model/processor to free device memory."""
    global qwen_model, qwen_processor, qwen_device, loaded_qwen_model_id
    try:
        del qwen_model
    except Exception:
        pass
    try:
        del qwen_processor
    except Exception:
        pass
    qwen_model = None
    qwen_processor = None
    loaded_qwen_model_id = None
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    qwen_device = None


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


def _generate_qwen_text(
    prompt: str,
    *,
    max_new_tokens: int = 128,
    use_system_prompt: bool = True,
) -> str:
    """Text-only generation with Qwen for small helper tasks (no images)."""
    model, processor = _ensure_qwen_ready()
    messages: List[Dict[str, Any]] = []
    sys_prompt = (active_qwen_metadata or {}).get("system_prompt")
    if use_system_prompt and sys_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_prompt}],
            }
        )
    messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )
    device = qwen_device or _resolve_qwen_device()
    inputs = inputs.to(device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0, top_p=1.0)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]
    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return decoded.strip()


def _ensure_prompt_llm():
    """Lazy-load the GPT-OSS text generation pipeline (preferred for prompt expansion)."""
    global GPT_OSS_PIPELINE, GPT_OSS_PIPELINE_ERROR
    if GPT_OSS_PIPELINE is not None:
        return GPT_OSS_PIPELINE
    if GPT_OSS_PIPELINE_ERROR is not None:
        raise GPT_OSS_PIPELINE_ERROR
    with _GPT_OSS_PIPELINE_LOCK:
        if GPT_OSS_PIPELINE is not None:
            return GPT_OSS_PIPELINE
        if GPT_OSS_PIPELINE_ERROR is not None:
            raise GPT_OSS_PIPELINE_ERROR
        try:
            from transformers import pipeline as hf_pipeline

            GPT_OSS_PIPELINE = hf_pipeline(
                "text-generation",
                model=GPT_OSS_MODEL_ID,
                torch_dtype="auto",
                device_map="auto",
            )
            return GPT_OSS_PIPELINE
        except Exception as exc:  # noqa: BLE001
            GPT_OSS_PIPELINE_ERROR = exc
            raise


def _unload_prompt_llm_runtime() -> None:
    """Release the prompt LLM pipeline (GPT-OSS) to free device memory."""
    global GPT_OSS_PIPELINE
    try:
        del GPT_OSS_PIPELINE
    except Exception:
        pass
    GPT_OSS_PIPELINE = None
    try:
        import gc

        gc.collect()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _build_harmony_prompt(system_text: str, developer_text: str, user_text: str) -> str:
    """Render a minimal Harmony prompt (system, developer, user, then assistant final-channel start)."""
    return (
        f"{_HARMONY_START}system{_HARMONY_MESSAGE}{system_text}{_HARMONY_END}"
        f"{_HARMONY_START}developer{_HARMONY_MESSAGE}{developer_text}{_HARMONY_END}"
        f"{_HARMONY_START}user{_HARMONY_MESSAGE}{user_text}{_HARMONY_END}"
        f"{_HARMONY_START}assistant{_HARMONY_CHANNEL}final{_HARMONY_MESSAGE}"
    )


def _extract_harmony_final(text: str) -> Tuple[str, bool]:
    """
    Extract the final-channel assistant content from a Harmony-formatted completion.
    Returns (content, valid) where valid indicates we found a properly marked final channel.
    """
    if not text:
        return "", False
    pattern = re.compile(
        r"<\|start\|>assistant(?:<\|channel\|>(\w+))?<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|<\|call\|>|$)",
        re.DOTALL,
    )
    matches = pattern.findall(text)
    if not matches:
        return text.strip(), False
    # Prefer the last final-channel message.
    finals = [m for m in matches if m[0] == "final"]
    chosen = finals[-1] if finals else matches[-1]
    content = chosen[1].strip()
    valid = bool(finals)
    return content, valid


def _parse_prompt_candidates(raw: str, seen: set[str], limit: int) -> List[str]:
    """Parse and validate a comma/list output into cleaned candidates; returns [] if invalid."""
    if not raw:
        return []
    parts = re.split(r"[,;\n]+", raw)
    parsed: List[str] = []
    for part in parts:
        cand = part.strip().strip('"').strip("'")
        cand = re.sub(r"(?i)^assistant\s+final[:\s]+", "", cand)
        if not cand:
            continue
        if cand.upper() == "STOP":
            break
        # Must be letters/spaces/hyphens only.
        if re.search(r"[^A-Za-z\s\-]", cand):
            continue
        words = cand.split()
        if not (1 <= len(words) <= 4):
            continue
        if any(len(w) < 2 for w in words):
            continue
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        parsed.append(cand)
        if limit and len(parsed) >= limit:
            break
    return parsed


def _generate_prompt_text(
    prompt: str,
    *,
    max_new_tokens: int = 128,
    reasoning: Literal["none", "low", "medium", "high"] = "high",
) -> str:
    """
    Text-only helper for prompt brainstorming/critique.
    Uses the GPT-OSS pipeline; returns empty string on failure.
    """
    system_msg = (
        "You are ChatGPT, a large language model trained by OpenAI.\n"
        "Knowledge cutoff: 2024-06\n"
        f"Current date: {time.strftime('%Y-%m-%d')}\n\n"
        f"Reasoning: {reasoning}\n\n"
        "# Valid channels: analysis, commentary, final. Channel must be included for every message."
    )
    developer_msg = (
        "# Instructions\n"
        "You generate short noun-phrase candidates for open-vocabulary detection. "
        "Respond once, ONLY on the final channel, with a comma-separated list (no prose). "
        "Do NOT emit analysis/commentary messages. "
        "Each candidate: 1-3 words, letters/spaces/hyphens only, no numbers, no punctuation beyond commas, no quotes, no numbering, no JSON. "
        "If no valid candidates, return an empty list."
    )
    user_msg = prompt
    harmony_prompt = _build_harmony_prompt(system_msg, developer_msg, user_msg)
    try:
        pipe = _ensure_prompt_llm()
        tokenizer = getattr(pipe, "tokenizer", None)
        stop_ids = None
        pad_id = None
        if tokenizer is not None:
            stop_ids = list({tok for tok in _HARMONY_STOP_IDS if tok is not None})
            try:
                end_id = _HARMONY_ENCODING.encode("<|end|>", allowed_special="all")[0]
                stop_ids.append(end_id)
            except Exception:
                pass
            try:
                if tokenizer.eos_token_id is not None:
                    stop_ids.append(tokenizer.eos_token_id)
            except Exception:
                pass
            if tokenizer.pad_token_id is not None:
                pad_id = tokenizer.pad_token_id
            elif tokenizer.eos_token_id is not None:
                pad_id = tokenizer.eos_token_id
        outputs = pipe(
            harmony_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=stop_ids,
            pad_token_id=pad_id,
            return_full_text=True,
        )
        if outputs:
            gen = outputs[0].get("generated_text") or outputs[0].get("text") or ""
            text = ""
            if isinstance(gen, str):
                text = gen
            elif isinstance(gen, list):
                # Some pipelines return list of dicts; fallback to stringify.
                text = str(gen)
            if text:
                # If reasoning is enabled, require a well-formed final channel.
                final, final_ok = _extract_harmony_final(text)
                if reasoning != "none":
                    if not final_ok or not final:
                        return ""
                    return final.strip()
                # Reasoning disabled: be lenient. If final channel not found, try to strip any headers
                # and grab the last assistant chunk.
                candidate = final if final else text
                # Strip any harmony markers or role prefixes.
                candidate = re.sub(r"<\\|[^>]+?\\|>", " ", candidate)
                candidate = re.sub(r"(?i)^(system|developer|user|assistant)\\s+final\\s+", " ", candidate)
                candidate = re.sub(r"\\s+", " ", candidate).strip()
                if candidate:
                    return candidate
    except Exception:
        pass
    return ""


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
    pil_img, np_img = _decode_image_base64(image_base64)
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

    @root_validator(skip_on_failure=True)
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

    @root_validator(skip_on_failure=True)
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

    @root_validator(skip_on_failure=True)
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
    gpu_total_mb: Optional[float] = None
    gpu_free_mb: Optional[float] = None


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

    @root_validator(skip_on_failure=True)
    def _ensure_multi_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        return values


class YoloBboxOutput(BaseModel):
    class_id: str
    bbox: List[float]
    uuid: Optional[str] = None
    image_token: Optional[str] = None
    mask: Optional[Dict[str, Any]] = None
    simplify_epsilon: Optional[float] = None


class YoloBboxClassOutput(BaseModel):
    class_id: int
    bbox: List[float]
    uuid: Optional[str] = None
    image_token: Optional[str] = None
    mask: Optional[Dict[str, Any]] = None
    simplify_epsilon: Optional[float] = None


class Sam3TextPrompt(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    text_prompt: str
    threshold: float = 0.5
    mask_threshold: float = 0.5
    simplify_epsilon: Optional[float] = None
    sam_variant: Optional[str] = None
    image_name: Optional[str] = None
    max_results: Optional[int] = None
    min_size: Optional[int] = None

    @root_validator(skip_on_failure=True)
    def _ensure_text_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        if not values.get("text_prompt"):
            raise ValueError("text_prompt_required")
        min_size = values.get("min_size")
        if min_size is not None:
            try:
                min_size_int = max(0, int(min_size))
            except (TypeError, ValueError):
                min_size_int = 0
            values["min_size"] = min_size_int
        eps = values.get("simplify_epsilon")
        if eps is not None:
            try:
                eps_val = float(eps)
            except (TypeError, ValueError):
                eps_val = None
            values["simplify_epsilon"] = eps_val if eps_val is None or eps_val >= 0 else 0.0
        return values


class Sam3VisualPrompt(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    bbox_left: float
    bbox_top: float
    bbox_width: float
    bbox_height: float
    threshold: float = 0.5
    mask_threshold: float = 0.5
    simplify_epsilon: Optional[float] = None
    sam_variant: Optional[str] = None
    image_name: Optional[str] = None
    max_results: Optional[int] = None
    min_size: Optional[int] = None

    @root_validator(skip_on_failure=True)
    def _validate_visual_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        for key in ("bbox_left", "bbox_top", "bbox_width", "bbox_height"):
            raw = values.get(key)
            try:
                values[key] = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"invalid_{key}")
        if values["bbox_width"] <= 0 or values["bbox_height"] <= 0:
            raise ValueError("invalid_bbox_dims")
        min_size = values.get("min_size")
        if min_size is not None:
            try:
                values["min_size"] = max(0, int(min_size))
            except (TypeError, ValueError):
                values["min_size"] = 0
        eps = values.get("simplify_epsilon")
        if eps is not None:
            try:
                eps_val = float(eps)
            except (TypeError, ValueError):
                eps_val = None
            values["simplify_epsilon"] = eps_val if eps_val is None or eps_val >= 0 else 0.0
        return values


class SamPointAutoResponse(BaseModel):
    prediction: Optional[str] = None
    proba: Optional[float] = None
    bbox: List[float]
    uuid: Optional[str] = None
    error: Optional[str] = None
    image_token: Optional[str] = None
    score: Optional[float] = None
    mask: Optional[Dict[str, Any]] = None
    simplify_epsilon: Optional[float] = None


class QwenDetection(BaseModel):
    bbox: List[float]
    qwen_label: Optional[str] = None
    source: Literal["bbox", "point", "bbox_sam", "sam3_text"]
    score: Optional[float] = None
    mask: Optional[Dict[str, Any]] = None
    simplify_epsilon: Optional[float] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    clip_head_prob: Optional[float] = None
    clip_head_margin: Optional[float] = None


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

    @root_validator(skip_on_failure=True)
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
    # Optional masks aligned to detections (packed and base64-encoded to stay compact)
    masks: Optional[List[Dict[str, Any]]] = None


class Sam3TextPromptAutoResponse(BaseModel):
    detections: List[SamPointAutoResponse] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    image_token: Optional[str] = None


class QwenPromptSection(BaseModel):
    base_prompt: str
    default_image_type: str = "image"
    default_extra_context: str = ""

    @root_validator(skip_on_failure=True)
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
    random_split: Optional[bool] = None
    val_percent: Optional[float] = None
    split_seed: Optional[int] = None
    train_limit: Optional[int] = None
    val_limit: Optional[int] = None


class Sam3TrainRequest(BaseModel):
    dataset_id: str
    run_name: Optional[str] = None
    experiment_log_dir: Optional[str] = None
    train_batch_size: Optional[int] = None
    val_batch_size: Optional[int] = None
    num_train_workers: Optional[int] = None
    num_val_workers: Optional[int] = None
    max_epochs: Optional[int] = None
    resolution: Optional[int] = None
    lr_scale: Optional[float] = None
    gradient_accumulation_steps: Optional[int] = None
    val_epoch_freq: Optional[int] = None
    target_epoch_size: Optional[int] = None
    scheduler_warmup: Optional[int] = None
    scheduler_timescale: Optional[int] = None
    num_gpus: Optional[int] = None
    enable_inst_interactivity: Optional[bool] = None
    balance_classes: Optional[bool] = None
    balance_strategy: Optional[str] = None
    balance_power: Optional[float] = None
    balance_clip: Optional[float] = None
    balance_beta: Optional[float] = None
    balance_gamma: Optional[float] = None
    train_limit: Optional[int] = None
    val_limit: Optional[int] = None
    log_freq: Optional[int] = None
    log_every_batch: Optional[bool] = None
    enable_segmentation_head: Optional[bool] = None
    train_segmentation: Optional[bool] = None
    freeze_language_backbone: Optional[bool] = None
    language_backbone_lr: Optional[float] = None
    prompt_variants: Optional[Dict[str, Any]] = None
    prompt_randomize: Optional[bool] = None
    val_score_thresh: Optional[float] = None
    val_max_dets: Optional[int] = None
    random_split: Optional[bool] = None
    val_percent: Optional[float] = None
    split_seed: Optional[int] = None


class Sam3ModelActivateRequest(BaseModel):
    checkpoint_path: Optional[str] = None
    label: Optional[str] = None
    enable_segmentation: Optional[bool] = None


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


class SegmentationBuildRequest(BaseModel):
    source_dataset_id: str = Field(..., description="Existing bbox dataset id (Qwen or SAM3)")
    output_name: Optional[str] = Field(None, description="Optional output dataset name")
    sam_variant: Literal["sam1", "sam3"] = Field("sam3", description="Generator to use for masks")
    output_format: Literal["yolo-seg"] = Field("yolo-seg", description="Target mask encoding (polygons)")
    mask_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Mask probability threshold")
    score_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Box confidence threshold")
    simplify_epsilon: float = Field(30.0, ge=0.0, description="Polygon simplification epsilon (px)")
    min_size: float = Field(0.0, ge=0.0, description="Minimum mask area (px^2)")
    max_results: int = Field(1, ge=1, description="Max detections per box prompt")


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
SAM3_JOB_ROOT = Path(os.environ.get("SAM3_TRAINING_ROOT", "./uploads/sam3_runs"))
SAM3_JOB_ROOT.mkdir(parents=True, exist_ok=True)
SAM3_DATASET_ROOT = SAM3_JOB_ROOT / "datasets"
SAM3_DATASET_ROOT.mkdir(parents=True, exist_ok=True)
SAM3_DATASET_META_NAME = "sam3_dataset.json"
DATASET_REGISTRY_ROOT = Path(os.environ.get("DATASET_ROOT", "./uploads/datasets"))
DATASET_REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
DATASET_META_NAME = "dataset.json"
PROMPT_HELPER_JOB_ROOT = Path(os.environ.get("SAM3_PROMPT_HELPER_ROOT", "./uploads/prompt_helper_jobs"))
PROMPT_HELPER_JOB_ROOT.mkdir(parents=True, exist_ok=True)
SEG_BUILDER_ROOT = Path(os.environ.get("SEGMENTATION_ROOT", "./uploads/seg_runs"))
SEG_BUILDER_ROOT.mkdir(parents=True, exist_ok=True)
SAM3_REPO_ROOT = Path(__file__).resolve().parent.resolve()
SAM3_VENDOR_ROOT = SAM3_REPO_ROOT / "sam3"
SAM3_PACKAGE_ROOT = SAM3_VENDOR_ROOT / "sam3"
SAM3_CONFIG_TEMPLATE = SAM3_REPO_ROOT / "sam3_local" / "local_yolo_ft.yaml"
SAM3_GENERATED_CONFIG_DIR = SAM3_PACKAGE_ROOT / "train/configs/generated"
SAM3_GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
SAM3_BPE_PATH = SAM3_VENDOR_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
SAM3_MAX_LOG_LINES = 500
SAM3_MAX_METRIC_POINTS = 2000
SAM3_STORAGE_SCOPES = {"all", "checkpoints", "logs", "tensorboard", "dumps"}


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


@dataclass
class Sam3TrainingJob:
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
    process: Optional[subprocess.Popen] = None
    log_seq: int = 0


@dataclass
class SegmentationBuildJob:
    job_id: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    config: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class PromptHelperJob:
    job_id: str
    status: str = "queued"
    message: str = "Queued"
    progress: float = 0.0
    request: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    total_steps: int = 0
    completed_steps: int = 0
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class AgentMiningJob:
    job_id: str
    status: str = "queued"
    message: str = "Queued"
    progress: float = 0.0
    request: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_event: threading.Event = field(default_factory=threading.Event)


QWEN_TRAINING_JOBS: Dict[str, QwenTrainingJob] = {}
QWEN_TRAINING_JOBS_LOCK = threading.Lock()
SAM3_TRAINING_JOBS: Dict[str, Sam3TrainingJob] = {}
SAM3_TRAINING_JOBS_LOCK = threading.Lock()
SEGMENTATION_BUILD_JOBS: Dict[str, SegmentationBuildJob] = {}
SEGMENTATION_BUILD_JOBS_LOCK = threading.Lock()
PROMPT_HELPER_JOBS: Dict[str, PromptHelperJob] = {}
PROMPT_HELPER_JOBS_LOCK = threading.Lock()
AGENT_MINING_JOBS: Dict[str, AgentMiningJob] = {}
AGENT_MINING_JOBS_LOCK = threading.Lock()
UPLOAD_ROOT = Path("uploads")
UPLOAD_ROOT.mkdir(exist_ok=True)
PROMPT_HELPER_PRESET_ROOT = UPLOAD_ROOT / "prompt_helper_presets"
PROMPT_HELPER_PRESET_ROOT.mkdir(parents=True, exist_ok=True)
PROMPT_RECIPE_PRESET_ROOT = UPLOAD_ROOT / "prompt_recipe_presets"
PROMPT_RECIPE_PRESET_ROOT.mkdir(parents=True, exist_ok=True)
CLIP_DATASET_UPLOAD_ROOT = UPLOAD_ROOT / "clip_dataset_uploads"
CLIP_DATASET_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
DATASET_UPLOAD_ROOT = UPLOAD_ROOT / "dataset_uploads"
DATASET_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


def _prune_job_registry(registry: Dict[str, Any], lock: threading.Lock, ttl_hours: Optional[int] = None) -> None:
    if ttl_hours is None:
        ttl_hours = JOB_REGISTRY_TTL_HOURS
    ttl_seconds = max(0, ttl_hours) * 3600
    if ttl_seconds == 0:
        return
    now = time.time()
    terminal = {"completed", "failed", "cancelled"}
    with lock:
        to_delete: List[str] = []
        for job_id, job in list(registry.items()):
            status = getattr(job, "status", "")
            updated = getattr(job, "updated_at", getattr(job, "created_at", now))
            if status in {"running", "queued", "cancelling"}:
                continue
            if status and status not in terminal:
                continue
            try:
                if now - float(updated) > ttl_seconds:
                    to_delete.append(job_id)
            except Exception:
                continue
        for job_id in to_delete:
            registry.pop(job_id, None)


def _purge_staging_dirs(
    root: Path,
    *,
    ttl_hours: Optional[int] = None,
    active_roots: Optional[set[str]] = None,
    prefix: Optional[str] = None,
) -> Dict[str, int]:
    """Delete old staging directories that are not active. Returns stats."""
    stats = {"deleted": 0, "bytes": 0}
    if ttl_hours is None:
        ttl_hours = STAGING_TTL_HOURS
    if not root.exists() or ttl_hours <= 0:
        return stats
    cutoff = time.time() - ttl_hours * 3600
    active_roots = active_roots or set()
    for entry in root.iterdir():
        try:
            if not entry.is_dir():
                continue
            if prefix and not entry.name.startswith(prefix):
                continue
            if str(entry.resolve()) in active_roots:
                continue
            mtime = entry.stat().st_mtime
            if mtime > cutoff:
                continue
            stats["bytes"] += _purge_directory(entry)
            stats["deleted"] += 1
        except Exception:
            continue
    return stats
JOB_REGISTRY_TTL_HOURS = _env_int("JOB_REGISTRY_TTL_HOURS", 72)
STAGING_TTL_HOURS = _env_int("STAGING_TTL_HOURS", 24)
AGENT_MINING_ROOT = UPLOAD_ROOT / "agent_mining"
AGENT_MINING_ROOT.mkdir(parents=True, exist_ok=True)
AGENT_MINING_JOB_ROOT = AGENT_MINING_ROOT / "jobs"
AGENT_MINING_JOB_ROOT.mkdir(parents=True, exist_ok=True)
AGENT_MINING_CACHE_ROOT = AGENT_MINING_ROOT / "cache"
AGENT_MINING_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
AGENT_MINING_META_ROOT = AGENT_MINING_ROOT / "meta"
AGENT_MINING_META_ROOT.mkdir(parents=True, exist_ok=True)
AGENT_MINING_DET_CACHE_ROOT = AGENT_MINING_ROOT / "detections"
AGENT_MINING_DET_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
AGENT_MINING_RECIPES_ROOT = AGENT_MINING_ROOT / "recipes"
AGENT_MINING_RECIPES_ROOT.mkdir(parents=True, exist_ok=True)


def _purge_dataset_artifacts(dataset_id: str) -> None:
    """Remove per-dataset agent/prompt-helper artifacts."""
    safe_dataset = _normalise_relative_path(dataset_id)
    for derived_root in (
        AGENT_MINING_META_ROOT / safe_dataset,
        AGENT_MINING_DET_CACHE_ROOT / safe_dataset,
    ):
        try:
            shutil.rmtree(derived_root, ignore_errors=True)
        except Exception:
            pass
    try:
        for preset_path in PROMPT_HELPER_PRESET_ROOT.glob("*.json"):
            try:
                with preset_path.open("r", encoding="utf-8") as handle:
                    preset_data = json.load(handle)
                if preset_data.get("dataset_id") in {dataset_id, safe_dataset}:
                    preset_path.unlink(missing_ok=True)
            except Exception:
                continue
    except Exception:
        pass
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


def _sam3_job_log(job: Sam3TrainingJob, message: str) -> None:
    job.log_seq += 1
    entry = {"timestamp": time.time(), "message": message, "seq": job.log_seq}
    job.logs.append(entry)
    if len(job.logs) > SAM3_MAX_LOG_LINES:
        job.logs[:] = job.logs[-SAM3_MAX_LOG_LINES:]
    job.updated_at = time.time()
    try:
        logger.info("[sam3-train %s] %s", job.job_id[:8], message)
    except Exception:
        pass


def _sam3_job_append_metric(job: Sam3TrainingJob, metric: Dict[str, Any]) -> None:
    if not metric:
        return
    job.metrics.append(metric)
    if SAM3_MAX_METRIC_POINTS and len(job.metrics) > SAM3_MAX_METRIC_POINTS:
        job.metrics[:] = job.metrics[-SAM3_MAX_METRIC_POINTS :]
    job.updated_at = time.time()


def _sam3_job_update(
    job: Sam3TrainingJob,
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
                _sam3_job_log(job, message)
        else:
            job.message = message
    if progress is not None:
        job.progress = max(0.0, min(1.0, progress))
    if error is not None:
        job.error = error
    if result is not None:
        job.result = result
    job.updated_at = time.time()


def _serialize_sam3_job(job: Sam3TrainingJob) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "logs": job.logs,
        "metrics": job.metrics,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def _seg_job_log(job: SegmentationBuildJob, message: str) -> None:
    entry = {"timestamp": time.time(), "message": message}
    job.logs.append(entry)
    if len(job.logs) > MAX_JOB_LOGS:
        job.logs[:] = job.logs[-MAX_JOB_LOGS:]
    job.updated_at = time.time()
    try:
        logger.info("[seg-build %s] %s", job.job_id[:8], message)
    except Exception:
        pass


def _seg_job_update(
    job: SegmentationBuildJob,
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
                _seg_job_log(job, message)
        else:
            job.message = message
    if progress is not None:
        job.progress = max(0.0, min(1.0, progress))
    if error is not None:
        job.error = error
    if result is not None:
        job.result = result
    job.updated_at = time.time()


def _serialize_seg_job(job: SegmentationBuildJob) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "logs": job.logs,
        "config": job.config,
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


def _load_json_metadata(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read metadata file %s: %s", path, exc)
    return None


def _ensure_qwen_dataset_signature(dataset_dir: Path, metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    signature = metadata.get("signature")
    if signature:
        return metadata, str(signature)
    signature = _compute_dir_signature(dataset_dir)
    metadata["signature"] = signature
    _persist_qwen_dataset_metadata(dataset_dir, metadata)
    return metadata, signature


def _find_qwen_dataset_by_signature(signature: str) -> Optional[Path]:
    if not signature:
        return None
    for path in QWEN_DATASET_ROOT.iterdir():
        if not path.is_dir():
            continue
        meta = _load_qwen_dataset_metadata(path)
        if not meta:
            continue
        _, sig = _ensure_qwen_dataset_signature(path, meta)
        if sig == signature:
            return path
    return None


def _load_registry_dataset_metadata(dataset_dir: Path) -> Optional[Dict[str, Any]]:
    return _load_json_metadata(dataset_dir / DATASET_META_NAME)


def _persist_dataset_metadata(dataset_dir: Path, metadata: Dict[str, Any]) -> None:
    meta_path = dataset_dir / DATASET_META_NAME
    try:
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write dataset metadata for %s: %s", dataset_dir, exc)


def _coerce_dataset_metadata(dataset_dir: Path, raw_meta: Optional[Dict[str, Any]], source: str) -> Dict[str, Any]:
    meta = dict(raw_meta or {})
    updated = False
    if "id" not in meta:
        meta["id"] = dataset_dir.name
        updated = True
    if "label" not in meta:
        meta["label"] = meta["id"]
        updated = True
    dataset_type = meta.get("type") or meta.get("dataset_type") or "bbox"
    meta["type"] = dataset_type
    if "classes" not in meta:
        meta["classes"] = []
        updated = True
    if "context" not in meta and meta.get("dataset_context"):
        meta["context"] = meta.get("dataset_context") or ""
        updated = True
    if "created_at" not in meta:
        meta["created_at"] = dataset_dir.stat().st_mtime
        updated = True
    if "source" not in meta:
        meta["source"] = source
        updated = True
    signature = meta.get("signature")
    if not signature:
        signature = _compute_dir_signature(dataset_dir)
        meta["signature"] = signature
        updated = True
    if source == "registry" and updated:
        _persist_dataset_metadata(dataset_dir, meta)
    return meta


def _list_all_datasets(prefer_registry: bool = True) -> List[Dict[str, Any]]:
    """Collect datasets across registry, SAM3, and Qwen roots."""
    entries: List[Dict[str, Any]] = []
    seen: Dict[str, Tuple[int, str]] = {}
    sources = [
        ("registry", DATASET_REGISTRY_ROOT, _load_registry_dataset_metadata),
        ("sam3", SAM3_DATASET_ROOT, _load_sam3_dataset_metadata),
        ("qwen", QWEN_DATASET_ROOT, _load_qwen_dataset_metadata),
    ]
    for source, root, loader in sources:
        if not root.exists():
            continue
        for path in root.iterdir():
            if not path.is_dir():
                continue
            raw_meta = loader(path)
            if not raw_meta and source == "registry":
                raw_meta = _load_sam3_dataset_metadata(path) or _load_qwen_dataset_metadata(path)
            if not raw_meta:
                continue
            meta = _coerce_dataset_metadata(path, raw_meta, source)
            sam3_meta = _load_sam3_dataset_metadata(path) if source != "sam3" else meta
            coco_train = None
            coco_val = None
            coco_ready = False
            if sam3_meta:
                coco_train = sam3_meta.get("coco_train_json")
                coco_val = sam3_meta.get("coco_val_json")
                coco_ready = bool(coco_train and coco_val)
            signature = meta.get("signature") or ""
            key = signature or meta["id"]
            entry = {
                "id": meta.get("id") or path.name,
                "label": meta.get("label") or path.name,
                "dataset_root": str(path),
                "created_at": meta.get("created_at") or path.stat().st_mtime,
                "image_count": meta.get("image_count"),
                "train_count": meta.get("train_count"),
                "val_count": meta.get("val_count"),
                "classes": meta.get("classes", []),
                "context": meta.get("context", "") or meta.get("dataset_context", ""),
                "signature": signature,
                "source": meta.get("source") or source,
                "type": meta.get("type", "bbox"),
                "coco_ready": coco_ready,
                "coco_train_json": coco_train,
                "coco_val_json": coco_val,
            }
            existing = seen.get(key)
            if existing is not None:
                existing_idx, existing_origin = existing
                if prefer_registry:
                    if existing_origin == "registry":
                        continue
                    if source == "registry":
                        entries[existing_idx] = entry
                        seen[key] = (existing_idx, source)
                        continue
                continue
            seen[key] = (len(entries), source)
            entries.append(entry)
    entries.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return entries


def _load_sam3_dataset_metadata(dataset_dir: Path) -> Optional[Dict[str, Any]]:
    meta_path = dataset_dir / SAM3_DATASET_META_NAME
    data = _load_json_metadata(meta_path)
    if not data:
        return None
    # Backfill defaults for older datasets.
    updated = False
    if "id" not in data:
        data["id"] = dataset_dir.name
        updated = True
    if "type" not in data:
        data["type"] = "bbox"
        updated = True
    if updated:
        _persist_sam3_dataset_metadata(dataset_dir, data)
    return data


def _persist_sam3_dataset_metadata(dataset_dir: Path, metadata: Dict[str, Any]) -> None:
    meta_path = dataset_dir / SAM3_DATASET_META_NAME
    try:
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write SAM3 dataset metadata for %s: %s", dataset_dir, exc)


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except Exception:
                continue
    return total


def _active_run_paths_for_variant(variant: str) -> set[Path]:
    paths: set[Path] = set()
    with SAM3_TRAINING_JOBS_LOCK:
        jobs = list(SAM3_TRAINING_JOBS.values())
    for job in jobs:
        if job.status not in {"running", "queued", "cancelling"}:
            continue
        exp_dir = None
        try:
            exp_dir = job.config.get("paths", {}).get("experiment_log_dir")
        except Exception:
            exp_dir = None
        if exp_dir:
            try:
                paths.add(Path(exp_dir).resolve())
            except Exception:
                continue
    return paths


def _describe_run_dir(run_dir: Path, variant: str, active_paths: set[Path]) -> Dict[str, Any]:
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    tensorboard_dir = run_dir / "tensorboard"
    dumps_dir = run_dir / "dumps"
    marker_path = run_dir / ".promoted"
    promoted = False
    promoted_at: Optional[float] = None
    if marker_path.exists():
        promoted = True
        try:
            meta = json.loads(marker_path.read_text())
            promoted_at = meta.get("timestamp")
        except Exception:
            promoted_at = None
    checkpoints: List[Dict[str, Any]] = []
    if checkpoints_dir.exists():
        for ckpt in sorted(checkpoints_dir.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
            if ckpt.is_file():
                try:
                    stat = ckpt.stat()
                    checkpoints.append(
                        {
                            "file": ckpt.name,
                            "path": str(ckpt),
                            "size_bytes": stat.st_size,
                            "updated_at": stat.st_mtime,
                        }
                    )
                except Exception:
                    continue
    try:
        dir_stat = run_dir.stat()
        created_at = dir_stat.st_ctime
        updated_at = dir_stat.st_mtime
    except Exception:
        created_at = time.time()
        updated_at = created_at
    entry = {
        "id": run_dir.name,
        "variant": variant,
        "path": str(run_dir),
        "created_at": created_at,
        "updated_at": updated_at,
        "size_bytes": _dir_size_bytes(run_dir),
        "checkpoints_size_bytes": _dir_size_bytes(checkpoints_dir),
        "logs_size_bytes": _dir_size_bytes(logs_dir),
        "tensorboard_size_bytes": _dir_size_bytes(tensorboard_dir),
        "dumps_size_bytes": _dir_size_bytes(dumps_dir),
        "checkpoints": checkpoints,
        "active": run_dir.resolve() in active_paths,
        "promoted": promoted,
        "promoted_at": promoted_at,
    }
    return entry


def _list_sam3_runs(variant: str) -> List[Dict[str, Any]]:
    root = SAM3_JOB_ROOT
    if not root.exists():
        return []
    active_paths = _active_run_paths_for_variant(variant)
    runs: List[Dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        if not child.is_dir():
            continue
        # Skip dataset folder and other non-run directories under the root
        if variant == "sam3" and child.resolve() == SAM3_DATASET_ROOT.resolve():
            continue
        if child.name.lower() == "datasets":
            continue
        try:
            runs.append(_describe_run_dir(child, variant, active_paths))
        except Exception:
            continue
    return runs


def _run_dir_for_request(run_id: str, variant: str) -> Path:
    root = SAM3_JOB_ROOT
    candidate = (root / run_id).resolve()
    if not str(candidate).startswith(str(root.resolve())):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="invalid_run_id")
    if not candidate.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="sam3_run_not_found")
    return candidate


def _delete_run_scope(run_dir: Path, scope: str) -> Tuple[List[str], int]:
    targets: List[Path] = []
    if scope == "all":
        targets.append(run_dir)
    else:
        mapping = {
            "checkpoints": run_dir / "checkpoints",
            "logs": run_dir / "logs",
            "tensorboard": run_dir / "tensorboard",
            "dumps": run_dir / "dumps",
        }
        target = mapping.get(scope)
        if target:
            targets.append(target)
    deleted: List[str] = []
    freed = 0
    for target in targets:
        if not target.exists():
            continue
        freed += _dir_size_bytes(target)
        try:
            shutil.rmtree(target)
        except Exception:
            continue
        deleted.append(str(target))
    return deleted, freed


def _strip_checkpoint_optimizer(ckpt_path: Path) -> Tuple[bool, int, int]:
    """Remove optimizer/scheduler state from a torch checkpoint to shrink size."""
    before = ckpt_path.stat().st_size if ckpt_path.exists() else 0
    if not ckpt_path.exists() or before == 0:
        return False, before, before
    try:
        payload = torch.load(ckpt_path, map_location="cpu")
        removed = False
        for key in ["optimizer", "optimizers", "lr_schedulers", "schedulers", "trainer"]:
            if key in payload:
                payload.pop(key, None)
                removed = True
        if not removed:
            return False, before, before
        tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_size = tmp_path.stat().st_size
        tmp_path.replace(ckpt_path)
        return True, before, tmp_size
    except Exception:
        return False, before, before


def _promote_run(run_id: str, variant: str) -> Dict[str, Any]:
    run_dir = _run_dir_for_request(run_id, variant)
    active_paths = _active_run_paths_for_variant(variant)
    if run_dir.resolve() in active_paths:
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="sam3_run_active")
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="sam3_checkpoint_dir_missing")
    ckpts = [p for p in ckpt_dir.iterdir() if p.is_file() and p.suffix in {".ckpt", ".pth", ".pt"}]
    if not ckpts:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="sam3_checkpoints_missing")
    # choose keep candidate: prefer last.ckpt else newest
    keep = None
    for p in ckpts:
        if p.name == "last.ckpt":
            keep = p
            break
    if keep is None:
        keep = max(ckpts, key=lambda p: p.stat().st_mtime if p.exists() else 0)
    deleted = []
    freed = 0
    for p in ckpts:
        if p == keep:
            continue
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        try:
            p.unlink()
            deleted.append(str(p))
            freed += size
        except Exception:
            continue
    stripped, before, after = _strip_checkpoint_optimizer(keep)
    freed += max(0, before - after)
    marker = run_dir / ".promoted"
    try:
        marker.write_text(json.dumps({"timestamp": time.time(), "keep": str(keep)}), encoding="utf-8")
    except Exception:
        pass
    return {
        "kept": str(keep),
        "kept_size_bytes": keep.stat().st_size if keep.exists() else 0,
        "stripped_optimizer": stripped,
        "deleted": deleted,
        "freed_bytes": freed,
        "run_path": str(run_dir),
        "promoted": True,
        "promoted_at": time.time(),
    }


def _resolve_dataset_legacy(dataset_id: str) -> Path:
    cleaned = (dataset_id or "").strip().replace("\\", "/")
    safe = re.sub(r"[^A-Za-z0-9._/-]", "_", cleaned)
    candidate_qwen = (QWEN_DATASET_ROOT / safe).resolve()
    if str(candidate_qwen).startswith(str(QWEN_DATASET_ROOT.resolve())) and candidate_qwen.exists():
        return candidate_qwen
    candidate_sam3 = (SAM3_DATASET_ROOT / safe).resolve()
    if str(candidate_sam3).startswith(str(SAM3_DATASET_ROOT.resolve())) and candidate_sam3.exists():
        return candidate_sam3
    candidate_registry = (DATASET_REGISTRY_ROOT / safe).resolve()
    if (
        str(candidate_registry).startswith(str(DATASET_REGISTRY_ROOT.resolve()))
        and candidate_registry.exists()
    ):
        return candidate_registry
    raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="sam3_dataset_not_found")


def _resolve_sam3_or_qwen_dataset(dataset_id: str) -> Path:
    cleaned = (dataset_id or "").strip()
    for entry in _list_all_datasets():
        if cleaned in (entry.get("id"), entry.get("signature")):
            path = Path(entry["dataset_root"]).resolve()
            if path.exists():
                return path
    # Fallback to legacy per-root resolution
    return _resolve_dataset_legacy(dataset_id)


def _stable_hash(entries: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for item in entries:
        digest.update(item.encode("utf-8"))
    return digest.hexdigest()


def _decode_image_base64(
    image_base64: str,
    *,
    max_bytes: Optional[int] = BASE64_IMAGE_MAX_BYTES,
    max_dim: Optional[int] = BASE64_IMAGE_MAX_DIM,
    allow_downscale: bool = True,
) -> Tuple[Image.Image, np.ndarray]:
    """Decode base64 image with size/dimension guards and optional downscale."""
    if not image_base64:
        raise HTTPException(status_code=HTTP_428_PRECONDITION_REQUIRED, detail="image_payload_missing")
    raw = image_base64
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    if max_bytes:
        est_bytes = (len(raw) * 3) // 4
        if est_bytes > max_bytes * 2:
            raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="image_base64_too_large")
    try:
        data = base64.b64decode(raw)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"invalid_base64:{exc}") from exc
    if max_bytes and len(data) > max_bytes:
        raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="image_bytes_too_large")
    try:
        pil_img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"invalid_image:{exc}") from exc
    if max_dim:
        width, height = pil_img.size
        if width > max_dim or height > max_dim:
            if not allow_downscale:
                raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="image_too_large_dim")
            try:
                resample = getattr(Image, "Resampling", Image).LANCZOS  # Pillow 10 compat
            except Exception:
                resample = Image.LANCZOS
            pil_img = pil_img.copy()
            pil_img.thumbnail((max_dim, max_dim), resample)
    np_img = np.array(pil_img)
    return pil_img, np_img


def _compute_dir_signature(root: Path, *, allowed_exts: Optional[set[str]] = None) -> str:
    """Return a stable signature for all files under ``root``."""
    entries: List[str] = []
    if not root.exists():
        return ""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if allowed_exts is not None and path.suffix.lower() not in allowed_exts:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = path.relative_to(root)
        entries.append(f"{rel}:{stat.st_mtime_ns}:{stat.st_size}")
    return _stable_hash(entries)


def _path_is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def _agent_mining_meta_dir(dataset_id: str) -> Path:
    cleaned = (dataset_id or "").strip().replace("\\", "/").strip("/")
    safe = re.sub(r"[^A-Za-z0-9._/-]", "_", cleaned)
    meta_dir = (AGENT_MINING_META_ROOT / safe).resolve()
    if not _path_is_within_root(meta_dir, AGENT_MINING_META_ROOT.resolve()):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_mining_dataset_invalid")
    meta_dir.mkdir(parents=True, exist_ok=True)
    return meta_dir


def _agent_mining_cache_dir(dataset_id: str) -> Path:
    cleaned = (dataset_id or "").strip().replace("\\", "/").strip("/")
    safe = re.sub(r"[^A-Za-z0-9._/-]", "_", cleaned)
    cache_dir = (AGENT_MINING_DET_CACHE_ROOT / safe).resolve()
    if not _path_is_within_root(cache_dir, AGENT_MINING_DET_CACHE_ROOT.resolve()):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_mining_dataset_invalid")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _normalize_agent_recipe_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    extra_keys = {
        "similarity_score",
        "seed_prompt",
        "fps",
        "gain",
        "source",
        "precision",
        "recall",
        "coverage",
        "duplicates",
    }
    for step in steps:
        prompt = step.get("prompt")
        threshold = step.get("threshold")
        has_exemplar = step.get("exemplar") is not None
        if (prompt is None and not has_exemplar) or threshold is None:
            continue
        try:
            thr_val = float(threshold)
        except Exception:
            continue
        if math.isnan(thr_val) or thr_val < 0.0 or thr_val > 1.0:
            continue
        entry = {
            "prompt": "" if prompt is None else str(prompt),
            "threshold": thr_val,
            "type": step.get("type"),
            "exemplar": dict(step["exemplar"]) if isinstance(step.get("exemplar"), dict) else step.get("exemplar"),
        }
        sim_raw = step.get("similarity_score")
        if sim_raw is not None:
            try:
                sim_val = float(sim_raw)
            except Exception:
                sim_val = None
            if sim_val is not None and 0.0 <= sim_val <= 1.0:
                entry["similarity_score"] = sim_val
        for key in extra_keys:
            if key in entry:
                continue
            if key in step:
                entry[key] = step[key]
        normalized.append(entry)
    return normalized


def _validate_agent_recipe_structure(recipe_obj: Dict[str, Any]) -> None:
    """Lightweight schema guard to avoid accepting malformed recipes."""
    if not isinstance(recipe_obj, dict):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
    # New greedy recipe format: prompt bank + positive/negative crop banks.
    if recipe_obj.get("mode") == "sam3_greedy" or isinstance(recipe_obj.get("text_prompts"), list) or isinstance(recipe_obj.get("positives"), list):
        text_prompts = recipe_obj.get("text_prompts")
        positives = recipe_obj.get("positives")
        negatives = recipe_obj.get("negatives")
        steps = recipe_obj.get("steps")
        if text_prompts is not None and not isinstance(text_prompts, list):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
        if positives is not None and not isinstance(positives, list):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
        if negatives is not None and not isinstance(negatives, list):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
        if steps is not None and not isinstance(steps, list):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
        if isinstance(text_prompts, list):
            for p in text_prompts:
                if not isinstance(p, str):
                    raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
        for crop_list in (positives, negatives):
            if not isinstance(crop_list, list):
                continue
            for ex in crop_list:
                if not isinstance(ex, dict):
                    raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
        if not (text_prompts or positives or steps):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
        return
    steps = recipe_obj.get("steps")
    if not isinstance(steps, list) or not steps:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
    for step in steps:
        if not isinstance(step, dict):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
        if "threshold" not in step:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
        try:
            thr_val = float(step.get("threshold"))
        except Exception:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
        if math.isnan(thr_val) or thr_val < 0.0 or thr_val > 1.0:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
        if "prompt" not in step and "exemplar" not in step:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
        if "exemplar" in step and step["exemplar"] is not None and not isinstance(step["exemplar"], dict):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
        sim_val = step.get("similarity_score")
        if sim_val is not None:
            try:
                sim_val_f = float(sim_val)
            except Exception:
                raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
            if math.isnan(sim_val_f) or sim_val_f < 0.0 or sim_val_f > 1.0:
                raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
    negatives = recipe_obj.get("negatives")
    if negatives is not None and not isinstance(negatives, list):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
    if isinstance(negatives, list):
        for neg in negatives:
            if not isinstance(neg, dict):
                raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")


def _compute_labelmap_hash(categories: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    names: List[Tuple[int, str]] = []
    for idx, cat in enumerate(categories):
        try:
            cid = int(cat.get("id", idx))
        except Exception:
            cid = idx
        names.append((cid, str(cat.get("name", f"class_{cid}"))))
    names.sort(key=lambda c: c[0])
    labels = [name for _, name in names]
    try:
        digest = hashlib.sha256("|".join(labels).encode("utf-8")).hexdigest()[:12]
    except Exception:
        digest = "unknown"
    return digest, labels


def _compute_dataset_signature(dataset_id: str, dataset_root: Path, images: Dict[int, Dict[str, Any]], categories: List[Dict[str, Any]]) -> str:
    """
    Create a location-agnostic signature for portability:
    - dataset_id
    - counts of images/categories
    - hashes of category names (sorted)
    - hashes of image file names (sorted)
    """
    try:
        cat_names = [str(c.get("name", f"class_{idx}")) for idx, c in enumerate(categories)]
        cat_hash = hashlib.sha256("|".join(sorted(cat_names)).encode("utf-8")).hexdigest()[:12]
        file_names = [Path(info.get("file_name") or "").name for info in images.values() if info.get("file_name")]
        file_hash = hashlib.sha256("|".join(sorted(file_names)).encode("utf-8")).hexdigest()[:12]
        payload = f"{dataset_id}|{len(images)}|{len(categories)}|{cat_hash}|{file_hash}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    except Exception:
        return "unknown"


def _save_exemplar_crop(
    *,
    exemplar: Dict[str, Any],
    images: Dict[int, Dict[str, Any]],
    crop_dir: Path,
    step_idx: int,
    crop_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Persist a single exemplar crop to disk and return enriched metadata."""
    img_id = exemplar.get("image_id")
    if img_id is None:
        return None
    info = images.get(int(img_id))
    if not info:
        return None
    bbox = exemplar.get("bbox")
    if not bbox or len(bbox) < 4:
        return None
    try:
        x, y, w, h = map(float, bbox[:4])
    except Exception:
        return None
    try:
        img_path = info.get("path")
        if not img_path:
            return None
        with Image.open(img_path) as pil_img:
            pil_img = pil_img.convert("RGB")
            width, height = pil_img.width, pil_img.height
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(width, x + w)
            y1 = min(height, y + h)
            crop = pil_img.crop((x0, y0, x1, y1))
            crop_dir.mkdir(parents=True, exist_ok=True)
            filename = crop_name or f"step_{step_idx:02d}_exemplar.png"
            crop_path = crop_dir / filename
            crop.save(crop_path, format="PNG")
    except Exception:
        return None
    bbox_norm = None
    try:
        bbox_norm = [x / width, y / height, w / width, h / height]
    except Exception:
        bbox_norm = None
    enriched = {
        **exemplar,
        "bbox": [x, y, w, h],
        "bbox_xyxy": [x0, y0, x1, y1],
        "bbox_norm": bbox_norm,
        "image_size": [width, height],
        "crop_path": str(Path("crops") / crop_path.name),
        "crop_size": [crop.width, crop.height],
    }
    return enriched


def _persist_agent_recipe(
    dataset_id: Optional[str],
    class_id: Optional[int],
    class_name: Optional[str],
    label: str,
    recipe: Dict[str, Any],
    *,
    crop_overrides: Optional[Dict[str, bytes]] = None,
    clip_head_overrides: Optional[Dict[str, bytes]] = None,
    meta_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(recipe, dict) or not recipe:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_empty")
    # Accept either a raw recipe body, or a wrapper containing {"recipe": {...}} (e.g., imported payload).
    recipe_body: Dict[str, Any] = recipe
    if (
        isinstance(recipe.get("recipe"), dict)
        and not any(k in recipe for k in ("steps", "text_prompts", "positives", "mode"))
    ):
        recipe_body = recipe.get("recipe") or {}
    if not isinstance(recipe_body, dict) or not recipe_body:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_empty")
    _validate_agent_recipe_structure(recipe_body)
    cleaned_label = label.strip() or "agent_recipe"
    recipe_id = f"ar_{uuid.uuid4().hex[:8]}"
    images: Dict[int, Dict[str, Any]] = {}
    categories: List[Dict[str, Any]] = []
    dataset_signature: Optional[str] = None
    labelmap_hash: Optional[str] = None
    labelmap_entries: Optional[List[str]] = None
    dataset_root: Optional[Path] = None
    dataset_id_clean = (dataset_id or "").strip()
    try:
        if dataset_id_clean:
            dataset_root = _resolve_sam3_or_qwen_dataset(dataset_id_clean)
            coco, _, images = _load_coco_index(dataset_root)
            categories = coco.get("categories") or []
            dataset_signature = _compute_dataset_signature(dataset_id_clean, dataset_root, images, categories)
            labelmap_hash, labelmap_entries = _compute_labelmap_hash(categories)
            if class_id is not None:
                try:
                    cid = int(class_id)
                except Exception:
                    cid = None
                if cid is not None:
                    found = any(int(cat.get("id", idx)) == cid for idx, cat in enumerate(categories))
                    if not found and not crop_overrides:
                        raise HTTPException(status_code=HTTP_412_PRECONDITION_FAILED, detail="agent_recipe_class_missing")
    except HTTPException:
        if not crop_overrides and not meta_overrides:
            raise
    except Exception:
        # Allow portability when importing with embedded crops; we'll fall back to meta overrides.
        pass
    if not dataset_signature and meta_overrides:
        dataset_signature = meta_overrides.get("dataset_signature")
    if not labelmap_hash and meta_overrides:
        labelmap_hash = meta_overrides.get("labelmap_hash")
        labelmap_entries = meta_overrides.get("labelmap")
    if not labelmap_entries:
        raise HTTPException(status_code=HTTP_412_PRECONDITION_FAILED, detail="agent_recipe_labelmap_missing")
    steps_raw = _normalize_agent_recipe_steps(recipe_body.get("steps") or [])
    text_prompts_raw = recipe_body.get("text_prompts")
    positives_raw = recipe_body.get("positives")
    negatives_raw = recipe_body.get("negatives")
    if text_prompts_raw is None:
        text_prompts_raw = recipe.get("text_prompts")
    if positives_raw is None:
        positives_raw = recipe.get("positives")
    if negatives_raw is None:
        negatives_raw = recipe.get("negatives")
    text_prompts: List[str] = []
    if isinstance(text_prompts_raw, list):
        text_prompts = _sanitize_prompts([str(p) for p in text_prompts_raw if str(p).strip()])
    positives_list: List[Dict[str, Any]] = [p for p in (positives_raw or []) if isinstance(p, dict)] if isinstance(positives_raw, list) else []
    negatives_list: List[Dict[str, Any]] = [n for n in (negatives_raw or []) if isinstance(n, dict)] if isinstance(negatives_raw, list) else []
    is_greedy = bool(recipe_body.get("mode") == "sam3_greedy" or text_prompts or positives_list)
    if not (steps_raw or text_prompts or positives_list):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_empty")
    recipe_dir = AGENT_MINING_RECIPES_ROOT / recipe_id
    cleanup_recipe_dir = True
    try:
        crops_dir = recipe_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        # Optional portable CLIP head artifacts (embedded into the recipe package).
        clip_head_cfg_raw: Optional[Dict[str, Any]] = None
        if isinstance(recipe_body.get("clip_head"), dict):
            clip_head_cfg_raw = recipe_body.get("clip_head")
        elif isinstance(recipe.get("clip_head"), dict):
            clip_head_cfg_raw = recipe.get("clip_head")

        clip_head_classifier_path: Optional[str] = None
        for src in (recipe_body, recipe):
            if isinstance(src, dict) and isinstance(src.get("_clip_head_classifier_path"), str):
                clip_head_classifier_path = str(src.get("_clip_head_classifier_path"))
                break

        clip_head_written = False
        clip_dir = recipe_dir / "clip_head"
        head_npz_bytes = None
        head_meta_bytes = None
        if clip_head_overrides:
            head_npz_bytes = clip_head_overrides.get("clip_head/head.npz") or clip_head_overrides.get("head.npz")
            head_meta_bytes = clip_head_overrides.get("clip_head/meta.json") or clip_head_overrides.get("meta.json")
        if head_npz_bytes:
            try:
                clip_dir.mkdir(parents=True, exist_ok=True)
                (clip_dir / "head.npz").write_bytes(head_npz_bytes)
                clip_head_written = True
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"agent_recipe_clip_head_write_failed:{exc}") from exc
        if head_meta_bytes:
            try:
                clip_dir.mkdir(parents=True, exist_ok=True)
                (clip_dir / "meta.json").write_bytes(head_meta_bytes)
                clip_head_written = True
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"agent_recipe_clip_head_meta_write_failed:{exc}") from exc
        if not clip_head_written and clip_head_classifier_path:
            resolved_classifier = _resolve_agent_clip_classifier_path(clip_head_classifier_path)
            if resolved_classifier is not None:
                head = _load_clip_head_from_classifier(resolved_classifier)
                if head is not None:
                    min_prob = 0.5
                    margin = 0.0
                    if clip_head_cfg_raw:
                        try:
                            if clip_head_cfg_raw.get("min_prob") is not None:
                                min_prob = float(clip_head_cfg_raw.get("min_prob"))
                            if clip_head_cfg_raw.get("margin") is not None:
                                margin = float(clip_head_cfg_raw.get("margin"))
                        except Exception:
                            min_prob = 0.5
                            margin = 0.0
                    _save_clip_head_artifacts(recipe_dir=recipe_dir, head=head, min_prob=min_prob, margin=margin)
                    clip_head_written = True

        clip_head_cfg_clean: Optional[Dict[str, Any]] = None
        if clip_head_written:
            loaded = _load_clip_head_artifacts(recipe_dir=recipe_dir, fallback_meta=clip_head_cfg_raw)
            if loaded is not None:
                min_prob = 0.5
                margin = 0.0
                if loaded.get("min_prob") is not None:
                    try:
                        min_prob = float(loaded.get("min_prob"))
                    except Exception:
                        min_prob = 0.5
                if loaded.get("margin") is not None:
                    try:
                        margin = float(loaded.get("margin"))
                    except Exception:
                        margin = 0.0
                clip_head_cfg_clean = {
                    "artifact": "clip_head/head.npz",
                    "clip_model": loaded.get("clip_model"),
                    "proba_mode": loaded.get("proba_mode"),
                    "classes": loaded.get("classes") if isinstance(loaded.get("classes"), list) else [],
                    "min_prob": float(max(0.0, min(1.0, min_prob))),
                    "margin": float(max(0.0, min(1.0, margin))),
                }
            try:
                total = 0
                if clip_dir.exists():
                    for f in clip_dir.iterdir():
                        if f.is_file():
                            total += f.stat().st_size
                if total > AGENT_RECIPE_MAX_CLIP_HEAD_BYTES:
                    raise HTTPException(
                        status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="agent_recipe_clip_head_too_large",
                    )
            except HTTPException:
                raise
            except Exception:
                pass
        def _safe_crop_filename(preferred: Optional[str], prefix: str, idx: int) -> str:
            try:
                name = Path(str(preferred)).name if preferred else ""
            except Exception:
                name = ""
            if not name:
                name = f"{prefix}_{idx:03d}.png"
            if not name.lower().endswith(".png"):
                name = f"{name}.png"
            name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
            base, ext = os.path.splitext(name)
            if not ext:
                ext = ".png"
            candidate = f"{base}{ext}"
            counter = 1
            while (crops_dir / candidate).exists():
                counter += 1
                candidate = f"{base}_{counter}{ext}"
            return candidate

        def _materialize_crop_entry(entry: Dict[str, Any], *, prefix: str, idx: int, fallback_step_idx: int) -> Optional[Dict[str, Any]]:
            """Materialize a crop into crops_dir and return a portable entry dict."""
            entry_copy = dict(entry)
            entry_copy.pop("crop_base64", None)
            crop_key = entry_copy.get("crop_path") or entry_copy.get("path")
            crop_bytes = None
            if crop_overrides and crop_key:
                crop_bytes = crop_overrides.get(str(crop_key))
                if crop_bytes is None:
                    try:
                        crop_bytes = crop_overrides.get(str(Path("crops") / Path(str(crop_key)).name))
                    except Exception:
                        crop_bytes = None
            filename = _safe_crop_filename(str(crop_key) if crop_key else None, prefix, idx)
            if crop_bytes is not None:
                crop_path = crops_dir / filename
                try:
                    with crop_path.open("wb") as fp:
                        fp.write(crop_bytes)
                    entry_copy["crop_path"] = str(Path("crops") / crop_path.name)
                    entry_copy.pop("path", None)
                    entry_copy.pop("embed_id", None)
                    entry_copy.pop("crop_base64", None)
                    return entry_copy
                except Exception:
                    # Fall back to a portable dict without guarantees if write fails.
                    entry_copy.pop("path", None)
                    entry_copy.pop("embed_id", None)
                    entry_copy.pop("crop_base64", None)
                    return entry_copy
            enriched = None
            if images:
                enriched = _save_exemplar_crop(
                    exemplar=entry_copy,
                    images=images,
                    crop_dir=crops_dir,
                    step_idx=fallback_step_idx,
                    crop_name=filename,
                )
            if enriched is None:
                entry_copy.pop("path", None)
                entry_copy.pop("embed_id", None)
                entry_copy.pop("crop_base64", None)
                # Ensure crop_path, if present, is made portable.
                if crop_key:
                    try:
                        entry_copy["crop_path"] = str(Path("crops") / Path(str(crop_key)).name)
                    except Exception:
                        pass
                return entry_copy
            enriched.pop("path", None)
            enriched.pop("embed_id", None)
            enriched.pop("crop_base64", None)
            return enriched

        portable_steps: List[Dict[str, Any]] = []
        portable_positives: List[Dict[str, Any]] = []
        portable_negatives: List[Dict[str, Any]] = []
        for idx, step in enumerate(steps_raw, start=1):
            entry = dict(step)
            ex = step.get("exemplar")
            if ex:
                enriched = None
                # Prefer provided crops if present (e.g., imported package), else derive from dataset.
                crop_key = None
                if isinstance(ex, dict):
                    crop_key = ex.get("crop_path")
                    crop_bytes = None
                    if crop_overrides and crop_key:
                        crop_bytes = crop_overrides.get(crop_key)
                        if crop_bytes is None:
                            try:
                                alt_key = str(Path("crops") / Path(crop_key).name)
                                crop_bytes = crop_overrides.get(alt_key)
                            except Exception:
                                crop_bytes = None
                    if crop_bytes is not None:
                        crop_path = crops_dir / Path(crop_key).name
                        try:
                            with crop_path.open("wb") as fp:
                                fp.write(crop_bytes)
                            if crop_path.exists():
                                pass
                            enriched = {
                                **ex,
                                "crop_path": str(Path("crops") / crop_path.name),
                            }
                        except Exception:
                            enriched = dict(ex)
                if enriched is None and images and isinstance(ex, dict):
                    enriched = _save_exemplar_crop(exemplar=ex, images=images, crop_dir=crops_dir, step_idx=idx)
                if enriched is None and isinstance(ex, dict):
                    enriched = dict(ex)
                entry["exemplar"] = enriched
            portable_steps.append(entry)

        # Greedy-mode crop banks.
        if is_greedy and positives_list:
            for p_idx, pos in enumerate(positives_list, start=1):
                enriched_pos = _materialize_crop_entry(pos, prefix="pos", idx=p_idx, fallback_step_idx=2000 + p_idx)
                if enriched_pos:
                    portable_positives.append(enriched_pos)
        for n_idx, neg in enumerate(negatives_list, start=1):
            enriched_neg = _materialize_crop_entry(neg, prefix="neg", idx=n_idx, fallback_step_idx=3000 + n_idx)
            if enriched_neg:
                portable_negatives.append(enriched_neg)
        def _normalize_crop_path(path_str: Optional[str]) -> Optional[str]:
            if not path_str:
                return None
            try:
                return str(Path("crops") / Path(path_str).name)
            except Exception:
                return None

        # Normalize crop paths in steps and negatives for portability.
        for entry in portable_steps:
            ex = entry.get("exemplar")
            if isinstance(ex, dict) and ex.get("crop_path"):
                normalized = _normalize_crop_path(ex.get("crop_path"))
                if normalized:
                    ex["crop_path"] = normalized
                ex.pop("path", None)
                ex.pop("crop_base64", None)
        for entry in portable_positives:
            if isinstance(entry, dict) and entry.get("crop_path"):
                normalized = _normalize_crop_path(entry.get("crop_path"))
                if normalized:
                    entry["crop_path"] = normalized
                entry.pop("path", None)
                entry.pop("crop_base64", None)
        for neg in portable_negatives:
            if isinstance(neg, dict) and neg.get("crop_path"):
                normalized = _normalize_crop_path(neg.get("crop_path"))
                if normalized:
                    neg["crop_path"] = normalized
                neg.pop("path", None)
                neg.pop("crop_base64", None)

        # Enforce crop count/byte limits after all crops are materialized.
        def _assert_crop_limits() -> None:
            if not crops_dir.exists():
                return
            count = 0
            total = 0
            try:
                for cf in crops_dir.glob("*.png"):
                    count += 1
                    try:
                        total += cf.stat().st_size
                    except Exception:
                        continue
                if count > AGENT_RECIPE_MAX_CROPS or total > AGENT_RECIPE_MAX_CROP_BYTES:
                    raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="agent_recipe_crops_too_large")
            except HTTPException as exc:
                raise exc

        _assert_crop_limits()

        params_src = recipe_body.get("params")
        if not isinstance(params_src, dict):
            params_src = recipe.get("params") if isinstance(recipe.get("params"), dict) else None
        params = params_src or {
            "mask_threshold": recipe_body.get("mask_threshold", recipe.get("mask_threshold")),
            "min_size": recipe_body.get("min_size", recipe.get("min_size")),
            "simplify_epsilon": recipe_body.get("simplify_epsilon", recipe.get("simplify_epsilon")),
            "max_results": recipe_body.get("max_results", recipe.get("max_results")),
            "similarity_score": recipe_body.get("similarity_score", recipe.get("similarity_score")),
            "seed_threshold": recipe_body.get("seed_threshold", recipe.get("seed_threshold")),
            "expand_threshold": recipe_body.get("expand_threshold", recipe.get("expand_threshold")),
            "max_visual_seeds": recipe_body.get("max_visual_seeds", recipe.get("max_visual_seeds")),
            "seed_dedupe_iou": recipe_body.get("seed_dedupe_iou", recipe.get("seed_dedupe_iou")),
            "dedupe_iou": recipe_body.get("dedupe_iou", recipe.get("dedupe_iou")),
            "use_clip_fp_guard": recipe_body.get("use_clip_fp_guard", recipe.get("use_clip_fp_guard")),
            "use_negative_exemplars": recipe_body.get("use_negative_exemplars", recipe.get("use_negative_exemplars")),
            "negative_strength": recipe_body.get("negative_strength", recipe.get("negative_strength")),
        }
        thresholds = sorted({float(s.get("threshold", 0.0)) for s in portable_steps if s.get("threshold") is not None})
        if thresholds:
            params["thresholds"] = thresholds
        payload = {
            "id": recipe_id,
            "dataset_id": dataset_id,
            "dataset_signature": dataset_signature,
            "labelmap_hash": labelmap_hash,
            "labelmap": labelmap_entries,
            "class_id": class_id,
            "class_name": class_name,
            "label": cleaned_label,
            "created_at": time.time(),
            "params": params,
            "recipe": {
                "mode": recipe_body.get("mode") or ("sam3_greedy" if is_greedy else None),
                "text_prompts": text_prompts if text_prompts else None,
                "positives": portable_positives if portable_positives else None,
                "steps": portable_steps,
                "negatives": portable_negatives,
                "clip_head": clip_head_cfg_clean,
                "summary": recipe_body.get("summary") or recipe.get("summary"),
            },
        }
        path = (AGENT_MINING_RECIPES_ROOT / f"{recipe_id}.json").resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not _path_is_within_root(path, AGENT_MINING_RECIPES_ROOT.resolve()):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_path_invalid")
        try:
            with path.open("w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)
            # Persist a portable zip alongside the JSON for download.
            zip_path = (AGENT_MINING_RECIPES_ROOT / f"{recipe_id}.zip").resolve()
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("recipe.json", json.dumps(payload, ensure_ascii=False, indent=2))
                if crops_dir.exists():
                    for crop_file in crops_dir.glob("*.png"):
                        try:
                            zf.write(crop_file, arcname=f"crops/{crop_file.name}")
                        except Exception:
                            continue
                clip_dir = recipe_dir / "clip_head"
                if clip_dir.exists():
                    for artifact in clip_dir.iterdir():
                        try:
                            if not artifact.is_file():
                                continue
                            if artifact.name not in {"head.npz", "meta.json"}:
                                continue
                            zf.write(artifact, arcname=f"clip_head/{artifact.name}")
                        except Exception:
                            continue
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"agent_recipe_save_failed:{exc}") from exc
        payload["_path"] = str(path)
        payload["_zip"] = str((AGENT_MINING_RECIPES_ROOT / f"{recipe_id}.zip").resolve())
        cleanup_recipe_dir = False
        return payload
    finally:
        if cleanup_recipe_dir:
            try:
                shutil.rmtree(recipe_dir, ignore_errors=True)
            except Exception:
                pass


def _load_agent_recipe(recipe_id: str) -> Dict[str, Any]:
    path = (AGENT_MINING_RECIPES_ROOT / f"{recipe_id}.json").resolve()
    if not _path_is_within_root(path, AGENT_MINING_RECIPES_ROOT.resolve()) or not path.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="agent_recipe_not_found")
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
        if not isinstance(data.get("recipe"), dict):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_invalid_schema")
        _validate_agent_recipe_structure(data.get("recipe") or {})
        data["_path"] = str(path)
        zip_path = (AGENT_MINING_RECIPES_ROOT / f"{recipe_id}.zip").resolve()
        if zip_path.exists():
            data["_zip"] = str(zip_path)
        # Inline a small number of crop previews if present on disk (kept small so
        # /agent_mining/apply payloads don't explode when the UI forwards recipes).
        recipe_block = data.get("recipe") or {}
        recipe_dir = (AGENT_MINING_RECIPES_ROOT / recipe_id).resolve()
        if not _path_is_within_root(recipe_dir, AGENT_MINING_RECIPES_ROOT.resolve()):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_path_invalid")
        crop_dir = (recipe_dir / "crops").resolve()
        if not _path_is_within_root(crop_dir, recipe_dir):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_path_invalid")
        max_inline = 8
        inlined = 0

        def _inline_crop(entry: Dict[str, Any]) -> None:
            nonlocal inlined
            if inlined >= max_inline:
                return
            crop_path = entry.get("crop_path")
            if not crop_path or not isinstance(crop_path, str):
                return
            try:
                crop_name = Path(crop_path).name
            except Exception:
                crop_name = ""
            abs_path = (crop_dir / crop_name).resolve() if crop_name else None
            if abs_path and _path_is_within_root(abs_path, crop_dir) and abs_path.exists() and abs_path.is_file():
                try:
                    with abs_path.open("rb") as cfp:
                        b64 = base64.b64encode(cfp.read()).decode("ascii")
                    entry["crop_base64"] = f"data:image/png;base64,{b64}"
                    entry["crop_path"] = str(Path("crops") / crop_name)
                    inlined += 1
                except Exception:
                    return
            else:
                # Normalize to relative path in case it was absolute originally.
                try:
                    entry["crop_path"] = f"crops/{Path(crop_path).name}"
                except Exception:
                    return

        steps = recipe_block.get("steps") or []
        if isinstance(steps, list):
            for step in steps:
                ex = step.get("exemplar") if isinstance(step, dict) else None
                if isinstance(ex, dict):
                    _inline_crop(ex)
        positives = recipe_block.get("positives") or []
        if isinstance(positives, list):
            for ex in positives:
                if isinstance(ex, dict):
                    _inline_crop(ex)
        negatives = recipe_block.get("negatives") or []
        if isinstance(negatives, list):
            for ex in negatives:
                if isinstance(ex, dict):
                    _inline_crop(ex)
        return data
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"agent_recipe_load_failed:{exc}") from exc


def _delete_agent_recipe(recipe_id: str) -> None:
    json_path = (AGENT_MINING_RECIPES_ROOT / f"{recipe_id}.json").resolve()
    zip_path = (AGENT_MINING_RECIPES_ROOT / f"{recipe_id}.zip").resolve()
    recipe_dir = (AGENT_MINING_RECIPES_ROOT / recipe_id).resolve()
    if not _path_is_within_root(json_path, AGENT_MINING_RECIPES_ROOT.resolve()):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_path_invalid")
    removed_any = False
    for path in (json_path, zip_path):
        if path.exists():
            try:
                path.unlink()
                removed_any = True
            except Exception:
                pass
    if recipe_dir.exists() and recipe_dir.is_dir():
        try:
            shutil.rmtree(recipe_dir)
            removed_any = True
        except Exception:
            pass
    if not removed_any:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="agent_recipe_not_found")


def _list_agent_recipes(dataset_id: Optional[str] = None) -> List[Dict[str, Any]]:
    recipes: List[Dict[str, Any]] = []
    for path in AGENT_MINING_RECIPES_ROOT.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            if dataset_id and data.get("dataset_id") != dataset_id:
                continue
            data["_path"] = str(path)
            zip_path = (AGENT_MINING_RECIPES_ROOT / f"{data.get('id','')}.zip").resolve()
            if zip_path.exists():
                data["_zip"] = str(zip_path)
            recipes.append(data)
        except Exception:
            continue
    recipes.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return recipes


def _ensure_recipe_zip(recipe: Dict[str, Any]) -> Path:
    recipe_id = recipe.get("id")
    if not recipe_id:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_missing_id")
    zip_path = (AGENT_MINING_RECIPES_ROOT / f"{recipe_id}.zip").resolve()
    if zip_path.exists():
        return zip_path
    recipe_dir = AGENT_MINING_RECIPES_ROOT / recipe_id
    crops_dir = recipe_dir / "crops"
    clip_head_dir = recipe_dir / "clip_head"
    try:
        # Never embed crop_base64 blobs inside the portable zip JSON; the PNGs are included separately.
        def _strip_unportable_fields(obj: Any) -> None:
            if isinstance(obj, dict):
                # UI-only / internal fields should not ship in portable zips.
                for k in list(obj.keys()):
                    if isinstance(k, str) and k.startswith("_"):
                        obj.pop(k, None)
                obj.pop("crop_base64", None)
                for v in obj.values():
                    _strip_unportable_fields(v)
            elif isinstance(obj, list):
                for v in obj:
                    _strip_unportable_fields(v)

        clean_recipe = json.loads(json.dumps(recipe))
        _strip_unportable_fields(clean_recipe)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("recipe.json", json.dumps(clean_recipe, ensure_ascii=False, indent=2))
            if crops_dir.exists():
                for crop_file in crops_dir.glob("*.png"):
                    try:
                        zf.write(crop_file, arcname=f"crops/{crop_file.name}")
                    except Exception:
                        continue
            if clip_head_dir.exists():
                for artifact in clip_head_dir.iterdir():
                    try:
                        if not artifact.is_file():
                            continue
                        if artifact.name not in {"head.npz", "meta.json"}:
                            continue
                        zf.write(artifact, arcname=f"clip_head/{artifact.name}")
                    except Exception:
                        continue
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"agent_recipe_export_failed:{exc}") from exc
    return zip_path


def _bbox_to_xyxy_pixels(
    bbox: Sequence[float],
    img_w: int,
    img_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    if not bbox or len(bbox) < 4:
        return None
    try:
        bbox4 = list(map(float, bbox[:4]))
    except Exception:
        return None
    if all(0.0 <= v <= 1.0 for v in bbox4):  # YOLO normalized (cx, cy, w, h)
        try:
            left, top, right, bottom = yolo_to_corners(bbox4, img_w, img_h)
            return float(left), float(top), float(right), float(bottom)
        except Exception:
            return None
    # Pixel xywh
    try:
        x, y, w, h = bbox4
    except Exception:
        return None
    x1 = float(x)
    y1 = float(y)
    x2 = float(x + w)
    y2 = float(y + h)
    return x1, y1, x2, y2


def _iou_xyxy(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _dedupe_qwen_detections_iou(
    dets: Sequence[QwenDetection],
    *,
    img_w: int,
    img_h: int,
    iou_thresh: float,
) -> List[QwenDetection]:
    """Simple NMS-style dedupe across a list of detections, keeping highest-score boxes."""
    iou_thresh = float(max(0.0, min(1.0, iou_thresh)))
    if not dets:
        return []
    boxes: List[Tuple[float, float, float, float]] = []
    scored: List[Tuple[float, int]] = []
    for idx, det in enumerate(dets):
        bbox = det.bbox or []
        box_xyxy = _bbox_to_xyxy_pixels(bbox, img_w, img_h)
        if box_xyxy is None:
            continue
        boxes.append(box_xyxy)
        score = det.score if det.score is not None else 0.0
        scored.append((float(score), idx))
    if not scored:
        return list(dets)
    scored.sort(key=lambda x: x[0], reverse=True)
    kept: List[QwenDetection] = []
    kept_boxes: List[Tuple[float, float, float, float]] = []
    for _, det_idx in scored:
        try:
            det = dets[det_idx]
        except Exception:
            continue
        bbox = det.bbox or []
        box_xyxy = _bbox_to_xyxy_pixels(bbox, img_w, img_h)
        if box_xyxy is None:
            continue
        if any(_iou_xyxy(box_xyxy, kb) > iou_thresh for kb in kept_boxes):
            continue
        kept.append(det)
        kept_boxes.append(box_xyxy)
    return kept


def _clip_encode_pil_batch(crops: Sequence[Image.Image]) -> Optional[np.ndarray]:
    """Encode a batch of PIL crops with raw CLIP, returning (N, D) float32 normalized embeddings."""
    model, preprocess = _ensure_clip_backbone_for_mining()
    if model is None or preprocess is None or not crops:
        return None
    try:
        with clip_lock:
            inp = torch.stack([preprocess(c) for c in crops], dim=0).to(device)
            try:
                target_dtype = next(model.parameters()).dtype
            except Exception:
                target_dtype = torch.float32
            if inp.dtype != target_dtype:
                inp = inp.to(dtype=target_dtype)
            with torch.no_grad():
                feats = model.encode_image(inp)
            feats = feats.to(dtype=torch.float32, device="cpu")
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return feats.cpu().numpy()
    except Exception as exc:  # noqa: BLE001
        logger.debug("CLIP batch encode failed: %s", exc)
        return None


def _resolve_agent_clip_classifier_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    allowed_root = (UPLOAD_ROOT / "classifiers").resolve()
    raw = Path(str(path_str))
    candidate = Path(os.path.abspath(str(path_str))).resolve()
    if not _path_is_within_root(candidate, allowed_root):
        # Also accept paths relative to the classifiers root (what /clip/classifiers returns as rel_path).
        try:
            candidate_alt = (allowed_root / raw).resolve()
        except Exception:
            candidate_alt = None
        if candidate_alt is None or not _path_is_within_root(candidate_alt, allowed_root):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_clip_classifier_path_not_allowed")
        candidate = candidate_alt
    if candidate.suffix.lower() not in CLASSIFIER_ALLOWED_EXTS:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_clip_classifier_ext_not_allowed")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="agent_clip_classifier_not_found")
    return candidate


def _load_clip_head_from_classifier(classifier_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a scikit-learn LogisticRegression head and return a portable dict containing:
    - classes: list[str]
    - coef: np.ndarray float32 (K,D) or (1,D) for binary
    - intercept: np.ndarray float32 (K,) or (1,)
    - clip_model: str|None
    - proba_mode: 'binary' | 'softmax' | 'ovr'
    """
    try:
        clf_obj = joblib.load(str(classifier_path))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"agent_clip_classifier_load_failed:{exc}") from exc
    classes_raw = getattr(clf_obj, "classes_", None)
    coef_raw = getattr(clf_obj, "coef_", None)
    intercept_raw = getattr(clf_obj, "intercept_", None)
    if classes_raw is None or coef_raw is None or intercept_raw is None:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_clip_classifier_invalid")
    try:
        classes = [str(c) for c in list(classes_raw)]
        coef = np.asarray(coef_raw, dtype=np.float32)
        intercept = np.asarray(intercept_raw, dtype=np.float32).reshape(-1)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"agent_clip_classifier_invalid:{exc}") from exc
    if coef.ndim != 2 or intercept.ndim != 1:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_clip_classifier_invalid_shape")
    if coef.shape[0] != intercept.shape[0]:
        # sklearn binary case often stores coef as (1,D) but classes has length 2; intercept is (1,).
        if not (coef.shape[0] == 1 and intercept.shape[0] == 1 and len(classes) == 2):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_clip_classifier_invalid_shape")

    clip_model_used = None
    solver_used = None
    meta_path = os.path.splitext(str(classifier_path))[0] + ".meta.pkl"
    if os.path.exists(meta_path):
        try:
            meta_obj = joblib.load(meta_path)
            if isinstance(meta_obj, dict):
                clip_model_used = meta_obj.get("clip_model")
                solver_used = meta_obj.get("solver")
        except Exception:
            clip_model_used = None
            solver_used = None

    n_classes = len(classes)
    proba_mode: str
    if n_classes == 2 and coef.shape[0] == 1:
        proba_mode = "binary"
    elif solver_used and str(solver_used).strip().lower() == "liblinear":
        proba_mode = "ovr"
    else:
        proba_mode = "softmax"

    return {
        "classes": classes,
        "coef": coef,
        "intercept": intercept,
        "clip_model": str(clip_model_used) if clip_model_used else None,
        "proba_mode": proba_mode,
    }


def _clip_head_predict_proba(feats: np.ndarray, head: Dict[str, Any]) -> Optional[np.ndarray]:
    """Compute predict_proba(feats) for an exported LogisticRegression head."""
    if feats is None:
        return None
    coef = head.get("coef")
    intercept = head.get("intercept")
    proba_mode = head.get("proba_mode")
    classes = head.get("classes") or []
    if not isinstance(classes, list) or not classes:
        return None
    try:
        X = np.asarray(feats, dtype=np.float32)
        W = np.asarray(coef, dtype=np.float32)
        b = np.asarray(intercept, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if X.ndim != 2 or W.ndim != 2 or b.ndim != 1:
        return None
    if W.shape[1] != X.shape[1]:
        return None
    if proba_mode == "binary":
        if len(classes) != 2 or W.shape[0] != 1 or b.shape[0] != 1:
            return None
        logits = (X @ W[0].reshape(-1, 1)).reshape(-1) + float(b[0])
        probs_1 = 1.0 / (1.0 + np.exp(-logits))
        probs_0 = 1.0 - probs_1
        return np.stack([probs_0, probs_1], axis=1).astype(np.float32)

    logits = X @ W.T
    if b.shape[0] == logits.shape[1]:
        logits = logits + b.reshape(1, -1)
    else:
        return None

    if proba_mode == "ovr":
        probs = 1.0 / (1.0 + np.exp(-logits))
        denom = probs.sum(axis=1, keepdims=True) + 1e-8
        return (probs / denom).astype(np.float32)

    # softmax
    max_logit = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logit)
    denom = exp_logits.sum(axis=1, keepdims=True) + 1e-8
    return (exp_logits / denom).astype(np.float32)


def _clip_head_keep_mask(
    proba: np.ndarray,
    *,
    target_index: int,
    min_prob: float,
    margin: float,
) -> Optional[np.ndarray]:
    """Return boolean keep mask for rows in proba."""
    try:
        probs = np.asarray(proba, dtype=np.float32)
    except Exception:
        return None
    if probs.ndim != 2 or probs.shape[0] == 0:
        return None
    if target_index < 0 or target_index >= probs.shape[1]:
        return None
    p_target = probs[:, target_index]
    if probs.shape[1] > 1:
        masked = probs.copy()
        masked[:, target_index] = -1.0
        p_other = np.max(masked, axis=1)
    else:
        p_other = np.zeros_like(p_target)
    keep = (p_target >= float(min_prob)) & (p_target >= (p_other + float(margin)))
    return keep


def _save_clip_head_artifacts(
    *,
    recipe_dir: Path,
    head: Dict[str, Any],
    min_prob: float,
    margin: float,
) -> None:
    """Persist a portable CLIP head artifact into a recipe package directory."""
    clip_dir = (recipe_dir / "clip_head").resolve()
    if not _path_is_within_root(clip_dir, recipe_dir.resolve()):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_clip_head_path_invalid")
    clip_dir.mkdir(parents=True, exist_ok=True)

    npz_path = clip_dir / "head.npz"
    meta_path = clip_dir / "meta.json"
    try:
        coef = np.asarray(head.get("coef"), dtype=np.float32)
        intercept = np.asarray(head.get("intercept"), dtype=np.float32).reshape(-1)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"agent_recipe_clip_head_invalid:{exc}") from exc
    if coef.ndim != 2 or intercept.ndim != 1:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_clip_head_invalid")

    try:
        np.savez_compressed(str(npz_path), coef=coef, intercept=intercept)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"agent_recipe_clip_head_write_failed:{exc}") from exc

    classes = head.get("classes") if isinstance(head.get("classes"), list) else []
    meta = {
        "clip_model": head.get("clip_model"),
        "proba_mode": head.get("proba_mode"),
        "classes": [str(c) for c in classes],
        "min_prob": float(min_prob),
        "margin": float(margin),
    }
    try:
        with meta_path.open("w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"agent_recipe_clip_head_meta_write_failed:{exc}") from exc

    try:
        total = npz_path.stat().st_size + meta_path.stat().st_size
    except Exception:
        total = 0
    if total and total > AGENT_RECIPE_MAX_CLIP_HEAD_BYTES:
        raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="agent_recipe_clip_head_too_large")


def _load_clip_head_artifacts(
    *,
    recipe_dir: Path,
    fallback_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Load a portable CLIP head artifact from a recipe package directory."""
    clip_dir = (recipe_dir / "clip_head").resolve()
    npz_path = (clip_dir / "head.npz").resolve()
    meta_path = (clip_dir / "meta.json").resolve()
    if not _path_is_within_root(npz_path, clip_dir) or not _path_is_within_root(meta_path, clip_dir):
        return None
    if not npz_path.exists() or not npz_path.is_file():
        return None
    try:
        with np.load(str(npz_path)) as data:
            coef = np.asarray(data["coef"], dtype=np.float32)
            intercept = np.asarray(data["intercept"], dtype=np.float32).reshape(-1)
    except Exception:
        return None
    meta: Dict[str, Any] = {}
    if meta_path.exists() and meta_path.is_file():
        try:
            with meta_path.open("r", encoding="utf-8") as fp:
                loaded = json.load(fp)
            if isinstance(loaded, dict):
                meta = loaded
        except Exception:
            meta = {}
    if not meta and isinstance(fallback_meta, dict):
        meta = fallback_meta
    classes_raw = meta.get("classes") if isinstance(meta.get("classes"), list) else []
    classes = [str(c) for c in classes_raw]
    proba_mode = meta.get("proba_mode")
    min_prob = meta.get("min_prob")
    margin = meta.get("margin")
    if not isinstance(proba_mode, str) or not proba_mode:
        if coef.shape[0] == 1 and len(classes) == 2:
            proba_mode = "binary"
        else:
            proba_mode = "softmax"
    min_prob_val: Optional[float] = None
    margin_val: Optional[float] = None
    try:
        if min_prob is not None:
            min_prob_val = float(min_prob)
    except Exception:
        min_prob_val = None
    try:
        if margin is not None:
            margin_val = float(margin)
    except Exception:
        margin_val = None

    return {
        "classes": classes,
        "coef": coef,
        "intercept": intercept,
        "clip_model": meta.get("clip_model"),
        "proba_mode": proba_mode,
        "min_prob": min_prob_val,
        "margin": margin_val,
    }


def _normalize_class_name_for_match(name: Optional[str]) -> str:
    if not name:
        return ""
    try:
        s = str(name).strip().lower()
    except Exception:
        return ""
    # Treat underscores/hyphens/spaces as equivalent and ignore punctuation.
    return re.sub(r"[^a-z0-9]+", "", s)


def _find_clip_head_target_index(classes: Sequence[str], class_name: Optional[str]) -> Optional[int]:
    target = _normalize_class_name_for_match(class_name)
    if not target:
        return None
    for idx, c in enumerate(classes):
        if _normalize_class_name_for_match(c) == target:
            return int(idx)
    return None


def _select_diverse_indices(
    feats: np.ndarray,
    *,
    k: int,
    scores: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Pick up to k diverse indices from feats using a greedy k-center heuristic.
    Distance is cosine distance (1 - cosine_sim). If scores provided, start from max score.
    """
    if feats is None:
        return []
    try:
        feats = np.asarray(feats, dtype=np.float32)
    except Exception:
        return []
    if feats.ndim != 2 or feats.shape[0] == 0:
        return []
    k = int(max(1, k))
    n = feats.shape[0]
    if k >= n:
        return list(range(n))
    # Normalize (should already be normalized).
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    vecs = feats / norms
    if scores is not None:
        try:
            start_idx = int(np.argmax(scores))
        except Exception:
            start_idx = 0
    else:
        start_idx = 0
    selected = [start_idx]
    dists = 1.0 - (vecs @ vecs[start_idx].reshape(-1, 1)).reshape(-1)
    while len(selected) < k:
        next_idx = int(np.argmax(dists))
        if next_idx in selected:
            break
        selected.append(next_idx)
        d_new = 1.0 - (vecs @ vecs[next_idx].reshape(-1, 1)).reshape(-1)
        dists = np.minimum(dists, d_new)
    return selected


def _apply_agent_recipe_to_image(
    recipe: Dict[str, Any],
    *,
    image: Dict[str, Any],
    dataset_id: str,
    images: Dict[int, Dict[str, Any]],
    mask_threshold: float,
    min_size: int,
    simplify_epsilon: float,
    max_results: int,
    class_id: Optional[int],
    class_name: Optional[str],
    warnings: Optional[List[str]] = None,
) -> List[QwenDetection]:
    """
    Apply a portable Agent Mining recipe to a single image.

    Current (greedy) semantics:
    1) Run seed text prompts at low threshold to generate many candidate boxes.
    2) Use CLIP similarity against the recipe's positive/negative crop banks to keep good seeds.
    3) Run SAM3 visual prompting from the kept seed boxes to expand detections.
    4) CLIP-filter and IoU-dedupe final detections to avoid double-labeling.
    """
    recipe_body = recipe.get("recipe") if "steps" not in recipe and isinstance(recipe.get("recipe"), dict) else recipe
    img_path = image.get("path")
    if not img_path:
        return []
    try:
        with Image.open(img_path) as pil_img_ctx:
            pil_img = pil_img_ctx.convert("RGB")
    except Exception:
        return []
    recipe_id = recipe.get("id")
    params_combined: Dict[str, Any] = {}
    for src in (recipe.get("params"), recipe_body.get("params")):
        if isinstance(src, dict):
            params_combined.update(src)
    # New-format fields (fallback to legacy step-based recipes for portability/back-compat).
    text_prompts_raw = recipe_body.get("text_prompts")
    positives_raw = recipe_body.get("positives")
    negatives_raw = recipe_body.get("negatives")
    steps = _normalize_agent_recipe_steps(recipe_body.get("steps") or [])
    if not isinstance(text_prompts_raw, list) or not text_prompts_raw:
        text_prompts_raw = [s.get("prompt") for s in steps if (s.get("type") or "text") == "text" and s.get("prompt")]
    if not isinstance(positives_raw, list) or not positives_raw:
        positives_raw = [s.get("exemplar") for s in steps if s.get("exemplar")]
    if not isinstance(negatives_raw, list):
        negatives_raw = []

    text_prompts = _sanitize_prompts([str(p) for p in (text_prompts_raw or []) if str(p).strip()])
    if not text_prompts and class_name:
        text_prompts = _sanitize_prompts([str(class_name)])

    # Optional embedded CLIP head (trained logistic regression) for filtering.
    recipe_target_class_name = str(recipe.get("class_name") or class_name or "").strip() if isinstance(recipe, dict) else str(class_name or "").strip()
    clip_head_cfg: Optional[Dict[str, Any]] = None
    if isinstance(recipe_body, dict) and isinstance(recipe_body.get("clip_head"), dict):
        clip_head_cfg = recipe_body.get("clip_head")
    elif isinstance(recipe, dict) and isinstance(recipe.get("clip_head"), dict):
        clip_head_cfg = recipe.get("clip_head")
    clip_head_min_prob = 0.5
    clip_head_margin = 0.0
    cfg_sets_min_prob = False
    cfg_sets_margin = False
    if clip_head_cfg:
        try:
            if clip_head_cfg.get("min_prob") is not None:
                clip_head_min_prob = float(clip_head_cfg.get("min_prob"))
                cfg_sets_min_prob = True
            if clip_head_cfg.get("margin") is not None:
                clip_head_margin = float(clip_head_cfg.get("margin"))
                cfg_sets_margin = True
        except Exception:
            clip_head_min_prob = 0.5
            clip_head_margin = 0.0
    clip_head: Optional[Dict[str, Any]] = None
    clip_head_target_index: Optional[int] = None
    if recipe_id:
        recipe_root = (AGENT_MINING_RECIPES_ROOT / str(recipe_id)).resolve()
        if _path_is_within_root(recipe_root, AGENT_MINING_RECIPES_ROOT.resolve()):
            clip_head = _load_clip_head_artifacts(recipe_dir=recipe_root, fallback_meta=clip_head_cfg)
    if clip_head:
        # If the recipe JSON doesn't specify these, fall back to embedded meta.json defaults.
        if not cfg_sets_min_prob and clip_head.get("min_prob") is not None:
            try:
                clip_head_min_prob = float(clip_head.get("min_prob"))
            except Exception:
                pass
        if not cfg_sets_margin and clip_head.get("margin") is not None:
            try:
                clip_head_margin = float(clip_head.get("margin"))
            except Exception:
                pass
        classes_list = clip_head.get("classes") if isinstance(clip_head.get("classes"), list) else []
        clip_head_target_index = _find_clip_head_target_index(classes_list, recipe_target_class_name)

    def _recipe_param(key: str) -> Any:
        if key in params_combined:
            return params_combined.get(key)
        if isinstance(recipe_body, dict):
            return recipe_body.get(key)
        return None

    def _bool_param(key: str, default: bool) -> bool:
        raw = _recipe_param(key)
        if raw is None:
            return default
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        if isinstance(raw, str):
            val = raw.strip().lower()
            if val in {"1", "true", "yes", "y", "on"}:
                return True
            if val in {"0", "false", "no", "n", "off"}:
                return False
        return default

    def _float_param(key: str, default: float) -> float:
        raw = _recipe_param(key)
        if raw is None:
            return default
        try:
            return float(raw)
        except Exception:
            return default

    def _int_param(key: str, default: int) -> int:
        raw = _recipe_param(key)
        if raw is None:
            return default
        try:
            return int(raw)
        except Exception:
            return default

    # Config knobs (recipe params override request params; request params are passed via function args).
    use_clip_guard = _bool_param("use_clip_fp_guard", True)
    use_negatives = _bool_param("use_negative_exemplars", True)
    neg_strength = _float_param("negative_strength", 0.5)
    similarity_floor = _float_param("similarity_score", 0.25)
    seed_threshold = _float_param("seed_threshold", 0.05)
    expand_threshold = _float_param("expand_threshold", 0.3)
    max_visual_seeds = _int_param("max_visual_seeds", 25)
    seed_dedupe_iou = _float_param("seed_dedupe_iou", 0.9)
    out_dedupe_iou = _float_param("dedupe_iou", 0.5)
    seed_threshold = max(0.0, min(1.0, seed_threshold))
    expand_threshold = max(0.0, min(1.0, expand_threshold))
    similarity_floor = max(0.0, min(1.0, similarity_floor))
    max_visual_seeds = max(0, min(500, max_visual_seeds))

    def _add_warning(code: str) -> None:
        if warnings is None:
            return
        if code not in warnings:
            warnings.append(code)

    def _crop_embed_key(prefix: str, crop_ref: Optional[str]) -> Optional[str]:
        if not crop_ref:
            return None
        try:
            crop_name = Path(str(crop_ref)).name
        except Exception:
            return None
        if not crop_name:
            return None
        return f"{prefix}:{crop_name}"

    # Build embeddings for exemplars/negatives if present.
    def _resolve_recipe_crop_path(recipe_id_val: Optional[str], crop_path_raw: Optional[str]) -> Optional[str]:
        """Resolve a crop path inside the recipe package; ignore absolute/external paths."""
        if not recipe_id_val or not crop_path_raw:
            return None
        try:
            crop_name = Path(crop_path_raw).name
        except Exception:
            return None
        recipe_root = (AGENT_MINING_RECIPES_ROOT / recipe_id_val).resolve()
        if not _path_is_within_root(recipe_root, AGENT_MINING_RECIPES_ROOT.resolve()):
            return None
        crop_root = (recipe_root / "crops").resolve()
        if not _path_is_within_root(crop_root, recipe_root):
            return None
        candidate = (crop_root / crop_name).resolve()
        try:
            if candidate.exists() and candidate.is_file() and _path_is_within_root(candidate, crop_root):
                return str(candidate)
        except Exception:
            return None
        return None

    exemplar_entries: List[Dict[str, Any]] = []
    for ex_raw in positives_raw or []:
        if not isinstance(ex_raw, dict):
            continue
        ex = dict(ex_raw)
        resolved = _resolve_recipe_crop_path(recipe_id, ex.get("crop_path") or ex.get("path"))
        if resolved:
            ex["path"] = resolved
        if ex.get("image_id") is None:
            ex["image_id"] = 0
        ex["bbox"] = [0.5, 0.5, 1.0, 1.0]
        ex_key = _crop_embed_key("crop", ex.get("crop_path") or ex.get("path"))
        if ex_key:
            ex["embed_id"] = ex_key
        if not ex.get("path"):
            _add_warning("missing_crop_exemplar")
        exemplar_entries.append(ex)
    exemplar_embeddings: Dict[str, np.ndarray] = {}
    exemplar_warnings: List[str] = []
    if exemplar_entries:
        exemplar_embeddings, exemplar_warnings = _clip_embed_regions(exemplar_entries, images, max_regions=len(exemplar_entries))
        for w in exemplar_warnings:
            _add_warning(w)

    negative_entries: List[Dict[str, Any]] = []
    if use_negatives and negatives_raw:
        for neg_raw in negatives_raw:
            if not isinstance(neg_raw, dict):
                continue
            entry = dict(neg_raw)
            resolved = _resolve_recipe_crop_path(recipe_id, entry.get("crop_path") or entry.get("path"))
            if resolved:
                entry["path"] = resolved
            if entry.get("image_id") is None:
                entry["image_id"] = 0
            entry["bbox"] = [0.5, 0.5, 1.0, 1.0]
            neg_key = _crop_embed_key("neg", entry.get("crop_path") or entry.get("path"))
            if neg_key:
                entry["embed_id"] = neg_key
            if not entry.get("path"):
                _add_warning("missing_crop_negative")
            negative_entries.append(entry)
    negative_embeddings: Dict[str, np.ndarray] = {}
    if use_clip_guard and use_negatives and negative_entries:
        negative_embeddings, _ = _clip_embed_regions(negative_entries, images, max_regions=len(negative_entries))

    ex_mat = None
    if exemplar_embeddings:
        try:
            ex_mat = np.stack(list(exemplar_embeddings.values())).astype(np.float32)
            ex_mat = ex_mat / (np.linalg.norm(ex_mat, axis=1, keepdims=True) + 1e-8)
        except Exception:
            ex_mat = None
    neg_mat = None
    if use_negatives and negative_embeddings:
        try:
            neg_mat = np.stack(list(negative_embeddings.values())).astype(np.float32)
            neg_mat = neg_mat / (np.linalg.norm(neg_mat, axis=1, keepdims=True) + 1e-8)
        except Exception:
            neg_mat = None

    clip_head_active = bool(isinstance(clip_head, dict) and clip_head_target_index is not None)
    if use_clip_guard and ex_mat is None:
        _add_warning("clip_missing_exemplars")
        use_clip_guard = False

    _, processor, _ = _ensure_sam3_text_runtime()
    try:
        state = processor.set_image(pil_img)
    except Exception:
        state = None

    def _reset_prompts() -> None:
        try:
            if hasattr(processor, "reset_all_prompts"):
                processor.reset_all_prompts(state)
                return
        except Exception:
            pass
        try:
            if isinstance(state, dict) and isinstance(state.get("backbone_out"), dict):
                for k in ("language_features", "language_mask", "language_embeds"):
                    state["backbone_out"].pop(k, None)
            if isinstance(state, dict):
                for k in ("geometric_prompt", "boxes", "masks", "masks_logits", "scores"):
                    state.pop(k, None)
        except Exception:
            pass

    def _clip_score(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # feats: (N, D), already normalized
        pos = np.zeros((feats.shape[0],), dtype=np.float32)
        neg = np.zeros((feats.shape[0],), dtype=np.float32)
        if ex_mat is not None:
            try:
                sims = feats @ ex_mat.T
                pos = np.max(sims, axis=1)
            except Exception:
                pos = np.zeros((feats.shape[0],), dtype=np.float32)
        if neg_mat is not None:
            try:
                sims_n = feats @ neg_mat.T
                neg = np.max(sims_n, axis=1)
            except Exception:
                neg = np.zeros((feats.shape[0],), dtype=np.float32)
        score = pos - max(0.0, neg_strength) * neg
        return pos, neg, score

    # 1) Seed with text prompts at low threshold.
    seed_dets: List[QwenDetection] = []
    for p in text_prompts:
        if not p:
            continue
        _reset_prompts()
        try:
            dets = _run_sam3_text_inference(
                pil_img,
                p,
                seed_threshold,
                mask_threshold,
                max_results,
                min_size=min_size if min_size > 0 else None,
                simplify_epsilon=simplify_epsilon,
                processor_override=processor,
                state=state,
            )
        except Exception:
            continue
        seed_dets.extend(dets or [])

    if not seed_dets:
        _add_warning("no_results")
        return []

    # Dedupe seed candidates aggressively before CLIP scoring.
    seed_dets = _dedupe_qwen_detections_iou(seed_dets, img_w=pil_img.width, img_h=pil_img.height, iou_thresh=seed_dedupe_iou)

    # 2) CLIP-filter seed candidates and select a diverse subset to expand.
    seed_boxes_xywh: List[Tuple[float, float, float, float]] = []
    seed_crops: List[Image.Image] = []
    seed_keep_refs: List[QwenDetection] = []
    for det in seed_dets:
        bbox_xyxy = _bbox_to_xyxy_pixels(det.bbox or [], pil_img.width, pil_img.height)
        if bbox_xyxy is None:
            continue
        x1, y1, x2, y2 = bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            continue
        try:
            crop = pil_img.crop((x1, y1, x2, y2))
        except Exception:
            continue
        seed_crops.append(crop)
        seed_boxes_xywh.append((float(x1), float(y1), float(x2 - x1), float(y2 - y1)))
        seed_keep_refs.append(det)
    need_seed_feats = bool(seed_crops) and (use_clip_guard or clip_head_active)
    feats = _clip_encode_pil_batch(seed_crops) if need_seed_feats else None
    seed_indices: List[int] = []
    if feats is not None and feats.size:
        keep_mask = np.ones((feats.shape[0],), dtype=bool)
        scores_for_diverse: Optional[np.ndarray] = None
        if use_clip_guard and ex_mat is not None:
            pos_s, _, score_s = _clip_score(feats)
            keep_mask &= (pos_s >= float(similarity_floor)) & (score_s >= 0.0)
            scores_for_diverse = score_s
        if clip_head_active:
            proba = _clip_head_predict_proba(feats, clip_head) if isinstance(clip_head, dict) else None
            if proba is not None and clip_head_target_index is not None:
                try:
                    scores_for_diverse = proba[:, int(clip_head_target_index)]
                except Exception:
                    pass
        kept_idx = [i for i, ok in enumerate(keep_mask.tolist()) if ok]
        if kept_idx:
            kept_feats = feats[kept_idx]
            kept_scores = scores_for_diverse[kept_idx] if scores_for_diverse is not None else None
            if max_visual_seeds > 0:
                diverse_local = _select_diverse_indices(kept_feats, k=min(max_visual_seeds, len(kept_idx)), scores=kept_scores)
                seed_indices = [kept_idx[i] for i in diverse_local]
            else:
                seed_indices = kept_idx
        else:
            seed_indices = []
            _add_warning("visual_seed_not_found")
    else:
        # No CLIP features available: expand from a small top-score subset to bound runtime.
        ranked = sorted(
            range(len(seed_keep_refs)),
            key=lambda i: (seed_keep_refs[i].score if seed_keep_refs[i].score is not None else 0.0),
            reverse=True,
        )
        seed_indices = ranked[: max(0, min(max_visual_seeds, len(ranked)))]

    # Always include the kept seed detections as potential outputs (they will be filtered/deduped later).
    kept_seed_dets: List[QwenDetection] = []
    if seed_indices:
        kept_seed_dets = [seed_keep_refs[i] for i in seed_indices if 0 <= i < len(seed_keep_refs)]

    # 3) Visual expansion from each chosen seed.
    expanded: List[QwenDetection] = []
    for i in seed_indices:
        if i < 0 or i >= len(seed_boxes_xywh):
            continue
        _reset_prompts()
        try:
            dets = _run_sam3_visual_inference(
                pil_img,
                seed_boxes_xywh[i],
                expand_threshold,
                mask_threshold,
                max_results,
                min_size=min_size if min_size > 0 else None,
                simplify_epsilon=simplify_epsilon,
                processor_override=processor,
                state=state,
            )
        except Exception:
            continue
        expanded.extend(dets or [])

    combined = [*kept_seed_dets, *expanded]
    if not combined:
        _add_warning("no_results")
        return []

    # 4) CLIP FP-guard over final detections (pos/neg crop banks).
    filtered_final: List[QwenDetection]
    if (use_clip_guard and ex_mat is not None) or clip_head_active:
        combined = _dedupe_qwen_detections_iou(combined, img_w=pil_img.width, img_h=pil_img.height, iou_thresh=out_dedupe_iou)
        det_crops: List[Image.Image] = []
        det_refs: List[QwenDetection] = []
        for det in combined:
            bbox_xyxy = _bbox_to_xyxy_pixels(det.bbox or [], pil_img.width, pil_img.height)
            if bbox_xyxy is None:
                continue
            x1, y1, x2, y2 = bbox_xyxy
            if x2 <= x1 or y2 <= y1:
                continue
            try:
                crop = pil_img.crop((x1, y1, x2, y2))
            except Exception:
                continue
            det_crops.append(crop)
            det_refs.append(det)
        det_feats = _clip_encode_pil_batch(det_crops) if det_crops else None
        if det_feats is None or det_feats.size == 0:
            filtered_final = combined
            _add_warning("clip_unavailable")
        else:
            keep_mask = np.ones((det_feats.shape[0],), dtype=bool)
            if use_clip_guard and ex_mat is not None:
                pos_s, _, score_s = _clip_score(det_feats)
                keep_mask &= (pos_s >= float(similarity_floor)) & (score_s >= 0.0)
            if clip_head_active and clip_head_target_index is not None:
                proba = _clip_head_predict_proba(det_feats, clip_head) if isinstance(clip_head, dict) else None
                if proba is not None:
                    head_keep = _clip_head_keep_mask(
                        proba,
                        target_index=int(clip_head_target_index),
                        min_prob=float(clip_head_min_prob),
                        margin=float(clip_head_margin),
                    )
                    if head_keep is not None:
                        keep_mask &= head_keep
            filtered_final = [d for d, ok in zip(det_refs, keep_mask.tolist()) if ok]
            if len(filtered_final) < len(det_refs):
                _add_warning("clip_fp_filtered")
            if det_refs and not filtered_final:
                _add_warning("clip_filtered_all")
    else:
        filtered_final = combined

    # 5) Final dedupe + class assignment.
    final = _dedupe_qwen_detections_iou(filtered_final, img_w=pil_img.width, img_h=pil_img.height, iou_thresh=out_dedupe_iou)
    for det in final:
        if det.class_id is None:
            det.class_id = class_id
        if det.class_name is None:
            det.class_name = class_name
        if det.qwen_label is None and class_name:
            det.qwen_label = class_name
    return _prune_detections_for_response(final, warnings=warnings)


def _reset_sam3_prompts_for_state(processor: Any, state: Any) -> None:
    """Best-effort prompt reset for a preloaded SAM3 image state."""
    try:
        if hasattr(processor, "reset_all_prompts"):
            processor.reset_all_prompts(state)
            return
    except Exception:
        pass
    try:
        if isinstance(state, dict) and isinstance(state.get("backbone_out"), dict):
            for k in ("language_features", "language_mask", "language_embeds"):
                state["backbone_out"].pop(k, None)
        if isinstance(state, dict):
            for k in ("geometric_prompt", "boxes", "masks", "masks_logits", "scores"):
                state.pop(k, None)
    except Exception:
        pass


def _infer_sam3_greedy_recipe_on_image(
    *,
    pil_img: Image.Image,
    processor: Any,
    text_prompts: Sequence[str],
    exemplar_embeddings: Optional[np.ndarray],
    negative_embeddings: Optional[np.ndarray],
    seed_threshold: float,
    expand_threshold: float,
    max_visual_seeds: int,
    seed_dedupe_iou: float,
    out_dedupe_iou: float,
    mask_threshold: float,
    min_size: int,
    simplify_epsilon: float,
    max_results: int,
    negative_strength: float,
    similarity_floor: float,
    clip_head: Optional[Dict[str, Any]] = None,
    clip_head_target_index: Optional[int] = None,
    clip_head_min_prob: float = 0.5,
    clip_head_margin: float = 0.0,
) -> List[QwenDetection]:
    """
    Greedy SAM3 recipe inference on a single PIL image using *precomputed* CLIP embedding banks.

    This matches the intended "portable recipe" semantics:
    1) low-threshold text prompt(s) to generate candidate boxes
    2) CLIP filter vs positive/negative crop banks to pick diverse seed boxes
    3) SAM3 visual prompting from those seeds to expand detections
    4) CLIP filter + IoU dedupe to suppress dupes/FPs
    """
    prompts = [p for p in (_sanitize_prompts([str(p) for p in text_prompts if str(p).strip()]) or []) if p]
    if not prompts:
        return []
    try:
        state = processor.set_image(pil_img)
    except Exception:
        return []

    ex_mat = exemplar_embeddings
    neg_mat = negative_embeddings
    use_clip = (ex_mat is not None and isinstance(ex_mat, np.ndarray) and ex_mat.size > 0) or (
        isinstance(clip_head, dict) and clip_head_target_index is not None
    )

    def _clip_score(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = np.zeros((feats.shape[0],), dtype=np.float32)
        neg = np.zeros((feats.shape[0],), dtype=np.float32)
        if ex_mat is not None:
            try:
                pos = np.max(feats @ ex_mat.T, axis=1)
            except Exception:
                pos = np.zeros((feats.shape[0],), dtype=np.float32)
        if neg_mat is not None and neg_mat.size:
            try:
                neg = np.max(feats @ neg_mat.T, axis=1)
            except Exception:
                neg = np.zeros((feats.shape[0],), dtype=np.float32)
        score = pos - max(0.0, float(negative_strength)) * neg
        return pos, neg, score

    def _clip_head_filter(feats: np.ndarray) -> Optional[np.ndarray]:
        if not isinstance(clip_head, dict) or clip_head_target_index is None:
            return None
        proba = _clip_head_predict_proba(feats, clip_head)
        if proba is None:
            return None
        return _clip_head_keep_mask(
            proba,
            target_index=int(clip_head_target_index),
            min_prob=float(clip_head_min_prob),
            margin=float(clip_head_margin),
        )

    seed_dets: List[QwenDetection] = []
    for prompt in prompts:
        _reset_sam3_prompts_for_state(processor, state)
        try:
            dets = _run_sam3_text_inference(
                pil_img,
                prompt,
                float(seed_threshold),
                float(mask_threshold),
                int(max_results) if max_results else None,
                min_size=min_size if min_size > 0 else None,
                simplify_epsilon=simplify_epsilon,
                processor_override=processor,
                state=state,
            )
        except Exception:
            continue
        seed_dets.extend(dets or [])
    if not seed_dets:
        return []

    seed_dets = _dedupe_qwen_detections_iou(seed_dets, img_w=pil_img.width, img_h=pil_img.height, iou_thresh=seed_dedupe_iou)
    if not seed_dets:
        return []

    seed_boxes_xywh: List[Tuple[float, float, float, float]] = []
    seed_refs: List[QwenDetection] = []
    seed_crops: List[Image.Image] = []
    for det in seed_dets:
        bbox_xyxy = _bbox_to_xyxy_pixels(det.bbox or [], pil_img.width, pil_img.height)
        if bbox_xyxy is None:
            continue
        x1, y1, x2, y2 = bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            continue
        try:
            seed_crops.append(pil_img.crop((x1, y1, x2, y2)))
        except Exception:
            continue
        seed_boxes_xywh.append((float(x1), float(y1), float(x2 - x1), float(y2 - y1)))
        seed_refs.append(det)
    if not seed_refs:
        return []

    seed_indices: List[int] = []
    had_seed_feats = False
    if use_clip and seed_crops:
        feats = _clip_encode_pil_batch(seed_crops)
        if feats is not None and feats.size:
            had_seed_feats = True
            keep_mask = np.ones((feats.shape[0],), dtype=bool)
            scores_for_diverse: Optional[np.ndarray] = None
            if ex_mat is not None:
                pos_s, _, score_s = _clip_score(feats)
                keep_mask &= (pos_s >= float(similarity_floor)) & (score_s >= 0.0)
                scores_for_diverse = score_s
            if isinstance(clip_head, dict) and clip_head_target_index is not None:
                try:
                    proba = _clip_head_predict_proba(feats, clip_head)
                    if proba is not None:
                        scores_for_diverse = proba[:, int(clip_head_target_index)]
                except Exception:
                    pass
            kept = [i for i, ok in enumerate(keep_mask.tolist()) if ok]
            if kept:
                kept_feats = feats[kept]
                kept_scores = scores_for_diverse[kept] if scores_for_diverse is not None else None
                if max_visual_seeds > 0:
                    diverse_local = _select_diverse_indices(kept_feats, k=min(max_visual_seeds, len(kept)), scores=kept_scores)
                    seed_indices = [kept[i] for i in diverse_local]
                else:
                    seed_indices = kept
    if not seed_indices and not had_seed_feats:
        ranked = sorted(
            range(len(seed_refs)),
            key=lambda i: (seed_refs[i].score if seed_refs[i].score is not None else 0.0),
            reverse=True,
        )
        seed_indices = ranked[: max(0, min(int(max_visual_seeds) if max_visual_seeds else 0, len(ranked)))]
    kept_seed_dets = [seed_refs[i] for i in seed_indices if 0 <= i < len(seed_refs)]

    expanded: List[QwenDetection] = []
    for i in seed_indices:
        if i < 0 or i >= len(seed_boxes_xywh):
            continue
        _reset_sam3_prompts_for_state(processor, state)
        try:
            dets = _run_sam3_visual_inference(
                pil_img,
                seed_boxes_xywh[i],
                float(expand_threshold),
                float(mask_threshold),
                int(max_results) if max_results else None,
                min_size=min_size if min_size > 0 else None,
                simplify_epsilon=simplify_epsilon,
                processor_override=processor,
                state=state,
            )
        except Exception:
            continue
        expanded.extend(dets or [])

    combined = [*kept_seed_dets, *expanded]
    if not combined:
        return []
    combined = _dedupe_qwen_detections_iou(combined, img_w=pil_img.width, img_h=pil_img.height, iou_thresh=out_dedupe_iou)

    if use_clip:
        det_crops: List[Image.Image] = []
        det_refs: List[QwenDetection] = []
        for det in combined:
            bbox_xyxy = _bbox_to_xyxy_pixels(det.bbox or [], pil_img.width, pil_img.height)
            if bbox_xyxy is None:
                continue
            x1, y1, x2, y2 = bbox_xyxy
            if x2 <= x1 or y2 <= y1:
                continue
            try:
                det_crops.append(pil_img.crop((x1, y1, x2, y2)))
            except Exception:
                continue
            det_refs.append(det)
        feats = _clip_encode_pil_batch(det_crops) if det_crops else None
        if feats is not None and feats.size:
            keep_mask = np.ones((feats.shape[0],), dtype=bool)
            if ex_mat is not None:
                pos_s, _, score_s = _clip_score(feats)
                keep_mask &= (pos_s >= float(similarity_floor)) & (score_s >= 0.0)
            if isinstance(clip_head, dict) and clip_head_target_index is not None:
                try:
                    proba = _clip_head_predict_proba(feats, clip_head)
                except Exception:
                    proba = None
                if proba is not None:
                    t_idx = int(clip_head_target_index)
                    try:
                        p_target = proba[:, t_idx]
                        if proba.shape[1] > 1:
                            masked = proba.copy()
                            masked[:, t_idx] = -1.0
                            p_other = np.max(masked, axis=1)
                        else:
                            p_other = np.zeros_like(p_target)
                        for det_obj, p_t, p_o in zip(det_refs, p_target.tolist(), p_other.tolist()):
                            det_obj.clip_head_prob = float(p_t)
                            det_obj.clip_head_margin = float(p_t - p_o)
                    except Exception:
                        pass
                    head_keep = _clip_head_keep_mask(
                        proba,
                        target_index=t_idx,
                        min_prob=float(clip_head_min_prob),
                        margin=float(clip_head_margin),
                    )
                    if head_keep is not None:
                        keep_mask &= head_keep
            filtered = [d for d, ok in zip(det_refs, keep_mask.tolist()) if ok]
            combined = filtered
    return _dedupe_qwen_detections_iou(combined, img_w=pil_img.width, img_h=pil_img.height, iou_thresh=out_dedupe_iou)


def _evaluate_sam3_greedy_recipe(
    *,
    cat_id: int,
    image_ids: Sequence[int],
    images: Dict[int, Dict[str, Any]],
    gt_by_image_cat: Dict[int, Dict[int, List[List[float]]]],
    processor: Any,
    text_prompts: Sequence[str],
    exemplar_embeddings: Optional[np.ndarray],
    negative_embeddings: Optional[np.ndarray],
    clip_head: Optional[Dict[str, Any]] = None,
    clip_head_target_index: Optional[int] = None,
    clip_head_min_prob: float = 0.5,
    clip_head_margin: float = 0.0,
    payload: AgentMiningRequest,
    cancel_event: Optional[threading.Event] = None,
    log_every: int = 25,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    total_gt = 0
    for img_id in image_ids:
        total_gt += len((gt_by_image_cat.get(int(img_id)) or {}).get(int(cat_id)) or [])

    head_active = bool(isinstance(clip_head, dict) and clip_head_target_index is not None)
    if head_active:
        # Sweep head thresholds without re-running SAM3: run the greedy recipe once with no head thresholding,
        # keep per-detection head probabilities, then apply different thresholds during scoring.
        fixed_margin = float(clip_head_margin)
        sweep_min_probs = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ]
        try:
            base_min = float(clip_head_min_prob)
            if 0.0 <= base_min <= 1.0:
                sweep_min_probs.append(base_min)
        except Exception:
            pass
        sweep_min_probs = sorted({float(max(0.0, min(1.0, p))) for p in sweep_min_probs})

        per_image_rows: Dict[int, List[Tuple[float, float, float, Optional[int]]]] = {}
        for idx, img_id in enumerate(image_ids, start=1):
            if cancel_event is not None and cancel_event.is_set():
                break
            info = images.get(int(img_id)) or {}
            path = info.get("path")
            if not path:
                continue
            try:
                with Image.open(path) as img:
                    pil_img = img.convert("RGB")
            except Exception:
                continue
            dets = _infer_sam3_greedy_recipe_on_image(
                pil_img=pil_img,
                processor=processor,
                text_prompts=text_prompts,
                exemplar_embeddings=exemplar_embeddings,
                negative_embeddings=negative_embeddings,
                seed_threshold=payload.seed_threshold,
                expand_threshold=payload.expand_threshold,
                max_visual_seeds=payload.max_visual_seeds,
                seed_dedupe_iou=payload.seed_dedupe_iou,
                out_dedupe_iou=payload.dedupe_iou,
                mask_threshold=payload.mask_threshold,
                min_size=payload.min_size,
                simplify_epsilon=payload.simplify_epsilon,
                max_results=payload.max_results,
                negative_strength=payload.negative_strength,
                similarity_floor=payload.similarity_score,
                clip_head=clip_head,
                clip_head_target_index=clip_head_target_index,
                clip_head_min_prob=0.0,
                clip_head_margin=0.0,
            )
            gt_boxes = (gt_by_image_cat.get(int(img_id)) or {}).get(int(cat_id)) or []
            gt_xyxy = [_xywh_to_xyxy(b) for b in gt_boxes]
            rows: List[Tuple[float, float, float, Optional[int]]] = []
            for det in dets:
                bbox = det.bbox or []
                if len(bbox) < 4:
                    continue
                try:
                    det_xyxy = yolo_to_corners(bbox, pil_img.width, pil_img.height)
                except Exception:
                    continue
                best_iou = 0.0
                best_idx: Optional[int] = None
                for j, gt in enumerate(gt_xyxy):
                    iou = _iou_xyxy(det_xyxy, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
                prob = float(det.clip_head_prob) if det.clip_head_prob is not None else 0.0
                margin = float(det.clip_head_margin) if det.clip_head_margin is not None else 0.0
                rows.append((prob, margin, float(best_iou), best_idx))
            per_image_rows[int(img_id)] = rows
            if log_fn and log_every > 0 and idx % log_every == 0:
                try:
                    log_fn(f"Processed {idx}/{len(image_ids)} val images for class {cat_id}")
                except Exception:
                    pass

        best_summary: Optional[Dict[str, Any]] = None
        best_key: Optional[Tuple[int, int, float, float]] = None
        for min_prob in sweep_min_probs:
            matched = 0
            fps = 0
            duplicates = 0
            preds = 0
            det_images = 0
            for img_id in image_ids:
                rows = per_image_rows.get(int(img_id)) or []
                used: set[int] = set()
                any_det = False
                for prob, margin, best_iou, best_idx in rows:
                    if prob < float(min_prob):
                        continue
                    if fixed_margin > 0.0 and margin < fixed_margin:
                        continue
                    any_det = True
                    preds += 1
                    if best_idx is not None and best_iou >= float(payload.iou_threshold):
                        if best_idx in used:
                            duplicates += 1
                        else:
                            used.add(best_idx)
                            matched += 1
                    else:
                        fps += 1
                if any_det:
                    det_images += 1
            recall = matched / total_gt if total_gt else 0.0
            precision = matched / max(1, matched + fps)
            det_rate = det_images / len(image_ids) if image_ids else 0.0
            # Prefer higher coverage, then fewer FPs, then higher precision; finally prefer higher threshold.
            key = (int(matched), -int(fps), float(precision), float(min_prob))
            if best_key is None or key > best_key:
                best_key = key
                best_summary = {
                    "gts": total_gt,
                    "matches": matched,
                    "fps": fps,
                    "duplicates": duplicates,
                    "preds": preds,
                    "precision": precision,
                    "recall": recall,
                    "coverage_rate": recall,
                    "det_rate": det_rate,
                    "clip_head_min_prob": float(min_prob),
                    "clip_head_margin": float(fixed_margin),
                }
        if best_summary is None:
            best_summary = {
                "gts": total_gt,
                "matches": 0,
                "fps": 0,
                "duplicates": 0,
                "preds": 0,
                "precision": 0.0,
                "recall": 0.0,
                "coverage_rate": 0.0,
                "det_rate": 0.0,
                "clip_head_min_prob": float(clip_head_min_prob),
                "clip_head_margin": float(clip_head_margin),
            }
        return best_summary

    matched = 0
    fps = 0
    duplicates = 0
    preds = 0
    det_images = 0
    for idx, img_id in enumerate(image_ids, start=1):
        if cancel_event is not None and cancel_event.is_set():
            break
        info = images.get(int(img_id)) or {}
        path = info.get("path")
        if not path:
            continue
        try:
            with Image.open(path) as img:
                pil_img = img.convert("RGB")
        except Exception:
            continue
        dets = _infer_sam3_greedy_recipe_on_image(
            pil_img=pil_img,
            processor=processor,
            text_prompts=text_prompts,
            exemplar_embeddings=exemplar_embeddings,
            negative_embeddings=negative_embeddings,
            seed_threshold=payload.seed_threshold,
            expand_threshold=payload.expand_threshold,
            max_visual_seeds=payload.max_visual_seeds,
            seed_dedupe_iou=payload.seed_dedupe_iou,
            out_dedupe_iou=payload.dedupe_iou,
            mask_threshold=payload.mask_threshold,
            min_size=payload.min_size,
            simplify_epsilon=payload.simplify_epsilon,
            max_results=payload.max_results,
            negative_strength=payload.negative_strength,
            similarity_floor=payload.similarity_score,
            clip_head=clip_head,
            clip_head_target_index=clip_head_target_index,
            clip_head_min_prob=clip_head_min_prob,
            clip_head_margin=clip_head_margin,
        )
        if dets:
            det_images += 1
        gt_boxes = (gt_by_image_cat.get(int(img_id)) or {}).get(int(cat_id)) or []
        gt_xyxy = [_xywh_to_xyxy(b) for b in gt_boxes]
        used: set[int] = set()
        for det in dets:
            preds += 1
            bbox = det.bbox or []
            if len(bbox) < 4:
                continue
            try:
                det_xyxy = yolo_to_corners(bbox, pil_img.width, pil_img.height)
            except Exception:
                continue
            best_iou = 0.0
            best_idx = None
            for j, gt in enumerate(gt_xyxy):
                iou = _iou_xyxy(det_xyxy, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_idx is not None and best_iou >= float(payload.iou_threshold):
                if best_idx in used:
                    duplicates += 1
                else:
                    used.add(best_idx)
                    matched += 1
            else:
                fps += 1
        if log_fn and log_every > 0 and idx % log_every == 0:
            try:
                log_fn(f"Processed {idx}/{len(image_ids)} val images for class {cat_id}")
            except Exception:
                pass
    recall = matched / total_gt if total_gt else 0.0
    precision = matched / max(1, matched + fps)
    det_rate = det_images / len(image_ids) if image_ids else 0.0
    return {
        "gts": total_gt,
        "matches": matched,
        "fps": fps,
        "duplicates": duplicates,
        "preds": preds,
        "precision": precision,
        "recall": recall,
        "coverage_rate": recall,
        "det_rate": det_rate,
    }


def _detections_to_eval_cache(
    detections: Sequence[Dict[str, Any]],
    images: Dict[int, Dict[str, Any]],
) -> Dict[int, List[Tuple[float, float, float, float, Optional[float]]]]:
    """Convert detection dicts into the format expected by _evaluate_prompt_candidate cached_detections."""
    by_image: Dict[int, List[Tuple[float, float, float, float, Optional[float]]]] = {}
    for det in detections:
        try:
            img_id = int(det.get("image_id"))
        except Exception:
            continue
        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        info = images.get(img_id)
        if not info:
            continue
        width = info.get("width")
        height = info.get("height")
        if width is None or height is None:
            path = info.get("path")
            if path:
                try:
                    with Image.open(path) as pil_img:
                        width = pil_img.width
                        height = pil_img.height
                        info["width"] = width
                        info["height"] = height
                except Exception:
                    continue
            else:
                continue
        try:
            x1, y1, x2, y2 = _yolo_to_xyxy(int(width), int(height), bbox)
        except Exception:
            continue
        score = det.get("score")
        by_image.setdefault(img_id, []).append((x1, y1, x2, y2, score))
    return by_image


def _load_agent_mining_split(dataset_id: str) -> Optional[Dict[str, Any]]:
    split_path = _agent_mining_meta_dir(dataset_id) / "split.json"
    if not split_path.exists():
        return None
    try:
        with split_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
            data["_path"] = str(split_path)
            return data
    except Exception:
        return None


def _ensure_agent_mining_split(
    dataset_id: str,
    dataset_root: Path,
    *,
    val_percent: float,
    seed: int,
    reuse_split: bool,
    test_mode: bool,
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
) -> Dict[str, Any]:
    coco, _, images = _load_coco_index(dataset_root)
    categories = coco.get("categories") or []
    dataset_signature = _compute_dataset_signature(dataset_id, dataset_root, images, categories)
    image_ids = [int(img.get("id", idx)) for idx, img in enumerate(coco.get("images") or []) if "id" in img or idx >= 0]
    if not image_ids:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_mining_no_images")
    # Always rebuild split deterministically; ignore any baked-in splits.
    dir_signature = _compute_dir_signature(dataset_root)
    if reuse_split and not test_mode:
        cached = _load_agent_mining_split(dataset_id)
        cached_seed = cached.get("seed") if cached else None
        if (
            cached
            and cached_seed is not None
            and cached.get("signature") == dir_signature
            and cached.get("dataset_signature") == dataset_signature
            and abs(cached.get("val_percent", val_percent) - val_percent) < 1e-6
            and int(cached_seed) == int(seed)
        ):
            return {**cached, "_cached": True}
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    total = len(image_ids)
    if test_mode:
        # In test mode, honor explicit limits (with sensible defaults) and ignore cached splits.
        if val_limit is None or val_limit <= 0:
            val_limit = 10
        if train_limit is None or train_limit <= 0:
            train_limit = 10
        if val_limit + train_limit >= total:
            # If requested caps exceed dataset size, split deterministically: reserve val_limit (or half) and the rest train.
            val_take = min(val_limit, max(1, total // 2))
            val_ids = image_ids[:val_take]
            train_ids = image_ids[val_take: min(total, val_take + train_limit)]
        else:
            val_ids = image_ids[: min(val_limit, total)]
            train_ids = image_ids[min(val_limit, total) : min(val_limit + train_limit, total)]
    else:
        if val_limit is not None and val_limit <= 0:
            val_limit = 1
        val_count = int(total * val_percent)
        if val_count <= 0:
            val_count = 1 if total > 1 else 0
        if val_count >= total:
            val_count = max(1, total - 1)
        val_ids = image_ids[:val_count]
        train_ids = image_ids[val_count:]
        if train_limit is not None and train_limit > 0:
            train_ids = train_ids[:train_limit]
        if val_limit is not None and val_limit > 0:
            val_ids = val_ids[:val_limit]
    if not train_ids and val_ids:
        train_ids, val_ids = val_ids, []
    if not train_ids or not val_ids:
        raise HTTPException(
            status_code=HTTP_412_PRECONDITION_FAILED,
            detail=f"agent_split_empty:{len(train_ids)}:{len(val_ids)}",
        )
    split = {
        "train": train_ids,
        "val": val_ids,
        "val_percent": val_percent,
        "seed": seed,
        "total_images": total,
        "signature": dir_signature,
        "dataset_signature": dataset_signature,
        "created_at": time.time(),
        "test_mode": bool(test_mode),
        "train_limit": train_limit,
        "val_limit": val_limit,
    }
    if not test_mode:
        split_path = _agent_mining_meta_dir(dataset_id) / "split.json"
        try:
            with split_path.open("w", encoding="utf-8") as fp:
                json.dump(split, fp)
        except Exception:
            logger.exception("Failed to persist agent mining split to %s", split_path)
    return {**split, "_cached": False}


def _agent_mining_cache_key(
    *,
    class_id: Optional[int] = None,
    prompt: Optional[str],
    visual_ref: Optional[Dict[str, Any]],
    threshold: float,
    mask_threshold: float,
    min_size: int,
    simplify: float,
    max_results: int,
    similarity_score: Optional[float] = None,
    context: Optional[str] = None,
) -> str:
    visual_key = ""
    if visual_ref:
        visual_key = f"{visual_ref.get('image_id','')}:{','.join(map(str, visual_ref.get('bbox') or []))}"
    key_parts = [
        f"class={class_id}" if class_id is not None else "class=?",
        prompt or "",
        visual_key,
        f"thr={threshold:.4f}",
        f"mthr={mask_threshold:.4f}",
        f"min={min_size}",
        f"simplify={simplify:.4f}",
        f"max={max_results}",
    ]
    if similarity_score is not None:
        key_parts.append(f"sim={similarity_score:.4f}")
    if context:
        key_parts.append(f"context={context}")
    return _stable_hash(key_parts)


def _agent_mining_cache_paths(cache_dir: Path, key: str) -> Tuple[Path, Path]:
    """Return (gz_path, legacy_json_path) for a cache key."""
    return cache_dir / f"{key}.json.gz", cache_dir / f"{key}.json"


def _slim_detections(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the minimal fields needed downstream to cut disk usage."""
    kept_fields = {"image_id", "bbox", "score", "label", "class_id", "class_idx"}
    slimmed: List[Dict[str, Any]] = []
    for det in detections or []:
        try:
            slim = {k: det.get(k) for k in kept_fields if k in det}
            # ensure bbox is JSON-serializable list
            if "bbox" in slim and isinstance(slim["bbox"], tuple):
                slim["bbox"] = list(slim["bbox"])
            slimmed.append(slim)
        except Exception:
            continue
    return slimmed


def _load_agent_mining_detections(cache_dir: Path, key: str) -> Optional[List[Dict[str, Any]]]:
    path_gz, path_json = _agent_mining_cache_paths(cache_dir, key)
    path = path_gz if path_gz.exists() else path_json if path_json.exists() else None
    if path is None:
        return None
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as fp:
                return json.load(fp)
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


def _save_agent_mining_detections(cache_dir: Path, key: str, detections: List[Dict[str, Any]]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path_gz, path_json = _agent_mining_cache_paths(cache_dir, key)
    # remove legacy uncompressed file if present
    if path_json.exists():
        try:
            path_json.unlink()
        except Exception:
            logger.debug("Failed to remove legacy cache file %s", path_json)
    slimmed = _slim_detections(detections)
    try:
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=cache_dir, suffix=".tmp") as fp:
            with gzip.GzipFile(fileobj=fp, mode="w") as gz:
                gz.write(json.dumps(slimmed).encode("utf-8"))
            tmp_name = fp.name
        Path(tmp_name).replace(path_gz)
    except Exception:
        logger.exception("Failed to persist agent mining detections to %s", path_gz)
    _enforce_agent_mining_cache_limits(cache_dir, allow_when_running=True)


def _agent_cache_running_jobs() -> bool:
    with AGENT_MINING_JOBS_LOCK:
        return any(j.status == "running" for j in AGENT_MINING_JOBS.values())


def _enforce_agent_mining_cache_limits(cache_root: Path, *, allow_when_running: bool = False) -> Dict[str, int]:
    """
    Enforce TTL and total size limits on a cache directory. Returns stats.
    """
    stats = {"deleted_files": 0, "deleted_bytes": 0}
    if not cache_root.exists():
        return stats
    if _agent_cache_running_jobs() and not allow_when_running:
        return stats
    ttl_seconds = max(0, AGENT_MINING_CACHE_TTL_HOURS) * 3600
    now = time.time()
    files: List[Tuple[Path, float, int]] = []
    try:
        for p in cache_root.rglob("*"):
            if p.is_file():
                try:
                    stat = p.stat()
                    files.append((p, stat.st_mtime, stat.st_size))
                except Exception:
                    continue
    except Exception:
        return stats
    # TTL purge first
    for path, mtime, size in files:
        try:
            if ttl_seconds and (now - mtime) > ttl_seconds:
                path.unlink(missing_ok=True)
                stats["deleted_bytes"] += size
                stats["deleted_files"] += 1
        except Exception:
            continue
    # Recompute remaining and total size
    remaining: List[Tuple[Path, float, int]] = []
    total_size = 0
    try:
        for p in cache_root.rglob("*"):
            if not p.is_file():
                continue
            try:
                stat = p.stat()
                total_size += stat.st_size
                remaining.append((p, stat.st_mtime, stat.st_size))
            except Exception:
                continue
    except Exception:
        return stats
    if AGENT_MINING_CACHE_MAX_BYTES > 0 and total_size > AGENT_MINING_CACHE_MAX_BYTES:
        remaining.sort(key=lambda t: t[1])  # oldest first
        for path, _, size in remaining:
            if total_size <= AGENT_MINING_CACHE_MAX_BYTES:
                break
            try:
                path.unlink(missing_ok=True)
                total_size -= size
                stats["deleted_bytes"] += size
                stats["deleted_files"] += 1
            except Exception:
                continue
    # Remove empty directories to keep the tree clean.
    try:
        for d in sorted({p.parent for p, _, _ in remaining}, key=lambda p: len(p.parts), reverse=True):
            if d.exists():
                try:
                    if not any(d.iterdir()):
                        d.rmdir()
                except Exception:
                    continue
    except Exception:
        pass
    return stats


def _collect_agent_mining_detections(
    *,
    images: Dict[int, Dict[str, Any]],
    image_ids: Sequence[int],
    prompt: Optional[str],
    visual_ref: Optional[Dict[str, Any]],
    threshold: float,
    mask_threshold: float,
    min_size: int,
    simplify: float,
    max_results: int,
    cache_dir: Path,
    context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    cache_key = _agent_mining_cache_key(
        prompt=prompt,
        visual_ref=visual_ref,
        threshold=threshold,
        mask_threshold=mask_threshold,
        min_size=min_size,
        simplify=simplify,
        max_results=max_results,
        context=context,
    )
    cached = _load_agent_mining_detections(cache_dir, cache_key)
    if cached is not None:
        return cached
    start_ts = time.time()
    results: List[Dict[str, Any]] = []
    for img_id in image_ids:
        img_info = images.get(img_id)
        if not img_info:
            continue
        img_path = img_info.get("path")
        if not img_path:
            continue
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception:
            logger.exception("Agent mining failed to open image %s", img_path)
            continue
        detections: List[QwenDetection]
        try:
            if visual_ref:
                bbox = visual_ref.get("bbox") if isinstance(visual_ref, dict) else None
                if not bbox or len(bbox) < 4:
                    continue
                detections = _run_sam3_visual_inference(
                    pil_img,
                    (
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ),
                    threshold,
                    mask_threshold,
                    max_results,
                    min_size=min_size if min_size > 0 else None,
                    simplify_epsilon=simplify,
                )
            else:
                detections = _run_sam3_text_inference(
                    pil_img,
                    prompt or "",
                    threshold,
                    mask_threshold,
                    max_results,
                    min_size=min_size if min_size > 0 else None,
                    simplify_epsilon=simplify,
                )
        except Exception:
            logger.exception("Agent mining prompt failed for image %s", img_id)
            continue
        for det in detections:
            det_dict = det.dict()
            det_dict["image_id"] = img_id
            results.append(det_dict)
    _save_agent_mining_detections(cache_dir, cache_key, results)
    try:
        elapsed = time.time() - start_ts
        logger.info(
            "Agent mining collected %d detections for prompt=%s visual=%s over %d images in %.2fs",
            len(results),
            prompt if prompt else "",
            bool(visual_ref),
            len(image_ids),
            elapsed,
        )
    except Exception:
        pass
    return results


def _build_sam3_text_processor_for_device(device: torch.device) -> Tuple[Any, Any]:
    if SAM3_NATIVE_IMAGE_IMPORT_ERROR is not None or build_sam3_image_model is None or Sam3ImageProcessor is None:
        raise RuntimeError(f"sam3_text_unavailable:{SAM3_NATIVE_IMAGE_IMPORT_ERROR}")
    device_str = "cuda" if device.type == "cuda" else "cpu"
    try:
        model = build_sam3_image_model(
            checkpoint_path=active_sam3_checkpoint,
            device=device_str,
            load_from_HF=active_sam3_checkpoint is None,
            enable_segmentation=True,
            bpe_path=str(SAM3_BPE_PATH),
        )
        if device:
            model = model.to(device)
        processor = Sam3ImageProcessor(model)
        return model, processor
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"sam3_text_load_failed:{exc}") from exc


class _Sam3MiningWorker:
    def __init__(self, device: torch.device):
        self.device = device
        self.model, self.processor = _build_sam3_text_processor_for_device(device)
        self.lock = threading.Lock()

    def close(self) -> None:
        try:
            del self.processor
        except Exception:  # noqa: BLE001
            pass
        try:
            del self.model
        except Exception:  # noqa: BLE001
            pass
        if torch.cuda.is_available() and self.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def process_image(
        self,
        image_id: int,
        pil_img: Image.Image,
        tasks: Sequence[Dict[str, Any]],
        *,
        min_threshold: float,
        mask_threshold: float,
        max_results: int,
        min_size: int,
        simplify: float,
        return_masks: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run all requested tasks against a single preloaded image on this worker.

        Important: SAM3 state is *not* safe to reuse across prompts without clearing previous
        prompts/results. We keep the expensive image backbone in `state["backbone_out"]` but
        reset text/geometric prompts between tasks.

        Visual tasks in Agent Mining are "seeded visual prompts": we do not use a fixed bbox from
        the training image. Instead, we:
        1) run (or reuse) a text prompt on the current image to get candidate boxes,
        2) pick the best seed box by CLIP similarity to the exemplar crop embedding,
        3) run SAM3 geometric prompt using that seed box.
        """
        if not tasks:
            return {}
        with self.lock:
            try:
                self.processor.set_confidence_threshold(float(min_threshold))
            except Exception:
                pass
            state = self.processor.set_image(pil_img)

            def _reset_prompts() -> None:
                try:
                    if hasattr(self.processor, "reset_all_prompts"):
                        self.processor.reset_all_prompts(state)
                        return
                except Exception:
                    pass
                # Best-effort fallback if upstream changes.
                try:
                    if isinstance(state.get("backbone_out"), dict):
                        for k in ("language_features", "language_mask", "language_embeds"):
                            state["backbone_out"].pop(k, None)
                    for k in ("geometric_prompt", "boxes", "masks", "masks_logits", "scores"):
                        state.pop(k, None)
                except Exception:
                    pass

            outputs: Dict[str, List[Dict[str, Any]]] = {}
            text_dets_by_prompt: Dict[str, List[QwenDetection]] = {}

            # Run all text tasks first so visual tasks can reuse them for seeding.
            text_tasks = [t for t in tasks if t.get("type") == "text"]
            visual_tasks = [t for t in tasks if t.get("type") != "text"]

            for task in text_tasks:
                task_id = task.get("id")
                if not task_id:
                    continue
                prompt_text = (task.get("prompt") or "").strip()
                _reset_prompts()
                det_masks: Optional[List[np.ndarray]] = None
                try:
                    if return_masks:
                        dets, det_masks = _run_sam3_text_inference(
                            pil_img,
                            prompt_text,
                            min_threshold,
                            mask_threshold,
                            max_results,
                            min_size=min_size if min_size > 0 else None,
                            simplify_epsilon=simplify,
                            processor_override=self.processor,
                            state=state,
                            return_masks=True,
                        )
                    else:
                        dets = _run_sam3_text_inference(
                            pil_img,
                            prompt_text,
                            min_threshold,
                            mask_threshold,
                            max_results,
                            min_size=min_size if min_size > 0 else None,
                            simplify_epsilon=simplify,
                            processor_override=self.processor,
                            state=state,
                        )
                except Exception:
                    continue
                if prompt_text:
                    text_dets_by_prompt[prompt_text] = dets
                det_dicts: List[Dict[str, Any]] = []
                for det_idx, det in enumerate(dets):
                    try:
                        det_data = det.dict()
                    except Exception:
                        continue
                    det_data["image_id"] = image_id
                    if return_masks and det_masks is not None:
                        if 0 <= det_idx < len(det_masks):
                            det_data["mask_array"] = det_masks[det_idx]
                    det_dicts.append(det_data)
                outputs[task_id] = det_dicts

            # Lazy CLIP seed embedding cache per seed prompt for this image.
            seed_cache: Dict[str, Tuple[List[Tuple[float, float, float, float]], np.ndarray]] = {}

            def _encode_clip_batch(crops: List[Image.Image]) -> Optional[np.ndarray]:
                model, preprocess = _ensure_clip_backbone_for_mining()
                if model is None or preprocess is None or not crops:
                    return None
                try:
                    with clip_lock:
                        inp = torch.stack([preprocess(c) for c in crops], dim=0).to(device)
                        try:
                            target_dtype = next(model.parameters()).dtype
                        except Exception:
                            target_dtype = torch.float32
                        if inp.dtype != target_dtype:
                            inp = inp.to(dtype=target_dtype)
                        with torch.no_grad():
                            feats = model.encode_image(inp)
                        feats = feats.to(dtype=torch.float32, device="cpu")
                    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
                    return feats.cpu().numpy()
                except Exception as exc:
                    logger.debug("CLIP batch encode failed: %s", exc)
                    return None

            def _seed_candidates_for_prompt(prompt_text: str) -> List[QwenDetection]:
                prompt_text = (prompt_text or "").strip()
                if not prompt_text:
                    return []
                existing = text_dets_by_prompt.get(prompt_text)
                if existing is not None:
                    return existing
                # If the seed prompt wasn't part of the task slice, run it on-demand.
                _reset_prompts()
                try:
                    dets = _run_sam3_text_inference(
                        pil_img,
                        prompt_text,
                        min_threshold,
                        mask_threshold,
                        max_results,
                        min_size=min_size if min_size > 0 else None,
                        simplify_epsilon=simplify,
                        processor_override=self.processor,
                        state=state,
                    )
                except Exception:
                    dets = []
                text_dets_by_prompt[prompt_text] = dets
                return dets

            def _best_seed_bbox_xywh(
                seed_prompt: str,
                *,
                exemplar_vec: Optional[np.ndarray],
            ) -> Optional[Tuple[float, float, float, float]]:
                dets = _seed_candidates_for_prompt(seed_prompt)
                if not dets:
                    return None
                # Prefer CLIP-based seed selection when an exemplar embedding is present.
                if exemplar_vec is None:
                    # Fallback: pick highest-scoring text detection.
                    best = max(dets, key=lambda d: (d.score if d.score is not None else 0.0))
                    try:
                        left, top, right, bottom = yolo_to_corners(best.bbox, pil_img.width, pil_img.height)
                    except Exception:
                        return None
                    return float(left), float(top), float(right - left), float(bottom - top)

                seed_prompt_key = (seed_prompt or "").strip()
                cached = seed_cache.get(seed_prompt_key)
                if cached is None:
                    # Only embed a small top-k for performance.
                    ranked = sorted(dets, key=lambda d: (d.score if d.score is not None else 0.0), reverse=True)
                    top_k = ranked[:50]
                    bboxes_xywh: List[Tuple[float, float, float, float]] = []
                    crops: List[Image.Image] = []
                    for det in top_k:
                        bbox = det.bbox or []
                        if len(bbox) < 4:
                            continue
                        try:
                            left, top, right, bottom = yolo_to_corners(bbox, pil_img.width, pil_img.height)
                        except Exception:
                            continue
                        if right <= left or bottom <= top:
                            continue
                        try:
                            crop = pil_img.crop((left, top, right, bottom))
                        except Exception:
                            continue
                        bboxes_xywh.append((float(left), float(top), float(right - left), float(bottom - top)))
                        crops.append(crop)
                    feats = _encode_clip_batch(crops)
                    if feats is None or feats.size == 0:
                        seed_cache[seed_prompt_key] = (bboxes_xywh, np.zeros((0, 1), dtype=np.float32))
                    else:
                        seed_cache[seed_prompt_key] = (bboxes_xywh, feats)
                    cached = seed_cache.get(seed_prompt_key)
                bboxes_xywh, feats = cached if cached is not None else ([], np.zeros((0, 1), dtype=np.float32))
                if feats is None or feats.size == 0 or not bboxes_xywh:
                    return None
                try:
                    ex = np.asarray(exemplar_vec, dtype=np.float32).reshape(-1)
                    ex = ex / (np.linalg.norm(ex) + 1e-8)
                    sims = feats @ ex.reshape(-1, 1)
                    sims = sims.squeeze(-1)
                    idx_best = int(np.argmax(sims)) if sims.size else -1
                except Exception:
                    idx_best = -1
                if idx_best < 0 or idx_best >= len(bboxes_xywh):
                    return None
                return bboxes_xywh[idx_best]

            for task in visual_tasks:
                task_id = task.get("id")
                if not task_id:
                    continue
                # Legacy: fixed bbox (pixel xywh) provided directly.
                bbox = task.get("bbox")
                seed_bbox: Optional[Tuple[float, float, float, float]] = None
                if bbox and len(bbox) >= 4:
                    try:
                        seed_bbox = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                    except Exception:
                        seed_bbox = None
                if seed_bbox is None:
                    seed_prompt = task.get("seed_prompt") or ""
                    exemplar_vec = task.get("exemplar_vec")
                    seed_bbox = _best_seed_bbox_xywh(seed_prompt, exemplar_vec=exemplar_vec)
                if seed_bbox is None:
                    outputs[task_id] = []
                    continue
                _reset_prompts()
                det_masks: Optional[List[np.ndarray]] = None
                try:
                    if return_masks:
                        dets, det_masks = _run_sam3_visual_inference(
                            pil_img,
                            seed_bbox,
                            min_threshold,
                            mask_threshold,
                            max_results,
                            min_size=min_size if min_size > 0 else None,
                            simplify_epsilon=simplify,
                            processor_override=self.processor,
                            state=state,
                            return_masks=True,
                        )
                    else:
                        dets = _run_sam3_visual_inference(
                            pil_img,
                            seed_bbox,
                            min_threshold,
                            mask_threshold,
                            max_results,
                            min_size=min_size if min_size > 0 else None,
                            simplify_epsilon=simplify,
                            processor_override=self.processor,
                            state=state,
                        )
                except Exception:
                    continue
                det_dicts: List[Dict[str, Any]] = []
                for det_idx, det in enumerate(dets):
                    try:
                        det_data = det.dict()
                    except Exception:
                        continue
                    det_data["image_id"] = image_id
                    if return_masks and det_masks is not None:
                        if 0 <= det_idx < len(det_masks):
                            det_data["mask_array"] = det_masks[det_idx]
                    det_dicts.append(det_data)
                outputs[task_id] = det_dicts
            return outputs


class _Sam1SegWorker:
    def __init__(self, device: torch.device):
        model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
        if device:
            try:
                model = model.to(device)
            except Exception:
                pass
        self.device = device
        self.predictor = SamPredictor(model)
        self.lock = threading.Lock()

    def close(self) -> None:
        try:
            del self.predictor
        except Exception:
            pass
        if torch.cuda.is_available() and self.device and self.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def process_image(
        self,
        image_id: int,
        pil_img: Image.Image,
        tasks: Sequence[Dict[str, Any]],
        *,
        simplify: float,
        min_size: int = 0,
        **_: Any,
    ) -> Dict[str, List[Dict[str, Any]]]:
        outputs: Dict[str, List[Dict[str, Any]]] = {}
        if not tasks:
            return outputs
        np_img = np.array(pil_img.convert("RGB"))
        with self.lock:
            try:
                self.predictor.set_image(np_img)
            except Exception:
                return outputs
            for task in tasks:
                task_id = task.get("id")
                bbox = task.get("bbox")
                class_idx = task.get("class_idx")
                fallback = task.get("fallback_poly")
                if not task_id or not bbox or len(bbox) < 4:
                    continue
                x, y, w, h = bbox
                xyxy = np.array([x, y, x + w, y + h])
                try:
                    masks, scores, _ = self.predictor.predict(
                        box=xyxy[None, :],
                        multimask_output=True,
                        return_logits=False,
                    )
                except Exception:
                    continue
                if masks is None or len(masks) == 0:
                    continue
                scores_arr = np.asarray(scores) if scores is not None else None
                idx_best = int(np.argmax(scores_arr)) if scores_arr is not None and scores_arr.size else 0
                mask_arr = masks[idx_best]
                area = float(np.count_nonzero(mask_arr))
                if min_size and area < float(min_size):
                    continue
                outputs[task_id] = [
                    {
                        "image_id": image_id,
                        "mask_array": mask_arr,
                        "score": float(scores_arr[idx_best]) if scores_arr is not None and scores_arr.size else None,
                        "class_idx": class_idx,
                        "fallback_poly": fallback,
                    }
                ]
        return outputs


class _Sam3MiningPool:
    def __init__(self, devices: Sequence[torch.device]):
        self.workers: List[_Sam3MiningWorker] = []
        for dev in devices:
            try:
                self.workers.append(_Sam3MiningWorker(dev))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to initialize SAM3 mining worker on %s: %s", dev, exc)
        if not self.workers:
            raise RuntimeError("sam3_mining_workers_unavailable")

    def close(self) -> None:
        for worker in self.workers:
            try:
                worker.close()
            except Exception:
                continue

    def run(
        self,
        image_entries: Sequence[Tuple[int, str]],
        tasks: Sequence[Dict[str, Any]],
        *,
        min_threshold: float,
        mask_threshold: float,
        max_results: int,
        min_size: int,
        simplify: float,
        cancel_event: Optional[threading.Event] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        return_masks: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        if not tasks or not image_entries:
            return {task.get("id"): [] for task in tasks if task.get("id")}
        results: Dict[str, List[Dict[str, Any]]] = {task.get("id"): [] for task in tasks if task.get("id")}
        max_workers = max(1, len(self.workers))

        # Fast path: single worker, run sequentially to avoid thread overhead and potential native race conditions.
        if max_workers == 1:
            worker = self.workers[0]
            for idx, (img_id, path) in enumerate(image_entries, start=1):
                if cancel_event is not None and cancel_event.is_set():
                    break
                try:
                    with Image.open(path) as img:
                        pil_img = img.convert("RGB")
                except Exception:
                    continue
                if cancel_event is not None and cancel_event.is_set():
                    break
                partial = worker.process_image(
                    img_id,
                    pil_img,
                    tasks,
                    min_threshold=min_threshold,
                    mask_threshold=mask_threshold,
                    max_results=max_results,
                    min_size=min_size,
                    simplify=simplify,
                    return_masks=return_masks,
                )
                for key, dets in partial.items():
                    if key is None or not dets:
                        continue
                    results.setdefault(key, []).extend(dets)
                if progress_callback:
                    try:
                        progress_callback(idx)
                    except Exception:
                        pass
            return results

        processed = 0
        proc_lock = threading.Lock()

        def _run_task(worker: _Sam3MiningWorker, img_id: int, path: str) -> Dict[str, List[Dict[str, Any]]]:
            if cancel_event is not None and cancel_event.is_set():
                return {}
            if not path:
                return {}
            try:
                with Image.open(path) as img:
                    pil_img = img.convert("RGB")
            except Exception:
                return {}
            if cancel_event is not None and cancel_event.is_set():
                return {}
            return worker.process_image(
                img_id,
                pil_img,
                tasks,
                min_threshold=min_threshold,
                mask_threshold=mask_threshold,
                max_results=max_results,
                min_size=min_size,
                simplify=simplify,
                return_masks=return_masks,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, (img_id, path) in enumerate(image_entries):
                if cancel_event is not None and cancel_event.is_set():
                    break
                worker = self.workers[idx % max_workers]
                futures.append(executor.submit(_run_task, worker, img_id, path))
            for future in as_completed(futures):
                try:
                    partial = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Agent mining worker failed: %s", exc)
                    continue
                if cancel_event is not None and cancel_event.is_set():
                    break
                for key, dets in partial.items():
                    if key is None:
                        continue
                    if dets:
                        results.setdefault(key, []).extend(dets)
                if progress_callback:
                    with proc_lock:
                        processed += 1
                        try:
                            progress_callback(processed)
                        except Exception:
                            pass
        return results


def _collect_agent_mining_detections_image_first(
    *,
    candidates: Sequence[Dict[str, Any]],
    thresholds: Sequence[float],
    similarity_scores: Optional[Sequence[float]],
    images: Dict[int, Dict[str, Any]],
    image_ids: Sequence[int],
    mask_threshold: float,
    min_size: int,
    simplify: float,
    max_results: int,
    cache_dir: Path,
    pool: _Sam3MiningPool,
    use_cache: bool = True,
    cancel_event: Optional[threading.Event] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
    cache_context: Optional[str] = None,
) -> Tuple[Dict[str, Dict[float, Dict[Optional[float], List[Dict[str, Any]]]]], Dict[str, Any]]:
    """
    Collect detections for all candidates over all images in an image-first manner. We run the
    lowest threshold once per image and reuse scores to materialize higher-threshold variants.
    """
    thresholds_list = [float(t) for t in thresholds] if thresholds else [0.3]
    thresholds_list = [t for t in thresholds_list if 0.0 <= t <= 1.0]
    if not thresholds_list:
        thresholds_list = [0.3]
    sim_list = [float(s) for s in (similarity_scores or []) if 0.0 <= float(s) <= 1.0]
    if not sim_list:
        sim_list = [None]
    min_threshold = min(thresholds_list)
    results: Dict[str, Dict[float, Dict[Optional[float], List[Dict[str, Any]]]]] = {}
    missing: List[Dict[str, Any]] = []
    # For streaming flush
    executed_keys: set[str] = set()
    executed_keys_with_dets: set[str] = set()
    cached_pairs = 0
    executed_pairs = 0
    executed_pairs_with_dets = 0
    # First try to satisfy from cache.
    if use_cache:
        for cand in candidates:
            cand_id = cand.get("id")
            if not cand_id:
                continue
            results[cand_id] = {}
            all_cached = True
            for thr in thresholds_list:
                for sim in sim_list:
                    cached = _load_agent_mining_detections(
                        cache_dir,
                        _agent_mining_cache_key(
                            class_id=cand.get("class_id"),
                            prompt=cand.get("prompt"),
                            visual_ref=cand.get("visual_ref"),
                            threshold=thr,
                            mask_threshold=mask_threshold,
                            min_size=min_size,
                            simplify=simplify,
                            max_results=max_results,
                            similarity_score=sim,
                            context=cache_context,
                        ),
                    )
                    if cached is not None:
                        results[cand_id].setdefault(thr, {})[sim] = cached
                        cached_pairs += 1
                    else:
                        all_cached = False
            if not all_cached:
                missing.append(cand)
    else:
        missing = list(candidates)
        for cand in candidates:
            cand_id = cand.get("id")
            if not cand_id:
                continue
            results[cand_id] = {}
    if missing and (cancel_event is None or not cancel_event.is_set()):
        image_entries: List[Tuple[int, str]] = []
        for img_id in image_ids:
            info = images.get(img_id) or {}
            path = info.get("path")
            if path:
                image_entries.append((img_id, path))
        # Remove stale cache files for the keys we will regenerate to avoid duplicate appends across runs.
        keys_to_reset: set[str] = set()
        for cand in missing:
            for thr in thresholds_list:
                for sim in sim_list:
                    cache_key = _agent_mining_cache_key(
                        class_id=cand.get("class_id"),
                        prompt=cand.get("prompt"),
                        visual_ref=cand.get("visual_ref"),
                        threshold=thr,
                        mask_threshold=mask_threshold,
                        min_size=min_size,
                        simplify=simplify,
                        max_results=max_results,
                        similarity_score=sim,
                        context=cache_context,
                    )
                    keys_to_reset.add(cache_key)
        for key in keys_to_reset:
            path_gz, path_json = _agent_mining_cache_paths(cache_dir, key)
            for p in (path_gz, path_json):
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        logger.debug("Failed to remove stale cache file %s", p)
        # Stream in chunks to limit memory, flushing each chunk to cache.
        # Keep chunk_size small and also cap how many candidates are evaluated per chunk to limit RAM/VRAM.
        chunk_size = 4
        max_cands_per_chunk = 16
        accumulator: Dict[str, List[Dict[str, Any]]] = {}
        processed_total = 0

        def flush_accumulator() -> None:
            nonlocal accumulator
            if not accumulator:
                return
            for key, items in accumulator.items():
                if cancel_event is not None and cancel_event.is_set():
                    break
                # We already cleared stale cache entries for these keys above,
                # so write the accumulated detections directly to avoid repeated read/extend churn.
                _save_agent_mining_detections(cache_dir, key, items)
            accumulator = {}

        for start in range(0, len(image_entries), chunk_size):
            if cancel_event is not None and cancel_event.is_set():
                break
            batch_entries = image_entries[start : start + chunk_size]
            # Optionally slice candidates to keep per-chunk work bounded.
            cand_chunks = [missing[i : i + max_cands_per_chunk] for i in range(0, len(missing), max_cands_per_chunk)]
            logger.info(
                "[agent-mining] chunk %d/%d images %d-%d/%d (candidates=%d, thresholds=%d)",
                (start // chunk_size) + 1,
                math.ceil(len(image_entries) / chunk_size),
                start + 1,
                min(start + len(batch_entries), len(image_entries)),
                len(image_entries),
                len(missing),
                len(thresholds_list),
            )
            for cand_slice in cand_chunks:
                if cancel_event is not None and cancel_event.is_set():
                    break
                try:
                    pooled = pool.run(
                        batch_entries,
                        cand_slice,
                        min_threshold=min_threshold,
                        mask_threshold=mask_threshold,
                        max_results=max_results,
                        min_size=min_size,
                        simplify=simplify,
                        cancel_event=cancel_event,
                        progress_callback=None,
                    )
                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        "SAM3 mining OOM; retrying slice with reduced candidate batch (cand_slice=%d)",
                        len(cand_slice),
                    )
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    # Retry one-by-one to limp forward
                    pooled = {}
                    for single_cand in cand_slice:
                        try:
                            partial_single = pool.run(
                                batch_entries,
                                [single_cand],
                                min_threshold=min_threshold,
                                mask_threshold=mask_threshold,
                                max_results=max_results,
                                min_size=min_size,
                                simplify=simplify,
                                cancel_event=cancel_event,
                                progress_callback=None,
                            )
                            pooled.update(partial_single)
                        except torch.cuda.OutOfMemoryError:
                            logger.error("SAM3 mining OOM even on single candidate; aborting batch.")
                            raise
                for cand in cand_slice:
                    cand_id = cand.get("id")
                    if not cand_id:
                        continue
                    base_dets = pooled.get(cand_id, [])
                    for thr in thresholds_list:
                        filtered = [
                            det
                            for det in base_dets
                            if ((det.get("score") is None and thr <= min_threshold) or (det.get("score") or 0.0) >= thr)
                        ]
                        for sim in sim_list:
                            cache_key = _agent_mining_cache_key(
                                class_id=cand.get("class_id"),
                                prompt=cand.get("prompt"),
                                visual_ref=cand.get("visual_ref"),
                                threshold=thr,
                                mask_threshold=mask_threshold,
                                min_size=min_size,
                                simplify=simplify,
                                max_results=max_results,
                                similarity_score=sim,
                                context=cache_context,
                            )
                            accumulator.setdefault(cache_key, []).extend(filtered)
                            executed_keys.add(cache_key)
                            if filtered:
                                executed_keys_with_dets.add(cache_key)
                flush_accumulator()
                try:
                    # Free pooled results explicitly to lower peak RAM.
                    pooled.clear()
                except Exception:
                    pass
                # Give the GPU a chance to release memory between slices.
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            processed_total += len(batch_entries)
            if progress_callback:
                try:
                    progress_callback(processed_total)
                except Exception:
                    pass
        flush_accumulator()
        # Reload from cache for all missing candidates.
        for cand in missing:
            cand_id = cand.get("id")
            if not cand_id:
                continue
            for thr in thresholds_list:
                for sim in sim_list:
                    cache_key = _agent_mining_cache_key(
                        class_id=cand.get("class_id"),
                        prompt=cand.get("prompt"),
                        visual_ref=cand.get("visual_ref"),
                        threshold=thr,
                        mask_threshold=mask_threshold,
                        min_size=min_size,
                        simplify=simplify,
                        max_results=max_results,
                        similarity_score=sim,
                        context=cache_context,
                    )
                    cached = _load_agent_mining_detections(cache_dir, cache_key) or []
                    results.setdefault(cand_id, {}).setdefault(thr, {})[sim] = cached
        executed_pairs += len(executed_keys)
        executed_pairs_with_dets += len(executed_keys_with_dets)
    stats = {
        "images": len(image_ids),
        "candidates": len(candidates),
        "thresholds": len(thresholds_list),
        "cached_pairs": cached_pairs,
        "executed_pairs": executed_pairs,
        "executed_pairs_with_dets": executed_pairs_with_dets,
    }
    logger.info(
        "[agent-mining] global sweep done: images=%d candidates=%d thresholds=%d cached=%d executed=%d det_pairs=%d",
        len(image_ids),
        len(candidates),
        len(thresholds_list),
        cached_pairs,
        executed_pairs,
        executed_pairs_with_dets,
    )
    return results, stats


def _clip_embed_regions(
    regions: List[Dict[str, Any]],
    images: Dict[int, Dict[str, Any]],
    *,
    max_regions: int = 256,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Embed cropped regions (image_id + YOLO bbox) with raw CLIP. Returns (id->embedding, warnings).
    """
    model, preprocess = _ensure_clip_backbone_for_mining()
    warnings: List[str] = []
    if model is None or preprocess is None:
        warnings.append("clip_unavailable")
        return {}, warnings
    embeddings: Dict[str, np.ndarray] = {}
    device_to_use = device if isinstance(device, str) or isinstance(device, torch.device) else "cpu"
    def _encode_image(image: Image.Image) -> Optional[np.ndarray]:
        try:
            target_dtype = next(model.parameters()).dtype
        except Exception:
            target_dtype = torch.float32
        try:
            inp = preprocess(image).unsqueeze(0).to(device_to_use)
            if inp.dtype != target_dtype:
                inp = inp.to(dtype=target_dtype)
            with torch.no_grad():
                feats = model.encode_image(inp)
            feats = feats.to(dtype=torch.float32, device="cpu")
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            return feats.squeeze(0).cpu().numpy()
        except Exception as exc:
            logger.debug("CLIP encode failed: %s", exc)
            return None
    for entry in regions[:max_regions]:
        try:
            img_id = int(entry.get("image_id"))
        except Exception:
            continue
        bbox = entry.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        info = images.get(img_id) or {}
        path = entry.get("path") or info.get("path")
        if not path:
            crop_path = entry.get("crop_path")
            if crop_path:
                try:
                    rel = Path(str(crop_path))
                except Exception:
                    rel = None
                if rel and not rel.is_absolute() and ".." not in rel.parts:
                    candidate = (AGENT_MINING_RECIPES_ROOT / rel).resolve()
                    if (
                        candidate.exists()
                        and candidate.is_file()
                        and _path_is_within_root(candidate, AGENT_MINING_RECIPES_ROOT.resolve())
                    ):
                        path = str(candidate)
        if not path:
            continue
        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception:
            continue
        try:
            bbox4 = list(map(float, bbox[:4]))
            if all(0.0 <= v <= 1.0 for v in bbox4):  # YOLO (cx, cy, w, h) normalized
                left, top, right, bottom = yolo_to_corners(bbox4, pil_img.width, pil_img.height)
            else:  # treat as COCO-style pixel xywh (x, y, w, h)
                x, y, w, h = map(float, bbox[:4])
                left = int(round(x))
                top = int(round(y))
                right = int(round(x + w))
                bottom = int(round(y + h))
                left = max(0, min(pil_img.width, left))
                top = max(0, min(pil_img.height, top))
                right = max(left, min(pil_img.width, right))
                bottom = max(top, min(pil_img.height, bottom))
            if right <= left or bottom <= top:
                continue
            crop = pil_img.crop((left, top, right, bottom))
            emb = _encode_image(crop)
            if emb is None:
                continue
            key = entry.get("embed_id") or f"{img_id}:{left},{top},{right},{bottom}"
            embeddings[key] = emb
        except Exception as exc:
            logger.debug("CLIP embed crop failed: %s", exc)
            continue
    if not embeddings:
        warnings.append("clip_embedding_empty")
    return embeddings, warnings


def _clip_fp_filter_detections(
    detections: List[Dict[str, Any]],
    *,
    exemplar_embeddings: Dict[str, np.ndarray],
    negative_embeddings: Optional[Dict[str, np.ndarray]] = None,
    negative_strength: float = 0.0,
    images: Dict[int, Dict[str, Any]],
    similarity_floor: float,
    max_regions: int = 512,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Filter detections whose CLIP similarity to any exemplar is below the floor."""
    warnings: List[str] = []
    if not exemplar_embeddings:
        return detections, warnings
    model, preprocess = _ensure_clip_backbone_for_mining()
    if model is None or preprocess is None:
        warnings.append("clip_unavailable")
        return detections, warnings
    filtered: List[Dict[str, Any]] = []
    exemplars_mat = np.stack(list(exemplar_embeddings.values()))
    ex_norm = np.linalg.norm(exemplars_mat, axis=1, keepdims=True) + 1e-8
    exemplars_mat = exemplars_mat / ex_norm

    neg_mat = None
    if negative_embeddings:
        neg_mat = np.stack(list(negative_embeddings.values()))
        neg_norm = np.linalg.norm(neg_mat, axis=1, keepdims=True) + 1e-8
        neg_mat = neg_mat / neg_norm
    device_to_use = device if isinstance(device, str) or isinstance(device, torch.device) else "cpu"
    def _encode_image(image: Image.Image) -> Optional[np.ndarray]:
        try:
            target_dtype = next(model.parameters()).dtype
        except Exception:
            target_dtype = torch.float32
        try:
            inp = preprocess(image).unsqueeze(0).to(device_to_use)
            if inp.dtype != target_dtype:
                inp = inp.to(dtype=target_dtype)
            with torch.no_grad():
                feats = model.encode_image(inp)
            feats = feats.to(dtype=torch.float32, device="cpu")
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            return feats.squeeze(0).cpu().numpy()
        except Exception as exc:
            logger.debug("CLIP encode failed: %s", exc)
            return None
    for det in detections[:max_regions]:
        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        try:
            img_id = int(det.get("image_id"))
        except Exception:
            continue
        info = images.get(img_id) or {}
        path = info.get("path")
        if not path:
            continue
        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception:
            continue
        try:
            bbox4 = list(map(float, bbox[:4]))
            if all(0.0 <= v <= 1.0 for v in bbox4):  # YOLO (cx, cy, w, h) normalized
                left, top, right, bottom = yolo_to_corners(bbox4, pil_img.width, pil_img.height)
            else:  # COCO-style pixel xywh
                x, y, w, h = map(float, bbox[:4])
                left = int(round(x))
                top = int(round(y))
                right = int(round(x + w))
                bottom = int(round(y + h))
                left = max(0, min(pil_img.width, left))
                top = max(0, min(pil_img.height, top))
                right = max(left, min(pil_img.width, right))
                bottom = max(top, min(pil_img.height, bottom))
            if right <= left or bottom <= top:
                continue
            crop = pil_img.crop((left, top, right, bottom))
            emb = _encode_image(crop)
            if emb is None:
                continue
            sims = (exemplars_mat @ emb.reshape(-1, 1)).squeeze()
            max_sim = float(np.max(sims)) if sims.size else 0.0
            max_neg_sim = 0.0
            if neg_mat is not None:
                neg_sims = (neg_mat @ emb.reshape(-1, 1)).squeeze()
                max_neg_sim = float(np.max(neg_sims)) if neg_sims.size else 0.0
            score = max_sim - max(0.0, negative_strength) * max_neg_sim
            if max_sim >= similarity_floor and score >= 0:
                filtered.append(det)
        except Exception as exc:
            logger.debug("CLIP filter crop failed: %s", exc)
            continue
    if len(filtered) < len(detections):
        warnings.append("clip_fp_filtered")
    return filtered if filtered else detections, warnings


def _sample_agent_mining_exemplars(
    cat_id: int,
    train_ids: Sequence[int],
    gt_by_image_cat: Dict[int, Dict[int, List[List[float]]]],
    images: Dict[int, Dict[str, Any]],
    *,
    limit: int,
    seed: int,
    candidate_mode: Literal["percent", "count"] = "percent",
    candidate_value: int = 25,
    use_clip_selection: bool = True,
    cluster: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], List[str], Dict[str, Any]]:
    """
    Sample exemplars for a class. Caps the candidate pool, optionally embeds with CLIP, and
    selects a diverse set via k-center. Returns (exemplars, exemplar_embeddings, warnings, stats).
    """
    if limit <= 0:
        return [], {}, [], {"candidates": 0, "embedded": 0, "selected": 0, "mode": candidate_mode, "value": candidate_value}
    rng = random.Random(seed + cat_id)
    candidates: List[Tuple[int, List[float]]] = []
    for img_id in train_ids:
        anns = gt_by_image_cat.get(img_id, {}).get(cat_id)
        if not anns:
            continue
        for bbox in anns:
            candidates.append((img_id, list(map(float, bbox[:4]))))
    total_candidates = len(candidates)
    if not candidates:
        return [], {}, ["no_candidates"], {"candidates": 0, "embedded": 0, "selected": 0, "mode": candidate_mode, "value": candidate_value}

    if candidate_mode == "percent":
        candidate_value = max(1, min(100, candidate_value))
        pool_cap = max(1, int(math.ceil(total_candidates * (candidate_value / 100.0))))
    else:
        pool_cap = max(1, min(candidate_value, total_candidates))

    rng.shuffle(candidates)
    pool = candidates[:pool_cap]
    pool_entries: List[Dict[str, Any]] = []
    for img_id, bbox in pool:
        try:
            x, y, w, h = bbox
            area = max(0.0, w * h)
        except Exception:
            x = y = w = h = 0.0
            area = 0.0
        img_info = images.get(img_id, {})
        embed_id = f"{img_id}:{x:.2f},{y:.2f},{w:.2f},{h:.2f}"
        pool_entries.append(
            {
                "image_id": img_id,
                "file_name": img_info.get("file_name"),
                "path": str(img_info.get("path") or ""),
                "bbox": [x, y, w, h],
                "area": area,
                "embed_id": embed_id,
            }
        )

    selection_target = limit * 3 if cluster else limit
    warnings: List[str] = []
    exemplar_embeddings: Dict[str, np.ndarray] = {}
    selected: List[Dict[str, Any]] = []

    def _select_k_center(regions: List[Dict[str, Any]], embeds: Dict[str, np.ndarray], k: int) -> List[Dict[str, Any]]:
        if k <= 0 or not regions or not embeds:
            return []
        keyed = [(r, embeds.get(r.get("embed_id", ""))) for r in regions if r.get("embed_id") in embeds]
        keyed = [(r, e) for r, e in keyed if e is not None]
        if not keyed:
            return []
        vecs = np.stack([e for _, e in keyed])
        areas = np.array([float(r.get("area", 0.0)) for r, _ in keyed])
        start_idx = int(np.argmax(areas))
        selected_indices = [start_idx]
        dists = np.ones(len(keyed), dtype=np.float32)
        dists *= np.inf
        dists = np.minimum(dists, 1.0 - (vecs @ vecs[start_idx]))
        while len(selected_indices) < min(k, len(keyed)):
            next_idx = int(np.argmax(dists))
            if not np.isfinite(dists[next_idx]):
                break
            selected_indices.append(next_idx)
            dists = np.minimum(dists, 1.0 - (vecs @ vecs[next_idx]))
        return [keyed[i][0] for i in selected_indices]

    embedded_count = 0
    if use_clip_selection and pool_entries:
        exemplar_embeddings, clip_warn = _clip_embed_regions(pool_entries, images, max_regions=len(pool_entries))
        warnings.extend(clip_warn)
        embedded_count = len(exemplar_embeddings)
        selected = _select_k_center(pool_entries, exemplar_embeddings, selection_target)

    if not selected:
        step = max(1, len(pool_entries) // selection_target) if selection_target > 0 else 1
        selected = pool_entries[::step][:selection_target]
        if not exemplar_embeddings and use_clip_selection:
            warnings.append("clip_selection_fallback_random")

    if cluster and len(selected) > limit:
        if exemplar_embeddings:
            selected = _k_center_select(selected, exemplar_embeddings, limit)
        else:
            selected = _cluster_agent_mining_exemplars(selected, max_items=limit, seed=seed + cat_id + 17)
    if len(selected) > limit:
        selected = selected[:limit]

    if exemplar_embeddings:
        selected_ids = {ex.get("embed_id") for ex in selected if ex.get("embed_id")}
        exemplar_embeddings = {k: v for k, v in exemplar_embeddings.items() if k in selected_ids}
    stats = {
        "candidates": total_candidates,
        "pool": len(pool_entries),
        "embedded": embedded_count,
        "selected": len(selected),
        "mode": candidate_mode,
        "value": candidate_value,
    }
    return selected, exemplar_embeddings, warnings, stats


def _cluster_agent_mining_exemplars(
    exemplars: List[Dict[str, Any]],
    *,
    max_items: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Placeholder clustering: down-sample exemplars evenly when clustering is requested."""
    if max_items <= 0 or len(exemplars) <= max_items:
        return exemplars
    rng = random.Random(seed)
    exemplars = exemplars[:]
    rng.shuffle(exemplars)
    step = max(1, len(exemplars) // max_items)
    clustered: List[Dict[str, Any]] = []
    for idx in range(0, len(exemplars), step):
        clustered.append(exemplars[idx])
        if len(clustered) >= max_items:
            break
    return clustered


def _k_center_select(regions: List[Dict[str, Any]], embeds: Dict[str, np.ndarray], k: int) -> List[Dict[str, Any]]:
    """Greedy k-center over normalized embeddings keyed by region['embed_id']."""
    if k <= 0 or not regions or not embeds:
        return []
    keyed = [(r, embeds.get(r.get("embed_id", ""))) for r in regions if r.get("embed_id") in embeds]
    keyed = [(r, e) for r, e in keyed if e is not None]
    if not keyed:
        return []
    vecs = np.stack([e for _, e in keyed])
    areas = np.array([float(r.get("area", 0.0)) for r, _ in keyed])
    start_idx = int(np.argmax(areas))
    selected_indices = [start_idx]
    dists = np.ones(len(keyed), dtype=np.float32)
    dists *= np.inf
    dists = np.minimum(dists, 1.0 - (vecs @ vecs[start_idx]))
    while len(selected_indices) < min(k, len(keyed)):
        next_idx = int(np.argmax(dists))
        if not np.isfinite(dists[next_idx]):
            break
        selected_indices.append(next_idx)
        dists = np.minimum(dists, 1.0 - (vecs @ vecs[next_idx]))
    return [keyed[i][0] for i in selected_indices]


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
        metadata, signature = _ensure_qwen_dataset_signature(path, metadata)
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
            "signature": signature,
            "type": metadata.get("type", "bbox"),
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
                updated = False
                if "id" not in data:
                    data["id"] = dataset_dir.name
                    updated = True
                if "type" not in data:
                    data["type"] = "bbox"
                    updated = True
                if updated:
                    _persist_qwen_dataset_metadata(dataset_dir, data)
                return data
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read Qwen dataset metadata from %s: %s", meta_path, exc)
    return None


def _normalize_qwen_image_rel(src_path: Path, dataset_root: Path, fallback: Path) -> Path:
    try:
        rel = src_path.relative_to(dataset_root)
    except Exception:
        rel = fallback
    parts = list(rel.parts)
    if parts and parts[0] in {"train", "val"}:
        parts = parts[1:]
    if parts and parts[0] == "images":
        parts = parts[1:]
    if not parts:
        parts = [fallback.name]
    return Path(*parts)


def _prepare_qwen_training_split(
    dataset_root: Path,
    job_id: str,
    *,
    random_split: bool,
    val_percent: float,
    split_seed: int,
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    log_messages: Optional[List[str]] = None,
) -> Path:
    if not random_split:
        return dataset_root
    meta = _load_qwen_dataset_metadata(dataset_root) or {}
    entries: List[Dict[str, Any]] = []
    for split in ("train", "val"):
        jsonl_path = dataset_root / split / "annotations.jsonl"
        if not jsonl_path.exists():
            continue
        try:
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        payload = json.loads(raw)
                    except Exception:
                        continue
                    image_rel = payload.get("image")
                    if not isinstance(image_rel, str) or not image_rel.strip():
                        continue
                    rel_path = Path(image_rel.strip())
                    candidates = [
                        dataset_root / split / rel_path,
                        dataset_root / split / "images" / rel_path,
                        dataset_root / "images" / rel_path,
                        dataset_root / "train" / rel_path,
                        dataset_root / "val" / rel_path,
                        dataset_root / rel_path,
                        dataset_root / "train" / "images" / rel_path,
                        dataset_root / "val" / "images" / rel_path,
                    ]
                    src_path = next((p for p in candidates if p.exists()), None)
                    if src_path is None:
                        continue
                    normalized_rel = _normalize_qwen_image_rel(src_path, dataset_root, rel_path)
                    entries.append(
                        {
                            "raw": raw,
                            "image_rel": normalized_rel,
                            "src": src_path,
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read Qwen annotations from %s: %s", jsonl_path, exc)
    if not entries:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="qwen_training_no_annotations")
    rnd = random.Random(split_seed)
    rnd.shuffle(entries)
    total = len(entries)
    vp = max(0.0, min(float(val_percent), 0.9))
    val_count = int(total * vp)
    if val_count <= 0 and total > 1:
        val_count = 1
    if val_limit is not None and val_limit > 0:
        val_count = min(val_count if val_count > 0 else val_limit, val_limit, total - 1 if total > 1 else total)
    val_entries = entries[:val_count]
    train_entries = entries[val_count:]
    if train_limit is not None and train_limit > 0:
        train_entries = train_entries[:train_limit]
    if not train_entries or not val_entries:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="qwen_training_split_empty")
    split_root = (QWEN_JOB_ROOT / "splits" / job_id).resolve()
    split_root.parent.mkdir(parents=True, exist_ok=True)
    if split_root.exists():
        shutil.rmtree(split_root, ignore_errors=True)
    (split_root / "train" / "images").mkdir(parents=True, exist_ok=True)
    (split_root / "val" / "images").mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    for split_name, split_entries in (("train", train_entries), ("val", val_entries)):
        ann_path = split_root / split_name / "annotations.jsonl"
        with ann_path.open("w", encoding="utf-8") as handle:
            for entry in split_entries:
                dst = split_root / split_name / "images" / entry["image_rel"]
                _link_or_copy_file(entry["src"], dst)
                try:
                    record = json.loads(entry["raw"])
                except Exception:
                    record = {"image": ""}
                record["image"] = entry["image_rel"].as_posix()
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split_name] += 1
    new_meta = {
        **meta,
        "id": meta.get("id") or dataset_root.name,
        "label": meta.get("label") or meta.get("id") or dataset_root.name,
        "classes": meta.get("classes") or _load_qwen_labelmap(dataset_root),
        "context": meta.get("context") or "",
        "created_at": meta.get("created_at") or time.time(),
        "train_count": counts["train"],
        "val_count": counts["val"],
        "image_count": counts["train"] + counts["val"],
    }
    _persist_qwen_dataset_metadata(split_root, new_meta)
    split_summary = (
        f"Qwen split: {counts['train']} train / {counts['val']} val "
        f"(seed={split_seed}, val_percent={vp:.2f}, src={dataset_root}) -> {split_root}"
    )
    logger.info(split_summary)
    if log_messages is not None:
        log_messages.append(split_summary)
    return split_root


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


def _build_qwen_config(payload: QwenTrainRequest, job_id: str, job_logs: Optional[List[str]] = None) -> QwenTrainingConfig:
    if QWEN_TRAINING_IMPORT_ERROR is not None or QwenTrainingConfig is None:
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"qwen_training_unavailable:{QWEN_TRAINING_IMPORT_ERROR}",
        )
    dataset_root = Path(os.path.abspath(payload.dataset_root))
    if not dataset_root.is_dir():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="dataset_root_not_found")
    val_percent = payload.val_percent if payload.val_percent is not None else 0.3
    split_seed = int(payload.split_seed) if payload.split_seed is not None else 42
    random_split = payload.random_split if payload.random_split is not None else True
    train_limit = int(payload.train_limit) if payload.train_limit is not None and payload.train_limit > 0 else None
    val_limit = int(payload.val_limit) if payload.val_limit is not None and payload.val_limit > 0 else None
    dataset_root = _prepare_qwen_training_split(
        dataset_root,
        job_id,
        random_split=random_split,
        val_percent=val_percent,
        split_seed=split_seed,
        train_limit=train_limit,
        val_limit=val_limit,
        log_messages=job_logs,
    )
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="dataset_missing_train_val")
    meta = _load_qwen_dataset_metadata(dataset_root) or {}
    if meta.get("train_count", 0) <= 0 or meta.get("val_count", 0) <= 0:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="dataset_empty_split")
    if not meta.get("classes"):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="dataset_missing_classes")
    run_name = payload.run_name or f"qwen_run_{job_id}"
    result_path = (QWEN_JOB_ROOT / run_name).resolve()
    if not str(result_path).startswith(str(QWEN_JOB_ROOT.resolve())):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="run_path_invalid")
    if result_path.exists():
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="run_name_exists")
    system_prompt = (payload.system_prompt or DEFAULT_SYSTEM_PROMPT).strip() or DEFAULT_SYSTEM_PROMPT
    cfg_kwargs: Dict[str, Any] = {
        "dataset_root": str(dataset_root),
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


def _safe_extract_zip(
    zf: zipfile.ZipFile,
    dest_dir: Path,
    *,
    strip_root: bool = False,
    max_bytes_per_file: Optional[int] = None,
    total_quota_bytes: Optional[int] = None,
) -> None:
    """Safely extract a zip into dest_dir, rejecting paths that escape."""
    dest_dir = dest_dir.resolve()
    extracted_bytes = 0
    for member in zf.namelist():
        # Skip directories explicitly; we'll create as needed.
        if not member or member.endswith("/"):
            continue
        try:
            info = zf.getinfo(member)
            file_size = getattr(info, "file_size", 0)
        except KeyError:
            file_size = 0
        if max_bytes_per_file and file_size > max_bytes_per_file:
            raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="zip_entry_too_large")
        if total_quota_bytes and extracted_bytes + file_size > total_quota_bytes:
            raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="zip_quota_exceeded")
        rel = _normalise_relative_path(member)
        if strip_root and len(rel.parts) > 1:
            rel = Path(*rel.parts[1:])
        target = (dest_dir / rel).resolve()
        if not str(target).startswith(str(dest_dir)):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="zip_entry_invalid_path")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member, "r") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        extracted_bytes += max(file_size, 0)


def _link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except Exception:
            continue
    return total


def _purge_directory(path: Path) -> int:
    if not path.exists():
        return 0
    deleted = 0
    for p in sorted(path.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        try:
            if p.is_file():
                deleted += p.stat().st_size
                p.unlink()
            elif p.is_dir():
                p.rmdir()
        except Exception:
            continue
    try:
        path.rmdir()
    except Exception:
        pass
    return deleted


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
    with CLIP_DATASET_JOBS_LOCK:
        active_roots = {str(j.root_dir) for j in CLIP_DATASET_JOBS.values()}
    _purge_staging_dirs(CLIP_DATASET_UPLOAD_ROOT, active_roots=active_roots)
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
    await _write_upload_file(
        file,
        dest_path,
        max_bytes=CLIP_DATASET_CHUNK_MAX_BYTES,
        quota_root=job.root_dir,
        quota_limit=CLIP_DATASET_UPLOAD_QUOTA_BYTES,
        allow_overwrite=True,  # allow idempotent retries for same relative_path
    )
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
    with QWEN_DATASET_JOBS_LOCK:
        active_roots = {str(j.root_dir) for j in QWEN_DATASET_JOBS.values()}
    # Limit purge to staging_* dirs to avoid deleting real datasets.
    _purge_staging_dirs(QWEN_DATASET_ROOT, active_roots=active_roots, prefix="staging_")
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
    await _write_upload_file(
        file,
        dest_path,
        max_bytes=QWEN_DATASET_CHUNK_MAX_BYTES,
        quota_root=job.root_dir,
        quota_limit=QWEN_DATASET_UPLOAD_QUOTA_BYTES,
        allow_overwrite=True,  # allow idempotent retries
    )
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
    signature = _compute_dir_signature(job.root_dir)
    existing = _find_qwen_dataset_by_signature(signature)
    if existing is not None:
        shutil.rmtree(job.root_dir, ignore_errors=True)
        existing_meta = _load_qwen_dataset_metadata(existing) or {}
        existing_meta, _ = _ensure_qwen_dataset_signature(existing, existing_meta)
        logger.info(
            "[qwen-dataset %s] reused existing dataset=%s (signature match)",
            job_id[:8],
            existing.name,
        )
        return {
            "dataset_root": str(existing),
            "run_name": existing.name,
            "metadata": existing_meta,
            "reused": True,
        }
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
        "signature": signature,
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
    return {"dataset_root": str(dest_dir), "run_name": safe_name, "metadata": dataset_meta, "reused": False}


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
    _purge_dataset_artifacts(dataset_id)
    return {"status": "deleted"}


def _find_yolo_dataset_root(extracted_dir: Path) -> Optional[Path]:
    candidates: List[Path] = [extracted_dir]
    for child in extracted_dir.iterdir():
        if child.is_dir():
            candidates.append(child)
    for candidate in candidates:
        labelmap_path = candidate / "labelmap.txt"
        train_images = candidate / "train" / "images"
        train_labels = candidate / "train" / "labels"
        val_images = candidate / "val" / "images"
        val_labels = candidate / "val" / "labels"
        root_images = candidate / "images"
        root_labels = candidate / "labels"
        if not labelmap_path.exists():
            continue
        if (train_images.exists() and train_labels.exists()) or (root_images.exists() and root_labels.exists()):
            return candidate
    return None


def _count_images_in_dir(images_dir: Path) -> int:
    if not images_dir.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    count = 0
    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            count += 1
    return count


def _infer_yolo_dataset_type(labels_dir: Path, fallback: str = "bbox") -> str:
    if not labels_dir.exists():
        return fallback
    try:
        for txt in labels_dir.rglob("*.txt"):
            with txt.open("r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) > 5:
                        return "seg"
    except Exception:
        return fallback
    return fallback


@app.post("/datasets/upload")
async def upload_dataset_zip(
    file: UploadFile = File(...),
    dataset_id: Optional[str] = Form(None),
    dataset_type: Optional[str] = Form(None),
):
    # Purge stale staging before creating a new one.
    _purge_staging_dirs(DATASET_UPLOAD_ROOT, active_roots=set(), prefix="dataset_upload_")
    filename = file.filename or "dataset.zip"
    safe_name = _safe_run_name(dataset_id, Path(filename).stem or f"dataset_{uuid.uuid4().hex[:6]}")
    tmp_root = Path(tempfile.mkdtemp(prefix="dataset_upload_", dir=str(DATASET_UPLOAD_ROOT)))
    zip_path = tmp_root / "payload.zip"
    try:
        await _write_upload_file(file, zip_path, max_bytes=DATASET_ZIP_MAX_BYTES)
        extracted_dir = tmp_root / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_extract_zip(
                zf,
                extracted_dir,
                max_bytes_per_file=DATASET_ZIP_ENTRY_MAX_BYTES,
                total_quota_bytes=DATASET_ZIP_MAX_BYTES,
            )
        dataset_root = _find_yolo_dataset_root(extracted_dir)
        if not dataset_root:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="yolo_layout_not_found")
        # Normalize layout: accept either train/val splits or root-level images/labels (no split).
        train_images = dataset_root / "train" / "images"
        train_labels = dataset_root / "train" / "labels"
        root_images = dataset_root / "images"
        root_labels = dataset_root / "labels"
        if root_images.exists() and root_labels.exists() and not train_images.exists():
            (dataset_root / "train").mkdir(parents=True, exist_ok=True)
            shutil.move(str(root_images), str(train_images.parent))
            shutil.move(str(root_labels), str(train_labels.parent))
        if not train_images.exists() or not train_labels.exists():
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="train_split_missing")
        target_dir = (DATASET_REGISTRY_ROOT / safe_name).resolve()
        if not str(target_dir).startswith(str(DATASET_REGISTRY_ROOT.resolve())):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="dataset_path_invalid")
        if target_dir.exists():
            raise HTTPException(status_code=HTTP_409_CONFLICT, detail="dataset_exists")
        # Ensure labelmap.txt exists and read classes.
        labelmap_path = dataset_root / "labelmap.txt"
        if not labelmap_path.exists():
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="labelmap_txt_missing")
        try:
            with labelmap_path.open("r", encoding="utf-8") as handle:
                labelmap = [line.strip() for line in handle if line.strip()]
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"labelmap_txt_invalid:{exc}") from exc
        shutil.move(str(dataset_root), str(target_dir))
        dataset_kind = (dataset_type or "").strip().lower() or _infer_yolo_dataset_type(target_dir / "train" / "labels", "bbox")
        if dataset_kind not in {"bbox", "seg"}:
            dataset_kind = "bbox"
        train_count = _count_images_in_dir(target_dir / "train" / "images")
        val_dir = target_dir / "val" / "images"
        val_count = _count_images_in_dir(val_dir) if val_dir.exists() else 0
        image_count = train_count + val_count
        signature = _compute_dir_signature(target_dir)
        metadata = {
            "id": safe_name,
            "label": safe_name,
            "dataset_root": str(target_dir),
            "type": dataset_kind,
            "source": "upload",
            "created_at": time.time(),
            "image_count": image_count,
            "train_count": train_count,
            "val_count": val_count,
            "classes": labelmap,
            "signature": signature,
        }
        _persist_sam3_dataset_metadata(target_dir, metadata)
        # Build COCO JSONs immediately so Qwen/SAM3 consumers have them ready.
        try:
            coco_meta = _convert_yolo_dataset_to_coco(target_dir, existing_meta=metadata)
            metadata.update(
                {
                    "coco_train_json": coco_meta.get("coco_train_json"),
                    "coco_val_json": coco_meta.get("coco_val_json"),
                }
            )
            _persist_sam3_dataset_metadata(target_dir, metadata)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[dataset-upload] failed COCO conversion for %s: %s", safe_name, exc)
        logger.info(
            "[dataset-upload] stored=%s type=%s train=%d val=%d classes=%d",
            safe_name,
            dataset_kind,
            train_count,
            val_count,
            len(labelmap),
        )
        return metadata
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


@app.get("/datasets")
def list_datasets():
    return _list_sam3_datasets()


@app.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    target_entry = None
    for entry in _list_all_datasets(prefer_registry=True):
        if dataset_id in (entry.get("id"), entry.get("signature")):
            target_entry = entry
            break
    if not target_entry:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="dataset_not_found")
    target_path = Path(target_entry["dataset_root"]).resolve()
    allowed_roots = {
        DATASET_REGISTRY_ROOT.resolve(),
        SAM3_DATASET_ROOT.resolve(),
        QWEN_DATASET_ROOT.resolve(),
    }
    if not any(str(target_path).startswith(str(root)) for root in allowed_roots):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="dataset_path_invalid")
    try:
        shutil.rmtree(target_path, ignore_errors=False)
    except FileNotFoundError:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="dataset_not_found")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    _purge_dataset_artifacts(dataset_id)
    return {"status": "deleted", "dataset_id": dataset_id}


@app.get("/sam3/datasets")
def list_sam3_datasets():
    return _list_sam3_datasets()


@app.get("/sam3/datasets/{dataset_id}/classes")
def get_sam3_dataset_classes(dataset_id: str):
    dataset_root = _resolve_sam3_or_qwen_dataset(dataset_id)
    coco, _, _ = _load_coco_index(dataset_root)
    categories = coco.get("categories") or []
    classes: List[str] = []
    class_ids: List[int] = []
    for idx, cat in enumerate(categories):
        try:
            cid = int(cat.get("id", idx))
        except Exception:
            cid = idx
        class_ids.append(cid)
        classes.append(str(cat.get("name", f"class_{cid}")))
    return {"dataset_id": dataset_id, "classes": classes, "class_ids": class_ids}


@app.post("/sam3/datasets/{dataset_id}/convert")
def sam3_convert_dataset(dataset_id: str):
    dataset_root = _resolve_sam3_or_qwen_dataset(dataset_id)
    annotations_path = dataset_root / "train" / "annotations.jsonl"
    train_images = dataset_root / "train" / "images"
    train_labels = dataset_root / "train" / "labels"
    if annotations_path.exists():
        meta = _convert_qwen_dataset_to_coco(dataset_root)
    elif train_images.exists() and train_labels.exists():
        meta = _convert_yolo_dataset_to_coco(dataset_root)
    else:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_dataset_type_unsupported")
    return meta


def _find_coco_split(dataset_root: Path) -> Tuple[Path, Path]:
    """Return (annotations_path, images_dir) preferring val split, then train."""
    val_ann = dataset_root / "val" / "_annotations.coco.json"
    if val_ann.exists():
        images_dir = val_ann.parent / "images"
        if not images_dir.exists():
            images_dir = val_ann.parent
        return val_ann, images_dir
    train_ann = dataset_root / "train" / "_annotations.coco.json"
    if train_ann.exists():
        images_dir = train_ann.parent / "images"
        if not images_dir.exists():
            images_dir = train_ann.parent
        return train_ann, images_dir
    raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="coco_annotations_missing")


def _load_coco_index(dataset_root: Path) -> Tuple[Dict[str, Any], Dict[int, Dict[int, List[List[float]]]], Dict[int, Dict[str, Any]]]:
    ann_paths: List[Tuple[Path, Path]] = []
    for split in ("train", "val"):
        ann_file = dataset_root / split / "_annotations.coco.json"
        if ann_file.exists():
            images_dir = ann_file.parent / "images"
            if not images_dir.exists():
                images_dir = ann_file.parent
            ann_paths.append((ann_file, images_dir))
    if not ann_paths:
        ann_path, images_dir = _find_coco_split(dataset_root)
        ann_paths = [(ann_path, images_dir)]
    images: Dict[int, Dict[str, Any]] = {}
    gt_by_image_cat: Dict[int, Dict[int, List[List[float]]]] = {}
    coco_merged: Dict[str, Any] = {"images": [], "annotations": [], "categories": []}
    categories_map: Dict[int, Dict[str, Any]] = {}
    for ann_path, images_dir in ann_paths:
        try:
            with ann_path.open("r", encoding="utf-8") as handle:
                coco = json.load(handle)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"coco_load_failed:{exc}") from exc
        for cat in coco.get("categories", []) or []:
            try:
                cid = int(cat.get("id"))
            except Exception:
                continue
            categories_map.setdefault(cid, cat)
        for img in coco.get("images", []) or []:
            try:
                img_id = int(img["id"])
            except Exception:
                continue
            path = (images_dir / img["file_name"]).resolve()
            images[img_id] = {**img, "path": path}
        for ann in coco.get("annotations", []) or []:
            try:
                img_id = int(ann["image_id"])
                cat_id = int(ann["category_id"])
            except Exception:
                continue
            bbox = ann.get("bbox")
            if bbox is None:
                continue
            gt_by_image_cat.setdefault(img_id, {}).setdefault(cat_id, []).append(list(bbox))
    coco_merged["categories"] = sorted(categories_map.values(), key=lambda c: int(c.get("id", 0)))
    coco_merged["images"] = list(
        {
            img_id: {
                "id": img_id,
                "file_name": str(img.get("file_name") or Path(img.get("path", "")).name),
                "width": img.get("width"),
                "height": img.get("height"),
            }
            for img_id, img in images.items()
        }.values()
    )
    # Rebuild annotations from gt_by_image_cat so downstream uses the merged mapping.
    annotations: List[Dict[str, Any]] = []
    ann_id = 1
    for img_id, cat_map in gt_by_image_cat.items():
        for cat_id, boxes in cat_map.items():
            for bbox in boxes:
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": bbox,
                    }
                )
                ann_id += 1
    coco_merged["annotations"] = annotations
    return coco_merged, gt_by_image_cat, images


def _yolo_to_xyxy(width: int, height: int, bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    cx, cy, bw, bh = map(float, bbox[:4])
    x1 = max(0.0, (cx - bw / 2.0) * width)
    y1 = max(0.0, (cy - bh / 2.0) * height)
    x2 = min(float(width), (cx + bw / 2.0) * width)
    y2 = min(float(height), (cy + bh / 2.0) * height)
    return x1, y1, x2, y2


def _xywh_to_xyxy(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = map(float, bbox[:4])
    return x, y, x + w, y + h


def _iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _humanize_class_name(name: str) -> str:
    return re.sub(r"[\\-_]+", " ", name).strip()


def _generate_prompt_variants_for_class(class_name: str, max_synonyms: int, use_qwen: bool) -> List[str]:
    base: List[str] = []
    cleaned = class_name.strip()
    if cleaned:
        base.append(cleaned)
    human = _humanize_class_name(cleaned)
    if human and human.lower() != cleaned.lower():
        base.append(human)
    base_lower = {b.lower() for b in base if b}
    base_words = []
    for entry in base_lower:
        base_words.extend(re.split(r"[\\s_\\-]+", entry))
    base_words = [w for w in base_words if w]
    variants: List[str] = []
    if use_qwen and max_synonyms > 0:
        try:
            text = _generate_prompt_text(
                (
                    f"Generate up to {max_synonyms} alternative, common English labels for the object class "
                    f"'{human or cleaned}'. Each label must be 1-3 full words, each word at least 3 letters. "
                    "No abbreviations, no partial/truncated words, no numbering, no JSON. Avoid repeating the original name. "
                    "Use labels typical of object-detection datasets (e.g., car -> car, automobile, sedan; "
                    "person -> person, human, individual; utility pole -> utility pole, telephone pole, power pole). "
                    "Return a single comma-separated list."
                ),
                max_new_tokens=96,
            )
            raw_parts = re.split(r"[\\n;,]+", text)
            for part in raw_parts:
                normalized = part.strip().strip('"').strip("'")
                if not normalized:
                    continue
                if any(ch in normalized for ch in "{}[]:\""):
                    continue
                alpha_chars = re.sub(r"[^A-Za-z]", "", normalized)
                if len(alpha_chars) < 3:
                    continue
                if len(normalized) > 40:
                    continue
                if not re.search(r"[A-Za-z]", normalized):
                    continue
                words = normalized.split()
                if len(words) > 4:
                    continue
                if any(len(w) < 3 for w in words):
                    continue
                lowered = normalized.lower()
                fragment_of_base = any(
                    (lowered != b and lowered.startswith(b[: max(1, len(b) - 2)]))
                    or (b.startswith(lowered) and (len(b) - len(lowered) <= 2))
                    for b in base_lower
                )
                if fragment_of_base:
                    continue
                # Drop obvious truncated tokens relative to base words.
                truncated_token = False
                for w in words:
                    lw = w.lower()
                    for bw in base_words:
                        if not bw:
                            continue
                        if bw.startswith(lw) and len(bw) - len(lw) >= 2:
                            truncated_token = True
                            break
                        if lw.startswith(bw) and len(lw) - len(bw) >= 3:
                            truncated_token = True
                            break
                    if truncated_token:
                        break
                if truncated_token:
                    continue
                variants.append(normalized)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prompt helper: Qwen generation failed for %s: %s", class_name, exc)
    seen = set()
    ordered: List[str] = []
    for item in [*base, *variants]:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    if max_synonyms <= 0:
        return ordered[:1]
    # Always keep the first/base entry, then up to max_synonyms additional candidates.
    head = ordered[:1]
    tail = ordered[1 : 1 + max_synonyms]
    return head + tail


def _suggest_prompts_for_dataset(payload: PromptHelperSuggestRequest) -> Dict[str, Any]:
    dataset_root = _resolve_sam3_or_qwen_dataset(payload.dataset_id)
    coco, _, _ = _load_coco_index(dataset_root)
    categories = coco.get("categories") or []
    cat_to_images: Dict[int, set[int]] = {}
    cat_to_gts: Dict[int, int] = {}
    for ann in coco.get("annotations", []):
        try:
            cat_id = int(ann["category_id"])
            img_id = int(ann["image_id"])
        except Exception:
            continue
        cat_to_images.setdefault(cat_id, set()).add(img_id)
        cat_to_gts[cat_id] = cat_to_gts.get(cat_id, 0) + 1
    classes: List[Dict[str, Any]] = []
    for idx, cat in enumerate(categories):
        cat_id = int(cat.get("id", idx))
        class_name = str(cat.get("name", f"class_{cat_id}"))
        prompts = _generate_prompt_variants_for_class(class_name, payload.max_synonyms, payload.use_qwen)
        classes.append(
            {
                "class_id": cat_id,
                "class_name": class_name,
                "default_prompts": prompts,
                "image_count": len(cat_to_images.get(cat_id, set())),
                "gt_count": cat_to_gts.get(cat_id, 0),
            }
        )
    return {
        "dataset_id": payload.dataset_id,
        "config": payload.dict(),
        "classes": classes,
    }


def _list_prompt_helper_presets() -> List[Dict[str, Any]]:
    presets: List[Dict[str, Any]] = []
    for path in PROMPT_HELPER_PRESET_ROOT.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            presets.append(data)
        except Exception:
            continue
    presets.sort(key=lambda p: p.get("created_at", 0), reverse=True)
    return presets


def _load_prompt_helper_preset(preset_id: str) -> Dict[str, Any]:
    path = (PROMPT_HELPER_PRESET_ROOT / f"{preset_id}.json").resolve()
    if not str(path).startswith(str(PROMPT_HELPER_PRESET_ROOT.resolve())) or not path.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="prompt_helper_preset_not_found")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"prompt_helper_preset_load_failed:{exc}") from exc


def _save_prompt_helper_preset(label: str, dataset_id: str, prompts_by_class: Dict[int, List[str]]) -> Dict[str, Any]:
    preset_id = f"phset_{uuid.uuid4().hex[:8]}"
    created_at = time.time()
    payload = {
        "id": preset_id,
        "label": label or preset_id,
        "dataset_id": dataset_id_clean or "portable",
        "created_at": created_at,
        "prompts_by_class": prompts_by_class,
    }
    path = (PROMPT_HELPER_PRESET_ROOT / f"{preset_id}.json").resolve()
    if not str(path).startswith(str(PROMPT_HELPER_PRESET_ROOT.resolve())):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="prompt_helper_preset_path_invalid")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def _list_prompt_recipe_presets() -> List[Dict[str, Any]]:
    presets: List[Dict[str, Any]] = []
    for path in PROMPT_RECIPE_PRESET_ROOT.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            presets.append(data)
        except Exception:
            continue
    presets.sort(key=lambda p: p.get("created_at", 0), reverse=True)
    return presets


def _load_prompt_recipe_preset(preset_id: str) -> Dict[str, Any]:
    path = (PROMPT_RECIPE_PRESET_ROOT / f"{preset_id}.json").resolve()
    if not str(path).startswith(str(PROMPT_RECIPE_PRESET_ROOT.resolve())) or not path.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="prompt_recipe_preset_not_found")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"prompt_recipe_preset_load_failed:{exc}"
        ) from exc


def _save_prompt_recipe_preset(
    label: str,
    class_name: str,
    class_id: Optional[int],
    steps: List[Dict[str, Any]],
    dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not steps:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="prompt_recipe_preset_empty_steps")
    preset_id = f"prset_{uuid.uuid4().hex[:8]}"
    created_at = time.time()
    payload = {
        "id": preset_id,
        "label": label or preset_id,
        "class_name": class_name,
        "class_id": class_id,
        "steps": [{"prompt": s.get("prompt"), "threshold": s.get("threshold")} for s in steps],
        "dataset_id": dataset_id,
        "created_at": created_at,
    }
    path = (PROMPT_RECIPE_PRESET_ROOT / f"{preset_id}.json").resolve()
    if not str(path).startswith(str(PROMPT_RECIPE_PRESET_ROOT.resolve())):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="prompt_recipe_preset_path_invalid")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


    # (Legacy single-split loader removed; combined loader above)


def _sample_images_for_category(cat_id: int, img_ids: List[int], sample_size: int, seed: int) -> List[int]:
    if not img_ids:
        return []
    rnd = random.Random(seed + cat_id * 9973)
    if len(img_ids) <= sample_size:
        return list(img_ids)
    return rnd.sample(img_ids, sample_size)


def _sample_negative_images(cat_id: int, all_img_ids: List[int], cat_to_images: Dict[int, set[int]], sample_size: int, seed: int) -> List[int]:
    """Pick images that do NOT contain the target category."""
    negative_pool = [img_id for img_id in all_img_ids if img_id not in cat_to_images.get(cat_id, set())]
    if not negative_pool or sample_size <= 0:
        return []
    rnd = random.Random(seed + cat_id * 15391)
    if len(negative_pool) <= sample_size:
        return negative_pool
    return rnd.sample(negative_pool, sample_size)


def _evaluate_prompt_for_class(
    prompt: str,
    *,
    cat_id: int,
    image_ids: List[int],
    gt_by_image_cat: Dict[int, Dict[int, List[List[float]]]],
    images: Dict[int, Dict[str, Any]],
    score_threshold: float,
    max_dets: int,
    iou_threshold: float,
    image_cache: Dict[int, Image.Image],
) -> Dict[str, Any]:
    total_gt = 0
    total_preds = 0
    matches = 0
    det_images = 0
    iou_sum = 0.0
    score_sum = 0.0
    matched_scores = 0
    for img_id in image_ids:
        info = images.get(img_id)
        if not info:
            continue
        path = info.get("path")
        width = info.get("width")
        height = info.get("height")
        if not path or width is None or height is None:
            continue
        gts = [*gt_by_image_cat.get(img_id, {}).get(cat_id, [])]
        gt_boxes = [_xywh_to_xyxy(b) for b in gts]
        total_gt += len(gt_boxes)
        if not gt_boxes:
            continue
        try:
            pil_img = image_cache[img_id]
        except KeyError:
            try:
                pil_img = Image.open(path).convert("RGB")
            except Exception:
                continue
            image_cache[img_id] = pil_img
        preds = _run_sam3_text_inference(
            pil_img,
            prompt,
            threshold=score_threshold,
            mask_threshold=0.0,
            limit=max_dets,
        )
        pred_boxes: List[Tuple[float, float, float, float, Optional[float]]] = []
        for det in preds:
            try:
                x1, y1, x2, y2 = _yolo_to_xyxy(pil_img.width, pil_img.height, det.bbox)
                pred_boxes.append((x1, y1, x2, y2, det.score))
            except Exception:
                continue
        if not pred_boxes:
            continue
        pred_boxes.sort(key=lambda b: (b[4] if b[4] is not None else 0.0), reverse=True)
        total_preds += len(pred_boxes)
        gt_used = [False] * len(gt_boxes)
        matched_in_image = 0
        for x1, y1, x2, y2, score in pred_boxes:
            best_iou = 0.0
            best_idx = -1
            for idx, gt_box in enumerate(gt_boxes):
                if gt_used[idx]:
                    continue
                iou = _iou((x1, y1, x2, y2), gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold and best_idx >= 0:
                gt_used[best_idx] = True
                matches += 1
                matched_in_image += 1
                iou_sum += best_iou
                if score is not None:
                    score_sum += score
                    matched_scores += 1
        if matched_in_image > 0:
            det_images += 1
    precision = matches / total_preds if total_preds else 0.0
    recall = matches / total_gt if total_gt else 0.0
    det_rate = det_images / len(image_ids) if image_ids else 0.0
    avg_iou = iou_sum / matches if matches else None
    avg_score = score_sum / matched_scores if matched_scores else None
    f1 = (2 * precision * recall) / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
    overall_score = f1 * (0.5 + 0.5 * det_rate)
    fps = max(0, total_preds - matches)
    return {
        "prompt": prompt,
        "precision": precision,
        "recall": recall,
        "det_rate": det_rate,
        "avg_iou": avg_iou,
        "avg_score": avg_score,
        "score": overall_score,
        "f1": f1,
        "preds": total_preds,
        "matches": matches,
        "gts": total_gt,
        "fps": fps,
    }


class PromptHelperSuggestRequest(BaseModel):
    dataset_id: str
    max_synonyms: int = Field(3, ge=0, le=10)
    use_qwen: bool = True

class PromptHelperPreset(BaseModel):
    id: str
    label: str
    dataset_id: str
    created_at: float
    prompts_by_class: Dict[int, List[str]]


class PromptHelperRequest(BaseModel):
    dataset_id: str
    sample_per_class: int = Field(10, ge=1, le=1000)
    max_synonyms: int = Field(3, ge=0, le=10)
    score_threshold: float = Field(0.2, ge=0.0, le=1.0)
    max_dets: int = Field(100, ge=1, le=2000)
    iou_threshold: float = Field(0.5, ge=0.0, le=1.0)
    seed: int = 42
    use_qwen: bool = True
    # Optional explicit prompts provided by the user; key is category_id.
    prompts_by_class: Optional[Dict[int, List[str]]] = None


class PromptHelperSearchRequest(BaseModel):
    dataset_id: str
    sample_per_class: int = Field(20, ge=1, le=2000)
    negatives_per_class: int = Field(20, ge=0, le=2000)
    score_threshold: float = Field(0.2, ge=0.0, le=1.0)
    max_dets: int = Field(100, ge=1, le=2000)
    iou_threshold: float = Field(0.5, ge=0.0, le=1.0)
    seed: int = 42
    precision_floor: float = Field(0.9, ge=0.0, le=1.0)
    prompts_by_class: Dict[int, List[str]]
    class_id: Optional[int] = None


class PromptRecipePrompt(BaseModel):
    prompt: str
    thresholds: Optional[List[float]] = None


class PromptRecipeRequest(BaseModel):
    dataset_id: str
    class_id: int
    prompts: List[PromptRecipePrompt]
    sample_size: int = Field(30, ge=1, le=5000)
    negatives: int = Field(0, ge=0, le=5000)
    max_dets: int = Field(100, ge=1, le=2000)
    iou_threshold: float = Field(0.5, ge=0.0, le=1.0)
    seed: int = 42
    score_threshold: float = Field(0.2, ge=0.0, le=1.0)
    threshold_candidates: Optional[List[float]] = None


class PromptRecipeExpandRequest(BaseModel):
    dataset_id: str
    class_id: int
    base_prompts: List[str]
    max_new: int = Field(10, ge=0, le=50)


class AgentMiningRequest(BaseModel):
    dataset_id: str
    classes: Optional[List[int]] = None
    val_percent: float = Field(0.3, ge=0.05, le=0.95)
    split_seed: int = 42
    reuse_split: bool = True

    # Prompt mining (GPT-OSS).
    text_prompts_by_class: Optional[Dict[int, List[str]]] = None
    prompt_llm_max_prompts: int = Field(10, ge=0, le=50)
    prompt_reasoning: Literal["none", "low", "medium", "high"] = "none"
    prompt_max_new_tokens: int = Field(160, ge=16, le=800)
    # Deprecated: hint text injected into prompts (no longer used).
    class_hints: Optional[Dict[str, str]] = None
    # Optional: extra user-provided prompt phrases per class (merged into the prompt list).
    extra_prompts_by_class: Optional[Dict[str, List[str]]] = None

    # Optional pretrained CLIP head (LogReg) to use for filtering instead of raw crop-similarity.
    # This should point at a classifier file on the backend (typically under uploads/classifiers/).
    clip_head_classifier_path: Optional[str] = None
    clip_head_min_prob: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum probability for the target class when using an embedded CLIP head.",
    )
    clip_head_margin: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Require target prob to exceed best other class by this margin when using an embedded CLIP head.",
    )

    # Exemplar (crop bank) selection.
    positives_per_class: int = Field(20, ge=0, le=500)
    cluster_exemplars: bool = True
    exemplar_candidate_mode: Literal["percent", "count"] = "percent"
    exemplar_candidate_value: int = Field(25, ge=1, le=10_000)

    # Cross-class negatives (crop bank) used for CLIP rejection.
    use_negative_exemplars: bool = True
    max_negatives_per_class: int = Field(25, ge=0, le=256)
    negative_strength: float = Field(0.5, ge=0.0, le=5.0)

    # Greedy SAM3 recipe parameters.
    seed_threshold: float = Field(0.05, ge=0.0, le=1.0)
    expand_threshold: float = Field(0.3, ge=0.0, le=1.0)
    max_visual_seeds: int = Field(25, ge=0, le=500)
    seed_dedupe_iou: float = Field(0.9, ge=0.0, le=1.0)
    dedupe_iou: float = Field(0.5, ge=0.0, le=1.0)
    mask_threshold: float = Field(0.5, ge=0.0, le=1.0)
    similarity_score: float = Field(0.25, ge=0.0, le=1.0)
    max_results: int = Field(100, ge=1, le=5000)
    min_size: int = Field(0, ge=0, le=10_000)
    simplify_epsilon: float = Field(0.0, ge=0.0, le=1_000.0)

    # Always-on guardrails in this mode.
    use_clip_fp_guard: bool = True

    # Evaluation / debug.
    iou_threshold: float = Field(0.5, ge=0.0, le=1.0)
    test_mode: bool = False
    test_train_limit: int = Field(10, ge=1, le=10_000)
    test_val_limit: int = Field(10, ge=1, le=10_000)


def _expand_prompts_with_prompt_llm(
    class_name: str,
    base_prompts: List[str],
    max_new: int,
    log_fn: Optional[Callable[[str], None]] = None,
    max_new_tokens: int = 128,
    reasoning: Literal["none", "low", "medium", "high"] = "none",
) -> List[str]:
    """Use the configured prompt LLM (GPT-OSS) to brainstorm additional prompt variants for a class."""
    cleaned_base = _sanitize_prompts(base_prompts)
    if max_new <= 0 or not cleaned_base:
        return []

    seen = {p.lower() for p in cleaned_base}
    suggestions: List[str] = []
    try:
        known_list_str = ", ".join(cleaned_base)

        def _log(msg: str) -> None:
            if log_fn:
                try:
                    log_fn(msg)
                except Exception:
                    pass

        def _run_brainstorm_with_retries(remaining: int, round_idx: int) -> List[str]:
            """Try up to 3 times (initial + 2 critiques) to get a clean list."""
            base_prompt = [
                "Generate diverse noun-phrase prompts for open-vocabulary object detection with SAM3.",
                f"Target class: '{_humanize_class_name(class_name)}'.",
                f"Known good prompts: {known_list_str}.",
            ]
            base_prompt.extend(
                [
                    f"Propose up to {remaining} NEW, concrete object names (1-3 words) that strictly describe this class.",
                    "Rules: letters/spaces/hyphens only; no numbers; no punctuation beyond commas between items; no adjectives alone; avoid repeats.",
                    "Return ONLY a comma-separated list. Example: thing one, thing two, thing three",
                ]
            )
            last_text = ""
            for attempt in range(3):
                prompt_lines = list(base_prompt)
                if attempt > 0:
                    prompt_lines.extend(
                        [
                            f"Previous output was invalid: {last_text or '(empty)'}",
                            f"Try again. Respond ONLY with up to {remaining} comma-separated noun phrases (1-3 words, letters/spaces/hyphens).",
                            "No commentary.",
                        ]
                    )
                prompt_text = "\n".join(prompt_lines)
                text = _generate_prompt_text(
                    prompt_text,
                    max_new_tokens=max_new_tokens,
                    reasoning=reasoning,
                )
                last_text = text
                if not text:
                    _log(f"GPT-OSS brainstorm (class={class_name}, round {round_idx + 1}, attempt {attempt + 1}) returned empty/invalid")
                    continue
                parsed = _parse_prompt_candidates(text, seen, remaining)
                if parsed and len(parsed) > remaining:
                    parsed = parsed[:remaining]
                _log(
                    f"GPT-OSS brainstorm (class={class_name}, round {round_idx + 1}, attempt {attempt + 1}"
                    "): "
                    f"{', '.join(parsed) if parsed else text}"
                )
                if parsed:
                    return parsed
                # If the only issue is duplication, don't treat it as a hard failure.
                dup_check = _parse_prompt_candidates(text, set(), remaining)
                if dup_check:
                    _log(
                        f"GPT-OSS brainstorm (class={class_name}, round {round_idx + 1}, attempt {attempt + 1}) "
                        "produced only duplicates; keeping existing list."
                    )
                    return []
                _log(f"GPT-OSS brainstorm (class={class_name}, round {round_idx + 1}, attempt {attempt + 1}) yielded no valid candidates")
            _log(f"GPT-OSS brainstorm (class={class_name}, round {round_idx + 1}) failed after 3 attempts")
            return []

        for round_idx in range(3):  # allow a few brainstorming rounds
            if len(suggestions) >= max_new:
                break
            remaining = max_new - len(suggestions)
            parsed = _run_brainstorm_with_retries(remaining, round_idx)
            if not parsed:
                continue
            suggestions.extend(parsed)
            if len(suggestions) >= max_new:
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning("Prompt recipe: GPT-OSS expansion failed for %s: %s", class_name, exc)
        suggestions = []
    # Final sanitize + dedupe. We intentionally avoid any Qwen-based refinement here so this
    # helper stays lightweight and doesn't contend with SAM3 for GPU memory.
    reviewed = _sanitize_prompts([*cleaned_base, *suggestions])
    reviewed_lower = {p.lower() for p in cleaned_base}
    final_new: List[str] = []
    for p in reviewed:
        low = p.lower()
        if low in reviewed_lower:
            continue
        reviewed_lower.add(low)
        final_new.append(p)
        if len(final_new) >= max_new:
            break
    # Fallback: if nothing survived, return the cleaned base prompts only.
    if not final_new:
        if log_fn:
            try:
                log_fn(f"GPT-OSS prompts fell back to base for {class_name}")
            except Exception:
                pass
        return cleaned_base
    return final_new


def _sanitize_prompts(prompts: List[str]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for p in prompts:
        if not isinstance(p, str):
            continue
        val = p.strip()
        if not val:
            continue
        words = val.split()
        if not (1 <= len(words) <= 4):
            continue
        if any(len(w) <= 1 for w in words):
            continue
        key = val.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(val)
    return cleaned


def _refine_prompts_with_qwen(prompts: List[str]) -> List[str]:
    prompts = _sanitize_prompts(prompts)
    if not prompts or Qwen2_5_VLForConditionalGeneration is None or QWEN_IMPORT_ERROR:
        return prompts
    try:
        prompt_text = (
            "You are validating candidate noun phrases for open-vocabulary object detection. "
            "Keep only entries that are concrete object-like noun phrases (1-4 words, nouns included). "
            "Reject fragments, verbs, partial words, or unrelated terms. "
            "Respond ONLY as a comma-separated list, no numbering, no explanations, ending with STOP.\n"
            f"Candidates: {', '.join(prompts)}"
        )
        text = _generate_prompt_text(prompt_text, max_new_tokens=160)
        if not text:
            return prompts
        parts = [t.strip() for t in re.split(r"[,\\n]+", text) if t.strip() and t.strip().upper() != "STOP"]
        cleaned = _sanitize_prompts(parts)
        return cleaned or prompts
    except Exception:
        return prompts


def _qwen_self_filter_prompts(class_name: str, prompts: List[str]) -> List[str]:
    """Ask Qwen to self-critique the candidate prompts against the target class and return only credible entries."""
    prompts = _sanitize_prompts(prompts)
    if not prompts or Qwen2_5_VLForConditionalGeneration is None or QWEN_IMPORT_ERROR:
        return prompts
    try:
        prompt_text = (
            "You are double-checking candidate noun phrases for object detection. "
            f"Target class: '{_humanize_class_name(class_name)}'. "
            "From the list, keep ONLY phrases that clearly describe that class (synonyms or sub-types). "
            "Drop anything ambiguous, misspelled, or unrelated. "
            "Return ONLY a comma-separated list, no explanations, ending with STOP.\n"
            f"Candidates: {', '.join(prompts)}"
        )
        text = _generate_prompt_text(prompt_text, max_new_tokens=160)
        if not text:
            return prompts
        parts = [t.strip() for t in re.split(r"[,\\n]+", text) if t.strip() and t.strip().upper() != "STOP"]
        cleaned = _sanitize_prompts(parts)
        return cleaned or prompts
    except Exception:
        return prompts


def _normalize_recipe_thresholds(thresholds: Optional[List[float]], fallback: float, limit: int = 20) -> List[float]:
    values = thresholds if thresholds is not None else [fallback]
    cleaned: List[float] = []
    seen = set()
    for raw in values:
        try:
            val = float(raw)
        except Exception:
            continue
        if val < 0.0 or val > 1.0:
            continue
        key = round(val, 4)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(val)
        if len(cleaned) >= limit:
            break
    return cleaned


def _expand_midpoints(values: List[float], *, fine_step: float = 0.05, clamp: Tuple[float, float] = (0.0, 1.0), limit: int = 20) -> List[float]:
    """Given a sorted list, add midpoints and small +/- offsets for coarse-to-fine sweeps."""
    if not values:
        return values
    lo, hi = clamp
    base = sorted({v for v in values if lo <= v <= hi})
    extras: List[float] = []
    for a, b in zip(base, base[1:]):
        mid = (a + b) / 2.0
        extras.append(mid)
    if fine_step > 0:
        for v in base:
            extras.append(v + fine_step)
            extras.append(v - fine_step)
    merged = sorted({v for v in [*base, *extras] if lo <= v <= hi})
    if len(merged) > limit:
        merged = merged[:limit]
    return merged


def _build_gt_index_for_class(
    gt_by_image_cat: Dict[int, Dict[int, List[List[float]]]], target_class: int
) -> Tuple[Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]], set[str], Dict[int, int]]:
    gt_index: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]] = {}
    all_keys: set[str] = set()
    per_image_counts: Dict[int, int] = {}
    for img_id, by_cat in gt_by_image_cat.items():
        boxes = by_cat.get(target_class)
        if not boxes:
            continue
        entries: List[Tuple[str, Tuple[float, float, float, float]]] = []
        for idx, bbox in enumerate(boxes):
            key = f"{img_id}:{idx}"
            entries.append((key, _xywh_to_xyxy(bbox)))
            all_keys.add(key)
        gt_index[img_id] = entries
        per_image_counts[img_id] = len(entries)
    return gt_index, all_keys, per_image_counts


def _evaluate_prompt_candidate(
    prompt: str,
    threshold: float,
    *,
    cat_id: int,
    image_ids: List[int],
    gt_index: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]],
    other_gt_index: Optional[Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]]] = None,
    images: Dict[int, Dict[str, Any]],
    iou_threshold: float,
    max_dets: int,
    image_cache: Dict[int, Image.Image],
    cached_detections: Optional[Dict[int, List[Tuple[float, float, float, float, Optional[float]]]]] = None,
) -> Dict[str, Any]:
    total_gt = sum(len(gt_index.get(img_id, [])) for img_id in image_ids)
    total_preds = 0
    conflicts = 0
    matches = 0
    fps = 0
    det_images = 0
    iou_sum = 0.0
    score_sum = 0.0
    matched_scores = 0
    matched_gt_keys: set[str] = set()
    matches_by_image: Dict[int, Dict[str, Any]] = {}
    for img_id in image_ids:
        info = images.get(img_id)
        if not info:
            continue
        path = info.get("path")
        width = info.get("width")
        height = info.get("height")
        if not path or width is None or height is None:
            continue
        gts = gt_index.get(img_id, [])
        gt_used = [False] * len(gts)
        pred_boxes: List[Tuple[float, float, float, float, Optional[float]]] = []
        if cached_detections is not None:
            pred_boxes = cached_detections.get(img_id, [])
        if not pred_boxes:
            try:
                pil_img = image_cache[img_id]
            except KeyError:
                try:
                    pil_img = Image.open(path).convert("RGB")
                except Exception:
                    continue
                image_cache[img_id] = pil_img
            preds = _run_sam3_text_inference(
                pil_img,
                prompt,
                threshold=threshold,
                mask_threshold=0.0,
                limit=max_dets,
            )
            for det in preds:
                try:
                    x1, y1, x2, y2 = _yolo_to_xyxy(pil_img.width, pil_img.height, det.bbox)
                    pred_boxes.append((x1, y1, x2, y2, det.score))
                except Exception:
                    continue
        if not pred_boxes:
            continue
        filtered = [b for b in pred_boxes if (b[4] if b[4] is not None else 0.0) >= threshold]
        filtered.sort(key=lambda b: (b[4] if b[4] is not None else 0.0), reverse=True)
        pred_boxes = filtered[:max_dets] if max_dets else filtered
        total_preds += len(pred_boxes)
        matched_in_image = 0
        fp_in_image = 0
        matched_keys: List[str] = []
        for x1, y1, x2, y2, score in pred_boxes:
            if other_gt_index:
                other_hits = other_gt_index.get(img_id, [])
                conflict_found = False
                for _, other_box in other_hits:
                    if _iou((x1, y1, x2, y2), other_box) >= iou_threshold:
                        conflicts += 1
                        conflict_found = True
                        break
                if conflict_found:
                    continue
            total_preds += 1
            best_iou = 0.0
            best_idx = -1
            for idx, (_, gt_box) in enumerate(gts):
                if gt_used[idx]:
                    continue
                iou = _iou((x1, y1, x2, y2), gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold and best_idx >= 0:
                gt_used[best_idx] = True
                matches += 1
                matched_in_image += 1
                matched_key = gts[best_idx][0]
                matched_keys.append(matched_key)
                matched_gt_keys.add(matched_key)
                iou_sum += best_iou
                if score is not None:
                    score_sum += score
                    matched_scores += 1
            else:
                fp_in_image += 1
        fps += fp_in_image
        if matched_in_image > 0:
            det_images += 1
        if matched_keys or fp_in_image:
            matches_by_image[img_id] = {"matched": matched_keys, "fps": fp_in_image}
    precision = matches / total_preds if total_preds else 0.0
    recall = matches / total_gt if total_gt else 0.0
    det_rate = det_images / len(image_ids) if image_ids else 0.0
    avg_iou = iou_sum / matches if matches else None
    avg_score = score_sum / matched_scores if matched_scores else None
    f1 = (2 * precision * recall) / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
    overall_score = f1 * (0.5 + 0.5 * det_rate)
    return {
        "prompt": prompt,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "det_rate": det_rate,
        "avg_iou": avg_iou,
        "avg_score": avg_score,
        "score": overall_score,
        "f1": f1,
        "preds": total_preds,
        "matches": matches,
        "gts": total_gt,
        "fps": fps,
        "conflicts": conflicts,
        "det_images": det_images,
        "matched_gt_keys": matched_gt_keys,
        "matches_by_image": matches_by_image,
    }


def _collect_prompt_detections(
    prompt: str,
    min_threshold: float,
    *,
    image_ids: List[int],
    images: Dict[int, Dict[str, Any]],
    image_cache: Dict[int, Image.Image],
    max_dets: int,
) -> Dict[int, List[Tuple[float, float, float, float, Optional[float]]]]:
    results: Dict[int, List[Tuple[float, float, float, float, Optional[float]]]] = {}
    for img_id in image_ids:
        info = images.get(img_id)
        if not info:
            continue
        path = info.get("path")
        width = info.get("width")
        height = info.get("height")
        if not path or width is None or height is None:
            continue
        try:
            pil_img = image_cache[img_id]
        except KeyError:
            try:
                pil_img = Image.open(path).convert("RGB")
            except Exception:
                continue
            image_cache[img_id] = pil_img
        preds = _run_sam3_text_inference(
            pil_img,
            prompt,
            threshold=min_threshold,
            mask_threshold=0.0,
            limit=max_dets,
        )
        boxes: List[Tuple[float, float, float, float, Optional[float]]] = []
        for det in preds:
            try:
                x1, y1, x2, y2 = _yolo_to_xyxy(pil_img.width, pil_img.height, det.bbox)
                boxes.append((x1, y1, x2, y2, det.score))
            except Exception:
                continue
        if boxes:
            results[img_id] = boxes
    return results


def _build_prompt_recipe(
    candidates: List[Dict[str, Any]],
    all_gt_keys: set[str],
    per_image_gt: Dict[int, int],
    images: Dict[int, Dict[str, Any]],
    image_ids: List[int],
    gt_index: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # Pick the best threshold per prompt (highest matched GTs, then lowest FPs, then higher precision).
    best_by_prompt: Dict[str, Dict[str, Any]] = {}
    for cand in candidates:
        prompt = cand.get("prompt") or ""
        matched_count = len(cand.get("matched_gt_keys") or [])
        fps = cand.get("fps", 0)
        precision = cand.get("precision", 0.0)
        current = best_by_prompt.get(prompt)
        if current is None:
            best_by_prompt[prompt] = cand
            continue
        curr_matched = len(current.get("matched_gt_keys") or [])
        curr_fps = current.get("fps", 0)
        curr_precision = current.get("precision", 0.0)
        better = (matched_count, -fps, precision) > (curr_matched, -curr_fps, curr_precision)
        if better:
            best_by_prompt[prompt] = cand
    ordered_candidates = list(best_by_prompt.values())

    # Simulate per-image early stop: run prompts only on images with uncovered GTs; negatives always contribute FPs.
    remaining_by_image: Dict[int, set[str]] = {
        img_id: {key for key, _ in entries} for img_id, entries in gt_index.items() if entries
    }
    remaining_total = sum(len(v) for v in remaining_by_image.values())
    active_images = {img_id for img_id, keys in remaining_by_image.items() if keys}
    negative_images = {img_id for img_id in image_ids if per_image_gt.get(img_id, 0) == 0}
    steps: List[Dict[str, Any]] = []
    total_fps = 0
    total_duplicates = 0
    used_prompt_keys: set[Tuple[str, float]] = set()
    while remaining_total > 0 and ordered_candidates:
        best = None
        best_score = (-1, -1, -1, -1)
        for cand in ordered_candidates:
            prompt_key = (cand.get("prompt"), cand.get("threshold"))
            if prompt_key in used_prompt_keys:
                continue
            matches_by_image = cand.get("matches_by_image") or {}
            step_gain = 0
            step_fps = 0
            step_duplicates = 0
            step_matches_total = 0
            step_hits_by_image: Dict[int, Dict[str, Any]] = {}
            for img_id in negative_images:
                img_hits = matches_by_image.get(img_id)
                if not img_hits:
                    continue
                matched_total = len(img_hits.get("matched") or [])
                fps_count = max(0, img_hits.get("fps", 0))
                step_matches_total += matched_total
                step_fps += fps_count
                if matched_total or fps_count:
                    step_hits_by_image[img_id] = {
                        "matched": [],
                        "matched_total": matched_total,
                        "fps": fps_count,
                    }
            for img_id in active_images:
                img_hits = matches_by_image.get(img_id)
                if not img_hits:
                    continue
                matched_list = img_hits.get("matched") or []
                matched_set = set(matched_list)
                unmatched = remaining_by_image.get(img_id, set())
                new_hits = matched_set & unmatched
                matched_total = len(matched_set)
                fps_count = max(0, img_hits.get("fps", 0))
                step_gain += len(new_hits)
                step_duplicates += max(0, matched_total - len(new_hits))
                step_matches_total += matched_total
                step_fps += fps_count
                step_hits_by_image[img_id] = {
                    "matched": list(new_hits),
                    "matched_total": matched_total,
                    "fps": fps_count,
                }
            if step_gain <= 0:
                continue
            zero_fp = step_fps == 0
            score_tuple = (
                1 if zero_fp else 0,
                step_gain,
                -step_fps,
                cand.get("precision", 0.0),
            )
            if score_tuple > best_score:
                best = (cand, step_hits_by_image, step_gain, step_fps, step_duplicates, step_matches_total)
                best_score = score_tuple
        if not best:
            break
        cand, step_hits_by_image, gain, step_fps, duplicate_hits, step_matches_total = best
        prompt_key = (cand.get("prompt"), cand.get("threshold"))
        used_prompt_keys.add(prompt_key)
        for img_id, hit_info in step_hits_by_image.items():
            new_hits = set(hit_info.get("matched") or [])
            if not new_hits:
                continue
            current_unmatched = remaining_by_image.get(img_id)
            if current_unmatched is None:
                continue
            remaining_by_image[img_id] = current_unmatched - new_hits
        active_images = {img_id for img_id, keys in remaining_by_image.items() if keys}
        remaining_total = max(0, remaining_total - gain)
        total_duplicates += duplicate_hits
        total_fps += max(0, step_fps)
        covered_after = len(all_gt_keys) - remaining_total
        cum_coverage = covered_after / len(all_gt_keys) if all_gt_keys else 0.0
        preds_in_step = step_matches_total + step_fps
        seq_precision = step_matches_total / preds_in_step if preds_in_step else 0.0
        steps.append(
            {
                "prompt": cand.get("prompt"),
                "threshold": cand.get("threshold"),
                "gain": gain,
                "matches": step_matches_total,
                "fps": step_fps,
                "precision": seq_precision,
                "recall": cand.get("recall"),
                "det_rate": cand.get("det_rate"),
                "avg_iou": cand.get("avg_iou"),
                "avg_score": cand.get("avg_score"),
                "duplicates": duplicate_hits,
                "covered_after": covered_after,
                "cum_coverage": cum_coverage,
                "cum_fps": total_fps,
                "_matches_by_image": step_hits_by_image,
                "_prompt_precision": cand.get("precision"),
                "similarity_score": cand.get("similarity_score"),
            }
        )
    coverage_rate = (len(all_gt_keys) - remaining_total) / len(all_gt_keys) if all_gt_keys else 0.0
    recipe = {
        "steps": [
            {
                **{k: v for k, v in step.items() if not k.startswith("_")},
                "coverage_after": (step.get("covered_after", 0) / len(all_gt_keys)) if all_gt_keys else 0.0,
            }
            for step in steps
        ],
        "summary": {
            "total_gt": len(all_gt_keys),
            "covered": len(all_gt_keys) - remaining_total,
            "coverage_rate": coverage_rate,
            "fps": total_fps,
            "duplicates": total_duplicates,
        },
    }
    coverage_by_image: List[Dict[str, Any]] = []
    coverage_map: Dict[int, Dict[str, Any]] = {}
    for img_id in image_ids:
        info = images.get(img_id, {})
        remaining_keys = remaining_by_image.get(img_id, set())
        is_positive = per_image_gt.get(img_id, 0) > 0
        entry = {
            "image_id": img_id,
            "file_name": info.get("file_name"),
            "gt": per_image_gt.get(img_id, 0),
            "hits": [],
            "type": "pos" if is_positive else "neg",
            "covered": per_image_gt.get(img_id, 0) == 0 or len(remaining_keys) == 0,
        }
        coverage_map[img_id] = entry
        coverage_by_image.append(entry)
    for idx, step in enumerate(steps):
        matches_by_image = step.get("_matches_by_image") or {}
        for img_id, img_info in matches_by_image.items():
            target = coverage_map.get(img_id)
            if not target:
                continue
            matched_list = img_info.get("matched") or []
            fp_count = img_info.get("fps", 0)
            target["hits"].append(
                {
                    "step": idx,
                    "prompt": step.get("prompt"),
                    "threshold": step.get("threshold"),
                    "matched": len(matched_list),
                    "fps": fp_count,
                }
            )
    return recipe, coverage_by_image


def _serialize_prompt_helper_job(job: PromptHelperJob) -> Dict[str, Any]:
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


def _serialize_agent_mining_job(job: AgentMiningJob) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "message": job.message,
        "progress": job.progress,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "request": job.request,
        "result": job.result,
        "logs": job.logs,
        "error": job.error,
    }


def _run_prompt_helper_job(job: PromptHelperJob, payload: PromptHelperRequest) -> None:
    with PROMPT_HELPER_JOBS_LOCK:
        PROMPT_HELPER_JOBS[job.job_id] = job
    job.status = "running"
    job.message = "Loading dataset…"
    job.request = payload.dict()
    job.updated_at = time.time()
    try:
        dataset_root = _resolve_sam3_or_qwen_dataset(payload.dataset_id)
        coco, gt_by_image_cat, images = _load_coco_index(dataset_root)
        categories = coco.get("categories") or []
        cat_to_images: Dict[int, set[int]] = {}
        for ann in coco.get("annotations", []):
            try:
                cat_id = int(ann["category_id"])
                img_id = int(ann["image_id"])
            except Exception:
                continue
            cat_to_images.setdefault(cat_id, set()).add(img_id)
        prompts_map: Dict[int, List[str]] = {}
        if payload.prompts_by_class:
            for k, vals in payload.prompts_by_class.items():
                try:
                    cid = int(k)
                except Exception:
                    continue
                cleaned = [v.strip() for v in vals if isinstance(v, str) and v.strip()]
                if cleaned:
                    prompts_map[cid] = cleaned
        results: List[Dict[str, Any]] = []
        total_classes = len(categories) or 1
        image_cache: Dict[int, Image.Image] = {}
        # Precompute total steps for progress: each prompt * each sampled image.
        total_steps = 0
        for idx, cat in enumerate(categories):
            cat_id = int(cat.get("id", idx))
            prompts = prompts_map.get(cat_id)
            if not prompts:
                prompts = _generate_prompt_variants_for_class(
                    str(cat.get("name", f"class_{cat_id}")),
                    payload.max_synonyms,
                    payload.use_qwen,
                )
            sample_ids = _sample_images_for_category(
                cat_id,
                list(cat_to_images.get(cat_id, set())),
                payload.sample_per_class,
                payload.seed,
            )
            total_steps += len(prompts) * max(1, len(sample_ids))
        job.total_steps = total_steps
        job.completed_steps = 0
        for idx, cat in enumerate(categories):
            cat_id = int(cat.get("id", idx))
            class_name = str(cat.get("name", f"class_{cat_id}"))
            job.message = f"Evaluating {class_name} ({idx + 1}/{total_classes})…"
            job.progress = (idx) / total_classes
            job.updated_at = time.time()
            candidates = prompts_map.get(cat_id)
            if not candidates:
                candidates = _generate_prompt_variants_for_class(
                    class_name,
                    payload.max_synonyms,
                    payload.use_qwen,
                )
            sampled_images = _sample_images_for_category(
                cat_id,
                list(cat_to_images.get(cat_id, set())),
                payload.sample_per_class,
                payload.seed,
            )
            candidate_results: List[Dict[str, Any]] = []
            for prompt in candidates:
                step_label = f"{class_name}: '{prompt}'"
                try:
                    job.logs.append({"ts": time.time(), "msg": f"Running {step_label} on {len(sampled_images)} images"})
                    if len(job.logs) > MAX_JOB_LOGS:
                        job.logs[:] = job.logs[-MAX_JOB_LOGS:]
                except Exception:
                    pass
                metrics = _evaluate_prompt_for_class(
                    prompt,
                    cat_id=cat_id,
                    image_ids=sampled_images,
                    gt_by_image_cat=gt_by_image_cat,
                    images=images,
                    score_threshold=payload.score_threshold,
                    max_dets=payload.max_dets,
                    iou_threshold=payload.iou_threshold,
                    image_cache=image_cache,
                )
                candidate_results.append(metrics)
                job.completed_steps += max(1, len(sampled_images))
                if job.total_steps:
                    job.progress = min(1.0, job.completed_steps / job.total_steps)
                job.updated_at = time.time()
            candidate_results.sort(key=lambda m: (m.get("score", 0.0), m.get("recall", 0.0), m.get("precision", 0.0)), reverse=True)
            results.append(
                {
                    "class_id": cat_id,
                    "class_name": class_name,
                    "images_sampled": len(sampled_images),
                    "candidates": candidate_results,
                }
            )
            job.progress = (idx + 1) / total_classes
            job.updated_at = time.time()
        job.status = "completed"
        job.message = "Done"
        job.result = {
            "classes": results,
            "config": payload.dict(),
            "dataset_id": payload.dataset_id,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prompt helper job %s failed", job.job_id)
        job.status = "failed"
        job.error = str(exc)
        job.message = "Failed"
    finally:
        job.updated_at = time.time()


def _run_prompt_helper_search_job(job: PromptHelperJob, payload: PromptHelperSearchRequest) -> None:
    with PROMPT_HELPER_JOBS_LOCK:
        PROMPT_HELPER_JOBS[job.job_id] = job
    job.status = "running"
    job.message = "Loading dataset…"
    job.request = {"mode": "search", **payload.dict()}
    job.updated_at = time.time()
    try:
        dataset_root = _resolve_sam3_or_qwen_dataset(payload.dataset_id)
        coco, gt_by_image_cat, images = _load_coco_index(dataset_root)
        categories = coco.get("categories") or []
        target_class_id = payload.class_id
        if not payload.prompts_by_class:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="search_prompts_required")
        cat_to_images: Dict[int, set[int]] = {}
        for ann in coco.get("annotations", []):
            try:
                cat_id = int(ann["category_id"])
                img_id = int(ann["image_id"])
            except Exception:
                continue
            cat_to_images.setdefault(cat_id, set()).add(img_id)
        prompts_map: Dict[int, List[str]] = {}
        for k, vals in payload.prompts_by_class.items():
            try:
                cid = int(k)
            except Exception:
                continue
            if not isinstance(vals, (list, tuple)):
                continue
            cleaned = [v.strip() for v in vals if isinstance(v, str) and v.strip()]
            if cleaned:
                prompts_map[cid] = cleaned
        if not prompts_map:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="search_prompts_empty")
        all_img_ids = list(images.keys())
        image_cache: Dict[int, Image.Image] = {}
        if target_class_id is not None:
            categories = [c for c in categories if int(c.get("id", categories.index(c))) == target_class_id]
            if not categories:
                raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="search_class_not_found")

        total_steps = 0
        for idx, cat in enumerate(categories):
            cat_id = int(cat.get("id", idx))
            prompts = prompts_map.get(cat_id)
            if not prompts:
                continue
            pos_ids = _sample_images_for_category(
                cat_id,
                list(cat_to_images.get(cat_id, set())),
                payload.sample_per_class,
                payload.seed,
            )
            neg_ids = _sample_negative_images(
                cat_id,
                all_img_ids,
                cat_to_images,
                payload.negatives_per_class,
                payload.seed,
            )
            eval_count = len(set(pos_ids + neg_ids)) or 1
            total_steps += len(prompts) * eval_count
        job.total_steps = total_steps
        job.completed_steps = 0
        results: List[Dict[str, Any]] = []
        total_classes = len(categories) or 1
        for idx, cat in enumerate(categories):
            cat_id = int(cat.get("id", idx))
            class_name = str(cat.get("name", f"class_{cat_id}"))
            prompts = prompts_map.get(cat_id)
            if not prompts:
                continue
            job.message = f"Searching prompts for {class_name} ({idx + 1}/{total_classes})…"
            job.progress = idx / total_classes
            pos_ids = _sample_images_for_category(
                cat_id,
                list(cat_to_images.get(cat_id, set())),
                payload.sample_per_class,
                payload.seed,
            )
            neg_ids = _sample_negative_images(
                cat_id,
                all_img_ids,
                cat_to_images,
                payload.negatives_per_class,
                payload.seed,
            )
            eval_ids = list(dict.fromkeys([*pos_ids, *neg_ids]))
            candidate_results: List[Dict[str, Any]] = []
            for prompt in prompts:
                try:
                    job.logs.append(
                        {
                            "ts": time.time(),
                            "msg": f"Eval '{prompt}' on {len(eval_ids)} imgs (+{len(pos_ids)} pos / {len(neg_ids)} neg)",
                        }
                    )
                    if len(job.logs) > MAX_JOB_LOGS:
                        job.logs[:] = job.logs[-MAX_JOB_LOGS:]
                except Exception:
                    pass
                metrics = _evaluate_prompt_for_class(
                    prompt,
                    cat_id=cat_id,
                    image_ids=eval_ids,
                    gt_by_image_cat=gt_by_image_cat,
                    images=images,
                    score_threshold=payload.score_threshold,
                    max_dets=payload.max_dets,
                    iou_threshold=payload.iou_threshold,
                    image_cache=image_cache,
                )
                penalty = 1.0
                if payload.precision_floor > 0:
                    penalty = min(1.0, metrics["precision"] / max(payload.precision_floor, 1e-6))
                metrics["precision_penalty"] = penalty
                metrics["search_score"] = metrics["recall"] * (0.5 + 0.5 * metrics["det_rate"]) * penalty
                metrics["images_evaluated"] = len(eval_ids)
                metrics["positive_images"] = len(pos_ids)
                metrics["negative_images"] = len(neg_ids)
                candidate_results.append(metrics)
                job.completed_steps += max(1, len(eval_ids))
                if job.total_steps:
                    job.progress = min(1.0, job.completed_steps / job.total_steps)
                job.updated_at = time.time()
            candidate_results.sort(
                key=lambda m: (
                    m.get("search_score", 0.0),
                    m.get("recall", 0.0),
                    m.get("precision", 0.0),
                ),
                reverse=True,
            )
            best = candidate_results[0] if candidate_results else None
            results.append(
                {
                    "class_id": cat_id,
                    "class_name": class_name,
                    "positive_images": len(pos_ids),
                    "negative_images": len(neg_ids),
                    "best": best,
                    "candidates": candidate_results,
                }
            )
            job.progress = (idx + 1) / total_classes
            job.updated_at = time.time()
        job.status = "completed"
        job.message = "Done"
        job.result = {
            "classes": results,
            "config": payload.dict(),
            "dataset_id": payload.dataset_id,
            "mode": "search",
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prompt search job %s failed", job.job_id)
        job.status = "failed"
        job.error = str(exc)
        job.message = "Failed"
    finally:
        job.updated_at = time.time()


def _run_prompt_recipe_job(job: PromptHelperJob, payload: PromptRecipeRequest) -> None:
    with PROMPT_HELPER_JOBS_LOCK:
        PROMPT_HELPER_JOBS[job.job_id] = job
    job.status = "running"
    job.message = "Loading dataset…"
    job.request = {"mode": "recipe", **payload.dict()}
    job.updated_at = time.time()
    try:
        if not payload.prompts:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="recipe_prompts_required")
        dataset_root = _resolve_sam3_or_qwen_dataset(payload.dataset_id)
        coco, gt_by_image_cat, images = _load_coco_index(dataset_root)
        categories = coco.get("categories") or []
        cat_entry = next((c for c in categories if int(c.get("id", categories.index(c))) == payload.class_id), None)
        if not cat_entry:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="recipe_class_not_found")
        class_name = str(cat_entry.get("name", f"class_{payload.class_id}"))
        cat_to_images: Dict[int, set[int]] = {}
        for ann in coco.get("annotations", []):
            try:
                cat_id = int(ann["category_id"])
                img_id = int(ann["image_id"])
            except Exception:
                continue
            cat_to_images.setdefault(cat_id, set()).add(img_id)
        pos_ids = _sample_images_for_category(
            payload.class_id,
            list(cat_to_images.get(payload.class_id, set())),
            payload.sample_size,
            payload.seed,
        )
        all_img_ids = list(images.keys())
        neg_ids = _sample_negative_images(
            payload.class_id,
            all_img_ids,
            cat_to_images,
            payload.negatives,
            payload.seed,
        )
        eval_ids = list(dict.fromkeys([*pos_ids, *neg_ids]))
        image_cache: Dict[int, Image.Image] = {}
        gt_index_all, all_gt_keys_all, per_image_gt_all = _build_gt_index_for_class(gt_by_image_cat, payload.class_id)
        gt_index = {img_id: entries for img_id, entries in gt_index_all.items() if img_id in eval_ids}
        per_image_gt = {img_id: per_image_gt_all.get(img_id, 0) for img_id in eval_ids}
        all_gt_keys = set()
        for entries in gt_index.values():
            for key, _ in entries:
                all_gt_keys.add(key)
        if not eval_ids:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="recipe_no_images_sampled")
        if not all_gt_keys:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="recipe_no_gt_for_class")
        thresholds_cache: Dict[str, List[float]] = {}
        total_steps = 0
        for prompt_entry in payload.prompts:
            key = prompt_entry.prompt
            thresholds_cache[key] = _normalize_recipe_thresholds(
                prompt_entry.thresholds or payload.threshold_candidates,
                payload.score_threshold,
            )
            total_steps += len(thresholds_cache[key]) * max(1, len(eval_ids))
        job.total_steps = total_steps
        job.completed_steps = 0
        candidates: List[Dict[str, Any]] = []
        for idx, prompt_entry in enumerate(payload.prompts):
            thresholds = thresholds_cache.get(prompt_entry.prompt) or [payload.score_threshold]
            min_threshold = min(thresholds) if thresholds else payload.score_threshold
            detections = _collect_prompt_detections(
                prompt_entry.prompt,
                min_threshold,
                image_ids=eval_ids,
                images=images,
                image_cache=image_cache,
                max_dets=payload.max_dets,
            )
            for thr in thresholds:
                try:
                    job.logs.append(
                        {
                            "ts": time.time(),
                            "msg": f"Eval prompt {idx + 1}/{len(payload.prompts)} @ {thr:.2f} on {len(eval_ids)} images",
                        }
                    )
                    if len(job.logs) > MAX_JOB_LOGS:
                        job.logs[:] = job.logs[-MAX_JOB_LOGS:]
                except Exception:
                    pass
                metrics = _evaluate_prompt_candidate(
                    prompt_entry.prompt,
                    thr,
                    cat_id=payload.class_id,
                    image_ids=eval_ids,
                    gt_index=gt_index,
                    images=images,
                    iou_threshold=payload.iou_threshold,
                    max_dets=payload.max_dets,
                    image_cache=image_cache,
                    cached_detections=detections,
                )
                metrics["class_name"] = class_name
                metrics["class_id"] = payload.class_id
                metrics["image_count"] = len(eval_ids)
                candidates.append(metrics)
                job.completed_steps += len(eval_ids)
                if job.total_steps:
                    job.progress = min(1.0, job.completed_steps / job.total_steps)
                job.message = f"Evaluated {prompt_entry.prompt} ({job.completed_steps}/{job.total_steps} images)"
                job.updated_at = time.time()
        recipe, coverage_by_image = _build_prompt_recipe(
            candidates,
            all_gt_keys,
            per_image_gt,
            images,
            eval_ids,
            gt_index,
        )
        job.status = "completed"
        job.message = "Done"
        job.result = {
            "mode": "recipe",
            "dataset_id": payload.dataset_id,
            "class_id": payload.class_id,
            "class_name": class_name,
            "positive_images": len(pos_ids),
            "negative_images": len(neg_ids),
            "positive_image_ids": pos_ids,
            "negative_image_ids": neg_ids,
            "evaluated_image_ids": eval_ids,
            "gt_count": len(all_gt_keys),
            "config": payload.dict(),
            "candidates": [
                {
                    **{k: v for k, v in cand.items() if k not in {"matched_gt_keys", "matches_by_image"}},
                    "matched_gt": len(cand.get("matched_gt_keys") or []),
                }
                for cand in candidates
            ],
            "recipe": recipe,
            "coverage_by_image": coverage_by_image,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prompt recipe job %s failed", job.job_id)
        job.status = "failed"
        job.error = str(exc)
        job.message = "Failed"
    finally:
        job.updated_at = time.time()


def _run_agent_mining_job_legacy(job: AgentMiningJob, payload: AgentMiningRequest) -> None:
    with AGENT_MINING_JOBS_LOCK:
        AGENT_MINING_JOBS[job.job_id] = job
    job.status = "running"
    job.message = "Preparing dataset split…"
    job.request = payload.dict()
    job.updated_at = time.time()
    mining_pool: Optional[_Sam3MiningPool] = None
    try:
        # Clean up stale caches before starting heavy work.
        _enforce_agent_mining_cache_limits(AGENT_MINING_DET_CACHE_ROOT, allow_when_running=False)
        dataset_root = _resolve_sam3_or_qwen_dataset(payload.dataset_id)
        job.logs.append({"ts": time.time(), "msg": f"Dataset resolved at {dataset_root}"})
        if len(job.logs) > MAX_JOB_LOGS:
            job.logs[:] = job.logs[-MAX_JOB_LOGS:]
        # Log received test settings to diagnose splits.
        job.logs.append(
            {
                "ts": time.time(),
                "msg": (
                    f"Request: test_mode={payload.test_mode} "
                    f"train_limit={payload.test_train_limit} val_limit={payload.test_val_limit} "
                    f"val_percent={payload.val_percent:.3f} ({payload.val_percent * 100:.1f}%) "
                    f"split_seed={payload.split_seed}"
                ),
            }
        )
        if len(job.logs) > MAX_JOB_LOGS:
            job.logs[:] = job.logs[-MAX_JOB_LOGS:]
        split = _ensure_agent_mining_split(
            payload.dataset_id,
            dataset_root,
            val_percent=payload.val_percent,
            seed=payload.split_seed,
            reuse_split=payload.reuse_split,
            test_mode=payload.test_mode,
            train_limit=payload.test_train_limit if payload.test_mode else None,
            val_limit=payload.test_val_limit if payload.test_mode else None,
        )
        job.logs.append(
            {
                "ts": time.time(),
                "msg": (
                    f"Split built (cached={split.get('_cached', False)}): "
                    f"train={len(split.get('train', []))} val={len(split.get('val', []))} "
                    f"test_mode={payload.test_mode} seed={payload.split_seed} val%={payload.val_percent:.3f}"
                ),
            }
        )
        if len(job.logs) > MAX_JOB_LOGS:
            job.logs[:] = job.logs[-MAX_JOB_LOGS:]
        job.progress = 0.05
        job.updated_at = time.time()
        cache_context = "|".join(
            [
                str(split.get("dataset_signature") or split.get("signature") or ""),
                f"seed={payload.split_seed}",
                f"val={payload.val_percent:.6f}",
                f"test={payload.test_mode}",
                f"trcap={payload.test_train_limit if payload.test_mode else ''}",
                f"vacap={payload.test_val_limit if payload.test_mode else ''}",
            ]
        )
        job.logs.append(
            {
                "ts": time.time(),
                "msg": (
                    f"Split { 'reused from cache' if split.get('_cached') else 'built fresh' } "
                    f"(seed={payload.split_seed}, val%={payload.val_percent * 100:.1f}%, "
                    f"reuse_split={payload.reuse_split})"
                ),
            }
        )
        job.logs.append(
            {
                "ts": time.time(),
                "msg": f"Prepared split with {len(split.get('train') or [])} train / {len(split.get('val') or [])} val images",
            }
        )
        if payload.test_mode:
            job.logs.append(
                {
                    "ts": time.time(),
                    "msg": f"Test mode enabled (train_limit={payload.test_train_limit}, val_limit={payload.test_val_limit})",
                }
            )
        job.logs.append(
            {
                "ts": time.time(),
                "msg": (
                    f"Config: mode={payload.search_mode} thresholds={payload.thresholds} mask_thr={payload.mask_threshold} "
                    f"min_size={payload.min_size} simplify={payload.simplify_epsilon} max_results={payload.max_results} "
                    f"clip_guard={payload.use_clip_fp_guard} neg_exemplars={payload.use_negative_exemplars} "
                    f"neg_strength={payload.negative_strength} adaptive_thr={payload.adaptive_threshold_search} "
                    f"adaptive_sim={payload.adaptive_similarity_search} cross_cleanup={payload.cross_class_cleanup} "
                    f"beam(k={payload.beam_width}, rounds={payload.beam_rounds}, eps={payload.beam_min_improve}, "
                    f"cap={payload.beam_eval_cap}, reuse_cache={payload.reuse_cache})"
                ),
            }
        )
        if len(job.logs) > MAX_JOB_LOGS:
            job.logs[:] = job.logs[-MAX_JOB_LOGS:]
        if job.cancel_event.is_set():
            job.status = "cancelled"
            job.message = "Cancelled"
            job.updated_at = time.time()
            return

        coco, gt_by_image_cat, images = _load_coco_index(dataset_root)
        categories = coco.get("categories") or []
        cat_filter: Optional[set[int]] = None
        if payload.classes:
            try:
                cat_filter = {int(c) for c in payload.classes}
            except Exception:
                cat_filter = None
        selected_cats: List[Dict[str, Any]] = []
        train_ids = list(split.get("train") or [])
        val_ids = list(split.get("val") or [])
        train_id_set = set(train_ids)
        val_id_set = set(val_ids)
        job.logs.append({"ts": time.time(), "msg": f"Loaded COCO with {len(categories)} categories"})
        if len(job.logs) > MAX_JOB_LOGS:
            job.logs[:] = job.logs[-MAX_JOB_LOGS:]
        job.updated_at = time.time()

        cache_dir = _agent_mining_cache_dir(payload.dataset_id)
        thresholds = [t for t in (payload.thresholds or []) if isinstance(t, (int, float))]
        if not thresholds:
            thresholds = [0.3]
        thresholds = [float(max(0.0, min(1.0, t))) for t in thresholds]
        if payload.search_mode == "grid" and payload.adaptive_threshold_search:
            thresholds = _expand_midpoints(thresholds, fine_step=0.05, clamp=(0.0, 1.0), limit=24)
        sim_scores_global = [s for s in (payload.similarity_scores or []) if isinstance(s, (int, float))]
        if not sim_scores_global:
            sim_scores_global = [payload.similarity_score]
        sim_scores_global = [float(max(0.0, min(1.0, s))) for s in sim_scores_global]
        if payload.search_mode == "grid" and payload.adaptive_similarity_search:
            sim_scores_global = _expand_midpoints(sim_scores_global, fine_step=0.05, clamp=(0.0, 1.0), limit=24)

        # Precompute prompts (including GPT-OSS expansions) before loading SAM3.
        prepared_prompts: Dict[int, List[str]] = {}
        raw_hints = payload.class_hints or {}
        normalized_hints: Dict[int, str] = {}
        name_to_id: Dict[str, int] = {}
        name_collisions: List[str] = []
        for idx, cat in enumerate(categories):
            try:
                cid = int(cat.get("id", idx))
            except Exception:
                cid = idx
            key = str(cat.get("name", "")).strip().lower()
            if key in name_to_id and name_to_id[key] != cid:
                name_collisions.append(key)
            name_to_id[key] = cid
        if name_collisions:
            job.logs.append({"ts": time.time(), "msg": f"Duplicate class names detected (normalized): {', '.join(sorted(set(name_collisions)))}"})
            if len(job.logs) > MAX_JOB_LOGS:
                job.logs[:] = job.logs[-MAX_JOB_LOGS:]
        for key, note in raw_hints.items():
            if not note or not isinstance(note, str):
                continue
            note_clean = note.strip()
            if not note_clean:
                continue
            class_id_val: Optional[int] = None
            try:
                class_id_val = int(key)
            except Exception:
                class_id_val = None
            if class_id_val is None:
                name_key = str(key).strip().lower()
                class_id_val = name_to_id.get(name_key)
            if class_id_val is None:
                continue
            normalized_hints[class_id_val] = note_clean
        for idx, cat in enumerate(categories):
            try:
                cat_id = int(cat.get("id", idx))
            except Exception:
                continue
            if cat_filter and cat_id not in cat_filter:
                continue
            cat_name = str(cat.get("name", f"class_{cat_id}"))
            hint_used = normalized_hints.get(cat_id)
            prompts_for_cat = (payload.text_prompts_by_class or {}).get(cat_id)
            if not prompts_for_cat:
                prompts_for_cat = [cat_name]
                if payload.auto_mine_prompts and payload.qwen_max_prompts > 0:
                    def _log_qwen(msg: str) -> None:
                        job.logs.append({"ts": time.time(), "msg": msg})
                        if len(job.logs) > MAX_JOB_LOGS:
                            job.logs[:] = job.logs[-MAX_JOB_LOGS:]
                    extra_prompts = _expand_prompts_with_prompt_llm(
                        cat_name,
                        prompts_for_cat,
                        payload.qwen_max_prompts,
                        log_fn=_log_qwen,
                        max_new_tokens=payload.prompt_max_new_tokens,
                        reasoning=payload.prompt_reasoning,
                    )
                    if len(extra_prompts) > payload.qwen_max_prompts:
                        extra_prompts = extra_prompts[: payload.qwen_max_prompts]
                    extra_prompts = _refine_prompts_with_qwen(extra_prompts)
                    merged = []
                    seen_prompts = set()
                    for entry in [*prompts_for_cat, *extra_prompts]:
                        key = entry.lower().strip()
                        if key in seen_prompts:
                            continue
                        seen_prompts.add(key)
                        merged.append(entry)
                    prompts_for_cat = merged
            else:
                prompts_for_cat = _refine_prompts_with_qwen(prompts_for_cat)
            prepared_prompts[cat_id] = prompts_for_cat
            if hint_used:
                job.logs.append({"ts": time.time(), "msg": f"Class hint applied for {cat_name}: {hint_used}"})
                if len(job.logs) > MAX_JOB_LOGS:
                    job.logs[:] = job.logs[-MAX_JOB_LOGS:]

        try:
            _unload_prompt_llm_runtime()
            job.logs.append({"ts": time.time(), "msg": "Prompt LLM unloaded to free memory before SAM3 init"})
            if len(job.logs) > MAX_JOB_LOGS:
                job.logs[:] = job.logs[-MAX_JOB_LOGS:]
            job.progress = max(job.progress, 0.1)
            job.updated_at = time.time()
        except Exception:
            pass

        base_devices = _resolve_sam3_mining_devices()
        expanded_devices: List[torch.device] = []
        per_dev_cap = max(1, payload.max_workers_per_device or 1)
        for dev in base_devices:
            expanded_devices.extend([dev] * per_dev_cap)
        if payload.max_workers and payload.max_workers > 0:
            mining_devices = expanded_devices[: payload.max_workers]
        else:
            mining_devices = expanded_devices
        if not mining_devices:
            raise RuntimeError("sam3_mining_no_devices")
        try:
            mining_pool = _Sam3MiningPool(mining_devices)
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.error = str(exc)
            job.message = "SAM3 mining init failed"
            job.updated_at = time.time()
            return
        try:
            device_labels = ", ".join(str(w.device) for w in mining_pool.workers)
        except Exception:
            device_labels = str(mining_devices)
        used_requested = len(mining_pool.workers)
        requested = len(mining_devices)
        log_msg = f"Using {used_requested} SAM3 worker(s): {device_labels}"
        if payload.max_workers and requested > payload.max_workers:
            log_msg += f" (capped at {payload.max_workers} max_workers)"
        elif used_requested != requested:
            log_msg += f" (requested {requested}, some devices invalid or unavailable)"
        if payload.max_workers_per_device and payload.max_workers_per_device > 1:
            log_msg += f"; up to {payload.max_workers_per_device} worker(s) per device"
        job.logs.append({"ts": time.time(), "msg": log_msg})
        if len(job.logs) > MAX_JOB_LOGS:
            job.logs[:] = job.logs[-MAX_JOB_LOGS:]

        class_entries: List[Dict[str, Any]] = []
        global_candidates: List[Dict[str, Any]] = []
        for idx, cat in enumerate(categories):
            try:
                cat_id = int(cat.get("id", idx))
            except Exception:
                continue
            if cat_filter and cat_id not in cat_filter:
                continue
            cat_name = str(cat.get("name", f"class_{cat_id}"))
            job.message = f"Class {idx + 1}/{len(categories)}: {cat_name} (prep)"
            job.logs.append({"ts": time.time(), "msg": f"Preparing class {cat_name} ({idx + 1}/{len(categories)})"})
            if len(job.logs) > MAX_JOB_LOGS:
                job.logs[:] = job.logs[-MAX_JOB_LOGS:]
            train_gt = 0
            val_gt = 0
            for img_id, cat_map in gt_by_image_cat.items():
                bboxes = cat_map.get(cat_id)
                if not bboxes:
                    continue
                count = len(bboxes)
                if img_id in train_id_set:
                    train_gt += count
                elif img_id in val_id_set:
                    val_gt += count
            sample_cap = payload.exemplar_per_class
            exemplars, exemplar_embeddings, clip_exemplar_warnings, exemplar_stats = _sample_agent_mining_exemplars(
                cat_id,
                train_ids,
                gt_by_image_cat,
                images,
                limit=sample_cap,
                seed=payload.split_seed,
                candidate_mode=payload.exemplar_candidate_mode,
                candidate_value=payload.exemplar_candidate_value,
                use_clip_selection=True,
                cluster=payload.cluster_exemplars,
            )
            # If FP guard is enabled but embeddings were not produced (e.g., CLIP unavailable), try embedding selected exemplars.
            if payload.use_clip_fp_guard and exemplars and not exemplar_embeddings:
                exemplar_embeddings, clip_exemplar_warnings = _clip_embed_regions(
                    exemplars,
                    images,
                    max_regions=max(16, payload.exemplar_per_class * 2),
                )
            # Log exemplar selection stats.
            try:
                job.logs.append(
                    {
                        "ts": time.time(),
                        "msg": (
                            f"Exemplars for {cat_name}: {exemplar_stats.get('candidates', 0)} candidates "
                            f"-> pool {exemplar_stats.get('pool', 0)} ({payload.exemplar_candidate_mode}="
                            f"{payload.exemplar_candidate_value}), embedded {exemplar_stats.get('embedded', 0)}, "
                            f"selected {len(exemplars)}"
                        ),
                    }
                )
                if len(job.logs) > MAX_JOB_LOGS:
                    job.logs[:] = job.logs[-MAX_JOB_LOGS:]
            except Exception:
                pass
            if payload.use_clip_fp_guard:
                job.logs.append(
                    {
                        "ts": time.time(),
                        "msg": f"CLIP guard embedded {len(exemplar_embeddings)} exemplars for {cat_name}; warnings: {len(clip_exemplar_warnings)}",
                    }
                )
                if len(job.logs) > MAX_JOB_LOGS:
                    job.logs[:] = job.logs[-MAX_JOB_LOGS:]
            # Negative exemplars from other classes.
            negatives: List[Dict[str, Any]] = []
            negative_embeddings: Dict[str, np.ndarray] = {}
            negative_stats: Dict[str, Any] = {"candidates": 0, "pool": 0, "embedded": 0, "selected": 0}
            negative_warnings: List[str] = []
            if payload.use_negative_exemplars and payload.max_negatives_per_class > 0:
                neg_candidates: List[Dict[str, Any]] = []
                for img_id in train_ids:
                    cat_map = gt_by_image_cat.get(img_id, {})
                    for other_cat, boxes in cat_map.items():
                        if other_cat == cat_id:
                            continue
                        for bbox in boxes:
                            try:
                                x, y, w, h = map(float, bbox[:4])
                                area = max(0.0, w * h)
                            except Exception:
                                x = y = w = h = 0.0
                                area = 0.0
                            img_info = images.get(img_id, {})
                            embed_id = f"{img_id}:{x:.2f},{y:.2f},{w:.2f},{h:.2f}"
                            neg_candidates.append(
                                {
                                    "image_id": img_id,
                                    "file_name": img_info.get("file_name"),
                                    "path": str(img_info.get("path") or ""),
                                    "bbox": [x, y, w, h],
                                    "area": area,
                                    "embed_id": embed_id,
                                }
                            )
                negative_stats["candidates"] = len(neg_candidates)
                if neg_candidates:
                    if payload.exemplar_candidate_mode == "percent":
                        pct = max(1, min(100, payload.exemplar_candidate_value))
                        pool_cap = max(1, int(math.ceil(len(neg_candidates) * (pct / 100.0))))
                    else:
                        pool_cap = max(1, min(payload.exemplar_candidate_value, len(neg_candidates)))
                    rng = random.Random(payload.split_seed + cat_id + 999)
                    rng.shuffle(neg_candidates)
                    pool = neg_candidates[:pool_cap]
                    negative_stats["pool"] = len(pool)
                    negative_embeddings, negative_warnings = _clip_embed_regions(pool, images, max_regions=len(pool))
                    negative_stats["embedded"] = len(negative_embeddings)
                    negatives = _k_center_select(pool, negative_embeddings, payload.max_negatives_per_class)
                    if negative_embeddings and negatives:
                        sel_ids = {n.get("embed_id") for n in negatives if n.get("embed_id")}
                        negative_embeddings = {k: v for k, v in negative_embeddings.items() if k in sel_ids}
                    negative_stats["selected"] = len(negatives)
                    if not negatives and not negative_embeddings:
                        negative_warnings.append("negatives_empty")
                else:
                    negative_warnings.append("negatives_empty")
            if payload.use_negative_exemplars:
                try:
                    job.logs.append(
                        {
                            "ts": time.time(),
                            "msg": (
                                f"Negatives for {cat_name}: {negative_stats.get('candidates', 0)} candidates "
                                f"-> pool {negative_stats.get('pool', 0)}, embedded {negative_stats.get('embedded', 0)}, "
                                f"selected {negative_stats.get('selected', 0)}"
                            ),
                        }
                    )
                    if len(job.logs) > MAX_JOB_LOGS:
                        job.logs[:] = job.logs[-MAX_JOB_LOGS:]
                except Exception:
                    pass
            text_prompts = prepared_prompts.get(cat_id) or [cat_name]
            try:
                preview_prompts = ", ".join(text_prompts) if text_prompts else "(none)"
                job.logs.append(
                    {
                        "ts": time.time(),
                        "msg": f"Prompt list for {cat_name}: {preview_prompts}",
                    }
                )
                if len(job.logs) > MAX_JOB_LOGS:
                    job.logs[:] = job.logs[-MAX_JOB_LOGS:]
            except Exception:
                pass
            job.logs.append(
                {
                    "ts": time.time(),
                    "msg": f"Evaluating {len(text_prompts)} text prompt(s) x {len(thresholds)} thresholds and {len(exemplars)} exemplar(s) for {cat_name}",
                }
            )
            entry = {
                "id": cat_id,
                "name": cat_name,
                "train_gt": train_gt,
                "val_gt": val_gt,
                "exemplars": exemplars,
                "clip_warnings": clip_exemplar_warnings,
                "exemplar_embeddings": exemplar_embeddings,
                "negatives": negatives,
                "negative_embeddings": negative_embeddings,
                "negative_warnings": negative_warnings,
                "text_prompts": text_prompts,
            }
            class_entries.append(entry)
            selected_cats.append(
                {
                    "id": cat_id,
                    "name": cat_name,
                    "train_gt": train_gt,
                    "val_gt": val_gt,
                    "exemplars": exemplars,
                    "clip_warnings": clip_exemplar_warnings,
                }
            )
            class_candidates: List[Dict[str, Any]] = []
            for prompt_text in text_prompts:
                cid = f"class{cat_id}::text::{prompt_text}"
                cand = {"id": cid, "type": "text", "prompt": prompt_text, "class_id": cat_id}
                class_candidates.append(cand)
                global_candidates.append(cand)
            for ex_idx, ex in enumerate(exemplars):
                cid = f"class{cat_id}::visual::{ex_idx}"
                ex_vec = None
                try:
                    embed_id = ex.get("embed_id") if isinstance(ex, dict) else None
                    if embed_id:
                        ex_vec = exemplar_embeddings.get(embed_id)
                except Exception:
                    ex_vec = None
                cand = {
                    "id": cid,
                    "type": "visual",
                    "visual_ref": ex,
                    "class_id": cat_id,
                    # Used to seed visual prompting when evaluating/applying this step.
                    # Visual steps are always seeded from a text prompt on the *current* image.
                    "seed_prompt": cat_name,
                    "exemplar_vec": ex_vec,
                }
                class_candidates.append(cand)
                global_candidates.append(cand)
            entry["candidates"] = class_candidates
            if job.cancel_event.is_set():
                job.status = "cancelled"
                job.message = "Cancelled"
                job.updated_at = time.time()
                return

        total_images = len(val_ids)
        if total_images == 0 and train_ids:
            # Fallback: carve a slice out of train for validation (keep splits disjoint).
            fallback = max(1, min(len(train_ids), payload.test_val_limit if payload.test_mode else 50))
            val_ids = train_ids[:fallback]
            train_ids = train_ids[fallback:]
            train_id_set = set(train_ids)
            val_id_set = set(val_ids)
            total_images = len(val_ids)
            job.logs.append(
                {
                    "ts": time.time(),
                    "msg": f"Val split empty; moved {total_images} image(s) from train to val for scoring.",
                }
            )
        job.logs.append(
            {
                "ts": time.time(),
                "msg": (
                    f"Starting global image-first sweep: {total_images} val images, "
                    f"{len(global_candidates)} candidates x {len(thresholds)} thresholds"
                ),
            }
        )
        if len(job.logs) > MAX_JOB_LOGS:
            job.logs[:] = job.logs[-MAX_JOB_LOGS:]

        progress_every_global = max(1, total_images // 20) if total_images else 1

        def _global_progress(done: int) -> None:
            if job.cancel_event.is_set():
                return
            if done == 1 or done == total_images or done % progress_every_global == 0:
                try:
                    pct = (done / total_images) if total_images else 0.0
                    job.progress = min(1.0, 0.1 + 0.6 * pct)
                    job.logs.append(
                        {
                            "ts": time.time(),
                            "msg": f"Processed {done}/{total_images} val images (global) for all candidates",
                        }
                    )
                    if len(job.logs) > MAX_JOB_LOGS:
                        job.logs[:] = job.logs[-MAX_JOB_LOGS:]
                    job.updated_at = time.time()
                except Exception:
                    pass

        det_cache, det_stats_global = _collect_agent_mining_detections_image_first(
            candidates=global_candidates,
            thresholds=thresholds,
            # Detections are cached independent of CLIP similarity floors; floors are applied during scoring.
            similarity_scores=None,
            images=images,
            image_ids=val_ids,
            mask_threshold=payload.mask_threshold,
            min_size=payload.min_size,
            simplify=payload.simplify_epsilon,
            max_results=payload.max_results,
            cache_dir=cache_dir,
            pool=mining_pool,
            use_cache=payload.reuse_cache,
            cancel_event=job.cancel_event,
            progress_callback=_global_progress,
            cache_context=cache_context,
        )
        job.logs.append(
            {
                "ts": time.time(),
                "msg": (
                    f"SAM3 global mining ran image-first on {det_stats_global.get('images')} val images for "
                    f"{det_stats_global.get('candidates')} candidates x {det_stats_global.get('thresholds')} thresholds "
                    f"(cached {det_stats_global.get('cached_pairs')} pairs, executed {det_stats_global.get('executed_pairs')} "
                    f"with {det_stats_global.get('executed_pairs_with_dets')} yielding detections)"
                ),
            }
        )
        if len(job.logs) > MAX_JOB_LOGS:
            job.logs[:] = job.logs[-MAX_JOB_LOGS:]
        job.progress = max(job.progress, 0.7)

        for idx, entry in enumerate(class_entries):
            cat_id = entry["id"]
            cat_name = entry["name"]
            exemplars = entry["exemplars"]
            exemplar_embeddings = entry.get("exemplar_embeddings") or {}
            negative_embeddings = entry.get("negative_embeddings") or {}
            negatives = entry.get("negatives") or []
            text_prompts = entry["text_prompts"]
            class_candidates = entry["candidates"]
            job.message = f"Class {idx + 1}/{len(class_entries)}: {cat_name} (eval prompts)"
            gt_index_all, all_gt_keys_all, per_image_gt_all = _build_gt_index_for_class(gt_by_image_cat, cat_id)
            gt_index_val = {img_id: entries for img_id, entries in gt_index_all.items() if img_id in val_id_set}
            per_image_gt_val = {img_id: per_image_gt_all.get(img_id, 0) for img_id in val_ids}
            all_gt_keys_val: set[str] = set()
            for entries in gt_index_val.values():
                for key, _ in entries:
                    all_gt_keys_val.add(key)
            other_gt_index_val: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]] = {}
            if payload.cross_class_cleanup:
                for img_id, cat_map in gt_by_image_cat.items():
                    if img_id not in val_id_set:
                        continue
                    other_entries: List[Tuple[str, Tuple[float, float, float, float]]] = []
                    for other_cat, boxes in cat_map.items():
                        if other_cat == cat_id:
                            continue
                        for idx_box, bbox in enumerate(boxes or []):
                            try:
                                other_entries.append((f"{other_cat}:{idx_box}", _xywh_to_xyxy(bbox)))
                            except Exception:
                                continue
                    if other_entries:
                        other_gt_index_val[img_id] = other_entries
            # Optional FP-derived negatives (single-run) for this class, before filtering
            fp_negatives: List[Dict[str, Any]] = []
            fp_negative_embeddings: Dict[str, np.ndarray] = {}
            fp_negative_warnings: List[str] = []
            if payload.use_negative_exemplars and payload.use_fp_negatives and payload.max_fp_negatives > 0:
                fp_candidates: List[Tuple[float, Dict[str, Any]]] = []
                for cand in class_candidates:
                    for thr in thresholds:
                        task_cache = det_cache.get(cand.get("id"), {}) or {}
                        dets_by_sim = task_cache.get(thr, {}) if isinstance(task_cache, dict) else {}
                        if isinstance(dets_by_sim, dict):
                            det_list = next(iter(dets_by_sim.values()), []) or []
                        elif isinstance(dets_by_sim, list):
                            det_list = dets_by_sim
                        else:
                            det_list = []
                        for det in det_list:
                            try:
                                lbl = det.get("label")
                                if lbl is not None and int(lbl) == int(cat_id):
                                    continue
                            except Exception:
                                pass
                            score = float(det.get("score", 0.0)) if det.get("score") is not None else 0.0
                            fp_candidates.append((score, det))
                fp_candidates.sort(key=lambda t: t[0], reverse=True)
                fp_candidates = fp_candidates[: max(0, payload.max_fp_negatives * 3)]
                fp_pool: List[Dict[str, Any]] = []
                for _, det in fp_candidates:
                    bbox = det.get("bbox")
                    if not bbox or len(bbox) < 4:
                        continue
                    try:
                        x, y, w, h = map(float, bbox[:4])
                        area = max(0.0, w * h)
                    except Exception:
                        x = y = w = h = 0.0
                        area = 0.0
                    img_id = det.get("image_id")
                    img_info = images.get(int(img_id)) if img_id is not None else {}
                    embed_id = f"fp:{img_id}:{x:.2f},{y:.2f},{w:.2f},{h:.2f}"
                    fp_pool.append(
                        {
                            "image_id": img_id,
                            "file_name": img_info.get("file_name"),
                            "path": str(img_info.get("path") or ""),
                            "bbox": [x, y, w, h],
                            "area": area,
                            "embed_id": embed_id,
                        }
                    )
                if fp_pool:
                    fp_pool = fp_pool[: max(1, min(len(fp_pool), payload.max_fp_negatives * 3))]
                    fp_negative_embeddings, fp_negative_warnings = _clip_embed_regions(fp_pool, images, max_regions=len(fp_pool))
                    fp_negatives = _k_center_select(fp_pool, fp_negative_embeddings, payload.max_fp_negatives)
                    if fp_negative_embeddings and fp_negatives:
                        sel_ids = {n.get("embed_id") for n in fp_negatives if n.get("embed_id")}
                        fp_negative_embeddings = {k: v for k, v in fp_negative_embeddings.items() if k in sel_ids}
                    try:
                        job.logs.append(
                            {
                                "ts": time.time(),
                                "msg": (
                                    f"FP negatives for {cat_name}: pool {len(fp_pool)} embedded {len(fp_negative_embeddings)} "
                                    f"selected {len(fp_negatives)}"
                                ),
                            }
                        )
                        if len(job.logs) > MAX_JOB_LOGS:
                            job.logs[:] = job.logs[-MAX_JOB_LOGS:]
                    except Exception:
                        pass

            combined_neg_embeddings = {}
            if payload.use_negative_exemplars:
                combined_neg_embeddings.update(negative_embeddings)
                combined_neg_embeddings.update(fp_negative_embeddings)
            combined_negatives = negatives + fp_negatives if payload.use_negative_exemplars else []

            eval_candidates: List[Dict[str, Any]] = []
            text_prompt_count = len(text_prompts or [])
            thresholds_count = len(thresholds or [])
            visual_candidate_count = len(exemplars) * thresholds_count
            sim_scores = [float(s) for s in sim_scores_global if s is not None]
            if not sim_scores:
                sim_scores = [payload.similarity_score]

            def _fetch_dets(task_cache: Dict[float, Any], thr: float, sim: Optional[float]) -> List[Dict[str, Any]]:
                dets_by_sim = task_cache.get(thr, {})
                if isinstance(dets_by_sim, list):
                    dets_by_sim = {sim_scores[0] if sim_scores else None: dets_by_sim}
                dets = dets_by_sim.get(sim)
                if dets is None and dets_by_sim:
                    dets = next(iter(dets_by_sim.values()))
                if dets is None and task_cache:
                    try:
                        base_thr = min(task_cache.keys())
                        base_map = task_cache.get(base_thr, {})
                        if isinstance(base_map, list):
                            base_list = base_map
                        else:
                            base_list = (
                                base_map.get(sim)
                                or (next(iter(base_map.values())) if base_map else [])
                                or []
                            )
                        dets = [d for d in base_list if (d.get("score") or 0.0) >= thr]
                    except Exception:
                        dets = []
                if dets is None:
                    dets = []
                return dets

            def _score_combo(
                cand: Dict[str, Any],
                thr: float,
                sim: Optional[float],
                task_cache: Dict[float, Any],
            ) -> Optional[Dict[str, Any]]:
                dets = _fetch_dets(task_cache, thr, sim)
                fp_warnings: List[str] = []
                if payload.use_clip_fp_guard and exemplar_embeddings:
                    dets, fp_warnings = _clip_fp_filter_detections(
                        dets,
                        exemplar_embeddings=exemplar_embeddings,
                        negative_embeddings=combined_neg_embeddings if payload.use_negative_exemplars else None,
                        negative_strength=payload.negative_strength if payload.use_negative_exemplars else 0.0,
                        images=images,
                        similarity_floor=sim if sim is not None else 0.0,
                    )
                cache_map = _detections_to_eval_cache(dets, images)
                metrics = _evaluate_prompt_candidate(
                    cand.get("prompt") if cand.get("type") == "text" else f"exemplar_{cand.get('id', '').split('::')[-1]}",
                    thr,
                    cat_id=cat_id,
                    image_ids=val_ids,
                    gt_index=gt_index_val,
                    other_gt_index=other_gt_index_val if payload.cross_class_cleanup else None,
                    images=images,
                    iou_threshold=0.5,
                    max_dets=payload.max_results,
                    image_cache={},
                    cached_detections=cache_map,
                )
                metrics["type"] = cand.get("type")
                metrics["candidate_id"] = cand.get("id")
                metrics["similarity_score"] = sim
                if cand.get("type") == "visual":
                    metrics["exemplar"] = {
                        "image_id": cand.get("visual_ref", {}).get("image_id"),
                        "bbox": cand.get("visual_ref", {}).get("bbox"),
                        "file_name": cand.get("visual_ref", {}).get("file_name"),
                    }
                    if cand.get("seed_prompt"):
                        metrics["seed_prompt"] = cand.get("seed_prompt")
                metrics["detections"] = len(dets)
                if fp_warnings:
                    metrics["warnings"] = fp_warnings
                return metrics

            if payload.search_mode == "grid":
                for cand in class_candidates:
                    task_id = cand.get("id")
                    cand_type = cand.get("type")
                    label = cand.get("prompt") if cand_type == "text" else f"exemplar_{task_id.split('::')[-1]}"
                    task_cache = det_cache.get(task_id, {})
                    for thr in thresholds:
                        for sim in sim_scores:
                            metrics = _score_combo(cand, thr, sim, task_cache)
                            if metrics:
                                metrics["prompt"] = label
                                eval_candidates.append(metrics)
            else:  # beam
                beam_k = max(1, min(payload.beam_width, 16))
                max_rounds = max(1, min(payload.beam_rounds, 10))
                min_improve = max(0.0, min(payload.beam_min_improve, 1.0))
                eval_cap = max(beam_k, min(payload.beam_eval_cap, 200))
                delta_thr = 0.1
                delta_sim = 0.05

                def _clamp01(val: float) -> float:
                    return max(0.0, min(1.0, val))

                for cand in class_candidates:
                    task_cache = det_cache.get(cand.get("id"), {})
                    seeds = []
                    for thr in thresholds:
                        for sim in sim_scores:
                            seeds.append((_clamp01(float(thr)), _clamp01(float(sim))))
                    if not seeds:
                        seeds = [(0.3, sim_scores[0] if sim_scores else 0.25)]
                    evaluated: Dict[Tuple[float, float], Dict[str, Any]] = {}
                    best_score = -1.0
                    proposals = list(dict.fromkeys(seeds))  # preserve order, unique
                    rounds_done = 0
                    while proposals and rounds_done < max_rounds and len(evaluated) < eval_cap:
                        new_metrics: List[Dict[str, Any]] = []
                        for thr, sim in proposals:
                            if len(evaluated) >= eval_cap:
                                break
                            key = (round(thr, 4), round(sim, 4))
                            if key in evaluated:
                                continue
                            metric = _score_combo(cand, thr, sim, task_cache)
                            if metric:
                                metric["prompt"] = metric.get("prompt") or (cand.get("prompt") or "")
                                evaluated[key] = metric
                                new_metrics.append(metric)
                        if not evaluated:
                            break
                        scored = sorted(evaluated.values(), key=lambda m: m.get("score", 0.0), reverse=True)
                        if scored:
                            improvement = scored[0].get("score", 0.0) - best_score
                            best_score = scored[0].get("score", 0.0)
                        else:
                            improvement = 0.0
                        top = scored[:beam_k] if scored else []
                        if improvement < min_improve:
                            break
                        rounds_done += 1
                        delta_thr *= 0.5
                        delta_sim *= 0.5
                        next_props: List[Tuple[float, float]] = []
                        for m in top:
                            base_thr = float(m.get("threshold", thresholds[0] if thresholds else 0.3))
                            base_sim = m.get("similarity_score")
                            try:
                                base_sim = float(base_sim) if base_sim is not None else 0.0
                            except Exception:
                                base_sim = 0.0
                            for dt in (-delta_thr, 0.0, delta_thr):
                                for ds in (-delta_sim, 0.0, delta_sim):
                                    thr_new = _clamp01(base_thr + dt)
                                    sim_new = _clamp01(base_sim + ds)
                                    key_new = (round(thr_new, 4), round(sim_new, 4))
                                    if key_new in evaluated:
                                        continue
                                    # diversity: require change
                                    if abs(thr_new - base_thr) < 1e-4 and abs(sim_new - base_sim) < 1e-4:
                                        continue
                                    next_props.append((thr_new, sim_new))
                        proposals = list(dict.fromkeys(next_props))
                    if evaluated:
                        best = max(evaluated.values(), key=lambda m: m.get("score", 0.0))
                        eval_candidates.append(best)
                        try:
                            thr_best = best.get("threshold")
                            sim_best = best.get("similarity_score")
                            score_best = best.get("score")
                            thr_txt = f"{float(thr_best):.3f}" if thr_best is not None else "n/a"
                            sim_txt = f"{float(sim_best):.3f}" if sim_best is not None else "n/a"
                            score_txt = f"{float(score_best):.3f}" if score_best is not None else "n/a"
                            job.logs.append(
                                {
                                    "ts": time.time(),
                                    "msg": (
                                        f"Beam for {cand.get('id')}: evals={len(evaluated)} "
                                        f"best_thr={thr_txt} sim={sim_txt} score={score_txt}"
                                    ),
                                }
                            )
                            if len(job.logs) > MAX_JOB_LOGS:
                                job.logs[:] = job.logs[-MAX_JOB_LOGS:]
                        except Exception:
                            pass

            recipe: Optional[Dict[str, Any]] = None
            coverage_by_image: Optional[List[Dict[str, Any]]] = None
            if eval_candidates and all_gt_keys_val:
                recipe, coverage_by_image = _build_prompt_recipe(
                    eval_candidates,
                    all_gt_keys_val,
                    per_image_gt_val,
                    images,
                    val_ids,
                    gt_index_val,
                )
                if recipe:
                    recipe["use_negative_exemplars"] = bool(payload.use_negative_exemplars)
                    recipe["negative_strength"] = float(payload.negative_strength) if payload.use_negative_exemplars else 0.0
                    recipe["negatives"] = combined_negatives if payload.use_negative_exemplars else []
                    meta_map: Dict[Tuple[str, float, Optional[float]], Dict[str, Any]] = {}
                    for cand in eval_candidates:
                        try:
                            key = (cand.get("prompt"), float(cand.get("threshold")), cand.get("similarity_score"))
                        except Exception:
                            continue
                        meta_map[key] = {
                            "type": cand.get("type"),
                            "exemplar": cand.get("exemplar"),
                            "warnings": cand.get("warnings"),
                            "similarity_score": cand.get("similarity_score"),
                            "seed_prompt": cand.get("seed_prompt"),
                        }
                    for step in recipe.get("steps", []):
                        key = (step.get("prompt"), float(step.get("threshold", 0)), step.get("similarity_score"))
                        # Try fallback without sim if not found
                        meta = meta_map.get(key) or meta_map.get((step.get("prompt"), float(step.get("threshold", 0)), None))
                        if meta:
                            step.update({k: v for k, v in meta.items() if v is not None})
                            if meta.get("candidate_id"):
                                step["candidate_id"] = meta.get("candidate_id")
                    summary_block = recipe.get("summary") or {}
                    fp_total = summary_block.get("fps") or 0
                    best_steps = recipe.get("steps") or []
                    best_labels = []
                    for s in best_steps:
                        if s.get("type") == "visual" and s.get("exemplar"):
                            ex = s.get("exemplar") or {}
                            best_labels.append(f"exemplar img {ex.get('image_id')} thr {s.get('threshold')}")
                        else:
                            best_labels.append(f"{s.get('prompt')} thr {s.get('threshold')}")
                    explanation = (
                        f"Tested {text_prompt_count} text prompt(s) x {thresholds_count} thresholds and {len(exemplars)} exemplar(s) "
                        f"({len(eval_candidates)} total candidates). "
                    )
                    if best_labels:
                        explanation += f"Kept {len(best_steps)} step(s): {', '.join(best_labels)}."
                    else:
                        explanation += "No steps kept."
                    if summary_block:
                        covered = summary_block.get("covered")
                        total_gt = summary_block.get("total_gt")
                        cov_rate = summary_block.get("coverage_rate")
                        cov_txt = ""
                        if cov_rate is not None:
                            try:
                                cov_txt = f"{cov_rate * 100:.1f}%"
                            except Exception:
                                cov_txt = ""
                        explanation += f" Coverage {covered}/{total_gt} ({cov_txt}), FPs {fp_total}."
                    recipe["explanation"] = explanation
            selected_cats[idx]["candidates"] = eval_candidates
            selected_cats[idx]["recipe"] = recipe
            selected_cats[idx]["coverage_by_image"] = coverage_by_image
            selected_cats[idx]["meta"] = {
                "text_prompts": text_prompt_count,
                "thresholds": thresholds_count,
                "exemplars": len(exemplars),
                "visual_candidates": visual_candidate_count,
                "total_candidates": len(eval_candidates),
            }
            if not recipe or not recipe.get("steps"):
                selected_cats[idx]["no_recipe_reason"] = "no_candidate_gain"
            job.progress = 0.7 + 0.3 * ((idx + 1) / max(1, len(class_entries)))
            summary = (recipe or {}).get("summary") or {}
            covered = summary.get("covered")
            total_gt = summary.get("total_gt")
            coverage_rate = summary.get("coverage_rate")
            fps = summary.get("fps")
            cov_pct_str = ""
            if coverage_rate is not None:
                try:
                    cov_pct_str = f"{coverage_rate * 100:.1f}%"
                except Exception:
                    cov_pct_str = ""
            job.message = f"Class {idx + 1}/{len(class_entries)}: {cat_name} complete ({len(eval_candidates)} candidates)"
            job.logs.append(
                {
                    "ts": time.time(),
                    "msg": (
                        f"Class {cat_name} complete with {len(eval_candidates)} candidates; "
                        f"recipe steps: {len(recipe.get('steps', [])) if recipe else 0}; "
                        f"text_prompts={len(text_prompts or [])} exemplars={len(exemplars or [])}; "
                        f"coverage={covered}/{total_gt} ({cov_pct_str}) fps={fps}"
                    ),
                }
            )
            if len(job.logs) > MAX_JOB_LOGS:
                job.logs[:] = job.logs[-MAX_JOB_LOGS:]
            job.updated_at = time.time()

        job.progress = max(job.progress, 0.99)
        job.updated_at = time.time()
        job.logs.append({"ts": time.time(), "msg": f"Prepared {len(selected_cats)} classes with exemplar seeds"})
        if len(job.logs) > MAX_JOB_LOGS:
            job.logs[:] = job.logs[-MAX_JOB_LOGS:]
        if job.cancel_event.is_set():
            job.status = "cancelled"
            job.message = "Cancelled"
            job.updated_at = time.time()
            return
        job.result = {
            "dataset_id": payload.dataset_id,
            "split": split,
            "config": payload.dict(),
            "status": "completed",
            "classes": selected_cats,
            "note": "Agent mining completed with splits, exemplars, and baseline candidate evals.",
        }
        try:
            coverages = []
            for cls in selected_cats:
                summary = (cls.get("recipe") or {}).get("summary") or {}
                rate = summary.get("coverage_rate")
                if rate is not None:
                    coverages.append(rate)
            if coverages:
                avg_cov = sum(coverages) / len(coverages)
                job.logs.append(
                    {
                        "ts": time.time(),
                        "msg": f"Job summary: avg coverage {avg_cov * 100:.1f}% across {len(coverages)} classes",
                    }
                )
        except Exception:
            pass
        job.status = "completed"
        job.message = "Split and exemplars ready"
        job.progress = 1.0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent mining job %s failed", job.job_id)
        job.status = "failed"
        job.error = str(exc)
        job.message = "Failed"
    finally:
        try:
            if mining_pool is not None:
                mining_pool.close()
        except Exception:
            pass
        job.updated_at = time.time()


def _run_agent_mining_job(job: AgentMiningJob, payload: AgentMiningRequest) -> None:
    """
    SAM3 Recipe Mining (sam3_greedy mode).

    Builds one greedy recipe per class:
    - prompt bank (text prompts)
    - positive crop bank (diverse train GT crops)
    - optional negative crop bank (diverse other-class crops)

    Then evaluates the full greedy recipe pipeline on the validation split.
    """
    with AGENT_MINING_JOBS_LOCK:
        AGENT_MINING_JOBS[job.job_id] = job
    job.status = "running"
    job.message = "Preparing dataset split…"
    job.request = payload.dict()
    job.updated_at = time.time()

    sam3_model: Optional[Any] = None
    sam3_processor: Optional[Any] = None
    sam3_shared_runtime = False

    def _log(msg: str) -> None:
        try:
            job.logs.append({"ts": time.time(), "msg": msg})
            if len(job.logs) > MAX_JOB_LOGS:
                job.logs[:] = job.logs[-MAX_JOB_LOGS:]
        except Exception:
            pass
        job.updated_at = time.time()

    def _cancelled() -> bool:
        return bool(job.cancel_event.is_set())

    try:
        _enforce_agent_mining_cache_limits(AGENT_MINING_DET_CACHE_ROOT, allow_when_running=False)

        dataset_root = _resolve_sam3_or_qwen_dataset(payload.dataset_id)
        _log(f"Dataset resolved at {dataset_root}")
        _log(
            "Request: "
            f"test_mode={payload.test_mode} "
            f"train_limit={payload.test_train_limit} val_limit={payload.test_val_limit} "
            f"val_percent={payload.val_percent:.3f} ({payload.val_percent * 100:.1f}%) "
            f"split_seed={payload.split_seed}"
        )

        split = _ensure_agent_mining_split(
            payload.dataset_id,
            dataset_root,
            val_percent=payload.val_percent,
            seed=payload.split_seed,
            reuse_split=payload.reuse_split,
            test_mode=payload.test_mode,
            train_limit=payload.test_train_limit if payload.test_mode else None,
            val_limit=payload.test_val_limit if payload.test_mode else None,
        )
        train_ids = [int(i) for i in (split.get("train") or [])]
        val_ids = [int(i) for i in (split.get("val") or [])]
        if not train_ids or not val_ids:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_mining_empty_split")
        _log(
            f"Prepared split with {len(train_ids)} train / {len(val_ids)} val images "
            f"(cached={split.get('_cached', False)})"
        )
        if payload.test_mode:
            _log(f"Test mode enabled (train_limit={payload.test_train_limit}, val_limit={payload.test_val_limit})")
        job.progress = 0.05

        coco, gt_by_image_cat, images = _load_coco_index(dataset_root)
        categories = coco.get("categories") or []
        _log(f"Loaded COCO with {len(categories)} categories")

        cat_filter: Optional[set[int]] = None
        if payload.classes:
            cat_filter = set()
            for cid in payload.classes:
                try:
                    cat_filter.add(int(cid))
                except Exception:
                    continue

        selected_categories: List[Tuple[int, str]] = []
        for idx, cat in enumerate(categories):
            try:
                cid = int(cat.get("id", idx))
            except Exception:
                cid = idx
            if cat_filter and cid not in cat_filter:
                continue
            selected_categories.append((cid, str(cat.get("name", f"class_{cid}"))))
        if not selected_categories:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_mining_no_classes_selected")

        _log(
            "Config: "
            f"seed_thr={payload.seed_threshold} expand_thr={payload.expand_threshold} sim_floor={payload.similarity_score} "
            f"pos_crops={payload.positives_per_class} "
            f"neg_crops={payload.max_negatives_per_class if payload.use_negative_exemplars else 0} "
            f"neg_strength={payload.negative_strength} "
            f"max_seeds={payload.max_visual_seeds} seed_iou={payload.seed_dedupe_iou} out_iou={payload.dedupe_iou} "
            f"mask_thr={payload.mask_threshold} max_results={payload.max_results} iou_eval={payload.iou_threshold} "
            f"cluster_exemplars={payload.cluster_exemplars} llm_prompts={payload.prompt_llm_max_prompts}"
        )

        # Optional pretrained CLIP head (LogReg) used as an additional filter during mining.
        clip_head_path = _resolve_agent_clip_classifier_path(payload.clip_head_classifier_path)
        clip_head: Optional[Dict[str, Any]] = None
        if clip_head_path is not None:
            clip_head = _load_clip_head_from_classifier(clip_head_path)
            _log(
                "Pretrained CLIP head enabled: "
                f"{clip_head_path.name} "
                f"(classes={len(clip_head.get('classes') or [])}, mode={clip_head.get('proba_mode')}) "
                f"(min_prob/margin are tuned per class; starting values min_prob={payload.clip_head_min_prob} margin={payload.clip_head_margin})"
            )

        # Prompt mining (GPT-OSS) per class.
        prepared_prompts: Dict[int, List[str]] = {}
        for cid, name in selected_categories:
            if _cancelled():
                break
            base = []
            if payload.text_prompts_by_class and cid in payload.text_prompts_by_class:
                base = [p for p in payload.text_prompts_by_class.get(cid) or [] if isinstance(p, str) and p.strip()]
            if not base:
                base = [name]
            user_extras: List[str] = []
            if payload.extra_prompts_by_class and isinstance(payload.extra_prompts_by_class, dict):
                raw_extra = payload.extra_prompts_by_class.get(name)
                if isinstance(raw_extra, list):
                    user_extras = [p for p in raw_extra if isinstance(p, str) and p.strip()]
                elif isinstance(raw_extra, str) and raw_extra.strip():
                    user_extras = [raw_extra.strip()]
            base = _sanitize_prompts([*base, *user_extras]) or [name]
            extras: List[str] = []
            if payload.prompt_llm_max_prompts > 0:
                extras = _expand_prompts_with_prompt_llm(
                    name,
                    base,
                    payload.prompt_llm_max_prompts,
                    log_fn=_log,
                    max_new_tokens=payload.prompt_max_new_tokens,
                    reasoning=payload.prompt_reasoning,
                )
            merged: List[str] = []
            seen = set()
            for p in [*base, *extras]:
                key = str(p).lower().strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(str(p))
            prepared_prompts[cid] = merged or base
            _log(f"Prompt list for {name}: {', '.join(prepared_prompts[cid])}")

        try:
            _unload_prompt_llm_runtime()
            _log("Prompt LLM unloaded to free memory before SAM3 init")
        except Exception:
            pass
        job.progress = max(job.progress, 0.15)

        # Prefer a dedicated SAM3 processor for mining so it doesn't interfere with interactive SAM3 use.
        # If it fails (e.g., transient GPU init issues), fall back to the shared SAM3 runtime.
        device = _resolve_sam3_mining_devices()[0]
        _log(f"Using SAM3 mining device: {device}")
        try:
            sam3_model, sam3_processor = _build_sam3_text_processor_for_device(device)
        except Exception as exc:  # noqa: BLE001
            sam3_shared_runtime = True
            _log(f"Dedicated SAM3 mining init failed; falling back to shared runtime: {exc}")
            sam3_model, sam3_processor, _ = _ensure_sam3_text_runtime()
        job.progress = max(job.progress, 0.2)

        train_set = set(train_ids)
        val_set = set(val_ids)
        results: List[Dict[str, Any]] = []
        total_classes = len(selected_categories)

        for class_idx, (cid, name) in enumerate(selected_categories, start=1):
            if _cancelled():
                break
            job.message = f"Mining {name} ({class_idx}/{total_classes})"
            job.updated_at = time.time()

            # Count GTs in split.
            train_gt = 0
            val_gt = 0
            for img_id, cat_map in gt_by_image_cat.items():
                bboxes = cat_map.get(cid)
                if not bboxes:
                    continue
                count = len(bboxes)
                if img_id in train_set:
                    train_gt += count
                elif img_id in val_set:
                    val_gt += count

            # Positive crop bank from train GTs.
            pos_candidates: List[Dict[str, Any]] = []
            for img_id in train_ids:
                boxes = (gt_by_image_cat.get(img_id) or {}).get(cid) or []
                for b_idx, bbox in enumerate(boxes):
                    if not bbox or len(bbox) < 4:
                        continue
                    try:
                        x, y, w, h = map(float, bbox[:4])
                    except Exception:
                        continue
                    area = max(0.0, w) * max(0.0, h)
                    embed_id = f"pos:{cid}:{img_id}:{b_idx}"
                    pos_candidates.append(
                        {
                            "id": embed_id,
                            "image_id": int(img_id),
                            "bbox": [x, y, w, h],
                            "area": area,
                            "embed_id": embed_id,
                        }
                    )
            rng = random.Random(payload.split_seed + cid)
            rng.shuffle(pos_candidates)
            if payload.exemplar_candidate_mode == "percent":
                pct = max(1, min(100, int(payload.exemplar_candidate_value)))
                pool_cap = max(1, int(math.ceil(len(pos_candidates) * (pct / 100.0)))) if pos_candidates else 0
            else:
                pool_cap = max(1, min(int(payload.exemplar_candidate_value), len(pos_candidates))) if pos_candidates else 0
            pool = pos_candidates[:pool_cap] if pool_cap else []
            positives_target = max(0, int(payload.positives_per_class))
            use_clip = bool(payload.use_clip_fp_guard)
            positives: List[Dict[str, Any]] = []
            pos_embeddings: Dict[str, np.ndarray] = {}
            pos_warnings: List[str] = []

            if positives_target > 0 and pool:
                if payload.cluster_exemplars and use_clip:
                    # CLIP-diverse crop bank selection requires embeddings for the pool.
                    pos_embeddings_all, pos_warnings = _clip_embed_regions(pool, images, max_regions=len(pool))
                    positives = _k_center_select(pool, pos_embeddings_all, positives_target) or pool[:positives_target]
                    sel_ids = {p.get("embed_id") for p in positives if p.get("embed_id")}
                    pos_embeddings = {k: v for k, v in pos_embeddings_all.items() if k in sel_ids}
                else:
                    # Random-but-deterministic (due to rng shuffle) selection.
                    positives = pool[:positives_target]
                    if use_clip and positives:
                        pos_embeddings, pos_warnings = _clip_embed_regions(positives, images, max_regions=len(positives))
            _log(
                f"Exemplars for {name}: {len(pos_candidates)} candidates -> pool {len(pool)}, "
                f"embedded {len(pos_embeddings)}, selected {len(positives)}"
            )
            for w in pos_warnings:
                _log(f"CLIP warning for {name}: {w}")

            neg_embeddings: Dict[str, np.ndarray] = {}
            negatives: List[Dict[str, Any]] = []
            if use_clip and payload.use_negative_exemplars and payload.max_negatives_per_class > 0:
                neg_candidates: List[Dict[str, Any]] = []
                for img_id in train_ids:
                    by_cat = gt_by_image_cat.get(img_id) or {}
                    for other_cid, boxes in by_cat.items():
                        try:
                            other_int = int(other_cid)
                        except Exception:
                            continue
                        if other_int == cid:
                            continue
                        for b_idx, bbox in enumerate(boxes or []):
                            if not bbox or len(bbox) < 4:
                                continue
                            try:
                                x, y, w, h = map(float, bbox[:4])
                            except Exception:
                                continue
                            area = max(0.0, w) * max(0.0, h)
                            embed_id = f"neg:{cid}:{img_id}:{other_int}:{b_idx}"
                            neg_candidates.append(
                                {
                                    "id": embed_id,
                                    "image_id": int(img_id),
                                    "bbox": [x, y, w, h],
                                    "area": area,
                                    "embed_id": embed_id,
                                }
                            )
                rng_n = random.Random(payload.split_seed + cid + 999)
                rng_n.shuffle(neg_candidates)
                if payload.exemplar_candidate_mode == "percent":
                    pct = max(1, min(100, int(payload.exemplar_candidate_value)))
                    pool_cap_n = max(1, int(math.ceil(len(neg_candidates) * (pct / 100.0)))) if neg_candidates else 0
                else:
                    pool_cap_n = max(1, min(int(payload.exemplar_candidate_value), len(neg_candidates))) if neg_candidates else 0
                pool_n = neg_candidates[:pool_cap_n] if pool_cap_n else []
                neg_target = max(0, int(payload.max_negatives_per_class))
                neg_warnings: List[str] = []
                if neg_target > 0 and pool_n:
                    if payload.cluster_exemplars:
                        neg_embeddings_all, neg_warnings = _clip_embed_regions(pool_n, images, max_regions=len(pool_n))
                        negatives = _k_center_select(pool_n, neg_embeddings_all, neg_target) or pool_n[:neg_target]
                        sel_n_ids = {n.get("embed_id") for n in negatives if n.get("embed_id")}
                        neg_embeddings = {k: v for k, v in neg_embeddings_all.items() if k in sel_n_ids}
                    else:
                        negatives = pool_n[:neg_target]
                        if negatives:
                            neg_embeddings, neg_warnings = _clip_embed_regions(negatives, images, max_regions=len(negatives))
                _log(
                    f"Negatives for {name}: {len(neg_candidates)} candidates -> pool {len(pool_n)}, "
                    f"embedded {len(neg_embeddings)}, selected {len(negatives)}"
                )
                for w in neg_warnings:
                    _log(f"CLIP warning (negatives) for {name}: {w}")

            ex_mat = None
            if pos_embeddings:
                try:
                    ex_mat = np.stack(list(pos_embeddings.values())).astype(np.float32)
                    ex_mat = ex_mat / (np.linalg.norm(ex_mat, axis=1, keepdims=True) + 1e-8)
                except Exception:
                    ex_mat = None
            neg_mat = None
            if neg_embeddings:
                try:
                    neg_mat = np.stack(list(neg_embeddings.values())).astype(np.float32)
                    neg_mat = neg_mat / (np.linalg.norm(neg_mat, axis=1, keepdims=True) + 1e-8)
                except Exception:
                    neg_mat = None

            prompts_for_class = prepared_prompts.get(cid) or [name]
            head_target_index: Optional[int] = None
            if clip_head:
                classes_list = clip_head.get("classes") if isinstance(clip_head.get("classes"), list) else []
                head_target_index = _find_clip_head_target_index(classes_list, name)
                if head_target_index is None:
                    _log(f"CLIP head: class '{name}' not found; skipping head filter for this class.")
            summary = _evaluate_sam3_greedy_recipe(
                cat_id=cid,
                image_ids=val_ids,
                images=images,
                gt_by_image_cat=gt_by_image_cat,
                processor=sam3_processor,
                text_prompts=prompts_for_class,
                exemplar_embeddings=ex_mat,
                negative_embeddings=neg_mat,
                clip_head=clip_head,
                clip_head_target_index=head_target_index,
                clip_head_min_prob=payload.clip_head_min_prob,
                clip_head_margin=payload.clip_head_margin,
                payload=payload,
                cancel_event=job.cancel_event,
                log_every=50 if not payload.test_mode else 5,
                log_fn=_log,
            )
            if clip_head is not None and head_target_index is not None and isinstance(summary, dict):
                try:
                    tuned_min_prob = summary.get("clip_head_min_prob")
                    tuned_margin = summary.get("clip_head_margin")
                    if tuned_min_prob is not None:
                        _log(
                            f"CLIP head tuned for {name}: min_prob={float(tuned_min_prob):.3f} "
                            f"margin={float(tuned_margin) if tuned_margin is not None else float(payload.clip_head_margin):.3f}"
                        )
                except Exception:
                    pass

            recipe = {
                "mode": "sam3_greedy",
                "text_prompts": prompts_for_class,
                "positives": positives,
                "negatives": negatives,
                "params": {
                    "use_clip_fp_guard": bool(payload.use_clip_fp_guard),
                    "use_negative_exemplars": bool(payload.use_clip_fp_guard and payload.use_negative_exemplars and negatives),
                    "negative_strength": float(payload.negative_strength),
                    "similarity_score": float(payload.similarity_score),
                    "seed_threshold": float(payload.seed_threshold),
                    "expand_threshold": float(payload.expand_threshold),
                    "max_visual_seeds": int(payload.max_visual_seeds),
                    "seed_dedupe_iou": float(payload.seed_dedupe_iou),
                    "dedupe_iou": float(payload.dedupe_iou),
                    "mask_threshold": float(payload.mask_threshold),
                    "min_size": int(payload.min_size),
                    "simplify_epsilon": float(payload.simplify_epsilon),
                    "max_results": int(payload.max_results),
                },
                "summary": summary,
            }
            if clip_head_path is not None and clip_head is not None and head_target_index is not None:
                tuned_min_prob = summary.get("clip_head_min_prob") if isinstance(summary, dict) else None
                tuned_margin = summary.get("clip_head_margin") if isinstance(summary, dict) else None
                try:
                    tuned_min_prob_f = float(tuned_min_prob) if tuned_min_prob is not None else float(payload.clip_head_min_prob)
                except Exception:
                    tuned_min_prob_f = float(payload.clip_head_min_prob)
                try:
                    tuned_margin_f = float(tuned_margin) if tuned_margin is not None else float(payload.clip_head_margin)
                except Exception:
                    tuned_margin_f = float(payload.clip_head_margin)
                recipe["_clip_head_classifier_path"] = str(clip_head_path)
                recipe["clip_head"] = {
                    "artifact": "clip_head/head.npz",
                    "clip_model": clip_head.get("clip_model"),
                    "proba_mode": clip_head.get("proba_mode"),
                    "classes": clip_head.get("classes") if isinstance(clip_head.get("classes"), list) else [],
                    "min_prob": tuned_min_prob_f,
                    "margin": tuned_margin_f,
                }
            results.append(
                {
                    "id": cid,
                    "name": name,
                    "train_gt": train_gt,
                    "val_gt": val_gt,
                    "recipe": recipe,
                }
            )

            job.progress = max(job.progress, 0.2 + 0.8 * (class_idx / max(1, total_classes)))
            job.updated_at = time.time()

        job.result = {
            "dataset_id": payload.dataset_id,
            "split": {"train": len(train_ids), "val": len(val_ids), "seed": payload.split_seed, "val_percent": payload.val_percent},
            "classes": results,
            "config": payload.dict(),
            "note": "Agent mining completed (sam3_greedy mode).",
        }
        if _cancelled():
            job.status = "cancelled"
            job.message = "Cancelled"
        else:
            job.status = "completed"
            job.message = "Done"
            job.progress = 1.0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent mining job %s failed", job.job_id)
        job.status = "failed"
        job.error = str(exc)
        job.message = "Failed"
    finally:
        try:
            if not sam3_shared_runtime:
                if sam3_processor is not None:
                    del sam3_processor
                if sam3_model is not None:
                    del sam3_model
        except Exception:
            pass
        if torch.cuda.is_available() and not sam3_shared_runtime:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        job.updated_at = time.time()


def _start_prompt_helper_job(payload: PromptHelperRequest) -> PromptHelperJob:
    job_id = f"ph_{uuid.uuid4().hex[:8]}"
    job = PromptHelperJob(job_id=job_id)
    with PROMPT_HELPER_JOBS_LOCK:
        PROMPT_HELPER_JOBS[job.job_id] = job
    thread = threading.Thread(target=_run_prompt_helper_job, args=(job, payload), daemon=True)
    thread.start()
    return job


def _start_agent_mining_job(payload: AgentMiningRequest) -> AgentMiningJob:
    job_id = f"am_{uuid.uuid4().hex[:8]}"
    job = AgentMiningJob(job_id=job_id)
    with AGENT_MINING_JOBS_LOCK:
        AGENT_MINING_JOBS[job.job_id] = job
    thread = threading.Thread(target=_run_agent_mining_job, args=(job, payload), daemon=True)
    thread.start()
    return job


def _cancel_agent_mining_job(job_id: str) -> AgentMiningJob:
    with AGENT_MINING_JOBS_LOCK:
        job = AGENT_MINING_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="agent_mining_job_not_found")
    if job.status in {"completed", "failed", "cancelled"}:
        return job
    job.cancel_event.set()
    job.status = "cancelled"
    job.message = "Cancelled"
    job.updated_at = time.time()
    return job


def _start_prompt_helper_search_job(payload: PromptHelperSearchRequest) -> PromptHelperJob:
    job_id = f"phs_{uuid.uuid4().hex[:8]}"
    job = PromptHelperJob(job_id=job_id)
    with PROMPT_HELPER_JOBS_LOCK:
        PROMPT_HELPER_JOBS[job.job_id] = job
    thread = threading.Thread(target=_run_prompt_helper_search_job, args=(job, payload), daemon=True)
    thread.start()
    return job


def _start_prompt_recipe_job(payload: PromptRecipeRequest) -> PromptHelperJob:
    job_id = f"phr_{uuid.uuid4().hex[:8]}"
    job = PromptHelperJob(job_id=job_id)
    with PROMPT_HELPER_JOBS_LOCK:
        PROMPT_HELPER_JOBS[job.job_id] = job
    thread = threading.Thread(target=_run_prompt_recipe_job, args=(job, payload), daemon=True)
    thread.start()
    return job


@app.post("/agent_mining/jobs")
def start_agent_mining_job(payload: AgentMiningRequest):
    job = _start_agent_mining_job(payload)
    return _serialize_agent_mining_job(job)


@app.get("/agent_mining/jobs")
def list_agent_mining_jobs():
    _prune_job_registry(AGENT_MINING_JOBS, AGENT_MINING_JOBS_LOCK)
    with AGENT_MINING_JOBS_LOCK:
        jobs = list(AGENT_MINING_JOBS.values())
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return [_serialize_agent_mining_job(j) for j in jobs]


@app.get("/agent_mining/jobs/{job_id}")
def get_agent_mining_job(job_id: str):
    with AGENT_MINING_JOBS_LOCK:
        job = AGENT_MINING_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="agent_mining_job_not_found")
    return _serialize_agent_mining_job(job)


@app.post("/agent_mining/jobs/{job_id}/cancel")
def cancel_agent_mining_job(job_id: str):
    job = _cancel_agent_mining_job(job_id)
    return _serialize_agent_mining_job(job)


@app.get("/agent_mining/results/latest")
def get_latest_agent_mining_result():
    with AGENT_MINING_JOBS_LOCK:
        jobs = [j for j in AGENT_MINING_JOBS.values() if j.status in {"running", "completed", "failed", "cancelled"}]
    if not jobs:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="agent_mining_result_not_found")
    jobs.sort(key=lambda j: j.updated_at, reverse=True)
    return _serialize_agent_mining_job(jobs[0])


@app.get("/agent_mining/cache_size")
def agent_mining_cache_size():
    cache_root = AGENT_MINING_DET_CACHE_ROOT
    # Light touch: enforce TTL/size only when no active job to avoid surprises.
    _enforce_agent_mining_cache_limits(cache_root, allow_when_running=False)
    total = 0
    files = 0
    try:
        for p in cache_root.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
                    files += 1
            except Exception:
                continue
    except Exception:
        total = 0
    return {
        "bytes": total,
        "files": files,
        "max_bytes": AGENT_MINING_CACHE_MAX_BYTES,
        "ttl_hours": AGENT_MINING_CACHE_TTL_HOURS,
    }


@app.post("/agent_mining/cache/purge")
def agent_mining_cache_purge():
    cache_root = AGENT_MINING_DET_CACHE_ROOT
    if _agent_cache_running_jobs():
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="agent_cache_purge_blocked_active_jobs")
    if not cache_root.exists():
        return {"status": "ok", "deleted_bytes": 0, "deleted_files": 0}
    deleted = 0
    deleted_files = 0
    paths = sorted(cache_root.rglob("*"), key=lambda x: len(x.parts), reverse=True)
    for p in paths:
        try:
            if p.is_file():
                deleted += p.stat().st_size
                deleted_files += 1
                p.unlink()
            elif p.is_dir():
                p.rmdir()
        except Exception:
            continue
    return {"status": "ok", "deleted_bytes": deleted, "deleted_files": deleted_files}


class AgentApplyRequest(BaseModel):
    dataset_id: str
    image_id: int
    recipe: Dict[str, Any]
    mask_threshold: float = Field(0.5, ge=0.0, le=1.0)
    min_size: int = Field(0, ge=0, le=10_000)
    simplify_epsilon: float = Field(0.0, ge=0.0, le=1_000.0)
    max_results: int = Field(100, ge=1, le=5000)
    override_class_id: Optional[int] = Field(None, ge=0)
    override_class_name: Optional[str] = None


@app.post("/agent_mining/apply", response_model=Sam3TextPromptResponse)
def agent_mining_apply(payload: AgentApplyRequest):
    dataset_root = _resolve_sam3_or_qwen_dataset(payload.dataset_id)
    coco, _, images = _load_coco_index(dataset_root)
    categories = coco.get("categories") or []
    labelmap_hash, _ = _compute_labelmap_hash(categories)
    warnings: List[str] = []
    recipe_meta = payload.recipe or {}
    recipe_labelmap_hash = recipe_meta.get("labelmap_hash")
    if recipe_labelmap_hash and recipe_labelmap_hash != labelmap_hash:
        warnings.append("labelmap_mismatch")
    recipe_signature = recipe_meta.get("dataset_signature")
    dataset_signature = _compute_dataset_signature(payload.dataset_id, dataset_root, images, categories)
    if recipe_signature and recipe_signature != dataset_signature:
        warnings.append("dataset_mismatch")
    target_class_id = recipe_meta.get("class_id")
    target_class_name = recipe_meta.get("class_name")
    class_id_int: Optional[int] = None
    override_class_id = payload.override_class_id
    override_class_name = payload.override_class_name
    if override_class_id is not None or override_class_name:
        # Explicit override takes precedence over the stored recipe class.
        if override_class_id is not None:
            present = any(int(cat.get("id", idx)) == override_class_id for idx, cat in enumerate(categories))
            if not present:
                raise HTTPException(status_code=HTTP_412_PRECONDITION_FAILED, detail="agent_override_class_missing")
            class_id_int = override_class_id
        elif override_class_name:
            target_lower = str(override_class_name).lower()
            for idx, cat in enumerate(categories):
                try:
                    cid = int(cat.get("id", idx))
                except Exception:
                    cid = idx
                if str(cat.get("name", "")).lower() == target_lower:
                    class_id_int = cid
                    break
            if class_id_int is None:
                raise HTTPException(status_code=HTTP_412_PRECONDITION_FAILED, detail="agent_override_class_missing")
        warnings.append("class_override_used")
    else:
        if target_class_id is not None:
            try:
                class_id_int = int(target_class_id)
            except Exception:
                class_id_int = None
        name_match_id: Optional[int] = None
        if target_class_name:
            target_lower = str(target_class_name).lower()
            for idx, cat in enumerate(categories):
                try:
                    cid = int(cat.get("id", idx))
                except Exception:
                    cid = idx
                if str(cat.get("name", "")).lower() == target_lower:
                    name_match_id = cid
                    break
        # Prefer ID match; if missing but name matches, use name match and add warning.
        if class_id_int is not None:
            present = any(int(cat.get("id", idx)) == class_id_int for idx, cat in enumerate(categories))
            if not present and name_match_id is not None:
                warnings.append("class_id_remapped_by_name")
                class_id_int = name_match_id
            elif not present:
                raise HTTPException(status_code=HTTP_412_PRECONDITION_FAILED, detail="agent_recipe_class_missing")
        elif name_match_id is not None:
            class_id_int = name_match_id
        else:
            raise HTTPException(status_code=HTTP_412_PRECONDITION_FAILED, detail="agent_recipe_class_missing")
    class_name_resolved: Optional[str] = None
    if class_id_int is not None:
        for idx, cat in enumerate(categories):
            try:
                cid = int(cat.get("id", idx))
            except Exception:
                cid = idx
            if cid == class_id_int:
                class_name_resolved = str(cat.get("name", class_name_resolved or target_class_name or ""))
                break
    if not class_name_resolved and target_class_name:
        class_name_resolved = str(target_class_name)
    img = images.get(payload.image_id)
    if not img:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="agent_mining_image_not_found")
    dets = _apply_agent_recipe_to_image(
        payload.recipe,
        image=img,
        dataset_id=payload.dataset_id,
        images=images,
        mask_threshold=payload.mask_threshold,
        min_size=payload.min_size,
        simplify_epsilon=payload.simplify_epsilon,
        max_results=payload.max_results,
        class_id=class_id_int,
        class_name=class_name_resolved,
        warnings=warnings,
    )
    return Sam3TextPromptResponse(detections=dets, warnings=warnings, image_token=None)


class AgentApplyImageRequest(BaseModel):
    image_base64: Optional[str] = None
    image_token: Optional[str] = None
    image_name: Optional[str] = None
    sam_variant: Optional[str] = None
    recipe: Dict[str, Any]
    mask_threshold: float = Field(0.5, ge=0.0, le=1.0)
    min_size: int = Field(0, ge=0, le=10_000)
    simplify_epsilon: float = Field(0.0, ge=0.0, le=1_000.0)
    max_results: int = Field(100, ge=1, le=5000)
    override_class_id: Optional[int] = Field(None, ge=0)
    override_class_name: Optional[str] = None

    @root_validator(skip_on_failure=True)
    def _ensure_agent_apply_image_payload(cls, values):  # noqa: N805
        if not values.get("image_base64") and not values.get("image_token"):
            raise ValueError("image_payload_missing")
        if not isinstance(values.get("recipe"), dict) or not values.get("recipe"):
            raise ValueError("recipe_required")
        return values


@app.post("/agent_mining/apply_image", response_model=Sam3TextPromptResponse)
def agent_mining_apply_image(payload: AgentApplyImageRequest):
    variant = _default_variant(payload.sam_variant or "sam3")
    if variant != "sam3":
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_apply_requires_sam3")
    pil_img, _, token = resolve_image_payload(payload.image_base64, payload.image_token, variant)
    warnings: List[str] = []
    recipe_meta = payload.recipe or {}

    class_id_val = recipe_meta.get("class_id")
    class_name_val = recipe_meta.get("class_name")
    if payload.override_class_id is not None or payload.override_class_name:
        warnings.append("class_override_used")
        if payload.override_class_id is not None:
            class_id_val = payload.override_class_id
        if payload.override_class_name:
            class_name_val = payload.override_class_name
    class_id_int: Optional[int] = None
    if class_id_val is not None:
        try:
            class_id_int = int(class_id_val)
        except Exception:
            class_id_int = None

    # Reuse the existing apply implementation by staging the in-memory image to a temp file.
    staging_dir = Path(tempfile.mkdtemp(prefix="agent_apply_image_"))
    try:
        img_path = staging_dir / "image.png"
        try:
            pil_img.save(img_path, format="PNG")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"agent_apply_image_encode_failed:{exc}") from exc
        dets = _apply_agent_recipe_to_image(
            payload.recipe,
            image={"path": str(img_path)},
            dataset_id="image_payload",
            images={},
            mask_threshold=payload.mask_threshold,
            min_size=payload.min_size,
            simplify_epsilon=payload.simplify_epsilon,
            max_results=payload.max_results,
            class_id=class_id_int,
            class_name=str(class_name_val) if class_name_val is not None else None,
            warnings=warnings,
        )
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return Sam3TextPromptResponse(detections=dets, warnings=warnings, image_token=token)


class AgentRecipeExportRequest(BaseModel):
    dataset_id: str
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    label: str = Field(..., min_length=1, max_length=128)
    recipe: Dict[str, Any]


@app.post("/agent_mining/recipes", response_model=Dict[str, Any])
def agent_mining_save_recipe(payload: AgentRecipeExportRequest):
    recipe = _persist_agent_recipe(
        payload.dataset_id,
        payload.class_id,
        payload.class_name,
        payload.label,
        payload.recipe,
    )
    return recipe


@app.get("/agent_mining/recipes", response_model=List[Dict[str, Any]])
def agent_mining_list_recipes(dataset_id: Optional[str] = None):
    return _list_agent_recipes(dataset_id)


@app.get("/agent_mining/recipes/{recipe_id}", response_model=Dict[str, Any])
def agent_mining_get_recipe(recipe_id: str):
    return _load_agent_recipe(recipe_id)


@app.get("/agent_mining/recipes/{recipe_id}/export")
def agent_mining_export_recipe(recipe_id: str):
    recipe = _load_agent_recipe(recipe_id)
    zip_path = _ensure_recipe_zip(recipe)
    filename = f"{recipe_id}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    try:
        stream = zip_path.open("rb")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"agent_recipe_export_failed:{exc}") from exc
    return StreamingResponse(stream, media_type="application/zip", headers=headers)


@app.post("/agent_mining/recipes/import", response_model=Dict[str, Any])
async def agent_mining_import_recipe(file: UploadFile = File(...)):
    staging_dir = Path(tempfile.mkdtemp(prefix="agent_recipe_import_", dir=str(AGENT_MINING_RECIPES_ROOT)))
    zip_path = staging_dir / "payload.zip"
    data: Dict[str, Any] = {}
    crops: Dict[str, bytes] = {}
    clip_head_files: Dict[str, bytes] = {}
    try:
        await _write_upload_file(
            file,
            zip_path,
            max_bytes=AGENT_RECIPE_MAX_BYTES,
            quota_root=staging_dir,
            quota_limit=AGENT_RECIPE_MAX_BYTES,
            allow_overwrite=True,
        )
        if not zipfile.is_zipfile(zip_path):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_import_zip_only")
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            json_name = None
            for name in names:
                if name.lower().endswith(".json"):
                    json_name = name
                    break
            if not json_name:
                raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_import_no_json")
            json_info = zf.getinfo(json_name)
            if json_info.file_size > AGENT_RECIPE_MAX_JSON_BYTES:
                raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="agent_recipe_import_json_too_large")
            json_path = Path(json_name)
            if json_path.is_absolute() or ".." in json_path.parts:
                raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_import_invalid_path")
            with zf.open(json_name) as jf:
                data = json.load(jf)

            total_bytes = 0
            crop_count = 0
            clip_head_bytes = 0
            for name in names:
                info = zf.getinfo(name)
                arc_path = Path(name)
                if arc_path.is_dir():
                    continue
                if arc_path.is_absolute() or ".." in arc_path.parts:
                    raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="agent_recipe_import_invalid_path")
                if len(arc_path.parts) < 2 or arc_path.parts[0] != "crops":
                    # Non-crop artifacts we support (portable CLIP head).
                    if len(arc_path.parts) == 2 and arc_path.parts[0] == "clip_head" and arc_path.name in {"head.npz", "meta.json"}:
                        clip_head_bytes += info.file_size
                        if clip_head_bytes > AGENT_RECIPE_MAX_CLIP_HEAD_BYTES:
                            raise HTTPException(
                                status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="agent_recipe_import_clip_head_too_large"
                            )
                        clip_head_files[f"clip_head/{arc_path.name}"] = zf.read(name)
                    continue
                if arc_path.suffix.lower() != ".png":
                    continue
                crop_count += 1
                total_bytes += info.file_size
                if crop_count > AGENT_RECIPE_MAX_CROPS or total_bytes > AGENT_RECIPE_MAX_CROP_BYTES:
                    raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="agent_recipe_import_crops_too_large")
                crops[f"crops/{arc_path.name}"] = zf.read(name)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"agent_recipe_import_failed:{exc}") from exc
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    dataset_id = data.get("dataset_id") or data.get("recipe", {}).get("dataset_id") or ""
    label = data.get("label") or data.get("recipe", {}).get("label") or "imported_recipe"
    class_id = data.get("class_id")
    class_name = data.get("class_name")
    recipe_body = data.get("recipe") or {}
    meta_overrides = {
        "dataset_signature": data.get("dataset_signature"),
        "labelmap_hash": data.get("labelmap_hash"),
        "labelmap": data.get("labelmap"),
    }
    persisted = _persist_agent_recipe(
        dataset_id,
        class_id,
        class_name,
        label,
        recipe_body,
        crop_overrides=crops,
        clip_head_overrides=clip_head_files,
        meta_overrides=meta_overrides,
    )
    return persisted


@app.delete("/agent_mining/recipes/{recipe_id}")
def agent_mining_delete_recipe(recipe_id: str):
    _delete_agent_recipe(recipe_id)
    return {"id": recipe_id, "deleted": True}


@app.post("/sam3/prompt_helper/suggest")
def prompt_helper_suggest(payload: PromptHelperSuggestRequest):
    return _suggest_prompts_for_dataset(payload)


@app.post("/sam3/prompt_helper/expand")
def prompt_helper_expand(payload: PromptRecipeExpandRequest):
    dataset_root = _resolve_sam3_or_qwen_dataset(payload.dataset_id)
    coco, _, _ = _load_coco_index(dataset_root)
    categories = coco.get("categories") or []
    cat_entry = next((c for c in categories if int(c.get("id", categories.index(c))) == payload.class_id), None)
    if not cat_entry:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="recipe_class_not_found")
    class_name = str(cat_entry.get("name", f"class_{payload.class_id}"))
    base_prompts = [p.strip() for p in payload.base_prompts if isinstance(p, str) and p.strip()]
    new_prompts = _expand_prompts_with_prompt_llm(
        class_name,
        base_prompts,
        payload.max_new,
        max_new_tokens=payload.max_new_tokens if hasattr(payload, "max_new_tokens") else 128,
        reasoning="high",
    )
    combined: List[str] = []
    seen = set()
    for prompt in [*base_prompts, *new_prompts]:
        low = prompt.lower()
        if low in seen:
            continue
        seen.add(low)
        combined.append(prompt)
    return {
        "class_id": payload.class_id,
        "class_name": class_name,
        "base_prompts": base_prompts,
        "new_prompts": new_prompts,
        "combined": combined,
    }


def _list_prompt_helper_presets() -> List[Dict[str, Any]]:
    presets: List[Dict[str, Any]] = []
    for path in PROMPT_HELPER_PRESET_ROOT.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            presets.append(data)
        except Exception:
            continue
    presets.sort(key=lambda p: p.get("created_at", 0), reverse=True)
    return presets


def _load_prompt_helper_preset(preset_id: str) -> Dict[str, Any]:
    path = (PROMPT_HELPER_PRESET_ROOT / f"{preset_id}.json").resolve()
    if not str(path).startswith(str(PROMPT_HELPER_PRESET_ROOT.resolve())) or not path.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="prompt_helper_preset_not_found")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"prompt_helper_preset_load_failed:{exc}") from exc


def _save_prompt_helper_preset(label: str, dataset_id: str, prompts_by_class: Dict[int, List[str]]) -> Dict[str, Any]:
    preset_id = f"phset_{uuid.uuid4().hex[:8]}"
    created_at = time.time()
    payload = {
        "id": preset_id,
        "label": label or preset_id,
        "dataset_id": dataset_id,
        "created_at": created_at,
        "prompts_by_class": prompts_by_class,
    }
    path = (PROMPT_HELPER_PRESET_ROOT / f"{preset_id}.json").resolve()
    if not str(path).startswith(str(PROMPT_HELPER_PRESET_ROOT.resolve())):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="prompt_helper_preset_path_invalid")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


@app.post("/sam3/prompt_helper/jobs")
def start_prompt_helper_job(payload: PromptHelperRequest):
    job = _start_prompt_helper_job(payload)
    return _serialize_prompt_helper_job(job)


@app.post("/sam3/prompt_helper/search")
def start_prompt_helper_search(payload: PromptHelperSearchRequest):
    job = _start_prompt_helper_search_job(payload)
    return _serialize_prompt_helper_job(job)


@app.post("/sam3/prompt_helper/recipe")
def start_prompt_helper_recipe(payload: PromptRecipeRequest):
    job = _start_prompt_recipe_job(payload)
    return _serialize_prompt_helper_job(job)


@app.get("/sam3/recipe_presets")
def list_prompt_recipe_presets():
    return _list_prompt_recipe_presets()


@app.get("/sam3/recipe_presets/{preset_id}")
def load_prompt_recipe_preset(preset_id: str):
    return _load_prompt_recipe_preset(preset_id)


class PromptRecipePresetSave(BaseModel):
    label: Optional[str] = None
    class_name: str
    class_id: Optional[int] = None
    dataset_id: Optional[str] = None
    steps: List[Dict[str, Any]]


@app.post("/sam3/recipe_presets")
def save_prompt_recipe_preset(payload: PromptRecipePresetSave):
    class_name = (payload.class_name or "").strip()
    if not class_name:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="prompt_recipe_preset_class_missing")
    cleaned_steps: List[Dict[str, Any]] = []
    for step in payload.steps:
        prompt = (step.get("prompt") or "").strip()
        thr = step.get("threshold")
        try:
            thr_val = float(thr)
        except Exception:
            continue
        if not prompt or thr_val < 0.0 or thr_val > 1.0:
            continue
        cleaned_steps.append({"prompt": prompt, "threshold": thr_val})
    return _save_prompt_recipe_preset(payload.label or "", class_name, payload.class_id, cleaned_steps, payload.dataset_id)


@app.get("/sam3/prompt_helper/presets")
def list_prompt_helper_presets():
    return _list_prompt_helper_presets()


@app.get("/sam3/prompt_helper/presets/{preset_id}")
def get_prompt_helper_preset(preset_id: str):
    return _load_prompt_helper_preset(preset_id)


@app.post("/sam3/prompt_helper/presets")
def create_prompt_helper_preset(
    dataset_id: str = Form(...),
    label: str = Form(""),
    prompts_json: str = Form(...),
):
    try:
        prompts_by_class = json.loads(prompts_json)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"invalid_prompts:{exc}") from exc
    if not isinstance(prompts_by_class, dict):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="prompts_must_be_object")
    normalized: Dict[int, List[str]] = {}
    for key, vals in prompts_by_class.items():
        try:
            cid = int(key)
        except Exception:
            continue
        if not isinstance(vals, (list, tuple)):
            continue
        cleaned = [str(v).strip() for v in vals if isinstance(v, str) and str(v).strip()]
        if cleaned:
            normalized[cid] = cleaned
    if not normalized:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="no_prompts_provided")
    preset = _save_prompt_helper_preset(label, dataset_id, normalized)
    return preset


@app.get("/sam3/prompt_helper/jobs")
def list_prompt_helper_jobs():
    _prune_job_registry(PROMPT_HELPER_JOBS, PROMPT_HELPER_JOBS_LOCK)
    with PROMPT_HELPER_JOBS_LOCK:
        jobs = list(PROMPT_HELPER_JOBS.values())
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return [_serialize_prompt_helper_job(j) for j in jobs]


@app.get("/sam3/prompt_helper/jobs/{job_id}")
def get_prompt_helper_job(job_id: str):
    with PROMPT_HELPER_JOBS_LOCK:
        job = PROMPT_HELPER_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="prompt_helper_job_not_found")
    return _serialize_prompt_helper_job(job)


@app.post("/segmentation/build/jobs")
def start_segmentation_build_job(request: SegmentationBuildRequest):
    job = _start_segmentation_build_job(request)
    return _serialize_seg_job(job)


@app.get("/segmentation/build/jobs")
def list_segmentation_build_jobs():
    _prune_job_registry(SEGMENTATION_BUILD_JOBS, SEGMENTATION_BUILD_JOBS_LOCK)
    with SEGMENTATION_BUILD_JOBS_LOCK:
        jobs = list(SEGMENTATION_BUILD_JOBS.values())
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return [_serialize_seg_job(job) for job in jobs]


@app.get("/segmentation/build/jobs/{job_id}")
def get_segmentation_build_job(job_id: str):
    with SEGMENTATION_BUILD_JOBS_LOCK:
        job = SEGMENTATION_BUILD_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="segmentation_job_not_found")
    return _serialize_seg_job(job)


def _collect_labels_from_qwen_jsonl(jsonl_path: Path) -> List[str]:
    labels: set[str] = set()
    if not jsonl_path.exists():
        return []
    try:
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                detections = payload.get("detections") or []
                if not isinstance(detections, list):
                    continue
                for det in detections:
                    if not isinstance(det, dict):
                        continue
                    label = str(det.get("label", "")).strip()
                    if label:
                        labels.add(label)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to scan labels from %s: %s", jsonl_path, exc)
    return sorted(labels)


def _load_qwen_labelmap(dataset_root: Path) -> List[str]:
    meta = _load_qwen_dataset_metadata(dataset_root) or {}
    classes = [str(cls).strip() for cls in meta.get("classes", []) if str(cls).strip()]
    if classes:
        return classes
    labels = set()
    for split in ("train", "val"):
        labels.update(_collect_labels_from_qwen_jsonl(dataset_root / split / "annotations.jsonl"))
    return sorted(labels)


def _load_labelmap_file(path: Path) -> List[str]:
    if not path.exists():
        return []
    lower = path.name.lower()
    try:
        if lower.endswith(".pkl"):
            obj = joblib.load(path)
            if isinstance(obj, list):
                return [str(x) for x in obj]
            raise ValueError("labelmap_pickle_not_list")
        with path.open("r", encoding="utf-8") as handle:
            classes = [ln.strip() for ln in handle if ln.strip()]
        return classes
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load labelmap from %s: %s", path, exc)
        return []


def _discover_yolo_labelmap(dataset_root: Path) -> List[str]:
    for name in ("labelmap.txt", "classes.txt", "labels.txt"):
        candidate = dataset_root / name
        classes = _load_labelmap_file(candidate)
        if classes:
            return classes
    return []


def _coco_info_block(dataset_id: str) -> Dict[str, Any]:
    """Minimal COCO info section to keep pycocotools happy."""
    return {
        "description": f"{dataset_id} generated by tator",
        "version": "1.0",
        "year": int(time.strftime("%Y", time.gmtime())),
        "contributor": "tator",
        "date_created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _write_coco_annotations(
    output_path: Path,
    *,
    dataset_id: str,
    categories: List[Dict[str, Any]],
    images: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
) -> None:
    payload = {
        "info": _coco_info_block(dataset_id),
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _ensure_coco_info_fields(path: Path, dataset_id: str, categories: List[Dict[str, Any]]) -> str:
    """Backfill missing COCO 'info'/'licenses' for older conversions."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load COCO file %s to backfill info: %s", path, exc)
        return str(path)
    if not isinstance(data, dict):
        return str(path)
    modified = False
    if "info" not in data or not isinstance(data["info"], dict):
        data["info"] = _coco_info_block(dataset_id)
        modified = True
    if "licenses" not in data or not isinstance(data["licenses"], list):
        data["licenses"] = []
        modified = True
    if categories and (not isinstance(data.get("categories"), list) or not data["categories"]):
        data["categories"] = categories
        modified = True
    if modified:
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to rewrite COCO file %s: %s", path, exc)
    return str(path)


def _convert_yolo_dataset_to_coco(dataset_root: Path) -> Dict[str, Any]:
    dataset_root = dataset_root.resolve()
    train_images = dataset_root / "train" / "images"
    train_labels = dataset_root / "train" / "labels"
    val_images = dataset_root / "val" / "images"
    val_labels = dataset_root / "val" / "labels"
    for path in (train_images, train_labels, val_images, val_labels):
        if not path.exists():
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_yolo_split_missing")

    labelmap = _discover_yolo_labelmap(dataset_root)
    if not labelmap:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_labelmap_missing")
    label_to_id = {label: idx + 1 for idx, label in enumerate(labelmap)}
    categories = [{"id": cid, "name": name} for name, cid in label_to_id.items()]
    signature = _compute_dir_signature(dataset_root)
    existing_meta = _load_sam3_dataset_metadata(dataset_root)
    # Preserve previously recorded metadata and infer dataset type (bbox vs seg) so downstream
    # training can enable masks when appropriate.
    dataset_type = (existing_meta or {}).get("type", "bbox")
    dataset_label = (existing_meta or {}).get("label", dataset_root.name)
    dataset_source = (existing_meta or {}).get("source", "yolo")
    if (
        existing_meta
        and existing_meta.get("signature") == signature
        and existing_meta.get("coco_train_json")
        and existing_meta.get("coco_val_json")
    ):
        # Backfill missing COCO info if this dataset was converted before we added it.
        _ensure_coco_info_fields(Path(existing_meta["coco_train_json"]), dataset_root.name, categories)
        _ensure_coco_info_fields(Path(existing_meta["coco_val_json"]), dataset_root.name, categories)
        return existing_meta

    image_id_counter = 1
    annotation_id = 1
    images_lookup: Dict[str, int] = {}
    image_sizes: Dict[str, Tuple[int, int]] = {}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def _image_path_for_label(labels_dir: Path, images_dir: Path, label_file: Path) -> Optional[Path]:
        stem = label_file.stem
        for ext in image_exts:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        for candidate in images_dir.glob(f"{stem}.*"):
            if candidate.suffix.lower() in image_exts:
                return candidate
        return None

    def _convert_split(split_images: Path, split_labels: Path, split_name: str) -> str:
        nonlocal image_id_counter, annotation_id, dataset_type
        images: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []
        for label_file in sorted(split_labels.rglob("*.txt")):
            image_path = _image_path_for_label(split_labels, split_images, label_file)
            if image_path is None:
                logger.warning("No matching image for label file %s", label_file)
                continue
            image_rel = str(image_path.relative_to(split_images.parent))
            if image_rel not in images_lookup:
                try:
                    with Image.open(image_path) as im:
                        width, height = im.size
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to read image %s: %s", image_path, exc)
                    continue
                images_lookup[image_rel] = image_id_counter
                image_sizes[image_rel] = (width, height)
                images.append(
                    {
                        "id": image_id_counter,
                        "file_name": image_rel,
                        "width": width,
                        "height": height,
                    }
                )
                image_id_counter += 1
            image_id = images_lookup[image_rel]
            width, height = image_sizes.get(image_rel, (None, None))
            try:
                with label_file.open("r", encoding="utf-8") as handle:
                    lines = [ln.strip() for ln in handle if ln.strip()]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read YOLO labels from %s: %s", label_file, exc)
                continue
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    class_idx = int(float(parts[0]))
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except (TypeError, ValueError):
                    continue
                if class_idx < 0 or class_idx >= len(labelmap):
                    continue
                if width is None or height is None:
                    continue
                # YOLO-seg polygons append x/y pairs after the box fields; treat presence as segmentation type.
                if len(parts) > 5:
                    dataset_type = "seg"
                abs_w = w * width
                abs_h = h * height
                x1 = cx * width - abs_w / 2.0
                y1 = cy * height - abs_h / 2.0
                if abs_w <= 0 or abs_h <= 0:
                    continue
                area = abs_w * abs_h
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_idx + 1,
                        "bbox": [x1, y1, abs_w, abs_h],
                        "area": area,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
        output_path = dataset_root / split_name / "_annotations.coco.json"
        try:
            _write_coco_annotations(
                output_path,
                dataset_id=dataset_root.name,
                categories=categories,
                images=images,
                annotations=annotations,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"sam3_coco_write_failed:{exc}") from exc
        return str(output_path)

    coco_train = _convert_split(train_images, train_labels, "train")
    coco_val = _convert_split(val_images, val_labels, "val")
    sam3_meta = {
        "id": dataset_root.name,
        "label": dataset_label,
        "source": dataset_source,
        "type": dataset_type,
        "dataset_root": str(dataset_root),
        "signature": signature,
        "classes": labelmap,
        "context": "",
        "image_count": None,
        "train_count": None,
        "val_count": None,
        "coco_train_json": coco_train,
        "coco_val_json": coco_val,
        "converted_at": time.time(),
    }
    _persist_sam3_dataset_metadata(dataset_root, sam3_meta)
    return sam3_meta


def _convert_qwen_dataset_to_coco(dataset_root: Path) -> Dict[str, Any]:
    dataset_root = dataset_root.resolve()
    metadata = _load_qwen_dataset_metadata(dataset_root) or {}
    metadata, signature = _ensure_qwen_dataset_signature(dataset_root, metadata)
    if "type" not in metadata:
        metadata["type"] = "bbox"
        _persist_qwen_dataset_metadata(dataset_root, metadata)
    dataset_id = metadata.get("id") or dataset_root.name
    labelmap = _load_qwen_labelmap(dataset_root)
    if not labelmap:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_labelmap_missing")
    label_to_id = {label: idx + 1 for idx, label in enumerate(labelmap)}
    categories = [{"id": cid, "name": name} for name, cid in label_to_id.items()]
    existing_meta = _load_sam3_dataset_metadata(dataset_root)
    if (
        existing_meta
        and existing_meta.get("signature") == signature
        and existing_meta.get("coco_train_json")
        and existing_meta.get("coco_val_json")
    ):
        _ensure_coco_info_fields(Path(existing_meta["coco_train_json"]), dataset_id, categories)
        _ensure_coco_info_fields(Path(existing_meta["coco_val_json"]), dataset_id, categories)
        return existing_meta

    annotation_id = 1
    images_lookup: Dict[str, int] = {}
    image_sizes: Dict[str, Tuple[int, int]] = {}

    def _convert_split(split: str) -> str:
        nonlocal annotation_id
        jsonl_path = dataset_root / split / "annotations.jsonl"
        if not jsonl_path.exists():
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"sam3_annotations_missing:{split}")
        images: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []
        try:
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    image_rel = payload.get("image")
                    if not isinstance(image_rel, str):
                        continue
                    if image_rel not in images_lookup:
                        image_path = dataset_root / split / image_rel
                        if not image_path.exists():
                            logger.warning("Missing image referenced in %s: %s", jsonl_path, image_path)
                            continue
                        try:
                            with Image.open(image_path) as im:
                                width, height = im.size
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Failed to read image %s: %s", image_path, exc)
                            continue
                        images_lookup[image_rel] = len(images_lookup) + 1
                        image_sizes[image_rel] = (width, height)
                        images.append(
                            {
                                "id": images_lookup[image_rel],
                                "file_name": image_rel,
                                "width": width,
                                "height": height,
                            }
                        )
                    image_id = images_lookup[image_rel]
                    width, height = image_sizes.get(image_rel, (None, None))
                    detections = payload.get("detections") or []
                    if not isinstance(detections, list):
                        continue
                    for det in detections:
                        if not isinstance(det, dict):
                            continue
                        label = str(det.get("label", "")).strip()
                        if not label or label not in label_to_id:
                            continue
                        bbox = det.get("bbox")
                        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                            try:
                                x1 = float(bbox[0])
                                y1 = float(bbox[1])
                                x2 = float(bbox[2])
                                y2 = float(bbox[3])
                            except (TypeError, ValueError):
                                continue
                            if width is not None and height is not None:
                                x1 = max(0.0, min(x1, width))
                                x2 = max(0.0, min(x2, width))
                                y1 = max(0.0, min(y1, height))
                                y2 = max(0.0, min(y2, height))
                            w = max(0.0, x2 - x1)
                            h = max(0.0, y2 - y1)
                            if w <= 0 or h <= 0:
                                continue
                            coco_bbox = [x1, y1, w, h]
                        else:
                            point = det.get("point")
                            if not (isinstance(point, (list, tuple)) and len(point) >= 2):
                                continue
                            try:
                                cx = float(point[0])
                                cy = float(point[1])
                            except (TypeError, ValueError):
                                continue
                            # Convert point to a tiny box to retain the signal.
                            size = 2.0
                            x1 = cx - size / 2.0
                            y1 = cy - size / 2.0
                            coco_bbox = [x1, y1, size, size]
                        area = coco_bbox[2] * coco_bbox[3]
                        if area <= 0:
                            continue
                        annotations.append(
                            {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": label_to_id[label],
                                "bbox": coco_bbox,
                                "area": area,
                                "iscrowd": 0,
                            }
                        )
                        annotation_id += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to convert %s to COCO: %s", jsonl_path, exc)
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"sam3_coco_conversion_failed:{split}")
        output_path = dataset_root / split / "_annotations.coco.json"
        try:
            _write_coco_annotations(
                output_path,
                dataset_id=dataset_id,
                categories=categories,
                images=images,
                annotations=annotations,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"sam3_coco_write_failed:{exc}") from exc
        return str(output_path)

    coco_train = _convert_split("train")
    coco_val = _convert_split("val")
    sam3_meta = {
        "id": metadata.get("id") or dataset_root.name,
        "label": metadata.get("label") or metadata.get("id") or dataset_root.name,
        "source": "qwen",
        "type": metadata.get("type", "bbox"),
        "dataset_root": str(dataset_root),
        "signature": signature,
        "classes": labelmap,
        "context": metadata.get("context", ""),
        "image_count": metadata.get("image_count"),
        "train_count": metadata.get("train_count"),
        "val_count": metadata.get("val_count"),
        "coco_train_json": coco_train,
        "coco_val_json": coco_val,
        "converted_at": time.time(),
    }
    _persist_sam3_dataset_metadata(dataset_root, sam3_meta)
    return sam3_meta


def _list_sam3_datasets() -> List[Dict[str, Any]]:
    return _list_all_datasets()


def _resolve_sam3_dataset_meta(dataset_id: str) -> Dict[str, Any]:
    dataset_root = _resolve_sam3_or_qwen_dataset(dataset_id)
    annotations_path = dataset_root / "train" / "annotations.jsonl"
    train_images = dataset_root / "train" / "images"
    train_labels = dataset_root / "train" / "labels"
    if annotations_path.exists():
        meta = _convert_qwen_dataset_to_coco(dataset_root)
    elif train_images.exists() and train_labels.exists():
        meta = _convert_yolo_dataset_to_coco(dataset_root)
    else:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_dataset_type_unsupported")
    meta["dataset_root"] = str(dataset_root)
    return meta


def _prepare_sam3_training_split(
    dataset_root: Path,
    meta: Dict[str, Any],
    job_id: str,
    *,
    random_split: bool,
    val_percent: float,
    split_seed: int,
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    log_messages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not random_split:
        return meta
    coco_train_path = Path(meta.get("coco_train_json", ""))
    coco_val_path = Path(meta.get("coco_val_json", ""))
    if not coco_train_path.exists() or not coco_val_path.exists():
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_coco_split_missing")
    try:
        with coco_train_path.open("r", encoding="utf-8") as handle:
            coco_train = json.load(handle)
        with coco_val_path.open("r", encoding="utf-8") as handle:
            coco_val = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"sam3_coco_load_failed:{exc}") from exc
    categories = coco_train.get("categories") or coco_val.get("categories") or []
    if not categories and meta.get("classes"):
        categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(meta.get("classes", []))]
    if not categories:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_categories_missing")
    images: Dict[int, Dict[str, Any]] = {}
    ann_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for coco_blob in (coco_train, coco_val):
        for img in coco_blob.get("images", []):
            try:
                img_id = int(img["id"])
            except Exception:
                continue
            images[img_id] = {**img, "id": img_id, "file_name": str(img.get("file_name", ""))}
        for ann in coco_blob.get("annotations", []):
            try:
                img_id = int(ann["image_id"])
            except Exception:
                continue
            ann_by_image.setdefault(img_id, []).append(ann)
    image_ids = list(images.keys())
    if not image_ids:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_training_no_images")
    rnd = random.Random(split_seed)
    rnd.shuffle(image_ids)
    total = len(image_ids)
    vp = max(0.0, min(float(val_percent), 0.9))
    val_count = int(total * vp)
    if val_count <= 0 and total > 1:
        val_count = 1
    if val_limit is not None and val_limit > 0:
        val_count = min(val_limit, val_count if val_count > 0 else val_limit, total - 1 if total > 1 else total)
    val_ids = image_ids[:val_count]
    train_ids = image_ids[val_count:]
    if train_limit is not None and train_limit > 0:
        train_ids = train_ids[:train_limit]
    if not train_ids or not val_ids:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_training_split_empty")
    split_root = (SAM3_JOB_ROOT / "splits" / job_id).resolve()
    split_root.parent.mkdir(parents=True, exist_ok=True)
    if split_root.exists():
        shutil.rmtree(split_root, ignore_errors=True)
    (split_root / "train" / "images").mkdir(parents=True, exist_ok=True)
    (split_root / "val" / "images").mkdir(parents=True, exist_ok=True)

    def _find_image_source(file_name: str) -> Optional[Path]:
        rel_path = Path(file_name)
        candidates = [
            dataset_root / rel_path,
            dataset_root / "train" / rel_path,
            dataset_root / "val" / rel_path,
            dataset_root / "train" / "images" / rel_path,
            dataset_root / "val" / "images" / rel_path,
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        if rel_path.is_absolute() and rel_path.exists():
            return rel_path
        return None

    def _write_split(target_ids: List[int], split_name: str) -> Tuple[str, int]:
        images_out: List[Dict[str, Any]] = []
        anns_out: List[Dict[str, Any]] = []
        for img_id in target_ids:
            info = images.get(img_id)
            if not info:
                continue
            file_name = info.get("file_name")
            if not file_name:
                continue
            src_path = _find_image_source(file_name)
            if src_path is None:
                continue
            dst_path = split_root / split_name / "images" / Path(file_name)
            _link_or_copy_file(src_path, dst_path)
            images_out.append(info)
            anns_out.extend(ann_by_image.get(img_id, []))
        ann_path = split_root / split_name / "_annotations.coco.json"
        _write_coco_annotations(
            ann_path,
            dataset_id=meta.get("id") or dataset_root.name,
            categories=categories,
            images=images_out,
            annotations=anns_out,
        )
        return str(ann_path), len(images_out)

    coco_train_new, train_count = _write_split(train_ids, "train")
    coco_val_new, val_count = _write_split(val_ids, "val")
    new_meta = {
        **meta,
        "dataset_root": str(split_root),
        "coco_train_json": coco_train_new,
        "coco_val_json": coco_val_new,
        "train_count": train_count,
        "val_count": val_count,
        "image_count": train_count + val_count,
        "signature": _compute_dir_signature(split_root),
        "source": meta.get("source", "resplit"),
    }
    _persist_sam3_dataset_metadata(split_root, new_meta)
    summary = (
        f"SAM3 split: {train_count} train / {val_count} val "
        f"(seed={split_seed}, val_percent={vp:.2f}, src={dataset_root}) -> {split_root}"
    )
    logger.info(summary)
    if log_messages is not None:
        log_messages.append(summary)
    return new_meta


def _plan_segmentation_build(request: SegmentationBuildRequest) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dataset_root = _resolve_sam3_or_qwen_dataset(request.source_dataset_id)
    source_meta = _load_qwen_dataset_metadata(dataset_root) or _load_sam3_dataset_metadata(dataset_root)
    if not source_meta:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="segmentation_source_metadata_missing")
    dataset_type = source_meta.get("type", "bbox")
    if dataset_type != "bbox":
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="segmentation_builder_requires_bbox")
    source_id = source_meta.get("id") or dataset_root.name
    suggested_name = f"{source_id}_seg"
    output_id = _safe_run_name(request.output_name, suggested_name)
    output_root = (SAM3_DATASET_ROOT / output_id).resolve()
    if not str(output_root).startswith(str(SAM3_DATASET_ROOT.resolve())):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="segmentation_output_path_invalid")
    if output_root.exists():
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="segmentation_output_exists")

    classes = source_meta.get("classes") or []
    context = source_meta.get("context") or source_meta.get("dataset_context") or ""
    source_signature = source_meta.get("signature") or _compute_dir_signature(dataset_root)
    planned_meta = {
        "id": output_id,
        "label": source_meta.get("label") or source_id,
        "type": "seg",
        "source": "segmentation_builder",
        "source_dataset_id": source_id,
        "source_dataset_root": str(dataset_root),
        "source_signature": source_signature,
        "generator_variant": request.sam_variant,
        "output_format": request.output_format,
        "classes": classes,
        "context": context,
        "created_at": time.time(),
    }
    planned_layout = {
        "dataset_root": str(output_root),
        "images_dir": str(output_root / "images"),
        "labels_dir": str(output_root / "labels"),
        "metadata_path": str(output_root / SAM3_DATASET_META_NAME),
        "log_dir": str(SEG_BUILDER_ROOT / "logs" / output_id),
    }
    return planned_meta, planned_layout


def _start_segmentation_build_job(request: SegmentationBuildRequest) -> SegmentationBuildJob:
    planned_meta, planned_layout = _plan_segmentation_build(request)
    job_id = str(uuid.uuid4())
    job = SegmentationBuildJob(
        job_id=job_id,
        status="queued",
        message="Queued",
        progress=0.0,
        config={
            "source_dataset_id": request.source_dataset_id,
            "sam_variant": request.sam_variant,
            "output_format": request.output_format,
            "planned_metadata": planned_meta,
            "planned_layout": planned_layout,
        },
    )
    with SEGMENTATION_BUILD_JOBS_LOCK:
        SEGMENTATION_BUILD_JOBS[job_id] = job

    def worker() -> None:
        try:
            _seg_job_update(job, status="running", progress=0.02, message="Preparing segmentation build…", error=None)
            source_meta = _resolve_sam3_dataset_meta(request.source_dataset_id)
            classes = source_meta.get("classes") or []
            if not classes:
                # Try to load from labelmap.txt directly.
                try:
                    labelmap_file = _resolve_sam3_or_qwen_dataset(request.source_dataset_id) / "labelmap.txt"
                    classes = _load_labelmap_file(labelmap_file)
                except Exception:
                    classes = []
            if not classes:
                raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="segmentation_builder_no_classes")
            dataset_root = Path(source_meta.get("dataset_root") or _resolve_sam3_or_qwen_dataset(request.source_dataset_id))
            labelmap_file = dataset_root / "labelmap.txt"
            if not labelmap_file.exists() and classes:
                # Backfill labelmap file if missing.
                try:
                    labelmap_file.write_text("\n".join(classes), encoding="utf-8")
                except Exception:
                    pass
            output_root = Path(planned_layout["dataset_root"]).resolve()
            train_out = output_root / "train"
            val_out = output_root / "val"
            (train_out / "images").mkdir(parents=True, exist_ok=True)
            (train_out / "labels").mkdir(parents=True, exist_ok=True)
            (val_out / "images").mkdir(parents=True, exist_ok=True)
            (val_out / "labels").mkdir(parents=True, exist_ok=True)
            # Copy/link labelmap.
            if labelmap_file.exists():
                shutil.copy2(labelmap_file, output_root / "labelmap.txt")
            splits = []
            image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

            def _find_image_for_label(labels_dir: Path, images_dir: Path, label_file: Path) -> Optional[Tuple[Path, Path]]:
                stem = label_file.stem
                for ext in image_exts:
                    candidate = images_dir / f"{stem}{ext}"
                    if candidate.exists():
                        try:
                            rel = candidate.relative_to(images_dir)
                        except Exception:
                            rel = Path(candidate.name)
                        return candidate, rel
                for candidate in images_dir.rglob(f"{stem}.*"):
                    if candidate.suffix.lower() in image_exts:
                        try:
                            rel = candidate.relative_to(images_dir)
                        except Exception:
                            rel = Path(candidate.name)
                        return candidate, rel
                return None

            image_uid = 1
            for split in ("train", "val"):
                images_dir = dataset_root / split / "images"
                labels_dir = dataset_root / split / "labels"
                if not images_dir.exists() or not labels_dir.exists():
                    continue
                entries = []
                for label_file in sorted(labels_dir.rglob("*.txt")):
                    match = _find_image_for_label(labels_dir, images_dir, label_file)
                    if match is None:
                        continue
                    image_path, rel_path = match
                    boxes = []
                    try:
                        with label_file.open("r", encoding="utf-8") as handle:
                            lines = [ln.strip() for ln in handle if ln.strip()]
                    except Exception:
                        continue
                    for ln in lines:
                        parts = ln.split()
                        if len(parts) < 5:
                            continue
                        try:
                            cls_idx = int(float(parts[0]))
                            cx = float(parts[1])
                            cy = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                        except (TypeError, ValueError):
                            continue
                        if classes and (cls_idx < 0 or cls_idx >= len(classes)):
                            continue
                        boxes.append({"class_idx": cls_idx, "bbox": (cx, cy, w, h)})
                    entries.append(
                        {
                            "label_file": label_file,
                            "image_path": image_path,
                            "rel_path": rel_path,
                            "boxes": boxes,
                            "split": split,
                            "image_id": image_uid,
                        }
                    )
                    image_uid += 1
                splits.append((split, entries))
            total_images = sum(len(e) for _, e in splits)
            if total_images == 0:
                raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="segmentation_builder_no_images")
            _seg_job_log(job, f"Queued {total_images} images for conversion using {request.sam_variant.upper()}")
            for split_name, entries in splits:
                _seg_job_log(job, f"{split_name}: {len(entries)} images")
            base_devices = _resolve_sam3_mining_devices() if request.sam_variant == "sam3" else _resolve_sam1_devices()
            expanded_devices: List[torch.device] = []
            per_dev = max(1, int(os.environ.get("SEG_BUILDER_WORKERS_PER_DEVICE", "1")))
            max_total_env = os.environ.get("SEG_BUILDER_MAX_WORKERS")
            max_total = None
            try:
                if max_total_env is not None:
                    max_total = max(1, int(max_total_env))
            except Exception:
                max_total = None
            for dev in base_devices:
                for _ in range(per_dev):
                    expanded_devices.append(dev)
            if max_total is not None:
                expanded_devices = expanded_devices[:max_total]
            if not expanded_devices:
                raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="segmentation_builder_no_devices")
            mining_pool = None
            sam1_workers: List[_Sam1SegWorker] = []
            try:
                if request.sam_variant == "sam3":
                    mining_pool = _Sam3MiningPool(expanded_devices)
                else:
                    for dev in expanded_devices:
                        sam1_workers.append(_Sam1SegWorker(dev))
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
            processed = 0
            simplify_eps = float(request.simplify_epsilon)
            mask_threshold = float(request.mask_threshold)
            min_threshold = float(request.score_threshold)
            max_results = int(max(1, request.max_results))
            min_area = float(request.min_size)
            progress_lock = threading.Lock()

            def _link_or_copy(src: Path, dst: Path) -> None:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    return
                try:
                    os.link(src, dst)
                except Exception:
                    shutil.copy2(src, dst)

            def _process_entry(entry: Dict[str, Any], worker: Any) -> None:
                nonlocal processed
                if job.cancel_event.is_set():
                    return
                image_path: Path = entry["image_path"]
                rel_path: Path = entry["rel_path"]
                split: str = entry["split"]
                boxes: List[Dict[str, Any]] = entry.get("boxes") or []
                tasks: List[Dict[str, Any]] = []
                try:
                    with Image.open(image_path) as im:
                        pil_img = im.convert("RGB")
                        width, height = pil_img.size
                        for idx, box in enumerate(boxes):
                            cx, cy, bw, bh = box["bbox"]
                            x1, y1, x2, y2 = _yolo_to_xyxy(width, height, (cx, cy, bw, bh))
                            tasks.append(
                                {
                                    "id": f"{rel_path}:{idx}",
                                    "type": "visual",
                                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                                    "class_idx": box["class_idx"],
                                    "fallback_poly": [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                                }
                            )
                        if not tasks:
                            outputs = {}
                        else:
                            outputs = worker.process_image(
                                image_id=entry.get("image_id", 0),
                                pil_img=pil_img,
                                tasks=tasks,
                                min_threshold=min_threshold,
                                mask_threshold=mask_threshold,
                                max_results=max_results,
                                min_size=min_area,
                                simplify=simplify_eps,
                                return_masks=True,
                            ) or {}
                            label_lines = []
                            for task in tasks:
                                task_id = task["id"]
                                class_idx = task["class_idx"]
                                fallback = task["fallback_poly"]
                                dets = outputs.get(task_id) or []
                                best = None
                                if dets:
                                    best = max(dets, key=lambda d: d.get("score") or 0.0)
                                mask_arr = None
                                best_score = best.get("score") if best else None
                                if best:
                                    mask_arr = best.get("mask_array")
                                    if mask_arr is None and best.get("mask"):
                                        mask_arr = decode_binary_mask(best.get("mask"))
                                polygon = mask_to_polygon(mask_arr, simplify_eps) if mask_arr is not None else []
                                if best_score is None or best_score < min_threshold:
                                    polygon = []
                                if len(polygon) < 3:
                                    polygon = fallback
                                coords: List[float] = []
                                for x, y in polygon:
                                    coords.extend(
                                        [
                                            max(0.0, min(1.0, x / width)),
                                            max(0.0, min(1.0, y / height)),
                                        ]
                                    )
                                if len(coords) >= 6:
                                    label_lines.append(f"{class_idx} " + " ".join(f"{v:.6f}" for v in coords))
                        dest_labels = (train_out if split == "train" else val_out) / "labels" / f"{rel_path.stem}.txt"
                        dest_images = (train_out if split == "train" else val_out) / "images" / rel_path
                        dest_labels.parent.mkdir(parents=True, exist_ok=True)
                        dest_images.parent.mkdir(parents=True, exist_ok=True)
                        _link_or_copy(image_path, dest_images)
                        dest_labels.write_text("\n".join(label_lines), encoding="utf-8")
                finally:
                    with progress_lock:
                        processed += 1
                        progress_val = min(1.0, 0.05 + 0.9 * (processed / max(total_images, 1)))
                    _seg_job_update(
                        job,
                        progress=progress_val,
                        message=f"Processed {processed}/{total_images} images ({progress_val*100:.1f}%)",
                        log_message=False,
                    )

            # Dispatch over workers
            try:
                workers_list = mining_pool.workers if mining_pool is not None else sam1_workers
                if not workers_list:
                    raise RuntimeError("segmentation_builder_no_workers")
                with ThreadPoolExecutor(max_workers=max(1, len(workers_list))) as executor:
                    futures = []
                    task_idx = 0
                    for _, entries in splits:
                        for entry in entries:
                            worker = workers_list[task_idx % len(workers_list)]
                            futures.append(executor.submit(_process_entry, entry, worker))
                            task_idx += 1
                    for fut in as_completed(futures):
                        if job.cancel_event.is_set():
                            break
                        try:
                            fut.result()
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Segmentation build worker failed: %s", exc)
            finally:
                try:
                    if mining_pool is not None:
                        mining_pool.close()
                except Exception:
                    pass
                if sam1_workers:
                    for worker in sam1_workers:
                        try:
                            worker.close()
                        except Exception:
                            pass
            if job.cancel_event.is_set():
                _seg_job_update(job, status="cancelled", message="Cancelled", progress=job.progress)
                return
            _seg_job_log(job, "Converting output to COCO…")
            try:
                coco_meta = _convert_yolo_dataset_to_coco(output_root)
            except Exception as exc:  # noqa: BLE001
                _seg_job_update(job, status="failed", message="COCO conversion failed", error=str(exc))
                return
            result_meta = _load_sam3_dataset_metadata(output_root) or coco_meta or planned_meta
            _seg_job_update(
                job,
                status="completed",
                progress=1.0,
                message="Segmentation build complete.",
                result={
                    "planned_metadata": planned_meta,
                    "output_dataset_id": result_meta.get("id") if isinstance(result_meta, dict) else planned_meta.get("id"),
                    "output_root": str(output_root),
                    "classes": classes,
                    "train_count": len(next((e for s, e in splits if s == "train"), [])),
                    "val_count": len(next((e for s, e in splits if s == "val"), [])),
                },
            )
        except HTTPException as exc:
            _seg_job_update(job, status="failed", message=str(exc.detail), error=str(exc.detail))
        except Exception as exc:  # noqa: BLE001
            _seg_job_update(job, status="failed", message=str(exc), error=str(exc))

    threading.Thread(target=worker, daemon=True, name=f"seg-build-{job_id[:8]}").start()
    return job


def _safe_run_name(desired: Optional[str], fallback: str) -> str:
    name = desired or fallback
    return re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("_") or fallback


def _latest_checkpoint_in_dir(checkpoint_dir: Path) -> Optional[str]:
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(
        checkpoint_dir.glob("*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])
    return None


def _save_sam3_config(cfg: OmegaConf, job_id: str) -> Tuple[str, Path]:
    SAM3_GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_file = SAM3_GENERATED_CONFIG_DIR / f"{job_id}.yaml"
    yaml_text = OmegaConf.to_yaml(cfg)
    config_file.write_text("# @package _global_\n" + yaml_text, encoding="utf-8")
    return f"configs/generated/{config_file.name}", config_file


def _start_sam3_training_worker(job: Sam3TrainingJob, cfg: OmegaConf, num_gpus: int) -> None:
    def worker():
        proc: Optional[subprocess.Popen] = None
        tail_logs: deque[str] = deque(maxlen=50)
        max_epochs = max(1, int(getattr(cfg.trainer, "max_epochs", 1) or 1))
        # Attempt to track steps per epoch from the config if present
        steps_per_epoch = None
        try:
            steps_per_epoch = int(cfg.scratch.target_epoch_size) if getattr(cfg.scratch, "target_epoch_size", None) else None
        except Exception:
            steps_per_epoch = None
        try:
            _sam3_job_update(job, status="running", progress=0.05, message="Preparing SAM3 training job ...")
            config_name, config_file = _save_sam3_config(cfg, job.job_id)
            script_path = SAM3_PACKAGE_ROOT / "train" / "train.py"
            cmd = [sys.executable, str(script_path), "-c", config_name, "--use-cluster", "0"]
            if num_gpus is not None:
                cmd.extend(["--num-gpus", str(num_gpus)])
            env = os.environ.copy()
            existing_py = env.get("PYTHONPATH", "")
            py_root = f"{SAM3_VENDOR_ROOT}:{SAM3_REPO_ROOT}"
            env["PYTHONPATH"] = f"{py_root}:{existing_py}" if existing_py else py_root
            env.setdefault("CUDA_LAUNCH_BLOCKING", "1")
            env.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
            env.setdefault("NCCL_DEBUG", "INFO")
            # Enable runtime monkeypatches (loaded via sitecustomize.py) to keep vendor tree untouched.
            env.setdefault("SAM3_MONKEYPATCH", "1")
            proc = subprocess.Popen(
                cmd,
                cwd=str(SAM3_VENDOR_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            job.process = proc
            _sam3_job_log(job, f"Spawned {' '.join(cmd)}")
            while True:
                if proc.stdout is None:
                    break
                line = proc.stdout.readline()
                if line == "" and proc.poll() is not None:
                    break
                if not line:
                    continue
                if job.cancel_event.is_set() and proc.poll() is None:
                    proc.terminate()
                    _sam3_job_update(job, status="cancelling", message="Cancellation requested ...")
                    continue
                cleaned = line.rstrip("\n")
                tail_logs.append(cleaned)
                _sam3_job_log(job, cleaned)
                if "sam3-balance" in cleaned.lower() or cleaned.startswith("[sam3-balance]"):
                    job.result = job.result or {}
                    job.result["balance_info"] = cleaned
                try:
                    match = re.search(r"Train Epoch:\s*\[(\d+)\]\[\s*(\d+)\s*/\s*(\d+)\]", cleaned)
                    val_match = re.search(r"Val Epoch:\s*\[(\d+)\]\[\s*(\d+)\s*/\s*(\d+)\]", cleaned)
                    if val_match:
                        val_epoch_idx = int(val_match.group(1))
                        val_step_idx = int(val_match.group(2))
                        val_total_steps = max(1, int(val_match.group(3)))
                        val_frac = max(0.0, min(1.0, val_step_idx / val_total_steps))
                        prog_val = max(job.progress or 0.0, min(0.99, 0.9 + 0.1 * val_frac))
                        _sam3_job_update(
                            job,
                            progress=prog_val,
                            message=f"Validation running ({val_step_idx}/{val_total_steps})",
                            log_message=False,
                        )
                        _sam3_job_append_metric(
                            job,
                            {
                                "phase": "val",
                                "val_step": val_step_idx,
                                "val_total": val_total_steps,
                                "epoch": val_epoch_idx + 1,
                                "total_epochs": max_epochs,
                            },
                        )
                    if match:
                        epoch_idx = int(match.group(1))
                        step_idx = int(match.group(2))
                        total_steps = max(1, int(match.group(3)))
                        # Prefer log-reported total steps; fall back to config target if present
                        steps_in_epoch = total_steps or steps_per_epoch or total_steps
                        frac_epoch = (step_idx / steps_in_epoch) if steps_in_epoch else 0.0
                        frac = (epoch_idx + frac_epoch) / max_epochs
                        prog_val = max(0.05, min(0.99, frac))
                        _sam3_job_update(job, progress=prog_val, log_message=False)
                    loss_match = re.search(
                        r"Losses\/train_all_loss:\s*(?:(?:last|batch)=)?([0-9.+-eE]+)(?:.*?(?:avg\d*=?\s*([0-9.+-eE]+)|\(\s*([0-9.+-eE]+)\s*\)))?",
                        cleaned,
                    )
                    if loss_match and match:
                        instant = float(loss_match.group(1))
                        avg_loss = None
                        if loss_match.group(2):
                            avg_loss = float(loss_match.group(2))
                        elif loss_match.group(3):
                            avg_loss = float(loss_match.group(3))
                        total_steps = max(1, int(match.group(3)))
                        steps_in_epoch = total_steps or steps_per_epoch or total_steps
                        global_step = epoch_idx * steps_in_epoch + step_idx
                        metric_payload = {
                            "phase": "train",
                            "train_loss_batch": instant,
                            "train_loss_avg10": avg_loss,
                            "batch": step_idx,
                            "batches_per_epoch": steps_in_epoch,
                            "epoch": epoch_idx + 1,
                            "total_epochs": max_epochs,
                            "step": global_step,
                            "timestamp": time.time(),
                        }
                        _sam3_job_append_metric(job, metric_payload)
                    if "Meters:" in cleaned and "coco_eval_bbox_AP" in cleaned:
                        try:
                            # Extract key/value pairs like '...': np.float64(0.123)
                            pairs = re.findall(r"'([^']+)':\s*np\.float64\(([0-9.eE+-]+)\)", cleaned)
                            meter_map = {k: float(v) for k, v in pairs}
                            epoch_meta = re.search(r"'Trainer/epoch':\s*([0-9]+)", cleaned)
                            epoch_val = int(epoch_meta.group(1)) + 1 if epoch_meta else None
                            val_payload: Dict[str, Any] = {
                                "phase": "val",
                                "timestamp": time.time(),
                            }
                            if epoch_val is not None:
                                val_payload["epoch"] = epoch_val
                            # Pick the first coco_eval_bbox_* metrics if present.
                            for key, field in [
                                ("coco_eval_bbox_AP", "coco_ap"),
                                ("coco_eval_bbox_AP_50", "coco_ap50"),
                                ("coco_eval_bbox_AP_75", "coco_ap75"),
                                ("coco_eval_bbox_AR_maxDets@10", "coco_ar10"),
                                ("coco_eval_bbox_AR_maxDets@100", "coco_ar100"),
                            ]:
                                for meter_key, meter_val in meter_map.items():
                                    if meter_key.endswith(key):
                                        val_payload[field] = meter_val
                                        break
                            _sam3_job_append_metric(job, val_payload)
                        except Exception:
                            pass
                except Exception:
                    pass
                _sam3_job_update(job, message=cleaned[-200:], log_message=False)
            retcode = proc.wait() if proc else 1
            if job.cancel_event.is_set():
                _sam3_job_update(job, status="cancelled", message="Training cancelled")
                return
            if retcode != 0:
                sig_note = ""
                if retcode < 0:
                    sig_num = -retcode
                    try:
                        sig_name = signal.Signals(sig_num).name
                    except Exception:
                        sig_name = f"SIG{sig_num}"
                    sig_desc = signal.strsignal(sig_num) or sig_name
                    sig_note = f" (signal {sig_num}: {sig_desc})"
                tail_text = "\n".join(tail_logs)
                _sam3_job_update(
                    job,
                    status="failed",
                    message=f"Training failed (exit {retcode}{sig_note})",
                    error=f"exit_code:{retcode}{sig_note}\nlast_logs:\n{tail_text}",
                )
                return
            log_dir = Path(cfg.paths.experiment_log_dir)
            checkpoint_dir = log_dir / "checkpoints"
            latest_ckpt = _latest_checkpoint_in_dir(checkpoint_dir)
            seg_head = bool(getattr(cfg.scratch, "enable_segmentation_head", getattr(cfg.scratch, "enable_segmentation", True)))
            load_seg = bool(getattr(cfg.scratch, "load_segmentation", seg_head))
            result_payload = {
                "experiment_log_dir": str(log_dir),
                "checkpoint": latest_ckpt,
                "config_path": str(config_file),
                "enable_segmentation": seg_head,
                "enable_segmentation_head": seg_head,
                "load_segmentation": load_seg,
            }
            _sam3_job_update(job, status="succeeded", message="Training complete", progress=1.0, result=result_payload)
        except Exception as exc:  # noqa: BLE001
            _sam3_job_update(job, status="failed", message="Training crashed", error=str(exc))
        finally:
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass

    thread = threading.Thread(target=worker, name=f"sam3-train-{job.job_id}", daemon=True)
    thread.start()


def _get_sam3_job(job_id: str) -> Sam3TrainingJob:
    with SAM3_TRAINING_JOBS_LOCK:
        job = SAM3_TRAINING_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="sam3_job_not_found")
        return job


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
    zip_path = dest_dir / "payload.zip"
    try:
        await _write_upload_file(
            file,
            zip_path,
            max_bytes=QWEN_DATASET_ZIP_MAX_BYTES,
            quota_root=dest_dir,
            quota_limit=QWEN_DATASET_ZIP_MAX_BYTES,
            allow_overwrite=True,
        )
        with zipfile.ZipFile(zip_path) as archive:
            _safe_extract_zip(
                archive,
                dest_dir,
                max_bytes_per_file=QWEN_DATASET_ZIP_MAX_BYTES,
                total_quota_bytes=QWEN_DATASET_ZIP_MAX_BYTES,
            )
    except zipfile.BadZipFile as exc:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"dataset_invalid_zip:{exc}") from exc
    except HTTPException:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"dataset_extract_failed:{exc}") from exc
    finally:
        try:
            zip_path.unlink()
        except Exception:
            pass
    for split in ("train", "val"):
        annotations = dest_dir / split / "annotations.jsonl"
        if not annotations.exists():
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"dataset_missing_annotations:{split}")
    # Backfill minimal metadata so downstream listings have type information.
    metadata = _load_qwen_dataset_metadata(dest_dir) or {}
    metadata.setdefault("id", safe_token)
    metadata.setdefault("label", safe_token)
    metadata.setdefault("created_at", time.time())
    metadata.setdefault("type", "bbox")
    _persist_qwen_dataset_metadata(dest_dir, metadata)
    return {"dataset_root": str(dest_dir), "run_name": safe_token}


async def _save_upload_file(
    upload: UploadFile,
    root: Path,
    *,
    max_bytes: Optional[int] = None,
    quota_root: Optional[Path] = None,
    quota_limit: Optional[int] = None,
) -> Path:
    rel_path = _normalise_relative_path(upload.filename)
    dest = (root / rel_path).resolve()
    if not str(dest).startswith(str(root.resolve())):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="invalid_relative_path")
    if dest.exists():
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="upload_exists")
    await _write_upload_file(
        upload,
        dest,
        max_bytes=max_bytes,
        quota_root=quota_root or root,
        quota_limit=quota_limit,
    )
    return dest


def _validate_upload_size(upload: UploadFile, *, max_bytes: int = BASE64_IMAGE_MAX_BYTES) -> None:
    if not max_bytes:
        return
    try:
        size = upload.size  # Starlette UploadFile may have size attr
    except Exception:
        size = None
    if size is not None and size > max_bytes:
        raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="upload_too_large")


def _validate_upload_extension(filename: str, allowed_exts: set[str], detail: str) -> None:
    suffix = Path(filename).suffix.lower()
    if allowed_exts and suffix not in allowed_exts:
        raise HTTPException(status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=detail)


async def _save_asset(
    upload: UploadFile,
    *,
    subdir: str,
    allowed_exts: Optional[set[str]] = None,
    max_bytes: Optional[int] = None,
    quota_bytes: Optional[int] = None,
) -> str:
    dest_dir = UPLOAD_ROOT / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    rel_name = Path(upload.filename or f"asset_{uuid.uuid4().hex}").name
    dest_path = dest_dir / rel_name
    if allowed_exts:
        _validate_upload_extension(rel_name, allowed_exts, "upload_extension_not_allowed")
    if dest_path.exists():
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="upload_exists")
    _validate_upload_size(upload, max_bytes=max_bytes or ASSET_MAX_BYTES)
    await _write_upload_file(
        upload,
        dest_path,
        max_bytes=max_bytes or ASSET_MAX_BYTES,
        quota_root=dest_dir,
        quota_limit=quota_bytes or ASSET_UPLOAD_QUOTA_BYTES,
    )
    return str(dest_path.resolve())


async def _write_upload_file(
    upload: UploadFile,
    dest: Path,
    *,
    max_bytes: Optional[int] = None,
    quota_root: Optional[Path] = None,
    quota_limit: Optional[int] = None,
    allow_overwrite: bool = False,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not allow_overwrite:
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="upload_exists")
    written = 0
    existing = _dir_size_bytes(quota_root) if quota_root and quota_limit else 0
    with dest.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if max_bytes and written > max_bytes:
                handle.close()
                try:
                    dest.unlink()
                except Exception:
                    pass
                raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="upload_too_large")
            if quota_root and quota_limit and existing + written > quota_limit:
                handle.close()
                try:
                    dest.unlink()
                except Exception:
                    pass
                raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="upload_quota_exceeded")
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
        "clip_error": clip_last_error,
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


def encode_binary_mask(mask: np.ndarray) -> Optional[Dict[str, Any]]:
    try:
        mask_arr = np.asarray(mask)
    except Exception:
        return None
    if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
        mask_arr = mask_arr[0]
    if mask_arr.ndim == 3 and mask_arr.shape[-1] == 1:
        mask_arr = mask_arr[..., 0]
    if mask_arr.ndim != 2:
        return None
    mask_bool = mask_arr.astype(bool)
    height, width = mask_bool.shape
    packed = np.packbits(mask_bool.astype(np.uint8), axis=None)
    try:
        packed_bytes = packed.tobytes()
    except Exception:
        return None
    if MASK_ENCODE_MAX_BYTES > 0 and len(packed_bytes) > MASK_ENCODE_MAX_BYTES:
        return None
    try:
        encoded = base64.b64encode(packed_bytes).decode("ascii")
    except Exception:
        return None
    return {"size": [int(height), int(width)], "counts": encoded}


def _prune_detections_for_response(dets: List[Any], warnings: Optional[List[str]] = None) -> List[Any]:
    """Clamp response payload size by limiting detection count and mask payloads."""
    if not dets:
        return dets
    limited: List[Any] = list(dets[: MAX_RESPONSE_DETECTIONS]) if MAX_RESPONSE_DETECTIONS > 0 else list(dets)
    if warnings is not None and MAX_RESPONSE_DETECTIONS > 0 and len(dets) > MAX_RESPONSE_DETECTIONS:
        warnings.append("detections_pruned")
    mask_budget = MAX_RESPONSE_MASKS if MAX_RESPONSE_MASKS > 0 else None
    masks_used = 0
    for det in limited:
        mask_val = getattr(det, "mask", None)
        if mask_val is not None and mask_budget is not None:
            if masks_used >= mask_budget:
                try:
                    det.mask = None  # type: ignore[attr-defined]
                except Exception:
                    pass
                else:
                    if warnings is not None:
                        warnings.append("masks_pruned")
            else:
                masks_used += 1
    return limited


def decode_binary_mask(payload: Dict[str, Any]) -> Optional[np.ndarray]:
    if not payload:
        return None
    counts = payload.get("counts")
    size = payload.get("size") or []
    if not counts or len(size) != 2:
        return None
    try:
        packed = np.frombuffer(base64.b64decode(counts), dtype=np.uint8)
        bits = np.unpackbits(packed)[: int(size[0]) * int(size[1])]
        return bits.reshape(int(size[0]), int(size[1]))
    except Exception:
        return None


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer–Douglas–Peucker simplification for 2D points."""
    if points.shape[0] < 3 or epsilon <= 0:
        return points

    def _perp_dist(pt, start, end):
        if np.allclose(start, end):
            return np.linalg.norm(pt - start)
        return np.abs(np.cross(end - start, start - pt)) / np.linalg.norm(end - start)

    start_pt = points[0]
    end_pt = points[-1]
    dmax = 0.0
    idx = 0
    for i in range(1, len(points) - 1):
        d = _perp_dist(points[i], start_pt, end_pt)
        if d > dmax:
            idx = i
            dmax = d
    if dmax > epsilon:
        rec1 = _rdp(points[: idx + 1], epsilon)
        rec2 = _rdp(points[idx:], epsilon)
        return np.concatenate((rec1[:-1], rec2), axis=0)
    return np.array([start_pt, end_pt])


def mask_to_polygon(mask: np.ndarray, simplify_epsilon: float) -> List[Tuple[float, float]]:
    """Extract a coarse polygon outline from a binary mask."""
    try:
        mask_arr = np.asarray(mask).astype(bool)
    except Exception:
        return []
    if mask_arr.ndim != 2 or not mask_arr.any():
        return []
    coords = np.argwhere(mask_arr)  # y, x
    if coords.shape[0] < 3:
        return []
    points = np.stack([coords[:, 1], coords[:, 0]], axis=1)  # x, y
    hull_pts = points
    if ConvexHull is not None:
        try:
            hull = ConvexHull(points)
            hull_pts = points[hull.vertices]
        except Exception:
            hull_pts = points
    if simplify_epsilon and simplify_epsilon > 0:
        hull_pts = _rdp(hull_pts, simplify_epsilon)
    # Ensure at least 3 points.
    if hull_pts.shape[0] < 3:
        # Fallback to simple bounding box.
        xs, ys = points[:, 0], points[:, 1]
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        hull_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return [(float(x), float(y)) for x, y in hull_pts]

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

    pil_img, _ = _decode_image_base64(payload.image_base64)
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


def _list_clip_classifiers() -> List[Dict[str, Any]]:
    """List classifier heads available for CLIP filtering (typically trained via the CLIP training tab)."""
    root = (UPLOAD_ROOT / "classifiers").resolve()
    classifiers: List[Dict[str, Any]] = []
    if not root.exists():
        return classifiers

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in CLASSIFIER_ALLOWED_EXTS:
            continue
        if path.name.endswith(".meta.pkl"):
            continue
        if not _path_is_within_root(path.resolve(), root):
            continue

        entry: Dict[str, Any] = {
            "filename": path.name,
            "path": str(path.resolve()),
            "rel_path": str(path.relative_to(root)),
        }

        meta_path = os.path.splitext(str(path))[0] + ".meta.pkl"
        if os.path.exists(meta_path):
            try:
                meta_obj = joblib.load(meta_path)
                if isinstance(meta_obj, dict):
                    entry["clip_model"] = meta_obj.get("clip_model")
                    entry["solver"] = meta_obj.get("solver")
                    entry["embedding_dim"] = meta_obj.get("embedding_dim")
                    entry["n_samples_train"] = meta_obj.get("n_samples_train")
                    entry["n_samples_test"] = meta_obj.get("n_samples_test")
            except Exception:
                pass

        try:
            clf_obj = joblib.load(str(path))
            classes_raw = getattr(clf_obj, "classes_", None)
            if classes_raw is not None:
                entry["classes"] = [str(c) for c in list(classes_raw)]
                entry["n_classes"] = len(entry["classes"])
            coef = getattr(clf_obj, "coef_", None)
            if coef is not None and hasattr(coef, "shape") and len(getattr(coef, "shape", [])) >= 2:
                entry["embedding_dim"] = int(coef.shape[1])
        except Exception as exc:  # noqa: BLE001
            entry["load_error"] = str(exc)

        try:
            entry["modified_at"] = path.stat().st_mtime
        except Exception:
            entry["modified_at"] = None
        classifiers.append(entry)

    classifiers.sort(key=lambda c: (c.get("modified_at") or 0), reverse=True)
    return classifiers


@app.get("/clip/classifiers")
def list_clip_classifiers():
    return _list_clip_classifiers()


@app.post("/fs/upload_classifier")
async def upload_classifier(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="filename_required")
    saved_path = await _save_asset(
        file,
        subdir="classifiers",
        allowed_exts=CLASSIFIER_ALLOWED_EXTS,
        max_bytes=ASSET_MAX_BYTES,
        quota_bytes=ASSET_UPLOAD_QUOTA_BYTES,
    )
    return {"path": saved_path}


@app.post("/fs/upload_labelmap")
async def upload_labelmap(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="filename_required")
    saved_path = await _save_asset(
        file,
        subdir="labelmaps",
        allowed_exts=LABELMAP_ALLOWED_EXTS,
        max_bytes=ASSET_MAX_BYTES,
        quota_bytes=ASSET_UPLOAD_QUOTA_BYTES,
    )
    return {"path": saved_path}


@app.get("/fs/select_directory")
def select_directory(request: Request, initial: str = "."):
    if not FS_DIALOG_ENABLED:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="fs_dialog_disabled")
    client_host = request.client.host if request and request.client else None
    if not FS_DIALOG_ALLOW_REMOTE and client_host not in {None, "127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="fs_dialog_remote_not_allowed")
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
        try:
            Path(abs_path).mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"output_dir_missing:{abs_path}") from exc
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


def _load_labelmap_simple(path: Optional[str]) -> List[str]:
    if not path:
        return []
    try:
        classes = _load_labelmap_file(Path(path))
        return classes
    except Exception:
        return []


def _validate_clip_dataset(inputs: Dict[str, str]) -> Dict[str, Any]:
    """
    Light validation of staged CLIP dataset to fail fast before launching a job.
    Expects keys: images_dir, labels_dir, optional labelmap_path.
    """
    images_dir = inputs.get("images_dir")
    labels_dir = inputs.get("labels_dir")
    labelmap_path = inputs.get("labelmap_path")
    if not images_dir or not labels_dir:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="clip_dataset_missing_paths")
    img_root = Path(images_dir)
    lbl_root = Path(labels_dir)
    if not img_root.is_dir() or not lbl_root.is_dir():
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="clip_dataset_missing_paths")
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_files: List[Path] = []
    for p in img_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in valid_exts:
            image_files.append(p)
    if not image_files:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="clip_images_missing")
    label_files = [p for p in lbl_root.rglob("*") if p.is_file() and p.suffix.lower() == ".txt"]
    if not label_files:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="clip_labels_missing")
    labelmap = _load_labelmap_simple(labelmap_path)
    max_cid = -1
    box_count = 0
    for lf in label_files:
        try:
            for line in lf.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                except Exception:
                    continue
                max_cid = max(max_cid, cid)
                box_count += 1
        except Exception:
            continue
    if box_count == 0:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="clip_labels_empty")
    if labelmap and max_cid >= len(labelmap):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="clip_labelmap_class_mismatch")
    return {
        "images": len(image_files),
        "labels": len(label_files),
        "boxes": box_count,
        "labelmap_classes": len(labelmap),
    }


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
            await _save_upload_file(
                upload,
                Path(images_dir),
                max_bytes=CLIP_TRAIN_UPLOAD_MAX_BYTES,
                quota_root=Path(temp_root),
                quota_limit=CLIP_TRAIN_UPLOAD_QUOTA_BYTES,
            )

        for upload in labels or []:
            await _save_upload_file(
                upload,
                Path(labels_dir),
                max_bytes=CLIP_TRAIN_UPLOAD_MAX_BYTES,
                quota_root=Path(temp_root),
                quota_limit=CLIP_TRAIN_UPLOAD_QUOTA_BYTES,
            )

    labelmap_path = None
    if labelmap_path_native:
        labelmap_path = os.path.abspath(labelmap_path_native)
        if not os.path.isfile(labelmap_path):
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="labelmap_not_found")
    elif labelmap is not None:
        if temp_root is None:
            temp_root = tempfile.mkdtemp(prefix="clip_train_")
        labelmap_path = str(
            await _save_upload_file(
                labelmap,
                Path(temp_root),
                max_bytes=ASSET_MAX_BYTES,
                quota_root=Path(temp_root),
                quota_limit=CLIP_TRAIN_UPLOAD_QUOTA_BYTES,
            )
        )

    job_id = uuid.uuid4().hex
    if images_dir is None or labels_dir is None:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="dataset_paths_unresolved")
    # Fail fast on obviously invalid staged datasets.
    _validate_clip_dataset({"images_dir": images_dir, "labels_dir": labels_dir, "labelmap_path": labelmap_path})
    logger.info("Starting training job %s (clip=%s, native_paths=%s)", job_id[:8], clip_model_name, use_native_paths)
    if staged_temp_dir:
        temp_root = os.path.abspath(staged_temp_dir)
    job = ClipTrainingJob(job_id=job_id, temp_dir=temp_root, images_dir=images_dir, labels_dir=labels_dir, labelmap_path=labelmap_path)
    job_message = "Job queued (native paths)" if use_native_paths else "Job queued (upload staging)"
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

    extras = [solver_name]
    if reuse_embeddings_flag:
        extras.append("cache")
    if hard_example_flag:
        extras.append(f"hard({hard_mis_weight_f:.1f}/{hard_low_conf_weight_f:.1f})")
    job_message += f" [{', '.join(extras)}]"
    _job_log(job, job_message)

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
    _prune_job_registry(TRAINING_JOBS, TRAINING_JOBS_LOCK)
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


def _build_sam3_config(
    payload: Sam3TrainRequest,
    meta: Dict[str, Any],
    job_id: str,
    job_logs: Optional[List[str]] = None,
) -> Tuple[OmegaConf, int]:
    dataset_root = Path(meta.get("dataset_root") or SAM3_DATASET_ROOT)
    val_percent = payload.val_percent if payload.val_percent is not None else 0.3
    split_seed = int(payload.split_seed) if payload.split_seed is not None else 42
    random_split = payload.random_split if payload.random_split is not None else True
    train_limit = int(payload.train_limit) if payload.train_limit is not None and payload.train_limit > 0 else None
    val_limit = int(payload.val_limit) if payload.val_limit is not None and payload.val_limit > 0 else None
    meta = _prepare_sam3_training_split(
        dataset_root,
        meta,
        job_id,
        random_split=random_split,
        val_percent=val_percent,
        split_seed=split_seed,
        train_limit=train_limit,
        val_limit=val_limit,
        log_messages=job_logs,
    )
    cfg = OmegaConf.load(str(SAM3_CONFIG_TEMPLATE))
    if not hasattr(cfg.scratch, "enable_segmentation_head"):
        cfg.scratch.enable_segmentation_head = True
    if not hasattr(cfg.scratch, "load_segmentation"):
        cfg.scratch.load_segmentation = False
    train_ann = Path(meta["coco_train_json"]).resolve()
    val_ann = Path(meta["coco_val_json"]).resolve()
    cfg.paths.train_img_folder = str(train_ann.parent)
    cfg.paths.train_ann_file = str(train_ann)
    cfg.paths.val_img_folder = str(val_ann.parent)
    cfg.paths.val_ann_file = str(val_ann)
    run_name = _safe_run_name(payload.run_name, f"sam3_run_{job_id}")
    exp_dir = Path(payload.experiment_log_dir) if payload.experiment_log_dir else (SAM3_JOB_ROOT / run_name)
    cfg.paths.experiment_log_dir = str(exp_dir.resolve())
    cfg.paths.bpe_path = str(SAM3_BPE_PATH)
    cfg.launcher.experiment_log_dir = cfg.paths.experiment_log_dir
    cfg.launcher.gpus_per_node = max(1, int(payload.num_gpus or cfg.launcher.gpus_per_node or 1))
    cfg.trainer.max_epochs = int(payload.max_epochs) if payload.max_epochs is not None else cfg.trainer.max_epochs
    cfg.trainer.val_epoch_freq = int(payload.val_epoch_freq) if payload.val_epoch_freq is not None else cfg.trainer.val_epoch_freq
    cfg.scratch.target_epoch_size = int(payload.target_epoch_size) if payload.target_epoch_size is not None else cfg.scratch.target_epoch_size
    dataset_type = meta.get("type", "bbox")
    seg_head_requested = payload.enable_segmentation_head
    train_seg_requested = payload.train_segmentation
    default_seg = dataset_type == "seg"
    enable_seg_head = bool(seg_head_requested) if seg_head_requested is not None else (bool(cfg.scratch.enable_segmentation_head) or default_seg)
    train_segmentation = bool(train_seg_requested) if train_seg_requested is not None else (bool(cfg.scratch.load_segmentation) or default_seg)
    cfg.scratch.enable_segmentation_head = enable_seg_head or train_segmentation
    cfg.scratch.load_segmentation = train_segmentation
    # Keep legacy flag aligned with head presence so downstream activation sees the capability.
    cfg.scratch.enable_segmentation = cfg.scratch.enable_segmentation_head
    if payload.resolution is not None:
        cfg.scratch.resolution = int(payload.resolution)
    if payload.lr_scale is not None:
        cfg.scratch.lr_scale = float(payload.lr_scale)
    if payload.gradient_accumulation_steps is not None:
        cfg.scratch.gradient_accumulation_steps = int(payload.gradient_accumulation_steps)
    cfg.trainer.gradient_accumulation_steps = cfg.scratch.gradient_accumulation_steps
    if cfg.trainer.gradient_accumulation_steps and cfg.trainer.gradient_accumulation_steps > 1:
        try:
            train_collate = cfg.trainer.data.train.collate_fn
            train_collate._target_ = "sam3.train.data.collator.collate_fn_api_with_chunking"
            train_collate.num_chunks = int(cfg.trainer.gradient_accumulation_steps)
            train_collate._partial_ = True
            if not hasattr(train_collate, "repeats"):
                train_collate.repeats = cfg.scratch.hybrid_repeats
        except Exception:
            pass
    if payload.scheduler_warmup is not None:
        cfg.scratch.scheduler_warmup = int(payload.scheduler_warmup)
    if payload.scheduler_timescale is not None:
        cfg.scratch.scheduler_timescale = int(payload.scheduler_timescale)
    if payload.train_batch_size is not None:
        cfg.scratch.train_batch_size = int(payload.train_batch_size)
    if payload.val_batch_size is not None:
        cfg.scratch.val_batch_size = int(payload.val_batch_size)
    if payload.num_train_workers is not None:
        cfg.scratch.num_train_workers = int(payload.num_train_workers)
    if payload.num_val_workers is not None:
        cfg.scratch.num_val_workers = int(payload.num_val_workers)
    if payload.enable_inst_interactivity is not None:
        cfg.scratch.enable_inst_interactivity = bool(payload.enable_inst_interactivity)
    if payload.train_limit is not None:
        cfg.dataset.num_images = int(payload.train_limit)
    elif payload.target_epoch_size is not None:
        try:
            batches = max(1, int(payload.target_epoch_size))
            batch_size = int(payload.train_batch_size) if payload.train_batch_size is not None else int(cfg.scratch.train_batch_size)
            cfg.dataset.num_images = max(1, batches * batch_size)
        except Exception:
            pass
    if payload.val_limit is not None:
        try:
            val_limit = max(1, int(payload.val_limit))
            cfg.dataset.val_num_images = val_limit
            if hasattr(cfg, "trainer") and hasattr(cfg.trainer, "data") and hasattr(cfg.trainer.data, "val"):
                if hasattr(cfg.trainer.data.val, "dataset"):
                    cfg.trainer.data.val.dataset.limit_ids = val_limit
        except Exception:
            pass
    if payload.log_every_batch:
        try:
            cfg.trainer.logging.log_freq = 1
        except Exception:
            pass
    elif payload.log_freq is not None and "logging" in cfg.trainer:
        cfg.trainer.logging.log_freq = int(payload.log_freq)
    # Language backbone tuning (text alignment preservation)
    if payload.language_backbone_lr is not None:
        try:
            cfg.scratch.lr_language_backbone = float(payload.language_backbone_lr)
        except Exception:
            pass
    if payload.freeze_language_backbone:
        try:
            cfg.scratch.lr_language_backbone = 0.0
        except Exception:
            pass
    # Balance strategy/config
    if payload.balance_strategy is not None:
        cfg.dataset.balance_strategy = payload.balance_strategy
        cfg.dataset.class_balance = payload.balance_strategy != "none"
    if payload.balance_classes is not None:
        cfg.dataset.class_balance = bool(payload.balance_classes)
    if payload.balance_power is not None:
        cfg.dataset.balance_power = float(payload.balance_power)
    if payload.balance_clip is not None:
        cfg.dataset.balance_clip = float(payload.balance_clip)
    if payload.balance_beta is not None:
        cfg.dataset.balance_beta = float(payload.balance_beta)
    if payload.balance_gamma is not None:
        cfg.dataset.balance_gamma = float(payload.balance_gamma)
    cfg.trainer.checkpoint.save_dir = f"{cfg.launcher.experiment_log_dir}/checkpoints"
    if "meters" in cfg.trainer and "val" in cfg.trainer.meters:
        try:
            cfg.trainer.meters.val.roboflow100.detection.dump_dir = f"{cfg.launcher.experiment_log_dir}/dumps/local"
            cfg.trainer.meters.val.roboflow100.detection.pred_file_evaluators[0].gt_path = cfg.paths.val_ann_file
            # Apply val filtering/tuning
            if payload.val_score_thresh is not None:
                cfg.trainer.meters.val.roboflow100.detection.min_confidence = float(payload.val_score_thresh)
            if payload.val_max_dets is not None:
                cfg.trainer.meters.val.roboflow100.detection.max_dets = int(payload.val_max_dets)
        except Exception:
            pass
    # Prompt vocab overrides: allow multiple variants per class and optional randomization during training
    user_prompts = payload.prompt_variants or {}
    prompt_map: Dict[int, List[str]] = {}
    classes = meta.get("classes") or []
    if classes and user_prompts:
        def _normalise_variants(raw: Any) -> List[str]:
            if raw is None:
                return []
            if isinstance(raw, str):
                parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
                return parts if parts else [raw.strip()] if raw.strip() else []
            if isinstance(raw, (list, tuple, set)):
                return [str(p).strip() for p in raw if str(p).strip()]
            return []

        for idx, label in enumerate(classes):
            # allow lookup by label or by (1-based) category id
            cat_id = idx + 1
            custom = (
                user_prompts.get(label)
                or user_prompts.get(str(label))
                or user_prompts.get(cat_id)
                or user_prompts.get(str(cat_id))
            )
            variants = _normalise_variants(custom)
            if variants:
                prompt_map[cat_id] = variants

    if prompt_map:
        prompt_randomize = bool(payload.prompt_randomize) if payload.prompt_randomize is not None else True
        # Train loader
        try:
            train_loader_cfg = cfg.trainer.data.train.dataset.get("coco_json_loader")  # type: ignore[index]
        except Exception:
            train_loader_cfg = None
        if train_loader_cfg is None:
            cfg.trainer.data.train.dataset["coco_json_loader"] = {}
            train_loader_cfg = cfg.trainer.data.train.dataset.get("coco_json_loader")  # type: ignore[index]
        try:
            train_loader_cfg["_target_"] = "sam3.train.data.coco_json_loaders.COCO_FROM_JSON"
            train_loader_cfg["_partial_"] = True
            train_loader_cfg["prompts"] = prompt_map
            train_loader_cfg["prompt_randomize"] = prompt_randomize
        except Exception:
            pass
        # Val loader (deterministic prompts)
        try:
            val_loader_cfg = cfg.trainer.data.val.dataset.coco_json_loader  # type: ignore[assignment]
        except Exception:
            val_loader_cfg = None
        if val_loader_cfg is None:
            try:
                cfg.trainer.data.val.dataset["coco_json_loader"] = {}
                val_loader_cfg = cfg.trainer.data.val.dataset.coco_json_loader  # type: ignore[assignment]
            except Exception:
                val_loader_cfg = None
        if val_loader_cfg is not None:
            try:
                val_loader_cfg["_target_"] = "sam3.train.data.coco_json_loaders.COCO_FROM_JSON"
                val_loader_cfg["_partial_"] = True
                val_loader_cfg["prompts"] = prompt_map
                val_loader_cfg["prompt_randomize"] = False
            except Exception:
                pass
    cfg.launcher.num_nodes = 1
    cfg.submitit.use_cluster = False
    cfg.submitit.cpus_per_task = max(cfg.scratch.num_train_workers, cfg.submitit.cpus_per_task or 0)
    Path(cfg.paths.experiment_log_dir).mkdir(parents=True, exist_ok=True)
    return cfg, int(cfg.launcher.gpus_per_node)


@app.post("/sam3/train/jobs")
def create_sam3_training_job(payload: Sam3TrainRequest):
    meta = _resolve_sam3_dataset_meta(payload.dataset_id)
    job_id = uuid.uuid4().hex
    prep_logs: List[str] = []
    cfg, num_gpus = _build_sam3_config(payload, meta, job_id, prep_logs)
    config_dict = OmegaConf.to_container(cfg, resolve=False)  # type: ignore[arg-type]
    job = Sam3TrainingJob(job_id=job_id, config=config_dict)
    with SAM3_TRAINING_JOBS_LOCK:
        SAM3_TRAINING_JOBS[job_id] = job
        for msg in prep_logs:
            _sam3_job_log(job, msg)
        _sam3_job_log(job, "Job queued")
    logger.info("[sam3-train %s] dataset=%s gpus=%s", job_id[:8], payload.dataset_id, num_gpus)
    _start_sam3_training_worker(job, cfg, num_gpus)
    return {"job_id": job_id}


@app.get("/sam3/train/jobs")
def list_sam3_training_jobs():
    _prune_job_registry(SAM3_TRAINING_JOBS, SAM3_TRAINING_JOBS_LOCK)
    with SAM3_TRAINING_JOBS_LOCK:
        jobs = sorted(SAM3_TRAINING_JOBS.values(), key=lambda job: job.created_at, reverse=True)
        return [_serialize_sam3_job(job) for job in jobs]


@app.get("/sam3/train/jobs/{job_id}")
def get_sam3_training_job(job_id: str):
    job = _get_sam3_job(job_id)
    return _serialize_sam3_job(job)


@app.post("/sam3/train/jobs/{job_id}/cancel")
def cancel_sam3_training_job(job_id: str):
    job = _get_sam3_job(job_id)
    with SAM3_TRAINING_JOBS_LOCK:
        if job.status in {"succeeded", "failed", "cancelled"}:
            raise HTTPException(status_code=HTTP_428_PRECONDITION_REQUIRED, detail="job_not_cancellable")
        if job.cancel_event.is_set():
            return {"status": job.status}
        job.cancel_event.set()
        if job.process and job.process.poll() is None:
            try:
                job.process.terminate()
            except Exception:  # noqa: BLE001
                pass
        next_status = job.status if job.status not in {"running", "queued"} else "cancelling"
        _sam3_job_update(job, status=next_status, message="Cancellation requested ...")
    return {"status": job.status}


@app.get("/sam3/train/cache_size")
def sam3_train_cache_size():
    cache_root = SAM3_JOB_ROOT / "splits"
    return {"bytes": _dir_size_bytes(cache_root)}


@app.post("/sam3/train/cache/purge")
def sam3_train_cache_purge():
    cache_root = SAM3_JOB_ROOT / "splits"
    deleted = _purge_directory(cache_root)
    return {"status": "ok", "deleted_bytes": deleted}


@app.get("/sam3/storage/runs")
def list_sam3_runs(variant: str = Query("sam3")):
    # SAM3-lite removed; always use sam3
    return _list_sam3_runs("sam3")


@app.delete("/sam3/storage/runs/{run_id}")
def delete_sam3_run(run_id: str, variant: str = Query("sam3"), scope: str = Query("all")):
    normalized = "sam3"
    if scope not in SAM3_STORAGE_SCOPES:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="invalid_scope")
    run_dir = _run_dir_for_request(run_id, normalized)
    active_paths = _active_run_paths_for_variant(normalized)
    if run_dir.resolve() in active_paths:
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail="sam3_run_active")
    deleted, freed = _delete_run_scope(run_dir, scope)
    return {"deleted": deleted, "freed_bytes": freed}


@app.post("/sam3/storage/runs/{run_id}/promote")
def promote_sam3_run(run_id: str, variant: str = Query("sam3")):
    return _promote_run(run_id, "sam3")


@app.get("/sam3/models/available")
def list_sam3_available_models(
    variant: str = Query("sam3"),
    promoted_only: bool = Query(False),
):
    """List run checkpoints for prompt model selection."""
    runs = _list_sam3_runs("sam3")
    models: List[Dict[str, Any]] = []
    # Always expose the base/active env model if available
    # Env/base model entry (always listed)
    env_base_path = SAM3_CHECKPOINT_PATH if SAM3_CHECKPOINT_PATH else None
    models.append(
        {
            "id": "Base SAM3",
            "key": "base",
            # Use None so activation loads from HF if no local checkpoint is present.
            "path": env_base_path,
            "size_bytes": None,
            "promoted": False,
            "active": active_sam3_checkpoint in {None, env_base_path},
            "variant": "sam3",
            "run_path": None,
            "source": "env",
        }
    )
    # Current active model entry (if different from env/base)
    if active_sam3_checkpoint and active_sam3_checkpoint != env_base_path:
        models.append(
            {
                "id": active_sam3_metadata.get("label") or active_sam3_metadata.get("id") or "active",
                "key": f"active:{active_sam3_checkpoint}",
                "path": active_sam3_checkpoint,
                "size_bytes": None,
                "promoted": False,
                "active": True,
                "variant": "sam3",
                "run_path": None,
                "source": active_sam3_metadata.get("source") or "env",
            }
        )
    for run in runs:
        if promoted_only and not run.get("promoted"):
            continue
        if run.get("active"):
            # allow listing active too, but mark status
            pass
        ckpts = run.get("checkpoints") or []
        if not ckpts:
            continue
        # prefer last.ckpt
        chosen = None
        for ck in ckpts:
            if ck.get("file") == "last.ckpt":
                chosen = ck
                break
        if chosen is None:
            chosen = ckpts[0]
        models.append(
            {
                "id": run.get("id"),
                "path": chosen.get("path"),
                "size_bytes": chosen.get("size_bytes"),
                "promoted": run.get("promoted", False),
                "active": run.get("active", False),
                "variant": run.get("variant"),
                "run_path": run.get("path"),
            }
        )
    return models


@app.get("/sam3/models/status")
def sam3_model_status():
    return {
        "checkpoint": active_sam3_checkpoint,
        "active": active_sam3_metadata,
        "enable_segmentation": active_sam3_enable_segmentation,
        "loaded_text": sam3_text_model is not None,
    }


@app.post("/sam3/models/activate")
def activate_sam3_model(payload: Sam3ModelActivateRequest):
    global active_sam3_checkpoint, active_sam3_model_id, active_sam3_metadata, active_sam3_enable_segmentation
    checkpoint_path = payload.checkpoint_path
    source = "huggingface"
    resolved_path: Optional[Path] = None
    if checkpoint_path:
        resolved_path = Path(checkpoint_path).resolve()
        if not resolved_path.exists():
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="sam3_checkpoint_not_found")
        checkpoint_path = str(resolved_path)
        source = "custom"
    enable_seg = active_sam3_enable_segmentation if checkpoint_path is not None else True
    if payload.enable_segmentation is not None:
        enable_seg = bool(payload.enable_segmentation)
    active_sam3_checkpoint = checkpoint_path
    active_sam3_enable_segmentation = enable_seg
    active_sam3_model_id = payload.label or (resolved_path.stem if resolved_path else "facebook/sam3")
    active_sam3_metadata = {
        "id": active_sam3_model_id,
        "label": payload.label or active_sam3_model_id,
        "checkpoint": active_sam3_checkpoint,
        "source": source,
        "enable_segmentation": active_sam3_enable_segmentation,
    }
    _reset_sam3_runtime()
    return {"active": active_sam3_metadata}


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
    prep_logs: List[str] = []
    config = _build_qwen_config(payload, job_id, prep_logs)
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
        for msg in prep_logs:
            _qwen_job_log(job, msg)
        _qwen_job_log(job, "Job queued")
    _start_qwen_training_worker(job, config)
    return {"job_id": job_id}


@app.get("/qwen/train/jobs")
def list_qwen_training_jobs(request: Request):
    _prune_job_registry(QWEN_TRAINING_JOBS, QWEN_TRAINING_JOBS_LOCK)
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


@app.get("/qwen/train/cache_size")
def qwen_train_cache_size():
    cache_root = QWEN_JOB_ROOT / "splits"
    return {"bytes": _dir_size_bytes(cache_root)}


@app.post("/qwen/train/cache/purge")
def qwen_train_cache_purge():
    cache_root = QWEN_JOB_ROOT / "splits"
    deleted = _purge_directory(cache_root)
    return {"status": "ok", "deleted_bytes": deleted}


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
    global active_classifier_path, active_labelmap_path, active_label_list, clip_last_error

    classifier_path = _normalise_optional_path(payload.classifier_path) or active_classifier_path
    labelmap_path = _normalise_optional_path(payload.labelmap_path)
    labelmap_provided = "labelmap_path" in payload.__fields_set__
    clip_name = _normalise_optional_path(payload.clip_model) or clip_model_name or DEFAULT_CLIP_MODEL

    if not classifier_path:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="classifier_required")
    classifier_path_abs = os.path.abspath(classifier_path)
    if not os.path.isfile(classifier_path_abs):
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="classifier_not_found")
    allowed_root = (UPLOAD_ROOT / "classifiers").resolve()
    if not str(Path(classifier_path_abs).resolve()).startswith(str(allowed_root)):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="classifier_path_not_allowed")
    _validate_upload_extension(classifier_path_abs, CLASSIFIER_ALLOWED_EXTS, "classifier_extension_not_allowed")

    try:
        new_clf = joblib.load(classifier_path_abs)
    except Exception as exc:  # noqa: BLE001
        clip_last_error = str(exc)
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
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="clip_model_not_allowed")

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
        allowed_label_root = (UPLOAD_ROOT / "labelmaps").resolve()
        if not str(Path(labelmap_path_abs).resolve()).startswith(str(allowed_label_root)):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="labelmap_path_not_allowed")
        _validate_upload_extension(labelmap_path_abs, LABELMAP_ALLOWED_EXTS, "labelmap_extension_not_allowed")
        labelmap_entries = _load_labelmap_file(labelmap_path_abs)
    elif not labelmap_provided and active_labelmap_path:
        labelmap_path_abs = active_labelmap_path
        labelmap_entries = list(active_label_list)
    try:
        clf_classes = int(getattr(new_clf, "coef_", None).shape[0]) if getattr(new_clf, "coef_", None) is not None else None
    except Exception:
        clf_classes = None
    if clf_classes is not None:
        if not labelmap_entries:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="labelmap_required_for_classifier")
        if clf_classes != len(labelmap_entries):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="labelmap_classifier_class_mismatch")

    with clip_lock:
        clf = new_clf
        clip_model = new_clip_model
        clip_preprocess = new_preprocess
        clip_model_name = clip_name
        clip_initialized = True
        active_classifier_path = classifier_path_abs
        active_labelmap_path = labelmap_path_abs
        active_label_list = labelmap_entries
        clip_last_error = None

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
    gpu_total_mb = None
    gpu_free_mb = None
    if torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            gpu_total_mb = _bytes_to_mb(int(total_bytes))
            gpu_free_mb = _bytes_to_mb(int(free_bytes))
        except Exception:
            gpu_total_mb = None
            gpu_free_mb = None
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
        gpu_total_mb=gpu_total_mb,
        gpu_free_mb=gpu_free_mb,
    )


@app.get("/predictor_settings", response_model=PredictorSettings)
def get_predictor_settings():
    return _predictor_settings_payload()


@app.post("/predictor_settings", response_model=PredictorSettings)
def update_predictor_settings(payload: PredictorSettingsUpdate):
    min_cap, max_cap = predictor_manager.capacity_limits()
    try:
        requested = int(payload.max_predictors)
    except Exception:
        requested = min_cap
    normalized = max(min_cap, min(max_cap, requested))
    predictor_manager.set_capacity(normalized)
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

    _validate_upload_size(file, max_bytes=BASE64_IMAGE_MAX_BYTES)
    image_bytes = file.file.read(BASE64_IMAGE_MAX_BYTES + 1)
    if len(image_bytes) > BASE64_IMAGE_MAX_BYTES:
        raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="upload_too_large")
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
    warnings: List[str] = []
    try:
        _, items = _extract_qwen_json_block(qwen_text)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        warnings.append(f"parse_error:{detail}")
        print(f"[Qwen parse error] {detail}; raw text follows:\n{qwen_text}")
        return QwenInferenceResponse(
            boxes=[],
            raw_response=qwen_text,
            prompt=final_prompt,
            prompt_type=prompt_type,  # type: ignore[arg-type]
            warnings=warnings,
            image_token=token,
        )
    normalized_items = _qwen_items_from_payload(items)
    if not normalized_items:
        print("[Qwen parsed but empty list]", qwen_text)
        warnings.append("no_results")
        return QwenInferenceResponse(
            boxes=[],
            raw_response=qwen_text,
            prompt=final_prompt,
            prompt_type=prompt_type,  # type: ignore[arg-type]
            warnings=warnings,
            image_token=token,
        )
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
    detections, masks_arr = _run_sam3_text_inference(
        pil_img,
        payload.text_prompt,
        payload.threshold,
        payload.mask_threshold,
        effective_limit,
        return_masks=True,
        min_size=payload.min_size,
        simplify_epsilon=payload.simplify_epsilon,
    )
    warnings: List[str] = []
    if not detections:
        warnings.append("no_results")
    encoded_masks = None
    if detections:
        encoded_masks = []
        for idx, det in enumerate(detections):
            payload = det.mask if isinstance(det, QwenDetection) else None
            if payload is None and masks_arr is not None and idx < len(masks_arr) and masks_arr[idx] is not None:
                try:
                    payload = encode_binary_mask(masks_arr[idx])
                except Exception:
                    payload = None
            encoded_masks.append(payload)
        if all(m is None for m in encoded_masks):
            encoded_masks = None
    return Sam3TextPromptResponse(
        detections=detections,
        warnings=warnings,
        image_token=token,
        masks=encoded_masks,
    )


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
    detections, masks_arr = _run_sam3_text_inference(
        pil_img,
        payload.text_prompt,
        payload.threshold,
        payload.mask_threshold,
        effective_limit,
        return_masks=True,
        min_size=payload.min_size,
        simplify_epsilon=payload.simplify_epsilon,
    )
    # TODO: enrich with masks for polygon mode consumers.
    responses: List[SamPointAutoResponse] = []
    warnings: List[str] = []
    if not detections:
        warnings.append("no_results")
    for idx, det in enumerate(detections):
        mask = masks_arr[idx] if masks_arr is not None and idx < len(masks_arr) else None
        mask_payload = det.mask if hasattr(det, "mask") else None
        if mask_payload is None and mask is not None:
            mask_payload = encode_binary_mask(mask)
        if mask is not None:
            try:
                x_min, y_min, x_max, y_max = mask_to_bounding_box(mask)
            except Exception:
                x_min, y_min, x_max, y_max = yolo_to_corners(det.bbox, pil_img.width, pil_img.height)
        else:
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
                    mask=mask_payload,
                    simplify_epsilon=getattr(det, "simplify_epsilon", None),
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
                mask=mask_payload,
                simplify_epsilon=getattr(det, "simplify_epsilon", None),
            )
        )
    return Sam3TextPromptAutoResponse(detections=responses, warnings=warnings, image_token=token)


@app.post("/sam3/visual_prompt", response_model=Sam3TextPromptResponse)
def sam3_visual_prompt(payload: Sam3VisualPrompt):
    variant = _default_variant(payload.sam_variant or "sam3")
    if variant != "sam3":
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="sam3_visual_requires_sam3")
    pil_img, np_img, token = resolve_image_payload(payload.image_base64, payload.image_token, variant)
    effective_limit = payload.max_results if payload.max_results is not None else 20
    try:
        detections, masks_arr = _run_sam3_visual_inference(
            pil_img,
            (
                float(payload.bbox_left),
                float(payload.bbox_top),
                float(payload.bbox_width),
                float(payload.bbox_height),
            ),
            payload.threshold,
            payload.mask_threshold,
            effective_limit,
            return_masks=True,
            min_size=payload.min_size,
            simplify_epsilon=payload.simplify_epsilon,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"sam3_visual_failed:{exc}") from exc
    warnings: List[str] = []
    if not detections:
        warnings.append("no_results")
    encoded_masks = None
    if detections:
        encoded_masks = []
        for idx, det in enumerate(detections):
            payload_mask = det.mask if isinstance(det, QwenDetection) else None
            if payload_mask is None and masks_arr is not None and idx < len(masks_arr) and masks_arr[idx] is not None:
                try:
                    payload_mask = encode_binary_mask(masks_arr[idx])
                except Exception:
                    payload_mask = None
            encoded_masks.append(payload_mask)
        if all(m is None for m in encoded_masks):
            encoded_masks = None
    return Sam3TextPromptResponse(
        detections=detections,
        warnings=warnings,
        image_token=token,
        masks=encoded_masks,
    )


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
    mask_arr = np.asarray(masks[0])
    if mask_arr.dtype != np.uint8:
        mask_arr = (mask_arr > 0).astype(np.uint8)
    left, top, right, bottom = mask_to_bounding_box(mask_arr)
    yolo_box = to_yolo(pil_img.width, pil_img.height, left, top, right, bottom)
    return YoloBboxOutput(
        class_id="0",
        bbox=yolo_box,
        uuid=prompt.uuid,
        image_token=token,
        mask=encode_binary_mask(mask_arr),
        simplify_epsilon=None,
    )


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
    mask_arr = np.asarray(masks[0])
    if mask_arr.dtype != np.uint8:
        mask_arr = (mask_arr > 0).astype(np.uint8)
    x_min, y_min, x_max, y_max = mask_to_bounding_box(mask_arr)
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
        mask=encode_binary_mask(mask_arr),
        simplify_epsilon=None,
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
    mask_arr = np.asarray(masks[0])
    if mask_arr.dtype != np.uint8:
        mask_arr = (mask_arr > 0).astype(np.uint8)
    left, top, right, bottom = mask_to_bounding_box(mask_arr)
    yolo_box = to_yolo(pil_img.width, pil_img.height, left, top, right, bottom)
    li = max(0, int(left))
    ti = max(0, int(top))
    ri = min(pil_img.width, int(right))
    bi = min(pil_img.height, int(bottom))
    if ri <= li or bi <= ti:
        return SamPointAutoResponse(
            prediction="unknown",
            bbox=yolo_box,
            uuid=prompt.uuid,
            error="empty_mask",
            image_token=token,
            mask=encode_binary_mask(mask_arr),
            simplify_epsilon=None,
        )
    subarr = np_img[ti:bi, li:ri, :]
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
        mask=encode_binary_mask(mask_arr),
        simplify_epsilon=None,
    )


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
    mask_arr = np.asarray(masks[0])
    if mask_arr.dtype != np.uint8:
        mask_arr = (mask_arr > 0).astype(np.uint8)
    left, top, right, bottom = mask_to_bounding_box(mask_arr)
    yolo_box = to_yolo(pil_img.width, pil_img.height, left, top, right, bottom)
    return YoloBboxOutput(
        class_id="0",
        bbox=yolo_box,
        uuid=prompt.uuid,
        image_token=token,
        mask=encode_binary_mask(mask_arr),
        simplify_epsilon=None,
    )


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
    mask_arr = np.asarray(masks[0])
    if mask_arr.dtype != np.uint8:
        mask_arr = (mask_arr > 0).astype(np.uint8)
    left, top, right, bottom = mask_to_bounding_box(mask_arr)
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
    return SamPointAutoResponse(
        prediction=str(pred_cls),
        bbox=yolo_box,
        uuid=prompt.uuid,
        image_token=token,
        mask=encode_binary_mask(mask_arr),
        simplify_epsilon=None,
    )


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
    mask_arr = np.asarray(masks[0])
    if mask_arr.dtype != np.uint8:
        mask_arr = (mask_arr > 0).astype(np.uint8)
    x_min, y_min, x_max, y_max = mask_to_bounding_box(mask_arr)
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
        mask=encode_binary_mask(mask_arr),
        simplify_epsilon=None,
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
    mask_arr = np.asarray(masks[0])
    if mask_arr.dtype != np.uint8:
        mask_arr = (mask_arr > 0).astype(np.uint8)
    x_min, y_min, x_max, y_max = mask_to_bounding_box(mask_arr)
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
        mask=encode_binary_mask(mask_arr),
        simplify_epsilon=None,
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
