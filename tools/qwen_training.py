"""Training utilities for Qwen 3 VL fine-tuning.

Supports two training modes:
- official_lora: LoRA training with full-precision base model.
- trl_qlora: QLoRA training using TRL SFTTrainer (wired in step 5).
"""

from __future__ import annotations

import json
import inspect
import logging
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from packaging import version as packaging_version
except Exception:  # noqa: BLE001
    packaging_version = None

try:
    from transformers import (
        AutoConfig,
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
except Exception as exc:  # noqa: BLE001
    AutoProcessor = None  # type: ignore[assignment]
    AutoConfig = None  # type: ignore[assignment]
    BitsAndBytesConfig = None  # type: ignore[assignment]
    Qwen3VLForConditionalGeneration = None  # type: ignore[assignment]
    Qwen3VLMoeForConditionalGeneration = None  # type: ignore[assignment]
    Trainer = None  # type: ignore[assignment]
    TrainingArguments = None  # type: ignore[assignment]
    TrainerCallback = object  # type: ignore[assignment]
    TRANSFORMERS_IMPORT_ERROR = exc
else:
    TRANSFORMERS_IMPORT_ERROR = None

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception as exc:  # noqa: BLE001
    LoraConfig = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]
    prepare_model_for_kbit_training = None  # type: ignore[assignment]
    PEFT_IMPORT_ERROR = exc
else:
    PEFT_IMPORT_ERROR = None

try:
    from trl import SFTConfig, SFTTrainer
except Exception as exc:  # noqa: BLE001
    SFTConfig = None  # type: ignore[assignment]
    SFTTrainer = None  # type: ignore[assignment]
    TRL_IMPORT_ERROR = exc
else:
    TRL_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], bool]
TelemetryCallback = Callable[[Dict[str, Any]], None]

QWEN_MIN_TRANSFORMERS = "4.57.0"
QWEN_PLATFORM_TRANSFORMERS = "transformers"
CAPTION_INSTRUCTION_ARCHIVE_FORMAT = "tator_caption_instruction_archive_v1"
CAPTION_INSTRUCTION_TRAINABLE_VALIDATION_STATUSES = {"accepted", "machine_validated"}
CAPTION_INSTRUCTION_NONTRAINABLE_VALIDATION_STATUSES = {"rejected", "failed", "invalid"}
CAPTION_INSTRUCTION_TRAINABLE_REVIEW_STATUSES = {"accepted", "unreviewed", "machine_validated"}
CAPTION_INSTRUCTION_NONTRAINABLE_REVIEW_STATUSES = {"rejected", "needs_revision"}
QWEN_PLATFORM_MLX = "mlx_vlm"

DEFAULT_SYSTEM_PROMPT = (
    "You are an annotation assistant that only returns JSON objects shaped like {\"detections\":[{\"label\":\"class\"," \
    "\"bbox\":[x1,y1,x2,y2]} or {\"label\":\"class\",\"point\":[x,y]}]}. Always reply with compact JSON and no prose."
)


def _build_legacy_prompt(context_text: str, mode: str) -> str:
    parts: List[str] = []
    context_text = (context_text or "").strip()
    if context_text:
        parts.append(context_text)
    parts.append("Return detections for every labeled object.")
    if mode == "point":
        parts.append(
            'Return a JSON object named "detections". Each detection must include "label" and "point" as [x,y] pixel coordinates near the object center. '
            'If nothing is present, respond with {"detections": []}. Respond with JSON only.'
        )
    else:
        parts.append(
            'Return a JSON object named "detections". Each detection must include "label" and "bbox" as [x1,y1,x2,y2] pixel coordinates (integers). '
            'If nothing is present, respond with {"detections": []}. Respond with JSON only.'
        )
    return " ".join(parts).strip()


def _build_output_payload(detections: List[Dict[str, Any]], mode: str) -> str:
    items: List[Dict[str, Any]] = []
    for det in detections:
        label = det.get("label")
        if not label:
            continue
        if mode == "point":
            point = det.get("point")
            if point:
                items.append({"label": label, "point": point})
        else:
            bbox = det.get("bbox")
            if bbox:
                items.append({"label": label, "bbox": bbox})
    return json.dumps({"detections": items}, ensure_ascii=False)


class TrainingError(RuntimeError):
    """Raised when the Qwen training pipeline fails in a recoverable way."""


@dataclass
class QwenTrainingConfig:
    dataset_root: str
    result_path: str
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    requested_model_id: Optional[str] = None
    requested_model_metadata: Dict[str, Any] = field(default_factory=dict)
    runtime_platform: str = QWEN_PLATFORM_TRANSFORMERS
    training_mode: str = "official_lora"  # official_lora | trl_qlora
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    run_name: Optional[str] = None
    devices: Optional[str] = None
    batch_size: int = 1
    max_epochs: int = 3
    lr: float = 2e-4
    accumulate_grad_batches: int = 8
    warmup_steps: int = 50
    num_workers: int = 0
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Sequence[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    seed: int = 1337
    log_every_n_steps: int = 10
    max_pixels: int = 28 * 28 * 576
    min_pixels: int = 28 * 28 * 16
    max_length: Optional[int] = None
    train_limit: Optional[int] = None
    val_limit: Optional[int] = None
    vram_estimate_mb: Optional[float] = None
    vram_estimate_note: Optional[str] = None


@dataclass
class QwenTrainingResult:
    config: QwenTrainingConfig
    checkpoints: List[str]
    latest_checkpoint: Optional[str]
    epochs_ran: int
    metadata: Dict[str, object] = field(default_factory=dict)


def _ensure_transformers_ready() -> None:
    if TRANSFORMERS_IMPORT_ERROR is not None or AutoProcessor is None:
        raise TrainingError(f"qwen_transformers_missing:{TRANSFORMERS_IMPORT_ERROR}")
    if packaging_version is None:
        return
    try:
        import transformers

        if packaging_version.parse(transformers.__version__) < packaging_version.parse(QWEN_MIN_TRANSFORMERS):
            raise TrainingError(
                f"qwen_transformers_too_old:{transformers.__version__}<{QWEN_MIN_TRANSFORMERS}"
            )
    except TrainingError:
        raise
    except Exception:
        return


def _is_qwen_moe_model_id(model_id: str) -> bool:
    lowered = str(model_id or "").lower()
    return "a3b" in lowered or "a22b" in lowered or "moe" in lowered


def _qwen_training_model_class(model_id: str):
    model_type = ""
    if AutoConfig is not None:
        try:
            hf_config = AutoConfig.from_pretrained(str(model_id))
            model_type = str(getattr(hf_config, "model_type", "") or "").lower()
        except Exception:
            model_type = ""
    if model_type == "qwen3_vl_moe" or _is_qwen_moe_model_id(model_id):
        if Qwen3VLMoeForConditionalGeneration is None:
            raise TrainingError(
                f"qwen3_vl_moe_unavailable:{TRANSFORMERS_IMPORT_ERROR}"
            )
        return Qwen3VLMoeForConditionalGeneration
    if Qwen3VLForConditionalGeneration is None:
        raise TrainingError(f"qwen3_vl_unavailable:{TRANSFORMERS_IMPORT_ERROR}")
    return Qwen3VLForConditionalGeneration


def _load_qwen_training_model(model_id: str, **kwargs):
    model_cls = _qwen_training_model_class(model_id)
    return model_cls.from_pretrained(str(model_id), **kwargs)


def _lora_target_modules(config: QwenTrainingConfig, *, qlora: bool) -> Any:
    targets = [str(item).strip() for item in (config.lora_target_modules or []) if str(item).strip()]
    if not targets:
        return "all-linear" if qlora else ["q_proj", "k_proj", "v_proj", "o_proj"]
    if len(targets) == 1 and targets[0] == "all-linear":
        return "all-linear"
    return targets


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _filter_kwargs_for(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if cls is None:
        return kwargs
    try:
        params = inspect.signature(cls).parameters
    except Exception:
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return kwargs
    allowed = set(params.keys())
    return {key: value for key, value in kwargs.items() if key in allowed}


def _select_eval_strategy_key(cls) -> Optional[str]:
    if cls is None:
        return None
    try:
        params = inspect.signature(cls).parameters
    except Exception:
        return None
    if "evaluation_strategy" in params:
        return "evaluation_strategy"
    if "eval_strategy" in params:
        return "eval_strategy"
    return None


def _resolve_image_path(dataset_root: Path, split: str, image_rel: str) -> Optional[Path]:
    raw = str(image_rel or "").replace("\\", "/").strip()
    win_path = PureWindowsPath(raw)
    if win_path.is_absolute() or win_path.drive:
        return None
    rel_path = Path(raw)
    if rel_path.is_absolute() or not raw:
        return None
    if any(part == ".." for part in rel_path.parts):
        return None
    try:
        dataset_root_resolved = dataset_root.resolve(strict=False)
    except (OSError, RuntimeError):
        return None
    roots = [
        dataset_root / split / "images",
        dataset_root / split,
        dataset_root / "images",
        dataset_root,
    ]
    for root in roots:
        try:
            root_resolved = root.resolve(strict=False)
            candidate = (root / rel_path).resolve(strict=False)
            root_resolved.relative_to(dataset_root_resolved)
            candidate.relative_to(dataset_root_resolved)
            candidate.relative_to(root_resolved)
        except (OSError, RuntimeError, ValueError):
            continue
        try:
            is_file = candidate.is_file()
        except OSError:
            is_file = False
        if is_file:
            return candidate
    return None


def _resolve_annotation_path(dataset_root: Path, split: str) -> Optional[Path]:
    split_norm = str(split or "").strip().lower()
    if split_norm not in {"train", "val"}:
        return None
    try:
        dataset_root_resolved = dataset_root.resolve(strict=False)
        ann_path = (dataset_root / split_norm / "annotations.jsonl").resolve(strict=False)
        ann_path.relative_to(dataset_root_resolved)
    except (OSError, RuntimeError, ValueError):
        return None
    try:
        if not ann_path.is_file():
            return None
    except OSError:
        return None
    return ann_path


def _conversation_to_messages(
    conversations: List[Dict[str, Any]],
    image: Image.Image,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    image_inserted = False
    cleaned_system = (system_prompt or "").strip()
    if cleaned_system:
        messages.append({"role": "system", "content": [{"type": "text", "text": cleaned_system}]})
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role_raw = str(turn.get("from") or turn.get("role") or "").strip().lower()
        if role_raw in {"human", "user"}:
            role = "user"
        elif role_raw in {"assistant", "gpt"}:
            role = "assistant"
        elif role_raw == "system":
            role = "system"
        else:
            role = "assistant"
        value = str(turn.get("value") or turn.get("text") or "").strip()
        if role == "user":
            parts = value.split("<image>")
            content: List[Dict[str, Any]] = []
            for idx, part in enumerate(parts):
                if part.strip():
                    content.append({"type": "text", "text": part.strip()})
                if idx < len(parts) - 1 and not image_inserted:
                    content.append({"type": "image", "image": image})
                    image_inserted = True
            messages.append({"role": "user", "content": content})
        elif role == "system":
            if value:
                messages.append({"role": "system", "content": [{"type": "text", "text": value}]})
        else:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": value}]})
    if not image_inserted:
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, list):
                content.insert(0, {"type": "image", "image": image})
                image_inserted = True
                break
    return messages


def _normalise_training_review_decision(value: Any) -> str:
    decision = re.sub(r"[\s-]+", "_", str(value or "").strip().lower())
    if decision in {"reject", "rejected", "deny", "denied", "drop", "dropped", "fail", "failed"}:
        return "rejected"
    if decision in {"revise", "revised", "needs_revision", "needs_review", "needs_rewrite", "edit", "edited"}:
        return "needs_revision"
    if decision in {"accept", "accepted", "approve", "approved", "keep", "kept", "pass", "passed"}:
        return "accepted"
    return decision or "unreviewed"


def _normalise_flat_training_question(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _normalise_flat_training_image_path(value: Any) -> str:
    raw = re.sub(r"/+", "/", str(value or "").replace("\\", "/").strip())
    while raw.startswith("./"):
        raw = raw[2:].strip()
    parts = [part for part in raw.split("/") if part and part != "."]
    return "/".join(parts)


def _normalise_flat_training_image_key(dataset_root: Path, split: str, value: Any) -> str:
    image_path = _normalise_flat_training_image_path(value)
    resolved = _resolve_image_path(dataset_root, split, str(value or ""))
    if resolved is not None:
        try:
            return resolved.resolve(strict=False).relative_to(dataset_root.resolve(strict=False)).as_posix()
        except (OSError, RuntimeError, ValueError):
            pass
    return image_path


def _flat_training_row_error_suffix(line_number: Optional[int]) -> str:
    return f":line_{line_number}" if line_number is not None else ""


def _validate_flat_training_row_metadata(
    payload: Dict[str, Any],
    *,
    line_number: Optional[int] = None,
) -> None:
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    source_archive = str(metadata.get("source_archive") or "").strip()
    is_caption_instruction_row = source_archive == CAPTION_INSTRUCTION_ARCHIVE_FORMAT
    if is_caption_instruction_row:
        for field in ("qa_id", "row_type", "answer_source", "answer_format"):
            if not str(metadata.get(field) or "").strip():
                raise TrainingError(
                    f"qwen_training_row_missing_{field}{_flat_training_row_error_suffix(line_number)}"
                )
    validation_status = str(metadata.get("validation_status") or "").strip().lower()
    if is_caption_instruction_row and not validation_status:
        raise TrainingError(
            f"qwen_training_row_missing_validation_status{_flat_training_row_error_suffix(line_number)}"
        )
    if validation_status in CAPTION_INSTRUCTION_NONTRAINABLE_VALIDATION_STATUSES:
        raise TrainingError(
            f"qwen_training_row_validation_rejected{_flat_training_row_error_suffix(line_number)}"
        )
    if (
        is_caption_instruction_row
        and validation_status
        and validation_status not in CAPTION_INSTRUCTION_TRAINABLE_VALIDATION_STATUSES
    ):
        raise TrainingError(
            f"qwen_training_row_unsupported_validation_status:{validation_status}{_flat_training_row_error_suffix(line_number)}"
        )
    raw_review_values = [
        value for value in (metadata.get("review_status"), metadata.get("review_decision"))
        if str(value or "").strip()
    ]
    review_statuses = [_normalise_training_review_decision(value) for value in raw_review_values]
    if is_caption_instruction_row and not raw_review_values:
        raise TrainingError(
            f"qwen_training_row_missing_review_status{_flat_training_row_error_suffix(line_number)}"
        )
    nontrainable_review_status = next(
        (status for status in review_statuses if status in CAPTION_INSTRUCTION_NONTRAINABLE_REVIEW_STATUSES),
        "",
    )
    if nontrainable_review_status:
        raise TrainingError(
            f"qwen_training_row_review_not_trainable:{nontrainable_review_status}{_flat_training_row_error_suffix(line_number)}"
        )
    unsupported_review_status = next(
        (status for status in review_statuses if status not in CAPTION_INSTRUCTION_TRAINABLE_REVIEW_STATUSES),
        "",
    )
    if is_caption_instruction_row and unsupported_review_status:
        raise TrainingError(
            f"qwen_training_row_unsupported_review_status:{unsupported_review_status}{_flat_training_row_error_suffix(line_number)}"
        )
    row_type = str(metadata.get("row_type") or "").strip().lower()
    answer_format = str(metadata.get("answer_format") or "").strip().lower()
    answer = str(payload.get("answer") or "").strip()
    if answer and (row_type.startswith("deterministic_") or answer_format == "json" or answer_format.endswith("_json")):
        try:
            json.loads(answer)
        except Exception as exc:  # noqa: BLE001
            raise TrainingError(
                f"qwen_training_row_invalid_json_answer{_flat_training_row_error_suffix(line_number)}"
            ) from exc


def _flat_training_row_to_conversation_entry(
    payload: Dict[str, Any],
    image_name: str,
    *,
    line_number: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    question = str(payload.get("question") or "").strip()
    answer = str(payload.get("answer") or "").strip()
    if not question or not answer:
        return None
    _validate_flat_training_row_metadata(payload, line_number=line_number)
    user_value = question if "<image>" in question else f"<image>\n{question}"
    entry: Dict[str, Any] = {
        "image": image_name,
        "conversations": [
            {"from": "human", "value": user_value},
            {"from": "gpt", "value": answer},
        ],
    }
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        entry["metadata"] = dict(metadata)
    return entry


class QwenConversationDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        split: str,
        processor,
        system_prompt: Optional[str] = None,
        max_items: Optional[int] = None,
    ) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.processor = processor
        self.system_prompt = system_prompt
        self.entries: List[Dict[str, Any]] = []
        jsonl_path = _resolve_annotation_path(dataset_root, split)
        if jsonl_path is None:
            raise TrainingError(f"qwen_annotations_missing:{dataset_root / split / 'annotations.jsonl'}")
        seen_flat_image_questions: Set[Tuple[str, str]] = set()
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                image_name = payload.get("image")
                if not isinstance(image_name, str):
                    image_name = payload.get("image_path")
                if not isinstance(image_name, str):
                    continue
                image_name = image_name.strip()
                if not image_name:
                    continue
                if "conversations" in payload:
                    entry = dict(payload)
                    entry["image"] = image_name
                    self.entries.append(entry)
                elif "detections" in payload:
                    detections = payload.get("detections") or []
                    if not isinstance(detections, list):
                        detections = []
                    context_text = payload.get("context") or ""
                    for mode in ("bbox", "point"):
                        prompt_text = _build_legacy_prompt(context_text, mode)
                        output_text = _build_output_payload(detections, mode)
                        self.entries.append(
                            {
                                "image": image_name,
                                "conversations": [
                                    {"from": "human", "value": f"<image>\n{prompt_text}"},
                                    {"from": "gpt", "value": output_text},
                                ],
                            }
                        )
                elif "question" in payload and "answer" in payload:
                    entry = _flat_training_row_to_conversation_entry(
                        payload,
                        image_name,
                        line_number=line_number,
                    )
                    if entry is not None:
                        question = _normalise_flat_training_question(payload.get("question"))
                        image_key = _normalise_flat_training_image_key(dataset_root, split, image_name)
                        flat_key = (image_key, question)
                        if flat_key in seen_flat_image_questions:
                            raise TrainingError(
                                f"qwen_training_duplicate_flat_question:line_{line_number}"
                            )
                        seen_flat_image_questions.add(flat_key)
                        self.entries.append(entry)
                else:
                    continue
                if max_items and len(self.entries) >= max_items:
                    break
        if not self.entries:
            raise TrainingError("qwen_training_no_annotations")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        image_rel = entry.get("image")
        if not isinstance(image_rel, str):
            raise TrainingError("qwen_training_bad_image")
        image_path = _resolve_image_path(self.dataset_root, self.split, image_rel)
        if image_path is None:
            raise TrainingError(f"qwen_training_image_missing:{image_rel}")
        try:
            with Image.open(image_path) as im:
                image = im.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise TrainingError(f"qwen_training_image_open_failed:{image_rel}:{exc}") from exc
        conversations = entry.get("conversations") or []
        if not isinstance(conversations, list):
            raise TrainingError("qwen_training_bad_conversations")
        messages = _conversation_to_messages(conversations, image, self.system_prompt)
        return {
            "messages": messages,
            "images": [image],
        }


class QwenConversationCollator:
    def __init__(self, processor, max_length: Optional[int]) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length

    @staticmethod
    def _flat_images(images_list: Sequence[Sequence[Image.Image]]) -> List[Image.Image]:
        images: List[Image.Image] = []
        for item_images in images_list:
            images.extend(list(item_images or []))
        return images

    def _masked_token_ids(self) -> List[int]:
        token_ids = set()
        for value in (
            getattr(self.tokenizer, "pad_token_id", None),
            getattr(self.tokenizer, "image_token_id", None),
            getattr(self.tokenizer, "video_token_id", None),
            getattr(self.tokenizer, "vision_token_id", None),
            getattr(self.tokenizer, "vision_start_token_id", None),
            getattr(self.tokenizer, "vision_end_token_id", None),
            getattr(self.processor, "image_token_id", None),
            getattr(self.processor, "video_token_id", None),
            getattr(self.processor, "vision_token_id", None),
            getattr(self.processor, "vision_start_token_id", None),
            getattr(self.processor, "vision_end_token_id", None),
        ):
            if isinstance(value, int) and value >= 0:
                token_ids.add(value)
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            for token in (
                "<|image_pad|>",
                "<|video_pad|>",
                "<|vision_start|>",
                "<|vision_end|>",
            ):
                try:
                    value = convert(token)
                except Exception:
                    continue
                if isinstance(value, int) and value >= 0:
                    token_ids.add(value)
        return sorted(token_ids)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        messages_list: List[List[Dict[str, Any]]] = []
        images_list: List[List[Image.Image]] = []
        prompt_texts: List[str] = []
        for item in batch:
            messages = item["messages"]
            messages_list.append(messages)
            images_list.append(item["images"])
            prompt_messages = messages[:-1] if len(messages) > 1 else messages
            prompt_texts.append(
                self.processor.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        texts = [
            self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in messages_list
        ]
        flat_images = self._flat_images(images_list)
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=flat_images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = self.processor(
            text=texts,
            images=flat_images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()
        prompt_attention = prompt_inputs.get("attention_mask")
        if prompt_attention is not None:
            prompt_lengths = prompt_attention.sum(dim=1).tolist()
        else:
            prompt_lengths = [int(prompt_inputs["input_ids"].shape[1])] * len(batch)
        for idx, length in enumerate(prompt_lengths):
            labels[idx, : min(int(length), int(labels.shape[1]))] = -100
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            labels[attention_mask == 0] = -100
        for token_id in self._masked_token_ids():
            labels[labels == int(token_id)] = -100
        inputs["labels"] = labels
        return inputs


class _MlxConversationDatasetAdapter:
    """Expose the repo's local dataset in the list/slice shape mlx-vlm expects."""

    def __init__(self, dataset: QwenConversationDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _collate(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        images: List[Any] = []
        messages: List[Any] = []
        for item in items:
            raw_images = item.get("images") or []
            if not isinstance(raw_images, list):
                raw_images = [raw_images]
            images.extend(raw_images)
            messages.append(item["messages"])
        return {"images": images, "messages": messages}

    def __getitem__(self, idx):  # noqa: ANN001
        if isinstance(idx, slice):
            indices = list(range(*idx.indices(len(self))))
            return self._collate([self.dataset[item_idx] for item_idx in indices])
        if isinstance(idx, range):
            return self._collate([self.dataset[item_idx] for item_idx in idx])
        if isinstance(idx, (list, tuple)):
            return self._collate([self.dataset[int(item_idx)] for item_idx in idx])
        return self.dataset[int(idx)]


class _QwenTrainingCallback(TrainerCallback):
    def __init__(self, progress_cb: Optional[ProgressCallback], cancel_cb: Optional[CancelCallback], metrics_cb: Optional[TelemetryCallback]):
        self.progress_cb = progress_cb
        self.cancel_cb = cancel_cb
        self.metrics_cb = metrics_cb
        self.total_steps: Optional[int] = None

    def on_init_end(self, args, state, control, **kwargs):  # noqa: ANN001
        return control

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        if self.total_steps is None:
            self.total_steps = state.max_steps
        if self.progress_cb:
            self.progress_cb(0.01, "Qwen3 training started")

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if logs is None:
            return
        if self.metrics_cb:
            payload = {"step": state.global_step, **logs}
            self.metrics_cb(payload)
        if self.progress_cb and self.total_steps:
            progress = min(0.99, max(0.01, state.global_step / max(1, self.total_steps)))
            self.progress_cb(progress, f"step {state.global_step}/{self.total_steps}")
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        if self.cancel_cb and self.cancel_cb():
            control.should_training_stop = True
            control.should_save = True
        return control


def _train_official_lora(
    config: QwenTrainingConfig,
    progress_cb: Optional[ProgressCallback],
    cancel_cb: Optional[CancelCallback],
    metrics_cb: Optional[TelemetryCallback],
) -> QwenTrainingResult:
    _ensure_transformers_ready()
    if PEFT_IMPORT_ERROR is not None or get_peft_model is None or LoraConfig is None:
        raise TrainingError(f"qwen_peft_missing:{PEFT_IMPORT_ERROR}")
    _seed_all(config.seed)

    dataset_root = Path(config.dataset_root)
    result_path = Path(config.result_path)
    result_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    processor = AutoProcessor.from_pretrained(
        config.model_id,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    model = _load_qwen_training_model(
        config.model_id,
        torch_dtype=dtype if device == "cuda" else torch.float32,
        low_cpu_mem_usage=False,
    )
    lora_cfg = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=_lora_target_modules(config, qlora=False),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.config.use_cache = False

    train_dataset = QwenConversationDataset(
        dataset_root,
        "train",
        processor,
        config.system_prompt,
        config.train_limit,
    )
    val_dataset = QwenConversationDataset(
        dataset_root,
        "val",
        processor,
        config.system_prompt,
        config.val_limit,
    )

    collator = QwenConversationCollator(processor, config.max_length)

    training_kwargs: Dict[str, Any] = {
        "output_dir": str(result_path),
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": max(1, config.batch_size),
        "gradient_accumulation_steps": config.accumulate_grad_batches,
        "learning_rate": config.lr,
        "num_train_epochs": config.max_epochs,
        "warmup_steps": config.warmup_steps,
        "logging_steps": config.log_every_n_steps,
        "save_strategy": "epoch",
        "report_to": [],
        "bf16": (device == "cuda" and torch.cuda.is_bf16_supported()),
        "fp16": (device == "cuda" and not torch.cuda.is_bf16_supported()),
        "remove_unused_columns": False,
    }
    eval_key = _select_eval_strategy_key(TrainingArguments)
    if eval_key:
        training_kwargs[eval_key] = "epoch"
    training_kwargs = _filter_kwargs_for(TrainingArguments, training_kwargs)
    training_args = TrainingArguments(**training_kwargs)

    callback = _QwenTrainingCallback(progress_cb, cancel_cb, metrics_cb)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=[callback],
    )

    trainer.train()
    latest_dir = result_path / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(latest_dir))
    processor.save_pretrained(str(latest_dir))

    result = QwenTrainingResult(
        config=config,
        checkpoints=[str(latest_dir)],
        latest_checkpoint=str(latest_dir),
        epochs_ran=int(config.max_epochs),
    )
    return result


def _train_trl_qlora(
    config: QwenTrainingConfig,
    progress_cb: Optional[ProgressCallback],
    cancel_cb: Optional[CancelCallback],
    metrics_cb: Optional[TelemetryCallback],
) -> QwenTrainingResult:
    _ensure_transformers_ready()
    if TRL_IMPORT_ERROR is not None or SFTTrainer is None or SFTConfig is None:
        raise TrainingError(f"qwen_trl_missing:{TRL_IMPORT_ERROR}")
    if BitsAndBytesConfig is None:
        raise TrainingError("qwen_bitsandbytes_config_missing")
    if PEFT_IMPORT_ERROR is not None or LoraConfig is None:
        raise TrainingError(f"qwen_peft_missing:{PEFT_IMPORT_ERROR}")
    if prepare_model_for_kbit_training is None:
        raise TrainingError(f"qwen_peft_kbit_missing:{PEFT_IMPORT_ERROR}")
    _seed_all(config.seed)

    dataset_root = Path(config.dataset_root)
    result_path = Path(config.result_path)
    result_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise TrainingError("qwen_trl_qlora_requires_cuda")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    processor = AutoProcessor.from_pretrained(
        config.model_id,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    model = _load_qwen_training_model(
        config.model_id,
        device_map="auto",
        quantization_config=quant_config,
        low_cpu_mem_usage=False,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    train_dataset = QwenConversationDataset(
        dataset_root,
        "train",
        processor,
        config.system_prompt,
        config.train_limit,
    )
    val_dataset = QwenConversationDataset(
        dataset_root,
        "val",
        processor,
        config.system_prompt,
        config.val_limit,
    )
    collator = QwenConversationCollator(processor, config.max_length)

    training_args = SFTConfig(
        output_dir=str(result_path),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=max(1, config.batch_size),
        gradient_accumulation_steps=config.accumulate_grad_batches,
        learning_rate=config.lr,
        num_train_epochs=config.max_epochs,
        warmup_steps=config.warmup_steps,
        logging_steps=config.log_every_n_steps,
        report_to="none",
    )

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=_lora_target_modules(config, qlora=True),
        bias="none",
        task_type="CAUSAL_LM",
    )

    callback = _QwenTrainingCallback(progress_cb, cancel_cb, metrics_cb)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        peft_config=peft_config,
        callbacks=[callback],
    )
    trainer.train()
    latest_dir = result_path / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(latest_dir))
    processor.save_pretrained(str(latest_dir))
    result = QwenTrainingResult(
        config=config,
        checkpoints=[str(latest_dir)],
        latest_checkpoint=str(latest_dir),
        epochs_ran=int(config.max_epochs),
    )
    return result


def _load_mlx_training_backend() -> Dict[str, Any]:
    try:
        import mlx.optimizers as mlx_optimizers
        from mlx_vlm.trainer.datasets import VisionDataset as MlxVisionDataset
        from mlx_vlm.trainer.sft_trainer import TrainingArgs as MlxTrainingArgs
        from mlx_vlm.trainer.sft_trainer import train as mlx_train
        from mlx_vlm.trainer.utils import (
            find_all_linear_names as mlx_find_all_linear_names,
            get_peft_model as mlx_get_peft_model,
            not_supported_for_training as mlx_not_supported_for_training,
            print_trainable_parameters as mlx_print_trainable_parameters,
        )
        from mlx_vlm.utils import load as mlx_load
    except Exception as modern_exc:  # noqa: BLE001
        try:
            import mlx.core as mlx_core
            import mlx.optimizers as mlx_optimizers
            from mlx_vlm.trainer import Dataset as MlxVisionDataset
            from mlx_vlm.trainer import Trainer as MlxTrainer
            from mlx_vlm.trainer import save_adapter as mlx_save_adapter
            from mlx_vlm.trainer.trainer import TrainingArgs as MlxTrainingArgs
            from mlx_vlm.trainer.utils import (
                find_all_linear_names as mlx_find_all_linear_names,
                get_peft_model as mlx_get_peft_model,
                print_trainable_parameters as mlx_print_trainable_parameters,
            )
            from mlx_vlm.utils import load as mlx_load
            from mlx_vlm.utils import load_image_processor as mlx_load_image_processor
        except Exception as legacy_exc:  # noqa: BLE001
            raise TrainingError(
                "qwen_mlx_training_unavailable:install mlx-vlm==0.3.9 in the macOS environment"
            ) from legacy_exc
        return {
            "api": "legacy",
            "modern_import_error": modern_exc,
            "mx": mlx_core,
            "optimizers": mlx_optimizers,
            "VisionDataset": MlxVisionDataset,
            "TrainingArgs": MlxTrainingArgs,
            "Trainer": MlxTrainer,
            "save_adapter": mlx_save_adapter,
            "find_all_linear_names": mlx_find_all_linear_names,
            "get_peft_model": mlx_get_peft_model,
            "not_supported_for_training": {"gemma3n", "qwen3_omni"},
            "print_trainable_parameters": mlx_print_trainable_parameters,
            "load": mlx_load,
            "load_image_processor": mlx_load_image_processor,
        }
    return {
        "api": "modern",
        "optimizers": mlx_optimizers,
        "VisionDataset": MlxVisionDataset,
        "TrainingArgs": MlxTrainingArgs,
        "train": mlx_train,
        "find_all_linear_names": mlx_find_all_linear_names,
        "get_peft_model": mlx_get_peft_model,
        "not_supported_for_training": mlx_not_supported_for_training,
        "print_trainable_parameters": mlx_print_trainable_parameters,
        "load": mlx_load,
    }


def _is_quantized_mlx_model_id(model_id: str) -> bool:
    lowered = str(model_id or "").lower()
    return any(
        marker in lowered
        for marker in (
            "3bit",
            "4bit",
            "5bit",
            "6bit",
            "8bit",
            "q2",
            "q3",
            "q4",
            "q6",
            "q8",
            "qx",
            "mxfp",
        )
    )


def _mlx_training_flavor(model_id: str) -> str:
    return "mlx_qlora" if _is_quantized_mlx_model_id(model_id) else "mlx_lora"


def _float_loss_value(loss: Any) -> Optional[float]:
    try:
        if hasattr(loss, "item"):
            return float(loss.item())
        return float(loss)
    except Exception:
        return None


def _write_mlx_adapter_config(
    adapter_dir: Path,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> Path:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    config_path = adapter_dir / "adapter_config.json"
    config_path.write_text(
        json.dumps(
            {
                "rank": int(rank),
                "alpha": float(alpha),
                "dropout": float(dropout),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return config_path


def _train_mlx_lora(
    config: QwenTrainingConfig,
    progress_cb: Optional[ProgressCallback],
    cancel_cb: Optional[CancelCallback],
    metrics_cb: Optional[TelemetryCallback],
) -> QwenTrainingResult:
    backend = _load_mlx_training_backend()
    _seed_all(config.seed)

    dataset_root = Path(config.dataset_root)
    result_path = Path(config.result_path)
    latest_dir = result_path / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = latest_dir / "adapters.safetensors"

    batch_size = max(1, int(config.batch_size or 1))
    train_raw = QwenConversationDataset(
        dataset_root,
        "train",
        processor=None,
        system_prompt=config.system_prompt,
        max_items=config.train_limit,
    )
    val_raw = QwenConversationDataset(
        dataset_root,
        "val",
        processor=None,
        system_prompt=config.system_prompt,
        max_items=config.val_limit,
    )
    batch_size = min(batch_size, max(1, len(train_raw)))
    epochs = max(1, int(config.max_epochs or 1))
    steps_per_epoch_estimate = (len(train_raw) + batch_size - 1) // batch_size
    iters = max(1, steps_per_epoch_estimate * epochs)
    if cancel_cb and cancel_cb():
        raise TrainingError("qwen_training_cancelled")
    if progress_cb:
        progress_cb(0.03, "Loading MLX-VLM model for Qwen training")

    model, processor = backend["load"](
        config.model_id,
        processor_config={"trust_remote_code": True},
    )
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type in backend["not_supported_for_training"]:
        raise TrainingError(f"qwen_mlx_model_training_unsupported:{model_type}")

    if cancel_cb and cancel_cb():
        raise TrainingError("qwen_training_cancelled")
    if progress_cb:
        progress_cb(0.12, "Preparing MLX LoRA adapters")

    modules = backend["find_all_linear_names"](model.language_model)
    mlx_lora_alpha = float(config.lora_alpha)
    if backend.get("api") == "legacy":
        mlx_lora_alpha = mlx_lora_alpha / max(1, int(config.lora_rank or 1))
    model = backend["get_peft_model"](
        model,
        modules,
        rank=config.lora_rank,
        alpha=mlx_lora_alpha,
        dropout=config.lora_dropout,
        verbose=False,
    )
    backend["print_trainable_parameters"](model)
    model_config = getattr(model, "config", None)
    model_config_dict = dict(getattr(model_config, "__dict__", {}) or {})
    train_source = _MlxConversationDatasetAdapter(train_raw)
    val_source = _MlxConversationDatasetAdapter(val_raw)
    vision_dataset_cls = backend["VisionDataset"]
    dataset_extra_kwargs: Dict[str, Any] = {}
    if backend.get("api") == "legacy":
        dataset_extra_kwargs["image_processor"] = backend["load_image_processor"](config.model_id)
    dataset_extra_kwargs = _filter_kwargs_for(vision_dataset_cls, dataset_extra_kwargs)
    train_dataset = vision_dataset_cls(train_source, model_config_dict, processor, **dataset_extra_kwargs)
    val_dataset = vision_dataset_cls(val_source, model_config_dict, processor, **dataset_extra_kwargs)
    training_args_cls = backend["TrainingArgs"]
    training_kwargs = {
        "batch_size": batch_size,
        "iters": iters,
        "steps_per_report": max(1, int(config.log_every_n_steps or 10)),
        "steps_per_eval": max(1, int(config.log_every_n_steps or 10) * 10),
        "steps_per_save": max(1, int(config.log_every_n_steps or 10) * 10),
        "val_batches": -1,
        "max_seq_length": int(config.max_length or 2048),
        "adapter_file": str(adapter_file),
        "grad_checkpoint": True,
        "learning_rate": float(config.lr),
        "warmup_steps": int(config.warmup_steps or 0),
        "gradient_accumulation_steps": max(1, int(config.accumulate_grad_batches or 1)),
        "full_finetune": False,
    }
    training_kwargs = _filter_kwargs_for(training_args_cls, training_kwargs)
    training_args = training_args_cls(**training_kwargs)
    optimizer = backend["optimizers"].Adam(learning_rate=float(config.lr))

    if progress_cb:
        progress_cb(0.2, "Running MLX-VLM Qwen LoRA training")
    if backend.get("api") == "legacy":
        trainer_cls = backend["Trainer"]
        trainer_kwargs = _filter_kwargs_for(trainer_cls, {"train_on_completions": True})
        trainer = trainer_cls(model, optimizer, **trainer_kwargs)
        if hasattr(model, "train"):
            model.train()
        steps_per_epoch = max(1, (len(train_dataset) + batch_size - 1) // batch_size)
        total_steps = max(1, steps_per_epoch * epochs)
        global_step = 0
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                if cancel_cb and cancel_cb():
                    _write_mlx_adapter_config(
                        latest_dir,
                        rank=config.lora_rank,
                        alpha=mlx_lora_alpha,
                        dropout=config.lora_dropout,
                    )
                    backend["save_adapter"](model, str(adapter_file))
                    raise TrainingError("qwen_training_cancelled")
                start = step * batch_size
                stop = min(len(train_dataset), start + batch_size)
                loss = trainer.train_step(train_dataset[start:stop])
                mx = backend.get("mx")
                if mx is not None:
                    try:
                        mx.eval(loss, model.trainable_parameters(), optimizer.state)
                    except Exception:
                        mx.eval(loss)
                global_step += 1
                loss_value = _float_loss_value(loss)
                progress = min(0.97, 0.2 + 0.76 * (global_step / total_steps))
                if metrics_cb and (global_step == 1 or global_step % max(1, int(config.log_every_n_steps or 10)) == 0):
                    payload: Dict[str, Any] = {
                        "runtime_platform": QWEN_PLATFORM_MLX,
                        "training_backend": "mlx_vlm",
                        "training_backend_api": "legacy",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "progress": progress,
                    }
                    if loss_value is not None:
                        payload["loss"] = loss_value
                    metrics_cb(payload)
                if progress_cb and (global_step == 1 or global_step % max(1, int(config.log_every_n_steps or 10)) == 0):
                    if loss_value is None:
                        progress_cb(progress, f"MLX step {global_step}/{total_steps}")
                    else:
                        progress_cb(progress, f"MLX step {global_step}/{total_steps} loss={loss_value:.4f}")
        _write_mlx_adapter_config(
            latest_dir,
            rank=config.lora_rank,
            alpha=mlx_lora_alpha,
            dropout=config.lora_dropout,
        )
        backend["save_adapter"](model, str(adapter_file))
    else:
        train_kwargs = {
            "model": model,
            "optimizer": optimizer,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "args": training_args,
            "train_on_completions": True,
        }
        train_fn = backend["train"]
        train_kwargs = _filter_kwargs_for(train_fn, train_kwargs)
        train_fn(**train_kwargs)
        _write_mlx_adapter_config(
            latest_dir,
            rank=config.lora_rank,
            alpha=mlx_lora_alpha,
            dropout=config.lora_dropout,
        )
    if progress_cb:
        progress_cb(0.98, "MLX-VLM Qwen adapter saved")
    if metrics_cb:
        metrics_cb(
            {
                "runtime_platform": QWEN_PLATFORM_MLX,
                "training_backend": "mlx_vlm",
                "training_backend_api": backend.get("api", "modern"),
                "training_flavor": _mlx_training_flavor(config.model_id),
                "progress": 0.98,
            }
        )
    return QwenTrainingResult(
        config=config,
        checkpoints=[str(latest_dir)],
        latest_checkpoint=str(latest_dir),
        epochs_ran=epochs,
        metadata={
            "runtime_platform": QWEN_PLATFORM_MLX,
            "training_backend": "mlx_vlm",
            "training_backend_api": backend.get("api", "modern"),
            "training_flavor": _mlx_training_flavor(config.model_id),
            "adapter_file": str(adapter_file),
        },
    )


def train_qwen_model(
    config: QwenTrainingConfig,
    *,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_cb: Optional[CancelCallback] = None,
    metrics_cb: Optional[TelemetryCallback] = None,
) -> QwenTrainingResult:
    previous_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda_visible_changed = False
    if config.runtime_platform != QWEN_PLATFORM_MLX and config.devices:
        cleaned = ",".join(part.strip() for part in str(config.devices).split(",") if part.strip())
        if cleaned:
            os.environ["CUDA_VISIBLE_DEVICES"] = cleaned
            cuda_visible_changed = True
    try:
        if config.runtime_platform == QWEN_PLATFORM_MLX:
            return _train_mlx_lora(config, progress_cb, cancel_cb, metrics_cb)
        if config.training_mode == "trl_qlora":
            return _train_trl_qlora(config, progress_cb, cancel_cb, metrics_cb)
        return _train_official_lora(config, progress_cb, cancel_cb, metrics_cb)
    finally:
        if cuda_visible_changed:
            if previous_cuda_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda_visible


__all__ = [
    "QwenTrainingConfig",
    "QwenTrainingResult",
    "TrainingError",
    "train_qwen_model",
    "DEFAULT_SYSTEM_PROMPT",
]
