"""Training utilities for Qwen 3 VL fine-tuning.

Supports two training modes:
- official_lora: LoRA training with full-precision base model.
- trl_qlora: QLoRA training using TRL SFTTrainer (wired in step 5).
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )
except Exception as exc:  # noqa: BLE001
    AutoProcessor = None  # type: ignore[assignment]
    Qwen3VLForConditionalGeneration = None  # type: ignore[assignment]
    Trainer = None  # type: ignore[assignment]
    TrainingArguments = None  # type: ignore[assignment]
    TRANSFORMERS_IMPORT_ERROR = exc
else:
    TRANSFORMERS_IMPORT_ERROR = None

try:
    from peft import LoraConfig, get_peft_model
except Exception as exc:  # noqa: BLE001
    LoraConfig = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]
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
    training_mode: str = "official_lora"  # official_lora | trl_qlora
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    run_name: Optional[str] = None
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


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_image_path(dataset_root: Path, split: str, image_rel: str) -> Optional[Path]:
    rel_path = Path(image_rel)
    candidates = [
        dataset_root / split / "images" / rel_path,
        dataset_root / split / rel_path,
        dataset_root / "images" / rel_path,
        dataset_root / rel_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _conversation_to_messages(
    conversations: List[Dict[str, Any]],
    image: Image.Image,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
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
                if idx < len(parts) - 1:
                    content.append({"type": "image", "image": image})
            messages.append({"role": "user", "content": content})
        elif role == "system":
            if value:
                messages.append({"role": "system", "content": [{"type": "text", "text": value}]})
        else:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": value}]})
    return messages


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
        jsonl_path = dataset_root / split / "annotations.jsonl"
        if not jsonl_path.exists():
            raise TrainingError(f"qwen_annotations_missing:{jsonl_path}")
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
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
                    continue
                if "conversations" in payload:
                    self.entries.append(payload)
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

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        messages_list: List[List[Dict[str, Any]]] = []
        images_list: List[List[Image.Image]] = []
        prompt_lengths: List[int] = []
        for item in batch:
            messages = item["messages"]
            messages_list.append(messages)
            images_list.append(item["images"])
            prompt_messages = messages[:-1] if len(messages) > 1 else messages
            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
            ).input_ids
            prompt_lengths.append(len(prompt_ids))

        texts = [
            self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in messages_list
        ]
        inputs = self.processor(
            text=texts,
            images=images_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()
        for idx, length in enumerate(prompt_lengths):
            labels[idx, :length] = -100
        inputs["labels"] = labels
        return inputs


class _QwenTrainingCallback:
    def __init__(self, progress_cb: Optional[ProgressCallback], cancel_cb: Optional[CancelCallback], metrics_cb: Optional[TelemetryCallback]):
        self.progress_cb = progress_cb
        self.cancel_cb = cancel_cb
        self.metrics_cb = metrics_cb
        self.total_steps: Optional[int] = None

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

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        if self.cancel_cb and self.cancel_cb():
            control.should_training_stop = True
            control.should_save = True


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
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_id,
        torch_dtype=dtype if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    lora_cfg = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
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

    training_args = TrainingArguments(
        output_dir=str(result_path),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=max(1, config.batch_size),
        gradient_accumulation_steps=config.accumulate_grad_batches,
        learning_rate=config.lr,
        num_train_epochs=config.max_epochs,
        warmup_steps=config.warmup_steps,
        logging_steps=config.log_every_n_steps,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to=[],
        bf16=(device == "cuda" and torch.cuda.is_bf16_supported()),
        fp16=(device == "cuda" and not torch.cuda.is_bf16_supported()),
    )

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
    if PEFT_IMPORT_ERROR is not None or LoraConfig is None:
        raise TrainingError(f"qwen_peft_missing:{PEFT_IMPORT_ERROR}")
    _seed_all(config.seed)

    dataset_root = Path(config.dataset_root)
    result_path = Path(config.result_path)
    result_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise TrainingError("qwen_trl_qlora_requires_cuda")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    processor = AutoProcessor.from_pretrained(
        config.model_id,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_id,
        device_map="auto",
        quantization_config=quant_config,
    )

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
        target_modules=list(config.lora_target_modules),
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


def train_qwen_model(
    config: QwenTrainingConfig,
    *,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_cb: Optional[CancelCallback] = None,
    metrics_cb: Optional[TelemetryCallback] = None,
) -> QwenTrainingResult:
    if config.training_mode == "trl_qlora":
        return _train_trl_qlora(config, progress_cb, cancel_cb, metrics_cb)
    return _train_official_lora(config, progress_cb, cancel_cb, metrics_cb)


__all__ = [
    "QwenTrainingConfig",
    "QwenTrainingResult",
    "TrainingError",
    "train_qwen_model",
    "DEFAULT_SYSTEM_PROMPT",
]
