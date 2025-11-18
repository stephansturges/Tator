"""Training utilities for Qwen 2.5 VL fine-tuning.

This module mirrors the workflow described in the Qwen fine-tuning blog post
so both the CLI and FastAPI layers can launch consistent training runs.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import lightning as L
    from lightning.pytorch.callbacks import Callback, EarlyStopping
except ImportError as exc:  # noqa: BLE001
    L = None
    Callback = None  # type: ignore[assignment]
    EarlyStopping = None  # type: ignore[assignment]
    LIGHTNING_IMPORT_ERROR = exc
else:
    LIGHTNING_IMPORT_ERROR = None

try:
    from nltk import edit_distance
except ImportError as exc:  # noqa: BLE001
    edit_distance = None  # type: ignore[assignment]
    NLTK_IMPORT_ERROR = exc
else:
    NLTK_IMPORT_ERROR = None

LIGHTNING_AVAILABLE = LIGHTNING_IMPORT_ERROR is None
NLTK_AVAILABLE = NLTK_IMPORT_ERROR is None

if not LIGHTNING_AVAILABLE:
    class _FallbackLightningModule:  # pragma: no cover - placeholder when Lightning missing
        pass

    class _FallbackLightning:
        LightningModule = _FallbackLightningModule

    class _FallbackCallback:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            pass

    class _FallbackEarlyStopping(_FallbackCallback):  # pragma: no cover
        pass

    L = _FallbackLightning()  # type: ignore[assignment]
    Callback = _FallbackCallback  # type: ignore[assignment]
    EarlyStopping = _FallbackEarlyStopping  # type: ignore[assignment]

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)


logger = logging.getLogger(__name__)


ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], bool]
TelemetryCallback = Callable[[Dict[str, Any]], None]


# Default system prompt teaches Qwen to emit JSON detections for either modality.
DEFAULT_SYSTEM_PROMPT = (
    "You are an annotation assistant that only returns JSON objects shaped like {\"detections\":[{\"label\":\"class\","
    "\"bbox\":[x1,y1,x2,y2]} or {\"label\":\"class\",\"point\":[x,y]}]}. Always reply with compact JSON and no prose."
)


class TrainingError(RuntimeError):
    """Raised when the Qwen training pipeline fails in a recoverable way."""


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch.cuda, "manual_seed"):
        try:
            torch.cuda.manual_seed(seed)
        except Exception:  # noqa: BLE001
            pass
    if hasattr(torch.cuda, "manual_seed_all"):
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:  # noqa: BLE001
            pass
    if hasattr(L, "seed_everything"):
        try:
            L.seed_everything(seed, workers=True)
        except Exception:  # noqa: BLE001
            pass


def _make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    def _worker_init(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return _worker_init


@dataclass
class QwenTrainingConfig:
    dataset_root: str
    result_path: str
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    system_prompt_noise: float = 0.05
    run_name: Optional[str] = None
    batch_size: int = 1
    max_epochs: int = 10
    lr: float = 2e-4
    accumulate_grad_batches: int = 8
    check_val_every_n_epoch: int = 2
    gradient_clip_val: float = 1.0
    warmup_steps: int = 50
    num_workers: int = 0
    use_qlora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Sequence[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    accelerator: str = "auto"
    devices: Optional[Sequence[int]] = None
    device_map: Optional[object] = None
    limit_val_batches: int = 1
    log_every_n_steps: int = 10
    patience: int = 3
    seed: int = 1337
    max_detections_per_sample: int = 200
    max_image_dim: int = 1024


@dataclass
class QwenTrainingResult:
    config: QwenTrainingConfig
    checkpoints: List[str]
    latest_checkpoint: Optional[str]
    epochs_ran: int
    metadata: Dict[str, object] = field(default_factory=dict)


def _json_dumps(data: object) -> str:
    return json.dumps(data, ensure_ascii=False)


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


class JSONLDataset(Dataset):
    """Loads JSONL annotations + images and emits chat-formatted entries."""

    def __init__(
        self,
        jsonl_path: Path,
        image_root: Path,
        system_prompt: str,
        prompt_noise: float = 0.05,
        seed: Optional[int] = None,
        max_detections: int = 200,
        max_image_dim: int = 1024,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.image_root = image_root
        self.system_prompt = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip() or DEFAULT_SYSTEM_PROMPT
        # Cap the noise ratio to avoid destroying the prompt entirely.
        self.prompt_noise = max(0.0, min(float(prompt_noise), 0.3))
        self.entries = self._load_entries()
        self.rng = random.Random(seed)
        self.max_detections = max(0, int(max_detections))
        try:
            max_dim = int(max_image_dim)
        except (TypeError, ValueError):
            max_dim = 1024
        self.max_image_dim = max(64, min(max_dim, 4096))

    def _load_entries(self) -> List[Dict[str, object]]:
        if not self.jsonl_path.exists():
            raise TrainingError(f"annotations file not found: {self.jsonl_path}")
        entries: List[Dict[str, object]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                entries.append(data)
        if not entries:
            raise TrainingError(f"annotations file is empty: {self.jsonl_path}")
        return entries

    def __len__(self) -> int:  # noqa: D401
        return len(self.entries)

    def __getitem__(self, index: int) -> Tuple[Image.Image, Dict[str, object], List[Dict[str, object]]]:
        if index < 0 or index >= len(self.entries):
            raise IndexError("Index out of range")
        entry = self.entries[index]
        image_rel = entry.get("image")
        if not isinstance(image_rel, str):
            raise TrainingError("Each annotation must include an 'image' key (string).")
        image_path = self.image_root / image_rel
        if not image_path.exists():
            raise TrainingError(f"Missing image referenced in annotations: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image = self._resize_image_if_needed(image)
        conversation, target_text = self._build_conversation(image, entry)
        payload = {
            "image": image_rel,
            "suffix": target_text,
        }
        return image, payload, conversation

    def _resize_image_if_needed(self, image: Image.Image) -> Image.Image:
        max_dim = self.max_image_dim or 1024
        width, height = image.size
        if width <= max_dim and height <= max_dim:
            return image
        ratio = max(width / max_dim, height / max_dim)
        new_size = (max(1, int(width / ratio)), max(1, int(height / ratio)))
        return image.resize(new_size, Image.BICUBIC)

    def _build_conversation(self, image: Image.Image, entry: Dict[str, object]) -> Tuple[List[Dict[str, object]], str]:
        context = str(entry.get("context") or "").strip()
        detections = entry.get("detections") or []
        if not isinstance(detections, list):
            detections = []
        prompt_labels, label_mode = self._select_label_prompt(detections)
        filtered_detections = self._filter_detections_for_prompt(detections, prompt_labels, label_mode)
        filtered_detections = self._apply_detection_budget(filtered_detections)
        use_bbox = self.rng.random() < 0.5
        system_prompt = self._apply_prompt_noise(self.system_prompt)
        if use_bbox:
            user_text = self._build_bbox_prompt(context, prompt_labels, label_mode)
            target_payload = self._format_bbox_targets(filtered_detections)
        else:
            user_text = self._build_point_prompt(context, prompt_labels, label_mode)
            target_payload = self._format_point_targets(filtered_detections)
        target_text = json.dumps({"detections": target_payload}, ensure_ascii=False)
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            },
        ]
        return conversation, target_text

    def _build_bbox_prompt(self, context: str, labels: List[str], mode: str) -> str:
        body = (
            "Return a JSON object named \"detections\". Each detection must include \"label\" and "
            "\"bbox\" as [x1,y1,x2,y2] pixel coordinates (integers). If nothing is present, respond with {\"detections\": []}. Respond with JSON only."
        )
        return self._compose_user_prompt(context, body, labels, mode)

    def _build_point_prompt(self, context: str, labels: List[str], mode: str) -> str:
        body = (
            "Return a JSON object named \"detections\". Each detection must include \"label\" and \"point\" as [x,y] pixel coordinates near the object center. "
            "If nothing is present, respond with {\"detections\": []}. Respond with JSON only."
        )
        return self._compose_user_prompt(context, body, labels, mode)

    def _compose_user_prompt(self, context: str, body: str, labels: List[str], mode: str) -> str:
        pieces: List[str] = []
        if context:
            pieces.append(context)
        clause = ""
        if labels:
            label_text = ", ".join(labels)
            if mode == "single":
                clause = f"Focus only on the class '{labels[0]}'."
            elif mode == "subset":
                clause = f"Focus only on these classes: {label_text}."
            else:
                clause = f"Return detections for these classes: {label_text}."
        else:
            clause = "Return detections for every labeled object."
        pieces.append(clause)
        pieces.append(body)
        return " ".join(part for part in pieces if part).strip()

    def _select_label_prompt(self, detections: List[Dict[str, object]]) -> Tuple[List[str], str]:
        labels = sorted({str(det.get("label", "")).strip() for det in detections if str(det.get("label", "")).strip()})
        if not labels:
            return [], "all"
        if len(labels) == 1:
            if self.rng.random() < 0.5:
                return labels, "single"
            return labels, "all"
        roll = self.rng.random()
        if roll < 0.34:
            return labels, "all"
        if roll < 0.67:
            return [self.rng.choice(labels)], "single"
        subset_size = self.rng.randint(2, len(labels))
        subset = sorted(self.rng.sample(labels, subset_size))
        return subset, "subset"

    def _filter_detections_for_prompt(
        self,
        detections: List[Dict[str, object]],
        labels: List[str],
        mode: str,
    ) -> List[Dict[str, object]]:
        if not labels or mode == "all":
            return detections
        normalized = {label.strip() for label in labels if label.strip()}
        filtered = [det for det in detections if str(det.get("label", "")).strip() in normalized]
        return filtered

    def _apply_prompt_noise(self, prompt: str) -> str:
        if self.prompt_noise <= 0 or len(prompt) < 2:
            return prompt
        total_chars = len(prompt)
        removal_count = max(1, int(total_chars * self.prompt_noise))
        removal_count = min(removal_count, total_chars - 1)
        indices = sorted(self.rng.sample(range(total_chars), removal_count), reverse=True)
        chars = list(prompt)
        for idx in indices:
            del chars[idx]
        return "".join(chars)

    def _format_bbox_targets(self, detections: List[Dict[str, object]]) -> List[Dict[str, object]]:
        formatted: List[Dict[str, object]] = []
        for det in detections:
            label = str(det.get("label", ""))
            bbox = det.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            try:
                x1 = int(round(float(bbox[0])))
                y1 = int(round(float(bbox[1])))
                x2 = int(round(float(bbox[2])))
                y2 = int(round(float(bbox[3])))
            except (TypeError, ValueError):
                continue
            formatted.append({"label": label, "bbox": [x1, y1, x2, y2]})
        return formatted

    def _apply_detection_budget(self, detections: List[Dict[str, object]]) -> List[Dict[str, object]]:
        budget = self.max_detections
        if budget <= 0 or len(detections) <= budget:
            return detections
        buckets: Dict[str, List[Dict[str, object]]] = {}
        for det in detections:
            label = str(det.get("label", "")).strip() or "__unlabeled__"
            buckets.setdefault(label, []).append(det)
        items = list(buckets.items())
        self.rng.shuffle(items)
        selected: List[Dict[str, object]] = []
        remaining = budget
        for label, bucket in items:
            count = len(bucket)
            if count <= remaining:
                selected.extend(bucket)
                remaining -= count
            elif not selected:
                selected.extend(bucket[:remaining])
                remaining = 0
            if remaining <= 0:
                break
        if not selected:
            # Fallback to the largest bucket if everything else failed.
            label, bucket = max(items, key=lambda entry: len(entry[1]))
            selected.extend(bucket[:budget])
        return selected

    def _format_point_targets(self, detections: List[Dict[str, object]]) -> List[Dict[str, object]]:
        formatted: List[Dict[str, object]] = []
        for det in detections:
            label = str(det.get("label", ""))
            point = det.get("point")
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    x_val = int(round(float(point[0])))
                    y_val = int(round(float(point[1])))
                    formatted.append({"label": label, "point": [x_val, y_val]})
                    continue
                except (TypeError, ValueError):
                    pass
            bbox = det.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                try:
                    x1 = float(bbox[0])
                    y1 = float(bbox[1])
                    x2 = float(bbox[2])
                    y2 = float(bbox[3])
                    cx = int(round((x1 + x2) / 2.0))
                    cy = int(round((y1 + y2) / 2.0))
                    formatted.append({"label": label, "point": [cx, cy]})
                except (TypeError, ValueError):
                    continue
        return formatted


class TrainCollator:
    def __init__(self, processor: Qwen2_5_VLProcessor) -> None:
        self.processor = processor

    def __call__(self, batch):
        _, _, conversations = zip(*batch)
        texts = [self.processor.apply_chat_template(item, tokenize=False) for item in conversations]
        image_inputs = [process_vision_info(item)[0] for item in conversations]
        model_inputs = self.processor(
            text=texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        )
        labels = model_inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        # Mask image tokens in the loss.
        image_tokens = [151652, 151653, 151655]
        for token_id in image_tokens:
            labels[labels == token_id] = -100
        return (
            model_inputs["input_ids"],
            model_inputs["attention_mask"],
            model_inputs["pixel_values"],
            model_inputs["image_grid_thw"],
            labels,
        )


class EvalCollator:
    def __init__(self, processor: Qwen2_5_VLProcessor) -> None:
        self.processor = processor

    def __call__(self, batch):
        _, entries, conversations = zip(*batch)
        suffixes = [entry.get("suffix", "") for entry in entries]
        prompt_only = [conv[:2] for conv in conversations]
        texts = [self.processor.apply_chat_template(item, tokenize=False) for item in prompt_only]
        image_inputs = [process_vision_info(item)[0] for item in prompt_only]
        model_inputs = self.processor(
            text=texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        )
        return (
            model_inputs["input_ids"],
            model_inputs["attention_mask"],
            model_inputs["pixel_values"],
            model_inputs["image_grid_thw"],
            suffixes,
        )


class QwenLightningModule(L.LightningModule):
    def __init__(self, model, processor, config: QwenTrainingConfig):
        super().__init__()
        self.model = model
        self.processor = processor
        self.config = config

    def training_step(self, batch, batch_idx):  # noqa: D401
        input_ids, attention_mask, pixel_values, image_grid_thw, labels = batch
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):  # noqa: D401
        input_ids, attention_mask, pixel_values, image_grid_thw, suffixes = batch
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=1024,
        )
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
        generated_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        scores: List[float] = []
        for predicted, target in zip(generated_texts, suffixes):
            target_text = target if isinstance(target, str) else _json_dumps(target)
            score = edit_distance(predicted, target_text) / max(len(predicted) or 1, len(target_text) or 1)
            scores.append(score)
        avg_score = sum(scores) / max(len(scores), 1)
        self.log("val_edit_distance", avg_score, prog_bar=True, on_epoch=True, batch_size=self.config.batch_size)
        return avg_score

    def configure_optimizers(self):  # noqa: D401
        return AdamW(self.model.parameters(), lr=self.config.lr)


class SaveCheckpoint(Callback):
    def __init__(self, result_path: Path) -> None:
        self.result_path = result_path
        self.epoch = 0

    def on_train_epoch_end(self, trainer, pl_module):  # noqa: D401
        checkpoint_path = self.result_path / str(self.epoch)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        pl_module.processor.save_pretrained(checkpoint_path)
        pl_module.model.save_pretrained(checkpoint_path)
        logger.info("Saved checkpoint to %s", checkpoint_path)
        self.epoch += 1

    def on_train_end(self, trainer, pl_module):  # noqa: D401
        checkpoint_path = self.result_path / "latest"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        pl_module.processor.save_pretrained(checkpoint_path)
        pl_module.model.save_pretrained(checkpoint_path)
        logger.info("Saved checkpoint to %s", checkpoint_path)


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:  # noqa: BLE001
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class ProgressReporter(Callback):
    def __init__(
        self,
        progress_cb: Optional[ProgressCallback],
        metrics_cb: Optional[TelemetryCallback],
        total_epochs: int,
        batches_per_epoch: int,
    ) -> None:
        self.progress_cb = progress_cb
        self.metrics_cb = metrics_cb
        self.total_epochs = max(1, total_epochs)
        self.batches_per_epoch = max(1, batches_per_epoch)
        self.progress_interval = 1
        self.metric_interval = 1
        self._recalculate_intervals(self.batches_per_epoch)

    def _recalculate_intervals(self, total_batches: int) -> None:
        self.batches_per_epoch = max(1, total_batches)
        self.progress_interval = 1
        self.metric_interval = 1

    def on_train_epoch_start(self, trainer, pl_module):  # noqa: D401
        total_batches = getattr(trainer, "num_training_batches", None)
        if isinstance(total_batches, int) and total_batches > 0:
            self._recalculate_intervals(total_batches)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):  # noqa: D401
        total_batches = max(1, self.batches_per_epoch)
        step_in_epoch = min(batch_idx + 1, total_batches)
        epoch_index = max(0, trainer.current_epoch)
        epoch_progress = step_in_epoch / total_batches
        global_progress = min(0.99, (epoch_index + epoch_progress) / self.total_epochs)
        metrics = getattr(trainer, "callback_metrics", {}) or {}
        train_loss = _safe_float(metrics.get("train_loss"))
        emit_progress = True
        emit_metrics = True
        message = f"Epoch {epoch_index + 1}/{self.total_epochs} • batch {step_in_epoch}/{total_batches}"
        if train_loss is not None:
            message += f" • loss {train_loss:.4f}"
        if emit_progress and self.progress_cb:
            try:
                self.progress_cb(global_progress, message)
            except Exception:  # noqa: BLE001
                pass
        if emit_metrics and self.metrics_cb and train_loss is not None:
            payload = {
                "timestamp": time.time(),
                "phase": "train",
                "epoch": epoch_index + 1,
                "total_epochs": self.total_epochs,
                "batch": step_in_epoch,
                "batches_per_epoch": total_batches,
                "step": int(getattr(trainer, "global_step", 0)),
                "train_loss": train_loss,
                "epoch_progress": epoch_progress,
                "progress": global_progress,
            }
            try:
                self.metrics_cb(payload)
            except Exception:  # noqa: BLE001
                pass

    def on_validation_epoch_end(self, trainer, pl_module):  # noqa: D401
        metrics = getattr(trainer, "callback_metrics", {}) or {}
        val_metric = _safe_float(metrics.get("val_edit_distance"))
        epoch_index = max(0, trainer.current_epoch)
        progress = min(0.99, (epoch_index + 1) / self.total_epochs)
        if self.metrics_cb and val_metric is not None:
            payload = {
                "timestamp": time.time(),
                "phase": "val",
                "epoch": epoch_index + 1,
                "total_epochs": self.total_epochs,
                "metric": "val_edit_distance",
                "value": val_metric,
                "progress": progress,
                "epoch_progress": 1.0,
            }
            try:
                self.metrics_cb(payload)
            except Exception:  # noqa: BLE001
                pass
        if self.progress_cb:
            message = (
                f"Epoch {epoch_index + 1}/{self.total_epochs} validation edit distance {val_metric:.4f}"
                if val_metric is not None
                else f"Epoch {epoch_index + 1}/{self.total_epochs} validation complete"
            )
            try:
                self.progress_cb(progress, message)
            except Exception:  # noqa: BLE001
                pass

    def on_train_epoch_end(self, trainer, pl_module):  # noqa: D401
        if not self.progress_cb:
            return
        fraction = (trainer.current_epoch + 1) / self.total_epochs
        message = f"Completed epoch {trainer.current_epoch + 1}/{self.total_epochs}"
        try:
            self.progress_cb(min(0.99, fraction), message)
        except Exception:  # noqa: BLE001
            pass


class CancellationWatcher(Callback):
    def __init__(self, cancel_cb: Optional[CancelCallback]) -> None:
        self.cancel_cb = cancel_cb

    def _should_cancel(self) -> bool:
        if not self.cancel_cb:
            return False
        try:
            return bool(self.cancel_cb())
        except Exception:  # noqa: BLE001
            return False

    def on_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):  # noqa: D401
        if self._should_cancel():
            trainer.should_stop = True
            raise KeyboardInterrupt("Training cancelled")


def _load_model(config: QwenTrainingConfig):
    quant_config = None
    if config.use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=torch.bfloat16,
        )
    device_map = config.device_map
    if not device_map:
        if config.devices and isinstance(config.devices, (tuple, list)) and len(config.devices) > 1:
            device_map = "auto"
        elif config.devices and isinstance(config.devices, (tuple, list)):
            device_map = {"": int(config.devices[0])}
        elif isinstance(config.devices, int):
            device_map = {"": int(config.devices)}
        elif config.use_qlora:
            device_map = {"": 0}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_id,
        device_map=device_map,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )
    if config.use_qlora:
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(config.target_modules),
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    return model


def _prepare_datasets(config: QwenTrainingConfig) -> Tuple[JSONLDataset, JSONLDataset]:
    dataset_root = Path(config.dataset_root)
    train_jsonl = dataset_root / "train" / "annotations.jsonl"
    val_jsonl = dataset_root / "val" / "annotations.jsonl"
    noise = max(0.0, min(float(config.system_prompt_noise), 0.3))
    base_seed = int(config.seed or 0)
    train_ds = JSONLDataset(
        train_jsonl,
        dataset_root / "train",
        config.system_prompt,
        noise,
        seed=base_seed,
        max_detections=config.max_detections_per_sample,
        max_image_dim=config.max_image_dim,
    )
    val_ds = JSONLDataset(
        val_jsonl,
        dataset_root / "val",
        config.system_prompt,
        noise,
        seed=base_seed + 1,
        max_detections=config.max_detections_per_sample,
        max_image_dim=config.max_image_dim,
    )
    return train_ds, val_ds


def train_qwen_model(
    config: QwenTrainingConfig,
    *,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_cb: Optional[CancelCallback] = None,
    metrics_cb: Optional[TelemetryCallback] = None,
) -> QwenTrainingResult:
    """Fine-tune Qwen 2.5 VL based on the supplied configuration."""

    if not LIGHTNING_AVAILABLE:
        raise TrainingError(
            "lightning_not_installed: install 'lightning' to enable Qwen training (pip install lightning)"
        )
    if not NLTK_AVAILABLE:
        raise TrainingError(
            "nltk_not_installed: install 'nltk' to enable Qwen training (pip install nltk)"
        )
    _ensure_dir(config.result_path)
    base_seed = int(config.seed or 0)
    _seed_all(base_seed)
    train_ds, val_ds = _prepare_datasets(config)
    dataset_meta = _load_dataset_metadata(config.dataset_root)
    processor = Qwen2_5_VLProcessor.from_pretrained(
        config.model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    train_worker_init = _make_worker_init_fn(base_seed)
    val_worker_init = _make_worker_init_fn(base_seed + 1)
    train_generator = torch.Generator()
    train_generator.manual_seed(base_seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(base_seed + 1)
    train_loader_kwargs = {
        "batch_size": config.batch_size,
        "shuffle": True,
        "num_workers": config.num_workers,
        "collate_fn": TrainCollator(processor),
        "generator": train_generator,
    }
    if config.num_workers > 0:
        train_loader_kwargs["worker_init_fn"] = train_worker_init
    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    eval_loader_kwargs = {
        "batch_size": config.batch_size,
        "shuffle": False,
        "num_workers": config.num_workers,
        "collate_fn": EvalCollator(processor),
        "generator": val_generator,
    }
    if config.num_workers > 0:
        eval_loader_kwargs["worker_init_fn"] = val_worker_init
    eval_loader = DataLoader(val_ds, **eval_loader_kwargs)

    model = _load_model(config)
    lightning_module = QwenLightningModule(model, processor, config)
    result_path = Path(config.result_path)
    try:
        batches_per_epoch = len(train_loader)
    except TypeError:
        batches_per_epoch = 1
    callbacks: List[Callback] = [
        SaveCheckpoint(result_path),
        ProgressReporter(progress_cb, metrics_cb, config.max_epochs, batches_per_epoch),
        CancellationWatcher(cancel_cb),
        EarlyStopping(monitor="val_edit_distance", patience=config.patience, mode="min"),
    ]

    devices = config.devices or 1

    trainer = L.Trainer(
        accelerator=config.accelerator,
        devices=devices,
        max_epochs=config.max_epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        gradient_clip_val=config.gradient_clip_val,
        limit_val_batches=config.limit_val_batches,
        num_sanity_val_steps=0,
        log_every_n_steps=config.log_every_n_steps,
        callbacks=callbacks,
        deterministic=True,
    )

    try:
        train_result = trainer.fit(lightning_module, train_loader, eval_loader)
    except KeyboardInterrupt as exc:
        raise TrainingError("Training cancelled") from exc
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(f"Training failed: {exc}") from exc

    checkpoints = [str(path) for path in sorted(result_path.glob("*/"))]
    latest = result_path / "latest"
    if progress_cb:
        try:
            progress_cb(1.0, "Training complete")
        except Exception:  # noqa: BLE001
            pass
    epochs_run = getattr(trainer, "current_epoch", config.max_epochs)
    return QwenTrainingResult(
        config=config,
        checkpoints=checkpoints,
        latest_checkpoint=str(latest) if latest.exists() else None,
        epochs_ran=epochs_run,
        metadata=dataset_meta,
    )


def _load_dataset_metadata(dataset_root: str) -> Dict[str, object]:
    root = Path(dataset_root)
    meta_path = root / "dataset_meta.json"
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, dict):
                    return data
        except Exception:  # noqa: BLE001
            pass
    fallback_context = ""
    fallback_classes: List[str] = []
    annotations = root / "train" / "annotations.jsonl"
    if annotations.exists():
        try:
            with annotations.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    fallback_context = str(data.get("context") or "")
                    dets = data.get("detections") or []
                    labels = sorted(
                        {
                            str(det.get("label", "")).strip()
                            for det in dets
                            if isinstance(det, dict) and str(det.get("label", "")).strip()
                        }
                    )
                    fallback_classes = labels
                    break
        except Exception:  # noqa: BLE001
            fallback_context = ""
            fallback_classes = []
    return {"context": fallback_context, "classes": fallback_classes}


__all__ = [
    "QwenTrainingConfig",
    "QwenTrainingResult",
    "train_qwen_model",
    "TrainingError",
    "DEFAULT_SYSTEM_PROMPT",
]
