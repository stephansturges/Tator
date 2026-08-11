"""Qwen generation helpers (token heuristics + logits processors)."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional, Set, Tuple

import torch

try:
    from transformers import LogitsProcessor, LogitsProcessorList
except Exception:  # noqa: BLE001
    LogitsProcessor = None
    LogitsProcessorList = None

_BASE_LOGITS_PROCESSOR = LogitsProcessor if LogitsProcessor is not None else object


def _resolve_qwen_max_seq_len(model: Any) -> Optional[int]:
    config = getattr(model, "config", None)
    if config is None:
        return None

    def _read_seq_len(cfg: Any) -> Optional[int]:
        if cfg is None:
            return None
        for attr in ("max_position_embeddings", "max_sequence_length", "seq_length"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        val = getattr(cfg, "max_length", None)
        if isinstance(val, int) and val > 0:
            return val
        return None

    for cfg in (getattr(config, "text_config", None), getattr(config, "language_config", None), config):
        val = _read_seq_len(cfg)
        if isinstance(val, int) and val >= 256:
            return val
    return None


def _qwen_estimate_vision_tokens(preview_inputs: Any) -> Optional[int]:
    grid = None
    if isinstance(preview_inputs, dict):
        grid = preview_inputs.get("image_grid_thw")
    else:
        grid = getattr(preview_inputs, "image_grid_thw", None)
    if grid is None:
        return None
    try:
        grid_vals = grid if isinstance(grid, torch.Tensor) else torch.as_tensor(grid)
        if grid_vals.ndim == 3:
            grid_vals = grid_vals[0]
        if grid_vals.ndim == 2 and grid_vals.shape[-1] == 3:
            tokens = (grid_vals[:, 0] * grid_vals[:, 1] * grid_vals[:, 2]).sum()
            return int(tokens.item())
        if grid_vals.ndim == 1 and grid_vals.numel() == 3:
            tokens = grid_vals[0] * grid_vals[1] * grid_vals[2]
            return int(tokens.item())
    except Exception:
        return None
    return None


def _qwen_effective_input_len(preview_inputs: Any, input_len: int, num_images: int) -> Tuple[int, Optional[int]]:
    vision_tokens = _qwen_estimate_vision_tokens(preview_inputs)
    if vision_tokens is None or num_images <= 0:
        return input_len, vision_tokens
    effective_len = max(1, input_len - num_images + vision_tokens)
    return effective_len, vision_tokens


def _qwen_supports_presence_penalty(model: Any) -> bool:
    gen_config = getattr(model, "generation_config", None)
    if gen_config is None:
        return False
    if hasattr(gen_config, "to_dict"):
        try:
            return "presence_penalty" in gen_config.to_dict()
        except Exception:
            pass
    return hasattr(gen_config, "presence_penalty")


class ThinkingEffortProcessor(_BASE_LOGITS_PROCESSOR):
    """Scale the </think> token logit to reduce or increase chain-of-thought length."""

    def __init__(self, end_thinking_token_id: int, thinking_effort: float = 1.0, scale_factor: float = 2.0):
        super().__init__()
        self.end_thinking_token_id = int(end_thinking_token_id)
        self.thinking_effort = float(thinking_effort)
        self.scale_factor = float(scale_factor)
        self.finished_sequences: Set[int] = set()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.end_thinking_token_id >= scores.size(1):
            return scores
        scale = self.scale_factor ** (1.0 - self.thinking_effort)
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            if i in self.finished_sequences:
                continue
            if (input_ids[i] == self.end_thinking_token_id).any():
                self.finished_sequences.add(i)
                continue
            scores[i, self.end_thinking_token_id] *= scale
        return scores


class ImmediateActionBiasProcessor(_BASE_LOGITS_PROCESSOR):
    """Boost </think> when 'wait' appears inside a think block after a minimum threshold."""

    def __init__(
        self,
        tokenizer: Any,
        end_thinking_token_id: int,
        *,
        min_think_chars: int = 200,
        min_think_seconds: float = 2.0,
        logit_bias: float = 6.0,
    ):
        super().__init__()
        self._tokenizer = tokenizer
        self.end_thinking_token_id = int(end_thinking_token_id)
        self.min_think_chars = max(1, int(min_think_chars))
        self.min_think_seconds = max(0.0, float(min_think_seconds))
        self.logit_bias = float(logit_bias)
        self._think_started_at: Dict[int, float] = {}
        self._wait_seen: Set[int] = set()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.logit_bias <= 0:
            return scores
        if self.end_thinking_token_id >= scores.size(1):
            return scores
        now = time.time()
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            if i in self._wait_seen:
                scores[i, self.end_thinking_token_id] += self.logit_bias
                continue
            try:
                text = self._tokenizer.decode(input_ids[i].tolist(), skip_special_tokens=False)
            except Exception:
                continue
            think_text = self._extract_open_think_text(text)
            if think_text is None:
                if i in self._think_started_at:
                    del self._think_started_at[i]
                continue
            if i not in self._think_started_at:
                self._think_started_at[i] = now
            if len(think_text) < self.min_think_chars:
                continue
            if (now - self._think_started_at.get(i, now)) < self.min_think_seconds:
                continue
            if re.search(r"\bwait\b", think_text, flags=re.IGNORECASE):
                self._wait_seen.add(i)
                scores[i, self.end_thinking_token_id] += self.logit_bias
        return scores

    @staticmethod
    def _extract_open_think_text(text: str) -> Optional[str]:
        if not text:
            return None
        if "<think>" not in text:
            return None
        after = text.split("<think>", 1)[1]
        if "</think>" in after:
            return None
        return after


def _qwen_find_end_think_token_id(tokenizer: Any) -> Optional[int]:
    if tokenizer is None:
        return None
    vocab_size = getattr(tokenizer, "vocab_size", None)
    vocab_size = int(vocab_size) if isinstance(vocab_size, int) and vocab_size > 0 else None
    candidates = [
        "</think>",
        "<|endofthink|>",
        "<|end_of_thought|>",
        "<|end_of_thinking|>",
    ]
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for token in candidates:
        try:
            tok_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            tok_id = None
        if tok_id is not None and tok_id != unk_id:
            if vocab_size is not None and int(tok_id) >= vocab_size:
                tok_id = None
            else:
                return int(tok_id)
        try:
            ids = tokenizer.encode(token, add_special_tokens=False)
        except Exception:
            ids = []
        if isinstance(ids, list) and len(ids) == 1:
            tok_id = int(ids[0])
            if vocab_size is None or tok_id < vocab_size:
                return tok_id
    return None


def _qwen_build_thinking_effort_processor(
    tokenizer: Any,
    thinking_effort: Optional[float],
    scale_factor: Optional[float],
) -> Optional[ThinkingEffortProcessor]:
    if LogitsProcessorList is None or LogitsProcessor is None:
        return None
    if thinking_effort is None:
        return None
    try:
        effort_val = float(thinking_effort)
    except (TypeError, ValueError):
        return None
    end_token_id = _qwen_find_end_think_token_id(tokenizer)
    if end_token_id is None:
        return None
    scale_val = 2.0
    if scale_factor is not None:
        try:
            scale_val = float(scale_factor)
        except (TypeError, ValueError):
            scale_val = 2.0
    return ThinkingEffortProcessor(end_token_id, thinking_effort=effort_val, scale_factor=scale_val)


def _qwen_build_immediate_action_processor(
    tokenizer: Any,
    immediate_action_bias: Optional[bool],
    min_think_chars: Optional[int],
    min_think_seconds: Optional[float],
    logit_bias: Optional[float],
) -> Optional[ImmediateActionBiasProcessor]:
    if LogitsProcessorList is None or LogitsProcessor is None:
        return None
    if not immediate_action_bias:
        return None
    end_token_id = _qwen_find_end_think_token_id(tokenizer)
    if end_token_id is None:
        return None
    chars_val = 200 if min_think_chars is None else int(min_think_chars)
    secs_val = 2.0 if min_think_seconds is None else float(min_think_seconds)
    bias_val = 6.0 if logit_bias is None else float(logit_bias)
    return ImmediateActionBiasProcessor(
        tokenizer,
        end_token_id,
        min_think_chars=chars_val,
        min_think_seconds=secs_val,
        logit_bias=bias_val,
    )


def _qwen_append_logits_processor(
    gen_kwargs: Dict[str, Any],
    processor: Optional[_BASE_LOGITS_PROCESSOR],
) -> None:
    if processor is None:
        return
    processors = gen_kwargs.get("logits_processor")
    if processors is None:
        gen_kwargs["logits_processor"] = LogitsProcessorList([processor])
    elif isinstance(processors, LogitsProcessorList):
        processors.append(processor)
    else:
        try:
            gen_kwargs["logits_processor"] = LogitsProcessorList(list(processors) + [processor])
        except Exception:
            gen_kwargs["logits_processor"] = LogitsProcessorList([processor])
