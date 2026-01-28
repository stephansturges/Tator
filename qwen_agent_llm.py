from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import copy

from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.schema import ASSISTANT, FUNCTION, SYSTEM, USER, ContentItem, Message
from qwen_agent.utils.utils import extract_text_from_message


def _lazy_localinferenceapi():
    import importlib

    return importlib.import_module("localinferenceapi")


def _message_content_items(content: Union[str, List[ContentItem]]) -> List[ContentItem]:
    if isinstance(content, list):
        return content
    return [ContentItem(text=str(content))]


def _content_item_to_qwen(item: ContentItem) -> Optional[Dict[str, Any]]:
    item_type, item_value = item.get_type_and_value()
    if item_type == "text":
        return {"type": "text", "text": str(item_value)}
    if item_type == "image":
        return {"type": "image", "image": item_value}
    if item_type == "audio":
        return {"type": "audio", "audio": item_value}
    if item_type == "video":
        return {"type": "video", "video": item_value}
    return None


def _messages_to_qwen(messages: Sequence[Message]) -> List[Dict[str, Any]]:
    qwen_messages: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.role == FUNCTION:
            role = "tool"
        else:
            role = msg.role
        content_items = _message_content_items(msg.content)
        qwen_content: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        has_non_text = False
        for item in content_items:
            item_type, item_value = item.get_type_and_value()
            if item_type == "text":
                text_parts.append(str(item_value))
                continue
            has_non_text = True
            payload = _content_item_to_qwen(item)
            if payload is not None:
                qwen_content.append(payload)
        if text_parts:
            if has_non_text:
                for chunk in text_parts:
                    qwen_content.append({"type": "text", "text": chunk})
            else:
                qwen_content = [{"type": "text", "text": "".join(text_parts)}]
        qwen_messages.append({"role": role, "content": qwen_content})
    return qwen_messages


class LocalQwenVLChatModel(BaseFnCallModel):
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__(cfg=cfg)
        cfg = cfg or {}
        self._model_id_override = cfg.get("model_id_override")

    @property
    def support_multimodal_input(self) -> bool:
        return True

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        local = _lazy_localinferenceapi()
        qwen_messages = _messages_to_qwen(messages)
        max_new_tokens = generate_cfg.get("max_new_tokens")
        assistant_prefix = generate_cfg.get("assistant_prefix")
        decode_override: Optional[Dict[str, Any]] = None
        if generate_cfg:
            decode_override = {
                k: v
                for k, v in generate_cfg.items()
                if k in {"temperature", "top_p", "top_k", "presence_penalty", "do_sample"}
            }
            if not decode_override:
                decode_override = None
        thinking_effort = generate_cfg.get("thinking_effort") if generate_cfg else None
        thinking_scale_factor = generate_cfg.get("thinking_scale_factor") if generate_cfg else None
        immediate_action_bias = generate_cfg.get("immediate_action_bias") if generate_cfg else None
        immediate_action_min_chars = generate_cfg.get("immediate_action_min_chars") if generate_cfg else None
        immediate_action_min_seconds = generate_cfg.get("immediate_action_min_seconds") if generate_cfg else None
        immediate_action_logit_bias = generate_cfg.get("immediate_action_logit_bias") if generate_cfg else None
        chat_template_kwargs = generate_cfg.get("chat_template_kwargs")
        stream = local._run_qwen_chat_stream(
            qwen_messages,
            max_new_tokens=max_new_tokens,
            decode_override=decode_override,
            model_id_override=self._model_id_override,
            tools=None,
            chat_template_kwargs=chat_template_kwargs,
            assistant_prefix=assistant_prefix,
            thinking_effort=thinking_effort,
            thinking_scale_factor=thinking_scale_factor,
            immediate_action_bias=immediate_action_bias,
            immediate_action_min_chars=immediate_action_min_chars,
            immediate_action_min_seconds=immediate_action_min_seconds,
            immediate_action_logit_bias=immediate_action_logit_bias,
        )
        for partial in stream:
            yield [Message(role=ASSISTANT, content=[ContentItem(text=partial)])]

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        local = _lazy_localinferenceapi()
        qwen_messages = _messages_to_qwen(messages)
        max_new_tokens = generate_cfg.get("max_new_tokens")
        assistant_prefix = generate_cfg.get("assistant_prefix")
        decode_override: Optional[Dict[str, Any]] = None
        if generate_cfg:
            decode_override = {
                k: v
                for k, v in generate_cfg.items()
                if k in {"temperature", "top_p", "top_k", "presence_penalty", "do_sample"}
            }
            if not decode_override:
                decode_override = None
        thinking_effort = generate_cfg.get("thinking_effort") if generate_cfg else None
        thinking_scale_factor = generate_cfg.get("thinking_scale_factor") if generate_cfg else None
        immediate_action_bias = generate_cfg.get("immediate_action_bias") if generate_cfg else None
        immediate_action_min_chars = generate_cfg.get("immediate_action_min_chars") if generate_cfg else None
        immediate_action_min_seconds = generate_cfg.get("immediate_action_min_seconds") if generate_cfg else None
        immediate_action_logit_bias = generate_cfg.get("immediate_action_logit_bias") if generate_cfg else None
        chat_template_kwargs = generate_cfg.get("chat_template_kwargs")
        output_text = local._run_qwen_chat(
            qwen_messages,
            max_new_tokens=max_new_tokens,
            decode_override=decode_override,
            model_id_override=self._model_id_override,
            tools=None,
            chat_template_kwargs=chat_template_kwargs,
            assistant_prefix=assistant_prefix,
            thinking_effort=thinking_effort,
            thinking_scale_factor=thinking_scale_factor,
            immediate_action_bias=immediate_action_bias,
            immediate_action_min_chars=immediate_action_min_chars,
            immediate_action_min_seconds=immediate_action_min_seconds,
            immediate_action_logit_bias=immediate_action_logit_bias,
        )
        return [Message(role=ASSISTANT, content=[ContentItem(text=output_text)])]

    def _continue_assistant_response(
        self,
        messages: List[Message],
        generate_cfg: dict,
        stream: bool,
    ) -> Iterator[List[Message]]:
        messages = copy.deepcopy(messages)
        assistant_prefix = None
        if messages and messages[-1].role == ASSISTANT:
            assistant_prefix = extract_text_from_message(messages[-1], add_upload_info=False)
            if assistant_prefix:
                messages = messages[:-1]
        gen_cfg = dict(generate_cfg)
        if assistant_prefix:
            gen_cfg["assistant_prefix"] = assistant_prefix
        return self._chat(messages, stream=stream, delta_stream=False, generate_cfg=gen_cfg)
