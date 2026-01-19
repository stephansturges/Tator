from __future__ import annotations

from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Union

from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.schema import ASSISTANT, FUNCTION, SYSTEM, USER, ContentItem, Message


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
        for item in content_items:
            payload = _content_item_to_qwen(item)
            if payload is not None:
                qwen_content.append(payload)
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
        response = self._chat_no_stream(messages, generate_cfg=generate_cfg)
        yield response

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        local = _lazy_localinferenceapi()
        qwen_messages = _messages_to_qwen(messages)
        max_new_tokens = generate_cfg.get("max_new_tokens")
        decode_override: Optional[Dict[str, Any]] = None
        if generate_cfg:
            decode_override = {
                k: v
                for k, v in generate_cfg.items()
                if k in {"temperature", "top_p", "top_k", "presence_penalty", "do_sample"}
            }
            if not decode_override:
                decode_override = None
        chat_template_kwargs = generate_cfg.get("chat_template_kwargs")
        output_text = local._run_qwen_chat(
            qwen_messages,
            max_new_tokens=max_new_tokens,
            decode_override=decode_override,
            model_id_override=self._model_id_override,
            tools=None,
            chat_template_kwargs=chat_template_kwargs,
        )
        return [Message(role=ASSISTANT, content=[ContentItem(text=output_text)])]
