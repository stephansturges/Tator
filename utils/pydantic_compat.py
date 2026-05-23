"""Compatibility helpers for Pydantic v1/v2 model APIs."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Type


def model_copy_update(model: Any, updates: Mapping[str, Any], **kwargs: Any) -> Any:
    update_dict = dict(updates or {})
    if hasattr(model, "model_copy"):
        return model.model_copy(update=update_dict, **kwargs)
    if hasattr(model, "copy"):
        return model.copy(update=update_dict, **kwargs)
    if isinstance(model, Mapping):
        copied = dict(model)
        copied.update(update_dict)
        return copied
    raise TypeError(f"model_copy_update_unsupported:{type(model).__name__}")


def model_dump_compat(model: Any, **kwargs: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return dict(model.model_dump(**kwargs))
    if hasattr(model, "dict"):
        return dict(model.dict(**kwargs))
    if isinstance(model, Mapping):
        return dict(model)
    raise TypeError(f"model_dump_unsupported:{type(model).__name__}")


def model_validate_compat(model_cls: Type[Any], payload: Any) -> Any:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    if hasattr(model_cls, "parse_obj"):
        return model_cls.parse_obj(payload)
    if isinstance(payload, Mapping):
        return model_cls(**payload)
    return model_cls(payload)
