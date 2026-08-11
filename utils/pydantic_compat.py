"""Compatibility helpers for Pydantic v1/v2 model APIs."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict, Mapping, Type

try:  # Pydantic v2.
    from pydantic import model_validator as _model_validator
except ImportError:  # pragma: no cover - exercised only with Pydantic v1.
    _model_validator = None

try:  # Prefer the non-deprecated v1 compatibility namespace when present.
    from pydantic.v1 import root_validator as _v1_root_validator
except ImportError:  # pragma: no cover - exercised only with Pydantic v1.
    from pydantic import root_validator as _v1_root_validator  # type: ignore[attr-defined]


def root_validator_compat(
    *,
    pre: bool = False,
    skip_on_failure: bool = False,
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Expose root-validator semantics without Pydantic v2 deprecation noise."""

    if _model_validator is None:
        return _v1_root_validator(pre=pre, skip_on_failure=skip_on_failure, **kwargs)

    if pre:
        def _decorate_before(fn: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(fn)
            def _wrapped(cls: Type[Any], values: Any) -> Any:
                return fn(cls, values)

            return _model_validator(mode="before")(classmethod(_wrapped))

        return _decorate_before

    def _decorate_after(fn: Callable[..., Any]) -> Callable[..., Any]:
        def _wrapped(self: Any) -> Any:
            values = dict(getattr(self, "__dict__", {}))
            updated = fn(type(self), values)
            if isinstance(updated, Mapping):
                for key, value in updated.items():
                    if hasattr(self, key):
                        object.__setattr__(self, key, value)
            return self

        _wrapped.__name__ = fn.__name__
        _wrapped.__qualname__ = fn.__qualname__
        _wrapped.__doc__ = fn.__doc__
        return _model_validator(mode="after")(_wrapped)

    return _decorate_after


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
