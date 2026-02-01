from __future__ import annotations

from typing import Any, Sequence


def _validate_cuda_device_ids_impl(
    device_ids: Sequence[int],
    *,
    torch_module: Any,
    http_exception_cls: Any,
) -> None:
    if not device_ids:
        return
    if not torch_module.cuda.is_available():
        raise http_exception_cls(status_code=400, detail="qwen_devices_unavailable")
    max_id = torch_module.cuda.device_count() - 1
    invalid = [device for device in device_ids if device < 0 or device > max_id]
    if invalid:
        raise http_exception_cls(
            status_code=400,
            detail=f"qwen_invalid_devices:available=0-{max_id}",
        )
