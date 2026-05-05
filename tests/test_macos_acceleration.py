from __future__ import annotations

from types import SimpleNamespace

import pytest

from services.sam3_runtime import (
    _resolve_sam3_device_impl,
    _resolve_sam3_mining_devices_impl,
)
from utils.gpu import _resolve_torch_inference_device_impl, _torch_mps_available_impl


class _FakeDevice:
    def __init__(self, name: str):
        self.name = name
        self.type = name.split(":", 1)[0]

    def __str__(self) -> str:
        return self.name


def _fake_torch(*, cuda: bool = False, mps: bool = False):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            device_count=lambda: 1 if cuda else 0,
        ),
        backends=SimpleNamespace(
            mps=SimpleNamespace(
                is_available=lambda: mps,
                is_built=lambda: mps,
            )
        ),
        device=lambda name: _FakeDevice(str(name)),
    )


def test_resolve_torch_inference_prefers_mps_when_cuda_absent() -> None:
    torch = _fake_torch(cuda=False, mps=True)

    assert _resolve_torch_inference_device_impl("auto", torch_module=torch) == "mps"


def test_resolve_torch_inference_can_disable_mps_preference() -> None:
    torch = _fake_torch(cuda=False, mps=True)

    assert (
        _resolve_torch_inference_device_impl(
            "auto",
            torch_module=torch,
            prefer_mps=False,
        )
        == "cpu"
    )


def test_resolve_torch_inference_rejects_unavailable_mps() -> None:
    torch = _fake_torch(cuda=False, mps=False)

    with pytest.raises(RuntimeError, match="mps_requested_but_unavailable"):
        _resolve_torch_inference_device_impl("mps", torch_module=torch)


def test_sam3_auto_device_uses_mps() -> None:
    torch = _fake_torch(cuda=False, mps=True)

    device = _resolve_sam3_device_impl(
        "auto",
        torch_module=torch,
        http_exception_cls=RuntimeError,
        http_400=400,
    )

    assert str(device) == "mps"


def test_sam3_mining_devices_fall_back_to_mps_before_cpu() -> None:
    torch = _fake_torch(cuda=False, mps=True)

    devices = _resolve_sam3_mining_devices_impl("auto", torch_module=torch, logger=SimpleNamespace(warning=lambda *_args: None))

    assert [str(device) for device in devices] == ["mps"]


def test_mps_available_probe_handles_missing_backend() -> None:
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), backends=SimpleNamespace())

    assert _torch_mps_available_impl(torch) is False
