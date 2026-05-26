"""Optional MLX-DINOv3 worker integration for Apple Silicon.

The backend keeps Torch/MPS as the portable DINOv3 implementation.  This module
adds a process-isolated Swift/MLX worker that can be selected explicitly, or by
``DINOV3_BACKEND=auto`` when the worker and converted model are already present.
"""

from __future__ import annotations

import atexit
import json
import os
import platform
import selectors
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


MLX_DINOV3_REPO_COMMIT = "3122d7905cca21012b4c249e8ddad19ff78f54bc"
MLX_DINOV3_CONVERSION_VERSION = "mlx-dinov3-vit-v1"
MLX_DINOV3_DEFAULT_TIMEOUT_SECONDS = 300.0


class MlxDinoV3Unavailable(RuntimeError):
    """Raised when MLX-DINOv3 is requested but cannot be used."""


class MlxDinoV3WorkerError(RuntimeError):
    """Raised when the MLX-DINOv3 worker fails after it was selected."""


@dataclass(frozen=True)
class MlxDinoV3BackendStatus:
    requested: str
    resolved: str
    available: bool
    platform_supported: bool
    worker_path: str
    worker_exists: bool
    metallib_exists: bool
    model_dir: str
    model_exists: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "default": "auto",
            "requested": self.requested,
            "resolved": self.resolved,
            "available": self.available,
            "platform_supported": self.platform_supported,
            "worker_path": self.worker_path,
            "worker_exists": self.worker_exists,
            "metallib_exists": self.metallib_exists,
            "model_dir": self.model_dir,
            "model_exists": self.model_exists,
            "repo_commit": MLX_DINOV3_REPO_COMMIT,
            "conversion_version": MLX_DINOV3_CONVERSION_VERSION,
            "detail": self.detail,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _safe_slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value or "").strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:160] or "dinov3_model"


def mlx_dinov3_model_cache_root() -> Path:
    return Path(
        os.environ.get(
            "MLX_DINOV3_MODEL_ROOT",
            str(_repo_root() / "uploads" / "model_cache" / "mlx_dinov3"),
        )
    ).expanduser()


def mlx_dinov3_model_dir(model_name: str) -> Path:
    return mlx_dinov3_model_cache_root() / _safe_slug(model_name)


def mlx_dinov3_worker_path() -> Path:
    return Path(
        os.environ.get(
            "MLX_DINOV3_WORKER",
            str(_repo_root() / "tools" / "mlx_dinov3_worker" / ".build" / "release" / "mlx-dinov3-worker"),
        )
    ).expanduser()


def _platform_supported() -> bool:
    return platform.system().lower() == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def _model_dir_ready(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file() and (path / "model.safetensors").is_file()


def mlx_dinov3_status(model_name: str, requested: Optional[str] = None) -> MlxDinoV3BackendStatus:
    raw_requested = str(requested or os.environ.get("DINOV3_BACKEND") or "auto").strip().lower() or "auto"
    if raw_requested in {"pytorch", "pt"}:
        raw_requested = "torch"
    if raw_requested not in {"auto", "torch", "mlx"}:
        raw_requested = "auto"

    worker = mlx_dinov3_worker_path()
    model_dir = mlx_dinov3_model_dir(model_name)
    platform_ok = _platform_supported()
    worker_ok = worker.is_file() and os.access(worker, os.X_OK)
    metallib_ok = (worker.parent / "mlx.metallib").is_file() or (worker.parent / "default.metallib").is_file()
    model_ok = _model_dir_ready(model_dir)
    available = bool(platform_ok and worker_ok and metallib_ok and model_ok)
    if raw_requested == "torch":
        resolved = "torch"
        detail = "Torch DINOv3 requested."
    elif raw_requested == "mlx":
        resolved = "mlx" if available else "unavailable"
        detail = "MLX-DINOv3 requested." if available else "MLX-DINOv3 requested but worker/model/platform is not ready."
    else:
        resolved = "mlx" if available else "torch"
        detail = "Auto selected MLX-DINOv3." if available else "Auto falls back to Torch DINOv3 until MLX worker and converted model are ready."
    return MlxDinoV3BackendStatus(
        requested=raw_requested,
        resolved=resolved,
        available=available,
        platform_supported=platform_ok,
        worker_path=str(worker),
        worker_exists=worker_ok,
        metallib_exists=metallib_ok,
        model_dir=str(model_dir),
        model_exists=model_ok,
        detail=detail,
    )


def resolve_mlx_dinov3_backend(model_name: str, requested: Optional[str] = None) -> str:
    status = mlx_dinov3_status(model_name, requested=requested)
    if status.requested == "mlx" and not status.available:
        raise MlxDinoV3Unavailable(status.detail)
    return "mlx" if status.resolved == "mlx" else "torch"


class MlxDinoV3Worker:
    """Persistent JSONL client for the Swift MLX-DINOv3 worker."""

    def __init__(
        self,
        *,
        model_name: str,
        model_dir: Path,
        worker_path: Path,
        timeout_seconds: float = MLX_DINOV3_DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.model_name = str(model_name or "")
        self.model_dir = Path(model_dir)
        self.worker_path = Path(worker_path)
        self.timeout_seconds = float(timeout_seconds)
        self._process: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()

    @property
    def hidden_size(self) -> int:
        config_path = self.model_dir / "config.json"
        try:
            payload = json.loads(config_path.read_text())
            return int(payload.get("hidden_size") or 0)
        except Exception:
            return 0

    def close(self) -> None:
        proc = self._process
        self._process = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _stderr_tail(self) -> str:
        proc = self._process
        if proc is None or proc.stderr is None:
            return ""
        try:
            return proc.stderr.read()[-2000:]
        except Exception:
            return ""

    def _read_json_line(self, timeout_seconds: float) -> Dict[str, Any]:
        proc = self._process
        if proc is None or proc.stdout is None:
            raise MlxDinoV3WorkerError("mlx_dinov3_worker_not_started")
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        try:
            events = selector.select(max(0.1, timeout_seconds))
            if not events:
                raise MlxDinoV3WorkerError("mlx_dinov3_worker_timeout")
            line = proc.stdout.readline()
        finally:
            selector.close()
        if not line:
            stderr = self._stderr_tail()
            raise MlxDinoV3WorkerError(f"mlx_dinov3_worker_exited:{stderr}".rstrip(":"))
        try:
            payload = json.loads(line)
        except Exception as exc:
            raise MlxDinoV3WorkerError(f"mlx_dinov3_worker_bad_json:{line[:200]}") from exc
        if not isinstance(payload, dict):
            raise MlxDinoV3WorkerError("mlx_dinov3_worker_bad_payload")
        return payload

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        if not self.worker_path.is_file():
            raise MlxDinoV3Unavailable(f"MLX-DINOv3 worker not found: {self.worker_path}")
        if not _model_dir_ready(self.model_dir):
            raise MlxDinoV3Unavailable(f"MLX-DINOv3 converted model not found: {self.model_dir}")
        self._process = subprocess.Popen(
            [str(self.worker_path), "--model-dir", str(self.model_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self._read_json_line(timeout_seconds=60.0)
        if not ready.get("ok"):
            raise MlxDinoV3WorkerError(str(ready.get("error") or "mlx_dinov3_worker_start_failed"))

    def encode_image_paths(
        self,
        image_paths: Sequence[str],
        *,
        include_patch_tokens: bool = True,
        include_last_hidden_state: bool = False,
        output_path: Optional[Path] = None,
    ) -> Dict[str, np.ndarray]:
        if not image_paths:
            raise ValueError("image_paths must not be empty")
        try:
            from safetensors.numpy import load_file
        except Exception as exc:  # noqa: BLE001
            raise MlxDinoV3Unavailable("safetensors.numpy is required for MLX-DINOv3 output loading") from exc

        with self._lock:
            self._ensure_started()
            owns_temp = output_path is None
            temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
            try:
                if output_path is None:
                    root = Path(os.environ.get("MLX_DINOV3_OUTPUT_ROOT", tempfile.gettempdir())).expanduser()
                    root.mkdir(parents=True, exist_ok=True)
                    temp_dir = tempfile.TemporaryDirectory(prefix="mlx_dinov3_", dir=str(root))
                    output_path = Path(temp_dir.name) / "tokens.safetensors"
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                request_id = uuid.uuid4().hex
                request = {
                    "id": request_id,
                    "image_paths": [str(path) for path in image_paths],
                    "output_path": str(output_path),
                    "include_patch_tokens": bool(include_patch_tokens),
                    "include_last_hidden_state": bool(include_last_hidden_state),
                }
                assert self._process is not None and self._process.stdin is not None
                self._process.stdin.write(json.dumps(request) + "\n")
                self._process.stdin.flush()
                response = self._read_json_line(timeout_seconds=self.timeout_seconds)
                if response.get("id") != request_id:
                    raise MlxDinoV3WorkerError("mlx_dinov3_worker_response_id_mismatch")
                if not response.get("ok"):
                    raise MlxDinoV3WorkerError(str(response.get("error") or "mlx_dinov3_worker_failed"))
                arrays = load_file(str(output_path))
                return {str(key): np.asarray(value, dtype=np.float32) for key, value in arrays.items()}
            finally:
                if owns_temp and temp_dir is not None:
                    temp_dir.cleanup()


_WORKERS: Dict[Tuple[str, str, str], MlxDinoV3Worker] = {}
_WORKERS_LOCK = threading.Lock()


def get_mlx_dinov3_worker(model_name: str) -> MlxDinoV3Worker:
    status = mlx_dinov3_status(model_name, requested="mlx")
    if not status.available:
        raise MlxDinoV3Unavailable(status.detail)
    key = (str(model_name), status.model_dir, status.worker_path)
    with _WORKERS_LOCK:
        worker = _WORKERS.get(key)
        if worker is None:
            worker = MlxDinoV3Worker(
                model_name=model_name,
                model_dir=Path(status.model_dir),
                worker_path=Path(status.worker_path),
            )
            _WORKERS[key] = worker
        return worker


def is_mlx_dinov3_encoder(value: Any) -> bool:
    return isinstance(value, MlxDinoV3Worker)


def stop_mlx_dinov3_workers() -> None:
    with _WORKERS_LOCK:
        workers = list(_WORKERS.values())
        _WORKERS.clear()
    for worker in workers:
        try:
            worker.close()
        except BaseException:
            pass


atexit.register(stop_mlx_dinov3_workers)
