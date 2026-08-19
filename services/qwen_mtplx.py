"""Managed local MTPLX sidecar for speculative MLX Qwen inference."""

from __future__ import annotations

import atexit
import base64
import ipaddress
import io
import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


_ROOT_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_EXECUTABLE = _ROOT_DIR / ".venv-mtplx" / "bin" / "mtplx"
_DEFAULT_CACHE_DIR = _ROOT_DIR / ".cache" / "tator" / "mtplx"


def mtplx_model_path(model_id: str) -> Path:
    configured = str(os.environ.get("TATOR_MTPLX_MODEL_ROOT") or "").strip()
    root = Path(configured).expanduser() if configured else Path.home() / ".mtplx" / "models"
    return root / str(model_id or "").strip().replace("/", "--")


def mtplx_model_is_local(model_id: str) -> bool:
    path = mtplx_model_path(model_id)
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    weights = list(path.glob("*.safetensors"))
    return bool(weights) and not any(path.glob("*.incomplete"))


@dataclass(frozen=True)
class MtplxRuntimeHandle:
    """Opaque marker carried by Tator's existing Qwen runtime container."""

    model_id: str
    base_url: str
    owned: bool


def is_mtplx_runtime_handle(value: Any) -> bool:
    return isinstance(value, MtplxRuntimeHandle)


class MtplxSidecarManager:
    """Own one localhost MTPLX server without touching Tator's shared MLX env."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: Optional[subprocess.Popen[Any]] = None
        self._model_id: Optional[str] = None
        self._owned = False
        self._log_handle: Optional[Any] = None

    @property
    def executable(self) -> Path:
        configured = str(os.environ.get("TATOR_MTPLX_EXECUTABLE") or "").strip()
        return Path(configured).expanduser() if configured else _DEFAULT_EXECUTABLE

    @property
    def host(self) -> str:
        configured = str(
            os.environ.get("TATOR_MTPLX_HOST") or "127.0.0.1"
        ).strip()
        if configured.lower() == "localhost":
            return "127.0.0.1"
        try:
            address = ipaddress.ip_address(configured)
        except ValueError as exc:
            raise RuntimeError("mtplx_host_must_be_loopback") from exc
        if not address.is_loopback:
            # The managed sidecar deliberately runs without authentication.
            # It must never expose prompts or images beyond this machine.
            raise RuntimeError("mtplx_host_must_be_loopback")
        return configured

    @property
    def port(self) -> int:
        configured = str(os.environ.get("TATOR_MTPLX_PORT") or "").strip()
        if not configured:
            return 18081
        try:
            port = int(configured)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("mtplx_port_invalid") from exc
        if not 1 <= port <= 65535:
            raise RuntimeError("mtplx_port_invalid")
        return port

    @property
    def base_url(self) -> str:
        host = self.host
        authority = f"[{host}]" if ":" in host else host
        return f"http://{authority}:{self.port}"

    @property
    def log_path(self) -> Path:
        configured = str(os.environ.get("TATOR_MTPLX_LOG") or "").strip()
        return Path(configured).expanduser() if configured else _DEFAULT_CACHE_DIR / "sidecar.log"

    def runtime_available(self) -> bool:
        return self.executable.is_file() and os.access(self.executable, os.X_OK)

    def _request_json(
        self,
        path: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        timeout: float = 3.0,
    ) -> Dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        method = "GET"
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
            method = "POST"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
        parsed = json.loads(raw.decode("utf-8")) if raw else {}
        return parsed if isinstance(parsed, dict) else {}

    def _healthy(self) -> bool:
        try:
            self._request_json("/health", timeout=1.0)
            return True
        except Exception:
            return False

    def _served_model_ids(self) -> List[str]:
        try:
            payload = self._request_json("/v1/models", timeout=2.0)
        except Exception:
            return []
        result: List[str] = []
        for item in payload.get("data") or []:
            if isinstance(item, dict) and str(item.get("id") or "").strip():
                result.append(str(item["id"]).strip())
        return result

    def _tail_log(self, max_bytes: int = 12000) -> str:
        try:
            with self.log_path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - max_bytes), os.SEEK_SET)
                return handle.read().decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    def ensure(
        self,
        model_id: str,
        *,
        cancel_check: Optional[Callable[[], None]] = None,
        wait_callback: Optional[Callable[[str], None]] = None,
    ) -> MtplxRuntimeHandle:
        requested = str(model_id or "").strip()
        if not requested:
            raise RuntimeError("mtplx_model_id_missing")
        with self._lock:
            if self._process is not None and self._process.poll() is not None:
                self._clear_process_state()
            if self._healthy():
                served = self._served_model_ids()
                if self._model_id == requested or requested in served:
                    self._model_id = requested
                    return MtplxRuntimeHandle(requested, self.base_url, self._owned)
                if not self._owned:
                    detail = ", ".join(served) if served else "unknown model"
                    raise RuntimeError(
                        f"mtplx_port_in_use:{self.host}:{self.port}:{detail}"
                    )
                self._stop_locked()
            elif self._process is not None:
                self._stop_locked()

            executable = self.executable
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise RuntimeError(
                    "mtplx_runtime_missing: run tools/setup_mtplx_runtime.sh"
                )

            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_path.open("ab", buffering=0)
            command = [
                str(executable),
                "serve",
                "--model",
                requested,
                "--download",
                "--profile",
                "turbo",
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--no-auth",
                "--yes",
                "--mtp",
                "--generation-mode",
                "mtp",
                "--reasoning",
                "off",
                "--no-stats-footer",
            ]
            self._process = subprocess.Popen(
                command,
                cwd=str(_ROOT_DIR),
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._model_id = requested
            self._owned = True
            try:
                timeout = max(
                    60.0,
                    float(os.environ.get("TATOR_MTPLX_STARTUP_TIMEOUT_SECONDS") or 3600),
                )
            except (TypeError, ValueError):
                timeout = 3600.0
            deadline = time.monotonic() + timeout
            next_update = 0.0
            while time.monotonic() < deadline:
                if cancel_check is not None:
                    try:
                        cancel_check()
                    except Exception:
                        self._stop_locked()
                        raise
                if self._process is None or self._process.poll() is not None:
                    status = self._process.poll() if self._process is not None else "unknown"
                    detail = self._tail_log()
                    self._clear_process_state()
                    raise RuntimeError(
                        f"mtplx_start_failed:exit={status}:{detail or 'no log output'}"
                    )
                if self._healthy():
                    return MtplxRuntimeHandle(requested, self.base_url, True)
                now = time.monotonic()
                if wait_callback is not None and now >= next_update:
                    wait_callback(
                        "Starting native MTPLX runtime; downloading weights on the first run"
                    )
                    next_update = now + 2.0
                time.sleep(0.5)
            detail = self._tail_log()
            self._stop_locked()
            raise RuntimeError(
                f"mtplx_start_timeout:{int(timeout)}s:{detail or 'no log output'}"
            )

    def _clear_process_state(self) -> None:
        self._process = None
        self._model_id = None
        self._owned = False
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None

    def _stop_locked(self) -> None:
        process = self._process
        owned = self._owned
        if process is not None and owned and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=12.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5.0)
                except Exception:
                    pass
            except ProcessLookupError:
                pass
        self._clear_process_state()

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    @staticmethod
    def _image_data_url(value: Any) -> str:
        if isinstance(value, str):
            if value.startswith("data:image/"):
                return value
            path = Path(value).expanduser()
            raw = path.read_bytes()
            suffix = path.suffix.lower()
            mime = "image/png" if suffix == ".png" else "image/jpeg"
        elif isinstance(value, (bytes, bytearray)):
            raw = bytes(value)
            mime = "image/png"
        elif hasattr(value, "save"):
            buffer = io.BytesIO()
            image = value.convert("RGB") if hasattr(value, "convert") else value
            image.save(buffer, format="JPEG", quality=95)
            raw = buffer.getvalue()
            mime = "image/jpeg"
        else:
            raise TypeError(f"unsupported_mtplx_image:{type(value).__name__}")
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    @classmethod
    def _normalize_messages(
        cls,
        messages: Sequence[Dict[str, Any]],
        *,
        assistant_prefix: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                parts: List[Dict[str, Any]] = []
                for item in content:
                    if not isinstance(item, dict):
                        parts.append({"type": "text", "text": str(item)})
                        continue
                    item_type = str(item.get("type") or "text")
                    if item_type in {"image", "input_image"}:
                        image_value = item.get("image") or item.get("image_url")
                        if isinstance(image_value, dict):
                            image_value = image_value.get("url")
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": cls._image_data_url(image_value)},
                            }
                        )
                    elif item_type == "image_url":
                        image_value = item.get("image_url")
                        if isinstance(image_value, dict):
                            image_value = image_value.get("url")
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": cls._image_data_url(image_value)},
                            }
                        )
                    else:
                        parts.append({"type": "text", "text": str(item.get("text") or "")})
                normalized.append({"role": str(message.get("role") or "user"), "content": parts})
            else:
                normalized.append(
                    {
                        "role": str(message.get("role") or "user"),
                        "content": str(content or ""),
                    }
                )
        if assistant_prefix:
            instruction = f"Begin the response exactly with this prefix: {assistant_prefix}"
            normalized.append({"role": "user", "content": instruction})
        return normalized

    @staticmethod
    def _content_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            result: List[str] = []
            for item in value:
                if isinstance(item, dict):
                    result.append(str(item.get("text") or item.get("content") or ""))
                else:
                    result.append(str(item))
            return "".join(result)
        return str(value or "")

    def chat(
        self,
        handle: MtplxRuntimeHandle,
        messages: Sequence[Dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int] = None,
        enable_thinking: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        assistant_prefix: Optional[str] = None,
        cancel_check: Optional[Callable[[], None]] = None,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        if not self._healthy():
            raise RuntimeError("mtplx_sidecar_unavailable")
        payload: Dict[str, Any] = {
            "model": handle.model_id,
            "messages": self._normalize_messages(
                messages,
                assistant_prefix=assistant_prefix,
            ),
            "max_tokens": max(1, int(max_tokens)),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": True,
            "enable_thinking": bool(enable_thinking),
        }
        if top_k is not None:
            payload["top_k"] = max(0, int(top_k))
        if tools is not None:
            payload["tools"] = tools
        request_id = uuid.uuid4().hex
        payload["metadata"] = {
            "cache_mode": "bypass",
            "mtplx_request_id": request_id,
            "client": "tator",
        }
        request = urllib.request.Request(
            f"{handle.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
                "X-MTPLX-Cache-Mode": "bypass",
                "X-MTPLX-Client": "tator",
                "X-MTPLX-Request-ID": request_id,
            },
            method="POST",
        )
        try:
            timeout = max(
                30.0,
                float(os.environ.get("TATOR_MTPLX_REQUEST_TIMEOUT_SECONDS") or 900),
            )
        except (TypeError, ValueError):
            timeout = 900.0
        pieces: List[str] = []
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                for raw_line in response:
                    if cancel_check is not None:
                        cancel_check()
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0] if isinstance(choices[0], dict) else {}
                    delta = choice.get("delta") if isinstance(choice, dict) else {}
                    value = delta.get("content") if isinstance(delta, dict) else None
                    if value is None and isinstance(choice, dict):
                        value = (choice.get("message") or {}).get("content")
                    piece = self._content_text(value)
                    if not piece:
                        continue
                    pieces.append(piece)
                    if token_callback is not None:
                        token_callback(piece)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"mtplx_http_{exc.code}:{detail}") from exc
        return "".join(pieces).strip()


mtplx_sidecar_manager = MtplxSidecarManager()
atexit.register(mtplx_sidecar_manager.stop)


__all__ = [
    "MtplxRuntimeHandle",
    "MtplxSidecarManager",
    "is_mtplx_runtime_handle",
    "mtplx_model_is_local",
    "mtplx_model_path",
    "mtplx_sidecar_manager",
]
