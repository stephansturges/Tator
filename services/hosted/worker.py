"""Private inference worker. Never import the application or its global models.

Only the supervisor supplies the adapter factory. The binary JSON protocol uses
a private stdout descriptor, so ordinary model logging cannot corrupt it.
"""
from __future__ import annotations

import ctypes
import importlib
import importlib.abc
import json
import os
import re
import signal
import struct
import subprocess
import sys
import threading
import time
from typing import Any

MAX_MESSAGE_BYTES = 64 * 1024 * 1024
_guardian = None


class _NoApplicationImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "localinferenceapi" or fullname == "app" or fullname.startswith("app."):
            raise ImportError("Hosted workers cannot import the API application")
        return None


def _read_exact(stream, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        chunk = stream.read(count - len(chunks))
        if not chunk:
            raise EOFError("Supervisor disconnected")
        chunks.extend(chunk)
    return bytes(chunks)


def read_message(stream) -> dict[str, Any]:
    size = struct.unpack("!I", _read_exact(stream, 4))[0]
    if not 0 < size <= MAX_MESSAGE_BYTES:
        raise ValueError("worker_message_size")
    value = json.loads(_read_exact(stream, size), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
    if not isinstance(value, dict):
        raise ValueError("worker_message_type")
    return value


def encode_message(value: dict[str, Any]) -> bytes:
    data = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if not 0 < len(data) <= MAX_MESSAGE_BYTES:
        raise ValueError("worker_message_size")
    return struct.pack("!I", len(data)) + data


def _watch_parent(parent_pid: int) -> None:
    global _guardian
    # Linux kills a worker even if a native model call never returns to Python.
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
    if os.getppid() != parent_pid:
        os._exit(71)
    # Linux PDEATHSIG kills this direct worker before a Python thread can
    # reliably stop grandchildren. A small independent guardian in our owned
    # group handles both worker crashes and control-process loss. It inherits
    # no protocol pipes, loads no models and never owns a different process group.
    _guardian = subprocess.Popen([sys.executable, "-m", "services.hosted.worker", "--guard-group",
        str(parent_pid), str(os.getpid()), str(os.getpgrp())], stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)

    def watch():
        while True:
            time.sleep(0.25)
            if os.getppid() != parent_pid:
                # Popen starts a new session: this group never contains the API.
                os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=watch, name="hosted-parent-watch", daemon=True).start()


def _guard_group(parent_pid: int, worker_pid: int, group_id: int) -> int:
    if worker_pid <= 1 or group_id != worker_pid or os.getpgrp() != group_id:
        return 72
    while os.getppid() == worker_pid:
        try:
            os.kill(parent_pid, 0)
        except ProcessLookupError:
            break
        except PermissionError:
            pass
        time.sleep(0.1)
    # Our own membership prevents this group ID being reused while we watch.
    # Detached descendants are deliberately outside this contract.
    os.killpg(group_id, signal.SIGKILL)
    return 0


def main() -> int:
    parent_pid = int(os.environ["TATOR_HOSTED_PARENT_PID"])
    _watch_parent(parent_pid)
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    sys.meta_path.insert(0, _NoApplicationImport())
    adapter = None
    try:
        message = read_message(sys.stdin.buffer)
        if message.get("command") != "initialize":
            raise ValueError("worker_initialization_required")
        factory_path = message.get("factory", "")
        if not isinstance(factory_path, str) or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", factory_path):
            raise ValueError("worker_factory_invalid")
        module, attribute = factory_path.split(":")
        factory = getattr(importlib.import_module(module), attribute)
        adapter = factory(config=message["config"], device=message["device"])
        if not callable(getattr(adapter, "execute", None)):
            raise TypeError("worker_adapter_execute_required")
        protocol.write(encode_message({"id": message["id"], "ok": True, "pid": os.getpid()}))
        while True:
            message = read_message(sys.stdin.buffer)
            if message.get("command") == "close":
                return 0
            if message.get("command") != "execute":
                raise ValueError("worker_command_invalid")
            try:
                result = adapter.execute(message["operation"], message["payload"], message["context"])
                response = encode_message({"id": message["id"], "ok": True, "result": result})
            except Exception as exc:
                # Details go to private diagnostics, never into another user's
                # response. No arbitrary exception message or payload is echoed.
                import traceback
                traceback.print_exc(file=sys.stderr)
                response = encode_message({"id": message["id"], "ok": False, "error_type": type(exc).__name__})
            protocol.write(response)
    except EOFError:
        return 0
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 70
    finally:
        if adapter is not None:
            close = getattr(adapter, "close", None)
            if callable(close):
                close()
        protocol.close()


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--guard-group":
        raise SystemExit(_guard_group(*(int(value) for value in sys.argv[2:])))
    raise SystemExit(main())
