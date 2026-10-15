"""Owner-scoped child processes and recoverable automation scheduling.

This module never loads a model or changes the control process's CUDA state.
Authentication and final annotation compare-and-swap remain the caller's job.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import struct
import subprocess
import sys
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping
import uuid

from services.hosted.worker import MAX_MESSAGE_BYTES, encode_message

TERMINAL = frozenset({"completed", "failed", "cancelled", "stale", "interrupted"})
_MISSING = object()


class RuntimeRejected(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _json(value) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _freeze(value):
    value = json.loads(_json(value))
    def immutable(item):
        if isinstance(item, dict):
            return MappingProxyType({key: immutable(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(immutable(child) for child in item)
        return item
    return immutable(value)


@dataclass(frozen=True)
class RequestContext:
    actor_id: str
    dataset_id: str
    image_id: str
    annotation_version: str
    ownership_generation: int
    editor_id: str
    workspace_id: str = ""
    model_config: Mapping[str, Any] = field(default_factory=dict)
    session_id: str = ""
    instance_id: str = ""
    actor_role: str = ""
    operation: str = ""

    def __post_init__(self):
        for name in ("actor_id", "dataset_id", "image_id", "annotation_version", "editor_id", "workspace_id", "session_id", "instance_id", "actor_role", "operation"):
            if not isinstance(getattr(self, name), str):
                raise ValueError("runtime_context_invalid")
        if not self.actor_id or not self.dataset_id:
            raise ValueError("runtime_context_identity_required")
        if type(self.ownership_generation) is not int or self.ownership_generation < 0:
            raise ValueError("runtime_context_generation_invalid")
        if not isinstance(self.model_config, Mapping):
            raise ValueError("runtime_model_config_invalid")
        object.__setattr__(self, "model_config", _freeze(self.model_config))

    def as_dict(self):
        return {name: _plain(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class AdapterSpec:
    factory: str
    config: Mapping[str, Any] = field(default_factory=dict)
    device: str = "cpu"

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", self.factory):
            raise ValueError("runtime_factory_invalid")
        if self.factory.split(":")[0].split(".")[0] in {"app", "localinferenceapi"}:
            raise ValueError("runtime_application_factory_forbidden")
        if not isinstance(self.config, Mapping) or not isinstance(self.device, str):
            raise ValueError("runtime_adapter_invalid")
        object.__setattr__(self, "config", _freeze(self.config))

    @property
    def identity(self):
        return hashlib.sha256(_json({"factory": self.factory, "config": self.config, "device": self.device}).encode()).hexdigest()


class _ProcessWorker:
    def __init__(self, spec: AdapterSpec, *, startup_timeout: float):
        self.spec = spec
        self.startup_timeout = startup_timeout
        self.stderr = deque(maxlen=64)
        env = dict(os.environ)
        env["TATOR_HOSTED_PARENT_PID"] = str(os.getpid())
        package_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (package_root, env.get("PYTHONPATH"))))
        self.process = subprocess.Popen(
            [sys.executable, "-m", "services.hosted.worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, start_new_session=True, bufsize=0,
        )
        self._reader = threading.Thread(target=self._read_stderr, daemon=True, name="hosted-worker-diagnostics")
        self._reader.start()

    def _read_stderr(self):
        try:
            while chunk := self.process.stderr.read(1024):
                self.stderr.append(chunk)
        except (OSError, ValueError):
            pass

    def _exchange(self, message, timeout):
        frame = encode_message(message)
        deadline = time.monotonic() + timeout
        # Nonblocking writes also bound a worker stuck before reading its input.
        fd = self.process.stdin.fileno()
        os.set_blocking(fd, False)
        view = memoryview(frame)
        while view:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
                raise RuntimeRejected("worker_timeout")
            try:
                count = os.write(fd, view)
            except BlockingIOError:
                continue
            if count <= 0:
                raise RuntimeRejected("worker_disconnected")
            view = view[count:]

        def read(count):
            output = bytearray()
            while len(output) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                    raise RuntimeRejected("worker_timeout")
                chunk = os.read(self.process.stdout.fileno(), count - len(output))
                if not chunk:
                    raise RuntimeRejected("worker_disconnected")
                output.extend(chunk)
            return bytes(output)

        size = struct.unpack("!I", read(4))[0]
        if not 0 < size <= MAX_MESSAGE_BYTES:
            raise RuntimeRejected("worker_protocol_invalid")
        response = json.loads(read(size))
        if not isinstance(response, dict) or response.get("id") != message["id"]:
            raise RuntimeRejected("worker_protocol_invalid")
        if response.get("ok") is not True:
            raise RuntimeRejected("worker_operation_failed")
        return response

    def initialize(self):
        self._exchange({"id": uuid.uuid4().hex, "command": "initialize", "factory": self.spec.factory,
                        "config": _plain(self.spec.config), "device": self.spec.device}, self.startup_timeout)

    def execute(self, request_id, operation, payload, context, timeout):
        return self._exchange({"id": request_id, "command": "execute", "operation": operation,
                               "payload": payload, "context": context}, timeout)["result"]

    def terminate(self):
        # This Popen created its own process group. Never signal the API group.
        self.kill()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            raise RuntimeRejected("worker_termination_failed")

    def kill(self):
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                # Darwin can return EPERM for a process group already exiting
                # after another cancellation path sent SIGKILL. Do not suppress
                # a genuine inability to terminate a still-running worker.
                try:
                    self.process.wait(timeout=0.2)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeRejected("worker_termination_failed") from exc

    def close(self):
        self.terminate()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        self._reader.join(timeout=0.5)


@dataclass
class _Owner:
    key: str
    kind: str
    owner_id: str
    spec: AdapterSpec
    queue: deque = field(default_factory=deque)
    worker: _ProcessWorker | None = None
    active_request: str | None = None
    state: str = "unloaded"
    wanted: bool = True
    used: bool = False
    last_actor: str | None = None
    last_used: float = field(default_factory=time.monotonic)
    thread: threading.Thread | None = None
    error: str | None = None


class RuntimeSupervisor:
    def __init__(self, state_dir: Path | str, *, authorize: Callable[[RequestContext, str], Any],
                 event_sink: Callable[[dict], None] | None = None, idle_seconds: float = 900,
                 queue_limit: int = 32, pending_limit: int = 128, history_limit: int = 1024,
                 payload_bytes_limit: int = 128 * 1024 * 1024, result_bytes_limit: int = 64 * 1024 * 1024,
                 startup_timeout: float = 180, request_timeout: float = 300, owner_limit: int = 128):
        limits = (idle_seconds, queue_limit, pending_limit, history_limit, payload_bytes_limit,
                  result_bytes_limit, startup_timeout, request_timeout, owner_limit)
        if any(not math.isfinite(value) or value <= 0 for value in limits):
            raise ValueError("runtime_limits_invalid")
        if any(type(value) is not int for value in (queue_limit, pending_limit, history_limit, payload_bytes_limit, result_bytes_limit, owner_limit)):
            raise ValueError("runtime_limits_invalid")
        self.root = Path(state_dir)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock_file = (self.root / "supervisor.lock").open("a+b")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_file.close()
            raise RuntimeRejected("runtime_supervisor_already_running")
        self.authorize, self.event_sink = authorize, event_sink
        self.idle_seconds, self.queue_limit, self.pending_limit = idle_seconds, queue_limit, pending_limit
        self.owner_limit = owner_limit
        self.history_limit, self.payload_bytes_limit, self.result_bytes_limit = history_limit, payload_bytes_limit, result_bytes_limit
        self.startup_timeout, self.request_timeout = startup_timeout, request_timeout
        self._condition = threading.Condition(threading.RLock())
        self._owners: dict[str, _Owner] = {}
        self._requests: dict[str, dict] = {}
        self._closed = False
        self._persistence_error = False
        self._payload_bytes = self._result_bytes = 0
        self._phase = "available"
        self._heavy_id = None
        self._heavy_worker = None
        self._heavy_thread = None
        self._events = deque(maxlen=256)
        self.instance_id = uuid.uuid4().hex
        try:
            self._recover()
            with self._condition:
                self._persist_locked()
        except BaseException:
            self._lock_file.close()
            raise
        self._event_thread = threading.Thread(target=self._event_loop, daemon=True, name="hosted-runtime-events")
        self._event_thread.start()

    def _recover(self):
        path = self.root / "runtime-state.json"
        if not path.exists():
            return
        previous = json.loads(path.read_text())
        if previous.get("schema") != "hosted-runtime-v1":
            raise RuntimeRejected("runtime_recovery_schema_invalid")
        for saved in previous["requests"][-self.history_limit:]:
            record = dict(saved)
            record["context"] = RequestContext(**record["context"])
            record.update(payload=None, payload_bytes=0, result=None, result_bytes=0)
            if record["state"] not in TERMINAL:
                record.update(state="interrupted", error="control_restarted", finished_at=time.time())
            self._requests[record["id"]] = record

    def _persist_locked(self):
        if self._persistence_error:
            return
        try:
            self._write_journal_locked()
        except OSError as exc:
            # Fail automation closed. Manual annotation persistence belongs to
            # the independent control service, which this supervisor never stops.
            self._persistence_error = True
            self._phase = "failed"
            for record in self._requests.values():
                if record["state"] not in TERMINAL:
                    record["cancel_requested"] = True
                    if record["state"] == "queued":
                        self._payload_bytes -= record.get("payload_bytes", 0)
                        record.update(state="interrupted", error="runtime_state_unavailable", finished_at=time.time(), payload=None, payload_bytes=0)
            for owner in self._owners.values():
                owner.queue.clear()
                if owner.worker:
                    owner.worker.kill()
            if self._heavy_worker:
                self._heavy_worker.kill()
            self._event_locked("runtime_storage_failed")
            raise RuntimeRejected("runtime_state_unavailable") from exc

    def _write_journal_locked(self):
        records = []
        for record in self._requests.values():
            row = {key: value for key, value in record.items() if key not in {"context", "payload", "result", "payload_bytes", "result_bytes"}}
            # Runtime credentials and pixels never belong in recovery metadata.
            row["context"] = {**record["context"].as_dict(), "model_config": {}}
            records.append(row)
        data = _json({"schema": "hosted-runtime-v1", "instance_id": self.instance_id,
                      "phase": self._phase, "heavy_id": self._heavy_id, "requests": records}).encode()
        partial = self.root / ".runtime-state.tmp"
        fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, self.root / "runtime-state.json")
        directory = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _event_locked(self, kind, **fields):
        self._events.append({"type": kind, "instance_id": self.instance_id, "phase": self._phase, **fields})
        self._condition.notify_all()

    def _event_loop(self):
        while True:
            with self._condition:
                while not self._events and not self._closed:
                    self._condition.wait()
                if not self._events and self._closed:
                    return
                event = self._events.popleft()
            if self.event_sink:
                try:
                    self.event_sink(event)
                except Exception:
                    pass  # Status is authoritative; notification is best effort.

    def _authorized(self, context, phase):
        try:
            accepted = self.authorize(context, phase)
        except Exception as exc:
            raise RuntimeRejected("runtime_authority_changed") from exc
        if accepted is False:
            raise RuntimeRejected("runtime_authority_changed")

    def register_user(self, user_id: str, spec: AdapterSpec):
        self._register("user", user_id, spec)

    def register_dataset(self, dataset_id: str, spec: AdapterSpec):
        self._register("dataset", dataset_id, spec)

    def _register(self, kind, owner_id, spec):
        if not isinstance(owner_id, str) or not owner_id or not isinstance(spec, AdapterSpec):
            raise ValueError("runtime_owner_invalid")
        key = kind + ":" + owner_id
        with self._condition:
            if self._closed:
                raise RuntimeRejected("runtime_closed")
            if self._persistence_error:
                raise RuntimeRejected("runtime_state_unavailable")
            existing = self._owners.get(key)
            if existing:
                existing.wanted = True
                if existing.spec.identity != spec.identity:
                    existing.spec = spec
                    for request_id in list(existing.queue):
                        self._finish_locked(self._requests[request_id], "stale", error="runtime_configuration_changed")
                    existing.queue.clear()
                self._condition.notify_all()
                return
            if len(self._owners) >= self.owner_limit:
                raise RuntimeRejected("runtime_owner_limit")
            owner = _Owner(key, kind, owner_id, spec)
            self._owners[key] = owner
            owner.thread = threading.Thread(target=self._owner_loop, args=(owner,), daemon=True, name="hosted-owner-" + kind)
            owner.thread.start()

    def set_active(self, owner: str, owner_id: str, active: bool):
        with self._condition:
            entry = self._owners[owner + ":" + owner_id]
            entry.wanted = bool(active)
            if active:
                entry.last_used = time.monotonic()
            self._condition.notify_all()

    def _new_record_locked(self, context, operation, payload, key, identity, timeout, request_id=None):
        if not isinstance(operation, str) or not operation or len(operation) > 160:
            raise ValueError("runtime_operation_invalid")
        payload_text = _json(payload)
        size = len(payload_text.encode())
        if size > MAX_MESSAGE_BYTES - 65536 or self._payload_bytes + size > self.payload_bytes_limit:
            raise RuntimeRejected("runtime_payload_budget_exceeded")
        if sum(record["state"] not in TERMINAL for record in self._requests.values()) >= self.pending_limit:
            raise RuntimeRejected("runtime_queue_full")
        while len(self._requests) >= self.history_limit:
            oldest = next((rid for rid, row in self._requests.items() if row["state"] in TERMINAL), None)
            if oldest is None:
                raise RuntimeRejected("runtime_queue_full")
            self._result_bytes -= self._requests[oldest].get("result_bytes", 0)
            del self._requests[oldest]
        request_id = request_id or uuid.uuid4().hex
        if not isinstance(request_id, str) or not re.fullmatch(r"[a-f0-9]{32}", request_id) or request_id in self._requests:
            raise RuntimeRejected("runtime_request_id_invalid")
        record = {"id": request_id, "context": context, "owner": key, "runtime_identity": identity,
                  "operation": operation, "payload": payload_text, "payload_bytes": size,
                  "state": "queued", "error": None, "created_at": time.time(), "finished_at": None,
                  "cancel_requested": False, "timeout": timeout, "result": None, "result_bytes": 0}
        self._requests[request_id] = record
        self._payload_bytes += size
        return record

    def submit(self, context: RequestContext, operation: str, payload: Mapping[str, Any], *, owner: str = "user", timeout: float | None = None) -> str:
        if owner not in {"user", "dataset"} or not context.image_id or not context.annotation_version or not context.editor_id:
            raise ValueError("runtime_inference_context_required")
        context = replace(context, operation=operation)
        self._authorized(context, "admit")
        key = owner + ":" + (context.actor_id if owner == "user" else context.dataset_id)
        timeout = self.request_timeout if timeout is None else timeout
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("runtime_timeout_invalid")
        with self._condition:
            if self._closed:
                raise RuntimeRejected("runtime_closed")
            if self._persistence_error:
                raise RuntimeRejected("runtime_state_unavailable")
            if self._phase != "available":
                raise RuntimeRejected("runtime_automation_paused")
            entry = self._owners.get(key)
            if entry is None:
                raise RuntimeRejected("runtime_owner_unconfigured")
            if len(entry.queue) >= self.queue_limit:
                raise RuntimeRejected("runtime_queue_full")
            if context.model_config and _json(context.model_config) != _json(entry.spec.config):
                raise RuntimeRejected("runtime_configuration_changed")
            context = replace(context, model_config=entry.spec.config)
            record = self._new_record_locked(context, operation, payload, key, entry.spec.identity, timeout)
            entry.queue.append(record["id"])
            entry.wanted = True
            self._persist_locked()
            self._event_locked("request_queued", request_id=record["id"], actor_id=context.actor_id)
            return record["id"]

    def _finish_locked(self, record, state, *, result=_MISSING, error=None):
        if record["state"] in TERMINAL:
            return
        self._payload_bytes -= record.get("payload_bytes", 0)
        record.update(payload=None, payload_bytes=0, state=state, error=error, finished_at=time.time())
        if state == "completed" and result is not _MISSING:
            text = _json(result)
            size = len(text.encode())
            for previous in self._requests.values():
                if self._result_bytes + size <= self.result_bytes_limit:
                    break
                if previous.get("result_bytes"):
                    self._result_bytes -= previous["result_bytes"]
                    previous.update(result=None, result_bytes=0)
            if size <= self.result_bytes_limit:
                record.update(result=text, result_bytes=size)
                self._result_bytes += size
            else:
                record.update(state="failed", error="runtime_result_budget_exceeded")
        self._persist_locked()
        self._event_locked("request_finished", request_id=record["id"], actor_id=record["context"].actor_id, state=record["state"])

    def _owner_loop(self, owner):
        while True:
            dispose = record = None
            try:
                with self._condition:
                    if self._closed:
                        return
                    if self._phase == "available" and owner.queue:
                        # Interleave actors on a dataset's sequential pipeline,
                        # while preserving submission order for each actor.
                        request_id = next((rid for rid in owner.queue if self._requests[rid]["context"].actor_id != owner.last_actor), owner.queue[0])
                        owner.queue.remove(request_id)
                        record = self._requests[request_id]
                        owner.last_actor = record["context"].actor_id
                        owner.active_request = record["id"]
                        owner.state = "busy"
                        record["state"] = "running"
                        spec = owner.spec
                        self._persist_locked()
                    elif self._phase == "available" and owner.worker and time.monotonic() - owner.last_used >= self.idle_seconds:
                        dispose, owner.worker = owner.worker, None
                        owner.state = "unloaded"
                        self._event_locked("runtime_unloaded", owner=owner.key)
                    else:
                        self._condition.wait(timeout=min(0.1, self.idle_seconds))
                        continue
                if dispose:
                    dispose.close()
                    continue
                self._authorized(record["context"], "dispatch")
                if record["runtime_identity"] != spec.identity:
                    raise RuntimeRejected("runtime_configuration_changed")
                with self._condition:
                    if record["cancel_requested"] or self._closed:
                        raise RuntimeRejected("runtime_cancelled")
                    worker = owner.worker
                if worker and worker.spec.identity != spec.identity:
                    worker.close()
                    worker = None
                    with self._condition:
                        owner.worker = None
                if worker is None:
                    with self._condition:
                        if record["cancel_requested"] or self._closed or self._persistence_error:
                            raise RuntimeRejected("runtime_cancelled")
                        worker = _ProcessWorker(spec, startup_timeout=self.startup_timeout)
                        owner.worker = worker
                        owner.used = True
                        owner.state = "loading"
                    worker.initialize()
                # Loading weights can take minutes. A logout, handoff or model
                # change during initialization must prevent the inference, not
                # merely discard its result after unnecessary computation.
                self._authorized(record["context"], "dispatch")
                with self._condition:
                    if record["cancel_requested"] or self._closed:
                        raise RuntimeRejected("runtime_cancelled")
                    if owner.spec.identity != record["runtime_identity"]:
                        raise RuntimeRejected("runtime_configuration_changed")
                    owner.state = "busy"
                result = worker.execute(record["id"], record["operation"], json.loads(record["payload"]), record["context"].as_dict(), record["timeout"])
                self._authorized(record["context"], "result")
                with self._condition:
                    if record["cancel_requested"]:
                        self._finish_locked(record, "cancelled", error="runtime_cancelled")
                    elif owner.spec.identity != record["runtime_identity"]:
                        self._finish_locked(record, "stale", error="runtime_configuration_changed")
                    else:
                        self._finish_locked(record, "completed", result=result)
                        owner.error = None
            except Exception as exc:
                code = exc.code if isinstance(exc, RuntimeRejected) else "worker_failed"
                with self._condition:
                    # A stale actor/image does not damage the loaded model.
                    # In particular, one revoked annotator must not evict the
                    # dataset detector serving other people's queued requests.
                    preserve = code == "runtime_authority_changed" and owner.worker is not None and owner.worker.spec.identity == owner.spec.identity
                    worker = None if preserve else owner.worker
                    if not preserve:
                        owner.worker = None
                    if dispose:
                        worker = dispose
                if worker:
                    try:
                        worker.close()
                    except Exception:
                        with self._condition:
                            self._phase = "failed"
                            owner.worker = worker
                with self._condition:
                    if record is not None:
                        state = "cancelled" if record["cancel_requested"] else "stale" if code in {"runtime_authority_changed", "runtime_configuration_changed"} else "failed"
                        try:
                            self._finish_locked(record, state, error=code)
                        except RuntimeRejected:
                            pass  # A journal failure already closed admission.
                    owner.error = None if preserve else code
            finally:
                if record is not None:
                    with self._condition:
                        owner.active_request = None
                        owner.last_used = time.monotonic()
                        owner.state = "ready" if owner.worker else "unloaded"
                        self._condition.notify_all()

    def request_status(self, request_id: str, *, actor_id: str) -> dict:
        with self._condition:
            record = self._requests.get(request_id)
            if record is None or record["context"].actor_id != actor_id:
                raise RuntimeRejected("runtime_request_not_found")
            return {key: record[key] for key in ("id", "state", "error", "created_at", "finished_at", "runtime_identity")} | {"result_available": record["result"] is not None and not self._persistence_error, "runtime_phase": self._phase}

    def result(self, request_id: str, *, actor_id: str) -> dict:
        with self._condition:
            if self._persistence_error:
                raise RuntimeRejected("runtime_state_unavailable")
            self.request_status(request_id, actor_id=actor_id)
            record = self._requests[request_id]
            context = record["context"]
        self._authorized(context, "heavy_result" if record["owner"] == "heavy" else "result")
        with self._condition:
            if self._persistence_error:
                raise RuntimeRejected("runtime_state_unavailable")
            entry = self._owners.get(record["owner"])
            if entry and entry.spec.identity != record["runtime_identity"]:
                raise RuntimeRejected("runtime_configuration_changed")
            if record["state"] != "completed":
                raise RuntimeRejected("runtime_result_not_ready")
            if record["result"] is None:
                raise RuntimeRejected("runtime_result_not_retained")
            return {"request_id": request_id, "context": context.as_dict(), "runtime_identity": record["runtime_identity"], "result": json.loads(record["result"])}

    def wait(self, request_id: str, *, actor_id: str, timeout: float = 30) -> dict:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("runtime_timeout_invalid")
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                status = self.request_status(request_id, actor_id=actor_id)
                if status["state"] in TERMINAL or time.monotonic() >= deadline:
                    return status
                self._condition.wait(timeout=max(0, deadline - time.monotonic()))

    def cancel(self, request_id: str, *, actor_id: str):
        worker = None
        with self._condition:
            status = self.request_status(request_id, actor_id=actor_id)
            record = self._requests[request_id]
            if status["state"] in TERMINAL:
                return status
            record["cancel_requested"] = True
            if record["owner"] == "heavy":
                worker = self._heavy_worker
            else:
                owner = self._owners[record["owner"]]
                if request_id in owner.queue:
                    owner.queue.remove(request_id)
                    self._finish_locked(record, "cancelled", error="runtime_cancelled")
                else:
                    worker = owner.worker
            self._persist_locked()
            self._condition.notify_all()
        if worker:
            worker.terminate()
        return self.request_status(request_id, actor_id=actor_id)

    def snapshot(self) -> dict:
        """Administrative status. Callers must filter owner information for users."""
        with self._condition:
            return {"instance_id": self.instance_id, "phase": self._phase, "heavy_id": self._heavy_id,
                    "owners": [{"owner": item.key, "state": item.state, "queued": len(item.queue),
                                "active_request": item.active_request, "runtime_identity": item.spec.identity,
                                "pid": item.worker.process.pid if item.worker else None, "error": item.error}
                               for item in self._owners.values()]}

    def begin_heavy(self, context: RequestContext, spec: AdapterSpec, operation: str, payload: Mapping[str, Any], *, drain_timeout: float = 30, timeout: float | None = None, request_id: str | None = None) -> str:
        context = replace(context, operation=operation)
        self._authorized(context, "heavy")  # The integrating authority requires admin.
        if not math.isfinite(drain_timeout) or drain_timeout <= 0 or (timeout is not None and (not math.isfinite(timeout) or timeout <= 0)):
            raise ValueError("runtime_timeout_invalid")
        with self._condition:
            if self._closed:
                raise RuntimeRejected("runtime_closed")
            if self._persistence_error:
                raise RuntimeRejected("runtime_state_unavailable")
            if self._phase != "available":
                raise RuntimeRejected("runtime_heavy_window_busy")
            context = replace(context, model_config=spec.config)
            record = self._new_record_locked(context, operation, payload, "heavy", spec.identity, timeout or self.request_timeout, request_id=request_id)
            self._heavy_id = record["id"]
            self._phase = "pausing"
            self._persist_locked()
            self._event_locked("automation_paused", actor_id=context.actor_id)
            restore_owners = frozenset(owner.key for owner in self._owners.values() if owner.worker or owner.active_request)
            self._heavy_thread = threading.Thread(target=self._heavy_loop, args=(record, spec, drain_timeout, restore_owners), daemon=True, name="hosted-heavy-window")
            self._heavy_thread.start()
            return record["id"]

    def _phase_locked(self, phase):
        self._phase = phase
        self._persist_locked()
        self._event_locked("runtime_phase_changed")

    def _heavy_loop(self, record, spec, drain_timeout, restore_owners):
        worker = None
        unloaded = False
        recovery_failed = False
        try:
            with self._condition:
                self._phase_locked("draining")
                deadline = time.monotonic() + drain_timeout
                while any(owner.active_request for owner in self._owners.values()) and time.monotonic() < deadline and not self._closed and not record["cancel_requested"]:
                    self._condition.wait(timeout=min(0.1, max(0, deadline - time.monotonic())))
                if self._closed or record["cancel_requested"]:
                    raise RuntimeRejected("runtime_cancelled")
                active = [(owner, owner.worker) for owner in self._owners.values() if owner.active_request]
                for owner, _ in active:
                    self._requests[owner.active_request]["cancel_requested"] = True
            for _, busy in active:
                if busy:
                    busy.terminate()
            with self._condition:
                deadline = time.monotonic() + self.startup_timeout + 3
                while any(owner.active_request for owner in self._owners.values()):
                    if time.monotonic() >= deadline:
                        recovery_failed = True
                        raise RuntimeRejected("runtime_drain_failed")
                    self._condition.wait(timeout=0.1)
                self._phase_locked("unloading")
                workers = [owner.worker for owner in self._owners.values() if owner.worker]
                for owner in self._owners.values():
                    owner.state = "unloading" if owner.worker else "unloaded"
            for instance in workers:
                try:
                    instance.close()
                except Exception:
                    recovery_failed = True
                    raise
            with self._condition:
                for owner in self._owners.values():
                    owner.worker = None
                    owner.state = "unloaded"
                unloaded = True
            self._authorized(record["context"], "heavy")
            with self._condition:
                if self._closed or record["cancel_requested"]:
                    raise RuntimeRejected("runtime_cancelled")
                self._phase_locked("heavy")
                record["state"] = "running"
                worker = _ProcessWorker(spec, startup_timeout=self.startup_timeout)
                self._heavy_worker = worker
                self._persist_locked()
            worker.initialize()
            self._authorized(record["context"], "heavy")
            with self._condition:
                if self._closed or record["cancel_requested"]:
                    raise RuntimeRejected("runtime_cancelled")
            result = worker.execute(record["id"], record["operation"], json.loads(record["payload"]), record["context"].as_dict(), record["timeout"])
            self._authorized(record["context"], "heavy_result")
            with self._condition:
                self._finish_locked(record, "cancelled" if record["cancel_requested"] else "completed", result=None if record["cancel_requested"] else result)
        except Exception as exc:
            with self._condition:
                if record["state"] not in TERMINAL:
                    try:
                        self._finish_locked(record, "cancelled" if record["cancel_requested"] or self._closed else "failed", error=exc.code if isinstance(exc, RuntimeRejected) else "worker_failed")
                    except RuntimeRejected:
                        pass
        finally:
            if worker:
                try:
                    worker.close()
                except Exception:
                    recovery_failed = True
            with self._condition:
                self._heavy_worker = None
                try:
                    self._phase_locked("restoring")
                except RuntimeRejected:
                    pass
                restore = [owner for owner in self._owners.values() if unloaded and owner.key in restore_owners and owner.wanted and not self._closed and not self._persistence_error and not recovery_failed]
            for owner in restore:
                instance = None
                try:
                    with self._condition:
                        if self._closed or not owner.wanted:
                            continue
                        selected = owner.spec
                        owner.state = "loading"
                        instance = _ProcessWorker(selected, startup_timeout=self.startup_timeout)
                        owner.worker = instance
                    instance.initialize()
                    with self._condition:
                        if self._closed or selected.identity != owner.spec.identity:
                            raise RuntimeRejected("runtime_configuration_changed")
                        owner.state, owner.error = "ready", None
                        owner.last_used = time.monotonic()
                        self._event_locked("runtime_ready", owner=owner.key)
                except Exception as exc:
                    if instance:
                        try:
                            instance.close()
                        except Exception:
                            recovery_failed = True
                    with self._condition:
                        owner.worker = None
                        owner.state, owner.error = "failed", exc.code if isinstance(exc, RuntimeRejected) else "worker_failed"
                        self._event_locked("runtime_reload_failed", owner=owner.key)
            with self._condition:
                self._heavy_id = None
                phase = "closed" if self._closed else "failed" if recovery_failed or self._persistence_error else "available"
                try:
                    self._phase_locked(phase)
                except RuntimeRejected:
                    pass

    def close(self):
        with self._condition:
            if self._closed and self._lock_file.closed:
                return
            self._closed = True
            workers = [owner.worker for owner in self._owners.values() if owner.worker]
            if self._heavy_worker:
                workers.append(self._heavy_worker)
            for record in self._requests.values():
                if record["state"] not in TERMINAL:
                    record["cancel_requested"] = True
            self._condition.notify_all()
        for worker in workers:
            worker.terminate()
        for owner in list(self._owners.values()):
            owner.thread.join(timeout=self.startup_timeout + 3)
            if owner.worker:
                owner.worker.close()
        if self._heavy_thread:
            self._heavy_thread.join(timeout=self.startup_timeout + 3)
        if any(owner.thread.is_alive() for owner in self._owners.values()) or (self._heavy_thread and self._heavy_thread.is_alive()):
            # Keep the exclusive supervisor lock: a second manager must not
            # start while a callback or worker cleanup still owns old work.
            raise RuntimeRejected("runtime_shutdown_incomplete")
        with self._condition:
            for record in self._requests.values():
                if record["state"] not in TERMINAL:
                    self._finish_locked(record, "interrupted", error="supervisor_closed")
            self._phase = "closed"
            self._persist_locked()
        self._event_thread.join(timeout=1)
        self._lock_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
