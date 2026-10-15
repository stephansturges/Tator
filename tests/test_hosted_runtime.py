"""Real subprocess scheduling proofs; the adapter deliberately does no ML work."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from services.hosted.runtime import AdapterSpec, RequestContext, RuntimeRejected, RuntimeSupervisor


class FixtureAdapter:
    """Only a fixture for the process/queue boundary, never a production model."""
    def __init__(self, *, config, device):
        self.config, self.device, self.count = config, device, 0
        if config.get("fail_start"):
            raise RuntimeError("fixture start failure")
        if config.get("start_marker"):
            Path(config["start_marker"]).write_text(str(os.getpid()))
            while not Path(config["start_release"]).exists():
                time.sleep(0.01)

    def execute(self, operation, payload, context):
        self.count += 1
        if operation == "child_hold":
            script = "import os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
            subprocess.Popen([sys.executable, "-c", script, payload["child_started"]])
            operation = "hold"
        if operation == "crash":
            os._exit(23)
        if operation == "fail":
            raise RuntimeError("private exception must not be exposed")
        if operation == "null":
            return None
        if operation == "forbidden_import":
            __import__("localinferenceapi")
        if operation == "hold":
            Path(payload["started"]).write_text(str(os.getpid()))
            while not Path(payload["release"]).exists():
                time.sleep(0.01)
        # Ordinary and native stdout logging must not corrupt the framing.
        print("fixture log")
        os.write(1, b"native fixture log\n")
        return {"pid": os.getpid(), "count": self.count, "context": context,
                "payload": payload, "config": self.config, "device": self.device,
                "api_loaded": "localinferenceapi" in sys.modules}


def context(actor="alice", dataset="dataset-a", version="version-1", generation=1):
    return RequestContext(actor, dataset, "image-a", version, generation, "editor-" + actor)


def eventually(check, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    tests_path = str(Path(__file__).resolve().parent)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (tests_path, os.environ.get("PYTHONPATH")))))
    instances = []

    def make(**kwargs):
        kwargs.setdefault("authorize", lambda ctx, phase: True)
        kwargs.setdefault("startup_timeout", 5)
        kwargs.setdefault("request_timeout", 5)
        instance = RuntimeSupervisor(tmp_path / str(len(instances)), **kwargs)
        instances.append(instance)
        return instance

    yield make
    for instance in instances:
        instance.close()


def spec(**config):
    return AdapterSpec("test_hosted_runtime:FixtureAdapter", config)


def completed(runtime, request, actor="alice"):
    assert runtime.wait(request, actor_id=actor, timeout=8)["state"] == "completed"
    return runtime.result(request, actor_id=actor)["result"]


def test_private_workers_immutable_context_and_actor_bound_results(supervisor):
    runtime = supervisor()
    configuration = {"nested": [1, 2]}
    selected = AdapterSpec("test_hosted_runtime:FixtureAdapter", configuration)
    runtime.register_user("alice", selected)
    runtime.register_user("bob", selected)
    configuration["nested"].append(3)
    payload = {"points": [[1, 2]]}
    first = runtime.submit(context(), "echo", payload)
    payload["points"][0][0] = 999
    second = runtime.submit(context("bob"), "echo", {})
    left, right = completed(runtime, first), completed(runtime, second, "bob")
    assert left["pid"] != right["pid"] != os.getpid()
    assert left["payload"] == {"points": [[1, 2]]}
    assert left["config"] == {"nested": [1, 2]}
    assert left["context"]["editor_id"] == "editor-alice"
    assert not left["api_loaded"] and not right["api_loaded"]
    with pytest.raises(TypeError):
        selected.config["new"] = 3
    with pytest.raises(RuntimeRejected, match="runtime_request_not_found"):
        runtime.result(first, actor_id="bob")
    null = runtime.submit(context(), "null", {})
    assert completed(runtime, null) is None
    # Even an indirect import from a trusted adapter cannot start the monolith.
    forbidden = runtime.submit(context(), "forbidden_import", {})
    assert runtime.wait(forbidden, actor_id="alice")["error"] == "worker_operation_failed"


def test_dataset_pipeline_is_sequential_bounded_and_separate(supervisor, tmp_path):
    runtime = supervisor(queue_limit=1)
    runtime.register_dataset("dataset-a", spec())
    runtime.register_dataset("dataset-b", spec())
    release, started = tmp_path / "release", tmp_path / "started"
    first = runtime.submit(context(), "hold", {"started": str(started), "release": str(release)}, owner="dataset")
    eventually(started.exists)
    queued = runtime.submit(context("bob"), "echo", {}, owner="dataset")
    with pytest.raises(RuntimeRejected, match="runtime_queue_full"):
        runtime.submit(context(), "echo", {}, owner="dataset")
    other = runtime.submit(context(dataset="dataset-b"), "echo", {}, owner="dataset")
    independent = completed(runtime, other)
    assert runtime.request_status(queued, actor_id="bob")["state"] == "queued"
    release.touch()
    first_result, queued_result = completed(runtime, first), completed(runtime, queued, "bob")
    assert first_result["pid"] == queued_result["pid"] != independent["pid"]
    assert queued_result["count"] == 2


def test_heavy_window_drains_restores_and_rechecks_queued_authority(supervisor, tmp_path):
    phases = []
    valid = {"alice": True, "bob": True}
    runtime = supervisor(authorize=lambda ctx, phase: phase.startswith("heavy") or valid[ctx.actor_id],
                         event_sink=lambda event: phases.append(event["phase"]))
    runtime.register_user("alice", spec())
    runtime.register_user("bob", spec())
    old_bob = completed(runtime, runtime.submit(context("bob"), "echo", {}), "bob")["pid"]
    release, started = tmp_path / "active-release", tmp_path / "active-started"
    active = runtime.submit(context(), "hold", {"started": str(started), "release": str(release)})
    eventually(started.exists)
    queued = runtime.submit(context(), "echo", {})
    heavy_release, heavy_started = tmp_path / "heavy-release", tmp_path / "heavy-started"
    heavy = runtime.begin_heavy(context("admin"), spec(), "hold", {"started": str(heavy_started), "release": str(heavy_release)})
    with pytest.raises(RuntimeRejected, match="runtime_automation_paused"):
        runtime.submit(context("bob"), "echo", {})
    assert not heavy_started.exists()
    release.touch()
    completed(runtime, active)
    eventually(heavy_started.exists)
    assert runtime.request_status(queued, actor_id="alice")["state"] == "queued"
    assert all(row["pid"] is None for row in runtime.snapshot()["owners"])
    valid["alice"] = False
    heavy_release.touch()
    completed(runtime, heavy, "admin")
    eventually(lambda: runtime.snapshot()["phase"] == "available")
    assert runtime.wait(queued, actor_id="alice")["state"] == "stale"
    new_bob = completed(runtime, runtime.submit(context("bob"), "echo", {}), "bob")["pid"]
    assert old_bob != new_bob
    eventually(lambda: "restoring" in phases and "available" in phases)
    for phase in ("pausing", "draining", "unloading", "heavy", "restoring"):
        assert phase in phases


def test_cancel_crash_idle_unload_and_config_change_recover(supervisor, tmp_path):
    runtime = supervisor(idle_seconds=0.2)
    runtime.register_user("alice", spec())
    release, started = tmp_path / "release", tmp_path / "started"
    active = runtime.submit(context(), "hold", {"started": str(started), "release": str(release)})
    eventually(started.exists)
    old_pid = int(started.read_text())
    runtime.cancel(active, actor_id="alice")
    assert runtime.wait(active, actor_id="alice")["state"] == "cancelled"
    new_pid = completed(runtime, runtime.submit(context(), "echo", {}))["pid"]
    assert old_pid != new_pid
    crash = runtime.submit(context(), "crash", {})
    assert runtime.wait(crash, actor_id="alice")["state"] == "failed"
    assert completed(runtime, runtime.submit(context(), "echo", {}))["pid"] != new_pid
    eventually(lambda: runtime.snapshot()["owners"][0]["pid"] is None)
    held_release, held_started = tmp_path / "held-release", tmp_path / "held-started"
    held = runtime.submit(context(), "hold", {"started": str(held_started), "release": str(held_release)})
    eventually(held_started.exists)
    queued = runtime.submit(context(), "echo", {})
    runtime.register_user("alice", spec(revision=2))
    assert runtime.wait(queued, actor_id="alice")["state"] == "stale"
    held_release.touch()
    assert runtime.wait(held, actor_id="alice")["state"] == "stale"
    assert completed(runtime, runtime.submit(context(), "echo", {}))["config"] == {"revision": 2}


@pytest.mark.parametrize("heavy_operation", ["fail", "hold"])
def test_heavy_failure_or_cancel_restores_after_forced_drain(supervisor, tmp_path, heavy_operation):
    runtime = supervisor()
    runtime.register_user("alice", spec())
    started = tmp_path / "started"
    active = runtime.submit(context(), "hold", {"started": str(started), "release": str(tmp_path / "never")})
    eventually(started.exists)
    heavy_started = tmp_path / "heavy-started"
    heavy = runtime.begin_heavy(context("admin"), spec(), heavy_operation,
                                {"started": str(heavy_started), "release": str(tmp_path / "never-heavy")}, drain_timeout=0.02)
    assert runtime.wait(active, actor_id="alice")["state"] == "cancelled"
    if heavy_operation == "hold":
        eventually(heavy_started.exists)
        runtime.cancel(heavy, actor_id="admin")
    assert runtime.wait(heavy, actor_id="admin")["state"] == ("cancelled" if heavy_operation == "hold" else "failed")
    eventually(lambda: runtime.snapshot()["phase"] == "available")
    assert completed(runtime, runtime.submit(context(), "echo", {}))["pid"] != int(started.read_text())


def test_authority_changes_during_execution_and_result_retrieval(supervisor, tmp_path):
    authorized = True
    runtime = supervisor(authorize=lambda ctx, phase: authorized)
    runtime.register_user("alice", spec())
    first = runtime.submit(context(), "echo", {})
    original = completed(runtime, first)
    release, started = tmp_path / "release", tmp_path / "started"
    active = runtime.submit(context(), "hold", {"started": str(started), "release": str(release)})
    eventually(started.exists)
    authorized = False
    release.touch()
    assert runtime.wait(active, actor_id="alice")["state"] == "stale"
    with pytest.raises(RuntimeRejected, match="runtime_authority_changed"):
        runtime.result(first, actor_id="alice")
    authorized = True
    assert completed(runtime, runtime.submit(context(), "echo", {}))["pid"] == original["pid"]


def test_heavy_does_not_reload_historical_idle_workers(supervisor):
    runtime = supervisor(idle_seconds=0.1)
    runtime.register_user("alice", spec())
    completed(runtime, runtime.submit(context(), "echo", {}))
    eventually(lambda: runtime.snapshot()["owners"][0]["pid"] is None)
    heavy = runtime.begin_heavy(context("admin"), spec(), "echo", {})
    completed(runtime, heavy, "admin")
    eventually(lambda: runtime.snapshot()["phase"] == "available")
    owner = runtime.snapshot()["owners"][0]
    assert owner["state"] == "unloaded" and owner["pid"] is None


@pytest.mark.parametrize("descendant", [False, True])
def test_restart_marks_interrupted_without_replaying_payload(supervisor, tmp_path, descendant):
    # A real disposable control process dies with a real worker mid-request.
    state, started = tmp_path / "crash-state", tmp_path / "started"
    script = """
import os, sys, time
from pathlib import Path
from services.hosted.runtime import RuntimeSupervisor
from test_hosted_runtime import spec, context
runtime = RuntimeSupervisor(sys.argv[1], authorize=lambda ctx, phase: True, startup_timeout=5)
runtime.register_user('alice', spec(secret='not-for-journal'))
request = runtime.submit(context(), sys.argv[4], {'started': sys.argv[2], 'release': sys.argv[3], 'child_started': sys.argv[5]})
while not Path(sys.argv[2]).exists(): time.sleep(.01)
if sys.argv[4]=='child_hold':
    while not Path(sys.argv[5]).exists(): time.sleep(.01)
print(request, flush=True)
os._exit(0)
"""
    child_started = tmp_path / "child-started"
    controller = subprocess.run([sys.executable, "-c", script, str(state), str(started), str(tmp_path / "never"), "child_hold" if descendant else "hold", str(child_started)],
                                capture_output=True, text=True, timeout=10)
    assert controller.returncode == 0, controller.stderr
    request = controller.stdout.strip()
    worker_pid = int(started.read_text())
    def worker_stopped(pid=worker_pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        # Linux may briefly retain an adopted zombie; it cannot execute work.
        status = Path(f"/proc/{pid}/stat")
        return status.exists() and status.read_text().split()[2] == "Z"
    eventually(worker_stopped)
    if descendant:
        eventually(lambda: worker_stopped(int(child_started.read_text())))
    with RuntimeSupervisor(state, authorize=lambda ctx, phase: True) as recovered:
        status = recovered.request_status(request, actor_id="alice")
        assert status["state"] == "interrupted" and status["error"] == "control_restarted"
        assert not status["result_available"]
        journal = (state / "runtime-state.json").read_text()
        assert "not-for-journal" not in journal and str(started) not in journal
        with pytest.raises(RuntimeRejected, match="runtime_supervisor_already_running"):
            RuntimeSupervisor(state, authorize=lambda ctx, phase: True)


def test_journal_failure_closes_automation_admission(supervisor, monkeypatch):
    runtime = supervisor()
    runtime.register_user("alice", spec())
    completed(runtime, runtime.submit(context(), "echo", {}))
    def full_disk():
        raise OSError("fixture disk full")
    monkeypatch.setattr(runtime, "_write_journal_locked", full_disk)
    with pytest.raises(RuntimeRejected, match="runtime_state_unavailable"):
        runtime.submit(context(), "echo", {})
    assert runtime.snapshot()["phase"] == "failed"
    with pytest.raises(RuntimeRejected, match="runtime_state_unavailable"):
        runtime.submit(context(), "echo", {})


def test_cancel_heavy_before_drain_preserves_active_runtime(supervisor, tmp_path):
    runtime = supervisor()
    runtime.register_user("alice", spec())
    release, started = tmp_path / "release", tmp_path / "started"
    active = runtime.submit(context(), "hold", {"started": str(started), "release": str(release)})
    eventually(started.exists)
    heavy = runtime.begin_heavy(context("admin"), spec(), "echo", {}, drain_timeout=4)
    runtime.cancel(heavy, actor_id="admin")
    assert runtime.wait(heavy, actor_id="admin")["state"] == "cancelled"
    eventually(lambda: runtime.snapshot()["phase"] == "available")
    release.touch()
    assert completed(runtime, active)["pid"] == int(started.read_text())


def test_cancellation_during_factory_loading_and_request_timeout(supervisor, tmp_path):
    runtime = supervisor()
    started = tmp_path / "factory-started"
    runtime.register_user("alice", spec(start_marker=str(started), start_release=str(tmp_path / "never")))
    loading = runtime.submit(context(), "echo", {})
    eventually(started.exists)
    runtime.cancel(loading, actor_id="alice")
    assert runtime.wait(loading, actor_id="alice")["state"] == "cancelled"
    runtime.register_user("alice", spec())
    timed = runtime.submit(context(), "hold", {"started": str(tmp_path / "execution"), "release": str(tmp_path / "never")}, timeout=0.05)
    assert runtime.wait(timed, actor_id="alice")["error"] == "worker_timeout"
    completed(runtime, runtime.submit(context(), "echo", {}))


@pytest.mark.parametrize("heavy", [False, True])
def test_revocation_during_weight_loading_prevents_execution(supervisor, tmp_path, heavy):
    allowed = True
    runtime = supervisor(authorize=lambda *_: allowed)
    started, release, executed = (tmp_path / name for name in ("loading", "loaded", "executed"))
    selected = spec(start_marker=str(started), start_release=str(release))
    payload = {"started": str(executed), "release": str(release)}
    if heavy:
        request = runtime.begin_heavy(context(), selected, "hold", payload)
    else:
        runtime.register_user("alice", selected)
        request = runtime.submit(context(), "hold", payload)
    eventually(started.exists)
    allowed = False
    release.touch()
    assert runtime.wait(request, actor_id="alice")["error"] == "runtime_authority_changed"
    assert not executed.exists()


def test_factory_failure_and_heavy_reload_failure_are_recoverable(supervisor, tmp_path):
    runtime = supervisor()
    runtime.register_user("alice", spec(fail_start=True))
    first = runtime.submit(context(), "echo", {})
    assert runtime.wait(first, actor_id="alice")["state"] == "failed"
    runtime.register_user("alice", spec())
    completed(runtime, runtime.submit(context(), "echo", {}))
    release, started = tmp_path / "release", tmp_path / "started"
    heavy = runtime.begin_heavy(context("admin"), spec(), "hold", {"started": str(started), "release": str(release)})
    eventually(started.exists)
    runtime.register_user("alice", spec(fail_start=True))
    release.touch()
    completed(runtime, heavy, "admin")
    eventually(lambda: runtime.snapshot()["phase"] == "available")
    assert runtime.snapshot()["owners"][0]["state"] == "failed"
    runtime.register_user("alice", spec())
    completed(runtime, runtime.submit(context(), "echo", {}))


def test_memory_bounds_and_old_configuration_result_rejection(supervisor):
    runtime = supervisor(owner_limit=1, result_bytes_limit=100, payload_bytes_limit=100)
    runtime.register_user("alice", spec())
    with pytest.raises(RuntimeRejected, match="runtime_owner_limit"):
        runtime.register_user("bob", spec())
    with pytest.raises(RuntimeRejected, match="runtime_payload_budget_exceeded"):
        runtime.submit(context(), "echo", {"large": "x" * 100})
    oversized = runtime.submit(context(), "echo", {})
    assert runtime.wait(oversized, actor_id="alice")["error"] == "runtime_result_budget_exceeded"
    old = runtime.submit(context(), "null", {})
    completed(runtime, old)
    runtime.register_user("alice", spec(revision=2))
    with pytest.raises(RuntimeRejected, match="runtime_configuration_changed"):
        runtime.result(old, actor_id="alice")
