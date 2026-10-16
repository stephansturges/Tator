"""Confirmed-snapshot caption runs, with durable per-attempt evidence.

The control process owns authority, retention and retries. One isolated heavy
worker owns Qwen for a run; it never opens the hosted database or changes labels.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from services.hosted.caption_case import CaptionSettings
from services.hosted.errors import DomainError
from services.hosted.runtime import AdapterSpec, RequestContext, RuntimeRejected, TERMINAL
from services.hosted.types import Actor, MutationContext
from services.hosted.paid_caption import PaidCaptionOptions
from services.hosted.instruction_review import bind_machine_validated_pairs


def _json(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read(path, limit=32 * 1024 * 1024):
    from services.hosted.artifacts import open_regular
    with open_regular(path) as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("caption_artifact_too_large")
    return json.loads(data, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def canonical_case(image, classes):
    names = {row["id"]: row["name"] for row in classes}
    hints, counts = [], Counter()
    for item in image["annotations"].get("objects", []):
        label = names[item["class_id"]]
        left, top, width, height = item["bbox"]
        hints.append({"label": label, "bbox": [left, top, left + width, top + height], "source_id": item["id"]})
        counts[label] += 1
    return {"case_id": image["id"], "name": image["id"], "stem": image["id"],
        "image_name": image["filename"], "label_count": len(hints), "class_counts": dict(counts), "label_hints": hints}


class PilotSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    min_cases: int = Field(default=20, ge=1, le=100000)
    max_duration_hours: float = Field(default=24, gt=0, le=8760)
    duration_safety_factor: float = Field(default=1.5, ge=1, le=10)
    deterministic_recovery_confidence: float = Field(default=0.95, ge=0, lt=1)
    require_prompt_budget: bool = True


class CaptionBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    snapshot_id: str
    operation_id: str = Field(min_length=8, max_length=128)
    caption_request: dict = Field(default_factory=dict)
    settings: CaptionSettings = Field(default_factory=CaptionSettings)
    sample_size: int = Field(default=20, ge=0, le=1000000)
    sample_seed: int = 0
    pilot: PilotSettings | None = Field(default_factory=PilotSettings)
    certified_pilot_id: str | None = None
    provider: Literal["local_qwen", "openai"] = "local_qwen"
    provider_options: PaidCaptionOptions | None = None


class CaptionProposalApply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_version: str
    generation: int = Field(ge=0)
    editor_id: str = Field(min_length=8, max_length=128)
    operation_id: str = Field(min_length=8, max_length=128)
    proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    include_generated_qa: bool = True
    replace_primary: bool = True


class CaptionBatchAdapter:
    """Private GPU worker, with an atomic result file after every attempt."""
    def __init__(self, *, config, device):
        self.config, self.device, self.qwen = config, device, None

    def _runtime(self):
        if self.qwen is None:
            from services.hosted.qwen_adapters import QwenRuntime
            self.qwen = QwenRuntime(self.config["qwen"], self.device)
        return self.qwen

    def execute(self, operation, payload, context):
        if operation != "qwen_caption_batch" or context["actor_id"] != self.config["qwen"]["actor_id"]:
            raise ValueError("caption_batch_context_invalid")
        from PIL import Image
        from services.hosted.analysis import read_canonical_snapshot
        from services.hosted.caption_case import execute_caption_case
        from tools import run_qwen_caption_flow_benchmark as runner
        manifest = read_canonical_snapshot(payload["canonical_snapshot"], allowed_root=self.config["canonical_root"])
        if manifest["snapshot_id"] != context["annotation_version"] or manifest["dataset_id"] != context["dataset_id"]:
            raise ValueError("caption_batch_snapshot_changed")
        directory = Path(self.config["output_directory"])
        settings = CaptionSettings(**payload["settings"])
        by_id = {image["id"]: image for image in manifest["images"]}
        for image_id in payload["image_ids"]:
            image = by_id[image_id]
            case = canonical_case(image, manifest["classes"])
            history = payload.get("previous", {}).get(image_id, [])
            if history and history[-1]["final_status"] == "ok":
                continue
            for attempt in range(len(history) + 1, settings.attempts + 1):
                started = time.time()
                _atomic(directory / "progress.json", {"image_id": image_id, "attempt": attempt,
                    "started_at": started, "timeout": settings.per_image_timeout_seconds})
                output, error = None, None
                try:
                    with Image.open(image["media_path"]) as opened:
                        opened.load()
                        pixels = opened.convert("RGB")
                    output = execute_caption_case(self._runtime(), pixels,
                        {"caption_request": payload["caption_request"], "settings": settings.model_dump(),
                         "case": case, "media_path": image["media_path"]},
                        {**context, "image_id": image_id, "annotation_version": image["version_id"], "snapshot_id": manifest["snapshot_id"]})
                    quality_failures = runner.caption_quality_failures(output.get("caption_quality"))
                    succeeded = output.get("status") in {"ok", "preview_only"} and (not quality_failures or settings.continue_on_quality_failures)
                except Exception as exc:
                    error = type(exc).__name__
                    quality_failures, succeeded = [], False
                    self.close()
                response = (output or {}).get("response", {})
                events = response.get("hosted_evidence", {}).get("model_io", [])
                telemetry = runner.summarize_qwen_caption_io_event_lines([_json(event) for event in events])
                # The raw events and this summary are written in the same
                # immutable attempt artifact; no historical log is substituted.
                telemetry["source"] = "hosted_private_runtime"
                record = {"case_id": image_id, "image_name": image["filename"], "version_id": image["version_id"],
                    "attempt": attempt, "elapsed_seconds": time.time() - started, "status": (output or {}).get("status", "error"),
                    "final_status": "ok" if succeeded else "failed_attempt", "error": error,
                    "caption_quality": (output or {}).get("caption_quality"), "quality_failures": quality_failures,
                    "recovery_events": response.get("recovery_events", []), "qwen_caption_io": telemetry}
                _atomic(directory / f"{image_id}.{attempt}.json", {"row": record, "output": output})
                if succeeded:
                    break
                if attempt < settings.attempts:
                    time.sleep(settings.cooldown_after_crash_seconds)
        _atomic(directory / "progress.json", {"complete": True})
        return {"snapshot_id": manifest["snapshot_id"], "processed": len(payload["image_ids"])}

    def close(self):
        if self.qwen is not None:
            self.qwen.close()
            self.qwen = None


def evaluate_batch(rows, image_ids, settings, pilot, *, target_cases):
    """Use the native health, timing and recovery-confidence calculations."""
    from tools.audit_qwen_caption_soak import _degraded_rate_checks
    from tools import certify_qwen_caption_soak as certify
    latest = {row["case_id"]: row for row in rows}
    values = list(latest.values())
    failed = sum(row.get("final_status") != "ok" for row in values) + len(set(image_ids) - set(latest))
    quality_failed = sum(bool(row.get("quality_failures")) for row in values)
    rates, violations = _degraded_rate_checks(latest_rows=values, rows=rows, expected_cases=len(image_ids),
        failed_cases=failed, pending_failed_attempt_cases=0, quality_failed=quality_failed, heartbeat_status="completed",
        **{key: getattr(settings, key) for key in CaptionSettings.model_fields if key.startswith("max_") and key.endswith("_rate")},
        min_rate_cases=settings.min_rate_cases)
    health = {"status": "error" if violations or failed else "ok", "rates": rates, "violations": violations,
              "failed": failed, "processed": len(latest), "total_cases": len(image_ids)}
    if pilot is None:
        return health, None
    generated = [row for row in values if row.get("final_status") == "ok" and row.get("status") != "preview_only"]
    timing = certify._timing_summary({key: sum(float(row.get("elapsed_seconds", 0)) for row in rows if row["case_id"] == key)
                                     for key in {row["case_id"] for row in generated}})
    confidence = certify._deterministic_recovery_reliability_summary(generated,
        limit=settings.max_deterministic_recovery_case_rate, confidence=pilot.deterministic_recovery_confidence,
        enabled=pilot.deterministic_recovery_confidence > 0 and settings.max_deterministic_recovery_case_rate >= 0)
    projected = timing["mean_case_seconds"] * target_cases / 3600 * pilot.duration_safety_factor
    checks = [
        {"name": "health", "passed": health["status"] == "ok"},
        {"name": "generated_sample", "passed": len(generated) >= pilot.min_cases, "actual": len(generated), "required": pilot.min_cases},
        {"name": "runtime_projection", "passed": bool(timing["timed_cases"]) and projected <= pilot.max_duration_hours,
         "projected_hours": projected, "maximum_hours": pilot.max_duration_hours},
        {"name": "runtime_tail", "passed": bool(timing["timed_cases"]) and timing["p95_case_seconds"] * target_cases / 3600 <= pilot.max_duration_hours},
        {"name": "prompt_budget", "passed": not pilot.require_prompt_budget or bool(generated) and all(row.get("qwen_caption_io", {}).get("prompt_budget_events", 0) > 0 for row in generated)},
        {"name": "deterministic_recovery_confidence", "passed": not confidence["enabled"] or confidence["upper_bound_rate"] <= confidence["limit"]},
    ]
    return health, {"status": "certified" if all(row["passed"] for row in checks) else "not_certified",
        "scope": "hosted_sequential_local_qwen", "checks": checks, "timing": timing, "recovery_confidence": confidence,
        "target_cases": target_cases, "annotation_writes": False}


class CaptionBatches:
    def __init__(self, tools):
        self.tools, self.store = tools, tools.store
        self.root = tools.root / "caption-batches"
        self.root.mkdir(exist_ok=True, mode=0o700)
        self.lock, self.stop, self.threads = threading.RLock(), threading.Event(), {}
        self.recovered = False
        with self.store.transaction() as con:
            con.execute("CREATE TABLE IF NOT EXISTS hosted_caption_batches(id TEXT PRIMARY KEY,actor_id TEXT NOT NULL,actor_json TEXT NOT NULL,request_json TEXT NOT NULL,identity TEXT NOT NULL,selection_json TEXT NOT NULL,state TEXT NOT NULL,worker_id TEXT,result_json TEXT,error TEXT,cancel_requested INTEGER NOT NULL DEFAULT 0,resumes INTEGER NOT NULL DEFAULT 0,created_at REAL NOT NULL,updated_at REAL NOT NULL)")
            con.execute("CREATE TABLE IF NOT EXISTS hosted_caption_batch_attempts(batch_id TEXT NOT NULL, image_id TEXT NOT NULL,attempt INTEGER NOT NULL,row_json TEXT NOT NULL,sha256 TEXT NOT NULL,size_bytes INTEGER NOT NULL,PRIMARY KEY(batch_id,image_id,attempt))")

    def _row(self, actor, job_id):
        with self.store.read_connection() as con:
            if actor:
                self.store._actor(con, actor, admin=True)
            row = con.execute("SELECT * FROM hosted_caption_batches WHERE id=?", (job_id,)).fetchone()
            if row is None or actor and row["actor_id"] != actor.user_id:
                raise DomainError("caption_batch_not_found", 404)
            return dict(row)

    def _update(self, job_id, **values):
        values["updated_at"] = self.store.clock()
        with self.store.transaction() as con:
            con.execute("UPDATE hosted_caption_batches SET " + ",".join(key + "=?" for key in values) + " WHERE id=?", (*values.values(), job_id))

    def _validate(self, actor, body):
        self.tools.auth.require_admin(actor)
        manifest = self.store.get_snapshot(actor, body.snapshot_id)["manifest"]
        if manifest["dataset_id"] != body.dataset_id:
            raise DomainError("tool_dataset_mismatch", 409)
        from services.hosted.tool_api import _parameters_for
        if set(body.caption_request) - _parameters_for("qwen_caption"):
            raise DomainError("tool_parameter_not_allowed", 422)
        from models.schemas import QwenCaptionRequest
        QwenCaptionRequest(**body.caption_request, image_token="canonical-snapshot")
        if body.provider == "openai":
            from services.hosted.paid_caption import validate_paid_request
            if body.provider_options is None:
                raise DomainError("paid_caption_options_required", 422)
            provider_identity = validate_paid_request(self.tools.provider_gateway, actor, body.dataset_id,
                body.caption_request, body.provider_options)
            identity = hashlib.sha256(_json({"provider_identity": provider_identity, "snapshot_id": body.snapshot_id,
                "caption_request": body.caption_request, "settings": body.settings.model_dump()}).encode()).hexdigest()
            return manifest, None, identity
        if body.provider_options is not None:
            raise DomainError("paid_caption_provider_required", 422)
        qwen = self.tools._qwen_config(actor, required=True)
        for key in ("model_id", "refinement_model_id", "caption_fallback_model_id"):
            selected = body.caption_request.get(key)
            if selected and selected not in {"auto", "active", "none", "same"} and selected not in qwen["models"]:
                raise DomainError("tool_model_not_allowed", 403)
        # Paths may change during a restore; immutable model content may not.
        model_ids = {key: self.tools.registry.model_identity(key) for key in qwen["models"]}
        identity = hashlib.sha256(_json({"models": model_ids, "qwen": {key: value for key, value in qwen.items() if key not in {"models", "actor_id"}},
            "snapshot_id": body.snapshot_id, "caption_request": body.caption_request, "settings": body.settings.model_dump()}).encode()).hexdigest()
        return manifest, qwen, identity

    def _recover(self):
        with self.lock:
            if self.recovered:
                return
            self.tools._ensure_runtime()
            with self.store.transaction() as con:
                con.execute("UPDATE hosted_caption_batches SET state='interrupted',error='control_restarted' WHERE state IN ('queued','running','paused')")
                rows = con.execute("SELECT id FROM hosted_caption_batches WHERE state='interrupted'").fetchall()
            self.recovered = True
            for row in rows:
                self._automatic_resume(row[0])

    def start(self, actor, body):
        self._recover()
        requested = body.model_dump()
        previous = self.store.raw_receipt(actor, body.operation_id)
        if not previous:
            from services.hosted.tool_api import qwen_guidance
            frozen = self.store.get_snapshot(actor, body.snapshot_id)["manifest"]
            body = body.model_copy(update={"caption_request": qwen_guidance(frozen.get("settings", {}), "qwen_caption", body.caption_request)})
            manifest, _, identity = self._validate(actor, body)
            from tools.run_qwen_caption_flow_benchmark import sample_cases_with_meta
            selected, selection = sample_cases_with_meta([canonical_case(image, manifest["classes"]) for image in manifest["images"]], sample_size=body.sample_size, sample_seed=body.sample_seed)
            if not selected:
                raise DomainError("caption_batch_empty_snapshot", 422)
            image_ids = [case["case_id"] for case in selected]
            if body.certified_pilot_id:
                pilot = self._row(actor, body.certified_pilot_id)
                result = json.loads(pilot["result_json"] or "{}")
                if pilot["identity"] != identity or result.get("certification", {}).get("status") != "certified":
                    raise DomainError("caption_pilot_not_compatible", 409)
        created = False
        def create(con):
            nonlocal created
            active = con.execute("SELECT COUNT(*) FROM hosted_caption_batches WHERE state IN ('queued','running','paused')").fetchone()[0]
            if active:
                raise DomainError("caption_batch_busy", 409)
            job_id, now = uuid.uuid4().hex, self.store.clock()
            con.execute("INSERT INTO hosted_caption_batches(id,actor_id,actor_json,request_json,identity,selection_json,state,created_at,updated_at) VALUES(?,?,?,?,?,?,'queued',?,?)",
                (job_id, actor.user_id, _json(asdict(actor)), body.model_dump_json(), identity, _json({"image_ids": image_ids, "sample_selection": selection}), now, now))
            created = True
            return {"job_id": job_id}
        receipt = self.store._command(actor, body.operation_id, "caption_batch_start", requested, create, admin=True)
        if created:
            self._launch(receipt["job_id"])
        return self.status(actor, receipt["job_id"])

    def _launch(self, job_id):
        with self.lock:
            if job_id in self.threads and self.threads[job_id].is_alive():
                return
            thread = threading.Thread(target=self._run, args=(job_id,), daemon=True, name="hosted-caption-batch")
            self.threads[job_id] = thread
            thread.start()

    def _attempts(self, job_id):
        with self.store.read_connection() as con:
            return [dict(row) for row in con.execute("SELECT * FROM hosted_caption_batch_attempts WHERE batch_id=? ORDER BY image_id,attempt", (job_id,))]

    def _retain(self, actor, job_id, directory, allowed, known=None):
        if known is None:
            known = {(row["image_id"], row["attempt"]) for row in self._attempts(job_id)}
        for path in directory.glob("*.json"):
            if path.name in {"progress.json", "canonical-snapshot.json"}:
                continue
            parts = path.name.split(".")
            if len(parts) != 3 or parts[0] not in allowed or not parts[1].isdigit():
                raise ValueError("caption_batch_evidence_invalid")
            key = (parts[0], int(parts[1]))
            if key in known:
                continue
            data = _read(path)
            if (data["row"]["case_id"], data["row"]["attempt"]) != key:
                raise ValueError("caption_batch_evidence_invalid")
            with self.tools.artifacts.exclusive():
                reference = self.tools.artifacts.put_bytes(_json(data).encode())
                with self.store.transaction() as con:
                    self.store._actor(con, actor, admin=True)
                    con.execute("INSERT OR IGNORE INTO hosted_artifact_references VALUES ('caption_batch',?,?,?,?)", (job_id, reference.sha256, reference.size_bytes, self.store.clock()))
                    con.execute("DELETE FROM hosted_orphan_observations WHERE sha256=?", (reference.sha256,))
                    con.execute("INSERT OR IGNORE INTO hosted_caption_batch_attempts VALUES (?,?,?,?,?,?)", (job_id, *key, _json(data["row"]), reference.sha256, reference.size_bytes))
            known.add(key)
            # The retained content hash is the recovery source. Removing its
            # working copy keeps scanning cost bounded across long runs.
            path.unlink()
        return known

    def _spec(self, actor, qwen, directory):
        return AdapterSpec("services.hosted.caption_batch:CaptionBatchAdapter", {"qwen": qwen,
            "canonical_root": str(self.root), "output_directory": str(directory)}, qwen.get("device", "cpu"))

    def _run(self, job_id):
        worker_id, directory, actor = None, None, None
        try:
            row = self._row(None, job_id)
            actor, body = Actor(**json.loads(row["actor_json"])), CaptionBatchRequest(**json.loads(row["request_json"]))
            manifest, qwen, identity = self._validate(actor, body)
            if identity != row["identity"]:
                raise DomainError("caption_batch_configuration_changed", 409)
            selection = json.loads(row["selection_json"])
            image_ids = selection["image_ids"]
            directory = self.root / job_id
            directory.mkdir(exist_ok=True, mode=0o700)
            progress_path = directory / "progress.json"
            if progress_path.exists() and body.provider == "local_qwen":
                self._record_interruption(actor, job_id, directory, _read(progress_path, 8192), "control_restarted")
            known = self._retain(actor, job_id, directory, image_ids)
            progress_path.unlink(missing_ok=True)
            history = {}
            for attempt in self._attempts(job_id):
                history.setdefault(attempt["image_id"], []).append(json.loads(attempt["row_json"]))
            # Capture only the selected confirmed members. The original snapshot
            # ID/revisions remain unchanged and source membership is checked.
            selected_ids = set(image_ids)
            content = json.loads(_json({**manifest, "snapshot_id": body.snapshot_id,
                "images": [image for image in manifest["images"] if image["id"] in selected_ids]}))
            with self.tools.artifacts.exclusive():
                for image in content["images"]:
                    view = self.tools.media.resolve(image["media_hash"])
                    ref = self.tools.artifacts.pin(self.tools.media.reference(view), directory)
                    image["working_media"] = view
                    image["media_path"] = str(directory / ref.sha256)
                _atomic(directory / "canonical-snapshot.json", content)
            if body.provider == "openai":
                self._run_paid(actor, job_id, body, identity, content, image_ids, history, directory, known)
                if self._validate(actor, body)[2] != identity:
                    raise DomainError("caption_batch_configuration_changed", 409)
                self._finish(actor, job_id, body, identity, manifest, selection)
                shutil.rmtree(directory)
                directory = None
                return
            reference = {key: content[key] for key in ("snapshot_id", "dataset_id", "revision", "class_version", "settings_version")}
            reference.update(manifest_path=str(directory / "canonical-snapshot.json"), manifest_sha256=hashlib.sha256((directory / "canonical-snapshot.json").read_bytes()).hexdigest())
            context = RequestContext(actor.user_id, body.dataset_id, "", body.snapshot_id, 0, "", actor_role=actor.role,
                session_id=actor.session_id, instance_id=actor.instance_id, operation="qwen_caption_batch")
            runtime, spec = self.tools._ensure_runtime(), self._spec(actor, qwen, directory)
            while True:
                self._check(job_id, actor, body, identity)
                try:
                    worker_id = runtime.begin_heavy(context, spec, "qwen_caption_batch", {"canonical_snapshot": reference,
                        "image_ids": image_ids, "caption_request": body.caption_request, "settings": body.settings.model_dump(), "previous": history},
                        timeout=max(300, len(image_ids) * body.settings.attempts * (body.settings.per_image_timeout_seconds + body.settings.cooldown_after_crash_seconds) + 300))
                    break
                except RuntimeRejected as exc:
                    if exc.code != "runtime_heavy_window_busy":
                        raise
                    self._update(job_id, state="paused")
                    self.stop.wait(.2)
            self._update(job_id, state="running", worker_id=worker_id)
            while True:
                self._check(job_id, actor, body, identity)
                self._retain(actor, job_id, directory, selected_ids, known)
                state = runtime.request_status(worker_id, actor_id=actor.user_id)
                if state["state"] in TERMINAL:
                    break
                progress_path = directory / "progress.json"
                if progress_path.exists():
                    progress = _read(progress_path, 8192)
                    if not progress.get("complete") and time.time() - progress["started_at"] > body.settings.per_image_timeout_seconds:
                        runtime.cancel(worker_id, actor_id=actor.user_id)
                        self._record_interruption(actor, job_id, directory, progress, "worker_timeout")
                        raise RuntimeRejected("worker_timeout")
                self.stop.wait(.1)
            self._retain(actor, job_id, directory, selected_ids, known)
            if state["state"] != "completed":
                if (directory / "progress.json").exists():
                    self._record_interruption(actor, job_id, directory, _read(directory / "progress.json", 8192), state.get("error") or "worker_failed")
                raise RuntimeRejected(state.get("error") or "worker_failed")
            runtime.result(worker_id, actor_id=actor.user_id)
            if self._validate(actor, body)[2] != identity:
                raise DomainError("caption_batch_configuration_changed", 409)
            self._finish(actor, job_id, body, identity, manifest, selection)
            shutil.rmtree(directory)
            directory = None
        except Exception as exc:
            code = exc.code if isinstance(exc, (DomainError, RuntimeRejected)) else "caption_batch_failed"
            state = "cancelled" if code == "caption_batch_cancelled" else "interrupted" if code == "caption_control_stopped" else "failed"
            if worker_id and actor:
                try:
                    self.tools._runtime.cancel(worker_id, actor_id=actor.user_id)
                except Exception:
                    pass
            if directory and actor:
                try:
                    if (directory / "progress.json").exists() and body.provider == "local_qwen":
                        self._record_interruption(actor, job_id, directory, _read(directory / "progress.json", 8192), code)
                    row = self._row(None, job_id)
                    self._retain(actor, job_id, directory, json.loads(row["selection_json"])["image_ids"])
                except Exception:
                    pass
            self._update(job_id, state=state, error=code)
        finally:
            with self.lock:
                self.threads.pop(job_id, None)
                if not self.stop.is_set():
                    row = self._row(None, job_id)
                    if row["state"] == "queued":
                        self._launch(job_id)
                    elif row["error"] in {"worker_timeout", "worker_disconnected", "worker_failed"}:
                        self._automatic_resume(job_id)

    def _run_paid(self, actor, job_id, body, identity, manifest, image_ids, history, directory, known):
        from PIL import Image
        from services.hosted.caption_case import execute_caption_case
        from services.hosted.paid_caption import PaidCaptionRuntime
        from services.hosted.providers import SnapshotAuthority
        from tools import run_qwen_caption_flow_benchmark as runner
        by_id = {image["id"]: image for image in manifest["images"]}
        self._update(job_id, state="running", worker_id=None)
        for image_id in image_ids:
            previous = history.get(image_id, [])
            if previous and previous[-1]["final_status"] == "ok":
                continue
            image = by_id[image_id]
            for attempt in range(len(previous) + 1, body.settings.attempts + 1):
                self._check(job_id, actor, body, identity)
                started = time.monotonic()
                deadline = started + body.settings.per_image_timeout_seconds
                def check():
                    nonlocal deadline
                    self._check(job_id, actor, body, identity)
                    if time.monotonic() >= deadline:
                        raise RuntimeRejected("paid_caption_attempt_timeout")
                    pause_started, paused = time.monotonic(), False
                    while self.tools._runtime.snapshot()["phase"] != "available":
                        self._check(job_id, actor, body, identity)
                        if not paused:
                            self._update(job_id, state="paused")
                            paused = True
                        self.stop.wait(.1)
                    if paused:
                        deadline += time.monotonic() - pause_started
                        self._update(job_id, state="running")
                def dispatch_guard(con):
                    row = con.execute("SELECT state,cancel_requested FROM hosted_caption_batches WHERE id=?", (job_id,)).fetchone()
                    if self.stop.is_set() or row is None or row["cancel_requested"] or row["state"] != "running":
                        raise DomainError("caption_batch_cancelled", 409)
                    if time.monotonic() >= deadline:
                        raise DomainError("paid_caption_attempt_timeout", 409)
                    if self.tools._runtime.snapshot()["phase"] != "available":
                        raise DomainError("runtime_automation_paused", 409)
                authority = SnapshotAuthority(body.snapshot_id, image_id, image["version_id"])
                runtime = PaidCaptionRuntime(self.tools.provider_gateway, actor, authority,
                    source_media_hash=image["media_hash"], options=body.provider_options,
                    stage_prefix=f"{job_id}:{image_id}:{attempt}", check=check, dispatch_guard=dispatch_guard)
                try:
                    with Image.open(image["media_path"]) as opened:
                        pixels = opened.convert("RGB")
                    output = execute_caption_case(runtime, pixels, {"caption_request": body.caption_request,
                        "settings": body.settings.model_dump(), "case": canonical_case(image, manifest["classes"]),
                        "media_path": image["media_path"]}, {"actor_id": actor.user_id})
                    if runtime.fatal is not None:
                        raise runtime.fatal
                finally:
                    runtime.close()
                check()
                failures = runner.caption_quality_failures(output.get("caption_quality"))
                succeeded = output.get("status") in {"ok", "preview_only"} and (not failures or body.settings.continue_on_quality_failures)
                response = output.get("response", {})
                telemetry = runner.summarize_qwen_caption_io_event_lines([_json(event) for event in response.get("hosted_evidence", {}).get("model_io", [])])
                telemetry["source"] = "hosted_metered_responses"
                record = {"case_id": image_id, "image_name": image["filename"], "version_id": image["version_id"],
                    "attempt": attempt, "elapsed_seconds": time.monotonic() - started, "status": output.get("status", "error"),
                    "final_status": "ok" if succeeded else "failed_attempt", "error": None,
                    "caption_quality": output.get("caption_quality"), "quality_failures": failures,
                    "recovery_events": response.get("recovery_events", []), "qwen_caption_io": telemetry}
                _atomic(directory / f"{image_id}.{attempt}.json", {"row": record, "output": output})
                self._retain(actor, job_id, directory, image_ids, known)
                if succeeded:
                    break
                until = time.monotonic() + body.settings.cooldown_after_crash_seconds
                while time.monotonic() < until and attempt < body.settings.attempts:
                    self._check(job_id, actor, body, identity)
                    self.stop.wait(.1)

    def _record_interruption(self, actor, job_id, directory, progress, error):
        if progress.get("complete"):
            return
        with self.store.read_connection() as con:
            if con.execute("SELECT 1 FROM hosted_caption_batch_attempts WHERE batch_id=? AND image_id=? AND attempt=?", (job_id, progress["image_id"], progress["attempt"])).fetchone():
                return
        path = directory / f"{progress['image_id']}.{progress['attempt']}.json"
        if not path.exists():
            _atomic(path, {"row": {"case_id": progress["image_id"], "attempt": progress["attempt"],
                "elapsed_seconds": max(0, time.time() - progress["started_at"]), "status": "interrupted",
                "final_status": "failed_attempt", "error": error}, "output": None})

    def _check(self, job_id, actor, body, identity):
        if self.stop.is_set():
            raise RuntimeRejected("caption_control_stopped")
        if self._row(None, job_id)["cancel_requested"]:
            raise RuntimeRejected("caption_batch_cancelled")
        # The captured snapshot and worker spec are immutable. Recheck live
        # authentication on every poll; re-reading a large snapshot here would
        # make monitoring cost grow with the entire dataset at 10 Hz.
        self.tools.auth.require_admin(actor)

    def _finish(self, actor, job_id, body, identity, manifest, selection):
        records = self._attempts(job_id)
        rows = [json.loads(row["row_json"]) for row in records]
        health, certification = evaluate_batch(rows, selection["image_ids"], body.settings, body.pilot, target_cases=len(manifest["images"]))
        if certification is not None and body.provider == "openai":
            certification["scope"] = "hosted_sequential_metered_responses"
        # Each response is retained separately; the API returns a bounded index
        # and paged captions, not every raw model response over one IPC frame.
        result = {"total_cases": len(selection["image_ids"]), "processed": health["processed"], "failed": health["failed"],
            "attempts_seen": len(rows), "saved_text_labels": 0, "health": health, "certification": certification,
            "snapshot_id": body.snapshot_id, "dataset_revision": manifest["revision"], "sample_selection": selection["sample_selection"],
            "snapshot_sha256": hashlib.sha256(_json(manifest).encode()).hexdigest(), "runtime_identity": identity}
        with self.tools.artifacts.exclusive():
            ref = self.tools.artifacts.put_bytes(_json({"result": result, "attempts": [{key: value for key, value in item.items() if key != "row_json"} for item in records]}).encode())
            with self.store.transaction() as con:
                self.store._actor(con, actor, admin=True)
                current = con.execute("SELECT cancel_requested FROM hosted_caption_batches WHERE id=?", (job_id,)).fetchone()
                if current[0] or self.stop.is_set():
                    raise RuntimeRejected("caption_batch_cancelled" if current[0] else "caption_control_stopped")
                con.execute("INSERT OR IGNORE INTO hosted_artifact_references VALUES ('caption_batch',?,?,?,?)", (job_id, ref.sha256, ref.size_bytes, self.store.clock()))
                con.execute("DELETE FROM hosted_orphan_observations WHERE sha256=?", (ref.sha256,))
                result["evidence"] = ref.as_dict()
                con.execute("UPDATE hosted_caption_batches SET state=?,result_json=?,error=?,updated_at=? WHERE id=?", (
                    "completed" if health["status"] == "ok" else "failed", _json(result), None if health["status"] == "ok" else "caption_health_gate_failed", self.store.clock(), job_id))

    def status(self, actor, job_id):
        self._recover()
        row = self._row(actor, job_id)
        with self.store.read_connection() as con:
            processed = con.execute("SELECT COUNT(DISTINCT image_id) FROM hosted_caption_batch_attempts WHERE batch_id=?", (job_id,)).fetchone()[0]
        result = json.loads(row["result_json"]) if row["result_json"] else {"total_cases": len(json.loads(row["selection_json"])["image_ids"]), "processed": processed, "saved_text_labels": 0}
        return {"job_id": job_id, "status": row["state"], "error": row["error"], "result": result,
            "auto_resume_count": row["resumes"], "request": json.loads(row["request_json"]), "created_at": row["created_at"], "updated_at": row["updated_at"]}

    def list(self, actor):
        self._recover()
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            rows = con.execute("SELECT id AS job_id,state AS status,error,request_json,created_at,updated_at FROM hosted_caption_batches WHERE actor_id=? ORDER BY created_at DESC LIMIT 100", (actor.user_id,)).fetchall()
        return [{**{key: row[key] for key in ("job_id", "status", "error", "created_at", "updated_at")},
                 "request": json.loads(row["request_json"])} for row in rows]

    def captions(self, actor, job_id, offset=0, limit=25):
        self._row(actor, job_id)
        if offset < 0 or not 1 <= limit <= 100:
            raise DomainError("caption_page_invalid", 422)
        latest = {}
        for record in self._attempts(job_id):
            latest[record["image_id"]] = record
        result = []
        for record in list(latest.values())[offset:offset + limit]:
            with self.tools.artifacts.open(record["sha256"]) as stream:
                data = json.load(stream)
            response = (data.get("output") or {}).get("response", {})
            result.append({"image_id": record["image_id"], "row": json.loads(record["row_json"]),
                "caption": response.get("caption"), "generated_qa_pairs": response.get("generated_qa_pairs", []),
                "evidence": {"sha256": record["sha256"], "size_bytes": record["size_bytes"]}})
        return {"total": len(latest), "offset": offset, "captions": result}

    def proposal(self, actor, job_id, image_id):
        row = self._row(actor, job_id)
        request = json.loads(row["request_json"])
        manifest = self.store.get_snapshot(actor, request["snapshot_id"])["manifest"]
        original = next((image for image in manifest["images"] if image["id"] == image_id), None)
        with self.store.read_connection() as con:
            attempt = con.execute("SELECT * FROM hosted_caption_batch_attempts WHERE batch_id=? AND image_id=? ORDER BY attempt DESC LIMIT 1", (job_id, image_id)).fetchone()
        if original is None or attempt is None:
            raise DomainError("caption_proposal_not_found", 404)
        with self.tools.artifacts.open(attempt["sha256"]) as stream:
            evidence = json.load(stream)
        response = (evidence.get("output") or {}).get("response", {})
        if not str(response.get("caption", "")).strip():
            raise DomainError("caption_proposal_unavailable", 409)
        return {"batch_id": job_id, "image_id": image_id, "dataset_id": manifest["dataset_id"],
            "snapshot_id": request["snapshot_id"], "source_version": original["version_id"],
            "caption": response["caption"], "generated_qa_pairs": bind_machine_validated_pairs(response.get("generated_qa_pairs", [])),
            "proposal_sha256": attempt["sha256"], "attempt": attempt["attempt"],
            "provider": request.get("provider", "local_qwen"), "runtime_identity": row["identity"],
            **{key: response[key] for key in ("used_counts", "used_boxes", "model_id", "runtime_platform") if key in response},
            "quality": evidence["row"].get("caption_quality"), "quality_failures": evidence["row"].get("quality_failures", []),
            "recovery_events": response.get("recovery_events", evidence["row"].get("recovery_events", [])), "requires_review": True}

    def apply(self, actor, job_id, image_id, body):
        # The command fingerprint contains the immutable proposal digest. Exact
        # replay recovers its original receipt after this save advances version.
        self._row(actor, job_id)
        context = MutationContext(image_id, body.expected_version, body.generation, body.editor_id, body.operation_id)
        previous = self.store.raw_receipt(actor, body.operation_id)
        proposal = None if previous else self.proposal(actor, job_id, image_id)
        if proposal is not None and proposal["proposal_sha256"] != body.proposal_sha256:
            raise DomainError("caption_proposal_changed", 409)
        def commit(con):
            batch = con.execute("SELECT request_json FROM hosted_caption_batches WHERE id=?", (job_id,)).fetchone()
            request = json.loads(batch[0])
            snapshot = con.execute("SELECT manifest_json FROM hosted_snapshots WHERE id=?", (request["snapshot_id"],)).fetchone()
            manifest = json.loads(snapshot[0])
            source = next((image for image in manifest["images"] if image["id"] == image_id), None)
            if source is None or source["version_id"] != context.expected_version:
                raise DomainError("caption_proposal_source_changed", 409)
            image = self.store._guard(con, actor, context)
            if image["dataset_id"] != manifest["dataset_id"] or image["state"] != "confirmed":
                raise DomainError("caption_proposal_image_not_confirmed", 409)
            dataset = self.store._dataset(con, manifest["dataset_id"])
            if dataset["class_version"] != manifest["class_version"]:
                raise DomainError("caption_proposal_classes_changed", 409)
            version = con.execute("SELECT annotations_json,complete,negative FROM hosted_annotation_versions WHERE id=?", (context.expected_version,)).fetchone()
            annotations = json.loads(version["annotations_json"])
            captions = annotations.setdefault("alternate_captions", [])
            if not isinstance(captions, list):
                raise DomainError("caption_alternates_format_invalid", 409)
            provenance = {"source": "hosted_qwen", "batch_id": job_id, "snapshot_id": request["snapshot_id"],
                "source_version": context.expected_version, "evidence_sha256": body.proposal_sha256,
                "provider": proposal["provider"], "runtime_identity": proposal["runtime_identity"],
                **({"caption_quality": proposal["quality"]} if proposal["quality"] is not None else {}),
                **{key: proposal[key] for key in ("quality_failures", "used_counts", "used_boxes", "model_id", "runtime_platform", "recovery_events") if key in proposal}}
            text = proposal["caption"]
            if body.replace_primary:
                previous = str(annotations.get("caption", "")).strip()
                if previous and previous != text and not any(item == previous or isinstance(item, dict) and item.get("caption") == previous for item in captions):
                    captions.append({"caption": previous, "source": "previous_primary"})
                annotations["caption"] = text
            captions.append({"caption": text, **provenance,
                **({"generated_qa_pairs": proposal["generated_qa_pairs"]} if body.include_generated_qa else {})})
            return self.store._save(con, actor, context, annotations, complete=bool(version["complete"]), negative=bool(version["negative"]))
        return self.store._command(actor, context.operation_id, "apply_caption_proposal", {"batch_id": job_id, "image_id": image_id, **body.model_dump()}, commit, admin=True)

    def cancel(self, actor, job_id, operation_id):
        self._row(actor, job_id)
        def change(con):
            con.execute("UPDATE hosted_caption_batches SET cancel_requested=1,state=CASE WHEN state='waiting_for_admin' THEN 'cancelled' ELSE state END WHERE id=?", (job_id,))
            return {"job_id": job_id}
        self.store._command(actor, operation_id, "caption_batch_cancel", {"job_id": job_id}, change, admin=True)
        return self.status(actor, job_id)

    def resume(self, actor, job_id, operation_id, *, automatic=False):
        self._recover()
        row = self._row(actor, job_id)
        body = CaptionBatchRequest(**json.loads(row["request_json"]))
        if not self.store.raw_receipt(actor, operation_id) and self._validate(actor, body)[2] != row["identity"]:
            raise DomainError("caption_batch_configuration_changed", 409)
        changed = False
        def change(con):
            nonlocal changed
            current = con.execute("SELECT state,cancel_requested,resumes FROM hosted_caption_batches WHERE id=?", (job_id,)).fetchone()
            if current[0] not in {"failed", "interrupted", "cancelled", "waiting_for_admin"}:
                raise DomainError("caption_batch_not_resumable", 409)
            if automatic and (current[1] or not body.settings.set_and_forget or current[2] >= body.settings.max_auto_resumes):
                raise DomainError("caption_auto_resume_unavailable", 409)
            con.execute("UPDATE hosted_caption_batches SET actor_json=?,state='queued',worker_id=NULL,cancel_requested=0,error=NULL,result_json=NULL,resumes=resumes+? WHERE id=?", (_json(asdict(actor)), int(automatic), job_id))
            changed = True
            return {"job_id": job_id}
        self.store._command(actor, operation_id, "caption_batch_resume", {"job_id": job_id, "automatic": automatic}, change, admin=True)
        if changed:
            self._launch(job_id)
        return self.status(actor, job_id)

    def _automatic_resume(self, job_id):
        row = self._row(None, job_id)
        body = CaptionBatchRequest(**json.loads(row["request_json"]))
        if not body.settings.set_and_forget or row["resumes"] >= body.settings.max_auto_resumes or row["cancel_requested"]:
            return
        try:
            self.resume(Actor(**json.loads(row["actor_json"])), job_id, f"batch-auto-{job_id}-{row['resumes'] + 1}", automatic=True)
        except (DomainError, RuntimeRejected) as exc:
            self._update(job_id, state="waiting_for_admin", error=exc.code)

    def close(self):
        self.stop.set()
        for thread in list(self.threads.values()):
            thread.join(timeout=2)
