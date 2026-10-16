"""Durable, actor-bound caption attempts over the private inference supervisor."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import logging
import threading
import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from services.hosted.caption_case import CaptionSettings
from services.hosted.errors import DomainError
from services.hosted.runtime import RuntimeRejected, TERMINAL
from services.hosted.types import Actor, MutationContext
from services.hosted.paid_caption import PaidCaptionOptions


class CaptionJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    image_id: str
    expected_version: str
    generation: int = Field(ge=0)
    editor_id: str = Field(min_length=8, max_length=128)
    workspace_id: str = Field(default="", max_length=128)
    operation_id: str = Field(min_length=8, max_length=128)
    caption_request: dict = Field(default_factory=dict)
    settings: CaptionSettings = Field(default_factory=CaptionSettings)
    provider: Literal["local_qwen", "openai"] = "local_qwen"
    provider_options: PaidCaptionOptions | None = None


class CaptionJobCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: str = Field(min_length=8, max_length=128)


class CaptionJobs:
    def __init__(self, tools):
        self.tools, self.store = tools, tools.store
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.threads = {}
        self.recovered = False
        with self.store.transaction() as con:
            con.execute("CREATE TABLE IF NOT EXISTS hosted_caption_jobs (id TEXT PRIMARY KEY,actor_id TEXT NOT NULL,actor_json TEXT NOT NULL,request_json TEXT NOT NULL,runtime_identity TEXT NOT NULL,state TEXT NOT NULL,worker_id TEXT,attempts_json TEXT NOT NULL,result_json TEXT,error TEXT,cancel_requested INTEGER NOT NULL DEFAULT 0,created_at REAL NOT NULL,updated_at REAL NOT NULL)")
            if "auto_resumes" not in {row[1] for row in con.execute("PRAGMA table_info(hosted_caption_jobs)")}:
                con.execute("ALTER TABLE hosted_caption_jobs ADD COLUMN auto_resumes INTEGER NOT NULL DEFAULT 0")

    def _recover(self):
        with self.lock:
            if self.recovered:
                return
            self.tools._ensure_runtime()  # Acquires the sole controller lock.
            with self.store.transaction() as con:
                con.execute("UPDATE hosted_caption_jobs SET state='interrupted',error='control_restarted',updated_at=? WHERE state IN ('queued','running','paused','retrying')", (self.store.clock(),))
                interrupted = con.execute("SELECT id,worker_id,attempts_json FROM hosted_caption_jobs WHERE state='interrupted' AND error IN ('control_restarted','caption_control_stopped')").fetchall()
                for row in interrupted:
                    attempts = json.loads(row["attempts_json"])
                    if row["worker_id"] and not any(item.get("worker_id") == row["worker_id"] for item in attempts):
                        attempts.append({"worker_id": row["worker_id"], "attempt": len(attempts) + 1,
                            "status": "interrupted", "error": "control_restarted", "final_status": "failed_attempt"})
                        con.execute("UPDATE hosted_caption_jobs SET attempts_json=? WHERE id=?", (json.dumps(attempts), row["id"]))
            self.recovered = True
            for row in interrupted:
                self._automatic_resume(row[0])

    def _row(self, job_id, actor=None):
        with self.store.read_connection() as con:
            if actor:
                self.store._actor(con, actor)
            row = con.execute("SELECT * FROM hosted_caption_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None or actor and row["actor_id"] != actor.user_id:
                raise DomainError("caption_job_not_found", 404)
            return dict(row)

    @staticmethod
    def _tool_request(body):
        from services.hosted.tool_api import ToolRequest
        return ToolRequest(**{key: getattr(body, key) for key in ("dataset_id", "image_id", "expected_version", "generation", "editor_id", "workspace_id")},
            operation="qwen_caption_case", parameters={"caption_request": body.caption_request, "settings": body.settings.model_dump()})

    def _validate(self, actor, body):
        from services.hosted.tool_api import _parameters_for
        if set(body.caption_request) - _parameters_for("qwen_caption"):
            raise DomainError("tool_parameter_not_allowed", 422)
        # Schema validation happens before consuming an idempotent command.
        from models.schemas import QwenCaptionRequest
        QwenCaptionRequest(**body.caption_request, image_token="hosted-current-image")
        if body.provider == "openai":
            from services.hosted.paid_caption import validate_paid_request
            if body.provider_options is None:
                raise DomainError("paid_caption_options_required", 422)
            image = self.store.authorize_context(actor, MutationContext(body.image_id, body.expected_version,
                body.generation, body.editor_id, body.operation_id), write=True)
            if image["dataset_id"] != body.dataset_id:
                raise DomainError("tool_dataset_mismatch", 409)
            identity = validate_paid_request(self.tools.provider_gateway, actor, body.dataset_id,
                body.caption_request, body.provider_options)
            classes = self.store.get_dataset(actor, body.dataset_id)["class_version"]
            return hashlib.sha256(f"{identity}:{classes}".encode()).hexdigest()
        if body.provider_options is not None:
            raise DomainError("paid_caption_provider_required", 422)
        tool_request = self._tool_request(body)
        context = self.tools._context(actor, tool_request)
        from dataclasses import replace
        context = replace(context, operation=tool_request.operation)
        self.tools.authorize(context, "admit")
        spec, _ = self.tools._spec(actor, body.dataset_id, tool_request.operation)
        for field in ("model_id", "refinement_model_id", "caption_fallback_model_id"):
            model_id = body.caption_request.get(field)
            if model_id and model_id not in {"active", "auto", "none", "same"} and model_id not in spec.config["qwen"]["models"]:
                raise DomainError("tool_model_not_allowed", 403)
        # Class names feed canonical label hints even when no classifier is
        # selected. Changing that map must not silently change a retry's input.
        classes = self.store.get_dataset(actor, body.dataset_id)["class_version"]
        return hashlib.sha256(f"{spec.identity}:{classes}".encode()).hexdigest()

    def start(self, actor, body):
        self._recover()
        requested = body.model_dump()
        previous = self.store.raw_receipt(actor, body.operation_id)
        if not previous:
            from services.hosted.tool_api import qwen_guidance
            settings = self.store.get_dataset(actor, body.dataset_id)["settings"]
            body = body.model_copy(update={"caption_request": qwen_guidance(settings, "qwen_caption", body.caption_request)})
        identity = self._validate(actor, body) if not previous else ""
        created = False
        def create(con):
            nonlocal created
            active = con.execute("SELECT COUNT(*) FROM hosted_caption_jobs WHERE actor_id=? AND state IN ('queued','running','paused','retrying')", (actor.user_id,)).fetchone()[0]
            if active >= int(self.tools.configuration.get("caption_jobs", {}).get("pending_per_user", 8)):
                raise DomainError("caption_job_queue_full", 429)
            created = True
            job_id, now = uuid.uuid4().hex, self.store.clock()
            con.execute("INSERT INTO hosted_caption_jobs (id,actor_id,actor_json,request_json,runtime_identity,state,worker_id,attempts_json,result_json,error,cancel_requested,created_at,updated_at) VALUES (?,?,?,?,?,'queued',NULL,'[]',NULL,NULL,0,?,?)",
                (job_id, actor.user_id, json.dumps(asdict(actor)), body.model_dump_json(), identity, now, now))
            return {"job_id": job_id}
        receipt = self.store._command(actor, body.operation_id, "start_caption_job", requested, create)
        if created:
            self._launch(receipt["job_id"])
        return self.status(actor, receipt["job_id"])

    def _launch(self, job_id):
        with self.lock:
            if job_id in self.threads and self.threads[job_id].is_alive():
                return
            thread = threading.Thread(target=self._run, args=(job_id,), daemon=True, name="hosted-caption-job")
            self.threads[job_id] = thread
            thread.start()

    def _update(self, job_id, **values):
        values["updated_at"] = self.store.clock()
        with self.store.transaction() as con:
            con.execute("UPDATE hosted_caption_jobs SET " + ",".join(key + "=?" for key in values) + " WHERE id=?", (*values.values(), job_id))

    def _check(self, job_id, actor, body, identity):
        row = self._row(job_id)
        if self.stop.is_set():
            raise RuntimeRejected("caption_control_stopped")
        if row["cancel_requested"]:
            raise RuntimeRejected("caption_job_cancelled")
        if self._validate(actor, body) != identity:
            raise RuntimeRejected("runtime_configuration_changed")

    def _wait(self, job_id, actor, body, identity, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._check(job_id, actor, body, identity)
            self.stop.wait(min(0.1, max(0, deadline - time.monotonic())))

    @staticmethod
    def _health(rows, settings, failed, quality_failed):
        from tools.audit_qwen_caption_soak import _degraded_rate_checks
        rates, violations = _degraded_rate_checks(latest_rows=rows[-1:], rows=rows, expected_cases=1,
            failed_cases=int(failed), pending_failed_attempt_cases=0, quality_failed=int(quality_failed),
            heartbeat_status="failed" if failed else "completed",
            **{key: getattr(settings, key) for key in CaptionSettings.model_fields if key.startswith("max_") and key.endswith("_rate")},
            min_rate_cases=settings.min_rate_cases)
        return {"status": "error" if violations else "ok", "rates": rates, "violations": violations}

    def _paid_attempt(self, job_id, actor, body, identity, attempt):
        """Reuse native orchestration on CPU; only the gateway can dispatch."""
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from PIL import Image
        from services.hosted.caption_batch import canonical_case
        from services.hosted.caption_case import execute_caption_case
        from services.hosted.paid_caption import PaidCaptionRuntime
        deadline = time.monotonic() + body.settings.per_image_timeout_seconds
        def check():
            nonlocal deadline
            self._check(job_id, actor, body, identity)
            if time.monotonic() >= deadline:
                raise RuntimeRejected("paid_caption_attempt_timeout")
            started, paused = time.monotonic(), False
            while self.tools._runtime.snapshot()["phase"] != "available":
                self._check(job_id, actor, body, identity)
                if not paused:
                    self._update(job_id, state="paused")
                    paused = True
                self.stop.wait(.1)
            if paused:
                deadline += time.monotonic() - started
                self._update(job_id, state="running")
        def dispatch_guard(con):
            # Runs inside the gateway's dispatch transaction. A committed
            # Cancel therefore prevents the next paid call from dispatching.
            row = con.execute("SELECT state,cancel_requested FROM hosted_caption_jobs WHERE id=?", (job_id,)).fetchone()
            if self.stop.is_set() or row is None or row["cancel_requested"] or row["state"] != "running":
                raise DomainError("caption_job_cancelled", 409)
            if time.monotonic() >= deadline:
                raise DomainError("paid_caption_attempt_timeout", 409)
            if self.tools._runtime.snapshot()["phase"] != "available":
                raise DomainError("runtime_automation_paused", 409)
        authority = MutationContext(body.image_id, body.expected_version, body.generation, body.editor_id, body.operation_id)
        image = self.store.image(actor, body.image_id)
        dataset = self.store.get_dataset(actor, body.dataset_id)
        runtime = PaidCaptionRuntime(self.tools.provider_gateway, actor, authority,
            source_media_hash=image["media_hash"], options=body.provider_options,
            stage_prefix=f"{job_id}:{attempt}", check=check, dispatch_guard=dispatch_guard)
        try:
            with TemporaryDirectory(prefix="paid-caption-", dir=self.tools.media_root) as temporary:
                with self.tools.artifacts.exclusive():
                    view = self.tools.media.resolve(image["media_hash"])
                    reference = self.tools.artifacts.pin(self.tools.media.reference(view), Path(temporary))
                path = Path(temporary) / reference.sha256
                from .media import decode_encoded
                pixels, _ = decode_encoded(path)
                output = execute_caption_case(runtime, pixels, {"caption_request": body.caption_request,
                    "settings": body.settings.model_dump(), "case": canonical_case(image, dataset["classes"]),
                    "media_path": str(path)}, {"actor_id": actor.user_id})
                if runtime.fatal is not None:
                    raise runtime.fatal
                return output
        finally:
            runtime.close()

    def _run(self, job_id):
        worker_id = None
        try:
            from tools import run_qwen_caption_flow_benchmark as runner
            row = self._row(job_id)
            actor = Actor(**json.loads(row["actor_json"]))
            body = CaptionJobRequest(**json.loads(row["request_json"]))
            history = json.loads(row["attempts_json"])
            identity = row["runtime_identity"]
            self._update(job_id, state="running", error=None)
            output = None
            if body.provider == "openai" and history and history[-1].get("evidence"):
                with self.tools.artifacts.open(history[-1]["evidence"]["sha256"]) as stream:
                    output = json.load(stream)
            for attempt in range(len(history) + 1 if body.provider == "openai" else 1, body.settings.attempts + 1):
                self._check(job_id, actor, body, identity)
                started = time.monotonic()
                if body.provider == "openai":
                    output = self._paid_attempt(job_id, actor, body, identity, len(history) + 1)
                    status = {"state": "completed"}
                while body.provider == "local_qwen":
                    self._check(job_id, actor, body, identity)
                    try:
                        submitted = self.tools.submit(actor, self._tool_request(body))
                        worker_id = submitted["id"]
                        self._update(job_id, state="running", worker_id=worker_id)
                        break
                    except RuntimeRejected as exc:
                        if exc.code != "runtime_automation_paused":
                            raise
                        self._update(job_id, state="paused")
                        self._wait(job_id, actor, body, identity, 0.25)
                while body.provider == "local_qwen":
                    self._check(job_id, actor, body, identity)
                    status = self.tools.status(actor, worker_id)
                    if status["state"] in TERMINAL:
                        break
                    self.stop.wait(0.1)
                record = {"attempt": len(history) + 1, "worker_id": worker_id,
                    "elapsed_seconds": time.monotonic() - started, "status": status["state"],
                    "error": status.get("error"), "final_status": "failed_attempt"}
                if status["state"] in {"stale", "cancelled", "interrupted"}:
                    raise RuntimeRejected(status.get("error") or "runtime_authority_changed")
                if status["state"] == "completed":
                    if body.provider == "local_qwen":
                        output = self.tools.result(actor, worker_id)["result"]
                    record.update(status=output.get("status", "error"), caption_quality=output.get("caption_quality"),
                        recovery_events=output.get("response", {}).get("recovery_events", []))
                    io_events = output.get("response", {}).get("hosted_evidence", {}).get("model_io", [])
                    record["qwen_caption_io"] = runner.summarize_qwen_caption_io_event_lines(
                        [json.dumps(event, allow_nan=False) for event in io_events])
                    record["quality_failures"] = runner.caption_quality_failures(record["caption_quality"])
                    succeeded = record["status"] in {"ok", "preview_only"} and (not record["quality_failures"] or body.settings.continue_on_quality_failures)
                else:
                    succeeded = False
                record["final_status"] = "ok" if succeeded else "failed_attempt"
                if output is not None and status["state"] == "completed":
                    with self.tools.artifacts.exclusive():
                        reference = self.tools.artifacts.put_bytes(json.dumps(output, allow_nan=False).encode())
                        with self.store.transaction() as con:
                            self.store._actor(con, actor)
                            self.store._guard(con, actor, MutationContext(body.image_id, body.expected_version, body.generation, body.editor_id, "caption-attempt"), renew=False)
                            con.execute("INSERT OR IGNORE INTO hosted_artifact_references VALUES ('caption_job',?,?,?,?)", (job_id, reference.sha256, reference.size_bytes, self.store.clock()))
                            con.execute("DELETE FROM hosted_orphan_observations WHERE sha256=?", (reference.sha256,))
                        record["evidence"] = reference.as_dict()
                history.append(record)
                self._update(job_id, attempts_json=json.dumps(history, allow_nan=False))
                if succeeded:
                    break
                if attempt < body.settings.attempts:
                    self._update(job_id, state="retrying")
                    self._wait(job_id, actor, body, identity, body.settings.cooldown_after_crash_seconds)
            failed = not history or history[-1]["final_status"] != "ok"
            health = self._health(history, body.settings, failed, bool(history and history[-1].get("quality_failures")))
            self._check(job_id, actor, body, identity)
            response = (output or {}).get("response", {})
            image = self.store.image(actor, body.image_id)
            record = {key: response[key] for key in ("caption", "used_counts", "used_boxes", "truncated", "recovery_events", "generated_qa_pairs", "generated_qa_pair_count", "generated_qa_target_pair_count", "generated_qa_warning") if key in response}
            record.update(image_name=image["filename"], caption_quality=(output or {}).get("caption_quality"))
            if "generated_qa_pairs" in record:
                from services.hosted.instruction_review import bind_machine_validated_pairs
                record["generated_qa_pairs"] = bind_machine_validated_pairs(record["generated_qa_pairs"])
            result = {"total_cases": 1, "processed": 1, "failed": int(failed), "attempts_seen": len(history),
                "saved_text_labels": 0, "captions": [record] if record.get("caption") else [],
                "latest_caption": record if record.get("caption") else None, "health": health,
                "preview": (output or {}).get("preview"), "degraded_rates": health["rates"]}
            state = "failed" if failed or health["status"] != "ok" else "completed"
            error = "caption_attempts_failed" if failed else "caption_health_gate_failed" if state == "failed" else None
            # Retain raw model evidence before exposing the completed result.
            # Actor/lease/version validation is repeated inside the reference
            # commit; no annotation row is changed by a caption job.
            with self.tools.artifacts.exclusive():
                evidence = self.tools.artifacts.put_bytes(json.dumps({"attempts": history, "output": output}, allow_nan=False).encode())
                with self.store.transaction() as con:
                    self.store._actor(con, actor)
                    self.store._guard(con, actor, MutationContext(body.image_id, body.expected_version, body.generation, body.editor_id, "caption-result"), renew=False)
                    cancelled = con.execute("SELECT cancel_requested FROM hosted_caption_jobs WHERE id=?", (job_id,)).fetchone()[0]
                    if cancelled or self.stop.is_set():
                        raise RuntimeRejected("caption_job_cancelled" if cancelled else "caption_control_stopped")
                    con.execute("INSERT OR IGNORE INTO hosted_artifact_references VALUES ('caption_job',?,?,?,?)", (job_id, evidence.sha256, evidence.size_bytes, self.store.clock()))
                    con.execute("DELETE FROM hosted_orphan_observations WHERE sha256=?", (evidence.sha256,))
                    result["evidence"] = evidence.as_dict()
                    con.execute("UPDATE hosted_caption_jobs SET state=?,result_json=?,error=?,updated_at=? WHERE id=?", (state, json.dumps(result, allow_nan=False), error, self.store.clock(), job_id))
        except Exception as exc:
            code = exc.code if isinstance(exc, (DomainError, RuntimeRejected)) else "caption_job_failed"
            state = "cancelled" if code == "caption_job_cancelled" else "interrupted" if code == "caption_control_stopped" else "failed"
            self._update(job_id, state=state, error=code)
            if worker_id:
                try:
                    self.tools.cancel(actor, worker_id)
                except Exception:
                    pass
        finally:
            with self.lock:
                self.threads.pop(job_id, None)
                # A resume can commit after the old attempt becomes terminal
                # but before its thread finishes cancelling its last worker.
                if not self.stop.is_set() and self._row(job_id)["state"] == "queued":
                    self._launch(job_id)
                elif not self.stop.is_set():
                    current = self._row(job_id)
                    attempts = json.loads(current["attempts_json"])
                    last_error = attempts[-1].get("error") if attempts else current["error"]
                    if current["state"] in {"failed", "interrupted"} and last_error in {"worker_disconnected", "worker_timeout", "worker_failed"}:
                        self._automatic_resume(job_id)

    def status(self, actor, job_id):
        self._recover()
        row = self._row(job_id, actor)
        body = CaptionJobRequest(**json.loads(row["request_json"]))
        result = json.loads(row["result_json"]) if row["result_json"] else {"total_cases": 1, "processed": 0, "attempts_seen": len(json.loads(row["attempts_json"]))}
        if row["result_json"]:
            if self._validate(actor, body) != row["runtime_identity"] or self._validate(Actor(**json.loads(row["actor_json"])), body) != row["runtime_identity"]:
                raise DomainError("caption_result_context_changed", 409)
        return {"job_id": job_id, "status": row["state"], "error": row["error"], "result": result,
            "auto_resume_count": row["auto_resumes"], "max_auto_resumes": body.settings.max_auto_resumes,
            "request": body.model_dump(exclude={"operation_id"}),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "context": {key: getattr(body, key) for key in ("dataset_id", "image_id", "expected_version", "generation", "editor_id")}}

    def list(self, actor):
        self._recover()
        with self.store.read_connection() as con:
            self.store._actor(con, actor)
            rows = con.execute("SELECT id AS job_id,state AS status,error,created_at,updated_at,request_json FROM hosted_caption_jobs WHERE actor_id=? ORDER BY created_at DESC LIMIT 100", (actor.user_id,)).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                request = json.loads(item.pop("request_json"))
                item["request"] = {key: request[key] for key in ("dataset_id", "image_id")}
                result.append(item)
            return result

    def cancel(self, actor, job_id, operation_id):
        row = self._row(job_id, actor)
        def change(con):
            con.execute("UPDATE hosted_caption_jobs SET cancel_requested=1,state=CASE WHEN state='waiting_for_editor' THEN 'cancelled' ELSE state END,updated_at=? WHERE id=?", (self.store.clock(), job_id))
            return {"job_id": job_id}
        self.store._command(actor, operation_id, "cancel_caption_job", {"job_id": job_id}, change)
        if row["worker_id"] and row["state"] in {"queued", "running", "paused", "retrying"}:
            try:
                self.tools.cancel(actor, row["worker_id"])
            except RuntimeRejected as exc:
                if exc.code != "runtime_request_not_found":
                    raise
        return self.status(actor, job_id)

    def resume(self, actor, job_id, body, *, automatic=False, generation_transition=None):
        self._recover()
        row = self._row(job_id, actor)
        requested = body.model_dump()
        previous = self.store.raw_receipt(actor, body.operation_id)
        if not previous:
            original = CaptionJobRequest(**json.loads(row["request_json"]))
            # Missing overrides retain this job's guidance across reconnects.
            caption = {key: original.caption_request[key] for key in ("user_prompt", "labelmap_glossary") if key in original.caption_request}
            body = body.model_copy(update={"caption_request": {**caption, **body.caption_request}})
            same_generation = original.generation == body.generation
            acquired_generation = automatic and generation_transition == (original.generation, body.generation) and body.generation == original.generation + 1
            if not (same_generation or acquired_generation) or any(getattr(original, key) != getattr(body, key) for key in ("dataset_id", "image_id", "expected_version", "caption_request", "settings", "provider", "provider_options")):
                raise DomainError("caption_resume_input_changed", 409)
            if row["state"] not in {"failed", "interrupted", "cancelled", "waiting_for_editor"}:
                raise DomainError("caption_job_not_resumable", 409)
            if self._validate(actor, body) != row["runtime_identity"]:
                raise DomainError("caption_resume_model_changed", 409)
        changed = False
        def change(con):
            nonlocal changed
            current = con.execute("SELECT state,cancel_requested,auto_resumes FROM hosted_caption_jobs WHERE id=?", (job_id,)).fetchone()
            if current[0] not in {"failed", "interrupted", "cancelled", "waiting_for_editor"}:
                raise DomainError("caption_job_not_resumable", 409)
            if automatic and (current[1] or not body.settings.set_and_forget or current[2] >= body.settings.max_auto_resumes):
                raise DomainError("caption_auto_resume_unavailable", 409)
            changed = True
            con.execute("UPDATE hosted_caption_jobs SET actor_json=?,request_json=?,state='queued',worker_id=NULL,result_json=NULL,error=NULL,cancel_requested=0,auto_resumes=auto_resumes+?,updated_at=? WHERE id=?",
                (json.dumps(asdict(actor)), body.model_dump_json(), int(automatic), self.store.clock(), job_id))
            return {"job_id": job_id}
        self.store._command(actor, body.operation_id, "resume_caption_job", {"job_id": job_id, "request": requested, "automatic": automatic}, change)
        if changed:
            self._launch(job_id)
        return self.status(actor, job_id)

    def _automatic_resume(self, job_id, *, actor=None, editor_id=None, generation_transition=None):
        row = self._row(job_id)
        body = CaptionJobRequest(**json.loads(row["request_json"]))
        if row["cancel_requested"] or not body.settings.set_and_forget or row["auto_resumes"] >= body.settings.max_auto_resumes:
            return None
        captured_actor = Actor(**json.loads(row["actor_json"]))
        if actor is not None and actor.user_id != captured_actor.user_id:
            return None
        actor = actor or captured_actor
        body.operation_id = "caption-auto-" + job_id + "-" + str(row["auto_resumes"] + 1)
        if editor_id is not None:
            body.editor_id = editor_id
        if generation_transition is not None:
            if body.generation != generation_transition[0]:
                self._update(job_id, state="failed", error="ownership_changed")
                return None
            body.generation = generation_transition[1]
        try:
            return self.resume(actor, job_id, body, automatic=True, generation_transition=generation_transition)
        except (DomainError, RuntimeRejected) as exc:
            if exc.code in {"caption_job_not_resumable", "caption_auto_resume_unavailable"}:
                return None
            # No lease is acquired or renewed here. A normal editor acquire
            # must succeed before a waiting job can use its fresh authority.
            if exc.code in {"caption_resume_model_changed", "annotation_version_changed", "ownership_changed", "image_not_found", "image_not_editable"}:
                self._update(job_id, state="failed", error=exc.code)
            else:
                self._update(job_id, state="waiting_for_editor", error=exc.code)
            return None

    def resume_waiting(self, actor, image_id, editor_id, *, previous_generation=None, generation=None, expected_version=None):
        """Call only after the ordinary acquire endpoint has granted a lease."""
        try:
            return self._resume_waiting(actor, image_id, editor_id, previous_generation=previous_generation,
                generation=generation, expected_version=expected_version)
        except Exception:
            # Lease acquisition already committed. Recovery is auxiliary and
            # must never turn that successful edit admission into an error.
            logging.getLogger(__name__).exception("Caption recovery after editor acquisition failed")
            return []

    def _resume_waiting(self, actor, image_id, editor_id, *, previous_generation=None, generation=None, expected_version=None):
        self._recover()
        with self.store.read_connection() as con:
            self.store._actor(con, actor)
            rows = con.execute("SELECT id,request_json FROM hosted_caption_jobs WHERE actor_id=? AND state='waiting_for_editor'", (actor.user_id,)).fetchall()
        resumed = []
        for row in rows:
            request = json.loads(row["request_json"])
            if request["image_id"] == image_id:
                if expected_version is not None and request["expected_version"] != expected_version:
                    self._update(row["id"], state="failed", error="annotation_version_changed")
                    continue
                transition = (previous_generation, generation) if previous_generation is not None and generation is not None else None
                try:
                    value = self._automatic_resume(row["id"], actor=actor, editor_id=editor_id, generation_transition=transition)
                except Exception:
                    logging.getLogger(__name__).exception("Caption recovery failed for job %s", row["id"])
                    try:
                        self._update(row["id"], state="failed", error="caption_recovery_failed")
                    except Exception:
                        logging.getLogger(__name__).exception("Could not record caption recovery failure")
                    continue
                if value:
                    resumed.append(value["job_id"])
        return resumed

    def close(self):
        self.stop.set()
        for thread in list(self.threads.values()):
            thread.join(timeout=2)
