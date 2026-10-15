"""Durable OpenAI Batch caption+QA jobs over canonical hosted images.

The existing Batch prompt builder and collector define the recipe. This module
owns only authority, immutable transport, provider submission and accounting.
It never imports the legacy API or reads a caller-supplied path/credential.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import threading
import time
from types import FunctionType, SimpleNamespace
from typing import Literal
import uuid

import httpx
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from services.hosted.caption_batch import canonical_case
from services.hosted.caption_case import CaptionSettings
from services.hosted.errors import DomainError
from services.hosted.instruction_review import bind_machine_validated_pairs
from services.hosted.providers import SnapshotAuthority
from services.hosted.types import Actor, MutationContext

MAX_BATCH_INPUT_BYTES = 200 * 1024 * 1024


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


class BatchOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1, max_length=160)
    detail: Literal["low", "high", "auto", "original"] = "high"
    max_output_tokens: int = Field(default=8192, ge=16, le=1000000)
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "high"


class BatchImageContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_id: str
    expected_version: str
    generation: int = Field(ge=0)
    editor_id: str = Field(min_length=8, max_length=128)


class PaidBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    operation_id: str = Field(min_length=8, max_length=128)
    snapshot_id: str | None = None
    image_context: BatchImageContext | None = None
    image_ids: list[str] | None = Field(default=None, max_length=50000)
    provider_options: BatchOptions
    caption_request: dict = Field(default_factory=dict)
    settings: CaptionSettings = Field(default_factory=CaptionSettings)
    catchup_job_id: str | None = None


class PaidBatchCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: str = Field(min_length=8, max_length=128)


class PaidBatchResume(PaidBatchCommand):
    image_context: BatchImageContext | None = None


class PaidBatchArchive(PaidBatchCommand):
    archived: bool = True


class RecoverBatch(PaidBatchCommand):
    remote_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,200}$")


class PaidBatchApply(PaidBatchCommand):
    image_id: str
    expected_version: str
    generation: int = Field(ge=0)
    editor_id: str = Field(min_length=8, max_length=128)
    proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    replace_primary: bool = True
    include_generated_qa: bool = True


class BatchTransport:
    """Fixed-host bounded transport. Mutating calls have no HTTP retries."""
    def request(self, secret, method, path, *, body=None, upload=None):
        if not re.fullmatch(r"/(?:files(?:/[A-Za-z0-9_-]+(?:/content)?)?|batches(?:/[A-Za-z0-9_-]+(?:/cancel)?)?)", path):
            raise DomainError("paid_batch_path_invalid", 500)
        kwargs = {"headers": {"Authorization": "Bearer " + secret}}
        if upload is not None:
            name, data, purpose = upload
            kwargs.update(data={"purpose": purpose}, files={"file": (name, data, "application/octet-stream")})
        elif body is not None:
            kwargs["json"] = body
        limit = 210 * 1024 * 1024 if path.endswith("/content") else 4 * 1024 * 1024
        with httpx.Client(timeout=180, trust_env=False, follow_redirects=False) as client:
            with client.stream(method, "https://api.openai.com/v1" + path, **kwargs) as response:
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > limit:
                        raise DomainError("paid_batch_response_too_large", 502)
                    chunks.append(chunk)
                if response.status_code not in {200, 201}:
                    raise DomainError("paid_batch_provider_request_failed", 502)
                raw = b"".join(chunks)
                return raw if path.endswith("/content") else json.loads(raw)


def native_line(image, classes, request, *, file_id="hosted-pinned-image", catchup=None):
    """Bind canonical hints into the exact native Batch recipe, without globals."""
    from tools import run_openai_caption_batch_smoke as native
    case = canonical_case(image, classes)
    case["image_path"] = image["filename"]
    if catchup:
        case.update(catchup)
    with tempfile.TemporaryDirectory(prefix="hosted-batch-prompt-") as temporary:
        request_path = Path(temporary) / "request.json"
        request_path.write_text(_json(request.caption_request))
        options, settings = request.provider_options, request.settings
        args = SimpleNamespace(**settings.model_dump(), model=options.model, image_detail=options.detail,
            reasoning_effort=options.reasoning_effort, max_output_tokens=options.max_output_tokens,
            max_boxes=int(request.caption_request.get("max_boxes") or 0), request_json=request_path,
            qa_count=settings.subcaptions_per_image if settings.instruction_dataset else 0)
        bindings = dict(native.build_batch_prompt_context.__globals__)
        bindings["_label_hints_for_case"] = lambda supplied_case, dataset_root: supplied_case["label_hints"]
        context = FunctionType(native.build_batch_prompt_context.__code__, bindings)
        bindings["build_batch_prompt_context"] = context
        build = FunctionType(native.build_batch_line.__code__, bindings)
        return case, build(case=case, file_id=file_id, dataset_root=Path(temporary), args=args)


class PaidBatches:
    def __init__(self, tools, *, transport=None):
        self.tools, self.store = tools, tools.store
        self.transport = transport or BatchTransport()
        self.stop, self.lock, self.threads = threading.Event(), threading.RLock(), {}
        self.recovered = False
        with self.store.transaction() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS hosted_paid_batches(
                id TEXT PRIMARY KEY,actor_id TEXT NOT NULL,actor_json TEXT NOT NULL,request_json TEXT NOT NULL,
                manifest_json TEXT NOT NULL,pricing_json TEXT NOT NULL,state TEXT NOT NULL,
                input_file_id TEXT,remote_id TEXT,remote_json TEXT,result_json TEXT,error TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0,archived INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,updated_at REAL NOT NULL)""")
            con.execute("""CREATE TABLE IF NOT EXISTS hosted_paid_batch_rows(
                job_id TEXT NOT NULL,image_id TEXT NOT NULL,call_id TEXT,request_json TEXT NOT NULL,
                upload_id TEXT,output_json TEXT,PRIMARY KEY(job_id,image_id))""")

    @property
    def gateway(self):
        if self.tools.provider_gateway is None:
            raise DomainError("paid_provider_not_configured", 503)
        return self.tools.provider_gateway

    def catalog(self, actor, dataset_id):
        allowed = self.gateway.catalog(actor, dataset_id)
        result = []
        for model in allowed["models"]:
            price = self.gateway.pricing.get(model["id"], {}).get("batch")
            if isinstance(price, dict) and self._price_valid(price):
                result.append({"id": model["id"], "max_output_tokens": price["max_output_tokens"],
                    "image_details": model.get("image_details", ["low", "high", "auto"])})
        return {"models": result, "recipe": "native_combined_caption_qa", "completion_window": "24h"}

    @staticmethod
    def _price_valid(price):
        return all(type(price.get(key)) is int and price[key] > 0 for key in (
            "input_microdollars_per_million", "cached_input_microdollars_per_million",
            "output_microdollars_per_million", "input_token_ceiling", "max_output_tokens"))

    def _row(self, actor, job_id, *, monitoring=False):
        with self.store.read_connection() as con:
            if actor:
                self.store._actor(con, actor)
            fields = "id,actor_id,actor_json,request_json,state,input_file_id,remote_id,remote_json,cancel_requested" if monitoring else "*"
            row = con.execute("SELECT " + fields + " FROM hosted_paid_batches WHERE id=?", (job_id,)).fetchone()
            if row is None or actor and row["actor_id"] != actor.user_id:
                raise DomainError("paid_batch_not_found", 404)
            return dict(row)

    def _update(self, job_id, **values):
        with self.store.transaction() as con:
            con.execute("UPDATE hosted_paid_batches SET " + ",".join(key + "=?" for key in values) + ",updated_at=? WHERE id=?",
                (*values.values(), self.store.clock(), job_id))

    def _rows(self, job_id):
        with self.store.read_connection() as con:
            return [dict(row) for row in con.execute("SELECT * FROM hosted_paid_batch_rows WHERE job_id=? ORDER BY image_id", (job_id,))]

    def _iter_rows(self, job_id):
        # Do not hold a read transaction across provider I/O or load every
        # retained prompt into the control process at once.
        after = ""
        while True:
            with self.store.read_connection() as con:
                rows = [dict(row) for row in con.execute("SELECT * FROM hosted_paid_batch_rows WHERE job_id=? AND image_id>? ORDER BY image_id LIMIT 16", (job_id, after))]
            if not rows:
                return
            yield from rows
            after = rows[-1]["image_id"]

    def _retain(self, job_id, data):
        return self._retain_stream(job_id, io.BytesIO(data))

    def _retain_stream(self, job_id, stream):
        with self.tools.artifacts.exclusive():
            reference = self.tools.artifacts.put_stream(stream)
            with self.store.transaction() as con:
                con.execute("INSERT OR IGNORE INTO hosted_artifact_references VALUES ('paid_batch',?,?,?,?)",
                    (job_id, reference.sha256, reference.size_bytes, self.store.clock()))
                con.execute("DELETE FROM hosted_orphan_observations WHERE sha256=?", (reference.sha256,))
            return reference.as_dict()

    def _capture(self, actor, body):
        if bool(body.snapshot_id) == bool(body.image_context):
            raise DomainError("paid_batch_source_required", 422)
        from services.hosted.tool_api import _parameters_for, qwen_guidance
        if set(body.caption_request) - _parameters_for("qwen_caption"):
            raise DomainError("tool_parameter_not_allowed", 422)
        if body.snapshot_id:
            self.tools.auth.require_admin(actor)
            manifest = self.store.get_snapshot(actor, body.snapshot_id)["manifest"]
            if manifest["dataset_id"] != body.dataset_id:
                raise DomainError("tool_dataset_mismatch", 409)
            if body.image_ids is not None:
                selected = set(body.image_ids)
                if len(selected) != len(body.image_ids) or not selected <= {image["id"] for image in manifest["images"]}:
                    raise DomainError("paid_batch_selection_invalid", 422)
                manifest = {**manifest, "images": [image for image in manifest["images"] if image["id"] in selected]}
        else:
            if body.image_ids is not None:
                raise DomainError("paid_batch_selection_invalid", 422)
            ctx = body.image_context
            image = self.store.authorize_context(actor, MutationContext(**ctx.model_dump(), operation_id=body.operation_id), write=True)
            if image["dataset_id"] != body.dataset_id:
                raise DomainError("tool_dataset_mismatch", 409)
            dataset = self.store.get_dataset(actor, body.dataset_id)
            manifest = {"dataset_id": body.dataset_id, "class_version": dataset["class_version"], "settings": dataset["settings"],
                "classes": dataset["classes"], "images": [{**image, "version_id": image["current_version"]}]}
        if not 1 <= len(manifest["images"]) <= 50000:
            raise DomainError("paid_batch_size_invalid", 422)
        body = body.model_copy(update={"caption_request": qwen_guidance(manifest.get("settings", {}), "qwen_caption", body.caption_request)})
        from models.schemas import QwenCaptionRequest
        QwenCaptionRequest(**body.caption_request, image_token="hosted-batch")
        permitted = next((item for item in self.catalog(actor, body.dataset_id)["models"] if item["id"] == body.provider_options.model), None)
        if not permitted or body.provider_options.max_output_tokens > permitted["max_output_tokens"] or body.provider_options.detail not in permitted["image_details"]:
            raise DomainError("paid_batch_model_options_not_allowed", 403)
        price = self.gateway.pricing[body.provider_options.model]["batch"]
        return body, manifest, price

    def _authority(self, body, image):
        if body.snapshot_id:
            return SnapshotAuthority(body.snapshot_id, image["id"], image["version_id"])
        return MutationContext(**body.image_context.model_dump(), operation_id="batch-stage")

    def preview(self, actor, body):
        body, manifest, price = self._capture(actor, body)
        case, line = native_line(manifest["images"][0], manifest["classes"], body)
        return {"recipe": "native_combined_caption_qa", "image_count": len(manifest["images"]),
            "image_id": case["case_id"], "full_text": line["body"]["input"][0]["content"][0]["text"],
            "row": line, "maximum_reserved_microdollars": len(manifest["images"]) * self._maximum(price, body)}

    @staticmethod
    def _maximum(price, body):
        return (price["input_token_ceiling"] * price["input_microdollars_per_million"] +
            body.provider_options.max_output_tokens * price["output_microdollars_per_million"] + 999999) // 1000000

    def start(self, actor, body):
        self._recover()
        requested = body.model_dump()
        if self.store.receipt(actor, body.operation_id):
            def unreachable(con):
                raise AssertionError("receipt missing")
            receipt = self.store._command(actor, body.operation_id, "paid_batch_start", requested, unreachable)
            return self.status(actor, receipt["job_id"])
        body, manifest, price = self._capture(actor, body)
        catchups = {}
        if body.catchup_job_id:
            previous = self._row(actor, body.catchup_job_id)
            old = json.loads(previous["manifest_json"])
            if {image["id"]: (image["version_id"], image["media_hash"]) for image in old["images"]} != {image["id"]: (image["version_id"], image["media_hash"]) for image in manifest["images"]}:
                raise DomainError("paid_batch_catchup_source_changed", 409)
            for row in self._rows(body.catchup_job_id):
                prior = json.loads(row["output_json"] or "{}")
                if prior.get("caption"):
                    target = body.settings.subcaptions_per_image if body.settings.instruction_dataset else 0
                    pairs = prior.get("generated_qa_pairs", [])
                    missing = max(0, target - len(pairs))
                    if missing:
                        catchups[row["image_id"]] = {"_openai_batch_existing_caption": prior["caption"],
                            "_openai_batch_existing_qa_pairs": pairs, "_openai_batch_target_qa": missing,
                            "_openai_batch_total_qa_target": target, "_openai_batch_catchup_source_job_id": body.catchup_job_id}
            manifest = {**manifest, "images": [image for image in manifest["images"] if image["id"] in catchups]}
            if not manifest["images"]:
                raise DomainError("paid_batch_no_missing_qa", 409)
        # Generate native prompts outside the database write transaction and
        # spill rows to disk. Reject the provider input bound before reservation
        # or upload; the longest permitted remote file ID is included in it.
        with tempfile.TemporaryFile(mode="w+b") as prepared:
            encoded_bytes = 0
            for image in manifest["images"]:
                case, line = native_line(image, manifest["classes"], body, catchup=catchups.get(image["id"]))
                encoded_bytes += len((_json(line) + "\n").encode()) + 200
                if encoded_bytes > MAX_BATCH_INPUT_BYTES:
                    raise DomainError("paid_batch_input_too_large", 422)
                record = {"case": case, "line": line, "source_media_hash": image["media_hash"]}
                prepared.write((_json([image["id"], _json(record)]) + "\n").encode())
            prepared.seek(0)
            def create(con):
                if con.execute("SELECT COUNT(*) FROM hosted_paid_batches WHERE actor_id=? AND state IN ('queued','uploading','creating','paused','submitted','validating','in_progress','finalizing','cancelling','submission_unknown')", (actor.user_id,)).fetchone()[0] >= 8:
                    raise DomainError("paid_batch_queue_full", 429)
                job_id, now = uuid.uuid4().hex, self.store.clock()
                con.execute("INSERT INTO hosted_paid_batches(id,actor_id,actor_json,request_json,manifest_json,pricing_json,state,created_at,updated_at) VALUES(?,?,?,?,?,?,'queued',?,?)",
                    (job_id, actor.user_id, _json(asdict(actor)), body.model_dump_json(), _json(manifest), _json(price), now, now))
                con.executemany("INSERT INTO hosted_paid_batch_rows(job_id,image_id,request_json) VALUES(?,?,?)",
                    ((job_id, *json.loads(row)) for row in prepared))
                return {"job_id": job_id}
            receipt = self.store._command(actor, body.operation_id, "paid_batch_start", requested, create)
        self._launch(receipt["job_id"])
        return self.status(actor, receipt["job_id"])

    def _recover(self):
        with self.lock:
            if self.recovered:
                return
            self.tools._ensure_runtime()
            with self.store.transaction() as con:
                con.execute("UPDATE hosted_paid_batches SET state='submission_unknown',error='control_restarted_during_submission' WHERE state='creating'")
                con.execute("UPDATE hosted_paid_batches SET state='queued' WHERE state IN ('uploading','paused') AND remote_id IS NULL")
                # Provider-terminal status precedes local collection. A restart
                # in that gap must finish retention and billing automatically.
                ids = [row[0] for row in con.execute("SELECT id FROM hosted_paid_batches WHERE state='queued' OR (remote_id IS NOT NULL AND state!='collected')")]
            self.recovered = True
            for job_id in ids:
                self._launch(job_id)

    def _launch(self, job_id):
        with self.lock:
            if job_id in self.threads and self.threads[job_id].is_alive():
                return
            thread = threading.Thread(target=self._run, args=(job_id,), name="hosted-paid-batch", daemon=True)
            self.threads[job_id] = thread
            thread.start()

    def _key(self, job_id, *, admission):
        row = self._row(None, job_id, monitoring=True)
        actor, body = Actor(**json.loads(row["actor_json"])), PaidBatchRequest(**json.loads(row["request_json"]))
        with self.store.read_connection() as con:
            provider = con.execute("SELECT * FROM hosted_providers WHERE name='openai'").fetchone()
            if provider is None or not provider["credential"]:
                raise DomainError("paid_provider_not_configured", 503)
            if admission:
                self.store._actor(con, actor)
                if row["cancel_requested"]:
                    raise DomainError("paid_batch_cancelled", 409)
                if not provider["enabled"] or body.dataset_id not in json.loads(provider["allowed_datasets_json"]) or body.provider_options.model not in json.loads(provider["models_json"]):
                    raise DomainError("paid_provider_not_allowed", 403)
                # Each reservation guards its image, and the dispatch commit
                # checks every captured image together. Repeating that full
                # population scan for each upload would be quadratic.
            return provider["credential"]

    def _wait(self, job_id):
        while self.tools._runtime.snapshot()["phase"] != "available":
            self._key(job_id, admission=True)
            self._update(job_id, state="paused")
            if self.stop.wait(.1):
                raise DomainError("paid_batch_control_stopped", 409)
        if self.stop.is_set():
            raise DomainError("paid_batch_control_stopped", 409)

    def _run(self, job_id):
        try:
            row = self._row(None, job_id, monitoring=True)
            if row["remote_id"] is None:
                if row["state"] == "submission_unknown":
                    return
                self._submit(job_id)
            while not self.stop.is_set():
                status = self._poll(job_id)
                if status in {"completed", "failed", "expired", "cancelled"}:
                    self._collect(job_id)
                    return
                self.stop.wait(30)
        except Exception as exc:
            row = self._row(None, job_id, monitoring=True)
            code = exc.code if isinstance(exc, DomainError) else "paid_batch_transport_failed"
            state = "submission_unknown" if row["state"] == "creating" else "cancelled" if code == "paid_batch_cancelled" else "interrupted" if code == "paid_batch_control_stopped" else "failed"
            self._update(job_id, state=state, error=code)
            if row["remote_id"] is None and state != "submission_unknown":
                for item in self._iter_rows(job_id):
                    if item["call_id"]:
                        self.gateway.accounts.release_reserved(item["call_id"], code)
        finally:
            with self.lock:
                self.threads.pop(job_id, None)
                if not self.stop.is_set() and self._row(None, job_id, monitoring=True)["state"] == "queued":
                    self._launch(job_id)

    def _upload_rows(self, job_id, actor, body, manifest, price):
        images = {image["id"]: image for image in manifest["images"]}
        with tempfile.TemporaryFile(mode="w+b") as content:
            for item in self._iter_rows(job_id):
                self._wait(job_id)
                secret = self._key(job_id, admission=True)
                self._update(job_id, state="uploading", error=None)
                prepared = json.loads(item["request_json"])
                image = images[item["image_id"]]
                authority = self._authority(body, image)
                from dataclasses import replace
                authority = replace(authority, operation_id="batch-row-" + _digest([job_id, image["id"]]))
                fingerprint = _digest({"prepared": prepared, "pricing": price, "transport": "openai_batch"})
                call = self.gateway.accounts.reserve(actor, authority, provider="openai", model=body.provider_options.model,
                    maximum_microdollars=self._maximum(price, body), request_fingerprint=fingerprint, replay_settled_stage=True)
                if call["state"] != "reserved":
                    raise DomainError("paid_batch_row_outcome_unresolved", 409)
                with self.store.transaction() as con:
                    con.execute("UPDATE hosted_paid_batch_rows SET call_id=? WHERE job_id=? AND image_id=?", (call["id"], job_id, item["image_id"]))
                file_id = item["upload_id"]
                if not file_id:
                    with self.tools.artifacts.open(image["media_hash"]) as stream:
                        data = stream.read(40 * 1024 * 1024 + 1)
                    if len(data) > 40 * 1024 * 1024:
                        raise DomainError("paid_batch_image_too_large", 422)
                    uploaded = self.transport.request(secret, "POST", "/files", upload=(image["filename"], data, "vision"))
                    file_id = self._remote_id(uploaded.get("id"))
                    self._retain(job_id, _json({"image_id": image["id"], "upload": uploaded, "source_media_hash": image["media_hash"]}).encode())
                    with self.store.transaction() as con:
                        con.execute("UPDATE hosted_paid_batch_rows SET upload_id=? WHERE job_id=? AND image_id=?", (file_id, job_id, item["image_id"]))
                line = prepared["line"]
                line["body"]["input"][0]["content"][1]["file_id"] = file_id
                content.write((_json(line) + "\n").encode())
                if content.tell() > MAX_BATCH_INPUT_BYTES:
                    raise DomainError("paid_batch_input_too_large", 422)
            content.seek(0)
            return self._retain_stream(job_id, content)

    def _submit(self, job_id):
        row = self._row(None, job_id)
        actor, body = Actor(**json.loads(row["actor_json"])), PaidBatchRequest(**json.loads(row["request_json"]))
        manifest, price = json.loads(row["manifest_json"]), json.loads(row["pricing_json"])
        images = {image["id"]: image for image in manifest["images"]}
        input_ref = self._upload_rows(job_id, actor, body, manifest, price)
        self._wait(job_id)
        secret = self._key(job_id, admission=True)
        row = self._row(None, job_id)
        input_id = row["input_file_id"]
        if not input_id:
            with self.tools.artifacts.open(input_ref["sha256"]) as content:
                uploaded = self.transport.request(secret, "POST", "/files", upload=("batch-input.jsonl", content, "batch"))
            input_id = self._remote_id(uploaded.get("id"))
            self._update(job_id, input_file_id=input_id)
        # All per-row reservations and authority are admitted in one commit.
        while True:
            self._wait(job_id)
            with self.store.transaction() as con:
                self.store._actor(con, actor)
                current = con.execute("SELECT cancel_requested FROM hosted_paid_batches WHERE id=?", (job_id,)).fetchone()
                if current[0]:
                    raise DomainError("paid_batch_cancelled", 409)
                if self.tools._runtime.snapshot()["phase"] != "available":
                    continue
                provider = con.execute("SELECT * FROM hosted_providers WHERE name='openai'").fetchone()
                if not provider["enabled"]:
                    raise DomainError("paid_provider_disabled", 409)
                for item in con.execute("SELECT * FROM hosted_paid_batch_rows WHERE job_id=?", (job_id,)):
                    self.gateway.accounts.guard_authority(con, actor, self._authority(body, images[item["image_id"]]), renew=True)
                    call = con.execute("SELECT * FROM hosted_provider_calls WHERE id=?", (item["call_id"],)).fetchone()
                    if call["state"] != "reserved" or call["provider_revision"] != provider["revision"]:
                        raise DomainError("paid_batch_reservation_changed", 409)
                    con.execute("UPDATE hosted_provider_calls SET state='dispatched' WHERE id=?", (call["id"],))
                con.execute("UPDATE hosted_paid_batches SET state='creating',error=NULL WHERE id=?", (job_id,))
                secret = provider["credential"]
            break
        response = self.transport.request(secret, "POST", "/batches", body={"input_file_id": input_id,
            "endpoint": "/v1/responses", "completion_window": "24h",
            "metadata": {"hosted_job_id": job_id, "input_sha256": input_ref["sha256"]}})
        remote_id = self._remote_id(response.get("id"))
        self._retain(job_id, _json(response).encode())
        self._update(job_id, remote_id=remote_id, remote_json=_json(response), state=str(response.get("status") or "submitted"))

    @staticmethod
    def _remote_id(value):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", value):
            raise DomainError("paid_batch_remote_id_invalid", 502)
        return value

    def _poll(self, job_id):
        row = self._row(None, job_id, monitoring=True)
        remote_id = self._remote_id(row["remote_id"])
        secret = self._key(job_id, admission=False)
        if row["cancel_requested"] and row["state"] not in {"cancelling", "cancelled", "completed", "failed", "expired"}:
            response = self.transport.request(secret, "POST", "/batches/" + remote_id + "/cancel")
        else:
            response = self.transport.request(secret, "GET", "/batches/" + remote_id)
        if response.get("id") != remote_id or response.get("input_file_id") != row["input_file_id"]:
            raise DomainError("paid_batch_remote_identity_changed", 502)
        state = response.get("status")
        if state not in {"validating", "in_progress", "finalizing", "completed", "failed", "expired", "cancelling", "cancelled"}:
            raise DomainError("paid_batch_remote_state_invalid", 502)
        self._retain(job_id, _json(response).encode())
        self._update(job_id, state=state, remote_json=_json(response), error=None)
        return state

    def _collect(self, job_id):
        row = self._row(None, job_id)
        body, remote, price = PaidBatchRequest(**json.loads(row["request_json"])), json.loads(row["remote_json"]), json.loads(row["pricing_json"])
        if remote.get("status") not in {"completed", "failed", "expired", "cancelled"}:
            raise DomainError("paid_batch_not_finished", 409)
        secret, rows = self._key(job_id, admission=False), self._rows(job_id)
        by_id = {item["image_id"]: item for item in rows}
        raw_files, observed, raw_evidence = {}, {}, []
        for field, name in (("output_file_id", "batch_output.jsonl"), ("error_file_id", "batch_error.jsonl")):
            value = remote.get(field)
            data = self.transport.request(secret, "GET", "/files/" + self._remote_id(value) + "/content") if value else b""
            raw_evidence.append(self._retain(job_id, data))
            raw_files[name] = data
            for raw_line in data.splitlines():
                if not raw_line.strip():
                    continue
                entry = json.loads(raw_line)
                cid = entry.get("custom_id")
                if cid not in by_id or cid in observed:
                    raise DomainError("paid_batch_output_identity_invalid", 502)
                observed[cid] = entry
        from tools import run_openai_caption_batch_smoke as native
        with tempfile.TemporaryDirectory(prefix="hosted-batch-collect-") as temporary:
            directory = Path(temporary)
            for name, data in raw_files.items():
                (directory / name).write_bytes(data)
            cases = [json.loads(item["request_json"])["case"] for item in rows]
            settings = body.settings
            summary = native.collect_results(cases=cases, output_dir=directory,
                target_qa=settings.subcaptions_per_image if settings.instruction_dataset else 0,
                answer_format=settings.answer_format, imposed_questions=settings.instruction_qa_imposed_questions,
                max_output_tokens=body.provider_options.max_output_tokens)
            collected = {}
            for name in ("captions.jsonl", "incomplete_captions.jsonl", "results.jsonl"):
                path = directory / name
                if path.exists():
                    data = path.read_bytes()
                    self._retain(job_id, data)
                    for value in native.read_jsonl(path):
                        cid = value["case_id"]
                        collected[cid] = {**collected.get(cid, {}), **value}
            for item in rows:
                entry = observed.get(item["image_id"], {})
                response = entry.get("response") or {}
                usage = (response.get("body") or {}).get("usage") or {}
                inputs, outputs, cached = usage.get("input_tokens"), usage.get("output_tokens"), (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
                valid = all(type(number) is int and number >= 0 for number in (inputs, outputs, cached)) and cached <= inputs
                charge = ((inputs - cached) * price["input_microdollars_per_million"] + cached * price["cached_input_microdollars_per_million"] + outputs * price["output_microdollars_per_million"] + 999999) // 1000000 if valid else None
                with self.store.read_connection() as con:
                    call = con.execute("SELECT state FROM hosted_provider_calls WHERE id=?", (item["call_id"],)).fetchone()
                if call["state"] in {"dispatched", "unknown"}:
                    self.gateway.accounts.settle(item["call_id"], actual_microdollars=charge,
                        provider_request_id=str((response.get("body") or {}).get("id") or ""))
                value = collected.get(item["image_id"], {"case_id": item["image_id"], "final_status": "failed", "failure_reason": "missing_batch_result"})
                value["call_id"] = item["call_id"]
                value["billing_resolved"] = valid
                value["evidence"] = raw_evidence
                with self.store.transaction() as con:
                    con.execute("UPDATE hosted_paid_batch_rows SET output_json=? WHERE job_id=? AND image_id=?", (_json(value), job_id, item["image_id"]))
        summary.update(saved_text_labels=0, transport="openai_batch", recipe="native_combined_caption_qa", raw_evidence=raw_evidence)
        evidence = self._retain(job_id, _json(summary).encode())
        self._update(job_id, state="collected", result_json=_json({**summary, "evidence": evidence}), error=None)

    def _visible(self, actor, row):
        manifest = json.loads(row["manifest_json"])
        with self.store.read_connection() as con:
            self.store._actor(con, actor)
            for image in manifest["images"]:
                self.store._can_view(actor, self.store._image(con, image["id"]))

    def status(self, actor, job_id):
        self._recover()
        row = self._row(actor, job_id)
        self._visible(actor, row)
        remote = json.loads(row["remote_json"] or "{}")
        return {"job_id": job_id, "status": row["state"], "error": row["error"], "openai_batch_id": row["remote_id"],
            "request_counts": remote.get("request_counts", {}), "request": json.loads(row["request_json"]),
            "result": json.loads(row["result_json"] or "{}"), "archived": bool(row["archived"]),
            "created_at": row["created_at"], "updated_at": row["updated_at"], "kind": "openai_caption_batch"}

    def list(self, actor, dataset_id=None, include_archived=False):
        self._recover()
        with self.store.read_connection() as con:
            self.store._actor(con, actor)
            ids = [row[0] for row in con.execute("SELECT id FROM hosted_paid_batches WHERE actor_id=? AND (archived=0 OR ?=1) ORDER BY created_at DESC LIMIT 100", (actor.user_id, int(include_archived)))]
        result = []
        for job_id in ids:
            try:
                value = self.status(actor, job_id)
            except DomainError as exc:
                if exc.status in {403, 404}:
                    continue
                raise
            if dataset_id is None or value["request"]["dataset_id"] == dataset_id:
                result.append(value)
        return result

    def poll(self, actor, job_id):
        row = self._row(actor, job_id)
        self._visible(actor, row)
        if row["remote_id"] and row["state"] != "collected":
            self._launch(job_id)
        return self.status(actor, job_id)

    def cancel(self, actor, job_id, operation_id):
        self._row(actor, job_id)
        def apply(con):
            con.execute("UPDATE hosted_paid_batches SET cancel_requested=1 WHERE id=?", (job_id,))
            return {"job_id": job_id}
        self.store._command(actor, operation_id, "paid_batch_cancel", {"job_id": job_id}, apply)
        self._launch(job_id)
        return self.status(actor, job_id)

    def archive(self, actor, job_id, operation_id, archived=True):
        self._row(actor, job_id)
        def apply(con):
            con.execute("UPDATE hosted_paid_batches SET archived=? WHERE id=?", (int(archived), job_id))
            return {"job_id": job_id}
        self.store._command(actor, operation_id, "paid_batch_archive", {"job_id": job_id, "archived": archived}, apply)
        return self.status(actor, job_id)

    def verify(self, actor, job_id):
        row = self._row(actor, job_id)
        self._visible(actor, row)
        manifest = json.loads(row["manifest_json"])
        changed = []
        with self.store.read_connection() as con:
            self.store._actor(con, actor)
            dataset = self.store._dataset(con, manifest["dataset_id"])
            for image in manifest["images"]:
                current = self.store._image(con, image["id"])
                archived = con.execute("SELECT 1 FROM hosted_archived_images WHERE image_id=?", (image["id"],)).fetchone()
                if archived or current["current_version"] != image["version_id"] or current["media_hash"] != image["media_hash"]:
                    changed.append(image["id"])
        classes_match = dataset["class_version"] == manifest["class_version"]
        matches = not changed and classes_match
        return {"ok": matches, "match_status": "exact_dataset_match" if matches else "same_dataset_id_image_mismatch" if changed else "label_snapshot_mismatch",
            "image_count": len(manifest["images"]), "changed_image_ids": changed, "classes_match": classes_match,
            "message": "All source image versions and classes still match." if matches else "Some source images or classes changed; affected proposals cannot be applied."}

    def resume(self, actor, job_id, body):
        row = self._row(actor, job_id)
        payload = {"job_id": job_id, **body.model_dump()}
        if self.store.receipt(actor, body.operation_id):
            def no_new_command(con):
                raise AssertionError("committed resume receipt disappeared")
            receipt = self.store._command(actor, body.operation_id, "paid_batch_resume", payload, no_new_command)
            try:
                return self.status(actor, job_id)
            except DomainError as exc:
                if exc.status not in {403, 404}:
                    raise
                return {**receipt, "operation_committed": True, "status": "access_changed"}
        original = PaidBatchRequest(**json.loads(row["request_json"]))
        if body.image_context:
            if original.image_context is None or (body.image_context.image_id, body.image_context.expected_version) != (original.image_context.image_id, original.image_context.expected_version):
                raise DomainError("paid_batch_source_changed", 409)
            original.image_context = body.image_context
        self._capture(actor, original)
        def change(con):
            current = con.execute("SELECT * FROM hosted_paid_batches WHERE id=?", (job_id,)).fetchone()
            if current["state"] in {"creating", "submission_unknown"}:
                raise DomainError("paid_batch_submission_unresolved", 409)
            if current["state"] not in {"failed", "interrupted", "cancelled"}:
                raise DomainError("paid_batch_not_resumable", 409)
            con.execute("UPDATE hosted_paid_batches SET actor_json=?,request_json=?,cancel_requested=0,error=NULL,state=? WHERE id=?",
                (_json(asdict(actor)), original.model_dump_json(), "submitted" if current["remote_id"] else "queued", job_id))
            return {"job_id": job_id}
        self.store._command(actor, body.operation_id, "paid_batch_resume", payload, change)
        self._launch(job_id)
        return self.status(actor, job_id)

    def recover_remote(self, actor, job_id, remote_id, operation_id):
        row = self._row(actor, job_id)
        if self.store.receipt(actor, operation_id):
            self.store._command(actor, operation_id, "paid_batch_recover", {"job_id": job_id, "remote_id": remote_id}, lambda con: None)
            return self.status(actor, job_id)
        if row["state"] != "submission_unknown" or not row["input_file_id"]:
            raise DomainError("paid_batch_recovery_not_needed", 409)
        response = self.transport.request(self._key(job_id, admission=False), "GET", "/batches/" + self._remote_id(remote_id))
        if response.get("input_file_id") != row["input_file_id"] or (response.get("metadata") or {}).get("hosted_job_id") != job_id or response.get("endpoint") != "/v1/responses":
            raise DomainError("paid_batch_recovery_identity_mismatch", 409)
        def change(con):
            con.execute("UPDATE hosted_paid_batches SET remote_id=?,remote_json=?,state='submitted',error=NULL WHERE id=?", (remote_id, _json(response), job_id))
            return {"job_id": job_id}
        self.store._command(actor, operation_id, "paid_batch_recover", {"job_id": job_id, "remote_id": remote_id}, change)
        self._launch(job_id)
        return self.status(actor, job_id)

    def proposals(self, actor, job_id, offset=0, limit=25):
        row = self._row(actor, job_id)
        self._visible(actor, row)
        if offset < 0 or not 1 <= limit <= 100:
            raise DomainError("caption_page_invalid", 422)
        manifest = json.loads(row["manifest_json"])
        images = {image["id"]: image for image in manifest["images"]}
        records = self._rows(job_id)
        result = []
        for item in records[offset:offset+limit]:
            data = json.loads(item["output_json"] or "{}")
            value = {"image_id": item["image_id"], "version_id": images[item["image_id"]]["version_id"],
                "class_version": manifest["class_version"], "caption": data.get("caption"),
                "generated_qa_pairs": bind_machine_validated_pairs(data.get("generated_qa_pairs", [])), "quality": data}
            value["proposal_sha256"] = _digest(value)
            result.append(value)
        return {"total": len(records), "offset": offset, "captions": result}

    def apply(self, actor, job_id, body):
        payload = {"job_id": job_id, **body.model_dump()}
        if self.store.receipt(actor, body.operation_id):
            def no_new_command(con):
                raise AssertionError("committed caption receipt disappeared")
            return self.store._command(actor, body.operation_id, "paid_batch_apply", payload, no_new_command)
        row = self._row(actor, job_id)
        manifest = json.loads(row["manifest_json"])
        image = next((item for item in manifest["images"] if item["id"] == body.image_id), None)
        if image is None or image["version_id"] != body.expected_version:
            raise DomainError("caption_proposal_source_changed", 409)
        records = self._rows(job_id)
        index = next(index for index, item in enumerate(records) if item["image_id"] == body.image_id)
        proposal = self.proposals(actor, job_id, index, 1)["captions"][0]
        if proposal["proposal_sha256"] != body.proposal_sha256 or not proposal["caption"]:
            raise DomainError("caption_proposal_changed", 409)
        request = json.loads(row["request_json"])
        quality = proposal["quality"]
        provenance = {"provider": "openai", "model_id": request["provider_options"]["model"],
            "runtime_platform": "openai_batch", "source_version": body.expected_version,
            "evidence": quality.get("evidence", []), "provider_call_id": quality.get("call_id"),
            "batch_output_status": quality.get("final_status"),
            **{key: quality[key] for key in ("caption_quality", "quality_failures", "used_counts", "used_boxes", "recovery_events") if key in quality}}
        context = MutationContext(body.image_id, body.expected_version, body.generation, body.editor_id, body.operation_id)
        def commit(con):
            current = self.store._guard(con, actor, context)
            dataset = self.store._dataset(con, current["dataset_id"])
            if dataset["class_version"] != manifest["class_version"]:
                raise DomainError("class_map_changed", 409)
            version = con.execute("SELECT * FROM hosted_annotation_versions WHERE id=?", (body.expected_version,)).fetchone()
            annotations = json.loads(version["annotations_json"])
            alternates = annotations.setdefault("alternate_captions", [])
            if not isinstance(alternates, list):
                raise DomainError("caption_alternates_format_invalid", 409)
            previous = str(annotations.get("caption") or "").strip()
            if body.replace_primary:
                if previous and previous != proposal["caption"]:
                    alternates.append({"caption": previous, "source": "previous_primary"})
                annotations["caption"] = proposal["caption"]
            alternates.append({"caption": proposal["caption"], "source": "hosted_paid_batch", "job_id": job_id,
                "proposal_sha256": body.proposal_sha256, **provenance,
                **({"generated_qa_pairs": proposal["generated_qa_pairs"]} if body.include_generated_qa else {})})
            return self.store._save(con, actor, context, annotations, complete=bool(version["complete"]), negative=bool(version["negative"]),
                trusted_instruction_pairs=proposal["generated_qa_pairs"] if body.include_generated_qa else ())
        return self.store._command(actor, body.operation_id, "paid_batch_apply", payload, commit)

    def close(self):
        self.stop.set()
        for thread in list(self.threads.values()):
            thread.join(timeout=2)
