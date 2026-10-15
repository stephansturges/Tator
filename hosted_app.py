"""Lightweight hosted control process. No import of the legacy model runtime.

Run ``uvicorn hosted_app:create_app --factory --host 127.0.0.1`` with
TATOR_HOSTED_STATE_DIR and TATOR_HOSTED_ORIGIN configured. Development callers
must explicitly construct a test app; production never disables authentication.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import asynccontextmanager
import fcntl
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.concurrency import run_in_threadpool

from services.hosted.artifacts import ArtifactError, ArtifactStore
from services.hosted.auth import AuthService
from services.hosted.errors import DomainError
from services.hosted.store import HostedStore
from services.hosted.operations_api import install_operations_api
from services.hosted.http_boundary import BodyLimitMiddleware
from services.hosted.tool_api import install_tool_api
from services.hosted.provider_api import install_provider_api
from services.hosted.analysis_api import install_analysis_api
from services.hosted.types import Actor, MutationContext

WEB_ROOT = Path(__file__).resolve().parent / "ybat-master"
COOKIE = "tator_session"


def public_identity(identity: Actor) -> dict:
    return {"user_id": identity.user_id, "username": identity.username, "role": identity.role}


class Credentials(BaseModel):
    username: str = Field(max_length=128)
    password: str = Field(max_length=1024)


class Redeem(BaseModel):
    token: str = Field(min_length=20, max_length=200)
    password: str = Field(min_length=12, max_length=1024)


class NewUser(BaseModel):
    username: str
    role: str = "annotator"


class UserChange(BaseModel):
    role: str
    disabled: bool
    expected_role: str
    expected_disabled: bool


class NewDataset(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    classes: list[str] = Field(default_factory=list, max_length=10000)
    operation_id: str = Field(min_length=8, max_length=128)


class DatasetGuidance(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    context: str = Field(max_length=32000)
    labelmap_glossary: str = Field(max_length=64000)
    expected_revision: int = Field(ge=0)
    operation_id: str = Field(min_length=8, max_length=128)


class Assignment(BaseModel):
    image_ids: list[str] = Field(min_length=1, max_length=10000)
    annotator_id: str
    operation_id: str = Field(min_length=8, max_length=128)
    expected: dict | None = None


class RecallPrecondition(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    expected_version: str = Field(min_length=1)
    generation: int = Field(ge=0)


class RecallImages(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    image_ids: list[str] = Field(min_length=1, max_length=50000)
    expected: dict[str, RecallPrecondition]
    reason: str = Field(min_length=1, max_length=10000)
    operation_id: str = Field(min_length=8, max_length=128)


class SplitPreview(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    image_ids: list[str] = Field(min_length=1, max_length=50000)
    target_split: Literal['train', 'val', 'test', 'unsplit']


class SplitAssignment(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    assignments: dict[str, Literal['train', 'val', 'test', 'unsplit']]
    expected_revision: int = Field(ge=0)
    expected: dict[str, RecallPrecondition]
    operation_id: str = Field(min_length=8, max_length=128)


class BulkReviewItem(BaseModel):
    image_id: str
    expected_version: str
    generation: int = Field(ge=0)
    operation_id: str = Field(min_length=8, max_length=128)


class BulkReview(BaseModel):
    operation_id: str = Field(min_length=8, max_length=128)
    editor_id: str = Field(min_length=8, max_length=128)
    images: list[BulkReviewItem] = Field(min_length=1, max_length=500)


class Editor(BaseModel):
    editor_id: str = Field(min_length=8, max_length=128)
    expected_version: str | None = None
    generation: int | None = Field(default=None, ge=0)
    takeover: bool = False


class Mutation(BaseModel):
    expected_version: str
    generation: int = Field(ge=0)
    editor_id: str = Field(min_length=8, max_length=128)
    operation_id: str = Field(min_length=8, max_length=128)

    def context(self, image_id: str) -> MutationContext:
        return MutationContext(image_id=image_id, **self.model_dump(include={"expected_version", "generation", "editor_id", "operation_id"}))


class Save(Mutation):
    annotations: dict
    complete: bool = False
    negative: bool = False


class Review(Mutation):
    feedback: str = Field(default="", max_length=10000)
    annotator_id: str | None = None


class ImportConflictChoice(Mutation):
    choice: str
    negative: bool = False


class ClassChange(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    operation_id: str = Field(min_length=8, max_length=128)
    expected_class_version: int = Field(ge=0)


class ClassRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=5000)
    operation_id: str = Field(min_length=8, max_length=128)


class ClassReview(BaseModel):
    approved: bool
    operation_id: str = Field(min_length=8, max_length=128)
    expected_class_version: int = Field(ge=0)
    feedback: str = Field(default="", max_length=10000)


def create_app(state_dir: str | Path | None = None, *, testing=False, origin: str | None = None) -> FastAPI:
    configured = state_dir or os.environ.get("TATOR_HOSTED_STATE_DIR")
    if not configured:
        raise RuntimeError("Set TATOR_HOSTED_STATE_DIR to a new dedicated hosted state directory.")
    origin = (origin or os.environ.get("TATOR_HOSTED_ORIGIN") or ("http://testserver" if testing else "")).rstrip("/")
    parsed = urlsplit(origin)
    if not parsed.hostname or parsed.path or parsed.query or parsed.fragment or parsed.username:
        raise RuntimeError("TATOR_HOSTED_ORIGIN must be an exact origin, without a path or credentials.")
    if not testing and parsed.scheme != "https":
        raise RuntimeError("Hosted production requires a configured HTTPS origin.")
    if not testing and sqlite3.sqlite_version_info < (3, 51, 3):
        raise RuntimeError("Hosted production requires SQLite 3.51.3 or later, including its WAL fix.")
    root = Path(configured).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not testing and root.stat().st_mode & 0o077:
        raise RuntimeError("The hosted state directory must be private (mode 0700).")
    store = HostedStore(root / "hosted.sqlite3")
    auth = AuthService(store)
    artifacts = ArtifactStore(root / "artifacts")
    @asynccontextmanager
    async def control_lifespan(_application):
        descriptor = os.open(root / "control.lock", os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another hosted control process owns this state directory. Use exactly one Uvicorn worker.") from error
            # Only the one live control process may reconcile interrupted calls.
            with store.transaction() as con:
                con.execute("UPDATE hosted_provider_calls SET state='unknown' WHERE state='dispatched'")
            yield
        finally:
            os.close(descriptor)

    app = FastAPI(title="Tator hosted authority", docs_url=None, redoc_url=None, openapi_url=None, lifespan=control_lifespan)
    app.state.store, app.state.auth, app.state.artifacts = store, auth, artifacts
    app.state.root = root
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[parsed.hostname])
    app.add_middleware(BodyLimitMiddleware)

    @app.exception_handler(DomainError)
    async def domain_error(_request, error):
        return JSONResponse({"error": error.code, "detail": error.detail}, status_code=error.status)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, error):
        # Pydantic's default errors include input values, sometimes the complete
        # request. Never echo credentials or annotation content in errors.
        return JSONResponse({"error": "invalid_request", "detail": "Check the request fields.",
                             "fields": [{"field": ".".join(map(str, item["loc"])), "type": item["type"]} for item in error.errors()]}, status_code=422)

    @app.exception_handler(ArtifactError)
    async def artifact_error(_request, _error):
        return JSONResponse({"error": "artifact_invalid", "detail": "The stored file failed validation."}, status_code=422)

    @app.middleware("http")
    async def boundary(request: Request, call_next):
        # Same-origin is enforced even for pre-login requests. No CORS wildcard
        # or caller-supplied editor identity grants a server permission.
        request_origin = request.headers.get("origin")
        if request_origin and request_origin != origin:
            return JSONResponse({"error": "origin_rejected"}, status_code=403)
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            return JSONResponse({"error": "invalid_content_length"}, status_code=400)
        if length > 260 * 1024 * 1024:
            return JSONResponse({"error": "request_too_large"}, status_code=413)
        if request.url.path.startswith("/api/hosted/") and request.url.path not in {"/api/hosted/login", "/api/hosted/redeem"}:
            # Authenticate before multipart/JSON parsers consume the request.
            # Domain mutations repeat the captured proof inside their transaction.
            try:
                request.state.hosted_actor = await run_in_threadpool(actor, request)
            except DomainError as error:
                return JSONResponse({"error": error.code, "detail": error.detail}, status_code=error.status)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Cache-Control"] = "no-store"
        return response

    def actor(request: Request) -> Actor:
        if hasattr(request.state, "hosted_actor"):
            return request.state.hosted_actor
        token = request.cookies.get(COOKIE, "")
        mutation = request.method not in ("GET", "HEAD", "OPTIONS")
        proof = request.headers.get("x-tator-csrf")
        identity = auth.authenticate(token, csrf=(proof or "") if mutation else proof)
        expected_instance = request.headers.get("x-tator-instance")
        if (mutation or expected_instance is not None) and expected_instance != auth.instance_id:
            raise DomainError("installation_changed", 409, "Reconnect before saving to this installation.")
        return identity

    def admin(identity: Actor = Depends(actor)) -> Actor:
        auth.require_admin(identity)
        return identity

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": "hosted"}

    @app.post("/api/hosted/login")
    def login(body: Credentials, request: Request):
        result = auth.login(body.username, body.password, request.client.host if request.client else "unknown")
        response = JSONResponse({"user": public_identity(result.actor), "csrf": result.csrf, "instance_id": auth.instance_id, "expires_at": result.expires_at})
        response.set_cookie(COOKIE, result.token, max_age=auth.session_seconds, secure=not testing, httponly=True, samesite="strict", path="/")
        return response

    @app.post("/api/hosted/redeem")
    def redeem(body: Redeem):
        auth.redeem(body.token, body.password)
        return {"ok": True}

    @app.get("/api/hosted/session")
    def session(request: Request, identity: Actor = Depends(actor)):
        return {"user": public_identity(identity), "csrf": auth.session_csrf(request.cookies[COOKIE]), "instance_id": auth.instance_id}

    @app.post("/api/hosted/logout")
    def logout(request: Request, _identity: Actor = Depends(actor)):
        auth.logout(request.cookies[COOKIE])
        response = JSONResponse({"ok": True})
        response.delete_cookie(COOKIE, path="/")
        return response

    @app.get("/api/hosted/users")
    def users(identity: Actor = Depends(admin)):
        return {"users": auth.list_users(identity)}

    @app.post("/api/hosted/users")
    def create_user(body: NewUser, identity: Actor = Depends(admin)):
        return auth.create_user(identity, body.username, body.role)

    @app.patch("/api/hosted/users/{user_id}")
    def change_user(user_id: str, body: UserChange, identity: Actor = Depends(admin)):
        return auth.set_user(identity, user_id, **body.model_dump())

    @app.post("/api/hosted/users/{user_id}/reset")
    def reset_user(user_id: str, identity: Actor = Depends(admin)):
        return auth.reset_link(identity, user_id)

    @app.get("/api/hosted/datasets")
    def datasets(identity: Actor = Depends(actor)):
        return {"datasets": store.list_datasets(identity)}

    @app.post("/api/hosted/datasets")
    def create_dataset(body: NewDataset, identity: Actor = Depends(admin)):
        return store.create_dataset(identity, body.name, body.classes, body.operation_id)

    @app.get("/api/hosted/datasets/{dataset_id}")
    def dataset(dataset_id: str, identity: Actor = Depends(actor)):
        return store.get_dataset(identity, dataset_id)

    @app.post("/api/hosted/datasets/{dataset_id}/guidance")
    def dataset_guidance(dataset_id: str, body: DatasetGuidance, identity: Actor = Depends(admin)):
        return store.patch_dataset_settings(identity, dataset_id,
            {"context": body.context, "labelmap_glossary": body.labelmap_glossary},
            body.expected_revision, body.operation_id)

    @app.post("/api/hosted/datasets/{dataset_id}/classes")
    def add_class(dataset_id: str, body: ClassChange, identity: Actor = Depends(admin)):
        return store.add_class(identity, dataset_id, **body.model_dump())

    @app.patch("/api/hosted/datasets/{dataset_id}/classes/{class_id}")
    def rename_class(dataset_id: str, class_id: str, body: ClassChange, identity: Actor = Depends(admin)):
        return store.rename_class(identity, dataset_id, class_id, **body.model_dump())

    @app.get("/api/hosted/class-requests")
    def class_requests(dataset_id: str | None = None, identity: Actor = Depends(actor)):
        return {"requests": store.list_class_requests(identity, dataset_id)}

    @app.post("/api/hosted/datasets/{dataset_id}/class-requests")
    def request_class(dataset_id: str, body: ClassRequest, identity: Actor = Depends(actor)):
        return store.request_class(identity, dataset_id, **body.model_dump())

    @app.post("/api/hosted/class-requests/{request_id}/resolve")
    def resolve_class_request(request_id: str, body: ClassReview, identity: Actor = Depends(admin)):
        return store.resolve_class_request(identity, request_id, **body.model_dump())

    @app.get("/api/hosted/submissions")
    def submissions(dataset_id: str | None = None, offset: int = 0, limit: int = 100, identity: Actor = Depends(actor)):
        return {"submissions": store.own_submission_status(identity, dataset_id, limit=limit, offset=offset)}

    @app.get("/api/hosted/images")
    def images(dataset_id: str | None = None, state: str | None = None, offset: int = 0, limit: int = 100, identity: Actor = Depends(actor)):
        if not 1 <= limit <= 500 or offset < 0:
            raise DomainError("invalid_pagination", 422)
        return {"images": store.list_images(identity, dataset_id=dataset_id, state=state, limit=limit, offset=offset)}

    @app.post("/api/hosted/datasets/{dataset_id}/incoming")
    def upload(dataset_id: str, file: UploadFile = File(...), annotations: str | None = Form(None),
               yolo_labels_json: str | None = Form(None), labelmap_names: str | None = Form(None),
               expected_class_version: int | None = Form(None), operation_id: str = Form(...), identity: Actor = Depends(admin)):
        if not 8 <= len(operation_id) <= 128 or len(annotations or "") > 16 * 1024 * 1024:
            raise DomainError("invalid_import", 422)
        if (yolo_labels_json is not None and (annotations is not None or labelmap_names is None or expected_class_version is None)
                or yolo_labels_json is None and (labelmap_names is not None or expected_class_version is not None)
                or len(yolo_labels_json or "") > 16 * 1024 * 1024 or len(labelmap_names or "") > 1024 * 1024):
            raise DomainError('invalid_yolo_import', 422)
        try:
            labels = json.loads(annotations) if annotations is not None else None
            names = json.loads(labelmap_names) if labelmap_names is not None else None
            yolo_labels = json.loads(yolo_labels_json) if yolo_labels_json is not None else None
            if yolo_labels_json is not None and not isinstance(yolo_labels, str):
                raise ValueError()
            if labels is not None and not isinstance(labels, dict):
                raise ValueError()
            with Image.open(file.file) as picture:
                width, height = picture.size
                if picture.format not in ("PNG", "JPEG", "WEBP", "TIFF", "BMP"):
                    raise ValueError()
                picture.verify()
            file.file.seek(0)
        except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
            raise DomainError("invalid_image_import", 422, "Select a supported image and valid annotations.") from None
        filename = str(file.filename or "image").replace("\\", "/").rsplit("/", 1)[-1]
        with artifacts.exclusive():
            ref = artifacts.put_stream(file.file, max_bytes=256 * 1024 * 1024)
            if yolo_labels is not None:
                return store.import_yolo_image(identity, dataset_id, ref.sha256, filename, yolo_labels, names,
                                              expected_class_version, operation_id, width=width, height=height, size_bytes=ref.size_bytes)
            return store.import_image(identity, dataset_id, ref.sha256, filename, labels, operation_id, width=width, height=height, size_bytes=ref.size_bytes, split='unsplit')

    @app.post("/api/hosted/datasets/{dataset_id}/assignments")
    def assign(dataset_id: str, body: Assignment, identity: Actor = Depends(admin)):
        return store.assign(identity, dataset_id, body.image_ids, body.annotator_id, body.operation_id, expected=body.expected)

    @app.post("/api/hosted/datasets/{dataset_id}/recall-images")
    def recall_images(dataset_id: str, body: RecallImages, identity: Actor = Depends(admin)):
        return store.recall_images(identity, dataset_id, body.image_ids,
                                   {key: value.model_dump() for key, value in body.expected.items()},
                                   body.reason, body.operation_id)

    @app.post('/api/hosted/datasets/{dataset_id}/splits/preview')
    def preview_splits(dataset_id: str, body: SplitPreview, identity: Actor = Depends(admin)):
        return store.preview_splits(identity, dataset_id, body.image_ids, body.target_split)

    @app.post('/api/hosted/datasets/{dataset_id}/splits')
    def set_splits(dataset_id: str, body: SplitAssignment, identity: Actor = Depends(admin)):
        return store.set_splits(identity, dataset_id, body.assignments, body.expected_revision,
                                {key: value.model_dump() for key, value in body.expected.items()}, body.operation_id)

    @app.post("/api/hosted/datasets/{dataset_id}/approve-images")
    def bulk_review(dataset_id: str, body: BulkReview, identity: Actor = Depends(admin)):
        if len({row.image_id for row in body.images}) != len(body.images) or len({row.operation_id for row in body.images}) != len(body.images):
            raise DomainError("invalid_bulk_targets", 422)
        results = []
        for row in body.images:
            ctx = MutationContext(row.image_id, row.expected_version, row.generation, body.editor_id, row.operation_id)
            try:
                results.append({"image_id": row.image_id, "status": "approved", "result": store.approve_from_review_queue(identity, ctx, dataset_id=dataset_id)})
            except DomainError as error:
                results.append({"image_id": row.image_id, "status": "rejected", "error": error.code, "detail": error.detail})
        return {"images": results}

    @app.get("/api/hosted/images/{image_id}")
    def image_record(image_id: str, identity: Actor = Depends(actor)):
        return store.image(identity, image_id)

    @app.get("/api/hosted/images/{image_id}/media")
    def media(image_id: str, identity: Actor = Depends(actor)):
        record = store.image(identity, image_id)
        digest = record["media_hash"]
        # Authorize by membership before resolving a content digest. Never expose
        # the artifact directory through a static-file mount.
        with artifacts.open(digest) as source:
            with Image.open(source) as picture:
                content_type = Image.MIME.get(picture.format, "application/octet-stream")
        def chunks():
            with artifacts.open(digest) as source:
                while chunk := source.read(1024 * 1024):
                    yield chunk
        return StreamingResponse(chunks(), media_type=content_type)

    @app.post("/api/hosted/images/{image_id}/editor")
    def acquire_editor(image_id: str, body: Editor, identity: Actor = Depends(actor)):
        acquired = store.acquire_editor(identity, image_id, body.editor_id, expected_version=body.expected_version, generation=body.generation, takeover=body.takeover)
        app.state.tools.caption_jobs.resume_waiting(identity, image_id, body.editor_id,
            previous_generation=body.generation, generation=acquired["generation"], expected_version=acquired["current_version"])
        return acquired

    @app.post("/api/hosted/images/{image_id}/save")
    def save(image_id: str, body: Save, identity: Actor = Depends(actor)):
        return store.save(identity, body.context(image_id), body.annotations, complete=body.complete, negative=body.negative)

    @app.post("/api/hosted/images/{image_id}/release")
    def release(image_id: str, body: Mutation, identity: Actor = Depends(actor)):
        return store.release_editor(identity, body.context(image_id))

    @app.post("/api/hosted/images/{image_id}/submit")
    def submit(image_id: str, body: Mutation, identity: Actor = Depends(actor)):
        return store.submit(identity, body.context(image_id))

    @app.post("/api/hosted/images/{image_id}/approve")
    def approve(image_id: str, body: Mutation, identity: Actor = Depends(admin)):
        return store.approve(identity, body.context(image_id))

    @app.post("/api/hosted/images/{image_id}/return")
    def return_image(image_id: str, body: Review, identity: Actor = Depends(admin)):
        return store.return_image(identity, body.context(image_id), feedback=body.feedback)

    @app.post("/api/hosted/images/{image_id}/override")
    def override(image_id: str, body: Review, identity: Actor = Depends(admin)):
        return store.override(identity, body.context(image_id), reason=body.feedback)

    @app.get("/api/hosted/images/{image_id}/history")
    def history(image_id: str, identity: Actor = Depends(actor)):
        return store.history(identity, image_id)

    @app.post("/api/hosted/images/{image_id}/import-conflicts/{conflict_id}/resolve")
    def resolve_import_conflict(image_id: str, conflict_id: str, body: ImportConflictChoice, identity: Actor = Depends(admin)):
        return store.resolve_import_conflict(identity, body.context(image_id), conflict_id, body.choice, negative=body.negative)

    @app.get("/api/hosted/receipts/{operation_id}")
    def receipt(operation_id: str, identity: Actor = Depends(actor)):
        return store.receipt(identity, operation_id)

    @app.post("/api/hosted/datasets/{dataset_id}/snapshot")
    def snapshot(dataset_id: str, identity: Actor = Depends(admin)):
        return store.snapshot(identity, dataset_id)

    @app.get("/api/hosted/schema")
    def schema(_identity: Actor = Depends(admin)):
        return app.openapi()

    install_operations_api(app, store, auth, artifacts, admin, root)
    install_tool_api(app, store, auth, artifacts, actor, admin, root)
    install_provider_api(app, store, artifacts, actor, admin)
    app.state.tools.provider_gateway = app.state.provider_gateway
    install_analysis_api(app, store, artifacts, actor, admin, app.state.tools.analysis)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (WEB_ROOT / "hosted.html").read_text()

    @app.get("/pruning", response_class=HTMLResponse)
    def pruning(job: str, request: Request, identity: Actor = Depends(admin)):
        status = app.state.tools.analysis.status(identity, job)
        if status["consumer"] != "pruning":
            raise DomainError("analysis_consumer_mismatch", 409)
        bootstrap = json.dumps({"job_id": job, "user_id": identity.user_id,
            "csrf": auth.session_csrf(request.cookies[COOKIE]), "instance_id": auth.instance_id}).replace("<", "\\u003c")
        source = (WEB_ROOT / "tator.html").read_text()
        start = source.index('<main id="datasetPruning"')
        panel = source[start:source.index("</main>", start) + len("</main>")]
        return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Tator · Dataset pruning</title><link rel="stylesheet" href="/hosted.css">'
            '<link rel="stylesheet" href="/dataset_pruning.css">'
            '<script>window.TatorPruningBoot=' + bootstrap + ';</script>'
            '<script src="/dataset_pruning.js" defer></script><script src="/hosted_pruning_ui.js" defer></script>'
            '</head><body><header class="hosted-header"><a class="hosted-brand" href="/">← Dataset workspace</a>'
            '<span>Confirmed dataset snapshot</span></header><p id="hostedPruningNotice" role="status"></p>'
            '<div id="tabDatasetPruning" class="active">' + panel + '</div></body></html>')

    @app.get("/annotate", response_class=HTMLResponse)
    def annotate(image: str, request: Request, identity: Actor = Depends(actor)):
        record = store.image(identity, image)
        bootstrap = json.dumps({"image_id": record["id"], "user": {"user_id": identity.user_id,
            "username": identity.username, "role": identity.role},
            "csrf": auth.session_csrf(request.cookies[COOKIE]), "instance_id": auth.instance_id}).replace("<", "\\u003c")
        html = (WEB_ROOT / "tator.html").read_text()
        html = html.replace('<script src="canvas.min.js', '<script>window.TatorHostedBoot=' + bootstrap + ';</script>\n<script src="/hosted_canvas.js"></script>\n<script src="/hosted_tools_ui.js"></script>\n<script src="canvas.min.js', 1)
        return html.replace("</head>", '<link rel="stylesheet" href="/hosted_canvas.css"><script src="/hosted_caption_review.js" defer></script><script src="/hosted_paid_batches_ui.js" defer></script><script src="/hosted_import_conflicts_ui.js" defer></script><script src="/hosted_instruction_review.js" defer></script></head>', 1)

    @app.get("/{asset_name}")
    def asset(asset_name: str):
        path = WEB_ROOT / asset_name
        if path.suffix not in {".js", ".css"} or not path.is_file() or path.resolve().parent != WEB_ROOT:
            raise DomainError("not_found", 404)
        return FileResponse(path)

    return app
