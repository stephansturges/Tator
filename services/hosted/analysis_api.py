"""Authenticated projections of retained native analysis results."""
import json
from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from .analysis_views import AnalysisViews
from .errors import DomainError
from .types import MutationContext


class AnnotationApplication(BaseModel):
    image_id: str
    expected_version: str
    generation: int = Field(ge=0)
    editor_id: str
    operation_id: str = Field(min_length=8, max_length=128)
    annotations: dict
    complete: bool = True
    negative: bool = False


def install_analysis_api(app, store, artifacts, actor_dependency, admin_dependency, analysis):
    views = AnalysisViews(analysis)
    from .dqe import DqeReview
    dqe = DqeReview(analysis)
    views.session_provider = dqe.session
    app.state.hosted_dqe = dqe
    app.state.analysis_views = views
    router = APIRouter(prefix="/api/hosted/analysis", tags=["hosted analysis"])

    def job_status(identity, job_id):
        result = analysis.status(identity, job_id)
        # Recover a running task after navigation without relying on a browser
        # variable. Request control remains restricted to its initiating user.
        with store.read_connection() as con:
            if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='hosted_tool_analysis'").fetchone():
                row = con.execute("SELECT request_id,actor_json FROM hosted_tool_analysis WHERE job_id=?", (job_id,)).fetchone()
                if row and json.loads(row['actor_json'])['user_id'] == identity.user_id:
                    result['runtime_request_id'] = row['request_id']
        return result

    @router.get("/jobs")
    def jobs(dataset_id: str | None = None, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), identity=Depends(admin_dependency)):
        with store.read_connection() as con:
            store._actor(con, identity, admin=True)
            where, values = ("WHERE s.dataset_id=?", [dataset_id]) if dataset_id else ("", [])
            rows = con.execute("SELECT j.id FROM hosted_analysis_jobs j JOIN hosted_snapshots s ON s.id=j.snapshot_id " + where + " ORDER BY j.created_at DESC,j.id LIMIT ? OFFSET ?", [*values, limit, offset]).fetchall()
        return {"jobs": [job_status(identity, row[0]) for row in rows], "offset": offset}

    @router.get("/jobs/{job_id}")
    def status(job_id: str, identity=Depends(admin_dependency)):
        return job_status(identity, job_id)

    @router.get("/jobs/{job_id}/view")
    def view(job_id: str, identity=Depends(admin_dependency)):
        return views.metadata(identity, job_id)

    @router.get("/jobs/{job_id}/result")
    def result(job_id: str, identity=Depends(admin_dependency)):
        reference = analysis.result_reference(identity, job_id)
        def chunks():
            with artifacts.open(reference) as stream:
                while chunk := stream.read(1024 * 1024):
                    yield chunk
        return StreamingResponse(chunks(), media_type="application/json")

    @router.get('/jobs/{job_id}/artifact')
    def artifact(job_id: str, path: str = Query(min_length=1, max_length=1000), identity=Depends(admin_dependency)):
        # Exact registered job-relative identity, never a caller filesystem path.
        reference = views._reference(identity, job_id, path)
        def chunks():
            with artifacts.open(reference) as stream:
                while chunk := stream.read(1024 * 1024):
                    yield chunk
        kind = 'application/zip' if path.endswith('.zip') else 'application/json' if path.endswith('.json') else 'application/octet-stream'
        return StreamingResponse(chunks(), media_type=kind, headers={'Content-Disposition': 'attachment; filename="tator-analysis-artifact' + ('.zip' if kind == 'application/zip' else '.json' if kind == 'application/json' else '.bin') + '"'})

    @router.get("/jobs/{job_id}/images/{image_id}/media")
    def frozen_image(job_id: str, image_id: str, identity=Depends(admin_dependency)):
        from PIL import Image
        reference = analysis.image_reference(identity, job_id, image_id)
        with artifacts.open(reference) as source:
            with Image.open(source) as picture:
                content_type = Image.MIME.get(picture.format, "application/octet-stream")
        def chunks():
            with artifacts.open(reference) as source:
                while chunk := source.read(1024 * 1024):
                    yield chunk
        return StreamingResponse(chunks(), media_type=content_type)

    @router.get("/jobs/{job_id}/dqe/graph")
    def graph(job_id: str, projection_mode: str | None = None, class_name: str | None = None,
              class_names: list[str] | None = Query(None), objects: str = "all", object_size: str = "all", reviewed: str = "any",
              limit: int = Query(5000, ge=1), cursor: str | None = None, identity=Depends(admin_dependency)):
        return views.graph(identity, job_id, projection_mode=projection_mode, class_name=class_name, class_names=class_names,
                           objects=objects, object_size=object_size, reviewed=reviewed, limit=limit, cursor=cursor)

    @router.get("/jobs/{job_id}/dqe/queue")
    def queue(job_id: str, category: str = "review", cursor: str | None = None, limit: int = Query(50, ge=1), identity=Depends(admin_dependency)):
        return views.queue(identity, job_id, category=category, cursor=cursor, limit=limit)

    @router.get("/jobs/{job_id}/dqe/points/{point_id}")
    def point(job_id: str, point_id: str, identity=Depends(admin_dependency)):
        return dqe.point(identity, job_id, point_id)

    @router.get("/jobs/{job_id}/dqe/points/{point_id}/evidence")
    def evidence(job_id: str, point_id: str, identity=Depends(admin_dependency)):
        return views.evidence(identity, job_id, point_id)

    @router.get("/jobs/{job_id}/dqe/points/{point_id}/image")
    def image(job_id: str, point_id: str, context: bool = False, size: int = Query(256, ge=32, le=1024), identity=Depends(admin_dependency)):
        return Response(views.thumbnail(identity, job_id, point_id, context=context, size=size), media_type="image/jpeg")

    @router.post("/jobs/{job_id}/annotations")
    def apply(job_id: str, body: AnnotationApplication, identity=Depends(admin_dependency)):
        ctx = MutationContext(body.image_id, body.expected_version, body.generation, body.editor_id, body.operation_id)
        return analysis.apply_annotations(identity, job_id, ctx, body.annotations, complete=body.complete, negative=body.negative)

    @router.get('/jobs/{job_id}/dqe/review')
    def review_state(job_id: str, identity=Depends(admin_dependency)):
        return dqe.state(identity, job_id)

    @router.get('/jobs/{job_id}/dqe/review/history')
    def review_history(job_id: str, limit: int = 100, offset: int = 0, identity=Depends(admin_dependency)):
        return dqe.history(identity, job_id, limit=limit, offset=offset)

    @router.post('/jobs/{job_id}/dqe/points/{point_id}/review')
    def disposition(job_id: str, point_id: str, body: dict, identity=Depends(admin_dependency)):
        try:
            return dqe.disposition(identity, job_id, point_id, **body)
        except TypeError as error:
            raise DomainError('dqe_review_request_invalid', 422) from error

    @router.post('/jobs/{job_id}/dqe/actions')
    def actions(job_id: str, body: dict, identity=Depends(admin_dependency)):
        try:
            context = MutationContext(**body['context'])
            return dqe.apply(identity, job_id, context, body['actions'], negative=body.get('negative', False), expected_review_revision=body.get('expected_review_revision'))
        except (KeyError, TypeError) as error:
            raise DomainError('dqe_action_request_invalid', 422) from error

    @router.post('/jobs/{job_id}/dqe/batches')
    def create_batch(job_id: str, body: dict, identity=Depends(admin_dependency)):
        try:
            return dqe.create_batch(identity, job_id, body['items'], body['operation_id'])
        except (KeyError, TypeError) as error:
            raise DomainError('dqe_batch_request_invalid', 422) from error

    @router.get('/jobs/{job_id}/dqe/batches/{batch_id}')
    def batch(job_id: str, batch_id: str, identity=Depends(admin_dependency)):
        return dqe.batch(identity, job_id, batch_id)

    @router.get('/jobs/{job_id}/dqe/batches')
    def batches(job_id: str, limit: int = 50, offset: int = 0, identity=Depends(admin_dependency)):
        return dqe.batches(identity, job_id, limit=limit, offset=offset)

    @router.post('/jobs/{job_id}/dqe/batches/{batch_id}/resume')
    def resume_batch(job_id: str, batch_id: str, body: dict, identity=Depends(admin_dependency)):
        try:
            return dqe.resume_batch(identity, job_id, batch_id, body['contexts'], body['operation_id'])
        except (KeyError, TypeError) as error:
            raise DomainError('dqe_batch_request_invalid', 422) from error

    @router.post('/jobs/{job_id}/dqe/batches/{batch_id}/process')
    def process_batch(job_id: str, batch_id: str, body: dict, identity=Depends(admin_dependency)):
        return dqe.process_batch(identity, job_id, batch_id, limit=body.get('limit', 5))

    @router.post('/jobs/{job_id}/dqe/batches/{batch_id}/cancel')
    def cancel_batch(job_id: str, batch_id: str, body: dict, identity=Depends(admin_dependency)):
        return dqe.cancel_batch(identity, job_id, batch_id, body.get('operation_id', ''))

    app.include_router(router)
    _install_pruning(app, admin_dependency, analysis)
    return views


def _install_pruning(app, admin_dependency, analysis):
    from contextvars import ContextVar
    from types import SimpleNamespace
    from api.dataset_pruning import build_dataset_pruning_router
    from .pruning import HostedPruning
    from services.dataset_pruning_jobs import read_json
    from .salad_profiles import install_salad_profiles_api
    profiles = install_salad_profiles_api(app, analysis.store, analysis.artifacts, admin_dependency)
    facade = HostedPruning(analysis)
    app.state.hosted_pruning = facade
    scope = ContextVar("hosted_pruning_scope")

    async def bind(hosted_job_id: str, identity=Depends(admin_dependency)):
        analysis.status(identity, hosted_job_id)
        token = scope.set((identity, hosted_job_id))
        try:
            yield
        finally:
            scope.reset(token)

    class Proxy:
        def __init__(self):
            self.actor, self.hosted_id = scope.get()
            self.service = facade.service(self.actor, self.hosted_id)
            self.dataset_id = analysis.status(self.actor, self.hosted_id)["dataset_id"]
            self.adapter = SimpleNamespace(capabilities=self.capabilities,
                api=SimpleNamespace(_get_dataset_salad_profiles=lambda: self,
                    _list_local_salad_heads=lambda: self.snapshot()["heads"]), guard=self.service.adapter.guard)
        def capabilities(self, analysis_job_id=""):
            path = self.service.directory(self.service.hosted_native_id) / "hosted-capabilities.json"
            if not path.is_file():
                raise DomainError("pruning_capabilities_unavailable", 409)
            value = read_json(path)
            current = self.snapshot()
            # Historical feature recipes remain frozen. Current deployable heads
            # are exposed separately for the next analysis and profile management.
            registered = {head["id"]: head for head in current["heads"]}
            heads = {head["id"]: head for head in value.get("heads", [])}
            for head_id, head in registered.items():
                if head_id in heads and heads[head_id].get("head_sha256") != head["head_sha256"]:
                    raise DomainError("salad_head_alias_exists", 409)
                heads[head_id] = head
            return {**value, "heads": list(heads.values()), "hosted_frozen_analysis": True,
                    "profile_revision": current["revision"]}
        def snapshot(self, source_identity=""):
            if source_identity and source_identity != "dataset:" + self.dataset_id:
                raise DomainError("salad_dataset_mismatch", 409)
            return profiles.snapshot(self.actor, self.dataset_id)
        def mutate(self, payload, available_heads=None):
            return profiles.mutate(self.actor, self.dataset_id, payload)
        def create(self, payload):
            raise DomainError("hosted_native_analysis_required", 409, "Start a new pruning analysis from Dataset analysis. This page reviews its frozen result.")
        def public_settings_preview(self, value):
            return self.service.public_settings_preview(value)
        def __getattr__(self, method):
            if method.startswith("_"):
                raise AttributeError(method)
            def invoke(native_job_id, *args, **kwargs):
                if native_job_id != self.service.hosted_native_id:
                    raise DomainError("pruning_job_not_in_snapshot", 404)
                return facade.call(self.actor, self.hosted_id, method, *args, **kwargs)
            return invoke

    native = build_dataset_pruning_router(Proxy)
    # Profile changes use current dataset CAS and intentionally stale this frozen
    # analysis; they do not rewrite its already measured feature recipe.
    app.include_router(native, prefix="/api/hosted/analysis/jobs/{hosted_job_id}", dependencies=[Depends(bind)])
