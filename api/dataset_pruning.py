"""API for human-reviewed, non-destructive dataset pruning variants."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from typing import Literal


class AnnotationSource(BaseModel):
    kind: Literal["dataset","transient"]
    id: str = Field(min_length=1)


class AnalysisRequest(BaseModel):
    request_id: str | None = Field(default=None,pattern=r"^[a-f0-9]{32}$")
    analysis_job_id: str = ""
    annotation_source: AnnotationSource
    expected_revision: StrictInt = Field(ge=0)
    salad_head_id: str = "auto"
    config: dict = Field(default_factory=dict)


class SelectionRequest(BaseModel):
    request_id: str | None = Field(default=None,pattern=r"^[a-f0-9]{32}$")
    selection_revision: StrictInt = Field(ge=1)
    removed_count: StrictInt = Field(default=0,ge=0)
    offset: StrictInt = Field(default=0,ge=0)
    limit: StrictInt = Field(default=12,ge=1,le=24)
    plan_revision: StrictInt | None = Field(default=None,ge=1)
    expected_review_revision: StrictInt | None = Field(default=None,ge=0)
    split_policy: Literal["grouped", "preserve"] = "grouped"
    query: str = ""
    filter: Literal["all", "unreviewed", "approved", "exceptions"] = "all"
    sort: Literal["stable", "size", "score"] = "stable"
    preview_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    view: Literal["impact", "family"] = "impact"


class PlanSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy: Literal["strict_duplicates", "budgeted_coverage"] = "strict_duplicates"
    object_weight: float = Field(default=.90, ge=0, le=1, allow_inf_nan=False, strict=True)
    max_removed_images: StrictInt = Field(default=0, ge=0)
    max_affected_appearance_groups: StrictInt = Field(default=0, ge=0)
    max_lost_appearance_weight_fraction: float = Field(default=0., ge=0, le=1, allow_inf_nan=False, strict=True)


class SettingsPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["settings"]
    selection_revision: StrictInt = Field(ge=1)
    expected_review_revision: StrictInt = Field(ge=0)
    expected_plan_revision: StrictInt = Field(ge=1)
    settings: PlanSettings
    client_sequence: StrictInt = Field(default=0, ge=0)


class KeepRequest(BaseModel):
    request_id: str | None = Field(default=None,pattern=r"^[a-f0-9]{32}$")
    selection_revision: StrictInt = Field(ge=1)
    keep_ids: list[str] = Field(default_factory=list)


def build_dataset_pruning_router(service_fn):
    router = APIRouter(prefix="/dataset_pruning",tags=["dataset pruning"])

    @router.get("/capabilities")
    def capabilities(analysis_job_id: str = ""):
        return service_fn().adapter.capabilities(analysis_job_id)

    @router.get("/profiles")
    def profiles(source_identity: str = ""):
        return service_fn().adapter.api._get_dataset_salad_profiles().snapshot(source_identity)

    @router.post("/profiles")
    def manage_profiles(payload: dict):
        adapter = service_fn().adapter
        with adapter.guard():
            return adapter.api._get_dataset_salad_profiles().mutate(payload, adapter.api._list_local_salad_heads())

    @router.post("/jobs")
    def create(payload:AnalysisRequest):
        return service_fn().create(payload.model_dump())

    @router.get("/jobs/{job_id}")
    def status(job_id:str): return service_fn().status(job_id)

    @router.post("/jobs/{job_id}/cancel")
    def cancel(job_id:str): return service_fn().cancel(job_id)

    @router.get("/jobs/{job_id}/result")
    def result(job_id:str): return service_fn().result(job_id)

    @router.post("/jobs/{job_id}/selection")
    def review(job_id:str,payload:SelectionRequest):
        try: return service_fn().review(job_id,payload.model_dump())
        except (ValueError,TypeError) as exc: raise HTTPException(400,str(exc)) from exc

    @router.post("/jobs/{job_id}/families")
    def families(job_id: str, payload: SelectionRequest):
        return service_fn().families(job_id, payload.model_dump())

    @router.get("/jobs/{job_id}/review")
    def review_snapshot(job_id: str):
        return service_fn().review_snapshot(job_id)

    @router.post('/jobs/{job_id}/families/{family_id}/members')
    def family_members(job_id: str, family_id: str, payload: dict):
        return service_fn().family_members(job_id, family_id, payload)

    @router.post("/jobs/{job_id}/review")
    def review_mutation(job_id: str, payload: dict):
        return service_fn().review_mutation(job_id, payload)

    @router.post("/jobs/{job_id}/plan/preview")
    def plan_preview(job_id: str, payload: dict):
        if payload.get("kind") == "settings":
            from pydantic import ValidationError
            try: payload = SettingsPreviewRequest.model_validate(payload).model_dump()
            except ValidationError as error: raise HTTPException(422, error.errors(include_context=False)) from error
        return service_fn().plan_preview(job_id, payload)

    @router.get("/jobs/{job_id}/plan/previews/{preview_id}")
    def plan_preview_status(job_id: str, preview_id: str):
        service = service_fn()
        return service.public_settings_preview(service.settings_preview_status(job_id, preview_id))

    @router.post("/jobs/{job_id}/plan/previews/{preview_id}/cancel")
    def plan_preview_cancel(job_id: str, preview_id: str):
        return service_fn().settings_preview_cancel(job_id, preview_id)

    @router.post("/jobs/{job_id}/comparison")
    def comparison(job_id: str, payload: dict):
        return service_fn().comparison(job_id, payload)

    @router.get("/jobs/{job_id}/recipe-lab")
    def recipe_lab(job_id: str):
        return service_fn().recipe_lab(job_id)

    @router.post("/jobs/{job_id}/recipe-lab/compare")
    def recipe_lab_compare(job_id: str, payload: dict):
        return service_fn().recipe_lab_compare(job_id, payload)

    @router.post("/jobs/{job_id}/recipe-lab/pair")
    def recipe_lab_pair(job_id: str, payload: dict):
        return service_fn().recipe_lab_pair(job_id, payload)

    @router.post("/jobs/{job_id}/recipe-lab/calibrate")
    def recipe_lab_calibrate(job_id: str, payload: dict):
        return service_fn().recipe_lab_calibrate(job_id, payload)

    @router.get("/jobs/{job_id}/images/{image_id}/objects/{label_row}")
    def crop(job_id: str, image_id: str, label_row: int):
        return service_fn().crop(job_id, image_id, label_row)

    @router.post("/jobs/{job_id}/keep")
    def keep(job_id:str,payload:KeepRequest):
        return service_fn().keep(job_id,payload.model_dump())

    @router.get("/jobs/{job_id}/images/{image_id}")
    def image(job_id:str,image_id:str): return service_fn().thumbnail(job_id,image_id)

    @router.post("/jobs/{job_id}/exports")
    def export(job_id:str,payload:SelectionRequest): return service_fn().export(job_id,payload.model_dump())

    @router.get("/jobs/{job_id}/exports/{export_id}")
    def export_status(job_id:str,export_id:str): return service_fn().export_status(job_id,export_id)

    @router.get("/jobs/{job_id}/exports/{export_id}/download")
    def download(job_id:str,export_id:str): return service_fn().download(job_id,export_id)

    return router
