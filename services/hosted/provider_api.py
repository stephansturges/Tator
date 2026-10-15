"""Administrator-only configuration; credentials are write-only over HTTP."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from .providers import ProviderAccounts
from .provider_gateway import PaidCaptionGateway
from .errors import DomainError
from .types import MutationContext


class Configuration(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    operation_id: str = Field(min_length=8, max_length=128)
    expected_revision: int = Field(ge=0)
    credential: str | None = None
    enabled: bool = False
    monthly_limit: int = Field(ge=0, le=10**12)
    request_limit: int = Field(ge=0, le=10**12)
    allowed_datasets: list[str] = Field(max_length=10000)
    models: list[str] = Field(max_length=100)


class Reconciliation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    operation_id: str = Field(min_length=8, max_length=128)
    charged: int = Field(ge=0, le=10**12)
    note: str = Field(min_length=3, max_length=2000)


class PaidCaption(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    image_id: str
    expected_version: str
    generation: int
    editor_id: str
    operation_id: str = Field(min_length=8, max_length=128)
    model: str = Field(max_length=160)
    prompt: str = Field(min_length=1, max_length=64000)
    max_output_tokens: int = Field(default=2048, ge=16)
    detail: str = "high"


def install_provider_api(app, store, artifacts, actor_dependency, admin_dependency):
    accounts = ProviderAccounts(store)
    gateway = PaidCaptionGateway(accounts, artifacts)
    router = APIRouter(prefix="/api/hosted/providers")

    def dispatch_guard(_con):
        # Check at the credential handoff, inside the accounting transaction.
        # Replaying a retained result does not dispatch and need not wait.
        runtime = app.state.tools._runtime
        if runtime is not None and runtime.snapshot()["phase"] != "available":
            raise DomainError("runtime_automation_paused", 409)

    @router.get("/available")
    def available(dataset_id: str, actor=Depends(actor_dependency)):
        return gateway.catalog(actor, dataset_id)

    @router.post("/caption")
    def caption(body: PaidCaption, actor=Depends(actor_dependency)):
        context = MutationContext(body.image_id, body.expected_version, body.generation, body.editor_id, body.operation_id)
        return gateway.caption(actor, context, model=body.model, prompt=body.prompt,
                               max_output_tokens=body.max_output_tokens, detail=body.detail,
                               dispatch_guard=dispatch_guard)


    @router.get("")
    def providers(actor=Depends(admin_dependency)):
        return {"providers": accounts.list(actor), "currency": "USD", "unit": "millionths"}

    @router.put("/{name}")
    def configure(name: str, body: Configuration, actor=Depends(admin_dependency)):
        return accounts.configure(actor, name, **body.model_dump())

    @router.get("/calls")
    def calls(limit: int = 100, offset: int = 0, actor=Depends(admin_dependency)):
        return {"calls": accounts.calls(actor, limit=limit, offset=offset)}

    @router.post("/calls/{call_id}/reconcile")
    def reconcile(call_id: str, body: Reconciliation, actor=Depends(admin_dependency)):
        return accounts.reconcile(actor, call_id, **body.model_dump())

    app.include_router(router)
    app.state.providers = accounts
    app.state.provider_gateway = gateway
    return accounts
