"""Authenticated tranche commands and keyset-scoped annotation queues."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from .preconditions import ImagePrecondition, precondition_payload
from .types import MutationContext


class Command(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    operation_id: str = Field(min_length=8, max_length=128)


class Revision(Command):
    expected_revision: int = Field(ge=0)


class Create(Command):
    name: str = Field(min_length=1, max_length=200)
    instructions: str = Field(default='', max_length=20000)
    annotator_id: str | None = None
    image_ids: list[str] = Field(default_factory=list, max_length=500)
    expected: dict[str, ImagePrecondition] = Field(default_factory=dict)


class Edit(Revision):
    name: str = Field(min_length=1, max_length=200)
    instructions: str = Field(max_length=20000)
    annotator_id: str | None = None


class Selection(Revision):
    image_ids: list[str] = Field(min_length=1, max_length=500)
    expected: dict[str, ImagePrecondition]


class Remove(Selection):
    reason: str = Field(min_length=1, max_length=10000)


class Move(Remove):
    destination_id: str
    expected_destination_revision: int = Field(ge=0)


class Activate(Revision):
    image_ids: list[str] = Field(min_length=1, max_length=10000)
    expected: dict[str, ImagePrecondition]
    annotator_id: str


class Assignee(Activate):
    image_ids: list[str] = Field(max_length=10000)


class Withdraw(Command):
    expected_version: str
    generation: int = Field(ge=0)
    editor_id: str = Field(min_length=1, max_length=200)
    submission_id: str
    reason: str = Field(min_length=1, max_length=10000)


def install_tranches_api(app, store, actor, admin):
    router = APIRouter(prefix='/api/hosted')

    def fields(body):
        data = body.model_dump(exclude_unset=True)
        if hasattr(body, 'expected'):
            data['expected'] = precondition_payload(body.expected)
        return data

    @router.get('/tranches')
    def listing(dataset_id: str | None = None, lifecycle: str | None = None,
                annotator_id: str | None = None, q: str = '', cursor: str | None = None,
                limit: int = 50, identity=Depends(actor)):
        return store.list_tranches(identity, dataset_id=dataset_id, lifecycle=lifecycle,
                                   annotator_id=annotator_id, q=q, cursor=cursor, limit=limit)

    @router.get('/tranches/{tranche_id}')
    def detail(tranche_id: str, identity=Depends(actor)):
        return store.tranche(identity, tranche_id)

    @router.get('/tranches/{tranche_id}/members')
    def members(tranche_id: str, state: str = 'all', cursor: str | None = None,
                limit: int = 100, identity=Depends(actor)):
        return store.tranche_members(identity, tranche_id, state=state, cursor=cursor, limit=limit)

    @router.get('/tranches/{tranche_id}/history')
    def history(tranche_id: str, cursor: str | None = None, limit: int = 100, identity=Depends(admin)):
        return store.tranche_history(identity, tranche_id, cursor=cursor, limit=limit)

    @router.get('/queue')
    def queue(dataset_id: str, tranche_id: str | None = None, queue: str = 'confirmed',
              cursor: str | None = None, limit: int = 100, anchor_id: str | None = None,
              direction: Literal['forward', 'backward'] = 'forward', q: str = '',
              annotator_id: str | None = None, include_total: bool = False, identity=Depends(actor)):
        return store.image_queue(identity, dataset_id, tranche_id=tranche_id, queue=queue,
                                 cursor=cursor, limit=limit, anchor_id=anchor_id, direction=direction,
                                 q=q, annotator_id=annotator_id, include_total=include_total)

    @router.get('/images/{image_id}/scope')
    def scope(image_id: str, tranche_id: str | None = None, queue: str | None = None, identity=Depends(actor)):
        return store.queue_scope(identity, image_id, tranche_id=tranche_id, queue=queue)

    @router.post('/datasets/{dataset_id}/tranches')
    def create(dataset_id: str, body: Create, identity=Depends(admin)):
        return store.create_tranche(identity, dataset_id, **fields(body))

    @router.post('/tranches/{tranche_id}/edit')
    def edit(tranche_id: str, body: Edit, identity=Depends(admin)):
        return store.edit_tranche(identity, tranche_id, **fields(body), set_annotator='annotator_id' in body.model_fields_set)

    @router.post('/tranches/{tranche_id}/members/add')
    def add(tranche_id: str, body: Selection, identity=Depends(admin)):
        return store.add_tranche_members(identity, tranche_id, **fields(body))

    @router.post('/tranches/{tranche_id}/members/remove')
    def remove(tranche_id: str, body: Remove, identity=Depends(admin)):
        return store.remove_tranche_members(identity, tranche_id, **fields(body))

    @router.post('/tranches/{tranche_id}/activate')
    def activate(tranche_id: str, body: Activate, identity=Depends(admin)):
        return store.activate_tranche(identity, tranche_id, **fields(body))

    @router.post('/tranches/{tranche_id}/move')
    def move(tranche_id: str, body: Move, identity=Depends(admin)):
        return store.move_tranche_members(identity, tranche_id, **fields(body))

    @router.post('/tranches/{tranche_id}/assignee')
    def assignee(tranche_id: str, body: Assignee, identity=Depends(admin)):
        return store.change_tranche_assignee(identity, tranche_id, **fields(body))

    @router.post('/tranches/{tranche_id}/close')
    def close(tranche_id: str, body: Revision, identity=Depends(admin)):
        return store.set_tranche_lifecycle(identity, tranche_id, **fields(body))

    @router.post('/tranches/{tranche_id}/reopen')
    def reopen(tranche_id: str, body: Revision, identity=Depends(admin)):
        return store.set_tranche_lifecycle(identity, tranche_id, **fields(body), reopen=True)

    @router.post('/images/{image_id}/withdraw-submission')
    def withdraw(image_id: str, body: Withdraw, identity=Depends(admin)):
        ctx = MutationContext(image_id, body.expected_version, body.generation, body.editor_id, body.operation_id)
        return store.withdraw_submission(identity, ctx, body.submission_id, body.reason)

    app.include_router(router)
