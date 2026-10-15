from concurrent.futures import ThreadPoolExecutor
import hashlib
import uuid

from fastapi.testclient import TestClient
import pytest

from hosted_app import create_app
from services.hosted.errors import DomainError
from services.hosted.types import Actor, MutationContext


def uid():
    return uuid.uuid4().hex


@pytest.fixture
def configured(tmp_path):
    app = create_app(tmp_path / "state", testing=True)
    user = app.state.auth.bootstrap("admin", "provider fixture password")
    actor = Actor(user["id"], "admin")
    ds = app.state.store.create_dataset(actor, "Provider fixture", ["Car"], uid())
    image = app.state.store.import_image(actor, ds["id"], "a" * 64, "fixture.png", None, uid(), width=10, height=10)
    image = app.state.store.acquire_editor(actor, image["id"], "provider-fixture", expected_version=image["current_version"], generation=image["generation"])
    account = app.state.providers
    account.configure(actor, "openai", expected_revision=0, operation_id=uid(), credential="fixture-key-never-a-real-secret", enabled=True,
                      monthly_limit=1000, request_limit=700, allowed_datasets=[ds["id"]], models=["fixture-model"])
    return app, actor, image


def reserve(app, actor, image, amount=700):
    ctx = MutationContext(image["id"], image["current_version"], image["generation"], "provider-fixture", uid())
    request = dict(provider="openai", model="fixture-model", maximum_microdollars=amount, request_fingerprint=hashlib.sha256(b"exact immutable provider input").hexdigest())
    return ctx, app.state.providers.reserve(actor, ctx, **request), request


def test_concurrent_reservations_and_unknown_cost_remain_bounded(configured):
    app, actor, image = configured
    def attempt():
        try: return reserve(app, actor, image)
        except DomainError as error: return error.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: attempt(), range(2)))
    assert results.count("paid_provider_budget_exceeded") == 1
    ctx, call, request = next(row for row in results if isinstance(row, tuple))
    assert app.state.providers.reserve(actor, ctx, **request) == call
    assert app.state.providers.dispatch(actor, ctx, call["id"]) == "fixture-key-never-a-real-secret"
    with pytest.raises(DomainError, match="provider_call_already_dispatched"):
        app.state.providers.dispatch(actor, ctx, call["id"])
    app.state.providers.settle(call["id"])
    assert app.state.providers.list(actor)[0]["spent_or_reserved"] == 700
    assert app.state.providers.list(actor)[0]["unresolved_calls"] == 1
    app.state.providers.reconcile(actor, call["id"], charged=200, note="Checked fixture provider usage", operation_id=uid())
    assert app.state.providers.list(actor)[0]["spent_or_reserved"] == 200
    reserve(app, actor, image)


def test_changed_authority_and_cost_underestimate_do_not_leak(configured):
    app, actor, image = configured
    ctx, call, _ = reserve(app, actor, image)
    image = app.state.store.acquire_editor(actor, image["id"], "different-tab", expected_version=image["current_version"], generation=image["generation"], takeover=True)
    with pytest.raises(DomainError, match="ownership_changed"):
        app.state.providers.dispatch(actor, ctx, call["id"])
    app.state.providers.reconcile(actor, call["id"], charged=0, note="Request never dispatched", operation_id=uid())
    image = app.state.store.acquire_editor(actor, image["id"], "provider-fixture", expected_version=image["current_version"], generation=image["generation"], takeover=True)
    ctx, call, _ = reserve(app, actor, image)
    app.state.providers.dispatch(actor, ctx, call["id"])
    app.state.providers.settle(call["id"], actual_microdollars=800)
    result = app.state.providers.list(actor)[0]
    assert result["spent_or_reserved"] == 800 and not result["enabled"]
    with pytest.raises(DomainError, match="paid_provider_disabled"):
        reserve(app, actor, image, 100)


def test_http_credentials_are_write_only_and_restore_pauses_billing(configured):
    app, actor, image = configured
    client = TestClient(app)
    response = client.post("/api/hosted/login", json={"username": "admin", "password": "provider fixture password"})
    identity = response.json(); client.headers.update({"x-tator-csrf": identity["csrf"], "x-tator-instance": identity["instance_id"]})
    result = client.get("/api/hosted/providers")
    assert result.status_code == 200 and "fixture-key" not in result.text
    body = {"operation_id": uid(), "expected_revision": 1, "credential": "new-fixture-key-not-a-real-secret", "enabled": True,
            "monthly_limit": 1000, "request_limit": 700, "models": ["fixture-model"], "allowed_datasets": [image["dataset_id"]]}
    response = client.put("/api/hosted/providers/openai", json=body)
    assert response.status_code == 200 and body["credential"] not in response.text
    invalid = client.put("/api/hosted/providers/openai", json={"credential": body["credential"]})
    assert invalid.status_code == 422 and body["credential"] not in invalid.text
    invite = app.state.auth.create_user(actor, "annotator"); app.state.auth.redeem(invite["invite_token"], "provider fixture password")
    other = TestClient(app); assert other.post("/api/hosted/login", json={"username": "annotator", "password": "provider fixture password"}).status_code == 200
    assert other.get("/api/hosted/providers").status_code == 403
    ctx, call, _ = reserve(app, actor, image)
    with app.state.store.transaction() as con:
        app.state.providers.after_restore(con)
    assert not app.state.providers.list(actor)[0]["enabled"]
    assert app.state.providers.calls(actor)[0]["state"] == "unknown"


def test_paid_gateway_uses_canonical_pixels_and_never_repeats_unknown_calls(configured):
    import base64
    import io
    from PIL import Image
    from services.hosted.provider_gateway import PaidCaptionGateway
    app, actor, previous = configured
    data = io.BytesIO(); Image.new('RGB', (32, 24), 'blue').save(data, format='PNG')
    media = app.state.artifacts.put_bytes(data.getvalue())
    image = app.state.store.import_image(actor, previous['dataset_id'], media.sha256, 'actual.png', None, uid(),
        width=32, height=24, size_bytes=media.size_bytes)
    image = app.state.store.acquire_editor(actor, image['id'], 'provider-fixture', expected_version=image['current_version'], generation=image['generation'])
    price = {'fixture-model': {'input_microdollars_per_million': 1000000, 'cached_input_microdollars_per_million': 500000,
        'output_microdollars_per_million': 2000000, 'input_token_ceiling': 100, 'max_output_tokens': 100}}
    requests = []
    def transport(secret, payload):
        assert secret == 'fixture-key-never-a-real-secret'
        assert payload['store'] is False and payload['service_tier'] == 'default'
        assert base64.b64decode(payload['input'][0]['content'][1]['image_url'].split(',')[1]) == data.getvalue()
        requests.append(payload)
        return {'id': 'fixture-response', 'status': 'completed',
                'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'A blue fixture.'}]}],
                'usage': {'input_tokens': 30, 'input_tokens_details': {'cached_tokens': 10}, 'output_tokens': 5}}
    gateway = PaidCaptionGateway(app.state.providers, app.state.artifacts, pricing=price, transport=transport)
    ctx = MutationContext(image['id'], image['current_version'], image['generation'], 'provider-fixture', uid())
    result = gateway.caption(actor, ctx, model='fixture-model', prompt='Describe this image.', max_output_tokens=32)
    assert result['caption'] == 'A blue fixture.' and result['estimated_microdollars'] == 35
    assert gateway.caption(actor, ctx, model='fixture-model', prompt='Describe this image.', max_output_tokens=32) == result
    assert len(requests) == 1
    def timeout(secret, payload):
        requests.append(payload)
        raise TimeoutError('Fixture loss after provider received request')
    gateway.transport = timeout
    ctx = MutationContext(image['id'], image['current_version'], image['generation'], 'provider-fixture', uid())
    with pytest.raises(DomainError, match='paid_call_failed_check_billing'):
        gateway.caption(actor, ctx, model='fixture-model', prompt='Describe.', max_output_tokens=32)
    with pytest.raises(DomainError, match='paid_call_outcome_unresolved'):
        gateway.caption(actor, ctx, model='fixture-model', prompt='Describe.', max_output_tokens=32)
    assert len(requests) == 2
    assert app.state.providers.list(actor)[0]['spent_or_reserved'] == 199
    assert gateway.catalog(actor, image['dataset_id'])['models'][0]['id'] == 'fixture-model'
    # A late successful response is billed and retained but cannot apply after a takeover.
    def takeover(secret, payload):
        output = transport(secret, payload)
        app.state.store.acquire_editor(actor, image['id'], 'replacement-tab', expected_version=image['current_version'], generation=image['generation'], takeover=True)
        return output
    gateway.transport = takeover
    ctx = MutationContext(image['id'], image['current_version'], image['generation'], 'provider-fixture', uid())
    with pytest.raises(DomainError, match='ownership_changed'):
        gateway.caption(actor, ctx, model='fixture-model', prompt='Describe again.', max_output_tokens=32)
    assert app.state.providers.list(actor)[0]['spent_or_reserved'] == 234
    assert len(requests) == 3


def test_paid_pipeline_stages_bind_pixels_prompts_snapshot_and_recovery(configured):
    import base64
    import io
    from PIL import Image
    from services.hosted.provider_gateway import PaidCaptionGateway
    from services.hosted.providers import SnapshotAuthority
    from services.hosted.operations import HostedOperations
    app, actor, previous = configured
    stream = io.BytesIO(); Image.new('RGB', (40, 30), 'red').save(stream, format='PNG')
    media = app.state.artifacts.put_bytes(stream.getvalue())
    image = app.state.store.import_image(actor, previous['dataset_id'], media.sha256, 'stage.png', None, uid(), width=40, height=30, size_bytes=media.size_bytes)
    image = app.state.store.acquire_editor(actor, image['id'], 'stage-editor')
    def ctx(value):
        return MutationContext(value['id'], value['current_version'], value['generation'], 'stage-editor', uid())
    image = app.state.store.save(actor, ctx(image), {'objects': []}, complete=True, negative=True)
    image = app.state.store.approve(actor, ctx(image))
    snapshot = app.state.store.create_snapshot(actor, image['dataset_id'], uid())
    authority = SnapshotAuthority(snapshot['snapshot_id'], image['id'], image['current_version'])
    pricing = {'fixture-model': {'input_microdollars_per_million': 1000000, 'cached_input_microdollars_per_million': 500000,
        'output_microdollars_per_million': 2000000, 'input_token_ceiling': 100, 'max_output_tokens': 100}}
    payloads = []
    def transport(secret, payload):
        payloads.append(payload)
        assert payload['store'] is False and payload['service_tier'] == 'default'
        return {'id': 'stage-fixture', 'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'A useful stage result.'}]}], 'usage': {'input_tokens': 20, 'output_tokens': 4}}
    gateway = PaidCaptionGateway(app.state.providers, app.state.artifacts, pricing=pricing, transport=transport)
    request = dict(stage_id='fixed-window-stage-1', model='fixture-model', user_prompt='Describe this crop.', system_prompt='Retain precise detail.', image=Image.new('RGB', (8, 9), 'blue'), source_media_hash=media.sha256, max_output_tokens=32, reasoning_effort='high')
    result = gateway.stage(actor, authority, **request)
    assert result['text'] == 'A useful stage result.'
    assert payloads[0]['instructions'] == request['system_prompt'] and payloads[0]['reasoning'] == {'effort': 'high'}
    derived = base64.b64decode(payloads[0]['input'][0]['content'][1]['image_url'].split(',')[1])
    assert Image.open(io.BytesIO(derived)).size == (8, 9)
    assert gateway.stage(actor, authority, **request) == result and len(payloads) == 1
    with pytest.raises(DomainError, match='operation_id_reused'):
        gateway.stage(actor, authority, **{**request, 'system_prompt': 'Changed instruction'})
    # Other dataset edits do not invalidate this exact source image.
    app.state.store.patch_dataset_settings(actor, image['dataset_id'], {'context': 'Changed dataset guidance'}, snapshot['manifest']['revision'], uid())
    text = gateway.stage(actor, authority, **{**request, 'stage_id': 'fixed-text-stage-2', 'image': None})
    assert len(payloads[1]['input'][0]['content']) == 1
    with app.state.store.read_connection() as con:
        retained = {row['sha256'] for row in HostedOperations.artifact_references(con)}
    assert {ref['sha256'] for ref in result['input_artifacts'] + text['input_artifacts']} <= retained
    member = app.state.auth.create_user(actor, 'another-user')
    with pytest.raises(DomainError):
        gateway.stage(Actor(member['id'], 'annotator'), authority, **{**request, 'stage_id': 'not-admin-stage'})
    changed = app.state.store.acquire_editor(actor, image['id'], 'stage-editor')
    app.state.store.save(actor, ctx(changed), {'objects': [], 'caption': 'A human correction'}, complete=True, negative=True)
    with pytest.raises(DomainError, match='paid_snapshot_image_changed'):
        gateway.stage(actor, authority, **request)
    assert len(payloads) == 2


def test_paid_stage_unknown_outcome_never_dispatches_twice(configured):
    from services.hosted.provider_gateway import PaidCaptionGateway
    app, actor, image = configured
    authority = MutationContext(image['id'], image['current_version'], image['generation'], 'provider-fixture', uid())
    pricing = {'fixture-model': {'input_microdollars_per_million': 1000000, 'cached_input_microdollars_per_million': 500000,
        'output_microdollars_per_million': 2000000, 'input_token_ceiling': 100, 'max_output_tokens': 100}}
    sent = []
    def timeout(secret, payload):
        sent.append(payload); raise TimeoutError('Reply lost after dispatch')
    gateway = PaidCaptionGateway(app.state.providers, app.state.artifacts, pricing=pricing, transport=timeout)
    request = dict(stage_id='fixed-lost-stage', model='fixture-model', user_prompt='Refine the previous caption.', source_media_hash=image['media_hash'], max_output_tokens=32)
    with pytest.raises(DomainError, match='paid_call_failed_check_billing'):
        gateway.stage(actor, authority, **request)
    with pytest.raises(DomainError, match='paid_call_outcome_unresolved'):
        gateway.stage(actor, authority, **request)
    reopened = app.state.store.acquire_editor(actor, image['id'], 'new-editor', expected_version=image['current_version'], generation=image['generation'], takeover=True)
    new_authority = MutationContext(image['id'], image['current_version'], reopened['generation'], 'new-editor', uid())
    with pytest.raises(DomainError, match='paid_call_outcome_unresolved'):
        gateway.stage(actor, new_authority, **request)
    assert len(sent) == 1 and app.state.providers.list(actor)[0]['unresolved_calls'] == 1


def test_paid_stage_replays_settled_output_after_reconnect_and_guards_dispatch(configured):
    from services.hosted.provider_gateway import PaidCaptionGateway
    app, actor, image = configured
    authority = MutationContext(image['id'], image['current_version'], image['generation'], 'provider-fixture', uid())
    pricing = {'fixture-model': {'input_microdollars_per_million': 1000000, 'cached_input_microdollars_per_million': 500000,
        'output_microdollars_per_million': 2000000, 'input_token_ceiling': 100, 'max_output_tokens': 100}}
    sent = []
    def transport(secret, payload):
        sent.append(payload)
        return {'id': 'reconnect-fixture', 'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Retained stage.'}]}], 'usage': {'input_tokens': 20, 'output_tokens': 4}}
    gateway = PaidCaptionGateway(app.state.providers, app.state.artifacts, pricing=pricing, transport=transport)
    request = dict(stage_id='settled-reconnect-stage', model='fixture-model', user_prompt='Refine the previous caption.', source_media_hash=image['media_hash'], max_output_tokens=32)
    result = gateway.stage(actor, authority, **request)
    reopened = app.state.store.acquire_editor(actor, image['id'], 'new-editor', expected_version=image['current_version'], generation=image['generation'], takeover=True)
    new_authority = MutationContext(image['id'], image['current_version'], reopened['generation'], 'new-editor', uid())
    assert gateway.stage(actor, new_authority, **request) == result
    assert len(sent) == 1
    with pytest.raises(DomainError, match='operation_id_reused'):
        gateway.stage(actor, new_authority, **{**request, 'user_prompt': 'Changed immutable input'})
    # This guard runs with the dispatch transaction, after pixels and budget
    # preparation. A cancelled job cannot slip another paid call through.
    def cancelled(con):
        assert con.in_transaction
        raise DomainError('caption_job_cancelled', 409)
    with pytest.raises(DomainError, match='caption_job_cancelled'):
        gateway.stage(actor, new_authority, **{**request, 'stage_id': 'cancel-before-dispatch'}, dispatch_guard=cancelled)
    calls = app.state.providers.calls(actor)
    assert len(sent) == 1 and len(calls) == 2
    assert any(row['state'] == 'reconciled' and row['charged'] == 0 for row in calls)
    assert app.state.providers.list(actor)[0]['spent_or_reserved'] == 28
    resumed = {**request, 'stage_id': 'cancel-before-dispatch'}
    # Explicit resume can reuse a provably unsent stage. If this admission then
    # loses its provider reply, even a zero-dollar manual reconciliation must
    # never make it eligible to dispatch again.
    def lose_reply(secret, payload):
        sent.append(payload)
        raise TimeoutError('Unknown after re-admission')
    gateway.transport = lose_reply
    with pytest.raises(DomainError, match='paid_call_failed_check_billing'):
        gateway.stage(actor, new_authority, **resumed)
    unknown = next(row for row in app.state.providers.calls(actor) if row['state'] == 'unknown')
    app.state.providers.reconcile(actor, unknown['id'], charged=0, note='Fixture billing reviewed; outcome remains unknown', operation_id=uid())
    with pytest.raises(DomainError, match='paid_call_outcome_unresolved'):
        gateway.stage(actor, new_authority, **resumed)
    assert len(sent) == 2
