"""Real browser/control-plane integration, opt in with TATOR_HOSTED_BROWSER_TESTS=1.

Only disposable databases and generated fixture media are used. No model doubles
are presented as evidence for GPU tools; this covers annotation authority/UI.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta

import httpx
import pytest
from PIL import Image

pytestmark = pytest.mark.skipif(os.environ.get('TATOR_HOSTED_BROWSER_TESTS') != '1', reason='Explicit isolated browser run required')


@pytest.fixture
def hosted_origin(tmp_path, request, monkeypatch):
    from hosted_app import create_app
    from services.hosted.types import Actor
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    origin = f'http://127.0.0.1:{port}'
    batch_fixture = getattr(request, "param", None) == "caption_batch"
    paid_batch_fixture = getattr(request, "param", None) in {"paid_batch", "paid_batch_billing"}
    caption_fixture = getattr(request, "param", None) == "caption" or batch_fixture
    analysis_fixture = getattr(request, "param", None) == "analysis"
    restore_fixture = getattr(request, "param", None) == "restore"
    pruning_fixture = getattr(request, "param", None) == "pruning"
    dqe_fixture = getattr(request, "param", None) == "dqe"
    if analysis_fixture:
        configuration = tmp_path / "models.json"
        configuration.write_text(json.dumps({"models": {}, "heavy_jobs": {"statistics": {
            "factory": "services.hosted.native_analysis:NativeAnalysisAdapter",
            "config": {"native_root": str(tmp_path / "native")}, "device": "cpu",
            "operation": "statistics", "allowed_parameters": [], "timeout": 30}}}))
        monkeypatch.setenv("TATOR_HOSTED_MODEL_CONFIG", str(configuration))
    if caption_fixture:
        source = tmp_path / "fixture-models"; source.mkdir()
        (source / "sam.pt").write_bytes(b"fixture weights never loaded")
        qwen = source / "qwen"; qwen.mkdir()
        (qwen / "config.json").write_text('{"model_type":"qwen3_vl"}')
        (qwen / "weights.safetensors").write_bytes(b"fixture weights never loaded")
        gate = tmp_path / "gate"; gate.mkdir()
        config = {"allowed_model_roots": [str(source)], "models": {
            "sam": {"kind": "sam1", "checkpoint": str(source / "sam.pt"), "model_type": "vit_b"},
            "qwen": {"kind": "qwen", "model_path": str(qwen), "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in qwen.iterdir()}}},
            "user_sam": {"default_model_id": "sam", "allowed_model_ids": ["sam"], "device": "cpu", "capacity": 1},
            "user_qwen": {"default_model_id": "qwen", "allowed_model_ids": ["qwen"], "device": "cpu"}, "heavy_jobs": {}}
        configuration = tmp_path / "models.json"; configuration.write_text(json.dumps(config))
        monkeypatch.setenv("TATOR_HOSTED_MODEL_CONFIG", str(configuration))
        monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
        monkeypatch.setenv("HOSTED_FIXTURE_GATE", str(gate))
    app = create_app(tmp_path / 'state', testing=True, origin=origin)
    auth, store = app.state.auth, app.state.store
    admin_user = auth.bootstrap('admin', 'test-only-fixture-password')
    admin = Actor(admin_user['id'], 'admin', 'admin')
    if caption_fixture:
        for model in ("sam", "qwen"):
            app.state.tools.install_model(admin, model, str(uuid.uuid4()))
    users = {}
    for name in ('alice', 'bob'):
        user = auth.create_user(admin, name); auth.redeem(user['invite_token'], 'test-only-fixture-password'); users[name] = user
    ds = store.create_dataset(admin, 'Hosted fixture', ['Car', 'Person'], str(uuid.uuid4()))
    first_image = None
    for number in range(4):
        data = io.BytesIO(); Image.new('RGB', (640, 480), (50+number*30, 70, 90)).save(data, format='PNG')
        ref = app.state.artifacts.put_bytes(data.getvalue())
        annotations = {'objects': []}
        if getattr(request, 'param', None) == 'caption_records' and number == 1:
            annotations.update(caption='Same wording from history.', captions=[
                {'id': 'shared-id', 'text': 'Alias from imported captions.', 'source': 'manual', 'metadata': {'opaque': {'ids': [1, 2]}}},
                {'caption': 'Same wording from history.', 'lifecycle_status': 'deleted', 'generated_qa_pairs': [{'question': 'Old question?', 'answer': 'Old answer.'}]},
                {'caption': None, 'text': 'Do not resurrect this alias.', 'opaque': 9},
                {'caption': '', 'generated_qa_pairs': [{'question': 'What is present?', 'answer': 'A car.'}]}],
                alternate_captions=[{'id': 'shared-id', 'text': 'Another alias.', 'model': 'legacy'},
                    'Plain string alternate.', {'caption': 'Retained inactive.', 'lifecycle_status': 'superseded'},
                    {'caption': 'Conflicting caption.', 'text': 'Ignored alias.'}, ''])
        image = store.import_image(admin, ds['id'], ref.sha256, f'frame-{number}.png', annotations, str(uuid.uuid4()), width=640, height=480, size_bytes=ref.size_bytes)
        if number == 0:
            store.assign(admin, ds['id'], [image['id']], users['alice']['id'], str(uuid.uuid4()), expected={image['id']: {'expected_version': image['current_version'], 'generation': image['generation']}})
            first_image = image['id']
        if ((analysis_fixture or batch_fixture or paid_batch_fixture) and number == 1) or ((pruning_fixture or dqe_fixture) and number in (1, 2)):
            from services.hosted.types import MutationContext
            image = store.acquire_editor(admin, image['id'], 'fixture-administrator', expected_version=image['current_version'], generation=image['generation'])
            def mutation(value):
                return MutationContext(value['id'], value['current_version'], value['generation'], 'fixture-administrator', str(uuid.uuid4()))
            image = store.save(admin, mutation(image), {'objects': [] if batch_fixture else [{'id': 'fixture-car-object', 'class_id': ds['classes'][0]['id'], 'bbox': [20, 30, 120, 90]}], **({'caption': 'A car.'} if analysis_fixture else {})}, complete=True, negative=batch_fixture)
            if paid_batch_fixture:
                annotations = store.image(admin, image['id'])['annotations']
                annotations['captions'] = [{'caption': 'Imported vehicle description.', 'source': 'migration',
                    'generated_qa_pairs': [{'question': 'What is in the imported image?', 'answer': 'A vehicle.', 'raw_model_reply': 'retained fixture evidence'}]}]
                image = store.save(admin, mutation(image), annotations, complete=True)
            store.approve(admin, mutation(image))
    if getattr(request, "param", None) == "multi_dataset":
        second = store.create_dataset(admin, 'Other assigned dataset', ['Car'], str(uuid.uuid4()))
        data = io.BytesIO(); Image.new('RGB', (640, 480), (30, 80, 110)).save(data, format='PNG')
        ref = app.state.artifacts.put_bytes(data.getvalue())
        extra = store.import_image(admin, second['id'], ref.sha256, 'other-assignment.png', {'objects': []}, str(uuid.uuid4()), width=640, height=480, size_bytes=ref.size_bytes)
        store.assign(admin, second['id'], [extra['id']], users['alice']['id'], str(uuid.uuid4()), expected={extra['id']: {'expected_version': extra['current_version'], 'generation': extra['generation']}})
    if paid_batch_fixture:
        from services.hosted.providers import ProviderAccounts
        ProviderAccounts(store).configure(admin, 'openai', expected_revision=0, operation_id=str(uuid.uuid4()),
            credential='fake-local-test-key', enabled=True, monthly_limit=1000000, request_limit=10000,
            allowed_datasets=[ds['id']], models=['test-paid-model'])
        prices = {'input_microdollars_per_million': 100, 'cached_input_microdollars_per_million': 50,
                  'output_microdollars_per_million': 200, 'input_token_ceiling': 10000, 'max_output_tokens': 4096}
        pricing = tmp_path / 'paid-pricing.json'
        pricing.write_text(json.dumps({'test-paid-model': {**prices, 'batch': prices}}))
        monkeypatch.setenv('TATOR_HOSTED_PROVIDER_PRICING', str(pricing))
        monkeypatch.setenv('PYTHONPATH', str(Path(__file__).resolve().parents[1]))
    if pruning_fixture:
        from hosted_pruning_fixture import publish_pruning_fixture
        publish_pruning_fixture(app.state.tools.analysis, admin, ds, tmp_path)
    if dqe_fixture:
        from hosted_dqe_fixture import publish_dqe_fixture
        publish_dqe_fixture(app.state.tools.analysis, admin, ds, tmp_path)
    server_root = tmp_path / 'state'
    if restore_fixture:
        from services.hosted.types import MutationContext
        from services.hosted.backup import create_encrypted_backup, restore_encrypted_backup
        from services.hosted.operations import recovery_artifact_references, prepare_restored_database
        from services.hosted.errors import DomainError
        if not shutil.which('age') or not shutil.which('age-keygen'):
            pytest.skip('Real age encryption is required for restored-browser evidence')
        session = auth.login('alice', 'test-only-fixture-password', 'fixture')
        alice = auth.authenticate(session.token)
        image = store.image(alice, first_image)
        image = store.acquire_editor(alice, first_image, 'before-restore', expected_version=image['current_version'], generation=image['generation'])
        store.save(alice, MutationContext(first_image, image['current_version'], image['generation'], 'before-restore', str(uuid.uuid4())),
            {'objects': [{'id': 'retained-object', 'class_id': ds['classes'][0]['id'], 'bbox': [20, 30, 100, 80]}], 'caption': 'Draft before encrypted backup'})
        key = tmp_path / 'recovery-key'
        subprocess.run(['age-keygen', '-o', str(key)], check=True, capture_output=True)
        recipient = subprocess.run(['age-keygen', '-y', str(key)], check=True, capture_output=True, text=True).stdout.strip()
        backup = create_encrypted_backup(server_root / 'hosted.sqlite3', app.state.artifacts, tmp_path / 'recovery.tar.age', recipient=recipient, references=recovery_artifact_references)
        server_root = tmp_path / 'restored'
        restore_encrypted_backup(backup.path, server_root, identity=key, restore_hook=prepare_restored_database)
        restored_app = create_app(server_root, testing=True, origin=origin)
        assert restored_app.state.auth.instance_id != auth.instance_id
        with pytest.raises(DomainError):
            restored_app.state.auth.authenticate(session.token)
        restored_app.state.tools.close()
    env = dict(os.environ, HOSTED_TEST_ROOT=str(server_root), HOSTED_TEST_ORIGIN=origin, HOSTED_TEST_PORT=str(port))
    code = "import os,uvicorn;from hosted_app import create_app;uvicorn.run(create_app(os.environ['HOSTED_TEST_ROOT'],testing=True,origin=os.environ['HOSTED_TEST_ORIGIN']),host='127.0.0.1',port=int(os.environ['HOSTED_TEST_PORT']),log_level='warning')"
    if paid_batch_fixture:
        code = """
import os, uvicorn
from hosted_app import create_app
from test_hosted_paid_batches import Transport
app = create_app(os.environ['HOSTED_TEST_ROOT'], testing=True, origin=os.environ['HOSTED_TEST_ORIGIN'])
app.state.tools.paid_batches.transport = Transport()
uvicorn.run(app, host='127.0.0.1', port=int(os.environ['HOSTED_TEST_PORT']), log_level='warning')
"""
        if getattr(request, 'param', None) == 'paid_batch_billing':
            # Only the external transport is doubled: duplicate provider output
            # IDs exercise the real terminal-status, collector and ledger path.
            code = code.replace('app.state.tools.paid_batches.transport = Transport()', '''
class DuplicateOutput(Transport):
    def request(self, *args, **kwargs):
        value = super().request(*args, **kwargs)
        if args[2].endswith('/content'):
            return value + value
        return value
app.state.tools.paid_batches.transport = DuplicateOutput()
''')
    if caption_fixture:
        code = """
import os, json, uvicorn
from hosted_app import create_app
from services.hosted.runtime import AdapterSpec
app = create_app(os.environ['HOSTED_TEST_ROOT'], testing=True, origin=os.environ['HOSTED_TEST_ORIGIN'])
original = app.state.tools._spec
def fixture_spec(actor, dataset_id, operation):
    selected, owner = original(actor, dataset_id, operation)
    config = json.loads(json.dumps(dict(selected.config), default=lambda value: dict(value) if hasattr(value, 'items') else list(value)))
    config['fixture_gate_root'] = os.environ['HOSTED_FIXTURE_GATE']
    return AdapterSpec('test_hosted_tools_api:FixtureTools', config, 'cpu'), owner
app.state.tools._spec = fixture_spec
original_batch = app.state.tools.caption_batches._spec
def fixture_batch(actor, qwen, directory):
    selected = original_batch(actor, qwen, directory)
    return AdapterSpec('test_hosted_tools_api:FixtureCaptionBatch', selected.config, 'cpu')
app.state.tools.caption_batches._spec = fixture_batch
uvicorn.run(app, host='127.0.0.1', port=int(os.environ['HOSTED_TEST_PORT']), log_level='warning')
"""
    log = (tmp_path / 'server.log').open('w')
    process = subprocess.Popen([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[2], env=env, stdout=log, stderr=log)
    try:
        for _ in range(100):
            if process.poll() is not None: pytest.fail((tmp_path / 'server.log').read_text())
            try:
                if httpx.get(origin + '/health').status_code == 200: break
            except httpx.ConnectError: pass
            time.sleep(.05)
        else: pytest.fail('Fixture control server did not become ready')
        yield origin
    finally:
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        log.close()


async def sign_in(page, origin, username):
    from playwright.async_api import expect
    await page.goto(origin)
    await page.locator('#loginForm').get_by_label('Username', exact=True).fill(username)
    await page.locator('#loginForm').get_by_label('Password', exact=True).fill('test-only-fixture-password')
    await page.get_by_role('button', name='Sign in', exact=True).click()
    await expect(page.locator('#workspace')).to_be_visible()


async def draw(page, left, top, right, bottom):
    canvas = await page.locator('#canvas').bounding_box(); assert canvas
    await page.mouse.move(canvas['x'] + canvas['width'] * left, canvas['y'] + canvas['height'] * top)
    await page.mouse.down()
    await page.mouse.move(canvas['x'] + canvas['width'] * right, canvas['y'] + canvas['height'] * bottom, steps=8)
    await page.mouse.up()


@pytest.mark.parametrize('hosted_origin', ['analysis'], indirect=True)
def test_administrator_runs_native_statistics_on_confirmed_images(hosted_origin):
    asyncio.run(_native_statistics(hosted_origin))


async def _native_statistics(origin):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []; page.on('pageerror', lambda error: errors.append(str(error)))
        await sign_in(page, origin, 'admin')
        await page.locator('#hostedAnalysis').click()
        await expect(page.locator('#hostedAnalysisDialog')).to_be_visible()
        await page.get_by_role('button', name='Start task', exact=True).click()
        await expect(page.get_by_role('button', name='Open results', exact=True)).to_be_visible(timeout=45000)
        await page.get_by_role('button', name='Open results', exact=True).click()
        await expect(page.locator('#hostedAnalysisDialog')).to_contain_text('1 confirmed images')
        await expect(page.locator('#hostedAnalysisDialog dl')).to_contain_text('Car1')
        jobs = await (await page.request.get(origin + '/api/hosted/analysis/jobs')).json()
        assert len(jobs['jobs']) == 1 and jobs['jobs'][0]['confirmed_count'] == 1
        assert jobs['jobs'][0]['state'] == 'complete'
        assert not errors
        alice = await browser.new_page()
        await sign_in(alice, origin, 'alice')
        await expect(alice.locator('#hostedAnalysis')).to_be_hidden()
        assert (await alice.request.get(origin + '/api/hosted/analysis/jobs')).status == 403
        await browser.close()


@pytest.mark.parametrize('hosted_origin', ['analysis'], indirect=True)
def test_native_caption_export_readiness_and_protected_bundle_download(hosted_origin):
    asyncio.run(_caption_export(hosted_origin))


async def _caption_export(origin):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []; page.on('pageerror', lambda error: errors.append(str(error)))
        await sign_in(page, origin, 'admin')
        await page.locator('#hostedCaptionJobs').click()
        await page.get_by_role('button', name='Export confirmed captions for training', exact=True).click()
        dialog = page.locator('#hostedAnalysisDialog')
        await expect(dialog.get_by_label('Compute task', exact=True)).to_have_value('caption_export')
        await dialog.get_by_role('button', name='Start task', exact=True).click()
        await expect(dialog).to_contain_text('Training readiness needs attention', timeout=45000)
        await expect(dialog.get_by_role('link', name='Download caption training bundle', exact=True)).to_have_count(0)
        await dialog.get_by_label('Require readiness checks to pass before creating a training bundle', exact=True).uncheck()
        await dialog.get_by_role('button', name='Start task', exact=True).click()
        download = dialog.get_by_role('link', name='Download diagnostic bundle', exact=True)
        await expect(download).to_be_visible(timeout=45000)
        target = await download.get_attribute('href')
        response = await page.request.get(origin + target)
        assert response.status == 200 and (await response.body()).startswith(b'PK')
        alice_context = await browser.new_context()
        alice = await alice_context.new_page()
        await sign_in(alice, origin, 'alice')
        assert (await alice.request.get(origin + target)).status == 403
        assert (await page.request.get(origin + target.split('?')[0] + '?path=../other.zip')).status == 404
        assert not errors
        await browser.close()


@pytest.mark.parametrize('hosted_origin', ['analysis'], indirect=True)
def test_confirmed_split_preview_recovers_committed_reply_after_reload(hosted_origin):
    asyncio.run(_split_change(hosted_origin))


async def _split_change(origin):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
        await sign_in(page, origin, 'admin')
        await page.locator('.hosted-card input[type=checkbox]').check()
        await page.locator('#hostedSplits').click()
        dialog = page.locator('#hostedSplitsDialog')
        await dialog.get_by_label('Place selected images in', exact=True).select_option('val')
        await dialog.get_by_role('button', name='Preview split change', exact=True).click()
        await expect(dialog).to_contain_text('1 images would change split')
        operations = []
        async def lose_reply(route):
            operations.append(route.request.post_data_json)
            response = await route.fetch(); assert response.status == 200
            await route.abort('failed')
        await page.route('**/api/hosted/datasets/*/splits', lose_reply)
        await dialog.get_by_role('button', name='Apply this split change', exact=True).click()
        await expect(dialog.get_by_role('button', name='Check split change', exact=True)).to_be_visible()
        await expect(dialog).to_contain_text('connection ended before confirmation')
        assert len(operations) == 2 and operations[0] == operations[1]
        await page.unroute('**/api/hosted/datasets/*/splits', lose_reply)
        await page.reload()
        await expect(page.locator('#selectionBar')).to_be_hidden()
        await page.locator('#hostedSplitRecovery').click()
        await dialog.get_by_role('button', name='Check split change', exact=True).click()
        await expect(dialog).not_to_be_visible()
        image_id = next(iter(operations[0]['assignments']))
        saved = await (await page.request.get(origin + '/api/hosted/images/' + image_id)).json()
        assert saved['split'] == 'val'
        assert saved['generation'] == operations[0]['expected'][image_id]['generation'] + 1
        assert await page.evaluate("Object.keys(sessionStorage).filter(k => k.startsWith('tator.hosted.splits:')).length") == 0
        await browser.close()


@pytest.mark.parametrize('hosted_origin', ['restore'], indirect=True)
def test_restored_encrypted_installation_continues_real_annotation(hosted_origin):
    asyncio.run(_restored_annotation(hosted_origin))


async def _restored_annotation(origin):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
        await sign_in(page, origin, 'alice')
        await page.locator('.hosted-card a').first.click()
        await expect(page.locator('#hostedSave')).to_be_enabled()
        before = await page.evaluate('window.TatorHostedCanvas.snapshot()')
        assert before['caption'] == 'Draft before encrypted backup'
        assert before['objects'][0]['id'] == 'retained-object'
        await draw(page, .4, .45, .65, .7)
        await page.locator('#hostedSave').click()
        await expect(page.locator('#hostedSaveState')).to_have_text('All changes saved')
        await page.locator('#hostedComplete').check()
        await page.locator('#hostedSubmit').click()
        await expect(page).to_have_url(origin + '/')
        admin = await browser.new_page()
        await sign_in(admin, origin, 'admin')
        rows = await (await admin.request.get(origin + '/api/hosted/images?state=submitted')).json()
        assert len(rows['images']) == 1
        record = await (await admin.request.get(origin + '/api/hosted/images/' + rows['images'][0]['id'])).json()
        assert len(record['annotations']['objects']) == 2
        assert record['annotations']['caption'] == before['caption']
        await browser.close()


@pytest.mark.parametrize('hosted_origin', ['pruning'], indirect=True)
def test_existing_pruning_controls_use_hosted_snapshot(hosted_origin):
    asyncio.run(_hosted_pruning(hosted_origin))


async def _hosted_pruning(origin):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []; page.on('pageerror', lambda error: errors.append(str(error)))
        await sign_in(page, origin, 'admin')
        await page.locator('#hostedAnalysis').click()
        await page.get_by_role('button', name='Open results', exact=True).click()
        await page.get_by_role('link', name='Open Dataset pruning →', exact=True).click()
        await expect(page.locator('[data-pruning="work"]')).to_be_visible()
        await expect(page.locator('[data-pruning="policyBudget"]')).to_be_enabled()
        await page.locator('[data-pruning="policyBudget"]').click()
        await page.locator('[data-pruning="planObjectWeight"]').fill('0.8')
        await page.locator('[data-pruning="planObjectWeight"]').dispatch_event('input')
        await expect(page.locator('[data-pruning="applyPlan"]')).to_be_enabled(timeout=15000)
        await page.locator('[data-pruning="applyPlan"]').click()
        await expect(page.locator('[data-pruning="status"]')).to_contain_text('Exact preview applied', timeout=15000)
        assert not errors
        await browser.close()


@pytest.mark.parametrize('hosted_origin', ['caption_batch'], indirect=True)
def test_admin_caption_batch_review_applies_only_the_selected_proposal(hosted_origin):
    asyncio.run(_caption_batch_review(hosted_origin))


async def _caption_batch_review(origin):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []; page.on('pageerror', lambda error: errors.append(str(error)))
        await sign_in(page, origin, 'admin')
        await page.locator('#hostedCaptionJobs').click()
        await page.get_by_role('button', name='Run pilot', exact=True).click()
        await expect(page.get_by_role('link', name='Review on image', exact=True)).to_be_visible(timeout=30000)
        await page.get_by_role('link', name='Review on image', exact=True).click()
        await expect(page.locator('#hostedUseCaption')).to_be_enabled()
        image_id = await page.evaluate('window.TatorHosted.imageId')
        before = await (await page.request.get(origin + '/api/hosted/images/' + image_id)).json()
        assert not before['annotations'].get('caption')
        await page.locator('#hostedUseCaption').click()
        await expect(page.locator('#hostedSaveState')).to_have_text('Caption saved · earlier captions preserved')
        after = await (await page.request.get(origin + '/api/hosted/images/' + image_id)).json()
        assert after['annotations']['caption'] == 'A red square on a plain surface.'
        assert after['annotations']['objects'] == before['annotations']['objects']
        assert after['annotations']['alternate_captions'][0]['evidence_sha256']
        assert not errors
        await browser.close()


@pytest.mark.parametrize('hosted_origin', ['paid_batch'], indirect=True)
def test_remote_paid_batch_preview_and_explicit_canvas_apply(hosted_origin):
    asyncio.run(_paid_batch_review(hosted_origin))


async def _paid_batch_review(origin):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []; page.on('pageerror', lambda error: errors.append(str(error)))
        await sign_in(page, origin, 'admin')
        await page.locator('#hostedPaidBatchJobs').click()
        dialog = page.locator('#hostedPaidBatchDialog')
        await dialog.get_by_label('QA pairs per image').fill('2')
        await dialog.get_by_role('button', name='Freeze inputs and preview cost limit', exact=True).click()
        await expect(dialog.locator('pre')).to_contain_text('Generate exactly 2 question-answer pairs')
        await dialog.get_by_role('button', name='Submit this frozen batch', exact=True).click()
        await expect(dialog.get_by_role('link', name='Review on image', exact=True)).to_be_visible(timeout=30000)
        await dialog.get_by_role('link', name='Review on image', exact=True).click()
        await expect(page.locator('#hostedSave')).to_be_enabled()
        await expect(page.locator('#hostedPaidBatchDialog').get_by_role('button', name='Use this caption', exact=True)).to_be_visible()
        image_id = await page.evaluate('window.TatorHosted.imageId')
        before = await (await page.request.get(origin + '/api/hosted/images/' + image_id)).json()
        assert not before['annotations'].get('caption')
        await page.locator('#hostedPaidBatchDialog').get_by_role('button', name='Use this caption', exact=True).click()
        await expect(page.locator('#hostedSaveState')).to_have_text('Caption saved · earlier captions preserved')
        after = await (await page.request.get(origin + '/api/hosted/images/' + image_id)).json()
        assert after['annotations']['caption'] == 'A red car on a paved road.'
        assert after['annotations']['objects'] == before['annotations']['objects']
        assert len(after['annotations']['alternate_captions'][-1]['generated_qa_pairs']) == 2
        await page.locator('#hostedPaidBatchDialog').get_by_role('button', name='Close', exact=True).click()
        await page.locator('#hostedReviewInstructions').click()
        review = page.locator('#hostedInstructionReviewDialog')
        await expect(review).to_contain_text('What detail 0 is visible?')
        await expect(review).to_contain_text('A red car.')
        await review.get_by_label('Review decision for caption:0', exact=True).select_option('accepted')
        await review.get_by_label('Review decision for qa:0:top:0', exact=True).select_option('accepted')
        await review.get_by_label('Review note for qa:0:top:0', exact=True).fill('Checked the car in the image')
        await review.get_by_label('Review decision for qa:0:top:1', exact=True).select_option('rejected')
        await review.get_by_label('Review decision for captions:qa:0:top:0', exact=True).select_option('accepted')
        await review.locator('#hostedSaveInstructionReview').click()
        await expect(review).to_contain_text('Review saved.')
        reviewed = await (await page.request.get(origin + '/api/hosted/images/' + image_id)).json()
        language = reviewed['annotations']['alternate_captions'][-1]
        assert language['generated_qa_pairs'][0]['metadata']['review_decision'] == 'accepted'
        assert language['generated_qa_pairs'][1]['review_status'] == 'rejected'
        assert language['evidence'] == after['annotations']['alternate_captions'][-1]['evidence']
        assert reviewed['annotations']['captions'][0]['generated_qa_pairs'][0]['review_status'] == 'accepted'
        await review.get_by_role('button', name='Close', exact=True).click()
        # Canvas normalization plus a genuine later annotation save must not
        # strip generated QA, human decisions, or provider evidence.
        await draw(page, .6, .6, .8, .8)
        await page.locator('#hostedSave').click()
        await expect(page.locator('#hostedSaveState')).to_have_text('All changes saved')
        later = await (await page.request.get(origin + '/api/hosted/images/' + image_id)).json()
        assert len(later['annotations']['objects']) == len(after['annotations']['objects']) + 1
        assert {key: later['annotations']['alternate_captions'][-1][key] for key in language} == language
        assert later['annotations']['captions'] == reviewed['annotations']['captions']
        assert not errors
        await browser.close()


@pytest.mark.parametrize('hosted_origin', ['paid_batch_billing'], indirect=True)
def test_admin_can_reconcile_terminal_batch_after_output_failure(hosted_origin):
    asyncio.run(_terminal_batch_billing(hosted_origin))


async def _terminal_batch_billing(origin):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []; page.on('pageerror', lambda error: errors.append(str(error)))
        await sign_in(page, origin, 'admin')
        await page.locator('#hostedPaidBatchJobs').click()
        batch = page.locator('#hostedPaidBatchDialog')
        await batch.get_by_role('button', name='Freeze inputs and preview cost limit', exact=True).click()
        async with page.expect_response(lambda r: r.request.method == 'POST' and r.url.endswith('/paid-batches')) as submitted:
            await batch.get_by_role('button', name='Submit this frozen batch', exact=True).click()
        job_id = (await (await submitted.value).json())['job_id']
        for _ in range(100):
            job = await (await page.request.get(origin + '/api/hosted/tools/paid-batches/' + job_id)).json()
            if job['status'] == 'failed': break
            await asyncio.sleep(.1)
        assert job['status'] == 'failed', job
        calls = (await (await page.request.get(origin + '/api/hosted/providers/calls?unresolved_only=true')).json())['calls']
        assert len(calls) == 1 and calls[0]['state'] == 'unknown' and calls[0]['can_reconcile']
        assert calls[0]['batch_job_id'] == job_id
        call_id = calls[0]['id']
        await batch.get_by_role('button', name='Close', exact=True).click()
        await page.locator('#hostedProviders').click()
        billing = page.locator('#hostedProvidersDialog')
        row = billing.locator(f'[data-provider-call="{call_id}"]')
        await expect(row).to_contain_text('The batch has finished; its bill still needs confirmation.')
        await expect(row.get_by_role('button', name='Record confirmed charge')).to_be_enabled()
        replies = iter(['0.002345', 'Checked the retained provider bill in this fixture'])
        page.on('dialog', lambda prompt: prompt.accept(next(replies)))
        await row.get_by_role('button', name='Record confirmed charge').click()
        await expect(row).to_have_count(0)
        calls = (await (await page.request.get(origin + '/api/hosted/providers/calls')).json())['calls']
        assert len(calls) == 1 and calls[0]['state'] == 'reconciled' and calls[0]['charged'] == 2345
        assert not (await (await page.request.get(origin + '/api/hosted/providers/calls?unresolved_only=true')).json())['calls']
        assert not errors
        await browser.close()


@pytest.mark.parametrize('hosted_origin', ['multi_dataset'], indirect=True)
def test_real_canvas_three_users_and_review(hosted_origin):
    asyncio.run(_three_users(hosted_origin))


async def _three_users(origin):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        contexts = [await browser.new_context(viewport={'width': 1440, 'height': 1000}) for _ in range(3)]
        admin, alice, bob = [await context.new_page() for context in contexts]
        errors = []
        for page, name in zip((admin, alice, bob), ('admin', 'alice', 'bob')):
            page.on('pageerror', lambda error: errors.append(str(error)))
            await sign_in(page, origin, name)
        await alice.locator('#datasetChoice').select_option(label='Hosted fixture')
        await expect(alice.locator('.hosted-card')).to_contain_text('frame-0.png')
        await alice.locator('.hosted-card a').first.click()
        await expect(alice.locator('#hostedSave')).to_be_enabled()
        image_url = alice.url
        image_id = image_url.split('image=')[1]
        assert (await contexts[2].request.get(origin + f'/api/hosted/images/{image_id}/media')).status == 404
        await admin.goto(image_url); await expect(admin.locator('#hostedSave')).to_be_disabled()
        await expect(admin.locator('#hostedOverride')).to_be_visible()
        await draw(alice, .22, .22, .42, .44)
        await alice.get_by_role('button', name='Save now', exact=True).click()
        await expect(alice.locator('#hostedSaveState')).to_have_text('All changes saved')
        drawn = await alice.evaluate('window.TatorHostedCanvas.snapshot()')
        assert len(drawn['objects']) == 1
        await alice.locator('#hostedComplete').check(); await alice.locator('#hostedSubmit').click()
        await expect(alice).to_have_url(origin + '/')
        await expect(alice.locator('#datasetChoice option:checked')).to_have_text('Other assigned dataset')
        await expect(alice.locator('#submissionList')).to_contain_text('frame-0.png')
        await expect(alice.locator('#submissionList')).to_contain_text('Awaiting review')
        await expect(alice.locator('#submissionList')).to_contain_text('Hosted fixture')
        await admin.goto(image_url); await expect(admin.locator('#hostedReturn')).to_be_enabled()
        await admin.locator('#hostedFeedback').fill('Please check the edge of the car.')
        await admin.locator('#hostedReturn').click(); await expect(admin).to_have_url(origin + '/')
        await alice.get_by_role('button', name='Refresh', exact=True).click()
        await alice.locator('#datasetChoice').select_option(label='Hosted fixture')
        await expect(alice.locator('.hosted-card')).to_contain_text('frame-0.png')
        await alice.locator('.hosted-card a').first.click()
        await expect(alice.locator('#hostedSubmit')).to_be_enabled()
        await expect(alice.locator('.hosted-review-note')).to_contain_text('Please check the edge')
        await alice.locator('#hostedComplete').check(); await alice.locator('#hostedSubmit').click(); await expect(alice).to_have_url(origin + '/')
        await admin.goto(image_url); await expect(admin.locator('#hostedApprove')).to_be_enabled()
        await admin.locator('#hostedComplete').check(); await admin.locator('#hostedApprove').click(); await expect(admin).to_have_url(origin + '/')
        record = await (await contexts[0].request.get(origin + f'/api/hosted/images/{image_id}')).json()
        assert record['state'] == 'confirmed'
        assert record['annotations']['objects'] == drawn['objects']
        await alice.get_by_role('button', name='Refresh', exact=True).click()
        await expect(alice.locator('#datasetChoice option:checked')).to_have_text('Other assigned dataset')
        await expect(alice.locator('#submissionList')).to_contain_text('Approved')
        assert not errors
        await browser.close()


def test_new_edits_survive_old_reply_and_lost_submission_reply(hosted_origin):
    asyncio.run(_network_ordering(hosted_origin))


@pytest.mark.parametrize('recovery_case', ['correct', 'delayed', 'reload', 'reload_invalid', 'storage_failure', 'untrusted_rejection'])
def test_rejected_save_preserves_latest_draft_and_retry_identity(hosted_origin, recovery_case):
    asyncio.run(_rejected_save(hosted_origin, recovery_case))


async def _rejected_save(origin, recovery_case):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('dialog', lambda dialog: dialog.accept())
        await sign_in(page, origin, 'alice')
        await page.locator('.hosted-card a').first.click()
        await expect(page.locator('#hostedSave')).to_be_enabled()
        await page.clock.install()
        await page.clock.pause_at(datetime.now() + timedelta(milliseconds=50))
        await draw(page, .2, .2, .45, .45)
        await page.locator('#hostedNegative').check()
        sent, started, release = [], asyncio.Event(), asyncio.Event()

        async def intercept(route):
            sent.append(route.request.post_data_json)
            response = await route.fetch()
            if len(sent) == 1:
                assert response.status == 422
                if recovery_case in {'delayed', 'storage_failure'}:
                    started.set()
                    await release.wait()
                if recovery_case == 'untrusted_rejection':
                    # HTTP status alone is not proof that the exact command
                    # rolled back (a proxy can generate this response).
                    await route.fulfill(status=422, json={'error': 'negative_has_objects'})
                    return
            await route.fulfill(response=response)

        await page.route('**/api/hosted/images/*/save', intercept)
        save = """async () => {try {return {ok:await window.TatorHosted.save({manual:true})};}
            catch(e){return {ok:false,code:e.code,message:e.message};}}"""
        first_task = asyncio.create_task(page.evaluate(save))
        if recovery_case in {'delayed', 'storage_failure'}:
            await asyncio.wait_for(started.wait(), 10)
            await page.locator('#hostedNegative').uncheck()
            await draw(page, .55, .45, .75, .65)
            if recovery_case == 'storage_failure':
                await page.evaluate("""() => {window.originalSetItem=Storage.prototype.setItem;
                    Storage.prototype.setItem=function(){throw new DOMException('Fixture quota','QuotaExceededError');};}""")
            release.set()
        first = await first_task
        assert not first['ok']
        if recovery_case == 'storage_failure':
            assert 'Keep this tab open' in first['message']
            await page.evaluate('() => { Storage.prototype.setItem=window.originalSetItem; }')
        elif recovery_case not in {'delayed', 'untrusted_rejection'}:
            assert first['code'] == 'negative_has_objects'
        if recovery_case in {'reload', 'reload_invalid'}:
            journal = await page.evaluate("JSON.parse(sessionStorage.getItem(Object.keys(sessionStorage).find(k=>k.startsWith('tator.hosted.draft:'))))")
            assert journal['schema'] == 2 and journal['state'] == 'rejected' and not journal.get('pending')
            # Keep the corrected latest draft while the rejected command itself
            # is already resolved. Closing the page must not resurrect it.
            if recovery_case == 'reload':
                await page.locator('#hostedNegative').uncheck()
            await page.evaluate("window.dispatchEvent(new Event('beforeunload'))")
            await page.clock.resume()
            await page.reload()
            await expect(page.locator('#hostedResolve')).to_be_visible()
            await page.locator('#hostedResolve').click()
            if recovery_case == 'reload_invalid':
                await expect(page.locator('#hostedSaveState')).to_contain_text('Clear the empty-image check')
                await page.clock.pause_at(datetime.now() + timedelta(milliseconds=50))
                await page.locator('#hostedNegative').uncheck()
                await page.locator('#hostedComplete').check()
                await page.evaluate("window.dispatchEvent(new Event('beforeunload'))")
                latest = await page.evaluate("JSON.parse(sessionStorage.getItem(Object.keys(sessionStorage).find(k=>k.startsWith('tator.hosted.draft:')))).latest")
                assert not latest['negative'] and latest['complete']
                assert (await page.evaluate(save))['ok']
            await expect(page.locator('#hostedSaveState')).to_have_text('All changes saved')
        else:
            if recovery_case == 'correct':
                # Repeating an unchanged invalid draft does not send another
                # autosave or create a new operation every two seconds.
                assert not (await page.evaluate(save))['ok']
                assert len(sent) == 1
            await page.locator('#hostedNegative').uncheck()
            second = await page.evaluate(save)
            if recovery_case == 'untrusted_rejection':
                assert not second['ok'] and sent[0] == sent[1]
                # Only the second, actual server rejection retires the command.
                second = await page.evaluate(save)
            assert second['ok'], second
        assert sent[-1]['operation_id'] != sent[0]['operation_id']
        assert not sent[-1]['negative']
        stored = await page.evaluate("window.TatorHosted.api('/api/hosted/images/'+window.TatorHosted.imageId)")
        assert len(stored['annotations']['objects']) == (2 if recovery_case in {'delayed', 'storage_failure'} else 1)
        assert not stored['negative']
        await browser.close()


@pytest.mark.parametrize('replace_lease', [False, True])
def test_reload_retries_unknown_save_before_restoring_draft(hosted_origin, replace_lease):
    asyncio.run(_reload_unknown_save(hosted_origin, replace_lease))


async def _reload_unknown_save(origin, replace_lease):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('dialog', lambda dialog: dialog.accept())
        await sign_in(page, origin, 'alice')
        await page.locator('.hosted-card a').first.click()
        await expect(page.locator('#hostedSave')).to_be_enabled()
        await page.clock.install()
        await page.clock.pause_at(datetime.now() + timedelta(milliseconds=50))
        await draw(page, .2, .2, .45, .45)
        sent, replies = [], []

        async def intercept(route):
            sent.append(route.request.post_data_json)
            # Simulate a proxy failing before delivery; the browser cannot
            # know whether the server committed. Fail the first recovery too.
            if len(sent) <= 2:
                await route.fulfill(status=503, json={'error': 'fixture_connection_lost'})
                return
            response = await route.fetch()
            replies.append(response.status)
            await route.fulfill(response=response)

        await page.route('**/api/hosted/images/*/save', intercept)
        await page.evaluate("async () => {try {await window.TatorHosted.save({manual:true});} catch (_) {}}")
        await draw(page, .55, .45, .75, .65)
        await page.evaluate("window.dispatchEvent(new Event('beforeunload'))")
        if replace_lease:
            await page.evaluate("""async () => {
                const path='/api/hosted/images/'+window.TatorHosted.imageId;
                const record=await window.TatorHosted.api(path);
                await window.TatorHosted.api(path+'/editor', {editor_id:'replacement-tab',
                    expected_version:record.current_version,generation:record.generation,takeover:true});
            }""")
        await page.clock.resume()
        await page.reload()
        await expect(page.locator('#hostedResolve')).to_be_visible()
        await page.locator('#hostedResolve').click()
        await expect(page.locator('#hostedSaveState')).to_contain_text('fixture_connection_lost')
        journal = await page.evaluate("JSON.parse(sessionStorage.getItem(Object.keys(sessionStorage).find(k=>k.startsWith('tator.hosted.draft:'))))")
        assert journal['pending']['body'] == sent[0] == sent[1]
        assert len(journal['latest']['annotations']['objects']) == 2
        await page.locator('#hostedResolve').click()
        await expect(page.locator('#hostedSaveState')).to_have_text('All changes saved')
        assert sent[2] == sent[0]
        assert replies == ([409, 200] if replace_lease else [200, 200])
        assert sent[-1]['operation_id'] != sent[0]['operation_id']
        assert sent[-1]['generation'] > sent[0]['generation']
        stored = await page.evaluate("window.TatorHosted.api('/api/hosted/images/'+window.TatorHosted.imageId)")
        assert len(stored['annotations']['objects']) == 2
        await browser.close()


@pytest.mark.parametrize('image_format', ['JPEG', 'PNG', 'WEBP', 'TIFF'])
def test_encoded_media_grid_matches_canvas_and_worker(hosted_origin, tmp_path, image_format):
    image = Image.new('RGB', (640, 480), 'red')
    image.paste(Image.new('RGB', (320, 480), 'blue'), (320, 0))
    exif = image.getexif(); exif[274] = 6
    source = tmp_path / ('oriented.' + {'JPEG': 'jpg', 'PNG': 'png', 'WEBP': 'webp', 'TIFF': 'tiff'}[image_format])
    image.save(source, format=image_format, exif=exif)
    asyncio.run(_encoded_media(hosted_origin, source))


async def _encoded_media(origin, source):
    from playwright.async_api import async_playwright, expect
    from services.hosted.adapters import _image
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
        await sign_in(page, origin, 'admin')
        await page.get_by_role('button', name='Incoming', exact=True).click()
        await page.locator('#uploadInput').set_input_files(str(source))
        await expect(page.locator('#notice')).to_contain_text('1 new images added')
        rows = await page.evaluate("window.TatorHostedDashboard.api('/images?dataset_id='+document.getElementById('datasetChoice').value)")
        record = next(row for row in rows['images'] if row['filename'] == source.name)
        assert record['media_hash'] == original_hash
        await page.goto(f"{origin}/annotate?image={record['id']}")
        await expect(page.locator('#hostedSave')).to_be_enabled()
        working = await page.context.request.get(f"{origin}/api/hosted/images/{record['id']}/media")
        assert working.status == 200
        raw = await working.body()
        working_path = source.parent / 'native-working.png'; working_path.write_bytes(raw)
        native, _ = _image({'media_path': str(working_path), 'media_sha256': hashlib.sha256(raw).hexdigest()},
                           {'media_root': str(source.parent)})
        await page.clock.install(); await page.clock.pause_at(datetime.now() + timedelta(milliseconds=50))
        await draw(page, .5, .16, .57, .25)
        draft = await page.evaluate('window.TatorHostedCanvas.snapshot()')
        box = draft['objects'][0]['bbox']
        center = [int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)]
        displayed = await page.evaluate("""async ([id,x,y]) => {const im=new Image();
            im.src='/api/hosted/images/'+id+'/media';await im.decode();
            const c=document.createElement('canvas');c.width=im.naturalWidth;c.height=im.naturalHeight;
            const ctx=c.getContext('2d');ctx.drawImage(im,0,0);
            return {width:c.width,height:c.height,pixel:[...ctx.getImageData(x,y,1,1).data].slice(0,3)};}""", [record['id'], *center])
        assert (displayed['width'], displayed['height']) == (640, 480) == native.size
        assert displayed['pixel'] == list(native.getpixel(tuple(center)))
        # Stay within the displayed landscape image, excluding canvas padding.
        await draw(page, .35, .40, .55, .55)
        assert await page.evaluate('window.TatorHosted.save({manual:true})')
        original = await page.context.request.get(f"{origin}/api/hosted/images/{record['id']}/original")
        assert original.status == 200 and hashlib.sha256(await original.body()).hexdigest() == original_hash
        assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
        native.close()
        await browser.close()


@pytest.mark.parametrize('mode', ['transparent', '16bit'])
def test_special_pixel_modes_match_browser_and_native(hosted_origin, tmp_path, mode):
    source = tmp_path / (mode + '.png')
    if mode == 'transparent':
        picture = Image.new('RGBA', (640, 480), (255, 0, 0, 0))
        picture.paste((0, 0, 255, 255), (320, 0, 640, 480))
    else:
        picture = Image.new('I;16', (640, 480), 1000)
    picture.save(source)
    asyncio.run(_special_pixel_modes(hosted_origin, source, mode))


async def _special_pixel_modes(origin, source, mode):
    from playwright.async_api import async_playwright, expect
    from services.hosted.adapters import _image
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
        await sign_in(page, origin, 'admin')
        await page.get_by_role('button', name='Incoming', exact=True).click()
        await page.locator('#uploadInput').set_input_files(str(source))
        await expect(page.locator('#notice')).to_contain_text('1 new images added')
        rows = await page.evaluate("window.TatorHostedDashboard.api('/images?dataset_id='+document.getElementById('datasetChoice').value)")
        image = next(row for row in rows['images'] if row['filename'] == source.name)
        await page.goto(origin + '/annotate?image=' + image['id'])
        await expect(page.locator('#hostedSave')).to_be_enabled()
        # Native consumers receive the captured working bytes, not an unsafe
        # original without a pixel-policy descriptor. Use the real served view.
        working = await page.context.request.get(origin + '/api/hosted/images/' + image['id'] + '/media')
        assert working.status == 200
        raw = await working.body()
        working_path = source.parent / 'native-working.png'
        working_path.write_bytes(raw)
        native, _ = _image({'media_path': str(working_path), 'media_sha256': hashlib.sha256(raw).hexdigest()},
                           {'media_root': str(source.parent)})
        samples = await page.evaluate("""async id => {
            const im = new Image(); im.src='/api/hosted/images/'+id+'/media'; await im.decode();
            const c = document.createElement('canvas'); c.width=im.naturalWidth;c.height=im.naturalHeight;
            const ctx=c.getContext('2d');ctx.drawImage(im,0,0);
            return [160,480].map(x=>[...ctx.getImageData(x,240,1,1).data]);
        }""", image['id'])
        assert samples == [list(native.getpixel((x, 240))) + [255] for x in (160, 480)]
        assert samples[0] == ([255, 255, 255, 255] if mode == 'transparent' else [3, 3, 3, 255])
        original = await page.context.request.get(origin + '/api/hosted/images/' + image['id'] + '/original')
        assert original.status == 200 and hashlib.sha256(await original.body()).hexdigest() == digest
        native.close()
        await page.screenshot(path=str(source.parent / (mode + '-canvas.png')))
        await browser.close()


@pytest.mark.parametrize('case', ['unsent_save', 'rejected_save', 'unsent_reload', 'rejected_reload', 'clear_failure'])
def test_undo_to_saved_canvas_retires_only_obsolete_draft(hosted_origin, case):
    asyncio.run(_undo_obsolete_draft(hosted_origin, case))


async def _undo_obsolete_draft(origin, case):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('dialog', lambda dialog: dialog.accept())
        await sign_in(page, origin, 'alice')
        await page.locator('.hosted-card a').first.click()
        await expect(page.locator('#hostedSave')).to_be_enabled()
        await page.clock.install(); await page.clock.pause_at(datetime.now() + timedelta(milliseconds=50))
        original = await page.evaluate('window.TatorHostedCanvas.snapshot()')
        await draw(page, .2, .2, .45, .45)
        if case.startswith('rejected'):
            await page.locator('#hostedNegative').check()
            code = await page.evaluate("async()=>{try{await window.TatorHosted.save({manual:true});}catch(e){return e.code;}}")
            assert code == 'negative_has_objects'
            await page.locator('#hostedNegative').uncheck()
        else:
            await page.evaluate("window.dispatchEvent(new Event('beforeunload'))")
        await page.evaluate('document.activeElement.blur()')
        await page.keyboard.press('q')
        assert await page.evaluate('window.TatorHostedCanvas.snapshot()') == original
        if case == 'clear_failure':
            await page.evaluate("""() => {
                const remove = Storage.prototype.removeItem;
                window.failDraftClear = true;
                Storage.prototype.removeItem = function(key) {
                    if (window.failDraftClear && key.startsWith('tator.hosted.draft:')) throw new Error('fixture storage failure');
                    return remove.call(this, key);
                };
            }""")
            message = await page.evaluate("async()=>{try{await window.TatorHosted.save({manual:true});}catch(e){return e.message;}}")
            assert 'could not clear' in message
            await expect(page.locator('#hostedSaveState')).not_to_have_text('All changes saved')
            assert await page.evaluate("Object.keys(sessionStorage).some(k=>k.startsWith('tator.hosted.draft:'))")
            await page.evaluate('window.failDraftClear = false')
        if not case.endswith('reload'):
            assert await page.evaluate('window.TatorHosted.save({manual:true})')
            await expect(page.locator('#hostedSaveState')).to_have_text('All changes saved')
            assert not await page.evaluate("Object.keys(sessionStorage).some(k=>k.startsWith('tator.hosted.draft:'))")
        await page.clock.resume(); await page.reload()
        await page.wait_for_function("window.TatorHostedCanvas?.ready() && (document.getElementById('hostedResolve') || window.TatorHosted?.editable)")
        assert await page.locator('#hostedResolve').count() == 0
        await expect(page.locator('#hostedSave')).to_be_enabled()
        assert await page.evaluate('window.TatorHostedCanvas.snapshot()') == original
        await browser.close()


@pytest.mark.parametrize('hosted_origin', ['caption_records'], indirect=True)
def test_hosted_caption_projection_preserves_original_records(hosted_origin):
    asyncio.run(_caption_projection(hosted_origin))


async def _caption_projection(origin):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
        await sign_in(page, origin, 'admin')
        datasets = (await (await page.context.request.get(origin + '/api/hosted/datasets')).json())['datasets']
        images = (await (await page.context.request.get(origin + '/api/hosted/images?dataset_id=' + datasets[0]['id'])).json())['images']
        target = next(row for row in images if row['filename'] == 'frame-1.png')
        url = origin + '/api/hosted/images/' + target['id']
        before = await (await page.context.request.get(url)).json()
        await page.goto(origin + '/annotate?image=' + target['id'])
        await expect(page.locator('#hostedSave')).to_be_enabled()
        await page.wait_for_function('window.TatorHostedCanvas.ready()')
        assert await page.evaluate('window.TatorHostedCanvas.snapshot()') == before['annotations']
        assert await page.evaluate('window.TatorHosted.save({manual:true})')
        unchanged = await (await page.context.request.get(url)).json()
        assert unchanged['current_version'] == before['current_version']
        await page.locator('#qwenCaptionOutput').evaluate("e => { for(let p=e.parentElement;p;p=p.parentElement) if(p.tagName==='DETAILS') p.open=true; }")
        buttons = page.locator('#qwenCaptionAlternates button')
        await expect(page.locator('#qwenCaptionAlternates')).not_to_contain_text('Retained inactive.')
        await expect(page.locator('#qwenCaptionAlternates')).not_to_contain_text('Do not resurrect')
        await buttons.filter(has_text='Alias from imported captions.').click()
        await page.locator('#qwenCaptionOutput').fill('Edited alias from imported captions.')
        await page.locator('#qwenCaptionUpdateSelected').click()
        await expect(page.locator('#hostedSaveState')).to_have_text('All changes saved')
        saved = await (await page.context.request.get(url)).json()
        edited = saved['annotations']['captions'][0]
        assert edited['text'] == 'Edited alias from imported captions.' and 'caption' not in edited
        assert edited['id'] == 'shared-id' and edited['metadata'] == before['annotations']['captions'][0]['metadata']
        assert saved['annotations']['captions'][1:] == before['annotations']['captions'][1:]
        assert saved['annotations']['alternate_captions'] == before['annotations']['alternate_captions']
        await page.reload(); await expect(page.locator('#hostedSave')).to_be_enabled()
        await page.wait_for_function('window.TatorHostedCanvas.ready()')
        assert await page.evaluate('window.TatorHostedCanvas.snapshot()') == saved['annotations']
        await page.locator('#qwenCaptionOutput').evaluate("e => { for(let p=e.parentElement;p;p=p.parentElement) if(p.tagName==='DETAILS') p.open=true; }")
        await page.locator('#qwenCaptionAlternates button').filter(has_text='Another alias.').click()
        await page.locator('#qwenCaptionDeleteSelected').click()
        await expect(page.locator('#hostedSaveState')).to_have_text('All changes saved')
        after = await (await page.context.request.get(url)).json()
        assert after['annotations']['captions'] == saved['annotations']['captions']
        assert after['annotations']['alternate_captions'] == saved['annotations']['alternate_captions'][1:]
        await browser.close()


def test_resumable_import_and_admin_recovery_controls(hosted_origin, tmp_path):
    paths = []
    for name, color in (("new-one.png", "magenta"), ("new-two.png", "cyan")):
        path = tmp_path / name; Image.new("RGB", (640, 480), color).save(path); paths.append(str(path))
    asyncio.run(_imports_and_recovery(hosted_origin, paths))


async def _imports_and_recovery(origin, paths):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1440, "height": 1000})
        page = await context.new_page(); errors = []; page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("dialog", lambda dialog: dialog.accept())
        await sign_in(page, origin, "admin")
        await page.locator("#createDataset").click()
        await page.locator('#datasetForm input[name="name"]').fill("Import recovery")
        await page.locator('#datasetForm textarea[name="classes"]').fill("Car")
        await page.locator('#datasetForm button[type="submit"]').click()
        await expect(page.locator("#datasetChoice")).not_to_be_empty()
        await expect(page.locator("#datasetDialog")).not_to_be_visible()
        await page.wait_for_function("document.querySelector('#datasetChoice').selectedOptions[0]?.text === 'Import recovery'")
        dataset_id = await page.locator("#datasetChoice").input_value()
        # Let Chromium send the actual File bytes. Fail its fetch promise after
        # the server commits; CDP upload interception omits file-part bytes.
        await page.evaluate("""() => {
            const original = window.fetch.bind(window); window.committedUploads = [];
            window.fetch = async (...args) => {
                const response = await original(...args);
                if (String(args[0]).endsWith('/incoming') && window.committedUploads.length < 2) {
                    window.committedUploads.push({status: response.status, body: await response.clone().json()});
                    throw new TypeError('Fixture: reply lost after upload reached server');
                }
                return response;
            };
        }""")
        await page.locator("#uploadInput").set_input_files(paths)
        await expect(page.locator("#resumeImport")).to_be_visible()
        responses = await page.evaluate("window.committedUploads")
        assert len(responses) == 2 and all(row["status"] == 200 for row in responses), responses
        commits = [row["body"] for row in responses]
        assert commits[0]["id"] == commits[1]["id"]
        await page.reload(); await expect(page.locator("#workspace")).to_be_visible()
        await page.locator("#datasetChoice").select_option(dataset_id)
        await expect(page.locator("#resumeImport")).to_be_visible()
        async with page.expect_file_chooser() as chooser:
            await page.locator("#resumeImport").click()
        await (await chooser.value).set_files(paths)
        await expect(page.locator("#notice")).to_contain_text("2 new images added")
        images = await (await context.request.get(f"{origin}/api/hosted/images?dataset_id={dataset_id}")).json()
        assert len(images["images"]) == 2 and commits[0]["id"] in {row["id"] for row in images["images"]}
        await page.locator("#hostedTools").click()
        await expect(page.locator("#hostedToolsDialog")).to_contain_text("No server model files have been configured yet")
        await page.locator("#hostedToolsDialog").get_by_role("button", name="Close", exact=True).click()
        await page.locator("#hostedProviders").click()
        await expect(page.locator("#hostedProvidersDialog")).to_contain_text("Monthly budget (USD)")
        await page.locator("#hostedProvidersDialog").get_by_role("button", name="Close", exact=True).click()
        await page.locator("#operations").click()
        await page.get_by_role("button", name="Archive this dataset…", exact=True).click()
        await expect(page.locator("#operationsDialog")).not_to_be_visible()
        await page.locator("#operations").click()
        await page.locator("#operationsDialog").get_by_role("button", name="Restore", exact=True).click()
        await expect(page.locator("#notice")).to_contain_text("Restored")
        after = await (await context.request.get(f"{origin}/api/hosted/images?dataset_id={dataset_id}")).json()
        assert {row["id"] for row in after["images"]} == {row["id"] for row in images["images"]}
        image_id = after['images'][0]['id']
        await page.goto(f'{origin}/annotate?image={image_id}')
        await expect(page.locator('#hostedArchive')).to_be_enabled()
        await draw(page, .15, .2, .45, .55)
        expected = await page.evaluate('window.TatorHostedCanvas.snapshot()')
        await page.locator('#hostedArchive').click()
        try:
            await expect(page).to_have_url(origin + '/')
        except AssertionError:
            pytest.fail(await page.locator('#hostedSaveState').inner_text())
        assert (await context.request.get(f'{origin}/api/hosted/images/{image_id}')).status == 404
        await page.locator('#operations').click()
        async with page.expect_response('**/api/hosted/archives/*/restore') as restored_response:
            await page.locator('#operationsDialog').get_by_role('button', name='Restore', exact=True).click()
        assert (await restored_response.value).status == 200
        restored = await (await context.request.get(f'{origin}/api/hosted/images/{image_id}')).json()
        assert restored['annotations']['objects'] == expected['objects']
        assert not errors
        await browser.close()


async def _network_ordering(origin):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1440, 'height': 1000})
        page = await context.new_page()
        await sign_in(page, origin, 'alice'); await page.locator('.hosted-card a').first.click()
        await expect(page.locator('#hostedSave')).to_be_enabled()
        started, release = asyncio.Event(), asyncio.Event()
        blocked = False
        async def hold_first(route):
            nonlocal blocked
            response = await route.fetch()
            if not blocked:
                blocked = True; started.set(); await release.wait()
            await route.fulfill(response=response)
        await page.route('**/api/hosted/images/*/save', hold_first)
        await draw(page, .12, .12, .3, .3)
        await asyncio.wait_for(started.wait(), timeout=10)
        await draw(page, .55, .45, .76, .67)
        assert len((await page.evaluate('window.TatorHostedCanvas.snapshot()'))['objects']) == 2
        release.set()
        await expect(page.locator('#hostedSaveState')).to_have_text('All changes saved', timeout=12000)
        image_id = page.url.split('image=')[1]
        stored = await (await context.request.get(origin + f'/api/hosted/images/{image_id}')).json()
        assert len(stored['annotations']['objects']) == 2
        await page.unroute('**/api/hosted/images/*/save', hold_first)
        saved_operations = []
        async def truncate_reply(route):
            saved_operations.append(route.request.post_data_json['operation_id'])
            response = await route.fetch(); assert response.status == 200
            if len(saved_operations) == 1:
                await route.fulfill(status=200, content_type='application/json', body='{"id":')
            else:
                await route.fulfill(response=response)
        await page.route('**/api/hosted/images/*/save', truncate_reply)
        await draw(page, .75, .15, .9, .3)
        await expect(page.locator('#hostedSaveState')).to_contain_text('reply was incomplete')
        assert await page.evaluate("Object.keys(sessionStorage).some(key => key.startsWith('tator.hosted.draft:'))")
        await page.locator('#hostedSave').click()
        await expect(page.locator('#hostedSaveState')).to_have_text('All changes saved')
        assert len(saved_operations) == 2 and saved_operations[0] == saved_operations[1]
        stored = await (await context.request.get(origin + f'/api/hosted/images/{image_id}')).json()
        assert len(stored['annotations']['objects']) == 3
        operation_ids = []
        async def lose_reply(route):
            operation_ids.append(route.request.post_data_json['operation_id'])
            response = await route.fetch(); assert response.status == 200
            await route.fulfill(status=200, content_type='application/json', body='{}')
        await page.route('**/api/hosted/images/*/submit', lose_reply)
        await page.locator('#hostedComplete').check(); await page.locator('#hostedSubmit').click()
        await expect(page.locator('#hostedHandoffRetry')).to_be_visible()
        await expect(page.locator('#hostedSubmit')).to_be_disabled()
        await page.locator('#hostedHandoffRetry').click(); await expect(page).to_have_url(origin + '/')
        assert len(operation_ids) == 1
        submitted = (await (await context.request.get(origin + '/api/hosted/submissions')).json())['submissions']
        assert len(submitted) == 1
        await browser.close()


def test_navigation_preserves_boxes_and_versioned_caption_alternates(hosted_origin):
    asyncio.run(_navigation_and_captions(hosted_origin))


async def _navigation_and_captions(origin):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1440, 'height': 1000})
        page = await context.new_page(); errors = []; page.on('pageerror', lambda error: errors.append(str(error)))
        await sign_in(page, origin, 'admin')
        datasets = (await (await context.request.get(origin + '/api/hosted/datasets')).json())['datasets']
        images = (await (await context.request.get(origin + '/api/hosted/images?dataset_id=' + datasets[0]['id'] + '&state=unassigned')).json())['images']
        first, second = images[:2]
        await page.goto(origin + '/annotate?image=' + first['id'])
        await expect(page.locator('#hostedSave')).to_be_enabled()
        await draw(page, .2, .2, .4, .4)
        # Reveal existing first-pane caption controls without bypassing handlers.
        await page.locator('#qwenCaptionOutput').evaluate("e => { for (let p=e.parentElement;p;p=p.parentElement) if(p.tagName==='DETAILS') p.open=true; }")
        await page.locator('#qwenCaptionOutput').fill('Two parked vehicles.')
        await page.locator('#qwenCaptionSaveAlternate').click()
        await expect(page.locator('#hostedSaveState')).to_have_text('All changes saved')
        await expect(page.locator('#qwenCaptionAlternates')).to_contain_text('Two parked vehicles.')
        saved = await (await context.request.get(origin + '/api/hosted/images/' + first['id'])).json()
        assert len(saved['annotations']['objects']) == 1
        assert any(item['caption'] == 'Two parked vehicles.' for item in saved['annotations']['alternate_captions'])
        await page.evaluate('window.navigationSentinel = "same document"')
        await page.evaluate('id => window.TatorHosted.navigate(id)', second['id'])
        assert await page.evaluate('window.navigationSentinel') == 'same document'
        assert (await page.evaluate('window.TatorHostedCanvas.snapshot()'))['objects'] == []
        await draw(page, .52, .51, .73, .74)
        await page.evaluate('id => window.TatorHosted.navigate(id)', first['id'])
        current = await page.evaluate('window.TatorHostedCanvas.snapshot()')
        assert current['objects'] == saved['annotations']['objects']
        assert current['alternate_captions'] == saved['annotations']['alternate_captions']
        other = await (await context.request.get(origin + '/api/hosted/images/' + second['id'])).json()
        assert len(other['annotations']['objects']) == 1
        assert other['annotations']['objects'] != current['objects']
        await page.locator('#hostedBack').click(); await expect(page).to_have_url(origin + '/')
        await page.goto(origin + '/annotate?image=' + first['id'])
        await expect(page.locator('#hostedSave')).to_be_enabled()
        assert (await page.evaluate('window.TatorHostedCanvas.snapshot()'))['alternate_captions'] == saved['annotations']['alternate_captions']
        assert not errors
        await browser.close()


@pytest.mark.parametrize('hosted_origin', ['caption'], indirect=True)
def test_existing_caption_button_runs_durable_private_job(hosted_origin):
    asyncio.run(_caption_button(hosted_origin))


async def _caption_button(origin):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1440, 'height': 1000})
        page = await context.new_page(); errors = []; page.on('pageerror', lambda error: errors.append(str(error)))
        await sign_in(page, origin, 'alice'); await page.locator('.hosted-card a').first.click()
        await expect(page.locator('#hostedSave')).to_be_enabled()
        await page.locator('#qwenCaptionBuildInstructionDataset').evaluate("e => { for (let p=e.parentElement;p;p=p.parentElement) if(p.tagName==='DETAILS') p.open=true; }")
        await page.locator('#qwenCaptionSubcaptionsPerImage').fill('1')
        await expect(page.locator('#qwenCaptionBuildInstructionDataset')).to_be_enabled(timeout=15000)
        await page.locator('#qwenCaptionBuildInstructionDataset').click()
        await expect(page.locator('#qwenCaptionOutput')).to_have_value('A red image.', timeout=30000)
        await page.locator('#hostedSave').click()
        await expect(page.locator('#hostedSaveState')).to_have_text('All changes saved')
        image_id = page.url.split('image=')[1]
        image = await (await context.request.get(origin + '/api/hosted/images/' + image_id)).json()
        assert image['annotations']['caption'] == 'A red image.'
        assert any(row['caption'] == 'A red image.' for row in image['annotations']['alternate_captions'])
        generated = next(row for row in image['annotations']['alternate_captions'] if row.get('backend_job_id'))
        assert generated['generated_qa_pairs'][0]['question'] == 'Which color is visible?'
        assert generated['generated_qa_pairs'][0]['validation_status'] == 'machine_validated'
        assert generated['generated_qa_pairs'][0]['metadata']['machine_validation_content_sha256']
        assert generated['caption_quality'] and generated['evidence']['sha256']
        await draw(page, .2, .2, .4, .4)
        await page.locator('#hostedSave').click()
        await expect(page.locator('#hostedSaveState')).to_have_text('All changes saved')
        later = await (await context.request.get(origin + '/api/hosted/images/' + image_id)).json()
        assert later['annotations']['alternate_captions'] == image['annotations']['alternate_captions']
        await page.reload()
        await expect(page.locator('#hostedSave')).to_be_enabled()
        reloaded = await page.evaluate('window.TatorHostedCanvas.snapshot()')
        assert reloaded['alternate_captions'] == image['annotations']['alternate_captions']
        assert not errors
        await browser.close()


@pytest.mark.parametrize('hosted_origin', ['dqe'], indirect=True)
def test_native_dqe_graph_selection_and_versioned_class_corrections(hosted_origin):
    asyncio.run(_dqe_corrections(hosted_origin))


async def _dqe_corrections(origin):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1440, 'height': 1100})
        page = await context.new_page(); errors = []; page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('dialog', lambda dialog: dialog.accept())
        await sign_in(page, origin, 'admin')
        await page.locator('#hostedAnalysis').click()
        await page.get_by_role('button', name='Open results', exact=True).click()
        graph = page.locator('.hosted-dqe-plot')
        await expect(graph.locator('.modebar')).to_be_visible()
        await graph.scroll_into_view_if_needed()
        # Use actual Plotly data-to-pixel coordinates to click a displayed dot.
        point = await graph.evaluate("e => {const r=e.getBoundingClientRect(), l=e._fullLayout; return {x:r.x+l.xaxis._offset+l.xaxis.l2p(0),y:r.y+l.yaxis._offset+l.yaxis.l2p(0)}}")
        await page.mouse.click(point['x'], point['y'])
        await expect(page.get_by_role('button', name='Save class correction')).to_be_visible()
        await page.get_by_label('Correct class', exact=True).select_option(label='Person')
        sent = []
        request_seen, release_reply = asyncio.Event(), asyncio.Event()
        async def lose_correction_reply(route):
            sent.append(route.request.post_data_json['context']['operation_id'])
            if len(sent) == 1:
                request_seen.set()
                await release_reply.wait()
            response = await route.fetch(); assert response.status == 200
            await route.abort('failed')
        await page.route('**/api/hosted/analysis/jobs/*/dqe/actions', lose_correction_reply)
        await page.get_by_role('button', name='Save class correction').click()
        await asyncio.wait_for(request_seen.wait(), 10)
        await expect(page.get_by_role('button', name='Check correction result')).to_be_disabled()
        release_reply.set()
        await expect(page.get_by_role('button', name='Check correction result')).to_be_enabled()
        await page.get_by_role('button', name='Check correction result').click()
        try:
            await expect(page.locator('.hosted-dqe')).to_contain_text('Correction confirmed')
        except AssertionError:
            pytest.fail(f"DQE recovery notice: {await page.locator('#notice').inner_text()}; browser errors: {errors}")
        assert len(sent) == 2 and sent[0] == sent[1]
        await page.unroute('**/api/hosted/analysis/jobs/*/dqe/actions', lose_correction_reply)
        ds = (await (await context.request.get(origin + '/api/hosted/datasets')).json())['datasets'][0]
        images = (await (await context.request.get(origin + '/api/hosted/images?dataset_id=' + ds['id'] + '&state=confirmed')).json())['images']
        detailed = [await (await context.request.get(origin + '/api/hosted/images/' + image['id'])).json() for image in images]
        classes = (await (await context.request.get(origin + '/api/hosted/datasets/' + ds['id'])).json())['classes']
        person = next(c['id'] for c in classes if c['name'] == 'Person')
        assert sum(i['annotations']['objects'][0]['class_id'] == person for i in detailed) == 1
        # Continue with the same frozen projection and batch-relabel both images.
        await graph.locator('[data-title="Box Select"]').click()
        bounds = await graph.evaluate("e => {const r=e.getBoundingClientRect(),s=e._fullLayout._size;return {x:r.x+s.l,y:r.y+s.t,w:s.w,h:s.h}}")
        await page.mouse.move(bounds['x']+1,bounds['y']+1); await page.mouse.down()
        await page.mouse.move(bounds['x']+bounds['w']-1,bounds['y']+bounds['h']-1, steps=15); await page.mouse.up()
        await expect(page.get_by_role('heading', name='2 selected objects')).to_be_visible()
        await page.get_by_label('New class for selected objects').select_option(label='Person')
        await page.get_by_role('button', name='Relabel selected objects').click()
        await expect(page.locator('.hosted-dqe')).to_contain_text('Corrections saved as new annotation versions')
        for image in images:
            current = await (await context.request.get(origin + '/api/hosted/images/' + image['id'])).json()
            assert current['annotations']['objects'][0]['class_id'] == person
        assert not errors
        await browser.close()
