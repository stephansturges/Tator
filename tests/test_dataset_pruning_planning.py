"""Actual planner/API/ledger composition; only external dataset admission is fake."""
import itertools
import json
import shutil
import time
import threading
from contextlib import nullcontext
from types import SimpleNamespace
import uuid

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dataset_pruning import build_dataset_pruning_router
from services.dataset_pruning import PruningConfig, build_order, digest, pair_evidence, parse_objects
from services.dataset_pruning_coverage import prepare_coverage
from services.dataset_pruning_evidence import PairStore
from services.dataset_pruning_jobs import PruningService, write_json, pruning_blas_policy
from services.dataset_pruning_planning import coverage_cache_key, coverage_directory, export_scene_groups
from services.dataset_pruning_review import ReviewLedger
from services.dataset_pruning_serving import states


def saved_replay(directory, result, source, objects):
    manifest = result['recipe']['component_manifest']
    binding = digest({'source': source['inventory_digest'], 'recipe': result['recipe'], 'manifest': manifest})
    target = directory / 'planning' / ('features-' + binding)
    target.mkdir(parents=True)
    path = target / 'objects.npy'
    np.save(path, objects, allow_pickle=False)
    write_json(target / 'commit.json', {'binding': binding,
        'file_states': [[p, list(s) if s else None] for p, s in states([path])]})


def clone_job(q):
    job = 'dp_' + 'b' * 16
    directory = q.service.root / job
    directory.mkdir()
    for name in ['source.json', 'images.json', 'result.json', 'state.json', 'scene.npy', 'components.json', 'evidence.sqlite3']:
        shutil.copy2(q.directory / name, directory / name)
    saved_replay(directory, json.loads((directory / 'result.json').read_text()),
                 json.loads((directory / 'source.json').read_text()), q.objects)
    return SimpleNamespace(client=q.client, service=q.service, job=job,
                           base='/dataset_pruning/jobs/' + job, directory=directory)


@pytest.fixture
def composed(tmp_path, request):
    config = PruningConfig(scene_threshold=.8, negative_scene_threshold=.85, object_threshold=.8,
                           rarity_scale_threshold=.6, index_threshold=.8)
    images = []
    options = getattr(request, 'param', [0] * 5)
    if isinstance(options, dict) and 'index_threshold' in options:
        config = PruningConfig(**{**vars(config), 'index_threshold': options['index_threshold']})
    class_ids = options.get('class_ids', [0] * 5) if isinstance(options, dict) else options
    negative_only = isinstance(options, dict) and options.get('negative_only')
    names = ['vehicle' if cls == 0 else f'class-{cls}' for cls in range(max(class_ids) + 1)]
    for i in range(5):
        lines = [] if negative_only else [f'{class_ids[i]} .5 .5 .2 .2']
        objects, errors = parse_objects(lines, len(names))
        if objects: objects[0]['feature_index'] = i
        images.append({'image_id': f'image-{i}', 'image_sha256': f'{i:064x}', 'width': 100, 'height': 100,
                       'image_relpath': f'image-{i}.png', 'image_path': str(tmp_path / f'image-{i}.png'),
                       'label_lines': lines, 'objects': objects, 'protection_errors': errors, 'split': 'train'})
    scene = np.array([[1., .01 * i] for i in range(5)], np.float32)
    scene /= np.linalg.norm(scene, axis=1, keepdims=True)
    objects = scene.copy()
    if negative_only: objects = np.empty((0, 0), np.float32)
    if isinstance(options, dict) and 'object_angles' in options:
        angles = np.deg2rad(options['object_angles'])
        objects = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
    source = {'identity': 'copied-source', 'revision': 1, 'inventory_digest': 'source-fixture', 'labelmap': names}
    manifest = {'schema': 'pruning-component-production-v1', 'source_images': 5, 'object_rows': len(objects),
                'components': {'scene_0': {'dimension': 2, 'weight': 1}}}
    if not negative_only: manifest['components']['bbox_fixture'] = {'dimension': 2, 'weight': 1}
    recipe = {'schema': 'copied-cached-feature-fixture', 'evidence_mode': 'compound', 'component_manifest': manifest}
    changed = {'value': False}
    def check(*_):
        if changed['value']:
            from fastapi import HTTPException
            raise HTTPException(409, 'Fixture source changed')
    adapter = SimpleNamespace(guard=lambda: nullcontext(), check_revision=check, check_inventory=check)
    service = PruningService(tmp_path / 'jobs', adapter)
    job = 'dp_' + 'a' * 16
    directory = service.root / job
    directory.mkdir()
    np.save(directory / 'scene.npy', scene, allow_pickle=False)
    evidence = PairStore(directory / 'evidence.sqlite3')
    for a, b in itertools.combinations(range(5), 2):
        evidence.append({'left': a, 'right': b, **pair_evidence(images[a], images[b], float(scene[a] @ scene[b]), objects, config)})
    evidence.finish()
    result = build_order(images, scene, evidence, config=config)
    if isinstance(options, dict) and options.get('isolated_groups'): result['groups'] = list(range(len(images)))
    result.update(recipe=recipe, config={key: getattr(config, key) for key in config.__dataclass_fields__},
                  selection_revision=1, image_count=len(images), source_revision=1, class_names=names)
    for name, value in [('state.json', {'status': 'completed'}), ('result.json', result), ('source.json', source), ('images.json', images)]:
        write_json(directory / name, value)
    write_json(directory / 'components.json', manifest)
    if not negative_only: saved_replay(directory, result, source, objects)
    index = prepare_coverage(coverage_directory(directory), images, scene, objects, config,
                             source_digest=source['inventory_digest'], recipe_digest=digest(recipe))
    key = coverage_cache_key(result, source, config, directory)
    if not negative_only: service.coverage_indexes[key] = index
    app = FastAPI(); app.include_router(build_dataset_pruning_router(lambda: service))
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, service=service, job=job, index=index, changed=changed,
                              base=f'/dataset_pruning/jobs/{job}', directory=directory, objects=objects)
    for live in {index, *service.coverage_indexes.values()}: live.close()


def settings_preview(q, **updates):
    snapshot = q.client.get(q.base + '/review').json()
    settings = {'policy': 'budgeted_coverage', 'object_weight': .9, 'max_removed_images': 2,
                'max_affected_appearance_groups': 0, 'max_lost_appearance_weight_fraction': 0.}
    settings.update(updates)
    response = q.client.post(q.base + '/plan/preview', json={'kind': 'settings', 'selection_revision': 1,
        'expected_review_revision': snapshot['review_revision'], 'expected_plan_revision': snapshot['plan_revision'], 'settings': settings})
    assert response.status_code == 200, response.text
    preview = response.json()
    for _ in range(200):
        response = q.client.get(q.base + '/plan/previews/' + preview['preview_id'])
        assert response.status_code == 200, response.text
        preview = response.json()
        if preview['status'] not in {'queued', 'running'}: break
        time.sleep(.01)
    assert preview['status'] == 'ready', preview
    assert preview['plan_projection'] == 'settings-preview-summary-v1'
    assert not {'cards', 'proposals', 'families'} & preview['plan'].keys()
    # Internal Apply/paged readers keep the complete immutable server plan.
    full = q.service.settings_preview_status(q.job, preview['preview_id'])
    assert full['preview_digest'] == preview['preview_digest']
    assert full['plan']['plan_digest'] == preview['plan']['plan_digest']
    return full


def mutate(q, action, **fields):
    state = q.client.get(q.base + '/review').json()
    payload = {'action': action, 'request_id': uuid.uuid4().hex, 'selection_revision': 1,
               'expected_review_revision': state['review_revision'], 'plan_revision': state['plan_revision'],
               'removed_count': state['budget'], **fields}
    result = q.client.post(q.base + '/review', json=payload)
    assert result.status_code == 200, result.text
    return result.json(), payload


def test_preview_apply_pin_no_refill_and_exact_retry(composed):
    q = composed
    before = q.client.get(q.base + '/review').json()
    scratch = coverage_directory(q.directory) / 'group-class-retired.npy'
    np.save(scratch, np.array([0], np.int32), allow_pickle=False)
    preview = settings_preview(q)
    assert scratch.name not in {name for name, _ in preview['plan']['accounting_files']}
    scratch.unlink()
    assert q.client.get(q.base + '/plan/previews/' + preview['preview_id']).status_code == 200
    assert q.client.get(q.base + '/review').json()['review_revision'] == before['review_revision']
    answer, payload = mutate(q, 'apply_plan', preview_id=preview['preview_id'])
    assert answer['snapshot']['approved_count'] == 0
    rows = q.client.post(q.base + '/selection', json={'selection_revision': 1, 'removed_count': 2}).json()['rows']
    assert len(rows) == 2
    kept = rows[0]['candidate_id']
    pinned, _ = mutate(q, 'pin', image_id=kept, keep=True)
    assert pinned['snapshot']['removed_count'] == 1
    assert pinned['snapshot']['active_settings']['max_removed_images'] == 2
    retried = q.client.post(q.base + '/review', json=payload)
    assert retried.status_code == 200
    assert retried.json()['snapshot']['removed_count'] == 1
    assert kept in retried.json()['snapshot']['pins']
    conflict = q.client.post(q.base + '/review', json={**payload, 'preview_id': '0' * 32})
    assert conflict.status_code == 409


def test_approval_subset_metrics_and_source_staleness(composed):
    q = composed
    preview = settings_preview(q)
    mutate(q, 'apply_plan', preview_id=preview['preview_id'])
    rows = q.client.post(q.base + '/selection', json={'selection_revision': 1, 'removed_count': 2}).json()['rows']
    row = rows[0]
    answer, _ = mutate(q, 'approve', candidate_id=row['candidate_id'], witness_id=row['witness_id'], evidence_digest=row['evidence_digest'])
    assert answer['snapshot']['approved_count'] == 1
    assert answer['snapshot']['approved_loss']['measured']
    prepared = q.service.prepare_active_export(q.job, {'selection_revision': 1,
        'expected_review_revision': answer['snapshot']['review_revision'], 'plan_revision': answer['snapshot']['plan_revision']})
    assert len(prepared['removed']) == 1
    assert prepared['metrics']['affected_appearance_group_count'] <= preview['plan']['proposed_loss']['affected_appearance_groups']
    pending = settings_preview(q)
    q.changed['value'] = True
    assert q.client.get(q.base + '/plan/previews/' + pending['preview_id']).status_code == 409


def test_strict_weight_preview_keeps_matching_gate_and_legacy_views(composed):
    q = composed
    legacy = q.client.post(q.base + '/selection', json={'selection_revision': 1, 'removed_count': 2, 'view': 'impact'})
    assert legacy.status_code == 200, legacy.text
    assert len(legacy.json()['rows']) == 2
    assert legacy.json()['rows'][0]['evidence_digest']
    preview = settings_preview(q, policy='strict_duplicates', object_weight=.25)
    assert preview['plan']['settings']['object_weight'] == .25
    assert all(not row['evidence']['left_unmatched_rows'] for row in preview['plan']['cards'])
    mutate(q, 'apply_plan', preview_id=preview['preview_id'])
    active = q.client.get(q.base + '/result').json()
    assert active['active_settings']['object_weight'] == .25
    assert active['index_weights']['objects'] == .25


@pytest.mark.parametrize('composed', [{'object_angles': [35, 0, 0, 0, 0], 'index_threshold': .87}], indirect=True)
@pytest.mark.parametrize('supplementary', [False, True])
def test_strict_weight_promotes_originally_invalid_measured_witness(composed, supplementary):
    q = composed
    saved = PairStore(q.directory / 'evidence.sqlite3')
    promoted = [saved.pair(0, i) for i in range(1, 5)]
    assert all(not edge['valid'] and not edge['left_unmatched_rows'] and not edge['right_unmatched_rows'] for edge in promoted)
    assert list(saved.edges_for(0)) == []  # Historical strict callers keep their original gate.
    assert len(list(saved.all_edges_for(0))) == 4
    if supplementary:
        with ReviewLedger(q.directory / 'review.sqlite3').transaction() as state:
            state['comparisons'] = {f"0:{edge['right']}": {'edge': edge} for edge in promoted}
    preview = settings_preview(q, policy='strict_duplicates', object_weight=.25, max_removed_images=4)
    assert preview['plan']['removed_count'] == 4
    assert preview['plan']['retained_ids'] == ['image-0']
    assert all(row['scene_witness']['image_id'] == 'image-0' for row in preview['plan']['cards'])
    assert all(row['evidence']['valid'] and row['evidence']['combined'] >= .87 for row in preview['plan']['cards'])
    mutate(q, 'apply_plan', preview_id=preview['preview_id'])
    rows = q.client.post(q.base + '/selection', json={'selection_revision': 1, 'removed_count': 4, 'view': 'impact'})
    assert rows.status_code == 200, rows.text
    assert len(rows.json()['rows']) == 4


def test_strict_compact_measurements_reuse_source_binding_and_reject_drift(composed):
    q = composed
    preview = settings_preview(q, policy='strict_duplicates', object_weight=.25)
    context, _, _ = q.service._capture_plan(q.job, check=False)
    first = q.service._strict_summary(context)
    before = first.path.stat()
    assert q.service._strict_summary(context) is first
    second = settings_preview(q, policy='strict_duplicates', object_weight=.5)
    assert second['plan']['settings']['object_weight'] == .5
    after = first.path.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) == (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    assert second['plan']['removed_ids'] == preview['plan']['removed_ids']
    with first.path.open('ab') as target: target.write(b'changed')
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error: q.service._strict_summary(context)
    assert error.value.status_code == 409


def test_pruning_blas_admission_can_cancel_without_waiting_for_owner():
    entered, release = threading.Event(), threading.Event()
    def owner():
        with pruning_blas_policy():
            entered.set()
            assert release.wait(3)
    worker = threading.Thread(target=owner)
    worker.start()
    try:
        assert entered.wait(2)
        started = time.monotonic()
        with pytest.raises(InterruptedError):
            with pruning_blas_policy(cancelled=lambda: True): pass
        assert time.monotonic() - started < 1.
    finally:
        release.set(); worker.join(3)
    assert not worker.is_alive()


def test_typed_settings_reject_boolean_and_stale_apply(composed):
    q = composed
    response = q.client.post(q.base + '/plan/preview', json={'kind': 'settings', 'selection_revision': 1,
        'expected_review_revision': 0, 'expected_plan_revision': 1, 'settings': {'max_removed_images': True}})
    assert response.status_code == 422
    preview = settings_preview(q)
    mutate(q, 'pin', image_id='image-4', keep=True)
    response = q.client.post(q.base + '/review', json={'action': 'apply_plan', 'preview_id': preview['preview_id'],
        'request_id': uuid.uuid4().hex, 'selection_revision': 1, 'expected_review_revision': 0, 'removed_count': 0})
    assert response.status_code == 409


def test_apply_undo_preserves_later_independent_pin(composed):
    q = composed
    preview = settings_preview(q)
    applied, _ = mutate(q, 'apply_plan', preview_id=preview['preview_id'])
    batch = applied['receipt']['batch_id']
    mutate(q, 'pin', image_id='image-4', keep=True)
    # The pin also owns a later active plan, so an earlier batch undo conflicts
    # atomically rather than restoring old settings over the independent edit.
    state = q.client.get(q.base + '/review').json()
    response = q.client.post(q.base + '/review', json={'action': 'bulk_undo', 'batch_id': batch,
        'request_id': uuid.uuid4().hex, 'selection_revision': 1, 'expected_review_revision': state['review_revision'],
        'plan_revision': state['plan_revision'], 'removed_count': state['budget']})
    assert response.status_code == 409
    assert q.client.get(q.base + '/review').json()['pins']['image-4']['value']


def test_family_keep_at_least_preview_bulk_capture_and_undo(composed):
    q = composed
    preview = settings_preview(q)
    mutate(q, 'apply_plan', preview_id=preview['preview_id'])
    state = q.client.get(q.base + '/review').json()
    response = q.client.post(q.base + '/plan/preview', json={'selection_revision': 1,
        'expected_review_revision': state['review_revision'], 'plan_revision': state['plan_revision'],
        'removed_count': state['budget'], 'keep_count': 2})
    assert response.status_code == 200, response.text
    family_preview = response.json()
    assert family_preview['captured_retained_count'] >= 2
    committed, _ = mutate(q, 'bulk_commit', preview_digest=family_preview['preview_digest'])
    assert committed['snapshot']['approved_count'] == family_preview['captured_eligible_count']
    undone, _ = mutate(q, 'bulk_undo', batch_id=committed['receipt']['batch_id'])
    assert undone['snapshot']['approved_count'] == 0


def test_cancelled_async_preview_does_not_hold_review_lock(composed, monkeypatch):
    q = composed
    entered, release = threading.Event(), threading.Event()
    original = q.service._compute_settings_plan
    def paused(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)
    monkeypatch.setattr(q.service, '_compute_settings_plan', paused)
    response = q.client.post(q.base + '/plan/preview', json={'kind': 'settings', 'selection_revision': 1,
        'expected_review_revision': 0, 'expected_plan_revision': 1,
        'settings': {'policy': 'budgeted_coverage', 'max_removed_images': 2}})
    assert response.status_code == 200
    identifier = response.json()['preview_id']
    assert entered.wait(2)
    # A reader can obtain the SQLite ledger while worker math is paused.
    assert q.client.get(q.base + '/review').status_code == 200
    assert q.client.post(q.base + '/plan/previews/' + identifier + '/cancel').status_code == 200
    release.set()
    for _ in range(100):
        status = q.client.get(q.base + '/plan/previews/' + identifier).json()
        if status['status'] == 'cancelled': break
        time.sleep(.01)
    assert status['status'] == 'cancelled', status
    assert q.client.get(q.base + '/review').json()['review_revision'] == 0


def test_legacy_direct_keep_binds_captured_set_without_refill(composed):
    q = composed
    with ReviewLedger(q.directory / 'review.sqlite3').transaction() as state: state['budget'] = 2
    before = q.client.post(q.base + '/selection', json={'selection_revision': 1, 'removed_count': 2}).json()['rows']
    captured = {row['candidate_id'] for row in before}
    answer, _ = mutate(q, 'pin', image_id=before[0]['candidate_id'], keep=True)
    assert answer['snapshot']['removed_count'] == 1
    after = q.client.post(q.base + '/selection', json={'selection_revision': 1, 'removed_count': 2}).json()['rows']
    assert {row['candidate_id'] for row in after} == captured - {before[0]['candidate_id']}
    undone, _ = mutate(q, 'bulk_undo', batch_id=answer['receipt']['batch_id'])
    assert undone['snapshot']['budget'] == 2
    assert undone['snapshot']['removed_count'] == 2


def test_undo_first_apply_restores_previous_ceiling_without_approvals(composed):
    q = composed
    preview = settings_preview(q)
    applied, _ = mutate(q, 'apply_plan', preview_id=preview['preview_id'])
    undone, _ = mutate(q, 'bulk_undo', batch_id=applied['receipt']['batch_id'])
    assert undone['snapshot']['budget'] == 0
    assert undone['snapshot']['active_settings']['policy'] == 'strict_duplicates'
    assert undone['snapshot']['approved_count'] == 0


def test_ready_preview_content_and_supplied_digest_are_bound(composed):
    q = composed
    preview = settings_preview(q)
    state_path = q.directory / 'planning' / 'previews' / preview['preview_id'] / 'state.json'
    original = state_path.read_bytes()
    changed = json.loads(original)
    changed['plan']['settings']['object_weight'] = .25
    state_path.write_text(json.dumps(changed))
    assert q.client.get(q.base + '/plan/previews/' + preview['preview_id']).status_code == 409
    state_path.write_bytes(original)
    state = q.client.get(q.base + '/review').json()
    payload = {'action': 'apply_plan', 'preview_id': preview['preview_id'], 'preview_digest': '0' * 64,
               'request_id': uuid.uuid4().hex, 'selection_revision': 1, 'expected_review_revision': state['review_revision'],
               'plan_revision': state['plan_revision'], 'removed_count': state['budget']}
    assert q.client.post(q.base + '/review', json=payload).status_code == 409
    assert q.client.get(q.base + '/review').json()['review_revision'] == state['review_revision']
    payload.update(request_id=uuid.uuid4().hex, preview_digest=preview['preview_digest'])
    response = q.client.post(q.base + '/review', json=payload)
    assert response.status_code == 200, response.text
    assert response.json()['snapshot']['active_settings']['object_weight'] == .9


@pytest.mark.parametrize('composed', [[0, 1, 2, 10, 11]], indirect=True)
def test_multiclass_ready_preview_digest_survives_json_round_trip(composed):
    q = composed
    preview = settings_preview(q)
    assert set(preview['plan']['metrics']['source_class_counts']) == {'0', '1', '2', '10', '11'}
    status = q.client.get(q.base + '/plan/previews/' + preview['preview_id'])
    assert status.status_code == 200, status.text
    assert status.json()['preview_digest'] == preview['preview_digest']
    answer, _ = mutate(q, 'apply_plan', preview_id=preview['preview_id'], preview_digest=preview['preview_digest'])
    assert answer['outcome'] == 'committed'
    assert answer['snapshot']['active_settings']['policy'] == 'budgeted_coverage'


def test_budgeted_export_keeps_new_scene_relations_and_video_groups(composed):
    q = composed
    snapshot = q.service.completed_snapshots.load(q.directory)
    images = [dict(image) for image in snapshot.images]
    images[2]['video_id'] = images[3]['video_id'] = 'clip-1'
    groups = export_scene_groups(images, list(range(5)), PruningConfig(), witness_metrics=[
        {'scene_witnesses': {'image-0': {'witness_id': 'image-1'}},
         'object_witnesses': {'image-0': {'witness_id': 'image-4'}}}])
    assert groups[0] == groups[1]
    assert groups[2] == groups[3]
    assert groups[0] != groups[2] != groups[4]
    # The proposal may remain unreviewed and therefore retained in the actual
    # approved subset; its scene witness must still be in the same split group.
    from services.dataset_pruning import grouped_splits
    split = grouped_splits(images, [0, 1, 2, 4], groups)
    assert split['assignments']['0'] == split['assignments']['1']


@pytest.mark.parametrize('composed', [{'isolated_groups': True}], indirect=True)
def test_approved_subset_export_groups_unreviewed_scene_proposals(composed):
    q = composed
    preview = settings_preview(q)
    mutate(q, 'apply_plan', preview_id=preview['preview_id'], preview_digest=preview['preview_digest'])
    rows = q.client.post(q.base + '/selection', json={'selection_revision': 1, 'removed_count': 2}).json()['rows']
    unreviewed, approved = rows
    state, _ = mutate(q, 'approve', candidate_id=approved['candidate_id'], witness_id=approved['witness_id'], evidence_digest=approved['evidence_digest'])
    prepared = q.service.prepare_active_export(q.job, {'selection_revision': 1,
        'expected_review_revision': state['snapshot']['review_revision'], 'plan_revision': state['snapshot']['plan_revision']})
    images = prepared['context'][3]
    indices = {image['image_id']: i for i, image in enumerate(images)}
    candidate, witness = indices[unreviewed['candidate_id']], indices[unreviewed['witness_id']]
    assert candidate in prepared['retained'] and witness in prepared['retained']
    assert prepared['groups'][candidate] == prepared['groups'][witness]
    from services.dataset_pruning import grouped_splits
    splits = grouped_splits(images, prepared['retained'], prepared['groups'])
    assert splits['assignments'][str(candidate)] == splits['assignments'][str(witness)]


def test_accounting_version_changes_cache_namespace_and_stales_ready_preview(composed, monkeypatch):
    q = composed
    preview = settings_preview(q)
    original_directory = coverage_directory(q.directory)
    snapshot = q.service.completed_snapshots.load(q.directory)
    config = PruningConfig.from_payload(snapshot.result['config'])
    original_key = coverage_cache_key(snapshot.result, snapshot.source, config, q.directory)
    import services.dataset_pruning_coverage as coverage
    monkeypatch.setattr(coverage, 'SCHEMA', coverage.SCHEMA + '0')
    assert coverage_directory(q.directory) != original_directory
    assert coverage_cache_key(snapshot.result, snapshot.source, config, q.directory) != original_key
    assert q.client.get(q.base + '/plan/previews/' + preview['preview_id']).status_code == 409
    assert original_directory.is_dir()
    assert (q.directory / 'planning' / 'previews' / preview['preview_id'] / 'plan.json').is_file()


def test_identical_source_jobs_own_their_coverage_and_survive_other_job_removal(composed):
    q = composed
    first = settings_preview(q)
    second = clone_job(q)
    other = settings_preview(second)
    assert other['plan']['removed_ids'] == first['plan']['removed_ids']
    second_context, _, _ = q.service._capture_plan(second.job, check=False)
    second_index = q.service._coverage_index(second_context)
    assert second_index is not q.index
    assert second_index.directory == coverage_directory(second.directory)
    assert second_index.accounting_digest == q.index.accounting_digest
    assert all(record.exists() for record in [second_index.directory / 'manifest.json', second_index.directory / 'proofs.sqlite3'])
    shutil.rmtree(q.directory)
    assert settings_preview(second)['plan']['removed_ids'] == other['plan']['removed_ids']
    restarted = PruningService(q.service.root, q.service.adapter)
    try:
        context, state, _ = restarted._capture_plan(second.job, check=False)
        restored = restarted._compute_settings_plan(context, state, other['settings'])
        assert restored['removed_ids'] == other['plan']['removed_ids']
        assert restored['encoder_calls'] == restored['training_calls'] == restored['refits'] == 0
    finally:
        for index in restarted.coverage_indexes.values(): index.close()


@pytest.mark.parametrize('phase,stale_source', [('evaluation', False), ('capture', False), ('evaluation', True), ('capture', True)])
def test_overlapping_same_token_retry_returns_committed_receipt(composed, monkeypatch, phase, stale_source):
    from concurrent.futures import ThreadPoolExecutor
    q = composed
    preview = settings_preview(q)
    mutate(q, 'apply_plan', preview_id=preview['preview_id'])
    row = preview['plan']['cards'][0]
    state = q.client.get(q.base + '/review').json()
    payload = {'action': 'approve', 'request_id': uuid.uuid4().hex, 'selection_revision': 1,
        'expected_review_revision': state['review_revision'], 'plan_revision': state['plan_revision'],
        'removed_count': state['budget'], 'candidate_id': row['candidate_id'],
        'witness_id': row['witness_id'], 'evidence_digest': row['evidence_digest']}
    first_entered, retry_entered, committed = threading.Event(), threading.Event(), threading.Event()
    count_lock = threading.Lock()
    calls = {'evaluation': 0, 'capture': 0}
    evaluate, capture = q.index.evaluate, q.service._capture_plan
    def controlled_evaluate(*args, **kwargs):
        with count_lock:
            calls['evaluation'] += 1
            ordinal = calls['evaluation']
        if ordinal == 1:
            first_entered.set()
            assert retry_entered.wait(5)
        elif ordinal == 2 and phase == 'evaluation':
            retry_entered.set()
            assert committed.wait(5)
        return evaluate(*args, **kwargs)
    def controlled_capture(*args, **kwargs):
        with count_lock:
            calls['capture'] += 1
            ordinal = calls['capture']
        if ordinal == 2 and phase == 'capture':
            retry_entered.set()
            assert committed.wait(5)
        return capture(*args, **kwargs)
    monkeypatch.setattr(q.index, 'evaluate', controlled_evaluate)
    monkeypatch.setattr(q.service, '_capture_plan', controlled_capture)
    with ThreadPoolExecutor(max_workers=2) as pool:
        original = pool.submit(q.client.post, q.base + '/review', json=payload)
        assert first_entered.wait(5)
        retry = pool.submit(q.client.post, q.base + '/review', json=payload)
        first = original.result(5)
        q.changed['value'] = stale_source
        committed.set()
        second = retry.result(5)
    assert first.status_code == second.status_code == 200, (first.text, second.text)
    assert first.json()['receipt'] == second.json()['receipt']
    if stale_source:
        assert second.json()['source_stale'] and second.json()['snapshot'] is None
    state = ReviewLedger(q.directory / 'review.sqlite3').read()
    assert state['review_revision'] == payload['expected_review_revision'] + 1
    assert len([row for row in state['history'] if row['request_id'] == payload['request_id']]) == 1
    assert sum(row.get('status') == 'approved' for row in state['decisions'].values()) == 1
    assert q.client.post(q.base + '/review', json={**payload, 'note': 'different payload'}).status_code == 409


def test_strict_reject_undo_restores_its_directed_match(composed):
    q = composed
    preview = settings_preview(q, policy='strict_duplicates')
    mutate(q, 'apply_plan', preview_id=preview['preview_id'])
    row = preview['plan']['cards'][0]
    rejected, _ = mutate(q, 'reject', candidate_id=row['candidate_id'], witness_id=row['witness_id'], evidence_digest=row['evidence_digest'])
    after_reject = q.client.post(q.base + '/selection', json={'selection_revision': 1, 'removed_count': 2}).json()['rows']
    assert next(value for value in after_reject if value['candidate_id'] == row['candidate_id'])['witness_id'] != row['witness_id']
    undone, _ = mutate(q, 'undo_decision', candidate_id=row['candidate_id'])
    assert undone['snapshot']['plan_revision'] > rejected['snapshot']['plan_revision']
    state = ReviewLedger(q.directory / 'review.sqlite3').read()
    assert state['rejected'] == {} and state['decisions'] == {}
    after_undo = q.client.post(q.base + '/selection', json={'selection_revision': 1, 'removed_count': 2}).json()['rows']
    fresh = settings_preview(q, policy='strict_duplicates')
    for rows in [after_undo, fresh['plan']['cards']]:
        assert next(value for value in rows if value['candidate_id'] == row['candidate_id'])['witness_id'] == row['witness_id']


def test_strict_undo_preserves_a_later_independent_edge_rejection(composed):
    q = composed
    preview = settings_preview(q, policy='strict_duplicates')
    mutate(q, 'apply_plan', preview_id=preview['preview_id'])
    row = preview['plan']['cards'][0]
    mutate(q, 'reject', candidate_id=row['candidate_id'], witness_id=row['witness_id'], evidence_digest=row['evidence_digest'])
    ledger = ReviewLedger(q.directory / 'review.sqlite3')
    with ledger.transaction() as state:
        key = next(iter(state['rejected']))
        state['rejected'][key].update(owner='later-operation', version=state['review_revision'] + 1)
    mutate(q, 'undo_decision', candidate_id=row['candidate_id'])
    assert ledger.read()['rejected'][key]['owner'] == 'later-operation'


@pytest.mark.parametrize('policy', ['budgeted_coverage', 'strict_duplicates'])
def test_inherited_keep_unpin_persists_no_refill_and_refreshes_measurements(composed, policy):
    q = composed
    second = clone_job(q)
    preview = settings_preview(q, policy=policy)
    mutate(q, 'apply_plan', preview_id=preview['preview_id'])
    candidate = preview['plan']['removed_ids'][0]
    mutate(second, 'pin', image_id=candidate, keep=True)
    # Commit the cheap reduction independently, then simulate a process restart
    # before any request has refreshed its derived coverage measurements.
    with q.service.lock:
        q.service.review_context(q.job, {'removed_count': 0}, False)
    ledger = ReviewLedger(q.directory / 'review.sqlite3')
    reduced = ledger.read()['plans']['active']['value']
    assert set(reduced['candidate_ids']) == set(preview['plan']['removed_ids']) - {candidate}
    persisted = json.loads((q.directory / 'planning' / 'artifacts' / (reduced['artifact'] + '.json')).read_text())
    assert persisted['coverage_refresh_required'] and not persisted['proposed_loss']['measured']
    assert persisted['curve'] == [] and persisted['metrics'] == {}
    restarted = PruningService(q.service.root, q.service.adapter)
    try:
        current = restarted.review_snapshot(q.job)
        assert current['removed_count'] == 1
        assert not current['coverage_refresh_required']
        if policy == 'budgeted_coverage': assert current['proposed_loss']['measured']
        context, state, _ = restarted._capture_plan(q.job, check=False)
        plan = restarted._load_active_plan(context, state)
        assert plan['removed_count'] == plan['metrics']['removed_count'] == 1
        assert plan['removed_percent'] == 20.
        assert plan['no_refill']
        mutate(second, 'pin', image_id=candidate, keep=False)
        assert restarted.review_snapshot(q.job)['removed_count'] == 1
        assert candidate not in ledger.read()['plans']['active']['value']['candidate_ids']
        # Explicit settings preview is the only operation allowed to expand it.
        assert settings_preview(q, policy=policy)['plan']['removed_count'] == 2
    finally:
        for index in restarted.coverage_indexes.values(): index.close()


@pytest.mark.parametrize('composed', [{'negative_only': True}], indirect=True)
def test_negative_only_api_replays_empty_objects_and_exports_approved_subset(composed):
    q = composed
    preview = settings_preview(q)
    assert preview['plan']['removed_count'] == 2
    assert preview['plan']['metrics']['uncovered_object_count'] == 0
    paths = list(q.directory.glob('planning/features-*/objects.npy'))
    assert len(paths) == 1 and np.load(paths[0], mmap_mode='r').shape == (0, 0)
    mutate(q, 'apply_plan', preview_id=preview['preview_id'])
    row = preview['plan']['cards'][0]
    approved, _ = mutate(q, 'approve', candidate_id=row['candidate_id'], witness_id=row['witness_id'], evidence_digest=row['evidence_digest'])
    prepared = q.service.prepare_active_export(q.job, {'selection_revision': 1,
        'expected_review_revision': approved['snapshot']['review_revision'], 'plan_revision': approved['snapshot']['plan_revision']})
    assert len(prepared['removed']) == 1
    assert prepared['metrics']['affected_appearance_group_count'] == 0
    assert prepared['metrics']['lost_appearance_weight_fraction'] == 0
    assert prepared['metrics']['ready']


def test_new_direct_qualification_stales_previews_and_requires_new_approval(composed, monkeypatch):
    import services.dataset_pruning_coverage as coverage
    q = composed
    preview = settings_preview(q)
    mutate(q, 'apply_plan', preview_id=preview['preview_id'])
    row = preview['plan']['cards'][0]
    mutate(q, 'approve', candidate_id=row['candidate_id'], witness_id=row['witness_id'], evidence_digest=row['evidence_digest'])
    pending = settings_preview(q)
    monkeypatch.setattr(coverage, 'QUALIFICATION', coverage.QUALIFICATION + '-next')
    assert q.client.get(q.base + '/plan/previews/' + pending['preview_id']).status_code == 409
    current = q.client.get(q.base + '/review').json()
    assert current['accounting_outdated'] and current['approved_count'] == 0
    assert q.client.get(q.base + '/result').json()['accounting_outdated']
    fresh = settings_preview(q)
    assert fresh['plan']['qualification'] == coverage.QUALIFICATION
    assert fresh['plan']['policy_digest'] != preview['plan']['policy_digest']
    applied, _ = mutate(q, 'apply_plan', preview_id=fresh['preview_id'])
    assert not applied['snapshot']['accounting_outdated']
    assert applied['snapshot']['approved_count'] == 0
    assert ReviewLedger(q.directory / 'review.sqlite3').read()['decisions'][row['candidate_id']]['status'] == 'superseded'


@pytest.mark.parametrize('composed', [{'object_angles': [0, 0, 90, 90, 180]}], indirect=True)
@pytest.mark.parametrize('later_keep', [False, True])
def test_strict_reject_sole_witness_undo_restores_only_its_owned_plan(composed, later_keep):
    q = composed
    preview = settings_preview(q, policy='strict_duplicates')
    assert preview['plan']['removed_count'] == 2
    mutate(q, 'apply_plan', preview_id=preview['preview_id'])
    row = preview['plan']['cards'][0]
    rejected, _ = mutate(q, 'reject', candidate_id=row['candidate_id'], witness_id=row['witness_id'], evidence_digest=row['evidence_digest'])
    assert rejected['snapshot']['removed_count'] == 1
    if later_keep: mutate(q, 'pin', image_id=row['candidate_id'], keep=True)
    undone, _ = mutate(q, 'undo_decision', candidate_id=row['candidate_id'])
    assert undone['snapshot']['removed_count'] == (1 if later_keep else 2)
    rows = q.client.post(q.base + '/selection', json={'selection_revision': 1, 'removed_count': 2}).json()['rows']
    assert (row['candidate_id'] in {value['candidate_id'] for value in rows}) is not later_keep
    assert not ReviewLedger(q.directory / 'review.sqlite3').read()['rejected']


@pytest.mark.parametrize('allow_action,legacy', [('unpin', False), ('unpin', True), ('undo', False)])
def test_keep_and_unpin_before_other_job_reopens_cannot_restore_its_capture(composed, allow_action, legacy):
    q = composed
    second = clone_job(q)
    preview = settings_preview(q)
    mutate(q, 'apply_plan', preview_id=preview['preview_id'])
    candidate = preview['plan']['removed_ids'][0]
    kept, _ = mutate(second, 'pin', image_id=candidate, keep=True)
    if allow_action == 'undo':
        mutate(second, 'bulk_undo', batch_id=kept['receipt']['batch_id'])
    else:
        mutate(second, 'pin', image_id=candidate, keep=False)
    if legacy:
        with ReviewLedger(second.directory / 'review.sqlite3').transaction() as state:
            for pin in state['pins'].values(): pin.pop('last_keep_at', None)
            for operation in state['history']:
                for field in operation.get('touched_fields', []):
                    if field['collection'] == 'pins' and field.get('before'):
                        field['before'].pop('last_keep_at', None)
    # There has been no request against job A between those two commits.
    restored = q.client.get(q.base + '/review').json()
    assert restored['removed_count'] == 1 and not restored['coverage_refresh_required']
    assert restored['pins'][candidate]['value'] is False
    assert restored['pins'][candidate]['last_keep_at'] > 0
    # Consuming old Keep history must not constrain a new explicit plan.
    fresh = settings_preview(q)
    assert fresh['plan']['removed_count'] == 2
    applied, _ = mutate(q, 'apply_plan', preview_id=fresh['preview_id'])
    assert applied['snapshot']['removed_count'] == 2
    assert q.client.get(q.base + '/review').json()['removed_count'] == 2


def test_later_committed_keep_is_not_lost_behind_a_newer_unpin_timestamp(composed):
    q = composed
    second = clone_job(q)
    preview = settings_preview(q)
    mutate(q, 'apply_plan', preview_id=preview['preview_id'])
    candidate = preview['plan']['removed_ids'][0]
    image = next(image for image in q.service.completed_snapshots.load(q.directory).images if image['image_id'] == candidate)
    # Two requests captured concurrently: the unpin's timestamp is newer,
    # but the older Keep commits last. Each committed field is valid locally.
    with ReviewLedger(q.directory / 'review.sqlite3').transaction() as state:
        state['pins'][candidate] = {'value': False, 'intent_at': 200, 'last_keep_at': 0,
            'image_sha256': image['image_sha256'], 'intent_origin_job': q.job, 'owner': 'unpin', 'version': state['review_revision']}
    with ReviewLedger(second.directory / 'review.sqlite3').transaction() as state:
        state['pins'][candidate] = {'value': True, 'intent_at': 100, 'last_keep_at': 100,
            'image_sha256': image['image_sha256'], 'intent_origin_job': second.job, 'owner': 'keep', 'version': state['review_revision']}
    current = q.client.get(q.base + '/review').json()
    assert current['pins'][candidate]['value'] is False
    assert current['pins'][candidate]['intent_at'] == 200
    assert current['pins'][candidate]['last_keep_at'] == 100
    assert current['removed_count'] == 1
    assert settings_preview(q)['plan']['removed_count'] == 2


@pytest.mark.parametrize('admission', ['status', 'download'])
def test_peer_keep_invalidates_prepared_export_without_opening_original_job(composed, admission):
    from fastapi import HTTPException
    q = composed
    second = clone_job(q)
    preview = settings_preview(q)
    mutate(q, 'apply_plan', preview_id=preview['preview_id'])
    row = preview['plan']['cards'][0]
    approved, _ = mutate(q, 'approve', candidate_id=row['candidate_id'], witness_id=row['witness_id'], evidence_digest=row['evidence_digest'])
    export_id = 'f' * 20
    directory = q.directory / 'exports' / export_id
    write_json(directory / 'state.json', {'status': 'completed', 'export_id': export_id, 'job_id': q.job,
        'review_revision': approved['snapshot']['review_revision'], 'plan_revision': approved['snapshot']['plan_revision']})
    (directory / 'dataset.zip').write_bytes(b'prepared owned fixture')
    assert q.service.export_status(q.job, export_id)['status'] == 'completed'
    mutate(second, 'pin', image_id=row['candidate_id'], keep=True)
    # No request reconciles job A before this direct admission boundary.
    if admission == 'status':
        assert q.service.export_status(q.job, export_id)['status'] == 'stale'
    else:
        with pytest.raises(HTTPException) as error: q.service.download(q.job, export_id)
        assert error.value.status_code == 409
    durable = ReviewLedger(q.directory / 'review.sqlite3').read()
    assert row['candidate_id'] not in durable['plans']['active']['value']['candidate_ids']
    assert durable['decisions'][row['candidate_id']]['status'] == 'superseded'


def test_new_qualification_invalidates_already_prepared_export(composed, monkeypatch):
    import services.dataset_pruning_coverage as coverage
    from fastapi import HTTPException
    q = composed
    preview = settings_preview(q)
    mutate(q, 'apply_plan', preview_id=preview['preview_id'])
    row = preview['plan']['cards'][0]
    approved, _ = mutate(q, 'approve', candidate_id=row['candidate_id'], witness_id=row['witness_id'], evidence_digest=row['evidence_digest'])
    export_id = 'e' * 20
    directory = q.directory / 'exports' / export_id
    write_json(directory / 'state.json', {'status': 'completed', 'export_id': export_id, 'job_id': q.job,
        'review_revision': approved['snapshot']['review_revision'], 'plan_revision': approved['snapshot']['plan_revision']})
    (directory / 'dataset.zip').write_bytes(b'prepared owned fixture')
    assert q.service.export_status(q.job, export_id)['status'] == 'completed'
    monkeypatch.setattr(coverage, 'QUALIFICATION', coverage.QUALIFICATION + '-next')
    assert q.service.export_status(q.job, export_id)['status'] == 'stale'
    with pytest.raises(HTTPException) as error: q.service.download(q.job, export_id)
    assert error.value.status_code == 409


@pytest.mark.parametrize('endpoint', ['review', 'result'])
def test_unversioned_read_retries_review_race_without_hiding_source_drift(composed, monkeypatch, endpoint):
    from concurrent.futures import ThreadPoolExecutor
    q = composed
    preview = settings_preview(q)
    mutate(q, 'apply_plan', preview_id=preview['preview_id'])
    row = preview['plan']['cards'][0]
    before = q.client.get(q.base + '/review').json()
    payload = {'action': 'approve', 'request_id': uuid.uuid4().hex, 'selection_revision': 1,
        'expected_review_revision': before['review_revision'], 'plan_revision': before['plan_revision'],
        'removed_count': before['budget'], 'candidate_id': row['candidate_id'],
        'witness_id': row['witness_id'], 'evidence_digest': row['evidence_digest']}
    captured, release = threading.Event(), threading.Event()
    original = q.service._capture_plan
    calls = 0
    count_lock = threading.Lock()
    def paused_capture(*args, **kwargs):
        nonlocal calls
        value = original(*args, **kwargs)
        with count_lock:
            calls += 1
            ordinal = calls
        if ordinal == 1:
            captured.set()
            assert release.wait(5)
        return value
    monkeypatch.setattr(q.service, '_capture_plan', paused_capture)
    with ThreadPoolExecutor(max_workers=1) as pool:
        read = pool.submit(q.client.get, q.base + '/' + endpoint)
        assert captured.wait(5)
        approved = q.client.post(q.base + '/review', json=payload)
        assert approved.status_code == 200, approved.text
        release.set()
        response = read.result(5)
    assert response.status_code == 200, response.text
    assert response.json()['approved_count'] == 1
    assert response.json()['review_revision'] == approved.json()['receipt']['revision']
    q.changed['value'] = True
    changed = q.client.get(q.base + '/' + endpoint)
    assert changed.status_code == 409 and changed.json()['detail'] == 'Fixture source changed'
