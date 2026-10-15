"""Explicit grouped split assignment uses the real SQLite authority."""
import copy

import pytest

from services.hosted.store import HostedStore
from services.hosted.native_analysis import _training_membership
from test_hosted_store import ADMIN, ADMIN2, ALICE, ctx, dataset, error, imported, objects, store


def confirmed(store, ds, name, metadata=None):
    image = imported(store, ds, name)
    leased = store.acquire_editor(ADMIN, image['id'], 'split-fixture')
    annotations = {**objects(ds), 'capture_metadata': metadata or {}}
    saved = store.save(ADMIN, ctx(leased, 'save-' + name, 'split-fixture'), annotations, complete=True)
    return store.approve(ADMIN, ctx(saved, 'approve-' + name, 'split-fixture'))


def commit(store, ds, plan, operation):
    return store.set_splits(ADMIN, ds['id'], plan['assignments'], plan['expected_revision'], plan['expected'], operation)


def test_preview_expands_transitive_groups_and_commit_preserves_versions_snapshots_and_receipts(store):
    ds = dataset(store)
    a = confirmed(store, ds, 'a', {'sequence_id': 'sequence-one'})
    b = confirmed(store, ds, 'b', {'sequence_id': 'sequence-one', 'video_id': 'video-two'})
    c = confirmed(store, ds, 'c', {'video_id': 'video-two'})
    d = confirmed(store, ds, 'd')
    incoming = imported(store, ds, 'incoming', {**objects(ds), 'capture_metadata': {'video_id': 'video-two'}})
    before = store.create_snapshot(ADMIN, ds['id'], 'before-split')
    plan = store.preview_splits(ADMIN, ds['id'], [a['id']], 'val')
    assert plan['selected_count'] == 1 and plan['group_added_count'] == 2 and plan['change_count'] == 3
    assert set(plan['assignments']) == {a['id'], b['id'], c['id']}
    assert plan['counts_before'] == {'train': 4, 'val': 0, 'test': 0, 'unsplit': 0}
    assert plan['counts_after'] == {'train': 1, 'val': 3, 'test': 0, 'unsplit': 0}
    result = commit(store, ds, plan, 'grouped-split')
    assert result['assignments'] == plan['assignments'] and result['changed_count'] == 3
    assert result['revision'] == plan['expected_revision'] + 1
    assert all(row['current_version'] == plan['expected'][row['id']]['expected_version'] and row['generation'] == plan['expected'][row['id']]['generation'] + 1 for row in result['images'])
    assert store.image(ADMIN, incoming['id'])['split'] == incoming['split']
    current = store.snapshot(ADMIN, ds['id'])
    assert _training_membership(current) == {a['id']: 'val', b['id']: 'val', c['id']: 'val', d['id']: 'train'}
    assert all(row['split'] == 'train' for row in store.get_snapshot(ADMIN, before['snapshot_id'])['manifest']['images'])
    later = store.acquire_editor(ADMIN, d['id'], 'later')
    store.save(ADMIN, ctx(later, 'later-save', 'later'), {**objects(ds), 'caption': 'Later human edit'}, complete=True)
    reopened = HostedStore(store.path, clock=store.clock)
    assert commit(reopened, ds, plan, 'grouped-split') == result
    changed = copy.deepcopy(plan); changed['assignments'][a['id']] = 'test'
    error('operation_id_reused', lambda: commit(reopened, ds, changed, 'grouped-split'))
    assert any(row['kind'] == 'split_changed' for row in store.history(ADMIN, a['id'])['events'])


@pytest.mark.parametrize('group_field', ['capture_group_id', 'video_group', 'source_group', 'duplicate_group', 'flight_id'])
def test_split_commit_rejects_partial_group_stale_generation_and_active_edit_atomically(store, group_field):
    ds = dataset(store)
    a = confirmed(store, ds, 'a', {group_field: 'capture'})
    b = confirmed(store, ds, 'b', {group_field: 'capture'})
    plan = store.preview_splits(ADMIN, ds['id'], [a['id']], 'test')
    partial = {**plan, 'assignments': {a['id']: 'test'}, 'expected': {a['id']: plan['expected'][a['id']]}}
    error('split_group_conflict', lambda: commit(store, ds, partial, 'partial-split'))
    lease = store.acquire_editor(ADMIN2, b['id'], 'other-tab')
    error('ownership_changed', lambda: commit(store, ds, plan, 'stale-split'))
    active = store.preview_splits(ADMIN, ds['id'], [a['id']], 'test')
    error('editor_lease_conflict', lambda: commit(store, ds, active, 'active-split'))
    assert {row['split'] for row in store.snapshot(ADMIN, ds['id'])['images']} == {'train'}
    store.release_editor(ADMIN2, ctx(lease, 'release-other', 'other-tab'))
    plan = store.preview_splits(ADMIN, ds['id'], [a['id']], 'test')
    result = commit(store, ds, plan, 'final-split')
    assert result['changed_count'] == 2
    unchanged = store.preview_splits(ADMIN, ds['id'], [a['id']], 'test')
    assert commit(store, ds, unchanged, 'unchanged-split')['revision'] == result['revision']


def test_split_admission_rejects_pending_foreign_roles_and_changed_dataset(store):
    ds = dataset(store)
    a = confirmed(store, ds, 'a')
    pending = imported(store, ds, 'pending')
    other_ds = dataset(store, 'other')
    foreign = confirmed(store, other_ds, 'foreign')
    error('split_image_not_confirmed', lambda: store.preview_splits(ADMIN, ds['id'], [pending['id']], 'val'))
    error('split_image_not_confirmed', lambda: store.preview_splits(ADMIN, ds['id'], [foreign['id']], 'val'))
    error('admin_required', lambda: store.preview_splits(ALICE, ds['id'], [a['id']], 'val'))
    plan = store.preview_splits(ADMIN, ds['id'], [a['id']], 'val')
    store.add_class(ADMIN, ds['id'], 'Bus', 'new-class', expected_class_version=ds['class_version'])
    error('dataset_revision_changed', lambda: commit(store, ds, plan, 'changed-dataset'))
    assert store.image(ADMIN, a['id'])['split'] == 'train'
