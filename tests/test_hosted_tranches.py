"""Composed tranche authority flows; no live hosted state or model access."""
from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3

import pytest

from services.hosted.store import HostedStore
from services.hosted.tranche_schema import migrate_store, validate_tranches
from services.hosted.operations import HostedOperations
from services.hosted.artifacts import ArtifactStore
from test_hosted_store import store, ADMIN, ADMIN2, ALICE, BOB, dataset, imported, ctx, expected, objects, error


def draft(store, ds, images=(), name='Work', annotator=ALICE):
    return store.create_tranche(ADMIN, ds['id'], name, 'create-' + name,
        instructions='Check every object', annotator_id=annotator.user_id if annotator else None,
        image_ids=[i['id'] for i in images], expected=expected(images))['tranche']


def activate(store, tranche, images):
    return store.activate_tranche(ADMIN, tranche['id'], tranche['revision'],
        [i['id'] for i in images], expected(images), tranche['annotator_id'], 'activate-' + tranche['id'])


def submit(store, ds, image, user=ALICE):
    image = store.acquire_editor(user, image['id'], 'editor')
    image = store.save(user, ctx(image, 'save-' + image['id'], 'editor'), objects(ds), complete=True)
    return store.submit(user, ctx(image, 'submit-' + image['id'], 'editor'))


def assert_counts(store, tranche_id, **counts):
    result = store.tranche(ADMIN, tranche_id)['counts']
    for key, value in counts.items():
        assert result[key] == value, result
    assert sum(result[key] for key in ('not_assigned','to_finish','changes_requested','awaiting_review','approved','removed','transferred','archived')) == result['ever_included']
    with store.read_connection() as con:
        validate_tranches(con)
    return result


def test_draft_reservation_conflict_and_no_annotator_access(store):
    ds = dataset(store); image = imported(store, ds)
    empty = draft(store, ds, name='empty', annotator=None)
    assert empty['counts']['retained'] == empty['counts']['approved'] == 0
    one = draft(store, ds, [image])
    error('tranche_not_found', lambda: store.tranche(ALICE, one['id']))
    error('image_not_found', lambda: store.image(ALICE, image['id']))
    error('image_not_assignable', lambda: draft(store, ds, [image], name='conflict'))
    error('image_not_assignable', lambda: store.assign(ADMIN, ds['id'], [image['id']], BOB.user_id, 'legacy-bypass', expected([image])))
    assert not store.image_queue(ADMIN, ds['id'], queue='unallocated')['images']
    assert store.queue_scope(ADMIN, image['id'])['scope']['queue'] == 'draft'
    updated = store.edit_tranche(ADMIN, one['id'], 0, 'Renamed', 'New instructions', 'rename-work')['tranche']
    assert updated['revision'] == 1
    error('tranche_revision_changed', lambda: activate(store, one, [image]))
    assert_counts(store, one['id'], not_assigned=1)


def test_two_admins_reserve_same_image_exactly_once(store):
    ds = dataset(store); image = imported(store, ds)
    def reserve(actor):
        try:
            return store.create_tranche(actor, ds['id'], actor.user_id, 'race-' + actor.user_id,
                image_ids=[image['id']], expected=expected([image]))['tranche']['id']
        except Exception as exc:
            return getattr(exc, 'code', str(exc))
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(reserve, [ADMIN, ADMIN2]))
    assert sum(value.startswith('tranche_') for value in results) == 1
    assert results.count('image_not_assignable') == 1
    assert len(store.list_tranches(ADMIN)['tranches']) == 1


def test_individual_submit_return_approve_and_personal_counts(store):
    ds = dataset(store); images = [imported(store, ds, str(n)) for n in range(3)]
    tranche = draft(store, ds, images); group = activate(store, tranche, images)
    assert len(store.image_queue(ALICE, ds['id'], tranche_id=tranche['id'], queue='work')['images']) == 3
    error('tranche_not_found', lambda: store.tranche(BOB, tranche['id']))
    submitted = submit(store, ds, group['images'][0])
    assert_counts(store, tranche['id'], to_finish=2, awaiting_review=1)
    error('image_not_found', lambda: store.image(ALICE, submitted['id']))
    review = store.acquire_editor(ADMIN, submitted['id'], 'admin-tab')
    returned = store.return_image(ADMIN, ctx(review, 'return-one', 'admin-tab'), 'One car is missing')
    assert_counts(store, tranche['id'], changes_requested=1)
    assert store.image(ALICE, returned['id'], include_review=True)['review_feedback']['feedback'] == 'One car is missing'
    row = next(m for m in store.tranche_members(ALICE, tranche['id'], state='changes_requested')['members'] if m['image_id'] == returned['id'])
    assert row['feedback'] == 'One car is missing'
    # Returned work arrives unchecked with its objects kept; the annotator confirms it again.
    assert returned['complete'] is False and returned['negative'] is False
    editing = store.acquire_editor(ALICE, returned['id'], 'editor')
    error('annotation_not_complete', lambda: store.submit(ALICE, ctx(editing, 'resubmit-unchecked', 'editor')))
    kept = store.image(ALICE, returned['id'])['annotations']
    assert len(kept['objects']) == len(objects(ds)['objects'])
    editing = store.save(ALICE, ctx(editing, 'recheck', 'editor'), kept, complete=True)
    submitted = store.submit(ALICE, ctx(editing, 'resubmit', 'editor'))
    store.approve_from_review_queue(ADMIN, ctx(submitted, 'approval'))
    assert_counts(store, tranche['id'], approved=1, to_finish=2)
    member = next(m for m in store.tranche_members(ALICE, tranche['id'])['members'] if m['image_id'] == submitted['id'])
    assert member['image'] is None and not member['can_view']
    assert store.tranche(ALICE, tranche['id'])['counts']['approved'] == 1
    assert store.tranche(ADMIN, tranche['id'])['revision'] == group['tranche']['revision']


def test_admin_approves_reserved_and_reopens_without_resurrection(store):
    ds = dataset(store); image = imported(store, ds)
    tranche = draft(store, ds, [image])
    current = store.acquire_editor(ADMIN, image['id'], 'editor')
    current = store.save(ADMIN, ctx(current, 'negative', 'editor'), {'objects': []}, negative=True)
    store.approve(ADMIN, ctx(current, 'approve-negative', 'editor'))
    assert_counts(store, tranche['id'], approved=1, not_assigned=0)
    closed = store.set_tranche_lifecycle(ADMIN, tranche['id'], tranche['revision'], 'close-tranche')['tranche']
    reopened = store.set_tranche_lifecycle(ADMIN, tranche['id'], closed['revision'], 'reopen-tranche', reopen=True)['tranche']
    assert reopened['lifecycle'] == 'draft'
    assert not store.image_queue(ADMIN, ds['id'], tranche_id=tranche['id'], queue='draft')['images']
    error('image_not_assignable', lambda: store.add_tranche_members(ADMIN, tranche['id'], reopened['revision'], [image['id']], expected([store.image(ADMIN,image['id'])]), 'cannot-demote'))


@pytest.mark.parametrize('destination_active', [False, True])
def test_atomic_transfer_saved_draft_source_outcome_and_private_feedback(store, destination_active):
    ds = dataset(store); image, seed = imported(store, ds, 'one'), imported(store, ds, 'seed')
    source = draft(store, ds, [image], 'Source'); activate(store, source, [image])
    submitted = submit(store, ds, image)
    review = store.acquire_editor(ADMIN, image['id'], 'reviewer')
    returned = store.return_image(ADMIN, ctx(review, 'return-source', 'reviewer'), 'Private Alice feedback')
    image = store.acquire_editor(ALICE, image['id'], 'alice-tab')
    image = store.save(ALICE, ctx(image, 'private-draft', 'alice-tab'), objects(ds))
    dest = draft(store, ds, [seed] if destination_active else [], 'Destination', BOB)
    if destination_active:
        dest = activate(store, dest, [seed])['tranche']
    source = store.tranche(ADMIN, source['id'])
    result = store.move_tranche_members(ADMIN, source['id'], source['revision'], dest['id'], dest['revision'], [image['id']], expected([image]), 'Move unfinished image', 'transfer-one')
    assert result == store.move_tranche_members(ADMIN, source['id'], source['revision'], dest['id'], dest['revision'], [image['id']], expected([image]), 'Move unfinished image', 'transfer-one')
    assert result['images'][0]['current_version'] == image['current_version']
    error('image_not_found', lambda: store.save(ALICE, ctx(image, 'late-save', 'alice-tab'), objects(ds)))
    assert_counts(store, source['id'], transferred=1, retained=0)
    if not destination_active:
        result = activate(store, result['destination'], result['images'])
    bob = store.image(BOB, image['id'], include_review=True)
    assert bob['annotations'] == objects(ds) and bob['review_feedback'] is None
    history = store.history(BOB, image['id'])
    assert not history['versions'] and not history['reviews'] and not history['submissions']
    assert all(row['actor_id'] == BOB.user_id for row in history['events'])
    approved = submit(store, ds, bob, BOB)
    store.approve_from_review_queue(ADMIN, ctx(approved, 'approve-bob'))
    assert_counts(store, source['id'], transferred=1, approved=0)
    assert store.tranche(ALICE, source['id'])['counts']['transferred'] == 1
    assert store.tranche(BOB, dest['id'])['counts']['approved'] == 1


def test_change_assignee_rejects_reviews_then_preserves_old_outcomes(store):
    ds = dataset(store); images = [imported(store, ds, str(n)) for n in range(2)]
    tranche = draft(store, ds, images); group = activate(store, tranche, images); tranche = group['tranche']
    submitted = submit(store, ds, images[0]); second = store.image(ADMIN, images[1]['id'])
    error('tranche_has_pending_review', lambda: store.change_tranche_assignee(ADMIN, tranche['id'], tranche['revision'], BOB.user_id, [second['id']], expected([second]), 'blocked-change'))
    store.approve_from_review_queue(ADMIN, ctx(submitted, 'approve-first'))
    changed = store.change_tranche_assignee(ADMIN, tranche['id'], tranche['revision'], BOB.user_id, [second['id']], expected([second]), 'change-annotator')
    assert changed['images'][0]['assignee_id'] == BOB.user_id
    assert store.tranche(ALICE, tranche['id'])['counts']['approved'] == 1
    assert store.tranche(ALICE, tranche['id'])['counts']['transferred'] == 1
    assert store.tranche(BOB, tranche['id'])['counts']['approved'] == 0
    assert_counts(store, tranche['id'], approved=1, to_finish=1, ever_included=2)


@pytest.mark.parametrize('disabled,role', [(1,'annotator'), (0,'admin')])
def test_unavailable_annotator_can_be_recalled_and_submission_withdrawn(store, disabled, role):
    ds = dataset(store); images = [imported(store, ds, str(n)) for n in range(2)]
    tranche = draft(store, ds, images); activate(store, tranche, images)
    submitted = submit(store, ds, images[0]); second = store.image(ADMIN, images[1]['id'])
    with store.transaction() as con:
        con.execute('UPDATE hosted_users SET disabled=?,role=? WHERE id=?', (disabled, role, ALICE.user_id))
    assert not store.tranche(ADMIN, tranche['id'])['annotator_available']
    store.recall_images(ADMIN, ds['id'], [second['id']], expected([second]), 'Annotator unavailable', 'recall-unavailable')
    result = store.withdraw_submission(ADMIN, ctx(submitted, 'withdraw-unavailable'), submitted['active_submission'], 'Author is unavailable')
    assert result['state'] == 'unassigned' and result['active_submission'] is None
    assert store.withdraw_submission(ADMIN, ctx(submitted, 'withdraw-unavailable'), submitted['active_submission'], 'Author is unavailable') == result
    assert_counts(store, tranche['id'], removed=2)
    with store.read_connection() as con:
        assert con.execute('SELECT action FROM hosted_reviews WHERE submission_id=?', (submitted['active_submission'],)).fetchone()[0] == 'withdrawn'


def test_repeat_member_is_distinct_and_close_rejects_unfinished(store):
    ds = dataset(store); image = imported(store, ds); tranche = draft(store, ds, [image])
    error('tranche_has_unfinished_members', lambda: store.set_tranche_lifecycle(ADMIN, tranche['id'], 0, 'cannot-close'))
    removed = store.remove_tranche_members(ADMIN, tranche['id'], 0, [image['id']], expected([image]), 'Not needed yet', 'remove-one')
    add = store.add_tranche_members(ADMIN, tranche['id'], removed['tranche']['revision'], [image['id']], expected(removed['images']), 'add-back')
    assert_counts(store, tranche['id'], ever_included=1, not_assigned=1, removed=0)
    with store.read_connection() as con:
        assert con.execute('SELECT COUNT(*) FROM hosted_tranche_members WHERE image_id=?', (image['id'],)).fetchone()[0] == 2
    assert len(store.tranche_members(ADMIN, tranche['id'])['members']) == 1


def test_stale_selection_and_mixed_move_commit_nothing(store):
    ds = dataset(store); images = [imported(store, ds, str(n)) for n in range(2)]
    source = draft(store, ds, images); activate(store, source, images); source = store.tranche(ADMIN,source['id'])
    destination = draft(store, ds, name='Target')
    captured = [store.image(ADMIN, i['id']) for i in images]
    submitted = submit(store, ds, images[1])
    error('tranche_member_not_movable', lambda: store.move_tranche_members(ADMIN, source['id'], source['revision'], destination['id'], 0, [i['id'] for i in captured], expected(captured), 'Move both', 'bad-move'))
    assert_counts(store, source['id'], to_finish=1, awaiting_review=1)
    assert_counts(store, destination['id'], ever_included=0)
    assert store.receipt(ADMIN, 'bad-move') is None
    editable = store.acquire_editor(ALICE, images[0]['id'], 'new-tab')
    error('ownership_changed', lambda: store.remove_tranche_members(ADMIN, source['id'], source['revision'], [captured[0]['id']], expected([captured[0]]), 'Recover', 'stale-remove'))
    assert store.image(ALICE, editable['id'])['editor_lease']


def test_queue_keysets_do_not_skip_after_submit_and_anchor_stays_scoped(store):
    ds = dataset(store); images = [imported(store, ds, str(n)) for n in range(12)]
    tranche = draft(store, ds, images); activate(store, tranche, images)
    full = store.image_queue(ALICE, ds['id'], tranche_id=tranche['id'], queue='work')['images']
    page = store.image_queue(ALICE, ds['id'], tranche_id=tranche['id'], queue='work', limit=3)
    submit(store, ds, page['images'][0])
    following = store.image_queue(ALICE, ds['id'], tranche_id=tranche['id'], queue='work', cursor=page['next_cursor'], limit=3)
    assert [i['id'] for i in following['images']] == [i['id'] for i in full[3:6]]
    anchor = store.image_queue(ALICE, ds['id'], tranche_id=tranche['id'], queue='work', anchor_id=full[8]['id'], limit=3)
    assert anchor['images'][0]['id'] == full[8]['id']
    preceding = store.image_queue(ALICE, ds['id'], tranche_id=tranche['id'], queue='work', cursor=anchor['previous_cursor'], limit=3)
    assert [i['id'] for i in preceding['images']] == [i['id'] for i in full[5:8]]
    error('invalid_cursor', lambda: store.image_queue(ADMIN, ds['id'], tranche_id=tranche['id'], queue='work', cursor=page['next_cursor']))
    error('image_left_queue', lambda: store.queue_scope(ADMIN, page['images'][0]['id'], tranche_id=tranche['id'], queue='work'))
    assert store.queue_scope(ADMIN, page['images'][0]['id'])['scope']['queue'] == 'review'


def test_draft_and_submission_archives_release_reservations_restore_unallocated(store, tmp_path):
    ds = dataset(store); first, second = imported(store, ds, 'first'), imported(store, ds, 'second')
    first_tranche = draft(store, ds, [first], 'Reserved'); second_tranche = draft(store, ds, [second], 'Submitted')
    activate(store, second_tranche, [second]); submitted = submit(store, ds, second)
    ops = HostedOperations(store, ArtifactStore(tmp_path / 'artifacts'), tmp_path)
    for image, tranche in ((first, first_tranche), (submitted, second_tranche)):
        image = store.acquire_editor(ADMIN, image['id'], 'admin')
        archive = ops.archive_image(ADMIN, ctx(image, 'archive-' + image['id'], 'admin'), 'Hold aside')
        assert_counts(store, tranche['id'], archived=1, retained=0)
        current_tranche = store.tranche(ADMIN, tranche['id'])
        store.set_tranche_lifecycle(ADMIN, tranche['id'], current_tranche['revision'], 'close-' + tranche['id'])
        ops.restore_archive(ADMIN, archive['archive_id'], 'restore-' + image['id'])
        restored = store.image(ADMIN, image['id'])
        assert restored['current_version'] == image['current_version'] and restored['state'] == 'unassigned'
        assert store.queue_scope(ADMIN, image['id'])['scope']['queue'] == 'unallocated'
        assert_counts(store, tranche['id'], archived=1)


def test_legacy_receipt_replay_and_reserved_reassign_cannot_bypass(store):
    ds = dataset(store); image = imported(store, ds); tranche = draft(store, ds, [image])
    image = store.acquire_editor(ADMIN, image['id'], 'admin')
    error('image_not_assignable', lambda: store.reassign(ADMIN, ctx(image, 'legacy-reassign', 'admin'), BOB.user_id))
    context = ctx(image, 'historic-reassign', 'admin')
    from dataclasses import asdict
    result = {'historical': 'original result'}
    payload = {'context': asdict(context), 'annotator_id': BOB.user_id}
    store._command(ADMIN, context.operation_id, 'reassign', payload, lambda con: result, admin=True)
    assert store.reassign(ADMIN, context, BOB.user_id) == result
    assert_counts(store, tranche['id'], not_assigned=1)


def test_raw_episode_insert_requires_tranche_and_failed_migration_rolls_back(store):
    with store.transaction() as con:
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("INSERT INTO hosted_assignments(id,dataset_id,annotator_id,created_by,created_at) VALUES('bad','bad','alice','admin',0)")
    with store.read_connection() as con:
        assert con.execute("SELECT version FROM hosted_schema WHERE component='store'").fetchone()[0] == 2
        assert not con.execute('PRAGMA foreign_key_check').fetchall()


def _downgrade_fixture_to_v1(store):
    """Represent the actual old tables, retaining real generated episode/history IDs."""
    with sqlite3.connect(store.path, isolation_level=None) as con:
        con.execute('PRAGMA foreign_keys=OFF'); con.execute('BEGIN IMMEDIATE')
        con.execute('''CREATE TABLE hosted_assignments_v1 (
            id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES hosted_datasets(id),
            annotator_id TEXT NOT NULL, created_by TEXT NOT NULL, created_at REAL NOT NULL)''')
        con.execute('INSERT INTO hosted_assignments_v1 SELECT id,dataset_id,annotator_id,created_by,created_at FROM hosted_assignments')
        con.execute('DROP TABLE hosted_assignments')
        con.execute('ALTER TABLE hosted_assignments_v1 RENAME TO hosted_assignments')
        for table in ('hosted_tranche_events','hosted_tranche_members','hosted_tranches','hosted_migration_outcomes'):
            con.execute('DROP TABLE ' + table)
        con.execute("UPDATE hosted_schema SET version=1 WHERE component='store'")
        con.commit(); con.execute('PRAGMA foreign_keys=ON')
        assert not con.execute('PRAGMA foreign_key_check').fetchall()


def _v1_population(store):
    ds = dataset(store)
    rows = [imported(store, ds, str(n)) for n in range(6)]
    group = store.assign(ADMIN, ds['id'], [i['id'] for i in rows], ALICE.user_id, 'old-assign', expected(rows))
    # One approved, one returned, one submitted, one transferred, one archived
    # submission and one live assigned row exercise the migration branches.
    approved = submit(store, ds, rows[0]); store.approve_from_review_queue(ADMIN, ctx(approved, 'old-approve'))
    returned = submit(store, ds, rows[1]); returned = store.acquire_editor(ADMIN, returned['id'], 'review')
    store.return_image(ADMIN, ctx(returned, 'old-return', 'review'), 'Alice only')
    submit(store, ds, rows[2])
    image = store.image(ADMIN, rows[3]['id'])
    image = store.override(ADMIN, ctx(image, 'old-override'))
    store.assign(ADMIN, ds['id'], [image['id']], BOB.user_id, 'old-assign-bob', expected([image]))
    archived = submit(store, ds, rows[4])
    with store.transaction() as con:
        capture = [{'image_id': archived['id'], 'version_id': archived['current_version'], 'generation': archived['generation'], 'state': 'submitted', 'media_hash': archived['media_hash']}]
        con.execute("INSERT INTO hosted_archives(id,kind,target_id,dataset_id,state,archived_at,actor_id,reason,captured_json) VALUES('legacy-archive','image',?,?,'archived',1000,'admin','old archive',?)", (archived['id'], ds['id'], json.dumps(capture)))
        con.execute("INSERT INTO hosted_archived_images VALUES(?,'legacy-archive',?)", (archived['id'], archived['current_version']))
    _downgrade_fixture_to_v1(store)
    with sqlite3.connect(store.path) as con:
        histories = {table: con.execute('SELECT * FROM ' + table + ' ORDER BY rowid').fetchall() for table in
                     ('hosted_annotation_versions','hosted_submissions','hosted_reviews','hosted_receipts','hosted_events','hosted_assignment_members')}
        states = con.execute('SELECT id,state,assignee_id,assignment_id,current_version,confirmed_version,active_submission,generation FROM hosted_images ORDER BY id').fetchall()
    return ds, archived, histories, states, group


@pytest.mark.parametrize('entrypoint', ['startup', 'restore'])
def test_v1_migration_preserves_history_backfills_every_episode_and_terminal_archives(store, tmp_path, entrypoint):
    ds, archived, histories, states, group = _v1_population(store)
    if entrypoint == 'startup':
        upgraded = HostedStore(store.path, clock=store.clock)
    else:
        from test_hosted_backup import TarTransport
        from services.hosted.backup import create_encrypted_backup, restore_encrypted_backup
        artifacts = ArtifactStore(tmp_path / 'backup-artifacts')
        # This fixture tests the schema seam, not recovery artifact closure (the
        # existing operations suite independently validates full closure).
        backup = create_encrypted_backup(store.path, artifacts, tmp_path / 'migration.tar', recipient='test', references=lambda con: [], age=TarTransport())
        calls = []
        def hook(con, manifest):
            assert con.execute("SELECT version FROM hosted_schema WHERE component='store'").fetchone()[0] == 2
            validate_tranches(con); calls.append(True)
        restored = restore_encrypted_backup(backup.path, tmp_path / 'restored', identity='test', restore_hook=hook, age=TarTransport())
        assert calls == [True]
        upgraded = HostedStore(restored.database_path, clock=store.clock)
    with upgraded.read_connection() as con:
        for table, old in histories.items():
            assert [tuple(row) for row in con.execute('SELECT * FROM ' + table + ' ORDER BY rowid')] == old
        current = {row['id']: row for row in con.execute('SELECT * FROM hosted_images')}
        for before in states:
            after = current[before[0]]
            assert after['current_version'] == before[4] and after['confirmed_version'] == before[5]
            if before[0] == archived['id']:
                assert after['state'] == 'unassigned' and after['assignment_id'] is None and after['active_submission'] is None
                assert after['generation'] > before[7]
            else:
                assert tuple(after[key] for key in ('id','state','assignee_id','assignment_id','current_version','confirmed_version','active_submission','generation')) == before
        assert not con.execute('PRAGMA foreign_key_check').fetchall()
        assert not con.execute('SELECT 1 FROM hosted_assignments WHERE tranche_id IS NULL').fetchone()
        assert con.execute('SELECT actor_kind FROM hosted_migration_outcomes WHERE image_id=?', (archived['id'],)).fetchone()[0] == 'system'
        validate_tranches(con)
    ops = HostedOperations(upgraded, ArtifactStore(tmp_path / ('restore-artifacts-' + entrypoint)), tmp_path)
    ops.restore_archive(ADMIN, 'legacy-archive', 'restore-old-archive')
    assert next(s for s in upgraded.own_submission_status(ALICE) if s['submission_id'] == archived['active_submission'])['status'] == 'archived'
    assert upgraded.image(ADMIN, archived['id'])['state'] == 'unassigned'
    # Pre-upgrade receipts keep their exact original response and operation hash.
    assert upgraded.receipt(ADMIN, 'old-assign')['result'] == group


def test_migration_failure_leaves_v1_and_foreign_keys_enabled(store):
    _, _, _, _, _ = _v1_population(store)
    with sqlite3.connect(store.path, isolation_level=None) as con:
        con.execute('PRAGMA foreign_keys=ON')
        con.execute("UPDATE hosted_images SET assignee_id='wrong' WHERE state='assigned'")
        with pytest.raises(Exception, match='tranche_ownership_inconsistent'):
            migrate_store(con)
        assert con.execute('PRAGMA foreign_keys').fetchone()[0] == 1
        assert con.execute("SELECT version FROM hosted_schema WHERE component='store'").fetchone()[0] == 1
        assert not con.execute("SELECT 1 FROM sqlite_master WHERE name='hosted_tranches'").fetchone()
        assert 'tranche_id' not in [row[1] for row in con.execute('PRAGMA table_info(hosted_assignments)')]


def test_more_than_500_members_activate_and_page_without_full_scan(store):
    ds = dataset(store)
    images = [imported(store, ds, 'queue-' + str(n)) for n in range(605)]
    tranche = draft(store, ds, images[:500], 'Large work')
    tranche = store.add_tranche_members(ADMIN, tranche['id'], tranche['revision'], [i['id'] for i in images[500:]], expected(images[500:]), 'append-large')['tranche']
    group = activate(store, tranche, images)
    assert group['tranche']['counts']['to_finish'] == 605
    seen, cursor, page_count = [], None, 0
    while True:
        page = store.image_queue(ALICE, ds['id'], tranche_id=tranche['id'], queue='work', cursor=cursor, limit=100)
        seen.extend(i['id'] for i in page['images']); page_count += 1
        cursor = page['next_cursor']
        if cursor is None:
            break
    assert page_count == 7 and len(seen) == len(set(seen)) == 605
    assert set(seen) == {i['id'] for i in images}
    deep = store.image_queue(ALICE, ds['id'], tranche_id=tranche['id'], queue='work', anchor_id=seen[550], limit=20)
    assert [i['id'] for i in deep['images']] == seen[550:570]


def test_direct_invalid_revision_and_confirmed_duplicate_remain_safe(store):
    ds = dataset(store); image = imported(store, ds)
    tranche = draft(store, ds, name='Metadata')
    error('tranche_revision_changed', lambda: store.edit_tranche(ADMIN, tranche['id'], None, 'No CAS', '', 'missing-revision'))
    owned = store.acquire_editor(ADMIN, image['id'], 'admin')
    owned = store.save(ADMIN, ctx(owned, 'negative-confirm', 'admin'), {'objects': []}, negative=True)
    confirmed = store.approve(ADMIN, ctx(owned, 'confirm-background', 'admin'))
    duplicate = store.import_image(ADMIN, ds['id'], image['media_hash'], 'copy.png', operation_id='duplicate-pixels', width=100, height=80, size_bytes=100)
    assert duplicate['id'] == confirmed['id']
    error('image_not_assignable', lambda: store.add_tranche_members(ADMIN, tranche['id'], 0, [duplicate['id']], expected([duplicate]), 'attach-confirmed'))
    assert store.image(ADMIN, image['id'])['state'] == 'confirmed'


def test_intake_search_review_filters_and_cursor_binding(store):
    ds = dataset(store)
    free, assigned, submitted, confirmed = [imported(store, ds, name) for name in ('free_100%', 'assigned', 'submitted', 'confirmed')]
    tranche = draft(store, ds, [assigned, submitted]); activate(store, tranche, [assigned, submitted])
    submit(store, ds, submitted)
    image = store.acquire_editor(ADMIN, confirmed['id'], 'admin')
    image = store.save(ADMIN, ctx(image, 'confirmed-save', 'admin'), {'objects': []}, negative=True)
    store.approve(ADMIN, ctx(image, 'confirmed-approve', 'admin'))
    incoming = store.image_queue(ADMIN, ds['id'], queue='incoming', include_total=True)
    assert incoming['total'] == 3
    rows = {i['id']: i for i in incoming['images']}
    assert rows[assigned['id']]['tranche_name'] == 'Work'
    assert rows[assigned['id']]['assignee_name'] == 'alice'
    assert [i['id'] for i in store.image_queue(ADMIN, ds['id'], queue='unallocated', q='100%')['images']] == [free['id']]
    assert store.image_queue(ADMIN, ds['id'], queue='unallocated', q='not_present', include_total=True)['total'] == 0
    review = store.image_queue(ADMIN, ds['id'], queue='review', tranche_id=tranche['id'], annotator_id=ALICE.user_id)
    assert [i['id'] for i in review['images']] == [submitted['id']]
    assert store.image_queue(ADMIN, ds['id'], queue='review', annotator_id=BOB.user_id, include_total=True)['total'] == 0
    first = store.image_queue(ADMIN, ds['id'], queue='incoming', limit=1)
    assert 'total' not in first
    error('invalid_cursor', lambda: store.image_queue(ADMIN, ds['id'], queue='incoming', q='changed', cursor=first['next_cursor']))
    error('admin_required', lambda: store.image_queue(ALICE, ds['id'], queue='incoming'))
    error('admin_required', lambda: store.image_queue(ALICE, ds['id'], tranche_id=tranche['id'], queue='work', annotator_id=BOB.user_id))
