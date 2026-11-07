"""Named work packages over the existing image/version/assignment authority.

No independent progress counters, media copies or nested command transactions.
The store's handoff hook settles current membership even for legacy callers.
"""
from __future__ import annotations

import base64
from dataclasses import asdict
import json
import sqlite3
import uuid

from .errors import DomainError
from .preconditions import validate_precondition

SELECTION_LIMIT = 500
TRANCHE_LIMIT = 10000
_NO_REVISION = object()
BUCKETS = {'reserved': 'not_assigned', 'assigned': 'to_finish', 'returned': 'changes_requested',
           'submitted': 'awaiting_review', 'approved': 'approved', 'withdrawn': 'removed',
           'transferred': 'transferred', 'archived': 'archived'}


def _id(prefix):
    return prefix + '_' + uuid.uuid4().hex


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _string(value, field, maximum, *, empty=False):
    if not isinstance(value, str) or len(value) > maximum or '\x00' in value or (not empty and not value.strip()):
        raise DomainError('invalid_field', 422, {'field': field})
    return value.strip()


def _cursor(scope, key, direction='forward'):
    return base64.urlsafe_b64encode(_json({'scope': scope, 'key': key, 'direction': direction}).encode()).decode().rstrip('=')


def _decode_cursor(value, scope):
    try:
        if not isinstance(value, str) or len(value) > 4096:
            raise ValueError()
        payload = json.loads(base64.urlsafe_b64decode(value + '=' * (-len(value) % 4)))
        if payload['scope'] != scope or payload['direction'] not in ('forward', 'backward'):
            raise ValueError()
        return payload['key'], payload['direction']
    except (ValueError, KeyError, TypeError, UnicodeError) as exc:
        raise DomainError('invalid_cursor', 422) from exc


class TrancheMixin:
    @staticmethod
    def _member(con, image_id):
        return con.execute('SELECT * FROM hosted_tranche_members WHERE image_id=? AND is_current=1', (image_id,)).fetchone()

    def _tranche(self, con, tranche_id, *, actor=None, revision=_NO_REVISION):
        row = con.execute('SELECT * FROM hosted_tranches WHERE id=?', (tranche_id,)).fetchone()
        if row is None:
            raise DomainError('tranche_not_found', 404)
        self._dataset(con, row['dataset_id'])
        if actor and actor.role != 'admin' and not con.execute(
                'SELECT 1 FROM hosted_tranche_members WHERE tranche_id=? AND annotator_id=? LIMIT 1',
                (tranche_id, actor.user_id)).fetchone():
            raise DomainError('tranche_not_found', 404)
        if revision is not _NO_REVISION and (type(revision) is not int or revision != row['revision']):
            raise DomainError('tranche_revision_changed', 409, {'tranche_id': tranche_id, 'revision': row['revision']})
        return row

    def _touch_tranche(self, con, tranche_id, actor_id, kind, data):
        now = self.clock()
        con.execute('UPDATE hosted_tranches SET revision=revision+1,updated_at=?,updated_by=? WHERE id=?',
                    (now, actor_id, tranche_id))
        self._tranche_event(con, tranche_id, actor_id, kind, data)

    def _tranche_event(self, con, tranche_id, actor_id, kind, data):
        con.execute('INSERT INTO hosted_tranche_events(tranche_id,actor_id,kind,data_json,created_at) VALUES(?,?,?,?,?)',
                    (tranche_id, actor_id, kind, _json(data), self.clock()))
        con.execute('UPDATE hosted_tranches SET updated_at=? WHERE id=?', (self.clock(), tranche_id))

    def _new_tranche(self, con, actor, dataset_id, name, instructions='', annotator_id=None):
        self._dataset(con, dataset_id)
        name = _string(name, 'name', 200)
        instructions = _string(instructions, 'instructions', 20000, empty=True)
        if annotator_id is not None:
            self._annotator(con, annotator_id)
        tranche_id, now = _id('tranche'), self.clock()
        con.execute('INSERT INTO hosted_tranches VALUES(?,?,?,?,?,?,?,?,?,?,?)', (
            tranche_id, dataset_id, name, instructions, annotator_id, 'draft', 0, actor.user_id, actor.user_id, now, now))
        self._tranche_event(con, tranche_id, actor.user_id, 'created', {'name': name})
        return self._tranche(con, tranche_id)

    def _reserve(self, con, actor, tranche_id, images):
        for image in images:
            con.execute('''INSERT INTO hosted_tranche_members
                (id,tranche_id,image_id,assignment_id,annotator_id,outcome,is_current,joined_at,ended_at,reason,joined_by,ended_by)
                VALUES(?,?,?,NULL,NULL,'reserved',1,?,NULL,'',?,NULL)''',
                (_id('member'), tranche_id, image['id'], self.clock(), actor.user_id))

    def _finish_member(self, con, member, outcome, actor_id, reason='', *, structural=False):
        con.execute('UPDATE hosted_tranche_members SET is_current=0,outcome=?,ended_at=?,ended_by=?,reason=? WHERE id=? AND is_current=1',
                    (outcome, self.clock(), actor_id, reason, member['id']))
        data = {'image_id': member['image_id'], 'membership_id': member['id'], 'reason': reason}
        if structural:
            self._touch_tranche(con, member['tranche_id'], actor_id, outcome, data)
        else:
            self._tranche_event(con, member['tranche_id'], actor_id, outcome, data)

    def _tranche_handoff(self, con, image, state, assignee, assignment, actor_id):
        """Called by *every* image handoff, inside its existing transaction."""
        member = self._member(con, image['id'])
        if state in ('assigned', 'returned', 'submitted'):
            episode = con.execute('SELECT * FROM hosted_assignments WHERE id=?', (assignment,)).fetchone()
            if not member or not episode:
                raise DomainError('tranche_ownership_inconsistent', 409)
            tranche = self._tranche(con, member['tranche_id'])
            if (tranche['lifecycle'] != 'active' or tranche['annotator_id'] != assignee
                    or episode['tranche_id'] != tranche['id'] or episode['annotator_id'] != assignee
                    or episode['dataset_id'] != image['dataset_id']
                    or not con.execute('SELECT 1 FROM hosted_assignment_members WHERE assignment_id=? AND image_id=?', (assignment, image['id'])).fetchone()):
                raise DomainError('tranche_ownership_inconsistent', 409)
            con.execute('UPDATE hosted_tranche_members SET outcome=?,assignment_id=?,annotator_id=? WHERE id=?',
                        (state, assignment, assignee, member['id']))
            self._tranche_event(con, tranche['id'], actor_id, state, {'image_id': image['id'], 'assignment_id': assignment})
        elif member:
            outcome = 'approved' if state == 'confirmed' else 'withdrawn'
            # Ownership recoveries alter membership structure. Normal approval
            # changes progress only and cannot make an unrelated title edit stale.
            self._finish_member(con, member, outcome, actor_id, structural=state == 'unassigned')

    def _episode(self, con, actor, tranche, images, annotator_id):
        self._annotator(con, annotator_id)
        assignment_id = _id('assignment')
        con.execute('INSERT INTO hosted_assignments(id,dataset_id,annotator_id,created_by,created_at,tranche_id) VALUES(?,?,?,?,?,?)',
                    (assignment_id, tranche['dataset_id'], annotator_id, actor.user_id, self.clock(), tranche['id']))
        for image in images:
            con.execute("INSERT INTO hosted_assignment_members VALUES(?,?,?,'assigned')", (assignment_id, image['id'], image['current_version']))
            self._handoff(con, image['id'], 'assigned', assignee=annotator_id, assignment=assignment_id, actor_id=actor.user_id)
            self._event(con, actor, tranche['dataset_id'], image['id'], 'assigned', {'assignment_id': assignment_id, 'tranche_id': tranche['id'], 'annotator_id': annotator_id})
        return assignment_id

    def _capture_tranche_images(self, con, tranche, image_ids, expected, *, member=False, whole=False, allow_empty=False, idle=False):
        bound = TRANCHE_LIMIT if whole else SELECTION_LIMIT
        if (not isinstance(image_ids, list) or len(image_ids) > bound or (not image_ids and not allow_empty)
                or any(not isinstance(v, str) for v in image_ids) or len(set(image_ids)) != len(image_ids)
                or not isinstance(expected, dict) or set(expected) != set(image_ids)):
            raise DomainError('exact_assignment_preconditions_required', 422, {'maximum': bound})
        if whole:
            current = {r[0] for r in con.execute('SELECT image_id FROM hosted_tranche_members WHERE tranche_id=? AND is_current=1', (tranche['id'],))}
            if current != set(image_ids):
                raise DomainError('tranche_membership_changed', 409)
        images = []
        for identifier in image_ids:
            image = self._image(con, identifier)
            membership = self._member(con, identifier)
            if image['dataset_id'] != tranche['dataset_id']:
                raise DomainError('image_dataset_mismatch', 409, {'image_id': identifier})
            if member:
                if not membership or membership['tranche_id'] != tranche['id'] or image['state'] not in ('unassigned','assigned','returned'):
                    raise DomainError('tranche_member_not_movable', 409, {'image_id': identifier})
            elif image['state'] != 'unassigned' or membership:
                raise DomainError('image_not_assignable', 409, {'image_id': identifier})
            pre = validate_precondition(expected[identifier], field='expected.' + identifier)
            self._expected(image, pre.get('expected_version', pre.get('version_id')), pre['generation'])
            if idle or image['state'] == 'unassigned':
                lease = con.execute('SELECT expires_at FROM hosted_editor_leases WHERE image_id=?', (identifier,)).fetchone()
                if lease and lease[0] > self.clock():
                    raise DomainError('editor_lease_conflict', 423, {'image_id': identifier})
            images.append(image)
        return images

    @staticmethod
    def _capacity(con, tranche_id, adding):
        current = con.execute('SELECT COUNT(*) FROM hosted_tranche_members WHERE tranche_id=? AND is_current=1', (tranche_id,)).fetchone()[0]
        if current + adding > TRANCHE_LIMIT:
            raise DomainError('tranche_capacity_exceeded', 409, {'maximum': TRANCHE_LIMIT, 'current': current, 'adding': adding})

    def _latest_members_sql(self, actor):
        # rowid orders intervals even with equal timestamps and random UUIDs.
        if actor.role == 'admin':
            return 'SELECT MAX(rowid) FROM hosted_tranche_members WHERE tranche_id=? GROUP BY image_id', []
        return 'SELECT MAX(rowid) FROM hosted_tranche_members WHERE tranche_id=? AND annotator_id=? GROUP BY image_id', [actor.user_id]

    def _tranche_summaries(self, con, actor, tranche_ids):
        """Two grouped reads for a bounded metadata page, independent of image count."""
        if not tranche_ids:
            return []
        marks = ','.join('?' for _ in tranche_ids)
        personal = actor.role != 'admin'
        personal_sql = ' AND annotator_id=?' if personal else ''
        counts_by_id = {identifier: dict.fromkeys(BUCKETS.values(), 0) for identifier in tranche_ids}
        query = ('SELECT tranche_id,outcome,COUNT(*) AS n FROM hosted_tranche_members WHERE rowid IN '
                 '(SELECT MAX(rowid) FROM hosted_tranche_members WHERE tranche_id IN (' + marks + ')' +
                 personal_sql + ' GROUP BY tranche_id,image_id) GROUP BY tranche_id,outcome')
        for row in con.execute(query, [*tranche_ids, *([actor.user_id] if personal else [])]):
            counts_by_id[row['tranche_id']][BUCKETS[row['outcome']]] = row['n']
        users = 'u.id=?' if personal else 'u.id=t.annotator_id'
        rows = con.execute('SELECT t.*,d.name AS dataset_name,COALESCE(u.display_name,u.username) AS annotator_name,u.role AS annotator_role,'
            'u.disabled AS annotator_disabled FROM hosted_tranches t JOIN hosted_datasets d ON d.id=t.dataset_id '
            'LEFT JOIN hosted_users u ON ' + users + ' WHERE t.id IN (' + marks + ')',
            [*([actor.user_id] if personal else []), *tranche_ids])
        by_id = {}
        for row in rows:
            tranche = dict(row); counts = counts_by_id[row['id']]
            counts['ever_included'] = sum(counts.values())
            counts['retained'] = counts['ever_included'] - counts['removed'] - counts['transferred'] - counts['archived']
            role = tranche.pop('annotator_role')
            disabled = tranche.pop('annotator_disabled')
            available = role == 'annotator' and not disabled
            if personal:
                tranche['annotator_id'] = actor.user_id
            tranche.update(annotator_available=available, counts=counts, personal=personal,
                           selection_limit=SELECTION_LIMIT, group_limit=TRANCHE_LIMIT)
            by_id[row['id']] = tranche
        return [by_id[identifier] for identifier in tranche_ids]

    def _tranche_result(self, con, actor, tranche_id):
        self._tranche(con, tranche_id, actor=actor)
        return self._tranche_summaries(con, actor, [tranche_id])[0]

    def tranche(self, actor, tranche_id):
        with self.read_connection() as con:
            self._actor(con, actor)
            return self._tranche_result(con, actor, tranche_id)

    def create_tranche(self, actor, dataset_id, name, operation_id, *, instructions='', annotator_id=None, image_ids=None, expected=None):
        image_ids = [] if image_ids is None else image_ids
        expected = {} if expected is None else expected
        payload = dict(dataset_id=dataset_id, name=name, instructions=instructions, annotator_id=annotator_id, image_ids=image_ids, expected=expected)
        def run(con):
            tranche = self._new_tranche(con, actor, dataset_id, name, instructions, annotator_id)
            images = self._capture_tranche_images(con, tranche, image_ids, expected, allow_empty=True, idle=True)
            self._reserve(con, actor, tranche['id'], images)
            return {'tranche': self._tranche_result(con, actor, tranche['id']), 'images': [self._image_result(con, i['id']) for i in images]}
        return self._command(actor, operation_id, 'create_tranche', payload, run, admin=True)

    def edit_tranche(self, actor, tranche_id, expected_revision, name, instructions, operation_id, *, annotator_id=None, set_annotator=False):
        payload = dict(tranche_id=tranche_id, expected_revision=expected_revision, name=name, instructions=instructions,
                       annotator_id=annotator_id, set_annotator=set_annotator)
        def run(con):
            tranche = self._tranche(con, tranche_id, revision=expected_revision)
            if tranche['lifecycle'] == 'closed':
                raise DomainError('tranche_closed', 409)
            text = _string(name, 'name', 200); guide = _string(instructions, 'instructions', 20000, empty=True)
            if set_annotator:
                if tranche['lifecycle'] != 'draft':
                    raise DomainError('use_tranche_assignee_command', 409)
                if annotator_id is not None:
                    self._annotator(con, annotator_id)
                con.execute('UPDATE hosted_tranches SET annotator_id=? WHERE id=?', (annotator_id, tranche_id))
            con.execute('UPDATE hosted_tranches SET name=?,instructions=? WHERE id=?', (text, guide, tranche_id))
            self._touch_tranche(con, tranche_id, actor.user_id, 'edited', {'name': text})
            return {'tranche': self._tranche_result(con, actor, tranche_id)}
        return self._command(actor, operation_id, 'edit_tranche', payload, run, admin=True)

    def add_tranche_members(self, actor, tranche_id, expected_revision, image_ids, expected, operation_id):
        payload = dict(tranche_id=tranche_id, expected_revision=expected_revision, image_ids=image_ids, expected=expected)
        def run(con):
            tranche = self._tranche(con, tranche_id, revision=expected_revision)
            if tranche['lifecycle'] == 'closed':
                raise DomainError('tranche_closed', 409)
            images = self._capture_tranche_images(con, tranche, image_ids, expected, idle=True)
            self._capacity(con, tranche_id, len(images))
            self._reserve(con, actor, tranche_id, images)
            if tranche['lifecycle'] == 'active':
                self._episode(con, actor, tranche, images, tranche['annotator_id'])
            self._touch_tranche(con, tranche_id, actor.user_id, 'members_added', {'image_ids': image_ids})
            return {'tranche': self._tranche_result(con, actor, tranche_id), 'images': [self._image_result(con, i['id']) for i in images]}
        return self._command(actor, operation_id, 'add_tranche_members', payload, run, admin=True)

    def activate_tranche(self, actor, tranche_id, expected_revision, image_ids, expected, annotator_id, operation_id):
        payload = dict(tranche_id=tranche_id, expected_revision=expected_revision, image_ids=image_ids, expected=expected, annotator_id=annotator_id)
        def run(con):
            tranche = self._tranche(con, tranche_id, revision=expected_revision)
            if tranche['lifecycle'] != 'draft':
                raise DomainError('tranche_not_draft', 409)
            self._annotator(con, annotator_id)
            images = self._capture_tranche_images(con, tranche, image_ids, expected, member=True, whole=True, idle=True)
            con.execute("UPDATE hosted_tranches SET lifecycle='active',annotator_id=? WHERE id=?", (annotator_id, tranche_id))
            tranche = self._tranche(con, tranche_id)
            self._episode(con, actor, tranche, images, annotator_id)
            self._touch_tranche(con, tranche_id, actor.user_id, 'activated', {'image_ids': image_ids, 'annotator_id': annotator_id})
            return {'tranche': self._tranche_result(con, actor, tranche_id), 'images': [self._image_result(con, i['id']) for i in images]}
        return self._command(actor, operation_id, 'activate_tranche', payload, run, admin=True)

    def remove_tranche_members(self, actor, tranche_id, expected_revision, image_ids, expected, reason, operation_id):
        payload = dict(tranche_id=tranche_id, expected_revision=expected_revision, image_ids=image_ids, expected=expected, reason=reason)
        def run(con):
            tranche = self._tranche(con, tranche_id, revision=expected_revision)
            explanation = _string(reason, 'reason', 10000)
            images = self._capture_tranche_images(con, tranche, image_ids, expected, member=True)
            for image in images:
                self._finish_member(con, self._member(con, image['id']), 'withdrawn', actor.user_id, explanation)
                self._handoff(con, image['id'], 'unassigned', assignee=None, assignment=None, actor_id=actor.user_id)
                self._event(con, actor, tranche['dataset_id'], image['id'], 'ownership_overridden', {'reason': explanation, 'tranche_id': tranche_id})
            self._touch_tranche(con, tranche_id, actor.user_id, 'members_removed', {'image_ids': image_ids, 'reason': explanation})
            return {'tranche': self._tranche_result(con, actor, tranche_id), 'images': [self._image_result(con, i['id']) for i in images]}
        return self._command(actor, operation_id, 'remove_tranche_members', payload, run, admin=True)

    def move_tranche_members(self, actor, tranche_id, expected_revision, destination_id, expected_destination_revision, image_ids, expected, reason, operation_id):
        payload = dict(tranche_id=tranche_id, expected_revision=expected_revision, destination_id=destination_id,
                       expected_destination_revision=expected_destination_revision, image_ids=image_ids, expected=expected, reason=reason)
        def run(con):
            source = self._tranche(con, tranche_id, revision=expected_revision)
            target = self._tranche(con, destination_id, revision=expected_destination_revision)
            if source['dataset_id'] != target['dataset_id'] or tranche_id == destination_id:
                raise DomainError('invalid_tranche_destination', 409)
            if target['lifecycle'] == 'closed':
                raise DomainError('tranche_closed', 409)
            explanation = _string(reason, 'reason', 10000)
            images = self._capture_tranche_images(con, source, image_ids, expected, member=True)
            self._capacity(con, destination_id, len(images))
            for image in images:
                self._finish_member(con, self._member(con, image['id']), 'transferred', actor.user_id, explanation)
                self._handoff(con, image['id'], 'unassigned', assignee=None, assignment=None, actor_id=actor.user_id)
                self._event(con, actor, source['dataset_id'], image['id'], 'tranche_transferred', {'source_id': tranche_id, 'destination_id': destination_id})
            self._reserve(con, actor, destination_id, images)
            if target['lifecycle'] == 'active':
                self._episode(con, actor, target, images, target['annotator_id'])
            self._touch_tranche(con, tranche_id, actor.user_id, 'members_transferred', {'image_ids': image_ids, 'destination_id': destination_id})
            self._touch_tranche(con, destination_id, actor.user_id, 'members_received', {'image_ids': image_ids, 'source_id': tranche_id})
            return {'tranche': self._tranche_result(con, actor, tranche_id), 'destination': self._tranche_result(con, actor, destination_id), 'images': [self._image_result(con, i['id']) for i in images]}
        return self._command(actor, operation_id, 'move_tranche_members', payload, run, admin=True)

    def change_tranche_assignee(self, actor, tranche_id, expected_revision, annotator_id, image_ids, expected, operation_id):
        payload = dict(tranche_id=tranche_id, expected_revision=expected_revision, annotator_id=annotator_id, image_ids=image_ids, expected=expected)
        def run(con):
            tranche = self._tranche(con, tranche_id, revision=expected_revision)
            if tranche['lifecycle'] != 'active':
                raise DomainError('tranche_not_active', 409)
            if con.execute("SELECT 1 FROM hosted_tranche_members WHERE tranche_id=? AND is_current=1 AND outcome='submitted'", (tranche_id,)).fetchone():
                raise DomainError('tranche_has_pending_review', 409, {'actions': ['review_submissions', 'move_unfinished_members']})
            self._annotator(con, annotator_id)
            images = self._capture_tranche_images(con, tranche, image_ids, expected, member=True, whole=True, allow_empty=True)
            for image in images:
                self._finish_member(con, self._member(con, image['id']), 'transferred', actor.user_id, 'Annotator changed')
                self._handoff(con, image['id'], 'unassigned', assignee=None, assignment=None, actor_id=actor.user_id)
            con.execute('UPDATE hosted_tranches SET annotator_id=? WHERE id=?', (annotator_id, tranche_id))
            self._reserve(con, actor, tranche_id, images)
            if images:
                self._episode(con, actor, self._tranche(con, tranche_id), images, annotator_id)
            self._touch_tranche(con, tranche_id, actor.user_id, 'annotator_changed', {'annotator_id': annotator_id, 'image_ids': image_ids})
            return {'tranche': self._tranche_result(con, actor, tranche_id), 'images': [self._image_result(con, i['id']) for i in images]}
        return self._command(actor, operation_id, 'change_tranche_assignee', payload, run, admin=True)

    def set_tranche_lifecycle(self, actor, tranche_id, expected_revision, operation_id, *, reopen=False):
        payload = dict(tranche_id=tranche_id, expected_revision=expected_revision)
        def run(con):
            tranche = self._tranche(con, tranche_id, revision=expected_revision)
            if reopen:
                if tranche['lifecycle'] != 'closed':
                    raise DomainError('tranche_not_closed', 409)
                state = 'draft'
            else:
                if tranche['lifecycle'] == 'closed':
                    raise DomainError('tranche_closed', 409)
                if con.execute('SELECT 1 FROM hosted_tranche_members WHERE tranche_id=? AND is_current=1', (tranche_id,)).fetchone():
                    raise DomainError('tranche_has_unfinished_members', 409)
                state = 'closed'
            con.execute('UPDATE hosted_tranches SET lifecycle=? WHERE id=?', (state, tranche_id))
            self._touch_tranche(con, tranche_id, actor.user_id, 'reopened' if reopen else 'closed', {})
            return {'tranche': self._tranche_result(con, actor, tranche_id)}
        return self._command(actor, operation_id, 'reopen_tranche' if reopen else 'close_tranche', payload, run, admin=True)

    def withdraw_submission(self, actor, ctx, submission_id, reason):
        payload = dict(context=asdict(ctx), submission_id=submission_id, reason=reason)
        def run(con):
            image = self._image(con, ctx.image_id)
            self._expected(image, ctx.expected_version, ctx.generation)
            _string(ctx.editor_id, 'editor_id', 200)
            if image['state'] != 'submitted' or image['active_submission'] != submission_id:
                raise DomainError('submission_changed', 409)
            lease = con.execute('SELECT * FROM hosted_editor_leases WHERE image_id=?', (ctx.image_id,)).fetchone()
            if lease and lease['expires_at'] > self.clock():
                self._guard(con, actor, ctx)
            explanation = _string(reason, 'reason', 10000)
            review_id = self._review(con, actor, image, 'withdrawn', explanation)
            member = self._member(con, image['id'])
            if member:
                self._finish_member(con, member, 'withdrawn', actor.user_id, explanation, structural=True)
            self._handoff(con, ctx.image_id, 'unassigned', assignee=None, assignment=None, actor_id=actor.user_id)
            self._event(con, actor, image['dataset_id'], image['id'], 'submission_withdrawn', {'review_id': review_id, 'reason': explanation})
            return self._image_result(con, image['id'])
        return self._command(actor, ctx.operation_id, 'withdraw_submission', payload, run, admin=True)

    def list_tranches(self, actor, *, dataset_id=None, lifecycle=None, annotator_id=None, q='', cursor=None, limit=50):
        self._pagination(limit, 0)
        if type(limit) is not int or limit > 500:
            raise DomainError('invalid_pagination', 422)
        if lifecycle not in (None, 'open', 'draft', 'active', 'closed'):
            raise DomainError('invalid_tranche_lifecycle', 422)
        q = _string(q, 'q', 200, empty=True)
        scope = [actor.user_id, 'tranches', dataset_id, lifecycle, annotator_id, q]
        with self.read_connection() as con:
            self._actor(con, actor)
            conditions = ['t.dataset_id NOT IN (SELECT dataset_id FROM hosted_archived_datasets)']; args = []
            if dataset_id:
                self._dataset(con, dataset_id)
                conditions.append('t.dataset_id=?'); args.append(dataset_id)
            if lifecycle == 'open':
                conditions.append("t.lifecycle IN ('draft','active')")
            elif lifecycle:
                conditions.append('t.lifecycle=?'); args.append(lifecycle)
            if actor.role != 'admin':
                if annotator_id not in (None, actor.user_id):
                    raise DomainError('admin_required', 403)
                conditions.append('EXISTS(SELECT 1 FROM hosted_tranche_members m WHERE m.tranche_id=t.id AND m.annotator_id=?)')
                args.append(actor.user_id)
            elif annotator_id:
                conditions.append('t.annotator_id=?'); args.append(annotator_id)
            if q:
                conditions.append('instr(lower(t.name),lower(?))>0'); args.append(q)
            if cursor:
                key, _ = _decode_cursor(cursor, scope)
                if not isinstance(key, list) or len(key) != 2 or type(key[0]) not in (int, float) or not isinstance(key[1], str):
                    raise DomainError('invalid_cursor', 422)
                conditions.append('(t.created_at,t.id)>(?,?)'); args.extend(key)
            rows = con.execute('SELECT t.id,t.created_at FROM hosted_tranches t WHERE ' + ' AND '.join(conditions) + ' ORDER BY t.created_at,t.id LIMIT ?', [*args, limit + 1]).fetchall()
            page = rows[:limit]
            return {'tranches': self._tranche_summaries(con, actor, [row['id'] for row in page]),
                    'next_cursor': _cursor(scope, [page[-1]['created_at'], page[-1]['id']]) if len(rows) > limit else None}

    def tranche_members(self, actor, tranche_id, *, state='all', cursor=None, limit=100):
        self._pagination(limit, 0)
        if type(limit) is not int or limit > 500:
            raise DomainError('invalid_pagination', 422)
        if state != 'all' and state not in BUCKETS.values():
            raise DomainError('invalid_member_state', 422)
        scope = [actor.user_id, 'members', tranche_id, state]
        with self.read_connection() as con:
            self._actor(con, actor)
            tranche = self._tranche_result(con, actor, tranche_id)
            latest, personal_args = self._latest_members_sql(actor)
            conditions = ['m.rowid IN (' + latest + ')']; args = [tranche_id, *personal_args]
            if state != 'all':
                conditions.append('m.outcome=?'); args.append(next(k for k, v in BUCKETS.items() if v == state))
            if cursor:
                key, _ = _decode_cursor(cursor, scope)
                if type(key) is not int:
                    raise DomainError('invalid_cursor', 422)
                conditions.append('m.rowid>?'); args.append(key)
            rows = con.execute('SELECT m.rowid AS ordinal,m.*,i.filename FROM hosted_tranche_members m JOIN hosted_images i ON i.id=m.image_id WHERE ' + ' AND '.join(conditions) + ' ORDER BY m.rowid LIMIT ?', [*args, limit + 1]).fetchall()
            members = []
            for row in rows[:limit]:
                item = {'membership_id': row['id'], 'image_id': row['image_id'], 'filename': row['filename'],
                        'state': BUCKETS[row['outcome']], 'outcome': row['outcome'], 'assignment_id': row['assignment_id'],
                        'annotator_id': row['annotator_id'], 'joined_at': row['joined_at'], 'ended_at': row['ended_at'],
                        'image': None, 'can_view': False}
                try:
                    image = self._image(con, row['image_id'])
                    self._can_view(actor, image)
                    # A former owner's personal row cannot become a window onto
                    # a later episode even when reassigned to them in another tranche.
                    if actor.role == 'admin' or (row['is_current'] and image['assignment_id'] == row['assignment_id']):
                        item['image'] = self._image_result(con, row['image_id'])
                        item['can_view'] = True
                except DomainError as exc:
                    if exc.code != 'image_not_found':
                        raise
                members.append(item)
            return {'tranche': tranche, 'members': members,
                    'next_cursor': _cursor(scope, rows[limit - 1]['ordinal']) if len(rows) > limit else None}

    def tranche_history(self, actor, tranche_id, *, cursor=None, limit=100):
        self._pagination(limit, 0)
        if type(limit) is not int or limit > 500:
            raise DomainError('invalid_pagination', 422)
        scope = [actor.user_id, 'history', tranche_id]
        with self.read_connection() as con:
            self._actor(con, actor, admin=True)
            self._tranche(con, tranche_id)
            key = 0
            if cursor:
                key, _ = _decode_cursor(cursor, scope)
                if type(key) is not int:
                    raise DomainError('invalid_cursor', 422)
            rows = con.execute('SELECT * FROM hosted_tranche_events WHERE tranche_id=? AND id>? ORDER BY id LIMIT ?', (tranche_id, key, limit + 1)).fetchall()
            events = []
            for row in rows[:limit]:
                value = dict(row); value['data'] = json.loads(value.pop('data_json')); events.append(value)
            return {'events': events, 'next_cursor': _cursor(scope, events[-1]['id']) if len(rows) > limit else None}

    def _queue_where(self, con, actor, dataset_id, tranche_id, queue):
        self._dataset(con, dataset_id)
        conditions = ['i.dataset_id=?', 'NOT EXISTS(SELECT 1 FROM hosted_archived_images a WHERE a.image_id=i.id)']
        args = [dataset_id]
        tranche = self._tranche(con, tranche_id, actor=actor) if tranche_id else None
        if tranche and tranche['dataset_id'] != dataset_id:
            raise DomainError('queue_scope_mismatch', 409)
        if actor.role != 'admin' and (queue != 'work' or not tranche):
            raise DomainError('admin_required', 403)
        if queue in ('work', 'draft'):
            if not tranche:
                raise DomainError('tranche_required', 422)
            wanted = 'active' if queue == 'work' else 'draft'
            if tranche['lifecycle'] != wanted:
                raise DomainError('queue_scope_changed', 409)
            conditions.append('EXISTS(SELECT 1 FROM hosted_tranche_members m WHERE m.image_id=i.id AND m.is_current=1 AND m.tranche_id=?)'); args.append(tranche_id)
            conditions.append("i.state IN ('assigned','returned')" if queue == 'work' else "i.state='unassigned'")
            if actor.role != 'admin':
                conditions.append('i.assignee_id=?'); args.append(actor.user_id)
        elif queue == 'review':
            conditions.append("i.state='submitted'")
            if tranche:
                conditions.append('EXISTS(SELECT 1 FROM hosted_tranche_members m WHERE m.image_id=i.id AND m.is_current=1 AND m.tranche_id=?)'); args.append(tranche_id)
        elif queue == 'unallocated':
            if tranche:
                raise DomainError('queue_scope_mismatch', 409)
            conditions.extend(["i.state='unassigned'", 'NOT EXISTS(SELECT 1 FROM hosted_tranche_members m WHERE m.image_id=i.id AND m.is_current=1)'])
        elif queue == 'incoming':
            if tranche:
                raise DomainError('queue_scope_mismatch', 409)
            conditions.append("i.state IN ('unassigned','assigned','returned','submitted')")
        elif queue == 'confirmed':
            if tranche:
                raise DomainError('queue_scope_mismatch', 409)
            conditions.append("i.state='confirmed'")
        else:
            raise DomainError('invalid_queue', 422)
        return conditions, args

    def queue_scope(self, actor, image_id, *, tranche_id=None, queue=None):
        with self.read_connection() as con:
            self._actor(con, actor)
            image = self._image(con, image_id); self._can_view(actor, image)
            member = self._member(con, image_id)
            if queue is None and tranche_id is None:
                if member:
                    tranche_id = member['tranche_id']
                    queue = 'draft' if member['outcome'] == 'reserved' else 'review' if image['state'] == 'submitted' else 'work'
                else:
                    queue = 'confirmed' if image['state'] == 'confirmed' else 'unallocated'
            elif queue is None:
                raise DomainError('queue_required', 422)
            conditions, args = self._queue_where(con, actor, image['dataset_id'], tranche_id, queue)
            if not con.execute('SELECT 1 FROM hosted_images i WHERE ' + ' AND '.join(conditions) + ' AND i.id=?', [*args, image_id]).fetchone():
                raise DomainError('image_left_queue', 409, {'tranche_id': tranche_id, 'queue': queue})
            return {'scope': {'dataset_id': image['dataset_id'], 'tranche_id': tranche_id, 'queue': queue},
                    'tranche': self._tranche_result(con, actor, tranche_id) if tranche_id else None}

    def image_queue(self, actor, dataset_id, *, tranche_id=None, queue='confirmed', cursor=None, limit=100, anchor_id=None, direction='forward', q='', annotator_id=None, include_total=False):
        self._pagination(limit, 0)
        if type(limit) is not int or limit > 500:
            raise DomainError('invalid_pagination', 422)
        if direction not in ('forward', 'backward') or (cursor and anchor_id):
            raise DomainError('invalid_cursor', 422)
        if not isinstance(q, str) or len(q) > 200:
            raise DomainError('invalid_search', 422)
        q = q.strip()
        scope = {'dataset_id': dataset_id, 'tranche_id': tranche_id, 'queue': queue}
        cursor_scope = [actor.user_id, 'queue', scope, q, annotator_id]
        with self.read_connection() as con:
            self._actor(con, actor)
            conditions, args = self._queue_where(con, actor, dataset_id, tranche_id, queue)
            if annotator_id:
                if actor.role != 'admin':
                    raise DomainError('admin_required', 403)
                conditions.append('i.assignee_id=?'); args.append(annotator_id)
            if q:
                conditions.append("i.filename LIKE ? ESCAPE '\\'")
                args.append('%' + q.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_') + '%')
            where = ' AND '.join(conditions)
            boundary, page_args, key = '', list(args), None
            inclusive = False
            if cursor:
                key, direction = _decode_cursor(cursor, cursor_scope)
                if not isinstance(key, list) or len(key) != 2 or type(key[0]) not in (float, int) or not isinstance(key[1], str):
                    raise DomainError('invalid_cursor', 422)
            elif anchor_id:
                anchor = con.execute('SELECT i.created_at,i.id FROM hosted_images i WHERE ' + where + ' AND i.id=?', [*args, anchor_id]).fetchone()
                if not anchor:
                    raise DomainError('image_left_queue', 409, {'tranche_id': tranche_id, 'queue': queue})
                key = [anchor['created_at'], anchor['id']]; inclusive = True
            if key:
                compare = ('>' if direction == 'forward' else '<') + ('=' if inclusive else '')
                boundary = ' AND (i.created_at,i.id)' + compare + '(?,?)'; page_args.extend(key)
            ordering = 'ASC' if direction == 'forward' else 'DESC'
            rows = con.execute('''SELECT i.*,v.id AS version_id,v.complete,v.negative,v.class_version,d.revision AS dataset_revision,
                (SELECT m.tranche_id FROM hosted_tranche_members m WHERE m.image_id=i.id AND m.is_current=1) AS tranche_id,
                (SELECT t.name FROM hosted_tranche_members m JOIN hosted_tranches t ON t.id=m.tranche_id WHERE m.image_id=i.id AND m.is_current=1) AS tranche_name,
                (SELECT COALESCE(u.display_name,u.username) FROM hosted_users u WHERE u.id=i.assignee_id) AS assignee_name
                FROM hosted_images i JOIN hosted_annotation_versions v ON v.id=i.current_version JOIN hosted_datasets d ON d.id=i.dataset_id
                WHERE ''' + where + boundary + ' ORDER BY i.created_at ' + ordering + ',i.id ' + ordering + ' LIMIT ?', [*page_args, limit]).fetchall()
            if direction == 'backward':
                rows = list(reversed(rows))
            images = [{**dict(row), 'complete': bool(row['complete']), 'negative': bool(row['negative']), 'editor_lease': None} for row in rows]
            previous = following = None
            if rows:
                first, last = [rows[0]['created_at'], rows[0]['id']], [rows[-1]['created_at'], rows[-1]['id']]
                if con.execute('SELECT 1 FROM hosted_images i WHERE ' + where + ' AND (i.created_at,i.id)<(?,?) LIMIT 1', [*args, *first]).fetchone():
                    previous = _cursor(cursor_scope, first, 'backward')
                if con.execute('SELECT 1 FROM hosted_images i WHERE ' + where + ' AND (i.created_at,i.id)>(?,?) LIMIT 1', [*args, *last]).fetchone():
                    following = _cursor(cursor_scope, last)
            result = {'scope': scope, 'images': images, 'next_cursor': following, 'previous_cursor': previous}
            if include_total:
                result['total'] = con.execute('SELECT COUNT(*) FROM hosted_images i WHERE ' + where, args).fetchone()[0]
            return result
