"""Hosted object review and resumable corrections over native DQE evidence.

The existing annotation-entity transformer handles relabel/delete. Hosted
leases, stable objects and immutable versions replace legacy file writers.
"""
from contextlib import contextmanager, closing
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import threading

from .analysis import _lines
from .analysis_views import AnalysisViews
from .artifacts import ArtifactRef
from .errors import DomainError
from .store import _id, _json
from .operations import _digest
from .types import MutationContext


class DqeReview:
    def __init__(self, analysis):
        self.analysis, self.store, self.artifacts = analysis, analysis.store, analysis.artifacts
        self.views = AnalysisViews(analysis)
        self.root = analysis.root / 'dqe-reviews'
        self.root.mkdir(exist_ok=True)
        self.lock, self.projections = threading.RLock(), {}
        with self.store.transaction() as con:
            con.execute('CREATE TABLE IF NOT EXISTS hosted_dqe_state(job_id TEXT PRIMARY KEY REFERENCES hosted_analysis_jobs(id),review_revision INTEGER NOT NULL,population_revision INTEGER NOT NULL)')
            con.execute('CREATE TABLE IF NOT EXISTS hosted_dqe_reviews(job_id TEXT NOT NULL,point_id TEXT NOT NULL,revision INTEGER NOT NULL,payload_json TEXT NOT NULL,PRIMARY KEY(job_id,point_id))')
            con.execute('CREATE TABLE IF NOT EXISTS hosted_dqe_images(job_id TEXT NOT NULL,image_id TEXT NOT NULL,version_id TEXT NOT NULL,PRIMARY KEY(job_id,image_id))')
            con.execute('CREATE TABLE IF NOT EXISTS hosted_dqe_batches(id TEXT PRIMARY KEY,job_id TEXT NOT NULL,actor_id TEXT NOT NULL,state TEXT NOT NULL,items_json TEXT NOT NULL,results_json TEXT NOT NULL,created_at REAL NOT NULL)')
    def _state(self, con, job_id):
        job, manifest, _ = self.analysis._job(con, job_id)
        if job['consumer'] != 'dqe' or job['state'] != 'complete':
            raise DomainError('dqe_result_unavailable', 409)
        con.execute('INSERT OR IGNORE INTO hosted_dqe_state VALUES(?,0,?)', (job_id, manifest['revision']))
        return con.execute('SELECT * FROM hosted_dqe_state WHERE job_id=?', (job_id,)).fetchone(), manifest
    def state(self, actor, job_id):
        with self.store.transaction() as con:
            self.store._actor(con, actor, admin=True)
            state, manifest = self._state(con, job_id)
            current = self.store._dataset(con, manifest['dataset_id'])
            return {**dict(state), 'dataset_id': manifest['dataset_id'], 'source_changed': current['revision'] != state['population_revision']}
    def _current(self, con, job_id, expected_review_revision=None):
        if expected_review_revision is not None and (type(expected_review_revision) is not int or expected_review_revision < 0):
            raise DomainError('dqe_review_revision_invalid', 422)
        state, manifest = self._state(con, job_id)
        current = self.store._dataset(con, manifest['dataset_id'])
        if current['revision'] != state['population_revision']:
            raise DomainError('dqe_population_changed', 409, 'Another edit changed this dataset. Run a fresh analysis before applying more corrections.')
        if expected_review_revision is not None and expected_review_revision != state['review_revision']:
            raise DomainError('dqe_review_changed', 409)
        return state, manifest
    def _points(self, actor, job_id, actions):
        if not isinstance(actions, list) or not actions or len(actions) > 10000:
            raise DomainError('dqe_actions_invalid', 422)
        mapped = []
        for action in actions:
            if not isinstance(action, dict) or action.get('action') not in {'relabel', 'delete', 'bbox'}:
                raise DomainError('dqe_action_invalid', 422)
            point = self.point(actor, job_id, action.get('point_id', ''))
            mapped.append((action, point['hosted_identity']))
        if len({identity['object_id'] for _, identity in mapped}) != len(mapped):
            raise DomainError('dqe_duplicate_object_action', 422)
        return mapped
    def point(self, actor, job_id, point_id):
        result = self.views.point(actor, job_id, point_id)
        identity = result['hosted_identity']
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            row = con.execute('SELECT version_id FROM hosted_dqe_images WHERE job_id=? AND image_id=?', (job_id, identity['image_id'])).fetchone()
            if row is not None:
                version = con.execute('SELECT annotations_json FROM hosted_annotation_versions WHERE id=?', (row[0],)).fetchone()
                obj = next((item for item in json.loads(version[0])['objects'] if item['id'] == identity['object_id']), None)
                result['hosted_identity'] = {**identity, 'frozen_version_id': identity['version_id'], 'version_id': row[0],
                                             'deleted': obj is None, **({'class_id': obj['class_id'], 'bbox': obj['bbox']} if obj else {})}
        return result
    def apply(self, actor, job_id, context, actions, *, negative=False, expected_review_revision=None, batch_id=None):
        if type(negative) is not bool:
            raise DomainError('dqe_negative_invalid', 422)
        payload = {'job_id': job_id, 'context': asdict(context), 'actions': actions, 'negative': negative,
                   'expected_review_revision': expected_review_revision, 'batch_id': batch_id}
        previous = self.analysis.operations._existing(actor, context.operation_id, 'dqe_apply_objects', payload)
        if previous is not None:
            return previous
        # Point lookup is immutable and runs outside the writer transaction.
        mapped = self._points(actor, job_id, actions)
        def run(con):
            state, manifest = self._current(con, job_id, expected_review_revision)
            if batch_id:
                batch = con.execute('SELECT * FROM hosted_dqe_batches WHERE id=? AND job_id=?', (batch_id, job_id)).fetchone()
                if batch is None or batch['actor_id'] != actor.user_id or batch['state'] != 'running':
                    raise DomainError('dqe_batch_not_running', 409)
            image = self.store._guard(con, actor, context)
            if image['state'] != 'confirmed' or image['dataset_id'] != manifest['dataset_id']:
                raise DomainError('dqe_image_not_confirmed', 409)
            frozen = next((row for row in manifest['images'] if row['id'] == context.image_id), None)
            owned = con.execute('SELECT version_id FROM hosted_dqe_images WHERE job_id=? AND image_id=?', (job_id, context.image_id)).fetchone()
            if frozen is None or (owned[0] if owned else frozen['version_id']) != context.expected_version:
                raise DomainError('dqe_image_version_changed', 409)
            if any(identity['image_id'] != context.image_id or identity['version_id'] != context.expected_version for _, identity in mapped):
                raise DomainError('dqe_object_not_in_image', 409)
            current_version = con.execute('SELECT annotations_json FROM hosted_annotation_versions WHERE id=?', (context.expected_version,)).fetchone()
            body = json.loads(current_version[0])
            if any(identity.get('deleted') or identity['object_id'] not in {obj['id'] for obj in body['objects']} for _, identity in mapped):
                raise DomainError('dqe_object_already_removed', 409)
            classes = sorted(manifest['classes'], key=lambda row: row['position'])
            class_indices = {row['id']: index for index, row in enumerate(classes)}
            lines, _ = _lines({**frozen, 'annotations': body}, class_indices)
            revision = int(hashlib.sha256(context.expected_version.encode()).hexdigest()[:12], 16) + 1
            from services.annotation_entities import _build_record, apply_annotation_entity_actions, canonical_label_line
            record = _build_record('hosted:' + manifest['dataset_id'], [
                {'annotation_entity_id': obj['id'], 'entity_revision': revision, 'source_row_ordinal': index, 'label_line': canonical_label_line(lines[index]), 'deleted': False}
                for index, obj in enumerate(body['objects'])])
            native_actions, boxes = [], {}
            for action, identity in mapped:
                if action['action'] == 'bbox':
                    boxes[identity['object_id']] = action.get('bbox')
                    continue
                request = {'annotation_entity_id': identity['object_id'], 'expected_entity_revision': revision, 'action': action['action']}
                if action['action'] == 'relabel':
                    if action.get('target_class_id') not in class_indices:
                        raise DomainError('dqe_target_class_unavailable', 422)
                    request['target_class_id'] = class_indices[action['target_class_id']]
                native_actions.append(request)
            _, transformed = apply_annotation_entity_actions(record, native_actions, expected_record_revision=record['record_revision'])
            results = {row['annotation_entity_id']: row for row in transformed}
            updated = []
            for obj in body['objects']:
                change = results.get(obj['id'])
                if change and change['deleted']:
                    continue
                if change:
                    obj['class_id'] = classes[int(change['label_line'].split()[0])]['id']
                if obj['id'] in boxes:
                    obj['bbox'] = boxes[obj['id']]
                    # A bounding box edit cannot pretend an old polygon follows
                    # the new geometry. Require the canvas for polygon edits.
                    if obj.get('polygon'):
                        raise DomainError('dqe_polygon_edit_requires_canvas', 409)
                updated.append(obj)
            body['objects'] = updated
            if not updated and not negative:
                raise DomainError('dqe_empty_image_requires_negative_confirmation', 409)
            saved = self.store._save(con, actor, context, body, complete=True, negative=negative)
            con.execute('INSERT OR REPLACE INTO hosted_dqe_images VALUES(?,?,?)', (job_id, context.image_id, saved['current_version']))
            next_review = state['review_revision'] + 1
            for action, identity in mapped:
                disposition = {'action': action['action'], 'object_id': identity['object_id'], 'image_id': context.image_id,
                               'previous_version': context.expected_version, 'version_id': saved['current_version'], 'actor_id': actor.user_id,
                               'operation_id': context.operation_id, 'reviewed_at': self.store.clock(), 'disposition': {'delete': 'deleted', 'relabel': 'relabeled', 'bbox': 'bbox_adjusted'}[action['action']]}
                con.execute('INSERT OR REPLACE INTO hosted_dqe_reviews VALUES(?,?,?,?)', (job_id, action['point_id'], next_review, _json(disposition)))
            con.execute('UPDATE hosted_dqe_state SET review_revision=?,population_revision=population_revision+1 WHERE job_id=?', (next_review, job_id))
            return {'image': saved, 'review_revision': next_review, 'operation_id': context.operation_id, 'action_count': len(mapped)}
        return self.store._command(actor, context.operation_id, 'dqe_apply_objects', payload, run, admin=True)
    def disposition(self, actor, job_id, point_id, *, disposition, note='', expected_revision, operation_id):
        if disposition not in {'keep', 'uncertain', 'reviewed'} or not isinstance(note, str) or len(note) > 10000:
            raise DomainError('dqe_disposition_invalid', 422)
        payload = {'job_id': job_id, 'point_id': point_id, 'disposition': disposition, 'note': note, 'expected_revision': expected_revision}
        previous = self.analysis.operations._existing(actor, operation_id, 'dqe_disposition', payload)
        if previous is not None:
            return previous
        identity = self.point(actor, job_id, point_id)['hosted_identity']
        def run(con):
            state, _ = self._current(con, job_id, expected_revision)
            current = self.store._image(con, identity['image_id'])
            if current['current_version'] != identity['version_id']:
                raise DomainError('dqe_image_version_changed', 409)
            if identity.get('deleted'):
                raise DomainError('dqe_object_already_removed', 409)
            revision = state['review_revision'] + 1
            value = {**identity, 'disposition': disposition, 'note': note, 'actor_id': actor.user_id, 'reviewed_at': self.store.clock(), 'operation_id': operation_id}
            con.execute('INSERT OR REPLACE INTO hosted_dqe_reviews VALUES(?,?,?,?)', (job_id, point_id, revision, _json(value)))
            con.execute('UPDATE hosted_dqe_state SET review_revision=? WHERE job_id=?', (revision, job_id))
            return {**value, 'review_revision': revision}
        return self.store._command(actor, operation_id, 'dqe_disposition', payload, run, admin=True)
    def history(self, actor, job_id, *, limit=100, offset=0):
        if type(limit) is not int or not 1 <= limit <= 500 or type(offset) is not int or offset < 0:
            raise DomainError('dqe_page_invalid', 422)
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            self.analysis._job(con, job_id)
            rows = con.execute('SELECT point_id,revision,payload_json FROM hosted_dqe_reviews WHERE job_id=? ORDER BY revision DESC,point_id LIMIT ? OFFSET ?', (job_id, limit, offset)).fetchall()
            return {'rows': [{'point_id': row[0], 'revision': row[1], **json.loads(row[2])} for row in rows]}
    @contextmanager
    def session(self, actor, job_id, relative):
        """Project authoritative review receipts into a disposable native index."""
        state = self.state(actor, job_id)
        with self.lock:
            path = self.root / (job_id + '.sqlite3')
            if not path.is_file():
                with self.views._file(actor, job_id, relative) as source:
                    with closing(sqlite3.connect(f'file:{source}?mode=ro', uri=True)) as src, closing(sqlite3.connect(path)) as dst:
                        src.backup(dst)
            if self.projections.get(job_id) != state['review_revision']:
                from services.class_analysis_session_store import replace_class_analysis_review_state
                with self.store.read_connection() as con:
                    self.store._actor(con, actor, admin=True)
                    rows = con.execute('SELECT point_id,revision,payload_json FROM hosted_dqe_reviews WHERE job_id=?', (job_id,)).fetchall()
                entries = []
                for point, revision, value in rows:
                    payload = json.loads(value)
                    entries.append({'point_id': point, 'revision': revision, 'disposition': payload['disposition'], 'reviewed_at': payload['reviewed_at'], 'payload': payload})
                replace_class_analysis_review_state(path, entries)
                self.projections[job_id] = state['review_revision']
            yield path
    def create_batch(self, actor, job_id, items, operation_id):
        if not isinstance(items, list) or not 1 <= len(items) <= 1000 or len(_json(items).encode()) > 8 * 1024 * 1024:
            raise DomainError('dqe_batch_invalid', 422)
        payload = {'job_id': job_id, 'items': items}
        previous = self.analysis.operations._existing(actor, operation_id, 'dqe_create_batch', payload)
        if previous is not None:
            return previous
        seen = set()
        for item in items:
            try:
                context = MutationContext(**item['context'])
                if type(item.get('negative', False)) is not bool:
                    raise ValueError('negative must be a boolean')
            except (KeyError, TypeError, ValueError) as error:
                raise DomainError('dqe_batch_invalid', 422) from error
            if context.image_id in seen:
                raise DomainError('dqe_batch_duplicate_image', 422)
            seen.add(context.image_id)
            self._points(actor, job_id, item['actions'])
        def run(con):
            self._current(con, job_id)
            for item in items:
                self.store._guard(con, actor, MutationContext(**item['context']))
            identifier = _id('dqe_batch')
            con.execute('INSERT INTO hosted_dqe_batches VALUES(?,?,?,\'queued\',?,\'{}\',?)', (identifier, job_id, actor.user_id, _json(items), self.store.clock()))
            return {'batch_id': identifier, 'state': 'queued', 'image_count': len(items)}
        return self.store._command(actor, operation_id, 'dqe_create_batch', {'job_id': job_id, 'items': items}, run, admin=True)
    def batch(self, actor, job_id, batch_id):
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            row = con.execute('SELECT * FROM hosted_dqe_batches WHERE id=? AND job_id=?', (batch_id, job_id)).fetchone()
            if row is None:
                raise DomainError('dqe_batch_not_found', 404)
            return {'batch_id': batch_id, 'state': row['state'], 'actor_id': row['actor_id'], 'items': json.loads(row['items_json']), 'results': json.loads(row['results_json'])}
    def batches(self, actor, job_id, *, limit=50, offset=0):
        if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or offset < 0:
            raise DomainError('dqe_page_invalid', 422)
        with self.store.read_connection() as con:
            self.store._actor(con, actor, admin=True)
            self.analysis._job(con, job_id)
            rows = con.execute('SELECT id,state,items_json,results_json,created_at FROM hosted_dqe_batches WHERE job_id=? AND actor_id=? ORDER BY created_at DESC,id LIMIT ? OFFSET ?', (job_id, actor.user_id, limit, offset)).fetchall()
            return {'batches': [{'batch_id': row[0], 'state': row[1], 'image_count': len(json.loads(row[2])),
                                 'completed_count': sum(item['state'] == 'complete' for item in json.loads(row[3]).values()), 'created_at': row[4]} for row in rows]}
    def resume_batch(self, actor, job_id, batch_id, contexts, operation_id):
        if not isinstance(contexts, list) or len(contexts) > 1000:
            raise DomainError('dqe_batch_contexts_invalid', 422)
        try:
            supplied = [MutationContext(**item) for item in contexts]
        except (TypeError, ValueError) as error:
            raise DomainError('dqe_batch_contexts_invalid', 422) from error
        replacements = {item.image_id: item for item in supplied}
        if len(replacements) != len(supplied):
            raise DomainError('dqe_batch_duplicate_image', 422)
        def run(con):
            _, manifest = self._current(con, job_id)
            row = con.execute('SELECT * FROM hosted_dqe_batches WHERE id=? AND job_id=? AND actor_id=?', (batch_id, job_id, actor.user_id)).fetchone()
            if row is None:
                raise DomainError('dqe_batch_not_found', 404)
            items, results = json.loads(row['items_json']), json.loads(row['results_json'])
            if set(replacements) - {item['context']['image_id'] for item in items}:
                raise DomainError('dqe_batch_contexts_invalid', 422)
            for index, item in enumerate(items):
                key = str(index)
                if results.get(key, {}).get('state') == 'complete':
                    continue
                original = MutationContext(**{**item['context'], 'operation_id': batch_id + ':' + key})
                payload = {'job_id': job_id, 'context': asdict(original), 'actions': item['actions'],
                           'negative': item.get('negative', False), 'expected_review_revision': None, 'batch_id': batch_id}
                receipt = con.execute('SELECT request_hash,response_json FROM hosted_receipts WHERE actor_id=? AND operation_id=?', (actor.user_id, original.operation_id)).fetchone()
                if receipt:
                    if receipt[0] != _digest({'operation': 'dqe_apply_objects', 'payload': payload}):
                        raise DomainError('operation_id_reused', 409)
                    results[key] = {'state': 'complete', 'result': json.loads(receipt[1])}
                    continue
                fresh = replacements.get(original.image_id)
                if fresh is None or fresh.expected_version != original.expected_version:
                    raise DomainError('dqe_batch_fresh_context_required', 409, {'image_id': original.image_id})
                self.store._guard(con, actor, fresh)
                item['context'] = asdict(fresh)
                results.pop(key, None)
            state = 'complete' if len(results) == len(items) else 'queued'
            con.execute('UPDATE hosted_dqe_batches SET items_json=?,results_json=?,state=? WHERE id=?', (_json(items), _json(results), state, batch_id))
            self.store._event(con, actor, manifest['dataset_id'], None, 'dqe_batch_resumed', {'batch_id': batch_id, 'operation_id': operation_id})
            return {'batch_id': batch_id, 'state': state, 'completed_count': len(results)}
        return self.store._command(actor, operation_id, 'dqe_resume_batch', {'job_id': job_id, 'batch_id': batch_id, 'contexts': contexts}, run, admin=True)
    def process_batch(self, actor, job_id, batch_id, *, limit=5):
        if type(limit) is not int or not 1 <= limit <= 25:
            raise DomainError('dqe_batch_limit_invalid', 422)
        with self.store.transaction() as con:
            self.store._actor(con, actor, admin=True)
            row = con.execute('SELECT actor_id,state FROM hosted_dqe_batches WHERE id=? AND job_id=?', (batch_id, job_id)).fetchone()
            if row is None or row['actor_id'] != actor.user_id:
                raise DomainError('dqe_batch_not_found', 404)
            if row['state'] in {'cancelled', 'complete'}:
                return self.batch(actor, job_id, batch_id)
            con.execute("UPDATE hosted_dqe_batches SET state='running' WHERE id=?", (batch_id,))
        current = self.batch(actor, job_id, batch_id)
        for index, item in enumerate(current['items']):
            if str(index) in current['results']:
                continue
            if limit <= 0:
                break
            limit -= 1
            context = MutationContext(**{**item['context'], 'operation_id': batch_id + ':' + str(index)})
            try:
                result = {'state': 'complete', 'result': self.apply(actor, job_id, context, item['actions'], negative=item.get('negative', False), batch_id=batch_id)}
            except DomainError as error:
                result = {'state': 'conflict', 'error': error.code, 'detail': error.detail}
            with self.store.transaction() as con:
                self.store._actor(con, actor, admin=True)
                row = con.execute('SELECT results_json,state FROM hosted_dqe_batches WHERE id=?', (batch_id,)).fetchone()
                results = json.loads(row[0]); results[str(index)] = result
                state = row[1]
                if state != 'cancelled':
                    state = ('needs_attention' if any(item['state'] != 'complete' for item in results.values())
                             else 'complete' if len(results) == len(current['items']) else 'running')
                con.execute('UPDATE hosted_dqe_batches SET results_json=?,state=? WHERE id=?', (_json(results), state, batch_id))
            if result['state'] != 'complete':
                break
        return self.batch(actor, job_id, batch_id)
    def cancel_batch(self, actor, job_id, batch_id, operation_id):
        def run(con):
            row = con.execute('SELECT actor_id,state FROM hosted_dqe_batches WHERE id=? AND job_id=?', (batch_id, job_id)).fetchone()
            if row is None or row[0] != actor.user_id:
                raise DomainError('dqe_batch_not_found', 404)
            if row[1] != 'complete':
                con.execute("UPDATE hosted_dqe_batches SET state='cancelled' WHERE id=?", (batch_id,))
            return {'batch_id': batch_id, 'state': 'complete' if row[1] == 'complete' else 'cancelled'}
        return self.store._command(actor, operation_id, 'dqe_cancel_batch', {'job_id': job_id, 'batch_id': batch_id}, run, admin=True)
