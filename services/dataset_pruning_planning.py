"""Cached-feature planning, asynchronous previews and evidence-bound plan commits.

Completed feature jobs are immutable. A preview captures one ledger generation,
does expensive math without the service/ledger lock, then brackets publication
with the source and generation checks. Applying the captured plan never approves
a removal. The content-addressed plan file and SQLite receipt are the authority.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import uuid

import numpy as np
from fastapi import HTTPException
from utils.capture_metadata import CAPTURE_GROUP_FIELDS, portable_capture_metadata

from services.dataset_pruning import (PruningConfig, build_order, coverage_for_retained,
    digest, reevaluate_pair_evidence, removal_evidence, selection, _assignment_image_input, _config, UnionFind)
from services.dataset_pruning_review import (FilteredEvidence, ReviewLedger, compute_plan,
    family_inventory, SupplementaryEvidence, directed_digest)
from services.dataset_pruning_serving import ServingContext, states, verify, thaw


SCHEMA = 'pruning-active-plan-v1'


class ReviewGenerationChanged(HTTPException):
    """Optimistic work lost its ledger generation, not its source binding."""
    def __init__(self, detail):
        super().__init__(409, detail)


def coverage_directory(job_directory):
    from services.dataset_pruning_coverage import SCHEMA as coverage_schema
    return Path(job_directory) / 'planning' / 'coverage' / coverage_schema


def coverage_cache_key(result, source, config, job_directory):
    from services.dataset_pruning_coverage import SCHEMA as coverage_schema
    return digest({'source': source['inventory_digest'], 'recipe': result['recipe'],
                   'config': config.recipe(), 'coverage_schema': coverage_schema,
                   'directory': str(coverage_directory(job_directory).resolve())})


def export_scene_groups(images, previous, config, *, witness_metrics=()):
    """Extend existing split groups with recorded whole-image relationships.

    Object appearance witnesses are deliberately excluded: finding the same
    kind of object in two different scenes does not make those scenes duplicates.
    Removed frames remain in the union so they can connect surviving neighbors.
    """
    if len(previous) != len(images): raise HTTPException(409, 'Saved split groups are incomplete')
    union = UnionFind(len(images))
    original, pixels, videos = {}, {}, {}
    identifiers = {image['image_id']: i for i, image in enumerate(images)}
    for i, image in enumerate(images):
        for mapping, key in [(original, previous[i]), (pixels, image['image_sha256'])]:
            if key in mapping: union.union(i, mapping[key])
            else: mapping[key] = i
        metadata = portable_capture_metadata(image)
        for field in CAPTURE_GROUP_FIELDS:
            value = metadata.get(field)
            if value is not None and value != '':
                key = (field, str(value))
                if key in videos: union.union(i, videos[key])
                else: videos[key] = i
    for metrics in witness_metrics:
        for candidate, record in metrics.get('scene_witnesses', {}).items():
            witness = record.get('witness_id') if isinstance(record, dict) else record
            if candidate in identifiers and witness in identifiers: union.union(identifiers[candidate], identifiers[witness])
    return [union.find(i) for i in range(len(images))]


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '-' + uuid.uuid4().hex)
    try:
        tmp.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read(path):
    path = Path(path)
    if path.is_symlink(): raise HTTPException(409, 'Planning artifact path changed')
    return json.loads(path.read_text())


def settings_for(result, state):
    active = state.get('plans', {}).get('active', {}).get('value')
    if active: return copy.deepcopy(active['settings'])
    return {'policy': 'strict_duplicates', 'object_weight': result.get('config', {}).get('object_weight', .9),
            'max_removed_images': state.get('budget', 0), 'max_affected_appearance_groups': 0,
            'max_lost_appearance_weight_fraction': 0.}


def _loss(value):
    value = value or {}
    return {'affected_appearance_groups': value.get('affected_appearance_group_count', value.get('affected_appearance_groups', 0)),
            'lost_appearance_weight_fraction': value.get('lost_appearance_weight_fraction', 0.),
            'uncovered_objects': value.get('uncovered_object_count', 0),
            'removed_annotations': value.get('removed_annotation_count', 0),
            'disappearing_classes': value.get('disappearing_classes', []),
            'accounting_digest': value.get('accounting_digest'),
            'measured': bool(value.get('accounting_digest'))}


class ReweightedEvidence:
    def __init__(self, images, evidence, original, current, scan_evidence=None, scan_binding=()):
        self.images, self.evidence, self.original, self.current = images, evidence, original, current
        self.scan_evidence = evidence if scan_evidence is None else scan_evidence
        self.scan_binding = scan_binding
        # These immutable image inputs recur in hundreds of measured pairs.
        # Preserve the exact existing pair JSON/hash while encoding each
        # annotation inventory only once per captured preview.
        self.assignment_inputs = [json.dumps(_assignment_image_input(image), sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode() for image in images]

    def convert(self, edge):
        expected = hashlib.sha256(b'[' + self.assignment_inputs[edge['left']] + b',' +
                                  self.assignment_inputs[edge['right']] + b']').hexdigest()
        return reevaluate_pair_evidence(self.images[edge['left']], self.images[edge['right']], edge,
                                        self.current, assignment_config=self.original, _assignment_identity=expected)

    def __iter__(self):
        verify(self.scan_binding)
        for edge in self.scan_evidence: yield self.convert(edge)
        verify(self.scan_binding)

    def coverage_rows(self):
        # These measured fields are invariant under the index-weight control.
        # Keep all pairs, including failed gates, exactly as the existing
        # coverage_for_retained lower-bound calculation does.
        verify(self.scan_binding)
        yield from self.scan_evidence
        verify(self.scan_binding)

    def edges_for(self, index):
        # The new weight can promote a pair that failed the original index
        # gate. Read all measured assignments before applying the new gate.
        iterator = self.evidence.all_edges_for(index) if hasattr(self.evidence, 'all_edges_for') else self.evidence
        for edge in iterator:
            if index in (edge['left'], edge['right']):
                converted = self.convert(edge)
                if converted['valid']: yield converted

    def valid_links(self):
        for edge in self:
            if edge['valid']: yield edge['left'], edge['right']

    def pair(self, a, b):
        edge = self.evidence.pair(a, b)
        return self.convert(edge) if edge is not None else None

    def comparisons_for(self, members):
        for edge in self.evidence.comparisons_for(members): yield self.convert(edge)


def replay_objects(service, context, cancelled, progress):
    """Bounded object-count batches replay only saved raw rows and saved fits."""
    from services.dataset_pruning_comparison import reconstruct_cached_object_rows
    directory, result, _, images, _, _, _, _ = context
    manifest_path = directory / 'components.json'
    manifest_binding = states([manifest_path])
    manifest = _read(manifest_path) if manifest_path.is_file() else {}
    if not manifest or digest(manifest) != digest(result['recipe'].get('component_manifest', {})):
        raise HTTPException(409, 'Saved component fits are unavailable; analyze with the selected cached recipe first')
    binding = digest({'source': context[2]['inventory_digest'], 'recipe': result['recipe'], 'manifest': manifest})
    target = directory / 'planning' / ('features-' + binding)
    target.mkdir(parents=True, exist_ok=True)
    path, meta_path = target / 'objects.npy', target / 'commit.json'
    total = sum(len(image['objects']) for image in images)
    width = sum(int(entry['dimension']) for key, entry in manifest['components'].items()
                if key.startswith('bbox_') and entry.get('weight', 0) > 0)
    if meta_path.is_file():
        meta = _read(meta_path)
        observed = states([path])
        if meta.get('binding') != binding or meta.get('file_states') != [[p, list(s) if s else None] for p, s in observed]:
            raise HTTPException(409, 'Saved planning feature artifact changed; preserve it for inspection')
        matrix = np.load(path, mmap_mode='r', allow_pickle=False)
        if matrix.shape != (total, width) or matrix.dtype != np.float32:
            raise HTTPException(409, 'Saved planning feature dimensions changed')
        return matrix
    all_indices = [obj['feature_index'] for image in images for obj in image['objects']]
    if sorted(all_indices) != list(range(len(all_indices))):
        raise HTTPException(409, 'Saved object row identities are incomplete')
    if total and width <= 0: raise HTTPException(409, 'Budgeted coverage requires saved object descriptors')
    tmp = target / ('objects-' + uuid.uuid4().hex + '.npy')
    output = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float32, shape=(total, width))
    count = 0
    try:
        for image in images:
            for start in range(0, len(image['objects']), 128):
                if cancelled(): raise InterruptedError('Preview cancelled')
                selected = {**image, 'objects': image['objects'][start:start + 128]}
                rows = reconstruct_cached_object_rows(service.adapter.api, service.cache, [selected], result['recipe'], manifest)
                for obj in selected['objects']:
                    vector = rows[obj['feature_index']]
                    if vector.shape != (width,) or not np.all(np.isfinite(vector)) or abs(float(np.linalg.norm(vector)) - 1.) > 1e-4:
                        raise HTTPException(409, 'Cached planning fingerprint is not a finite unit vector')
                    output[obj['feature_index']] = vector
                count += len(selected['objects'])
                progress(.04 + .20 * count / max(1, total), 'Replaying cached object fingerprints with saved size correction')
        output.flush()
        verify(context.binding)
        verify(manifest_binding)
        os.replace(tmp, path)
        _write(meta_path, {'binding': binding, 'rows': total, 'dimension': width,
                          'encoder_calls': 0, 'training_calls': 0, 'refits': 0,
                          'file_states': [[p, list(s) if s else None] for p, s in states([path])]})
    finally:
        del output
        tmp.unlink(missing_ok=True)
    return np.load(path, mmap_mode='r', allow_pickle=False)


class PlanningMixin:
    def init_planning(self):
        self.preview_lock = threading.RLock()
        self.preview_events = {}
        self.coverage_indexes = {}
        self.coverage_locks = {}
        self.planning_artifacts = {}
        self.strict_summaries = {}
        self.strict_summary_locks = {}

    def _capture_plan(self, job_id, payload=None, check=True, refresh=True):
        payload = payload or {}
        with self.adapter.guard(), self.lock:
            context = self.review_context(job_id, {**payload, 'removed_count': 0}, check_selection=check)
            context = ServingContext(tuple(context), context.binding + states([context[0] / 'components.json']))
            state = context[4].read()
            capture = {'selection_revision': context[1]['selection_revision'], 'review_revision': state['review_revision'],
                       'plan_revision': state['plan_revision'], 'source_digest': context[2]['inventory_digest'],
                       'recipe_digest': digest(context[1]['recipe'])}
        if refresh:
            state = self._refresh_reduced_plan(context, state, capture)
            capture.update(review_revision=state['review_revision'], plan_revision=state['plan_revision'])
        if check:
            if payload.get('expected_review_revision') != state['review_revision'] or payload.get('expected_plan_revision', payload.get('plan_revision')) != state['plan_revision']:
                raise HTTPException(409, 'The review plan changed. Refresh before previewing these settings')
        return context, state, capture

    def _current_review_plan(self, job_id):
        """Read a coherent current snapshot while allowing concurrent reviews.

        Unlike a mutation or preview, an unversioned GET does not promise an
        earlier ledger generation. Retry only generation races; source or
        artifact failures remain errors. Any derived measurements still run
        outside commit ownership in _capture_plan.
        """
        while True:
            try:
                context, state, capture = self._capture_plan(job_id, check=False)
                with self.adapter.guard(), self.lock:
                    self._check_capture(context, capture)
                    plan = self.plan_for(context, state, state.get('budget', 0))
                    return context, state, plan
            except ReviewGenerationChanged:
                continue

    def _refresh_reduced_plan(self, context, state, capture):
        """Measure a durably reduced proposal without expanding its candidates.

        Cross-job Keep first commits a cheap, monotone reduction. Expensive
        cached measurements happen here, outside the ledger transaction and
        service lock; their installation is conditional on that same revision.
        """
        active = self._load_active_plan(context, state)
        if not active or not active.get('coverage_refresh_required') or active.get('accounting_outdated'):
            return state
        plan = self._compute_settings_plan(context, state, active['settings'], candidate_ids=active['removed_ids'])
        plan['no_refill'] = True
        record = self._store_plan(context, plan)
        draft = copy.deepcopy(state)
        self.supersede_approvals(draft, plan)
        approved = [i for i, image in enumerate(context[3])
                    if plan['proposals'].get(image['image_id'], {}).get('status') == 'approved'
                    and draft['decisions'].get(image['image_id'], {}).get('status') == 'approved']
        retained = sorted(set(range(len(context[3]))) - set(approved))
        if plan['settings']['policy'] == 'budgeted_coverage':
            metrics = self._coverage_index(context).evaluate(retained)
            self._check_budget_metrics(plan['settings'], metrics)
        else:
            config, edges = self._strict_evidence(context, draft, plan['settings'])
            metrics = coverage_for_retained(context[3], retained, edges.coverage_rows(), config)
        prior_artifact = state['plans']['active']['value']['artifact']
        with self.adapter.guard(), self.lock:
            current = context[4].read()
            if (current.get('plans', {}).get('active', {}).get('refresh_of') == prior_artifact
                    and current['review_revision'] == capture['review_revision'] + 1
                    and current['plan_revision'] == capture['plan_revision'] + 1):
                # Two readers can refresh the same immutable reduction. The
                # first committed measurement is authoritative for both.
                verify(context.binding)
                self.adapter.check_revision(context[2]); self.adapter.check_inventory(context[2], context[3])
                return current
            self._check_capture(context, capture)
            self._check_plan_integrity(context, plan)
            with context[4].transaction() as committed:
                if (committed['review_revision'], committed['plan_revision']) != (capture['review_revision'], capture['plan_revision']):
                    raise ReviewGenerationChanged('Review changed while the reduced proposal was being measured; refresh again')
                committed['review_revision'] += 1
                committed['plan_revision'] += 1
                committed['plans']['active'] = {'value': record, 'owner': 'inherited-keep-refresh',
                    'version': committed['review_revision'], 'refresh_of': prior_artifact}
                self.supersede_approvals(committed, plan)
                committed['approved_metrics'] = metrics
                committed['approved_metrics_digest'] = digest(sorted(approved))
                return copy.deepcopy(committed)

    def _check_capture(self, context, capture):
        verify(context.binding)
        key = coverage_cache_key(context[1], context[2], context[-1], context[0]) if context[-1] else None
        index = self.coverage_indexes.get(key)
        if index is not None:
            try:
                getattr(index, 'validate', index._validate)()
                verify(getattr(index, '_planning_feature_binding', ()))
            except ValueError as error: raise HTTPException(409, str(error)) from error
        self.adapter.check_revision(context[2]); self.adapter.check_inventory(context[2], context[3])
        state = context[4].read()
        if state['review_revision'] != capture['review_revision'] or state['plan_revision'] != capture['plan_revision']:
            raise ReviewGenerationChanged('Review changed while the preview was being computed')
        return state

    def _preview_directory(self, job_id, preview_id):
        if not re.fullmatch(r'[a-f0-9]{32}', str(preview_id)):
            raise HTTPException(404, 'Planning preview not found')
        directory = self.directory(job_id) / 'planning' / 'previews' / preview_id
        if directory.is_symlink(): raise HTTPException(409, 'Planning preview path changed')
        return directory

    def settings_preview(self, job_id, payload):
        from api.dataset_pruning import SettingsPreviewRequest
        payload = SettingsPreviewRequest.model_validate(payload).model_dump()
        context, state, capture = self._capture_plan(job_id, payload)
        settings = payload['settings']
        if settings['max_removed_images'] > len(context[3]): raise HTTPException(400, 'The image ceiling exceeds this dataset')
        if settings['policy'] == 'budgeted_coverage' and (context[-1] is None or context[-1].evidence_mode != 'compound'):
            raise HTTPException(409, 'Budgeted appearance coverage requires whole-image and object features')
        if settings['policy'] == 'budgeted_coverage' and context[-1].scale_protection != 'native_pixels_and_occupancy':
            raise HTTPException(409, 'Budgeted coverage requires modern native-pixel and occupancy protections; run that analysis using cached features first')
        identifier = uuid.uuid4().hex
        directory = self._preview_directory(job_id, identifier)
        initial = {'preview_id': identifier, 'job_id': job_id, 'status': 'queued', 'progress': 0.,
                   'message': 'Preparing cached planning evidence', 'settings': settings,
                   'capture': capture, 'client_sequence': payload.get('client_sequence', 0), 'created_at': time.time()}
        _write(directory / 'state.json', initial)
        event = threading.Event()
        with self.preview_lock: self.preview_events[(job_id, identifier)] = event
        threading.Thread(target=self._run_settings_preview, args=(context, state, initial, event), daemon=True,
                         name='pruning-plan-' + identifier[:8]).start()
        return initial

    def _run_settings_preview(self, context, state, initial, event):
        job_id, identifier = initial['job_id'], initial['preview_id']
        directory = self._preview_directory(job_id, identifier)
        current = dict(initial)
        def progress(value, message='Computing cached coverage'):
            if event.is_set(): raise InterruptedError('Preview cancelled')
            current.update(status='running', progress=float(value), message=str(message), updated_at=time.time())
            _write(directory / 'state.json', current)
        try:
            progress(.01, 'Checking the frozen source and planning settings')
            plan = self._compute_settings_plan(context, state, initial['settings'], event.is_set, progress)
            if initial['settings']['policy'] == 'budgeted_coverage':
                approved = {key for key, value in plan['proposals'].items() if value['status'] == 'approved'}
                approved_metrics = self._coverage_index(context).evaluate([image['image_id'] for image in context[3] if image['image_id'] not in approved], cancelled=event.is_set)
                plan['approved_loss'] = _loss(approved_metrics)
            plan['superseded_count'] = sum(value.get('status') == 'approved' and
                (key not in plan['proposals'] or plan['proposals'][key]['evidence_digest'] != value.get('evidence_digest'))
                for key, value in state['decisions'].items())
            if event.is_set(): raise InterruptedError('Preview cancelled')
            with self.adapter.guard(), self.lock:
                self._check_capture(context, initial['capture'])
                if event.is_set(): raise InterruptedError('Preview cancelled')
                plan['capture'] = initial['capture']
                plan['preview_id'] = identifier
                # JSON turns integer class-map keys into strings. Hash the
                # actual wire representation so class10/11 sorts identically
                # before publication and after a process/disk round trip.
                plan = json.loads(json.dumps(plan, allow_nan=False))
                plan['preview_digest'] = digest(plan)
                _write(directory / 'plan.json', plan)
                current.update(status='ready', progress=1., message='Preview ready; no removals have been approved',
                               preview_digest=plan['preview_digest'], plan=plan)
                _write(directory / 'state.json', current)
        except InterruptedError:
            current.update(status='cancelled', message='Preview cancelled; the applied plan is unchanged')
            current.pop('plan', None); _write(directory / 'state.json', current)
        except Exception as error:
            current.update(status='failed', message=str(getattr(error, 'detail', error)),
                           error_status=getattr(error, 'status_code', 409 if isinstance(error, ValueError) else 500))
            current.pop('plan', None); _write(directory / 'state.json', current)
        finally:
            with self.preview_lock: self.preview_events.pop((job_id, identifier), None)

    def settings_preview_status(self, job_id, preview_id):
        directory = self._preview_directory(job_id, preview_id)
        if not (directory / 'state.json').is_file(): raise HTTPException(404, 'Planning preview not found')
        value = _read(directory / 'state.json')
        if value.get('preview_id') != preview_id or value.get('job_id') != job_id:
            raise HTTPException(409, 'Planning preview identity changed')
        if value['status'] in {'queued', 'running'} and (job_id, preview_id) not in self.preview_events:
            value.update(status='interrupted', message='Preview interrupted by restart; request a new cached preview')
        if value['status'] == 'ready':
            try: plan = _read(directory / 'plan.json')
            except (OSError, ValueError) as error: raise HTTPException(409, 'The immutable planning preview is unavailable') from error
            measured_digest = digest({key: item for key, item in plan.items() if key != 'preview_digest'})
            if (plan.get('preview_digest') != measured_digest or value.get('preview_digest') != measured_digest
                    or digest(value.get('plan')) != digest(plan) or value.get('settings') != plan.get('settings')
                    or value.get('capture') != plan.get('capture') or plan.get('preview_id') != preview_id):
                raise HTTPException(409, 'The saved planning preview changed; request a fresh preview')
            if plan['settings']['policy'] == 'budgeted_coverage' and plan.get('accounting_schema') != coverage_directory(self.directory(job_id)).name:
                raise HTTPException(409, 'Appearance accounting changed; request a new cached preview')
            if plan['settings']['policy'] == 'budgeted_coverage':
                from services.dataset_pruning_coverage import QUALIFICATION
                if plan.get('qualification') != QUALIFICATION:
                    raise HTTPException(409, 'Direct coverage qualification changed; request a new cached preview')
            value['plan'] = plan
            context, _, _ = self._capture_plan(job_id, check=False)
            with self.adapter.guard():
                self._check_capture(context, value['capture'])
                self._check_plan_integrity(context, value['plan'])
        return value

    def settings_preview_cancel(self, job_id, preview_id):
        directory = self._preview_directory(job_id, preview_id)
        if not (directory / 'state.json').is_file(): raise HTTPException(404, 'Planning preview not found')
        with self.preview_lock:
            event = self.preview_events.get((job_id, preview_id))
            if event: event.set()
        value = _read(directory / 'state.json')
        if value['status'] in {'queued', 'running'}: value['cancel_requested'] = True
        return value

    @staticmethod
    def public_settings_preview(value):
        """HTTP summary; Apply and paged views retain the complete bound plan.

        This additive projection marker identifies the settings-preview
        response contract. Native pair proof and family inventories are served
        through their existing paged endpoints, not repeated in every poll.
        """
        if value.get('status') != 'ready': return value
        omitted = {'cards', 'proposals', 'families', 'removed_ids', 'retained_ids',
                   'chosen', 'order_result', 'accounting_files', 'metrics'}
        return {**value, 'plan': {key: item for key, item in value['plan'].items() if key not in omitted},
                'plan_projection': 'settings-preview-summary-v1'}

    def _coverage_index(self, context, cancelled=lambda: False, progress=lambda *_: None):
        from services.dataset_pruning_coverage import prepare_coverage
        key = coverage_cache_key(context[1], context[2], context[-1], context[0])
        with self.preview_lock:
            lock = self.coverage_locks.setdefault(key, threading.Lock())
        while not lock.acquire(timeout=.1):
            if cancelled(): raise InterruptedError('Preview cancelled')
        try:
            verify(context.binding)
            if key not in self.coverage_indexes:
                objects = replay_objects(self, context, cancelled, progress)
                neighbor_adapter = copy.copy(self.adapter)
                neighbor_adapter.cancel_event = type('PreviewCancellation', (), {'is_set': staticmethod(cancelled)})()
                neighbors = getattr(neighbor_adapter, 'group_neighbors', None) or getattr(neighbor_adapter, 'neighbors', None)
                index = prepare_coverage(coverage_directory(context[0]), context[3], context[5], objects, context[-1],
                                         source_digest=context[2]['inventory_digest'], recipe_digest=digest(context[1]['recipe']),
                                         cancelled=cancelled, progress=progress, neighbors_fn=neighbors)
                index._planning_feature_binding = states([objects.filename]) if getattr(objects, 'filename', None) else ()
                self.coverage_indexes[key] = index
            index = self.coverage_indexes[key]
            try:
                getattr(index, 'validate', index._validate)()
                verify(getattr(index, '_planning_feature_binding', ()))
            except ValueError as error: raise HTTPException(409, str(error)) from error
            return index
        finally: lock.release()

    def _strict_summary(self, context, cancelled=lambda: False, progress=lambda *_: None):
        """Disk-backed compact scans; complete saved proof is never replaced."""
        from services.dataset_pruning_evidence import COMPACT_PAIR_SCHEMA, PairStore, compact_pair
        # Only the saved measured pairs and labelled image inputs determine
        # this projection. Review revisions, the removal ceiling and weight
        # controls cannot alter it. The complete source bracket is still
        # checked on both admission and return/publication.
        inputs = states([context[6].path, context[0] / 'evidence.sqlite3-wal', context[0] / 'images.json'])
        key = digest({'schema': COMPACT_PAIR_SCHEMA, 'inputs': inputs})
        with self.preview_lock:
            lock = self.strict_summary_locks.setdefault(key, threading.Lock())
        while not lock.acquire(timeout=.1):
            if cancelled(): raise InterruptedError('Preview cancelled')
        try:
            verify(context.binding)
            cached = self.strict_summaries.get(key)
            if cached:
                verify(cached[1])
                return cached[0]
            directory = context[0] / 'planning' / 'strict-evidence'
            directory.mkdir(parents=True, exist_ok=True)
            target, receipt_path = directory / (key + '.sqlite3'), directory / (key + '.json')
            if target.is_symlink(): raise HTTPException(409, 'Compact pair artifact path changed')
            if receipt_path.is_file():
                if not target.is_file(): raise HTTPException(409, 'Compact pair measurements are unavailable')
                receipt = _read(receipt_path)
                binding = states([target, receipt_path])
                if receipt.get('binding') != key or receipt.get('schema') != COMPACT_PAIR_SCHEMA:
                    raise HTTPException(409, 'Compact pair measurements belong to another source')
                value = hashlib.sha256()
                with target.open('rb') as source:
                    for block in iter(lambda: source.read(1024 * 1024), b''): value.update(block)
                if value.hexdigest() != receipt.get('sha256'):
                    raise HTTPException(409, 'Compact pair measurements changed; preserve this analysis for inspection')
                verify(binding)
            else:
                descriptor, name = tempfile.mkstemp(prefix='.' + key + '-', suffix='.sqlite3', dir=directory)
                os.close(descriptor)
                temporary = Path(name)
                output = PairStore(temporary)
                try:
                    count = 0
                    for count, edge in enumerate(context[6], 1):
                        if count % 256 == 1:
                            if cancelled(): raise InterruptedError('Preview cancelled')
                        output.append(compact_pair(edge))
                        if count % 8192 == 0: progress(.05, 'Preparing compact cached pair measurements: ' + str(count))
                    output.finish()
                    if cancelled(): raise InterruptedError('Preview cancelled')
                    verify(context.binding)
                    value = hashlib.sha256()
                    with temporary.open('rb') as source:
                        for block in iter(lambda: source.read(1024 * 1024), b''): value.update(block)
                    os.replace(temporary, target)
                    _write(receipt_path, {'schema': COMPACT_PAIR_SCHEMA, 'binding': key, 'pair_count': count,
                                          'sha256': value.hexdigest()})
                    binding = states([target, receipt_path])
                finally:
                    output.abort()
                    temporary.unlink(missing_ok=True)
            verify(context.binding)
            store = PairStore(target)
            store._planning_binding = binding
            self.strict_summaries[key] = (store, binding)
            while len(self.strict_summaries) > 4: self.strict_summaries.pop(next(iter(self.strict_summaries)))
            return store
        finally: lock.release()

    def _strict_evidence(self, context, state, settings, cancelled=lambda: False, progress=lambda *_: None):
        original = context[-1] or _config(None)
        config = replace(original, object_weight=settings['object_weight'])
        combined = SupplementaryEvidence(context[6], state.get('comparisons', {}))
        summary = self._strict_summary(context, cancelled, progress)
        scan = SupplementaryEvidence(summary, state.get('comparisons', {}))
        base = ReweightedEvidence(context[3], combined, original, config, scan_evidence=scan,
                                  scan_binding=summary._planning_binding)
        return config, FilteredEvidence(base, state['rejected'])

    def _compute_settings_plan(self, context, state, settings, cancelled=lambda: False, progress=lambda *_: None,
                               candidate_ids=None):
        # NumPy's limits are process-wide. Reuse the existing serialized
        # pruning policy, with cancellable admission, outside ledger locks.
        from services.dataset_pruning_jobs import pruning_blas_policy
        with pruning_blas_policy(cancelled=cancelled):
            return self._compute_settings_plan_with_blas(context, state, settings, cancelled, progress, candidate_ids)

    def _compute_settings_plan_with_blas(self, context, state, settings, cancelled, progress, candidate_ids):
        images = context[3]
        pins = {key for key, field in state['pins'].items() if field['value']}
        for field in state['constraints'].values(): pins.update(field['value']['keeper_ids'])
        if settings['policy'] == 'budgeted_coverage':
            index = self._coverage_index(context, cancelled, progress)
            raw = index.plan(settings, pinned_ids=sorted(pins), candidate_ids=candidate_ids,
                             cancelled=cancelled, progress=progress)
            retained = raw.get('retained_ids', raw.get('metrics', {}).get('retained_ids'))
            if retained is None: raise ValueError('Coverage planner omitted retained identities')
            metrics = raw.get('metrics') or index.evaluate(retained, cancelled=cancelled)
            impacts = raw.get('impacts')
            if impacts is None: impacts = index.impacts(retained, candidate_ids=raw.get('removed_ids'), cancelled=cancelled)
            plan = self._assemble_plan(context, state, settings, raw, metrics, impacts)
            index_directory = coverage_directory(context[0])
            from services.dataset_pruning_coverage import FROZEN_ARRAYS
            manifest_path = index_directory / 'manifest.json'
            manifest_binding = states([manifest_path])
            manifest = _read(manifest_path)
            artifacts = manifest.get('artifacts')
            expected = {name + '.npy' for name in FROZEN_ARRAYS}
            if (not isinstance(artifacts, dict) or set(artifacts) != expected or
                    any(not isinstance(name, str) or Path(name).name != name for name in artifacts) or
                    not manifest.get('complete') or manifest.get('accounting_digest') != index.accounting_digest):
                raise HTTPException(409, 'Completed appearance accounting has invalid frozen-artifact bindings')
            files = [manifest_path, *(index_directory / name for name in sorted(artifacts))]
            plan['accounting_files'] = [[Path(path).name, list(value) if value else None] for path, value in states(files)]
            verify(manifest_binding)
            index.validate()
            plan['accounting_directory'] = index_directory.name
            plan['accounting_schema'] = index_directory.name
        else:
            config, evidence = self._strict_evidence(context, state, settings, cancelled, progress)
            order = build_order(images, context[5], evidence, sorted(pins), cancelled=cancelled, config=config)
            # The final assembler builds the family/proposal records. Reusing
            # compute_plan here built and discarded those records, repeatedly
            # reading every large saved assignment a second time.
            chosen = selection(images, order, min(settings['max_removed_images'], order['max_removed_count']))
            raw = {'chosen': chosen}
            selected = set(chosen['removed'])
            if candidate_ids is not None:
                allowed = set(candidate_ids)
                selected = {i for i, image in enumerate(images) if image['image_id'] in allowed and image['image_id'] not in pins and not image.get('protection_errors')}
                while True:
                    retained = set(range(len(images))) - selected
                    invalid = []
                    for i in sorted(selected):
                        try: removal_evidence(i, retained, evidence)
                        except ValueError: invalid.append(i)
                    if not invalid: break
                    selected.difference_update(invalid)
            retained = set(range(len(images))) - selected
            metrics = coverage_for_retained(images, retained, evidence.coverage_rows(), config)
            impacts = []
            for i in sorted(selected):
                proof = removal_evidence(i, retained, evidence)
                witness = proof.pop('retained_index')
                impacts.append({'candidate_id': images[i]['image_id'], 'scene_witness_id': images[witness]['image_id'],
                                'evidence': proof, 'restored_appearance_groups': None, 'restored_appearance_weight': None})
            raw.update(removed_ids=[images[i]['image_id'] for i in sorted(selected)],
                       retained_ids=[images[i]['image_id'] for i in sorted(retained)], curve=order['curve'])
            plan = self._assemble_plan(context, state, settings, raw, metrics, impacts)
            plan['order_result'] = order
        return plan

    def _assemble_plan(self, context, state, settings, raw, metrics, impacts):
        images, source, result = context[3], context[2], context[1]
        ids = {image['image_id']: i for i, image in enumerate(images)}
        retained_ids = list(raw.get('retained_ids', metrics.get('retained_ids', [])))
        removed_ids = list(raw.get('removed_ids', metrics.get('removed_ids', [])))
        retained_set, removed_set = set(retained_ids), set(removed_ids)
        if retained_set & removed_set or retained_set | removed_set != set(ids):
            raise ValueError('Planner returned an incomplete image partition')
        accounting = raw.get('accounting_digest', metrics.get('accounting_digest'))
        from services.dataset_pruning_coverage import QUALIFICATION
        qualification = QUALIFICATION if settings['policy'] == 'budgeted_coverage' else None
        policy_digest = digest({'settings': settings, 'source': source['inventory_digest'], 'recipe': result['recipe'],
                                'accounting': accounting, **({'qualification': qualification} if qualification else {})})
        proposals, cards = {}, []
        impact_by_id = {row.get('candidate_id', row.get('image_id')): row for row in impacts}
        impact_rank = {row.get('candidate_id', row.get('image_id')): rank + 1 for rank, row in enumerate(impacts)}
        for rank, identifier in enumerate(removed_ids):
            impact = copy.deepcopy(impact_by_id.get(identifier, {'candidate_id': identifier}))
            witness = impact.get('scene_witness_id') or impact.get('witness_id')
            if not witness:
                record = metrics.get('scene_witnesses', {}).get(identifier, {})
                witness = record.get('witness_id') if isinstance(record, dict) else record
            if witness not in retained_set: raise ValueError('Every removal requires a final-retained scene witness')
            evidence = impact.get('evidence') or impact
            if settings['policy'] == 'strict_duplicates' and settings['object_weight'] == (context[-1] or _config(None)).object_weight:
                evidence_digest = directed_digest(identifier, witness, evidence, source['inventory_digest'], digest(result['recipe']))
            else:
                evidence_digest = digest({'policy': policy_digest, 'candidate': identifier, 'witness': witness, 'evidence': evidence})
            decision = state['decisions'].get(identifier)
            status = 'approved' if decision and decision.get('status') == 'approved' and decision.get('evidence_digest') == evidence_digest and decision.get('witness_id') == witness else 'unreviewed'
            proposal = {'candidate_id': identifier, 'witness_id': witness, 'evidence_digest': evidence_digest,
                        'evidence': evidence, 'status': status, 'note': decision.get('note', '') if decision else '',
                        'scene_witness': self.public_image(images[ids[witness]])}
            proposals[identifier] = proposal
            cards.append({**impact, **proposal, 'candidate': self.public_image(images[ids[identifier]]),
                          'scene_witness': self.public_image(images[ids[witness]]), 'restoration_rank': impact_rank.get(identifier, rank + 1),
                          'restored_appearance_groups': impact.get('restored_appearance_group_count'),
                          'restored_appearance_weight': impact.get('restored_appearance_weight_fraction'),
                          'restored_objects': impact.get('restored_object_count'),
                          'partially_restored_groups': impact.get('partially_restored_group_count'),
                          'uncovered_objects': impact.get('uncovered_objects', []), 'object_witnesses': impact.get('object_witnesses', [])})
        population = family_inventory(images, context[6])
        covered = {i for members in population.values() for i in members}
        for identifier in removed_ids:
            i = ids[identifier]
            if i not in covered:
                members = sorted({i, ids[proposals[identifier]['witness_id']]})
                population[digest([images[j]['image_id'] for j in members])[:24]] = members
                covered.add(i)
        families = {}
        for family_id, members in population.items():
            keepers = [images[i]['image_id'] for i in members if images[i]['image_id'] in retained_set]
            constraint = state['constraints'].get(family_id, {}).get('value', {})
            families[family_id] = {'id': family_id, 'member_indices': members, 'keeper_ids': keepers,
                'requested_count': constraint.get('requested_count', 1),
                'confirmed': bool(constraint and set(constraint.get('keeper_ids', [])) <= set(keepers)),
                'proposals': [proposals[images[i]['image_id']] for i in members if images[i]['image_id'] in proposals],
                'hard_keep_ids': [images[i]['image_id'] for i in members if images[i].get('protection_errors')],
                'pins': [key for key in keepers if state['pins'].get(key, {}).get('value')],
                'uncovered_ids': [], 'exception_reasons': [],
                'requirements': ['Keep at least the selected representatives; global loss ceilings may retain more',
                                 'Object witnesses may come from several retained images']}
        family_of, confirmed_candidates = {}, set()
        for family_id, family in families.items():
            for proposal in family['proposals']:
                family_of.setdefault(proposal['candidate_id'], family_id)
                if family['confirmed']: confirmed_candidates.add(proposal['candidate_id'])
        for card in cards:
            card['family_id'] = family_of.get(card['candidate_id'])
            card['requires_confirmation'] = settings['policy'] == 'strict_duplicates' and card['candidate_id'] not in confirmed_candidates
            details = card.get('objects', card.get('object_witnesses', []))
            if details and not card['uncovered_objects']: card['uncovered_objects'] = [obj for obj in details if not obj.get('covered', False)]
            for obj in card['uncovered_objects']:
                obj.setdefault('row', obj.get('label_row'))
                obj['crop_url'] = f'/dataset_pruning/jobs/{context[0].name}/images/{card["candidate_id"]}/objects/{obj["row"]}'
        cards.sort(key=lambda row: row['restoration_rank'])
        explicit_keepers = {key for key, field in state['pins'].items() if field['value']}
        for field in state['constraints'].values(): explicit_keepers.update(field['value']['keeper_ids'])
        protected = {image['image_id'] for image in images if image.get('protection_errors')} | explicit_keepers
        plan = {'schema': SCHEMA, 'settings': settings, 'active_settings': settings, 'policy_digest': policy_digest,
                'qualification': qualification, 'proof_generation': raw.get('proof_generation', metrics.get('proof_generation')),
                'accounting_digest': accounting, 'recipe_digest': digest(result['recipe']), 'source_digest': source['inventory_digest'],
                'removed_ids': removed_ids, 'retained_ids': retained_ids, 'removed_count': len(removed_ids),
                'retained_count': len(retained_ids), 'removed_percent': 100 * len(removed_ids) / max(1, len(images)),
                'proposals': proposals, 'cards': cards, 'families': families,
                'metrics': metrics, 'proposed_loss': _loss(metrics), 'approved_loss': _loss({}),
                'appearance_group_count': raw.get('appearance_group_count', metrics.get('appearance_group_count')),
                'total_appearance_weight': raw.get('total_appearance_weight', metrics.get('total_appearance_weight', metrics.get('appearance_weight_total'))),
                'curve': [{**point, 'affected_appearance_groups': point.get('affected_appearance_group_count', point.get('affected_appearance_groups', 0))} for point in raw.get('curve', [])],
                'budget': settings['max_removed_images'], 'approved_count': sum(row['status'] == 'approved' for row in proposals.values()),
                'approved_indices': [ids[key] for key, row in proposals.items() if row['status'] == 'approved'],
                'chosen': {'removed': [ids[key] for key in removed_ids], 'retained': [ids[key] for key in retained_ids], **metrics},
                'order_result': {'max_removed_count': len(removed_ids), 'protected_count': len(protected),
                                 'curve': raw.get('curve', []), 'keep_ids': sorted(explicit_keepers),
                                 'protection_reasons': {str(i): ['Incomplete or conflicting annotation evidence'] for i, image in enumerate(images) if image.get('protection_errors')}},
                'review_revision': state['review_revision'], 'plan_revision': state['plan_revision'], 'encoder_calls': 0, 'training_calls': 0, 'refits': 0}
        plan['plan_digest'] = digest({'policy': policy_digest, 'removed': removed_ids,
                                      'evidence': [(key, value['evidence_digest']) for key, value in sorted(proposals.items())]})
        return plan

    def _store_plan(self, context, plan):
        content = json.dumps(plan, sort_keys=True, allow_nan=False).encode()
        identifier = hashlib.sha256(content).hexdigest()
        path = context[0] / 'planning' / 'artifacts' / (identifier + '.json')
        if not path.exists(): _write(path, plan)
        return {'artifact': identifier, 'settings': plan['settings'], 'plan_digest': plan['plan_digest'],
                'accounting_digest': plan.get('accounting_digest'), 'candidate_ids': plan['removed_ids']}

    @staticmethod
    def _check_plan_integrity(context, plan):
        files = plan.get('accounting_files', [])
        if files:
            directory = context[0] / 'planning' / 'coverage'
            if plan.get('accounting_directory'):
                name = plan['accounting_directory']
                if not re.fullmatch(r'pruning-direct-appearance-coverage-v[1-9][0-9]*', name):
                    raise HTTPException(409, 'Invalid accounting generation identity')
                directory = directory / name
            if any(Path(name).name != name for name, _ in files): raise HTTPException(409, 'Invalid accounting artifact identity')
            verify(tuple((str(directory / name), tuple(value) if value else None) for name, value in files))

    def _load_active_plan(self, context, state):
        record = state.get('plans', {}).get('active', {}).get('value')
        if not record: return None
        identifier = record['artifact']
        if not re.fullmatch('[a-f0-9]{64}', identifier): raise HTTPException(409, 'Saved active-plan identity is invalid')
        path = context[0] / 'planning' / 'artifacts' / (identifier + '.json')
        binding = states([path])
        cached = self.planning_artifacts.get(identifier)
        if not cached or cached[0] != binding:
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != identifier: raise HTTPException(409, 'Saved active plan changed')
            plan = json.loads(content)
            self.planning_artifacts[identifier] = (binding, plan)
            while len(self.planning_artifacts) > 4: self.planning_artifacts.pop(next(iter(self.planning_artifacts)))
        else: plan = cached[1]
        plan = copy.deepcopy(plan)
        self._check_plan_integrity(context, plan)
        if plan['settings']['policy'] == 'budgeted_coverage':
            from services.dataset_pruning_coverage import QUALIFICATION
            plan['accounting_outdated'] = (plan.get('accounting_schema') != coverage_directory(context[0]).name or
                                          plan.get('qualification') != QUALIFICATION)
        if plan['source_digest'] != context[2]['inventory_digest'] or plan['recipe_digest'] != digest(context[1]['recipe']):
            raise HTTPException(409, 'Saved active plan belongs to another source or representation')
        ids = {image['image_id']: i for i, image in enumerate(context[3])}
        # Committed Keep intent may arrive from another open job. Restoring
        # these candidates is monotone and never refills the captured set.
        required = {key for key, field in state.get('pins', {}).items() if field['value']}
        for field in state.get('constraints', {}).values(): required.update(field['value']['keeper_ids'])
        newly_kept = required & set(plan['removed_ids'])
        if newly_kept:
            plan['removed_ids'] = [key for key in plan['removed_ids'] if key not in newly_kept]
            plan['retained_ids'] = sorted(set(plan['retained_ids']) | newly_kept)
            plan['proposals'] = {key: value for key, value in plan['proposals'].items() if key not in newly_kept}
            plan['cards'] = [row for row in plan['cards'] if row['candidate_id'] not in newly_kept]
            plan.update(removed_count=len(plan['removed_ids']), retained_count=len(plan['retained_ids']), no_refill=True,
                        removed_percent=100 * len(plan['removed_ids']) / max(1, len(ids)),
                        coverage_refresh_required=True, proposed_loss={'measured': False}, curve=[], metrics={})
            plan['chosen'] = {'removed': [ids[key] for key in plan['removed_ids']], 'retained': [ids[key] for key in plan['retained_ids']]}
            plan['order_result'].update(curve=[], max_removed_count=len(plan['removed_ids']))
            for row in plan['cards']:
                for field in ['restored_appearance_groups', 'restored_appearance_weight', 'restored_objects',
                              'partially_restored_groups', 'restoration_rank']:
                    row[field] = None
            plan['plan_digest'] = digest({'previous': plan['plan_digest'], 'restored_manual_keepers': sorted(newly_kept)})
            for family in plan['families'].values():
                family['proposals'] = [row for row in family['proposals'] if row['candidate_id'] not in newly_kept]
                family['keeper_ids'] = sorted(set(family['keeper_ids']) | {key for key in newly_kept if ids[key] in family['member_indices']})
        approved = []
        for identifier, proposal in plan['proposals'].items():
            decision = state['decisions'].get(identifier, {})
            proposal['status'] = 'approved' if not plan.get('accounting_outdated') and decision.get('status') == 'approved' and decision.get('evidence_digest') == proposal['evidence_digest'] and decision.get('witness_id') == proposal['witness_id'] else 'unreviewed'
            if proposal['status'] == 'approved': approved.append(ids[identifier])
        for row in plan['cards']: row['status'] = plan['proposals'][row['candidate_id']]['status']
        for family in plan['families'].values():
            family['proposals'] = [plan['proposals'][row['candidate_id']] for row in family['proposals']]
            constraint = state['constraints'].get(family['id'], {}).get('value', {})
            family['confirmed'] = bool(constraint and set(constraint.get('keeper_ids', [])) <= set(family['keeper_ids']))
        for row in plan['cards']:
            row['requires_confirmation'] = plan['settings']['policy'] == 'strict_duplicates' and not any(
                family['confirmed'] and any(value['candidate_id'] == row['candidate_id'] for value in family['proposals'])
                for family in plan['families'].values())
        plan.update(approved_indices=sorted(approved), approved_count=len(approved),
                    review_revision=state['review_revision'], plan_revision=state['plan_revision'])
        approved_metrics = state.get('approved_metrics', {})
        plan['approved_loss'] = _loss(approved_metrics) if state.get('approved_metrics_digest') == digest(sorted(approved)) else {**_loss({}), 'measured': False}
        if plan.get('accounting_outdated'):
            plan['proposed_loss'] = {**plan['proposed_loss'], 'measured': False}
            plan['approved_loss'] = {**plan['approved_loss'], 'measured': False}
        return plan

    def planning_selection(self, job_id, payload):
        context, state, _ = self._capture_plan(job_id, check=False)
        if payload.get('selection_revision') != context[1]['selection_revision']:
            raise HTTPException(409, 'The feature analysis changed')
        if payload.get('preview_id'):
            value = self.settings_preview_status(job_id, payload['preview_id'])
            if value['status'] != 'ready': raise HTTPException(409, 'The preview is not ready')
            plan = value['plan']
        else:
            if payload.get('expected_review_revision') is not None and payload['expected_review_revision'] != state['review_revision']:
                raise HTTPException(409, 'Review changed')
            if payload.get('plan_revision') is not None and payload['plan_revision'] != state['plan_revision']:
                raise HTTPException(409, 'The applied plan changed')
            plan = self.plan_for(context, state, state.get('budget', payload.get('removed_count', 0)))
            if 'cards' not in plan: plan = self._legacy_public_plan(context, state, plan)
        rows = plan.get('cards', [])
        if payload.get('filter') in {'approved', 'unreviewed'}: rows = [row for row in rows if row['status'] == payload['filter']]
        query = str(payload.get('query') or '').casefold()
        if query: rows = [row for row in rows if query in row['candidate']['image_relpath'].casefold() or query in row['candidate_id']]
        offset, limit = payload.get('offset', 0), payload.get('limit', 12)
        return {key: copy.deepcopy(value) for key, value in plan.items() if key not in {'cards', 'families', 'proposals', 'order_result', 'chosen'}} | {
            'rows': rows[offset:offset + limit], 'offset': offset,
            'next_offset': offset + limit if offset + limit < len(rows) else None,
            'selection_revision': context[1]['selection_revision']}

    def planning_preview_members(self, job_id, family_id, payload):
        preview = self.settings_preview_status(job_id, payload['preview_id'])
        if preview['status'] != 'ready': raise HTTPException(409, 'The preview is not ready')
        context, _, _ = self._capture_plan(job_id, check=False)
        plan = preview['plan']
        family = plan['families'].get(family_id)
        if family is None: raise HTTPException(404, 'Family not present in this preview')
        offset, limit = payload.get('offset', 0), payload.get('limit', 120)
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 120:
            raise HTTPException(400, 'Invalid member continuation bounds')
        members = family['member_indices']
        return {'family_id': family_id, 'preview_id': payload['preview_id'],
                'members': [self.public_image(context[3][i]) for i in members[offset:offset + limit]],
                'offset': offset, 'next_offset': offset + limit if offset + limit < len(members) else None,
                'total_member_count': len(members), 'review_revision': plan['review_revision'], 'plan_revision': plan['plan_revision']}

    def _legacy_public_plan(self, context, state, plan):
        plan = copy.deepcopy(plan)
        ids = {image['image_id']: image for image in context[3]}
        cards = []
        for identifier, proposal in plan['proposals'].items():
            family_id = next((key for key, family in plan['families'].items() if any(row['candidate_id'] == identifier for row in family['proposals'])), None)
            family = plan['families'].get(family_id, {})
            candidate, witness = self.public_image(ids[identifier]), self.public_image(ids[proposal['witness_id']])
            cards.append({**proposal, 'candidate': candidate, 'scene_witness': witness, 'removed': candidate, 'retained': witness,
                          'family_id': family_id, 'requires_confirmation': not family.get('confirmed'),
                          'restoration_rank': len(cards) + 1, 'restored_appearance_groups': None,
                          'restored_appearance_weight': None, 'uncovered_objects': [], 'object_witnesses': []})
        plan.update(cards=cards, settings=settings_for(context[1], state), active_settings=settings_for(context[1], state),
                    proposed_loss=_loss({}), approved_loss=_loss({}), curve=plan['order_result']['curve'])
        return plan

    def planning_families(self, job_id, payload):
        context, state, _ = self._capture_plan(job_id, check=False)
        if payload.get('selection_revision') != context[1]['selection_revision']:
            raise HTTPException(409, 'The feature analysis changed')
        if payload.get('preview_id'):
            preview = self.settings_preview_status(job_id, payload['preview_id'])
            if preview['status'] != 'ready': raise HTTPException(409, 'The preview is not ready')
            plan = preview['plan']
        else:
            if payload.get('plan_revision') is not None and payload['plan_revision'] != state['plan_revision']:
                raise HTTPException(409, 'The applied plan changed')
            if payload.get('expected_review_revision') is not None and payload['expected_review_revision'] != state['review_revision']:
                raise HTTPException(409, 'Review changed')
            plan = self._load_active_plan(context, state)
        query = str(payload.get('query') or '').casefold()
        filter_name = payload.get('filter', 'all')
        values = []
        for original in plan['families'].values():
            if query and not any(query in context[3][i]['image_relpath'].casefold() for i in original['member_indices']) and query not in original['id']: continue
            if filter_name in {'approved', 'unreviewed'} and not any(row['status'] == filter_name for row in original['proposals']): continue
            if filter_name == 'exceptions' and not original['exception_reasons']: continue
            values.append(original)
        if payload.get('sort') == 'size': values.sort(key=lambda row: (-len(row['member_indices']), row['id']))
        else: values.sort(key=lambda row: row['id'])
        offset, limit = payload.get('offset', 0), payload.get('limit', 12)
        rows = []
        for original in values[offset:offset + limit]:
            row = copy.deepcopy(original)
            members = row.pop('member_indices')
            row['members'] = [self.public_image(context[3][i]) for i in members[:120]]
            points = [{'image_id': context[3][i]['image_id'], 'x': float(np.cos(j * 2 * np.pi / max(1, len(members)))),
                       'y': float(np.sin(j * 2 * np.pi / max(1, len(members))))} for j, i in enumerate(members[:120])]
            row.update(graph={'points': points, 'edges': [{'candidate_id': p['candidate_id'], 'witness_id': p['witness_id'],
                                'combined': p['evidence'].get('combined'), 'scene': p['evidence'].get('scene'), 'object': p['evidence'].get('object')}
                               for p in row['proposals']], 'layout_note': 'Navigation positions only. Budgeted object witnesses may come from other families.'},
                       coverage_curve=[], total_member_count=len(members), graph_truncated=len(members) > 120,
                       members_truncated=len(members) > 120, member_next_offset=120 if len(members) > 120 else None)
            rows.append(row)
        return {**self.review_public(state, plan, context[1]), 'families': rows, 'family_count': len(plan['families']),
                'filtered_family_count': len(values), 'filtered_family_ids': [row['id'] for row in values],
                'offset': offset, 'next_offset': offset + limit if offset + limit < len(values) else None,
                'generation': digest({'source': context[2]['inventory_digest'], 'recipe': plan['recipe_digest'],
                                     'plan': plan['plan_digest'], 'review_revision': state['review_revision']})}

    def planning_review_mutation(self, job_id, payload):
        token = str(payload.get('request_id') or '')
        if not re.fullmatch('[a-f0-9]{32}', token): raise HTTPException(400, 'A stable review request identity is required')
        identity = digest(payload)
        ledger = self.review_ledger(self.directory(job_id) / 'review.sqlite3')
        # A retry may overlap both the expensive calculation and the initial
        # capture of the original request. A durable receipt always wins over
        # a subsequently stale source/plan; a different payload never does.
        with self.adapter.guard(), self.lock:
            answer = self._planning_receipt_response(job_id, ledger.read(), token, identity)
            if answer is not None: return answer
        try:
            return self._planning_review_mutation_once(job_id, payload, token, identity)
        except HTTPException as error:
            with self.adapter.guard(), self.lock:
                answer = self._planning_receipt_response(job_id, ledger.read(), token, identity)
                if answer is not None: return answer
            raise

    def _planning_receipt_response(self, job_id, state, token, identity):
        saved = state['operations'].get(token)
        if not saved: return None
        if saved['identity'] != identity: raise HTTPException(409, 'This request identity belongs to another operation')
        try:
            context, current, _ = self._capture_plan(job_id, check=False, refresh=False)
            snapshot = self.review_public(current, self.plan_for(context, current, current.get('budget', 0)), context[1])
        except HTTPException as error:
            return {'outcome': 'committed', 'receipt': saved, 'snapshot': None, 'source_stale': True,
                    'message': 'The operation committed; refresh the source before continuing', 'current_source_error': error.detail}
        return {'outcome': 'committed', 'receipt': saved, 'snapshot': snapshot}

    def _planning_review_mutation_once(self, job_id, payload, token, identity):
        context, original, capture = self._capture_plan(job_id, check=False)
        if payload.get('selection_revision') != capture['selection_revision'] or payload.get('expected_review_revision') != capture['review_revision']:
            raise HTTPException(409, 'Review generation changed. Refresh before applying this operation')
        if payload.get('plan_revision') is not None and payload['plan_revision'] != capture['plan_revision']:
            raise HTTPException(409, 'The applied plan changed')
        if payload.get('expected_plan_revision') is not None and payload['expected_plan_revision'] != capture['plan_revision']:
            raise HTTPException(409, 'The applied plan changed')
        if payload.get('removed_count', original.get('budget', 0)) != original.get('budget', 0):
            raise HTTPException(409, 'The applied image ceiling changed')
        state = copy.deepcopy(original)
        state.setdefault('plans', {})
        revision = state['review_revision'] + 1
        touched = []
        def assign(collection, key, value):
            state.setdefault(collection, {})
            before = copy.deepcopy(state[collection].get(key))
            state[collection][key] = {**value, 'owner': token, 'version': revision}
            touched.append({'collection': collection, 'key': key, 'before': before})
        active = self._load_active_plan(context, state)
        action = payload.get('action')
        if active and active.get('accounting_outdated') and action not in {'apply_plan', 'pin', 'bulk_undo'}:
            raise HTTPException(409, 'Appearance accounting changed; request and Apply a new cached preview before reviewing removals')
        if active is None and action == 'pin':
            legacy = self.plan_for(context, state, state.get('budget', 0))
            captured = [context[3][i]['image_id'] for i in legacy['chosen']['removed']]
            active = self._compute_settings_plan(context, state, settings_for(context[1], state), candidate_ids=captured)
        plan_changed, plan = False, active
        if action == 'apply_plan':
            preview = self.settings_preview_status(job_id, payload.get('preview_id'))
            if preview['status'] != 'ready': raise HTTPException(409, 'The planning preview is not ready')
            if payload.get('preview_digest') is not None and payload['preview_digest'] != preview['preview_digest']:
                raise HTTPException(409, 'The supplied preview digest does not match the captured plan')
            if preview['capture'] != capture: raise HTTPException(409, 'The preview belongs to another review generation')
            plan = copy.deepcopy(preview['plan'])
            previous_policy = settings_for(context[1], state)['policy']
            if previous_policy != plan['settings']['policy']:
                for decision in state['decisions'].values():
                    if decision.get('status') == 'approved': decision.update(status='superseded', superseded_at_revision=revision)
            assign('plans', 'active', {'value': self._store_plan(context, plan)})
            state['budget'] = plan['settings']['max_removed_images']
            plan_changed = True
        elif active is None:
            raise HTTPException(409, 'Apply a settings preview before editing this plan')
        elif action == 'set_budget':
            raise HTTPException(409, 'Preview and Apply the updated planning settings before changing the image ceiling')
        elif action == 'pin':
            identifier, keep = payload.get('image_id'), payload.get('keep')
            image = next((image for image in context[3] if image['image_id'] == identifier), None)
            if image is None or type(keep) is not bool: raise HTTPException(400, 'Keep a specific image with an explicit value')
            intent_at = time.time_ns()
            previous = self.manual_intent_fields(state).get(identifier, {})
            assign('pins', identifier, {'value': keep, 'note': str(payload.get('note') or '')[:2000],
                'image_sha256': image['image_sha256'], 'intent_at': intent_at, 'intent_origin_job': job_id,
                'last_keep_at': intent_at if keep else previous.get('last_keep_at', 0)})
            # The captured allowed set cannot grow after Keep or unpin.
            plan = self._compute_settings_plan(context, state, active['settings'], candidate_ids=active['removed_ids'])
            plan['no_refill'] = True
            assign('plans', 'active', {'value': self._store_plan(context, plan)})
            plan_changed = True
        elif action in {'approve', 'reject', 'undo_decision'}:
            candidate = payload.get('candidate_id')
            proposal = active['proposals'].get(candidate)
            if action != 'undo_decision' and (not proposal or proposal['witness_id'] != payload.get('witness_id') or proposal['evidence_digest'] != payload.get('evidence_digest')):
                raise HTTPException(409, 'The proposed removal or surviving evidence changed')
            if action == 'undo_decision':
                before = state['decisions'].pop(candidate, None)
                touched.append({'collection': 'decisions', 'key': candidate, 'before': before})
                if before and before.get('status') == 'rejected':
                    ids = {image['image_id']: i for i, image in enumerate(context[3])}
                    key = f"{ids[candidate]}:{ids[before['witness_id']]}"
                    rejected = state['rejected'].get(key)
                    if rejected and (rejected.get('owner'), rejected.get('version')) == (before.get('owner'), before.get('version')):
                        state['rejected'].pop(key)
                        touched.append({'collection': 'rejected', 'key': key, 'before': rejected})
                        candidates = active['removed_ids']
                        active_field = state['plans']['active']
                        if ((active_field.get('owner'), active_field.get('version')) == (before.get('owner'), before.get('version'))
                                and before.get('prior_candidate_ids') is not None):
                            # Reject can remove the candidate altogether if
                            # its sole witness is rejected. Undo restores only
                            # that owned captured set, never a later Keep/plan.
                            candidates = before['prior_candidate_ids']
                        plan = self._compute_settings_plan(context, state, active['settings'], candidate_ids=candidates)
                        assign('plans', 'active', {'value': self._store_plan(context, plan)})
                        plan_changed = True
            elif action == 'reject' and active['settings']['policy'] == 'budgeted_coverage':
                raise HTTPException(400, 'Use Keep image for a budgeted proposal; individual object matches have separate witnesses')
            else:
                if action == 'approve' and active['settings']['policy'] == 'strict_duplicates' and not any(family['confirmed'] and any(row['candidate_id'] == candidate for row in family['proposals']) for family in active['families'].values()):
                    raise HTTPException(409, 'Confirm the family keeper plan before approving a strict duplicate removal')
                assign('decisions', candidate, {'status': 'approved' if action == 'approve' else 'rejected',
                    'witness_id': proposal['witness_id'], 'evidence_digest': proposal['evidence_digest'], 'note': str(payload.get('note') or '')[:2000],
                    **({'prior_candidate_ids': active['removed_ids']} if action == 'reject' else {})})
                if action == 'reject':
                    ids = {image['image_id']: i for i, image in enumerate(context[3])}
                    assign('rejected', f"{ids[candidate]}:{ids[proposal['witness_id']]}", {'value': True})
                    plan = self._compute_settings_plan(context, state, active['settings'], candidate_ids=active['removed_ids'])
                    assign('plans', 'active', {'value': self._store_plan(context, plan)})
                    plan_changed = True
        elif action in {'confirm_plan', 'bulk_commit'}:
            preview = state['previews'].get(payload.get('preview_digest'))
            if not preview or preview['review_revision'] != state['review_revision'] or preview['plan_revision'] != state['plan_revision'] or preview['selection_revision'] != capture['selection_revision']:
                raise HTTPException(409, 'This captured family preview is stale')
            plan = self._plan_from_record(context, preview['plan_record'])
            for key, value in preview['constraints'].items(): assign('constraints', key, {'value': value})
            assign('plans', 'active', {'value': preview['plan_record']})
            if action == 'bulk_commit':
                for candidate, proposal in preview['proposals'].items():
                    assign('decisions', candidate, {'status': 'approved', 'witness_id': proposal['witness_id'],
                                                   'evidence_digest': proposal['evidence_digest'], 'note': ''})
            plan_changed = True
        elif action == 'bulk_undo':
            batch = state['batches'].get(payload.get('batch_id'))
            if not batch or batch['undone']: raise HTTPException(409, 'This batch cannot be undone again')
            if any(not state.get(field['collection'], {}).get(field['key']) or
                   state[field['collection']][field['key']].get('owner') != batch['batch_id'] or
                   state[field['collection']][field['key']].get('version') != batch['revision'] for field in batch['fields']):
                raise HTTPException(409, 'A later independent edit owns part of this batch; no undo changes were applied')
            for field in reversed(batch['fields']):
                if field['collection'] == 'pins':
                    image = next(image for image in context[3] if image['image_id'] == field['key'])
                    prior = copy.deepcopy(field['before'] or {'value': False, 'note': ''})
                    intent_at = time.time_ns()
                    last_keep_at = self.manual_intent_fields(state).get(field['key'], {}).get('last_keep_at', 0)
                    prior.update(owner=token, version=revision, image_sha256=image['image_sha256'],
                                 intent_at=intent_at, intent_origin_job=job_id,
                                 last_keep_at=intent_at if prior['value'] else last_keep_at)
                    state['pins'][field['key']] = prior
                elif field['before'] is None: state[field['collection']].pop(field['key'])
                else:
                    value = copy.deepcopy(field['before'])
                    if field['collection'] == 'decisions' and value.get('status') == 'approved': value['status'] = 'superseded'
                    state[field['collection']][field['key']] = value
            batch['undone'] = True
            restored = self._load_active_plan(context, state)
            if restored:
                plan = self._compute_settings_plan(context, state, restored['settings'], candidate_ids=restored['removed_ids'])
                assign('plans', 'active', {'value': self._store_plan(context, plan)})
                state['budget'] = restored['settings']['max_removed_images']
            else:
                plan = None
                state['budget'] = batch.get('previous_budget', 0)
            plan_changed = True
        else: raise HTTPException(400, 'Unknown planning review operation')
        if action in {'apply_plan', 'pin', 'confirm_plan', 'bulk_commit'}:
            state['batches'][token] = {'batch_id': token, 'fields': touched, 'revision': revision,
                                      'at': time.time(), 'action': action, 'undone': False,
                                      'previous_budget': original.get('budget', 0)}
        state['review_revision'] = revision
        state['plan_revision'] += int(plan_changed)
        final = self._load_active_plan(context, state) if state.get('plans', {}).get('active') else self.plan_for(context, state, state.get('budget', 0))
        self.supersede_approvals(state, final)
        # Approved subset is evaluated with precisely the same frozen accounting.
        approved = [i for i, image in enumerate(context[3]) if final['proposals'].get(image['image_id'], {}).get('status') == 'approved' and state['decisions'].get(image['image_id'], {}).get('status') == 'approved']
        retained = sorted(set(range(len(context[3]))) - set(approved))
        if final.get('settings', {}).get('policy') == 'budgeted_coverage':
            metrics = self._coverage_index(context).evaluate(retained)
            self._check_budget_metrics(final['settings'], metrics)
        else:
            settings = final.get('settings', settings_for(context[1], state))
            cfg, edges = self._strict_evidence(context, state, settings)
            metrics = coverage_for_retained(context[3], retained, edges.coverage_rows(), cfg)
        state['approved_metrics'], state['approved_metrics_digest'] = metrics, digest(sorted(approved))
        receipt = {'request_id': token, 'identity': identity, 'revision': revision, 'plan_revision': state['plan_revision'],
                   'action': action, 'batch_id': token if action in {'apply_plan', 'pin', 'bulk_commit', 'confirm_plan'} else None,
                   'conflicts': [], 'at': time.time()}
        state['operations'][token] = receipt
        state.setdefault('history', []).append({'request_id': token, 'action': action, 'revision': revision, 'at': receipt['at'],
            'candidate_id': payload.get('candidate_id'), 'image_id': payload.get('image_id'), 'touched_fields': touched,
            'source_digest': context[2]['inventory_digest'], 'recipe_digest': digest(context[1]['recipe']), 'plan_digest': final['plan_digest']})
        # No expensive coverage work is performed inside these locks/transaction.
        with self.adapter.guard(), self.lock:
            answer = self._planning_receipt_response(job_id, context[4].read(), token, identity)
            if answer is not None: return answer
            self._check_capture(context, capture)
            with context[4].transaction() as committed:
                saved = committed['operations'].get(token)
                if saved:
                    if saved['identity'] != identity: raise HTTPException(409, 'This request identity belongs to another operation')
                    # Do not recapture inside the SQLite transaction: manual
                    # intent reconciliation also owns this ledger. Read its
                    # current snapshot after this transaction releases it.
                    state, receipt = copy.deepcopy(committed), saved
                else:
                    if committed['review_revision'] != capture['review_revision'] or committed['plan_revision'] != capture['plan_revision']:
                        raise HTTPException(409, 'Review changed before this operation could commit')
                    verify(context.binding)
                    self._check_plan_integrity(context, final)
                    self.adapter.check_revision(context[2]); self.adapter.check_inventory(context[2], context[3])
                    committed.clear(); committed.update(state)
        final = self._load_active_plan(context, state) if state.get('plans', {}).get('active') else self.plan_for(context, state, state.get('budget', 0))
        return {'outcome': 'committed', 'receipt': receipt, 'snapshot': self.review_public(state, final, context[1])}

    def _plan_from_record(self, context, record):
        return self._load_active_plan(context, {'plans': {'active': {'value': record}}, 'decisions': {}, 'constraints': {},
                                               'review_revision': 0, 'plan_revision': 0})

    @staticmethod
    def _check_budget_metrics(settings, metrics):
        if not metrics.get('ready', True): raise HTTPException(409, 'Direct final-retained coverage is not verified')
        if metrics.get('affected_appearance_group_count', 0) > settings['max_affected_appearance_groups'] or metrics.get('lost_appearance_weight_fraction', 0) > settings['max_lost_appearance_weight_fraction'] + 1e-12 or metrics['removed_count'] > settings['max_removed_images']:
            raise HTTPException(409, 'The approved variant exceeds the applied appearance or image loss ceilings')

    def planning_family_preview(self, job_id, payload):
        context, state, capture = self._capture_plan(job_id, check=False)
        if payload.get('expected_review_revision') != capture['review_revision']: raise HTTPException(409, 'Review changed')
        active = self._load_active_plan(context, state)
        population = active['families']
        selected = [payload['family_id']] if payload.get('family_id') else payload.get('family_ids', sorted(population))
        if not isinstance(selected, list) or len(set(selected)) != len(selected) or set(selected) - set(population):
            raise HTTPException(400, 'The captured family population is invalid')
        count = payload.get('keep_count', 1)
        if type(count) is not int or count < 1: raise HTTPException(400, 'Keep count must be a positive integer')
        draft, constraints = copy.deepcopy(state), {}
        for family_id in selected:
            family = population[family_id]
            members = family['member_indices']
            explicit = payload.get('keeper_ids') if payload.get('family_id') == family_id else None
            member_ids = {context[3][i]['image_id'] for i in members}
            if explicit is not None and (not isinstance(explicit, list) or not explicit or len(set(explicit)) != len(explicit) or set(explicit) - member_ids):
                raise HTTPException(400, 'Choose distinct keepers from this family')
            pinned = {key for key, field in state['pins'].items() if field['value']}
            if active['settings']['policy'] == 'strict_duplicates':
                from services.dataset_pruning_review import choose_family_keepers
                cfg, evidence = self._strict_evidence(context, state, active['settings'])
                keepers = choose_family_keepers(context[3], members, evidence, context[5], min(count, len(members)), pinned, explicit)
            else:
                # Explicit keep-at-least-X; no automatic rare/class/direct-frame seed.
                from services.data_ingestion import greedy_diverse_indices_with_scores
                required = (pinned & member_ids) | set(explicit or []) | {context[3][i]['image_id'] for i in members if context[3][i].get('protection_errors')}
                initial = [j for j, i in enumerate(members) if context[3][i]['image_id'] in required]
                order, _, _ = greedy_diverse_indices_with_scores(np.asarray(context[5][members]), keep_fraction=1,
                    initial_indices=initial, target_count=max(len(initial), min(count, len(members))))
                keepers = sorted(required | {context[3][members[j]]['image_id'] for j in order[:max(len(initial), min(count, len(members)))]})
            value = {'keeper_ids': keepers, 'requested_count': count}
            constraints[family_id] = value
            draft['constraints'][family_id] = {'value': value, 'owner': 'preview', 'version': capture['review_revision'] + 1}
        plan = self._compute_settings_plan(context, draft, active['settings'])
        for family_id, value in constraints.items(): value['confirmed_keeper_ids'] = plan['families'].get(family_id, {}).get('keeper_ids', value['keeper_ids'])
        record = self._store_plan(context, plan)
        member_ids = {context[3][i]['image_id'] for key in selected for i in population[key]['member_indices']}
        proposals = {key: value for key, value in plan['proposals'].items() if key in member_ids}
        source_counts = {}
        retained_counts = {}
        retained_ids = set(plan['retained_ids'])
        for image in context[3]:
            for obj in image['objects']:
                cls = str(obj['class_id']); source_counts[cls] = source_counts.get(cls, 0) + 1
                if image['image_id'] in retained_ids: retained_counts[cls] = retained_counts.get(cls, 0) + 1
        preview = {'selection_revision': capture['selection_revision'], 'review_revision': capture['review_revision'],
            'plan_revision': capture['plan_revision'], 'source_revision': context[2]['revision'], 'recipe_digest': plan['recipe_digest'],
            'family_ids': selected, 'requested_keep_count': count, 'constraints': constraints, 'proposals': proposals,
            'plan_record': record, 'plan_digest': plan['plan_digest'], 'removed_count': plan['removed_count'], 'retained_count': plan['retained_count'],
            'removed_percent': plan['removed_percent'], 'budget': active['settings']['max_removed_images'],
            'captured_eligible_count': len(proposals), 'captured_member_count': len(member_ids),
            'captured_retained_count': len(member_ids) - len(proposals), 'new_approval_count': len(proposals),
            'superseded_count': sum(value.get('status') == 'approved' and (key not in plan['proposals'] or plan['proposals'][key]['evidence_digest'] != value.get('evidence_digest')) for key, value in state['decisions'].items()),
            'additional_keeper_count': sum(max(0, len(plan['families'].get(key, {}).get('keeper_ids', [])) - count) for key in selected),
            'class_impact': {cls: {'source': value, 'retained': retained_counts.get(cls, 0)} for cls, value in source_counts.items()},
            'exceptions': [], 'plan': {key: {field: value for field, value in plan['families'].get(key, {}).items() if field != 'member_indices'} for key in selected}}
        identifier = digest(preview)
        with self.adapter.guard(), self.lock:
            self._check_capture(context, capture)
            with context[4].transaction() as committed:
                if committed['review_revision'] != capture['review_revision'] or committed['plan_revision'] != capture['plan_revision']: raise HTTPException(409, 'Review changed')
                committed['previews'][identifier] = preview
        return {'preview_digest': identifier, **{key: value for key, value in preview.items() if key not in {'constraints', 'proposals', 'plan_record'}}}

    def prepare_active_export(self, job_id, payload):
        context, state, capture = self._capture_plan(job_id, check=False)
        active = self._load_active_plan(context, state)
        if not active: return None
        if payload.get('selection_revision') != capture['selection_revision'] or payload.get('expected_review_revision') != capture['review_revision'] or payload.get('plan_revision') != capture['plan_revision']:
            raise HTTPException(409, 'The approved plan changed; refresh before exporting')
        removed = active['approved_indices']
        if not removed: raise HTTPException(400, 'Approve at least one exact proposed removal before exporting')
        retained = sorted(set(range(len(context[3]))) - set(removed))
        if active['settings']['policy'] == 'budgeted_coverage':
            if active.get('accounting_outdated'):
                raise HTTPException(409, 'Appearance accounting changed; request and Apply a new cached preview before exporting')
            index = self._coverage_index(context)
            metrics = index.evaluate(retained)
            self._check_budget_metrics(active['settings'], metrics)
            groups = export_scene_groups(context[3], context[1]['groups'], context[-1],
                                         witness_metrics=[active['metrics'], metrics])
        else:
            cfg, evidence = self._strict_evidence(context, state, active['settings'])
            for i in removed: removal_evidence(i, set(retained), evidence)
            metrics = coverage_for_retained(context[3], retained, evidence.coverage_rows(), cfg)
            groups = list(context[1]['groups'])
        self._check_capture(context, capture)
        return {'context': context, 'capture': capture, 'plan': active, 'metrics': metrics,
                'removed': removed, 'retained': retained, 'groups': groups}
