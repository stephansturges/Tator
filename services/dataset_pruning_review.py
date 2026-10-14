"""One durable, revisioned authority for family keepers and exact approvals.

The SQLite transaction commits the plan and every decision together. Request
receipts are durable; replay returns the current snapshot and the original
receipt. Undo owns only fields still bearing its operation/version marker.
"""
from contextlib import contextmanager
from collections import defaultdict, Counter
import copy
import json
from pathlib import Path
import re
import sqlite3
import time

import numpy as np
from fastapi import HTTPException

from services.dataset_pruning import UnionFind, build_order, digest, removal_evidence, selection


class SupplementaryEvidence:
    """Base pair measurements plus atomic ledger-owned cache-only comparisons."""
    def __init__(self, evidence, supplementary=None):
        self.evidence, self.supplementary = evidence, supplementary or {}

    @staticmethod
    def key(left, right): return ':'.join(str(value) for value in sorted((left, right)))

    def __iter__(self):
        seen = set()
        for edge in self.evidence:
            key = self.key(edge['left'], edge['right'])
            if key in self.supplementary: seen.add(key)
            yield self.supplementary.get(key, {}).get('edge', edge)
        for key, field in sorted(self.supplementary.items()):
            if key not in seen: yield field['edge']

    def edges_for(self, index):
        for edge in self.evidence.edges_for(index):
            if self.key(edge['left'], edge['right']) not in self.supplementary: yield edge
        for field in self.supplementary.values():
            edge = field['edge']
            if edge['valid'] and index in (edge['left'], edge['right']): yield edge

    def all_edges_for(self, index):
        """Overlay every measured pair without filtering by its original gate."""
        iterator = self.evidence.all_edges_for(index) if hasattr(self.evidence, 'all_edges_for') else self.evidence
        for edge in iterator:
            if index in (edge['left'], edge['right']) and self.key(edge['left'], edge['right']) not in self.supplementary:
                yield edge
        for field in self.supplementary.values():
            edge = field['edge']
            if index in (edge['left'], edge['right']): yield edge

    def valid_links(self):
        iterator = (self.evidence.valid_links() if hasattr(self.evidence, 'valid_links') else
                    ((edge['left'], edge['right']) for edge in self.evidence if edge['valid']))
        for left, right in iterator:
            if self.key(left, right) not in self.supplementary: yield left, right
        for field in self.supplementary.values():
            edge = field['edge']
            if edge['valid']: yield edge['left'], edge['right']

    def pair(self, left, right):
        field = self.supplementary.get(self.key(left, right))
        return field['edge'] if field else self.evidence.pair(left, right)

    def comparisons_for(self, members):
        selected, seen = set(members), set()
        iterator = self.evidence.comparisons_for(members) if hasattr(self.evidence, 'comparisons_for') else self.evidence
        for edge in iterator:
            if edge['left'] not in selected or edge['right'] not in selected: continue
            key = self.key(edge['left'], edge['right'])
            if key in self.supplementary: seen.add(key)
            yield self.supplementary.get(key, {}).get('edge', edge)
        for key, field in self.supplementary.items():
            edge = field['edge']
            if key not in seen and edge['left'] in selected and edge['right'] in selected: yield edge


class FilteredEvidence:
    def __init__(self, evidence, rejected, supplementary=None):
        self.evidence = SupplementaryEvidence(evidence, supplementary) if supplementary is not None else evidence
        self.rejected = rejected

    def filtered(self, edge):
        value = dict(edge)
        for candidate, witness, side in [(edge['left'], edge['right'], 'left'), (edge['right'], edge['left'], 'right')]:
            if f'{candidate}:{witness}' in self.rejected:
                value[side + '_valid'] = False
        value['valid'] = value['left_valid'] or value['right_valid']
        return value

    def __iter__(self):
        for edge in self.evidence:
            yield self.filtered(edge)

    def coverage_rows(self):
        # Reported descriptor coverage uses scene/class coverage measurements,
        # not the removal-valid flags changed by rejection or reweighting.
        source = getattr(self.evidence, 'coverage_rows', None)
        return source() if source is not None else iter(self)

    def edges_for(self, index):
        for edge in self.evidence.edges_for(index):
            yield self.filtered(edge)

    def comparisons_for(self, members):
        iterator = self.evidence.comparisons_for(members) if hasattr(self.evidence, 'comparisons_for') else self.evidence
        selected = set(members)
        for edge in iterator:
            if edge['left'] in selected and edge['right'] in selected: yield self.filtered(edge)


def empty_state():
    return {'schema': 'pruning-review-ledger-v1', 'review_revision': 0,
            'plan_revision': 1, 'pins': {}, 'constraints': {}, 'decisions': {},
            'rejected': {}, 'batches': {}, 'operations': {}, 'previews': {}, 'history': [], 'comparisons': {}, 'plans': {}}


class ReviewLedger:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def transaction(self):
        if self.path.is_symlink():
            raise HTTPException(409, 'Review path changed; preserve it for inspection')
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute('PRAGMA synchronous=FULL')
            db.execute('CREATE TABLE IF NOT EXISTS ledger (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)')
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT payload FROM ledger WHERE id=1').fetchone()
            state = json.loads(row[0]) if row else empty_state()
            if state.get('schema') != 'pruning-review-ledger-v1':
                raise HTTPException(503, 'Saved review cannot be read; preserve it for repair')
            state.setdefault('plans', {})
            yield state
            db.execute('INSERT OR REPLACE INTO ledger VALUES(1, ?)', (json.dumps(state, sort_keys=True, allow_nan=False),))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def read(self):
        if self.path.is_symlink(): raise HTTPException(409, 'Review path changed; preserve it for inspection')
        if self.path.exists():
            try:
                with sqlite3.connect(f'file:{self.path}?mode=ro', uri=True, timeout=30) as db:
                    row = db.execute('SELECT payload FROM ledger WHERE id=1').fetchone()
                if row:
                    state = json.loads(row[0])
                    if state.get('schema') != 'pruning-review-ledger-v1':
                        raise HTTPException(503, 'Saved review cannot be read; preserve it for repair')
                    state.setdefault('plans', {})
                    return state
            except sqlite3.OperationalError as error:
                if 'no such table' not in str(error): raise
        with self.transaction() as state: return copy.deepcopy(state)


def family_inventory(images, evidence):
    union = UnionFind(len(images))
    links = (evidence.valid_links() if hasattr(evidence, 'valid_links') else
             ((edge['left'], edge['right']) for edge in evidence if edge['valid']))
    for left, right in links: union.union(left, right)
    groups = defaultdict(list)
    for index in range(len(images)):
        groups[union.find(index)].append(index)
    return {digest(sorted(images[i]['image_id'] for i in members))[:24]: members
            for members in groups.values() if len(members) > 1}


def family_edges(evidence, members):
    selected = set(members)
    if hasattr(evidence, 'edges_for'):
        for index in members:
            for edge in evidence.edges_for(index):
                if edge['left'] == index and edge['right'] in selected:
                    yield edge
    else:
        for edge in evidence:
            if edge['left'] in selected and edge['right'] in selected:
                yield edge


def directed_digest(candidate, witness, edge, source_digest, recipe_digest):
    return digest({'candidate': candidate, 'witness': witness, 'source': source_digest,
                   'recipe': recipe_digest, 'evidence': edge})


def family_comparison_summary(images, members, keeper_ids, evidence):
    selected_ids = set(keeper_ids)
    keepers = {i for i in members if images[i]['image_id'] in selected_ids}
    measured, failed, seen = 0, 0, defaultdict(set)
    for edge in evidence.comparisons_for(members):
        for candidate, witness, side in [(edge['left'], edge['right'], 'left'), (edge['right'], edge['left'], 'right')]:
            if witness in keepers:
                measured += 1; seen[candidate].add(witness)
                failed += int(not edge[side+'_valid'])
    possible = len(members)*len(keepers)-len(keepers)
    examples = []
    for candidate in members:
        for witness in sorted(keepers-seen[candidate]-{candidate}):
            examples.append({'candidate_id': images[candidate]['image_id'], 'witness_id': images[witness]['image_id']})
            if len(examples) == 128: break
        if len(examples) == 128: break
    return {'measured_count': measured, 'measured_failing_or_rejected_count': failed,
            'unmeasured_count': possible-measured, 'unmeasured_examples': examples,
            'scope': 'Directed comparisons from family members to current keepers, excluding self; missing evidence does not prove novelty'}


def compute_plan(images, result, source, scene, evidence, state, budget, config, *, family_population=None, order_result=None):
    if type(budget) is not int or not 0 <= budget <= len(images):
        raise HTTPException(400, 'The global removal ceiling must be a valid image count')
    id_index = {image['image_id']: i for i, image in enumerate(images)}
    families = family_inventory(images, SupplementaryEvidence(evidence, state.get('comparisons'))) if family_population is None else family_population
    filtered = FilteredEvidence(evidence, state['rejected'], state.get('comparisons', {}))
    independent_pins = {identifier for identifier, field in state['pins'].items() if field['value']}
    pinned = set(independent_pins)
    for constraint in state['constraints'].values():
        pinned.update(constraint['value']['keeper_ids'])
    order = build_order(images, scene, filtered, sorted(pinned), config=config) if order_result is None else order_result
    actual = min(budget, order['max_removed_count'])
    chosen = selection(images, order, actual)
    retained = set(chosen['retained'])
    recipe_digest = digest(result['recipe'])
    proposals = {}
    for index in chosen['removed']:
        edge = removal_evidence(index, retained, filtered)
        witness = edge.pop('retained_index')
        candidate_id, witness_id = images[index]['image_id'], images[witness]['image_id']
        evidence_digest = directed_digest(candidate_id, witness_id, edge, source['inventory_digest'], recipe_digest)
        decision = state['decisions'].get(candidate_id)
        status = decision['status'] if decision and decision.get('evidence_digest') == evidence_digest and decision.get('witness_id') == witness_id else 'unreviewed'
        proposals[candidate_id] = {'candidate_id': candidate_id, 'witness_id': witness_id,
                                  'evidence_digest': evidence_digest, 'evidence': edge, 'status': status,
                                  'note': decision.get('note', '') if decision and decision.get('evidence_digest') == evidence_digest else ''}
    family_plans = {}
    for family_id, members in families.items():
        keeper_ids = [images[i]['image_id'] for i in members if i in retained]
        constraint = state['constraints'].get(family_id)
        value = constraint['value'] if constraint else {}
        family_proposals = [proposals[images[i]['image_id']] for i in members if images[i]['image_id'] in proposals]
        hard = [images[i]['image_id'] for i in members if images[i].get('protection_errors') or images[i]['image_id'] in independent_pins]
        exceptions = []
        requested = value.get('requested_count', 1)
        measured = family_comparison_summary(images, members, keeper_ids, filtered)
        if len(keeper_ids) > requested:
            exceptions.append('Direct witness coverage, protections, pins or the global ceiling retain additional images')
            if measured['unmeasured_count']:
                exceptions.append(f"{measured['unmeasured_count']} direct comparisons to current keepers are unmeasured. Extra keepers may reflect missing evidence rather than measured novelty; compute cached direct comparisons for selected keepers.")
        family_plans[family_id] = {'id': family_id, 'member_indices': members,
                                  'keeper_ids': keeper_ids, 'requested_count': requested,
                                  'confirmed': bool(constraint and sorted(value.get('confirmed_keeper_ids', [])) == sorted(keeper_ids)),
                                  'proposals': family_proposals,
                                  'hard_keep_ids': hard, 'pins': [key for key in keeper_ids if key in independent_pins],
                                  'uncovered_ids': [], 'exception_reasons': exceptions,
                                  'direct_comparisons': measured,
                                  'requirements': ['Every removal has a direct retained witness',
                                                   'Class/count inventory is covered; bbox appearance is unmeasured' if config and config.evidence_mode == 'scene_only' else
                                                   'Class-aware appearance and original scale coverage remain protected',
                                                   'Whole-image scene coverage is unmeasured' if config and config.evidence_mode == 'objects_only' else
                                                   'Whole-image scene evidence is measured']}
    plan_digest = digest({'budget': budget, 'keepers': sorted(images[i]['image_id'] for i in retained),
                          'proposals': [(key, value['witness_id'], value['evidence_digest']) for key, value in sorted(proposals.items())],
                          'constraints': {key: value['value'] for key, value in state['constraints'].items()},
                          'pins': {key: value['value'] for key, value in state['pins'].items()},
                          'rejected': sorted(state['rejected'])})
    approved = [id_index[key] for key, proposal in proposals.items() if proposal['status'] == 'approved']
    return {'order_result': order, 'chosen': chosen, 'families': family_plans, 'proposals': proposals,
            'plan_digest': plan_digest, 'budget': budget, 'approved_indices': sorted(approved),
            'review_revision': state['review_revision'], 'plan_revision': state['plan_revision'],
            'approved_count': len(approved), 'removed_count': actual, 'retained_count': len(retained),
            'recipe_digest': recipe_digest}


def choose_family_keepers(images, members, evidence, scene, requested, pins=(), explicit=None):
    """Greedy direct-coverage seed then the existing farthest-first selector."""
    from services.data_ingestion import greedy_diverse_indices_with_scores
    members = sorted(members, key=lambda i: images[i]['image_id'])
    if type(requested) is not int or requested < 1 or requested > len(members):
        raise HTTPException(400, 'Keep count must be between1andthe family size')
    ids = {images[i]['image_id']: i for i in members}
    if explicit is not None and (not isinstance(explicit, list) or not explicit or len(set(explicit)) != len(explicit) or set(explicit) - set(ids)):
        raise HTTPException(400, 'Explicit keepers must be distinct members of this family')
    required = {i for i in members if images[i].get('protection_errors') or images[i]['image_id'] in pins}
    covered_by = {i: {i} for i in members}
    for edge in family_edges(evidence, members):
        if edge['left'] not in covered_by or edge['right'] not in covered_by:
            continue
        if edge['left_valid']: covered_by[edge['right']].add(edge['left'])
        if edge['right_valid']: covered_by[edge['left']].add(edge['right'])
    keepers = required | ({ids[key] for key in explicit} if explicit else set())
    if not keepers:
        keepers.add(min(members, key=lambda i: (-len(covered_by[i]), images[i]['image_id'])))
    # Complete the direct-coverage seed independently of X. Completing coverage
    # after taking an X-dependent prefix can drop a former keeper when X grows.
    covered = set().union(*(covered_by[i] for i in keepers))
    while set(members) - covered:
        uncovered = set(members) - covered
        best = min(uncovered, key=lambda i: (-len(covered_by[i] & uncovered), images[i]['image_id']))
        keepers.add(best)
        covered.update(covered_by[best])
    selected_local = [members.index(i) for i in sorted(keepers, key=lambda i: images[i]['image_id'])]
    count = max(requested, len(keepers))
    if count == len(members): return sorted(ids)
    if count == len(keepers): return sorted(images[i]['image_id'] for i in keepers)
    order, _, _ = greedy_diverse_indices_with_scores(np.asarray(scene[members]), keep_fraction=1,
                                                     initial_indices=selected_local, target_count=count)
    keepers.update(members[i] for i in order[:count])
    return sorted(images[i]['image_id'] for i in keepers)
