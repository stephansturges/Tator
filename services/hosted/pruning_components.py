"""Portable raw pruning components for the existing cache-only recipe lab.

Rows come from DescriptorCache, with its original normalization and producer
qualification. Replay applies the same saved geometry transform and fusion;
there is no encoder, training or fitting path in this module.
"""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

import numpy as np
from fastapi import HTTPException

from services.dataset_pruning import digest
from services.dataset_pruning_cache import DescriptorCache, file_state
from services.dataset_pruning_comparison import SelectedObjectFeatures
from utils.class_analysis_salad import fuse_class_analysis_feature_branches
from utils.embedding_recipe import (crop_bounds, embedding_view_specs, parse_yolo_geometry,
                                    replay_fitted_embedding_adjustment)
from .artifacts import ArtifactRef


SCHEMA = 'hosted-pruning-raw-components-v1'


def geometry_records(images, preprocessing):
    """The native parser/crop arithmetic, restricted to retained replay fields."""
    primary, padding, _ = embedding_view_specs(crop_mode=preprocessing['crop_mode'],
        padding_ratio=preprocessing['padding_ratio'], view_mode=preprocessing['embedding_view_mode'])[0]
    records = []
    for image in images:
        for obj in image['objects']:
            geometry = parse_yolo_geometry(image['label_lines'][obj['label_row']],
                image_width=image['width'], image_height=image['height'])
            if not geometry or geometry['kind'] != 'bbox':
                raise HTTPException(409, 'Saved raw object geometry is incompatible')
            bbox = geometry['bbox_xyxy']
            bounds = crop_bounds(bbox, image_width=image['width'], image_height=image['height'],
                                 crop_mode=primary, padding_ratio=padding)
            records.append({'image_id': image['image_id'], 'label_line_index': obj['label_row'],
                'feature_index': obj['feature_index'], '_image_sha256': image['image_sha256'],
                'geometry_key': obj['geometry_key'], 'bbox_xyxy': bbox, 'crop_xyxy': list(bounds),
                'width': int(round(bbox[2]-bbox[0])), 'height': int(round(bbox[3]-bbox[1]))})
    return records


def _sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def retain_components(directory, images, recipe, manifest, cache):
    """Copy bounded cached rows into one self-contained, immutable job bundle."""
    directory = Path(directory)
    if not recipe.get('bbox', {}).get('components'):
        return False
    records = geometry_records(images, recipe['bbox']['preprocessing'])
    if not records:
        return False
    if sorted(row['feature_index'] for row in records) != list(range(len(records))):
        raise HTTPException(409, 'Saved raw object identities are incomplete')
    records.sort(key=lambda row: row['feature_index'])
    components = {}
    with tempfile.TemporaryDirectory(prefix='.raw-components-', dir=directory) as temporary:
        temporary = Path(temporary)
        for component in recipe['bbox']['components']:
            name = 'bbox_' + component['id']
            if not name.replace('_', '').isalnum():
                raise HTTPException(409, 'Saved object component identity is invalid')
            entry = manifest.get('components', {}).get(name, {})
            width = int(entry.get('dimension', 0))
            if not entry.get('raw_recipe') or width <= 0:
                raise HTTPException(409, 'Saved raw producer bindings are unavailable')
            path = temporary / (name + '.npy')
            matrix = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(records), width))
            try:
                for start in range(0, len(records), 256):
                    batch = records[start:start + 256]
                    keys = [cache.key(row['_image_sha256'], entry['raw_recipe'], row['geometry_key']) for row in batch]
                    values = cache.get_many(keys)
                    if any(value is None or value.shape != (width,) or not np.isfinite(value).all() for value in values):
                        raise HTTPException(409, 'Raw descriptors are unavailable in the retained producer artifacts')
                    matrix[start:start + len(batch)] = np.asarray(values, np.float32)
                matrix.flush()
            finally:
                matrix._mmap.close()
            components[name] = {'file': path.name, 'dimension': width, 'sha256': _sha(path),
                                'raw_recipe_digest': digest(entry['raw_recipe'])}
        metadata = {'schema': SCHEMA, 'recipe_digest': digest(recipe), 'manifest_digest': digest(manifest),
                    'records': records, 'components': components}
        (temporary / 'manifest.json').write_text(json.dumps(metadata, sort_keys=True, separators=(',', ':')))
        target = directory / 'hosted_raw_components'
        if target.exists():
            raise HTTPException(409, 'Retained raw component bundle already exists')
        # Publish only a complete set. No partial bundle can enable the lab.
        temporary.rename(target)
    return True


class RetainedComponents:
    def __init__(self, directory):
        self.directory = Path(directory) / 'hosted_raw_components'
        self.verified = {}

    def replay(self, images, recipe, manifest, original):
        path = self.directory / 'manifest.json'
        if not path.is_file() or path.is_symlink():
            raise HTTPException(409, 'The saved analysis has no retained raw object rows for recipe comparison')
        saved = json.loads(path.read_text())
        if (saved.get('schema') != SCHEMA or saved.get('recipe_digest') != digest(original)
                or saved.get('manifest_digest') != digest(original.get('component_manifest', {}))
                or recipe.get('bbox', {}).get('preprocessing') != original.get('bbox', {}).get('preprocessing')):
            raise HTTPException(409, 'Saved raw object recipe binding changed')
        wanted = {(image['image_id'], obj['label_row']): obj['feature_index'] for image in images for obj in image['objects']}
        records = [row for row in saved['records'] if (row['image_id'], row['label_line_index']) in wanted]
        if len(records) != len(wanted) or any(wanted[(row['image_id'], row['label_line_index'])] != row['feature_index'] for row in records):
            raise HTTPException(409, 'Saved raw object membership changed')
        matrices, weights = [], []
        for component in recipe['bbox']['components']:
            name = 'bbox_' + component['id']; entry = manifest['components'][name]
            retained = saved['components'].get(name)
            if not retained or retained['raw_recipe_digest'] != digest(entry['raw_recipe']):
                raise HTTPException(409, 'Saved raw object producer changed')
            if retained['file'] != name + '.npy' or not name.replace('_', '').isalnum():
                raise HTTPException(409, 'Saved raw component path is invalid')
            path = self.directory / retained['file']
            state = file_state(path)
            if path.is_symlink():
                raise HTTPException(409, 'Saved raw component changed')
            if self.verified.get(path) != state:
                if _sha(path) != retained['sha256'] or file_state(path) != state:
                    raise HTTPException(409, 'Saved raw component changed')
                self.verified[path] = state
            values = np.load(path, mmap_mode='r', allow_pickle=False)
            try:
                if values.shape != (len(saved['records']), entry['dimension']) or values.dtype != np.float32:
                    raise HTTPException(409, 'Saved raw component dimensions changed')
                raw = np.asarray(values[[row['feature_index'] for row in records]], np.float32)
            finally:
                values._mmap.close()
            if file_state(path) != state or not np.isfinite(raw).all():
                raise HTTPException(409, 'Saved raw component changed')
            try:
                matrices.append(replay_fitted_embedding_adjustment(raw, records, entry['adjustment']))
            except ValueError as error:
                raise HTTPException(409, str(error)) from error
            weights.append(entry['weight'])
        fused = fuse_class_analysis_feature_branches(matrices, weights)
        return SelectedObjectFeatures({row['feature_index']: fused[index] for index, row in enumerate(records)}, fused.shape[1])


def recover_retained_components(directory, images, recipe, manifest, artifacts, references):
    """Upgrade an older publication using only its own retained cache artifacts.

    Absolute producer paths are never opened. A referenced matrix needs its
    recorded content hash in this job's immutable artifact index.
    """
    by_path = {row['relative_path']: row for row in references}
    database = by_path.get('class_analysis/pruning/cache/descriptors.sqlite3')
    if not database:
        return False
    by_hash = {row['sha256']: ArtifactRef(row['sha256'], row['size_bytes']) for row in references}
    with tempfile.TemporaryDirectory(prefix='.raw-recovery-', dir=directory) as temporary:
        temporary = Path(temporary)
        dbref = ArtifactRef(database['sha256'], database['size_bytes'])
        artifacts.pin(dbref, temporary)
        with closing(sqlite3.connect(f'file:{temporary / dbref.sha256}?mode=ro&immutable=1', uri=True)) as con:
            class Cache:
                key = staticmethod(DescriptorCache.key)
                def get_many(self, keys):
                    output = []
                    for key in keys:
                        row = con.execute('SELECT dim,vector,reference FROM descriptors WHERE key=?', (key,)).fetchone()
                        value = None
                        if row is not None:
                            width, blob, ref = row
                            if blob is not None:
                                value = np.frombuffer(blob, np.float32)
                            else:
                                ref = json.loads(ref)
                                artifact = by_hash.get(ref.get('sha256'))
                                if artifact is not None:
                                    artifacts.pin(artifact, temporary)
                                    matrix = np.load(temporary / artifact.sha256, mmap_mode='r', allow_pickle=False)
                                    try:
                                        value = np.array(matrix[ref['row'], ref.get('start', 0):ref['end']], np.float32, copy=True)
                                    finally:
                                        matrix._mmap.close()
                            if value is not None:
                                if value.shape != (width,) or not np.isfinite(value).all() or np.linalg.norm(value) <= 1e-8:
                                    value = None
                                else:
                                    value = value / max(float(np.linalg.norm(value)), 1e-12)
                        output.append(value)
                    return output
            return retain_components(directory, images, recipe, manifest, Cache())
