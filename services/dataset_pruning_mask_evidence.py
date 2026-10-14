"""Diagnostic SAM provenance; independent of vectors, scores and approvals."""
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import time
import uuid

from services.dataset_pruning import digest

SCHEMA = 'pruning-sam-mask-evidence-v1'
FILENAME = 'sam-mask-evidence.json'
COMMIT = 'sam-mask-evidence.commit.json'


def sam_recipe_digest(recipe):
    component = next((value for value in recipe.get('bbox', {}).get('components', [])
                      if value.get('id') == 'sam3'), None)
    return digest(component['raw_recipe']) if component and component.get('raw_recipe') else None


def row_key(image_sha256, geometry_key):
    return str(image_sha256) + ':' + str(geometry_key)


def valid_row(row):
    if not isinstance(row, dict) or not isinstance(row.get('used_bbox_fallback'), bool): return False
    if type(row.get('descriptor_dimension')) is not int or row['descriptor_dimension'] != 768: return False
    dimensions = row.get('source_dimensions')
    if not isinstance(dimensions, list) or len(dimensions) != 2 or any(type(value) is not int or value <= 0 for value in dimensions): return False
    score = row.get('mask_score')
    if score is not None and (type(score) not in {int, float} or not math.isfinite(score) or not 0 <= score <= 1): return False
    observed = row.get('observed_at', 0.)
    if type(observed) not in {int, float} or not math.isfinite(observed) or observed < 0: return False
    return all(isinstance(row.get(key), str) and len(row[key]) == 64 for key in ['image_sha256', 'geometry_key'])


def valid_unknown(row):
    if not isinstance(row, dict) or not isinstance(row.get('reason'), str): return False
    dimensions, observed = row.get('source_dimensions'), row.get('observed_at')
    return (isinstance(dimensions, list) and len(dimensions) == 2 and all(type(value) is int and value > 0 for value in dimensions)
            and type(observed) in {int, float} and math.isfinite(observed) and observed >= 0
            and all(isinstance(row.get(key), str) and len(row[key]) == 64 for key in ['image_sha256', 'geometry_key']))


def read_sidecar(directory, raw_digest, *, source_digest=None):
    """Changed, incomplete or incompatible diagnostics become unknown."""
    try:
        path, receipt = Path(directory) / FILENAME, Path(directory) / COMMIT
        if path.is_symlink() or receipt.is_symlink(): return None
        encoded = path.read_bytes(); commit = json.loads(receipt.read_text())
        if not isinstance(commit, dict): return None
        if commit.get('schema') != SCHEMA or hashlib.sha256(encoded).hexdigest() != commit.get('sha256'): return None
        value = json.loads(encoded)
        if not isinstance(value, dict) or not isinstance(value.get('source_inventory_digest'), str): return None
        if value.get('schema') != SCHEMA or value.get('sam_raw_recipe_digest') != raw_digest: return None
        if source_digest is not None and value.get('source_inventory_digest') != source_digest: return None
        if not isinstance(value.get('rows'), dict): return None
        if any(not valid_row(row) or key != row_key(row['image_sha256'], row['geometry_key']) for key, row in value['rows'].items()): return None
        unknown = value.get('unknown_rows', {})
        if not isinstance(unknown, dict) or any(not valid_unknown(row) or key != row_key(row['image_sha256'], row['geometry_key']) for key, row in unknown.items()): return None
        return value
    except (OSError, ValueError, TypeError, KeyError): return None


def _write_json(path, value):
    temporary = path.with_name('.' + path.name + '-' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w') as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_sidecar(directory, source, recipe, images, rows, *, lineage=None, diagnostic_issues=None, unknown_rows=None):
    """Rebind label-agnostic diagnostics only to exact current pixels/geometry."""
    directory = Path(directory); raw_digest = sam_recipe_digest(recipe)
    if not raw_digest: return {'used': False, 'known_objects': 0, 'unknown_objects': 0}
    bound, unknown_bound, count = {}, {}, 0
    unknown_rows = unknown_rows or {}
    total = sum(len(image['objects']) for image in images)
    for image in images:
        for obj in image['objects']:
            key = row_key(image['image_sha256'], obj['geometry_key']); value = rows.get(key)
            unknown = unknown_rows.get(key)
            if unknown and valid_unknown(unknown) and unknown['source_dimensions'] == [image['width'], image['height']]:
                if not value or unknown['observed_at'] >= value.get('observed_at', 0.):
                    unknown_bound[key] = unknown; continue
            if not value or not valid_row(value) or value['source_dimensions'] != [image['width'], image['height']]: continue
            bound[key] = value; count += 1
    value = {'schema': SCHEMA, 'job_id': directory.name,
             'source_inventory_digest': source['inventory_digest'], 'source_revision': source['revision'],
             'sam_raw_recipe_digest': raw_digest, 'rows': bound, 'unknown_rows': unknown_bound, 'lineage': lineage or [],
             'summary': {'used': True, 'known_objects': count, 'unknown_objects': total-count,
                         'known_unique_content_geometries': len(bound),
                         'diagnostic_issues': diagnostic_issues or [],
                         'scope': 'Per-object mask method and segmentation score; not object similarity or accuracy.'}}
    _write_json(directory / FILENAME, value)
    _write_json(directory / COMMIT, {'schema': SCHEMA, 'sha256': hashlib.sha256((directory / FILENAME).read_bytes()).hexdigest()})
    return value['summary']


def collect_run_evidence(root, directory, source, recipe, images, records, stats):
    """Retain exact known cached metadata and new producer JSONL before cleanup."""
    raw_digest = sam_recipe_digest(recipe)
    if not raw_digest: return save_sidecar(directory, source, recipe, images, {})
    rows, unknown_rows, lineage, conflicts, issues = {}, {}, [], set(), []
    candidates = [Path(directory)] + sorted(Path(root).glob('dp_*'))
    for previous in dict.fromkeys(candidates):
        evidence = read_sidecar(previous, raw_digest)
        if evidence is None: continue
        lineage.append({'path': str(previous / FILENAME), 'sha256': hashlib.sha256((previous / FILENAME).read_bytes()).hexdigest(),
                        'source_inventory_digest': evidence['source_inventory_digest']})
        for key, value in evidence['rows'].items():
            existing = rows.get(key)
            # Different observations cannot silently become the metadata of a
            # shared cached descriptor. Keep an explicit unknown in that case.
            fields = ['used_bbox_fallback', 'mask_score', 'source_dimensions', 'descriptor_dimension']
            if existing and any(existing[field] != value[field] for field in fields):
                conflicts.add(key)
                unknown_rows[key] = {field: value[field] for field in ['image_sha256', 'geometry_key', 'source_dimensions']}
                unknown_rows[key].update(reason='Conflicting compatible SAM observations; this bbox mask method is unknown',
                                         observed_at=max(existing.get('observed_at', 0.), value.get('observed_at', 0.)))
            elif not existing or value.get('observed_at', 0.) >= existing.get('observed_at', 0.): rows[key] = value
        for key, value in evidence.get('unknown_rows', {}).items():
            if key not in unknown_rows or value['observed_at'] >= unknown_rows[key]['observed_at']: unknown_rows[key] = value
    runtime_value = stats.get('components', {}).get('bbox_sam3', {}).get('mask_evidence', {})
    runtime = runtime_value if isinstance(runtime_value, dict) else {}
    raw_value = runtime.get('per_object_evidence_path') if isinstance(runtime_value, dict) else True
    fresh_records = [record for record in records if 'sam3_mask_evidence' in record]
    if raw_value or fresh_records:
        # Invalid diagnostic data never invalidates valid descriptor vectors.
        # Discard old observations of freshly encoded rows as well, so an
        # unknown new observation cannot silently inherit an old mask claim.
        observed_at = time.time()
        dimensions = {image['image_sha256']: [image['width'], image['height']] for image in images}
        def mark_unknown(record, reason):
            key = row_key(record['_image_sha256'], record['geometry_key'])
            rows.pop(key, None)
            unknown_rows[key] = {'image_sha256': record['_image_sha256'], 'geometry_key': record['geometry_key'],
                                 'source_dimensions': dimensions[record['_image_sha256']], 'observed_at': observed_at,
                                 'reason': reason, 'origin_job_id': Path(directory).name}
        try:
            if not raw_value:
                raise ValueError('Freshly encoded SAM rows have no per-object diagnostic path; their mask method is unknown')
            raw_path = Path(raw_value)
            if raw_path.is_symlink() or not raw_path.resolve().is_relative_to((Path(directory) / 'features').resolve()):
                raise ValueError('SAM diagnostics are outside the current producer directory')
            encoded = raw_path.read_bytes(); raw_sha = hashlib.sha256(encoded).hexdigest()
            by_crop = {}
            for record in records: by_crop.setdefault(record['crop_cache_key'], []).append(record)
            fresh_crops = {record['crop_cache_key'] for record in records if 'sam3_mask_evidence' in record}
            producer_rows = [json.loads(line) for line in encoded.splitlines() if line.strip()]
            if len(producer_rows) != runtime.get('object_count'): raise ValueError('SAM diagnostic row count changed')
            produced, seen_crops = {}, set()
            for item in producer_rows:
                matches = by_crop.get(item.get('crop_cache_key'), []) if isinstance(item, dict) else []
                if not matches: raise ValueError('SAM diagnostic crop is absent from the current source')
                crop = item['crop_cache_key']
                if crop in seen_crops: raise ValueError('Duplicate producer SAM diagnostic crop; fresh observations are unknown')
                seen_crops.add(crop)
                if crop not in fresh_crops: raise ValueError('SAM diagnostic crop is not a freshly encoded row')
                fields = ['used_bbox_fallback', 'mask_score', 'source_dimensions', 'descriptor_dimension']
                if any(any(item.get(field) != record['sam3_mask_evidence'].get(field) for field in fields)
                       for record in matches if 'sam3_mask_evidence' in record):
                    raise ValueError('SAM diagnostics changed from the fresh producer observation')
                for record in matches:
                    key = row_key(record['_image_sha256'], record['geometry_key'])
                    rows.pop(key, None)
                    value = {field: item.get(field) for field in ['used_bbox_fallback', 'mask_score', 'source_dimensions', 'descriptor_dimension']}
                    value.update(image_sha256=record['_image_sha256'], geometry_key=record['geometry_key'],
                                 origin_job_id=Path(directory).name, origin_source_digest=source['inventory_digest'],
                                 producer_evidence_sha256=raw_sha, crop_cache_key=record['crop_cache_key'], observed_at=observed_at)
                    if not valid_row(value) or value['source_dimensions'] != [record['_image_width'], record['_image_height']]:
                        reason = 'An encoded bbox has invalid per-object SAM diagnostics; its mask method is unknown'
                        issues.append(reason); mark_unknown(record, reason)
                    else: produced[key] = value
            if hashlib.sha256(raw_path.read_bytes()).hexdigest() != raw_sha: raise ValueError('SAM diagnostics changed during preservation')
            rows.update(produced)
            for key in produced: conflicts.discard(key); unknown_rows.pop(key, None)
            lineage.append({'producer_path': str(raw_path), 'sha256': raw_sha, 'object_count': len(producer_rows)})
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(str(error))
            affected = fresh_records or records
            for record in affected: mark_unknown(record, str(error))
    for key in conflicts: rows.pop(key, None)
    return save_sidecar(directory, source, recipe, images, rows, lineage=lineage, diagnostic_issues=issues, unknown_rows=unknown_rows)


def copy_sidecar(source, destination, raw_digest):
    if read_sidecar(source, raw_digest) is None: return False
    for name in [FILENAME, COMMIT]: shutil.copy2(Path(source) / name, Path(destination) / name)
    return read_sidecar(destination, raw_digest) is not None


def object_mask_evidence(sidecar, image, label_row, *, sam_used):
    if not sam_used: return {'status': 'not_used', 'reason': 'SAM is absent from the selected object recipe'}
    obj = next((value for value in image['objects'] if value['label_row'] == label_row), None)
    value = sidecar.get('rows', {}).get(row_key(image['image_sha256'], obj['geometry_key'])) if sidecar and obj else None
    if not value or value['source_dimensions'] != [image['width'], image['height']]:
        unknown = sidecar.get('unknown_rows', {}).get(row_key(image['image_sha256'], obj['geometry_key'])) if sidecar and obj else None
        return {'status': 'unknown', 'reason': unknown.get('reason') if unknown else 'No compatible per-object mask observation; historical aggregate evidence cannot identify this bbox'}
    return {'status': 'known', **value}
