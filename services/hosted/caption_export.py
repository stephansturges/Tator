"""Native instruction export with identities supplied by a confirmed snapshot.

Called only inside NativeAnalysisAdapter's private child. The native readiness,
row selection, bundle and checksum implementations remain unchanged. These
bindings replace live-folder discovery with the exact frozen image population.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil

from .caption_semantics import caption_lifecycle, caption_text, primary_caption_index


def _native_caption_source(record):
    """Expose hosted model output to native provenance rules without relabeling history."""
    source = str(record.get('source') or '').strip().lower()
    if source in {'hosted_qwen', 'hosted_paid_batch'}:
        return 'qwen_caption_' + source
    return record.get('source', 'manual')


def _native_lifecycle(record, parent=None):
    """Preserve inactivity despite the native reader's smaller vocabulary."""
    status = caption_lifecycle(record, parent)
    return 'superseded' if status == 'deleted' else status


def _effective_review(record, kind):
    """Human and machine qualification apply only to their original text."""
    from .instruction_review import review_content_digest

    metadata = record.get('metadata') if isinstance(record.get('metadata'), dict) else {}
    digest = review_content_digest(record, kind)
    human_changed = (metadata.get('review_recorded_via') == 'hosted_instruction_review'
                     and metadata.get('review_content_sha256') != digest)
    machine_changed = (kind == 'generated_qa' and metadata.get('machine_validation_recorded_via') == 'hosted_caption_result'
                       and metadata.get('machine_validation_content_sha256') != digest)
    if not human_changed and not machine_changed:
        return record
    # The original decision and all its attribution remain in hosted_source.
    # Clear both native precedence paths so neither can revive that decision.
    metadata = dict(metadata)
    effective = {**record, 'metadata': metadata}
    if human_changed:
        for key in ('review_notes', 'reviewer', 'reviewed_at'):
            metadata.pop(key, None)
        metadata.update(review_decision='unreviewed', review_status='unreviewed',
                        hosted_review_invalidated='content_changed')
        effective.update(review_decision='unreviewed', review_status='unreviewed')
    if machine_changed:
        metadata['hosted_machine_validation_invalidated'] = 'content_changed'
    if kind == 'generated_qa':
        # Native machine-validated QA does not require human review. That
        # qualification also describes the old text; rerun the native checks
        # and require a new human decision for the changed candidate.
        validation = dict(record.get('validation')) if isinstance(record.get('validation'), dict) else {}
        metadata_validation = dict(metadata.get('validation')) if isinstance(metadata.get('validation'), dict) else {}
        for value, key in ((effective, 'validation_status'), (metadata, 'validation_status'), (validation, 'status'),
                           (metadata_validation, 'status'),
                           (effective, 'review_status'), (effective, 'review_decision'),
                           (metadata, 'review_status'), (metadata, 'review_decision')):
            if re.sub(r'[\s-]+', '_', str(value.get(key) or '').strip().lower()) == 'machine_validated':
                value[key] = 'unreviewed' if key.startswith('review_') else 'unknown'
        if validation:
            effective['validation'] = validation
        if metadata_validation:
            metadata['validation'] = metadata_validation
    return effective


def export_native_instructions(api, manifest, source, directory, options):
    from api.datasets import _instruction_export_not_ready_reason

    dataset_id = manifest['dataset_id']
    entry = {**api._resolve_dataset_entry(dataset_id), 'yolo_layout': 'split'}
    rows = [{**row, 'split': row['original_split'], 'image_name': row['image_relpath'],
             'image_width': row['width'], 'image_height': row['height'],
             'label_source': 'hosted_snapshot'} for row in source['images']]
    by_key = {(row['split'], row['image_relpath']): row for row in rows}
    captions, instructions = [], []
    for row in rows:
        identity = {'image_id': row['hosted_image_id'], 'version_id': row['hosted_version_id'],
                    'snapshot_id': manifest['snapshot_id'], 'media_sha256': row['image_sha256']}
        image_key = row['split'] + '/' + row['image_relpath']
        records = [dict(value) if isinstance(value, dict) else {'caption': value}
                   for value in row.get('caption_records', []) if isinstance(value, (str, dict))]
        primary = str(row.get('text_label') or '').strip()
        selected = primary_caption_index(records, primary)
        if primary and selected is None:
            records.insert(0, {'caption': primary, 'source': 'manual', 'lifecycle_status': 'active', 'is_primary': True})
            selected = 0
        for index, record in enumerate(records):
            text = caption_text(record)
            identifier = 'hosted-caption-' + hashlib.sha256(f"{image_key}:{index}:{text}".encode()).hexdigest()
            recorded_metadata = record.get('metadata') if isinstance(record.get('metadata'), dict) else {}
            metadata = {**recorded_metadata, 'hosted_source': dict(record), 'hosted_identity': identity}
            for key in ('caption_quality', 'quality_failures', 'used_counts', 'used_boxes', 'recovery_events'):
                if key in record:
                    metadata[key] = record[key]
            # Equal text can carry different model, quality or review evidence.
            # Native export owns row selection; retain each original record.
            if text:
                captions.append(_effective_review({**record, 'id': identifier, 'image_name': row['image_relpath'], 'image_key': image_key,
                    'split': row['split'], 'caption': text, 'source': _native_caption_source(record),
                    'lifecycle_status': _native_lifecycle(record),
                    'is_primary': index == selected, 'metadata': metadata}, 'caption0'))
            pairs = record.get('generated_qa_pairs') or recorded_metadata.get('generated_qa_pairs') or []
            for number, pair in enumerate(pairs):
                if not isinstance(pair, dict):
                    raise ValueError('Confirmed instruction row must be an object')
                instructions.append(_effective_review({**pair, 'id': identifier + '-qa-' + str(number), 'caption_id': identifier,
                    'lifecycle_status': _native_lifecycle(pair, record),
                    'image_name': row['image_relpath'], 'image_key': image_key, 'split': row['split'],
                    'caption': text, 'metadata': {**(pair.get('metadata') if isinstance(pair.get('metadata'), dict) else {}),
                        'hosted_source': dict(pair), 'hosted_caption_source': dict(record), 'hosted_identity': identity}}, 'generated_qa'))

    def split(value):
        if value not in {'train', 'val', 'test', 'unsplit'}:
            raise ValueError('Frozen instruction split changed')
        return value

    def resolve(identifier):
        if identifier != dataset_id:
            raise ValueError('Frozen instruction dataset changed')
        return entry

    bindings = {
        '_resolve_dataset_entry': resolve,
        '_annotation_normalise_split': split,
        '_annotation_collect_images': lambda *_args, **_kwargs: rows,
        '_annotation_manifest_for_entry': lambda *_args, **_kwargs: {**source, 'images': rows, 'yolo_layout': 'split'},
        '_annotation_effective_text_label': lambda _entry, relative, selected: by_key[(split(selected), Path(relative).as_posix())]['text_label'],
        '_annotation_effective_label_payload': lambda _entry, selected, relative: {
            'label_lines': by_key[(split(selected), Path(relative).as_posix())]['label_lines'],
            'label_source': 'hosted_snapshot',
            'label_source_present': by_key[(split(selected), Path(relative).as_posix())]['label_source_present']},
        '_load_dataset_caption_records': lambda *_args, **_kwargs: captions,
        '_load_dataset_caption_instruction_records': lambda *_args, **_kwargs: instructions,
    }
    previous = {name: getattr(api, name) for name in bindings}
    output = Path(directory) / 'caption_export'
    output.mkdir()
    try:
        for name, value in bindings.items():
            setattr(api, name, value)
        exported = api.export_captions(dataset_id, options)
        report = exported['instruction_report']
        for filename, value in [('captions-export.json', exported), ('readiness.json', report), ('hosted-snapshot.json', manifest)]:
            (output / filename).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
        reason = _instruction_export_not_ready_reason(exported)
        bundle = None
        if (not options.get('require_ready_instruction_export', True) or not reason) and exported.get('instruction_archive_rows'):
            response = api.download_caption_instruction_bundle(dataset_id, options, export_payload=exported)
            temporary = Path(response.path)
            bundle = output / 'caption-training.zip'
            try:
                shutil.copyfile(temporary, bundle)
                api._caption_instruction_validate_bundle_manifest(bundle)
            finally:
                if temporary.parent.name.startswith('caption_instruction_bundle_'):
                    shutil.rmtree(temporary.parent)
        return {'summary': exported['instruction_summary'], 'readiness': report.get('training_readiness', {}),
                'bundle_path': 'caption_export/caption-training.zip' if bundle else None,
                'export_path': 'caption_export/captions-export.json', 'report_path': 'caption_export/readiness.json',
                'snapshot_path': 'caption_export/hosted-snapshot.json', 'blocked_reason': reason,
                'diagnostic_export': bool(bundle and reason), 'image_count': len(rows),
                'mechanisms': ['export_captions', '_instruction_export_not_ready_reason', 'download_caption_instruction_bundle', '_caption_instruction_validate_bundle_manifest']}
    finally:
        for name, value in previous.items():
            setattr(api, name, value)
