"""Retained native projection fixture; no model-extraction claim."""
import hashlib
import json
import shutil
from services.class_analysis_session_store import build_class_analysis_session_store


def publish_dqe_fixture(service, actor, dataset, root):
    job = service.capture(actor, dataset['id'], 'dqe', {}, 'browser-dqe-capture')
    request = service.dqe_request(actor, job['job_id'])
    images = request['workspace_manifest']['images']
    directory = root / 'native-dqe' / 'run'
    native_id = 'ca_browser_fixture'
    session = directory / f'class_analysis/{native_id}/session.sqlite3'
    session.parent.mkdir(parents=True)
    points = [{'point_id': f'point-{i}', 'class_id': '0', 'class_name': 'Car',
               'image_relpath': image['hosted_image_id'] + '.png', 'label_line_index': 0,
               'bbox_xyxy': [20, 30, 140, 120], 'projection': [float(i), float(-i)],
               'identity_status': 'ready', 'annotation_entity_id': f'fixture-object-{i}', 'annotation_entity_revision': 1,
               'annotation_entity_record_revision': 'fixture-version', 'annotation_source_identity': 'hosted:' + dataset['id'],
               'source_record_key': image['hosted_image_id'] + '.png', 'annotation_attestation': f'fixture-{i}',
               'quality_review_candidate': True, 'review_priority_score': 1 - i/10}
              for i, image in enumerate(images)]
    build_class_analysis_session_store(session, points, summary={'projection_mode': 'umap'}, request={'projection_mode': 'umap'}, expected_point_count=len(points))
    config = session.parent / 'config.json'; config.write_text(json.dumps({'projection_mode': 'umap'}))
    result = directory / 'result.json'
    result.write_text(json.dumps({'snapshot_id': job['snapshot_id'], 'dataset_id': dataset['id'], 'revision': job['dataset_revision'],
        'engine_result': {'native_job_id': native_id, 'result': {'summary': {'object_count': len(points)}, 'points': points}}}))
    entries = [{'relative_path': p.relative_to(directory).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'size_bytes': p.stat().st_size} for p in (session, config, result)]
    receipt = directory / 'receipt.json'
    receipt.write_text(json.dumps({'operation': 'dqe', 'state': 'complete', 'result_path': 'result.json', 'artifacts': entries}))
    service.publish_native_result(actor, job['job_id'], {'operation': 'dqe', 'snapshot_id': job['snapshot_id'], 'dataset_id': dataset['id'], 'revision': job['dataset_revision'], 'native_directory': str(directory), 'receipt_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest(), 'receipt_size_bytes': receipt.stat().st_size}, 'browser-dqe-publish', native_root=directory.parent)
    shutil.rmtree(directory)
