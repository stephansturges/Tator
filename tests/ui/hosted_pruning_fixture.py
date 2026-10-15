"""Synthetic feature inputs, real native planner/artifact publication for browser tests."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
from services.dataset_pruning import PruningConfig, build_order, pair_evidence, parse_objects
from services.dataset_pruning_evidence import PairStore
from services.dataset_pruning_jobs import write_json


def publish_pruning_fixture(analysis, actor, ds, tmp_path):
    captured = analysis.capture(actor, ds['id'], 'pruning', {}, 'pruning-capture')
    prepared = analysis.prepare(actor, captured['job_id'])
    rows = prepared['manifest']['images']
    native_id = 'dp_' + 'b' * 16
    directory = tmp_path / 'native-pruning'
    jobdir = directory / 'class_analysis/pruning' / native_id
    jobdir.mkdir(parents=True)
    config = PruningConfig(scene_threshold=.8, object_threshold=.8, index_threshold=.8)
    images = []
    for index, row in enumerate(rows):
        objects, errors = parse_objects(row['label_lines'], 1)
        objects[0]['feature_index'] = index
        images.append({**row, 'image_id': 'pruning-' + str(index), 'objects': objects, 'protection_errors': errors})
    values = np.array([[1., 0.], [1., 0.]], dtype=np.float32)
    np.save(jobdir / 'scene.npy', values)
    np.save(jobdir / 'hosted_objects.npy', values)
    evidence = PairStore(jobdir / 'evidence.sqlite3')
    evidence.append({'left': 0, 'right': 1, **pair_evidence(images[0], images[1], 1., values, config)})
    evidence.finish()
    manifest = {'components': {'bbox_fixture': {'dimension': 2, 'weight': 1}}}
    recipe = {'evidence_mode': 'compound', 'component_manifest': manifest}
    result = build_order(images, values, evidence, config=config)
    result.update(recipe=recipe, config=vars(config), selection_revision=1, image_count=2, source_revision=captured['dataset_revision'], class_names=['Car'])
    source = {'identity': 'hosted:' + ds['id'] + ':' + captured['snapshot_id'], 'revision': captured['dataset_revision'],
              'inventory_digest': 'frozen-fixture', 'labelmap': ['Car'], 'annotation_states': {}}
    for name, value in [('source.json', source), ('images.json', images), ('result.json', result), ('components.json', manifest),
                        ('state.json', {'status': 'completed', 'job_id': native_id}), ('hosted-capabilities.json', {'version': 3, 'default_config': vars(config), 'heads': []})]:
        write_json(jobdir / name, value)
    measured = {'snapshot_id': captured['snapshot_id'], 'dataset_id': ds['id'], 'revision': captured['dataset_revision'],
                'engine_result': {'native_job_id': native_id, 'result': result}}
    write_json(directory / 'result.json', measured)
    entries = [{'relative_path': path.relative_to(directory).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'size_bytes': path.stat().st_size}
               for path in directory.rglob('*') if path.is_file()]
    write_json(directory / 'receipt.json', {'operation': 'pruning', 'state': 'complete', 'result_path': 'result.json', 'artifacts': entries})
    receipt = directory / 'receipt.json'
    output = {'operation': 'pruning', 'snapshot_id': captured['snapshot_id'], 'dataset_id': ds['id'], 'revision': captured['dataset_revision'],
              'native_directory': str(directory), 'receipt_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest(), 'receipt_size_bytes': receipt.stat().st_size}
    analysis.publish_native_result(actor, captured['job_id'], output, 'pruning-publish', native_root=tmp_path)
    shutil.rmtree(directory)
    return captured['job_id']
