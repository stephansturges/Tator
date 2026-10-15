"""Original recipe-lab math over retained raw rows; no model extraction."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from services.dataset_pruning import PruningConfig, build_order, digest, pair_evidence, parse_objects
from services.dataset_pruning_cache import DescriptorCache
from services.dataset_pruning_comparison import reconstruct_cached_object_rows
from services.dataset_pruning_components import object_records
from services.dataset_pruning_correction import effective_transform
from services.dataset_pruning_evidence import PairStore
from services.dataset_pruning_jobs import write_json
from services.dataset_pruning_recipe_lab import variant_recipe
from services.hosted.pruning import FrozenRows, HostedPruning
from services.hosted.pruning_components import geometry_records, retain_components, recover_retained_components
from utils.class_analysis_salad import fuse_class_analysis_feature_branches
from utils.embedding_recipe import crop_bounds, embedding_view_specs, parse_yolo_geometry, replay_fitted_embedding_adjustment
from test_hosted_analysis import setup


def material(tmp_path, images):
    preprocessing = {'crop_mode': 'padded_square', 'padding_ratio': .15, 'embedding_view_mode': 'standard',
                     'preprocess_mode': 'canonical', 'canonical_size': 224, 'background_mode': 'none'}
    api = SimpleNamespace(_embedding_view_specs=embedding_view_specs, _class_analysis_parse_yolo_geometry=parse_yolo_geometry,
        _class_analysis_crop_bounds=crop_bounds, _class_analysis_cache_digest=digest, CLASS_ANALYSIS_CACHE_VERSION='fixture',
        _class_analysis_image_pack_group_key=lambda sha, rows: sha,
        _class_analysis_replay_fitted_embedding_adjustment=replay_fitted_embedding_adjustment,
        fuse_class_analysis_feature_branches=fuse_class_analysis_feature_branches)
    fit = {'applied': True, 'correction_strength': 1., 'covariates': ['log_bbox_area'], 'fitted_transform': {
        'schema': 'independent_size_residualization_v1', 'normalization': 'subtract_design_then_l2',
        'covariate_mean': [3.], 'covariate_std': [1.], 'coefficients': [[.12, .06], [.04, -.03]],
        'record_digest': 'fixed-fit', 'fit_population_count': 20, 'design_sha256': 'fixed-design'}}
    manifest = {'components': {'scene_0': {'dimension': 2, 'weight': 1., 'raw_recipe': {'producer': 'scene'}, 'adjustment': {'applied': False}}}}
    components = []
    cache = DescriptorCache(tmp_path / 'cache')
    for ordinal, name in enumerate(('dinov3', 'sam3', 'salad', 'cradio')):
        strength = 0. if name == 'cradio' else 1.
        components.append({'id': name, 'weight': .25, 'adjustment_strength': strength})
        entry = {'dimension': 2, 'weight': .25, 'raw_recipe': {'producer': name}, 'adjustment': effective_transform(fit, strength)}
        manifest['components']['bbox_' + name] = entry
        for index, image in enumerate(images):
            for obj in image['objects']:
                cache.put(cache.key(image['image_sha256'], entry['raw_recipe'], obj['geometry_key']),
                          np.array([1. + ordinal * .3, .3 + index * (.3 + ordinal * .2)], np.float32))
    recipe = {'evidence_mode': 'compound', 'bbox': {'components': components, 'preprocessing': preprocessing},
              'scene': {'components': [{'weight': 1.}]}, 'component_manifest': manifest}
    return api, cache, recipe, manifest


def images_from_rows(rows):
    images = []
    for index, row in enumerate(rows):
        objects, errors = parse_objects(row['label_lines'], 1)
        for obj in objects:
            obj['feature_index'] = index
        images.append({**row, 'image_id': 'pruning-' + str(index), 'objects': objects, 'protection_errors': errors})
    return images


def test_retained_raw_rows_equal_native_replay_and_preserve_variant_differences(setup, tmp_path):
    _, _, analysis, actor, ds, _ = setup
    captured = analysis.capture(actor, ds['id'], 'pruning', {}, 'capture-components')
    images = images_from_rows(analysis.prepare(actor, captured['job_id'])['manifest']['images'])
    api, cache, recipe, manifest = material(tmp_path, images)
    expected_records = object_records(api, copy.deepcopy(images), recipe['bbox']['preprocessing'])
    for native, retained in zip(expected_records, geometry_records(images, recipe['bbox']['preprocessing'])):
        for name in ('bbox_xyxy', 'crop_xyxy', 'width', 'height', 'geometry_key', 'image_id', 'label_line_index'):
            assert list(native[name]) == list(retained[name]) if name in ('bbox_xyxy', 'crop_xyxy') else native[name] == retained[name]
    target = tmp_path / 'job'; target.mkdir()
    write_json(target / 'result.json', {'recipe': recipe})
    original = reconstruct_cached_object_rows(api, cache, images, recipe, manifest)
    np.save(target / 'hosted_objects.npy', original[[0, 1]])
    retain_components(target, images, recipe, manifest, cache)
    frozen = FrozenRows(cache, target)
    scores = []
    for alpha, weight in ((0., 0.), (0., .25), (1., .25), (.5, .6)):
        varied, produced = variant_recipe(recipe, manifest, {'alpha': alpha, 'cradio_weight': weight})
        expected = reconstruct_cached_object_rows(api, cache, images, varied, produced)
        actual = reconstruct_cached_object_rows(None, frozen, images, varied, produced)
        np.testing.assert_array_equal(actual[[0, 1]], expected[[0, 1]])
        scores.append(float(actual[0] @ actual[1]))
    assert len(set(scores)) == 4
    np.testing.assert_array_equal(frozen.replay_object_rows(images, recipe, manifest)[[0, 1]], original[[0, 1]])
    # Changed producer bytes must never become a different plausible measurement.
    path = target / 'hosted_raw_components/bbox_cradio.npy'
    values = np.load(path); values[0] = [0., 1.]; np.save(path, values)
    with pytest.raises(HTTPException, match='component changed'):
        frozen.replay_object_rows(images, varied, produced)


def test_older_raw_cache_recovery_uses_only_this_jobs_retained_artifacts(setup, tmp_path):
    _, artifacts, analysis, actor, ds, _ = setup
    captured = analysis.capture(actor, ds['id'], 'pruning', {}, 'capture-legacy')
    images = images_from_rows(analysis.prepare(actor, captured['job_id'])['manifest']['images'])
    api, cache, recipe, manifest = material(tmp_path, images)
    producer = tmp_path / 'producer.npy'; np.save(producer, np.array([[1., 2.], [3., 4.]], np.float32))
    producer_ref = artifacts.put_file(producer)
    keys = [cache.key(image['image_sha256'], manifest['components']['bbox_cradio']['raw_recipe'], image['objects'][0]['geometry_key']) for image in images]
    with cache.connect() as con:
        for index, key in enumerate(keys):
            con.execute('UPDATE descriptors SET vector=NULL,reference=? WHERE key=?',
                (json.dumps({'path': '/forbidden/original.npy', 'sha256': producer_ref.sha256, 'row': index, 'end': 2}), key))
    with cache.connect() as con:
        con.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    cache_ref = artifacts.put_file(cache.path)
    refs = [{'relative_path': 'class_analysis/pruning/cache/descriptors.sqlite3', **cache_ref.as_dict()},
            {'relative_path': 'producer.npy', **producer_ref.as_dict()}]
    shutil.rmtree(tmp_path / 'cache'); producer.unlink()
    target = tmp_path / 'old-job'; target.mkdir()
    assert recover_retained_components(target, images, recipe, manifest, artifacts, refs)
    np.testing.assert_array_equal(np.load(target / 'hosted_raw_components/bbox_cradio.npy'),
                                  np.array([[1., 2.], [3., 4.]], np.float32) / np.array([[np.sqrt(5)], [5]], np.float32))
    missing = tmp_path / 'missing'; missing.mkdir()
    with pytest.raises(HTTPException, match='retained producer'):
        recover_retained_components(missing, images, recipe, manifest, artifacts, refs[:1])
    assert not (missing / 'hosted_raw_components').exists()


def test_hosted_recipe_lab_uses_original_comparison_and_checkpoint_after_restart(setup, tmp_path, monkeypatch):
    store, artifacts, analysis, actor, ds, _ = setup
    captured = analysis.capture(actor, ds['id'], 'pruning', {}, 'capture-lab')
    rows = analysis.prepare(actor, captured['job_id'])['manifest']['images']
    images = images_from_rows(rows)
    api, cache, recipe, manifest = material(tmp_path, images)
    native_id = 'dp_' + 'c' * 16
    directory = tmp_path / 'native'; target = directory / 'class_analysis/pruning' / native_id; target.mkdir(parents=True)
    values = reconstruct_cached_object_rows(api, cache, images, recipe, manifest)[[0, 1]]
    scene = np.array([[1., 0.], [1., 0.]], np.float32)
    np.save(target / 'scene.npy', scene); np.save(target / 'hosted_objects.npy', values)
    config = PruningConfig(scene_threshold=.8, object_threshold=.8, index_threshold=.8)
    evidence = PairStore(target / 'evidence.sqlite3')
    evidence.append({'left': 0, 'right': 1, **pair_evidence(images[0], images[1], 1., values, config)}); evidence.finish()
    result = build_order(images, scene, evidence, config=config)
    result.update(recipe=recipe, config=vars(config), selection_revision=1, image_count=2, source_revision=captured['dataset_revision'], class_names=['Car'])
    source = {'identity': 'hosted:' + ds['id'] + ':' + captured['snapshot_id'], 'revision': captured['dataset_revision'],
              'inventory_digest': 'raw-fixture', 'labelmap': ['Car'], 'annotation_states': {}}
    for filename, value in [('source.json', source), ('images.json', images), ('result.json', result),
                             ('components.json', manifest), ('state.json', {'status': 'completed', 'job_id': native_id})]:
        write_json(target / filename, value)
    retain_components(target, images, recipe, manifest, cache)
    write_json(directory / 'result.json', {'snapshot_id': captured['snapshot_id'], 'dataset_id': ds['id'], 'revision': captured['dataset_revision'], 'engine_result': {'native_job_id': native_id, 'result': result}})
    entries = [{'relative_path': path.relative_to(directory).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'size_bytes': path.stat().st_size} for path in directory.rglob('*') if path.is_file()]
    write_json(directory / 'receipt.json', {'operation': 'pruning', 'state': 'complete', 'result_path': 'result.json', 'artifacts': entries})
    receipt = directory / 'receipt.json'
    analysis.publish_native_result(actor, captured['job_id'], {'operation': 'pruning', 'snapshot_id': captured['snapshot_id'], 'dataset_id': ds['id'],
        'revision': captured['dataset_revision'], 'native_directory': str(directory), 'receipt_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest(),
        'receipt_size_bytes': receipt.stat().st_size}, 'publish-lab', native_root=tmp_path)
    shutil.rmtree(directory); shutil.rmtree(tmp_path / 'cache')
    facade = HostedPruning(analysis)
    metadata = facade.call(actor, captured['job_id'], 'recipe_lab')
    assert metadata['available'] and metadata['pairs']
    payload = {'selection_revision': 1, 'expected_digest': metadata['base_digest'], 'pairs': metadata['pairs'][:1],
               'variants': [{'id': 'historical', 'alpha': 0., 'cradio_weight': .25}, {'id': 'corrected', 'alpha': 1., 'cradio_weight': .25}]}
    compared = facade.call(actor, captured['job_id'], 'recipe_lab_compare', payload)
    assert compared['encoder_calls'] == compared['head_training_calls'] == compared['geometry_refits'] == compared['review_mutations'] == 0
    assert compared['variants'][0]['rows'][0]['score'] != compared['variants'][1]['rows'][0]['score']
    calibration = {**payload, 'request_id': 'a' * 32, 'variant': payload['variants'][1],
                   'labels': [{'case_id': payload['pairs'][0]['case_id'], 'label': 'useful_variation_keep'}]}
    calibrated = facade.call(actor, captured['job_id'], 'recipe_lab_calibrate', calibration)
    assert calibrated['outcome'] == 'committed' and calibrated['geometry_refits'] == 0
    shutil.rmtree(facade.root)
    restarted = HostedPruning(analysis)
    again = restarted.call(actor, captured['job_id'], 'recipe_lab_compare', payload)
    assert again == compared
    assert restarted.call(actor, captured['job_id'], 'recipe_lab_calibrate', calibration) == calibrated
    import services.dataset_pruning_recipe_lab as lab
    recheck = lab._recheck
    def changed_before_commit(*args):
        recheck(*args)
        current = store.get_dataset(actor, ds['id'])
        store.patch_dataset_settings(actor, ds['id'], {'review_note': 'changed'}, current['revision'], 'change-before-calibration')
    monkeypatch.setattr(lab, '_recheck', changed_before_commit)
    with pytest.raises(HTTPException, match='before calibration commit'):
        restarted.call(actor, captured['job_id'], 'recipe_lab_calibrate', {**calibration, 'request_id': 'b' * 32})
    import sqlite3
    with sqlite3.connect(restarted.root / captured['job_id'] / native_id / 'calibration/recipe-lab.sqlite3') as connection:
        assert connection.execute('SELECT count(*) FROM operations').fetchone()[0] == 1
    assert len(store.get_snapshot(actor, captured['snapshot_id'])['manifest']['images']) == 2
