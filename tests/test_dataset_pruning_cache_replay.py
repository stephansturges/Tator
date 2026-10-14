import numpy as np

from services.dataset_pruning_cache import DescriptorCache, file_state


def test_batch_cache_reads_preserve_order_missing_and_normalization(tmp_path):
    cache = DescriptorCache(tmp_path / 'cache')
    cache.put_many([('first', np.array([3., 4.])), ('second', np.array([0., 9.]))])
    keys = ['second', 'missing', 'first', 'second']
    values = cache.get_many(keys)
    assert len(values) == len(keys)
    assert values[1] is None
    for key, actual in zip(keys, values):
        expected = cache.get(key)
        if expected is not None:
            np.testing.assert_array_equal(actual, expected)
    assert cache.get_many([]) == []


def test_batch_cache_rejects_changed_referenced_producer(tmp_path):
    import json
    path = tmp_path / 'producer.npy'
    np.save(path, np.array([[3., 4.], [0., 7.]], np.float32))
    cache = DescriptorCache(tmp_path / 'cache')
    reference = {'path': str(path), 'state': file_state(path), 'row': 0, 'start': 0, 'end': 2}
    with cache.connect() as db:
        db.execute('INSERT INTO descriptors VALUES (?,?,?,?)', ('reference', 2, None, json.dumps(reference)))
    np.testing.assert_array_equal(cache.get_many(['reference'])[0], cache.get('reference'))
    np.save(path, np.array([[5., 12.], [2., 3.]], np.float32))
    assert cache.get_many(['reference']) == [None]


def test_batch_cache_rejects_invalid_vectors_and_crosses_sql_chunks(tmp_path):
    cache = DescriptorCache(tmp_path / 'cache')
    entries = [(str(index), np.array([index + 1., 1.], np.float32)) for index in range(520)]
    cache.put_many(entries)
    with cache.connect() as db:
        db.execute('INSERT INTO descriptors VALUES (?,?,?,NULL)', ('invalid', 2, np.array([np.nan, 1.], np.float32).tobytes()))
    values = cache.get_many([str(index) for index in reversed(range(520))] + ['invalid'])
    assert len(values) == 521 and values[-1] is None
    np.testing.assert_array_equal(values[0], cache.get('519'))
    np.testing.assert_array_equal(values[-2], cache.get('0'))


def test_replay_still_rejects_missing_descriptors_when_objects_exist(tmp_path):
    from types import SimpleNamespace
    import pytest
    from fastapi import HTTPException
    from services.dataset_pruning import parse_objects
    from services.dataset_pruning_jobs import write_json
    from services.dataset_pruning_planning import replay_objects
    from services.dataset_pruning_serving import ServingContext
    manifest = {'components': {'scene_0': {'dimension': 2, 'weight': 1}}}
    write_json(tmp_path / 'components.json', manifest)
    objects, _ = parse_objects(['0 .5 .5 .2 .2'], 1)
    objects[0]['feature_index'] = 0
    context = ServingContext((tmp_path, {'recipe': {'component_manifest': manifest}},
        {'inventory_digest': 'positive-fixture'}, [{'objects': objects}], None, None, None, None), ())
    with pytest.raises(HTTPException, match='saved object descriptors') as error:
        replay_objects(SimpleNamespace(), context, lambda: False, lambda *_: None)
    assert error.value.status_code == 409
