"""Direct coverage/planner mechanism tests; no API, encoders or model imports."""
import math
import json

import numpy as np
import pytest

from services.dataset_pruning import PruningConfig
from services.dataset_pruning_coverage import CoverageCancelled, CoverageUnavailable, CoverageIndex, _midranks, _cosine_screen_error, prepare_coverage, recompute_exact_rarity


def vector(angle):
    angle = math.radians(angle)
    return [math.cos(angle), math.sin(angle)]


def fixture_index(tmp_path, entries, *, scenes=None, boxes=None, config=None, neighbors_fn=None, progress=None, cancelled=None, source_groups=None, protected=(), scene_dtype=np.float32):
    images, features = [], []
    for i, objects in enumerate(entries):
        rows = []
        for row, (cls, value) in enumerate(objects):
            box = boxes.get((i, row), [.5, .5, .2, .2]) if boxes else [.5, .5, .2, .2]
            rows.append({"class_id": cls, "box": box, "label_row": row, "feature_index": len(features), "geometry_key": str(box)})
            features.append(value)
        images.append({"image_id": f"image-{i}", "image_sha256": f"sha-{i}", "width": 100, "height": 100,
                       "objects": rows, "protection_errors": []})
        if source_groups is not None:
            images[-1]['source_group']=source_groups[i]
        if i in protected:
            images[-1]['protection_errors']=['Fixture source integrity protection']
    objects = np.asarray(features, np.float32).reshape(-1, len(features[0]) if features else 2)
    scene = np.asarray(scenes if scenes is not None else [vector(0)] * len(images), scene_dtype)
    cfg = config or PruningConfig()
    index = prepare_coverage(tmp_path, images, scene, objects, cfg,
                             source_digest="source", recipe_digest="recipe", neighbors_fn=neighbors_fn,
                             progress=progress,cancelled=cancelled)
    return index, images, scene, objects, cfg


def settings(count=100, groups=0, fraction=0., weight=1.):
    return {"policy": "budgeted_coverage", "object_weight": weight, "max_removed_images": count,
            "max_affected_appearance_groups": groups, "max_lost_appearance_weight_fraction": fraction}


def test_mutual_backups_leave_one_real_survivor(tmp_path):
    index, *_ = fixture_index(tmp_path, [[(0, vector(0))], [(0, vector(0))]])
    result = index.plan(settings())
    assert result["removed_count"] == 1
    assert result["metrics"]["affected_appearance_group_count"] == 0
    candidate = result["removed_ids"][0]
    assert result["metrics"]["scene_witnesses"][candidate]["witness_id"] in result["retained_ids"]
    assert result["metrics"]["uncovered_object_count"] == 0
    index.close()


def test_multiple_remaining_frames_cover_different_objects(tmp_path):
    index, *_ = fixture_index(tmp_path, [[(0, vector(0)), (1, vector(90))], [(0, vector(0))], [(1, vector(90))]])
    result = index.plan(settings(), candidate_ids=["image-0"])
    assert result["removed_ids"] == ["image-0"]
    assert result["metrics"]["affected_appearance_group_count"] == 0
    assert {r["witness_image_id"] for r in result["impacts"][0]["objects"]} == {"image-1", "image-2"}
    index.close()


def test_both_binary_caps_and_class_extinction_are_real(tmp_path):
    index, *_ = fixture_index(tmp_path, [[(0, vector(0))], [(1, vector(90))]])
    result = index.plan(settings(groups=1, fraction=1), candidate_ids=[0])
    assert result["removed_indices"] == [0]
    assert result["metrics"]["disappearing_classes"] == [0]
    assert result["metrics"]["affected_appearance_group_count"] == 1
    assert result["metrics"]["lost_appearance_weight_fraction"] == pytest.approx(.5)
    assert index.plan(settings(groups=0, fraction=1), candidate_ids=[0])["removed_count"] == 0
    assert index.plan(settings(groups=1, fraction=.49), candidate_ids=[0])["removed_count"] == 0
    index.close()


def test_representative_coverage_does_not_cover_its_members(tmp_path):
    index, *_ = fixture_index(tmp_path, [[(0, vector(a))] for a in [0, 15, -15, -30]])
    assert len(set(index.group_of[:3])) == 1
    result = index.evaluate([3])
    assert result["affected_appearance_group_count"] == 1
    assert result["uncovered_object_count"] == 1
    details = index.describe("image-1", [3])
    assert details["objects"][0]["covered"] is False
    assert index.describe("image-0", [3])["objects"][0]["covered"] is True
    index.close()


def test_repeated_frames_are_one_counting_unit_not_five_losses(tmp_path):
    index, *_ = fixture_index(tmp_path, [[(0, vector(0))]] * 5 + [[(1, vector(90))]])
    result = index.evaluate([5])
    assert result["affected_appearance_group_count"] == 1
    assert result["uncovered_object_count"] == 5
    assert result["removed_annotation_count"] == 5
    assert result["appearance_group_count"] == 2
    index.close()


def test_native_pixels_remain_a_direct_coverage_condition(tmp_path):
    boxes = {(0, 0): [.5, .5, .4, .4], (1, 0): [.5, .5, .02, .02]}
    index, *_ = fixture_index(tmp_path, [[(0, vector(0))], [(0, vector(0))]], boxes=boxes)
    assert index.evaluate([1])["uncovered_object_count"] == 1
    assert index.plan(settings(), candidate_ids=[0])["removed_count"] == 0
    assert index.plan(settings(groups=1, fraction=1), candidate_ids=[0])["removed_count"] == 1
    index.close()


def test_scene_chain_rechecks_already_removed_images(tmp_path):
    cfg = PruningConfig(scene_threshold=.9)
    index, *_ = fixture_index(tmp_path, [[(0, vector(0))]] * 3,
                             scenes=[vector(0), vector(20), vector(40)], config=cfg)
    result = index.plan(settings(), candidate_ids=[0, 1])
    assert result["removed_indices"] == [0]
    assert result["metrics"]["scene_witnesses"]["image-0"]["witness_id"] == "image-1"
    assert index.evaluate([2])["ready"] is False
    index.close()


def test_similarity_deterioration_does_not_consume_binary_budget(tmp_path):
    index, *_ = fixture_index(tmp_path, [[(0, vector(0))], [(0, vector(20))]])
    result = index.plan(settings(), candidate_ids=[0])
    assert result["removed_count"] == 1
    assert result["metrics"]["affected_appearance_group_count"] == 0
    assert result["metrics"]["lost_appearance_weight_fraction"] == 0
    assert result["curve"][-1]["marginal_ranking_loss"] > 0
    index.close()


def test_partial_restoration_recovers_objects_without_whole_group(tmp_path):
    boxes = {(0, 0): [.5, .5, .2, .2], (1, 0): [.5, .5, .08, .08], (2, 0): [.5, .5, .5, .5]}
    index, *_ = fixture_index(tmp_path, [[(0, vector(0))]] * 3 + [[(1, vector(90))]], boxes=boxes)
    assert len(set(index.group_of[:3])) == 1
    impacts = {row["image_id"]: row for row in index.impacts([3])}
    assert impacts["image-1"]["restored_object_count"] == 2
    assert impacts["image-1"]["restored_appearance_group_count"] == 0
    assert impacts["image-1"]["partially_restored_group_count"] == 1
    assert impacts["image-0"]["restored_object_count"] == 3
    assert impacts["image-0"]["restored_appearance_group_count"] == 1
    index.close()


def test_no_refill_and_explicit_pins(tmp_path):
    index, *_ = fixture_index(tmp_path, [[(0, vector(0))]] * 4)
    assert index.plan(settings(), pinned_ids=[0], candidate_ids=[0, 1])["removed_indices"] == [1]
    assert index.plan(settings(), candidate_ids=[])["removed_count"] == 0
    index.close()


def test_frozen_units_and_plan_binding_survive_cache_warming(tmp_path):
    index, *_ = fixture_index(tmp_path, [[(0, vector(0))], [(0, vector(0))], [(1, vector(90))]])
    groups, weights = index.group_of.copy(), index.weights.copy()
    first = index.plan(settings(), candidate_ids=[0])
    index._exact("object", 2, None)
    second = index.plan(settings(), candidate_ids=[0])
    assert first["plan_digest"] == second["plan_digest"]
    np.testing.assert_array_equal(groups, index.group_of)
    np.testing.assert_array_equal(weights, index.weights)
    index.plan(settings(weight=.25), candidate_ids=[0])
    np.testing.assert_array_equal(weights, index.weights)
    index.close()


def test_exact_certificates_reopen_without_models_or_feature_loading(tmp_path):
    index, images, scene, objects, config = fixture_index(tmp_path, [[(0, vector(0))], [(1, vector(90))]])
    expected = index.evaluate([1])
    index.close()
    reopened = prepare_coverage(tmp_path, images, None, None, config, source_digest="source", recipe_digest="recipe")
    actual = reopened.evaluate([1])
    assert actual == expected
    with pytest.raises(CoverageUnavailable):
        reopened.evaluate([0])
    reopened.close()
    np.testing.assert_array_equal(objects, np.asarray([vector(0), vector(90)], np.float32))


def test_cancel_leaves_no_ready_partial_generation(tmp_path):
    with pytest.raises(CoverageCancelled):
        prepare_coverage(tmp_path, [], np.empty((0, 2)), np.empty((0, 2)), PruningConfig(),
                         source_digest="s", recipe_digest="r", cancelled=lambda: True)
    assert not (tmp_path / "manifest.json").exists()


def test_unknown_is_not_false_absence_and_ready_requires_resolution(tmp_path):
    index, *_ = fixture_index(tmp_path, [[(0, vector(0))], [(1, vector(90))]])
    draft = index.evaluate([1], verify=False)
    assert draft["unknown_object_count"] == 1
    assert draft["affected_appearance_group_count"] == 0
    assert draft["affected_appearance_group_count_upper"] == 1
    assert draft["ready"] is False
    final = index.evaluate([1])
    assert final["unknown_object_count"] == 0
    assert final["affected_appearance_group_count"] == 1
    index.close()


def test_proof_generation_and_saved_generation_reject_source_drift(tmp_path):
    index, images, scene, objects, config = fixture_index(tmp_path, [[(0, vector(0))], [(0, vector(0))]])
    index.close()
    with pytest.raises(ValueError, match="another source"):
        prepare_coverage(tmp_path, images, scene, objects, config, source_digest="changed", recipe_digest="recipe")


def test_verified_negative_scene_needs_negative_keeper(tmp_path):
    index, *_ = fixture_index(tmp_path, [[], [(0, vector(0))], []])
    result = index.plan(settings(groups=10, fraction=1), candidate_ids=[0])
    assert result["metrics"]["scene_witnesses"]["image-0"]["witness_id"] == "image-2"
    assert index.evaluate([1])["ready"] is False
    index.close()


def test_frozen_array_drift_fails_before_evaluation_and_reopen(tmp_path):
    index, images, scene, objects, config = fixture_index(tmp_path, [[(0, vector(0))], [(0, vector(0))]])
    path = tmp_path / 'weights.npy'
    changed = np.load(path).copy()
    changed[0] *= 2
    np.save(path, changed)
    with pytest.raises(ValueError, match='artifact changed'):
        index.evaluate([0])
    index.close()
    with pytest.raises(ValueError, match='artifact changed'):
        prepare_coverage(tmp_path, images, scene, objects, config, source_digest='source', recipe_digest='recipe')


def test_grouping_exact_fallback_is_independent_of_ann_omissions(tmp_path):
    entries = [[(0, vector(a))] for a in [0, 8, 15, 50, 58, 0]]
    exact, *_ = fixture_index(tmp_path / 'exact', entries)
    def empty_neighbors(matrix, k, groups):
        return np.empty((len(matrix), 0), np.int64)
    sparse, *_ = fixture_index(tmp_path / 'sparse', entries, neighbors_fn=empty_neighbors)
    np.testing.assert_array_equal(exact.group_of, sparse.group_of)
    np.testing.assert_array_equal(exact.weights, sparse.weights)
    result = sparse.plan(settings(), candidate_ids=[0])
    assert result['removed_count'] == 1  # Exact expansion finds a retained match.
    exact.close(); sparse.close()


def test_full_source_certificate_covers_restoration_outside_frozen64(tmp_path):
    index, images, scene, objects, config = fixture_index(tmp_path, [[(0, vector(0))]] * 70)
    ranked = set(index.object_rank_images[0].tolist())
    extra = next(i for i in range(1, 70) if i not in ranked)
    result = index.evaluate([extra])
    assert result['uncovered_object_count'] == 0
    assert result['unknown_object_count'] == 0
    proof = index._proof('object', 0)
    assert proof is not None
    assert np.unpackbits(proof, bitorder='little', count=70).all()
    # A different final retainer reuses the same full-source certificate.
    assert index.evaluate([(extra + 1) % 70])['uncovered_object_count'] == 0
    index.close()


def test_progress_and_cancellation_contract_and_source_arrays_unchanged(tmp_path):
    progress = []
    images = [{'image_id':'a','image_sha256':'a','width':100,'height':100,'objects':[
        {'class_id':0,'box':[.5,.5,.2,.2],'label_row':0,'feature_index':0}], 'protection_errors':[]}]
    features = np.asarray([vector(0)],np.float32); before = features.copy()
    index = prepare_coverage(tmp_path, images, features, features, PruningConfig(), source_digest='s',recipe_digest='r',
                             progress=lambda value,message:progress.append((value,message)))
    assert progress and all(0 <= value <= 1 and isinstance(message,str) for value,message in progress)
    with pytest.raises(InterruptedError):
        index.plan(settings(), cancelled=lambda:True)
    np.testing.assert_array_equal(before,features)
    index.close()


def test_same_source_digest_cannot_hide_changed_box_rows(tmp_path):
    index,images,scene,objects,config=fixture_index(tmp_path,[[(0,vector(0))],[(0,vector(0))]])
    index.close();images[0]['objects'][0]['box']=[.5,.5,.3,.3]
    with pytest.raises(ValueError,match='another source'):
        prepare_coverage(tmp_path,images,scene,objects,config,source_digest='source',recipe_digest='recipe')


def test_cancelled_exact_scan_never_publishes_partial_certificate(tmp_path):
    index,*_=fixture_index(tmp_path,[[(0,vector(0))]]*3)
    with pytest.raises(CoverageCancelled):
        index._exact('object',0,lambda:True)
    assert index._proof('object',0) is None
    assert index._exact('object',0,None) is not None
    index.close()


def test_corrupt_compressed_certificate_fails_with_integrity_error(tmp_path):
    index,*_=fixture_index(tmp_path,[[(0,vector(0))]]*2)
    index._exact('object',0,None)
    with index._db:
        index._db.execute("UPDATE proofs SET bits=? WHERE kind='object' AND q=0",(b'invalid zlib',))
    with pytest.raises(ValueError,match='certificate is corrupt'):
        index._proof('object',0)
    index.close()


def test_rarity_is_exact_even_when_ann_returns_five_wrong_neighbors(tmp_path):
    angles=[0,1,2,5,14,25,40,70,100,140,170,220]
    entries=[[(0,vector(a))] for a in angles]
    cfg=PruningConfig(object_threshold=.999999)
    def omit_nearest(matrix,k,groups):
        scores=np.asarray(matrix @ matrix.T)
        return np.asarray([np.argsort(row)[0:5] for row in scores])
    exact,*_=fixture_index(tmp_path/'exact',entries,config=cfg)
    omitted,*_=fixture_index(tmp_path/'omitted',entries,config=cfg,neighbors_fn=omit_nearest)
    assert len(omitted.representatives)==len(angles)
    np.testing.assert_array_equal(exact.isolation,omitted.isolation)
    np.testing.assert_array_equal(exact.weights,omitted.weights)
    rows=np.asarray(omitted.representatives)
    graph=omit_nearest(omitted.objects[rows],5,np.arange(len(rows)))
    actual=omitted._exact_rarity_means(rows,graph,None,None,0)
    original=np.asarray(omitted.objects[rows],dtype=np.float64)
    original/=np.linalg.norm(original,axis=1,keepdims=True)
    distances=np.sqrt(np.maximum(0,2*(1-np.clip(original @ original.T,-1,1))))
    np.fill_diagonal(distances,np.inf)
    expected=np.sort(distances,axis=1)[:,:5].mean(axis=1)
    np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=1e-12)
    exact.close();omitted.close()


def test_cancel_resume_reuses_completed_class_neighbors_and_rankings(tmp_path):
    entries=[[(0,vector(0))]]*3+[[(1,vector(90))]]*3
    calls=[];stop=[False];messages=[]
    def provider(matrix,k,groups):
        calls.append(float(matrix[0,0]))
        return np.asarray([[v for v in range(len(matrix)) if v!=q][:k] for q in range(len(matrix))])
    def progress(value,phase):
        messages.append(phase)
        if phase=='neighbors_class_1':
            stop[0]=True
    with pytest.raises(CoverageCancelled):
        fixture_index(tmp_path,entries,neighbors_fn=provider,progress=progress,cancelled=lambda:stop[0])
    checkpoint=json.loads((tmp_path/'prepare-checkpoint.json').read_text())
    assert checkpoint['classes']['0']['neighbors'] and checkpoint['classes']['0']['ranking']
    assert not (tmp_path/'manifest.json').exists()
    index,*_=fixture_index(tmp_path,entries,neighbors_fn=provider)
    assert len(calls)==2  # First class is reused; second was cancelled before ANN.
    assert 'neighbors_class_0' in messages
    assert index.plan(settings(),candidate_ids=[0])['removed_count']==1
    index.close()


def test_completed_v1_upgrade_preserves_frozen_groups_and_rankings(tmp_path):
    index,images,scene,objects,config=fixture_index(tmp_path,[[(0,vector(a))] for a in [0,30,70]])
    groups=index.group_of.copy();rankings=index.object_rank_images.copy();old_digest=index.accounting_digest
    index.close()
    manifest_path=tmp_path/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['schema']='pruning-direct-appearance-coverage-v1'
    manifest['binding']['schema']=manifest['schema'];manifest.pop('rarity')
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='another source'):
        prepare_coverage(tmp_path,images,scene,objects,config,source_digest='source',recipe_digest='recipe')
    upgraded=recompute_exact_rarity(tmp_path,images,scene,objects,config,source_digest='source',recipe_digest='recipe')
    np.testing.assert_array_equal(groups,upgraded.group_of)
    np.testing.assert_array_equal(rankings,upgraded.object_rank_images)
    assert upgraded.accounting_digest==old_digest  # Same exact computation reproduces its v2 digest.
    assert upgraded.plan(settings(),candidate_ids=[])['ready']
    upgraded.close()


def test_exact_rarity_checks_nearer_targets_hidden_entirely_in_prefix_remainder(tmp_path):
    rng=np.random.default_rng(7)
    features=np.zeros((12,1544),np.float32)
    features[:,1536:]=rng.normal(size=(12,8))
    features/=np.linalg.norm(features,axis=1,keepdims=True)
    def far_neighbors(matrix,k,groups):
        return np.asarray([np.argsort(row)[:5] for row in matrix @ matrix.T])
    index,*_=fixture_index(tmp_path,[[(0,row)] for row in features],
                           config=PruningConfig(object_threshold=.999999),neighbors_fn=far_neighbors)
    rows=np.asarray(index.representatives)
    values=np.asarray(features[rows],np.float64)
    values/=np.linalg.norm(values,axis=1,keepdims=True)
    distances=np.sqrt(np.maximum(0,2*(1-np.clip(values @ values.T,-1,1))))
    np.fill_diagonal(distances,np.inf)
    actual=index._exact_rarity_means(rows,far_neighbors(features[rows],5,np.arange(len(rows))),None,None,0)
    np.testing.assert_allclose(actual,np.sort(distances,axis=1)[:,:5].mean(axis=1),atol=1e-14,rtol=1e-12)
    index.close()


@pytest.mark.parametrize('group_cap,fraction_cap',[(0,1),(100,0)])
def test_first_excess_loss_rejects_without_resolving_remaining_boxes(tmp_path,monkeypatch,group_cap,fraction_cap):
    entries=[[(0,vector(0)),(1,vector(30)),(2,vector(60))],[(3,vector(90))],[(3,vector(90))]]
    index,*_=fixture_index(tmp_path,entries)
    exact=index._exact;calls=[]
    def record(kind,q,cancelled):
        calls.append((kind,q))
        return exact(kind,q,cancelled)
    monkeypatch.setattr(index,'_exact',record)
    config=settings(groups=group_cap,fraction=fraction_cap)
    result=index.plan(config,candidate_ids=[0,1])
    assert result['removed_indices']==[1]
    assert [('object',0)]==[value for value in calls if value[0]=='object']
    assert index._proof('object',0) is not None  # Completed proof survives rejection.
    assert index._proof('object',1) is None and index._proof('object',2) is None
    # Eagerly resolving every rejected box changes neither the selection nor
    # its evidence binding; the optimized trial simply avoids this work.
    exact('object',1,None);exact('object',2,None)
    eager=index.plan(config,candidate_ids=[0,1])
    assert eager==result
    index.close()


def test_zero_floors_use_identical_clipped_metric_with_or_without_ann_edges(tmp_path):
    entries=[[(0,vector(0))],[(0,vector(180))]]
    cfg=PruningConfig(object_threshold=0,scene_threshold=0,negative_scene_threshold=0)
    def omitted(matrix,k,groups):
        return np.empty((len(matrix),0),np.int64)
    normal,*_=fixture_index(tmp_path/'normal',entries,scenes=[vector(0),vector(180)],config=cfg)
    sparse,*_=fixture_index(tmp_path/'sparse',entries,scenes=[vector(0),vector(180)],config=cfg,neighbors_fn=omitted)
    assert normal.evaluate([1])['uncovered_object_count']==0
    assert sparse.evaluate([1])['uncovered_object_count']==0
    assert np.unpackbits(sparse._exact('object',0,None),bitorder='little',count=2).all()
    assert np.unpackbits(sparse._exact('scene',0,None),bitorder='little',count=2).all()
    assert normal.plan(settings(),candidate_ids=[0])['removed_ids']==sparse.plan(settings(),candidate_ids=[0])['removed_ids']
    normal.close();sparse.close()


def test_exact_witness_image_uses_qualifying_crop_when_ann_found_wrong_object(tmp_path):
    def incomplete(matrix,k,groups):
        return np.asarray([[1],[0],[0]],np.int64)
    index,images,scene,objects,config=fixture_index(tmp_path,[[(0,vector(0))],[(0,vector(60)),(0,vector(0))]],neighbors_fn=incomplete)
    assert index.object_rank_scores[0,1]<config.object_threshold
    detail=index.describe('image-0',[1])['objects'][0]
    assert detail['covered'] and detail['witness_image_id']=='image-1'
    assert detail['witness_row']==1 and detail['similarity']==pytest.approx(1.)
    index.close()
    reopened=prepare_coverage(tmp_path,images,None,None,config,source_digest='source',recipe_digest='recipe')
    missing=reopened.describe('image-0',[1])['objects'][0]
    assert missing['covered'] and missing['witness_row'] is None
    assert missing['correspondence_available'] is False
    reopened.close()


def test_v2_migration_invalidates_legacy_signed_zero_floor_certificates(tmp_path):
    def omitted(matrix,k,groups):
        return np.empty((len(matrix),0),np.int64)
    cfg=PruningConfig(object_threshold=0,scene_threshold=0,negative_scene_threshold=0)
    index,images,scene,objects,config=fixture_index(tmp_path,[[(0,vector(0))],[(0,vector(180))]],
                              scenes=[vector(0),vector(180)],config=cfg,neighbors_fn=omitted)
    index._save_proof('object',0,[True,False])
    index._save_proof('scene',0,[True,False])
    with index._db:
        index._db.execute('DROP TABLE certificate_meta')
    index.close()
    path=tmp_path/'manifest.json';manifest=json.loads(path.read_text())
    manifest['schema']='pruning-direct-appearance-coverage-v2'
    manifest['binding']['schema']=manifest['schema'];manifest['binding'].pop('ordering_digest')
    path.write_text(json.dumps(manifest))
    upgraded=recompute_exact_rarity(tmp_path,images,scene,objects,config,source_digest='source',recipe_digest='recipe')
    assert upgraded._proof('object',0) is None and upgraded._proof('scene',0) is None
    assert upgraded.evaluate([1])['uncovered_object_count']==0
    assert np.unpackbits(upgraded._exact('object',0,None),bitorder='little',count=2).all()
    assert np.unpackbits(upgraded._exact('scene',0,None),bitorder='little',count=2).all()
    upgraded.close()


def sequential_group_reference(index):
    """Small scalar oracle: ordered exemplars, no ANN or pruning bounds."""
    result=np.full(index.object_count,-1,np.int32);representatives=[]
    threshold=1-(1-index.config.object_threshold)/4
    for cls,rows in sorted(index.by_class.items()):
        sources={}
        for q in rows:
            image=index.images[index.image_of[q]]
            source=str(image.get('source_group') or image.get('video_id') or image['image_id'])
            sources.setdefault(source,[]).append(int(q))
        for values in sources.values():
            values.sort(key=lambda q:(-np.prod(index.boxes[q,2:]*index.dimensions[index.image_of[q]]),
                                      index.image_ids[index.image_of[q]],index.flat[q]['label_row']))
        order=[]
        for position in range(max(map(len,sources.values()),default=0)):
            order.extend(sources[source][position] for source in sorted(sources) if position<len(sources[source]))
        own=[]
        for q in order:
            found=None
            for group in own:
                ref=representatives[group]
                if index._scale(q,[ref])[0]>=index.config.rarity_scale_threshold and index._cosines(q,[ref])[0]>=threshold:
                    found=group;break
            if found is None:
                found=len(representatives);representatives.append(q);own.append(found)
            result[q]=found
    return result,np.asarray(representatives)


@pytest.mark.parametrize('seed',[7,17])
def test_batched_groups_match_scalar_reference_with_missing_ann_and_native_scales(tmp_path,seed):
    rng=np.random.default_rng(seed);entries=[];boxes={}
    bases=rng.normal(size=(12,8));bases/=np.linalg.norm(bases,axis=1,keepdims=True)
    for image in range(140):
        objects=[]
        for row in range(2):
            values=np.zeros(1544,np.float32);values[:4]=bases[image%12,:4];values[1536:1540]=bases[image%12,4:]
            values[:4]+=rng.normal(scale=.015,size=4);values/=np.linalg.norm(values)
            objects.append((row,values))
            size=[.025,.07,.2,.5][int(rng.integers(0,4))]
            boxes[image,row]=[.5,.5,size,size]
        entries.append(objects)
    def no_edges(matrix,k,groups):
        return np.empty((len(matrix),0),np.int64)
    index,*_=fixture_index(tmp_path,entries,boxes=boxes,neighbors_fn=no_edges,
                           source_groups=[f'video-{i%7}' for i in range(len(entries))])
    groups,representatives=sequential_group_reference(index)
    np.testing.assert_array_equal(groups,index.group_of)
    np.testing.assert_array_equal(representatives,index.representatives)
    index.close()


def test_batched_group_radius_and_scale_boundary_match_scalar_reference(tmp_path):
    cfg=PruningConfig();threshold=1-(1-cfg.object_threshold)/4
    angles=[0,math.degrees(math.acos(threshold+1e-5)),math.degrees(math.acos(threshold-1e-5))]
    entries=[[(0,vector(angle))] for angle in angles]+[[(1,vector(0))]]*3
    boxes={(3,0):[.5,.5,.1,.1]}
    for i,scale in [(4,cfg.rarity_scale_threshold+1e-5),(5,cfg.rarity_scale_threshold-1e-5)]:
        size=.1/(scale*scale);boxes[i,0]=[.5,.5,size,size]
    index,*_=fixture_index(tmp_path,entries,boxes=boxes)
    groups,representatives=sequential_group_reference(index)
    np.testing.assert_array_equal(groups,index.group_of)
    np.testing.assert_array_equal(representatives,index.representatives)
    index.close()


def test_cancelled_group_batch_resumes_from_checkpoint_without_ann_redo(tmp_path):
    entries=[[(0,vector(0))]]*1100;stop=[False];calls=[]
    def no_edges(matrix,k,groups):
        calls.append(len(matrix));return np.empty((len(matrix),0),np.int64)
    def progress(done,total,phase):
        if phase=='groups' and done>=1024:
            stop[0]=True
    with pytest.raises(CoverageCancelled):
        fixture_index(tmp_path,entries,neighbors_fn=no_edges,progress=progress,cancelled=lambda:stop[0])
    saved=json.loads(next(tmp_path.glob('group-checkpoint-*.json')).read_text())
    assert saved['classes']['0']['processed']==1024
    phases=[]
    index,*_=fixture_index(tmp_path,entries,neighbors_fn=no_edges,progress=lambda value,phase:phases.append(phase))
    assert calls==[1100] and 'groups_resumed' in phases
    assert len(index.representatives)==1 and np.all(index.group_of==0)
    index.close()


def rarity_kernel(tmp_path,values):
    tmp_path.mkdir(parents=True,exist_ok=True)
    images=[{'image_id':str(i),'width':100,'height':100,'protection_errors':[],
             'objects':[{'class_id':0,'label_row':0,'feature_index':i,'box':[.5,.5,.2,.2]}]} for i in range(len(values))]
    return CoverageIndex(tmp_path,images,None,values,PruningConfig(),{'source':'test'})


@pytest.mark.parametrize('mode',['repeated','near_ties','overflow'])
def test_certified_fp32_rarity_preserves_fp64_midrank_ties_and_weights(tmp_path,mode):
    rng=np.random.default_rng(91)
    bases=rng.normal(size=(8,7552)).astype(np.float32);bases/=np.linalg.norm(bases,axis=1,keepdims=True)
    values=np.repeat(bases,8,axis=0)
    if mode=='near_ties':
        values[:,-1]+=np.arange(64,dtype=np.float32)*1e-7
    if mode=='overflow':
        values*=np.float32(1e30)
    graph=np.tile(np.arange(8,48,8),(len(values),1))
    index=rarity_kernel(tmp_path,values)
    actual=index._exact_rarity_means(np.arange(len(values)),graph,None,None,0)
    original=np.asarray(values,np.float64);norms=np.linalg.norm(original,axis=1)
    expected=[]
    for q in range(len(values)):
        targets=np.delete(np.arange(len(values)),q)
        scores=np.clip((original[targets] @ original[q])/(norms[targets]*norms[q]),-1,1)
        selected=np.lexsort((targets,-scores))[:5]
        expected.append(np.sqrt(np.maximum(0.,2*(1-scores[selected]))).mean())
    np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=1e-12)
    np.testing.assert_array_equal(_midranks(actual),_midranks(expected))
    np.testing.assert_array_equal(1+_midranks(actual),1+_midranks(expected))
    assert _cosine_screen_error(7552)>9e-4
    index.close()


def test_exact_rarity_query_checkpoint_resumes_completed_blocks(tmp_path):
    cfg=PruningConfig(object_threshold=.999999);stop=[False]
    entries=[[(0,vector(angle))] for angle in range(300)]
    def no_edges(matrix,k,groups):
        return np.empty((len(matrix),0),np.int64)
    def progress(done,total,phase):
        if phase=='exact_rarity_class_0' and done==256:
            stop[0]=True
    with pytest.raises(CoverageCancelled):
        fixture_index(tmp_path,entries,config=cfg,neighbors_fn=no_edges,progress=progress,cancelled=lambda:stop[0])
    checkpoint=json.loads(next(tmp_path.glob('rarity-blocks-*.json')).read_text())
    assert checkpoint['blocks'][0]['start']==0 and checkpoint['blocks'][0]['stop']==256
    phases=[]
    index,*_=fixture_index(tmp_path,entries,config=cfg,neighbors_fn=no_edges,progress=lambda value,phase:phases.append(phase))
    assert 'exact_rarity_class_0_resumed' in phases
    completed=json.loads(next(tmp_path.glob('rarity-blocks-*.json')).read_text())
    assert [(row['start'],row['stop']) for row in completed['blocks']]==[(0,256),(256,300)]
    index.close()


def exhaustive_scene_bits(index,q):
    """Authoritative original scan, including its exact target block shape."""
    covered=np.zeros(index.image_count,bool)
    threshold=index.config.negative_scene_threshold if index.negative[q] else index.config.scene_threshold
    query=np.asarray(index.scene[q],np.float32)
    for start in range(0,index.image_count,256):
        targets=np.arange(start,min(start+256,index.image_count))
        with np.errstate(over='ignore',invalid='ignore'):
            scores=np.clip(np.asarray(index.scene[targets]) @ query,0,1)
        covered[targets]=(scores>=threshold)&index.valid_images[targets]&((not index.negative[q])|index.negative[targets])
    covered[q]=True
    return np.packbits(covered,bitorder='little')


@pytest.mark.parametrize('threshold,negative_threshold',[(0,.94),(.5,.7),(.92,.94),(1,1)])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_scene_gemm_certificates_equal_authoritative_gemv_bits(tmp_path,threshold,negative_threshold,dtype):
    rng=np.random.default_rng(56)
    scenes=rng.normal(size=(270,33)).astype(dtype)
    scenes/=np.linalg.norm(scenes,axis=1,keepdims=True)
    scenes[0]=0;scenes[0,0]=1
    scenes[1]=0;scenes[1,0]=threshold;scenes[1,1]=math.sqrt(max(0,1-threshold*threshold))
    scenes[7]=0;scenes[8]*=3;scenes[9]*=.01
    scenes[10]=np.asarray([1e30,-1e30]+[0]*31,dtype=dtype)
    scenes[11]=np.asarray([1e30,1e30]+[0]*31,dtype=dtype)
    entries=[[] if i%3==0 else [(0,vector(0))] for i in range(len(scenes))]
    cfg=PruningConfig(scene_threshold=threshold,negative_scene_threshold=negative_threshold)
    index,*_=fixture_index(tmp_path,entries,scenes=scenes,scene_dtype=dtype,config=cfg,protected=(12,268))
    for q in range(len(scenes)):
        np.testing.assert_array_equal(index._proof('scene',q),exhaustive_scene_bits(index,q))
    accounting=index.accounting_digest;generation=index.proof_generation
    with index._db:
        index._db.execute("DELETE FROM proofs WHERE kind='scene'")
    warmed=index.prewarm_scene_proofs()
    assert warmed['inserted']==len(scenes) and warmed['computed_queries']==len(scenes)
    for q in range(len(scenes)):
        np.testing.assert_array_equal(index._proof('scene',q),exhaustive_scene_bits(index,q))
    assert index.accounting_digest==accounting and index.proof_generation==generation
    assert index.prewarm_scene_proofs()['computed_queries']==0
    index.close()


def test_scene_prewarm_cancel_resume_and_unavailable_features(tmp_path):
    index,images,scene,objects,config=fixture_index(tmp_path,[[(0,vector(0))]]*70)
    generation=index.proof_generation;stop=[False]
    with index._db:
        index._db.execute("DELETE FROM proofs WHERE kind='scene'")
    def progress(done,total,phase):
        if done>=32:
            stop[0]=True
    with pytest.raises(CoverageCancelled):
        index.prewarm_scene_proofs(cancelled=lambda:stop[0],progress=progress)
    assert index._db.execute("SELECT COUNT(*) FROM proofs WHERE kind='scene'").fetchone()[0]==32
    assert index.validate() and index.proof_generation==generation
    index.close()
    reopened=prepare_coverage(tmp_path,images,None,None,config,source_digest='source',recipe_digest='recipe')
    with pytest.raises(CoverageUnavailable):
        reopened.prewarm_scene_proofs()
    reopened.close()
    restored=prepare_coverage(tmp_path,images,scene,None,config,source_digest='source',recipe_digest='recipe')
    assert restored.prewarm_scene_proofs()['inserted']==38
    restored.close()
    featureless=prepare_coverage(tmp_path,images,None,None,config,source_digest='source',recipe_digest='recipe')
    assert featureless.prewarm_scene_proofs()['already_cached']==70
    featureless.close()


def test_concurrent_scene_prewarm_uses_atomic_equal_sqlite_batches(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    first,images,scene,objects,config=fixture_index(tmp_path,[[(0,vector(0))]]*65)
    second=prepare_coverage(tmp_path,images,scene,objects,config,source_digest='source',recipe_digest='recipe')
    with first._db:
        first._db.execute("DELETE FROM proofs WHERE kind='scene'")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(index.prewarm_scene_proofs) for index in (first,second)]
        results=[future.result() for future in futures]
    assert sum(row['inserted'] for row in results)==65
    for q in range(65):
        np.testing.assert_array_equal(first._proof('scene',q),exhaustive_scene_bits(first,q))
    first.close();second.close()


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_scene_certificates_refine_underflow_and_subnormal_rows(tmp_path,dtype):
    tiny=np.nextafter(dtype(0),dtype(1),dtype=dtype)
    scenes=np.asarray([[1,0],[tiny,tiny],[1e-20,-1e-20],[1e-20,1e-20],[0,0]],dtype=dtype)
    if dtype==np.float64:
        scenes=np.concatenate((scenes,np.asarray([[1e-200,1e-200]],dtype=dtype)))
    index,*_=fixture_index(tmp_path,[[(0,vector(0))]]*len(scenes),scenes=scenes,scene_dtype=dtype,
                           config=PruningConfig(scene_threshold=1e-41,negative_scene_threshold=1e-41))
    for q in range(len(scenes)):
        np.testing.assert_array_equal(index._proof('scene',q),exhaustive_scene_bits(index,q))
    index.close()


@pytest.mark.parametrize('dtype,magnitude',[(np.float32,1e19),(np.float64,1e160)])
def test_scene_overflow_cancellation_refines_even_when_gemm_is_finite_and_mask_is_sparse(tmp_path,dtype,magnitude):
    rng=np.random.default_rng(41)
    # Float64 targets retain the old float32 query cast, so keep the query
    # finite while making target products overflow the promoted accumulator.
    scenes=np.full((270,8),magnitude,dtype=dtype)
    if dtype==np.float64:
        scenes[:32]=np.float64(1e30)
        scenes[32:]=np.float64(1e280)
    for row in range(32,270):
        scenes[row]*=rng.permutation(np.asarray([1]*4+[-1]*4,dtype=dtype))
    index=rarity_kernel(tmp_path,np.asarray([vector(0)]*270,np.float32))
    index.scene=scenes
    index.valid_images[:]=False;index.valid_images[[0,37]]=True
    index.negative[:]=False
    index.config=PruningConfig(scene_threshold=0,negative_scene_threshold=0)
    selected=np.arange(32)
    with np.errstate(over='ignore',invalid='ignore'):
        raw=np.asarray(scenes[:32],np.float32) @ scenes.T
    actual,refined=index._scene_bitmaps(selected,raw,index._scene_norms(None),None)
    assert refined>0
    for q in selected:
        np.testing.assert_array_equal(np.packbits(actual[q],bitorder='little'),exhaustive_scene_bits(index,int(q)))
    index.close()


# These normalized fp32 rows exposed differing shortlist/exact block rounding.
# Retain the real values, and compare against the authoritative calculation;
# a different BLAS may round differently but must still honor its certificate.
SCENE_THRESHOLD_ROWS = np.asarray([[0.09108594059944153,
  -0.1582496166229248,
  0.6376433968544006,
  -0.23035496473312378,
  -0.056670766323804855,
  0.2730228304862976,
  -0.6290979385375977,
  -0.1830042600631714],
 [0.19074292480945587,
  -0.1899397075176239,
  0.44213932752609253,
  -0.3386130630970001,
  -0.3557223975658417,
  0.33040058612823486,
  -0.5704010725021362,
  -0.23734718561172485]], np.float32)
OBJECT_THRESHOLD_ROWS = np.asarray([[-0.011772203259170055,
  -0.39034485816955566,
  -0.023335270583629608,
  0.1302158683538437,
  -0.2594090402126312,
  0.04646527022123337,
  -0.0520436130464077,
  0.07761846482753754,
  0.15504330396652222,
  -0.03564520552754402,
  0.09523996710777283,
  -0.024564368650317192,
  0.050805188715457916,
  0.029541023075580597,
  -0.17457741498947144,
  0.08215029537677765,
  0.21751154959201813,
  0.12850980460643768,
  0.08695682138204575,
  0.4815395176410675,
  -0.09991809725761414,
  -0.18843358755111694,
  0.19963997602462769,
  0.034553952515125275,
  -0.09302136301994324,
  -0.1157028004527092,
  -0.17453734576702118,
  0.03542917221784592,
  0.1160365417599678,
  0.3809443712234497,
  0.14778020977973938,
  -0.011184575967490673,
  0.22809840738773346],
 [0.04859757423400879,
  -0.26072677969932556,
  -0.02173607610166073,
  0.08465564250946045,
  -0.2653379440307617,
  0.11316289007663727,
  -0.07529938220977783,
  -0.16149447858333588,
  0.24430863559246063,
  -0.042348045855760574,
  0.0445350743830204,
  -0.0024203769862651825,
  0.08180274814367294,
  -0.19251371920108795,
  -0.2165004462003708,
  0.0009922310709953308,
  0.25524070858955383,
  -0.12273941189050674,
  0.12179113924503326,
  0.40734678506851196,
  -0.1787346601486206,
  -0.10565381497144699,
  -0.002578943967819214,
  0.022755825892090797,
  -0.14608345925807953,
  -0.1351986676454544,
  -0.26735326647758484,
  0.0899120643734932,
  0.07987295091152191,
  0.3937147855758667,
  0.07636100798845291,
  0.24472631514072418,
  0.04538483917713165]], np.float32)


def test_scene_shortlist_cannot_override_exact_threshold_certificate(tmp_path):
    index, *_ = fixture_index(tmp_path, [[(0, vector(0))]] * 2, scenes=SCENE_THRESHOLD_ROWS)
    expected = bool(np.unpackbits(index._proof('scene', 0), bitorder='little', count=2)[1])
    retained = np.asarray([False, True])
    assert (index._witness('scene', 0, retained, None) == 1) == expected
    evaluation = index.evaluate([1])
    assert evaluation['ready'] == expected
    assert evaluation['scene_missing_ids'] == ([] if expected else ['image-0'])
    assert index.plan(settings(count=1), candidate_ids=[0])['removed_ids'] == (['image-0'] if expected else [])
    index.close()


@pytest.mark.parametrize('warm_full_certificate', [False, True])
def test_object_shortlist_and_final_evaluation_use_canonical_threshold(tmp_path, warm_full_certificate):
    outcomes = []
    for mode in ('seeded', 'empty_ann'):
        neighbors = (lambda matrix, k: np.empty((len(matrix), 0), np.int64)) if mode == 'empty_ann' else None
        index, *_ = fixture_index(tmp_path / mode, [[(0, row)] for row in OBJECT_THRESHOLD_ROWS], neighbors_fn=neighbors)
        expected = bool(index._cosines(0, np.arange(2))[1] >= index.config.object_threshold)
        if warm_full_certificate:
            index._exact('object', 0, None)
        else:
            assert index._proof('object', 0) is None
        evaluation = index.evaluate([1])  # Exercises the former vectorized bypass.
        assert evaluation['uncovered_object_count'] == int(not expected)
        assert evaluation['affected_appearance_group_count'] == int(not expected)
        assert evaluation['lost_appearance_weight_fraction'] == (0. if expected else .5)
        plan = index.plan(settings(count=1), candidate_ids=[0])
        assert plan['removed_ids'] == (['image-0'] if expected else [])
        outcomes.append((evaluation['affected_appearance_group_count'], plan['removed_ids']))
        index.close()
    assert outcomes[0] == outcomes[1]


def test_sparse_object_certificates_reuse_features_and_keep_canonical_shape(tmp_path):
    # A shortlisted target in the second class block must be evaluated with
    # its original validity/scale filter, not with a one-row dot product.
    entries = [[(0, vector(0))]] * 260
    index, images, scene, objects, config = fixture_index(tmp_path, entries, protected=(257,))
    original = index._cosines
    calls = []
    def observe(q, rows):
        calls.append(list(rows))
        return original(q, rows)
    index._cosines = observe
    scores = index._object_row_scores(0, [256, 258])
    assert calls == [[256, 258, 259]]
    assert scores == {256: 1., 258: 1.}
    assert index._proof('object', 0) is None
    generation = index.proof_generation
    index.close()
    reopened = prepare_coverage(tmp_path, images, None, None, config, source_digest='source', recipe_digest='recipe')
    assert reopened._object_row_scores(0, [256, 258], verify=False) == scores
    assert reopened.proof_generation == generation
    assert reopened._object_row_scores(0, [259], verify=False) == {}
    with pytest.raises(CoverageUnavailable):
        reopened._object_row_scores(0, [259])
    reopened.close()
