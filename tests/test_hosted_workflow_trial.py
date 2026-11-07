"""Workflow trial staging against actual media, inventory and hosted import."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageCms, PngImagePlugin
import pytest

from services.hosted.media import HostedMedia
from services.hosted.migration import HostedMigrator
from services.hosted.types import MutationContext
from test_hosted_migration import authority, hashes
from tools.hosted_workflow_trial import stage_trial


def encoded(kind, color):
    pixels = Image.new('RGB', (96, 64), color)
    stream = io.BytesIO()
    if kind == 'xmp':
        metadata = PngImagePlugin.PngInfo()
        metadata.add_itxt('XML:com.adobe.xmp', '<x:xmpmeta xmlns:x="adobe:ns:meta/">retained</x:xmpmeta>')
        pixels.save(stream, format='PNG', pnginfo=metadata)
    elif kind == 'icc':
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        pixels.save(stream, format='JPEG', icc_profile=profile)
    elif kind == 'oriented':
        exif = Image.Exif(); exif[274] = 6
        pixels.save(stream, format='JPEG', exif=exif)
    else:
        pixels.save(stream, format='PNG')
    return stream.getvalue()


def canonical(tmp_path, kind='plain', *, split='train'):
    store, artifacts, admin = authority(tmp_path)
    ds = store.create_dataset(admin, 'Canonical source', ['Car','Building'], 'source-create')
    media = HostedMedia(store, artifacts)
    for index, color in enumerate(('red','green','blue')):
        data = encoded(kind, color); raw = tmp_path / f'source-{index}.bin'; raw.write_bytes(data)
        prepared = media.stage(raw)
        annotations = {'objects': [], 'caption': None, 'captions': None, 'alternate_captions': None, 'text_labels': None}
        if index == 0:
            annotations.update(objects=[{'id':'source-polygon','class_id':ds['classes'][0]['id'], 'bbox':[12,8,36,24],
                'polygon':[[12,8],[48,8],[48,32],[12,32]],'quality':0.8}],
                caption='A red car', captions=['A useful string alternative', {'id':'same-id','caption':'Old candidate','lifecycle_status':'deleted','metadata':{'source_model':'fixture'}}],
                alternate_captions=[{'id':'same-id','text':'Another candidate','lifecycle_status':'superseded','generated_qa_pairs':[{'question':'Color?','answer':'Red'}]}],
                grouping={'flight':'test-flight'}, capture_time='2026-01-01')
        elif index == 2:
            annotations.update(objects=[{'id':'source-box','class_id':ds['classes'][1]['id'],'bbox':[24,16,32,20]}], caption='',
                captions=[{'id':'active-but-not-primary','caption':'Candidate only','is_primary':True}],
                text_labels=[{'id':'text-label','text':'Roadside'}])
        image = store.import_image(admin, ds['id'], prepared['source_sha256'], f'original-{index}.png', annotations,
            'import-source-' + str(index), width=96, height=64, size_bytes=len(data), split=split)
        image = store.acquire_editor(admin, image['id'], 'test')
        context = MutationContext(image['id'],image['current_version'],image['generation'],'test','save-source-' + str(index))
        image = store.save(admin, context, annotations, complete=True, negative=not annotations['objects'])
        store.approve(admin, MutationContext(image['id'],image['current_version'],image['generation'],'test','approve-source-' + str(index)))
    return store, artifacts, admin, ds, store.snapshot(admin, ds['id'])


@pytest.mark.parametrize('kind', ['plain','xmp','icc','oriented'])
def test_trial_roundtrips_original_pixels_geometry_captions_and_independent_ids(tmp_path, kind):
    store, artifacts, admin, ds, manifest = canonical(tmp_path, kind)
    original_artifacts = hashes(artifacts.root)
    destination = tmp_path / 'trial-inputs'; output = tmp_path / 'reviewed-plan.json'
    result = stage_trial(manifest, artifacts.root, destination, output,
                         image_ids=[image['id'] for image in manifest['images']], namespace='test-workflow')
    assert hashes(artifacts.root) == original_artifacts
    plan = json.loads(output.read_text())
    assert not plan['blockers'] and not plan['missing_labels']
    assert result['images'] == 3 and len(result['provenance_only_fields']) == 1
    for row in plan['images']:
        source = next(image for image in manifest['images'] if image['media_hash'] == row['media_hash'])
        assert Path(plan['files'][row['file']]['path']).read_bytes() == (tmp_path / ('source-' + source['filename'].split('-')[1].split('.')[0] + '.bin')).read_bytes()
        assert row['metadata']['workflow_trial']['original_annotations'] == source['annotations']
        assert row['effective_caption'] == (source['annotations'].get('caption') or '')
        assert row['split'] == source['split']
    migrator = HostedMigrator(store, artifacts)
    imported = migrator.apply(admin, plan, name='Workflow trial', mode='incoming')
    assert migrator.apply(admin, plan, name='Workflow trial', mode='incoming') == imported
    assert store.snapshot(admin, ds['id']) == manifest
    trials = store.list_images(admin, imported['dataset_id'])
    assert len(trials) == 3 and all(row['state'] == 'unassigned' and not row['complete'] and not row['negative'] for row in trials)
    assert not {row['id'] for row in trials} & {row['id'] for row in manifest['images']}
    source_objects = {obj['id'] for row in manifest['images'] for obj in row['annotations']['objects']}
    source_classes = {row['id'] for row in manifest['classes']}
    for row in trials:
        value = store.image(admin, row['id']); original = next(i for i in manifest['images'] if i['media_hash'] == row['media_hash'])
        provenance = value['annotations']['migration']['metadata']['workflow_trial']
        assert provenance['source_image_id'] == original['id']
        assert provenance['original_annotations'] == original['annotations']
        for obj in value['annotations']['objects']:
            assert obj['id'] not in source_objects and obj['class_id'] not in source_classes
        records = [c for c in value['annotations']['captions'] if c.get('workflow_trial_source')]
        original_count = sum(len(original['annotations'].get(field) or []) for field in ('captions','alternate_captions','text_labels'))
        assert len(records) == original_count
        assert len({c['id'] for c in records}) == len(records)
        if original['annotations'].get('captions') and isinstance(original['annotations']['captions'][0], str):
            assert any(c['caption'] == 'A useful string alternative' for c in records)
            assert {c['lifecycle_status'] for c in records} >= {'deleted','superseded','active'}
        with artifacts.open(row['media_hash']) as stream:
            assert hashlib.sha256(stream.read()).hexdigest() == original['media_hash']


def test_trial_unsplit_kept_and_mixed_unsplit_rejected(tmp_path):
    _, artifacts, _, _, manifest = canonical(tmp_path, split='unsplit')
    stage_trial(manifest, artifacts.root, tmp_path / 'flat', tmp_path / 'flat.json',
        image_ids=[row['id'] for row in manifest['images']], namespace='unsplit')
    assert {row['split'] for row in json.loads((tmp_path / 'flat.json').read_text())['images']} == {'unsplit'}
    mixed = copy.deepcopy(manifest); mixed['images'][0]['split'] = 'train'
    with pytest.raises(ValueError, match='Stage unsplit images separately'):
        stage_trial(mixed, artifacts.root, tmp_path / 'mixed', tmp_path / 'mixed.json', image_ids=[row['id'] for row in mixed['images']], namespace='mixed')
    assert not (tmp_path / 'mixed').exists()


@pytest.mark.parametrize('path_kind', ['inputs','plan','report_collision'])
def test_trial_rejects_writing_into_source_or_colliding_outputs(tmp_path, path_kind):
    _, artifacts, _, _, manifest = canonical(tmp_path)
    before = hashes(artifacts.root)
    destination = artifacts.root / 'new-inputs' if path_kind == 'inputs' else tmp_path / 'inputs'
    plan = artifacts.root / 'new-plan.json' if path_kind == 'plan' else tmp_path / ('plan.rst' if path_kind == 'report_collision' else 'plan.json')
    with pytest.raises(ValueError):
        stage_trial(manifest, artifacts.root, destination, plan,
                    image_ids=[manifest['images'][0]['id']], namespace='safe')
    assert hashes(artifacts.root) == before and not destination.exists()


def test_trial_rejects_non_roundtrippable_polygon_without_publishing(tmp_path):
    _, artifacts, _, _, manifest = canonical(tmp_path)
    source = next(row for row in manifest['images'] if row['annotations']['objects'] and 'polygon' in row['annotations']['objects'][0])
    source['annotations']['objects'][0]['bbox'][2] += 1
    before = hashes(artifacts.root)
    with pytest.raises(ValueError, match='cannot round-trip'):
        stage_trial(manifest, artifacts.root, tmp_path / 'bad-inputs', tmp_path / 'bad-plan.json', image_ids=[source['id']], namespace='geometry')
    assert not (tmp_path / 'bad-inputs').exists() and not (tmp_path / 'bad-plan.json').exists()
    assert hashes(artifacts.root) == before
