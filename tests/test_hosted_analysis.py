from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from PIL import Image
import pytest

from services.hosted.analysis import CanonicalAnalysis, read_candidate_set, read_canonical_snapshot
from services.hosted.artifacts import ArtifactStore
from services.hosted.auth import AuthService
from services.hosted.errors import DomainError
from services.hosted.store import HostedStore
from services.hosted.types import Actor, MutationContext


def ctx(image, operation, editor="tab"):
    return MutationContext(image["id"], image["current_version"], image["generation"], editor, operation)


@pytest.fixture
def setup(tmp_path):
    store = HostedStore(tmp_path / "hosted.sqlite3")
    auth = AuthService(store)
    row = auth.bootstrap("administrator", "a canonical analysis fixture password")
    actor = Actor(row["id"], "admin", row["username"])
    artifacts = ArtifactStore(tmp_path / "artifacts")
    service = CanonicalAnalysis(store, artifacts, tmp_path / "analysis")
    ds = store.create_dataset(actor, "Canonical analysis", ["Car"], "dataset")
    images = []
    for split, color, approve in (("train", "red", True), ("val", "blue", True), ("train", "green", False)):
        stream = io.BytesIO()
        Image.new("RGB", (40, 30), color).save(stream, format="PNG")
        with artifacts.exclusive():
            ref = artifacts.put_bytes(stream.getvalue())
            image = store.import_image(actor, ds["id"], ref.sha256, color + ".png", {}, "import-" + color, width=40, height=30, size_bytes=ref.size_bytes, split=split)
        if approve:
            image = store.acquire_editor(actor, image["id"], "tab")
            annotations = {"objects": [{"id": "object-" + color, "class_id": ds["classes"][0]["id"], "bbox": [2, 3, 10, 15]}],
                           "capture_metadata": {"sequence_id": "flight-" + color, "image_path": "/must-not-override"},
                           "caption": color + " car"}
            image = store.save(actor, ctx(image, "save-" + color), annotations, complete=True)
            image = store.approve(actor, ctx(image, "approve-" + color))
        images.append(image)
    return store, artifacts, service, actor, ds, images


def test_native_dqe_input_keeps_identity_and_frozen_population(setup):
    store, artifacts, service, actor, ds, images = setup
    job = service.capture(actor, ds["id"], "dqe", {"quality_recipe": "thorough_quality_v1", "workspace_dir": "/untrusted", "source_mode": "linked", "dataset_id": "other"}, "analysis")
    request = service.dqe_request(actor, job["job_id"])
    assert request["source_mode"] == "active_workspace" and request["dataset_id"] == ""
    assert request["quality_recipe"] == "thorough_quality_v1"
    rows = request["workspace_manifest"]["images"]
    assert {row["hosted_image_id"] for row in rows} == {images[0]["id"], images[1]["id"]}
    assert {row["sequence_id"] for row in rows} == {"flight-red", "flight-blue"}
    assert all(row["image_path"] != "/must-not-override" for row in rows)
    assert {row["hosted_objects"][0]["object_id"] for row in rows} == {"object-red", "object-blue"}
    assert {row["hosted_objects"][0]["class_id"] for row in rows} == {ds["classes"][0]["id"]}
    assert service.dqe_request(actor, job["job_id"]) == request
    edit = store.acquire_editor(actor, images[0]["id"], "edit")
    changed = edit["annotations"]
    changed["objects"][0]["bbox"][0] = 12
    store.save(actor, ctx(edit, "edit-newer", "edit"), changed, complete=True)
    assert service.status(actor, job["job_id"])["stale"]
    assert service.dqe_request(actor, job["job_id"]) == request


def test_job_media_keeps_archived_frozen_references_and_rejects_unrelated_images(setup, monkeypatch):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient
    from services.hosted.analysis_api import install_analysis_api
    from services.hosted.operations import HostedOperations
    store, artifacts, service, actor, ds, images = setup
    candidates = service.capture_incoming(actor, ds["id"], [images[2]["id"]], "capture-media-candidates")
    job = service.capture(actor, ds["id"], "ingestion", {"candidate_set_id": candidates["candidate_set_id"]}, "capture-media-job")
    unrelated = service.capture(actor, ds["id"], "statistics", {}, "capture-other-media-job")
    app = FastAPI()
    @app.exception_handler(DomainError)
    async def domain_error(_, error):
        return JSONResponse({"error": error.code}, status_code=error.status)
    install_analysis_api(app, store, artifacts, lambda: actor, lambda: actor, service)
    client = TestClient(app)
    base = f"/api/hosted/analysis/jobs/{job['job_id']}/images"
    original = {image["id"]: client.get(f"{base}/{image['id']}/media").content for image in (images[0], images[2])}
    assert client.get(f"/api/hosted/analysis/jobs/{unrelated['job_id']}/images/{images[2]['id']}/media").status_code == 404
    assert client.get(f"{base}/not-in-this-job/media").status_code == 404
    operations = HostedOperations(store, artifacts)
    for image in (images[0], images[2]):
        leased = store.acquire_editor(actor, image["id"], "archive")
        operations.archive_image(actor, ctx(leased, "archive-" + image["id"], "archive"), "Archive after capture")
        response = client.get(f"{base}/{image['id']}/media")
        assert response.status_code == 200 and response.headers["content-type"] == "image/png"
        assert response.content == original[image["id"]]
    assert service.status(actor, job["job_id"])["stale"]
    # Completed pre-upgrade jobs have no request media maps. Both canonical
    # and candidate images retain their own v1 view; unrelated sources are not
    # resolved while serving a single historical image.
    from services.hosted.media import HostedMedia, LEGACY_MEDIA_VIEW_POLICY
    media = HostedMedia(store, artifacts)
    for image in (images[0], images[2]):
        view = media.resolve(image['media_hash']); view['policy'] = LEGACY_MEDIA_VIEW_POLICY
        with artifacts.open(media.reference(view)) as stream:
            media._retain(stream.read(), view)
    with store.transaction() as con:
        request = json.loads(con.execute('SELECT request_json FROM hosted_analysis_jobs WHERE id=?', (job['job_id'],)).fetchone()[0])
        for field in ('working_media', 'candidate_working_media', 'media_policy'):
            request.pop(field, None)
        con.execute("UPDATE hosted_analysis_jobs SET request_json=?,state='complete' WHERE id=?", (json.dumps(request), job['job_id']))
    resolve = HostedMedia.resolve
    def only_requested(instance, source, **options):
        assert getattr(source, 'sha256', source) != images[1]['media_hash']
        return resolve(instance, source, **options)
    monkeypatch.setattr(HostedMedia, 'resolve', only_requested)
    for image in (images[0], images[2]):
        response = client.get(f"{base}/{image['id']}/media")
        assert response.status_code == 200 and response.content == original[image['id']]
    with pytest.raises(DomainError, match="admin_required"):
        with store.transaction() as con:
            con.execute("UPDATE hosted_users SET role='annotator' WHERE id=?", (actor.user_id,))
        service.image_reference(Actor(actor.user_id, "annotator", actor.username), job["job_id"], images[0]["id"])


@pytest.mark.parametrize('registered_legacy', [False, True])
def test_pending_legacy_analysis_requires_known_compatible_policy(setup, registered_legacy):
    from services.hosted.media import HostedMedia, LEGACY_MEDIA_VIEW_POLICY
    store, artifacts, service, actor, ds, images = setup
    job = service.capture(actor, ds['id'], 'statistics', {}, 'legacy-policy-job')
    if registered_legacy:
        media = HostedMedia(store, artifacts)
        for image in images[:2]:
            view = media.resolve(image['media_hash']); view['policy'] = LEGACY_MEDIA_VIEW_POLICY
            with artifacts.open(media.reference(view)) as stream:
                media._retain(stream.read(), view)
    with store.transaction() as con:
        request = json.loads(con.execute('SELECT request_json FROM hosted_analysis_jobs WHERE id=?', (job['job_id'],)).fetchone()[0])
        for field in ('working_media', 'candidate_working_media', 'media_policy'):
            request.pop(field, None)
        con.execute('UPDATE hosted_analysis_jobs SET request_json=? WHERE id=?', (json.dumps(request), job['job_id']))
    if registered_legacy:
        prepared = service.prepare(actor, job['job_id'])
        mapping = json.loads((Path(prepared['dataset_root']) / 'hosted-media.json').read_text())
        assert mapping['policy'] == LEGACY_MEDIA_VIEW_POLICY
    else:
        with pytest.raises(DomainError, match='media_policy_upgrade_required'):
            service.prepare(actor, job['job_id'])
        assert service.status(actor, job['job_id'])['state'] == 'interrupted'


def test_existing_yolo_dataset_resolver_uses_only_exact_snapshot(setup):
    _, _, service, actor, ds, images = setup
    job = service.capture(actor, ds["id"], "yolo_training", {"epochs": 2}, "training")
    resolved = service.resolve_yolo_input(actor, job["job_id"])
    assert resolved["yolo_ready"] and resolved["yolo_layout"] == "split"
    assert set(resolved["identity_map"]) == {images[0]["id"], images[1]["id"]}
    assert resolved["signature"] == resolved["hosted_provenance"]["snapshot_sha256"]
    # A subsequent consumer may not ingest altered labels from a reused view.
    from pathlib import Path
    label = next(Path(resolved["yolo_labels_dir"]).glob("*.txt"))
    label.chmod(0o600)
    label.write_text("0 0.2 0.2 0.3 0.3\n")
    with pytest.raises(DomainError, match="analysis_materialization_changed"):
        service.resolve_yolo_input(actor, job["job_id"])


def test_result_application_checks_population_lease_and_retry_inside_transaction(setup):
    store, artifacts, service, actor, ds, images = setup
    job = service.capture(actor, ds["id"], "dqe", {"quality_recipe": "thorough_quality_v1"}, "analysis")
    fit = artifacts.put_bytes(b"frozen actual-engine-fit fixture")
    published = service.publish_result(actor, job["job_id"], {"recipe": {"unchanged": True}, "fit_population_snapshot": job["snapshot_id"]}, "publish", artifacts=[fit.as_dict()])
    assert service.result_reference(actor, job["job_id"]).sha256 == published["result"]["sha256"]
    edit = store.acquire_editor(actor, images[0]["id"], "tab")
    changed = edit["annotations"]
    changed["objects"][0]["bbox"][0] = 5
    with pytest.raises(DomainError, match="editor_lease_conflict"):
        service.apply_annotations(actor, job["job_id"], ctx(edit, "bad-tab", "wrong-tab"), changed)
    request = ctx(edit, "apply-correction")
    accepted = service.apply_annotations(actor, job["job_id"], request, changed)
    assert store.image(actor, accepted["id"])["annotations"]["objects"][0]["bbox"][0] == 5
    assert service.apply_annotations(actor, job["job_id"], request, changed) == accepted
    other = store.acquire_editor(actor, images[1]["id"], "tab")
    with pytest.raises(DomainError, match="analysis_snapshot_stale"):
        service.apply_annotations(actor, job["job_id"], ctx(other, "old-population"), other["annotations"])
    reopened = CanonicalAnalysis(HostedStore(store.path), ArtifactStore(artifacts.root), service.root)
    assert reopened.status(actor, job["job_id"])["stale"]
    assert reopened.result_reference(actor, job["job_id"]) == service.result_reference(actor, job["job_id"])


def test_heavy_manifest_transport_binds_paths_hashes_and_snapshot_identity(setup, tmp_path):
    store, artifacts, _, actor, ds, _ = setup
    frozen = store.create_snapshot(actor, ds["id"], "heavy-snapshot")
    root = tmp_path / "heavy"
    directory = root / "private-job"
    directory.mkdir(parents=True)
    manifest = frozen["manifest"] | {"snapshot_id": frozen["snapshot_id"]}
    with artifacts.exclusive():
        for image in manifest["images"]:
            ref = artifacts.pin(image["media_hash"], directory)
            image["media_path"] = str(directory / ref.sha256)
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    reference = {key: manifest[key] for key in ("snapshot_id", "dataset_id", "revision", "class_version", "settings_version")}
    reference.update(manifest_path=str(path), manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert read_canonical_snapshot(reference, allowed_root=root) == manifest
    with pytest.raises(DomainError, match="analysis_snapshot_identity_changed"):
        read_canonical_snapshot(reference | {"revision": 999}, allowed_root=root)
    outside = root / "another-job"
    outside.mkdir()
    manifest["images"][0]["media_path"] = str(outside / "elsewhere")
    (outside / "elsewhere").write_bytes(b"other user's bytes")
    path.write_text(json.dumps(manifest))
    reference["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(DomainError, match="analysis_path_invalid"):
        read_canonical_snapshot(reference, allowed_root=root)


def test_native_statistics_child_uses_frozen_input_and_copies_cache_safely(setup, tmp_path, monkeypatch):
    from pathlib import Path
    from services.hosted.native_analysis import NativeAnalysisAdapter, native_environment
    store, artifacts, service, actor, ds, images = setup
    frozen = store.create_snapshot(actor, ds["id"], "native-snapshot")
    root = tmp_path / "heavy"
    directory = root / "private-job"
    directory.mkdir(parents=True)
    manifest = frozen["manifest"] | {"snapshot_id": frozen["snapshot_id"]}
    with artifacts.exclusive():
        for image in manifest["images"]:
            ref = artifacts.pin(image["media_hash"], directory)
            image["media_path"] = str(directory / ref.sha256)
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    reference = {key: manifest[key] for key in ("snapshot_id", "dataset_id", "revision", "class_version", "settings_version")}
    reference.update(manifest_path=str(path), manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    original = tmp_path / "existing-cache"
    original.mkdir()
    cache = original / "descriptor.bin"
    cache.write_bytes(b"immutable cached feature fixture")
    config = {"canonical_root": str(root), "native_root": str(tmp_path / "native"), "cache_input_roots": [str(original)],
              "cache_inputs": [{"namespace": "dqe_cache", "relative_path": "descriptor.bin", "path": str(cache),
                                "sha256": hashlib.sha256(cache.read_bytes()).hexdigest(), "size_bytes": cache.stat().st_size}]}
    config['cache_inputs'].append({**config['cache_inputs'][0], 'relative_path': 'same-content-alias.bin'})
    import numpy as np
    from services.dataset_pruning_cache import DescriptorCache
    matrix = original / "embeddings.npy"
    np.save(matrix, np.array([[3., 4.]], dtype=np.float32))
    original_index = DescriptorCache(original / "pruning")
    original_index.put_references([("cached-pixel-and-geometry", 0)], matrix, 2)
    for file, namespace, relative in ((matrix, "dqe_cache", "embeddings.npy"), (original_index.path, "pruning_cache", "descriptors.sqlite3")):
        config["cache_inputs"].append({"namespace": namespace, "relative_path": relative, "path": str(file),
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "size_bytes": file.stat().st_size})
    original_database = original_index.path.read_bytes()
    monkeypatch.setenv("DATASET_ROOT", "/original-must-never-be-used")
    monkeypatch.setenv("PRIVATE_PROVIDER_SECRET", "must-not-reach-child")
    assert "PRIVATE_PROVIDER_SECRET" not in native_environment(tmp_path, "cpu")
    adapter = NativeAnalysisAdapter(config=config, device="cpu")
    context = {"dataset_id": ds["id"], "annotation_version": frozen["snapshot_id"]}
    with pytest.raises(DomainError, match="native_snapshot_context_changed"):
        adapter.execute("statistics", {"canonical_snapshot": reference}, context | {"annotation_version": "other"})
    with pytest.raises(DomainError, match="native_parameters_not_allowed"):
        adapter.execute("statistics", {"canonical_snapshot": reference, "dataset_root": "/other"}, context)
    result = adapter.execute("statistics", {"canonical_snapshot": reference}, context)
    output = Path(result["native_directory"])
    receipt = json.loads((output / "receipt.json").read_text())
    assert hashlib.sha256((output / "receipt.json").read_bytes()).hexdigest() == result["receipt_sha256"]
    measured = json.loads((output / receipt["result_path"]).read_text())
    assert measured["engine_result"]["image_count"] == 2
    assert measured["engine_result"]["object_counts"] == {"Car": 2}
    assert measured["engine_result"]["splits"] == {"train": 1, "val": 1, "test": 0, "unsplit": 0}
    assert measured["cache_rebase"] == {"rebound_rows": 1, "discarded_unavailable_rows": 0}
    assert original_index.path.read_bytes() == original_database
    assert {image["hosted_image_id"] for image in measured["source"]["images"]} == {images[0]["id"], images[1]["id"]}
    copied = output / "class_analysis/cache/descriptor.bin"
    assert copied.read_bytes() == cache.read_bytes() and copied.stat().st_ino != cache.stat().st_ino
    assert all(item["relative_path"] != "native.log" for item in receipt["artifacts"])
    captured = service.capture(actor, ds["id"], "statistics", {}, "native-analysis", snapshot_id=frozen["snapshot_id"])
    published = service.publish_native_result(actor, captured["job_id"], result, "native-publish", native_root=tmp_path / "native")
    # DescriptorCache opening creates/checkpoints its private SQLite state;
    # publication therefore precedes this deliberate post-run cache consumer.
    reused = DescriptorCache(output / "class_analysis/pruning/cache")
    assert reused.get("cached-pixel-and-geometry") == pytest.approx([.6, .8])
    with artifacts.open(published["result"]["sha256"]) as stream:
        retained = json.load(stream)
    assert retained["result"]["engine_result"]["image_count"] == 2
    assert any(item["relative_path"] == "class_analysis/cache/descriptor.bin" for item in retained["artifacts"])
    import shutil
    shutil.rmtree(output)
    assert service.publish_native_result(actor, captured["job_id"], result, "native-publish", native_root=tmp_path / "native") == published
    assert cache.read_bytes() == b"immutable cached feature fixture"


def test_native_caption_export_uses_frozen_records_and_original_readiness(setup, tmp_path):
    from pathlib import Path
    import zipfile
    from services.hosted.native_analysis import NativeAnalysisAdapter
    store, artifacts, service, actor, ds, images = setup
    for split, color in [('test', 'yellow'), ('unsplit', 'purple')]:
        stream = io.BytesIO()
        Image.new('RGB', (40, 30), color).save(stream, format='PNG')
        with artifacts.exclusive():
            ref = artifacts.put_bytes(stream.getvalue())
            image = store.import_image(actor, ds['id'], ref.sha256, color + '.png', {}, 'import-' + color, width=40, height=30, size_bytes=ref.size_bytes, split=split)
        image = store.acquire_editor(actor, image['id'], 'tab')
        annotations = {'objects': [{'id': 'object-' + color, 'class_id': ds['classes'][0]['id'], 'bbox': [2, 3, 10, 15]}],
                       'caption': color + ' car', 'alternate_captions': [{'caption': 'A car.', 'metadata': None, 'source': 'hosted_qwen',
                       'generated_qa_pairs': [{'question': 'What is present?', 'answer': 'A car.', 'validation_status': 'machine_validated', 'review_status': 'unreviewed'}]},
                       {'caption': 'A car.', 'source': 'hosted_paid_batch', 'model_id': 'test-provenance', 'quality_failures': ['retained fixture warning']},
                       {'caption': 'Deleted caption.', 'lifecycle_status': 'deleted',
                        'generated_qa_pairs': [{'question': 'Deleted question?', 'answer': 'Old answer.', 'validation_status': 'machine_validated'}]},
                       {'caption': 'Superseded caption.', 'lifecycle_status': 'superseded',
                        'generated_qa_pairs': [{'question': 'Superseded question?', 'answer': 'Old answer.', 'validation_status': 'machine_validated'}]}]}
        image = store.save(actor, ctx(image, 'save-' + color), annotations, complete=True)
        store.approve(actor, ctx(image, 'approve-' + color))
    frozen = store.create_snapshot(actor, ds['id'], 'caption-export-snapshot')
    directory = tmp_path / 'caption-input'
    directory.mkdir()
    manifest = frozen['manifest'] | {'snapshot_id': frozen['snapshot_id']}
    with artifacts.exclusive():
        for image in manifest['images']:
            ref = artifacts.pin(image['media_hash'], directory)
            image['media_path'] = str(directory / ref.sha256)
    path = directory / 'manifest.json'
    path.write_text(json.dumps(manifest))
    reference = {key: manifest[key] for key in ('snapshot_id', 'dataset_id', 'revision', 'class_version', 'settings_version')}
    reference.update(manifest_path=str(path), manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    config = {'canonical_root': str(directory), 'native_root': str(tmp_path / 'native'),
              'allowed_parameters': ['require_ready_instruction_export']}
    adapter = NativeAnalysisAdapter(config=config, device='cpu')
    context = {'dataset_id': ds['id'], 'annotation_version': frozen['snapshot_id']}
    first = adapter.execute('caption_export', {'canonical_snapshot': reference}, context)
    first_dir = Path(first['native_directory'])
    receipt = json.loads((first_dir / 'receipt.json').read_text())
    measured = json.loads((first_dir / receipt['result_path']).read_text())['engine_result']
    assert measured['image_count'] == 4
    assert measured['blocked_reason'] and not measured['bundle_path']
    # An explicit diagnostic export retains rejected rows/readiness; it never
    # manufactures ready status just to produce a downloadable archive.
    diagnostic = adapter.execute('caption_export', {'canonical_snapshot': reference, 'require_ready_instruction_export': False}, context)
    output = Path(diagnostic['native_directory'])
    receipt = json.loads((output / 'receipt.json').read_text())
    result = json.loads((output / receipt['result_path']).read_text())['engine_result']
    assert result['diagnostic_export'] and result['blocked_reason'] == measured['blocked_reason']
    exported = json.loads((output / result['export_path']).read_text())
    assert {row['caption'] for row in exported['captions']} == {'red car', 'blue car', 'yellow car', 'purple car', 'A car.', 'Deleted caption.', 'Superseded caption.'}
    inactive = [row for row in exported['captions'] if row['caption'] in {'Deleted caption.', 'Superseded caption.'}]
    assert all(row['lifecycle_status'] == 'superseded' and not row['is_primary'] for row in inactive)
    assert {row['metadata']['hosted_source']['lifecycle_status'] for row in inactive} == {'deleted', 'superseded'}
    assert {row['split'] for row in exported['captions']} == {'train', 'val', 'test', 'unsplit'}
    assert exported['instruction_summary']['generated_qa_pair_count'] == 2
    generated = [row for row in exported['captions'] if row['caption'] == 'A car.']
    assert len(generated) == 4
    assert all(row['source'].startswith('qwen_caption_hosted_') for row in generated)
    assert {row['metadata']['hosted_source']['source'] for row in generated} == {'hosted_qwen', 'hosted_paid_batch'}
    assert all('caption_quality' not in row['metadata'] for row in generated)
    assert all(row.get('review_status') != 'accepted' for row in generated)
    with zipfile.ZipFile(output / result['bundle_path']) as bundle:
        assert len([name for name in bundle.namelist() if name.endswith('.png')]) == 4
    captured = service.capture(actor, ds['id'], 'caption_export', {}, 'export-job', snapshot_id=frozen['snapshot_id'])
    service.publish_native_result(actor, captured['job_id'], diagnostic, 'export-publish', native_root=tmp_path / 'native')
    from services.hosted.analysis_views import AnalysisViews
    views = AnalysisViews(service)
    with artifacts.open(views._reference(actor, captured['job_id'], result['bundle_path'])) as stream:
        assert stream.read(2) == b'PK'
    with pytest.raises(DomainError, match='analysis_artifact_unavailable'):
        views._reference(actor, captured['job_id'], '../private/other.zip')


def test_native_dqe_read_views_reuse_session_graph_and_canonical_crops(setup, tmp_path):
    import shutil
    from services.class_analysis_session_store import build_class_analysis_session_store
    from services.hosted.analysis_views import AnalysisViews
    store, artifacts, service, actor, ds, images = setup
    bus = store.add_class(actor, ds['id'], 'Bus', 'review-bus-class', ds['class_version'])['class']['id']
    job = service.capture(actor, ds["id"], "dqe", {}, "view-analysis")
    directory = tmp_path / "native/view-job"
    native_job = "ca_fixture"
    session = directory / f"class_analysis/{native_job}/session.sqlite3"
    session.parent.mkdir(parents=True)
    points = [{"point_id": "point-" + str(index), "class_id": "0", "class_name": "Car",
               "image_relpath": image["id"] + ".png", "label_line_index": 0,
               "bbox_xyxy": [2, 3, 12, 18], "projection": [float(index), float(-index)],
               "identity_status": "ready", "annotation_entity_id": "object-" + str(index), "annotation_entity_revision": 1,
               "annotation_entity_record_revision": image["current_version"], "annotation_source_identity": "hosted:" + ds["id"],
               "source_record_key": image["split"] + ":" + image["id"] + ".png", "annotation_attestation": "fixture-" + str(index),
               "quality_review_candidate": True, "review_priority_score": 1 - index / 10}
              for index, image in enumerate(images[:2])]
    build_class_analysis_session_store(session, points, summary={"projection_mode": "umap"},
                                      request={"projection_mode": "umap"}, expected_point_count=2)
    measured = {"snapshot_id": job["snapshot_id"], "dataset_id": ds["id"], "revision": job["dataset_revision"],
                "engine_result": {"native_job_id": native_job, "result": {"summary": {"object_count": 2}, "points": points}}}
    (directory / "result.json").write_text(json.dumps(measured))
    config_path = session.parent / 'config.json'
    config_path.write_text(json.dumps({'projection_mode': 'umap'}))
    entries = [{"relative_path": path.relative_to(directory).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
               for path in (session, config_path, directory / "result.json")]
    receipt = directory / "receipt.json"
    receipt.write_text(json.dumps({"operation": "dqe", "state": "complete", "result_path": "result.json", "artifacts": entries}))
    output = {"operation": "dqe", "snapshot_id": job["snapshot_id"], "dataset_id": ds["id"], "revision": job["dataset_revision"],
              "native_directory": str(directory), "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(), "receipt_size_bytes": receipt.stat().st_size}
    service.publish_native_result(actor, job["job_id"], output, "view-publish", native_root=directory.parent)
    shutil.rmtree(directory)
    pinned = service.prepare_review_parent(actor, job['job_id'], tmp_path / 'review-parent-pins')
    assert pinned['snapshot_id'] == job['snapshot_id'] and pinned['parent_analysis']['native_job_id'] == native_job
    assert {item['relative_path'] for item in pinned['cache_inputs']} == {'session.sqlite3', 'config.json'}
    for item in pinned['cache_inputs']:
        from pathlib import Path
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
    views = AnalysisViews(service)
    assert views.metadata(actor, job["job_id"])["result"] == {"summary": {"object_count": 2}}
    graph = views.graph(actor, job["job_id"])
    assert graph["point_count"] == 2 and graph["columns"]["x"] == [0., 1.]
    detail = views.point(actor, job["job_id"], "point-0")
    assert detail["hosted_identity"]["object_id"] == "object-red"
    assert detail["hosted_identity"]["version_id"] == images[0]["current_version"]
    assert views.queue(actor, job["job_id"])["total"] == 2
    thumb = views.thumbnail(actor, job["job_id"], "point-0")
    with Image.open(io.BytesIO(thumb)) as image:
        assert image.size == (10, 15) and image.getpixel((0, 0))[0] > 240
    from services.hosted.dqe import DqeReview
    from dataclasses import asdict
    review = DqeReview(service)
    kept = review.disposition(actor, job['job_id'], 'point-0', disposition='keep', expected_revision=0, operation_id='dqe-keep')
    assert kept['review_revision'] == 1
    views.session_provider = review.session
    assert views.graph(actor, job['job_id'], reviewed='reviewed')['returned'] == 1
    items = []
    for index, original in enumerate(images[:2]):
        editor = store.acquire_editor(actor, original['id'], 'tab')
        items.append({'context': asdict(ctx(editor, 'batch-item-' + str(index))),
            'actions': [{'point_id': 'point-' + str(index), 'action': 'bbox', 'bbox': [3 + index, 3, 10, 15]}]})
    batch = review.create_batch(actor, job['job_id'], items, 'dqe-batch')
    partial = review.process_batch(actor, job['job_id'], batch['batch_id'], limit=1)
    assert len(partial['results']) == 1 and partial['state'] == 'running'
    # Simulate shutdown after the image receipt commits but before progress is
    # stored, then let the old browser leases expire. Resume must recover that
    # receipt and rebind only the still-pending image to freshly acquired leases.
    with store.transaction() as con:
        con.execute('UPDATE hosted_dqe_batches SET results_json=? WHERE id=?', ('{}', batch['batch_id']))
    old_clock = store.clock
    store.clock = lambda: old_clock() + 1000
    refreshed = [asdict(ctx(store.acquire_editor(actor, image['id'], 'tab'), 'new-tab-context')) for image in images[:2]]
    rebound = review.resume_batch(actor, job['job_id'], batch['batch_id'], refreshed, 'resume-expired-leases')
    assert rebound['completed_count'] == 1
    assert review.batches(actor, job['job_id'])['batches'][0]['batch_id'] == batch['batch_id']
    resumed = DqeReview(service).process_batch(actor, job['job_id'], batch['batch_id'])
    assert resumed['state'] == 'complete' and all(item['state'] == 'complete' for item in resumed['results'].values())
    assert review.process_batch(actor, job['job_id'], batch['batch_id']) == resumed
    assert store.image(actor, images[1]['id'])['annotations']['objects'][0]['bbox'][0] == 4
    assert views.graph(actor, job['job_id'], reviewed='reviewed')['returned'] == 2
    # A second correction on the same image uses this review's owned version,
    # without overwriting an unrelated edit or returning to frozen annotations.
    current = store.acquire_editor(actor, images[0]['id'], 'tab')
    relabel = review.apply(actor, job['job_id'], ctx(current, 'relabel-after-bbox'),
                          [{'point_id': 'point-0', 'action': 'relabel', 'target_class_id': bus}])
    relabeled_image = store.image(actor, images[0]['id'])
    assert relabeled_image['annotations']['objects'][0]['class_id'] == bus
    assert relabeled_image['annotations']['objects'][0]['bbox'][0] == 3
    assert review.point(actor, job['job_id'], 'point-0')['hosted_identity']['version_id'] == relabel['image']['current_version']
    delete_context = ctx(relabel['image'], 'last-object-delete')
    with pytest.raises(DomainError, match='dqe_empty_image_requires_negative_confirmation'):
        review.apply(actor, job['job_id'], delete_context, [{'point_id': 'point-0', 'action': 'delete'}])
    deleted = review.apply(actor, job['job_id'], delete_context, [{'point_id': 'point-0', 'action': 'delete'}], negative=True)
    deleted_image = store.image(actor, images[0]['id'])
    assert deleted_image['negative'] and deleted_image['annotations']['objects'] == []
    assert review.apply(actor, job['job_id'], delete_context, [{'point_id': 'point-0', 'action': 'delete'}], negative=True) == deleted
    # Earlier corrections advance the canonical population, but remain tied to
    # their exact receipts. Deleting the last object needs explicit intent.
    newer = service.capture(actor, ds['id'], 'dqe', {}, 'next-dqe-capture')
    assert newer['dataset_revision'] == job['dataset_revision'] + 4
    with pytest.raises(DomainError, match='analysis_snapshot_stale'):
        service.prepare_review_parent(actor, job['job_id'], tmp_path / 'stale-review-pins')
    edit = store.acquire_editor(actor, images[1]["id"], "tab")
    store.save(actor, ctx(edit, "view-newer"), edit["annotations"], complete=True)
    assert views.graph(actor, job["job_id"])["stale"]


def test_native_vlm_review_reuses_parent_and_preserves_full_response(setup, tmp_path, monkeypatch):
    """Adapter contract: model execution is doubled; native VLM requires GPU acceptance."""
    import sys
    from types import SimpleNamespace
    from services.hosted.native_analysis import _materialize, _run_native
    store, _, service, actor, ds, _ = setup
    captured = store.create_snapshot(actor, ds['id'], 'review-frozen')
    manifest = captured['manifest'] | {'snapshot_id': captured['snapshot_id']}
    pins = tmp_path / 'native-review-media'; pins.mkdir()
    for image in manifest['images']:
        ref = service.artifacts.pin(image['media_hash'], pins)
        image['media_path'] = str(pins / ref.sha256)
    directory = tmp_path / 'native-review'; directory.mkdir()
    (directory / 'registry').mkdir()
    dataset, source, _ = _materialize(manifest, directory)
    parent = directory / 'class_analysis/review_parent'; parent.mkdir()
    (parent / 'config.json').write_text(json.dumps({'source_mode': 'active_workspace', 'workspace_dir': '/stale/unavailable/source'}))
    calls = []
    response = {'review_id': 'cqr_fixture', 'status': 'completed', 'result': {'decision': 'keep', 'reason': 'Visible class cue'},
                'trace': [{'model_input': 'actual payload location', 'raw_output': 'preserved response'}]}
    def create(job_id, point_id, payload):
        calls.append((job_id, point_id, payload))
        saved = json.loads((directory / 'class_analysis/ca_frozen/config.json').read_text())
        assert saved['workspace_dir'] == str(dataset) and saved['workspace_manifest']['snapshot_id'] == manifest['snapshot_id']
        return {'review_id': 'cqr_fixture'}
    from services.annotation_entity_store import AnnotationEntityStore
    entity_store = AnnotationEntityStore(directory / 'annotation_entities.sqlite3')
    monkeypatch.setitem(sys.modules, 'localinferenceapi', SimpleNamespace(create_class_analysis_qwen_review=create, get_class_analysis_qwen_review=lambda _: response,
        _class_analysis_annotation_entity_store=lambda: entity_store,
        _class_analysis_annotation_image_key=lambda row: f"{row['split']}:{row['image_relpath']}"))
    result = _run_native({'operation': 'dqe_review', 'parameters': {'point_id': 'point-0', 'max_turns': 4, 'model_id': '/untrusted'},
                         'parent_analysis': {'job_id': 'hosted-parent', 'native_job_id': 'ca_frozen'},
                         'review_model_directory': 'models/review', 'timeout': 1}, manifest, dataset, source, directory)
    assert calls[0][2]['model_id'] == str(directory / 'models/review') and calls[0][2]['keep_vlm_loaded'] is False
    assert result['review'] == response and result['parent_analysis_job_id'] == 'hosted-parent'


def test_ingestion_candidates_freeze_only_unassigned_media_and_survive_recovery(setup, tmp_path):
    import sqlite3
    from services.hosted.operations import HostedOperations
    store, artifacts, service, actor, ds, images = setup
    pending = images[2]
    captured = service.capture_incoming(actor, ds['id'], [pending['id']], 'candidate-capture')
    assert captured['manifest']['images'][0]['version_id'] == pending['current_version']
    with pytest.raises(DomainError, match='ingestion_candidate_not_unassigned_incoming'):
        service.capture_incoming(actor, ds['id'], [images[0]['id']], 'canonical-is-not-candidate')
    user = AuthService(store).create_user(actor, 'candidate-reviewer')
    store.assign(actor, ds['id'], [pending['id']], user['id'], 'assign-candidate',
                 expected={pending['id']: {'expected_version': pending['current_version'], 'generation': pending['generation']}})
    assert service.capture_incoming(actor, ds['id'], [pending['id']], 'candidate-capture') == captured
    with pytest.raises(DomainError, match='ingestion_candidate_not_unassigned_incoming'):
        service.capture_incoming(actor, ds['id'], [pending['id']], 'new-capture-after-assignment')
    pins = tmp_path / 'candidate-pins'; pins.mkdir()
    frozen = json.loads(json.dumps(captured['manifest']))
    for image in frozen['images']:
        ref = artifacts.pin(image['media_hash'], pins)
        image['media_path'] = str(pins / ref.sha256)
    path = pins / 'candidates.json'; path.write_text(json.dumps(frozen))
    reference = {key: captured[key] for key in ('candidate_set_id', 'dataset_id')}
    reference.update(manifest_path=str(path), manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert read_candidate_set(reference, allowed_root=tmp_path) == frozen
    assert CanonicalAnalysis(store, artifacts, tmp_path / 'restarted-analysis').get_candidate_set(actor, captured['candidate_set_id']) == captured
    with store.read_connection() as con:
        refs = HostedOperations.artifact_references(con)
        restored = sqlite3.connect(':memory:')
        con.backup(restored)
    try:
        HostedOperations.restore_hook(restored, {'artifacts': refs})
        assert restored.execute('SELECT COUNT(*) FROM hosted_ingestion_candidates').fetchone()[0] == 1
        with pytest.raises(DomainError, match='recovery_artifact_missing'):
            HostedOperations.restore_hook(restored, {'artifacts': [row for row in refs if row['sha256'] != pending['media_hash']]})
    finally:
        restored.close()


def test_native_ingestion_and_salad_training_preserve_upload_population_and_scores(setup, tmp_path, monkeypatch):
    """Real adapter/files, doubled native job/model boundary; GPU work remains open."""
    import sys
    from types import SimpleNamespace
    from services.hosted.native_analysis import _materialize, _run_native
    store, artifacts, service, actor, ds, images = setup
    snapshot = store.create_snapshot(actor, ds['id'], 'salad-reference')
    manifest = snapshot['manifest'] | {'snapshot_id': snapshot['snapshot_id']}
    # Self-supervised SALAD must not inherit detector train/validation requirements.
    for image in manifest['images']:
        image['split'] = 'unsplit'
    candidate = service.capture_incoming(actor, ds['id'], [images[2]['id']], 'native-candidates')['manifest']
    pins = tmp_path / 'native-ingestion-pins'; pins.mkdir()
    for image in [*manifest['images'], *candidate['images']]:
        ref = artifacts.pin(image['media_hash'], pins)
        image['media_path'] = str(pins / ref.sha256)
    candidate_path = pins / 'candidates.json'; candidate_path.write_text(json.dumps(candidate))
    candidate_reference = {key: candidate[key] for key in ('candidate_set_id', 'dataset_id')}
    candidate_reference.update(manifest_path=str(candidate_path), manifest_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest())
    calls, outputs = [], {}
    original_match = lambda metadata, incoming: False
    api = SimpleNamespace(_local_salad_head_reference_matches_request=original_match)
    async def read_all(uploads):
        result = []
        for upload in uploads:
            data = await upload.read(23)
            data += await upload.read()
            await upload.close()
            result.append({'filename': upload.filename, 'sha256': hashlib.sha256(data).hexdigest()})
        return result
    async def train(encoded, uploads):
        payload = json.loads(encoded)
        calls.append(('training', payload, await read_all(uploads)))
        outputs['salad_test'] = {'salad_head_id': 'salad_learned', 'losses': [2., 1.]}
        return {'job_id': 'salad_test'}
    async def ingest(encoded, candidates, references):
        payload = json.loads(encoded)
        assert api._local_salad_head_reference_matches_request({'reference_dataset_id': 'old-imported-id'}, payload)
        assert not api._local_salad_head_reference_matches_request({}, payload | {'salad_head_id': 'not-attached'})
        candidate_rows, reference_rows = await read_all(candidates), await read_all(references)
        calls.append(('ingestion', payload, candidate_rows, reference_rows))
        outputs['di_test'] = {'summary': {'selection_method': 'farthest_first_coverage'},
                              'items': [row | {'selection_score': .317, 'keep': True} for row in candidate_rows],
                              'reference_items': reference_rows}
        return {'job_id': 'di_test'}
    api.create_local_salad_training_job, api.create_data_ingestion_analysis_job = train, ingest
    api.get_data_ingestion_job = lambda _: {'status': 'completed'}
    api.get_data_ingestion_result = lambda identifier: outputs[identifier]
    api.get_data_ingestion_distribution = lambda identifier: {'summary': {'projection': 'pca', 'projection_fit_basis': 'reference'},
        'points': [{'filename': row['filename'], 'projection': [1., 2.]} for key in ('items', 'reference_items') for row in outputs[identifier][key]]}
    monkeypatch.setitem(sys.modules, 'localinferenceapi', api)
    for operation in ('salad_training', 'ingestion'):
        directory = tmp_path / operation; directory.mkdir(); (directory / 'registry').mkdir()
        dataset, source, _ = _materialize(manifest, directory)
        (directory / 'salad_heads').mkdir()
        head = directory / 'salad_heads/attached.pt'; head.write_bytes(b'head-boundary-fixture')
        request = {'operation': operation, 'parameters': {'salad_head_id': 'attached', 'keep_fraction': .5},
                   'timeout': 1, 'canonical_root': str(tmp_path), 'candidate_reference': candidate_reference,
                   'hosted_salad_heads': {'attached': hashlib.sha256(head.read_bytes()).hexdigest()}}
        result = _run_native(request, manifest, dataset, source, directory)
        assert result['reference_snapshot_id'] == manifest['snapshot_id']
        assert api._local_salad_head_reference_matches_request is original_match
        if operation == 'ingestion':
            assert result['result']['items'][0]['selection_score'] == .317
            assert result['result']['items'][0]['hosted_identity']['id'] == images[2]['id']
            assert result['candidate_set_id'] == candidate['candidate_set_id']
            assert {point['hosted_identity']['id'] for point in result['result']['distribution']['points']} == {image['id'] for image in images}
        else:
            assert result['result']['losses'] == [2., 1.]
    assert {row['sha256'] for row in calls[0][2]} == {row['media_hash'] for row in manifest['images']}
    assert {row['sha256'] for row in calls[1][2]} == {images[2]['media_hash']}
    assert {row['sha256'] for row in calls[1][3]} == {row['media_hash'] for row in manifest['images']}
    assert all(not call[1]['use_server_reference_dataset'] for call in calls)


def test_classifier_admission_preserves_declared_splits_and_capture_groups():
    from services.hosted.native_analysis import _classifier_split, _training_membership
    membership = {"a": "train", "b": "train", "c": "val", "d": "test"}
    train, validation, grouped = _classifier_split(membership)(["car", "bus", "car", "bus"], ["a", "b", "c", "c"])
    assert train.tolist() == [0, 1] and validation.tolist() == [2, 3] and grouped
    with pytest.raises(RuntimeError, match="lacks classes"):
        _classifier_split(membership)(["car", "bus"], ["a", "c"])
    manifest = {"images": [{"id": key, "split": value, "annotations": {"capture_metadata": {"video_id": "one-flight"}}} for key, value in membership.items()]}
    with pytest.raises(RuntimeError, match="group crosses"):
        _training_membership(manifest)


def test_hosted_pruning_reuses_actual_planner_ledger_and_retained_export(setup, tmp_path):
    from pathlib import Path
    import shutil
    import time
    import uuid
    import numpy as np
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from services.dataset_pruning import PruningConfig, build_order, digest, pair_evidence, parse_objects
    from services.dataset_pruning_evidence import PairStore
    from services.dataset_pruning_jobs import write_json
    from services.hosted.analysis_api import install_analysis_api
    from services.hosted.pruning import HostedPruning
    store, artifacts, analysis, actor, ds, _ = setup
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
    app = FastAPI()
    from fastapi.responses import JSONResponse
    @app.exception_handler(DomainError)
    async def domain_error(_, error):
        return JSONResponse({"error": error.code}, status_code=error.status)
    install_analysis_api(app, store, artifacts, lambda: actor, lambda: actor, analysis)
    client = TestClient(app)
    base = f"/api/hosted/analysis/jobs/{captured['job_id']}/dataset_pruning/jobs/{native_id}"
    def ok(response):
        assert response.status_code == 200, response.text
        return response.json()
    state = ok(client.get(base + '/review'))
    assert client.get(base.replace(native_id, 'dp_' + 'c' * 16) + '/review').status_code == 404
    settings = {'policy': 'budgeted_coverage', 'object_weight': .8, 'max_removed_images': 1,
                'max_affected_appearance_groups': 0, 'max_lost_appearance_weight_fraction': 0.}
    preview = ok(client.post(base + '/plan/preview', json={'kind': 'settings', 'selection_revision': 1,
        'expected_review_revision': state['review_revision'], 'expected_plan_revision': state['plan_revision'], 'settings': settings}))
    for _ in range(200):
        preview = ok(client.get(base + '/plan/previews/' + preview['preview_id']))
        if preview['status'] not in {'queued', 'running'}: break
        time.sleep(.01)
    assert preview['status'] == 'ready', preview
    def mutate(action, **extra):
        state = ok(client.get(base + '/review'))
        body = {'action': action, 'request_id': uuid.uuid4().hex, 'selection_revision': 1,
                'expected_review_revision': state['review_revision'], 'plan_revision': state['plan_revision'], 'removed_count': state['budget'], **extra}
        return ok(client.post(base + '/review', json=body)), body
    accepted, body = mutate('apply_plan', preview_id=preview['preview_id'], preview_digest=preview['preview_digest'])
    state = accepted['snapshot']
    assert state['approved_count'] == 0 and state['budget'] == 1
    selection = ok(client.post(base + '/selection', json={'selection_revision': 1, 'removed_count': 1, 'plan_revision': state['plan_revision'],
        'expected_review_revision': state['review_revision'], 'view': 'impact'}))
    row = selection['rows'][0]
    approved, body = mutate('approve', candidate_id=row['candidate_id'], witness_id=row['witness_id'], evidence_digest=row['evidence_digest'])
    assert approved['snapshot']['approved_count'] == 1
    assert ok(client.post(base + '/review', json=body))['receipt'] == approved['receipt']
    state = approved['snapshot']
    export = ok(client.post(base + '/exports', json={'request_id': uuid.uuid4().hex, 'selection_revision': 1, 'removed_count': 1,
        'plan_revision': state['plan_revision'], 'expected_review_revision': state['review_revision']}))
    for _ in range(200):
        export = ok(client.get(base + '/exports/' + export['export_id']))
        if export['status'] not in {'queued', 'running'}: break
        time.sleep(.01)
    assert export['status'] == 'completed', export
    download = client.get(base + '/exports/' + export['export_id'] + '/download')
    assert download.status_code == 200
    import zipfile
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        assert len([name for name in archive.namelist() if '/images/' in name]) == 1
    # Whole recovery closure reconstructs the review without native outputs.
    facade = app.state.hosted_pruning
    for service in facade.services.values():
        for index in service.coverage_indexes.values(): index.close()
    shutil.rmtree(facade.root)
    reopened = HostedPruning(analysis)
    restored = reopened.call(actor, captured['job_id'], 'review_snapshot')
    assert restored['approved_count'] == 0  # file-identity proofs need new review
    assert restored['review_revision'] > state['review_revision']
    # The original confirmed population is untouched by variant export/recovery.
    assert store.get_snapshot(actor, captured['snapshot_id'])['manifest']['images']


def test_pruning_ledger_rechecks_logout_and_canonical_revision_at_commit(setup, tmp_path):
    from types import SimpleNamespace
    from services.hosted.pruning import FencedLedger
    from services.dataset_pruning_review import ReviewLedger
    from fastapi import HTTPException
    store, _, analysis, actor, ds, images = setup
    auth = AuthService(store)
    session = auth.login(actor.username, 'a canonical analysis fixture password', 'fixture')
    bound = auth.authenticate(session.token)
    manifest = store.create_snapshot(actor, ds['id'], 'fenced-snapshot')['manifest']
    adapter = SimpleNamespace(owner=SimpleNamespace(store=store), actor=bound, manifest=manifest)
    path = tmp_path / 'fenced-review.sqlite3'
    with pytest.raises(DomainError):
        with FencedLedger(path, adapter).transaction() as state:
            state['budget'] = 12
            auth.logout(session.token)
    assert 'budget' not in ReviewLedger(path).read()
    adapter.actor = actor
    with pytest.raises(HTTPException, match='approved dataset changed'):
        with FencedLedger(path, adapter).transaction() as state:
            state['budget'] = 9
            editor = store.acquire_editor(actor, images[0]['id'], 'tab')
            store.save(actor, ctx(editor, 'fenced-new-version'), editor['annotations'], complete=True)
    assert 'budget' not in ReviewLedger(path).read()
