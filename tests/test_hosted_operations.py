from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import zipfile

from PIL import Image
import pytest

from services.hosted.artifacts import ArtifactStore
from services.hosted.auth import AuthService
from services.hosted.backup import create_encrypted_backup, restore_encrypted_backup
from services.hosted.errors import DomainError
from services.hosted.operations import HostedOperations, MINIMUM_RETENTION_SECONDS
from services.hosted.store import HostedStore
from services.hosted.types import Actor, MutationContext


PASSWORD = "an actual fixture password"


@pytest.fixture
def workspace(tmp_path):
    now = [1000.0]
    store = HostedStore(tmp_path / "hosted.sqlite3", clock=lambda: now[0])
    auth = AuthService(store, clock=lambda: now[0])
    user = auth.bootstrap("administrator", PASSWORD)
    admin = Actor(user["id"], "admin", "administrator")
    alice_row = auth.create_user(admin, "alice")
    auth.redeem(alice_row["invite_token"], PASSWORD)
    alice = Actor(alice_row["id"], "annotator", "alice")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    operations = HostedOperations(store, artifacts)
    ds = store.create_dataset(admin, "Dataset", ["Vehicle"], "dataset")
    return store, auth, artifacts, operations, admin, alice, ds, now


def context(image, operation, editor="tab"):
    return MutationContext(image["id"], image["current_version"], image["generation"], editor, operation)


def expected(image, editor=None):
    value = {"expected_version": image["current_version"], "generation": image["generation"]}
    if editor:
        value["editor_id"] = editor
    return value


def import_image(workspace, color="red", ds=None, approved=True):
    store, _, artifacts, _, admin, _, default, _ = workspace
    ds = ds or default
    stream = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(stream, format="PNG")
    data = stream.getvalue()
    with artifacts.exclusive():
        reference = artifacts.put_bytes(data)
        image = store.import_image(admin, ds["id"], reference.sha256, color + ".misleading-suffix", {}, "import-" + ds["id"] + color,
                                   width=32, height=24, size_bytes=len(data))
    if approved:
        image = store.acquire_editor(admin, image["id"], "tab")
        annotations = {"objects": [{"id": "obj-" + image["id"], "class_id": ds["classes"][0]["id"], "bbox": [1, 2, 10, 12],
                                    "polygon": [[1, 2], [11, 2], [11, 14], [1, 14]]}],
                       "caption": "An inspected vehicle", "captions": [{"text": "Alternative caption", "origin": "human"}],
                       "alternate_captions": [{"text": "Reviewed model alternative", "origin": "hosted_qwen", "evidence_sha256": "a" * 64, "generated_qa_pairs": [{"question": "What vehicle?", "answer": "Car"}]}],
                       "text_labels": ["example"]}
        image = store.save(admin, context(image, "save-" + image["id"]), annotations, complete=True)
        image = store.approve(admin, context(image, "approve-" + image["id"]))
    return image, data


def assert_error(code, callback):
    with pytest.raises(DomainError) as exc:
        callback()
    assert exc.value.code == code


@pytest.mark.parametrize('mode', ['RGB', 'L', 'RGBA', 'LA', 'palette', 'uint16', 'gamma'])
def test_media_v2_pixels_passthrough_and_native_identity(tmp_path, mode):
    import numpy as np
    import struct
    from PIL.PngImagePlugin import PngInfo
    from services.hosted.media import working_media_bytes, MEDIA_VIEW_POLICY
    from services.hosted.adapters import _image
    expected, passthrough, options = (20, 40, 60), mode in {'RGB', 'L'}, {}
    if mode == 'RGBA':
        picture = Image.new('RGBA', (4, 2), (255, 0, 0, 0)); picture.putpixel((1, 0), (255, 0, 0, 128)); expected = (255, 255, 255)
    elif mode == 'LA':
        picture = Image.new('LA', (4, 2), (0, 0)); expected = (255, 255, 255)
    elif mode == 'palette':
        picture = Image.new('P', (4, 2), 0); picture.putpalette([255, 0, 0] + [0, 0, 0] * 255); options['transparency'] = 0; expected = (255, 255, 255)
    elif mode == 'uint16':
        picture = Image.fromarray(np.array([[1000, 0, 32768, 65535]] * 2, dtype=np.uint16)); expected = (3, 3, 3)
    elif mode == 'L':
        picture = Image.new('L', (4, 2), 20); expected = (20, 20, 20)
    else:
        picture = Image.new('RGB', (4, 2), expected)
        if mode == 'gamma':
            metadata = PngInfo(); metadata.add(b'gAMA', struct.pack('>I', 45455)); options['pnginfo'] = metadata
    buffer = io.BytesIO(); picture.save(buffer, format='PNG', **options); original = buffer.getvalue()
    working, view = working_media_bytes(original)
    assert (working == original) is passthrough
    assert view['policy'] == MEDIA_VIEW_POLICY and view['source_sha256'] == hashlib.sha256(original).hexdigest()
    path = tmp_path / 'working.png'; path.write_bytes(working)
    native, _ = _image({'media_path': str(path), 'media_sha256': view['sha256'], 'source_media_hash': view['source_sha256'], 'working_media': view}, {'media_root': str(tmp_path)})
    with Image.open(io.BytesIO(working)) as browser_input:
        assert browser_input.convert('RGB').getpixel((0, 0)) == native.getpixel((0, 0)) == expected
        assert browser_input.mode in {'RGB', 'L'} and 'transparency' not in browser_input.info and 'gamma' not in browser_input.info
    if mode == 'uint16':
        assert [native.getpixel((x, 0))[0] for x in range(4)] == [3, 0, 128, 255]
    if mode == 'RGBA':
        assert native.getpixel((1, 0)) == (255, 127, 127)
    native.close()


def test_media_v2_retains_legacy_views_and_frozen_jobs_without_reinterpreting_pixels(workspace, tmp_path):
    from services.hosted.media import HostedMedia, MEDIA_VIEW_POLICY, LEGACY_MEDIA_VIEW_POLICY, validate_working_view
    from services.hosted.analysis import CanonicalAnalysis
    from services.hosted.native_analysis import NativeAnalysisAdapter
    store, _, artifacts, ops, actor, _, ds, _ = workspace
    media = HostedMedia(store, artifacts)
    buffer = io.BytesIO(); Image.new('RGBA', (32, 24), (255, 0, 0, 0)).save(buffer, format='PNG'); original = buffer.getvalue()
    source = artifacts.put_bytes(original)
    legacy = {'policy': LEGACY_MEDIA_VIEW_POLICY, 'coordinate_space': 'encoded', 'source_sha256': source.sha256,
        'source_size_bytes': source.size_bytes, 'source_format': 'PNG', 'orientation': 1, 'frame_index': 0, 'frame_count': 1,
        'sha256': source.sha256, 'size_bytes': source.size_bytes, 'width': 32, 'height': 24, 'content_type': 'image/png', 'suffix': '.png'}
    media._retain(original, legacy)
    current = media.resolve(source)
    assert current['policy'] == MEDIA_VIEW_POLICY and current['sha256'] != legacy['sha256']
    assert media.resolve(source, policy=LEGACY_MEDIA_VIEW_POLICY) == legacy
    with artifacts.open(source) as stream: assert stream.read() == original
    image = store.import_image(actor, ds['id'], source.sha256, 'transparent.png', {}, 'transparent-import', width=32, height=24, size_bytes=source.size_bytes)
    image = store.acquire_editor(actor, image['id'], 'tab'); image = store.save(actor, context(image, 'transparent-save'), {'objects': []}, complete=True, negative=True); store.approve(actor, context(image, 'transparent-approve'))
    analysis = CanonicalAnalysis(store, artifacts, tmp_path / 'analysis')
    new = analysis.capture(actor, ds['id'], 'statistics', {}, 'current-capture')
    prepared = analysis.prepare(actor, new['job_id'])
    with Image.open(prepared['manifest']['images'][0]['image_path']) as pixels: assert pixels.getpixel((0, 0)) == (255, 255, 255)
    old = analysis.capture(actor, ds['id'], 'statistics', {}, 'legacy-capture')
    with store.transaction() as con:
        request = json.loads(con.execute('SELECT request_json FROM hosted_analysis_jobs WHERE id=?', (old['job_id'],)).fetchone()[0]); request.pop('working_media'); request.pop('media_policy'); con.execute('UPDATE hosted_analysis_jobs SET request_json=? WHERE id=?', (json.dumps(request), old['job_id']))
    assert_error('media_policy_upgrade_required', lambda: analysis.prepare(actor, old['job_id']))
    assert analysis.status(actor, old['job_id'])['state'] == 'interrupted'
    historical = analysis.capture(actor, ds['id'], 'statistics', {}, 'historical-capture')
    with store.transaction() as con:
        request = json.loads(con.execute('SELECT request_json FROM hosted_analysis_jobs WHERE id=?', (historical['job_id'],)).fetchone()[0]); request['working_media'] = {image['id']: legacy}; request['media_policy'] = LEGACY_MEDIA_VIEW_POLICY; con.execute('UPDATE hosted_analysis_jobs SET request_json=? WHERE id=?', (json.dumps(request), historical['job_id']))
    analysis.publish_result(actor, historical['job_id'], {'count': 1}, 'historical-result')
    retained = analysis.prepare(actor, historical['job_id'])
    assert Path(retained['manifest']['images'][0]['image_path']).read_bytes() == original
    assert analysis.image_view(actor, historical['job_id'], image['id']) == legacy
    with artifacts.open(source) as stream:
        assert_error('media_policy_upgrade_required', lambda: validate_working_view(stream, legacy, require_current_compatible=True))
    # A real CPU child consumes the newly captured v2 view and keeps source identity.
    snapshot = store.get_snapshot(actor, new['snapshot_id']); manifest = {**snapshot['manifest'], 'snapshot_id': new['snapshot_id']}
    capture = tmp_path / 'capture'; capture.mkdir()
    for item in manifest['images']:
        ref = artifacts.pin(media.reference(current), capture); item.update(media_path=str(capture / ref.sha256), working_media=current)
    path = capture / 'manifest.json'; path.write_text(json.dumps(manifest)); reference = {key: manifest[key] for key in ('snapshot_id', 'dataset_id', 'revision', 'class_version', 'settings_version')}; reference.update(manifest_path=str(path), manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    adapter = NativeAnalysisAdapter(config={'canonical_root': str(capture), 'native_root': str(tmp_path / 'native')}, device='cpu')
    result = adapter.execute('statistics', {'canonical_snapshot': reference}, {'dataset_id': ds['id'], 'annotation_version': new['snapshot_id']})
    output = Path(result['native_directory']); exported = json.loads((output / 'class_analysis/input/hosted-media.json').read_text())
    assert exported['images'][image['id']]['sha256'] == current['sha256']
    assert exported['images'][image['id']]['source_sha256'] == source.sha256


def test_encoded_media_export_native_map_and_recovery_retain_original_identity(workspace, tmp_path):
    from services.hosted.media import HostedMedia
    from services.hosted.analysis import legacy_source_manifest
    from services.hosted.native_analysis import _materialize
    store, _, artifacts, operations, admin, _, ds, _ = workspace
    media = HostedMedia(store, artifacts)
    stream = io.BytesIO(); exif = Image.Exif(); exif[274] = 6
    picture = Image.new("RGB", (64, 48), "red"); picture.paste("blue", (0, 0, 24, 48))
    picture.save(stream, format="TIFF", exif=exif)
    original = stream.getvalue(); view = media.ingest(original)
    assert view["sha256"] != view["source_sha256"]
    assert media.resolve(view["source_sha256"]) == view
    assert HostedMedia(HostedStore(store.path), ArtifactStore(artifacts.root)).resolve(view["source_sha256"]) == view
    with pytest.raises(sqlite3.IntegrityError, match="immutable media view"):
        with store.transaction() as con:
            con.execute("UPDATE hosted_media_views SET view_json='{}'")
    image = store.import_image(admin, ds["id"], view["source_sha256"], "oriented.tif", {}, "oriented-import", width=64, height=48, size_bytes=len(original))
    image = store.acquire_editor(admin, image["id"], "tab")
    image = store.save(admin, context(image, "oriented-save"), {"objects": []}, complete=True, negative=True)
    image = store.approve(admin, context(image, "oriented-approve"))
    snapshot = store.create_snapshot(admin, ds["id"], "oriented-snapshot")
    manifest = store.get_snapshot(admin, snapshot["snapshot_id"])["manifest"]
    target = tmp_path / "encoded-view"
    operations.materialize_snapshot(admin, snapshot["snapshot_id"], target)
    mapped = json.loads((target / "hosted-media.json").read_text())["images"][image["id"]]
    exported = target / mapped["path"]
    assert exported.suffix == ".png" and hashlib.sha256(exported.read_bytes()).hexdigest() == view["sha256"]
    assert json.loads((target / "hosted-snapshot.json").read_text()) == manifest
    with Image.open(exported) as pixels:
        assert pixels.size == (64, 48) and pixels.getpixel((8, 20)) == (0, 0, 255)
    source = legacy_source_manifest(manifest, target)
    assert source["images"][0]["image_sha256"] == view["sha256"]
    assert source["images"][0]["source_image_sha256"] == view["source_sha256"]
    # The supervised child receives only pinned working bytes and no store.
    pinned = json.loads(json.dumps(manifest))
    pinned["images"][0].update(media_path=str(exported), working_media=view)
    _, child_source, _ = _materialize(pinned, tmp_path / "child")
    assert child_source["images"][0]["source_image_sha256"] == view["source_sha256"]
    with store.read_connection() as con:
        references = HostedOperations.artifact_references(con)
        restored = sqlite3.connect(":memory:"); con.backup(restored)
    try:
        assert {view["sha256"], view["source_sha256"]}.issubset({ref["sha256"] for ref in references})
        HostedOperations.restore_hook(restored, {"artifacts": references})
        restored.execute("DELETE FROM hosted_artifact_references WHERE kind='media_view' AND sha256=?", (view["sha256"],))
        with pytest.raises(DomainError, match="recovery_media_view_missing"):
            HostedOperations.restore_hook(restored, {"artifacts": references})
    finally:
        restored.close()
    with artifacts.open(view["source_sha256"]) as raw:
        assert raw.read() == original


def test_historical_media_audit_reports_ambiguity_and_corruption_without_mutation(workspace):
    from services.hosted.media import HostedMedia
    store, _, artifacts, _, admin, _, ds, _ = workspace
    stream = io.BytesIO(); exif = Image.Exif(); exif[274] = 6
    Image.new("RGB", (64, 48), "orange").save(stream, format="JPEG", quality=95, exif=exif)
    data = stream.getvalue()
    ambiguous_ref, corrupt_ref = artifacts.put_bytes(data), artifacts.put_bytes(data[:-40])
    body = {"objects": [{"id": "legacy-object", "class_id": ds["classes"][0]["id"], "bbox": [1, 2, 8, 10]}]}
    ambiguous = store.import_image(admin, ds["id"], ambiguous_ref.sha256, "legacy.jpg", body, "legacy-oriented", width=64, height=48, size_bytes=len(data))
    corrupt = store.import_image(admin, ds["id"], corrupt_ref.sha256, "legacy-broken.jpg", {}, "legacy-corrupt", width=64, height=48, size_bytes=len(data)-40)
    media = HostedMedia(store, artifacts)
    with store.read_connection() as con:
        before = [tuple(row) for row in con.execute("SELECT * FROM hosted_annotation_versions ORDER BY id")]
    issues = {row["image_id"]: row["issue"] for row in media.audit()}
    assert issues == {ambiguous["id"]: "historical_label_coordinate_space_unknown", corrupt["id"]: "invalid_image_import"}
    with store.read_connection() as con:
        assert [tuple(row) for row in con.execute("SELECT * FROM hosted_annotation_versions ORDER BY id")] == before
        assert con.execute("SELECT COUNT(*) FROM hosted_media_views").fetchone()[0] == 0
    with artifacts.open(ambiguous_ref) as raw:
        assert raw.read() == data


def test_failed_admission_does_not_permanently_retain_media_views(workspace):
    from services.hosted.media import HostedMedia
    store, _, artifacts, operations, admin, _, ds, _ = workspace
    stream = io.BytesIO(); exif = Image.Exif(); exif[274] = 6
    Image.new("RGB", (64, 48), "purple").save(stream, format="JPEG", exif=exif)
    data = stream.getvalue(); media = HostedMedia(store, artifacts)
    with artifacts.exclusive():
        view = media.stage(data)
        with pytest.raises(DomainError, match="dataset_not_found"):
            store.import_image(admin, "unknown-dataset", view["source_sha256"], "example.jpg", {}, "rejected-import", width=64, height=48, size_bytes=len(data))
    with store.read_connection() as con:
        assert con.execute("SELECT COUNT(*) FROM hosted_media_views").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM hosted_artifact_references WHERE kind='media_view'").fetchone()[0] == 0
    assert not operations.preview_orphan_purge(admin, [view["source_sha256"]])["artifacts"][0]["referenced"]
    with artifacts.exclusive():
        media.stage(data)
        store.import_image(admin, ds["id"], view["source_sha256"], "example.jpg", {}, "accepted-import", width=64, height=48, size_bytes=len(data))
    assert operations.preview_orphan_purge(admin, [view["source_sha256"]])["artifacts"][0]["referenced"]
    assert media.resolve(view["source_sha256"]) == view
    assert operations.preview_orphan_purge(admin, [view["sha256"]])["artifacts"][0]["referenced"]


@pytest.mark.parametrize("invalid", [None, [], {"expected_version": "v", "generation": True}, {"version_id": "v", "generation": -1}])
def test_archive_dataset_nested_preconditions_are_typed_without_mutation(workspace, invalid):
    store, _, _, operations, admin, _, ds, _ = workspace
    image, _ = import_image(workspace)
    with pytest.raises(DomainError, match="invalid_precondition"):
        operations.archive_dataset(admin, ds["id"], {image["id"]: invalid}, "bad-archive", expected_revision=store.get_dataset(admin, ds["id"])["revision"])
    assert store.image(admin, image["id"])["current_version"] == image["current_version"]


def test_recovery_paid_stage_requires_exact_registered_input_artifacts(workspace):
    """A real retained stage with a doubled provider transport; no network call."""
    from services.hosted.providers import ProviderAccounts
    from services.hosted.provider_gateway import PaidCaptionGateway
    store, _, artifacts, _, admin, _, ds, _ = workspace
    image, _ = import_image(workspace)
    image = store.acquire_editor(admin, image['id'], 'tab')
    accounts = ProviderAccounts(store)
    accounts.configure(admin, 'openai', expected_revision=0, operation_id='configure-recovery-provider',
                       credential='synthetic-recovery-test-credential', enabled=True,
                       monthly_limit=10000, request_limit=1000, allowed_datasets=[ds['id']], models=['fixture'])
    gateway = PaidCaptionGateway(accounts, artifacts, pricing={'fixture': {
        'input_microdollars_per_million': 1000, 'cached_input_microdollars_per_million': 100,
        'output_microdollars_per_million': 1000, 'input_token_ceiling': 100, 'max_output_tokens': 100}},
        transport=lambda *_: {'id': 'fixture-response', 'status': 'completed', 'usage': {'input_tokens': 2, 'output_tokens': 2},
                              'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Retained test output'}]}]})
    result = gateway.stage(admin, context(image, 'stage-context'), stage_id='recovery-stage', model='fixture',
                           user_prompt='Synthetic recovery input.', source_media_hash=image['media_hash'], max_output_tokens=20)
    assert result['input_artifacts']
    with store.read_connection() as con:
        references = HostedOperations.artifact_references(con)
        restored = sqlite3.connect(':memory:'); con.backup(restored)
    try:
        HostedOperations.restore_hook(restored, {'artifacts': references})
        digest = result['input_artifacts'][0]['sha256']
        # The backup still contains the bytes. Removing the authority reference
        # must not permit a dangling secondary result index to pass recovery.
        restored.execute("DELETE FROM hosted_artifact_references WHERE kind='paid_stage' AND owner_id=? AND sha256=?", (result['call_id'], digest))
        with pytest.raises(DomainError, match='recovery_paid_stage_artifact_missing'):
            HostedOperations.restore_hook(restored, {'artifacts': references})
    finally:
        restored.close()


def test_archive_removes_current_membership_preserves_history_and_restores_exact_version(workspace):
    store, _, artifacts, ops, admin, alice, ds, now = workspace
    image, original = import_image(workspace)
    frozen = store.create_snapshot(admin, ds["id"], "frozen")
    image = store.acquire_editor(admin, image["id"], "tab")
    archived = ops.archive_image(admin, context(image, "archive"), "Review later")
    assert archived["files_deleted"] == 0
    assert not store.snapshot(admin, ds["id"])["images"]
    assert not store.list_images(admin, ds["id"])
    assert store.get_dataset(admin, ds["id"])["counts"] == {}
    assert_error("image_not_found", lambda: store.image(admin, image["id"]))
    assert_error("image_not_found", lambda: store.save(admin, context(image, "late"), {"objects": []}))
    assert store.get_snapshot(admin, frozen["snapshot_id"])["manifest"]["images"][0]["version_id"] == image["current_version"]
    with artifacts.open(image["media_hash"]) as handle:
        assert handle.read() == original
    assert_error("admin_required", lambda: ops.list_archives(alice))
    preview = ops.purge_preview(admin, archived["archive_id"])
    assert not preview["retention_elapsed"]
    assert_error("purge_retention_not_elapsed", lambda: ops.purge(admin, archived["archive_id"], preview["preview_hash"], "too-soon"))
    now[0] += MINIMUM_RETENTION_SECONDS
    preview = ops.purge_preview(admin, archived["archive_id"])
    assert preview["retention_elapsed"] and not preview["can_purge"]
    assert preview["references"][0]["snapshot_references"] == 1
    assert_error("purge_references_retained", lambda: ops.purge(admin, archived["archive_id"], preview["preview_hash"], "protected"))
    restored = ops.restore_archive(admin, archived["archive_id"], "restore")
    assert ops.restore_archive(admin, archived["archive_id"], "restore") == restored
    current = store.image(admin, image["id"])
    assert current["current_version"] == image["current_version"] and current["generation"] > image["generation"]
    assert store.snapshot(admin, ds["id"])["images"][0]["version_id"] == image["current_version"]


def test_dataset_archive_requires_exact_idle_members_and_independent_image_archive_survives(workspace):
    store, _, _, ops, admin, alice, ds, _ = workspace
    first, _ = import_image(workspace)
    second, _ = import_image(workspace, "blue", approved=False)
    group = store.assign(admin, ds["id"], [second["id"]], alice.user_id, "assign", {second["id"]: expected(second)})
    second = group["images"][0]
    revision = store.get_dataset(admin, ds["id"])["revision"]
    captures = {row["id"]: expected(row) for row in (first, second)}
    assert_error("image_owned_by_annotator", lambda: ops.archive_dataset(admin, ds["id"], captures, "blocked", expected_revision=revision))
    second = store.override(admin, context(second, "override"))
    first = store.acquire_editor(admin, first["id"], "tab")
    captures = {row["id"]: expected(row) for row in (first, second)}
    assert_error("editor_lease_conflict", lambda: ops.archive_dataset(admin, ds["id"], captures, "open-editor", expected_revision=revision))
    individually = ops.archive_image(admin, context(first, "individual"))
    revision = store.get_dataset(admin, ds["id"])["revision"]
    archived = ops.archive_dataset(admin, ds["id"], {second["id"]: expected(second)}, "whole-dataset", expected_revision=revision)
    assert not store.list_datasets(admin) and not store.list_images(admin)
    assert_error("dataset_not_found", lambda: store.snapshot(admin, ds["id"]))
    assert_error("dataset_still_archived", lambda: ops.restore_archive(admin, individually["archive_id"], "premature"))
    ops.restore_archive(admin, archived["archive_id"], "restore-dataset")
    assert [row["id"] for row in store.list_images(admin)] == [second["id"]]
    assert_error("image_not_found", lambda: store.image(admin, first["id"]))
    ops.restore_archive(admin, individually["archive_id"], "restore-individual")
    assert len(store.list_images(admin)) == 2
    # Archiving a handed-off submission removes it from active review, but its
    # author still needs an honest metadata-only progress state.
    third, _ = import_image(workspace, "green", approved=False)
    store.assign(admin, ds["id"], [third["id"]], alice.user_id, "third-assignment", {third["id"]: expected(third)})
    third = store.acquire_editor(alice, third["id"], "tab")
    third = store.save(alice, context(third, "third-negative"), {"objects": []}, negative=True)
    third = store.submit(alice, context(third, "third-submit"))
    third = store.acquire_editor(admin, third["id"], "tab")
    archived_submission = ops.archive_image(admin, context(third, "third-archive"), "Waiting for source review")
    progress = store.own_submission_status(alice)[0]
    assert progress["archived"] and progress["status"] == "archived"
    assert progress["archive_reason"] == "Waiting for source review"
    assert "annotations" not in progress and "media_hash" not in progress
    assert_error("image_not_found", lambda: store.image(alice, third["id"]))
    ops.restore_archive(admin, archived_submission["archive_id"], "third-restore")
    assert store.own_submission_status(alice)[0]["status"] == "awaiting_review"


def test_materializer_and_zip_use_frozen_approved_versions_and_preserve_pixels(workspace, tmp_path):
    store, _, artifacts, ops, admin, alice, ds, _ = workspace
    approved, original = import_image(workspace)
    pending, _ = import_image(workspace, "blue", approved=False)
    frozen = store.create_snapshot(admin, ds["id"], "before-change")
    edit = store.acquire_editor(admin, approved["id"], "tab")
    changed = store.image(admin, approved["id"])["annotations"]
    changed["objects"][0]["bbox"][0] = 10
    store.save(admin, context(edit, "later-edit"), changed, complete=True)
    materialized = ops.materialize_snapshot(admin, frozen["snapshot_id"], tmp_path / "view")
    root = tmp_path / "view"
    assert materialized["image_count"] == 1
    assert (root / "train/images" / (approved["id"] + ".png")).read_bytes() == original
    label = (root / "train/labels" / (approved["id"] + ".txt")).read_text().strip().split()
    assert float(label[1]) == pytest.approx(6 / 32)
    assert len(list((root / "train/images").iterdir())) == 1
    assert (root / "train/captions" / (approved["id"] + ".txt")).read_text() == "An inspected vehicle"
    caption_record = json.loads((root / 'captions.jsonl').read_text().splitlines()[0])
    assert caption_record['alternate_captions'][0]['evidence_sha256'] == 'a' * 64
    assert caption_record['alternate_captions'][0]['generated_qa_pairs'][0]['answer'] == 'Car'
    assert (root / "train/segments" / (approved["id"] + ".txt")).exists()
    assert json.loads((root / "hosted-snapshot.json").read_text())["images"][0]["annotations"]["objects"][0]["bbox"][0] == 1
    assert_error("admin_required", lambda: ops.materialize_snapshot(alice, frozen["snapshot_id"], tmp_path / "forbidden"))
    export = ops.export_canonical(admin, ds["id"], tmp_path / "dataset.zip", "export", frozen["snapshot_id"])
    assert ops.export_canonical(admin, ds["id"], tmp_path / "dataset.zip", "export", frozen["snapshot_id"]) == export
    with zipfile.ZipFile(tmp_path / "dataset.zip") as archive:
        assert archive.read("train/images/" + approved["id"] + ".png") == original
        assert not any(pending["id"] in name for name in archive.namelist())
        assert json.loads(archive.read("data.yaml"))["names"] == ["Vehicle"]
    assert ops.export_reference(admin, export["export_id"]).sha256 == export["sha256"]
    with artifacts.open(approved["media_hash"]) as source:
        assert hashlib.sha256(source.read()).hexdigest() == approved["media_hash"]


def test_unicode_class_zip_indices_and_invalid_historical_snapshot_replay(workspace, tmp_path):
    from services.detectors import _yolo_load_labelmap_impl
    store, _, _, ops, admin, _, _, _ = workspace
    names = ['Véhicule', '自転車', 'Καράβι']
    dataset = store.create_dataset(admin, 'Unicode map', names, 'unicode-dataset')
    image, _ = import_image(workspace, ds=dataset)
    image = store.acquire_editor(admin, image['id'], 'tab')
    annotations = store.image(admin, image['id'])['annotations']
    annotations['objects'][0]['class_id'] = dataset['classes'][2]['id']
    image = store.save(admin, context(image, 'unicode-save'), annotations, complete=True)
    frozen = store.create_snapshot(admin, dataset['id'], 'unicode-snapshot')
    destination = tmp_path / 'unicode.zip'
    exported = ops.export_canonical(admin, dataset['id'], destination, 'unicode-export', frozen['snapshot_id'])
    original = destination.read_bytes()
    with zipfile.ZipFile(destination) as archive:
        labelmap = tmp_path / 'labelmap.txt'; labelmap.write_bytes(archive.read('labelmap.txt'))
        assert _yolo_load_labelmap_impl(labelmap) == names
        assert archive.read('train/labels/' + image['id'] + '.txt').split()[0] == b'2'
    # Emulate a pre-validator snapshot without mutating any current class map.
    manifest = frozen['manifest']; invalid = manifest['classes'][1]
    invalid['name'] = 'Car\u2028Boat'
    with store.transaction() as con:
        con.execute('DROP TRIGGER hosted_snapshots_immutable_update')
        con.execute('UPDATE hosted_snapshots SET manifest_json=? WHERE id=?', (json.dumps(manifest), frozen['snapshot_id']))
    with pytest.raises(DomainError) as failure:
        ops.export_canonical(admin, dataset['id'], tmp_path / 'new.zip', 'new-unicode-export', frozen['snapshot_id'])
    assert failure.value.code == 'invalid_snapshot_class_names'
    assert failure.value.detail == {'class_id': invalid['id'], 'name': invalid['name']}
    assert_error('invalid_snapshot_class_names', lambda: ops.materialize_snapshot(admin, frozen['snapshot_id'], tmp_path / 'invalid'))
    assert ops.export_canonical(admin, dataset['id'], destination, 'unicode-export', frozen['snapshot_id']) == exported
    assert destination.read_bytes() == original


@pytest.mark.parametrize('grayscale,interlace', [(True, False), (False, False), (False, True)])
def test_media_v2_matches_16bit_transparency_before_reducing_samples(grayscale, interlace, monkeypatch):
    import png
    from services.hosted.media import working_media_bytes
    key = (1000,) if grayscale else (1000, 2000, 3000)
    adjacent = (1001,) if grayscale else (1001, 2000, 3000)
    stream = io.BytesIO()
    png.Writer(2, 1, greyscale=grayscale, bitdepth=16, transparent=key, interlace=interlace).write(stream, [key + adjacent])
    source = stream.getvalue()
    working, view = working_media_bytes(source)
    with Image.open(io.BytesIO(working)) as image:
        assert image.getpixel((0, 0)) == (255, 255, 255)
        assert image.getpixel((1, 0)) == ((3, 3, 3) if grayscale else (3, 7, 11))
    assert view['source_sha256'] == hashlib.sha256(source).hexdigest()
    monkeypatch.setitem(__import__('sys').modules, 'png', None)
    assert_error('media_decoder_unavailable', lambda: working_media_bytes(source))


def test_real_orphan_purge_retention_reference_fencing_and_interrupted_unlink_recovery(workspace):
    store, _, artifacts, ops, admin, _, _, now = workspace
    image, original = import_image(workspace)
    orphan = artifacts.put_bytes(b"unreferenced abandoned result")
    preview = ops.preview_orphan_purge(admin, [orphan.sha256])
    assert not preview["can_purge"]
    assert_error("purge_not_eligible", lambda: ops.purge_orphans(admin, [orphan.sha256], preview["preview_hash"], "early"))
    now[0] += MINIMUM_RETENTION_SECONDS
    preview = ops.preview_orphan_purge(admin, [orphan.sha256])
    assert preview["can_purge"]
    with store.transaction() as con:
        con.execute("CREATE TRIGGER fail_purge AFTER UPDATE ON hosted_purge_items BEGIN SELECT RAISE(ABORT,'interrupted after unlink'); END")
    with pytest.raises(sqlite3.IntegrityError, match="interrupted after unlink"):
        ops.purge_orphans(admin, [orphan.sha256], preview["preview_hash"], "purge")
    assert_error("operation_id_reused", lambda: store.create_dataset(admin, "wrong reuse", [], "purge"))
    with store.transaction() as con:
        con.execute("DROP TRIGGER fail_purge")
    recovered = ops.purge_orphans(admin, [orphan.sha256], preview["preview_hash"], "purge")
    assert recovered["already_absent_digests"] == [orphan.sha256]
    assert recovered["logical_bytes_removed"] == 0
    assert ops.purge_orphans(admin, [orphan.sha256], preview["preview_hash"], "purge") == recovered
    second = artifacts.put_bytes(b"another orphan")
    ops.preview_orphan_purge(admin, [second.sha256])
    now[0] += MINIMUM_RETENTION_SECONDS
    preview = ops.preview_orphan_purge(admin, [second.sha256])
    removed = ops.purge_orphans(admin, [second.sha256], preview["preview_hash"], "purge-second")
    assert removed["deleted_digests"] == [second.sha256] and removed["logical_bytes_removed"] == second.size_bytes
    preview = ops.preview_orphan_purge(admin, [image["media_hash"]])
    assert preview["artifacts"][0]["referenced"] and not preview["can_purge"]
    with artifacts.open(image["media_hash"]) as stream:
        assert stream.read() == original


def test_real_encrypted_restore_checks_blob_closure_and_preserves_continued_work(workspace, tmp_path):
    from services.hosted.media import HostedMedia, LEGACY_MEDIA_VIEW_POLICY
    if not shutil.which("age") or not shutil.which("age-keygen"):
        pytest.skip("actual age executable required for composed encrypted recovery proof")
    store, auth, artifacts, ops, admin, alice, ds, now = workspace
    canonical, _ = import_image(workspace)
    oriented = io.BytesIO(); exif = Image.Exif(); exif[274] = 6
    Image.new("RGB", (24, 16), "orange").save(oriented, format="JPEG", exif=exif)
    view = HostedMedia(store, artifacts).ingest(oriented.getvalue())
    with artifacts.open(HostedMedia.reference(view)) as stream:
        legacy_view = HostedMedia(store, artifacts)._retain(stream.read(), {**view, 'policy': LEGACY_MEDIA_VIEW_POLICY})
    pending, _ = import_image(workspace, "blue", approved=False)
    store.assign(admin, ds["id"], [pending["id"]], alice.user_id, "assignment", {pending["id"]: expected(pending)})
    login = auth.login("alice", PASSWORD, address="127.0.0.1")
    session_actor = auth.authenticate(login.token)
    lease = store.acquire_editor(session_actor, pending["id"], "tab")
    draft = store.save(session_actor, context(lease, "negative-draft"), {"objects": []}, negative=True)
    custom = artifacts.put_bytes(b"registered custom model weights")
    ops.register_artifact(admin, custom, "tool_model", "fixture-model", "register-model")
    with store.transaction() as con:
        con.execute("CREATE TABLE hosted_tool_models(model_id TEXT PRIMARY KEY,definition_json TEXT NOT NULL,updated_at REAL NOT NULL)")
        con.execute("INSERT INTO hosted_tool_models VALUES (?,?,?)", ("fixture-model", json.dumps({"checkpoint_sha256": custom.sha256}), now[0]))
    identity = tmp_path / "private-age-key.txt"
    subprocess.run([shutil.which("age-keygen"), "-o", str(identity)], check=True, capture_output=True)
    recipient = subprocess.run([shutil.which("age-keygen"), "-y", str(identity)], check=True, capture_output=True, text=True).stdout.strip()
    backup = create_encrypted_backup(store.path, artifacts, tmp_path / "recovery.age", recipient=recipient, references=ops.artifact_references)
    restored = restore_encrypted_backup(backup.path, tmp_path / "restored", identity=identity, restore_hook=ops.restore_hook)
    other = HostedStore(restored.database_path, clock=lambda: now[0])
    other_auth = AuthService(other, clock=lambda: now[0])
    assert_error("login_required", lambda: other_auth.authenticate(login.token))
    assert_error("login_required", lambda: other.authorize_context(session_actor, context(draft, "old-session")))
    continued_login = other_auth.login("alice", PASSWORD, address="127.0.0.1")
    continued = other_auth.authenticate(continued_login.token)
    current = other.image(continued, pending["id"])
    assert current["current_version"] == draft["current_version"] and current["negative"]
    assert current["generation"] > draft["generation"] and current["editor_lease"] is None
    current = other.acquire_editor(continued, pending["id"], "new-tab")
    assert other.submit(continued, context(current, "submit-restored", "new-tab"))["state"] == "submitted"
    assert other.snapshot(admin, ds["id"])["images"][0]["id"] == canonical["id"]
    with ArtifactStore(restored.artifacts_root).open(custom) as stream:
        assert stream.read() == b"registered custom model weights"
    restored_media = HostedMedia(other, ArtifactStore(restored.artifacts_root))
    assert restored_media.resolve(view["source_sha256"]) == view
    assert restored_media.resolve(view['source_sha256'], policy=LEGACY_MEDIA_VIEW_POLICY) == legacy_view
    with restored_media.artifacts.open(view["source_sha256"]) as stream:
        assert stream.read() == oriented.getvalue()
    with restored_media.artifacts.open(restored_media.reference(view)) as stream, Image.open(stream) as pixels:
        assert pixels.size == (24, 16) and pixels.getexif().get(274, 1) == 1
    # A valid encrypted archive can still be incomplete if enumeration was
    # wrong. The schema-aware hook must reject it before publishing a root.
    incomplete = create_encrypted_backup(store.path, artifacts, tmp_path / "incomplete.age", recipient=recipient, references=lambda con: [])
    assert_error("recovery_artifact_missing", lambda: restore_encrypted_backup(incomplete.path, tmp_path / "must-not-publish", identity=identity, restore_hook=ops.restore_hook))
    assert not (tmp_path / "must-not-publish").exists()
    # A declaration without its owning registry reference is not recoverable,
    # even if another owner happens to retain the same physical bytes.
    with store.transaction() as con:
        con.execute("INSERT INTO hosted_tool_models VALUES (?,?,?)", ("unregistered-model", json.dumps({"checkpoint_sha256": custom.sha256}), now[0]))
    missing_model = create_encrypted_backup(store.path, artifacts, tmp_path / "missing-model.age", recipient=recipient, references=ops.artifact_references)
    assert_error("recovery_model_artifact_missing", lambda: restore_encrypted_backup(missing_model.path, tmp_path / "missing-model-root", identity=identity, restore_hook=ops.restore_hook))
    assert not (tmp_path / "missing-model-root").exists()
