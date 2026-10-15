from __future__ import annotations

import hashlib
import io
import json
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
    if not shutil.which("age") or not shutil.which("age-keygen"):
        pytest.skip("actual age executable required for composed encrypted recovery proof")
    store, auth, artifacts, ops, admin, alice, ds, now = workspace
    canonical, _ = import_image(workspace)
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
