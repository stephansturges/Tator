"""HTTP composition with actual authority, immutable files and age recovery."""
from __future__ import annotations

import io
import shutil
import subprocess
import uuid
import zipfile
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from hosted_app import create_app
from services.hosted.backup import restore_encrypted_backup
from services.hosted.operations import prepare_restored_database


PASSWORD = "a real operations fixture password"


def command():
    return {"operation_id": str(uuid.uuid4())}


def login(app):
    client = TestClient(app)
    response = client.post("/api/hosted/login", json={"username": "administrator", "password": PASSWORD})
    assert response.status_code == 200, response.text
    session = response.json()
    client.headers.update({"x-tator-csrf": session["csrf"], "x-tator-instance": session["instance_id"]})
    return client


def successful(response):
    assert response.status_code == 200, response.text
    return response.json()


def test_orphan_cleanup_http_retention_reference_race_and_exact_replay(tmp_path):
    from services.hosted.types import Actor
    from services.hosted.operations import MINIMUM_RETENTION_SECONDS
    app = create_app(tmp_path / 'cleanup', testing=True)
    now = [time.time()]
    app.state.auth.clock = app.state.store.clock = lambda: now[0]
    account = app.state.auth.bootstrap('administrator', PASSWORD)
    actor = Actor(account['id'], 'admin')
    client = login(app)
    refs = [app.state.artifacts.put_bytes(value) for value in (b'preserved model', b'old unused input', b'late saved input')]
    app.state.operations.register_artifact(actor, refs[0], 'fixture_model', 'fixture', str(uuid.uuid4()))
    rows, cursor = [], ''
    while True:
        page = successful(client.post('/api/hosted/storage/orphans/scan', json={'after': cursor, 'limit': 1}))
        assert page['scanned'] <= 1
        rows.extend(page['artifacts']); cursor = page['next_cursor']
        if cursor is None:
            break
    assert {row['sha256'] for row in rows} == {ref.sha256 for ref in refs[1:]}
    assert all(not row['eligible'] for row in rows)
    digests = [ref.sha256 for ref in refs[1:]]
    preview = successful(client.post('/api/hosted/storage/orphans/preview', json={'digests': digests}))
    assert client.post('/api/hosted/storage/orphans/purge', json={'digests': digests, 'preview_hash': preview['preview_hash'], **command()}).status_code == 409
    now[0] += MINIMUM_RETENTION_SECONDS + 1
    client = login(app)
    preview = successful(client.post('/api/hosted/storage/orphans/preview', json={'digests': digests}))
    assert preview['can_purge']
    app.state.operations.register_artifact(actor, refs[2], 'fixture_input', 'fixture', str(uuid.uuid4()))
    assert client.post('/api/hosted/storage/orphans/purge', json={'digests': digests, 'preview_hash': preview['preview_hash'], **command()}).status_code == 409
    for ref in refs:
        app.state.artifacts.verify(ref)
    selection = {'digests': [refs[1].sha256]}
    preview = successful(client.post('/api/hosted/storage/orphans/preview', json=selection))
    body = {**selection, 'preview_hash': preview['preview_hash'], **command()}
    deleted = successful(client.post('/api/hosted/storage/orphans/purge', json=body))
    assert deleted['deleted_digests'] == [refs[1].sha256]
    assert successful(client.post('/api/hosted/storage/orphans/purge', json=body)) == deleted
    app.state.artifacts.verify(refs[0]); app.state.artifacts.verify(refs[2])
    assert client.post('/api/hosted/storage/orphans/preview', json={'digests': ['../outside']}).status_code == 422
    invite = app.state.auth.create_user(actor, 'annotator'); app.state.auth.redeem(invite['invite_token'], PASSWORD)
    member = TestClient(app); response = member.post('/api/hosted/login', json={'username': 'annotator', 'password': PASSWORD}).json()
    member.headers.update({'x-tator-csrf': response['csrf'], 'x-tator-instance': response['instance_id']})
    assert member.post('/api/hosted/storage/orphans/scan', json={}).status_code == 403


def test_export_archive_and_encrypted_backup_restore_through_actual_http(tmp_path):
    if not shutil.which("age") or not shutil.which("age-keygen"):
        pytest.skip("actual age executable required for HTTP recovery composition")
    app = create_app(tmp_path / "source", testing=True)
    app.state.auth.bootstrap("administrator", PASSWORD)
    client = login(app)
    ds = successful(client.post("/api/hosted/datasets", json={"name": "HTTP operations", "classes": ["Vehicle"], **command()}))
    stream = io.BytesIO()
    Image.new("RGB", (32, 24), "orange").save(stream, format="PNG")
    original = stream.getvalue()
    image = successful(client.post(f"/api/hosted/datasets/{ds['id']}/incoming", files={"file": ("fixture.png", original, "image/png")}, data=command()))
    editor_id = "operations-editor"
    image = successful(client.post(f"/api/hosted/images/{image['id']}/editor", json={"editor_id": editor_id, "expected_version": image["current_version"], "generation": image["generation"]}))
    def mutation(image):
        return {"expected_version": image["current_version"], "generation": image["generation"], "editor_id": editor_id, **command()}
    image = successful(client.post(f"/api/hosted/images/{image['id']}/save", json={**mutation(image), "annotations": {"objects": []}, "negative": True}))
    image = successful(client.post(f"/api/hosted/images/{image['id']}/approve", json=mutation(image)))
    export_body = command()
    export = successful(client.post(f"/api/hosted/datasets/{ds['id']}/export", json=export_body))
    download = client.get(export["download_url"])
    assert download.status_code == 200
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        assert image['split'] == 'unsplit'
        assert archive.read(f"unsplit/images/{image['id']}.png") == original
    assert not list((app.state.root / "downloads").iterdir())
    assert successful(client.post(f"/api/hosted/datasets/{ds['id']}/export", json=export_body)) == export
    assert client.get(export["download_url"]).content == download.content

    image = successful(client.post(f"/api/hosted/images/{image['id']}/editor", json={"editor_id": editor_id, "expected_version": image["current_version"], "generation": image["generation"]}))
    archived = successful(client.post(f"/api/hosted/images/{image['id']}/archive", json={**mutation(image), "reason": "Temporary review"}))
    assert client.get(f"/api/hosted/images/{image['id']}/media").status_code == 404
    assert successful(client.post(f"/api/hosted/datasets/{ds['id']}/snapshot"))["images"] == []
    assert client.get(export["download_url"]).content == download.content
    successful(client.post(f"/api/hosted/archives/{archived['archive_id']}/restore", json=command()))
    assert client.get(f"/api/hosted/images/{image['id']}/media").content == original

    key = tmp_path / "identity.txt"
    subprocess.run([shutil.which("age-keygen"), "-o", str(key)], check=True, capture_output=True)
    recipient = subprocess.run([shutil.which("age-keygen"), "-y", str(key)], check=True, capture_output=True, text=True).stdout.strip()
    backup_body = {"recipient": recipient, **command()}
    backup = successful(client.post("/api/hosted/recovery-backups", json=backup_body))
    assert successful(client.post("/api/hosted/recovery-backups", json=backup_body)) == backup
    encrypted = client.get(backup["download_url"])
    assert encrypted.status_code == 200 and encrypted.content.startswith(b"age-encryption.org/")
    archive_path = tmp_path / "downloaded.age"
    archive_path.write_bytes(encrypted.content)
    restored = restore_encrypted_backup(archive_path, tmp_path / "restored", identity=key, restore_hook=prepare_restored_database)
    restored_app = create_app(restored.root, testing=True)
    stale = TestClient(restored_app)
    stale.cookies.update(client.cookies)
    assert stale.get("/api/hosted/session").status_code == 401
    recovered = login(restored_app)
    snapshot = successful(recovered.post(f"/api/hosted/datasets/{ds['id']}/snapshot"))
    assert [row["id"] for row in snapshot["images"]] == [image["id"]]
    assert recovered.get(export["download_url"]).content == download.content
    assert recovered.get(f"/api/hosted/images/{image['id']}/media").content == original
