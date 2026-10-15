from __future__ import annotations

import io
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from hosted_app import create_app


PASSWORD = "fixture password sufficiently long"


def op():
    return str(uuid.uuid4())


def login(app, username="administrator"):
    client = TestClient(app)
    response = client.post("/api/hosted/login", json={"username": username, "password": PASSWORD})
    assert response.status_code == 200, response.text
    body = response.json()
    client.headers.update({"x-tator-csrf": body["csrf"], "x-tator-instance": body["instance_id"]})
    return client


@pytest.fixture
def application(tmp_path):
    app = create_app(tmp_path / "state", testing=True)
    app.state.auth.bootstrap("administrator", PASSWORD)
    return app


def provision(client, username):
    response = client.post("/api/hosted/users", json={"username": username})
    assert response.status_code == 200, response.text
    user = response.json()
    assert client.post("/api/hosted/redeem", json={"token": user["invite_token"], "password": PASSWORD}).status_code == 200
    return user


def dataset(client):
    response = client.post("/api/hosted/datasets", json={"name": "Hosted test", "classes": ["Vehicle"], "operation_id": op()})
    assert response.status_code == 200, response.text
    return response.json()


def upload(client, dataset_id, *, color="orange"):
    content = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(content, format="PNG")
    response = client.post(f"/api/hosted/datasets/{dataset_id}/incoming", files={"file": ("image.png", content.getvalue(), "image/png")}, data={"operation_id": op()})
    assert response.status_code == 200, response.text
    return response.json()


def context(record, editor="fixture-editor-one"):
    return {"expected_version": record["current_version"], "generation": record["generation"], "editor_id": editor, "operation_id": op()}


def acquire(client, image, editor="fixture-editor-one", **extra):
    response = client.post(f"/api/hosted/images/{image['id']}/editor", json={"editor_id": editor, "expected_version": image["current_version"], "generation": image["generation"], **extra})
    assert response.status_code == 200, response.text
    return response.json()


def test_real_auth_csrf_revocation_and_last_admin(application):
    anon = TestClient(application)
    assert anon.get("/api/hosted/datasets").status_code == 401
    admin = login(application)
    auth = admin.get("/api/hosted/session").json()
    assert auth["csrf"] == admin.headers["x-tator-csrf"]  # another tab does not invalidate saves
    assert admin.post("/api/hosted/datasets", headers={"x-tator-csrf": "wrong"}, json={"name": "x", "operation_id": op()}).status_code == 403
    assert admin.post("/api/hosted/datasets", headers={"x-tator-instance": "restored-other-server"}, json={"name": "x", "operation_id": op()}).status_code == 409
    assert admin.post("/api/hosted/login", headers={"origin": "https://elsewhere.invalid"}, json={"username": "administrator", "password": PASSWORD}).status_code == 403
    user_id = auth["user"]["user_id"]
    assert admin.patch(f"/api/hosted/users/{user_id}", json={"role": "annotator", "disabled": False, "expected_role": "admin", "expected_disabled": False}).status_code == 409
    user = provision(admin, "alice")
    alice = login(application, "alice")
    assert alice.get("/api/hosted/users").status_code == 403
    # Another tab can replace the shared cookie; the old tab's read proof must
    # not silently turn its existing workspace into a different user's view.
    assert alice.get('/api/hosted/datasets', headers={'x-tator-csrf': auth['csrf']}).status_code == 403
    assert alice.get('/api/hosted/datasets', headers={'x-tator-instance': 'restored-other-server'}).status_code == 409
    assert admin.patch(f"/api/hosted/users/{user['id']}", json={"role": "annotator", "disabled": True, "expected_role": "annotator", "expected_disabled": False}).status_code == 200
    assert alice.get("/api/hosted/session").status_code == 401


def test_full_incoming_review_admission_and_receipt_recovery(application):
    admin = login(application)
    alice_user = provision(admin, "alice")
    provision(admin, "bob")
    alice, bob = login(application, "alice"), login(application, "bob")
    ds = dataset(admin)
    image = upload(admin, ds["id"])
    expected = {image["id"]: {"expected_version": image["current_version"], "generation": image["generation"]}}
    response = admin.post(f"/api/hosted/datasets/{ds['id']}/assignments", json={"image_ids": [image["id"]], "annotator_id": alice_user["id"], "expected": expected, "operation_id": op()})
    assert response.status_code == 200, response.text
    image = response.json()["images"][0]
    assert bob.get(f"/api/hosted/images/{image['id']}/media").status_code == 404
    assert admin.post(f"/api/hosted/images/{image['id']}/editor", json={"editor_id": "administrator-tab"}).status_code == 409
    assert alice.get(f"/api/hosted/images/{image['id']}/media").status_code == 200
    image = acquire(alice, image)
    saved = alice.post(f"/api/hosted/images/{image['id']}/save", json={**context(image), "annotations": {"objects": [], "caption": "A reviewed empty frame", "alternate_captions": ["Nothing to label"]}, "negative": True})
    assert saved.status_code == 200, saved.text
    image = saved.json()
    assert admin.post(f"/api/hosted/datasets/{ds['id']}/snapshot").json()["images"] == []
    submission = context(image)
    response = alice.post(f"/api/hosted/images/{image['id']}/submit", json=submission)
    assert response.status_code == 200, response.text
    submitted = response.json()
    assert alice.post(f"/api/hosted/images/{image['id']}/submit", json=submission).json() == submitted
    assert alice.get(f"/api/hosted/images/{image['id']}/media").status_code == 404
    image = acquire(admin, submitted, "administrator-tab")
    returned = admin.post(f"/api/hosted/images/{image['id']}/return", json={**context(image, "administrator-tab"), "feedback": "Please check the upper edge."})
    assert returned.status_code == 200, returned.text
    image = acquire(alice, returned.json())
    resubmitted = alice.post(f"/api/hosted/images/{image['id']}/submit", json=context(image))
    assert resubmitted.status_code == 200, resubmitted.text
    image = acquire(admin, resubmitted.json(), "administrator-tab")
    approval = context(image, "administrator-tab")
    accepted = admin.post(f"/api/hosted/images/{image['id']}/approve", json=approval)
    assert accepted.status_code == 200, accepted.text
    # Lost reply recovery, including a completely new app using the same database.
    restarted = create_app(application.state.root, testing=True)
    again = login(restarted)
    assert again.post(f"/api/hosted/images/{image['id']}/approve", json=approval).json() == accepted.json()
    snapshot = again.post(f"/api/hosted/datasets/{ds['id']}/snapshot").json()
    assert len(snapshot["images"]) == 1
    assert snapshot["images"][0]["annotations"]["alternate_captions"] == ["Nothing to label"]
    assert snapshot["images"][0]["version_id"] == image["current_version"]
    assert again.get(f"/api/hosted/receipts/{approval['operation_id']}").json()["result"] == accepted.json()


def test_login_link_is_single_use_and_passwords_are_argon2(application):
    admin = login(application)
    created = admin.post("/api/hosted/users", json={"username": "new-person"}).json()
    token = created["invite_token"]
    assert admin.post("/api/hosted/redeem", json={"token": token, "password": PASSWORD}).status_code == 200
    assert admin.post("/api/hosted/redeem", json={"token": token, "password": PASSWORD}).status_code == 400
    with application.state.store.read_connection() as db:
        rows = db.execute("SELECT password_hash FROM hosted_users").fetchall()
        assert all(row[0].startswith("$argon2id$") for row in rows)
        assert not db.execute("SELECT 1 FROM hosted_auth_links WHERE token_hash=?", (token,)).fetchone()


def test_streamed_body_limit_and_preparse_authentication(application):
    anon = TestClient(application)
    def chunks():
        for _ in range(18):
            yield b' ' * (1024 * 1024)
    response = anon.post('/api/hosted/login', content=chunks(), headers={'content-type': 'application/json'})
    assert response.status_code == 413
    # Protected endpoints reject before parsing even a deliberately invalid body.
    response = anon.post('/api/hosted/datasets', content=b'{invalid json', headers={'content-type': 'application/json'})
    assert response.status_code == 401


def test_bulk_review_exact_replay_and_plain_reimport_preserves_labels(application):
    admin = login(application)
    user = provision(admin, "alice")
    alice = login(application, "alice")
    ds = dataset(admin)
    images = []
    for color in ("red", "blue"):
        image = upload(admin, ds["id"], color=color)
        assigned = admin.post(f"/api/hosted/datasets/{ds['id']}/assignments", json={"image_ids": [image["id"]], "annotator_id": user["id"], "operation_id": op(), "expected": {image["id"]: context(image)}}).json()["images"][0]
        image = acquire(alice, assigned)
        image = alice.post(f"/api/hosted/images/{image['id']}/save", json={**context(image), "annotations": {"objects": [], "caption": "Reviewed"}, "complete": True, "negative": True}).json()
        image = alice.post(f"/api/hosted/images/{image['id']}/submit", json=context(image)).json()
        images.append(image)
    images[1] = acquire(admin, images[1], "other-review-tab")
    body = {"operation_id": op(), "editor_id": "bulk-review-tab", "images": [{"image_id": image["id"], "expected_version": image["current_version"], "generation": image["generation"], "operation_id": op()} for image in images]}
    response = admin.post(f"/api/hosted/datasets/{ds['id']}/approve-images", json=body)
    assert response.status_code == 200, response.text
    assert [row["status"] for row in response.json()["images"]] == ["approved", "rejected"]
    assert response.json()["images"][1]["error"] == "editor_lease_conflict"
    assert admin.post(f"/api/hosted/datasets/{ds['id']}/approve-images", json=body).json() == response.json()
    repeated = upload(admin, ds["id"], color="red")
    assert repeated["duplicate"] and repeated["conflict_id"] is None
    actual = admin.get(f"/api/hosted/images/{repeated['id']}").json()
    assert actual["annotations"]["caption"] == "Reviewed" and actual["state"] == "confirmed"


def test_control_directory_rejects_second_live_process(application):
    # Separate application lifespans share a kernel flock, not an in-memory flag.
    second = create_app(application.state.root, testing=True)
    with TestClient(application):
        with pytest.raises(RuntimeError, match='Another hosted control process'):
            with TestClient(second):
                pass
    with TestClient(second) as client:
        assert client.get('/health').status_code == 200


def test_dataset_guidance_is_admin_only_versioned_and_cannot_replace_models(application):
    admin = login(application)
    ds = dataset(admin)
    provision(admin, 'annotator')
    member = login(application, 'annotator')
    body = {'context': 'Overhead urban scenes', 'labelmap_glossary': 'Vehicle: car, bus', 'expected_revision': ds['revision'], 'operation_id': op()}
    route = f"/api/hosted/datasets/{ds['id']}/guidance"
    assert member.post(route, json=body).status_code == 403
    saved = admin.post(route, json=body)
    assert saved.status_code == 200, saved.text
    assert saved.json()['settings']['context'] == body['context']
    assert admin.post(route, json=body).json() == saved.json()
    assert admin.post(route, json={**body, 'operation_id': op()}).status_code == 409
    assert admin.post(route, json={**body, 'operation_id': op(), 'detector': 'arbitrary'}).status_code == 422
