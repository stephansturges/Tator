"""Cross-layer races at the authenticated HTTP / durable mutation boundary."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from test_hosted_api import acquire, application, context, dataset, login, op, provision, upload


@pytest.mark.parametrize("revocation", ["logout", "password_reset", "disable"])
def test_revoked_session_cannot_commit_an_already_admitted_save(application, monkeypatch, revocation):
    admin = login(application)
    alice_user = provision(admin, "alice")
    alice = login(application, "alice")
    ds = dataset(admin)
    image = upload(admin, ds["id"])
    assigned = admin.post(f"/api/hosted/datasets/{ds['id']}/assignments", json={
        "image_ids": [image["id"]], "annotator_id": alice_user["id"], "operation_id": op(),
        "expected": {image["id"]: {"expected_version": image["current_version"], "generation": image["generation"]}},
    })
    assert assigned.status_code == 200, assigned.text
    image = acquire(alice, assigned.json()["images"][0])
    command = {**context(image), "annotations": {"objects": [], "caption": "late authenticated save"}}
    admitted, resume = Event(), Event()
    original = application.state.store.save

    def delayed_save(*args, **kwargs):
        admitted.set()
        assert resume.wait(10), "test failed to release admitted request"
        return original(*args, **kwargs)

    monkeypatch.setattr(application.state.store, "save", delayed_save)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(alice.post, f"/api/hosted/images/{image['id']}/save", json=command)
        try:
            assert admitted.wait(10), "request did not pass authentication"
            if revocation == "logout":
                application.state.auth.logout(alice.cookies.get("tator_session"))
            elif revocation == "password_reset":
                reset = admin.post(f"/api/hosted/users/{alice_user['id']}/reset")
                assert reset.status_code == 200, reset.text
            else:
                disabled = admin.patch(f"/api/hosted/users/{alice_user['id']}", json={"role": "annotator", "disabled": True,
                    "expected_role": "annotator", "expected_disabled": False})
                assert disabled.status_code == 200, disabled.text
            assert alice.get("/api/hosted/session").status_code == 401
        finally:
            resume.set()
        response = pending.result(timeout=10)

    assert response.status_code in (401, 403), response.text
    current = admin.get(f"/api/hosted/images/{image['id']}").json()
    assert current["current_version"] == image["current_version"]
    with application.state.store.read_connection() as db:
        assert not db.execute("SELECT 1 FROM hosted_receipts WHERE actor_id=? AND operation_id=?",
                              (alice_user["id"], command["operation_id"])).fetchone()


def test_stale_role_change_cannot_reenable_a_disabled_account(application):
    first_admin, second_admin = login(application), login(application)
    alice = provision(first_admin, "alice")
    first_view = next(user for user in first_admin.get("/api/hosted/users").json()["users"] if user["id"] == alice["id"])
    second_view = next(user for user in second_admin.get("/api/hosted/users").json()["users"] if user["id"] == alice["id"])
    disabled = first_admin.patch(f"/api/hosted/users/{alice['id']}", json={
        "role": "annotator", "disabled": True,
        "expected_role": first_view["role"], "expected_disabled": bool(first_view["disabled"]),
    })
    assert disabled.status_code == 200, disabled.text
    stale_role_change = second_admin.patch(f"/api/hosted/users/{alice['id']}", json={
        "role": "admin", "disabled": False,
        "expected_role": second_view["role"], "expected_disabled": bool(second_view["disabled"]),
    })
    assert stale_role_change.status_code == 409, stale_role_change.text
    current = next(user for user in first_admin.get("/api/hosted/users").json()["users"] if user["id"] == alice["id"])
    assert current["disabled"] and current["role"] == "annotator"
