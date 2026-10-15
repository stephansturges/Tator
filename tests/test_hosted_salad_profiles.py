"""Actual saved heads, hosted transactions and portable capture; no GPU claim."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
import shutil
import sqlite3

from fastapi import HTTPException
import pytest

from services.hosted.artifacts import ArtifactStore
from services.hosted.auth import AuthService
from services.hosted.errors import DomainError
from services.hosted.operations import HostedOperations
from services.hosted.salad_profiles import HostedSaladProfiles
from services.hosted.store import HostedStore
from services.hosted.types import Actor


@pytest.fixture
def workspace(tmp_path):
    store = HostedStore(tmp_path / "hosted.sqlite3")
    auth = AuthService(store)
    user = auth.bootstrap("administrator", "A real fixture password for SALAD")
    actor = Actor(user["id"], "admin", "administrator")
    annotator = auth.create_user(actor, "annotator")
    other = Actor(annotator["id"], "annotator", "annotator")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    operations = HostedOperations(store, artifacts)
    service = HostedSaladProfiles(store, artifacts)
    dataset = store.create_dataset(actor, "Images", ["Vehicle"], "new-dataset")
    return store, artifacts, operations, service, actor, other, dataset


def saved_head(scope="whole_image", *, valid=True):
    import torch
    from utils.local_salad import LocalSALADConfig, LocalSALADHead, LOCAL_SALAD_POLICY, LOCAL_SALAD_TRAINER
    config = LocalSALADConfig(num_channels=4, num_clusters=2, cluster_dim=3, token_dim=2, hidden_dim=4)
    head = LocalSALADHead(config, seed=42)
    payload = {"format": config.file_format, "config": config.to_dict(), "state_dict": head.state_dict(),
               "metadata": {"policy": LOCAL_SALAD_POLICY if valid else "external checkpoint",
                            "trainer": LOCAL_SALAD_TRAINER, "label": "Trained vehicle head",
                            "encoder_type": "dinov3", "encoder_model": "facebook/dinov3-fixture",
                            "representation_scope": scope, "epochs": 2}}
    data = io.BytesIO()
    torch.save(payload, data)
    return data.getvalue()


def retained_source(workspace, data, *, name="retained-head.pt"):
    store, artifacts, operations, service, actor, _, ds = workspace
    ref = artifacts.put_bytes(data)
    operations.register_artifact(actor, ref, "migration_salad_weights", ds["id"], "retain-" + ref.sha256)
    current = store.get_dataset(actor, ds["id"])
    previous = current["settings"].get("migration", {}).get("artifacts", [])
    settings = {"migration": {"artifacts": [*previous, {"source_file": name, "role": "salad_weights", **ref.as_dict()}]}}
    store.patch_dataset_settings(actor, ds["id"], settings, current["revision"], "source-" + ref.sha256)
    return ref, service.discover(actor, ds["id"])["sources"][-1]


def register(workspace, source, *, alias="vehicle_head", operation="register-head"):
    _, _, _, service, actor, _, ds = workspace
    payload = {"source_id": source["source_id"], "head_id": alias,
               "expected_revision": service.snapshot(actor, ds["id"])["revision"], "operation_id": operation}
    return service.register(actor, ds["id"], payload), payload


def mutate(workspace, **payload):
    _, _, _, service, actor, _, ds = workspace
    payload.setdefault("expected_revision", service.snapshot(actor, ds["id"])["revision"])
    return service.mutate(actor, ds["id"], payload)


def test_actual_head_registration_recipe_retry_and_scope_rules(workspace):
    store, artifacts, _, service, actor, other, ds = workspace
    ref, source = retained_source(workspace, saved_head())
    assert source["status"] == "available" and source["label"] == "Trained vehicle head"
    assert source["encoder_model"] == "facebook/dinov3-fixture"
    assert source["metadata"]["descriptor_dim"] == 8
    result, original = register(workspace, source)
    assert result["profiles"]["dataset"]["head_ids"] == ["vehicle_head"]
    assert result["profiles"]["heads"][0]["head_sha256"] == ref.sha256
    renamed = mutate(workspace, action="rename", head_id="vehicle_head", display_label="Vehicles", operation_id="rename-head")
    assert service.register(actor, ds["id"], original) == result
    with pytest.raises(DomainError, match="operation_id_reused"):
        service.register(actor, ds["id"], {**original, "head_id": "other_alias"})
    recipe = {"components": [{"encoder_type": "dinov3", "head_id": "vehicle_head", "head_sha256": ref.sha256, "weight": 1.0}]}
    preferred = mutate(workspace, action="prefer_recipe", role="whole_image", recipe=recipe, operation_id="prefer-whole")
    assert preferred["profiles"]["dataset"]["preferred_recipes"]["whole_image"] == recipe
    for action in ("detach", "archive"):
        with pytest.raises(HTTPException) as error:
            mutate(workspace, action=action, head_id="vehicle_head", operation_id=action + "-used")
        assert error.value.status_code == 409
    with pytest.raises(HTTPException) as error:
        mutate(workspace, action="prefer_recipe", role="bbox_crop", recipe=recipe, operation_id="wrong-scope")
    assert error.value.status_code == 400
    with pytest.raises(DomainError, match="admin_required"):
        service.snapshot(other, ds["id"])
    assert store.get_dataset(actor, ds["id"])["settings"]["migration"]["artifacts"][0]["sha256"] == ref.sha256


def test_invalid_trainer_and_unretained_paths_never_register(workspace):
    store, _, _, service, actor, _, ds = workspace
    _, source = retained_source(workspace, saved_head(valid=False))
    assert source["status"] == "invalid" and source["error"]["code"] == "salad_head_invalid"
    before = service.snapshot(actor, ds["id"])
    with pytest.raises(DomainError, match="salad_head_invalid"):
        register(workspace, source)
    with pytest.raises(DomainError, match="salad_source_unavailable"):
        service.register(actor, ds["id"], {"source_id": "/tmp/arbitrary.pt", "operation_id": "path-attempt", "expected_revision": before["revision"]})
    assert service.snapshot(actor, ds["id"]) == before
    assert store.receipt(actor, "register-head") is None


def test_concurrent_cas_and_dataset_scoped_archive_preserve_receipts(workspace):
    store, _, _, service, actor, _, ds = workspace
    _, source = retained_source(workspace, saved_head())
    register(workspace, source)
    revision = service.snapshot(actor, ds["id"])["revision"]
    def rename(index):
        payload = {"action": "rename", "head_id": "vehicle_head", "display_label": "Name " + str(index),
                   "expected_revision": revision, "operation_id": "concurrent-" + str(index)}
        try:
            return payload, service.mutate(actor, ds["id"], payload)
        except HTTPException as error:
            return payload, error
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(rename, (1, 2)))
    success = [(request, result) for request, result in outcomes if isinstance(result, dict)]
    assert len(success) == 1
    assert [result.status_code for _, result in outcomes if isinstance(result, HTTPException)] == [409]
    mutate(workspace, action="archive", head_id="vehicle_head", archived=True, operation_id="archive-head")
    assert service.mutate(actor, ds["id"], success[0][0]) == success[0][1]
    second = store.create_dataset(actor, "Independent", ["Vehicle"], "second-dataset")
    snapshot = service.snapshot(actor, second["id"])
    assert not snapshot["dataset"]["head_ids"] and not snapshot["heads"][0].get("archived", False)
    service.mutate(actor, second["id"], {"action": "attach", "head_id": "vehicle_head", "expected_revision": snapshot["revision"], "operation_id": "second-attach"})
    assert service.snapshot(actor, second["id"])["dataset"]["head_ids"] == ["vehicle_head"]
    with pytest.raises(DomainError, match="salad_dataset_mismatch"):
        mutate(workspace, action="attach", head_id="vehicle_head", source_identity="dataset:" + second["id"], operation_id="cross-dataset")


def test_frozen_capture_retains_exact_heads_and_defaults_after_current_archive_and_relocation(workspace, tmp_path):
    store, artifacts, operations, service, actor, _, ds = workspace
    ref, source = retained_source(workspace, saved_head())
    register(workspace, source)
    recipe = {"components": [{"encoder_type": "dinov3", "head_id": "vehicle_head", "head_sha256": ref.sha256, "weight": 1.0}]}
    mutate(workspace, action="prefer_recipe", role="whole_image", recipe=recipe, operation_id="save-preference")
    frozen = store.create_snapshot(actor, ds["id"], "before-deployment-change")
    # Replacing the default with a pooled descriptor permits archiving only the
    # current deployment. Earlier frozen jobs still own the exact trained head.
    mutate(workspace, action="prefer_recipe", role="whole_image", recipe={"components": [{"encoder_type": "dinov3", "weight": 1.0}]}, operation_id="pooled-default")
    mutate(workspace, action="archive", head_id="vehicle_head", archived=True, operation_id="archive-current")
    capture = service.capture(actor, frozen["snapshot_id"], tmp_path / "pins")
    assert capture["pruning_defaults"]["scene_components"] == recipe["components"]
    assert hashlib.sha256(Path(capture["cache_inputs"][0]["path"]).read_bytes()).hexdigest() == ref.sha256
    assert capture["cache_inputs"][0]["relative_path"] == "vehicle_head.pt"
    with store.read_connection() as con:
        assert ref.as_dict() in operations.artifact_references(con)
    relocated = tmp_path / "recovered"
    relocated.mkdir()
    with store.read_connection() as source_db, sqlite3.connect(relocated / "hosted.sqlite3") as target:
        source_db.backup(target)
        operations.restore_hook(target, {"artifacts": operations.artifact_references(target)})
    shutil.copytree(artifacts.root, relocated / "artifacts")
    recovered = HostedSaladProfiles(HostedStore(relocated / "hosted.sqlite3"), ArtifactStore(relocated / "artifacts"))
    after = recovered.capture(actor, frozen["snapshot_id"], relocated / "pins")
    assert after["cache_inputs"][0]["sha256"] == ref.sha256
    assert after["pruning_defaults"] == capture["pruning_defaults"]
    assert str(relocated) in after["cache_inputs"][0]["path"]
    # A generic migration ref still retains these bytes, but it must not hide a
    # broken immutable deployment registry during full recovery admission.
    with recovered.store.transaction() as con:
        con.execute("DELETE FROM hosted_artifact_references WHERE kind='salad_head'")
        with pytest.raises(DomainError, match="recovery_salad_head_missing"):
            operations.restore_hook(con, {"artifacts": operations.artifact_references(con)})


def test_native_retained_head_discovery_uses_snapshot_dataset(workspace):
    from services.hosted.analysis import CanonicalAnalysis
    store, artifacts, _, service, actor, _, ds = workspace
    analysis = CanonicalAnalysis(store, artifacts, artifacts.root.parent / "analysis")
    job = analysis.capture(actor, ds["id"], "statistics", {}, "native-result")
    ref = artifacts.put_bytes(saved_head())
    # Retain a native published artifact index with the same schema consumed by
    # publish_native_result; numerical analysis itself is covered separately.
    with store.transaction() as con:
        con.execute("INSERT INTO hosted_analysis_artifacts VALUES (?,?,?,?)", (job["job_id"], "salad_heads/local_trained.pt", ref.sha256, ref.size_bytes))
        con.execute("INSERT INTO hosted_artifact_references VALUES ('analysis_result',?,?,?,?)", (job["job_id"], ref.sha256, ref.size_bytes, store.clock()))
        con.execute("UPDATE hosted_analysis_jobs SET state='complete' WHERE id=?", (job["job_id"],))
    sources = service.discover(actor, ds["id"])["sources"]
    assert len(sources) == 1 and sources[0]["origin"] == "analysis" and sources[0]["status"] == "available"
    second = store.create_dataset(actor, "Other", ["Vehicle"], "second")
    assert service.discover(actor, second["id"])["sources"] == []


def test_hosted_http_profile_administration_uses_real_login_csrf_and_revision(tmp_path):
    from fastapi.testclient import TestClient
    from hosted_app import create_app
    app = create_app(tmp_path / "http-state", testing=True)
    password = "An actual HTTP SALAD fixture password"
    admin = app.state.auth.bootstrap("administrator", password)
    client = TestClient(app)
    login = client.post("/api/hosted/login", json={"username": "administrator", "password": password}).json()
    client.headers.update({"x-tator-csrf": login["csrf"], "x-tator-instance": login["instance_id"]})
    response = client.post("/api/hosted/datasets", json={"name": "Profiles", "classes": ["Car"], "operation_id": "http-dataset"})
    assert response.status_code == 200, response.text
    ds = response.json()
    base = "/api/hosted/datasets/" + ds["id"] + "/salad-profiles"
    assert TestClient(app).get(base).status_code == 401
    snapshot = client.get(base).json()
    payload = {"action": "prefer_recipe", "role": "whole_image", "recipe": {"components": [{"encoder_type": "dinov3", "weight": 1.0}]},
               "expected_revision": snapshot["revision"], "operation_id": "http-prefer"}
    without_csrf = TestClient(app, cookies=client.cookies)
    assert without_csrf.post(base, json=payload).status_code == 403
    changed = client.post(base, json=payload)
    assert changed.status_code == 200, changed.text
    assert changed.json()["profiles"]["revision"] == snapshot["revision"] + 1
    assert client.post(base, json=payload).json() == changed.json()
    assert client.post(base, json={**payload, "operation_id": "http-stale"}).status_code == 409
    assert client.get(base + "/sources").json()["sources"] == []
    client.post("/api/hosted/logout")
    assert client.get(base).status_code == 401
