"""Real auth/store/media/runtime API composition; model calls are fixture-only."""
from __future__ import annotations

import io
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from services.hosted.artifacts import ArtifactStore
from services.hosted.auth import AuthService
from services.hosted.errors import DomainError
from services.hosted.runtime import AdapterSpec
from services.hosted.store import HostedStore
from services.hosted.tool_api import install_tool_api
from services.hosted.types import MutationContext

PASSWORD = "this is a long fixture password"


class FixtureTools:
    def __init__(self, *, config, device):
        self.config = config

    def execute(self, operation, payload, context):
        tag = str(payload.get("uuid", "") or payload.get("caption_request", {}).get("user_prompt", "") or "")
        if tag.startswith("hold"):
            root = Path(self.config["fixture_gate_root"])
            (root / (tag + ".started")).touch()
            while not (root / (tag + ".release")).exists():
                time.sleep(0.01)
        if "canonical_snapshot" in payload:
            from services.hosted.analysis import read_canonical_snapshot
            manifest = read_canonical_snapshot(payload["canonical_snapshot"], allowed_root=self.config["canonical_root"])
            return {"images": [image["id"] for image in manifest["images"]]}
        assert Path(payload["media_path"]).is_file()
        if operation == "qwen_caption_case":
            from tools.run_qwen_caption_flow_benchmark import summarize_caption, _qa_machine_validated_pair
            marker = Path(self.config["fixture_gate_root"]) / ("caption-attempt-" + context["image_id"])
            text = "A bounding box is visible." if not marker.exists() and tag == "retry-fixture" else "A red image."
            marker.touch()
            # This fixture doubles model output, including the verifier's
            # qualified row, while exercising real job retention/save binding.
            pairs = [_qa_machine_validated_pair({"question": "Which color is visible?", "answer": "Red."}, verifier_decision="accept")] if payload.get("settings", {}).get("instruction_dataset") else []
            return {"status": "ok", "response": {"caption": text, "used_counts": {}, "generated_qa_pairs": pairs, "recovery_events": [{"type": "fixture_recovery"}] if tag == "recovery-fixture" else [],
                "hosted_evidence": {"model_io": [{"raw_output": text}]}}, "caption_quality": summarize_caption(text, {})}
        return {"bbox": [0.5, 0.5, 0.25, 0.25], "image_id": context["image_id"]}


from services.hosted.caption_batch import CaptionBatchAdapter


class FixtureCaptionBatch(CaptionBatchAdapter):
    """Exercise the real batch/case/health path with only generation doubled."""
    def _runtime(self):
        gate, held = self.config.get("fixture_gate"), self.config.get("fixture_hold_image")
        class Qwen:
            default_model = "qwen"
            io = [{"event": "prompt_budget", "prompt_tokens": 5, "input_tokens": 5,
                   "requested_max_new_tokens": 20, "effective_max_new_tokens": 20}]
            events, events_dropped, loaded = [], 0, {"qwen": True}
            def execute(self, operation, image, payload, context):
                if operation == "qwen_caption_preview":
                    return {"full_text": "Describe the actual image."}
                if gate:
                    marker = Path(gate) / (context["image_id"] + ".calls")
                    with marker.open("a") as stream:
                        stream.write("called\n")
                    if context["image_id"] == held:
                        (Path(gate) / "batch-held").touch()
                        while not (Path(gate) / "batch-release").exists():
                            time.sleep(.01)
                return {"caption": "A red square on a plain surface.", "used_counts": {}}
            def close(self):
                pass
        self.qwen = Qwen()
        return self.qwen


def op():
    return uuid.uuid4().hex


def eventually(check):
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


@pytest.fixture
def system(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (str(Path(__file__).parent), os.environ.get("PYTHONPATH")))))
    source = tmp_path / "source-models"
    source.mkdir()
    (source / "model.pt").write_bytes(b"fixture model - never loaded as a neural network")
    (source / "head.meta.pkl").write_bytes(b"fixture recipe - never unpickled")
    qwen = source / "qwen"
    qwen.mkdir()
    (qwen / "config.json").write_text('{"model_type":"qwen3_vl"}')
    (qwen / "weights.safetensors").write_bytes(b"fixture weights - never loaded")
    gate = tmp_path / "gate"
    gate.mkdir()
    configuration = {
        "allowed_model_roots": [str(source)],
        "models": {
            "sam": {"kind": "sam1", "checkpoint": str(source / "model.pt"), "model_type": "vit_b"},
            "detector": {"kind": "yolo", "checkpoint": str(source / "model.pt"), "labels": ["vehicle"]},
            "classifier": {"kind": "classifier", "checkpoint": str(source / "model.pt"), "metadata_path": str(source / "head.meta.pkl"), "labels": ["vehicle"]},
            "qwen": {"kind": "qwen", "model_path": str(qwen), "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in qwen.iterdir()}},
        },
        "user_sam": {"default_model_id": "sam", "allowed_model_ids": ["sam"], "device": "cpu", "capacity": 1},
        "detector_device": "cpu",
        "user_qwen": {"default_model_id": "qwen", "allowed_model_ids": ["qwen"], "device": "cpu"},
        "heavy_jobs": {"fixture": {"factory": "test_hosted_tools_api:FixtureTools", "config": {"fixture_gate_root": str(gate)},
            "device": "cpu", "operation": "fixture_heavy", "allowed_parameters": ["uuid"], "timeout": 10}},
    }
    config_file = tmp_path / "models.json"
    config_file.write_text(json.dumps(configuration))
    monkeypatch.setenv("TATOR_HOSTED_MODEL_CONFIG", str(config_file))
    root = tmp_path / "state"
    store = HostedStore(root / "hosted.sqlite3")
    auth = AuthService(store)
    auth.bootstrap("admin", PASSWORD)
    admin_login = auth.login("admin", PASSWORD, "fixture")
    admin = auth.authenticate(admin_login.token)
    user = auth.create_user(admin, "alice")
    auth.redeem(user["invite_token"], PASSWORD)
    alice_login = auth.login("alice", PASSWORD, "fixture")
    alice = auth.authenticate(alice_login.token)
    artifacts = ArtifactStore(root / "artifacts")
    app = FastAPI()
    @app.exception_handler(DomainError)
    async def domain_error(request, error):
        return JSONResponse({"error": error.code, "detail": error.detail}, status_code=error.status)
    def identity(request: Request):
        return auth.authenticate(request.headers.get("authorization", ""),
            csrf=request.headers.get("x-tator-csrf", "") if request.method != "GET" else None)
    def administrator(actor=Depends(identity)):
        auth.require_admin(actor)
        return actor
    controller = install_tool_api(app, store, auth, artifacts, identity, administrator, root)
    original_spec = controller._spec
    def fixture_spec(actor, dataset_id, operation):
        selected, owner = original_spec(actor, dataset_id, operation)
        config = json.loads(json.dumps(dict(selected.config), default=lambda value: dict(value) if hasattr(value, "items") else list(value)))
        config["fixture_gate_root"] = str(gate)
        return AdapterSpec("test_hosted_tools_api:FixtureTools", config, "cpu"), owner
    monkeypatch.setattr(controller, "_spec", fixture_spec)
    admin_client = TestClient(app, headers={"authorization": admin_login.token, "x-tator-csrf": admin_login.csrf})
    client = TestClient(app, headers={"authorization": alice_login.token, "x-tator-csrf": alice_login.csrf})
    dataset = store.create_dataset(admin, "Runtime fixture", ["vehicle"], op())
    images = []
    for color in ("red", "green"):
        content = io.BytesIO()
        Image.new("RGB", (64, 48), color).save(content, format="PNG")
        ref = artifacts.put_bytes(content.getvalue())
        images.append(store.import_image(admin, dataset["id"], ref.sha256, color + ".png", {}, op(), width=64, height=48, size_bytes=ref.size_bytes))
    expected = {image["id"]: {"expected_version": image["current_version"], "generation": image["generation"]} for image in images}
    assignment = store.assign(admin, dataset["id"], [image["id"] for image in images], alice.user_id, op(), expected=expected)
    images = assignment["images"]
    yield {"app": app, "store": store, "auth": auth, "artifacts": artifacts, "controller": controller,
           "admin": admin, "alice": alice, "admin_client": admin_client, "client": client, "images": images,
           "dataset": dataset, "gate": gate, "root": root, "configuration": configuration}
    controller.close()


def request(image, operation="sam_point", **parameters):
    return {"dataset_id": image["dataset_id"], "image_id": image["id"], "expected_version": image["current_version"],
        "generation": image["generation"], "editor_id": "alice-editor", "operation": operation,
        "parameters": {"point_x": 20, "point_y": 15, **parameters}}


def install(system, model="sam"):
    response = system["admin_client"].post(f"/api/hosted/tools/models/{model}/install", json={"operation_id": op()})
    assert response.status_code == 200, response.text


def acquire(system, image):
    return system["store"].acquire_editor(system["alice"], image["id"], "alice-editor", expected_version=image["current_version"], generation=image["generation"])


def test_saved_sam_recipe_metadata_private_cascades_and_admission(system):
    """Installed identity and private preset ownership compose with real auth/store."""
    controller, client, admin = system["controller"], system["client"], system["admin_client"]
    source = Path(controller.configuration["models"]["sam"]["checkpoint"]).parent
    (source / "vocab.gz").write_bytes(b"fixture vocabulary")
    recipe = source / "vehicle-recipe"
    recipe.mkdir()
    content = json.dumps({"class_name": "vehicle", "recipe": {"mode": "sam3_steps", "steps": [{"type": "text", "text_prompt": "vehicle"}], "clip_head": {"path": "head.npz"}}})
    (recipe / "recipe.json").write_text(content)
    controller.configuration["models"].update({
        "sam3": {"kind": "sam3", "checkpoint": str(source / "model.pt"), "bpe_path": str(source / "vocab.gz")},
        "vehicle-recipe": {"kind": "sam3_recipe", "model_path": str(recipe), "files": {"recipe.json": hashlib.sha256(content.encode()).hexdigest()},
            "recipe_file": "recipe.json", "components": {"sam3": "sam3", "extra_classifier": "classifier"}},
    })
    for model in ("sam", "sam3", "classifier", "vehicle-recipe"):
        install(system, model)
    dataset_id = system["dataset"]["id"]
    path = f"/api/hosted/tools/datasets/{dataset_id}"
    selection = {"model_ids": ["vehicle-recipe"], "expected_revision": system["store"].get_dataset(system["admin"], dataset_id)["revision"], "operation_id": op()}
    assert client.post(path + "/sam3-recipes", json=selection).status_code == 403
    response = admin.post(path + "/sam3-recipes", json=selection)
    assert response.status_code == 200, response.text
    assert admin.post(path + "/sam3-recipes", json=selection).json() == response.json()
    metadata = client.get(f"/api/hosted/tools/metadata?dataset_id={dataset_id}").json()
    assert metadata["sam3_recipes"][0]["class_id"] == 0
    assert metadata["sam3_recipes"][0]["clip_head"] == {"registered": True}
    assert metadata["classifiers"][0]["path"] == "classifier"
    assert [model["id"] for model in metadata["sam_models"]] == ["sam"]  # Attached component is not a selectable user model.
    saved = {"label": "Vehicle pass", "steps": [{"recipe_id": "vehicle-recipe"}], "operation_id": op()}
    result = client.post(path + "/cascades", json=saved)
    assert result.status_code == 200, result.text
    cascade = result.json()
    assert client.post(path + "/cascades", json=saved).json() == cascade
    assert admin.get(path + "/cascades").json() == []
    assert admin.get(path + "/cascades/" + cascade["id"]).status_code == 404
    assert client.post(path + "/cascades", json={**saved, "operation_id": op(), "steps": [{"recipe_id": "../outside"}]}).status_code == 403
    image = acquire(system, system["images"][0])
    body = request(image, "sam3_recipe_chain")
    body["parameters"] = {"steps": [{"recipe_id": "vehicle-recipe"}]}
    accepted = client.post("/api/hosted/tools/request", json=body)
    assert accepted.status_code == 202, accepted.text
    assert finish(client, accepted.json()["id"])[0]["state"] == "completed"
    body["parameters"]["steps"][0]["override_class_id"] = 8
    assert client.post("/api/hosted/tools/request", json=body).status_code == 422
    removal = {"expected_revision": cascade["revision"], "operation_id": op()}
    assert client.post(path + "/cascades/" + cascade["id"] + "/delete", json=removal).status_code == 200
    assert client.get(path + "/cascades").json() == []


def test_caption_guidance_is_frozen_and_explicit_empty_override_wins(system):
    install(system)
    install(system, "qwen")
    store, admin, client = system["store"], system["admin"], system["client"]
    dataset_id = system["dataset"]["id"]
    dataset = store.get_dataset(admin, dataset_id)
    dataset = store.patch_dataset_settings(admin, dataset_id, {"context": "Original scene context", "labelmap_glossary": "vehicle: road transport"}, dataset["revision"], op())
    image = acquire(system, system["images"][0])
    body = {key: value for key, value in request(image).items() if key not in {"operation", "parameters"}}
    body.update(operation_id=op(), caption_request={}, settings={"attempts": 1})
    first = client.post("/api/hosted/tools/caption-jobs", json=body)
    assert first.status_code == 202, first.text
    job_id = first.json()["job_id"]
    finished = eventually(lambda: (row if (row := client.get("/api/hosted/tools/caption-jobs/" + job_id).json())["status"] == "completed" else None))
    assert finished["request"]["caption_request"] == {"user_prompt": "Original scene context", "labelmap_glossary": "vehicle: road transport"}
    store.patch_dataset_settings(admin, dataset_id, {"context": "New scene context"}, dataset["revision"], op())
    replay = client.post("/api/hosted/tools/caption-jobs", json=body)
    assert replay.status_code == 202 and replay.json()["job_id"] == job_id
    assert replay.json()["request"]["caption_request"] == finished["request"]["caption_request"]
    body.update(operation_id=op(), caption_request={"user_prompt": "", "labelmap_glossary": ""})
    explicit = client.post("/api/hosted/tools/caption-jobs", json=body)
    assert explicit.status_code == 202, explicit.text
    assert explicit.json()["request"]["caption_request"] == body["caption_request"]


def finish(client, request_id):
    url = "/api/hosted/tools/requests/" + request_id
    status = eventually(lambda: (value if (value := client.get(url).json()).get("state") in {"completed", "stale", "failed", "cancelled"} else None))
    return status, client.get(url + "/result")


def test_real_api_media_authority_preload_and_server_only_model_setup(system):
    client, admin = system["client"], system["admin_client"]
    image = system["images"][0]
    assert client.post("/api/hosted/tools/models/sam/install", json={"operation_id": op()}).status_code == 403
    assert client.post("/api/hosted/tools/request", json=request(image, "sam_preload")).status_code == 503
    install(system)
    preload = client.post("/api/hosted/tools/request", json=request(image, "sam_preload", slot="next"))
    assert preload.status_code == 202, preload.text
    assert finish(client, preload.json()["id"])[0]["state"] == "completed"
    assert client.post("/api/hosted/tools/request", json=request(image)).status_code == 409
    image = acquire(system, image)
    accepted = client.post("/api/hosted/tools/request", json=request(image))
    assert accepted.status_code == 202, accepted.text
    status, result = finish(client, accepted.json()["id"])
    assert status["state"] == "completed", status
    assert result.status_code == 200 and result.json()["result"]["image_id"] == image["id"]
    assert "session_id" not in result.text and "media_path" not in result.text and "checkpoint" not in result.text
    assert admin.get(f"/api/hosted/tools/requests/{accepted.json()['id']}").status_code == 404
    bad = request(image, media_path="/etc/passwd")
    assert client.post("/api/hosted/tools/request", json=bad).status_code == 422
    bad = request(image)
    bad["dataset_id"] = "different-dataset"
    assert client.post("/api/hosted/tools/request", json=bad).status_code == 409
    assert client.post("/api/hosted/tools/request", json=request(image, "sam_point_auto")).status_code == 503
    eventually(lambda: not list(system["controller"].media_root.iterdir()))


def test_queued_authority_is_bound_to_original_login_session(system):
    install(system)
    image = acquire(system, system["images"][0])
    client = system["client"]
    active = client.post("/api/hosted/tools/request", json=request(image, uuid="hold-session")).json()["id"]
    eventually(lambda: (system["gate"] / "hold-session.started").exists())
    queued = client.post("/api/hosted/tools/request", json=request(image)).json()["id"]
    system["auth"].logout(client.headers["authorization"])
    (system["gate"] / "hold-session.release").touch()
    runtime = system["controller"]._runtime
    assert runtime.wait(active, actor_id=system["alice"].user_id, timeout=12)["state"] == "stale"
    assert runtime.wait(queued, actor_id=system["alice"].user_id, timeout=12)["state"] == "stale"
    new_login = system["auth"].login("alice", PASSWORD, "fixture")
    fresh = TestClient(system["app"], headers={"authorization": new_login.token, "x-tator-csrf": new_login.csrf})
    assert fresh.get(f"/api/hosted/tools/requests/{active}/result").status_code == 409
    assert client.get(f"/api/hosted/tools/requests/{active}").status_code == 401


def test_detector_selection_cas_preserves_settings_and_recovers_receipt(system):
    install(system, "detector")
    store, admin, dataset_id = system["store"], system["admin"], system["dataset"]["id"]
    current = store.get_dataset(admin, dataset_id)
    current = store.set_dataset_settings(admin, dataset_id, {"salad": {"id": "keep-recipe"}}, current["revision"], op())
    selection = {"model_id": "detector", "expected_revision": current["revision"], "operation_id": op()}
    url = f"/api/hosted/tools/datasets/{dataset_id}/detector"
    assert system["client"].post(url, json=selection).status_code == 403
    response = system["admin_client"].post(url, json=selection)
    assert response.status_code == 200, response.text
    assert response.json()["settings"]["salad"] == {"id": "keep-recipe"}
    latest = store.patch_dataset_settings(admin, dataset_id, {"other": True}, response.json()["revision"], op())
    assert system["admin_client"].post(url, json=selection).json() == response.json()
    stale = {**selection, "operation_id": op()}
    assert system["admin_client"].post(url, json=stale).status_code == 409
    image = acquire(system, system["images"][0])
    queued = system["client"].post("/api/hosted/tools/request", json=request(image, "detect_full"))
    assert queued.status_code == 202, queued.text
    assert finish(system["client"], queued.json()["id"])[0]["state"] == "completed"
    with store.read_connection() as con:
        refs = con.execute("SELECT sha256 FROM hosted_artifact_references WHERE kind='tool_model'").fetchall()
        assert len(refs) == 1
        definition = json.loads(con.execute("SELECT definition_json FROM hosted_tool_models WHERE model_id='detector'").fetchone()[0])
        assert "checkpoint" not in definition and definition["checkpoint_sha256"] == refs[0][0]


def test_classifier_artifacts_class_binding_and_private_runtime_state(system):
    install(system)
    install(system, "classifier")
    install(system, "detector")
    store, admin, dataset_id = system["store"], system["admin"], system["dataset"]["id"]
    dataset = store.get_dataset(admin, dataset_id)
    body = {"model_id": "classifier", "expected_revision": dataset["revision"], "operation_id": op()}
    url = f"/api/hosted/tools/datasets/{dataset_id}/classifier"
    selected = system["admin_client"].post(url, json=body)
    assert selected.status_code == 200, selected.text
    assert selected.json()["settings"]["classifier"]["class_mapping"] == {"vehicle": dataset["classes"][0]["id"]}
    with store.read_connection() as con:
        refs = con.execute("SELECT sha256 FROM hosted_artifact_references WHERE kind='tool_model' AND owner_id='classifier'").fetchall()
        assert len(refs) == 2
        definition = json.loads(con.execute("SELECT definition_json FROM hosted_tool_models WHERE model_id='classifier'").fetchone()[0])
        assert definition["metadata_sha256"] in {row[0] for row in refs}
        assert "metadata_path" not in definition
    image = acquire(system, system["images"][0])
    pending = system["client"].post("/api/hosted/tools/request", json=request(image, "sam_point_auto", uuid="hold-class-change"))
    assert pending.status_code == 202, pending.text
    eventually(lambda: (system["gate"] / "hold-class-change.started").exists())
    renamed = store.rename_class(admin, dataset_id, dataset["classes"][0]["id"], "renamed vehicle", op(), expected_class_version=dataset["class_version"])
    # Class identity survives renaming; results captured with old names do not.
    assert renamed["name"] == "renamed vehicle"
    (system["gate"] / "hold-class-change.release").touch()
    assert finish(system["client"], pending.json()["id"])[0]["state"] == "stale"
    assert system["admin_client"].post(url, json=body).json() == selected.json()
    state = system["client"].get("/api/hosted/tools/state", params={"dataset_id": dataset_id})
    assert state.status_code == 200, state.text
    assert state.json()["capabilities"]["sam_point_auto"]["configured"] is True
    assert "pid" not in state.text and "owners" not in state.text and "session_id" not in state.text
    catalog = system["client"].get("/api/hosted/tools/catalog", params={"dataset_id": dataset_id}).json()
    assert "classifier" in {row["model_id"] for row in catalog["models"]}
    dataset = store.get_dataset(admin, dataset_id)
    # Unmatched label names require an explicit admin mapping; no numeric fallback.
    detector_body = {"model_id": "detector", "expected_revision": dataset["revision"], "operation_id": op()}
    detector_url = f"/api/hosted/tools/datasets/{dataset_id}/detector"
    assert system["admin_client"].post(detector_url, json=detector_body).status_code == 422
    detector_body["class_mapping"] = {"vehicle": dataset["classes"][0]["id"]}
    assert system["admin_client"].post(detector_url, json=detector_body).status_code == 200
    spec, _ = system["controller"]._spec(system["alice"], dataset_id, "detect_full")
    assert spec.config["detector"]["class_bindings"]["vehicle"]["name"] == "renamed vehicle"


def test_heavy_uses_confirmed_snapshot_and_durable_no_replay_receipt(system):
    store, alice, admin = system["store"], system["alice"], system["admin"]
    image = acquire(system, system["images"][0])
    def ctx(value, editor="alice-editor"):
        return MutationContext(value["id"], value["current_version"], value["generation"], editor, op())
    image = store.save(alice, ctx(image), {"objects": []}, complete=True, negative=True)
    image = store.submit(alice, ctx(image))
    image = store.acquire_editor(admin, image["id"], "admin-editor", expected_version=image["current_version"], generation=image["generation"])
    confirmed = store.approve(admin, ctx(image, "admin-editor"))
    body = {"dataset_id": system["dataset"]["id"], "job": "fixture", "operation_id": op(), "parameters": {"uuid": "hold-heavy-api"}}
    assert system["client"].post("/api/hosted/tools/heavy", json=body).status_code == 403
    accepted = system["admin_client"].post("/api/hosted/tools/heavy", json=body)
    assert accepted.status_code == 202, accepted.text
    request_id = accepted.json()["id"]
    eventually(lambda: (system["gate"] / "hold-heavy-api.started").exists())
    # Real actor/session/version/lease guarded manual storage remains available
    # while the isolated heavy worker is active; it does not enter that queue.
    second = acquire(system, system["images"][1])
    draft = store.save(alice, ctx(second), {"objects": []}, complete=True, negative=True)
    assert draft["current_version"] != second["current_version"]
    assert system["client"].get("/api/hosted/tools/state").json()["phase"] == "heavy"
    (system["gate"] / "hold-heavy-api.release").touch()
    status, result = finish(system["admin_client"], request_id)
    assert status["state"] == "completed", status
    assert result.json()["result"] == {"images": [confirmed["id"]]}  # Other Incoming excluded.
    retried = system["admin_client"].post("/api/hosted/tools/heavy", json=body)
    assert retried.status_code == 202 and retried.json()["id"] == request_id
    assert retried.json()["state"] == "completed"


def test_qwen_install_is_portable_registered_and_model_choice_restricted(system):
    install(system)
    image = acquire(system, system["images"][0])
    body = request(image, "qwen_caption")
    body["parameters"] = {"user_prompt": "Describe this fixture", "model_id": "qwen"}
    assert system["client"].post("/api/hosted/tools/request", json=body).status_code == 503
    install(system, "qwen")
    accepted = system["client"].post("/api/hosted/tools/request", json=body)
    assert accepted.status_code == 202, accepted.text
    assert finish(system["client"], accepted.json()["id"])[0]["state"] == "completed"
    body["operation"] = "qwen_caption_preview"
    preview = system["client"].post("/api/hosted/tools/request", json=body)
    assert preview.status_code == 202, preview.text
    assert finish(system["client"], preview.json()["id"])[0]["state"] == "completed"
    catalog = system["client"].get("/api/hosted/tools/catalog").json()
    assert catalog["parameters"]["qwen_caption_preview"] == catalog["parameters"]["qwen_caption"]
    body["parameters"]["model_id"] = "unregistered-model"
    assert system["client"].post("/api/hosted/tools/request", json=body).status_code == 403
    body["parameters"] = {"unload_others": True}
    assert system["client"].post("/api/hosted/tools/request", json=body).status_code == 422
    with system["store"].read_connection() as con:
        definition = json.loads(con.execute("SELECT definition_json FROM hosted_tool_models WHERE model_id='qwen'").fetchone()[0])
        refs = {row[0] for row in con.execute("SELECT sha256 FROM hosted_artifact_references WHERE kind='tool_model' AND owner_id='qwen'")}
    assert "model_path" not in definition
    assert {definition["checkpoint_sha256"], *definition["files"].values()} <= refs
    # Lose the configured external source; the installed definition still
    # reconstructs only its registered immutable files on a fresh controller.
    source = Path(system["configuration"]["models"]["qwen"]["model_path"])
    source.rename(source.with_name("source-offline"))
    from services.hosted.tool_api import ToolController
    controller = ToolController(system["store"], system["auth"], system["artifacts"], system["root"], configuration=system["configuration"])
    try:
        definition = controller.registry.definition("qwen")
        for relative, digest in definition["files"].items():
            assert hashlib.sha256((Path(definition["model_path"]) / relative).read_bytes()).hexdigest() == digest
        assert "qwen" in {row["model_id"] for row in controller.catalog(system["alice"])["models"]}
    finally:
        controller.close()


def test_native_heavy_publishes_artifacts_and_recovers_result_without_reexecution(system):
    from services.hosted.tool_api import NATIVE_ANALYSIS_FACTORY, ToolController
    store, alice, admin = system["store"], system["alice"], system["admin"]
    image = acquire(system, system["images"][0])
    def context(value, editor="alice-editor"):
        return MutationContext(value["id"], value["current_version"], value["generation"], editor, op())
    image = store.save(alice, context(image), {"objects": []}, complete=True, negative=True)
    image = store.submit(alice, context(image))
    image = store.acquire_editor(admin, image["id"], "admin-editor", expected_version=image["current_version"], generation=image["generation"])
    store.approve(admin, context(image, "admin-editor"))
    controller = system["controller"]
    native_root = (system["root"] / "native").resolve()
    controller.configuration["heavy_jobs"]["statistics"] = {"factory": NATIVE_ANALYSIS_FACTORY,
        "config": {"native_root": str(native_root)}, "device": "cpu", "operation": "statistics",
        "allowed_parameters": [], "timeout": 30}
    body = {"dataset_id": system["dataset"]["id"], "job": "statistics", "operation_id": op(), "parameters": {}}
    response = system["admin_client"].post("/api/hosted/tools/heavy", json=body)
    assert response.status_code == 202, response.text
    request_id, job_id = response.json()["id"], response.json()["analysis_job_id"]
    status, result = finish(system["admin_client"], request_id)
    assert status["state"] == "completed", status
    assert status["publication_state"] == "complete"
    published = result.json()
    assert published["analysis_job_id"] == job_id and published["result"]["state"] == "complete"
    assert "native_directory" not in result.text
    assert controller.analysis.status(admin, job_id)["state"] == "complete"
    reference = controller.analysis.result_reference(admin, job_id)
    system["artifacts"].verify(reference)
    eventually(lambda: request_id not in controller._pins)
    # Retained artifacts and receipt survive private native worktree loss and a
    # new supervisor. Neither result retrieval nor command retry runs compute.
    import shutil
    shutil.rmtree(native_root)
    controller.close()
    restarted = ToolController(store, system["auth"], system["artifacts"], system["root"], configuration=controller.configuration)
    try:
        assert restarted.result(admin, request_id) == published
        from services.hosted.tool_api import HeavyRequest
        assert restarted.heavy(admin, HeavyRequest(**body))["state"] == "completed"
        assert not native_root.exists()
    finally:
        restarted.close()


def test_ingestion_admission_pins_separate_unassigned_candidates(system, monkeypatch):
    from services.hosted.analysis import read_canonical_snapshot, read_candidate_set
    from services.hosted.runtime import RuntimeRejected
    from services.hosted.tool_api import NATIVE_ANALYSIS_FACTORY
    controller, store, admin = system["controller"], system["store"], system["admin"]
    image_bytes = io.BytesIO()
    Image.new("RGB", (16, 16), "blue").save(image_bytes, "PNG")
    media = system["artifacts"].put_bytes(image_bytes.getvalue())
    image = store.import_image(admin, system["dataset"]["id"], media.sha256, "new.png", {}, op(), width=16, height=16, size_bytes=media.size_bytes)
    controller.configuration["heavy_jobs"]["ingestion"] = {"factory": NATIVE_ANALYSIS_FACTORY,
        "config": {"native_root": str(system["root"] / "native")}, "device": "cpu", "operation": "ingestion",
        "allowed_parameters": ["candidate_image_ids"], "timeout": 30}
    observed = {}
    def inspect_admission(context, spec, operation, payload, **kwargs):
        reference = read_canonical_snapshot(payload["canonical_snapshot"], allowed_root=controller.media_root)
        candidates = read_candidate_set(payload["candidate_set"], allowed_root=controller.media_root)
        assert operation == "ingestion" and reference["images"] == []
        assert [item["id"] for item in candidates["images"]] == [image["id"]]
        assert "candidate_image_ids" not in payload
        assert dict(spec.config["hosted_salad_heads"]) == {}
        observed.update(candidates=candidates)
        raise RuntimeRejected("fixture_stops_before_neural_compute")
    monkeypatch.setattr(controller._ensure_runtime(), "begin_heavy", inspect_admission)
    body = {"dataset_id": system["dataset"]["id"], "job": "ingestion", "operation_id": op(), "parameters": {"candidate_image_ids": [image["id"]]}}
    response = system["admin_client"].post("/api/hosted/tools/heavy", json=body)
    assert response.status_code == 409 and response.json()["error"] == "fixture_stops_before_neural_compute"
    frozen = controller.analysis.get_candidate_set(admin, observed["candidates"]["candidate_set_id"])
    assert frozen["manifest"]["images"][0]["version_id"] == image["current_version"]
    assert "media_path" not in frozen["manifest"]["images"][0]
    assert store.image(admin, image["id"])["state"] == "unassigned"
    body.update(operation_id=op(), parameters={"candidate_image_ids": [system["images"][0]["id"]]})
    assert system["admin_client"].post("/api/hosted/tools/heavy", json=body).json()["error"] == "ingestion_candidate_not_unassigned_incoming"


def test_caption_jobs_retry_quality_preserve_evidence_and_never_save_annotations(system):
    install(system)
    install(system, "qwen")
    image = acquire(system, system["images"][0])
    body = {key: value for key, value in request(image).items() if key not in {"operation", "parameters"}}
    body.update(operation_id=op(), caption_request={"user_prompt": "retry-fixture"},
        settings={"attempts": 2, "cooldown_after_crash_seconds": 0, "max_failed_attempt_row_rate": 1})
    url = "/api/hosted/tools/caption-jobs"
    response = system["client"].post(url, json=body)
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    result = eventually(lambda: (row if (row := system["client"].get(url + "/" + job_id).json()).get("status") in {"completed", "failed"} else None))
    assert result["status"] == "completed", result
    assert result["result"]["attempts_seen"] == 2
    assert result["result"]["latest_caption"]["caption"] == "A red image."
    assert result["result"]["saved_text_labels"] == 0
    unchanged = system["store"].image(system["alice"], image["id"])
    assert unchanged["current_version"] == image["current_version"]
    with system["store"].read_connection() as con:
        row = con.execute("SELECT attempts_json FROM hosted_caption_jobs WHERE id=?", (job_id,)).fetchone()
        attempts = json.loads(row[0])
    assert attempts[0]["quality_failures"] == ["prompt leak"]
    assert attempts[0]["final_status"] == "failed_attempt" and attempts[1]["final_status"] == "ok"
    from services.hosted.artifacts import ArtifactRef
    for attempt in attempts:
        system["artifacts"].verify(ArtifactRef(**attempt["evidence"]))
    retry = system["client"].post(url, json=body)
    assert retry.status_code == 202 and retry.json()["job_id"] == job_id
    assert retry.json()["result"]["attempts_seen"] == 2
    assert system["admin_client"].get(url + "/" + job_id).status_code == 404
    mutation = MutationContext(image["id"], image["current_version"], image["generation"], "alice-editor", op())
    system["store"].save(system["alice"], mutation, {"objects": []}, complete=True, negative=True)
    assert system["client"].get(url + "/" + job_id + "/result").status_code == 409


def test_machine_qualified_qa_requires_exact_durable_job_and_survives_edits(system):
    import copy
    from services.hosted.caption_export import _effective_review
    install(system)
    install(system, "qwen")
    image = acquire(system, system["images"][0])
    body = {key: value for key, value in request(image).items() if key not in {"operation", "parameters"}}
    body.update(operation_id=op(), caption_request={}, settings={"instruction_dataset": True, "subcaptions_per_image": 1})
    response = system["client"].post("/api/hosted/tools/caption-jobs", json=body)
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    job = eventually(lambda: (value if (value := system["client"].get("/api/hosted/tools/caption-jobs/" + job_id).json()).get("status") in {"completed", "failed"} else None))
    assert job["status"] == "completed", job
    generated = job["result"]["latest_caption"]
    pair = generated["generated_qa_pairs"][0]
    assert pair["metadata"]["machine_validation_content_sha256"]
    annotations = {"objects": [], "alternate_captions": [{"id": "generated-record", "caption": generated["caption"],
        "backend_job_id": job_id, "generated_qa_pairs": copy.deepcopy(generated["generated_qa_pairs"]),
        "caption_quality": generated["caption_quality"], "evidence": job["result"]["evidence"]}]}
    def mutation(value):
        return MutationContext(value["id"], value["current_version"], value["generation"], "alice-editor", op())
    forged = copy.deepcopy(annotations)
    forged["alternate_captions"][0]["generated_qa_pairs"][0]["answer"] = "Forged qualification."
    with pytest.raises(DomainError, match="machine_validation_requires_server_evidence"):
        system["store"].save(system["alice"], mutation(image), forged)
    unknown = copy.deepcopy(annotations)
    unknown["alternate_captions"][0]["backend_job_id"] = "does-not-exist"
    with pytest.raises(DomainError, match="machine_validation_requires_server_evidence"):
        system["store"].save(system["alice"], mutation(image), unknown)
    other = acquire(system, system["images"][1])
    with pytest.raises(DomainError, match="machine_validation_source_changed"):
        system["store"].save(system["alice"], mutation(other), annotations)
    saved = system["store"].save(system["alice"], mutation(image), annotations)
    assert system["store"].image(system["alice"], image["id"])["annotations"]["alternate_captions"][0]["generated_qa_pairs"] == generated["generated_qa_pairs"]
    # A later correction keeps original evidence, while losing qualification
    # for the changed answer. It does not trap the annotator in a failed save.
    edited = copy.deepcopy(annotations)
    edited["alternate_captions"][0]["generated_qa_pairs"][0]["answer"] = "Blue."
    system["store"].save(system["alice"], mutation(saved), edited)
    current_pair = system["store"].image(system["alice"], image["id"])["annotations"]["alternate_captions"][0]["generated_qa_pairs"][0]
    assert current_pair["metadata"] == pair["metadata"]
    assert _effective_review(current_pair, "generated_qa")["validation_status"] != "machine_validated"


def test_caption_job_cancel_resume_and_health_gate_are_durable(system):
    install(system)
    install(system, "qwen")
    image = acquire(system, system["images"][0])
    body = {key: value for key, value in request(image).items() if key not in {"operation", "parameters"}}
    body.update(operation_id=op(), caption_request={"user_prompt": "hold-caption-job"}, settings={"attempts": 1})
    url = "/api/hosted/tools/caption-jobs"
    response = system["client"].post(url, json=body)
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    eventually(lambda: (system["gate"] / "hold-caption-job.started").exists())
    cancelled = system["client"].post(url + "/" + job_id + "/cancel", json={"operation_id": op()})
    assert cancelled.status_code == 200, cancelled.text
    eventually(lambda: system["client"].get(url + "/" + job_id).json()["status"] == "cancelled")
    (system["gate"] / "hold-caption-job.release").touch()
    body["operation_id"] = op()
    resumed = system["client"].post(url + "/" + job_id + "/resume", json=body)
    assert resumed.status_code == 202, resumed.text
    eventually(lambda: system["client"].get(url + "/" + job_id).json()["status"] == "completed")
    assert system["client"].post(url + "/" + job_id + "/resume", json=body).json()["status"] == "completed"
    body.update(operation_id=op(), caption_request={"user_prompt": "recovery-fixture"})
    gated = system["client"].post(url, json=body)
    assert gated.status_code == 202, gated.text
    gated_id = gated.json()["job_id"]
    failed = eventually(lambda: (value if (value := system["client"].get(url + "/" + gated_id).json()).get("status") == "failed" else None))
    assert failed["error"] == "caption_health_gate_failed"
    assert failed["result"]["latest_caption"]["caption"] == "A red image."
    assert failed["result"]["health"]["violations"][0]["rate"] == "recovery_event_case_rate"


def test_edr_package_is_portable_pinned_and_shares_one_stable_dataset_queue(system):
    install(system, "detector")
    controller = system["controller"]
    source = Path(system["configuration"]["models"]["detector"]["checkpoint"]).parent / "edr"
    source.mkdir()
    recipe = source / "recipe.json"
    recipe.write_text('{"fixture":"package transport only; no ensemble loaded"}')
    controller.configuration["models"]["edr"] = {"kind": "edr", "model_path": str(source),
        "files": {"recipe.json": hashlib.sha256(recipe.read_bytes()).hexdigest()},
        "recipe_file": "recipe.json", "labels": ["vehicle"], "components": {"yolo": "detector"}}
    install(system, "edr")
    dataset_id = system["dataset"]["id"]
    for kind, model in (("detector", "detector"), ("edr", "edr")):
        dataset = system["store"].get_dataset(system["admin"], dataset_id)
        selected = system["admin_client"].post(f"/api/hosted/tools/datasets/{dataset_id}/{kind}",
            json={"model_id": model, "expected_revision": dataset["revision"], "operation_id": op()})
        assert selected.status_code == 200, selected.text
    direct, owner = controller._spec(system["alice"], dataset_id, "detect_full")
    ensemble, second_owner = controller._spec(system["alice"], dataset_id, "edr_prepass")
    assert owner == second_owner == "dataset" and direct.identity == ensemble.identity
    package = ensemble.config["edr"]["model"]
    assert package["component_models"]["yolo"]["checkpoint_sha256"] == controller.registry.definition("detector")["checkpoint_sha256"]
    with system["store"].read_connection() as con:
        definition = json.loads(con.execute("SELECT definition_json FROM hosted_tool_models WHERE model_id='edr'").fetchone()[0])
        registered = {row[0] for row in con.execute("SELECT sha256 FROM hosted_artifact_references WHERE kind='tool_model' AND owner_id='edr'")}
    assert "model_path" not in definition
    assert {definition["checkpoint_sha256"], *definition["files"].values()} <= registered
    # A changed component is never silently substituted into an installed
    # ensemble, even if its user-facing model ID remains the same.
    replacement = controller.model_root / "replacement.pt"
    replacement.write_bytes(b"new fixture detector version")
    controller.registry.register("detector", {"kind": "yolo", "checkpoint": str(replacement), "labels": ["vehicle"]})
    with pytest.raises(DomainError, match="tool_model_selection_changed|tool_edr_component_changed"):
        controller._spec(system["alice"], dataset_id, "edr_prepass")


def test_caption_restart_waits_for_real_lease_reacquire_and_consumes_resume_budget(system, monkeypatch):
    from services.hosted.tool_api import ToolController
    install(system)
    install(system, "qwen")
    image = acquire(system, system["images"][0])
    body = {key: value for key, value in request(image).items() if key not in {"operation", "parameters"}}
    body.update(operation_id=op(), caption_request={"user_prompt": "hold-caption-restart"},
        settings={"attempts": 1, "max_auto_resumes": 1, "max_failed_attempt_row_rate": 1})
    response = system["client"].post("/api/hosted/tools/caption-jobs", json=body)
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    eventually(lambda: (system["gate"] / "hold-caption-restart.started").exists())
    system["controller"].close()
    with system["store"].transaction() as con:
        con.execute("UPDATE hosted_editor_leases SET expires_at=0 WHERE image_id=?", (image["id"],))
    restarted = ToolController(system["store"], system["auth"], system["artifacts"], system["root"], configuration=system["configuration"])
    original = restarted._spec
    def fixture_spec(actor, dataset_id, operation):
        spec, owner = original(actor, dataset_id, operation)
        config = json.loads(json.dumps(dict(spec.config), default=lambda value: dict(value) if hasattr(value, "items") else list(value)))
        return AdapterSpec("test_hosted_tools_api:FixtureTools", {**config, "fixture_gate_root": str(system["gate"])}, "cpu"), owner
    monkeypatch.setattr(restarted, "_spec", fixture_spec)
    try:
        state = restarted.caption_jobs.status(system["alice"], job_id)
        assert state["status"] == "waiting_for_editor" and state["auto_resume_count"] == 0
        assert state["error"] == "editor_lease_expired"
        (system["gate"] / "hold-caption-restart.release").touch()
        fresh = system["store"].acquire_editor(system["alice"], image["id"], "alice-new-editor",
            expected_version=image["current_version"], generation=image["generation"])
        resumed = restarted.caption_jobs.resume_waiting(system["alice"], image["id"], "alice-new-editor",
            previous_generation=image["generation"], generation=fresh["generation"], expected_version=fresh["current_version"])
        assert resumed == [job_id]
        done = eventually(lambda: (value if (value := restarted.caption_jobs.status(system["alice"], job_id))["status"] in {"completed", "failed"} else None))
        assert done["status"] == "completed", done
        assert done["auto_resume_count"] == 1 and done["context"]["generation"] == fresh["generation"]
        assert done["result"]["attempts_seen"] == 2
        assert system["store"].image(system["alice"], image["id"])["current_version"] == image["current_version"]
        assert restarted.caption_jobs._automatic_resume(job_id) is None
    finally:
        restarted.close()


def test_caption_batch_uses_frozen_confirmed_members_and_real_health_gates(system, monkeypatch):
    install(system, "qwen")
    store, admin, client = system["store"], system["admin"], system["admin_client"]
    # Incoming and assigned records remain outside the approved snapshot.
    content = io.BytesIO()
    Image.new("RGB", (40, 30), "red").save(content, format="PNG")
    ref = system["artifacts"].put_bytes(content.getvalue())
    image = store.import_image(admin, system["dataset"]["id"], ref.sha256, "confirmed.png", {}, op(), width=40, height=30, size_bytes=ref.size_bytes)
    image = store.acquire_editor(admin, image["id"], "admin-editor")
    image = store.save(admin, MutationContext(image["id"], image["current_version"], image["generation"], "admin-editor", op()), {"objects": []}, complete=True, negative=True)
    image = store.approve(admin, MutationContext(image["id"], image["current_version"], image["generation"], "admin-editor", op()))
    snapshot = store.create_snapshot(admin, image["dataset_id"], op(), "Caption fixture")
    batches = system["controller"].caption_batches
    original = batches._spec
    def fixture_spec(actor, qwen, directory):
        spec = original(actor, qwen, directory)
        return AdapterSpec("test_hosted_tools_api:FixtureCaptionBatch", spec.config, "cpu")
    monkeypatch.setattr(batches, "_spec", fixture_spec)
    body = {"dataset_id": image["dataset_id"], "snapshot_id": snapshot["snapshot_id"], "operation_id": op(),
            "sample_size": 0, "settings": {"attempts": 1}, "pilot": {"min_cases": 1, "deterministic_recovery_confidence": 0}}
    url = "/api/hosted/tools/caption-batches"
    assert system["client"].post(url, json=body).status_code == 403
    response = client.post(url, json=body)
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    done = eventually(lambda: (value if (value := client.get(url + "/" + job_id).json()).get("status") in {"completed", "failed"} else None))
    assert done["status"] == "completed", done
    assert done["result"]["certification"]["status"] == "certified", done
    assert done["result"]["total_cases"] == 1 and done["result"]["saved_text_labels"] == 0
    captions = client.get(url + "/" + job_id + "/captions").json()
    assert [row["image_id"] for row in captions["captions"]] == [image["id"]]
    assert captions["captions"][0]["caption"] == "A red square on a plain surface."
    assert store.image(admin, image["id"])["current_version"] == image["current_version"]
    assert client.post(url, json=body).json()["job_id"] == job_id
    # Native Wilson gate remains strict; finite zero-event samples cannot
    # certify an exact zero true recovery rate with positive confidence.
    body["operation_id"] = op()
    body["pilot"]["deterministic_recovery_confidence"] = .95
    again = client.post(url, json=body)
    assert again.status_code == 202, again.text
    second = eventually(lambda: (value if (value := client.get(url + "/" + again.json()["job_id"]).json()).get("status") in {"completed", "failed"} else None))
    assert second["status"] == "completed", second
    assert second["result"]["certification"]["status"] == "not_certified"
    body.update(operation_id=op(), certified_pilot_id=again.json()["job_id"], pilot=None)
    assert client.post(url, json=body).status_code == 409
    proposal_url = url + "/" + job_id + "/images/" + image["id"]
    proposal = client.get(proposal_url + "/proposal").json()
    current = store.acquire_editor(admin, image["id"], "admin-caption-review", expected_version=image["current_version"], generation=image["generation"])
    apply_body = {"expected_version": current["current_version"], "generation": current["generation"],
        "editor_id": "admin-caption-review", "operation_id": op(), "proposal_sha256": proposal["proposal_sha256"]}
    applied = client.post(proposal_url + "/apply", json=apply_body)
    assert applied.status_code == 200, applied.text
    annotations = store.image(admin, image["id"])["annotations"]
    assert annotations["caption"] == proposal["caption"] and annotations["objects"] == []
    assert annotations["alternate_captions"][-1]["evidence_sha256"] == proposal["proposal_sha256"]
    assert annotations["alternate_captions"][-1]["caption_quality"] == proposal["quality"]
    assert annotations["alternate_captions"][-1]["provider"] == "local_qwen"
    assert client.post(proposal_url + "/apply", json=apply_body).json() == applied.json()
    apply_body["operation_id"] = op()
    assert client.post(proposal_url + "/apply", json=apply_body).status_code == 409


def test_caption_batch_restart_reuses_retained_success_without_generating_it_again(system, monkeypatch):
    from services.hosted.caption_batch import CaptionBatchRequest
    from services.hosted.tool_api import ToolController
    store, admin = system["store"], system["admin"]
    install(system, "qwen")
    for color in ("blue", "yellow"):
        stream = io.BytesIO()
        Image.new("RGB", (20, 20), color).save(stream, format="PNG")
        ref = system["artifacts"].put_bytes(stream.getvalue())
        image = store.import_image(admin, system["dataset"]["id"], ref.sha256, color + ".png", {}, op(), width=20, height=20, size_bytes=ref.size_bytes)
        image = store.acquire_editor(admin, image["id"], "admin-editor")
        image = store.save(admin, MutationContext(image["id"], image["current_version"], image["generation"], "admin-editor", op()), {"objects": []}, complete=True, negative=True)
        store.approve(admin, MutationContext(image["id"], image["current_version"], image["generation"], "admin-editor", op()))
    snapshot = store.create_snapshot(admin, system["dataset"]["id"], op(), "Recovery fixture")
    images = store.get_snapshot(admin, snapshot["snapshot_id"])["manifest"]["images"]
    first, second = [image["id"] for image in images]
    def bind(controller):
        original = controller.caption_batches._spec
        def spec(actor, qwen, directory):
            value = original(actor, qwen, directory)
            return AdapterSpec("test_hosted_tools_api:FixtureCaptionBatch", {**dict(value.config),
                "fixture_gate": str(system["gate"]), "fixture_hold_image": second}, "cpu")
        monkeypatch.setattr(controller.caption_batches, "_spec", spec)
    bind(system["controller"])
    body = CaptionBatchRequest(dataset_id=system["dataset"]["id"], snapshot_id=snapshot["snapshot_id"], operation_id=op(), sample_size=0,
        settings={"attempts": 2, "max_auto_resumes": 1, "max_failed_attempt_row_rate": 1}, pilot=None)
    job = system["controller"].caption_batches.start(admin, body)
    eventually(lambda: (system["gate"] / "batch-held").exists())
    eventually(lambda: len(system["controller"].caption_batches._attempts(job["job_id"])) == 1)
    system["controller"].close()
    (system["gate"] / "batch-release").touch()
    restarted = ToolController(store, system["auth"], system["artifacts"], system["root"], configuration=system["configuration"])
    bind(restarted)
    try:
        state = restarted.caption_batches.status(admin, job["job_id"])
        finished = eventually(lambda: (value if (value := restarted.caption_batches.status(admin, job["job_id"]))["status"] in {"completed", "failed"} else None))
        assert finished["status"] == "completed", finished
        assert finished["auto_resume_count"] == 1
        assert finished["result"]["attempts_seen"] == 3
        assert (system["gate"] / (first + ".calls")).read_text().count("called") == 1
        assert (system["gate"] / (second + ".calls")).read_text().count("called") == 2
        assert not (restarted.caption_batches.root / job["job_id"]).exists()
        # Retrying the committed resume must resolve its receipt even if the
        # configured model subsequently changes; it must not execute again.
        def changed_configuration(*_):
            raise DomainError("caption_batch_configuration_changed", 409)
        monkeypatch.setattr(restarted.caption_batches, "_validate", changed_configuration)
        replay = restarted.caption_batches.resume(admin, job["job_id"], f"batch-auto-{job['job_id']}-1", automatic=True)
        assert replay["status"] == "completed" and replay["auto_resume_count"] == 1
    finally:
        restarted.close()
