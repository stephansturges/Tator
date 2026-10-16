"""Real native Batch recipe/accounting; fixed-host network replaced locally."""
import json
import sqlite3
import threading
from pathlib import Path
import pytest

from test_hosted_tools_api import system, acquire, eventually, op
from test_hosted_paid_caption import configure_paid
from services.hosted.errors import DomainError


class Transport:
    def __init__(self):
        self.files, self.batches, self.requests = {}, {}, []
        self.fail_create = False
        self.before_create = None

    def request(self, secret, method, path, *, body=None, upload=None):
        assert secret == "fake-local-test-key"
        self.requests.append((method, path, body, upload))
        if path == "/files":
            name, data, purpose = upload
            if hasattr(data, "read"):
                data = data.read()
            file_id = "file_" + str(len(self.files))
            self.files[file_id] = data
            return {"id": file_id, "purpose": purpose}
        if path == "/batches":
            if self.before_create:
                self.before_create()
            lines = [json.loads(line) for line in self.files[body["input_file_id"]].splitlines()]
            outputs = []
            for line in lines:
                count = line["body"]["text"]["format"]["schema"]["properties"]["qa_pairs"]["minItems"]
                qa = [{"question": f"What detail {index} is visible?", "answer": "A red car."} for index in range(count)]
                outputs.append({"custom_id": line["custom_id"], "response": {"status_code": 200,
                    "body": {"id": "response_" + line["custom_id"], "status": "completed",
                        "output_text": json.dumps({"caption": "A red car on a paved road.", "qa_pairs": qa}),
                        "usage": {"input_tokens": 30, "output_tokens": 20}}}})
            output_id = "file_output_" + str(len(self.batches))
            self.files[output_id] = b"".join((json.dumps(row) + "\n").encode() for row in outputs)
            remote_id = "batch_" + str(len(self.batches))
            result = {**body, "id": remote_id, "status": "completed", "output_file_id": output_id,
                "request_counts": {"total": len(lines), "completed": len(lines), "failed": 0}}
            self.batches[remote_id] = result
            if self.fail_create:
                raise TimeoutError("response lost after provider accepted")
            return result
        if path.endswith("/content"):
            return self.files[path.split("/")[2]]
        if path.startswith("/batches/"):
            result = self.batches[path.split("/")[2]]
            return result
        raise AssertionError((method, path))


def configure(system):
    gateway, _ = configure_paid(system)
    gateway.pricing["test-paid-model"]["batch"] = dict(gateway.pricing["test-paid-model"])
    transport = Transport()
    system["controller"].paid_batches.transport = transport
    return gateway, transport


def body_for(image):
    return {"dataset_id": image["dataset_id"], "operation_id": op(),
        "image_context": {"image_id": image["id"], "expected_version": image["current_version"],
            "generation": image["generation"], "editor_id": "alice-editor"},
        "provider_options": {"model": "test-paid-model", "max_output_tokens": 4096},
        "settings": {"instruction_dataset": True, "subcaptions_per_image": 2}}


def test_batch_exact_native_prompt_upload_collect_and_explicit_apply(system):
    gateway, transport = configure(system)
    image = acquire(system, system["images"][0])
    body, client = body_for(image), system["client"]
    url = "/api/hosted/tools/paid-batches"
    preview = client.post(url + "/preview", json=body)
    assert preview.status_code == 200, preview.text
    assert "Generate exactly 2 question-answer pairs" in preview.json()["full_text"]
    result = client.post(url, json=body)
    assert result.status_code == 202, result.text
    job_id = result.json()["job_id"]
    done = eventually(lambda: (row if (row := client.get(url + "/" + job_id).json()).get("status") in {"collected", "failed", "submission_unknown"} else None))
    assert done["status"] == "collected", done
    assert done["result"]["accepted_cases"] == 1
    assert len(transport.batches) == 1
    ledger = gateway.accounts.calls(system["admin"])
    assert len(ledger) == 1 and ledger[0]["state"] == "settled" and ledger[0]["charged"] > 0
    proposals = client.get(url + "/" + job_id + "/captions").json()["captions"]
    assert len(proposals[0]["generated_qa_pairs"]) == 2
    assert client.post(url + "/" + job_id + "/verify", json={}).json()["ok"]
    assert system["store"].image(system["alice"], image["id"])["current_version"] == image["current_version"]
    command = {**body["image_context"], "operation_id": op(), "proposal_sha256": proposals[0]["proposal_sha256"]}
    saved = client.post(url + "/" + job_id + "/apply", json=command)
    assert saved.status_code == 200, saved.text
    annotations = system["store"].image(system["alice"], image["id"])["annotations"]
    assert annotations["caption"] == proposals[0]["caption"]
    assert annotations["alternate_captions"][-1]["generated_qa_pairs"] == proposals[0]["generated_qa_pairs"]
    assert annotations["alternate_captions"][-1]["provider_call_id"] == ledger[0]["id"]
    assert annotations["alternate_captions"][-1]["evidence"] == done["result"]["raw_evidence"]
    assert client.post(url + "/" + job_id + "/apply", json=command).json() == saved.json()
    assert not client.post(url + "/" + job_id + "/verify", json={}).json()["ok"]
    assert client.post(url + "/" + job_id + "/archive", json={"operation_id": op()}).json()["archived"]
    assert not client.post(url + "/" + job_id + "/archive", json={"operation_id": op(), "archived": False}).json()["archived"]
    assert client.post(url, json=body).json()["job_id"] == job_id
    assert len(transport.batches) == 1
    # The same retained output must survive recovery with every secondary
    # artifact index bound to the central backup reference registry.
    from services.hosted.operations import HostedOperations
    with system["store"].read_connection() as con:
        references = HostedOperations.artifact_references(con)
        restored = sqlite3.connect(":memory:")
        con.backup(restored)
    try:
        HostedOperations.restore_hook(restored, {"artifacts": references})
        digest = done["result"]["raw_evidence"][0]["sha256"]
        restored.execute("DELETE FROM hosted_artifact_references WHERE kind='paid_batch' AND owner_id=? AND sha256=?", (job_id, digest))
        with pytest.raises(DomainError, match="recovery_paid_batch_missing"):
            HostedOperations.restore_hook(restored, {"artifacts": references})
    finally:
        restored.close()
    # A lost apply reply remains resolvable after explicit administrative recall.
    current = system["store"].image(system["alice"], image["id"])
    system["store"].recall_images(system["admin"], image["dataset_id"], [image["id"]],
        {image["id"]: {"expected_version": current["current_version"], "generation": current["generation"]}},
        "Fixture recall after committed caption", op())
    replay = client.post(url + "/" + job_id + "/apply", json=command)
    assert replay.status_code == 403 and replay.json()["error"] == "committed_result_restricted"
    receipt = system["store"].receipt(system["alice"], command["operation_id"])
    assert receipt["commit_outcome"] == "committed" and receipt["result_restricted"] and receipt["result"] is None
    changed_command = {**command, "include_generated_qa": False}
    mismatch = client.post(url + "/" + job_id + "/apply", json=changed_command)
    assert mismatch.status_code == 409 and mismatch.json()["error"] == "operation_id_reused"


def test_unknown_batch_submission_never_resends_and_recovers_only_bound_remote_id(system):
    gateway, transport = configure(system)
    transport.fail_create = True
    image = acquire(system, system["images"][0])
    body, client = body_for(image), system["client"]
    url = "/api/hosted/tools/paid-batches"
    result = client.post(url, json=body)
    assert result.status_code == 202, result.text
    job_id = result.json()["job_id"]
    eventually(lambda: client.get(url + "/" + job_id).json().get("status") == "submission_unknown")
    resume = client.post(url + "/" + job_id + "/resume", json={"operation_id": op()})
    assert resume.status_code == 409 and resume.json()["error"] == "paid_batch_submission_unresolved"
    assert len(transport.batches) == 1
    transport.fail_create = False
    remote_id = next(iter(transport.batches))
    transport.batches[remote_id]["metadata"]["hosted_job_id"] = "wrong-job"
    bad = client.post(url + "/" + job_id + "/recover", json={"operation_id": op(), "remote_id": remote_id})
    assert bad.status_code == 409
    transport.batches[remote_id]["metadata"]["hosted_job_id"] = job_id
    recovered = client.post(url + "/" + job_id + "/recover", json={"operation_id": op(), "remote_id": remote_id})
    assert recovered.status_code == 202, recovered.text
    done = eventually(lambda: (row if (row := client.get(url + "/" + job_id).json()).get("status") in {"collected", "failed"} else None))
    assert done["status"] == "collected", done
    assert len(transport.batches) == 1
    assert gateway.accounts.calls(system["admin"])[0]["state"] == "settled"


def test_batch_source_change_during_upload_prevents_any_inference_submission(system):
    from services.hosted.types import MutationContext
    gateway, transport = configure(system)
    image = acquire(system, system["images"][0])
    upload_entered, release = threading.Event(), threading.Event()
    original = transport.request
    def paused(secret, method, path, **kwargs):
        value = original(secret, method, path, **kwargs)
        if path == "/files" and kwargs.get("upload", (None, None, None))[2] == "vision":
            upload_entered.set()
            assert release.wait(10)
        return value
    transport.request = paused
    url, client = "/api/hosted/tools/paid-batches", system["client"]
    response = client.post(url, json=body_for(image))
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    try:
        assert upload_entered.wait(10)
        system["store"].save(system["alice"], MutationContext(image["id"], image["current_version"], image["generation"], "alice-editor", op()), {"objects": [], "caption": "New manual text"})
    finally:
        release.set()
    done = eventually(lambda: (row if (row := client.get(url + "/" + job_id).json()).get("status") == "failed" else None))
    assert done["error"] == "annotation_version_changed"
    assert not transport.batches
    assert gateway.accounts.calls(system["admin"])[0]["charged"] == 0


def test_batch_restart_collects_already_terminal_remote_without_resubmitting(system, monkeypatch):
    gateway, transport = configure(system)
    controller = system["controller"].paid_batches
    image = acquire(system, system["images"][0])
    native_collect = controller._collect
    # Model abrupt controller loss after remote completion was persisted, but
    # before any output or usage was retained. No synthetic settled state.
    def stop_before_collection(job_id):
        controller.stop.set()
    monkeypatch.setattr(controller, "_collect", stop_before_collection)
    url, client = "/api/hosted/tools/paid-batches", system["client"]
    response = client.post(url, json=body_for(image))
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    eventually(lambda: controller.stop.is_set() and not controller.threads)
    assert controller._row(None, job_id)["state"] == "completed"
    assert gateway.accounts.calls(system["admin"])[0]["state"] == "dispatched"
    monkeypatch.setattr(controller, "_collect", native_collect)
    controller.stop.clear()
    controller.recovered = False
    done = eventually(lambda: (row if (row := client.get(url + "/" + job_id).json()).get("status") == "collected" else None))
    assert done["result"]["accepted_cases"] == 1
    assert len(transport.batches) == 1
    assert gateway.accounts.calls(system["admin"])[0]["state"] == "settled"


def test_admin_batch_uses_all_frozen_images_and_keeps_source_versions_until_apply(system):
    from services.hosted.types import MutationContext
    gateway, transport = configure(system)
    store, admin, alice = system["store"], system["admin"], system["alice"]
    confirmed = []
    for source in system["images"]:
        image = acquire(system, source)
        def context(value, editor="alice-editor"):
            return MutationContext(value["id"], value["current_version"], value["generation"], editor, op())
        image = store.save(alice, context(image), {"objects": []}, complete=True, negative=True)
        image = store.submit(alice, context(image))
        image = store.acquire_editor(admin, image["id"], "admin-editor", expected_version=image["current_version"], generation=image["generation"])
        confirmed.append(store.approve(admin, context(image, "admin-editor")))
    snapshot = store.create_snapshot(admin, system["dataset"]["id"], op())
    body = {"dataset_id": system["dataset"]["id"], "snapshot_id": snapshot["snapshot_id"], "operation_id": op(),
        "provider_options": {"model": "test-paid-model", "max_output_tokens": 4096},
        "settings": {"instruction_dataset": True, "subcaptions_per_image": 2}}
    url = "/api/hosted/tools/paid-batches"
    assert system["client"].post(url, json=body).status_code == 403
    response = system["admin_client"].post(url, json=body)
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    done = eventually(lambda: (row if (row := system["admin_client"].get(url + "/" + job_id).json()).get("status") in {"collected", "failed"} else None))
    assert done["status"] == "collected", done
    assert done["result"]["accepted_cases"] == 2
    assert len(gateway.accounts.calls(admin)) == 2 and len(transport.batches) == 1
    for image in confirmed:
        assert store.image(admin, image["id"])["current_version"] == image["current_version"]


def test_remote_running_batches_still_count_toward_pending_limit(system):
    _, transport = configure(system)
    native_request = transport.request
    def keep_running(*args, **kwargs):
        result = native_request(*args, **kwargs)
        if isinstance(result, dict) and str(result.get("id", "")).startswith("batch_"):
            result["status"] = "in_progress"
        return result
    transport.request = keep_running
    image = acquire(system, system["images"][0])
    client, url = system["client"], "/api/hosted/tools/paid-batches"
    for _ in range(8):
        response = client.post(url, json=body_for(image))
        assert response.status_code == 202, response.text
        job_id = response.json()["job_id"]
        eventually(lambda: client.get(url + "/" + job_id).json()["status"] == "in_progress")
    blocked = client.post(url, json=body_for(image))
    assert blocked.status_code == 429 and blocked.json()["error"] == "paid_batch_queue_full"
    assert len(transport.batches) == 8


def test_oversize_native_prompt_is_rejected_before_reservation_or_upload(system, monkeypatch):
    from services.hosted import paid_batches
    gateway, transport = configure(system)
    image = acquire(system, system["images"][0])
    monkeypatch.setattr(paid_batches, "MAX_BATCH_INPUT_BYTES", 256)
    response = system["client"].post("/api/hosted/tools/paid-batches", json=body_for(image))
    assert response.status_code == 422 and response.json()["error"] == "paid_batch_input_too_large"
    assert not gateway.accounts.calls(system["admin"])
    assert not transport.requests
    with system["store"].read_connection() as con:
        assert not con.execute("SELECT 1 FROM hosted_paid_batches").fetchone()


def _oriented_batch_image(system, encoding):
    import io
    from PIL import Image
    picture = Image.new('RGB', (64, 48), 'red')
    picture.paste('blue', (0, 0, 24, 48))
    exif = Image.Exif(); exif[274] = 6
    data = io.BytesIO(); picture.save(data, format=encoding, exif=exif)
    original = data.getvalue()
    view = system['controller'].media.stage(original)
    store, admin = system['store'], system['admin']
    image = store.import_image(admin, system['dataset']['id'], view['source_sha256'], 'oriented.' + encoding.lower(), {}, op(),
        width=64, height=48, size_bytes=len(original))
    assigned = store.assign(admin, image['dataset_id'], [image['id']], system['alice'].user_id, op(),
        expected={image['id']: {'expected_version': image['current_version'], 'generation': image['generation']}})
    return acquire(system, assigned['images'][0]), original, view


@pytest.mark.parametrize('encoding', ['JPEG', 'PNG', 'WEBP', 'TIFF'])
def test_paid_batch_uploads_encoded_working_view_and_keeps_source_identity(system, encoding):
    import hashlib
    import io
    from PIL import Image, ImageOps
    from services.hosted.media import decode_encoded
    from services.hosted.paid_batches import _digest
    gateway, transport = configure(system)
    image, original, view = _oriented_batch_image(system, encoding)
    client, url = system['client'], '/api/hosted/tools/paid-batches'
    body = body_for(image)
    started = client.post(url, json=body)
    assert started.status_code == 202, started.text
    job_id = started.json()['job_id']
    done = eventually(lambda: row if (row := client.get(url + '/' + job_id).json()).get('status') in {'collected', 'failed', 'submission_unknown'} else None)
    assert done['status'] == 'collected', done
    uploads = [request[3] for request in transport.requests if request[3] and request[3][2] == 'vision']
    assert len(uploads) == 1
    name, data, _ = uploads[0]
    assert name.endswith('.png')
    assert hashlib.sha256(data).hexdigest() == view['sha256'] != view['source_sha256']
    expected, _ = decode_encoded(original)
    with Image.open(io.BytesIO(data)) as actual:
        assert actual.size == ImageOps.exif_transpose(actual).size == (64, 48)
        assert actual.getexif().get(274, 1) == 1
        assert actual.convert('RGB').tobytes() == expected.tobytes()
    expected.close()
    with system['store'].read_connection() as con:
        retained = con.execute('SELECT * FROM hosted_paid_batch_rows WHERE job_id=?', (job_id,)).fetchone()
        prepared = json.loads(retained['request_json'])
        assert prepared['source_media_hash'] == view['source_sha256']
        assert prepared['working_media'] == view
        assert retained['upload_sha256'] == view['sha256']
        price = json.loads(con.execute('SELECT pricing_json FROM hosted_paid_batches WHERE id=?', (job_id,)).fetchone()[0])
        call = con.execute('SELECT * FROM hosted_provider_calls WHERE id=?', (retained['call_id'],)).fetchone()
        assert call['request_hash'] == _digest({'prepared': prepared, 'pricing': price, 'transport': 'openai_batch'})
        assert call['state'] == 'settled' and call['charged'] > 0
    with system['artifacts'].open(view['source_sha256']) as stream:
        assert stream.read() == original
    current = system['store'].image(system['alice'], image['id'])
    assert current['media_hash'] == view['source_sha256'] and current['current_version'] == image['current_version']
    assert client.post(url, json=body).json()['job_id'] == job_id
    assert len(transport.batches) == 1
    assert len([request for request in transport.requests if request[3] and request[3][2] == 'vision']) == 1


def test_paid_batch_replaces_unsent_legacy_oriented_upload_without_new_reservation(system, monkeypatch):
    """An old original-byte upload and JSONL must not survive into a new dispatch."""
    from dataclasses import replace
    from services.hosted.paid_batches import PaidBatchRequest, _digest, _json
    gateway, transport = configure(system)
    image, original, view = _oriented_batch_image(system, 'JPEG')
    ctl = system['controller'].paid_batches
    launch = ctl._launch
    monkeypatch.setattr(ctl, '_launch', lambda job_id: None)
    body = body_for(image)
    started = system['client'].post('/api/hosted/tools/paid-batches', json=body)
    assert started.status_code == 202
    job_id = started.json()['job_id']
    row = ctl._rows(job_id)[0]
    prepared = json.loads(row['request_json']); prepared.pop('working_media')
    old_file = transport.request('fake-local-test-key', 'POST', '/files', upload=('oriented.jpg', original, 'vision'))['id']
    old_line = json.loads(_json(prepared['line']))
    old_line['body']['input'][0]['content'][1]['file_id'] = old_file
    old_input = transport.request('fake-local-test-key', 'POST', '/files', upload=('old-input.jsonl', (_json(old_line) + '\n').encode(), 'batch'))['id']
    captured = ctl._row(None, job_id)
    manifest, price = json.loads(captured['manifest_json']), json.loads(captured['pricing_json'])
    frozen_body = PaidBatchRequest(**json.loads(captured['request_json']))
    authority = replace(ctl._authority(frozen_body, manifest['images'][0]), operation_id='batch-row-' + _digest([job_id, image['id']]))
    fingerprint = _digest({'prepared': prepared, 'pricing': price, 'transport': 'openai_batch'})
    call = gateway.accounts.reserve(system['alice'], authority, provider='openai', model=frozen_body.provider_options.model,
        maximum_microdollars=ctl._maximum(price, frozen_body), request_fingerprint=fingerprint, replay_settled_stage=True)
    with system['store'].transaction() as con:
        con.execute('UPDATE hosted_paid_batch_rows SET request_json=?,upload_id=?,call_id=?,upload_sha256=NULL WHERE job_id=?', (_json(prepared), old_file, call['id'], job_id))
        con.execute('UPDATE hosted_paid_batches SET input_file_id=? WHERE id=?', (old_input, job_id))
    monkeypatch.setattr(ctl, '_launch', launch)
    ctl._launch(job_id)
    done = eventually(lambda: row if (row := system['client'].get('/api/hosted/tools/paid-batches/' + job_id).json()).get('status') in {'collected', 'failed'} else None)
    assert done['status'] == 'collected', done
    retained = ctl._rows(job_id)[0]
    assert retained['upload_id'] != old_file and retained['upload_sha256'] == view['sha256']
    remote = next(iter(transport.batches.values()))
    assert remote['input_file_id'] != old_input
    line = json.loads(transport.files[remote['input_file_id']].splitlines()[0])
    assert line['body']['input'][0]['content'][1]['file_id'] == retained['upload_id']
    assert len(gateway.accounts.calls(system['admin'])) == 1
    with system['store'].read_connection() as con:
        final = con.execute('SELECT request_hash,state FROM hosted_provider_calls WHERE id=?', (call['id'],)).fetchone()
        assert final['request_hash'] == fingerprint and final['state'] == 'settled'
    assert json.loads(retained['request_json']) == prepared
