"""Real native caption orchestration with a metered-transport double; no charges."""
from PIL import Image
import pytest

from services.hosted.paid_caption import PaidCaptionOptions, PaidCaptionRuntime
from services.hosted.types import Actor, MutationContext
from test_hosted_tools_api import system, acquire, eventually, op


class Gateway:
    def __init__(self):
        self.calls = []
    def stage(self, actor, authority, **kwargs):
        self.calls.append(kwargs)
        return {"call_id": str(len(self.calls)), "caption": "A red vehicle is parked on a road.",
            "usage": {"input_tokens": 10, "output_tokens": 12}, "provider_response_id": "fixture"}


def runtime(gateway):
    return PaidCaptionRuntime(gateway, Actor("alice", "annotator"), MutationContext("image", "version", 1, "editor-id", "operation-id"),
        source_media_hash="a" * 64, options=PaidCaptionOptions(model="test-paid-model"), stage_prefix="job-attempt-1")


def test_paid_runtime_preserves_native_full_caption_and_system_prompt():
    gateway = Gateway()
    value = runtime(gateway)
    result = value.execute("qwen_caption", Image.new("RGB", (64, 48), "red"),
        {"user_prompt": "Describe this street", "caption_system_prompt": "Use plain English.", "max_new_tokens": 256,
         "restrict_to_labels": False, "caption_loop_recovery_mode": "off"}, {"actor_id": "alice"})
    assert result["caption"] == "A red vehicle is parked on a road."
    assert gateway.calls[0]["image"].size == (64, 48)
    assert "Use plain English" in gateway.calls[0]["system_prompt"]
    assert "Describe this street" in gateway.calls[0]["user_prompt"]
    assert gateway.calls[0]["source_media_hash"] == "a" * 64
    assert result["hosted_evidence"]["model_io"][0]["runtime_platform"] == "openai_responses"
    value.close()


def test_paid_transport_failure_cannot_become_an_automatic_second_charge():
    from services.hosted.errors import DomainError
    class Failed(Gateway):
        def stage(self, *args, **kwargs):
            self.calls.append(kwargs)
            raise DomainError("paid_call_outcome_unresolved", 409)
    gateway = Failed()
    value = runtime(gateway)
    with pytest.raises(DomainError, match="paid_call_outcome_unresolved"):
        value.execute("qwen_caption", Image.new("RGB", (64, 48)), {}, {"actor_id": "alice"})
    assert len(gateway.calls) == 1


def test_paid_windowed_caption_preserves_crops_and_text_merge():
    gateway = Gateway()
    value = runtime(gateway)
    result = value.execute("qwen_caption", Image.new("RGB", (800, 400)), {
        "caption_mode": "windowed", "window_size": 400, "window_overlap": 0,
        "caption_windowed_full_image_strategy": "text_only", "caption_all_windows": True,
        "restrict_to_labels": False, "include_counts": False, "max_new_tokens": 128,
        "caption_loop_recovery_mode": "off"}, {"actor_id": "alice"})
    assert result["caption"]
    assert [call["image"].size for call in gateway.calls if call["image"]][:2] == [(400, 400), (400, 400)]
    assert any(call["image"] is None for call in gateway.calls)
    assert len({call["stage_id"] for call in gateway.calls}) == len(gateway.calls)
    value.close()


def configure_paid(system, transport=None):
    from services.hosted.providers import ProviderAccounts
    from services.hosted.provider_gateway import PaidCaptionGateway
    accounts = ProviderAccounts(system["store"])
    accounts.configure(system["admin"], "openai", expected_revision=0, operation_id=op(),
        credential="fake-local-test-key", enabled=True, monthly_limit=1000000, request_limit=10000,
        allowed_datasets=[system["dataset"]["id"]], models=["test-paid-model"])
    calls = []
    def respond(secret, payload):
        calls.append(payload)
        if transport:
            return transport(secret, payload)
        return {"id": str(len(calls)), "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "A red image."}]}],
            "usage": {"input_tokens": 10, "output_tokens": 8}}
    gateway = PaidCaptionGateway(accounts, system["artifacts"], pricing={"test-paid-model": {
        "input_microdollars_per_million": 100, "cached_input_microdollars_per_million": 50,
        "output_microdollars_per_million": 200, "input_token_ceiling": 10000, "max_output_tokens": 4096}}, transport=respond)
    system["controller"].provider_gateway = gateway
    return gateway, calls


def test_paid_current_image_job_uses_native_flow_and_retains_real_ledger(system):
    gateway, calls = configure_paid(system)
    image = acquire(system, system["images"][0])
    body = {"dataset_id": image["dataset_id"], "image_id": image["id"], "expected_version": image["current_version"],
        "generation": image["generation"], "editor_id": "alice-editor", "operation_id": op(),
        "provider": "openai", "provider_options": {"model": "test-paid-model"},
        "caption_request": {"max_new_tokens": 256, "restrict_to_labels": False, "caption_loop_recovery_mode": "off"},
        "settings": {"attempts": 1}}
    url = "/api/hosted/tools/caption-jobs"
    result = system["client"].post(url, json=body)
    assert result.status_code == 202, result.text
    job_id = result.json()["job_id"]
    done = eventually(lambda: (row if (row := system["client"].get(url + "/" + job_id).json()).get("status") in {"completed", "failed"} else None))
    assert done["status"] == "completed", done
    assert done["result"]["latest_caption"]["caption"] == "A red image."
    assert len(calls) >= 1
    ledger = gateway.accounts.calls(system["admin"])
    assert len(ledger) == len(calls) and all(row["state"] == "settled" for row in ledger)
    assert system["store"].image(system["alice"], image["id"])["current_version"] == image["current_version"]
    assert system["client"].post(url, json=body).json()["job_id"] == job_id
    assert len(gateway.accounts.calls(system["admin"])) == len(calls)


def test_paid_caption_and_qa_share_encoded_pixels_and_original_identity(system):
    """Run HTTP admission and both native stages; double only provider transport."""
    import base64
    import hashlib
    import io
    import json
    from services.hosted.media import decode_encoded

    source = Image.new("RGB", (64, 48), "red")
    source.paste("blue", (0, 0, 24, 48))
    exif = Image.Exif()
    exif[274] = 6
    content = io.BytesIO()
    source.save(content, format="TIFF", exif=exif)
    original = content.getvalue()
    reference = system["artifacts"].put_bytes(original)
    store, admin = system["store"], system["admin"]
    image = store.import_image(admin, system["dataset"]["id"], reference.sha256,
        "oriented.tif", {}, op(), width=64, height=48, size_bytes=reference.size_bytes)
    image = store.assign(admin, image["dataset_id"], [image["id"]], system["alice"].user_id, op(),
        expected={image["id"]: {"expected_version": image["current_version"], "generation": image["generation"]}})["images"][0]
    image = acquire(system, image)
    decoded, _ = decode_encoded(original)
    expected_pixels = hashlib.sha256(decoded.tobytes()).hexdigest()
    sent_images = []

    def respond(_secret, payload):
        for message in payload["input"]:
            for item in message.get("content", []):
                if item.get("type") == "input_image":
                    raw = base64.b64decode(item["image_url"].split(",", 1)[1])
                    with Image.open(io.BytesIO(raw)) as submitted:
                        sent_images.append((submitted.size, hashlib.sha256(submitted.convert("RGB").tobytes()).hexdigest()))
        # Empty QA is a valid provider response for this pixel-grid check; the
        # native helper still executes its real generation and retention path.
        text = "A red vehicle is parked on a road." if len(sent_images) == 1 else '{"pairs": []}'
        return {"id": str(len(sent_images)), "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 10, "output_tokens": 12}}

    gateway, calls = configure_paid(system, respond)
    body = paid_body(image)
    body["settings"].update(instruction_dataset=True, subcaptions_per_image=1, instruction_qa_max_topup_attempts=0)
    url = "/api/hosted/tools/caption-jobs"
    response = system["client"].post(url, json=body)
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    done = eventually(lambda: (row if (row := system["client"].get(url + "/" + job_id).json()).get("status") in {"completed", "failed"} else None))
    assert len(sent_images) >= 2, done
    assert all(value == ((64, 48), expected_pixels) for value in sent_images)
    ledger = gateway.accounts.calls(admin)
    assert len(ledger) == len(calls) and all(row["state"] == "settled" for row in ledger)
    with store.read_connection() as con:
        results = [json.loads(row[0]) for row in con.execute("SELECT result_json FROM hosted_provider_results")]
    for result in results:
        with system["artifacts"].open(result["input_artifacts"][-1]["sha256"]) as stream:
            identity = json.load(stream)
        assert identity["source_media_hash"] == reference.sha256
    with system["artifacts"].open(reference) as stream:
        assert stream.read() == original
    assert store.image(system["alice"], image["id"])["current_version"] == image["current_version"]


def test_paid_snapshot_batch_keeps_canonical_source_and_real_proposal(system):
    gateway, calls = configure_paid(system)
    store, actor = system["store"], system["admin"]
    source = store.image(actor, system["images"][0]["id"])
    import io
    content = io.BytesIO()
    Image.new("RGB", (64, 48), "orange").save(content, format="PNG")
    reference = system["artifacts"].put_bytes(content.getvalue())
    image = store.import_image(actor, source["dataset_id"], reference.sha256, "confirmed-paid.png", {}, op(), width=64, height=48, size_bytes=reference.size_bytes)
    image = store.acquire_editor(actor, image["id"], "admin-editor")
    context = lambda: MutationContext(image["id"], image["current_version"], image["generation"], "admin-editor", op())
    image = store.save(actor, context(), {"objects": []}, complete=True, negative=True)
    image = store.approve(actor, context())
    snapshot = store.create_snapshot(actor, image["dataset_id"], op(), "Paid snapshot fixture")
    body = {"dataset_id": image["dataset_id"], "snapshot_id": snapshot["snapshot_id"], "operation_id": op(),
        "provider": "openai", "provider_options": {"model": "test-paid-model"}, "sample_size": 0, "pilot": None,
        "caption_request": {"max_new_tokens": 256, "restrict_to_labels": False, "caption_loop_recovery_mode": "off"},
        "settings": {"attempts": 1}}
    url = "/api/hosted/tools/caption-batches"
    result = system["admin_client"].post(url, json=body)
    assert result.status_code == 202, result.text
    job_id = result.json()["job_id"]
    done = eventually(lambda: (row if (row := system["admin_client"].get(url + "/" + job_id).json()).get("status") in {"completed", "failed"} else None))
    assert done["status"] == "completed", done
    proposal = system["admin_client"].get(url + "/" + job_id + "/images/" + image["id"] + "/proposal")
    assert proposal.status_code == 200, proposal.text
    assert proposal.json()["caption"] == "A red image."
    assert calls and store.image(actor, image["id"])["current_version"] == image["current_version"]


def paid_body(image):
    return {"dataset_id": image["dataset_id"], "image_id": image["id"], "expected_version": image["current_version"],
        "generation": image["generation"], "editor_id": "alice-editor", "operation_id": op(),
        "provider": "openai", "provider_options": {"model": "test-paid-model"},
        "caption_request": {"max_new_tokens": 256, "restrict_to_labels": False, "caption_loop_recovery_mode": "off"},
        "settings": {"attempts": 1}}


def test_paid_job_replays_settled_stage_after_attempt_retention_interruption(system, monkeypatch):
    from services.hosted.runtime import RuntimeRejected
    gateway, calls = configure_paid(system)
    image = acquire(system, system["images"][0])
    jobs = system["controller"].caption_jobs
    attempt = jobs._paid_attempt
    interrupted = []
    def stop_after_result(*args):
        result = attempt(*args)
        if not interrupted:
            interrupted.append(True)
            raise RuntimeRejected("caption_control_stopped")
        return result
    monkeypatch.setattr(jobs, "_paid_attempt", stop_after_result)
    body, url = paid_body(image), "/api/hosted/tools/caption-jobs"
    result = system["client"].post(url, json=body)
    assert result.status_code == 202, result.text
    job_id = result.json()["job_id"]
    eventually(lambda: system["client"].get(url + "/" + job_id).json().get("status") == "interrupted")
    count = len(calls)
    # The new editor owns the same unchanged version; settled stages are
    # replayed under fresh authority without rewriting their original ledger.
    acquired = system["store"].acquire_editor(system["alice"], image["id"], "alice-second-editor",
        expected_version=image["current_version"], generation=image["generation"], takeover=True)
    # Exercise the normal acquire hook, which alone admits a +1 generation.
    with system["store"].transaction() as con:
        con.execute("UPDATE hosted_caption_jobs SET state='waiting_for_editor' WHERE id=?", (job_id,))
    resumed = jobs.resume_waiting(system["alice"], image["id"], "alice-second-editor",
        previous_generation=image["generation"], generation=acquired["generation"], expected_version=acquired["current_version"])
    assert resumed == [job_id]
    done = eventually(lambda: (row if (row := system["client"].get(url + "/" + job_id).json()).get("status") in {"completed", "failed"} else None))
    assert done["status"] == "completed", done
    assert count == len(calls) >= 1
    assert len(gateway.accounts.calls(system["admin"])) == count


def test_cancel_committed_after_stage_preparation_prevents_paid_dispatch(system, monkeypatch):
    import threading
    gateway, calls = configure_paid(system)
    image = acquire(system, system["images"][0])
    entered, release = threading.Event(), threading.Event()
    dispatch = gateway.accounts.dispatch
    def paused(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return dispatch(*args, **kwargs)
    monkeypatch.setattr(gateway.accounts, "dispatch", paused)
    body, url = paid_body(image), "/api/hosted/tools/caption-jobs"
    result = system["client"].post(url, json=body)
    assert result.status_code == 202, result.text
    job_id = result.json()["job_id"]
    try:
        assert entered.wait(10)
        cancelled = system["client"].post(url + "/" + job_id + "/cancel", json={"operation_id": op()})
        assert cancelled.status_code == 200, cancelled.text
    finally:
        release.set()
    eventually(lambda: system["client"].get(url + "/" + job_id).json().get("status") == "cancelled")
    assert calls == []
    assert gateway.accounts.calls(system["admin"])[0]["charged"] == 0


@pytest.mark.parametrize("cancel_while_paused", [False, True])
def test_paid_dispatch_waits_for_real_heavy_window_without_new_reservation(system, monkeypatch, cancel_while_paused):
    import threading
    gateway, calls = configure_paid(system)
    image = acquire(system, system["images"][0])
    entered, release = threading.Event(), threading.Event()
    dispatch = gateway.accounts.dispatch
    def paused(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(10)
        return dispatch(*args, **kwargs)
    monkeypatch.setattr(gateway.accounts, "dispatch", paused)
    body, url = paid_body(image), "/api/hosted/tools/caption-jobs"
    result = system["client"].post(url, json=body)
    assert result.status_code == 202, result.text
    job_id = result.json()["job_id"]
    try:
        assert entered.wait(10)
        heavy = system["admin_client"].post("/api/hosted/tools/heavy", json={"dataset_id": image["dataset_id"],
            "job": "fixture", "operation_id": op(), "parameters": {"uuid": "hold-paid-pause"}})
        assert heavy.status_code == 202, heavy.text
        eventually(lambda: (system["gate"] / "hold-paid-pause.started").exists())
        release.set()
        eventually(lambda: system["client"].get(url + "/" + job_id).json().get("status") == "paused")
        assert calls == []
        ledger = gateway.accounts.calls(system["admin"])
        assert len(ledger) == 1 and ledger[0]["state"] == "reserved"
        if cancel_while_paused:
            response = system["client"].post(url + "/" + job_id + "/cancel", json={"operation_id": op()})
            assert response.status_code == 200, response.text
        else:
            (system["gate"] / "hold-paid-pause.release").touch()
        done = eventually(lambda: (row if (row := system["client"].get(url + "/" + job_id).json()).get("status") in {"completed", "cancelled", "failed"} else None))
        assert done["status"] == ("cancelled" if cancel_while_paused else "completed"), done
        final_ledger = gateway.accounts.calls(system["admin"])
        assert ledger[0]["id"] in {row["id"] for row in final_ledger}
        assert len(final_ledger) == (1 if cancel_while_paused else len(calls))
        assert (not calls) if cancel_while_paused else bool(calls)
        if cancel_while_paused:
            assert gateway.accounts.calls(system["admin"])[0]["charged"] == 0
    finally:
        release.set()
        (system["gate"] / "hold-paid-pause.release").touch()
