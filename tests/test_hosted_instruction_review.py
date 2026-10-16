"""Versioned human review preserves generated evidence and native readiness."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from test_hosted_api import application, login, dataset, upload, acquire, context, op, provision


def test_nested_qa_shapes_reject_without_versions_and_preserve_legacy_extensions(application):
    admin = login(application)
    ds = dataset(admin)
    image = acquire(admin, upload(admin, ds["id"]))
    base = f"/api/hosted/images/{image['id']}"
    for collection in ("captions", "alternate_captions"):
        for nested in (False, True):
            for location in ((), ("metadata",), ("validation",), ("metadata", "validation")):
                for field in ("validated_against", "source_fields"):
                    pair = {"question": "What color?", "answer": "Orange", "validated_against": ["image"]}
                    target = pair
                    for key in location:
                        target = target.setdefault(key, {})
                    target[field] = 17
                    record = {"caption": "Orange scene", "generated_qa_pairs": []}
                    destination = record.setdefault("metadata", {}) if nested else record
                    destination["generated_qa_pairs"] = [pair]
                    response = admin.post(base + "/save", json={**context(image), "annotations": {"objects": [], collection: [record]}})
                    assert response.status_code == 422, response.text
                    expected_path = collection + "[0]." + ("metadata." if nested else "") + "generated_qa_pairs[0]." + ".".join((*location, field))
                    assert response.json()["detail"]["field"] == expected_path
                    assert admin.get(base).json()["current_version"] == image["current_version"]
    valid = {"objects": [], "caption": None, "captions": ["Legacy plain text"], "alternate_captions": [{
        "caption": None, "metadata": {"generated_qa_pairs": [{"question": None, "answer": "Orange",
            "source_fields": None, "validated_against": ["image", {"legacy": "extension"}],
            "metadata": {"validation": {"source_fields": [], "validated_against": None}, "custom": 17}, "custom": {"anything": True}}]}, "custom": 17}]}
    saved = admin.post(base + "/save", json={**context(image), "annotations": valid, "negative": True})
    assert saved.status_code == 200, saved.text
    assert admin.get(base).json()["annotations"] == valid
    approved = admin.post(base + "/approve", json=context(saved.json()))
    assert approved.status_code == 200


def test_historical_nested_qa_preflight_before_actual_native_export(application, tmp_path):
    from services.hosted.errors import DomainError
    from services.hosted.native_analysis import NativeAnalysisAdapter
    admin = login(application)
    ds = dataset(admin)
    image = acquire(admin, upload(admin, ds["id"]))
    annotations = {"objects": [], "caption": "An orange scene.", "alternate_captions": [{"caption": "An orange scene.",
        "generated_qa_pairs": [{"question": "What color?", "answer": "Orange.", "metadata": {"validated_against": ["image"], "source_fields": []}}]}]}
    base = f"/api/hosted/images/{image['id']}"
    saved = admin.post(base + "/save", json={**context(image), "annotations": annotations, "negative": True})
    assert saved.status_code == 200
    assert admin.post(base + "/approve", json=context(saved.json())).status_code == 200
    snapshot = admin.post(f"/api/hosted/datasets/{ds['id']}/snapshots", json={"operation_id": op()}).json()
    manifest = snapshot["manifest"] | {"snapshot_id": snapshot["snapshot_id"]}
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    with application.state.artifacts.exclusive():
        for member in manifest["images"]:
            ref = application.state.artifacts.pin(member["media_hash"], frozen)
            member["media_path"] = str(frozen / ref.sha256)
    def reference(value, name):
        path = frozen / name
        path.write_text(json.dumps(value))
        return {**{key: value[key] for key in ("snapshot_id", "dataset_id", "revision", "class_version", "settings_version")},
                "manifest_path": str(path), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    adapter = NativeAnalysisAdapter(config={"canonical_root": str(frozen), "native_root": str(tmp_path / "native")}, device="cpu")
    request_context = {"dataset_id": ds["id"], "annotation_version": snapshot["snapshot_id"]}
    result = adapter.execute("caption_export", {"canonical_snapshot": reference(manifest, "valid.json")}, request_context)
    assert result["state"] == "complete" and (Path(result["native_directory"]) / "receipt.json").is_file()
    bad = copy.deepcopy(manifest)
    bad["images"][0]["annotations"]["alternate_captions"][0]["generated_qa_pairs"][0]["metadata"]["validated_against"] = 17
    bad_reference = reference(bad, "historical-invalid.json")
    before = sorted((tmp_path / "native").iterdir())
    with pytest.raises(DomainError) as caught:
        adapter.execute("caption_export", {"canonical_snapshot": bad_reference}, request_context)
    assert caught.value.status == 422
    assert caught.value.detail == {"image_id": image["id"], "field": "alternate_captions[0].generated_qa_pairs[0].metadata.validated_against", "expected": "list"}
    assert sorted((tmp_path / "native").iterdir()) == before
    assert json.loads((frozen / "historical-invalid.json").read_text()) == bad


def test_explicit_review_is_versioned_receipted_and_keeps_model_evidence(application):
    admin = login(application)
    ds = dataset(admin)
    image = acquire(admin, upload(admin, ds["id"]))
    annotations = {"objects": [], "caption": "A quiet scene.", "captions": [{"caption": "Imported description.", "migration_source": "retained",
        "generated_qa_pairs": [{"question": "What was imported?", "answer": "This image.", "raw_model_reply": "keep"}]}], "alternate_captions": [
        {"caption": "A quiet scene.", "source": "hosted_qwen", "caption_quality": {"passed": True},
         "unknown_evidence": {"keep": [1, 2]}, "metadata": {"review_status": "accepted", "original": "retained"},
         "generated_qa_pairs": [
             {"question": "What is visible?", "answer": "An open area.", "validation_status": "accepted", "metadata": {"model_trace": "original"}},
             {"question": "What is its mood?", "answer": "Peaceful.", "validation_status": "accepted", "raw_model_response": "retained"}]},
        {"caption": "An alternate view.", "metadata": {"generated_qa_pairs": [
             {"question": "What is above?", "answer": "Sky.", "validation_status": "machine_validated"}]}}]}
    base = f"/api/hosted/images/{image['id']}"
    image = admin.post(base + "/save", json={**context(image), "annotations": annotations, "complete": True, "negative": True}).json()
    review_path = f"/api/hosted/tools/images/{image['id']}/instruction-review"
    review = admin.get(review_path).json()
    assert len(review["rows"]) == 7
    body = {**context(image), "source_sha256": review["source_sha256"], "decisions": [
        {"target_id": "caption:0", "decision": "unreviewed", "notes": "Reconsider this caption"},
        {"target_id": "qa:0:top:0", "decision": "accepted", "notes": "Checked against the image"},
        {"target_id": "qa:0:top:1", "decision": "rejected", "notes": "Subjective"},
        {"target_id": "qa:1:metadata:0", "decision": "needs_revision", "notes": "Check the crop"},
        {"target_id": "captions:qa:0:top:0", "decision": "accepted", "notes": "Review imported QA"}]}
    accepted = admin.post(review_path, json=body)
    assert accepted.status_code == 200, accepted.text
    fresh = admin.get(base).json()
    saved = fresh["annotations"]["alternate_captions"][0]
    assert saved["unknown_evidence"] == annotations["alternate_captions"][0]["unknown_evidence"]
    assert saved["caption_quality"] == {"passed": True}
    assert saved["metadata"]["review_status"] == "unreviewed"
    pair = saved["generated_qa_pairs"][0]
    assert pair["validation_status"] == "accepted" and pair["metadata"]["model_trace"] == "original"
    assert pair["metadata"]["reviewer"] == admin.get("/api/hosted/session").json()["user"]["user_id"]
    assert pair["metadata"]["review_decision"] == "accepted"
    assert pair["metadata"]["review_source_version"] == image["current_version"]
    assert fresh["annotations"]["alternate_captions"][1]["metadata"]["generated_qa_pairs"][0]["review_status"] == "needs_revision"
    assert fresh["annotations"]["captions"][0]["generated_qa_pairs"][0]["review_status"] == "accepted"
    assert fresh["annotations"]["captions"][0]["generated_qa_pairs"][0]["raw_model_reply"] == "keep"
    # Later ordinary save keeps the human decision and all original evidence.
    edited = copy.deepcopy(fresh["annotations"])
    edited["caption"] = "A quiet open area."
    later = admin.post(base + "/save", json={**context(fresh), "annotations": edited, "complete": True, "negative": True})
    assert later.status_code == 200
    assert admin.get(base).json()["annotations"]["alternate_captions"][0] == saved
    assert admin.get(base).json()["annotations"]["captions"] == fresh["annotations"]["captions"]
    assert admin.post(review_path, json=body).json() == accepted.json()
    assert admin.post(review_path, json={**body, "decisions": body["decisions"][:1]}).status_code == 409
    assert admin.post(review_path, json={**body, "operation_id": op()}).status_code == 409
    provision(admin, "alice")
    assert login(application, "alice").get(review_path).status_code == 403
    assert login(application, "alice").post(review_path, json=body).status_code == 403
    # Editing reviewed language is not permission to approve the new wording.
    current = admin.get(base).json()
    current["annotations"]["alternate_captions"][0]["generated_qa_pairs"][0]["answer"] = "A different answer."
    changed = admin.post(base + "/save", json={**context(current), "annotations": current["annotations"], "complete": True, "negative": True})
    assert changed.status_code == 200
    changed_row = next(row for row in admin.get(review_path).json()["rows"] if row["target_id"] == "qa:0:top:0")
    assert changed_row["decision"] == "unreviewed" and changed_row["text_changed_since_review"]


def test_primary_only_caption_review_reaches_original_export_readiness(application, tmp_path):
    from services.hosted.native_analysis import NativeAnalysisAdapter
    admin = login(application)
    ds = dataset(admin)
    image = acquire(admin, upload(admin, ds["id"]))
    base = f"/api/hosted/images/{image['id']}"
    annotations = {"objects": [{"id": "object-one", "class_id": ds["classes"][0]["id"], "bbox": [2, 3, 10, 12]}], "caption": "A vehicle on a road."}
    image = admin.post(base + "/save", json={**context(image), "annotations": annotations, "complete": True}).json()
    image = admin.post(base + "/approve", json=context(image)).json()
    image = acquire(admin, image)
    before = admin.post(f"/api/hosted/datasets/{ds['id']}/snapshots", json={"operation_id": op(), "name": "Before review"}).json()
    path = f"/api/hosted/tools/images/{image['id']}/instruction-review"
    review = admin.get(path).json()
    assert review["rows"][0]["target_id"] == "primary"
    response = admin.post(path, json={**context(image), "source_sha256": review["source_sha256"], "decisions": [
        {"target_id": "primary", "decision": "accepted", "notes": "Checked the complete image"}]})
    assert response.status_code == 200, response.text
    after = admin.post(f"/api/hosted/datasets/{ds['id']}/snapshots", json={"operation_id": op(), "name": "After review"}).json()

    def export(snapshot, name, *, include_metadata=False):
        # Invoke the actual isolated native CPU export, not a readiness double.
        manifest = snapshot["manifest"] | {"snapshot_id": snapshot["snapshot_id"]}
        directory = tmp_path / name
        directory.mkdir()
        with application.state.artifacts.exclusive():
            for row in manifest["images"]:
                ref = application.state.artifacts.pin(row["media_hash"], directory)
                row["media_path"] = str(directory / ref.sha256)
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        reference = {key: manifest[key] for key in ("snapshot_id", "dataset_id", "revision", "class_version", "settings_version")}
        reference.update(manifest_path=str(manifest_path), manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        adapter = NativeAnalysisAdapter(config={"canonical_root": str(directory), "native_root": str(tmp_path / "native"), "allowed_parameters": ["include_deterministic_metadata_qa"]}, device="cpu")
        result = adapter.execute("caption_export", {"canonical_snapshot": reference, "include_deterministic_metadata_qa": include_metadata}, {"dataset_id": ds["id"], "annotation_version": snapshot["snapshot_id"]})
        output = Path(result["native_directory"])
        receipt = json.loads((output / "receipt.json").read_text())
        measured = json.loads((output / receipt["result_path"]).read_text())["engine_result"]
        measured["readiness"] = json.loads((output / "caption_export/readiness.json").read_text())["training_readiness"]
        return measured

    unreviewed, reviewed = export(before, "before"), export(after, "after")
    assert unreviewed["blocked_reason"] and not unreviewed["bundle_path"]
    assert unreviewed["readiness"]["pending_manual_review_row_count"] == 1
    assert reviewed["readiness"]["pending_manual_review_row_count"] == 0
    # Human approval is not a bypass for the independent native coverage rule.
    assert reviewed["readiness"]["quality_warnings"] == ["source_class_coverage_rate_below_threshold"]
    complete = export(after, "with-structured-metadata", include_metadata=True)
    assert not complete["blocked_reason"] and complete["bundle_path"]
    current = admin.get(base).json()
    current["annotations"]["alternate_captions"][0]["generated_qa_pairs"] = [{
        "question": "Describe the surface.", "answer": "Paved.", "validation_status": "machine_validated",
        "metadata": {"validation_status": "machine_validated"}, "validation": {"status": "machine_validated"}}]
    response = admin.post(base + "/save", json={**context(current), "annotations": current["annotations"], "complete": True})
    assert response.status_code == 200
    qa_review = admin.get(path).json()
    response = admin.post(path, json={**context(response.json()), "source_sha256": qa_review["source_sha256"], "decisions": [
        {"target_id": "qa:0:top:0", "decision": "accepted", "notes": "Checked the surface"}]})
    assert response.status_code == 200
    current = admin.get(base).json()
    current["annotations"]["caption"] = "A vehicle near a building."
    current["annotations"]["alternate_captions"][0]["caption"] = current["annotations"]["caption"]
    current["annotations"]["alternate_captions"][0]["generated_qa_pairs"][0]["answer"] = "An entirely different surface."
    response = admin.post(base + "/save", json={**context(current), "annotations": current["annotations"], "complete": True})
    assert response.status_code == 200
    changed = admin.post(f"/api/hosted/datasets/{ds['id']}/snapshots", json={"operation_id": op(), "name": "Changed wording"}).json()
    invalidated = export(changed, "changed-wording", include_metadata=True)
    assert invalidated["readiness"]["pending_manual_review_row_count"] == 2
    assert invalidated["blocked_reason"] and not invalidated["bundle_path"]
