from concurrent.futures import ThreadPoolExecutor
import copy
from hashlib import sha256
import sqlite3

import pytest

from services.hosted.errors import DomainError
from services.hosted.store import HostedStore
from services.hosted.types import Actor, MutationContext


ADMIN = Actor("admin", "admin", "administrator")
ADMIN2 = Actor("admin2", "admin", "other administrator")
ALICE = Actor("alice", "annotator", "alice")
BOB = Actor("bob", "annotator", "bob")


@pytest.fixture
def store(tmp_path):
    # AuthService owns this schema. These rows exercise the store component's
    # actual transaction-time user checks without importing a password service.
    now = [1000.0]
    value = HostedStore(tmp_path / "hosted.sqlite3", clock=lambda: now[0], lease_seconds=30)
    value.test_clock = now
    with value.transaction() as con:
        con.execute("CREATE TABLE hosted_users(id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE COLLATE NOCASE, password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('admin','annotator')), disabled INTEGER NOT NULL DEFAULT 0,created_at REAL NOT NULL,password_changed_at REAL NOT NULL)")
        for actor in (ADMIN, ADMIN2, ALICE, BOB):
            con.execute("INSERT INTO hosted_users VALUES (?,?,? ,?,0,0,0)",
                        (actor.user_id, actor.username, "unused by store tests", actor.role))
    return value


def dataset(store, name="test"):
    return store.create_dataset(ADMIN, name, ["car"], "create-" + name)


def imported(store, ds, name="one", annotations=None):
    return store.import_image(ADMIN, ds["id"], sha256(name.encode()).hexdigest(), name + ".png",
                              annotations, "import-" + ds["id"] + name,
                              width=100, height=80, size_bytes=100)


def objects(ds, object_id="object-1"):
    return {"objects": [{"id": object_id, "class_id": ds["classes"][0]["id"],
                         "bbox": [1, 2, 10, 15]}], "caption": "A car"}


def ctx(image, operation_id, editor="tab"):
    return MutationContext(image["id"], image["current_version"], image["generation"], editor, operation_id)


def expected(images):
    return {image["id"]: {"expected_version": image["current_version"], "generation": image["generation"]}
            for image in images}


def assigned(store, ds, image, user=ALICE):
    return store.assign(ADMIN, ds["id"], [image["id"]], user.user_id,
                        "assign-" + image["id"], expected([image]))["images"][0]


def error(code, callback):
    with pytest.raises(DomainError) as exc:
        callback()
    assert exc.value.code == code


def test_direct_approval_negative_and_canonical_snapshot_survive_lost_reply(store):
    ds = dataset(store)
    image = imported(store, ds)
    assert store.snapshot(ADMIN, ds["id"])["images"] == []
    lease = store.acquire_editor(ADMIN, image["id"], "tab")
    draft = store.save(ADMIN, ctx(lease, "empty-review"), {"objects": []}, complete=True)
    error("empty_annotation_requires_negative_review", lambda: store.approve(ADMIN, ctx(draft, "approve-empty")))
    assert store.snapshot(ADMIN, ds["id"])["images"] == []
    negative = store.save(ADMIN, ctx(draft, "negative-review"), {"objects": []}, negative=True)
    approval = ctx(negative, "approve-negative")
    approved = store.approve(ADMIN, approval)
    assert approved["state"] == "confirmed" and approved["dataset_revision"] == 1
    reopened = HostedStore(store.path, clock=store.clock)
    assert reopened.approve(ADMIN, approval) == approved
    assert reopened.receipt(ADMIN, "approve-negative")["result"] == approved
    assert reopened.receipt(ALICE, "approve-negative") is None
    error("operation_id_reused", lambda: reopened.approve(ADMIN, approval, "different request"))
    frozen = reopened.create_snapshot(ADMIN, ds["id"], "freeze", "before edit")
    canonical = reopened.acquire_editor(ADMIN, image["id"], "new-tab")
    changed = reopened.save(ADMIN, ctx(canonical, "canonical-edit", "new-tab"), objects(ds), complete=True)
    assert changed["dataset_revision"] == 2
    assert reopened.snapshot(ADMIN, ds["id"])["images"][0]["annotations"] == objects(ds)
    assert reopened.get_snapshot(ADMIN, frozen["snapshot_id"])["manifest"]["images"][0]["negative"] == 1
    error("image_not_found", lambda: reopened.image(ALICE, image["id"]))
    error("admin_required", lambda: reopened.snapshot(ALICE, ds["id"]))
    with reopened.transaction() as con:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            con.execute("UPDATE hosted_annotation_versions SET complete=0 WHERE id=?", (negative["current_version"],))


def test_dataset_settings_patch_receipt_survives_unrelated_changes(store):
    ds = dataset(store)
    selected = store.patch_dataset_settings(ADMIN, ds["id"], {"detector_model_id": "detector-a"}, ds["revision"], "choose-detector")
    unrelated = store.patch_dataset_settings(ADMIN, ds["id"], {"recipe": {"weight": 0.8}}, selected["revision"], "change-recipe")
    assert unrelated["settings"] == {"detector_model_id": "detector-a", "recipe": {"weight": 0.8}}
    assert store.patch_dataset_settings(ADMIN, ds["id"], {"detector_model_id": "detector-a"}, ds["revision"], "choose-detector") == selected
    assert store.get_dataset(ADMIN, ds["id"])["settings"] == unrelated["settings"]
    error("dataset_revision_changed", lambda: store.patch_dataset_settings(ADMIN, ds["id"], {"detector_model_id": "detector-b"}, ds["revision"], "stale-choice"))
    error("operation_id_reused", lambda: store.patch_dataset_settings(ADMIN, ds["id"], {"detector_model_id": "detector-b"}, ds["revision"], "choose-detector"))
    frozen = store.create_snapshot(ADMIN, ds["id"], "settings-snapshot")
    assert frozen["manifest"]["settings"] == unrelated["settings"]


def test_assignment_submission_return_correction_preserves_exact_versions_and_receipts(store):
    ds = dataset(store)
    one, two = imported(store, ds, "one"), imported(store, ds, "two")
    assignment = store.assign(ADMIN, ds["id"], [one["id"], two["id"]], ALICE.user_id, "batch", expected([one, two]))
    assert len(store.list_images(ALICE)) == 2 and not store.list_images(BOB)
    assert len(store.list_datasets(ALICE)) == 1 and not store.list_datasets(BOB)
    error("image_owned_by_annotator", lambda: store.acquire_editor(ADMIN, one["id"], "admin-tab"))
    error("image_not_found", lambda: store.image(BOB, one["id"]))
    lease = store.acquire_editor(ALICE, one["id"], "tab")
    draft = store.save(ALICE, ctx(lease, "alice-save"), objects(ds), complete=True)
    submission_context = ctx(draft, "alice-submit")
    submitted = store.submit(ALICE, submission_context)
    assert store.submit(ALICE, submission_context) == submitted
    assert len(store.list_images(ALICE)) == 1 and not store.snapshot(ADMIN, ds["id"])["images"]
    assert store.image(ALICE, two["id"])["assignment_id"] == assignment["assignment_id"]
    error("image_not_found", lambda: store.save(ALICE, ctx(submitted, "late-save"), objects(ds)))
    review = store.acquire_editor(ADMIN, one["id"], "tab")
    returned = store.return_image(ADMIN, ctx(review, "return"), "Please include the second car.")
    assert returned["state"] == "returned" and returned["generation"] > submitted["generation"]
    lease = store.acquire_editor(ALICE, one["id"], "tab")
    second = objects(ds)
    second["objects"].append({"id": "object-2", "class_id": ds["classes"][0]["id"], "bbox": [30, 30, 10, 10]})
    saved = store.save(ALICE, ctx(lease, "second-save"), second, complete=True)
    store.submit(ALICE, ctx(saved, "second-submit"))
    review = store.acquire_editor(ADMIN2, one["id"], "tab")
    corrected = store.save(ADMIN2, ctx(review, "admin-correction"), {**second, "caption": "Two cars"}, complete=True)
    approved = store.approve(ADMIN2, ctx(corrected, "admin-approval"))
    assert approved["state"] == "confirmed"
    history = store.history(ADMIN, one["id"])
    assert [row["version_id"] for row in history["submissions"]] == [draft["current_version"], saved["current_version"]]
    assert {row["action"] for row in history["reviews"]} == {"returned", "approved"}
    assert history["reviews"][0]["feedback"] == "Please include the second car."
    assert store.submit(ALICE, submission_context) == submitted  # exact old receipt, not a new handoff
    assert store.snapshot(ADMIN, ds["id"])["images"][0]["version_id"] == corrected["current_version"]
    statuses = store.own_submission_status(ALICE)
    assert [row["status"] for row in statuses] == ["approved", "returned"]
    assert all("annotations" not in row and "media_hash" not in row for row in statuses)
    assert not store.own_submission_status(BOB)


def test_annotator_preserves_human_review_evidence_but_cannot_forge_acceptance(store):
    from services.hosted.instruction_review import InstructionReviewApply, InstructionReviews
    from services.hosted.caption_export import _effective_review

    ds = dataset(store)
    annotations = {**objects(ds), 'captions': [{'caption': 'Old accepted wording', 'created_at': '2026-10-01',
        'review_decision': 'accepted', 'reviewer': 'historic-admin'}], 'alternate_captions': [
        {'id': 'generated', 'caption': 'A generated caption', 'raw_model_reply': 'retain',
         'generated_qa_pairs': [{'question': 'What is visible?', 'answer': 'A car.', 'raw_model_reply': 'qa'}],
         'metadata': {'generated_qa_pairs': [{'question': 'Hidden list?', 'answer': 'Still protected.'}]}}]}
    image = imported(store, ds, annotations=annotations)
    lease = store.acquire_editor(ADMIN, image['id'], 'review-tab')
    reviews = InstructionReviews(store)
    inspected = reviews.inspect(ADMIN, image['id'])
    reviewed = reviews.apply(ADMIN, image['id'], InstructionReviewApply(
        expected_version=lease['current_version'], generation=lease['generation'], editor_id='review-tab',
        operation_id='review-generated', source_sha256=inspected['source_sha256'], decisions=[
            {'target_id': 'caption:0', 'decision': 'accepted'}, {'target_id': 'qa:0:top:0', 'decision': 'accepted'}]))
    released = store.release_editor(ADMIN, ctx(reviewed, 'release-review', 'review-tab'))
    assigned(store, ds, released)
    lease = store.acquire_editor(ALICE, image['id'], 'tab')
    before = store.image(ALICE, image['id'])['annotations']
    attempts = []
    changed = copy.deepcopy(before); changed['alternate_captions'][0]['metadata']['reviewer'] = 'alice'; attempts.append(changed)
    changed = copy.deepcopy(before); changed['captions'].append({'caption': 'Forged native', 'review_decision': 'accepted'}); attempts.append(changed)
    changed = copy.deepcopy(before); changed['alternate_captions'].append(copy.deepcopy(changed['alternate_captions'][0])); attempts.append(changed)
    changed = copy.deepcopy(before); changed['captions'][0]['caption'] = 'New wording inheriting old approval'; attempts.append(changed)
    changed = copy.deepcopy(before); changed['alternate_captions'][0]['metadata']['generated_qa_pairs'][0]['review_status'] = 'accepted'; attempts.append(changed)
    changed = copy.deepcopy(before); changed['alternate_captions'][0].pop('metadata'); changed['alternate_captions'][0].pop('review_status'); attempts.append(changed)
    for number, changed in enumerate(attempts):
        error('instruction_review_requires_admin', lambda: store.save(ALICE, ctx(lease, 'forge-' + str(number)), changed, complete=True))
        assert store.image(ALICE, image['id'])['current_version'] == lease['current_version']

    edited = copy.deepcopy(before)
    edited['alternate_captions'][0]['caption'] = 'Corrected wording'
    edited['alternate_captions'][0]['updated_at'] = '2026-10-15T18:00:00Z'
    edited['alternate_captions'][0]['generated_qa_pairs'][0]['answer'] = 'Two cars.'
    edited['objects'][0]['bbox'] = [2, 3, 10, 15]
    # Existing normalizeCaptionRecord adds these display fields and supplies
    # updated_at from created_at. Imported captions may acquire their first ID.
    for number, record in enumerate(edited['captions'] + edited['alternate_captions']):
        record.update(id=record.get('id', 'native-ui-' + str(number)), image_name='one.png',
            image='one.png', image_key='one.png', split=None, caption_index=number,
            title='', source=record.get('source', 'manual'), is_primary=False,
            created_at=record.get('created_at', ''), updated_at=record.get('updated_at', record.get('created_at', '')))
        record.setdefault('metadata', {})
    edited['alternate_captions'].append({'caption': 'New model output', 'review_status': 'unreviewed',
        'metadata': {'review_status': 'unreviewed'}, 'generated_qa_pairs': [
            {'question': 'What color?', 'answer': 'Blue', 'review_status': 'unreviewed',
             'validation_status': 'accepted', 'metadata': {'review_decision': 'unreviewed'}}]})
    saved = store.save(ALICE, ctx(lease, 'legitimate-edit'), edited, complete=True)
    carried = store.image(ALICE, image['id'])['annotations']
    assert carried['alternate_captions'][0]['metadata'] == before['alternate_captions'][0]['metadata']
    assert _effective_review(carried['alternate_captions'][0], 'caption0')['review_status'] == 'unreviewed'
    assert _effective_review(carried['alternate_captions'][0]['generated_qa_pairs'][0], 'generated_qa')['review_status'] == 'unreviewed'
    assert carried['captions'][0]['review_decision'] == before['captions'][0]['review_decision']
    assert carried['captions'][0]['id'] == 'native-ui-0'
    assert carried['alternate_captions'][1] == edited['alternate_captions'][1]
    deleted = {**carried, 'captions': []}
    store.save(ALICE, ctx(saved, 'remove-whole-record'), deleted, complete=True)
    # Deleting a complete record never erases its prior immutable version.
    with store.read_connection() as con:
        import json
        prior = json.loads(con.execute('SELECT annotations_json FROM hosted_annotation_versions WHERE id=?', (saved['current_version'],)).fetchone()[0])
        assert prior['captions'] == carried['captions']


def test_machine_validation_requires_matching_server_result_and_preserves_content_binding(store):
    from dataclasses import asdict
    import json
    from types import SimpleNamespace
    from services.hosted.caption_jobs import CaptionJobs
    from services.hosted.instruction_review import bind_machine_validated_pairs
    from services.hosted.caption_export import _effective_review

    ds = dataset(store)
    image = assigned(store, ds, imported(store, ds))
    lease = store.acquire_editor(ALICE, image['id'], 'caption-tab')
    CaptionJobs(SimpleNamespace(store=store))  # Create the actual durable job schema.
    pair = bind_machine_validated_pairs([{'question': 'What is visible?', 'answer': 'A car.',
        'validation_status': 'machine_validated', 'review_status': 'unreviewed',
        'qa_verifier': {'grounded': True, 'score': 0.9},
        'validation': {'status': 'machine_validated', 'warnings': ['retained']},
        'metadata': {'verifier_model': 'actual-fixture-model'}}])[0]
    quality = {'passed': True, 'score': 0.9}
    evidence = {'sha256': 'e' * 64, 'size_bytes': 100}
    caption = {'caption': 'A car.', 'backend_job_id': 'verified-caption-job', 'caption_quality': quality,
               'evidence': evidence, 'generated_qa_pairs': [pair]}
    body = {**objects(ds), 'alternate_captions': [caption]}
    request = {'dataset_id': ds['id'], 'image_id': image['id'], 'expected_version': lease['current_version'],
               'generation': lease['generation'], 'editor_id': 'caption-tab'}
    result = {'captions': [{'caption': 'A car.', 'caption_quality': quality, 'generated_qa_pairs': [pair]}],
              'latest_caption': {'caption': 'A car.', 'generated_qa_pairs': [pair]}, 'evidence': evidence}

    def register(identifier, request_value, *, actor_id=ALICE.user_id, state='completed'):
        with store.transaction() as con:
            con.execute('INSERT INTO hosted_caption_jobs(id,actor_id,actor_json,request_json,runtime_identity,state,attempts_json,result_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)',
                (identifier, actor_id, json.dumps(asdict(ALICE)), json.dumps(request_value), 'fixture-model', state, '[]', json.dumps(result), store.clock(), store.clock()))

    error('machine_validation_requires_server_evidence', lambda: store.save(ALICE, ctx(lease, 'missing-result', 'caption-tab'), body))
    register('verified-caption-job', request)
    for name, changed_request, actor_id, state in (
            ('wrong-image', {**request, 'image_id': 'different'}, ALICE.user_id, 'completed'),
            ('wrong-version', {**request, 'expected_version': 'different'}, ALICE.user_id, 'completed'),
            ('wrong-owner', request, BOB.user_id, 'completed'),
            ('failed-result', request, ALICE.user_id, 'failed')):
        register(name, changed_request, actor_id=actor_id, state=state)
        changed = copy.deepcopy(body); changed['alternate_captions'][0]['backend_job_id'] = name
        code = 'machine_validation_source_changed' if name in {'wrong-image', 'wrong-version'} else 'machine_validation_requires_server_evidence'
        error(code, lambda: store.save(ALICE, ctx(lease, name, 'caption-tab'), changed))
    attempts = []
    changed = copy.deepcopy(body); changed['alternate_captions'][0]['generated_qa_pairs'][0]['qa_verifier']['grounded'] = False; attempts.append(changed)
    changed = copy.deepcopy(body); changed['alternate_captions'][0]['caption_quality']['passed'] = False; attempts.append(changed)
    changed = copy.deepcopy(body); changed['alternate_captions'][0]['generated_qa_pairs'][0]['answer'] = 'An invented building.'; attempts.append(changed)
    changed = copy.deepcopy(body); changed['alternate_captions'][0]['generated_qa_pairs'] *= 2; attempts.append(changed)
    for number, changed in enumerate(attempts):
        error('machine_validation_requires_server_evidence', lambda: store.save(ALICE, ctx(lease, 'tampered-' + str(number), 'caption-tab'), changed))
    for namespace in ('top', 'metadata', 'validation'):
        fake = {'question': 'Unverified?', 'answer': 'Yes'}
        if namespace == 'top': fake['validation_status'] = ' Machine-Validated '
        else: fake[namespace] = {'status' if namespace == 'validation' else 'validation_status': 'machine_validated'}
        changed = {**objects(ds), 'captions': [{'caption': 'Fake', 'generated_qa_pairs': [fake]}]}
        error('machine_validation_requires_server_evidence', lambda: store.save(ALICE, ctx(lease, 'alias-' + namespace, 'caption-tab'), changed))
    for namespace in ('top', 'metadata'):
        fake = {'caption': 'Fake accepted caption'}
        if namespace == 'top': fake['review_decision'] = 'machine_validated'
        else: fake['metadata'] = {'review_status': 'machine_validated'}
        error('instruction_review_requires_admin', lambda: store.save(ALICE, ctx(lease, 'caption-alias-' + namespace, 'caption-tab'), {**objects(ds), 'captions': [fake]}))

    saved = store.save(ALICE, ctx(lease, 'genuine-result', 'caption-tab'), body, complete=True)
    without_origin = copy.deepcopy(body)
    without_origin['alternate_captions'][0].pop('backend_job_id')
    error('machine_validation_requires_server_evidence', lambda: store.save(ALICE, ctx(saved, 'erase-origin', 'caption-tab'), without_origin))
    normalized = copy.deepcopy(body)
    normalized['alternate_captions'][0].update(id='native-ui-id', image='one.png', caption_index=0, metadata={})
    normalized['alternate_captions'][0]['generated_qa_pairs'][0].update(id='native-qa-id', image_key='one.png', created_at='', updated_at='')
    later = store.save(ALICE, ctx(saved, 'normalized-carry', 'caption-tab'), normalized, complete=True)
    edited = copy.deepcopy(normalized)
    edited['alternate_captions'][0]['generated_qa_pairs'][0]['answer'] = 'Two cars.'
    store.save(ALICE, ctx(later, 'edit-machine-wording', 'caption-tab'), edited, complete=True)
    retained = store.image(ALICE, image['id'])['annotations']['alternate_captions'][0]['generated_qa_pairs'][0]
    assert retained['metadata'] == pair['metadata']
    effective = _effective_review(retained, 'generated_qa')
    assert effective['validation_status'] == 'unknown' and effective['validation']['status'] == 'unknown'
    assert effective['qa_verifier'] == pair['qa_verifier']


def test_same_user_tabs_expiry_override_and_disabled_actor_cannot_apply_late_work(store):
    ds = dataset(store)
    image = assigned(store, ds, imported(store, ds))
    lease = store.acquire_editor(ALICE, image["id"], "tab-a")
    error("editor_lease_conflict", lambda: store.acquire_editor(ALICE, image["id"], "tab-b"))
    renewal = store.acquire_editor(ALICE, image["id"], "tab-a")
    assert renewal["generation"] == lease["generation"]
    store.test_clock[0] += 31
    replacement = store.acquire_editor(ALICE, image["id"], "tab-b")
    error("ownership_changed", lambda: store.save(ALICE, ctx(lease, "late-a", "tab-a"), objects(ds)))
    override = store.override(ADMIN, ctx(replacement, "override", "admin-tab"), "Reassign this image")
    error("image_not_found", lambda: store.save(ALICE, ctx(replacement, "late-b", "tab-b"), objects(ds)))
    assert override["current_version"] == image["current_version"]
    admin_lease = store.acquire_editor(ADMIN, image["id"], "admin-tab")
    reassigned = store.reassign(ADMIN, ctx(admin_lease, "reassign", "admin-tab"), BOB.user_id)
    assert reassigned["assignee_id"] == BOB.user_id
    error("image_not_found", lambda: store.image(ALICE, image["id"]))
    bob_lease = store.acquire_editor(BOB, image["id"], "tab")
    with store.transaction() as con:
        con.execute("UPDATE hosted_users SET disabled=1 WHERE id=?", (BOB.user_id,))
    error("actor_inactive", lambda: store.save(BOB, ctx(bob_lease, "disabled-save"), objects(ds)))
    error("actor_inactive", lambda: store.image(BOB, image["id"]))


def test_import_identity_conflicts_cross_dataset_visibility_and_atomic_assignment(store):
    first, second = dataset(store, "first"), dataset(store, "second")
    one = imported(store, first, "same", objects(first))
    other = imported(store, second, "same", objects(second))
    assert one["media_hash"] == other["media_hash"] and one["id"] != other["id"]
    assigned(store, first, one, ALICE)
    assigned(store, second, other, BOB)
    assert len(store.media_references()) == 1
    assert store.image(ALICE, one["id"])["annotations"] == objects(first)
    error("image_not_found", lambda: store.image(ALICE, other["id"]))
    duplicate = store.import_image(ADMIN, first["id"], one["media_hash"], "new-name.png", {"objects": []}, "conflict")
    assert duplicate["duplicate"] and duplicate["conflict_id"]
    assert store.image(ADMIN, one["id"])["annotations"] == objects(first)
    assert store.history(ADMIN, one["id"])["import_conflicts"][0]["annotations"] == {"objects": []}
    third, fourth = imported(store, first, "third"), imported(store, first, "fourth")
    stale = expected([third, fourth])
    live = store.acquire_editor(ADMIN, fourth["id"], "tab")
    released = store.release_editor(ADMIN, ctx(live, "release"))
    error("ownership_changed", lambda: store.assign(ADMIN, first["id"], [third["id"], fourth["id"]], BOB.user_id, "failed-batch", stale))
    assert store.image(ADMIN, third["id"])["state"] == "unassigned"
    store.assign(ADMIN, first["id"], [third["id"], fourth["id"]], BOB.user_id, "good-batch", expected([third, released]))
    error("invalid_class", lambda: store.import_image(ADMIN, first["id"], sha256(b"x").hexdigest(), "x.png", objects(second), "wrong-class"))


def test_class_addition_preserves_open_drafts_class_mapping_and_snapshot(store):
    ds = dataset(store)
    image = assigned(store, ds, imported(store, ds))
    lease = store.acquire_editor(ALICE, image["id"], "tab")
    error("invalid_class_name", lambda: store.request_class(ALICE, ds["id"], "Car\nBus", "Invalid class list", "request-invalid-name"))
    error("invalid_class_name", lambda: store.add_class(ADMIN, ds["id"], "Car\rBus", "add-invalid-name", 0))
    error("invalid_class_name", lambda: store.create_dataset(ADMIN, "bad names", ["Car\nBus"], "create-invalid-name"))
    request = store.request_class(ALICE, ds["id"], "bus", "A bus is present", "request-bus")
    error("admin_required", lambda: store.add_class(ALICE, ds["id"], "bus", "illegal-add", 0))
    resolved = store.resolve_class_request(ADMIN, request["id"], True, "allow-bus", 0)
    assert resolved["class_id"] and store.list_class_requests(ALICE)[0]["state"] == "approved"
    # An additive map change does not invalidate an unchanged image draft.
    saved = store.save(ALICE, ctx(lease, "after-class-added"), objects(ds), complete=True)
    store.submit(ALICE, ctx(saved, "submit"))
    review = store.acquire_editor(ADMIN, image["id"], "tab")
    store.approve(ADMIN, ctx(review, "approve"))
    snapshot = store.create_snapshot(ADMIN, ds["id"], "snapshot")
    store.rename_class(ADMIN, ds["id"], ds["classes"][0]["id"], "Car", "rename", 1)
    assert store.get_snapshot(ADMIN, snapshot["snapshot_id"])["manifest"]["classes"][0]["name"] == "car"
    assert store.snapshot(ADMIN, ds["id"])["classes"][0]["name"] == "Car"
    assert store.snapshot(ADMIN, ds["id"])["images"][0]["annotations"]["objects"][0]["class_id"] == ds["classes"][0]["id"]


def test_concurrent_retries_competing_writes_and_snapshot_capture(store):
    ds = dataset(store)
    image = imported(store, ds)
    lease = store.acquire_editor(ADMIN, image["id"], "tab")
    command = ctx(lease, "same-save")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(store.save, ADMIN, command, objects(ds), complete=True) for _ in range(2)]
        results = [future.result() for future in futures]
    assert results[0] == results[1]
    assert len(store.history(ADMIN, image["id"])["versions"]) == 2
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(store.save, ADMIN, ctx(results[0], "different-" + str(i)), objects(ds), complete=True) for i in range(2)]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except DomainError as exc:
                assert exc.code == "annotation_version_changed"
                outcomes.append(None)
    assert sum(outcome is not None for outcome in outcomes) == 1
    current = next(outcome for outcome in outcomes if outcome)
    with store.read_connection() as read:
        assert store._snapshot(read, ds["id"])["images"] == []
        store.approve(ADMIN, ctx(current, "approve"))
        assert store._snapshot(read, ds["id"])["images"] == []  # coherent read transaction
    assert len(store.snapshot(ADMIN, ds["id"])["images"]) == 1


def test_geometry_takeover_and_transaction_failure_leave_current_version_intact(store):
    ds = dataset(store)
    image = imported(store, ds)
    lease = store.acquire_editor(ADMIN, image["id"], "old")
    error("expected_version_required", lambda: store.acquire_editor(ADMIN, image["id"], "new", takeover=True))
    switched = store.acquire_editor(ADMIN, image["id"], "new", lease["current_version"], lease["generation"], takeover=True)
    assert switched["generation"] == lease["generation"] + 1
    error("ownership_changed", lambda: store.authorize_context(ADMIN, ctx(lease, "old-request", "old")))
    error("editor_lease_conflict", lambda: store.acquire_editor(ADMIN2, image["id"], "other", switched["current_version"], switched["generation"], takeover=True))
    assert store.authorize_context(ADMIN, ctx(switched, "read", "new"))["annotations"] == {"objects": []}
    for bbox, code in (([1, 2, 0, 5], "bbox_out_of_bounds"), ([1, 2, 100, 5], "bbox_out_of_bounds"),
                       ([True, 2, 10, 5], "invalid_bbox"), ([1, 2, 3], "invalid_bbox")):
        body = objects(ds)
        body["objects"][0]["bbox"] = bbox
        error(code, lambda: store.save(ADMIN, ctx(switched, "invalid", "new"), body))
    body = objects(ds)
    body["objects"][0]["polygon"] = [[1, 2], [101, 2], [3, 3]]
    error("polygon_out_of_bounds", lambda: store.save(ADMIN, ctx(switched, "bad-polygon", "new"), body))
    # Inject an actual SQLite commit-path failure after version insertion.
    with store.transaction() as con:
        con.execute("CREATE TRIGGER fail_save BEFORE UPDATE OF current_version ON hosted_images BEGIN SELECT RAISE(ABORT,'injected write failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match="injected write failure"):
        store.save(ADMIN, ctx(switched, "retryable", "new"), objects(ds), complete=True)
    assert store.image(ADMIN, image["id"])["current_version"] == image["current_version"]
    assert len(store.history(ADMIN, image["id"])["versions"]) == 1
    assert store.receipt(ADMIN, "retryable") is None
    with store.transaction() as con:
        con.execute("DROP TRIGGER fail_save")
    assert store.save(ADMIN, ctx(switched, "retryable", "new"), objects(ds), complete=True)["complete"]


def test_assignment_override_preserves_submitted_members_and_bulk_reports_each_version(store):
    ds = dataset(store)
    one, two = imported(store, ds, "one"), imported(store, ds, "two")
    group = store.assign(ADMIN, ds["id"], [one["id"], two["id"]], ALICE.user_id, "group", expected([one, two]))
    captured = expected(group["images"])
    lease = store.acquire_editor(ALICE, one["id"], "tab")
    draft = store.save(ALICE, ctx(lease, "save"), objects(ds), complete=True)
    store.submit(ALICE, ctx(draft, "submit"))
    recovered = store.override_assignment(ADMIN, group["assignment_id"], "recover-group", captured)
    assert [row["id"] for row in recovered["images"]] == [two["id"]]
    assert recovered["skipped"][0]["image_id"] == one["id"]
    assert store.image(ADMIN, one["id"])["state"] == "submitted"
    assert not store.list_images(ALICE)
    review = store.acquire_editor(ADMIN, one["id"], "tab")
    other = store.acquire_editor(ADMIN, two["id"], "tab")
    other = store.save(ADMIN, ctx(other, "negative"), {"objects": []}, negative=True)
    commands = [ctx(review, "approve-one"), ctx(other, "approve-two")]
    newer = store.save(ADMIN, ctx(other, "newer-negative"), {"objects": []}, negative=True)
    result = store.bulk_approve(ADMIN, commands)
    assert [item["status"] for item in result] == ["committed", "rejected"]
    assert result[1]["error"]["code"] == "annotation_version_changed"
    assert len(store.snapshot(ADMIN, ds["id"])["images"]) == 1
    assert store.image(ADMIN, two["id"])["current_version"] == newer["current_version"]
    assert store.bulk_approve(ADMIN, commands) == result


def test_whole_selection_recall_is_atomic_preserves_drafts_and_replays_after_reassignment(store):
    ds = dataset(store)
    one = assigned(store, ds, imported(store, ds, "one"), ALICE)
    two = assigned(store, ds, imported(store, ds, "two"), BOB)
    one = store.acquire_editor(ALICE, one["id"], "tab")
    one = store.save(ALICE, ctx(one, "alice-draft"), objects(ds))
    two = store.acquire_editor(BOB, two["id"], "tab")
    captured = expected([one, two])
    two = store.save(BOB, ctx(two, "bob-later-draft"), objects(ds, "bob-object"))
    error("annotation_version_changed", lambda: store.recall_images(ADMIN, ds["id"], [one["id"], two["id"]], captured, "Recover unfinished work", "recall-batch"))
    assert store.receipt(ADMIN, "recall-batch") is None
    assert store.image(ALICE, one["id"])["generation"] == one["generation"]
    assert store.image(BOB, two["id"])["editor_lease"]
    captured = expected([one, two])
    error("invalid_field", lambda: store.recall_images(ADMIN, ds["id"], [one["id"], two["id"]], captured, "  ", "empty-reason"))
    error("admin_required", lambda: store.recall_images(ALICE, ds["id"], [one["id"], two["id"]], captured, "Recover work", "not-admin"))
    result = store.recall_images(ADMIN, ds["id"], [one["id"], two["id"]], captured, "Recover unfinished work", "recall-batch")
    for before, after in zip([one, two], result["images"]):
        assert after["state"] == "unassigned" and after["assignee_id"] is None
        assert after["current_version"] == before["current_version"]
        assert after["generation"] == before["generation"] + 1 and after["editor_lease"] is None
    assert store.image(ADMIN, one["id"])["annotations"] == objects(ds)
    assert store.image(ADMIN, two["id"])["annotations"] == objects(ds, "bob-object")
    error("image_not_found", lambda: store.authorize_context(ALICE, ctx(one, "late-result")))
    error("image_not_found", lambda: store.save(BOB, ctx(two, "late-save"), objects(ds)))
    with store.read_connection() as con:
        assert not con.execute("SELECT 1 FROM hosted_editor_leases").fetchone()
        assert {row[0] for row in con.execute("SELECT state FROM hosted_assignment_members")} == {"overridden"}
    reassigned = store.assign(ADMIN, ds["id"], [one["id"], two["id"]], BOB.user_id, "assign-again", expected(result["images"]))
    reopened = HostedStore(store.path, clock=store.clock)
    assert reopened.recall_images(ADMIN, ds["id"], [one["id"], two["id"]], captured, "Recover unfinished work", "recall-batch") == result
    assert reopened.image(ADMIN, one["id"])["assignment_id"] == reassigned["assignment_id"]
    error("operation_id_reused", lambda: reopened.recall_images(ADMIN, ds["id"], [one["id"], two["id"]], captured, "Changed reason", "recall-batch"))


def test_recall_accepts_returned_work_but_rejects_submission_or_other_dataset_atomically(store):
    ds = dataset(store)
    one = assigned(store, ds, imported(store, ds, "one"))
    two = assigned(store, ds, imported(store, ds, "two"))
    drafts = []
    for image in [one, two]:
        leased = store.acquire_editor(ALICE, image["id"], "tab")
        saved = store.save(ALICE, ctx(leased, "save-" + image["id"]), objects(ds), complete=True)
        drafts.append(store.submit(ALICE, ctx(saved, "submit-" + image["id"])))
    review = store.acquire_editor(ADMIN, one["id"], "tab")
    returned = store.return_image(ADMIN, ctx(review, "return-one"), "Please check this car")
    submitted = drafts[1]
    error("image_not_overridable", lambda: store.recall_images(ADMIN, ds["id"], [returned["id"], submitted["id"]], expected([returned, submitted]), "Recover work", "cannot-recall-submission"))
    other_ds = dataset(store, "other")
    other = assigned(store, other_ds, imported(store, other_ds))
    error("image_not_overridable", lambda: store.recall_images(ADMIN, ds["id"], [returned["id"], other["id"]], expected([returned, other]), "Recover work", "cannot-cross-dataset"))
    assert store.image(ALICE, one["id"])["state"] == "returned"
    result = store.recall_images(ADMIN, ds["id"], [returned["id"]], expected([returned]), "Administrator will finish it", "recall-returned")
    assert result["images"][0]["current_version"] == returned["current_version"]
    assert store.image(ADMIN, two["id"])["state"] == "submitted"
    with store.read_connection() as con:
        assert con.execute("SELECT count(*) FROM hosted_submissions").fetchone()[0] == 2
        assert con.execute("SELECT feedback FROM hosted_reviews WHERE action='returned'").fetchone()[0] == "Please check this car"


def test_yolo_import_freezes_class_mapping_and_conflict_choice_preserves_other_content(store):
    ds = store.create_dataset(ADMIN, "import-map", ["Car", "Bus"], "import-map-dataset")
    image = imported(store, ds)
    leased = store.acquire_editor(ADMIN, image['id'], 'tab')
    current = store.save(ADMIN, ctx(leased, 'caption-before-import'), {'objects': [], 'caption': 'Human caption', 'alternate_captions': [{'caption': 'Earlier caption'}]})
    payload = dict(actor=ADMIN, dataset_id=ds['id'], media_hash=image['media_hash'], filename=image['filename'],
                   label_text='0 0.5 0.5 0.2 0.25\n', class_names=['Bus', 'Car'], expected_class_version=ds['class_version'],
                   operation_id='label-import', width=100, height=80, size_bytes=100)
    imported_labels = store.import_yolo_image(**payload)
    assert imported_labels['duplicate'] and imported_labels['conflict_id']
    assert store.image(ADMIN, image['id'])['current_version'] == current['current_version']
    conflict = store.history(ADMIN, image['id'])['import_conflicts'][0]
    assert conflict['annotations']['objects'][0]['class_id'] == ds['classes'][1]['id']
    assert conflict['annotations']['objects'][0]['bbox'] == pytest.approx([40, 30, 20, 20])
    assert conflict['annotations']['import_provenance']['label_text'] == payload['label_text']
    store.rename_class(ADMIN, ds['id'], ds['classes'][1]['id'], 'Coach', expected_class_version=ds['class_version'], operation_id='rename-bus')
    assert store.import_yolo_image(**payload) == imported_labels
    error('operation_id_reused', lambda: store.import_yolo_image(**{**payload, 'label_text': ''}))
    error('class_version_changed', lambda: store.import_yolo_image(**{**payload, 'operation_id': 'stale-labelmap'}))
    apply_context = ctx(current, 'use-import')
    result = store.resolve_import_conflict(ADMIN, apply_context, conflict['id'], 'use_imported_objects')
    annotations = store.image(ADMIN, image['id'])['annotations']
    assert annotations['caption'] == 'Human caption' and annotations['alternate_captions'] == [{'caption': 'Earlier caption'}]
    assert annotations['objects'] == conflict['annotations']['objects']
    assert store.resolve_import_conflict(ADMIN, apply_context, conflict['id'], 'use_imported_objects') == result
    error('import_conflict_already_reviewed', lambda: store.resolve_import_conflict(ADMIN, ctx(result, 'other-choice'), conflict['id'], 'keep_current'))
    empty = store.import_yolo_image(**{**payload, 'label_text': '', 'class_names': ['Car', 'Coach'], 'expected_class_version': ds['class_version'] + 1, 'operation_id': 'empty-import'})
    error('empty_annotation_requires_negative_review', lambda: store.resolve_import_conflict(ADMIN, ctx(result, 'empty-choice'), empty['conflict_id'], 'use_imported_objects'))
    kept = store.resolve_import_conflict(ADMIN, ctx(result, 'keep-labels'), empty['conflict_id'], 'keep_current')
    assert kept['current_version'] == result['current_version']
    assert len(store.history(ADMIN, image['id'])['import_conflict_decisions']) == 2


def test_authenticated_session_revocation_is_rechecked_inside_store_commit(tmp_path):
    from services.hosted.auth import AuthService

    now = [1000.0]
    store = HostedStore(tmp_path / "auth.sqlite3", clock=lambda: now[0])
    auth = AuthService(store, clock=lambda: now[0], session_seconds=100)
    auth.bootstrap("administrator", "a-long-test-password")
    login = auth.login("administrator", "a-long-test-password", address="127.0.0.1")
    actor = auth.authenticate(login.token)
    ds = store.create_dataset(actor, "test", ["car"], "create")
    image = store.import_image(actor, ds["id"], sha256(b"pixel").hexdigest(), "one.png", {}, "import", width=100,height=80)
    lease = store.acquire_editor(actor, image["id"], "tab")
    auth.logout(login.token)
    error("login_required", lambda: store.save(actor, ctx(lease, "late-save"), objects(ds)))
    error("login_required", lambda: store.receipt(actor, "import"))
    login = auth.login("administrator", "a-long-test-password", address="127.0.0.1")
    actor = auth.authenticate(login.token)
    now[0] += 101
    error("login_required", lambda: store.authorize_context(actor, ctx(lease, "expired")))
    assert store.image(Actor(actor.user_id, "admin"), image["id"])["current_version"] == image["current_version"]


def test_queue_approval_is_atomic_idempotent_and_never_bypasses_an_editor(store):
    ds = dataset(store)
    image = assigned(store, ds, imported(store, ds))
    image = store.acquire_editor(ALICE, image["id"], "tab")
    image = store.save(ALICE, ctx(image, "complete"), objects(ds), complete=True)
    image = store.submit(ALICE, ctx(image, "submit"))
    original = ctx(image, "queue-approval", "queue")
    edit = store.acquire_editor(ADMIN2, image["id"], "other-admin")
    current_request = ctx(edit, "queue-blocked", "queue")
    error("editor_lease_conflict", lambda: store.approve_from_review_queue(ADMIN, current_request))
    error("ownership_changed", lambda: store.approve_from_review_queue(ADMIN, original))
    store.release_editor(ADMIN2, ctx(edit, "release", "other-admin"))
    current = store.image(ADMIN, image["id"])
    request = ctx(current, "queue-current", "queue")
    error("admin_required", lambda: store.approve_from_review_queue(ALICE, request))
    approved = store.approve_from_review_queue(ADMIN, request)
    assert approved["state"] == "confirmed"
    assert HostedStore(store.path, clock=store.clock).approve_from_review_queue(ADMIN, request) == approved
    assert len(store.snapshot(ADMIN, ds["id"])["images"]) == 1
