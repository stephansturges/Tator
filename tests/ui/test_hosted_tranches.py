"""Composed browser proofs for named work, exact retries and progressive queues.

Uses only the disposable hosted workflow fixture. Native CUDA qualification is
separate; these assertions do not claim inference or user acceptance.
"""
import asyncio
import os
import uuid

import pytest

from test_hosted_workflow import hosted_origin, sign_in, draw

pytestmark = pytest.mark.skipif(os.environ.get("TATOR_HOSTED_BROWSER_TESTS") != "1", reason="Explicit isolated browser run required")


async def api(page, path, body=None):
    return await page.evaluate("([path,body]) => window.TatorHostedDashboard.api(path, body === null ? {} : {method:'POST',body})", [path, body])


def test_named_tranche_create_retry_partial_submit_review_and_private_progress(hosted_origin, tmp_path):
    asyncio.run(_named_flow(hosted_origin, tmp_path))


async def _named_flow(origin, output):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        admin = await browser.new_page(viewport={"width": 1440, "height": 1000})
        alice = await browser.new_page(viewport={"width": 1440, "height": 1000})
        bob = await browser.new_page()
        errors = []
        for page in (admin, alice, bob): page.on("pageerror", lambda error: errors.append(str(error)))
        await sign_in(admin, origin, "admin")
        dataset = await admin.locator("#datasetChoice").input_value()
        users = (await api(admin, "/users"))["users"]
        alice_id = next(user["id"] for user in users if user["username"] == "alice")
        await admin.get_by_role("button", name="Incoming", exact=True).click()
        await admin.locator("#incomingFilter").select_option("unallocated")
        await admin.get_by_role("button", name="Apply filters", exact=True).click()
        await expect(admin.locator("#imageGrid .hosted-card")).to_have_count(3)
        for check in (admin.locator("#imageGrid input").nth(0), admin.locator("#imageGrid input").nth(1)): await check.check()
        await admin.get_by_role("button", name="Create tranche from selection").click()
        dialog = admin.locator("#trancheDialog")
        await dialog.get_by_label("Name", exact=True).fill("Road junctions")
        await dialog.get_by_label("Instructions for this tranche").fill("Check the parked cars and confirm empty scenes.")
        await dialog.get_by_label("Annotator", exact=True).select_option(alice_id)
        await dialog.get_by_label("When should the annotator receive it?").select_option("active")
        sent = []
        async def lose_reply(route):
            sent.append(route.request.post_data_json)
            response = await route.fetch(); assert response.status == 200
            await route.abort("failed")
        endpoint = f"**/api/hosted/datasets/{dataset}/tranches"
        await admin.route(endpoint, lose_reply)
        await dialog.get_by_role("button", name="Create tranche", exact=True).click()
        await expect(admin.locator("#trancheRetry")).to_be_enabled()
        assert len(sent) == 2 and sent[0] == sent[1]
        await admin.unroute(endpoint, lose_reply)
        await admin.reload()
        await expect(admin.locator("#workspace")).to_be_visible()
        await admin.locator("#trancheRetry").click()
        await expect(admin.locator("#tranchePanel")).to_contain_text("Road junctions")
        await expect(admin.locator("#trancheRecovery")).to_be_hidden()
        tranches = (await api(admin, f"/tranches?dataset_id={dataset}"))["tranches"]
        tranche = next(t for t in tranches if t["name"] == "Road junctions")
        assert tranche["lifecycle"] == "active" and tranche["counts"]["to_finish"] == 2
        await admin.screenshot(path=str(output / "admin-tranche.png"))
        await sign_in(alice, origin, "alice")
        card = alice.locator(".tranche-card").filter(has=alice.get_by_role("heading", name="Road junctions", exact=True))
        await card.get_by_role("button", name="Open tranche", exact=True).click()
        await expect(alice.locator(".tranche-grid")).to_have_count(0)
        await alice.get_by_role("button", name="Continue annotation", exact=True).click()
        await expect(alice.locator("#hostedSave")).to_be_enabled()
        assert f"tranche={tranche['id']}" in alice.url and "queue=work" in alice.url
        first = await alice.evaluate("window.TatorHosted.imageId")
        await expect(alice.locator("#hostedTrancheInstructions")).to_contain_text("parked cars")
        await alice.locator("#hostedNegative").check()
        await alice.locator("#hostedSubmit").click()
        await expect(alice.locator("#hostedSave")).to_be_enabled()
        await alice.wait_for_function("id => window.TatorHosted?.imageId !== id", arg=first)
        assert f"tranche={tranche['id']}" in alice.url
        second = await alice.evaluate("window.TatorHosted.imageId")
        assert second != first
        await draw(alice, .2, .2, .4, .4)
        retained = await alice.evaluate("window.TatorHostedCanvas.snapshot()")
        await admin.reload(); await expect(admin.locator("#tranchePanel")).to_contain_text("Road junctions")
        await admin.get_by_role("button", name="Add Incoming images", exact=True).click()
        await admin.locator(".tranche-picker input").check()
        await admin.locator("#trancheDialog").get_by_role("button", name="Add selected images", exact=True).click()
        await admin.get_by_role("button", name="Edit name & instructions", exact=True).click()
        await admin.locator("#trancheDialog").get_by_label("Instructions for this tranche").fill("Updated review guidance: include distant cars.")
        await admin.locator("#trancheDialog").get_by_role("button", name="Save changes", exact=True).click()
        await alice.locator("#hostedRefreshQueue").click()
        await expect(alice.locator("#hostedSaveState")).to_have_text("Work list updated. Your current annotations are unchanged.")
        assert await alice.evaluate("window.TatorHostedCanvas.snapshot()") == retained
        assert await alice.evaluate("window.TatorHosted.imageId") == second
        await expect(alice.locator("#imageList option")).to_have_count(2)
        await expect(alice.locator("#hostedTrancheInstructions")).to_contain_text("distant cars")
        await sign_in(bob, origin, "bob")
        assert (await bob.request.get(f"{origin}/api/hosted/images/{second}/media")).status == 404
        await expect(bob.locator("#tranchePanel")).not_to_contain_text("Road junctions")
        await admin.reload(); await expect(admin.locator("#tranchePanel")).to_contain_text("1 waiting for review")
        await admin.get_by_role("button", name="Review 1 image", exact=True).click()
        await expect(admin.locator("#hostedApprove")).to_be_enabled()
        await admin.locator("#hostedApprove").click()
        await expect(admin.locator("#workspace")).to_be_visible()
        assert f"tranche={tranche['id']}" in admin.url
        await expect(admin.locator("#tranchePanel")).to_contain_text("1 confirmed")
        await alice.locator("#hostedBack").click()
        await expect(alice.locator("#tranchePanel")).to_contain_text("1 confirmed")
        members = await api(alice, f"/tranches/{tranche['id']}/members")
        approved = next(m for m in members["members"] if m["image_id"] == first)
        assert not approved["can_view"] and approved["image"] is None
        assert (await alice.request.get(f"{origin}/api/hosted/images/{first}/media")).status == 404
        await alice.set_viewport_size({"width": 390, "height": 844})
        await alice.screenshot(path=str(output / "annotator-tranche-narrow.png"))
        assert await alice.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert not errors
        await browser.close()


@pytest.mark.parametrize("hosted_origin", ["tranche_queue"], indirect=True)
def test_scope_keysets_beyond_500_and_append_preserves_current_draft(hosted_origin, tmp_path):
    asyncio.run(_queue_flow(hosted_origin, tmp_path))


async def _queue_flow(origin, output):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 1000})
        await sign_in(page, origin, "admin")
        dataset = await page.locator("#datasetChoice").input_value()
        rows, cursor = [], None
        while True:
            params = f"dataset_id={dataset}&queue=unallocated&limit=500" + (f"&cursor={cursor}" if cursor else "")
            response = await api(page, f"/queue?{params}"); rows.extend(response["images"]); cursor = response["next_cursor"]
            if not cursor: break
        assert len(rows) > 500
        created = await api(page, f"/datasets/{dataset}/tranches", {"operation_id": str(uuid.uuid4()), "name": "Long draft", "image_ids": [r["id"] for r in rows[:500]], "expected": {r["id"]: {"expected_version": r["current_version"], "generation": r["generation"]} for r in rows[:500]}})
        tranche = created["tranche"]
        tail = rows[500:]
        await api(page, f"/tranches/{tranche['id']}/members/add", {"operation_id": str(uuid.uuid4()), "expected_revision": tranche["revision"], "image_ids": [r["id"] for r in tail], "expected": {r["id"]: {"expected_version": r["current_version"], "generation": r["generation"]} for r in tail}})
        # Deep link must start at its own anchor without fetching earlier pages.
        queue_requests = []; page.on("request", lambda req: queue_requests.append(req.url) if "/api/hosted/queue?" in req.url else None)
        await page.goto(f"{origin}/annotate?image={rows[499]['id']}&tranche={tranche['id']}&queue=draft")
        await expect(page.locator("#hostedSave")).to_be_enabled()
        assert len(queue_requests) <= 2 and "anchor_id=" in queue_requests[0]
        assert await page.locator("#imageList option").count() <= 200
        await draw(page, .2, .2, .4, .4)
        snapshot = await page.evaluate("window.TatorHostedCanvas.snapshot()")
        assert snapshot["objects"]
        await page.locator("#hostedMoreImages").click()
        assert (await page.evaluate("window.TatorHostedCanvas.snapshot()")) == snapshot
        await page.evaluate("window.TatorHosted.step(1)")
        await page.wait_for_function("id => window.TatorHosted?.imageId === id && window.TatorHostedCanvas.ready()", arg=rows[500]["id"])
        await page.locator("#hostedNegative").check()
        await page.locator("#hostedApprove").click()
        await expect(page.locator("#hostedSave")).to_be_enabled()
        await page.wait_for_function("id => window.TatorHosted?.imageId === id", arg=rows[501]["id"])
        assert "queue=draft" in page.url and f"tranche={tranche['id']}" in page.url
        await page.screenshot(path=str(output / "scoped-draft-queue.png"))
        await browser.close()


def test_uploaded_tranche_membership_retry_and_membership_controls(hosted_origin, tmp_path):
    asyncio.run(_upload_membership(hosted_origin, tmp_path))


async def _upload_membership(origin, output):
    from PIL import Image
    from playwright.async_api import async_playwright, expect
    source = output / "incoming-upload.png"
    Image.new("RGB", (320, 240), (148, 60, 20)).save(source)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 950})
        await sign_in(page, origin, "admin")
        dataset = await page.locator("#datasetChoice").input_value()
        await page.get_by_role("button", name="Tranches", exact=True).click()
        await page.get_by_role("button", name="New tranche", exact=True).click()
        await page.locator("#trancheDialog").get_by_label("Name", exact=True).fill("Upload draft")
        await page.locator("#trancheDialog").get_by_role("button", name="Create tranche", exact=True).click()
        await expect(page.locator("#tranchePanel")).to_contain_text("Upload draft")
        tranche = next(t for t in (await api(page, f"/tranches?dataset_id={dataset}"))["tranches"] if t["name"] == "Upload draft")
        calls = []
        async def lose_reply(route):
            calls.append(route.request.post_data_json)
            response = await route.fetch(); assert response.status == 200
            await route.abort("failed")
        endpoint = f"**/api/hosted/tranches/{tranche['id']}/members/add"
        await page.route(endpoint, lose_reply)
        async with page.expect_file_chooser() as chooser:
            await page.get_by_role("button", name="Upload images here", exact=True).click()
        await (await chooser.value).set_files(source)
        await expect(page.locator("#trancheRetry")).to_be_enabled()
        assert len(calls) == 2 and calls[0] == calls[1]
        await page.unroute(endpoint, lose_reply)
        await page.reload(); await expect(page.locator("#trancheRetry")).to_be_enabled()
        await page.locator("#trancheRetry").click()
        await expect(page.locator("#trancheRecovery")).to_be_hidden()
        assert not await page.evaluate("Object.keys(sessionStorage).some(k => k.startsWith('tator.hosted.import:'))")
        await expect(page.locator("#tranchePanel .hosted-card")).to_have_count(1)
        await page.get_by_role("button", name="Edit name & instructions", exact=True).click()
        await page.locator("#trancheDialog").get_by_label("Name", exact=True).fill("Reviewed upload")
        await page.locator("#trancheDialog").get_by_role("button", name="Save changes", exact=True).click()
        await expect(page.locator("#tranchePanel")).to_contain_text("Reviewed upload")
        await page.locator("#tranchePanel .hosted-card input").check()
        await page.get_by_role("button", name="Remove from tranche", exact=True).click()
        await page.locator("#trancheDialog").get_by_label("Reason", exact=True).fill("Hold for another task.")
        await page.locator("#trancheDialog").get_by_role("button", name="Remove from tranche", exact=True).click()
        await expect(page.locator("#tranchePanel")).to_contain_text("1 removed")
        await page.get_by_role("button", name="Close tranche", exact=True).click()
        await expect(page.locator("#tranchePanel")).to_contain_text("Closed")
        await page.get_by_role("button", name="Reopen as Draft", exact=True).click()
        await expect(page.locator("#tranchePanel")).to_contain_text("Draft")
        await expect(page.locator("#tranchePanel")).to_contain_text("0 currently retained")
        await page.get_by_role("button", name="Add Incoming images", exact=True).click()
        await page.locator(".tranche-picker label").filter(has_text="incoming-upload.png").locator("input").check()
        await page.locator("#trancheDialog").get_by_role("button", name="Add selected images", exact=True).click()
        await expect(page.locator("#tranchePanel")).to_contain_text("1 currently retained")
        current = await api(page, f"/tranches/{tranche['id']}")
        assert current["counts"]["ever_included"] == 1 and current["counts"]["not_assigned"] == 1
        await browser.close()


@pytest.mark.parametrize('case', ['same_owner_move', 'incoming_reserved'])
def test_retained_draft_recovers_after_scope_change(hosted_origin, tmp_path, case):
    asyncio.run(_scope_recovery(hosted_origin, tmp_path, case))


async def _scope_recovery(origin, output, case):
    import sqlite3
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        admin = await browser.new_page(); worker = await browser.new_page(viewport={'width': 1440, 'height': 1000})
        await sign_in(admin, origin, 'admin'); dataset = await admin.locator('#datasetChoice').input_value()
        users = (await api(admin, '/users'))['users']; alice_id = next(u['id'] for u in users if u['username'] == 'alice')
        selected = (await api(admin, f'/queue?dataset_id={dataset}&queue=unallocated'))['images']
        def capture(images): return {'image_ids': [i['id'] for i in images], 'expected': {i['id']: {'expected_version': i['current_version'], 'generation': i['generation']} for i in images}}
        async def create(name, images, active=False):
            result = await api(admin, f'/datasets/{dataset}/tranches', {'operation_id': str(uuid.uuid4()), 'name': name, **capture(images)})
            if active:
                result = await api(admin, f"/tranches/{result['tranche']['id']}/activate", {'operation_id': str(uuid.uuid4()), 'expected_revision': result['tranche']['revision'], 'annotator_id': alice_id, **capture(result['images'])})
            return result['tranche']
        if case == 'same_owner_move':
            source = await create('Original work', selected[:1], True)
            destination = await create('New work', selected[1:2], True)
            await sign_in(worker, origin, 'alice')
            url = f"{origin}/annotate?image={selected[0]['id']}&tranche={source['id']}&queue=work"
        else:
            source = None; destination = await create('Reserved later', [])
            await sign_in(worker, origin, 'admin')
            url = f"{origin}/annotate?image={selected[0]['id']}&queue=unallocated"
        await worker.goto(url); await expect(worker.locator('#hostedSave')).to_be_enabled()
        image_id = selected[0]['id']; save_route = f'**/api/hosted/images/{image_id}/save'
        await worker.route(save_route, lambda route: route.abort('failed'))
        await draw(worker, .15, .2, .45, .5)
        retained = await worker.evaluate('window.TatorHostedCanvas.snapshot()')
        await worker.locator('#hostedSave').click()
        await worker.wait_for_function("Object.keys(sessionStorage).some(k => k.startsWith('tator.hosted.draft:') && JSON.parse(sessionStorage[k]).pending)")
        fresh = await api(admin, f'/images/{image_id}')
        if case == 'same_owner_move':
            await api(admin, f"/tranches/{source['id']}/move", {'operation_id': str(uuid.uuid4()), 'expected_revision': source['revision'], 'destination_id': destination['id'], 'expected_destination_revision': destination['revision'], 'reason': 'Keep the same author in a different group', **capture([fresh])})
        else:
            # Only the disposable fixture clock is advanced; no product lease bypass.
            with sqlite3.connect(output / 'state' / 'hosted.sqlite3') as con:
                con.execute('UPDATE hosted_editor_leases SET expires_at=0 WHERE image_id=?', (image_id,))
            await api(admin, f"/tranches/{destination['id']}/members/add", {'operation_id': str(uuid.uuid4()), 'expected_revision': destination['revision'], **capture([fresh])})
        await worker.unroute(save_route)
        response = await worker.reload(); assert response.status == 200
        await expect(worker.locator('#hostedResolve')).to_be_visible()
        await expect(worker.locator('#hostedScopeNotice')).to_contain_text('left its previous')
        await expect(worker.locator('#hostedMoreImages')).to_be_disabled()
        # Image-only links derive the new scope but must retain the old draft too.
        if case == 'same_owner_move':
            await worker.goto(f'{origin}/annotate?image={image_id}')
            await expect(worker.locator('#hostedResolve')).to_be_visible()
        worker.on('dialog', lambda dialog: dialog.accept())
        await worker.locator('#hostedResolve').click()
        await expect(worker.locator('#hostedResolve')).to_have_count(0)
        await expect(worker.locator('#hostedSaveState')).to_have_text('All changes saved')
        assert await worker.evaluate('window.TatorHostedCanvas.snapshot()') == retained
        assert f"tranche={destination['id']}" in worker.url
        current = await api(admin, f'/images/{image_id}')
        assert current['annotations'] == retained
        await expect(worker.locator('#hostedBreadcrumb')).to_contain_text('Reserved later' if source is None else 'New work')
        await worker.screenshot(path=str(output / f'{case}-recovered.png'))
        await browser.close()


@pytest.mark.parametrize('handoff', ['approve', 'archive'])
def test_lost_draft_handoff_reply_recovers_after_scope_exit(hosted_origin, handoff):
    asyncio.run(_lost_draft_handoff(hosted_origin, handoff))


async def _lost_draft_handoff(origin, handoff):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True); page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
        await sign_in(page, origin, 'admin'); dataset = await page.locator('#datasetChoice').input_value()
        image = (await api(page, f'/queue?dataset_id={dataset}&queue=unallocated'))['images'][0]
        created = await api(page, f'/datasets/{dataset}/tranches', {'operation_id': str(uuid.uuid4()), 'name': 'Lost reply trial', 'image_ids': [image['id']], 'expected': {image['id']: {'expected_version': image['current_version'], 'generation': image['generation']}}})
        tranche = created['tranche']; await page.goto(f"{origin}/annotate?image={image['id']}&tranche={tranche['id']}&queue=draft")
        await expect(page.locator('#hostedSave')).to_be_enabled()
        await page.locator('#hostedNegative').check(); await page.locator('#hostedSave').click(); await expect(page.locator('#hostedSaveState')).to_have_text('All changes saved')
        captured = []
        async def lose(route):
            captured.append(route.request.post_data_json)
            response = await route.fetch(); assert response.status == 200
            await route.abort('failed')
        endpoint = f"**/api/hosted/images/{image['id']}/{handoff}"
        await page.route(endpoint, lose)
        page.on('dialog', lambda dialog: dialog.accept())
        await page.locator('#hostedApprove' if handoff == 'approve' else '#hostedArchive').click()
        await expect(page.locator('#hostedHandoffRetry')).to_be_visible()
        assert 'recovery=' in page.url
        await page.unroute(endpoint, lose)
        response = await page.reload(); assert response.status == 200
        await expect(page.locator('#workspace')).to_be_visible()
        assert f"tranche={tranche['id']}" in page.url
        assert not await page.evaluate("Object.keys(sessionStorage).some(k => k.startsWith('tator.hosted.draft:'))")
        detail = await api(page, f"/tranches/{tranche['id']}")
        assert detail['counts']['approved' if handoff == 'approve' else 'archived'] == 1
        assert len(captured) == 1
        await browser.close()


def test_incoming_exact_selection_and_global_review_filters(hosted_origin):
    asyncio.run(_incoming_and_review_filters(hosted_origin))


async def _incoming_and_review_filters(origin):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True); admin = await browser.new_page(); alice = await browser.new_page()
        await sign_in(admin, origin, 'admin'); dataset = await admin.locator('#datasetChoice').input_value()
        users = (await api(admin, '/users'))['users']; alice_id = next(u['id'] for u in users if u['username'] == 'alice'); bob_id = next(u['id'] for u in users if u['username'] == 'bob')
        await admin.get_by_role('button', name='Incoming', exact=True).click()
        await expect(admin.locator('#imageGrid .hosted-card')).to_have_count(4)
        await expect(admin.locator('#imageGrid')).to_contain_text('Tranche:')
        await expect(admin.locator('#imageGrid input[type=checkbox]')).to_have_count(3)
        # Range selection cannot select the assigned row mixed into Incoming.
        await admin.get_by_label('Select frame-1.png', exact=True).check()
        await admin.get_by_label('Select frame-3.png', exact=True).click(modifiers=['Shift'])
        await expect(admin.locator('#selectionCount')).to_have_text('3 images selected')
        await admin.locator('#imageSearch').fill('frame-1.png')
        await admin.get_by_role('button', name='Apply filters', exact=True).click()
        await expect(admin.locator('#imageGrid .hosted-card')).to_have_count(1)
        await expect(admin.locator('#selectionBar')).to_be_hidden()
        confirmations = []
        async def confirm(dialog): confirmations.append(dialog.message); await dialog.accept()
        admin.on('dialog', confirm)
        await admin.get_by_role('button', name='Select all matching', exact=True).click()
        await expect(admin.locator('#selectionCount')).to_have_text('1 image selected')
        assert any('all 1 eligible matching image,' in message for message in confirmations)
        create_requests = []
        async def capture(route): create_requests.append(route.request.post_data_json); await route.continue_()
        await admin.route(f'**/api/hosted/datasets/{dataset}/tranches', capture)
        await admin.get_by_role('button', name='Create tranche from selection', exact=True).click()
        await admin.locator('#trancheDialog').get_by_label('Name', exact=True).fill('Filtered review')
        await admin.locator('#trancheDialog').get_by_label('Annotator', exact=True).select_option(alice_id)
        await admin.locator('#trancheDialog').get_by_label('When should the annotator receive it?').select_option('active')
        await admin.locator('#trancheDialog').get_by_role('button', name='Create tranche', exact=True).click()
        await expect(admin.locator('#tranchePanel')).to_contain_text('Filtered review')
        assert len(create_requests) == 1
        request = create_requests[0]; assert len(request['image_ids']) == 1 and set(request['expected']) == set(request['image_ids'])
        tranches = (await api(admin, f'/tranches?dataset_id={dataset}'))['tranches']; target = next(t for t in tranches if t['name'] == 'Filtered review')
        await sign_in(alice, origin, 'alice')
        # Two separate tranches submitted through normal authenticated HTTP.
        for tranche in tranches:
            work = (await api(alice, f"/queue?dataset_id={dataset}&tranche_id={tranche['id']}&queue=work"))['images']
            if not work: continue
            image = work[0]; editor = str(uuid.uuid4())
            current = await api(alice, f"/images/{image['id']}/editor", {'editor_id': editor, 'expected_version': image['current_version'], 'generation': image['generation']})
            def context(row): return {'editor_id': editor, 'expected_version': row['current_version'], 'generation': row['generation'], 'operation_id': str(uuid.uuid4())}
            saved = await api(alice, f"/images/{image['id']}/save", {**context(current), 'annotations': {'objects': []}, 'complete': True, 'negative': True})
            await api(alice, f"/images/{image['id']}/submit", context(saved))
        await admin.get_by_role('button', name='Review', exact=True).click()
        await admin.locator('#imageSearch').fill('')
        await admin.get_by_role('button', name='Apply filters', exact=True).click()
        await expect(admin.locator('#imageGrid .hosted-card')).to_have_count(2)
        await admin.locator('#reviewAnnotator').select_option(bob_id)
        await admin.get_by_role('button', name='Apply filters', exact=True).click()
        await expect(admin.locator('#imageGrid .hosted-card')).to_have_count(0)
        await admin.locator('#reviewAnnotator').select_option(alice_id)
        await admin.locator('#reviewTranche').select_option(target['id'])
        await admin.get_by_role('button', name='Apply filters', exact=True).click()
        await expect(admin.locator('#imageGrid .hosted-card')).to_have_count(1)
        await expect(admin.locator('#imageMatchCount')).to_contain_text('1 matching image ')
        await admin.get_by_role('button', name='Select all matching', exact=True).click()
        await admin.get_by_role('button', name='Approve selected images', exact=True).click()
        await expect(admin.locator('#imageGrid .hosted-card')).to_have_count(0)
        review = await api(admin, f'/queue?dataset_id={dataset}&queue=review&include_total=true')
        assert review['total'] == 1 and review['images'][0]['id'] not in request['image_ids']
        approved = await api(admin, f'/queue?dataset_id={dataset}&queue=confirmed&include_total=true')
        assert approved['total'] == 1 and approved['images'][0]['id'] == request['image_ids'][0]
        await browser.close()


def test_delayed_queue_and_selection_cannot_cross_view_or_dataset(hosted_origin):
    asyncio.run(_delayed_projection(hosted_origin))


async def _delayed_projection(origin):
    """Real responses arrive while the next Review filter request is held."""
    from urllib.parse import parse_qs, urlsplit
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await sign_in(page, origin, 'admin')
        first = await page.locator('#datasetChoice').input_value()
        second = await api(page, '/datasets', {'name': 'Other dataset', 'classes': ['Vehicle'], 'operation_id': str(uuid.uuid4())})
        await page.evaluate('window.TatorHostedDashboard.refresh()')
        await page.select_option('#datasetChoice', first)
        await expect(page.locator('#imageGrid .hosted-card')).to_have_count(0)
        queue_started, queue_release, queue_done = asyncio.Event(), asyncio.Event(), asyncio.Event()
        filters_started, filters_release = asyncio.Event(), asyncio.Event()

        async def hold_queue(route):
            response = await route.fetch()
            queue_started.set(); await queue_release.wait()
            await route.fulfill(response=response); queue_done.set()

        async def hold_filters(route):
            response = await route.fetch()
            filters_started.set(); await filters_release.wait()
            await route.fulfill(response=response)

        await page.route('**/api/hosted/queue?*', hold_queue)
        await page.route('**/api/hosted/tranches?*', hold_filters)
        await page.get_by_role('button', name='Incoming', exact=True).click()
        await asyncio.wait_for(queue_started.wait(), 5)
        await page.get_by_role('button', name='Review', exact=True).click()
        await asyncio.wait_for(filters_started.wait(), 5)
        # The previous Incoming response contains four rows. The new Review
        # request cannot start until its filter metadata is released.
        async with page.expect_response(lambda r: '/api/hosted/queue?' in r.url):
            queue_release.set()
        await asyncio.wait_for(queue_done.wait(), 5)
        await expect(page.locator('#workspaceTitle')).to_have_text('Review')
        await expect(page.locator('#imageGrid .hosted-card')).to_have_count(0)
        await expect(page.locator('#selectionBar')).to_be_hidden()
        await page.unroute('**/api/hosted/queue?*', hold_queue)
        filters_release.set()
        await expect(page.locator('#imageMatchCount')).to_contain_text('0 matching images')
        await page.unroute('**/api/hosted/tranches?*', hold_filters)

        await page.get_by_role('button', name='Incoming', exact=True).click()
        await expect(page.locator('#imageGrid .hosted-card')).to_have_count(4)
        selection_started, selection_release, selection_done = asyncio.Event(), asyncio.Event(), asyncio.Event()
        filters_started.clear(); filters_release.clear()
        confirmations = []
        async def confirm(dialog):
            confirmations.append(dialog.message); await dialog.accept()
        page.on('dialog', confirm)

        async def hold_selection(route):
            query = parse_qs(urlsplit(route.request.url).query)
            if query.get('limit') != ['500']:
                await route.continue_(); return
            response = await route.fetch()
            selection_started.set(); await selection_release.wait()
            await route.fulfill(response=response); selection_done.set()

        await page.route('**/api/hosted/queue?*', hold_selection)
        await page.get_by_role('button', name='Select all matching', exact=True).click()
        await asyncio.wait_for(selection_started.wait(), 5)
        await page.get_by_role('button', name='Review', exact=True).click()
        await expect(page.locator('#imageMatchCount')).to_contain_text('0 matching images')
        await page.route('**/api/hosted/tranches?*', hold_filters)
        await page.select_option('#datasetChoice', second['id'])
        await asyncio.wait_for(filters_started.wait(), 5)
        async with page.expect_response(lambda r: '/api/hosted/queue?' in r.url and 'limit=500' in r.url):
            selection_release.set()
        await asyncio.wait_for(selection_done.wait(), 5)
        # The request belongs to the first dataset. It must not even ask for
        # confirmation in the new dataset, much less populate its selection.
        await expect(page.get_by_role('button', name='Select all matching', exact=True)).to_be_enabled()
        assert confirmations == []
        assert await page.evaluate('window.TatorHostedDashboard.selectedImages.length') == 0
        await expect(page.locator('#imageGrid .hosted-card')).to_have_count(0)
        await expect(page.locator('#selectionBar')).to_be_hidden()
        filters_release.set()
        await expect(page.locator('#imageMatchCount')).to_contain_text('0 matching images')
        assert await page.locator('#datasetChoice').input_value() == second['id']
        await browser.close()


def test_global_review_draft_recovery_keeps_cross_tranche_queue(hosted_origin):
    asyncio.run(_global_review_recovery(hosted_origin))


async def _global_review_recovery(origin):
    from urllib.parse import parse_qs, urlsplit
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        admin = await browser.new_page(viewport={'width': 1440, 'height': 1000})
        alice = await browser.new_page()
        await sign_in(admin, origin, 'admin')
        dataset = await admin.locator('#datasetChoice').input_value()
        users = (await api(admin, '/users'))['users']
        alice_id = next(user['id'] for user in users if user['username'] == 'alice')
        image = (await api(admin, f'/queue?dataset_id={dataset}&queue=unallocated'))['images'][0]
        def captured(images):
            return {'image_ids': [row['id'] for row in images], 'expected': {row['id']: {'expected_version': row['current_version'], 'generation': row['generation']} for row in images}}
        created = await api(admin, f'/datasets/{dataset}/tranches', {'name': 'Second review tranche', 'operation_id': str(uuid.uuid4()), **captured([image])})
        await api(admin, f"/tranches/{created['tranche']['id']}/activate", {'operation_id': str(uuid.uuid4()), 'expected_revision': created['tranche']['revision'], 'annotator_id': alice_id, **captured(created['images'])})
        await sign_in(alice, origin, 'alice')
        tranches = (await api(alice, f'/tranches?dataset_id={dataset}'))['tranches']
        assert len(tranches) == 2
        for tranche in tranches:
            image = (await api(alice, f"/queue?dataset_id={dataset}&tranche_id={tranche['id']}&queue=work"))['images'][0]
            editor = str(uuid.uuid4())
            current = await api(alice, f"/images/{image['id']}/editor", {'editor_id': editor, 'expected_version': image['current_version'], 'generation': image['generation']})
            def context(row):
                return {'editor_id': editor, 'expected_version': row['current_version'], 'generation': row['generation'], 'operation_id': str(uuid.uuid4())}
            saved = await api(alice, f"/images/{image['id']}/save", {**context(current), 'annotations': {'objects': []}, 'complete': True, 'negative': True})
            await api(alice, f"/images/{image['id']}/submit", context(saved))
        reviewed = (await api(admin, f'/queue?dataset_id={dataset}&queue=review'))['images']
        first, second = reviewed
        assert first['tranche_id'] != second['tranche_id']
        await admin.goto(f"{origin}/annotate?image={first['id']}&queue=review")
        await expect(admin.locator('#hostedSave')).to_be_enabled()
        save_route = f"**/api/hosted/images/{first['id']}/save"
        await admin.route(save_route, lambda route: route.abort('failed'))
        await admin.locator('#hostedNegative').uncheck()
        await draw(admin, .15, .2, .45, .5)
        retained = await admin.evaluate('window.TatorHostedCanvas.snapshot()')
        await admin.locator('#hostedSave').click()
        await admin.wait_for_function("Object.keys(sessionStorage).some(k => k.startsWith('tator.hosted.draft:') && JSON.parse(sessionStorage[k]).pending)")
        await admin.unroute(save_route)
        await admin.reload()
        await expect(admin.locator('#hostedResolve')).to_be_visible()
        await expect(admin.locator('#hostedScopeNotice')).to_have_count(0)
        admin.on('dialog', lambda dialog: dialog.accept())
        await admin.locator('#hostedResolve').click()
        await expect(admin.locator('#hostedResolve')).to_have_count(0)
        await expect(admin.locator('#hostedSaveState')).to_have_text('All changes saved')
        assert await admin.evaluate('window.TatorHostedCanvas.snapshot()') == retained
        query = parse_qs(urlsplit(admin.url).query)
        assert query.get('queue') == ['review'] and 'tranche' not in query
        await admin.locator('#hostedApprove').click()
        await admin.wait_for_function('id => window.TatorHosted?.imageId === id && window.TatorHostedCanvas.ready()', arg=second['id'])
        query = parse_qs(urlsplit(admin.url).query)
        assert query.get('queue') == ['review'] and 'tranche' not in query
        await admin.locator('#hostedBack').click()
        await expect(admin.locator('#workspace')).to_be_visible()
        query = parse_qs(urlsplit(admin.url).query)
        assert query.get('view') == ['submitted'] and 'tranche' not in query
        await browser.close()
