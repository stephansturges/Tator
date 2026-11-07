"""Actual upload composition and exact tranche-attachment recovery (TRV-02)."""
import asyncio
import os
import uuid

import pytest
from test_hosted_workflow import hosted_origin, sign_in
from test_hosted_tranches import api

pytestmark = pytest.mark.skipif(os.environ.get('TATOR_HOSTED_BROWSER_TESTS') != '1', reason='Explicit isolated browser run required')


@pytest.mark.parametrize('hosted_origin', ['analysis'], indirect=True)
def test_mixed_uploads_attach_only_unique_available_images_and_bound_before_upload(hosted_origin, tmp_path):
    asyncio.run(_mixed(hosted_origin, tmp_path))


async def _mixed(origin, output):
    from PIL import Image
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await sign_in(page, origin, 'admin')
        dataset = await page.locator('#datasetChoice').input_value()
        incoming = (await api(page, f'/queue?dataset_id={dataset}&queue=unallocated'))['images'][0]
        confirmed = (await api(page, f'/queue?dataset_id={dataset}&queue=confirmed'))['images'][0]
        tranche = (await api(page, f'/datasets/{dataset}/tranches', {'operation_id': str(uuid.uuid4()), 'name': 'Mixed uploads', 'image_ids': [incoming['id']], 'expected': {incoming['id']: {'expected_version': incoming['current_version'], 'generation': incoming['generation']}}}))['tranche']
        files = []
        for filename, image in [('confirmed.png', confirmed), ('reserved.png', incoming)]:
            data = await (await page.request.get(f'{origin}/api/hosted/images/{image["id"]}/media')).body()
            path = output / filename; path.write_bytes(data); files.append(path)
        fresh = output / 'new.png'; Image.new('RGB', (120, 80), (187, 40, 21)).save(fresh)
        copy = output / 'same-pixels.png'; copy.write_bytes(fresh.read_bytes())
        await page.goto(f'{origin}/?dataset={dataset}&view=tranches&tranche={tranche["id"]}')
        async with page.expect_file_chooser() as chooser:
            await page.get_by_role('button', name='Upload images here', exact=True).click()
        await (await chooser.value).set_files([fresh, *files, copy])
        await expect(page.locator('#notice')).to_contain_text('1 images attached to the tranche')
        await expect(page.locator('#notice')).to_contain_text('confirmed.png: already in the confirmed dataset')
        await expect(page.locator('#notice')).to_contain_text('reserved.png: already in “Mixed uploads”')
        await expect(page.locator('#notice')).to_contain_text('same-pixels.png: same image as new.png')
        assert (await api(page, f'/tranches/{tranche["id"]}'))['counts']['retained'] == 2
        assert not await page.evaluate("Object.keys(sessionStorage).some(k => k.startsWith('tator.hosted.import:') || k.startsWith('tator.hosted.tranche-command:'))")
        uploads = []
        page.on('request', lambda request: uploads.append(request.url) if request.method == 'POST' and request.url.endswith('/incoming') else None)
        async with page.expect_file_chooser() as chooser:
            await page.get_by_role('button', name='Upload images here', exact=True).click()
        await (await chooser.value).set_files([{'name': f'candidate-{i}.png', 'mimeType': 'image/png', 'buffer': fresh.read_bytes()} for i in range(501)])
        await expect(page.locator('#notice')).to_contain_text('No files were uploaded')
        assert not uploads
        assert not await page.evaluate("Object.keys(sessionStorage).some(k => k.startsWith('tator.hosted.import:'))")
        # A concurrent tranche edit rejects the exact attachment; its uploaded image
        # remains in Incoming, with both retry journals retired rather than looping.
        newer = output / 'later.png'; Image.new('RGB', (120, 80), (188, 42, 23)).save(newer)
        endpoint = f'**/api/hosted/tranches/{tranche["id"]}/members/add'
        async def change_before_attach(route):
            current = await api(page, f'/tranches/{tranche["id"]}')
            await api(page, f'/tranches/{tranche["id"]}/edit', {'operation_id': str(uuid.uuid4()), 'expected_revision': current['revision'], 'name': current['name'], 'instructions': 'Changed concurrently'})
            await route.continue_()
        await page.route(endpoint, change_before_attach)
        async with page.expect_file_chooser() as chooser:
            await page.get_by_role('button', name='Upload images here', exact=True).click()
        await (await chooser.value).set_files(newer)
        await expect(page.locator('#notice')).to_contain_text('attachment failed')
        assert not await page.evaluate("Object.keys(sessionStorage).some(k => k.startsWith('tator.hosted.import:') || k.startsWith('tator.hosted.tranche-command:'))")
        assert any(row['filename'] == 'later.png' for row in (await api(page, f'/queue?dataset_id={dataset}&queue=unallocated'))['images'])
        await browser.close()


@pytest.mark.parametrize('hosted_origin', ['tranche_queue'], indirect=True)
def test_picker_exact_matching_limit_search_range_and_annotator_list(hosted_origin):
    asyncio.run(_picker(hosted_origin))


async def _picker(origin):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await sign_in(page, origin, 'admin')
        dataset = await page.locator('#datasetChoice').input_value()
        tranche = (await api(page, f'/datasets/{dataset}/tranches', {'operation_id': str(uuid.uuid4()), 'name': 'Searchable draft'}))['tranche']
        await page.goto(f'{origin}/?dataset={dataset}&view=tranches&tranche={tranche["id"]}')
        await page.get_by_role('button', name='Add Incoming images', exact=True).click()
        dialog = page.locator('#trancheDialog')
        await expect(dialog.get_by_role('status')).to_contain_text('604 matching images')
        await dialog.get_by_role('button', name='Select all matching images', exact=True).click()
        await expect(dialog).to_contain_text('604 images match. Narrow the search')
        await expect(dialog.get_by_role('status')).to_contain_text('0 selected')
        await dialog.get_by_label('Find Incoming images').fill('frame-60')
        await dialog.get_by_role('button', name='Search images', exact=True).click()
        await expect(dialog.get_by_role('status')).to_contain_text('6 matching images')
        await dialog.locator('.tranche-picker input').nth(0).click()
        await dialog.locator('.tranche-picker input').nth(2).click(modifiers=['Shift'])
        await expect(dialog.get_by_role('status')).to_contain_text('3 selected')
        await dialog.get_by_role('button', name='Select all matching images', exact=True).click()
        await expect(dialog.get_by_role('status')).to_contain_text('6 selected')
        await dialog.get_by_role('button', name='Add selected images', exact=True).click()
        await expect(page.locator('#tranchePanel')).to_contain_text('6 currently retained')
        users = (await api(page, '/users'))['users']; alice = next(u['id'] for u in users if u['username'] == 'alice'); bob = next(u['id'] for u in users if u['username'] == 'bob')
        await page.get_by_role('button', name='Assign tranche', exact=True).click()
        await dialog.get_by_label('Annotator', exact=True).select_option(alice)
        await dialog.get_by_role('button', name='Assign unfinished images', exact=True).click()
        await expect(page.locator('#tranchePanel')).to_contain_text('Active · alice')
        await page.get_by_role('button', name='Change annotator', exact=True).click()
        await dialog.get_by_label('Annotator', exact=True).select_option(bob)
        await dialog.get_by_role('button', name='Assign unfinished images', exact=True).click()
        await expect(page.locator('#tranchePanel')).to_contain_text('Active · bob')
        other = (await api(page, f'/datasets/{dataset}/tranches', {'operation_id': str(uuid.uuid4()), 'name': 'Destination draft'}))['tranche']
        await page.locator('#tranchePanel .hosted-card input').first.check()
        await page.get_by_role('button', name='Move to another tranche', exact=True).click()
        await dialog.get_by_label('Destination tranche', exact=True).select_option(other['id'])
        await dialog.get_by_label('Reason', exact=True).fill('Separate follow-up')
        await dialog.get_by_role('button', name='Move images', exact=True).click()
        await expect(page.locator('#tranchePanel')).to_contain_text('1 transferred')
        await page.get_by_role('button', name='← All tranches', exact=True).click()
        await page.get_by_label('Filter by annotator').select_option(bob)
        await expect(page.locator('.tranche-card')).to_have_count(1)
        await expect(page.locator('.tranche-card')).to_contain_text('Last activity')
        await expect(page.locator('.tranche-card')).to_contain_text('Searchable draft')
        await browser.close()


def test_direct_continue_and_tranche_bulk_review_exact_retry(hosted_origin):
    asyncio.run(_review(hosted_origin))


async def _review(origin):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        admin = await browser.new_page(); alice_page = await browser.new_page()
        await sign_in(admin, origin, 'admin')
        dataset = await admin.locator('#datasetChoice').input_value()
        rows = (await api(admin, f'/queue?dataset_id={dataset}&queue=unallocated'))['images'][:2]
        alice = next(u['id'] for u in (await api(admin, '/users'))['users'] if u['username'] == 'alice')
        captured = lambda images: {'image_ids': [r['id'] for r in images], 'expected': {r['id']: {'expected_version': r['current_version'], 'generation': r['generation']} for r in images}}
        created = await api(admin, f'/datasets/{dataset}/tranches', {'operation_id': str(uuid.uuid4()), 'name': 'Review batch', **captured(rows)})
        tranche = created['tranche']
        await api(admin, f'/tranches/{tranche["id"]}/activate', {'operation_id': str(uuid.uuid4()), 'expected_revision': tranche['revision'], 'annotator_id': alice, **captured(created['images'])})
        await sign_in(alice_page, origin, 'alice')
        card = alice_page.locator('.tranche-card').filter(has_text='Review batch')
        await card.get_by_role('button', name='Continue annotation', exact=True).click()
        for index in range(2):
            await expect(alice_page.locator('#hostedSave')).to_be_enabled()
            first = await alice_page.evaluate('window.TatorHosted.imageId')
            await alice_page.locator('#hostedNegative').check()
            await alice_page.locator('#hostedSubmit').click()
            if index == 0: await alice_page.wait_for_function('id => window.TatorHosted?.imageId !== id', arg=first)
        await expect(alice_page.locator('#workspace')).to_be_visible()
        await admin.goto(f'{origin}/?dataset={dataset}&view=tranches&tranche={tranche["id"]}')
        await admin.locator('#trancheMemberFilter').select_option('awaiting_review')
        await expect(admin.locator('#tranchePanel .hosted-card')).to_have_count(2)
        await admin.get_by_role('button', name='Select eligible images on this page', exact=True).click()
        calls = []
        async def lose(route):
            calls.append(route.request.post_data_json); response = await route.fetch(); assert response.status == 200; await route.abort('failed')
        endpoint = f'**/api/hosted/datasets/{dataset}/approve-images'
        await admin.route(endpoint, lose)
        admin.on('dialog', lambda dialog: dialog.accept())
        await admin.get_by_role('button', name='Approve selected submissions', exact=True).click()
        await expect(admin.locator('#trancheRetry')).to_be_enabled()
        assert len(calls) == 2 and calls[0] == calls[1]
        await admin.unroute(endpoint, lose)
        await admin.reload(); await expect(admin.locator('#trancheRetry')).to_be_enabled()
        await admin.locator('#trancheRetry').click()
        await expect(admin.locator('#trancheRecovery')).to_be_hidden()
        await expect(admin.locator('#tranchePanel')).to_contain_text('2 approved')
        await admin.get_by_role('button', name='← All tranches', exact=True).click()
        await expect(admin.locator('.tranche-card').filter(has_text='Review batch')).to_contain_text('Complete')
        await browser.close()
