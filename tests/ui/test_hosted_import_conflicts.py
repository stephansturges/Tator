"""Annotated upload and explicit conflict choice through the actual hosted UI."""
import asyncio
import json
import os
import uuid

import pytest

from test_hosted_workflow import hosted_origin, sign_in

pytestmark = pytest.mark.skipif(os.environ.get('TATOR_HOSTED_BROWSER_TESTS') != '1', reason='Explicit isolated browser run required')


def test_yolo_sidecar_import_compare_and_apply_preserves_caption(hosted_origin, tmp_path):
    asyncio.run(_exercise(hosted_origin, tmp_path))


async def _exercise(origin, output):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []; page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('dialog', lambda dialog: dialog.accept())
        await sign_in(page, origin, 'admin')
        async def api(path, body=None):
            return await page.evaluate("([path,body]) => window.TatorHostedDashboard.api(path, body === null ? {} : {method:'POST',body})", [path, body])
        ds = await page.locator('#datasetChoice').input_value()
        dataset = await api('/datasets/' + ds)
        incoming = [row for row in (await api('/images?dataset_id=' + ds))['images'] if row['state'] == 'unassigned']
        image, other_image = incoming[:2]
        leased = await api(f"/images/{image['id']}/editor", {'editor_id': 'import-test-editor', 'expected_version': image['current_version'], 'generation': image['generation']})
        context = lambda row: {'expected_version': row['current_version'], 'generation': row['generation'], 'editor_id': 'import-test-editor', 'operation_id': str(uuid.uuid4())}
        saved = await api(f"/images/{image['id']}/save", {**context(leased), 'annotations': {'objects': [], 'caption': 'Preserved human caption', 'alternate_captions': [{'caption': 'Preserved earlier caption'}]}})
        await api(f"/images/{image['id']}/release", context(saved))
        photo = output / image['filename']; photo.write_bytes(await (await page.request.get(f"{origin}/api/hosted/images/{image['id']}/media")).body())
        labels = photo.with_suffix('.txt'); labels.write_text('0 0.5 0.5 0.2 0.25\n')
        labelmap = output / 'labelmap.txt'; labelmap.write_text('Person\nCar\n')
        await page.get_by_role('button', name='Incoming', exact=True).click()
        await page.locator('#uploadInput').set_input_files([str(photo), str(labels), str(labelmap)])
        await expect(page.locator('#notice')).to_contain_text('1 imported label conflicts retained')
        before = await api(f"/images/{image['id']}")
        assert before['annotations']['objects'] == []
        await page.goto(f"{origin}/annotate?image={image['id']}")
        try:
            await page.wait_for_function('window.TatorHosted?.editable && window.TatorHostedCanvas?.ready()')
        except Exception:
            await page.screenshot(path=str(output / 'canvas-readiness-failure.png'))
            (output / 'canvas-readiness-failure.json').write_text(json.dumps({
                'errors': errors, 'body': await page.locator('body').inner_text(),
                'state': await page.evaluate('({editable:window.TatorHosted?.editable,ready:window.TatorHostedCanvas?.ready()})'),
            }, indent=2))
            raise
        assert await page.evaluate('window.TatorHosted.save({manual:true})')
        # Hold an actual committed save reply. The comparison sees its new server
        # version, while applying it must wait for the canvas save acknowledgement.
        # Closing the modal during that wait must not allow another image to take
        # ownership of the proposal's eventual reply or draft journal.
        save_committed, release_save = asyncio.Event(), asyncio.Event()
        async def delay_save_reply(route):
            response = await route.fetch()
            save_committed.set()
            await release_save.wait()
            await route.fulfill(response=response)
        await page.route(f"**/api/hosted/images/{image['id']}/save", delay_save_reply)
        await page.locator('#hostedComplete').check()
        await page.evaluate('() => { window.importRaceSave = window.TatorHosted.save({manual:true}); }')
        await asyncio.wait_for(save_committed.wait(), 10)
        await page.locator('#hostedImportConflicts').click()
        await expect(page.locator('#importConflictsDialog canvas')).to_have_count(2)
        await expect(page.locator('#importConflictsDialog')).to_contain_text('1 Person')
        await page.screenshot(path=str(output / 'import-conflict-comparison.png'))
        try:
            await page.get_by_role('button', name='Use imported objects', exact=True).click()
            await page.locator('#importConflictsDialog').get_by_role('button', name='Close', exact=True).click()
            await page.evaluate("""id => {
                window.importRaceNavigation = {state: 'pending'};
                window.TatorHosted.navigate(id).then(
                    () => { window.importRaceNavigation = {state: 'navigated'}; },
                    error => { window.importRaceNavigation = {state: 'blocked', message: error.message}; });
            }""", other_image['id'])
            await page.wait_for_function("window.importRaceNavigation.state !== 'pending'", timeout=2000)
            navigation = await page.evaluate('window.importRaceNavigation')
            assert navigation['state'] == 'blocked', navigation
            assert 'Wait for the current save or handoff' in navigation['message']
            assert await page.evaluate('window.TatorHosted.imageId') == image['id']
            await expect(page.locator('#hostedSave')).to_be_disabled()
        finally:
            release_save.set()
        await page.evaluate('window.importRaceSave')
        await expect(page.locator('#hostedSaveState')).to_contain_text('Import comparison resolved')
        assert await page.evaluate('window.TatorHosted.imageId') == image['id']
        await page.locator('#hostedImportConflicts').click()
        await expect(page.locator('#importConflictsDialog [role=status]')).to_contain_text('imported objects used')
        final = await page.evaluate("id => window.TatorHosted.api('/api/hosted/images/' + id)", image['id'])
        assert final['annotations']['caption'] == 'Preserved human caption'
        assert final['annotations']['alternate_captions'][0]['caption'] == 'Preserved earlier caption'
        assert final['annotations']['objects'][0]['class_id'] == next(row['id'] for row in dataset['classes'] if row['name'] == 'Person')
        history = await page.evaluate("id => window.TatorHosted.api('/api/hosted/images/' + id + '/history')", image['id'])
        assert len(history['import_conflicts']) == len(history['import_conflict_decisions']) == 1
        assert history['import_conflicts'][0]['annotations']['import_provenance']['label_text'] == labels.read_text()
        assert not errors
        await browser.close()
