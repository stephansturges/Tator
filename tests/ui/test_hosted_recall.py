"""One actual browser/HTTP recall proof using the existing disposable fixture."""
import asyncio
import os
import uuid

import pytest

from test_hosted_workflow import hosted_origin, sign_in

pytestmark = pytest.mark.skipif(os.environ.get("TATOR_HOSTED_BROWSER_TESTS") != "1",
                                reason="Explicit isolated browser run required")


def test_whole_annotator_recall_conflict_and_lost_reply_recovery(hosted_origin, tmp_path):
    asyncio.run(_exercise(hosted_origin, tmp_path))


async def _exercise(origin, output):
    from playwright.async_api import async_playwright, expect
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        admin = await browser.new_page(viewport={"width": 1280, "height": 950})
        alice = await browser.new_page()
        errors = []
        admin.on("pageerror", lambda error: errors.append(str(error)))
        await sign_in(admin, origin, "admin")
        await sign_in(alice, origin, "alice")

        async def api(page, path, body=None):
            return await page.evaluate("([path,body]) => window.TatorHostedDashboard.api(path, body === null ? {} : {method:'POST',body})", [path, body])

        ds = await admin.locator("#datasetChoice").input_value()
        users = (await api(admin, "/users"))["users"]
        owners = {user["username"]: user["id"] for user in users}
        images = (await api(admin, f"/images?dataset_id={ds}"))["images"]
        owned = next(image for image in images if image["assignee_id"] == owners["alice"])
        incoming = [image for image in images if image["state"] == "unassigned"]

        async def assign(images, owner):
            return await api(admin, f"/datasets/{ds}/assignments", {
                "image_ids": [image["id"] for image in images], "annotator_id": owners[owner],
                "expected": {image["id"]: {"expected_version": image["current_version"], "generation": image["generation"]} for image in images},
                "operation_id": str(uuid.uuid4())})

        await assign([incoming[0]], "alice")
        await assign([incoming[1]], "bob")
        await admin.locator("#recallAssigned").click()
        await expect(admin.locator("#recallImages input")).to_have_count(3)
        await admin.locator("#recallAnnotator").select_option(owners["alice"])
        await admin.get_by_role("button", name="Select all in this list", exact=True).click()
        await expect(admin.locator("#recallImages input:checked")).to_have_count(2)
        await admin.locator("#recallReason").fill("Administrator will redistribute this work.")
        await admin.screenshot(path=str(output / "recall-selection.png"))

        leased = await api(alice, f"/images/{owned['id']}/editor", {"editor_id": "recall-browser-editor", "expected_version": owned["current_version"], "generation": owned["generation"]})
        saved = await api(alice, f"/images/{owned['id']}/save", {"expected_version": leased["current_version"], "generation": leased["generation"],
            "editor_id": "recall-browser-editor", "operation_id": str(uuid.uuid4()), "annotations": {"objects": [], "caption": "Draft preserved across recall"}})
        await admin.locator("#recallSubmit").click()
        await expect(admin.locator("#recallDialog [role=status]")).to_contain_text("No images were recalled")
        assert (await api(admin, f"/images/{incoming[0]['id']}"))["assignee_id"] == owners["alice"]
        await admin.get_by_role("button", name="Refresh list", exact=True).click()
        await expect(admin.locator("#recallImages input")).to_have_count(3)
        await admin.locator("#recallAnnotator").select_option(owners["alice"])
        await admin.get_by_role("button", name="Select all in this list", exact=True).click()
        calls = []

        async def lose_reply(route):
            calls.append(route.request.post_data_json)
            await route.fetch()
            await route.abort("failed")

        endpoint = f"**/api/hosted/datasets/{ds}/recall-images"
        await admin.route(endpoint, lose_reply)
        await admin.locator("#recallSubmit").click()
        await expect(admin.locator("#recallDialog [role=status]")).to_contain_text("original recall is retained")
        assert len(calls) == 2 and calls[0] == calls[1]
        recalled = [(await api(admin, f"/images/{identifier}")) for identifier in calls[0]["image_ids"]]
        assert all(image["state"] == "unassigned" for image in recalled)
        assert next(image for image in recalled if image["id"] == owned["id"])["current_version"] == saved["current_version"]
        new_assignment = await assign(recalled, "bob")
        await admin.unroute(endpoint, lose_reply)
        await admin.reload()
        await expect(admin.locator("#workspace")).to_be_visible()
        await admin.locator("#recallAssigned").click()
        await expect(admin.locator("#recallSubmit")).to_have_text("Check and retry saved recall")

        async def malformed_reply(route):
            await route.fetch()
            await route.fulfill(status=200, content_type="application/json", body="{}")

        await admin.route(endpoint, malformed_reply)
        await admin.locator("#recallSubmit").click()
        await expect(admin.locator("#recallDialog [role=status]")).to_contain_text("result is still unconfirmed")
        assert await admin.evaluate("Object.keys(sessionStorage).some(key => key.startsWith('tator.hosted.recall:'))")
        await admin.unroute(endpoint, malformed_reply)
        await admin.locator("#recallSubmit").click()
        await expect(admin.locator("#recallDialog")).not_to_be_visible()
        current = await api(admin, f"/images/{owned['id']}")
        assert current["assignment_id"] == new_assignment["assignment_id"]
        assert current["annotations"]["caption"] == "Draft preserved across recall"
        assert not await admin.evaluate("Object.keys(sessionStorage).some(key => key.startsWith('tator.hosted.recall:'))")
        assert not errors
        await browser.close()
