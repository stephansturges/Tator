/* Explicit split assignment for confirmed images, with capture groups intact. */
(() => {
    "use strict";
    const app = window.TatorHostedDashboard;
    if (!app) return;
    const node = (tag, text) => { const value = document.createElement(tag); if (text !== undefined) value.textContent = text; return value; };
    const button = node("button", "Set training split"); button.id = "hostedSplits"; button.hidden = true;
    document.getElementById("selectionBar").append(button);
    const recover = node("button", "Check pending split change"); recover.hidden = true; recover.id = "hostedSplitRecovery";
    document.querySelector(".hosted-actions").append(recover);
    const dialog = node("dialog"), notice = node("p"), content = node("section"), close = node("button", "Close");
    dialog.id = "hostedSplitsDialog"; notice.setAttribute("role", "status"); close.onclick = () => dialog.close();
    dialog.append(node("h2", "Training splits"), node("p", "Training images teach the model; validation and test images measure it. Related capture or video frames stay together. This changes split membership only; saved annotations and earlier snapshots remain intact."), notice, content, close);
    document.body.append(dialog);
    let epoch = 0, datasetId = "", key = "", sessionKey = "", recoveryMode = false;
    const valid = ticket => ticket === epoch && dialog.open && (recoveryMode || datasetId === app.datasetId) && app.session?.user.role === "admin";
    function retainedKeys() {
        if (app.session?.user.role !== "admin") return [];
        const prefix = `tator.hosted.splits:${app.session.instance_id}:${app.session.user.user_id}:`;
        return Object.keys(sessionStorage).filter(value => value.startsWith(prefix)).sort();
    }
    function action(parent, text, callback) {
        const ticket = epoch, control = node("button", text); control.type = "button";
        control.onclick = async () => { if (!valid(ticket)) return; control.disabled = true; try { await callback(); } catch (e) { if (valid(ticket) && !e.stale) notice.textContent = e.message; } finally { control.disabled = false; } };
        parent.append(control); return control;
    }
    async function commit(body, ticket) {
        let result;
        try { result = await app.api(`/datasets/${datasetId}/splits`, {method: "POST", body}); }
        catch (error) {
            if (valid(ticket) && [400, 409, 422].includes(error.status)) {
                // This atomic command was rejected. Unknown responses and auth
                // failures retain the exact command for later reconciliation.
                content.replaceChildren(node("p", "The server rejected this split change. No split was changed by this command. Refresh the preview after resolving the reported conflict."));
                action(content, "Discard rejected preview", () => { sessionStorage.removeItem(key); selection(); dialog.close(); });
            }
            throw error;
        }
        if (!valid(ticket)) return;
        const sorted = value => Object.entries(value || {}).sort(([a], [b]) => a.localeCompare(b));
        if (result.dataset_id !== datasetId || !Number.isInteger(result.revision) || !Number.isInteger(result.changed_count) || JSON.stringify(sorted(result.assignments)) !== JSON.stringify(sorted(body.assignments))) throw new Error("The split reply was incomplete. The original command is retained; check its result before making another change.");
        sessionStorage.removeItem(key); selection(); dialog.close(); await app.refresh(); app.notice(`Training split updated for ${result.changed_count} images. Earlier snapshots keep their original splits.`);
    }
    function pending(body, ticket) {
        content.replaceChildren(node("p", "A split change is awaiting confirmation. Check the saved command to recover its result; this does not create a new assignment."));
        action(content, "Check split change", () => commit(body, ticket));
    }
    async function open() {
        const ticket = ++epoch; datasetId = app.datasetId; recoveryMode = false;
        key = `tator.hosted.splits:${app.session.instance_id}:${app.session.user.user_id}:${datasetId}`;
        notice.textContent = ""; content.replaceChildren(); dialog.showModal();
        const retained = sessionStorage.getItem(key);
        if (retained) { pending(JSON.parse(retained), ticket); return; }
        const selected = app.selectedImages.filter(row => row.dataset_id === datasetId && row.state === "confirmed");
        const label = node("label", "Place selected images in"), target = node("select"); target.setAttribute("aria-label", "Place selected images in");
        for (const [value, name] of [["train", "Training"], ["val", "Validation"], ["test", "Test"], ["unsplit", "Not assigned to a split"]]) target.add(new Option(name, value));
        label.append(target); content.append(label);
        const previewArea = node("section");
        target.onchange = () => previewArea.replaceChildren();
        action(content, "Preview split change", async () => {
            const choice = target.value;
            const preview = await app.api(`/datasets/${datasetId}/splits/preview`, {method: "POST", body: {image_ids: selected.map(row => row.id), target_split: choice}});
            if (!valid(ticket) || target.value !== choice) return;
            previewArea.replaceChildren(node("p", `${preview.selected_count} selected images; ${preview.group_added_count} related images included to keep capture groups together. ${preview.change_count} images would change split.`));
            const counts = node("p"); counts.textContent = ["train", "val", "test", "unsplit"].map(split => `${split === "unsplit" ? "Unassigned" : split === "val" ? "Validation" : split === "train" ? "Training" : "Test"}: ${preview.counts_before[split] || 0} → ${preview.counts_after[split] || 0}`).join(" · "); previewArea.append(counts);
            const details = node("details"), list = node("ul"); details.append(node("summary", "Images in this change"), list);
            for (const image of preview.images) list.append(node("li", `${image.filename}: ${image.split} → ${preview.assignments[image.id]}`));
            previewArea.append(details);
            if (!preview.change_count) return;
            action(previewArea, "Apply this split change", async () => {
                const body = {assignments: preview.assignments, expected: preview.expected, expected_revision: preview.expected_revision, operation_id: crypto.randomUUID()};
                sessionStorage.setItem(key, JSON.stringify(body)); selection(); pending(body, ticket); await commit(body, ticket);
            });
        });
        content.append(previewArea);
    }
    button.onclick = () => { const ticket = epoch + 1; open().catch(e => { if (valid(ticket)) notice.textContent = e.message; }); };
    const selection = () => { button.hidden = app.session?.user.role !== "admin" || !app.selectedImages.some(row => row.state === "confirmed"); recover.hidden = retainedKeys().length === 0; };
    recover.onclick = () => {
        const retained = retainedKeys()[0]; if (!retained) return;
        const ticket = ++epoch; key = retained; datasetId = key.split(":").at(-1); recoveryMode = true;
        notice.textContent = ""; dialog.showModal(); pending(JSON.parse(sessionStorage.getItem(key)), ticket);
    };
    window.addEventListener("tator:hosted-selection", selection);
    window.addEventListener("tator:hosted-session", () => {
        const next = JSON.stringify([app.session?.instance_id, app.session?.user.user_id, app.session?.csrf]);
        if (sessionKey !== next) { sessionKey = next; ++epoch; dialog.close(); content.replaceChildren(); }
        selection();
    });
    dialog.addEventListener("close", () => { ++epoch; });
})();
