(() => {
    "use strict";
    const dashboard = window.TatorHostedDashboard;
    if (!dashboard) return;
    const node = (tag, text) => { const result = document.createElement(tag); if (text !== undefined) result.textContent = text; return result; };
    const uid = () => crypto.randomUUID();
    const button = node("button", "Export & recovery"); button.type = "button"; button.id = "operations"; button.hidden = true;
    document.querySelector(".hosted-actions").append(button);
    const dialog = node("dialog"); dialog.id = "operationsDialog";
    const heading = node("h2", "Export and recovery");
    const notice = node("p"); notice.className = "dialog-notice"; notice.dataset.dialogNotice = ""; notice.setAttribute("role", "status"); notice.hidden = true;
    const datasetPanel = node("section"), backups = node("section"), archived = node("section"), downloads = node("div"), cleanup = node("section");
    const close = node("button", "Close"); close.type = "button"; close.onclick = () => dialog.close();
    let generation = 0, sessionKey = "";
    const current = (ticket) => ticket === generation && dialog.open && dashboard.session?.user.role === "admin";
    dialog.append(heading, notice, datasetPanel, backups, downloads, archived, cleanup, close); document.body.append(dialog);
    function action(parent, title, fn) { const ticket = generation; const control = node("button", title); control.type = "button"; control.onclick = () => { if (current(ticket)) return dashboard.action(control, fn); }; parent.append(control); return control; }
    function link(result, name) { const a = node("a", name); a.href = result.download_url; a.download = ""; a.className = "hosted-download"; downloads.append(a); }
    function storageCleanup(ticket) {
        cleanup.replaceChildren(node("h3", "Unused storage"), node("p", "Find files that no image, annotation history, saved analysis or registered model uses. Files must remain unreferenced for at least 30 days before deletion. Archived dataset files remain protected."));
        const key = `tator.hosted.purge:${dashboard.session.instance_id}:${dashboard.session.user.user_id}`;
        const resultArea = node("div"); cleanup.append(resultArea);
        async function complete(body) {
            const result = await dashboard.api('/storage/orphans/purge', {method: 'POST', body});
            if (!current(ticket)) return;
            const expected = [...body.digests].sort(), actual = [...(result.deleted_digests || []), ...(result.already_absent_digests || [])].sort();
            if (result.status !== "complete" || JSON.stringify(expected) !== JSON.stringify(actual) || !Number.isFinite(result.logical_bytes_removed)) throw new Error("The cleanup reply was incomplete. Its command remains saved; check its result before starting another cleanup.");
            sessionStorage.removeItem(key);
            storageCleanup(ticket); dashboard.notice(`Unused files removed: ${result.deleted_digests.length}. Logical size: ${(result.logical_bytes_removed / 1048576).toFixed(1)} MiB.`);
        }
        const retained = sessionStorage.getItem(key);
        if (retained) {
            cleanup.append(node("p", "A cleanup command is awaiting confirmation. Checking it repeats the same command and resumes only its original file list."));
            action(cleanup, "Check cleanup result", () => complete(JSON.parse(retained))); return;
        }
        async function scan(after = "") {
            const result = await dashboard.api('/storage/orphans/scan', {method: 'POST', body: {after, limit: 500}});
            if (!current(ticket)) return;
            const eligible = result.artifacts.filter(item => item.eligible && !item.referenced);
            resultArea.replaceChildren(node("p", `Checked ${result.scanned} stored files on this page. ${result.artifacts.length} are unused; ${eligible.length} have passed the 30-day recovery period.`));
            for (const item of result.artifacts) resultArea.append(node("small", `${item.sha256.slice(0, 16)}… · ${(item.size_bytes / 1048576).toFixed(2)} MiB · ${item.eligible ? "Eligible for cleanup" : "Protected until " + new Date(item.eligible_at * 1000).toLocaleDateString()}`));
            if (eligible.length) action(resultArea, "Preview eligible cleanup", async () => {
                const digests = eligible.map(item => item.sha256);
                const preview = await dashboard.api('/storage/orphans/preview', {method: 'POST', body: {digests}});
                if (!current(ticket)) return;
                if (!preview.can_purge) throw new Error("Storage references changed. Scan again before cleaning up.");
                const size = preview.artifacts.reduce((sum, item) => sum + item.size_bytes, 0);
                if (!confirm(`Permanently delete these ${digests.length} unused files (${(size / 1048576).toFixed(1)} MiB)? The server will check every reference again before deletion.`)) return;
                const body = {digests, preview_hash: preview.preview_hash, operation_id: uid()};
                sessionStorage.setItem(key, JSON.stringify(body)); storageCleanup(ticket); await complete(body);
            });
            if (result.next_cursor) action(resultArea, "Check next storage page", () => scan(result.next_cursor));
        }
        action(cleanup, "Inspect unused storage", () => scan());
    }
    async function archives() {
        const ticket = generation;
        const result = await dashboard.api("/archives");
        if (!current(ticket)) return;
        archived.replaceChildren(node("h3", "Archived images and datasets"));
        if (!result.archives.length) { archived.append(node("p", "Nothing is archived.")); return; }
        for (const entry of result.archives) {
            const row = node("div"); row.className = "hosted-user";
            row.append(node("span", `${entry.kind === "dataset" ? "Dataset" : "Image"} · ${entry.target_id} · ${new Date(entry.archived_at * 1000).toLocaleDateString()}${entry.reason ? ` · ${entry.reason}` : ""}`));
            if (entry.state === "archived") {
                action(row, "Restore", async () => { await dashboard.api(`/archives/${entry.archive_id}/restore`, { method: "POST", body: { operation_id: uid() } }); await archives(); await dashboard.refresh(); if (current(ticket)) dashboard.notice("Restored. Annotations and history were preserved."); });
                action(row, "Storage details", async () => {
                    const preview = await dashboard.api(`/archives/${entry.archive_id}/purge-preview`);
                    if (!current(ticket)) return;
                    const detail = node("p", `Files stay available for annotation history and saved dataset snapshots. The minimum recovery period ends ${new Date(preview.eligible_at * 1000).toLocaleDateString()}; referenced files remain protected afterward.`);
                    row.querySelector("p")?.remove(); row.append(detail);
                });
            }
            else row.append(node("small", "Restored"));
            archived.append(row);
        }
    }
    async function open() {
        const ticket = ++generation;
        datasetPanel.replaceChildren(); backups.replaceChildren(); downloads.replaceChildren(); notice.hidden = true;
        dialog.showModal();
        storageCleanup(ticket);
        const datasetId = dashboard.datasetId;
        if (datasetId) {
            const dataset = await dashboard.api(`/datasets/${datasetId}`);
            if (!current(ticket) || datasetId !== dashboard.datasetId) return;
            datasetPanel.append(node("h3", dataset.name), node("p", "A dataset export contains confirmed images, labels and captions. Incoming work is excluded."));
            action(datasetPanel, "Export confirmed dataset", async () => {
                dashboard.notice("Preparing the confirmed dataset export…");
                const result = await dashboard.api(`/datasets/${datasetId}/export`, { method: "POST", body: { operation_id: uid() } });
                if (!current(ticket)) return;
                link(result, `Download dataset · ${result.image_count} images`); dashboard.notice("Export ready. Your dataset has not changed.");
            });
            action(datasetPanel, "Archive this dataset…", async () => {
                const current = await dashboard.api(`/datasets/${datasetId}`), images = [];
                for (let offset = 0; ; offset += 500) { const result = await dashboard.api(`/images?dataset_id=${datasetId}&limit=500&offset=${offset}`); images.push(...result.images); if (result.images.length < 500) break; }
                if (images.some((image) => ["assigned", "returned"].includes(image.state))) throw new Error("This dataset still has assigned work. Recover those assignments before archiving it.");
                if (images.some((image) => image.editor_lease)) throw new Error("Close the editing tabs for this dataset before archiving it.");
                if (!confirm(`Archive “${current.name}” and its ${images.length} images? It will leave active views. Original files and annotation history stay preserved and can be restored.`)) return;
                const expected_images = Object.fromEntries(images.map((image) => [image.id, { expected_version: image.current_version, generation: image.generation }]));
                await dashboard.api(`/datasets/${datasetId}/archive`, { method: "POST", body: { expected_images, expected_revision: current.revision, operation_id: uid(), reason: "Archived by an administrator from the dataset workspace." } });
                dialog.close(); await dashboard.refresh(); dashboard.notice("Dataset archived. You can restore it from Export & recovery.");
            });
        }
        backups.append(node("h3", "Full server recovery backup"), node("p", "This encrypted backup includes accounts, assignments, drafts, confirmed data and registered model files. Keep the private decryption key separately on your computer. A dataset export cannot restore the server."));
        const label = node("label", "Public age encryption recipient"), recipient = node("input"); recipient.placeholder = "age1…"; recipient.autocomplete = "off"; label.append(recipient); backups.append(label);
        action(backups, "Create encrypted recovery backup", async () => {
            if (!/^age1[0-9a-z]{58}$/.test(recipient.value.trim())) throw new Error("Enter the public age recipient from your recovery key. Do not enter its private key.");
            dashboard.notice("Creating an encrypted backup. Annotation work can continue.");
            const result = await dashboard.api("/recovery-backups", { method: "POST", body: { operation_id: uid(), recipient: recipient.value.trim() } });
            if (!current(ticket)) return;
            link(result, "Download encrypted server backup"); dashboard.notice("Backup ready. Download it and keep your private recovery key separately.");
        });
        await archives();
    }
    button.onclick = () => dashboard.action(button, open);
    dialog.addEventListener("close", () => { ++generation; });
    function sessionChanged() {
        const next = JSON.stringify([dashboard.session?.instance_id, dashboard.session?.user.user_id, dashboard.session?.csrf]);
        if (next !== sessionKey) { sessionKey = next; ++generation; dialog.close(); datasetPanel.replaceChildren(); backups.replaceChildren(); downloads.replaceChildren(); archived.replaceChildren(); cleanup.replaceChildren(); }
        button.hidden = dashboard.session?.user.role !== "admin";
        if (button.hidden) { dialog.close(); datasetPanel.replaceChildren(); backups.replaceChildren(); downloads.replaceChildren(); archived.replaceChildren(); }
    }
    window.addEventListener("tator:hosted-session", sessionChanged); sessionChanged();
})();
