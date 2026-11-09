/* An image's history for administrators: who changed it, when and how. Read-only. */
(() => {
    "use strict";
    const canvas = () => window.TatorHosted, workspace = window.TatorHostedDashboard;
    if (!workspace && !window.TatorHostedBoot) return;
    const role = () => window.TatorHostedBoot ? window.TatorHostedBoot.user?.role : workspace?.session?.user.role;
    const request = path => window.TatorHostedBoot ? canvas().api(`/api/hosted${path}`) : workspace.api(path);
    const node = (tag, text, className) => { const element = document.createElement(tag); if (text !== undefined) element.textContent = text; if (className) element.className = className; return element; };
    const plural = (n, word, many = `${word}es`) => `${n} ${n === 1 ? word : many}`;
    let dialog, title, status, list, ticket = 0;
    function build() {
        dialog = node("dialog"); dialog.id = "imageHistoryDialog"; dialog.style.width = "min(720px, calc(100% - 32px))";
        // The canvas page has no dialog styling of its own; the workspace's hosted.css refines this.
        if (window.TatorHostedBoot) dialog.style.cssText += ";padding:24px 28px;border-radius:12px;border:1px solid #8888;max-height:85vh";
        title = node("h2", "History"); status = node("p"); status.setAttribute("role", "status");
        list = node("ol", undefined, "hosted-history"); list.style.cssText = "list-style:none;padding:0;margin:0;max-height:60vh;overflow:auto";
        const close = node("button", "Close"); close.type = "button"; close.onclick = () => dialog.close();
        const actions = node("div", undefined, "dialog-actions"); actions.append(close);
        dialog.append(title, status, list, actions); document.body.append(dialog);
        dialog.addEventListener("close", () => { ++ticket; });
        // The canvas page claims Escape for drawing; while this dialog is open, Escape closes it.
        document.addEventListener("keydown", event => { if (event.key === "Escape" && dialog.open) { event.preventDefault(); event.stopPropagation(); dialog.close(); } }, true);
    }
    function describe(event, versions, reviews, who) {
        const data = event.data || {}, version = versions.get(data.version_id), reason = data.reason ? `: ${data.reason}` : "";
        const content = item => { if (!item) return ""; const boxes = (item.annotations?.objects || []).length; return item.negative ? "no target objects" : `${boxes ? plural(boxes, "box") : "no labels"}${item.complete ? ", checked" : ""}`; };
        const feedback = text => text ? `: “${text}”` : "";
        switch (event.kind) {
            case "imported": return `Added to Incoming${version ? ` with ${content(version)}` : ""}${data.filename_collisions?.length ? ` · ${data.filename_collisions.length === 1 ? "another image has" : `${data.filename_collisions.length} other images have`} the same file name` : ""}`;
            case "duplicate_import": return `Added again as ${data.filename || "the same file"}; it was already here and kept as it was${data.conflict_id ? ", with the new labels kept for comparison" : ""}`;
            case "assigned": return `Assigned to ${who(data.annotator_id)}`;
            case "reassigned": return `Reassigned to ${who(data.annotator_id)}`;
            case "annotation_saved": return `Saved with ${content(version) || "changes"}`;
            case "submitted": return "Submitted for review";
            case "approved": return `Approved into the confirmed dataset${feedback(reviews.get(data.review_id)?.feedback)}`;
            case "returned": return `Returned for changes${feedback(data.feedback ?? reviews.get(data.review_id)?.feedback)}`;
            case "submission_withdrawn": return `Submission withdrawn to Incoming${reason}`;
            case "ownership_overridden": return `Recalled from ${data.previous_assignee ? who(data.previous_assignee) : "the annotator"}${reason}`;
            case "tranche_transferred": return "Moved to another tranche";
            case "image_archived": return `Archived${reason}`;
            case "image_restored": return "Restored from the archive";
            case "editor_taken_over": return "Editing moved to another tab";
            case "import_conflict_resolved": return `Imported labels decided: ${data.choice === "keep_current" ? "current labels kept" : "imported objects used"}`;
            case "split_changed": return `Split set to ${data.split || data.target_split || "a new value"}`;
            default: { const text = String(event.kind || "change").replaceAll("_", " "); return `${text.charAt(0).toUpperCase()}${text.slice(1)}`; }
        }
    }
    async function open(imageId, filename) {
        if (role() !== "admin" || !imageId) return;
        if (!dialog) build();
        const seq = ++ticket; title.textContent = `History · ${filename || "image"}`; list.replaceChildren(); status.textContent = "Loading the history…"; status.dataset.error = "false";
        if (!dialog.open) dialog.showModal();
        try {
            const [history, users] = await Promise.all([request(`/images/${encodeURIComponent(imageId)}/history`), request("/users").catch(() => ({users: []}))]);
            if (seq !== ticket || !dialog.open) return;
            const names = new Map(users.users.map(user => [user.id, user.display_name || user.username]));
            const who = id => names.get(id) || "a removed account";
            const versions = new Map(history.versions.map(item => [item.id, item])), reviews = new Map(history.reviews.map(item => [item.id, item]));
            // Saving the same draft repeatedly is one line per save; the editor's own tab handovers are noise.
            const events = history.events.filter(event => event.kind !== "editor_released").sort((a, b) => b.created_at - a.created_at);
            status.textContent = events.length ? `${plural(events.length, "change", "changes")}, newest first. Every saved version stays in the history.` : "No changes are recorded for this image.";
            for (const event of events) {
                const row = node("li"); row.style.cssText = "padding:.55rem 0;border-bottom:1px solid var(--border, #8884)";
                const when = node("small", `${new Date(event.created_at * 1000).toLocaleString()} · ${who(event.actor_id)}`); when.style.display = "block";
                row.append(when, node("span", describe(event, versions, reviews, who))); list.append(row);
            }
        } catch (error) { if (seq === ticket) { status.textContent = error.message; status.dataset.error = "true"; } }
    }
    window.TatorHostedHistory = {open};
    // On the canvas: a History button beside the image's other review tools.
    if (window.TatorHostedBoot?.user?.role === "admin") window.addEventListener("load", () => {
        const button = node("button", "History"); button.type = "button"; button.id = "hostedImageHistory";
        button.onclick = () => open(canvas()?.imageId, canvas()?.filename || document.querySelector(".hosted-canvas-panel strong")?.textContent);
        const attach = () => { const target = document.querySelector(".hosted-canvas-feedback"); if (!target) return false; target.append(button); return true; };
        if (!attach()) { const observer = new MutationObserver(() => { if (attach()) observer.disconnect(); }); observer.observe(document.body, {childList: true, subtree: true}); }
        window.addEventListener("tator:hosted-image", () => { if (dialog?.open) dialog.close(); });
    }, {once: true});
})();
