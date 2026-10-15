/* Explicit, atomic recovery of assigned work. Saved annotations never change. */
(() => {
    "use strict";
    const app = window.TatorHostedDashboard;
    if (!app) return;
    const node = (tag, text) => { const element = document.createElement(tag); if (text !== undefined) element.textContent = text; return element; };
    const button = node("button", "Recall assigned work"); button.id = "recallAssigned"; button.type = "button"; button.hidden = true;
    document.querySelector(".hosted-actions").append(button);
    const dialog = node("dialog"), heading = node("h2", "Recall assigned work"); dialog.id = "recallDialog";
    const explanation = node("p", "Return selected images to unassigned Incoming. Saved drafts and feedback stay intact. The annotators' editing tabs and pending tool results lose permission to change them. Submitted and confirmed images are excluded.");
    const notice = node("p"); notice.className = "dialog-notice"; notice.setAttribute("role", "status"); notice.hidden = true;
    const filterLabel = node("label", "Annotator"), filter = node("select"); filter.id = "recallAnnotator"; filterLabel.append(filter);
    const selectAll = node("button", "Select all in this list"), clear = node("button", "Clear selection"), refresh = node("button", "Refresh list");
    const controls = node("div"); controls.className = "dialog-actions"; controls.append(selectAll, clear, refresh);
    const count = node("p"), list = node("div"), more = node("button", "Show more"); list.id = "recallImages";
    list.style.maxHeight = "40vh"; list.style.overflowY = "auto";
    const reasonLabel = node("label", "Why are you recalling this work?"), reason = node("textarea"); reason.id = "recallReason"; reason.rows = 2; reason.maxLength = 10000; reason.required = true; reasonLabel.append(reason);
    const submit = node("button", "Recall selected images"), close = node("button", "Close"); submit.id = "recallSubmit"; submit.className = "primary";
    const actions = node("div"); actions.className = "dialog-actions"; actions.append(close, submit);
    for (const control of [selectAll, clear, refresh, more, submit, close]) control.type = "button";
    dialog.append(heading, explanation, notice, filterLabel, controls, count, list, more, reasonLabel, actions); document.body.append(dialog);
    let sequence = 0, sessionKey = "", datasetId = "", rows = [], selected = new Set(), pending = null, busy = false, visible = 100, storageKey = "";
    const identity = () => JSON.stringify([app.session?.instance_id, app.session?.user.user_id, app.session?.csrf]);
    const current = ticket => ticket === sequence && dialog.open && app.session?.user.role === "admin" && identity() === sessionKey && app.datasetId === datasetId;
    const tell = (text, error = false) => { notice.textContent = text; notice.hidden = !text; notice.dataset.error = String(error); };
    const filtered = () => rows.filter(row => !filter.value || row.assignee_id === filter.value);
    function update() {
        const frozen = busy || !!pending;
        for (const control of [filter, selectAll, clear, refresh, reason]) control.disabled = frozen;
        for (const check of list.querySelectorAll("input")) check.disabled = frozen;
        submit.disabled = busy || (!pending && (!selected.size || !reason.value.trim()));
        submit.textContent = pending ? "Check and retry saved recall" : `Recall ${selected.size || "selected"} image${selected.size === 1 ? "" : "s"}`;
        count.textContent = pending ? `${pending.body.image_ids.length} images in the saved recall. Its original selection and reason will be reused.` : `${selected.size} selected · ${filtered().length} images in this list`;
    }
    function render() {
        list.replaceChildren();
        const matching = filtered();
        for (const image of matching.slice(0, visible)) {
            const label = node("label"), check = node("input"); check.type = "checkbox"; check.checked = selected.has(image.id);
            check.setAttribute("aria-label", `Recall ${image.filename}`);
            check.onchange = () => { if (check.checked) selected.add(image.id); else selected.delete(image.id); update(); };
            const title = node("span", `${image.filename} · ${image.state === "returned" ? "Changes requested" : "Assigned"}${image.editor_lease ? " · Editing tab open" : ""}`);
            label.style.display = "flex"; label.style.flexDirection = "row"; label.style.alignItems = "center"; label.style.gap = "0.75rem";
            check.style.width = "auto"; check.style.flex = "0 0 auto"; label.append(check, title); list.append(label);
        }
        more.hidden = matching.length <= visible || !!pending;
        if (!matching.length && !pending) list.append(node("p", "No assigned work in this list."));
        update();
    }
    async function load(ticket) {
        busy = true; update(); tell("Loading assigned work…");
        try {
            const users = await app.api("/users");
            if (!current(ticket)) return;
            const loaded = [];
            for (let offset = 0; ; offset += 500) {
                const result = await app.api(`/images?${new URLSearchParams({ dataset_id: datasetId, state: "incoming", limit: "500", offset: String(offset) })}`);
                if (!current(ticket)) return;
                loaded.push(...result.images.filter(image => ["assigned", "returned"].includes(image.state)));
                if (result.images.length < 500) break;
            }
            rows = loaded; selected.clear(); visible = 100;
            filter.replaceChildren(); const all = node("option", "All annotators"); all.value = ""; filter.append(all);
            const owners = [...new Set(rows.map(image => image.assignee_id))];
            for (const owner of owners) {
                const user = users.users.find(user => user.id === owner), amount = rows.filter(image => image.assignee_id === owner).length;
                const option = node("option", `${user?.username || owner}${user?.disabled ? " (disabled)" : ""} · ${amount} images`); option.value = owner; filter.append(option);
            }
            tell(""); render();
        } finally { if (current(ticket)) { busy = false; update(); } }
    }
    async function open() {
        if (app.session?.user.role !== "admin" || !app.datasetId) return;
        datasetId = app.datasetId; sessionKey = identity(); const ticket = ++sequence;
        storageKey = `tator.hosted.recall:${app.session.instance_id}:${app.session.user.user_id}:${datasetId}`;
        rows = []; selected.clear(); reason.value = ""; pending = null; busy = false; filter.replaceChildren(); list.replaceChildren();
        heading.textContent = `Recall assigned work · ${document.getElementById("datasetChoice").selectedOptions[0]?.textContent || "Dataset"}`;
        dialog.showModal(); tell("");
        try { pending = JSON.parse(sessionStorage.getItem(storageKey) || "null"); }
        catch (_) { tell("The saved recall could not be read. Check this account's recent changes before starting another recall.", true); busy = true; update(); return; }
        if (pending) {
            reason.value = pending.body.reason; render();
            tell("A previous recall has no confirmed reply in this tab. Check it first; retrying uses the original command and cannot recall newly assigned work.");
        } else { update(); await load(ticket); }
    }
    async function commit() {
        const ticket = sequence;
        if (!current(ticket) || busy) return;
        if (!pending) {
            const chosen = rows.filter(row => selected.has(row.id));
            if (!chosen.length || !reason.value.trim()) return;
            if (chosen.length > 50000) throw new Error("Select no more than 50,000 images in one recall.");
            const body = { image_ids: chosen.map(row => row.id), expected: Object.fromEntries(chosen.map(row => [row.id, { expected_version: row.current_version, generation: row.generation }])), reason: reason.value.trim(), operation_id: crypto.randomUUID() };
            const captured = { dataset_id: datasetId, body };
            try { sessionStorage.setItem(storageKey, JSON.stringify(captured)); }
            catch (_) { throw new Error("This browser could not retain the recall for reply recovery. Free some browser storage before continuing."); }
            pending = captured;
        }
        busy = true; update(); tell("Checking the selected images and recalling their assignments…");
        try {
            const result = await app.api(`/datasets/${encodeURIComponent(pending.dataset_id)}/recall-images`, { method: "POST", body: pending.body });
            if (!current(ticket)) return;
            const received = result?.images;
            if (result?.dataset_id !== pending.dataset_id || !Array.isArray(received)
                    || received.length !== pending.body.image_ids.length
                    || new Set(received.map(row => row?.id)).size !== received.length
                    || received.some(row => !row || !pending.body.expected[row.id]
                        || row.dataset_id !== pending.dataset_id || row.state !== "unassigned" || row.assignee_id !== null
                        || row.current_version !== pending.body.expected[row.id].expected_version
                        || row.generation !== pending.body.expected[row.id].generation + 1 || row.editor_lease !== null)) {
                throw new Error("The server reply did not identify the complete saved recall. Its result is still unconfirmed.");
            }
            sessionStorage.removeItem(storageKey); pending = null;
            const message = `${result.images.length} images recalled to unassigned Incoming. Saved work and history were preserved.`;
            dialog.close(); app.notice(message);
            try { await app.refresh(); app.notice(message); }
            catch (error) { if (!error.stale) app.notice(`${message} Refresh the workspace to see the saved result.`, true); }
        } catch (error) {
            if (!current(ticket)) return;
            if ([400, 404, 409, 422].includes(error.status)) {
                sessionStorage.removeItem(storageKey); pending = null;
                tell("No images were recalled by this command. The selection changed or is no longer eligible. Refresh the list and check it again.", true);
                selected.clear(); render();
            } else tell(`${error.message} The original recall is retained; use “Check and retry saved recall” to recover its result.`, true);
        } finally { if (current(ticket)) { busy = false; update(); } }
    }
    filter.onchange = () => { selected.clear(); visible = 100; render(); };
    selectAll.onclick = () => { for (const row of filtered()) selected.add(row.id); render(); };
    clear.onclick = () => { selected.clear(); render(); };
    more.onclick = () => { visible += 100; render(); };
    reason.oninput = update;
    refresh.onclick = () => app.action(refresh, () => load(++sequence));
    close.onclick = () => dialog.close();
    button.onclick = () => app.action(button, open);
    submit.onclick = () => { commit().catch(error => tell(error.message, true)); };
    dialog.addEventListener("close", () => { ++sequence; });
    document.getElementById("datasetChoice").addEventListener("change", () => { ++sequence; dialog.close(); updateVisibility(); });
    function updateVisibility() {
        button.hidden = app.session?.user.role !== "admin"; button.disabled = !app.datasetId;
        if (identity() !== sessionKey || button.hidden) { ++sequence; dialog.close(); rows = []; selected.clear(); list.replaceChildren(); pending = null; busy = false; }
    }
    window.addEventListener("tator:hosted-session", updateVisibility);
    window.addEventListener("tator:hosted-workspace", updateVisibility);
    updateVisibility();
})();
