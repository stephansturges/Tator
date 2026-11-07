/* Named work packages. The server owns membership; this journal owns exact retries. */
(() => {
    "use strict";
    const app = window.TatorHostedDashboard;
    if (!app) return;
    const $ = id => document.getElementById(id);
    const el = (tag, text, cls) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; };
    const button = (text, fn, cls) => { const b = el("button", text, cls); b.type = "button"; b.onclick = () => app.action(b, fn); return b; };
    const labels = {not_assigned: "Not assigned", to_finish: "To finish", changes_requested: "Changes requested", awaiting_review: "Awaiting review", approved: "Approved", removed: "Removed", transferred: "Transferred", archived: "Archived"};
    const panel = el("section", undefined, "hosted-tranches"); panel.id = "tranchePanel"; panel.hidden = true; $("imageGrid").before(panel);
    const recovery = el("aside", undefined, "hosted-tranche-recovery"); recovery.id = "trancheRecovery"; recovery.hidden = true; panel.before(recovery);
    const dialog = el("dialog"); dialog.id = "trancheDialog"; document.body.append(dialog);
    let sequence = 0, shown = false, current = null, members = [], selected = new Map(), cursor = null, pending = null, running = false, owner = "";
    let filter = "all", lifecycle = "open", query = "", assigneeFilter = "", listRows = [], listCursor = null;
    const identity = () => app.session ? `${app.session.instance_id}:${app.session.user.user_id}` : "";
    const key = () => `tator.hosted.tranche-command:${identity()}`;
    const scopeLink = (id, tranche, queue) => `/annotate?${new URLSearchParams({image: id, ...(tranche ? {tranche} : {}), queue})}`;
    const currentQuery = () => new URLSearchParams(location.search);
    const admin = () => app.session?.user.role === "admin";
    function persist(value) {
        try { if (value) sessionStorage.setItem(key(), JSON.stringify(value)); else sessionStorage.removeItem(key()); }
        catch (_) { throw new Error("This browser could not retain the exact command. Keep this tab open and free browser storage before continuing."); }
        pending = value; renderRecovery();
    }
    function renderRecovery() {
        recovery.replaceChildren(); recovery.hidden = !pending;
        if (!pending) return;
        recovery.append(el("strong", "Finish the saved tranche command"), el("p", `${pending.label}. Its original images and settings are retained. Check its result before starting another tranche change.`));
        const retry = button(running ? "Checking…" : "Check and continue", resume, "primary"); retry.id = "trancheRetry"; retry.disabled = running; recovery.append(retry);
    }
    function readPending() {
        if (owner === identity()) return;
        ++sequence; owner = identity(); current = null; members = []; selected.clear(); dialog.close(); pending = null;
        if (owner) { try { pending = JSON.parse(sessionStorage.getItem(key()) || "null"); } catch (_) { app.notice("The retained tranche command cannot be read. Do not start another change until browser recovery is resolved.", true); pending = {label: "Unreadable retained command"}; } }
        renderRecovery();
    }
    const expected = rows => Object.fromEntries(rows.map(row => [row.id, {expected_version: row.current_version, generation: row.generation}]));
    const captured = rows => ({image_ids: rows.map(row => row.id), expected: expected(rows)});
    function validate(result, operation) {
        if (operation.bulk_review) {
            const ids = new Set(operation.body.images.map(row => row.image_id));
            if (!Array.isArray(result?.images) || result.images.length !== ids.size || new Set(result.images.map(row => row.image_id)).size !== ids.size
                || result.images.some(row => !ids.has(row.image_id) || !["approved", "rejected"].includes(row.status))) throw new Error("The review reply was incomplete. Its exact selections are retained.");
            return;
        }
        if (operation.image_command) { if (result?.id !== operation.path.split("/")[2] || result.state !== "unassigned" || typeof result.current_version !== "string" || !Number.isInteger(result.generation)) throw new Error("The withdrawal reply was incomplete. The exact command is retained."); return; }
        if (!result?.tranche || typeof result.tranche.id !== "string" || !Number.isInteger(result.tranche.revision)
            || result.tranche.dataset_id !== operation.dataset_id) throw new Error("The reply did not confirm this tranche command. Its exact request is retained.");
        if (operation.body.image_ids && (!Array.isArray(result.images) || result.images.length !== operation.body.image_ids.length
            || new Set(result.images.map(i => i.id)).size !== result.images.length
            || result.images.some(i => !operation.body.expected[i.id]))) throw new Error("The reply did not confirm every selected image. Its exact request is retained.");
    }
    async function resume() {
        readPending();
        if (running || !pending?.path) return;
        const capturedOwner = owner; let committed = null; running = true; renderRecovery();
        try {
            while (pending && owner === capturedOwner) {
                const operation = structuredClone(pending);
                let result;
                try { result = await app.api(operation.path, {method: "POST", body: operation.body}); }
                catch (error) {
                    if (owner !== capturedOwner || error.stale) return;
                    if (error.commit_outcome === "committed" && error.result_restricted) { if (operation.import_storage_key) sessionStorage.removeItem(operation.import_storage_key); persist(null); throw new Error("The change was saved, but this account can no longer see its result. Refresh the workspace."); }
                    if ([400, 403, 404, 409, 422, 423].includes(error.status)) {
                        if (operation.import_storage_key) sessionStorage.removeItem(operation.import_storage_key);
                        persist(null);
                        throw new Error(`${error.message} No changes from this command were applied.${operation.stage === "activate" ? " The created Draft remains available; review it and assign it again." : " Refresh the selection before trying again."}`);
                    }
                    throw error;
                }
                if (owner !== capturedOwner) return;
                validate(result, operation);
                if (operation.assign_after_create) {
                    const tranche = result.tranche;
                    persist({label: `Assign “${tranche.name}”`, dataset_id: tranche.dataset_id, stage: "activate", path: `/tranches/${encodeURIComponent(tranche.id)}/activate`, body: {operation_id: crypto.randomUUID(), expected_revision: tranche.revision,
                        annotator_id: operation.assign_after_create, ...captured(result.images)}, tranche_id: tranche.id});
                } else {
                    committed = result; if (operation.import_storage_key) sessionStorage.removeItem(operation.import_storage_key); const trancheId = result.tranche?.id || operation.tranche_id; persist(null); dialog.close();
                    app.notice(`${operation.bulk_review ? `${result.images.filter(row => row.status === "approved").length} images approved. ${result.images.filter(row => row.status === "rejected").map(row => `${operation.review_names[row.image_id]}: ${row.detail || row.error}`).join("; ")}` : `${operation.label}: saved.`}${operation.import_summary?.excluded.length ? ` Not attached: ${operation.import_summary.excluded.join("; ")}.` : ""}`);
                    if (app.datasetId === operation.dataset_id) {
                        history.replaceState(null, "", `/?${new URLSearchParams({dataset: operation.dataset_id, view: "tranches", tranche: trancheId})}`);
                        try { await app.showView(admin() ? "tranches" : "mine"); } catch (_) { app.notice("The change was saved. Refresh to load the current tranche.", true); }
                    }
                }
            }
        } catch (error) { if (owner === capturedOwner && !error.stale) app.notice(`${error.message}${pending ? " Use Check and continue to recover the original command." : ""}`, true); }
        finally { running = false; renderRecovery(); }
        return committed;
    }
    async function command(label, path, body, extra = {}) {
        if (dialog.open && dialog._context !== `${identity()}:${app.datasetId}`) throw new Error("The account or dataset changed. Reopen this action in its original dataset.");
        readPending();
        if (pending || running) throw new Error("Check the saved tranche command before starting another change.");
        persist({label, path, body: {...body, operation_id: crypto.randomUUID()}, dataset_id: app.datasetId, ...extra});
        return resume();
    }
    function totals(tranche) {
        const box = el("div", undefined, "tranche-counts");
        for (const [key, text] of Object.entries(labels)) if (tranche.counts[key] || ["to_finish", "awaiting_review", "approved"].includes(key)) {
            const badge = el("span"); badge.append(el("strong", String(tranche.counts[key] || 0)), document.createTextNode(` ${text.toLowerCase()}`)); box.append(badge);
        }
        box.append(el("small", `${tranche.counts.ever_included || 0} ever included · ${tranche.counts.retained || 0} currently retained${tranche.personal ? " · your work only" : ""}`));
        return box;
    }
    function route(id = null) {
        history.replaceState(null, "", `/?${new URLSearchParams({dataset: app.datasetId, view: "tranches", ...(id ? {tranche: id} : {})})}`);
        return show();
    }
    async function list(append = false) {
        const ticket = ++sequence, dataset = app.datasetId;
        if (!append) { listRows = []; listCursor = null; panel.replaceChildren(); }
        const params = new URLSearchParams({dataset_id: dataset, limit: "50"});
        if (lifecycle) params.set("lifecycle", lifecycle); if (assigneeFilter && admin()) params.set("annotator_id", assigneeFilter); if (query) params.set("q", query);
        if (append && listCursor) params.set("cursor", listCursor);
        const result = await app.api(`/tranches?${params}`);
        if (ticket !== sequence || !shown || dataset !== app.datasetId) return;
        listRows.push(...result.tranches); listCursor = result.next_cursor; renderList();
    }
    function personName(user) { return app.personLabel ? app.personLabel(user) : user.username; }
    function statusLine(tranche) {
        if (tranche.lifecycle === "active") return `Active · assigned to ${tranche.annotator_name || "an annotator"}`;
        if (tranche.lifecycle === "draft") return `Draft · not given to anyone yet${tranche.annotator_name ? ` (planned for ${tranche.annotator_name})` : ""}`;
        return "Closed · kept for history";
    }
    function renderList() {
        panel.replaceChildren();
        const toolbar = el("div", undefined, "tranche-toolbar"), stateLabel = el("label", "Status"), stateSelect = el("select");
        for (const [value, text] of [["open", "Open (Draft and Active)"], ["draft", "Draft"], ["active", "Active"], ["closed", "Closed"], ["", "All, including closed"]]) stateSelect.append(new Option(text, value));
        stateSelect.value = lifecycle; stateSelect.onchange = () => { lifecycle = stateSelect.value; app.action(stateSelect, () => list()); }; stateLabel.append(stateSelect);
        const search = el("input"); search.type = "search"; search.placeholder = "Find a tranche"; search.value = query; search.setAttribute("aria-label", "Find a tranche");
        const form = el("form"); form.className = "tranche-search"; form.append(search, button("Search", () => { query = search.value.trim(); return list(); })); form.onsubmit = e => { e.preventDefault(); query = search.value.trim(); app.action(search, () => list()); };
        toolbar.append(stateLabel, form); if (admin()) {
            const label = el("label", "Annotator"), select = el("select"); select.setAttribute("aria-label", "Filter by annotator"); select.append(new Option("All annotators", ""));
            app.users.filter(user => user.role === "annotator" || listRows.some(row => row.annotator_id === user.id)).forEach(user => select.append(new Option(personName(user), user.id)));
            select.value = assigneeFilter; select.onchange = () => { assigneeFilter = select.value; app.action(select, () => list()); }; label.append(select); toolbar.append(label, button("New tranche", () => create(), "primary"));
        } panel.append(toolbar);
        if (admin()) {
            const steps = el("ol", undefined, "tranche-steps");
            for (const text of ["Add new images in Incoming.", "Group them into a tranche and assign it to one annotator. Drafts stay private until assigned.", "The annotator submits images one by one. Approve them in Review to add them to the Confirmed dataset."]) steps.append(el("li", text));
            panel.append(steps);
        }
        const grid = el("div", undefined, "tranche-grid");
        for (const tranche of listRows) {
            const card = el("article", undefined, "tranche-card"); card.dataset.trancheId = tranche.id;
            const complete = tranche.counts.ever_included > 0 && ["not_assigned", "to_finish", "changes_requested", "awaiting_review"].every(key => !tranche.counts[key]);
            card.append(el("h2", tranche.name), el("p", `${complete ? "Complete · " : ""}${statusLine(tranche)}${tranche.annotator_id && !tranche.annotator_available ? " · Annotator unavailable" : ""}`), totals(tranche));
            if (tranche.instructions) card.append(el("p", tranche.instructions, "tranche-instructions"));
            if (tranche.updated_at) card.append(el("small", `Last activity ${new Date(tranche.updated_at * 1000).toLocaleString()}`, "tranche-activity"));
            if (!admin() && tranche.counts.to_finish + tranche.counts.changes_requested) card.append(button("Continue annotation", () => openQueue("work", tranche), "primary"));
            card.append(button(admin() ? "Manage tranche" : "Open tranche", () => route(tranche.id))); grid.append(card);
        }
        panel.append(grid);
        if (!listRows.length) panel.append(el("p", admin() ? `${lifecycle === "open" ? "No open tranches. Closed tranches are listed under Status. " : ""}Create a tranche from Incoming images, or start an empty Draft here.` : "No tranches match this view. Your submitted work remains available below."));
        if (listCursor) panel.append(button("Load more tranches", () => list(true)));
    }
    async function detail(id, append = false) {
        const ticket = ++sequence, dataset = app.datasetId;
        if (!append) { members = []; selected.clear(); cursor = null; }
        const params = new URLSearchParams({state: filter, limit: "100"}); if (append && cursor) params.set("cursor", cursor);
        const result = await app.api(`/tranches/${encodeURIComponent(id)}/members?${params}`);
        if (ticket !== sequence || !shown || dataset !== app.datasetId) return;
        current = result.tranche; members.push(...result.members); cursor = result.next_cursor; renderDetail();
    }
    async function allMembers(tranche, states) {
        const authority = identity(), output = [];
        for (const state of states) {
            let next = null;
            do { const params = new URLSearchParams({state, limit: "500"}); if (next) params.set("cursor", next);
                const page = await app.api(`/tranches/${encodeURIComponent(tranche.id)}/members?${params}`);
                if (identity() !== authority || app.datasetId !== tranche.dataset_id) throw new Error("The account or dataset changed. Reopen this action in its original dataset.");
                if (page.tranche.revision !== tranche.revision) throw new Error("The tranche changed. Refresh before assigning it.");
                output.push(...page.members.map(m => m.image).filter(Boolean)); next = page.next_cursor;
                if (output.length > 10000) throw new Error("A tranche can contain at most 10,000 unfinished images.");
            } while (next);
        }
        return output;
    }
    async function openQueue(queue, tranche = current) {
        const result = await app.api(`/queue?${new URLSearchParams({dataset_id: tranche.dataset_id, tranche_id: tranche.id, queue, limit: "1"})}`);
        if (!result.images.length) { app.notice(queue === "work" ? "All your images are submitted. The administrator can review them now." : "There are no images in this queue."); await route(tranche.id); return; }
        location.assign(scopeLink(result.images[0].id, tranche.id, queue));
    }
    function renderDetail() {
        panel.replaceChildren();
        const tranche = current, head = el("div", undefined, "tranche-toolbar");
        head.append(button("← All tranches", () => route()), el("h2", tranche.name)); panel.append(head);
        panel.append(el("p", `${statusLine(tranche)}${tranche.annotator_id && !tranche.annotator_available ? " · Annotator unavailable: recall unfinished work or withdraw pending submissions to reassign it." : ""}`), totals(tranche));
        if (admin() && tranche.lifecycle === "draft") panel.append(el("p", "Annotators cannot see a Draft. Choose Assign tranche to give its images to one annotator.", "tranche-hint"));
        if (tranche.instructions) panel.append(el("p", tranche.instructions, "tranche-instructions"));
        const actions = el("div", undefined, "tranche-toolbar");
        if (admin()) {
            if (tranche.lifecycle !== "closed") {
                actions.append(button("Edit name & instructions", () => edit(tranche)), button("Add Incoming images", () => pickIncoming(tranche)));
                actions.append(button("Upload images here", () => window.TatorHostedImport?.start({tranche_id: tranche.id})));
                if (tranche.lifecycle === "draft") actions.append(button("Assign tranche", () => chooseAssignee(tranche), "primary"));
                else { const change = button("Change annotator", () => chooseAssignee(tranche)); change.disabled = Boolean(tranche.counts.awaiting_review); if (change.disabled) change.title = "Review pending submissions first, or move selected unfinished images to another tranche."; actions.append(change); }
                actions.append(button("Close tranche", () => command(`Close “${tranche.name}”`, `/tranches/${tranche.id}/close`, {expected_revision: tranche.revision})));
            } else actions.append(button("Reopen as Draft", () => command(`Reopen “${tranche.name}”`, `/tranches/${tranche.id}/reopen`, {expected_revision: tranche.revision})));
            if (tranche.counts.awaiting_review) actions.append(button(`Review ${tranche.counts.awaiting_review} images`, () => openQueue("review"), "primary"));
            if (tranche.lifecycle === "draft" && tranche.counts.not_assigned) actions.append(button("Annotate Draft yourself", () => openQueue("draft")));
            actions.append(button("View history", () => showHistory(tranche)));
        } else if (tranche.counts.to_finish + tranche.counts.changes_requested) actions.append(button("Continue annotation", () => openQueue("work"), "primary"));
        else panel.append(el("p", tranche.counts.awaiting_review ? "All your images are submitted. You can track the administrator’s review here." : "No images are currently assigned to you in this tranche. Your earlier outcomes remain below."));
        panel.append(actions);
        const filters = el("label", "Show images"), select = el("select"); select.id = "trancheMemberFilter";
        select.append(new Option("All outcomes", "all")); for (const [key, text] of Object.entries(labels)) select.append(new Option(text, key)); select.value = filter;
        select.onchange = () => { filter = select.value; app.action(select, () => detail(tranche.id)); }; filters.append(select); panel.append(filters);
        const selectionBar = el("div", undefined, "tranche-toolbar"); selectionBar.id = "trancheSelection";
        if (admin() && tranche.lifecycle !== "closed") selectionBar.append(button("Select eligible images on this page", () => { members.forEach(m => { if (m.image && ["not_assigned", "to_finish", "changes_requested", "awaiting_review"].includes(m.state)) selected.set(m.image_id, m.image); }); renderDetail(); }), button("Clear selection", () => { selected.clear(); renderDetail(); }));
        selectionBar.append(el("span", `${selected.size} selected · maximum 500 per change`));
        if (selected.size) {
            const rows = [...selected.values()];
            if (rows.every(image => image.state === "submitted")) selectionBar.append(button("Approve selected submissions", () => approveSelected(tranche), "primary"));
            else if (rows.every(image => image.state !== "submitted")) selectionBar.append(button("Remove from tranche", () => removal(tranche)), button("Move to another tranche", () => move(tranche)));
            else selectionBar.append(el("span", "Select submitted images separately to approve them, or unfinished images to move them."));
        }
        panel.append(selectionBar);
        const grid = el("div", undefined, "hosted-grid tranche-members");
        for (const member of members) {
            const card = el("article", undefined, "hosted-card"); card.dataset.imageId = member.image_id;
            const info = el("div", undefined, "card-info"); info.append(el("strong", member.filename), el("small", labels[member.state] || member.state));
            if (member.can_view && member.image) {
                const image = member.image, link = el("a"), photo = el("img"); photo.loading = "lazy"; photo.alt = member.filename; photo.decoding = "async"; photo.onerror = () => { photo.onerror = null; photo.src = `/api/hosted/images/${encodeURIComponent(image.id)}/media`; }; photo.src = `/api/hosted/images/${encodeURIComponent(image.id)}/thumbnail`;
                const queue = {not_assigned: "draft", to_finish: "work", changes_requested: "work", awaiting_review: "review", approved: "confirmed"}[member.state];
                link.href = queue ? scopeLink(image.id, queue === "confirmed" ? null : tranche.id, queue) : `/annotate?image=${encodeURIComponent(image.id)}`;
                link.append(photo, info); card.append(link);
                if (admin() && ["not_assigned", "to_finish", "changes_requested", "awaiting_review"].includes(member.state)) {
                    const check = el("input"); check.type = "checkbox"; check.checked = selected.has(image.id); check.setAttribute("aria-label", `Select ${member.filename}`);
                    check.onchange = () => { if (check.checked) selected.set(image.id, image); else selected.delete(image.id); renderDetail(); }; card.append(check);
                }
                if (admin() && member.state === "awaiting_review") card.append(button("Withdraw to Incoming…", () => withdraw(image)));
            } else card.append(info);
            grid.append(card);
        }
        panel.append(grid); if (cursor) panel.append(button("Load more images", () => detail(tranche.id, true)));
        if (!members.length) panel.append(el("p", "No images match this filter."));
    }
    function form(title) { dialog._context = `${identity()}:${app.datasetId}`; dialog.replaceChildren(el("h2", title)); const notice = el("p"); notice.dataset.dialogNotice = ""; notice.className = "dialog-notice"; notice.hidden = true; dialog.append(notice); const f = el("form"); dialog.append(f); dialog.showModal(); return f; }
    function field(f, text, tag = "input", value = "") { const label = el("label", text), input = el(tag); input.value = value; input.setAttribute("aria-label", text); if (tag === "textarea") { input.rows = 3; input.maxLength = 10000; } label.append(input); f.append(label); return input; }
    function submit(f, text, work) { const row = el("div", undefined, "dialog-actions"), save = el("button", text, "primary"); save.type = "submit"; row.append(button("Cancel", () => dialog.close()), save); f.append(row); f.onsubmit = e => { e.preventDefault(); app.action(save, work); }; }
    function annotators(f, selectedId = "") { const choice = field(f, "Annotator", "select"); choice.append(new Option("Choose an annotator", "")); for (const user of app.users.filter(u => u.role === "annotator" && !u.disabled)) choice.append(new Option(personName(user), user.id)); choice.value = selectedId; return choice; }
    function checkSelection(rows) { if (!rows.length || rows.length > 500) throw new Error("Select between 1 and 500 images for one membership change."); }
    function create(rows = []) {
        if (rows.length > 500) { app.notice("Create a tranche with up to 500 selected images, then add more in further selections.", true); return; }
        const f = form("Create an annotation tranche"), name = field(f, "Name"), instructions = field(f, "Instructions for this tranche", "textarea"), assignee = annotators(f), mode = field(f, "When should the annotator receive it?", "select");
        name.required = true; name.maxLength = 200; mode.append(new Option("Save as Draft", "draft"), new Option("Assign now", "active"));
        f.append(el("p", `${rows.length} selected images will be reserved. A Draft is visible only to administrators. You can add up to 10,000 unfinished images in selections of at most 500.`));
        submit(f, "Create tranche", () => { if (mode.value === "active" && (!assignee.value || !rows.length)) throw new Error("Choose an annotator and at least one image to assign now."); return command(`Create “${name.value.trim()}”`, `/datasets/${app.datasetId}/tranches`, {name: name.value.trim(), instructions: instructions.value.trim(), annotator_id: assignee.value || null, ...captured(rows)}, mode.value === "active" ? {assign_after_create: assignee.value} : {}); });
    }
    function edit(tranche) {
        const f = form("Edit tranche"), name = field(f, "Name", "input", tranche.name), instructions = field(f, "Instructions for this tranche", "textarea", tranche.instructions); name.required = true; name.maxLength = 200;
        const assignee = tranche.lifecycle === "draft" ? annotators(f, tranche.annotator_id) : null;
        submit(f, "Save changes", () => command(`Update “${name.value.trim()}”`, `/tranches/${tranche.id}/edit`, {expected_revision: tranche.revision, name: name.value.trim(), instructions: instructions.value.trim(), ...(assignee ? {annotator_id: assignee.value || null} : {})}));
    }
    function chooseAssignee(tranche) {
        const f = form(tranche.lifecycle === "draft" ? "Assign this tranche" : "Change its annotator"), choice = annotators(f, tranche.annotator_id); choice.required = true;
        f.append(el("p", "This assigns all unfinished images together. Saved annotations remain intact. Review submitted images before changing an active tranche’s annotator."));
        submit(f, "Assign unfinished images", async () => { const rows = await allMembers(tranche, tranche.lifecycle === "draft" ? ["not_assigned"] : ["to_finish", "changes_requested"]); return command(`Assign “${tranche.name}”`, `/tranches/${tranche.id}/${tranche.lifecycle === "draft" ? "activate" : "assignee"}`, {expected_revision: tranche.revision, annotator_id: choice.value, ...captured(rows)}); });
    }
    function removal(tranche) {
        const rows = [...selected.values()]; checkSelection(rows); const f = form(`Remove ${rows.length} images from this tranche`), reason = field(f, "Reason", "textarea"); reason.required = true;
        f.append(el("p", "The images return to unallocated Incoming with their saved annotations. Their current annotator loses editing access. Nothing is deleted."));
        submit(f, "Remove from tranche", () => command(`Remove ${rows.length} images`, `/tranches/${tranche.id}/members/remove`, {expected_revision: tranche.revision, reason: reason.value.trim(), ...captured(rows)}));
    }
    async function move(tranche) {
        const rows = [...selected.values()]; checkSelection(rows); const f = form(`Move ${rows.length} images`), choice = field(f, "Destination tranche", "select"), reason = field(f, "Reason", "textarea"); reason.required = true;
        const destinations = []; let next = null; do { const q = new URLSearchParams({dataset_id: tranche.dataset_id, limit: "100"}); if (next) q.set("cursor", next); const page = await app.api(`/tranches?${q}`); destinations.push(...page.tranches.filter(t => t.id !== tranche.id && t.lifecycle !== "closed")); next = page.next_cursor; } while (next);
        destinations.forEach(t => choice.append(new Option(`${t.name} · ${t.lifecycle}${t.annotator_name ? ` · ${t.annotator_name}` : ""}`, t.id))); choice.required = true;
        f.append(el("p", "Moving to an Active tranche hands the images to its annotator. Moving to a Draft reserves them for an administrator. Saved annotations remain intact."));
        submit(f, "Move images", () => { const target = destinations.find(t => t.id === choice.value); if (!target) throw new Error("Create another Draft first."); return command(`Move ${rows.length} images`, `/tranches/${tranche.id}/move`, {expected_revision: tranche.revision, destination_id: target.id, expected_destination_revision: target.revision, reason: reason.value.trim(), ...captured(rows)}); });
    }
    async function approveSelected(tranche) {
        const rows = [...selected.values()]; checkSelection(rows);
        if (!rows.every(row => row.state === "submitted" && row.active_submission)) throw new Error("Select only images awaiting review.");
        if (!confirm(`Approve these ${rows.length} submitted images into the confirmed dataset? Only continue after checking their annotations.`)) return;
        await command(`Review ${rows.length} submissions`, `/datasets/${tranche.dataset_id}/approve-images`, {editor_id: crypto.randomUUID(), images: rows.map(image => ({image_id: image.id, expected_version: image.current_version, generation: image.generation, operation_id: crypto.randomUUID()}))}, {bulk_review: true, tranche_id: tranche.id, review_names: Object.fromEntries(rows.map(row => [row.id, row.filename]))});
    }
    async function pickIncoming(tranche) {
        const f = form("Add unallocated Incoming images"), grid = el("div", undefined, "tranche-picker"), chosen = new Map();
        let next = null, rows = [], total = 0, searchText = "", generation = 0, lastIndex = null;
        const authority = identity();
        const valid = () => { if (identity() !== authority || app.datasetId !== tranche.dataset_id || !dialog.open) throw new Error("Reopen this selection in its original dataset."); };
        const params = limit => new URLSearchParams({dataset_id: tranche.dataset_id, queue: "unallocated", limit: String(limit), q: searchText, include_total: "true"});
        const search = el("input"); search.type = "search"; search.placeholder = "Find images by filename"; search.setAttribute("aria-label", "Find Incoming images");
        const counter = el("p", "0 selected"); counter.setAttribute("role", "status");
        const update = () => { counter.textContent = `${chosen.size} selected · ${total} matching images · maximum 500 per change`; grid.querySelectorAll("input").forEach(check => { check.checked = chosen.has(check.dataset.imageId); }); };
        const selectRows = additions => { const fresh = new Map(chosen); additions.forEach(row => fresh.set(row.id, row)); if (fresh.size > 500) throw new Error(`This would select ${fresh.size} images. Narrow the search or choose a range of at most 500. Your existing selection is unchanged.`); chosen.clear(); fresh.forEach((row, id) => chosen.set(id, row)); update(); };
        const render = () => {
            grid.replaceChildren(); rows.forEach((image, index) => {
                const label = el("label"), check = el("input"), img = el("img"); check.type = "checkbox"; check.dataset.imageId = image.id; check.setAttribute("aria-label", `Select ${image.filename}`);
                check.onclick = event => { try {
                    if (event.shiftKey && lastIndex !== null && check.checked) selectRows(rows.slice(Math.min(lastIndex, index), Math.max(lastIndex, index) + 1));
                    else if (check.checked) selectRows([image]); else { chosen.delete(image.id); update(); }
                    lastIndex = index;
                } catch (error) { update(); app.notice(error.message, true); } };
                img.decoding = "async"; img.onerror = () => { img.onerror = null; img.src = `/api/hosted/images/${encodeURIComponent(image.id)}/media`; }; img.src = `/api/hosted/images/${encodeURIComponent(image.id)}/thumbnail`; img.loading = "lazy"; img.alt = ""; label.append(check, img, document.createTextNode(image.filename)); grid.append(label);
            });
            if (!rows.length) grid.append(el("p", "No unallocated Incoming images match this search.")); more.hidden = !next; update();
        };
        const load = async (append = false) => {
            const ticket = ++generation, q = params(100); if (append && next) q.set("cursor", next);
            const page = await app.api(`/queue?${q}`); valid(); if (ticket !== generation) return;
            next = page.next_cursor; total = page.total; rows = append ? [...rows, ...page.images] : page.images; render();
        };
        const find = () => { searchText = search.value.trim(); chosen.clear(); lastIndex = null; return load(); };
        search.onkeydown = event => { if (event.key === "Enter") { event.preventDefault(); app.action(search, find); } };
        const toolbar = el("div", undefined, "tranche-toolbar"); toolbar.append(search, button("Search images", find), button("Select visible images", () => selectRows(rows)), button("Select all matching images", async () => {
            const ticket = generation, page = await app.api(`/queue?${params(500)}`); valid(); if (ticket !== generation) throw new Error("The search changed; select from the current results.");
            if (page.total > 500 || page.next_cursor) throw new Error(`${page.total} images match. Narrow the search to at most 500; no selection was changed.`);
            selectRows(page.images);
        }), button("Clear selection", () => { chosen.clear(); update(); }));
        const more = button("Load more Incoming images", () => load(true));
        f.append(el("p", `Select images, or shift-click to select a range. ${tranche.lifecycle === "active" ? "They will be assigned immediately to this tranche’s annotator." : "They will be reserved in this Draft."}`), toolbar, counter, grid, more);
        submit(f, "Add selected images", () => { valid(); const selectedRows = [...chosen.values()]; checkSelection(selectedRows); return command(`Add ${selectedRows.length} images`, `/tranches/${tranche.id}/members/add`, {expected_revision: tranche.revision, ...captured(selectedRows)}); }); await load();
    }
    async function showHistory(tranche) {
        const f = form(`${tranche.name} · History`), list = el("div"); let next = null; f.append(list);
        const load = async () => { const q = new URLSearchParams({limit: "100"}); if (next) q.set("cursor", next); const result = await app.api(`/tranches/${tranche.id}/history?${q}`); result.events.forEach(event => list.append(el("p", `${new Date((event.created_at || event.at) * 1000).toLocaleString()} · ${event.kind || event.action || event.event_type}${event.data?.reason ? ` · ${event.data.reason}` : ""}`))); next = result.next_cursor; more.hidden = !next; };
        const more = button("Load earlier history", load); f.append(more, button("Close", () => dialog.close())); await load();
    }
    async function withdraw(image) {
        // This image command has its own existing receipt contract, not a group result.
        const f = form("Withdraw submission to Incoming"), reason = field(f, "Reason", "textarea"); reason.required = true;
        f.append(el("p", "The submission is closed without approval. Its saved annotations and review history remain. It can then be assigned again."));
        submit(f, "Withdraw submission", async () => {
            if (pending) throw new Error("Finish the saved tranche command first.");
            const authority = identity();
            const fresh = await app.api(`/images/${image.id}`);
            if (identity() !== authority || app.datasetId !== image.dataset_id) throw new Error("The account or dataset changed. Reopen this submission before withdrawing it.");
            persist({label: `Withdraw ${image.filename}`, dataset_id: image.dataset_id, image_command: true, path: `/images/${image.id}/withdraw-submission`, body: {operation_id: crypto.randomUUID(), expected_version: fresh.current_version, generation: fresh.generation, editor_id: crypto.randomUUID(), submission_id: fresh.active_submission, reason: reason.value.trim()}, tranche_id: current.id}); await resume();
        });
    }
    async function show() { shown = true; panel.hidden = false; readPending(); if (!app.datasetId) { panel.replaceChildren(el("p", "No datasets are available for this account.")); return; } const id = currentQuery().get("tranche"); return id ? detail(id) : list(); }
    function hide() { shown = false; ++sequence; panel.hidden = true; }
    window.TatorHostedTranches = {show, hide, create,
        async attachImported(trancheId, entries, importStorageKey) {
            const authority = identity(), tranche = await app.api(`/tranches/${encodeURIComponent(trancheId)}`);
            const rows = [], excluded = [], seen = new Map();
            for (const entry of entries) {
                if (seen.has(entry.image_id)) { excluded.push(`${entry.name}: same image as ${seen.get(entry.image_id)}`); continue; }
                seen.set(entry.image_id, entry.name);
                const image = await app.api(`/images/${encodeURIComponent(entry.image_id)}`);
                if (image.state === "unassigned" && !image.tranche_id && !image.archived_at) rows.push(image);
                else excluded.push(`${entry.name}: ${image.tranche_name ? `already in “${image.tranche_name}”` : image.state === "confirmed" ? "already in the confirmed dataset" : image.archived_at ? "archived" : ({submitted: "awaiting review", assigned: "already assigned", returned: "changes requested"}[image.state] || image.state)}`);
            }
            if (identity() !== authority || app.datasetId !== tranche.dataset_id) throw new Error("The account or dataset changed. The uploaded images remain saved; reopen their original tranche.");
            if (!rows.length || rows.length > 500 || tranche.lifecycle === "closed") {
                sessionStorage.removeItem(importStorageKey);
                if (rows.length > 500 || tranche.lifecycle === "closed") throw new Error(`All uploads are saved in Incoming. ${tranche.lifecycle === "closed" ? "This tranche is closed." : `${rows.length} eligible images exceed the 500-image attachment limit.`} Use Add Incoming images to make a fresh selection.`);
                return {attached: 0, excluded};
            }
            const summary = {attached: rows.length, excluded};
            const result = await command(`Attach ${rows.length} uploaded images`, `/tranches/${tranche.id}/members/add`, {expected_revision: tranche.revision, ...captured(rows)}, {import_storage_key: importStorageKey, import_summary: summary});
            if (!result) throw new Error(pending ? "The attachment reply is unconfirmed. Use Check and continue; uploading again is unnecessary." : "The upload is saved in Incoming, but attachment failed. Refresh the tranche and select the eligible images there.");
            return summary;
        },
        get hasPending() { return !!pending; }};
    window.addEventListener("tator:hosted-session", () => { readPending(); if (!app.session) hide(); });
    window.addEventListener("tator:hosted-workspace", readPending);
    $("datasetChoice").addEventListener("change", () => { dialog.close(); current = null; });
})();
